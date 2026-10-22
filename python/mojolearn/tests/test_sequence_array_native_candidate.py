"""Qualify the native array candidate before replacing the public protocol.

These tests deliberately call the internal native candidate. They do not
qualify the unchanged public SequenceDeviceTensor.__array__ wrapper.
"""
import warnings
import struct

import numpy as np
import pytest

from mojolearn._training_impl import _OPT_NATIVE, _OPT_CONTEXT
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn._x_sequence_device import SequenceDeviceTensor, to_device


DTYPES = [
    None, np.float32, np.float64, "f2", "f4", "f8", np.longdouble,
    "?", "i1", "u1", "i2", "u2", "i4", "u4", "i8", "u8", "c8", "c16",
    np.clongdouble, object, str, bytes, "U0", "U8", "S0", "S8", "U1", "S1",
    "datetime64[D]", "timedelta64[D]", "V4",
    np.dtype([("a", "f4"), ("b", "i4")]), np.dtype(("f4", (2,))), ">f4", ">f8", ">i4",
]


@pytest.fixture(scope="module")
def resident():
    values = np.array([0., -0., .5, -.5, 1., -1., 255., 256., -128., 32768.,
                       1e20, np.inf, -np.inf, np.nan, np.finfo("f4").tiny,
                       np.nextafter(np.float32(0), np.float32(1))], dtype="f4")
    tensor = to_device(values, numeric_mode="identical")
    if type(tensor) is not SequenceDeviceTensor:
        pytest.skip("candidate array qualification requires a device tensor")
    return values, tensor


def native_candidate(tensor, dtype, copy=None):
    return audit_native_call(_OPT_NATIVE, "seqdev_array", tensor, (dtype, copy),
                             None, _OPT_CONTEXT).assert_native()


def assert_outputs(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.flags.writeable == expected.flags.writeable
    if actual.dtype.hasobject:
        assert all(type(x) is float for x in actual.ravel())
        assert [struct.pack("=d", x) for x in actual.ravel()] == [
            struct.pack("=d", x) for x in expected.ravel()]
    else:
        assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda dtype: str(dtype))
@pytest.mark.parametrize("copy", [None, False, True])
def test_native_candidate_preserves_retained_dtype_copy_contract(resident, dtype, copy):
    values, tensor = resident
    with warnings.catch_warnings(record=True) as old:
        warnings.simplefilter("always")
        expected = values.copy() if dtype is None else values.astype(dtype, copy=False)
    with warnings.catch_warnings(record=True) as new:
        warnings.simplefilter("always")
        actual = native_candidate(tensor, dtype, copy)
    assert_outputs(actual, expected)
    assert [(type(w.message), str(w.message)) for w in new] == [(type(w.message), str(w.message)) for w in old]


@pytest.mark.parametrize("mode", ["ignore", "warn", "raise"])
@pytest.mark.parametrize("dtype", ["f2", "i4", "u8"])
def test_native_candidate_preserves_numpy_error_state(resident, mode, dtype):
    values, tensor = resident
    with np.errstate(all=mode), warnings.catch_warnings(record=True) as old:
        warnings.simplefilter("always")
        if mode == "raise":
            with pytest.raises(FloatingPointError) as previous:
                values.astype(dtype)
        else:
            expected = values.astype(dtype)
    with np.errstate(all=mode), warnings.catch_warnings(record=True) as new:
        warnings.simplefilter("always")
        if mode == "raise":
            audit = audit_native_call(_OPT_NATIVE, "seqdev_array", tensor, (dtype, None), None, _OPT_CONTEXT)
            assert type(audit.exception) is FloatingPointError
            assert str(audit.exception) == str(previous.value)
            report = audit.report()
            assert not report["python_frames"] and not report["forbidden_native_calls"], report
        else:
            assert_outputs(native_candidate(tensor, dtype), expected)
    assert [str(w.message) for w in new] == [str(w.message) for w in old]


def test_native_candidate_shortest_float_strings_and_special_payloads(resident):
    _, tensor = resident
    values = np.random.default_rng(42).integers(0, 2 ** 32, size=4096, dtype="u4").view("f4")
    device = to_device(values, numeric_mode="identical")
    with np.errstate(all="ignore"):
        for dtype in ("U32", "S8", ">f4", "f8", "c8", "c16", "f2"):
            assert_outputs(native_candidate(device, dtype), values.astype(dtype))


def test_native_candidate_refuses_dtype_hooks_before_invoking_them(resident):
    _, tensor = resident
    calls = []
    class Protocol:
        @property
        def dtype(self):
            calls.append("dtype")
            return np.dtype("f4")
    audit = audit_native_call(_OPT_NATIVE, "seqdev_array", tensor, (Protocol(), None), None, _OPT_CONTEXT)
    assert type(audit.exception) is TypeError
    assert not calls
    report = audit.report()
    assert not report["python_frames"] and not report["forbidden_native_calls"], report


def test_native_candidate_warning_errors_preserve_native_exception(resident):
    _, tensor = resident
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        audit = audit_native_call(_OPT_NATIVE, "seqdev_array", tensor, ("f2", None), None, _OPT_CONTEXT)
    assert type(audit.exception) is RuntimeWarning
    assert str(audit.exception) == "overflow encountered in cast"
    report = audit.report()
    assert not report["python_frames"] and not report["forbidden_native_calls"], report


def test_native_candidate_refuses_python_dtype_alias_discovery(resident, monkeypatch):
    _, tensor = resident
    calls = []
    class Alias:
        @property
        def dtype(self):
            calls.append("dtype")
            return np.dtype("f4")
    monkeypatch.setitem(np.sctypeDict, "mojolearn_foreign_dtype", Alias())
    audit = audit_native_call(_OPT_NATIVE, "seqdev_array", tensor,
                              ("mojolearn_foreign_dtype", None), None, _OPT_CONTEXT)
    assert type(audit.exception) is TypeError
    assert not calls
    report = audit.report()
    assert not report["python_frames"] and not report["forbidden_native_calls"], report


def test_native_candidate_preserves_native_error_log_stream(resident):
    import io
    _, tensor = resident
    stream = io.StringIO()
    with np.errstate(over="log", call=stream):
        native_candidate(tensor, "f2")
    assert stream.getvalue() == "Warning: overflow encountered in cast\n"


def test_native_candidate_refuses_replaced_numpy_dtype_before_call(resident, monkeypatch):
    _, tensor = resident
    calls = []
    def replaced(*args, **kwargs):
        calls.append("dtype")
        raise AssertionError("Python dtype constructor executed")
    monkeypatch.setattr(np, "dtype", replaced)
    audit = audit_native_call(_OPT_NATIVE, "seqdev_array", tensor, ("f8", None), None, _OPT_CONTEXT)
    assert type(audit.exception) is TypeError
    assert not calls
    report = audit.report()
    assert not report["python_frames"] and not report["forbidden_native_calls"], report


SCHEMAS = [
    "float32", b"float64", "int32", "2f4", "(2,)f4", "(2,3)>f4", "2,3f4",
    "()f4", "i4,f4", "i4, f4", "f4,f8,", ">2f4", "2f4 ",
    ("f4", 2), ("f4", (2, 3)), ("f4", (0,)),
    [("a", "f4"), ("b", "i4")],
    [("a", "f4", (2,)), ("b", [("c", ">f4"), ("d", "f8")])],
    [(("title", "a"), "f4"), ("b", "i4")],
    {"names": ["a", "b"], "formats": ["f4", "i4"]},
    {"names": ["a", "b"], "formats": ["f4", "i4"], "offsets": [4, 0]},
    {"names": ["a", "b"], "formats": ["i1", "i4"], "aligned": True},
    {"names": ["a"], "formats": [[("x", "i1"), ("y", "i4")]], "aligned": True},
    {"names": ["a"], "formats": ["f4"], "titles": ["title"], "itemsize": 8},
    {"a": ("f4", 4), "b": ("i4", 0)},
    {"a": ("f4", 4, "title"), "b": ("i4", 0), -1: ["a", "b"]},
]


def assert_schema_outputs(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.flags.writeable == expected.flags.writeable
    if actual.dtype.names:
        for name in actual.dtype.names:
            # Padding bytes are uninitialized in NumPy's original astype;
            # compare every field and its exact floating words, not padding.
            assert_schema_outputs(actual[name], expected[name])
    else:
        assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("schema", SCHEMAS, ids=lambda value: repr(value))
def test_native_candidate_normalizes_ordinary_dtype_schemas(resident, schema):
    values, tensor = resident
    with warnings.catch_warnings(record=True) as old:
        warnings.simplefilter("always")
        expected = values.astype(schema, copy=False)
    with warnings.catch_warnings(record=True) as new:
        warnings.simplefilter("always")
        actual = native_candidate(tensor, schema)
    assert_schema_outputs(actual, expected)
    assert [(type(w.message), str(w.message)) for w in new] == [(type(w.message), str(w.message)) for w in old]


@pytest.mark.parametrize("position", ["shape", "offset", "title"])
def test_native_candidate_refuses_schema_protocols_before_hooks(resident, position):
    _, tensor = resident
    calls = []
    class Protocol:
        def __index__(self):
            calls.append("index")
            return 2
        def __hash__(self):
            calls.append("hash")
            return 1
        def __eq__(self, other):
            calls.append("eq")
            return False
    value = Protocol()
    schema = {"shape": ("f4", (value,)),
              "offset": {"names": ["a"], "formats": ["f4"], "offsets": [value]},
              "title": [((value, "a"), "f4")]}[position]
    audit = audit_native_call(_OPT_NATIVE, "seqdev_array", tensor, (schema, None), None, _OPT_CONTEXT)
    assert type(audit.exception) is TypeError
    assert not calls
    report = audit.report()
    assert not report["python_frames"] and not report["forbidden_native_calls"], report
