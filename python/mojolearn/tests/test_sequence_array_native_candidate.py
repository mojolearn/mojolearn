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


def assert_defined_u16_outputs(actual, expected, source):
    # FIXTURE CORRECTION (2026-10-22): the retained NumPy1.26 mixed-input
    # failure asserted exact bytes for negative/out-of-range float->uint16
    # values. NumPy explicitly leaves those results undefined under C99:
    # https://numpy.org/doc/1.26/release/1.24.0-notes.html#numpy-now-gives-floating-point-errors-in-casts
    # Keep the entire operation and its warnings/native audit. Restrict only
    # this uint16 byte oracle to finite values whose truncated integer fits;
    # the original full-byte mismatch remains retained qualification evidence.
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.flags.writeable == expected.flags.writeable
    finite = np.isfinite(source)
    truncated = np.trunc(source[finite].astype("f8"))
    defined = np.zeros(source.shape, dtype=bool)
    defined[finite] = (truncated >= 0) & (truncated <= 65535)
    assert np.any(defined)
    assert actual[defined].tobytes() == expected[defined].tobytes()


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
    if expected.dtype.kind == "u" and expected.dtype.itemsize == 2:
        assert_defined_u16_outputs(actual, expected, values)
    else:
        assert_outputs(actual, expected)
    assert [(type(w.message), str(w.message)) for w in new] == [(type(w.message), str(w.message)) for w in old]


@pytest.mark.parametrize("length", [1, 7, 8, 15, 16, 17, 31, 32, 33])
@pytest.mark.parametrize("reversed_input", [False, True])
def test_native_u16_defined_boundaries_and_neighbor_lengths(length, reversed_input):
    # Every truncated value fits uint16, including negative fractions that
    # truncate to zero. Compare every output byte across neighboring lengths
    # and source strides; no product dispatch keys on these test dimensions.
    boundary = np.array([65535., np.nextafter(np.float32(65536), np.float32(0)),
                         0., -0., .5, -.5, 1., 255., 256., 32768.,
                         np.nextafter(np.float32(-1), np.float32(0)), 65534.], dtype="f4")
    values = np.resize(boundary, length)
    if reversed_input:
        values = values[::-1]
    tensor = to_device(values, numeric_mode="identical")
    if type(tensor) is not SequenceDeviceTensor:
        pytest.skip("uint16 output qualification requires a device tensor")
    with warnings.catch_warnings(record=True) as old, np.errstate(all="raise"):
        warnings.simplefilter("always")
        expected = values.astype("u2", copy=False)
    with warnings.catch_warnings(record=True) as new, np.errstate(all="raise"):
        warnings.simplefilter("always")
        actual = native_candidate(tensor, "u2")
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
    "1f4", ("f4", 1), ("f4", (1,)), ("f4", np.int64(2)),
    ("f4", 0), ("f8", (0, 2)), "V0", [], [("empty", "V0")],
    [("a", "f4", 2)], ("S", 4), ("U", 4), ("V", 4), [("a", "S", 4)],
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


MALFORMED_SCHEMAS = [
    "dontknow", "f4,garbage", "f4,,i4", "(2)f4", "2f4junk", "f4 ",
    ("f4", (-1,)), ("f4", (2**31,)), ("f4", True), ("S", -1),
    ("U", 2**31), [("a", "f4"), ("a", "i4")],
    {"names": ["a"], "formats": []},
    {"names": ["a"], "formats": ["f4"], "offsets": [-1]},
    [("a", "f4", "bad")], (), ("f4",), [],
]


@pytest.mark.parametrize("index,schema", list(enumerate(MALFORMED_SCHEMAS)), ids=lambda value: repr(value))
def test_native_candidate_preserves_malformed_dtype_exception_class(resident, index, schema):
    values, tensor = resident
    with warnings.catch_warnings(record=True) as old:
        warnings.simplefilter("always")
        try:
            expected = values.astype(schema, copy=False)
            failure = None
        except Exception as error:
            failure = error
    with warnings.catch_warnings(record=True) as new:
        warnings.simplefilter("always")
        audit = audit_native_call(_OPT_NATIVE, "seqdev_array", tensor, (schema, None), None, _OPT_CONTEXT)
    report = audit.report()
    assert not report["python_frames"] and not report["forbidden_native_calls"], report
    assert not report["unobserved_existing_threads"], report
    if failure is None:
        assert_schema_outputs(audit.assert_native(), expected)
    else:
        assert type(audit.exception) is type(failure)
        # These seven defects are the retained a5 repair scope. Other
        # existing admission messages remain explicit native diagnostics.
        if index in {2, 6, 7, 8, 11, 13, 14}:
            assert str(audit.exception) == str(failure)
        if type(failure) is SyntaxError:
            assert audit.exception.args == failure.args
    assert [(type(w.message), str(w.message)) for w in new] == [(type(w.message), str(w.message)) for w in old]
