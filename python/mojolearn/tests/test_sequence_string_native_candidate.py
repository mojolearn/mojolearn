"""Qualify managed StringDType storage without changing public __array__."""
import struct
import warnings

import numpy as np
import pytest

from mojolearn._training_impl import _OPT_NATIVE, _OPT_CONTEXT
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn._x_sequence_device import SequenceDeviceTensor, to_device


pytestmark = pytest.mark.skipif(not hasattr(np.dtypes, "StringDType"), reason="NumPy 2 StringDType is unavailable")


def candidate(tensor, dtype, copy=None):
    return audit_native_call(_OPT_NATIVE, "seqdev_array", tensor, (dtype, copy), None, _OPT_CONTEXT)


def assert_native_boundary(audit):
    report = audit.report()
    assert not report["python_frames"] and not report["forbidden_native_calls"], report
    assert not report["unobserved_existing_threads"], report


@pytest.fixture(scope="module")
def resident():
    words = np.array([0, 0x80000000, 0x3fa00000, 0x60ad78ec, 0x7f800000,
                      0xff800000, 0x7fc00001, 0x7f800001, 1], dtype="u4")
    values = words.view("f4")
    tensor = to_device(values, numeric_mode="identical")
    if type(tensor) is not SequenceDeviceTensor:
        pytest.skip("managed string candidate requires a device tensor")
    return values, tensor


def missing_value(name):
    if name == "none": return None
    if name == "text": return "nan"
    if name == "float": return float("nan")
    if name == "complex": return complex(float("nan"), 0)
    if name == "numpy": return np.float32("nan")
    if name == "nat": return np.datetime64("NaT", "ns")
    if name == "integer": return 7
    if name == "finite": return 1.25
    if name == "bytes": return b"missing"
    if name == "tuple": return (None, 7)
    if name == "list": return [None, 7]
    if name == "dict": return {"missing": None}
    if name == "rawvoid": return np.void(b"missing")
    raise AssertionError(name)


def assert_array_values(actual, expected):
    # Structured String fields can contain subarrays, while neighboring
    # numeric/object leaves retain NaN payloads. Compare consumed leaf values.
    if expected.dtype.fields is not None:
        for name in expected.dtype.names:
            assert_array_values(actual[name], expected[name])
    elif expected.dtype.kind == "T":
        assert actual.tolist() == expected.tolist()
    elif expected.dtype.kind == "O":
        assert [struct.pack("=d", value) for value in actual.flat] == [struct.pack("=d", value) for value in expected.flat]
    else:
        assert actual.tobytes() == expected.tobytes()


def structured_string_dtype(name):
    strings = np.dtypes.StringDType(na_object="missing", coerce=False)
    simple = {"names": ["text"], "formats": [strings]}
    if name == "field":
        return simple
    if name == "list_subarray":
        return [("text", (strings, (2,)))]
    if name == "mixed":
        return {"names": ["text", "number", "object"], "formats": [strings, "f4", "O"]}
    if name == "nested":
        return {"names": ["nested"], "formats": [simple]}
    if name == "nested_subarray":
        return {"names": ["nested"], "formats": [(simple, (2,))]}
    if name == "outer_subarray":
        return (simple, (2,))
    if name == "distinct_metadata":
        return {"names": ["left", "right"], "formats": [strings, np.dtypes.StringDType(na_object=None)]}
    raise AssertionError(name)


@pytest.mark.parametrize("name", ["field", "list_subarray", "mixed", "nested", "nested_subarray", "outer_subarray", "distinct_metadata"])
@pytest.mark.parametrize("copy", [None, False, True])
def test_native_string_uses_each_allocated_structured_leaf(resident, name, copy):
    values, tensor = resident
    dtype = structured_string_dtype(name)
    first = compare_cast(values, tensor, dtype, copy)
    # A second allocation may clone a String descriptor. Keep the first
    # output alive and consume it again after populating the second owner.
    second = compare_cast(values, tensor, dtype, copy)
    assert not np.shares_memory(first, second)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = values.astype(dtype, copy=False)
    assert_array_values(first, expected)
    assert first.flags.owndata == second.flags.owndata == expected.flags.owndata


def compare_cast(values, tensor, dtype, copy=None):
    with warnings.catch_warnings(record=True) as old:
        warnings.simplefilter("always")
        expected = values.astype(dtype, copy=False)
    with warnings.catch_warnings(record=True) as new:
        warnings.simplefilter("always")
        actual = candidate(tensor, dtype, copy).assert_native()
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.flags.writeable == expected.flags.writeable
    assert_array_values(actual, expected)
    assert [(type(w.message), str(w.message)) for w in new] == [(type(w.message), str(w.message)) for w in old]
    return actual


@pytest.mark.parametrize("name", ["unset", "none", "text", "float", "complex", "numpy", "nat"])
@pytest.mark.parametrize("coerce", [False, True])
@pytest.mark.parametrize("copy", [None, False, True])
def test_native_string_preserves_missing_value_coerce_and_copy(resident, name, coerce, copy):
    values, tensor = resident
    metadata = {} if name == "unset" else {"na_object": missing_value(name)}
    dtype = np.dtypes.StringDType(coerce=coerce, **metadata)
    actual = compare_cast(values, tensor, dtype, copy)
    assert actual.dtype.coerce == coerce
    if metadata:
        assert actual.dtype.na_object is metadata["na_object"]


@pytest.mark.parametrize("name", ["integer", "finite", "bytes", "tuple", "list", "dict", "rawvoid"])
def test_native_string_preserves_ordinary_native_metadata(resident, name):
    values, tensor = resident
    missing = missing_value(name)
    dtype = np.dtypes.StringDType(na_object=missing)
    actual = compare_cast(values, tensor, dtype)
    assert actual.dtype.na_object is missing


@pytest.mark.parametrize("schema", ["T", "2T", ("T", (2,)), ("T", 0)])
def test_native_string_preserves_alias_and_subarray_layouts(resident, schema):
    values, tensor = resident
    compare_cast(values, tensor, schema)


@pytest.mark.parametrize("name", ["Float32DType", "UInt8DType", "StringDType", "StrDType", "BytesDType", "VoidDType", "DateTime64DType", "TimeDelta64DType"])
def test_native_descriptor_classes_preserve_top_level_cast_semantics(resident, name):
    values, tensor = resident
    compare_cast(values, tensor, getattr(np.dtypes, name))


@pytest.mark.parametrize("name", ["Float32DType", "StringDType"])
def test_native_descriptor_classes_inside_schemas_keep_object_semantics(resident, name):
    values, tensor = resident
    compare_cast(values, tensor, {"names": ["value"], "formats": [getattr(np.dtypes, name)]})


def signaling_missing(name):
    if name == "float":
        return struct.unpack("=d", struct.pack("=Q", 0x7ff0000000000001))[0]
    if name == "complex":
        return complex(float("nan"), 0)
    if name == "numpy16":
        return np.array([0x7c01], dtype="u2").view("f2")[0]
    if name == "numpy32":
        return np.array([0x7f800001], dtype="u4").view("f4")[0]
    if name == "numpy64":
        return np.array([0x7ff0000000000001], dtype="u8").view("f8")[0]
    if name == "numpy_complex64":
        return np.array([0x7f800001, 0], dtype="u4").view("c8")[0]
    raise AssertionError(name)


@pytest.mark.parametrize("name", ["float", "complex", "numpy16", "numpy32", "numpy64", "numpy_complex64"])
@pytest.mark.parametrize("word", [0x3fa00000, 0x7fc00001, 0x7f800001])
@pytest.mark.parametrize("mode", ["ignore", "warn", "raise"])
def test_native_string_preserves_signaling_metadata_status(resident, name, word, mode):
    # The module fixture performs the supported-target skip before uploading
    # these individual words. Separating words exposes sentinel-only flags.
    values = np.array([word], dtype="u4").view("f4")
    tensor = to_device(values, numeric_mode="identical")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dtype = np.dtypes.StringDType(na_object=signaling_missing(name))
    with warnings.catch_warnings(record=True) as old, np.errstate(invalid=mode):
        warnings.simplefilter("always")
        try:
            expected = values.astype(dtype, copy=False)
            failure = None
        except Exception as error:
            failure = (type(error), str(error))
    with warnings.catch_warnings(record=True) as new, np.errstate(invalid=mode):
        warnings.simplefilter("always")
        audit = candidate(tensor, dtype)
    assert_native_boundary(audit)
    if failure is None:
        actual = audit.assert_native()
        assert actual.tolist() == expected.tolist()
    else:
        assert (type(audit.exception), str(audit.exception)) == failure
    assert [(type(w.message), str(w.message)) for w in new] == [(type(w.message), str(w.message)) for w in old]


def test_native_string_refuses_missing_value_hooks_before_allocation(resident):
    _, tensor = resident
    calls = []
    class Missing:
        def __ne__(self, other):
            calls.append("ne")
            return True
        def __eq__(self, other):
            calls.append("eq")
            return False
        def __str__(self):
            calls.append("str")
            return "missing"
    dtype = np.dtypes.StringDType(na_object=Missing())
    np.ndarray((1,), dtype=dtype)  # Ensure a subsequent allocation would clone.
    calls.clear()
    audit = candidate(tensor, dtype)
    assert type(audit.exception) is TypeError
    assert not calls
    assert_native_boundary(audit)


@pytest.mark.parametrize("kind", ["structured_void", "array"])
def test_native_string_refuses_metadata_with_transitive_python_hooks(resident, kind):
    _, tensor = resident
    missing = (np.array([(1., 2)], dtype=[("a", "f4"), ("b", "i4")])[0]
               if kind == "structured_void" else np.array([1., 2.]))
    dtype = np.dtypes.StringDType(na_object=missing)
    np.ndarray((1,), dtype=dtype)
    audit = candidate(tensor, dtype)
    assert type(audit.exception) is TypeError
    assert_native_boundary(audit)


@pytest.mark.parametrize("words", [
    [0x7f800001, 0x3fa00000], [0x3fa00000, 0x7f800001],
    [0x7f800001, 0x7f800001], [0x7f800001, 0x7fc00001],
], ids=["signaling-finite", "finite-signaling", "two-signaling", "signaling-quiet"])
@pytest.mark.parametrize("mode", ["ignore", "warn", "raise"])
@pytest.mark.parametrize("action", ["always", "error"])
def test_native_string_preserves_element_order_and_warning_error_outcome(resident, words, mode, action):
    # The prior test assumed every complex-NaN cast raises under an error
    # filter. NumPy clears earlier equality flags at each following element:
    # terminal and nonterminal sNaNs therefore have different outcomes.
    values = np.array(words, dtype="u4").view("f4")
    tensor = to_device(values, numeric_mode="identical")
    dtype = np.dtypes.StringDType(na_object=complex(float("nan"), 0))
    with warnings.catch_warnings(record=True) as old, np.errstate(invalid=mode):
        warnings.simplefilter(action)
        try:
            expected = values.astype(dtype, copy=False)
            failure = None
        except Exception as error:
            failure = error
    with warnings.catch_warnings(record=True) as new, np.errstate(invalid=mode):
        warnings.simplefilter(action)
        audit = candidate(tensor, dtype)
    assert_native_boundary(audit)
    if failure is None:
        assert_array_values(audit.assert_native(), expected)
    else:
        assert (type(audit.exception), str(audit.exception)) == (type(failure), str(failure))
    assert [(type(w.message), str(w.message)) for w in new] == [(type(w.message), str(w.message)) for w in old]


@pytest.mark.parametrize("layout", ["T_O", "O_T", "T_f2", "f2_T", "T_O_T", "O_T_O"])
@pytest.mark.parametrize("words", [[0x7f800001, 0x3fa00000], [0x3fa00000, 0x7f800001], [0x60ad78ec, 0x3fa00000]],
                         ids=["signaling-finite", "finite-signaling", "overflow-finite"])
def test_native_string_preserves_status_across_structured_fields(resident, layout, words):
    values = np.array(words, dtype="u4").view("f4")
    tensor = to_device(values, numeric_mode="identical")
    strings = np.dtypes.StringDType(na_object=complex(float("nan"), 0))
    formats = [strings if kind == "T" else kind for kind in layout.split("_")]
    dtype = {"names": [f"f{i}" for i in range(len(formats))], "formats": formats}
    with np.errstate(all="warn"):
        compare_cast(values, tensor, dtype)


def test_native_string_refuses_replaced_class_before_hooks(resident, monkeypatch):
    _, tensor = resident
    calls = []
    def replacement(*args, **kwargs):
        calls.append("class")
    monkeypatch.setattr(np.dtypes, "StringDType", replacement)
    audit = candidate(tensor, "T")
    assert type(audit.exception) is TypeError
    assert not calls
    assert_native_boundary(audit)


def test_native_string_refuses_foreign_storage_capsule(resident, monkeypatch):
    import ctypes
    _, tensor = resident
    words = (ctypes.c_void_p * 400)()
    constructor = ctypes.pythonapi.PyCapsule_New
    constructor.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    constructor.restype = ctypes.py_object
    capsule = constructor(ctypes.addressof(words), None, None)
    monkeypatch.setattr(np._core._multiarray_umath, "_ARRAY_API", capsule)
    audit = candidate(tensor, "T")
    assert type(audit.exception) is TypeError
    assert_native_boundary(audit)
