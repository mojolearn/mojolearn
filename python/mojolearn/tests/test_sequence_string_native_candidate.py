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
    assert actual.tolist() == expected.tolist()
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


@pytest.mark.parametrize("name", ["Float32DType", "UInt8DType", "StringDType"])
def test_native_descriptor_classes_keep_object_output(resident, name):
    values, tensor = resident
    descriptor_class = getattr(np.dtypes, name)
    expected = values.astype(descriptor_class, copy=False)
    actual = candidate(tensor, descriptor_class).assert_native()
    assert actual.dtype == expected.dtype == np.dtype(object)
    assert actual.shape == expected.shape
    assert [struct.pack("=d", value) for value in actual] == [struct.pack("=d", value) for value in expected]


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


def test_native_string_warning_error_reports_final_cast(resident):
    values, tensor = resident
    dtype = np.dtypes.StringDType(na_object=complex(float("nan"), 0))
    with warnings.catch_warnings(), np.errstate(invalid="warn"):
        warnings.simplefilter("error")
        with pytest.raises(RuntimeWarning) as old:
            values.astype(dtype)
        audit = candidate(tensor, dtype)
    assert type(audit.exception) is RuntimeWarning
    assert str(audit.exception) == str(old.value) == "invalid value encountered in cast"
    assert_native_boundary(audit)


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
