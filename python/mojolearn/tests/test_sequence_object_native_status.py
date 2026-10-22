"""Native object output must preserve signaling-NaN words and cast status."""
import struct
import warnings

import numpy as np
import pytest

from mojolearn._training_impl import _OPT_NATIVE, _OPT_CONTEXT
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn._x_sequence_device import SequenceDeviceTensor, to_device


@pytest.fixture(scope="module")
def resident():
    values = np.array([0x3fa00000, 0x7f800001, 0xff800001, 0x7fc00001], dtype="u4").view("f4")
    tensor = to_device(values, numeric_mode="identical")
    if type(tensor) is not SequenceDeviceTensor:
        pytest.skip("object output qualification requires a device tensor")
    return values, tensor


def assert_values(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    if expected.dtype.names:
        for name in expected.dtype.names:
            assert_values(actual[name], expected[name])
    elif expected.dtype.kind == "O":
        assert [struct.pack("=d", value) for value in actual.flat] == [struct.pack("=d", value) for value in expected.flat]
    else:
        assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("dtype", [object, [("number", "f4"), ("boxed", "O")], [("boxed", "O"), ("number", "f4")]])
@pytest.mark.parametrize("mode", ["ignore", "warn", "raise"])
def test_native_object_output_preserves_signaling_status(resident, dtype, mode):
    values, tensor = resident
    with warnings.catch_warnings(record=True) as old, np.errstate(invalid=mode):
        warnings.simplefilter("always")
        try:
            expected = values.astype(dtype, copy=False)
            failure = None
        except Exception as error:
            failure = error
    with warnings.catch_warnings(record=True) as new, np.errstate(invalid=mode):
        warnings.simplefilter("always")
        audit = audit_native_call(_OPT_NATIVE, "seqdev_array", tensor, (dtype, None), None, _OPT_CONTEXT)
    report = audit.report()
    assert not report["python_frames"] and not report["forbidden_native_calls"], report
    assert not report["unobserved_existing_threads"], report
    if failure is None:
        assert_values(audit.assert_native(), expected)
    else:
        assert (type(audit.exception), str(audit.exception)) == (type(failure), str(failure))
    assert [(type(w.message), str(w.message)) for w in new] == [(type(w.message), str(w.message)) for w in old]


@pytest.mark.parametrize("name", ["Float32DType", "UInt8DType", "StrDType", "BytesDType", "VoidDType", "DateTime64DType", "TimeDelta64DType"])
def test_native_legacy_descriptor_classes_preserve_versioned_cast_contract(resident, name):
    # These classes exist on NumPy1.26 too. Keep this qualification outside
    # the StringDType module, whose module-level skip requires NumPy2.
    values, tensor = resident
    dtype = getattr(np.dtypes, name)
    with warnings.catch_warnings(record=True) as old:
        warnings.simplefilter("always")
        expected = values.astype(dtype, copy=False)
    with warnings.catch_warnings(record=True) as new:
        warnings.simplefilter("always")
        actual = audit_native_call(_OPT_NATIVE, "seqdev_array", tensor, (dtype, None), None, _OPT_CONTEXT).assert_native()
    assert_values(actual, expected)
    assert [(type(w.message), str(w.message)) for w in new] == [(type(w.message), str(w.message)) for w in old]
