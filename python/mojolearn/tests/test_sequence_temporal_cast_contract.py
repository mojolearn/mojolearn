"""Source-intended NaN -> NaT semantics, with strict ordinary cast controls."""
import warnings

import numpy as np
import pytest

from mojolearn._training_impl import _OPT_NATIVE, _OPT_CONTEXT
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn._x_sequence_device import SequenceDeviceTensor, to_device


def call(values, dtype, mode):
    tensor = to_device(values, numeric_mode="identical")
    if type(tensor) is not SequenceDeviceTensor:
        pytest.skip("temporal candidate qualification requires a device tensor")
    with warnings.catch_warnings(record=True) as caught, np.errstate(invalid=mode):
        warnings.simplefilter("always")
        result = audit_native_call(_OPT_NATIVE, "seqdev_array", tensor, (dtype, None), None, _OPT_CONTEXT)
    report = result.report()
    assert not report["python_frames"] and not report["forbidden_native_calls"], report
    assert not report["unobserved_existing_threads"], report
    return result, [(type(w.message).__name__, str(w.message)) for w in caught]


@pytest.mark.parametrize("name", ["DateTime64DType", "TimeDelta64DType"])
@pytest.mark.parametrize("mode", ["ignore", "warn", "raise"])
@pytest.mark.parametrize("words", [
    [0x7fc00001, 0xffc00001],
    [0x7f800001, 0xff800001],
    [0x3fa00000, 0x7f800001, 0xff800001, 0x7fc00001],
], ids=["quiet", "signaling", "mixed"])
def test_native_temporal_nan_to_nat_has_no_cast_warning(name, mode, words, record_property):
    # NumPy's temporal cast explicitly says "Avoid undefined behaviour and
    # warning for NaN -> NaT". Its old ARM float isnan instruction may set
    # invalid for an sNaN before choosing NaT; newer builds classify by bits.
    # This test adopts the stated algorithm contract, not that old compiler
    # side effect. No warning is filtered out of the product execution.
    # https://github.com/numpy/numpy/blob/v1.26.4/numpy/core/src/multiarray/arraytypes.c.src#L1317-L1327
    values = np.array(words, dtype="u4").view("f4")
    dtype = getattr(np.dtypes, name)
    with np.errstate(invalid="ignore"):
        expected = values.astype(dtype, copy=False)
    with warnings.catch_warnings(record=True) as observed, np.errstate(invalid=mode):
        warnings.simplefilter("always")
        try:
            values.astype(dtype, copy=False)
            failure = None
        except FloatingPointError as error:
            failure = (type(error).__name__, str(error))
    record_property("numpy_nan_to_nat_warnings", repr([(type(w.message).__name__, str(w.message)) for w in observed]))
    record_property("numpy_nan_to_nat_exception", repr(failure))
    result, caught = call(values, dtype, mode)
    actual = result.assert_native()
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert actual.flags.writeable == expected.flags.writeable
    assert actual.tobytes() == expected.tobytes()
    nan_mask = (values.view("u4") & np.uint32(0x7fffffff)) > np.uint32(0x7f800000)
    assert np.all(actual.view("u8")[nan_mask] == np.uint64(0x8000000000000000))
    assert not caught


@pytest.mark.parametrize("name", ["DateTime64DType", "TimeDelta64DType"])
@pytest.mark.parametrize("mode", ["ignore", "warn", "raise"])
@pytest.mark.parametrize("values", [
    [0., -0., 1.25, -1.25, 2 ** 62, -(2 ** 63)],
    [np.inf], [-np.inf], [2 ** 63], [-(2 ** 64)],
], ids=["ordinary", "positive-infinity", "negative-infinity", "positive-overflow", "negative-overflow"])
def test_native_temporal_non_nan_controls_keep_exact_warning_contract(name, mode, values, record_property):
    values = np.array(values, dtype="f4")
    dtype = getattr(np.dtypes, name)
    with warnings.catch_warnings(record=True) as caught, np.errstate(invalid=mode):
        warnings.simplefilter("always")
        try:
            expected = values.astype(dtype, copy=False)
            failure = None
        except FloatingPointError as error:
            failure = (type(error), str(error))
    result, actual_warnings = call(values, dtype, mode)
    assert actual_warnings == [(type(w.message).__name__, str(w.message)) for w in caught]
    if failure is not None:
        assert (type(result.exception), str(result.exception)) == failure
    else:
        actual = result.assert_native()
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        assert actual.flags.writeable == expected.flags.writeable
        # Ordinary inputs have defined truncated int64 values. Out-of-range
        # float-to-int values are unspecified, but dtype/shape/status above
        # remain checked and both complete word sequences remain evidence.
        record_property("numpy_output_words", repr(expected.view("u8").tolist()))
        record_property("native_output_words", repr(actual.view("u8").tolist()))
        defined = np.isfinite(values) & (values >= -(2 ** 63)) & (values < 2 ** 63)
        assert actual[defined].tobytes() == expected[defined].tobytes()
