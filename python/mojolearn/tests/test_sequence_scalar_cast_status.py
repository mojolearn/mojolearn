"""Preserve native scalar cast words and signaling status across NumPy versions."""
import warnings

import numpy as np
import pytest

from mojolearn._training_impl import _OPT_NATIVE, _OPT_CONTEXT
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn._x_sequence_device import SequenceDeviceTensor, to_device


WORDS = [
    [0x7FC00001], [0x7F800001], [0xFF800001],
    [0x7F803FFF, 0x7FC12345, 0xFFBFFFFF],
    [0, 0x80000000, 0x3FA00000, 0x7F800001, 0x7F800000],
]


@pytest.mark.parametrize("words", WORDS,
                         ids=["quiet", "signaling", "negative-signaling", "payloads", "mixed"])
@pytest.mark.parametrize("dtype", ["?", "f2"])
@pytest.mark.parametrize("mode", ["ignore", "warn", "raise"])
def test_native_scalar_cast_preserves_versioned_nan_words_and_status(words, dtype, mode):
    # Test preparation and the independent NumPy oracle are outside the
    # candidate operation. The native call includes download and conversion.
    values = np.array(words, dtype="u4").view("f4")
    tensor = to_device(values, numeric_mode="identical")
    if type(tensor) is not SequenceDeviceTensor:
        pytest.skip("scalar output qualification requires a device tensor")
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
    for field in ("python_frames", "forbidden_native_calls", "wrapper_work", "unobserved_existing_threads"):
        assert not report[field], report
    if failure is None:
        actual = audit.assert_native()
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        assert actual.tobytes() == expected.tobytes()
    else:
        assert (type(audit.exception), str(audit.exception)) == (type(failure), str(failure))
    assert [(type(w.message), str(w.message)) for w in new] == [
        (type(w.message), str(w.message)) for w in old]
