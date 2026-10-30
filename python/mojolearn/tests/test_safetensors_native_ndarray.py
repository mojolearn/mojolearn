"""Native widening admits ndarray bits without invoking array protocols."""
import numpy as np
import pytest

from mojolearn import Array
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.models.safetensors import widen_f16


def expected_word(half):
    half = int(half)
    sign, exponent, fraction = (half & 0x8000) << 16, (half >> 10) & 31, half & 1023
    if exponent == 31:
        return sign | 0x7f800000 | (fraction << 13)
    if exponent:
        return sign | ((exponent + 112) << 23) | (fraction << 13)
    if not fraction:
        return sign
    exponent = 113
    while not fraction & 1024:
        fraction <<= 1
        exponent -= 1
    return sign | (exponent << 23) | ((fraction & 1023) << 13)


@pytest.mark.parametrize('dtype', [np.uint16, np.float16])
def test_every_half_pattern_preserves_sign_and_nan_payload(dtype):
    words = np.arange(65536, dtype=np.uint16)
    bits = words.view(dtype)
    before = bits.tobytes()
    result = audit_native_call(widen_f16, bits).assert_native()
    actual = np.asarray(result)
    assert actual.shape == bits.shape and actual.dtype == np.float32
    np.testing.assert_array_equal(actual.view(np.uint32),
                                  np.array([expected_word(x) for x in words], dtype=np.uint32))
    assert bits.tobytes() == before


@pytest.mark.parametrize('case', ['reverse', 'transpose', 'empty', 'scalar'])
def test_captured_ndarray_shape_and_strides(case):
    bits = np.array([[0, 1, 0x8000], [0x3c00, 0x7c00, 0x7c01]], dtype=np.uint16)
    if case == 'reverse': bits = bits[::-1, ::-1]
    if case == 'transpose': bits = bits.T
    if case == 'empty': bits = bits[:0]
    if case == 'scalar': bits = np.array(0x8001, dtype=np.uint16)
    bits.flags.writeable = False
    actual = np.asarray(widen_f16(bits))
    expected = np.array([expected_word(x) for x in bits.flat], dtype=np.uint32).reshape(bits.shape)
    assert actual.shape == bits.shape
    np.testing.assert_array_equal(actual.view(np.uint32), expected)


def test_native_array_input_still_keeps_its_descriptor():
    words = np.array([0, 1, 0x8000, 0x7c01], dtype=np.uint16)
    bits = Array.from_buffer(words.reshape(2, 2))
    actual = np.asarray(widen_f16(bits))
    assert bits.dtype == '<u2' and bits.shape == (2, 2)
    np.testing.assert_array_equal(actual.view(np.uint32),
        np.array([expected_word(x) for x in words], dtype=np.uint32).reshape(2, 2))


@pytest.mark.parametrize('dtype', [np.float32, np.int16, '>u2'])
def test_other_storage_is_refused(dtype):
    with pytest.raises(TypeError, match='float16|uint16'):
        widen_f16(np.zeros((2, 3), dtype=dtype))


def test_subclass_and_foreign_array_hooks_never_execute():
    calls = []

    class Foreign:
        @property
        def __array_interface__(self):
            calls.append('interface')
            raise AssertionError('foreign array hook ran')

    class Subclass(np.ndarray):
        def __array__(self, *args, **kwargs):
            calls.append('array')
            raise AssertionError('subclass array hook ran')

    for value in [Foreign(), np.zeros(3, dtype=np.uint16).view(Subclass)]:
        with pytest.raises(TypeError):
            widen_f16(value)
    assert calls == []
