# SPDX-License-Identifier: Apache-2.0
"""Native buffer scalar conversion through public fit and archive output."""
import io
import struct

import numpy as np
import pytest

from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.tests import _legacy_host_iforest_fit as legacy
from mojolearn.tests.test_host_archive_native import _state
from mojolearn.tests.test_host_iforest_fit_native import _model, _pair
from mojolearn.tests.test_host_kde_fit_native import _x


@pytest.mark.parametrize('dtype', ['i1', 'u1', 'i2', 'u2', 'i4', 'u4', 'i8', 'u8', 'q', 'Q', 'f2', 'f4', 'f8', 'g'])
@pytest.mark.parametrize('array', [False, True])
def test_fit_integer_options_decode_native_storage(dtype, array):
    value = np.dtype(dtype).type(2.75)
    if array:
        value = np.asarray(value)
    _pair(_x(), max_depth=value)


@pytest.mark.parametrize('dtype', ['f2', 'f4', 'f8', 'g'])
@pytest.mark.parametrize('big_endian', [False, True])
def test_fit_float_options_preserve_native_rounding_and_endianness(dtype, big_endian):
    # The adjacent representable value exercises a consumed saved float,
    # including longdouble->double rounding on Linux extended formats.
    value = np.nextafter(np.dtype(dtype).type(.5), np.dtype(dtype).type(1.))
    value = np.asarray(value, dtype=np.dtype(dtype).newbyteorder('>' if big_endian else '='))
    _pair(_x(), max_features=value)


@pytest.mark.parametrize('dtype', ['f2', 'f4', 'f8', 'g'])
def test_fit_integer_conversion_does_not_round_before_truncation(dtype):
    value = np.nextafter(np.dtype(dtype).type(2), np.dtype(dtype).type(0))
    assert int(value) == 1
    _, model = _pair(_x(), max_depth=np.asarray(value))
    assert model._max_depth == 1


@pytest.mark.parametrize('dtype,value', [('f2', -0.), ('f4', 1e-30), ('f8', -0.),
                                       ('g', '1e-300'), ('F', 1j), ('D', -0j), ('G', 1j)])
def test_fit_truth_reads_native_scalar_components(dtype, value):
    # Complex truth examines both components without calling NumPy bool.
    _pair(_x(), bootstrap=np.asarray(value, dtype=dtype))


@pytest.mark.parametrize('dtype', ['f2', 'f4', 'f8', 'g'])
@pytest.mark.parametrize('value', ['nan', 'inf', '-inf'])
def test_fit_nonfinite_integer_conversion_keeps_error_class_and_text(dtype, value):
    supplied = np.asarray(value, dtype=dtype)
    old, model = _model(max_depth=supplied), _model(max_depth=supplied)
    with pytest.raises((ValueError, OverflowError)) as expected:
        legacy.fit(old, _x())
    audit = audit_native_call(model.fit, _x())
    assert type(audit.exception) is type(expected.value), audit.report()
    assert str(audit.exception) == str(expected.value)
    assert not audit.report()['python_frames'] and not audit.report()['forbidden_native_calls']
    assert _state(vars(model)) == _state(vars(old))


@pytest.mark.parametrize('dtype,bits,array', [
    pytest.param('f4', 0x7f800001, False, id='f4-snan'),
    pytest.param('f8', 0x7ff0000000000001, False, id='f8-snan'),
    pytest.param('f4', 0x80000000, False, id='f4-negative-zero'),
    pytest.param('f8', 1, False, id='f8-smallest-subnormal'),
    pytest.param('u8', 0xffffffffffffffff, False, id='u8-max'),
    pytest.param('i8', 0x8000000000000000, False, id='i8-min'),
    pytest.param('f4', 0x7f800001, True, id='zero-d-f4-snan'),
    pytest.param('f8', 0x8000000000000000, True, id='zero-d-f8-negative-zero'),
])
def test_save_consumes_exact_native_scalar_conversion_bytes(dtype, bits, array):
    from mojolearn.tests.test_host_density_save_native import _prepared

    # Raw storage preserves signaling NaNs until the conversion under test.
    storage_type = np.dtype(dtype).newbyteorder('<')
    scalar = np.frombuffer(bits.to_bytes(storage_type.itemsize, 'little'),
                           dtype=storage_type).copy()[0]
    value = np.array(scalar) if array else scalar
    expected = struct.pack('<d', float(value))
    model = _prepared('kde')
    model.bandwidth = value
    output = io.BytesIO()
    audit = audit_native_call(model.save, output)
    assert audit.assert_native() is output
    assert model.bandwidth is value
    # Save accepts these metadata values; inference/load validation is a
    # separate interface. Consume the saved bytes, not just error/truth state.
    with np.load(io.BytesIO(output.getvalue()), allow_pickle=False) as saved:
        assert saved['bandwidth'].tobytes() == expected
