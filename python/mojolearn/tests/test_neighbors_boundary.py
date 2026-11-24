# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Python API boundary tests of the neighbors family: refusals, exception types and words, returned types,
buffer admission and framework interop. Numerical results are held by the verify lanes."""

# moved from python/mojolearn/tests/test_native_scalar_decode_public.py
# Native buffer scalar conversion of KernelDensity metadata through the public save.
import io
import struct

import numpy as np
import pytest

from mojolearn._verify_native_runtime import audit_native_call


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


@pytest.mark.parametrize('value', ['negative_zero', 'adjacent_half', 'smallest_normal'])
def test_save_consumes_non_native_longdouble_scalar_bytes(value):
    from mojolearn.tests.test_host_density_save_native import _prepared

    scalar = (np.longdouble('-0') if value == 'negative_zero' else
              np.nextafter(np.longdouble('.5'), np.longdouble(1)) if value == 'adjacent_half' else
              np.finfo(np.longdouble).tiny)
    supplied = np.asarray(scalar, dtype=np.dtype('g').newbyteorder('>'))
    expected = struct.pack('<d', float(supplied))
    before = supplied.tobytes()
    model = _prepared('kde')
    model.bandwidth = supplied
    output = io.BytesIO()
    assert audit_native_call(model.save, output).assert_native() is output
    assert model.bandwidth is supplied and supplied.tobytes() == before
    with np.load(io.BytesIO(output.getvalue()), allow_pickle=False) as saved:
        assert saved['bandwidth'].tobytes() == expected
