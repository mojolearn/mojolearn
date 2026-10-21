# SPDX-License-Identifier: Apache-2.0
"""Independent wire descriptors, numerical byte oracles and runtime audits."""
import math

import numpy as np
import pytest

from mojolearn import Array, _backend
from mojolearn.lowbit import BF16Weight, Int8Weight
from mojolearn._verify_native_runtime import audit_native_call


@pytest.fixture
def codec():
    operation = _backend.load_host_module('_mojolearn_tokenizer_host').native_codec_call
    operation('initialize', (Array, BF16Weight, Int8Weight))
    return operation


def test_native_pack_and_unpack_use_independent_array_descriptor(codec):
    data = np.array([[1., -0., 3.], [4., 5., 6.]], np.float32)
    encoded = audit_native_call(codec, 'pack', (data, 1024)).assert_native()
    header, payload = encoded
    assert header == dict(kind='array', begin=0, size=24, shape=[2, 3],
                          dtype='<f4', order='C', readonly=False)
    assert payload == data.tobytes()
    # Decode an independently authored descriptor, not only our own serializer.
    reference = dict(kind='array', begin=0, size=8, shape=[2],
                     dtype='<i4', order='C', readonly=False)
    result = audit_native_call(codec, 'unpack', (reference, b'\x07\0\0\0\xfd\xff\xff\xff', 8)).assert_native()
    assert np.asarray(result).tolist() == [7, -3]


def test_nested_state_tuple_bytes_nonfinite_metadata_and_empty_arrays(codec):
    source = {'layers': (None, True, 7, -0., float('inf'), -float('inf'), float('nan')),
              'bytes': b'\xff\0', 'mutable': bytearray(b'abc'),
              'empty': np.empty((0, 3), np.float32), 8: ['x', 'λ']}
    header, payload = audit_native_call(codec, 'pack', (source, 1024)).assert_native()
    result = audit_native_call(codec, 'unpack', (header, payload, 1024)).assert_native()
    assert isinstance(result['layers'], tuple)
    assert result['layers'][:3] == (None, True, 7)
    assert math.copysign(1, result['layers'][3]) < 0
    assert result['layers'][4:6] == (float('inf'), -float('inf'))
    assert math.isnan(result['layers'][6])
    assert result['bytes'] == b'\xff\0' and type(result['bytes']) is bytes
    assert result['mutable'] == bytearray(b'abc') and type(result['mutable']) is bytearray
    assert result['empty'].shape == (0, 3)
    assert result[8] == ['x', 'λ']


def test_native_array_fortran_layout_and_lowbit_types_survive_without_callbacks(codec):
    f = Array.from_buffer(np.asfortranarray(np.arange(6, dtype=np.float32).reshape(2, 3)))
    bf = BF16Weight(np.array([[0x3f80, 0x4000]], np.uint16))
    i8 = Int8Weight(np.array([[1, -2], [3, -4]], np.int8), np.array([-1, 2], np.int32))
    header, payload = audit_native_call(codec, 'pack', ([f, bf, i8], 1024)).assert_native()
    result = audit_native_call(codec, 'unpack', (header, payload, 1024)).assert_native()
    assert result[0].order == 'F'
    np.testing.assert_array_equal(np.asarray(result[0]), np.asarray(f))
    assert type(result[1]) is BF16Weight and type(result[2]) is Int8Weight
    np.testing.assert_array_equal(np.asarray(result[1].bits), np.asarray(bf.bits))
    np.testing.assert_array_equal(np.asarray(result[2].codes), np.asarray(i8.codes))
    np.testing.assert_array_equal(np.asarray(result[2].exponents), np.asarray(i8.exponents))


@pytest.mark.parametrize('descriptor,payload,limit', [
    (dict(kind='bytes', begin=1, size=1), b'ab', 2),
    (dict(kind='bytes', begin=0, size=3), b'ab', 3),
    (dict(kind='array', begin=0, size=4, shape=[2], dtype='<f4', order='C', readonly=False), b'1234', 8),
    (dict(kind='array', begin=0, size=0, shape=[1 << 63], dtype='<f4', order='C', readonly=False), b'', 8),
    (dict(kind='dict', items=[['a', 1], ['a', 2]]), b'', 8),
    (dict(kind='float', value='arbitrary'), b'', 8),
    (None, b'unclaimed', 32),
])
def test_malformed_native_descriptor_fails_without_python_work(codec, descriptor, payload, limit):
    audited = audit_native_call(codec, 'unpack', (descriptor, payload, limit))
    assert isinstance(audited.exception, ValueError), audited.report()
    report = audited.report()
    for key in ('python_frames', 'forbidden_native_calls', 'wrapper_work', 'unobserved_existing_threads'):
        assert not report[key], report
