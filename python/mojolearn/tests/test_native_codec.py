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
    _assert_native_rejection(audited)


def _assert_native_rejection(audited):
    assert isinstance(audited.exception, ValueError), audited.report()
    report = audited.report()
    for key in ('python_frames', 'forbidden_native_calls', 'wrapper_work', 'unobserved_existing_threads'):
        assert not report[key], report


@pytest.mark.parametrize('dtype', ['<f4', '<f8', '<f2', '<i4', '<i8', '<u4', '<u2', '<u1', '<i1'])
@pytest.mark.parametrize('order', ['C', 'F'])
@pytest.mark.parametrize('readonly', [False, True])
def test_independent_wire_roundtrip_preserves_every_dtype_order_and_readonly_flag(codec, dtype, order, readonly):
    # The expected bytes/header come from the wire contract, not codec_pack.
    source = np.arange(6, dtype=np.dtype(dtype)).reshape(2, 3)
    payload = source.tobytes(order=order)
    header = dict(kind='array', begin=0, size=len(payload), shape=[2, 3],
                  dtype=dtype, order=order, readonly=readonly)
    decoded = audit_native_call(codec, 'unpack', (header, payload, len(payload))).assert_native()
    assert decoded.order == order and decoded._readonly is readonly
    np.testing.assert_array_equal(np.asarray(decoded), source)
    encoded = audit_native_call(codec, 'pack', (decoded, len(payload))).assert_native()
    assert encoded == (header, payload)


@pytest.mark.parametrize('layout', ['fortran', 'strided', 'empty', 'scalar'])
def test_foreign_native_exporters_keep_logical_c_order_and_readonly(codec, layout):
    source = np.arange(12, dtype=np.float32).reshape(3, 4)
    if layout == 'fortran':
        source = np.asfortranarray(source)
    elif layout == 'strided':
        source = source[::-1, ::2]
    elif layout == 'empty':
        source = source[:0]
    else:
        source = source[0, 0:1].reshape(())
    source.flags.writeable = False
    header, payload = audit_native_call(codec, 'pack', (memoryview(source), 1024)).assert_native()
    assert header == dict(kind='array', begin=0, size=source.nbytes, shape=list(source.shape),
                          dtype='<f4', order='C', readonly=True)
    assert payload == source.tobytes(order='C')


@pytest.mark.parametrize('operation', ['pack', 'unpack'])
def test_foreign_dictionary_key_is_rejected_before_colliding_value_lookup(codec, operation):
    calls = []
    ordinary = 'kind' if operation == 'unpack' else 'safe'

    class ForeignKey:
        def __hash__(self):
            return hash(ordinary)

        def __eq__(self, other):
            calls.append(other)
            return False

    foreign = ForeignKey()
    source = {foreign: 0, ordinary: 'bytes' if operation == 'unpack' else 1}
    # Reinsert changes iteration order while reusing the earlier probe slot.
    # Prove the collision is active using a test-only lookup, then clear it.
    del source[foreign]
    source[foreign] = 0
    calls.clear()
    _ = source[ordinary]
    assert calls
    calls.clear()
    args = (source, b'', 64) if operation == 'unpack' else (source, 64)
    audited = audit_native_call(codec, operation, args)
    assert not calls
    _assert_native_rejection(audited)


@pytest.mark.parametrize('field', ['_mv', 'size', 'dtype', '_readonly'])
def test_native_array_mutated_metadata_is_rejected_without_protocol_calls(codec, field):
    calls = []

    class Foreign:
        def __buffer__(self, flags):
            calls.append('buffer')
            return memoryview(b'abcd')

        def __int__(self):
            calls.append('int')
            return 1

        def __str__(self):
            calls.append('str')
            return '<f4'

        def __bool__(self):
            calls.append('bool')
            return False

    value = Array.from_buffer(np.array([1.0], np.float32))
    setattr(value, field, Foreign())
    audited = audit_native_call(codec, 'pack', (value, 64))
    assert not calls
    _assert_native_rejection(audited)


def test_foreign_exporter_is_rejected_before_buffer_protocol(codec):
    calls = []

    class Foreign:
        def __buffer__(self, flags):
            calls.append(flags)
            return memoryview(b'abcd')

    audited = audit_native_call(codec, 'pack', (Foreign(), 64))
    assert not calls
    _assert_native_rejection(audited)


@pytest.mark.parametrize('owner', [Array, BF16Weight, Int8Weight])
def test_mutated_output_setter_is_rejected_before_publication(codec, monkeypatch, owner):
    calls = []
    array = dict(kind='array', begin=0, size=4, shape=[1, 2],
                 dtype='<u2', order='C', readonly=False)
    descriptor, payload = array, b'\x80\x3f\0\x40'
    if owner is BF16Weight:
        descriptor = dict(kind='bf16', bits=array)
    elif owner is Int8Weight:
        descriptor = dict(kind='int8',
                          codes=dict(kind='array', begin=0, size=2, shape=[1, 2],
                                     dtype='<i1', order='C', readonly=False),
                          exponents=dict(kind='array', begin=2, size=4, shape=[1],
                                         dtype='<i4', order='C', readonly=False))
        payload = b'\x01\xfe\0\0\0\0'

    def setter(self, name, value):
        calls.append(name)

    with monkeypatch.context() as patch:
        patch.setattr(owner, '__setattr__', setter)
        audited = audit_native_call(codec, 'unpack', (descriptor, payload, 64))
    assert not calls
    _assert_native_rejection(audited)


@pytest.mark.parametrize('owner,field', [(BF16Weight, 'bits'), (Int8Weight, 'codes')])
def test_packed_replaced_field_descriptor_is_rejected_before_read(codec, monkeypatch, owner, field):
    calls = []
    value = object.__new__(owner)

    def fields(self):
        calls.append(field)
        return {}

    with monkeypatch.context() as patch:
        patch.setattr(owner, field, property(fields), raising=False)
        audited = audit_native_call(codec, 'pack', (value, 64))
    assert not calls
    _assert_native_rejection(audited)


@pytest.mark.parametrize('packed_index', [1, 2])
def test_initialize_refuses_packed_dictionary_properties_without_reading_them(codec, packed_index):
    calls = []

    class ForeignPacked:
        @property
        def __dict__(self):
            calls.append('dict')
            return {}

    classes = [Array, None, None]
    classes[packed_index] = ForeignPacked
    audited = audit_native_call(codec, 'initialize', tuple(classes))
    assert not calls
    _assert_native_rejection(audited)


@pytest.mark.parametrize('target', ['operation', 'arguments', 'limit', 'payload', 'tag'])
def test_entry_and_descriptor_protocols_are_rejected_before_conversion(codec, target):
    calls = []

    class Foreign:
        def __str__(self):
            calls.append('str')
            return 'pack'

        def __getitem__(self, index):
            calls.append('getitem')
            return (None, 64)[index]

        def __len__(self):
            calls.append('len')
            return 0

        def __int__(self):
            calls.append('int')
            return 64

        __index__ = __int__

    foreign = Foreign()
    operation, args = 'pack', (None, 64)
    if target == 'operation':
        operation = foreign
    elif target == 'arguments':
        args = foreign
    elif target == 'limit':
        args = (None, foreign)
    elif target == 'payload':
        operation, args = 'unpack', (None, foreign, 64)
    else:
        operation, args = 'unpack', (dict(kind=foreign), b'', 64)
    audited = audit_native_call(codec, operation, args)
    assert not calls
    _assert_native_rejection(audited)
