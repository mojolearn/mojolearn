# SPDX-License-Identifier: Apache-2.0
"""The public row materializer owns typed native resident snapshots."""
from array import array
import struct
import numpy as np
import pytest
from mojolearn._array import Array
from mojolearn._arena_io import DeviceRows
from mojolearn._buffer import _native
from mojolearn.model_selection import _Resident
from mojolearn._verify_native_runtime import audit_native_call


def _refused(audit, category=TypeError):
    assert type(audit.exception) is category, audit.report()
    assert not audit.report()['python_frames'] and not audit.report()['forbidden_native_calls'], audit.report()


@pytest.fixture
def owner():
    # Test setup/cleanup deliberately remains outside the public operation:
    # the existing _Resident constructor/close are separate migration debt.
    live = _native('msel_live')
    before = int(live())
    result = _Resident()
    try:
        yield result
    finally:
        result.close()
        assert int(live()) == before


@pytest.mark.parametrize('dtype', ['f4', 'f8', 'i8', 'i4', 'u1', 'i1', 'u2', 'i2', 'u8', 'f2', 'bool'])
@pytest.mark.parametrize('layout', ['contiguous', 'fortran', 'strided', 'bigendian'])
def test_materialize_preserves_typed_bytes_and_cached_identity(owner, dtype, layout):
    values = (np.arange(24).reshape(4, 6) % 2 if dtype == 'bool' else np.arange(24).reshape(4, 6)).astype(dtype)
    base = (np.asfortranarray(values) if layout == 'fortran' else values[:, ::2] if layout == 'strided'
            else values.astype(values.dtype.newbyteorder('>')) if layout == 'bigendian' else values)
    indices = np.array([3, 0, 2], dtype='i8')
    rows = DeviceRows(base, indices, owner.take)
    output = audit_native_call(rows.materialize).assert_native()
    natural = {'i2': '<i4', 'u8': '<i8', 'bool': '<u1'}
    expected_dtype = natural.get(dtype, '<' + dtype)
    if dtype == 'u2' and layout in ('strided', 'bigendian'):
        expected_dtype = '<u4'
    assert output.dtype == expected_dtype and output.shape == (3, base.shape[1])
    assert output.tobytes() == base[indices].astype(expected_dtype).tobytes()
    assert audit_native_call(rows.materialize).assert_native() is output
    assert len(owner._entries) == 2
    assert owner._entries[id(base)][0] is base and owner._entries[id(indices)][0] is indices
    if layout == 'contiguous' and dtype not in natural:
        assert owner._entries[id(base)][1]._store is base


@pytest.mark.parametrize('shape', [(4, 3), (4, 1)])
@pytest.mark.parametrize('order', ['C', 'F'])
def test_array_storage_preserves_alias_and_order_contract(owner, shape, order):
    values = np.arange(np.prod(shape), dtype='f8').reshape(shape)
    base = Array._owned(array('d', values.ravel(order=order).tolist()), shape, '<f8', order)
    indices = Array.from_list([3, 0], '<i8')
    output = audit_native_call(DeviceRows(base, indices, owner.take).materialize).assert_native()
    assert output.tobytes() == values[[3, 0]].tobytes()
    prepared = owner._entries[id(base)][1]
    if order == 'C':
        assert prepared is base
    elif shape[1] == 1:
        assert prepared._base is base and prepared._mv is base._mv
    else:
        assert prepared is not base and prepared.order == 'C'


@pytest.mark.parametrize('dtype', ['f4', 'f2'])
@pytest.mark.parametrize('layout', ['contiguous', 'fortran', 'strided', 'bigendian'])
def test_materialize_preserves_nan_payload_copy_rules(owner, dtype, layout):
    words = [0x7f800001, 0xff800001, 0x7fc01234, 0xffc03456, 0x80000000, 0] if dtype == 'f4' else [0x7c01, 0xfc01, 0x7e12, 0xfe34, 0x8000, 0]
    unit = 'u4' if dtype == 'f4' else 'u2'
    original = np.array(words * 4, dtype=unit).reshape(4, 6).view(dtype)
    base = (np.asfortranarray(original) if layout == 'fortran' else original[:, ::2] if layout == 'strided'
            else original.astype('>' + dtype) if layout == 'bigendian' else original)
    expected = original[:, ::2] if layout == 'strided' else original
    expected = expected[[0, 2]].view(unit).copy()
    if layout == 'strided':
        if dtype == 'f4':
            expected[(expected & 0x7fffffff) > 0x7f800000] |= 0x00400000
        elif struct.pack('<e', struct.unpack('<e', b'\x01\x7c')[0]) == b'\x00\x7e':
            mask = (expected & 0x7fff) > 0x7c00
            expected[mask] = (expected[mask] & 0x8000) | 0x7e00
    output = audit_native_call(DeviceRows(base, np.array([0, 2]), owner.take).materialize).assert_native()
    assert output.tobytes() == expected.tobytes()


@pytest.mark.parametrize('kind', ['rank1', 'rank3', 'empty_rows', 'empty_columns', 'empty_indices'])
def test_materialize_preserves_rank_and_zero_size(owner, kind):
    shape = {'rank1': (4,), 'rank3': (4, 2, 3), 'empty_rows': (0, 3), 'empty_columns': (3, 0), 'empty_indices': (4, 3)}[kind]
    base = np.arange(np.prod(shape), dtype='f4').reshape(shape)
    indices = np.array([] if kind == 'empty_indices' else [0], dtype='i8')
    output = audit_native_call(DeviceRows(base, indices, owner.take).materialize).assert_native()
    expected = np.zeros((len(indices), *shape[1:]), dtype='f4') if kind == 'empty_rows' else base[indices]
    assert output.shape == expected.shape and output.tobytes() == expected.tobytes()


@pytest.mark.parametrize('kind', ['integers', 'floats', 'numpy', 'longlong'])
def test_materialize_preserves_native_object_elements(owner, kind):
    values = ([[1, 2], [3, 4]] if kind == 'integers' else [[1., 2], [3, 4]] if kind == 'floats'
              else [[np.longlong(1), np.ulonglong(2)], [np.longlong(3), np.ulonglong(4)]] if kind == 'longlong'
              else [[np.float32(1), np.int64(2)], [np.float32(3), np.int64(4)]])
    base = np.array(values, dtype='O')
    output = audit_native_call(DeviceRows(base, np.array([1, 0]), owner.take).materialize).assert_native()
    dtype = '<i8' if kind == 'integers' else '<f8'
    assert output.dtype == dtype and output.tobytes() == np.array(values[::-1], dtype=dtype).tobytes()


@pytest.mark.parametrize('dtype', ['i4', 'f4', 'bool'])
def test_materialize_converts_native_indices(owner, dtype):
    base = np.arange(12, dtype='f4').reshape(4, 3)
    indices = np.array([True, False] if dtype == 'bool' else [3.9, 0.1] if dtype == 'f4' else [3, 0], dtype=dtype)
    output = audit_native_call(DeviceRows(base, indices, owner.take).materialize).assert_native()
    assert output.tobytes() == base[indices.astype('i8')].tobytes()
    assert owner._entries[id(indices)][1].dtype == '<i8'


@pytest.mark.parametrize('dtype', ['i8', 'f4', 'O'])
def test_scalar_index_keeps_existing_prepared_shape(owner, dtype):
    base = np.arange(12, dtype='f4').reshape(4, 3)
    indices = np.array(1, dtype=dtype)
    output = audit_native_call(DeviceRows(base, indices, owner.take).materialize).assert_native()
    assert output.tobytes() == base[[1]].tobytes()
    assert owner._entries[id(indices)][1].shape == (() if dtype == 'O' else (1,))


@pytest.mark.parametrize('dtype,values,category', [
    ('f4', [float('nan')], ValueError), ('f8', [float('inf')], OverflowError),
    ('f8', [1e30], TypeError), ('f8', [-1e30], TypeError),
    ('f8', [float('inf'), float('nan')], OverflowError), ('f8', [float('nan'), float('inf')], ValueError),
    ('f2', [float('inf')], OverflowError), ('O', [2**80], TypeError), ('O', [float(2**80)], TypeError),
    ('O', [np.bool_(True)], TypeError), ('i8', [-1], Exception), ('i8', [99], Exception),
])
def test_materialize_preserves_native_conversion_and_gather_errors(owner, dtype, values, category):
    rows = DeviceRows(np.arange(12, dtype='f4').reshape(4, 3), np.array(values, dtype=dtype), owner.take)
    _refused(audit_native_call(rows.materialize), category)
    assert rows._host is None


def test_materialize_reuses_resident_uploads_after_host_mutation(owner):
    base = Array.from_list([[1., 2.], [3., 4.], [5., 6.]], '<f4')
    indices = Array.from_list([0, 2], '<i8')
    first = audit_native_call(DeviceRows(base, indices, owner.take).materialize).assert_native()
    count = int(_native('msel_live')())
    base._mv[0], indices._mv[0] = 99., 1
    second = audit_native_call(DeviceRows(base, indices, owner.take).materialize).assert_native()
    assert first.tobytes() == second.tobytes()
    assert len(owner._entries) == 2 and int(_native('msel_live')()) == count


def test_materialize_does_not_call_python_resident_helpers(owner, monkeypatch):
    rows = DeviceRows(np.arange(12, dtype='f4').reshape(4, 3), np.array([2, 0]), owner.take)
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError('Python resident helper executed')
    with monkeypatch.context() as patch:
        patch.setattr(_Resident, '_entry', forbidden)
        patch.setattr(_Resident, 'take', forbidden)
        result = audit_native_call(rows.materialize).assert_native()
    assert not calls and result.tobytes() == np.array([[6., 7., 8.], [0., 1., 2.]], dtype='f4').tobytes()


def test_materialize_preserves_preexisting_cached_object(owner):
    rows = DeviceRows(np.arange(6, dtype='f4').reshape(2, 3), np.array([0]), owner.take)
    value = object()
    rows._host = value
    assert audit_native_call(rows.materialize).assert_native() is value
    assert not owner._entries


@pytest.mark.parametrize('hook', ['take', 'object_float', 'resident_entry', 'array_slot', 'row_key', 'cache_key'])
def test_materialize_refuses_application_hooks_before_execution(owner, hook, monkeypatch):
    calls = []
    def forbidden(*args):
        calls.append(hook)
        raise AssertionError('application hook ran')
    class Foreign:
        def __float__(self):
            forbidden()
    class ForeignKey(str):
        __hash__ = str.__hash__
        def __eq__(self, other):
            forbidden()
    class ForeignIdentity(int):
        __hash__ = int.__hash__
        def __eq__(self, other):
            forbidden()
    base = Array.from_list([[1., 2.], [3., 4.]], '<f4')
    indices = np.array([1, 0])
    rows = DeviceRows(base, indices, owner.take)
    with monkeypatch.context() as patch:
        if hook == 'take':
            rows._take = forbidden
        elif hook == 'object_float':
            rows.base = np.array([[Foreign()]], dtype='O')
        elif hook == 'resident_entry':
            owner._take = forbidden
        elif hook == 'array_slot':
            patch.setattr(Array, '_mv', property(forbidden))
        elif hook == 'row_key':
            # An instance's shared-key dictionary can retain the ordinary
            # 'base' key after pop, so reinsertion itself calls __eq__ before
            # the product boundary. Build the malformed dictionary afresh,
            # keeping the foreign key first, and retain the actual no-hook
            # rejection check below against the public materialize call.
            malformed = {ForeignKey('base'): rows.base}
            malformed.update((key, value) for key, value in vars(rows).items()
                             if key != 'base')
            object.__setattr__(rows, '__dict__', malformed)
            assert type(next(iter(vars(rows)))) is ForeignKey
            assert calls == []
        else:
            owner._entries[ForeignIdentity(id(base))] = None
        calls.clear()
        audit = audit_native_call(rows.materialize)
    _refused(audit)
    assert not calls
    if hook == 'cache_key':
        owner._entries.clear()  # controller teardown of deliberately malformed cache


@pytest.mark.parametrize('field,value', [('itemsize', 1), ('shape', (100, 3)), ('size', 300)])
def test_materialize_refuses_inconsistent_array_metadata_before_pointer_use(owner, field, value):
    base = Array.from_list([[1., 2., 3.], [4., 5., 6.]], '<f8')
    rows = DeviceRows(base, np.array([0]), owner.take)
    setattr(base, field, value)
    audit = audit_native_call(rows.materialize)
    _refused(audit, ValueError)
    assert not owner._entries


@pytest.mark.parametrize('hook', ['__new__', '__init__', '__setattr__'])
def test_materialize_allocates_array_storage_without_python_class_constructors(owner, monkeypatch, hook):
    rows = DeviceRows(np.arange(12, dtype='f4').reshape(4, 3), np.array([2, 0]), owner.take)
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(hook)
        raise AssertionError('Python Array construction hook ran')
    with monkeypatch.context() as patch:
        patch.setattr(Array, hook, staticmethod(forbidden) if hook == '__new__' else forbidden)
        result = audit_native_call(rows.materialize).assert_native()
    assert not calls and result.tobytes() == np.array([[6., 7., 8.], [0., 1., 2.]], dtype='f4').tobytes()


@pytest.mark.parametrize('constructor', ['bytes', 'list', 'tuple', 'object', 'memoryview'])
def test_materialize_uses_connected_native_output_constructors(owner, monkeypatch, constructor):
    import builtins
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(constructor)
        raise AssertionError('mutable builtin output hook ran')
    # The auditor itself constructs a list of disassembled wrapper operations.
    # Its inherited native constructor stays usable; product use of the new
    # subclass's length or indexing remains an observable forbidden callback.
    if constructor in ('list', 'tuple'):
        replacement = type('ForeignSequence', (getattr(builtins, constructor),),
                           {'__len__': forbidden, '__getitem__': forbidden})
    elif constructor == 'object':
        replacement = type('ForeignObject', (), {'__new__': staticmethod(forbidden)})
    else:
        replacement = forbidden
    rows = DeviceRows(np.arange(12, dtype='f4').reshape(4, 3), np.array([2, 0]), owner.take)
    with monkeypatch.context() as patch:
        patch.setattr(builtins, constructor, replacement)
        audit = audit_native_call(rows.materialize)
    output = audit.assert_native()
    assert not calls and output.tobytes() == np.array([[6., 7., 8.], [0., 1., 2.]], dtype='f4').tobytes()


@pytest.mark.parametrize('name', ['int', 'float', 'bool'])
def test_materialize_scalar_admission_uses_connected_native_identities(owner, monkeypatch, name):
    import builtins
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(name)
        raise AssertionError('mutable builtin scalar hook ran')
    cls = type('ForeignScalar', (() if name == 'bool' else (getattr(builtins, name),)),
               {'__float__': forbidden, '__int__': forbidden, '__bool__': forbidden})
    value = cls() if name == 'bool' else cls(1)
    # A genuine float keeps the old conversion's float64 branch selected, so
    # incorrectly admitting ForeignScalar would invoke its __float__ slot.
    base = np.array([[value, 1.0]], dtype='O')
    rows = DeviceRows(base, np.array([0]), owner.take)
    with monkeypatch.context() as patch:
        patch.setattr(builtins, name, cls)
        audit = audit_native_call(rows.materialize)
    _refused(audit)
    assert not calls and not owner._entries
