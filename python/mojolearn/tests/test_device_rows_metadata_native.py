# SPDX-License-Identifier: Apache-2.0
"""DeviceRows constructor metadata preserves aliases without Python work."""
from array import array
import numpy as np
import pytest
from mojolearn._array import Array
from mojolearn._arena_io import DeviceRows
from mojolearn._verify_native_runtime import audit_native_call


@pytest.mark.parametrize('source', ['array', 'numpy'])
@pytest.mark.parametrize('dtype', ['<f4', '<f8', '<i8'])
@pytest.mark.parametrize('shape', [(4,), (4,3), (4,2,3), (0,3)])
def test_constructor_and_length_preserve_native_metadata_and_aliases(source, dtype, shape):
    values = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    base = values if source == 'numpy' else Array._owned(
        array({'<f4':'f','<f8':'d','<i8':'q'}[dtype], values.ravel().tolist()), shape, dtype, 'C')
    indices = np.array([2,0], dtype='i8') if source == 'numpy' else Array.from_list([2,0], '<i8')
    calls = []
    def forbidden(*args):
        calls.append(True)
        raise AssertionError('opaque materializer was called by constructor')
    rows = audit_native_call(DeviceRows, base, indices, forbidden).assert_native()
    assert rows.base is base and rows.indices is indices and rows._take is forbidden
    assert rows._host is None and not calls
    assert rows.shape == (2, *shape[1:])
    assert rows.dtype == base.dtype and rows.ndim == base.ndim and rows.itemsize == base.itemsize
    assert rows.size == np.prod(rows.shape)
    assert rows.row_words == (base.size // shape[0] if shape[0] else 0)
    assert audit_native_call(rows.__len__).assert_native() == 2


@pytest.mark.parametrize('field', ['shape', 'size', 'dtype', 'itemsize', 'ndim'])
def test_constructor_refuses_metadata_protocols_before_hooks(field):
    calls = []
    class Foreign:
        def __getitem__(self, key):
            calls.append('getitem')
            raise AssertionError('metadata protocol called')
        def __int__(self):
            calls.append('int')
            return 2
        def __str__(self):
            calls.append('str')
            return '<f4'
    base = Array.from_list([[1.,2.],[3.,4.]], '<f4')
    setattr(base, field, Foreign())
    indices = Array.from_list([0], '<i8')
    audit = audit_native_call(DeviceRows, base, indices, None)
    assert isinstance(audit.exception, TypeError), audit.report()
    assert not calls
    assert not audit.report()['python_frames'] and not audit.report()['forbidden_native_calls']


def test_constructor_refuses_replaced_array_descriptor_before_hook(monkeypatch):
    calls = []
    base = Array.from_list([[1.,2.],[3.,4.]], '<f4')
    indices = Array.from_list([0], '<i8')
    def forbidden(value):
        calls.append(True)
        raise AssertionError('replaced Array slot executed')
    with monkeypatch.context() as patch:
        patch.setattr(Array, 'shape', property(forbidden))
        audit = audit_native_call(DeviceRows, base, indices, None)
    assert isinstance(audit.exception, TypeError), audit.report()
    assert not calls and not audit.report()['python_frames']


def test_constructor_preserves_scalar_base_error():
    base = Array._owned(array('f', [1.]), (), '<f4', 'C')
    audit = audit_native_call(DeviceRows, base, Array.from_list([0], '<i8'), None)
    assert isinstance(audit.exception, IndexError), audit.report()
    assert not audit.report()['python_frames'] and not audit.report()['forbidden_native_calls']


def test_constructor_preserves_an_ordinary_subclass():
    class Rows(DeviceRows):
        pass
    base = Array.from_list([[1.,2.],[3.,4.]], '<f4')
    indices = Array.from_list([0], '<i8')
    rows = audit_native_call(Rows, base, indices, None).assert_native()
    assert type(rows) is Rows and rows.base is base and rows.indices is indices
    assert audit_native_call(rows.__len__).assert_native() == 1


def _fresh_rows(source='array', take=None):
    if source == 'numpy':
        return DeviceRows(np.arange(6, dtype='f4').reshape(2, 3), np.array([1], dtype='i8'), take)
    return DeviceRows(Array.from_list([[1.,2.,3.],[4.,5.,6.]], '<f4'), Array.from_list([1], '<i8'), take)


@pytest.mark.parametrize('source', ['array', 'numpy'])
def test_reinitialization_preserves_native_aliases_and_resets_snapshot(source):
    calls = []
    def materializer(*args):
        calls.append('take')
    rows = _fresh_rows(source, materializer)
    rows._host = Array.from_list([[4.,5.,6.]], '<f4')
    marker = object()
    rows.application_state = marker
    if source == 'numpy':
        base, indices = np.arange(8, dtype='f8').reshape(4, 2), np.array([3, 0], dtype='i8')
    else:
        base, indices = Array.from_list([[1.,2.],[3.,4.],[5.,6.],[7.,8.]], '<f8'), Array.from_list([3,0], '<i8')
    assert audit_native_call(rows.__init__, base, indices, materializer).assert_native() is None
    assert rows.base is base and rows.indices is indices and rows._take is materializer
    assert rows._host is None and rows.application_state is marker and not calls
    assert rows.shape == (2, 2) and rows.size == 4 and rows.row_words == 2
    assert rows.dtype == base.dtype and rows.itemsize == 8 and rows.ndim == 2
    assert audit_native_call(rows.__len__).assert_native() == 2


def test_reinitialization_preserves_a_renewed_bound_method_owner():
    calls = []
    class Materializer:
        def take(self, *args):
            calls.append('take')
    owner = Materializer()
    rows = _fresh_rows(take=owner.take)
    base, indices = Array.from_list([[1.,2.],[3.,4.]], '<f4'), Array.from_list([0], '<i8')
    assert audit_native_call(rows.__init__, base, indices, owner.take).assert_native() is None
    assert rows._take.__self__ is owner and rows._take.__func__ is Materializer.take and not calls


@pytest.mark.parametrize('field', ['base', 'indices', '_take', '_host', 'shape', 'dtype', 'itemsize', 'ndim', 'size', 'row_words'])
def test_reinitialization_refuses_last_owned_finalizers_before_any_replacement(field):
    calls = []
    class Finalizer:
        def __del__(self):
            calls.append('finalizer')
    rows = _fresh_rows()
    setattr(rows, field, Finalizer())  # the rows dictionary is the only owner
    previous_ids = {key: id(value) for key, value in vars(rows).items()}
    base, indices = Array.from_list([[9.,8.]], '<f4'), Array.from_list([0], '<i8')
    audit = audit_native_call(rows.__init__, base, indices, None)
    assert isinstance(audit.exception, TypeError), audit.report()
    assert 'reinitialization' in str(audit.exception)
    assert not calls and not audit.report()['python_frames'] and not audit.report()['forbidden_native_calls']
    assert {key: id(value) for key, value in vars(rows).items()} == previous_ids
    setattr(rows, field, None)  # application cleanup, outside the audited product operation
    assert calls == ['finalizer']


@pytest.mark.parametrize('slot', ['_pin', '_store', 'strides', 'order'])
def test_reinitialization_admits_all_previous_array_slot_owners(slot):
    calls = []
    class Finalizer:
        def __del__(self):
            calls.append('finalizer')
    rows = _fresh_rows()
    setattr(rows.base, slot, Finalizer())
    previous_base_id = id(rows.base)
    base, indices = Array.from_list([[9.,8.]], '<f4'), Array.from_list([0], '<i8')
    audit = audit_native_call(rows.__init__, base, indices, None)
    assert isinstance(audit.exception, TypeError), audit.report()
    assert not calls and id(rows.base) == previous_base_id
    assert not audit.report()['python_frames'] and not audit.report()['forbidden_native_calls']
    setattr(rows.base, slot, None)
    assert calls == ['finalizer']


@pytest.mark.parametrize('source', ['array', 'numpy'])
def test_reinitialization_refuses_owned_buffer_weakref_callbacks(source):
    import weakref
    calls = []
    rows = _fresh_rows(source)
    owner = rows.base._store if source == 'array' else rows.base
    reference = weakref.ref(owner, lambda unused: calls.append('weakref'))
    del owner
    previous_base_id = id(rows.base)
    base, indices = Array.from_list([[9.,8.]], '<f4'), Array.from_list([0], '<i8')
    audit = audit_native_call(rows.__init__, base, indices, None)
    assert isinstance(audit.exception, TypeError), audit.report()
    assert 'weak-reference' in str(audit.exception)
    assert not calls and id(rows.base) == previous_base_id and reference() is not None
    assert not audit.report()['python_frames'] and not audit.report()['forbidden_native_calls']
    rows.base = None
    assert reference() is None and calls == ['weakref']


def test_reinitialization_refuses_nested_dtype_metadata_without_releasing_it():
    calls = []
    class Finalizer:
        def __del__(self):
            calls.append('finalizer')
    rows = DeviceRows(np.zeros((2, 3), dtype=np.dtype('f4', metadata={'owner': Finalizer()})), np.array([0], dtype='i8'), None)
    previous_base_id, previous_dtype_id = id(rows.base), id(rows.dtype)
    base, indices = np.zeros((2, 2), dtype='f8'), np.array([1], dtype='i8')
    audit = audit_native_call(rows.__init__, base, indices, None)
    assert isinstance(audit.exception, TypeError), audit.report()
    assert not calls and id(rows.base) == previous_base_id and id(rows.dtype) == previous_dtype_id
    assert not audit.report()['python_frames'] and not audit.report()['forbidden_native_calls']
    rows.base = None
    rows.dtype = None
    assert calls == ['finalizer']


@pytest.mark.parametrize('layout', ['structured', 'subarray', 'metadata', 'object', 'structured_object'])
def test_reinitialization_preserves_native_dtype_layouts_and_owned_metadata(layout):
    dtype = {
        'structured': np.dtype([('left', '<i4'), ('right', '<f8')]),
        'subarray': np.dtype([('vector', '<f8', (2,))]),
        'metadata': np.dtype('<f4', metadata={'shape': (2, 3), 'units': ['native', np.int32(1)]}),
        'object': np.dtype('O'),
        'structured_object': np.dtype([('name', 'O'), ('weight', '<f8')]),
    }[layout]
    base = np.zeros((2, 3), dtype=dtype)
    if layout == 'object':
        base[0, 0] = {'values': [1, 2.5, None, np.longdouble(3), np.complex64(2j), np.str_('native')]}
    elif layout == 'structured_object':
        base['name'][0, 0] = ['native', 3]
    rows = DeviceRows(base, np.array([1], dtype='i8'), None)
    del base, dtype  # rows owns the old buffer and dtype references
    replacement = np.arange(4, dtype='f8').reshape(2, 2)
    indices = np.array([0], dtype='i8')
    assert audit_native_call(rows.__init__, replacement, indices, None).assert_native() is None
    assert rows.base is replacement and rows.indices is indices
    assert rows.dtype == np.dtype('f8') and rows.shape == (1, 2)


@pytest.mark.parametrize('structured', [False, True])
def test_reinitialization_refuses_owned_object_buffer_finalizers(structured):
    calls = []
    class Finalizer:
        def __del__(self):
            calls.append('finalizer')
    dtype = np.dtype([('owner', 'O'), ('value', 'f8')]) if structured else np.dtype('O')
    rows = DeviceRows(np.zeros((2, 3), dtype=dtype), np.array([1], dtype='i8'), None)
    if structured:
        rows.base['owner'][0, 0] = Finalizer()
    else:
        rows.base[0, 0] = Finalizer()
    previous_base_id = id(rows.base)
    replacement = np.zeros((2, 2), dtype='f8')
    audit = audit_native_call(rows.__init__, replacement, np.array([0], dtype='i8'), None)
    assert isinstance(audit.exception, TypeError), audit.report()
    assert not calls and id(rows.base) == previous_base_id
    assert not audit.report()['python_frames'] and not audit.report()['forbidden_native_calls']
    rows.base = None
    assert calls == ['finalizer']
