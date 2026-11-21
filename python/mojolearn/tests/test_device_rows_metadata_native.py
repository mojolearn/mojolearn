# SPDX-License-Identifier: Apache-2.0
"""DeviceRows constructor metadata preserves aliases without Python work.

The view's metadata and its base / indices pins are native state behind the
instance's one `_owner` slot; every public attribute is read-only.

Rewritten with the native state (2026-11-21, arena unit E): `_take` and
`_host` are gone (a `take` other than `model_selection._Resident.take` is
not kept, and `materialize` refuses it in words). Reinitialization admits
every owner the old view's release would drop before anything changes: an
old base, indices or host rows whose release could run a finalizer or a
weak-reference callback is refused in words (TypeError) and the view is
left as it was (restored natively, lane fx-leftovers item 7).
"""
from array import array
import numpy as np
import pytest
from mojolearn._array import Array
from mojolearn._arena_io import DeviceRows
from mojolearn._verify_native_runtime import audit_native_call


@pytest.mark.parametrize('dtype', ['q', 'Q'])
@pytest.mark.parametrize('storage', ['scalar', 'object_array'])
def test_reinitialization_preserves_native_longlong_owner_aliases(dtype, storage):
    scalar = np.dtype(dtype).type(3)
    old = scalar if storage == 'scalar' else np.array([scalar], dtype=object)
    base = Array.from_list([[1.]], '<f4')
    indices = Array.from_list([0], '<i8')
    rows = DeviceRows(base, indices, old)
    assert audit_native_call(rows.__init__, base, indices, None).assert_native() is None
    assert rows.base is base and rows.indices is indices


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
    assert rows.base is base and rows.indices is indices and not calls
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
def test_reinitialization_preserves_native_aliases_and_stays_read_only(source):
    calls = []
    def materializer(*args):
        calls.append('take')
    rows = _fresh_rows(source, materializer)
    for name in ('_host', '_take', 'application_state', 'base'):
        with pytest.raises(AttributeError, match='read-only view'):
            setattr(rows, name, object())
    if source == 'numpy':
        base, indices = np.arange(8, dtype='f8').reshape(4, 2), np.array([3, 0], dtype='i8')
    else:
        base, indices = Array.from_list([[1.,2.],[3.,4.],[5.,6.],[7.,8.]], '<f8'), Array.from_list([3,0], '<i8')
    assert audit_native_call(rows.__init__, base, indices, materializer).assert_native() is None
    assert rows.base is base and rows.indices is indices and not calls
    assert rows.shape == (2, 2) and rows.size == 4 and rows.row_words == 2
    assert rows.dtype == base.dtype and rows.itemsize == 8 and rows.ndim == 2
    assert audit_native_call(rows.__len__).assert_native() == 2
    # A Python gather callback is never invoked.
    with pytest.raises(TypeError, match='_Resident.take'):
        rows.materialize()
    assert not calls


def test_reinitialization_with_a_foreign_bound_method_never_calls_it():
    calls = []
    class Materializer:
        def take(self, *args):
            calls.append('take')
    owner = Materializer()
    rows = _fresh_rows(take=owner.take)
    base, indices = Array.from_list([[1.,2.],[3.,4.]], '<f4'), Array.from_list([0], '<i8')
    assert audit_native_call(rows.__init__, base, indices, owner.take).assert_native() is None
    with pytest.raises(TypeError, match='_Resident.take'):
        rows.materialize()
    assert not calls


class _Finalizer:
    def __init__(self, calls):
        self.calls = calls

    def __del__(self):
        self.calls.append('finalizer')


def _refused_reinitialization(rows, base=None, indices=None):
    # The audited call must refuse BEFORE the view changes and run no Python.
    previous_base = rows.base
    if base is None:
        base, indices = Array.from_list([[9.,8.]], '<f4'), Array.from_list([0], '<i8')
    audit = audit_native_call(rows.__init__, base, indices, None)
    assert isinstance(audit.exception, TypeError), audit.report()
    assert not audit.report()['python_frames'] and not audit.report()['forbidden_native_calls']
    assert rows.base is previous_base
    return audit.exception


@pytest.mark.parametrize('field', ['base', 'indices'])
def test_reinitialization_refuses_a_previous_view_pin_with_a_finalizer(field):
    calls = []
    held = _Finalizer(calls)
    previous = (np.array([[held, 1.]], dtype='O') if field == 'base' else np.array([0], dtype='i8'))
    rows = (DeviceRows(previous, np.array([0], dtype='i8'), None) if field == 'base' else
            DeviceRows(np.zeros((2, 3), dtype='f4'), previous, None))
    del held, previous
    if field == 'base':
        _refused_reinitialization(rows)
        assert calls == []
        rows.base[0, 0] = None  # application cleanup, outside the audited product operation
        assert calls == ['finalizer']
    else:
        # plain int64 indices own no finalizer: the view is replaced natively
        base, indices = Array.from_list([[9.,8.]], '<f4'), Array.from_list([0], '<i8')
        assert audit_native_call(rows.__init__, base, indices, None).assert_native() is None
        assert rows.base is base and calls == []


@pytest.mark.parametrize('slot', ['_pin', '_store', 'strides', 'order'])
def test_reinitialization_admits_all_previous_array_slot_owners(slot):
    calls = []
    rows = _fresh_rows()
    setattr(rows.base, slot, _Finalizer(calls))
    _refused_reinitialization(rows)
    assert calls == []
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
    error = _refused_reinitialization(rows)
    assert 'weak-reference' in str(error)
    assert not calls and reference() is not None


def test_reinitialization_refuses_nested_dtype_metadata_without_releasing_it():
    calls = []
    rows = DeviceRows(np.zeros((2, 3), dtype=np.dtype('f4', metadata={'owner': _Finalizer(calls)})),
                      np.array([0], dtype='i8'), None)
    _refused_reinitialization(rows, np.zeros((2, 2), dtype='f8'), np.array([1], dtype='i8'))
    assert not calls and rows.dtype == np.dtype('f4')


def test_reinitialization_keeps_a_base_the_caller_passes_again():
    # The same base passed again is not released: nothing to admit, no refusal.
    calls = []
    held = _Finalizer(calls)
    base = np.array([[held, 1.]], dtype='O')
    rows = DeviceRows(base, np.array([0], dtype='i8'), None)
    assert audit_native_call(rows.__init__, base, np.array([0], dtype='i8'), None).assert_native() is None
    assert rows.base is base and calls == []


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
    old = rows.base
    del base, dtype
    replacement = np.arange(4, dtype='f8').reshape(2, 2)
    indices = np.array([0], dtype='i8')
    assert audit_native_call(rows.__init__, replacement, indices, None).assert_native() is None
    assert rows.base is replacement and rows.indices is indices
    assert rows.dtype == np.dtype('f8') and rows.shape == (1, 2)
    del old


@pytest.mark.parametrize('structured', [False, True])
def test_reinitialization_refuses_owned_object_buffer_finalizers(structured):
    calls = []
    dtype = np.dtype([('owner', 'O'), ('value', 'f8')]) if structured else np.dtype('O')
    rows = DeviceRows(np.zeros((2, 3), dtype=dtype), np.array([1], dtype='i8'), None)
    if structured:
        rows.base['owner'][0, 0] = _Finalizer(calls)
    else:
        rows.base[0, 0] = _Finalizer(calls)
    _refused_reinitialization(rows, np.zeros((2, 2), dtype='f8'), np.array([0], dtype='i8'))
    assert not calls
