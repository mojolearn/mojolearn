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
