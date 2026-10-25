# SPDX-License-Identifier: Apache-2.0
"""Escaping views through the shared input converter, using public operations."""
import gc
import weakref

import pytest

from mojolearn import KernelPCA, Ridge
from mojolearn._array import Array
from mojolearn._verify_native_runtime import audit_native_call


def public_read(model, name):
    return getattr(model, name)


def read_native(model, name):
    # The actual Ridge attribute slot calls classical_getattribute and then
    # sequential_input. This must exercise that owner, not the new shared
    # PublicOutputsMixin installer or a private converter in isolation.
    return audit_native_call(public_read, model, name).assert_native()


@pytest.mark.parametrize('order,dtype,readonly', [('C', 'f4', False), ('F', 'f4', False),
                                                 ('C', 'f2', False), ('F', 'f8', True)])
def test_ridge_attribute_view_retains_external_exporter(order, dtype, readonly):
    np = pytest.importorskip('numpy')
    storage = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype, order=order)
    if readonly:
        storage.flags.writeable = False
    reference = weakref.ref(storage)
    array = Array.from_buffer(storage)
    model = object.__new__(Ridge)
    object.__getattribute__(model, '__dict__')['coef_'] = array
    output = read_native(model, 'coef_')
    assert type(output) is np.ndarray
    assert np.shares_memory(output, storage)
    assert output.strides == storage.strides
    assert output.flags.writeable is (not readonly)
    del array, model, storage
    gc.collect()
    assert reference() is not None
    np.testing.assert_array_equal(output, [[1, 2, 3], [4, 5, 6]])
    del output
    gc.collect()
    assert reference() is None


def test_ridge_row_view_keeps_offset_and_owner():
    np = pytest.importorskip('numpy')
    storage = np.array([[1, 2], [3, 4]], dtype='f4')
    reference = weakref.ref(storage)
    array = Array.from_buffer(storage)
    row = array._row_view(1, (2,), 2)
    model = object.__new__(Ridge)
    object.__getattribute__(model, '__dict__')['coef_'] = row
    output = read_native(model, 'coef_')
    assert np.shares_memory(output, storage)
    del storage, array, row, model
    gc.collect()
    assert reference() is not None
    np.testing.assert_array_equal(output, [3, 4])


def test_mapped_preprocessing_storage_without_store_keeps_owning_slice():
    import mmap
    np = pytest.importorskip('numpy')
    mapping = mmap.mmap(-1, 32)
    reference = weakref.ref(mapping)
    view = memoryview(mapping).cast('f')
    view[2], view[3] = 5, 7
    part = view[2:4]
    # Testing preparation reproduces xprep_take's documented mapped output
    # layout at a small size. It does not claim a full workload measurement.
    array = object.__new__(Array)
    array._store, array._base, array._pin = None, view, None
    array._mv = part
    array._addr = np.ndarray((2,), dtype='f4', buffer=part).__array_interface__['data'][0]
    array._readonly, array.shape, array.dtype, array.order = False, (2,), '<f4', 'C'
    model = object.__new__(Ridge)
    object.__getattribute__(model, '__dict__')['coef_'] = array
    output = read_native(model, 'coef_')
    del array, model, part, view, mapping
    gc.collect()
    assert reference() is not None
    np.testing.assert_array_equal(output, [5, 7])
    del output
    gc.collect()
    assert reference() is None


def test_kernel_pca_retained_training_view_survives_argument_lifetime():
    np = pytest.importorskip('numpy')
    storage = np.array([[0, 1], [1, 2], [2, 0], [3, 1]], dtype='f4')
    reference = weakref.ref(storage)
    array = Array.from_buffer(storage)
    model = KernelPCA(n_components=2, kernel='linear', numeric_mode='identical')
    model.fit(array)
    query = np.array([[0.5, 1.0], [2.5, 0.5]], dtype='f4')
    expected = np.asarray(model.transform(query)).copy()
    fitted = object.__getattribute__(model, '__dict__')['_fit_X']
    assert np.shares_memory(np.asarray(fitted), storage)
    del array, storage, fitted
    gc.collect()
    assert reference() is not None
    np.testing.assert_array_equal(np.asarray(model.transform(query)), expected)
