"""Small public scaler snapshot/copy-back controls for final integration."""
import mmap

import numpy as np
import pytest

import mojolearn as ml
from mojolearn._array import Array
from mojolearn._verify_native_runtime import audit_native_call

__mojolearn_user_boundary__ = True


@pytest.mark.parametrize('name', ['StandardScaler', 'MinMaxScaler'])
@pytest.mark.parametrize('direct', ['0', '1'])
@pytest.mark.parametrize('layout', ['native', 'strided', 'bigendian', 'readonly', 'array_view', 'mapped_array'])
def test_fit_transform_reuses_preparation_and_preserves_copyback(name, direct, layout, monkeypatch):
    monkeypatch.setenv('MOJOLEARN_SCALER_DIRECT', direct)
    x = np.arange(27, dtype=np.float32).reshape(9, 3) / 8
    # A NaN forces the combined entry's normal fallback after its initial
    # snapshot; direct=0 covers the route with no combined attempt.
    x[1, 1] = np.nan
    cls = getattr(ml, name)
    expected = cls().fit(x).transform(x)
    backing = None
    if layout == 'strided':
        backing = np.zeros((18, 6), dtype=np.float32)
        backing[::2, ::2] = x
        source = backing[::2, ::2]
    elif layout == 'bigendian':
        source = x.astype('>f4')
    elif layout == 'readonly':
        source = x.copy()
        source.flags.writeable = False
    elif layout == 'array_view':
        backing = Array.from_list(x.tolist(), '<f4')
        source = backing[:]
    elif layout == 'mapped_array':
        backing = mmap.mmap(-1, x.nbytes)
        storage = memoryview(backing).cast('f')
        storage[:] = memoryview(x).cast('B').cast('f')
        source = Array.from_buffer(storage.cast('B').cast('f', x.shape))
        # The mapped arena's finished Array keeps this exact memoryview.
        source._mv = storage
    else:
        source = x.copy()
    before = np.asarray(source).copy()
    model = cls(copy=False)
    actual = audit_native_call(model.fit_transform, source).assert_native()
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    assert np.asarray(actual).shape == (9, 3)
    assert np.asarray(actual).dtype == np.float32
    if layout in ('native', 'array_view', 'mapped_array'):
        np.testing.assert_array_equal(np.asarray(source), np.asarray(expected))
    else:
        np.testing.assert_array_equal(np.asarray(source), before)
    if backing is not None and layout == 'array_view':
        np.testing.assert_array_equal(np.asarray(backing), np.asarray(expected))


@pytest.mark.parametrize('name', ['StandardScaler', 'MinMaxScaler'])
def test_readonly_array_view_is_not_written_by_copy_false(name):
    x = np.arange(18, dtype=np.float32).reshape(6, 3)
    original = Array.from_list(x.tolist(), '<f4')
    view = original[:]
    # The Array's readonly slot is part of its supported copy-back contract,
    # independently of whether its underlying exported storage is writable.
    view._readonly = True
    result = getattr(ml, name)(copy=False).fit_transform(view)
    np.testing.assert_array_equal(np.asarray(original), x)
    assert np.asarray(result).dtype == np.float32
    assert not np.array_equal(np.asarray(result), x)


@pytest.mark.parametrize('name', ['StandardScaler', 'MinMaxScaler'])
def test_array_storage_hooks_are_refused_before_metadata_access(name):
    calls = []

    class Storage:
        @property
        def readonly(self):
            calls.append('readonly')
            return False

        def cast(self, *args):
            calls.append('cast')
            raise AssertionError('storage conversion hook ran')

    source = Array.from_list([[1., 2.], [3., 4.]], '<f4')
    original = source._mv
    source._mv = Storage()
    try:
        with pytest.raises(TypeError, match='memoryview'):
            getattr(ml, name)().fit_transform(source)
    finally:
        source._mv = original
    assert calls == []
