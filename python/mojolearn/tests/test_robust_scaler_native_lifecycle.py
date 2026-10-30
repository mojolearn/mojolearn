# SPDX-License-Identifier: Apache-2.0
"""Public RobustScaler lifecycle; inherited sklearn tags are separate scope."""
import os
import subprocess
import sys
import warnings

import numpy as np
import pytest

import mojolearn as ml
from mojolearn._array import Array
from mojolearn._arena_io import DeviceRows
from mojolearn._verify_native_runtime import audit_native_call


def _data(rows=19):
    x = np.random.default_rng(741).normal(size=(rows, 5)).astype('f4')
    x[:, -1] = 3
    x[::3, 0] = np.nan
    x[:, 1] = np.nan
    return x


def _refused(audit, category=TypeError):
    assert isinstance(audit.exception, category), audit.report()
    report = audit.report()
    for name in ('python_frames', 'forbidden_native_calls', 'wrapper_work', 'unobserved_existing_threads'):
        assert not report[name], report


def test_constructor_parameter_and_fitted_state_lifecycle_is_native():
    model = audit_native_call(ml.RobustScaler, copy=False).assert_native()
    assert audit_native_call(model.get_params).assert_native() == dict(
        copy=False, with_centering=True, with_scaling=True,
        quantile_range=(25., 75.), unit_variance=False)
    assert not audit_native_call(model.__sklearn_is_fitted__).assert_native()
    audit_native_call(model.fit, _data()).assert_native()
    before = model.scale_.tobytes()
    assert audit_native_call(model.set_params, with_centering=False).assert_native() is model
    assert audit_native_call(model.__sklearn_is_fitted__).assert_native()
    assert model.scale_.tobytes() == before  # set_params preserves existing fitted state.
    _refused(audit_native_call(model.set_params, unknown=True), ValueError)


@pytest.mark.parametrize('blocked', ['0', '1'])
@pytest.mark.parametrize('center,scale', [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize('rows', [19, 2049])
def test_public_fit_and_transform_keep_quantiles_nan_and_constant_columns(blocked, center, scale, rows, monkeypatch):
    monkeypatch.setenv('MOJOLEARN_XPREP_BLOCKED', blocked)
    x = _data(rows)
    before = x.tobytes()
    model = ml.RobustScaler(with_centering=center, with_scaling=scale, copy=False)
    actual = audit_native_call(model.fit_transform, x).assert_native()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        quantiles = np.nanpercentile(x, [25., 50., 75.], axis=0)
    # The existing x_prep quantile kernel returns zero when a feature has no
    # observed samples; its zero scale then becomes one. Preserve that contract.
    quantiles[:, np.isnan(x).all(axis=0)] = 0
    expected_center = quantiles[1].astype('f4') if center else None
    expected_scale = (quantiles[2] - quantiles[0]).astype('f4') if scale else None
    if scale:
        expected_scale[expected_scale < 10 * np.finfo('f4').eps] = 1
        np.testing.assert_allclose(model.scale_, expected_scale, rtol=2e-6, atol=2e-6)
    else:
        assert model.scale_ is None
    if center:
        np.testing.assert_allclose(model.center_, expected_center, rtol=2e-6, atol=2e-6)
    else:
        assert model.center_ is None
    expected = x.copy()
    if center:
        expected -= expected_center
    if scale:
        expected /= expected_scale
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    assert np.array_equal(np.isnan(actual), np.isnan(x))
    assert actual is not x and x.tobytes() == before
    assert actual.tobytes() == audit_native_call(model.transform, x).assert_native().tobytes()


@pytest.mark.parametrize('layout', ['f64', 'integer', 'fortran', 'strided', 'readonly', 'memoryview', 'array', 'list', 'bigendian'])
def test_ordinary_input_preparation_and_consumed_outputs_are_native(layout):
    x = np.nan_to_num(_data())
    if layout == 'integer':
        x = np.round(x).astype('i8')
    sources = dict(f64=lambda: x.astype('f8'), integer=lambda: x,
                   fortran=lambda: np.asfortranarray(x),
                   strided=lambda: np.repeat(x, 2, axis=0)[::2], readonly=lambda: x.copy(),
                   memoryview=lambda: memoryview(x), array=lambda: Array.from_list(x.tolist(), '<f4'),
                   list=lambda: x.tolist(), bigendian=lambda: x.astype('>f4'))
    source = sources[layout]()
    if layout == 'readonly':
        source.flags.writeable = False
    actual = audit_native_call(ml.RobustScaler().fit_transform, source).assert_native()
    expected = ml.RobustScaler().fit_transform(x.astype('f4'))
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize('quantiles', [(10., 90.), [10, 90], np.asarray([10., 90.]), Array.from_list([10., 90.], '<f8')])
def test_native_quantile_metadata_and_unit_variance_are_preserved(quantiles):
    model = ml.RobustScaler(quantile_range=quantiles, unit_variance=True)
    actual = audit_native_call(model.fit_transform, _data()).assert_native()
    expected = ml.RobustScaler(quantile_range=(10., 90.), unit_variance=True).fit_transform(_data())
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize('kind', ['object', 'str', 'bytes', 'f2', 'longdouble', 'complex', 'strided', 'bigendian', 'halfview', 'halfarray'])
def test_native_quantile_buffers_preserve_supported_scalar_kinds(kind):
    source = dict(object=lambda: np.array([10., 90.], dtype=object),
                  str=lambda: np.array(['10', '90']), bytes=lambda: np.array([b'10', b'90']),
                  f2=lambda: np.array([10., 90.], dtype='f2'),
                  longdouble=lambda: np.array([10., 90.], dtype='g'),
                  complex=lambda: np.array([10.+1j, 90.+2j]),
                  strided=lambda: np.array([90., 10.])[::-1],
                  bigendian=lambda: np.array([10., 90.], dtype='>f8'),
                  halfview=lambda: memoryview(np.array([10., 90.], dtype='f2')),
                  halfarray=lambda: Array.from_list([10., 90.], '<f2'))[kind]()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        actual = audit_native_call(ml.RobustScaler(quantile_range=source).fit_transform, _data()).assert_native()
    expected = ml.RobustScaler(quantile_range=(10., 90.)).fit_transform(_data())
    assert actual.tobytes() == expected.tobytes()
    assert len(caught) == (2 if kind == 'complex' else 0)
    if kind == 'complex':
        assert all(record.category is np.exceptions.ComplexWarning for record in caught)


@pytest.mark.parametrize('storage', ['object', 'structured', 'object_view', 'structured_view', 'ndarray_subclass', 'foreign_buffer'])
def test_quantile_storage_refuses_before_element_or_exporter_hooks(storage):
    calls = []
    class Hook:
        def __float__(self):
            calls.append('float')
            return 10.
    class Foreign:
        def __buffer__(self, flags):
            calls.append('buffer')
            return memoryview(np.array([10., 90.]))
        def __release_buffer__(self, view):
            calls.append('release')
    class Subclass(np.ndarray):
        def __getitem__(self, key):
            calls.append('getitem')
            return super().__getitem__(key)
    if storage == 'foreign_buffer':
        if sys.version_info < (3, 12):
            pytest.skip('Python-defined buffer exporters require Python 3.12')
        exporter = Foreign()
        source = memoryview(exporter)
    elif storage == 'ndarray_subclass':
        source = np.array([10., 90.]).view(Subclass)
    else:
        source = np.empty(2, dtype=[('q', object)] if 'structured' in storage else object)
        source[:] = [(Hook(),), (90.,)] if 'structured' in storage else [Hook(), 90.]
        if storage.endswith('_view'):
            source = memoryview(source)
    calls.clear()  # Construction by the application precedes product execution.
    _refused(audit_native_call(ml.RobustScaler(quantile_range=source).fit, _data()))
    assert not calls
    if storage == 'foreign_buffer':
        # Release outside the audited product call while the fixture class and
        # exporter are still live; do not defer its callback to cyclic GC.
        source.release()
        assert calls == ['release']


@pytest.mark.parametrize('metadata', ['name', 'key'])
def test_complex_quantile_warning_metadata_cannot_invoke_python_hooks(metadata, monkeypatch):
    calls = []
    class Text(str):
        __hash__ = str.__hash__
        def __str__(self):
            calls.append('str')
            return str.__str__(self)
        def __eq__(self, other):
            calls.append('eq')
            return str.__eq__(self, other)
    fields = {Text('__module__') if metadata == 'key' else '__module__': 'numpy.exceptions'}
    category = type('ComplexWarning', (RuntimeWarning,), fields)
    if metadata == 'name':
        category.__name__ = Text('ComplexWarning')
    monkeypatch.setattr(np.exceptions, 'ComplexWarning', category)
    source = np.array([10.+1j, 90.+2j])
    calls.clear()
    _refused(audit_native_call(ml.RobustScaler(quantile_range=source).fit, _data()))
    assert not calls


@pytest.mark.parametrize('field', ['dtype', 'shape', '_mv'])
def test_quantile_array_metadata_refuses_python_value_protocols(field):
    calls = []
    class Hook:
        def __str__(self):
            calls.append('str')
            return '<f8'
        def __index__(self):
            calls.append('index')
            return 2
        def __float__(self):
            calls.append('float')
            return 10.
    source = Array.from_list([10.,90.], '<f8')
    setattr(source, field, (Hook(),) if field == 'shape' else Hook())
    _refused(audit_native_call(ml.RobustScaler(quantile_range=source).fit, _data()))
    assert not calls


@pytest.mark.parametrize('options', [dict(quantile_range=(-1., 75.)), dict(quantile_range=(80., 20.)), dict(unit_variance=True, quantile_range=(0., 100.))])
def test_invalid_quantiles_are_refused_natively(options):
    _refused(audit_native_call(ml.RobustScaler(**options).fit, _data()), ValueError)


def test_target_width_and_unfitted_checks_are_native():
    model = ml.RobustScaler()
    from mojolearn.exceptions import NotFittedError
    _refused(audit_native_call(model.transform, _data()), NotFittedError)
    _refused(audit_native_call(model.fit, _data(), [1, 2]), ValueError)
    audit_native_call(model.fit, _data(), np.arange(19)).assert_native()
    _refused(audit_native_call(model.transform, np.ones((3, 2), dtype='f4')), ValueError)


@pytest.mark.parametrize('position', ['input', 'quantiles', 'truth'])
def test_python_protocols_refuse_before_their_hooks(position):
    calls = []
    class Hook:
        def __array__(self, *args, **kwargs):
            calls.append('array')
            return _data()
        def __float__(self):
            calls.append('float')
            return 25.
        def __bool__(self):
            calls.append('bool')
            return True
    model = ml.RobustScaler(quantile_range=(Hook(), 75.)) if position == 'quantiles' else ml.RobustScaler(with_centering=Hook()) if position == 'truth' else ml.RobustScaler()
    _refused(audit_native_call(model.fit, Hook() if position == 'input' else _data()))
    assert not calls


def test_native_row_gather_does_not_call_application_take_hook():
    x = _data()
    calls = []
    def take(*args):
        calls.append(True)
        raise AssertionError('Python row gather ran')
    indices = [0, 3, 11, 2]
    rows = DeviceRows(Array.from_list(x.tolist(), '<f4'), Array.from_list(indices, '<i8'), take)
    actual = audit_native_call(ml.RobustScaler().fit_transform, rows).assert_native()
    expected = ml.RobustScaler().fit_transform(x[indices])
    assert actual.tobytes() == expected.tobytes()
    assert not calls


def test_application_subclass_overrides_keep_guards_and_are_not_called_from_native():
    calls = []
    class Child(ml.RobustScaler):
        def fit(self, X, y=None):
            calls.append('fit')
            return self
    assert Child.fit._mojolearn_tracks_fit
    child = Child()
    _refused(audit_native_call(child.fit_transform, _data()))
    assert not calls
    with pytest.raises(ValueError):
        child.fit(_data(), [1, 2])
    assert not calls


def test_numpy_free_public_lifecycle_does_not_import_numpy():
    script = '''
import sys
# Model an environment without the optional dependency before package import.
import importlib.abc
class BlockNumPy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "numpy" or fullname.startswith("numpy."):
            raise ModuleNotFoundError("NumPy is absent in this fixture", name="numpy")
sys.meta_path.insert(0, BlockNumPy())
from array import array
import mojolearn as ml
from mojolearn._verify_native_runtime import audit_native_call
assert 'numpy' not in sys.modules
x=memoryview(array('f',[1,2,3,4,5,6])).cast('B').cast('f',(3,2))
m=audit_native_call(ml.RobustScaler).assert_native()
y=audit_native_call(m.fit_transform,x).assert_native()
assert y.shape==(3,2) and y.tolist()==[[-1.,-1.],[0.,0.],[1.,1.]]
assert 'numpy' not in sys.modules
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, env=os.environ)
    assert result.returncode == 0, result.stderr[-4000:]
