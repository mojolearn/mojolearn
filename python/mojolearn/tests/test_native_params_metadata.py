# SPDX-License-Identifier: Apache-2.0
"""Installed native parameter reads; the generic attribute path remains separate."""
import array
import inspect

import pytest

from mojolearn import (BaggingRegressor, GaussianProcessRegressor, KMeans,
                       RandomForestRegressor, Ridge, RidgeCV, StandardScaler)
from mojolearn import _mode
from mojolearn.gaussian_process import ConstantKernel, RBF
from mojolearn.model_selection import KFold, RepeatedKFold
from mojolearn._verify_native_runtime import audit_native_call


def clean(operation, *args, error=None):
    audit = audit_native_call(operation, *args)
    report = audit.report()
    assert not any(report[key] for key in ('python_frames', 'forbidden_native_calls',
                                          'unresolved_native_calls', 'wrapper_work',
                                          'unobserved_existing_threads')), report
    if error is not None:
        assert isinstance(audit.exception, error), report
        return audit.exception
    return audit.assert_native()


def test_shallow_constructor_names_and_values_keep_their_identity():
    model = KMeans(n_clusters=3, numeric_mode='identical')
    getter = model.get_params
    expected = {name: getattr(model, name) for name in model._get_param_names()}
    actual = clean(getter, False)
    assert list(actual) == sorted(expected)
    assert all(actual[name] is value for name, value in expected.items())


def test_nested_builtin_parameters_expand_and_keep_estimator_identity():
    child = Ridge(alpha=0.25, numeric_mode='identical')
    model = BaggingRegressor(estimator=child, numeric_mode='identical')
    getter = model.get_params
    deep = clean(getter)
    assert deep['estimator'] is child
    assert deep['estimator__alpha'] is child.alpha
    shallow = clean(getter, False)
    assert shallow['estimator'] is child
    assert not any(name.startswith('estimator__') for name in shallow)


def test_multiple_nested_levels_keep_each_object():
    leaf = Ridge(alpha=0.75, numeric_mode='identical')
    child = BaggingRegressor(estimator=leaf, numeric_mode='identical')
    model = BaggingRegressor(estimator=child, numeric_mode='identical')
    result = clean(model.get_params)
    assert result['estimator'] is child
    assert result['estimator__estimator'] is leaf
    assert result['estimator__estimator__alpha'] is leaf.alpha


def test_gp_kernels_remain_values_without_an_invented_parameter_protocol():
    kernel = ConstantKernel(2.0) * RBF(0.5)
    model = GaussianProcessRegressor(kernel=kernel, numeric_mode='identical')
    result = clean(model.get_params)
    assert result['kernel'] is kernel
    assert not any(name.startswith('kernel__') for name in result)


@pytest.mark.parametrize('splitter', [KFold(n_splits=3), RepeatedKFold(n_splits=3, n_repeats=2)])
def test_native_nested_splitter_schema_matches_its_public_parameters(splitter):
    model = RidgeCV(cv=splitter)
    expected = splitter.get_params()
    result = clean(model.get_params)
    assert result['cv'] is splitter
    assert {name[4:]: value for name, value in result.items() if name.startswith('cv__')} == expected


def test_nested_forest_and_scaler_keep_their_existing_shallow_contracts():
    model = BaggingRegressor(estimator=RandomForestRegressor(n_estimators=2, numeric_mode='identical'), numeric_mode='identical')
    getter = model.get_params
    result = clean(getter)
    assert result['estimator__n_estimators'] == 2
    child = StandardScaler(numeric_mode='identical')
    # Parameter inspection reports stored objects without fitting or validating
    # whether that object is an appropriate learner for a later Bagging.fit.
    model.__dict__['estimator'] = child
    result = clean(getter)
    assert result['estimator'] is child
    assert result['estimator__with_mean'] is child.with_mean


def test_classes_and_native_buffers_are_borrowed_values():
    model = BaggingRegressor(numeric_mode='identical')
    getter = model.get_params
    buffer = array.array('f', [1.0, 2.0])
    for value in (Ridge, buffer):
        model.__dict__['estimator'] = value
        result = clean(getter)
        assert result['estimator'] is value
        assert not any(name.startswith('estimator__') for name in result)


def test_numpy_buffer_identity_is_preserved_without_conversion():
    np = pytest.importorskip('numpy')
    value = np.arange(8, dtype='f4')[::2]
    model = BaggingRegressor(numeric_mode='identical')
    model.__dict__['estimator'] = value
    result = clean(model.get_params)
    assert result['estimator'] is value


def test_native_torch_buffer_identity_is_preserved_without_conversion():
    torch = pytest.importorskip('torch')
    value = torch.arange(8, dtype=torch.float32)[::2]
    model = BaggingRegressor(numeric_mode='identical')
    model.__dict__['estimator'] = value
    assert clean(model.get_params)['estimator'] is value


def test_dataframe_constructor_values_are_borrowed_without_column_lookup():
    pd = pytest.importorskip('pandas')
    frame = pd.DataFrame({'a': [0.0, 1.0], 'get_params': [2.0, 3.0]})
    model = KMeans(n_clusters=2, init='array', init_centroids=frame, numeric_mode='identical')
    result = clean(model.get_params)
    assert result['init_centroids'] is frame
    alpha = pd.Series([0.1, 0.2], index=['get_params', 'other'])
    gp = GaussianProcessRegressor(alpha=alpha, numeric_mode='identical')
    assert clean(gp.get_params)['alpha'] is alpha


def test_leaf_metaclass_hooks_are_not_needed_to_read_native_storage():
    calls = []
    class Meta(type):
        def __getattribute__(cls, name):
            calls.append(name)
            return super().__getattribute__(name)
    class Value(metaclass=Meta):
        pass
    value = Value()
    model = BaggingRegressor(numeric_mode='identical')
    model.__dict__['estimator'] = value
    getter = model.get_params
    calls.clear()
    assert clean(getter)['estimator'] is value
    assert not calls


def test_nested_python_getter_refuses_before_callback_but_shallow_read_is_safe():
    calls = []
    class Callback:
        def get_params(self):
            calls.append('called')
            return {}
    value = Callback()
    model = BaggingRegressor(numeric_mode='identical')
    model.__dict__['estimator'] = value
    getter = model.get_params
    assert clean(getter, False)['estimator'] is value
    clean(getter, True, error=TypeError)
    assert not calls


def test_nested_property_and_attribute_hooks_are_refused_before_invocation():
    calls = []
    class Property:
        @property
        def get_params(self):
            calls.append('property')
            return lambda: {}
    class Missing:
        def __getattr__(self, name):
            calls.append('missing')
            return None
    model = BaggingRegressor(numeric_mode='identical')
    getter = model.get_params
    for value in (Property(), Missing()):
        model.__dict__['estimator'] = value
        clean(getter, error=TypeError)
    assert not calls


def test_parameter_descriptor_change_is_refused_without_running_property(monkeypatch):
    model = BaggingRegressor(numeric_mode='identical')
    getter = model.get_params
    calls = []
    def invalid(self):
        calls.append('called')
        return None
    monkeypatch.setattr(type(model), 'estimator', property(invalid), raising=False)
    clean(getter, error=TypeError)
    assert not calls


def test_deep_boolean_callback_refuses_before_invocation():
    calls = []
    class Deep:
        def __bool__(self):
            calls.append('called')
            return True
    clean(KMeans(numeric_mode='identical').get_params, Deep(), error=TypeError)
    assert not calls


def test_cycles_raise_without_unbounded_native_recursion():
    model = BaggingRegressor(numeric_mode='identical')
    model.__dict__['estimator'] = model
    clean(model.get_params, error=RecursionError)


def test_missing_constructor_field_preserves_parameter_error():
    model = KMeans(numeric_mode='identical')
    getter = model.get_params
    del model.__dict__['n_clusters']
    clean(getter, False, error=TypeError)


def test_all_public_generic_getters_have_connected_schemas():
    # Dynamic installers are visible only in the installed runtime; the source
    # inventory cannot prove that every constructor signature connected.
    import mojolearn
    pytest.importorskip('numpy')
    # Resolve the documented optional sequence class imports as an application
    # does; their setup must connect new classes without rebuilding old plans.
    from mojolearn import _expansion_sequence
    for name in _expansion_sequence.__all__:
        getattr(_expansion_sequence, name)
    context = _mode._PARAMS_METADATA_CONTEXT
    for name in mojolearn.__all__:
        cls = getattr(mojolearn, name)
        if type(cls) is type:
            declared = next((vars(base)['get_params'] for base in cls.__mro__ if 'get_params' in vars(base)), None)
            if declared is not None:
                assert cls not in context['pending'], (name, context['pending'][cls])
    seen = set()
    for module in tuple(__import__('sys').modules.values()):
        if getattr(module, '__name__', '').startswith('mojolearn'):
            for cls in vars(module).values():
                if type(cls) is type and cls not in seen:
                    seen.add(cls)
                    declared = next((vars(base)['get_params'] for base in cls.__mro__ if 'get_params' in vars(base)), None)
                    if declared is _mode.ParamsMixin.get_params:
                        assert cls in context['plans'], (cls, context['pending'].get(cls))
