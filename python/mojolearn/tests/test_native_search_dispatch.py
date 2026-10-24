"""Public search construction, configuration, sampling and callback refusal."""
import numpy as np
import pytest

import mojolearn as ml
from mojolearn import model_selection as ms
from mojolearn._verify_native_runtime import audit_native_call


@pytest.mark.parametrize('cls, options', [
    (ms.GridSearchCV, {'param_grid': {'alpha': [.1, 1.]}}),
    (ms.RandomizedSearchCV, {'param_distributions': {'alpha': [.1, 1.]}, 'n_iter': 2}),
])
def test_search_constructor_and_parameter_metadata_are_native(cls, options):
    estimator = ml.Ridge()
    search = audit_native_call(cls, estimator, cv=3, **options).assert_native()
    parameters = audit_native_call(search.get_params).assert_native()
    assert parameters['estimator'] is estimator and parameters['cv'] == 3
    assert audit_native_call(search.set_params, cv=4).assert_native() is search
    assert search.cv == 4


def test_parameter_grid_native_iteration_retains_mutable_value_lists():
    values = [.1, 1.]
    grid = audit_native_call(ms.ParameterGrid, {'alpha': values}).assert_native()
    values.append(10.)
    iterator = audit_native_call(grid.__iter__).assert_native()
    assert list(iterator) == [{'alpha': v} for v in values]
    assert audit_native_call(grid.__len__).assert_native() == 3


def test_parameter_sampler_native_iteration_and_seed():
    options = {'alpha': np.array([.1, 1., 10.], dtype='f8')}
    first = audit_native_call(ms.ParameterSampler, options, 2, random_state=17).assert_native()
    second = ms.ParameterSampler(options, 2, random_state=17)
    got = list(audit_native_call(first.__iter__).assert_native())
    assert got == list(second) and len(got) == 2
    assert len({float(row['alpha']) for row in got}) == 2


def test_search_refit_and_grid_protocols_refuse_before_python_hooks():
    calls = []

    class Callback:
        def __call__(self, *args, **kwargs):
            calls.append('call')
        def __getattribute__(self, name):
            calls.append(name)
            raise AssertionError('Python protocol was invoked')

    grid = audit_native_call(ms.ParameterGrid, {'alpha': Callback()})
    assert isinstance(grid.exception, TypeError), grid.report()
    assert calls == [] and not grid.report()['python_frames'], grid.report()
    search = ms.RandomizedSearchCV(ml.Ridge(), {'alpha': [1.]}, n_iter=1, cv=2, refit=Callback())
    x = np.arange(18, dtype='f4').reshape(6, 3)
    y = np.arange(6, dtype='f4')
    fit = audit_native_call(search.fit, x, y)
    assert isinstance(fit.exception, TypeError), fit.report()
    assert calls == [] and not fit.report()['python_frames'], fit.report()


def test_parameter_sampler_distribution_property_is_not_called():
    calls = []

    class Distribution:
        @property
        def rvs(self):
            calls.append('rvs')
            raise AssertionError('distribution property executed')

    audit = audit_native_call(ms.ParameterSampler, {'alpha': Distribution()}, 2)
    assert isinstance(audit.exception, NotImplementedError), audit.report()
    assert calls == [] and not audit.report()['python_frames'], audit.report()


def test_learning_curve_native_fit_failure_uses_error_score():
    x = np.arange(36, dtype='f4').reshape(12, 3)
    y = np.arange(12, dtype='f4')
    result = audit_native_call(ms.learning_curve, ml.Ridge(alpha=-1), x, y, cv=3,
                               train_sizes=[4, 8], error_score=-123., n_jobs=1).assert_native()
    sizes, train, test = result
    assert np.array_equal(np.asarray(sizes), [4, 8])
    assert np.array_equal(np.asarray(train), np.full((2, 3), -123.))
    assert np.array_equal(np.asarray(test), np.full((2, 3), -123.))
