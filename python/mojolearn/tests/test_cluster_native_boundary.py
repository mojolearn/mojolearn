"""Public cluster boundary controls; run against the compiled installed candidate."""
import sys

import numpy as np
import pytest
import mojolearn as ml

__mojolearn_user_boundary__ = True


def _poison(calls):
    def fail(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError('Python product callback executed')
    return fail


@pytest.mark.parametrize('name', ['MiniBatchKMeans', 'BisectingKMeans', 'MeanShift',
                                 'OPTICS', 'AffinityPropagation', 'BayesianGaussianMixture'])
def test_cluster_validation_enters_native_without_legacy_python_wrappers(name, monkeypatch):
    import mojolearn._mode as mode
    import mojolearn._sk_validate as validate
    calls = []
    model = getattr(ml, name)(numeric_mode='identical')
    monkeypatch.setattr(mode.NumericModeMixin, '_bind', _poison(calls))
    for field in ('before_fit', 'before_call', 'after_fit', 'after_call'):
        monkeypatch.setattr(validate, field, _poison(calls))
    with pytest.raises(ValueError, match='Expected 2D array'):
        model.fit(np.ones(3, dtype=np.float32))
    assert calls == []


@pytest.mark.parametrize('name', ['MiniBatchKMeans', 'BisectingKMeans', 'MeanShift',
                                 'AffinityPropagation', 'BayesianGaussianMixture'])
def test_cluster_not_fitted_error_is_native(name):
    from mojolearn.exceptions import NotFittedError
    model = getattr(ml, name)(numeric_mode='identical')
    with pytest.raises(NotFittedError):
        model.predict(np.ones((3, 2), dtype=np.float32))


def test_cluster_rejects_callback_initializer_without_calling_it():
    calls = []
    model = ml.MiniBatchKMeans(n_clusters=2, init=_poison(calls), numeric_mode='identical')
    with pytest.raises(TypeError, match='explicit initial centers'):
        model.fit(np.ones((4, 2), dtype=np.float32))
    assert calls == []


@pytest.mark.parametrize('field,base,hook', [('damping', float, '__float__'),
                                          ('max_iter', int, '__int__'),
                                          ('random_state', int, '__lt__')])
def test_affinity_parameter_protocols_are_rejected_without_execution(field, base, hook):
    calls = []
    hostile = type('Hostile', (base,), {hook: _poison(calls)})
    model = ml.AffinityPropagation(numeric_mode='identical')
    setattr(model, field, hostile(1))
    with pytest.raises(TypeError):
        model.fit(np.ones((4, 2), dtype=np.float32))
    assert calls == []


def test_cluster_fit_returns_finished_public_result_and_tracks_state(monkeypatch):
    import mojolearn._mode as mode
    import mojolearn._sk_validate as validate
    calls = []
    model = ml.MiniBatchKMeans(n_clusters=2, max_iter=2, batch_size=4,
                             n_init=1, random_state=7, numeric_mode='identical')
    monkeypatch.setattr(mode.NumericModeMixin, '_bind', _poison(calls))
    for field in ('before_fit', 'before_call', 'after_fit', 'after_call'):
        monkeypatch.setattr(validate, field, _poison(calls))
    x = np.asarray([[0, 0], [0, 1], [8, 8], [8, 9]], dtype=np.float32)
    assert model.fit(x) is model
    result = model.predict(x)
    assert type(result) is np.ndarray and result.dtype == np.dtype('int32')
    assert result.shape == (4,)
    assert vars(model)['_mojolearn_fitted'] is True
    assert '_mojolearn_fitting' not in vars(model)
    assert calls == []


def test_cluster_missing_context_does_not_invoke_module_hook(monkeypatch):
    import mojolearn._native_public_attributes as attributes
    calls = []
    model = ml.AffinityPropagation(numeric_mode='identical')
    fit = model.fit
    monkeypatch.delattr(attributes, '_CONTEXT')
    monkeypatch.setattr(attributes, '__getattr__', _poison(calls), raising=False)
    with pytest.raises(TypeError, match='connected dictionary'):
        fit(np.ones((4, 2), dtype=np.float32))
    assert calls == []
