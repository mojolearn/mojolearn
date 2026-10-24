# SPDX-License-Identifier: Apache-2.0
"""The class-level GaussianProcessClassifier drivers against the plain estimator.

Both drivers are one native call (bindings/parallel_shard_native.mojo): one
binary Laplace fit, or one latent pass, per one-vs-rest class column in a
worker process per device, the columns folded in canonical order. On a
CPU-only install each device index is one worker process.
"""
import numpy as np
import pytest
from mojolearn import parallel_gaussian_process as pg
from mojolearn._gpc_impl import GaussianProcessClassifier as GPC
from mojolearn._native_parallel_shard import _native_parallel_shard


@pytest.fixture(autouse=True)
def setup(monkeypatch):
    monkeypatch.setattr('mojolearn._backend.default_mode', lambda: 'identical')
    from mojolearn._cpu_reference import reference_training
    with reference_training():
        yield


@pytest.mark.parametrize('labels', [[9, 3, 9, 3, 9], ['c', 'a', 'b', 'a', 'c']])
def test_fit_targets_state_and_predictions_equal_plain(labels):
    x = np.arange(15, dtype=np.float32).reshape(5, 3)
    expected = GPC().fit(x, labels)
    model = GPC()
    _native_parallel_shard('trace_take', ())
    assert pg.fit_gaussian_process_classifier(model, x, labels, devices=(0,)) is model
    served = [c['operation'] for e in _native_parallel_shard('trace_take', ()) for c in e['calls']]
    assert served == ['gpc_class_fit'] * len(model.estimators_)
    assert model.classes_ == expected.classes_
    assert model.n_iter_ == expected.n_iter_
    assert model.log_marginal_likelihood_value_ == expected.log_marginal_likelihood_value_
    assert [e.y_train_.tolist() for e in model.estimators_] == [e.y_train_.tolist() for e in expected.estimators_]
    for method in ('predict', 'predict_proba'):
        actual = pg.predict_gaussian_process_classifier(model, x[:4], method=method, devices=(0,))
        wanted = getattr(expected, method)(x[:4])
        assert np.asarray(actual).tobytes() == np.asarray(wanted).tobytes()
        assert np.asarray(actual).shape == np.asarray(wanted).shape


def test_failure_does_not_publish_partial_fit(monkeypatch):
    model = GPC().fit(np.ones((3, 2), np.float32), [1, 2, 3])
    before = dict(model.__dict__)
    # the transport negative control drops one class's result
    monkeypatch.setenv('MOJOLEARN_PAR_TRANSPORT_FAULT', 'drop_result')
    monkeypatch.setenv('MOJOLEARN_HOST_ALLOW_SABOTAGE', '1')
    _native_parallel_shard('faults_take', ())
    with pytest.raises(ValueError, match='incomplete'):
        pg.fit_gaussian_process_classifier(model, np.ones((3, 2), np.float32), [3, 4, 5])
    assert _native_parallel_shard('faults_take', ()) == 1
    assert model.__dict__.keys() == before.keys()
    assert all(model.__dict__[k] is before[k] for k in before)


@pytest.mark.parametrize('labels', [[1, 1], [1], []])
def test_invalid_targets(labels):
    with pytest.raises(ValueError):
        pg.fit_gaussian_process_classifier(GPC(), np.ones((2, 3), np.float32), labels)


def test_prediction_validation():
    with pytest.raises(ValueError, match='method'):
        pg.predict_gaussian_process_classifier(GPC(), np.ones((2, 3), np.float32), method='score')
    with pytest.raises(ValueError, match='fit'):
        pg.predict_gaussian_process_classifier(GPC(), np.ones((2, 3), np.float32))
