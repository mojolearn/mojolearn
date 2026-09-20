"""Ranking stochastic defaults exercise the native grouped-gradient path.

The greedy searcher bootstraps rows after forming whole-query gradients:
CatBoost weak_objective_impl.h:21-43, querywise_targets_impl.h:162-188.
Temperature zero must recover the no-bootstrap model; nonzero temperature
must affect a model, and a fixed seed must reproduce it exactly.
"""
import numpy as np
import pytest

from mojolearn import GradientBoosting
from mojolearn._cpu_reference import reference_training


def _fit(loss, **kwargs):
    rng = np.random.default_rng(901)
    X = rng.normal(size=(96, 4)).astype(np.float32)
    y = np.digitize(X[:, 0] + X[:, 1] * .3, [-.7, 0., .7]).astype(np.float32)
    groups = np.repeat(np.arange(12), 8)
    model = GradientBoosting(loss=loss, n_estimators=5, max_depth=3,
                             learning_rate=.1, random_state=19, **kwargs)
    model.fit(X, y, group_id=groups)
    return np.asarray(model.predict(X)), np.asarray(model.loss_curve_)


@pytest.mark.parametrize('loss', ['QueryRMSE', 'PairLogit', 'YetiRank'])
@reference_training()
def test_ranking_defaults_train_and_reproduce(loss):
    pred, curve = _fit(loss)
    repeated, repeated_curve = _fit(loss)
    assert np.isfinite(pred).all() and np.isfinite(curve).all()
    assert np.any(pred != 0)
    assert pred.tobytes() == repeated.tobytes()
    assert curve.tobytes() == repeated_curve.tobytes()


@pytest.mark.parametrize('loss', ['QueryRMSE', 'PairLogit', 'YetiRank'])
@reference_training()
def test_ranking_bootstrap_temperature_zero_and_nonzero(loss):
    plain, _ = _fit(loss, bootstrap_type='No', random_strength=0.)
    zero, _ = _fit(loss, bagging_temperature=0., random_strength=0.)
    sampled, _ = _fit(loss, bagging_temperature=1., random_strength=0.)
    assert plain.tobytes() == zero.tobytes()
    assert sampled.tobytes() != plain.tobytes()


@pytest.mark.parametrize('loss', ['QueryRMSE', 'PairLogit', 'YetiRank'])
@pytest.mark.parametrize('bootstrap', ['Bernoulli', 'Poisson'])
@reference_training()
def test_ranking_discrete_bootstrap(loss, bootstrap):
    pred, _ = _fit(loss, bootstrap_type=bootstrap, subsample=.7)
    assert np.isfinite(pred).all() and np.any(pred != 0)
