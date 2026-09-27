# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""MultiRMSE, multi-target regression GradientBoosting (lane/algos-trees,
2026-09-27).

What the fit must do, independent of any vendor: take `y` of shape
`(n_samples, n_targets)` with `n_targets >= 2`, learn every target, predict
RAW `(n_samples, n_targets)`, refuse `predict_proba`, and refuse by name the
arms this implementation does not carry (eval_set, Ordered, the per-dimension
boost_from_average, class weights, the non-symmetric policies). Where
CatBoost is importable, its own MultiRMSE fit on the same pool is a loose
quality reference: the reference here is their GPU learner, which this Mac
cannot run, so only the order of magnitude of the training error is compared,
never a bit. Bit identity across the Metal, CUDA, HIP and CPU columns is the
gbdt-multirmse lane of tools/identity_break.py, not this module.

    cd python && python3 -m pytest -q mojolearn/tests/test_gbdt_multirmse.py
"""
import numpy as np
import pytest

from mojolearn import GradientBoosting
from mojolearn._cpu_reference import reference_training


def _fixture(n=512, d=4, seed=11):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = np.stack([
        X[:, 0] + 0.5 * X[:, 1],
        np.sin(X[:, 2]) - X[:, 0],
        0.25 * X[:, 3] * X[:, 1],
    ], axis=1).astype(np.float32)
    return X, y


def _model(**kw):
    # pinned the way the other loss tests pin themselves: no stochastic arm,
    # an explicit rate (MultiRMSE has no auto learning-rate row, 0.03)
    params = dict(n_estimators=40, max_depth=4, loss="MultiRMSE",
                  learning_rate=0.1, bootstrap_type="No", random_strength=0.0)
    params.update(kw)
    return GradientBoosting(**params)


@reference_training()
def test_learns_every_target_and_predicts_raw():
    X, y = _fixture()
    m = _model().fit(X, y)
    pred = np.asarray(m.predict(X))
    assert pred.shape == y.shape
    assert np.all(np.isfinite(pred))
    assert m.approx_dim_ == 3
    assert m.n_classes_ is None
    assert m.loss_curve_[-1] < m.loss_curve_[0]
    base = np.mean((y - y.mean(axis=0)) ** 2, axis=0)
    err = np.mean((y - pred) ** 2, axis=0)
    # every dimension is learned, not only the first
    assert np.all(err < base), (err, base)


@reference_training()
def test_same_seed_twice_is_the_same_model():
    X, y = _fixture()
    a = _model().fit(X, y)
    b = _model().fit(X, y)
    assert a.model_ == b.model_
    assert np.asarray(a.predict(X)).tobytes() == np.asarray(b.predict(X)).tobytes()


@reference_training()
def test_refusals_by_name():
    X, y = _fixture()
    with pytest.raises(Exception, match="n_targets >= 2|ndim|2-D|dimension"):
        _model().fit(X, y[:, 0])
    with pytest.raises(Exception, match="n_targets >= 2"):
        _model().fit(X, y[:, :1])
    with pytest.raises(NotImplementedError, match="eval_set"):
        _model().fit(X, y, eval_set=(X[:32], y[:32]))
    with pytest.raises(NotImplementedError, match="Ordered"):
        _model(boosting_type="Ordered").fit(X, y)
    with pytest.raises(NotImplementedError, match="boost_from_average"):
        _model(boost_from_average=True).fit(X, y)
    with pytest.raises(Exception, match="optimization scheme is not supported"):
        _model(grow_policy="Depthwise").fit(X, y)
    m = _model(n_estimators=4).fit(X, y)
    with pytest.raises(ValueError, match="predict_proba"):
        m.predict_proba(X)


def test_catboost_multirmse_quality_is_comparable():
    catboost = pytest.importorskip("catboost")
    X, y = _fixture(n=2000)
    Xt, yt = X[:1500], y[:1500]
    Xv, yv = X[1500:], y[1500:]
    kw = dict(iterations=100, depth=4, learning_rate=0.1, l2_leaf_reg=3.0)
    cb = catboost.CatBoostRegressor(
        loss_function="MultiRMSE", boost_from_average=False,
        bootstrap_type="No", random_strength=0.0, border_count=128,
        random_seed=0, verbose=False, allow_writing_files=False, **kw,
    ).fit(Xt, yt)
    with reference_training():
        ours = _model(n_estimators=kw["iterations"], max_depth=kw["depth"],
                      learning_rate=kw["learning_rate"],
                      l2_leaf_reg=kw["l2_leaf_reg"]).fit(Xt, yt)
        ours_pred = np.asarray(ours.predict(Xv), dtype=np.float64)
    cb_pred = np.asarray(cb.predict(Xv), dtype=np.float64)
    ours_mse = float(np.mean((yv - ours_pred) ** 2))
    cb_mse = float(np.mean((yv - cb_pred) ** 2))
    base = float(np.mean((yv - yt.mean(axis=0)) ** 2))
    # LOOSE: their CPU learner is not the GPU learner this restates
    assert ours_mse < 0.5 * base, (ours_mse, base)
    assert ours_mse < 1.5 * cb_mse + 1e-3, (ours_mse, cb_mse)
