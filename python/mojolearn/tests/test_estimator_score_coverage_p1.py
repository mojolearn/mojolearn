# SPDX-License-Identifier: Apache-2.0
"""Real-path and refusal tests for estimator ``score`` methods changed since 0.8.55.

Each estimator is fitted on a small problem; ``score`` must equal the accuracy
or R^2 of its own ``predict`` and must refuse an unfitted model, a label vector
of the wrong length and a feature count other than the fitted one.
"""
import warnings

import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score, r2_score

import mojolearn as ml

_rng = np.random.default_rng(0)
_N = 60
X = _rng.normal(size=(_N, 4)).astype(np.float32)
_W = np.array([1.0, -2.0, 0.5, 0.0], np.float32)
Y_REG = (X @ _W + 0.1 * _rng.normal(size=_N)).astype(np.float32)
Y_POS = (np.exp(0.15 * (X @ _W)) + 0.1).astype(np.float32)
Y_BIN = (X[:, 0] + 0.3 * _rng.normal(size=_N) > 0).astype(np.int32)
X_BOOL = (X > 0).astype(np.float32)
X_CAT = _rng.integers(0, 3, (_N, 4)).astype(np.float32)
Y_CAT = ((X_CAT[:, 0] + (_rng.random(_N) < 0.2)) % 3).astype(np.int32)
Y_SEMI = Y_BIN.copy()
Y_SEMI[::3] = -1
X_OUT = X.copy()
X_OUT[:3] += 8
Y_OUT = np.ones(_N, np.int32)
Y_OUT[:3] = -1
X_1D = _rng.random(_N).astype(np.float32)
Y_1D = (X_1D + 0.1 * _rng.normal(size=_N)).astype(np.float32)

REG, CLF = "regressor", "classifier"
# (estimator, init kwargs, X, y for fit (None: unsupervised), y for score, kind)
CASES = [
    pytest.param(ml.EllipticEnvelope, dict(random_state=0), X_OUT, None, Y_OUT, CLF, id="EllipticEnvelope"),
    pytest.param(ml.ARDRegression, {}, X, Y_REG, Y_REG, REG, id="ARDRegression"),
    pytest.param(ml.BayesianRidge, {}, X, Y_REG, Y_REG, REG, id="BayesianRidge"),
    pytest.param(ml.GammaRegressor, {}, X, Y_POS, Y_POS, REG, id="GammaRegressor"),
    pytest.param(ml.PoissonRegressor, {}, X, Y_POS, Y_POS, REG, id="PoissonRegressor"),
    pytest.param(ml.TweedieRegressor, dict(power=1.5), X, Y_POS, Y_POS, REG, id="TweedieRegressor"),
    pytest.param(ml.IsotonicRegression, {}, X_1D, Y_1D, Y_1D, REG, id="IsotonicRegression"),
    pytest.param(ml.Lars, {}, X, Y_REG, Y_REG, REG, id="Lars"),
    pytest.param(ml.LassoLars, dict(alpha=0.01), X, Y_REG, Y_REG, REG, id="LassoLars"),
    pytest.param(ml.LogisticRegressionCV, dict(cv=3), X, Y_BIN, Y_BIN, CLF, id="LogisticRegressionCV"),
    pytest.param(ml.PassiveAggressiveClassifier, dict(random_state=0), X, Y_BIN, Y_BIN, CLF,
                 id="PassiveAggressiveClassifier"),
    pytest.param(ml.PassiveAggressiveRegressor, dict(random_state=0), X, Y_REG, Y_REG, REG,
                 id="PassiveAggressiveRegressor"),
    pytest.param(ml.Perceptron, dict(random_state=0), X, Y_BIN, Y_BIN, CLF, id="Perceptron"),
    pytest.param(ml.QuantileRegressor, dict(alpha=0.0), X, Y_REG, Y_REG, REG, id="QuantileRegressor"),
    pytest.param(ml.RidgeCV, {}, X, Y_REG, Y_REG, REG, id="RidgeCV"),
    pytest.param(ml.RidgeClassifier, {}, X, Y_BIN, Y_BIN, CLF, id="RidgeClassifier"),
    pytest.param(ml.SGDClassifier, dict(random_state=0), X, Y_BIN, Y_BIN, CLF, id="SGDClassifier"),
    pytest.param(ml.SGDRegressor, dict(random_state=0), X, Y_REG, Y_REG, REG, id="SGDRegressor"),
    pytest.param(ml.LabelPropagation, {}, X, Y_SEMI, Y_BIN, CLF, id="LabelPropagation"),
    pytest.param(ml.LabelSpreading, {}, X, Y_SEMI, Y_BIN, CLF, id="LabelSpreading"),
    pytest.param(ml.NearestCentroid, {}, X, Y_BIN, Y_BIN, CLF, id="NearestCentroid"),
    pytest.param(ml.SVGP, dict(n_inducing=8), X, Y_REG, Y_REG, REG, id="SVGP"),
    pytest.param(ml.BernoulliNB, {}, X_BOOL, Y_BIN, Y_BIN, CLF, id="BernoulliNB"),
    pytest.param(ml.CategoricalNB, {}, X_CAT, Y_CAT, Y_CAT, CLF, id="CategoricalNB"),
    pytest.param(ml.ComplementNB, {}, X_CAT, Y_CAT, Y_CAT, CLF, id="ComplementNB"),
]
ARGS = "est,kw,Xf,yf,ys,kind"


def _fitted(est, kw, Xf, yf):
    model = est(**kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(Xf) if yf is None else model.fit(Xf, yf)


@pytest.mark.parametrize(ARGS, CASES)
def test_score_matches_own_predictions(est, kw, Xf, yf, ys, kind):
    model = _fitted(est, kw, Xf, yf)
    got = model.score(Xf, ys)
    assert isinstance(float(got), float)
    pred = model.predict(Xf)
    want = accuracy_score(ys, pred) if kind == CLF else r2_score(ys, pred)
    assert got == pytest.approx(want, rel=1e-4, abs=1e-5)
    # Small, separable problems: every model learns something.
    assert got > (0.5 if kind == CLF else 0.3)


@pytest.mark.parametrize(ARGS, CASES)
def test_score_refuses_unfitted_model(est, kw, Xf, yf, ys, kind):
    with pytest.raises(NotFittedError):
        est(**kw).score(Xf, ys)


@pytest.mark.parametrize(ARGS, CASES)
def test_score_refuses_short_labels(est, kw, Xf, yf, ys, kind):
    model = _fitted(est, kw, Xf, yf)
    with pytest.raises(ValueError):
        model.score(Xf, ys[:-1])


@pytest.mark.parametrize(ARGS, [c for c in CASES if c.values[2].ndim == 2])
def test_score_refuses_other_feature_count(est, kw, Xf, yf, ys, kind):
    model = _fitted(est, kw, Xf, yf)
    with pytest.raises(ValueError, match="features"):
        model.score(Xf[:, :3], ys)


def test_ridgecv_fit_selects_alpha_from_grid():
    model = ml.RidgeCV(alphas=[0.1, 1.0, 10.0]).fit(X, Y_REG)
    assert float(model.alpha_) == pytest.approx(0.1)
    np.testing.assert_allclose(model.coef_, _W, atol=0.05)


@pytest.mark.parametrize("kw,args", [
    pytest.param(dict(alphas=[-1.0]), (X, Y_REG), id="negative-alpha"),
    pytest.param(dict(alphas=[]), (X, Y_REG), id="empty-alphas"),
    pytest.param({}, (X, Y_REG[:-1]), id="short-y"),
    pytest.param({}, (X[:, :, None], Y_REG), id="3d-X"),
])
def test_ridgecv_fit_refuses_bad_input(kw, args):
    with pytest.raises(ValueError):
        ml.RidgeCV(**kw).fit(*args)
