# SPDX-License-Identifier: Apache-2.0
"""LinearSVC, LinearSVR and QNRegressor (lane/expose-qn-objectives,
2026-09-20): the Python surface over the six one-target quasi-Newton
objectives.

Runs as a module, `cd python && python3 -m mojolearn.tests.test_qn_objectives_surface`,
and under pytest. The refusal tests need no binding. The fit tests go
through `qn_fit` of whichever `_mojolearn_estimators` this install routes
to, and check the SURFACE and that the reported objective is the documented
one recomputed in float64. Cross-vendor identity is `tools/identity_break.py`'s
six lanes, not this file.
"""
import numpy as np
import pytest

import mojolearn as ml
from mojolearn._cpu_reference import reference_training
from mojolearn.svm import LinearSVC, LinearSVR
from mojolearn.linear_model import QNRegressor


@pytest.fixture(autouse=True)
def _reference_fits():
    """A CPU-only install fits only inside the verifier's scope."""
    with reference_training():
        yield


def _data(n=400, d=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    w = np.arange(1, d + 1, dtype=np.float32) / d
    yr = (X @ w + 0.5 + 0.1 * rng.standard_normal(n)).astype(np.float32)
    yc = (yr > 0.5).astype(np.int64)
    return X, yc, yr


def _z(m, X):
    coef = np.asarray(m.coef_, dtype=np.float64).reshape(-1)
    b = float(np.asarray(m.intercept_).reshape(-1)[0])
    return X.astype(np.float64) @ coef + b, coef


def test_public_names():
    assert ml.svm.LinearSVC is LinearSVC and ml.LinearSVC is LinearSVC
    assert ml.svm.LinearSVR is LinearSVR and ml.LinearSVR is LinearSVR
    assert ml.svm.SVC is ml.SVC and ml.svm.SVR is ml.SVR
    assert ml.QNRegressor is QNRegressor
    for name in ("LinearSVC", "LinearSVR", "QNRegressor", "svm"):
        assert name in ml.__all__


@pytest.mark.parametrize("make, exc, word", [
    (lambda: LinearSVC(loss="log"), ValueError, "loss"),
    (lambda: LinearSVC(penalty="elasticnet"), ValueError, "penalty"),
    (lambda: LinearSVC(C=0.0), ValueError, "C must be positive"),
    (lambda: LinearSVC(penalized_intercept=True), NotImplementedError, "penalized_intercept"),
    (lambda: LinearSVC(class_weight="balanced"), NotImplementedError, "class_weight"),
    (lambda: LinearSVC(multi_class="crammer_singer"), ValueError, "multi_class"),
    (lambda: LinearSVR(loss="hinge"), ValueError, "loss"),
    (lambda: LinearSVR(epsilon=-0.1), ValueError, "epsilon"),
    (lambda: LinearSVR(epsilon=float("nan")), ValueError, "epsilon"),
    (lambda: LinearSVR(penalized_intercept=True), NotImplementedError, "penalized_intercept"),
    (lambda: LinearSVR(tol=0.0), ValueError, "tol"),
    (lambda: QNRegressor(loss="huber"), ValueError, "loss"),
    (lambda: QNRegressor(warm_start=True), NotImplementedError, "warm_start"),
    (lambda: QNRegressor(l2_strength=-1.0), ValueError, "l2_strength"),
    (lambda: QNRegressor(max_iter=0), ValueError, "max_iter"),
])
def test_refused_by_name(make, exc, word):
    with pytest.raises(exc, match=word):
        make()


def test_sample_weight_refused_by_name():
    X, yc, yr = _data(n=16)
    w = np.ones(16, dtype=np.float32)
    for est, y in ((LinearSVC(), yc), (LinearSVR(), yr), (QNRegressor(), yr)):
        with pytest.raises(NotImplementedError, match="sample_weight"):
            est.fit(X, y, sample_weight=w)


@pytest.mark.parametrize("loss", ["hinge", "squared_hinge"])
def test_linear_svc_fits_the_documented_objective(loss):
    X, yc, _ = _data()
    m = LinearSVC(loss=loss, C=2.0).fit(X, yc)
    assert m.classes_ == [0, 1]
    assert np.asarray(m.coef_).shape == (1, 5) and np.asarray(m.coef_).dtype == np.float32
    assert np.asarray(m.intercept_).shape == (1,)
    scores = np.asarray(m.decision_function(X))
    assert scores.shape == (400,) and scores.dtype == np.float32
    pred = np.asarray(m.predict(X))
    assert np.array_equal(pred, (scores > 0).astype(np.int64))
    assert m.score(X, yc) > 0.9
    z, coef = _z(m, X)
    t = np.maximum(0.0, 1.0 - (2.0 * yc - 1.0) * z)
    lz = t if loss == "hinge" else t * t
    objective = lz.mean() + 0.5 * (1.0 / 2.0) / len(X) * float(coef @ coef)
    assert abs(m.objective_ - objective) < 1e-4 * max(1.0, objective)


def test_linear_svc_one_vs_rest_and_labels():
    X, _, yr = _data()
    y3 = np.digitize(yr, np.quantile(yr, [1 / 3, 2 / 3]))
    names = np.array(["low", "mid", "top"])[y3]
    m = LinearSVC(max_iter=200).fit(X, names.tolist())
    assert m.classes_ == ["low", "mid", "top"]
    assert np.asarray(m.coef_).shape == (3, 5) and np.asarray(m.intercept_).shape == (3,)
    scores = np.asarray(m.decision_function(X))
    assert scores.shape == (400, 3)
    ref = X @ np.asarray(m.coef_).T + np.asarray(m.intercept_)
    assert np.max(np.abs(scores - ref)) < 1e-4
    pred = m.predict(X)
    assert [m.classes_[i] for i in scores.argmax(axis=1)] == list(pred)
    # each machine is the two-class fit of its own class against the rest
    one = LinearSVC(max_iter=200).fit(X, (y3 == 2).astype(np.int64))
    assert np.array_equal(np.asarray(one.coef_)[0], np.asarray(m.coef_)[2])


@pytest.mark.parametrize("loss", ["epsilon_insensitive", "squared_epsilon_insensitive"])
@pytest.mark.parametrize("penalty", ["l1", "l2"])
def test_linear_svr_fits_the_documented_objective(loss, penalty):
    X, _, yr = _data()
    m = LinearSVR(loss=loss, penalty=penalty, C=4.0, epsilon=0.05).fit(X, yr)
    assert np.asarray(m.coef_).shape == (5,) and isinstance(m.intercept_, float)
    pred = np.asarray(m.predict(X))
    assert pred.shape == (400,) and pred.dtype == np.float32
    assert m.score(X, yr) > 0.9
    z, coef = _z(m, X)
    d = np.maximum(0.0, np.abs(yr - z) - 0.05)
    lz = d if loss == "epsilon_insensitive" else d * d
    pen = np.abs(coef).sum() if penalty == "l1" else 0.5 * float(coef @ coef)
    objective = lz.mean() + (1.0 / 4.0) / len(X) * pen
    assert abs(m.objective_ - objective) < 1e-4 * max(1.0, objective)


def test_epsilon_is_read():
    X, _, yr = _data()
    a = LinearSVR(epsilon=0.0, penalty="l2").fit(X, yr)
    b = LinearSVR(epsilon=0.5, penalty="l2").fit(X, yr)
    assert not np.array_equal(np.asarray(a.coef_), np.asarray(b.coef_))


@pytest.mark.parametrize("loss", ["squared_error", "absolute_error"])
def test_qn_regressor_fits_the_documented_objective(loss):
    X, _, yr = _data()
    m = QNRegressor(loss=loss, l2_strength=0.5).fit(X, yr)
    assert np.asarray(m.coef_).shape == (5,) and isinstance(m.intercept_, float)
    assert m.score(X, yr) > 0.9
    z, coef = _z(m, X)
    lz = 0.5 * (z - yr) ** 2 if loss == "squared_error" else np.abs(z - yr)
    objective = lz.mean() + 0.5 * 0.5 / len(X) * float(coef @ coef)
    assert abs(m.objective_ - objective) < 1e-4 * max(1.0, objective)


def test_qn_squared_agrees_with_ridge_loosely():
    """The same minimizer as Ridge(alpha) at l2_strength = alpha, reached by
    a different solver in float32: close, not equal."""
    X, _, yr = _data()
    q = QNRegressor(l2_strength=1.0, tol=1e-6).fit(X, yr)
    r = ml.Ridge(alpha=1.0).fit(X, yr)
    assert np.max(np.abs(np.asarray(q.coef_) - np.asarray(r.coef_).reshape(-1))) < 5e-3


def test_binding_refuses_a_mismatched_loss_by_name():
    X, yc, yr = _data(n=32)
    est = QNRegressor()
    from mojolearn.linear_model import _qn_fit_one_target
    from mojolearn._buffer import as_f32_c
    x, _ = as_f32_c(X, ndim=2, name="X")
    t, _ = as_f32_c(yr, ndim=1, name="y")
    with pytest.raises(Exception, match="n_classes == 1"):
        _qn_fit_one_target(est, x, t, 2, 1, 0.0, 0.0, 1e-4, 1e-6)
    with pytest.raises(Exception, match="svr_eps"):
        _qn_fit_one_target(est, x, t, 1, 1, 0.0, 0.0, 1e-4, 1e-6, svr_eps=0.1)
    with pytest.raises(Exception, match="qn_loss_type"):
        _qn_fit_one_target(est, x, t, 1, 42, 0.0, 0.0, 1e-4, 1e-6)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
