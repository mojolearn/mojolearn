# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE LINEAR LANE'S SANITY CHECK against scikit-learn (lane/algos-linear).

    cd python && python -m mojolearn.tests.test_x_linear_sanity [name ...]

A tolerance check on small data, not an identity claim: each case fits the
mojolearn estimator and scikit-learn's on the same data and compares the
quantity a user reads (score, coefficients or predictions) within a stated
tolerance. NumPy and scikit-learn are test oracles here only.
"""
import sys

import numpy as np

import mojolearn as ml

CASES = {}


def case(name):
    def deco(fn):
        CASES[name] = fn
        return fn
    return deco


def _data(n=600, d=8, seed=0, noise=0.1):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    w = rng.standard_normal(d).astype(np.float32)
    yr = (X @ w + noise * rng.standard_normal(n)).astype(np.float32)
    yc = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(np.int64)
    y3 = np.digitize(X[:, 0] + 0.3 * X[:, 2], [-0.5, 0.5]).astype(np.int64)
    return X, yr, yc, y3


def _close(tag, ours, theirs, tol):
    ours, theirs = np.asarray(ours, np.float64), np.asarray(theirs, np.float64)
    err = float(np.max(np.abs(ours - theirs))) if ours.size else 0.0
    ok = err <= tol
    print(f"  {tag}: max|diff| {err:.4g} (tol {tol}) {'ok' if ok else 'FAIL'}")
    return ok


@case("sgd")
def _():
    from sklearn import linear_model as sk
    X, yr, yc, y3 = _data()
    ok = True
    for loss in ("hinge", "log_loss", "modified_huber", "squared_hinge", "perceptron"):
        for y in (yc, y3):
            a = ml.SGDClassifier(loss=loss, random_state=0).fit(X, y).score(X, y)
            b = sk.SGDClassifier(loss=loss, random_state=0).fit(X, y).score(X, y)
            ok &= _close(f"SGDClassifier {loss} k={len(set(y))} accuracy {a:.3f} vs {b:.3f}", a, b, 0.06)
    for loss in ("squared_error", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"):
        for pen in ("l2", "l1", "elasticnet"):
            a = ml.SGDRegressor(loss=loss, penalty=pen, random_state=0).fit(X, yr)
            b = sk.SGDRegressor(loss=loss, penalty=pen, random_state=0).fit(X, yr)
            ok &= _close(f"SGDRegressor {loss}/{pen} R2 {a.score(X, yr):.3f} vs {b.score(X, yr):.3f}",
                         a.score(X, yr), b.score(X, yr), 0.05)
    return ok


@case("glm")
def _():
    from sklearn import linear_model as sk
    X, yr, yc, y3 = _data()
    rng = np.random.default_rng(1)
    w = (0.3 * rng.standard_normal(X.shape[1])).astype(np.float32)
    mu = np.exp(X @ w + 0.2)
    ycount = rng.poisson(mu).astype(np.float32)
    ygam = rng.gamma(2.0, mu / 2.0).astype(np.float32)
    ok = True
    for name, a, b, y in (
        ("Poisson", ml.PoissonRegressor(alpha=0.01), sk.PoissonRegressor(alpha=0.01), ycount),
        ("Gamma", ml.GammaRegressor(alpha=0.01), sk.GammaRegressor(alpha=0.01), ygam),
        ("Tweedie1.5", ml.TweedieRegressor(power=1.5, alpha=0.01), sk.TweedieRegressor(power=1.5, alpha=0.01), ygam),
        ("Tweedie0", ml.TweedieRegressor(power=0, alpha=0.01), sk.TweedieRegressor(power=0, alpha=0.01), yr),
        ("Tweedie3", ml.TweedieRegressor(power=3, alpha=0.01), sk.TweedieRegressor(power=3, alpha=0.01), ygam),
    ):
        a.fit(X, y)
        b.fit(X.astype(np.float64), y.astype(np.float64))
        ok &= _close(f"{name} coef", a.coef_, b.coef_, 2e-3)
        ok &= _close(f"{name} intercept", [a.intercept_], [b.intercept_], 2e-3)
        ok &= _close(f"{name} predict (relative)", np.asarray(a.predict(X[:50])) / b.predict(X[:50]) if name != "Tweedie0" else a.predict(X[:50]), 1.0 if name != "Tweedie0" else b.predict(X[:50]), 5e-3)
    return ok


@case("huber")
def _():
    from sklearn import linear_model as sk
    X, yr, yc, y3 = _data()
    y = yr.copy()
    y[::13] += 20.0
    ok = True
    for fi in (True, False):
        a = ml.HuberRegressor(fit_intercept=fi).fit(X, y)
        b = sk.HuberRegressor(fit_intercept=fi).fit(X.astype(np.float64), y.astype(np.float64))
        ok &= _close(f"fit_intercept={fi} coef", a.coef_, b.coef_, 5e-3)
        ok &= _close(f"fit_intercept={fi} intercept", [a.intercept_], [b.intercept_], 5e-3)
        ok &= _close(f"fit_intercept={fi} scale", [a.scale_], [b.scale_], 5e-3 * max(1, b.scale_))
    return ok


@case("bayes")
def _():
    from sklearn import linear_model as sk
    X, yr, yc, y3 = _data(noise=0.5)
    Xs = X.copy()
    Xs[:, 3] = 0.0  # an irrelevant feature for ARD to prune
    ok = True
    for fi in (True, False):
        a = ml.BayesianRidge(fit_intercept=fi).fit(X, yr)
        b = sk.BayesianRidge(fit_intercept=fi).fit(X.astype(np.float64), yr.astype(np.float64))
        ok &= _close(f"BayesianRidge fi={fi} coef", a.coef_, b.coef_, 1e-3)
        ok &= _close(f"BayesianRidge fi={fi} intercept", [a.intercept_], [b.intercept_], 1e-3)
        ok &= _close(f"BayesianRidge fi={fi} alpha (relative)", [a.alpha_ / b.alpha_], [1.0], 1e-2)
        a = ml.ARDRegression(fit_intercept=fi).fit(X, yr)
        b = sk.ARDRegression(fit_intercept=fi).fit(X.astype(np.float64), yr.astype(np.float64))
        ok &= _close(f"ARD fi={fi} coef", a.coef_, b.coef_, 2e-3)
        ok &= _close(f"ARD fi={fi} intercept", [a.intercept_], [b.intercept_], 2e-3)
        ok &= _close(f"ARD fi={fi} predict", a.predict(X[:50]), b.predict(X[:50]), 5e-3)
    return ok


@case("lars")
def _():
    from sklearn import linear_model as sk
    X, yr, yc, y3 = _data(noise=0.5)
    ok = True
    for fi in (True, False):
        for nz in (3, 500):
            a = ml.Lars(fit_intercept=fi, n_nonzero_coefs=nz).fit(X, yr)
            b = sk.Lars(fit_intercept=fi, n_nonzero_coefs=nz).fit(X.astype(np.float64), yr.astype(np.float64))
            ok &= _close(f"Lars fi={fi} nz={nz} coef", a.coef_, b.coef_, 2e-3)
            ok &= _close(f"Lars fi={fi} nz={nz} intercept", [a.intercept_], [b.intercept_], 2e-3)
        for alpha in (0.5, 0.05, 0.001):
            a = ml.LassoLars(alpha=alpha, fit_intercept=fi).fit(X, yr)
            b = sk.LassoLars(alpha=alpha, fit_intercept=fi).fit(X.astype(np.float64), yr.astype(np.float64))
            ok &= _close(f"LassoLars fi={fi} alpha={alpha} coef", a.coef_, b.coef_, 2e-3)
            ok &= _close(f"LassoLars fi={fi} alpha={alpha} intercept", [a.intercept_], [b.intercept_], 2e-3)
    return ok


@case("quantile")
def _():
    from sklearn import linear_model as sk
    X, yr, yc, y3 = _data(noise=1.0)
    ok = True
    for q, alpha, fi in ((0.5, 0.01, True), (0.8, 0.05, True), (0.3, 0.0, False), (0.5, 1.0, True)):
        a = ml.QuantileRegressor(quantile=q, alpha=alpha, fit_intercept=fi).fit(X, yr)
        b = sk.QuantileRegressor(quantile=q, alpha=alpha, fit_intercept=fi).fit(X.astype(np.float64), yr.astype(np.float64))

        def obj(coef, icpt):
            r = yr - X.astype(np.float64) @ np.asarray(coef, np.float64) - icpt
            return np.mean(np.where(r >= 0, q * r, (q - 1) * r)) + alpha * np.abs(coef).sum()
        oa, ob = obj(a.coef_, a.intercept_), obj(b.coef_, b.intercept_)
        print(f"  q={q} alpha={alpha} fi={fi} n_iter={a.n_iter_} objective {oa:.6f} vs {ob:.6f}")
        ok &= _close("objective (relative)", [oa / ob], [1.0], 2e-3)
        ok &= _close("coef", a.coef_, b.coef_, 3e-2)
    return ok


@case("perceptron")
def _():
    from sklearn import linear_model as sk
    X, yr, yc, y3 = _data()
    ok = True
    for y in (yc, y3):
        for pen in (None, "l2"):
            a = ml.Perceptron(penalty=pen).fit(X, y).score(X, y)
            b = sk.Perceptron(penalty=pen).fit(X, y).score(X, y)
            ok &= _close(f"Perceptron k={len(set(y))} penalty={pen} accuracy {a:.3f} vs {b:.3f}", a, b, 0.06)
    return ok


@case("pa")
def _():
    from sklearn import linear_model as sk
    X, yr, yc, y3 = _data()
    ok = True
    for y in (yc, y3):
        for loss in ("hinge", "squared_hinge"):
            a = ml.PassiveAggressiveClassifier(loss=loss, random_state=0).fit(X, y).score(X, y)
            b = sk.PassiveAggressiveClassifier(loss=loss, random_state=0).fit(X, y).score(X, y)
            ok &= _close(f"PAClassifier k={len(set(y))} {loss} accuracy {a:.3f} vs {b:.3f}", a, b, 0.06)
    for loss in ("epsilon_insensitive", "squared_epsilon_insensitive"):
        a = ml.PassiveAggressiveRegressor(loss=loss, random_state=0).fit(X, yr).score(X, yr)
        b = sk.PassiveAggressiveRegressor(loss=loss, random_state=0).fit(X, yr).score(X, yr)
        ok &= _close(f"PARegressor {loss} R2 {a:.4f} vs {b:.4f}", a, b, 0.02)
    return ok


@case("ocsvm")
def _():
    from sklearn import linear_model as sk
    X, yr, yc, y3 = _data()
    X = X + 3.0
    ok = True
    for nu in (0.1, 0.5):
        a = ml.SGDOneClassSVM(nu=nu, random_state=0).fit(X)
        b = sk.SGDOneClassSVM(nu=nu, random_state=0).fit(X)
        fa_ = float(np.mean(np.asarray(a.predict(X)) == -1))
        fb_ = float(np.mean(b.predict(X) == -1))
        ok &= _close(f"nu={nu} outlier fraction {fa_:.3f} vs {fb_:.3f}", fa_, fb_, 0.08)
    return ok


@case("ridge-clf")
def _():
    from sklearn import linear_model as sk
    X, yr, yc, y3 = _data()
    ok = True
    for y in (yc, y3):
        for fi in (True, False):
            a = ml.RidgeClassifier(alpha=2.0, fit_intercept=fi).fit(X, y)
            b = sk.RidgeClassifier(alpha=2.0, fit_intercept=fi).fit(X.astype(np.float64), y)
            ok &= _close(f"k={len(set(y))} fi={fi} coef", a.coef_, b.coef_, 1e-4)
            ok &= _close(f"k={len(set(y))} fi={fi} intercept", a.intercept_, b.intercept_, 1e-4)
            ok &= _close(f"k={len(set(y))} fi={fi} predict", np.asarray(a.predict(X)), b.predict(X), 0)
    return ok


def main(argv):
    names = argv or list(CASES)
    bad = []
    for name in names:
        print(f"[{name}]")
        if not CASES[name]():
            bad.append(name)
    print("SANITY:", "PASS" if not bad else f"FAIL {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
