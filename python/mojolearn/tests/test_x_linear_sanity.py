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


@case("ridge-cv")
def _():
    from sklearn import linear_model as sk
    X, yr, yc, y3 = _data(noise=2.0)
    ok = True
    for fi in (True, False):
        alphas = (0.1, 3.0, 30.0, 300.0)
        a = ml.RidgeCV(alphas=alphas, fit_intercept=fi).fit(X, yr)
        b = sk.RidgeCV(alphas=alphas, fit_intercept=fi).fit(X.astype(np.float64), yr.astype(np.float64))
        ok &= _close(f"fi={fi} alpha_ {a.alpha_} vs {b.alpha_}", [a.alpha_], [b.alpha_], 0)
        ok &= _close(f"fi={fi} best_score_ (relative)", [a.best_score_ / b.best_score_], [1.0], 1e-4)
        ok &= _close(f"fi={fi} coef", a.coef_, b.coef_, 1e-4)
    return ok


@case("lasso-cv")
def _():
    from sklearn import linear_model as sk
    X, yr, yc, y3 = _data(noise=2.0)
    X = X.copy()
    X[:, 5:] *= 0.05
    ok = True
    for fi in (True, False):
        a = ml.LassoCV(fit_intercept=fi).fit(X, yr)
        b = sk.LassoCV(fit_intercept=fi).fit(X.astype(np.float64), yr.astype(np.float64))
        print(f"  fi={fi} alpha_ {a.alpha_:.6g} vs {b.alpha_:.6g}")
        ok &= _close("alphas_ (relative)", np.asarray(a.alphas_) / b.alphas_, np.ones(len(b.alphas_)), 1e-4)
        ok &= _close("mse_path_ (relative)", np.asarray(a.mse_path_) / b.mse_path_, np.ones(b.mse_path_.shape), 5e-3)
        ok &= _close("alpha_ (relative)", [a.alpha_ / b.alpha_], [1.0], 1e-3)
        ok &= _close("coef", a.coef_, b.coef_, 2e-3)
    return ok


@case("enet-cv")
def _():
    from sklearn import linear_model as sk
    X, yr, yc, y3 = _data(noise=2.0)
    X = X.copy()
    X[:, 5:] *= 0.05
    ok = True
    for l1 in (0.5, [0.1, 0.5, 0.9]):
        a = ml.ElasticNetCV(l1_ratio=l1, cv=4).fit(X, yr)
        b = sk.ElasticNetCV(l1_ratio=l1, cv=4).fit(X.astype(np.float64), yr.astype(np.float64))
        print(f"  l1={l1} alpha_ {a.alpha_:.6g} vs {b.alpha_:.6g}, l1_ratio_ {a.l1_ratio_} vs {b.l1_ratio_}")
        ok &= _close("alphas_ (relative)", np.asarray(a.alphas_) / b.alphas_, np.ones(np.shape(b.alphas_)), 1e-4)
        ok &= _close("mse_path_ (relative)", np.asarray(a.mse_path_) / b.mse_path_, np.ones(b.mse_path_.shape), 5e-3)
        ok &= _close("l1_ratio_", [a.l1_ratio_], [b.l1_ratio_], 1e-6)
        ok &= _close("coef", a.coef_, b.coef_, 2e-3)
    return ok


@case("logistic-cv")
def _():
    from sklearn import linear_model as sk
    from sklearn.model_selection import StratifiedKFold
    from mojolearn._expansion_linear import _stratified_kfold_ids
    X, yr, yc, y3 = _data(noise=1.0)
    rng = np.random.default_rng(3)
    flip = rng.random(len(yc)) < 0.15
    ycn = np.where(flip, 1 - yc, yc)
    ok = True
    ids = _stratified_kfold_ids(list(y3), 5)
    ref = np.empty(len(y3), int)
    for f, (_, te) in enumerate(StratifiedKFold(5).split(X, y3)):
        ref[te] = f
    ok &= _close("stratified fold ids", ids, ref, 0)
    for y in (ycn, y3):
        # tol 1e-6: at the default 1e-4 both stop far from the multinomial
        # optimum (weak penalty), each in its own place
        a = ml.LogisticRegressionCV(max_iter=1000, tol=1e-6).fit(X, y)
        b = sk.LogisticRegressionCV(max_iter=1000, tol=1e-6).fit(X.astype(np.float64), y)
        k = len(set(y))
        print(f"  k={k} C_ {a.C_[0]:.4g} vs {b.C_[0]:.4g}")
        key = sorted(b.scores_)[-1]
        ok &= _close(f"k={k} scores_", np.asarray(a.scores_[key]), b.scores_[key], 0.02)
        pa, pb = np.asarray(a.predict_proba(X)), b.predict_proba(X)
        ok &= _close(f"k={k} predict_proba", pa, pb, 0.02)
    return ok


@case("isotonic")
def _():
    from sklearn import isotonic as sk
    rng = np.random.default_rng(4)
    ok = True
    for n, ties in ((500, False), (400, True)):
        x = rng.standard_normal(n).astype(np.float32)
        if ties:
            x = np.round(x * 4).astype(np.float32) / 4
        y = (x + 0.7 * rng.standard_normal(n)).astype(np.float32)
        w = rng.uniform(0.5, 2.0, n).astype(np.float32)
        q = np.linspace(-4, 4, 97).astype(np.float32)
        for kw in (dict(), dict(increasing=False), dict(increasing="auto", out_of_bounds="clip"),
                   dict(y_min=-0.5, y_max=0.8, out_of_bounds="clip")):
            a = ml.IsotonicRegression(**kw).fit(x, y, sample_weight=w)
            b = sk.IsotonicRegression(**kw).fit(x.astype(np.float64), y.astype(np.float64), sample_weight=w.astype(np.float64))
            ok &= _close(f"n={n} ties={ties} {kw} X_thresholds_", a.X_thresholds_, b.X_thresholds_, 1e-6) if len(a.X_thresholds_) == len(b.X_thresholds_) else _close(f"n={n} {kw} threshold count", [len(a.X_thresholds_)], [len(b.X_thresholds_)], 0)
            pa, pb = np.asarray(a.predict(q), np.float64), b.predict(q.astype(np.float64))
            same_nan = np.array_equal(np.isnan(pa), np.isnan(pb))
            ok &= same_nan
            ok &= _close(f"n={n} ties={ties} {kw} predict (nan mask {'=' if same_nan else '!='})",
                         np.nan_to_num(pa), np.nan_to_num(pb), 2e-5)
    return ok


@case("glm-sw")
def _():
    from sklearn import linear_model as sk
    X, yr, yc, y3 = _data()
    rng = np.random.default_rng(1)
    w = (0.3 * rng.standard_normal(X.shape[1])).astype(np.float32)
    mu = np.exp(X @ w + 0.2)
    ycount = rng.poisson(mu).astype(np.float32)
    ygam = rng.gamma(2.0, mu / 2.0).astype(np.float32)
    sw = rng.uniform(0, 3, len(yr)).astype(np.float32)
    sw[::7] = 0
    ok = True
    for name, a, b, y in (
        ("Poisson", ml.PoissonRegressor(alpha=0.01), sk.PoissonRegressor(alpha=0.01), ycount),
        ("Gamma", ml.GammaRegressor(alpha=0.01), sk.GammaRegressor(alpha=0.01), ygam),
        ("Tweedie1.5", ml.TweedieRegressor(power=1.5, alpha=0.01), sk.TweedieRegressor(power=1.5, alpha=0.01), ygam),
        ("Tweedie0", ml.TweedieRegressor(power=0, alpha=0.01), sk.TweedieRegressor(power=0, alpha=0.01), yr),
    ):
        a.fit(X, y, sample_weight=sw)
        b.fit(X.astype(np.float64), y.astype(np.float64), sample_weight=sw.astype(np.float64))
        ok &= _close(f"{name} weighted coef", a.coef_, b.coef_, 2e-3)
        ok &= _close(f"{name} weighted intercept", [a.intercept_], [b.intercept_], 2e-3)
    return ok


@case("huber-sw")
def _():
    from sklearn import linear_model as sk
    X, yr, yc, y3 = _data()
    y = yr.copy()
    y[::13] += 20.0
    sw = np.random.default_rng(5).uniform(0, 3, len(y)).astype(np.float32)
    sw[::9] = 0
    a = ml.HuberRegressor().fit(X, y, sample_weight=sw)
    b = sk.HuberRegressor().fit(X.astype(np.float64), y.astype(np.float64), sample_weight=sw.astype(np.float64))
    ok = _close("weighted coef", a.coef_, b.coef_, 5e-3)
    ok &= _close("weighted intercept", [a.intercept_], [b.intercept_], 5e-3)
    ok &= _close("weighted scale", [a.scale_], [b.scale_], 5e-3 * max(1, b.scale_))
    return ok


@case("quantile-sw")
def _():
    from sklearn import linear_model as sk
    X, yr, yc, y3 = _data(noise=1.0)
    sw = np.random.default_rng(6).uniform(0, 3, len(yr)).astype(np.float32)
    sw[::9] = 0
    ok = True
    for q, alpha in ((0.5, 0.01), (0.8, 0.05)):
        a = ml.QuantileRegressor(quantile=q, alpha=alpha).fit(X, yr, sample_weight=sw)
        b = sk.QuantileRegressor(quantile=q, alpha=alpha).fit(X.astype(np.float64), yr.astype(np.float64), sample_weight=sw.astype(np.float64))

        def obj(coef, icpt):
            r = yr - X.astype(np.float64) @ np.asarray(coef, np.float64) - icpt
            return np.sum(sw * np.where(r >= 0, q * r, (q - 1) * r)) / sw.sum() + alpha * np.abs(coef).sum()
        oa, ob = obj(a.coef_, a.intercept_), obj(b.coef_, b.intercept_)
        print(f"  q={q} weighted objective {oa:.6f} vs {ob:.6f}")
        ok &= _close("weighted objective (relative)", [oa / ob], [1.0], 2e-3)
    return ok


@case("sgd-w")
def _():
    from sklearn import linear_model as sk
    from sklearn.metrics import balanced_accuracy_score
    X, yr, yc, y3 = _data()
    keep = (y3 != 2) | (np.arange(len(y3)) % 5 == 0)   # an imbalanced third class
    X, yr, y3 = X[keep], yr[keep], y3[keep]
    sw = np.random.default_rng(7).uniform(0.2, 2, len(yr)).astype(np.float32)
    ok = True
    for cw in (None, "balanced", {0: 1.0, 1: 2.0, 2: 5.0}):
        a = ml.SGDClassifier(class_weight=cw, random_state=0).fit(X, y3, sample_weight=sw)
        b = sk.SGDClassifier(class_weight=cw, random_state=0).fit(X, y3, sample_weight=sw)
        ba = balanced_accuracy_score(y3, np.asarray(a.predict(X)))
        bb = balanced_accuracy_score(y3, b.predict(X))
        ok &= _close(f"class_weight={cw} balanced accuracy {ba:.3f} vs {bb:.3f}", ba, bb, 0.06)
    for Est, name in ((ml.Perceptron, "Perceptron"), (ml.PassiveAggressiveClassifier, "PA")):
        SkEst = getattr(sk, name if name != "PA" else "PassiveAggressiveClassifier")
        kw = dict(sample_weight=sw) if name == "Perceptron" else {}  # their PA.fit takes no sample_weight
        a = Est(class_weight="balanced", random_state=0).fit(X, y3, **kw)
        b = SkEst(class_weight="balanced", random_state=0).fit(X, y3, **kw)
        ba = balanced_accuracy_score(y3, np.asarray(a.predict(X)))
        bb = balanced_accuracy_score(y3, b.predict(X))
        ok &= _close(f"{name} balanced accuracy {ba:.3f} vs {bb:.3f}", ba, bb, 0.08)
    a = ml.SGDRegressor(random_state=0).fit(X, yr, sample_weight=sw).score(X, yr)
    b = sk.SGDRegressor(random_state=0).fit(X, yr, sample_weight=sw).score(X, yr)
    ok &= _close(f"SGDRegressor weighted R2 {a:.4f} vs {b:.4f}", a, b, 0.02)
    return ok


@case("ridge-w")
def _():
    from sklearn import linear_model as sk
    X, yr, yc, y3 = _data(noise=2.0)
    sw = np.random.default_rng(8).uniform(0, 3, len(yr)).astype(np.float32)
    sw[::9] = 0
    ok = True
    for fi in (True, False):
        for cw in (None, "balanced"):
            a = ml.RidgeClassifier(alpha=2.0, fit_intercept=fi, class_weight=cw).fit(X, y3, sample_weight=sw)
            b = sk.RidgeClassifier(alpha=2.0, fit_intercept=fi, class_weight=cw).fit(X.astype(np.float64), y3, sample_weight=sw.astype(np.float64))
            ok &= _close(f"RidgeClassifier fi={fi} cw={cw} coef", a.coef_, b.coef_, 1e-4)
            ok &= _close(f"RidgeClassifier fi={fi} cw={cw} intercept", a.intercept_, b.intercept_, 1e-4)
        alphas = (0.1, 3.0, 30.0, 300.0)
        a = ml.RidgeCV(alphas=alphas, fit_intercept=fi).fit(X, yr, sample_weight=sw)
        b = sk.RidgeCV(alphas=alphas, fit_intercept=fi).fit(X.astype(np.float64), yr.astype(np.float64), sample_weight=sw.astype(np.float64))
        ok &= _close(f"RidgeCV fi={fi} alpha_ {a.alpha_} vs {b.alpha_}", [a.alpha_], [b.alpha_], 0)
        ok &= _close(f"RidgeCV fi={fi} best_score_ (relative)", [a.best_score_ / b.best_score_], [1.0], 1e-4)
        ok &= _close(f"RidgeCV fi={fi} coef", a.coef_, b.coef_, 1e-4)
    return ok


@case("bayes-sw")
def _():
    from sklearn import linear_model as sk
    X, yr, yc, y3 = _data(noise=0.5)
    sw = np.random.default_rng(9).uniform(0, 3, len(yr)).astype(np.float32)
    sw[::9] = 0
    ok = True
    for fi in (True, False):
        a = ml.BayesianRidge(fit_intercept=fi).fit(X, yr, sample_weight=sw)
        b = sk.BayesianRidge(fit_intercept=fi).fit(X.astype(np.float64), yr.astype(np.float64), sample_weight=sw.astype(np.float64))
        ok &= _close(f"fi={fi} weighted coef", a.coef_, b.coef_, 1e-3)
        ok &= _close(f"fi={fi} weighted intercept", [a.intercept_], [b.intercept_], 1e-3)
        ok &= _close(f"fi={fi} weighted alpha (relative)", [a.alpha_ / b.alpha_], [1.0], 1e-2)
    return ok


@case("logistic-cv-w")
def _():
    from sklearn import linear_model as sk
    X, yr, yc, y3 = _data(noise=1.0)
    sw = np.random.default_rng(10).uniform(0.2, 3, len(yr)).astype(np.float32)
    ok = True
    for y, cw in ((yc, None), (y3, "balanced")):
        a = ml.LogisticRegressionCV(max_iter=1000, tol=1e-6, class_weight=cw).fit(X, y, sample_weight=sw)
        b = sk.LogisticRegressionCV(max_iter=1000, tol=1e-6, class_weight=cw).fit(X.astype(np.float64), y, sample_weight=sw.astype(np.float64))
        k = len(set(y))
        print(f"  k={k} cw={cw} C_ {a.C_[0]:.4g} vs {b.C_[0]:.4g}")
        pa, pb = np.asarray(a.predict_proba(X)), b.predict_proba(X)
        ok &= _close(f"k={k} weighted predict_proba", pa, pb, 0.02)
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
