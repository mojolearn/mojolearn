# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Paired FAST quality check for the x_linear fits (lane linear-apple3).

For each seed, a seeded train / held-out split of taxi (regression targets)
and of HIGGS (binary) is fitted on the GPU binding under the environment's
numeric mode, and on the CPU host binding (the reference column), and both
are scored on the held-out rows. Run once per arm (each arm's binding built
in turn); the arms are compared seed by seed, each against the reference.

    python bench/linear_apple3_quality.py --arm <label> [--seeds 0,1,2,3,4]
        [--train 100000] [--test 100000] [--cases poisson,...] [--no-host]

Lines: `QUAL3 <arm> <case> <column> seed=<s> <metric>=<v> ... n_iter=<k> fit=<s>`.
Metrics are computed in float64 from the predictions (lower is better for
every deviance and loss; higher for r2 and acc).
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))

from x_linear_speed import _load, _modules, _pin  # noqa: E402


def _arr(v):
    return np.asarray(v.tolist() if hasattr(v, "tolist") else v, dtype=np.float64)


def _poisson_dev(y, mu):
    mu = np.maximum(mu, 1e-12)
    return float(2.0 * np.mean(y * np.log(np.maximum(y, 1e-12) / mu) - (y - mu)))


def _gamma_dev(y, mu):
    mu = np.maximum(mu, 1e-12)
    return float(2.0 * np.mean(np.log(mu / y) + y / mu - 1.0))


def _tweedie_dev(y, mu, p=1.5):
    mu = np.maximum(mu, 1e-12)
    return float(2.0 * np.mean(y ** (2 - p) / ((1 - p) * (2 - p)) - y * mu ** (1 - p) / (1 - p)
                               + mu ** (2 - p) / (2 - p)))


def _r2(y, pr):
    return 1.0 - float(np.sum((y - pr) ** 2) / np.sum((y - y.mean()) ** 2))


def _pinball(y, pr, q=0.5):
    e = y - pr
    return float(np.mean(np.maximum(q * e, (q - 1) * e)))


def _logloss(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _wb(m):
    """(w, b) in float64 from coef_ and intercept_ (one output)."""
    w = _arr(m.coef_).ravel()
    b = float(np.asarray(_arr(m.intercept_)).ravel()[0])
    return w, b


def _glm(kind, power=1.5):
    """Held-out deviance, and the TRAINING objective the solver minimizes
    (mean half deviance + alpha / 2 |w|^2), both in float64 from the
    coefficients: `obj` says which arm is the better minimizer."""
    def half(y, mu):
        if kind == "poisson":
            return 0.5 * _poisson_dev(y, mu)
        if kind == "gamma":
            return 0.5 * _gamma_dev(y, mu)
        return 0.5 * _tweedie_dev(y, mu, power)

    def score(m, X, y, Xtr, ytr):
        w, b = _wb(m)
        mu = np.exp(X.astype(np.float64) @ w + b)
        mut = np.exp(Xtr.astype(np.float64) @ w + b)
        obj = half(np.asarray(ytr, np.float64), mut) + 0.5 * float(m.alpha) * float(w @ w)
        return [("dev", 2.0 * half(y, mu)), ("obj", obj)]
    return score


def _huber_score(m, X, y, Xtr, ytr):
    w, b = _wb(m)
    pr = X.astype(np.float64) @ w + b
    r = np.asarray(ytr, np.float64) - (Xtr.astype(np.float64) @ w + b)
    sg, eps = float(m.scale_), float(m.epsilon)
    out = np.abs(r) > eps * sg
    obj = (len(r) * sg + np.sum(r[~out] ** 2) / sg + np.sum(2 * eps * np.abs(r[out]) - sg * eps * eps)
           + float(m.alpha) * float(w @ w)) / len(r)
    return [("r2", _r2(y, pr)), ("mae", float(np.mean(np.abs(y - pr)))), ("obj", float(obj))]


def _quantile_score(q):
    def score(m, X, y, Xtr, ytr):
        w, b = _wb(m)
        pr = X.astype(np.float64) @ w + b
        prt = Xtr.astype(np.float64) @ w + b
        obj = _pinball(np.asarray(ytr, np.float64), prt, q) + float(m.alpha) * float(np.sum(np.abs(w)))
        return [("pinball", _pinball(y, pr, q)), ("obj", obj)]
    return score


def _logcv_score(m, X, y, Xtr, ytr):
    w, b = _wb(m)
    z = X.astype(np.float64) @ w + b
    zt = Xtr.astype(np.float64) @ w + b
    yt = np.asarray(ytr, np.float64)
    c = float(m.C_[0])
    obj = float(np.mean(np.logaddexp(0.0, zt) - yt * zt)) + float(w @ w) / (2.0 * c * len(yt))
    p = 1.0 / (1.0 + np.exp(-z))
    return [("logloss", _logloss(p, y)), ("acc", float(np.mean((z > 0) == (y > 0.5)))), ("C", c), ("obj", obj)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--train", type=int, default=100_000)
    ap.add_argument("--test", type=int, default=100_000)
    ap.add_argument("--cases", default="")
    ap.add_argument("--no-host", action="store_true")
    a = ap.parse_args()
    d = _load(1_000_000)
    import mojolearn._expansion_linear as lm
    only = set(s for s in a.cases.split(",") if s)
    tx, hx = d["tx"], d["hx"]
    fare, pos, hy = d["fare"], d["pos"], d["hy"]
    table = [
        ("poisson", lambda: lm.PoissonRegressor(max_iter=50), tx, pos, _glm("poisson")),
        ("gamma", lambda: lm.GammaRegressor(max_iter=50), tx, pos, _glm("gamma")),
        ("tweedie", lambda: lm.TweedieRegressor(power=1.5, max_iter=50), tx, pos, _glm("tweedie", 1.5)),
        ("huber", lambda: lm.HuberRegressor(max_iter=50), tx, fare, _huber_score),
        ("huber-200", lambda: lm.HuberRegressor(max_iter=200), tx, fare, _huber_score),
        ("quantile", lambda: lm.QuantileRegressor(alpha=0.0), tx, fare, _quantile_score(0.5)),
        ("quantile-l1", lambda: lm.QuantileRegressor(alpha=0.01, quantile=0.9), tx, fare, _quantile_score(0.9)),
        ("logistic-cv", lambda: lm.LogisticRegressionCV(Cs=4, cv=3, max_iter=50), hx, hy, _logcv_score),
        ("logistic-cv-taxi", lambda: lm.LogisticRegressionCV(Cs=4, cv=3, max_iter=50), tx, d["card"],
         _logcv_score),
    ]
    acc = lambda m, X, y, Xtr, ytr: [("acc", float(np.mean(_arr(m.predict(X)) == y)))]
    r2 = lambda m, X, y, Xtr, ytr: [("r2", _r2(y, _arr(m.predict(X)))),
                                    ("r2train", _r2(np.asarray(ytr, np.float64), _arr(m.predict(Xtr))))]
    iso = np.ascontiguousarray(tx[:, 0])
    table += [
        ("sgd-clf", lambda: lm.SGDClassifier(max_iter=5, tol=None, random_state=0), hx, hy, acc),
        ("sgd-reg", lambda: lm.SGDRegressor(max_iter=5, tol=None, random_state=0), tx, fare, r2),
        ("perceptron", lambda: lm.Perceptron(max_iter=5, tol=None, random_state=0), hx, hy, acc),
        ("pa-clf", lambda: lm.PassiveAggressiveClassifier(max_iter=5, tol=None, random_state=0), hx, hy, acc),
        ("pa-reg", lambda: lm.PassiveAggressiveRegressor(max_iter=5, tol=None, random_state=0), tx, fare, r2),
        ("sgd-ocsvm", lambda: lm.SGDOneClassSVM(max_iter=5, tol=None, random_state=0), tx, None,
         lambda m, X, y, Xtr, ytr: [("inliers", float(np.mean(_arr(m.predict(X)) > 0)))]),
        ("bayes-ridge", lambda: lm.BayesianRidge(), tx, fare, r2),
        ("ard", lambda: lm.ARDRegression(max_iter=50), tx, fare, r2),
        ("lars", lambda: lm.Lars(), tx, fare, r2),
        ("lasso-lars", lambda: lm.LassoLars(alpha=0.01), tx, fare, r2),
        ("ridge-clf", lambda: lm.RidgeClassifier(), hx, hy, acc),
        ("ridge-cv", lambda: lm.RidgeCV(), tx, fare,
         lambda m, X, y, Xtr, ytr: [("r2", _r2(y, _arr(m.predict(X)))),
                                    ("r2train", _r2(np.asarray(ytr, np.float64), _arr(m.predict(Xtr)))),
                                    ("alpha", float(m.alpha_))]),
        ("lasso-cv", lambda: lm.LassoCV(cv=3, n_alphas=10), tx, fare,
         lambda m, X, y, Xtr, ytr: [("r2", _r2(y, _arr(m.predict(X)))),
                                    ("r2train", _r2(np.asarray(ytr, np.float64), _arr(m.predict(Xtr)))),
                                    ("alpha", float(m.alpha_))]),
        ("enet-cv", lambda: lm.ElasticNetCV(cv=3, n_alphas=10, l1_ratio=[0.5, 0.9]), tx, fare,
         lambda m, X, y, Xtr, ytr: [("r2", _r2(y, _arr(m.predict(X)))),
                                    ("r2train", _r2(np.asarray(ytr, np.float64), _arr(m.predict(Xtr)))),
                                    ("alpha", float(m.alpha_))]),
        ("isotonic", lambda: lm.IsotonicRegression(), iso, fare, r2),
    ]
    cols = ["gpu"] if a.no_host else ["gpu", "host"]
    for seed in [int(s) for s in a.seeds.split(",")]:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(1_000_000)
        tr, te = perm[:a.train], perm[a.train:a.train + a.test]
        for name, make, X, y, score in table:
            if only and name not in only:
                continue
            Xtr, Xte = np.ascontiguousarray(X[tr]), np.ascontiguousarray(X[te])
            ytr = None if y is None else y[tr]
            yte = None if y is None else np.asarray(y[te], dtype=np.float64)
            for col in cols:
                try:
                    est = _pin(make(), _modules(col))
                    t0 = time.perf_counter()
                    est.fit(Xtr) if ytr is None else est.fit(Xtr, ytr)
                    t1 = time.perf_counter()
                    ms = " ".join(f"{k}={v:.8f}" for k, v in score(est, Xte, yte, Xtr, ytr))
                    print(f"QUAL3 {a.arm} {name} {col} seed={seed} {ms} "
                          f"n_iter={getattr(est, 'n_iter_', '?')} fit={t1 - t0:.3f}", flush=True)
                except Exception as e:  # one case must not end the arm
                    print(f"QUAL3 {a.arm} {name} {col} seed={seed} ERROR {e!r}"[:300], flush=True)


if __name__ == "__main__":
    main()
