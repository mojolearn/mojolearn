# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Paired FAST quality check for the x_linear fits (lane linear-apple3).

For each seed, a seeded train / held-out split of taxi (regression targets)
and of HIGGS (binary) is fitted on the GPU binding under the environment's
numeric mode, and on the CPU host binding (the reference column), and both
are scored on the held-out rows. Run once per arm (each arm's binding built
in turn); the arms are compared seed by seed, each against the reference.

    python bench/linear_apple3_quality.py --arm <label> [--seeds 0,1,2]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seeds", default="0,1,2")
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
        ("poisson", lambda: lm.PoissonRegressor(max_iter=50), tx, pos,
         lambda m, X, y: [("dev", _poisson_dev(y, _arr(m.predict(X))))]),
        ("gamma", lambda: lm.GammaRegressor(max_iter=50), tx, pos,
         lambda m, X, y: [("dev", _gamma_dev(y, _arr(m.predict(X))))]),
        ("tweedie", lambda: lm.TweedieRegressor(power=1.5, max_iter=50), tx, pos,
         lambda m, X, y: [("dev", _tweedie_dev(y, _arr(m.predict(X))))]),
        ("huber", lambda: lm.HuberRegressor(max_iter=50), tx, fare,
         lambda m, X, y: [("r2", _r2(y, _arr(m.predict(X)))),
                          ("mae", float(np.mean(np.abs(y - _arr(m.predict(X))))))]),
        ("huber-200", lambda: lm.HuberRegressor(max_iter=200), tx, fare,
         lambda m, X, y: [("r2", _r2(y, _arr(m.predict(X)))),
                          ("mae", float(np.mean(np.abs(y - _arr(m.predict(X))))))]),
        ("quantile", lambda: lm.QuantileRegressor(alpha=0.0), tx, fare,
         lambda m, X, y: [("pinball", _pinball(y, _arr(m.predict(X))))]),
        ("quantile-l1", lambda: lm.QuantileRegressor(alpha=0.01, quantile=0.9), tx, fare,
         lambda m, X, y: [("pinball", _pinball(y, _arr(m.predict(X)), 0.9))]),
        ("logistic-cv", lambda: lm.LogisticRegressionCV(Cs=4, cv=3, max_iter=50), hx, hy,
         lambda m, X, y: [("logloss", _logloss(_arr(m.predict_proba(X))[:, 1], y)),
                          ("acc", float(np.mean(_arr(m.predict(X)) == y))),
                          ("C", float(m.C_[0]))]),
        ("logistic-cv-taxi", lambda: lm.LogisticRegressionCV(Cs=4, cv=3, max_iter=50), tx, d["card"],
         lambda m, X, y: [("logloss", _logloss(_arr(m.predict_proba(X))[:, 1], y)),
                          ("acc", float(np.mean(_arr(m.predict(X)) == y))),
                          ("C", float(m.C_[0]))]),
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
            ytr, yte = y[tr], np.asarray(y[te], dtype=np.float64)
            for col in cols:
                try:
                    est = _pin(make(), _modules(col))
                    t0 = time.perf_counter()
                    est.fit(Xtr, ytr)
                    t1 = time.perf_counter()
                    ms = " ".join(f"{k}={v:.7f}" for k, v in score(est, Xte, yte))
                    print(f"QUAL3 {a.arm} {name} {col} seed={seed} {ms} "
                          f"n_iter={getattr(est, 'n_iter_', '?')} fit={t1 - t0:.3f}", flush=True)
                except Exception as e:  # one case must not end the arm
                    print(f"QUAL3 {a.arm} {name} {col} seed={seed} ERROR {e!r}"[:300], flush=True)


if __name__ == "__main__":
    main()
