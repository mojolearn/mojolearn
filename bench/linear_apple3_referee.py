# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The referee of bench/linear_apple3_quality.py (lane linear-apple3):
scikit-learn, float64, tight tolerance, on the SAME seeded splits, scored
with the same float64 expressions. It says where the minimizer is, so an
arm that stops at another point than the reference can be judged.

    python bench/linear_apple3_referee.py [--seeds 0,1,2,3,4] [--train 50000]
        [--test 50000] [--cases poisson,gamma,tweedie,huber,logistic]

CPU only (the `bench` pixi environment holds scikit-learn). Lines:
`REF3 <case> seed=<s> <metric>=<v> ... obj=<training objective>`.
"""
import argparse
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import linear_apple3_quality as q  # noqa: E402  (the scoring functions)
from x_linear_speed import _load  # noqa: E402


def _view(coef, intercept, **kw):
    m = types.SimpleNamespace(coef_=np.asarray(coef, np.float64).ravel(),
                              intercept_=np.asarray([float(np.ravel(intercept)[0])]), **kw)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--train", type=int, default=50_000)
    ap.add_argument("--test", type=int, default=50_000)
    ap.add_argument("--cases", default="poisson,gamma,tweedie,huber,logistic")
    ap.add_argument("--logistic-c", type=float, default=21.5443469)
    a = ap.parse_args()
    from sklearn import linear_model as sk
    d = _load(1_000_000)
    want = set(a.cases.split(","))
    tx, hx = d["tx"].astype(np.float64), d["hx"].astype(np.float64)
    for seed in [int(s) for s in a.seeds.split(",")]:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(1_000_000)
        tr, te = perm[:a.train], perm[a.train:a.train + a.test]
        for name, kind, power in (("poisson", "poisson", 1.0), ("gamma", "gamma", 2.0), ("tweedie", "tweedie", 1.5)):
            if name not in want:
                continue
            y = d["pos"].astype(np.float64)
            est = sk.TweedieRegressor(power=power, link="log", alpha=1.0, tol=1e-10, max_iter=1000,
                                      solver="newton-cholesky").fit(tx[tr], y[tr])
            m = _view(est.coef_, est.intercept_, alpha=1.0)
            ms = q._glm(kind, power)(m, tx[te], y[te], tx[tr], y[tr])
            print(f"REF3 {name} seed={seed} " + " ".join(f"{k}={v:.8f}" for k, v in ms)
                  + f" n_iter={est.n_iter_}", flush=True)
        if "huber" in want:
            y = d["fare"].astype(np.float64)
            est = sk.HuberRegressor(epsilon=1.35, alpha=1e-4, tol=1e-10, max_iter=5000).fit(tx[tr], y[tr])
            m = _view(est.coef_, est.intercept_, alpha=1e-4, epsilon=1.35, scale_=est.scale_)
            ms = q._huber_score(m, tx[te], y[te], tx[tr], y[tr])
            print(f"REF3 huber seed={seed} " + " ".join(f"{k}={v:.8f}" for k, v in ms)
                  + f" n_iter={est.n_iter_}", flush=True)
        if "logistic" in want:
            y = d["hy"].astype(np.float64)
            est = sk.LogisticRegression(C=a.logistic_c, tol=1e-10, max_iter=5000).fit(hx[tr], y[tr])
            m = _view(est.coef_, est.intercept_, C_=[a.logistic_c])
            ms = q._logcv_score(m, hx[te], y[te], hx[tr], y[tr])
            print(f"REF3 logistic-cv seed={seed} " + " ".join(f"{k}={v:.8f}" for k, v in ms), flush=True)


if __name__ == "__main__":
    main()
