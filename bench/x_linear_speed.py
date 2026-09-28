# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The linear lane's GPU speed board (lane `linear`, speed phase).

Times every linear-family estimator's fit (and its predict) at a realistic
shape on the two R2 datasets (taxi: regression and a binary label from the
card flag; HIGGS: binary), on the GPU binding and on the CPU host binding,
under the numeric mode of the environment (MOJOLEARN_NUMERIC_MODE). Each case
runs once; the fit wall time and a digest of (coef_, intercept_, predict) are
printed, so a before and an after on IDENTICAL show the same bits by eye (the
lane check proves it by column) and GPU == host shows by digest.

    python bench/x_linear_speed.py [--rows 1000000] [--only name,...] [--column gpu|host|both]
        [--warm-rows 10000]

Data: GBM_BENCH_DATA (default ~/datasets/gbm-bench), staged from R2 with
`tools/dataset_store.sh stage` (taxi/taxi_speed.npz, higgs/higgs_speed.npz).
Lines: `XLSPEED <case> <column> fit=<s> predict=<s> <digest>`.
"""
import argparse
import hashlib
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))


def _root():
    return os.environ.get("GBM_BENCH_DATA", os.path.join(os.path.expanduser("~"), "datasets", "gbm-bench"))


def _digest(parts):
    h = hashlib.sha256()
    for p in parts:
        a = np.asarray(p.tolist() if hasattr(p, "tolist") else p, dtype=np.float64)
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def _load(n):
    taxi = np.load(os.path.join(_root(), "taxi", "taxi_speed.npz"))
    higgs = np.load(os.path.join(_root(), "higgs", "higgs_speed.npz"))
    tx = np.ascontiguousarray(np.asarray(taxi["x"][:n], dtype=np.float32)[:, :16])
    # standardize so every solver sees a well-scaled problem (the board times
    # the solvers, not their reaction to raw taxi units)
    tx = ((tx - tx.mean(0)) / (tx.std(0) + 1e-6)).astype(np.float32)
    fare = np.asarray(taxi["fare"][:n], dtype=np.float32)
    card = (np.asarray(taxi["card"][:n]) != 0).astype(np.int64)
    hx = np.ascontiguousarray(np.asarray(higgs["x"][:n], dtype=np.float32))
    hy = np.asarray(higgs["y"][:n]).astype(np.int64)
    pos = np.maximum(fare, np.float32(0.5))
    return dict(tx=tx, fare=fare, pos=pos, card=card, hx=hx, hy=hy)


def cases(d):
    import mojolearn._expansion_linear as lm
    from mojolearn import linear_model as LM
    from mojolearn import svm as SV
    import mojolearn as ML  # Lasso and ElasticNet are top-level, not in linear_model
    tx, hx = d["tx"], d["hx"]
    iso_x = tx[:, 0].copy()
    return [
        ("sgd-clf", "x", lambda: lm.SGDClassifier(max_iter=5, tol=None, random_state=0), hx, d["hy"]),
        ("sgd-reg", "x", lambda: lm.SGDRegressor(max_iter=5, tol=None, random_state=0), tx, d["fare"]),
        ("perceptron", "x", lambda: lm.Perceptron(max_iter=5, tol=None, random_state=0), hx, d["hy"]),
        ("pa-clf", "x", lambda: lm.PassiveAggressiveClassifier(max_iter=5, tol=None, random_state=0), hx, d["hy"]),
        ("pa-reg", "x", lambda: lm.PassiveAggressiveRegressor(max_iter=5, tol=None, random_state=0), tx, d["fare"]),
        ("sgd-ocsvm", "x", lambda: lm.SGDOneClassSVM(max_iter=5, tol=None, random_state=0), tx, None),
        ("poisson", "x", lambda: lm.PoissonRegressor(max_iter=50), tx, d["pos"]),
        ("gamma", "x", lambda: lm.GammaRegressor(max_iter=50), tx, d["pos"]),
        ("tweedie", "x", lambda: lm.TweedieRegressor(power=1.5, max_iter=50), tx, d["pos"]),
        ("huber", "x", lambda: lm.HuberRegressor(max_iter=50), tx, d["fare"]),
        ("bayes-ridge", "x", lambda: lm.BayesianRidge(), tx, d["fare"]),
        ("ard", "x", lambda: lm.ARDRegression(max_iter=50), tx, d["fare"]),
        ("lars", "x", lambda: lm.Lars(), tx, d["fare"]),
        ("lasso-lars", "x", lambda: lm.LassoLars(alpha=0.01), tx, d["fare"]),
        ("quantile", "x", lambda: lm.QuantileRegressor(alpha=0.0), tx, d["fare"]),
        ("ridge-clf", "x", lambda: lm.RidgeClassifier(), hx, d["hy"]),
        ("ridge-cv", "x", lambda: lm.RidgeCV(), tx, d["fare"]),
        ("lasso-cv", "x", lambda: lm.LassoCV(cv=3, n_alphas=10), tx, d["fare"]),
        ("enet-cv", "x", lambda: lm.ElasticNetCV(cv=3, n_alphas=10, l1_ratio=[0.5, 0.9]), tx, d["fare"]),
        ("logistic-cv", "x", lambda: lm.LogisticRegressionCV(Cs=4, cv=3, max_iter=50), hx, d["hy"]),
        ("isotonic", "x", lambda: lm.IsotonicRegression(), iso_x, d["fare"]),
        ("ols", "core", lambda: LM.LinearRegression(), tx, d["fare"]),
        ("ridge", "core", lambda: LM.Ridge(alpha=1.0), tx, d["fare"]),
        ("lasso", "core", lambda: ML.Lasso(alpha=0.01), tx, d["fare"]),
        ("elasticnet", "core", lambda: ML.ElasticNet(alpha=0.01, l1_ratio=0.5), tx, d["fare"]),
        ("logistic", "core", lambda: LM.LogisticRegression(max_iter=100), hx, d["hy"]),
        ("linear-svc", "core", lambda: SV.LinearSVC(max_iter=100), hx, d["hy"]),
        ("linear-svr", "core", lambda: SV.LinearSVR(max_iter=100), tx, d["fare"]),
    ]


def _pin(est, module):
    from mojolearn._expansion_linear import _BINDING
    orig = est._bind
    est._bind = lambda name=None: module if (name or est._BINDING) == _BINDING else orig(name)
    return est


def _modules(column):
    from mojolearn import _backend
    from mojolearn._expansion_linear import _BINDING
    mode = os.environ.get("MOJOLEARN_NUMERIC_MODE", "identical").lower()
    if column == "host":
        return _backend.load_host_module("_mojolearn_x_linear_host")
    return _backend.binding(_BINDING, mode)


def run_case(name, kind, make, X, y, column, warm_rows=0):
    est = make()
    if kind == "x":
        _pin(est, _modules(column))
    elif column == "host":
        return None  # the existing models' host column is the CPU speed lane's
    if warm_rows > 0:
        # an untimed fit of the first rows: the context and this fit's
        # pipelines exist before the clock starts (lane linear-apple3)
        w = make()
        if kind == "x":
            _pin(w, _modules(column))
        k = min(warm_rows, len(X))
        w.fit(X[:k]) if y is None else w.fit(X[:k], y[:k])
    t0 = time.perf_counter()
    est.fit(X) if y is None else est.fit(X, y)
    t1 = time.perf_counter()
    pred = est.predict(X)
    t2 = time.perf_counter()
    parts = [pred]
    for attr in ("coef_", "intercept_"):
        if hasattr(est, attr):
            parts.append(getattr(est, attr))
    return t1 - t0, t2 - t1, _digest(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1_000_000)
    ap.add_argument("--only", default="")
    ap.add_argument("--column", default="gpu", choices=["gpu", "host", "both"])
    ap.add_argument("--warm-rows", type=int, default=0)
    a = ap.parse_args()
    d = _load(a.rows)
    only = set(s for s in a.only.split(",") if s)
    cols = ["gpu", "host"] if a.column == "both" else [a.column]
    for name, kind, make, X, y in cases(d):
        if only and name not in only:
            continue
        for col in cols:
            r = run_case(name, kind, make, X, y, col, a.warm_rows)
            if r is None:
                continue
            print(f"XLSPEED {name} {col} fit={r[0]:.3f} predict={r[1]:.3f} {r[2]}", flush=True)


if __name__ == "__main__":
    main()
