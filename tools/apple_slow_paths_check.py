#!/usr/bin/env python3
"""Checks for lane apple-slow-paths (orchestrator tool, not runtime code).

fa   : FactorAnalysis 20000 x 64 (seeded synthetic, 8 factors) in both modes,
       each mode fitted twice in one process; prints n_iter, score and the
       wall ms of the first (cold) and second (warm) fit, so a one-time cost
       (binding import, pipeline creation) is told apart from the fit.
       Run it twice, once per order (--order fast,identical and
       --order identical,fast).
gbdt : GradientBoostingClassifier and GradientBoostingRegressor defaults
       (Ordered below 50k rows; Logloss walks 10 Newton iterations) on a
       seeded 569 x 30 pool, n_estimators=N; prints sha256 of the model text
       and of the predictions. Timing-free: run it on a build of each arm
       (default, -D MOJOLEARN_GBDT_WALK_DEV_STEP_OFF,
       -D MOJOLEARN_GBDT_ORD_TREE_BUFS_OFF) and on each vendor; every line
       must be equal (same bits).
"""
import argparse
import hashlib
import time

import numpy as np


def _fa(order):
    try:
        from mojolearn.decomposition import FactorAnalysis
    except ImportError:
        from mojolearn import FactorAnalysis
    rng = np.random.default_rng(7)
    n, d, k = 20000, 64, 8
    W = rng.standard_normal((k, d)).astype(np.float32)
    Z = rng.standard_normal((n, k)).astype(np.float32)
    X = (Z @ W + 0.5 * rng.standard_normal((n, d))).astype(np.float32)
    for mode in order:
        for rep in ("cold", "warm"):
            fa = FactorAnalysis(n_components=k, numeric_mode=mode)
            t0 = time.perf_counter()
            fa.fit(X)
            ms = (time.perf_counter() - t0) * 1e3
            print(f"FA mode={mode} {rep} n_iter={fa.n_iter_} score={fa.score(X):.6f} fit_ms={ms:.1f}")


def _gbdt(n_estimators, mode):
    from mojolearn import GradientBoostingClassifier, GradientBoostingRegressor
    rng = np.random.default_rng(11)
    X = rng.standard_normal((569, 30)).astype(np.float32)
    logit = X[:, 0] - 0.5 * X[:, 1] + 0.25 * X[:, 2] * X[:, 3]
    yc = (logit + 0.3 * rng.standard_normal(569) > 0).astype(np.int64)
    yr = (logit + 0.1 * rng.standard_normal(569)).astype(np.float32)
    for name, est, y in (("GBC", GradientBoostingClassifier, yc), ("GBR", GradientBoostingRegressor, yr)):
        m = est(n_estimators=n_estimators, numeric_mode=mode).fit(X, y)
        text = str(m.model_).encode()
        pred = np.ascontiguousarray(m.predict(X)).tobytes()
        print(f"GBDT {name} mode={mode} n_estimators={n_estimators} "
              f"model_sha={hashlib.sha256(text).hexdigest()[:16]} "
              f"pred_sha={hashlib.sha256(pred).hexdigest()[:16]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("fa", "gbdt"))
    ap.add_argument("--order", default="fast,identical")
    ap.add_argument("--n-estimators", type=int, default=50)
    ap.add_argument("--mode", default="identical")
    a = ap.parse_args()
    if a.what == "fa":
        _fa(a.order.split(","))
    else:
        _gbdt(a.n_estimators, a.mode)


if __name__ == "__main__":
    main()
