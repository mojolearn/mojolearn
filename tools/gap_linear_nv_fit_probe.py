#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Stage probe for lane/gap-linear-nv: the board's LinearSVR, LogisticRegression
and KNeighborsClassifier at the board's shapes (tools/bench_board_more.py:
LIN_ROWS 1,000,000 fit rows; KNN_FIT 200,000 / KNN_QUERIES 4,000; k = 10) on
synthetic standardized rows, D = 11 (taxi-sized) and 220 (Istella-S-sized).

Prints one line per (estimator, D):

    FITPROBE <name> d=<D> ms=<fit ms> n_iter=<iters> ms_per_iter=<..>
    KNNPROBE d=<D> fit_ms=<..> predict_ms=<..>

The QN lines split a fit into its per-iteration price; the board race says
how many iterations the real data takes. Device runs only (nv / amd via lq).
"""
import os
import sys
import time

import numpy as np

import mojolearn as ml

REPS = int(os.environ.get("GAP_PROBE_REPS", "2"))


def _rows(n, d, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d), dtype=np.float32)
    w = rng.standard_normal(d).astype(np.float32) / np.float32(np.sqrt(d))
    z = x @ w + np.float32(0.3) * rng.standard_normal(n, dtype=np.float32)
    return x, z.astype(np.float32)


def _time(fn):
    best = None
    out = None
    for _ in range(REPS):
        t0 = time.perf_counter()
        out = fn()
        ms = (time.perf_counter() - t0) * 1e3
        best = ms if best is None else min(best, ms)
    return best, out


def qn(d):
    x, z = _rows(1_000_000, d, 7 + d)
    y_cls = (z > 0).astype(np.float32)
    makers = {
        "linearsvr": (lambda: ml.LinearSVR(epsilon=0.0, penalty="l2", loss="epsilon_insensitive",
                                           C=1.0, tol=1e-4, max_iter=1000, fit_intercept=True,
                                           penalized_intercept=False), z),
        "logreg": (lambda: ml.LogisticRegression(penalty="l2", C=1.0, tol=1e-4, max_iter=1000,
                                                 fit_intercept=True, solver="qn",
                                                 class_weight=None), y_cls),
    }
    for name, (make, y) in makers.items():
        ms, est = _time(lambda: make().fit(x, y))
        it = int(np.asarray(est.n_iter_).reshape(-1)[0])
        print(f"FITPROBE {name} d={d} ms={ms:.1f} n_iter={it} "
              f"ms_per_iter={ms / max(it, 1):.3f}", flush=True)


def knn(d):
    x, z = _rows(204_000, d, 11 + d)
    xf, xq = x[:200_000], x[200_000:]
    y = (z[:200_000] > 0).astype(np.int32)

    def fit():
        return ml.KNeighborsClassifier(n_neighbors=10, weights="uniform", metric="euclidean",
                                       algorithm="brute", p=2).fit(xf, y)

    fit_ms, est = _time(fit)
    pred_ms, _ = _time(lambda: est.predict(xq))
    fresh_ms, _ = _time(lambda: fit().predict(xq))
    print(f"KNNPROBE d={d} fit_ms={fit_ms:.1f} predict_ms={pred_ms:.1f} "
          f"fit_plus_first_predict_ms={fresh_ms:.1f}", flush=True)


def main():
    which = sys.argv[1:] or ["qn", "knn"]
    for d in (11, 220):
        if "qn" in which:
            qn(d)
        if "knn" in which:
            knn(d)


if __name__ == "__main__":
    main()
