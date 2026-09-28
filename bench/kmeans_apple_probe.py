# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Apple k-means probe (lane cluster-apple2): the IVF-shaped k-means fits.

The IVF fits (x_ann) are dominated by cluster/'s k-means: the coarse fit
(1M x 28 HIGGS, 1024 lists, k-means|| init, 10 to 20 Lloyd iterations) and
the IVF-PQ codebooks (1M x 2 subspaces, 256 codes, 20 iterations). This
times those shapes, plus the board's 1M x 8 k = 8 fit, at max_iter N and at
max_iter 1, so the difference splits the init from the Lloyd loop without
instrumenting the fit. Every line prints a digest of labels + centers.

    python bench/kmeans_apple_probe.py [--only coarse,cb,board] [--reps 1]

Lines: `KMPROBE <case> <rows>x<cols> k=<k> it=<max_iter> <seconds> <digest> <inertia>`.
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


def higgs(n, cols):
    x = np.load(os.path.join(_root(), "higgs", "higgs_speed.npz"))["x"][:n, cols]
    x = np.asarray(x, dtype=np.float64)
    x = (x - x.mean(0)) / np.where(x.std(0) > 0, x.std(0), 1.0)
    return np.ascontiguousarray(x, dtype=np.float32)


def digest(*vs):
    h = hashlib.sha256()
    for v in vs:
        a = np.asarray(v.to_numpy() if hasattr(v, "to_numpy") else v)
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="coarse,cb,board")
    ap.add_argument("--reps", type=int, default=1)
    a = ap.parse_args()
    import mojolearn as ml
    # name -> (x columns, k, [max_iter values])
    table = {
        "coarse": (slice(0, 28), 1024, [10, 1]),
        "cb": (slice(0, 2), 256, [20, 1]),
        "board": (slice(1, 9), 8, [300, 1]),
    }
    for name in [s for s in a.only.split(",") if s]:
        if name == "ivfsq":
            # the IVF coarse fit inside the x_ann binding (1024 lists, 10 iterations)
            x = higgs(1_000_000, slice(0, 28))
            for rep in range(2):
                est = ml.IVFSQIndex(n_lists=1024, n_probes=32, kmeans_n_iters=10, random_state=0)
                t0 = time.perf_counter()
                est.fit(x)
                print(f"KMPROBE ivfsq-fit 1000000x28 k=1024 it=10 rep{rep} {time.perf_counter() - t0:.4f}", flush=True)
            continue
        cols, k, iters = table[name]
        x = higgs(1_000_000, cols)
        for it in iters:
            build = lambda: ml.KMeans(n_clusters=k, max_iter=it, random_state=0)
            try:
                build().fit(x)
                best = float("inf")
                est = None
                for _ in range(a.reps):
                    est = build()
                    t0 = time.perf_counter()
                    est.fit(x)
                    best = min(best, time.perf_counter() - t0)
                lab = np.asarray(est.labels_)
                print(f"KMPROBE {name} {x.shape[0]}x{x.shape[1]} k={k} it={it} {best:.4f} "
                      f"{digest(lab, est.cluster_centers_)} {float(est.inertia_):.6g}", flush=True)
            except Exception as e:
                print(f"KMPROBE {name} k={k} it={it} ERROR {type(e).__name__}: {str(e)[:200]}", flush=True)


if __name__ == "__main__":
    main()
