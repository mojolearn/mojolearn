#!/usr/bin/env python3
"""scikit-learn's side of the dimensionality sweep: same fixture, same eps
schedule (0.30 * sqrt(d/8)), same n = 200,000, their `algorithm='auto'` and
`n_jobs=-1`. Prints which algorithm auto actually chose, because THAT is the
sweep's mechanism: kd-trees degrade with d and auto walks itself back to
brute force, while the GPU arm's cost is d-linear."""
import math, sys, time
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

REPEATS = 3
M1, M2, M3 = 0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9, 0x94D049BB133111EB


def u01(rows, cols, salt):
    r = np.arange(rows, dtype=np.uint64)[:, None]
    k = np.arange(cols, dtype=np.uint64)[None, :]
    z = (r * np.uint64(M1) + (k + np.uint64(1)) * np.uint64(M2)
         + np.uint64(salt + 1) * np.uint64(M3))
    z = (z ^ (z >> np.uint64(30))) * np.uint64(M2)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(M3)
    z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)


def main():
    n = 200000
    for d in (32, 64, 128):
        idx = np.ascontiguousarray(u01(n, d, 1), dtype=np.float32)
        q = np.ascontiguousarray(u01(2000, d, 2), dtype=np.float32)
        for _ in range(REPEATS):
            t = time.perf_counter()
            nn = NearestNeighbors(n_neighbors=10, n_jobs=-1).fit(idx)
            nn.kneighbors(q)
            dt = time.perf_counter() - t
            print(f"ARM knn_d{d}@{n} {dt*1000:.4f}", flush=True)
        print(f"# knn d={d}: auto chose {nn._fit_method}", flush=True)

    for d in (8, 32, 64):
        x = np.ascontiguousarray(u01(n, d, 4) * 4.0, dtype=np.float32)
        eps = 0.30 * math.sqrt(d / 8.0)
        for _ in range(REPEATS):
            t = time.perf_counter()
            db = DBSCAN(eps=eps, min_samples=5, n_jobs=-1).fit(x)
            dt = time.perf_counter() - t
            print(f"ARM dbscan_d{d}@{n} {dt*1000:.4f}", flush=True)
        n_noise = int((db.labels_ == -1).sum())
        print(f"# dbscan d={d}: eps={eps:.3f}, {n_noise}/{n} noise", flush=True)


if __name__ == "__main__":
    main()
