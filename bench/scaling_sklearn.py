#!/usr/bin/env python3

"""scikit-learn's scaling curve on the same sizes and the same data.

**scikit-learn gets its DEFAULT `algorithm='auto'`, not `'brute'`.** That
matters and it is deliberate. Our implementation is brute force; theirs picks
a ball tree or a kd-tree when it thinks that is better. Forcing them to brute
force would flatter us by making them run an algorithm no real user would
run. The comparison a user cares about is our best against their best.

The consequence is that for DBSCAN and k-NN they are running a fundamentally
different algorithm at large n, with better asymptotics and worse constants.
Where the curves cross is the whole point of this file.
**`n_jobs=-1` ON EVERY ESTIMATOR THAT TAKES IT.** Added 2026-08-19 after the
DBSCAN result was already written up. `DBSCAN(n_jobs=None)` and
`NearestNeighbors(n_jobs=None)` are scikit-learn's defaults and they mean ONE
CORE. DBSCAN's neighbour queries are its entire cost, so on a 10-core M4 that
arm was racing a whole GPU on a tenth of the CPU.

That is the same unfairness, pointed the other way, that this harness already
refuses when it declines to hand scikit-learn `algorithm='brute'`. A default
that cripples the incumbent is not their default in any sense a user would
recognise. `KMeans`, `PCA` and the OLS arms reach OpenMP and LAPACK regardless
and were never affected.
"""
import sys, time
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
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 120.0
    for n in (20000, 50000, 100000, 200000, 400000):
        idx = np.ascontiguousarray(u01(n, 32, 1), dtype=np.float32)
        q = np.ascontiguousarray(u01(2000, 32, 2), dtype=np.float32)
        for _ in range(REPEATS):
            t = time.perf_counter()
            NearestNeighbors(n_neighbors=10, n_jobs=-1).fit(idx).kneighbors(q)
            dt = time.perf_counter() - t
            print(f"ARM knn@{n} {dt*1000:.4f}", flush=True)
        if dt > budget:
            print(f"# knn stopped after {n}: {dt:.1f}s exceeds budget", flush=True)
            break

    # 400k and 800k added 2026-08-19 to FIND THE CROSSOVER rather than
    # extrapolate it. With n_jobs=-1 scikit-learn wins from 50,000 up, but the
    # ratio is 0.48x -> 0.72x -> 0.90x, i.e. the gap closes with n. Either it
    # crosses or it asymptotes, and only these two sizes can say which.
    for n in (4000, 16000, 50000, 100000, 200000, 400000, 800000):
        x = np.ascontiguousarray(u01(n, 8, 4) * 4.0, dtype=np.float32)
        for _ in range(REPEATS):
            t = time.perf_counter()
            DBSCAN(eps=0.30, min_samples=5, n_jobs=-1).fit(x)
            dt = time.perf_counter() - t
            print(f"ARM dbscan@{n} {dt*1000:.4f}", flush=True)
        if dt > budget:
            print(f"# dbscan stopped after {n}: {dt:.1f}s exceeds budget", flush=True)
            break


if __name__ == "__main__":
    main()
