#!/usr/bin/env python3
"""Time scikit-learn on the SAME shapes and the SAME data as bench_main.mojo.

Prints `ARM <name> <milliseconds>` so `run_bench.py` can alternate the two
processes and compare.

THE DATA IS BIT-FOR-BIT THE SAME. `_u01` is the identical splitmix64 mixer
the Mojo benchmark uses, so neither side gets an easier dataset. Generating
it in numpy from the same recurrence is slow but it happens outside the
timing loop, where it belongs.

WHAT IS BEING COMPARED, AND WHAT IS NOT
---------------------------------------
Our GPU against scikit-learn's CPU, on this Mac. That is the comparison the
project exists to make: cuML and cuVS cannot run on Apple silicon at all, so
there is no GPU arm on the other side to compare against.

It is NOT a claim that we beat cuML. It is a claim about what a user on this
machine can actually reach.

Parameters are matched where they mean the same thing: same k, same
n_components, same eps and min_samples, same max_iter, n_init pinned to 1 on
both sides. Where they cannot be matched the difference is printed rather
than hidden: sklearn's LinearRegression uses LAPACK gelsd (an SVD route)
against our normal-equations route, and sklearn's KMeans uses Elkan or Lloyd
with its own initialization.
"""

import sys
import time

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors

REPEATS = 5

M1 = 0x9E3779B97F4A7C15
M2 = 0xBF58476D1CE4E5B9
M3 = 0x94D049BB133111EB
MASK = (1 << 64) - 1


def u01(rows, cols, salt):
    """The Mojo benchmark's splitmix64, vectorized."""
    r = np.arange(rows, dtype=np.uint64)[:, None]
    k = np.arange(cols, dtype=np.uint64)[None, :]
    z = (r * np.uint64(M1) + (k + np.uint64(1)) * np.uint64(M2)
         + np.uint64(salt + 1) * np.uint64(M3))
    z = (z ^ (z >> np.uint64(30))) * np.uint64(M2)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(M3)
    z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)


def emit(name, seconds):
    print(f"ARM {name} {seconds * 1000.0:.4f}")


def main():
    km_rows, km_cols, km_k, km_iter = 4000000, 32, 64, 20
    knn_index, knn_queries, knn_cols, knn_k = 400000, 4000, 32, 10
    pca_rows, pca_cols, pca_comp = 4000000, 32, 8
    db_rows, db_cols = 4000, 16
    ols_rows, ols_cols = 4000000, 32

    km_x = np.ascontiguousarray(u01(km_rows, km_cols, 0) * 10.0, dtype=np.float32)
    km_init = np.ascontiguousarray(
        np.stack([u01(1, km_cols, 5)[0] * 10.0 for _ in range(km_k)]),
        dtype=np.float32,
    )
    # Match the Mojo side's per-centroid seed exactly.
    km_init = np.ascontiguousarray(
        np.stack([u01(c * 7919 + 1, km_cols, 5)[0] * 10.0 for c in range(km_k)]),
        dtype=np.float32,
    )

    kn_idx = np.ascontiguousarray(u01(knn_index, knn_cols, 1), dtype=np.float32)
    kn_q = np.ascontiguousarray(u01(knn_queries, knn_cols, 2), dtype=np.float32)
    pc_x = np.ascontiguousarray(u01(pca_rows, pca_cols, 3) * 4.0, dtype=np.float32)
    db_x = np.ascontiguousarray(u01(db_rows, db_cols, 4) * 2.0, dtype=np.float32)

    ol_a = np.ascontiguousarray(u01(ols_rows, ols_cols, 6) - 0.5, dtype=np.float32)
    ol_w = 1.0 + 0.1 * np.arange(ols_cols)
    ol_b = np.ascontiguousarray(ol_a @ ol_w, dtype=np.float32)

    for _ in range(REPEATS):
        t = time.perf_counter()
        KMeans(n_clusters=km_k, init=km_init, n_init=1,
               max_iter=km_iter, tol=1e-7).fit(km_x)
        emit("kmeans", time.perf_counter() - t)

        t = time.perf_counter()
        NearestNeighbors(n_neighbors=knn_k, algorithm="brute").fit(
            kn_idx).kneighbors(kn_q)
        emit("knn", time.perf_counter() - t)

        t = time.perf_counter()
        PCA(n_components=pca_comp, svd_solver="covariance_eigh").fit(pc_x)
        emit("pca", time.perf_counter() - t)

        t = time.perf_counter()
        DBSCAN(eps=0.35, min_samples=5, algorithm="brute").fit(db_x)
        emit("dbscan", time.perf_counter() - t)

        t = time.perf_counter()
        LinearRegression(fit_intercept=False).fit(ol_a, ol_b)
        emit("ols", time.perf_counter() - t)


if __name__ == "__main__":
    main()
