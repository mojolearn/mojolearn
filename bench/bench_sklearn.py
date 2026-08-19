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
both sides.

WHERE THE ALGORITHM DIFFERS, BOTH ARMS ARE REPORTED
---------------------------------------------------
Revised 2026-08-19. Two arms here were comparing OUR algorithm against a
DIFFERENT one of theirs and reporting the ratio as if it were hardware
against hardware. Naming the difference in a docstring, which is what this
file used to do, is not enough: the number still went into a table.

So each of those two now emits TWO measurements, and a reader can see the
algorithm penalty and the hardware difference separately:

    pca            PCA()                 svd_solver='auto'. THEIR DEFAULT.
    pca_cov_eigh   PCA(covariance_eigh)  Forms the covariance and
                                         eigendecomposes it -- OUR route.

    ols            LinearRegression      LAPACK gelsd, an SVD route.
                                         THEIR DEFAULT: what a user gets.
    ols_normal_eq  Ridge(alpha=0,        Forms X^T X and solves it. The SAME
                   solver="cholesky")    algorithm class as our lstsq_eig,
                                         so this ratio is hardware only.

    dbscan         algorithm="auto"      A kd-tree/ball-tree, O(n log n)
                                         queries. THEIR DEFAULT.
    dbscan_brute   algorithm="brute"     All n^2 pairs, like ours.

**The default arm is the honest headline and the matched arm is the
diagnostic.** A user choosing a library gets the default; forcing scikit-learn
to run something no user would run flatters us. This file previously gave
DBSCAN only `algorithm="brute"` while `scaling_sklearn.py` gave it `auto`, so
the two files disagreed about the same comparison, and the flattering one was
the headline.

Cholesky-vs-eigendecomposition is a real but minor difference INSIDE the
normal-equations route: both form the Gram matrix, which is the only step
that touches rows and therefore all of the cost at 4,000,000 x 32. What
follows is O(cols^3) on a 32 x 32 matrix.

`bench/ols_conditioning.py` prices the accuracy side of that trade, which a
timing harness cannot see.

sklearn's KMeans still uses Elkan or Lloyd with its own initialization and
that one is NOT split into two arms, because there is no sklearn option that
matches ours.

TWO THINGS THIS HARNESS DOES NOT TIME, STATED RATHER THAN BURIED
----------------------------------------------------------------
1. **`n_jobs=-1` is now passed to every estimator that accepts it.** Their
   default is `None`, which means ONE CORE, and DBSCAN's and k-NN's whole cost
   is neighbour queries. Racing a 10-core GPU against a single CPU core is the
   same unfairness this file refuses when it declines to pass
   `algorithm='brute'`. `KMeans`, `PCA` and OLS reach OpenMP and LAPACK
   regardless.

2. **Host-to-device transfer is OUTSIDE our timed region.** Every
   `enqueue_copy` in `bench_main.mojo` happens before the repeat loop, so our
   arms time compute on data already resident. That is the right shape for a
   repeated-call workload and the wrong shape for a one-shot
   `fit(numpy_array)`: at 4,000,000 x 32 it is 512 MB a user would pay once.
   Immaterial against k-means at ~120 ms; **potentially decisive against PCA
   at ~10 ms**, which is the arm where it should be read as a caveat rather
   than a footnote. scikit-learn's arms have no transfer to hide.
"""

import sys
import time

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
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
        # THEIR DEFAULT. `algorithm='auto'` and NOT a hardcoded 'brute',
        # which is what this file used to pass while `scaling_sklearn.py`
        # passed auto -- the identical file-disagreement this docstring
        # claims to have fixed for DBSCAN, left standing for k-NN.
        # Verified on sklearn 1.9.0: at d=32 `auto` resolves to `brute`
        # anyway, so the number does not move. The hardcode was still wrong,
        # because it made the harness assert a choice instead of reporting
        # theirs, and it would have silently stopped tracking them the moment
        # their heuristic changed.
        NearestNeighbors(n_neighbors=knn_k, n_jobs=-1).fit(
            kn_idx).kneighbors(kn_q)
        emit("knn", time.perf_counter() - t)

        t = time.perf_counter()
        # THEIR DEFAULT. `svd_solver='auto'`, which is what a user gets.
        PCA(n_components=pca_comp).fit(pc_x)
        emit("pca", time.perf_counter() - t)

        # ALGORITHM-MATCHED. `covariance_eigh` forms the covariance matrix
        # and eigendecomposes it, which is our route.
        #
        # This file previously ran ONLY this arm and reported it as "pca".
        # Probed on sklearn 1.9.0 at 200,000 x 32: `auto` RESOLVES TO
        # `covariance_eigh` at this shape, so the hardcode moved no number and
        # was not a bias in either direction -- the same finding as the k-NN
        # `algorithm="brute"` hardcode above. It was still wrong for the same
        # reason: a harness that asserts their choice instead of reporting it
        # stops tracking them the moment their heuristic changes, and this one
        # would have gone on printing "sklearn PCA" for a solver sklearn no
        # longer picks.
        #
        # The useful consequence: **PCA is already apples-to-apples.** Their
        # default and ours are the same algorithm, so that ratio needs no
        # second arm to be honest -- unlike DBSCAN and OLS. Both are emitted
        # anyway, so the claim stays checkable.
        t = time.perf_counter()
        PCA(n_components=pca_comp, svd_solver="covariance_eigh").fit(pc_x)
        emit("pca", time.perf_counter() - t)

        # THEIR DEFAULT. `auto` picks a kd-tree or ball tree and does
        # O(n log n) queries; it is what a user gets and it is the arm the
        # scaling curve already used.
        t = time.perf_counter()
        DBSCAN(eps=0.35, min_samples=5, n_jobs=-1).fit(db_x)
        emit("dbscan", time.perf_counter() - t)

        # ALGORITHM-MATCHED. All n^2 pairs, like ours. The gap between this
        # and `dbscan` above is the index, not the hardware.
        t = time.perf_counter()
        DBSCAN(eps=0.35, min_samples=5, algorithm="brute", n_jobs=-1).fit(db_x)
        emit("dbscan_brute", time.perf_counter() - t)

        # THEIR DEFAULT. LAPACK gelsd: an SVD of the full 4,000,000 x 32
        # design. Much more work than ours, and stable on collinear X.
        t = time.perf_counter()
        LinearRegression(fit_intercept=False).fit(ol_a, ol_b)
        emit("ols", time.perf_counter() - t)

        # ALGORITHM-MATCHED. `Ridge(alpha=0, solver="cholesky")` forms X^T X
        # and solves it -- the normal equations, which is our route. alpha=0
        # makes it ordinary least squares rather than ridge.
        t = time.perf_counter()
        Ridge(alpha=0.0, solver="cholesky", fit_intercept=False).fit(ol_a, ol_b)
        emit("ols_normal_eq", time.perf_counter() - t)


if __name__ == "__main__":
    main()
