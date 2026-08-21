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
project exists to make: cuML and cuVS cannot run on Apple silicon at all.

That used to be followed by "so there is no GPU arm on the other side to
compare against", which was an ASSUMPTION and is now a MEASUREMENT. It was
too strong. scikit-learn 1.9 has Array API dispatch, so a torch tensor on the
`mps` device does reach the Apple GPU for some estimators. Measured
2026-08-20 (bench/results/SKLEARN_GPU_BASELINE_2026-08-20.md,
bench/bench_sklearn_gpu.py):

  * `PCA(svd_solver="auto")` -- THEIR DEFAULT, and the `pca` arm below --
    CANNOT reach it. `auto` resolves to `covariance_eigh` at our shape and
    `aten::_linalg_eigh` is unimplemented on MPS.
  * `Ridge(alpha=0, solver="cholesky")` -- the `ols` arm below, and the only
    one -- CANNOT reach it. Array API dispatch supports only `svd`.
  * `PCA(full)` and `Ridge(svd)` CAN, and both are ~1.2x SLOWER on MPS than
    the same call on torch's CPU, because `linalg_svd` has no MPS kernel and
    falls back to the host anyway.

So every arm THIS file times is CPU-only on the other side for a reason that
was checked rather than assumed, and turning scikit-learn's GPU support on
makes their PCA about 10x worse rather than better. The comparison stands;
the justification for it is no longer a guess.

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

    pca            PCA()                 svd_solver='auto'. THEIR DEFAULT,
                                         and asserted at run time to resolve
                                         to covariance_eigh -- OUR route.

    ols            Ridge(alpha=0,        Forms X^T X and solves it. The SAME
                   solver="cholesky")    algorithm class as our lstsq_eig,
                                         so this ratio is hardware only.
                                         The `LinearRegression` gelsd arm was
                                         DELETED 2026-08-20: an SVD of the
                                         full design is a different algorithm
                                         doing more work, and its 25.6x was
                                         not a device result. See the note at
                                         the call site.

    dbscan         algorithm="auto"      A kd-tree/ball-tree, O(n log n)
                                         queries. THEIR DEFAULT, and the
                                         MATCHED arm: their spatial index
                                         against our ball cover (EPS_NN_RBC).

Every arm this file emits is algorithm-matched. Two were deleted on
2026-08-20 for failing that bar: `ols` as `LinearRegression` (gelsd SVD, a
different decomposition) and `dbscan_brute` (their unindexed path against our
ball cover). Both were unfair in OUR favour.

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
   is neighbor queries. Racing a 10-core GPU against a single CPU core is the
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
from sklearn.linear_model import Ridge
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

    # THE PCA ARM'S FAIRNESS, ASSERTED RATHER THAN TIMED.
    #
    # `pca` above passes their default `svd_solver="auto"`. That is only
    # apples-to-apples if `auto` picks `covariance_eigh`, which forms the
    # covariance matrix and eigendecomposes it -- our route. This file used
    # to time a second `pca_cov_eigh` arm to keep that checkable, but a
    # one-sided row on the board is noise, and two timings of the SAME solver
    # invited exactly the misreading it was meant to prevent (they differed
    # 34% on one run, purely from sampling).
    #
    # So it is a check now, not an arm. Probed 2026-08-20 at both 200,000 x 32
    # and 4,000,000 x 32: `auto` resolves to `covariance_eigh` at both. If
    # scikit-learn's heuristic ever moves, this stops the run instead of
    # silently making the `pca` ratio an algorithm comparison.
    _solver = PCA(n_components=pca_comp).fit(pc_x[:200000])._fit_svd_solver
    if _solver != "covariance_eigh":
        raise SystemExit(
            "pca arm is no longer algorithm-matched: sklearn's auto chose "
            f"{_solver!r}, not 'covariance_eigh'. Our route is the covariance "
            "eigendecomposition. Fix the arm before quoting the ratio."
        )
    print(f"# pca: auto resolves to {_solver!r} -- algorithm-matched", flush=True)

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


        # THEIR DEFAULT. `auto` picks a kd-tree or ball tree and does
        # O(n log n) queries; it is what a user gets and it is the arm the
        # scaling curve already used.
        t = time.perf_counter()
        DBSCAN(eps=0.35, min_samples=5, n_jobs=-1).fit(db_x)
        emit("dbscan", time.perf_counter() - t)

        # `dbscan_brute` (`algorithm="brute"`) WAS TIMED HERE AND IS DELETED,
        # 2026-08-20. Its comment claimed "all n^2 pairs, LIKE OURS", which
        # has been false since RBC became our default: `bench_main.mojo:252`
        # calls `dbscan_fit_impl` with `eps_nn_method=EPS_NN_RBC`, a
        # ball-cover INDEX. So it timed their UNINDEXED path against our
        # INDEXED one -- unfair in OUR favour, and it printed as a one-sided
        # row a reader could mistake for a win.
        #
        # The matched arm is `dbscan` above: their spatial index (auto picks
        # a kd-tree or ball tree) against our ball cover, each side's own
        # best structure, device the only variable. Do not re-add this.

        # ALGORITHM-MATCHED, AND THE ONLY OLS ARM. `Ridge(alpha=0,
        # solver="cholesky")` forms X^T X and solves it -- the normal
        # equations, which is our route. alpha=0 makes it ordinary least
        # squares rather than ridge.
        #
        # WHAT WAS DELETED HERE, 2026-08-20, AND WHY IT MUST NOT COME BACK.
        # This file also ran `LinearRegression(fit_intercept=False)`, which
        # is LAPACK `gelsd`: an SVD of the full 4,000,000 x 32 design. It was
        # emitted as `ols` and it was the headline number on the board at
        # 25.6x. That ratio was NOT the device. It was mostly SVD against
        # normal equations -- a different algorithm doing substantially more
        # work, and the more numerically stable one. We are partly faster
        # there because we do less and are more fragile on collinear X, and
        # quoting it as a GPU result was measuring the wrong variable.
        #
        # The rule this now follows, Andrew 2026-08-20: the comparison holds
        # everything the same and varies the DEVICE. An arm where their side
        # runs a different decomposition cannot be on the board, however
        # flattering. If someone wants the gelsd figure for a
        # what-does-a-user-get note, take it somewhere that is not this
        # table and label it as a solver difference, not a speedup.
        t = time.perf_counter()
        Ridge(alpha=0.0, solver="cholesky", fit_intercept=False).fit(ol_a, ol_b)
        emit("ols", time.perf_counter() - t)


if __name__ == "__main__":
    main()
