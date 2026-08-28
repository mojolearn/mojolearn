#!/usr/bin/env python3
"""E2U driver: the UNSUPERVISED sub-feature matrix, one identity card per cell.

The unsupervised analog of `tools/e2_matrix_fit.py` (read that docstring
first; the discipline is the same and is not repeated here). `E1U_RESULTS.md`
certified ONE configuration each of k-means, k-NN and DBSCAN across Apple
and AMD through a Mojo driver, and named what it did NOT cover: k-means++'s
float scan, the fused/tiled arms at other k, the ball-cover DBSCAN arm, the
memory-budget batching, and PCA / truncated SVD / OLS entirely. This driver
sweeps the PYTHON surface -- `mojolearn.KMeans`, `.NearestNeighbors`,
`.KNeighborsClassifier`, `.KNeighborsRegressor` (2026-08-23), `.DBSCAN`,
`.PCA`, `.TruncatedSVD`, `.LinearRegression` -- one subprocess per cell, a `MOJOLEARN_IDENTITY_TRACE` card per cell (every one of the six
paths traces: k-means and DBSCAN inside the ported fit, k-NN in
`neighbors/estimator.mojo`, OLS through `ols_fit_traced` since DEVIATION 517,
PCA / tSVD at the host surface since DEVIATION 518), and a sha256 over every
caller-visible output: labels, centroids, inertia, indices, distances,
components, explained variance, singular values, coefficients, intercept,
predictions, transforms, class probabilities, class sets.

THREE VERDICTS PER CELL, ALL OF THEM RESULTS (the E2 set): identical /
divergent-at-stage / refused. A REFUSED cell is a parameter the surface
refuses BY NAME -- `whiten=True`, `metric='cosine'`, `k > 256` under
NUMERIC_IDENTICAL -- and two machines refusing with the same message is
that cell's passing result.

AND A FOURTH THING THIS DRIVER MEASURES THAT E2 DID NOT: REACH. `REACH`
below declares pairs of cells that differ in exactly one parameter and
says whether the outputs are EXPECTED to differ (the parameter steers the
answer) or to be the SAME (the parameter is a memory/scheduling number and
`check_*_invariant` gates say the answer must not move). After the matrix
runs, the parent evaluates every pair and writes the table into
`e2u_cells.json["reach"]` and to stdout. A "differ" pair whose hashes are
equal is a parameter that is accepted and ignored -- the ET `max_features`
finding of E2, which is a BUG, not a result -- and the parent's exit code
says so. A "same" pair whose hashes differ is a broken invariance gate.

INPUTS ARE A PURE FUNCTION OF THE SEED, integer-exact where a float op
could go through a platform BLAS or libm, exactly as E2: numpy's MT19937
stream, targets from an integer matmul scaled by a power of two, planted
ties on a 1/16 grid, DBSCAN blobs at integer centers with power-of-two
half-widths and hashed noise. `inputs` hashes in `e2u_cells.json` prove the
inputs matched before any fit is compared.

MODES. `mojolearn.numeric_mode()` is recorded; the mode is chosen by
`MOJOLEARN_NUMERIC_MODE=identical` at import (a build define, see
`python/mojolearn/_backend.py`). Run the matrix in BOTH modes, TWICE each,
and diff run against run with `tools/e2_matrix_diff.py` (which reads
`e2u_cells.json` as well as `e2_cells.json`): under IDENTICAL two runs must
be byte-identical on every card; under FAST they need not be, and the cells
that are not are a finding to list, not a failure. See
`tools/e2u_README.md` for the exact commands.

usage: PYTHONPATH=python python3 tools/e2u_matrix_fit.py <out_dir> [--only SUBSTR]
       (internal) ... --cell NAME <out_dir>
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback

import numpy as np

SEED = 20260823
N_ROWS, N_COLS = 20000, 24
N_QUERIES = 2000


def sha256_of(arr):
    a = np.ascontiguousarray(np.asarray(arr))
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def make_inputs():
    """Every array here is bit-identical on every platform by construction
    (module docstring). ORDER OF DRAWS IS THE CONTRACT: a new array goes at
    the END so every existing hash is untouched."""
    rng = np.random.RandomState(SEED)
    X = rng.rand(N_ROWS, N_COLS).astype(np.float32)
    Xq = rng.rand(N_QUERIES, N_COLS).astype(np.float32)
    w = rng.rand(N_COLS).astype(np.float32)
    q = (X * 1024.0).astype(np.int64)
    wq = (w * 1024.0).astype(np.int64)
    # the OLS target: an integer matmul scaled by a power of two (exact),
    # plus a hashed integer residual so the fit is not exactly linear
    # (an exactly linear target makes every coefficient a tiny rounding
    # of a known value and the intercept 0.0, which certifies less)
    idx = np.arange(N_ROWS, dtype=np.int64)
    resid = ((idx * 2654435761) % 17).astype(np.float64) - 8.0
    y_reg = ((q @ wq).astype(np.float64) * 2.0 ** -20
             + resid * 2.0 ** -6).astype(np.float32)
    # hashed integer sample weights 1..5 (E2's recipe)
    sw = (1 + (idx * 7919) % 5).astype(np.float32)
    # THE TIE FIXTURE for k-NN: every coordinate on a 1/16 grid, so
    # distances take few distinct values and equidistant neighbours are
    # the rule, not the exception; queries ARE index points (hashed rows),
    # so the self-match at distance 0 and its tie class are in every row.
    # This is the shape on which FAST k-NN returned three different index
    # sets in three runs (UNSUPERVISED_IDENTITY.md).
    X_ties = ((q // 64).astype(np.float64) / 16.0).astype(np.float32)
    tie_rows = (np.arange(500, dtype=np.int64) * 7919) % N_ROWS
    Xq_ties = np.ascontiguousarray(X_ties[tie_rows])
    # THE DBSCAN FIXTURE: 8 blobs at integer centers in [0, 8)^4, side
    # 1/4 (every coordinate is center + rand * 2^-2, exact), of SIZES
    # 1200, 900, 600, 300, 100, 40, 20, 10 -- so `min_samples` has
    # something to move: at eps 0.3 a blob's points all see each other,
    # and min_samples 5 / 50 / 200 keep 8 / 5 / 3 of them as clusters and
    # turn the rest into noise. Plus 400 noise points uniform over the
    # cube, the whole thing shuffled by a seeded permutation.
    centers = rng.randint(0, 8, size=(8, 4)).astype(np.float64)
    sizes = [1200, 900, 600, 300, 100, 40, 20, 10]
    assign = np.repeat(np.arange(8), sizes)
    nb = int(assign.shape[0])
    blob = centers[assign] + rng.rand(nb, 4) * 0.25
    noise = rng.rand(400, 4) * 8.0
    X_blobs = np.vstack([blob, noise])
    perm = rng.permutation(nb + 400)
    X_blobs = np.ascontiguousarray(X_blobs[perm]).astype(np.float32)
    # THE CHAIN FIXTURE: 1000 points on a line at spacing 1/8 (exact),
    # in hashed order, eps 3/16 so each point sees exactly its two
    # neighbours. Label propagation moves one hop per pass, so the
    # connected component takes ~1000 passes to label -- FIVE TIMES cuML's
    # default `max_iterations=200`. Under FAST the shipped default returns
    # a truncated labelling (cuML's behaviour); under IDENTICAL it is
    # refused (DEVIATION 507). Both are results this matrix records.
    chain_pos = ((np.arange(1000, dtype=np.int64) * 7919) % 1000).astype(
        np.float64) * 0.125
    X_chain = np.zeros((1000, 4), dtype=np.float64)
    X_chain[:, 0] = chain_pos
    X_chain = np.ascontiguousarray(X_chain).astype(np.float32)
    # the wide-column case
    X_wide = rng.rand(4096, 200).astype(np.float32)
    # refused shapes for OLS: more columns than rows, and one column
    X_ols_wide = rng.rand(64, 100).astype(np.float32)
    y_ols_wide = (((X_ols_wide * 1024.0).astype(np.int64).sum(axis=1))
                  .astype(np.float64) * 2.0 ** -16).astype(np.float32)
    # ROW-COUNT REGIME: 200k rows, drawn LAST
    Xb = rng.rand(200000, N_COLS).astype(np.float32)
    qb = (Xb * 1024.0).astype(np.int64)
    idxb = np.arange(200000, dtype=np.int64)
    residb = ((idxb * 2654435761) % 17).astype(np.float64) - 8.0
    yb_reg = ((qb @ wq).astype(np.float64) * 2.0 ** -20
              + residb * 2.0 ** -6).astype(np.float32)
    # hashed starting centroids for init='array': 8 rows of X
    c0_rows = (np.arange(8, dtype=np.int64) * 7919 + 13) % N_ROWS
    C0 = np.ascontiguousarray(X[c0_rows])
    # THE k-NN CLASSIFIER LABELS (added 2026-08-23, drawn from no rng, so
    # every hash above is untouched): hashed integer classes over the
    # 20,000 index rows, 3-class as {-1, 0, 1} (a NEGATIVE label, so the
    # class set is not range(n) and the ported monotonic map does work)
    # and 2-class as {0, 1}. The 2-class set at an EVEN k on the tie
    # fixture is where 2-2 VOTE ties happen and the lowest-class rule is
    # exercised.
    y_clf3 = (((idx * 2654435761) % 3)).astype(np.int64) - 1
    y_clf2 = (((idx * 40503) >> 3) % 2).astype(np.int64)
    return dict(X=X, Xq=Xq, y_reg=y_reg, sw=sw, X_ties=X_ties,
                Xq_ties=Xq_ties, X_blobs=X_blobs, X_chain=X_chain,
                X_wide=X_wide,
                X_ols_wide=X_ols_wide, y_ols_wide=y_ols_wide,
                Xb=Xb, yb_reg=yb_reg, C0=C0,
                y_clf3=y_clf3, y_clf2=y_clf2)


# ---------------------------------------------------------------- the cells
# name -> (kind, spec). kind selects the class; spec is the constructor
# kwargs plus driver keys prefixed with '_' (which arrays, fit/query kwargs).
# Names are STABLE -- they are the row labels of the cross-vendor table and
# the card file names.

CELLS = {}


def cell(name, kind, **spec):
    assert name not in CELLS, name
    CELLS[name] = (kind, spec)


def km(**kw):
    return kw


# --- KMeans: X is 20000 x 24, seed 7 -------------------------------------
KM_BASE = dict(n_clusters=8, random_state=7)
cell("kmeans_k8", "kmeans", **KM_BASE)
cell("kmeans_k3", "kmeans", n_clusters=3, random_state=7)
cell("kmeans_k32", "kmeans", n_clusters=32, random_state=7)
cell("kmeans_k8_random", "kmeans", init="random", **KM_BASE)
cell("kmeans_k8_ninit3", "kmeans", n_init=3, **KM_BASE)
cell("kmeans_k8_tol1e-2", "kmeans", tol=1e-2, **KM_BASE)
cell("kmeans_k8_maxiter2", "kmeans", max_iter=2, **KM_BASE)
cell("kmeans_k8_seed11", "kmeans", n_clusters=8, random_state=11)
cell("kmeans_k8_sw", "kmeans", _fit=dict(sample_weight="sw"), **KM_BASE)
cell("kmeans_k8_array", "kmeans", init="array", _init_centroids="C0",
     **KM_BASE)
cell("kmeans_k32_random", "kmeans", n_clusters=32, init="random",
     random_state=7)
cell("kmeans_k8_200k", "kmeans", _X="Xb", **KM_BASE)
cell("kmeans_k8_wide", "kmeans", _X="X_wide", **KM_BASE)
cell("kmeans_k8_ties", "kmeans", _X="X_ties", **KM_BASE)
cell("kmeans_init_bad", "kmeans", init="pca-ish", **KM_BASE)  # REFUSED

# --- NearestNeighbors: index X (20000 x 24), queries Xq (2000 x 24) --------
cell("knn_k1", "knn", n_neighbors=1)
cell("knn_k10", "knn", n_neighbors=10)
cell("knn_k100", "knn", n_neighbors=100)
cell("knn_k10_self", "knn", n_neighbors=10, _Xq="X_head")  # queries in index
cell("knn_k10_ties", "knn", n_neighbors=10, _X="X_ties", _Xq="Xq_ties")
cell("knn_k64_ties", "knn", n_neighbors=64, _X="X_ties", _Xq="Xq_ties")
cell("knn_k100_ties", "knn", n_neighbors=100, _X="X_ties", _Xq="Xq_ties")
cell("knn_k300", "knn", n_neighbors=300)  # REFUSED under IDENTICAL (k>256)
cell("knn_k10_tile64", "knn", n_neighbors=10, query_tile=64)
cell("knn_k10_200k", "knn", n_neighbors=10, _X="Xb")
cell("knn_metric_cosine", "knn", n_neighbors=10, metric="cosine")  # REFUSED
cell("knn_metric_minkowski_p1", "knn", n_neighbors=10, metric="minkowski",
     p=1)  # REFUSED
cell("knn_algorithm_kd_tree", "knn", n_neighbors=10,
     algorithm="kd_tree")  # REFUSED
cell("knn_metric_l2", "knn", n_neighbors=10, metric="l2")  # == knn_k10

# --- KNeighborsClassifier / KNeighborsRegressor (2026-08-23): index X,
# queries Xq; labels y_clf3 ({-1,0,1}) / y_clf2 ({0,1}); targets y_reg.
# The vote and the mean are serial per-row folds in both modes (DEVIATION
# 542), so under IDENTICAL the whole card (search stages + knn_clf.* /
# knn_reg.*) must be byte-identical run to run, and under FAST the vote
# stages are identical whenever the SEARCH's index set is (row 11).
cell("knn_clf_k5", "knn_clf", n_neighbors=5, _y="y_clf3")
cell("knn_clf_k15_3class", "knn_clf", n_neighbors=15, _y="y_clf3")
# 2 classes, k=4, on the 1/16-grid tie fixture: 2-2 vote ties AND
# equidistant neighbours in one cell; the lowest-class rule decides the
# former, the IDENTICAL selector's lowest-index rule the latter
cell("knn_clf_ties", "knn_clf", n_neighbors=4, _y="y_clf2", _X="X_ties",
     _Xq="Xq_ties")
cell("knn_reg_k5", "knn_reg", n_neighbors=5)
cell("knn_reg_k50", "knn_reg", n_neighbors=50)
cell("knn_clf_weights_distance_refused", "knn_clf", n_neighbors=5,
     weights="distance", _y="y_clf3")  # REFUSED

# --- DBSCAN: X_blobs (6400 x 4; 8 blobs + 400 noise) ------------------------
DB_BASE = dict(eps=0.3, min_samples=5)
cell("dbscan_rbc", "dbscan", **DB_BASE)
cell("dbscan_brute", "dbscan", algorithm="brute", **DB_BASE)
cell("dbscan_eps0.15", "dbscan", eps=0.15, min_samples=5)
cell("dbscan_eps1.0", "dbscan", eps=1.0, min_samples=5)
cell("dbscan_min2", "dbscan", eps=0.3, min_samples=2)
cell("dbscan_min50", "dbscan", eps=0.3, min_samples=50)
cell("dbscan_min200", "dbscan", eps=0.3, min_samples=200)
cell("dbscan_min50_brute", "dbscan", eps=0.3, min_samples=50,
     algorithm="brute")
cell("dbscan_rbc_budget1mb", "dbscan", max_mbytes_per_batch=1, **DB_BASE)
cell("dbscan_brute_budget1mb", "dbscan", algorithm="brute",
     max_mbytes_per_batch=1, **DB_BASE)
cell("dbscan_maxiter1", "dbscan", max_iterations=1, **DB_BASE)
# the chain: an EXPLICIT cap of 200 -- what the surface used to default to
# before DEVIATION 519 -- truncates (FAST) or refuses (IDENTICAL); the
# default (None, the fixed point) and an explicit 5000 both reach the fixed
# point -- one cluster, no noise, ~730 passes
cell("dbscan_chain_default", "dbscan", _X="X_chain", eps=0.1875,
     min_samples=2)
cell("dbscan_chain_iter200", "dbscan", _X="X_chain", eps=0.1875,
     min_samples=2, max_iterations=200)
cell("dbscan_chain_iter5000", "dbscan", _X="X_chain", eps=0.1875,
     min_samples=2, max_iterations=5000)
cell("dbscan_chain_iter5000_brute", "dbscan", _X="X_chain", eps=0.1875,
     min_samples=2, max_iterations=5000, algorithm="brute")
cell("dbscan_metric_manhattan", "dbscan", metric="manhattan",
     **DB_BASE)  # REFUSED
cell("dbscan_algorithm_bad", "dbscan", algorithm="kd_tree",
     **DB_BASE)  # REFUSED
cell("dbscan_sw", "dbscan", _fit=dict(sample_weight="sw_blobs"),
     **DB_BASE)  # REFUSED
# a 24-feature uniform fixture: eps chosen so the core/border/noise arms
# are all populated (the driver prints the noise fraction)
cell("dbscan_uniform24", "dbscan", _X="X", eps=1.25, min_samples=5)
cell("dbscan_uniform24_brute", "dbscan", _X="X", eps=1.25, min_samples=5,
     algorithm="brute")

# --- PCA: X (20000 x 24) -----------------------------------------------------
cell("pca_c2", "pca", n_components=2)
cell("pca_c8", "pca", n_components=8)
cell("pca_all", "pca", n_components=None)
cell("pca_c8_wide", "pca", n_components=8, _X="X_wide")
cell("pca_c8_200k", "pca", n_components=8, _X="Xb")
cell("pca_whiten", "pca", n_components=2, whiten=True)  # REFUSED
cell("pca_solver_full", "pca", n_components=2, svd_solver="full")  # REFUSED

# --- TruncatedSVD ------------------------------------------------------------
cell("tsvd_c2", "tsvd", n_components=2)
cell("tsvd_c8", "tsvd", n_components=8)
cell("tsvd_c8_wide", "tsvd", n_components=8, _X="X_wide")
cell("tsvd_randomized", "tsvd", n_components=2,
     algorithm="randomized")  # REFUSED

# --- LinearRegression --------------------------------------------------------
cell("ols_intercept", "ols")
cell("ols_nointercept", "ols", fit_intercept=False)
cell("ols_narrow4", "ols", _X="X_narrow4")
cell("ols_200k", "ols", _X="Xb", _y="yb_reg")
cell("ols_wide_refused", "ols", _X="X_ols_wide", _y="y_ols_wide")  # REFUSED
cell("ols_onecol_refused", "ols", _X="X_col1")  # REFUSED
cell("ols_sw_refused", "ols", _fit=dict(sample_weight="sw"))  # REFUSED

# --- Ridge (DEVIATION 545) -----------------------------------------------------
cell("ridge_a1", "ridge", alpha=1.0)
cell("ridge_a100", "ridge", alpha=100.0)
cell("ridge_nointercept", "ridge", alpha=1.0, fit_intercept=False)

# --- LogisticRegression (DEVIATIONS 546-549) -----------------------------------
# the binary target is derived from X with no new RNG draw (see _derived)
cell("logreg_c1", "logreg", C=1.0, _y="y_bin")
cell("logreg_c100", "logreg", C=100.0, _y="y_bin")
cell("logreg_nointercept", "logreg", C=1.0, fit_intercept=False, _y="y_bin")
cell("logreg_l1_refused", "logreg", penalty="l1", _y="y_bin")  # REFUSED

# --- KernelDensity (kde/, DEVIATIONS 600-604; surface 2026-08-23): X is the
# 20000 x 24 matrix, queries Xq; bandwidth 8 on that scale so every kernel
# with compact support sees neighbours on most rows and none on some (the
# -3.4e38 sentinel rows, cuML's, are part of the answer). One card per
# cell through kde/estimator.mojo's trace.
KDE_BASE = dict(bandwidth=8.0)
cell("kde_gaussian", "kde", kernel="gaussian", **KDE_BASE)
cell("kde_tophat", "kde", kernel="tophat", **KDE_BASE)
cell("kde_epanechnikov", "kde", kernel="epanechnikov", **KDE_BASE)
cell("kde_exponential", "kde", kernel="exponential", **KDE_BASE)
cell("kde_linear", "kde", kernel="linear", **KDE_BASE)
cell("kde_cosine", "kde", kernel="cosine", **KDE_BASE)  # DEVIATION 602: ours, not sklearn's, at even d
cell("kde_bw2", "kde", kernel="gaussian", bandwidth=2.0)
# a compact kernel at a bandwidth most rows cannot reach: the -3.4e38
# sentinel rows (cuML's, DEVIATION 603 keeps them -inf-not-NaN) are the cell
cell("kde_tophat_bw1", "kde", kernel="tophat", bandwidth=1.0)
cell("kde_l1", "kde", kernel="gaussian", metric="l1", **KDE_BASE)
cell("kde_sqeuclidean", "kde", kernel="gaussian", metric="sqeuclidean", bandwidth=64.0)
cell("kde_weighted", "kde", kernel="gaussian", _fit=dict(sample_weight="sw"), **KDE_BASE)
cell("kde_metric_cosine_refused", "kde", kernel="gaussian", metric="cosine", **KDE_BASE)  # REFUSED (unported metric)
cell("kde_bw_scott_refused", "kde", kernel="gaussian", bandwidth="scott")  # REFUSED


# ---------------------------------------------------------------- reach
# (parameter, cell A, cell B, expectation). "differ": the parameter
# steers the answer and the hashes MUST differ. "same": the parameter is
# a memory / scheduling / arm choice and the caller-visible hashes MUST
# agree (an invariance gate), while the named side field shows the
# parameter reached the kernel.
REACH = [
    ("KernelDensity.kernel", "kde_gaussian", "kde_tophat", "differ", None),
    ("KernelDensity.bandwidth", "kde_gaussian", "kde_bw2", "differ", None),
    ("KernelDensity.metric", "kde_gaussian", "kde_l1", "differ", None),
    ("KernelDensity.sample_weight", "kde_gaussian", "kde_weighted", "differ", None),
    ("KMeans.n_clusters", "kmeans_k8", "kmeans_k3", "differ", None),
    ("KMeans.init", "kmeans_k8", "kmeans_k8_random", "differ", None),
    ("KMeans.n_init", "kmeans_k8", "kmeans_k8_ninit3", "differ", None),
    ("KMeans.tol", "kmeans_k8", "kmeans_k8_tol1e-2", "differ", "n_iter"),
    ("KMeans.max_iter", "kmeans_k8", "kmeans_k8_maxiter2", "differ",
     "n_iter"),
    ("KMeans.random_state", "kmeans_k8", "kmeans_k8_seed11", "differ",
     None),
    ("KMeans.sample_weight", "kmeans_k8", "kmeans_k8_sw", "differ", None),
    ("KMeans.init_centroids", "kmeans_k8", "kmeans_k8_array", "differ",
     None),
    ("NearestNeighbors.n_neighbors", "knn_k10", "knn_k100", "differ", None),
    ("NearestNeighbors.query_tile", "knn_k10", "knn_k10_tile64", "same",
     "used_query_tile"),
    ("NearestNeighbors.metric='l2'", "knn_k10", "knn_metric_l2", "same",
     None),
    ("KNeighborsClassifier.n_neighbors", "knn_clf_k5", "knn_clf_k15_3class",
     "differ", None),
    ("KNeighborsRegressor.n_neighbors", "knn_reg_k5", "knn_reg_k50",
     "differ", None),
    ("DBSCAN.eps", "dbscan_rbc", "dbscan_eps1.0", "differ", None),
    ("DBSCAN.min_samples", "dbscan_rbc", "dbscan_min50", "differ", None),
    ("DBSCAN.algorithm", "dbscan_rbc", "dbscan_brute", "same", None),
    ("DBSCAN.algorithm (min50)", "dbscan_min50", "dbscan_min50_brute",
     "same", None),
    ("DBSCAN.algorithm (uniform24)", "dbscan_uniform24",
     "dbscan_uniform24_brute", "same", None),
    ("DBSCAN.max_mbytes_per_batch (rbc)", "dbscan_rbc",
     "dbscan_rbc_budget1mb", "same", "batches"),
    ("DBSCAN.max_mbytes_per_batch (brute)", "dbscan_brute",
     "dbscan_brute_budget1mb", "same", "batches"),
    ("DBSCAN.max_iterations (chain, cap 200 binds)", "dbscan_chain_iter5000",
     "dbscan_chain_iter200", "differ-or-refused", "n_iter"),
    ("DBSCAN.max_iterations (chain, None = fixed point)",
     "dbscan_chain_iter5000", "dbscan_chain_default", "same", None),
    ("DBSCAN.max_iterations (blobs, converges in 1)", "dbscan_rbc",
     "dbscan_maxiter1", "same-or-refused", "n_iter"),
    ("DBSCAN.algorithm (chain)", "dbscan_chain_iter5000",
     "dbscan_chain_iter5000_brute", "same", None),
    ("PCA.n_components", "pca_c2", "pca_c8", "differ", None),
    ("TruncatedSVD.n_components", "tsvd_c2", "tsvd_c8", "differ", None),
    ("LinearRegression.fit_intercept", "ols_intercept", "ols_nointercept",
     "differ", None),
    ("Ridge.alpha", "ridge_a1", "ridge_a100", "differ", None),
    ("Ridge.fit_intercept", "ridge_a1", "ridge_nointercept", "differ", None),
    ("LogisticRegression.C", "logreg_c1", "logreg_c100", "differ", "n_iter"),
    ("LogisticRegression.fit_intercept", "logreg_c1", "logreg_nointercept",
     "differ", "n_iter"),
]

_HASH_KEYS = ("labels", "centroids", "inertia", "distances", "indices",
              "components", "explained_variance", "explained_variance_ratio",
              "singular_values", "mean", "noise_variance", "transformed",
              "coef", "intercept", "predictions", "proba", "classes",
              "log_density")


# ---------------------------------------------------------------- one cell
def _derived(data, name):
    """A few views derived from the base arrays, by name."""
    if name == "X_head":
        return np.ascontiguousarray(data["X"][:N_QUERIES])
    if name == "X_narrow4":
        return np.ascontiguousarray(data["X"][:, :4])
    if name == "X_col1":
        return np.ascontiguousarray(data["X"][:, :1])
    if name == "y_bin":
        # THE LOGISTIC TARGET: the sign of the integer matmul about its
        # median -- integers only, no float op, no new draw, so every
        # existing hash is untouched and the two classes are balanced.
        q = (data["X"] * 1024.0).astype(np.int64)
        # wq is the same integer weight vector make_inputs used; recompute
        # it from the same stream position by redrawing X, Xq, w
        r = np.random.RandomState(SEED)
        r.rand(N_ROWS, N_COLS); r.rand(N_QUERIES, N_COLS)
        wq = (r.rand(N_COLS).astype(np.float32) * 1024.0).astype(np.int64)
        qw = q @ wq
        return (qw > np.median(qw)).astype(np.int64)
    if name == "sw_blobs":
        n = data["X_blobs"].shape[0]
        return (1 + (np.arange(n, dtype=np.int64) * 7919) % 5).astype(
            np.float32)
    return data[name]


def _is_refusal(exc):
    if isinstance(exc, (NotImplementedError, ValueError)):
        return True
    return any(t in str(exc) for t in (
        "not ported", "refus", "not supported", "NOT PORTED",
        "does not support", "is refused", "cannot"))


def run_cell(name, out_dir):
    kind, spec = CELLS[name]
    spec = dict(spec)
    data = make_inputs()
    fit_kw = {k: _derived(data, v) for k, v in spec.pop("_fit", {}).items()}
    X = _derived(data, spec.pop("_X", "X_blobs" if kind == "dbscan" else "X"))
    y = _derived(data, spec.pop("_y", "y_reg"))
    Xq = _derived(data, spec.pop("_Xq", "Xq"))
    if "_init_centroids" in spec:
        spec["init_centroids"] = _derived(data, spec.pop("_init_centroids"))

    import mojolearn
    card = os.path.join(out_dir, name + ".card")
    if os.path.exists(card):
        os.remove(card)  # the trace APPENDS
    entry = {"kind": kind, "spec": {k: v for k, v in spec.items()
                                   if k != "init_centroids"}}
    if "init_centroids" in spec:
        entry["spec"]["init_centroids"] = "C0"
    t0 = time.time()
    os.environ["MOJOLEARN_IDENTITY_TRACE"] = card
    try:
        if kind == "kmeans":
            m = mojolearn.KMeans(**spec).fit(X, **fit_kw)
            entry["labels"] = sha256_of(m.labels_)
            entry["centroids"] = sha256_of(m.cluster_centers_)
            entry["inertia"] = sha256_of(np.float64(m.inertia_))
            entry["inertia_value"] = float(m.inertia_)
            entry["n_iter"] = int(m.n_iter_)
            entry["sum_scale"] = float(m.sum_scale_)
            entry["weight_scale"] = float(m.weight_scale_)
        elif kind == "knn":
            m = mojolearn.NearestNeighbors(**spec).fit(X)
            dist, ind = m.kneighbors(Xq)
            entry["distances"] = sha256_of(dist)
            entry["indices"] = sha256_of(ind)
            entry["used_query_tile"] = int(m.used_query_tile_)
        elif kind == "knn_clf":
            m = mojolearn.KNeighborsClassifier(**spec).fit(X, y)
            # predict is the TRACED call (search stages + knn_clf.uniq_labels
            # / votes / labels). predict_proba is a SECOND search and vote
            # -- cuML's predict / predict_proba are two kneighbors calls
            # too -- and a second Mojo call would append a second `seq 0`
            # into the same card, which the differ refuses (the DEVIATION
            # 518 defect); so the trace is unset for it and only its
            # output hash is recorded. `_predict` raises by name if the
            # ported class set disagrees with np.unique (estimator policy
            # 7).
            entry["predictions"] = sha256_of(m.predict(Xq))
            os.environ.pop("MOJOLEARN_IDENTITY_TRACE", None)
            entry["proba"] = sha256_of(m.predict_proba(Xq))
            entry["classes"] = sha256_of(m.classes_)
            entry["n_classes"] = int(m.classes_.shape[0])
            entry["used_query_tile"] = int(m.used_query_tile_)
        elif kind == "knn_reg":
            m = mojolearn.KNeighborsRegressor(**spec).fit(X, y)
            entry["predictions"] = sha256_of(m.predict(Xq))
            entry["used_query_tile"] = int(m.used_query_tile_)
        elif kind == "dbscan":
            m = mojolearn.DBSCAN(**spec).fit(X, **fit_kw)
            entry["labels"] = sha256_of(m.labels_)
            entry["n_clusters"] = int(m.labels_.max()) + 1
            entry["n_noise"] = int((m.labels_ == -1).sum())
            entry["n_iter"] = int(m.n_iter_)
        elif kind == "pca":
            m = mojolearn.PCA(**spec).fit(X)
            entry["components"] = sha256_of(m.components_)
            entry["explained_variance"] = sha256_of(m.explained_variance_)
            entry["explained_variance_ratio"] = sha256_of(
                m.explained_variance_ratio_)
            entry["singular_values"] = sha256_of(m.singular_values_)
            entry["mean"] = sha256_of(m.mean_)
            entry["noise_variance"] = sha256_of(np.float64(m.noise_variance_))
            entry["transformed"] = sha256_of(m.transform(X[:N_QUERIES]))
        elif kind == "tsvd":
            m = mojolearn.TruncatedSVD(**spec).fit(X)
            entry["components"] = sha256_of(m.components_)
            entry["singular_values"] = sha256_of(m.singular_values_)
            entry["transformed"] = sha256_of(m.transform(X[:N_QUERIES]))
        elif kind == "ols":
            m = mojolearn.LinearRegression(**spec).fit(X, y, **fit_kw)
            entry["coef"] = sha256_of(m.coef_)
            entry["intercept"] = sha256_of(np.float64(m.intercept_))
            entry["intercept_value"] = float(m.intercept_)
            entry["predictions"] = sha256_of(m.predict(X[:N_QUERIES]))
        elif kind == "ridge":
            m = mojolearn.Ridge(**spec).fit(X, y, **fit_kw)
            entry["coef"] = sha256_of(m.coef_)
            entry["intercept"] = sha256_of(np.float64(m.intercept_))
            entry["intercept_value"] = float(m.intercept_)
            entry["predictions"] = sha256_of(m.predict(X[:N_QUERIES]))
        elif kind == "kde":
            m = mojolearn.KernelDensity(**spec).fit(X, **fit_kw)
            ld = m.score_samples(Xq)
            entry["log_density"] = sha256_of(ld)
            entry["n_sentinel"] = int((ld <= -1e38).sum())
            entry["score"] = float(np.sum(ld[ld > -1e38], dtype=np.float64))
        elif kind == "logreg":
            m = mojolearn.LogisticRegression(**spec).fit(X, y, **fit_kw)
            entry["coef"] = sha256_of(m.coef_)
            entry["intercept"] = sha256_of(m.intercept_)
            entry["predictions"] = sha256_of(m.predict(X[:N_QUERIES]))
            entry["proba"] = sha256_of(m.predict_proba(X[:N_QUERIES]))
            entry["n_iter"] = int(m.n_iter_[0])
            entry["retcode"] = int(m.retcode_)
            entry["objective"] = float(m.objective_)
        else:
            raise AssertionError(kind)
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        if _is_refusal(exc):
            entry["refused"] = msg
        else:
            entry["error"] = msg
            entry["traceback"] = traceback.format_exc()[-2000:]
        return entry
    finally:
        os.environ.pop("MOJOLEARN_IDENTITY_TRACE", None)
        entry["seconds"] = round(time.time() - t0, 2)
    if os.path.exists(card):
        entry["card"] = os.path.basename(card)
        # the DBSCAN header carries the batch count (a device-memory
        # number, deliberately NOT a stage); lift it so the budget's reach
        # is visible without opening the card
        with open(card) as fh:
            head = fh.readline()
        if "batches=" in head:
            entry["batches"] = int(head.split("batches=")[1].split()[0])
        entry["stages"] = sum(1 for l in open(card) if "\t" in l)
    else:
        entry["card"] = None
    return entry


def _hashes(entry):
    return {k: entry[k] for k in _HASH_KEYS if k in entry}


def judge_reach(cells):
    rows = []
    bad = 0
    for param, a, b, expect, side in REACH:
        ea, eb = cells.get(a), cells.get(b)
        if ea is None or eb is None:
            rows.append((param, "MISSING", ""))
            bad += 1
            continue
        ref_a, ref_b = "refused" in ea, "refused" in eb
        if expect.endswith("-or-refused") and (ref_a or ref_b):
            rows.append((param, "REFUSED", (ea.get("refused") or
                                            eb.get("refused"))[:70]))
            continue
        if ref_a or ref_b or "error" in ea or "error" in eb:
            rows.append((param, "NOT-JUDGED", "a side refused or failed"))
            bad += 1
            continue
        ha, hb = _hashes(ea), _hashes(eb)
        common = sorted(set(ha) & set(hb))
        moved = [k for k in common if ha[k] != hb[k]]
        note = ""
        if side is not None:
            note = f"{side}: {ea.get(side)} -> {eb.get(side)}"
        if expect in ("differ", "differ-or-refused"):
            # (the refused branch of "-or-refused" returned above)
            if moved:
                rows.append((param, "REACHES", f"moved {','.join(moved)}"
                             + (f"; {note}" if note else "")))
            else:
                rows.append((param, "INERT", f"all of {','.join(common)} "
                             "equal" + (f"; {note}" if note else "")))
                bad += 1
        else:  # same
            side_moved = (side is not None and ea.get(side) != eb.get(side))
            if not moved and (side is None or side_moved):
                rows.append((param, "INVARIANT", note or
                             f"{','.join(common)} equal"))
            elif not moved:
                rows.append((param, "INERT", f"{note} (side field did not "
                             "move: the parameter may not have reached the "
                             "kernel)"))
                bad += 1
            else:
                rows.append((param, "BROKEN-INVARIANCE",
                             f"moved {','.join(moved)}; {note}"))
                bad += 1
    return rows, bad


def _numeric_mode():
    import mojolearn
    return mojolearn.numeric_mode()



def _commit(repo):
    """The commit, or the archive's own identity when there is no `.git`.

    DEVIATION 1935, 2026-08-28. This was a bare
    `subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo)`, and on
    a box that received the tree as a `git archive` -- which is exactly what
    `tools/gemm_remote_leg.sh` ships, because the repo is private and the box
    has no credentials -- it raises CalledProcessError 128 and takes the whole
    driver down with it.

    That is not a hypothetical. The NVIDIA column of the 2026-08-28 round
    (`cc499f7`) built all ten identical bindings and then lost phase 3's
    traced fits, phase 4's E2 tree matrix and phase 7's E2U matrix to this
    line, so the round had an Apple column, an AMD column, and an NVIDIA
    column with no matrix in it -- for a provenance string, not for anything
    numeric. The DigitalOcean leg never hit it because it ships a `git
    bundle` and clones it, so its boxes do have a `.git`.

    The fallbacks are the same evidence the leg already writes, in the order
    they are trustworthy: the bootstrap's own `commit.txt`, then
    MOJOLEARN_COMMIT from the environment, then the honest string "unknown".
    A provenance field must never be able to end a measurement.
    """
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        pass
    for cand in (os.path.join(repo, "commit.txt"),):
        try:
            with open(cand) as fh:
                v = fh.read().strip()
            if v:
                return v
        except Exception:
            pass
    return os.environ.get("MOJOLEARN_COMMIT", "unknown")

def main():
    argv = sys.argv[1:]
    if argv and argv[0] == "--cell":
        name, out_dir = argv[1], os.path.abspath(argv[2])
        entry = run_cell(name, out_dir)
        with open(os.path.join(out_dir, name + ".cell.json"), "w") as fh:
            json.dump(entry, fh, indent=2, sort_keys=True)
        return 0

    out_dir = os.path.abspath(argv[0])
    only = None
    if "--only" in argv:
        only = argv[argv.index("--only") + 1]
    os.makedirs(out_dir, exist_ok=True)
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data = make_inputs()
    record = {
        "driver": "e2u",
        "commit": _commit(repo),
        "numeric_mode": _numeric_mode(),
        "platform": platform.platform(),
        "machine": platform.node(),
        "inputs": {k: sha256_of(v) for k, v in data.items()},
        "cells": {},
    }
    names = [n for n in CELLS if only is None or only in n]
    print(f"E2U matrix ({record['numeric_mode']}): {len(names)} cells -> "
          f"{out_dir}")
    for i, name in enumerate(names, 1):
        cell_json = os.path.join(out_dir, name + ".cell.json")
        if os.path.exists(cell_json):
            os.remove(cell_json)
        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--cell", name,
             out_dir], cwd=repo, capture_output=True, text=True)
        dt = round(time.time() - t0, 1)
        if os.path.exists(cell_json):
            with open(cell_json) as fh:
                entry = json.load(fh)
        else:
            entry = {"crashed": proc.returncode,
                     "stderr_tail": proc.stderr[-1500:]}
        record["cells"][name] = entry
        if "refused" in entry:
            verdict = "REFUSED " + entry["refused"][:40]
        elif "error" in entry:
            verdict = "ERROR " + entry["error"][:40]
        elif "crashed" in entry:
            verdict = "CRASHED"
        else:
            first = next((entry[k][:16] for k in _HASH_KEYS if k in entry),
                         "?")
            extra = " ".join(f"{k}={entry[k]}" for k in
                             ("n_iter", "n_clusters", "n_noise", "batches",
                              "used_query_tile", "stages") if k in entry)
            verdict = f"{first} {extra}"
        print(f"[{i:2d}/{len(names)}] {name:28s} {verdict:60s} {dt}s",
              flush=True)
        if "stderr_tail" in entry:
            print(entry["stderr_tail"][-600:])
        with open(os.path.join(out_dir, "e2u_cells.json"), "w") as fh:
            json.dump(record, fh, indent=2, sort_keys=True)
    rows, bad = judge_reach(record["cells"])
    record["reach"] = [dict(parameter=p, verdict=v, note=n)
                       for p, v, n in rows]
    with open(os.path.join(out_dir, "e2u_cells.json"), "w") as fh:
        json.dump(record, fh, indent=2, sort_keys=True)
    print("\nREACH")
    for p, v, n in rows:
        print(f"  {p:44s} {v:18s} {n}")
    print("wrote", os.path.join(out_dir, "e2u_cells.json"))
    if only is None and bad:
        print(f"{bad} reach row(s) are INERT / broken / not judged")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
