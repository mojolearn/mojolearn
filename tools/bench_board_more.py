#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The `classical2` family of tools/bench_board.py: the wheel's remaining
classical estimators against scikit-learn, umap-learn, statsmodels, FAISS,
cuML and cuVS on the same box, interleaved round by round, quality beside
every time.

    python3 tools/bench_board_more.py prep --data DIR --lanes umap,gmm --datasets taxi
    python3 tools/bench_board_more.py race --lane umap --dataset taxi --data DIR \\
        --arms ours,ours-fast,umap-learn-cpu --rounds 5 --out DIR --work DIR \\
        --ours-python PY --theirs-python PY

It speaks tools/classical_two_datasets.py's protocol and reuses its `Worker`
and its block helpers (stride sampling, the Istella sentinel clean, the fit
rows' standardization): one persistent worker process per arm, one warm-up
round then `--rounds` timed rounds, the arm order rotated every round, and
one race JSON whose shape tools/bench_board.py's `classical_cells` reads.

THE ARMS
--------
  ours          the wheel's public estimator, MOJOLEARN_NUMERIC_MODE=identical,
                read back from the binary it holds (`numeric_mode_used()`, or
                the lane's binding's `<prefix>_numeric_mode()` for the classes
                without the mixin). The input is a host float32 array and the
                upload is inside the clock.
  ours-fast     the same estimator under MOJOLEARN_NUMERIC_MODE=fast (Apple);
                a readback other than `fast` refuses the arm by name.
  sklearn-cpu   scikit-learn on every core as installed (no thread caps,
                n_jobs=-1 where the estimator takes one).
  umap-learn-cpu           umap-learn with random_state=7: umap-learn itself
                           then runs ONE thread (its rule for a seeded fit).
  umap-learn-cpu-unseeded  umap-learn with random_state=None and n_jobs=-1,
                           every core; the one setting that differs is named.
  statsmodels-cpu  statsmodels, one fit per series, the series spread over
                every core with joblib (loky, n_jobs=-1).
  faiss-cpu     faiss IndexIVFFlat on every OpenMP thread.
  cuml-gpu      RAPIDS cuML, inputs uploaded to the device before the clock
                (their fast arm, as tools/classical_two_datasets.py does).
  cuvs-gpu      cuVS ivf_flat from the pinned rapids set, the same way.

Every lane's settings, and every mismatch that cannot be avoided with its
one-line reason, are in LANE_CONFIG below; tools/bench_board.py copies them
into every cell's `settings`.

THE DATA (prep, untimed, once per box and row cap)
--------------------------------------------------
taxi and Istella-S through tools/speed_gbdt_arm.py's loaders (the trees
harness's caches, staged from R2; nothing is downloaded), the Istella
missing sentinel cleaned to 0.0, stride samples of the train and test
splits. Supervised and kernel blocks are STANDARDIZED by the fit rows (a
Gaussian kernel or a penalized linear model on raw Istella columns spanning
seven orders of magnitude is not a problem any user fits). The blocks:

    manifold  X 20,000 stride rows                       umap, spectral-embedding
    cls       X 1,000,000 + Xq 100,000, y binary          logreg, linearsvc, knn-clf,
                                                          spectral, agglomerative, gpc
    reg       X 1,000,000 + Xq 100,000, y real            ridge, lasso, elasticnet,
                                                          linearsvr, knn-reg, gmm, gpr,
                                                          svr, kernel-ridge, nystroem,
                                                          rbf-sampler
    tsvd      X 1,000,000 raw (sentinel cleaned only)     tsvd
    ivf       tools/knn_datasets.real_block (the knn lane's block, raw)   ivf
    arma      64 synthetic ARMA(1,1) series, 2,100 points (seed 7)        arima
    seasonal  64 synthetic hourly series, period 24, 1,488 points (seed 7) ets

Lanes that are O(n^2) or O(n^3) take a DOCUMENTED stride subset of their
block, the SAME rows for every arm (`lane_arrays`): knn-* 200,000 fit rows
and 4,000 queries; spectral and agglomerative 10,000 rows; gmm 100,000 fit
rows and 20,000 held-out rows; gpr and gpc 3,000 fit and 3,000 held-out rows;
svr and kernel-ridge 10,000 and 10,000; nystroem and rbf-sampler 100,000 rows
transformed, the kernel check on 1,000 of them.

THE TIME SERIES ARE SYNTHETIC. The repo's own ARIMA quality work
(bench/results/fast_quality_audit_2026-09-26/arima_quality.py) fits seeded
synthetic ARMA series, and no ARIMA or Holt-Winters bench reads taxi or
Istella, so these two lanes do the same: `arma` is 64 ARMA(1,1) series (phi
0.6, theta 0.3, 200 burn-in points dropped) and `seasonal` is 64 series of
level + trend + a period-24 sine + noise, every one from default_rng(7). The
last 100 (arma) and 48 (seasonal) points of each series are held out for
the forecast error.

QUALITY (the conductor, float64 NumPy, from each arm's saved outputs; never
the libraries' own scorers)
-------------------------------------------------------------------------
  umap, spectral-embedding  trustworthiness at k = 15 over every embedded row
                (sklearn.manifold.trustworthiness's formula, computed in row
                chunks so 20,000 rows fit in memory; equal to scikit-learn's
                on small data in tools/test_bench_board_more.py)
  gmm           held-out mean log-likelihood and BIC on the fit rows, both
                from each arm's weights, means and covariances
  logreg, gpc   held-out accuracy and log loss
  linearsvc, knn-clf  held-out accuracy
  ridge, lasso, elasticnet, linearsvr, knn-reg, svr, kernel-ridge
                held-out R2 and RMSE
  gpr           held-out RMSE, R2 and mean log predictive density
  tsvd          explained-variance ratio sum (scikit-learn's TruncatedSVD
                definition) and relative reconstruction error
  spectral, agglomerative  cluster count, silhouette on every clustered row,
                ARI against ours
  nystroem, rbf-sampler  relative Frobenius error of Z Z^T against the exact
                RBF kernel on the check rows
  arima         mean log-likelihood, mean AIC (2N - 2 llf with the same N on
                every arm), forecast RMSE over the held-out points, in-sample
                one-step RMSE
  ets           forecast RMSE over the held-out points, in-sample one-step RMSE
                (times t >= 2 * period, where every arm has a prediction)
  ivf           recall@10 against a float64 NumPy brute force (the classical
                knn lane's function)
"""
import argparse
import hashlib
import importlib.util
import json
import os
import shlex
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

SEED = 7
FAMILY = "classical2"

# ---------------------------------------------------------------------------
# Shapes (the full board). `--max-rows` caps every row count for a smoke.
# ---------------------------------------------------------------------------
MANIFOLD_ROWS = 20_000
LIN_ROWS = 1_000_000
EVAL_ROWS = 100_000
TSVD_ROWS = 1_000_000
KNN_FIT = 200_000
KNN_QUERIES = 4_000
CLUSTER_ROWS = 10_000
GMM_FIT = 100_000
GMM_EVAL = 20_000
GP_FIT = 3_000
GP_EVAL = 3_000
#: The GP ridge: the value the IDENTICAL profile pins (mojolearn.identical.gp.fp32.v1
#: refuses any other alpha), used on every arm.
GP_ALPHA = 2.0 ** -20
#: The regressor's observation noise, carried by a WhiteKernel on every arm.
GP_NOISE = 1e-2
KERNEL_FIT = 10_000
KERNEL_EVAL = 10_000
KAPPROX_ROWS = 100_000
KAPPROX_CHECK = 1_000
KAPPROX_COMPONENTS = 256
IVF_INDEX = 400_000
IVF_QUERIES = 4_000
TS_SERIES = 64
ARMA_FIT, ARMA_H = 2_000, 100
SEASON_PERIOD = 24
SEASON_FIT, SEASON_H = SEASON_PERIOD * 60, 48

K_TRUST = 15
N_CLUSTERS = 8
KNN_K = 10
#: cuML benchmark tSVD shared_args n_components=10 (tools/bench_board_harness.py)
TSVD_COMPONENTS = 10
#: cuML benchmark SpectralClustering shared_args: n_init=1, random_state=42
#: (the board's seed for this lane, bench_board_params.LANE_SEED)
SPECTRAL_N_INIT = 1
SPECTRAL_SEED = 42
#: cuML benchmark ElasticNet shared_args alpha=0.1, l1_ratio=0.5
ENET_ALPHA = 0.1
#: cuML benchmark UMAP shared_args n_neighbors=5, n_epochs=500
UMAP_NEIGHBORS = 5
UMAP_EPOCHS = 500
GMM_COMPONENTS = 8
IVF_NLIST, IVF_NPROBE, IVF_K = 1024, 32, 10
ARIMA_ORDER = (1, 0, 1)
ARIMA_SEASONAL = (0, 0, 0, 0)
#: Holt-Winters convergence tolerance: ours' and cuML's parameter (their
#: default, set explicitly); statsmodels has none.
ETS_EPS = 2.24e-3
#: The SVR kernel cache on every arm that takes one (scikit-learn's value
#: here since the lane began; ours honors it at predict only, DEVIATION 871).
SVR_CACHE_MB = 2000.0

#: lane -> (block, the binding its estimator answers from, datasets)
LANES = {
    "umap": ("manifold", "_mojolearn_metrics", ("taxi", "istella")),
    "spectral-embedding": ("manifold", "_mojolearn_metrics", ("taxi", "istella")),
    "gmm": ("reg", "_mojolearn_mixture", ("taxi", "istella")),
    "logreg": ("cls", "_mojolearn_estimators", ("taxi", "istella")),
    "linearsvc": ("cls", "_mojolearn_estimators", ("taxi", "istella")),
    "ridge": ("reg", "_mojolearn_estimators", ("taxi", "istella")),
    "lasso": ("reg", "_mojolearn_solver", ("taxi", "istella")),
    "elasticnet": ("reg", "_mojolearn_solver", ("taxi", "istella")),
    "linearsvr": ("reg", "_mojolearn_estimators", ("taxi", "istella")),
    "tsvd": ("tsvd", "_mojolearn_estimators", ("taxi", "istella")),
    "knn-clf": ("cls", "_mojolearn", ("taxi", "istella")),
    "knn-reg": ("reg", "_mojolearn", ("taxi", "istella")),
    "spectral": ("cls", "_mojolearn_metrics", ("taxi", "istella")),
    "agglomerative": ("cls", "_mojolearn_estimators", ("taxi", "istella")),
    "gpr": ("reg", "_mojolearn_gp", ("taxi", "istella")),
    "gpc": ("cls", "_mojolearn_gp", ("taxi", "istella")),
    "svr": ("reg", "_mojolearn_svm", ("taxi", "istella")),
    "kernel-ridge": ("reg", "_mojolearn_kernel_methods", ("taxi", "istella")),
    "nystroem": ("reg", "_mojolearn_kernel_methods", ("taxi", "istella")),
    "rbf-sampler": ("reg", "_mojolearn_kernel_methods", ("taxi", "istella")),
    "arima": ("arma", "_mojolearn_arima", ("synthetic",)),
    "ets": ("seasonal", "_mojolearn_tsa", ("synthetic",)),
    "ivf": ("ivf", "_mojolearn_ivf", ("taxi", "istella")),
}
LANE_ORDER = tuple(LANES)

#: The bindings the 0.8.22 wheel ships a FAST tier for
#: (mojolearn._backend._CLASSICAL_FAST, read from the wheel). Every lane above
#: answers from one of them, so every lane races FAST beside IDENTICAL on
#: Apple; an `ours-fast` worker whose readback is not `fast` refuses by name.
FAST_BINDINGS = frozenset({
    "_mojolearn", "_mojolearn_estimators", "_mojolearn_svm", "_mojolearn_solver",
    "_mojolearn_metrics", "_mojolearn_preprocessing", "_mojolearn_tsa",
    "_mojolearn_linalg", "_mojolearn_arima", "_mojolearn_gp",
    "_mojolearn_kernel_methods", "_mojolearn_mixture", "_mojolearn_hdbscan",
    "_mojolearn_resample", "_mojolearn_ivf"})


def has_fast(lane):
    return LANES[lane][1] in FAST_BINDINGS


def block_of(lane):
    return LANES[lane][0]


def datasets_of(lane):
    return LANES[lane][2]


#: Opponent arms per vendor and lane. Apple and AMD: the CPU libraries on every
#: core (none of them has an Apple or AMD GPU path). NVIDIA: cuML or cuVS where
#: they carry the algorithm, scikit-learn or statsmodels on the CPU where they
#: do not (NOT_PLANNED says which and why).
_CPU = {
    "umap": ("umap-learn-cpu", "umap-learn-cpu-unseeded"),
    "arima": ("statsmodels-cpu",), "ets": ("statsmodels-cpu",),
    "ivf": ("faiss-cpu",),
}
OPPONENTS = {
    "apple": {lane: _CPU.get(lane, ("sklearn-cpu",)) for lane in LANE_ORDER},
    "amd": {lane: _CPU.get(lane, ("sklearn-cpu",)) for lane in LANE_ORDER},
    "nvidia": {
        "umap": ("cuml-gpu",),
        "spectral-embedding": ("cuml-gpu", "sklearn-cpu"),
        "gmm": ("sklearn-cpu",),
        "logreg": ("cuml-gpu",), "linearsvc": ("cuml-gpu",), "ridge": ("cuml-gpu",),
        "lasso": ("cuml-gpu",), "elasticnet": ("cuml-gpu",), "linearsvr": ("cuml-gpu",),
        "tsvd": ("cuml-gpu",), "knn-clf": ("cuml-gpu",), "knn-reg": ("cuml-gpu",),
        "spectral": ("cuml-gpu", "sklearn-cpu"),
        "agglomerative": ("cuml-gpu",),
        "gpr": ("sklearn-cpu",), "gpc": ("sklearn-cpu",),
        "svr": ("cuml-gpu",), "kernel-ridge": ("cuml-gpu",),
        "nystroem": ("sklearn-cpu",), "rbf-sampler": ("sklearn-cpu",),
        "arima": ("cuml-gpu", "statsmodels-cpu"), "ets": ("cuml-gpu", "statsmodels-cpu"),
        "ivf": ("cuvs-gpu",),
    },
}

#: Opponents that are NOT on the board, by name, with the reason. The board
#: prints this list; nothing is dropped silently.
NOT_PLANNED = {
    "apple": [
        "cuML and cuVS: CUDA only; no Apple build exists.",
        "faiss-gpu: CUDA only; faiss-cpu is the arm on this box.",
    ],
    "amd": [
        "cuML and cuVS: CUDA only; no ROCm build is pinned.",
        "faiss-gpu: the pinned FAISS GPU builds are CUDA; faiss-cpu is the arm on this box.",
    ],
    "nvidia": [
        "cuML GaussianMixture, GaussianProcessRegressor/Classifier, Nystroem, RBFSampler: "
        "cuML 26.8.0 has none; scikit-learn on the CPU is the arm.",
        "faiss-gpu: no pinned PyPI wheel for this image; cuVS ivf_flat from the pinned "
        "rapids set (cuvs-cu12==26.8.1) is the IVF-Flat GPU arm.",
        "umap-learn and faiss-cpu: not installed on NVIDIA; cuML UMAP and cuVS are the arms.",
        "cuML SpectralClustering/SpectralEmbedding: present only in newer cuML; if the "
        "pinned 26.8.0 lacks them the arm refuses by name and scikit-learn stands beside it.",
    ],
}

#: What every arm of a lane is set to, and every mismatch that could not be
#: avoided (one line each). tools/bench_board.py copies this into each cell.
_STD = "standardized by the fit rows"
LANE_CONFIG = {
    "umap": {
        "rows": "%d stride rows of the train split, %s" % (MANIFOLD_ROWS, _STD),
        "params": "n_neighbors=5, n_epochs=500 (the cuML benchmark's UMAP), n_components=2, "
                  "min_dist=0.1, spread=1.0, "
                  "metric='euclidean', init='spectral', learning_rate=1.0, repulsion_strength=1.0, "
                  "negative_sample_rate=5, set_op_mix_ratio=1.0, local_connectivity=1.0, random_state=7",
        "timed": "fit (ours fit_transform) from host rows to the embedding",
        "quality": "trustworthiness k=15 over every row (sklearn's formula, chunked)",
        "mismatches": [
            "neighbors: ours exact brute force; umap-learn NN-descent (its choice above 4,096 rows); "
            "cuML build_algo='brute_force_knn' (exact)",
            "umap-learn-cpu: random_state=7 makes umap-learn run one thread (its rule)",
            "umap-learn-cpu-unseeded: random_state=None and n_jobs=-1, the every-core setting; "
            "the seed is the one parameter that differs",
            "spectral init: each library's own eigensolver and tolerance (ours: Lanczos, at most "
            "20 basis vectors)",
        ],
    },
    "spectral-embedding": {
        "rows": "the umap block (%d stride rows, %s)" % (MANIFOLD_ROWS, _STD),
        "params": "n_components=2, affinity='nearest_neighbors', n_neighbors=10, random_state=7",
        "timed": "fit_transform",
        "quality": "trustworthiness k=15 over every row",
        "mismatches": ["eigensolver: ours Lanczos (default tolerance); scikit-learn arpack "
                       "(its default); cuML its own"],
    },
    "gmm": {
        "rows": ("%d fit and %d held-out stride rows of the reg block (%s); "
                 "an explicitly full_dataset_coverage recipe retains all fit/eval rows")
                % (GMM_FIT, GMM_EVAL, _STD),
        "opponent_precision": "scikit-learn receives the same values promoted to float64 inside the fit clock; original float32 covariance failure retained",
        "params": "n_components=8, covariance_type='full', tol=1e-3, reg_covar=1e-6 on taxi and "
                  "3e-3 on Istella-S (GMM_REG_COVAR), max_iter=100, "
                  "init_params='kmeans', n_init=1, warm_start=False, random_state=7",
        "timed": "fit",
        "quality": "held-out mean log-likelihood, BIC on the fit rows (from each arm's parameters)",
        "mismatches": ["init_params='kmeans': each library seeds its own k-means (ours the "
                       "identity-certified k-means, scikit-learn KMeans(n_init=1, k-means++))"],
        "reg_covar": "Istella-S takes 3e-3 on every arm: on its 100,000 x 200 float32 fit rows "
                     "scikit-learn's GaussianMixture refuses at 1e-3 (ill-defined empirical "
                     "covariance) and both arms fit at 3e-3 (m2pro, 2026-09-29, "
                     "tools/gmm_istella_probe.py); taxi keeps 1e-6, where both arms fit",
        "data": "constant columns dropped: the columns constant on the fit rows (Istella-S: 20 of 220 on the 100,000 fit rows) are removed from X and Xq before the clock, the same for every arm; a full covariance over them is singular, and on the raw float32 rows ours, scikit-learn float32 and both Bayesian mixtures refused (ill-defined empirical covariance) where only scikit-learn float64 fitted (m3ultra-b, 2026-09-29, tools/gmm_istella_probe.py)",
    },
    "logreg": {
        "rows": "%d fit and %d held-out stride rows (%s)" % (LIN_ROWS, EVAL_ROWS, _STD),
        "params": "penalty='l2', C=1.0, tol=1e-4, max_iter=1000, fit_intercept=True, "
                  "class_weight=None; scikit-learn random_state=7",
        "timed": "fit",
        "quality": "held-out accuracy and log loss",
        "mismatches": ["solver: ours 'qn' (L-BFGS, cuML's), scikit-learn 'lbfgs', cuML 'qn'; "
                       "each library's own stopping rule reads tol",
                       "seed: ours and cuML LogisticRegression have no seed argument",
                       "l1_ratio: ours None, scikit-learn None or 0.0 (its l2 spelling from 1.8)"],
    },
    "linearsvc": {
        "rows": "%d fit and %d held-out stride rows (%s)" % (LIN_ROWS, EVAL_ROWS, _STD),
        "params": "penalty='l2', loss='squared_hinge', C=1.0, tol=1e-4, max_iter=1000, fit_intercept=True, "
                  "class_weight=None; ours and cuML penalized_intercept=False; scikit-learn "
                  "intercept_scaling=1.0, random_state=7",
        "timed": "fit",
        "quality": "held-out accuracy",
        "mismatches": ["solver: ours and cuML L-BFGS on the primal with an unpenalized intercept; "
                       "scikit-learn liblinear (dual='auto'), which penalizes the intercept",
                       "seed: ours and cuML LinearSVC have no seed argument"],
    },
    "ridge": {
        "rows": "%d fit and %d held-out stride rows (%s)" % (LIN_ROWS, EVAL_ROWS, _STD),
        "params": "alpha=1.0, fit_intercept=True; scikit-learn positive=False, random_state=7",
        "timed": "fit", "quality": "held-out R2 and RMSE",
        "mismatches": ["solver, named on every arm: ours and cuML 'eig' (eigendecomposition of "
                       "the normal equations), scikit-learn 'cholesky' (it has no 'eig')",
                       "seed: ours and cuML Ridge have no seed argument"],
    },
    "lasso": {
        "rows": "%d fit and %d held-out stride rows (%s)" % (LIN_ROWS, EVAL_ROWS, _STD),
        "params": "alpha=0.01, fit_intercept=True, max_iter=1000, tol=1e-4, selection='cyclic', "
                  "precompute=False, positive=False; ours and cuML solver='cd'",
        "timed": "fit", "quality": "held-out R2 and RMSE",
        "mismatches": ["seed: ours refuses random_state (it selects nothing with "
                       "selection='cyclic'), cuML has none; scikit-learn random_state=7",
                       "tol: each library's own stopping rule reads it"],
    },
    "elasticnet": {
        "rows": "%d fit and %d held-out stride rows (%s)" % (LIN_ROWS, EVAL_ROWS, _STD),
        "params": "alpha=0.1, l1_ratio=0.5 (the cuML benchmark's ElasticNet), fit_intercept=True, "
                  "max_iter=1000, tol=1e-4, selection='cyclic', precompute=False, positive=False; "
                  "ours and cuML solver='cd'",
        "timed": "fit", "quality": "held-out R2 and RMSE",
        "mismatches": ["seed: ours refuses random_state (it selects nothing with "
                       "selection='cyclic'), cuML has none; scikit-learn random_state=7",
                       "tol: each library's own stopping rule reads it"],
    },
    "linearsvr": {
        "rows": "%d fit and %d held-out stride rows (%s)" % (LIN_ROWS, EVAL_ROWS, _STD),
        "params": "penalty='l2', loss='epsilon_insensitive', epsilon=0.0, C=1.0, tol=1e-4, "
                  "max_iter=1000, fit_intercept=True",
        "timed": "fit", "quality": "held-out R2 and RMSE",
        "mismatches": ["penalty='l2' set on ours and cuML (their default is 'l1'); scikit-learn "
                       "has only l2", "solver: ours and cuML L-BFGS on the primal; scikit-learn "
                       "liblinear dual coordinate descent (dual=True, the only form for this loss)",
                       "intercept: ours and cuML penalized_intercept=False; scikit-learn "
                       "intercept_scaling=1.0 (liblinear penalizes it)",
                       "seed: ours and cuML LinearSVR have no seed argument; scikit-learn 7"],
    },
    "tsvd": {
        "rows": "%d stride rows of the train split, raw (sentinel cleaned, not scaled)" % TSVD_ROWS,
        "params": "n_components=10 (the cuML benchmark's tSVD), tol=0.0, n_iter=5, "
                  "n_oversamples=10, random_state=7",
        "timed": "fit",
        "quality": "explained-variance ratio sum (TruncatedSVD's definition), relative "
                   "reconstruction error",
        "mismatches": ["algorithm: ours 'covariance_eigh' (eigh of X^T X), scikit-learn "
                       "'arpack' (tol=0, exact to ARPACK's tolerance), cuML 'full'"],
    },
    "knn-clf": {
        "rows": "%d fit rows, %d queries (stride subsets of the cls block, %s)" % (KNN_FIT, KNN_QUERIES, _STD),
        "params": "n_neighbors=10, weights='uniform', metric='euclidean', algorithm='brute'",
        "timed": "fit + predict of the queries", "quality": "held-out accuracy",
        "mismatches": ["seed: no arm has a seed argument (exact search)"],
    },
    "knn-reg": {
        "rows": "%d fit rows, %d queries (stride subsets of the reg block, %s)" % (KNN_FIT, KNN_QUERIES, _STD),
        "params": "n_neighbors=10, weights='uniform', metric='euclidean', algorithm='brute'",
        "timed": "fit + predict of the queries", "quality": "held-out R2 and RMSE",
        "mismatches": ["seed: no arm has a seed argument (exact search)"],
    },
    "spectral": {
        "rows": "%d stride rows of the cls block (%s); O(n^2) affinity" % (CLUSTER_ROWS, _STD),
        "params": "n_clusters=8, affinity='nearest_neighbors', n_neighbors=10, n_init=1, "
                  "random_state=42 (the cuML benchmark's SpectralClustering), "
                  "assign_labels='kmeans', n_components=8",
        "timed": "fit", "quality": "cluster count, silhouette, ARI vs ours",
        "mismatches": ["eigensolver: ours Lanczos eigen_tol 1e-5 (its default); scikit-learn "
                       "arpack eigen_tol='auto'; each library's own k-means on the embedding",
                       "gamma, degree, coef0: not read by the nearest_neighbors affinity; ours "
                       "refuses any value (None), scikit-learn holds 1.0, 3, 1"],
    },
    "agglomerative": {
        "rows": "%d stride rows of the cls block (%s); O(n^2)" % (CLUSTER_ROWS, _STD),
        "params": "n_clusters=8, linkage='single', metric='euclidean' (ours and cuML "
                  "connectivity='pairwise')",
        "timed": "fit", "quality": "cluster count, silhouette, ARI vs ours",
        "mismatches": ["single linkage only: ours and cuML implement no other linkage",
                       "seed: no arm has a seed argument (deterministic)"],
    },
    "gpr": {
        "rows": "%d fit and %d held-out stride rows of the reg block (%s); O(n^3)" % (GP_FIT, GP_EVAL, _STD),
        "params": "kernel=ConstantKernel(1.0) * RBF(length_scale=sqrt(d)) + WhiteKernel(noise_level=%g), "
                  "alpha=2**-20 (the one ridge IDENTICAL accepts beside 0; the noise lives in the "
                  "WhiteKernel so the float32 factor of K exists on taxi's near-duplicate rows), "
                  "optimizer=None, normalize_y=False, n_restarts_optimizer=0, random_state=7" % 1e-2,
        "timed": "fit",
        "quality": "held-out RMSE, R2, mean log predictive density (variance std^2 + alpha)",
        "mismatches": ["kernel: the same kernel built from each library's own classes"],
    },
    "gpc": {
        "rows": "%d fit and %d held-out stride rows of the cls block (%s); O(n^3)" % (GP_FIT, GP_EVAL, _STD),
        "params": "kernel=ConstantKernel(1.0) * RBF(length_scale=sqrt(d)), optimizer=None, "
                  "max_iter_predict=100, n_restarts_optimizer=0",
        "timed": "fit", "quality": "held-out accuracy and log loss",
        "mismatches": ["seed: ours refuses random_state (optimizer=None draws nothing); "
                       "scikit-learn random_state=7"],
    },
    "svr": {
        "rows": "%d fit and %d held-out stride rows of the reg block (%s)" % (KERNEL_FIT, KERNEL_EVAL, _STD),
        "params": "kernel='rbf', gamma=1/d, C=1.0, epsilon=0.1, tol=1e-3, degree=3, coef0=0.0, "
                  "max_iter=-1, cache_size=2000 MB",
        "timed": "fit", "quality": "held-out R2 and RMSE",
        "mismatches": ["cache_size=2000 on every arm; ours honors it at predict only (DEVIATION "
                       "871); libsvm is single-threaded; shrinking=True is scikit-learn's only",
                       "seed: no arm has a seed argument"],
    },
    "kernel-ridge": {
        "rows": "%d fit and %d held-out stride rows of the reg block (%s)" % (KERNEL_FIT, KERNEL_EVAL, _STD),
        "params": "alpha=1.0, kernel='rbf', gamma=1/d, degree=3, coef0=1.0",
        "timed": "fit", "quality": "held-out R2 and RMSE",
        "mismatches": ["seed: no arm has a seed argument (closed-form fit)"],
    },
    "nystroem": {
        "rows": "%d stride rows of the reg block (%s); kernel check on the first %d" % (
            KAPPROX_ROWS, _STD, KAPPROX_CHECK),
        "params": "kernel='rbf', gamma=1/d, n_components=%d, random_state=7" % KAPPROX_COMPONENTS,
        "timed": "fit_transform of every row",
        "quality": "relative Frobenius error of Z Z^T against the exact kernel on the check rows",
        "mismatches": ["the landmark rows are each library's own random draw from seed 7",
                       "degree=3, coef0=1.0 on every arm; the rbf kernel reads neither"],
    },
    "rbf-sampler": {
        "rows": "%d stride rows of the reg block (%s); kernel check on the first %d" % (
            KAPPROX_ROWS, _STD, KAPPROX_CHECK),
        "params": "gamma=1/d, n_components=%d, random_state=7" % KAPPROX_COMPONENTS,
        "timed": "fit_transform of every row",
        "quality": "relative Frobenius error of Z Z^T against the exact kernel on the check rows",
        "mismatches": ["the random Fourier features are each library's own draw from seed 7"],
    },
    "arima": {
        "rows": "%d synthetic ARMA(1,1) series, %d fit points, %d held out" % (TS_SERIES, ARMA_FIT, ARMA_H),
        "params": "order=(1,0,1), seasonal_order=(0,0,0,0), trend='c' (cuML fit_intercept=True), "
                  "maxiter=1000, maximum likelihood",
        "timed": "fit of every series",
        "quality": "mean llf, mean AIC (2N - 2 llf, N=4 on every arm), forecast RMSE, in-sample RMSE",
        "mismatches": ["ours and cuML fit the whole batch in one call; statsmodels fits one "
                       "series per call (the state-space model, L-BFGS), spread over every core with joblib",
                       "statsmodels enforce_stationarity and enforce_invertibility at its default "
                       "(True); ours and cuML have no such parameter",
                       "seed: no arm has a seed argument (maximum likelihood)"],
    },
    "ets": {
        "rows": "%d synthetic hourly series, period %d, %d fit points, %d held out" % (
            TS_SERIES, SEASON_PERIOD, SEASON_FIT, SEASON_H),
        "params": "trend additive, seasonal additive, seasonal_periods=24, "
                  "initialization_method='estimated'; ours and cuML start_periods=2, eps=2.24e-3; "
                  "statsmodels damped_trend=False, use_boxcox=False",
        "timed": "construct + fit of every series",
        "quality": "forecast RMSE, in-sample one-step RMSE (t >= 48)",
        "mismatches": ["initialization: ours 'estimated' (its default, statsmodels' definition), "
                       "statsmodels 'estimated'; cuML has only its heuristic start "
                       "(start_periods=2), so its row fits the older initialization",
                       "cuML returns no in-sample predictions; that quality cell is empty",
                       "trend: ours and cuML are additive-trend with no parameter; statsmodels "
                       "trend='additive'. eps is ours' and cuML's only; statsmodels fit() uses "
                       "its own optimizer",
                       "seed: no arm has a seed argument"],
    },
    "ivf": {
        "rows": "the classical knn lane's block (tools/knn_datasets.real_block): %d index rows, "
                "%d queries, raw" % (IVF_INDEX, IVF_QUERIES),
        "params": "IVF-Flat, n_lists=%d, n_probes=%d, k=%d, squared L2, k-means 20 iterations, "
                  "seed 7" % (IVF_NLIST, IVF_NPROBE, IVF_K),
        "timed": "build + search of every query",
        "quality": "recall@10 against a float64 NumPy brute force",
        "mismatches": ["quantizer training set: each library's own (FAISS subsamples to 256 rows "
                       "per list; cuVS kmeans_trainset_fraction 0.5; ours its own)",
                       "seed: ours random_state=7, faiss cp.seed=7; cuVS IndexParams takes none"],
    },
}


def now_utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _load(name):
    path = os.path.join(HERE, name + ".py")
    spec = importlib.util.spec_from_file_location("bbm_" + name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cap(n, cap, floor=0):
    return n if not cap else max(min(floor, n), min(n, cap))


# ---------------------------------------------------------------------------
# prep: the blocks, written once
# ---------------------------------------------------------------------------

def arma_series(n_series, n_obs, seed=SEED, phi=0.6, theta=0.3, burn=200):
    import numpy as np
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((n_series, n_obs + burn))
    x = np.zeros_like(e)
    for t in range(n_obs + burn):
        x[:, t] = e[:, t]
        if t >= 1:
            x[:, t] += phi * x[:, t - 1] + theta * e[:, t - 1]
    level = rng.uniform(-2.0, 2.0, size=(n_series, 1))
    return np.ascontiguousarray((x[:, burn:] + level).astype(np.float32))


def seasonal_series(n_series, n_obs, period=SEASON_PERIOD, seed=SEED):
    import numpy as np
    rng = np.random.default_rng(seed)
    t = np.arange(n_obs, dtype=np.float64)[None, :]
    level = rng.uniform(50.0, 150.0, size=(n_series, 1))
    slope = rng.uniform(-0.01, 0.02, size=(n_series, 1))
    amp = rng.uniform(5.0, 20.0, size=(n_series, 1))
    phase = rng.uniform(0.0, 2 * np.pi, size=(n_series, 1))
    noise = rng.normal(0.0, 1.0, size=(n_series, n_obs))
    y = level + slope * t + amp * np.sin(2 * np.pi * t / period + phase) + noise
    return np.ascontiguousarray(y.astype(np.float32))


def prep(args):
    import numpy as np
    ctd = _load("classical_two_datasets")
    lanes = [l for l in args.lanes.split(",") if l]
    for l in lanes:
        if l not in LANES:
            raise SystemExit("unknown lane %r; choose from %s" % (l, ",".join(LANE_ORDER)))
    datasets = [d for d in args.datasets.split(",") if d]
    cap = int(args.max_rows) if args.max_rows else None
    os.makedirs(args.data, exist_ok=True)
    blocks = sorted({block_of(l) for l in lanes})
    base = {"rule": "tools/bench_board_more.py prep", "smoke_max_rows": cap, "seed": SEED}
    if "arma" in blocks:
        n = _cap(ARMA_FIT, cap, 200) + ARMA_H
        Y = arma_series(TS_SERIES, n)
        ctd._write_block(args.data, "arma-synthetic", {"Y": Y}, dict(
            base, block="arma", dataset="synthetic", series=TS_SERIES, n_obs=n, h=ARMA_H,
            generator="ARMA(1,1) phi 0.6 theta 0.3, 200 burn-in, level U(-2,2), default_rng(7)"))
    if "seasonal" in blocks:
        n = _cap(SEASON_FIT, cap, SEASON_PERIOD * 8) + SEASON_H
        Y = seasonal_series(TS_SERIES, n)
        ctd._write_block(args.data, "seasonal-synthetic", {"Y": Y}, dict(
            base, block="seasonal", dataset="synthetic", series=TS_SERIES, n_obs=n, h=SEASON_H,
            period=SEASON_PERIOD,
            generator="level U(50,150) + slope U(-.01,.02) t + amp U(5,20) sin(2 pi t/24 + "
                      "phase) + N(0,1), default_rng(7)"))
    table = [b for b in blocks if b in ("manifold", "cls", "reg", "tsvd", "ivf")]
    if not table:
        return 0
    harness = ctd._module("speed_gbdt_arm")
    for ds in datasets:
        if ds not in ("taxi", "istella"):
            continue
        t0 = time.perf_counter()
        drec = dict(base, dataset=ds, data_root=harness.data_root())
        loaded = {}

        def load(regression):
            if regression not in loaded:
                if ds == "taxi":
                    d = harness.load_taxi("shipped", regression=regression)
                    loaded[regression] = (ctd._taxi_numeric(harness, d.X_train),
                                          ctd._taxi_numeric(harness, d.X_test),
                                          np.asarray(d.y_train), np.asarray(d.y_test),
                                          "speed_gbdt_arm.load_taxi('shipped', regression=%s), "
                                          "TAXI_NUMERIC columns" % regression)
                else:
                    d = harness.load_istella("shipped", regression=True)
                    ytr, yte = np.asarray(d.y_train), np.asarray(d.y_test)
                    if not regression:
                        ytr, yte = (ytr > 0).astype(np.float32), (yte > 0).astype(np.float32)
                    loaded[regression] = (d.X_train, d.X_test, ytr, yte,
                                          "speed_gbdt_arm.load_istella('shipped', regression=True)"
                                          + ("" if regression else ", label relevance > 0"))
            return loaded[regression]

        for blk in table:
            if blk == "ivf":
                knn_ds = ctd._module("knn_datasets")
                b = knn_ds.real_block(ds, n_index=_cap(IVF_INDEX, cap, 2048),
                                      n_queries=_cap(IVF_QUERIES, cap and max(64, cap // 50), 64))
                index, bad_i = ctd.clean_sentinel(b["index"])
                queries, bad_q = ctd.clean_sentinel(b["queries"])
                ctd._write_block(args.data, "ivf-" + ds, {"index": index, "queries": queries}, dict(
                    drec, block="ivf", loader="knn_datasets.real_block(%r)" % ds,
                    index_rows=b["index_rows"], query_rows=b["query_rows"],
                    sentinel_cells_replaced={"index": bad_i, "queries": bad_q}, scaling="none"))
                continue
            regression = blk != "cls"
            xtr, xte, ytr, yte, loader = load(regression)
            if blk == "manifold":
                idx = ctd.stride_rows(xtr.shape[0], _cap(MANIFOLD_ROWS, cap, 64))
                X, bad = ctd.clean_sentinel(xtr[idx])
                (X,) = ctd.standardize(X)
                ctd._write_block(args.data, "manifold-" + ds, {"X": X}, dict(
                    drec, block="manifold", loader=loader,
                    rows="stride sample of %d of %d train rows" % (X.shape[0], xtr.shape[0]),
                    sentinel_cells_replaced={"X": bad}, scaling="standardized by these rows"))
            elif blk == "tsvd":
                idx = ctd.stride_rows(xtr.shape[0], _cap(TSVD_ROWS, cap, 64))
                X, bad = ctd.clean_sentinel(xtr[idx])
                ctd._write_block(args.data, "tsvd-" + ds, {"X": X}, dict(
                    drec, block="tsvd", loader=loader,
                    rows="stride sample of %d of %d train rows" % (X.shape[0], xtr.shape[0]),
                    sentinel_cells_replaced={"X": bad}, scaling="none"))
            else:
                fi = ctd.stride_rows(xtr.shape[0], _cap(LIN_ROWS, cap, 256))
                ei = ctd.stride_rows(xte.shape[0], _cap(EVAL_ROWS, cap, 256))
                X, bad_x = ctd.clean_sentinel(xtr[fi])
                Xq, bad_q = ctd.clean_sentinel(xte[ei])
                X, Xq = ctd.standardize(X, Xq)
                y = np.ascontiguousarray(np.asarray(ytr)[fi], dtype=np.float32)
                yq = np.ascontiguousarray(np.asarray(yte)[ei], dtype=np.float32)
                if blk == "cls" and len(np.unique(y)) != 2:
                    raise RuntimeError("cls-%s: the fit rows do not hold both classes" % ds)
                ctd._write_block(args.data, "%s-%s" % (blk, ds), {"X": X, "y": y, "Xq": Xq, "yq": yq}, dict(
                    drec, block=blk, loader=loader,
                    fit_rows="stride sample of %d of %d train rows" % (X.shape[0], xtr.shape[0]),
                    eval_rows="stride sample of %d of %d test rows" % (Xq.shape[0], xte.shape[0]),
                    target=("binary" if blk == "cls" else "real"),
                    sentinel_cells_replaced={"X": bad_x, "Xq": bad_q},
                    scaling="standardized by the fit rows (float64 mean and std)"))
        loaded.clear()
        print("MORE-PREP dataset=%s seconds=%.1f" % (ds, time.perf_counter() - t0), flush=True)
    return 0


# ---------------------------------------------------------------------------
# Each lane's arrays: the SAME rows for every arm and for the quality pass
# ---------------------------------------------------------------------------

#: the gmm lane's reg_covar per dataset, the same on every arm
#: (LANE_CONFIG['gmm']['reg_covar'] gives the measurement behind it)
GMM_REG_COVAR = {"taxi": 1e-6, "istella": 3e-3}


def _gmm_reg_covar(rec):
    return GMM_REG_COVAR[(rec or {}).get("dataset")]


def drop_constant_columns(D):
    """X and Xq without the columns constant on X (the fit rows): the same
    columns for every arm, removed before the clock. A full-covariance mixture
    over a constant column is singular (LANE_CONFIG['gmm']['data'])."""
    import numpy as np
    X = D["X"]
    keep = np.flatnonzero(X.max(axis=0) != X.min(axis=0))
    if keep.shape[0] == X.shape[1]:
        return D
    out = dict(D)
    for k in ("X", "Xq"):
        if k in out:
            out[k] = np.ascontiguousarray(out[k][:, keep])
    out["_dropped_constant_columns"] = int(X.shape[1] - keep.shape[0])
    return out


def shape_desc(D):
    """'X 100000x200; Xq ...' for the race record. Not every value is an array:
    the gmm lane also carries its dropped-column count (an int)."""
    return "; ".join("%s %s" % (k, "x".join(str(s) for s in v.shape) if hasattr(v, "shape") else v)
                     for k, v in sorted(D.items()))


def lane_arrays(lane, B, rec=None):
    """The arrays one lane reads from its block, subsets taken by stride (the
    same rows for every arm and for the conductor's quality pass)."""
    import numpy as np

    def sub(a, m):
        m = min(m, a.shape[0])
        idx = (np.arange(m, dtype=np.int64) * a.shape[0]) // m
        return np.ascontiguousarray(a[idx])

    blk = block_of(lane)
    if blk in ("arma", "seasonal"):
        Y = B["Y"]
        h = ARMA_H if blk == "arma" else SEASON_H
        return {"Yfit": np.ascontiguousarray(Y[:, :-h]), "Yhold": np.ascontiguousarray(Y[:, -h:])}
    if blk == "ivf":
        return {"index": B["index"], "queries": B["queries"]}
    if blk in ("manifold", "tsvd"):
        return {"X": B["X"]}
    X, y, Xq, yq = B["X"], B["y"], B["Xq"], B["yq"]
    if lane in ("knn-clf", "knn-reg"):
        return {"X": sub(X, KNN_FIT), "y": sub(y, KNN_FIT), "Xq": sub(Xq, KNN_QUERIES),
                "yq": sub(yq, KNN_QUERIES)}
    if lane in ("spectral", "agglomerative"):
        return {"X": sub(X, CLUSTER_ROWS)}
    if lane == "gmm":
        if (rec or {}).get("full_dataset_coverage") is True:
            # An explicitly complete input recipe must not regain the legacy
            # diagnostic GMM row caps in either the worker or its quality pass.
            # Keep the same constant-column policy on every measurement arm.
            if (rec.get("fit_rows") != [0, X.shape[0]]
                    or rec.get("fit_rows_available") != X.shape[0]
                    or rec.get("eval_rows_available") != Xq.shape[0]):
                raise ValueError("Full GMM recipe does not attest complete fit/eval rows")
            return drop_constant_columns({"X": X, "Xq": Xq})
        return drop_constant_columns({"X": sub(X, GMM_FIT), "Xq": sub(Xq, GMM_EVAL)})
    if lane in ("gpr", "gpc"):
        return {"X": sub(X, GP_FIT), "y": sub(y, GP_FIT), "Xq": sub(Xq, GP_EVAL), "yq": sub(yq, GP_EVAL)}
    if lane in ("svr", "kernel-ridge"):
        return {"X": sub(X, KERNEL_FIT), "y": sub(y, KERNEL_FIT), "Xq": sub(Xq, KERNEL_EVAL),
                "yq": sub(yq, KERNEL_EVAL)}
    if lane in ("nystroem", "rbf-sampler"):
        Xk = sub(X, KAPPROX_ROWS)
        return {"X": Xk, "Xcheck": np.ascontiguousarray(Xk[:KAPPROX_CHECK])}
    return {"X": X, "y": y, "Xq": Xq, "yq": yq}


def gamma_of(D):
    return 1.0 / float(D["X"].shape[1])


def length_scale_of(D):
    return float(D["X"].shape[1]) ** 0.5


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

class Runner:
    """`call` is timed; `outputs` is run after the clock, once per round."""

    def __init__(self, info, call, outputs, sync=None, params=None, model_for_hash=None):
        self.model_for_hash = model_for_hash
        self.info, self._call, self._outputs, self._sync = info, call, outputs, sync
        # the constructed estimator (or a declared dict for a function arm);
        # the worker sends its tools/bench_board_params.py record
        self.params = params

    def call(self):
        self._call()

    def sync(self):
        if self._sync:
            self._sync()

    def outputs(self):
        return self._outputs()


def _np():
    import numpy as np
    return np


def _host(a):
    if type(a).__module__.split(".")[0] == "cudf":
        # cuML 26.08 HoltWinters.forecast returns a cuDF frame even when the
        # fit input is CuPy. cuDF rejects implicit NumPy conversion; explicitly
        # copy its values to the host, preserving dtype and shape for the
        # existing series-major quality normalization below.
        return a.to_numpy(copy=True)
    ctd = _load("classical_two_datasets")
    return ctd._to_host(a)


def _expected_mode():
    return os.environ.get("MOJOLEARN_NUMERIC_MODE", "identical").strip().lower()


def _mode_readback(ml, est, binding):
    """The tier `est` runs on, read back from the binary: the mixin's
    `numeric_mode_used()`, else the lane binding's own constant through a
    probe of the same mixin (the classes without it resolve the process tier
    from the same `_backend.binding` call)."""
    fn = getattr(est, "numeric_mode_used", None)
    if callable(fn):
        return fn(), "numeric_mode_used()"
    from mojolearn import _backend
    name = getattr(est, "_BINDING", None) or binding
    mod = _backend.binding(name, None)
    short = mod.__name__.rsplit(".", 1)[-1]
    getter = getattr(mod, _backend._vendor_fn(short).removesuffix("_vendor") + "_numeric_mode", None)
    if getter is not None:
        return _backend._CODE_MODE[getter()], "%s numeric_mode constant" % name
    # 0.8.22's _mojolearn_solver and _mojolearn_tsa carry no numeric-mode
    # constant: the tier is read from the directory the loaded binary sits in.
    here = os.path.dirname(os.path.abspath(getattr(mod, "__file__", "") or ""))
    for m in ("fast", "identical", "deterministic"):
        if here == os.path.abspath(_backend.tier_dir(m)):
            return m, "%s has no numeric-mode constant; tier from its directory %s" % (name, here)
    return "unknown (%s)" % here, "%s tier directory not recognized" % name


def _ours(lane, est):
    import mojolearn as ml
    want = _expected_mode()
    mode, how = _mode_readback(ml, est, LANES[lane][1])
    info = {"library": "mojolearn", "version": getattr(ml, "__version__", "unknown"),
            "numeric_mode_env": os.environ.get("MOJOLEARN_NUMERIC_MODE"),
            "numeric_mode_used": mode, "numeric_mode_how": how, "device": "gpu",
            "module_path": getattr(ml, "__file__", None), "pre_clock_fit": False,
            "input_home": "host"}
    try:
        info["vendor_used"] = ml.vendor()
    except Exception as exc:  # noqa: BLE001
        info["vendor_used"] = "unavailable (%r)" % (exc,)
    if mode != want:
        raise RuntimeError("REFUSED: ours is not %s: the binary reads back %r" % (want.upper(), mode))
    return info


def _sk_info():
    ctd = _load("classical_two_datasets")
    try:
        ctd._sklearn_pools_touch()
    except Exception:  # noqa: BLE001
        pass
    return ctd._sklearn_info()


def _cuml(arrays):
    ctd = _load("classical_two_datasets")
    dev, info = ctd._cuml_setup(arrays)
    return dev, info, ctd._cupy_sync


def build(lane, arm, D, rec):
    """The runner for (lane, arm) on the lane's arrays D."""
    np = _np()
    S = {}
    if arm in ("ours", "ours-fast"):
        return _build_ours(lane, D, rec, S)
    if arm == "sklearn-cpu":
        return _build_sklearn(lane, D, rec, S)
    if arm in ("umap-learn-cpu", "umap-learn-cpu-unseeded"):
        import umap
        seeded = arm == "umap-learn-cpu"
        info = _sk_info()
        info.update(library="umap-learn", version=umap.__version__)
        kw = _umap_kw()
        kw.update(random_state=SEED if seeded else None, n_jobs=-1)
        info["config"] = "umap.UMAP(%s)" % ", ".join("%s=%r" % kv for kv in sorted(kw.items()))
        if seeded:
            info["threads_note"] = "random_state set: umap-learn runs one thread (its rule)"

        def call():
            S["e"] = umap.UMAP(**kw).fit(D["X"]).embedding_
        return Runner(info, call, lambda: {"embedding": np.asarray(S["e"], dtype=np.float32)},
                      params=umap.UMAP(**kw))
    if arm == "statsmodels-cpu":
        return _build_statsmodels(lane, D, rec, S)
    if arm == "faiss-cpu":
        import faiss
        info = {"library": "faiss", "version": faiss.__version__, "device": "cpu",
                "pre_clock_fit": False, "input_home": "host",
                "omp_threads": faiss.omp_get_max_threads(),
                "config": "IndexIVFFlat(IndexFlatL2(d), d, %d, METRIC_L2); cp.niter=20, cp.seed=7; "
                          "train + add + search, nprobe=%d, k=%d" % (IVF_NLIST, IVF_NPROBE, IVF_K)}
        info.update(_load("classical_two_datasets")._host_info())
        X, Q = D["index"], D["queries"]
        nlist = min(IVF_NLIST, X.shape[0] // 4)
        declared = {"__library__": "faiss", "seed": SEED, "nlist": nlist,
                    "nprobe": min(IVF_NPROBE, nlist), "n_neighbors": IVF_K,
                    "metric": "sqeuclidean", "kmeans_n_iters": 20}

        def call():
            d = X.shape[1]
            quant = faiss.IndexFlatL2(d)
            index = faiss.IndexIVFFlat(quant, d, nlist, faiss.METRIC_L2)
            index.cp.niter = 20
            index.cp.seed = SEED
            index.train(X)
            index.add(X)
            index.nprobe = min(IVF_NPROBE, nlist)
            _dist, ind = index.search(Q, IVF_K)
            S["ind"], S["keep"] = ind, (quant, index)
        return Runner(info, call, lambda: {"ind": np.asarray(S["ind"], dtype=np.int64)},
                      params=declared)
    if arm == "cuvs-gpu":
        from cuvs.neighbors import ivf_flat
        import cuvs
        dev, info, sync = _cuml({"index": D["index"], "queries": D["queries"]})
        info.update(library="cuvs", version=getattr(cuvs, "__version__", "unknown"),
                    config="cuvs.neighbors.ivf_flat build(IndexParams(n_lists=%d, metric='sqeuclidean', "
                           "kmeans_n_iters=20)) + search(SearchParams(n_probes=%d), k=%d)"
                           % (IVF_NLIST, IVF_NPROBE, IVF_K))
        nlist = min(IVF_NLIST, D["index"].shape[0] // 4)
        # cuVS ivf_flat IndexParams takes no seed
        declared = {"__library__": "cuvs", "nlist": nlist, "nprobe": min(IVF_NPROBE, nlist),
                    "n_neighbors": IVF_K, "metric": "sqeuclidean", "kmeans_n_iters": 20}

        def call():
            idx = ivf_flat.build(ivf_flat.IndexParams(n_lists=nlist, metric="sqeuclidean",
                                                      kmeans_n_iters=20), dev["index"])
            _d, ind = ivf_flat.search(ivf_flat.SearchParams(n_probes=min(IVF_NPROBE, nlist)), idx,
                                      dev["queries"], IVF_K)
            S["ind"] = ind
        return Runner(info, call, lambda: {"ind": np.asarray(_host(S["ind"]), dtype=np.int64)}, sync,
                      params=declared)
    if arm == "cuml-gpu":
        return _build_cuml(lane, D, rec, S)
    raise SystemExit("no arm %r for lane %r" % (arm, lane))


def _umap_kw():
    return dict(n_neighbors=UMAP_NEIGHBORS, n_components=2, min_dist=0.1, spread=1.0,
                n_epochs=UMAP_EPOCHS,
                metric="euclidean", init="spectral", learning_rate=1.0, repulsion_strength=1.0,
                negative_sample_rate=5, set_op_mix_ratio=1.0, local_connectivity=1.0)


def _gmm_out(np, w, m, c, n_iter):
    return {"weights": np.asarray(w, dtype=np.float64), "means": np.asarray(m, dtype=np.float64),
            "covariances": np.asarray(c, dtype=np.float64),
            "n_iter": np.array([int(n_iter)], dtype=np.int64)}


def _supervised_out(lane, np, est, Xq):
    if lane in ("logreg", "gpc"):
        P = np.asarray(_host(est.predict_proba(Xq)), dtype=np.float64)
        cls = np.asarray(_host(est.classes_), dtype=np.float64).reshape(-1)
        return {"proba1": P[:, list(cls).index(1.0)] if P.ndim == 2 else P,
                "pred": np.asarray(_host(est.predict(Xq)), dtype=np.float64).reshape(-1)}
    return {"pred": np.asarray(_host(est.predict(Xq)), dtype=np.float64).reshape(-1)}


def _build_ours(lane, D, rec, S, *, separate_inference=False, classical_variant=None):
    np = _np()
    import mojolearn as ml
    d = D["X"].shape[1] if "X" in D else (D["index"].shape[1] if "index" in D else None)
    X = D.get("X")
    if lane == "umap":
        make = lambda: ml.UMAP(random_state=SEED, **_umap_kw())  # noqa: E731
        def call():
            S["est"] = make()
            S["e"] = S["est"].fit_transform(X)
        out = lambda: {"embedding": np.asarray(S["e"], dtype=np.float32)}  # noqa: E731
    elif lane == "spectral-embedding":
        make = lambda: ml.SpectralEmbedding(n_components=2, affinity="nearest_neighbors",  # noqa: E731
                                            n_neighbors=10, random_state=SEED)
        def call():
            S["est"] = make()
            S["e"] = S["est"].fit_transform(X)
        out = lambda: {"embedding": np.asarray(S["e"], dtype=np.float32)}  # noqa: E731
    elif lane == "gmm":
        make = lambda: ml.GaussianMixture(n_components=GMM_COMPONENTS, covariance_type="full",  # noqa: E731
                                          tol=1e-3, reg_covar=_gmm_reg_covar(rec), max_iter=100,
                                          init_params="kmeans", n_init=1, warm_start=False,
                                          random_state=SEED)
        call = lambda: S.update(est=make().fit(X))  # noqa: E731

        def out():
            e = S["est"]
            return _gmm_out(np, e.weights_, e.means_, e.covariances_, e.n_iter_)
    elif lane in ("tsvd",):
        make = lambda: ml.TruncatedSVD(n_components=TSVD_COMPONENTS, algorithm="covariance_eigh",  # noqa: E731
                                       n_iter=5, n_oversamples=10, tol=0.0, random_state=SEED)
        call = lambda: S.update(est=make().fit(X))  # noqa: E731
        out = lambda: {"components": np.asarray(S["est"].components_, dtype=np.float64)}  # noqa: E731
    elif lane in ("spectral", "agglomerative"):
        if lane == "spectral":
            make = lambda: ml.SpectralClustering(n_clusters=N_CLUSTERS, affinity="nearest_neighbors",  # noqa: E731
                                                 n_neighbors=10, assign_labels="kmeans",
                                                 n_init=SPECTRAL_N_INIT, n_components=N_CLUSTERS,
                                                 random_state=SPECTRAL_SEED)
        else:
            make = lambda: ml.AgglomerativeClustering(n_clusters=N_CLUSTERS, metric="euclidean",  # noqa: E731
                                                      connectivity="pairwise", linkage="single",
                                                      compute_full_tree="auto", distance_threshold=None)
            # C42's separate saved-dataset variant exercises the public
            # x_cluster route (per-merge distances or another supported linkage).
            # Both arms receive the same explicit settings. The existing board
            # single-linkage recipe remains its original hierarchy operation.
            # NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
            if classical_variant is not None:
                make = lambda: ml.AgglomerativeClustering(**classical_variant)
        call = lambda: S.update(est=make().fit(X))  # noqa: E731
        out = lambda: {"labels": np.asarray(S["est"].labels_, dtype=np.int64).reshape(-1)}  # noqa: E731
    elif lane in ("knn-clf", "knn-reg"):
        cls = ml.KNeighborsClassifier if lane == "knn-clf" else ml.KNeighborsRegressor
        make = lambda: cls(n_neighbors=KNN_K, weights="uniform", metric="euclidean", algorithm="brute", p=2)  # noqa: E731
        # the classifier takes integer labels only (cuML's check_dtype=np.int32);
        # the cast is outside the clock, the same 0/1 values every arm reads
        yfit = D["y"].astype(np.int32) if lane == "knn-clf" else D["y"]

        def call():
            est = make()
            est.fit(X, yfit)
            S["est"] = est
            if not separate_inference:
                S["pred"] = est.predict(D["Xq"])
        out = lambda: {"pred": np.asarray(S["pred"], dtype=np.float64).reshape(-1)}  # noqa: E731
    elif lane in ("nystroem", "rbf-sampler"):
        g = gamma_of(D)
        if lane == "nystroem":
            make = lambda: ml.Nystroem(kernel="rbf", gamma=g, n_components=KAPPROX_COMPONENTS,  # noqa: E731
                                       random_state=SEED)
        else:
            make = lambda: ml.RBFSampler(gamma=g, n_components=KAPPROX_COMPONENTS, random_state=SEED)  # noqa: E731

        def call():
            est = make()
            S["z"] = est.fit_transform(X)
            S["est"] = est
        out = lambda: {"zcheck": np.asarray(S["est"].transform(D["Xcheck"]), dtype=np.float64)}  # noqa: E731
    elif lane == "arima":
        make = lambda: ml.ARIMA(order=ARIMA_ORDER, seasonal_order=ARIMA_SEASONAL, trend="c",  # noqa: E731
                                method="ml", maxiter=1000)
        call = lambda: S.update(est=make().fit(D["Yfit"]))  # noqa: E731

        def out():
            e = S["est"]
            return {"llf": np.asarray(e.llf_, dtype=np.float64).reshape(-1),
                    "forecast": np.asarray(e.forecast(ARMA_H), dtype=np.float64),
                    "insample": np.asarray(e.predict(0, D["Yfit"].shape[1]), dtype=np.float64)}
    elif lane == "ets":
        Y = D["Yfit"]
        make = lambda: ml.ExponentialSmoothing(Y, seasonal="additive", seasonal_periods=SEASON_PERIOD,  # noqa: E731
                                               start_periods=2, ts_num=Y.shape[0], eps=ETS_EPS,
                                               initialization_method="estimated")
        call = lambda: S.update(est=make().fit())  # noqa: E731

        def out():
            e = S["est"]
            return {"forecast": np.asarray(e.forecast(SEASON_H), dtype=np.float64).T,
                    "insample": np.asarray(e.predict(0, Y.shape[1]), dtype=np.float64).T}
    elif lane == "ivf":
        make = lambda: ml.IVFIndex(n_lists=min(IVF_NLIST, D["index"].shape[0] // 4),  # noqa: E731
                                   n_probes=min(IVF_NPROBE, IVF_NLIST, D["index"].shape[0] // 4),
                                   n_neighbors=IVF_K, kmeans_n_iters=20, metric="sqeuclidean",
                                   random_state=SEED)

        def call():
            est = make()
            est.fit(D["index"])
            S["est"] = est
            if not separate_inference:
                _dist, ind = est.search(D["queries"])
                S["ind"] = ind
        out = lambda: {"ind": np.asarray(S["ind"], dtype=np.int64)}  # noqa: E731
    else:
        g, ls = (gamma_of(D), length_scale_of(D))
        ctors = {
            "logreg": lambda: ml.LogisticRegression(penalty="l2", C=1.0, tol=1e-4, max_iter=1000,
                                                    fit_intercept=True, solver="qn",
                                                    class_weight=None),
            "linearsvc": lambda: ml.LinearSVC(penalty="l2", loss="squared_hinge", C=1.0, tol=1e-4,
                                              max_iter=1000, fit_intercept=True,
                                              penalized_intercept=False, class_weight=None),
            # 'auto' resolves to 'eig' here (solver_); named, not left to resolve
            "ridge": lambda: ml.Ridge(alpha=1.0, fit_intercept=True, solver="eig"),
            # random_state stays None: ours refuses any other value (it selects
            # nothing once selection='random' is refused)
            "lasso": lambda: ml.Lasso(alpha=0.01, fit_intercept=True, max_iter=1000, tol=1e-4,
                                      selection="cyclic", solver="cd", precompute=False,
                                      positive=False, warm_start=False),
            "elasticnet": lambda: ml.ElasticNet(alpha=ENET_ALPHA, l1_ratio=0.5, fit_intercept=True,
                                                max_iter=1000, tol=1e-4, selection="cyclic",
                                                solver="cd", precompute=False, positive=False,
                                                warm_start=False),
            "linearsvr": lambda: ml.LinearSVR(epsilon=0.0, penalty="l2", loss="epsilon_insensitive",
                                              C=1.0, tol=1e-4, max_iter=1000, fit_intercept=True,
                                              penalized_intercept=False),
            "gpr": lambda: ml.GaussianProcessRegressor(
                kernel=ml.ConstantKernel(1.0) * ml.RBF(length_scale=ls) + ml.WhiteKernel(noise_level=GP_NOISE), alpha=GP_ALPHA, optimizer=None,
                normalize_y=False, n_restarts_optimizer=0, random_state=SEED),
            # random_state stays None: ours refuses any other value
            "gpc": lambda: ml.GaussianProcessClassifier(
                kernel=ml.ConstantKernel(1.0) * ml.RBF(length_scale=ls), optimizer=None,
                max_iter_predict=100, n_restarts_optimizer=0),
            "svr": lambda: ml.SVR(kernel="rbf", gamma=g, C=1.0, epsilon=0.1, tol=1e-3, degree=3,
                                  coef0=0.0, max_iter=-1, cache_size=SVR_CACHE_MB),
            "kernel-ridge": lambda: ml.KernelRidge(alpha=1.0, kernel="rbf", gamma=g, degree=3,
                                                   coef0=1.0),
        }
        make = ctors[lane]
        call = lambda: S.update(est=make().fit(X, D["y"]))  # noqa: E731
        if lane == "gpr":
            def out():
                mu, sd = S["est"].predict(D["Xq"], return_std=True)
                return {"pred": np.asarray(mu, dtype=np.float64).reshape(-1),
                        "std": np.asarray(sd, dtype=np.float64).reshape(-1)}
        else:
            out = lambda: _supervised_out(lane, np, S["est"], D["Xq"])  # noqa: E731
    probe = make()
    info = _ours(lane, probe)
    if d is not None:
        info["n_features"] = d
    return Runner(info, call, out, params=probe, model_for_hash=lambda: S.get("est"))


def _build_sklearn(lane, D, rec, S):
    np = _np()
    info = _sk_info()
    X = D.get("X")
    if lane == "spectral-embedding":
        from sklearn.manifold import SpectralEmbedding
        make = lambda: SpectralEmbedding(n_components=2, affinity="nearest_neighbors", n_neighbors=10,  # noqa: E731
                                         random_state=SEED, n_jobs=-1)
        call = lambda: S.update(e=make().fit_transform(X))  # noqa: E731
        out = lambda: {"embedding": np.asarray(S["e"], dtype=np.float32)}  # noqa: E731
    elif lane == "gmm":
        from sklearn.mixture import GaussianMixture
        make = lambda: GaussianMixture(n_components=GMM_COMPONENTS, covariance_type="full", tol=1e-3,  # noqa: E731
                                       reg_covar=_gmm_reg_covar(rec), max_iter=100, init_params="kmeans", n_init=1,
                                       warm_start=False, random_state=SEED)
        # Float32 covariance accumulation can lose positive definiteness even
        # after standardization. The retained sklearn failure recommends its
        # float64 path; promoting these same values changes no samples or
        # estimator parameters. Include the conversion in the timed fit call.
        info.update(input_dtype=str(X.dtype), fit_dtype="float64",
                    input_conversion_inside_clock=True,
                    fit_rows=int(X.shape[0]), eval_rows=int(D["Xq"].shape[0]))

        def call():
            S.update(est=make().fit(np.ascontiguousarray(X, dtype=np.float64)))

        def out():
            e = S["est"]
            return _gmm_out(np, e.weights_, e.means_, e.covariances_, e.n_iter_)
    elif lane == "tsvd":
        from sklearn.decomposition import TruncatedSVD
        make = lambda: TruncatedSVD(n_components=TSVD_COMPONENTS, algorithm="arpack", tol=0.0,  # noqa: E731
                                    n_iter=5, n_oversamples=10, random_state=SEED)
        call = lambda: S.update(est=make().fit(X))  # noqa: E731
        out = lambda: {"components": np.asarray(S["est"].components_, dtype=np.float64)}  # noqa: E731
    elif lane in ("spectral", "agglomerative"):
        if lane == "spectral":
            from sklearn.cluster import SpectralClustering
            make = lambda: SpectralClustering(n_clusters=N_CLUSTERS, affinity="nearest_neighbors",  # noqa: E731
                                              n_neighbors=10, assign_labels="kmeans",
                                              n_init=SPECTRAL_N_INIT, n_components=N_CLUSTERS,
                                              random_state=SPECTRAL_SEED, n_jobs=-1)
        else:
            from sklearn.cluster import AgglomerativeClustering
            make = lambda: AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage="single",  # noqa: E731
                                                   metric="euclidean", compute_full_tree="auto",
                                                   distance_threshold=None)
        call = lambda: S.update(est=make().fit(X))  # noqa: E731
        out = lambda: {"labels": np.asarray(S["est"].labels_, dtype=np.int64).reshape(-1)}  # noqa: E731
    elif lane in ("knn-clf", "knn-reg"):
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        cls = KNeighborsClassifier if lane == "knn-clf" else KNeighborsRegressor
        make = lambda: cls(n_neighbors=KNN_K, weights="uniform", metric="euclidean",  # noqa: E731
                           algorithm="brute", p=2, n_jobs=-1)

        def call():
            est = make()
            est.fit(X, D["y"])
            S["pred"] = est.predict(D["Xq"])
        out = lambda: {"pred": np.asarray(S["pred"], dtype=np.float64).reshape(-1)}  # noqa: E731
    elif lane in ("nystroem", "rbf-sampler"):
        from sklearn.kernel_approximation import Nystroem, RBFSampler
        g = gamma_of(D)
        if lane == "nystroem":
            # degree and coef0 are not read by the rbf kernel; set to ours' values
            make = lambda: Nystroem(kernel="rbf", gamma=g, degree=3, coef0=1.0,  # noqa: E731
                                    n_components=KAPPROX_COMPONENTS, random_state=SEED, n_jobs=-1)
        else:
            make = lambda: RBFSampler(gamma=g, n_components=KAPPROX_COMPONENTS, random_state=SEED)  # noqa: E731

        def call():
            est = make()
            S["z"] = est.fit_transform(X)
            S["est"] = est
        out = lambda: {"zcheck": np.asarray(S["est"].transform(D["Xcheck"]), dtype=np.float64)}  # noqa: E731
    else:
        from sklearn import gaussian_process as gp, kernel_ridge, linear_model as lm, svm
        g, ls = gamma_of(D), length_scale_of(D)
        K = gp.kernels
        ctors = {
            "logreg": lambda: lm.LogisticRegression(penalty="l2", C=1.0, tol=1e-4, max_iter=1000,
                                                    fit_intercept=True, solver="lbfgs",
                                                    class_weight=None, random_state=SEED),
            "linearsvc": lambda: svm.LinearSVC(penalty="l2", loss="squared_hinge", C=1.0, tol=1e-4,
                                               max_iter=1000, fit_intercept=True, dual="auto",
                                               intercept_scaling=1.0, class_weight=None,
                                               random_state=SEED),
            # 'auto' is 'cholesky' for this dense input; named, not left to resolve
            "ridge": lambda: lm.Ridge(alpha=1.0, fit_intercept=True, solver="cholesky",
                                      positive=False, random_state=SEED),
            "lasso": lambda: lm.Lasso(alpha=0.01, fit_intercept=True, max_iter=1000, tol=1e-4,
                                      selection="cyclic", precompute=False, positive=False,
                                      warm_start=False, random_state=SEED),
            "elasticnet": lambda: lm.ElasticNet(alpha=ENET_ALPHA, l1_ratio=0.5, fit_intercept=True,
                                                max_iter=1000, tol=1e-4, selection="cyclic",
                                                precompute=False, positive=False,
                                                warm_start=False, random_state=SEED),
            "linearsvr": lambda: svm.LinearSVR(epsilon=0.0, loss="epsilon_insensitive", C=1.0,
                                               tol=1e-4, max_iter=1000, fit_intercept=True,
                                               intercept_scaling=1.0, dual=True, random_state=SEED),
            "gpr": lambda: gp.GaussianProcessRegressor(
                kernel=K.ConstantKernel(1.0) * K.RBF(length_scale=ls) + K.WhiteKernel(noise_level=GP_NOISE), alpha=GP_ALPHA, optimizer=None,
                normalize_y=False, n_restarts_optimizer=0, random_state=SEED),
            "gpc": lambda: gp.GaussianProcessClassifier(
                kernel=K.ConstantKernel(1.0) * K.RBF(length_scale=ls), optimizer=None,
                max_iter_predict=100, n_restarts_optimizer=0, random_state=SEED),
            "svr": lambda: svm.SVR(kernel="rbf", gamma=g, C=1.0, epsilon=0.1, tol=1e-3, degree=3,
                                   coef0=0.0, max_iter=-1, shrinking=True, cache_size=SVR_CACHE_MB),
            "kernel-ridge": lambda: kernel_ridge.KernelRidge(alpha=1.0, kernel="rbf", gamma=g,
                                                             degree=3, coef0=1.0),
        }
        if lane not in ctors:
            raise SystemExit("no sklearn-cpu arm for lane %r" % lane)
        make = ctors[lane]
        call = lambda: S.update(est=make().fit(X, D["y"]))  # noqa: E731
        if lane == "gpr":
            def out():
                mu, sd = S["est"].predict(D["Xq"], return_std=True)
                return {"pred": np.asarray(mu, dtype=np.float64).reshape(-1),
                        "std": np.asarray(sd, dtype=np.float64).reshape(-1)}
        else:
            out = lambda: _supervised_out(lane, np, S["est"], D["Xq"])  # noqa: E731
    probe = make()
    info["config"] = repr(probe)
    return Runner(info, call, out, params=probe)


def _sm_arima_one(y):
    import warnings
    from statsmodels.tsa.arima.model import ARIMA
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ARIMA(y.astype("float64"), order=ARIMA_ORDER, seasonal_order=ARIMA_SEASONAL,
                    trend="c").fit(method_kwargs={"maxiter": 1000})
    return float(res.llf), res.forecast(ARMA_H), res.fittedvalues


def _sm_ets_one(y):
    import warnings
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ExponentialSmoothing(y.astype("float64"), trend="additive", seasonal="additive",
                                   seasonal_periods=SEASON_PERIOD, damped_trend=False,
                                   use_boxcox=False, initialization_method="estimated").fit()
    return res.forecast(SEASON_H), res.fittedvalues


def _build_statsmodels(lane, D, rec, S):
    np = _np()
    import statsmodels
    from joblib import Parallel, delayed
    info = _sk_info()
    info.update(library="statsmodels", version=statsmodels.__version__)
    Y = D["Yfit"]
    fn = _sm_arima_one if lane == "arima" else _sm_ets_one
    pool = Parallel(n_jobs=-1)
    info["config"] = ("statsmodels.tsa.arima.model.ARIMA(order=(1,0,1), seasonal_order=(0,0,0,0), "
                      "trend='c').fit(method_kwargs={'maxiter': 1000})" if lane == "arima" else
                      "statsmodels.tsa.holtwinters.ExponentialSmoothing(trend='additive', "
                      "seasonal='additive', seasonal_periods=24, damped_trend=False, use_boxcox=False, "
                      "initialization_method='estimated').fit()")
    info["config"] += "; one series per fit, joblib Parallel(n_jobs=-1) over the %d series" % Y.shape[0]
    # Declared (a function per series, not an estimator): the values _sm_*_one pass.
    if lane == "arima":
        declared = {"__library__": "statsmodels", "order": list(ARIMA_ORDER),
                    "seasonal_order": list(ARIMA_SEASONAL), "trend": "c", "maxiter": 1000}
    else:
        declared = {"__library__": "statsmodels", "trend": "additive", "seasonal": "additive",
                    "seasonal_periods": SEASON_PERIOD, "damped_trend": False,
                    "initialization_method": "estimated"}

    def call():
        S["res"] = pool(delayed(fn)(Y[i]) for i in range(Y.shape[0]))

    def out():
        r = S["res"]
        if lane == "arima":
            return {"llf": np.array([x[0] for x in r], dtype=np.float64),
                    "forecast": np.array([np.asarray(x[1]) for x in r], dtype=np.float64),
                    "insample": np.array([np.asarray(x[2]) for x in r], dtype=np.float64)}
        return {"forecast": np.array([np.asarray(x[0]) for x in r], dtype=np.float64),
                "insample": np.array([np.asarray(x[1]) for x in r], dtype=np.float64)}
    return Runner(info, call, out, params=declared)


def _series_major(a, n_series):
    """(n_series, t) from a cuML (t, n_series) or (n_series, t) array."""
    a = _np().asarray(a, dtype=_np().float64)
    if a.ndim == 1:
        return a.reshape(n_series, -1)
    if a.shape[0] != n_series and a.shape[1] == n_series:
        return a.T
    return a


def _build_cuml(lane, D, rec, S):
    np = _np()
    import cuml
    ctd = _load("classical_two_datasets")
    arrays = {k: v for k, v in D.items() if k in ("X", "y", "Xq")}
    if lane == "arima":
        arrays = {"Yt": np.ascontiguousarray(D["Yfit"].T)}          # cuML: (n_obs, batch)
    elif lane == "ets":
        arrays = {"Y": D["Yfit"]}                                     # cuML: (ts_num, n)
    dev, info, sync = _cuml(arrays)
    X = dev.get("X")
    out_type = {"output_type": "cupy"}

    def tolerant(cls, **kw):
        """cls with the keywords this cuML build takes; a dropped one is named
        in info['params_not_in_this_build'] (never silently)."""
        _obj, kept = ctd.construct_tolerant(cls, kw, info)
        return lambda: cls(**kept)  # noqa: E731

    if lane == "umap":
        from cuml.manifold import UMAP
        kw = _umap_kw()
        kw.update(random_state=SEED, build_algo="brute_force_knn", **out_type)
        make = tolerant(UMAP, **kw)
        call = lambda: S.update(e=make().fit_transform(X))  # noqa: E731
        out = lambda: {"embedding": np.asarray(_host(S["e"]), dtype=np.float32)}  # noqa: E731
    elif lane == "spectral-embedding":
        from cuml.manifold import SpectralEmbedding
        make = tolerant(SpectralEmbedding, n_components=2, affinity="nearest_neighbors",
                        n_neighbors=10, random_state=SEED)
        call = lambda: S.update(e=make().fit_transform(X))  # noqa: E731
        out = lambda: {"embedding": np.asarray(_host(S["e"]), dtype=np.float32)}  # noqa: E731
    elif lane == "tsvd":
        from cuml.decomposition import TruncatedSVD
        make = tolerant(TruncatedSVD, n_components=TSVD_COMPONENTS, algorithm="full",
                        random_state=SEED, **out_type)
        call = lambda: S.update(est=make().fit(X))  # noqa: E731
        out = lambda: {"components": np.asarray(_host(S["est"].components_), dtype=np.float64)}  # noqa: E731
    elif lane in ("spectral", "agglomerative"):
        if lane == "spectral":
            from cuml.cluster import SpectralClustering
            make = tolerant(SpectralClustering, n_clusters=N_CLUSTERS, affinity="nearest_neighbors",
                            n_neighbors=10, n_init=SPECTRAL_N_INIT, n_components=N_CLUSTERS,
                            random_state=SPECTRAL_SEED)
        else:
            from cuml.cluster import AgglomerativeClustering
            make = tolerant(AgglomerativeClustering, n_clusters=N_CLUSTERS, metric="euclidean",
                            linkage="single", connectivity="pairwise", **out_type)
        call = lambda: S.update(est=make().fit(X))  # noqa: E731
        out = lambda: {"labels": np.asarray(_host(S["est"].labels_), dtype=np.int64).reshape(-1)}  # noqa: E731
    elif lane in ("knn-clf", "knn-reg"):
        from cuml.neighbors import KNeighborsClassifier, KNeighborsRegressor
        cls = KNeighborsClassifier if lane == "knn-clf" else KNeighborsRegressor
        make = tolerant(cls, n_neighbors=KNN_K, weights="uniform", metric="euclidean",
                        algorithm="brute", p=2, **out_type)

        def call():
            est = make()
            est.fit(X, dev["y"])
            S["pred"] = est.predict(dev["Xq"])
        out = lambda: {"pred": np.asarray(_host(S["pred"]), dtype=np.float64).reshape(-1)}  # noqa: E731
    elif lane == "arima":
        from cuml.tsa.arima import ARIMA
        nb = D["Yfit"].shape[0]
        make = tolerant(ARIMA, endog=dev["Yt"], order=ARIMA_ORDER, seasonal_order=ARIMA_SEASONAL,
                        fit_intercept=True, output_type="numpy")

        def call():
            est = make()
            est.fit(method="ml", maxiter=1000)
            S["est"] = est

        def out():
            e = S["est"]
            return {"llf": np.asarray(_host(e.llf), dtype=np.float64).reshape(-1),
                    "forecast": _series_major(_host(e.forecast(ARMA_H)), nb),
                    "insample": _series_major(_host(e.predict(0, D["Yfit"].shape[1])), nb)}
    elif lane == "ets":
        from cuml import ExponentialSmoothing
        nb = D["Yfit"].shape[0]
        make = tolerant(ExponentialSmoothing, endog=dev["Y"], seasonal="additive",
                        seasonal_periods=SEASON_PERIOD, start_periods=2, ts_num=nb, eps=ETS_EPS)
        call = lambda: S.update(est=make().fit())  # noqa: E731

        def out():
            return {"forecast": _series_major(_host(S["est"].forecast(SEASON_H)), nb)}
    else:
        from cuml import linear_model as lm, svm
        g = gamma_of(D)
        ctors = {
            "logreg": (lm.LogisticRegression, dict(penalty="l2", C=1.0, tol=1e-4, max_iter=1000,
                                                   fit_intercept=True, solver="qn",
                                                   class_weight=None)),
            "linearsvc": (svm.LinearSVC, dict(penalty="l2", loss="squared_hinge", C=1.0, tol=1e-4,
                                              max_iter=1000, fit_intercept=True,
                                              penalized_intercept=False, class_weight=None)),
            "ridge": (lm.Ridge, dict(alpha=1.0, fit_intercept=True, solver="eig")),
            "lasso": (lm.Lasso, dict(alpha=0.01, fit_intercept=True, max_iter=1000, tol=1e-4,
                                     selection="cyclic", solver="cd")),
            "elasticnet": (lm.ElasticNet, dict(alpha=ENET_ALPHA, l1_ratio=0.5, fit_intercept=True,
                                               max_iter=1000, tol=1e-4, selection="cyclic",
                                               solver="cd")),
            "linearsvr": (svm.LinearSVR, dict(epsilon=0.0, penalty="l2", loss="epsilon_insensitive",
                                              C=1.0, tol=1e-4, max_iter=1000, fit_intercept=True,
                                              penalized_intercept=False)),
            "svr": (svm.SVR, dict(kernel="rbf", gamma=g, C=1.0, epsilon=0.1, tol=1e-3, degree=3,
                                  coef0=0.0, max_iter=-1, cache_size=SVR_CACHE_MB)),
            "kernel-ridge": (cuml.KernelRidge, dict(alpha=1.0, kernel="rbf", gamma=g, degree=3,
                                                    coef0=1.0)),
        }
        if lane not in ctors:
            raise SystemExit("no cuml-gpu arm for lane %r" % lane)
        cls, kw = ctors[lane]
        make = tolerant(cls, **kw, **out_type)
        call = lambda: S.update(est=make().fit(X, dev["y"]))  # noqa: E731
        out = lambda: _supervised_out(lane, np, S["est"], dev["Xq"])  # noqa: E731
    probe = make()
    info["config"] = repr(probe)
    return Runner(info, call, out, sync, params=probe)


def worker(args):
    proto = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    def say(obj):
        proto.write(json.dumps(obj, sort_keys=True, default=str) + "\n")
        proto.flush()

    np = _np()
    block = os.path.join(args.data, "%s-%s" % (block_of(args.lane), args.dataset))
    try:
        with np.load(block + ".npz") as z:
            B = {k: np.ascontiguousarray(z[k]) for k in z.files}
        with open(block + ".json") as fh:
            rec = json.load(fh)
        runner = build(args.lane, args.arm, lane_arrays(args.lane, B, rec), rec)
        # the parameters this arm really got, read back from what it constructed
        params = _load("classical_two_datasets").params_record(runner.params)
    except BaseException as exc:  # noqa: BLE001 (SystemExit from a missing arm too)
        import traceback
        traceback.print_exc()
        say({"event": "error", "stage": "ready", "error": repr(exc)})
        return 1
    if isinstance(runner.info, dict):   # the arm's own library version and GPU (the store's key)
        runner.info.update(_load("bench_board_probe").library_identity(runner.info))
    say({"event": "ready", "info": runner.info, "pid": os.getpid(), "params": params})
    # peak memory per round, reset and read OUTSIDE the clock
    mem = _load("bench_board_probe").MemProbe((runner.info or {}).get("device", "gpu"),
        library=(runner.info or {}).get("library") or "?")
    last = None
    for line in sys.stdin:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "round":
            r = int(parts[1])
            try:
                mem.start()
                whole_start = time.perf_counter()
                whole_requested = os.environ.get("MOJOLEARN_BENCH_WHOLE_OPERATION") == "1"
                if whole_requested:
                    # Recreate per operation so constructor-side fit/preparation
                    # (notably kNN/KDE) is included, on every scored arm equally.
                    # Release the previous fitted device model before allocating
                    # its replacement, avoiding a transient double allocation.
                    runner = None
                    runner = build(args.lane, args.arm, lane_arrays(args.lane, B, rec), rec)
                    runner.sync()
                preparation_ms = (time.perf_counter() - whole_start) * 1000.0
                t0 = time.perf_counter()
                runner.call()
                runner.sync()
                ms = (time.perf_counter() - t0) * 1000.0
                last = runner.outputs()
                runner.sync()
                operation_ms = (time.perf_counter() - whole_start) * 1000.0
                m = mem.stop()
                h = hashlib.sha256()
                for k in sorted(last):
                    h.update(k.encode())
                    h.update(np.ascontiguousarray(last[k]).data)
                digest = h.hexdigest()[:16]
                from bench_board_state import scored_receipt
                state_receipt = scored_receipt(last, runner) if r > 0 else None
            except Exception as exc:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                say({"event": "error", "stage": "round %d" % r, "error": repr(exc)})
                return 1
            say({"event": "round", "round": r, "ms": ms, "digest": digest, "mem": m, "state_receipt": state_receipt,
                 "operation": {"ms": operation_ms, "preparation_ms": preparation_ms,
                               "scope": "prepare-fit-consume" if whole_requested else "call-consume",
                               "dataset_load_included": False,
                               "fresh_process": False, "startup_probe_preparation_excluded": True}})
        elif parts[0] == "save":
            try:
                path = parts[1]
                tmp = path + ".tmp.npz"
                np.savez(tmp, **last)
                os.replace(tmp, path)
                say({"event": "saved", "path": path, "info": runner.info})
            except Exception as exc:  # noqa: BLE001
                say({"event": "error", "stage": "save", "error": repr(exc)})
                return 1
        elif parts[0] == "quit":
            say({"event": "bye"})
            return 0
    return 0


# ---------------------------------------------------------------------------
# Quality (the conductor, float64 NumPy)
# ---------------------------------------------------------------------------

def _sqdist(A, B):
    np = _np()
    d = (A * A).sum(axis=1)[:, None] + (B * B).sum(axis=1)[None, :] - 2.0 * (A @ B.T)
    return np.maximum(d, 0.0)


def trustworthiness(X, E, k=K_TRUST, chunk=256):
    """scikit-learn's `trustworthiness(X, E, n_neighbors=k)` (euclidean), in
    row chunks: for each row, the rank (in X-space order, self excluded) of
    each of its k nearest neighbours in the embedding; the penalty is the sum
    of rank - k over ranks above k."""
    np = _np()
    X = np.asarray(X, dtype=np.float64)
    E = np.asarray(E, dtype=np.float64)
    n = X.shape[0]
    if k >= n / 2:
        raise ValueError("trustworthiness needs k < n / 2")
    t = 0.0
    ranks_base = np.arange(1, n + 1, dtype=np.int64)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        rows = np.arange(s, e)
        dx = np.sqrt(_sqdist(X[s:e], X))
        dx[np.arange(e - s), rows] = np.inf
        ind_x = np.argsort(dx, axis=1)
        inv = np.empty((e - s, n), dtype=np.int64)
        inv[np.arange(e - s)[:, None], ind_x] = ranks_base[None, :]
        de = _sqdist(E[s:e], E)
        de[np.arange(e - s), rows] = np.inf
        nn = np.argpartition(de, k - 1, axis=1)[:, :k]
        r = np.take_along_axis(inv, nn, axis=1) - k
        t += float(r[r > 0].sum())
    return 1.0 - t * (2.0 / (n * k * (2.0 * n - 3.0 * k - 1.0)))


def adjusted_rand(a, b):
    np = _np()
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    cont = np.zeros((ai.max() + 1, bi.max() + 1), dtype=np.float64)
    np.add.at(cont, (ai, bi), 1.0)
    c2 = lambda x: (x * (x - 1.0) / 2.0).sum()  # noqa: E731
    s = c2(cont)
    sa, sb = c2(cont.sum(axis=1)), c2(cont.sum(axis=0))
    n = float(a.shape[0])
    exp = sa * sb / (n * (n - 1.0) / 2.0)
    mx = (sa + sb) / 2.0
    if mx == exp:
        return 1.0
    return float((s - exp) / (mx - exp))


def silhouette(X, labels, chunk=512):
    """Mean silhouette (euclidean), scikit-learn's definition: a sample in a
    singleton cluster scores 0. None when there are fewer than 2 clusters."""
    np = _np()
    X = np.asarray(X, dtype=np.float64)
    _, lab = np.unique(np.asarray(labels).reshape(-1), return_inverse=True)
    k = int(lab.max()) + 1
    n = X.shape[0]
    if k < 2 or k >= n:
        return None
    onehot = np.zeros((n, k))
    onehot[np.arange(n), lab] = 1.0
    counts = onehot.sum(axis=0)
    s = np.empty(n)
    for st in range(0, n, chunk):
        e = min(st + chunk, n)
        d = np.sqrt(_sqdist(X[st:e], X))
        sums = d @ onehot
        own = lab[st:e]
        own_n = counts[own]
        a = np.where(own_n > 1, sums[np.arange(e - st), own] / np.maximum(own_n - 1, 1), 0.0)
        other = sums / counts[None, :]
        other[np.arange(e - st), own] = np.inf
        b = other.min(axis=1)
        v = (b - a) / np.maximum(a, b)
        s[st:e] = np.where(own_n > 1, v, 0.0)
    return float(s.mean())


def gmm_loglik(X, weights, means, covs):
    """Per-row log-likelihood of a full-covariance mixture, float64."""
    np = _np()
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    k = means.shape[0]
    lp = np.empty((n, k))
    for j in range(k):
        L = np.linalg.cholesky(covs[j])
        z = np.linalg.solve(L, (X - means[j]).T)
        lp[:, j] = (np.log(weights[j]) - 0.5 * (d * np.log(2 * np.pi) + (z * z).sum(axis=0))
                    - np.log(np.diag(L)).sum())
    m = lp.max(axis=1, keepdims=True)
    return m[:, 0] + np.log(np.exp(lp - m).sum(axis=1))


def _reg(y, p):
    np = _np()
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    res = y - p
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"r2": 1.0 - float((res * res).sum()) / ss_tot if ss_tot else None,
            "rmse": float(np.sqrt(np.mean(res * res))),
            "finite": bool(np.all(np.isfinite(p)))}


def quality(lane, D, outs):
    """Arm -> quality dict, one float64 NumPy function per lane."""
    np = _np()
    q = {}
    if lane in ("umap", "spectral-embedding"):
        for arm, o in outs.items():
            E = o["embedding"]
            q[arm] = {"trustworthiness_k15": trustworthiness(D["X"], E) if np.all(np.isfinite(E))
                      else None}
    elif lane == "gmm":
        X, Xq = D["X"], D["Xq"]
        n, d = X.shape
        for arm, o in outs.items():
            w, m, c = o["weights"], o["means"], o["covariances"]
            k = m.shape[0]
            p = k * d + k * d * (d + 1) / 2.0 + k - 1
            ll_fit = float(gmm_loglik(X, w, m, c).sum())
            q[arm] = {"mean_log_likelihood": float(gmm_loglik(Xq, w, m, c).mean()),
                      "bic": -2.0 * ll_fit + p * np.log(n), "n_iter": int(o["n_iter"][0])}
    elif lane in ("logreg", "gpc", "linearsvc", "knn-clf"):
        yq = D["yq"].astype(np.float64)
        for arm, o in outs.items():
            ent = {"accuracy": float((o["pred"] == yq).mean())}
            if "proba1" in o:
                raw = o["proba1"].astype(np.float64)
                bad = int((~np.isfinite(raw)).sum())
                # a non-finite probability is the arm's answer, counted by
                # name; a log loss over the remaining rows would not be
                # comparable, so it is left empty
                ent["nonfinite_proba_rows"] = bad
                p = np.clip(raw, 1e-15, 1 - 1e-15)
                ent["logloss"] = (float(-np.mean(yq * np.log(p) + (1 - yq) * np.log(1 - p)))
                                  if not bad else None)
            q[arm] = ent
    elif lane in ("ridge", "lasso", "elasticnet", "linearsvr", "knn-reg", "svr", "kernel-ridge"):
        for arm, o in outs.items():
            q[arm] = _reg(D["yq"], o["pred"])
    elif lane == "gpr":
        yq = D["yq"].astype(np.float64)
        for arm, o in outs.items():
            ent = _reg(yq, o["pred"])
            var = o["std"].astype(np.float64) ** 2 + GP_ALPHA
            ent["mean_log_predictive_density"] = float(np.mean(
                -0.5 * np.log(2 * np.pi * var) - (yq - o["pred"]) ** 2 / (2 * var)))
            q[arm] = ent
    elif lane == "tsvd":
        X = D["X"]
        tot_var = 0.0
        mu = X.astype(np.float64).mean(axis=0)
        fro = 0.0
        acc = {arm: [0.0, 0.0, np.zeros(o["components"].shape[0]), np.zeros(o["components"].shape[0])]
               for arm, o in outs.items()}
        n = X.shape[0]
        for s in range(0, n, 250_000):
            xb = X[s:s + 250_000].astype(np.float64)
            tot_var += float(((xb - mu) ** 2).sum())
            fro += float((xb * xb).sum())
            for arm, o in outs.items():
                V = o["components"]
                P = xb @ V.T
                R = xb - P @ V
                acc[arm][0] += float((R * R).sum())
                acc[arm][2] += P.sum(axis=0)
                acc[arm][3] += (P * P).sum(axis=0)
        for arm in outs:
            pm = acc[arm][2] / n
            pvar = acc[arm][3] / n - pm * pm
            q[arm] = {"explained_variance_ratio_sum": float(pvar.sum() * n / tot_var) if tot_var else None,
                      "relative_reconstruction_error": float(np.sqrt(acc[arm][0] / fro)) if fro else None}
    elif lane in ("spectral", "agglomerative"):
        ref = outs.get("ours")
        for arm, o in outs.items():
            lab = o["labels"]
            ent = {"n_clusters": int(np.unique(lab[lab >= 0]).shape[0]),
                   "silhouette": silhouette(D["X"], lab)}
            if ref is not None and arm != "ours" and ref["labels"].size == lab.size:
                ent["ari_vs_ours"] = adjusted_rand(ref["labels"], lab)
            q[arm] = ent
    elif lane in ("nystroem", "rbf-sampler"):
        Xc = D["Xcheck"].astype(np.float64)
        K = np.exp(-gamma_of(D) * _sqdist(Xc, Xc))
        kn = float(np.linalg.norm(K))
        for arm, o in outs.items():
            Z = o["zcheck"]
            q[arm] = {"kernel_rel_error": float(np.linalg.norm(Z @ Z.T - K)) / kn}
    elif lane in ("arima", "ets"):
        hold = D["Yhold"].astype(np.float64)
        fit = D["Yfit"].astype(np.float64)
        skip = 1 if lane == "arima" else 2 * SEASON_PERIOD
        for arm, o in outs.items():
            fc = _series_major(o["forecast"], hold.shape[0])
            ent = {"forecast_rmse": float(np.sqrt(np.mean((fc - hold) ** 2)))}
            if "insample" in o:
                ins = _series_major(o["insample"], fit.shape[0])
                err = (ins - fit)[:, skip:]
                ok = np.isfinite(err)
                ent["insample_rmse"] = float(np.sqrt(np.mean(err[ok] ** 2))) if ok.any() else None
            if lane == "arima":
                llf = o["llf"].astype(np.float64)
                n_par = ARIMA_ORDER[0] + ARIMA_ORDER[2] + 2
                ent["mean_llf"] = float(llf.mean())
                ent["mean_aic"] = float((2 * n_par - 2 * llf).mean())
            q[arm] = ent
    elif lane == "ivf":
        ctd = _load("classical_two_datasets")
        kq = ctd.quality("knn", {"index": D["index"], "queries": D["queries"]},
                         {a: {"ind": o["ind"]} for a, o in outs.items()}, {}, knn_k=IVF_K)
        for arm, ent in kq.items():
            q[arm] = {k: v for k, v in ent.items() if k != "reference"}
    return q


# ---------------------------------------------------------------------------
# race: the conductor for one (lane, dataset)
# ---------------------------------------------------------------------------

def _worker_env(arm):
    ctd = _load("classical_two_datasets")
    env = dict(os.environ)
    for k in ctd.THREAD_ENV:
        env.pop(k, None)
    ctd.apply_cpu_quota(env)
    if arm in ("ours", "ours-fast"):
        env["MOJOLEARN_NUMERIC_MODE"] = "fast" if arm == "ours-fast" else "identical"
        if os.environ.get("MOJOLEARN_BENCH_INSTALLED", "0").strip() in ("", "0"):
            tree = os.path.join(REPO, "python")
            env["PYTHONPATH"] = tree + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def race(args):
    np = _np()
    ctd = _load("classical_two_datasets")
    lane, ds = args.lane, args.dataset
    arms = [a for a in args.arms.split(",") if a]
    _load("bench_board_probe").refuse_our_cpu_arms(arms, "bench_board_more")
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.work, exist_ok=True)
    block = os.path.join(args.data, "%s-%s" % (block_of(lane), ds))
    with open(block + ".json") as fh:
        rec = json.load(fh)
    with np.load(block + ".npz") as z:
        D = lane_arrays(lane, {k: z[k] for k in z.files}, rec)
    shape = shape_desc(D)
    result = {"lane": lane, "dataset": ds, "block": rec, "shape": shape, "arms": {},
              "lane_config": LANE_CONFIG[lane], "rounds_requested": args.rounds,
              "started": now_utc(), "script": "tools/bench_board_more.py",
              "commit": os.environ.get("MOJOLEARN_REPO_COMMIT", "unknown")}
    tag = "%s-%s" % (lane, ds)
    workers = {}
    for arm in arms:
        py = args.ours_python if arm in ("ours", "ours-fast") else args.theirs_python
        cmd = shlex.split(py) + [os.path.abspath(__file__), "worker", "--arm", arm, "--lane", lane,
                                 "--dataset", ds, "--data", args.data]
        workers[arm] = ctd.Worker(arm, cmd, _worker_env(arm),
                                  os.path.join(args.out, "%s-%s.log" % (tag, arm)), REPO)
        result["arms"][arm] = {"command": cmd, "warmup_ms": None, "ms": [], "digests": [],
                               "mem": [], "status": "ok"}
    for arm, w in workers.items():
        msg = w.read(args.ready_seconds)
        if msg is None or msg.get("event") != "ready":
            w.kill("not_ready", msg)
            result["arms"][arm].update(status="not_ready", error=msg)
            print("MORE-REFUSED lane=%s dataset=%s arm=%s stage=ready detail=%s"
                  % (lane, ds, arm, json.dumps(msg)), flush=True)
            continue
        w.info = msg["info"]
        result["arms"][arm]["info"] = msg["info"]
        result["arms"][arm]["params_record"] = msg.get("params")
    # SAME SEED, SAME TUNING PARAMETERS, checked before the first timed round
    # (tools/bench_board_params.py). A refusal fails the race by name.
    if getattr(args, "params_only", False):
        return ctd.params_only_exit(result, workers, arms, lane, FAMILY,
                                    os.path.join(args.out, tag + ".params.json"))
    records = {a: result["arms"][a].get("params_record") for a in arms
               if workers[a].alive and result["arms"][a].get("params_record") is not None}
    refused = ctd.enforce_params(lane, FAMILY, records, result, "MORE")
    if refused is not None:
        for arm in arms:
            w = workers[arm]
            if w.alive:
                w.kill("params_refused", refused[:500])
                result["arms"][arm].update(status="params_refused", error=refused[:2000])
            w.log.close()
        result["finished"] = now_utc()
        out_json = os.path.join(args.out, "%s.json" % tag)
        with open(out_json + ".tmp", "w") as fh:
            json.dump(result, fh, indent=2, sort_keys=True, default=str)
        os.replace(out_json + ".tmp", out_json)
        return 3
    for r in range(args.rounds + 1):
        live = [a for a in arms if workers[a].alive]
        if not live:
            break
        shift = r % len(live)
        for arm in live[shift:] + live[:shift]:
            w = workers[arm]
            w.send("round %d" % r)
            msg = w.read(args.warmup_seconds if r == 0 else args.round_seconds)
            if msg is None or msg.get("event") != "round":
                status = "timeout" if msg is None else "error"
                w.kill(status, msg)
                result["arms"][arm].update(status=status, error=msg, failed_round=r)
                print("MORE-REFUSED lane=%s dataset=%s arm=%s stage=round%d detail=%s"
                      % (lane, ds, arm, r, json.dumps(msg)), flush=True)
                continue
            if r == 0:
                result["arms"][arm]["warmup_ms"] = msg["ms"]
            else:
                result["arms"][arm]["ms"].append(msg["ms"])
            result["arms"][arm]["digests"].append(msg["digest"])
            if msg.get("operation") is not None:
                result["arms"][arm].setdefault("operations", []).append(dict(msg["operation"], round=r, warmup=r == 0))
            if msg.get("state_receipt") is not None:
                result["arms"][arm].setdefault("state_receipts", []).append(msg["state_receipt"])
            result["arms"][arm]["mem"].append(msg.get("mem"))
            print("MORE-ROUND lane=%s dataset=%s arm=%s round=%d ms=%.3f digest=%s"
                  % (lane, ds, arm, r, msg["ms"], msg["digest"]), flush=True)
    outs = {}
    for arm in arms:
        w = workers[arm]
        if w.alive and len(result["arms"][arm]["ms"]) == args.rounds:
            path = os.path.join(args.work, "%s-%s.npz" % (tag, arm))
            w.send("save %s" % path)
            msg = w.read(args.round_seconds)
            if msg is not None and msg.get("event") == "saved":
                with np.load(path) as z:
                    outs[arm] = {k: z[k] for k in z.files}
                os.remove(path)
            else:
                result["arms"][arm].update(status="save_failed", error=msg)
        w.close()
    try:
        result["quality"] = quality(lane, D, outs)
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        result["quality"] = {"error": repr(exc)}
    for arm in arms:
        a = result["arms"][arm]
        ok = a["status"] == "ok" and len(a["ms"]) == args.rounds
        a["median_ms"] = statistics.median(a["ms"]) if ok else None
        timed = a["digests"][1:]
        a["digest_stable"] = (len(set(timed)) == 1) if ok and len(timed) >= 2 else None
        info = a.get("info") or {}
        home = info.get("input_home") or ("device" if info.get("library") in ("cuml", "cuvs")
                                          or "upload_ms_untimed" in info else "host")
        a["span"] = {"input_home": home, "pre_clock_fit": bool(info.get("pre_clock_fit")),
                     "upload_ms_untimed": info.get("upload_ms_untimed"),
                     "inside_clock": LANE_CONFIG[lane]["timed"]}
        print("MORE lane=%s dataset=%s arm=%s status=%s median_ms=%s quality=%s"
              % (lane, ds, arm, a["status"], a["median_ms"],
                 json.dumps(result["quality"].get(arm, {}), sort_keys=True, default=str)), flush=True)
    result["finished"] = now_utc()
    out_json = os.path.join(args.out, "%s.json" % tag)
    with open(out_json + ".tmp", "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True, default=str)
    os.replace(out_json + ".tmp", out_json)
    failed = [a for a in arms if result["arms"][a]["status"] != "ok"]
    return 1 if failed and len(failed) == len(arms) else 0


def build_parser():
    p = argparse.ArgumentParser(prog="bench_board_more", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("prep")
    pr.add_argument("--data", required=True)
    pr.add_argument("--lanes", default=",".join(LANE_ORDER))
    pr.add_argument("--datasets", default="taxi,istella")
    pr.add_argument("--max-rows", type=int, default=0,
                    help="SMOKE shape: cap every row count near this (0 = the lane shapes)")
    r = sub.add_parser("race")
    r.add_argument("--lane", required=True, choices=LANE_ORDER)
    r.add_argument("--dataset", required=True)
    r.add_argument("--data", required=True)
    r.add_argument("--arms", required=True)
    r.add_argument("--rounds", type=int, default=5)
    r.add_argument("--out", required=True)
    r.add_argument("--work", required=True)
    r.add_argument("--ours-python", default=sys.executable)
    r.add_argument("--theirs-python", default=sys.executable)
    r.add_argument("--ready-seconds", type=float, default=1800)
    r.add_argument("--params-only", action="store_true",
                   help="construct every arm, read its parameters back, write <out>/<tag>.params.json "
                        "and stop before the warm-up (the board's opponent-store lookup)")
    r.add_argument("--warmup-seconds", type=float, default=1800)
    r.add_argument("--round-seconds", type=float, default=1800)
    w = sub.add_parser("worker")
    w.add_argument("--arm", required=True)
    w.add_argument("--lane", required=True, choices=LANE_ORDER)
    w.add_argument("--dataset", required=True)
    w.add_argument("--data", required=True)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.cmd == "prep":
        return prep(args)
    if args.cmd == "worker":
        return worker(args)
    if args.rounds < 1:
        raise SystemExit("--rounds must be >= 1")
    return race(args)


if __name__ == "__main__":
    sys.exit(main())
