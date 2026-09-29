#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Classical lanes on the two benchmark datasets (CONTRIBUTING.md (Performance claims), DEVIATION 2570): OUR IDENTICAL arm beside scikit-learn on every
CPU core and a torch GPU arm, interleaved round by round, quality beside
every cell.

    python3 tools/classical_two_datasets.py prep --data DIR \\
        [--lanes kmeans,pca,ols,knn,kde,svc] [--datasets taxi,istella]
    python3 tools/classical_two_datasets.py race --data DIR --out DIR \\
        --lane kmeans --dataset taxi \\
        --ours-python "pixi run python3" --theirs-python /root/ctd-venv/bin/python
    python3 tools/classical_two_datasets.py summary --out DIR

This is THE timing path for kmeans, pca, ols, knn, kde and svc from
2026-09-11. The splitmix64 fixtures of `bench/bench_main.mojo`,
`bench/speed/classical_speed_main.mojo`, `tools/speed_cuml_arm.py` and
`bench/bench_sklearn.py` stay as correctness and smoke fixtures; a timing or a
default flip quotes this file's two real datasets only.

THE DATA (prep, untimed, once per box)
======================================
Read through the trees harness's own loaders, `tools/speed_gbdt_arm.py::
load_taxi` and `::load_istella` (their NumPy caches, built by `--download`),
and for kNN through `tools/knn_datasets.py::real_block` (DEVIATION 2524), so
no decoding is duplicated and the column selection is the trees lane's
`TAXI_NUMERIC`. Every block is written ONCE as an uncompressed `.npz` plus a
JSON record (shapes, sha256 of every array, rows taken, what was cleaned),
and every arm of every round reads those same bytes.

    block   lanes            taxi (11 numeric columns)       Istella-S (220)
    big     kmeans pca ols   train rows [0, 4,000,000)        train split, all
                             eval: the 500,000 test rows      2,043,304 rows
                                                              eval: 500,000 test
    knn     knn              real_block: index [0, 400,000), queries
                             [400,000, 404,000), k = 10
    kde     kde              100,000 fit rows, 2,000 queries, stride-sampled
    svc     svc              10,000 fit rows, 10,000 eval rows, stride-sampled

THE SHAPES, one line each:
  kmeans, pca, ols  4,000,000 rows is section 9's shape; Istella-S stops at
                    its cached train split, 2,043,304 rows (the cache holds
                    train plus 500,000 test rows; the 3.4M count includes the
                    validation file, which the trees cache does not decode).
  knn               DEVIATION 2524's shape, unchanged.
  kde               the score is O(n_train x n_query x d) with no fit, so
                    2e8 kernel evaluations is where our device time is the
                    pairwise kernel and not the launch, and it is the largest
                    shape at which scikit-learn's single-threaded tree score
                    (rtol = atol = 0, no pruning) finishes six rounds inside
                    one lease.
  svc               SMO work grows with n^2 kernel rows; at 10,000 rows the
                    kernel rows are the cost on the device, and scikit-learn's
                    single-threaded libsvm finishes six fits inside one lease.

WHAT IS CLEANED, IDENTICALLY FOR EVERY ARM:
  * Istella's missing-value sentinel (float32 max, `_decode_letor`) becomes
    0.0 in every block. A float32-max cell overflows a squared distance or a
    Gram entry to inf on EVERY library, so leaving it would time inf
    arithmetic. The count replaced is in the record (0 means no-op). Taxi's
    -1 missing marker is finite and stays.
  * kde and svc are STANDARDIZED with the fit rows' float64 column mean and
    standard deviation (a zero deviation divides by 1). A Gaussian kernel on
    raw Istella columns spanning seven orders of magnitude is exp(-huge) = 0
    for every pair; any user scales first. kmeans, pca, ols and knn run on the
    raw columns (knn exactly as DEVIATION 2524 reads them).
  * kde's bandwidth is Scott's rule on the standardized fit rows,
    n ** (-1 / (d + 4)); svc's gamma is 1 / d (scikit-learn's 'scale' on
    standardized data, and ours' 'auto').

THE ARMS (0b-iii: only OUR IDENTICAL arm against the opponent's FAST arm)
=========================================================================
  ours          mojolearn's public estimator under MOJOLEARN_NUMERIC_MODE=
                identical, read back by `numeric_mode_used()`. Timed from a
                host float32 array to host results: its upload is INSIDE the
                clock, because the public call uploads.
  sklearn-cpu   scikit-learn on every core as installed: no OMP_NUM_THREADS,
                OPENBLAS_NUM_THREADS or MKL_NUM_THREADS, no n_jobs cap
                (n_jobs=-1 where the estimator takes one). The ready record
                names the BLAS and OpenMP pools threadpoolctl reports.
  cuml-gpu      RAPIDS cuML on NVIDIA (the GPU library on the box is the
                opponent, DEVIATION 2571): KMeans, PCA(svd_solver='full'),
                LinearRegression(algorithm='eig'), NearestNeighbors(brute),
                KernelDensity and SVC, output_type='cupy'. Their FAST arm:
                inputs are cupy arrays uploaded BEFORE the clock (upload_ms in
                the ready record), kNN and KDE fit before the clock (kneighbors
                and score_samples timed), the clock ends at
                cupy.cuda.runtime.deviceSynchronize().
  torch-gpu     PyTorch on the box's GPU (ROCm on the MI325X, CUDA on NVIDIA),
                written the way a competent user writes it: Lloyd iterations
                with a chunked addmm assignment; PCA by covariance eigh; OLS
                by torch.linalg.lstsq on centered data (on CUDA its only
                driver is gels, QR without pivoting, which assumes full rank);
                brute-force kNN by torch.cdist plus topk per 1024-query chunk.
                Their FAST arm: the inputs are uploaded BEFORE the clock
                (upload_ms is in the ready record), the clock ends at
                torch.cuda.synchronize(). Not written for kde and svc.
  torch-gpu-eigh  OLS only: centered normal equations through eigh with a
                pseudo-inverse cutoff (the algorithm class of ours and of
                cuML's algorithm='eig'; Istella has near-constant columns, so
                it is kept beside the lstsq arm).

INTERLEAVING. One worker process per arm holds the data and its library for
the whole (lane, dataset); the conductor sends `round r` to each arm in turn,
the order rotated every round, one warm-up (round 0, excluded) and
`--rounds` timed rounds. A worker's protocol lines go to a duplicated file
descriptor and its fd 1 is pointed at its log, so a library that prints
cannot corrupt the protocol.

QUALITY is computed HERE, in the conductor, from each arm's saved outputs,
by one float64 NumPy function per lane, never by the libraries' own scorers:
  kmeans  inertia of the final centroids over the fit rows, and n_iter
  pca     explained-variance ratio of the 8 components over the fit rows
  ols     R2 and RMSE on the eval rows
  knn     tie-aware recall@10 against a float64 NumPy brute force computed
          here (a neighbour counts when its float64 distance is within the
          reference's 10th distance), distinct ids only
  (kmeans inertia is also given as a ratio over ours)
  kde     mean log-likelihood of the queries
  svc     accuracy on the eval rows

OUTPUT under --out: <lane>-<dataset>.json (every round, every hash, the
ready records, quality, ratios), <lane>-<dataset>-<arm>.log (worker stderr),
CTD lines on stdout, and `summary` writes summary.tsv. Ratios are OURS
median / OPPONENT median per cell (below 1 means our median is lower); a
ratio is never a sentence here.

CTD-SPAN, one per arm, says what was INSIDE that arm's clock and what was
outside it: `input_home` (a device-resident input was uploaded before the
clock started), `upload_ms_untimed`, `fit_ms_untimed`, `pre_clock_fit`.
CTD-RATIO carries a `span_asymmetry=` field naming every opponent whose clock
covers less work than ours. Every term of it was already being recorded in the
ready records and none of it used to reach the line a reader reads. Ours
uploads and validates INSIDE its clock and the GPU opponents do not, which on
Istella-S KDE is roughly 19.7 ms of host work at d = 220 (7.28 ms of it
validation) against 36.9 ms of device entry -- an asymmetry that runs AGAINST
us. Nothing is equalized: no work moves into or out of any clock and no ratio
changes. The point is that a ratio spanning two different amounts of work
cannot be read as like-for-like without saying so.

knn and kde fit BEFORE the clock on EVERY arm, ours included (each worker fits
its index or its density at construction and times kneighbors or
score_samples), so `pre_clock_fit` is declared true on ours there too and
`fit_before_its_clock` no longer marks those races. What remains marked is
the device-resident input of the torch and cuML arms (their FAST arm, by the
board's rule).

SAME SEED, SAME TUNING PARAMETERS (2026-09-29): every arm gets seed 7 through
its library's own argument and every parameter the libraries share is set
explicitly; CONFIG names what cannot be matched. Each worker sends the
read-back of what it constructed and `race` runs tools/bench_board_params.py's
check before the first timed round; a refusal fails the race by name (exit 3,
`params_refused` in the JSON).
"""
import argparse
import hashlib
import importlib.util
import json
import os
import select
import shlex
import signal
import statistics
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

LANES = ("kmeans", "pca", "ols", "knn", "kde", "svc", "dbscan", "hdbscan")
DATASETS = ("taxi", "istella")
#: Every arm a lane has. A leg names the ones it races with --arms (NVIDIA:
#: ours, cuml-gpu and torch; AMD: ours, torch ROCm and scikit-learn on the CPU).
ARMS = {
    "kmeans": ("ours", "cuml-gpu", "torch-gpu", "sklearn-cpu"),
    "pca": ("ours", "cuml-gpu", "torch-gpu", "sklearn-cpu"),
    "ols": ("ours", "cuml-gpu", "torch-gpu", "torch-gpu-eigh", "sklearn-cpu"),
    "knn": ("ours", "cuml-gpu", "torch-gpu", "sklearn-cpu"),
    "kde": ("ours", "cuml-gpu", "sklearn-cpu"),
    "svc": ("ours", "cuml-gpu", "sklearn-cpu"),
    # Lane linear-cluster-istella (2026-09-11): DBSCAN at 1,000,000 rows,
    # measurement only. `cuml-gpu` is cuML's default eps search ('brute'),
    # `cuml-gpu-rbc` its random ball cover; ours runs its default ('rbc').
    # HDBSCAN: ours (mojolearn.HDBSCAN, shipped since 0.8.x), cuML on NVIDIA,
    # scikit-learn on the CPU, every arm with the SAME min_samples,
    # min_cluster_size, metric and cluster selection (CumlHDBSCAN's).
    "dbscan": ("ours", "cuml-gpu", "cuml-gpu-rbc", "sklearn-cpu"),
    "hdbscan": ("ours", "cuml-gpu", "sklearn-cpu"),
}
BLOCK_OF = {"kmeans": "big", "pca": "big", "ols": "big", "knn": "knn",
            "kde": "kde", "svc": "svc", "dbscan": "dbscan", "hdbscan": "dbscan"}
#: DBSCAN block rows and parameters. eps and min_samples come from the
#: environment (MOJOLEARN_CTD_DBSCAN_<DATASET>="eps,min_samples"), the SAME
#: value for every arm, chosen per dataset by the lane's rule and recorded in
#: each race's arm info; the block itself is standardized like kde and svc.
DBSCAN_ROWS = 1_000_000
HDBSCAN_ROWS = 100_000

BIG_ROWS = 4_000_000
#: The board's one seed (tools/bench_board.py SEED), given to every arm through
#: the library's own seed argument; an arm without one is named in CONFIG.
SEED = 7
KMEANS_K = 64
KMEANS_ITER = 20
#: One tol on every arm that takes one. ours and cuML refuse tol <= 0
#: ("invalid parameter (tol<=0)"), so 0.0 cannot be the shared value.
KMEANS_TOL = 1e-7
PCA_COMPONENTS = 8
KNN_K = 10
KDE_TRAIN = 100_000
KDE_QUERY = 2_000
KDE_LEAF_SIZE = 40
SVC_TRAIN = 10_000
SVC_EVAL = 10_000
SVC_C = 1.0
SVC_TOL = 1e-3
SVC_CACHE_MB = 2000.0
SVC_DEGREE = 3
SVC_COEF0 = 0.0
SVC_MAX_ITER = -1
TORCH_CHUNK_ROWS = 1 << 20
TORCH_KNN_QUERY_CHUNK = 1024

#: What every arm of a lane is set to and every mismatch that could not be
#: removed, one line each (the classical2 driver's LANE_CONFIG shape). `race`
#: writes it into the race JSON as `lane_config` and prints it as CTD-SETTINGS
#: and CTD-MISMATCH lines. Rule: same seed and same value of every parameter
#: both sides have; what exists on one side only, or cannot be matched, is
#: named here with each side's value.
CONFIG = {
    "kmeans": {
        "rows": "big block: 4,000,000 taxi rows or the Istella-S train split, raw",
        "params": "n_clusters=64, init=<the same 64 block rows on every arm>, n_init=1, "
                  "max_iter=20, tol=1e-7, metric='euclidean', Lloyd; seed 7 (ours, scikit-learn "
                  "and cuML random_state=7, torch.manual_seed(7))",
        "timed": "fit",
        "mismatches": [
            "tol=1e-7 on ours, scikit-learn and cuML (ours and cuML refuse 0); each library "
            "applies it through its own convergence test. torch-gpu has no tol and always runs "
            "20 iterations. n_iter is in every quality cell",
            "algorithm: scikit-learn algorithm='lloyd'; ours and cuML have no such parameter "
            "(Lloyd); torch-gpu is written as Lloyd",
            "seed: with init given as an array no arm draws a random number; 7 is set anyway",
        ],
    },
    "pca": {
        "rows": "big block, raw",
        "params": "n_components=8, whiten=False, random_state=7; ours and scikit-learn "
                  "svd_solver='covariance_eigh'",
        "timed": "fit",
        "mismatches": [
            "svd_solver: cuML has no 'covariance_eigh'; cuml-gpu runs svd_solver='full'",
            "torch-gpu: no estimator; covariance eigh written out, torch.manual_seed(7) (it "
            "draws nothing)",
            "random_state is read by none of the covariance solvers; set to 7 on every arm",
        ],
    },
    "ols": {
        "rows": "big block, raw; R2 and RMSE on the 500,000 eval rows",
        "params": "fit_intercept=True",
        "timed": "fit",
        "mismatches": [
            "seed: no arm has a seed argument (closed-form fit); torch-gpu torch.manual_seed(7)",
            "solver: ours eig of the normal equations (no parameter), scikit-learn scipy lstsq "
            "gelsd (no parameter), cuML algorithm='eig', torch-gpu torch.linalg.lstsq (gels on "
            "CUDA), torch-gpu-eigh eigh with a pseudo-inverse cutoff",
            "scikit-learn positive=False and copy_X=True: parameters ours does not have",
        ],
    },
    "knn": {
        "rows": "knn block: 400,000 index rows, 4,000 queries, raw",
        "params": "n_neighbors=10, metric='euclidean', algorithm='brute' (ours, scikit-learn, "
                  "cuML); torch cdist p=2 plus topk",
        "timed": "kneighbors; the fit (index) is before the clock on every arm",
        "mismatches": [
            "seed: no arm has a seed argument (exact search); torch-gpu torch.manual_seed(7)",
            "query_tile: ours only (tiling, results unchanged); n_jobs=-1: scikit-learn only",
        ],
    },
    "kde": {
        "rows": "kde block: 100,000 fit rows, 2,000 queries, standardized",
        "params": "bandwidth=Scott's rule, kernel='gaussian', metric='euclidean' on every arm; "
                  "ours and scikit-learn atol=0, rtol=0, algorithm='auto', leaf_size=40, "
                  "breadth_first=True",
        "timed": "score_samples; the fit is before the clock on every arm",
        "mismatches": [
            "seed: no arm has a seed argument (exact density)",
            "cuML has no atol, rtol, algorithm, leaf_size or breadth_first (exact brute force)",
        ],
    },
    "svc": {
        "rows": "svc block: 10,000 fit rows, 10,000 eval rows, standardized",
        "params": "C=1.0, kernel='rbf', gamma=1/d, tol=1e-3, degree=3, coef0=0.0, max_iter=-1, "
                  "cache_size=2000 MB, class_weight=None",
        "timed": "fit",
        "mismatches": [
            "seed: scikit-learn and cuML random_state=7 (read only with probability=True); "
            "ours refuses random_state without probability=True, so ours stays None",
            "cache_size=2000 on every arm; ours honors it only as the prediction buffer "
            "(DEVIATION 871), so its training is unaffected",
            "shrinking=True: scikit-learn only; nochange_steps=1000: ours and cuML only",
        ],
    },
    "dbscan": {
        "rows": "dbscan block: 1,000,000 rows, standardized",
        "params": "eps and min_samples from MOJOLEARN_CTD_DBSCAN_<DATASET>, the same on every "
                  "arm; metric='euclidean'",
        "timed": "fit",
        "mismatches": [
            "seed: no arm has a seed argument (deterministic)",
            "algorithm (an exact neighbor search on every arm, results unchanged): ours 'rbc' "
            "(its default), scikit-learn 'auto' (a tree on taxi, brute on Istella-S; it has no "
            "'rbc'), cuml-gpu 'brute', cuml-gpu-rbc 'rbc'",
            "leaf_size=30 and n_jobs=-1: scikit-learn only",
        ],
    },
    "hdbscan": {
        "rows": "the dbscan block's first 100,000 rows",
        "params": "min_samples=10, min_cluster_size=100, metric='euclidean', "
                  "cluster_selection_method='eom', cluster_selection_epsilon=0.0, alpha=1.0, "
                  "allow_single_cluster=False",
        "timed": "fit",
        "mismatches": [
            "seed: no arm has a seed argument (deterministic)",
            "max_cluster_size: ours and cuML 0, scikit-learn None (both mean no limit)",
            "scikit-learn algorithm='auto', leaf_size=40, n_jobs=-1: its own; cuML build_algo "
            "at its default",
        ],
    },
}


def construct_tolerant(cls, kw, info):
    """`(cls(**kw), kw)`. A keyword this build of an OPPONENT library does not
    take is dropped and named in info['params_not_in_this_build'] with the
    value it would have had, never silently. Ours never goes through here: a
    parameter our wheel refuses fails the arm by name."""
    kw = dict(kw)
    while True:
        try:
            return cls(**kw), kw
        except TypeError as exc:
            msg = str(exc)
            bad = [k for k in kw if "'%s'" % k in msg]
            if not bad:
                raise
            for k in bad:
                info.setdefault("params_not_in_this_build", {})[k] = repr(kw.pop(k))

THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _module(name):
    """A tools/ module by file path (the trees harness and the kNN loader
    import stdlib and numpy only at import time)."""
    path = os.path.join(HERE, name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha256_array(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.data)
    return h.hexdigest()


def stride_rows(n, m):
    """`m` evenly spaced row indices over [0, n), deterministic."""
    m = min(m, n)
    return (np.arange(m, dtype=np.int64) * n) // m


def clean_sentinel(x):
    """Istella's missing sentinel (float32 max) and any non-finite cell to
    0.0, in a float32 C-contiguous copy. Returns (array, cells replaced)."""
    x = np.array(x, dtype=np.float32, order="C", copy=True)
    bad = ~np.isfinite(x) | (x >= np.finfo(np.float32).max)
    count = int(bad.sum())
    if count:
        x[bad] = 0.0
    return x, count


def standardize(fit, *others):
    """Column mean and standard deviation of `fit` in float64 (zero
    deviation divides by 1), applied to `fit` and every other array,
    returned as float32 C-contiguous arrays."""
    f64 = fit.astype(np.float64)
    mu = f64.mean(axis=0)
    sd = f64.std(axis=0)
    sd[sd == 0.0] = 1.0
    out = [np.ascontiguousarray(((f64 - mu) / sd).astype(np.float32))]
    for o in others:
        out.append(np.ascontiguousarray(((o.astype(np.float64) - mu) / sd)
                                        .astype(np.float32)))
    return out


def distinct_init_rows(x, k):
    """`k` distinct rows of `x` for the k-means initial centroids: rows
    c * (n // k), walking forward past a row equal to one already taken."""
    n = x.shape[0]
    step = max(1, n // k)
    taken, rows = set(), []
    for c in range(k):
        i = c * step
        while i < n and x[i].tobytes() in taken:
            i += 1
        if i >= n:
            raise RuntimeError("fewer than %d distinct rows for the init" % k)
        taken.add(x[i].tobytes())
        rows.append(i)
    return np.ascontiguousarray(x[rows]), rows


def median(xs):
    return float(statistics.median(xs)) if xs else None


def _bp():
    """tools/bench_board_params.py, the board's one parameter check."""
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import bench_board_params
    return bench_board_params


def _array_tag(a):
    """An array-valued parameter (k-means' shared init rows) as its digest,
    so two arms given the same rows compare equal and a different array does
    not."""
    return "array sha256:%s" % sha256_array(np.asarray(_to_host(a)))[:16]


def params_record(obj):
    """What a worker sends for its arm: BP.arm_record of the object it
    constructed (or the declared dict of a function arm), array values given
    as digests. An unreadable object sends an empty record, which the check
    refuses by name (no seed read back)."""
    BP = _bp()
    if obj is None:
        return {"__record__": True, "library": "?", "source": "no parameter object", "params": {}}
    try:
        rec = BP.arm_record(obj)
        _lib, _src, raw = BP.read_params(obj)
    except Exception as exc:  # noqa: BLE001
        return {"__record__": True, "library": "?", "source": "unreadable (%r)" % (exc,),
                "params": {}}
    for k, v in raw.items():
        if k in rec["params"] and getattr(v, "ndim", 0) >= 1 and hasattr(v, "shape"):
            rec["params"][k] = _array_tag(v)
    # ours takes k-means' shared rows as init='array' plus init_centroids; the
    # opponents take the array as init. Both are the same rows.
    if rec["library"] == "mojolearn" and rec["params"].get("init") == "array" \
            and isinstance(rec["params"].get("init_centroids"), str):
        rec["params"]["init"] = rec["params"]["init_centroids"]
    return rec


def enforce_params(lane, family, records, result, tag):
    """BP.enforce over the arms that came up. Returns the refusal text, or
    None when every arm matched. The report goes into the race JSON either way."""
    BP = _bp()
    try:
        result["params_check"] = BP.enforce(lane, records, family=family)
        return None
    except BP.ParamsRefused as exc:
        result["params_check"] = BP.check(lane, records, family=family)
        result["params_refused"] = str(exc)
        print("%s-PARAMS-REFUSED lane=%s %s" % (tag, lane, exc), flush=True)
        return str(exc)


def now_utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------------------
# prep: the blocks, written once
# ---------------------------------------------------------------------------

def _taxi_numeric(harness, x):
    cols = [harness.TAXI_FEATURES.index(c) for c in harness.TAXI_NUMERIC]
    return np.ascontiguousarray(x[:, cols])


def _write_block(data_dir, name, arrays, record):
    path = os.path.join(data_dir, name + ".npz")
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)
    record["arrays"] = {k: {"shape": list(v.shape), "dtype": str(v.dtype),
                            "sha256": sha256_array(v)}
                        for k, v in arrays.items()}
    record["written"] = now_utc()
    with open(os.path.join(data_dir, name + ".json"), "w") as fh:
        json.dump(record, fh, indent=2, sort_keys=True, default=str)
    print("CTD-PREP block=%s %s" % (name, " ".join(
        "%s=%s" % (k, "x".join(str(s) for s in v.shape))
        for k, v in arrays.items())), flush=True)


def prep(args):
    lanes = [l for l in args.lanes.split(",") if l]
    datasets = [d for d in args.datasets.split(",") if d]
    for l in lanes:
        if l not in LANES:
            raise SystemExit("unknown lane %r; choose from %s" % (l, LANES))
    for d in datasets:
        if d not in DATASETS:
            raise SystemExit("unknown dataset %r; choose from %s" % (d, DATASETS))
    os.makedirs(args.data, exist_ok=True)
    harness = _module("speed_gbdt_arm")
    blocks = sorted({BLOCK_OF[l] for l in lanes})
    # --max-rows: the SMOKE shape (a harness bug costs minutes, not a lease).
    # Every block is capped; a smoke block is never a timing row.
    cap = int(args.max_rows) if args.max_rows else None
    big_rows = min(BIG_ROWS, cap) if cap else BIG_ROWS
    eval_cap = cap
    knn_index = min(400_000, cap) if cap else 400_000
    knn_queries = min(4_000, max(64, cap // 50)) if cap else 4_000
    kde_train = min(KDE_TRAIN, cap) if cap else KDE_TRAIN
    kde_query = min(KDE_QUERY, max(64, cap // 100)) if cap else KDE_QUERY
    svc_train = min(SVC_TRAIN, max(256, cap // 10)) if cap else SVC_TRAIN
    svc_eval = min(SVC_EVAL, max(256, cap // 10)) if cap else SVC_EVAL
    for ds in datasets:
        t0 = time.perf_counter()
        base = {"dataset": ds, "data_root": harness.data_root(),
                "rule": "CONTRIBUTING.md (Performance claims); DEVIATION 2570",
                "smoke_max_rows": cap}
        reg = None
        if "big" in blocks or "kde" in blocks:
            if ds == "taxi":
                reg = harness.load_taxi("shipped", regression=True)
                xtr = _taxi_numeric(harness, reg.X_train)
                xte = _taxi_numeric(harness, reg.X_test)
                loader = "speed_gbdt_arm.load_taxi('shipped', regression=True), TAXI_NUMERIC columns"
            else:
                reg = harness.load_istella("shipped", regression=True)
                xtr, xte = reg.X_train, reg.X_test
                loader = "speed_gbdt_arm.load_istella('shipped', regression=True)"
            ytr = np.ascontiguousarray(reg.y_train, dtype=np.float32)
            yte = np.ascontiguousarray(reg.y_test, dtype=np.float32)
        if "big" in blocks:
            n = min(big_rows, xtr.shape[0])
            X, bad_x = clean_sentinel(xtr[:n])
            n_eval = min(eval_cap, xte.shape[0]) if eval_cap else xte.shape[0]
            Xq, bad_q = clean_sentinel(xte[:n_eval])
            yte = yte[:n_eval]
            init, init_rows = distinct_init_rows(X, KMEANS_K)
            rec = dict(base, block="big", lanes=["kmeans", "pca", "ols"],
                       loader=loader, fit_rows=[0, n],
                       fit_rows_available=int(xtr.shape[0]),
                       capped_below_4M=bool(n < BIG_ROWS),
                       eval_rows="the loader's test split, all %d rows" % Xq.shape[0],
                       target="fare_amount" if ds == "taxi" else "relevance grade 0..4",
                       sentinel_cells_replaced={"X": bad_x, "Xq": bad_q},
                       scaling="none",
                       kmeans={"k": KMEANS_K, "max_iter": KMEANS_ITER,
                               "init_rows": init_rows},
                       pca={"n_components": PCA_COMPONENTS})
            _write_block(args.data, "big-" + ds,
                         {"X": X, "y": ytr[:n], "Xq": Xq, "yq": yte,
                          "init": init}, rec)
        if "dbscan" in blocks:
            # Lane linear-cluster-istella: the first DBSCAN_ROWS train rows,
            # sentinel cleaned, standardized by themselves (float64 mean and
            # std), so one eps means the same thing on every column.
            if ds == "taxi":
                db_src = harness.load_taxi("shipped", regression=True)
                db_x = _taxi_numeric(harness, db_src.X_train)
                db_loader = "speed_gbdt_arm.load_taxi('shipped', regression=True), TAXI_NUMERIC columns"
            else:
                db_src = reg if reg is not None else harness.load_istella("shipped", regression=True)
                db_x = db_src.X_train
                db_loader = "speed_gbdt_arm.load_istella('shipped', regression=True)"
            n_db = min(DBSCAN_ROWS, cap, db_x.shape[0]) if cap else min(DBSCAN_ROWS, db_x.shape[0])
            Xd, bad_x = clean_sentinel(db_x[:n_db])
            (Xd,) = standardize(Xd)
            rec = dict(base, block="dbscan", lanes=["dbscan", "hdbscan"], loader=db_loader,
                       fit_rows=[0, n_db], fit_rows_available=int(db_x.shape[0]),
                       sentinel_cells_replaced={"X": bad_x},
                       scaling="standardized by these rows (float64 mean and std)",
                       dbscan={"rows": n_db,
                               "params_env": "MOJOLEARN_CTD_DBSCAN_%s" % ds.upper()},
                       hdbscan={"rows": min(HDBSCAN_ROWS, n_db)})
            _write_block(args.data, "dbscan-" + ds, {"X": Xd}, rec)
            db_src = db_x = Xd = None
        if "kde" in blocks:
            fit_idx = stride_rows(xtr.shape[0], kde_train)
            q_idx = stride_rows(xte.shape[0], kde_query)
            Xf, bad_x = clean_sentinel(xtr[fit_idx])
            Xq, bad_q = clean_sentinel(xte[q_idx])
            Xf, Xq = standardize(Xf, Xq)
            d = Xf.shape[1]
            bw = float(Xf.shape[0] ** (-1.0 / (d + 4)))
            rec = dict(base, block="kde", lanes=["kde"], loader=loader,
                       fit_rows="stride sample of %d of the train split's %d rows" % (Xf.shape[0], xtr.shape[0]),
                       query_rows="stride sample of %d of the test split's %d rows" % (Xq.shape[0], xte.shape[0]),
                       sentinel_cells_replaced={"X": bad_x, "Xq": bad_q},
                       scaling="standardized by the fit rows (float64 mean and std)",
                       kde={"bandwidth": bw, "bandwidth_rule": "scott n**(-1/(d+4))",
                            "kernel": "gaussian", "metric": "euclidean"})
            _write_block(args.data, "kde-" + ds, {"X": Xf, "Xq": Xq}, rec)
        if "svc" in blocks:
            if ds == "taxi":
                clf = harness.load_taxi("shipped", regression=False)
                ctr = _taxi_numeric(harness, clf.X_train)
                cte = _taxi_numeric(harness, clf.X_test)
                cy_tr, cy_te = clf.y_train, clf.y_test
                cloader = "speed_gbdt_arm.load_taxi('shipped', regression=False) (card trips, tip >= 20%), TAXI_NUMERIC columns"
            else:
                if reg is None:
                    reg = harness.load_istella("shipped", regression=True)
                ctr, cte = reg.X_train, reg.X_test
                cy_tr = (reg.y_train > 0).astype(np.float32)
                cy_te = (reg.y_test > 0).astype(np.float32)
                cloader = "speed_gbdt_arm.load_istella('shipped', regression=True), label relevance > 0"
            fit_idx = stride_rows(ctr.shape[0], svc_train)
            ev_idx = stride_rows(cte.shape[0], svc_eval)
            Xf, bad_x = clean_sentinel(ctr[fit_idx])
            Xq, bad_q = clean_sentinel(cte[ev_idx])
            Xf, Xq = standardize(Xf, Xq)
            y = np.ascontiguousarray(np.asarray(cy_tr)[fit_idx], dtype=np.float32)
            yq = np.ascontiguousarray(np.asarray(cy_te)[ev_idx], dtype=np.float32)
            if len(np.unique(y)) != 2:
                raise RuntimeError("svc-%s: the fit rows do not hold both classes" % ds)
            rec = dict(base, block="svc", lanes=["svc"], loader=cloader,
                       fit_rows="stride sample of %d of %d train rows" % (Xf.shape[0], ctr.shape[0]),
                       eval_rows="stride sample of %d of %d test rows" % (Xq.shape[0], cte.shape[0]),
                       positive_fraction_fit=float(y.mean()),
                       sentinel_cells_replaced={"X": bad_x, "Xq": bad_q},
                       scaling="standardized by the fit rows (float64 mean and std)",
                       svc={"C": SVC_C, "kernel": "rbf", "gamma": 1.0 / Xf.shape[1],
                            "tol": SVC_TOL})
            _write_block(args.data, "svc-" + ds, {"X": Xf, "y": y, "Xq": Xq, "yq": yq}, rec)
        if "knn" in blocks:
            knn_ds = _module("knn_datasets")
            blk = knn_ds.real_block(ds, n_index=knn_index, n_queries=knn_queries)
            index, bad_i = clean_sentinel(blk["index"])
            queries, bad_q = clean_sentinel(blk["queries"])
            src = {k: v for k, v in blk["source"].items()}
            rec = dict(base, block="knn", lanes=["knn"],
                       loader="knn_datasets.real_block(%r) (DEVIATION 2524)" % ds,
                       index_rows=blk["index_rows"], query_rows=blk["query_rows"],
                       real_block_sha256={"block": blk["sha256_block"],
                                          "index": blk["sha256_index"],
                                          "queries": blk["sha256_queries"]},
                       real_block_source=src,
                       sentinel_cells_replaced={"index": bad_i, "queries": bad_q},
                       scaling="none", knn={"k": KNN_K})
            _write_block(args.data, "knn-" + ds, {"index": index, "queries": queries}, rec)
        reg = None
        print("CTD-PREP dataset=%s seconds=%.1f" % (ds, time.perf_counter() - t0), flush=True)
    return 0


# ---------------------------------------------------------------------------
# worker: one arm, one (lane, dataset), driven over a pipe
# ---------------------------------------------------------------------------

def _cpu_model():
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    import platform
    return platform.processor() or platform.machine()


def _host_info():
    info = {"cpu_model": _cpu_model(), "os_cpu_count": os.cpu_count(),
            "thread_env": {k: os.environ.get(k) for k in THREAD_ENV}}
    try:
        info["sched_affinity_cpus"] = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        info["sched_affinity_cpus"] = None
    info["numpy"] = np.__version__
    info["python"] = sys.version.split()[0]
    return info


#: "cuda" (CUDA or ROCm, both spelled torch.cuda) or "mps" (Apple silicon),
#: set by `_torch_setup` in the worker that owns the torch arm.
_TORCH_KIND = "cuda"


def _torch_sync():
    import torch
    if _TORCH_KIND == "mps":
        torch.mps.synchronize()
    else:
        torch.cuda.synchronize()


def _to_host(a):
    """Host NumPy copy of a torch tensor, cupy array, mojolearn Array or
    NumPy array."""
    if hasattr(a, "detach"):
        return a.detach().cpu().numpy()
    if type(a).__module__.split(".")[0] == "cupy":
        return a.get()
    if hasattr(a, "copy_to_host"):
        # cuVS/pylibraft device_ndarray: np.array() of it reads host garbage
        # (measured 2026-09-27: every cuvs-gpu recall read ~0 through it)
        return np.asarray(a.copy_to_host())
    return np.array(a, copy=True)


# ---- ours ----------------------------------------------------------------

def _probe():
    """tools/bench_board_probe.py (memory per round, the ours-cpu readback)."""
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import bench_board_probe
    return bench_board_probe


def _ours_module():
    os.environ.setdefault("MOJOLEARN_NUMERIC_MODE", "identical")
    import mojolearn
    return mojolearn


def _expected_ours_mode():
    """The tier this `ours` worker was started for: `identical` for `ours`
    and `ours-base`, `fast` for `ours-fast` (the Apple board's FAST arm,
    tools/bench_board.py). `_worker_env` sets it; the readback must match."""
    return os.environ.get("MOJOLEARN_NUMERIC_MODE", "identical").strip().lower()


def _ours_info(ml, est):
    info = {"library": "mojolearn", "version": getattr(ml, "__version__", "unknown"),
            "numeric_mode_env": os.environ.get("MOJOLEARN_NUMERIC_MODE")}
    for name in ("numeric_mode_used", "vendor_used"):
        try:
            info[name] = getattr(est, name)()
        except Exception as exc:  # noqa: BLE001
            info[name] = "unavailable (%r)" % (exc,)
    info["device"] = "gpu"
    # an ours-cpu worker: the wheel must have loaded its CPU set (refuses by name)
    info.update(_probe().ours_cpu_check(ml))
    # Which mojolearn answered: the installed wheel or an in-repo tree.
    info["module_path"] = getattr(ml, "__file__", None)
    # Ours fits INSIDE its clock: the public call is what is timed, upload and
    # host validation included. Declared rather than left absent so
    # `_span_facts` never has to guess it from prose (see the note there).
    info["pre_clock_fit"] = False
    want = _expected_ours_mode()
    if info.get("numeric_mode_used") != want:
        raise RuntimeError("ours is not %s: numeric_mode_used() = %r"
                           % (want.upper(), info.get("numeric_mode_used"),))
    return info


class OursKMeans:
    def __init__(self, data, rec):
        self.ml = _ours_module()
        self.X, self.init = data["X"], data["init"]
        self.est = None
        self.params_obj = self.make()
        self.info = _ours_info(self.ml, self.params_obj)
        # The binding refuses tol = 0 with a plain Exception('invalid
        # parameter (tol<=0)') (RunPod MI300X, 2026-09-11), so every arm now
        # takes KMEANS_TOL instead of ours alone falling back to it.
        self.info["config"] = ("mojolearn.KMeans(n_clusters=64, init='array', n_init=1, "
                               "max_iter=20, tol=%g, metric='euclidean', random_state=%d)"
                               % (KMEANS_TOL, SEED))

    def make(self):
        return self.ml.KMeans(n_clusters=KMEANS_K, init="array", n_init=1,
                              max_iter=KMEANS_ITER, tol=KMEANS_TOL, metric="euclidean",
                              random_state=SEED, init_centroids=self.init)

    def call(self):
        est = self.make()
        est.fit(self.X)
        self.est = est

    def sync(self):
        pass

    def outputs(self):
        return {"centers": np.array(self.est.cluster_centers_, dtype=np.float32),
                "labels": np.array(self.est.labels_, dtype=np.int32),
                "n_iter": np.array([int(self.est.n_iter_)], dtype=np.int64)}


class OursPCA:
    def __init__(self, data, rec):
        self.ml = _ours_module()
        self.X = data["X"]
        self.est = None
        self.params_obj = self.make()
        self.info = _ours_info(self.ml, self.params_obj)

    def make(self):
        return self.ml.PCA(n_components=PCA_COMPONENTS, svd_solver="covariance_eigh",
                           whiten=False, random_state=SEED)

    def call(self):
        est = self.make()
        est.fit(self.X)
        self.est = est

    def sync(self):
        pass

    def outputs(self):
        return {"components": np.array(self.est.components_, dtype=np.float32),
                "explained_variance": np.array(self.est.explained_variance_, dtype=np.float32),
                "mean": np.array(self.est.mean_, dtype=np.float32)}


class OursOLS:
    def __init__(self, data, rec):
        self.ml = _ours_module()
        self.X, self.y = data["X"], data["y"]
        self.est = None
        self.params_obj = self.ml.LinearRegression(fit_intercept=True)
        self.info = _ours_info(self.ml, self.params_obj)

    def call(self):
        est = self.ml.LinearRegression(fit_intercept=True)
        est.fit(self.X, self.y)
        self.est = est

    def sync(self):
        pass

    def outputs(self):
        return {"coef": np.array(self.est.coef_, dtype=np.float64),
                "intercept": np.array([float(self.est.intercept_)], dtype=np.float64)}


class OursKNN:
    def __init__(self, data, rec):
        self.ml = _ours_module()
        self.q = data["queries"]
        self.nn = self.ml.NearestNeighbors(n_neighbors=KNN_K, metric="euclidean",
                                           algorithm="brute", p=2)
        self.nn.fit(data["index"])
        self.index = data["index"]
        self.out = None
        self.info = _ours_info(self.ml, self.nn)
        self.params_obj = self.nn
        # Declared where it becomes true: our fit above runs before the clock,
        # exactly as every opponent's kNN fit does; `kneighbors` alone is timed.
        self.info["pre_clock_fit"] = True

    def call(self):
        self.out = self.nn.kneighbors(self.q)

    def sync(self):
        pass

    def outputs(self):
        dist, ind = self.out
        return {"ind": np.array(ind, dtype=np.int64),
                "dist": np.array(dist, dtype=np.float32)}


class OursKDE:
    def __init__(self, data, rec):
        self.ml = _ours_module()
        self.q = data["Xq"]
        self.kd = self.ml.KernelDensity(bandwidth=rec["kde"]["bandwidth"], kernel="gaussian",
                                        metric="euclidean", algorithm="auto", atol=0.0, rtol=0.0,
                                        breadth_first=True, leaf_size=KDE_LEAF_SIZE)
        self.kd.fit(data["X"])
        self.X = data["X"]
        self.scores = None
        self.info = _ours_info(self.ml, self.kd)
        self.params_obj = self.kd
        # Our fit above is before the clock, as scikit-learn's tree build and
        # cuML's fit are; `score_samples` alone is timed on every arm.
        self.info["pre_clock_fit"] = True

    def call(self):
        self.scores = self.kd.score_samples(self.q)

    def sync(self):
        pass

    def outputs(self):
        return {"scores": np.array(self.scores, dtype=np.float64)}


class OursSVC:
    def __init__(self, data, rec):
        self.ml = _ours_module()
        self.X, self.y, self.Xq = data["X"], data["y"], data["Xq"]
        self.gamma = float(rec["svc"]["gamma"])
        self.est = None
        probe = self.make()
        # SVC calls `_svm_impl._extension`, not `_bind`, and carries no
        # `_BINDING`, so the mixin's witness would read the BASE binding.
        # Point it at the extension the fit actually calls.
        probe._BINDING = "_mojolearn_svm"
        self.info = _ours_info(self.ml, probe)
        self.params_obj = probe

    def make(self):
        # random_state stays None: ours refuses it without probability=True.
        return self.ml.SVC(C=SVC_C, kernel="rbf", gamma=self.gamma, tol=SVC_TOL,
                           degree=SVC_DEGREE, coef0=SVC_COEF0, max_iter=SVC_MAX_ITER,
                           cache_size=SVC_CACHE_MB, class_weight=None)

    def call(self):
        est = self.make()
        est.fit(self.X, self.y)
        self.est = est

    def sync(self):
        pass

    def outputs(self):
        pred = self.est.predict(self.Xq)
        return {"pred": np.array(pred, dtype=np.float64),
                "n_support": np.array([int(self.est.n_support_)], dtype=np.int64)}


# ---- scikit-learn ------------------------------------------------------------

def _sklearn_info():
    import scipy
    import sklearn
    # `pre_clock_fit` defaults to False for every scikit-learn arm and SkKDE
    # overrides it, because its tree is built before the clock. Declared here
    # so the field is present on EVERY ready record and its absence is an
    # anomaly `_span_facts` can shout about rather than a case it must guess.
    info = {"library": "scikit-learn", "version": sklearn.__version__,
            "scipy": scipy.__version__, "device": "cpu",
            "pre_clock_fit": False}
    info.update(_host_info())
    try:
        from threadpoolctl import threadpool_info
        pools = []
        for p in threadpool_info():
            pools.append({k: p.get(k) for k in ("user_api", "internal_api", "num_threads",
                                                  "version", "threading_layer", "architecture")}
                         | {"filepath": os.path.basename(str(p.get("filepath")))})
        info["threadpools"] = pools
    except Exception as exc:  # noqa: BLE001
        info["threadpools"] = "unavailable (%r)" % (exc,)
    return info


def _sklearn_pools_touch():
    """Load scikit-learn's BLAS and OpenMP pools before threadpool_info reads
    them (a pool that has not been loaded is not listed)."""
    import sklearn.cluster  # noqa: F401
    import sklearn.utils.extmath  # noqa: F401
    import scipy.linalg  # noqa: F401


class SkKMeans:
    def __init__(self, data, rec):
        from sklearn.cluster import KMeans
        _sklearn_pools_touch()
        self.KMeans = KMeans
        self.X, self.init = data["X"], data["init"]
        self.est = None
        self.info = _sklearn_info()
        self.info["config"] = ("KMeans(n_clusters=64, init=<the shared array>, n_init=1, max_iter=20, "
                               "tol=%g, algorithm='lloyd', random_state=%d)" % (KMEANS_TOL, SEED))
        self.params_obj = self.make()

    def make(self):
        return self.KMeans(n_clusters=KMEANS_K, init=self.init, n_init=1,
                           max_iter=KMEANS_ITER, tol=KMEANS_TOL, algorithm="lloyd",
                           random_state=SEED)

    def call(self):
        est = self.make()
        est.fit(self.X)
        self.est = est

    def sync(self):
        pass

    def outputs(self):
        return {"centers": np.asarray(self.est.cluster_centers_, dtype=np.float32),
                "labels": np.asarray(self.est.labels_, dtype=np.int32),
                "n_iter": np.array([int(self.est.n_iter_)], dtype=np.int64)}


class SkPCA:
    def __init__(self, data, rec):
        from sklearn.decomposition import PCA
        _sklearn_pools_touch()
        self.PCA = PCA
        self.X = data["X"]
        self.est = None
        self.info = _sklearn_info()
        self.info["config"] = ("PCA(n_components=8, svd_solver='covariance_eigh', whiten=False, "
                               "random_state=%d)" % SEED)
        self.params_obj = self.make()

    def make(self):
        return self.PCA(n_components=PCA_COMPONENTS, svd_solver="covariance_eigh", whiten=False,
                        random_state=SEED)

    def call(self):
        est = self.make()
        est.fit(self.X)
        self.est = est

    def sync(self):
        pass

    def outputs(self):
        return {"components": np.asarray(self.est.components_, dtype=np.float32),
                "explained_variance": np.asarray(self.est.explained_variance_, dtype=np.float32),
                "mean": np.asarray(self.est.mean_, dtype=np.float32)}


class SkOLS:
    def __init__(self, data, rec):
        from sklearn.linear_model import LinearRegression
        _sklearn_pools_touch()
        self.LR = LinearRegression
        self.X, self.y = data["X"], data["y"]
        self.est = None
        self.info = _sklearn_info()
        self.info["config"] = ("LinearRegression(fit_intercept=True, positive=False, copy_X=True) "
                               "(scipy.linalg.lstsq, gelsd)")
        self.params_obj = self.make()

    def make(self):
        return self.LR(fit_intercept=True, positive=False, copy_X=True)

    def call(self):
        est = self.make()
        est.fit(self.X, self.y)
        self.est = est

    def sync(self):
        pass

    def outputs(self):
        return {"coef": np.asarray(self.est.coef_, dtype=np.float64).ravel(),
                "intercept": np.array([float(self.est.intercept_)], dtype=np.float64)}


class SkKNN:
    def __init__(self, data, rec):
        from sklearn.neighbors import NearestNeighbors
        _sklearn_pools_touch()
        self.q = data["queries"]
        self.nn = NearestNeighbors(n_neighbors=KNN_K, algorithm="brute", metric="euclidean", p=2,
                                   n_jobs=-1)
        self.nn.fit(data["index"])
        self.out = None
        self.info = _sklearn_info()
        self.info["config"] = ("NearestNeighbors(n_neighbors=10, algorithm='brute', metric='euclidean', "
                               "p=2, n_jobs=-1); fit before the clock, kneighbors timed")
        self.info["pre_clock_fit"] = True
        self.params_obj = self.nn

    def call(self):
        self.out = self.nn.kneighbors(self.q)

    def sync(self):
        pass

    def outputs(self):
        dist, ind = self.out
        return {"ind": np.asarray(ind, dtype=np.int64),
                "dist": np.asarray(dist, dtype=np.float32)}


class SkKDE:
    def __init__(self, data, rec):
        from sklearn.neighbors import KernelDensity
        _sklearn_pools_touch()
        self.q = data["Xq"]
        t0 = time.perf_counter()
        self.kd = KernelDensity(bandwidth=rec["kde"]["bandwidth"], kernel="gaussian",
                                metric="euclidean", algorithm="auto", rtol=0.0, atol=0.0,
                                breadth_first=True, leaf_size=KDE_LEAF_SIZE)
        self.kd.fit(data["X"])
        fit_ms = (time.perf_counter() - t0) * 1000.0
        self.scores = None
        self.params_obj = self.kd
        self.info = _sklearn_info()
        self.info["config"] = ("KernelDensity(bandwidth=scott, kernel='gaussian', metric='euclidean', "
                               "algorithm='auto', rtol=0, atol=0, breadth_first=True, leaf_size=40); "
                               "score_samples timed; single-threaded by design")
        self.info["tree_fit_ms_untimed"] = fit_ms
        # The tree above was built before the clock, which times
        # `score_samples` alone. The magnitude is recorded beside it; the flag
        # is the fact, and it is what `_span_facts` reads.
        self.info["pre_clock_fit"] = True

    def call(self):
        self.scores = self.kd.score_samples(self.q)

    def sync(self):
        pass

    def outputs(self):
        return {"scores": np.asarray(self.scores, dtype=np.float64)}


class SkSVC:
    def __init__(self, data, rec):
        from sklearn.svm import SVC
        _sklearn_pools_touch()
        self.SVC = SVC
        self.X, self.y, self.Xq = data["X"], data["y"], data["Xq"]
        self.gamma = float(rec["svc"]["gamma"])
        self.est = None
        self.info = _sklearn_info()
        self.info["config"] = ("SVC(C=1.0, kernel='rbf', gamma=1/d, tol=1e-3, degree=3, coef0=0.0, "
                               "max_iter=-1, cache_size=2000, shrinking=True, class_weight=None, "
                               "random_state=%d); fit timed; libsvm is single-threaded" % SEED)
        self.params_obj = self.make()

    def make(self):
        return self.SVC(C=SVC_C, kernel="rbf", gamma=self.gamma, tol=SVC_TOL, degree=SVC_DEGREE,
                        coef0=SVC_COEF0, max_iter=SVC_MAX_ITER, cache_size=SVC_CACHE_MB,
                        shrinking=True, class_weight=None, random_state=SEED)

    def call(self):
        est = self.make()
        est.fit(self.X, self.y)
        self.est = est

    def sync(self):
        pass

    def outputs(self):
        return {"pred": np.asarray(self.est.predict(self.Xq), dtype=np.float64),
                "n_support": np.array([int(np.sum(self.est.n_support_))], dtype=np.int64)}


# ---- torch ---------------------------------------------------------------------

def _torch_setup(arrays):
    """Upload `arrays` (name -> host array) to the GPU before any clock.
    Returns (torch, device, tensors, info)."""
    import torch
    global _TORCH_KIND
    # Apple silicon: torch's GPU is MPS (tools/bench_board.py, the Apple
    # column). The same arm, the same algorithm, written the same way; an op
    # MPS lacks fails that arm by name at its warm-up, never silently on CPU.
    if torch.cuda.is_available():
        _TORCH_KIND = "cuda"
        dev = torch.device("cuda")
        dev_name = torch.cuda.get_device_name(0)
    elif (getattr(torch.backends, "mps", None) is not None
          and torch.backends.mps.is_available()):
        _TORCH_KIND = "mps"
        dev = torch.device("mps")
        dev_name = "Apple MPS"
    else:
        raise RuntimeError("torch sees no CUDA, ROCm or MPS device; no GPU arm on this box")
    # The board's seed through torch's own argument (no torch arm here draws a
    # random number; it is set so every arm carries the same seed).
    torch.manual_seed(SEED)
    _torch_sync()
    t0 = time.perf_counter()
    tensors = {k: torch.from_numpy(np.ascontiguousarray(v)).to(dev) for k, v in arrays.items()}
    _torch_sync()
    upload_ms = (time.perf_counter() - t0) * 1000.0
    info = {"library": "torch", "version": torch.__version__,
            "torch_version_hip": getattr(torch.version, "hip", None),
            "torch_version_cuda": torch.version.cuda,
            "torch_backend": _TORCH_KIND,
            "device": "gpu", "device_name": dev_name,
            "upload_ms_untimed": upload_ms, "seed": SEED,
            # No torch arm fits before its clock; declared, not left absent.
            "pre_clock_fit": False}
    info.update(_host_info())
    return torch, dev, tensors, info


class TorchKMeans:
    def __init__(self, data, rec):
        self.torch, self.dev, t, self.info = _torch_setup({"X": data["X"], "init": data["init"]})
        self.x, self.init = t["X"], t["init"]
        self.ones = self.torch.ones(self.x.shape[0], device=self.dev, dtype=self.torch.float32)
        self.info["config"] = ("Lloyd, 20 iterations, no early stop; assignment = chunked "
                               "addmm(||c||^2, x, c.T, alpha=-2).argmin; update = index_add_")
        self.c = None
        self.labels = None
        # Declared (a function, not an estimator): the values `call` really uses.
        self.params_obj = {"__library__": "torch", "seed": SEED, "n_clusters": KMEANS_K,
                           "init": _array_tag(data["init"]), "n_init": 1,
                           "max_iter": KMEANS_ITER, "metric": "euclidean"}

    def call(self):
        torch = self.torch
        x = self.x
        n, d = x.shape
        c = self.init.clone()
        k = c.shape[0]
        labels = torch.empty(n, dtype=torch.long, device=self.dev)
        for _ in range(KMEANS_ITER):
            c2 = (c * c).sum(dim=1)
            for s in range(0, n, TORCH_CHUNK_ROWS):
                e = min(s + TORCH_CHUNK_ROWS, n)
                labels[s:e] = torch.addmm(c2.unsqueeze(0), x[s:e], c.T,
                                          beta=1.0, alpha=-2.0).argmin(dim=1)
            sums = torch.zeros((k, d), device=self.dev, dtype=torch.float32)
            counts = torch.zeros(k, device=self.dev, dtype=torch.float32)
            sums.index_add_(0, labels, x)
            counts.index_add_(0, labels, self.ones)
            c = torch.where((counts > 0)[:, None], sums / counts.clamp(min=1.0)[:, None], c)
        self.c, self.labels = c, labels

    def sync(self):
        _torch_sync()

    def outputs(self):
        return {"centers": _to_host(self.c).astype(np.float32),
                "labels": _to_host(self.labels).astype(np.int32),
                "n_iter": np.array([KMEANS_ITER], dtype=np.int64)}


class TorchPCA:
    def __init__(self, data, rec):
        self.torch, self.dev, t, self.info = _torch_setup({"X": data["X"]})
        self.x = t["X"]
        self.info["config"] = "mean; xc = x - mean; cov = xc.T @ xc / (n - 1); torch.linalg.eigh(cov); top 8"
        self.components = self.ev = self.mean = None
        self.params_obj = {"__library__": "torch", "seed": SEED, "n_components": PCA_COMPONENTS,
                           "svd_solver": "covariance_eigh", "whiten": False}

    def call(self):
        torch = self.torch
        x = self.x
        mean = x.mean(dim=0)
        xc = x - mean
        cov = (xc.T @ xc) / float(x.shape[0] - 1)
        evals, evecs = torch.linalg.eigh(cov)
        self.components = evecs[:, -PCA_COMPONENTS:].flip(1).T.contiguous()
        self.ev = evals[-PCA_COMPONENTS:].flip(0)
        self.mean = mean

    def sync(self):
        _torch_sync()

    def outputs(self):
        return {"components": _to_host(self.components).astype(np.float32),
                "explained_variance": _to_host(self.ev).astype(np.float32),
                "mean": _to_host(self.mean).astype(np.float32)}


class TorchOLS:
    """torch.linalg.lstsq on centered data, the call a competent user writes.
    On CUDA its only driver is gels (QR without pivoting, full rank assumed);
    the quality beside the cell says what that costs on Istella's
    near-constant columns."""

    def __init__(self, data, rec):
        self.torch, self.dev, t, self.info = _torch_setup({"X": data["X"], "y": data["y"]})
        self.x, self.y = t["X"], t["y"]
        self.info["config"] = ("xc = x - mean, yc = y - mean; torch.linalg.lstsq(xc, yc[:, None]) "
                               "(default driver; gels on CUDA); intercept = ymean - xmean @ coef")
        self.coef = self.intercept = None
        self.params_obj = {"__library__": "torch", "seed": SEED, "fit_intercept": True}

    def call(self):
        torch = self.torch
        x, y = self.x, self.y
        xm = x.mean(dim=0)
        ym = y.mean()
        sol = torch.linalg.lstsq(x - xm, (y - ym).unsqueeze(1)).solution
        coef = sol[: x.shape[1], 0]
        self.coef = coef
        self.intercept = ym - xm @ coef

    def sync(self):
        _torch_sync()

    def outputs(self):
        return {"coef": _to_host(self.coef).astype(np.float64),
                "intercept": np.array([float(_to_host(self.intercept))], dtype=np.float64)}


class TorchOLSEigh:
    def __init__(self, data, rec):
        self.torch, self.dev, t, self.info = _torch_setup({"X": data["X"], "y": data["y"]})
        self.x, self.y = t["X"], t["y"]
        self.info["config"] = ("centered normal equations: G = xc.T @ xc, b = xc.T @ yc, "
                               "torch.linalg.eigh(G), pseudo-inverse cutoff d * eps32 * max|eig|")
        self.coef = self.intercept = None
        self.params_obj = {"__library__": "torch", "seed": SEED, "fit_intercept": True}

    def call(self):
        torch = self.torch
        x, y = self.x, self.y
        xm = x.mean(dim=0)
        ym = y.mean()
        xc = x - xm
        yc = y - ym
        g = xc.T @ xc
        b = xc.T @ yc
        evals, evecs = torch.linalg.eigh(g)
        cutoff = evals.abs().max() * float(g.shape[0]) * float(torch.finfo(torch.float32).eps)
        inv = torch.where(evals.abs() > cutoff, 1.0 / evals, torch.zeros_like(evals))
        coef = evecs @ (inv * (evecs.T @ b))
        self.coef = coef
        self.intercept = ym - xm @ coef

    def sync(self):
        _torch_sync()

    def outputs(self):
        return {"coef": _to_host(self.coef).astype(np.float64),
                "intercept": np.array([float(_to_host(self.intercept))], dtype=np.float64)}


class TorchKNN:
    def __init__(self, data, rec):
        self.torch, self.dev, t, self.info = _torch_setup({"index": data["index"],
                                                           "queries": data["queries"]})
        self.index, self.q = t["index"], t["queries"]
        self.info["config"] = ("brute force: per 1024-query chunk torch.cdist(q, index) "
                               "(default compute_mode) then topk(10, largest=False)")
        self.ind = None
        self.params_obj = {"__library__": "torch", "seed": SEED, "n_neighbors": KNN_K,
                           "metric": "euclidean", "p": 2, "algorithm": "brute"}

    def call(self):
        torch = self.torch
        q, index = self.q, self.index
        nq = q.shape[0]
        ind = torch.empty((nq, KNN_K), dtype=torch.long, device=self.dev)
        for s in range(0, nq, TORCH_KNN_QUERY_CHUNK):
            e = min(s + TORCH_KNN_QUERY_CHUNK, nq)
            d = torch.cdist(q[s:e], index)
            _vals, idx = torch.topk(d, KNN_K, dim=1, largest=False)
            ind[s:e] = idx
        self.ind = ind

    def sync(self):
        _torch_sync()

    def outputs(self):
        return {"ind": _to_host(self.ind).astype(np.int64)}


# ---- cuML (DEVIATION 2571: the GPU library on the box is the opponent) ----------

def _cupy_sync():
    import cupy as cp
    cp.cuda.runtime.deviceSynchronize()


def _cuml_setup(arrays):
    """Upload `arrays` to the GPU as cupy arrays before any clock. Returns
    (tensors, info)."""
    import cupy as cp
    import cuml
    cp.cuda.runtime.deviceSynchronize()
    t0 = time.perf_counter()
    dev = {k: cp.asarray(np.ascontiguousarray(v)) for k, v in arrays.items()}
    cp.cuda.runtime.deviceSynchronize()
    upload_ms = (time.perf_counter() - t0) * 1000.0
    try:
        name = cp.cuda.runtime.getDeviceProperties(0)["name"]
        name = name.decode() if isinstance(name, bytes) else str(name)
    except Exception as exc:  # noqa: BLE001
        name = "unavailable (%r)" % (exc,)
    info = {"library": "cuml", "version": cuml.__version__, "cupy": cp.__version__,
            "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
            "cuda_driver_api": cp.cuda.runtime.driverGetVersion(),
            "device": "gpu", "device_name": name, "upload_ms_untimed": upload_ms,
            # False by default; CumlKNN and CumlKDE set it True after the fit
            # they perform before the clock starts.
            "pre_clock_fit": False}
    info.update(_host_info())
    return dev, info


def _scalar(v):
    return float(np.asarray(_to_host(v)).reshape(-1)[0]) if not isinstance(v, (int, float)) else float(v)


class CumlKMeans:
    def __init__(self, data, rec):
        from cuml.cluster import KMeans
        t, self.info = _cuml_setup({"X": data["X"], "init": data["init"]})
        self.x, self.init = t["X"], t["init"]
        # cuVS refuses tol <= 0 ("RAFT failure ... invalid parameter (tol<=0)",
        # H100 pod 22up9vbhj3tbeg, 2026-09-11), exactly as our binding does, so
        # every arm takes KMEANS_TOL.
        self.params_obj, self.kw = construct_tolerant(
            KMeans, dict(n_clusters=KMEANS_K, init=self.init, n_init=1, max_iter=KMEANS_ITER,
                         tol=KMEANS_TOL, random_state=SEED, output_type="cupy"), self.info)
        self.KMeans = KMeans
        self.info["config"] = ("cuml.cluster.KMeans(n_clusters=64, init=<the shared array, on device>, "
                               "n_init=1, max_iter=20, tol=%g, random_state=%d, output_type='cupy'); "
                               "fit timed" % (KMEANS_TOL, SEED))
        self.est = None

    def call(self):
        est = self.KMeans(**self.kw)
        est.fit(self.x)
        self.est = est

    def sync(self):
        _cupy_sync()

    def outputs(self):
        return {"centers": np.asarray(_to_host(self.est.cluster_centers_), dtype=np.float32),
                "labels": np.asarray(_to_host(self.est.labels_), dtype=np.int32),
                "n_iter": np.array([int(_scalar(self.est.n_iter_))], dtype=np.int64)}


class CumlPCA:
    def __init__(self, data, rec):
        from cuml.decomposition import PCA
        self.PCA = PCA
        t, self.info = _cuml_setup({"X": data["X"]})
        self.x = t["X"]
        self.params_obj, self.kw = construct_tolerant(
            PCA, dict(n_components=PCA_COMPONENTS, svd_solver="full", whiten=False,
                      random_state=SEED, output_type="cupy"), self.info)
        self.info["config"] = ("cuml.decomposition.PCA(n_components=8, svd_solver='full', whiten=False, "
                               "random_state=%d, output_type='cupy'); fit timed" % SEED)
        self.est = None

    def call(self):
        est = self.PCA(**self.kw)
        est.fit(self.x)
        self.est = est

    def sync(self):
        _cupy_sync()

    def outputs(self):
        return {"components": np.asarray(_to_host(self.est.components_), dtype=np.float32),
                "explained_variance": np.asarray(_to_host(self.est.explained_variance_), dtype=np.float32),
                "mean": np.asarray(_to_host(self.est.mean_), dtype=np.float32).reshape(-1)}


class CumlOLS:
    def __init__(self, data, rec):
        from cuml.linear_model import LinearRegression
        self.LR = LinearRegression
        t, self.info = _cuml_setup({"X": data["X"], "y": data["y"]})
        self.x, self.y = t["X"], t["y"]
        # copy_X=True where the build takes it: fit_intercept centers the
        # design matrix, and a centered device X would change every later round.
        self.kw = dict(algorithm="eig", fit_intercept=True, output_type="cupy")
        try:
            self.LR(copy_X=True, **self.kw)
            self.kw["copy_X"] = True
        except TypeError:
            self.info["copy_X"] = "not a parameter of this build"
        self.info["config"] = "cuml.linear_model.LinearRegression(%s); fit timed" % (
            ", ".join("%s=%r" % kv for kv in sorted(self.kw.items())),)
        self.params_obj = self.LR(**self.kw)
        self.est = None

    def call(self):
        est = self.LR(**self.kw)
        est.fit(self.x, self.y)
        self.est = est

    def sync(self):
        _cupy_sync()

    def outputs(self):
        return {"coef": np.asarray(_to_host(self.est.coef_), dtype=np.float64).reshape(-1),
                "intercept": np.array([_scalar(self.est.intercept_)], dtype=np.float64)}


class CumlKNN:
    def __init__(self, data, rec):
        from cuml.neighbors import NearestNeighbors
        t, self.info = _cuml_setup({"index": data["index"], "queries": data["queries"]})
        self.q = t["queries"]
        self.nn, _kw = construct_tolerant(
            NearestNeighbors, dict(n_neighbors=KNN_K, algorithm="brute", metric="euclidean", p=2,
                                   output_type="cupy"), self.info)
        self.params_obj = self.nn
        self.nn.fit(t["index"])
        _cupy_sync()
        # THE FACT, DECLARED AT THE PLACE IT BECOMES TRUE: the fit above ran
        # before any clock, so this arm's timed region covers `kneighbors`
        # alone. The config string below says so too, for a human; the flag is
        # what `_span_facts` reads, so rewording that prose cannot silently
        # turn this fact false.
        self.info["pre_clock_fit"] = True
        self.info["config"] = ("cuml.neighbors.NearestNeighbors(n_neighbors=10, algorithm='brute', "
                               "metric='euclidean', output_type='cupy'); fit before the clock, kneighbors timed")
        self.out = None

    def call(self):
        self.out = self.nn.kneighbors(self.q)

    def sync(self):
        _cupy_sync()

    def outputs(self):
        dist, ind = self.out
        return {"ind": np.asarray(_to_host(ind), dtype=np.int64),
                "dist": np.asarray(_to_host(dist), dtype=np.float32)}


class CumlKDE:
    def __init__(self, data, rec):
        from cuml.neighbors import KernelDensity
        t, self.info = _cuml_setup({"X": data["X"], "Xq": data["Xq"]})
        self.q = t["Xq"]
        self.kd = KernelDensity(bandwidth=rec["kde"]["bandwidth"], kernel="gaussian",
                                metric="euclidean", output_type="cupy")
        self.params_obj = self.kd
        self.kd.fit(t["X"])
        _cupy_sync()
        # Declared where it becomes true: the fit above is outside the clock,
        # which times `score_samples` alone. See CumlKNN.
        self.info["pre_clock_fit"] = True
        self.info["config"] = ("cuml.neighbors.KernelDensity(bandwidth=scott, kernel='gaussian', "
                               "metric='euclidean', output_type='cupy'); fit before the clock, score_samples timed")
        self.scores = None

    def call(self):
        self.scores = self.kd.score_samples(self.q)

    def sync(self):
        _cupy_sync()

    def outputs(self):
        return {"scores": np.asarray(_to_host(self.scores), dtype=np.float64).reshape(-1)}


class CumlSVC:
    def __init__(self, data, rec):
        from cuml.svm import SVC
        self.SVC = SVC
        t, self.info = _cuml_setup({"X": data["X"], "y": data["y"], "Xq": data["Xq"]})
        self.x, self.y, self.xq = t["X"], t["y"], t["Xq"]
        self.gamma = float(rec["svc"]["gamma"])
        self.params_obj, self.kw = construct_tolerant(
            SVC, dict(C=SVC_C, kernel="rbf", gamma=self.gamma, tol=SVC_TOL, degree=SVC_DEGREE,
                      coef0=SVC_COEF0, max_iter=SVC_MAX_ITER, cache_size=SVC_CACHE_MB,
                      class_weight=None, random_state=SEED, output_type="cupy"), self.info)
        self.info["config"] = ("cuml.svm.SVC(C=1.0, kernel='rbf', gamma=1/d, tol=1e-3, degree=3, coef0=0.0, "
                               "max_iter=-1, cache_size=2000, class_weight=None, random_state=%d, "
                               "output_type='cupy'); fit timed" % SEED)
        self.est = None

    def call(self):
        est = self.SVC(**self.kw)
        est.fit(self.x, self.y)
        self.est = est

    def sync(self):
        _cupy_sync()

    def outputs(self):
        pred = self.est.predict(self.xq)
        return {"pred": np.asarray(_to_host(pred), dtype=np.float64).reshape(-1),
                "n_support": np.array([int(np.asarray(_to_host(self.est.support_)).shape[0])],
                                      dtype=np.int64)}


def _cgroup_cpus():
    """(CPUs the cgroup CPU quota allows, where it was read), or (None, reason).
    A container can see every host CPU (os.cpu_count, sched_getaffinity) while
    a CFS quota holds it to far fewer: a RunPod MI300X pod on 2026-09-11 saw
    192 and was allowed 20.4."""
    try:
        with open("/sys/fs/cgroup/cpu.max") as fh:
            quota, period = fh.read().split()[:2]
        if quota != "max":
            return float(quota) / float(period), "cgroup v2 cpu.max"
    except (OSError, ValueError):
        pass
    for base in ("/sys/fs/cgroup/cpu", "/sys/fs/cgroup/cpu,cpuacct"):
        try:
            with open(base + "/cpu.cfs_quota_us") as fh:
                quota = int(fh.read())
            with open(base + "/cpu.cfs_period_us") as fh:
                period = int(fh.read())
            if quota > 0 and period > 0:
                return quota / period, "cgroup v1 %s/cpu.cfs_quota_us" % base
        except (OSError, ValueError):
            pass
    return None, "no cgroup CPU quota"


class SkQuota:
    """sklearn-cpu-quota: the same scikit-learn call, its BLAS and OpenMP pools
    (and kNN's n_jobs) capped at the whole CPUs the cgroup quota allows. The
    uncapped sklearn-cpu arm sizes its pools to every VISIBLE CPU, which on a
    quota-limited container oversubscribes the quota; this arm is the
    opponent's better configuration on such a box, and both are reported."""

    def __init__(self, inner_cls, data, rec):
        from threadpoolctl import threadpool_limits
        self.limits = threadpool_limits
        cpus, source = _cgroup_cpus()
        self.cap = max(1, int(cpus)) if cpus else (os.cpu_count() or 1)
        with self.limits(limits=self.cap):
            self.inner = inner_cls(data, rec)
            if hasattr(self.inner, "nn"):
                self.inner.nn.set_params(n_jobs=self.cap)
            self.info = _sklearn_info()
        self.info["config"] = self.inner.info.get("config", "") + (
            "; threadpool_limits(limits=%d) around construction and every call" % self.cap)
        self.info["thread_cap"] = {"threads": self.cap, "quota_cpus": cpus, "source": source}
        for k in ("tree_fit_ms_untimed", "pre_clock_fit"):
            if k in self.inner.info:
                self.info[k] = self.inner.info[k]
        self.params_obj = getattr(self.inner, "params_obj", None)

    def call(self):
        with self.limits(limits=self.cap):
            self.inner.call()

    def sync(self):
        pass

    def outputs(self):
        return self.inner.outputs()


# ---- DBSCAN and HDBSCAN (lane linear-cluster-istella, measurement only) -------

def _dbscan_params(ds):
    """(eps, min_samples) for this dataset, from MOJOLEARN_CTD_DBSCAN_<DATASET>
    ("eps,min_samples"). One value for every arm of a race, never a default."""
    key = "MOJOLEARN_CTD_DBSCAN_%s" % ds.upper()
    raw = os.environ.get(key, "")
    if "," not in raw:
        raise RuntimeError("%s is not set (want 'eps,min_samples')" % key)
    eps, ms = raw.split(",", 1)
    return float(eps), int(ms)


class OursDBSCAN:
    def __init__(self, data, rec):
        self.ml = _ours_module()
        self.X = data["X"]
        self.eps, self.min_samples = _dbscan_params(rec["dataset"])
        self.est = None
        self.params_obj = self.make()
        self.info = _ours_info(self.ml, self.params_obj)
        self.info["config"] = ("mojolearn.DBSCAN(eps=%r, min_samples=%d, metric='euclidean', "
                               "algorithm='rbc' (its default)); fit timed" % (self.eps, self.min_samples))

    def make(self):
        return self.ml.DBSCAN(eps=self.eps, min_samples=self.min_samples, metric="euclidean",
                              algorithm="rbc")

    def call(self):
        est = self.make()
        est.fit(self.X)
        self.est = est

    def sync(self):
        pass

    def outputs(self):
        return {"labels": np.array(self.est.labels_, dtype=np.int32).reshape(-1)}


class CumlDBSCAN:
    ALGO = "brute"

    def __init__(self, data, rec):
        from cuml.cluster import DBSCAN
        self.DBSCAN = DBSCAN
        t, self.info = _cuml_setup({"X": data["X"]})
        self.x = t["X"]
        self.eps, self.min_samples = _dbscan_params(rec["dataset"])
        self.params_obj, self.kw = construct_tolerant(
            DBSCAN, dict(eps=self.eps, min_samples=self.min_samples, metric="euclidean",
                         algorithm=self.ALGO, calc_core_sample_indices=False, output_type="cupy"),
            self.info)
        self.info["config"] = ("cuml.cluster.DBSCAN(eps=%r, min_samples=%d, metric='euclidean', "
                               "algorithm=%r, calc_core_sample_indices=False, output_type='cupy'); "
                               "fit timed" % (self.eps, self.min_samples, self.ALGO))
        self.est = None

    def call(self):
        est = self.DBSCAN(**self.kw)
        est.fit(self.x)
        self.est = est

    def sync(self):
        _cupy_sync()

    def outputs(self):
        return {"labels": np.asarray(_to_host(self.est.labels_), dtype=np.int32).reshape(-1)}


class CumlDBSCANRbc(CumlDBSCAN):
    ALGO = "rbc"


class SkDBSCAN:
    """sklearn.cluster.DBSCAN with the race's eps and min_samples on the same
    1,000,000-row standardized block, on every core (n_jobs=-1; its default is
    one). algorithm='auto' is scikit-learn's own choice: a tree index on
    low-dimensional taxi, brute force on 220-column Istella-S, where a round can
    run for hours; the board's per-round limit then records the timeout."""

    def __init__(self, data, rec):
        from sklearn.cluster import DBSCAN
        _sklearn_pools_touch()
        self.DBSCAN = DBSCAN
        self.X = data["X"]
        self.eps, self.min_samples = _dbscan_params(rec["dataset"])
        self.est = None
        self.info = _sklearn_info()
        self.info["config"] = ("sklearn.cluster.DBSCAN(eps=%r, min_samples=%d, metric='euclidean', "
                               "algorithm='auto', n_jobs=-1); fit timed" % (self.eps, self.min_samples))
        self.params_obj = self.make()

    def make(self):
        return self.DBSCAN(eps=self.eps, min_samples=self.min_samples, metric="euclidean",
                           algorithm="auto", n_jobs=-1)

    def call(self):
        est = self.make()
        est.fit(self.X)
        self.est = est

    def sync(self):
        pass

    def outputs(self):
        return {"labels": np.asarray(self.est.labels_, dtype=np.int32).reshape(-1)}


#: HDBSCAN: one setting for every arm, every parameter the libraries share
#: set explicitly (max_cluster_size: 0 on ours and cuML, None on scikit-learn;
#: both mean no limit).
HDBSCAN_KW = dict(min_samples=10, min_cluster_size=100, metric="euclidean",
                  cluster_selection_method="eom", cluster_selection_epsilon=0.0, alpha=1.0,
                  allow_single_cluster=False)


class CumlHDBSCAN:
    MIN_SAMPLES = HDBSCAN_KW["min_samples"]
    MIN_CLUSTER_SIZE = HDBSCAN_KW["min_cluster_size"]

    def __init__(self, data, rec):
        from cuml.cluster import HDBSCAN
        self.HDBSCAN = HDBSCAN
        n = int(rec.get("hdbscan", {}).get("rows", HDBSCAN_ROWS))
        t, self.info = _cuml_setup({"X": data["X"][:n]})
        self.x = t["X"]
        self.params_obj, self.kw = construct_tolerant(
            HDBSCAN, dict(HDBSCAN_KW, max_cluster_size=0, output_type="cupy"), self.info)
        self.info["config"] = ("cuml.cluster.HDBSCAN(%s) on the block's first %d rows; fit timed"
                               % (", ".join("%s=%r" % kv for kv in sorted(self.kw.items())), n))
        self.est = None

    def call(self):
        est = self.HDBSCAN(**self.kw)
        est.fit(self.x)
        self.est = est

    def sync(self):
        _cupy_sync()

    def outputs(self):
        return {"labels": np.asarray(_to_host(self.est.labels_), dtype=np.int32).reshape(-1)}


class OursHDBSCAN:
    """mojolearn.HDBSCAN with the same settings on the same first rows."""
    MIN_SAMPLES = CumlHDBSCAN.MIN_SAMPLES
    MIN_CLUSTER_SIZE = CumlHDBSCAN.MIN_CLUSTER_SIZE

    def __init__(self, data, rec):
        self.ml = _ours_module()
        n = int(rec.get("hdbscan", {}).get("rows", HDBSCAN_ROWS))
        self.X = data["X"][:n]
        self.est = None
        self.params_obj = self.make()
        self.info = _ours_info(self.ml, self.params_obj)
        self.info["config"] = ("mojolearn.HDBSCAN(%s, max_cluster_size=0) on the block's first %d rows; "
                               "fit timed" % (", ".join("%s=%r" % kv for kv in sorted(HDBSCAN_KW.items())), n))

    def make(self):
        return self.ml.HDBSCAN(max_cluster_size=0, **HDBSCAN_KW)

    def call(self):
        est = self.make()
        est.fit(self.X)
        self.est = est

    def sync(self):
        pass

    def outputs(self):
        return {"labels": np.array(self.est.labels_, dtype=np.int32).reshape(-1)}


class SkHDBSCAN:
    """sklearn.cluster.HDBSCAN with the same settings, on every core
    (n_jobs=-1: its default is ONE core, and every other scikit-learn arm
    here runs on every core)."""
    MIN_SAMPLES = CumlHDBSCAN.MIN_SAMPLES
    MIN_CLUSTER_SIZE = CumlHDBSCAN.MIN_CLUSTER_SIZE

    def __init__(self, data, rec):
        from sklearn.cluster import HDBSCAN
        _sklearn_pools_touch()
        self.HDBSCAN = HDBSCAN
        n = int(rec.get("hdbscan", {}).get("rows", HDBSCAN_ROWS))
        self.X = data["X"][:n]
        self.est = None
        self.info = _sklearn_info()
        self.info["config"] = ("sklearn.cluster.HDBSCAN(%s, max_cluster_size=None, n_jobs=-1) on the "
                               "block's first %d rows; fit timed"
                               % (", ".join("%s=%r" % kv for kv in sorted(HDBSCAN_KW.items())), n))
        self.params_obj = self.make()

    def make(self):
        return self.HDBSCAN(max_cluster_size=None, n_jobs=-1, **HDBSCAN_KW)

    def call(self):
        est = self.make()
        est.fit(self.X)
        self.est = est

    def sync(self):
        pass

    def outputs(self):
        return {"labels": np.asarray(self.est.labels_, dtype=np.int32).reshape(-1)}


BUILDERS = {
    ("dbscan", "ours"): OursDBSCAN, ("dbscan", "cuml-gpu"): CumlDBSCAN,
    ("dbscan", "cuml-gpu-rbc"): CumlDBSCANRbc, ("dbscan", "sklearn-cpu"): SkDBSCAN,
    ("hdbscan", "cuml-gpu"): CumlHDBSCAN,
    ("hdbscan", "ours"): OursHDBSCAN, ("hdbscan", "sklearn-cpu"): SkHDBSCAN,
    ("kmeans", "ours"): OursKMeans, ("kmeans", "sklearn-cpu"): SkKMeans, ("kmeans", "torch-gpu"): TorchKMeans,
    ("kmeans", "cuml-gpu"): CumlKMeans,
    ("pca", "ours"): OursPCA, ("pca", "sklearn-cpu"): SkPCA, ("pca", "torch-gpu"): TorchPCA,
    ("pca", "cuml-gpu"): CumlPCA,
    ("ols", "ours"): OursOLS, ("ols", "sklearn-cpu"): SkOLS, ("ols", "torch-gpu"): TorchOLS,
    ("ols", "torch-gpu-eigh"): TorchOLSEigh, ("ols", "cuml-gpu"): CumlOLS,
    ("knn", "ours"): OursKNN, ("knn", "sklearn-cpu"): SkKNN, ("knn", "torch-gpu"): TorchKNN,
    ("knn", "cuml-gpu"): CumlKNN,
    ("kde", "ours"): OursKDE, ("kde", "sklearn-cpu"): SkKDE, ("kde", "cuml-gpu"): CumlKDE,
    ("svc", "ours"): OursSVC, ("svc", "sklearn-cpu"): SkSVC, ("svc", "cuml-gpu"): CumlSVC,
}
# `ours-base`: OUR SAME estimator from a second Python tree
# (MOJOLEARN_CTD_BASE_PY, a copy of `python/` holding the BEFORE bindings), so
# a before/after A/B interleaves round by round in one race instead of two
# races minutes apart (lane linear-cluster-speed f3de0b36, 2026-09-11; taken
# here for the SVC before/after as well). It is never an opponent: ratios
# against it are ours-vs-ours and are not quoted as one.
for _lane in LANES:
    # A lane without an `ours` arm has no before/after to interleave.
    if (_lane, "ours") in BUILDERS:
        BUILDERS[(_lane, "ours-base")] = BUILDERS[(_lane, "ours")]
        # `ours-fast`: the SAME estimator in a worker started under
        # MOJOLEARN_NUMERIC_MODE=fast (`_worker_env`), so the Apple board
        # interleaves FAST beside IDENTICAL round by round in one race.
        BUILDERS[(_lane, "ours-fast")] = BUILDERS[(_lane, "ours")]
        # `ours-cpu`: the SAME estimator in a worker started under
        # MOJOLEARN_VENDOR=cpu (the wheel's public CPU switch,
        # tools/bench_board_probe.py), IDENTICAL, read back as vendor cpu.
        BUILDERS[(_lane, "ours-cpu")] = BUILDERS[(_lane, "ours")]
for _lane in LANES:
    # Same for a lane with no scikit-learn arm: there is
    # nothing to wrap in the CPU quota.
    if (_lane, "sklearn-cpu") not in BUILDERS:
        continue
    BUILDERS[(_lane, "sklearn-cpu-quota")] = (
        lambda data, rec, _c=BUILDERS[(_lane, "sklearn-cpu")]: SkQuota(_c, data, rec))
    ARMS[_lane] = ARMS[_lane] + ("sklearn-cpu-quota",)


def _digest(outputs):
    h = hashlib.sha256()
    for k in sorted(outputs):
        h.update(k.encode())
        h.update(np.ascontiguousarray(outputs[k]).data)
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# inference (`race --infer`, OFF by default: without it nothing below runs and
# the race's output and JSON are unchanged)
# ---------------------------------------------------------------------------
#
# After the fit rounds, every arm of a lane with a public predict or transform
# times it on the eval rows `Xq` with ITS OWN model from the last fit round
# (the fit is not retimed): kmeans predict, pca transform, ols predict, svc
# predict. kNN (kneighbors) and KDE (score_samples) already time inference as
# their race; DBSCAN and HDBSCAN have no predict. The span is the fit's: ours
# takes host rows and returns host results inside its clock; the torch and
# cuML arms upload `Xq` BEFORE their clock (upload_ms_untimed) and their clock
# ends at the device synchronize, exactly as their fit rounds do.

INFER_LANES = ("kmeans", "pca", "ols", "svc")
INFER_CALL = {"kmeans": "predict", "pca": "transform", "ols": "predict", "svc": "predict"}


class _InferRunner:
    def __init__(self, call, sync, outputs, desc, info=None):
        self.call, self.sync, self.outputs, self.desc = call, sync, outputs, desc
        self.info = info or {}


def _torch_upload(runner, x):
    t0 = time.perf_counter()
    dev = runner.torch.from_numpy(np.ascontiguousarray(x)).to(runner.dev)
    _torch_sync()
    return dev, (time.perf_counter() - t0) * 1000.0


def _cupy_upload(x):
    import cupy as cp
    t0 = time.perf_counter()
    dev = cp.asarray(np.ascontiguousarray(x))
    cp.cuda.runtime.deviceSynchronize()
    return dev, (time.perf_counter() - t0) * 1000.0


def infer_runner(lane, runner, data):
    """The timed inference call for one fitted runner, or a RuntimeError
    naming why this arm has none. Setup here (an upload) is untimed."""
    if lane not in INFER_LANES:
        raise RuntimeError("lane %s has no inference phase (its race already times "
                           "inference, or its estimator has no predict)" % lane)
    quota = runner if isinstance(runner, SkQuota) else None
    inner = quota.inner if quota else runner
    xq = data["Xq"]
    out = {}
    host_sync = (lambda: None)
    kind = INFER_CALL[lane]

    def keep(v):
        out["v"] = v

    if isinstance(inner, (OursKMeans, OursPCA, OursOLS, OursSVC, SkKMeans, SkPCA, SkOLS, SkSVC)):
        est = inner.est
        if est is None:
            raise RuntimeError("no fitted estimator")
        method = getattr(est, kind)
        lib = "mojolearn" if isinstance(inner, (OursKMeans, OursPCA, OursOLS, OursSVC)) else "sklearn"
        def call(method=method):
            if quota is None:
                keep(method(xq))
                return
            with quota.limits(limits=quota.cap):
                keep(method(xq))
        desc = "%s %s.%s(Xq), host rows in, host result out" % (lib, type(est).__name__, kind)
        return _InferRunner(call, host_sync, lambda: {"pred": _infer_host(lane, out["v"])}, desc)
    if isinstance(inner, (TorchKMeans, TorchPCA, TorchOLS, TorchOLSEigh)):
        torch = inner.torch
        xd, up = _torch_upload(inner, xq)
        info = {"upload_ms_untimed": up}
        if isinstance(inner, TorchKMeans):
            c = inner.c

            def call():
                c2 = (c * c).sum(dim=1)
                n = xd.shape[0]
                labels = torch.empty(n, dtype=torch.long, device=inner.dev)
                for s in range(0, n, TORCH_CHUNK_ROWS):
                    e = min(s + TORCH_CHUNK_ROWS, n)
                    labels[s:e] = torch.addmm(c2.unsqueeze(0), xd[s:e], c.T,
                                              beta=1.0, alpha=-2.0).argmin(dim=1)
                keep(labels)
            desc = "torch chunked addmm(||c||^2, Xq, c.T, alpha -2).argmin over the fitted centers"
        elif isinstance(inner, TorchPCA):
            comp, mean = inner.components, inner.mean
            call = (lambda: keep((xd - mean) @ comp.T))
            desc = "torch (Xq - mean) @ components.T"
        else:
            coef, icpt = inner.coef, inner.intercept
            call = (lambda: keep(xd @ coef + icpt))
            desc = "torch Xq @ coef + intercept"
        return _InferRunner(call, _torch_sync, lambda: {"pred": _infer_host(lane, out["v"])},
                            desc + "; Xq uploaded before the clock, which ends at the device synchronize",
                            info)
    if isinstance(inner, (CumlKMeans, CumlPCA, CumlOLS, CumlSVC)):
        est = inner.est
        if est is None:
            raise RuntimeError("no fitted estimator")
        if isinstance(inner, CumlSVC):
            xd, up = inner.xq, None
        else:
            xd, up = _cupy_upload(xq)
        method = getattr(est, kind)
        call = (lambda: keep(method(xd)))
        desc = ("cuml %s.%s(Xq on the device, output_type cupy); Xq uploaded before the clock, "
                "which ends at the device synchronize" % (type(est).__name__, kind))
        return _InferRunner(call, _cupy_sync, lambda: {"pred": _infer_host(lane, out["v"])}, desc,
                            {"upload_ms_untimed": up})
    raise RuntimeError("no inference call wired for %s" % type(inner).__name__)


def _infer_host(lane, v):
    """One dtype per lane on every arm, so outputs compare bit for bit."""
    a = np.asarray(_to_host(v))
    if lane == "kmeans":
        return np.ascontiguousarray(a.reshape(-1), dtype=np.int32)
    if lane == "svc":
        return np.ascontiguousarray(a.reshape(-1), dtype=np.float64)
    if lane == "ols":
        return np.ascontiguousarray(a.reshape(-1), dtype=np.float32)
    return np.ascontiguousarray(a, dtype=np.float32)


def infer_quality(lane, data, infer_outs, fit_outs):
    """Arm -> quality of its timed inference output: the lane's metric on the
    eval rows, agreement with a float64 NumPy evaluation of the arm's OWN
    fitted model, and agreement with ours (`bits_equal_vs_ours` is the FAST
    against IDENTICAL check for `ours-fast`)."""
    q = {}
    Xq = data["Xq"].astype(np.float64)
    for arm, o in infer_outs.items():
        p = o["pred"]
        f = fit_outs.get(arm) or {}
        e = {}
        try:
            if lane == "kmeans" and "centers" in f:
                C = f["centers"].astype(np.float64)
                c2 = (C * C).sum(axis=1)
                agree = total = 0.0
                for s in range(0, Xq.shape[0], 250_000):
                    xb = Xq[s:s + 250_000]
                    d = np.maximum((xb * xb).sum(axis=1)[:, None] - 2.0 * (xb @ C.T) + c2[None, :], 0.0)
                    dmin = d.min(axis=1)
                    lab = p[s:s + 250_000].astype(np.int64)
                    ok = (lab >= 0) & (lab < C.shape[0])
                    got = np.where(ok, d[np.arange(d.shape[0]), np.clip(lab, 0, C.shape[0] - 1)], np.inf)
                    agree += float((got <= dmin * (1.0 + 1e-6) + 1e-9).sum())
                    total += float(np.where(ok, got, 0.0).sum())
                e["eval_inertia"] = total
                e["label_agreement_own_centers"] = agree / Xq.shape[0]
            elif lane == "pca" and "components" in f:
                ref = (Xq - f["mean"].astype(np.float64).reshape(-1)) @ f["components"].astype(np.float64).T
                scale = float(np.max(np.abs(ref))) or 1.0
                e["transform_max_rel_err_own_fp64"] = float(np.max(np.abs(p.astype(np.float64) - ref))) / scale
            elif lane == "ols":
                yq = data["yq"].astype(np.float64)
                pr = p.astype(np.float64)
                ss_tot = float(((yq - yq.mean()) ** 2).sum())
                res = yq - pr
                e["r2_eval"] = 1.0 - float((res * res).sum()) / ss_tot if ss_tot else None
                e["rmse_eval"] = float(np.sqrt(float((res * res).sum()) / yq.shape[0]))
                if "coef" in f:
                    ref = Xq @ f["coef"].astype(np.float64).reshape(-1) + float(f["intercept"][0])
                    scale = float(np.max(np.abs(ref))) or 1.0
                    e["predict_max_rel_err_own_fp64"] = float(np.max(np.abs(pr - ref))) / scale
            elif lane == "svc":
                e["accuracy_eval"] = float((p == data["yq"].astype(np.float64)).mean())
        except Exception as exc:  # noqa: BLE001
            e["error"] = repr(exc)[:200]
        ref = infer_outs.get("ours")
        if ref is not None and arm != "ours":
            r = ref["pred"]
            same = r.shape == p.shape
            e["bits_equal_vs_ours"] = bool(same and r.tobytes() == p.tobytes())
            if same and lane in ("kmeans", "svc"):
                e["agreement_vs_ours"] = float((r == p).mean())
            elif same:
                e["max_abs_diff_vs_ours"] = float(np.max(np.abs(r.astype(np.float64) - p.astype(np.float64)))) if p.size else 0.0
        q[arm] = e
    return q


def worker(args):
    # The protocol gets its own descriptor; fd 1 becomes stderr (the log), so
    # a library that prints cannot interleave with a protocol line.
    proto = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    def say(obj):
        proto.write(json.dumps(obj, sort_keys=True) + "\n")
        proto.flush()

    block = os.path.join(args.data, "%s-%s" % (BLOCK_OF[args.lane], args.dataset))
    try:
        with np.load(block + ".npz") as z:
            data = {k: np.ascontiguousarray(z[k]) for k in z.files}
        with open(block + ".json") as fh:
            rec = json.load(fh)
        runner = BUILDERS[(args.lane, args.arm)](data, rec)
        # the parameters this arm really got, read back from what it constructed
        params = params_record(getattr(runner, "params_obj", None))
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        say({"event": "error", "stage": "ready", "error": repr(exc)})
        return 1
    say({"event": "ready", "info": runner.info, "pid": os.getpid(), "params": params})
    # peak memory per round, reset and read OUTSIDE the clock
    mem = _probe().MemProbe((runner.info or {}).get("device", "gpu"))
    inf = None
    for line in sys.stdin:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "round":
            r = int(parts[1])
            try:
                mem.start()
                t0 = time.perf_counter()
                runner.call()
                runner.sync()
                ms = (time.perf_counter() - t0) * 1000.0
                m = mem.stop()
                digest = _digest(runner.outputs())
            except Exception as exc:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                say({"event": "error", "stage": "round %d" % r, "error": repr(exc)})
                return 1
            say({"event": "round", "round": r, "ms": ms, "digest": digest, "mem": m})
        elif parts[0] == "infer":
            # `race --infer` only. An inference failure is reported and the
            # worker stays up: its fit outputs still have to be saved.
            r = int(parts[1])
            try:
                if inf is None:
                    inf = infer_runner(args.lane, runner, data)
                mem.start()
                t0 = time.perf_counter()
                inf.call()
                inf.sync()
                ms = (time.perf_counter() - t0) * 1000.0
                m = mem.stop()
                digest = _digest(inf.outputs())
            except Exception as exc:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                say({"event": "infer_error", "stage": "infer %d" % r, "error": repr(exc)})
                continue
            say({"event": "infer", "round": r, "ms": ms, "digest": digest,
                 "call": inf.desc, "info": inf.info, "mem": m})
        elif parts[0] == "infer_save":
            try:
                path = parts[1]
                tmp = path + ".tmp.npz"
                np.savez(tmp, **inf.outputs())
                os.replace(tmp, path)
                say({"event": "infer_saved", "path": path})
            except Exception as exc:  # noqa: BLE001
                say({"event": "infer_error", "stage": "infer_save", "error": repr(exc)})
        elif parts[0] == "save":
            try:
                path = parts[1]
                tmp = path + ".tmp.npz"
                np.savez(tmp, **runner.outputs())
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
# race: the conductor for one (lane, dataset)
# ---------------------------------------------------------------------------

class Worker:
    def __init__(self, arm, cmd, env, log_path, cwd):
        self.arm = arm
        self.log = open(log_path, "w")
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self.log, env=env, cwd=cwd,
                                     start_new_session=True)
        self.buf = b""
        self.noise = []
        self.alive = True
        self.status = "ok"
        self.error = None
        self.info = None

    def send(self, text):
        self.proc.stdin.write((text + "\n").encode())
        self.proc.stdin.flush()

    def read(self, seconds):
        """One protocol object, or None at the deadline or at EOF. A line
        that is not a JSON object (a launcher such as `pixi run` printing
        before the interpreter owns fd 1) is kept in `noise` and skipped."""
        deadline = time.monotonic() + seconds
        fd = self.proc.stdout.fileno()
        while True:
            while b"\n" not in self.buf:
                left = deadline - time.monotonic()
                if left <= 0:
                    return None
                ready, _, _ = select.select([fd], [], [], min(left, 5.0))
                if not ready:
                    continue
                chunk = os.read(fd, 65536)
                if not chunk:
                    return None
                self.buf += chunk
            line, self.buf = self.buf.split(b"\n", 1)
            text = line.decode(errors="replace").strip()
            if text.startswith("{"):
                try:
                    return json.loads(text)
                except ValueError:
                    pass
            self.noise.append(text[:200])

    def kill(self, status, error):
        self.alive = False
        self.status = status
        self.error = error
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except OSError:
            pass
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass

    def close(self):
        if self.alive:
            try:
                self.send("quit")
                self.read(60)
            except OSError:
                pass
        try:
            self.proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.kill(self.status, self.error)
        self.log.close()


def _worker_env(arm, root):
    env = dict(os.environ)
    for k in THREAD_ENV:
        env.pop(k, None)
    if arm in ("ours", "ours-base", "ours-fast", "ours-cpu"):
        env["MOJOLEARN_NUMERIC_MODE"] = "fast" if arm == "ours-fast" else "identical"
        if arm == "ours-cpu":
            _probe().ours_cpu_env(env)
        tree = os.path.join(root, "python")
        if arm == "ours-base":
            tree = os.environ.get("MOJOLEARN_CTD_BASE_PY", "")
            if not tree or not os.path.isdir(tree):
                raise SystemExit("arm ours-base needs MOJOLEARN_CTD_BASE_PY, a python/ tree holding the before bindings")
        elif os.environ.get("MOJOLEARN_BENCH_INSTALLED", "0").strip() not in ("", "0"):
            # tools/bench_board.py measures the INSTALLED wheel: the in-repo
            # python/ (no compiled bindings in a shipped tree) must not shadow it.
            return env
        env["PYTHONPATH"] = tree + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def quality(lane, data, outs, rec):
    """Arm name -> quality dict, one float64 NumPy function per lane."""
    q = {}
    if lane == "kmeans":
        X = data["X"]
        for arm, o in outs.items():
            C = o["centers"].astype(np.float64)
            c2 = (C * C).sum(axis=1)
            total = 0.0
            for s in range(0, X.shape[0], 250_000):
                xb = X[s:s + 250_000].astype(np.float64)
                d = (xb * xb).sum(axis=1)[:, None] - 2.0 * (xb @ C.T) + c2[None, :]
                total += float(np.maximum(d.min(axis=1), 0.0).sum())
            q[arm] = {"inertia": total, "n_iter": int(o["n_iter"][0])}
        ref = q.get("ours", {}).get("inertia")
        for arm in q:
            if ref:
                q[arm]["inertia_over_ours"] = q[arm]["inertia"] / ref
    elif lane == "pca":
        X = data["X"]
        n = X.shape[0]
        mu = np.zeros(X.shape[1])
        for s in range(0, n, 250_000):
            mu += X[s:s + 250_000].astype(np.float64).sum(axis=0)
        mu /= n
        total = 0.0
        proj = {arm: 0.0 for arm in outs}
        comps = {arm: o["components"].astype(np.float64) for arm, o in outs.items()}
        for s in range(0, n, 250_000):
            xb = X[s:s + 250_000].astype(np.float64) - mu
            total += float((xb * xb).sum())
            for arm, V in comps.items():
                p = xb @ V.T
                proj[arm] += float((p * p).sum())
        for arm in outs:
            q[arm] = {"explained_variance_ratio_sum": proj[arm] / total if total else None}
    elif lane == "ols":
        Xq, yq = data["Xq"].astype(np.float64), data["yq"].astype(np.float64)
        ss_tot = float(((yq - yq.mean()) ** 2).sum())
        for arm, o in outs.items():
            pred = Xq @ o["coef"].astype(np.float64) + float(o["intercept"][0])
            res = yq - pred
            ss_res = float((res * res).sum())
            q[arm] = {"r2": 1.0 - ss_res / ss_tot if ss_tot else None,
                      "rmse": float(np.sqrt(ss_res / yq.shape[0])),
                      "finite": bool(np.all(np.isfinite(pred)))}
    elif lane == "knn":
        index = data["index"].astype(np.float64)
        Q = data["queries"].astype(np.float64)
        nq = Q.shape[0]
        shortlist = min(4 * KNN_K, index.shape[0])

        def d64(qrows, ids):
            diff = Q[qrows][:, None, :] - index[ids]
            return (diff * diff).sum(axis=2)

        # THE REFERENCE IS OURS TO COMPUTE, NOT AN ARM'S: float64 NumPy brute
        # force. A norm-expansion shortlist of 40 per query, unioned with every
        # arm's returned ids, then exact float64 differences over that union;
        # the reference is its 10th smallest distance. An id an arm found that
        # the shortlist missed is in the union, so it cannot be scored a miss.
        x2 = (index * index).sum(axis=1)
        kth = np.empty(nq)
        for s in range(0, nq, 64):
            e = min(s + 64, nq)
            qb = Q[s:e]
            dd = (qb * qb).sum(axis=1)[:, None] - 2.0 * (qb @ index.T) + x2[None, :]
            cand = np.argpartition(dd, shortlist - 1, axis=1)[:, :shortlist]
            union = np.concatenate([cand] + [o["ind"][s:e].astype(np.int64) for o in outs.values()],
                                   axis=1)
            exact = d64(np.arange(s, e), union)
            for r in range(e - s):
                _u, first = np.unique(union[r], return_index=True)
                kth[s + r] = np.sort(exact[r, first])[KNN_K - 1]
        for arm, o in outs.items():
            ids = o["ind"].astype(np.int64)
            entry = {"reference": "float64 NumPy brute force (shortlist 40 plus every arm's ids, exact differences)"}
            srt = np.sort(ids, axis=1)
            distinct = 1 + (np.diff(srt, axis=1) != 0).sum(axis=1)
            entry["rows_with_repeated_ids"] = int((distinct < KNN_K).sum())
            d = np.concatenate([d64(np.arange(s, min(s + 256, nq)), ids[s:s + 256])
                                for s in range(0, nq, 256)], axis=0)
            within = d <= kth[:, None] * (1.0 + 1e-9) + 1e-12
            hits = np.minimum(within.sum(axis=1), distinct)
            entry["recall_at_10"] = float(hits.mean() / KNN_K)
            q[arm] = entry
    elif lane == "kde":
        sentinel = -3.0e38
        for arm, o in outs.items():
            s = o["scores"].astype(np.float64)
            ok = np.isfinite(s) & (s > sentinel)
            q[arm] = {"mean_log_likelihood": float(s[ok].mean()) if ok.any() else None,
                      "rows_without_density": int((~ok).sum())}
    elif lane in ("dbscan", "hdbscan"):
        # Cluster count, noise share, and agreement with OUR labels (adjusted
        # Rand index over every row both arms labeled; noise is its own label).
        ref = outs.get("ours")
        for arm, o in outs.items():
            lab = o["labels"].astype(np.int64).reshape(-1)
            ent = {"rows": int(lab.shape[0]),
                   "n_clusters": int(np.unique(lab[lab >= 0]).shape[0]),
                   "noise_fraction": float((lab < 0).mean())}
            if ref is not None and arm != "ours" and ref["labels"].size == lab.size:
                from sklearn.metrics import adjusted_rand_score
                r = ref["labels"].astype(np.int64).reshape(-1)
                ent["ari_vs_ours"] = float(adjusted_rand_score(r, lab))
                ent["noise_agreement_vs_ours"] = float(((r < 0) == (lab < 0)).mean())
            q[arm] = ent
    elif lane == "svc":
        yq = data["yq"].astype(np.float64)
        for arm, o in outs.items():
            q[arm] = {"accuracy": float((o["pred"] == yq).mean()),
                      "n_support": int(o["n_support"][0])}
    return q


#: What each library's arm has OUTSIDE its clock, keyed on `info["library"]`.
#: Read from the ready record every arm already sends; nothing here measures
#: anything new and nothing here changes what is timed.
def _span_facts(arm, info):
    """What is inside this arm's timed region and what is outside it.

    THE NUMBER WAS ALREADY MEASURED AND THEN THROWN AWAY (2026-09-12, lane
    harness-honesty). `_cuml_setup` and `_torch_setup` have always recorded
    `upload_ms_untimed`, `SkKDE` has always recorded `tree_fit_ms_untimed`,
    and the module docstring has always said which arms fit before the clock.
    All of it reached the JSON and none of it reached `CTD-RATIO`, which is
    the one line a reader actually reads. This assembles it.

    WHY IT MATTERS, WITH THE DIRECTION STATED. On Istella-S KDE the opponent's
    X is device-resident before its clock starts while ours uploads inside the
    timed call -- about 19.7 ms of host work at d = 220, of which 7.28 ms is
    host validation (DEVIATION 604), against 36.9 ms of device entry
    (`bench/results/kde_fused_2026-09-12/README.md`). That asymmetry runs
    AGAINST US: our published KDE gap is worse than the code deserves. It is
    surfaced for the same reason a gap in our favour would be -- a ratio whose
    two sides cover different spans of work is not a like-for-like number, and
    a reader is entitled to know before drawing a conclusion from it.

    NOTHING IS EQUALIZED HERE. No work moves into or out of any clock and no
    ratio changes; the asymmetry is named, not corrected. Correcting it on our
    side is a bindings question (keeping the DeviceBuffer alive across fit),
    it is legitimate only as a user-facing improvement because a user
    re-fitting pays that upload too, and it must never be done to improve a
    ratio."""
    info = info or {}
    library = info.get("library", "?")
    config = str(info.get("config", ""))
    facts = {
        "upload_ms_untimed": info.get("upload_ms_untimed"),
        "fit_ms_untimed": info.get("tree_fit_ms_untimed"),
    }
    # A PRE-CLOCK FIT IS A DECLARED FACT, NEVER AN INFERENCE FROM PROSE
    # (2026-09-12, lane harness-honesty-3). This used to substring-match
    # "fit before the clock" in the arm's own config string. That is a
    # silent-zero generator of exactly the class this reporting exists to
    # eliminate: reword the prose -- a tidy-up, a rename, a translation, a
    # copy-edit -- and a TRUE fact turns false with no error, no missing
    # output and nothing for a reader to notice. It is strictly worse than
    # the problem it solved. Every ready record now carries the flag, set at
    # the place the pre-clock fit actually happens (CumlKNN, CumlKDE, SkKDE)
    # and defaulted to False in the four builders, so the field is ALWAYS
    # present on a record built by this file.
    #
    # The prose fallback survives only for a record this file did not build
    # (an older JSON replayed, a worker from another checkout), and it is
    # LOUD: the source is reported on the CTD-SPAN line and a warning rides
    # with it, because a guess that succeeds silently is the thing being
    # removed here.
    declared = info.get("pre_clock_fit")
    if declared is None:
        facts["pre_clock_fit"] = ("fit before the clock" in config
                                  or info.get("tree_fit_ms_untimed") is not None)
        facts["pre_clock_fit_source"] = "INFERRED-FROM-PROSE"
        facts["pre_clock_fit_warning"] = (
            "this arm's ready record carries no pre_clock_fit flag, so the "
            "fact was GUESSED from its config text; a reworded string would "
            "silently flip it. Set info['pre_clock_fit'] where the fit "
            "happens.")
    else:
        facts["pre_clock_fit"] = bool(declared)
        facts["pre_clock_fit_source"] = "declared"
    if library == "mojolearn":
        # Our public call uploads and validates inside the clock, because that
        # is what a caller of this surface pays. Stated, not measured here: a
        # per-stage split needs the binding's own instrument, and this file
        # must not invent a number it did not take in this run.
        facts["inside_clock"] = "host_to_device_upload,host_validation,call"
        facts["input_home"] = "host"
    elif library in ("cuml", "torch"):
        facts["inside_clock"] = "call"
        facts["input_home"] = "device"
    else:
        facts["inside_clock"] = "call"
        facts["input_home"] = "host"
    facts["library"] = library
    return facts


def _fmt_span(v):
    return "-" if v is None else ("%.3f" % v if isinstance(v, float) else str(v))


def _span_qty(v):
    """A magnitude for the asymmetry field, or the WORD `unmeasured`.

    Not `-`: a dash beside `fit_before_its_clock` reads as a formatting fault
    or, worse, as zero. The two facts are independent -- cuML's kNN and KDE
    demonstrably fit before their clock and NO term in their ready record
    times that fit, while scikit-learn's KDE records one. So the fact is
    reported as measured and the missing magnitude is reported as missing."""
    return "unmeasured" if v is None else "%.3f_ms" % v


def _span_warnings(spans, opponents):
    """Which opponents' clocks cover LESS WORK than ours, and why.

    A separate function rather than a loop inside `race` so a check can reach
    it without workers, a GPU and a dataset. The lane that added it is the one
    that keeps finding numbers nobody could prove were produced by the code
    they were attributed to; an emission path exercised only by a full rented
    run is exactly that shape of risk (`tools/test_ctd_span.py`).

    Returns a list of `arm:reason+reason` strings, empty when every arm's
    clock covers the same span. Only asymmetries AGAINST ours are reported,
    because those are the ones that make our own ratio look worse than the
    code deserves and would otherwise be read as a like-for-like loss."""
    ours_span = spans.get("ours", {})
    warn = []
    for k in sorted(opponents):
        s = spans.get(k, {})
        why = []
        if s.get("input_home") == "device" and ours_span.get("input_home") == "host":
            why.append("upload_outside_its_clock(%s)"
                       % _span_qty(s.get("upload_ms_untimed")))
        if s.get("pre_clock_fit") and not ours_span.get("pre_clock_fit"):
            why.append("fit_before_its_clock(%s)"
                       % _span_qty(s.get("fit_ms_untimed")))
        if why:
            warn.append("%s:%s" % (k, "+".join(why)))
    return warn


def infer_phase(args, lane, ds, arms, workers, result):
    """`race --infer`: after the fit rounds, one warm-up and `--rounds` timed
    inference rounds per arm that completed every fit round, the order rotated
    each round like the fit's. Returns the `infer` record of the race JSON."""
    xq = (result.get("block") or {}).get("arrays", {}).get("Xq", {}).get("shape")
    infer = {"call": INFER_CALL[lane], "batch": "Xq", "rows": (xq or [None])[0], "arms": {}}
    eligible = [a for a in arms if workers[a].alive and len(result["arms"][a]["ms"]) == args.rounds]
    for arm in arms:
        infer["arms"][arm] = {"warmup_ms": None, "ms": [], "digests": [], "mem": [],
                              "status": "ok" if arm in eligible else "no_fit"}
    for r in range(args.rounds + 1):
        live = [a for a in eligible if workers[a].alive and infer["arms"][a]["status"] == "ok"]
        if not live:
            break
        shift = r % len(live)
        for arm in live[shift:] + live[:shift]:
            w = workers[arm]
            rec = infer["arms"][arm]
            w.send("infer %d" % r)
            msg = w.read(args.warmup_seconds if r == 0 else args.round_seconds)
            if msg is None or msg.get("event") != "infer":
                rec["status"] = "timeout" if msg is None else "error"
                rec["error"] = msg
                rec["failed_round"] = r
                if msg is None:
                    w.kill("infer_timeout", None)
                print("CTD-INFER-REFUSED lane=%s dataset=%s arm=%s stage=infer%d detail=%s"
                      % (lane, ds, arm, r, json.dumps(msg)), flush=True)
                continue
            rec["mem"].append(msg.get("mem"))
            if r == 0:
                rec["warmup_ms"] = msg["ms"]
                rec["call_text"] = msg.get("call")
                rec["info"] = msg.get("info")
            else:
                rec["ms"].append(msg["ms"])
            rec["digests"].append(msg["digest"])
            print("CTD-INFER-ROUND lane=%s dataset=%s arm=%s round=%d ms=%.3f digest=%s"
                  % (lane, ds, arm, r, msg["ms"], msg["digest"]), flush=True)
            time.sleep(args.pause)
    return infer


def infer_summary(lane, ds, infer, rounds):
    for arm, a in infer["arms"].items():
        ok = a["status"] == "ok" and len(a["ms"]) == rounds
        a["median_ms"] = median(a["ms"]) if ok else None
        a["min_ms"] = min(a["ms"]) if ok else None
        a["max_ms"] = max(a["ms"]) if ok else None
        a["digest_stable"] = (len(set(a["digests"][1:])) == 1) if ok else None
        print("CTD-INFER lane=%s dataset=%s arm=%s call=%s rows=%s status=%s median_ms=%s "
              "digest_stable=%s quality=%s"
              % (lane, ds, arm, infer["call"], infer["rows"], a["status"], a["median_ms"],
                 a["digest_stable"], json.dumps((infer.get("quality") or {}).get(arm, {}),
                                                sort_keys=True)), flush=True)


def race(args):
    lane, ds = args.lane, args.dataset
    arms = [a for a in (args.arms.split(",") if args.arms else ARMS[lane]) if a]
    for a in arms:
        if (lane, a) not in BUILDERS:
            raise SystemExit("no arm %r for lane %r" % (a, lane))
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.work, exist_ok=True)
    block = os.path.join(args.data, "%s-%s" % (BLOCK_OF[lane], ds))
    with open(block + ".json") as fh:
        rec = json.load(fh)
    result = {"lane": lane, "dataset": ds, "block": rec, "arms": {},
              "rounds_requested": args.rounds, "started": now_utc(),
              "script": "tools/classical_two_datasets.py",
              "commit": os.environ.get("MOJOLEARN_REPO_COMMIT", "unknown")}
    tag = "%s-%s" % (lane, ds)
    workers = {}
    for arm in arms:
        py = args.ours_python if arm in ("ours", "ours-base", "ours-fast", "ours-cpu") \
            else args.theirs_python
        cmd = shlex.split(py) + [os.path.abspath(__file__), "worker", "--arm", arm,
                                 "--lane", lane, "--dataset", ds, "--data", args.data]
        workers[arm] = Worker(arm, cmd, _worker_env(arm, args.root),
                              os.path.join(args.out, "%s-%s.log" % (tag, arm)), args.root)
        result["arms"][arm] = {"command": cmd, "warmup_ms": None, "ms": [],
                               "digests": [], "mem": [], "status": "ok"}
    # Ready, in parallel: each worker loads its data and its library.
    for arm, w in workers.items():
        msg = w.read(args.ready_seconds)
        if msg is None or msg.get("event") != "ready":
            w.kill("not_ready", msg)
            result["arms"][arm]["status"] = "not_ready"
            result["arms"][arm]["error"] = msg
            print("CTD-REFUSED lane=%s dataset=%s arm=%s stage=ready detail=%s"
                  % (lane, ds, arm, json.dumps(msg)), flush=True)
            continue
        w.info = msg["info"]
        result["arms"][arm]["info"] = msg["info"]
        result["arms"][arm]["params_record"] = msg.get("params")
    # SAME SEED, SAME TUNING PARAMETERS, checked before the first timed round
    # (tools/bench_board_params.py). A refusal fails the race by name.
    result["lane_config"] = CONFIG[lane]
    print("CTD-SETTINGS lane=%s dataset=%s %s" % (lane, ds, CONFIG[lane]["params"]), flush=True)
    for mm in CONFIG[lane]["mismatches"]:
        print("CTD-MISMATCH lane=%s dataset=%s %s" % (lane, ds, mm), flush=True)
    records = {a: result["arms"][a].get("params_record") for a in arms
               if workers[a].alive and result["arms"][a].get("params_record") is not None}
    refused = enforce_params(lane, "classical", records, result, "CTD")
    if refused is not None:
        for arm in arms:
            w = workers[arm]
            if w.alive:
                w.kill("params_refused", refused[:500])
                result["arms"][arm]["status"] = "params_refused"
                result["arms"][arm]["error"] = refused[:2000]
            w.log.close()
        result["finished"] = now_utc()
        with open(os.path.join(args.out, tag + ".json"), "w") as fh:
            json.dump(result, fh, indent=2, sort_keys=True, default=str)
        return 3
    for r in range(args.rounds + 1):
        live = [a for a in arms if workers[a].alive]
        if not live:
            break
        shift = r % len(live)
        order = live[shift:] + live[:shift]
        for arm in order:
            w = workers[arm]
            if not w.alive:
                continue
            w.send("round %d" % r)
            msg = w.read(args.warmup_seconds if r == 0 else args.round_seconds)
            if msg is None or msg.get("event") != "round":
                status = "timeout" if msg is None else "error"
                w.kill(status, msg)
                result["arms"][arm]["status"] = status
                result["arms"][arm]["error"] = msg
                result["arms"][arm]["failed_round"] = r
                print("CTD-REFUSED lane=%s dataset=%s arm=%s stage=round%d detail=%s"
                      % (lane, ds, arm, r, json.dumps(msg)), flush=True)
                continue
            if r == 0:
                result["arms"][arm]["warmup_ms"] = msg["ms"]
            else:
                result["arms"][arm]["ms"].append(msg["ms"])
            result["arms"][arm]["digests"].append(msg["digest"])
            result["arms"][arm]["mem"].append(msg.get("mem"))
            print("CTD-ROUND lane=%s dataset=%s arm=%s round=%d ms=%.3f digest=%s"
                  % (lane, ds, arm, r, msg["ms"], msg["digest"]), flush=True)
            time.sleep(args.pause)
    infer = None
    if getattr(args, "infer", False) and lane in INFER_LANES:
        infer = infer_phase(args, lane, ds, arms, workers, result)
    outs = {}
    infer_outs = {}
    for arm in arms:
        w = workers[arm]
        if w.alive and len(result["arms"][arm]["ms"]) == args.rounds:
            path = os.path.join(args.work, "%s-%s.npz" % (tag, arm))
            w.send("save %s" % path)
            msg = w.read(args.round_seconds)
            if msg is not None and msg.get("event") == "saved":
                with np.load(path) as z:
                    outs[arm] = {k: z[k] for k in z.files}
                result["arms"][arm]["info"] = msg.get("info", result["arms"][arm].get("info"))
            else:
                result["arms"][arm]["status"] = "save_failed"
                result["arms"][arm]["error"] = msg
        if infer is not None and w.alive and infer["arms"].get(arm, {}).get("status") == "ok":
            ipath = os.path.join(args.work, "%s-%s-infer.npz" % (tag, arm))
            w.send("infer_save %s" % ipath)
            msg = w.read(args.round_seconds)
            if msg is not None and msg.get("event") == "infer_saved":
                with np.load(ipath) as z:
                    infer_outs[arm] = {k: z[k] for k in z.files}
            else:
                infer["arms"][arm]["status"] = "save_failed"
                infer["arms"][arm]["error"] = msg
        w.close()
    result["finished_rounds"] = now_utc()
    try:
        with np.load(block + ".npz") as z:
            data = {k: z[k] for k in z.files}
        result["quality"] = quality(lane, data, outs, rec)
        # our CPU tier against our GPU IDENTICAL, bit for bit (the promise)
        if "ours-cpu" in outs and "ours" in outs:
            result["quality"].setdefault("ours-cpu", {})["bits_equal_vs_ours_identical"] = \
                _probe().bits_equal(outs["ours-cpu"], outs["ours"])
    except Exception as exc:  # noqa: BLE001
        result["quality"] = {"error": repr(exc)}
        data = None
    if infer is not None:
        try:
            infer["quality"] = infer_quality(lane, data, infer_outs, outs) if data else {}
        except Exception as exc:  # noqa: BLE001
            infer["quality"] = {"error": repr(exc)}
        infer_summary(lane, ds, infer, args.rounds)
        result["infer"] = infer
    ours = result["arms"].get("ours", {})
    ours_med = median(ours.get("ms", [])) if len(ours.get("ms", [])) == args.rounds else None
    result["ratios_ours_over"] = {}
    spans = {}
    for arm in arms:
        a = result["arms"][arm]
        ok = a["status"] == "ok" and len(a["ms"]) == args.rounds
        a["median_ms"] = median(a["ms"]) if ok else None
        a["min_ms"] = min(a["ms"]) if ok else None
        a["max_ms"] = max(a["ms"]) if ok else None
        a["digest_stable"] = (len(set(a["digests"][1:])) == 1) if ok else None
        dev = (a.get("info") or {}).get("device", "?")
        qual = result.get("quality", {}).get(arm, {})
        print("CTD lane=%s dataset=%s arm=%s device=%s status=%s median_ms=%s min_ms=%s max_ms=%s "
              "digest_stable=%s quality=%s"
              % (lane, ds, arm, dev, a["status"], a["median_ms"], a["min_ms"], a["max_ms"],
                 a["digest_stable"], json.dumps(qual, sort_keys=True)), flush=True)
        # WHAT WAS INSIDE THIS ARM'S CLOCK, beside its time. Assembled from the
        # ready record the arm already sent; no timing is affected.
        span = _span_facts(arm, a.get("info"))
        spans[arm] = span
        a["span"] = span
        print("CTD-SPAN lane=%s dataset=%s arm=%s input_home=%s inside_clock=%s "
              "upload_ms_untimed=%s fit_ms_untimed=%s pre_clock_fit=%s "
              "pre_clock_fit_source=%s"
              % (lane, ds, arm, span["input_home"], span["inside_clock"],
                 _fmt_span(span["upload_ms_untimed"]), _fmt_span(span["fit_ms_untimed"]),
                 str(span["pre_clock_fit"]).lower(),
                 span.get("pre_clock_fit_source", "declared")), flush=True)
        if span.get("pre_clock_fit_warning"):
            print("CTD-SPAN-WARNING lane=%s dataset=%s arm=%s %s"
                  % (lane, ds, arm, span["pre_clock_fit_warning"]), flush=True)
        if arm != "ours" and ours_med and a["median_ms"]:
            result["ratios_ours_over"][arm] = ours_med / a["median_ms"]
    result["spans"] = spans
    if result["ratios_ours_over"]:
        # THE RATIO IS UNCHANGED. The `ours/<arm>=<float>` tokens are exactly
        # what they were; a `span_asymmetry=` field is appended naming every
        # opponent whose clock covers less work than ours, so the one line a
        # reader reads cannot be read as like-for-like when it is not.
        warn = _span_warnings(spans, result["ratios_ours_over"])
        print("CTD-RATIO lane=%s dataset=%s %s%s" % (lane, ds, " ".join(
            "ours/%s=%.4f" % (k, v) for k, v in sorted(result["ratios_ours_over"].items())),
            (" span_asymmetry=" + ",".join(warn)) if warn else " span_asymmetry=none"),
            flush=True)
        if warn:
            print("CTD-SPAN-NOTE lane=%s dataset=%s the ratios above are NOT "
                  "like-for-like spans: ours uploads (and validates) inside its "
                  "clock while these arms do not. The asymmetry is named, not "
                  "corrected; no work was moved into or out of any clock."
                  % (lane, ds), flush=True)
    result["finished"] = now_utc()
    with open(os.path.join(args.out, tag + ".json"), "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True, default=str)
    return 0


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def _quality_text(q):
    if not q:
        return "-"
    return ",".join("%s=%s" % (k, ("%.6g" % v) if isinstance(v, float) else v)
                    for k, v in sorted(q.items()) if k not in ("reference",))


def summary(args):
    rows = []
    for name in sorted(os.listdir(args.out)):
        if not name.endswith(".json") or name.count("-") != 1:
            continue
        lane, ds = name[:-5].split("-")
        if lane not in LANES or ds not in DATASETS:
            continue
        with open(os.path.join(args.out, name)) as fh:
            r = json.load(fh)
        shapes = r["block"].get("arrays", {})
        first = shapes.get("X") or shapes.get("index") or {}
        shape = "x".join(str(s) for s in first.get("shape", []))
        for arm, a in r["arms"].items():
            info = a.get("info") or {}
            rows.append([lane, ds, shape, arm, info.get("device", "?"),
                         info.get("device_name") or info.get("cpu_model") or info.get("vendor_used") or "-",
                         a.get("status"), a.get("median_ms"), a.get("min_ms"), a.get("max_ms"),
                         len(a.get("ms", [])), a.get("digest_stable"),
                         _quality_text(r.get("quality", {}).get(arm)),
                         r.get("ratios_ours_over", {}).get(arm)])
    head = ["lane", "dataset", "shape", "arm", "device", "device_name", "status", "median_ms",
            "min_ms", "max_ms", "rounds", "digest_stable", "quality", "ours_over_this"]
    with open(os.path.join(args.out, "summary.tsv"), "w") as fh:
        fh.write("\t".join(head) + "\n")
        for row in rows:
            fh.write("\t".join("" if v is None else (("%.4f" % v) if isinstance(v, float) else str(v))
                               for v in row) + "\n")
    with open(os.path.join(args.out, "summary.tsv")) as fh:
        sys.stdout.write(fh.read())
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--data", required=True)
    p.add_argument("--lanes", default=",".join(LANES))
    p.add_argument("--datasets", default=",".join(DATASETS))
    p.add_argument("--max-rows", type=int, default=0,
                   help="SMOKE shape: cap every block near this many rows (0 = the lane shapes)")
    w = sub.add_parser("worker")
    w.add_argument("--arm", required=True)
    w.add_argument("--lane", required=True, choices=LANES)
    w.add_argument("--dataset", required=True, choices=DATASETS)
    w.add_argument("--data", required=True)
    r = sub.add_parser("race")
    r.add_argument("--lane", required=True, choices=LANES)
    r.add_argument("--dataset", required=True, choices=DATASETS)
    r.add_argument("--data", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--work", default="/root/ctd-work",
                   help="where arm outputs (labels, components) are saved; kept OUT of the fetched tree")
    r.add_argument("--root", default=REPO)
    r.add_argument("--arms", default="")
    r.add_argument("--rounds", type=int, default=5)
    r.add_argument("--ours-python", default="pixi run python3")
    r.add_argument("--theirs-python", default=sys.executable)
    r.add_argument("--ready-seconds", type=float, default=600.0)
    r.add_argument("--warmup-seconds", type=float, default=600.0)
    r.add_argument("--round-seconds", type=float, default=300.0)
    r.add_argument("--pause", type=float, default=0.5)
    r.add_argument("--infer", action="store_true",
                   help="after the fit rounds, time the lane's public predict or transform "
                        "on the eval rows (kmeans, pca, ols, svc; CTD-INFER lines and an "
                        "`infer` record). Off by default: the race is then unchanged")
    s = sub.add_parser("summary")
    s.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "prep":
        return prep(args)
    if args.cmd == "worker":
        return worker(args)
    if args.cmd == "race":
        if args.rounds < 1:
            raise SystemExit("--rounds must be >= 1")
        return race(args)
    return summary(args)


if __name__ == "__main__":
    sys.exit(main())
