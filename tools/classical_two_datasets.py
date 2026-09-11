#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Classical lanes on the two benchmark datasets (ENGINEERING_RULES.md
section 9, DEVIATION 2570): OUR IDENTICAL arm beside scikit-learn on every
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

LANES = ("kmeans", "pca", "ols", "knn", "kde", "svc")
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
}
BLOCK_OF = {"kmeans": "big", "pca": "big", "ols": "big", "knn": "knn",
            "kde": "kde", "svc": "svc"}

BIG_ROWS = 4_000_000
KMEANS_K = 64
KMEANS_ITER = 20
PCA_COMPONENTS = 8
KNN_K = 10
KDE_TRAIN = 100_000
KDE_QUERY = 2_000
SVC_TRAIN = 10_000
SVC_EVAL = 10_000
SVC_C = 1.0
SVC_TOL = 1e-3
TORCH_CHUNK_ROWS = 1 << 20
TORCH_KNN_QUERY_CHUNK = 1024

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
                "rule": "ENGINEERING_RULES.md section 9; DEVIATION 2570",
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


def _torch_sync():
    import torch
    torch.cuda.synchronize()


def _to_host(a):
    """Host NumPy copy of a torch tensor, cupy array, mojolearn Array or
    NumPy array."""
    if hasattr(a, "detach"):
        return a.detach().cpu().numpy()
    if type(a).__module__.split(".")[0] == "cupy":
        return a.get()
    return np.array(a, copy=True)


# ---- ours ----------------------------------------------------------------

def _ours_module():
    os.environ.setdefault("MOJOLEARN_NUMERIC_MODE", "identical")
    import mojolearn
    return mojolearn


def _ours_info(ml, est):
    info = {"library": "mojolearn", "version": getattr(ml, "__version__", "unknown"),
            "numeric_mode_env": os.environ.get("MOJOLEARN_NUMERIC_MODE")}
    for name in ("numeric_mode_used", "vendor_used"):
        try:
            info[name] = getattr(est, name)()
        except Exception as exc:  # noqa: BLE001
            info[name] = "unavailable (%r)" % (exc,)
    info["device"] = "gpu"
    if info.get("numeric_mode_used") != "identical":
        raise RuntimeError("ours is not IDENTICAL: numeric_mode_used() = %r"
                           % (info.get("numeric_mode_used"),))
    return info


class OursKMeans:
    def __init__(self, data, rec):
        self.ml = _ours_module()
        self.X, self.init = data["X"], data["init"]
        self.tol = 0.0
        self.est = None
        probe = self.ml.KMeans(n_clusters=KMEANS_K, init="array", n_init=1,
                               max_iter=KMEANS_ITER, tol=self.tol,
                               init_centroids=self.init)
        self.info = _ours_info(self.ml, probe)

    def call(self):
        try:
            est = self.ml.KMeans(n_clusters=KMEANS_K, init="array", n_init=1,
                                 max_iter=KMEANS_ITER, tol=self.tol,
                                 init_centroids=self.init)
            est.fit(self.X)
        except ValueError:
            if self.tol != 0.0:
                raise
            # tol = 0 refused by the surface: the speed driver's 1e-7.
            self.tol = 1e-7
            self.info["tol_fallback"] = 1e-7
            est = self.ml.KMeans(n_clusters=KMEANS_K, init="array", n_init=1,
                                 max_iter=KMEANS_ITER, tol=self.tol,
                                 init_centroids=self.init)
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
        self.info = _ours_info(self.ml, self.ml.PCA(n_components=PCA_COMPONENTS,
                                                    svd_solver="covariance_eigh"))

    def call(self):
        est = self.ml.PCA(n_components=PCA_COMPONENTS, svd_solver="covariance_eigh")
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
        self.info = _ours_info(self.ml, self.ml.LinearRegression(fit_intercept=True))

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
        self.nn = self.ml.NearestNeighbors(n_neighbors=KNN_K)
        self.nn.fit(data["index"])
        self.index = data["index"]
        self.out = None
        self.info = _ours_info(self.ml, self.nn)

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
        self.kd = self.ml.KernelDensity(bandwidth=rec["kde"]["bandwidth"], kernel="gaussian")
        self.kd.fit(data["X"])
        self.X = data["X"]
        self.scores = None
        self.info = _ours_info(self.ml, self.kd)

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
        probe = self.ml.SVC(C=SVC_C, kernel="rbf", gamma=self.gamma, tol=SVC_TOL)
        # SVC calls `_svm_impl._extension`, not `_bind`, and carries no
        # `_BINDING`, so the mixin's witness would read the BASE binding.
        # Point it at the extension the fit actually calls.
        probe._BINDING = "_mojolearn_svm"
        self.info = _ours_info(self.ml, probe)

    def call(self):
        est = self.ml.SVC(C=SVC_C, kernel="rbf", gamma=self.gamma, tol=SVC_TOL)
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
    info = {"library": "scikit-learn", "version": sklearn.__version__,
            "scipy": scipy.__version__, "device": "cpu"}
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
        self.info["config"] = "KMeans(n_clusters=64, init=<the shared array>, n_init=1, max_iter=20, tol=0.0, algorithm='lloyd')"

    def call(self):
        est = self.KMeans(n_clusters=KMEANS_K, init=self.init, n_init=1,
                          max_iter=KMEANS_ITER, tol=0.0, algorithm="lloyd",
                          random_state=0)
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
        self.info["config"] = "PCA(n_components=8, svd_solver='covariance_eigh')"

    def call(self):
        est = self.PCA(n_components=PCA_COMPONENTS, svd_solver="covariance_eigh")
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
        self.info["config"] = "LinearRegression(fit_intercept=True) (scipy.linalg.lstsq, gelsd)"

    def call(self):
        est = self.LR(fit_intercept=True)
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
        self.nn = NearestNeighbors(n_neighbors=KNN_K, algorithm="brute", n_jobs=-1)
        self.nn.fit(data["index"])
        self.out = None
        self.info = _sklearn_info()
        self.info["config"] = "NearestNeighbors(n_neighbors=10, algorithm='brute', n_jobs=-1); kneighbors timed"

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
                                rtol=0.0, atol=0.0)
        self.kd.fit(data["X"])
        fit_ms = (time.perf_counter() - t0) * 1000.0
        self.scores = None
        self.info = _sklearn_info()
        self.info["config"] = ("KernelDensity(bandwidth=scott, kernel='gaussian', rtol=0, atol=0); "
                               "score_samples timed; single-threaded by design")
        self.info["tree_fit_ms_untimed"] = fit_ms

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
        self.info["config"] = ("SVC(C=1.0, kernel='rbf', gamma=1/d, tol=1e-3, cache_size=2000); "
                               "fit timed; libsvm is single-threaded")

    def call(self):
        est = self.SVC(C=SVC_C, kernel="rbf", gamma=self.gamma, tol=SVC_TOL,
                       cache_size=2000.0)
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
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False; no GPU arm on this box")
    dev = torch.device("cuda")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    tensors = {k: torch.from_numpy(np.ascontiguousarray(v)).to(dev) for k, v in arrays.items()}
    torch.cuda.synchronize()
    upload_ms = (time.perf_counter() - t0) * 1000.0
    info = {"library": "torch", "version": torch.__version__,
            "torch_version_hip": getattr(torch.version, "hip", None),
            "torch_version_cuda": torch.version.cuda,
            "device": "gpu", "device_name": torch.cuda.get_device_name(0),
            "upload_ms_untimed": upload_ms}
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
            "device": "gpu", "device_name": name, "upload_ms_untimed": upload_ms}
    info.update(_host_info())
    return dev, info


def _scalar(v):
    return float(np.asarray(_to_host(v)).reshape(-1)[0]) if not isinstance(v, (int, float)) else float(v)


class CumlKMeans:
    def __init__(self, data, rec):
        from cuml.cluster import KMeans
        self.KMeans = KMeans
        t, self.info = _cuml_setup({"X": data["X"], "init": data["init"]})
        self.x, self.init = t["X"], t["init"]
        self.info["config"] = ("cuml.cluster.KMeans(n_clusters=64, init=<the shared array, on device>, "
                               "n_init=1, max_iter=20, tol=0.0, output_type='cupy'); fit timed")
        self.est = None

    def call(self):
        est = self.KMeans(n_clusters=KMEANS_K, init=self.init, n_init=1, max_iter=KMEANS_ITER,
                          tol=0.0, output_type="cupy")
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
        self.info["config"] = "cuml.decomposition.PCA(n_components=8, svd_solver='full', output_type='cupy'); fit timed"
        self.est = None

    def call(self):
        est = self.PCA(n_components=PCA_COMPONENTS, svd_solver="full", output_type="cupy")
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
        self.nn = NearestNeighbors(n_neighbors=KNN_K, algorithm="brute", metric="euclidean",
                                   output_type="cupy")
        self.nn.fit(t["index"])
        _cupy_sync()
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
        self.kd.fit(t["X"])
        _cupy_sync()
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
        self.info["config"] = ("cuml.svm.SVC(C=1.0, kernel='rbf', gamma=1/d, tol=1e-3, cache_size default, "
                               "output_type='cupy'); fit timed")
        self.est = None

    def call(self):
        est = self.SVC(C=SVC_C, kernel="rbf", gamma=self.gamma, tol=SVC_TOL, output_type="cupy")
        est.fit(self.x, self.y)
        self.est = est

    def sync(self):
        _cupy_sync()

    def outputs(self):
        pred = self.est.predict(self.xq)
        return {"pred": np.asarray(_to_host(pred), dtype=np.float64).reshape(-1),
                "n_support": np.array([int(np.asarray(_to_host(self.est.support_)).shape[0])],
                                      dtype=np.int64)}


BUILDERS = {
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


def _digest(outputs):
    h = hashlib.sha256()
    for k in sorted(outputs):
        h.update(k.encode())
        h.update(np.ascontiguousarray(outputs[k]).data)
    return h.hexdigest()[:16]


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
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        say({"event": "error", "stage": "ready", "error": repr(exc)})
        return 1
    say({"event": "ready", "info": runner.info, "pid": os.getpid()})
    for line in sys.stdin:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "round":
            r = int(parts[1])
            try:
                t0 = time.perf_counter()
                runner.call()
                runner.sync()
                ms = (time.perf_counter() - t0) * 1000.0
                digest = _digest(runner.outputs())
            except Exception as exc:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                say({"event": "error", "stage": "round %d" % r, "error": repr(exc)})
                return 1
            say({"event": "round", "round": r, "ms": ms, "digest": digest})
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
    if arm == "ours":
        env["MOJOLEARN_NUMERIC_MODE"] = "identical"
        env["PYTHONPATH"] = os.path.join(root, "python") + (
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
    elif lane == "svc":
        yq = data["yq"].astype(np.float64)
        for arm, o in outs.items():
            q[arm] = {"accuracy": float((o["pred"] == yq).mean()),
                      "n_support": int(o["n_support"][0])}
    return q


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
        py = args.ours_python if arm == "ours" else args.theirs_python
        cmd = shlex.split(py) + [os.path.abspath(__file__), "worker", "--arm", arm,
                                 "--lane", lane, "--dataset", ds, "--data", args.data]
        workers[arm] = Worker(arm, cmd, _worker_env(arm, args.root),
                              os.path.join(args.out, "%s-%s.log" % (tag, arm)), args.root)
        result["arms"][arm] = {"command": cmd, "warmup_ms": None, "ms": [],
                               "digests": [], "status": "ok"}
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
            print("CTD-ROUND lane=%s dataset=%s arm=%s round=%d ms=%.3f digest=%s"
                  % (lane, ds, arm, r, msg["ms"], msg["digest"]), flush=True)
            time.sleep(args.pause)
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
                result["arms"][arm]["info"] = msg.get("info", result["arms"][arm].get("info"))
            else:
                result["arms"][arm]["status"] = "save_failed"
                result["arms"][arm]["error"] = msg
        w.close()
    result["finished_rounds"] = now_utc()
    try:
        with np.load(block + ".npz") as z:
            data = {k: z[k] for k in z.files}
        result["quality"] = quality(lane, data, outs, rec)
    except Exception as exc:  # noqa: BLE001
        result["quality"] = {"error": repr(exc)}
    ours = result["arms"].get("ours", {})
    ours_med = median(ours.get("ms", [])) if len(ours.get("ms", [])) == args.rounds else None
    result["ratios_ours_over"] = {}
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
        if arm != "ours" and ours_med and a["median_ms"]:
            result["ratios_ours_over"][arm] = ours_med / a["median_ms"]
    if result["ratios_ours_over"]:
        print("CTD-RATIO lane=%s dataset=%s %s" % (lane, ds, " ".join(
            "ours/%s=%.4f" % (k, v) for k, v in sorted(result["ratios_ours_over"].items()))),
            flush=True)
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
