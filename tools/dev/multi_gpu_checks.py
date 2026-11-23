#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE MULTI-GPU DRIVER CHECKS. Optional. Not required for any merge or release. Not shipped to users.

(Andrew, 2026-11-20: the multi-GPU tests are kept separately, as something to optionally run. Developer doc:
docs/dev/MULTI_GPU_CHECKS.md.)

On a box with two or more GPUs of one vendor, for each unit in UNITS below this tool calls the PUBLIC multi-GPU
driver a user would call (`mojolearn.parallel_classical.fit_svm(..., devices=...)` and its siblings) on the
INSTALLED wheel in three arms, each in its own process:

    one     devices = (g,)            one physical GPU (g rotates over the box's GPUs, unit by unit)
    multi   devices = (0, 1, ..., N-1) every GPU of the box
    fault   the multi arm with a fault switched on (below), so the check is shown able to fail

and compares the arms by output digest: sha256 of each named output's raw C-order bytes with its shape and dtype
(the same definition as `mojolearn._verify_distributed.digest`), folded into one short digest per arm. One run per
arm; a mismatch is a bug to fix, never a reason to rerun.

    MULTIGPU <unit> one=<d> multi=<d> fault=<d> OK|DIFFER|FAIL <reason>

A unit is OK only when (1) one == multi, (2) the multi arm's work is seen on two or more ranks where the unit has a
placement witness, and (3) the fault arm's digest differs from the one arm's. It writes <out>/multi_gpu_checks.json
and exits 1 unless every selected unit is OK, 2 when it cannot run at all (fewer than two GPUs, a malformed table).
FAIL CLOSED: a unit that raises, times out or returns no digest is FAIL with a one-line reason, never a skip.

THE FAULT ARM, two kinds (the `fault` column of UNITS):
  shift   the drivers' own native negative control (bindings/parallel_shard_common_run.mojo `shard_read_shift`):
          with MOJOLEARN_PAR_DRIVER_SABOTAGE=1 and MOJOLEARN_HOST_ALLOW_SABOTAGE=1 every shard on an owner above
          rank 0 reads one position early while its block is still placed at its true offset. Inert at one device,
          so a differing digest also proves the multi arm really used a second owner. The entries that honour it:
          fit_scaler, transform_scaler, transform_rbf_sampler, fit_forest, ParallelQueries, ReferenceShardedNeighbors,
          fit_arima, fit_exponential_smoothing.
  input   every other driver has no device-level fault switch in a shipped binary, so its fault arm runs the multi
          arm on a perturbed fixture (FAULT_NOTE). That shows the unit's digest covers a computed result and that
          the verdict can read DIFFER; it does NOT show that a device-level defect would be caught. Say so in any
          report. A device-level fault for the cooperative drivers needs a native switch: none exists today.

PLACEMENT WITNESS (the `witness` column): `plan` units read the device-plan work counter of every loaded binding
(`device_plan_log()` -> (plans, ranks), core/device_plan.mojo `used`) after the multi arm; fewer than two ranks is a
FAIL ("spread"), as in tools/logical_shard_identity.py. `seen` units record the count and print it but are judged
only under --strict-spread, because their split rule against these fixture sizes has not been read. `none` units
have no counter in the caller's process (their shards run in worker processes); for the shift units the fault arm
is the witness.

PARALLEL PLAN: every (unit, arm) is an independent process. They run at once on a thread pool whose size is the
smallest of: the jobs, the allocation's cores (cgroup quota and affinity, not the host's CPUs), the host memory
bound (--host-gb-per-job) and the device memory bound (--gpu-gb-per-job against the smallest GPU's free memory;
every job opens a runtime context on each device it names). The one arms are spread over the GPUs round robin;
multi and fault arms use every GPU. The two per-job memory figures are estimates until the first run on a pod:
read the peaks it prints and pass measured values. tools/ops/multi_gpu_checks_job.sh samples busy cores of total.

NOT REQUIRED ANYWHERE. No release gate, ledger row, admit step, CI trigger, merge gate or push hook reads this
tool or its result; tools/release.py does not know it exists (tools/test_multi_gpu_checks_optional.py fails if a
release or gate module imports or names it). It is run by hand: tools/ops/multi_gpu_checks_job.sh on a pod.
Nothing here is in the wheel and `python -m mojolearn verify` has no option for it.

WHAT THIS DOES NOT COVER (say it plainly):
  * mixed card models in one plan, and more than one vendor in one box;
  * Apple (multi-device is not a Metal feature) and the CPU column (a CPU install's device indices are processes);
  * the plain one-device estimator: both arms go through the driver. The removed par-* lanes held each driver to
    the plain fit; a driver that is wrong on one device and equally wrong on two passes here. The one-device
    driver against the plain estimator is the release verification's and the logical-shard gate's business;
  * ParallelByteLanguageModelTrainer(schedule='offload') (one device by design: the removed lane ran it on devices[:1]);
  * the Mojo-level `training/checks/*_parallel_check.mojo` programs (they need a source build on the box:
    tools/two_device_checks.sh runs them; the job script can call it as a second phase).

Usage (on the box, with the interpreter the wheels are installed in):
    python3 tools/dev/multi_gpu_checks.py --python <venv>/bin/python --out DIR [--devices 0,1] [--units a,b]
        [--jobs N] [--no-fault] [--strict-spread] [--list]
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SCHEMA = "mojolearn.multi-gpu-checks.v1"
SUMMARY = "multi_gpu_checks.json"
ARMS = ("one", "multi", "fault")
UNIT_TIMEOUT = 1800
#: the fixture every unit slices: rows x features float32 standard normal, a binary and a regression target, and a
#: four-blob copy. Sizes are from the drivers' split rules (tools/logical_shard_identity.py THE LIST), never from a
#: benchmark shape.
ROWS, FEATURES = 6000, 16
FAULT_ENV = {"MOJOLEARN_PAR_DRIVER_SABOTAGE": "1", "MOJOLEARN_HOST_ALLOW_SABOTAGE": "1"}
FAULT_NOTE = "input fault: every third row of X scaled by 0.5 and shifted by 0.25, yr shifted on the same rows, " \
             "every seventh class label flipped"
MIN_SPREAD = 2
RETIRED_TABLE = Path(__file__).resolve().parent / "multi_gpu_retired_par_cells.json"


# ---------------------------------------------------------------------------------------------------- FIXTURE
# Test control only: built with NumPy in the harness, never inside the library's runtime.
def fixture(np, seed, rows=ROWS, features=FEATURES, fault=False):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((rows, features)).astype(np.float32)
    w = rng.standard_normal(features).astype(np.float32)
    yr = (X @ w + 0.1 * rng.standard_normal(rows)).astype(np.float32)
    yc = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(np.int32)
    centers = 4.0 * rng.standard_normal((4, features))
    yb = rng.integers(0, 4, rows).astype(np.int32)
    Xb = (centers[yb] + 0.3 * rng.standard_normal((rows, features))).astype(np.float32)
    if fault:
        X[::3] = X[::3] * np.float32(0.5) + np.float32(0.25)
        Xb[::3] = Xb[::3] * np.float32(0.5) + np.float32(0.25)
        yr[::3] = yr[::3] + np.float32(0.25)
        yc[::7] = 1 - yc[::7]
        yb[::7] = (yb[::7] + 1) % 4
    c = np.ascontiguousarray
    return dict(X=c(X), yc=c(yc), yr=c(yr), Xb=c(Xb), yb=c(yb))


def token_ids(np, X, batch, length, vocab=256):
    """A (batch, length) int32 token stream: the first batch*length bytes of X's float32 values."""
    raw = np.frombuffer(np.ascontiguousarray(X).tobytes()[: batch * length], dtype=np.uint8)
    return np.ascontiguousarray((raw.astype(np.int32) % vocab).reshape(batch, length))


# ---------------------------------------------------------------------------------------------------- RECIPES
# Each recipe takes (ml, np, fx, devices) and returns [(name, value), ...]: only public estimators, the public
# multi-GPU entries and their documented attributes. A fresh estimator per arm. Ported from the removed par-* lanes
# (experiments/removed/MOJOLEARN_PAR_VERIFY.patch, tools/identity_break.py) and from the logical-shard gate's
# recipes; the plain-fit side and the harness helpers of the removed lanes are not carried.
def _svm(ml, np, fx, devices):
    from mojolearn import parallel_classical as pc
    X, y = fx["X"][:600], fx["yc"][:600]
    est = pc.fit_svm(ml.SVC(kernel="rbf"), X, y, devices=devices)
    return [("dual_coef_", est.dual_coef_), ("intercept_", est.intercept_),
            ("decision", pc.predict_svm(est, fx["X"][600:856], devices=devices, method="decision_function")),
            ("predict", pc.predict_svm(est, fx["X"][600:856], devices=devices, method="predict"))]


def _svr(ml, np, fx, devices):
    from mojolearn import parallel_classical as pc
    est = pc.fit_svm(ml.SVR(C=1.0, kernel="rbf", epsilon=0.1, max_iter=200), fx["X"][:2000], fx["yr"][:2000],
                     devices=devices)
    return [("predict", pc.predict_svm(est, fx["X"][2000:2256], devices=devices, method="predict"))]


def _gp(ml, np, fx, devices):
    from mojolearn import parallel_classical as pc
    X, y = np.ascontiguousarray(fx["X"][:512, :8]), fx["yr"][:512]
    est = pc.fit_gaussian_process(ml.GaussianProcessRegressor(), X, y, devices=devices)
    Xq = np.ascontiguousarray(fx["X"][512:576, :8])
    mean, std = pc.predict_gaussian_process(est, Xq, devices=devices, return_std=True)
    return [("alpha_", est.alpha_), ("L_", est.L_), ("mean", mean), ("std", std)]


def _lasso(ml, np, fx, devices):
    from mojolearn import parallel_classical as pc
    est = pc.fit_coordinate_descent(ml.Lasso(alpha=0.01), fx["X"][:2000], fx["yr"][:2000], devices=devices)
    return [("coef_", est.coef_), ("intercept_", est.intercept_)]


def _elasticnet(ml, np, fx, devices):
    from mojolearn import parallel_classical as pc
    est = pc.fit_coordinate_descent(ml.ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=200), fx["X"][:2000],
                                    fx["yr"][:2000], devices=devices)
    return [("coef_", est.coef_), ("intercept_", est.intercept_)]


def _tsvd(ml, np, fx, devices):
    from mojolearn import parallel_classical as pc
    est = pc.fit_gram_estimator(ml.TruncatedSVD(n_components=3, algorithm="covariance_eigh"), fx["X"][:4000],
                                devices=devices)
    return [("components_", est.components_), ("singular_values_", est.singular_values_),
            ("explained_variance_", est.explained_variance_)]


def _pca(ml, np, fx, devices):
    from mojolearn import parallel_classical as pc
    est = pc.fit_gram_estimator(ml.PCA(n_components=4), fx["X"][:4000], devices=devices)
    return [("components_", est.components_), ("explained_variance_", est.explained_variance_),
            ("transform", est.transform(fx["X"][:256]))]


def _ridge(ml, np, fx, devices):
    from mojolearn import parallel_classical as pc
    est = pc.fit_gram_estimator(ml.Ridge(alpha=1.0), fx["X"][:2000], fx["yr"][:2000], devices=devices)
    return [("coef_", est.coef_), ("intercept_", [est.intercept_])]


def _ols(ml, np, fx, devices):
    from mojolearn import parallel_classical as pc
    # every row of the 13000-row fixture: the TSQR row blocks are 4096 rows (tools/logical_shard_identity.py)
    est = pc.fit_gram_estimator(ml.LinearRegression(), np.ascontiguousarray(fx["X"][:, :8]), fx["yr"], devices=devices)
    return [("coef_", est.coef_), ("intercept_", [est.intercept_])]


def _logistic(ml, np, fx, devices):
    from mojolearn import parallel_classical as pc
    X, y = np.ascontiguousarray(fx["Xb"][:2000, :8]), fx["yb"][:2000]
    multi = pc.fit_logistic(ml.LogisticRegression(max_iter=50), X, np.minimum(y, 2), devices=devices)
    binary = pc.fit_logistic(ml.LogisticRegression(max_iter=50), X, (y > 1).astype(np.int32), devices=devices)
    return [("multi coef_", multi.coef_), ("multi intercept_", multi.intercept_),
            ("binary coef_", binary.coef_), ("binary intercept_", binary.intercept_)]


def _kmeans(ml, np, fx, devices):
    from mojolearn import parallel_classical as pc
    X = np.ascontiguousarray(fx["Xb"][:4000, :8])
    est = pc.fit_kmeans(ml.KMeans(n_clusters=4, random_state=0), X, devices=devices)
    return [("cluster_centers_", est.cluster_centers_), ("labels_", est.labels_)]


def _dbscan(ml, np, fx, devices):
    from mojolearn import parallel_classical as pc
    # algorithm='brute': the default 'rbc' bitmap passes have no device plan (tools/logical_shard_identity.py)
    X = np.ascontiguousarray(fx["Xb"][:1500, :4])
    est = pc.fit_dbscan(ml.DBSCAN(eps=0.5, min_samples=5, algorithm="brute"), X, devices=devices)
    return [("labels_", est.labels_)]


def _hdbscan(ml, np, fx, devices):
    from mojolearn import parallel_classical as pc
    est = pc.fit_hdbscan(ml.HDBSCAN(min_cluster_size=5), np.ascontiguousarray(fx["Xb"][:6000, :4]), devices=devices)
    return [("labels_", est.labels_), ("core_distances_", est.core_distances_)]


def _gmm(ml, np, fx, devices):
    from mojolearn import parallel_classical as pc
    X = np.ascontiguousarray(fx["Xb"][:3000, :4])
    est = pc.fit_gaussian_mixture(ml.GaussianMixture(n_components=3, random_state=0), X, devices=devices)
    return [("weights_", est.weights_), ("means_", est.means_), ("covariances_", est.covariances_),
            ("predict_proba", pc.predict_gaussian_mixture(est, X, devices=devices, method="predict_proba")),
            ("predict", pc.predict_gaussian_mixture(est, X, devices=devices, method="predict"))]


def _resample(ml, np, fx, devices):
    from mojolearn import parallel_classical as pc
    x = fx["yr"][:4096]
    b = pc.bootstrap(x, devices=devices, statistic="mean", n_resamples=2048, random_state=3)
    p = pc.permutation_test(fx["yr"][:512], fx["yr"][512:1024], devices=devices, statistic="diff_means",
                            n_resamples=2048, random_state=3)
    r = pc.monte_carlo_integrate("product", [0.0, 0.0], [1.0, 2.0], 65536 + 300, devices=devices, random_state=1)
    # the Monte Carlo integral has no data input: only the two arrays above move under the input fault
    return [("bootstrap distribution", b.distribution), ("bootstrap point_estimate", [b.point_estimate]),
            ("permutation null", p.null_distribution), ("permutation statistic", [p.statistic, p.pvalue]),
            ("monte carlo", [r.integral, r.mean])]


def _cholesky(ml, np, fx, devices):
    from mojolearn import parallel_classical as pc
    M = fx["X"][:384].astype(np.float64)
    A = np.ascontiguousarray((M @ M.T / 384 + np.eye(384)).astype(np.float32))  # test control: an SPD matrix
    B = np.ascontiguousarray(fx["X"][384:768, :4].astype(np.float32))
    est = pc.fit_cholesky(ml.Cholesky(), A, devices=devices)
    return [("L_", est.L_), ("solve", pc.solve_cholesky(est, B, devices=devices))]


def _kernel_ridge(ml, np, fx, devices):
    from mojolearn import parallel_classical as pc
    Y = np.ascontiguousarray(np.stack([fx["yr"][:600], fx["yr"][600:1200]], 1).astype(np.float32))
    X = np.ascontiguousarray(fx["X"][:600, :4])
    est = pc.fit_kernel_method(ml.KernelRidge(alpha=0.1, kernel="rbf", gamma=0.5), X, Y, devices=devices)
    return [("dual_coef_", est.dual_coef_),
            ("predict", pc.apply_kernel_method(est, np.ascontiguousarray(fx["X"][600:856, :4]), devices=devices))]


def _nystroem(ml, np, fx, devices):
    from mojolearn import parallel_classical as pc
    est = pc.fit_kernel_method(ml.Nystroem(kernel="rbf", gamma=0.5, n_components=32, random_state=7),
                               np.ascontiguousarray(fx["X"][:600, :4]), devices=devices)
    return [("components_", est.components_), ("normalization_", est.normalization_),
            ("transform", pc.apply_kernel_method(est, np.ascontiguousarray(fx["X"][600:1624, :4]), devices=devices))]


def _rbf_sampler(ml, np, fx, devices):
    from mojolearn import parallel_classical as pc
    m = ml.RBFSampler(gamma=0.5, n_components=1024, random_state=1).fit(fx["X"])
    return [("transform", pc.transform_rbf_sampler(m, fx["X"][:1000], devices=devices, rows_per_shard=300))]


def _arima(ml, np, fx, devices):
    from mojolearn import parallel_classical as pc
    series = np.ascontiguousarray(fx["X"][:512, :4].T)
    est = pc.fit_arima(ml.ARIMA(order=(1, 0, 0)), series, devices=devices, series_per_shard=1)
    return [("ar_", est.ar_), ("mu_", est.mu_), ("sigma2_", est.sigma2_), ("forecast", est.forecast(24))]


def _hw_series(np, X):
    S = np.cumsum(X[:512, :4], axis=0).T.astype(np.float32) + np.float32(50.0)
    return np.ascontiguousarray((S - S.min(axis=1, keepdims=True) + np.float32(1.0)).astype(np.float32))


def _holtwinters(ml, np, fx, devices):
    from mojolearn import parallel_classical as pc
    est = pc.fit_exponential_smoothing(
        ml.ExponentialSmoothing(_hw_series(np, fx["X"]), seasonal="additive", seasonal_periods=12, ts_num=4),
        devices=devices, series_per_shard=1)
    return [(n, getattr(est, n)) for n in ("level_", "trend_", "season_", "sse_", "alpha_", "beta_", "gamma_")] \
        + [("forecast", est.forecast(24))]


def _forecast_arima(ml, np, fx, devices):
    from mojolearn.parallel_forecasting import predict_arima, forecast_arima
    m = ml.ARIMA(order=(1, 0, 0)).fit(np.ascontiguousarray(fx["X"][:512, :4].T))
    return [("predict", predict_arima(m, 0, m.n_obs_, devices=devices, series_per_shard=1)),
            ("forecast", forecast_arima(m, 24, devices=devices, series_per_shard=1))]


def _forecast_holtwinters(ml, np, fx, devices):
    from mojolearn.parallel_forecasting import forecast_exponential_smoothing, predict_exponential_smoothing
    m = ml.ExponentialSmoothing(_hw_series(np, fx["X"]), seasonal="additive", seasonal_periods=12, ts_num=4).fit()
    return [("predict", predict_exponential_smoothing(m, m.n, m.n + 24, devices=devices, series_per_shard=1)),
            ("forecast", forecast_exponential_smoothing(m, 24, devices=devices, series_per_shard=1)),
            ("indexed", forecast_exponential_smoothing(m, 24, index=2, devices=devices, series_per_shard=1))]


def _scaler(cls_name):
    def recipe(ml, np, fx, devices):
        from mojolearn.parallel_preprocessing import fit_scaler, transform_scaler
        X = fx["X"]
        est = fit_scaler(getattr(ml, cls_name)(), X, devices=devices, columns_per_shard=4)
        t = transform_scaler(est, X[:256], devices=devices, columns_per_shard=4)
        back = transform_scaler(est, t, devices=devices, columns_per_shard=4, inverse=True)
        stats = ("mean_", "var_") if cls_name == "StandardScaler" else ("data_min_", "data_max_", "scale_", "min_")
        return [(n, getattr(est, n)) for n in stats] + [("transform", t), ("inverse", back)]
    return recipe


def _forest(cls_name, target):
    def recipe(ml, np, fx, devices):
        from mojolearn.parallel_ensemble import fit_forest
        est = fit_forest(getattr(ml, cls_name)(n_estimators=16, max_depth=8, random_state=7), fx["X"], fx[target],
                         devices=devices, trees_per_shard=4)
        parts = [("predict", est.predict(fx["X"]))]
        if target == "yc":
            parts.append(("predict_proba", est.predict_proba(fx["X"])))
        return parts
    return recipe


def _forest_pool(ml, np, fx, devices):
    from mojolearn.parallel_ensemble import ParallelForestPredictor
    forest = ml.RandomForestClassifier(n_estimators=40, max_depth=8, random_state=7,
                                       inference_engine="parallel_groves").fit(fx["X"], fx["yc"])
    with ParallelForestPredictor(forest, devices=devices) as pool:
        return [("predict", pool.predict(fx["X"])), ("predict_proba", pool.predict_proba(fx["X"]))]


def _boosting(cls_name, target, **kw):
    def recipe(ml, np, fx, devices):
        from mojolearn.parallel_ensemble import fit_boosting
        est = fit_boosting(getattr(ml, cls_name)(n_estimators=20, max_depth=6, **kw), fx["X"][:2000],
                           fx[target][:2000], devices=devices)
        parts = [("predict", est.predict(fx["X"][:2000]))]
        if target == "yc":
            parts.append(("predict_proba", est.predict_proba(fx["X"][:2000])))
        return parts
    return recipe


def _iforest(ml, np, fx, devices):
    from mojolearn import parallel_ensemble as pe
    X = np.ascontiguousarray(fx["X"][:2000, :8])
    est = pe.fit_isolation_forest(ml.IsolationForest(random_state=0), X, devices=devices)
    return [("score_samples", pe.score_isolation_forest(est, X, devices=devices, method="score_samples"))]


def _ordered_rmse(ml, np, fx, devices):
    from mojolearn.parallel_ensemble import fit_ordered_rmse
    X, y = fx["X"][:2000], fx["yr"][:2000]
    perm = np.random.default_rng(5).permutation(X.shape[0]).astype(np.int64)  # test control: a fixed permutation
    est = fit_ordered_rmse(ml.OrderedRMSE(n_estimators=20, max_depth=6), X, y, permutation=perm, devices=devices)
    return [("predict", est.predict(X))]


def _feature_freq(ml, np, fx, devices):
    from mojolearn.parallel_ensemble import fit_feature_freq
    X = fx["X"][:2000]
    c0 = (X[:, 4] > np.median(X[:, 4])).astype(np.float32)
    c1 = (2 * (X[:, 5] > np.median(X[:, 5])).astype(np.int32)
          + (X[:, 6] > np.median(X[:, 6])).astype(np.int32)).astype(np.float32)
    Xc = np.ascontiguousarray(np.column_stack([c0, c1, X[:, :8]]).astype(np.float32))
    est = fit_feature_freq(ml.ExperimentalTwoLevelFeatureFreq(sources=[0, 1], random_state=7), Xc, fx["yr"][:2000],
                           devices=devices)
    return [("predict", est.predict(Xc))]


def _queries(kind):
    def recipe(ml, np, fx, devices):
        from mojolearn.parallel_neighbors import ParallelQueries
        X = fx["X"]
        index, q = X[:4096], np.ascontiguousarray(X[4096:4160])
        parts = []
        if kind == "knn":
            m = ml.KNeighborsClassifier(n_neighbors=8).fit(index, fx["yc"][:4096])
            methods = ("kneighbors", "predict", "predict_proba")
        elif kind == "nn":
            m, methods = ml.NearestNeighbors(n_neighbors=8).fit(index), ("kneighbors",)
        elif kind == "kde":
            m = ml.KernelDensity(bandwidth=0.7).fit(np.ascontiguousarray(index[:, :4]))
            q, methods = np.ascontiguousarray(X[4096:4352, :4]), ("score_samples",)
        else:
            # test control: a radius that gives each query a few neighbours (the 5 percent distance quantile)
            d = np.sqrt(((q[:, None, :].astype(np.float64) - index[None, :512, :]) ** 2).sum(-1))
            radius = float(np.float32(np.quantile(d, 0.05)))
            m, methods = ml.RadiusNeighbors(radius=radius).fit(index), ("radius_neighbors",)
        with ParallelQueries(m, devices=devices, rows_per_shard=16) as pq:
            for method in methods:
                kw = dict(sort_results=True) if method == "radius_neighbors" else {}
                parts.append((method, pq.query(q, method=method, **kw)))
        return parts
    return recipe


def _reference(cls_name, target):
    def recipe(ml, np, fx, devices):
        from mojolearn.parallel_neighbors_reference import ReferenceShardedNeighbors
        X = fx["X"]
        m = getattr(ml, cls_name)(n_neighbors=8).fit(X[:4096], fx[target][:4096])
        q = np.ascontiguousarray(X[4096:4160])
        with ReferenceShardedNeighbors(m, devices=devices, reference_rows_per_shard=1024,
                                       query_rows_per_shard=16) as rs:
            parts = [("kneighbors", rs.kneighbors(q)), ("predict", rs.predict(q))]
            if target == "yc":
                parts.append(("predict_proba", rs.predict_proba(q)))
        return parts
    return recipe


def _graph(kind):
    def recipe(ml, np, fx, devices):
        from mojolearn.parallel_graph import fit_graph, transform_umap
        if kind == "agglomerative":
            est = fit_graph(ml.AgglomerativeClustering(n_clusters=4), np.ascontiguousarray(fx["Xb"][:2000, :4]),
                            devices=devices)
            return [("labels_", est.labels_), ("children_", est.children_)]
        if kind == "spectral":
            est = fit_graph(ml.SpectralClustering(n_clusters=4, random_state=3),
                            np.ascontiguousarray(fx["Xb"][:2000, :4]), devices=devices)
            return [("labels_", est.labels_), ("embedding_", est.embedding_)]
        est = fit_graph(ml.UMAP(n_neighbors=8, n_components=2, n_epochs=8, random_state=3),
                        np.ascontiguousarray(fx["X"][:1024, :8]), devices=devices)
        return [("embedding_", est.embedding_),
                ("transform", transform_umap(est, np.ascontiguousarray(fx["X"][1024:1088, :8]), devices=devices))]
    return recipe


def _ivf(ml, np, fx, devices):
    from mojolearn.parallel_ivf import DistributedIVFIndex
    X = fx["X"]
    q = np.ascontiguousarray(X[4096:4160])
    m = ml.IVFIndex(n_lists=16, n_probes=4, n_neighbors=8, random_state=3).fit(X[:4096])
    with DistributedIVFIndex.from_index(m, devices=devices) as shards:
        d, i = shards.search(q)
        # n_candidates_ is per shard and so differs between one and N devices by design: not a compared part
        return [("distances", d), ("ids", i)]


def _gpc(ml, np, fx, devices):
    from mojolearn.parallel_gaussian_process import (fit_gaussian_process_classifier,
                                                     predict_gaussian_process_classifier)
    X = fx["X"]
    xs, q = np.ascontiguousarray(X[:256, :4]), np.ascontiguousarray(X[256:320, :4])
    # three classes, so the one-vs-rest fit has more than one binary problem to place
    y3 = ((X[:256, 3] > 0).astype(np.int32) + (X[:256, 4] > 0).astype(np.int32)).astype(np.int32)
    kernel = ml.ConstantKernel(1.0) * ml.RBF(1.0)
    est = fit_gaussian_process_classifier(ml.GaussianProcessClassifier(kernel=kernel), xs, y3, devices=devices)
    return [("L_", [e.L_ for e in est.estimators_]), ("pi_", [e.pi_ for e in est.estimators_]),
            ("proba", predict_gaussian_process_classifier(est, q, devices=devices, method="predict_proba")),
            ("predict", predict_gaussian_process_classifier(est, q, devices=devices, method="predict"))]


def _cross_val(ml, np, fx, devices):
    from mojolearn.parallel_model_selection import cross_val_score
    est = ml.GradientBoostingRegressor(n_estimators=8, max_depth=4)
    # one fold per worker at most: as many folds as devices, and never fewer than three
    return [("scores", cross_val_score(est, fx["X"][:2000], fx["yr"][:2000], devices=devices,
                                       cv=max(3, len(devices))))]


def _mlp(ml, np, fx, devices):
    from mojolearn.parallel_training import ParallelNeuralTrainer
    w = [((np.arange(int(np.prod(s)), dtype=np.float32) % 7 - 3) / 32).reshape(s).astype(np.float32)
         for s in ((16, 8), (16,), (3, 16), (3,))]
    m = ml.SmallMLPTrainer(*w, data_schedule={"dataset": "multi_gpu_checks", "order": "sequential"})
    X = fx["X"]
    Xm = np.ascontiguousarray(X[:576, :8])
    t = ((X[:576, 3] > 0).astype(np.int32) + (X[:576, 4] > 0).astype(np.int32)).astype(np.int32)
    # three steps of 192 rows; each step is cut into one shard per logical shard (three, or one per device)
    logical = max(3, len(devices))
    per = 192 // logical
    steps = [[(Xm[192 * s + per * k: 192 * s + per * (k + 1)], t[192 * s + per * k: 192 * s + per * (k + 1)])
              for k in range(logical)] for s in range(3)]
    with ParallelNeuralTrainer(m, devices=devices, logical_shards=logical) as tr:
        for shards in steps:
            tr.train_step(shards)
    return [("weights_[%s]" % k, m.weights_[k]) for k in sorted(m.weights_)] \
        + [("logits", m.predict_logits(Xm[:256]))]


def _samba(max_norm):
    def recipe(ml, np, fx, devices):
        from mojolearn.parallel_training import ParallelNeuralTrainer
        cfg = ml.SambaConfig(vocab=256, d_model=32, layers=("mamba3", "attention"), n_heads=2, intermediate=64)
        kw = {} if max_norm is None else dict(max_norm=max_norm)
        m = ml.SambaStack(cfg, generator=ml.training.Generator(1), lr=1e-3, **kw)
        ids = token_ids(np, fx["X"], 8, 17)
        steps = [[(ids[4 * s + 2 * k: 4 * s + 2 * k + 2, :-1], ids[4 * s + 2 * k: 4 * s + 2 * k + 2, 1:])
                  for k in range(2)] for s in range(2)]
        with ParallelNeuralTrainer(m, devices=devices, logical_shards=2) as tr:
            for shards in steps:
                tr.train_step(shards)
            grads = tr.export_gradients()
        params = m.parameters()
        return [("parameters[%s]" % k, params[k]) for k in sorted(params)] \
            + [("gradients", list(grads)), ("forward", m.forward(ids[:2, :-1]))]
    return recipe


def _byte_lm(trainer_module, trainer_name, **options):
    def recipe(ml, np, fx, devices):
        import importlib
        cls = getattr(importlib.import_module(trainer_module), trainer_name)
        shape = ml.ByteLanguageModelConfig()
        # test control: the weights are the same in every arm; the fixture is the token stream
        rng = np.random.default_rng(31)
        named = {}
        for name, shp in zip(shape.parameter_names, shape.parameter_shapes):
            if name.endswith("norm1_w") or name.endswith("norm2_w"):
                named[name] = np.ones(shp, dtype=np.float32)
            else:
                named[name] = rng.uniform(-0.125, 0.125, size=shp).astype(np.float32)
        seed = ml.SmallByteLanguageModelTrainer(
            named, data_schedule={"dataset": "multi_gpu_checks", "order": "sequential"}, shape=shape)
        ids = token_ids(np, fx["X"], 6 * shape.batch, shape.length + 1)
        losses = []
        with cls(seed.state_dict(), devices=devices, logical_shards=2, **options) as tr:
            for step in range(3):
                got = tr.train_step([ids[4 * step: 4 * step + 2], ids[4 * step + 2: 4 * step + 4]])
                losses.append([float(v) for v in got["losses"]])
            state = tr.state_dict()
            grads = tr.export_gradients()
        return [("losses", np.asarray(losses, dtype=np.float64)), ("parameters", state["parameters"]),
                ("m", state["m"]), ("v", state["v"]), ("gradients", grads)]
    return recipe


def _causal_lm(ml, np, fx, devices):
    import tempfile
    from mojolearn import _causal_lm_fixtures as fixtures
    from mojolearn.models import ParallelCausalLM
    ids = token_ids(np, fx["X"], 4, 12, 64)
    cfg, tensors = fixtures.family_fixture("llama", False)
    with tempfile.TemporaryDirectory(prefix="multi-gpu-checks-causal-lm-") as d:
        root = fixtures._write_checkpoint(os.path.join(d, "llama"), cfg, tensors)
        plain = ml.models.causal_lm.CausalLM.load(root, numeric_profile="fp32_v1")
        n_layers = plain.plan.n_layers
        owners = tuple(devices[i % len(devices)] for i in range(n_layers))
        with ParallelCausalLM.load(root, layer_devices=owners, numeric_profile="fp32_v1") as par:
            return [("logits", np.ascontiguousarray(np.asarray(par.forward(ml.Array.from_buffer(ids)))))]


def _distributed_program(ml, np, fx, devices):
    from mojolearn import ops
    X = np.ascontiguousarray(fx["X"][:19, :3])
    y = np.ascontiguousarray(fx["X"][:19, 3:5])
    rng = np.random.default_rng(104)  # test control: the same starting weights in every arm
    with ops.Session(numeric_mode="identical") as s:
        x = s.tensor(np.zeros((3, 3), np.float32))
        w1 = s.tensor(rng.normal(0, 0.1, size=(3, 5)).astype(np.float32))
        bias = s.tensor(np.zeros((1, 5), np.float32))
        w2 = s.tensor(rng.normal(0, 0.1, size=(5, 2)).astype(np.float32))
        h = s.dropout(s.tanh(s.linear(x, w1, bias)), p=0.25, seed=17)
        p = s.capture([x], [s.matmul(h, w2)])
        opt = s.optimizer("adamw", lr=0.01, weight_decay=0.02, schedule="step", step_size=2, gamma=0.5)
        with p.distribute(devices, logical_shards=max(4, len(devices)), optimizer=opt, seed=91) as d:
            losses = d.fit(X, targets=y, epochs=5, batch_size=7)
            if tuple(d.residency()["devices"]) != tuple(devices):
                raise RuntimeError("DistributedProgram.residency() does not name the requested devices")
            state = d.state()
            return [("losses", losses), ("parameters", list(state["parameters"])), ("run", d.run(X))]


# ---------------------------------------------------------------------------------------------------- THE LIST
# unit, driver module, recipe, witness (plan | seen | none), fault (shift | input), fixture rows.
# One line per unit; adding a unit = adding a line and a recipe. `plan` is given only to the units whose split
# rule is written down in tools/logical_shard_identity.py THE LIST (the same driver calls on the same sizes).
UNITS = (
    ("svm-rbf", "parallel_classical", _svm, "plan", "input", ROWS),
    ("svm-svr", "parallel_classical", _svr, "seen", "input", ROWS),
    ("gp-regress", "parallel_classical", _gp, "plan", "input", ROWS),
    ("lasso-cd", "parallel_classical", _lasso, "plan", "input", ROWS),
    ("elasticnet-cd", "parallel_classical", _elasticnet, "seen", "input", ROWS),
    ("tsvd-gram", "parallel_classical", _tsvd, "plan", "input", ROWS),
    ("pca-gram", "parallel_classical", _pca, "seen", "input", ROWS),
    ("ridge-gram", "parallel_classical", _ridge, "plan", "input", ROWS),
    ("ols-tsqr", "parallel_classical", _ols, "plan", "input", 13000),
    ("glm-logistic", "parallel_classical", _logistic, "plan", "input", ROWS),
    ("kmeans", "parallel_classical", _kmeans, "plan", "input", ROWS),
    ("dbscan", "parallel_classical", _dbscan, "plan", "input", ROWS),
    ("hdbscan", "parallel_classical", _hdbscan, "seen", "input", ROWS),
    ("gmm", "parallel_classical", _gmm, "plan", "input", ROWS),
    ("resample", "parallel_classical", _resample, "plan", "input", ROWS),
    ("cholesky", "parallel_classical", _cholesky, "plan", "input", ROWS),
    ("kernel-ridge", "parallel_classical", _kernel_ridge, "seen", "input", ROWS),
    ("nystroem", "parallel_classical", _nystroem, "seen", "input", ROWS),
    ("rbf-sampler", "parallel_classical", _rbf_sampler, "none", "shift", ROWS),
    ("arima-fit", "parallel_classical", _arima, "none", "shift", ROWS),
    ("holtwinters-fit", "parallel_classical", _holtwinters, "none", "shift", ROWS),
    ("arima-forecast", "parallel_forecasting", _forecast_arima, "none", "input", ROWS),
    ("holtwinters-forecast", "parallel_forecasting", _forecast_holtwinters, "none", "input", ROWS),
    ("scaler-standard", "parallel_preprocessing", _scaler("StandardScaler"), "none", "shift", ROWS),
    ("scaler-minmax", "parallel_preprocessing", _scaler("MinMaxScaler"), "none", "shift", ROWS),
    ("forest-rf-clf", "parallel_ensemble", _forest("RandomForestClassifier", "yc"), "none", "shift", ROWS),
    ("forest-rf-reg", "parallel_ensemble", _forest("RandomForestRegressor", "yr"), "none", "shift", ROWS),
    ("forest-et-clf", "parallel_ensemble", _forest("ExtraTreesClassifier", "yc"), "none", "shift", ROWS),
    ("forest-et-reg", "parallel_ensemble", _forest("ExtraTreesRegressor", "yr"), "none", "shift", ROWS),
    ("forest-predict-pool", "parallel_ensemble", _forest_pool, "plan", "input", ROWS),
    ("boosting-clf", "parallel_ensemble", _boosting("GradientBoostingClassifier", "yc"), "seen", "input", ROWS),
    ("boosting-reg", "parallel_ensemble", _boosting("GradientBoostingRegressor", "yr"), "plan", "input", ROWS),
    ("iforest", "parallel_ensemble", _iforest, "plan", "input", ROWS),
    ("ordered-rmse", "parallel_ensemble", _ordered_rmse, "seen", "input", ROWS),
    ("feature-freq", "parallel_ensemble", _feature_freq, "seen", "input", ROWS),
    ("queries-knn", "parallel_neighbors", _queries("knn"), "none", "shift", ROWS),
    ("queries-nn", "parallel_neighbors", _queries("nn"), "none", "shift", ROWS),
    ("queries-radius", "parallel_neighbors", _queries("radius"), "none", "shift", ROWS),
    ("queries-kde", "parallel_neighbors", _queries("kde"), "none", "shift", ROWS),
    ("reference-knn-clf", "parallel_neighbors_reference", _reference("KNeighborsClassifier", "yc"), "none", "shift",
     ROWS),
    ("reference-knn-reg", "parallel_neighbors_reference", _reference("KNeighborsRegressor", "yr"), "none", "shift",
     ROWS),
    ("graph-agglomerative", "parallel_graph", _graph("agglomerative"), "seen", "input", ROWS),
    ("graph-spectral", "parallel_graph", _graph("spectral"), "seen", "input", ROWS),
    ("graph-umap", "parallel_graph", _graph("umap"), "seen", "input", ROWS),
    ("ivf-search", "parallel_ivf", _ivf, "none", "input", ROWS),
    ("gpc", "parallel_gaussian_process", _gpc, "none", "input", ROWS),
    ("cross-val", "parallel_model_selection", _cross_val, "none", "input", ROWS),
    ("neural-mlp", "parallel_training", _mlp, "none", "input", ROWS),
    ("neural-samba", "parallel_training", _samba(None), "none", "input", ROWS),
    ("neural-samba-clip", "parallel_training", _samba(0.5), "none", "input", ROWS),
    ("byte-lm", "parallel_training", _byte_lm("mojolearn.parallel_training", "ParallelByteLanguageModelTrainer"),
     "none", "input", ROWS),
    ("byte-lm-model-pool", "parallel_training",
     _byte_lm("mojolearn.parallel_training", "ParallelByteLanguageModelTrainer", schedule="model_pool"),
     "none", "input", ROWS),
    ("causal-lm", "models.parallel_causal_lm", _causal_lm, "none", "input", ROWS),
    ("distributed-program", "_distributed_program", _distributed_program, "none", "input", ROWS),
)

#: Where each of the 59 retired par-* lanes of the removed verification went: the unit of this suite that calls the
#: same driver, or a sentence saying why none does. The names and their old reference cells are history in
#: multi_gpu_retired_par_cells.json (moved out of the shipped verify table); nothing here reads those cells.
RETIRED_PAR_LANES = {
    "par-arima": "arima-fit", "par-boosting": "boosting-clf", "par-boosting-clf": "boosting-clf",
    "par-boosting-pointwise": "not covered: use_pointwise_searcher is a constructor option of the removed harness "
                              "configuration; boosting-reg reaches gbdt/methods/pointwise_multi_gpu.mojo",
    "par-boosting-reg": "boosting-reg",
    "par-border-types": "not covered: one fit per feature_border_type; boosting-clf covers the driver",
    "par-byte-lm": "byte-lm", "par-byte-lm-model-pool": "byte-lm-model-pool",
    "par-byte-lm-offload": "not covered: ParallelByteLanguageModelTrainer(schedule='offload') is one device by design",
    "par-causal-lm": "causal-lm", "par-cd": "lasso-cd", "par-cd-elasticnet": "elasticnet-cd",
    "par-cholesky": "cholesky", "par-cross-val": "cross-val", "par-dbscan": "dbscan",
    "par-feature-freq": "feature-freq", "par-forecast-arima": "arima-forecast",
    "par-forecast-holtwinters": "holtwinters-forecast", "par-forest": "forest-rf-clf",
    "par-forest-et": "forest-et-reg", "par-forest-et-clf": "forest-et-clf", "par-forest-pool": "forest-predict-pool",
    "par-forest-reg": "forest-rf-reg", "par-gmm": "gmm", "par-gp": "gp-regress", "par-gpc-fit": "gpc",
    "par-gpc-predict": "gpc", "par-gram": "ridge-gram", "par-gram-ols": "ols-tsqr", "par-gram-pca": "pca-gram",
    "par-gram-tsvd": "tsvd-gram", "par-graph-agglomerative": "graph-agglomerative",
    "par-graph-spectral": "graph-spectral", "par-graph-umap": "graph-umap", "par-hdbscan": "hdbscan",
    "par-holtwinters": "holtwinters-fit", "par-iforest": "iforest", "par-ivf": "ivf-search",
    "par-kernel-ridge": "kernel-ridge", "par-kmeans": "kmeans", "par-logistic": "glm-logistic",
    "par-mlp": "neural-mlp", "par-nystroem": "nystroem",
    "par-ordered": "not covered: boosting_type='Ordered' through fit_boosting; ordered-rmse covers the ordered driver",
    "par-ordered-rmse": "ordered-rmse", "par-queries-kde": "queries-kde", "par-queries-knn": "queries-knn",
    "par-queries-nn": "queries-nn", "par-queries-radius": "queries-radius", "par-rbf-sampler": "rbf-sampler",
    "par-reference-knn": "reference-knn-clf", "par-reference-knn-reg": "reference-knn-reg",
    "par-resample": "resample", "par-samba": "neural-samba", "par-samba-clip": "neural-samba-clip",
    "par-scaler": "scaler-standard", "par-scaler-minmax": "scaler-minmax", "par-svm": "svm-rbf",
    "par-svm-svr": "svm-svr",
}


def unit_names():
    return [row[0] for row in UNITS]


def unit_row(name):
    return {row[0]: row for row in UNITS}[name]


def unit_errors(units=UNITS):
    """Why the list is malformed, or [] (checked before anything runs, and by the tools test)."""
    errors = []
    names = [row[0] for row in units]
    if len(set(names)) != len(names):
        errors.append("duplicate unit names")
    for row in units:
        if len(row) != 6:
            errors.append("%s: a unit line has 6 fields" % (row[0],))
            continue
        name, module, recipe, witness, fault, rows = row
        if not callable(recipe):
            errors.append("%s: the recipe is not callable" % name)
        if witness not in ("plan", "seen", "none"):
            errors.append("%s: witness %r" % (name, witness))
        if fault not in ("shift", "input"):
            errors.append("%s: fault %r" % (name, fault))
        if type(rows) is not int or rows < 4352:
            errors.append("%s: fixture rows must be an int of at least 4352 (the recipes slice that far)" % name)
        if not (isinstance(module, str) and module):
            errors.append("%s: no driver module" % name)
    for lane, target in RETIRED_PAR_LANES.items():
        if not target.startswith("not covered: ") and target not in names:
            errors.append("%s maps to %s, which is not a unit" % (lane, target))
    return errors


# ---------------------------------------------------------------------------------------------------- DIGESTS
def flatten(np, name, value):
    """[(name, ndarray), ...] of one recipe output: nested tuples, lists and dicts (a ragged radius result, a
    (distances, indices) pair, a parameter dict) become one array per leaf, named by position."""
    if value is None:
        return [(name + " (none)", np.zeros(0, dtype=np.uint8))]
    if isinstance(value, dict):
        out = []
        for key in sorted(value, key=str):
            out.extend(flatten(np, "%s.%s" % (name, key), value[key]))
        return out
    if isinstance(value, (tuple, list)):
        if all(isinstance(v, (bool, int, float, np.generic)) for v in value):
            return [(name, np.asarray(value))]
        out = [(name + " (length)", np.asarray([len(value)], dtype=np.int64))]
        for k, item in enumerate(value):
            out.extend(flatten(np, "%s[%d]" % (name, k), item))
        return out
    array = np.asarray(value)
    if array.dtype == object:
        return flatten(np, name, list(array.reshape(-1)))
    return [(name, array)]


def part_digest(np, array):
    array = np.ascontiguousarray(array)
    return dict(shape=list(array.shape), dtype=array.dtype.str, sha256=hashlib.sha256(array.tobytes()).hexdigest())


def arm_digest(parts):
    """The short digest of one arm: sha256 over its parts' (name, shape, dtype, sha256) lines, or None without
    parts (an arm that produced nothing never matches)."""
    if not parts:
        return None
    text = "\n".join("%s %s %s %s" % (p["name"], p["shape"], p["dtype"], p["sha256"]) for p in parts)
    return hashlib.sha256(text.encode()).hexdigest()


def one_line(text):
    return " ".join(str(text).split())[:240]


def first_difference(a, b):
    """The first part whose digest differs between two arms, as a short phrase."""
    pa, pb = {p["name"]: p for p in a or []}, {p["name"]: p for p in b or []}
    for name in [p["name"] for p in a or []] + [n for n in pb if n not in pa]:
        if pa.get(name) != pb.get(name):
            return name
    return "?"


# ---------------------------------------------------------------------------------------------------- VERDICTS
def unit_verdict(name, arms, n_devices, want_fault=True, strict_spread=False):
    """(verdict, reason, {arm: digest}) of one unit from its arm documents {arm: doc or None}.

    OK      one == multi, the placement witness (where the unit has one) saw two or more ranks, and the fault arm
            differs from the one arm
    DIFFER  the multi arm's digest is not the one arm's: the defect this suite exists to find
    FAIL    anything missing, raised or timed out; a multi arm that stayed on one rank; a fault that was not seen
    """
    _, _module, _recipe, witness, fault, _rows = unit_row(name)
    digests = {}
    needed = ARMS if want_fault else ARMS[:2]
    for arm in needed:
        doc = arms.get(arm)
        if not isinstance(doc, dict) or doc.get("unit") != name or doc.get("arm") != arm:
            return "FAIL", "arm %s: no worker result" % arm, digests
        if doc.get("error"):
            return "FAIL", "arm %s: %s" % (arm, one_line(doc["error"])), digests
        d = arm_digest(doc.get("parts"))
        if d is None:
            return "FAIL", "arm %s produced no digest" % arm, digests
        digests[arm] = d
    if len(arms["one"].get("devices") or []) != 1 or len(arms["multi"].get("devices") or []) != n_devices:
        return "FAIL", "the arms did not run on 1 and %d devices" % n_devices, digests
    if digests["multi"] != digests["one"]:
        return ("DIFFER", "multi differs from one at %r" % first_difference(arms["one"]["parts"],
                                                                           arms["multi"]["parts"]), digests)
    ranks = arms["multi"].get("ranks")
    if witness == "plan" or (strict_spread and witness == "seen"):
        if type(ranks) is not int:
            why = arms["multi"].get("rank_errors") or []
            return ("FAIL", "multi: no rank record (no loaded binding reports the device-plan work counter)%s"
                    % (("; " + one_line(why[0])) if why else ""), digests)
        if ranks < MIN_SPREAD:
            return ("FAIL", "multi spread its work over %d rank(s), not >= %d: a trivial match" % (ranks, MIN_SPREAD),
                    digests)
        if ranks > n_devices:
            return "FAIL", "multi reports %d ranks, more than its %d devices" % (ranks, n_devices), digests
    if want_fault and digests["fault"] == digests["one"]:
        return ("FAIL", "the %s fault was not seen: the fault arm's digest equals the one arm's, so this check "
                "cannot fail as written" % fault, digests)
    return "OK", "", digests


def summary_errors(doc, units=None):
    """Why a summary does not show every selected unit OK, or []."""
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
        return ["no multi-GPU checks summary"]
    errors = []
    rows = doc.get("units") or {}
    want = sorted(units if units is not None else doc.get("selected") or [])
    if sorted(rows) != want:
        errors.append("units %s, not the selection %s" % (sorted(rows), want))
    if not isinstance(doc.get("devices"), list) or len(doc["devices"]) < 2:
        errors.append("fewer than two devices")
    for name, row in sorted(rows.items()):
        if (row or {}).get("verdict") != "OK":
            errors.append("%s %s %s" % (name, (row or {}).get("verdict"), (row or {}).get("reason", "")))
    return errors


def check_line(name, verdict, reason, digests, ranks=None):
    arms = " ".join("%s=%s" % (arm, (digests.get(arm) or "-")[:12]) for arm in ARMS)
    seen = "" if ranks is None else " ranks=%s" % ranks
    return ("MULTIGPU %s %s%s %s %s" % (name, arms, seen, verdict, reason)).rstrip()


# ---------------------------------------------------------------------------------------------------- WORKER
def ranks_take(modules, errors=None):
    """The ranks that received work since the last take, over every loaded binding (`device_plan_log()` ->
    (plans, ranks)); the record is cleared. None when no loaded binding reports ranks. As
    tools/logical_shard_identity.py `ranks_take`."""
    best = None
    seen = set()
    for module in list(modules.values()):
        if id(module) in seen:
            continue
        seen.add(id(module))
        try:
            log = getattr(module, "device_plan_log", None)
        except Exception:
            continue
        if not callable(log):
            continue
        try:
            got = log()
        except Exception as error:
            if errors is not None:
                errors.append(one_line("%s.device_plan_log: %s: %s" % (getattr(module, "__name__", "?"),
                                                                        type(error).__name__, error)))
            continue
        if isinstance(got, tuple) and len(got) == 2 and type(got[1]) is int:
            best = got[1] if best is None else max(best, got[1])
    return best


def worker(name, arm, devices, out_path):
    """One (unit, arm) in this (installed-wheel) interpreter; writes the arm's document to out_path."""
    doc = dict(unit=name, arm=arm, devices=list(devices), parts=[], error=None)
    started = time.time()
    try:
        _, _module, recipe, _witness, fault, rows = unit_row(name)
        import numpy as np
        import mojolearn as ml
        doc.update(mojolearn_version=getattr(ml, "__version__", None), mojolearn_file=ml.__file__)
        fx = fixture(np, 1000 + unit_names().index(name), rows=rows, fault=(arm == "fault" and fault == "input"))
        ranks_take(sys.modules)  # clear what the imports recorded
        for part, value in recipe(ml, np, fx, tuple(devices)):
            for leaf, array in flatten(np, part, value):
                doc["parts"].append(dict(name=leaf, **part_digest(np, array)))
        rank_errors = []
        doc["ranks"] = ranks_take(sys.modules, rank_errors)
        if rank_errors:
            doc["rank_errors"] = rank_errors[:4]
    except Exception as error:  # fail closed: recorded, judged FAIL
        doc["error"] = "%s: %s" % (type(error).__name__, error)
        doc["parts"] = []
    doc["seconds"] = round(time.time() - started, 3)
    try:
        import resource
        doc["peak_rss_kb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:
        pass
    Path(out_path).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return 0


def worker_env(arm, fault):
    """The worker's environment: the shipped defaults of the installed wheel. Every MOJOLEARN_* variable of the box
    and every *_DEVICE_COUNT is dropped (a job-level knob would change a route or keep a unit on one rank), then
    the mode is set, and the two switches of the native negative control for a shift fault arm only."""
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONPATH", "PYTHONHOME") and not k.endswith("_DEVICE_COUNT")
           and not k.startswith("MOJOLEARN_")}
    env["MOJOLEARN_NUMERIC_MODE"] = "identical"
    env["PYTHONUNBUFFERED"] = "1"
    if arm == "fault" and fault == "shift":
        env.update(FAULT_ENV)
    return env


def run_arm(python, name, arm, devices, out, timeout):
    """Run one (unit, arm) in its own process (an external cwd, isolated mode: the installed wheel, never the
    source tree); returns the arm's document, or a document carrying the reason it has none."""
    work = Path(out) / "arms"
    work.mkdir(parents=True, exist_ok=True)
    stem = "%s.%s" % (name, arm)
    result, log = work / (stem + ".json"), work / (stem + ".log")
    if result.exists():
        result.unlink()
    argv = [python, "-I", str(Path(__file__).resolve()), "--worker", name, "--arm", arm,
            "--devices", ",".join(str(d) for d in devices), "--result", str(result)]
    failed = dict(unit=name, arm=arm, devices=list(devices), parts=[])
    try:
        with log.open("w") as fh:
            proc = subprocess.run(argv, cwd=str(work), env=worker_env(arm, unit_row(name)[4]), stdout=fh,
                                  stderr=subprocess.STDOUT, timeout=timeout)
    except subprocess.TimeoutExpired:
        return dict(failed, error="timed out after %ss" % timeout)
    try:
        doc = json.loads(result.read_text())
    except (OSError, ValueError):
        return dict(failed, error="worker exited %s without a result (see arms/%s.log)" % (proc.returncode, stem))
    if proc.returncode != 0 and not doc.get("error"):
        doc["error"] = "worker exited %s" % proc.returncode
    return doc


# ---------------------------------------------------------------------------------------------------- THE BOX
def alloc_cores():
    """The cores of this allocation: affinity, capped by the cgroup quota (tools/ops/box_cores_lib.sh
    `alloc_cores`), not the host's visible CPUs."""
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:
        cores = os.cpu_count() or 1
    for path, parse in (("/sys/fs/cgroup/cpu.max", lambda t: t.split()),
                        ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", None)):
        try:
            text = Path(path).read_text().strip()
            if parse:
                quota, period = parse(text)
            else:
                quota, period = text, Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text().strip()
            if quota != "max" and int(quota) > 0:
                cores = min(cores, -(-int(quota) // int(period)))
            break
        except (OSError, ValueError):
            continue
    return max(1, cores)


def alloc_mem_gb():
    """Memory available to this allocation in GiB (MemAvailable, capped by the cgroup's memory.max), or None."""
    try:
        kb = [int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()
              if line.startswith("MemAvailable")][0]
    except (OSError, IndexError, ValueError):
        return None
    try:
        cap = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if cap != "max":
            kb = min(kb, int(cap) // 1024)
    except (OSError, ValueError):
        pass
    return kb / 1048576.0


def detect_gpus():
    """(vendor, [free MiB per GPU]) from nvidia-smi or rocm-smi, or (None, []) when neither answers."""
    try:
        got = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=60)
        if got.returncode == 0:
            free = [int(line.strip()) for line in got.stdout.splitlines() if line.strip()]
            if free:
                return "nvidia", free
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    for tool in ("rocm-smi", "/opt/rocm/bin/rocm-smi"):
        try:
            got = subprocess.run([tool, "--showmeminfo", "vram", "--json"], capture_output=True, text=True,
                                 timeout=60)
            if got.returncode != 0:
                continue
            free = []
            for _card, row in sorted(json.loads(got.stdout).items()):
                if not isinstance(row, dict) or "VRAM Total Memory (B)" not in row:
                    continue
                free.append((int(row["VRAM Total Memory (B)"]) - int(row.get("VRAM Total Used Memory (B)", 0)))
                            // 1048576)
            if free:
                return "amd", free
        except (OSError, ValueError, subprocess.TimeoutExpired):
            continue
    return None, []


def plan_workers(n_jobs, cores, host_gb, gpu_free_mib, host_gb_per_job, gpu_gb_per_job, asked=0):
    """(workers, {bound: value}): the smallest of the jobs, the cores and the two memory bounds; `asked` overrides
    the cores bound only (the memory bounds still hold). Never less than 1."""
    bounds = dict(jobs=n_jobs, cores=asked or cores)
    if host_gb is not None and host_gb_per_job > 0:
        bounds["host_memory"] = max(1, int(host_gb // host_gb_per_job))
    if gpu_free_mib and gpu_gb_per_job > 0:
        bounds["gpu_memory"] = max(1, int(min(gpu_free_mib) / 1024.0 // gpu_gb_per_job))
    return max(1, min(bounds.values())), bounds


def jobs_for(names, devices, want_fault=True):
    """Every (unit, arm, devices) of the run. The one arms rotate over the GPUs, so a box's cards share them."""
    jobs = []
    for k, name in enumerate(names):
        jobs.append((name, "one", (devices[k % len(devices)],)))
        jobs.append((name, "multi", tuple(devices)))
        if want_fault:
            jobs.append((name, "fault", tuple(devices)))
    # the longest-looking arms first is not known before a first run; submit in list order
    return jobs


def parse_devices(text):
    devices = tuple(int(x) for x in text.split(",") if x.strip() != "")
    if len(set(devices)) != len(devices) or any(d < 0 for d in devices):
        raise ValueError("devices must be distinct nonnegative indices")
    return devices


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--python", default=sys.executable, help="the interpreter the wheels are installed in")
    ap.add_argument("--devices", default="", help="comma-separated device indices (default: every GPU found)")
    ap.add_argument("--units", default="", help="comma-separated unit names (default: every unit)")
    ap.add_argument("--out", default="")
    ap.add_argument("--jobs", type=int, default=0, help="arms at once (default: the allocation's cores)")
    ap.add_argument("--host-gb-per-job", type=float, default=2.0, help="host memory per arm (an estimate)")
    ap.add_argument("--gpu-gb-per-job", type=float, default=1.5, help="device memory per arm per GPU (an estimate)")
    ap.add_argument("--timeout", type=int, default=UNIT_TIMEOUT, help="seconds per arm")
    ap.add_argument("--no-fault", action="store_true", help="skip the fault arms (the verdicts then say so)")
    ap.add_argument("--strict-spread", action="store_true", help="judge the placement of `seen` units too")
    ap.add_argument("--list", action="store_true", help="print the units and exit")
    ap.add_argument("--worker", default="", help=argparse.SUPPRESS)
    ap.add_argument("--arm", default="", help=argparse.SUPPRESS)
    ap.add_argument("--result", default="", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.worker:
        return worker(a.worker, a.arm, parse_devices(a.devices), a.result)
    errors = unit_errors()
    if errors:
        print("MULTIGPU list refused: " + "; ".join(errors))
        return 2
    if a.list:
        for name, module, _recipe, witness, fault, rows in UNITS:
            print("%-24s %-32s witness=%-5s fault=%-5s rows=%d" % (name, module, witness, fault, rows))
        return 0
    if not a.out:
        ap.error("--out is required")
    names = [n for n in a.units.split(",") if n] or unit_names()
    unknown = sorted(set(names) - set(unit_names()))
    if unknown:
        print("MULTIGPU unknown units: " + " ".join(unknown))
        return 2
    vendor, gpu_free = detect_gpus()
    try:
        devices = parse_devices(a.devices) if a.devices else tuple(range(len(gpu_free)))
    except ValueError as error:
        print("MULTIGPU refused: %s" % error)
        return 2
    if len(devices) < 2:
        print("MULTIGPU not applicable: %d GPU(s) named or found; the checks need two or more of one vendor"
              % len(devices))
        return 2
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = jobs_for(names, devices, want_fault=not a.no_fault)
    cores = alloc_cores()
    workers, bounds = plan_workers(len(jobs), cores, alloc_mem_gb(), gpu_free, a.host_gb_per_job, a.gpu_gb_per_job,
                                   asked=a.jobs)
    print("MULTIGPU plan: %d units, %d arms, %d at once (bounds %s) on %d cores, devices %s (%s)"
          % (len(names), len(jobs), workers, json.dumps(bounds, sort_keys=True), cores,
             ",".join(map(str, devices)), vendor or "vendor not detected"), flush=True)
    started = time.time()
    docs = {name: {} for name in names}
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_arm, a.python, name, arm, devs, out, a.timeout): (name, arm, devs)
                   for name, arm, devs in jobs}
        for future in cf.as_completed(futures):
            name, arm, devs = futures[future]
            try:
                docs[name][arm] = future.result()
            except Exception as error:
                docs[name][arm] = dict(unit=name, arm=arm, devices=list(devs), parts=[],
                                       error=one_line("%s: %s" % (type(error).__name__, error)))
    units = {}
    for name in names:
        verdict, reason, digests = unit_verdict(name, docs[name], len(devices), want_fault=not a.no_fault,
                                                strict_spread=a.strict_spread)
        row = unit_row(name)
        arms = docs[name]
        units[name] = dict(verdict=verdict, reason=reason, digests=digests, module=row[1], witness=row[3],
                           fault=(row[4] if not a.no_fault else "not run"),
                           ranks=(arms.get("multi") or {}).get("ranks"),
                           seconds={arm: (arms.get(arm) or {}).get("seconds") for arm in arms},
                           peak_rss_kb={arm: (arms.get(arm) or {}).get("peak_rss_kb") for arm in arms},
                           devices={arm: (arms.get(arm) or {}).get("devices") for arm in arms})
        print(check_line(name, verdict, reason, digests, units[name]["ranks"]))
    files = sorted({str(d.get("mojolearn_file")) for arms in docs.values() for d in arms.values()
                    if d.get("mojolearn_file")})
    peaks = [v for row in units.values() for v in row["peak_rss_kb"].values() if type(v) is int]
    doc = dict(schema=SCHEMA, optional=True, required_by=[], vendor=vendor, devices=list(devices), selected=names,
               workers=workers, bounds=bounds, cores=cores, python=a.python, mojolearn_files=files,
               fault_arms=not a.no_fault, fault_note=FAULT_NOTE, strict_spread=a.strict_spread,
               seconds=round(time.time() - started, 1), peak_rss_kb_max=max(peaks) if peaks else None, units=units)
    doc["ok"] = not summary_errors(doc, names)
    (out / SUMMARY).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    bad = sorted(n for n, r in units.items() if r["verdict"] != "OK")
    print("MULTIGPU summary %d/%d OK%s; %d arms at once on %d cores; largest worker RSS %s kB (%s)"
          % (len(units) - len(bad), len(units), (", not: " + " ".join(bad)) if bad else "", workers, cores,
             doc["peak_rss_kb_max"], out / SUMMARY))
    return 0 if doc["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
