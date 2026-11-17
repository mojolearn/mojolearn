# SPDX-License-Identifier: Apache-2.0
"""Census fixtures: the multi-device drivers (parallel_classical, parallel_ensemble, parallel_forecasting,
parallel_gaussian_process, parallel_graph, parallel_model_selection, parallel_preprocessing, parallel_neighbors,
parallel_neighbors_reference, parallel_ivf).

Testing tools, never product (tools/ops/python_frames_census.py, fixture units). The shapes follow
python/mojolearn/tests/test_parallel_public_drivers.py, cut to a few dozen rows. Devices are (0, 1): two workers.

Two routes exist on a CPU-only wheel (bindings/parallel_public_native_door.mojo, bindings/parallel_plan_door.mojo):
  * shard / forecast / fold-process routes (fit_forest, transform_rbf_sampler, the forecasting drivers, the GP
    classifier, the scalers, ParallelQueries, ReferenceShardedNeighbors, DistributedIVFIndex, cross_val_score) run
    one host worker process per index, so (0, 1) is a real two-worker pool here;
  * the cooperative drivers (family_par_run, the Cholesky and kernel-method `direct` doors) shard inside one GPU
    binding and refuse on a CPU-only install with "no CPU implementation of the cooperative multi-GPU driver". Their
    fixture still makes the public call (the Python boundary up to the native refusal is measured; the census
    classes the refusal as cpu_refusal), one member per fixture so a refusal loses nothing else; NEEDS_GPU lists
    them for the GPU box's smoke, which runs the same fixtures to completion.
Loading this module imports only the standard library.
"""

DEV = (0, 1)
_PC = "mojolearn.parallel_classical."
_PE = "mojolearn.parallel_ensemble."
_PF = "mojolearn.parallel_forecasting."
_COOP = "cooperative multi-GPU driver: shards inside one GPU binding; a CPU-only install refuses it (measured as cpu_refusal on the CPU runner)"


def _data(rows=48, cols=4, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    X = np.ascontiguousarray(rng.standard_normal((rows, cols)).astype(np.float32))
    yc = (X[:, 0] + X[:, 1] > 0).astype(np.int64)
    yr = np.ascontiguousarray((X[:, 0] - 0.5 * X[:, min(2, cols - 1)]).astype(np.float32))
    return X, yc, yr


# ---------------------------------------------------------------- parallel_classical: cooperative (GPU) drivers

def fit_logistic(ml, cx):
    from mojolearn.parallel_classical import fit_logistic as f
    X, yc, _ = _data()
    f(ml.LogisticRegression(max_iter=5), X, yc, devices=DEV)


def fit_gram_estimator(ml, cx):
    from mojolearn.parallel_classical import fit_gram_estimator as f
    X, _, yr = _data()
    f(ml.Ridge(alpha=1.0), X, yr, devices=DEV)


def fit_kmeans(ml, cx):
    from mojolearn.parallel_classical import fit_kmeans as f
    X, _, _ = _data()
    f(ml.KMeans(n_clusters=3, random_state=3), X, devices=DEV)


def fit_coordinate_descent(ml, cx):
    from mojolearn.parallel_classical import fit_coordinate_descent as f
    X, _, yr = _data()
    f(ml.Lasso(alpha=0.01, max_iter=20), X, yr, devices=DEV)


def fit_svm(ml, cx):
    from mojolearn.parallel_classical import fit_svm as f
    X, yc, _ = _data()
    f(ml.SVC(C=1.0, kernel="rbf", max_iter=50), X, yc, devices=DEV)


def predict_svm(ml, cx):
    from mojolearn.parallel_classical import predict_svm as f
    X, yc, _ = _data()
    m = ml.SVC(C=1.0, kernel="rbf", max_iter=50).fit(X[:32], yc[:32])
    f(m, X[32:], devices=DEV, method="predict")


def _gp_kernel(ml):
    return ml.ConstantKernel(1.0) * ml.RBF(1.0) + ml.WhiteKernel(0.1)


def fit_gaussian_process(ml, cx):
    from mojolearn.parallel_classical import fit_gaussian_process as f
    X, _, yr = _data(32)
    f(ml.GaussianProcessRegressor(kernel=_gp_kernel(ml)), X, yr, devices=DEV)


def predict_gaussian_process(ml, cx):
    from mojolearn.parallel_classical import predict_gaussian_process as f
    X, _, yr = _data(40)
    m = ml.GaussianProcessRegressor(kernel=_gp_kernel(ml)).fit(X[:32], yr[:32])
    f(m, X[32:], devices=DEV, return_std=True)


def fit_dbscan(ml, cx):
    from mojolearn.parallel_classical import fit_dbscan as f
    X, _, _ = _data()
    f(ml.DBSCAN(eps=0.9, min_samples=3), X, devices=DEV)


def fit_gaussian_mixture(ml, cx):
    from mojolearn.parallel_classical import fit_gaussian_mixture as f
    X, _, _ = _data()
    f(ml.GaussianMixture(n_components=2, max_iter=5, random_state=3), X, devices=DEV)


def predict_gaussian_mixture(ml, cx):
    from mojolearn.parallel_classical import predict_gaussian_mixture as f
    X, _, _ = _data()
    m = ml.GaussianMixture(n_components=2, max_iter=5, random_state=3).fit(X)
    f(m, X, devices=DEV, method="predict")


def fit_hdbscan(ml, cx):
    from mojolearn.parallel_classical import fit_hdbscan as f
    X, _, _ = _data()
    f(ml.HDBSCAN(min_cluster_size=5), X, devices=DEV)


def bootstrap(ml, cx):
    from mojolearn.parallel_classical import bootstrap as f
    _, _, yr = _data(64)
    f(yr, devices=DEV, statistic="mean", n_resamples=32, random_state=3)


def permutation_test(ml, cx):
    from mojolearn.parallel_classical import permutation_test as f
    _, _, yr = _data(64)
    f(yr[:32], yr[32:], devices=DEV, statistic="diff_means", n_resamples=32, random_state=3)


def monte_carlo_integrate(ml, cx):
    from mojolearn.parallel_classical import monte_carlo_integrate as f
    f("product", [0.0, 0.0], [1.0, 2.0], 256, devices=DEV, random_state=1)


def _spd(n=24):
    import numpy as np
    X, _, yr = _data(n)
    d2 = ((X[:, None, :].astype(np.float64) - X[None, :, :]) ** 2).sum(-1)
    A = 1.0 / (1.0 + d2)
    A[np.arange(n), np.arange(n)] += 1.0
    B = np.ascontiguousarray(np.stack([yr, yr[::-1]], 1).astype(np.float32))
    return np.ascontiguousarray(A.astype(np.float32)), B


def fit_cholesky(ml, cx):
    from mojolearn.parallel_classical import fit_cholesky as f
    A, _ = _spd()
    f(ml.Cholesky(), A, devices=DEV)


def solve_cholesky(ml, cx):
    from mojolearn.parallel_classical import solve_cholesky as f
    A, B = _spd()
    f(ml.Cholesky().fit(A), B, devices=DEV)


def fit_kernel_method(ml, cx):
    from mojolearn.parallel_classical import fit_kernel_method as f
    X, _, yr = _data()
    f(ml.KernelRidge(alpha=0.1, kernel="rbf", gamma=0.5), X, yr, devices=DEV)


def apply_kernel_method(ml, cx):
    from mojolearn.parallel_classical import apply_kernel_method as f
    X, _, yr = _data()
    m = ml.KernelRidge(alpha=0.1, kernel="rbf", gamma=0.5).fit(X[:32], yr[:32])
    f(m, X[32:], devices=DEV)


# ---------------------------------------------------------------- parallel_classical: host-sharded drivers

def _hw_series(rows=32, series=2):
    import numpy as np
    X, _, _ = _data(rows, series)
    return np.ascontiguousarray(np.cumsum(np.abs(X), axis=0).T.astype(np.float32) + np.float32(10.0))


def _arima_series(rows=48, series=2):
    import numpy as np
    X, _, _ = _data(rows, series)
    return np.ascontiguousarray(np.cumsum(X, axis=0).T.astype(np.float32))


def fit_exponential_smoothing(ml, cx):
    from mojolearn.parallel_classical import fit_exponential_smoothing as f
    S = _hw_series()
    f(ml.ExponentialSmoothing(S, seasonal="additive", seasonal_periods=4, ts_num=2), devices=DEV,
      series_per_shard=1)


def fit_arima(ml, cx):
    from mojolearn.parallel_classical import fit_arima as f
    f(ml.ARIMA(order=(1, 0, 0)), _arima_series(), devices=DEV, series_per_shard=1)


def transform_rbf_sampler(ml, cx):
    from mojolearn.parallel_classical import transform_rbf_sampler as f
    X, _, _ = _data(40, 4)
    m = ml.RBFSampler(gamma=0.5, n_components=16, random_state=1).fit(X)
    f(m, X, devices=DEV, rows_per_shard=16)


# ---------------------------------------------------------------- parallel_forecasting (host workers)

def forecasting_exponential_smoothing(ml, cx):
    from mojolearn.parallel_forecasting import forecast_exponential_smoothing, predict_exponential_smoothing
    m = ml.ExponentialSmoothing(_hw_series(32, 3), seasonal="additive", seasonal_periods=4, ts_num=3).fit()
    predict_exponential_smoothing(m, m.n, m.n + 4, devices=DEV, series_per_shard=2)
    forecast_exponential_smoothing(m, 4, devices=DEV, series_per_shard=2)


def forecasting_arima(ml, cx):
    from mojolearn.parallel_forecasting import forecast_arima, predict_arima
    m = ml.ARIMA(order=(1, 0, 0)).fit(_arima_series(48, 3))
    predict_arima(m, 48, 52, devices=DEV, series_per_shard=2)
    forecast_arima(m, 4, devices=DEV, series_per_shard=2)


# ---------------------------------------------------------------- parallel_ensemble

def fit_forest(ml, cx):
    from mojolearn.parallel_ensemble import fit_forest as f
    X, yc, _ = _data(64, 4)
    f(ml.RandomForestClassifier(n_estimators=4, max_depth=3, random_state=7), X, yc, devices=DEV,
      trees_per_shard=2)


def fit_boosting(ml, cx):
    from mojolearn.parallel_ensemble import fit_boosting as f
    X, yc, _ = _data(64, 4)
    f(ml.GradientBoosting(n_estimators=3, max_depth=2, loss="Logloss", learning_rate=0.1), X, yc, devices=DEV)


def fit_ordered_rmse(ml, cx):
    import numpy as np
    from mojolearn.parallel_ensemble import fit_ordered_rmse as f
    X, _, yr = _data(64, 4)
    perm = np.arange(X.shape[0])[::-1].copy()
    f(ml.OrderedRMSE(n_estimators=3, max_depth=2), X, yr, permutation=perm, devices=DEV)


def fit_feature_freq(ml, cx):
    import numpy as np
    from mojolearn.parallel_ensemble import fit_feature_freq as f
    X, _, yr = _data(64, 6)
    codes = np.column_stack([(X[:, 3] > 0), (X[:, 4] > 0) * 2 + (X[:, 5] > 0)]).astype(np.float32)
    Xc = np.ascontiguousarray(np.column_stack([codes, X]).astype(np.float32))
    f(ml.ExperimentalTwoLevelFeatureFreq(sources=[0, 1], random_state=7), Xc, yr, devices=DEV)


def fit_isolation_forest(ml, cx):
    from mojolearn.parallel_ensemble import fit_isolation_forest as f
    X, _, _ = _data(64, 4)
    f(ml.IsolationForest(n_estimators=4, random_state=5), X, devices=DEV)


def score_isolation_forest(ml, cx):
    from mojolearn.parallel_ensemble import score_isolation_forest as f
    X, _, _ = _data(64, 4)
    m = ml.IsolationForest(n_estimators=4, random_state=5).fit(X)
    f(m, X, devices=DEV, method="score_samples")


def _groves_forest(ml):
    X, yc, _ = _data(64, 4)
    m = ml.RandomForestClassifier(n_estimators=4, max_depth=3, random_state=7,
                                  inference_engine="parallel_groves").fit(X, yc)
    return m, X


def forest_predictor_predict(ml, cx):
    from mojolearn.parallel_ensemble import ParallelForestPredictor
    m, X = _groves_forest(ml)
    pool = ParallelForestPredictor(m, devices=DEV)
    try:
        pool.predict(X)
    finally:
        pool.close()


def forest_predictor_predict_proba(ml, cx):
    from mojolearn.parallel_ensemble import ParallelForestPredictor
    m, X = _groves_forest(ml)
    pool = ParallelForestPredictor(m, devices=DEV)
    try:
        pool.predict_proba(X)
    finally:
        pool.close()


# ---------------------------------------------------------------- parallel_gaussian_process (host workers)

def gaussian_process_classifier(ml, cx):
    from mojolearn.parallel_gaussian_process import (fit_gaussian_process_classifier,
                                                     predict_gaussian_process_classifier)
    X, yc, _ = _data(40, 4)
    k = ml.ConstantKernel(1.0) * ml.RBF(1.0)
    fitted = fit_gaussian_process_classifier(ml.GaussianProcessClassifier(kernel=k), X[:32], yc[:32], devices=DEV)
    predict_gaussian_process_classifier(fitted, X[32:], devices=DEV, method="predict_proba")


# ---------------------------------------------------------------- parallel_graph (cooperative)

def fit_graph(ml, cx):
    from mojolearn.parallel_graph import fit_graph as f
    X, _, _ = _data(48, 4)
    f(ml.AgglomerativeClustering(n_clusters=3), X, devices=DEV)


def transform_umap(ml, cx):
    from mojolearn.parallel_graph import transform_umap as f
    X, _, _ = _data(48, 4)
    m = ml.UMAP(n_neighbors=8, n_components=2, n_epochs=4, random_state=3).fit(X[:40])
    f(m, X[40:], devices=DEV)


# ---------------------------------------------------------------- parallel_model_selection (fold processes)

def cross_val_score(ml, cx):
    from mojolearn.parallel_model_selection import cross_val_score as f
    X, _, yr = _data(48, 4)
    f(ml.LinearRegression(), X, yr, devices=DEV, cv=2)


# ---------------------------------------------------------------- parallel_preprocessing (host workers)

def scaler(ml, cx):
    from mojolearn.parallel_preprocessing import fit_scaler, transform_scaler
    X, _, _ = _data(32, 6)
    par = fit_scaler(ml.StandardScaler(), X, devices=DEV, columns_per_shard=3)
    t = transform_scaler(par, X, devices=DEV, columns_per_shard=3)
    transform_scaler(par, t, devices=DEV, columns_per_shard=3, inverse=True)


# ---------------------------------------------------------------- parallel_neighbors, parallel_neighbors_reference

def parallel_queries(ml, cx):
    from mojolearn.parallel_neighbors import ParallelQueries
    X, yc, _ = _data(64, 4)
    m = ml.KNeighborsClassifier(n_neighbors=3).fit(X[:48], yc[:48])
    pq = ParallelQueries(m, devices=DEV, rows_per_shard=8)
    try:
        pq.query(X[48:], method="kneighbors")
    finally:
        pq.close()


def reference_sharded(ml, cx):
    from mojolearn.parallel_neighbors_reference import ReferenceShardedNeighbors
    X, yc, _ = _data(64, 4)
    m = ml.KNeighborsClassifier(n_neighbors=3).fit(X[:48], yc[:48])
    rs = ReferenceShardedNeighbors(m, devices=DEV, reference_rows_per_shard=24, query_rows_per_shard=8)
    try:
        rs.kneighbors(X[48:])
        rs.predict(X[48:])
    finally:
        rs.close()


# ---------------------------------------------------------------- parallel_ivf

def distributed_ivf(ml, cx):
    from mojolearn.parallel_ivf import DistributedIVFIndex
    X, _, _ = _data(96, 8)
    try:   # the constructor always refuses ("use DistributedIVFIndex.from_index"): the refusal is its behaviour
        DistributedIVFIndex()
    except TypeError:
        pass
    m = ml.IVFIndex(4, 4, 5, kmeans_n_iters=3, metric="euclidean", numeric_mode="identical").fit(X[:88])
    shards = DistributedIVFIndex.from_index(m, devices=DEV)
    try:
        shards.search(X[88:])
    finally:
        shards.close()


FIXTURES = {
    "fit_logistic": (fit_logistic, [_PC + "fit_logistic"]),
    "fit_gram_estimator": (fit_gram_estimator, [_PC + "fit_gram_estimator"]),
    "fit_kmeans": (fit_kmeans, [_PC + "fit_kmeans"]),
    "fit_coordinate_descent": (fit_coordinate_descent, [_PC + "fit_coordinate_descent"]),
    "fit_svm": (fit_svm, [_PC + "fit_svm"]),
    "predict_svm": (predict_svm, [_PC + "predict_svm"]),
    "fit_gaussian_process": (fit_gaussian_process, [_PC + "fit_gaussian_process"]),
    "predict_gaussian_process": (predict_gaussian_process, [_PC + "predict_gaussian_process"]),
    "fit_dbscan": (fit_dbscan, [_PC + "fit_dbscan"]),
    "fit_gaussian_mixture": (fit_gaussian_mixture, [_PC + "fit_gaussian_mixture"]),
    "predict_gaussian_mixture": (predict_gaussian_mixture, [_PC + "predict_gaussian_mixture"]),
    "fit_hdbscan": (fit_hdbscan, [_PC + "fit_hdbscan"]),
    "bootstrap": (bootstrap, [_PC + "bootstrap"]),
    "permutation_test": (permutation_test, [_PC + "permutation_test"]),
    "monte_carlo_integrate": (monte_carlo_integrate, [_PC + "monte_carlo_integrate"]),
    "fit_cholesky": (fit_cholesky, [_PC + "fit_cholesky"]),
    "solve_cholesky": (solve_cholesky, [_PC + "solve_cholesky"]),
    "fit_kernel_method": (fit_kernel_method, [_PC + "fit_kernel_method"]),
    "apply_kernel_method": (apply_kernel_method, [_PC + "apply_kernel_method"]),
    "fit_exponential_smoothing": (fit_exponential_smoothing, [_PC + "fit_exponential_smoothing"]),
    "fit_arima": (fit_arima, [_PC + "fit_arima"]),
    "transform_rbf_sampler": (transform_rbf_sampler, [_PC + "transform_rbf_sampler"]),
    "forecasting_exponential_smoothing": (forecasting_exponential_smoothing,
                                          [_PF + "predict_exponential_smoothing",
                                           _PF + "forecast_exponential_smoothing"]),
    "forecasting_arima": (forecasting_arima, [_PF + "predict_arima", _PF + "forecast_arima"]),
    "fit_forest": (fit_forest, [_PE + "fit_forest"]),
    "fit_boosting": (fit_boosting, [_PE + "fit_boosting"]),
    "fit_ordered_rmse": (fit_ordered_rmse, [_PE + "fit_ordered_rmse"]),
    "fit_feature_freq": (fit_feature_freq, [_PE + "fit_feature_freq"]),
    "fit_isolation_forest": (fit_isolation_forest, [_PE + "fit_isolation_forest"]),
    "score_isolation_forest": (score_isolation_forest, [_PE + "score_isolation_forest"]),
    "forest_predictor_predict": (forest_predictor_predict, [_PE + "ParallelForestPredictor.predict"]),
    "forest_predictor_predict_proba": (forest_predictor_predict_proba, [_PE + "ParallelForestPredictor.predict_proba"]),
    "gaussian_process_classifier": (gaussian_process_classifier,
                                    ["mojolearn.parallel_gaussian_process.fit_gaussian_process_classifier",
                                     "mojolearn.parallel_gaussian_process.predict_gaussian_process_classifier"]),
    "fit_graph": (fit_graph, ["mojolearn.parallel_graph.fit_graph"]),
    "transform_umap": (transform_umap, ["mojolearn.parallel_graph.transform_umap"]),
    "cross_val_score": (cross_val_score, ["mojolearn.parallel_model_selection.cross_val_score"]),
    "scaler": (scaler, ["mojolearn.parallel_preprocessing.fit_scaler",
                        "mojolearn.parallel_preprocessing.transform_scaler"]),
    "parallel_queries": (parallel_queries, ["mojolearn.parallel_neighbors.ParallelQueries.__init__",
                                            "mojolearn.parallel_neighbors.ParallelQueries.query",
                                            "mojolearn.parallel_neighbors.ParallelQueries.close"]),
    "reference_sharded": (reference_sharded,
                          ["mojolearn.parallel_neighbors_reference.ReferenceShardedNeighbors.kneighbors",
                           "mojolearn.parallel_neighbors_reference.ReferenceShardedNeighbors.predict"]),
    "distributed_ivf": (distributed_ivf, ["mojolearn.DistributedIVFIndex.__init__",
                                          "mojolearn.DistributedIVFIndex.from_index",
                                          "mojolearn.DistributedIVFIndex.search",
                                          "mojolearn.DistributedIVFIndex.close"]),
}

#: Called on the CPU runner (the boundary is measured up to the native refusal, a cpu_refusal); their work runs only
#: on a GPU box, whose smoke runs the same fixtures to completion.
NEEDS_GPU = {m: _COOP for m in (
    [_PC + n for n in ("fit_logistic", "fit_gram_estimator", "fit_kmeans", "fit_coordinate_descent", "fit_svm",
                       "predict_svm", "fit_gaussian_process", "predict_gaussian_process", "fit_dbscan",
                       "fit_gaussian_mixture", "predict_gaussian_mixture", "fit_hdbscan", "bootstrap",
                       "permutation_test", "monte_carlo_integrate", "fit_cholesky", "solve_cholesky",
                       "fit_kernel_method", "apply_kernel_method")]
    + [_PE + n for n in ("fit_boosting", "fit_ordered_rmse", "fit_feature_freq", "fit_isolation_forest",
                         "score_isolation_forest", "ParallelForestPredictor.predict",
                         "ParallelForestPredictor.predict_proba")]
    + ["mojolearn.parallel_graph.fit_graph", "mojolearn.parallel_graph.transform_umap"])}

WHY_NOT = {}
