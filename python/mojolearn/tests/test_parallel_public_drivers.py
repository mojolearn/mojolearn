# SPDX-License-Identifier: Apache-2.0
"""The public multi-GPU driver modules, called the way a user calls them.

parallel_classical, parallel_ensemble, parallel_graph, parallel_neighbors,
parallel_neighbors_reference and parallel_preprocessing are imported by
`mojolearn` and list their drivers in `__all__`. Each driver here runs on the
real path (worker processes, the estimator's own binding) over a small
fixture and must return the bytes the plain one-device estimator returns.

Devices come from MOJOLEARN_PAR_DEVICES (comma-separated indices, default
"0"); set it to "0,1" on a two-GPU box for the physical multi-device run.
A case skips, and says why, when the installed package has no route for it
(a CPU-only package refuses the cooperative drivers by name) or its binding
is not installed.

    cd python && python3 -m pytest mojolearn/tests/test_parallel_public_drivers.py
"""
import os
import re
from pathlib import Path

import numpy as np
import pytest

import mojolearn

MODULES = ("parallel_classical", "parallel_ensemble", "parallel_graph", "parallel_neighbors",
           "parallel_neighbors_reference", "parallel_preprocessing")

#: Text that belongs in an engineering comment, never in a user docstring.
_NOT_USER_VOICE = re.compile(r"lane/|\b20\d\d-\d\d-\d\d\b|\.mojo\b|\.py:\d|\bTODO\b|NOT YET|\bIDENTICAL\b")


def _devices():
    raw = os.environ.get("MOJOLEARN_PAR_DEVICES", "0")
    return tuple(int(v) for v in raw.split(",") if v.strip())


def _same(label, actual, expected):
    a, e = np.asarray(actual), np.asarray(expected)
    assert a.shape == e.shape, (label, a.shape, e.shape)
    assert a.tobytes() == e.tobytes(), label


def _run(case):
    try:
        return case(_devices())
    except NotImplementedError as exc:
        if "no CPU implementation" in str(exc):
            pytest.skip(f"this package has no route for the driver: {exc}")
        raise
    except ValueError as exc:
        if "device selection is unavailable" in str(exc):
            pytest.skip(f"device group unavailable here: {exc}")
        raise
    except (ImportError, OSError) as exc:
        pytest.skip(f"binding not installed: {exc}")


def _data(rows=256, cols=8, seed=0):
    rng = np.random.default_rng(seed)
    X = np.ascontiguousarray(rng.standard_normal((rows, cols)).astype(np.float32))
    yc = (X[:, 0] + X[:, 1] > 0).astype(np.int64)
    yr = np.ascontiguousarray((X[:, 0] - 0.5 * X[:, 2]).astype(np.float32))
    return X, yc, yr


# ---------------------------------------------------------------- source checks (no device)

@pytest.mark.parametrize("name", MODULES)
def test_module_is_public_and_its_drivers_are_documented(name):
    module = getattr(mojolearn, name)
    assert name in mojolearn.__all__
    assert module.__all__, name
    assert module.__doc__ and not _NOT_USER_VOICE.search(module.__doc__), name
    for member in module.__all__:
        obj = getattr(module, member)
        assert callable(obj), (name, member)
        doc = obj.__doc__ or ""
        assert doc.strip(), (name, member)
        assert not _NOT_USER_VOICE.search(doc), (name, member)


def test_catboost_auto_learning_rate_is_public():
    from mojolearn import ensemble
    assert "catboost_auto_learning_rate" in ensemble.__all__
    assert not _NOT_USER_VOICE.search(ensemble.catboost_auto_learning_rate.__doc__)
    assert ensemble.catboost_auto_learning_rate("Logloss", 10000, 1000, False, False) == 0.029701
    assert ensemble.catboost_auto_learning_rate("MAE", 1000, 1000, False, True) is None


def test_docs_leave_the_unqualified_drivers_out():
    exclude = Path(__file__).resolve().parents[3] / "docs-site" / "api_exclude.txt"
    if not exclude.is_file():
        pytest.skip("docs-site is not part of an installed package")
    names = {line.split("\t", 1)[0].strip() for line in exclude.read_text().splitlines()
             if line.strip() and not line.startswith("#")}
    assert set(MODULES) <= names
    assert "ensemble.catboost_auto_learning_rate" in names


# ---------------------------------------------------------------- real path (one device by default)

def _scaler(dev):
    from mojolearn.parallel_preprocessing import fit_scaler, transform_scaler
    X, _, _ = _data(64, 10)
    par = fit_scaler(mojolearn.StandardScaler(), X, devices=dev, columns_per_shard=4)
    plain = mojolearn.StandardScaler().fit(X)
    t = transform_scaler(par, X, devices=dev, columns_per_shard=4)
    back = transform_scaler(par, t, devices=dev, columns_per_shard=4, inverse=True)
    return [("fit_scaler mean_", par.mean_, plain.mean_), ("fit_scaler var_", par.var_, plain.var_),
            ("transform_scaler", t, plain.transform(X)),
            ("transform_scaler inverse", back, plain.inverse_transform(plain.transform(X)))]


def _graph(dev):
    from mojolearn.parallel_graph import fit_graph
    X, _, _ = _data(200, 4)
    par = fit_graph(mojolearn.AgglomerativeClustering(n_clusters=3), X, devices=dev)
    plain = mojolearn.AgglomerativeClustering(n_clusters=3).fit(X)
    return [("fit_graph labels_", par.labels_, plain.labels_),
            ("fit_graph children_", par.children_, plain.children_)]


def _umap(dev):
    from mojolearn.parallel_graph import fit_graph, transform_umap
    X, _, _ = _data(288, 8)
    kw = dict(n_neighbors=8, n_components=2, n_epochs=8, random_state=3)
    par = fit_graph(mojolearn.UMAP(**kw), X[:256], devices=dev)
    plain = mojolearn.UMAP(**kw).fit(X[:256])
    return [("fit_graph embedding_", par.embedding_, plain.embedding_),
            ("transform_umap", transform_umap(par, X[256:], devices=dev), plain.transform(X[256:]))]


def _queries(dev):
    from mojolearn.parallel_neighbors import ParallelQueries
    X, yc, _ = _data(320, 8)
    m = mojolearn.KNeighborsClassifier(n_neighbors=5).fit(X[:256], yc[:256])
    q = np.ascontiguousarray(X[256:])
    with ParallelQueries(m, devices=dev, rows_per_shard=16) as pq:
        d, i = pq.query(q, method="kneighbors")
        pp = pq.query(q, method="predict_proba")
    d0, i0 = m.kneighbors(q)
    return [("ParallelQueries distances", d, d0), ("ParallelQueries indices", i, i0),
            ("ParallelQueries predict_proba", pp, m.predict_proba(q))]


def _reference(dev):
    from mojolearn.parallel_neighbors_reference import ReferenceShardedNeighbors
    X, yc, _ = _data(320, 8)
    m = mojolearn.KNeighborsClassifier(n_neighbors=5).fit(X[:256], yc[:256])
    q = np.ascontiguousarray(X[256:])
    with ReferenceShardedNeighbors(m, devices=dev, reference_rows_per_shard=64,
                                   query_rows_per_shard=32) as rs:
        d, i = rs.kneighbors(q)
        pr, pp = rs.predict(q), rs.predict_proba(q)
    d0, i0 = m.kneighbors(q)
    return [("ReferenceShardedNeighbors distances", d, d0), ("ReferenceShardedNeighbors indices", i, i0),
            ("ReferenceShardedNeighbors predict", pr, m.predict(q)),
            ("ReferenceShardedNeighbors predict_proba", pp, m.predict_proba(q))]


def _forest(dev):
    from mojolearn.parallel_ensemble import fit_forest
    X, yc, _ = _data(256, 8)
    kw = dict(n_estimators=8, max_depth=4, random_state=7)
    par = fit_forest(mojolearn.RandomForestClassifier(**kw), X, yc, devices=dev, trees_per_shard=2)
    plain = mojolearn.RandomForestClassifier(**kw).fit(X, yc)
    return [("fit_forest predict_proba", par.predict_proba(X), plain.predict_proba(X))]


def _forest_predictor(dev):
    from mojolearn.parallel_ensemble import ParallelForestPredictor
    X, yc, _ = _data(256, 8)
    m = mojolearn.RandomForestClassifier(n_estimators=8, max_depth=4, random_state=7,
                                         inference_engine="parallel_groves").fit(X, yc)
    with ParallelForestPredictor(m, devices=dev) as pool:
        pred, proba = pool.predict(X), pool.predict_proba(X)
    return [("ParallelForestPredictor predict", pred, m.predict(X)),
            ("ParallelForestPredictor predict_proba", proba, m.predict_proba(X))]


def _boosting(dev):
    from mojolearn.parallel_ensemble import fit_boosting
    X, yc, _ = _data(256, 8)
    kw = dict(n_estimators=5, max_depth=3, loss="Logloss", learning_rate=0.1)
    par = fit_boosting(mojolearn.GradientBoosting(**kw), X, yc, devices=dev)
    plain = mojolearn.GradientBoosting(**kw).fit(X, yc)
    return [("fit_boosting predict_proba", par.predict_proba(X), plain.predict_proba(X))]


def _ordered_rmse(dev):
    from mojolearn.parallel_ensemble import fit_ordered_rmse
    X, _, yr = _data(256, 8)
    perm = np.arange(X.shape[0])[::-1].copy()
    par = fit_ordered_rmse(mojolearn.OrderedRMSE(n_estimators=5, max_depth=3), X, yr,
                           permutation=perm, devices=dev)
    plain = mojolearn.OrderedRMSE(n_estimators=5, max_depth=3).fit(X, yr, permutation=perm)
    return [("fit_ordered_rmse predict", par.predict(X), plain.predict(X))]


def _feature_freq(dev):
    from mojolearn.parallel_ensemble import fit_feature_freq
    X, _, yr = _data(256, 8)
    codes = np.column_stack([(X[:, 4] > 0), (X[:, 5] > 0) * 2 + (X[:, 6] > 0)]).astype(np.float32)
    Xc = np.ascontiguousarray(np.column_stack([codes, X]).astype(np.float32))
    kw = dict(sources=[0, 1], random_state=7)
    par = fit_feature_freq(mojolearn.ExperimentalTwoLevelFeatureFreq(**kw), Xc, yr, devices=dev)
    plain = mojolearn.ExperimentalTwoLevelFeatureFreq(**kw).fit(Xc, yr)
    return [("fit_feature_freq predict", par.predict(Xc), plain.predict(Xc))]


def _isolation_forest(dev):
    from mojolearn.parallel_ensemble import fit_isolation_forest, score_isolation_forest
    X, _, _ = _data(256, 8)
    par = fit_isolation_forest(mojolearn.IsolationForest(n_estimators=8, random_state=5), X, devices=dev)
    plain = mojolearn.IsolationForest(n_estimators=8, random_state=5).fit(X)
    s = score_isolation_forest(par, X, devices=dev, method="score_samples")
    return [("score_isolation_forest", s, plain.score_samples(X))]


def _logistic(dev):
    from mojolearn.parallel_classical import fit_logistic
    X, yc, _ = _data()
    par = fit_logistic(mojolearn.LogisticRegression(max_iter=20), X, yc, devices=dev)
    plain = mojolearn.LogisticRegression(max_iter=20).fit(X, yc)
    return [("fit_logistic coef_", par.coef_, plain.coef_)]


def _exponential_smoothing(dev):
    from mojolearn.parallel_classical import fit_exponential_smoothing
    X, _, _ = _data(96, 2)
    S = np.cumsum(np.abs(X), axis=0).T.astype(np.float32) + np.float32(10.0)
    S = np.ascontiguousarray(S)
    kw = dict(seasonal="additive", seasonal_periods=4, ts_num=2)
    par = fit_exponential_smoothing(mojolearn.ExponentialSmoothing(S, **kw), devices=dev, series_per_shard=1)
    plain = mojolearn.ExponentialSmoothing(S, **kw).fit()
    return [("fit_exponential_smoothing level_", par.level_, plain.level_),
            ("fit_exponential_smoothing forecast", par.forecast(8), plain.forecast(8))]


def _gram(dev):
    from mojolearn.parallel_classical import fit_gram_estimator
    X, _, yr = _data()
    par = fit_gram_estimator(mojolearn.Ridge(alpha=1.0), X, yr, devices=dev)
    plain = mojolearn.Ridge(alpha=1.0).fit(X, yr)
    return [("fit_gram_estimator coef_", par.coef_, plain.coef_)]


def _arima(dev):
    from mojolearn.parallel_classical import fit_arima
    X, _, _ = _data(128, 2)
    series = np.ascontiguousarray(np.cumsum(X, axis=0).T.astype(np.float32))
    par = fit_arima(mojolearn.ARIMA(order=(1, 0, 0)), series, devices=dev, series_per_shard=1)
    plain = mojolearn.ARIMA(order=(1, 0, 0)).fit(series)
    return [("fit_arima ar_", par.ar_, plain.ar_), ("fit_arima forecast", par.forecast(4), plain.forecast(4))]


def _kmeans(dev):
    from mojolearn.parallel_classical import fit_kmeans
    X, _, _ = _data()
    par = fit_kmeans(mojolearn.KMeans(n_clusters=4, random_state=3), X, devices=dev)
    plain = mojolearn.KMeans(n_clusters=4, random_state=3).fit(X)
    return [("fit_kmeans cluster_centers_", par.cluster_centers_, plain.cluster_centers_),
            ("fit_kmeans labels_", par.labels_, plain.labels_)]


def _coordinate_descent(dev):
    from mojolearn.parallel_classical import fit_coordinate_descent
    X, _, yr = _data()
    par = fit_coordinate_descent(mojolearn.Lasso(alpha=0.01, max_iter=50), X, yr, devices=dev)
    plain = mojolearn.Lasso(alpha=0.01, max_iter=50).fit(X, yr)
    return [("fit_coordinate_descent coef_", par.coef_, plain.coef_)]


def _svm(dev):
    from mojolearn.parallel_classical import fit_svm, predict_svm
    X, yc, _ = _data()
    kw = dict(C=1.0, kernel="rbf", max_iter=100)
    par = fit_svm(mojolearn.SVC(**kw), X[:192], yc[:192], devices=dev)
    plain = mojolearn.SVC(**kw).fit(X[:192], yc[:192])
    dec = predict_svm(par, X[192:], devices=dev, method="decision_function")
    return [("predict_svm decision_function", dec, plain.decision_function(X[192:])),
            ("predict_svm predict", predict_svm(par, X[192:], devices=dev), plain.predict(X[192:]))]


def _gaussian_process(dev):
    from mojolearn.parallel_classical import fit_gaussian_process, predict_gaussian_process
    X, _, yr = _data(96, 4)
    k = mojolearn.ConstantKernel(1.0) * mojolearn.RBF(1.0) + mojolearn.WhiteKernel(0.1)
    par = fit_gaussian_process(mojolearn.GaussianProcessRegressor(kernel=k), X[:64], yr[:64], devices=dev)
    plain = mojolearn.GaussianProcessRegressor(kernel=k).fit(X[:64], yr[:64])
    mean, std = predict_gaussian_process(par, X[64:], devices=dev, return_std=True)
    pm, ps = plain.predict(X[64:], return_std=True)
    return [("fit_gaussian_process alpha_", par.alpha_, plain.alpha_),
            ("predict_gaussian_process mean", mean, pm), ("predict_gaussian_process std", std, ps)]


def _dbscan(dev):
    from mojolearn.parallel_classical import fit_dbscan
    X, _, _ = _data(400, 4)
    par = fit_dbscan(mojolearn.DBSCAN(eps=0.9, min_samples=5), X, devices=dev)
    plain = mojolearn.DBSCAN(eps=0.9, min_samples=5).fit(X)
    return [("fit_dbscan labels_", par.labels_, plain.labels_)]


def _gaussian_mixture(dev):
    from mojolearn.parallel_classical import fit_gaussian_mixture, predict_gaussian_mixture
    X, _, _ = _data(400, 4)
    kw = dict(n_components=3, max_iter=10, random_state=3)
    par = fit_gaussian_mixture(mojolearn.GaussianMixture(**kw), X, devices=dev)
    plain = mojolearn.GaussianMixture(**kw).fit(X)
    return [("fit_gaussian_mixture covariances_", par.covariances_, plain.covariances_),
            ("predict_gaussian_mixture", predict_gaussian_mixture(par, X, devices=dev, method="predict"),
             plain.predict(X))]


def _resample(dev):
    from mojolearn import parallel_classical as pc
    _, _, yr = _data(512, 4)
    x = np.ascontiguousarray(yr)
    b = pc.bootstrap(x, devices=dev, statistic="mean", n_resamples=256, random_state=3)
    b0 = mojolearn.resample.bootstrap(x, statistic="mean", n_resamples=256, random_state=3)
    p = pc.permutation_test(x[:256], x[256:], devices=dev, statistic="diff_means", n_resamples=256,
                            random_state=3)
    p0 = mojolearn.resample.permutation_test(x[:256], x[256:], statistic="diff_means", n_resamples=256,
                                             random_state=3)
    r = pc.monte_carlo_integrate("product", [0.0, 0.0], [1.0, 2.0], 4096, devices=dev, random_state=1)
    r0 = mojolearn.resample.monte_carlo_integrate("product", [0.0, 0.0], [1.0, 2.0], 4096, random_state=1)
    return [("bootstrap distribution", b.distribution, b0.distribution),
            ("permutation_test null", p.null_distribution, p0.null_distribution),
            ("monte_carlo_integrate", np.asarray([r.integral, r.mean]), np.asarray([r0.integral, r0.mean]))]


def _hdbscan(dev):
    from mojolearn.parallel_classical import fit_hdbscan
    X, _, _ = _data(400, 4)
    par = fit_hdbscan(mojolearn.HDBSCAN(min_cluster_size=5), X, devices=dev)
    plain = mojolearn.HDBSCAN(min_cluster_size=5).fit(X)
    return [("fit_hdbscan labels_", par.labels_, plain.labels_),
            ("fit_hdbscan core_distances_", par.core_distances_, plain.core_distances_)]


def _cholesky(dev):
    from mojolearn.parallel_classical import fit_cholesky, solve_cholesky
    X, _, yr = _data(96, 4)
    d2 = ((X[:, None, :].astype(np.float64) - X[None, :, :]) ** 2).sum(-1)
    A = 1.0 / (1.0 + d2)
    A[np.arange(96), np.arange(96)] += 1.0
    A = np.ascontiguousarray(A.astype(np.float32))
    B = np.ascontiguousarray(np.stack([yr, yr[::-1]], 1).astype(np.float32))
    par = fit_cholesky(mojolearn.Cholesky(), A, devices=dev)
    plain = mojolearn.Cholesky().fit(A)
    return [("fit_cholesky L_", par.L_, plain.L_),
            ("solve_cholesky", solve_cholesky(par, B, devices=dev), plain.solve(B))]


def _kernel_method(dev):
    from mojolearn.parallel_classical import fit_kernel_method, apply_kernel_method
    X, _, yr = _data(160, 4)
    kw = dict(alpha=0.1, kernel="rbf", gamma=0.5)
    par = fit_kernel_method(mojolearn.KernelRidge(**kw), X[:128], yr[:128], devices=dev)
    plain = mojolearn.KernelRidge(**kw).fit(X[:128], yr[:128])
    return [("fit_kernel_method dual_coef_", par.dual_coef_, plain.dual_coef_),
            ("apply_kernel_method", apply_kernel_method(par, X[128:], devices=dev), plain.predict(X[128:]))]


def _rbf_sampler(dev):
    from mojolearn.parallel_classical import transform_rbf_sampler
    X, _, _ = _data(100, 8)
    m = mojolearn.RBFSampler(gamma=0.5, n_components=64, random_state=1).fit(X)
    return [("transform_rbf_sampler", transform_rbf_sampler(m, X, devices=dev, rows_per_shard=30),
             m.transform(X))]


#: case -> the public names it calls
CASES = {
    _scaler: ("parallel_preprocessing.fit_scaler", "parallel_preprocessing.transform_scaler"),
    _graph: ("parallel_graph.fit_graph",),
    _umap: ("parallel_graph.fit_graph", "parallel_graph.transform_umap"),
    _queries: ("parallel_neighbors.ParallelQueries",),
    _reference: ("parallel_neighbors_reference.ReferenceShardedNeighbors",),
    _forest: ("parallel_ensemble.fit_forest",),
    _forest_predictor: ("parallel_ensemble.ParallelForestPredictor",),
    _boosting: ("parallel_ensemble.fit_boosting",),
    _ordered_rmse: ("parallel_ensemble.fit_ordered_rmse",),
    _feature_freq: ("parallel_ensemble.fit_feature_freq",),
    _isolation_forest: ("parallel_ensemble.fit_isolation_forest", "parallel_ensemble.score_isolation_forest"),
    _logistic: ("parallel_classical.fit_logistic",),
    _exponential_smoothing: ("parallel_classical.fit_exponential_smoothing",),
    _gram: ("parallel_classical.fit_gram_estimator",),
    _arima: ("parallel_classical.fit_arima",),
    _kmeans: ("parallel_classical.fit_kmeans",),
    _coordinate_descent: ("parallel_classical.fit_coordinate_descent",),
    _svm: ("parallel_classical.fit_svm", "parallel_classical.predict_svm"),
    _gaussian_process: ("parallel_classical.fit_gaussian_process", "parallel_classical.predict_gaussian_process"),
    _dbscan: ("parallel_classical.fit_dbscan",),
    _gaussian_mixture: ("parallel_classical.fit_gaussian_mixture", "parallel_classical.predict_gaussian_mixture"),
    _resample: ("parallel_classical.bootstrap", "parallel_classical.permutation_test",
                "parallel_classical.monte_carlo_integrate"),
    _hdbscan: ("parallel_classical.fit_hdbscan",),
    _cholesky: ("parallel_classical.fit_cholesky", "parallel_classical.solve_cholesky"),
    _kernel_method: ("parallel_classical.fit_kernel_method", "parallel_classical.apply_kernel_method"),
    _rbf_sampler: ("parallel_classical.transform_rbf_sampler",),
}


def test_every_public_driver_has_a_real_path_case():
    exported = {f"{m}.{n}" for m in MODULES for n in getattr(mojolearn, m).__all__}
    covered = {name for names in CASES.values() for name in names}
    assert exported == covered, (sorted(exported - covered), sorted(covered - exported))


@pytest.mark.parametrize("case", list(CASES), ids=[c.__name__.lstrip("_") for c in CASES])
def test_driver_equals_the_one_device_estimator(case):
    for label, actual, expected in _run(case):
        _same(label, actual, expected)
