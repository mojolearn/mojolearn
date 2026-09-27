# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The neighbors expansion lane against scikit-learn, at a tolerance, on tiny
data (a sanity check, not an identity claim). Needs scikit-learn and numpy;
runs on the lane's pod: `python -m pytest python/mojolearn/tests/test_x_neighbors_sanity.py`."""
import numpy as np
import pytest

sk = pytest.importorskip("sklearn")
import mojolearn as ml  # noqa: E402


def _data(n=200, d=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    X[:5] *= 6.0
    return X


def test_lof():
    from sklearn.neighbors import LocalOutlierFactor as R
    X = _data()
    ours = ml.LocalOutlierFactor(n_neighbors=12, contamination=0.1)
    lab = np.asarray(ours.fit_predict(X))
    ref = R(n_neighbors=12, contamination=0.1)
    rl = ref.fit_predict(X)
    np.testing.assert_allclose(np.asarray(ours.negative_outlier_factor_), ref.negative_outlier_factor_, rtol=1e-4)
    assert (lab == rl).mean() > 0.98
    Xh = _data(50, seed=1)
    a = ml.LocalOutlierFactor(n_neighbors=12, novelty=True).fit(X)
    b = R(n_neighbors=12, novelty=True).fit(X)
    np.testing.assert_allclose(np.asarray(a.score_samples(Xh)), b.score_samples(Xh), rtol=1e-4)
    np.testing.assert_allclose(np.asarray(a.decision_function(Xh)), b.decision_function(Xh), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("kw", [dict(), dict(metric="manhattan"), dict(shrink_threshold=0.3),
                                dict(priors="empirical")])
def test_nearest_centroid(kw):
    from sklearn.neighbors import NearestCentroid as R
    X = _data(300)
    y = (X[:, 0] + X[:, 1] > 0).astype(int) + (X[:, 2] > 0.5).astype(int)
    a = ml.NearestCentroid(**kw).fit(X, y)
    b = R(**kw).fit(X, y)
    np.testing.assert_allclose(np.asarray(a.centroids_), b.centroids_, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(np.asarray(a.within_class_std_dev_), b.within_class_std_dev_, rtol=1e-4)
    assert (np.asarray(a.predict(X)) == b.predict(X)).mean() > 0.99
    if kw.get("metric", "euclidean") == "euclidean":
        np.testing.assert_allclose(np.asarray(a.decision_function(X)), b.decision_function(X), rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(np.asarray(a.predict_proba(X)), b.predict_proba(X), rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("kernel", ["rbf", "linear", "poly", "sigmoid"])
def test_ocsvm(kernel):
    from sklearn.svm import OneClassSVM as R
    X = _data(150)
    if kernel == "linear":
        X = X + np.float32(2.0)       # centred data makes the linear problem degenerate (decision ~ 0)
    kw = dict(kernel=kernel, nu=0.3, gamma=0.1 if kernel != "rbf" else "scale")
    if kernel == "sigmoid":
        kw["coef0"] = 0.0
    if kernel == "poly":
        kw["tol"] = 1e-5              # the cubic kernel on the scaled rows is ill conditioned: at 1e-3
        #                               both solvers stop KKT-feasible but 12% apart; they meet as tol drops
    a = ml.OneClassSVM(**kw).fit(X)
    b = R(**kw).fit(X)
    da = np.asarray(a.decision_function(X))
    db = b.decision_function(X)
    scale = np.abs(db).max()
    assert np.abs(da - db).max() < 2e-2 * scale, (np.abs(da - db).max(), scale)
    clear = np.abs(db) > 2e-2 * scale  # margin rows (free support vectors) sit at decision ~ 0 on both
    assert (np.asarray(a.predict(X)) == b.predict(X))[clear].mean() > 0.97


@pytest.mark.parametrize("kernel", ["rbf", "linear", "poly", "cosine"])
def test_kernel_pca(kernel):
    from sklearn.decomposition import KernelPCA as R
    X = _data(80)
    a = ml.KernelPCA(n_components=3, kernel=kernel)
    b = R(n_components=3, kernel=kernel, eigen_solver="dense")
    Za = np.asarray(a.fit_transform(X))
    Zb = b.fit_transform(X)
    np.testing.assert_allclose(np.asarray(a.eigenvalues_), b.eigenvalues_, rtol=1e-3)
    np.testing.assert_allclose(np.abs(Za), np.abs(Zb), rtol=2e-3, atol=2e-3)
    Xh = _data(20, seed=3)
    np.testing.assert_allclose(np.abs(np.asarray(a.transform(Xh))), np.abs(b.transform(Xh)), rtol=2e-3, atol=2e-3)


def test_polynomial_count_sketch():
    from sklearn.kernel_approximation import PolynomialCountSketch as R
    X = _data(60)
    a = ml.PolynomialCountSketch(degree=3, gamma=0.5, coef0=1.0, n_components=32, random_state=7).fit(X)
    b = R(degree=3, gamma=0.5, coef0=1.0, n_components=32, random_state=7).fit(X)
    np.testing.assert_array_equal(np.asarray(a.indexHash_), b.indexHash_)
    np.testing.assert_array_equal(np.asarray(a.bitHash_), b.bitHash_)
    np.testing.assert_allclose(np.asarray(a.transform(X)), b.transform(X), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("steps", [1, 2, 3])
def test_additive_chi2(steps):
    from sklearn.kernel_approximation import AdditiveChi2Sampler as R
    X = np.abs(_data(40))
    X[::5, 1] = 0
    np.testing.assert_allclose(np.asarray(ml.AdditiveChi2Sampler(sample_steps=steps).fit_transform(X)),
                               R(sample_steps=steps).fit_transform(X), rtol=1e-5, atol=1e-6)


def test_skewed_chi2():
    from sklearn.kernel_approximation import SkewedChi2Sampler as R
    X = np.abs(_data(40))
    a = ml.SkewedChi2Sampler(skewedness=0.5, n_components=50, random_state=4).fit(X)
    b = R(skewedness=0.5, n_components=50, random_state=4).fit(X)
    np.testing.assert_allclose(np.asarray(a.random_weights_), b.random_weights_, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(np.asarray(a.random_offset_), b.random_offset_, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(a.transform(X)), b.transform(X), rtol=1e-3, atol=2e-4)


@pytest.mark.parametrize("cls,kw", [("LabelPropagation", dict(kernel="rbf", gamma=0.2)),
                                    ("LabelPropagation", dict(kernel="knn", n_neighbors=5)),
                                    ("LabelSpreading", dict(kernel="rbf", gamma=0.2, alpha=0.3)),
                                    ("LabelSpreading", dict(kernel="knn", n_neighbors=5))])
def test_label_propagation(cls, kw):
    import sklearn.semi_supervised as S
    X = _data(120)
    y = (X[:, 0] > 0).astype(int)
    y[::3] = -1
    a = getattr(ml, cls)(**kw).fit(X, y)
    b = getattr(S, cls)(**kw).fit(X, y)
    np.testing.assert_allclose(np.asarray(a.label_distributions_), b.label_distributions_, rtol=2e-3, atol=2e-4)
    assert (np.asarray(a.transduction_) == b.transduction_).mean() > 0.98
    Xh = _data(30, seed=5)
    np.testing.assert_allclose(np.asarray(a.predict_proba(Xh)), b.predict_proba(Xh), rtol=2e-3, atol=2e-4)


@pytest.mark.parametrize("kw", [dict(), dict(weights="distance", n_neighbors=3), dict(add_indicator=True)])
def test_knn_imputer(kw):
    from sklearn.impute import KNNImputer as R
    X = _data(80)
    X[::4, 2] = np.nan
    X[1::5, 0] = np.nan
    X[7, :] = np.nan
    np.testing.assert_allclose(np.asarray(ml.KNNImputer(**kw).fit_transform(X)), R(**kw).fit_transform(X),
                               rtol=1e-4, atol=1e-5)


def _graph(n=60, seed=0, directed=False):
    rng = np.random.default_rng(seed)
    A = (rng.random((n, n)) < 0.08).astype(np.float32) * rng.integers(1, 4, (n, n)).astype(np.float32)
    np.fill_diagonal(A, 0)
    if not directed:
        A = np.triu(A)
        A = A + A.T
    return A


def test_pagerank():
    nx = pytest.importorskip("networkx")
    A = _graph(directed=True)
    A[3] = 0                                            # a dangling node
    ref = nx.pagerank(nx.from_numpy_array(A, create_using=nx.DiGraph), alpha=0.85, tol=1e-6)
    ours = np.asarray(ml.PageRank(alpha=0.85, tol=1e-6).fit(A).pagerank_)
    np.testing.assert_allclose(ours, [ref[i] for i in range(len(A))], rtol=2e-4, atol=1e-6)


def test_connected_components():
    from scipy.sparse.csgraph import connected_components as R
    A = _graph(80)
    A[:, 70:] = 0
    A[70:, :] = 0
    k, lab = ml.connected_components(A, directed=False)
    rk, rl = R(A, directed=False)
    assert k == rk
    np.testing.assert_array_equal(np.asarray(lab), rl)


def test_louvain():
    nx = pytest.importorskip("networkx")
    A = _graph(80)
    G = nx.from_numpy_array(A)
    m = ml.Louvain().fit(A)
    comms = {}
    for i, c in enumerate(np.asarray(m.labels_)):
        comms.setdefault(int(c), set()).add(i)
    q_ours = nx.community.modularity(G, list(comms.values()))
    assert abs(q_ours - m.modularity_) < 1e-4
    q_ref = max(nx.community.modularity(G, nx.community.louvain_communities(G, seed=s)) for s in range(5))
    assert q_ours > q_ref - 0.03, (q_ours, q_ref)


def test_svgp():
    X = _data(150, 3)
    y = np.sin(X[:, 0]) + 0.1 * X[:, 1]
    Z = X[::10]
    kv, ls, nv = 1.5, 1.2, 0.2
    m = ml.SVGP(inducing_points=Z, kernel_variance=kv, lengthscale=ls, noise_variance=nv, jitter=1e-6).fit(X, y)
    X64, Z64, y64 = X.astype(np.float64), Z.astype(np.float64), y.astype(np.float64)

    def k(a, b):
        d = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        return kv * np.exp(-d / (2 * ls * ls))
    Kuu = k(Z64, Z64) + 1e-6 * np.eye(len(Z64))
    Kuf = k(Z64, X64)
    S = Kuu + Kuf @ Kuf.T / nv
    alpha = np.linalg.solve(S, Kuf @ y64) / nv
    Xs = _data(20, 3, seed=9).astype(np.float64)
    Ksu = k(Xs, Z64)
    mean = Ksu @ alpha
    var = kv - np.einsum("ij,jk,ik->i", Ksu, np.linalg.inv(Kuu) - np.linalg.inv(S), Ksu)
    a_mean, a_var = m.predict_f(Xs.astype(np.float32))
    np.testing.assert_allclose(np.asarray(a_mean), mean, rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(np.asarray(a_var), var, rtol=5e-2, atol=5e-3)
    n = len(X64)
    Qff = Kuf.T @ np.linalg.solve(Kuu, Kuf)
    C = Qff + nv * np.eye(n)
    _, logdet = np.linalg.slogdet(C)
    elbo = (-0.5 * n * np.log(2 * np.pi) - 0.5 * logdet - 0.5 * y64 @ np.linalg.solve(C, y64)
            - 0.5 / nv * (n * kv - np.trace(Qff)))
    assert abs(m.elbo_ - elbo) < 1e-3 * abs(elbo) + 0.05, (m.elbo_, elbo)


# ---------------------------------------------------------------- option parity (pass 2)
@pytest.mark.parametrize("kw", [dict(), dict(shrink_threshold=0.3)])
def test_nearest_centroid_options(kw):
    from sklearn.neighbors import NearestCentroid as R
    sp = pytest.importorskip("scipy.sparse")
    X = _data(300)
    y = (X[:, 0] + X[:, 1] > 0).astype(int) + (X[:, 2] > 0.5).astype(int)
    a = ml.NearestCentroid(**kw).fit(X, y)
    b = R(**kw).fit(X, y)
    np.testing.assert_allclose(np.asarray(a.deviations_), b.deviations_, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(np.asarray(a.predict_log_proba(X)), b.predict_log_proba(X), rtol=1e-3, atol=1e-3)
    assert abs(a.score(X, y) - b.score(X, y)) < 0.01
    c = ml.NearestCentroid(**kw).fit(sp.csr_matrix(X), y)
    assert np.array_equal(np.asarray(c.centroids_), np.asarray(a.centroids_))


def test_ocsvm_sample_weight_and_precomputed():
    from sklearn.svm import OneClassSVM as R
    X = _data(150)
    w = (1 + np.arange(150) % 3).astype(np.float64) * 0.5
    w[7] = 0.0
    a = ml.OneClassSVM(nu=0.3).fit(X, sample_weight=w)
    b = R(nu=0.3).fit(X, sample_weight=w)
    Xh = _data(40, seed=3)
    np.testing.assert_allclose(np.asarray(a.decision_function(Xh)), b.decision_function(Xh), rtol=2e-3, atol=2e-3)
    # scikit-learn's support_ indexes the rows left after libsvm drops the
    # zero-weight samples; ours indexes the caller's rows (the docstring), so
    # map ours down before comparing, and compare the vectors themselves.
    kept = np.flatnonzero(w > 0)
    ours = np.asarray(a.support_)
    assert np.array_equal(np.searchsorted(kept, ours), b.support_)
    assert np.array_equal(np.asarray(a.support_vectors_), b.support_vectors_.astype(np.float32))
    K = np.asarray(ml.OneClassSVM()._kernel(X, X, "rbf", 0.1, 0.0, 0))
    Kh = np.asarray(ml.OneClassSVM()._kernel(Xh, X, "rbf", 0.1, 0.0, 0))
    p = ml.OneClassSVM(kernel="precomputed", nu=0.3).fit(K)
    q = ml.OneClassSVM(kernel="rbf", gamma=0.1, nu=0.3).fit(X)
    assert np.array_equal(np.asarray(p.decision_function(Kh)), np.asarray(q.decision_function(Xh)))


def test_kernel_pca_options():
    from sklearn.decomposition import KernelPCA as R
    X = _data(80)
    a = ml.KernelPCA(n_components=3, kernel="rbf", gamma=0.2, iterated_power=5).fit(X)
    K = np.asarray(a._kernel(X, X, "rbf", a._gamma, 1.0, 3))
    p = ml.KernelPCA(n_components=3, kernel="precomputed").fit(K)
    assert np.array_equal(np.asarray(p.eigenvalues_), np.asarray(a.eigenvalues_))
    assert list(a.get_feature_names_out()) == list(R(n_components=3).fit(X).get_feature_names_out())


def test_samplers_options():
    from sklearn.kernel_approximation import AdditiveChi2Sampler, PolynomialCountSketch, SkewedChi2Sampler
    sp = pytest.importorskip("scipy.sparse")
    X = np.abs(_data(60))
    X[X < 0.5] = 0
    a = ml.AdditiveChi2Sampler().fit(X)
    b = AdditiveChi2Sampler().fit(X)
    S = a.transform(sp.csr_matrix(X))
    R = b.transform(sp.csr_matrix(X))
    assert sp.issparse(S)
    np.testing.assert_allclose(S.toarray(), R.toarray(), rtol=1e-4, atol=1e-5)
    assert np.array_equal(S.toarray(), np.asarray(a.transform(X)))
    assert list(a.get_feature_names_out()) == list(b.get_feature_names_out())
    rs1, rs2 = np.random.RandomState(4), np.random.RandomState(4)
    p = ml.PolynomialCountSketch(n_components=16, random_state=rs1).fit(X)
    q = PolynomialCountSketch(n_components=16, random_state=rs2).fit(X)
    assert np.array_equal(np.asarray(p.indexHash_), q.indexHash_)
    assert np.array_equal(np.asarray(p.bitHash_), q.bitHash_)
    assert list(p.get_feature_names_out()) == list(q.get_feature_names_out())
    s1 = ml.SkewedChi2Sampler(n_components=12, random_state=np.random.RandomState(9)).fit(X)
    s2 = SkewedChi2Sampler(n_components=12, random_state=np.random.RandomState(9)).fit(X)
    np.testing.assert_allclose(np.asarray(s1.random_weights_), s2.random_weights_, rtol=1e-5, atol=1e-6)
    assert list(s1.get_feature_names_out()) == list(s2.get_feature_names_out())
    assert ml.SkewedChi2Sampler(random_state=None).fit(X).random_weights_.shape == (6, 100)


def test_label_propagation_score():
    from sklearn.semi_supervised import LabelSpreading as R
    X = _data(120)
    y = (X[:, 0] > 0).astype(int)
    yl = y.copy()
    yl[::3] = -1
    a = ml.LabelSpreading(kernel="knn", n_neighbors=7).fit(X, yl)
    b = R(kernel="knn", n_neighbors=7).fit(X, yl)
    assert abs(a.score(X, y) - b.score(X, y)) < 0.02


def test_knn_imputer_options():
    from sklearn.impute import KNNImputer as R
    X = _data(80)
    X[::7, 1] = -1.0
    X[::5, 3] = -1.0
    a = ml.KNNImputer(missing_values=-1.0, add_indicator=True).fit(X)
    b = R(missing_values=-1.0, add_indicator=True).fit(X)
    np.testing.assert_allclose(np.asarray(a.transform(X)), b.transform(X), rtol=1e-4, atol=1e-5)
    assert list(a.get_feature_names_out()) == list(b.get_feature_names_out())


def test_pagerank_options():
    nx = pytest.importorskip("networkx")
    rng = np.random.default_rng(5)
    A = (rng.random((40, 40)) < 0.1).astype(np.float32) * (1 + rng.integers(0, 3, (40, 40))).astype(np.float32)
    np.fill_diagonal(A, 0)
    A[3] = 0
    A[11] = 0
    G = nx.from_numpy_array(A.astype(np.float64), create_using=nx.DiGraph)
    dang = {i: float(1 + i % 4) for i in range(40)}
    start = {i: float(1 + (i * 3) % 7) for i in range(40)}
    ref = nx.pagerank(G, alpha=0.85, dangling=dang, nstart=start, tol=1e-6)
    ours = ml.PageRank(alpha=0.85, dangling=[dang[i] for i in range(40)],
                       nstart=[start[i] for i in range(40)]).fit(A)
    np.testing.assert_allclose(np.asarray(ours.pagerank_), [ref[i] for i in range(40)], rtol=1e-3, atol=1e-5)
    base = ml.PageRank(alpha=0.85).fit(A)
    same = ml.PageRank(alpha=0.85, dangling=np.ones(40)).fit(A)
    assert np.array_equal(np.asarray(base.pagerank_), np.asarray(same.pagerank_))
