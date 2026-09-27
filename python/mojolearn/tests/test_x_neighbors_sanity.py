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
