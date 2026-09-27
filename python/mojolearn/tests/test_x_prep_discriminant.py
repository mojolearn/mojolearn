# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Prep lane sanity: LDA and QDA against scikit-learn at a float32
tolerance (binary and three-class; transform up to component sign)."""
import numpy as np
try:
    import pytest
    pytest.importorskip("sklearn")
except ImportError:
    pass
import sklearn.discriminant_analysis as skda
import mojolearn as ml


def _data(seed=0, n=500, d=5, k=3):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n)
    A = rng.standard_normal((d, d)) * 0.5 + np.eye(d)
    X = (rng.standard_normal((n, d)) @ A + y[:, None] * np.linspace(0.5, 1.5, d)).astype(np.float32)
    return X, y


def test_lda():
    for k in (2, 3):
        X, y = _data(k=k)
        Xh, _ = _data(1, k=k)
        m = ml.LinearDiscriminantAnalysis().fit(X, y)
        r = skda.LinearDiscriminantAnalysis().fit(X.astype(np.float64), y)
        np.testing.assert_allclose(np.asarray(m.coef_), r.coef_, rtol=2e-3, atol=2e-3)
        np.testing.assert_allclose(np.asarray(m.intercept_), r.intercept_, rtol=2e-3, atol=2e-3)
        np.testing.assert_allclose(np.asarray(m.predict_proba(Xh)), r.predict_proba(Xh.astype(np.float64)),
                                   atol=2e-3)
        np.testing.assert_allclose(np.asarray(m.explained_variance_ratio_), r.explained_variance_ratio_, atol=1e-3)
        a, b = np.asarray(m.transform(Xh)), r.transform(Xh.astype(np.float64))
        for j in range(b.shape[1]):
            s = np.sign(np.dot(a[:, j], b[:, j]))
            np.testing.assert_allclose(s * a[:, j], b[:, j], rtol=2e-3, atol=2e-3)
        assert np.mean(np.asarray(m.predict(Xh)) == r.predict(Xh.astype(np.float64))) > 0.99
        if k == 2:
            np.testing.assert_allclose(np.asarray(m.decision_function(Xh)),
                                       r.decision_function(Xh.astype(np.float64)), rtol=2e-3, atol=2e-3)


def test_qda():
    for k, reg in ((2, 0.0), (3, 0.1)):
        X, y = _data(2, k=k)
        Xh, _ = _data(3, k=k)
        m = ml.QuadraticDiscriminantAnalysis(reg_param=reg).fit(X, y)
        r = skda.QuadraticDiscriminantAnalysis(reg_param=reg).fit(X.astype(np.float64), y)
        for a, b in zip(m.scalings_, r.scalings_):
            np.testing.assert_allclose(np.asarray(a), b, rtol=2e-3, atol=1e-4)
        np.testing.assert_allclose(np.asarray(m.predict_proba(Xh)), r.predict_proba(Xh.astype(np.float64)),
                                   atol=2e-3)
        assert np.mean(np.asarray(m.predict(Xh)) == r.predict(Xh.astype(np.float64))) > 0.99
        if k == 2:
            np.testing.assert_allclose(np.asarray(m.decision_function(Xh)),
                                       r.decision_function(Xh.astype(np.float64)), rtol=2e-3, atol=2e-3)


if __name__ == "__main__":
    test_lda()
    test_qda()
    print("PASS test_x_prep_discriminant")


def test_priors():
    X, y = _data(4, k=3)
    Xh, _ = _data(5, k=3)
    import sklearn.naive_bayes as sknb
    pairs = ((ml.GaussianNB(priors=[0.2, 0.5, 0.3]), sknb.GaussianNB(priors=[0.2, 0.5, 0.3]), X),
             (ml.MultinomialNB(class_prior=[0.1, 0.6, 0.3]), sknb.MultinomialNB(class_prior=[0.1, 0.6, 0.3]), np.abs(X)),
             (ml.LinearDiscriminantAnalysis(priors=[1.0, 2.0, 1.0]), skda.LinearDiscriminantAnalysis(priors=[1.0, 2.0, 1.0]), X),
             (ml.QuadraticDiscriminantAnalysis(priors=[0.25, 0.25, 0.5]), skda.QuadraticDiscriminantAnalysis(priors=[0.25, 0.25, 0.5]), X))
    for m, r, Xa in pairs:
        m.fit(Xa, y)
        r.fit(Xa.astype(np.float64), y)
        Xt = np.abs(Xh) if Xa is not X else Xh
        a, b = np.asarray(m.predict_proba(Xt)), r.predict_proba(Xt.astype(np.float64))
        # float32 against float64: a point far out on a thin class covariance
        # can move; hold 99% of the probabilities to the tolerance
        assert np.mean(np.abs(a - b) <= 3e-3) > 0.99, (type(m).__name__, np.mean(np.abs(a - b) <= 3e-3))


if __name__ == "__main__":
    test_priors()
    print("PASS test_x_prep_discriminant (priors)")


def test_solvers_and_shrinkage():
    for k in (2, 3):
        X, y = _data(6, k=k)
        X[:, 4] = X[:, 3] * 0.7 + X[:, 4] * 0.3         # a correlated pair: shrinkage matters
        Xh, _ = _data(7, k=k)
        for kw in (dict(solver="lsqr"), dict(solver="lsqr", shrinkage="auto"), dict(solver="lsqr", shrinkage=0.3),
                   dict(solver="eigen"), dict(solver="eigen", shrinkage="auto"), dict(solver="eigen", shrinkage=0.6),
                   dict(solver="svd", store_covariance=True)):
            m = ml.LinearDiscriminantAnalysis(**kw).fit(X, y)
            r = skda.LinearDiscriminantAnalysis(**kw).fit(X.astype(np.float64), y)
            np.testing.assert_allclose(np.asarray(m.covariance_), r.covariance_, rtol=2e-3, atol=2e-4)
            np.testing.assert_allclose(np.asarray(m.coef_), r.coef_, rtol=5e-3, atol=5e-3)
            np.testing.assert_allclose(np.asarray(m.intercept_), r.intercept_, rtol=5e-3, atol=5e-3)
            np.testing.assert_allclose(np.asarray(m.predict_proba(Xh)), r.predict_proba(Xh.astype(np.float64)),
                                       atol=3e-3)
            if kw["solver"] == "eigen" and np.abs(r.explained_variance_ratio_).max() <= 1:
                # (with shrinkage Sb = St - Sw is indefinite; a ratio above 1 means sum(evals)
                # cancels to near 0 and float32 cannot follow the reference's float64 there)
                np.testing.assert_allclose(np.asarray(m.explained_variance_ratio_), r.explained_variance_ratio_,
                                           atol=1e-3)
                a, b = np.asarray(m.transform(Xh)), r.transform(Xh.astype(np.float64))
                for j in range(b.shape[1]):
                    s = np.sign(np.dot(a[:, j], b[:, j]))
                    np.testing.assert_allclose(s * a[:, j], b[:, j], rtol=5e-3, atol=5e-3)
    X, y = _data(8, k=3)
    Xh, _ = _data(9, k=3)
    for kw in (dict(solver="eigen"), dict(solver="eigen", shrinkage="auto", store_covariance=True),
               dict(solver="eigen", shrinkage=0.2), dict(solver="svd", reg_param=0.1, store_covariance=True)):
        m = ml.QuadraticDiscriminantAnalysis(**kw).fit(X, y)
        r = skda.QuadraticDiscriminantAnalysis(**kw).fit(X.astype(np.float64), y)
        for a, b in zip(m.scalings_, r.scalings_):
            np.testing.assert_allclose(np.asarray(a), b, rtol=2e-3, atol=1e-4)
        if kw.get("store_covariance"):
            for a, b in zip(m.covariance_, r.covariance_):
                np.testing.assert_allclose(np.asarray(a), b, rtol=2e-3, atol=2e-4)
        np.testing.assert_allclose(np.asarray(m.predict_proba(Xh)), r.predict_proba(Xh.astype(np.float64)),
                                   atol=3e-3)


if __name__ == "__main__":
    test_solvers_and_shrinkage()
    print("PASS test_x_prep_discriminant (solvers)")
