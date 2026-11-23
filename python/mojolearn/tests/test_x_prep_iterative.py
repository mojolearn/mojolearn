# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Prep lane sanity: IterativeImputer (BayesianRidge) against scikit-learn
on correlated data with 15% missing."""
import numpy as np
try:
    import pytest
    pytest.importorskip("sklearn")
except ImportError:
    pass
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer as SkII
import mojolearn as ml


def _data(seed=0, n=400, d=5):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, 2))
    X = (Z @ rng.standard_normal((2, d)) + 0.1 * rng.standard_normal((n, d))).astype(np.float32)
    X[rng.random((n, d)) < 0.15] = np.nan
    return X


def test_iterative():
    X = _data()
    Xh = _data(1)
    for kw in (dict(), dict(max_iter=3, imputation_order="descending", initial_strategy="median"),
               dict(min_value=-1.0, max_value=1.0)):
        m, r = ml.IterativeImputer(**kw), SkII(**kw)
        a, b = np.asarray(m.fit_transform(X)), r.fit_transform(X.astype(np.float64))
        np.testing.assert_allclose(a, b, rtol=2e-3, atol=2e-3)
        assert m.n_iter_ == r.n_iter_, (m.n_iter_, r.n_iter_)
        np.testing.assert_allclose(np.asarray(m.transform(Xh)), r.transform(Xh.astype(np.float64)),
                                   rtol=2e-3, atol=2e-3)


def test_iterative_options():
    from sklearn.linear_model import Ridge as SkRidge
    X = _data(2)
    Xh = _data(3)
    miss = np.isnan(X)
    # add_indicator, and another (mojolearn) estimator: the rounds over its fit / predict
    for kw, skw in ((dict(add_indicator=True), dict(add_indicator=True)),
                    (dict(estimator=ml.Ridge(alpha=2.0)), dict(estimator=SkRidge(alpha=2.0))),
                    (dict(estimator=ml.Ridge(alpha=2.0), add_indicator=True, max_iter=4),
                     dict(estimator=SkRidge(alpha=2.0), add_indicator=True, max_iter=4))):
        m, r = ml.IterativeImputer(**kw), SkII(**skw)
        a, b = np.asarray(m.fit_transform(X)), r.fit_transform(X.astype(np.float64))
        np.testing.assert_allclose(a, b, rtol=2e-3, atol=2e-3, err_msg=str(kw))
        assert m.n_iter_ == r.n_iter_, (kw, m.n_iter_, r.n_iter_)
        np.testing.assert_allclose(np.asarray(m.transform(Xh)), r.transform(Xh.astype(np.float64)),
                                   rtol=2e-3, atol=2e-3, err_msg=str(kw))
    # the random order and the drawn neighbours: a different draw than the reference's, the
    # same fixed point to within the imputation's own spread; the same bits for one seed
    for kw in (dict(imputation_order="random", random_state=3), dict(n_nearest_features=3, random_state=3),
               dict(n_nearest_features=2, imputation_order="random", random_state=5, max_iter=20)):
        a = np.asarray(ml.IterativeImputer(**kw).fit_transform(X))
        assert np.array_equal(a, np.asarray(ml.IterativeImputer(**kw).fit_transform(X))), kw
        b = SkII(**kw).fit_transform(X.astype(np.float64))
        b2 = SkII(**dict(kw, random_state=kw["random_state"] + 1)).fit_transform(X.astype(np.float64))
        spread = np.abs(b - b2)[miss].mean()          # the reference against itself, another seed
        assert np.abs(a - b)[miss].mean() < 2 * spread + 0.02, (kw, np.abs(a - b)[miss].mean(), spread)
    # sample_posterior: draws from the truncated predictive normal; over seeds, the mean and the
    # spread of the draws match the reference's
    for kw in (dict(sample_posterior=True), dict(sample_posterior=True, min_value=-0.5, max_value=0.5)):
        A = np.array([np.asarray(ml.IterativeImputer(random_state=s, **kw).fit_transform(X))[miss]
                      for s in range(12)])
        B = np.array([SkII(random_state=s, **kw).fit_transform(X.astype(np.float64))[miss] for s in range(12)])
        if "min_value" in kw:
            assert A.min() >= -0.5 and A.max() <= 0.5
        assert abs(A.mean() - B.mean()) < 0.05, (kw, A.mean(), B.mean())
        assert abs(A.std(0).mean() / B.std(0).mean() - 1) < 0.2, (kw, A.std(0).mean(), B.std(0).mean())
    try:
        ml.IterativeImputer(estimator=ml.Ridge(), sample_posterior=True).fit(X)
        raise AssertionError("no ValueError")
    except ValueError:
        pass
    # a user estimator must be a mojolearn estimator
    try:
        ml.IterativeImputer(estimator=SkRidge()).fit(X)
        raise AssertionError("no TypeError")
    except TypeError as error:
        assert "requires a mojolearn estimator; got Ridge" in str(error)


def _labels(seed=0, n=400, d=4):
    # every column the row's label plus a column shift, at most one NaN a row (integer-valued targets)
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 3, n)
    X = np.stack([labels + (j % 2) for j in range(d)], axis=1).astype(np.float32)
    rows = rng.random(n) < 0.6
    cols = rng.integers(0, d, n)
    X[rows, cols[rows]] = np.nan
    return X


def test_iterative_degenerate_shapes():
    # scikit-learn's edge cases (lane mf-iterimp3): one kept feature returns the initial imputation with
    # n_iter_ = 0 and no step; a kept all-NaN column (keep_empty_features) is not missing and keeps its fill
    X = _data(4)
    X[:, 1:] = np.nan
    for est in (None, ml.Ridge(alpha=1.0)):
        m, r = ml.IterativeImputer(estimator=est, add_indicator=True), SkII(add_indicator=True)
        np.testing.assert_allclose(np.asarray(m.fit_transform(X)), r.fit_transform(X.astype(np.float64)),
                                   rtol=1e-5, atol=1e-5)
        assert m.n_iter_ == 0 == r.n_iter_ and len(m.imputation_sequence_) == 0
    X = _data(5)
    X[:, 2] = np.nan
    from sklearn.linear_model import Ridge as SkRidge
    for est, sk in ((None, None), (ml.Ridge(alpha=1.0), SkRidge(alpha=1.0))):
        m = ml.IterativeImputer(estimator=est, keep_empty_features=True)
        r = SkII(estimator=sk, keep_empty_features=True)
        a, b = np.asarray(m.fit_transform(X)), r.fit_transform(X.astype(np.float64))
        assert np.array_equal(a[:, 2], b[:, 2]), (est, a[:, 2][:4], b[:, 2][:4])
        np.testing.assert_allclose(a, b, rtol=2e-3, atol=2e-3, err_msg=str(est))
        assert m.n_iter_ == r.n_iter_, (est, m.n_iter_, r.n_iter_)
        assert m.n_features_with_missing_ == r.n_features_with_missing_


def test_iterative_classifier():
    # a classifier estimator is fitted on the float target as scikit-learn fits it (lane mf-iterimp3)
    from sklearn.naive_bayes import GaussianNB as SkGNB
    from sklearn.tree import DecisionTreeClassifier as SkTree
    X, Xh = _labels(0), _labels(1)
    for est, sk in ((ml.GaussianNB(), SkGNB()),
                    (ml.DecisionTreeClassifier(random_state=0), SkTree(random_state=0))):
        m, r = ml.IterativeImputer(estimator=est), SkII(estimator=sk)
        a, b = np.asarray(m.fit_transform(X)), r.fit_transform(X.astype(np.float64))
        assert np.array_equal(a, b), (type(est).__name__, np.argwhere(a != b)[:4])
        assert m.n_iter_ == r.n_iter_
        assert np.array_equal(np.asarray(m.transform(Xh)), r.transform(Xh.astype(np.float64)))
    try:
        ml.IterativeImputer(estimator=ml.GaussianNB()).fit(_data(0))
        raise AssertionError("no ValueError")
    except ValueError as error:
        assert "Unknown label type: continuous" in str(error), error


if __name__ == "__main__":
    test_iterative()
    test_iterative_options()
    test_iterative_degenerate_shapes()
    test_iterative_classifier()
    print("PASS test_x_prep_iterative")
