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
    # add_indicator, and another estimator (the rounds in Python over its fit / predict)
    for kw, skw in ((dict(add_indicator=True), dict(add_indicator=True)),
                    (dict(estimator=SkRidge(alpha=2.0)), dict(estimator=SkRidge(alpha=2.0))),
                    (dict(estimator=SkRidge(alpha=2.0), add_indicator=True, max_iter=4),
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
        ml.IterativeImputer(estimator=SkRidge(), sample_posterior=True).fit(X)
        raise AssertionError("no ValueError")
    except ValueError:
        pass


if __name__ == "__main__":
    test_iterative()
    test_iterative_options()
    print("PASS test_x_prep_iterative")
