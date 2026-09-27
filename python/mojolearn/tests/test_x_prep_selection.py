# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Prep lane sanity: feature selection against scikit-learn."""
import numpy as np
try:
    import pytest
    pytest.importorskip("sklearn")
except ImportError:
    pass
import sklearn.feature_selection as skfs
import mojolearn as ml


def _data(seed=0, n=300, d=6):
    rng = np.random.default_rng(seed)
    X = (rng.standard_normal((n, d)) * np.linspace(0.3, 2.0, d)).astype(np.float32)
    X[:, 2] = np.float32(4.0)
    return X


def test_variance_threshold():
    X = _data()
    for th in (0.0, 0.5):
        m, r = ml.VarianceThreshold(th).fit(X), skfs.VarianceThreshold(th).fit(X.astype(np.float64))
        np.testing.assert_allclose(np.asarray(m.variances_), r.variances_, rtol=1e-4, atol=1e-6)
        assert list(m.get_support()) == list(r.get_support())
        np.testing.assert_array_equal(np.asarray(m.transform(X)), r.transform(X))


if __name__ == "__main__":
    test_variance_threshold()
    print("PASS test_x_prep_selection")


def test_scores_and_kbest():
    rng = np.random.default_rng(3)
    n, d = 400, 7
    y = rng.integers(0, 3, n)
    X = (rng.standard_normal((n, d)) + y[:, None] * np.linspace(0, 0.6, d)).astype(np.float32)
    yr = (X[:, 3] * 0.4 + rng.standard_normal(n)).astype(np.float32)
    for mine, ref, Xa, ya in ((ml.f_classif, skfs.f_classif, X, y), (ml.chi2, skfs.chi2, np.abs(X), y),
                              (ml.f_regression, skfs.f_regression, X, yr)):
        s, p = mine(Xa, ya)
        rs, rp = ref(Xa.astype(np.float64), ya)
        np.testing.assert_allclose(np.asarray(s), rs, rtol=2e-3, atol=1e-4)
        ok = rp > 1e-6
        np.testing.assert_allclose(np.asarray(p)[ok], rp[ok], rtol=2e-2, atol=1e-6)
    m = ml.SelectKBest(k=3).fit(X, y)
    r = skfs.SelectKBest(k=3).fit(X, y)
    assert list(m.get_support()) == list(r.get_support())


if __name__ == "__main__":
    test_scores_and_kbest()
    print("PASS test_x_prep_selection (kbest)")


def test_mutual_info():
    rng = np.random.default_rng(5)
    n, d = 500, 4
    y = rng.integers(0, 3, n)
    X = (rng.standard_normal((n, d)) + y[:, None] * np.array([0.0, 0.3, 1.0, 2.0])).astype(np.float32)
    yr = (X[:, 2] + 0.5 * rng.standard_normal(n)).astype(np.float32)
    a = np.asarray(ml.mutual_info_classif(X, y, random_state=0))
    b = skfs.mutual_info_classif(X, y, random_state=0)
    np.testing.assert_allclose(a, b, atol=3e-3)
    a = np.asarray(ml.mutual_info_regression(X, yr, random_state=0))
    b = skfs.mutual_info_regression(X, yr, random_state=0)
    np.testing.assert_allclose(a, b, atol=3e-3)


def test_mutual_info_discrete():
    rng = np.random.default_rng(6)
    n = 500
    y = rng.integers(0, 3, n)
    Xd = np.stack([rng.integers(0, 4, n), (y + rng.integers(0, 2, n)) % 4, y * 2.5,
                   np.full(n, 7)], axis=1).astype(np.float32)
    Xc = (rng.standard_normal((n, 2)) + y[:, None] * np.array([0.2, 1.0])).astype(np.float32)
    X = np.concatenate([Xd[:, :2], Xc[:, :1], Xd[:, 2:], Xc[:, 1:]], axis=1)
    yr = (X[:, 1] + 0.7 * rng.standard_normal(n)).astype(np.float32)
    Xd4 = X[:, [0, 1, 3, 4]]
    for A, df in ((Xd4, True), (X, [0, 1, 3, 4]), (X, np.array([True, True, False, True, True, False])),
                  (X, [-6, 1, 3, -2]), (X, False), (X, True)):
        a = np.asarray(ml.mutual_info_classif(A, y, discrete_features=df, random_state=0))
        b = skfs.mutual_info_classif(A, y, discrete_features=df, random_state=0)
        np.testing.assert_allclose(a, b, atol=3e-3, err_msg=f"classif {df}")
        if A is X and df is True:     # continuous columns as categories: one sample per value
            for f in (ml.mutual_info_regression, skfs.mutual_info_regression):
                try:
                    f(A, yr, discrete_features=df, random_state=0)
                    raise AssertionError("no ValueError")
                except ValueError:
                    pass
            continue
        a = np.asarray(ml.mutual_info_regression(A, yr, discrete_features=df, random_state=0))
        b = skfs.mutual_info_regression(A, yr, discrete_features=df, random_state=0)
        np.testing.assert_allclose(a, b, atol=3e-3, err_msg=f"regression {df}")


if __name__ == "__main__":
    test_mutual_info()
    test_mutual_info_discrete()
    print("PASS test_x_prep_selection (mutual info)")


def test_rfe():
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as SkLDA
    rng = np.random.default_rng(7)
    n, d = 400, 8
    y = rng.integers(0, 3, n)
    X = (rng.standard_normal((n, d)) + y[:, None] * np.linspace(0, 1.4, d)).astype(np.float32)
    m = ml.RFE(ml.LinearDiscriminantAnalysis(), n_features_to_select=3, step=2).fit(X, y)
    r = skfs.RFE(SkLDA(), n_features_to_select=3, step=2).fit(X.astype(np.float64), y)
    np.testing.assert_array_equal(np.asarray(m.ranking_), r.ranking_)
    assert np.mean(np.asarray(m.predict(X)) == r.predict(X.astype(np.float64))) > 0.99


if __name__ == "__main__":
    test_rfe()
    print("PASS test_x_prep_selection (rfe)")


def test_score_edges_and_rfe_getter():
    """The reference's NaN / inf score edges, f_regression / r_regression
    force_finite, RFE importance_getter as a str and a callable."""
    import warnings
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as SkLDA
    rng = np.random.default_rng(11)
    n = 300
    y = rng.integers(0, 3, n)
    X = rng.standard_normal((n, 5)).astype(np.float32)
    X[:, 1] = 2.5                                   # constant: NaN
    X[:, 2] = y.astype(np.float32)                  # constant within classes: +inf
    Xc = np.abs(X)
    Xc[:, 4] = 0                                    # all-zero: chi2 NaN
    yr = (X[:, 0] * 0.5 + rng.standard_normal(n)).astype(np.float32)
    Xr = X.copy()
    Xr[:, 3] = yr * 2                               # |r| = 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cases = [(ml.f_classif(X, y), skfs.f_classif(X.astype(np.float64), y)),
                 (ml.chi2(Xc, y), skfs.chi2(Xc.astype(np.float64), y))]
        # the constant column 1: the reference divides its float noise by a zero norm
        # (+-inf or NaN by rounding); mojolearn gives the mathematical value, r 0 / F 0 /
        # p 1 with force_finite and NaN without it
        keep = np.array([0, 2, 3, 4])
        for ff in (True, False):
            fs, fp = ml.f_regression(Xr, yr, force_finite=ff)
            mr = np.asarray(ml.r_regression(X, yr, force_finite=ff))
            edge = (0.0, 1.0, 0.0) if ff else (np.nan, np.nan, np.nan)
            np.testing.assert_array_equal([np.asarray(fs)[1], np.asarray(fp)[1], mr[1]], edge)
            rs, rp = skfs.f_regression(Xr.astype(np.float64), yr, force_finite=ff)
            cases.append(((np.asarray(fs)[keep], np.asarray(fp)[keep]), (rs[keep], rp[keep])))
            rr = skfs.r_regression(X.astype(np.float64), yr, force_finite=ff)
            np.testing.assert_allclose(mr[keep], rr[keep], rtol=2e-3, atol=1e-4)
    for (s, p), (rs, rp) in cases:
        s, p = np.asarray(s, dtype=np.float64), np.asarray(p, dtype=np.float64)
        huge = rs > 1e10                            # |r| = 1 / separable: float64 may stop short of inf
        assert np.all(s[huge] > 1e10) and np.all(p[huge] < 1e-6), (s, rs)
        for mine, ref in ((s[~huge], rs[~huge]), (p[~huge], rp[~huge])):
            assert np.array_equal(np.isnan(mine), np.isnan(ref)), (mine, ref)
            fin = np.isfinite(ref)
            np.testing.assert_allclose(mine[fin], ref[fin], rtol=2e-2, atol=1e-5)
    m = ml.SelectKBest(k=2).fit(X, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = skfs.SelectKBest(k=2).fit(X.astype(np.float64), y)
    assert list(m.get_support()) == list(r.get_support())
    Xg = (rng.standard_normal((n, 8)) + y[:, None] * np.linspace(0, 1.4, 8)).astype(np.float32)
    for g in ("coef_", lambda e: e.coef_[0]):
        m = ml.RFE(ml.LinearDiscriminantAnalysis(), n_features_to_select=3, step=2, importance_getter=g).fit(Xg, y)
        r = skfs.RFE(SkLDA(), n_features_to_select=3, step=2, importance_getter=g).fit(Xg.astype(np.float64), y)
        np.testing.assert_array_equal(np.asarray(m.ranking_), r.ranking_)


if __name__ == "__main__":
    test_score_edges_and_rfe_getter()
    print("PASS test_x_prep_selection (score edges, rfe getter)")
