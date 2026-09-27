# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Prep lane sanity: the naive Bayes classifiers against scikit-learn at a
float32 tolerance (three classes, string labels, a constant column)."""
import numpy as np
try:
    import pytest
    pytest.importorskip("sklearn")
except ImportError:
    pass
import sklearn.naive_bayes as sknb
import mojolearn as ml


def _data(seed=0, n=600, d=6):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    X = (rng.standard_normal((n, d)) + y[:, None] * np.linspace(0.2, 1.0, d)).astype(np.float32)
    X[:, 5] = np.float32(1.0)
    labels = np.array(["a", "b", "c"])[y]
    return X, labels


def _check(m, r, X, Xh, attrs, rtol=2e-4, atol=2e-5):
    for a in attrs:
        np.testing.assert_allclose(np.asarray(getattr(m, a)), getattr(r, a), rtol=rtol, atol=atol, err_msg=a)
    np.testing.assert_allclose(np.asarray(m.predict_proba(Xh)), r.predict_proba(Xh), rtol=1e-3, atol=1e-5)
    np.testing.assert_allclose(np.asarray(m.predict_log_proba(Xh)), r.predict_log_proba(Xh), rtol=1e-3, atol=1e-4)
    assert np.mean(np.asarray(m.predict(Xh)) == r.predict(Xh)) > 0.99


def test_gaussian():
    X, y = _data()
    Xh, _ = _data(1)
    _check(ml.GaussianNB().fit(X, y), sknb.GaussianNB().fit(X, y), X, Xh, ("theta_", "var_", "class_prior_"))


def test_multinomial():
    X, y = _data()
    X, Xh = np.abs(X), np.abs(_data(1)[0])
    for kw in (dict(alpha=1.0), dict(alpha=0.3, fit_prior=False)):
        _check(ml.MultinomialNB(**kw).fit(X, y), sknb.MultinomialNB(**kw).fit(X, y), X, Xh,
               ("feature_count_", "feature_log_prob_", "class_log_prior_"))


def test_bernoulli():
    X, y = _data()
    Xh, _ = _data(1)
    for kw in (dict(), dict(binarize=0.5, alpha=2.0)):
        _check(ml.BernoulliNB(**kw).fit(X, y), sknb.BernoulliNB(**kw).fit(X, y), X, Xh,
               ("feature_count_", "feature_log_prob_", "class_log_prior_"))


if __name__ == "__main__":
    test_gaussian()
    test_multinomial()
    test_bernoulli()
    print("PASS test_x_prep_naive_bayes")


def test_complement():
    X, y = _data()
    X, Xh = np.abs(X), np.abs(_data(1)[0])
    for kw in (dict(), dict(alpha=0.4, norm=True)):
        _check(ml.ComplementNB(**kw).fit(X, y), sknb.ComplementNB(**kw).fit(X, y), X, Xh,
               ("feature_count_", "feature_log_prob_", "class_log_prior_"))


if __name__ == "__main__":
    test_complement()
    print("PASS test_x_prep_naive_bayes (complement)")


def test_categorical():
    rng = np.random.default_rng(4)
    n, d = 500, 4
    y = rng.integers(0, 3, n)
    X = np.clip(rng.integers(0, 4, (n, d)) + (y[:, None] > 0) * rng.integers(0, 2, (n, d)), 0, 4).astype(np.float32)
    Xh = np.clip(rng.integers(0, 5, (100, d)), 0, 4).astype(np.float32)
    Xh = np.minimum(Xh, X.max(axis=0))
    m, r = ml.CategoricalNB(alpha=0.5).fit(X, y), sknb.CategoricalNB(alpha=0.5).fit(X, y)
    for a, b in zip(m.feature_log_prob_, r.feature_log_prob_):
        np.testing.assert_allclose(np.asarray(a), b, rtol=2e-4, atol=2e-5)
    np.testing.assert_allclose(np.asarray(m.predict_proba(Xh)), r.predict_proba(Xh), rtol=1e-3, atol=1e-5)
    assert np.mean(np.asarray(m.predict(Xh)) == r.predict(Xh)) > 0.99


if __name__ == "__main__":
    test_categorical()
    print("PASS test_x_prep_naive_bayes (categorical)")


def test_sample_weight_and_min_categories():
    X, y = _data(5)
    Xh, _ = _data(6)
    w = np.random.default_rng(7).uniform(0.1, 3.0, len(y)).astype(np.float32)
    _check(ml.GaussianNB().fit(X, y, sample_weight=w), sknb.GaussianNB().fit(X, y, sample_weight=w), X, Xh,
           ("theta_", "var_", "class_prior_", "class_count_"), rtol=1e-3, atol=1e-4)
    Xa, Xha = np.abs(X), np.abs(Xh)
    for nm in ("MultinomialNB", "ComplementNB", "BernoulliNB"):
        m = getattr(ml, nm)().fit(Xa, y, sample_weight=w)
        r = getattr(sknb, nm)().fit(Xa, y, sample_weight=w)
        _check(m, r, Xa, Xha, ("feature_count_", "class_count_", "feature_log_prob_"), rtol=1e-3, atol=1e-3)
    rng = np.random.default_rng(8)
    Xc = rng.integers(0, 4, (500, 3)).astype(np.float32)
    yc = rng.integers(0, 3, 500)
    wc = rng.uniform(0.1, 2.0, 500)
    for kw in (dict(), dict(min_categories=6), dict(min_categories=[2, 7, 5])):
        m = ml.CategoricalNB(**kw).fit(Xc, yc, sample_weight=wc)
        r = sknb.CategoricalNB(**kw).fit(Xc, yc, sample_weight=wc)
        np.testing.assert_array_equal(np.asarray(m.n_categories_), r.n_categories_)
        for a, b in zip(m.feature_log_prob_, r.feature_log_prob_):
            np.testing.assert_allclose(np.asarray(a), b, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(np.asarray(m.predict_proba(Xc[:50])), r.predict_proba(Xc[:50]), rtol=1e-3, atol=1e-5)


if __name__ == "__main__":
    test_sample_weight_and_min_categories()
    print("PASS test_x_prep_naive_bayes (sample_weight)")
