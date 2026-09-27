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
