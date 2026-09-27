# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""SVC probability=True: libsvm's Platt scaling over a seeded 5-fold CV
(lane x-neighbors-svc-probability, 2026-09-27).

The host arithmetic (the shuffle, sigmoid_train, the pairwise coupling) is
checked with no binding. The fitted checks run through whichever SVM binding
this install loads and are skipped, saying so, without one; they compare with
scikit-learn to a tolerance (libsvm's fold shuffle is C rand(), so the folds
and hence the bits differ by design).
"""

import os
import tempfile

import numpy as np
import pytest

from mojolearn import _svm_impl
from mojolearn._svm_impl import SVC


def _data(n=180, k=3, seed=5):
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=2.5, size=(k, 5))
    y = np.arange(n) % k
    x = (centers[y] + rng.normal(size=(n, 5))).astype(np.float32)
    return x, y.astype(np.int64)


def _fit_or_skip(x, y, **kw):
    try:
        return SVC(**kw).fit(x, y)
    except (ImportError, NotImplementedError) as exc:
        pytest.skip(f"no SVM binding that fits on this install: {exc}")


def test_shuffle_is_a_seeded_permutation():
    a = _svm_impl._splitmix_perm(50, 3)
    assert sorted(a) == list(range(50))
    assert a == _svm_impl._splitmix_perm(50, 3)
    assert a != _svm_impl._splitmix_perm(50, 4)


def test_sigmoid_train_separates_and_orients():
    dec = [-2.0, -1.5, -1.0, -0.2, 0.3, 1.0, 1.4, 2.2]
    labels = [-1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0]
    a, b = _svm_impl._sigmoid_train(dec, labels)
    assert a < 0.0                     # P(+1) rises with the decision value
    assert _svm_impl._sigmoid_predict(2.0, a, b) > 0.5 > _svm_impl._sigmoid_predict(-2.0, a, b)


def test_coupling_is_a_distribution():
    r = [[0.0, 0.7, 0.8], [0.3, 0.0, 0.6], [0.2, 0.4, 0.0]]
    p = _svm_impl._multiclass_probability(3, r)
    assert abs(sum(p) - 1.0) < 1e-9 and p[0] > p[1] > p[2]


def test_random_state_needs_probability():
    with pytest.raises(NotImplementedError, match="random_state"):
        SVC(random_state=1)
    with pytest.raises(ValueError, match="random_state"):
        SVC(probability=True, random_state=-1)


def test_predict_proba_needs_probability():
    x, y = _data(n=60, k=2)
    m = _fit_or_skip(x, y, kernel="rbf", gamma=0.1)
    with pytest.raises(AttributeError, match="probability"):
        m.predict_proba(x)


@pytest.mark.parametrize("k", [2, 3])
def test_probabilities_agree_with_sklearn_and_repeat(k):
    sk = pytest.importorskip("sklearn.svm")
    x, y = _data(k=k)
    m = _fit_or_skip(x, y, kernel="rbf", gamma=0.1, probability=True, random_state=3)
    p = np.asarray(m.predict_proba(x))
    assert p.shape == (len(x), k)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6)
    assert np.allclose(np.exp(np.asarray(m.predict_log_proba(x))), p, atol=1e-12)
    ref = sk.SVC(kernel="rbf", gamma=0.1, probability=True, random_state=3).fit(x, y).predict_proba(x)
    assert np.mean(np.argmax(p, axis=1) == np.argmax(ref, axis=1)) > 0.95
    assert np.max(np.abs(p - ref)) < 0.25
    again = SVC(kernel="rbf", gamma=0.1, probability=True, random_state=3).fit(x, y)
    assert np.asarray(again.predict_proba(x)).tobytes() == p.tobytes()
    assert len(np.asarray(m.probA_)) == k * (k - 1) // 2


def test_save_load_keeps_probability_bits():
    x, y = _data()
    m = _fit_or_skip(x, y, kernel="rbf", gamma=0.1, probability=True)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "svc")
        m.save(path)
        back = SVC.load(path)
    assert np.asarray(back.predict_proba(x)).tobytes() == np.asarray(m.predict_proba(x)).tobytes()
