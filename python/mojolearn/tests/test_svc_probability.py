# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""SVC probability=True: libsvm's Platt scaling over a seeded 5-fold CV
(lane x-neighbors-svc-probability, 2026-09-27).

Shuffle, calibration and pairwise coupling numerical contracts run through
actual native kernels in checks/native_tests/svc_probability_{cpu,gpu}_suite.mojo.
The fitted checks run through whichever SVM binding
this install loads and are skipped, saying so, without one; they compare with
scikit-learn to a tolerance (libsvm's fold shuffle is C rand(), so the folds
and hence the bits differ by design).
"""

import os
import tempfile

import numpy as np
import pytest

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


def test_random_state_needs_probability():
    # random_state without probability is accepted and unused, as in
    # scikit-learn (04bb03dff); a bad value is still refused
    assert SVC(random_state=1).get_params()["random_state"] == 1
    with pytest.raises(ValueError, match="random_state"):
        SVC(probability=True, random_state=-1).fit([[0.0], [1.0]], [0, 1])


def test_predict_proba_needs_probability():
    x, y = _data(n=60, k=2)
    m = _fit_or_skip(x, y, kernel="rbf", gamma=0.1)
    # scikit-learn's wording: "This 'SVC' has no attribute 'predict_proba'".
    # The native attribute owner (bindings/public_attributes_native.mojo,
    # availability rule svm_probability) raises it directly; the chained
    # Python cause of scikit-learn's available_if is not reproduced natively.
    with pytest.raises(AttributeError, match=r"^This 'SVC' has no attribute 'predict_proba'$"):
        m.predict_proba(x)
    assert not hasattr(m, "predict_proba")


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
