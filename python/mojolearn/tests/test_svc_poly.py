# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""SVC(kernel='poly') (lane/cpu-training-small-gaps, 2026-09-15).

Refusal checks always run. The value checks run through whichever SVM binding
this install loads (the Metal identical set, or the SVM host binding under
MOJOLEARN_HOST_DIR) and are skipped, saying so, without one. The kernel is
the identical linear Gram followed by kernel_methods' DEVIATION 1663
polynomial epilogue (one fused multiply-add, then an ascending repeated
product), so values are compared with scikit-learn to a tolerance, and with
themselves bitwise.
"""

import os
import tempfile

import numpy as np
import pytest

from mojolearn._svm_impl import SVC, SVR


def _fit_or_skip(**kw):
    rng = np.random.default_rng(11)
    x = rng.normal(size=(300, 4)).astype(np.float32)
    y = ((x[:, 0] * x[:, 1] + 0.5 * x[:, 2]) > 0).astype(np.float32)
    try:
        m = SVC(**kw).fit(x, y)
    except (ImportError, NotImplementedError) as exc:
        pytest.skip(f"no SVM binding that fits on this install: {exc}")
    return m, x, y


@pytest.mark.parametrize("kw, error, match", [
    (dict(kernel="poly", degree=2.5), TypeError, "integer"),
    (dict(kernel="poly", degree=True), TypeError, "integer"),
    (dict(kernel="poly", degree=-1), ValueError, r"\[0, 32\]"),
    (dict(kernel="poly", degree=33), ValueError, r"\[0, 32\]"),
    (dict(kernel="poly", coef0=float("inf")), ValueError, "finite"),
    (dict(kernel="polynomial"), NotImplementedError, "poly"),
    (dict(kernel="rbf", degree=2), NotImplementedError, "read only by kernel='poly'"),
    (dict(kernel="linear", coef0=1.0), NotImplementedError, "read only by kernel='poly'"),
    (dict(kernel="tanh"), NotImplementedError, "sigmoid"),
    (dict(kernel="sigmoid", coef0=float("nan")), ValueError, "finite"),
])
def test_refusals_by_name(kw, error, match):
    with pytest.raises(error, match=match):
        SVC(**kw).fit([[0.0], [1.0]], [0, 1])


def test_svr_carries_poly_and_sigmoid():
    assert SVR(kernel="poly", degree=2).degree == 2
    assert SVR(kernel="sigmoid", coef0=-0.5).coef0 == -0.5
    with pytest.raises(NotImplementedError, match="poly"):
        SVR(kernel="polynomial").fit([[0.0], [1.0]], [0.0, 1.0])


def test_poly_fit_repeats_bitwise_and_tracks_sklearn():
    kw = dict(C=1.0, kernel="poly", degree=3, gamma=0.25, coef0=1.0, max_iter=-1)
    m, x, y = _fit_or_skip(**kw)
    m2, _, _ = _fit_or_skip(**kw)
    d1 = np.asarray(m.decision_function(x))
    assert d1.tobytes() == np.asarray(m2.decision_function(x)).tobytes()
    svm = pytest.importorskip("sklearn.svm")
    ref = svm.SVC(C=1.0, kernel="poly", degree=3, gamma=0.25, coef0=1.0, tol=1e-3).fit(x, y)
    agree = float(np.mean(np.asarray(m.predict(x)) == ref.predict(x)))
    assert agree >= 0.97, agree


def test_poly_save_load_keeps_coef0_and_degree():
    m, x, _ = _fit_or_skip(C=1.0, kernel="poly", degree=2, gamma=0.5, coef0=-0.75)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "svc_poly.npz")
        m.save(path)
        back = SVC.load(path)
    assert back.kernel == "poly" and back.degree == 2 and back.coef0 == -0.75
    assert np.asarray(back.decision_function(x)).tobytes() == np.asarray(m.decision_function(x)).tobytes()
