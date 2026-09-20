# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""SVC(kernel='poly') (lane/cpu-training-small-gaps, 2026-09-15).

Source and refusal checks always run. The value checks run through whichever
SVM binding this install loads (the Metal identical set, or the SVM host
binding under MOJOLEARN_HOST_DIR) and are skipped, saying so, without one.
The kernel is the identical linear Gram followed by kernel_methods' DEVIATION
1663 polynomial epilogue (one fused multiply-add, then an ascending repeated
product), so values are compared with scikit-learn to a tolerance, and with
themselves bitwise.
"""

import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

import mojolearn
from mojolearn import host_surface
from mojolearn._svm_impl import SVC, SVR

ROOT = Path(__file__).resolve().parents[3]


def _fit_or_skip(**kw):
    rng = np.random.default_rng(11)
    x = rng.normal(size=(300, 4)).astype(np.float32)
    y = ((x[:, 0] * x[:, 1] + 0.5 * x[:, 2]) > 0).astype(np.float32)
    try:
        from mojolearn._cpu_reference import reference_training
        with reference_training():
            m = SVC(**kw).fit(x, y)
    except (ImportError, NotImplementedError) as exc:
        pytest.skip(f"no SVM binding that fits on this install: {exc}")
    return m, x, y


def test_both_bindings_take_degree_and_coef0():
    for rel in ("bindings/_mojolearn_svm.mojo", "bindings/_mojolearn_svm_host.mojo"):
        text = (ROOT / rel).read_text()
        fit = text.split("def svc_fit_binding(", 1)[1].split("\ndef ", 1)[0]
        predict = text.split("def svc_predict_binding(", 1)[1].split("\ndef ", 1)[0]
        assert "len(params) != 10" in fit and "params[9]" in fit, rel
        assert "len(params) != 12" in predict and "params[11]" in predict, rel


def test_the_gram_and_the_oracle_carry_the_polynomial_arm():
    km = (ROOT / "svm/impl/distance/kernel_matrices.mojo").read_text()
    assert "kp.kernel == KERNEL_POLYNOMIAL" in km and "polynomial_epilogue_kernel" in km
    oracle = (ROOT / "svm/host/smo_oracle.mojo").read_text()
    cell = oracle.split("def _kernel_cell[", 1)[1].split("\ndef ", 1)[0]
    assert "KERNEL_POLYNOMIAL" in cell and "_mul[dt](acc, t)" in cell
    tsv = (ROOT / "svm/NOT_IMPLEMENTED.tsv").read_text()
    assert "one `identical_pow` away" not in tsv


def test_host_update_f_parallelizes_rows_not_kernel_folds():
    oracle = (ROOT / "svm/host/smo_oracle.mojo").read_text()
    fit = oracle.split("def smo_oracle_fit[", 1)[1].split("# Results.", 1)[0]
    assert "sync_parallelize(_update_f, update_tasks)" in fit
    assert "host_predict_task_count(n_rows)" in fit
    assert "if n_rows * nnz * k < (1 << 18):\n                update_tasks = 1" in fit
    worker = fit.split("def _update_f(", 1)[1].split("if update_tasks == 1:", 1)[0]
    assert "for i in range(lo, hi):" in worker
    assert "for rr in range(nnz):" in worker
    assert worker.index("for i in range(lo, hi):") < worker.index("for rr in range(nnz):")
    assert "f[i + n_rows] = _flush[dt](f[i + n_rows] + acc)" in worker


def test_manifest_covers_the_lane():
    assert "svc-poly" in host_surface.family("svm")["training_lanes"]


@pytest.mark.parametrize("kw, error, match", [
    (dict(kernel="poly", degree=2.5), TypeError, "integer"),
    (dict(kernel="poly", degree=True), TypeError, "integer"),
    (dict(kernel="poly", degree=-1), ValueError, r"\[0, 32\]"),
    (dict(kernel="poly", degree=33), ValueError, r"\[0, 32\]"),
    (dict(kernel="poly", coef0=float("inf")), ValueError, "finite"),
    (dict(kernel="polynomial"), NotImplementedError, "poly"),
    (dict(kernel="rbf", degree=2), NotImplementedError, "read only by kernel='poly'"),
    (dict(kernel="linear", coef0=1.0), NotImplementedError, "read only by kernel='poly'"),
    (dict(kernel="sigmoid"), NotImplementedError, "TANH"),
])
def test_refusals_by_name(kw, error, match):
    with pytest.raises(error, match=match):
        SVC(**kw)


def test_svr_still_refuses_poly():
    with pytest.raises(NotImplementedError, match="poly"):
        SVR(kernel="poly")


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
