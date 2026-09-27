# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""SVC multiclass: one-vs-one over the binary solver (lane
x-neighbors-svc-multiclass, 2026-09-27).

Refusal checks always run. The value checks run through whichever SVM binding
this install loads and are skipped, saying so, without one. Each pair machine
is the binary solver on that pair's rows, so a pair's decision must be the
bytes of a binary SVC fitted on those rows; the multiclass layout and vote are
compared with scikit-learn (to a tolerance on the decisions, which come from
a different solver).
"""

import os
import tempfile

import numpy as np
import pytest

import mojolearn
from mojolearn._svm_impl import SVC


def _data(n=240, k=4, seed=5):
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


def test_break_ties_with_ovo_is_refused():
    with pytest.raises(ValueError, match="break_ties"):
        SVC(break_ties=True, decision_function_shape="ovo")
    with pytest.raises(ValueError, match="decision_function_shape"):
        SVC(decision_function_shape="ovx")


def test_one_class_is_refused():
    x, _ = _data(n=20)
    with pytest.raises(ValueError, match="at least two"):
        SVC().fit(x, np.zeros(20, dtype=np.int64))


def test_each_pair_is_the_binary_solver_on_its_rows():
    x, y = _data()
    m = _fit_or_skip(x, y, kernel="rbf", gamma=0.1)
    q = x[:50]
    ovo = np.asarray(m.decision_function(q))
    assert ovo.shape == (50, 6) and ovo.dtype == np.float32
    col = 0
    for i in range(4):
        for j in range(i + 1, 4):
            rows = (y == i) | (y == j)
            b = SVC(kernel="rbf", gamma=0.1).fit(x[rows], (y[rows] == j).astype(np.int64))
            ref = -np.asarray(b.decision_function(q))
            assert ref.tobytes() == ovo[:, col].tobytes(), (i, j)
            col += 1


def test_gamma_scale_is_resolved_on_the_whole_x():
    x, y = _data()
    m = _fit_or_skip(x, y, gamma="scale")
    from mojolearn._scale_gamma import scale_gamma
    assert m._gamma == scale_gamma(x.ravel().tolist(), x.shape[1])


def test_matches_scikit_learn_layout_and_votes():
    sk = pytest.importorskip("sklearn.svm")
    x, y = _data()
    m = _fit_or_skip(x, y, kernel="rbf", gamma=0.1, tol=1e-4)
    r = sk.SVC(kernel="rbf", gamma=0.1, tol=1e-4, decision_function_shape="ovo").fit(x, y)
    assert list(np.asarray(m.n_support_)) == list(r.n_support_)
    assert np.asarray(m.dual_coef_).shape == r.dual_coef_.shape
    assert np.asarray(m.intercept_).shape == r.intercept_.shape
    np.testing.assert_allclose(np.asarray(m.intercept_), r.intercept_, atol=5e-2)
    np.testing.assert_allclose(np.asarray(m.decision_function(x)), r.decision_function(x), atol=5e-2)
    assert (np.asarray(m.predict(x)) == r.predict(x)).mean() >= 0.99
    m.decision_function_shape = "ovr"
    r.decision_function_shape = "ovr"
    np.testing.assert_allclose(np.asarray(m.decision_function(x)), r.decision_function(x), atol=5e-2)
    t = _fit_or_skip(x, y, kernel="rbf", gamma=0.1, tol=1e-4, decision_function_shape="ovr", break_ties=True)
    rt = sk.SVC(kernel="rbf", gamma=0.1, tol=1e-4, decision_function_shape="ovr", break_ties=True).fit(x, y)
    assert (np.asarray(t.predict(x)) == rt.predict(x)).mean() >= 0.99


def test_linear_coef_matches_scikit_learn():
    sk = pytest.importorskip("sklearn.svm")
    x, y = _data(k=3)
    m = _fit_or_skip(x, y, kernel="linear", C=0.5, tol=1e-4)
    r = sk.SVC(kernel="linear", C=0.5, tol=1e-4).fit(x, y)
    np.testing.assert_allclose(np.asarray(m.coef_), r.coef_, atol=5e-2)


def test_string_labels_and_weights():
    x, y = _data()
    names = np.array(["a", "b", "c", "d"])[y]
    w = 0.5 + 0.5 * (np.arange(len(y)) % 3)
    m = _fit_or_skip(x, list(names), class_weight="balanced")
    assert m.classes_ == ["a", "b", "c", "d"]
    n = _fit_or_skip(x, y, class_weight={0: 2.0})
    s = _fit_or_skip(x, y)
    s2 = SVC().fit(x, y, sample_weight=w)
    assert np.asarray(n.dual_coef_).tobytes() != np.asarray(s.dual_coef_).tobytes()
    assert np.asarray(s2.dual_coef_).tobytes() != np.asarray(s.dual_coef_).tobytes()


def test_repeat_calls_and_save_load_round_trip():
    x, y = _data()
    m = _fit_or_skip(x, y, decision_function_shape="ovr", break_ties=True)
    a = (np.asarray(m.decision_function(x)).tobytes(), np.asarray(m.predict(x)).tobytes())
    b = (np.asarray(m.decision_function(x)).tobytes(), np.asarray(m.predict(x)).tobytes())
    assert a == b
    m2 = SVC(decision_function_shape="ovr", break_ties=True).fit(x, y)
    assert np.asarray(m2.dual_coef_).tobytes() == np.asarray(m.dual_coef_).tobytes()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "svc.npz")
        m.save(path)
        back = SVC.load(path)
        for name in ("dual_coef_", "support_", "support_vectors_", "intercept_", "n_support_"):
            assert np.asarray(getattr(back, name)).tobytes() == np.asarray(getattr(m, name)).tobytes(), name
        assert np.asarray(back.decision_function(x)).tobytes() == a[0]
        assert np.asarray(back.predict(x)).tobytes() == a[1]
        try:
            host = mojolearn.host_model(path)
        except ImportError as exc:
            pytest.skip(f"no host binding: {exc}")
        assert np.asarray(host.decision_function(x)).tobytes() == a[0]


def _int_data(n=120, k=3, seed=9):
    rng = np.random.default_rng(seed)
    y = np.arange(n) % k
    a = (rng.integers(-3, 4, size=(n, 4)) + 2 * y[:, None]).astype(np.float32)
    return a, y.astype(np.int64)


def test_precomputed_is_the_linear_kernel_bit_for_bit():
    """An integer-valued X has an exact linear Gram, so kernel='precomputed'
    on that Gram must give kernel='linear''s bits: binary, multiclass, SVR."""
    a, y = _int_data()
    k = (a.astype(np.int64) @ a.astype(np.int64).T).astype(np.float32)
    q = a[:30]
    kq = (q.astype(np.int64) @ a.astype(np.int64).T).astype(np.float32)
    for yy in ((y > 0).astype(np.int64), y):
        lin = _fit_or_skip(a, yy, kernel="linear", C=0.01)
        pre = SVC(kernel="precomputed", C=0.01).fit(k, yy)
        for name in ("dual_coef_", "support_", "intercept_"):
            assert np.asarray(getattr(pre, name)).tobytes() == np.asarray(getattr(lin, name)).tobytes(), name
        assert np.asarray(pre.decision_function(kq)).tobytes() == np.asarray(lin.decision_function(q)).tobytes()
        assert np.asarray(pre.predict(kq)).tobytes() == np.asarray(lin.predict(q)).tobytes()
    from mojolearn._svm_impl import SVR
    t = (a[:, 0] - a[:, 1]).astype(np.float32)
    lin = SVR(kernel="linear", C=0.01).fit(a, t)
    pre = SVR(kernel="precomputed", C=0.01).fit(k, t)
    assert np.asarray(pre.dual_coef_).tobytes() == np.asarray(lin.dual_coef_).tobytes()
    assert np.asarray(pre.predict(kq)).tobytes() == np.asarray(lin.predict(q)).tobytes()


def test_precomputed_needs_a_square_matrix():
    a, y = _int_data()
    with pytest.raises(ValueError, match="square"):
        SVC(kernel="precomputed").fit(a, y)


def test_precomputed_save_load_round_trip():
    a, y = _int_data()
    k = (a.astype(np.int64) @ a.astype(np.int64).T).astype(np.float32)
    for yy in ((y > 0).astype(np.int64), y):
        m = _fit_or_skip(k, yy, kernel="precomputed", C=0.01)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "svc.npz")
            m.save(path)
            back = SVC.load(path)
            assert np.asarray(back.decision_function(k)).tobytes() == np.asarray(m.decision_function(k)).tobytes()
