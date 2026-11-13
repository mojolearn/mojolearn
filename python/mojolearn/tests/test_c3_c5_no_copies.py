# SPDX-License-Identifier: Apache-2.0
"""Fits and block forwards never change the caller's float32 buffer.

0.8.66 regressions C3 and C5 (docs/plans/REGRESSIONS_0866_VS_0855.md): a
k-NN fit keeps a caller's float32 C matrix by reference (as scikit-learn
does) and never freezes or writes it; IsolationForest borrows it too on a
binary that keeps the forest resident (as 0.8.55 did) and keeps its own copy
on one that refits per scoring call, which reads the training matrix again. A block forward reads a
C-contiguous float32 `x` in place and never writes it or its flags. The
outputs are byte-identical to the copied path's.
"""
import os

import numpy as np
import pytest

os.environ.setdefault("MOJOLEARN_NUMERIC_MODE", "identical")

import mojolearn as ml  # noqa: E402


def _x(rows=64, cols=6, seed=3):
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(rng.standard_normal((rows, cols)).astype(np.float32))


@pytest.mark.parametrize("cls", ["NearestNeighbors", "KNeighborsRegressor", "KNeighborsClassifier"])
def test_knn_fit_keeps_the_float32_matrix_by_reference_and_never_changes_it(cls):
    X = _x()
    y = (np.arange(len(X)) % 3).astype(np.int64)
    reference = X.copy()
    a = getattr(ml, cls)(n_neighbors=3)
    b = getattr(ml, cls)(n_neighbors=3)
    a.fit(X) if cls == "NearestNeighbors" else a.fit(X, y)
    b.fit(reference.copy()) if cls == "NearestNeighbors" else b.fit(reference.copy(), y)
    # The caller's array is never frozen or written.
    assert X.flags.writeable
    assert X.tobytes() == reference.tobytes()
    da, ia = a.kneighbors(reference[:5])
    db, ib = b.kneighbors(reference[:5])
    assert np.asarray(ia).tobytes() == np.asarray(ib).tobytes()
    assert np.asarray(da).tobytes() == np.asarray(db).tobytes()
    # The documented contract: after an in-place edit, fit again.
    X[:] = X[::-1].copy()
    a.fit(X) if cls == "NearestNeighbors" else a.fit(X, y)
    c = getattr(ml, cls)(n_neighbors=3)
    c.fit(X.copy()) if cls == "NearestNeighbors" else c.fit(X.copy(), y)
    assert np.asarray(a.kneighbors(reference[:5])[1]).tobytes() == np.asarray(c.kneighbors(reference[:5])[1]).tobytes()


def test_knn_fit_casts_a_strided_view_and_leaves_its_owner_alone():
    base = _x(rows=128)
    before_bytes = base.tobytes()
    model = ml.NearestNeighbors(n_neighbors=2).fit(base[::2])
    assert base.flags.writeable and base.tobytes() == before_bytes
    query = base[::2][:4].copy()
    before = np.asarray(model.kneighbors(query)[1]).tobytes()
    base[:] = 0.0  # the strided view was cast into the model's own Array
    assert np.asarray(model.kneighbors(query)[1]).tobytes() == before


def test_iforest_fit_copies_and_never_changes_the_caller_matrix():
    X = _x(rows=256)
    reference = X.copy()
    a = ml.IsolationForest(n_estimators=8, random_state=5).fit(X)
    b = ml.IsolationForest(n_estimators=8, random_state=5).fit(reference.copy())
    assert X.flags.writeable and X.tobytes() == reference.tobytes()
    sa = np.asarray(a.score_samples(reference)).tobytes()
    assert sa == np.asarray(b.score_samples(reference)).tobytes()
    X[:] = 0.0  # scoring reads the fit's own copy of the training matrix
    assert np.asarray(a.score_samples(reference)).tobytes() == sa


def test_iforest_public_scores_are_the_float32_scores_widened():
    # Lane restore-iforest-kde: the device epilogue writes the float64
    # scores itself (IDN_IF_SCORES_F64); they are float32 values widened
    # exactly, as the door's cast wrote them, and decision_function is
    # score_samples - offset_ rounded once in float32.
    X = _x(rows=256)
    a = ml.IsolationForest(n_estimators=8, random_state=5).fit(X)
    s = np.asarray(a.score_samples(X))
    d = np.asarray(a.decision_function(X))
    assert s.dtype == np.float64 and d.dtype == np.float64
    assert s.tobytes() == s.astype(np.float32).astype(np.float64).tobytes()
    expected = (s.astype(np.float32) - np.float32(a.offset_)).astype(np.float64)
    assert d.tobytes() == expected.tobytes()


@pytest.mark.parametrize("form", ["float64", "strided", "fortran"])
def test_iforest_fit_keeps_its_cast_copy_and_matches_the_float32_fit(form):
    # Lane perf-iforest: a non-float32-C X is cast once into storage the fit
    # owns and kept as the training matrix (no zero-filled store, no second
    # copy); the forest and the scores equal the float32 C fit's.
    reference = _x(rows=256)
    if form == "float64":
        X = reference.astype(np.float64)
    elif form == "strided":
        X = np.repeat(reference, 2, axis=1)[:, ::2]
    else:
        X = np.asfortranarray(reference)
    a = ml.IsolationForest(n_estimators=8, random_state=5).fit(X)
    b = ml.IsolationForest(n_estimators=8, random_state=5).fit(reference.copy())
    assert type(a.__dict__["_x"]) is type(b.__dict__["_x"])
    assert np.asarray(a.__dict__["_x"]).tobytes() == reference.tobytes()
    sa = np.asarray(a.score_samples(reference)).tobytes()
    assert sa == np.asarray(b.score_samples(reference)).tobytes()
    X[...] = 0.0  # the fit kept its own cast copy
    assert np.asarray(a.score_samples(reference)).tobytes() == sa
    assert np.asarray(a.__dict__["_x"]).tobytes() == reference.tobytes()


@pytest.mark.parametrize("kind", ["mamba1", "mamba2", "mamba3"])
def test_block_forward_reads_x_in_place_and_matches_the_copy(kind):
    from mojolearn.tests.test_mamba_native_public import _block, _tokens
    block, dm = _block(kind)
    x = _tokens(2, 7, dm)
    before = x.tobytes()
    y_in_place = np.asarray(block.forward(x)).tobytes()
    y_copy = np.asarray(block.forward(x.copy())).tobytes()
    assert y_in_place == y_copy
    assert x.tobytes() == before
    assert x.flags.writeable  # a forward never freezes its input
    strided = np.ascontiguousarray(np.repeat(x, 2, axis=2))[:, :, ::2]
    assert np.asarray(block.forward(strided)).tobytes() == y_copy
