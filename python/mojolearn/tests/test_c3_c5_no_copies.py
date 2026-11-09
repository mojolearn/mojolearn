# SPDX-License-Identifier: Apache-2.0
"""Fits and block forwards read the caller's float32 C-order buffer in place.

0.8.66 regressions C3 and C5 (docs/plans/REGRESSIONS_0866_VS_0855.md): k-NN
fit and IsolationForest fit keep a caller's float32 C matrix that owns its
memory, frozen read-only, instead of a zero-filled copy; a view keeps the
copy and its owner stays writable. A block forward reads a C-contiguous
float32 `x` in place and never writes it. The outputs are byte-identical to
the copied path's.
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
def test_knn_fit_keeps_an_owning_float32_matrix_read_only(cls):
    X = _x()
    y = (np.arange(len(X)) % 3).astype(np.int64)
    reference = X.copy()
    a = getattr(ml, cls)(n_neighbors=3)
    b = getattr(ml, cls)(n_neighbors=3)
    a.fit(X) if cls == "NearestNeighbors" else a.fit(X, y)
    b.fit(reference.copy()) if cls == "NearestNeighbors" else b.fit(reference.copy(), y)
    assert not X.flags.writeable
    with pytest.raises(ValueError):
        X[0, 0] = 1.0
    da, ia = a.kneighbors(reference[:5])
    db, ib = b.kneighbors(reference[:5])
    assert np.asarray(ia).tobytes() == np.asarray(ib).tobytes()
    assert np.asarray(da).tobytes() == np.asarray(db).tobytes()


def test_knn_fit_copies_a_view_and_leaves_its_owner_writable():
    fresh = _x(rows=128)
    for view in (lambda a: a[::2], lambda a: a[:64]):  # strided, then C-contiguous
        base = fresh.copy()
        source = view(base)
        query = view(fresh)[:4].copy()
        model = ml.NearestNeighbors(n_neighbors=2).fit(source)
        assert base.flags.writeable and source.flags.writeable
        before = np.asarray(model.kneighbors(query)[1]).tobytes()
        base[:] = 0.0  # the fitted index is a copy: the caller's edit cannot reach it
        assert np.asarray(model.kneighbors(query)[1]).tobytes() == before


def test_iforest_fit_keeps_an_owning_float32_matrix_read_only():
    X = _x(rows=256)
    reference = X.copy()
    a = ml.IsolationForest(n_estimators=8, random_state=5).fit(X)
    b = ml.IsolationForest(n_estimators=8, random_state=5).fit(reference.copy())
    assert a.input_copied_ is False
    assert not X.flags.writeable
    with pytest.raises(ValueError):
        X[0, 0] = 1.0
    sa = np.asarray(a.score_samples(reference))
    sb = np.asarray(b.score_samples(reference))
    assert sa.tobytes() == sb.tobytes()


def test_iforest_fit_copies_a_view():
    base = _x(rows=512)
    X = base[:256]
    model = ml.IsolationForest(n_estimators=8, random_state=5).fit(X)
    assert base.flags.writeable and X.flags.writeable
    before = np.asarray(model.score_samples(X.copy())).tobytes()
    snapshot = X.copy()
    base[:] = 0.0
    assert np.asarray(model.score_samples(snapshot)).tobytes() == before


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
