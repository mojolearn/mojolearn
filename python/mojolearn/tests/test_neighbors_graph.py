# SPDX-License-Identifier: Apache-2.0
"""The public boundary of `mojolearn.neighbors_graph` and `UMAP.graph_` (lane sc-leftovers-1, 2026-11-28): the
returned CSR tuples (dtypes, lengths, no self edge, symmetric connectivities), `graph_` before a fit
(AttributeError), built on first read and kept until the next fit, the same CSR as the connectivities, and the
refusals. Numerics are the native checks' (umap/checks/neighbors_graph_check.mojo); no reference is computed here."""

import numpy as np
import pytest

import mojolearn as ml


_RNG = np.random.default_rng(5)
X = _RNG.standard_normal((300, 6)).astype(np.float32)


def _skip_unbuilt(call):
    try:
        return call()
    except (ImportError, ModuleNotFoundError) as error:  # noqa: BLE001 - a binding not built here
        pytest.skip(f"compiled binding unavailable: {error}")


def _csr(t):
    indptr, indices, data = (np.asarray(a) for a in t)
    return indptr, indices, data


def test_csr_forms():
    dist, conn = _skip_unbuilt(lambda: ml.neighbors_graph(X, 8))
    ip, ix, dv = _csr(dist)
    assert ip.dtype == np.int64 and ix.dtype == np.int32 and dv.dtype == np.float32
    assert ip.shape == (301,) and ip[-1] == 300 * 7 and ix.shape == dv.shape == (300 * 7,)
    rows = np.repeat(np.arange(300), np.diff(ip))
    assert not (ix == rows).any()
    cp, cx, cv = _csr(conn)
    assert cp.dtype == np.int64 and cx.dtype == np.int32 and cv.dtype == np.float32
    assert cp.shape == (301,) and cx.shape == cv.shape == (int(cp[-1]),)
    dense = np.zeros((300, 300), dtype=np.float32)
    dense[np.repeat(np.arange(300), np.diff(cp)), cx] = cv
    assert np.array_equal(dense, dense.T)


def test_graph_attribute_lifecycle():
    model = ml.UMAP(n_neighbors=8, n_epochs=4, random_state=3)
    with pytest.raises(AttributeError):
        model.graph_
    _skip_unbuilt(lambda: model.fit(X))
    first = model.graph_
    assert isinstance(first, tuple) and len(first) == 3
    assert model.graph_ is first
    _, conn = ml.neighbors_graph(X, 8)
    for a, b in zip(first, conn):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    model.fit(X[:200])
    again = model.graph_
    assert again is not first and np.asarray(again[0]).shape == (201,)


def test_refusals():
    with pytest.raises(ValueError):
        _skip_unbuilt(lambda: ml.neighbors_graph(X, 8, method="hnsw"))
    with pytest.raises(ValueError):
        ml.neighbors_graph(X, 1)
    with pytest.raises(TypeError):
        ml.neighbors_graph(X, 8.5)
