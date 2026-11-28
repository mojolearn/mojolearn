# SPDX-License-Identifier: Apache-2.0
"""The single-cell graph functions at the Python API boundary (lane sc-graph-1, 2026-11-28):
mojolearn.shared_neighbors_graph, mojolearn.PAGA and mojolearn.manifold.DiffusionMap. The input forms they admit,
their refusals, their parameters and what they return; NumPy serves as the independent external oracle (no SciPy,
Andrew 2026-11-28). The kernels' numerical checks are native (checks/native_tests/x_neighbors_snn_paga_check.mojo)."""
from types import SimpleNamespace

import numpy as np
import pytest

import mojolearn


def _dense(g):
    """The dense matrix of a CSR record (NumPy)."""
    out = np.zeros(g.shape, np.float64)
    indptr, indices, data = (np.asarray(g.indptr), np.asarray(g.indices), np.asarray(g.data))
    rows = np.repeat(np.arange(g.shape[0]), np.diff(indptr))
    out[rows, indices] = data
    return out


def _knn_record(knn):
    n, k = knn.shape
    return SimpleNamespace(indptr=np.arange(0, n * k + 1, k, dtype=np.int32), indices=knn.ravel().astype(np.int32),
                           data=np.ones(n * k, np.float32), shape=(n, n), format="csr")


def _knn(n=60, k=8, seed=0):
    """An (n, k) neighbor-index array of random points, the point itself first (NumPy brute force)."""
    x = np.random.default_rng(seed).normal(size=(n, 4))
    d = ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1)
    d[np.arange(n), np.arange(n)] = -1.0
    return np.argsort(d, axis=1, kind="stable")[:, :k].astype(np.int64)


def _jaccard(knn, prune):
    n, k = knn.shape
    member = np.zeros((n, n), bool)
    member[np.repeat(np.arange(n), k), knn.ravel()] = True
    s = (member[:, None, :] & member[None, :, :]).sum(-1)
    j = np.where(s > 0, s / (2 * k - s), 0.0)
    return np.where(j >= prune, j, 0.0)


# ---------------------------------------------------------------- shared_neighbors_graph


def test_snn_matches_seurat_formula():
    knn = _knn()
    g = mojolearn.shared_neighbors_graph(knn)
    assert g.format == "csr" and np.asarray(g.data).dtype == np.float32
    assert g.shape == (60, 60)
    want = _jaccard(knn, 1 / 15)
    d = _dense(g)
    np.testing.assert_allclose(d, want, rtol=1e-6, atol=0)
    assert np.array_equal(d, d.T)
    assert np.all(np.diag(d) == 1.0)
    for i in range(g.shape[0]):
        row = g.indices[g.indptr[i]:g.indptr[i + 1]]
        assert np.all(np.diff(row) > 0)


def test_snn_prune_and_input_forms():
    knn = _knn(seed=1)
    full = mojolearn.shared_neighbors_graph(knn, prune_snn=0.0)
    np.testing.assert_allclose(_dense(full), _jaccard(knn, 0.0), rtol=1e-6, atol=0)
    strict = mojolearn.shared_neighbors_graph(knn, prune_snn=0.5)
    assert len(strict.indices) < len(full.indices)
    n, k = knn.shape
    tup = (np.arange(0, n * k + 1, k, dtype=np.int32), knn.ravel().astype(np.int32), n)
    base = _dense(mojolearn.shared_neighbors_graph(knn))
    np.testing.assert_array_equal(_dense(mojolearn.shared_neighbors_graph(tup)), base)
    np.testing.assert_array_equal(_dense(mojolearn.shared_neighbors_graph(_knn_record(knn))), base)


def test_snn_feeds_louvain_and_leiden():
    g = mojolearn.shared_neighbors_graph(_knn(n=80, k=10, seed=2))
    for est in (mojolearn.Louvain(), mojolearn.Leiden()):
        labels = est.fit_predict(g)
        assert labels.shape == (80,)


def test_snn_refusals():
    knn = _knn()
    with pytest.raises(ValueError, match="prune_snn"):
        mojolearn.shared_neighbors_graph(knn, prune_snn=1.5)
    bad = knn.copy()
    bad[0, 1] = 60
    with pytest.raises(ValueError, match=r"outside \[0, n\)"):
        mojolearn.shared_neighbors_graph(bad)
    dup = knn.copy()
    dup[3, 2] = dup[3, 1]
    with pytest.raises(ValueError, match="twice"):
        mojolearn.shared_neighbors_graph(dup)
    with pytest.raises(TypeError, match="integer"):
        mojolearn.shared_neighbors_graph(knn.astype(np.float32))


def test_label_order_size():
    g = mojolearn.shared_neighbors_graph(_knn(n=120, k=10, seed=7))
    for cls in (mojolearn.Louvain, mojolearn.Leiden):
        first = cls().fit_predict(g)
        size = cls(label_order="size").fit_predict(g)
        counts = np.bincount(size)
        assert np.all(np.diff(counts) <= 0)
        # the same partition, renumbered
        pairs = set(zip(np.asarray(first).tolist(), np.asarray(size).tolist()))
        assert len(pairs) == len(set(np.asarray(first).tolist()))
        with pytest.raises(ValueError, match="label_order"):
            cls(label_order="largest").fit(g)


# ---------------------------------------------------------------- PAGA


def _paga_oracle(graph, groups):
    """Scanpy's PAGA v1.2 statistic in NumPy (every stored entry one directed edge)."""
    c = groups.max() + 1
    ns = np.bincount(groups, minlength=c)
    indptr, indices = np.asarray(graph.indptr), np.asarray(graph.indices)
    rows = np.repeat(np.arange(graph.shape[0]), np.diff(indptr))
    counts = np.zeros((c, c))
    np.add.at(counts, (groups[rows], groups[indices]), 1)
    es = counts.sum(axis=1)
    inter = counts.copy()
    np.fill_diagonal(inter, 0)
    v = inter + inter.T
    n = graph.shape[0]
    expected = (es[:, None] * ns[None, :] + es[None, :] * ns[:, None]) / (n - 1)
    con = np.where(v > 0, np.minimum(1.0, np.divide(v, expected, out=np.ones_like(v), where=expected != 0)), 0.0)
    return con, v, ns


def test_paga_statistic_and_tree():
    knn = _knn(n=90, k=6, seed=3)
    n, k = knn.shape
    graph = _knn_record(knn)
    groups = (np.arange(n) * 5 // n).astype(np.int32)
    p = mojolearn.PAGA().fit(graph, groups)
    con, v, ns = _paga_oracle(graph, groups)
    assert p.n_groups_ == 5
    np.testing.assert_array_equal(np.asarray(p.group_sizes_), ns)
    assert p.connectivities_.format == "csr" and p.connectivities_tree_.format == "csr"
    np.testing.assert_allclose(_dense(p.connectivities_), con, rtol=2e-6, atol=0)
    tree = _dense(p.connectivities_tree_)
    assert np.array_equal(tree, tree.T)
    edges = np.argwhere(np.triu(tree) > 0)
    assert len(edges) <= 4
    # a maximum spanning forest of the edge counts: its weight equals Kruskal's (NumPy, union-find)
    parent = list(range(v.shape[0]))

    def root(a):
        while parent[a] != a:
            a = parent[a]
        return a
    want = 0
    for a, b in sorted(zip(*np.nonzero(np.triu(v, 1))), key=lambda e: -v[e]):
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[ra] = rb
            want += v[a, b]
    assert v[edges[:, 0], edges[:, 1]].sum() == want


def test_paga_dense_equals_sparse_and_refusals():
    knn = _knn(n=40, k=5, seed=4)
    n, k = knn.shape
    graph = _knn_record(knn)
    groups = (np.arange(n) % 3).astype(np.int32)
    a = mojolearn.PAGA().fit(graph, groups)
    b = mojolearn.PAGA().fit(_dense(graph).astype(np.float32), groups)
    np.testing.assert_array_equal(_dense(a.connectivities_), _dense(b.connectivities_))
    with pytest.raises(ValueError, match="v1.2"):
        mojolearn.PAGA(model="v1.0").fit(graph, groups)
    with pytest.raises(ValueError, match="negative"):
        mojolearn.PAGA().fit(graph, np.where(groups == 0, -1, groups).astype(np.int32))
    with pytest.raises(ValueError, match="entries"):
        mojolearn.PAGA().fit(graph, groups[:-1])


# ---------------------------------------------------------------- DiffusionMap


def _transitions_sym(w):
    """Scanpy's density-normalized symmetric transitions (Neighbors.compute_transitions), dense NumPy."""
    w = np.asarray(w, np.float64)
    np.fill_diagonal(w, 0)
    q = w.sum(axis=0)
    k = w / np.outer(q, q)
    z = np.sqrt(k.sum(axis=0))
    return k / np.outer(z, z)


def test_diffmap_matches_scanpy_eigenpairs():
    w = _dense(mojolearn.shared_neighbors_graph(_knn(n=100, k=10, seed=5))).astype(np.float32)
    dm = mojolearn.manifold.DiffusionMap(n_comps=6).fit(w)
    emb = np.asarray(dm.embedding_)
    evals = np.asarray(dm.eigenvalues_)
    assert emb.shape == (100, 6) and evals.shape == (6,)
    assert np.all(np.diff(evals) <= 1e-6)
    t = _transitions_sym(w)
    ref = np.sort(np.linalg.eigvalsh(t))[::-1][:6]
    np.testing.assert_allclose(evals, ref, atol=1e-4)
    # each column is an eigenvector of T with its eigenvalue, unit norm
    np.testing.assert_allclose(np.linalg.norm(emb, axis=0), 1.0, atol=1e-4)
    np.testing.assert_allclose(t @ emb, emb * evals[None, :], atol=2e-3)


def test_diffmap_sparse_input_fit_transform_and_refusals():
    g = mojolearn.shared_neighbors_graph(_knn(n=50, k=8, seed=6))
    dm = mojolearn.manifold.DiffusionMap(n_comps=4, random_state=3)
    y = dm.fit_transform(g)
    np.testing.assert_array_equal(np.asarray(y), np.asarray(dm.embedding_))
    np.testing.assert_array_equal(np.asarray(mojolearn.manifold.DiffusionMap(n_comps=4, random_state=3)
                                             .fit(_dense(g).astype(np.float32)).embedding_), np.asarray(y))
    params = mojolearn.manifold.DiffusionMap().get_params()
    assert params["n_comps"] == 15 and params["random_state"] == 0
    with pytest.raises(ValueError, match="n_comps"):
        mojolearn.manifold.DiffusionMap(n_comps=50).fit(g)
    with pytest.raises(ValueError, match="negative"):
        mojolearn.manifold.DiffusionMap(n_comps=2).fit(-_dense(g).astype(np.float32))
    with pytest.raises(ValueError, match="n_comps"):
        mojolearn.manifold.DiffusionMap(n_comps=0).fit(g)
