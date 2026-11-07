"""lane/py-dn-kern (2026-09-28): the float32 dense-to-COO scan
(`_spectral_impl._DenseCOO` over `nonzero_f32_count/fill`) and the
precomputed kNN affinity (`knn_affinity_f32`) against the Python loops they
replaced, kept here verbatim as the oracles, on raw bytes.

Planted values exercise the scan's test (0.0, -0.0 a zero, NaN kept,
subnormals, inf) and the selection's order (ties between equal distances go
to the lower column, -0.0 equal to 0.0, rows with exactly k candidates,
and the refusals)."""
import array
import random

import pytest

from mojolearn._array import Array
from mojolearn._buffer import as_f32_c, frombytes
from mojolearn._expansion_decomp import _M
from mojolearn._spectral_impl import SpectralEmbedding, _DenseCOO


def _dense_oracle(m):
    rows, cols, vals = array.array("i"), array.array("i"), array.array("f")
    for i in range(m.r):
        base = i * m.c
        for j in range(m.c):
            v = m.s[base + j]
            if v != 0.0:
                rows.append(i)
                cols.append(j)
                vals.append(v)
    return rows.tobytes(), cols.tobytes(), vals.tobytes()


def _knn_oracle(k, X):
    tocoo = getattr(X, "tocoo", None)
    if callable(tocoo):
        coo = tocoo()
        n = int(coo.shape[0])
        per = [[] for _ in range(n)]
        for r, c, v in zip(coo.row.tolist(), coo.col.tolist(), as_f32_c(coo.data, ndim=1, name="data")[0].tolist()):
            per[int(r)].append((v, int(c)))
    else:
        d, _ = as_f32_c(X, ndim=2, name="X")
        n = int(d.shape[0])
        per = [[(v, j) for j, v in enumerate(row)] for row in d.tolist()]
    C = _M.zeros(n, n)
    for i, cand in enumerate(per):
        if any(v != v or v < 0 for v, _ in cand):
            raise ValueError("negative or NaN")
        if len(cand) < k:
            raise ValueError(f"row {i} has {len(cand)} stored distances, fewer than n_neighbors={k}")
        for _, j in sorted(cand)[:k]:
            C.s[i * n + j] = 1.0
    A = _M.zeros(n, n)
    for i in range(n):
        for j in range(n):
            t = C.s[i * n + j] + C.s[j * n + i]
            A.s[i * n + j] = 0.5 if t == 1.0 else t / 2
    return A.s.tobytes()


def _matrix(n, seed, plant):
    rnd = random.Random(seed)
    pool = [0.0, 0.0, -0.0, 1.0, 1.0, 2.5, 1e-45, 1.1754943508222875e-38, 0.25, 0.25]
    vals = [rnd.choice(pool) if plant else rnd.random() for _ in range(n * n)]
    return _M(array.array("f", vals), n, n)


@pytest.mark.parametrize("seed", range(6))
def test_dense_coo_matches_the_python_scan(seed):
    m = _matrix(17 + seed, seed, plant=True)
    m.s[3] = float("nan")
    m.s[5] = float("inf")
    got = _DenseCOO(m)
    want = _dense_oracle(m)
    assert (got.row.tobytes(), got.col.tobytes(), got.data.tobytes()) == want


@pytest.mark.parametrize("seed", range(6))
def test_dense_knn_affinity_matches_the_python_loops(seed):
    n = 23 + seed
    m = _matrix(n, 100 + seed, plant=seed % 2 == 0)
    X = frombytes(m.s.tobytes(), "<f4", (n, n))
    se = SpectralEmbedding(affinity="precomputed_nearest_neighbors", n_neighbors=3 + seed)
    se._validate_params()  # the fit-time parameter checks derive the private settings
    assert se._precomputed_knn_affinity(X).tobytes() == _knn_oracle(3 + seed, X)


# DELETED test_sparse_knn_affinity_matches_the_python_loops (lane ne1-k-fail-b,
# 2026-11-07): SpectralEmbedding takes dense X only (the public door's refusal
# below); the dense kNN affinity is test_dense_knn_affinity_matches_the_python_loops.


def test_refusals():
    """A sparse graph is refused by name; a negative or NaN distance in a
    dense matrix is refused."""
    sparse = pytest.importorskip("scipy.sparse")
    import numpy as np
    n = 6
    se = SpectralEmbedding(affinity="precomputed_nearest_neighbors", n_neighbors=2)
    se._validate_params()  # the fit-time parameter checks derive the private settings
    graph = sparse.coo_matrix((np.ones(3, "<f4"), (np.arange(3), np.arange(1, 4))), shape=(n, n))
    with pytest.raises(TypeError, match="SpectralEmbedding takes dense X only"):
        se.fit(graph)
    for bad in (-1.0, float("nan")):
        D = np.ones((n, n), "<f4")
        D[2, 4] = bad
        with pytest.raises(ValueError, match="negative or NaN"):
            se._precomputed_knn_affinity(D)
