"""lane/py-dn-kern (2026-09-28): the float32 dense-to-COO scan
(`_spectral_impl._DenseCOO` over `nonzero_f32_count/fill`) and the
precomputed kNN affinity (`knn_affinity_f32`) against the Python loops they
replaced, kept here verbatim as the oracles, on raw bytes.

Planted values exercise the scan's test (0.0, -0.0 a zero, NaN kept,
subnormals, inf) and the selection's order (ties between equal distances go
to the lower column, -0.0 equal to 0.0, duplicate COO entries, rows with
exactly k candidates, and the two refusals in row order)."""
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


class _Coo:
    def __init__(self, n, r, c, v):
        self.shape = (n, n)
        self.row = Array.from_list(r, "<i4")
        self.col = Array.from_list(c, "<i4")
        self.data = Array.from_list(v, "<f4")

    def tocoo(self):
        return self


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
    assert se._precomputed_knn_affinity(X).s.tobytes() == _knn_oracle(3 + seed, X)


@pytest.mark.parametrize("seed", range(6))
def test_sparse_knn_affinity_matches_the_python_loops(seed):
    rnd = random.Random(200 + seed)
    n, k = 19, 2 + seed % 3
    r, c, v = [], [], []
    for i in range(n):
        for _ in range(k + rnd.randrange(4)):
            r.append(i)
            c.append(rnd.randrange(n))
            v.append(rnd.choice([0.0, -0.0, 0.5, 0.5, 1.0, rnd.random()]))
    rnd_order = list(range(len(r)))
    rnd.shuffle(rnd_order)
    X = _Coo(n, [r[q] for q in rnd_order], [c[q] for q in rnd_order], [v[q] for q in rnd_order])
    se = SpectralEmbedding(affinity="precomputed_nearest_neighbors", n_neighbors=k)
    assert se._precomputed_knn_affinity(X).s.tobytes() == _knn_oracle(k, X)


def test_refusals_follow_the_row_order():
    n = 6
    X = _Coo(n, [0, 0, 1, 2, 2, 3, 3, 4, 4, 5, 5], [1, 2, 0, 1, 3, 0, 1, 2, 3, 0, 4],
             [1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, float("nan"), 1.0, 1.0])
    se = SpectralEmbedding(affinity="precomputed_nearest_neighbors", n_neighbors=2)
    with pytest.raises(ValueError, match="row 1 of the precomputed graph has 1 stored"):
        se._precomputed_knn_affinity(X)
    X2 = _Coo(n, [0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, -0 + 1], [1.0, -1.0, 1.0, 1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="negative or NaN"):
        se._precomputed_knn_affinity(X2)
