# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Every `_mojolearn_x_decomp` entry point, called TWICE in one process, on
the GPU binding and on the CPU host binding (CURRENT DIRECTIVES 2026-09-27:
x_cluster and x_neighbors hung on the second GPU call of a process because
each call built a DeviceContext whose buffers outlived it; x_decomp keeps
ONE process-lifetime context, x_decomp/device.mojo `xd_ctx`). The two calls
must also return the same bytes, and the GPU the host's.

    .pixi/envs/test/bin/python -m pytest python/mojolearn/tests/test_x_decomp_repeat.py -q
"""
import array

import pytest

from mojolearn import _backend
from mojolearn._expansion_decomp import _Kit, _M


def _m(r, c, seed, lo=-1.0, hi=1.0):
    s = array.array("f")
    x = seed * 2654435761 + 1
    for _ in range(r * c):
        x = (x * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        s.append(lo + (hi - lo) * ((x >> 40) / float(1 << 24)))
    return _M(s, r, c)


def _spd(n, seed):
    A = _m(n + 3, n, seed)
    k = _Kit("identical", _backend.load_host_module("_mojolearn_x_decomp_host"))
    G = k.mm(A, A, ta=True)
    for i in range(n):
        G.s[i * n + i] += 1.0
    return G


def _calls(k):
    """(name, bytes) for every entry point, through the kit."""
    A, B = _m(7, 5, 1), _m(6, 5, 2)
    S = _spd(5, 3)
    out = []

    def rec(name, *ms):
        out.append((name, b"".join(bytes(m.s) if isinstance(m, _M) else bytes(array.array("f", m)) for m in ms)))

    rec("ew", k.ew("add", A, A), k.ew("exp", A))
    rec("gemm", k.mm(A, B, tb=True))
    rec("colsum", k.colsum(A))
    rec("rowsum", k.rowsum(A))
    rec("sqdist", k.sqdist(A, B), k.pdist(A, B, 1), k.pdist(A, B, 3, 1.5), k.pdist(A, B, 4))
    rec("rand", k.rand(3, 4, 7, 1, 0), k.rand(3, 4, 7, 2, 1))
    lu, piv, _ = k.lu(S)
    rec("lu", lu, [float(p) for p in piv])
    rec("lu_solve", k.lu_solve(lu, piv, _m(5, 2, 4)), k.lu_solve(lu, piv, _m(5, 2, 4), trans=1))
    rec("chol", k.chol(S)[0])
    w, v = k.eigh(S)
    rec("eigh", w, v)
    s, vt = k.svd(A)
    rec("svd", s, vt)
    rec("orth", k.orth(A))
    rec("qr_r", k.qr_r(A))
    rec("absmax_sign", [1.0 if f else 0.0 for f in k.absmax_flags(A, True)])
    W = _m(7, 5, 5, 0.0, 1.0)
    H = _m(5, 5, 6, 0.0, 1.0)
    rec("cd_rows", [k.cd_rows(W, k.mm(H, H, tb=True), k.mm(_m(7, 5, 7, 0.0, 1.0), H, tb=True), list(range(5)))], W)
    G = k.mm(B, B, tb=True)
    Q = k.mm(A, B, tb=True)
    rec("lasso_rows", k.lasso_rows(G, Q, _M.zeros(7, 6), 0.1, 50, 1e-6, False))
    rec("omp_rows", k.omp_rows(G, Q, 2))
    rec("rand_gamma", k.rand_gamma(2, 5, 3, 1, 100.0))
    D = k.ew("sqrt", k.sqdist(A, A))
    rec("dijkstra_rows", k.dijkstra(D))
    rec("barycenter_rows", k.barycenter(A, A, [[1, 2], [0, 2], [0, 1], [2, 4], [3, 5], [4, 6], [4, 5]], 1e-3))
    C = _m(7, 6, 8, 0.0, 2.0)
    rec("als_rows", k.als(C, _m(6, 3, 9), 0.1))
    X = _m(4, 6, 10, 0.0, 3.0)
    EW = _m(3, 6, 11, 0.1, 1.0)
    rec("lda_rows", *k.lda_rows(X, EW, _m(4, 3, 12, 0.5, 1.5), _m(4, 3, 13, 0.5, 1.5), 0.1, 20, 1e-3)[:1])
    return out


def _kits():
    host = _Kit("identical", _backend.load_host_module("_mojolearn_x_decomp_host"))
    try:
        gpu = _Kit("identical")
    except Exception:  # a CPU-only install has no GPU binding
        gpu = None
    return host, gpu


def test_every_entry_twice_host_and_gpu():
    host, gpu = _kits()
    h1, h2 = _calls(host), _calls(host)
    assert h1 == h2, [n for (n, a), (_, b) in zip(h1, h2) if a != b]
    if gpu is None:
        pytest.skip("no GPU binding in this process")
    g1, g2 = _calls(gpu), _calls(gpu)
    assert g1 == g2, [n for (n, a), (_, b) in zip(g1, g2) if a != b]
    assert g1 == h1, [n for (n, a), (_, b) in zip(g1, h1) if a != b]
