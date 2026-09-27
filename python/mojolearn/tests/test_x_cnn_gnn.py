# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Correctness sanity of GCNConv and SAGEConv against float64 NumPy
references of PyG's gcn_conv.py / sage_conv.py semantics (PyG is not on the
dev pod; the references restate its gcn_norm, add_remaining_self_loops and
mean aggregation)."""
import numpy as np
import pytest


def _graph(n, seed):
    rng = np.random.default_rng(seed)
    e = rng.integers(0, n, size=(2, 3 * n))
    e = np.concatenate([e, [[1, 2, 2], [1, 3, 3]]], axis=1)  # a self loop and a duplicate
    return e[:, e[1] != 0]  # node 0 has no incoming edge


def ref_gcn(x, ei, w, W, b, improved, g):
    n = x.shape[0]
    src, dst = ei
    w = np.ones(len(src)) if w is None else w.astype(np.float64)
    loop = src == dst
    lw = np.full(n, 2.0 if improved else 1.0)
    lw[src[loop]] = w[loop]
    src = np.concatenate([src[~loop], np.arange(n)])
    dst = np.concatenate([dst[~loop], np.arange(n)])
    w = np.concatenate([w[~loop], lw])
    deg = np.zeros(n)
    np.add.at(deg, dst, w)
    dis = np.where(deg > 0, deg ** -0.5, 0)
    norm = dis[src] * w * dis[dst]
    A = np.zeros((n, n))
    np.add.at(A, (dst, src), norm)
    h = x.astype(np.float64) @ W.T.astype(np.float64)
    out = A @ h + b
    dh = A.T @ g
    return out, dh @ W.astype(np.float64), dh.T @ x.astype(np.float64), g.sum(0)


def ref_sage(x, ei, Wl, bl, Wr, aggr, g):
    n = x.shape[0]
    src, dst = ei
    A = np.zeros((n, n))
    np.add.at(A, (dst, src), 1.0)
    if aggr == "mean":
        deg = A.sum(1, keepdims=True)
        A = np.where(deg > 0, A / np.maximum(deg, 1), 0)
    x64 = x.astype(np.float64)
    agg = A @ x64
    out = agg @ Wl.T + bl + x64 @ Wr.T
    dx = (A.T @ (g @ Wl)) + g @ Wr
    return out, dx, g.T @ agg, g.sum(0), g.T @ x64


@pytest.mark.parametrize("improved,weighted", [(False, False), (True, True)])
def test_gcn(improved, weighted):
    import mojolearn as ml
    rng = np.random.default_rng(0)
    n = 40
    x = rng.standard_normal((n, 6)).astype(np.float32)
    ei = _graph(n, 1)
    w = (rng.random(ei.shape[1]) + 0.5).astype(np.float32) if weighted else None
    conv = ml.GCNConv(6, 4, improved=improved, random_state=2)
    conv.bias_ = rng.standard_normal(4).astype(np.float32)
    y = conv.forward(x, ei, w)
    g = rng.standard_normal(y.shape).astype(np.float32)
    dx = conv.backward(g)
    ry, rdx, rdw, rdb = ref_gcn(x, ei, w, conv.weight_, conv.bias_, improved, g.astype(np.float64))
    np.testing.assert_allclose(y, ry, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(dx, rdx, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(conv.grad_weight_, rdw, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(conv.grad_bias_, rdb, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("aggr", ["mean", "sum"])
def test_sage(aggr):
    import mojolearn as ml
    rng = np.random.default_rng(3)
    n = 40
    x = rng.standard_normal((n, 6)).astype(np.float32)
    ei = _graph(n, 4)
    conv = ml.SAGEConv(6, 5, aggr=aggr, random_state=5)
    y = conv.forward(x, ei)
    g = rng.standard_normal(y.shape).astype(np.float32)
    dx = conv.backward(g)
    ry, rdx, rdwl, rdbl, rdwr = ref_sage(x, ei, conv.lin_l.weight_, conv.lin_l.bias_, conv.weight_r_, aggr,
                                         g.astype(np.float64))
    np.testing.assert_allclose(y, ry, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(dx, rdx, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(conv.lin_l.grad_weight_, rdwl, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(conv.lin_l.grad_bias_, rdbl, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(conv.grad_weight_r_, rdwr, rtol=1e-5, atol=1e-5)
