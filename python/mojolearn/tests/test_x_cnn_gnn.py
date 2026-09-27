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


def ref_sage_max_norm(x, ei, Wl, bl, Wr, normalize, g):
    n, F = x.shape
    src, dst = ei
    x64 = x.astype(np.float64)
    agg = np.zeros((n, F))
    cnt = np.zeros((n, F))
    for t in range(n):
        m = src[dst == t]
        if len(m):
            vals = x64[m]
            agg[t] = vals.max(0)
            cnt[t] = (vals == agg[t]).sum(0)
    pre = agg @ Wl.T + bl + x64 @ Wr.T
    if normalize:
        den = np.maximum(np.linalg.norm(pre, axis=1, keepdims=True), 1e-12)
        y = pre / den
        G = (g - y * (g * y).sum(1, keepdims=True)) / den
    else:
        y, G = pre, g
    dagg = G @ Wl
    dx = G @ Wr
    for e in range(len(src)):
        s_, t = src[e], dst[e]
        hit = x64[s_] == agg[t]
        dx[s_] += np.where(hit, dagg[t] / np.maximum(cnt[t], 1), 0)
    return y, dx


@pytest.mark.parametrize("normalize", [False, True])
def test_sage_max_normalize(normalize):
    import mojolearn as ml
    rng = np.random.default_rng(6)
    n = 30
    x = rng.integers(-2, 3, size=(n, 5)).astype(np.float32)  # ties inside the max
    ei = _graph(n, 7)
    conv = ml.SAGEConv(5, 4, aggr="max", normalize=normalize, random_state=8)
    y = conv.forward(x, ei)
    g = rng.standard_normal(y.shape).astype(np.float32)
    dx = conv.backward(g)
    ry, rdx = ref_sage_max_norm(x, ei, conv.lin_l.weight_, conv.lin_l.bias_, conv.weight_r_, normalize,
                                g.astype(np.float64))
    np.testing.assert_allclose(y, ry, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(dx, rdx, rtol=1e-4, atol=1e-5)


def test_max_reference_matches_torch():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(6)
    n = 30
    # values kept away from 0.0: torch's amax backward (2.4) also counts the
    # zero-initialized output as a tied candidate when include_self=False, so a
    # maximum of exactly 0.0 gets half its gradient. That is not carried.
    x = rng.integers(1, 5, size=(n, 5)).astype(np.float64)
    ei = _graph(n, 7)
    tx = torch.tensor(x, requires_grad=True)
    src, dst = torch.tensor(ei[0]), torch.tensor(ei[1])
    out = torch.zeros(n, 5, dtype=torch.float64).scatter_reduce(0, dst[:, None].expand(-1, 5), tx[src], "amax",
                                                                include_self=False)
    g = rng.standard_normal((n, 5))
    out.backward(torch.tensor(g))
    Wl, Wr = np.eye(5), np.zeros((5, 5))
    ry, rdx = ref_sage_max_norm(x.astype(np.float32), ei, Wl, np.zeros(5), Wr, False, g)
    np.testing.assert_allclose(out.detach().numpy(), ry, atol=1e-12)
    np.testing.assert_allclose(tx.grad.numpy(), rdx, atol=1e-12)
    y = torch.tensor(rng.standard_normal((4, 6)), requires_grad=True)
    z = torch.nn.functional.normalize(y, p=2, dim=-1)
    g2 = rng.standard_normal((4, 6))
    z.backward(torch.tensor(g2))
    yy = y.detach().numpy()
    den = np.maximum(np.linalg.norm(yy, axis=1, keepdims=True), 1e-12)
    zz = yy / den
    np.testing.assert_allclose(z.detach().numpy(), zz, atol=1e-12)
    np.testing.assert_allclose(y.grad.numpy(), (g2 - zz * (g2 * zz).sum(1, keepdims=True)) / den, atol=1e-12)


@pytest.mark.parametrize("aggr", ["mean", "max"])
def test_sage_project(aggr):
    """project=True (PyG sage_conv.py): x_j -> relu(lin(x_j)) before the
    aggregation; the root term lin_r keeps the unprojected x_i."""
    import mojolearn as ml
    rng = np.random.default_rng(9)
    n = 30
    x = rng.standard_normal((n, 5)).astype(np.float32)
    ei = _graph(n, 10)
    conv = ml.SAGEConv(5, 4, aggr=aggr, project=True, random_state=11)
    y = conv.forward(x, ei)
    g = rng.standard_normal(y.shape).astype(np.float32)
    dx = conv.backward(g)
    Wp, bp = conv.lin.weight_.astype(np.float64), conv.lin.bias_.astype(np.float64)
    Wl, bl, Wr = conv.lin_l.weight_, conv.lin_l.bias_, conv.weight_r_
    x64, g64 = x.astype(np.float64), g.astype(np.float64)
    h = x64 @ Wp.T + bp
    p = np.maximum(h, 0)
    src, dst = ei
    if aggr == "mean":
        A = np.zeros((n, n))
        np.add.at(A, (dst, src), 1.0)
        deg = A.sum(1, keepdims=True)
        A = np.where(deg > 0, A / np.maximum(deg, 1), 0)
        agg = A @ p
        dp = A.T @ (g64 @ Wl)
    else:
        # the max reference with identity lin_l/lin_r gives agg and d(agg)->dp
        agg, _ = ref_sage_max_norm(p, ei, np.eye(5), np.zeros(5), np.zeros((5, 5)), False, np.zeros((n, 5)))
        _, dp = ref_sage_max_norm(p, ei, np.eye(5), np.zeros(5), np.zeros((5, 5)), False, g64 @ Wl)
    ry = agg @ Wl.T + bl + x64 @ Wr.T
    rdx = (dp * (h > 0)) @ Wp + g64 @ Wr
    np.testing.assert_allclose(y, ry, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(dx, rdx, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(conv.lin.grad_weight_, (dp * (h > 0)).T @ x64, rtol=1e-4, atol=1e-5)
