# SPDX-License-Identifier: Apache-2.0
"""Rank-N tensors in mojolearn.ops.Session (docs/plans/AUTOGRAD_RANKN.md):
views (reshape, flatten, squeeze, unsqueeze, permute, transpose), sums,
means and max over axis sets, NumPy broadcasting, the batched product,
concat / split / take, softmax along an axis, conv2d and the 2-D pools, and
their gradients in a recording session. Small, odd, non-board shapes. Every
case checks values against NumPy (exactly for copies and single roundings,
with a tolerance for sums), runs twice for the same digest, and prints
RANKN_DIGEST lines to diff across CPU, Metal, NVIDIA and AMD."""
import hashlib

import numpy as np
import pytest

from mojolearn import _backend


def _has_rankn():
    try:
        b = _backend.binding("_mojolearn_x_sequence")
        if not hasattr(b, "ops_open"):
            return False
        from mojolearn import ops
        with ops.Session() as s:
            return s.shape(s.tensor(np.zeros((2, 1, 3), dtype=np.float32))) == (2, 1, 3)
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_rankn(),
                                reason="the sequence binding has no rank-N tensors (rebuild it)")


def _ops():
    from mojolearn import ops
    return ops


def _digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a, dtype=np.float32).tobytes())
    return h.hexdigest()[:16]


def _emit(name, digest):
    print(f"RANKN_DIGEST {name} {digest}")


def _twice(name, run):
    """Run `run` twice; the results must have the same digest. Emits it."""
    first = run()
    second = run()
    d = _digest(*first)
    assert d == _digest(*second), f"{name}: two runs differ"
    _emit(name, d)
    return first


def _rng(seed):
    return np.random.default_rng(seed)


# ---------------------------------------------------------------- views
def _views_run():
    X = np.arange(105, dtype=np.float32).reshape(3, 5, 7) * 0.25 - 4.0
    with _ops().Session() as s:
        x = s.tensor(X)
        assert s.shape(x) == (3, 5, 7)
        r = s.reshape(x, (-1, 7))
        f = s.flatten(x)
        u = s.unsqueeze(x, 1)
        q = s.squeeze(u)
        p = s.permute(x, (2, 0, 1))
        t = s.transpose(x, 0, 2)
        tm = s.transpose(x)
        assert s.shape(r) == (15, 7) and s.shape(f) == (3, 35) and s.shape(u) == (3, 1, 5, 7)
        assert s.shape(q) == (3, 5, 7) and s.shape(p) == (7, 3, 5) and s.shape(t) == (7, 5, 3)
        assert s.shape(tm) == (3, 7, 5)
        # a reshape of a permuted tensor reads the permuted order
        pr = s.reshape(p, (21, 5))
        out = [s.numpy(v) for v in (r, f, u, q, p, t, tm, pr)]
    return out


def test_views():
    X = np.arange(105, dtype=np.float32).reshape(3, 5, 7) * 0.25 - 4.0
    r, f, u, q, p, t, tm, pr = _twice("views", _views_run)
    assert np.array_equal(r, X.reshape(15, 7)) and np.array_equal(f, X.reshape(3, 35))
    assert np.array_equal(u, X[:, None]) and np.array_equal(q, X)
    assert np.array_equal(p, X.transpose(2, 0, 1)) and np.array_equal(t, X.transpose(2, 1, 0))
    assert np.array_equal(tm, X.transpose(0, 2, 1))
    assert np.array_equal(pr, X.transpose(2, 0, 1).reshape(21, 5))


def test_matrices_keep_their_shapes():
    X = _rng(1).standard_normal((6, 4)).astype(np.float32)
    with _ops().Session() as s:
        x = s.tensor(X)
        assert s.shape(s.sum(x)) == (1, 1)
        assert s.shape(s.sum(x, axis=0)) == (1, 4) and s.shape(s.mean(x, axis=1)) == (6, 1)
        assert s.shape(s.sum(x, axis=0, keepdims=False)) == (4,)
        assert s.shape(s.transpose(x)) == (4, 6)
        assert s.shape(s.tensor(np.arange(5))) == (5, 1)
        assert s.shape(s.full((2, 3), 1.5)) == (2, 3) and s.shape(s.full((2, 3, 4), 1.5)) == (2, 3, 4)


# ---------------------------------------------------------------- reductions
def _reduce_run():
    X = _rng(2).standard_normal((4, 6, 5)).astype(np.float32)
    X[1, 2, :] = 3.0  # ties along the last axis
    X[2, :, 3] = 7.0  # ties along the middle axis
    with _ops().Session() as s:
        x = s.tensor(X)
        outs = [s.sum(x, axis=1), s.sum(x, axis=(0, 2)), s.sum(x, axis=-1, keepdims=True),
                s.mean(x, axis=(1, 2)), s.mean(x), s.max(x, axis=1), s.max(x, axis=2, keepdims=True),
                s.max(x, axis=(0, 1))]
        shapes = [s.shape(o) for o in outs]
        assert shapes == [(4, 5), (6,), (4, 6, 1), (4,), (1, 1), (4, 5), (4, 6, 1), (5,)]
        return [s.numpy(o) for o in outs]


def test_reductions():
    X = _rng(2).standard_normal((4, 6, 5)).astype(np.float32)
    X[1, 2, :] = 3.0
    X[2, :, 3] = 7.0
    got = _twice("reductions", _reduce_run)
    want = [X.sum(1), X.sum((0, 2)), X.sum(-1, keepdims=True), X.mean((1, 2)), np.full((1, 1), X.mean())]
    for g, w in zip(got[:5], want):
        assert np.allclose(g, w, rtol=1e-5, atol=1e-5)
    assert np.array_equal(got[5], X.max(1)) and np.array_equal(got[6], X.max(2, keepdims=True))
    assert np.array_equal(got[7], X.max((0, 1)))


def test_max_nan_wins_and_gradient_takes_the_first_maximum():
    X = np.array([[[1.0, 4.0, 4.0, 2.0], [np.nan, 0.0, 5.0, 5.0]]], dtype=np.float32)
    with _ops().Session(record=True) as s:
        x = s.tensor(X, requires_grad=True)
        m = s.max(x, axis=2)
        s.backward(s.sum(m))
        values, grad = s.numpy(m), s.grad(x)
    assert values[0, 0] == 4.0 and np.isnan(values[0, 1])
    want = np.zeros_like(X)
    want[0, 0, 1] = 1.0
    want[0, 1, 0] = 1.0
    assert np.array_equal(grad, want)


# ---------------------------------------------------------------- broadcasting
def _bcast_run():
    rng = _rng(3)
    A = rng.standard_normal((2, 3, 4)).astype(np.float32)
    B = rng.standard_normal((1, 3, 1)).astype(np.float32)
    C = rng.standard_normal((4,)).astype(np.float32).reshape(1, 4)
    D = (rng.random((2, 1, 4)) + 0.5).astype(np.float32)
    with _ops().Session(record=True) as s:
        a = s.tensor(A, requires_grad=True)
        b = s.tensor(B, requires_grad=True)
        c = s.tensor(C, requires_grad=True)
        d = s.tensor(D, requires_grad=True)
        y = s.div(s.mul(s.add(a, b), c), d)
        s.backward(s.sum(y))
        return [s.numpy(y), s.grad(a), s.grad(b), s.grad(c), s.grad(d)]


def test_broadcasting_and_its_gradients():
    rng = _rng(3)
    A = rng.standard_normal((2, 3, 4)).astype(np.float32)
    B = rng.standard_normal((1, 3, 1)).astype(np.float32)
    C = rng.standard_normal((4,)).astype(np.float32).reshape(1, 4)
    D = (rng.random((2, 1, 4)) + 0.5).astype(np.float32)
    y, ga, gb, gc, gd = _twice("broadcast", _bcast_run)
    assert np.array_equal(y, ((A + B) * C) / D)
    assert np.allclose(ga, np.broadcast_to(C / D, A.shape), rtol=1e-6)
    assert np.allclose(gb, (C / D * np.ones_like(A)).sum((0, 2), keepdims=True), rtol=1e-5, atol=1e-5)
    assert np.allclose(gc, ((A + B) / D).sum((0, 1)).reshape(1, 4), rtol=1e-5, atol=1e-5)
    assert np.allclose(gd, (-(A + B) * C / D ** 2).sum(1, keepdims=True), rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------- batched product
def _bmm_run():
    rng = _rng(4)
    Q = rng.standard_normal((2, 3, 5, 4)).astype(np.float32)
    K = rng.standard_normal((2, 3, 7, 4)).astype(np.float32)
    W = rng.standard_normal((4, 6)).astype(np.float32)
    with _ops().Session(record=True) as s:
        q = s.tensor(Q, requires_grad=True)
        k = s.tensor(K, requires_grad=True)
        w = s.tensor(W, requires_grad=True)
        scores = s.matmul(q, s.transpose(k))
        att = s.softmax(s.scale(scores, 0.5))
        out = s.matmul(s.matmul(att, k), w)
        assert s.shape(scores) == (2, 3, 5, 7) and s.shape(out) == (2, 3, 5, 6)
        s.backward(s.mean(out))
        return [s.numpy(scores), s.numpy(out), s.grad(q), s.grad(k), s.grad(w)]


def test_batched_matmul_and_attention():
    rng = _rng(4)
    Q = rng.standard_normal((2, 3, 5, 4)).astype(np.float32)
    K = rng.standard_normal((2, 3, 7, 4)).astype(np.float32)
    W = rng.standard_normal((4, 6)).astype(np.float32)
    scores, out, gq, gk, gw = _twice("bmm_attention", _bmm_run)
    S = np.einsum("bhqd,bhkd->bhqk", Q.astype(np.float64), K.astype(np.float64))
    assert np.allclose(scores, S, rtol=1e-4, atol=1e-4)
    E = np.exp(0.5 * S - (0.5 * S).max(-1, keepdims=True))
    P = E / E.sum(-1, keepdims=True)
    O = np.einsum("bhqk,bhkd->bhqd", P, K) @ W
    assert np.allclose(out, O, rtol=1e-4, atol=1e-4)
    for g, shape in ((gq, Q.shape), (gk, K.shape), (gw, W.shape)):
        assert g.shape == shape and np.isfinite(g).all()
    gO = np.full(O.shape, 1.0 / O.size)
    assert np.allclose(gw, np.einsum("bhqd,bhqo->do", np.einsum("bhqk,bhkd->bhqd", P, K), gO),
                       rtol=1e-4, atol=1e-6)


# ---------------------------------------------------------------- layout
def _layout_run():
    rng = _rng(5)
    parts = [rng.standard_normal((3, k, 5)).astype(np.float32) for k in (2, 1, 4)]
    T = rng.standard_normal((3, 7, 5)).astype(np.float32)
    with _ops().Session(record=True) as s:
        xs = [s.tensor(p, requires_grad=True) for p in parts]
        cat = s.concat(xs, axis=1)
        a, b, c = s.split(cat, [2, 1, 4], axis=1)
        halves = s.split(s.tensor(T), 1, axis=0)
        t = s.tensor(T, requires_grad=True)
        picked = s.take(t, np.array([6, 0, 6, 3]), axis=1)
        assert s.shape(cat) == (3, 7, 5) and s.shape(picked) == (3, 4, 5) and len(halves) == 1
        weights = s.tensor(np.arange(60, dtype=np.float32).reshape(3, 4, 5))
        loss = s.add(s.sum(s.mul(a, a)), s.sum(s.mul(picked, weights)))
        loss = s.add(loss, s.sum(s.scale(c, 3.0)))
        s.backward(loss)
        return [s.numpy(cat), s.numpy(a), s.numpy(b), s.numpy(c), s.numpy(picked)] + \
            [s.grad(x) for x in xs] + [s.grad(t)]


def test_concat_split_take():
    rng = _rng(5)
    parts = [rng.standard_normal((3, k, 5)).astype(np.float32) for k in (2, 1, 4)]
    T = rng.standard_normal((3, 7, 5)).astype(np.float32)
    out = _twice("layout", _layout_run)
    cat, a, b, c, picked, g0, g1, g2, gt = out
    assert np.array_equal(cat, np.concatenate(parts, axis=1))
    for got, want in zip((a, b, c), parts):
        assert np.array_equal(got, want)
    assert np.array_equal(picked, T[:, [6, 0, 6, 3]])
    assert np.array_equal(g0, 2 * parts[0]) and np.array_equal(g1, np.zeros_like(parts[1]))
    assert np.array_equal(g2, np.full_like(parts[2], 3.0))
    W = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
    want = np.zeros_like(T)
    for j, idx in enumerate((6, 0, 6, 3)):  # the reference scatter-add, in the test only
        want[:, idx] += W[:, j]
    assert np.array_equal(gt, want)


# ---------------------------------------------------------------- softmax along an axis
def _softmax_run():
    X = _rng(6).standard_normal((4, 6, 5)).astype(np.float32)
    with _ops().Session() as s:
        x = s.tensor(X)
        return [s.numpy(s.softmax(x, axis=1)), s.numpy(s.log_softmax(x, axis=0)), s.numpy(s.softmax(x))]


def test_softmax_along_an_axis():
    X = _rng(6).standard_normal((4, 6, 5)).astype(np.float64)
    s1, l0, s2 = _twice("softmax_axis", _softmax_run)

    def soft(v, axis):
        e = np.exp(v - v.max(axis, keepdims=True))
        return e / e.sum(axis, keepdims=True)
    assert np.allclose(s1, soft(X, 1), rtol=1e-5, atol=1e-6)
    assert np.allclose(l0, np.log(soft(X, 0)), rtol=1e-5, atol=1e-5)
    assert np.allclose(s2, soft(X, 2), rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------- conv2d and pools
def _conv_reference(X, W, b, stride, pad):
    """Direct cross-correlation in float64 (the test's own oracle)."""
    N, C, H, Wd = X.shape
    OC, _, KH, KW = W.shape
    Xp = np.pad(X.astype(np.float64), ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    OH = (H + 2 * pad - KH) // stride + 1
    OW = (Wd + 2 * pad - KW) // stride + 1
    out = np.zeros((N, OC, OH, OW))
    for i in range(OH):
        for j in range(OW):
            win = Xp[:, :, i * stride:i * stride + KH, j * stride:j * stride + KW]
            out[:, :, i, j] = np.einsum("nckl,ockl->no", win, W)
    return out + b.reshape(1, OC, 1, 1)


def _pool_reference(X, k, stride, kind):
    N, C, H, W = X.shape
    OH, OW = (H - k) // stride + 1, (W - k) // stride + 1
    out = np.zeros((N, C, OH, OW), dtype=np.float64)
    for i in range(OH):
        for j in range(OW):
            win = X[:, :, i * stride:i * stride + k, j * stride:j * stride + k].astype(np.float64)
            out[:, :, i, j] = win.max((2, 3)) if kind == "max" else win.mean((2, 3))
    return out


def _conv_inputs():
    rng = _rng(7)
    X = rng.standard_normal((2, 3, 9, 11)).astype(np.float32)
    W = (0.3 * rng.standard_normal((4, 3, 3, 3))).astype(np.float32)
    b = rng.standard_normal((4,)).astype(np.float32)
    return X, W, b


def _conv_run():
    X, W, b = _conv_inputs()
    with _ops().Session(record=True) as s:
        x = s.tensor(X, requires_grad=True)
        w = s.tensor(W, requires_grad=True)
        bb = s.tensor(b.reshape(1, 4), requires_grad=True)
        y1 = s.conv2d(x, w, bb, stride=1, padding=1)
        y2 = s.conv2d(x, w, stride=2, padding=0)
        mp = s.max_pool2d(y1, 2)
        ap = s.avg_pool2d(x, 3, stride=2)
        assert s.shape(y1) == (2, 4, 9, 11) and s.shape(y2) == (2, 4, 4, 5)
        assert s.shape(mp) == (2, 4, 4, 5) and s.shape(ap) == (2, 3, 4, 5)
        loss = s.add(s.mean(s.mul(mp, mp)), s.mean(ap))
        loss = s.add(loss, s.mean(y2))
        s.backward(loss)
        return [s.numpy(y1), s.numpy(y2), s.numpy(mp), s.numpy(ap), s.grad(x), s.grad(w), s.grad(bb)]


def test_conv2d_and_pools():
    X, W, b = _conv_inputs()
    y1, y2, mp, ap, gx, gw, gb = _twice("conv2d_pools", _conv_run)
    assert np.allclose(y1, _conv_reference(X, W, b, 1, 1), rtol=1e-4, atol=1e-4)
    assert np.allclose(y2, _conv_reference(X, W, np.zeros(4), 2, 0), rtol=1e-4, atol=1e-4)
    assert np.array_equal(mp, _pool_reference(y1, 2, 2, "max").astype(np.float32))
    assert np.allclose(ap, _pool_reference(X, 3, 2, "avg"), rtol=1e-5, atol=1e-6)
    assert gx.shape == X.shape and gw.shape == W.shape and gb.shape == (1, 4)
    assert np.isfinite(gx).all() and np.isfinite(gw).all() and np.isfinite(gb).all()


def test_conv2d_gradient_by_finite_differences():
    rng = _rng(8)
    X = rng.standard_normal((1, 2, 5, 6)).astype(np.float32)
    W = (0.5 * rng.standard_normal((3, 2, 3, 3))).astype(np.float32)
    G = rng.standard_normal((1, 3, 5, 6))

    def loss_of(Wv):
        return float((_conv_reference(X, Wv, np.zeros(3), 1, 1) * G).sum())
    with _ops().Session(record=True) as s:
        w = s.tensor(W, requires_grad=True)
        y = s.conv2d(s.tensor(X), w, padding=1)
        s.backward(s.sum(s.mul(y, s.tensor(G.astype(np.float32)))))
        gw = s.grad(w)
    eps = 1e-3
    for idx in ((0, 0, 0, 0), (1, 1, 2, 1), (2, 0, 1, 2)):  # three weights, central differences
        Wp, Wm = W.astype(np.float64), W.astype(np.float64)
        Wp[idx] += eps
        Wm[idx] -= eps
        fd = (loss_of(Wp) - loss_of(Wm)) / (2 * eps)
        assert abs(gw[idx] - fd) <= 1e-2 * max(1.0, abs(fd))


# ---------------------------------------------------------------- refusals
def test_refusals():
    ops = _ops()
    with ops.Session() as s:
        with pytest.raises(ValueError):
            s.tensor(np.zeros((1,) * 9, dtype=np.float32))
        with pytest.raises(ValueError, match="C-contiguous"):
            s.tensor(np.zeros((4, 3, 2), dtype=np.float32).transpose(2, 1, 0))
        x = s.tensor(np.zeros((2, 3, 4), dtype=np.float32))
        with pytest.raises(Exception):
            s.permute(x, (0, 0, 1))
        with pytest.raises(Exception):
            s.reshape(x, (5, -1))
        with pytest.raises(Exception):
            s.split(x, [1, 1], axis=1)
        with pytest.raises(Exception):
            s.split(x, 2, axis=1)
        with pytest.raises(Exception):
            s.take(x, np.array([0, 3]), axis=1)
        with pytest.raises(Exception):
            s.squeeze(x, 0)
        with pytest.raises(Exception):
            s.sum(x, axis=(1, 1))
        with pytest.raises(Exception):
            s.conv2d(x, x)
