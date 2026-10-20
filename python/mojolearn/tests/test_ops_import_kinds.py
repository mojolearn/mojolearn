# SPDX-License-Identifier: Apache-2.0
"""The elementwise operations of imported models in mojolearn.ops.Session
(docs/plans/MODEL_IMPORT.md section 3.3): erf, gelu, softplus, silu, leaky
relu, elu, hard sigmoid and swish, clip, abs, floor, ceil, round, sin, cos,
sign, reciprocal, pow, minimum, maximum, where, the comparisons and logical
operations, expand, argmax and pad, with their gradients in a recording
session. Small, odd, non-board shapes. Values are checked against NumPy
(exactly where the operation rounds nothing, with a tolerance otherwise);
every case runs twice for the same digest and prints OPS_IMPORT_DIGEST
lines to diff across CPU, Metal, NVIDIA and AMD."""
import hashlib
import math

import numpy as np
import pytest

from mojolearn import _backend


def _has_import_kinds():
    try:
        b = _backend.binding("_mojolearn_x_sequence")
        if not hasattr(b, "ops_open"):
            return False
        from mojolearn import ops
        with ops.Session() as s:
            return s.shape(s.erf(s.tensor(np.zeros((2, 3), dtype=np.float32)))) == (2, 3)
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_import_kinds(),
                                reason="the sequence binding has no import kinds (rebuild it)")


def _ops():
    from mojolearn import ops
    return ops


def _digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a, dtype=np.float32).tobytes())
    return h.hexdigest()[:16]


def _twice(name, run):
    first = run()
    second = run()
    d = _digest(*first)
    assert d == _digest(*second), f"{name}: two runs differ"
    print(f"OPS_IMPORT_DIGEST {name} {d}")
    return first


def _x(shape, seed, low=-2.0, high=2.0):
    return np.random.default_rng(seed).uniform(low, high, size=shape).astype(np.float32)


_erf = np.vectorize(math.erf, otypes=[np.float64])


def _unary_reference(name, x):
    x = x.astype(np.float64)
    if name == "erf":
        return _erf(x)
    if name == "gelu":
        return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))
    if name == "gelu_tanh":
        return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))
    if name == "softplus":
        return np.log1p(np.exp(x))
    if name == "silu":
        return x / (1.0 + np.exp(-x))
    if name == "leaky_relu":
        return np.where(x >= 0, x, 0.1 * x)
    if name == "elu":
        return np.where(x >= 0, x, 1.3 * (np.exp(x) - 1.0))
    if name == "hard_sigmoid":
        return np.clip(0.2 * x + 0.5, 0.0, 1.0)
    if name == "hard_swish":
        return x * np.clip(x / 6.0 + 0.5, 0.0, 1.0)
    if name == "sin":
        return np.sin(x)
    if name == "cos":
        return np.cos(x)
    if name == "reciprocal":
        return 1.0 / x
    raise ValueError(name)


def _apply(s, name, h):
    if name == "gelu_tanh":
        return s.gelu(h, approximate="tanh")
    if name == "leaky_relu":
        return s.leaky_relu(h, alpha=0.1)
    if name == "elu":
        return s.elu(h, alpha=1.3)
    return getattr(s, name)(h)


_SMOOTH = ("erf", "gelu", "gelu_tanh", "softplus", "silu", "leaky_relu", "elu", "hard_sigmoid",
           "hard_swish", "sin", "cos", "reciprocal")


@pytest.mark.parametrize("name", _SMOOTH)
def test_smooth_unary_values(name):
    X = _x((3, 7, 5), 1)
    if name == "reciprocal":
        X = np.abs(X) + np.float32(0.25)

    def run():
        with _ops().Session() as s:
            y = s.numpy(_apply(s, name, s.tensor(X)))
        return (y,)

    (y,) = _twice(f"unary/{name}", run)
    assert y.shape == X.shape
    np.testing.assert_allclose(y, _unary_reference(name, X), rtol=2e-6, atol=2e-6)


def test_exact_unary_values():
    X = (np.arange(-23, 82, dtype=np.float32) * np.float32(0.25)).reshape(3, 5, 7)
    X[0, 0, 0] = np.float32(-0.0)

    def run():
        with _ops().Session() as s:
            x = s.tensor(X)
            outs = [s.numpy(f(x)) for f in (s.abs, s.floor, s.ceil, s.round, s.sign, s.logical_not)]
            outs.append(s.numpy(s.clip(x, -1.3, 2.1)))
            outs.append(s.numpy(s.clip(x, min=0.5)))
        return tuple(outs)

    a, f, c, r, sg, nt, cl, lo = _twice("unary/exact", run)
    np.testing.assert_array_equal(a, np.abs(X))
    np.testing.assert_array_equal(f, np.floor(X))
    np.testing.assert_array_equal(c, np.ceil(X))
    np.testing.assert_array_equal(r, np.round(X))
    np.testing.assert_array_equal(sg, np.sign(X))
    np.testing.assert_array_equal(nt, (X == 0).astype(np.float32))
    np.testing.assert_array_equal(cl, np.clip(X, np.float32(-1.3), np.float32(2.1)))
    np.testing.assert_array_equal(lo, np.maximum(X, np.float32(0.5)))


def test_pow_minimum_maximum_broadcast():
    A = _x((3, 4, 5), 2, 0.5, 2.0)
    B = _x((3, 1, 5), 3, -1.5, 1.5)
    N = -(np.arange(60, dtype=np.float32).reshape(3, 4, 5) % 7) * np.float32(0.25) - np.float32(0.5)
    E = (np.arange(60, dtype=np.float32).reshape(3, 4, 5) % 5) - np.float32(2)

    def run():
        with _ops().Session() as s:
            a, b = s.tensor(A), s.tensor(B)
            p = s.numpy(s.pow(a, b))
            ps = s.numpy(s.pow_scalar(a, 2.5))
            pn = s.numpy(s.pow(s.tensor(N), s.tensor(E)))
            mn = s.numpy(s.minimum(a, b))
            mx = s.numpy(s.maximum(a, b))
        return p, ps, pn, mn, mx

    p, ps, pn, mn, mx = _twice("binary/pow-min-max", run)
    np.testing.assert_allclose(p, np.power(A.astype(np.float64), B), rtol=3e-6)
    np.testing.assert_allclose(ps, np.power(A.astype(np.float64), 2.5), rtol=3e-6)
    np.testing.assert_allclose(pn, np.power(N.astype(np.float64), E), rtol=3e-6)
    np.testing.assert_array_equal(mn, np.minimum(A, B))
    np.testing.assert_array_equal(mx, np.maximum(A, B))


def test_comparisons_logical_and_where():
    A = (np.arange(60, dtype=np.float32).reshape(3, 4, 5) % 5) - np.float32(2)
    B = (np.arange(15, dtype=np.float32).reshape(3, 1, 5) % 3) - np.float32(1)
    C = (np.arange(15, dtype=np.float32).reshape(3, 1, 5) % 2)
    Y = _x((1, 4, 1), 4)

    def run():
        with _ops().Session() as s:
            a, b, c, y = s.tensor(A), s.tensor(B), s.tensor(C), s.tensor(Y)
            outs = [s.numpy(f(a, b)) for f in (s.equal, s.not_equal, s.greater, s.greater_equal,
                                               s.less, s.less_equal, s.logical_and, s.logical_or)]
            outs.append(s.numpy(s.where(c, a, y)))
        return tuple(outs)

    eq, ne, gt, ge, lt, le, la, lo, wh = _twice("binary/compare-where", run)
    f = np.float32
    np.testing.assert_array_equal(eq, (A == B).astype(f))
    np.testing.assert_array_equal(ne, (A != B).astype(f))
    np.testing.assert_array_equal(gt, (A > B).astype(f))
    np.testing.assert_array_equal(ge, (A >= B).astype(f))
    np.testing.assert_array_equal(lt, (A < B).astype(f))
    np.testing.assert_array_equal(le, (A <= B).astype(f))
    np.testing.assert_array_equal(la, ((A != 0) & (B != 0)).astype(f))
    np.testing.assert_array_equal(lo, ((A != 0) | (B != 0)).astype(f))
    np.testing.assert_array_equal(wh, np.where(C != 0, A, Y))


def test_expand_argmax_pad():
    X = _x((3, 1, 5), 5)
    T = np.round(_x((5, 7, 6), 6) * 2) / 2  # many ties
    T[1, 3, 2] = np.nan

    def run():
        with _ops().Session() as s:
            x, t = s.tensor(X), s.tensor(T)
            e = s.numpy(s.expand(x, (3, 4, 5)))
            ones = s.numpy(s.mul(s.full((3, 4, 5), 1.0), x))
            am = s.numpy(s.argmax(t, axis=1))
            ak = s.numpy(s.argmax(t, axis=-2, keepdims=True, int32=True))
            pd = s.numpy(s.pad(x, (1, 0, 2, 0, 3, -1), value=0.5))
        return e, ones, am, ak, pd

    e, ones, am, ak, pd = _twice("shape/expand-argmax-pad", run)
    np.testing.assert_array_equal(e, np.broadcast_to(X, (3, 4, 5)))
    assert e.tobytes() == ones.tobytes()
    want = np.argmax(T, axis=1)
    assert am.shape == (5, 6) and ak.shape == (5, 1, 6)
    np.testing.assert_array_equal(am, want.astype(np.float32))
    np.testing.assert_array_equal(ak.view(np.int32)[:, 0, :], want)
    ref = np.pad(X, ((1, 0), (0, 3), (2, 0)), constant_values=0.5)[:, :, :6]
    np.testing.assert_array_equal(pd, ref)


def _numeric_grad(f, X, eps=1e-3):
    g = np.zeros(X.shape, dtype=np.float64)
    flat = X.reshape(-1)
    for i in range(flat.size):
        old = flat[i]
        flat[i] = old + eps
        hi = f(X)
        flat[i] = old - eps
        lo = f(X)
        flat[i] = old
        g.reshape(-1)[i] = (hi - lo) / (2 * eps)
    return g


@pytest.mark.parametrize("name", ("erf", "gelu", "gelu_tanh", "softplus", "silu", "elu", "sin", "cos"))
def test_unary_gradients(name):
    X = _x((2, 3, 5), 7, -1.5, 1.5)

    def run():
        with _ops().Session(record=True) as s:
            x = s.tensor(X, requires_grad=True)
            s.backward(s.sum(_apply(s, name, x)))
            return (s.grad(x),)

    (g,) = _twice(f"grad/{name}", run)
    want = _numeric_grad(lambda v: float(np.sum(_unary_reference(name, v))), X.astype(np.float64))
    np.testing.assert_allclose(g, want, rtol=1e-4, atol=1e-5)


def test_broadcast_gradients():
    A = _x((3, 4, 5), 8, 0.5, 2.0)
    B = _x((3, 1, 5), 9, -1.0, 1.0)
    C = (np.arange(15, dtype=np.float32).reshape(3, 1, 5) % 2)
    X = _x((3, 1, 5), 10)

    def run():
        with _ops().Session(record=True) as s:
            a = s.tensor(A, requires_grad=True)
            b = s.tensor(B, requires_grad=True)
            c = s.tensor(C)
            x = s.tensor(X, requires_grad=True)
            loss = s.add(s.sum(s.pow(a, b)), s.sum(s.where(c, a, b)))
            loss = s.add(loss, s.sum(s.mul(s.expand(x, (3, 4, 5)), s.tensor(A))))
            loss = s.add(loss, s.sum(s.pad(x, (0, 1, 0, 2, 0, 0), value=3.0)))
            s.backward(loss)
            return s.grad(a), s.grad(b), s.grad(x)

    ga, gb, gx = _twice("grad/broadcast", run)
    A64, B64 = A.astype(np.float64), B.astype(np.float64)
    take = np.broadcast_to(C != 0, A.shape)
    want_a = B64 * np.power(A64, B64 - 1) + take
    want_b = np.sum(np.power(A64, B64) * np.log(A64) + (~take), axis=1, keepdims=True)
    want_x = np.sum(A64, axis=1, keepdims=True) + 1.0
    np.testing.assert_allclose(ga, want_a, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(gb, want_b, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(gx, want_x, rtol=1e-5, atol=1e-5)


def test_refusals():
    with _ops().Session() as s:
        x = s.tensor(np.zeros((2, 3), dtype=np.float32))
        y = s.tensor(np.zeros((4, 3), dtype=np.float32))
        with pytest.raises(Exception):
            s.maximum(x, y)
        with pytest.raises(ValueError):
            s.gelu(x, approximate="sigmoid")
        with pytest.raises(Exception):
            s.pad(x, (1, 1))
        with pytest.raises(Exception):
            s.clip(x, 1.0, -1.0)
