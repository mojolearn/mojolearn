# SPDX-License-Identifier: Apache-2.0
"""Higher-order gradients through mojolearn.ops.Session: backward with
create_graph=True, grad_tensor, hvp, vjp and jvp. Tiny data. Hessian-vector
products are compared with float64 NumPy nested central differences (the
oracle is test code; every computed tensor comes from the session), jvp with
vjp by the adjoint identity <v, J u> = <J^T v, u>, a one-input tanh
network's d2u/dx2 with its closed form, and the refusals' errors. Every case
prints OPS2_DIGEST lines to diff across CPU, Metal, NVIDIA and AMD."""
import hashlib

import numpy as np
import pytest

from mojolearn import _backend


def _has_higher_entries():
    try:
        b = _backend.binding("_mojolearn_x_sequence")
        return all(hasattr(b, name) for name in ("ops_backward_graph", "ops_grad_handle", "ops_hvp", "ops_jvp"))
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_higher_entries(),
                                reason="the sequence binding has no higher-order entries (rebuild it)")


def _ops():
    from mojolearn import ops
    return ops


def _digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a, dtype=np.float32).tobytes())
    return h.hexdigest()[:16]


def _emit(name, digest):
    print(f"OPS2_DIGEST {name} {digest}")


def _f64(a):
    return np.asarray(a, dtype=np.float32).astype(np.float64)


def _close(actual, expected, what):
    actual = np.asarray(actual, dtype=np.float64)
    tol = 3e-4 + 3e-4 * np.abs(expected)
    assert actual.shape == expected.shape, what
    assert np.isfinite(actual).all(), what
    assert (np.abs(actual - expected) <= tol).all(), f"{what}: max error {np.abs(actual - expected).max():.3g}"


def _hvp_oracle(fn, point, v, h=1e-3):
    """H v of the scalar fn at point (name -> array) along v (name -> array):
    nested central differences, Richardson-extrapolated from h and h/2."""
    names = list(v)

    def at(step):
        out = {}
        for q in names:
            r = np.zeros_like(point[q])
            for idx in np.ndindex(point[q].shape):
                def f(sv, se):
                    d = {k: point[k] + (sv * step * v[k] if k in v else 0) for k in point}
                    d[q] = d[q].copy()
                    d[q][idx] += se * step
                    return fn(d)
                r[idx] = (f(1, 1) - f(1, -1) - f(-1, 1) + f(-1, -1)) / (4 * step * step)
            out[q] = r
        return out
    full, half = at(h), at(h / 2)
    return {q: (4 * half[q] - full[q]) / 3 for q in names}


def _jacobian(fn, point, names, h=1e-6):
    out = np.asarray(fn(point))
    jac = {}
    for k in names:
        j = np.zeros((out.size, point[k].size))
        for n, idx in enumerate(np.ndindex(point[k].shape)):
            plus = {q: a.copy() for q, a in point.items()}
            minus = {q: a.copy() for q, a in point.items()}
            plus[k][idx] += h
            minus[k][idx] -= h
            j[:, n] = ((np.asarray(fn(plus)) - np.asarray(fn(minus))) / (2 * h)).ravel()
        jac[k] = j
    return jac


# ---------------------------------------------------------------- the guide's example

def _example_inputs():
    rng = np.random.default_rng(5)
    X = (0.5 * rng.standard_normal((8, 3))).astype(np.float32)
    W0 = (0.5 * rng.standard_normal((3, 2))).astype(np.float32)
    Y = (0.5 * rng.standard_normal((8, 2))).astype(np.float32)
    V = rng.standard_normal((3, 2)).astype(np.float32)
    return X, W0, Y, V


def _example_run():
    X, W0, Y, V = _example_inputs()
    ops = _ops()
    with ops.Session(record=True) as s:
        x = s.tensor(X)
        w = s.tensor(W0, requires_grad=True)
        loss = s.mse(s.linear(x, w, activation="tanh"), s.tensor(Y))

        s.backward(loss, create_graph=True)
        g = s.grad_tensor(w)
        assert s.shape(g) == W0.shape
        first = s.numpy(g)
        v = s.tensor(V)
        s.backward(s.sum(s.mul(g, v)))
        hv = s.grad(w)

        hv2 = s.numpy(s.hvp(loss, [w], [v])[0])
        return first, hv, hv2


def test_example_hvp():
    X, W0, Y, V = _example_inputs()
    point = {"w": _f64(W0)}

    def fn(d):
        return float(np.mean((np.tanh(_f64(X) @ d["w"]) - _f64(Y)) ** 2))
    oracle = _hvp_oracle(fn, point, {"w": _f64(V)})["w"]
    z = np.tanh(_f64(X) @ _f64(W0))
    gradient = _f64(X).T @ ((2.0 / Y.size) * (z - _f64(Y)) * (1 - z * z))
    first, hv, hv2 = _example_run()
    _close(first, gradient, "first-order gradient under create_graph")
    _close(hv, oracle, "H v by two backward calls")
    _close(hv2, oracle, "H v by hvp")
    again = _example_run()
    assert _digest(first, hv, hv2) == _digest(*again)
    _emit("example_first", _digest(first))
    _emit("example_hvp", _digest(hv))
    _emit("example_hvp_helper", _digest(hv2))


# ---------------------------------------------------------------- hvp over several parameters

def _two_param_inputs():
    rng = np.random.default_rng(6)
    X = (0.6 * rng.standard_normal((6, 3))).astype(np.float32)
    W = (0.6 * rng.standard_normal((3, 4))).astype(np.float32)
    B = (0.3 * rng.standard_normal((1, 4))).astype(np.float32)
    C = rng.integers(0, 4, size=(6, 1)).astype(np.float32)
    VW = rng.standard_normal((3, 4)).astype(np.float32)
    VB = rng.standard_normal((1, 4)).astype(np.float32)
    return X, W, B, C, VW, VB


def _two_param_run():
    X, W, B, C, VW, VB = _two_param_inputs()
    with _ops().Session(record=True) as s:
        w = s.tensor(W, requires_grad=True)
        b = s.tensor(B, requires_grad=True)
        loss = s.cross_entropy(s.linear(s.tensor(X), w, b), s.tensor(C))
        hw, hb = s.hvp(loss, [w, b], [s.tensor(VW), s.tensor(VB)])
        assert s.shape(hw) == W.shape and s.shape(hb) == B.shape
        return s.numpy(hw), s.numpy(hb)


def test_hvp_softmax_cross_entropy():
    X, W, B, C, VW, VB = _two_param_inputs()
    codes = C.ravel().astype(int)

    def fn(d):
        z = _f64(X) @ d["w"] + d["b"]
        z = z - z.max(axis=1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return float(-np.mean(logp[np.arange(len(codes)), codes]))
    oracle = _hvp_oracle(fn, {"w": _f64(W), "b": _f64(B)}, {"w": _f64(VW), "b": _f64(VB)})
    hw, hb = _two_param_run()
    _close(hw, oracle["w"], "softmax cross-entropy H v (weights)")
    _close(hb, oracle["b"], "softmax cross-entropy H v (bias)")
    assert _digest(hw, hb) == _digest(*_two_param_run())
    _emit("hvp_softmax_cross_entropy", _digest(hw, hb))


def test_hvp_zero_curvature():
    rng = np.random.default_rng(7)
    X = rng.standard_normal((4, 3)).astype(np.float32)
    W = rng.standard_normal((3, 2)).astype(np.float32)
    with _ops().Session(record=True) as s:
        w = s.tensor(W, requires_grad=True)
        loss = s.sum(s.matmul(s.tensor(X), w))
        (h,) = s.hvp(loss, [w], [s.tensor(np.ones_like(W))])
        hv = s.numpy(h)
    assert hv.shape == W.shape and not hv.any()
    _emit("hvp_zero_curvature", _digest(hv))


# ---------------------------------------------------------------- vjp and jvp

def _adjoint_inputs():
    rng = np.random.default_rng(8)
    X = (0.6 * rng.standard_normal((3, 2))).astype(np.float32)
    W1 = (0.6 * rng.standard_normal((2, 3))).astype(np.float32)
    W2 = (0.6 * rng.standard_normal((3, 2))).astype(np.float32)
    V = rng.standard_normal((3, 2)).astype(np.float32)
    UX = rng.standard_normal((3, 2)).astype(np.float32)
    UW = rng.standard_normal((2, 3)).astype(np.float32)
    return X, W1, W2, V, UX, UW


def _adjoint_run():
    X, W1, W2, V, UX, UW = _adjoint_inputs()
    with _ops().Session(record=True) as s:
        x = s.tensor(X, requires_grad=True)
        w1 = s.tensor(W1, requires_grad=True)
        out = s.tanh(s.matmul(s.tanh(s.matmul(x, w1)), s.tensor(W2)))
        jtv_x, jtv_w = s.vjp(out, [x, w1], s.tensor(V))
        ju = s.jvp(out, [x, w1], [s.tensor(UX), s.tensor(UW)])
        assert s.shape(jtv_x) == X.shape and s.shape(jtv_w) == W1.shape and s.shape(ju) == V.shape
        return s.numpy(jtv_x), s.numpy(jtv_w), s.numpy(ju)


def test_jvp_vjp_adjoint():
    X, W1, W2, V, UX, UW = _adjoint_inputs()

    def fn(d):
        return np.tanh(np.tanh(d["x"] @ d["w1"]) @ _f64(W2))
    jac = _jacobian(fn, {"x": _f64(X), "w1": _f64(W1)}, ["x", "w1"])
    v = _f64(V).ravel()
    jtv_x, jtv_w, ju = _adjoint_run()
    _close(jtv_x, (jac["x"].T @ v).reshape(X.shape), "J^T v (x)")
    _close(jtv_w, (jac["w1"].T @ v).reshape(W1.shape), "J^T v (w1)")
    _close(ju, (jac["x"] @ _f64(UX).ravel() + jac["w1"] @ _f64(UW).ravel()).reshape(V.shape), "J u")
    lhs = float(np.sum(_f64(V) * _f64(ju)))
    rhs = float(np.sum(_f64(jtv_x) * _f64(UX)) + np.sum(_f64(jtv_w) * _f64(UW)))
    assert abs(lhs - rhs) <= 1e-4 * (1 + abs(lhs))
    assert _digest(jtv_x, jtv_w, ju) == _digest(*_adjoint_run())
    _emit("vjp", _digest(jtv_x, jtv_w))
    _emit("jvp", _digest(ju))


def test_vjp_create_graph():
    """A vjp with create_graph=True is differentiable: d/dx <J^T v, 1> for
    out = x * x * x (elementwise) is 6 x v."""
    X = np.array([[0.5, -1.0, 1.5]], dtype=np.float32)
    V = np.array([[1.0, 2.0, -0.5]], dtype=np.float32)
    with _ops().Session(record=True) as s:
        x = s.tensor(X, requires_grad=True)
        out = s.mul(s.mul(x, x), x)
        (g,) = s.vjp(out, [x], s.tensor(V), create_graph=True)
        s.backward(s.sum(g))
        second = s.grad(x)
    _close(second, 6 * _f64(X) * _f64(V), "derivative of a create_graph vjp")
    _emit("vjp_create_graph", _digest(second))


# ---------------------------------------------------------------- the guide's physics-informed derivative

def _physics_inputs():
    X = np.array([[-0.9], [-0.2], [0.4], [1.1]], dtype=np.float32)
    W1 = np.array([[0.8, -1.3, 0.5]], dtype=np.float32)
    B1 = np.array([[0.1, 0.3, -0.2]], dtype=np.float32)
    W2 = np.array([[0.9], [0.4], [-0.7]], dtype=np.float32)
    return X, W1, B1, W2


def _physics_run():
    X, W1, B1, W2 = _physics_inputs()
    with _ops().Session(record=True) as s:
        x = s.tensor(X, requires_grad=True)
        u = s.matmul(s.tanh(s.add(s.matmul(x, s.tensor(W1)), s.tensor(B1))), s.tensor(W2))
        s.backward(s.sum(u), create_graph=True)
        du = s.grad_tensor(x)
        first = s.numpy(du)
        s.backward(s.sum(du))
        return first, s.grad(x)


def test_physics_second_derivative():
    X, W1, B1, W2 = (_f64(a) for a in _physics_inputs())
    t = np.tanh(X @ W1 + B1)
    w1, w2 = W1.ravel(), W2.ravel()
    du = ((1 - t * t) * w1 * w2).sum(axis=1, keepdims=True)
    d2u = (-2 * t * (1 - t * t) * w1 * w1 * w2).sum(axis=1, keepdims=True)
    first, second = _physics_run()
    _close(first, du, "du/dx per row")
    _close(second, d2u, "d2u/dx2 per row")
    assert _digest(first, second) == _digest(*_physics_run())
    _emit("physics_du", _digest(first))
    _emit("physics_d2u", _digest(second))


# ---------------------------------------------------------------- refusals

def test_refusals():
    ops = _ops()
    with ops.Session(record=True) as s:
        x = s.tensor(np.array([[0.3, -0.8, 1.2], [0.7, 0.2, -0.4]], dtype=np.float32), requires_grad=True)
        loss = s.sum(s.mul(x, x))
        with pytest.raises(ValueError):
            s.backward(loss, accumulate=True, create_graph=True)
        s.backward(loss)
        with pytest.raises(Exception):
            s.grad_tensor(x)  # the last backward did not record
        wt = s.tensor(np.ones((1, 3), dtype=np.float32), requires_grad=True)
        bs = s.tensor(np.zeros((1, 3), dtype=np.float32), requires_grad=True)
        normed = s.sum(s.mul(s.layer_norm(x, wt, bs), x))
        with pytest.raises(Exception) as info:
            s.backward(normed, create_graph=True)
        text = str(info.value)
        assert "first-order gradient only" in text or "create_graph cannot differentiate through" in text, text
        # the plain first-order backward still works on the same graph
        s.backward(normed)
        assert np.isfinite(s.grad(x)).all()
    with ops.Session() as s:
        with pytest.raises(Exception):
            s.backward(s.tensor(np.ones((1, 1), dtype=np.float32)), create_graph=True)
