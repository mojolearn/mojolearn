# SPDX-License-Identifier: Apache-2.0
"""Writes the inputs and float64 oracles of checks/native_tests/ops_higher_cases.mojo.

Moved from python/mojolearn/tests/test_ops_higher.py (2026-11-14): the inputs
are the same NumPy PCG64 draws (default_rng(5..8)) the Python tests made, and
the oracles are the same float64 NumPy nested central differences
(Richardson-extrapolated from h and h/2) and Jacobians, computed here once and
stored so the Mojo check needs no NumPy stream. The closed-form oracles
(vjp of x*x*x, the physics network's derivatives, exp) are computed in Mojo.

Run from the repository root:  python checks/native_tests/fixtures/ops_higher/generate.py
Files: <case>-<name>.f32 (little-endian float32 inputs, C order) and
<case>-<name>.f64 (little-endian float64 oracles, C order).
"""
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _f64(a):
    return np.asarray(a, dtype=np.float32).astype(np.float64)


def _hvp_oracle(fn, point, v, h=1e-3):
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


def _put(case, name, array, dtype):
    np.ascontiguousarray(array, dtype=dtype).astype(dtype.newbyteorder("<")).tofile(HERE / f"{case}-{name}.{dtype.char == 'f' and 'f32' or 'f64'}")


def f32(case, name, array):
    _put(case, name, array, np.dtype(np.float32))


def f64(case, name, array):
    _put(case, name, array, np.dtype(np.float64))


def example():
    rng = np.random.default_rng(5)
    X = (0.5 * rng.standard_normal((8, 3))).astype(np.float32)
    W0 = (0.5 * rng.standard_normal((3, 2))).astype(np.float32)
    Y = (0.5 * rng.standard_normal((8, 2))).astype(np.float32)
    V = rng.standard_normal((3, 2)).astype(np.float32)

    def fn(d):
        return float(np.mean((np.tanh(_f64(X) @ d["w"]) - _f64(Y)) ** 2))
    oracle = _hvp_oracle(fn, {"w": _f64(W0)}, {"w": _f64(V)})["w"]
    z = np.tanh(_f64(X) @ _f64(W0))
    gradient = _f64(X).T @ ((2.0 / Y.size) * (z - _f64(Y)) * (1 - z * z))
    for n, a in (("x", X), ("w", W0), ("y", Y), ("v", V)):
        f32("example", n, a)
    f64("example", "gradient", gradient)
    f64("example", "hvp", oracle)


def softmax_ce():
    rng = np.random.default_rng(6)
    X = (0.6 * rng.standard_normal((6, 3))).astype(np.float32)
    W = (0.6 * rng.standard_normal((3, 4))).astype(np.float32)
    B = (0.3 * rng.standard_normal((1, 4))).astype(np.float32)
    C = rng.integers(0, 4, size=(6, 1)).astype(np.float32)
    VW = rng.standard_normal((3, 4)).astype(np.float32)
    VB = rng.standard_normal((1, 4)).astype(np.float32)
    codes = C.ravel().astype(int)

    def fn(d):
        z = _f64(X) @ d["w"] + d["b"]
        z = z - z.max(axis=1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return float(-np.mean(logp[np.arange(len(codes)), codes]))
    oracle = _hvp_oracle(fn, {"w": _f64(W), "b": _f64(B)}, {"w": _f64(VW), "b": _f64(VB)})
    for n, a in (("x", X), ("w", W), ("b", B), ("c", C), ("vw", VW), ("vb", VB)):
        f32("softmax_ce", n, a)
    f64("softmax_ce", "hw", oracle["w"])
    f64("softmax_ce", "hb", oracle["b"])


def zero_curvature():
    rng = np.random.default_rng(7)
    f32("zero_curvature", "x", rng.standard_normal((4, 3)).astype(np.float32))
    f32("zero_curvature", "w", rng.standard_normal((3, 2)).astype(np.float32))


def adjoint():
    rng = np.random.default_rng(8)
    X = (0.6 * rng.standard_normal((3, 2))).astype(np.float32)
    W1 = (0.6 * rng.standard_normal((2, 3))).astype(np.float32)
    W2 = (0.6 * rng.standard_normal((3, 2))).astype(np.float32)
    V = rng.standard_normal((3, 2)).astype(np.float32)
    UX = rng.standard_normal((3, 2)).astype(np.float32)
    UW = rng.standard_normal((2, 3)).astype(np.float32)

    def fn(d):
        return np.tanh(np.tanh(d["x"] @ d["w1"]) @ _f64(W2))
    jac = _jacobian(fn, {"x": _f64(X), "w1": _f64(W1)}, ["x", "w1"])
    v = _f64(V).ravel()
    for n, a in (("x", X), ("w1", W1), ("w2", W2), ("v", V), ("ux", UX), ("uw", UW)):
        f32("adjoint", n, a)
    f64("adjoint", "jtv_x", (jac["x"].T @ v).reshape(X.shape))
    f64("adjoint", "jtv_w", (jac["w1"].T @ v).reshape(W1.shape))
    f64("adjoint", "ju", (jac["x"] @ _f64(UX).ravel() + jac["w1"] @ _f64(UW).ravel()).reshape(V.shape))


def layer_norm():
    point = {
        "x": _f64([[0.3, -0.8, 1.2], [0.7, 0.2, -0.4]]),
        "w": _f64([[0.8, -0.3, 1.1]]),
        "b": _f64([[0.1, 0.2, -0.1]]),
    }
    direction = {k: np.full_like(v, 0.2) for k, v in point.items()}
    direction["x"] = _f64([[0.2, -0.4, 0.1], [0.3, -0.2, 0.5]])

    def oracle(d):
        center = d["x"] - d["x"].mean(axis=1, keepdims=True)
        normalized = center / np.sqrt((center ** 2).mean(axis=1, keepdims=True) + 0.03)
        return np.sum((normalized * d["w"] + d["b"]) ** 2)
    expected = _hvp_oracle(oracle, point, direction)
    for k in point:
        f32("layer_norm", k, point[k])
        f32("layer_norm", "v" + k, direction[k])
        f64("layer_norm", "h" + k, expected[k])


if __name__ == "__main__":
    example()
    softmax_ce()
    zero_curvature()
    adjoint()
    layer_norm()
