# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The sequence lane's optimizers against float64 NumPy restatements of
torch.optim's update rules, at a tolerance. Parity with torch itself was
checked on the lane's pod (docs/lanes/progress/sequence.md)."""
import numpy as np

import mojolearn as ml


def _run(cls, ref, steps=6, **kw):
    rng = np.random.default_rng(0)
    p = rng.standard_normal((4, 3)).astype(np.float32)
    grads = [rng.standard_normal((4, 3)).astype(np.float32) for _ in range(steps)]
    opt = cls([p], **kw)
    q = p.astype(np.float64)
    state = {}
    for t, g in enumerate(grads, 1):
        opt.step([g])
        q = ref(q, g.astype(np.float64), t, state, **kw)
    np.testing.assert_allclose(p, q, rtol=2e-5, atol=2e-6)


def ref_rmsprop(p, g, t, st, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0.0, momentum=0.0, centered=False):
    g = g + weight_decay * p
    st["v"] = alpha * st.get("v", 0.0) + (1 - alpha) * g * g
    v = st["v"]
    if centered:
        st["ga"] = alpha * st.get("ga", 0.0) + (1 - alpha) * g
        v = v - st["ga"] ** 2
    avg = np.sqrt(v) + eps
    if momentum > 0:
        st["b"] = momentum * st.get("b", 0.0) + g / avg
        return p - lr * st["b"]
    return p - lr * g / avg


def test_rmsprop():
    _run(ml.RMSprop, ref_rmsprop)
    _run(ml.RMSprop, ref_rmsprop, lr=3e-3, centered=True, momentum=0.7, weight_decay=1e-2)


def ref_adagrad(p, g, t, st, lr=1e-2, lr_decay=0.0, weight_decay=0.0, initial_accumulator_value=0.0,
                eps=1e-10):
    g = g + weight_decay * p
    clr = lr / (1 + (t - 1) * lr_decay)
    st["s"] = st.get("s", initial_accumulator_value) + g * g
    return p - clr * g / (np.sqrt(st["s"]) + eps)


def test_adagrad():
    _run(ml.Adagrad, ref_adagrad)
    _run(ml.Adagrad, ref_adagrad, lr=5e-2, lr_decay=0.1, weight_decay=1e-2, initial_accumulator_value=0.1)


def ref_lion(p, g, t, st, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
    b1, b2 = betas
    m = st.get("m", 0.0)
    p = p * (1 - lr * weight_decay)
    p = p - lr * np.sign(b1 * m + (1 - b1) * g)
    st["m"] = b2 * m + (1 - b2) * g
    return p


def test_lion():
    _run(ml.Lion, ref_lion, lr=1e-2)
    _run(ml.Lion, ref_lion, lr=3e-3, betas=(0.95, 0.98), weight_decay=0.1)


def test_adafactor_against_torch_rule():
    """A float64 restatement of torch 2.5's _single_tensor_adafactor."""
    rng = np.random.default_rng(1)
    W = rng.standard_normal((5, 4)).astype(np.float32)
    v = rng.standard_normal(3).astype(np.float32)
    opt = ml.Adafactor([W, v], lr=2e-2, weight_decay=0.05)
    qW, qv = W.astype(np.float64), v.astype(np.float64)
    row, col, var = np.zeros(5), np.zeros(4), np.zeros(3)
    eps1, eps2 = float(np.finfo(np.float32).eps), 1e-3
    for t in range(1, 7):
        gW = rng.standard_normal((5, 4)).astype(np.float32)
        gv = rng.standard_normal(3).astype(np.float32)
        opt.step([gW, gv])
        w, rho = t ** -0.8, min(2e-2, t ** -0.5)
        out = []
        for q, g, kind in ((qW, gW.astype(np.float64), "m"), (qv, gv.astype(np.float64), "v")):
            alpha = max(eps2, np.linalg.norm(q) / np.sqrt(q.size)) * rho
            q = q * (1 - 2e-2 * 0.05)
            if kind == "m":
                row[:] = row + w * ((g * g).mean(1) - row)
                col[:] = col + w * ((g * g).mean(0) - col)
                est = np.outer(row, col) / max(row.mean(), eps1)
            else:
                var[:] = var + w * (g * g - var)
                est = var.copy()
            u = g / np.sqrt(np.maximum(est, eps1 * eps1))
            q = q - alpha / max(1.0, np.linalg.norm(u) / np.sqrt(u.size)) * u
            out.append(q)
        qW, qv = out
    np.testing.assert_allclose(W, qW, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(v, qv, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
