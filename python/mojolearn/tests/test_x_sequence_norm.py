# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""LayerNorm against a float64 NumPy restatement of F.layer_norm and its
gradients (parity with torch itself within 9e-6 was checked on the pod)."""
import numpy as np

import mojolearn as ml


def test_forward_backward():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((32, 6)) * 2 + 1).astype(np.float32)
    dy = rng.standard_normal((32, 6)).astype(np.float32)
    w = rng.standard_normal(6).astype(np.float32)
    b = rng.standard_normal(6).astype(np.float32)
    xd = x.astype(np.float64)
    mu = xd.mean(1, keepdims=True)
    rstd = 1 / np.sqrt(xd.var(1, keepdims=True) + 1e-5)
    xh = (xd - mu) * rstd
    np.testing.assert_allclose(ml.layer_norm_forward(x, weight=w, bias=b), xh * w + b, atol=2e-5)
    g = dy * w
    dx_ref = rstd * (g - g.mean(1, keepdims=True) - xh * (g * xh).mean(1, keepdims=True))
    dx, dw, db = ml.layer_norm_backward(dy, x, weight=w, bias=b)
    np.testing.assert_allclose(dx, dx_ref, atol=2e-5)
    np.testing.assert_allclose(dw, (dy * xh).sum(0), atol=2e-4)
    np.testing.assert_allclose(db, dy.sum(0), atol=2e-5)


def test_module():
    ln = ml.LayerNorm(4)
    x = np.arange(8, dtype=np.float32).reshape(2, 4)
    y = ln(x)
    np.testing.assert_allclose(y.mean(1), 0.0, atol=1e-6)
    ln.backward(np.ones_like(x))
    assert ln.weight_grad.shape == (4,) and ln.bias_grad.shape == (4,)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
