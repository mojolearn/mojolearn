# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Dropout2d's eval mode (the public layer returns x without a mask) and the
float64 NumPy reference of PyTorch's adaptive pools checked against torch
(ORACLE). The mask, adaptive pool and BasicBlock numerics are native
(checks/native_tests/cnn_blocks.mojo, `pixi run
check-native-cnn-blocks-cpu`); the BasicBlock is also compared with
torchvision's by a two-stage check."""
import numpy as np


def _x(shape, seed):
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


def test_dropout2d_eval_returns_x():
    import mojolearn as ml
    x = _x((64, 8, 3, 3), 0) + np.float32(3)
    d = ml.Dropout2d(p=0.3, random_state=7)
    d.forward(x)
    d.eval()
    assert np.array_equal(d.forward(x), x)


def _ref_adaptive(x, g, osize, is_max):
    n, c, h, w = x.shape
    oh, ow = osize
    out = np.zeros((n, c, oh, ow))
    dx = np.zeros(x.shape)
    for i in range(oh):
        hs, he = (i * h) // oh, -(-((i + 1) * h) // oh)
        for j in range(ow):
            ws, we = (j * w) // ow, -(-((j + 1) * w) // ow)
            win = x[:, :, hs:he, ws:we].astype(np.float64).reshape(n, c, -1)
            if is_max:
                t = win.argmax(2)
                out[:, :, i, j] = win.max(2)
                for a in range(n):
                    for b in range(c):
                        dx[a, b, hs + t[a, b] // (we - ws), ws + t[a, b] % (we - ws)] += g[a, b, i, j]
            else:
                out[:, :, i, j] = win.mean(2)
                dx[:, :, hs:he, ws:we] += g[:, :, i:i + 1, j:j + 1] / ((he - hs) * (we - ws))
    return out, dx


def test_adaptive_reference_matches_torch():
    import pytest
    torch = pytest.importorskip("torch")
    F = torch.nn.functional
    x = _x((2, 3, 7, 5), 8)
    for fn, is_max in ((F.adaptive_avg_pool2d, False), (F.adaptive_max_pool2d, True)):
        for osize in ((3, 2), (4, 4), (2, 5)):
            tx = torch.tensor(x, dtype=torch.float64, requires_grad=True)
            y = fn(tx, osize)
            g = _x(tuple(y.shape), 9)
            y.backward(torch.tensor(g, dtype=torch.float64))
            ry, rdx = _ref_adaptive(x, g, osize, is_max)
            # torch's avg-pool backward rounds through float32 somewhere (2e-8 absolute)
            np.testing.assert_allclose(y.detach().numpy(), ry, rtol=1e-6, atol=1e-7)
            np.testing.assert_allclose(tx.grad.numpy(), rdx, rtol=1e-6, atol=1e-7)
