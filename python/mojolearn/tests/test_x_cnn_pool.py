# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The float64 NumPy reference of PyTorch's MaxPool2d / AvgPool2d (floor and
ceil mode), checked against torch where torch is installed (ORACLE). The
layers are held to the same reference natively in
checks/native_tests/cnn_pool.mojo (`pixi run check-native-cnn-pool-cpu`)."""
import numpy as np
import pytest


def _windows(h, w, k, s, p, d, oh, ow):
    for i in range(oh):
        for j in range(ow):
            yield i, j, [(i * s[0] - p[0] + a * d[0], j * s[1] - p[1] + b * d[1])
                         for a in range(k[0]) for b in range(k[1])]


def _out_1(size, k, s, p, d, ceil):
    num = size + 2 * p - d * (k - 1) - 1 + ((s - 1) if ceil else 0)
    out = num // s + 1
    if ceil and (out - 1) * s >= size + p:
        out -= 1
    return out


def _out_hw(h, w, k, s, p, d, ceil=False):
    return _out_1(h, k[0], s[0], p[0], d[0], ceil), _out_1(w, k[1], s[1], p[1], d[1], ceil)


def ref_maxpool(x, g, k, s, p, d, ceil=False):
    n, c, h, w = x.shape
    oh, ow = _out_hw(h, w, k, s, p, d, ceil)
    out = np.zeros((n, c, oh, ow))
    dx = np.zeros(x.shape)
    for i, j, taps in _windows(h, w, k, s, p, d, oh, ow):
        taps = [(a, b) for a, b in taps if 0 <= a < h and 0 <= b < w]
        for nn in range(n):
            for cc in range(c):
                vals = [x[nn, cc, a, b] for a, b in taps]
                t = int(np.argmax(vals))  # first maximum
                out[nn, cc, i, j] = vals[t]
                dx[nn, cc, taps[t][0], taps[t][1]] += g[nn, cc, i, j]
    return out, dx


def ref_avgpool(x, g, k, s, p, include_pad, ceil=False, override=None):
    n, c, h, w = x.shape
    oh, ow = _out_hw(h, w, k, s, p, (1, 1), ceil)
    out = np.zeros((n, c, oh, ow))
    dx = np.zeros(x.shape)
    for i, j, taps in _windows(h, w, k, s, p, (1, 1), oh, ow):
        hs, ws = i * s[0] - p[0], j * s[1] - p[1]
        he, we = min(hs + k[0], h + p[0]), min(ws + k[1], w + p[1])
        div = (he - hs) * (we - ws) if include_pad else (min(he, h) - max(hs, 0)) * (min(we, w) - max(ws, 0))
        div = override or div
        taps = [(a, b) for a, b in taps if 0 <= a < h and 0 <= b < w]
        for a, b in taps:
            out[:, :, i, j] += x[:, :, a, b]
            dx[:, :, a, b] += g[:, :, i, j] / div
        out[:, :, i, j] /= div
    return out, dx


MAX_CASES = [((2, 3), (1, 1), (1, 1), (1, 1)), ((3, 3), (2, 2), (1, 1), (1, 1)), ((2, 2), (1, 2), (1, 1), (1, 2)),
             ((2, 2), (2, 2), (0, 0), (1, 1))]
AVG_CASES = [((3, 3), (1, 1), (1, 1), True), ((2, 3), (1, 2), (1, 1), False), ((2, 2), (2, 2), (0, 0), True),
             ((3, 2), (2, 1), (1, 1), True)]


def _x(shape, seed):
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


def test_reference_matches_torch():
    torch = pytest.importorskip("torch")
    F = torch.nn.functional
    for k, s, p, d in MAX_CASES:
        x = _x((2, 3, 6, 7), 0)
        tx = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        out = F.max_pool2d(tx, k, s, p, d)
        g = _x(tuple(out.shape), 1)
        out.backward(torch.tensor(g, dtype=torch.float64))
        rout, rdx = ref_maxpool(x, g, k, s, p, d)
        np.testing.assert_allclose(out.detach().numpy(), rout, atol=1e-12)
        np.testing.assert_allclose(tx.grad.numpy(), rdx, atol=1e-12)
    for k, s, p, inc in AVG_CASES:
        x = _x((2, 3, 6, 7), 0)
        tx = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        out = F.avg_pool2d(tx, k, s, p, count_include_pad=inc)
        g = _x(tuple(out.shape), 1)
        out.backward(torch.tensor(g, dtype=torch.float64))
        rout, rdx = ref_avgpool(x, g, k, s, p, inc)
        np.testing.assert_allclose(out.detach().numpy(), rout, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(tx.grad.numpy(), rdx, rtol=1e-5, atol=1e-7)


CEIL_CASES = [((3, 3), (2, 2), (1, 1), True, None), ((2, 3), (2, 2), (0, 1), False, None),
              ((3, 2), (2, 3), (1, 0), True, 5), ((2, 2), (3, 3), (1, 1), False, 3)]


def test_ceil_reference_matches_torch():
    torch = pytest.importorskip("torch")
    F = torch.nn.functional
    for k, s, p, inc, ovr in CEIL_CASES:
        x = _x((2, 3, 6, 7), 0)
        tx = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        out = F.max_pool2d(tx, k, s, p, 1, ceil_mode=True)
        g = _x(tuple(out.shape), 1)
        out.backward(torch.tensor(g, dtype=torch.float64))
        rout, rdx = ref_maxpool(x, g, k, s, p, (1, 1), ceil=True)
        np.testing.assert_allclose(out.detach().numpy(), rout, atol=1e-12)
        np.testing.assert_allclose(tx.grad.numpy(), rdx, atol=1e-12)
        tx = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        out = F.avg_pool2d(tx, k, s, p, ceil_mode=True, count_include_pad=inc, divisor_override=ovr)
        g = _x(tuple(out.shape), 2)
        out.backward(torch.tensor(g, dtype=torch.float64))
        rout, rdx = ref_avgpool(x, g, k, s, p, inc, ceil=True, override=ovr)
        np.testing.assert_allclose(out.detach().numpy(), rout, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(tx.grad.numpy(), rdx, rtol=1e-5, atol=1e-7)
