# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Correctness sanity of Dropout2d, the adaptive (global) pools and the
ResNet BasicBlock against float64 NumPy references of PyTorch's semantics.
The BasicBlock is also compared with torchvision's by a two-stage check
recorded in docs/lanes/progress/cnn.md."""
import numpy as np

from mojolearn.tests.test_x_cnn_batchnorm import ref_bn
from mojolearn.tests.test_x_cnn_conv import ref_conv2d, ref_conv2d_backward


def _x(shape, seed):
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


def test_dropout2d():
    import mojolearn as ml
    x = _x((64, 8, 3, 3), 0) + np.float32(3)
    d = ml.Dropout2d(p=0.3, random_state=7)
    y = d.forward(x)
    m = d.mask_
    assert np.all(m.reshape(64, 8, -1).min(2) == m.reshape(64, 8, -1).max(2))  # one draw per channel
    kept = m[:, :, 0, 0] != 0
    assert 0.6 < kept.mean() < 0.8
    np.testing.assert_allclose(m[m != 0], 1 / 0.7, rtol=1e-6)
    np.testing.assert_allclose(y, x * m, rtol=1e-6)
    g = _x(x.shape, 1)
    np.testing.assert_allclose(d.backward(g), g * m, rtol=1e-6)
    y2 = d.forward(x)
    assert not np.array_equal(y, y2)  # a fresh mask per call
    assert np.array_equal(ml.Dropout2d(p=0.3, random_state=7).forward(x), y)  # reproducible
    assert np.all(ml.Dropout2d(p=1.0).forward(x) == 0)
    assert np.array_equal(ml.Dropout2d(p=0.0).forward(x), x)
    d.eval()
    assert np.array_equal(d.forward(x), x)


def test_adaptive_pools():
    import mojolearn as ml
    x = _x((3, 4, 6, 8), 2)
    a = ml.AdaptiveAvgPool2d(1)
    y = a.forward(x)
    np.testing.assert_allclose(y[:, :, 0, 0], x.astype(np.float64).mean((2, 3)), rtol=1e-5, atol=1e-6)
    g = _x(y.shape, 3)
    np.testing.assert_allclose(a.backward(g), np.broadcast_to(g / 48, x.shape), rtol=1e-5, atol=1e-7)
    m = ml.AdaptiveMaxPool2d((2, 4))
    y = m.forward(x)
    np.testing.assert_array_equal(y, x.reshape(3, 4, 2, 3, 4, 2).max((3, 5)))


def test_basic_block():
    import mojolearn as ml
    for inplanes, planes, stride in ((4, 4, 1), (3, 5, 2)):
        blk = ml.BasicBlock(inplanes, planes, stride=stride, random_state=4)
        x = _x((2, inplanes, 6, 6), 5)
        y = blk.forward(x)
        # float64 reference of the same block
        def bn(layer, t, g=None):
            c = t.shape[1]
            return ref_bn(t, layer.weight_, layer.bias_, np.zeros(c), np.ones(c), True, 0.1, 1e-5,
                          np.zeros(t.shape) if g is None else g)
        c1, _, _ = ref_conv2d(x, blk.conv1.weight_, None, (stride, stride), (1, 1), (1, 1))
        b1 = bn(blk.bn1, c1)[0]
        r1 = np.maximum(b1, 0)
        c2, _, _ = ref_conv2d(r1, blk.conv2.weight_, None, (1, 1), (1, 1), (1, 1))
        b2 = bn(blk.bn2, c2)[0]
        ident = x.astype(np.float64)
        if blk.downsample:
            cd, _, _ = ref_conv2d(x, blk.downsample[0].weight_, None, (stride, stride), (0, 0), (1, 1))
            ident = bn(blk.downsample[1], cd)[0]
        pre = b2 + ident
        np.testing.assert_allclose(y, np.maximum(pre, 0), rtol=1e-3, atol=1e-4)
        g = _x(y.shape, 6)
        dx = blk.backward(g)
        gp = g * (pre > 0)
        gb2 = bn(blk.bn2, c2, gp)[1]
        gr1 = ref_conv2d_backward(r1, blk.conv2.weight_, gb2, (1, 1), (1, 1), (1, 1))[0]
        gb1 = bn(blk.bn1, c1, gr1 * (b1 > 0))[1]
        rdx = ref_conv2d_backward(x, blk.conv1.weight_, gb1, (stride, stride), (1, 1), (1, 1))[0]
        if blk.downsample:
            gbd = bn(blk.downsample[1], cd, gp)[1]
            rdx = rdx + ref_conv2d_backward(x, blk.downsample[0].weight_, gbd, (stride, stride), (0, 0), (1, 1))[0]
        else:
            rdx = rdx + gp
        np.testing.assert_allclose(dx, rdx, rtol=1e-3, atol=1e-3)


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


def test_adaptive_non_dividing():
    import mojolearn as ml
    x = _x((2, 3, 7, 5), 8)
    for cls, is_max in ((ml.AdaptiveAvgPool2d, False), (ml.AdaptiveMaxPool2d, True)):
        for osize in ((3, 2), (4, 4), (2, 5)):
            m = cls(osize)
            y = m.forward(x)
            g = _x(y.shape, 9)
            ry, rdx = _ref_adaptive(x, g, osize, is_max)
            np.testing.assert_allclose(y, ry, rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(m.backward(g), rdx, rtol=1e-5, atol=1e-6)


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
