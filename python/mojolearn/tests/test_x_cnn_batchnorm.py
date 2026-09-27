# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Correctness sanity of BatchNorm2d/1d against a float64 NumPy reference of
PyTorch's batch_norm (training and eval, forward and backward, running
statistics), itself checked against torch where torch is installed."""
import numpy as np
import pytest


def ref_bn(x, gamma, beta, rm, rv, training, momentum, eps, g):
    x = x.astype(np.float64)
    g = g.astype(np.float64)
    ax = (0,) + tuple(range(2, x.ndim))
    shape = (1, -1) + (1,) * (x.ndim - 2)
    count = x.size // x.shape[1]
    if training:
        mean = x.mean(ax)
        var = x.var(ax)
        rm = (1 - momentum) * rm + momentum * mean
        rv = (1 - momentum) * rv + momentum * var * count / (count - 1)
    else:
        mean, var = rm.astype(np.float64), rv.astype(np.float64)
    invstd = 1 / np.sqrt(var + eps)
    xhat = (x - mean.reshape(shape)) * invstd.reshape(shape)
    y = xhat * gamma.reshape(shape) + beta.reshape(shape)
    dbeta = g.sum(ax)
    dgamma = (g * xhat).sum(ax)
    if training:
        dx = (g - (dbeta / count).reshape(shape) - xhat * (dgamma / count).reshape(shape)) * \
            (invstd * gamma).reshape(shape)
    else:
        dx = g * (invstd * gamma).reshape(shape)
    return y, dx, dgamma, dbeta, rm, rv


def _data(seed, shape):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape) * 2 + 0.5).astype(np.float32)


@pytest.mark.parametrize("shape", [(4, 3, 5, 6), (7, 5), (6, 4, 9)])
def test_batchnorm(shape):
    import mojolearn as ml
    c = shape[1]
    bn = (ml.BatchNorm2d if len(shape) == 4 else ml.BatchNorm1d)(c, momentum=0.2)
    bn.weight_ = _data(9, (c,))
    bn.bias_ = _data(8, (c,))
    rm, rv = bn.running_mean_.astype(np.float64), bn.running_var_.astype(np.float64)
    for step in range(2):
        x, g = _data(step, shape), _data(10 + step, shape)
        y = bn.forward(x)
        dx = bn.backward(g)
        ry, rdx, rdg, rdb, rm, rv = ref_bn(x, bn.weight_, bn.bias_, rm, rv, True, 0.2, 1e-5, g)
        np.testing.assert_allclose(y, ry, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(dx, rdx, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(bn.grad_weight_, rdg, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(bn.grad_bias_, rdb, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(bn.running_mean_, rm, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(bn.running_var_, rv, rtol=1e-5, atol=1e-6)
    bn.eval()
    x, g = _data(5, shape), _data(6, shape)
    y = bn.forward(x)
    dx = bn.backward(g)
    ry, rdx, rdg, rdb, _, _ = ref_bn(x, bn.weight_, bn.bias_, bn.running_mean_, bn.running_var_, False, 0.2, 1e-5, g)
    np.testing.assert_allclose(y, ry, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(dx, rdx, rtol=1e-4, atol=1e-5)


def test_reference_matches_torch():
    torch = pytest.importorskip("torch")
    for shape in [(4, 3, 5, 6), (7, 5)]:
        c = shape[1]
        gamma, beta = _data(9, (c,)).astype(np.float64), _data(8, (c,)).astype(np.float64)
        m = (torch.nn.BatchNorm2d if len(shape) == 4 else torch.nn.BatchNorm1d)(c, momentum=0.2).double()
        with torch.no_grad():
            m.weight.copy_(torch.tensor(gamma))
            m.bias.copy_(torch.tensor(beta))
        rm, rv = np.zeros(c), np.ones(c)
        for training in (True, False):
            m.train(training)
            x, g = _data(1, shape), _data(2, shape)
            tx = torch.tensor(x, dtype=torch.float64, requires_grad=True)
            out = m(tx)
            out.backward(torch.tensor(g, dtype=torch.float64))
            ry, rdx, rdg, rdb, rm, rv = ref_bn(x, gamma, beta, rm, rv, training, 0.2, 1e-5, g)
            np.testing.assert_allclose(out.detach().numpy(), ry, rtol=1e-9, atol=1e-9)
            np.testing.assert_allclose(tx.grad.numpy(), rdx, rtol=1e-9, atol=1e-9)
            np.testing.assert_allclose(m.running_mean.numpy(), rm, rtol=1e-9, atol=1e-12)
            np.testing.assert_allclose(m.running_var.numpy(), rv, rtol=1e-9, atol=1e-12)
            if training:
                np.testing.assert_allclose(m.weight.grad.numpy(), rdg, rtol=1e-9, atol=1e-9)
                np.testing.assert_allclose(m.bias.grad.numpy(), rdb, rtol=1e-9, atol=1e-9)


def test_batchnorm_options():
    import mojolearn as ml
    shape = (5, 3, 4, 4)
    # momentum=None: the cumulative average over the batches seen
    bn = ml.BatchNorm2d(3, momentum=None)
    rm, rv = np.zeros(3), np.ones(3)
    for step in range(3):
        x, g = _data(20 + step, shape), _data(30 + step, shape)
        bn.forward(x)
        _, _, _, _, rm, rv = ref_bn(x, bn.weight_, bn.bias_, rm, rv, True, 1.0 / (step + 1), 1e-5, g)
        np.testing.assert_allclose(bn.running_mean_, rm, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(bn.running_var_, rv, rtol=1e-5, atol=1e-6)
    # track_running_stats=False: batch statistics in eval mode as well
    bn = ml.BatchNorm2d(3, track_running_stats=False).eval()
    x, g = _data(40, shape), _data(41, shape)
    y = bn.forward(x)
    dx = bn.backward(g)
    ry, rdx, _, _, _, _ = ref_bn(x, bn.weight_, bn.bias_, np.zeros(3), np.ones(3), True, 0.1, 1e-5, g)
    np.testing.assert_allclose(y, ry, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(dx, rdx, rtol=1e-4, atol=1e-4)
    assert bn.running_mean_ is None


def test_options_reference_matches_torch():
    torch = pytest.importorskip("torch")
    shape = (5, 3, 4, 4)
    m = torch.nn.BatchNorm2d(3, momentum=None).double()
    rm, rv = np.zeros(3), np.ones(3)
    for step in range(3):
        x, g = _data(20 + step, shape), _data(30 + step, shape)
        m(torch.tensor(x, dtype=torch.float64))
        _, _, _, _, rm, rv = ref_bn(x, np.ones(3), np.zeros(3), rm, rv, True, 1.0 / (step + 1), 1e-5, g)
        np.testing.assert_allclose(m.running_mean.numpy(), rm, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(m.running_var.numpy(), rv, rtol=1e-10, atol=1e-12)
    m = torch.nn.BatchNorm2d(3, track_running_stats=False).double().eval()
    x, g = _data(40, shape), _data(41, shape)
    tx = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    out = m(tx)
    out.backward(torch.tensor(g, dtype=torch.float64))
    ry, rdx, _, _, _, _ = ref_bn(x, np.ones(3), np.zeros(3), np.zeros(3), np.ones(3), True, 0.1, 1e-5, g)
    np.testing.assert_allclose(out.detach().numpy(), ry, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(tx.grad.numpy(), rdx, rtol=1e-9, atol=1e-9)
