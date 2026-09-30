# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Sanity of the CNN trainer's pieces (Linear, softmax cross entropy, SGD)
against float64 NumPy, and that CNNClassifier learns a separable task. The
whole trajectory against torch is a two-stage check (torch and the bindings
live in different interpreters on the dev pod)."""
import numpy as np


def test_linear_softmax_sgd():
    import mojolearn as ml
    from mojolearn import _expansion_cnn as E
    rng = np.random.default_rng(0)
    x = rng.standard_normal((9, 5)).astype(np.float32)
    lin = E._Linear(5, 3, random_state=1)
    out = lin.forward(x)
    np.testing.assert_allclose(out, x.astype(np.float64) @ lin.weight_.T + lin.bias_, rtol=1e-5, atol=1e-5)
    y = rng.integers(0, 3, 9).astype(np.int32)
    loss, g, proba = E._softmax_xent(lin._binding(), out, y)
    z = out.astype(np.float64)
    p = np.exp(z - z.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    np.testing.assert_allclose(proba, p, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(loss, -np.log(p[np.arange(9), y]).mean(), rtol=1e-5)
    onehot = np.eye(3)[y]
    np.testing.assert_allclose(g, (p - onehot) / 9, rtol=1e-5, atol=1e-6)
    dx = lin.backward(g)
    np.testing.assert_allclose(dx, g.astype(np.float64) @ lin.weight_, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(lin.grad_weight_, g.astype(np.float64).T @ x, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(lin.grad_bias_, g.astype(np.float64).sum(0), rtol=1e-5, atol=1e-6)
    w = lin.weight_.copy()
    buf = np.full_like(w, 0.5)
    E._sgd(lin._binding(), lin.weight_, lin.grad_weight_, buf, 0.1, 0.9, 0.01)
    d = lin.grad_weight_.astype(np.float64) + 0.01 * w
    v = 0.9 * 0.5 + d
    np.testing.assert_allclose(buf, v, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(lin.weight_, w - 0.1 * v, rtol=1e-5, atol=1e-6)


def test_cnn_classifier_learns():
    import mojolearn as ml
    rng = np.random.default_rng(1)
    X = rng.standard_normal((128, 1, 6, 6)).astype(np.float32)
    y = (X[:, 0, :3].sum((1, 2)) > X[:, 0, 3:].sum((1, 2))).astype(int)
    m = ml.CNNClassifier(input_shape=(1, 6, 6), conv_channels=(4,), learning_rate=0.1, batch_size=32,
                         max_iter=15, random_state=0).fit(X.reshape(128, -1), y)
    assert m.loss_curve_[-1] < 0.5 * m.loss_curve_[0]
    assert (m.predict(X) == y).mean() > 0.9
