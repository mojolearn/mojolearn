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
    # the softmax cross-entropy and SGD steps run inside CNNClassifier's one
    # native fit now (the Python _softmax_xent and _sgd doors are gone); the
    # layer's own backward is checked against an arbitrary upstream gradient
    g = rng.standard_normal((9, 3)).astype(np.float32)
    dx = lin.backward(g)
    np.testing.assert_allclose(dx, g.astype(np.float64) @ lin.weight_, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(lin.grad_weight_, g.astype(np.float64).T @ x, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(lin.grad_bias_, g.astype(np.float64).sum(0), rtol=1e-5, atol=1e-6)

def test_cnn_classifier_learns():
    import mojolearn as ml
    rng = np.random.default_rng(1)
    X = rng.standard_normal((128, 1, 6, 6)).astype(np.float32)
    y = (X[:, 0, :3].sum((1, 2)) > X[:, 0, 3:].sum((1, 2))).astype(int)
    m = ml.CNNClassifier(input_shape=(1, 6, 6), conv_channels=(4,), learning_rate=0.1, batch_size=32,
                         max_iter=15, random_state=0).fit(X.reshape(128, -1), y)
    assert m.loss_curve_[-1] < 0.5 * m.loss_curve_[0]
    assert (m.predict(X) == y).mean() > 0.9
