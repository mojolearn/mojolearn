# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The public `fit_loop_` attribute of the neural estimators (lane alm-neural,
2026-11-24) at the Python boundary: absent before fit (AttributeError, as the
other fitted attributes), and after fit a tuple of (int, float) whose count
is the capped epochs. The loop record itself is checked natively in
checks/native_tests/neural_fit_loop.mojo."""
import numpy as np
import pytest

import mojolearn as ml

K = 2


def _rows(n=32, d=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    return X, (X[:, 0] > 0).astype(np.int64)


def _sequences(n=32, t=5, d=3, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, t, d)).astype(np.float32)
    return X, (X[:, -1, 0] + 0.5 * X[:, -2, 1]).astype(np.float32)


def _check(model, iterations):
    loop = model.fit_loop_
    assert isinstance(loop, tuple) and len(loop) == 2
    assert type(loop[0]) is int and type(loop[1]) is float
    assert loop[0] == iterations
    assert loop[1] >= 0.0


def _unfitted(model):
    with pytest.raises(AttributeError):
        model.fit_loop_


def test_mlp_fit_loop():
    X, y = _rows()
    for model, target in ((ml.MLPClassifier(hidden_layer_sizes=(8,), max_iter=K, tol=0.0, n_iter_no_change=1000000,
                                            batch_size=8, random_state=0), y),
                          (ml.MLPRegressor(hidden_layer_sizes=(8,), max_iter=K, tol=0.0, n_iter_no_change=1000000,
                                           batch_size=8, random_state=0), y.astype(np.float32))):
        _unfitted(model)
        model.fit(X, target)
        _check(model, K)
        assert model.n_iter_ == K


def test_recurrent_fit_loop():
    X, y = _sequences()
    for cls in (ml.LSTMRegressor, ml.GRURegressor, ml.RNNRegressor):
        model = cls(hidden_size=4, batch_size=8, max_epochs=K, random_state=0)
        _unfitted(model)
        model.fit(X, y)
        # iterations count epochs; n_iter_ counts optimizer steps (K * ceil(32 / 8))
        _check(model, K)
        assert model.n_iter_ == K * 4
    model = ml.LSTMClassifier(hidden_size=4, batch_size=8, max_epochs=K, random_state=0)
    _unfitted(model)
    model.fit(X, (y > 0).astype(np.int64))
    _check(model, K)


def test_sequential_fit_loop():
    X, y = _rows()
    model = ml.Sequential().linear(8).relu().linear(1)
    model.compile(loss="bce_with_logits", optimizer="adam", lr=1e-3, tol=0.0, n_iter_no_change=0)
    _unfitted(model)
    model.fit(X, y, epochs=K, batch_size=8, shuffle=True, seed=7)
    _check(model, K)
    assert model.n_iter_ == K


def test_cnn_fit_loop():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((32, 36)).astype(np.float32)
    y = np.arange(32) % 3
    model = ml.CNNClassifier(input_shape=(1, 6, 6), conv_channels=(2,), batch_size=8, max_iter=K, random_state=0)
    _unfitted(model)
    model.fit(X, y)
    _check(model, K)
