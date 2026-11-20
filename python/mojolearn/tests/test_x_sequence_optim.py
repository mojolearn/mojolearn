# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Public sequence optimizer options and state/output publication.

Elementwise equations moved to sequence_optimizer_{cpu,gpu}.mojo; factored
matrix/vector equations moved to adafactor_{cpu,gpu}.mojo. These public calls
retain keyword decoding and state/output publication coverage.
"""
import numpy as np

import mojolearn as ml


def _public_step(cls, **options):
    params = np.arange(12, dtype=np.float32).reshape(4, 3)
    optimizer = cls([params], **options)
    optimizer.step([np.full_like(params, 0.125)])
    assert optimizer.t == 1
    assert params.shape == (4, 3) and params.dtype == np.float32
    state = optimizer.state_dict()
    assert type(state) is dict
    assert state["t"] == 1


def test_rmsprop():
    _public_step(ml.RMSprop)
    _public_step(ml.RMSprop, lr=3e-3, centered=True, momentum=0.7, weight_decay=1e-2)


def test_adagrad():
    _public_step(ml.Adagrad)
    _public_step(ml.Adagrad, lr=5e-2, lr_decay=0.1, weight_decay=1e-2, initial_accumulator_value=0.1)


def test_lion():
    _public_step(ml.Lion, lr=1e-2)
    _public_step(ml.Lion, lr=3e-3, betas=(0.95, 0.98), weight_decay=0.1)


def test_adafactor_matrix_and_vector_publication():
    # Independent six-step matrix/vector equations now live in the native
    # adafactor_{cpu,gpu} checks, including the resident GPU state route.
    matrix = np.arange(20, dtype=np.float32).reshape(5, 4)
    vector = np.array([0.5, -0.25, 1.0], dtype=np.float32)
    optimizer = ml.Adafactor([matrix, vector], lr=0.02, weight_decay=0.05)
    optimizer.step([np.full_like(matrix, 0.125), np.full_like(vector, -0.25)])
    assert optimizer.t == 1
    assert matrix.shape == (5, 4) and matrix.dtype == np.float32
    assert vector.shape == (3,) and vector.dtype == np.float32
    assert isinstance(optimizer.state, (list, tuple))


def test_adamax():
    _public_step(ml.Adamax)
    _public_step(ml.Adamax, lr=1e-2, betas=(0.8, 0.99), weight_decay=0.05)


def test_nadam():
    _public_step(ml.NAdam)
    _public_step(ml.NAdam, lr=1e-2, weight_decay=0.05, decoupled_weight_decay=True, momentum_decay=0.01)
