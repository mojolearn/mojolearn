# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Public sequence optimizer options and the remaining Adafactor oracle.

Elementwise optimizer equations moved to sequence_optimizer_{cpu,gpu}.mojo.
These public calls retain keyword decoding and state publication coverage;
Adafactor's factored matrix/vector numerical migration is still pending.
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




def test_adamax():
    _public_step(ml.Adamax)
    _public_step(ml.Adamax, lr=1e-2, betas=(0.8, 0.99), weight_decay=0.05)




def test_nadam():
    _public_step(ml.NAdam)
    _public_step(ml.NAdam, lr=1e-2, weight_decay=0.05, decoupled_weight_decay=True, momentum_decay=0.01)
