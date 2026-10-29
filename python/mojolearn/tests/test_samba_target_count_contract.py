# SPDX-License-Identifier: Apache-2.0
"""Installed public regressions for Samba target-count ownership and dropout.

Run against actual compiled bindings; missing bindings are failures, not skips.
The small cases complement the standalone integer-count check and the full
samba-optimizer-state verifier lane without replacing vendor identity evidence.
"""
import numpy as np
import pytest

from mojolearn import SambaConfig, SambaStack, training


def _bytes(value):
    return np.asarray(value, dtype="f4").tobytes()


@pytest.mark.parametrize("length", [1, 7, 32])
def test_public_dropout_count_matches_explicit_normalization_and_trains(length):
    cfg = SambaConfig(19, 8, ["attention"], n_heads=2, intermediate=11,
                      dropout=0.125)
    model = SambaStack(cfg, generator=training.Generator(17), lr=0.001)
    inputs = (np.arange(length, dtype="i4") % 19).reshape(1, length)
    targets = ((inputs + 3) % 19).copy()
    if length > 1:
        targets[0, 1::3] = -100
    count = int(np.count_nonzero(targets != -100))
    before = model.generator.state_dict()
    expected = model.loss_and_grads(inputs, targets, num_items=count,
                                    dropout_stream=7)
    actual = model.loss_and_grads(inputs, targets, dropout_stream=7)
    assert _bytes(actual[0]) == _bytes(expected[0])
    assert len(actual[1]) == len(expected[1])
    for got, wanted in zip(actual[1], expected[1]):
        assert _bytes(got) == _bytes(wanted)
    assert model.generator.state_dict() == before
    model.optimizer.maximize = True
    result = model.train_step(inputs, targets)
    assert result["step"] == 1
    assert np.isfinite(result["loss"])
    assert model.generator.counter == before["counter"] + 1
    assert np.isfinite(np.asarray(model.flat)).all()


def test_public_optimizer_snapshot_resume_then_dropout_maximize():
    cfg = SambaConfig(256, 32, ["mamba3", "attention"], n_heads=2,
                      intermediate=64)
    ids = (np.arange(8 * 17, dtype="i4") % 256).reshape(8, 17)
    windows = [(ids[2*k:2*k+2, :-1], ids[2*k:2*k+2, 1:]) for k in range(4)]
    a = SambaStack(cfg, generator=training.Generator(1), lr=0.001)
    a.train_step(*windows[0])
    saved = a.state_dict()
    a.train_step(*windows[1])
    b = SambaStack(cfg, generator=training.Generator(1), lr=0.001)
    b.train_step(*windows[2])
    b.load_state_dict(saved)
    b.train_step(*windows[1])
    assert _bytes(a.flat) == _bytes(b.flat)
    assert _bytes(a.forward(ids[:2, :-1])) == _bytes(b.forward(ids[:2, :-1]))
    resumed = b.state_dict()
    continuous = a.state_dict()
    for name in ("parameters", "exp_avg", "exp_avg_sq"):
        assert _bytes(resumed[name]) == _bytes(continuous[name])
    c = SambaStack(SambaConfig(256, 32, ["mamba3", "attention"], n_heads=2,
                              intermediate=64, dropout=0.1),
                   generator=training.Generator(1), lr=0.001)
    c.optimizer.maximize = True
    initial_counter = c.generator.counter  # Weight initialization also draws streams.
    for step in range(2):
        result = c.train_step(*windows[step])
        assert result["step"] == step + 1
        assert np.isfinite(result["loss"])
    assert c.generator.counter == initial_counter + 2
    assert np.isfinite(np.asarray(c.flat)).all()
