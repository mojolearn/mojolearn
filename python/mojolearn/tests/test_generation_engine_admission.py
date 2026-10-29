# SPDX-License-Identifier: Apache-2.0
"""Installed native generation admission checks, CPU or explicit GPU selection."""
import numpy as np
import pytest
from mojolearn.tests.test_generation_native import load_fixture, DEVICE


@pytest.mark.parametrize("engine", ["auto", "1"])
def test_default_dispatch_constructs_native_engine_owners(tmp_path, monkeypatch, engine):
    monkeypatch.delenv("MOJOLEARN_NEURAL_NATIVE_DISPATCH", raising=False)
    monkeypatch.setenv("MOJOLEARN_CAUSAL_LM_RESIDENT_FORWARD", "0")
    monkeypatch.setenv("MOJOLEARN_GENERATION_NATIVE_ENGINE", engine)
    model = load_fixture(tmp_path, "llama")
    if DEVICE == "gpu":
        assert model._blocks[0]._native_session is not None
    prompt = np.array([[1, 3]], dtype="i4")
    with monkeypatch.context() as legacy:
        legacy.setenv("MOJOLEARN_GENERATION_NATIVE_ENGINE", "0")
        expected_logits = np.asarray(model.forward(prompt)).copy()
        expected_tokens = np.asarray(model.stream(prompt, 2).read(2)).copy()
    state = model.allocate_state(1, 5)
    actual = np.asarray(model.forward(prompt, state))
    assert np.array_equal(actual.view("u4"), expected_logits.view("u4"))
    assert state.positions == 2
    stream = model.stream(prompt, 2)
    assert np.array_equal(np.asarray(stream.read(2)), expected_tokens)
    assert stream.info()["generated_tokens"] == 2


def test_explicit_dispatch_opt_out_keeps_auto_legacy_route(tmp_path, monkeypatch):
    monkeypatch.setenv("MOJOLEARN_NEURAL_NATIVE_DISPATCH", "0")
    monkeypatch.setenv("MOJOLEARN_CAUSAL_LM_RESIDENT_FORWARD", "0")
    monkeypatch.setenv("MOJOLEARN_GENERATION_NATIVE_ENGINE", "auto")
    model = load_fixture(tmp_path, "llama", uniform=True)
    prompt = np.array([[1, 3]], dtype="i4")
    state = model.allocate_state(1, 5)
    assert np.asarray(model.forward(prompt, state)).shape == (1, 2, model.vocab_size)
    assert state.positions == 2
    assert np.asarray(model.stream(prompt, 2).read(2)).tolist() == [[0, 0]]


def test_forced_engine_does_not_swallow_explicit_opt_out(tmp_path, monkeypatch):
    monkeypatch.setenv("MOJOLEARN_NEURAL_NATIVE_DISPATCH", "0")
    monkeypatch.setenv("MOJOLEARN_CAUSAL_LM_RESIDENT_FORWARD", "0")
    monkeypatch.setenv("MOJOLEARN_GENERATION_NATIVE_ENGINE", "1")
    model = load_fixture(tmp_path, "llama", uniform=True)
    with pytest.raises(Exception, match="native|session|lock"):
        model.forward(np.array([[1]], dtype="i4"))
