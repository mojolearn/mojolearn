# SPDX-License-Identifier: Apache-2.0
"""Installed native generation admission checks, CPU or explicit GPU selection."""
import numpy as np
import pytest
from mojolearn.tests.test_generation_native import load_fixture, DEVICE


@pytest.mark.parametrize("engine", ["auto", "1"])
def test_default_dispatch_constructs_native_engine_owners(tmp_path, monkeypatch, engine):
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


# test_explicit_dispatch_opt_out_keeps_auto_legacy_route and
# test_forced_engine_does_not_swallow_explicit_opt_out set
# MOJOLEARN_NEURAL_NATIVE_DISPATCH=0 at run time. Native dispatch is now a build
# constant (lane flag-native-dispatch-gate, 2026-11-15): their opt-out arm is
# the -D MOJOLEARN_NEURAL_NATIVE_DISPATCH_OFF build, an A/B arm, not a runtime
# route, so no process of the default build can reach it.
