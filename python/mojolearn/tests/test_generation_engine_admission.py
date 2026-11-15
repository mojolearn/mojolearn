# SPDX-License-Identifier: Apache-2.0
"""Installed native generation admission checks, CPU or explicit GPU selection."""
import hashlib

import numpy as np
import pytest
from mojolearn.tests.test_generation_native import load_fixture, DEVICE
from mojolearn.tests._build_arms import built_with, two_build_compare

# The generation switches are build defines since lane envgate-neural-lm
# (2026-11-15). This check ran with MOJOLEARN_CAUSAL_LM_RESIDENT_FORWARD=0 and
# engine "auto" / "1" against an in-process engine "0" reference; now each arm
# is a wheel: on a GPU every wheel adds -D MOJOLEARN_CAUSAL_LM_RESIDENT_FORWARD_OFF,
# the engine arms are the default wheel ("auto") and -D
# MOJOLEARN_GENERATION_NATIVE_ENGINE_FORCE ("1"), and the reference is the -D
# MOJOLEARN_GENERATION_NATIVE_ENGINE_OFF wheel, compared through
# MOJOLEARN_TWO_BUILD_RECORD / _REFERENCE (python/mojolearn/tests/_build_arms.py).
pytestmark = pytest.mark.skipif(
    DEVICE == "gpu" and not built_with("MOJOLEARN_CAUSAL_LM_RESIDENT_FORWARD_OFF"),
    reason="needs a wheel built with -D MOJOLEARN_CAUSAL_LM_RESIDENT_FORWARD_OFF on a GPU",
)


def _words(logits, tokens):
    return dict(logits=hashlib.sha256(np.ascontiguousarray(logits).view("u4").tobytes()).hexdigest(),
                tokens=np.asarray(tokens).tolist())


def test_default_dispatch_constructs_native_engine_owners(tmp_path):
    model = load_fixture(tmp_path, "llama")
    prompt = np.array([[1, 3]], dtype="i4")
    if built_with("MOJOLEARN_GENERATION_NATIVE_ENGINE_OFF"):
        expected_logits = np.asarray(model.forward(prompt)).copy()
        expected_tokens = np.asarray(model.stream(prompt, 2).read(2)).copy()
        two_build_compare("generation_engine_admission", _words(expected_logits, expected_tokens))
        return
    if DEVICE == "gpu":
        assert model._blocks[0]._native_session is not None
    state = model.allocate_state(1, 5)
    actual = np.asarray(model.forward(prompt, state)).copy()
    assert state.positions == 2
    stream = model.stream(prompt, 2)
    tokens = np.asarray(stream.read(2)).copy()
    assert stream.info()["generated_tokens"] == 2
    two_build_compare("generation_engine_admission", _words(actual, tokens))


# test_explicit_dispatch_opt_out_keeps_auto_legacy_route and
# test_forced_engine_does_not_swallow_explicit_opt_out set
# MOJOLEARN_NEURAL_NATIVE_DISPATCH=0 at run time. Native dispatch is now a build
# constant (lane flag-native-dispatch-gate, 2026-11-15): their opt-out arm is
# the -D MOJOLEARN_NEURAL_NATIVE_DISPATCH_OFF build, an A/B arm, not a runtime
# route, so no process of the default build can reach it.
