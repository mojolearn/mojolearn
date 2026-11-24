# SPDX-License-Identifier: Apache-2.0
"""Installed native generation admission checks, CPU or explicit GPU selection."""
import hashlib

import numpy as np
import pytest
from mojolearn.tests.test_generation_native import load_fixture, DEVICE
from mojolearn.tests._build_arms import built_with, two_build_compare

# The generation switches are build defines since lane envgate-neural-lm
# (2026-11-15). The engine arms are the default wheel ("auto") and -D
# MOJOLEARN_GENERATION_NATIVE_ENGINE_FORCE ("1"), and the reference is the -D
# MOJOLEARN_GENERATION_NATIVE_ENGINE_OFF wheel, compared through
# MOJOLEARN_TWO_BUILD_RECORD / _REFERENCE (python/mojolearn/tests/_build_arms.py).
# On a GPU the resident forward takes this fixture, so the engine admission is
# checked on the CPU install.
# TOMBSTONE: MOJOLEARN_CAUSAL_LM_RESIDENT_FORWARD_OFF (old path no faster: NV L40S 2 cells, median 1.00) deleted 2026-11-24 by lane/ab-outcome-2; code recoverable at 1e8aa88da1.
# Restore: git apply ~/mojolearn-evidence/experiments-removed/MOJOLEARN_CAUSAL_LM_RESIDENT_FORWARD_OFF.ab-outcome-2.patch
pytestmark = pytest.mark.skipif(
    DEVICE == "gpu",
    reason="on a GPU the resident forward takes this fixture; the engine admission runs on the CPU install",
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
# constant (lane flag-native-dispatch-gate, 2026-11-15) and, since 2026-11-23,
# the one form: the -D MOJOLEARN_NEURAL_NATIVE_DISPATCH_OFF opt-out arm is
# deleted, so no build can reach it.
