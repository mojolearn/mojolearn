# SPDX-License-Identifier: Apache-2.0
"""GPU resident-generation API regressions, separate from identity columns.

Run explicitly with MOJOLEARN_GENERATION_DEVICE=gpu against current bindings.
Absent resident capabilities FAIL that run; ordinary CPU test collection skips
this GPU-only module. These bounded fixtures are not performance measurements.
The alias tests also exercise the native optional-resident admission door,
because ordinary streams own their state and do not expose writable KV aliases.
"""
import os

import numpy as np
import pytest

from mojolearn import Array
from mojolearn.models import Tokenizer
from mojolearn.models.causal_lm import _NATIVE
from mojolearn.tests.test_generation_native import load_fixture
from mojolearn.tests._build_arms import built_with, two_build_compare


pytestmark = pytest.mark.skipif(
    os.environ.get("MOJOLEARN_GENERATION_DEVICE") != "gpu",
    reason="explicit GPU resident-generation regression run required",
)


# The resident route is the build default since lane envgate-neural-lm
# (2026-11-15); the former autouse fixture set it in the environment.
# TOMBSTONE: MOJOLEARN_TRANSFORMER_LEGACY_SETUP (removed without measurement: plumbing, rule 2026-11-22) deleted 2026-11-23 by lane/ab-prep-transformer; code recoverable at 4d76a9e16e.
# Restore: git apply ~/mojolearn-evidence/experiments-removed/MOJOLEARN_TRANSFORMER_LEGACY_SETUP.patch; record in docs/TOMBSTONES.md.
RESIDENT_ARMS_OFF = ("MOJOLEARN_CAUSAL_LM_RESIDENT_FORWARD_OFF",)


def resident_model(path, *, uniform=False):
    present = [d for d in RESIDENT_ARMS_OFF if built_with(d)]
    if present:
        pytest.skip("resident-route check; this wheel has " + " ".join("-D " + d for d in present))
    model = load_fixture(path, "llama", uniform=uniform)
    entries = model._native_entries
    assert model.kind == "transformer"
    assert model._head_int15 is None and model._norm is not None
    assert entries["program_read"] is not None
    assert entries["native_block_operations"] is not None
    assert entries.get("parallel_call") is None
    # The resident sessions are native registry handles reached through the
    # CausalLM session table; admission records only that the table is there.
    assert entries["resident"] is True
    # Loaded blocks carry their compiled arithmetic-profile tag. Keep that
    # metadata intact: successful public tests below must prove actual resident
    # selection through the retained cache, not bypass admission in the fixture.
    return model


def test_resident_greedy_read_matches_legacy_forward_and_chunking(tmp_path):
    prompt = np.array([[1, 3], [2, 4]], dtype="i4")
    if built_with("MOJOLEARN_CAUSAL_LM_RESIDENT_FORWARD_OFF"):
        # The legacy-route wheel (-D MOJOLEARN_CAUSAL_LM_RESIDENT_FORWARD_OFF
        # -D MOJOLEARN_GENERATION_NATIVE_ENGINE_OFF; the former in-process
        # environment switches) records the words the resident wheel compares.
        legacy = load_fixture(tmp_path, "llama")
        two_build_compare("resident_greedy_read", np.asarray(legacy.stream(prompt, 5).read(5)).tolist())
        return
    model = resident_model(tmp_path)
    expected = np.asarray(model.stream(prompt, 5).read(5)).copy()
    whole = model.stream(prompt, 5)
    actual = np.asarray(whole.read(5)).copy()
    split = model.stream(prompt, 5)
    prefix = split.read(2)
    saved_prefix = np.asarray(prefix).copy()
    tail = np.asarray(split.read(3)).copy()
    assert np.array_equal(actual, expected)
    assert np.array_equal(np.concatenate((saved_prefix, tail), axis=1), expected)
    assert np.array_equal(np.asarray(prefix), saved_prefix), "later read overwrote a retained result"
    assert whole.info() == dict(done=True, generated_tokens=5, lengths=[5, 5], usable=True)
    assert model._resident_sessions is not None
    assert model._resident_busy is False
    two_build_compare("resident_greedy_read", expected.tolist())


@pytest.mark.parametrize("eos", [0, [0, 7]])
def test_resident_greedy_eos_stops_and_text_omits_eos(tmp_path, eos):
    model = resident_model(tmp_path, uniform=True)
    prompt = np.array([[33], [34]], dtype="i4")
    stream = model.stream(prompt, 8, eos_token_id=eos, pad_token_id=9)
    assert np.asarray(stream.read(8)).tolist() == [[0], [0]]
    assert stream.info() == dict(done=True, generated_tokens=1, lengths=[1, 1], usable=True)
    assert stream.read(1).shape == (2, 0)
    tokenizer = Tokenizer([bytes([i]) for i in range(256)], [], pattern="gpt2")
    assert model.generate_text("!", 8, tokenizer=tokenizer, eos_token_id=eos) == ""
    assert model._resident_busy is False
    assert model._resident_sessions is not None


@pytest.mark.parametrize("overlap", ["none", "same-layer", "cross-layer", "partial"])
def test_optional_resident_door_declines_overlapping_kv_without_mutation(tmp_path, overlap):
    model = resident_model(tmp_path, uniform=True)
    state = model.allocate_state(1, 7)
    if overlap == "same-layer":
        state.layers[0].v_cache = state.layers[0].k_cache
    elif overlap == "cross-layer":
        state.layers[1].k_cache = state.layers[0].k_cache
    elif overlap == "partial":
        size = int(np.asarray(state.layers[0].k_cache).size)
        storage = np.zeros(size + 1, dtype="f4")
        state.layers[0].k_cache = Array.from_buffer(storage[:-1])
        state.layers[0].v_cache = Array.from_buffer(storage[1:])
        assert state.layers[0].v_cache._addr == state.layers[0].k_cache._addr + 4
    before = [np.asarray(getattr(layer, name)).copy()
              for layer in state.layers for name in ("k_cache", "v_cache")]
    result = _NATIVE.causal_lm_resident(model, np.array([[1, 3]], dtype="i4"), 3, None, state)
    if overlap == "none":
        # Positive control: a missing/nonworking resident door cannot make
        # every alias-refusal case pass merely by returning None universally.
        assert np.asarray(result).tolist() == [[1, 3, 0, 0, 0]]
        return
    assert result is None, "overlapping KV must keep the legacy publication route"
    assert state.positions == 0
    assert all(layer.cached_tokens == 0 for layer in state.layers)
    after = [np.asarray(getattr(layer, name))
             for layer in state.layers for name in ("k_cache", "v_cache")]
    assert all(np.array_equal(a, b) for a, b in zip(before, after))


def test_resident_failure_publishes_completed_steps_and_releases_cache(tmp_path):
    model = resident_model(tmp_path, uniform=True)
    prompt = np.array([[33, 34]], dtype="i4")
    stream = model.stream(prompt, 5)
    old_limit = model.max_positions
    # Admit a valid state first, then lower the public position limit. Prompt
    # prefill and one decode complete; the next step must fail inside the read.
    model.max_positions = 3
    with pytest.raises(Exception, match="exceeds max_positions"):
        stream.read(5)
    info = stream.info()
    assert info["generated_tokens"] == 2
    assert info["lengths"] == [2]
    assert info["usable"] is False
    assert model._resident_busy is False
    assert model._resident_sessions is None
    with pytest.raises(Exception, match="lost after a failed call"):
        stream.read(1)
    # Cleanup must permit a new independent operation, not revive failed state.
    model.max_positions = old_limit
    assert np.asarray(model.stream(prompt, 2).read(2)).tolist() == [[0, 0]]


def test_sampling_upload_reuse_preserves_integer_oracle_and_eos(tmp_path):
    # Both arms hold the same integer oracle: the default wheel (reuse off) and
    # the -D MOJOLEARN_EXPERIMENT_SAMPLING_KV_UPLOAD_REUSE wheel (the former
    # in-process =1 parameter) each run this test.
    model = resident_model(tmp_path, uniform=True)
    stream = model.stream(np.array([[33], [34]], dtype="i4"), 8,
                          greedy=False, seed=0, top_k=4, eos_token_id=3, pad_token_id=9)
    assert np.asarray(stream.read(20)).tolist() == [
        [3, 9, 9, 9, 9, 9, 9], [1, 0, 2, 2, 0, 0, 3],
    ]
    assert stream.info() == dict(done=True, generated_tokens=7, lengths=[1, 7], usable=True)
    assert model._resident_busy is False
    assert model._resident_sessions is not None
