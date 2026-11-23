# SPDX-License-Identifier: Apache-2.0
"""Opt-in resident FP32 sampling contracts; explicit GPU runs only.

These compare candidate routes, never the old FP64 sequence. The routes are
build defines since lane envgate-neural-lm (2026-11-15), so the candidate is a
wheel built with -D MOJOLEARN_EXPERIMENT_RESIDENT_SAMPLING_FP32 and
-D MOJOLEARN_GENERATION_NATIVE_ENGINE_OFF (what the former autouse fixture set
in the environment), and the host-policy reference (the former in-process
MOJOLEARN_CAUSAL_LM_RESIDENT_FORWARD=0, or engine "1") is a second wheel that
adds -D MOJOLEARN_CAUSAL_LM_RESIDENT_FORWARD_OFF (or swaps in
-D MOJOLEARN_GENERATION_NATIVE_ENGINE_FORCE). Run this file on both with
MOJOLEARN_TWO_BUILD_RECORD / _REFERENCE (python/mojolearn/tests/_build_arms.py).
Compile-only delivery leaves execution and cross-vendor identity pending.
"""
import os

import numpy as np
import pytest

from mojolearn.tests.test_generation_native import load_fixture
from mojolearn.tests._build_arms import default_build_only, requires_build, two_build_compare

FP32 = "MOJOLEARN_EXPERIMENT_RESIDENT_SAMPLING_FP32"

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("MOJOLEARN_GENERATION_DEVICE") != "gpu",
        reason="explicit GPU resident sampling regression run required",
    ),
    # TOMBSTONE: MOJOLEARN_TRANSFORMER_LEGACY_SETUP (removed without measurement: plumbing, rule 2026-11-22) deleted 2026-11-23 by lane/ab-prep-transformer; code recoverable at 4d76a9e16e.
    # Restore: git apply ~/mojolearn-evidence/experiments-removed/MOJOLEARN_TRANSFORMER_LEGACY_SETUP.patch; record in docs/TOMBSTONES.md.
]


@pytest.mark.parametrize("uniform,temperature,top_k,top_p", [
    (False, 0.7, 0, 0.83), (False, 1.3, 17, 0.95), (True, 1.0, 4, 0.75),
    (False, 1e-45, 0, 1e-320),
])
@requires_build(FP32, "MOJOLEARN_GENERATION_NATIVE_ENGINE_OFF")
def test_candidate_matches_host_policy_and_chunk_checkpoint(
        tmp_path, uniform, temperature, top_k, top_p):
    model = load_fixture(tmp_path, "llama", uniform=uniform)
    assert model._native_entries["program_read"] is not None
    prompt = np.array([[1, 3], [2, 4]], dtype="i4")
    options = dict(greedy=False, seed=123, temperature=temperature,
                   top_k=top_k, top_p=top_p)
    # The host-policy words are the -D MOJOLEARN_CAUSAL_LM_RESIDENT_FORWARD_OFF
    # wheel's run of this same test (two_build_compare below).
    expected = np.asarray(model.stream(prompt, 6, **options).read(6)).copy()
    whole = np.asarray(model.stream(prompt, 6, **options).read(6)).copy()
    split = model.stream(prompt, 6, **options)
    prefix = np.asarray(split.read(2)).copy()
    saved = split.state_dict()
    assert saved["schema"] == "mojolearn.generation.v2"
    assert saved["sampling_policy"] == "mojolearn.sampling.fp32-compensated.v1"
    resumed = model.resume_generation(saved)
    tail = np.asarray(resumed.read(4)).copy()
    np.testing.assert_array_equal(whole, expected)
    np.testing.assert_array_equal(np.concatenate((prefix, tail), axis=1), expected)
    np.testing.assert_array_equal(np.asarray(split.read(4)), expected[:, 2:])
    assert resumed.info()["generated_tokens"] == 6
    assert model._resident_busy is False
    two_build_compare("resident_sampling_policy_%s_%r_%d_%r" % (uniform, temperature, top_k, top_p),
                      expected.tolist())


@default_build_only(FP32)
def test_legacy_checkpoint_does_not_adopt_enabled_policy(tmp_path):
    # Within the default build: a v1 (policy off) checkpoint resumes as v1.
    # Resuming that checkpoint on the FP32 wheel (the former in-process flip to
    # =1) needs the checkpoint carried across builds: OWED as a two-build
    # artifact check (lane envgate-neural-lm report).
    model = load_fixture(tmp_path, "llama", uniform=True)
    prompt = np.array([[33]], dtype="i4")
    stream = model.stream(prompt, 5, greedy=False, seed=0, top_k=4)
    stream.read(2)
    saved = stream.state_dict()
    assert saved["schema"] == "mojolearn.generation.v1"
    expected = np.asarray(stream.read(3)).copy()
    resumed = model.resume_generation(saved)
    assert resumed.state_dict()["schema"] == "mojolearn.generation.v1"
    np.testing.assert_array_equal(np.asarray(resumed.read(3)), expected)


@requires_build(FP32, "MOJOLEARN_GENERATION_NATIVE_ENGINE_OFF")
def test_candidate_failure_releases_lease_and_preserves_completed_count(tmp_path):
    model = load_fixture(tmp_path, "llama", uniform=True)
    stream = model.stream(np.array([[33, 34]], dtype="i4"), 5,
                          greedy=False, seed=7, top_k=4)
    old_limit = model.max_positions
    model.max_positions = 3
    with pytest.raises(Exception, match="exceeds max_positions"):
        stream.read(5)
    assert stream.info() == dict(done=False, generated_tokens=2,
                                 lengths=[2], usable=False)
    assert model._resident_busy is False
    assert model._resident_sessions is None
    with pytest.raises(Exception, match="lost after a failed call"):
        stream.read(1)
    model.max_positions = old_limit
    assert model.stream(np.array([[33]], dtype="i4"), 2,
                        greedy=False, seed=7, top_k=4).read(2).shape == (1, 2)


@requires_build(FP32, "MOJOLEARN_GENERATION_NATIVE_ENGINE_OFF")
@pytest.mark.parametrize("temperature,top_p", [(0.8, 0.83), (1e-45, 1e-320)])
def test_exposed_parameter_fallback_keeps_candidate_policy(tmp_path, temperature, top_p):
    model = load_fixture(tmp_path, "llama", uniform=False)
    prompt = np.array([[1, 3], [2, 4]], dtype="i4")
    options = dict(greedy=False, seed=(1 << 64) - 1, top_k=17,
                   top_p=top_p, temperature=temperature)
    expected = np.asarray(model.stream(prompt, 5, **options).read(5)).copy()
    owners = model.parameters()
    assert owners
    assert model._parameters_exposed is True
    stream = model.stream(prompt, 5, **options)
    np.testing.assert_array_equal(np.asarray(stream.read(5)), expected)
    assert stream.state_dict()["sampling_policy"] == "mojolearn.sampling.fp32-compensated.v1"


@requires_build(FP32, "MOJOLEARN_GENERATION_NATIVE_ENGINE_OFF")
def test_candidate_eos_and_padding_match_host_policy(tmp_path):
    # The host words are the -D MOJOLEARN_CAUSAL_LM_RESIDENT_FORWARD_OFF wheel's
    # run of this test (two_build_compare); within a build, eos and padding hold.
    model = load_fixture(tmp_path, "llama", uniform=True)
    prompt = np.array([[33], [34]], dtype="i4")
    options = dict(greedy=False, seed=0, top_k=4, top_p=0.9)
    first = np.asarray(model.stream(prompt, 1, **options).read(1))
    options.update(eos_token_id=int(first[0, 0]), pad_token_id=9)
    device = model.stream(prompt, 8, **options)
    expected = np.asarray(device.read(8)).copy()
    info = device.info()
    assert info["lengths"][0] == 1
    assert np.all(expected[0, 1:] == 9)
    two_build_compare("resident_sampling_eos_padding", dict(tokens=expected.tolist(), info=info))


@requires_build(FP32, "MOJOLEARN_CAUSAL_LM_RESIDENT_FORWARD_OFF")
@pytest.mark.parametrize("temperature,top_p", [(0.7, 0.83), (1e-45, 1e-320)])
def test_candidate_native_engine_uses_same_policy(tmp_path, temperature, top_p):
    # Engine "0" against engine "1": the -D MOJOLEARN_GENERATION_NATIVE_ENGINE_OFF
    # and -D MOJOLEARN_GENERATION_NATIVE_ENGINE_FORCE wheels (both with FP32 and
    # RESIDENT_FORWARD_OFF) record and compare the same words.
    model = load_fixture(tmp_path, "llama", uniform=False)
    prompt = np.array([[1, 3], [2, 4]], dtype="i4")
    options = dict(greedy=False, seed=123, top_k=17, top_p=top_p, temperature=temperature)
    stream = model.stream(prompt, 5, **options)
    tokens = np.asarray(stream.read(5)).copy()
    assert stream.state_dict()["sampling_policy"] == "mojolearn.sampling.fp32-compensated.v1"
    two_build_compare("resident_sampling_engine_%r_%r" % (temperature, top_p), tokens.tolist())
