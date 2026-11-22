# SPDX-License-Identifier: Apache-2.0
"""Two threads generating on ONE CausalLM model: both work.

The model keeps one set of resident sessions on the device and one lease on
them. A call takes the lease in one step and uses the kept sessions; a call
that arrives while another holds it opens a private session set of its own,
proceeds, and closes it. Every round here starts its callers together and
checks that EVERY call completed with the tokens (or logits) of the same
call made alone, for the same seed: a kept session used by two holders at
once would mix their caches and change them. The take itself (never two
holders inside) is checked natively, checks/native_tests/causal_owner.mojo.

Run explicitly with MOJOLEARN_GENERATION_DEVICE=gpu (the resident sessions
are GPU state). Bounded fixtures; not a performance measurement.
"""
import os
import threading

import numpy as np
import pytest

from mojolearn.tests.test_resident_generation_contract import resident_model


pytestmark = pytest.mark.skipif(
    os.environ.get("MOJOLEARN_GENERATION_DEVICE") != "gpu",
    reason="explicit GPU resident-generation regression run required",
)

ROUNDS = 200
# One prompt per caller: callers that shared a kept session would show the
# other's tokens.
PROMPTS = [
    np.array([[1, 3], [2, 4]], dtype="i4"),
    np.array([[5, 7], [6, 8]], dtype="i4"),
    np.array([[9, 2], [3, 1]], dtype="i4"),
    np.array([[4, 4], [7, 5]], dtype="i4"),
]
TOKENS = 5


def _generate(model, index):
    return np.asarray(model.generate(PROMPTS[index], TOKENS, resident=True)).copy()


def _read(model, index):
    return np.asarray(model.stream(PROMPTS[index], TOKENS).read(TOKENS)).copy()


def _sampled_read(model, index):
    stream = model.stream(PROMPTS[index], TOKENS, greedy=False, seed=123 + index, top_k=4)
    return np.asarray(stream.read(TOKENS)).copy()


def _forward(model, index):
    prompt = PROMPTS[index]
    state = model.allocate_state(prompt.shape[0], prompt.shape[1] + TOKENS)
    return np.asarray(model.forward(prompt, state)).copy()


def _round(call, model, callers):
    start = threading.Barrier(callers)
    results = [None] * callers
    errors = [None] * callers

    def run(index):
        start.wait()
        try:
            results[index] = call(model, index)
        except Exception as error:  # reported by the assertion below
            errors[index] = error

    threads = [threading.Thread(target=run, args=(index,)) for index in range(callers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return results, errors


@pytest.mark.parametrize("callers", [2, 4])
@pytest.mark.parametrize("call", [_generate, _read, _sampled_read, _forward],
                         ids=["generate", "stream-read", "sampled-read", "forward"])
def test_every_concurrent_caller_completes_with_its_single_thread_output(tmp_path, call, callers):
    model = resident_model(tmp_path, uniform=False)
    references = [call(model, index) for index in range(callers)]
    assert model._resident_busy is False
    for _ in range(ROUNDS):
        results, errors = _round(call, model, callers)
        assert errors == [None] * callers, [str(error) for error in errors]
        for index in range(callers):
            np.testing.assert_array_equal(results[index], references[index])
        # Whoever held the lease gave it back before its call returned.
        assert model._resident_busy is False
    # The model answers as before, alone, and is not marked exposed.
    for index in range(callers):
        np.testing.assert_array_equal(call(model, index), references[index])
    assert model._resident_busy is False
    assert model._parameters_exposed is False
    if call is _generate or call is _forward:
        # The last call ran alone, took the lease and left its sessions kept.
        assert model._resident_sessions is not None


def test_exposed_model_runs_both_callers_on_private_sessions(tmp_path):
    model = resident_model(tmp_path, uniform=False)
    references = [_generate(model, index) for index in range(2)]
    assert model.parameters()
    assert model._parameters_exposed is True
    assert model._resident_sessions is None
    for _ in range(ROUNDS):
        results, errors = _round(_generate, model, 2)
        assert errors == [None, None], [str(error) for error in errors]
        for index in range(2):
            np.testing.assert_array_equal(results[index], references[index])
    assert model._resident_busy is False
    assert model._resident_sessions is None


def test_load_state_dict_drops_the_lease_with_the_weights(tmp_path):
    # A model that is not generating loads (the refusal reads the live
    # lease word); the kept sessions go with the replaced weights.
    model = resident_model(tmp_path, uniform=False)
    reference = _generate(model, 0)
    weights = model.state_dict()
    model.load_state_dict(weights)
    np.testing.assert_array_equal(_generate(model, 0), reference)
    assert model._resident_busy is False
