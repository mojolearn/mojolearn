# SPDX-License-Identifier: Apache-2.0
"""Two threads generating on ONE CausalLM model.

The model keeps one set of resident sessions on the device and one lease on
them. A call takes the lease in one step; a call that arrives while another
holds it is refused in words at that moment and touches nothing. Every round
here starts its callers together and checks that each call either returned
the single-thread tokens or was refused with those words, that at least one
call per round proceeded, and that the model works afterwards.

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

REFUSAL = "transformer decode session: busy"
ROUNDS = 200
PROMPT = np.array([[1, 3], [2, 4]], dtype="i4")
TOKENS = 5


def _generate(model):
    return np.asarray(model.generate(PROMPT, TOKENS, resident=True)).copy()


def _read(model):
    return np.asarray(model.stream(PROMPT, TOKENS).read(TOKENS)).copy()


def _forward(model):
    state = model.allocate_state(PROMPT.shape[0], PROMPT.shape[1] + TOKENS)
    return np.asarray(model.forward(PROMPT, state)).copy()


def _round(call, model, callers):
    start = threading.Barrier(callers)
    results = [None] * callers
    errors = [None] * callers

    def run(index):
        start.wait()
        try:
            results[index] = call(model)
        except Exception as error:  # the refusal is the subject of the test
            errors[index] = error

    threads = [threading.Thread(target=run, args=(index,)) for index in range(callers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return results, errors


@pytest.mark.parametrize("callers", [2, 4])
@pytest.mark.parametrize("call", [_generate, _read, _forward], ids=["generate", "stream-read", "forward"])
def test_one_caller_proceeds_and_the_other_is_refused_in_words(tmp_path, call, callers):
    model = resident_model(tmp_path, uniform=False)
    reference = call(model)
    assert model._resident_busy is False
    refused = 0
    for _ in range(ROUNDS):
        results, errors = _round(call, model, callers)
        proceeded = [value for value in results if value is not None]
        failed = [error for error in errors if error is not None]
        assert len(proceeded) + len(failed) == callers
        # Somebody held the lease, so somebody proceeded.
        assert proceeded, [str(error) for error in failed]
        for value in proceeded:
            np.testing.assert_array_equal(value, reference)
        for error in failed:
            assert REFUSAL in str(error), str(error)
        refused += len(failed)
        # The last holder gave the lease back before its call returned.
        assert model._resident_busy is False
    # No corruption: the model answers as before, and keeps its sessions.
    np.testing.assert_array_equal(call(model), reference)
    assert model._resident_busy is False
    assert model._parameters_exposed is False
    if call is not _read:
        assert model._resident_sessions is not None
    if refused == 0:
        pytest.skip("the callers never overlapped in %d rounds; the take itself is checked natively "
                    "(checks/native_tests/causal_owner.mojo)" % ROUNDS)


def test_exposed_model_runs_both_callers_on_private_sessions(tmp_path):
    model = resident_model(tmp_path, uniform=False)
    reference = _generate(model)
    assert model.parameters()
    assert model._parameters_exposed is True
    assert model._resident_sessions is None
    for _ in range(ROUNDS):
        results, errors = _round(_generate, model, 2)
        assert errors == [None, None], [str(error) for error in errors]
        for value in results:
            np.testing.assert_array_equal(value, reference)
    assert model._resident_busy is False
    assert model._resident_sessions is None


def test_load_state_dict_drops_the_lease_with_the_weights(tmp_path):
    # A model that is not generating loads (the refusal reads the live
    # lease word); the kept sessions go with the replaced weights.
    model = resident_model(tmp_path, uniform=False)
    reference = _generate(model)
    weights = model.state_dict()
    model.load_state_dict(weights)
    np.testing.assert_array_equal(_generate(model), reference)
    assert model._resident_busy is False
