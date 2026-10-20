# SPDX-License-Identifier: Apache-2.0
"""Native Generator state boundaries, including the last usable stream."""
import pytest

from mojolearn import _training_impl as training
from mojolearn._verify_native_runtime import audit_native_call


def test_generator_last_stream_and_exhausted_restore_are_native():
    gen = training.Generator.__new__(training.Generator)
    audit_native_call(gen.__init__, (1 << 64) - 1).assert_native()
    audit_native_call(gen.load_state_dict,
                      {"seed": 7, "counter": (1 << 32) - 1}).assert_native()
    assert audit_native_call(gen.next_stream).assert_native() == (1 << 32) - 1
    exhausted = audit_native_call(gen.state_dict).assert_native()
    assert exhausted == {"seed": 7, "counter": 1 << 32}
    restored = training.Generator(1)
    audit_native_call(restored.load_state_dict, exhausted).assert_native()
    for value in (gen, restored):
        result = audit_native_call(value.next_stream)
        assert isinstance(result.exception, RuntimeError), result.report()
        assert "exhausted" in str(result.exception)
        for key in ("python_frames", "forbidden_native_calls", "wrapper_work",
                    "unobserved_existing_threads"):
            assert not result.report()[key], result.report()
        assert value.state_dict() == exhausted


@pytest.mark.parametrize("state", [
    {"seed": -1, "counter": 0},
    {"seed": 1 << 64, "counter": 0},
    {"seed": 3, "counter": -1},
    {"seed": 3, "counter": (1 << 32) + 1},
])
def test_generator_invalid_restore_preserves_existing_state(state):
    gen = training.Generator(123)
    gen.next_stream()
    before = gen.state_dict()
    result = audit_native_call(gen.load_state_dict, state)
    assert isinstance(result.exception, ValueError), result.report()
    for key in ("python_frames", "forbidden_native_calls", "wrapper_work",
                "unobserved_existing_threads"):
        assert not result.report()[key], result.report()
    assert gen.state_dict() == before
