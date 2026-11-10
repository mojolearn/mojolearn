"""External mapping admission preserves order without a Python key-list copy."""
from collections import OrderedDict

import numpy as np
import pytest

from mojolearn import _training_impl as T
from mojolearn._verify_native_runtime import audit_native_call


@pytest.mark.parametrize("move_key,last_error", [
    (False, "state requires native string keys"),
    (True, "requires native buffers and lists/tuples"),
])
def test_ordered_checkpoint_admission_preserves_reordered_refusal(move_key, last_error):
    class Callback:
        def __iter__(self):
            raise AssertionError("checkpoint admission entered a Python iterator")

    owner = T.Adam(np.ones(3, dtype="f4"), numeric_mode="identical", resident=False)
    state = OrderedDict([(1, None), ("callback", Callback())])
    if move_key:
        state.move_to_end(1)
    original_params, original_step = owner.params, owner.t
    result = audit_native_call(owner.load_state_dict, state)
    assert type(result.exception) is TypeError, result.report()
    assert last_error in str(result.exception), result.report()
    for field in ("python_frames", "forbidden_native_calls", "wrapper_work", "unobserved_existing_threads"):
        assert not result.report()[field], result.report()
    assert owner.params is original_params and owner.t == original_step
