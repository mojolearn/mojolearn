# SPDX-License-Identifier: Apache-2.0
"""RUN half of the byte LM `_state` admit-once record (lane rs-neural, wave 8
run split, from bindings/byte_lm_state_admission.mojo): the names of the
step fields that every step, session adoption or load rewrites. Typed Mojo
only, no Python import (tools/python_free_dirs.sh). Same bodies."""


def _byte_state_step_fields() -> List[String]:
    # The fields every step, session adoption or load rewrites.
    return [String("parameters"), String("m"), String("v"), String("flags"), String("completed_steps"),
            String("next_batch_index")]


def _byte_state_is_step_field(name: String) -> Bool:
    var steps = _byte_state_step_fields()
    for i in range(len(steps)):  # small-loop(the six step fields)
        if steps[i] == name:
            return True
    return False
