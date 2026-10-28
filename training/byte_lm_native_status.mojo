# SPDX-License-Identifier: Apache-2.0
# One native snapshot of existing resident status and attention-stage metadata.
# No launches, downloads or synchronization occur here.
from training.byte_lm_binding_impl import ByteLMSession
from training.byte_lm import byte_attention_eager_cells


def byte_lm_status(owner: ByteLMSession) raises -> List[Int]:
    var result: List[Int] = [
        owner.trainer.value().completed_steps if owner.trainer else -1,
        owner.trainer.value().grad_step if owner.trainer else -1,
        1 if owner.usable else 0,
        1 if owner.trainer else 0,
    ]
    var eager = List[Int](length=6, fill=0)
    if owner.trainer:
        eager = byte_attention_eager_cells(owner.trainer.value())
    for value in eager:  # driver-loop: over existing session stage metadata
        result.append(value)
    return result^
