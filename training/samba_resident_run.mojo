# SPDX-License-Identifier: Apache-2.0
"""RUN half of the resident Samba step door (lane rs-neural, wave 8 run
split, from providers/python/samba_resident_pyapi.mojo): the integer words
of one resident train step (reserved validation slot, shape, accumulation
and the typed optimizer plan). Typed Mojo only, no Python import
(tools/python_free_dirs.sh). Same words in the same order."""
from core.samba_optimizer_plan import SambaOptimizerPlan


def samba_resident_step_words(h: Int, rows: Int, length: Int, accumulation: Int, total_count: Int,
                              plan: SambaOptimizerPlan) -> List[Int]:
    var p: List[Int] = [h, rows, length]
    if accumulation > 1:
        p.append(accumulation)
        p.append(total_count)
    p.append(plan.tensors)
    p.append(plan.hyper.kind)
    p.append(plan.step)
    p.append(plan.hyper.nesterov)
    return p^
