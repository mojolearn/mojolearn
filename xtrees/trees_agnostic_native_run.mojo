# SPDX-License-Identifier: Apache-2.0
"""Typed parts of the model-agnostic explainer door (RUN): the explainer name, the per-binary
switch-bit cache keyed on the compiled entry address, and the rows-per-chunk budget."""
from core.shared_slot import SharedSlot


#: xtrees/agnostic.mojo AGN_BUDGET: synthetic words per explainer chunk.
comptime _AGN_BUDGET = 1 << 25


def _name(permutation: Bool) -> String:
    return "PermutationExplainer" if permutation else "KernelExplainer"


struct _SwitchFacts(Defaultable, Movable):
    """`x_trees_fast_switches()` of each loaded x_trees binary, keyed on the
    compiled entry's address (its connected table holds it for the life of
    the process). The bits are compile-time constants; every shap_values
    call (twice on the permutation path) called the entry again (N252,
    ne1-a-trees, 2026-11-06)."""
    var answers: Dict[Int, Int]

    def __init__(out self):
        self.answers = Dict[Int, Int]()


def _chunk(per_row: Int, n: Int) -> Int:
    """Rows per chunk: as many as keep a chunk's synthetic matrix under the
    budget (xtrees/agnostic.mojo AGN_BUDGET words), at least one."""
    return max(1, min(n, _AGN_BUDGET // max(1, per_row)))


comptime _SWITCH_FACTS = SharedSlot[_SwitchFacts, "MojolearnTreesAgnosticSwitchFactsV2"]
