# SPDX-License-Identifier: Apache-2.0
"""NN31/NN32 native tape selection; scheduling never changes the arithmetic."""
from std.sys.compile import is_defined, get_defined_int
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

# L11 (2026-10-07): activation retention is ONE switch with arms,
# -D MOJOLEARN_IDN_ACT_RETAIN=1|2|3: 1 = NN32 retain every attention forward,
# 2 = NN31 budgeted checkpoints (default), 3 = NI48 owned Samba forward tapes
# (training/neural_identical_experiments.mojo, bindings/_mojolearn_mamba.mojo);
# -D MOJOLEARN_IDN_ACT_RETAIN_OFF = arm 0, replay. Scheduling only; no
# arithmetic changes.
# Arm 2 promoted 2026-10-08 (lane grid-act-4, IDENTICAL grid ge123e6f9, one run
# per arm, replay -> budgeted checkpoints ms): samba-train-step NV 96.0 -> 77.6,
# AMD 147.2 -> 144.3 (0.890x combined); output hashes equal to the incumbent on
# both vendors, no bit moves. The rule keeps a forward only when its actual
# retained bytes fit the budget and its replay work per byte clears the floor,
# so it holds for any shape that fits (no shape or dataset key). Measured alone;
# it now combines with the promoted resident Samba step and m3 angle carry
# cache, and the post-merge race measures the combination.
comptime IDN_ACT_RETAIN_ARM = 0 if is_defined["MOJOLEARN_IDN_ACT_RETAIN_OFF"]() else get_defined_int["MOJOLEARN_IDN_ACT_RETAIN", 2]()
comptime NN32_RETAIN_FORWARD = (GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and IDN_ACT_RETAIN_ARM == 1
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]())
comptime NN31_BOUNDED_CHECKPOINTS = (GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and IDN_ACT_RETAIN_ARM == 2
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]())


def attention_checkpoint_retain(bytes: Int, budget: Int, replay_ops: Int,
                                minimum_ops_per_byte: Int) -> Bool:
    """Actual retained storage bound and optional shape-derived replay cost.

    Input/weight snapshots are outside the activation budget. The current
    forward is transient even when its completed stages will be discarded.
    No dataset, exact shape, vendor or measured timing threshold is encoded.
    """
    if bytes <= 0 or bytes > budget:
        return False
    comptime if NN31_BOUNDED_CHECKPOINTS:
        return minimum_ops_per_byte <= 0 or replay_ops // bytes >= minimum_ops_per_byte
    comptime if NN32_RETAIN_FORWARD:
        return True
    return False
