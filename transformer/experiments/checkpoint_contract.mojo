# SPDX-License-Identifier: Apache-2.0
"""NN31/NN32 native tape selection; scheduling never changes the arithmetic."""
from std.sys.compile import is_defined, get_defined_int
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

# L11 (2026-10-07): activation retention is ONE switch with arms,
# -D MOJOLEARN_IDN_ACT_RETAIN=0|1|2|3: 0 replay (default), 1 = NN32 retain
# every attention forward, 2 = NN31 budgeted checkpoints, 3 = NI48 owned
# Samba forward tapes (training/neural_identical_experiments.mojo,
# bindings/_mojolearn_mamba.mojo). Scheduling only; no arithmetic changes.
comptime IDN_ACT_RETAIN_ARM = get_defined_int["MOJOLEARN_IDN_ACT_RETAIN", 0]()
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
