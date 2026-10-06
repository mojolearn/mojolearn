# SPDX-License-Identifier: Apache-2.0
"""NN31/NN32 native tape selection; scheduling never changes the arithmetic."""
from std.sys.compile import is_defined
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

comptime NN32_RETAIN_FORWARD = (GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and is_defined["MOJOLEARN_NN32_RETAIN_FORWARD"]()
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]())
comptime NN31_BOUNDED_CHECKPOINTS = (GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and is_defined["MOJOLEARN_NN31_BOUNDED_CHECKPOINTS"]()
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
