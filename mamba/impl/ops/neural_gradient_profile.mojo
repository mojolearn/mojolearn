# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""NN39 Mamba-2 parameter-gradient fold v2; source-only opt-in profile.

Existing fixed 256-row leaves are unchanged. Merge adjacent equal-size
leaf groups as they become complete, then drain occupied binary slots low
to high (older left subtree + newer right subtree). An odd subtree is
carried, never padded by a fabricated +0. A single leaf is returned
verbatim. The only internal arithmetic is ftz(left + ftz(right)), matching
on host/NVIDIA/AMD/Apple. No real-number associativity claim is a bit proof.
Backward quality and cross-column execution have NOT been checked.
"""
from std.memory import stack_allocation
from std.sys.compile import is_defined
from checks.numerics import ftz, GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

comptime NN39_M2_GRAD_TREE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and is_defined["MOJOLEARN_NN39_M2_GRAD_TREE"]()
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)


@always_inline
def nn39_gradient_tree(part: MutPointer[Float32, MutAnyOrigin], tiles: Int, cols: Int, c: Int) -> Float32:
    # 32 levels cover every positive Int32 tile count used by the caller.
    var slots = stack_allocation[32, Float32]()
    var occupied = 0
    for tile in range(tiles):
        var carry = part.unsafe_load(tile * cols + c)
        var level = 0
        while (occupied & (1 << level)) != 0:
            carry = ftz(slots[level] + ftz(carry))
            level += 1
        slots[level] = carry
        occupied += 1
    var acc = Float32(0.0)
    var have = False
    for level in range(32):
        if (occupied & (1 << level)) != 0:
            if have:
                acc = ftz(slots[level] + ftz(acc))
            else:
                acc = slots[level]
                have = True
    return acc
