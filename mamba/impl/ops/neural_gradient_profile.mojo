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
from checks.numerics import ftz

# L11 (2026-10-07): the Mamba-2 gradient fold is v1 (256-row leaves,
# ascending tile merge). The NN39 tile tree and NI44 128-row leaves arms are
# deleted; the two aliases keep their names as constants for
# autograd/kernels/mamba2_ssd.mojo, which imports them.
# TOMBSTONE: MOJOLEARN_IDN_M2_GRAD_FOLD (unmeasured opt-in arms 1 (NN39 tile tree) and 2 (NI44 128-row leaves) of the Mamba-2 gradient fold) deleted 2026-11-15 by lane/expdel2-28-neural; code recoverable at 92dbaa2ecb.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_M2_GRAD_FOLD.patch; record in docs/TOMBSTONES.md.
comptime IDN_M2_GRAD_FOLD_ARM = 0
comptime NN39_M2_GRAD_TREE = False


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
