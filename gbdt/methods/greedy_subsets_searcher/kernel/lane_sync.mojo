# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The turn-taking sync the CatBoost histogram accumulators need, in ONE
place, keyed by the matrix row that owns it.

WHAT THIS REPLACES. Six kernels in this directory each wrote `syncwarp()`
inline where CatBoost writes `tiled_partition<8>::sync()` or
`tiled_partition<32>::sync()`, each with its own copy of the argument for why
a 32-lane sync is a legal widening of an 8-lane one. That argument is sound
on a machine whose wave IS 32 lanes and is exactly what fails to transfer to
a 64-wide wavefront or an 8-wide one, so it belongs in the table and not in
six comments. `sub_byte_lane_sync_for` is the row; this file is its only
consumer.

THE SUBSTITUTION IS SCHEDULING, NOT ARITHMETIC. Widening a sync changes which
threads wait, never what is added to what. Within one 8-lane tile the lanes
hold eight distinct values of `tid & 7`, so on any one iteration they touch
eight distinct slots; two tiles never share a slot (`tid & 24` gives each its
own eighth of the 32 floats a bin spans); two logical 32-groups never share a
replica (`512 * (tid // 32)`, or `1024 * (tid // 32)` in the one-byte
ladder). So for any given slot the sequence of adds is program order inside
one tile, whatever the barrier's width, and the histogram is bit-for-bit the
same either way. That is what makes this a per-column row at all.

THE PARTICIPATION REQUIREMENT IS ALREADY MET, which is the other half of why
`barrier()` may stand here. Every call in these families is UNCONDITIONAL
inside a loop whose trip count is block-uniform by construction
(`requires_uniform_iteration_for`, and the `max_iters` derivation each kernel
carries beside it), so no thread can skip a barrier its neighbours reach.
That property was bought for a different reason -- CatBoost's warp-local sync
is what our threadgroup barrier could not be -- and it is what lets this file
take a threadgroup barrier without a second thought.

THE COLUMNS. `apple`, `nvidia`, `amd-rdna` and the identity column are
exactly 32 lanes wide and keep `syncwarp`, byte for byte and cycle for cycle.
`amd` (CDNA, 64) and the variable-width columns take `barrier()`. See
`sub_byte_lane_sync_for` for the full argument, including the part that could
NOT be established by reading: what `max.gpu.sync.syncwarp` lowers to on a
wave that is not 32 wide.
"""

from max.gpu.sync import barrier, syncwarp

from checks.kernel_matrix import (
    SYNC_LANE,
    TARGET_COLUMN,
    sub_byte_lane_sync_for,
)


#: The row, resolved once for this build. `SYNC_LANE` on a 32-lane column,
#: `SYNC_BLOCK` on every other.
comptime TURN_SYNC = sub_byte_lane_sync_for[TARGET_COLUMN]()

#: Comptime-visible spelling of the same answer, for a kernel that wants to
#: say in its own source which barrier it compiled with.
comptime TURN_SYNC_IS_LANE = TURN_SYNC == SYNC_LANE


@always_inline
def turn_sync():
    """One turn of the accumulator's write-turn sync.

    `syncwarp()` where the hardware wave is the logical replication group;
    `barrier()` everywhere else. The branch is comptime: exactly one of the
    two is emitted, and on every column that has ever run a histogram it is
    the same `syncwarp` that was there before.
    """

    @parameter
    if TURN_SYNC_IS_LANE:
        syncwarp()
    else:
        barrier()
