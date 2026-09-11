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

THE PARTICIPATION REQUIREMENT, which is the other half of why `barrier()` may
stand here. Every call in these families is UNCONDITIONAL, so what must be
block-uniform is the trip count of every loop around one. The striped loop's
is (`requires_uniform_iteration_for`, and the `max_iters` derivation each
kernel carries beside it).

THE HEAD/TAIL PEEL'S WAS NOT, UNTIL DEVIATION 2600 (2026-09-11). It copied
CatBoost's `for (idx = tid; idx < alignSize; idx += BlockSize)`, which gives a
thread with `tid >= alignSize` NO trip. `alignSize` is 128 (half-byte) or 256
(binary, hist_2) against a 512-thread block, so threads 0..127 or 0..255 ran
the peel's `AddPoint` syncs and the rest of the block did not. Their sync is
warp-local and `alignSize` is a multiple of the warp, so every warp stays
uniform and nothing is wrong on a column where `turn_sync` is `syncwarp`. On
the 64-lane AMD column `turn_sync` is this `barrier()`, a barrier part of the
block skips. The IDENTICAL binary, half-byte, 5-bit and 6-bit kernels carry
it (the 7-bit and 8-bit IDENTICAL arms issue no turn sync, and the one-byte
PASS family's shared-Int32 block equals its `alignSize`). Each peel now runs
to `PEEL_END`, `alignSize` rounded up to the block, so every thread makes the
same trips. A trip at or past `alignSize` fails both load guards (`head_len`
and `tail_len` never exceed `alignSize`) and adds a zero point into cells the
kernel zeroed a statement earlier, so no histogram bit moves on any column.

MEASURED 2026-09-11. Hot Aisle MI300X (gfx942, 64 lanes), the same source
built with and without the bound: without it the half-byte, 5-bit and 6-bit
fits moved between two fits in one process and the binary fit gave one wrong
answer every time; the stage trace first parted at `tree000.depth02.hist`;
with it `checks/gbdt_sub_byte_identity_check.py` read 16/16 against the H100
and taxi 1M symmetric held the H100's hash for ten rounds. Restoring the old
bound in one kernel file failed exactly that file's fixtures. On an H100 the
two builds gave the same bits in every cell.

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
