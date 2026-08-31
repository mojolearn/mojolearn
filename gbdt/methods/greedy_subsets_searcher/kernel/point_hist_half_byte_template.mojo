# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The conflict-free shared-memory histogram accumulator.

PORT OF `catboost/cuda/methods/greedy_subsets_searcher/kernel/
point_hist_half_byte_template.cuh` at CatBoost `54a8143a`.
Transliterated. Do not improve.

**This is the file the whole experiment is about.** It is how CatBoost
accumulates a histogram on the GPU with NO ATOMICS in the inner loop, and it
is the thing mojotrees does not do.

The mechanism, in their words rearranged:

- The threadgroup holds `BlockSize * 16` floats of scratch.
- `SliceOffset()` = `512 * (tid / 32) + (tid & 24)` gives every 32-lane warp
  its own 512-float copy, and within a warp gives each group of 8 lanes its
  own sub-copy at offset 0, 8, 16 or 24.
- `AddPoint` loops `i` over 0..7 and has lane `tid` handle feature
  `f = (tid + i) & 7`. Because the 8 lanes of a tile hold 8 distinct values
  of `tid & 7`, the rotation is a permutation: within one iteration the 8
  lanes touch 8 DISTINCT features, hence 8 distinct slots. No two lanes
  collide, so the update is a plain `Histogram[bin] += t`.
- The slot is `bin = ((ci >> (28 - 4*f)) & 15) << 5 | f`. The `<< 5` spaces
  consecutive bins 32 floats apart so that the 8 lanes' writes fall in
  different banks.

`ci` is one `UInt32` of the compressed index holding EIGHT features at 4 bits
each, so one load feeds eight histogram updates. That is the read-density
win: mojotrees issues eight one-byte loads for the same work.

DEVIATION (PORTING.md 1): `BLOCK_SIZE` is 512 rather than their 768 because
Apple caps threadgroup memory at 32 KB and this accumulator wants 16 floats
per thread, so 32,768 / (16 * 4) = 512 threads is the largest block that
fits. Their `tiled_partition<8>::sync()` becomes `syncwarp()`, a 32-lane
sync: it orders a SUPERSET of the 8 lanes theirs orders, so it is correct,
and it is the narrowest sync Mojo 1.0 exposes. Warp SHUFFLES are what Mojo
lacks, not warp SYNC.
"""

from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.memory import AddressSpace
from max.gpu.sync import syncwarp

from mojo_only.kernel_matrix import (
    lane_width_for,
    K_HIST_HALF_BYTE,
    TARGET_COLUMN,
    block_size_for,
    hist_floats_per_thread_for,
)
from mojo_only.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_FAST,
    NUMERIC_IDENTICAL,
    ftz,
)


#: READ FROM THE MATRIX, not restated here. `mojo_only/kernel_matrix.mojo`
#: owns every knob that is a BUDGET, and a kernel that hardcodes one makes
#: the table decoration. CatBoost uses 768; Apple's 32 KB ceiling over 16
#: floats per thread yields 512, which is exactly 32,768 bytes.
comptime BLOCK_SIZE = block_size_for[K_HIST_HALF_BYTE, TARGET_COLUMN]()


#: The mode this build compiles against. See `mojo_only/numerics.mojo`. FAST
#: is the default, and under FAST the flush is CatBoost's own float
#: `atomicAdd` (`hist_half_byte.cu:45-51`) on every vendor Mojo builds for.
#: The fixed-point Int32 accumulator is what IDENTICAL selects, and it is
#: dead code in this build.
#:
#: "No vendor forces the deterministic row" stood here and is deleted rather
#: than annotated (2026-08-21): the `qualcomm` column and the portable
#: baseline have no CORE float atomic -- it arrives through an optional
#: extension -- so `deterministic_flush_for` returns True for them in BOTH
#: modes. That row is a vendor override now, not only a mode's choice.
#:
#: WORTH KNOWING WHERE THIS FILE SITS IN THE IDENTITY ARGUMENT. The flush is
#: fixed point under IDENTICAL, but the SHARED-MEMORY accumulation in this
#: family is float (`smem[slot] = smem[slot] + stat`, below), unlike the
#: hist_2 family's shared Int32. So for THIS family the block size is a
#: NUMERIC row -- it sets how many private slices exist, hence which floats
#: add to which -- and that is what the identity floor's 32 KB is protecting.
#: On the hist_2 path the same knob is pure scheduling, because integer
#: addition is associative. Two families, two answers, one table.
comptime BUILD_MODE = GLOBAL_NUMERIC_MODE

#: Lanes moving in lockstep. READ FROM THE MATRIX, not pinned here.
#:
#: This used to be a literal 32 in this file and in three others, with a
#: comment pointing at `kernel_matrix.column_lane_width` for why AMD's 64
#: must not reach it. That is a matrix row that EXISTS and was bypassed:
#: `column_lane_width` had fifteen call sites and every one of them was
#: inside the matrix itself or the table printer, so changing
#: `TARGET_COLUMN` to AMD would have compiled and silently kept 32 while the
#: replication geometry assumed a 32-wide slice on a 64-wide wavefront.
#: One number now flows through, which is the point of having the table.
comptime LANE_WIDTH = lane_width_for[
    TARGET_COLUMN, BUILD_MODE == NUMERIC_IDENTICAL
]()

#: `Reduce()`'s stage-1 width (`point_hist_half_byte_template.cuh:115-134`).
#:
#: THE DERIVATION, because the old one was a coincidence. Their literal 512
#: is NOT "512 because BlockSize is 768". It is the size of ONE WARP'S
#: PRIVATE REPLICA, established by `SliceOffset()` at
#: `point_hist_half_byte_template.cuh:48-52`:
#:
#:     const int warpOffset = 512 * (threadIdx.x / 32);
#:
#: i.e. `LANE_WIDTH * 16`, lanes per warp times floats per thread. Folding
#: the scratch at exactly that stride is what sums a bin ACROSS the warps'
#: private copies while leaving the 512-float layout of a single copy
#: standing for stage 2 to un-scramble. Any other width sums the wrong
#: cells, so this is a NUMERIC quantity and not a tunable one.
#:
#: It used to be read from `mojo_only/kernel_matrix.mojo`, whose
#: `reduce_width_for` returns the BLOCK SIZE under `NUMERIC_FAST`. That is
#: 512 on Apple by accident and 768 on CatBoost's own configuration, where
#: it would be wrong. The block size does not belong in this quantity, so
#: the matrix is no longer consulted for it.
#:
#: Stage 1 strides `slot_i += BLOCK_SIZE` while `slot_i < REDUCE_WIDTH`, and
#: it carries barriers, so BLOCK_SIZE must divide or equal REDUCE_WIDTH for
#: the trip count to stay uniform. At 512 and 512 every thread makes exactly
#: one trip.
comptime REDUCE_WIDTH = LANE_WIDTH * hist_floats_per_thread_for[
    K_HIST_HALF_BYTE
]()

#: `GetHistSize()`, `BlockSize * 16`, with the 16 from the matrix.
comptime HIST_SIZE = BLOCK_SIZE * hist_floats_per_thread_for[
    K_HIST_HALF_BYTE
]()


def slice_offset(tid: Int) -> Int:
    """`TPointHistHalfByteBase::SliceOffset()`, copied exactly.

        const int warpOffset = 512 * (threadIdx.x / 32);
        const int innerHistStart = threadIdx.x & 24;
        return warpOffset + innerHistStart;

    `& 24` keeps bits 3 and 4, giving 0, 8, 16 or 24: four sub-copies inside
    each warp's 512-float region. Do not "simplify" it to `& 31`; the low
    three bits are the lane's position inside the 8-lane tile and are carried
    by `f` instead.

    THE 512 AND THE 32 ARE THEIRS, TRANSCRIBED, AND THEY ARE 32-LANE
    ---------------------------------------------------------------
    512 is not `LANE_WIDTH * 16` that happens to be right; it is 16 bins of
    32 slots, and the 32 is 8 features times the 4 sub-copies a 32-lane warp
    is cut into. `add_point_slot` spaces bins by `<< 5`, so the layout has
    room for exactly four 8-lane tiles. A 64-lane wavefront would need eight,
    and two of them would collide on every bin.

    So this geometry does not generalise, CatBoost never had to make it, and
    guessing at it is inventing. The assert below turns the AMD case into a
    COMPILE ERROR instead of a silently mis-folded histogram: `REDUCE_WIDTH`
    is derived from `LANE_WIDTH` and would move to 1024 while these two
    literals stayed at 512 and 32, and stage 1 would then fold cells that
    belong to different bins.

    A 64-lane FAST column never instantiates this template:
    `greedy_sub_byte_excluded_for` (DEVIATION 1910) comptime-excludes the
    binary and half-byte launch sites, which refuse at runtime by name --
    they cannot take the one-byte route, their cindex packing is 8 and 32
    features per word against the fused kernel's 4. The assert stays for
    whatever reaches this file some other way.
    """
    comptime assert LANE_WIDTH == 32, (
        "the half-byte accumulator's slice layout is 32-lane by construction"
        " (`point_hist_half_byte_template.cuh:48-52` plus the `<< 5` bin"
        " spacing); write the wide-wavefront layout before letting"
        " LANE_WIDTH be 64"
    )
    return 512 * (tid // 32) + (tid & 24)


def add_point_slot(ci: UInt32, tid: Int, i: Int) -> Int:
    """The slot `AddPoint` writes on iteration `i`, as pure arithmetic.

        const int f = (threadIdx.x + i) & 7;
        int bin = (ci >> (28 - 4 * f)) & 15;
        bin <<= 5;
        bin += f;

    CatBoost keeps this inside `AddPoint`. Here the ARITHMETIC is factored
    out so that `AddPoint` (below) and `AddPointsImpl`'s inner batch loop,
    which is inlined at each call site, can share one copy of it. Same
    operations, same order.

    Why the slot is collision-free: the 8 lanes of a tile hold 8 distinct
    values of `tid & 7`, so `(tid + i) & 7` is a permutation of them and the
    8 lanes touch 8 distinct features on every iteration. `bin << 5` then
    spaces consecutive bins 32 floats apart so those 8 writes fall in
    different banks.
    """
    var f = (tid + i) & 7
    var bin = Int((ci >> UInt32(28 - 4 * f)) & UInt32(15))
    bin <<= 5
    bin += f
    return bin


def add_half_byte_point(
    ci: UInt32,
    stat: Float32,
    tid: Int,
    slice_base: Int,
    smem: UnsafePointer[
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
        origin=MutUntrackedOrigin,
    ],
):
    """`TPointHistHalfByteBase::AddPoint`, ONE point, as a callable
    (`point_hist_half_byte_template.cuh:65-77`):

        thread_block_tile<8> addToHistTile = tiled_partition<8>(...);
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const int f = (threadIdx.x + i) & 7;
            int bin = (ci >> (28 - 4 * f)) & 15;
            bin <<= 5;
            bin += f;
            Histogram[bin] += t;
            addToHistTile.sync();
        }

    Their `AddPoint` is a method, so `AlignMemoryAccess`
    (`compute_hist_loop_one_stat.cuh:81`, `:129`) calls the SAME code the
    main loop does. Ours used to be inlined in the loop only, which is fine
    right up until the head/tail peel needs it too; duplicating a
    sync-carrying accumulation in four places is how the four copies drift.

    DEVIATION BLOCK
    ---------------
    THEIRS: `addToHistTile.sync()` on a `tiled_partition<8>`.
    OURS:   `syncwarp()`, 32 lanes.
    WHY:    Mojo 1.0 exposes no 8-lane tile. 32 lanes is a SUPERSET of the 8
            theirs orders, so the ordering guarantee is strictly stronger and
            the result is unchanged. It is warp-local either way, so callers
            need a uniform trip count per WARP only, not per block.
    """

    @parameter
    for i in range(8):
        var slot = slice_base + add_point_slot(ci, tid, i)
        # IDENTITY_PATHS ROW 10: this family accumulates in FLOAT in both
        # modes (unlike hist_2's Int32), and a running cell can pass
        # through the denormal range under cancellation -- flushed on
        # Metal's hardware, kept on CUDA's default -- so each stored
        # intermediate goes through `ftz`. Comptime no-op under FAST.
        smem[slot] = ftz(smem[slot] + stat)
        syncwarp()


def reduce_stage2_slot(tid: Int, group: Int) -> Int:
    """`Reduce()` stage 2's read offset: `32 * fold + featureId + 8 * group`
    for `fold = (tid >> 3) & 15` and `featureId = tid & 7`."""
    var fold = (tid >> 3) & 15
    var feature_id = tid & 7
    return 32 * fold + feature_id + 8 * group
