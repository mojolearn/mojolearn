# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The 5-bit specialization of the fused two-stat one-byte accumulator.

PORT OF `hist_2_one_byte_5bit.cu` at CatBoost `54a8143a`,
`TPointHist2OneByte<5, BlockSize>`. Transliterated. Do not improve.

This is the family CatBoost's own one-byte dispatch takes at
`maxBins <= 32` (`hist_one_byte.cu:315-316`), NOT the `TPointHistOneByte`
PASS family this repository ported first. The two differ in everything but
the writeback: this one processes TWO stat columns per pass, keys its slot
parity on the stat, and pays for the second stat with a second warp-local
sync phase instead of a second launch.

THE PARITY TRICK, which is the whole file
-----------------------------------------
`flag = threadIdx.x & 1` swaps which stat a lane calls "stat1"
(`hist_2_one_byte_5bit.cu:41-50`), and the two write phases store at
`offset + flag` and `offset + !flag`. So at any instant the EVEN lanes and
the ODD lanes are writing OPPOSITE stat parities, every lane of the warp
writes a distinct slot, and one warp-local sync between the phases is the
only ordering needed. The net effect is `stat1 -> even slot, stat2 -> odd
slot` for every lane, which is what `Reduce` unscrambles by `threadIdx.x & 1`.

Slot layout per warp (1024 floats): `innerHistStart (tid & 24, four 8-slot
sub-copies) + f (0/2/4/6, the feature) + flag (the stat) + 32 * (bin & 31)`.

DEVIATION (sync width): their `AddPointsImpl` syncs a `tiled_partition<8>`
(`hist_2_one_byte_5bit.cu:39`, `:66`, `:76`). Mojo 1.0 exposes `syncwarp`
(32 lanes) and `barrier()` only, so the sync is WIDENED to the warp, exactly
as `hist_one_byte.mojo` widens their `tiled_partition<32>`'s siblings. Every
lane executes the same unconditional sync count, so the widening is legal;
it is stricter than theirs, never looser, and it changes no sum: the slot
map above gives every lane of the warp a distinct slot within a phase, so
serializing more lanes than necessary reorders nothing.
"""

from max.gpu.memory import AddressSpace
from gbdt.methods.greedy_subsets_searcher.kernel.lane_sync import turn_sync

#: DEVIATION 1947, 2026-09-01. `syncwarp()` stood at each of the write-turn
#: sites below and is now `turn_sync()`, which IS `syncwarp()` on a column
#: whose hardware wave is 32 lanes -- Apple, NVIDIA, RDNA and the identity
#: column, byte for byte and instruction for instruction -- and a threadgroup
#: `barrier()` on any other width. The reason is not the slice layout, which
#: is a logical partition of thread indices and always travelled; it is that
#: what `syncwarp` EMITS on a 64-wide wavefront cannot be established by
#: reading anything in this tree, and these turns are what keeps two PLAIN
#: float adds off one slot. See `sub_byte_lane_sync_for`.

from gbdt.methods.greedy_subsets_searcher.kernel.histogram_utils import (
    hist2_smem_add,
)


def hist2_slice_offset_5(tid: Int) -> Int:
    """`TPointHist2OneByte<5>::SliceOffset()` (`hist_2_one_byte_5bit.cu:25-31`).

        const int warpId = (threadIdx.x / 32);
        const int warpOffset = 1024 * warpId;
        const int blocks = 4;
        const int innerHistStart = (threadIdx.x & ((blocks - 1) << 3));
        return warpOffset + innerHistStart;
    """
    var warp_offset = 1024 * (tid // 32)
    var inner_hist_start = tid & ((4 - 1) << 3)
    return warp_offset + inner_hist_start


def hist2_add_points_5[
    n: Int, dt: DType
](
    ci: InlineArray[UInt32, n],
    s1: InlineArray[Float32, n],
    s2: InlineArray[Float32, n],
    q1: InlineArray[Int32, n],
    q2: InlineArray[Int32, n],
    tid: Int,
    slice_base: Int,
    smem: UnsafePointer[
        Scalar[dt],
        address_space = AddressSpace.SHARED,
        origin=MutUntrackedOrigin,
    ],
):
    """`TPointHist2OneByte<5>::AddPointsImpl<N>`
    (`hist_2_one_byte_5bit.cu:34-87`), branch for branch.

    `pass[k] = bin != 32` is theirs and it is EXACTLY 32, not `bin > 31`:
    32 is the one encodable value past a full 5-bit feature, so it is the
    skip mark, while every smaller bin is real. Copied, not generalized.

    `dt`, `q1` and `q2` exist for the `HIST_SMEM_SHARED2_I32` matrix row
    (`q1`/`q2` are the batch's stats through `hist2_quantize`, computed once
    per row at load time):
    the pass structure, bin decode, parity trick and sync phases are theirs
    in both modes, and WHERE the add lands is decided once in
    `hist2_smem_add` (see its DEVIATION BLOCK for the probe numbers).
    """
    var flag = tid & 1

    # stat1[k] = flag ? s2[k] : s1[k];  stat2[k] = flag ? s1[k] : s2[k];
    # The pre-quantized pair rides the same swap.
    var stat1 = InlineArray[Float32, n](fill=Float32(0.0))
    var stat2 = InlineArray[Float32, n](fill=Float32(0.0))
    var qstat1 = InlineArray[Int32, n](fill=Int32(0))
    var qstat2 = InlineArray[Int32, n](fill=Int32(0))

    @parameter
    for k in range(n):
        if flag == 1:
            stat1[k] = s2[k]
            stat2[k] = s1[k]
            qstat1[k] = q2[k]
            qstat2[k] = q1[k]
        else:
            stat1[k] = s1[k]
            stat2[k] = s2[k]
            qstat1[k] = q1[k]
            qstat2[k] = q2[k]

    @parameter
    for i in range(4):
        var f = (2 * i + tid) & 6

        var offsets = InlineArray[Int, n](fill=0)
        var keep = InlineArray[Bool, n](fill=False)

        @parameter
        for k in range(n):
            var bin = Int((ci[k] >> UInt32(24 - (f << 2))) & UInt32(255))
            offsets[k] = f + 32 * (bin & 31)
            keep[k] = bin != 32

        # syncTile.sync() -- tiled_partition<8>, widened to the warp. See
        # the DEVIATION note in the module docstring.
        turn_sync()

        @parameter
        for k in range(n):
            var offset1 = slice_base + offsets[k] + flag
            var add1 = Float32(0.0)
            var qadd1 = Int32(0)
            if keep[k]:
                add1 = stat1[k]
                qadd1 = qstat1[k]
            hist2_smem_add[dt](smem, offset1, add1, qadd1)

        turn_sync()

        @parameter
        for k in range(n):
            var offset2 = slice_base + offsets[k] + (1 - flag)
            var add2 = Float32(0.0)
            var qadd2 = Int32(0)
            if keep[k]:
                add2 = stat2[k]
                qadd2 = qstat2[k]
            hist2_smem_add[dt](smem, offset2, add2, qadd2)


def hist2_reduce_tail_5[
    dt: DType
](
    tid: Int,
    smem: UnsafePointer[
        Scalar[dt],
        address_space = AddressSpace.SHARED,
        origin=MutUntrackedOrigin,
    ],
):
    """The 5-bit tail of `Reduce()` after `ReduceToOneWarp`
    (`hist_2_one_byte_5bit.cu:96-118`): gather the four 8-slot sub-copies
    out of the combined warp histogram at `smem + 2048` and park the final
    `[stat][feature][fold]` layout at `smem[0..255]`.

    The caller supplies the trailing `__syncthreads()` (`:119`).

    `if fold < maxFoldCount` is vacuous at 5 bits (`fold` is masked to 5
    bits two lines up) and is kept because it is theirs.
    """
    if tid < 256:
        var is_second_stat = tid & 1
        var f = tid // 64
        var acc = Scalar[dt](0)
        var fold = (tid >> 1) & 31
        comptime max_fold_count = 32

        if fold < max_fold_count:
            var src = 2048 + 32 * fold + 2 * f + is_second_stat

            @parameter
            for in_warp_hist in range(4):
                acc += smem[src + (in_warp_hist << 3)]

            smem[max_fold_count * 4 * is_second_stat + max_fold_count * f + fold] = acc
