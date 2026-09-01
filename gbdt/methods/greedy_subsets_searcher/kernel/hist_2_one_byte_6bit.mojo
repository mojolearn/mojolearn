# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The 6-bit specialization of the fused two-stat one-byte accumulator.

PORT OF `hist_2_one_byte_6bit.cu` at CatBoost `54a8143a`,
`TPointHist2OneByte<6, BlockSize>`. Transliterated. Do not improve.

The family CatBoost's one-byte dispatch takes at `32 < maxBins <= 64`
(`hist_one_byte.cu:317-319`).

Where the 5-bit variant has four 8-slot sub-copies per warp, six bits leave
room for TWO 16-slot sub-copies (`blocks = 2`, `innerHistStart = tid & 16`),
and the slot spends its odd bin bit inline: `offset = f + 16 * (bin & 62) +
8 * (bin & 1) + flag`. With only two sub-copies, half the warp's lanes would
collide inside a phase, so the writes are additionally serialized by
`writeFirstFlag = threadIdx.x & 8`: the two halves of each 16-lane tile take
turns (`hist_2_one_byte_6bit.cu:72-115`).

DEVIATION (sync width): their sync is a `tiled_partition<16>`
(`hist_2_one_byte_6bit.cu:38`). Widened to `syncwarp` (32 lanes), the same
widening `hist_2_one_byte_5bit.mojo` documents: every lane executes the same
unconditional sync count, so it is stricter than theirs and reorders
nothing.
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


def hist2_slice_offset_6(tid: Int) -> Int:
    """`TPointHist2OneByte<6>::SliceOffset()` (`hist_2_one_byte_6bit.cu:25-31`).

        const int warpId = (threadIdx.x / 32);
        const int warpOffset = 1024 * warpId;
        const int blocks = 2;
        const int innerHistStart = (threadIdx.x & ((blocks - 1) << 4));
        return warpOffset + innerHistStart;
    """
    var warp_offset = 1024 * (tid // 32)
    var inner_hist_start = tid & ((2 - 1) << 4)
    return warp_offset + inner_hist_start


def hist2_add_points_6[
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
    """`TPointHist2OneByte<6>::AddPointsImpl<N>`
    (`hist_2_one_byte_6bit.cu:33-117`), branch for branch.

    Theirs folds the skip into arithmetic, `pass = bin != 64 ? 1.0f : 0.0f`
    then `val = pass * stat`, rather than the 5-bit variant's conditional;
    copied that way. The stat-2 phase reuses the SAME offsets shifted by
    `flag ? -1 : 1`, which lands each lane on the opposite parity slot
    (`:92-96`).

    `dt`, `q1` and `q2` exist for the `HIST_SMEM_SHARED2_I32` matrix row
    (`q1`/`q2` are the batch's stats through `hist2_quantize`, computed once
    per row at load time):
    pass structure, bin decode, slot arithmetic and both sync/turn
    disciplines are theirs in both modes, and WHERE the add lands is decided
    once in `hist2_smem_add` (see its DEVIATION BLOCK).
    """
    var flag = tid & 1

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

    var val1 = InlineArray[Float32, n](fill=Float32(0.0))
    var val2 = InlineArray[Float32, n](fill=Float32(0.0))
    var qval1 = InlineArray[Int32, n](fill=Int32(0))
    var qval2 = InlineArray[Int32, n](fill=Int32(0))
    var offset = InlineArray[Int, n](fill=0)

    @parameter
    for i in range(4):
        var f = (2 * i + tid) & 6

        @parameter
        for k in range(n):
            var bin = Int((ci[k] >> UInt32(24 - (f << 2))) & UInt32(255))
            var keep = Float32(0.0)
            var qkeep = Int32(0)
            if bin != 64:
                keep = Float32(1.0)
                qkeep = Int32(1)
            val1[k] = keep * stat1[k]
            val2[k] = keep * stat2[k]
            qval1[k] = qkeep * qstat1[k]
            qval2[k] = qkeep * qstat2[k]
            offset[k] = (
                slice_base + f + 16 * (bin & 62) + 8 * (bin & 1) + flag
            )

        # const bool writeFirstFlag = threadIdx.x & 8;
        var write_first = (tid & 8) != 0

        # syncTile.sync() -- tiled_partition<16>, widened to the warp. See
        # the DEVIATION note in the module docstring.
        turn_sync()

        if write_first:

            @parameter
            for k in range(n):
                hist2_smem_add[dt](smem, offset[k], val1[k], qval1[k])

        turn_sync()

        if not write_first:

            @parameter
            for k in range(n):
                hist2_smem_add[dt](smem, offset[k], val1[k], qval1[k])

        # int shift = flag ? -1 : 1;
        var shift = 1
        if flag == 1:
            shift = -1

        @parameter
        for k in range(n):
            offset[k] += shift

        turn_sync()

        if write_first:

            @parameter
            for k in range(n):
                hist2_smem_add[dt](smem, offset[k], val2[k], qval2[k])

        turn_sync()

        if not write_first:

            @parameter
            for k in range(n):
                hist2_smem_add[dt](smem, offset[k], val2[k], qval2[k])


def hist2_reduce_tail_6[
    dt: DType
](
    tid: Int,
    smem: UnsafePointer[
        Scalar[dt],
        address_space = AddressSpace.SHARED,
        origin=MutUntrackedOrigin,
    ],
):
    """The 6-bit tail of `Reduce()` after `ReduceToOneWarp`
    (`hist_2_one_byte_6bit.cu:126-152`): each of the 256 participating
    threads gathers TWO folds, `fold0` and `fold0 + 32`, out of the combined
    warp histogram at `smem + 2048` (two sub-copies, stride 16; the upper
    32 folds sit 512 floats up) and parks the final `[stat][feature][fold]`
    layout at `smem[0..511]`.

    The caller supplies the trailing `__syncthreads()` (`:153`).
    """
    if tid < 256:
        var is_second_stat = tid & 1
        var f = tid // 64
        var sum0 = Scalar[dt](0)
        var sum1 = Scalar[dt](0)
        var fold0 = (tid >> 1) & 31
        comptime max_fold_count = 64

        var src = (
            2048
            + 2 * f
            + 8 * (fold0 & 1)
            + 32 * (fold0 >> 1)
            + is_second_stat
        )

        @parameter
        for in_warp_hist in range(2):
            sum0 += smem[src + (in_warp_hist << 4)]
            sum1 += smem[src + (in_warp_hist << 4) + 512]

        smem[
            max_fold_count * 4 * is_second_stat + max_fold_count * f + fold0
        ] = sum0
        smem[
            max_fold_count * 4 * is_second_stat
            + max_fold_count * f
            + fold0
            + 32
        ] = sum1
