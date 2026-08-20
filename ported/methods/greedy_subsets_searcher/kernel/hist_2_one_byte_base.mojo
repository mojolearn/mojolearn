"""The fused TWO-STAT one-byte histogram: both stat columns in one pass.

PORT OF `hist_2_one_byte_base.cuh` (`TPointHist2OneByteBase`, the
`ComputeHist2OneByteBits` launch plumbing) and the two-stat kernel loop it
runs under, `compute_hist_loop_two_stats.cuh`
(`ComputeSplitPropertiesDirectLoadsTwoStastImpl`,
`ComputeSplitPropertiesTwoStatsGatherImpl`, `TComputeHistogramTwoStatsImpl`
and its `AlignMemoryAccess`), at CatBoost `54a8143a`. Transliterated. Do not
improve.

The per-bit accumulators live beside this file exactly as theirs do:
`hist_2_one_byte_5bit.mojo`, `_6bit.mojo`, `_7bit.mojo` mirror their
`hist_2_one_byte_{5,6,7}bit.cu`. Their CRTP (`TImpl* impl =
static_cast<TImpl*>(this)`) becomes a comptime `bits` dispatch in the three
`hist2_*` helpers below, which is the same static resolution spelled the way
Mojo can say it. The two-stat loop file is INLINED into the two kernels here
rather than kept as a separate module, exactly as the PASS family inlines
`compute_hist_loop_one_stat.cuh` into `hist_one_byte.mojo` (PORTING.md 10
and 13 record why the loop cannot be a freestanding function yet).

WHY THIS FAMILY EXISTS, AND WHEN CATBOOST TAKES IT
--------------------------------------------------
`ComputeHistOneByte` selects by `maxBins` (`hist_one_byte.cu:314-328`):

    maxBins <= 32   HIST2_PASS(5)
    maxBins <= 64   HIST2_PASS(6)
    maxBins <= 128  HIST2_PASS(7)    <- their own GPU default border count
    maxBins <= 255  PASS(8, numStats)

so for every one-byte shape up to 128 bins their dispatch runs THIS family,
which processes stat columns TWO AT A TIME (`numBlocks.z = numStats / 2`,
each block reading `stats` and `stats + statLineSize`), and the
`TPointHistOneByte` PASS family only above 128. Until 2026-08-19 this
repository routed everything through the PASS family, which is the
wrong-kernel-family misport PORTING_RULES 0b-i describes. When `numStats` is
odd, their `HIST2_PASS` macro first covers stat 0 with a one-stat
`PASS(Bits, 1)` launch and then runs this family over the remaining even
count with `SkipFirst = true` (`hist_one_byte.cu:306-312`); the `skip_first`
comptime parameter below is that template bool.

DEVIATION (PORTING.md 1): CatBoost runs this at `BlockSize = 384`
(`hist_2_one_byte_base.cuh:169`), so `384 * 32` floats is 49,152 bytes and
Apple gives 32,768. The matrix row `K_HIST_2_ONE_BYTE` resolves the block to
256, which asks for exactly 32,768 bytes and keeps their per-warp slice
arithmetic intact (8 warps times 1024 floats). The BLOCK shrinks, the
LAYOUT does not: every 1024-float warp slice, the 2048-float combined-hist
offset and the 256-thread reduce/writeback stages are theirs verbatim, and
all of them fit under a 256-thread block because their stages already cap
participation at `threadIdx.x < 256`.

DEVIATION (load batch): their `Unroll` is 1 below Volta and 2 at or above
(`hist_2_one_byte_base.cuh:28-35`), their `LoadSize` is `FourElements` on
everything after Maxwell (`:41-47`), and `TLoadSizeHist2<FourElements>` is 8
at or above Volta (`tuning_policy_enums.cuh:73-80`). This port takes the
MODERN side of each arch test, the same choice `hist_one_byte.mojo` records
for the PASS family: UNROLL 2, LOAD 4, batch 8. Scheduling, not numeric: the
same values are added in the same per-lane order.
"""

from std.atomic import Atomic
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.gpu.intrinsics import ldg
from std.memory import stack_allocation

from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from mojo_only.kernel_matrix import (
    K_HIST_2_ONE_BYTE,
    TARGET_COLUMN,
    block_size_for,
    deterministic_flush_for,
    hist_floats_per_thread_for,
    lane_width_for,
    requires_uniform_iteration_for,
)
from mojo_only.numerics import NUMERIC_FAST, NUMERIC_IDENTICAL

from ported.methods.greedy_subsets_searcher.kernel.hist_2_one_byte_5bit import (
    hist2_add_points_5,
    hist2_reduce_tail_5,
    hist2_slice_offset_5,
)
from ported.methods.greedy_subsets_searcher.kernel.hist_2_one_byte_6bit import (
    hist2_add_points_6,
    hist2_reduce_tail_6,
    hist2_slice_offset_6,
)
from ported.methods.greedy_subsets_searcher.kernel.hist_2_one_byte_7bit import (
    hist2_add_points_7,
    hist2_reduce_tail_7,
    hist2_slice_offset_7,
)


#: Same build mode as the other histogram kernels; the flush follows the
#: matrix.
comptime BUILD_MODE = NUMERIC_FAST

#: Lanes moving in lockstep. READ FROM THE MATRIX, not pinned here.
comptime LANE_WIDTH = lane_width_for[
    TARGET_COLUMN, BUILD_MODE == NUMERIC_IDENTICAL
]()

#: READ FROM THE MATRIX. Theirs is 384 (`hist_2_one_byte_base.cuh:169`);
#: Apple's 32 KB over 32 floats per thread yields 256. See the module
#: docstring's first DEVIATION.
comptime HIST2_BLOCK_SIZE = block_size_for[K_HIST_2_ONE_BYTE, TARGET_COLUMN]()

#: `GetHistSize()` = `BlockSize * 32` (`hist_2_one_byte_base.cuh:20-22`),
#: the 32 from the matrix.
comptime HIST2_HIST_SIZE = HIST2_BLOCK_SIZE * hist_floats_per_thread_for[
    K_HIST_2_ONE_BYTE
]()

#: `Unroll(ECIndexLoadType)` (`hist_2_one_byte_base.cuh:28-35`): 1 below
#: Volta, 2 at or above. The modern side, per the module docstring.
comptime HIST2_UNROLL = 2

#: `ELoadSize::FourElements` (`hist_2_one_byte_base.cuh:41-47`).
comptime HIST2_LOAD_SIZE = 4

#: `AddPointsBatchSize()` = `TLoadSizeHist2<FourElements>::Size()`
#: (`hist_2_one_byte_base.cuh:24-26`, `tuning_policy_enums.cuh:73-80`): 4
#: below Volta, 8 at or above. Equal to `loadSize * Unroll` here, so their
#: `AddPoints<N>` loop over batches (`hist_2_one_byte_base.cuh:68-78`)
#: degenerates to ONE `AddPointsImpl<8>` call per iteration.
comptime HIST2_ADD_POINTS_BATCH = 8

#: `loadSize * N`, the points one thread takes per iteration.
comptime HIST2_POINTS_PER_ITER = HIST2_LOAD_SIZE * HIST2_UNROLL

#: `BlockLoadSize(indexLoadType)` = `TLoadSizeHist2<LoadSize()>::Size() *
#: BlockSize * Unroll(...)` (`hist_2_one_byte_base.cuh:49-51`). It is what
#: decides `activeBlockCount`, and it is NOT `loadSize * BlockSize * Unroll`
#: as the PASS family's is: the hist_2 batch size doubles it.
comptime HIST2_MIN_DOCS_PER_BLOCK = (
    HIST2_ADD_POINTS_BATCH * HIST2_BLOCK_SIZE * HIST2_UNROLL
)


def hist2_slice_offset[bits: Int](tid: Int) -> Int:
    """`TImpl::SliceOffset()`, resolved by `bits` the way their CRTP does.

    THE 1024 IS THEIRS AND IT IS 32-LANE: 32 lanes times the 32 floats per
    thread this accumulator takes. A 64-lane wavefront needs a 2048-float
    slice and CatBoost never wrote that layout, so the assert makes it a
    compile error rather than two warps quietly sharing one private copy --
    the same guard `hist_one_byte.mojo` carries.
    """
    comptime assert LANE_WIDTH == 32, (
        "the hist_2 accumulator's slice layout is 32-lane by construction"
        " (`hist_2_one_byte_{5,6,7}bit.cu` SliceOffset, and ReduceToOneWarp's"
        " warpHistSize at `hist_2_one_byte_base.cuh:92`); write the"
        " wide-wavefront layout before letting LANE_WIDTH be 64"
    )

    @parameter
    if bits == 5:
        return hist2_slice_offset_5(tid)
    elif bits == 6:
        return hist2_slice_offset_6(tid)
    else:
        return hist2_slice_offset_7(tid)


def hist2_add_points[
    bits: Int, n: Int
](
    ci: InlineArray[UInt32, n],
    s1: InlineArray[Float32, n],
    s2: InlineArray[Float32, n],
    tid: Int,
    slice_base: Int,
    smem: UnsafePointer[
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
        origin=MutUntrackedOrigin,
    ],
):
    """`TImpl::AddPointsImpl<N>`, resolved by `bits`. At `n == 1` this is
    their `AddPoint` (`hist_2_one_byte_base.cuh:80-85`), which is what the
    head/tail peel calls."""

    @parameter
    if bits == 5:
        hist2_add_points_5[n](ci, s1, s2, tid, slice_base, smem)
    elif bits == 6:
        hist2_add_points_6[n](ci, s1, s2, tid, slice_base, smem)
    else:
        hist2_add_points_7[n](ci, s1, s2, tid, slice_base, smem)


def hist2_reduce_to_one_warp(
    tid: Int,
    smem: UnsafePointer[
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
        origin=MutUntrackedOrigin,
    ],
):
    """`TPointHist2OneByteBase::ReduceToOneWarp`
    (`hist_2_one_byte_base.cuh:87-104`), copied: fold the per-warp 1024-float
    copies down to one, parked at `smem[2048 + start]`.

        const int warpHistSize = 1024;
        for (int start = threadIdx.x; start < warpHistSize; start += BlockSize) {
            float sum = 0;
            for (int i = start; i < 32 * BlockSize; i += warpHistSize)
                sum += Histogram[i];
            Histogram[2048 + start] = sum;
        }

    Stride 1024 means one thread owns each residue class, so the write at
    `2048 + start` is by the same thread that already read that slot as part
    of its own sum. That is why their loop needs no barrier inside; the two
    `__syncthreads()` bracketing it are theirs.

    Their `Histogram -= impl->SliceOffset()` before the loop is the CRTP way
    of getting back to the raw buffer base; here the raw `smem` is passed
    directly, which is the same pointer.
    """
    barrier()
    comptime WARP_HIST_SIZE = 1024
    var start = tid
    while start < WARP_HIST_SIZE:
        var acc = Float32(0.0)
        var i = start
        while i < HIST2_HIST_SIZE:
            acc += smem[i]
            i += WARP_HIST_SIZE
        smem[2048 + start] = acc
        start += HIST2_BLOCK_SIZE
    barrier()


def hist2_reduce[bits: Int](
    tid: Int,
    smem: UnsafePointer[
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
        origin=MutUntrackedOrigin,
    ],
):
    """`TImpl::Reduce()`: `TParent::ReduceToOneWarp()` then the per-bit tail,
    then the tail's own trailing `__syncthreads()` (each `Reduce` ends with
    one; the tails leave it to this caller so it exists exactly once)."""
    hist2_reduce_to_one_warp(tid, smem)

    @parameter
    if bits == 5:
        hist2_reduce_tail_5(tid, smem)
    elif bits == 6:
        hist2_reduce_tail_6(tid, smem)
    else:
        hist2_reduce_tail_7(tid, smem)
    barrier()


def hist2_add_to_global_memory[
    bits: Int
](
    stat_id: Int,
    stat_count: Int,
    block_count: Int,
    feature_folds: MutPointer[UInt32, MutAnyOrigin],
    feature_fold_offset: MutPointer[UInt32, MutAnyOrigin],
    feature_group_offset: MutPointer[UInt32, MutAnyOrigin],
    feature_group_size: MutPointer[UInt32, MutAnyOrigin],
    feature_offset: Int,
    f_count: Int,
    leaf_id: Int,
    leaf_count: Int,
    bin_sums: MutPointer[Float32, MutAnyOrigin],
    acc_i32: MutPointer[Int32, MutAnyOrigin],
    fixed_scale: Float32,
    tid: Int,
    smem: UnsafePointer[
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
        origin=MutUntrackedOrigin,
    ],
):
    """`TPointHist2OneByteBase::AddToGlobalMemory`
    (`hist_2_one_byte_base.cuh:117-145`), copied. TWO stats leave in one
    call: threads 128..255 carry `isSecondStatFlag` and write `statId + 1`'s
    plane, threads 0..127 write `statId`'s, 32 fold-lanes per feature.

        const int isSecondStatFlag = threadIdx.x >= 128;
        const int fid = (threadIdx.x & 127) / 32;
        const int firstFoldIdx = threadIdx.x & 31;
        const int histSize = 1 << TImpl::MaxBits();
        ...
        for (int fold = firstFoldIdx; fold < features[fid].Folds; fold += 32)
            val = Histogram[isSecondStatFlag * 4 * histSize + fid * histSize + fold];

    `DstOffset` (`:107-115`) is the same expression the PASS family's
    writeback uses; `leaf_id` is the DENSE `blockIdx.y` and `leaf_count` is
    the caller's `max_leaves`, both exactly as `hist_one_byte.mojo`'s
    writeback takes them.

    THE FLUSH is their multi-block branch (`:135-141`): `atomicAdd` when
    `blockCount > 1`, a plain store otherwise. Wired through BOTH modes the
    way `hist_one_byte.mojo` wires it: `NUMERIC_FAST` takes CatBoost's float
    atomic verbatim; `NUMERIC_IDENTICAL` sends the replicated flush through
    the Int32 accumulator instead, because integer addition is associative
    and the histogram then does not depend on which block lands first.
    """
    comptime hist_size = 1 << bits

    if tid < 256:
        var is_second_stat = 0
        if tid >= 128:
            is_second_stat = 1
        var fid = (tid & 127) // 32
        var first_fold_idx = tid & 31

        if fid < f_count:
            var folds = Int(feature_folds.unsafe_load(feature_offset + fid))
            var group_offset = Int(
                feature_group_offset.unsafe_load(feature_offset + fid)
            )
            var group_size = Int(
                feature_group_size.unsafe_load(feature_offset + fid)
            )
            var fold_off = Int(
                feature_fold_offset.unsafe_load(feature_offset + fid)
            )

            # `DstOffset(statId + isSecondStatFlag, statCount, group, ...)`
            var device_offset = group_offset * stat_count * leaf_count
            var entries_per_leaf = stat_count * group_size
            var dst_base = (
                device_offset
                + leaf_id * entries_per_leaf
                + (stat_id + is_second_stat) * group_size
                + fold_off
            )
            var dst = bin_sums + dst_base

            var fold = first_fold_idx
            while fold < folds:
                var val = smem[
                    is_second_stat * 4 * hist_size + fid * hist_size + fold
                ]
                if abs(val) > Float32(1e-20):
                    comptime det = deterministic_flush_for[
                        TARGET_COLUMN, BUILD_MODE == NUMERIC_IDENTICAL
                    ]()

                    @parameter
                    if det:
                        if block_count > 1:
                            # `NUMERIC_IDENTICAL`: partials sum as Int32.
                            var q = Int32(val * fixed_scale)
                            _ = Atomic.fetch_add(
                                acc_i32.unsafe_offset(dst_base + fold), q
                            )
                        else:
                            dst.unsafe_store(fold, val)
                    else:
                        # `atomicAdd(dst + fold, val)`, theirs verbatim
                        # (`hist_2_one_byte_base.cuh:137`).
                        if block_count > 1:
                            _ = Atomic.fetch_add(dst.unsafe_offset(fold), val)
                        else:
                            dst.unsafe_store(fold, val)
                fold += 32


def hist2_one_byte_kernel[bits: Int, skip_first: Bool](
    # `TFeatureInBlock*`, flattened to four parallel arrays so the kernel
    # takes plain pointers, exactly as the PASS family's kernels do.
    feature_folds: MutPointer[UInt32, MutAnyOrigin],
    feature_fold_offset: MutPointer[UInt32, MutAnyOrigin],
    feature_group_offset: MutPointer[UInt32, MutAnyOrigin],
    feature_group_size: MutPointer[UInt32, MutAnyOrigin],
    f_count_in32: Int32,
    bins: MutPointer[UInt32, MutAnyOrigin],
    bins_line_size_in: Int32,
    cindex_base_in: Int32,
    stats: MutPointer[Float32, MutAnyOrigin],
    stat_line_size_in: Int32,
    part_offset: MutPointer[UInt32, MutAnyOrigin],
    part_size: MutPointer[UInt32, MutAnyOrigin],
    part_ids: MutPointer[UInt32, MutAnyOrigin],
    bin_sums: MutPointer[Float32, MutAnyOrigin],
    acc_i32: MutPointer[Int32, MutAnyOrigin],
    fixed_scale: Float32,
    leaf_count_in: Int32,
    stat_count_in: Int32,
):
    """`ComputeSplitPropertiesDirectLoadsTwoStastImpl` with `GroupSize = 4`
    (`compute_hist_loop_two_stats.cuh:495-557`), the multi-part overload.

    Grid, copied from `hist_2_one_byte_base.cuh:172-180`:
        z = (numStats - IsOdd) / 2 STAT PAIRS, y = partCount,
        x = ceil(fCount/4) * replication

    **`blockIdx.z` is a STAT PAIR, not a stat.** The block reads BOTH of its
    pair's columns, `stats` and `stats + statsLineSize`
    (`compute_hist_loop_two_stats.cuh:202-203`), and the writeback emits both
    planes in one call. `skip_first` is their `SkipFirst` template bool: true
    when an odd `numStats` had its stat 0 covered by a one-stat `PASS(Bits,
    1)` prelude, shifting every pair up by one column (`:530`, `:548-550`).

    `stat_count_in` is their `statCount = gridDim.z * 2 + (SkipFirst ? 1 :
    0)` (`:548`), passed as an argument because every kernel in this port
    takes it that way; the launch constructs the grid from the same number,
    so the two agree by construction.
    """
    var f_count_in = Int(f_count_in32)
    var bins_line_size = Int(bins_line_size_in)
    var stat_line_size = Int(stat_line_size_in)
    var leaf_count = Int(leaf_count_in)
    var stat_count = Int(stat_count_in)
    var tid = Int(thread_idx.x)

    var part_id = Int(part_ids.unsafe_load(Int(block_idx.y)))
    var p_offset = Int(part_offset.unsafe_load(part_id))
    var p_size = Int(part_size.unsafe_load(part_id))

    # `maxBlocksPerPart = gridDim.x / ceil(fCount / GroupSize)`
    var feature_blocks = (f_count_in + 3) // 4
    var max_blocks_per_part = Int(grid_dim.x) // feature_blocks
    var feature_offset = (Int(block_idx.x) // max_blocks_per_part) * 4
    var f_count = min(f_count_in - feature_offset, 4)

    # `bins += (binsLineSize * (blockIdx.x / maxBlocksPerPart))` plus the
    # policy's own column base, exactly as the PASS family's kernel notes.
    var bins_p = bins + Int(cindex_base_in) + bins_line_size * (
        Int(block_idx.x) // max_blocks_per_part
    )

    var local_block_idx = Int(block_idx.x) % max_blocks_per_part
    var active_block_count = min(
        (p_size + HIST2_MIN_DOCS_PER_BLOCK - 1) // HIST2_MIN_DOCS_PER_BLOCK,
        max_blocks_per_part,
    )
    if local_block_idx >= active_block_count:
        return

    # `stats += ((SkipFirst ? 1 : 0) + 2 * blockIdx.z) * statsLineSize`
    # (`compute_hist_loop_two_stats.cuh:530`). The pair's SECOND column is
    # read at `+ statsLineSize` throughout.
    comptime skip_int = 1 if skip_first else 0
    var stats_p = stats + (skip_int + 2 * Int(block_idx.z)) * stat_line_size

    # `TPointHist2OneByteBase`'s constructor, inlined (PORTING.md 10):
    #     for (i = threadIdx.x; i < histSize; i += BlockSize) hist[i] = 0;
    #     Histogram = hist + impl->SliceOffset();
    #     __syncthreads();
    var smem = stack_allocation[
        HIST2_HIST_SIZE,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var z = tid
    while z < HIST2_HIST_SIZE:
        smem[z] = Float32(0.0)
        z += HIST2_BLOCK_SIZE
    barrier()
    var slice_base = hist2_slice_offset[bits](tid)

    # --- AlignMemoryAccess, two-stat direct overload
    # (`compute_hist_loop_two_stats.cuh:57-108`): peel the unaligned HEAD and
    # TAIL of the partition on block 0 through `AddPoint`, so the striped
    # loop below sees a whole number of aligned warp iterations. Identical in
    # structure to the PASS family's peel; the difference is the second stat
    # column riding along (`:83`, `:102`).
    comptime ALIGN_SIZE = HIST2_LOAD_SIZE * LANE_WIDTH * HIST2_UNROLL

    var head_len = p_size
    var to_align = ALIGN_SIZE - (p_offset % ALIGN_SIZE)
    if to_align < head_len:
        head_len = to_align
    if head_len < 0:
        head_len = 0

    var body_size = p_size - head_len
    if body_size < 0:
        body_size = 0
    var tail_len = body_size % ALIGN_SIZE
    var tail_start = p_offset + head_len + (body_size - tail_len)

    var pe = tid
    while pe < ALIGN_SIZE:
        var hb = InlineArray[UInt32, 1](fill=UInt32(0))
        var hs1 = InlineArray[Float32, 1](fill=Float32(0.0))
        var hs2 = InlineArray[Float32, 1](fill=Float32(0.0))
        if local_block_idx == 0 and pe < head_len:
            # `Ldg(bins, idx)`, `Ldg(stats, idx)`,
            # `Ldg(stats, idx + statsLineSize)`
            # (`compute_hist_loop_two_stats.cuh:81-83`).
            hb[0] = ldg(bins_p + (p_offset + pe))
            hs1[0] = ldg(stats_p + (p_offset + pe))
            hs2[0] = ldg(stats_p + (p_offset + pe + stat_line_size))
        hist2_add_points[bits, 1](hb, hs1, hs2, tid, slice_base, smem)

        var tb = InlineArray[UInt32, 1](fill=UInt32(0))
        var ts1 = InlineArray[Float32, 1](fill=Float32(0.0))
        var ts2 = InlineArray[Float32, 1](fill=Float32(0.0))
        if local_block_idx == 0 and pe < tail_len:
            # `Ldg(bins, tailOffset + idx)` and the two stat loads
            # (`compute_hist_loop_two_stats.cuh:100-102`).
            tb[0] = ldg(bins_p + (tail_start + pe))
            ts1[0] = ldg(stats_p + (tail_start + pe))
            ts2[0] = ldg(stats_p + (tail_start + pe + stat_line_size))
        hist2_add_points[bits, 1](tb, ts1, ts2, tid, slice_base, smem)
        pe += HIST2_BLOCK_SIZE

    # the striped loop sees the ALIGNED MIDDLE only
    var aligned_offset = p_offset + head_len
    var aligned_size = body_size - tail_len

    var warps_per_block = HIST2_BLOCK_SIZE // LANE_WIDTH
    var global_warp_id = local_block_idx * warps_per_block + (
        tid // LANE_WIDTH
    )
    var entries_per_warp = LANE_WIDTH * HIST2_UNROLL * HIST2_LOAD_SIZE
    var stripe_size = entries_per_warp * warps_per_block * active_block_count
    var remaining = max(aligned_size - global_warp_id * entries_per_warp, 0)
    var local_idx = (tid & (LANE_WIDTH - 1)) * HIST2_LOAD_SIZE

    var base = aligned_offset + global_warp_id * entries_per_warp + local_idx
    var iter_count = (remaining - local_idx + stripe_size - 1) // stripe_size

    # THE BARRIER MUST NOT DIVERGE. Same matrix row and same uniform-count
    # workaround as the PASS family; see `hist_one_byte.mojo`'s block comment
    # at this point, which applies verbatim.
    comptime uniform = requires_uniform_iteration_for[TARGET_COLUMN]()

    @parameter
    if not uniform:
        return

    var max_iters = (aligned_size + stripe_size - 1) // stripe_size
    if max_iters < 1:
        max_iters = 1

    var b_ptr = bins_p + base
    var s_ptr = stats_p + base

    for it in range(max_iters):
        var active = it < iter_count
        var local_bins = InlineArray[UInt32, HIST2_POINTS_PER_ITER](fill=0)
        var local_stats1 = InlineArray[Float32, HIST2_POINTS_PER_ITER](
            fill=Float32(0.0)
        )
        var local_stats2 = InlineArray[Float32, HIST2_POINTS_PER_ITER](
            fill=Float32(0.0)
        )

        # `Ldg((uint4*) bins, warpSize * k)`, `Ldg((float4*) stats, ...)`,
        # `Ldg((float4*) (stats + statsLineSize), ...)`
        # (`compute_hist_loop_two_stats.cuh:386-396`). Same element-space
        # stride and the same `alignment=4` note as `hist_one_byte.mojo`.
        @parameter
        for k in range(HIST2_UNROLL):
            if active:
                var vb = ldg[width=HIST2_LOAD_SIZE, alignment=4](
                    b_ptr + LANE_WIDTH * HIST2_LOAD_SIZE * k
                )
                var vs1 = ldg[width=HIST2_LOAD_SIZE, alignment=4](
                    s_ptr + LANE_WIDTH * HIST2_LOAD_SIZE * k
                )
                var vs2 = ldg[width=HIST2_LOAD_SIZE, alignment=4](
                    s_ptr + stat_line_size + LANE_WIDTH * HIST2_LOAD_SIZE * k
                )

                @parameter
                for e in range(HIST2_LOAD_SIZE):
                    local_bins[k * HIST2_LOAD_SIZE + e] = vb[e]
                    local_stats1[k * HIST2_LOAD_SIZE + e] = vs1[e]
                    local_stats2[k * HIST2_LOAD_SIZE + e] = vs2[e]
            else:
                # No row: contribute zero, stay inside every sync.
                @parameter
                for e in range(HIST2_LOAD_SIZE):
                    local_bins[k * HIST2_LOAD_SIZE + e] = UInt32(0)
                    local_stats1[k * HIST2_LOAD_SIZE + e] = Float32(0.0)
                    local_stats2[k * HIST2_LOAD_SIZE + e] = Float32(0.0)

        # `hist.AddPoints<loadSize * N>(...)`: batch size equals the whole
        # iteration here, so this is ONE `AddPointsImpl<8>` call, their
        # `AddPoints` loop unrolled at trip count one.
        hist2_add_points[bits, HIST2_POINTS_PER_ITER](
            local_bins, local_stats1, local_stats2, tid, slice_base, smem
        )

        b_ptr += stripe_size
        s_ptr += stripe_size

    # `hist.Reduce()` then the kernel's own `__syncthreads()`
    # (`compute_hist_loop_two_stats.cuh:214-215`, `:546`).
    hist2_reduce[bits](tid, smem)
    barrier()

    # `hist.AddToGlobalMemory((SkipFirst ? 1 : 0) + 2 * blockIdx.z,
    #                         statCount, activeBlockCount, ...)` (`:550-556`)
    hist2_add_to_global_memory[bits](
        skip_int + 2 * Int(block_idx.z),
        stat_count,
        active_block_count,
        feature_folds,
        feature_fold_offset,
        feature_group_offset,
        feature_group_size,
        feature_offset,
        f_count,
        # `blockIdx.y`, DENSE, not `partIds[blockIdx.y]`, exactly as the
        # PASS family's writeback.
        Int(block_idx.y),
        leaf_count,
        bin_sums,
        acc_i32,
        fixed_scale,
        tid,
        smem,
    )


def hist2_one_byte_gather_kernel[bits: Int, skip_first: Bool](
    feature_folds: MutPointer[UInt32, MutAnyOrigin],
    feature_fold_offset: MutPointer[UInt32, MutAnyOrigin],
    feature_group_offset: MutPointer[UInt32, MutAnyOrigin],
    feature_group_size: MutPointer[UInt32, MutAnyOrigin],
    f_count_in32: Int32,
    cindex: MutPointer[UInt32, MutAnyOrigin],
    bins_line_size_in: Int32,
    cindex_base_in: Int32,
    indices: MutPointer[UInt32, MutAnyOrigin],
    stats: MutPointer[Float32, MutAnyOrigin],
    stat_line_size_in: Int32,
    part_offset: MutPointer[UInt32, MutAnyOrigin],
    part_size: MutPointer[UInt32, MutAnyOrigin],
    part_ids: MutPointer[UInt32, MutAnyOrigin],
    bin_sums: MutPointer[Float32, MutAnyOrigin],
    acc_i32: MutPointer[Int32, MutAnyOrigin],
    fixed_scale: Float32,
    leaf_count_in: Int32,
    stat_count_in: Int32,
):
    """`ComputeSplitPropertiesTwoStatsGatherImpl` with `GroupSize = 4`
    (`compute_hist_loop_two_stats.cuh:560-625`), the multi-part overload.
    Identical to the direct kernel except the bin is read through `indices`
    (their `LoadByIndexBins`; see `launch_one_byte`'s naming note, which
    applies to this family unchanged)."""
    var f_count_in = Int(f_count_in32)
    var bins_line_size = Int(bins_line_size_in)
    var stat_line_size = Int(stat_line_size_in)
    var leaf_count = Int(leaf_count_in)
    var stat_count = Int(stat_count_in)
    var tid = Int(thread_idx.x)

    var part_id = Int(part_ids.unsafe_load(Int(block_idx.y)))
    var p_offset = Int(part_offset.unsafe_load(part_id))
    var p_size = Int(part_size.unsafe_load(part_id))

    var feature_blocks = (f_count_in + 3) // 4
    var max_blocks_per_part = Int(grid_dim.x) // feature_blocks
    var feature_offset = (Int(block_idx.x) // max_blocks_per_part) * 4
    var f_count = min(f_count_in - feature_offset, 4)

    var cindex_p = cindex + Int(cindex_base_in) + bins_line_size * (
        Int(block_idx.x) // max_blocks_per_part
    )
    var idx_p = indices

    var local_block_idx = Int(block_idx.x) % max_blocks_per_part
    var active_block_count = min(
        (p_size + HIST2_MIN_DOCS_PER_BLOCK - 1) // HIST2_MIN_DOCS_PER_BLOCK,
        max_blocks_per_part,
    )
    if local_block_idx >= active_block_count:
        return

    comptime skip_int = 1 if skip_first else 0
    var stats_p = stats + (skip_int + 2 * Int(block_idx.z)) * stat_line_size

    var smem = stack_allocation[
        HIST2_HIST_SIZE,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var z = tid
    while z < HIST2_HIST_SIZE:
        smem[z] = Float32(0.0)
        z += HIST2_BLOCK_SIZE
    barrier()
    var slice_base = hist2_slice_offset[bits](tid)

    # --- AlignMemoryAccess, two-stat gather overload
    # (`compute_hist_loop_two_stats.cuh:110-163`).
    comptime ALIGN_SIZE = HIST2_LOAD_SIZE * LANE_WIDTH * HIST2_UNROLL

    var head_len = p_size
    var to_align = ALIGN_SIZE - (p_offset % ALIGN_SIZE)
    if to_align < head_len:
        head_len = to_align
    if head_len < 0:
        head_len = 0

    var body_size = p_size - head_len
    if body_size < 0:
        body_size = 0
    var tail_len = body_size % ALIGN_SIZE
    var tail_start = p_offset + head_len + (body_size - tail_len)

    var pe = tid
    while pe < ALIGN_SIZE:
        var hb = InlineArray[UInt32, 1](fill=UInt32(0))
        var hs1 = InlineArray[Float32, 1](fill=Float32(0.0))
        var hs2 = InlineArray[Float32, 1](fill=Float32(0.0))
        if local_block_idx == 0 and pe < head_len:
            # `Ldg(indices, idx)`, `Ldg(cindex, loadIdx)`, both stat loads
            # (`compute_hist_loop_two_stats.cuh:134-137`).
            var hrow = Int(ldg(indices + (p_offset + pe)))
            hb[0] = ldg(cindex_p + hrow)
            hs1[0] = ldg(stats_p + (p_offset + pe))
            hs2[0] = ldg(stats_p + (p_offset + pe + stat_line_size))
        hist2_add_points[bits, 1](hb, hs1, hs2, tid, slice_base, smem)

        var tb = InlineArray[UInt32, 1](fill=UInt32(0))
        var ts1 = InlineArray[Float32, 1](fill=Float32(0.0))
        var ts2 = InlineArray[Float32, 1](fill=Float32(0.0))
        if local_block_idx == 0 and pe < tail_len:
            # (`compute_hist_loop_two_stats.cuh:154-157`)
            var trow = Int(ldg(indices + (tail_start + pe)))
            tb[0] = ldg(cindex_p + trow)
            ts1[0] = ldg(stats_p + (tail_start + pe))
            ts2[0] = ldg(stats_p + (tail_start + pe + stat_line_size))
        hist2_add_points[bits, 1](tb, ts1, ts2, tid, slice_base, smem)
        pe += HIST2_BLOCK_SIZE

    var aligned_offset = p_offset + head_len
    var aligned_size = body_size - tail_len

    var warps_per_block = HIST2_BLOCK_SIZE // LANE_WIDTH
    var global_warp_id = local_block_idx * warps_per_block + (
        tid // LANE_WIDTH
    )
    var entries_per_warp = LANE_WIDTH * HIST2_UNROLL * HIST2_LOAD_SIZE
    var stripe_size = entries_per_warp * warps_per_block * active_block_count
    var remaining = max(aligned_size - global_warp_id * entries_per_warp, 0)
    var local_idx = (tid & (LANE_WIDTH - 1)) * HIST2_LOAD_SIZE

    var base = aligned_offset + global_warp_id * entries_per_warp + local_idx
    var iter_count = (remaining - local_idx + stripe_size - 1) // stripe_size

    comptime uniform = requires_uniform_iteration_for[TARGET_COLUMN]()

    @parameter
    if not uniform:
        return

    var max_iters = (aligned_size + stripe_size - 1) // stripe_size
    if max_iters < 1:
        max_iters = 1

    var i_ptr = idx_p + base
    var s_ptr = stats_p + base

    for it in range(max_iters):
        var active = it < iter_count
        var local_bins = InlineArray[UInt32, HIST2_POINTS_PER_ITER](fill=0)
        var local_stats1 = InlineArray[Float32, HIST2_POINTS_PER_ITER](
            fill=Float32(0.0)
        )
        var local_stats2 = InlineArray[Float32, HIST2_POINTS_PER_ITER](
            fill=Float32(0.0)
        )

        # Their gather batch (`compute_hist_loop_two_stats.cuh:424-449`):
        # indices and BOTH stat columns load 4-wide; only the bins are
        # gathered one at a time, because a gather has no vector form.
        @parameter
        for k in range(HIST2_UNROLL):
            if active:
                var vi = ldg[width=HIST2_LOAD_SIZE, alignment=4](
                    i_ptr + LANE_WIDTH * HIST2_LOAD_SIZE * k
                )
                var vs1 = ldg[width=HIST2_LOAD_SIZE, alignment=4](
                    s_ptr + LANE_WIDTH * HIST2_LOAD_SIZE * k
                )
                var vs2 = ldg[width=HIST2_LOAD_SIZE, alignment=4](
                    s_ptr + stat_line_size + LANE_WIDTH * HIST2_LOAD_SIZE * k
                )

                @parameter
                for e in range(HIST2_LOAD_SIZE):
                    local_bins[k * HIST2_LOAD_SIZE + e] = ldg(
                        cindex_p + Int(vi[e])
                    )
                    local_stats1[k * HIST2_LOAD_SIZE + e] = vs1[e]
                    local_stats2[k * HIST2_LOAD_SIZE + e] = vs2[e]
            else:

                @parameter
                for e in range(HIST2_LOAD_SIZE):
                    local_bins[k * HIST2_LOAD_SIZE + e] = UInt32(0)
                    local_stats1[k * HIST2_LOAD_SIZE + e] = Float32(0.0)
                    local_stats2[k * HIST2_LOAD_SIZE + e] = Float32(0.0)

        hist2_add_points[bits, HIST2_POINTS_PER_ITER](
            local_bins, local_stats1, local_stats2, tid, slice_base, smem
        )

        i_ptr += stripe_size
        s_ptr += stripe_size

    hist2_reduce[bits](tid, smem)
    barrier()

    hist2_add_to_global_memory[bits](
        skip_int + 2 * Int(block_idx.z),
        stat_count,
        active_block_count,
        feature_folds,
        feature_fold_offset,
        feature_group_offset,
        feature_group_size,
        feature_offset,
        f_count,
        Int(block_idx.y),
        leaf_count,
        bin_sums,
        acc_i32,
        fixed_scale,
        tid,
        smem,
    )
