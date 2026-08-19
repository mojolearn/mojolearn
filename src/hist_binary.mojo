"""The binary-feature histogram kernel: 32 features per 4-byte load.

PORT OF `hist_binary.cu` plus the loop it instantiates,
`compute_hist_loop_one_stat.cuh` (`ALIGN_MEMORY`,
`TComputeHistogramImpl<OneElement>::Compute`,
`ComputeSplitPropertiesDirectLoadsImpl`), at CatBoost `54a8143a`.
Transliterated. Do not improve.

**This is the kernel that should matter most on covtype**, where 44 of 54
columns are 0/1 and route to `BinaryFeatures`. One `UInt32` of the compressed
index holds 32 of them, so one load feeds 32 features' worth of histogram
work. mojotrees issues 32 separate one-byte loads for the same result.

The nibble trick in the writeback is worth reading twice. Binary features are
packed 1 bit each, but the ACCUMULATOR is the half-byte one with 16 bins per
group, so one 4-bit nibble carries FOUR binary features and its 16 bins are
the 16 combinations of those four. To recover feature `fid`'s "bin 0" count
you sum the eight of those sixteen combinations in which its bit is clear:

    groupId = fid / 4
    fMask   = 1 << (3 - (fid & 3))
    val     = sum over i in 0..15 where !(i & fMask) of Histogram[8 * i + groupId]

Only the zero side is written. The one side is `total - val`, recovered by
the caller, which is why a binary feature costs one fold and not two.
"""

from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from .point_hist_half_byte import (
    BLOCK_SIZE,
    HIST_SIZE,
    add_point,
    hist_reduce,
    hist_zero_and_slice,
    slice_offset,
)


#: `THist::Unroll(ECIndexLoadType::Direct)` for `__CUDA_ARCH__ >= 700`
#: (`hist_binary.cu:20-26`). SCHEDULING row: it changes how many loads are in
#: flight, not what is summed into what.
comptime UNROLL = 2

#: DEVIATION (PORTING.md 5): CatBoost selects a 2- or 4-element vector load
#: by arch (`point_hist_half_byte_template.cuh:34-41`) and instantiates a
#: different `TComputeHistogramImpl` for each. This port takes the
#: `OneElement` specialization only. Scheduling, not numeric: the same values
#: are added in the same order, fewer at a time.
comptime LOAD_SIZE = 1

#: Lanes moving in lockstep. Pinned rather than read from the device; see
#: `kernel_matrix.column_lane_width` for why AMD's 64 must not reach this.
comptime LANE_WIDTH = 32


def binary_hist_kernel(
    # `TFeatureInBlock*`, flattened to four parallel arrays so the kernel
    # takes plain pointers.
    feature_folds: MutPointer[UInt32, MutAnyOrigin],
    feature_fold_offset: MutPointer[UInt32, MutAnyOrigin],
    feature_group_offset: MutPointer[UInt32, MutAnyOrigin],
    feature_group_size: MutPointer[UInt32, MutAnyOrigin],
    f_count_in: Int,
    bins: MutPointer[UInt32, MutAnyOrigin],
    bins_line_size: Int,
    stats: MutPointer[Float32, MutAnyOrigin],
    stat_line_size: Int,
    part_offset: MutPointer[UInt32, MutAnyOrigin],
    part_size: MutPointer[UInt32, MutAnyOrigin],
    part_ids: MutPointer[UInt32, MutAnyOrigin],
    bin_sums: MutPointer[Float32, MutAnyOrigin],
    leaf_count: Int,
    stat_count: Int,
):
    """`ComputeSplitPropertiesDirectLoadsImpl` with `GroupSize = 32`.

    Grid, copied from `hist_binary.cu:86-97`:
        z = numStats, y = partCount, x = ceil(fCount/32) * replication

    **`blockIdx.y` is the LEAF.** All leaves of a level are one launch, and
    the number of launches does not depend on the leaf count or on the
    dataset size. That is the design point (`compute_by_blocks_helper.h:87-92`
    states it outright) and it is the opposite of a per-leaf launch.
    """
    var tid = Int(thread_idx.x)

    var part_id = Int(part_ids[Int(block_idx.y)])
    var p_offset = Int(part_offset[part_id])
    var p_size = Int(part_size[part_id])

    # `maxBlocksPerPart = gridDim.x / ceil(fCount / GroupSize)`
    var feature_blocks = (f_count_in + 31) // 32
    var max_blocks_per_part = Int(grid_dim.x) // feature_blocks
    var feature_offset = (Int(block_idx.x) // max_blocks_per_part) * 32
    var f_count = min(f_count_in - feature_offset, 32)

    var bins_p = bins + bins_line_size * (
        Int(block_idx.x) // max_blocks_per_part
    )

    # THE RAGGED-LEAF TRICK, and it costs nothing. The grid is sized for the
    # worst-case leaf; every block reads its partition's size ON THE DEVICE
    # and the ones with no work return. No host round trip, no per-leaf
    # launch, no dependence on how unbalanced the level is.
    var local_block_idx = Int(block_idx.x) % max_blocks_per_part
    var min_docs_per_block = LANE_WIDTH * UNROLL * LOAD_SIZE * (
        BLOCK_SIZE // LANE_WIDTH
    )
    var active_block_count = min(
        (p_size + min_docs_per_block - 1) // min_docs_per_block,
        max_blocks_per_part,
    )
    if local_block_idx >= active_block_count:
        return

    var stats_p = stats + Int(block_idx.z) * stat_line_size

    var smem = stack_allocation[
        HIST_SIZE,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var hist = hist_zero_and_slice(smem, tid)

    # --- ALIGN_MEMORY(1), copied ---------------------------------------
    #
    # DEVIATION (PORTING.md 6): `AlignMemoryAccess` peels an unaligned prefix
    # so the vector loads that follow are aligned. At LOAD_SIZE 1 there is
    # nothing to align, so the peel is omitted rather than ported. It becomes
    # required the moment LOAD_SIZE moves above 1.
    var warps_per_block = BLOCK_SIZE // LANE_WIDTH
    var global_warp_id = local_block_idx * warps_per_block + (tid // LANE_WIDTH)
    var entries_per_warp = LANE_WIDTH * UNROLL * LOAD_SIZE
    var stripe_size = entries_per_warp * warps_per_block * active_block_count
    var remaining = max(p_size - global_warp_id * entries_per_warp, 0)
    var local_idx = (tid & (LANE_WIDTH - 1)) * LOAD_SIZE

    var base = p_offset + global_warp_id * entries_per_warp + local_idx
    var iter_count = (remaining - local_idx + stripe_size - 1) // stripe_size

    var b_ptr = bins_p + base
    var s_ptr = stats_p + base

    for _ in range(iter_count):
        # Their two unrolled loops: gather the batch, then add it. Kept in
        # that order because it is what keeps the loads in flight.
        var local_bins = InlineArray[Scalar[DType.uint32], UNROLL](fill=0)
        var local_stats = InlineArray[Scalar[DType.float32], UNROLL](fill=0)

        @parameter
        for k in range(UNROLL):
            local_bins[k] = b_ptr[LANE_WIDTH * k]

        @parameter
        for k in range(UNROLL):
            local_stats[k] = s_ptr[LANE_WIDTH * k]

        @parameter
        for k in range(UNROLL):
            add_point(hist, local_bins[k], local_stats[k], tid)

        b_ptr += stripe_size
        s_ptr += stripe_size

    var reduced = hist_reduce(hist, tid)
    barrier()

    # --- TPointHistBinary::AddToGlobalMemory, copied --------------------
    var fid = tid
    var fold = 0
    if fid < f_count:
        var folds = Int(feature_folds[feature_offset + fid])
        if folds != 0:
            var group_offset = Int(feature_group_offset[feature_offset + fid])
            var group_size = Int(feature_group_size[feature_offset + fid])
            var device_offset = group_offset * stat_count * leaf_count
            var entries_per_leaf = stat_count * group_size
            var dst = (
                bin_sums
                + device_offset
                + Int(part_ids[Int(block_idx.y)]) * entries_per_leaf
                + Int(block_idx.z) * group_size
                + Int(feature_fold_offset[feature_offset + fid])
            )

            var group_id = fid // 4
            var f_mask = 1 << (3 - (fid & 3))

            var val = Scalar[DType.float32](0.0)

            @parameter
            for i in range(16):
                if (i & f_mask) == 0:
                    val += reduced[8 * i + group_id]

            # Their guard, copied: skip a flush that cannot matter, so the
            # non-deterministic atomic is not paid for nothing.
            if abs(val) > Scalar[DType.float32](1e-20):
                # DEVIATION (numerics.mojo): CatBoost uses atomicAdd when
                # blockCount > 1 and a plain store otherwise. The atomic is
                # order-nondeterministic and is what `NUMERIC_IDENTICAL`
                # replaces with a fixed-point accumulator. Direct store here;
                # the multi-block flush lands with the replication work.
                dst[fold] = val
