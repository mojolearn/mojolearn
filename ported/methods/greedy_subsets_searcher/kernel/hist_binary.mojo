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

from std.atomic import Atomic
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from mojo_only.kernel_matrix import (
    TARGET_COLUMN,
    deterministic_flush_for,
    requires_uniform_iteration_for,
)
from mojo_only.numerics import NUMERIC_FAST, NUMERIC_IDENTICAL
from ported.methods.greedy_subsets_searcher.kernel.point_hist_half_byte_template import (
    BLOCK_SIZE,
    HIST_SIZE,
    REDUCE_WIDTH,
    add_point_slot,
    reduce_stage2_slot,
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

#: The mode this build compiles against; see `mojo_only/numerics.mojo`.
comptime BUILD_MODE = NUMERIC_FAST


def binary_hist_kernel(
    # `TFeatureInBlock*`, flattened to four parallel arrays so the kernel
    # takes plain pointers.
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
    """`ComputeSplitPropertiesDirectLoadsImpl` with `GroupSize = 32`.

    Grid, copied from `hist_binary.cu:86-97`:
        z = numStats, y = partCount, x = ceil(fCount/32) * replication

    **`blockIdx.y` is the LEAF.** All leaves of a level are one launch, and
    the number of launches does not depend on the leaf count or on the
    dataset size. That is the design point (`compute_by_blocks_helper.h:87-92`
    states it outright) and it is the opposite of a per-leaf launch.
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
    var feature_blocks = (f_count_in + 31) // 32
    var max_blocks_per_part = Int(grid_dim.x) // feature_blocks
    var feature_offset = (Int(block_idx.x) // max_blocks_per_part) * 32
    var f_count = min(f_count_in - feature_offset, 32)

    # `cindex += features->CompressedIndexOffset` in theirs: the base of
    # THIS POLICY's columns. Distinct from `bins_line_size`, which is the
    # stride between FEATURE BLOCKS inside the policy. Conflating them makes
    # every policy after the first read the first one's bits, which looks
    # like a tree that will not split.
    var bins_p = bins + Int(cindex_base_in) + bins_line_size * (
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

    # `TPointHistHalfByteBase`'s constructor, inlined (PORTING.md 10):
    #     for (i = threadIdx.x; i < histSize; i += BlockSize) buff[i] = 0;
    #     __syncthreads();
    #     Histogram = buff + SliceOffset();
    var smem = stack_allocation[
        HIST_SIZE,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var z = tid
    while z < HIST_SIZE:
        smem[z] = Float32(0.0)
        z += BLOCK_SIZE
    barrier()
    var slice_base = slice_offset(tid)

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

    # THE BARRIER MUST NOT DIVERGE. The requirement is a MATRIX ROW,
    # `requires_uniform_iteration_for`, not a local decision, so that when
    # Mojo exposes lane primitives the kernels follow the table instead of
    # each carrying its own workaround.
    comptime uniform = requires_uniform_iteration_for[TARGET_COLUMN]()

    @parameter
    if not uniform:
        # Deliberately unimplemented rather than silently wrong: the literal
        # CatBoost loop needs a lane-local sync, and reaching here means the
        # matrix claims one exists. Write that path before flipping the row.
        return

    # This is where the port stops being a transliteration. CatBoost syncs a `tiled_partition<8>`, which is
    # WARP-LOCAL, so warps with different iteration counts never wait on each
    # other. Mojo 1.0 has only the threadgroup-wide `barrier()`, and a
    # threadgroup barrier that some warps reach and others skip is undefined
    # behavior.
    #
    # It bites immediately rather than rarely: a 64-row partition over a
    # 512-thread block gives warp 0 one iteration and warps 1 to 15 zero, so
    # the first fifteen sixteenths of the block walk past barriers warp 0 is
    # waiting on. Measured result before this fix: every feature's histogram
    # came back 0.0.
    #
    # So every thread runs the SAME iteration count and the ones with no rows
    # contribute nothing. `block.max` is not available here, so the count is
    # derived from the partition size, which every thread already has.
    var max_iters = (p_size + stripe_size - 1) // stripe_size
    if max_iters < 1:
        max_iters = 1

    var b_ptr = bins_p + base
    var s_ptr = stats_p + base

    for it in range(max_iters):
        var active = it < iter_count
        # Their two unrolled loops: gather the batch, then add it. Kept in
        # that order because it is what keeps the loads in flight.
        var local_bins = InlineArray[UInt32, UNROLL](fill=0)
        var local_stats = InlineArray[Float32, UNROLL](fill=0)

        @parameter
        for k in range(UNROLL):
            if active and base + it * stripe_size + LANE_WIDTH * k < p_offset + p_size:
                local_bins[k] = b_ptr.unsafe_load(LANE_WIDTH * k)
                local_stats[k] = s_ptr.unsafe_load(LANE_WIDTH * k)
            else:
                # No row: contribute zero. The slot it lands in is harmless
                # because the stat is 0.0, and it keeps this lane inside every
                # barrier below.
                local_bins[k] = UInt32(0)
                local_stats[k] = Float32(0.0)

        # `AddPoint`, inlined. The 8 lanes of a tile touch 8 distinct slots
        # per iteration (see `add_point_slot`), so the update is a plain
        # `+=` with no atomic. The barrier between iterations is CatBoost's
        # 8-lane `addToHistTile.sync()` widened to the threadgroup, which is
        # correct and strictly more expensive (PORTING.md 2). It is NOT safe
        # to drop: distinctness holds WITHIN an iteration only.
        @parameter
        for k in range(UNROLL):
            @parameter
            for i in range(8):
                var slot = slice_base + add_point_slot(local_bins[k], tid, i)
                smem[slot] = smem[slot] + local_stats[k]
                barrier()

        b_ptr += stripe_size
        s_ptr += stripe_size

    # `Reduce()`, inlined. Stage 1 folds the whole replicated scratch to
    # REDUCE_WIDTH slots, stage 2 folds those to 128 at
    # `featureId + 8 * fold`.
    barrier()
    var slot_i = tid
    while slot_i < REDUCE_WIDTH:
        var acc = Float32(0.0)
        var i2 = slot_i
        while i2 < HIST_SIZE:
            acc += smem[i2]
            i2 += REDUCE_WIDTH
        barrier()
        smem[slot_i] = acc
        barrier()
        slot_i += BLOCK_SIZE

    var acc2 = Float32(0.0)
    if tid < 128:
        @parameter
        for group in range(4):
            acc2 += smem[reduce_stage2_slot(tid, group)]
    barrier()
    if tid < 128:
        smem[tid] = acc2
    barrier()

    # --- TPointHistBinary::AddToGlobalMemory, copied --------------------
    var fid = tid
    var fold = 0
    if fid < f_count:
        var folds = Int(feature_folds.unsafe_load(feature_offset + fid))
        if folds != 0:
            var group_offset = Int(feature_group_offset.unsafe_load(feature_offset + fid))
            var group_size = Int(feature_group_size.unsafe_load(feature_offset + fid))
            var device_offset = group_offset * stat_count * leaf_count
            var entries_per_leaf = stat_count * group_size
            var dst = (
                bin_sums
                + device_offset
                # `blockIdx.y`, DENSE, not `partIds[blockIdx.y]`. See the
                # note above `entries_per_leaf`.
                + Int(block_idx.y) * entries_per_leaf
                + Int(block_idx.z) * group_size
                + Int(feature_fold_offset.unsafe_load(feature_offset + fid))
            )

            var group_id = fid // 4
            var f_mask = 1 << (3 - (fid & 3))

            var val = Float32(0.0)

            @parameter
            for i in range(16):
                if (i & f_mask) == 0:
                    val += smem[8 * i + group_id]

            # Their guard, copied: skip a flush that cannot matter, so the
            # non-deterministic atomic is not paid for nothing.
            if abs(val) > Float32(1e-20):
                # THE FLUSH, and the one row `mojo_only/numerics.mojo` says a
                # vendor can override. CatBoost writes
                #
                #     if (blockCount > 1) atomicAdd(dst + fold, val);
                #     else                dst[fold] = val;
                #
                # A plain store is only correct when ONE block owns the
                # partition. Replicating blocks across a partition, which is
                # what fills the machine, makes every block hold a PARTIAL
                # histogram, and partials must be summed.
                #
                # Metal has no float atomic add, so the partial sum goes
                # through the FIXED-POINT accumulator: `val * scale` into an
                # Int32 with an integer atomic, which Metal does have.
                # Integer addition is associative, so the result does not
                # depend on which block lands first, and the histogram is
                # reproducible run to run rather than merely correct. That is
                # the property CatBoost's float atomic gives up.
                # THE ROW, read from the matrix rather than assumed. On
                # Apple it is forced true because Metal has no float atomic;
                # on NVIDIA and AMD it follows the mode.
                # THE ROW, read from the matrix rather than assumed. On
                # Apple it is forced true because Metal has no float atomic;
                # on NVIDIA and AMD it follows the mode. Comptime, because
                # the two flushes are different code and not a configured
                # value, which is the distinction numerics.mojo draws.
                comptime det = deterministic_flush_for[
                    TARGET_COLUMN, BUILD_MODE == NUMERIC_IDENTICAL
                ]()

                @parameter
                if det:
                    if active_block_count > 1:
                        # Replicated blocks hold PARTIAL histograms. Metal
                        # has no float atomic, so partials sum as Int32
                        # through an integer atomic, which is associative and
                        # therefore reproducible run to run.
                        var q = Int32(val * fixed_scale)
                        _ = Atomic.fetch_add(
                            acc_i32.unsafe_offset(
                                (
                                    Int(
                                        part_ids.unsafe_load(
                                            Int(block_idx.y)
                                        )
                                    )
                                    * stat_count
                                    + Int(block_idx.z)
                                )
                                * group_size
                                + Int(
                                    feature_fold_offset.unsafe_load(
                                        feature_offset + fid
                                    )
                                )
                                + fold
                            ),
                            q,
                        )
                    else:
                        dst.unsafe_store(fold, val)
                else:
                    # CatBoost's float-atomic path. Unreachable while
                    # TARGET_COLUMN is apple. Left as a plain store rather
                    # than a float atomic Mojo cannot emit; the day a CUDA
                    # column is built this is where their branch goes.
                    dst.unsafe_store(fold, val)


def binary_hist_gather_kernel(
    # `TFeatureInBlock*`, flattened to four parallel arrays so the kernel
    # takes plain pointers.
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
    """`ComputeSplitPropertiesGatherImpl` with `GroupSize = 32`.

    **The difference from the direct variant is one line, and it is the whole
    reason both exist.** Direct reads the bin at a POSITION:

        featureVal[k] = bins[i + k * BlockSize]

    Gather reads it through the row index:

        loadIndex[k]  = indices[i + k * BlockSize]
        featureVal[k] = cindex[loadIndex[k]]

    Which is correct depends on whether the rows have been reordered. The
    compressed index is NEVER permuted, deliberately (`split_points.mojo`),
    so after the first split a leaf's position `i` no longer names row `i` of
    the bin matrix. Direct loads are right at depth 0, where the index is the
    identity, and silently wrong below it: every feature's histogram becomes
    noise from unrelated rows, every score looks alike, and the level re-picks
    the same split. Measured here as a depth-6 tree with 64 partitions of
    which 2 held rows.

    The STAT columns are read by position, because those ARE permuted, by
    `gather_in_leaves_kernel`. So one plane is indexed through the map and
    the other is not, which is easy to get backwards.

    Grid, copied from `hist_binary.cu:86-97`:
        z = numStats, y = partCount, x = ceil(fCount/32) * replication

    **`blockIdx.y` is the LEAF.** All leaves of a level are one launch, and
    the number of launches does not depend on the leaf count or on the
    dataset size. That is the design point (`compute_by_blocks_helper.h:87-92`
    states it outright) and it is the opposite of a per-leaf launch.
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
    var feature_blocks = (f_count_in + 31) // 32
    var max_blocks_per_part = Int(grid_dim.x) // feature_blocks
    var feature_offset = (Int(block_idx.x) // max_blocks_per_part) * 32
    var f_count = min(f_count_in - feature_offset, 32)

    # `cindex += features->CompressedIndexOffset` in theirs; the port takes
    # one feature group, so the offset is the group's column.
    # See the note in the direct variant: this is the policy's column base,
    # not the feature-block stride.
    var cindex_p = cindex + Int(cindex_base_in) + bins_line_size * (
        Int(block_idx.x) // max_blocks_per_part
    )
    var idx_p = indices

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

    # `TPointHistHalfByteBase`'s constructor, inlined (PORTING.md 10):
    #     for (i = threadIdx.x; i < histSize; i += BlockSize) buff[i] = 0;
    #     __syncthreads();
    #     Histogram = buff + SliceOffset();
    var smem = stack_allocation[
        HIST_SIZE,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var z = tid
    while z < HIST_SIZE:
        smem[z] = Float32(0.0)
        z += BLOCK_SIZE
    barrier()
    var slice_base = slice_offset(tid)

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

    # THE BARRIER MUST NOT DIVERGE. The requirement is a MATRIX ROW,
    # `requires_uniform_iteration_for`, not a local decision, so that when
    # Mojo exposes lane primitives the kernels follow the table instead of
    # each carrying its own workaround.
    comptime uniform = requires_uniform_iteration_for[TARGET_COLUMN]()

    @parameter
    if not uniform:
        # Deliberately unimplemented rather than silently wrong: the literal
        # CatBoost loop needs a lane-local sync, and reaching here means the
        # matrix claims one exists. Write that path before flipping the row.
        return

    # This is where the port stops being a transliteration. CatBoost syncs a `tiled_partition<8>`, which is
    # WARP-LOCAL, so warps with different iteration counts never wait on each
    # other. Mojo 1.0 has only the threadgroup-wide `barrier()`, and a
    # threadgroup barrier that some warps reach and others skip is undefined
    # behavior.
    #
    # It bites immediately rather than rarely: a 64-row partition over a
    # 512-thread block gives warp 0 one iteration and warps 1 to 15 zero, so
    # the first fifteen sixteenths of the block walk past barriers warp 0 is
    # waiting on. Measured result before this fix: every feature's histogram
    # came back 0.0.
    #
    # So every thread runs the SAME iteration count and the ones with no rows
    # contribute nothing. `block.max` is not available here, so the count is
    # derived from the partition size, which every thread already has.
    var max_iters = (p_size + stripe_size - 1) // stripe_size
    if max_iters < 1:
        max_iters = 1

    var i_ptr = idx_p + base
    var s_ptr = stats_p + base

    for it in range(max_iters):
        var active = it < iter_count
        # Their two unrolled loops: gather the batch, then add it. Kept in
        # that order because it is what keeps the loads in flight.
        var local_bins = InlineArray[UInt32, UNROLL](fill=0)
        var local_stats = InlineArray[Float32, UNROLL](fill=0)

        @parameter
        for k in range(UNROLL):
            if active and base + it * stripe_size + LANE_WIDTH * k < p_offset + p_size:
                # THE GATHER: position -> row -> bin.
                var row = Int(i_ptr.unsafe_load(LANE_WIDTH * k))
                local_bins[k] = cindex_p.unsafe_load(row)
                local_stats[k] = s_ptr.unsafe_load(LANE_WIDTH * k)
            else:
                # No row: contribute zero. The slot it lands in is harmless
                # because the stat is 0.0, and it keeps this lane inside every
                # barrier below.
                local_bins[k] = UInt32(0)
                local_stats[k] = Float32(0.0)

        # `AddPoint`, inlined. The 8 lanes of a tile touch 8 distinct slots
        # per iteration (see `add_point_slot`), so the update is a plain
        # `+=` with no atomic. The barrier between iterations is CatBoost's
        # 8-lane `addToHistTile.sync()` widened to the threadgroup, which is
        # correct and strictly more expensive (PORTING.md 2). It is NOT safe
        # to drop: distinctness holds WITHIN an iteration only.
        @parameter
        for k in range(UNROLL):
            @parameter
            for i in range(8):
                var slot = slice_base + add_point_slot(local_bins[k], tid, i)
                smem[slot] = smem[slot] + local_stats[k]
                barrier()

        i_ptr += stripe_size
        s_ptr += stripe_size

    # `Reduce()`, inlined. Stage 1 folds the whole replicated scratch to
    # REDUCE_WIDTH slots, stage 2 folds those to 128 at
    # `featureId + 8 * fold`.
    barrier()
    var slot_i = tid
    while slot_i < REDUCE_WIDTH:
        var acc = Float32(0.0)
        var i2 = slot_i
        while i2 < HIST_SIZE:
            acc += smem[i2]
            i2 += REDUCE_WIDTH
        barrier()
        smem[slot_i] = acc
        barrier()
        slot_i += BLOCK_SIZE

    var acc2 = Float32(0.0)
    if tid < 128:
        @parameter
        for group in range(4):
            acc2 += smem[reduce_stage2_slot(tid, group)]
    barrier()
    if tid < 128:
        smem[tid] = acc2
    barrier()

    # --- TPointHistBinary::AddToGlobalMemory, copied --------------------
    var fid = tid
    var fold = 0
    if fid < f_count:
        var folds = Int(feature_folds.unsafe_load(feature_offset + fid))
        if folds != 0:
            var group_offset = Int(feature_group_offset.unsafe_load(feature_offset + fid))
            var group_size = Int(feature_group_size.unsafe_load(feature_offset + fid))
            var device_offset = group_offset * stat_count * leaf_count
            var entries_per_leaf = stat_count * group_size
            var dst = (
                bin_sums
                + device_offset
                # `blockIdx.y`, DENSE, not `partIds[blockIdx.y]`. See the
                # note above `entries_per_leaf`.
                + Int(block_idx.y) * entries_per_leaf
                + Int(block_idx.z) * group_size
                + Int(feature_fold_offset.unsafe_load(feature_offset + fid))
            )

            var group_id = fid // 4
            var f_mask = 1 << (3 - (fid & 3))

            var val = Float32(0.0)

            @parameter
            for i in range(16):
                if (i & f_mask) == 0:
                    val += smem[8 * i + group_id]

            # Their guard, copied: skip a flush that cannot matter, so the
            # non-deterministic atomic is not paid for nothing.
            if abs(val) > Float32(1e-20):
                # THE FLUSH, and the one row `mojo_only/numerics.mojo` says a
                # vendor can override. CatBoost writes
                #
                #     if (blockCount > 1) atomicAdd(dst + fold, val);
                #     else                dst[fold] = val;
                #
                # A plain store is only correct when ONE block owns the
                # partition. Replicating blocks across a partition, which is
                # what fills the machine, makes every block hold a PARTIAL
                # histogram, and partials must be summed.
                #
                # Metal has no float atomic add, so the partial sum goes
                # through the FIXED-POINT accumulator: `val * scale` into an
                # Int32 with an integer atomic, which Metal does have.
                # Integer addition is associative, so the result does not
                # depend on which block lands first, and the histogram is
                # reproducible run to run rather than merely correct. That is
                # the property CatBoost's float atomic gives up.
                # THE ROW, read from the matrix rather than assumed. On
                # Apple it is forced true because Metal has no float atomic;
                # on NVIDIA and AMD it follows the mode.
                # THE ROW, read from the matrix rather than assumed. On
                # Apple it is forced true because Metal has no float atomic;
                # on NVIDIA and AMD it follows the mode. Comptime, because
                # the two flushes are different code and not a configured
                # value, which is the distinction numerics.mojo draws.
                comptime det = deterministic_flush_for[
                    TARGET_COLUMN, BUILD_MODE == NUMERIC_IDENTICAL
                ]()

                @parameter
                if det:
                    if active_block_count > 1:
                        # Replicated blocks hold PARTIAL histograms. Metal
                        # has no float atomic, so partials sum as Int32
                        # through an integer atomic, which is associative and
                        # therefore reproducible run to run.
                        var q = Int32(val * fixed_scale)
                        _ = Atomic.fetch_add(
                            acc_i32.unsafe_offset(
                                (
                                    Int(
                                        part_ids.unsafe_load(
                                            Int(block_idx.y)
                                        )
                                    )
                                    * stat_count
                                    + Int(block_idx.z)
                                )
                                * group_size
                                + Int(
                                    feature_fold_offset.unsafe_load(
                                        feature_offset + fid
                                    )
                                )
                                + fold
                            ),
                            q,
                        )
                    else:
                        dst.unsafe_store(fold, val)
                else:
                    # CatBoost's float-atomic path. Unreachable while
                    # TARGET_COLUMN is apple. Left as a plain store rather
                    # than a float atomic Mojo cannot emit; the day a CUDA
                    # column is built this is where their branch goes.
                    dst.unsafe_store(fold, val)
