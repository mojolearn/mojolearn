"""The binary-feature histogram kernel: 32 features per 4-byte load.

PORT OF `hist_binary.cu` plus the loop it instantiates,
`compute_hist_loop_one_stat.cuh` (`ALIGN_MEMORY`,
`TComputeHistogramImpl<FourElements>::Compute`,
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

from std.atomic import Atomic, Ordering
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.gpu.intrinsics import ldg
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier, syncwarp

from mojo_only.kernel_matrix import (
    lane_width_for,
    TARGET_COLUMN,
    deterministic_flush_for,
    requires_uniform_iteration_for,
)
from mojo_only.numerics import (
    PIN_DETERMINISM,
    GLOBAL_NUMERIC_MODE,
    NUMERIC_FAST,
    NUMERIC_IDENTICAL,
    ftz,
)
from gbdt.methods.greedy_subsets_searcher.kernel.point_hist_half_byte_template import (
    BLOCK_SIZE,
    HIST_SIZE,
    REDUCE_WIDTH,
    add_half_byte_point,
    add_point_slot,
    reduce_stage2_slot,
    slice_offset,
)


#: `TPointHistBinary::Unroll(ECIndexLoadType)` for `__CUDA_ARCH__ >= 700`
#: (`hist_binary.cu:18-26`, which returns 4 / 1 / 2 by arch). SCHEDULING row:
#: it changes how many loads are in flight, not what is summed into what.
#: The half-byte kernel has its OWN table and returns 1; the two are not
#: interchangeable, which is why each file cites its own.
comptime UNROLL = 2

#: `ELoadSize::FourElements`, selected by the SHARED base for both this
#: kernel and the half-byte one (`point_hist_half_byte_template.cuh:34-42`).
#: CatBoost picks it on every arch except Maxwell-to-Volta, and it is the
#: reason a thread of theirs consumes four points per load where ours used to
#: consume one. The histogram is bandwidth bound, so this is the load path,
#: not a detail.
#:
#: Raising it REQUIRES the head/tail peel below: a partition offset is not
#: 4-aligned in general, so a 4-wide load without `AlignMemoryAccess` is
#: ILLEGAL and not merely slow. The claim this replaces, that the load width
#: is "scheduling and not numeric", was the same sentence that was wrong for
#: `hist_one_byte.mojo`.
comptime LOAD_SIZE = 4

#: `loadSize * N`, the points one thread takes per iteration.
comptime POINTS_PER_ITER = UNROLL * LOAD_SIZE


#: The mode this build compiles against; see `mojo_only/numerics.mojo`.
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
    fixed_scale_ptr: MutPointer[Float32, MutAnyOrigin],
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
    var fixed_scale = fixed_scale_ptr.unsafe_load(0)
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

    # --- AlignMemoryAccess, ported --------------------------------------
    # (`compute_hist_loop_one_stat.cuh:57-105`, the direct overload.)
    #
    # REQUIRED by LOAD_SIZE 4, not an optimization: a partition offset is not
    # 4-aligned in general, so the vector load below is illegal without this.
    #
    # Peels the unaligned HEAD and TAIL of the partition on block 0 with
    # scalar adds, so what remains starts and ends on an `alignSize`
    # boundary. `alignSize = LoadSize * warpSize * N` is exactly one warp
    # iteration, which is what lets the striped loop issue ALIGNED 4-wide
    # loads and carry no per-element bounds test.
    #
    #     int lastId = min(partSize, alignSize - (partOffset % alignSize));
    #     if (blockId == 0) for (idx = tid; idx < alignSize; idx += BlockSize)
    #     partSize = max(partSize - lastId, 0);
    #     const int unalignedTail = (partSize % alignSize);
    #     if (unalignedTail) { if (blockId == 0) { ...tail... } }
    #     partSize -= unalignedTail;
    #
    # Their head and tail loops run to a FIXED `alignSize` bound, so every
    # thread of block 0 makes the same number of trips and the syncs inside
    # `AddPoint` stay uniform across the warp. Blocks other than 0 make the
    # trips too and contribute zeros, which keeps the count uniform without a
    # second code path.
    comptime ALIGN_SIZE = LOAD_SIZE * LANE_WIDTH * UNROLL

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
        var hb = UInt32(0)
        var hs = Float32(0.0)
        if local_block_idx == 0 and pe < head_len:
            # `Ldg(bins, idx)` and `Ldg(stats, idx)`
            # (`compute_hist_loop_one_stat.cuh:80-81`). `Ldg` is
            # `cub::ThreadLoad<cub::LOAD_LDG>` (`kernel_helpers.cuh:180`),
            # the read-only non-coherent load; `std.gpu.intrinsics.ldg` is
            # its Mojo spelling.
            hb = ldg(bins_p + (p_offset + pe))
            hs = ldg(stats_p + (p_offset + pe))
        add_half_byte_point(hb, hs, tid, slice_base, smem)

        var tb = UInt32(0)
        var ts = Float32(0.0)
        if local_block_idx == 0 and pe < tail_len:
            # `Ldg(bins, tailOffset + idx)` and `Ldg(stats, tailOffset + idx)`
            # (`compute_hist_loop_one_stat.cuh:98-99`).
            tb = ldg(bins_p + (tail_start + pe))
            ts = ldg(stats_p + (tail_start + pe))
        add_half_byte_point(tb, ts, tid, slice_base, smem)
        pe += BLOCK_SIZE

    # the striped loop sees the ALIGNED MIDDLE only
    var aligned_offset = p_offset + head_len
    var aligned_size = body_size - tail_len

    var warps_per_block = BLOCK_SIZE // LANE_WIDTH
    var global_warp_id = local_block_idx * warps_per_block + (tid // LANE_WIDTH)
    var entries_per_warp = LANE_WIDTH * UNROLL * LOAD_SIZE
    var stripe_size = entries_per_warp * warps_per_block * active_block_count
    var remaining = max(aligned_size - global_warp_id * entries_per_warp, 0)
    var local_idx = (tid & (LANE_WIDTH - 1)) * LOAD_SIZE

    var base = aligned_offset + global_warp_id * entries_per_warp + local_idx
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
    var max_iters = (aligned_size + stripe_size - 1) // stripe_size
    if max_iters < 1:
        max_iters = 1

    var b_ptr = bins_p + base
    var s_ptr = stats_p + base

    for it in range(max_iters):
        var active = it < iter_count
        # Their two unrolled loops: gather the batch, then add it. Kept in
        # that order because it is what keeps the loads in flight.
        var local_bins = InlineArray[UInt32, POINTS_PER_ITER](fill=0)
        var local_stats = InlineArray[Float32, POINTS_PER_ITER](fill=0)

        # `Ldg((uint4*) bins, warpSize * k)` and its `float4` twin
        # (`compute_hist_loop_one_stat.cuh:366-375`). Indexing a `uint4*` by
        # `warpSize * k` advances `warpSize * k * 4` ELEMENTS, which is the
        # element-space stride written here.
        #
        # NO per-element bounds test. The peel above leaves a whole number of
        # warp iterations, so an ACTIVE iteration is wholly in range. That is
        # the entire reason `AlignMemoryAccess` exists, and it is what makes
        # a 4-wide load legal as well as fast. Only the uniform-iteration
        # guard remains, and that one is ours.
        @parameter
        for k in range(UNROLL):
            if active:
                # `Ldg((uint4*) bins, warpSize * k)` and
                # `Ldg((float4*) stats, warpSize * k)`
                # (`compute_hist_loop_one_stat.cuh:368`, `:374`).
                # `alignment=4` is OURS and is stated rather than
                # inherited. Their `Ldg((uint4*) ...)` is a 16-byte load
                # because their column stride is `AlignedColumnSize()`; ours
                # is `n_rows` (`greedy_search_helper.mojo`), so a column base
                # is not 4-element aligned in general and the width-4 load
                # keeps the element alignment the plain load already assumed.
                var vb = ldg[width=LOAD_SIZE, alignment=4](
                    b_ptr + LANE_WIDTH * LOAD_SIZE * k
                )
                var vs = ldg[width=LOAD_SIZE, alignment=4](
                    s_ptr + LANE_WIDTH * LOAD_SIZE * k
                )

                @parameter
                for e in range(LOAD_SIZE):
                    local_bins[k * LOAD_SIZE + e] = vb[e]
                    local_stats[k * LOAD_SIZE + e] = vs[e]
            else:
                # No row: contribute zero. The slot it lands in is harmless
                # because the stat is 0.0, and it keeps this lane inside
                # every sync below.
                @parameter
                for e in range(LOAD_SIZE):
                    local_bins[k * LOAD_SIZE + e] = UInt32(0)
                    local_stats[k * LOAD_SIZE + e] = Float32(0.0)

        # `hist.AddPoints<loadSize * N>(...)`
        # (`point_hist_half_byte_template.cuh:103-112`), which hands the
        # points to `AddPointsImpl<N>` (:79-101) in batches of
        # `AddPointsBatchSize()` = LOAD_SIZE.
        #
        # THE NEST IS THEIRS, and its ORDER is the whole point: `i`, the
        # feature rotation, is OUTER and `k`, the batch, is INNER, with
        # exactly ONE sync per `i` -- 8 syncs per batch of LOAD_SIZE points.
        # Ours had `k` outer and `i` inner, which pays 8 syncs PER POINT.
        #
        # It stays collision-free because distinctness is a property of `i`
        # and not of `k`: within one `i` the 8 lanes of a tile hold 8
        # distinct values of `f = (tid + i) & 7`, and the slot carries `f` in
        # its low three bits, so no two lanes of the tile can land on the
        # same slot however many `k` they run. That is exactly why the sync
        # belongs on `i`.
        @parameter
        for batch in range(UNROLL):

            @parameter
            for i in range(8):

                @parameter
                for e in range(LOAD_SIZE):
                    var t = local_stats[batch * LOAD_SIZE + e]
                    var slot = slice_base + add_point_slot(
                        local_bins[batch * LOAD_SIZE + e], tid, i
                    )
                    # IDENTITY_PATHS ROW 10: float-family intermediate,
                    # flushed -- see `add_half_byte_point`'s note.
                    # Comptime no-op under FAST.
                    smem[slot] = ftz(smem[slot] + t)
                # `addToHistTile.sync()`, an 8-lane `tiled_partition<8>` in
                # theirs. `syncwarp` is 32 lanes, so it orders a SUPERSET and
                # is still correct; it is the narrowest sync Mojo exposes.
                syncwarp()

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
            # row 10: reduce-stage intermediate, flushed (no-op under
            # FAST).
            acc = ftz(acc + smem[i2])
            i2 += REDUCE_WIDTH
        barrier()
        smem[slot_i] = acc
        barrier()
        slot_i += BLOCK_SIZE

    var acc2 = Float32(0.0)
    if tid < 128:
        @parameter
        for group in range(4):
            # row 10: reduce-stage intermediate, flushed (no-op under
            # FAST).
            acc2 = ftz(acc2 + smem[reduce_stage2_slot(tid, group)])
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
                # `NUMERIC_IDENTICAL` sends the partial sum through the
                # FIXED-POINT accumulator instead: `val * scale` into an
                # Int32 with an integer atomic. Integer addition is
                # associative, so the result does not depend on which block
                # lands first, and the histogram is reproducible run to run
                # rather than merely correct. That is the property CatBoost's
                # float atomic gives up, and it is the only reason the
                # integer path exists.
                #
                # THE ROW, read from the matrix rather than assumed. It
                # follows the MODE on every vendor. Comptime, because the two
                # flushes are different code and not a configured value,
                # which is the distinction numerics.mojo draws.
                comptime det = deterministic_flush_for[
                    TARGET_COLUMN, PIN_DETERMINISM
                ]()

                @parameter
                if det:
                    if active_block_count > 1:
                        # Replicated blocks hold PARTIAL histograms, and
                        # under `NUMERIC_IDENTICAL` they sum as Int32 through
                        # an integer atomic, which is associative and
                        # therefore reproducible run to run.
                        var q = Int32(val * fixed_scale)
                        # THE ACCUMULATOR MIRRORS `bin_sums` EXACTLY.
                        #
                        # It is the fixed-point twin of the same cell, so it
                        # has to use the same address arithmetic: the block
                        # layout, `device_offset` included, keyed by the
                        # DENSE `blockIdx.y`. It used to be keyed by
                        # `partIds[blockIdx.y]` and to omit `device_offset`,
                        # which coincides with this expression only when
                        # there is ONE feature block and the id list is the
                        # identity. That is every single-policy check in this
                        # repository, which is why it survived.
                        # DEVIATION 1898: upstream's atomicAdd is relaxed; the
                        # non-Apple Mojo default is seq_cst.
                        _ = Atomic.fetch_add[ordering = Ordering.RELAXED](
                            acc_i32.unsafe_offset(
                                device_offset
                                + Int(block_idx.y) * entries_per_leaf
                                + Int(block_idx.z) * group_size
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
                    # `atomicAdd(dst + fold, val)` -- CatBoost's flush, in
                    # every `AddToGlobalMemory`.
                    #
                    # This was a plain STORE with a comment saying Mojo could
                    # not emit a float atomic and that the branch was
                    # unreachable on Apple. Both halves were false: probed
                    # 2026-08-19, 1024 threads each adding 1.0 through
                    # `Atomic.fetch_add` give exactly 1024.0 on the M4.
                    #
                    # Order-nondeterministic, exactly as theirs is, and
                    # CatBoost ships it that way. `NUMERIC_IDENTICAL` selects
                    # the fixed-point branch above when reproducibility is
                    # wanted; that branch is now a CHOICE rather than the
                    # only thing that compiles.
                    if active_block_count > 1:
                        # DEVIATION 1898: upstream's atomicAdd is relaxed; the
                        # non-Apple Mojo default is seq_cst.
                        _ = Atomic.fetch_add[ordering = Ordering.RELAXED](
                            dst.unsafe_offset(fold), val
                        )
                    else:
                        dst.unsafe_store(fold, val)


def binary_hist_gather_kernel[ridx_stats: Bool = False](
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
    fixed_scale_ptr: MutPointer[Float32, MutAnyOrigin],
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
    the other is not, which is easy to get backwards. Under `ridx_stats`
    (DEVIATION 1902) the stat planes stop being permuted and the stat read
    goes through the SAME row id as the bin -- same bits, see
    `split_points_ridx.mojo`.

    Grid, copied from `hist_binary.cu:86-97`:
        z = numStats, y = partCount, x = ceil(fCount/32) * replication

    **`blockIdx.y` is the LEAF.** All leaves of a level are one launch, and
    the number of launches does not depend on the leaf count or on the
    dataset size. That is the design point (`compute_by_blocks_helper.h:87-92`
    states it outright) and it is the opposite of a per-leaf launch.
    """
    var fixed_scale = fixed_scale_ptr.unsafe_load(0)
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

    # --- AlignMemoryAccess (gather), ported
    # (`compute_hist_loop_one_stat.cuh:107-157`). Same peel as the direct
    # variant; the difference is only that the bin comes through `indices`.
    #
    # REQUIRED by LOAD_SIZE 4, not an optimization: a partition offset is not
    # 4-aligned in general, so the vector load below is illegal without this.
    #
    # Peels the unaligned HEAD and TAIL of the partition on block 0 with
    # scalar adds, so what remains starts and ends on an `alignSize`
    # boundary. `alignSize = LoadSize * warpSize * N` is exactly one warp
    # iteration, which is what lets the striped loop issue ALIGNED 4-wide
    # loads and carry no per-element bounds test.
    #
    #     int lastId = min(partSize, alignSize - (partOffset % alignSize));
    #     if (blockId == 0) for (idx = tid; idx < alignSize; idx += BlockSize)
    #     partSize = max(partSize - lastId, 0);
    #     const int unalignedTail = (partSize % alignSize);
    #     if (unalignedTail) { if (blockId == 0) { ...tail... } }
    #     partSize -= unalignedTail;
    #
    # Their head and tail loops run to a FIXED `alignSize` bound, so every
    # thread of block 0 makes the same number of trips and the syncs inside
    # `AddPoint` stay uniform across the warp. Blocks other than 0 make the
    # trips too and contribute zeros, which keeps the count uniform without a
    # second code path.
    comptime ALIGN_SIZE = LOAD_SIZE * LANE_WIDTH * UNROLL

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
        var hb = UInt32(0)
        var hs = Float32(0.0)
        if local_block_idx == 0 and pe < head_len:
            # `Ldg(indices, idx)`, `Ldg(cindex, loadIdx)`, `Ldg(stats, idx)`
            # (`compute_hist_loop_one_stat.cuh:130-132`). All three streams
            # are read-only for the kernel's lifetime, which is what `Ldg`
            # declares.
            var hrow = Int(ldg(indices + (p_offset + pe)))
            hb = ldg(cindex_p + hrow)

            @parameter
            if ridx_stats:
                # DEVIATION 1902: the stat plane is stationary and the
                # stat rides the SAME gathered row id as the bin
                # (`split_points_ridx.mojo`'s invariant).
                hs = ldg(stats_p + hrow)
            else:
                hs = ldg(stats_p + (p_offset + pe))
        add_half_byte_point(hb, hs, tid, slice_base, smem)

        var tb = UInt32(0)
        var ts = Float32(0.0)
        if local_block_idx == 0 and pe < tail_len:
            # `Ldg(indices, tailOffset + idx)`, `Ldg(cindex, loadIdx)`,
            # `Ldg(stats, tailOffset + idx)`
            # (`compute_hist_loop_one_stat.cuh:149-151`).
            var trow = Int(ldg(indices + (tail_start + pe)))
            tb = ldg(cindex_p + trow)

            @parameter
            if ridx_stats:
                # DEVIATION 1902, as on the head peel above.
                ts = ldg(stats_p + trow)
            else:
                ts = ldg(stats_p + (tail_start + pe))
        add_half_byte_point(tb, ts, tid, slice_base, smem)
        pe += BLOCK_SIZE

    # the striped loop sees the ALIGNED MIDDLE only
    var aligned_offset = p_offset + head_len
    var aligned_size = body_size - tail_len

    var warps_per_block = BLOCK_SIZE // LANE_WIDTH
    var global_warp_id = local_block_idx * warps_per_block + (tid // LANE_WIDTH)
    var entries_per_warp = LANE_WIDTH * UNROLL * LOAD_SIZE
    var stripe_size = entries_per_warp * warps_per_block * active_block_count
    var remaining = max(aligned_size - global_warp_id * entries_per_warp, 0)
    var local_idx = (tid & (LANE_WIDTH - 1)) * LOAD_SIZE

    var base = aligned_offset + global_warp_id * entries_per_warp + local_idx
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
    var max_iters = (aligned_size + stripe_size - 1) // stripe_size
    if max_iters < 1:
        max_iters = 1

    var i_ptr = idx_p + base
    var s_ptr = stats_p + base

    for it in range(max_iters):
        var active = it < iter_count
        # Their two unrolled loops: gather the batch, then add it. Kept in
        # that order because it is what keeps the loads in flight.
        var local_bins = InlineArray[UInt32, POINTS_PER_ITER](fill=0)
        var local_stats = InlineArray[Float32, POINTS_PER_ITER](fill=0)

        # Their gather batch (`compute_hist_loop_one_stat.cuh:406-424`):
        #
        #     localIndices[k] = Ldg((int4*) indices, warpSize * k);
        #     localBins[k].x  = Ldg(cindex, localIndices[k].x);   ...
        #     localStats[k]   = Ldg((float4*) stats, warpSize * k);
        #
        # The INDICES and the STATS are contiguous, so both load 4-wide. Only
        # the BINS are gathered, one at a time, because a gather has no
        # vector form. That asymmetry is theirs and it is the point: two of
        # the three streams still get the wide load.
        #
        # NO per-element bounds test: the peel above leaves a whole number of
        # warp iterations, so an ACTIVE iteration is wholly in range.
        @parameter
        for k in range(UNROLL):
            if active:
                # `Ldg((int4*) indices, warpSize * k)` and
                # `Ldg((float4*) stats, warpSize * k)`
                # (`compute_hist_loop_one_stat.cuh:407`, `:423`).
                # `alignment=4` is OURS and is stated rather than
                # inherited. Their `Ldg((uint4*) ...)` is a 16-byte load
                # because their column stride is `AlignedColumnSize()`; ours
                # is `n_rows` (`greedy_search_helper.mojo`), so a column base
                # is not 4-element aligned in general and the width-4 load
                # keeps the element alignment the plain load already assumed.
                var vi = ldg[width=LOAD_SIZE, alignment=4](
                    i_ptr + LANE_WIDTH * LOAD_SIZE * k
                )
                var vs = SIMD[DType.float32, LOAD_SIZE](0.0)

                @parameter
                if not ridx_stats:
                    vs = ldg[width=LOAD_SIZE, alignment=4](
                        s_ptr + LANE_WIDTH * LOAD_SIZE * k
                    )

                @parameter
                for e in range(LOAD_SIZE):
                    # THE GATHER: position -> row -> bin.
                    # `localBins[k].x = Ldg(cindex, localIndices[k].x)` and
                    # its y, z, w twins
                    # (`compute_hist_loop_one_stat.cuh:415-418`). Scalar,
                    # because a gather has no vector form.
                    local_bins[k * LOAD_SIZE + e] = ldg(
                        cindex_p + Int(vi[e])
                    )

                    @parameter
                    if ridx_stats:
                        # DEVIATION 1902: the stat joins the bins' scalar
                        # gather through the same loaded row id; the wide
                        # load above is traded for it.
                        vs[e] = ldg(stats_p + Int(vi[e]))
                    local_stats[k * LOAD_SIZE + e] = vs[e]
            else:
                # No row: contribute zero. The slot it lands in is harmless
                # because the stat is 0.0, and it keeps this lane inside
                # every sync below.
                @parameter
                for e in range(LOAD_SIZE):
                    local_bins[k * LOAD_SIZE + e] = UInt32(0)
                    local_stats[k * LOAD_SIZE + e] = Float32(0.0)

        # `hist.AddPoints<loadSize * N>(...)`
        # (`point_hist_half_byte_template.cuh:103-112`), which hands the
        # points to `AddPointsImpl<N>` (:79-101) in batches of
        # `AddPointsBatchSize()` = LOAD_SIZE.
        #
        # THE NEST IS THEIRS, and its ORDER is the whole point: `i`, the
        # feature rotation, is OUTER and `k`, the batch, is INNER, with
        # exactly ONE sync per `i` -- 8 syncs per batch of LOAD_SIZE points.
        # Ours had `k` outer and `i` inner, which pays 8 syncs PER POINT.
        #
        # It stays collision-free because distinctness is a property of `i`
        # and not of `k`: within one `i` the 8 lanes of a tile hold 8
        # distinct values of `f = (tid + i) & 7`, and the slot carries `f` in
        # its low three bits, so no two lanes of the tile can land on the
        # same slot however many `k` they run. That is exactly why the sync
        # belongs on `i`.
        @parameter
        for batch in range(UNROLL):

            @parameter
            for i in range(8):

                @parameter
                for e in range(LOAD_SIZE):
                    var t = local_stats[batch * LOAD_SIZE + e]
                    var slot = slice_base + add_point_slot(
                        local_bins[batch * LOAD_SIZE + e], tid, i
                    )
                    # IDENTITY_PATHS ROW 10: float-family intermediate,
                    # flushed -- see `add_half_byte_point`'s note.
                    # Comptime no-op under FAST.
                    smem[slot] = ftz(smem[slot] + t)
                # `addToHistTile.sync()`, an 8-lane `tiled_partition<8>` in
                # theirs. `syncwarp` is 32 lanes, so it orders a SUPERSET and
                # is still correct; it is the narrowest sync Mojo exposes.
                syncwarp()

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
            # row 10: reduce-stage intermediate, flushed (no-op under
            # FAST).
            acc = ftz(acc + smem[i2])
            i2 += REDUCE_WIDTH
        barrier()
        smem[slot_i] = acc
        barrier()
        slot_i += BLOCK_SIZE

    var acc2 = Float32(0.0)
    if tid < 128:
        @parameter
        for group in range(4):
            # row 10: reduce-stage intermediate, flushed (no-op under
            # FAST).
            acc2 = ftz(acc2 + smem[reduce_stage2_slot(tid, group)])
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
                # `NUMERIC_IDENTICAL` sends the partial sum through the
                # FIXED-POINT accumulator instead: `val * scale` into an
                # Int32 with an integer atomic. Integer addition is
                # associative, so the result does not depend on which block
                # lands first, and the histogram is reproducible run to run
                # rather than merely correct. That is the property CatBoost's
                # float atomic gives up, and it is the only reason the
                # integer path exists.
                #
                # THE ROW, read from the matrix rather than assumed. It
                # follows the MODE on every vendor. Comptime, because the two
                # flushes are different code and not a configured value,
                # which is the distinction numerics.mojo draws.
                comptime det = deterministic_flush_for[
                    TARGET_COLUMN, PIN_DETERMINISM
                ]()

                @parameter
                if det:
                    if active_block_count > 1:
                        # Replicated blocks hold PARTIAL histograms, and
                        # under `NUMERIC_IDENTICAL` they sum as Int32 through
                        # an integer atomic, which is associative and
                        # therefore reproducible run to run.
                        var q = Int32(val * fixed_scale)
                        # THE ACCUMULATOR MIRRORS `bin_sums` EXACTLY.
                        #
                        # It is the fixed-point twin of the same cell, so it
                        # has to use the same address arithmetic: the block
                        # layout, `device_offset` included, keyed by the
                        # DENSE `blockIdx.y`. It used to be keyed by
                        # `partIds[blockIdx.y]` and to omit `device_offset`,
                        # which coincides with this expression only when
                        # there is ONE feature block and the id list is the
                        # identity. That is every single-policy check in this
                        # repository, which is why it survived.
                        # DEVIATION 1898: upstream's atomicAdd is relaxed; the
                        # non-Apple Mojo default is seq_cst.
                        _ = Atomic.fetch_add[ordering = Ordering.RELAXED](
                            acc_i32.unsafe_offset(
                                device_offset
                                + Int(block_idx.y) * entries_per_leaf
                                + Int(block_idx.z) * group_size
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
                    # `atomicAdd(dst + fold, val)` -- CatBoost's flush, in
                    # every `AddToGlobalMemory`.
                    #
                    # This was a plain STORE with a comment saying Mojo could
                    # not emit a float atomic and that the branch was
                    # unreachable on Apple. Both halves were false: probed
                    # 2026-08-19, 1024 threads each adding 1.0 through
                    # `Atomic.fetch_add` give exactly 1024.0 on the M4.
                    #
                    # Order-nondeterministic, exactly as theirs is, and
                    # CatBoost ships it that way. `NUMERIC_IDENTICAL` selects
                    # the fixed-point branch above when reproducibility is
                    # wanted; that branch is now a CHOICE rather than the
                    # only thing that compiles.
                    if active_block_count > 1:
                        # DEVIATION 1898: upstream's atomicAdd is relaxed; the
                        # non-Apple Mojo default is seq_cst.
                        _ = Atomic.fetch_add[ordering = Ordering.RELAXED](
                            dst.unsafe_offset(fold), val
                        )
                    else:
                        dst.unsafe_store(fold, val)
