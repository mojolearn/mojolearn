"""The one-byte histogram kernel: 4 features per 4-byte load, 32 to 256 bins.

PORT OF `hist_one_byte.cu` and the base it derives from,
`hist_2_one_byte_base.cuh`, at CatBoost `54a8143a`. Transliterated. Do not
improve.

This is the odd one of the three and it does not derive from the other two.
Where binary and half-byte share `TPointHistHalfByteBase` at 16 floats per
thread, this one uses 32 and trades replication for serialization to make
wide features fit.

THE INNER-BITS TRICK, which is the whole file
---------------------------------------------
A one-byte feature has up to 256 bins, far more than a private per-lane slot
can hold. So the bin is split:

    InnerHistBitsCount = Bits - 5
    higherBin = (bin >> 5) & ((1 << InnerHistBitsCount) - 1)
    offset    = 4 * higherBin + f + ((bin & 31) << 5)

The low 5 bits of the bin index the slot directly. The HIGH bits are handled
by running `1 << InnerHistBitsCount` PASSES, where on pass `p` only the lanes
whose `higherBin == p` write:

    for (k = 0; k < (1 << InnerHistBitsCount); ++k) {
        const int pass = ((threadIdx.x >> 2) + k) & mask;
        syncTile.sync();
        if (pass == higherBin) Histogram[offset] += statToAdd;
    }

So a 5-bit feature costs one pass and a 8-bit feature costs eight. **Wide
features are paid for in TIME rather than in shared memory**, which is the
opposite trade from replicating the histogram, and it is why this kernel can
exist at all inside a 32 KB budget.

Their `SliceOffset` differs to match: `1024 * (threadIdx.x / 32)` gives each
warp 1024 floats (32 per lane), and the inner offset is masked by the number
of blocks the inner bits leave.

DEVIATION (PORTING.md 1): CatBoost runs this at `BlockSize = 384`, so
`384 * 32` floats is 49,152 bytes and Apple gives 32,768. `BLOCK_SIZE = 256`
asks for exactly 32,768 and keeps their per-warp slice arithmetic intact,
since 8 warps times 1024 floats is 8192 floats.
"""

from std.gpu import block_dim, block_idx, grid_dim, thread_idx

from mojo_only.kernel_matrix import (
    TARGET_COLUMN,
    requires_uniform_iteration_for,
)
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from mojo_only.kernel_matrix import (
    K_HIST_ONE_BYTE,
    TARGET_COLUMN,
    block_size_for,
    hist_floats_per_thread_for,
)


#: READ FROM THE MATRIX. Theirs is 384; Apple's 32 KB over 32 floats per
#: thread yields 256, which is exactly 32,768 bytes and keeps their per-warp
#: slice arithmetic intact at 8 warps of 1024 floats.
comptime ONE_BYTE_BLOCK_SIZE = block_size_for[K_HIST_ONE_BYTE, TARGET_COLUMN]()

#: `GetHistSize()` = `BlockSize * 32`, the 32 from the matrix.
comptime ONE_BYTE_HIST_SIZE = ONE_BYTE_BLOCK_SIZE * hist_floats_per_thread_for[
    K_HIST_ONE_BYTE
]()

#: `Unroll` for `__CUDA_ARCH__ >= 700`.
comptime ONE_BYTE_UNROLL = 2

#: Their unroll and load width, named as the loop expects them.
comptime UNROLL = ONE_BYTE_UNROLL

#: DEVIATION (PORTING.md 5): CatBoost picks a 4-element vector load here
#: (`hist_2_one_byte_base.cuh:44-48`); the port takes one element, which is
#: scheduling and not numeric.
comptime LOAD_SIZE = 1

comptime LANE_WIDTH = 32


def one_byte_slice_offset[bits: Int](tid: Int) -> Int:
    """`TPointHistOneByte::SliceOffset()`, copied.

        const int warpOffset = 1024 * (threadIdx.x / 32);
        const int blocks = 8 >> InnerHistBitsCount;
        const int innerHistStart =
            threadIdx.x & ((blocks - 1) << (InnerHistBitsCount + 2));
        return warpOffset + innerHistStart;

    `blocks` shrinks as the feature widens: a 5-bit feature gets 8 private
    sub-copies per warp, an 8-bit feature gets 1. That is the same trade as
    the pass loop, seen from the memory side.
    """
    comptime inner_bits = bits - 5
    var warp_offset = 1024 * (tid // LANE_WIDTH)
    comptime blocks = 8 >> inner_bits
    var inner_hist_start = tid & ((blocks - 1) << (inner_bits + 2))
    return warp_offset + inner_hist_start


def one_byte_bin_offset[bits: Int](ci: UInt32, tid: Int, i: Int) -> Int:
    """The slot for iteration `i`, as pure arithmetic (PORTING.md 10).

        int f = (threadIdx.x + i) & 3;
        int bin = (ci >> (24 - 8 * f)) & 255;
        const int higherBin = (bin >> 5) & mask;
        int offset = 4 * higherBin + f + ((bin & 31) << 5);

    Four features per word here, so the rotation is `& 3` rather than `& 7`.
    """
    comptime inner_bits = bits - 5
    comptime mask = (1 << inner_bits) - 1
    var f = (tid + i) & 3
    var bin = Int((ci >> UInt32(24 - 8 * f)) & UInt32(255))
    var higher_bin = (bin >> 5) & mask
    return 4 * higher_bin + f + ((bin & 31) << 5)


def one_byte_higher_bin[bits: Int](ci: UInt32, tid: Int, i: Int) -> Int:
    """`higherBin` alone: which PASS of the inner-bits loop may write."""
    comptime inner_bits = bits - 5
    comptime mask = (1 << inner_bits) - 1
    var f = (tid + i) & 3
    var bin = Int((ci >> UInt32(24 - 8 * f)) & UInt32(255))
    return (bin >> 5) & mask


def one_byte_hist_kernel[bits: Int](
    # `TFeatureInBlock*`, flattened to four parallel arrays so the kernel
    # takes plain pointers.
    feature_folds: MutPointer[UInt32, MutAnyOrigin],
    feature_fold_offset: MutPointer[UInt32, MutAnyOrigin],
    feature_group_offset: MutPointer[UInt32, MutAnyOrigin],
    feature_group_size: MutPointer[UInt32, MutAnyOrigin],
    f_count_in32: Int32,
    bins: MutPointer[UInt32, MutAnyOrigin],
    bins_line_size_in: Int32,
    stats: MutPointer[Float32, MutAnyOrigin],
    stat_line_size_in: Int32,
    part_offset: MutPointer[UInt32, MutAnyOrigin],
    part_size: MutPointer[UInt32, MutAnyOrigin],
    part_ids: MutPointer[UInt32, MutAnyOrigin],
    bin_sums: MutPointer[Float32, MutAnyOrigin],
    leaf_count_in: Int32,
    stat_count_in: Int32,
):
    """`ComputeSplitPropertiesDirectLoadsImpl` with `GroupSize = 4`.

    Grid, copied from `hist_one_byte.cu:284-290`:
        z = numStats, y = partCount, x = ceil(fCount/4) * replication

    `bits` is their template parameter, 5 to 8, and it decides how many
    PASSES the inner loop runs: a 5-bit feature costs one and an 8-bit
    feature costs eight. See `one_byte_bin_offset`.

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
    var feature_blocks = (f_count_in + 3) // 4
    var max_blocks_per_part = Int(grid_dim.x) // feature_blocks
    var feature_offset = (Int(block_idx.x) // max_blocks_per_part) * 4
    var f_count = min(f_count_in - feature_offset, 4)

    var bins_p = bins + bins_line_size * (
        Int(block_idx.x) // max_blocks_per_part
    )

    # THE RAGGED-LEAF TRICK, and it costs nothing. The grid is sized for the
    # worst-case leaf; every block reads its partition's size ON THE DEVICE
    # and the ones with no work return. No host round trip, no per-leaf
    # launch, no dependence on how unbalanced the level is.
    var local_block_idx = Int(block_idx.x) % max_blocks_per_part
    var min_docs_per_block = LANE_WIDTH * UNROLL * LOAD_SIZE * (
        ONE_BYTE_BLOCK_SIZE // LANE_WIDTH
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
        ONE_BYTE_HIST_SIZE,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var z = tid
    while z < ONE_BYTE_HIST_SIZE:
        smem[z] = Float32(0.0)
        z += ONE_BYTE_BLOCK_SIZE
    barrier()
    var slice_base = one_byte_slice_offset[bits](tid)

    # --- ALIGN_MEMORY(1), copied ---------------------------------------
    #
    # DEVIATION (PORTING.md 6): `AlignMemoryAccess` peels an unaligned prefix
    # so the vector loads that follow are aligned. At LOAD_SIZE 1 there is
    # nothing to align, so the peel is omitted rather than ported. It becomes
    # required the moment LOAD_SIZE moves above 1.
    var warps_per_block = ONE_BYTE_BLOCK_SIZE // LANE_WIDTH
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
        # `TPointHistOneByte::AddPoint`, copied. Four features per word, so
        # the rotation is `& 3`. The PASS LOOP is the difference from the
        # half-byte accumulator: the low five bits of the bin index the slot
        # directly and the HIGH bits are serialized, one pass each, with only
        # the lanes whose `higherBin == pass` writing. A 5-bit feature runs
        # one pass and an 8-bit feature eight, so wide features are paid for
        # in TIME rather than in shared memory.
        @parameter
        for k in range(UNROLL):
            @parameter
            for i in range(4):
                var slot = slice_base + one_byte_bin_offset[bits](
                    local_bins[k], tid, i
                )
                comptime inner_bits = bits - 5

                @parameter
                if inner_bits == 0:
                    barrier()
                    smem[slot] = smem[slot] + local_stats[k]
                else:
                    var higher = one_byte_higher_bin[bits](
                        local_bins[k], tid, i
                    )
                    comptime mask = (1 << inner_bits) - 1

                    @parameter
                    for kk in range(1 << inner_bits):
                        var p = ((tid >> 2) + kk) & mask
                        barrier()
                        if p == higher:
                            smem[slot] = smem[slot] + local_stats[k]

        b_ptr += stripe_size
        s_ptr += stripe_size

    # `TPointHistOneByte::Reduce`, the one-byte shape: fold the warp copies
    # down so feature `fid`'s bin `fold` lands at `Histogram[fid * histSize +
    # fold]`, which is what the writeback below reads.
    barrier()
    comptime hist_size_bins = 1 << (5 + (bits - 5))
    var slot_i = tid
    while slot_i < hist_size_bins * 4:
        var acc = Float32(0.0)
        var j = slot_i
        while j < ONE_BYTE_HIST_SIZE:
            acc += smem[j]
            j += hist_size_bins * 4
        barrier()
        smem[slot_i] = acc
        barrier()
        slot_i += ONE_BYTE_BLOCK_SIZE

    # `AddToGlobalMemory`, one-byte: one thread per FOLD, looping features.
    var fold = tid
    for fid in range(f_count):
        var folds = Int(feature_folds.unsafe_load(feature_offset + fid))
        if fold < folds:
            var group_offset = Int(feature_group_offset.unsafe_load(feature_offset + fid))
            var group_size = Int(feature_group_size.unsafe_load(feature_offset + fid))
            var device_offset = group_offset * stat_count * leaf_count
            var entries_per_leaf = stat_count * group_size
            var dst = (
                bin_sums
                + device_offset
                + Int(part_ids.unsafe_load(Int(block_idx.y))) * entries_per_leaf
                + Int(block_idx.z) * group_size
                + Int(feature_fold_offset.unsafe_load(feature_offset + fid))
            )

            var val = smem[fid * hist_size_bins + fold]

            if abs(val) > Float32(1e-20):
                # DEVIATION (numerics.mojo): CatBoost uses atomicAdd when
                # blockCount > 1 and a plain store otherwise. The atomic is
                # order-nondeterministic and is what `NUMERIC_IDENTICAL`
                # replaces with a fixed-point accumulator. Direct store here;
                # the multi-block flush lands with the replication work.
                dst.unsafe_store(fold, val)


def one_byte_hist_gather_kernel[bits: Int](
    # `TFeatureInBlock*`, flattened to four parallel arrays so the kernel
    # takes plain pointers.
    feature_folds: MutPointer[UInt32, MutAnyOrigin],
    feature_fold_offset: MutPointer[UInt32, MutAnyOrigin],
    feature_group_offset: MutPointer[UInt32, MutAnyOrigin],
    feature_group_size: MutPointer[UInt32, MutAnyOrigin],
    f_count_in32: Int32,
    cindex: MutPointer[UInt32, MutAnyOrigin],
    bins_line_size_in: Int32,
    indices: MutPointer[UInt32, MutAnyOrigin],
    stats: MutPointer[Float32, MutAnyOrigin],
    stat_line_size_in: Int32,
    part_offset: MutPointer[UInt32, MutAnyOrigin],
    part_size: MutPointer[UInt32, MutAnyOrigin],
    part_ids: MutPointer[UInt32, MutAnyOrigin],
    bin_sums: MutPointer[Float32, MutAnyOrigin],
    leaf_count_in: Int32,
    stat_count_in: Int32,
):
    """`ComputeSplitPropertiesGatherImpl` with `GroupSize = 4`.

    The gather instantiation, `GroupSize = 4`. Identical to the direct
    one except the bin is read through `indices`.

    Grid, copied from `hist_one_byte.cu:284-290`:
        z = numStats, y = partCount, x = ceil(fCount/4) * replication

    `bits` is their template parameter, 5 to 8, and it decides how many
    PASSES the inner loop runs: a 5-bit feature costs one and an 8-bit
    feature costs eight. See `one_byte_bin_offset`.

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
    var feature_blocks = (f_count_in + 3) // 4
    var max_blocks_per_part = Int(grid_dim.x) // feature_blocks
    var feature_offset = (Int(block_idx.x) // max_blocks_per_part) * 4
    var f_count = min(f_count_in - feature_offset, 4)

    var cindex_p = cindex + bins_line_size * (
        Int(block_idx.x) // max_blocks_per_part
    )
    var idx_p = indices

    # THE RAGGED-LEAF TRICK, and it costs nothing. The grid is sized for the
    # worst-case leaf; every block reads its partition's size ON THE DEVICE
    # and the ones with no work return. No host round trip, no per-leaf
    # launch, no dependence on how unbalanced the level is.
    var local_block_idx = Int(block_idx.x) % max_blocks_per_part
    var min_docs_per_block = LANE_WIDTH * UNROLL * LOAD_SIZE * (
        ONE_BYTE_BLOCK_SIZE // LANE_WIDTH
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
        ONE_BYTE_HIST_SIZE,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var z = tid
    while z < ONE_BYTE_HIST_SIZE:
        smem[z] = Float32(0.0)
        z += ONE_BYTE_BLOCK_SIZE
    barrier()
    var slice_base = one_byte_slice_offset[bits](tid)

    # --- ALIGN_MEMORY(1), copied ---------------------------------------
    #
    # DEVIATION (PORTING.md 6): `AlignMemoryAccess` peels an unaligned prefix
    # so the vector loads that follow are aligned. At LOAD_SIZE 1 there is
    # nothing to align, so the peel is omitted rather than ported. It becomes
    # required the moment LOAD_SIZE moves above 1.
    var warps_per_block = ONE_BYTE_BLOCK_SIZE // LANE_WIDTH
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
                # THE GATHER: position -> row -> bin. The compressed index
                # is never permuted, so after a reorder position no longer
                # names a row. See hist_binary for the full note.
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
        # `TPointHistOneByte::AddPoint`, copied. Four features per word, so
        # the rotation is `& 3`. The PASS LOOP is the difference from the
        # half-byte accumulator: the low five bits of the bin index the slot
        # directly and the HIGH bits are serialized, one pass each, with only
        # the lanes whose `higherBin == pass` writing. A 5-bit feature runs
        # one pass and an 8-bit feature eight, so wide features are paid for
        # in TIME rather than in shared memory.
        @parameter
        for k in range(UNROLL):
            @parameter
            for i in range(4):
                var slot = slice_base + one_byte_bin_offset[bits](
                    local_bins[k], tid, i
                )
                comptime inner_bits = bits - 5

                @parameter
                if inner_bits == 0:
                    barrier()
                    smem[slot] = smem[slot] + local_stats[k]
                else:
                    var higher = one_byte_higher_bin[bits](
                        local_bins[k], tid, i
                    )
                    comptime mask = (1 << inner_bits) - 1

                    @parameter
                    for kk in range(1 << inner_bits):
                        var p = ((tid >> 2) + kk) & mask
                        barrier()
                        if p == higher:
                            smem[slot] = smem[slot] + local_stats[k]

        i_ptr += stripe_size
        s_ptr += stripe_size

    # `TPointHistOneByte::Reduce`, the one-byte shape: fold the warp copies
    # down so feature `fid`'s bin `fold` lands at `Histogram[fid * histSize +
    # fold]`, which is what the writeback below reads.
    barrier()
    comptime hist_size_bins = 1 << (5 + (bits - 5))
    var slot_i = tid
    while slot_i < hist_size_bins * 4:
        var acc = Float32(0.0)
        var j = slot_i
        while j < ONE_BYTE_HIST_SIZE:
            acc += smem[j]
            j += hist_size_bins * 4
        barrier()
        smem[slot_i] = acc
        barrier()
        slot_i += ONE_BYTE_BLOCK_SIZE

    # `AddToGlobalMemory`, one-byte: one thread per FOLD, looping features.
    var fold = tid
    for fid in range(f_count):
        var folds = Int(feature_folds.unsafe_load(feature_offset + fid))
        if fold < folds:
            var group_offset = Int(feature_group_offset.unsafe_load(feature_offset + fid))
            var group_size = Int(feature_group_size.unsafe_load(feature_offset + fid))
            var device_offset = group_offset * stat_count * leaf_count
            var entries_per_leaf = stat_count * group_size
            var dst = (
                bin_sums
                + device_offset
                + Int(part_ids.unsafe_load(Int(block_idx.y))) * entries_per_leaf
                + Int(block_idx.z) * group_size
                + Int(feature_fold_offset.unsafe_load(feature_offset + fid))
            )

            var val = smem[fid * hist_size_bins + fold]

            if abs(val) > Float32(1e-20):
                # DEVIATION (numerics.mojo): CatBoost uses atomicAdd when
                # blockCount > 1 and a plain store otherwise. The atomic is
                # order-nondeterministic and is what `NUMERIC_IDENTICAL`
                # replaces with a fixed-point accumulator. Direct store here;
                # the multi-block flush lands with the replication work.
                dst.unsafe_store(fold, val)
