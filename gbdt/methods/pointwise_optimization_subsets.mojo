# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CatBoost optimization-subset state and its depth-to-depth transition."""

from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.host.device_attribute import DeviceAttribute
from std.gpu import block_dim, block_idx, grid_dim, thread_idx

from gbdt.gpu_util.kernel.fill import launch_make_sequence
from gbdt.gpu_util.kernel.radix_sort import launch_radix_sort_bins
from gbdt.gpu_util.kernel.reorder_one_bit import REORDER_BLOCK
from gbdt.gpu_util.kernel.transform import launch_gather_with_mask_f32
from gbdt.methods.kernel.pointwise_scores import update_partition_props
from gbdt.gpu_util.partitions_reduce import (
    compute_partition_stats,
    partition_stats_chunks,
)


comptime SPLIT_BLOCK_SIZE = 256
"""`constexpr int blockSize = 256` (`gpu_data/kernel/split.cu:211`)."""

comptime SPLIT_MAX_BLOCKS = 65535
"""Stand-in for `TArchProps::MaxBlockCount()`; the kernel grid-strides, so any large constant gives the same answer."""

comptime POINTWISE_STAT_COUNT = 2
"""How many columns `TL2Target` carries and how many planes the partition reduce actually sums."""

comptime L2_PLANE_WEIGHT = 0
comptime L2_PLANE_TARGET = 1
"""`TL2Target`'s two columns, in the ORDER THE PARTITION STAT RECORD USES -- weight first, gradient second, matching `TPartitionStatistics{Weight, Sum}` and `greedy_search_helper.mojo:244`. What survives is the ORDER, which is still load-bearing -- it is the order `pack_partition_stats_kernel` writes and the order any future caller must hand `compute_hist2` its `target` and `weight`."""

comptime PART_OFFSET = 0
comptime PART_SIZE = 1
"""`TDataPartition`'s two `ui32`, in their declaration order (`cuda_util/gpu_data/partitions.h`)."""

comptime PARTITION_RECORD = 2
"""`sizeof(TDataPartition) / sizeof(ui32)`."""

comptime PART_STAT_WEIGHT = 0
comptime PART_STAT_SUM = 1
comptime PART_STAT_COUNT = 2
"""`TPartitionStatistics`'s three members, in their declaration order (`gpu_data/gpu_structures.h:113-116`)."""

comptime PARTITION_STAT_STRIDE = 3
"""`TPartitionStatistics` is three wide, so the record is three wide, even though only two of the three planes are reduced and only two are ever read."""

comptime GATHER_NO_MASK = UInt32(0xFFFFFFFF)
"""`GatherTarget` calls plain `Gather` (`weak_target_helpers.h:28-29`), not `GatherWithMask`."""


struct TL2Target(Movable):
    """`TL2Target<TStripeMapping>` (`methods/weak_target_helpers.h:11-14`), as TWO buffers, which is what it is upstream."""

    var weights: DeviceBuffer[DType.float32]
    """`Weights`. `TPartitionStatistics::Weight` is summed from this."""

    var weighted_target: DeviceBuffer[DType.float32]
    """`WeightedTarget`. `TPartitionStatistics::Sum` is summed from this."""

    var line_size: Int
    """Documents per column. Their `GetObjectsSlice().Size()`."""

    def __init__(
        out self,
        var weights: DeviceBuffer[DType.float32],
        var weighted_target: DeviceBuffer[DType.float32],
        line_size: Int,
    ):
        self.weights = weights^
        self.weighted_target = weighted_target^
        self.line_size = line_size


struct TOptimizationSubsets(Movable):
    """`TOptimizationSubsets<TStripeMapping, false>` (`pointwise_optimization_subsets.h:14-50`)."""

    var bins: DeviceBuffer[DType.uint32]
    """`Bins`."""

    var indices: DeviceBuffer[DType.uint32]
    """`Indices`."""

    var partitions: DeviceBuffer[DType.uint32]
    """`Partitions`: `TDataPartition[]` reinterpreted, two `UInt32` per record (DEVIATION 97.1)."""

    var partition_stats: DeviceBuffer[DType.float32]
    """`PartitionStats`, `[part * PARTITION_STAT_STRIDE + stat]`, three wide."""

    var count_dummy: DeviceBuffer[DType.float32]
    """Their `counts` argument, which `pointwise_kernels.h:240` passes as `nullptr` on this path so `PartitionUpdateImpl` takes `else { tmp = size; }`."""

    var gathered_weight: DeviceBuffer[DType.float32]
    var gathered_target: DeviceBuffer[DType.float32]
    """`Weights` and `WeightedTarget` AFTER `GatherTarget`, in current partition order. Handing it `buf.unsafe_ptr()` and `buf.unsafe_ptr().unsafe_offset(doc_count)` is refused: error: aliasing values passed mutably to 'target' argument and passed mutably to 'weight' argument `unsafe_bitcast[Float32]()` does not launder the origin, and the check fires at `enqueue_function` itself rather than only at `def` boundaries, so..."""

    var doc_count: Int
    var max_part_count: Int
    """`1 << (FoldBits + maxDepth)` (`pointwise_optimization_subsets.cpp:12`)."""

    var fold_count: UInt32
    """`FoldCount`."""

    var current_depth: UInt32
    """`CurrentDepth`."""

    var fold_bits: UInt32
    """`FoldBits`."""

    var sm_count: Int
    """`TArchProps::SMCount()`, cached at construction."""

    var tmp_bins: DeviceBuffer[DType.uint32]
    var tmp_indices: DeviceBuffer[DType.uint32]
    var scan_offsets: DeviceBuffer[DType.int32]
    var block_sums: DeviceBuffer[DType.int32]
    """`ReorderBins`'s double buffer and scan scratch."""

    var stat_partials: DeviceBuffer[DType.float32]
    """Phase 1's per-(chunk, part, stat) partials, their `tempVars`."""

    var reduce_offsets: DeviceBuffer[DType.uint32]
    var reduce_sizes: DeviceBuffer[DType.uint32]
    var reduce_weight: DeviceBuffer[DType.float32]
    var reduce_target: DeviceBuffer[DType.float32]
    """THE ADAPTER, and it is ours, not theirs."""

    var part_ids: DeviceBuffer[DType.uint32]
    """`compute_partition_stats` takes a `partIds` list, their argument of the same name."""

    def __init__(
        out self,
        var bins: DeviceBuffer[DType.uint32],
        var indices: DeviceBuffer[DType.uint32],
        var partitions: DeviceBuffer[DType.uint32],
        var partition_stats: DeviceBuffer[DType.float32],
        var count_dummy: DeviceBuffer[DType.float32],
        var gathered_weight: DeviceBuffer[DType.float32],
        var gathered_target: DeviceBuffer[DType.float32],
        var tmp_bins: DeviceBuffer[DType.uint32],
        var tmp_indices: DeviceBuffer[DType.uint32],
        var scan_offsets: DeviceBuffer[DType.int32],
        var block_sums: DeviceBuffer[DType.int32],
        var stat_partials: DeviceBuffer[DType.float32],
        var part_ids: DeviceBuffer[DType.uint32],
        var reduce_offsets: DeviceBuffer[DType.uint32],
        var reduce_sizes: DeviceBuffer[DType.uint32],
        var reduce_weight: DeviceBuffer[DType.float32],
        var reduce_target: DeviceBuffer[DType.float32],
        doc_count: Int,
        max_part_count: Int,
        sm_count: Int,
    ):
        self.bins = bins^
        self.indices = indices^
        self.partitions = partitions^
        self.partition_stats = partition_stats^
        self.count_dummy = count_dummy^
        self.gathered_weight = gathered_weight^
        self.gathered_target = gathered_target^
        self.tmp_bins = tmp_bins^
        self.tmp_indices = tmp_indices^
        self.scan_offsets = scan_offsets^
        self.block_sums = block_sums^
        self.stat_partials = stat_partials^
        self.part_ids = part_ids^
        self.reduce_offsets = reduce_offsets^
        self.reduce_sizes = reduce_sizes^
        self.reduce_weight = reduce_weight^
        self.reduce_target = reduce_target^
        self.doc_count = doc_count
        self.max_part_count = max_part_count
        self.fold_count = 0
        self.current_depth = 0
        self.fold_bits = 0
        self.sm_count = sm_count

    def current_part_count(self) -> Int:
        """`CurrentPartsView`'s slice length: `1ULL << (CurrentDepth + FoldBits)` (`pointwise_optimization_subsets.h:99`, `:112`, `:118`)."""
        return 1 << Int(self.current_depth + self.fold_bits)


def update_partition_offsets_kernel(
    partitions: MutPointer[UInt32, MutAnyOrigin],
    part_count_in: Int32,
    sorted_bins: MutPointer[UInt32, MutAnyOrigin],
    size_in: Int32,
):
    """`UpdatePartitionOffsets<TPartitionOffsetWriter, false>` (`cuda_util/kernel/partitions.cu:81-107`), the instantiation `UpdatePartitionDimensions` dispatches (`:128`)."""
    var size = Int(size_in)
    var part_count = UInt32(Int(part_count_in))
    var last_bin = UInt32(0xFFFFFFFF)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    while i < size:
        var bin0 = sorted_bins.unsafe_load(i)
        var bin1: UInt32
        if i > 0:
            bin1 = sorted_bins.unsafe_load(i - 1)
        else:
            bin1 = UInt32(0xFFFFFFFF)
        if bin0 != bin1:
            var b = bin0
            while b != bin1:
                partitions.unsafe_store(
                    Int(b) * PARTITION_RECORD + PART_OFFSET, UInt32(i)
                )
                b -= 1  # wraps past 0 to 0xFFFFFFFF, ending the i==0 walk
        if i + 1 == size:
            var limit = last_bin
            if part_count < limit:
                limit = part_count
            var b = bin0 + 1
            while b < limit:
                partitions.unsafe_store(
                    Int(b) * PARTITION_RECORD + PART_OFFSET, UInt32(size)
                )
                b += 1
        i += stride


def update_partition_sizes_kernel(
    partitions: MutPointer[UInt32, MutAnyOrigin],
    part_count_in: Int32,
    sorted_bins: MutPointer[UInt32, MutAnyOrigin],
    size_in: Int32,
):
    """`UpdatePartitionSizes` (`cuda_util/kernel/partitions.cu:14-38`), transcribed. **It READS `parts[b].Offset`, so the offsets kernel must have run first.** `Size` is computed as a difference against the offset, not counted."""
    var size = Int(size_in)
    var part_count = UInt32(Int(part_count_in))
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    while i < size:
        var bin0 = sorted_bins.unsafe_load(i)
        var bin1: UInt32
        if i > 0:
            bin1 = sorted_bins.unsafe_load(i - 1)
        else:
            bin1 = UInt32(0)
        if bin0 != bin1:
            var b = bin1
            while b < bin0:
                var off = partitions.unsafe_load(
                    Int(b) * PARTITION_RECORD + PART_OFFSET
                )
                partitions.unsafe_store(
                    Int(b) * PARTITION_RECORD + PART_SIZE, UInt32(i) - off
                )
                b += 1
        if (i + 1) == size:
            var off0 = partitions.unsafe_load(
                Int(bin0) * PARTITION_RECORD + PART_OFFSET
            )
            partitions.unsafe_store(
                Int(bin0) * PARTITION_RECORD + PART_SIZE, UInt32(size) - off0
            )
            var b = bin0 + 1
            while b < part_count:
                partitions.unsafe_store(
                    Int(b) * PARTITION_RECORD + PART_SIZE, UInt32(0)
                )
                b += 1
        i += stride


def zero_partitions_kernel(
    partitions: MutPointer[UInt32, MutAnyOrigin],
    part_count_in: Int32,
):
    """`ZeroPartitions` (`cuda_util/kernel/partitions.cu:109-117`)."""
    var part_count = Int(part_count_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    while i < part_count:
        partitions.unsafe_store(i * PARTITION_RECORD + PART_SIZE, UInt32(0))
        partitions.unsafe_store(i * PARTITION_RECORD + PART_OFFSET, UInt32(0))
        i += stride


def deinterleave_partitions_kernel(
    partitions: MutPointer[UInt32, MutAnyOrigin],
    part_offset: MutPointer[UInt32, MutAnyOrigin],
    part_size: MutPointer[UInt32, MutAnyOrigin],
    part_count_in: Int32,
):
    """ADAPTER, NOT A PORT."""
    var part_count = Int(part_count_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    while i < part_count:
        part_offset.unsafe_store(
            i, partitions.unsafe_load(i * PARTITION_RECORD + PART_OFFSET)
        )
        part_size.unsafe_store(
            i, partitions.unsafe_load(i * PARTITION_RECORD + PART_SIZE)
        )
        i += stride


def pack_partition_stats_kernel(
    reduced_weight: MutPointer[Float32, MutAnyOrigin],
    reduced_target: MutPointer[Float32, MutAnyOrigin],
    partitions: MutPointer[UInt32, MutAnyOrigin],
    part_stats: MutPointer[Float32, MutAnyOrigin],
    part_count_in: Int32,
):
    """ADAPTER, plus ONE line that IS theirs."""
    var part_count = Int(part_count_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    while i < part_count:
        part_stats.unsafe_store(
            i * PARTITION_STAT_STRIDE + PART_STAT_WEIGHT,
            reduced_weight.unsafe_load(i),
        )
        part_stats.unsafe_store(
            i * PARTITION_STAT_STRIDE + PART_STAT_SUM,
            reduced_target.unsafe_load(i),
        )
        part_stats.unsafe_store(
            i * PARTITION_STAT_STRIDE + PART_STAT_COUNT,
            Float32(
                Int(partitions.unsafe_load(i * PARTITION_RECORD + PART_SIZE))
            ),
        )
        i += stride


def launch_update_partition_dimensions(
    ctx: DeviceContext,
    mut partitions: DeviceBuffer[DType.uint32],
    part_count: Int,
    mut sorted_bins: DeviceBuffer[DType.uint32],
    size: Int,
) raises:
    """`UpdatePartitionDimensions` (`cuda_util/kernel/partitions.cu:121-135`)."""
    var num_blocks = (size + SPLIT_BLOCK_SIZE - 1) // SPLIT_BLOCK_SIZE
    if num_blocks > SPLIT_MAX_BLOCKS:
        num_blocks = SPLIT_MAX_BLOCKS

    if num_blocks > 0:
        ctx.enqueue_function[update_partition_offsets_kernel](
            partitions.unsafe_ptr(),
            Int32(part_count),
            sorted_bins.unsafe_ptr(),
            Int32(size),
            grid_dim=(num_blocks, 1, 1),
            block_dim=(SPLIT_BLOCK_SIZE, 1, 1),
        )
        ctx.enqueue_function[update_partition_sizes_kernel](
            partitions.unsafe_ptr(),
            Int32(part_count),
            sorted_bins.unsafe_ptr(),
            Int32(size),
            grid_dim=(num_blocks, 1, 1),
            block_dim=(SPLIT_BLOCK_SIZE, 1, 1),
        )
    else:
        var clear_blocks = (part_count + SPLIT_BLOCK_SIZE - 1) // (
            SPLIT_BLOCK_SIZE
        )
        if clear_blocks == 0:
            return
        ctx.enqueue_function[zero_partitions_kernel](
            partitions.unsafe_ptr(),
            Int32(part_count),
            grid_dim=(clear_blocks, 1, 1),
            block_dim=(SPLIT_BLOCK_SIZE, 1, 1),
        )


def update_bins_from_compressed_index_kernel(
    compressed_index: MutPointer[UInt32, MutAnyOrigin],
    indices: MutPointer[UInt32, MutAnyOrigin],
    size_in: Int32,
    feature_offset: UInt32,
    feature_mask: UInt32,
    feature_shift: UInt32,
    feature_one_hot: Int32,
    bin_idx: UInt32,
    depth: UInt32,
    bins: MutPointer[UInt32, MutAnyOrigin],
):
    """`UpdateBinsFromCompressedIndexImpl` (`gpu_data/kernel/split.cu:179-201`), transcribed."""
    var size = Int(size_in)
    var value = bin_idx << feature_shift
    var mask = feature_mask << feature_shift
    var f_offset = Int(feature_offset)
    var one_hot = feature_one_hot != Int32(0)

    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    while i < size:
        var idx = Int(indices.unsafe_load(i))
        var feature_val = compressed_index.unsafe_load(f_offset + idx) & mask
        var goes_right: Bool
        if one_hot:
            goes_right = feature_val == value
        else:
            goes_right = feature_val > value
        if goes_right:
            bins.unsafe_store(i, bins.unsafe_load(i) | (UInt32(1) << depth))
        i += stride


def update_bins_from_desc_kernel(
    compressed_index: MutPointer[UInt32, MutAnyOrigin],
    indices: MutPointer[UInt32, MutAnyOrigin],
    size_in: Int32,
    split_desc: MutPointer[UInt32, MutAnyOrigin],
    depth: UInt32,
    bins: MutPointer[UInt32, MutAnyOrigin],
):
    """`update_bins_from_compressed_index_kernel` with the five feature scalars read from the DEVICE descriptor the pack kernel wrote (`kernel/pointwise_split_resolve.mojo`) instead of arriving as kernel arguments -- DEVIATION 207, the blind level loop."""
    var size = Int(size_in)
    var f_offset = Int(split_desc.unsafe_load(0))
    var feature_shift = split_desc.unsafe_load(2)
    var value = split_desc.unsafe_load(4) << feature_shift
    var mask = split_desc.unsafe_load(1) << feature_shift
    var one_hot = split_desc.unsafe_load(3) != UInt32(0)

    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    while i < size:
        var idx = Int(indices.unsafe_load(i))
        var feature_val = compressed_index.unsafe_load(f_offset + idx) & mask
        var goes_right: Bool
        if one_hot:
            goes_right = feature_val == value
        else:
            goes_right = feature_val > value
        if goes_right:
            bins.unsafe_store(i, bins.unsafe_load(i) | (UInt32(1) << depth))
        i += stride


def launch_update_bin_from_compressed_index(
    ctx: DeviceContext,
    mut compressed_index: DeviceBuffer[DType.uint32],
    mut docs_for_bins: DeviceBuffer[DType.uint32],
    size: Int,
    feature_offset: UInt32,
    feature_mask: UInt32,
    feature_shift: UInt32,
    feature_one_hot: Bool,
    bin_idx: UInt32,
    depth: UInt32,
    mut bins: DeviceBuffer[DType.uint32],
) raises:
    """`UpdateBinsFromCompressedIndex` (`gpu_data/kernel/split.cu:203-218`), which is what `UpdateBinFromCompressedIndex` (`gpu_data/splitter.h:174-182`) dispatches through `TUpdateBinsFromCompressedIndexKernel`."""
    var num_blocks = (size + SPLIT_BLOCK_SIZE - 1) // SPLIT_BLOCK_SIZE
    if num_blocks > SPLIT_MAX_BLOCKS:
        num_blocks = SPLIT_MAX_BLOCKS
    if num_blocks == 0:
        return
    ctx.enqueue_function[update_bins_from_compressed_index_kernel](
        compressed_index.unsafe_ptr(),
        docs_for_bins.unsafe_ptr(),
        Int32(size),
        feature_offset,
        feature_mask,
        feature_shift,
        Int32(1) if feature_one_hot else Int32(0),
        bin_idx,
        depth,
        bins.unsafe_ptr(),
        grid_dim=(num_blocks, 1, 1),
        block_dim=(SPLIT_BLOCK_SIZE, 1, 1),
    )


def update_subsets_stats(
    ctx: DeviceContext,
    mut source: TL2Target,
    mut subsets: TOptimizationSubsets,
) raises:
    """`UpdateSubsetsStats` (`pointwise_optimization_subsets.h:53-70`), call for call."""
    var part_count = subsets.current_part_count()
    var adapt_blocks = (part_count + SPLIT_BLOCK_SIZE - 1) // SPLIT_BLOCK_SIZE
    if adapt_blocks < 1:
        adapt_blocks = 1

    launch_update_partition_dimensions(
        ctx,
        subsets.partitions,
        part_count,
        subsets.bins,
        subsets.doc_count,
    )

    launch_gather_with_mask_f32(
        ctx,
        subsets.gathered_weight,
        source.weights,
        subsets.indices,
        subsets.doc_count,
        GATHER_NO_MASK,
    )
    launch_gather_with_mask_f32(
        ctx,
        subsets.gathered_target,
        source.weighted_target,
        subsets.indices,
        subsets.doc_count,
        GATHER_NO_MASK,
    )

    update_partition_props(
        ctx,
        subsets.gathered_target,
        subsets.gathered_weight,
        subsets.count_dummy,
        True,
        True,
        False,
        subsets.partitions,
        subsets.partition_stats,
        part_count,
    )


def create_subsets(
    ctx: DeviceContext,
    max_depth: Int,
    mut source: TL2Target,
    fold_count: Int = 0,
    fold_bits: Int = 0,
    sm_count: Int = -1,
) raises -> TOptimizationSubsets:
    """`TSubsetsHelper<TStripeMapping>::CreateSubsets` (`pointwise_optimization_subsets.cpp:4-24`), line for line."""
    if max_depth < 0:
        raise Error(
            String("maxDepth must be non-negative, got ") + String(max_depth)
        )
    var doc_count = source.line_size

    var max_part_count = 1 << (fold_bits + max_depth)

    var bins = ctx.enqueue_create_buffer[DType.uint32](doc_count)
    var indices = ctx.enqueue_create_buffer[DType.uint32](doc_count)
    var partitions = ctx.enqueue_create_buffer[DType.uint32](
        max_part_count * PARTITION_RECORD
    )
    var part_stats = ctx.enqueue_create_buffer[DType.float32](
        max_part_count * PARTITION_STAT_STRIDE
    )
    var gathered_weight = ctx.enqueue_create_buffer[DType.float32](doc_count)
    var gathered_target = ctx.enqueue_create_buffer[DType.float32](doc_count)

    var tmp_bins = ctx.enqueue_create_buffer[DType.uint32](doc_count)
    var tmp_indices = ctx.enqueue_create_buffer[DType.uint32](doc_count)
    var scan_offsets = ctx.enqueue_create_buffer[DType.int32](doc_count)
    var n_scan_blocks = (doc_count + REORDER_BLOCK - 1) // REORDER_BLOCK
    if n_scan_blocks < 1:
        n_scan_blocks = 1
    var block_sums = ctx.enqueue_create_buffer[DType.int32](n_scan_blocks)

    var sm = sm_count
    if sm < 0:
        sm = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
    var chunks = partition_stats_chunks(sm, 1)
    var stat_partials = ctx.enqueue_create_buffer[DType.float32](
        max_part_count * 1 * chunks
    )

    var part_ids = ctx.enqueue_create_buffer[DType.uint32](max_part_count)

    var reduce_offsets = ctx.enqueue_create_buffer[DType.uint32](max_part_count)
    var reduce_sizes = ctx.enqueue_create_buffer[DType.uint32](max_part_count)
    var reduce_weight = ctx.enqueue_create_buffer[DType.float32](max_part_count)
    var reduce_target = ctx.enqueue_create_buffer[DType.float32](max_part_count)

    var count_dummy = ctx.enqueue_create_buffer[DType.float32](1)

    var subsets = TOptimizationSubsets(
        bins^,
        indices^,
        partitions^,
        part_stats^,
        count_dummy^,
        gathered_weight^,
        gathered_target^,
        tmp_bins^,
        tmp_indices^,
        scan_offsets^,
        block_sums^,
        stat_partials^,
        part_ids^,
        reduce_offsets^,
        reduce_sizes^,
        reduce_weight^,
        reduce_target^,
        doc_count,
        max_part_count,
        sm,
    )

    reset_subsets(ctx, subsets, source, fold_count, fold_bits)
    return subsets^


def reset_subsets(
    ctx: DeviceContext,
    mut subsets: TOptimizationSubsets,
    mut source: TL2Target,
    fold_count: Int = 0,
    fold_bits: Int = 0,
) raises:
    """`CreateSubsets`' STATE half, split from its ALLOCATION half so a pooled `TOptimizationSubsets` can be reused across trees: every buffer is shape-keyed (`doc_count`, `max_part_count`) and only what this function writes depends on the TREE -- the target changes with every boosting iteration, the counters and seeds do not carry."""
    if source.line_size != subsets.doc_count:
        raise Error(
            "reset_subsets: source has "
            + String(source.line_size)
            + " documents but the pooled subsets were built for "
            + String(subsets.doc_count)
            + " -- the pool key must include doc_count"
        )

    subsets.current_depth = 0
    subsets.fold_count = UInt32(fold_count)
    subsets.fold_bits = UInt32(fold_bits)

    ctx.enqueue_memset(subsets.bins, UInt32(0))
    launch_make_sequence(ctx, UInt32(0), subsets.indices, subsets.doc_count)
    launch_make_sequence(
        ctx, UInt32(0), subsets.part_ids, subsets.max_part_count
    )

    update_subsets_stats(ctx, source, subsets)


def split_subsets(
    ctx: DeviceContext,
    mut source: TL2Target,
    mut compressed_index: DeviceBuffer[DType.uint32],
    mut docs_for_bins: DeviceBuffer[DType.uint32],
    feature_offset: UInt32,
    feature_mask: UInt32,
    feature_shift: UInt32,
    feature_one_hot: Bool,
    bin_idx: UInt32,
    mut subsets: TOptimizationSubsets,
) raises:
    """`TSubsetsHelper<TStripeMapping>::Split` (`pointwise_optimization_subsets.cpp:26-52`), call for call. **`ReorderBins(..., offset, 1)` IS ONE BIT AND IT MUST BE STABLE.** The documents were already grouped by their first `d` bits; sorting on bit `d` alone splits every existing group in two IN PLACE only because equal keys keep their relative order."""
    var depth = subsets.current_depth + subsets.fold_bits
    if Int(depth) >= 32:
        raise Error(
            String("Split at depth ") + String(depth)
            + " would write bit " + String(depth)
            + " of a ui32 bin; CatBoost's ReorderBins asserts"
            " (offset + bits) <= 32 (cuda_util/sort.cpp:557)"
        )

    launch_update_bin_from_compressed_index(
        ctx,
        compressed_index,
        docs_for_bins,
        subsets.doc_count,
        feature_offset,
        feature_mask,
        feature_shift,
        feature_one_hot,
        bin_idx,
        depth,
        subsets.bins,
    )

    launch_radix_sort_bins(
        ctx,
        subsets.doc_count,
        Int(depth),
        Int(depth) + 1,
        subsets.bins,
        subsets.indices,
        subsets.tmp_bins,
        subsets.tmp_indices,
        subsets.scan_offsets,
        subsets.block_sums,
    )

    subsets.current_depth += 1

    update_subsets_stats(ctx, source, subsets)


def split_subsets_from_desc(
    ctx: DeviceContext,
    mut source: TL2Target,
    mut compressed_index: DeviceBuffer[DType.uint32],
    mut docs_for_bins: DeviceBuffer[DType.uint32],
    mut split_desc: DeviceBuffer[DType.uint32],
    mut subsets: TOptimizationSubsets,
) raises:
    """`split_subsets` consuming the winner from the DEVICE descriptor the pack kernel wrote, instead of five host scalars -- DEVIATION 207, the blind level loop's split."""
    var depth = subsets.current_depth + subsets.fold_bits
    if Int(depth) >= 32:
        raise Error(
            String("Split at depth ") + String(depth)
            + " would write bit " + String(depth)
            + " of a ui32 bin; CatBoost's ReorderBins asserts"
            " (offset + bits) <= 32 (cuda_util/sort.cpp:557)"
        )

    var num_blocks = (subsets.doc_count + SPLIT_BLOCK_SIZE - 1) // (
        SPLIT_BLOCK_SIZE
    )
    if num_blocks > SPLIT_MAX_BLOCKS:
        num_blocks = SPLIT_MAX_BLOCKS
    if num_blocks > 0:
        ctx.enqueue_function[update_bins_from_desc_kernel](
            compressed_index.unsafe_ptr(),
            docs_for_bins.unsafe_ptr(),
            Int32(subsets.doc_count),
            split_desc.unsafe_ptr(),
            depth,
            subsets.bins.unsafe_ptr(),
            grid_dim=(num_blocks, 1, 1),
            block_dim=(SPLIT_BLOCK_SIZE, 1, 1),
        )

    launch_radix_sort_bins(
        ctx,
        subsets.doc_count,
        Int(depth),
        Int(depth) + 1,
        subsets.bins,
        subsets.indices,
        subsets.tmp_bins,
        subsets.tmp_indices,
        subsets.scan_offsets,
        subsets.block_sums,
    )

    subsets.current_depth += 1

    update_subsets_stats(ctx, source, subsets)
