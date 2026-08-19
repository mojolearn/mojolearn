"""Per-leaf totals of every stat column.

MIRRORS `catboost/cuda/cuda_util/partitions_reduce.h`, whose
`ComputePartitionStats` is called once per level from
`split_properties_helper.cpp:1068`.

# =========================================================================
# DEVIATION BLOCK: their implementation is a CUB segmented reduce.
#
# `partitions_reduce.h` delegates to `SegmentedReduce` in `cuda_util/reduce.h`,
# which is CUB. There is no CUB in Mojo, so there is no line-for-line port and
# a reviewer diffing against their header will find no counterpart for what
# follows. Same situation as the stable partition in `split_points.mojo`.
#
# THE SEARCH, so it is not repeated: re-run against the docs 2026-08-19. What
# ships that is reduction-shaped is `max.gpu.primitives.block.{sum,max,min}`
# (BLOCK scope, and this file calls it), `std.gpu.primitives.warp.{sum,max,
# min,prefix_sum}` (WARP scope), and the `algorithm` package's row-wise
# scaffolder, which reduces the INNERMOST axis of a dense tensor. None of them
# is a device-wide reduce over RAGGED segments, which is what a leaf partition
# is: our segments are `[part_offset[leaf], + part_size[leaf])` and vary per
# leaf per level. `nn.argmaxmin` and `nn.cumsum` are dense-axis too, and
# `cumsum` is CPU-only besides. So the two phases below stay ours.
#
# What is NOT ours any more is the reduction inside each block. See the
# kernels.
# =========================================================================

WHY IT EXISTS, and what breaks without it
------------------------------------------
The score kernel derives each leaf's RIGHT side as `total - left`, so it needs
that leaf's total weight and total gradient. At depth 0 there is one leaf and
the total is the whole dataset, which is why a depth-0-only driver can hardcode
it and appear to work.

Below depth 0 it cannot. A tree grown with per-leaf totals left at zero runs,
conserves every row, produces `2^depth` partitions, and **splits nothing**:
measured here as 64 leaves of which 2 were non-empty at depth 6, because a
degenerate score re-picks the same feature every level and every row in a
child goes the same way. Everything about that failure looks like healthy
plumbing, which is why it needs its own kernel rather than a comment.

TWO PHASES, for the reason the partition has two
-------------------------------------------------
Phase 1 is grid-parallel over (chunk, leaf) and writes a partial sum per
chunk. Phase 2 reduces the partials per leaf. A one-block-per-leaf version
would be correct and would serialize the whole dataset through one threadgroup
at depth 0, which is the mistake this repository has now made twice and
measured twice.
"""

from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.primitives.block import sum as block_sum


comptime STATS_BLOCK = 256


def partition_stats_partial_kernel(
    leaves: MutPointer[UInt32, MutAnyOrigin],
    part_offset: MutPointer[UInt32, MutAnyOrigin],
    part_size: MutPointer[UInt32, MutAnyOrigin],
    stats: MutPointer[Float32, MutAnyOrigin],
    line_size_in: Int32,
    partials: MutPointer[Float32, MutAnyOrigin],
    max_chunks_in: Int32,
):
    """PHASE 1: one partial sum per (chunk, leaf, stat).

    Grid x is the chunk, y the leaf, z the stat plane, so every plane of
    every leaf is summed in one launch.

    The reduction is `max.gpu.primitives.block.sum`, `cub::BlockReduce`'s
    counterpart. It used to be a `prefix_sum` whose last lane was broadcast
    back, on the belief that Mojo exposed a scan and not a block reduce. That
    belief was false, and it cost a scan's dependency chain and a broadcast
    on every partial.
    """
    var line_size = Int(line_size_in)
    var max_chunks = Int(max_chunks_in)
    var leaf_slot = Int(block_idx.y)
    var leaf_id = Int(leaves.unsafe_load(leaf_slot))
    var stat = Int(block_idx.z)
    var n_stats = Int(grid_dim.z)

    var offset = Int(part_offset.unsafe_load(leaf_id))
    var size = Int(part_size.unsafe_load(leaf_id))

    var chunk = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var i = chunk * STATS_BLOCK + tid

    var v = Float32(0.0)
    if i < size:
        v = stats.unsafe_load(stat * line_size + offset + i)
    var total = block_sum[block_size=STATS_BLOCK](v)
    if tid == 0:
        partials.unsafe_store(
            (leaf_slot * n_stats + stat) * max_chunks + chunk, total
        )


def partition_stats_finish_kernel(
    leaves: MutPointer[UInt32, MutAnyOrigin],
    part_size: MutPointer[UInt32, MutAnyOrigin],
    partials: MutPointer[Float32, MutAnyOrigin],
    out_stats: MutPointer[Float32, MutAnyOrigin],
    max_chunks_in: Int32,
):
    """PHASE 2: sum a leaf's per-chunk partials into `partStats`.

    Writes `[leaf][stat]`, the layout `compute_optimal_splits_kernel` reads
    as `part_stats[leaf * stat_count + stat]`.
    """
    var max_chunks = Int(max_chunks_in)
    var leaf_slot = Int(block_idx.y)
    var leaf_id = Int(leaves.unsafe_load(leaf_slot))
    var stat = Int(block_idx.z)
    var n_stats = Int(grid_dim.z)
    var size = Int(part_size.unsafe_load(leaf_id))
    var n_chunks = (size + STATS_BLOCK - 1) // STATS_BLOCK
    var tid = Int(thread_idx.x)

    var acc = Float32(0.0)
    var c = tid
    while c < n_chunks:
        acc += partials.unsafe_load(
            (leaf_slot * n_stats + stat) * max_chunks + c
        )
        c += STATS_BLOCK
    var total = block_sum[block_size=STATS_BLOCK](acc)
    if tid == 0:
        out_stats.unsafe_store(leaf_id * n_stats + stat, total)


def compute_partition_stats(
    ctx: DeviceContext,
    n_leaf_slots: Int,
    max_leaf_rows: Int,
    n_stats: Int,
    line_size: Int,
    mut leaves: DeviceBuffer[DType.uint32],
    mut part_offset: DeviceBuffer[DType.uint32],
    mut part_size: DeviceBuffer[DType.uint32],
    mut stats: DeviceBuffer[DType.float32],
    mut partials: DeviceBuffer[DType.float32],
    mut out_stats: DeviceBuffer[DType.float32],
) raises:
    """`ComputePartitionStats`, both phases."""
    var max_chunks = (max_leaf_rows + STATS_BLOCK - 1) // STATS_BLOCK
    if max_chunks < 1:
        max_chunks = 1

    ctx.enqueue_function[partition_stats_partial_kernel](
        leaves.unsafe_ptr(),
        part_offset.unsafe_ptr(),
        part_size.unsafe_ptr(),
        stats.unsafe_ptr(),
        Int32(line_size),
        partials.unsafe_ptr(),
        Int32(max_chunks),
        grid_dim=(max_chunks, n_leaf_slots, n_stats),
        block_dim=(STATS_BLOCK, 1, 1),
    )
    ctx.enqueue_function[partition_stats_finish_kernel](
        leaves.unsafe_ptr(),
        part_size.unsafe_ptr(),
        partials.unsafe_ptr(),
        out_stats.unsafe_ptr(),
        Int32(max_chunks),
        grid_dim=(1, n_leaf_slots, n_stats),
        block_dim=(STATS_BLOCK, 1, 1),
    )
