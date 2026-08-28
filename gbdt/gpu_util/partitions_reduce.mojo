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
#
# DEVIATION BLOCK 2: they accumulate in `double`, we accumulate in Float32.
#
# `UpdatePartitionsPropsImpl` (`cuda_util/kernel/update_part_props.cu`) is
# double from end to end: `ComputeSum` returns `double`, `localBuffer` is a
# `volatile double[BlockSize]`, `tempVars` is `double*`, and `statSums` is
# `double`. Every buffer below is Float32 instead.
#
# THE REASON IS THE TARGET, not a judgement about accuracy: Metal has no
# double, which is the same wall `jacobi_eigh_device.mojo` records and the
# reason the device Jacobi is Float32 as well. There is nothing to trade off
# here and no knob to expose; a Float64 buffer does not compile for the
# device we ship on first.
#
# WHAT IT COSTS IS UNPRICED and this block does not pretend otherwise. The
# per-chunk partial sums at most `STATS_BLOCK` values, so phase 1 is short.
# Phase 2 sums up to one partial per chunk of the widest leaf, and the leaf
# values that come out of it feed a Newton ratio, so the error is relative
# and small rather than catastrophic. No measurement has been taken against a
# Float64 host reduction of the same fixture. Until one is,
# `mojo_only/partitions_reduce_check.mojo` sidesteps the question entirely by
# planting integer-valued stats small enough that every intermediate sum is
# exact in Float32, which is what lets it compare with EQUALITY and no
# tolerance at all.
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
Phase 1 is grid-parallel over (block, leaf) and writes one partial sum per
block, each block STRIPING over the rows it owns the way `ComputeSum` does.
Phase 2 reduces one partial per launched block, per leaf, the way
`SaveResultsImpl` does. A one-block-per-leaf version would be correct and
would serialize the whole dataset through one threadgroup at depth 0, which
is the mistake this repository has now made twice and measured twice.

Neither phase re-derives the block count from a partition's length. The
launcher owns that number and hands it to both, which is theirs
(`tempVarsBlockCount` is `numBlocks.x`) and is the reason `max_leaf_rows` is
a grid hint here rather than a correctness precondition. It was one for a
while, silently. `mojo_only/partitions_reduce_check.mojo` runs a 40,000-row
leaf through a one-block grid to keep it from becoming one again.
"""

from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.host.device_attribute import DeviceAttribute
from max.gpu.primitives.block import sum as block_sum

from gbdt.targets.kernel.pointwise_targets import pinned_block_sum

from mojo_only.kernel_matrix import partition_chunks_sm_for
from mojo_only.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL


# `const ui32 blockSize = 512` (`update_part_props.cu:209`). Was 256, which
# was ours, not theirs.
comptime STATS_BLOCK = 512


def partition_stats_partial_kernel(
    leaves: MutPointer[UInt32, MutAnyOrigin],
    part_offset: MutPointer[UInt32, MutAnyOrigin],
    part_size: MutPointer[UInt32, MutAnyOrigin],
    stats: MutPointer[Float32, MutAnyOrigin],
    line_size_in: Int32,
    partials: MutPointer[Float32, MutAnyOrigin],
    max_chunks_in: Int32,
):
    """PHASE 1: one partial sum per (block, leaf, stat).

    Grid x is the block, y the leaf, z the stat plane, so every plane of
    every leaf is summed in one launch.

    The reduction is `max.gpu.primitives.block.sum`, `cub::BlockReduce`'s
    counterpart. It used to be a `prefix_sum` whose last lane was broadcast
    back, on the belief that Mojo exposed a scan and not a block reduce. That
    belief was false, and it cost a scan's dependency chain and a broadcast
    on every partial.

    THE STRIPE IS THEIRS AND IT IS NOT DECORATION. `ComputeSum`
    (`cuda_util/kernel/update_part_props.cu`) walks its partition with
    `stripeSize = entriesPerWarp * warpsPerBlock * blockCount`, so block `b`
    reads its own slice and then every `blockCount`-th slice after it and the
    grid covers the whole partition WHATEVER its width. Theirs has to: their
    grid x is `CeilDivide(2 * TArchProps::SMCount(), statCount)`, a machine
    constant that knows nothing about how long a leaf is.

    This kernel used to read exactly ONE chunk per block and stop, so every
    row past `gridDim.x * STATS_BLOCK` of a partition was DROPPED, in
    silence, with a well-formed answer on every leaf that happened to fit.
    That made `max_leaf_rows` a correctness precondition the signature never
    stated. It is now what it is in their code: a grid-sizing hint. Passing
    one that is too small costs launches, not rows, and
    `check_partitions_reduce_narrow_grid` runs exactly that case.

    TWIN (DEVIATION 1902): `partition_stats_partial_gather_kernel`
    (`gpu_util/kernel/partition_stats_gather.mojo`) is this body with the
    stat load gathered through `row_index`, for the ridx-only split route.
    An edit to this loop must visit that one; the reason the body is
    doubled rather than parameterized is on that file's banner.
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

    # `blockCount` is `gridDim.x`, exactly as theirs is: the stride has to be
    # the grid that is actually running, not the buffer's row width.
    var stride = Int(grid_dim.x) * STATS_BLOCK
    var v = Float32(0.0)
    var i = chunk * STATS_BLOCK + tid
    while i < size:
        v += stats.unsafe_load(stat * line_size + offset + i)
        i += stride
    # IDENTITY_PATHS row 8, the last producer pair (E1 2026-08-22: the
    # gbdt_rmse card diverged Apple<->AMD at the FIRST pstats stage --
    # this fold, at hardware warp width). `pinned_block_sum`'s FAST arm
    # IS the library call; IDENTICAL folds one shape on every vendor.
    var total = pinned_block_sum[STATS_BLOCK](v)
    if tid == 0:
        partials.unsafe_store(
            (leaf_slot * n_stats + stat) * max_chunks + chunk, total
        )


def partition_stats_finish_kernel(
    leaves: MutPointer[UInt32, MutAnyOrigin],
    partials: MutPointer[Float32, MutAnyOrigin],
    out_stats: MutPointer[Float32, MutAnyOrigin],
    max_chunks_in: Int32,
):
    """PHASE 2: sum a leaf's per-block partials into `partStats`.

    `SaveResultsImpl` (`cuda_util/kernel/update_part_props.cu`), whose loop is

        for (int x = 0; x < tempVarsBlockCount; ++x) {
            total += tempVars[i];  tempVars += statCount * partCount;
        }

    `tempVarsBlockCount` is `numBlocks.x` handed down from the host, so they
    read one partial PER LAUNCHED BLOCK and never re-derive the count from
    the partition. `max_chunks` is that argument.

    This used to recompute `ceil(part_size / STATS_BLOCK)` from the size
    instead. Two things were wrong with that. It read `max_chunks` entries'
    worth of stride while indexing up to the leaf's own chunk count, so a
    leaf longer than `max_leaf_rows` walked out of its own row of `partials`
    and into the next leaf's. And it duplicated a number the launcher already
    knows, which is how the two could disagree at all. `part_size` is
    therefore no longer an argument, exactly as it is not one of theirs.

    Blocks past a leaf's length wrote `0.0` in phase 1, so summing all of
    them is the same arithmetic as summing the occupied ones.

    Writes `[leaf][stat]`, the layout `compute_optimal_splits_kernel` reads
    as `part_stats[leaf * stat_count + stat]`.
    """
    var max_chunks = Int(max_chunks_in)
    var leaf_slot = Int(block_idx.y)
    var leaf_id = Int(leaves.unsafe_load(leaf_slot))
    var stat = Int(block_idx.z)
    var n_stats = Int(grid_dim.z)
    var tid = Int(thread_idx.x)

    var acc = Float32(0.0)
    var c = tid
    while c < max_chunks:
        acc += partials.unsafe_load(
            (leaf_slot * n_stats + stat) * max_chunks + c
        )
        c += STATS_BLOCK
    # IDENTITY_PATHS row 8 (see the sibling fold above).
    var total = pinned_block_sum[STATS_BLOCK](acc)
    if tid == 0:
        out_stats.unsafe_store(leaf_id * n_stats + stat, total)


def partition_stats_chunks(sm_count: Int, n_stats: Int) -> Int:
    """`numBlocks.x = CeilDivide(2 * TArchProps::SMCount(), statCount)`
    (`update_part_props.cu:215`). Exported so the check derives its expected
    `partials` layout from the SAME formula the launch uses; an expectation
    that recomputes this its own way is an expectation that follows nothing.

    **AND IT IS A NUMERIC ROW, WHICH IS WHY THE PIN IS APPLIED HERE AND NOT
    AT A CALL SITE.** The chunk count partitions a FLOAT sum -- each chunk is
    reduced by `block.sum` and the partials are added in phase 2 -- so the
    machine's core count decides the last bits of every per-leaf stat, and
    per-leaf stats are the LEAF VALUES. Under `IDENTICAL` the count therefore
    comes from `kernel_matrix.partition_chunks_sm_for`, pinned, so the
    partition has one shape on every vendor.

    This function is the only place the formula lives, and both readers --
    the launch and the `stat_partials` buffer sizing in
    `greedy_search_helper` -- go through it. That matters more than it looks:
    if the pin were applied at the launch only, the buffer would be sized
    from the device count and the kernel would index past it. One formula,
    one pin, both readers.
    """
    comptime _identical = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var sm = partition_chunks_sm_for[_identical](sm_count)
    var chunks = (2 * sm + n_stats - 1) // n_stats
    if chunks < 1:
        chunks = 1
    return chunks


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
    sm_count: Int = -1,
) raises:
    """`ComputePartitionStats`, both phases.

    `max_leaf_rows` sizes the grid and NOTHING ELSE. Both kernels are correct
    at any width, because phase 1 stripes the way `ComputeSum` does and phase
    2 reads one partial per launched block the way `SaveResultsImpl` does. A
    caller that under-reports the widest leaf pays in occupancy, not in
    dropped rows.

    THE GRID IS NOW THEIRS: `numBlocks.x = CeilDivide(2 *
    TArchProps::SMCount(), statCount)` at `blockSize = 512`
    (`update_part_props.cu:209-215`), MACHINE-sized. This stood as an OPEN
    deviation, data-sized at `ceil(max_leaf_rows / STATS_BLOCK)`, waiting
    for a measurement; the 2026-08-19 phase itemization supplied one (5.4
    ms/tree at 800k x 100 depth 6, ~40x under the streaming bandwidth this
    box measures on the same buffers). At depth 0 on 800k rows the old
    formula asked for 3125 blocks in x where theirs asks for a few dozen.

    `max_leaf_rows` is no longer read; the parameter stays because the call
    sites live in `greedy_search_helper.mojo` and are not this file's to
    edit this round.

    NUMERIC NOTE, not scheduling: the x count is a summation-order choice
    for a FLOAT tree reduction, so moving it moves last bits. That is
    THEIR design too -- their count reads `SMCount()` on whatever device
    they run -- and the oracle gate is the arbiter of whether the model
    moved. It did not (48 of 48 after this change).
    """
    _ = max_leaf_rows
    # `sm_count` is threaded in rather than queried here because ONE
    # `ctx.get_attribute` costs 1.26 ms on this Metal device (measured, 100
    # calls in 126 ms) and this function runs once per level. Their
    # `TArchProps::SMCount()` is a CACHED static read once at init, so the
    # caching is theirs; querying per call was ours and cost ~9 ms/tree.
    var sm = sm_count
    if sm < 0:
        sm = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
    var max_chunks = partition_stats_chunks(sm, n_stats)

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
        partials.unsafe_ptr(),
        out_stats.unsafe_ptr(),
        Int32(max_chunks),
        grid_dim=(1, n_leaf_slots, n_stats),
        block_dim=(STATS_BLOCK, 1, 1),
    )
