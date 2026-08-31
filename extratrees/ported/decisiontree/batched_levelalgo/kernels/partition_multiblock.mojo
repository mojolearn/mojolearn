# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Derivative work: the upstream file and its pinned commit are recorded in this lane's PORTED_MAP.tsv and in this file's own docstring. See NOTICE.
"""The frontier partition, spread across blocks instead of one block per node.

==========================================================================
DEVIATION BLOCK -- DEVIATION 203. THE PARTITION IS MULTI-BLOCK.

THEIRS. `launchNodeSplitKernel` is `<<<work_items_size, TPB>>>`
(`builder_kernels_impl.cuh:109-134`): ONE BLOCK PER NODE, and inside it
`partitionSamples` (`:43-88`) walks two cursors down the node's range
swapping misfits in pairs. `node_split_kernel` in
`builder_kernels_impl.mojo` is that, transcribed, and DEVIATION 177
recorded the shape as theirs and kept it.

WHY IT IS BEING CHANGED, MEASURED. That grid has `n_nodes` blocks, and
the root level has ONE node. On covtype at 581,012 rows the root's
partition is therefore one threadgroup walking 581,012 rows, and the
first four levels are nearly serial. The signature is visible from
outside the kernel: at 145,253 rows a whole fit is FASTER with
`max_features=54` (10.7 ms/level, 11,080 nodes) than with
`max_features=5` (14.4 ms/level, 1,244 nodes), even though the wide arm
does ten times the split-search work. The only quantity that moves the
right way is the NODE COUNT, and the only per-node-serial pass is this
one. A100s hide it under 108 SMs; ten Apple GPU cores do not.

WHAT REPLACES IT. A counting partition, in three passes over the same
`WorkloadInfo` flattening the split search already uses, so the grid is
`(n_blocks_dimx, 1, 1)` -- blocks over ROWS, not over nodes:

  1. `partition_count_kernel`   -- each block counts its own chunk's
     left-going rows into `blk_left[b]`;
  2. `partition_scan_kernel`    -- one block per node runs an exclusive
     scan over that node's block counts. Still per-node-serial, but over
     `ceil(count/TPB)` elements instead of `count` -- 4,540 instead of
     581,012 at the root, a 128x reduction in the serial term;
  3. `partition_scatter_kernel` -- each block scatters its chunk into
     `row_ids_alt` at the offsets the scan produced, and
     `partition_writeback_kernel` copies the touched ranges back.

THE ANSWER IS UNCHANGED AND THAT IS CHECKABLE, NOT HOPED FOR. This
produces a DIFFERENT ORDER within each side than their pairwise-swap
does -- theirs is whatever the swap sequence leaves, this is stable by
block. Nothing downstream reads that order: a node's split search takes
min, max and INTEGER class counts over its rows (deviation 135 put the
regression accumulator in fixed point precisely so that sums are
order-independent), the leaf pass sums the same integers, and the child
ranges come from `split.n_left`, not from the arrangement. So the tree
must come out BIT-IDENTICAL, and `partition_multiblock_check` asserts
exactly that against the one-block kernel, cell by cell, rather than
asserting the two partitions are "equivalent".

WHAT IS KEPT. `split_not_valid` is called on the same four fields, in
the same place, so a node the ported kernel skips is a node this skips.
`node_split_kernel` STAYS in the tree and stays checked: it is the
oracle this is compared against, and deleting the thing you are
verified against leaves nothing to verify against tomorrow.
==========================================================================
"""

from std.gpu import block_idx, thread_idx
from max.gpu.primitives.block import prefix_sum
from max.gpu.primitives.block import sum as block_sum

from extratrees.ported.decisiontree.batched_levelalgo.kernels.builder_kernels import (
    NodeWorkItem,
    WorkloadInfo,
    split_not_valid,
)
from extratrees.ported.decisiontree.batched_levelalgo.split import Split


comptime PART_MB_SAB_NONE: Int32 = 0
comptime PART_MB_SAB_NO_VALID_GUARD: Int32 = 1
"""Ignore `split_not_valid`, so a node the shipping path skips is partitioned.
Rule 8: the guard has two sides and both are exercised."""
comptime PART_MB_SAB_SINGLE_BLOCK: Int32 = 2
"""Every block claims to be block 0 of its node, which collapses the scatter
onto one chunk's offsets. The point of the multi-block form is that
`offset_blockid` is READ; if it were ignored the answer would still look
plausible for one-block nodes."""
comptime PART_MB_SAB_NO_SCAN: Int32 = 3
"""Scatter with a zero cross-block offset, i.e. as if every block were first in
its node. Distinguishes "the scan ran" from "the scan happened to be zero",
which is the only thing that separates a working prefix from a no-op at the
root, where there IS only one block."""


def _block_scan[
    TPB: Int
](flag: Int32) -> Tuple[Int32, Int32]:
    """`(exclusive_prefix_of_this_thread, block_total)` over `TPB` flags.

    cuML calls `cub::BlockScan(...).ExclusiveSum(flag, prefix, aggregate)`
    (`builder_kernels_impl.cuh:69-71`), and rule 0b-i says port the CALL, not a
    hand-written replacement: MAX's `prefix_sum` and `sum` are the
    counterparts. CUB returns the aggregate through a third argument; MAX
    spells it as a separate broadcast reduction, so this is two collectives
    where theirs is one.
    """
    var excl = prefix_sum[block_size=TPB, exclusive=True](flag)
    var total = block_sum[block_size=TPB, broadcast=True](flag)
    return (excl, total)


def _goes_left(
    data: MutPointer[Float32, MutAnyOrigin],
    row_ids: MutPointer[Int32, MutAnyOrigin],
    slot: Int,
    col_offset: Int,
    quesval: Float32,
) -> Int32:
    """`col[row_ids[slot]] <= quesval`.

    THE DIRECTION IS THEIRS. `partitionSamples` calls a LEFT slot a misfit
    when `col[row_ids[loff]] > quesval` (`builder_kernels_impl.cuh:65-66`), so
    a row EQUAL to the threshold belongs LEFT. sklearn's partitioner agrees
    (`_partitioner.pyx:233-236`). Written once, here, so the count pass and
    the scatter pass cannot drift from each other.
    """
    var row = Int(row_ids[unsafe_offset=slot])
    return 1 if data[unsafe_offset = col_offset + row] <= quesval else 0


def _skip_node(
    work_items: MutPointer[NodeWorkItem, MutAnyOrigin],
    splits: MutPointer[Split, MutAnyOrigin],
    nid: Int,
    min_impurity_decrease: Float32,
    min_samples_leaf: Int32,
    sabotage: Int32,
) -> Bool:
    """`split_not_valid` on the same four fields `node_split_kernel` reads."""
    var range_len = Int(work_items[unsafe_offset=nid].instances.count)
    var invalid = split_not_valid(
        Split(
            splits[unsafe_offset=nid].quesval,
            splits[unsafe_offset=nid].colid,
            splits[unsafe_offset=nid].best_metric_val,
            splits[unsafe_offset=nid].n_left,
        ),
        min_impurity_decrease,
        min_samples_leaf,
        Int32(range_len),
    )
    if sabotage == PART_MB_SAB_NO_VALID_GUARD:
        return False
    return invalid


def partition_count_kernel[
    TPB: Int
](
    blk_left: MutPointer[Int32, MutAnyOrigin],
    row_ids: MutPointer[Int32, MutAnyOrigin],
    data: MutPointer[Float32, MutAnyOrigin],
    work_items: MutPointer[NodeWorkItem, MutAnyOrigin],
    workload: MutPointer[WorkloadInfo, MutAnyOrigin],
    splits: MutPointer[Split, MutAnyOrigin],
    m_in: Int32,
    min_impurity_decrease: Float32,
    min_samples_leaf: Int32,
    sabotage_in: Int32,
):
    """Pass 1: how many of block `b`'s rows go left.

    GRID `(n_blocks_dimx, 1, 1)`, BLOCK `(TPB, 1, 1)`. Block `b` serves
    `workload[b]`, and covers rows `[offset_blockid * TPB, +TPB)` of that
    node -- the same tiling `build_workload_info` counted with
    `n_blocks_per_node = ceildiv(count, tpb)`.
    """
    var b = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var nid = Int(workload[unsafe_offset=b].nodeid)
    var ob = Int(workload[unsafe_offset=b].offset_blockid)
    if sabotage_in == PART_MB_SAB_SINGLE_BLOCK:
        ob = 0

    if _skip_node(
        work_items,
        splits,
        nid,
        min_impurity_decrease,
        min_samples_leaf,
        sabotage_in,
    ):
        if tid == 0:
            blk_left[unsafe_offset=b] = Int32(0)
        return

    var range_start = Int(work_items[unsafe_offset=nid].instances.begin)
    var range_len = Int(work_items[unsafe_offset=nid].instances.count)
    var col_offset = Int(splits[unsafe_offset=nid].colid) * Int(m_in)
    var quesval = splits[unsafe_offset=nid].quesval

    var row_index = ob * TPB + tid
    var flag = Int32(0)
    if row_index < range_len:
        flag = _goes_left(
            data, row_ids, range_start + row_index, col_offset, quesval
        )

    var scanned = _block_scan[TPB](flag)
    if tid == 0:
        blk_left[unsafe_offset=b] = scanned[1]


def partition_scan_kernel[
    TPB: Int
](
    blk_off: MutPointer[Int32, MutAnyOrigin],
    blk_left: MutPointer[Int32, MutAnyOrigin],
    blk_base: MutPointer[Int32, MutAnyOrigin],
    work_items: MutPointer[NodeWorkItem, MutAnyOrigin],
    splits: MutPointer[Split, MutAnyOrigin],
    min_impurity_decrease: Float32,
    min_samples_leaf: Int32,
    sabotage_in: Int32,
):
    """Pass 2: exclusive scan of `blk_left` WITHIN each node.

    GRID `(n_nodes, 1, 1)`. This is the one pass that is still one block per
    node, and that is the point: it scans `ceil(count/TPB)` block counts, not
    `count` rows. `blk_base[n]` is where node `n`'s blocks start in the
    flattened workload array, which the host knows because
    `build_workload_info` lays them out contiguously in node order.
    """
    var n = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    if _skip_node(
        work_items,
        splits,
        n,
        min_impurity_decrease,
        min_samples_leaf,
        sabotage_in,
    ):
        return

    var base = Int(blk_base[unsafe_offset=n])
    var count = Int(work_items[unsafe_offset=n].instances.count)
    var nb = (count + TPB - 1) // TPB
    if nb < 1:
        nb = 1

    var running = Int32(0)
    var chunk = 0
    while chunk < nb:
        var i = chunk + tid
        var v = Int32(0)
        if i < nb:
            v = blk_left[unsafe_offset = base + i]
        var scanned = _block_scan[TPB](v)
        if i < nb:
            blk_off[unsafe_offset = base + i] = running + scanned[0]
        running += scanned[1]
        chunk += TPB


def partition_scatter_kernel[
    TPB: Int
](
    row_ids_out: MutPointer[Int32, MutAnyOrigin],
    row_ids: MutPointer[Int32, MutAnyOrigin],
    blk_off: MutPointer[Int32, MutAnyOrigin],
    data: MutPointer[Float32, MutAnyOrigin],
    work_items: MutPointer[NodeWorkItem, MutAnyOrigin],
    workload: MutPointer[WorkloadInfo, MutAnyOrigin],
    splits: MutPointer[Split, MutAnyOrigin],
    m_in: Int32,
    min_impurity_decrease: Float32,
    min_samples_leaf: Int32,
    sabotage_in: Int32,
):
    """Pass 3: write every row to its final slot.

    A left-going row lands at `range_start + (left rows before it)`; a
    right-going row at `range_start + n_left + (right rows before it)`. "Before
    it" is the cross-block prefix from `blk_off` plus the within-block prefix
    from the same scan the count pass ran. Rows before this block within the
    node number exactly `offset_blockid * TPB`, because the tiling is dense.
    """
    var b = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var nid = Int(workload[unsafe_offset=b].nodeid)
    var ob = Int(workload[unsafe_offset=b].offset_blockid)
    if sabotage_in == PART_MB_SAB_SINGLE_BLOCK:
        ob = 0

    if _skip_node(
        work_items,
        splits,
        nid,
        min_impurity_decrease,
        min_samples_leaf,
        sabotage_in,
    ):
        return

    var range_start = Int(work_items[unsafe_offset=nid].instances.begin)
    var range_len = Int(work_items[unsafe_offset=nid].instances.count)
    var col_offset = Int(splits[unsafe_offset=nid].colid) * Int(m_in)
    var quesval = splits[unsafe_offset=nid].quesval
    var n_left = Int(splits[unsafe_offset=nid].n_left)

    var row_index = ob * TPB + tid
    var flag = Int32(0)
    if row_index < range_len:
        flag = _goes_left(
            data, row_ids, range_start + row_index, col_offset, quesval
        )

    var scanned = _block_scan[TPB](flag)

    var left_before = Int(blk_off[unsafe_offset=b])
    if sabotage_in == PART_MB_SAB_NO_SCAN:
        left_before = 0
    var rows_before = ob * TPB
    var right_before = rows_before - left_before

    if row_index < range_len:
        var row = row_ids[unsafe_offset = range_start + row_index]
        if flag != 0:
            row_ids_out[
                unsafe_offset = range_start + left_before + Int(scanned[0])
            ] = row
        else:
            var within = tid - Int(scanned[0])
            row_ids_out[
                unsafe_offset = range_start + n_left + right_before + within
            ] = row


def partition_writeback_kernel[
    TPB: Int
](
    row_ids: MutPointer[Int32, MutAnyOrigin],
    row_ids_out: MutPointer[Int32, MutAnyOrigin],
    work_items: MutPointer[NodeWorkItem, MutAnyOrigin],
    workload: MutPointer[WorkloadInfo, MutAnyOrigin],
    splits: MutPointer[Split, MutAnyOrigin],
    min_impurity_decrease: Float32,
    min_samples_leaf: Int32,
    sabotage_in: Int32,
):
    """Copy the partitioned ranges back, leaving every other row alone.

    Only the frontier's ranges are touched, which is what makes `row_ids`
    resident: a node NOT in this batch, and a node whose split was refused,
    keeps the slots it already had.
    """
    var b = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var nid = Int(workload[unsafe_offset=b].nodeid)
    var ob = Int(workload[unsafe_offset=b].offset_blockid)
    if _skip_node(
        work_items,
        splits,
        nid,
        min_impurity_decrease,
        min_samples_leaf,
        sabotage_in,
    ):
        return
    var range_start = Int(work_items[unsafe_offset=nid].instances.begin)
    var range_len = Int(work_items[unsafe_offset=nid].instances.count)
    var row_index = ob * TPB + tid
    if row_index < range_len:
        row_ids[unsafe_offset = range_start + row_index] = row_ids_out[
            unsafe_offset = range_start + row_index
        ]
