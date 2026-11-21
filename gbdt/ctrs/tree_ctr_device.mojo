# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Tree CTR (feature combination) columns on the device.

The statement of what is computed is `gbdt/ctrs/tree_ctr.mojo`. Here the
columns come from the launches the simple CTRs already use:

    TCtrBinBuilderGpu            rows in tensor order, segment starts marked;
                                 one `add_cat_feature_bins_device` per tensor
                                 member (a split's 0/1 side is a member of
                                 cardinality 2), each a stable radix sort
    THistoryBasedCtrCalcerGpu    the ordered Borders columns
    TWeightedBinFreqCalcerGpu    the FeatureFreq columns

`tree_ctr_base_builder` orders the rows by the tree's tensor once per level
and permutation; every candidate copies that ordering device to device and
adds its one categorical feature (the reference's `SetIndices(base)` +
`AddCompressedBins`), so a level costs one sort per candidate and not one
per member per candidate.

The FeatureFreq columns' Median grid (`tree_ctr.mojo`, THE MEDIAN GRID)
takes its order statistics here: `launch_tree_median_words` sorts one
column exactly (`core/sort_ops.mojo`'s stable radix sort of the values'
bits), gathers the sorted copy and writes the column's `2 + 2 * borders`
words into a level-wide record buffer the caller reads home ONCE per level
for all candidates.

Parallel: every launch is one thread per row (or the calcers' own grids).
The candidates of a level are enqueued one after another on the fit's one
queue. Nothing here reads a row on the host; the calcers drain once per
candidate, as they do for a simple CTR.

Bits: integer counts held in Float32 and one Float32 divide per value, the
arithmetic `tree_ctr.mojo` states for the host column.
"""

from max.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from gbdt.ctrs.ctr import (
    CTR_BORDERS,
    CTR_FEATURE_FREQ,
    TCtrConfig,
    is_equal_up_to_prior_and_binarization,
)
from gbdt.ctrs.ctr_bins_builder import TCtrBinBuilderGpu
from gbdt.ctrs.ctr_calcers import (
    THistoryBasedCtrCalcerGpu,
    TWeightedBinFreqCalcerGpu,
)
from core.sort_ops import (
    SortScratch,
    sort_op_f32,
    sort_op_gather_rows_f32,
)
from core.step_gate import step_gate_open_word
from gbdt.ctrs.tree_ctr import (
    TreeF32Any,
    tree_ctr_check_configs,
    tree_ctr_median_pair_into,
    tree_ctr_median_words,
)
from gbdt.gpu_data.gpu_structures import CFeature

comptime TREE_CTR_BLOCK = 256


def _tree_ctr_blocks(n: Int) -> Int:
    return min(max((n + TREE_CTR_BLOCK - 1) // TREE_CTR_BLOCK, 1), 65535)


def tree_split_bits_kernel(
    cindex: MutPointer[UInt32, MutAnyOrigin],
    base_in: Int32,
    mask: UInt32,
    shift: UInt32,
    bin: UInt32,
    take_bin: Int32,
    dst: MutPointer[UInt32, MutAnyOrigin],
    n_in: Int32,
):
    """Per row, the 0/1 side of one binary split read off the compressed
    index: `bin == split bin` for a one-hot split, `bin > split bin`
    otherwise."""
    var n = Int(n_in)
    var base = Int(base_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    while i < n:
        var value = (cindex.unsafe_load(base + i) >> shift) & mask
        var side = UInt32(0)
        if take_bin != Int32(0):
            if value == bin:
                side = UInt32(1)
        else:
            if value > bin:
                side = UInt32(1)
        dst.unsafe_store(i, side)
        i += stride


def launch_tree_split_bits(
    ctx: DeviceContext,
    mut cindex: DeviceBuffer[DType.uint32],
    feature: CFeature,
    n_rows: Int,
    bin: Int,
    take_bin: Bool,
    mut dst: DeviceBuffer[DType.uint32],
) raises:
    """`dst[0:n_rows]` = the side of every row under the split (`feature`
    is the split column's layout entry in `cindex`)."""
    if n_rows <= 0:
        return
    ctx.enqueue_function[tree_split_bits_kernel](
        cindex.unsafe_ptr(),
        Int32(Int(feature.offset) * n_rows),
        feature.mask,
        feature.shift,
        UInt32(bin),
        Int32(1) if take_bin else Int32(0),
        dst.unsafe_ptr(),
        Int32(n_rows),
        grid_dim=(_tree_ctr_blocks(n_rows), 1, 1),
        block_dim=(TREE_CTR_BLOCK, 1, 1),
    )


def tree_slot_write_kernel(
    cindex: MutPointer[UInt32, MutAnyOrigin],
    base_in: Int32,
    mask: UInt32,
    shift: UInt32,
    bins: MutPointer[UInt32, MutAnyOrigin],
    n_in: Int32,
):
    """Per row, replace one feature's bits of its compressed-index word
    with `bins[row]`. One thread owns one word, so the read-modify-write
    needs no atomic; the neighbouring features of the word are kept."""
    var n = Int(n_in)
    var base = Int(base_in)
    var field = mask << shift
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    while i < n:
        var word = cindex.unsafe_load(base + i) & ~field
        cindex.unsafe_store(
            base + i, word | ((bins.unsafe_load(i) & mask) << shift)
        )
        i += stride


def launch_tree_slot_write(
    ctx: DeviceContext,
    mut cindex: DeviceBuffer[DType.uint32],
    feature: CFeature,
    n_rows: Int,
    mut bins: DeviceBuffer[DType.uint32],
) raises:
    """Install `bins[0:n_rows]` as the column of layout entry `feature`
    (a tree-CTR slot of a pinned layout) in `cindex`."""
    if n_rows <= 0:
        return
    ctx.enqueue_function[tree_slot_write_kernel](
        cindex.unsafe_ptr(),
        Int32(Int(feature.offset) * n_rows),
        feature.mask,
        feature.shift,
        bins.unsafe_ptr(),
        Int32(n_rows),
        grid_dim=(_tree_ctr_blocks(n_rows), 1, 1),
        block_dim=(TREE_CTR_BLOCK, 1, 1),
    )


def tree_ctr_base_builder(
    ctx: DeviceContext,
    order: DeviceBuffer[DType.uint32],
    n_rows: Int,
    member_codes: List[DeviceBuffer[DType.uint32]],
    member_cards: List[Int],
) raises -> TCtrBinBuilderGpu:
    """The rows of one permutation ordered by the tree's tensor: `order` is
    the permutation's CTR estimation order (resident, row ids only), and
    each member is `n_rows` codes below its cardinality (a categorical
    feature's dense codes, or a split's sides from `launch_tree_split_bits`
    with cardinality 2), in the tensor's canonical member order."""
    if len(member_codes) != len(member_cards):
        raise Error("tree CTR base: one cardinality per member")
    var builder = TCtrBinBuilderGpu(ctx, device_order=order, n_rows=n_rows)
    # driver-loop: over the tensor's members
    for m in range(len(member_codes)):
        builder.add_cat_feature_bins_device(
            ctx, member_codes[m], member_cards[m]
        )
    return builder^


def tree_ctr_candidate_columns_device(
    ctx: DeviceContext,
    base: TCtrBinBuilderGpu,
    codes: DeviceBuffer[DType.uint32],
    cardinality: Int,
    n_rows: Int,
    configs: List[TCtrConfig],
    binarized_target: DeviceBuffer[DType.uint8],
) raises -> List[DeviceBuffer[DType.float32]]:
    """One candidate's columns, one device buffer per config in config
    order, indexed by original row: the base ordering extended by the
    candidate's categorical feature, then one calcer pass per group of
    configs equal up to the prior. `binarized_target` is read only when a
    Borders config is present."""
    tree_ctr_check_configs(configs)
    # the base's indices carry its segment starts; the copy adopts them
    var builder = TCtrBinBuilderGpu(
        ctx, device_order=base.indices, n_rows=n_rows
    )
    builder.add_cat_feature_bins_device(ctx, codes, cardinality)

    var bufs = List[DeviceBuffer[DType.float32]]()
    var slot_buf = List[Int](length=len(configs), fill=-1)
    var done = List[Bool](length=len(configs), fill=False)
    # driver-loop: over the CTR configs (grouped equal up to the prior)
    for i in range(len(configs)):
        if done[i]:
            continue
        var group = List[TCtrConfig]()
        var group_slots = List[Int]()
        # driver-loop: over the CTR configs
        for j in range(i, len(configs)):
            if not done[j] and is_equal_up_to_prior_and_binarization(
                configs[j], configs[i]
            ):
                done[j] = True
                group.append(configs[j])
                group_slots.append(j)
        var columns: List[DeviceBuffer[DType.float32]]
        if group[0].ctr_type == CTR_FEATURE_FREQ:
            var freq = TWeightedBinFreqCalcerGpu.trivial(ctx, n_rows)
            # a tensor has at most one segment per row
            columns = freq.visit_equal_up_to_prior_freq_ctrs(
                ctx, builder, group, n_rows
            )
            _ = freq^
        else:
            var calcer = THistoryBasedCtrCalcerGpu(ctx, builder)
            calcer.set_binarized_sample_device(ctx, binarized_target)
            columns = calcer.visit_cat_feature_ctr(ctx, group)
            # the calcer's scratch is read by the enqueued launches
            ctx.synchronize()
            _ = calcer^
        # driver-loop: over the configs of the group
        for k in range(len(group_slots)):
            slot_buf[group_slots[k]] = len(bufs)
            bufs.append(columns[k])

    var out = List[DeviceBuffer[DType.float32]]()
    # driver-loop: over the CTR configs
    for c in range(len(configs)):
        out.append(bufs[slot_buf[c]])
    ctx.synchronize()
    _ = builder^
    return out^


def tree_median_gather_kernel(
    sorted: TreeF32Any,
    n_in: Int32,
    border_count_in: Int32,
    words: TreeF32Any,
    base_in: Int32,
):
    """One column's Median record from its ascending copy: the minimum and
    maximum (thread 0), and one thread per border for its pair
    (`tree_ctr_median_pair_into`, the statement the host checks run)."""
    var n = Int(n_in)
    var border_count = Int(border_count_in)
    var base = Int(base_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    if i == 0:
        words.unsafe_store(base, sorted.unsafe_load(0))
        words.unsafe_store(base + 1, sorted.unsafe_load(n - 1))
    while i < border_count:
        tree_ctr_median_pair_into(sorted, n, border_count, i, words, base)
        i += stride


struct TreeMedianScratch(Movable):
    """The Median gather's device scratch for columns of up to `capacity`
    rows: the sort's order and scratch, the sorted copy and the open gate
    the sort ops take outside a loop. Allocate once per fit."""

    var capacity: Int
    var gate: DeviceBuffer[DType.int32]
    var order: DeviceBuffer[DType.int32]
    var sorted: DeviceBuffer[DType.float32]
    var sort: SortScratch

    def __init__(out self, ctx: DeviceContext, capacity: Int) raises:
        var cap = max(capacity, 1)
        self.capacity = cap
        self.gate = step_gate_open_word(ctx)
        self.order = ctx.enqueue_create_buffer[DType.int32](cap)
        self.sorted = ctx.enqueue_create_buffer[DType.float32](cap)
        self.sort = SortScratch(ctx, cap)


def launch_tree_median_words(
    ctx: DeviceContext,
    mut column: DeviceBuffer[DType.float32],
    n_rows: Int,
    border_count: Int,
    mut scratch: TreeMedianScratch,
    mut words: DeviceBuffer[DType.float32],
    base: Int,
) raises:
    """ENQUEUED: the Median record of `column[0:n_rows]` (a FeatureFreq
    tree-CTR column: finite values, no NaN) into `words[base : base +
    tree_ctr_median_words(border_count)]`. The caller reads `words` home
    once for every candidate of the level and hands each record to
    `tree_ctr_median_from_words`. `column` is read only. The scratch is
    reused by the next call on the same queue, in queue order."""
    if n_rows <= 0:
        raise Error("tree CTR Median grid: no rows")
    if border_count < 1:
        raise Error("tree CTR Median grid: no borders asked")
    if n_rows > scratch.capacity or len(column) < n_rows:
        raise Error("tree CTR Median grid: the column exceeds the scratch")
    if base < 0 or base + tree_ctr_median_words(border_count) > len(words):
        raise Error("tree CTR Median grid: the words buffer is too small")
    sort_op_f32(
        ctx, scratch.gate, column, scratch.order, n_rows, False, scratch.sort
    )
    sort_op_gather_rows_f32(
        ctx, scratch.gate, column, scratch.order, scratch.sorted, n_rows, 1
    )
    ctx.enqueue_function[tree_median_gather_kernel](
        scratch.sorted.unsafe_ptr(),
        Int32(n_rows),
        Int32(border_count),
        words.unsafe_ptr(),
        Int32(base),
        grid_dim=(1, 1, 1),
        block_dim=(TREE_CTR_BLOCK, 1, 1),
    )


# ===========================================================================
# THE FIT ARM'S SLOT LAUNCHES (`gbdt/methods/tree_ctr_search.mojo`)
# ===========================================================================


def tree_ctr_bins_kernel(
    values: TreeF32Any,
    n_in: Int32,
    borders: TreeF32Any,
    border_base_in: Int32,
    border_count_in: Int32,
    dst: MutPointer[UInt32, MutAnyOrigin],
):
    """Per row, the number of borders the value strictly exceeds
    (`tree_ctr.tree_ctr_bins_into`, the host column's statement): the
    column's bin under `borders[border_base : border_base + border_count]`,
    ascending. Comparisons only."""
    var n = Int(n_in)
    var base = Int(border_base_in)
    var count = Int(border_count_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    while i < n:
        var v = values.unsafe_load(i)
        var bin = UInt32(0)
        var b = 0
        while b < count:
            if v > borders.unsafe_load(base + b):
                bin += UInt32(1)
            b += 1
        dst.unsafe_store(i, bin)
        i += stride


def launch_tree_ctr_bins(
    ctx: DeviceContext,
    mut values: DeviceBuffer[DType.float32],
    n_rows: Int,
    mut borders: DeviceBuffer[DType.float32],
    border_base: Int,
    border_count: Int,
    mut dst: DeviceBuffer[DType.uint32],
) raises:
    """ENQUEUED: `dst[0:n_rows]` = the bins of `values[0:n_rows]` under one
    grid of the level's borders buffer."""
    if n_rows <= 0:
        return
    if border_base < 0 or border_count < 0 or (
        border_base + border_count > len(borders)
    ):
        raise Error("tree CTR bins: the grid is outside the borders buffer")
    if len(values) < n_rows or len(dst) < n_rows:
        raise Error("tree CTR bins: a column is shorter than the row count")
    ctx.enqueue_function[tree_ctr_bins_kernel](
        values.unsafe_ptr(),
        Int32(n_rows),
        borders.unsafe_ptr(),
        Int32(border_base),
        Int32(border_count),
        dst.unsafe_ptr(),
        grid_dim=(_tree_ctr_blocks(n_rows), 1, 1),
        block_dim=(TREE_CTR_BLOCK, 1, 1),
    )


def tree_slot_copy_kernel(
    src: MutPointer[UInt32, MutAnyOrigin],
    src_base_in: Int32,
    src_mask: UInt32,
    src_shift: UInt32,
    dst: MutPointer[UInt32, MutAnyOrigin],
    dst_base_in: Int32,
    dst_mask: UInt32,
    dst_shift: UInt32,
    n_in: Int32,
):
    """Per row, copy one feature's bin from a compressed index into another
    feature's bits (of the same index or of another one). One thread owns
    one row: when both features share a word, the thread reads it before it
    writes it."""
    var n = Int(n_in)
    var src_base = Int(src_base_in)
    var dst_base = Int(dst_base_in)
    var field = dst_mask << dst_shift
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    while i < n:
        var bin = (src.unsafe_load(src_base + i) >> src_shift) & src_mask
        var word = dst.unsafe_load(dst_base + i) & ~field
        dst.unsafe_store(dst_base + i, word | ((bin & dst_mask) << dst_shift))
        i += stride


def launch_tree_slot_copy(
    ctx: DeviceContext,
    mut src: DeviceBuffer[DType.uint32],
    src_feature: CFeature,
    mut dst: DeviceBuffer[DType.uint32],
    dst_feature: CFeature,
    n_rows: Int,
) raises:
    """ENQUEUED: the column of layout entry `src_feature` in `src` becomes
    the column of `dst_feature` in `dst` (a won candidate slot into the
    tree's next persist slot, on the search index or on another
    permutation's)."""
    if n_rows <= 0:
        return
    ctx.enqueue_function[tree_slot_copy_kernel](
        src.unsafe_ptr(),
        Int32(Int(src_feature.offset) * n_rows),
        src_feature.mask,
        src_feature.shift,
        dst.unsafe_ptr(),
        Int32(Int(dst_feature.offset) * n_rows),
        dst_feature.mask,
        dst_feature.shift,
        Int32(n_rows),
        grid_dim=(_tree_ctr_blocks(n_rows), 1, 1),
        block_dim=(TREE_CTR_BLOCK, 1, 1),
    )


def tree_slot_clear_kernel(
    cindex: MutPointer[UInt32, MutAnyOrigin],
    base_in: Int32,
    mask: UInt32,
    shift: UInt32,
    n_in: Int32,
):
    """Per row, zero one feature's bits of its compressed-index word (bin 0
    for every row: a slot no candidate holds offers no split)."""
    var n = Int(n_in)
    var base = Int(base_in)
    var field = mask << shift
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    while i < n:
        cindex.unsafe_store(base + i, cindex.unsafe_load(base + i) & ~field)
        i += stride


def launch_tree_slot_clear(
    ctx: DeviceContext,
    mut cindex: DeviceBuffer[DType.uint32],
    feature: CFeature,
    n_rows: Int,
) raises:
    """ENQUEUED: every row of layout entry `feature` in `cindex` to bin 0."""
    if n_rows <= 0:
        return
    ctx.enqueue_function[tree_slot_clear_kernel](
        cindex.unsafe_ptr(),
        Int32(Int(feature.offset) * n_rows),
        feature.mask,
        feature.shift,
        Int32(n_rows),
        grid_dim=(_tree_ctr_blocks(n_rows), 1, 1),
        block_dim=(TREE_CTR_BLOCK, 1, 1),
    )


def tree_group_count_kernel(
    scanned: MutPointer[UInt32, MutAnyOrigin],
    n_in: Int32,
    dst: MutPointer[UInt32, MutAnyOrigin],
):
    """One thread: the number of segments of a bin builder whose
    `compute_current_bins` just ran. `scanned` is the builder's `bins`, which
    that call leaves holding the exclusive scan of the segment-end flags in
    sorted order, so its last word is the last segment's id."""
    if Int(block_idx.x) == 0 and Int(thread_idx.x) == 0:
        dst.unsafe_store(0, scanned.unsafe_load(Int(n_in) - 1) + UInt32(1))


def launch_tree_group_count(
    ctx: DeviceContext,
    mut builder: TCtrBinBuilderGpu,
    mut dst: DeviceBuffer[DType.uint32],
) raises:
    """ENQUEUED: `builder.compute_current_bins` (after which
    `builder.current_bins` is every original row's group id under the
    members added so far) and the group count into `dst[0]`."""
    builder.compute_current_bins(ctx)
    ctx.enqueue_function[tree_group_count_kernel](  # small-launch(1: one word): the group count of one combination
        builder.bins.unsafe_ptr(),
        Int32(builder.size),
        dst.unsafe_ptr(),
        grid_dim=(1, 1, 1),
        block_dim=(1, 1, 1),
    )
