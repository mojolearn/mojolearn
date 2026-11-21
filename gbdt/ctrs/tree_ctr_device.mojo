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
from gbdt.ctrs.tree_ctr import tree_ctr_check_configs
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
