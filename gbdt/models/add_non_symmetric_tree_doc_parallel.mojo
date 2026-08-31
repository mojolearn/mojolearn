# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# This lane MIRRORS CatBoost (pinned commit 54a8143a). Per-file provenance is in gbdt/DERIVATION_MAP.tsv, in this file's own docstring, and in NOTICE; files with no CatBoost counterpart say so.
"""Apply a NON-SYMMETRIC tree to rows: compute each row's bin, add its leaf.

PORT OF `catboost/cuda/models/add_non_symmetric_tree_doc_parallel.{h,cpp}`
at CatBoost `54a8143a`. Transliterated. Do not improve.

Two of their three entry points are here, collapsed to one device and one
task each, which is what the caller in `doc_parallel_boosting` has:

* `ComputeBinsForModel(const TNonSymmetricTreeStructure&, dataSet, bins)`
  (`:208-216`) -- ONE task of `TComputeNonSymmetricTreeLeavesDocParallel`,
  which packs one `TCFeature` per INTERNAL NODE beside the node array
  (`:88-92`) and launches `ComputeNonSymmetricDecisionTreeBins`. This is
  `compute_non_symmetric_bins_for_model` below, and it is what the leaves
  estimator calls through `task.Model->ComputeBins(dataSet, &bins)`
  (`doc_parallel_leaves_estimator.cpp:48`) before building its oracle.
* `TAddModelDocParallel<TNonSymmetricTree>::Proceed` (`:182-206`) -- bins
  first, then `AddBinModelValues(taskValues, TempBins, cursor)`. This is
  `add_non_symmetric_tree_to_cursor` below, the apply `predict` and the
  held-out arm use. `AddBinModelValues` is `add_bin_model_value_kernel`
  (`AddBinModelValueImpl`, `add_model_value.cu:14-53`), already ported for
  the estimator's `MoveTo`.

The third, the streamed multi-task `AddTask`/`Proceed` pairing over several
cursors, has no counterpart because nothing here applies one tree to
several datasets in one launch batch; each caller applies one tree to one
cursor.

THE PER-NODE FEATURE PLANES are the DEVIATION BLOCK of
`compute_non_symmetric_decision_tree_bins_kernel`: theirs walks a
`TCFeature*` array by pointer, ours seven parallel planes by index, for the
Metal reason recorded there. The packing here is the same packing
`original/depthwise_check.apply_bins` did inline before this file existed;
that check keeps its own copy, because a gate that imports the thing it
gates cannot catch the thing drifting, and this file's caller is the
boosting loop that check does not run. DEVIATION 259.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from gbdt.gpu_data.compressed_index_builder import CompressedIndexLayout
from gbdt.methods.kernel_add_model_value import (
    ABMV_BLOCK,
    ABMV_ELEMENTS,
    add_bin_model_value_kernel,
)
from gbdt.models.kernel.add_bin_values import (
    compute_non_symmetric_decision_tree_bins_kernel,
)
from gbdt.models.non_symmetric_tree import (
    TNonSymmetricTree,
    TNonSymmetricTreeStructure,
)
from gbdt.models.oblivious_model import BIN_SPLIT_TAKE_BIN

#: their `ComputeNonSymmetricDecisionTreeBins` launch shape
#: (`add_model_value.cu:399-412`): 256 threads, `CeilDivide(size, 256)`
#: blocks, strided.
comptime NS_BINS_BLOCK_SIZE = 256


def compute_non_symmetric_bins_for_model(
    ctx: DeviceContext,
    layout: CompressedIndexLayout,
    structure: TNonSymmetricTreeStructure,
    mut cindex: DeviceBuffer[DType.uint32],
    n_rows: Int,
    mut out_bins: DeviceBuffer[DType.uint32],
) raises:
    """`ComputeBinsForModel` for a `TNonSymmetricTreeStructure`
    (`add_non_symmetric_tree_doc_parallel.cpp:208-216`).

    One `TCFeature` per internal node, in node order
    (`FeaturesBuilder.Add(dataSet.GetTCFeature(node.FeatureId))`, `:88-92`),
    plus the node's bin, subtree sizes and its predicate. The predicate
    comes off the MODEL's `split_types`, not the layout, for the reason
    `predict` records: a model read back from a file is applied against a
    layout rebuilt from its own fold counts, and the file has to be the
    authority. The layout is cross-checked, which is their
    `CB_ENSURE(dataSet.IsOneHot(split.FeatureId))` pair
    (`add_oblivious_tree_model_doc_parallel.cpp:43-47`) on this shape.

    A CONSTANT TREE (no nodes) is their `nodes == nullptr`: every row lands
    in bin 0. One slot is still staged so no buffer is zero-length.
    """
    var n_nodes = len(structure.nodes)
    if len(structure.split_types) != n_nodes:
        raise Error(
            "compute_non_symmetric_bins_for_model: " + String(n_nodes)
            + " nodes and " + String(len(structure.split_types))
            + " split types"
        )
    var slots = n_nodes if n_nodes > 0 else 1

    var h_off = ctx.enqueue_create_host_buffer[DType.uint32](slots)
    var h_mask = ctx.enqueue_create_host_buffer[DType.uint32](slots)
    var h_shift = ctx.enqueue_create_host_buffer[DType.uint32](slots)
    var h_oh = ctx.enqueue_create_host_buffer[DType.uint8](slots)
    var h_bin = ctx.enqueue_create_host_buffer[DType.uint32](slots)
    var h_ls = ctx.enqueue_create_host_buffer[DType.uint32](slots)
    var h_rs = ctx.enqueue_create_host_buffer[DType.uint32](slots)
    for i in range(slots):
        h_off.unsafe_ptr().unsafe_store(i, UInt32(0))
        h_mask.unsafe_ptr().unsafe_store(i, UInt32(0))
        h_shift.unsafe_ptr().unsafe_store(i, UInt32(0))
        h_oh.unsafe_ptr().unsafe_store(i, UInt8(0))
        h_bin.unsafe_ptr().unsafe_store(i, UInt32(0))
        h_ls.unsafe_ptr().unsafe_store(i, UInt32(1))
        h_rs.unsafe_ptr().unsafe_store(i, UInt32(1))
    for i in range(n_nodes):
        ref n = structure.nodes[i]
        var fid = Int(n.feature_id)
        if fid < 0 or fid >= len(layout.features):
            raise Error(
                "compute_non_symmetric_bins_for_model: node " + String(i)
                + " splits on feature " + String(fid) + " of "
                + String(len(layout.features))
            )
        ref f = layout.features[fid]
        var take_bin = Int(structure.split_types[i]) == BIN_SPLIT_TAKE_BIN
        if take_bin != f.one_hot_feature:
            raise Error(
                "non-symmetric node " + String(i) + " is a "
                + String("TakeBin" if take_bin else "TakeGreater")
                + " split on feature " + String(fid)
                + ", which the layout says is "
                + String("one-hot" if f.one_hot_feature else "ordered")
            )
        # their `feature.Offset` is a COLUMN index and the kernel adds the
        # row; ours is the column times the row count, which is how this
        # port lays the compressed index out
        h_off.unsafe_ptr().unsafe_store(i, UInt32(Int(f.offset) * n_rows))
        h_mask.unsafe_ptr().unsafe_store(i, f.mask)
        h_shift.unsafe_ptr().unsafe_store(i, f.shift)
        h_oh.unsafe_ptr().unsafe_store(i, UInt8(1) if take_bin else UInt8(0))
        h_bin.unsafe_ptr().unsafe_store(i, UInt32(Int(n.bin)))
        h_ls.unsafe_ptr().unsafe_store(i, UInt32(Int(n.left_subtree)))
        h_rs.unsafe_ptr().unsafe_store(i, UInt32(Int(n.right_subtree)))

    var d_off = ctx.enqueue_create_buffer[DType.uint32](slots)
    var d_mask = ctx.enqueue_create_buffer[DType.uint32](slots)
    var d_shift = ctx.enqueue_create_buffer[DType.uint32](slots)
    var d_oh = ctx.enqueue_create_buffer[DType.uint8](slots)
    var d_bin = ctx.enqueue_create_buffer[DType.uint32](slots)
    var d_ls = ctx.enqueue_create_buffer[DType.uint32](slots)
    var d_rs = ctx.enqueue_create_buffer[DType.uint32](slots)
    ctx.enqueue_copy(dst_buf=d_off, src_ptr=h_off.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_mask, src_ptr=h_mask.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_shift, src_ptr=h_shift.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_oh, src_ptr=h_oh.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_bin, src_ptr=h_bin.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_ls, src_ptr=h_ls.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_rs, src_ptr=h_rs.unsafe_ptr())

    var blocks = (n_rows + NS_BINS_BLOCK_SIZE - 1) // NS_BINS_BLOCK_SIZE
    if blocks < 1:
        blocks = 1
    ctx.enqueue_function[compute_non_symmetric_decision_tree_bins_kernel](
        d_off.unsafe_ptr(), d_mask.unsafe_ptr(), d_shift.unsafe_ptr(),
        d_oh.unsafe_ptr(), d_bin.unsafe_ptr(),
        d_ls.unsafe_ptr(), d_rs.unsafe_ptr(),
        Int32(n_nodes),
        cindex.unsafe_ptr(),
        Int32(n_rows),
        out_bins.unsafe_ptr(),
        grid_dim=(blocks, 1, 1),
        block_dim=(NS_BINS_BLOCK_SIZE, 1, 1),
    )
    ctx.synchronize()
    # [[mojo-buffer-freed-at-last-use]]: every plane's last textual use is
    # the launch above; the keep-alives sit past the drain
    _ = d_off^
    _ = d_mask^
    _ = d_shift^
    _ = d_oh^
    _ = d_bin^
    _ = d_ls^
    _ = d_rs^
    _ = h_off^
    _ = h_mask^
    _ = h_shift^
    _ = h_oh^
    _ = h_bin^
    _ = h_ls^
    _ = h_rs^


def add_non_symmetric_tree_to_cursor(
    ctx: DeviceContext,
    layout: CompressedIndexLayout,
    tree: TNonSymmetricTree,
    mut cindex: DeviceBuffer[DType.uint32],
    n_rows: Int,
    mut cursor: DeviceBuffer[DType.float32],
) raises:
    """`TAddModelDocParallel<TNonSymmetricTree>::Proceed`
    (`add_non_symmetric_tree_doc_parallel.cpp:182-206`), one task: compute
    the bins, broadcast the leaf values, `AddBinModelValues`.

    THE LEAF VALUES ARE ADDED AS STORED. `fit_with_test` folds their
    `Rescale(step)` into the stored values exactly as it does for the
    oblivious shape, so this adds them at 1.0 and must not reapply the
    rate. `cursor` is PLANE-MAJOR `[dim * n_rows + row]`, and `leaf_values`
    is BIN-MAJOR `[bin * dim + d]` -- their `LeafValues.data() + bin * Dim`
    (`non_symmetric_tree.h:191`) -- which is the pair
    `add_bin_model_value_kernel` was written against.
    """
    var dim = tree.dim
    if dim < 1:
        raise Error("add_non_symmetric_tree_to_cursor: dim " + String(dim))
    var n_bins = tree.bin_count()
    if len(tree.leaf_values) != n_bins * dim:
        raise Error(
            "add_non_symmetric_tree_to_cursor: " + String(len(tree.leaf_values))
            + " leaf values for " + String(n_bins) + " bins x "
            + String(dim)
        )
    var bins = ctx.enqueue_create_buffer[DType.uint32](n_rows)
    compute_non_symmetric_bins_for_model(
        ctx, layout, tree.model_structure, cindex, n_rows, bins
    )
    var h_vals = ctx.enqueue_create_host_buffer[DType.float32](n_bins * dim)
    for i in range(n_bins * dim):
        h_vals.unsafe_ptr().unsafe_store(i, tree.leaf_values[i])
    var d_vals = ctx.enqueue_create_buffer[DType.float32](n_bins * dim)
    ctx.enqueue_copy(dst_buf=d_vals, src_ptr=h_vals.unsafe_ptr())
    # `CeilDivide(size, blockSize * elementsPerThreads)` (`:62`)
    var per_block = ABMV_BLOCK * ABMV_ELEMENTS
    var blocks = (n_rows + per_block - 1) // per_block
    if blocks < 1:
        blocks = 1
    ctx.enqueue_function[add_bin_model_value_kernel](
        d_vals.unsafe_ptr(),
        bins.unsafe_ptr(),
        Int32(n_rows),
        Int32(dim),
        Int32(n_rows),
        cursor.unsafe_ptr(),
        grid_dim=(blocks, 1, 1),
        block_dim=(ABMV_BLOCK, 1, 1),
    )
    ctx.synchronize()
    _ = d_vals^  # past the drain (step-33 race class, device side)
    _ = bins^  # past the drain (step-33 race class, device side)
    _ = h_vals^  # past the drain (step-33 race class)
