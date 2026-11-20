# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Apply a NON-SYMMETRIC tree to rows: compute each row's bin, add its leaf.

Reference: `catboost/cuda/models/add_non_symmetric_tree_doc_parallel.{h,cpp}`
(CatBoost `54a8143a`).

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
  (`AddBinModelValueImpl`, `add_model_value.cu:14-53`), already implemented for
  the estimator's `MoveTo`.

The third, the streamed multi-task `AddTask`/`Proceed` pairing over several
cursors, has no counterpart because nothing here applies one tree to
several datasets in one launch batch; each caller applies one tree to one
cursor.

THE PER-NODE FEATURE PLANES are the DEVIATION BLOCK of
`compute_non_symmetric_decision_tree_bins_kernel`: theirs walks a
`TCFeature*` array by pointer, ours seven parallel planes by index, for the
Metal reason recorded there. The packing here is the same packing
`checks/depthwise_check.apply_bins` did inline before this file existed;
that check keeps its own copy, because a gate that imports the thing it
gates cannot catch the thing drifting, and this file's caller is the
boosting loop that check does not run. DEVIATION 259.
"""

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
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
    for i in range(n_nodes):  # small-loop(n_nodes: tree nodes): stages each model node's split descriptor, model parameters, no row data
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
        # implementation lays the compressed index out
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


#: lane/fam-gbdt (2026-10-04), IDN_NS_PREDICT_PACKED: IDENTICAL, every
#: vendor, default on. `predict` on a Depthwise / Lossguide model applied
#: one tree at a time through `add_non_symmetric_tree_to_cursor`: per tree
#: sixteen staging and device buffers, a fresh `n_rows` bins buffer and TWO
#: drains. `add_non_symmetric_trees_packed` below packs every tree's node
#: table and leaf values once (eight uploads for the whole ensemble), reuses
#: one bins buffer, enqueues the same two kernels per tree back to back on
#: the stream and drains ONCE, the shape the oblivious arm of `predict`
#: already has. Same kernels, same arguments per tree, same tree order: each
#: row's cursor takes the same float adds in the same order, so no bit moves
#: on any column and the host column is untouched.
# TOMBSTONE: MOJOLEARN_IDN_GBDT_NS_PREDICT_PACKED_OFF (DELETE-OLD-PATH, no A/B recorded for the old path, rule b) deleted 2026-11-15 by lane/expdel2-44-trees; code recoverable at c48d22c288.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_GBDT_NS_PREDICT_PACKED_OFF.patch; record in docs/TOMBSTONES.md.
# TOMBSTONE: MOJOLEARN_IDN_ALL_OFF (DELETE-OLD-PATH, EXP:2438 master old-path switch already deleted, rule e; non-owner fold) deleted 2026-11-15 by lane/expdel2-44-trees; code recoverable at 931e7ca76a.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_ALL_OFF.expdel2-44-trees.patch; record in docs/TOMBSTONES.md.
comptime IDN_NS_PREDICT_PACKED = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL


def add_non_symmetric_trees_packed(
    ctx: DeviceContext,
    layout: CompressedIndexLayout,
    trees: List[TNonSymmetricTree],
    mut cindex: DeviceBuffer[DType.uint32],
    n_rows: Int,
    mut cursor: DeviceBuffer[DType.float32],
) raises:
    """`add_non_symmetric_tree_to_cursor` over every tree of an ensemble, in
    order, with the node tables and leaf values packed once and ONE drain
    (`IDN_NS_PREDICT_PACKED`). Tree `t`'s node records sit at
    `[node_at[t], node_at[t] + max(n_nodes, 1))` of the seven node planes and
    its leaf values at `[val_at[t], val_at[t] + n_bins * dim)`; the kernels
    take the planes at those offsets, so a node's subtree sizes stay relative
    to its own tree exactly as in the per-tree buffers. The refusals are the
    per-tree functions', sentence for sentence."""
    var n_trees = len(trees)
    if n_trees == 0:
        ctx.synchronize()
        return
    var total_slots = 0
    var total_vals = 0
    var node_at = List[Int](capacity=n_trees)
    var val_at = List[Int](capacity=n_trees)
    for t in range(n_trees):  # small-loop(n_trees: trees): sizes each tree's node and leaf slabs, model parameters
        ref tree = trees[t]
        var dim = tree.dim
        if dim < 1:
            raise Error("add_non_symmetric_tree_to_cursor: dim " + String(dim))
        var n_bins = tree.bin_count()
        if len(tree.leaf_values) != n_bins * dim:
            raise Error(
                "add_non_symmetric_tree_to_cursor: "
                + String(len(tree.leaf_values))
                + " leaf values for " + String(n_bins) + " bins x "
                + String(dim)
            )
        var n_nodes = len(tree.model_structure.nodes)
        if len(tree.model_structure.split_types) != n_nodes:
            raise Error(
                "compute_non_symmetric_bins_for_model: " + String(n_nodes)
                + " nodes and "
                + String(len(tree.model_structure.split_types))
                + " split types"
            )
        node_at.append(total_slots)
        val_at.append(total_vals)
        total_slots += n_nodes if n_nodes > 0 else 1
        total_vals += n_bins * dim
    var val_cap = total_vals if total_vals > 0 else 1

    var h_off = ctx.enqueue_create_host_buffer[DType.uint32](total_slots)
    var h_mask = ctx.enqueue_create_host_buffer[DType.uint32](total_slots)
    var h_shift = ctx.enqueue_create_host_buffer[DType.uint32](total_slots)
    var h_oh = ctx.enqueue_create_host_buffer[DType.uint8](total_slots)
    var h_bin = ctx.enqueue_create_host_buffer[DType.uint32](total_slots)
    var h_ls = ctx.enqueue_create_host_buffer[DType.uint32](total_slots)
    var h_rs = ctx.enqueue_create_host_buffer[DType.uint32](total_slots)
    var h_vals = ctx.enqueue_create_host_buffer[DType.float32](val_cap)
    for i in range(total_slots):
        h_off.unsafe_ptr().unsafe_store(i, UInt32(0))
        h_mask.unsafe_ptr().unsafe_store(i, UInt32(0))
        h_shift.unsafe_ptr().unsafe_store(i, UInt32(0))
        h_oh.unsafe_ptr().unsafe_store(i, UInt8(0))
        h_bin.unsafe_ptr().unsafe_store(i, UInt32(0))
        h_ls.unsafe_ptr().unsafe_store(i, UInt32(1))
        h_rs.unsafe_ptr().unsafe_store(i, UInt32(1))
    if total_vals == 0:
        h_vals.unsafe_ptr().unsafe_store(0, Float32(0.0))
    for t in range(n_trees):  # small-loop(n_trees: trees): stages each tree's node descriptors and leaf values for upload, model parameters
        ref tree = trees[t]
        ref structure = tree.model_structure
        var base = node_at[t]
        for i in range(len(structure.nodes)):  # small-loop(structure.nodes: nodes of one tree): stages one tree's node descriptors, model parameters
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
            h_off.unsafe_ptr().unsafe_store(
                base + i, UInt32(Int(f.offset) * n_rows)
            )
            h_mask.unsafe_ptr().unsafe_store(base + i, f.mask)
            h_shift.unsafe_ptr().unsafe_store(base + i, f.shift)
            h_oh.unsafe_ptr().unsafe_store(
                base + i, UInt8(1) if take_bin else UInt8(0)
            )
            h_bin.unsafe_ptr().unsafe_store(base + i, UInt32(Int(n.bin)))
            h_ls.unsafe_ptr().unsafe_store(
                base + i, UInt32(Int(n.left_subtree))
            )
            h_rs.unsafe_ptr().unsafe_store(
                base + i, UInt32(Int(n.right_subtree))
            )
        var vbase = val_at[t]
        for i in range(len(tree.leaf_values)):
            h_vals.unsafe_ptr().unsafe_store(vbase + i, tree.leaf_values[i])

    var d_off = ctx.enqueue_create_buffer[DType.uint32](total_slots)
    var d_mask = ctx.enqueue_create_buffer[DType.uint32](total_slots)
    var d_shift = ctx.enqueue_create_buffer[DType.uint32](total_slots)
    var d_oh = ctx.enqueue_create_buffer[DType.uint8](total_slots)
    var d_bin = ctx.enqueue_create_buffer[DType.uint32](total_slots)
    var d_ls = ctx.enqueue_create_buffer[DType.uint32](total_slots)
    var d_rs = ctx.enqueue_create_buffer[DType.uint32](total_slots)
    var d_vals = ctx.enqueue_create_buffer[DType.float32](val_cap)
    ctx.enqueue_copy(dst_buf=d_off, src_ptr=h_off.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_mask, src_ptr=h_mask.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_shift, src_ptr=h_shift.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_oh, src_ptr=h_oh.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_bin, src_ptr=h_bin.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_ls, src_ptr=h_ls.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_rs, src_ptr=h_rs.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_vals, src_ptr=h_vals.unsafe_ptr())

    # TOMBSTONE: MOJOLEARN_GBDT_NS_PREDICT_BITVEC (DELETE-ARM, opt-in arm, never tested or measured, rule a) deleted 2026-11-15 by lane/expdel2-44-trees; code recoverable at 4cdd684efa.
    # Restore: git apply experiments/removed/MOJOLEARN_GBDT_NS_PREDICT_BITVEC.patch; record in docs/TOMBSTONES.md.

    # TOMBSTONE: MOJOLEARN_TREES_C50_GB_PACKED (DELETE-ARM, opt-in arm, no A/B recorded, rule a; non-owner fold) deleted 2026-11-15 by lane/expdel2-44-trees; code recoverable at 5ffd788147.
    # Restore: git apply experiments/removed/MOJOLEARN_TREES_C50_GB_PACKED.expdel2-44-trees.patch; record in docs/TOMBSTONES.md.

    # ONE bins buffer for the ensemble: the stream runs tree t's add before
    # tree t+1's bins kernel rewrites it
    var bins = ctx.enqueue_create_buffer[DType.uint32](
        n_rows if n_rows > 0 else 1
    )
    var bin_blocks = (n_rows + NS_BINS_BLOCK_SIZE - 1) // NS_BINS_BLOCK_SIZE
    if bin_blocks < 1:
        bin_blocks = 1
    var per_block = ABMV_BLOCK * ABMV_ELEMENTS
    var add_blocks = (n_rows + per_block - 1) // per_block
    if add_blocks < 1:
        add_blocks = 1
    for t in range(n_trees):
        var base = node_at[t]
        var vbase = val_at[t]
        ctx.enqueue_function[compute_non_symmetric_decision_tree_bins_kernel](
            d_off.unsafe_ptr() + base,
            d_mask.unsafe_ptr() + base,
            d_shift.unsafe_ptr() + base,
            d_oh.unsafe_ptr() + base,
            d_bin.unsafe_ptr() + base,
            d_ls.unsafe_ptr() + base,
            d_rs.unsafe_ptr() + base,
            Int32(len(trees[t].model_structure.nodes)),
            cindex.unsafe_ptr(),
            Int32(n_rows),
            bins.unsafe_ptr(),
            grid_dim=(bin_blocks, 1, 1),
            block_dim=(NS_BINS_BLOCK_SIZE, 1, 1),
        )
        ctx.enqueue_function[add_bin_model_value_kernel](
            d_vals.unsafe_ptr() + vbase,
            bins.unsafe_ptr(),
            Int32(n_rows),
            Int32(trees[t].dim),
            Int32(n_rows),
            cursor.unsafe_ptr(),
            grid_dim=(add_blocks, 1, 1),
            block_dim=(ABMV_BLOCK, 1, 1),
        )
    ctx.synchronize()
    # past the drain (step-33 race class): every buffer's last use above was
    # an enqueue
    _ = bins^
    _ = d_vals^
    _ = d_rs^
    _ = d_ls^
    _ = d_bin^
    _ = d_oh^
    _ = d_shift^
    _ = d_mask^
    _ = d_off^
    _ = h_vals^
    _ = h_rs^
    _ = h_ls^
    _ = h_bin^
    _ = h_oh^
    _ = h_shift^
    _ = h_mask^
    _ = h_off^

#: lane fix-gbdt-prebinned-infer (2026-11-13), IDENTICAL, every vendor,
#: default on: the resident model (`gbdt/resident_model.mojo`) packs a
#: Depthwise / Lossguide ensemble ONCE, at its first predict, and keeps the
#: packed node planes, leaf values and the bins buffer on the device.
#: `add_non_symmetric_trees_packed` above, which every resident predict
#: called through `predict`, re-ran per call: the layout build, a host pass
#: over every node of every tree (seven planes plus the leaf values),
#: sixteen staging and device allocations and eight uploads, and an
#: `n_rows` bins buffer. Now a call writes the node offsets (`offset *
#: n_rows`) and the bins buffer only when the row count changes, and
#: enqueues the SAME two kernels per tree with the SAME arguments in the
#: SAME tree order, then drains once: each row's cursor takes the same float
#: adds in the same order, so no bit moves on any column and the host
#: column is untouched. The refusals are `add_non_symmetric_trees_packed`'s,
#: sentence for sentence, made at the same first predict.
#: A/B ab3 (NVIDIA L40S, wheel d652a4b0a1, 2026-11-18, job
#: nvc-ab-20261118T182558Z-cb6e02, one run per arm, digests and quality
#: equal on all 24 cells): the per-call pack was arm/base 1.034 geomean
#: (depthwise taxi prebinned infer 15.1 -> 20.2 ms 1.34, lossguide taxi infer
#: 1.10-1.17 one-call vs median-of-20 scoring; the rest 0.99-1.06).
#: TOMBSTONE: MOJOLEARN_IDN_GBDT_PREBINNED_INFER_OLD (LOSER: per-call pack, arm/base geomean 1.034 over 24 cells, NV L40S ab3 2026-11-18) deleted 2026-11-18 by lane/fx-verdicts2; code recoverable at 85c36e324f.
#: Restore: git apply experiments/removed/MOJOLEARN_IDN_GBDT_PREBINNED_INFER_OLD.patch; record in docs/TOMBSTONES.md.
comptime IDN_NS_RESIDENT_PACK = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and IDN_NS_PREDICT_PACKED
)


struct NsResidentPack(Movable):
    """`add_non_symmetric_trees_packed`'s packed ensemble, held: the seven
    node planes and the leaf values on the device, each node's feature word
    offset (`f.offset`, times `n_rows` per call), and the per-row-count
    pieces (the uploaded offsets and the bins buffer) of the last call."""

    var n_trees: Int
    var total_slots: Int
    var node_at: List[Int]
    var val_at: List[Int]
    var nodes_of: List[Int]
    var dims: List[Int]
    #: `Int(f.offset)` per node slot; 0 on a constant tree's placeholder
    var off_unit: List[Int]
    var h_off: HostBuffer[DType.uint32]
    var d_off: DeviceBuffer[DType.uint32]
    var d_mask: DeviceBuffer[DType.uint32]
    var d_shift: DeviceBuffer[DType.uint32]
    var d_oh: DeviceBuffer[DType.uint8]
    var d_bin: DeviceBuffer[DType.uint32]
    var d_ls: DeviceBuffer[DType.uint32]
    var d_rs: DeviceBuffer[DType.uint32]
    var d_vals: DeviceBuffer[DType.float32]
    #: the row count `d_off` holds offsets for; -1 before the first call
    var off_rows: Int
    var bins: Optional[DeviceBuffer[DType.uint32]]
    #: the bins buffer's capacity in rows (its high-water row count); -1 before the first call
    var bins_rows: Int

    def __init__(
        out self,
        ctx: DeviceContext,
        layout: CompressedIndexLayout,
        trees: List[TNonSymmetricTree],
    ) raises:
        """The staging of `add_non_symmetric_trees_packed`, statement for
        statement, without the row count; one drain at the end so the
        staging buffers can go."""
        var n_trees = len(trees)
        var total_slots = 0
        var total_vals = 0
        var node_at = List[Int](capacity=n_trees)
        var val_at = List[Int](capacity=n_trees)
        var nodes_of = List[Int](capacity=n_trees)
        var dims = List[Int](capacity=n_trees)
        for t in range(n_trees):  # small-loop(n_trees: trees): sizes each tree's node and leaf slabs once per model, model parameters
            ref tree = trees[t]
            var dim = tree.dim
            if dim < 1:
                raise Error("add_non_symmetric_tree_to_cursor: dim " + String(dim))
            var n_bins = tree.bin_count()
            if len(tree.leaf_values) != n_bins * dim:
                raise Error(
                    "add_non_symmetric_tree_to_cursor: "
                    + String(len(tree.leaf_values))
                    + " leaf values for " + String(n_bins) + " bins x "
                    + String(dim)
                )
            var n_nodes = len(tree.model_structure.nodes)
            if len(tree.model_structure.split_types) != n_nodes:
                raise Error(
                    "compute_non_symmetric_bins_for_model: " + String(n_nodes)
                    + " nodes and "
                    + String(len(tree.model_structure.split_types))
                    + " split types"
                )
            node_at.append(total_slots)
            val_at.append(total_vals)
            nodes_of.append(n_nodes)
            dims.append(dim)
            total_slots += n_nodes if n_nodes > 0 else 1
            total_vals += n_bins * dim
        var slot_cap = total_slots if total_slots > 0 else 1
        var val_cap = total_vals if total_vals > 0 else 1

        var off_unit = List[Int](length=slot_cap, fill=0)
        var h_off = ctx.enqueue_create_host_buffer[DType.uint32](slot_cap)
        var h_mask = ctx.enqueue_create_host_buffer[DType.uint32](slot_cap)
        var h_shift = ctx.enqueue_create_host_buffer[DType.uint32](slot_cap)
        var h_oh = ctx.enqueue_create_host_buffer[DType.uint8](slot_cap)
        var h_bin = ctx.enqueue_create_host_buffer[DType.uint32](slot_cap)
        var h_ls = ctx.enqueue_create_host_buffer[DType.uint32](slot_cap)
        var h_rs = ctx.enqueue_create_host_buffer[DType.uint32](slot_cap)
        var h_vals = ctx.enqueue_create_host_buffer[DType.float32](val_cap)
        ctx.synchronize()
        for i in range(slot_cap):  # small-loop(slot_cap: node slots of the ensemble): the placeholder records, once per model, model parameters
            h_off.unsafe_ptr().unsafe_store(i, UInt32(0))
            h_mask.unsafe_ptr().unsafe_store(i, UInt32(0))
            h_shift.unsafe_ptr().unsafe_store(i, UInt32(0))
            h_oh.unsafe_ptr().unsafe_store(i, UInt8(0))
            h_bin.unsafe_ptr().unsafe_store(i, UInt32(0))
            h_ls.unsafe_ptr().unsafe_store(i, UInt32(1))
            h_rs.unsafe_ptr().unsafe_store(i, UInt32(1))
        if total_vals == 0:
            h_vals.unsafe_ptr().unsafe_store(0, Float32(0.0))
        for t in range(n_trees):  # small-loop(n_trees: trees): stages each tree's node descriptors and leaf values once per model, model parameters
            ref tree = trees[t]
            ref structure = tree.model_structure
            var base = node_at[t]
            for i in range(len(structure.nodes)):  # small-loop(structure.nodes: nodes of one tree): stages one tree's node descriptors, model parameters
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
                off_unit[base + i] = Int(f.offset)
                h_mask.unsafe_ptr().unsafe_store(base + i, f.mask)
                h_shift.unsafe_ptr().unsafe_store(base + i, f.shift)
                h_oh.unsafe_ptr().unsafe_store(
                    base + i, UInt8(1) if take_bin else UInt8(0)
                )
                h_bin.unsafe_ptr().unsafe_store(base + i, UInt32(Int(n.bin)))
                h_ls.unsafe_ptr().unsafe_store(
                    base + i, UInt32(Int(n.left_subtree))
                )
                h_rs.unsafe_ptr().unsafe_store(
                    base + i, UInt32(Int(n.right_subtree))
                )
            var vbase = val_at[t]
            for i in range(len(tree.leaf_values)):  # small-loop(leaf_values: leaves of one tree): stages one tree's leaf values, model parameters
                h_vals.unsafe_ptr().unsafe_store(vbase + i, tree.leaf_values[i])

        var d_off = ctx.enqueue_create_buffer[DType.uint32](slot_cap)
        var d_mask = ctx.enqueue_create_buffer[DType.uint32](slot_cap)
        var d_shift = ctx.enqueue_create_buffer[DType.uint32](slot_cap)
        var d_oh = ctx.enqueue_create_buffer[DType.uint8](slot_cap)
        var d_bin = ctx.enqueue_create_buffer[DType.uint32](slot_cap)
        var d_ls = ctx.enqueue_create_buffer[DType.uint32](slot_cap)
        var d_rs = ctx.enqueue_create_buffer[DType.uint32](slot_cap)
        var d_vals = ctx.enqueue_create_buffer[DType.float32](val_cap)
        ctx.enqueue_copy(dst_buf=d_mask, src_ptr=h_mask.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=d_shift, src_ptr=h_shift.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=d_oh, src_ptr=h_oh.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=d_bin, src_ptr=h_bin.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=d_ls, src_ptr=h_ls.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=d_rs, src_ptr=h_rs.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=d_vals, src_ptr=h_vals.unsafe_ptr())
        ctx.synchronize()
        # past the drain: the uploads above read the staging buffers
        _ = h_vals^
        _ = h_rs^
        _ = h_ls^
        _ = h_bin^
        _ = h_oh^
        _ = h_shift^
        _ = h_mask^

        self.n_trees = n_trees
        self.total_slots = slot_cap
        self.node_at = node_at^
        self.val_at = val_at^
        self.nodes_of = nodes_of^
        self.dims = dims^
        self.off_unit = off_unit^
        self.h_off = h_off^
        self.d_off = d_off^
        self.d_mask = d_mask^
        self.d_shift = d_shift^
        self.d_oh = d_oh^
        self.d_bin = d_bin^
        self.d_ls = d_ls^
        self.d_rs = d_rs^
        self.d_vals = d_vals^
        self.off_rows = -1
        self.bins = Optional[DeviceBuffer[DType.uint32]]()
        self.bins_rows = -1


def ns_resident_apply(
    ctx: DeviceContext,
    mut pack: NsResidentPack,
    mut cindex: DeviceBuffer[DType.uint32],
    n_rows: Int,
    mut cursor: DeviceBuffer[DType.float32],
) raises:
    """`add_non_symmetric_trees_packed`'s launches over a held pack
    (`IDN_NS_RESIDENT_PACK`): the node offsets are rewritten only when the
    row count changes (the previous call's
    drain has finished the last upload from `h_off`) and the bins buffer
    grown only past its high-water capacity, then the same two
    kernels per tree, in tree order, and ONE drain."""
    if pack.n_trees == 0:
        ctx.synchronize()
        return
    if pack.off_rows != n_rows:
        var ho = pack.h_off.unsafe_ptr()
        for i in range(pack.total_slots):  # small-loop(total_slots: node slots of the ensemble): the per-row-count word offsets, model parameters
            ho.unsafe_store(i, UInt32(pack.off_unit[i] * n_rows))
        ctx.enqueue_copy(dst_buf=pack.d_off, src_ptr=ho)
        pack.off_rows = n_rows
    # Lane fix-wave2-leftovers (audit NOTE ns_resident_apply): the bins
    # buffer keeps its high-water capacity; it is replaced only when a call
    # has more rows than it holds (was: on every row-count change, so
    # alternating batch sizes reallocated per call). Both kernels take
    # n_rows and touch bins[0:n_rows] only, so the tail past n_rows is never
    # read (same bits). The previous call's drain ended every read of the
    # old buffer.
    var need = n_rows if n_rows > 0 else 1
    if pack.bins_rows < need:
        pack.bins = None
        pack.bins = ctx.enqueue_create_buffer[DType.uint32](need)
        pack.bins_rows = need
    var bin_blocks = (n_rows + NS_BINS_BLOCK_SIZE - 1) // NS_BINS_BLOCK_SIZE
    if bin_blocks < 1:
        bin_blocks = 1
    var per_block = ABMV_BLOCK * ABMV_ELEMENTS
    var add_blocks = (n_rows + per_block - 1) // per_block
    if add_blocks < 1:
        add_blocks = 1
    # driver-loop: over the trees of the ensemble (model structure)
    for t in range(pack.n_trees):
        var base = pack.node_at[t]
        var vbase = pack.val_at[t]
        ctx.enqueue_function[compute_non_symmetric_decision_tree_bins_kernel](
            pack.d_off.unsafe_ptr() + base,
            pack.d_mask.unsafe_ptr() + base,
            pack.d_shift.unsafe_ptr() + base,
            pack.d_oh.unsafe_ptr() + base,
            pack.d_bin.unsafe_ptr() + base,
            pack.d_ls.unsafe_ptr() + base,
            pack.d_rs.unsafe_ptr() + base,
            Int32(pack.nodes_of[t]),
            cindex.unsafe_ptr(),
            Int32(n_rows),
            pack.bins.value().unsafe_ptr(),
            grid_dim=(bin_blocks, 1, 1),
            block_dim=(NS_BINS_BLOCK_SIZE, 1, 1),
        )
        ctx.enqueue_function[add_bin_model_value_kernel](
            pack.d_vals.unsafe_ptr() + vbase,
            pack.bins.value().unsafe_ptr(),
            Int32(n_rows),
            Int32(pack.dims[t]),
            Int32(n_rows),
            cursor.unsafe_ptr(),
            grid_dim=(add_blocks, 1, 1),
            block_dim=(ABMV_BLOCK, 1, 1),
        )
    ctx.synchronize()
