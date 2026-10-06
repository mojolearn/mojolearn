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

from max.gpu.host import DeviceBuffer, DeviceContext
from std.sys.compile import is_defined
from std.gpu import block_idx, block_dim, thread_idx
from core.forest_experiments import C50_GB_PACKED

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
#: `-D MOJOLEARN_IDN_GBDT_NS_PREDICT_PACKED_OFF` (or the master
#: `-D MOJOLEARN_IDN_ALL_OFF`) restores the per-tree loop.
comptime IDN_NS_PREDICT_PACKED = C50_GB_PACKED or (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not (
        is_defined["MOJOLEARN_IDN_GBDT_NS_PREDICT_PACKED_OFF"]()
        or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
    )
)


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

    comptime if C50_GB_PACKED:
        if total_slots > 2147483647//8 or total_vals > 2147483647 or n_trees > 2147483647//4:
            raise Error("C50 GB packed node/metadata offsets exceed Int32")
        var packed = ctx.enqueue_create_buffer[DType.uint32](8*total_slots)
        ctx.enqueue_function[_c50_pack_ns_kernel](d_off.unsafe_ptr(), d_mask.unsafe_ptr(), d_shift.unsafe_ptr(), d_oh.unsafe_ptr(),
            d_bin.unsafe_ptr(), d_ls.unsafe_ptr(), d_rs.unsafe_ptr(), packed.unsafe_ptr(), Int32(total_slots),
            grid_dim=(total_slots+255)//256, block_dim=256)
        var hm = ctx.enqueue_create_host_buffer[DType.int32](4*n_trees)
        for t in range(n_trees):
            hm.unsafe_ptr().unsafe_store(4*t, Int32(node_at[t]))
            hm.unsafe_ptr().unsafe_store(4*t+1, Int32(val_at[t]))
            hm.unsafe_ptr().unsafe_store(4*t+2, Int32(len(trees[t].model_structure.nodes)))
            hm.unsafe_ptr().unsafe_store(4*t+3, Int32(trees[t].dim))
        var dm = ctx.enqueue_create_buffer[DType.int32](4*n_trees)
        ctx.enqueue_copy(dst_buf=dm, src_ptr=hm.unsafe_ptr())
        var first = 0
        while first < n_trees:
            # Cache working set, not benchmark shape, limits the tree group.
            # A single larger tree remains indivisible; constant trees cost one
            # record so every group is bounded even in an all-constant forest.
            var last = first
            var bytes = 0
            while last < n_trees:
                var next_bytes = 32*max(1, len(trees[last].model_structure.nodes))
                if last > first and bytes+next_bytes > 64*1024:
                    break
                bytes += next_bytes
                last += 1
            ctx.enqueue_function[_c50_ns_rows_kernel](packed.unsafe_ptr(), dm.unsafe_ptr(), cindex.unsafe_ptr(), d_vals.unsafe_ptr(),
                cursor.unsafe_ptr(), Int32(n_rows), Int32(first), Int32(last), grid_dim=max(1,(n_rows+127)//128), block_dim=128)
            first = last
        ctx.synchronize()
        _ = dm^
        _ = hm^
        _ = packed^
        return

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

# C50 GB non-symmetric inference: exact packed integer records; no compiler
# struct lowering or unsupported reinterpretation. Same shift/mask and tree fold.
# NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
def _c50_pack_ns_kernel(offset: MutPointer[UInt32, MutAnyOrigin], mask: MutPointer[UInt32, MutAnyOrigin],
    shift: MutPointer[UInt32, MutAnyOrigin], equal: MutPointer[UInt8, MutAnyOrigin], value: MutPointer[UInt32, MutAnyOrigin],
    left: MutPointer[UInt32, MutAnyOrigin], right: MutPointer[UInt32, MutAnyOrigin], packed: MutPointer[UInt32, MutAnyOrigin], n: Int32):
    var i = Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if i < Int(n):
        packed.unsafe_store(8*i, offset.unsafe_load(i))
        packed.unsafe_store(8*i+1, mask.unsafe_load(i))
        packed.unsafe_store(8*i+2, shift.unsafe_load(i))
        packed.unsafe_store(8*i+3, UInt32(equal.unsafe_load(i)))
        packed.unsafe_store(8*i+4, value.unsafe_load(i))
        packed.unsafe_store(8*i+5, left.unsafe_load(i))
        packed.unsafe_store(8*i+6, right.unsafe_load(i))
        packed.unsafe_store(8*i+7, UInt32(0))


def _c50_ns_rows_kernel(packed: MutPointer[UInt32, MutAnyOrigin], metadata: MutPointer[Int32, MutAnyOrigin],
    cindex: MutPointer[UInt32, MutAnyOrigin], values: MutPointer[Float32, MutAnyOrigin], cursor: MutPointer[Float32, MutAnyOrigin],
    rows: Int32, first: Int32, last: Int32):
    var row = Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if row >= Int(rows):
        return
    for t in range(Int(first), Int(last)):
        var desc = metadata.unsafe_load[width=4](4*t)
        var base = Int(desc[0])
        var node = 0
        var leaf = 0
        var stop = desc[2] == 0
        while not stop:
            var feature = packed.unsafe_load[width=4](8*(base+node))
            var decision = packed.unsafe_load[width=4](8*(base+node)+4)
            var value = (cindex.unsafe_load(Int(feature[0])+row) >> feature[2]) & feature[1]
            var split = value == decision[0] if feature[3] != 0 else value > decision[0]
            if split:
                leaf += Int(decision[1])
                stop = decision[2] == 1
                if not stop:
                    node += Int(decision[1])
            else:
                stop = decision[1] == 1
                if not stop:
                    node += 1
        for d in range(Int(desc[3])):
            var at = d*Int(rows)+row
            cursor.unsafe_store(at, cursor.unsafe_load(at)+values.unsafe_load(Int(desc[1])+leaf*Int(desc[3])+d))
