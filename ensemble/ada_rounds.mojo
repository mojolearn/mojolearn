# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""AdaBoost's boosting rounds on the device (lane n5-prophet-adaboost-loops,
2026-11-09; plan docs/plans/EXECUTOR_CONSOLIDATION_AND_CLEANUP.md section 4,
row "AdaBoost").

A forest data session (`bindings/_mojolearn_rf.mojo`, `RfDataSession`) keeps
the ensemble's state resident across rounds (`AdaState`): the boosting
weights as binary64 words, the labels (class codes) or targets, and the
class_weight row weights. One round is one native call: the member's
float32 weights (`w * n`, then times the class_weight rows) built on the
device, the member fit (the RF builder, unchanged) on the session's X, the
member's predictions of the training X here (`ada_codes_kernel` /
`ada_values_kernel`), and the SAMME / R2 step on the device
(`xtrees/ops_device_boost.mojo` `samme_step_on` / `r2_step_on`, the same
kernels and folds as the per-call step). Only the step's four scalar words
and the member's model cross to the host.

THE PREDICTIONS are the member's own predict, operation for operation: the
walk of `core/forest_inference.mojo` `reached_leaf` (RF input: the value
flushed, the finite-key `<=` comparison, the local left-child index), the
sum over trees by `forest_add` from +0.0 in tree order and the
`identical_div` by the tree count, then the first maximum over the outputs
(numpy's argmax) for a classifier. The session's X is the column-major
plane, so a row's feature f is `x[f * n + row]`; the member's predict reads
the same float32 values."""

from std.gpu import block_dim, block_idx, thread_idx
from std.math import ceildiv
from std.memory import bitcast
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import ftz, identical_div
from core.forest_inference import finite_key, forest_add
from ensemble.decisiontree.decisiontree import TreeMetaDataNode
from xtrees.ops_device_elem import mul_f32_kernel, scale_kernel, scale_to_f32_kernel

comptime ADA_TPB = 256


struct AdaState(Movable):
    """The device-resident state of one AdaBoost fit: `w` the boosting
    weights (n binary64 words, updated in place every round), `codes` the
    class codes (classifier; one placeholder word for a regressor), `values`
    the targets (regressor; one placeholder float otherwise), `cw` the
    class_weight row weights when `has_cw`. `k` is the class count, 0 for a
    regressor."""

    var n: Int
    var k: Int
    var w: DeviceBuffer[DType.uint64]
    var codes: DeviceBuffer[DType.int32]
    var values: DeviceBuffer[DType.float32]
    var cw: DeviceBuffer[DType.float32]
    var has_cw: Bool

    def __init__(
        out self,
        n: Int,
        k: Int,
        var w: DeviceBuffer[DType.uint64],
        var codes: DeviceBuffer[DType.int32],
        var values: DeviceBuffer[DType.float32],
        var cw: DeviceBuffer[DType.float32],
        has_cw: Bool,
    ):
        self.n = n
        self.k = k
        self.w = w^
        self.codes = codes^
        self.values = values^
        self.cw = cw^
        self.has_cw = has_cw


@always_inline
def _f64_word(v: Float64) -> UInt64:
    return bitcast[DType.uint64](v)


@always_inline
def _leaf_colmajor(
    offsets: MutPointer[Int32, MutAnyOrigin], columns: MutPointer[Int32, MutAnyOrigin],
    thresholds: MutPointer[Float32, MutAnyOrigin], left: MutPointer[Int32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin], tree: Int, row: Int, n: Int,
) -> Int:
    """`reached_leaf[RF_INPUT=True]` over a column-major plane."""
    var base = Int(offsets.unsafe_load(tree))
    var node = base
    var child = Int(left.unsafe_load(node))
    while child != -1:
        var value = ftz(x.unsafe_load(Int(columns.unsafe_load(node)) * n + row))
        var go_left = finite_key(value) <= finite_key(thresholds.unsafe_load(node))
        node = base + child + (0 if go_left else 1)
        child = Int(left.unsafe_load(node))
    return node


def ada_codes_kernel(
    offsets: MutPointer[Int32, MutAnyOrigin], columns: MutPointer[Int32, MutAnyOrigin],
    thresholds: MutPointer[Float32, MutAnyOrigin], left: MutPointer[Int32, MutAnyOrigin],
    leaves: MutPointer[Float32, MutAnyOrigin], x: MutPointer[Float32, MutAnyOrigin],
    codes: MutPointer[Int32, MutAnyOrigin], n_in: Int32, outputs_in: Int32, trees_in: Int32,
):
    """One thread per row: the class code a classifier member predicts (the
    first maximum of its mean leaf distribution)."""
    var n = Int(n_in)
    var outputs = Int(outputs_in)
    var trees = Int(trees_in)
    var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if row >= n:
        return
    var best = 0
    var best_v = Float32(0)
    for o in range(outputs):
        var total = Float32(0)
        for tree in range(trees):
            var node = _leaf_colmajor(offsets, columns, thresholds, left, x, tree, row, n)
            total = forest_add(total, leaves.unsafe_load(node * outputs + o))
        var v = ftz(identical_div(ftz(total), Float32(trees)))
        if o == 0 or v > best_v:
            best = o
            best_v = v
    codes.unsafe_store(row, Int32(best))


def ada_values_kernel(
    offsets: MutPointer[Int32, MutAnyOrigin], columns: MutPointer[Int32, MutAnyOrigin],
    thresholds: MutPointer[Float32, MutAnyOrigin], left: MutPointer[Int32, MutAnyOrigin],
    leaves: MutPointer[Float32, MutAnyOrigin], x: MutPointer[Float32, MutAnyOrigin],
    values: MutPointer[Float32, MutAnyOrigin], n_in: Int32, trees_in: Int32,
):
    """One thread per row: the value a regressor member predicts (its mean
    leaf value)."""
    var n = Int(n_in)
    var trees = Int(trees_in)
    var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if row >= n:
        return
    var total = Float32(0)
    for tree in range(trees):
        var node = _leaf_colmajor(offsets, columns, thresholds, left, x, tree, row, n)
        total = forest_add(total, leaves.unsafe_load(node))
    values.unsafe_store(row, ftz(identical_div(ftz(total), Float32(trees))))


struct AdaNodes(Movable):
    """A fitted member's flat model on the device (the export layout:
    tree offsets, per-node column / threshold / local left child, the leaf
    vectors), for its predictions of the training X."""

    var offsets: DeviceBuffer[DType.int32]
    var columns: DeviceBuffer[DType.int32]
    var thresholds: DeviceBuffer[DType.float32]
    var left: DeviceBuffer[DType.int32]
    var leaves: DeviceBuffer[DType.float32]
    var trees: Int
    var outputs: Int

    def __init__(
        out self,
        var offsets: DeviceBuffer[DType.int32],
        var columns: DeviceBuffer[DType.int32],
        var thresholds: DeviceBuffer[DType.float32],
        var left: DeviceBuffer[DType.int32],
        var leaves: DeviceBuffer[DType.float32],
        trees: Int,
        outputs: Int,
    ):
        self.offsets = offsets^
        self.columns = columns^
        self.thresholds = thresholds^
        self.left = left^
        self.leaves = leaves^
        self.trees = trees
        self.outputs = outputs


def ada_upload_nodes(ctx: DeviceContext, trees: List[TreeMetaDataNode[DType.float32]]) raises -> AdaNodes:
    """The member's trees in the export layout (`rf_export_trees_into`'s
    fields), uploaded once: model-sized, a few nodes for a stump."""
    var n_trees = len(trees)
    if n_trees < 1:
        raise Error("AdaBoost round: the member has no tree")
    var outputs = Int(trees[0].num_outputs)
    var offsets = List[Int32](capacity=n_trees + 1)
    offsets.append(Int32(0))
    var total = 0
    for t in range(n_trees):  # driver-loop: over the member's fitted trees (model structure)
        total += len(trees[t].sparsetree)
        offsets.append(Int32(total))
    var columns = List[Int32](capacity=total)
    var thresholds = List[Float32](capacity=total)
    var left = List[Int32](capacity=total)
    var leaves = List[Float32](capacity=total * outputs)
    for t in range(n_trees):  # driver-loop: over the member's fitted trees (model structure)
        ref tree = trees[t]
        for i in range(len(tree.sparsetree)):  # driver-loop: over one tree's nodes (model structure)
            ref node = tree.sparsetree[i]
            columns.append(Int32(node.ColumnId()))
            thresholds.append(node.QueryValue())
            left.append(Int32(node.LeftChildId()))
        leaves.extend(tree.vector_leaf.copy())
    var d_off = ctx.enqueue_create_buffer[DType.int32](n_trees + 1)
    var d_col = ctx.enqueue_create_buffer[DType.int32](max(1, total))
    var d_thr = ctx.enqueue_create_buffer[DType.float32](max(1, total))
    var d_left = ctx.enqueue_create_buffer[DType.int32](max(1, total))
    var d_leaf = ctx.enqueue_create_buffer[DType.float32](max(1, len(leaves)))
    ctx.enqueue_copy(dst_buf=d_off, src_ptr=offsets.unsafe_ptr())
    if total > 0:
        ctx.enqueue_copy(dst_buf=d_col, src_ptr=columns.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=d_thr, src_ptr=thresholds.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=d_left, src_ptr=left.unsafe_ptr())
    if len(leaves) > 0:
        ctx.enqueue_copy(dst_buf=d_leaf, src_ptr=leaves.unsafe_ptr())
    # the host lists are read by the copies above: wait before they drop
    ctx.synchronize()
    _ = offsets^
    _ = columns^
    _ = thresholds^
    _ = left^
    _ = leaves^
    return AdaNodes(d_off^, d_col^, d_thr^, d_left^, d_leaf^, n_trees, outputs)


def ada_predict_codes(ctx: DeviceContext, nodes: AdaNodes, x: DeviceBuffer[DType.float32], n: Int,
                      codes: DeviceBuffer[DType.int32]) raises:
    """The member's class codes for the n rows of the column-major `x`."""
    ctx.enqueue_function[ada_codes_kernel](
        nodes.offsets.unsafe_ptr(), nodes.columns.unsafe_ptr(), nodes.thresholds.unsafe_ptr(),
        nodes.left.unsafe_ptr(), nodes.leaves.unsafe_ptr(), x.unsafe_ptr(), codes.unsafe_ptr(),
        Int32(n), Int32(nodes.outputs), Int32(nodes.trees),
        grid_dim=ceildiv(n, ADA_TPB), block_dim=ADA_TPB,
    )


def ada_predict_values(ctx: DeviceContext, nodes: AdaNodes, x: DeviceBuffer[DType.float32], n: Int,
                       values: DeviceBuffer[DType.float32]) raises:
    """The member's values for the n rows of the column-major `x`."""
    ctx.enqueue_function[ada_values_kernel](
        nodes.offsets.unsafe_ptr(), nodes.columns.unsafe_ptr(), nodes.thresholds.unsafe_ptr(),
        nodes.left.unsafe_ptr(), nodes.leaves.unsafe_ptr(), x.unsafe_ptr(), values.unsafe_ptr(),
        Int32(n), Int32(nodes.trees),
        grid_dim=ceildiv(n, ADA_TPB), block_dim=ADA_TPB,
    )


def ada_member_weights(ctx: DeviceContext, state: AdaState, dsw: DeviceBuffer[DType.float32]) raises:
    """The member's float32 weights from the resident boosting weights:
    `float32(w * n)` (`ops.scale_to_f32`: one binary64 product, one
    narrowing), then times the class_weight row weights (`ops.mul_f32`: one
    correctly rounded float32 product), the host's `_w32` +
    `trees_weighted_rows` per element."""
    var n = state.n
    var grid = max(1, min(ceildiv(n, ADA_TPB), 65535))
    ctx.enqueue_function[scale_to_f32_kernel](
        state.w.unsafe_ptr(), Int64(n), _f64_word(Float64(n)), dsw.unsafe_ptr(),
        grid_dim=grid, block_dim=ADA_TPB,
    )
    if state.has_cw:
        ctx.enqueue_function[mul_f32_kernel](
            dsw.unsafe_ptr(), state.cw.unsafe_ptr(), Int64(n), dsw.unsafe_ptr(),
            grid_dim=grid, block_dim=ADA_TPB,
        )


def ada_rescale(ctx: DeviceContext, state: AdaState, total: Float64) raises:
    """w /= total in place (`ops.scale_f64`, the host's `_scale`)."""
    var n = state.n
    ctx.enqueue_function[scale_kernel](
        state.w.unsafe_ptr(), Int64(n), _f64_word(total),
        grid_dim=max(1, min(ceildiv(n, ADA_TPB), 65535)), block_dim=ADA_TPB,
    )
