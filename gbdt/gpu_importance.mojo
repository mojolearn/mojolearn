# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`feature_importances_` of a fitted gradient-boosting model, THE DEVICE
COLUMN (lane mf-imp6, 2026-11-23). What is summed, the bit rules and the
plan are `core/gbdt_importance.mojo`'s; every kernel here calls that
module's `@always_inline` bodies (`gbdt_imp_bin_body`, `gbdt_imp_row_leaf`,
`gbdt_imp_weight_word`, `gbdt_imp_tree_body`, `gbdt_imp_fold_body`,
`gbdt_imp_finish_body`), so the host column and every vendor return the same
bits. No kernel calls another kernel.

Launches, in order, on the image's process context:
  1 `_imp_exponent_kernel`  one thread per row: the largest weight exponent
                            (an integer `Atomic.max`, order free);
  2 `_imp_bins_kernel`      one thread per (row, model column): the bin byte;
  3 `_imp_leaf_kernel`      one thread per (row, tree): the row's leaf, its
                            weight word added to the leaf's (low, high) pair
                            with two 32-bit atomics and an exact carry (the
                            64-bit sum in any order; Apple has no 64-bit
                            atomics);
  4 `_imp_tree_kernel`      one thread per tree: its decreases;
  5 `_imp_fold_kernel`      one thread per input column: the sum over trees;
  6 `_imp_finish_kernel`    one thread: the normalisation (its fold order is
                            the bits).
One upload of the plan, X (the model columns) and the weights; one download
of the `n_input` results and the refusal word.

The CTR expansion of a categorical model's columns is the host statement the
GPU predict also runs (`gbdt/train.mojo::_expand_model_columns`), done once
by `gbdt_importance_input` before the upload; the importance work itself is
on the device.
"""
from std.atomic import Atomic
from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from core.fold_x_device import fold_x_fill

from core.gbdt_importance import (
    GbdtImpF32, GbdtImpI32, GbdtImpU32, GbdtImpU8, GbdtImportanceInput, gbdt_importance_input,
    gbdt_importance_fold_rows,
    gbdt_imp_bin_body, gbdt_imp_exponent, gbdt_imp_finish_body, gbdt_imp_fold_body, gbdt_imp_row_leaf,
    gbdt_imp_tree_body, gbdt_imp_weight_word,
)

comptime IMP_TPB = 256
#: the trees of one leaf-weight launch (the grid's y extent; CUDA caps it at 65535)
comptime IMP_TREES_PER_LAUNCH = 32768


def _imp_exponent_kernel(w: GbdtImpF32, e_max: GbdtImpI32, n_rows_: Int32):
    var n_rows = Int(n_rows_)
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if r >= n_rows:
        return
    var e = gbdt_imp_exponent(w[r])
    if e > 1:
        _ = Atomic.max(e_max.unsafe_offset(0), Int32(e))


def _imp_bins_kernel(x: GbdtImpF32, bins: GbdtImpU8, border_lo: GbdtImpI32, borders: GbdtImpF32,
                     treat: GbdtImpI32, sorted: GbdtImpI32, n_rows_: Int32, n_cols_: Int32, row_major_: Int32):
    var n_rows = Int(n_rows_)
    var n_cols = Int(n_cols_)
    var row_major = Int(row_major_)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= n_rows * n_cols:
        return
    var c = cell // n_rows
    var r = cell - c * n_rows
    var v: Float32
    if row_major != 0:
        v = x[r * n_cols + c]
    else:
        v = x[c * n_rows + r]
    var lo = Int(border_lo[c])
    bins[cell] = gbdt_imp_bin_body(v, lo, Int(border_lo[c + 1]) - lo, sorted[c] != 0, Int(treat[c]), borders)


def _imp_leaf_kernel(bins: GbdtImpU8, w: GbdtImpF32, e_max: GbdtImpI32, wpair: GbdtImpU32, fail: GbdtImpI32,
                     tree_lo: GbdtImpI32, leaf_lo: GbdtImpI32, split_feature: GbdtImpI32, split_bin: GbdtImpI32,
                     split_take: GbdtImpI32, node_left: GbdtImpI32, node_right: GbdtImpI32, n_rows_: Int32,
                     dim_: Int32, non_symmetric_: Int32, n_weights_: Int32, t0_: Int32):
    var n_rows = Int(n_rows_)
    var dim = Int(dim_)
    var non_symmetric = Int(non_symmetric_)
    var n_weights = Int(n_weights_)
    var t0 = Int(t0_)
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var t = t0 + Int(block_idx.y)
    if r >= n_rows:
        return
    var leaf = gbdt_imp_row_leaf(t, r, bins, n_rows, tree_lo, split_feature, split_bin, split_take, node_left,
                                 node_right, non_symmetric != 0)
    if leaf < 0:
        _ = Atomic.max(fail.unsafe_offset(0), Int32(t + 1))
        return
    var q = UInt32(1)
    if n_weights > 0:
        q = gbdt_imp_weight_word(w[r], Int(e_max[0]))
    if q == UInt32(0):
        return
    var slot = Int(leaf_lo[t]) // dim + leaf
    var old = Atomic.fetch_add(wpair.unsafe_offset(2 * slot), q)
    if old > UInt32(0xFFFFFFFF) - q:
        # the low word wrapped: exactly one carry into the high word
        _ = Atomic.fetch_add(wpair.unsafe_offset(2 * slot + 1), UInt32(1))


def _imp_tree_kernel(tree_lo: GbdtImpI32, leaf_lo: GbdtImpI32, split_feature: GbdtImpI32, node_left: GbdtImpI32,
                     node_right: GbdtImpI32, leaves: GbdtImpF32, wpair: GbdtImpU32, map_lo: GbdtImpI32,
                     map_col: GbdtImpI32, map_share: GbdtImpF32, contrib: GbdtImpF32, n_trees_: Int32, dim_: Int32,
                     non_symmetric_: Int32, n_cols_: Int32, n_input_: Int32):
    var n_trees = Int(n_trees_)
    var dim = Int(dim_)
    var non_symmetric = Int(non_symmetric_)
    var n_cols = Int(n_cols_)
    var n_input = Int(n_input_)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= n_trees:
        return
    gbdt_imp_tree_body(t, tree_lo, leaf_lo, dim, non_symmetric != 0, split_feature, node_left, node_right, leaves,
                       wpair, n_cols, map_lo, map_col, map_share, n_input, contrib)


def _imp_fold_kernel(contrib: GbdtImpF32, total: GbdtImpF32, n_trees_: Int32, n_input_: Int32):
    var n_trees = Int(n_trees_)
    var n_input = Int(n_input_)
    var c = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if c >= n_input:
        return
    gbdt_imp_fold_body(c, n_trees, n_input, contrib, total)


def _imp_finish_kernel(total: GbdtImpF32, dst: GbdtImpF32, n_input_: Int32):
    var n_input = Int(n_input_)
    if Int(block_idx.x) != 0 or Int(thread_idx.x) != 0:
        return
    gbdt_imp_finish_body(n_input, total, dst)


def _blocks(n: Int) -> Int:
    return max((n + IMP_TPB - 1) // IMP_TPB, 1)


def _up_i32(ctx: DeviceContext, values: List[Int32]) raises -> DeviceBuffer[DType.int32]:
    var buf = ctx.enqueue_create_buffer[DType.int32](max(len(values), 1))
    if len(values) > 0:
        ctx.enqueue_copy(dst_buf=buf, src_ptr=values.unsafe_ptr())
    return buf^


def _up_f32(ctx: DeviceContext, values: List[Float32]) raises -> DeviceBuffer[DType.float32]:
    var buf = ctx.enqueue_create_buffer[DType.float32](max(len(values), 1))
    if len(values) > 0:
        ctx.enqueue_copy(dst_buf=buf, src_ptr=values.unsafe_ptr())
    return buf^


def gbdt_feature_importances_device(
    ctx: DeviceContext, text: String, x_addr: Int, n_rows: Int, n_input: Int, row_major: Bool, w_addr: Int,
    n_weights: Int, dst_addr: Int,
) raises -> Int:
    """`feature_importances_` (THE DEVICE COLUMN) of the model `text` fitted
    on the `n_rows` x `n_input` float32 host rows at `x_addr` (C order when
    `row_major`) with `n_weights` (0 or `n_rows`) host sample weights at
    `w_addr`; the `n_input` float32 results to the host `dst_addr`. Returns
    `n_input`. The caller keeps its buffers alive until this returns (it
    synchronizes)."""
    var request = gbdt_importance_input(text, x_addr, n_rows, n_input, row_major, n_weights)
    var fold_rows = List[Float32]()
    if request.needs_rows:
        # A CTR or expanded model fitted on THE FOLD X WORD (lane
        # vdf2-gbdtcat2): its model columns are the predict path's expansion
        # (`gbdt_host_expand_ctr`, a host statement on every column, the GPU
        # predict's too) of the fit's rows, so the slot's rows (the
        # categorical columns already the fit's codes) come home once, here,
        # after the fit; the expanded columns then go up as for a host fit.
        fold_rows = List[Float32](unsafe_uninit_length=max(n_rows * n_input, 1))
        var d_rows = ctx.enqueue_create_buffer[DType.float32](max(n_rows * n_input, 1))
        fold_x_fill(ctx, d_rows, x_addr, n_rows * n_input)
        ctx.enqueue_copy(
            dst_ptr=MutPointer[Float32, MutAnyOrigin](unsafe_from_address=Int(fold_rows.unsafe_ptr())), src_buf=d_rows
        )
        ctx.synchronize()
        _ = d_rows^
        gbdt_importance_fold_rows(request, Int(fold_rows.unsafe_ptr()), n_rows, n_input)
    ref plan = request.plan
    var n_cols = plan.n_cols
    var n_trees = plan.n_trees
    # the model columns: the caller's X (n_cols == n_input) or the expanded copy
    var n_x = n_rows * n_cols
    var d_x = ctx.enqueue_create_buffer[DType.float32](max(n_x, 1))
    # THE FOLD X WORD (lane vdf-scope-trees4, core/fold_x_device.mojo): a
    # negative address is the fit's fold slot, copied device to device; a
    # host address is the old upload
    fold_x_fill(ctx, d_x, request.x_addr, n_x)
    var d_w = ctx.enqueue_create_buffer[DType.float32](max(n_weights, 1))
    if n_weights > 0:
        ctx.enqueue_copy(dst_buf=d_w, src_ptr=MutPointer[Float32, MutAnyOrigin](unsafe_from_address=w_addr))
    var d_border_lo = _up_i32(ctx, plan.border_lo)
    var d_borders = _up_f32(ctx, plan.borders)
    var d_treat = _up_i32(ctx, plan.treat)
    var d_sorted = _up_i32(ctx, plan.sorted)
    var d_tree_lo = _up_i32(ctx, plan.tree_lo)
    var d_leaf_lo = _up_i32(ctx, plan.leaf_lo)
    var d_split_feature = _up_i32(ctx, plan.split_feature)
    var d_split_bin = _up_i32(ctx, plan.split_bin)
    var d_split_take = _up_i32(ctx, plan.split_take)
    var d_node_left = _up_i32(ctx, plan.node_left)
    var d_node_right = _up_i32(ctx, plan.node_right)
    var d_leaves = _up_f32(ctx, plan.leaves)
    var d_map_lo = _up_i32(ctx, plan.map_lo)
    var d_map_col = _up_i32(ctx, plan.map_col)
    var d_map_share = _up_f32(ctx, plan.map_share)
    var d_bins = ctx.enqueue_create_buffer[DType.uint8](max(n_rows * n_cols, 1))
    # two buffers, never two views of one (a launch takes both mutably)
    var d_emax = ctx.enqueue_create_buffer[DType.int32](1)
    d_emax.enqueue_fill(Int32(1))
    var d_fail = ctx.enqueue_create_buffer[DType.int32](1)
    d_fail.enqueue_fill(Int32(0))
    var d_wpair = ctx.enqueue_create_buffer[DType.uint32](2 * max(plan.n_leaf_slots, 1))
    d_wpair.enqueue_fill(UInt32(0))
    var d_contrib = ctx.enqueue_create_buffer[DType.float32](max(n_trees * plan.n_input, 1))
    var d_total = ctx.enqueue_create_buffer[DType.float32](max(plan.n_input, 1))
    var d_out = ctx.enqueue_create_buffer[DType.float32](max(plan.n_input, 1))
    var h_fail = ctx.enqueue_create_host_buffer[DType.int32](1)
    if n_weights > 0:
        ctx.enqueue_function[_imp_exponent_kernel](
            d_w.unsafe_ptr(), d_emax.unsafe_ptr(), Int32(n_rows),
            grid_dim=(_blocks(n_rows), 1, 1), block_dim=(IMP_TPB, 1, 1),
        )
    ctx.enqueue_function[_imp_bins_kernel](
        d_x.unsafe_ptr(), d_bins.unsafe_ptr(), d_border_lo.unsafe_ptr(), d_borders.unsafe_ptr(),
        d_treat.unsafe_ptr(), d_sorted.unsafe_ptr(), Int32(n_rows), Int32(n_cols), Int32(1 if request.row_major else 0),
        grid_dim=(_blocks(n_rows * n_cols), 1, 1), block_dim=(IMP_TPB, 1, 1),
    )
    var t0 = 0
    while t0 < n_trees:  # driver-loop: over launches of at most IMP_TREES_PER_LAUNCH trees
        var count = min(IMP_TREES_PER_LAUNCH, n_trees - t0)
        ctx.enqueue_function[_imp_leaf_kernel](
            d_bins.unsafe_ptr(), d_w.unsafe_ptr(), d_emax.unsafe_ptr(), d_wpair.unsafe_ptr(), d_fail.unsafe_ptr(),
            d_tree_lo.unsafe_ptr(), d_leaf_lo.unsafe_ptr(), d_split_feature.unsafe_ptr(), d_split_bin.unsafe_ptr(),
            d_split_take.unsafe_ptr(), d_node_left.unsafe_ptr(), d_node_right.unsafe_ptr(), Int32(n_rows), Int32(plan.dim),
            Int32(1 if plan.non_symmetric else 0), Int32(n_weights), Int32(t0),
            grid_dim=(_blocks(n_rows), count, 1), block_dim=(IMP_TPB, 1, 1),
        )
        t0 += count
    if n_trees > 0:
        ctx.enqueue_function[_imp_tree_kernel](
            d_tree_lo.unsafe_ptr(), d_leaf_lo.unsafe_ptr(), d_split_feature.unsafe_ptr(), d_node_left.unsafe_ptr(),
            d_node_right.unsafe_ptr(), d_leaves.unsafe_ptr(), d_wpair.unsafe_ptr(), d_map_lo.unsafe_ptr(),
            d_map_col.unsafe_ptr(), d_map_share.unsafe_ptr(), d_contrib.unsafe_ptr(), Int32(n_trees), Int32(plan.dim),
            Int32(1 if plan.non_symmetric else 0), Int32(n_cols), Int32(plan.n_input),
            grid_dim=(_blocks(n_trees), 1, 1), block_dim=(IMP_TPB, 1, 1),
        )
    ctx.enqueue_function[_imp_fold_kernel](
        d_contrib.unsafe_ptr(), d_total.unsafe_ptr(), Int32(n_trees), Int32(plan.n_input),
        grid_dim=(_blocks(plan.n_input), 1, 1), block_dim=(IMP_TPB, 1, 1),
    )
    ctx.enqueue_function[_imp_finish_kernel](
        d_total.unsafe_ptr(), d_out.unsafe_ptr(), Int32(plan.n_input),
        grid_dim=(1, 1, 1), block_dim=(1, 1, 1),
    )
    ctx.enqueue_copy(dst_ptr=h_fail.unsafe_ptr(), src_buf=d_fail)
    ctx.enqueue_copy(dst_ptr=MutPointer[Float32, MutAnyOrigin](unsafe_from_address=dst_addr), src_buf=d_out)
    ctx.synchronize()
    var failed = Int(h_fail.unsafe_ptr().unsafe_load(0))
    # every buffer a queued launch read stays alive past the synchronize
    _ = d_x^
    _ = d_w^
    _ = d_border_lo^
    _ = d_borders^
    _ = d_treat^
    _ = d_sorted^
    _ = d_tree_lo^
    _ = d_leaf_lo^
    _ = d_split_feature^
    _ = d_split_bin^
    _ = d_split_take^
    _ = d_node_left^
    _ = d_node_right^
    _ = d_leaves^
    _ = d_map_lo^
    _ = d_map_col^
    _ = d_map_share^
    _ = d_bins^
    _ = d_fail^
    _ = d_emax^
    _ = d_wpair^
    _ = d_contrib^
    _ = d_total^
    _ = d_out^
    _ = h_fail^
    _ = len(request.model_x)  # the upload read the expanded model columns through `x_addr`
    _ = len(fold_rows)  # the expansion read the fold's rows
    if failed > 0:
        raise Error("gbdt importances: non-symmetric tree " + String(failed - 1) + " walks a row out of its tree")
    return plan.n_input
