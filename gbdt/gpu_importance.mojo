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

The CTR expansion of a categorical model's columns (lane vd-leftovers,
2026-11-24) runs ON THE DEVICE before launch 2 (`gbdt_imp_expand_device`):
the fit's raw rows (one upload of host rows, or THE FOLD X WORD's device
copy) go through the resident predict's kernels
(`gbdt/models/kernel/resident_ctr.mojo` for the simple and tensor CTRs,
`gbdt/models/kernel/combination_code.mojo` for the tree CTRs), which write
the model columns, column-major, straight into the buffer launch 2 reads.
Those kernels compute the host statements' words (every float32 operation
correctly rounded once, `checks/soft_f64.mojo`), so the host column's
`gbdt_importance_model_x` and every vendor give the same model columns. The
refusal words come home with the results. No row visits the host.
"""
from std.atomic import Atomic
from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from core.fold_x_device import fold_x_fill

from core.gbdt_importance import (
    GbdtImpF32, GbdtImpI32, GbdtImpU32, GbdtImpU8, GbdtImportanceInput, gbdt_importance_input,
    gbdt_imp_bin_body, gbdt_imp_exponent, gbdt_imp_finish_body, gbdt_imp_fold_body, gbdt_imp_row_leaf,
    gbdt_imp_tree_body, gbdt_imp_weight_word,
)

from gbdt.models.kernel.combination_code import ResidentCombinationPlan, enqueue_combination_columns
from gbdt.models.kernel.resident_ctr import (
    CTR_BORDER_SLAB,
    ResidentCtrPlan,
    _upload_f32,
    ctr_plan_raise_if_bad_device,
    ctr_plan_size_bins,
    enqueue_ctr_expand_kernels,
)
from gbdt.models.tensor_ctr_apply import TTensorCtrApplyTable
from gbdt.models.trained_model import TrainedModel

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


struct GbdtImpCtrStage(Movable):
    """What the device expansion's queued launches read, kept alive by the
    caller until its drain: the CTR plans, the raw rows on the device and
    the border slabs (lane vd-leftovers)."""

    var ctr: ResidentCtrPlan
    var comb: Optional[ResidentCombinationPlan]
    var d_in: DeviceBuffer[DType.float32]
    var d_borders: DeviceBuffer[DType.float32]

    def __init__(
        out self,
        var ctr: ResidentCtrPlan,
        var comb: Optional[ResidentCombinationPlan],
        var d_in: DeviceBuffer[DType.float32],
        var d_borders: DeviceBuffer[DType.float32],
    ):
        self.ctr = ctr^
        self.comb = comb^
        self.d_in = d_in^
        self.d_borders = d_borders^


def gbdt_imp_expand_device(
    ctx: DeviceContext, tm: TrainedModel, x_word: Int, n_rows: Int, n_raw: Int, row_major: Bool, n_columns: Int,
    mut d_out: DeviceBuffer[DType.float32],
) raises -> GbdtImpCtrStage:
    """A CTR model's `n_columns` MODEL columns (column-major, `n_columns *
    n_rows`) into `d_out`, ENQUEUED on `ctx`, from the fit's `n_rows` x
    `n_raw` raw rows at `x_word` (a host address: one upload; THE FOLD X
    WORD: one device-to-device copy of the slot; C order when `row_major`;
    the categorical columns as the fit's dense codes). The plans are the
    resident predict's (`gbdt/resident_model.mojo` builds the same ones
    from the same model): the simple plan over the base columns, the
    tree-CTR plan over the model's last columns, the tensor plan with the
    model's border slabs. The refusal words' copy home is enqueued last;
    after its drain the caller calls `ctr_plan_raise_if_bad_device` on the
    returned stage, which it keeps alive until then."""
    var tensor_tables = List[TTensorCtrApplyTable]()
    ref reg = tm.tensor_ctr_registry
    # driver-loop: over the registered tensor CTR model columns (model structure)
    for i in range(len(reg.features)):  # small-loop(features: registered tensor CTR model columns): builds the apply plan once per call
        if reg.features[i].model_column != reg.first_model_column + i:
            raise Error("tensor CTR registry model columns are not contiguous")
        tensor_tables.append(reg.features[i].table.apply_table())
    var n_comb = len(tm.combination_ctrs.tables)
    var n_base = n_columns - n_comb
    var base_one_hot = List[Bool]()
    # driver-loop: over the base model columns (their one-hot flags)
    for c in range(min(n_base, len(tm.one_hot))):  # small-loop(n_base: model columns): packs the one-hot flags once per call
        base_one_hot.append(tm.one_hot[c])
    var ctr = ResidentCtrPlan(ctx, tm.ctr_tables, tensor_tables, n_base, base_one_hot, n_raw)
    var comb = Optional[ResidentCombinationPlan](None)
    if n_comb > 0:
        comb = ResidentCombinationPlan(ctx, tm.combination_ctrs, n_columns, n_raw)
    # the border slabs the tensor kernels read (`resident_model`'s layout:
    # per model column `[count, border_0, ...]`); a simple plan reads none
    var slab_words = 1
    if len(tensor_tables) != 0:
        slab_words = n_columns * CTR_BORDER_SLAB
    var slab = List[Float32](length=slab_words, fill=Float32(0.0))
    if len(tensor_tables) != 0:
        if len(tm.borders) != n_columns:
            raise Error("gbdt importances: the model's borders do not match its columns")
        # driver-loop: over the model columns (model data)
        for f in range(n_columns):  # small-loop(n_columns: model columns): packs the border slabs once per call
            var count = len(tm.borders[f])
            if count >= CTR_BORDER_SLAB:
                raise Error(
                    "feature " + String(f) + " has " + String(count)
                    + " borders; the staging slab holds " + String(CTR_BORDER_SLAB - 1)
                )
            slab[f * CTR_BORDER_SLAB] = Float32(count)
            for b in range(count):  # small-loop(count: borders of one model column, under 256)
                slab[f * CTR_BORDER_SLAB + 1 + b] = tm.borders[f][b]
    var d_borders = _upload_f32(ctx, slab)
    _ = len(slab)  # the upload copied from its storage (and waited)
    # the raw rows on the device, once: the old upload or the slot's copy
    var d_in = ctx.enqueue_create_buffer[DType.float32](max(n_rows * n_raw, 1))
    fold_x_fill(ctx, d_in, x_word, n_rows * n_raw)
    ctr_plan_size_bins(ctx, ctr, n_rows)
    enqueue_ctr_expand_kernels(ctx, ctr, d_in.unsafe_ptr(), n_rows, row_major, d_borders, d_out)
    if comb:
        # the tree-CTR columns flag into the same refusal words
        enqueue_combination_columns(ctx, comb.value(), d_in, n_rows, row_major, d_out, ctr.d_bad)
    ctx.enqueue_copy(dst_ptr=ctr.h_bad.unsafe_ptr(), src_buf=ctr.d_bad)
    return GbdtImpCtrStage(ctr^, comb^, d_in^, d_borders^)


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
    var request = gbdt_importance_input(text, x_addr, n_rows, n_input, row_major, n_weights, True)
    ref plan = request.plan
    var n_cols = plan.n_cols
    var n_trees = plan.n_trees
    # the model columns: the caller's X (n_cols == n_input) or, for a CTR
    # model, its expansion on the device (column-major)
    var n_x = n_rows * n_cols
    var d_x = ctx.enqueue_create_buffer[DType.float32](max(n_x, 1))
    var x_row_major = request.row_major
    var stage = Optional[GbdtImpCtrStage](None)
    if request.expand_on_device:
        # lane vd-leftovers: the raw rows (host or THE FOLD X WORD) are
        # expanded by the device kernels into `d_x`; nothing comes home
        stage = gbdt_imp_expand_device(
            ctx, request.model, request.x_addr, n_rows, n_input, request.row_major, n_cols, d_x
        )
        x_row_major = False
    else:
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
        d_treat.unsafe_ptr(), d_sorted.unsafe_ptr(), Int32(n_rows), Int32(n_cols), Int32(1 if x_row_major else 0),
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
    _ = len(request.model_x)  # empty on the device column (the expansion is on the device)
    if stage:
        # the expansion's launches read the stage's plans and rows until the drain above
        ctr_plan_raise_if_bad_device(stage.value().ctr)
    _ = stage^
    if failed > 0:
        raise Error("gbdt importances: non-symmetric tree " + String(failed - 1) + " walks a row out of its tree")
    return plan.n_input
