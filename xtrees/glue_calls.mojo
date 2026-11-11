# SPDX-License-Identifier: Apache-2.0
"""Typed native calls of the x_trees glue operations through the trees entry
table (lane calltable-trees, 2026-11-11). No Python: a caller passes buffer
addresses and plain scalars and keeps the buffers alive until the call
returns. Each wrapper lays its arguments out in the positions the run
documents (`xtrees/glue_runs.mojo`, ABI in `xtrees/glue_abi.mojo`): an
integer parameter goes to the int view, a float parameter to the float view.

`xt_glue` reads the published table once per call (a lock-free acquire load
and a schema compare) and calls the pointer; a missing or mismatched
publication refuses with the table's error. The refusal text of a run is the
text its former Python entry raised.
"""
from core.native_descriptor_call import native_descriptor_call
from core.trees_entry_table import TreesEntryKey, trees_table_entry, TREES_TABLE_X_TREES
from xtrees.glue_abi import (
    XT_GLUE_ABI, XT_OP_COUNT, XT_ACCUMULATE_COLS_MANY, XT_OVR_ASSEMBLE, XT_CALIBRATED_PROBA,
    XT_STACK_W64, XT_ACCUMULATE, XT_ACCUMULATE_COLS, XT_ACCUMULATE_ONEHOT, XT_ACCUMULATE_ROWS, XT_APPLY,
    XT_ARGMAX_ROWS, XT_BAG_ROWS, XT_BINARY_PROBA, XT_CLASS_COUNTS, XT_CLASS_PRIORS, XT_COMPLEMENT_PAIRS,
    XT_COUNT_EQUAL, XT_COUNT_ROWS, XT_DART_THRESHOLDS, XT_DART_RESCALE, XT_DEVICE_FOLDS, XT_EXACT_SUM,
    XT_FILL_CLASS_MAJOR_F64, XT_GATHER_F32, XT_GATHER_I32, XT_INDICATOR_CODES, XT_IOTA_I32, XT_ISOTONIC_FIT,
    XT_ISOTONIC_PREDICT_STRIDED, XT_LEAF_NUMBERING, XT_LOG64, XT_MARGIN2, XT_NORMALIZE_ROWS, XT_ONEHOT_LEAVES,
    XT_OOB_R2, XT_PLATT_APPLY_STRIDED, XT_PLATT_FIT, XT_POSITIVE_CODES, XT_PUT_F32, XT_R2_STEP, XT_REMAP_COLS,
    XT_ROWS_ANY_UNSET, XT_SAMME_STEP, XT_SAMPLE_INDICES, XT_SCALE, XT_SCALE_TO_F32, XT_SCATTER, XT_SOFTMAX_ROWS,
    XT_TRANSPOSE_F32, XT_TRANSPOSE_F64, XT_UNIFORM, XT_UNSEEN_ROWS, XT_WEIGHTED_MEDIAN, XT_WEIGHTED_SAMPLE,
)


def xt_glue(key: TreesEntryKey, op: Int, addrs: List[Int], ints: List[Int], floats: List[Float64]) raises -> Int:
    """One glue operation in the x_trees image the key names."""
    var entry = trees_table_entry[TREES_TABLE_X_TREES](key, 0, String(XT_GLUE_ABI), XT_OP_COUNT)
    return native_descriptor_call(entry, op, addrs, ints, floats)


def xt_accumulate(key: TreesEntryKey, acc: Int, x: Int, n: Int, weight: Float64) raises -> Int:
    var addrs: List[Int] = [acc, x]
    var ints: List[Int] = [n, 0]
    var floats: List[Float64] = [Float64(n), weight]
    return xt_glue(key, XT_ACCUMULATE, addrs, ints, floats)


def xt_accumulate_cols(key: TreesEntryKey, acc: Int, x: Int, cols: Int, n: Int, k: Int, ks: Int, weight: Float64) raises -> Int:
    var addrs: List[Int] = [acc, x, cols]
    var ints: List[Int] = [n, k, ks, 0]
    var floats: List[Float64] = [Float64(n), Float64(k), Float64(ks), weight]
    return xt_glue(key, XT_ACCUMULATE_COLS, addrs, ints, floats)


def xt_accumulate_onehot(key: TreesEntryKey, acc: Int, codes: Int, n: Int, k: Int, on: Float64, off: Float64) raises -> Int:
    var addrs: List[Int] = [acc, codes]
    var ints: List[Int] = [n, k, 0, 0]
    var floats: List[Float64] = [Float64(n), Float64(k), on, off]
    return xt_glue(key, XT_ACCUMULATE_ONEHOT, addrs, ints, floats)


def xt_accumulate_rows(key: TreesEntryKey, acc: Int, x: Int, rows: Int, n: Int, k: Int, m: Int) raises -> Int:
    var addrs: List[Int] = [acc, x, rows]
    var ints: List[Int] = [n, k, m]
    var floats: List[Float64] = [Float64(n), Float64(k), Float64(m)]
    return xt_glue(key, XT_ACCUMULATE_ROWS, addrs, ints, floats)


def xt_apply(key: TreesEntryKey, offsets: Int, colid: Int, quesval: Int, left: Int, x: Int, res: Int, n: Int, d: Int, t0: Int, t1: Int) raises -> Int:
    var addrs: List[Int] = [offsets, colid, quesval, left, x, res]
    var ints: List[Int] = [n, d, t0, t1]
    var floats: List[Float64] = [Float64(n), Float64(d), Float64(t0), Float64(t1)]
    return xt_glue(key, XT_APPLY, addrs, ints, floats)


def xt_argmax_rows(key: TreesEntryKey, x: Int, res: Int, n: Int, k: Int) raises -> Int:
    var addrs: List[Int] = [x, res]
    var ints: List[Int] = [n, k]
    var floats: List[Float64] = [Float64(n), Float64(k)]
    return xt_glue(key, XT_ARGMAX_ROWS, addrs, ints, floats)


def xt_bag_rows(key: TreesEntryKey, res: Int, n: Int, seed: Int, stream: Int, frac: Float64) raises -> Int:
    var addrs: List[Int] = [res]
    var ints: List[Int] = [n, seed, stream, 0]
    var floats: List[Float64] = [Float64(n), Float64(seed), Float64(stream), frac]
    return xt_glue(key, XT_BAG_ROWS, addrs, ints, floats)


def xt_binary_proba(key: TreesEntryKey, p: Int, res: Int, n: Int) raises -> Int:
    var addrs: List[Int] = [p, res]
    var ints: List[Int] = [n]
    var floats: List[Float64] = [Float64(n)]
    return xt_glue(key, XT_BINARY_PROBA, addrs, ints, floats)


def xt_class_counts(key: TreesEntryKey, y: Int, counts: Int, n: Int, k: Int) raises -> Int:
    var addrs: List[Int] = [y, counts]
    var ints: List[Int] = [n, k]
    var floats: List[Float64] = [Float64(n), Float64(k)]
    return xt_glue(key, XT_CLASS_COUNTS, addrs, ints, floats)


def xt_class_priors(key: TreesEntryKey, counts: Int, priors: Int, n: Int, k: Int) raises -> Int:
    var addrs: List[Int] = [counts, priors]
    var ints: List[Int] = [n, k]
    var floats: List[Float64] = [Float64(n), Float64(k)]
    return xt_glue(key, XT_CLASS_PRIORS, addrs, ints, floats)


def xt_complement_pairs(key: TreesEntryKey, x: Int, n: Int) raises -> Int:
    var addrs: List[Int] = [x]
    var ints: List[Int] = [n]
    var floats: List[Float64] = [Float64(n)]
    return xt_glue(key, XT_COMPLEMENT_PAIRS, addrs, ints, floats)


def xt_count_equal(key: TreesEntryKey, a: Int, b: Int, n: Int) raises -> Int:
    var addrs: List[Int] = [a, b]
    var ints: List[Int] = [n]
    var floats: List[Float64] = [Float64(n)]
    return xt_glue(key, XT_COUNT_EQUAL, addrs, ints, floats)


def xt_count_rows(key: TreesEntryKey, counts: Int, rows: Int, n: Int, m: Int) raises -> Int:
    var addrs: List[Int] = [counts, rows]
    var ints: List[Int] = [n, m]
    var floats: List[Float64] = [Float64(n), Float64(m)]
    return xt_glue(key, XT_COUNT_ROWS, addrs, ints, floats)


def xt_dart_thresholds(key: TreesEntryKey, weights: Int, coefs: Int, coef32: Int, thr: Int, state: Int, t: Int, k: Int, uniform_drop: Int, max_drop: Int, rate: Float64) raises -> Int:
    var addrs: List[Int] = [weights, coefs, coef32, thr, state]
    var ints: List[Int] = [t, k, uniform_drop, max_drop, 0]
    var floats: List[Float64] = [Float64(t), Float64(k), Float64(uniform_drop), Float64(max_drop), rate]
    return xt_glue(key, XT_DART_THRESHOLDS, addrs, ints, floats)


def xt_dart_rescale(key: TreesEntryKey, weights: Int, coefs: Int, flags: Int, state: Int, res3: Int, t: Int, k: Int, uniform_drop: Int, xgb: Int, lr: Float64) raises -> Int:
    var addrs: List[Int] = [weights, coefs, flags, state, res3]
    var ints: List[Int] = [t, k, uniform_drop, xgb, 0]
    var floats: List[Float64] = [Float64(t), Float64(k), Float64(uniform_drop), Float64(xgb), lr]
    return xt_glue(key, XT_DART_RESCALE, addrs, ints, floats)


def xt_device_folds(key: TreesEntryKey, codes: Int, rows: Int, counts: Int, n: Int, n_splits: Int, n_classes: Int) raises -> Int:
    var addrs: List[Int] = [codes, rows, counts]
    var ints: List[Int] = [n, n_splits, n_classes]
    var floats: List[Float64] = [Float64(n), Float64(n_splits), Float64(n_classes)]
    return xt_glue(key, XT_DEVICE_FOLDS, addrs, ints, floats)


def xt_exact_sum(key: TreesEntryKey, x: Int, res: Int, flags: Int, n: Int, want_div: Int) raises -> Int:
    var addrs: List[Int] = [x, res, flags]
    var ints: List[Int] = [n, want_div]
    var floats: List[Float64] = [Float64(n), Float64(want_div)]
    return xt_glue(key, XT_EXACT_SUM, addrs, ints, floats)


def xt_fill_class_major_f64(key: TreesEntryKey, inits: Int, dst: Int, n: Int, k: Int) raises -> Int:
    var addrs: List[Int] = [inits, dst]
    var ints: List[Int] = [n, k]
    var floats: List[Float64] = [Float64(n), Float64(k)]
    return xt_glue(key, XT_FILL_CLASS_MAJOR_F64, addrs, ints, floats)


def xt_gather_f32(key: TreesEntryKey, src: Int, rows: Int, cols: Int, dst: Int, n_src_rows: Int, n_src_cols: Int, n_rows: Int, n_cols: Int) raises -> Int:
    var addrs: List[Int] = [src, rows, cols, dst]
    var ints: List[Int] = [n_src_rows, n_src_cols, n_rows, n_cols]
    var floats: List[Float64] = [Float64(n_src_rows), Float64(n_src_cols), Float64(n_rows), Float64(n_cols)]
    return xt_glue(key, XT_GATHER_F32, addrs, ints, floats)


def xt_gather_i32(key: TreesEntryKey, src: Int, rows: Int, dst: Int, n_src: Int, n_rows: Int) raises -> Int:
    var addrs: List[Int] = [src, rows, dst]
    var ints: List[Int] = [n_src, n_rows]
    var floats: List[Float64] = [Float64(n_src), Float64(n_rows)]
    return xt_glue(key, XT_GATHER_I32, addrs, ints, floats)


def xt_indicator_codes(key: TreesEntryKey, codes: Int, out_i: Int, out_f: Int, n: Int, cls: Int) raises -> Int:
    var addrs: List[Int] = [codes, out_i, out_f]
    var ints: List[Int] = [n, cls]
    var floats: List[Float64] = [Float64(n), Float64(cls)]
    return xt_glue(key, XT_INDICATOR_CODES, addrs, ints, floats)


def xt_iota_i32(key: TreesEntryKey, dst: Int, n: Int) raises -> Int:
    var addrs: List[Int] = [dst]
    var ints: List[Int] = [n]
    var floats: List[Float64] = [Float64(n)]
    return xt_glue(key, XT_IOTA_I32, addrs, ints, floats)


def xt_isotonic_fit(key: TreesEntryKey, x: Int, y: Int, kx: Int, ky: Int, n: Int) raises -> Int:
    var addrs: List[Int] = [x, y, kx, ky]
    var ints: List[Int] = [n]
    var floats: List[Float64] = [Float64(n)]
    return xt_glue(key, XT_ISOTONIC_FIT, addrs, ints, floats)


def xt_isotonic_predict_strided(key: TreesEntryKey, kx: Int, ky: Int, t: Int, res: Int, m: Int, n: Int, t_len: Int, t_stride: Int, t_off: Int, res_len: Int, res_stride: Int, res_off: Int) raises -> Int:
    var addrs: List[Int] = [kx, ky, t, res]
    var ints: List[Int] = [m, n, t_len, t_stride, t_off, res_len, res_stride, res_off]
    var floats: List[Float64] = [Float64(m), Float64(n), Float64(t_len), Float64(t_stride), Float64(t_off), Float64(res_len), Float64(res_stride), Float64(res_off)]
    return xt_glue(key, XT_ISOTONIC_PREDICT_STRIDED, addrs, ints, floats)


def xt_leaf_numbering(key: TreesEntryKey, left: Int, node_col: Int, n_nodes: Int) raises -> Int:
    var addrs: List[Int] = [left, node_col]
    var ints: List[Int] = [n_nodes]
    var floats: List[Float64] = [Float64(n_nodes)]
    return xt_glue(key, XT_LEAF_NUMBERING, addrs, ints, floats)


def xt_log64(key: TreesEntryKey, res: Int, x: Float64) raises -> Int:
    var addrs: List[Int] = [res]
    var ints: List[Int] = [0]
    var floats: List[Float64] = [x]
    return xt_glue(key, XT_LOG64, addrs, ints, floats)


def xt_margin2(key: TreesEntryKey, acc: Int, dst: Int, n: Int, mode: Int) raises -> Int:
    var addrs: List[Int] = [acc, dst]
    var ints: List[Int] = [n, mode]
    var floats: List[Float64] = [Float64(n), Float64(mode)]
    return xt_glue(key, XT_MARGIN2, addrs, ints, floats)


def xt_normalize_rows(key: TreesEntryKey, x: Int, n: Int, k: Int) raises -> Int:
    var addrs: List[Int] = [x]
    var ints: List[Int] = [n, k]
    var floats: List[Float64] = [Float64(n), Float64(k)]
    return xt_glue(key, XT_NORMALIZE_ROWS, addrs, ints, floats)


def xt_onehot_leaves(key: TreesEntryKey, nodes: Int, tree_base: Int, node_col: Int, res: Int, n: Int, n_trees: Int, n_cols: Int) raises -> Int:
    var addrs: List[Int] = [nodes, tree_base, node_col, res]
    var ints: List[Int] = [n, n_trees, n_cols]
    var floats: List[Float64] = [Float64(n), Float64(n_trees), Float64(n_cols)]
    return xt_glue(key, XT_ONEHOT_LEAVES, addrs, ints, floats)


def xt_oob_r2(key: TreesEntryKey, acc: Int, counts: Int, y: Int, pred: Int, words: Int, flags: Int, n: Int) raises -> Int:
    var addrs: List[Int] = [acc, counts, y, pred, words, flags]
    var ints: List[Int] = [n]
    var floats: List[Float64] = [Float64(n)]
    return xt_glue(key, XT_OOB_R2, addrs, ints, floats)


def xt_platt_apply_strided(key: TreesEntryKey, f: Int, res: Int, n: Int, a: Float64, b: Float64, f_len: Int, f_stride: Int, f_off: Int, res_len: Int, res_stride: Int, res_off: Int) raises -> Int:
    var addrs: List[Int] = [f, res]
    var ints: List[Int] = [n, 0, 0, f_len, f_stride, f_off, res_len, res_stride, res_off]
    var floats: List[Float64] = [Float64(n), a, b, Float64(f_len), Float64(f_stride), Float64(f_off), Float64(res_len), Float64(res_stride), Float64(res_off)]
    return xt_glue(key, XT_PLATT_APPLY_STRIDED, addrs, ints, floats)


def xt_platt_fit(key: TreesEntryKey, f: Int, y: Int, ab: Int, n: Int) raises -> Int:
    var addrs: List[Int] = [f, y, ab]
    var ints: List[Int] = [n]
    var floats: List[Float64] = [Float64(n)]
    return xt_glue(key, XT_PLATT_FIT, addrs, ints, floats)


def xt_positive_codes(key: TreesEntryKey, x: Int, codes: Int, n: Int) raises -> Int:
    var addrs: List[Int] = [x, codes]
    var ints: List[Int] = [n]
    var floats: List[Float64] = [Float64(n)]
    return xt_glue(key, XT_POSITIVE_CODES, addrs, ints, floats)


def xt_put_f32(key: TreesEntryKey, dst: Int, src: Int, offset: Int, n: Int) raises -> Int:
    var addrs: List[Int] = [dst, src]
    var ints: List[Int] = [offset, n]
    var floats: List[Float64] = [Float64(offset), Float64(n)]
    return xt_glue(key, XT_PUT_F32, addrs, ints, floats)


def xt_r2_step(key: TreesEntryKey, w: Int, pred: Int, y: Int, stats: Int, n: Int, loss: Int, learning_rate: Float64, last: Int, tree_base: Int) raises -> Int:
    var addrs: List[Int] = [w, pred, y, stats]
    var ints: List[Int] = [n, loss, 0, last, tree_base]
    var floats: List[Float64] = [Float64(n), Float64(loss), learning_rate, Float64(last), Float64(tree_base)]
    return xt_glue(key, XT_R2_STEP, addrs, ints, floats)


def xt_remap_cols(key: TreesEntryKey, colid: Int, cols: Int, n_nodes: Int, m: Int) raises -> Int:
    var addrs: List[Int] = [colid, cols]
    var ints: List[Int] = [n_nodes, m]
    var floats: List[Float64] = [Float64(n_nodes), Float64(m)]
    return xt_glue(key, XT_REMAP_COLS, addrs, ints, floats)


def xt_rows_any_unset(key: TreesEntryKey, eq: Int, out_i: Int, n: Int, m: Int, col_major: Int) raises -> Int:
    var addrs: List[Int] = [eq, out_i]
    var ints: List[Int] = [n, m, col_major]
    var floats: List[Float64] = [Float64(n), Float64(m), Float64(col_major)]
    return xt_glue(key, XT_ROWS_ANY_UNSET, addrs, ints, floats)


def xt_samme_step(key: TreesEntryKey, w: Int, pred: Int, y: Int, stats: Int, n: Int, n_classes: Int, learning_rate: Float64, last: Int, tree_base: Int) raises -> Int:
    var addrs: List[Int] = [w, pred, y, stats]
    var ints: List[Int] = [n, n_classes, 0, last, tree_base]
    var floats: List[Float64] = [Float64(n), Float64(n_classes), learning_rate, Float64(last), Float64(tree_base)]
    return xt_glue(key, XT_SAMME_STEP, addrs, ints, floats)


def xt_sample_indices(key: TreesEntryKey, dst: Int, n_pool: Int, n_draw: Int, replace: Int, seed: Int, stream: Int) raises -> Int:
    var addrs: List[Int] = [dst]
    var ints: List[Int] = [n_pool, n_draw, replace, seed, stream]
    var floats: List[Float64] = [Float64(n_pool), Float64(n_draw), Float64(replace), Float64(seed), Float64(stream)]
    return xt_glue(key, XT_SAMPLE_INDICES, addrs, ints, floats)


def xt_scale(key: TreesEntryKey, x: Int, n: Int, divisor: Float64) raises -> Int:
    var addrs: List[Int] = [x]
    var ints: List[Int] = [n, 0]
    var floats: List[Float64] = [Float64(n), divisor]
    return xt_glue(key, XT_SCALE, addrs, ints, floats)


def xt_scale_to_f32(key: TreesEntryKey, x: Int, res: Int, n: Int, factor: Float64) raises -> Int:
    var addrs: List[Int] = [x, res]
    var ints: List[Int] = [n, 0]
    var floats: List[Float64] = [Float64(n), factor]
    return xt_glue(key, XT_SCALE_TO_F32, addrs, ints, floats)


def xt_scatter(key: TreesEntryKey, dst: Int, src: Int, rows: Int, n_dst_rows: Int, n_dst_cols: Int, m: Int, c: Int, col0: Int) raises -> Int:
    var addrs: List[Int] = [dst, src, rows]
    var ints: List[Int] = [n_dst_rows, n_dst_cols, m, c, col0]
    var floats: List[Float64] = [Float64(n_dst_rows), Float64(n_dst_cols), Float64(m), Float64(c), Float64(col0)]
    return xt_glue(key, XT_SCATTER, addrs, ints, floats)


def xt_softmax_rows(key: TreesEntryKey, x: Int, n: Int, k: Int) raises -> Int:
    var addrs: List[Int] = [x]
    var ints: List[Int] = [n, k]
    var floats: List[Float64] = [Float64(n), Float64(k)]
    return xt_glue(key, XT_SOFTMAX_ROWS, addrs, ints, floats)


def xt_transpose_f32(key: TreesEntryKey, src: Int, dst: Int, n: Int, d: Int) raises -> Int:
    var addrs: List[Int] = [src, dst]
    var ints: List[Int] = [n, d]
    var floats: List[Float64] = [Float64(n), Float64(d)]
    return xt_glue(key, XT_TRANSPOSE_F32, addrs, ints, floats)


def xt_transpose_f64(key: TreesEntryKey, src: Int, dst: Int, n: Int, d: Int) raises -> Int:
    var addrs: List[Int] = [src, dst]
    var ints: List[Int] = [n, d]
    var floats: List[Float64] = [Float64(n), Float64(d)]
    return xt_glue(key, XT_TRANSPOSE_F64, addrs, ints, floats)


def xt_uniform(key: TreesEntryKey, res: Int, n: Int, seed: Int, stream: Int) raises -> Int:
    var addrs: List[Int] = [res]
    var ints: List[Int] = [n, seed, stream]
    var floats: List[Float64] = [Float64(n), Float64(seed), Float64(stream)]
    return xt_glue(key, XT_UNIFORM, addrs, ints, floats)


def xt_unseen_rows(key: TreesEntryKey, rows: Int, res: Int, m: Int, n: Int) raises -> Int:
    var addrs: List[Int] = [rows, res]
    var ints: List[Int] = [m, n]
    var floats: List[Float64] = [Float64(m), Float64(n)]
    return xt_glue(key, XT_UNSEEN_ROWS, addrs, ints, floats)


def xt_weighted_median(key: TreesEntryKey, preds: Int, weights: Int, res: Int, n: Int, m: Int) raises -> Int:
    var addrs: List[Int] = [preds, weights, res]
    var ints: List[Int] = [n, m]
    var floats: List[Float64] = [Float64(n), Float64(m)]
    return xt_glue(key, XT_WEIGHTED_MEDIAN, addrs, ints, floats)


def xt_weighted_sample(key: TreesEntryKey, w: Int, dst: Int, n: Int, n_draw: Int, seed: Int, stream: Int) raises -> Int:
    var addrs: List[Int] = [w, dst]
    var ints: List[Int] = [n, n_draw, seed, stream]
    var floats: List[Float64] = [Float64(n), Float64(n_draw), Float64(seed), Float64(stream)]
    return xt_glue(key, XT_WEIGHTED_SAMPLE, addrs, ints, floats)

def xt_stack_w64(key: TreesEntryKey, addrs: List[Int], n: Int) raises -> Int:
    """addrs = [dst, column 0, ..., column m-1] (see stack_w64_run)."""
    var ints: List[Int] = [n]
    var floats: List[Float64] = [Float64(n)]
    return xt_glue(key, XT_STACK_W64, addrs, ints, floats)


def xt_accumulate_cols_many(key: TreesEntryKey, addrs: List[Int], ints: List[Int], floats: List[Float64]) raises -> Int:
    """addrs = [acc, x_0, cols_0, ...]; params = [n, k, ks_0, weight_0, ...]
    as int/float views (see accumulate_cols_many_run)."""
    return xt_glue(key, XT_ACCUMULATE_COLS_MANY, addrs, ints, floats)


def xt_ovr_assemble(key: TreesEntryKey, addrs: List[Int], ints: List[Int], floats: List[Float64]) raises -> Int:
    """addrs = [out, blk_0, ...]; params = [n, k, finish, bc_0, at_0, ...]
    (see ovr_assemble_run)."""
    return xt_glue(key, XT_OVR_ASSEMBLE, addrs, ints, floats)


def xt_calibrated_proba(key: TreesEntryKey, addrs: List[Int], ints: List[Int], floats: List[Float64]) raises -> Int:
    """The flattened pairs and calibrators of calibrated_proba_run."""
    return xt_glue(key, XT_CALIBRATED_PROBA, addrs, ints, floats)
