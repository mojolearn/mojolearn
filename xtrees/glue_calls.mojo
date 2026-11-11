# SPDX-License-Identifier: Apache-2.0
"""Typed native calls of the x_trees glue operations through the trees entry
table (lane calltable-trees, 2026-11-11). No Python: a caller passes buffer
addresses and plain scalars and keeps the buffers alive until the call
returns. Each wrapper lays its arguments out in the positions the run
documents (`xtrees/glue_runs.mojo`, ABI in `xtrees/glue_abi.mojo`): an
integer parameter goes to the int view, a float parameter to the float view.

Lane entry-table-zero-alloc (2026-11-11): the caller resolves the table once
(`xt_ref`, where a missing or mismatched publication refuses with the table's
error) and passes the `EntryRef` to every wrapper; `xt_glue` is one indirect
call through it, with no table read. The refusal text of a run is the text its
former Python entry raised.
"""
from core.entry_table import EntryRef, entry_ref_call, entry_ref_word
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


def xt_ref(key: TreesEntryKey) raises -> EntryRef:
    """The glue entry of the x_trees image the key names, resolved once."""
    return EntryRef(trees_table_entry[TREES_TABLE_X_TREES](key, 0, String(XT_GLUE_ABI), XT_OP_COUNT),
                    entry_ref_word(key.mode, key.vendor, XT_OP_COUNT))


def xt_glue(glue: EntryRef, op: Int, addrs: List[Int], ints: List[Int], floats: List[Float64]) raises -> Int:
    """One glue operation through the resolved entry."""
    return entry_ref_call(glue, op, addrs, ints, floats)


def xt_accumulate(glue: EntryRef, acc: Int, x: Int, n: Int, weight: Float64) raises -> Int:
    var addrs: List[Int] = [acc, x]
    var ints: List[Int] = [n, 0]
    var floats: List[Float64] = [Float64(n), weight]
    return xt_glue(glue, XT_ACCUMULATE, addrs, ints, floats)


def xt_accumulate_cols(glue: EntryRef, acc: Int, x: Int, cols: Int, n: Int, k: Int, ks: Int, weight: Float64) raises -> Int:
    var addrs: List[Int] = [acc, x, cols]
    var ints: List[Int] = [n, k, ks, 0]
    var floats: List[Float64] = [Float64(n), Float64(k), Float64(ks), weight]
    return xt_glue(glue, XT_ACCUMULATE_COLS, addrs, ints, floats)


def xt_accumulate_onehot(glue: EntryRef, acc: Int, codes: Int, n: Int, k: Int, on: Float64, off: Float64) raises -> Int:
    var addrs: List[Int] = [acc, codes]
    var ints: List[Int] = [n, k, 0, 0]
    var floats: List[Float64] = [Float64(n), Float64(k), on, off]
    return xt_glue(glue, XT_ACCUMULATE_ONEHOT, addrs, ints, floats)


def xt_accumulate_rows(glue: EntryRef, acc: Int, x: Int, rows: Int, n: Int, k: Int, m: Int) raises -> Int:
    var addrs: List[Int] = [acc, x, rows]
    var ints: List[Int] = [n, k, m]
    var floats: List[Float64] = [Float64(n), Float64(k), Float64(m)]
    return xt_glue(glue, XT_ACCUMULATE_ROWS, addrs, ints, floats)


def xt_apply(glue: EntryRef, offsets: Int, colid: Int, quesval: Int, left: Int, x: Int, res: Int, n: Int, d: Int, t0: Int, t1: Int) raises -> Int:
    var addrs: List[Int] = [offsets, colid, quesval, left, x, res]
    var ints: List[Int] = [n, d, t0, t1]
    var floats: List[Float64] = [Float64(n), Float64(d), Float64(t0), Float64(t1)]
    return xt_glue(glue, XT_APPLY, addrs, ints, floats)


def xt_argmax_rows(glue: EntryRef, x: Int, res: Int, n: Int, k: Int) raises -> Int:
    var addrs: List[Int] = [x, res]
    var ints: List[Int] = [n, k]
    var floats: List[Float64] = [Float64(n), Float64(k)]
    return xt_glue(glue, XT_ARGMAX_ROWS, addrs, ints, floats)


def xt_bag_rows(glue: EntryRef, res: Int, n: Int, seed: Int, stream: Int, frac: Float64) raises -> Int:
    var addrs: List[Int] = [res]
    var ints: List[Int] = [n, seed, stream, 0]
    var floats: List[Float64] = [Float64(n), Float64(seed), Float64(stream), frac]
    return xt_glue(glue, XT_BAG_ROWS, addrs, ints, floats)


def xt_binary_proba(glue: EntryRef, p: Int, res: Int, n: Int) raises -> Int:
    var addrs: List[Int] = [p, res]
    var ints: List[Int] = [n]
    var floats: List[Float64] = [Float64(n)]
    return xt_glue(glue, XT_BINARY_PROBA, addrs, ints, floats)


def xt_class_counts(glue: EntryRef, y: Int, counts: Int, n: Int, k: Int) raises -> Int:
    var addrs: List[Int] = [y, counts]
    var ints: List[Int] = [n, k]
    var floats: List[Float64] = [Float64(n), Float64(k)]
    return xt_glue(glue, XT_CLASS_COUNTS, addrs, ints, floats)


def xt_class_priors(glue: EntryRef, counts: Int, priors: Int, n: Int, k: Int) raises -> Int:
    var addrs: List[Int] = [counts, priors]
    var ints: List[Int] = [n, k]
    var floats: List[Float64] = [Float64(n), Float64(k)]
    return xt_glue(glue, XT_CLASS_PRIORS, addrs, ints, floats)


def xt_complement_pairs(glue: EntryRef, x: Int, n: Int) raises -> Int:
    var addrs: List[Int] = [x]
    var ints: List[Int] = [n]
    var floats: List[Float64] = [Float64(n)]
    return xt_glue(glue, XT_COMPLEMENT_PAIRS, addrs, ints, floats)


def xt_count_equal(glue: EntryRef, a: Int, b: Int, n: Int) raises -> Int:
    var addrs: List[Int] = [a, b]
    var ints: List[Int] = [n]
    var floats: List[Float64] = [Float64(n)]
    return xt_glue(glue, XT_COUNT_EQUAL, addrs, ints, floats)


def xt_count_rows(glue: EntryRef, counts: Int, rows: Int, n: Int, m: Int) raises -> Int:
    var addrs: List[Int] = [counts, rows]
    var ints: List[Int] = [n, m]
    var floats: List[Float64] = [Float64(n), Float64(m)]
    return xt_glue(glue, XT_COUNT_ROWS, addrs, ints, floats)


def xt_dart_thresholds(glue: EntryRef, weights: Int, coefs: Int, coef32: Int, thr: Int, state: Int, t: Int, k: Int, uniform_drop: Int, max_drop: Int, rate: Float64) raises -> Int:
    var addrs: List[Int] = [weights, coefs, coef32, thr, state]
    var ints: List[Int] = [t, k, uniform_drop, max_drop, 0]
    var floats: List[Float64] = [Float64(t), Float64(k), Float64(uniform_drop), Float64(max_drop), rate]
    return xt_glue(glue, XT_DART_THRESHOLDS, addrs, ints, floats)


def xt_dart_rescale(glue: EntryRef, weights: Int, coefs: Int, flags: Int, state: Int, res3: Int, t: Int, k: Int, uniform_drop: Int, xgb: Int, lr: Float64) raises -> Int:
    var addrs: List[Int] = [weights, coefs, flags, state, res3]
    var ints: List[Int] = [t, k, uniform_drop, xgb, 0]
    var floats: List[Float64] = [Float64(t), Float64(k), Float64(uniform_drop), Float64(xgb), lr]
    return xt_glue(glue, XT_DART_RESCALE, addrs, ints, floats)


def xt_device_folds(glue: EntryRef, codes: Int, rows: Int, counts: Int, n: Int, n_splits: Int, n_classes: Int) raises -> Int:
    var addrs: List[Int] = [codes, rows, counts]
    var ints: List[Int] = [n, n_splits, n_classes]
    var floats: List[Float64] = [Float64(n), Float64(n_splits), Float64(n_classes)]
    return xt_glue(glue, XT_DEVICE_FOLDS, addrs, ints, floats)


def xt_exact_sum(glue: EntryRef, x: Int, res: Int, flags: Int, n: Int, want_div: Int) raises -> Int:
    var addrs: List[Int] = [x, res, flags]
    var ints: List[Int] = [n, want_div]
    var floats: List[Float64] = [Float64(n), Float64(want_div)]
    return xt_glue(glue, XT_EXACT_SUM, addrs, ints, floats)


def xt_fill_class_major_f64(glue: EntryRef, inits: Int, dst: Int, n: Int, k: Int) raises -> Int:
    var addrs: List[Int] = [inits, dst]
    var ints: List[Int] = [n, k]
    var floats: List[Float64] = [Float64(n), Float64(k)]
    return xt_glue(glue, XT_FILL_CLASS_MAJOR_F64, addrs, ints, floats)


def xt_gather_f32(glue: EntryRef, src: Int, rows: Int, cols: Int, dst: Int, n_src_rows: Int, n_src_cols: Int, n_rows: Int, n_cols: Int) raises -> Int:
    var addrs: List[Int] = [src, rows, cols, dst]
    var ints: List[Int] = [n_src_rows, n_src_cols, n_rows, n_cols]
    var floats: List[Float64] = [Float64(n_src_rows), Float64(n_src_cols), Float64(n_rows), Float64(n_cols)]
    return xt_glue(glue, XT_GATHER_F32, addrs, ints, floats)


def xt_gather_i32(glue: EntryRef, src: Int, rows: Int, dst: Int, n_src: Int, n_rows: Int) raises -> Int:
    var addrs: List[Int] = [src, rows, dst]
    var ints: List[Int] = [n_src, n_rows]
    var floats: List[Float64] = [Float64(n_src), Float64(n_rows)]
    return xt_glue(glue, XT_GATHER_I32, addrs, ints, floats)


def xt_indicator_codes(glue: EntryRef, codes: Int, out_i: Int, out_f: Int, n: Int, cls: Int) raises -> Int:
    var addrs: List[Int] = [codes, out_i, out_f]
    var ints: List[Int] = [n, cls]
    var floats: List[Float64] = [Float64(n), Float64(cls)]
    return xt_glue(glue, XT_INDICATOR_CODES, addrs, ints, floats)


def xt_iota_i32(glue: EntryRef, dst: Int, n: Int) raises -> Int:
    var addrs: List[Int] = [dst]
    var ints: List[Int] = [n]
    var floats: List[Float64] = [Float64(n)]
    return xt_glue(glue, XT_IOTA_I32, addrs, ints, floats)


def xt_isotonic_fit(glue: EntryRef, x: Int, y: Int, kx: Int, ky: Int, n: Int) raises -> Int:
    var addrs: List[Int] = [x, y, kx, ky]
    var ints: List[Int] = [n]
    var floats: List[Float64] = [Float64(n)]
    return xt_glue(glue, XT_ISOTONIC_FIT, addrs, ints, floats)


def xt_isotonic_predict_strided(glue: EntryRef, kx: Int, ky: Int, t: Int, res: Int, m: Int, n: Int, t_len: Int, t_stride: Int, t_off: Int, res_len: Int, res_stride: Int, res_off: Int) raises -> Int:
    var addrs: List[Int] = [kx, ky, t, res]
    var ints: List[Int] = [m, n, t_len, t_stride, t_off, res_len, res_stride, res_off]
    var floats: List[Float64] = [Float64(m), Float64(n), Float64(t_len), Float64(t_stride), Float64(t_off), Float64(res_len), Float64(res_stride), Float64(res_off)]
    return xt_glue(glue, XT_ISOTONIC_PREDICT_STRIDED, addrs, ints, floats)


def xt_leaf_numbering(glue: EntryRef, left: Int, node_col: Int, n_nodes: Int) raises -> Int:
    var addrs: List[Int] = [left, node_col]
    var ints: List[Int] = [n_nodes]
    var floats: List[Float64] = [Float64(n_nodes)]
    return xt_glue(glue, XT_LEAF_NUMBERING, addrs, ints, floats)


def xt_log64(glue: EntryRef, res: Int, x: Float64) raises -> Int:
    var addrs: List[Int] = [res]
    var ints: List[Int] = [0]
    var floats: List[Float64] = [x]
    return xt_glue(glue, XT_LOG64, addrs, ints, floats)


def xt_margin2(glue: EntryRef, acc: Int, dst: Int, n: Int, mode: Int) raises -> Int:
    var addrs: List[Int] = [acc, dst]
    var ints: List[Int] = [n, mode]
    var floats: List[Float64] = [Float64(n), Float64(mode)]
    return xt_glue(glue, XT_MARGIN2, addrs, ints, floats)


def xt_normalize_rows(glue: EntryRef, x: Int, n: Int, k: Int) raises -> Int:
    var addrs: List[Int] = [x]
    var ints: List[Int] = [n, k]
    var floats: List[Float64] = [Float64(n), Float64(k)]
    return xt_glue(glue, XT_NORMALIZE_ROWS, addrs, ints, floats)


def xt_onehot_leaves(glue: EntryRef, nodes: Int, tree_base: Int, node_col: Int, res: Int, n: Int, n_trees: Int, n_cols: Int) raises -> Int:
    var addrs: List[Int] = [nodes, tree_base, node_col, res]
    var ints: List[Int] = [n, n_trees, n_cols]
    var floats: List[Float64] = [Float64(n), Float64(n_trees), Float64(n_cols)]
    return xt_glue(glue, XT_ONEHOT_LEAVES, addrs, ints, floats)


def xt_oob_r2(glue: EntryRef, acc: Int, counts: Int, y: Int, pred: Int, words: Int, flags: Int, n: Int) raises -> Int:
    var addrs: List[Int] = [acc, counts, y, pred, words, flags]
    var ints: List[Int] = [n]
    var floats: List[Float64] = [Float64(n)]
    return xt_glue(glue, XT_OOB_R2, addrs, ints, floats)


def xt_platt_apply_strided(glue: EntryRef, f: Int, res: Int, n: Int, a: Float64, b: Float64, f_len: Int, f_stride: Int, f_off: Int, res_len: Int, res_stride: Int, res_off: Int) raises -> Int:
    var addrs: List[Int] = [f, res]
    var ints: List[Int] = [n, 0, 0, f_len, f_stride, f_off, res_len, res_stride, res_off]
    var floats: List[Float64] = [Float64(n), a, b, Float64(f_len), Float64(f_stride), Float64(f_off), Float64(res_len), Float64(res_stride), Float64(res_off)]
    return xt_glue(glue, XT_PLATT_APPLY_STRIDED, addrs, ints, floats)


def xt_platt_fit(glue: EntryRef, f: Int, y: Int, ab: Int, n: Int) raises -> Int:
    var addrs: List[Int] = [f, y, ab]
    var ints: List[Int] = [n]
    var floats: List[Float64] = [Float64(n)]
    return xt_glue(glue, XT_PLATT_FIT, addrs, ints, floats)


def xt_positive_codes(glue: EntryRef, x: Int, codes: Int, n: Int) raises -> Int:
    var addrs: List[Int] = [x, codes]
    var ints: List[Int] = [n]
    var floats: List[Float64] = [Float64(n)]
    return xt_glue(glue, XT_POSITIVE_CODES, addrs, ints, floats)


def xt_put_f32(glue: EntryRef, dst: Int, src: Int, offset: Int, n: Int) raises -> Int:
    var addrs: List[Int] = [dst, src]
    var ints: List[Int] = [offset, n]
    var floats: List[Float64] = [Float64(offset), Float64(n)]
    return xt_glue(glue, XT_PUT_F32, addrs, ints, floats)


def xt_r2_step(glue: EntryRef, w: Int, pred: Int, y: Int, stats: Int, n: Int, loss: Int, learning_rate: Float64, last: Int, tree_base: Int) raises -> Int:
    var addrs: List[Int] = [w, pred, y, stats]
    var ints: List[Int] = [n, loss, 0, last, tree_base]
    var floats: List[Float64] = [Float64(n), Float64(loss), learning_rate, Float64(last), Float64(tree_base)]
    return xt_glue(glue, XT_R2_STEP, addrs, ints, floats)


def xt_remap_cols(glue: EntryRef, colid: Int, cols: Int, n_nodes: Int, m: Int) raises -> Int:
    var addrs: List[Int] = [colid, cols]
    var ints: List[Int] = [n_nodes, m]
    var floats: List[Float64] = [Float64(n_nodes), Float64(m)]
    return xt_glue(glue, XT_REMAP_COLS, addrs, ints, floats)


def xt_rows_any_unset(glue: EntryRef, eq: Int, out_i: Int, n: Int, m: Int, col_major: Int) raises -> Int:
    var addrs: List[Int] = [eq, out_i]
    var ints: List[Int] = [n, m, col_major]
    var floats: List[Float64] = [Float64(n), Float64(m), Float64(col_major)]
    return xt_glue(glue, XT_ROWS_ANY_UNSET, addrs, ints, floats)


def xt_samme_step(glue: EntryRef, w: Int, pred: Int, y: Int, stats: Int, n: Int, n_classes: Int, learning_rate: Float64, last: Int, tree_base: Int) raises -> Int:
    var addrs: List[Int] = [w, pred, y, stats]
    var ints: List[Int] = [n, n_classes, 0, last, tree_base]
    var floats: List[Float64] = [Float64(n), Float64(n_classes), learning_rate, Float64(last), Float64(tree_base)]
    return xt_glue(glue, XT_SAMME_STEP, addrs, ints, floats)


def xt_sample_indices(glue: EntryRef, dst: Int, n_pool: Int, n_draw: Int, replace: Int, seed: Int, stream: Int) raises -> Int:
    var addrs: List[Int] = [dst]
    var ints: List[Int] = [n_pool, n_draw, replace, seed, stream]
    var floats: List[Float64] = [Float64(n_pool), Float64(n_draw), Float64(replace), Float64(seed), Float64(stream)]
    return xt_glue(glue, XT_SAMPLE_INDICES, addrs, ints, floats)


def xt_scale(glue: EntryRef, x: Int, n: Int, divisor: Float64) raises -> Int:
    var addrs: List[Int] = [x]
    var ints: List[Int] = [n, 0]
    var floats: List[Float64] = [Float64(n), divisor]
    return xt_glue(glue, XT_SCALE, addrs, ints, floats)


def xt_scale_to_f32(glue: EntryRef, x: Int, res: Int, n: Int, factor: Float64) raises -> Int:
    var addrs: List[Int] = [x, res]
    var ints: List[Int] = [n, 0]
    var floats: List[Float64] = [Float64(n), factor]
    return xt_glue(glue, XT_SCALE_TO_F32, addrs, ints, floats)


def xt_scatter(glue: EntryRef, dst: Int, src: Int, rows: Int, n_dst_rows: Int, n_dst_cols: Int, m: Int, c: Int, col0: Int) raises -> Int:
    var addrs: List[Int] = [dst, src, rows]
    var ints: List[Int] = [n_dst_rows, n_dst_cols, m, c, col0]
    var floats: List[Float64] = [Float64(n_dst_rows), Float64(n_dst_cols), Float64(m), Float64(c), Float64(col0)]
    return xt_glue(glue, XT_SCATTER, addrs, ints, floats)


def xt_softmax_rows(glue: EntryRef, x: Int, n: Int, k: Int) raises -> Int:
    var addrs: List[Int] = [x]
    var ints: List[Int] = [n, k]
    var floats: List[Float64] = [Float64(n), Float64(k)]
    return xt_glue(glue, XT_SOFTMAX_ROWS, addrs, ints, floats)


def xt_transpose_f32(glue: EntryRef, src: Int, dst: Int, n: Int, d: Int) raises -> Int:
    var addrs: List[Int] = [src, dst]
    var ints: List[Int] = [n, d]
    var floats: List[Float64] = [Float64(n), Float64(d)]
    return xt_glue(glue, XT_TRANSPOSE_F32, addrs, ints, floats)


def xt_transpose_f64(glue: EntryRef, src: Int, dst: Int, n: Int, d: Int) raises -> Int:
    var addrs: List[Int] = [src, dst]
    var ints: List[Int] = [n, d]
    var floats: List[Float64] = [Float64(n), Float64(d)]
    return xt_glue(glue, XT_TRANSPOSE_F64, addrs, ints, floats)


def xt_uniform(glue: EntryRef, res: Int, n: Int, seed: Int, stream: Int) raises -> Int:
    var addrs: List[Int] = [res]
    var ints: List[Int] = [n, seed, stream]
    var floats: List[Float64] = [Float64(n), Float64(seed), Float64(stream)]
    return xt_glue(glue, XT_UNIFORM, addrs, ints, floats)


def xt_unseen_rows(glue: EntryRef, rows: Int, res: Int, m: Int, n: Int) raises -> Int:
    var addrs: List[Int] = [rows, res]
    var ints: List[Int] = [m, n]
    var floats: List[Float64] = [Float64(m), Float64(n)]
    return xt_glue(glue, XT_UNSEEN_ROWS, addrs, ints, floats)


def xt_weighted_median(glue: EntryRef, preds: Int, weights: Int, res: Int, n: Int, m: Int) raises -> Int:
    var addrs: List[Int] = [preds, weights, res]
    var ints: List[Int] = [n, m]
    var floats: List[Float64] = [Float64(n), Float64(m)]
    return xt_glue(glue, XT_WEIGHTED_MEDIAN, addrs, ints, floats)


def xt_weighted_sample(glue: EntryRef, w: Int, dst: Int, n: Int, n_draw: Int, seed: Int, stream: Int) raises -> Int:
    var addrs: List[Int] = [w, dst]
    var ints: List[Int] = [n, n_draw, seed, stream]
    var floats: List[Float64] = [Float64(n), Float64(n_draw), Float64(seed), Float64(stream)]
    return xt_glue(glue, XT_WEIGHTED_SAMPLE, addrs, ints, floats)

def xt_stack_w64(glue: EntryRef, addrs: List[Int], n: Int) raises -> Int:
    """addrs = [dst, column 0, ..., column m-1] (see stack_w64_run)."""
    var ints: List[Int] = [n]
    var floats: List[Float64] = [Float64(n)]
    return xt_glue(glue, XT_STACK_W64, addrs, ints, floats)


def xt_accumulate_cols_many(glue: EntryRef, addrs: List[Int], ints: List[Int], floats: List[Float64]) raises -> Int:
    """addrs = [acc, x_0, cols_0, ...]; params = [n, k, ks_0, weight_0, ...]
    as int/float views (see accumulate_cols_many_run)."""
    return xt_glue(glue, XT_ACCUMULATE_COLS_MANY, addrs, ints, floats)


def xt_ovr_assemble(glue: EntryRef, addrs: List[Int], ints: List[Int], floats: List[Float64]) raises -> Int:
    """addrs = [out, blk_0, ...]; params = [n, k, finish, bc_0, at_0, ...]
    (see ovr_assemble_run)."""
    return xt_glue(glue, XT_OVR_ASSEMBLE, addrs, ints, floats)


def xt_calibrated_proba(glue: EntryRef, addrs: List[Int], ints: List[Int], floats: List[Float64]) raises -> Int:
    """The flattened pairs and calibrators of calibrated_proba_run."""
    return xt_glue(glue, XT_CALIBRATED_PROBA, addrs, ints, floats)
