# SPDX-License-Identifier: Apache-2.0
"""The trees entry table's x_trees glue entry, reached natively.

Ordered preconditions first (they share the one process table): a caller
refuses before publication, then the same publication both x_trees bindings
make at module init (`publish_x_trees_glue`), a key and a schema mismatch
refuse, and an unknown operation refuses through the entry's sink. Then one
case per published glue operation, each through `xtrees/glue_calls.mojo` and
the published pointer, with literal expected values or the run's own refusal
text (the text its former Python entry raised). No Python participates.

The cases are independent units: on the host column they run across the
native host worker pool; on a device column they run one after another,
because every case launches on the process's one device context.
Run in a fresh process (the first check needs an unpublished table).
"""
from checks.kernel_matrix import COLUMN_CPU, TARGET_COLUMN
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.host_parallel import host_parallelize
from core.trees_entry_table import TreesEntryKey, trees_table_entry, TREES_TABLE_X_TREES
from xtrees.glue_abi import XT_OP_COUNT
from xtrees.glue_runs import publish_x_trees_glue
from xtrees.glue_calls import (
    xt_glue, xt_accumulate, xt_accumulate_cols, xt_accumulate_onehot, xt_accumulate_rows, xt_apply, xt_argmax_rows,
    xt_bag_rows, xt_binary_proba, xt_class_counts, xt_class_priors, xt_complement_pairs, xt_count_equal,
    xt_count_rows, xt_dart_rescale, xt_dart_thresholds, xt_device_folds, xt_exact_sum, xt_fill_class_major_f64,
    xt_gather_f32, xt_gather_i32, xt_indicator_codes, xt_iota_i32, xt_isotonic_fit, xt_isotonic_predict_strided,
    xt_leaf_numbering, xt_log64, xt_margin2, xt_normalize_rows, xt_onehot_leaves, xt_oob_r2,
    xt_platt_apply_strided, xt_platt_fit, xt_positive_codes, xt_put_f32, xt_r2_step, xt_remap_cols,
    xt_rows_any_unset, xt_samme_step, xt_sample_indices, xt_scale, xt_scale_to_f32, xt_scatter, xt_softmax_rows,
    xt_stack_w64, xt_transpose_f32, xt_transpose_f64, xt_uniform, xt_unseen_rows, xt_weighted_median,
    xt_weighted_sample,
)

comptime TREES_GLUE_CASES = 50


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _key() -> TreesEntryKey:
    return TreesEntryKey(Int(GLOBAL_NUMERIC_MODE), 0)


def _f64(mut l: List[Float64]) -> Int:
    return Int(l.unsafe_ptr())


def _f32(mut l: List[Float32]) -> Int:
    return Int(l.unsafe_ptr())


def _i32(mut l: List[Int32]) -> Int:
    return Int(l.unsafe_ptr())


def _u8(mut l: List[UInt8]) -> Int:
    return Int(l.unsafe_ptr())


def _i64(mut l: List[Int64]) -> Int:
    return Int(l.unsafe_ptr())


def _refuses(problem: Error, text: String) raises:
    require(text in String(problem), "unexpected refusal: " + String(problem))


def precondition_unpublished() raises:
    var out: List[Int32] = [0]
    try:
        _ = xt_iota_i32(_key(), _i32(out), 1)
    except problem:
        _refuses(problem, "provider is not published")
        _ = out
        return
    raise Error("an unpublished trees table did not refuse")


def preconditions_published() raises:
    var out: List[Int32] = [0]
    try:
        _ = xt_iota_i32(TreesEntryKey(1 - Int(GLOBAL_NUMERIC_MODE), 0), _i32(out), 1)
        raise Error("a mismatched tier key did not refuse")
    except problem:
        require("did not refuse" not in String(problem), String(problem))
    try:
        _ = trees_table_entry[TREES_TABLE_X_TREES](_key(), 0, String("mojolearn.trees.other.v1"), XT_OP_COUNT)
        raise Error("a mismatched schema did not refuse")
    except problem:
        _refuses(problem, "ABI schema differs")
    try:
        var none = List[Int]()
        _ = xt_glue(_key(), XT_OP_COUNT, none, none, List[Float64]())
        raise Error("an unknown operation did not refuse")
    except problem:
        _refuses(problem, "unknown operation")
    _ = out


def case(unit: Int) raises:
    var k = _key()
    if unit == 0:
        var out: List[Int32] = [9, 9, 9, 9]
        require(xt_iota_i32(k, _i32(out), 4) == 4 and out[0] == 0 and out[3] == 3, "iota_i32")
        _ = out
    elif unit == 1:
        var acc: List[Float64] = [1.0, 2.0]
        var x: List[Float32] = [1.0, 1.0]
        _ = xt_accumulate(k, _f64(acc), _f32(x), 2, 0.5)
        require(acc[0] == 1.5 and acc[1] == 2.5, "accumulate")
        _ = x
    elif unit == 2:
        var acc: List[Float64] = [0.0, 0.0, 0.0]
        var x: List[Float32] = [2.0, 3.0]
        var cols: List[Int32] = [2, 0]
        _ = xt_accumulate_cols(k, _f64(acc), _f32(x), _i32(cols), 1, 3, 2, 1.0)
        require(acc[0] == 3.0 and acc[1] == 0.0 and acc[2] == 2.0, "accumulate_cols")
        _ = x
        _ = cols
    elif unit == 3:
        var acc: List[Float64] = [0.0, 0.0, 0.0, 0.0]
        var codes: List[Int32] = [1, 0]
        _ = xt_accumulate_onehot(k, _f64(acc), _i32(codes), 2, 2, 1.0, 0.0)
        require(acc[0] == 0.0 and acc[1] == 1.0 and acc[2] == 1.0 and acc[3] == 0.0, "accumulate_onehot")
        _ = codes
    elif unit == 4:
        var acc: List[Float64] = [0.0, 0.0, 0.0]
        var x: List[Float64] = [5.0, 7.0]
        var rows: List[Int32] = [2, 0]
        _ = xt_accumulate_rows(k, _f64(acc), _f64(x), _i32(rows), 3, 1, 2)
        require(acc[0] == 7.0 and acc[1] == 0.0 and acc[2] == 5.0, "accumulate_rows")
        _ = x
        _ = rows
    elif unit == 5:
        var x: List[Float64] = [0.0, 2.0, 1.0, 3.0, 1.0, 0.0]
        var res: List[Int32] = [9, 9]
        _ = xt_argmax_rows(k, _f64(x), _i32(res), 2, 3)
        require(res[0] == 1 and res[1] == 0, "argmax_rows")
        _ = x
    elif unit == 6:
        var x: List[Float64] = [2.0, 4.0]
        _ = xt_scale(k, _f64(x), 2, 2.0)
        require(x[0] == 1.0 and x[1] == 2.0, "scale")
    elif unit == 7:
        var x: List[Float64] = [0.5, 0.25]
        var res: List[Float32] = [0.0, 0.0]
        _ = xt_scale_to_f32(k, _f64(x), _f32(res), 2, 4.0)
        require(res[0] == 2.0 and res[1] == 1.0, "scale_to_f32")
        _ = x
    elif unit == 8:
        var dst: List[Float32] = [0.0, 0.0, 0.0, 0.0]
        var src: List[Float32] = [7.0, 8.0]
        _ = xt_put_f32(k, _f32(dst), _f32(src), 1, 2)
        require(dst[0] == 0.0 and dst[1] == 7.0 and dst[2] == 8.0 and dst[3] == 0.0, "put_f32")
        _ = src
    elif unit == 9:
        var x: List[Float64] = [0.0, 0.0]
        _ = xt_softmax_rows(k, _f64(x), 1, 2)
        require(x[0] == 0.5 and x[1] == 0.5, "softmax_rows")
    elif unit == 10:
        var x: List[Float64] = [1.0, 3.0]
        _ = xt_normalize_rows(k, _f64(x), 1, 2)
        require(x[0] == 0.25 and x[1] == 0.75, "normalize_rows")
    elif unit == 11:
        var src: List[Float32] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        var dst: List[Float32] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        require(xt_transpose_f32(k, _f32(src), _f32(dst), 2, 3) == 6, "transpose_f32 count")
        require(dst[0] == 1.0 and dst[1] == 4.0 and dst[2] == 2.0 and dst[5] == 6.0, "transpose_f32")
        _ = src
    elif unit == 12:
        var src: List[Float64] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        var dst: List[Float64] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        _ = xt_transpose_f64(k, _f64(src), _f64(dst), 2, 3)
        require(dst[0] == 1.0 and dst[1] == 4.0 and dst[2] == 2.0 and dst[5] == 6.0, "transpose_f64")
        _ = src
    elif unit == 13:
        var src: List[Float32] = [1.0, 2.0, 3.0, 4.0]
        var rows: List[Int32] = [1]
        var cols: List[Int32] = [0, 1]
        var dst: List[Float32] = [0.0, 0.0]
        _ = xt_gather_f32(k, _f32(src), _i32(rows), _i32(cols), _f32(dst), 2, 2, 1, 2)
        require(dst[0] == 3.0 and dst[1] == 4.0, "gather_f32")
        _ = src
        _ = rows
        _ = cols
    elif unit == 14:
        var src: List[Int32] = [5, 6, 7]
        var rows: List[Int32] = [2, 0]
        var dst: List[Int32] = [0, 0]
        _ = xt_gather_i32(k, _i32(src), _i32(rows), _i32(dst), 3, 2)
        require(dst[0] == 7 and dst[1] == 5, "gather_i32")
        _ = src
        _ = rows
    elif unit == 15:
        var x: List[Float64] = [0.0, 0.25]
        _ = xt_complement_pairs(k, _f64(x), 1)
        require(x[0] == 0.75 and x[1] == 0.25, "complement_pairs")
    elif unit == 16:
        var p: List[Float32] = [0.25]
        var res: List[Float64] = [0.0, 0.0]
        _ = xt_binary_proba(k, _f32(p), _f64(res), 1)
        require(res[0] == 0.75 and res[1] == 0.25, "binary_proba")
        _ = p
    elif unit == 17:
        var x: List[Float64] = [-1.0, 2.0, 0.0]
        var codes: List[Int32] = [9, 9, 9]
        _ = xt_positive_codes(k, _f64(x), _i32(codes), 3)
        require(codes[0] == 0 and codes[1] == 1 and codes[2] == 0, "positive_codes")
        _ = x
    elif unit == 18:
        var a: List[Int32] = [1, 2, 3]
        var b: List[Int32] = [1, 0, 3]
        require(xt_count_equal(k, _i32(a), _i32(b), 3) == 2, "count_equal")
        _ = a
        _ = b
    elif unit == 19:
        var counts: List[Int32] = [0, 0, 0]
        var rows: List[Int32] = [2, 2, 0]
        require(xt_count_rows(k, _i32(counts), _i32(rows), 3, 3) == 3, "count_rows count")
        require(counts[0] == 1 and counts[1] == 0 and counts[2] == 2, "count_rows")
        _ = rows
    elif unit == 20:
        var codes: List[Int32] = [0, 2, 1, 2]
        var out_i: List[Int32] = [9, 9, 9, 9]
        var out_f: List[Float64] = [9.0, 9.0, 9.0, 9.0]
        _ = xt_indicator_codes(k, _i32(codes), _i32(out_i), _f64(out_f), 4, 2)
        require(out_i[0] == 0 and out_i[1] == 1 and out_i[2] == 0 and out_i[3] == 1, "indicator_codes int")
        require(out_f[0] == 0.0 and out_f[1] == 1.0 and out_f[3] == 1.0, "indicator_codes float")
        _ = codes
    elif unit == 21:
        var dst: List[Float64] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        var src: List[Float32] = [5.0, 6.0]
        var rows: List[Int32] = [1]
        require(xt_scatter(k, _f64(dst), _f32(src), _i32(rows), 2, 3, 1, 2, 1) == 2, "scatter count")
        require(dst[3] == 0.0 and dst[4] == 5.0 and dst[5] == 6.0 and dst[1] == 0.0, "scatter")
        _ = src
        _ = rows
    elif unit == 22:
        var a: List[Float64] = [1.0, 2.0]
        var b: List[Float64] = [3.0, 4.0]
        var dst: List[Float64] = [0.0, 0.0, 0.0, 0.0]
        require(xt_stack_w64(k, [_f64(dst), _f64(a), _f64(b)], 2) == 4, "stack_w64 count")
        require(dst[0] == 1.0 and dst[1] == 3.0 and dst[2] == 2.0 and dst[3] == 4.0, "stack_w64")
        _ = a
        _ = b
    elif unit == 23:
        var acc: List[Float64] = [1.0, 3.0, 4.0, 2.0]
        var d: List[Float64] = [0.0, 0.0]
        _ = xt_margin2(k, _f64(acc), _f64(d), 2, 0)
        require(d[0] == 2.0 and d[1] == -2.0, "margin2 values")
        var codes: List[Int32] = [9, 9]
        _ = xt_margin2(k, _f64(acc), _i32(codes), 2, 1)
        require(codes[0] == 1 and codes[1] == 0, "margin2 codes")
        _ = acc
    elif unit == 24:
        var rows: List[Int32] = [0, 2]
        var res: List[Int32] = [9, 9, 9, 9]
        require(xt_unseen_rows(k, _i32(rows), _i32(res), 2, 4) == 2, "unseen_rows count")
        require(res[0] == 1 and res[1] == 3, "unseen_rows")
        _ = rows
    elif unit == 25:
        var colid: List[Int32] = [1, 0]
        var cols: List[Int32] = [5, 7]
        _ = xt_remap_cols(k, _i32(colid), _i32(cols), 2, 2)
        require(colid[0] == 7 and colid[1] == 5, "remap_cols")
        _ = cols
    elif unit == 26:
        var eq: List[UInt8] = [1, 1, 1, 0]
        var out_i: List[Int32] = [9, 9]
        _ = xt_rows_any_unset(k, _u8(eq), _i32(out_i), 2, 2, 0)
        require(out_i[0] == 0 and out_i[1] == 1, "rows_any_unset")
        _ = eq
    elif unit == 27:
        var y: List[Float32] = [0.0, 1.0, 1.0]
        var counts: List[Int32] = [0, 0]
        _ = xt_class_counts(k, _f32(y), _i32(counts), 3, 2)
        require(counts[0] == 1 and counts[1] == 2, "class_counts")
        _ = y
    elif unit == 28:
        var counts: List[Int32] = [3]
        var priors: List[Float64] = [9.0]
        _ = xt_class_priors(k, _i32(counts), _f64(priors), 3, 1)
        require(priors[0] == 0.0, "class_priors")
        _ = counts
    elif unit == 29:
        var res: List[Float64] = [9.0]
        _ = xt_log64(k, _f64(res), 1.0)
        require(res[0] == 0.0, "log64")
    elif unit == 30:
        var x: List[Float32] = [1.0, 2.0, 3.0]
        var res: List[Float64] = [0.0]
        var flags: List[Int32] = [0, 0, 0, 0, 0, 0, 0, 0]
        _ = xt_exact_sum(k, _f32(x), _f64(res), _i32(flags), 3, 0)
        require(res[0] == 6.0 and flags[0] == 0 and flags[3] == 0, "exact_sum")
        _ = x
    elif unit == 31:
        var preds: List[Float32] = [3.0, 4.0]
        var weights: List[Float64] = [1.0]
        var res: List[Float32] = [0.0, 0.0]
        _ = xt_weighted_median(k, _f32(preds), _f64(weights), _f32(res), 2, 1)
        require(res[0] == 3.0 and res[1] == 4.0, "weighted_median")
        _ = preds
        _ = weights
    elif unit == 32:
        var inits: List[Float64] = [1.0, 2.0]
        var dst: List[Float64] = [0.0, 0.0, 0.0, 0.0]
        _ = xt_fill_class_major_f64(k, _f64(inits), _f64(dst), 2, 2)
        require(dst[0] == 1.0 and dst[1] == 1.0 and dst[2] == 2.0 and dst[3] == 2.0, "fill_class_major_f64")
        _ = inits
    elif unit == 33:
        var dst: List[Int32] = [-1, -1, -1, -1]
        _ = xt_sample_indices(k, _i32(dst), 5, 4, 1, 7, 0)
        for i in range(4):  # small-loop(four drawn indices)
            require(dst[i] >= 0 and dst[i] < 5, "sample_indices range")
    elif unit == 34:
        var res: List[Float64] = [-1.0, -1.0, -1.0, -1.0]
        _ = xt_uniform(k, _f64(res), 4, 7, 0)
        for i in range(4):  # small-loop(four draws)
            require(res[i] >= 0.0 and res[i] < 1.0, "uniform range")
    elif unit == 35:
        var w: List[Float64] = [0.0, 1.0, 0.0]
        var dst: List[Int32] = [-1, -1, -1, -1]
        _ = xt_weighted_sample(k, _f64(w), _i32(dst), 3, 4, 7, 0)
        for i in range(4):  # small-loop(four weighted draws)
            require(dst[i] == 1, "weighted_sample takes the only weighted row")
        _ = w
    elif unit == 36:
        var res: List[Int32] = [-1, -1, -1, -1]
        require(xt_bag_rows(k, _i32(res), 4, 7, 0, 2.0) == 4, "bag_rows count")
        require(res[0] == 0 and res[3] == 3, "bag_rows ascending")
    elif unit == 37:
        var codes: List[Int32] = [0, 0, 0, 0]
        var rows: List[Int32] = [-1, -1, -1, -1, -1, -1, -1, -1]
        var counts: List[Int32] = [0, 0, 9]
        require(xt_device_folds(k, _i32(codes), _i32(rows), _i32(counts), 4, 2, 0) == 0, "device_folds status")
        require(counts[0] + counts[1] == 4, "device_folds sizes")
        _ = codes
    elif unit == 38:
        var x: List[Float64] = [1.0, 2.0, 3.0]
        var y: List[Float64] = [1.0, 2.0, 3.0]
        var kx: List[Float64] = [0.0, 0.0, 0.0]
        var ky: List[Float64] = [0.0, 0.0, 0.0]
        var m = xt_isotonic_fit(k, _f64(x), _f64(y), _f64(kx), _f64(ky), 3)
        require(m >= 1 and m <= 3 and kx[0] == 1.0 and ky[0] == 1.0, "isotonic_fit")
        _ = x
        _ = y
    elif unit == 39:
        var kx: List[Float64] = [0.0, 1.0]
        var ky: List[Float64] = [0.0, 1.0]
        var t: List[Float64] = [0.5]
        var res: List[Float64] = [9.0]
        _ = xt_isotonic_predict_strided(k, _f64(kx), _f64(ky), _f64(t), _f64(res), 2, 1, 1, 1, 0, 1, 1, 0)
        require(res[0] == 0.5, "isotonic_predict_strided")
        _ = kx
        _ = ky
        _ = t
    elif unit == 40:
        var f: List[Float64] = [-1.0, 1.0, -2.0, 2.0]
        var y: List[Int32] = [0, 1, 0, 1]
        var ab: List[Float64] = [0.0, 0.0]
        _ = xt_platt_fit(k, _f64(f), _i32(y), _f64(ab), 4)
        require(ab[0] == ab[0] and ab[1] == ab[1] and ab[0] < 0.0, "platt_fit")
        _ = f
        _ = y
    elif unit == 41:
        var f: List[Float64] = [0.0]
        var res: List[Float64] = [9.0]
        _ = xt_platt_apply_strided(k, _f64(f), _f64(res), 1, 0.0, 0.0, 1, 1, 0, 1, 1, 0)
        require(res[0] == 0.5, "platt_apply_strided")
        _ = f
    elif unit == 42:
        var left: List[Int32] = [0]
        var node_col: List[Int32] = [0]
        require(xt_leaf_numbering(k, _i32(left), _i32(node_col), 0) == 0, "leaf_numbering empty")
        _ = left
        _ = node_col
    elif unit == 43:
        var offsets: List[Int32] = [0, 1]
        var colid: List[Int32] = [-1]
        var quesval: List[Float32] = [0.0]
        var left: List[Int32] = [-1]
        var x: List[Float32] = [0.0]
        var res: List[Int32] = [0]
        try:
            _ = xt_apply(k, _i32(offsets), _i32(colid), _f32(quesval), _i32(left), _f32(x), _i32(res), 0, 1, 1, 1)
            raise Error("apply did not refuse t1 <= t0")
        except problem:
            _refuses(problem, "x_trees_apply: need t1 > t0")
        _ = offsets
        _ = colid
        _ = quesval
        _ = left
        _ = x
        _ = res
    elif unit == 44:
        var nodes: List[Int32] = [0]
        var base: List[Int32] = [0]
        var node_col: List[Int32] = [0]
        var res: List[Float64] = [0.0]
        require(xt_onehot_leaves(k, _i32(nodes), _i32(base), _i32(node_col), _f64(res), 0, 1, 1) == 0,
                "onehot_leaves empty")
        _ = nodes
        _ = base
        _ = node_col
        _ = res
    elif unit == 45:
        var w: List[Float64] = [1.0]
        var pred: List[Int32] = [0]
        var y: List[Int32] = [0]
        var stats: List[Float64] = [0.0, 0.0, 0.0, 0.0]
        try:
            _ = xt_samme_step(k, _f64(w), _i32(pred), _i32(y), _f64(stats), 0, 2, 1.0, 1, 0)
            raise Error("samme_step did not refuse n = 0")
        except problem:
            _refuses(problem, "x_trees_samme_step: n must be positive")
        _ = w
        _ = pred
        _ = y
        _ = stats
    elif unit == 46:
        var w: List[Float64] = [1.0]
        var pred: List[Float32] = [0.0]
        var y: List[Float32] = [0.0]
        var stats: List[Float64] = [0.0, 0.0, 0.0, 0.0]
        try:
            _ = xt_r2_step(k, _f64(w), _f32(pred), _f32(y), _f64(stats), 0, 0, 1.0, 1, 0)
            raise Error("r2_step did not refuse n = 0")
        except problem:
            _refuses(problem, "x_trees_r2_step: n must be positive")
        _ = w
        _ = pred
        _ = y
        _ = stats
    elif unit == 47:
        var acc: List[Float64] = [2.0, 4.0]
        var counts: List[Int32] = [1, 2]
        var y: List[Float32] = [2.0, 2.0]
        var pred: List[Float64] = [0.0, 0.0]
        var words: List[Float64] = [0.0, 0.0, 0.0, 0.0]
        var flags: List[Int32] = [9, 9, 9, 9]
        require(xt_oob_r2(k, _f64(acc), _i32(counts), _f32(y), _f64(pred), _f64(words), _i32(flags), 2) == 2,
                "oob_r2 count")
        require(pred[0] == 2.0 and pred[1] == 2.0 and words[0] == 4.0 and words[3] == 2.0, "oob_r2")
        _ = acc
        _ = counts
        _ = y
    elif unit == 48:
        var weights: List[Float64] = [0.0]
        var coefs: List[Float64] = [0.0]
        var coef32: List[Float32] = [0.0]
        var thr: List[Int64] = [0]
        var state: List[Float64] = [0.0]
        require(xt_dart_thresholds(k, _f64(weights), _f64(coefs), _f32(coef32), _i64(thr), _f64(state),
                                   0, 1, 0, 0, 0.1) == 0, "dart_thresholds empty")
        _ = weights
        _ = coefs
        _ = coef32
        _ = thr
        _ = state
    else:
        var weights: List[Float64] = [0.0]
        var coefs: List[Float64] = [0.0]
        var flags: List[Int32] = [0]
        var state: List[Float64] = [0.0]
        var res: List[Float64] = [0.0, 0.0, 0.0]
        try:
            _ = xt_dart_rescale(k, _f64(weights), _f64(coefs), _i32(flags), _f64(state), _f64(res), 0, 0, 0, 0, 0.1)
            raise Error("dart_rescale did not refuse k = 0")
        except problem:
            _refuses(problem, "x_trees_dart_rescale: needs k >= 1")
        _ = weights
        _ = coefs
        _ = flags
        _ = state
        _ = res


def run_checks() raises:
    precondition_unpublished()
    # The publication _mojolearn_x_trees_host makes at module init.
    publish_x_trees_glue("cpu")
    preconditions_published()
    var failures = List[Int](length=TREES_GLUE_CASES, fill=0)
    var flags = failures.unsafe_ptr()
    def task(unit: Int) {imm flags}:
        try:
            case(unit)
        except problem:
            flags[unit] = 1
            print("FAIL trees glue entry case", unit, String(problem))
    comptime if TARGET_COLUMN == COLUMN_CPU:
        host_parallelize(task, TREES_GLUE_CASES)
    else:
        for unit in range(TREES_GLUE_CASES):  # driver-loop: device cases share one device context
            task(unit)
    var failed = 0
    for failure in failures:  # collect independent case status words
        failed += failure
    _ = failures
    require(failed == 0, "trees glue entry cases failed: " + String(failed))
    print("PASS trees glue entry", TREES_GLUE_CASES)
