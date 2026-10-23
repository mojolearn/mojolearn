# SPDX-License-Identifier: Apache-2.0
# roc_auc_score and precision_recall_curve of python/mojolearn/_metrics_impl.py,
# one native call each (lane mcb2-mcls).
#
# The binary unweighted full-AUC call and the default precision-recall curve
# keep their cuML-backed entries of the metrics binding (`roc_auc_score`,
# `precision_recall_curve`) with the same encoded labels, scores and
# parameter lists. The options run the x_metrics PROGRAMS of
# _expansion_metrics.py's `roc_auc_options` (the device curve fold
# `_fold_curves` / `_fold_stages`, the one-vs-rest tail `_OvrTail`, the
# micro average's onehot and rep_rows inputs, the one-vs-one programs of
# `_ovo_native`) and `precision_recall_curve_options` (`_curve_out`),
# rebuilt here with the same arena layout, the same stage records in the same
# order and the same declared outputs, run by the shared runner. The
# fold words become the score with the same handful of binary64 operations
# `_FoldCurve.auc` made, every product pinned (checks/numerics.mojo
# `pinned_mul_f64`: Python never fuses). Same kernels: the bits do not move.
from std.memory import bitcast
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_contains
from bindings.classical_arrays import classical_address, classical_dtype_str
from bindings.family_common import (
    FAMILY_F32P, FAMILY_I32P, family_repr, family_str, family_call, family_array, family_as_f32,
    family_shape_repr,
)
from bindings.mcls_common import (
    MC_NONE, MC_INT32_MAX, McLabels, McProg, mc_b, mc_len, mc_bool, mc_eq, mc_ne, mc_in, mc_is_str,
    mc_is_bool, mc_is_real, mc_float, mc_int, mc_sorted, mc_set, mc_nan, mc_warn, mc_undefined, mc_entry,
    mc_shape_of, mc_rank, mc_dim, mc_labels, mc_map_order, mc_map_equal, mc_selected, mc_positive, mc_weights,
    mc_f32_lists, mc_run, mc_arena_int, mc_arena_float, mc_arena_double, mc_arena_f64_array, mc_group,
)
from bindings.mcls_classify import mc_targets, mc_pos_label, mc_binary_sets
from core.family_cast_impl import family_all_finite_f32
from core.mcls_impl import mcls_copy_words
from checks.numerics import pinned_mul_f64

comptime _CF_OUT = 10
comptime _CF_CHUNK = 1024
comptime _OVO_WORDS = 1 << 26
comptime _RANK_ROW_SUM = 5
comptime _RANK_CURVE_SCORE = 6
comptime _RANK_AVERAGE = 7
comptime _RANK_OVO_MASK = 8
comptime _RANK_OVO_PAIR = 9
comptime _CO_PR = 1
comptime _CO_EMPTY = 1
comptime _CO_ZERO_CLASS = 2
comptime _ROC_ONE_CLASS = "Only one class is present in y_true. ROC AUC score is not defined in that case."
comptime _ROWS_MSG = "Target scores need to be probabilities for multiclass roc_auc, i.e. they should sum up to 1.0 over classes"


# ------------------------------------------------------------------ inputs


def _scores(y_score: PythonObject, n: Int, caller: String, ndim: Int, context: PythonObject) raises -> PythonObject:
    """`_expansion_metrics._scores(y_score, n, caller, ndim=ndim)`: finite
    float32 scores, C order (an (n, 1) column keeps its words)."""
    var name = String("y_score")
    var a = mc_f32_lists(y_score, name, context)
    if classical_dtype_str(a, context) != "<f4":
        raise Error("TypeError: mojolearn " + caller + ": " + name + " must have dtype float32; cast explicitly")
    var nd = Int(py=a.ndim)
    var column = nd == 2 and Int(py=a.shape[1]) == 1 and ndim == 1
    if nd != ndim and not column:
        raise Error("mojolearn " + caller + ": " + name + " must be " + String(ndim) + "-D, got shape "
                    + family_shape_repr(a))
    var rows = Int(py=a.shape[0])
    if rows != n:
        raise Error("mojolearn " + caller + ": " + name + " has " + String(rows) + " rows for " + String(n) + " samples")
    var out = family_as_f32(a, nd, name, context)
    if not family_all_finite_f32(FAMILY_F32P(unsafe_from_address=Int(py=classical_address(out))), Int(py=out.size)):
        raise Error("mojolearn " + caller + ": " + name + " must be finite")
    return out


def _binary_scores(y_score: PythonObject, n: Int, context: PythonObject) raises -> PythonObject:
    """`_binary_ranking_inputs`' score checks: 1-D finite float32 scores, n entries."""
    var scores = mc_f32_lists(y_score, "input", context)
    if classical_dtype_str(scores, context) != "<f4":
        raise Error("TypeError: binary ranking scores must have dtype float32")
    if Int(py=scores.ndim) != 1 or Int(py=scores.shape[0]) != n:
        raise Error("binary ranking scores must be one-dimensional with the same length as y_true")
    var out = family_as_f32(scores, 1, "scores", context)
    if not family_all_finite_f32(FAMILY_F32P(unsafe_from_address=Int(py=classical_address(out))), n):
        raise Error("binary ranking scores must be finite")
    if n > MC_INT32_MAX:
        raise Error("binary ranking metrics exceed the native Int32 indexing bound")
    return out


def _binary_labels(y_true: PythonObject, context: PythonObject) raises -> McLabels:
    """`_binary_ranking_inputs`' label checks."""
    var true = mc_labels(y_true, "y_true", context)
    if mc_len(true.classes) > 2:
        raise Error("binary ranking metrics support at most two observed classes")
    return true^


# -------------------------------------------------------- the curve fold


def _f32_bits(x: Float64) -> Int:
    return Int(bitcast[DType.int32](Float32(x)))


def mc_fold_stages(mut prog: McProg, s: Int, stride: Int, pos: Int, w: Int, n: Int, problems: Int,
                   partial: Bool, max_fpr: Float64, out_at: Int) raises -> Int:
    """`_fold_stages(..., auc=True, max_fpr, out)`: the compacted curve
    (bin_curve) and its AUC fold (curve_fold); the fold's CF_OUT words per
    problem, at `out_at` (allocated and declared when MC_NONE)."""
    var total = n * problems
    var order = prog.scratch(total)
    var cnt = prog.scratch(problems)
    var fps = prog.scratch(total)
    var tps = prog.scratch(total)
    var thr = prog.scratch(total)
    var keep = prog.scratch(total)
    var cf = prog.scratch(3 * total)
    var cm = prog.scratch(problems)
    prog.stage("bin_curve", problems, [s, stride, pos, w, n, order, fps, tps, thr, cnt, keep, 1, cf, cm])
    var out = out_at
    if out == MC_NONE:
        out = prog.want(prog.alloc(_CF_OUT * problems), _CF_OUT * problems)
    var mode = 0
    var mh = 0
    var ml = 0
    if partial:
        # `_f32_split(max_fpr)`: the float-float hi + lo as Int32 bit words
        var hi = Float32(max_fpr)
        mode = 2
        mh = Int(bitcast[DType.int32](hi))
        ml = _f32_bits(max_fpr - Float64(hi))
    prog.stage("curve_fold", problems, [n, cf, cf + total, cm, out, mode, mh, ml, _CF_CHUNK])
    return out


def mc_fold_auc(arena: PythonObject, off: Int, partial: Bool, max_fpr: Float64, context: PythonObject) raises -> PythonObject:
    """`_FoldCurve.auc(max_fpr)` from the fold's words."""
    var s = mc_arena_float(arena, off) + mc_arena_float(arena, off + 1)
    var f_tot = mc_arena_float(arena, off + 2)
    var t_tot = mc_arena_float(arena, off + 3)
    var k = mc_arena_int(arena, off + 4)
    var a = mc_arena_float(arena, off + 5)
    var b = mc_arena_float(arena, off + 6)
    var u = mc_arena_float(arena, off + 7)
    var v = mc_arena_float(arena, off + 8)
    var c = mc_arena_int(arena, off + 9)
    if c <= 0 or not (f_tot > 0 and t_tot > 0):
        mc_undefined(_ROC_ONE_CLASS, context)
        return mc_nan()
    var whole = s / pinned_mul_f64(pinned_mul_f64(2.0, f_tot), t_tot)
    if not partial or k >= c:
        return PythonObject(whole)
    var x0 = a / f_tot
    var x1 = b / f_tot
    var y0 = u / t_tot
    var y1 = v / t_tot
    var yi = y0
    if x1 != x0:
        yi = y0 + pinned_mul_f64(max_fpr - x0, y1 - y0) / (x1 - x0)
    var part = whole + pinned_mul_f64(max_fpr - x0, yi + y0) / 2.0
    var min_area = pinned_mul_f64(pinned_mul_f64(0.5, max_fpr), max_fpr)
    return PythonObject(pinned_mul_f64(0.5, 1 + (part - min_area) / (max_fpr - min_area)))


def mc_stage_average(mut prog: McProg, sc: Int, sup: Int, m: Int, weighted: Bool) raises -> Int:
    """`_stage_average`: rank_epi AVERAGE; the declared binary64 result."""
    var res = prog.want(prog.alloc(2), 2)
    prog.stage("rank_epi", 1, [_RANK_AVERAGE, sc, sup, m, 1 if weighted else 0, res])
    return res


# ------------------------------------------------------------ roc_auc_score


def _roc_binary(true: McLabels, args: PythonObject, max_fpr: PythonObject, context: PythonObject) raises -> PythonObject:
    """roc_auc_options' binary arm: one fold program."""
    var present = true.classes
    var average = args[3]
    var averages = Python.tuple(PythonObject(None), PythonObject("micro"), PythonObject("macro"),
                                PythonObject("samples"), PythonObject("weighted"))
    if not mc_in(average, averages):
        raise Error("average has to be one of (None, 'micro', 'macro', 'weighted', 'samples')")
    if mc_len(present) != 2:
        mc_undefined(_ROC_ONE_CLASS, context)
        return mc_nan()
    var n = true.n
    var s = _scores(args[2], n, "roc_auc_score", 1, context)
    var w = mc_weights(args[4], n, "roc_auc_score", context) if args[4] is not None else PythonObject(None)
    var flags = mc_map_equal(true, present[1], context)
    var partial = max_fpr is not None and mc_float(max_fpr) != 1
    var mf = mc_float(max_fpr) if partial else 0.0
    # `_fold_curves(s, flags, w, n, 1, fold="auc", max_fpr=mf)`
    var prog = McProg(context)
    var s_off = prog.put(s)
    var pos = prog.put(flags)
    var w_off = MC_NONE
    if w is not None:
        w_off = prog.put(w)
    var out = mc_fold_stages(prog, s_off, 1, pos, w_off, n, 1, partial, mf, MC_NONE)
    var arena = mc_run(prog, args[8], context)
    _ = s^
    _ = flags^
    return mc_fold_auc(arena, out, partial, mf, context)


def _roc_ovr(true: McLabels, s: PythonObject, k: Int, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_ovr` with `_OvrTail("auc", True, average)`, then the micro
    average's own fold program."""
    var labels = args[7]
    var average = args[3]
    var present = true.classes
    var n = true.n
    var classes = present
    if labels is not None:
        classes = mc_selected(labels, true.kind, present)
        if mc_ne(classes, mc_sorted(classes)):
            raise Error("Parameter 'labels' must be ordered")
        if mc_len(family_call(mc_set(present).difference, classes)) > 0:
            raise Error("'y_true' contains labels not in parameter 'labels'")
    if mc_len(classes) != k:
        raise Error("Number of classes in y_true not equal to the number of columns in 'y_score'")
    var w = mc_weights(args[4], n, "roc_auc_score", context) if args[4] is not None else PythonObject(None)
    var codes = mc_map_order(true, classes, context)
    var micro = average is not None and String(average) == "micro"
    var weighted_avg = average is not None and String(average) == "weighted"
    # `_fold_curves(s, _OneHot(codes, n, k, 0), w, n, k, stride=k, fold="auc", tail=tail)`
    var prog = McProg(context)
    var s_off = prog.put(s)
    var c_off = prog.put(codes)
    var pos = prog.scratch(n * k)
    prog.stage("onehot", n * k, [c_off, pos, n, k, 0])
    var w_off = MC_NONE
    if w is not None:
        w_off = prog.put(w)
    var out = mc_fold_stages(prog, s_off, k, pos, w_off, n, k, False, 0.0, MC_NONE)
    # the tail: ROW_SUM, then (not micro) the supports, scores and average
    var rows = prog.want(prog.alloc(1), 1)
    if n:
        prog.stage("rank_epi", n, [_RANK_ROW_SUM, s_off, n, k, rows])
    var sc = MC_NONE
    var fl = MC_NONE
    var res = MC_NONE
    if not micro:
        var got = mc_group(prog, c_off, n, k, MC_NONE, 1, w_off, 1)
        sc = prog.want(prog.alloc(2 * k), 2 * k)
        fl = prog.want(prog.alloc(k), k)
        var sup = prog.want(prog.alloc(2 * k), 2 * k)
        prog.stage("rank_epi", k, [_RANK_CURVE_SCORE, out, k, 0, sc, fl, got[1] if w is not None else got[0],
                                   1 if w is not None else 0, sup])
        if average is not None:
            res = mc_stage_average(prog, sc, sup, k, weighted_avg)
    var arena = mc_run(prog, args[8], context)
    if mc_arena_int(arena, rows) != 0:
        raise Error(_ROWS_MSG)
    if micro:
        # `_micro_inputs` and `_curves(s.reshape((n * k,)), flags, wm, n * k, 1, fold="auc")`
        var p2 = McProg(context)
        var s2 = p2.put(s)
        var c2 = p2.put(codes)
        var pos2 = p2.scratch(n * k)
        p2.stage("onehot", n * k, [c2, pos2, n, k, 1])
        var w2 = MC_NONE
        if w is not None:
            var w_in = p2.put(w)
            w2 = p2.scratch(n * k)
            p2.stage("rep_rows", n * k, [w_in, w2, k])
        var out2 = mc_fold_stages(p2, s2, 1, pos2, w2, n * k, 1, False, 0.0, MC_NONE)
        var arena2 = mc_run(p2, args[8], context)
        _ = codes^
        return mc_fold_auc(arena2, out2, False, 0.0, context)
    _ = codes^
    # `_OvrTail.result`: one warning per class without both labels (an AUC
    # never takes the AP curve rule)
    for c in range(k):  # driver-loop: over the classes
        if mc_arena_int(arena, fl + c) == 1:
            mc_undefined(_ROC_ONE_CLASS, context)
    if average is None:
        return mc_arena_f64_array(arena, sc, k, Python.tuple(k), context)
    return PythonObject(mc_arena_double(arena, res))


def _roc_ovo(true: McLabels, s: PythonObject, k: Int, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_ovo_native`: per program the scores and codes go up once, OVO_MASK
    weights each pair's rows, two curve folds per pair, CURVE_SCORE,
    OVO_PAIR and AVERAGE; pairs past `_OVO_WORDS` go to further programs
    whose pair words meet in one AVERAGE program."""
    var labels = args[7]
    var present = true.classes
    var n = true.n
    var classes = present if labels is None else mc_selected(labels, true.kind, present)
    if mc_len(classes) != k:
        raise Error("Number of classes in y_true not equal to the number of columns in 'y_score'")
    var weighted_avg = String(args[3]) == "weighted"
    var codes = mc_map_order(true, classes, context)
    var n_pairs = k * (k - 1) // 2
    var per_pair = 17 * n + 4 + 4 * _CF_OUT
    var chunk = max(1, min(n_pairs, _OVO_WORDS // max(per_pair, 1)))
    var pair_a = List[Int]()
    var pair_b = List[Int]()
    for a in range(k):  # driver-loop: over the classes (the pair table)
        for b in range(a + 1, k):  # driver-loop: over the classes after a
            pair_a.append(a)
            pair_b.append(b)
    var ps_words = List[Int32]()
    var prev_words = List[Int32]()
    var g0 = 0
    for _ in range(n_pairs + 1):  # driver-loop: over the chunks of class pairs (one program each)
        if g0 >= n_pairs:
            break
        var m = min(chunk, n_pairs - g0)
        var last_only = g0 == 0 and m == n_pairs
        var prog = McProg(context)
        var s_off = prog.put(s)
        var c_off = prog.put(codes)
        var oh = prog.scratch(n * k)
        if n:
            prog.stage("onehot", n * k, [c_off, oh, n, k, 0])
        var rows = MC_NONE
        if g0 == 0:
            rows = prog.want(prog.alloc(1), 1)
            if n:
                prog.stage("rank_epi", n, [_RANK_ROW_SUM, s_off, n, k, rows])
        var got = mc_group(prog, c_off, n, k, MC_NONE, 1, MC_NONE, 1)
        var off = got[0]
        _ = prog.want(off + k, 1)
        var cf = prog.scratch(2 * m * _CF_OUT)
        for t in range(m):  # driver-loop: over the class pairs of this program
            var a = pair_a[g0 + t]
            var b = pair_b[g0 + t]
            var w = prog.scratch(n)
            if n:
                prog.stage("rank_epi", n, [_RANK_OVO_MASK, c_off, n, a, b, w])
            _ = mc_fold_stages(prog, s_off + a, k, oh + a * n, w, n, 1, False, 0.0, cf + 2 * t * _CF_OUT)
            _ = mc_fold_stages(prog, s_off + b, k, oh + b * n, w, n, 1, False, 0.0, cf + (2 * t + 1) * _CF_OUT)
        var sc = prog.scratch(4 * m)
        var fl = prog.want(prog.alloc(2 * m), 2 * m)
        prog.stage("rank_epi", 2 * m, [_RANK_CURVE_SCORE, cf, 2 * m, 0, sc, fl, MC_NONE, 0, 0])
        var ps: Int
        var prev: Int
        if last_only:
            ps = prog.scratch(2 * m)
            prev = prog.scratch(2 * m)
        else:
            ps = prog.want(prog.alloc(2 * m), 2 * m)
            prev = prog.want(prog.alloc(2 * m), 2 * m)
        prog.stage("rank_epi", m, [_RANK_OVO_PAIR, sc, off, k, n, g0, m, ps, prev])
        var res = MC_NONE
        if last_only:
            res = mc_stage_average(prog, ps, prev, n_pairs, weighted_avg)
        var arena = mc_run(prog, args[8], context)
        if g0 == 0:
            if mc_arena_int(arena, rows) != 0:
                raise Error(_ROWS_MSG)
            if mc_arena_int(arena, off + k) != n:
                raise Error("mojolearn roc_auc_score: a y_true label is not among the classes")
        for j in range(2 * m):  # driver-loop: over the pair directions of this program
            if mc_arena_int(arena, fl + j) == 1:
                mc_undefined(_ROC_ONE_CLASS, context)
        if last_only:
            _ = codes^
            return PythonObject(mc_arena_double(arena, res))
        for j in range(2 * m):  # driver-loop: over the pair words of this program
            ps_words.append(Int32(mc_arena_int(arena, ps + j)))
            prev_words.append(Int32(mc_arena_int(arena, prev + j)))
        g0 += m
    _ = codes^
    var p2 = McProg(context)
    var ps_in = family_array(Python.tuple(len(ps_words)), "<i4", context)
    var prev_in = family_array(Python.tuple(len(prev_words)), "<i4", context)
    mcls_copy_words(Int(ps_words.unsafe_ptr()), Int(py=classical_address(ps_in)), len(ps_words))
    mcls_copy_words(Int(prev_words.unsafe_ptr()), Int(py=classical_address(prev_in)), len(prev_words))
    var res = mc_stage_average(p2, p2.put(ps_in), p2.put(prev_in), n_pairs, weighted_avg)
    var arena = mc_run(p2, args[8], context)
    _ = ps_words^
    _ = prev_words^
    return PythonObject(mc_arena_double(arena, res))


def _roc_options(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`roc_auc_options`; args = (binding, y_true, y_score, average,
    sample_weight, max_fpr, multi_class, labels, numeric_mode)."""
    var shape = mc_f32_lists(args[2], "y_score", context).shape
    var true = mc_targets(args[1], "roc_auc_score", context)
    var present = true.classes
    var multiclass = mc_len(present) > 2 or (mc_len(shape) == 2 and Int(py=shape[1]) > 2)
    var max_fpr = args[5]
    if max_fpr is not None:
        var ok = not mc_is_bool(max_fpr) and mc_is_real(max_fpr)
        if ok:
            var v = mc_float(max_fpr)
            ok = v > 0 and v <= 1
        if not ok:
            raise Error("Expected max_fpr in range (0, 1], got: " + family_repr(max_fpr))
    if not multiclass:
        return _roc_binary(true, args, max_fpr, context)
    if max_fpr is not None and mc_float(max_fpr) != 1:
        raise Error("Partial AUC computation not available in multiclass setting, 'max_fpr' must be "
                    + "set to `None`, received `max_fpr=" + family_str(max_fpr) + "` instead")
    var multi_class = args[6]
    var ovr = mc_is_str(multi_class) and String(multi_class) == "ovr"
    var ovo = mc_is_str(multi_class) and String(multi_class) == "ovo"
    if not (ovr or ovo):
        raise Error("multi_class must be in ('ovo', 'ovr')")
    var average = args[3]
    var options: PythonObject
    if ovr:
        options = Python.tuple(PythonObject("micro"), PythonObject("macro"), PythonObject("weighted"), PythonObject(None))
    else:
        options = Python.tuple(PythonObject("macro"), PythonObject("weighted"), PythonObject(None))
    if not mc_in(average, options):
        raise Error("average must be one of " + family_str(options) + " for multiclass problems")
    if average is None and ovo:
        raise Error("NotImplementedError: average=None is not implemented for multi_class='ovo'.")
    if ovo and args[4] is not None:
        raise Error("sample_weight is not supported for multiclass one-vs-one ROC AUC, "
                    + "'sample_weight' must be None in this case.")
    var s = _scores(args[2], true.n, "roc_auc_score", 2, context)
    var k = Int(py=s.shape[1])
    if ovr:
        return _roc_ovr(true, s, k, args, context)
    return _roc_ovo(true, s, k, args, context)


def _roc_auc(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """roc_auc_score; args = (binding, y_true, y_score, average,
    sample_weight, max_fpr, multi_class, labels, numeric_mode)."""
    var max_fpr = args[5]
    var full = max_fpr is None
    if not full and not mc_is_bool(max_fpr) and mc_is_real(max_fpr):
        full = mc_eq(max_fpr, PythonObject(1))
    if (mc_ne(args[3], PythonObject("macro")) or mc_ne(args[6], PythonObject("raise")) or args[7] is not None
            or args[4] is not None or not full or mc_rank(mc_shape_of(args[2])) == 2):
        return _roc_options(args, context)
    var true = _binary_labels(args[1], context)
    var scores = _binary_scores(args[2], true.n, context)
    var classes = true.classes
    if mc_len(classes) != 2:
        raise Error("roc_auc_score requires both positive and negative classes")
    var encoded = mc_map_equal(true, classes[1], context)
    var result = family_array(Python.tuple(1), "<f4", context)
    _ = family_call(mc_entry(args[0], "roc_auc_score"), classical_address(encoded), classical_address(scores),
                    classical_address(result), Python.list(PythonObject(true.n)))
    _ = result  # keep the Python object alive through the native call above
    var value = Float64(FAMILY_F32P(unsafe_from_address=Int(py=classical_address(result)))[0])
    _ = result  # keep the Python object alive through the native call above
    _ = encoded^
    _ = scores^
    var _keep_ret_result = PythonObject(value)
    _ = result  # keep the Python object alive through the native call above
    return _keep_ret_result^


# ---------------------------------------------------- precision_recall_curve


def _pr_options(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`precision_recall_curve_options` (`_curve_out(..., _CO_PR, drop)`);
    args = (binding, y_true, y_score, pos_label, sample_weight,
    drop_intermediate, numeric_mode)."""
    var caller = String("precision_recall_curve")
    var true = mc_targets(args[1], caller, context)
    var classes = true.classes
    var pos_label = args[3]
    if mc_len(classes) > 2 and pos_label is None:
        raise Error("multiclass format is not supported")
    var pos = mc_pos_label(pos_label, true.kind, classes, caller)
    var n = true.n
    var s = _scores(args[2], n, caller, 1, context)
    var w = mc_weights(args[4], n, caller, context) if args[4] is not None else PythonObject(None)
    var flags = mc_map_equal(true, pos, context)
    var drop = mc_bool(args[5])
    var prog = McProg(context)
    var s_off = prog.put(s)
    var p_off = prog.put(flags)
    var w_off = MC_NONE
    if w is not None:
        w_off = prog.put(w)
    var order = prog.scratch(n)
    var cnt = prog.scratch(1)
    var fps = prog.scratch(n)
    var tps = prog.scratch(n)
    var thr = prog.scratch(n)
    prog.stage("bin_curve", 1, [s_off, 1, p_off, w_off, n, order, fps, tps, thr, cnt, MC_NONE, 0, 0, 0])
    var head = prog.want(prog.alloc(6), 6)
    var out = prog.alloc(6 * (n + 1))
    for b in range(3):  # driver-loop: over the three output arrays
        _ = prog.want(out + 2 * b * (n + 1), 2 * (n + 1), head, 2)
    prog.stage("curve_out", 1, [_CO_PR, n, fps, tps, thr, cnt, 1 if drop else 0, out, head])
    var arena = mc_run(prog, args[6], context)
    _ = s^
    _ = flags^
    var m = mc_arena_int(arena, head)
    var status = mc_arena_int(arena, head + 1)
    if status == _CO_EMPTY:
        raise Error("mojolearn metrics: an empty curve has no thresholds")
    var t_tot = mc_arena_double(arena, head + 4)
    if t_tot == 0:
        mc_warn("No positive class found in y_true, recall is set to one for all thresholds.", mc_b().UserWarning)
    if status == _CO_ZERO_CLASS:
        return PythonObject(None)
    var precision = mc_arena_f64_array(arena, out, m, Python.tuple(m), context)
    var recall = mc_arena_f64_array(arena, out + 2 * (n + 1), m, Python.tuple(m), context)
    var thresholds = mc_arena_f64_array(arena, out + 4 * (n + 1), m - 1, Python.tuple(m - 1), context)
    return Python.tuple(precision, recall, thresholds)


def _pr_curve(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """precision_recall_curve; args = (binding, y_true, y_score, pos_label,
    sample_weight, drop_intermediate, numeric_mode)."""
    var drop = args[5]
    if not mc_is_bool(drop):
        raise Error("drop_intermediate must be a bool")
    if mc_bool(drop) or args[4] is not None:
        return _pr_options(args, context)
    var true = _binary_labels(args[1], context)
    var n = true.n
    var scores = _binary_scores(args[2], n, context)
    var classes = true.classes
    var positive: PythonObject
    if args[3] is None:
        if true.kind != "integer" or not mc_in(classes, mc_binary_sets()):
            raise Error("pos_label must be specified for labels outside {0,1} or {-1,1}")
        positive = PythonObject(1)
    else:
        var pos = mc_positive(args[3])
        if true.kind != String(pos[1]):
            raise Error("pos_label must use the same type as y_true")
        positive = pos[0]
    var encoded = mc_map_equal(true, positive, context)
    var precision = family_array(Python.tuple(n + 1), "<f4", context)
    var recall = family_array(Python.tuple(n + 1), "<f4", context)
    var thresholds = family_array(Python.tuple(n), "<f4", context)
    var size = mc_int(family_call(mc_entry(args[0], "precision_recall_curve"), classical_address(encoded),
                                  classical_address(scores), classical_address(precision), classical_address(recall),
                                  classical_address(thresholds), Python.list(PythonObject(n))))
    _ = encoded^
    _ = scores^
    if not (1 <= size and size <= n):
        raise Error("RuntimeError: precision_recall_curve returned an invalid threshold count")
    if not mc_in(positive, classes):
        mc_warn("No positive class found in y_true; recall is set to one for all thresholds.", mc_b().UserWarning)
    # `precision[:size + 1].copy()`, `recall[:size + 1].copy()`, `thresholds[:size].copy()`
    var p_out = family_array(Python.tuple(size + 1), "<f4", context)
    var r_out = family_array(Python.tuple(size + 1), "<f4", context)
    var t_out = family_array(Python.tuple(size), "<f4", context)
    mcls_copy_words(Int(py=classical_address(precision)), Int(py=classical_address(p_out)), size + 1)
    mcls_copy_words(Int(py=classical_address(recall)), Int(py=classical_address(r_out)), size + 1)
    mcls_copy_words(Int(py=classical_address(thresholds)), Int(py=classical_address(t_out)), size)
    _ = precision^
    _ = recall^
    _ = thresholds^
    return Python.tuple(p_out, r_out, t_out)


def mcls_ranking_operation(operation: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    if operation == "family_mcls_roc_auc":
        return _roc_auc(args, context)
    if operation == "family_mcls_pr_curve":
        return _pr_curve(args, context)
    raise Error("RuntimeError: unknown family operation " + operation)
