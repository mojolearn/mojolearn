# SPDX-License-Identifier: Apache-2.0
# The curve metrics of python/mojolearn/_expansion_metrics.py, one native
# call each: roc_curve, det_curve, auc, average_precision_score and
# top_k_accuracy_score.
#
# This owner is the Python `_curve_out`, `_fold_curves` / `_fold_stages`,
# `_FoldCurve`, `_ovr` / `_OvrTail` and the top-k program: the same refusals,
# then the same x_metrics PROGRAMS (arena layout, stage records, parameter
# words, order), run by the shared program runner; the fold words are turned
# into the score with the same binary64 operations in the same order. The
# micro-average precision no longer runs the one-vs-rest curve program
# whose results it never read. Same kernels: the bits do not move.
from std.memory import memcpy, bitcast
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_contains, py_get
from bindings.family_common import (
    FAMILY_F32P, FAMILY_F64P, FAMILY_I32P, family_array, family_str, family_view, family_cast_view_f64,
)
from bindings.classical_arrays import classical_address, classical_is_array, classical_dtype_str
from bindings.classical_validate import classical_view
from core.family_cast_impl import family_min_f32, family_max_f32
from bindings.metrics_exp_common import (
    MX_NONE, MxProg, MxArena, MxLab, MxMat, mx_run, mx_b, mx_len, mx_truthy, mx_is_bool, mx_is_str,
    mx_is_integral, mx_float, mx_nan, mx_isnan, mx_zero_division, mx_warn, mx_undefined, mx_sorted, mx_list,
    mx_missing, mx_encode, mx_selected, mx_codes, mx_map_eq, mx_targets, mx_pos_label, mx_scores, mx_weights,
    mx_eq, mx_list_labels,
)

comptime _CF_OUT = 10
comptime _CF_CHUNK = 1024
comptime _CO_ROC = 0
comptime _CO_PR = 1
comptime _CO_DET = 2
comptime _CO_EMPTY = 1
comptime _CO_ZERO_CLASS = 2
comptime _AX_CHUNK = 1024
comptime _AX_NEG = 1
comptime _AX_POS = 2
comptime _RANK_CURVE_SCORE = 6
comptime _RANK_AVERAGE = 7
comptime _ROW_TOPK = 0
comptime _ROC_ONE_CLASS = "Only one class is present in y_true. ROC AUC score is not defined in that case."


# ---------------------------------------------------------------- roc_curve / det_curve


def _curve_out(args: PythonObject, caller: String, kind: Int,
               context: PythonObject) raises -> Tuple[PythonObject, Float64, Float64, PythonObject]:
    """`_curve_out`; args = (y_true, y_score, pos_label, sample_weight,
    drop_intermediate, mode): (the three arrays or None, F, T, classes)."""
    var pos_label = args[2]
    var drop = args[4]
    var true = mx_targets(args[0], caller, context)
    if mx_len(true.classes) > 2 and pos_label is None:
        raise Error("multiclass format is not supported")
    var pos = mx_pos_label(pos_label, true.kind, true.classes, caller, context)
    var n = true.n
    var s = mx_scores(args[1], n, caller, 1, "y_score", context)
    var w = mx_weights(args[3], n, caller, context)
    var flags = mx_map_eq(true, pos, context)
    if n <= 0:
        raise Error("mojolearn metrics: an empty curve has no thresholds")
    var prog = MxProg(context)
    var S = prog.put(s.a)
    var POS = prog.put_i32(flags)
    var W = MX_NONE
    if w is not None:
        W = prog.put(w)
    var order = prog.scratch(n)
    var cnt = prog.scratch(1)
    var fps = prog.scratch(n)
    var tps = prog.scratch(n)
    var thr = prog.scratch(n)
    var roc_drop = kind == _CO_ROC and mx_truthy(drop)
    var keep = MX_NONE
    var CF = 0
    var CM = 0
    if roc_drop:
        keep = prog.scratch(n)
        CF = prog.scratch(3 * n)
        CM = prog.scratch(1)
    prog.stage("bin_curve", 1, [S, 1, POS, W, n, order, fps, tps, thr, cnt, keep, 1 if roc_drop else 0, CF, CM])
    var LEN = prog.want(prog.alloc(6), 6)
    var OUT = prog.alloc(6 * (n + 1))
    for b in range(3):  # small-loop(3: the three output arrays): fixed
        _ = prog.want_count(OUT + 2 * b * (n + 1), 2 * (n + 1), LEN, 2)
    var drop_flag = 1 if (mx_truthy(drop) and kind != _CO_ROC) else 0
    if roc_drop:
        prog.stage("curve_out", 1, [kind, n, CF, CF + n, CF + 2 * n, CM, drop_flag, OUT, LEN])
    else:
        prog.stage("curve_out", 1, [kind, n, fps, tps, thr, cnt, drop_flag, OUT, LEN])
    var arena = mx_run(prog, args[5], context)
    var m = arena.i32(LEN)
    var status = arena.i32(LEN + 1)
    if status == _CO_EMPTY:
        raise Error("mojolearn metrics: an empty curve has no thresholds")
    var F = arena.f64(LEN + 2)
    var T = arena.f64(LEN + 4)
    if status == _CO_ZERO_CLASS:
        return (PythonObject(None), F, T, true.classes)
    var arrays = Python.list()
    for b in range(3):  # small-loop(3: the three output arrays): fixed
        var length = m - 1 if (kind == _CO_PR and b == 2) else m
        _ = py_call(arrays.append, arena.words(OUT + 2 * b * (n + 1), 2 * length, Python.tuple(PythonObject(length)),
                                               "<f8", context))
    return (py_call(mx_b().tuple, arrays), F, T, true.classes)


def _roc_curve(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """args = (y_true, y_score, pos_label, sample_weight, drop_intermediate, mode)."""
    var r = _curve_out(args, "roc_curve", _CO_ROC, context)
    if r[1] <= 0:
        mx_undefined("No negative samples in y_true, false positive value should be meaningless", context)
    if r[2] <= 0:
        mx_undefined("No positive samples in y_true, true positive value should be meaningless", context)
    return r[0]


def _det_curve(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """args = (y_true, y_score, pos_label, sample_weight, drop_intermediate, mode)."""
    var r = _curve_out(args, "det_curve", _CO_DET, context)
    if mx_len(r[3]) != 2:
        raise Error("Only one class is present in y_true. Detection error tradeoff curve is not "
                    + "defined in that case.")
    if r[0] is None:
        mx_zero_division("float division by zero")
    return r[0]


# ---------------------------------------------------------------- auc


def _auc_values(v: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    """`_auc_f64(v)`, else `Array.from_list(flatten_mo(v), "<f8")`: the
    values as a C-order float64 Array. A flat list goes through
    `array('d')` (each number converted as float() does); a nested list, an
    integer buffer or a 2-D buffer is read in C order and cast to float64."""
    var b = mx_b()
    var it = Python.import_module("itertools")
    var values = v
    var seq = Python.tuple(b.list, b.tuple)
    var is_buffer = classical_is_array(values, context) or _is_ndarray(values, context)
    if not is_buffer and not Bool(py=py_call(b.isinstance, values, seq)) and Bool(py=py_call(b.hasattr, values, PythonObject("tolist"))):
        values = py_call(values.tolist)
    if Bool(py=py_call(b.isinstance, values, seq)):
        var nested = Bool(py=py_call(b.any, py_call(b.map, b.isinstance, values, py_call(it.repeat, seq))))
        if not nested:
            var store = py_call(context["array_type"], PythonObject("d"), values)
            var n = mx_len(store)
            var out = family_array(Python.tuple(PythonObject(n)), "<f8", context)
            if n > 0:
                memcpy(dest=FAMILY_F64P(unsafe_from_address=Int(py=classical_address(out))),
                       src=FAMILY_F64P(unsafe_from_address=Int(py=py_call(store.buffer_info)[0])), count=n)
                _ = out  # keep the Python object alive through the native call above
            return out
    var view = classical_view(values, context, name)
    var flat = family_cast_view_f64(view, name, context)
    _ = view^
    return flat


def _is_ndarray(v: PythonObject, context: PythonObject) raises -> Bool:
    var nd = py_get(context, "ndarray")
    return nd is not None and Bool(py=py_call(mx_b().isinstance, v, nd))


def _auc(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """args = (x, y)."""
    var xa = _auc_values(args[0], "x", context)
    var ya = _auc_values(args[1], "y", context)
    var n = Int(py=xa.size)
    if n != Int(py=ya.size):
        raise Error("x and y must have the same length")
    if n < 2:
        raise Error("At least 2 points are needed to compute area under curve, but x.shape = (" + String(n) + ",)")
    var prog = MxProg(context)
    var X = prog.put_i32(_f64_words(xa, n, context))
    _ = xa  # keep the Python object alive through the native call above
    var Y = prog.put_i32(_f64_words(ya, n, context))
    _ = ya  # keep the Python object alive through the native call above
    var OUT = prog.want(prog.alloc(3), 3)
    prog.stage("auc_xy", 1, [n, X, Y, OUT, _AX_CHUNK])
    _ = ya  # keep the Python object alive through the native call above
    _ = xa  # keep the Python object alive through the native call above
    var arena = mx_run(prog, PythonObject(None), context)
    var flags = arena.i32(OUT + 2)
    if (flags & _AX_NEG) and (flags & _AX_POS):
        var store = py_call(context["array_type"], PythonObject("d"), py_call(context["native_bytes_type"], 8 * n))
        memcpy(dest=FAMILY_F64P(unsafe_from_address=Int(py=py_call(store.buffer_info)[0])),
               src=FAMILY_F64P(unsafe_from_address=Int(py=classical_address(xa))), count=n)
        _ = xa  # keep the Python object alive through the native call above
        raise Error("x is neither increasing nor decreasing : " + family_str(py_call(store.tolist)) + ".")
    return PythonObject(arena.f64(OUT))


def _f64_words(values: PythonObject, n: Int, context: PythonObject) raises -> PythonObject:
    """`_f64_array_words`: the float64 values as an Int32 word array."""
    var words = family_array(Python.tuple(PythonObject(2 * n)), "<i4", context)
    memcpy(dest=FAMILY_F64P(unsafe_from_address=Int(py=classical_address(words))),
           src=FAMILY_F64P(unsafe_from_address=Int(py=classical_address(values))), count=n)
    _ = words  # keep the Python object alive through the native call above
    return words


# ---------------------------------------------------------------- the device curve fold


def _f32_bits(x: Float32) -> Int:
    return Int(bitcast[DType.int32](x))


def _fold_stages(mut prog: MxProg, S: Int, stride: Int, POS: Int, W: Int, n: Int, problems: Int, auc: Bool,
                 has_max_fpr: Bool, max_fpr: Float64, out_in: Int) raises -> Int:
    """`_fold_stages`: the curve and its fold; the fold's CF_OUT words per problem."""
    var N = n * problems
    var order = prog.scratch(N)
    var cnt = prog.scratch(problems)
    var fps = prog.scratch(N)
    var tps = prog.scratch(N)
    var thr = prog.scratch(N)
    var keep = MX_NONE
    var CF = 0
    var CM = 0
    if auc:
        keep = prog.scratch(N)
        CF = prog.scratch(3 * N)
        CM = prog.scratch(problems)
    prog.stage("bin_curve", problems, [S, stride, POS, W, n, order, fps, tps, thr, cnt, keep, 1 if auc else 0, CF, CM])
    var out = out_in
    if out < 0:
        out = prog.want(prog.alloc(_CF_OUT * problems), _CF_OUT * problems)
    var mode = 1
    var mh = 0
    var ml = 0
    if auc:
        mode = 0
        if has_max_fpr:
            # `_f32_split`: hi the nearest float32, lo the nearest float32 of the remainder
            var hi = Float32(max_fpr)
            var lo = Float32(max_fpr - Float64(hi))
            mode = 2
            mh = _f32_bits(hi)
            ml = _f32_bits(lo)
    if auc:
        prog.stage("curve_fold", problems, [n, CF, CF + N, CM, out, mode, mh, ml, _CF_CHUNK])
    else:
        prog.stage("curve_fold", problems, [n, fps, tps, cnt, out, mode, mh, ml, _CF_CHUNK])
    return out


struct _Fold(Copyable, Movable):
    """`_FoldCurve._words`: s, F, T, k, a, b, u, v, c."""
    var s: Float64
    var F: Float64
    var T: Float64
    var k: Int
    var a: Float64
    var b: Float64
    var u: Float64
    var v: Float64
    var c: Int

    def __init__(out self, arena: MxArena, off: Int):
        self.s = arena.f32(off) + arena.f32(off + 1)
        self.F = arena.f32(off + 2)
        self.T = arena.f32(off + 3)
        self.k = arena.i32(off + 4)
        self.a = arena.f32(off + 5)
        self.b = arena.f32(off + 6)
        self.u = arena.f32(off + 7)
        self.v = arena.f32(off + 8)
        self.c = arena.i32(off + 9)


def _fold_ap(arena: MxArena, off: Int) raises -> Float64:
    """`_FoldCurve.ap`."""
    var f = _Fold(arena, off)
    if f.c <= 0 or f.T == 0:
        return 0.0
    if f.k > 0:
        mx_zero_division("float division by zero")
    var q = f.s / f.T
    return q if q > 0.0 else 0.0


def _stage_average(mut prog: MxProg, SC: Int, SUP: Int, m: Int, weighted: Bool) raises -> Int:
    var res = prog.want(prog.alloc(2), 2)
    prog.stage("rank_epi", 1, [_RANK_AVERAGE, SC, SUP, m, 1 if weighted else 0, res])
    return res


# ---------------------------------------------------------------- average_precision_score


def _ap_binary(true: MxLab, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """The binary average precision: one curve and its fold."""
    var caller = String("average_precision_score")
    var pos_label = args[2]
    var present = true.classes
    if mx_len(present) == 2 and not py_contains(present, pos_label):
        raise Error("pos_label=" + family_str(pos_label) + " is not a valid label. It should be one of "
                    + family_str(present))
    var n = true.n
    var s = mx_scores(args[1], n, caller, 1, "y_score", context)
    var w = mx_weights(args[3], n, caller, context)
    var flags = mx_map_eq(true, pos_label, context)
    var prog = MxProg(context)
    var S = prog.put(s.a)
    var POS = prog.put_i32(flags)
    var W = MX_NONE
    if w is not None:
        W = prog.put(w)
    var out = _fold_stages(prog, S, 1, POS, W, n, 1, False, False, 0.0, -1)
    var arena = mx_run(prog, args[4], context)
    return PythonObject(_fold_ap(arena, out))


def _average_precision(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """args = (y_true, y_score, pos_label, sample_weight, mode, average)."""
    var caller = String("average_precision_score")
    var average = args[5]
    var averages = Python.tuple(PythonObject(None), PythonObject("micro"), PythonObject("macro"),
                                PythonObject("weighted"), PythonObject("samples"))
    if not py_contains(averages, average):
        raise Error("average has to be one of (None, 'micro', 'macro', 'weighted', 'samples')")
    var true = mx_targets(args[0], caller, context)
    if mx_len(true.classes) <= 2:
        return _ap_binary(true, args, context)
    if not mx_eq(args[2], PythonObject(1)):
        raise Error("Parameter pos_label is fixed to 1 for multiclass y_true. Do not set pos_label "
                    + "or set pos_label to 1.")
    var avg = String("none") if average is None else String(py=average)
    if avg == "samples":
        raise Error("NotImplementedError: mojolearn average_precision_score: average='samples' applies to "
                    + "multilabel targets, which are NOT IMPLEMENTED")
    # `_ovr`: the class columns, the codes, the weights
    var n = true.n
    var s = mx_scores(args[1], n, caller, 2, "y_score", context)
    var k = s.cols
    var classes = true.classes
    if mx_len(classes) != k:
        raise Error("Number of classes in y_true not equal to the number of columns in 'y_score'")
    var w = mx_weights(args[3], n, caller, context)
    var codes = mx_codes(true, classes, True, context)
    var prog = MxProg(context)
    if avg == "micro":
        # `_micro_inputs` + `_curves(s.reshape((n*k,)), ...)`: the row-major
        # one-hot flags and the repeated weights, formed by stages
        var S = prog.put(s.a)
        var C = prog.put_i32(codes)
        var POS = prog.scratch(n * k)
        prog.stage("onehot", n * k, [C, POS, n, k, 1])
        var W = MX_NONE
        if w is not None:
            var W0 = prog.put(w)
            W = prog.scratch(n * k)
            prog.stage("rep_rows", n * k, [W0, W, k])
        var out = _fold_stages(prog, S, 1, POS, W, n * k, 1, False, False, 0.0, -1)
        var arena = mx_run(prog, args[4], context)
        return PythonObject(_fold_ap(arena, out))
    # one-vs-rest: the class-major one-hot flags, the folds and the tail
    var S = prog.put(s.a)
    var C = prog.put_i32(codes)
    var POS = prog.scratch(n * k)
    prog.stage("onehot", n * k, [C, POS, n, k, 0])
    var W = MX_NONE
    if w is not None:
        W = prog.put(w)
    var out = _fold_stages(prog, S, k, POS, W, n, k, False, False, 0.0, -1)
    # `_OvrTail("ap", False, average)`
    var weighted = W != MX_NONE
    var g = prog.group(C, n, k, MX_NONE, 1, W, 1)
    var sc = prog.want(prog.alloc(2 * k), 2 * k)
    var fl = prog.want(prog.alloc(k), k)
    var sup = prog.want(prog.alloc(2 * k), 2 * k)
    prog.stage("rank_epi", k, [_RANK_CURVE_SCORE, out, k, 1, sc, fl, g[1] if weighted else g[0], 1 if weighted else 0, sup])
    var res = MX_NONE
    if average is not None:
        res = _stage_average(prog, sc, sup, k, avg == "weighted")
    var arena = mx_run(prog, args[4], context)
    var edge = False
    for c in range(k):  # driver-loop: over the classes' fold flags
        if arena.i32(fl + c) == 2:
            edge = True
    if not edge:
        if average is None:
            var _keep_ret_sc = arena.words(sc, 2 * k, Python.tuple(PythonObject(k)), "<f8", context)
            _ = sc  # keep the Python object alive through the native call above
            var _keep_ret_sc = _keep_ret_sc^
            _ = sc  # keep the Python object alive through the native call above
            return _keep_ret_sc^
        return PythonObject(arena.f64(res))
    # a class whose AP needs the curve rule takes the fold's own rule
    var scores = List[Float64]()
    for c in range(k):  # driver-loop: over the classes' scores
        if arena.i32(fl + c) == 2:
            scores.append(_fold_ap(arena, out + _CF_OUT * c))
        else:
            scores.append(arena.f64(sc + 2 * c))
    var words = family_array(Python.tuple(PythonObject(2 * k)), "<i4", context)
    var dst = FAMILY_F64P(unsafe_from_address=Int(py=classical_address(words)))
    _ = words  # keep the Python object alive through the native call above
    for c in range(k):  # driver-loop: over the classes' scores
        dst[c] = scores[c]
        _ = words  # keep the Python object alive through the native call above
    if average is None:
        var result = family_array(Python.tuple(PythonObject(k)), "<f8", context)
        memcpy(dest=FAMILY_F64P(unsafe_from_address=Int(py=classical_address(result))), src=dst, count=k)
        _ = result  # keep the Python object alive through the native call above
        _ = words  # keep the Python object alive through the native call above
        return result
    var p2 = MxProg(context)
    var SC = p2.put_i32(words)
    var SUP = p2.put_i32(arena.words(sup, 2 * k, Python.tuple(PythonObject(2 * k)), "<i4", context))
    _ = sup  # keep the Python object alive through the native call above
    var res2 = _stage_average(p2, SC, SUP, k, avg == "weighted")
    _ = sup  # keep the Python object alive through the native call above
    var a2 = mx_run(p2, args[4], context)
    return PythonObject(a2.f64(res2))


# ---------------------------------------------------------------- top_k_accuracy_score


def _top_k(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """args = (y_true, y_score, k, normalize, sample_weight, labels, mode)."""
    var caller = String("top_k_accuracy_score")
    var k_obj = args[2]
    var labels = args[5]
    if mx_is_bool(k_obj) or not mx_is_integral(k_obj, context) or Bool(py=k_obj < PythonObject(1)):
        raise Error("k must be a positive integer")
    var k = Int(py=py_call(mx_b().int, k_obj))
    var true = mx_targets(args[0], caller, context)
    var present = true.classes
    var n = true.n
    var s = mx_scores(args[1], n, caller, -1, "y_score", context)
    var binary = mx_len(present) <= 2 and not (labels is not None and mx_len(labels) > 2)
    var classes = present if labels is None else mx_selected(labels, true.kind, present, context)
    if labels is not None:
        if not mx_eq(classes, mx_sorted(classes)):
            raise Error("Parameter 'labels' must be ordered.")
        if mx_missing(present, classes):
            raise Error("'y_true' contains labels not in parameter 'labels'.")
    var n_score = s.cols if s.ndim == 2 else 2
    if binary and s.ndim == 2 and s.cols != 1:
        raise Error("`y_true` is binary while y_score is 2d with " + String(s.cols) + " classes. If `y_true` "
                    + "does not contain all the labels, `labels` must be provided.")
    var nc = mx_len(classes)
    if nc != n_score:
        raise Error("Number of classes in 'y_true' (" + String(nc) + ") not equal to the number of "
                    + "classes in 'y_score' (" + String(n_score) + ").")
    if k >= nc:
        mx_undefined("'k' (" + family_str(k_obj) + ") greater than or equal to 'n_classes' (" + String(nc)
                     + ") will result in a perfect score and is therefore meaningless.", context)
    var w = mx_weights(args[4], n, caller, context)
    var codes = mx_codes(true, classes, True, context)
    var prog = MxProg(context)
    var S: Int
    var kk: Int
    var cols: Int
    if binary:
        if k == 1:
            var p = FAMILY_F32P(unsafe_from_address=Int(py=classical_address(s.a)))
            var thr = Float32(0.5) if (family_min_f32(p, n) >= 0 and family_max_f32(p, n) <= 1) else Float32(0.0)
            var S1 = prog.put(s.a)
            S = prog.scratch(2 * n)
            prog.stage("pair_cols", n, [S1, S, _f32_bits(thr)])
            kk = 1
            cols = 2
        else:
            S = prog.scratch(2 * n)
            kk = 2
            cols = 2
    else:
        S = prog.put(s.a)
        kk = k
        cols = s.cols
    var Y = prog.put_i32(codes)
    var hit = prog.scratch(n)
    prog.stage("row_metric", n, [S, cols, Y, hit, _ROW_TOPK, kk, 0])
    var zero = prog.scratch(n)
    prog.stage("pair_key", n, [0, 0, zero, 1, 2])
    var W = MX_NONE
    if w is not None:
        W = prog.put(w)
    var tot = prog.group(zero, n, 1, hit, 1, W, 1)[1]
    var sw = MX_NONE
    if w is not None:
        sw = prog.group(zero, n, 1, MX_NONE, 1, W, 1)[1]
    var arena = mx_run(prog, args[6], context)
    var hits = arena.f32(tot)
    if not mx_truthy(args[3]):
        return PythonObject(hits)
    if w is None:
        return PythonObject(hits / Float64(n))
    return PythonObject(hits / arena.f32(sw))


def metrics_exp_curve_operation(operation: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """The curve metrics; each shell passes its arguments in the order its
    docstring's owner function names (see python/mojolearn/_expansion_metrics.py)."""
    if operation == "family_mexp_curve_roc":
        return _roc_curve(args, context)
    if operation == "family_mexp_curve_det":
        return _det_curve(args, context)
    if operation == "family_mexp_curve_auc":
        return _auc(args, context)
    if operation == "family_mexp_curve_average_precision":
        return _average_precision(args, context)
    if operation == "family_mexp_curve_top_k":
        return _top_k(args, context)
    raise Error("RuntimeError: unknown curve metric " + operation)
