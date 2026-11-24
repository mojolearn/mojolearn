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
from bindings.metrics_keys_unpack import metrics_keys
from bindings.py_balanced import py_item
from bindings.operation_names_mexp import (
    MEXP_OPN_FAMILY_MEXP_CURVE_AUC,
    MEXP_OPN_FAMILY_MEXP_CURVE_AVERAGE_PRECISION,
    MEXP_OPN_FAMILY_MEXP_CURVE_DET,
    MEXP_OPN_FAMILY_MEXP_CURVE_RANKING_REPORT,
    MEXP_OPN_FAMILY_MEXP_CURVE_ROC,
    MEXP_OPN_FAMILY_MEXP_CURVE_TOP_K,
)
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from bindings.admitted_family_door import admit_family_f64
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_contains, py_get, py_type, py_setitem
from bindings.family_common_unpack import (
    FAMILY_F32P,
    FAMILY_F64P,
    FAMILY_I32P,
    family_str,
)
from bindings.family_common_pack import (
    family_array,
)
from bindings.classical_arrays_door import classical_address, classical_is_array, classical_dtype_str
from bindings.classical_validate_unpack import classical_view
from bindings.metrics_exp_common_run import MX_NONE, mx_nan, mx_isnan
from bindings.metrics_program_pack import metric_arena_array, metric_f64_array
from bindings.metrics_program_door import MetricDoorProg, metric_door_run, metric_plan_run
from bindings.metrics_exp_curve_run import (
    curve_fold_stages, curve_f32_bits, CURVE_CF_OUT, CURVE_CF_CHUNK, ap_fold_value, ap_binary_stages,
    ap_ovr_stages, ap_ovr_edge, ap_ovr_scores, ap_edge_average_plan, top_k_threshold, top_k_stages,
    top_k_reads_scores, top_k_value,
)
from bindings.arena_prog_run import MetricArenaOwner
from bindings.metrics_exp_common_unpack import (
    MxLab, MxMat, mx_b, mx_len, mx_truthy, mx_is_bool, mx_is_str, mx_is_integral, mx_float, mx_zero_division,
    mx_warn, mx_undefined, mx_sorted, mx_list, mx_missing, mx_encode, mx_selected, mx_codes, mx_map_eq, mx_targets,
    mx_pos_label, mx_scores, mx_weights, mx_eq, mx_list_labels,
)

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
    var prog = MetricDoorProg()
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
    var arena = metric_door_run(prog, args[5], context)
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
        _ = py_call(arrays.append, metric_arena_array(arena, OUT + 2 * b * (n + 1), 2 * length, Python.tuple(PythonObject(length)),
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
    var it = native_loaded_module("itertools")
    var values = v
    var seq = Python.tuple(b.list, b.tuple)
    var is_buffer = classical_is_array(values, context) or _is_ndarray(values, context)
    if not is_buffer and not Bool(py=py_call(b.isinstance, values, seq)) and Bool(py=py_call(b.hasattr, values, PythonObject("tolist"))):
        values = py_call(values.tolist)
    if Bool(py=py_call(b.isinstance, values, seq)):
        var nested = Bool(py=py_call(b.any, py_call(b.map, b.isinstance, values, py_call(it.repeat, seq))))
        if not nested:
            var nk = metrics_keys()
            var store = py_call(py_item(context, nk[].k_array_type), PythonObject("d"), values)
            var n = mx_len(store)
            var out = family_array(Python.tuple(PythonObject(n)), "<f8", context)
            if n > 0:
                unsafe_memcpy(dest=FAMILY_F64P(unsafe_from_address=Int(py=classical_address(out))),
                       src=FAMILY_F64P(unsafe_from_address=Int(py=py_call(store.buffer_info)[0])), count=n)
                _ = out  # keep the Python object alive through the native call above
            return out
    var view = classical_view(values, context, name)
    var flat = admit_family_f64(view, name, context)
    _ = view^
    return flat


def _is_ndarray(v: PythonObject, context: PythonObject) raises -> Bool:
    var nk = metrics_keys()
    var nd = py_get(context, nk[].k_ndarray)
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
    var prog = MetricDoorProg()
    var X = prog.put_words(_f64_word_list(xa, n))
    var Y = prog.put_words(_f64_word_list(ya, n))
    var OUT = prog.want(prog.alloc(3), 3)
    prog.stage("auc_xy", 1, [n, X, Y, OUT, _AX_CHUNK])
    _ = ya  # keep the Python object alive through the native call above
    _ = xa  # keep the Python object alive through the native call above
    var arena = metric_door_run(prog, PythonObject(None), context)
    var flags = arena.i32(OUT + 2)
    if (flags & _AX_NEG) and (flags & _AX_POS):
        var nk = metrics_keys()
        var store = py_call(py_item(context, nk[].k_array_type), PythonObject("d"), py_call(py_item(context, nk[].k_native_bytes_type), 8 * n))
        unsafe_memcpy(dest=FAMILY_F64P(unsafe_from_address=Int(py=py_call(store.buffer_info)[0])),
               src=FAMILY_F64P(unsafe_from_address=Int(py=classical_address(xa))), count=n)
        _ = xa  # keep the Python object alive through the native call above
        raise Error("x is neither increasing nor decreasing : " + family_str(py_call(store.tolist)) + ".")
    return PythonObject(arena.f64(OUT))


def _f64_word_list(values: PythonObject, n: Int) raises -> List[Int32]:
    """UNPACK `_f64_array_words`: the n float64 values of an admitted Array
    as program-owned Int32 words, one buffer read (was a second Python Array
    pinned as a caller buffer)."""
    var words = List[Int32](length=2 * n, fill=0)
    unsafe_memcpy(dest=FAMILY_F64P(unsafe_from_address=Int(words.unsafe_ptr())),
           src=FAMILY_F64P(unsafe_from_address=Int(py=classical_address(values))), count=n)
    _ = values  # keep the Python object alive through the native copy above
    return words^


# ---------------------------------------------------------------- the device curve fold


def _fold_ap(arena: MetricArenaOwner, off: Int) raises -> Float64:
    """`_FoldCurve.ap` (bindings/metrics_exp_curve_run.mojo `ap_fold_value`,
    the one body the search scorers' RUN also reads)."""
    var zero = False
    var value = ap_fold_value(arena, off, zero)
    if zero:
        mx_zero_division("float division by zero")
    return value


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
    var prog = MetricDoorProg()
    # the typed builder (bindings/metrics_exp_curve_run.mojo `ap_binary_stages`,
    # lane mf-score5): its inputs S, POS, W pinned in that order
    var out = ap_binary_stages(prog.plan, n, w is not None)
    prog.pins.append(s.a)
    prog.pins.append(flags)
    if w is not None:
        prog.pins.append(w)
    var arena = metric_door_run(prog, args[4], context)
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
                    + "multilabel targets, which are NOT IMPLEMENTED (metrics/NOT_IMPLEMENTED.tsv)")
    # `_ovr`: the class columns, the codes, the weights
    var n = true.n
    var s = mx_scores(args[1], n, caller, 2, "y_score", context)
    var k = s.cols
    var classes = true.classes
    if mx_len(classes) != k:
        raise Error("Number of classes in y_true not equal to the number of columns in 'y_score'")
    var w = mx_weights(args[3], n, caller, context)
    var codes = mx_codes(true, classes, True, context)
    var prog = MetricDoorProg()
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
        var out = curve_fold_stages(prog.plan, S, 1, POS, W, n * k, 1, False, False, 0.0, -1)
        var arena = metric_door_run(prog, args[4], context)
        return PythonObject(_fold_ap(arena, out))
    # one-vs-rest: the class-major one-hot flags, the folds and the tail, by
    # the typed builder (bindings/metrics_exp_curve_run.mojo `ap_ovr_stages`,
    # lane mf-score5): its inputs S, C, W pinned in that order
    var o = ap_ovr_stages(prog.plan, n, k, w is not None, average is not None, avg == "weighted")
    prog.pins.append(s.a)
    prog.pins.append(codes)
    if w is not None:
        prog.pins.append(w)
    var arena = metric_door_run(prog, args[4], context)
    if not ap_ovr_edge(arena, o, k):
        if average is None:
            var per_class = metric_arena_array(arena, o.sc, 2 * k, Python.tuple(PythonObject(k)), "<f8", context)
            return per_class^
        return PythonObject(arena.f64(o.res))
    # a class whose AP needs the curve rule takes the fold's own rule
    var zero = False
    var scores = ap_ovr_scores(arena, o, k, zero)
    if zero:
        mx_zero_division("float division by zero")
    if average is None:
        return metric_f64_array(scores, context)
    # the second program reads the scores and the support as its own words
    var res2 = MX_NONE
    var p2 = ap_edge_average_plan(scores, arena.i32_words(o.sup, 2 * k), k, avg == "weighted", res2)
    var a2 = metric_plan_run(p2, List[PythonObject](), args[4], context)
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
    var classes = present if labels is None else mx_selected(labels, true.kind, present, context)
    var binary = mx_len(present) <= 2 and not (labels is not None and mx_len(classes) > 2)
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
    var prog = MetricDoorProg()
    # the typed builder (bindings/metrics_exp_curve_run.mojo `top_k_stages`,
    # lane mf-score5): its inputs (the scores when read, the codes, the
    # weights) pinned in that order
    var threshold = Float32(0.0)
    if binary and k == 1:
        threshold = top_k_threshold(Int(py=classical_address(s.a)), n)
        _ = s.a  # keep the Python object alive through the native read above
    var got = top_k_stages(prog.plan, n, binary, k, s.cols, threshold, w is not None)
    if top_k_reads_scores(binary, k):
        prog.pins.append(s.a)
    prog.pins.append(codes)
    if w is not None:
        prog.pins.append(w)
    var arena = metric_door_run(prog, args[6], context)
    return PythonObject(top_k_value(arena, got[0], got[1], n, mx_truthy(args[3])))


def _ranking_report(args: PythonObject, context: PythonObject) raises -> PythonObject:
    # Keep both incumbent curve producers and their existing stage order.
    # Candidate switches remain in the shared kernels, with no new defaults.
    var b = mx_b()
    if py_type(args[5]) is not b.str or String(args[5]) != "identical":
        raise Error("ranking_report requires numeric_mode='identical'")
    from bindings.host_inference_scalars_door import host_inference_scalar
    host_inference_scalar(args[4], context, True)
    var include_curves = Bool(py=args[4])
    var true = mx_targets(args[0], "ranking_report", context)
    if mx_len(true.classes) > 2:
        raise Error("ranking_report currently requires binary targets")
    var pos = mx_pos_label(args[2], true.kind, true.classes, "ranking_report", context)
    var n = true.n
    if n <= 0:
        raise Error("mojolearn metrics: an empty curve has no thresholds")
    var scores = mx_scores(args[1], n, "ranking_report", 1, "y_score", context)
    var weights = mx_weights(args[3], n, "ranking_report", context)
    var flags = mx_map_eq(true, pos, context)
    var prog = MetricDoorProg()
    var S = prog.put(scores.a)
    var POS = prog.put_i32(flags)
    var W = MX_NONE
    if weights is not None:
        W = prog.put(weights)
    var curves = List[Int]()
    for _ in range(2):  # driver-loop: the report's fixed ROC/AP pair
        var order = prog.scratch(n)
        var fps = prog.scratch(n)
        var tps = prog.scratch(n)
        var thr = prog.scratch(n)
        var count = prog.scratch(1)
        curves.extend([fps, tps, thr, count])
        prog.stage("bin_curve", 1, [S, 1, POS, W, n, order, fps, tps, thr, count, MX_NONE, 0, 0, 0])
    var outputs = List[Int]()
    var totals = List[Int]()
    var curve_outputs = List[Int]()
    for fold in range(2):  # driver-loop: fixed ROC/AP scalar and optional curve outputs
        var fps = curves[4 * fold]
        var tps = curves[4 * fold + 1]
        var thr = curves[4 * fold + 2]
        var count = curves[4 * fold + 3]
        var folded = prog.alloc(CURVE_CF_OUT)
        totals.append(prog.want(folded + 2, 2))
        var result = prog.want(prog.alloc(2), 2)
        var status = prog.want(prog.alloc(1), 1)
        prog.stage("curve_fold", 1, [n, fps, tps, count, folded, fold, 0, 0, CURVE_CF_CHUNK])
        prog.stage("rank_epi", 1, [_RANK_CURVE_SCORE, folded, 1, fold, result, status, MX_NONE, 0, 0])
        outputs.extend([result, status])
        if include_curves:
            var lengths = prog.want(prog.alloc(6), 6)
            var output = prog.alloc(6 * (n + 1))
            for channel in range(3):  # driver-loop: three curve output channels
                _ = prog.want_count(output + 2 * channel * (n + 1), 2 * (n + 1), lengths, 2)
            prog.stage("curve_out", 1, [_CO_ROC if fold == 0 else _CO_PR, n, fps, tps, thr, count, 0, output, lengths])
            curve_outputs.extend([output, lengths])
    var arena = metric_door_run(prog, args[5], context)
    if arena.i32(outputs[1]) == 1:
        mx_undefined(String(_ROC_ONE_CLASS), context)
    if arena.i32(outputs[3]) == 2:
        mx_zero_division("float division by zero")
    if arena.f32(totals[1] + 1) == 0:
        mx_warn("No positive class found in y_true, recall is set to one for all thresholds.", b.UserWarning, 2, context)
    var result = Python.dict()
    var nk = metrics_keys()
    py_setitem(result, nk[].k_roc_auc, PythonObject(arena.f64(outputs[0])))
    py_setitem(result, nk[].k_average_precision, PythonObject(arena.f64(outputs[2])))
    if include_curves:
        for kind in range(2):  # driver-loop: fixed ROC/PR result tuples
            var output = curve_outputs[2 * kind]
            var length = arena.i32(curve_outputs[2 * kind + 1])
            var arrays = Python.list()
            for channel in range(3):  # driver-loop: fixed curve channels with their reported lengths
                var size = length - 1 if kind == 1 and channel == 2 else length
                py_call(arrays.append, metric_arena_array(arena, output + 2 * channel * (n + 1), 2 * size, Python.tuple(size), "<f8", context))
            py_setitem(result, "roc_curve" if kind == 0 else "precision_recall_curve", py_call(b.tuple, arrays))
    return result


def metrics_exp_curve_operation(operation: Int, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """The curve metrics; each shell passes its arguments in the order its
    docstring's owner function names (see python/mojolearn/_expansion_metrics.py)."""
    if operation == MEXP_OPN_FAMILY_MEXP_CURVE_RANKING_REPORT:
        return _ranking_report(args, context)
    if operation == MEXP_OPN_FAMILY_MEXP_CURVE_ROC:
        return _roc_curve(args, context)
    if operation == MEXP_OPN_FAMILY_MEXP_CURVE_DET:
        return _det_curve(args, context)
    if operation == MEXP_OPN_FAMILY_MEXP_CURVE_AUC:
        return _auc(args, context)
    if operation == MEXP_OPN_FAMILY_MEXP_CURVE_AVERAGE_PRECISION:
        return _average_precision(args, context)
    if operation == MEXP_OPN_FAMILY_MEXP_CURVE_TOP_K:
        return _top_k(args, context)
    raise Error("RuntimeError: unknown curve metric " + String(operation))
