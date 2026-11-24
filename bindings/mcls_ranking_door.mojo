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
from bindings.metrics_slot_door import mc_slot_inputs
from metrics.slot_ops import MS_ROC_AUC, MS_PR_CURVE
from bindings.operation_names_mcls import (
    MCLS_OPN_FAMILY_MCLS_PR_CURVE,
    MCLS_OPN_FAMILY_MCLS_ROC_AUC,
)
from std.memory import bitcast
from std.python import Python, PythonObject
from bindings.admitted_family_door import admit_family
from bindings.arena_prog_run import MetricArenaOwner
from bindings.py_balanced import py_call, py_type, py_contains
from bindings.classical_arrays_door import classical_address, classical_dtype_str
from bindings.family_common_unpack import (
    FAMILY_F32P,
    FAMILY_I32P,
    family_repr,
    family_str,
    family_shape_repr,
)
from bindings.family_common_pack import (
    family_call,
    family_array,
)
from bindings.mcls_common_run import MC_NONE, MC_INT32_MAX
from bindings.mcls_common_unpack import (
    McLabels, mc_b, mc_len, mc_bool, mc_eq, mc_ne, mc_in, mc_is_str, mc_is_bool, mc_is_real, mc_float, mc_int,
    mc_sorted, mc_set, mc_nan, mc_warn, mc_undefined, mc_entry, mc_shape_of, mc_rank, mc_dim, mc_labels,
    mc_map_order, mc_map_equal, mc_selected, mc_positive, mc_weights, mc_f32_lists,
)
from bindings.metrics_program_door import MetricDoorProg, metric_door_run, metric_route_open
from bindings.mcls_ranking_run import McAuc, RocMicro, mc_fold_auc_value, roc_ovr_micro_run
from std.python._cpython import GILReleased
from bindings.metrics_program_pack import metric_arena_array
from bindings.mcls_ranking_run import (
    _CF_OUT, _CF_CHUNK, _OVO_WORDS, _RANK_ROW_SUM, _RANK_CURVE_SCORE, _RANK_AVERAGE, _RANK_OVO_MASK, _RANK_OVO_PAIR,
    _CO_PR, _CO_EMPTY, _CO_ZERO_CLASS, mc_fold_stages, mc_stage_average, mc_roc_binary_stages, mc_roc_ovr_stages,
    RocOvo, ROC_OVO_OK, ROC_OVO_ROWS, ROC_OVO_LABEL, roc_ovo_run,
)
from bindings.mcls_classify_door import mc_targets, mc_pos_label, mc_binary_sets
from core.host_finite_scan import host_all_finite_f32
from core.mcls_impl import mcls_copy_words
from checks.numerics import pinned_mul_f64

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
    var out = admit_family(a, nd, name, context).array
    if not host_all_finite_f32(FAMILY_F32P(unsafe_from_address=Int(py=classical_address(out))), Int(py=out.size)):
        raise Error("mojolearn " + caller + ": " + name + " must be finite")
    _ = out  # keep the Python object alive through the native calls above
    return out


def _binary_scores(y_score: PythonObject, n: Int, context: PythonObject) raises -> PythonObject:
    """`_binary_ranking_inputs`' score checks: 1-D finite float32 scores, n entries."""
    var scores = mc_f32_lists(y_score, "input", context)
    if classical_dtype_str(scores, context) != "<f4":
        raise Error("TypeError: binary ranking scores must have dtype float32")
    if Int(py=scores.ndim) != 1 or Int(py=scores.shape[0]) != n:
        raise Error("binary ranking scores must be one-dimensional with the same length as y_true")
    var out = admit_family(scores, 1, "scores", context).array
    if not host_all_finite_f32(FAMILY_F32P(unsafe_from_address=Int(py=classical_address(out))), n):
        raise Error("binary ranking scores must be finite")
    _ = out  # keep the Python object alive through the native calls above
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
# (the typed stages: bindings/mcls_ranking_run.mojo)


def mc_fold_auc(arena: MetricArenaOwner, off: Int, partial: Bool, max_fpr: Float64, context: PythonObject) raises -> PythonObject:
    """`_FoldCurve.auc(max_fpr)` from the fold's words (the typed body
    `mc_fold_auc_value`); a one-class fold warns and answers NaN."""
    var got = mc_fold_auc_value(arena, off, partial, max_fpr)
    if not got.defined:
        mc_undefined(_ROC_ONE_CLASS, context)
        return mc_nan()
    return PythonObject(got.value)


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
    var partial = max_fpr is not None and mc_float(max_fpr, context) != 1
    var mf = mc_float(max_fpr, context) if partial else 0.0
    # `_fold_curves(s, flags, w, n, 1, fold="auc", max_fpr=mf)`
    # the typed builder (bindings/mcls_ranking_run.mojo `mc_roc_binary_stages`):
    # inputs s, flags and (weighted) w, pinned in that order
    var prog = MetricDoorProg()
    var out = mc_roc_binary_stages(prog.plan, n, w is not None, partial, mf)
    prog.pins.append(s)
    prog.pins.append(flags)
    if w is not None:
        prog.pins.append(w)
    var arena = metric_door_run(prog, args[8], context)
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
        classes = mc_selected(labels, true.kind, present, context)
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
    if micro:
        # One program for the micro average (lane ne2-b-members, N080):
        # before, the one-vs-rest program (its onehot, k curve folds and the
        # ROW_SUM check) ran only for its ROW_SUM flag and a second program
        # uploaded the scores and codes again for the micro fold. Now the
        # micro program carries the same ROW_SUM stage after its fold (the
        # place it had in the one-vs-rest program, over the same untouched
        # scores), and the flag is read before the AUC as before: the same
        # refusal, the same fold over the same words, the same bits.
        # `_micro_inputs` and `_curves(s.reshape((n * k,)), flags, wm, n * k, 1, fold="auc")`
        # Lane bm-meta-3 (2026-11-23): UNPACK here (the route, the pinned
        # scores, codes and weights), RUN `roc_ovr_micro_run`
        # (bindings/mcls_ranking_run.mojo: the same program, typed, through
        # the one program runner `metric_inputs_run`), PACK the float.
        var pin = metric_route_open(args[8], context)
        var w_at = 0
        var w_words = -1
        if w is not None:
            w_at = Int(py=classical_address(w))
            w_words = Int(py=w.size)
        var got = RocMicro(False, McAuc(False, 0.0))
        # every Python read before the GIL is released (they ran without it: segfault in roc_auc_score micro;
        # nn-integrate, lane bm-meta-3's RUN)
        var s_at = Int(py=classical_address(s))
        var s_words = Int(py=s.size)
        var codes_at = Int(py=classical_address(codes))
        var codes_words = Int(py=codes.size)
        with GILReleased(Python()):
            got = roc_ovr_micro_run(pin.route, s_at, s_words, codes_at, codes_words, w_at, w_words, n, k)
        _ = s  # the RUN reads the pinned words
        _ = codes^
        _ = w
        _ = pin^
        if not got.rows_ok:
            raise Error(_ROWS_MSG)
        if not got.auc.defined:
            mc_undefined(_ROC_ONE_CLASS, context)
            return mc_nan()
        return PythonObject(got.auc.value)
    # `_fold_curves(s, _OneHot(codes, n, k, 0), w, n, k, stride=k, fold="auc", tail=tail)`: the typed
    # builder (bindings/mcls_ranking_run.mojo `mc_roc_ovr_stages`), inputs s, codes and (weighted) w
    var prog = MetricDoorProg()
    var slots = mc_roc_ovr_stages(prog.plan, n, k, w is not None, average is not None, weighted_avg)
    prog.pins.append(s)
    prog.pins.append(codes)
    if w is not None:
        prog.pins.append(w)
    var rows = slots.rows
    var sc = slots.sc
    var fl = slots.fl
    var res = slots.res
    var arena = metric_door_run(prog, args[8], context)
    if arena.i32(rows) != 0:
        raise Error(_ROWS_MSG)
    _ = codes^
    # `_OvrTail.result`: one warning per class without both labels (an AUC
    # never takes the AP curve rule)
    for c in range(k):  # driver-loop: over the classes
        if arena.i32(fl + c) == 1:
            mc_undefined(_ROC_ONE_CLASS, context)
    if average is None:
        return metric_arena_array(arena, sc, 2 * k, Python.tuple(k), "<f8", context)
    return PythonObject(arena.f64(res))


def _roc_ovo(true: McLabels, s: PythonObject, k: Int, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_ovo_native`: per program the scores and codes go up once, OVO_MASK
    weights each pair's rows, two curve folds per pair, CURVE_SCORE,
    OVO_PAIR and AVERAGE; pairs past `_OVO_WORDS` go to further programs
    whose pair words meet in one AVERAGE program."""
    var labels = args[7]
    var present = true.classes
    var n = true.n
    var classes = present if labels is None else mc_selected(labels, true.kind, present, context)
    if mc_len(classes) != k:
        raise Error("Number of classes in y_true not equal to the number of columns in 'y_score'")
    var weighted_avg = String(args[3]) == "weighted"
    var codes = mc_map_order(true, classes, context)
    # UNPACK here (the route, the pinned scores and codes, every address read
    # with the GIL held), RUN `roc_ovo_run` (bindings/mcls_ranking_run.mojo:
    # the same programs, typed, lane mf-metric4), PACK the warnings and float.
    var pin = metric_route_open(args[8], context)
    var s_at = Int(py=classical_address(s))
    var codes_at = Int(py=classical_address(codes))
    var got = RocOvo(ROC_OVO_OK, 0, 0.0)
    with GILReleased(Python()):
        got = roc_ovo_run(pin.route, s_at, codes_at, n, k, weighted_avg)
    _ = s  # the RUN reads the pinned words
    _ = codes^
    _ = pin^
    if got.status == ROC_OVO_ROWS:
        raise Error(_ROWS_MSG)
    if got.status == ROC_OVO_LABEL:
        raise Error("mojolearn roc_auc_score: a y_true label is not among the classes")
    for _ in range(got.undefined):  # driver-loop: over the one-class pair directions (one warning each)
        mc_undefined(_ROC_ONE_CLASS, context)
    return PythonObject(got.value)


def _roc_options(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`roc_auc_options`; args = (binding, y_true, y_score, average,
    sample_weight, max_fpr, multi_class, labels, numeric_mode)."""
    # p3-labels-metrics: y_score is read and (for a list) cast once here;
    # the multiclass loaders below take this array, not args[2] again.
    var score_view = mc_f32_lists(args[2], "y_score", context)
    var shape = score_view.shape
    var true = mc_targets(args[1], "roc_auc_score", context)
    var present = true.classes
    var multiclass = mc_len(present) > 2 or (mc_len(shape) == 2 and Int(py=shape[1]) > 2)
    var max_fpr = args[5]
    if max_fpr is not None:
        var ok = not mc_is_bool(max_fpr) and mc_is_real(max_fpr, context)
        if ok:
            var v = mc_float(max_fpr, context)
            ok = v > 0 and v <= 1
        if not ok:
            raise Error("Expected max_fpr in range (0, 1], got: " + family_repr(max_fpr))
    if not multiclass:
        return _roc_binary(true, args, max_fpr, context)
    if max_fpr is not None and mc_float(max_fpr, context) != 1:
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
    var s = _scores(score_view, true.n, "roc_auc_score", 2, context)
    var k = Int(py=s.shape[1])
    if ovr:
        return _roc_ovr(true, s, k, args, context)
    return _roc_ovo(true, s, k, args, context)


def _roc_auc(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """roc_auc_score; args = (binding, y_true, y_score, average,
    sample_weight, max_fpr, multi_class, labels, numeric_mode)."""
    var max_fpr = args[5]
    var full = max_fpr is None
    if not full and not mc_is_bool(max_fpr) and mc_is_real(max_fpr, context):
        full = mc_eq(max_fpr, PythonObject(1))
    if (mc_ne(args[3], PythonObject("macro")) or mc_ne(args[6], PythonObject("raise")) or args[7] is not None
            or args[4] is not None or not full or mc_rank(mc_shape_of(args[2], context)) == 2):
        return _roc_options(args, context)
    var true = _binary_labels(args[1], context)
    var scores = _binary_scores(args[2], true.n, context)
    var classes = true.classes
    if mc_len(classes) != 2:
        raise Error("roc_auc_score requires both positive and negative classes")
    var encoded = mc_map_equal(true, classes[1], context)
    # A score vector resident in the active resident() scope is read in place
    # (lane vdf-xm-kernels); the labels' codes are this door's host encode.
    var roc_values: List[PythonObject] = [encoded, scores]
    var roc_ints: List[Int] = [true.n]
    var device = mc_slot_inputs(args[0], MS_ROC_AUC, roc_values, roc_ints, List[Float64](), List[Int](), context)
    if device:
        return family_call(mc_b().float, device.value())
    var result = family_array(Python.tuple(1), "<f4", context)
    _ = family_call(mc_entry(args[0], "roc_auc_score"), classical_address(encoded), classical_address(scores),
                    classical_address(result), Python.list(PythonObject(true.n)))
    _ = encoded  # keep the Python object alive through the native call above
    _ = scores  # keep the Python object alive through the native call above
    _ = result  # keep the Python object alive through the native call above
    var value = Float64(FAMILY_F32P(unsafe_from_address=Int(py=classical_address(result)))[0])
    _ = result  # keep the Python object alive through the native call above
    _ = encoded^
    _ = scores^
    var _keep_ret_result_r449 = PythonObject(value)
    _ = result  # keep the Python object alive through the native call above
    return _keep_ret_result_r449^


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
    var pos = mc_pos_label(pos_label, true.kind, classes, caller, context)
    var n = true.n
    var s = _scores(args[2], n, caller, 1, context)
    var w = mc_weights(args[4], n, caller, context) if args[4] is not None else PythonObject(None)
    var flags = mc_map_equal(true, pos, context)
    var drop = mc_bool(args[5])
    var prog = MetricDoorProg()
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
    var arena = metric_door_run(prog, args[6], context)
    _ = s^
    _ = flags^
    var m = arena.i32(head)
    var status = arena.i32(head + 1)
    if status == _CO_EMPTY:
        raise Error("mojolearn metrics: an empty curve has no thresholds")
    var t_tot = arena.f64(head + 4)
    if t_tot == 0:
        mc_warn("No positive class found in y_true, recall is set to one for all thresholds.", mc_b().UserWarning, context)
    if status == _CO_ZERO_CLASS:
        return PythonObject(None)
    var precision = metric_arena_array(arena, out, 2 * m, Python.tuple(m), "<f8", context)
    var recall = metric_arena_array(arena, out + 2 * (n + 1), 2 * m, Python.tuple(m), "<f8", context)
    var thresholds = metric_arena_array(arena, out + 4 * (n + 1), 2 * (m - 1), Python.tuple(m - 1), "<f8", context)
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
        var pos = mc_positive(args[3], context)
        if true.kind != String(pos[1]):
            raise Error("pos_label must use the same type as y_true")
        positive = pos[0]
    var encoded = mc_map_equal(true, positive, context)
    var precision = family_array(Python.tuple(n + 1), "<f4", context)
    var recall = family_array(Python.tuple(n + 1), "<f4", context)
    var thresholds = family_array(Python.tuple(n), "<f4", context)
    var curve_values: List[PythonObject] = [encoded, scores]
    var curve_ints: List[Int] = [n]
    var curve_outs: List[Int] = [Int(py=classical_address(precision)), Int(py=classical_address(recall)),
                                 Int(py=classical_address(thresholds))]
    var device = mc_slot_inputs(args[0], MS_PR_CURVE, curve_values, curve_ints, List[Float64](), curve_outs, context)
    var size: Int
    if device:
        size = mc_int(device.value())
    else:
        size = mc_int(family_call(mc_entry(args[0], "precision_recall_curve"), classical_address(encoded),
                                  classical_address(scores), classical_address(precision), classical_address(recall),
                                  classical_address(thresholds), Python.list(PythonObject(n))))
    _ = encoded  # keep the Python object alive through the native call above
    _ = recall  # keep the Python object alive through the native call above
    _ = precision  # keep the Python object alive through the native call above
    _ = scores  # keep the Python object alive through the native call above
    _ = thresholds  # keep the Python object alive through the native call above
    _ = encoded^
    _ = scores^
    if not (1 <= size and size <= n):
        raise Error("RuntimeError: precision_recall_curve returned an invalid threshold count")
    _ = thresholds  # keep the Python object alive through the native calls above
    _ = recall  # keep the Python object alive through the native calls above
    _ = precision  # keep the Python object alive through the native calls above
    if not mc_in(positive, classes):
        mc_warn("No positive class found in y_true; recall is set to one for all thresholds.", mc_b().UserWarning, context)
    # `precision[:size + 1].copy()`, `recall[:size + 1].copy()`, `thresholds[:size].copy()`
    var p_out = family_array(Python.tuple(size + 1), "<f4", context)
    _ = thresholds  # keep the Python object alive through the native call above
    _ = recall  # keep the Python object alive through the native call above
    _ = precision  # keep the Python object alive through the native call above
    var r_out = family_array(Python.tuple(size + 1), "<f4", context)
    _ = thresholds  # keep the Python object alive through the native call above
    _ = recall  # keep the Python object alive through the native call above
    _ = precision  # keep the Python object alive through the native call above
    var t_out = family_array(Python.tuple(size), "<f4", context)
    _ = thresholds  # keep the Python object alive through the native call above
    _ = recall  # keep the Python object alive through the native call above
    _ = precision  # keep the Python object alive through the native call above
    mcls_copy_words(Int(py=classical_address(precision)), Int(py=classical_address(p_out)), size + 1)
    _ = p_out  # keep the Python object alive through the native call above
    _ = thresholds  # keep the Python object alive through the native call above
    _ = recall  # keep the Python object alive through the native call above
    _ = precision  # keep the Python object alive through the native call above
    mcls_copy_words(Int(py=classical_address(recall)), Int(py=classical_address(r_out)), size + 1)
    _ = r_out  # keep the Python object alive through the native call above
    _ = thresholds  # keep the Python object alive through the native call above
    _ = recall  # keep the Python object alive through the native call above
    _ = precision  # keep the Python object alive through the native call above
    mcls_copy_words(Int(py=classical_address(thresholds)), Int(py=classical_address(t_out)), size)
    _ = t_out  # keep the Python object alive through the native call above
    _ = thresholds  # keep the Python object alive through the native call above
    _ = recall  # keep the Python object alive through the native call above
    _ = precision  # keep the Python object alive through the native call above
    _ = precision^
    _ = recall^
    _ = thresholds^
    return Python.tuple(p_out, r_out, t_out)


def mcls_ranking_operation(operation: Int, args: PythonObject, context: PythonObject) raises -> PythonObject:
    if operation == MCLS_OPN_FAMILY_MCLS_ROC_AUC:
        return _roc_auc(args, context)
    if operation == MCLS_OPN_FAMILY_MCLS_PR_CURVE:
        return _pr_curve(args, context)
    raise Error("RuntimeError: unknown family operation " + String(operation))
