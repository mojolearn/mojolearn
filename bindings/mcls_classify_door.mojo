# SPDX-License-Identifier: Apache-2.0
# accuracy_score, confusion_matrix, precision_score / recall_score /
# f1_score and log_loss of python/mojolearn/_metrics_impl.py, one native
# call each (lane mcb2-mcls).
#
# The default calls keep their cuML-backed entry of the metrics binding
# (`accuracy_score`, `accuracy_score_weighted`, `confusion_matrix`,
# `precision_recall_fscore`) with the same encoded labels, addresses and
# parameter lists. The options run the x_metrics PROGRAMS the Python helpers
# of _expansion_metrics.py built (`accuracy_count`,
# `confusion_matrix_weighted`, `_prf_weighted` through
# `precision_recall_fscore_support`, the log-loss `_row_mean`,
# `log_loss_options`), rebuilt here with the same arena layout, the same
# stage records in the same order and the same declared outputs, run by the
# shared runner. Same kernels, same programs: the bits do not move.
from bindings.metrics_slot_door import (
    mc_slot_inputs, mc_slots_open, mc_slots_call, mc_resident_pair, mc_resident_none, mc_resident_order,
    McSlots, McResidentPair,
)
from metrics.slot_ops import MS_ACCURACY, MS_ACCURACY_W, MS_CONFUSION, MS_PRFS
from bindings.operation_names_mcls import (
    MCLS_OPN_FAMILY_MCLS_ACCURACY,
    MCLS_OPN_FAMILY_MCLS_ACCURACY_FRACTION,
    MCLS_OPN_FAMILY_MCLS_CONFUSION,
    MCLS_OPN_FAMILY_MCLS_LOG_LOSS,
    MCLS_OPN_FAMILY_MCLS_PRF,
)
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from bindings.admitted_family_door import admit_family
from bindings.arena_prog_run import MetricArenaOwner
from bindings.py_balanced import py_call, py_type, py_contains
from bindings.classical_arrays_door import classical_address, classical_dtype_str, classical_is_array
from bindings.classical_pandas_door import classical_admit_value
from bindings.native_loaded_module_door import native_loaded_module
from bindings.family_common_unpack import (
    FAMILY_F32P,
    family_repr,
    family_str,
)
from bindings.family_common_pack import (
    family_call,
    family_array,
)
from bindings.mcls_common_run import MC_NONE, MC_INT32_MAX
from bindings.mcls_common_unpack import (
    McLabels, mc_b, mc_len, mc_bool, mc_eq, mc_in, mc_is_str, mc_is_bool, mc_is_real, mc_float, mc_int, mc_sorted,
    mc_set, mc_list, mc_index, mc_nan, mc_warn, mc_undefined, mc_entry, mc_shape_of, mc_rank, mc_dim, mc_labels,
    mc_union, mc_map_order, mc_map_equal, mc_selected, mc_extend_order, mc_positive, mc_refuse_multilabel,
    mc_i32_1d, mc_pair_check, mc_weights, mc_f32_lists, mc_integer_view, mc_i32_array,
)
from bindings.metrics_program_door import MetricDoorProg, metric_door_run
from bindings.metrics_program_pack import metric_arena_array
from bindings.mcls_classify_run import (
    _CM_COUNT, _CM_NORM, _CM_WSUM_LEAF, _CM_WSUM_FOLD, _WSUM_BLOCK, _ROW_LOGLOSS, _RANK_MEAN,
    McEpi, McSums, mc_wsum_sf64, mc_count_tail, mc_row_mean, mc_proba_stages, mc_proba_code,
    mc_sums_stages, mc_epi_stage, mc_log_loss_stages,
)

# ------------------------------------------------------------------ targets


struct McPair(Movable):
    """`_expansion_metrics._pair`'s answer: both label vectors, the sorted
    union of their labels and the float32 weights (or None)."""
    var true: McLabels
    var pred: McLabels
    var present: PythonObject
    var w: PythonObject

    def __init__(out self, var true: McLabels, var pred: McLabels, present: PythonObject, w: PythonObject):
        self.true = true^
        self.pred = pred^
        self.present = present
        self.w = w


def mc_pair(y_true: PythonObject, y_pred: PythonObject, sample_weight: PythonObject, caller: String,
            context: PythonObject, true_admitted: Bool = False, pred_admitted: Bool = False) raises -> McPair:
    """`_expansion_metrics._pair(y_true, y_pred, sample_weight, caller)`.

    Each label input is admitted (`classical_admit_value`) once here, in the
    order the shape probes admitted them, and the multilabel refusal, the
    fast encoder and the label input step reuse that admission instead of
    walking a label list again. `true_admitted`: the caller already admitted
    y_true in this native call (GaussianProcessClassifier.score).
    `pred_admitted` likewise for y_pred (p3-labels-metrics: the metrics door,
    bindings/metrics_cls_native.mojo, admits every argument before
    `metric_arguments` snapshots it; a snapshot holds the same admitted
    leaves, so the door's calls skip this second walk of a label list)."""
    if not true_admitted:
        classical_admit_value(y_true, context)
    mc_refuse_multilabel(y_true, "y_true", caller, context, True)
    if not pred_admitted:
        classical_admit_value(y_pred, context)
    mc_refuse_multilabel(y_pred, "y_pred", caller, context, True)
    var true = mc_labels(y_true, "y_true", context, True)
    var pred = mc_labels(y_pred, "y_pred", context, True)
    if true.n != pred.n:
        raise Error("mojolearn " + caller + ": y_true and y_pred lengths differ")
    if true.kind != pred.kind:
        raise Error("TypeError: mojolearn " + caller + ": y_true and y_pred must use the same label type")
    if true.n > MC_INT32_MAX:
        raise Error("mojolearn " + caller + ": at most INT32_MAX rows")
    var w = PythonObject(None)
    if sample_weight is not None:
        w = mc_weights(sample_weight, true.n, caller, context)
    var present = mc_union(true, pred)
    return McPair(true^, pred^, present, w)


def mc_classification_pair(y_true: PythonObject, y_pred: PythonObject, context: PythonObject) raises -> McPair:
    """`_metrics_impl._classification_pair` (unweighted)."""
    var true = mc_labels(y_true, "y_true", context)
    var pred = mc_labels(y_pred, "y_pred", context)
    if true.n != pred.n:
        raise Error("y_true and y_pred lengths differ")
    if true.kind != pred.kind:
        raise Error("TypeError: y_true and y_pred must use the same label type")
    if true.n > MC_INT32_MAX:
        raise Error("classification counts require at most INT32_MAX rows")
    var present = mc_union(true, pred)
    return McPair(true^, pred^, present, PythonObject(None))


# ---------------------------------------------------------- the sums program


def mc_sums(mut prog: MetricDoorProg, pair: McPair, order: PythonObject, epis: List[McEpi],
            args: PythonObject, w_arg: Int, context: PythonObject) raises -> McSums:
    """`_Sums(true, pred, w, order, numeric_mode, epi=...)` up to its tail:
    the codes over `order`, the pair key, the three (four weighted)
    groupings, the count slots and the cls_epi stages, in that order.
    `args[w_arg]`: the sample_weight argument the weights were admitted from
    (-1: none named); the mapped codes are this call's temporaries, never
    kept resident (lane vdf-xm-wire2)."""
    var n = pair.true.n
    var labels = mc_len(order)
    var yt = mc_map_order(pair.true, order, context)
    var yp = mc_map_order(pair.pred, order, context)
    # the typed builder (bindings/mcls_classify_run.mojo `mc_sums_stages`):
    # inputs a, b and (weighted) w, pinned here in that order
    var s = mc_sums_stages(prog.plan, n, labels, pair.w is not None)
    prog.pin(yt, False)
    prog.pin(yp, False)
    if s.weighted:
        prog.pin_arg(pair.w, args, w_arg)
    for e in range(len(epis)):  # driver-loop: over the requested set-wise epilogues
        _ = mc_epi_stage(prog.plan, s, epis[e])
    _ = yt^
    _ = yp^
    return s^


# --------------------------------------------------------------- accuracy


def _accuracy(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """args = (binding, y_true, y_pred, normalize, sample_weight, numeric_mode)."""
    var normalize = args[3]
    if not mc_is_bool(normalize):
        raise Error("normalize must be a bool")
    if not mc_bool(normalize):
        # `accuracy_count`: the COUNT tail over the match sums
        var pair = mc_pair(args[1], args[2], args[4], "accuracy_score", context, True, True)
        var prog = MetricDoorProg()
        var s = mc_sums(prog, pair, pair.present, List[McEpi](), args, 4, context)
        var out = mc_count_tail(prog.plan, s, mc_len(pair.present))
        var arena = metric_door_run(prog, args[5], context)
        if not s.weighted:
            return PythonObject(arena.i32(out + 1))
        return PythonObject(arena.f64(out + 2))
    var yt = mc_i32_1d(args[1], "y_true", context)
    var yp = mc_i32_1d(args[2], "y_pred", context)
    mc_pair_check(yt, yp, "y_true", "y_pred")
    var n = Int(py=yt.size)
    var params = Python.list(PythonObject(n))
    var result: PythonObject
    if args[4] is not None:
        var w = mc_weights(args[4], n, "accuracy_score", context)
        # resident() inputs read in place (lane vdf-xm-kernels): raw int32 labels and the weights
        var w_values: List[PythonObject] = [yt, yp, w]
        var w_ints: List[Int] = [n]
        var device = mc_slot_inputs(args[0], MS_ACCURACY_W, w_values, w_ints, List[Float64](), List[Int](), context)
        if device:
            result = device.value()
        else:
            result = family_call(mc_entry(args[0], "accuracy_score_weighted"), classical_address(yt),
                                 classical_address(yp), classical_address(w), params)
        _ = yt  # keep the Python object alive through the native call above
        _ = w  # keep the Python object alive through the native call above
        _ = yp  # keep the Python object alive through the native call above
        _ = w^
    else:
        var a_values: List[PythonObject] = [yt, yp]
        var a_ints: List[Int] = [n]
        var device = mc_slot_inputs(args[0], MS_ACCURACY, a_values, a_ints, List[Float64](), List[Int](), context)
        if device:
            result = device.value()
        else:
            result = family_call(mc_entry(args[0], "accuracy_score"), classical_address(yt), classical_address(yp), params)
        _ = yp  # keep the Python object alive through the native call above
        _ = yt  # keep the Python object alive through the native call above
    _ = yt^
    _ = yp^
    return family_call(mc_b().float, result)


# ------------------------------------------------ resident label pairs


def _resident_labels(binding: PythonObject, y_true: PythonObject, y_pred: PythonObject, want_classes: Bool,
                     mut state: McSlots, context: PythonObject) raises -> McResidentPair:
    """The label pair of an unweighted classification metric encoded on the
    device (lane vdf-xm-wire; bindings/metrics_slot_door.mojo
    `mc_resident_pair`): both are contiguous 1-D signed integer arrays of one
    width (4 or 8 bytes) and length, at least one resident in the active
    `resident()` scope. Their union codes stay on the device; k (and, with
    `want_classes`, the sorted union = `pair.present`) come back once.
    Anything else (outside a scope, lists, strings, bool / unsigned / mixed
    widths, a strided view, a length mismatch the host route words) is not
    live, and the caller takes `mc_classification_pair`, unchanged."""
    if not state.resident.active():
        return mc_resident_none()
    if not classical_is_array(y_true, context) or not classical_is_array(y_pred, context):
        return mc_resident_none()
    var vt = mc_integer_view(y_true, context)
    var vp = mc_integer_view(y_pred, context)
    if not vt or not vp:
        return mc_resident_none()
    var it = vt.value().item
    var n = vt.value().count
    if (not vt.value().signed or not vp.value().signed or it != vp.value().item or vt.value().stride != it
            or vp.value().stride != it or n != vp.value().count or n > MC_INT32_MAX):
        return mc_resident_none()
    var ta = Int(py=classical_address(vt.value().view))
    var pa = Int(py=classical_address(vp.value().view))
    var pair = mc_resident_pair(binding, y_true, ta, y_pred, pa, n, it, want_classes, state)
    _ = vt^  # the views' buffers are read by address inside the step above
    _ = vp^
    return pair^


# -------------------------------------------------------- confusion matrix


def _confusion_weighted(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`confusion_matrix_weighted`; args = (binding, y_true, y_pred, labels,
    sample_weight, normalize, numeric_mode)."""
    var pair = mc_pair(args[1], args[2], args[4], "confusion_matrix", context, True, True)
    var chosen = mc_selected(args[3], pair.true.kind, pair.present, context)
    if mc_len(family_call(mc_set(chosen).intersection, pair.true.classes)) == 0:
        raise Error("At least one label specified must be in y_true")
    var k = mc_len(chosen)
    var n = pair.true.n
    # `_confusion`
    if k > 4096:
        raise Error("mojolearn metrics: at most 4096 labels in a confusion matrix")
    var prog = MetricDoorProg()
    var yt = mc_map_order(pair.true, chosen, context)
    var yp = mc_map_order(pair.pred, chosen, context)
    # the mapped codes are this call's temporaries: never kept resident (lane vdf-xm-wire2)
    var a = prog.put_temp(yt)
    var b = prog.put_temp(yp)
    var key = prog.scratch(n)
    prog.stage("pair_key", n, [a, b, key, k, 0])
    var w_off = prog.put_arg(pair.w, args, 4)
    var got = prog.group(key, n, k * k, MC_NONE, 1, w_off, 1)
    var cells = got[1]
    var col = 1
    # `_normalize`
    var normalize = args[5]
    var out = prog.want(prog.alloc(2 * k * k), 2 * k * k)
    var ts = prog.scratch(2)
    var mode = 4
    if normalize is not None:
        var name = String(normalize)
        if name == "all":
            mode = 1
            prog.stage("cm_epi", 1, [_CM_NORM, cells, k, col, 0, out, ts])
        elif name == "true":
            mode = 2
        else:
            mode = 3
    if k:
        prog.stage("cm_epi", k, [_CM_NORM, cells, k, col, mode, out, ts])
    var arena = metric_door_run(prog, args[6], context)
    _ = yt^
    _ = yp^
    return metric_arena_array(arena, out, 2 * (k * k), Python.tuple(k, k), "<f8", context)


def _confusion(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """args = (binding, y_true, y_pred, labels, sample_weight, normalize, numeric_mode)."""
    var normalize = args[5]
    if normalize is not None:
        var ok = mc_is_str(normalize)
        if ok:
            var name = String(normalize)
            ok = name == "true" or name == "pred" or name == "all"
        if not ok:
            raise Error("normalize must be None, 'true', 'pred' or 'all'")
    if args[4] is not None:
        return _confusion_weighted(args, context)
    var norm = 0
    if normalize is not None:
        var name = String(normalize)
        norm = 1 if name == "true" else (2 if name == "pred" else 3)
    if args[3] is None:
        # resident labels (lane vdf-xm-wire): the device union step's codes ARE the codes over
        # `present` (the sorted union), the order `labels=None` selects; read in place
        var state = mc_slots_open(context)
        var dev = _resident_labels(args[0], args[1], args[2], False, state, context)
        if dev.live():
            if dev.k > 4096:
                _ = state.close()
                raise Error("confusion_matrix supports at most 4096 output labels")
            var cm = family_array(Python.tuple(dev.k, dev.k), "<i8" if norm == 0 else "<f4", context)
            var addrs: List[Int] = [0, 0]
            var slots: List[Int] = [dev.codes_t, dev.codes_p]
            var ints: List[Int] = [dev.n, dev.k, norm]
            var outs: List[Int] = [Int(py=classical_address(cm))]
            _ = mc_slots_call(args[0], MS_CONFUSION, addrs, slots, ints, List[Float64](), outs, state)
            return cm
    var pair = mc_classification_pair(args[1], args[2], context)
    var selected = mc_selected(args[3], pair.true.kind, pair.present, context)
    if mc_len(family_call(mc_set(selected).intersection, pair.true.classes)) == 0:
        raise Error("At least one label specified must be in y_true")
    var k = mc_len(selected)
    if k > 4096:
        raise Error("confusion_matrix supports at most 4096 output labels")
    var yt = mc_map_order(pair.true, selected, context)
    var yp = mc_map_order(pair.pred, selected, context)
    var result = family_array(Python.tuple(k, k), "<i8" if norm == 0 else "<f4", context)
    _ = family_call(mc_entry(args[0], "confusion_matrix"), classical_address(yt), classical_address(yp),
                    classical_address(result), Python.list(PythonObject(pair.true.n), PythonObject(k), PythonObject(norm)))
    _ = yp  # keep the Python object alive through the native call above
    _ = yt  # keep the Python object alive through the native call above
    _ = result  # keep the Python object alive through the native call above
    _ = yt^
    _ = yp^
    return result


# ----------------------------------------------- precision, recall, f-score


def _zero_division_value(zero_division: PythonObject, context: PythonObject) raises -> Float64:
    """`_expansion_metrics._zero_division_value`."""
    if mc_is_str(zero_division):
        if String(zero_division) == "warn":
            return 0.0
    elif not mc_is_bool(zero_division) and mc_is_real(zero_division, context):
        if mc_in(zero_division, Python.tuple(PythonObject(0), PythonObject(1))):
            return mc_float(zero_division, context)
        var v = mc_float(zero_division, context)
        if v != v:
            return v
    raise Error("zero_division must be 'warn', 0.0, 1.0 or np.nan")


def _set_wise_labels(present: PythonObject, kind: String, average: PythonObject, labels: PythonObject,
                     pos_label: PythonObject, caller: String, context: PythonObject) raises -> PythonObject:
    """`_expansion_metrics._set_wise_labels`."""
    var b = mc_b()
    var averages = Python.tuple(PythonObject(None), PythonObject("micro"), PythonObject("macro"),
                                PythonObject("weighted"), PythonObject("samples"), PythonObject("binary"))
    if not mc_in(average, averages):
        raise Error("average has to be one of (None, 'micro', 'macro', 'weighted', 'samples', 'binary')")
    if average is not None and String(average) == "samples":
        raise Error("NotImplementedError: mojolearn " + caller + ": average='samples' applies to multilabel targets, "
                    + "which are NOT IMPLEMENTED (metrics/NOT_IMPLEMENTED.tsv)")
    if average is not None and String(average) == "binary":
        if mc_len(present) > 2:
            raise Error("Target is multiclass but average='binary'. Please choose another "
                        + "average setting, one of [None, 'micro', 'macro', 'weighted'].")
        var positive = mc_positive(pos_label, context)
        if String(positive[1]) != kind:
            raise Error("pos_label must use the same type as the targets")
        if not mc_in(positive[0], present) and mc_len(present) >= 2:
            raise Error("pos_label=" + family_str(pos_label) + " is not a valid label. It should be one of "
                        + family_str(present))
        return Python.list(positive[0])
    if not mc_in(pos_label, Python.tuple(PythonObject(None), PythonObject(1))):
        mc_warn("Note that pos_label (set to " + family_repr(pos_label) + ") is ignored when average != 'binary' "
                + "(got " + family_repr(average) + "). You may use labels=[pos_label] to specify a single positive class.",
                b.UserWarning, context)
    if labels is not None:
        return mc_selected(labels, kind, present, context)
    return mc_list(present)


def _prf_weighted(args: PythonObject, metric: Int, context: PythonObject) raises -> PythonObject:
    """`_prf_weighted`: `precision_recall_fscore_support(beta=1.0,
    warn_for=(names[metric],), ...)[metric]`; args = (binding, metric,
    y_true, y_pred, labels, pos_label, average, sample_weight,
    zero_division, numeric_mode)."""
    var zero_division = args[8]
    var zv = _zero_division_value(zero_division, context)
    var pair = mc_pair(args[2], args[3], args[7], "precision_recall_fscore_support", context, True, True)
    var average = args[6]
    var chosen = _set_wise_labels(pair.present, pair.true.kind, average, args[4], args[5],
                                  "precision_recall_fscore_support", context)
    var order = mc_extend_order(chosen, pair.present)
    var k = mc_len(chosen)
    # `_epi_codes(average, zero_division)`; beta = 1: FMODE 2, beta^2 = 1
    var avg = 2
    if average is None:
        avg = 0
    elif String(average) == "micro":
        avg = 1
    elif String(average) == "weighted":
        avg = 3
    var zd = 2 if zv != zv else Int(zv)
    var epis = List[McEpi]()
    epis.append(McEpi(2, k, avg, zd, 2, 1.0))
    var prog = MetricDoorProg()
    var s = mc_sums(prog, pair, order, epis, args, 7, context)
    var arena = metric_door_run(prog, args[9], context)
    var eo = s.eos[0]
    var bad = arena.i32(eo)
    if mc_is_str(zero_division):
        var bit = 1 if metric == 0 else (2 if metric == 1 else 4)
        var what = String("Precision") if metric == 0 else (String("Recall") if metric == 1 else String("F-score"))
        if bad & bit:
            mc_undefined(what + " is ill-defined and being set to 0.0 in labels with no predicted/true samples. "
                         + "Use `zero_division` parameter to control this behavior.", context)
    if average is None:
        return metric_arena_array(arena, eo + 8 + 2 * metric * k, 2 * k, Python.tuple(k), "<f8", context)
    return PythonObject(arena.f64(eo + 2 + 2 * metric))


def _prf(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_precision_recall_fscore`; args = (binding, metric, y_true, y_pred,
    labels, pos_label, average, sample_weight, zero_division, numeric_mode)."""
    var metric = Int(py=args[1])
    if args[7] is not None:
        return _prf_weighted(args, metric, context)
    var average = args[6]
    var code = 0
    if average is not None:
        var ok = mc_is_str(average)
        if ok:
            var name = String(average)
            if name == "binary":
                code = 1
            elif name == "micro":
                code = 2
            elif name == "macro":
                code = 3
            elif name == "weighted":
                code = 4
            elif name == "samples":
                # The weighted path's words (`_prf_weighted` -> `_set_wise_labels`) and a search scorer's
                # (core/msel_scorer_run.mojo `row.samples`), one text per door (lane w3f2-words: this
                # unweighted path said "average must be ..." for an average scikit-learn accepts).
                raise Error("NotImplementedError: mojolearn precision_recall_fscore_support: average='samples'"
                            + " applies to multilabel targets, which are NOT IMPLEMENTED (metrics/NOT_IMPLEMENTED.tsv)")
            else:
                ok = False
        if not ok:
            raise Error("average must be 'binary', 'micro', 'macro', 'weighted' or None")
    var zero_division = args[8]
    var warn = mc_is_str(zero_division) and String(zero_division) == "warn"
    if not warn and (mc_is_bool(zero_division) or not mc_is_real(zero_division, context)
                     or not mc_in(zero_division, Python.tuple(PythonObject(0), PythonObject(1)))):
        raise Error("zero_division must be 'warn', 0 or 1")
    var zero = 0 if warn else mc_int(zero_division)
    # resident labels (lane vdf-xm-wire): the union codes stay on the device; `observed` (the
    # sorted union) comes back once for the label choices below
    var state = mc_slots_open(context)
    var dev = _resident_labels(args[0], args[2], args[3], True, state, context)
    if dev.live():
        try:
            var order = _prf_order(code, dev.classes, "integer", args, context)
            var all_labels = order[1]
            var ct = dev.codes_t
            var cp = dev.codes_p
            var prefix = family_call(mc_b().list, family_call(native_loaded_module("itertools").islice, all_labels,
                                                              PythonObject(dev.k)))
            if not mc_eq(prefix, dev.classes):
                # the union codes mapped to `all_labels` (a superset of the union) on the device;
                # when `all_labels` starts with the union in its order the codes already are its indices
                var table = _order_table(dev.classes, all_labels, context)
                ct = mc_resident_order(args[0], ct, dev.n, table, state)
                cp = mc_resident_order(args[0], cp, dev.n, table, state)
            var width = order[0] if average is None else 1
            var output = family_array(Python.tuple(3 * width + 3), "<f4", context)
            var addrs: List[Int] = [0, 0]
            var slots: List[Int] = [ct, cp]
            var ints: List[Int] = [dev.n, mc_len(all_labels), code, order[2], zero, order[0]]
            var outs: List[Int] = [Int(py=classical_address(output))]
            _ = mc_slots_call(args[0], MS_PRFS, addrs, slots, ints, List[Float64](), outs, state)
            return _prf_pack(output, width, metric, warn, average, context)
        except error:
            _ = state.close()  # a refusal of the label choices: the call's slots go with it
            raise error^
    var pair = mc_classification_pair(args[2], args[3], context)
    var order = _prf_order(code, pair.present, pair.true.kind, args, context)
    var selected_count = order[0]
    var all_labels = order[1]
    var yt = mc_map_order(pair.true, all_labels, context)
    var yp = mc_map_order(pair.pred, all_labels, context)
    var width = selected_count if average is None else 1
    var output = family_array(Python.tuple(3 * width + 3), "<f4", context)
    _ = family_call(mc_entry(args[0], "precision_recall_fscore"), classical_address(yt), classical_address(yp),
                    classical_address(output),
                    Python.list(PythonObject(pair.true.n), PythonObject(mc_len(all_labels)), PythonObject(code),
                                PythonObject(order[2]), PythonObject(zero), PythonObject(selected_count)))
    _ = yp  # keep the Python object alive through the native call above
    _ = yt  # keep the Python object alive through the native call above
    _ = output  # keep the Python object alive through the native call above
    _ = yt^
    _ = yp^
    return _prf_pack(output, width, metric, warn, average, context)


def _prf_order(code: Int, observed: PythonObject, kind: String, args: PythonObject,
               context: PythonObject) raises -> Tuple[Int, PythonObject, Int]:
    """The label choices of `_precision_recall_fscore` over the observed
    labels (the sorted union): (selected count, all labels in class-index
    order, the positive label's index). Shared by the host and the
    resident-label routes (lane vdf-xm-wire): the same checks, words and order."""
    var positive = 0
    var selected: PythonObject
    if code == 1:
        if mc_len(observed) > 2:
            raise Error("average='binary' requires at most two observed classes")
        var pos = mc_positive(args[5], context)
        if String(pos[1]) != kind:
            raise Error("pos_label must use the same type as the targets")
        var label = pos[0]
        if not mc_in(label, observed) and mc_len(observed) == 2:
            raise Error("pos_label is not a valid observed label")
        selected = mc_list(observed)
        if not mc_in(label, selected):
            _ = family_call(selected.append, label)
        positive = Int(py=family_call(selected.index, label))
    else:
        selected = mc_selected(args[4], kind, observed, context)
    var selected_count = mc_len(selected)
    var all_labels = mc_extend_order(selected, observed)
    if mc_len(all_labels) > 715827882:
        raise Error("classification metrics exceed the native class-index bound")
    return (selected_count, all_labels, positive)


def _order_table(classes: PythonObject, order: PythonObject, context: PythonObject) raises -> PythonObject:
    """`mc_map_order`'s per-class table (the index of each class in `order`,
    -1 outside it) as a C-contiguous int32 array: O(k) host words."""
    var b = mc_b()
    var index = mc_index(order)
    var table = family_call(b.list, family_call(b.map, index.get, classes,
                                                family_call(native_loaded_module("itertools").repeat, PythonObject(-1))))
    return mc_i32_array(table, context)


def _prf_pack(output: PythonObject, width: Int, metric: Int, warn: Bool, average: PythonObject,
              context: PythonObject) raises -> PythonObject:
    """PACK of `_precision_recall_fscore`: the zero-division warning and the
    requested metric's value(s) out of the entry's 3 * width + 3 words."""
    var words = FAMILY_F32P(unsafe_from_address=Int(py=classical_address(output)))
    _ = output  # keep the Python object alive through the native call above
    if warn and words[3 * width + metric] != 0:
        var name = String("Precision") if metric == 0 else (String("Recall") if metric == 1 else String("F-score"))
        mc_undefined(name + " is ill-defined and being set to 0.0 due to zero "
                     + "denominator; use zero_division to control this behavior.", context)
    _ = output  # keep the Python object alive through the native calls above
    if average is None:
        var values = family_array(Python.tuple(width), "<f4", context)
        unsafe_memcpy(dest=FAMILY_F32P(unsafe_from_address=Int(py=classical_address(values))),
               src=words + metric * width, count=width)
        _ = values  # keep the Python object alive through the native call above
        _ = output  # keep the Python object alive through the native call above
        return values
    var value = Float64(words[metric * width])
    _ = output
    return PythonObject(value)


# ------------------------------------------------------------------ log_loss


# Removed (a plain door path, not an experiment): `mc_put_proba` and `mc_row_mean_door`, the pinned halves of the
# log-loss program, removed 2026-11-23 by lane mf-metric4; both callers take the typed `mc_log_loss_stages`.


def mc_targets(y_true: PythonObject, caller: String, context: PythonObject) raises -> McLabels:
    """`_targets`: the multilabel refusal, then the encoded labels."""
    mc_refuse_multilabel(y_true, "y_true", caller, context)
    return mc_labels(y_true, "y_true", context)


def mc_pos_label(pos_label: PythonObject, kind: String, classes: PythonObject, caller: String, context: PythonObject) raises -> PythonObject:
    """`_expansion_metrics._pos_label`."""
    if pos_label is None:
        if kind != "integer" or not mc_in(classes, mc_binary_sets()):
            raise Error("y_true takes value in " + family_str(classes) + " and pos_label is not specified: either make "
                        + "y_true take value in {0, 1} or {-1, 1} or pass pos_label explicitly.")
        return PythonObject(1)
    var positive = mc_positive(pos_label, context)
    if String(positive[1]) != kind:
        raise Error("pos_label must use the same type as y_true")
    return positive[0]


def mc_binary_sets() raises -> PythonObject:
    """`([0], [1], [-1], [0, 1], [-1, 1])`."""
    return Python.tuple(Python.list(PythonObject(0)), Python.list(PythonObject(1)), Python.list(PythonObject(-1)),
                        Python.list(PythonObject(0), PythonObject(1)), Python.list(PythonObject(-1), PythonObject(1)))


# TOMBSTONE: _caller_words (the per-call address compare of log_loss's y_proba, on every route; lane
# vdf-xm-wire) deleted 2026-11-24 by lane vdf-xm-wire2: the runner compares inside resident() only
# (bindings/arena_prog_native_door.mojo `metric_reads_source`); code recoverable at 82e076645c.


def _log_loss_options(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`log_loss_options` (`_proba`, then one program); args = (y_true,
    y_pred, normalize, sample_weight, labels, numeric_mode)."""
    var caller = String("log_loss")
    var true = mc_targets(args[0], caller, context)
    var present = true.classes
    var n = true.n
    var a = mc_f32_lists(args[1], "y_proba", context)
    # float64 too: the classifiers' predict_proba is float64, and the
    # admission below rounds it to the float32 words the program reads
    # (`admit_family`). Refusing it made metrics.log_loss(y, predict_proba(X))
    # and the neg_log_loss scorer fail on every estimator (proving suite
    # 2026-11-22).
    var a_dtype = classical_dtype_str(a, context)
    if a_dtype != "<f4" and a_dtype != "<f8":
        raise Error("TypeError: mojolearn " + caller + ": y_proba must have dtype float32 or float64; cast explicitly")
    var nd = Int(py=a.ndim)
    var binary = nd == 1 or (nd == 2 and Int(py=a.shape[1]) == 1)
    var codes: PythonObject
    var k: Int
    if binary:
        if mc_len(present) > 2:
            raise Error("The type of the target inferred from y_true is multiclass but should be binary "
                        + "according to the shape of y_prob.")
        # `_pos_label(None, ...)`, its ValueError answered by the last class
        var pos = present[mc_len(present) - 1]
        if true.kind == "integer" and mc_in(present, mc_binary_sets()):
            pos = PythonObject(1)
        codes = mc_map_equal(true, pos, context)
        k = 2
    else:
        var labels = args[4]
        var classes = present
        if labels is not None:
            classes = mc_selected(labels, true.kind, present, context)
            if mc_len(family_call(mc_set(present).difference, classes)) > 0:
                raise Error("y_true contains values not belonging to the passed labels")
        classes = mc_sorted(classes)
        if nd < 2:
            raise Error("IndexError: tuple index out of range")
        k = Int(py=a.shape[1])
        if mc_len(classes) != k:
            raise Error("y_true and y_proba contain different number of classes: " + String(mc_len(classes))
                        + " vs " + String(k))
        codes = mc_map_order(true, classes, context)
    var rows = 0 if nd == 0 else Int(py=a.shape[0])
    if rows != n:
        raise Error("y_true and y_proba have different numbers of rows")
    # `as_f32_c(a, ndim=a.ndim)`: an (n, 1) column is the words of the (n,) vector
    var proba = admit_family(a, nd, "y_proba", context).array
    var w = PythonObject(None)
    if args[3] is not None:
        w = mc_weights(args[3], n, caller, context)
    # the typed builder (bindings/mcls_classify_run.mojo `mc_log_loss_stages`):
    # inputs proba, codes and (weighted) w, pinned in that order
    var prog = MetricDoorProg()
    var put = mc_log_loss_stages(prog.plan, n, k, binary, w is not None, mc_bool(args[2]))
    # the slot form (lane vdf-xm-wire): y_proba and the weights, when the admitted words ARE the
    # caller's buffer, are read from their resident copies inside resident(); a converted copy and the
    # codes are this call's temporaries. The runner compares the buffers inside a scope only (lane
    # vdf-xm-wire2; was an address compare per call here, `_caller_words`, on every route).
    prog.pin_arg(proba, args, 1)
    prog.pin(codes, False)
    if w is not None:
        prog.pin_arg(w, args, 3)
    var res = put[1]
    var arena = metric_door_run(prog, args[5], context)
    var code = mc_proba_code(arena, put[0])  # put = (FLAGS, result)
    if code == 1:
        raise Error("mojolearn " + caller + ": y_proba must be finite")
    if code == 2:
        raise Error("mojolearn " + caller + ": y_proba must lie in [0, 1]")
    if code == 3:
        raise Error("mojolearn " + caller + ": y_proba rows must sum to one within sqrt(float32 eps)")
    _ = proba^
    _ = codes^
    return PythonObject(arena.f64(res))


def _log_loss(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """args = (y_true, y_pred, normalize, sample_weight, labels, numeric_mode)."""
    var normalize = args[2]
    if not mc_is_bool(normalize):
        raise Error("normalize must be a bool")
    if args[3] is not None:
        return _log_loss_options(args, context)
    var true = mc_labels(args[0], "y_true", context)
    var selected = mc_selected(args[4], true.kind, true.classes, context)
    if mc_len(selected) < 2:
        raise Error("log_loss requires at least two labels; pass labels for one observed class")
    if mc_len(family_call(mc_set(true.classes).difference, selected)) > 0:
        raise Error("y_true contains a label missing from labels")
    var probabilities = mc_f32_lists(args[1], "input", context)
    # float32 or float64 (rounded to float32 by `admit_family` below), as in
    # `_log_loss_options`.
    var p_dtype = classical_dtype_str(probabilities, context)
    if p_dtype != "<f4" and p_dtype != "<f8":
        raise Error("TypeError: log_loss probabilities must have dtype float32 or float64")
    var nd = Int(py=probabilities.ndim)
    if nd != 1 and nd != 2:
        raise Error("log_loss probabilities must have shape (n,), (n,1) or (n,k)")
    var n = true.n
    if Int(py=probabilities.shape[0]) != n:
        raise Error("y_true and probability row counts differ")
    var binary = nd == 1 or Int(py=probabilities.shape[1]) == 1
    var k = mc_len(selected)
    if binary:
        if k != 2:
            raise Error("one-column probabilities require exactly two labels")
    elif Int(py=probabilities.shape[1]) < 2 or Int(py=probabilities.shape[1]) != k:
        raise Error("probability columns must match the label count (at least two)")
    if n > MC_INT32_MAX or Int(py=probabilities.size) > MC_INT32_MAX or (binary and 2 * n > MC_INT32_MAX):
        raise Error("log_loss exceeds the native Int32 indexing bound")
    var proba = admit_family(probabilities, nd, "probabilities", context).array
    var encoded = mc_map_order(true, selected, context)
    var prog = MetricDoorProg()
    var put = mc_log_loss_stages(prog.plan, n, k, binary, False, mc_bool(normalize))
    prog.pin_arg(proba, args, 1)  # the caller's buffer: the slot form inside resident() (lanes vdf-xm-wire, vdf-xm-wire2)
    prog.pin(encoded, False)
    var res = put[1]
    var arena = metric_door_run(prog, args[5], context)
    var code = mc_proba_code(arena, put[0])  # put = (FLAGS, result)
    if code == 1:
        raise Error("log_loss probabilities must be finite")
    if code == 2:
        raise Error("log_loss probabilities must lie in [0,1]")
    if code == 3:
        raise Error("log_loss probabilities rows must sum to one within sqrt(float32 eps)")
    _ = proba^
    _ = encoded^
    return PythonObject(arena.f64(res))


def mcls_accuracy_fraction(y_true: PythonObject, y_pred: PythonObject, sample_weight: PythonObject,
                           numeric_mode: PythonObject, context: PythonObject,
                           true_admitted: Bool = False, pred_admitted: Bool = False) raises -> PythonObject:
    """`_expansion_metrics.accuracy_fraction` (the estimators' `score`): the
    COUNT tail of `_accuracy_sums` over the match sums, the (weighted)
    match count over the row count or the weight total; a zero weight total
    raises as the division did. The same program as accuracy_count's."""
    var pair = mc_pair(y_true, y_pred, sample_weight, "accuracy_score", context, true_admitted, pred_admitted)
    var prog = MetricDoorProg()
    # no argument tuple here: the weights stay unmarked (kept as the caller's, as before)
    var s = mc_sums(prog, pair, pair.present, List[McEpi](), PythonObject(None), -1, context)
    var out = mc_count_tail(prog.plan, s, mc_len(pair.present))
    var arena = metric_door_run(prog, numeric_mode, context)
    if s.weighted and arena.i32(out) != 0:
        raise Error("ZeroDivisionError: float division by zero")
    return PythonObject(arena.f64(out + 8))


def mcls_classify_operation(operation: Int, args: PythonObject, context: PythonObject) raises -> PythonObject:
    if operation == MCLS_OPN_FAMILY_MCLS_ACCURACY:
        return _accuracy(args, context)
    if operation == MCLS_OPN_FAMILY_MCLS_ACCURACY_FRACTION:
        # args = (y_true, y_pred, sample_weight, numeric_mode)
        return mcls_accuracy_fraction(args[0], args[1], args[2], args[3], context, True, True)
    if operation == MCLS_OPN_FAMILY_MCLS_CONFUSION:
        return _confusion(args, context)
    if operation == MCLS_OPN_FAMILY_MCLS_PRF:
        return _prf(args, context)
    if operation == MCLS_OPN_FAMILY_MCLS_LOG_LOSS:
        return _log_loss(args, context)
    raise Error("RuntimeError: unknown family operation " + String(operation))
