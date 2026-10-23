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
from std.memory import memcpy
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_contains
from bindings.classical_arrays import classical_address, classical_dtype_str
from bindings.family_common import (
    FAMILY_F32P, family_repr, family_str, family_call, family_array, family_as_f32,
)
from bindings.mcls_common import (
    MC_NONE, MC_INT32_MAX, McLabels, McProg, mc_b, mc_len, mc_bool, mc_eq, mc_in, mc_is_str, mc_is_bool,
    mc_is_real, mc_float, mc_int, mc_sorted, mc_set, mc_list, mc_index, mc_nan, mc_warn, mc_undefined, mc_entry,
    mc_shape_of, mc_rank, mc_dim, mc_labels, mc_union, mc_map_order, mc_map_equal, mc_selected, mc_extend_order,
    mc_positive, mc_refuse_multilabel, mc_i32_1d, mc_pair_check, mc_weights, mc_f32_lists, mc_run,
    mc_arena_int, mc_arena_double, mc_arena_float, mc_arena_f64_array, mc_f64_words, mc_group, mc_counts,
)

# cm_epi kinds (x_metrics/cm_epi.mojo, op 59) and the NORM modes.
comptime _CM_COUNT = 2
comptime _CM_NORM = 3
comptime _CM_WSUM_LEAF = 7
comptime _CM_WSUM_FOLD = 8
comptime _WSUM_BLOCK = 256
# row_metric kinds (x_metrics ranking.mojo) and rank_epi MEAN.
comptime _ROW_LOGLOSS = 2
comptime _RANK_MEAN = 0


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
            context: PythonObject) raises -> McPair:
    """`_expansion_metrics._pair(y_true, y_pred, sample_weight, caller)`."""
    mc_refuse_multilabel(y_true, "y_true", caller)
    mc_refuse_multilabel(y_pred, "y_pred", caller)
    var true = mc_labels(y_true, "y_true", context)
    var pred = mc_labels(y_pred, "y_pred", context)
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


@fieldwise_init
struct McEpi(Copyable, Movable):
    """One `cls_epi` request of `_Sums(epi=...)`: (kind, k, AVG, ZD, FMODE, beta^2)."""
    var kind: Int
    var k: Int
    var avg: Int
    var zd: Int
    var fmode: Int
    var b2: Float64


struct McSums(Movable):
    """The offsets `_Sums` keeps: the groupings' (OFF, OUT) pairs (match,
    true, pred, then the weight total's), the count slots (unweighted) and
    the epilogue slots; the match key and the weights for a COUNT tail."""
    var groups: List[Int]
    var cnt: List[Int]
    var eos: List[Int]
    var match_off: Int
    var w_off: Int
    var n: Int
    var labels: Int
    var weighted: Bool

    def __init__(out self):
        self.groups = List[Int]()
        self.cnt = List[Int]()
        self.eos = List[Int]()
        self.match_off = MC_NONE
        self.w_off = MC_NONE
        self.n = 0
        self.labels = 0
        self.weighted = False


def mc_sums(mut prog: McProg, pair: McPair, order: PythonObject, epis: List[McEpi],
            context: PythonObject) raises -> McSums:
    """`_Sums(true, pred, w, order, numeric_mode, epi=...)` up to its tail:
    the codes over `order`, the pair key, the three (four weighted)
    groupings, the count slots and the cls_epi stages, in that order."""
    var s = McSums()
    var n = pair.true.n
    var labels = mc_len(order)
    s.n = n
    s.labels = labels
    s.weighted = pair.w is not None
    var yt = mc_map_order(pair.true, order, context)
    var yp = mc_map_order(pair.pred, order, context)
    var a = prog.put(yt)
    var b = prog.put(yp)
    var match_off = prog.scratch(n)
    prog.stage("pair_key", n, [a, b, match_off, labels, 1])
    var w_off = MC_NONE
    if s.weighted:
        w_off = prog.put(pair.w)
    s.match_off = match_off
    s.w_off = w_off
    var keys: List[Int] = [match_off, a, b]
    for g in range(3):  # driver-loop: over the three count groupings
        var got = mc_group(prog, keys[g], n, labels, MC_NONE, 1, w_off, 1)
        s.groups.append(got[0])
        s.groups.append(got[1])
    if s.weighted:
        var zero = prog.scratch(n)
        prog.stage("pair_key", n, [a, a, zero, 1, 2])
        var got = mc_group(prog, zero, n, 1, MC_NONE, 1, w_off, 1)
        s.groups.append(got[0])
        s.groups.append(got[1])
    else:
        for g in range(3):  # driver-loop: over the three count slots
            s.cnt.append(mc_counts(prog, s.groups[2 * g], labels))
    var col = 1 if s.weighted else 0
    for e in range(len(epis)):  # driver-loop: over the requested set-wise epilogues
        var bo = 0
        if epis[e].fmode == 2:
            var pairs: List[Float64] = [1 + epis[e].b2, epis[e].b2]
            bo = prog.put(mc_f64_words(pairs, context))
            _ = pairs  # keep the Python object alive through the native call above
        var eo = prog.alloc(8 + 6 * max(labels, 1))
        prog.stage("cls_epi", 1, [s.groups[col], s.groups[2 + col], s.groups[4 + col], epis[e].k, col, epis[e].kind,
                                  epis[e].avg, epis[e].zd, eo, epis[e].fmode, bo])
        s.eos.append(eo)
    _ = yt^
    _ = yp^
    return s^


def mc_wsum_sf64(mut prog: McProg, s: McSums, k: Int) raises -> Int:
    """`_wsum_sf64`: the weighted COUNT's hit and weight total as soft
    binary64 sums (one leaf unit per WSUM_BLOCK rows, then one fold stage
    per tree level)."""
    var nb = max(1, (s.n + _WSUM_BLOCK - 1) // _WSUM_BLOCK)
    var out = prog.scratch(4 * nb)
    prog.stage("cm_epi", nb, [_CM_WSUM_LEAF, s.match_off, s.w_off, s.n, out, k])
    var stride = 1
    for _ in range(64):  # driver-loop: over the fold tree levels (log2 of the block count)
        if stride >= nb:
            break
        prog.stage("cm_epi", (nb - stride + 2 * stride - 1) // (2 * stride), [_CM_WSUM_FOLD, out, stride, nb])
        stride *= 2
    return out


def mc_count_tail(mut prog: McProg, s: McSums, k: Int) raises -> Int:
    """`_cm_stage(..., _CM_COUNT, 11)`: the COUNT tail's 11-word slot."""
    var col = 1 if s.weighted else 0
    var out = prog.alloc(11)
    var tp = s.groups[col]
    var tot = s.n
    if s.weighted:
        tot = mc_wsum_sf64(prog, s, k)
    prog.stage("cm_epi", 1, [_CM_COUNT, tp, k, col, tot, out])
    return out


# --------------------------------------------------------------- accuracy


def _accuracy(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """args = (binding, y_true, y_pred, normalize, sample_weight, numeric_mode)."""
    var normalize = args[3]
    if not mc_is_bool(normalize):
        raise Error("normalize must be a bool")
    if not mc_bool(normalize):
        # `accuracy_count`: the COUNT tail over the match sums
        var pair = mc_pair(args[1], args[2], args[4], "accuracy_score", context)
        var prog = McProg(context)
        var s = mc_sums(prog, pair, pair.present, List[McEpi](), context)
        var out = mc_count_tail(prog, s, mc_len(pair.present))
        var arena = mc_run(prog, args[5], context)
        if not s.weighted:
            return PythonObject(mc_arena_int(arena, out + 1))
        return PythonObject(mc_arena_double(arena, out + 2))
    var yt = mc_i32_1d(args[1], "y_true", context)
    var yp = mc_i32_1d(args[2], "y_pred", context)
    mc_pair_check(yt, yp, "y_true", "y_pred")
    var n = Int(py=yt.size)
    var params = Python.list(PythonObject(n))
    var result: PythonObject
    if args[4] is not None:
        var w = mc_weights(args[4], n, "accuracy_score", context)
        result = family_call(mc_entry(args[0], "accuracy_score_weighted"), classical_address(yt),
                             classical_address(yp), classical_address(w), params)
        _ = w^
    else:
        result = family_call(mc_entry(args[0], "accuracy_score"), classical_address(yt), classical_address(yp), params)
    _ = yt^
    _ = yp^
    return family_call(mc_b().float, result)


# -------------------------------------------------------- confusion matrix


def _confusion_weighted(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`confusion_matrix_weighted`; args = (binding, y_true, y_pred, labels,
    sample_weight, normalize, numeric_mode)."""
    var pair = mc_pair(args[1], args[2], args[4], "confusion_matrix", context)
    var chosen = mc_selected(args[3], pair.true.kind, pair.present)
    if mc_len(family_call(mc_set(chosen).intersection, pair.true.classes)) == 0:
        raise Error("At least one label specified must be in y_true")
    var k = mc_len(chosen)
    var n = pair.true.n
    # `_confusion`
    if k > 4096:
        raise Error("mojolearn metrics: at most 4096 labels in a confusion matrix")
    var prog = McProg(context)
    var yt = mc_map_order(pair.true, chosen, context)
    var yp = mc_map_order(pair.pred, chosen, context)
    var a = prog.put(yt)
    var b = prog.put(yp)
    var key = prog.scratch(n)
    prog.stage("pair_key", n, [a, b, key, k, 0])
    var w_off = prog.put(pair.w)
    var got = mc_group(prog, key, n, k * k, MC_NONE, 1, w_off, 1)
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
    var arena = mc_run(prog, args[6], context)
    _ = yt^
    _ = yp^
    return mc_arena_f64_array(arena, out, k * k, Python.tuple(k, k), context)


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
    var pair = mc_classification_pair(args[1], args[2], context)
    var selected = mc_selected(args[3], pair.true.kind, pair.present)
    if mc_len(family_call(mc_set(selected).intersection, pair.true.classes)) == 0:
        raise Error("At least one label specified must be in y_true")
    var k = mc_len(selected)
    if k > 4096:
        raise Error("confusion_matrix supports at most 4096 output labels")
    var yt = mc_map_order(pair.true, selected, context)
    var yp = mc_map_order(pair.pred, selected, context)
    var norm = 0
    if normalize is not None:
        var name = String(normalize)
        norm = 1 if name == "true" else (2 if name == "pred" else 3)
    var result = family_array(Python.tuple(k, k), "<i8" if norm == 0 else "<f4", context)
    _ = family_call(mc_entry(args[0], "confusion_matrix"), classical_address(yt), classical_address(yp),
                    classical_address(result), Python.list(PythonObject(pair.true.n), PythonObject(k), PythonObject(norm)))
    _ = yt^
    _ = yp^
    return result


# ----------------------------------------------- precision, recall, f-score


def _zero_division_value(zero_division: PythonObject) raises -> Float64:
    """`_expansion_metrics._zero_division_value`."""
    if mc_is_str(zero_division):
        if String(zero_division) == "warn":
            return 0.0
    elif not mc_is_bool(zero_division) and mc_is_real(zero_division):
        if mc_in(zero_division, Python.tuple(PythonObject(0), PythonObject(1))):
            return mc_float(zero_division)
        var v = mc_float(zero_division)
        if v != v:
            return v
    raise Error("zero_division must be 'warn', 0.0, 1.0 or np.nan")


def _set_wise_labels(present: PythonObject, kind: String, average: PythonObject, labels: PythonObject,
                     pos_label: PythonObject, caller: String) raises -> PythonObject:
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
        var positive = mc_positive(pos_label)
        if String(positive[1]) != kind:
            raise Error("pos_label must use the same type as the targets")
        if not mc_in(positive[0], present) and mc_len(present) >= 2:
            raise Error("pos_label=" + family_str(pos_label) + " is not a valid label. It should be one of "
                        + family_str(present))
        return Python.list(positive[0])
    if not mc_in(pos_label, Python.tuple(PythonObject(None), PythonObject(1))):
        mc_warn("Note that pos_label (set to " + family_repr(pos_label) + ") is ignored when average != 'binary' "
                + "(got " + family_repr(average) + "). You may use labels=[pos_label] to specify a single positive class.",
                b.UserWarning)
    if labels is not None:
        return mc_selected(labels, kind, present)
    return mc_list(present)


def _prf_weighted(args: PythonObject, metric: Int, context: PythonObject) raises -> PythonObject:
    """`_prf_weighted`: `precision_recall_fscore_support(beta=1.0,
    warn_for=(names[metric],), ...)[metric]`; args = (binding, metric,
    y_true, y_pred, labels, pos_label, average, sample_weight,
    zero_division, numeric_mode)."""
    var zero_division = args[8]
    var zv = _zero_division_value(zero_division)
    var pair = mc_pair(args[2], args[3], args[7], "precision_recall_fscore_support", context)
    var average = args[6]
    var chosen = _set_wise_labels(pair.present, pair.true.kind, average, args[4], args[5],
                                  "precision_recall_fscore_support")
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
    var prog = McProg(context)
    var s = mc_sums(prog, pair, order, epis, context)
    var arena = mc_run(prog, args[9], context)
    var eo = s.eos[0]
    var bad = mc_arena_int(arena, eo)
    if mc_is_str(zero_division):
        var bit = 1 if metric == 0 else (2 if metric == 1 else 4)
        var what = String("Precision") if metric == 0 else (String("Recall") if metric == 1 else String("F-score"))
        if bad & bit:
            mc_undefined(what + " is ill-defined and being set to 0.0 in labels with no predicted/true samples. "
                         + "Use `zero_division` parameter to control this behavior.", context)
    if average is None:
        return mc_arena_f64_array(arena, eo + 8 + 2 * metric * k, k, Python.tuple(k), context)
    return PythonObject(mc_arena_double(arena, eo + 2 + 2 * metric))


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
            else:
                ok = False
        if not ok:
            raise Error("average must be 'binary', 'micro', 'macro', 'weighted' or None")
    var zero_division = args[8]
    var warn = mc_is_str(zero_division) and String(zero_division) == "warn"
    if not warn and (mc_is_bool(zero_division) or not mc_is_real(zero_division)
                     or not mc_in(zero_division, Python.tuple(PythonObject(0), PythonObject(1)))):
        raise Error("zero_division must be 'warn', 0 or 1")
    var pair = mc_classification_pair(args[2], args[3], context)
    var observed = pair.present
    var positive = 0
    var selected: PythonObject
    if code == 1:
        if mc_len(observed) > 2:
            raise Error("average='binary' requires at most two observed classes")
        var pos = mc_positive(args[5])
        if String(pos[1]) != pair.true.kind:
            raise Error("pos_label must use the same type as the targets")
        var label = pos[0]
        if not mc_in(label, observed) and mc_len(observed) == 2:
            raise Error("pos_label is not a valid observed label")
        selected = mc_list(observed)
        if not mc_in(label, selected):
            _ = family_call(selected.append, label)
        positive = Int(py=family_call(selected.index, label))
    else:
        selected = mc_selected(args[4], pair.true.kind, observed)
    var selected_count = mc_len(selected)
    var all_labels = mc_extend_order(selected, observed)
    if mc_len(all_labels) > 715827882:
        raise Error("classification metrics exceed the native class-index bound")
    var yt = mc_map_order(pair.true, all_labels, context)
    var yp = mc_map_order(pair.pred, all_labels, context)
    var width = selected_count if average is None else 1
    var output = family_array(Python.tuple(3 * width + 3), "<f4", context)
    var zero = 0 if warn else mc_int(zero_division)
    _ = family_call(mc_entry(args[0], "precision_recall_fscore"), classical_address(yt), classical_address(yp),
                    classical_address(output),
                    Python.list(PythonObject(pair.true.n), PythonObject(mc_len(all_labels)), PythonObject(code),
                                PythonObject(positive), PythonObject(zero), PythonObject(selected_count)))
    _ = yt^
    _ = yp^
    var words = FAMILY_F32P(unsafe_from_address=Int(py=classical_address(output)))
    if warn and words[3 * width + metric] != 0:
        var name = String("Precision") if metric == 0 else (String("Recall") if metric == 1 else String("F-score"))
        mc_undefined(name + " is ill-defined and being set to 0.0 due to zero "
                     + "denominator; use zero_division to control this behavior.", context)
    if average is None:
        var values = family_array(Python.tuple(width), "<f4", context)
        memcpy(dest=FAMILY_F32P(unsafe_from_address=Int(py=classical_address(values))),
               src=words + metric * width, count=width)
        _ = output^
        return values
    var value = Float64(words[metric * width])
    _ = output^
    return PythonObject(value)


# ------------------------------------------------------------------ log_loss


def mc_put_proba(mut prog: McProg, a: PythonObject, n: Int, k: Int, binary: Bool) raises -> Tuple[Int, Int]:
    """`_put_proba`: (S, FLAGS); the probabilities' check and layout."""
    var a_off = prog.put(a)
    var flags = prog.alloc(3)
    var s = a_off
    if binary:
        s = prog.scratch(2 * n)
    if n:
        prog.stage("proba_rows", n, [a_off, n, 1 if binary else k, 1 if binary else 0, s, flags])
    return (s, flags)


def mc_proba_code(arena: PythonObject, flags: Int) raises -> Int:
    """`_proba_code`: 0, or the first failed check."""
    if mc_arena_int(arena, flags) != 0:
        return 1
    if mc_arena_int(arena, flags + 1) != 0:
        return 2
    if mc_arena_int(arena, flags + 2) != 0:
        return 3
    return 0


def mc_row_mean(mut prog: McProg, s: Int, cols: Int, y: Int, n: Int, kind: Int, w: PythonObject,
                normalize: Bool, half: Bool) raises -> Int:
    """`_row_mean`'s stages (`_row_fold`, then rank_epi MEAN) before the
    run: the binary64 result's offset."""
    var out = prog.scratch(n)
    prog.stage("row_metric", n, [s, cols, y, out, kind, 0, MC_NONE])
    var zero = prog.scratch(n)
    prog.stage("pair_key", n, [0, 0, zero, 1, 2])
    var w_off = MC_NONE
    if w is not None:
        w_off = prog.put(w)
    var tot = mc_group(prog, zero, n, 1, out, 1, w_off, 1)[1]
    var sw = MC_NONE
    if w is not None:
        sw = mc_group(prog, zero, n, 1, MC_NONE, 1, w_off, 1)[1]
    var res = prog.alloc(2)
    prog.stage("rank_epi", 1, [_RANK_MEAN, tot, n, sw, 1 if normalize else 0, 1 if half else 0, res])
    return res


def mc_targets(y_true: PythonObject, caller: String, context: PythonObject) raises -> McLabels:
    """`_targets`: the multilabel refusal, then the encoded labels."""
    mc_refuse_multilabel(y_true, "y_true", caller)
    return mc_labels(y_true, "y_true", context)


def mc_pos_label(pos_label: PythonObject, kind: String, classes: PythonObject, caller: String) raises -> PythonObject:
    """`_expansion_metrics._pos_label`."""
    if pos_label is None:
        if kind != "integer" or not mc_in(classes, mc_binary_sets()):
            raise Error("y_true takes value in " + family_str(classes) + " and pos_label is not specified: either make "
                        + "y_true take value in {0, 1} or {-1, 1} or pass pos_label explicitly.")
        return PythonObject(1)
    var positive = mc_positive(pos_label)
    if String(positive[1]) != kind:
        raise Error("pos_label must use the same type as y_true")
    return positive[0]


def mc_binary_sets() raises -> PythonObject:
    """`([0], [1], [-1], [0, 1], [-1, 1])`."""
    return Python.tuple(Python.list(PythonObject(0)), Python.list(PythonObject(1)), Python.list(PythonObject(-1)),
                        Python.list(PythonObject(0), PythonObject(1)), Python.list(PythonObject(-1), PythonObject(1)))


def _log_loss_options(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`log_loss_options` (`_proba`, then one program); args = (y_true,
    y_pred, normalize, sample_weight, labels, numeric_mode)."""
    var caller = String("log_loss")
    var true = mc_targets(args[0], caller, context)
    var present = true.classes
    var n = true.n
    var a = mc_f32_lists(args[1], "y_proba", context)
    if classical_dtype_str(a, context) != "<f4":
        raise Error("TypeError: mojolearn " + caller + ": y_proba must have dtype float32; cast explicitly")
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
            classes = mc_selected(labels, true.kind, present)
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
    var proba = family_as_f32(a, nd, "y_proba", context)
    var w = PythonObject(None)
    if args[3] is not None:
        w = mc_weights(args[3], n, caller, context)
    var prog = McProg(context)
    var put = mc_put_proba(prog, proba, n, k, binary)
    var y = prog.put(codes)
    var res = mc_row_mean(prog, put[0], k, y, n, _ROW_LOGLOSS, w, mc_bool(args[2]), False)
    var arena = mc_run(prog, args[5], context)
    var code = mc_proba_code(arena, put[1])
    if code == 1:
        raise Error("mojolearn " + caller + ": y_proba must be finite")
    if code == 2:
        raise Error("mojolearn " + caller + ": y_proba must lie in [0, 1]")
    if code == 3:
        raise Error("mojolearn " + caller + ": y_proba rows must sum to one within sqrt(float32 eps)")
    _ = proba^
    _ = codes^
    return PythonObject(mc_arena_double(arena, res))


def _log_loss(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """args = (y_true, y_pred, normalize, sample_weight, labels, numeric_mode)."""
    var normalize = args[2]
    if not mc_is_bool(normalize):
        raise Error("normalize must be a bool")
    if args[3] is not None:
        return _log_loss_options(args, context)
    var true = mc_labels(args[0], "y_true", context)
    var selected = mc_selected(args[4], true.kind, true.classes)
    if mc_len(selected) < 2:
        raise Error("log_loss requires at least two labels; pass labels for one observed class")
    if mc_len(family_call(mc_set(true.classes).difference, selected)) > 0:
        raise Error("y_true contains a label missing from labels")
    var probabilities = mc_f32_lists(args[1], "input", context)
    if classical_dtype_str(probabilities, context) != "<f4":
        raise Error("TypeError: log_loss probabilities must have dtype float32")
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
    var proba = family_as_f32(probabilities, nd, "probabilities", context)
    var encoded = mc_map_order(true, selected, context)
    var prog = McProg(context)
    var put = mc_put_proba(prog, proba, n, k, binary)
    var y = prog.put(encoded)
    var res = mc_row_mean(prog, put[0], k, y, n, _ROW_LOGLOSS, PythonObject(None), mc_bool(normalize), False)
    var arena = mc_run(prog, args[5], context)
    var code = mc_proba_code(arena, put[1])
    if code == 1:
        raise Error("log_loss probabilities must be finite")
    if code == 2:
        raise Error("log_loss probabilities must lie in [0,1]")
    if code == 3:
        raise Error("log_loss probabilities rows must sum to one within sqrt(float32 eps)")
    _ = proba^
    _ = encoded^
    return PythonObject(mc_arena_double(arena, res))


def mcls_accuracy_fraction(y_true: PythonObject, y_pred: PythonObject, sample_weight: PythonObject,
                           numeric_mode: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_expansion_metrics.accuracy_fraction` (the estimators' `score`): the
    COUNT tail of `_accuracy_sums` over the match sums, the (weighted)
    match count over the row count or the weight total; a zero weight total
    raises as the division did. The same program as accuracy_count's."""
    var pair = mc_pair(y_true, y_pred, sample_weight, "accuracy_score", context)
    var prog = McProg(context)
    var s = mc_sums(prog, pair, pair.present, List[McEpi](), context)
    var out = mc_count_tail(prog, s, mc_len(pair.present))
    var arena = mc_run(prog, numeric_mode, context)
    if s.weighted and mc_arena_int(arena, out) != 0:
        raise Error("ZeroDivisionError: float division by zero")
    return PythonObject(mc_arena_double(arena, out + 8))


def mcls_classify_operation(operation: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    if operation == "family_mcls_accuracy":
        return _accuracy(args, context)
    if operation == "family_mcls_accuracy_fraction":
        # args = (y_true, y_pred, sample_weight, numeric_mode)
        return mcls_accuracy_fraction(args[0], args[1], args[2], args[3], context)
    if operation == "family_mcls_confusion":
        return _confusion(args, context)
    if operation == "family_mcls_prf":
        return _prf(args, context)
    if operation == "family_mcls_log_loss":
        return _log_loss(args, context)
    raise Error("RuntimeError: unknown family operation " + operation)
