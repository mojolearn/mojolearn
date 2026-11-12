# SPDX-License-Identifier: Apache-2.0
# The probability, hinge and ranking metrics of
# python/mojolearn/_expansion_metrics.py, one native call each:
# brier_score_loss, d2_log_loss_score, d2_brier_score, hinge_loss,
# dcg_score, ndcg_score, coverage_error,
# label_ranking_average_precision_score and label_ranking_loss.
#
# This owner is the Python `_proba` / `_put_proba`, `_row_fold` /
# `_row_mean`, `_relevance` / `_relevance_put` and `_dcg_table`: the same
# refusals, then the same x_metrics PROGRAM (arena layout, stage records,
# parameter words, order), run by the shared program runner, and the same
# flag checks after the run. Same kernels: the bits do not move.
from std.python import Python, PythonObject
from bindings.admitted_family_door import admit_family
from bindings.py_balanced import py_call, py_contains
from bindings.family_common_unpack import FAMILY_F32P, family_str
from bindings.family_common_pack import family_array
from bindings.classical_arrays_door import classical_address
from core.host_finite_scan import host_all_finite_f32
from bindings.metrics_exp_common_run import MX_NONE, mx_nan
from bindings.metrics_exp_rank_run import (
    _ROW_BRIER, _ROW_LOGLOSS, _ROW_HINGE_BIN, _ROW_HINGE_MC, _ROW_DCG, _ROW_COVERAGE, _ROW_LRAP, _ROW_RANKLOSS,
    _ROW_DCG_IGNORE_TIES, _RANK_MEAN, _RANK_NDCG_ROW, _RANK_D2_LOG, _RANK_D2_BRIER, _SCAN_INDICATOR, _SCAN_NEGATIVE,
    ProbaShape, row_fold, row_mean_stage, proba_stages, proba_check, dcg_table,
)
from bindings.metrics_program_door import MetricDoorProg, metric_door_run
from bindings.arena_prog_run import MetricArenaOwner
from bindings.metrics_exp_common_unpack import (
    MxLab, MxMat, mx_b, mx_len, mx_truthy, mx_is_str, mx_float, mx_zero_division, mx_undefined, mx_sorted, mx_list,
    mx_missing, mx_selected, mx_codes, mx_map_eq, mx_targets, mx_pos_label, mx_pos_label_ok, mx_scores, mx_weights,
    mx_f32_view, mx_shape_text,
)

# TOMBSTONE: the Python-holding `_Proba` fields beside its shape, `_row_fold`, `_row_mean_stage`,
# `_proba_check` and the stage half of `_put_proba` / `_dcg_table` moved to the typed
# bindings/metrics_exp_rank_run.mojo 2026-11-12 by lane metric-programs-2; code recoverable at 4f5a5ae6d8.


def _pin_weights(mut prog: MetricDoorProg, w: PythonObject):
    """Pin the weights a typed fold declared as the plan's next input."""
    if w is not None:
        prog.pins.append(w)


def _row_mean(mut prog: MetricDoorProg, S: Int, cols: Int, Y: Int, n: Int, kind: Int, w: PythonObject,
              K: Int, Dt: Int, normalize: Bool, half: Bool) raises -> Int:
    var res = row_mean_stage(prog.plan, S, cols, Y, n, kind, w is not None, K, Dt, normalize, half)
    _pin_weights(prog, w)
    return res


# ---------------------------------------------------------------- probabilities


struct _Proba(Movable):
    """`_proba`'s answer: the typed shape and the two arrays the program
    reads (the door's pins: the probabilities, then the codes)."""
    var codes: PythonObject
    var a: PythonObject
    var shape: ProbaShape

    def __init__(out self, codes: PythonObject, a: PythonObject, shape: ProbaShape):
        self.codes = codes
        self.a = a
        self.shape = shape


def _proba(y_true: PythonObject, y_proba: PythonObject, labels: PythonObject, pos_label: PythonObject,
           caller: String, context: PythonObject) raises -> _Proba:
    var true = mx_targets(y_true, caller, context)
    var present = true.classes
    var n = true.n
    var got = mx_f32_view(y_proba, "y_proba", context)
    var view = got[0]
    if got[1] != "<f4":
        raise Error("TypeError: mojolearn " + caller + ": y_proba must have dtype float32; cast explicitly")
    var nd = Int(py=view.ndim)
    var rows = Int(py=view.shape[0]) if nd >= 1 else 0
    var binary = nd == 1 or (nd == 2 and Int(py=view.shape[1]) == 1)
    var codes: PythonObject
    var k: Int
    if binary:
        if mx_len(present) > 2:
            raise Error("The type of the target inferred from y_true is multiclass but should be binary "
                        + "according to the shape of y_prob.")
        var pos: PythonObject
        if pos_label is None:
            if mx_pos_label_ok(true.kind, present):
                pos = PythonObject(1)
            else:
                pos = present[mx_len(present) - 1]
        else:
            pos = mx_pos_label(pos_label, true.kind, present, caller, context)
        codes = mx_map_eq(true, pos, context)
        k = 2
    else:
        var classes = present if labels is None else mx_selected(labels, true.kind, present, context)
        if labels is not None and mx_missing(present, classes):
            raise Error("y_true contains values not belonging to the passed labels")
        classes = mx_sorted(classes)
        k = Int(py=view.shape[1])
        if mx_len(classes) != k:
            raise Error("y_true and y_proba contain different number of classes: " + String(mx_len(classes))
                        + " vs " + String(k))
        codes = mx_codes(true, classes, True, context)
    if rows != n:
        raise Error("y_true and y_proba have different numbers of rows")
    # the rank and emptiness refusals and the cast in one admission (family_view deleted)
    var a = admit_family(view, -1, "y_proba", context).array
    return _Proba(codes, a, ProbaShape(n, k, binary))


def _put_proba(mut prog: MetricDoorProg, p: _Proba) raises -> Tuple[Int, Int]:
    """`_put_proba`: (S, FLAGS); the probabilities pinned, the stages typed."""
    var A = prog.put(p.a)
    return proba_stages(prog.plan, A, p.shape)


def _brier(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """args = (y_true, y_proba, sample_weight, pos_label, labels, scale_by_half, mode)."""
    var caller = String("brier_score_loss")
    var p = _proba(args[0], args[1], args[4], args[3], caller, context)
    var w = mx_weights(args[2], p.shape.n, caller, context)
    var sbh = args[5]
    var half: Bool
    if mx_is_str(sbh) and String(py=sbh) == "auto":
        half = p.shape.binary or p.shape.k < 3
    else:
        half = mx_truthy(sbh)
    var prog = MetricDoorProg()
    var put = _put_proba(prog, p)
    var Y = prog.put_i32(p.codes)
    var res = _row_mean(prog, put[0], p.shape.k, Y, p.shape.n, _ROW_BRIER, w, 0, MX_NONE, True, half)
    var arena = metric_door_run(prog, args[6], context)
    proba_check(arena, put[1], caller)
    return PythonObject(arena.f64(res))


def _d2_proba(args: PythonObject, caller: String, logloss: Bool, context: PythonObject) raises -> PythonObject:
    """d2_log_loss_score / d2_brier_score; args = (y_true, y_proba,
    sample_weight, labels, pos_label, mode)."""
    var p = _proba(args[0], args[1], args[3], args[4], caller, context)
    if p.shape.n < 2:
        mx_undefined("D^2 score is not well-defined with less than two samples.", context)
        return PythonObject(mx_nan())
    var w = mx_weights(args[2], p.shape.n, caller, context)
    var prog = MetricDoorProg()
    var put = _put_proba(prog, p)
    var Y = prog.put_i32(p.codes)
    var f = row_fold(prog.plan, put[0], p.shape.k, Y, p.shape.n, _ROW_LOGLOSS if logloss else _ROW_BRIER, w is not None,
                     0, MX_NONE)
    _pin_weights(prog, w)
    var g = prog.group(Y, p.shape.n, p.shape.k, MX_NONE, 1, f[2], 1)
    var PER = g[0] if w is None else g[1]
    var weighted = 0 if w is None else 1
    var res = prog.alloc(2)
    if logloss:
        prog.stage("rank_epi", 1, [_RANK_D2_LOG, f[0], PER, weighted, p.shape.k, res])
    else:
        prog.stage("rank_epi", 1, [_RANK_D2_BRIER, f[0], p.shape.n, f[1], PER, weighted, p.shape.k, res])
    var arena = metric_door_run(prog, args[5], context)
    proba_check(arena, put[1], caller)
    return PythonObject(arena.f64(res))


def _hinge(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """args = (y_true, pred_decision, labels, sample_weight, mode)."""
    var caller = String("hinge_loss")
    var labels = args[2]
    var true = mx_targets(args[0], caller, context)
    var n = true.n
    var uniq = true.classes if labels is None else mx_sorted(mx_selected(labels, true.kind, true.classes, context))
    var s = mx_scores(args[1], n, caller, -1, "pred_decision", context)
    var w = mx_weights(args[3], n, caller, context)
    var prog = MetricDoorProg()
    var nu = mx_len(uniq)
    if nu > 2:
        if s.ndim != 2:
            raise Error("The shape of pred_decision cannot be 1d array with a multiclass target.")
        if s.cols != nu:
            if labels is None:
                raise Error("Please include all labels in y_true or pass labels as third argument")
            raise Error("The shape of pred_decision is not consistent with the number of classes.")
        var codes = mx_codes(true, uniq, True, context)
        var S = prog.put(s.a)
        var Y = prog.put_i32(codes)
        var res = _row_mean(prog, S, s.cols, Y, n, _ROW_HINGE_MC, w, 0, MX_NONE, True, False)
        var arena = metric_door_run(prog, args[4], context)
        return PythonObject(arena.f64(res))
    if s.ndim == 2 and s.cols != 1:
        raise Error("pred_decision must be 1-D for a binary target")
    var S = prog.put(s.a)
    var Y: Int
    if nu == 2:
        Y = prog.put_i32(mx_map_eq(true, uniq[nu - 1], context))
    else:
        # all-zero codes: program-owned words (was a zeroed Array pinned as a caller buffer)
        Y = prog.put_words(List[Int32](length=n, fill=0))
    var res = _row_mean(prog, S, 1, Y, n, _ROW_HINGE_BIN, w, 0, MX_NONE, True, False)
    var arena = metric_door_run(prog, args[4], context)
    return PythonObject(arena.f64(res))


# ---------------------------------------------------------------- ranking


def _relevance(y_true: PythonObject, y_score: PythonObject, caller: String,
               context: PythonObject) raises -> Tuple[MxMat, MxMat]:
    """`_relevance`: y_true as a finite float32 (n, c) matrix and y_score."""
    var got = mx_f32_view(y_true, "y_true", context)
    var view = got[0]
    var nd = Int(py=view.ndim)
    if nd != 2:
        raise Error("mojolearn " + caller + ": y_true must be a 2-D (n_samples, n_labels) array")
    # the rank and emptiness refusals and the cast in one admission (family_view deleted)
    var y = admit_family(view, 2, "y_true", context).array
    var n = Int(py=view.shape[0])
    var c = Int(py=view.shape[1])
    if not host_all_finite_f32(FAMILY_F32P(unsafe_from_address=Int(py=classical_address(y))), n * c):
        raise Error("mojolearn " + caller + ": y_true must be finite")
    _ = y  # keep the Python object alive through the native calls above
    var s = mx_scores(y_score, n, caller, 2, "y_score", context)
    if s.cols != c:
        raise Error("y_true and y_score have different shape")
    return (MxMat(y, n, c, 2), s^)


def _dcg_table(mut prog: MetricDoorProg, c: Int, log_base: PythonObject, context: PythonObject) raises -> Int:
    """`_dcg_table`: the device discount table's slot."""
    var b = mx_float(log_base, context)
    if not b > 0.0:
        raise Error("math domain error")
    if b == 1.0:
        mx_zero_division("float division by zero")
    return dcg_table(prog.plan, c, b)


def _top_k_param(k: PythonObject) raises -> Int:
    return 0 if k is None else Int(py=py_call(mx_b().int, k))


def _dcg(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """args = (y_true, y_score, k, log_base, sample_weight, ignore_ties, mode)."""
    var caller = String("dcg_score")
    var r = _relevance(args[0], args[1], caller, context)
    var n = r[0].rows
    var c = r[0].cols
    var w = mx_weights(args[4], n, caller, context)
    var prog = MetricDoorProg()
    var S = prog.put(r[1].a)
    var Y = prog.put(r[0].a)
    var Dt = _dcg_table(prog, c, args[3], context)
    var kind = _ROW_DCG_IGNORE_TIES if mx_truthy(args[5]) else _ROW_DCG
    var res = _row_mean(prog, S, c, Y, n, kind, w, _top_k_param(args[2]), Dt, True, False)
    var arena = metric_door_run(prog, args[6], context)
    return PythonObject(arena.f64(res))


def _ndcg(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """args = (y_true, y_score, k, sample_weight, ignore_ties, mode)."""
    var caller = String("ndcg_score")
    var r = _relevance(args[0], args[1], caller, context)
    var n = r[0].rows
    var c = r[0].cols
    if c <= 1:
        raise Error("Computing NDCG is only meaningful when there is more than 1 document. Got "
                    + String(c) + " instead.")
    var w = mx_weights(args[3], n, caller, context)
    var K = _top_k_param(args[2])
    var prog = MetricDoorProg()
    var S = prog.put(r[1].a)
    var Y = prog.put(r[0].a)
    var neg = prog.scan_flag(Y, n * c, _SCAN_NEGATIVE)
    var Dt = _dcg_table(prog, c, PythonObject(2), context)
    var gain = prog.scratch(n)
    var ideal = prog.scratch(n)
    var ratio = prog.scratch(n)
    var kind = _ROW_DCG_IGNORE_TIES if mx_truthy(args[4]) else _ROW_DCG
    prog.stage("row_metric", n, [S, c, Y, gain, kind, K, Dt])
    prog.stage("row_metric", n, [Y, c, Y, ideal, _ROW_DCG, K, Dt])
    prog.stage("rank_epi", n, [_RANK_NDCG_ROW, gain, ideal, n, ratio])
    var zero = prog.scratch(n)
    prog.stage("pair_key", n, [0, 0, zero, 1, 2])
    var W = MX_NONE
    if w is not None:
        W = prog.put(w)
    var tot = prog.group(zero, n, 1, ratio, 1, W, 1)[1]
    var sw = MX_NONE
    if w is not None:
        sw = prog.group(zero, n, 1, MX_NONE, 1, W, 1)[1]
    var res = prog.alloc(2)
    prog.stage("rank_epi", 1, [_RANK_MEAN, tot, n, sw, 1, 0, res])
    var arena = metric_door_run(prog, args[5], context)
    if arena.i32(neg) != 0:
        raise Error("ndcg_score should not be used on negative y_true values.")
    return PythonObject(arena.f64(res))


def _label_ranking(args: PythonObject, kind: Int, caller: String, context: PythonObject) raises -> PythonObject:
    """coverage_error / label_ranking_average_precision_score /
    label_ranking_loss; args = (y_true, y_score, sample_weight, mode)."""
    var r = _relevance(args[0], args[1], caller, context)
    var n = r[0].rows
    var c = r[0].cols
    var w = mx_weights(args[2], n, caller, context)
    var prog = MetricDoorProg()
    var S = prog.put(r[1].a)
    var Y = prog.put(r[0].a)
    var flag = prog.scan_flag(Y, n * c, _SCAN_INDICATOR)
    var res = _row_mean(prog, S, c, Y, n, kind, w, 0, MX_NONE, True, False)
    var arena = metric_door_run(prog, args[3], context)
    if arena.i32(flag) != 0:
        raise Error(caller + " requires a binary label indicator y_true")
    return PythonObject(arena.f64(res))


def metrics_exp_rank_operation(operation: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """The probability, hinge and ranking metrics (argument orders: see
    python/mojolearn/_expansion_metrics.py)."""
    if operation == "family_mexp_rank_brier":
        return _brier(args, context)
    if operation == "family_mexp_rank_d2_log_loss":
        return _d2_proba(args, "d2_log_loss_score", True, context)
    if operation == "family_mexp_rank_d2_brier":
        return _d2_proba(args, "d2_brier_score", False, context)
    if operation == "family_mexp_rank_hinge":
        return _hinge(args, context)
    if operation == "family_mexp_rank_dcg":
        return _dcg(args, context)
    if operation == "family_mexp_rank_ndcg":
        return _ndcg(args, context)
    if operation == "family_mexp_rank_coverage":
        return _label_ranking(args, _ROW_COVERAGE, "coverage_error", context)
    if operation == "family_mexp_rank_lrap":
        return _label_ranking(args, _ROW_LRAP, "label_ranking_average_precision_score", context)
    if operation == "family_mexp_rank_loss":
        return _label_ranking(args, _ROW_RANKLOSS, "label_ranking_loss", context)
    raise Error("RuntimeError: unknown ranking metric " + operation)
