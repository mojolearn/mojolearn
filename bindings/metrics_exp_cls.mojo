# SPDX-License-Identifier: Apache-2.0
# The classification metrics of python/mojolearn/_expansion_metrics.py, one
# native call each: multilabel_confusion_matrix,
# precision_recall_fscore_support, fbeta_score, jaccard_score,
# balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score,
# hamming_loss, zero_one_loss, class_likelihood_ratios and
# classification_report.
#
# This owner is the Python `_Sums`, `_confusion` and their tails: the same
# refusals (wording and order), then the same x_metrics PROGRAM with the same
# arena layout, the same stage records in the same order and the same
# parameter words, run by the shared program runner. classification_report
# builds the one program (the sums and the four set-wise epilogues) its
# Python memo made, and the support total's program. Same programs, same
# kernels: the bits do not move.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_contains, py_setitem
from bindings.family_common import FAMILY_F32P, FAMILY_I32P, family_array, family_str, family_repr, family_call_kw
from bindings.classical_arrays import classical_address
from bindings.metrics_exp_common import (
    MX_NONE, MxProg, MxArena, MxLab, MxPair, mx_run, mx_b, mx_len, mx_truthy, mx_eq, mx_is_bool, mx_is_str,
    mx_is_real, mx_float, mx_nan, mx_inf, mx_isnan, mx_isinf, mx_raise_obj, mx_zero_division, mx_warn, mx_undefined,
    mx_sorted, mx_set, mx_list, mx_label_order, mx_str_list, mx_encode, mx_selected, mx_codes, mx_pair,
    mx_list_labels, mx_union_sorted, mx_zero_division_value, mx_attr,
)

# cm_epi kinds (x_metrics/cm_epi.mojo, op 59)
comptime _CM_MLCM = 0
comptime _CM_MCC = 1
comptime _CM_COUNT = 2
comptime _CM_NORM = 3
comptime _CM_KAPPA = 4
comptime _CM_CLR = 5
comptime _CM_TOTAL = 6
comptime _CM_WSUM_LEAF = 7
comptime _CM_WSUM_FOLD = 8
comptime _WSUM_BLOCK = 256


struct MxSums(Movable):
    """`_Sums`: the program with the per-label tp / true / pred groupings
    (and the weighted total's), their count slots and the staged epilogues."""
    var prog: MxProg
    var n: Int
    var L: Int
    var weighted: Bool
    var match_key: Int
    var W: Int
    var g_off: List[Int]
    var g_out: List[Int]
    var cnt: List[Int]
    var eos: List[Int]

    def __init__(out self, var prog: MxProg, n: Int, L: Int, weighted: Bool):
        self.prog = prog^
        self.n = n
        self.L = L
        self.weighted = weighted
        self.match_key = MX_NONE
        self.W = MX_NONE
        self.g_off = List[Int]()
        self.g_out = List[Int]()
        self.cnt = List[Int]()
        self.eos = List[Int]()

    def col(self) -> Int:
        return 1 if self.weighted else 0

    def grp(self, i: Int) -> Int:
        """groups[i][col]: the OFF table (unweighted) or the PairSums (weighted)."""
        return self.g_out[i] if self.weighted else self.g_off[i]

    def add_epi(mut self, kind: Int, k: Int, avg: Int, zd: Int, fmode: Int, b2: Float64,
                context: PythonObject) raises:
        """One `cls_epi` stage (x_metrics/cls_epi.mojo)."""
        var col = self.col()
        var bo = 0
        if fmode == 2:
            var words = family_array(Python.tuple(4), "<i4", context)
            var dst = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(words)))
            dst[0] = 1 + b2
            dst[1] = b2
            bo = self.prog.put_i32(words)
        var eo = self.prog.alloc(8 + 6 * max(self.L, 1))
        self.prog.stage("cls_epi", 1, [self.grp(0), self.grp(1), self.grp(2), k, col, kind, avg, zd, eo, fmode, bo])
        self.eos.append(eo)


def _sums(pair: MxPair, order: PythonObject, context: PythonObject) raises -> MxSums:
    """`_Sums.__init__` up to its epilogues: the codes, the match key, the
    three count groupings (and the weighted total's), the count slots."""
    var n = pair.true.n
    var L = mx_len(order)
    var yt = mx_codes(pair.true, order, False, context)
    var yp = mx_codes(pair.pred, order, False, context)
    var s = MxSums(MxProg(context), n, L, pair.w is not None)
    var a = s.prog.put_i32(yt)
    var b = s.prog.put_i32(yp)
    var matched = s.prog.scratch(n)
    s.prog.stage("pair_key", n, [a, b, matched, L, 1])
    var W = MX_NONE
    if pair.w is not None:
        W = s.prog.put(pair.w)
    s.match_key = matched
    s.W = W
    var keys: List[Int] = [matched, a, b]
    for i in range(3):  # driver-loop: over the three count groupings
        var g = s.prog.group(keys[i], n, L, MX_NONE, 1, W, 1)
        s.g_off.append(g[0])
        s.g_out.append(g[1])
    if pair.w is not None:
        var zero = s.prog.scratch(n)
        s.prog.stage("pair_key", n, [a, a, zero, 1, 2])
        var g = s.prog.group(zero, n, 1, MX_NONE, 1, W, 1)
        s.g_off.append(g[0])
        s.g_out.append(g[1])
    else:
        for i in range(3):  # driver-loop: over the three count groupings
            s.cnt.append(s.prog.counts(s.g_off[i], L))
    return s^


def _wsum_sf64(mut s: MxSums, k: Int) raises -> Int:
    """`_wsum_sf64`: the weighted COUNT's hit and total as soft binary64 sums."""
    var n = s.n
    var nb = max(1, (n + _WSUM_BLOCK - 1) // _WSUM_BLOCK)
    var slot = s.prog.scratch(4 * nb)
    s.prog.stage("cm_epi", nb, [_CM_WSUM_LEAF, s.match_key, s.W, n, slot, k])
    var stride = 1
    while stride < nb:  # driver-loop: over the fold levels (log2 of the block count)
        s.prog.stage("cm_epi", (nb - stride + 2 * stride - 1) // (2 * stride), [_CM_WSUM_FOLD, slot, stride, nb])
        stride *= 2
    return slot


def _cm_stage(mut s: MxSums, k: Int, kind: Int, size: Int) raises -> Int:
    """`_cm_stage`: a cm_epi tail over the groupings; the output slot."""
    var col = s.col()
    var out = s.prog.alloc(max(size, 1))
    var tp = s.grp(0)
    var tr = s.grp(1)
    var pr = s.grp(2)
    var tot = s.g_out[3] if s.weighted else s.n
    if kind == _CM_COUNT and s.weighted:
        tot = _wsum_sf64(s, k)
    if kind == _CM_MLCM:
        if k:
            s.prog.stage("cm_epi", k, [kind, tp, tr, pr, k, col, tot, out])
    elif kind == _CM_MCC:
        s.prog.stage("cm_epi", 1, [kind, tp, tr, pr, k, col, out])
    else:
        s.prog.stage("cm_epi", 1, [kind, tp, k, col, tot, out])
    return out


def _epi_avg(average: PythonObject) raises -> Int:
    """`_epi_codes`' AVG: 0 per label, 1 micro, 3 weighted, 2 the plain mean."""
    if average is None:
        return 0
    if mx_is_str(average):
        var a = String(py=average)
        if a == "micro":
            return 1
        if a == "weighted":
            return 3
    return 2


def _epi_zd(zero_division: PythonObject, context: PythonObject) raises -> Int:
    var zv = mx_zero_division_value(zero_division, context)
    return 2 if mx_isnan(zv) else Int(zv)


def _set_wise_labels(pair: MxPair, average: PythonObject, labels: PythonObject, pos_label: PythonObject,
                     caller: String, context: PythonObject) raises -> PythonObject:
    """`_set_wise_labels`."""
    var averages = Python.tuple(PythonObject(None), PythonObject("micro"), PythonObject("macro"),
                                PythonObject("weighted"), PythonObject("samples"), PythonObject("binary"))
    if not py_contains(averages, average):
        raise Error("average has to be one of (None, 'micro', 'macro', 'weighted', 'samples', 'binary')")
    if mx_is_str(average) and String(py=average) == "samples":
        raise Error("NotImplementedError: mojolearn " + caller + ": average='samples' applies to multilabel "
                    + "targets, which are NOT IMPLEMENTED (metrics/NOT_IMPLEMENTED.tsv)")
    if mx_is_str(average) and String(py=average) == "binary":
        if mx_len(pair.present) > 2:
            raise Error("Target is multiclass but average='binary'. Please choose another "
                        + "average setting, one of [None, 'micro', 'macro', 'weighted'].")
        var listed = mx_list_labels(Python.list(pos_label), "pos_label", False, context)
        if listed[1] != pair.kind:
            raise Error("pos_label must use the same type as the targets")
        var value = listed[0][0]
        if not py_contains(pair.present, value) and mx_len(pair.present) >= 2:
            raise Error("pos_label=" + family_str(pos_label) + " is not a valid label. It should be one of "
                        + family_str(pair.present))
        return Python.list(value)
    if not py_contains(Python.tuple(PythonObject(None), PythonObject(1)), pos_label):
        mx_warn("Note that pos_label (set to " + family_repr(pos_label) + ") is ignored when average != 'binary' (got "
                + family_repr(average) + "). You may use labels=[pos_label] to specify a single positive class.",
                mx_b().UserWarning, 2)
    if labels is not None:
        return mx_selected(labels, pair.kind, pair.present, context)
    return mx_list(pair.present)


def _check_beta(beta: PythonObject, context: PythonObject) raises -> Float64:
    if mx_is_bool(beta) or not mx_is_real(beta, context) or not mx_truthy(beta >= PythonObject(0)):
        raise Error("beta should be >=0 in the F-beta score")
    return mx_float(beta)


def _fmode(fb: Float64) -> Int:
    return 0 if mx_isinf(fb) else (1 if fb == 0 else 2)


def _prf_warn(bad: Int, warn_for: PythonObject, zero_division: PythonObject, context: PythonObject) raises:
    if not mx_is_str(zero_division):
        return
    var bits: List[Int] = [1, 2, 4]
    var whats: List[String] = ["precision", "recall", "f-score"]
    var names: List[String] = ["Precision", "Recall", "F-score"]
    for i in range(3):  # small-loop(3: the three metric names): fixed
        if (bad & bits[i]) != 0 and py_contains(warn_for, PythonObject(whats[i])):
            mx_undefined(names[i] + " is ill-defined and being set to 0.0 in labels with no predicted/true "
                         + "samples. Use `zero_division` parameter to control this behavior.", context)


def _prf_result(s: MxSums, arena: MxArena, eo: Int, k: Int, average: PythonObject,
                context: PythonObject) raises -> PythonObject:
    """The answer of one set-wise epilogue: per-label arrays, or the scalars."""
    if average is None:
        var p = arena.words(eo + 8, 2 * k, Python.tuple(PythonObject(k)), "<f8", context)
        var r = arena.words(eo + 8 + 2 * k, 2 * k, Python.tuple(PythonObject(k)), "<f8", context)
        var f = arena.words(eo + 8 + 4 * k, 2 * k, Python.tuple(PythonObject(k)), "<f8", context)
        var support = _support_array(s, arena, k, context)
        return Python.tuple(p, r, f, support)
    return Python.tuple(PythonObject(arena.f64(eo + 2)), PythonObject(arena.f64(eo + 4)),
                        PythonObject(arena.f64(eo + 6)), PythonObject(None))


def _support_array(s: MxSums, arena: MxArena, k: Int, context: PythonObject) raises -> PythonObject:
    """`s.true[:k]` as float64 (weighted PairSums) or int64 (counts)."""
    if s.weighted:
        var out = family_array(Python.tuple(PythonObject(k)), "<f8", context)
        var dst = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(out)))
        for j in range(k):  # driver-loop: over the reported labels
            dst[j] = arena.f32(s.g_out[1] + j)
        return out
    var out = family_array(Python.tuple(PythonObject(k)), "<i8", context)
    var dst = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(out)))
    for j in range(k):  # driver-loop: over the reported labels
        dst[j] = Int64(arena.i32(s.cnt[1] + j))
    return out


def _prfs(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """precision_recall_fscore_support; args = (y_true, y_pred, beta, labels,
    pos_label, average, warn_for, sample_weight, zero_division, mode)."""
    var caller = String("precision_recall_fscore_support")
    var fb = _check_beta(args[2], context)
    _ = mx_zero_division_value(args[8], context)
    var pair = mx_pair(args[0], args[1], args[7], caller, context)
    var average = args[5]
    var chosen = _set_wise_labels(pair, average, args[3], args[4], caller, context)
    var order = mx_label_order(chosen, pair.present)
    var k = mx_len(chosen)
    var s = _sums(pair, order, context)
    s.add_epi(2, k, _epi_avg(average), _epi_zd(args[8], context), _fmode(fb), fb * fb, context)
    var arena = mx_run(s.prog, args[9], context)
    var eo = s.eos[0]
    _prf_warn(arena.i32(eo), args[6], args[8], context)
    return _prf_result(s, arena, eo, k, average, context)


def _fbeta(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """fbeta_score: precision_recall_fscore_support's F with warn_for=("f-score",);
    args = (y_true, y_pred, beta, labels, pos_label, average, sample_weight, zero_division, mode)."""
    var full = Python.tuple(args[0], args[1], args[2], args[3], args[4], args[5],
                            Python.tuple(PythonObject("f-score")), args[6], args[7], args[8])
    return _prfs(full, context)[2]


def _jaccard(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """args = (y_true, y_pred, labels, pos_label, average, sample_weight, zero_division, mode)."""
    var caller = String("jaccard_score")
    _ = mx_zero_division_value(args[6], context)
    var pair = mx_pair(args[0], args[1], args[5], caller, context)
    var average = args[4]
    var chosen = _set_wise_labels(pair, average, args[2], args[3], caller, context)
    var k = mx_len(chosen)
    var s = _sums(pair, mx_label_order(chosen, pair.present), context)
    s.add_epi(0, k, _epi_avg(average), _epi_zd(args[6], context), 0, 0.0, context)
    var arena = mx_run(s.prog, args[7], context)
    var eo = s.eos[0]
    if arena.i32(eo) != 0 and mx_is_str(args[6]):
        mx_undefined("Jaccard is ill-defined and being set to 0.0 in labels with no true or "
                     + "predicted samples. Use `zero_division` parameter to control this behavior.", context)
    if average is None:
        return arena.words(eo + 8, 2 * k, Python.tuple(PythonObject(k)), "<f8", context)
    return PythonObject(arena.f64(eo + 2))


def _balanced_accuracy(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """args = (y_true, y_pred, sample_weight, adjusted, mode)."""
    var pair = mx_pair(args[0], args[1], args[2], "balanced_accuracy_score", context)
    var L = mx_len(pair.present)
    var s = _sums(pair, pair.present, context)
    s.add_epi(1, L, 0, 1 if mx_truthy(args[3]) else 0, 0, 0.0, context)
    var arena = mx_run(s.prog, args[4], context)
    var eo = s.eos[0]
    if arena.i32(eo + 1) != L:
        mx_warn("y_pred contains classes not in y_true", mx_b().UserWarning, 2)
    return PythonObject(arena.f64(eo + 2))


def _matthews(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """args = (y_true, y_pred, sample_weight, mode)."""
    var pair = mx_pair(args[0], args[1], args[2], "matthews_corrcoef", context)
    var s = _sums(pair, pair.present, context)
    var tail = _cm_stage(s, mx_len(pair.present), _CM_MCC, 4)
    var arena = mx_run(s.prog, args[3], context)
    if arena.i32(tail) != 0:
        raise Error("math domain error")
    return PythonObject(arena.f64(tail + 2))


def _mlcm(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """args = (y_true, y_pred, sample_weight, labels, samplewise, mode)."""
    if mx_truthy(args[4]):
        raise Error("NotImplementedError: mojolearn multilabel_confusion_matrix: samplewise=True applies to "
                    + "multilabel targets, which are NOT IMPLEMENTED")
    var pair = mx_pair(args[0], args[1], args[2], "multilabel_confusion_matrix", context)
    var chosen = mx_list(pair.present) if args[3] is None else mx_selected(args[3], pair.kind, pair.present, context)
    var k = mx_len(chosen)
    var s = _sums(pair, mx_label_order(chosen, pair.present), context)
    var tail = _cm_stage(s, k, _CM_MLCM, 8 * k)
    var arena = mx_run(s.prog, args[5], context)
    var shape = Python.tuple(PythonObject(k), PythonObject(2), PythonObject(2))
    var _keep_ret_tail = arena.words(tail, 8 * k, shape, "<f8" if s.weighted else "<i8", context)
    _ = tail  # keep the Python object alive through the native call above
    return _keep_ret_tail^


def _cm_count(args: PythonObject, caller: String, context: PythonObject) raises -> Tuple[MxArena, Int, Bool]:
    """`_cm_count`; args = (y_true, y_pred, sample_weight, ...): the arena,
    the COUNT tail's 11-word slot and whether it is weighted."""
    var pair = mx_pair(args[0], args[1], args[2], caller, context)
    var s = _sums(pair, pair.present, context)
    var out = _cm_stage(s, mx_len(pair.present), _CM_COUNT, 11)
    var weighted = s.weighted
    return (mx_run(s.prog, args[mx_len(args) - 1], context), out, weighted)


def _hamming(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """args = (y_true, y_pred, sample_weight, mode)."""
    var r = _cm_count(args, "hamming_loss", context)
    if r[0].i32(r[1]) != 0:
        mx_zero_division("float division by zero")
    return PythonObject(r[0].f64(r[1] + 6))


def _zero_one(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """args = (y_true, y_pred, sample_weight, normalize, mode)."""
    if not mx_is_bool(args[3]):
        raise Error("normalize must be a bool")
    var r = _cm_count(args, "zero_one_loss", context)
    var out = r[1]
    if mx_truthy(args[3]):
        if r[0].i32(out) != 0:
            mx_zero_division("float division by zero" if r[2] else "division by zero")
        return PythonObject(r[0].f64(out + 6))
    if r[2]:
        return PythonObject(r[0].f64(out + 4))
    return PythonObject(r[0].i32(out + 10))


# ---------------------------------------------------------------- the confusion matrix


struct MxConf(Movable):
    """`_confusion`'s program: the k*k groupings over `order`."""
    var prog: MxProg
    var cells: Int
    var col: Int

    def __init__(out self, var prog: MxProg, cells: Int, col: Int):
        self.prog = prog^
        self.cells = cells
        self.col = col


def _confusion(pair: MxPair, order: PythonObject, context: PythonObject) raises -> MxConf:
    var n = pair.true.n
    var k = mx_len(order)
    if k > 4096:
        raise Error("mojolearn metrics: at most 4096 labels in a confusion matrix")
    var prog = MxProg(context)
    var a = prog.put_i32(mx_codes(pair.true, order, False, context))
    var b = prog.put_i32(mx_codes(pair.pred, order, False, context))
    var key = prog.scratch(n)
    prog.stage("pair_key", n, [a, b, key, k, 0])
    var W = MX_NONE
    if pair.w is not None:
        W = prog.put(pair.w)
    var g = prog.group(key, n, k * k, MX_NONE, 1, W, 1)
    var weighted = pair.w is not None
    return MxConf(prog^, g[1] if weighted else g[0], 1 if weighted else 0)


def _kappa(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """args = (y1, y2, labels, weights, sample_weight, replace_undefined_by, mode)."""
    var weights = args[3]
    var wm: Int
    if weights is None:
        wm = 0
    elif mx_is_str(weights) and String(py=weights) == "linear":
        wm = 1
    elif mx_is_str(weights) and String(py=weights) == "quadratic":
        wm = 2
    else:
        raise Error("weights must be None, 'linear' or 'quadratic'")
    var pair = mx_pair(args[0], args[1], args[4], "cohen_kappa_score", context)
    var chosen = mx_selected(args[2], pair.kind, pair.present, context)
    if mx_len(py_call(mx_set(chosen).intersection, pair.true.classes)) == 0:
        raise Error("At least one label in `labels` must be present in `y1` (even though "
                    + "`cohen_kappa_score` is otherwise agnostic to the order of `y1` and `y2`).")
    var k = mx_len(chosen)
    var c = _confusion(pair, chosen, context)
    var out = c.prog.want(c.prog.alloc(4), 4)
    var sc = c.prog.scratch(8 * k + 2)
    var phases: List[Int] = [0, 1, 2, 3]
    var units: List[Int] = [k, 1, k, 1]
    for i in range(4):  # small-loop(4: the four kappa stages): fixed
        if units[i]:
            c.prog.stage("cm_epi", units[i], [_CM_KAPPA, c.cells, k, c.col, phases[i], wm, out, sc])
    var arena = mx_run(c.prog, args[6], context)
    var flags = arena.i32(out)
    if flags & 1:
        mx_undefined("`y2` contains no labels that are present in both `y1` and `labels`.", context)
        return args[5]
    if flags & 2:
        mx_undefined("`y1`, `y2` and `labels` have only one label in common.", context)
        return args[5]
    return PythonObject(arena.f64(out + 2))


def _rub_ok(value: PythonObject, low: Float64, high: Float64, context: PythonObject) raises -> Bool:
    """`isinstance(v, Real)` and (low <= v <= high or v is NaN)."""
    if not mx_is_real(value, context):
        return False
    var v = mx_float(value)
    return (low <= v and v <= high) or mx_isnan(v)


def _likelihood_ratios(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """args = (y_true, y_pred, labels, sample_weight, replace_undefined_by, mode)."""
    var b = mx_b()
    var pair = mx_pair(args[0], args[1], args[3], "class_likelihood_ratios", context)
    if mx_len(pair.present) > 2:
        raise Error("class_likelihood_ratios only supports binary classification problems, "
                    + "got targets of type: multiclass")
    var rub = args[4]
    var is_dict = False
    if mx_is_real(rub, context) and not mx_is_bool(rub) and mx_float(rub) == 1.0:
        rub = Python.dict()
        py_setitem(rub, "LR+", PythonObject(1.0))
        py_setitem(rub, "LR-", PythonObject(1.0))
    if _is_dict(rub):
        is_dict = True
        var keys = mx_set(Python.list(PythonObject("LR+"), PythonObject("LR-")))
        var pos = py_call(rub.get, PythonObject("LR+"))
        var neg = py_call(rub.get, PythonObject("LR-"))
        if (not mx_eq(mx_set(rub), keys) or not _rub_ok(pos, 1.0, mx_inf(), context)
                or not _rub_ok(neg, 0.0, 1.0, context)):
            raise Error("The dictionary passed as `replace_undefined_by` needs to be in the form "
                        + "`{'LR+': `value_1`, 'LR-': `value_2`}`")
    elif not (_is_float(rub) and mx_isnan(mx_float(rub))):
        raise Error("replace_undefined_by must be np.nan, 1.0 or a dict")
    var chosen = mx_list(pair.present) if args[2] is None else mx_selected(args[2], pair.kind, pair.present, context)
    if mx_len(chosen) == 1:
        chosen = mx_label_order(chosen, pair.present)
    var k = mx_len(chosen)
    if k > 4096:
        raise Error("mojolearn metrics: at most 4096 labels in a confusion matrix")
    if k != 2:
        raise Error("class_likelihood_ratios needs a 2 x 2 confusion matrix")
    var c = _confusion(pair, chosen, context)
    var out = c.prog.want(c.prog.alloc(6), 6)
    c.prog.stage("cm_epi", 1, [_CM_CLR, c.cells, c.col, out])
    var arena = mx_run(c.prog, args[5], context)
    var flags = arena.i32(out)
    if flags & 1:
        mx_undefined("No samples of the positive class are present in `y_true`.", context)
        return Python.tuple(PythonObject(mx_nan()), PythonObject(mx_nan()))
    var lr_pos = PythonObject(arena.f64(out + 2))
    var lr_neg = PythonObject(arena.f64(out + 4))
    if flags & 2:
        mx_undefined("`positive_likelihood_ratio` is ill-defined and set to `np.nan`.", context)
        lr_pos = rub[PythonObject("LR+")] if is_dict else rub
    if flags & 4:
        mx_undefined("`negative_likelihood_ratio` is ill-defined and set to `np.nan`.", context)
        lr_neg = rub[PythonObject("LR-")] if is_dict else rub
    return Python.tuple(py_call(b.float, lr_pos), py_call(b.float, lr_neg))


def _is_dict(value: PythonObject) raises -> Bool:
    return Bool(py=py_call(mx_b().isinstance, value, mx_b().dict))


def _is_float(value: PythonObject) raises -> Bool:
    return Bool(py=py_call(mx_b().isinstance, value, mx_b().float))


# ---------------------------------------------------------------- classification_report


def _support_total(s: MxSums, arena: MxArena, k: Int, mode: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_support_total`: the report's support total (cm_epi TOTAL) in its
    own program, from the k supports' words (Float32 PairSum images or the
    Int32 counts, the copies `put` / `put_i32` made of the support array)."""
    var prog = MxProg(context)
    var src: Int
    if s.weighted:
        var w = family_array(Python.tuple(PythonObject(k)), "<f4", context)
        var dst = FAMILY_F32P(unsafe_from_address=Int(py=classical_address(w)))
        for j in range(k):  # driver-loop: over the reported labels
            dst[j] = Float32(arena.f32(s.g_out[1] + j))
        src = prog.put(w)
    else:
        var c = family_array(Python.tuple(PythonObject(k)), "<i4", context)
        var dst = FAMILY_I32P(unsafe_from_address=Int(py=classical_address(c)))
        for j in range(k):  # driver-loop: over the reported labels
            dst[j] = Int32(arena.i32(s.cnt[1] + j))
        src = prog.put_i32(c)
    var out = prog.want(prog.alloc(2), 2)
    prog.stage("cm_epi", 1, [_CM_TOTAL, src, k, 1 if s.weighted else 0, out])
    var res = mx_run(prog, mode, context)
    if s.weighted:
        return PythonObject(res.f64(out))
    return PythonObject(res.i64(out))


def _report(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """classification_report; args = (y_true, y_pred, labels, target_names,
    sample_weight, digits, output_dict, zero_division, mode)."""
    var b = mx_b()
    var labels = args[2]
    var target_names = args[3]
    var digits = args[5]
    var zero_division = args[7]
    var mode = args[8]
    var t0 = mx_encode(args[0], "y_true", context)
    var p0 = mx_encode(args[1], "y_pred", context)
    var present = mx_union_sorted(t0.classes, p0.classes)
    var labels_given = labels is not None
    var chosen = mx_list(labels) if labels_given else present
    var micro_is_accuracy = not labels_given or Bool(py=py_call(mx_set(chosen).issuperset, present))
    if target_names is not None and mx_len(target_names) != mx_len(chosen):
        raise Error("Number of classes, " + String(mx_len(chosen)) + ", does not match size of target_names, "
                    + String(mx_len(target_names)) + ". Try specifying the labels parameter")
    var names = mx_str_list(target_names) if target_names is not None else mx_str_list(chosen)
    var headers = Python.list(PythonObject("precision"), PythonObject("recall"), PythonObject("f1-score"),
                              PythonObject("support"))
    # precision_recall_fscore_support(labels=chosen, average=...) four
    # times over one encoded pair: one program, the four epilogues
    var caller = String("precision_recall_fscore_support")
    _ = mx_zero_division_value(zero_division, context)
    var pair = mx_pair(args[0], args[1], args[4], caller, context)
    var sel = _set_wise_labels(pair, PythonObject(None), chosen, PythonObject(1), caller, context)
    var order = mx_label_order(sel, pair.present)
    var k = mx_len(sel)
    var s = _sums(pair, order, context)
    var zd = _epi_zd(zero_division, context)
    var avg_codes: List[Int] = [0, 1, 2, 3]
    for i in range(4):  # small-loop(4: the report's four averages): fixed
        s.add_epi(2, k, avg_codes[i], zd, 2, 1.0, context)
    var arena = mx_run(s.prog, mode, context)
    var all_for = Python.tuple(PythonObject("precision"), PythonObject("recall"), PythonObject("f-score"))
    _prf_warn(arena.i32(s.eos[0]), all_for, zero_division, context)
    var per = _prf_result(s, arena, s.eos[0], k, PythonObject(None), context)
    var rows = mx_list(py_call(b.zip, names, py_call(per[0].tolist), py_call(per[1].tolist), py_call(per[2].tolist),
                               py_call(per[3].tolist)))
    var report = Python.dict()
    var n_rows = mx_len(rows)
    for i in range(n_rows):  # driver-loop: over the report's label rows
        var row = rows[i]
        py_setitem(report, row[0], py_call(b.dict, py_call(b.zip, headers,
                   Python.tuple(row[1], row[2], row[3], row[4]))))
    var total = _support_total(s, arena, k, mode, context)
    if micro_is_accuracy:
        _prf_warn(arena.i32(s.eos[1]), all_for, zero_division, context)
        py_setitem(report, "accuracy", PythonObject(arena.f64(s.eos[1] + 2)))
    var avg_names = Python.list()
    var avg_slots = List[Int]()
    if not micro_is_accuracy:
        _ = py_call(avg_names.append, PythonObject("micro avg"))
        avg_slots.append(1)
    _ = py_call(avg_names.append, PythonObject("macro avg"))
    avg_slots.append(2)
    _ = py_call(avg_names.append, PythonObject("weighted avg"))
    avg_slots.append(3)
    var avg_rows = Python.list()
    for i in range(len(avg_slots)):  # small-loop(3: the report's average rows): at most three
        var eo = s.eos[avg_slots[i]]
        _prf_warn(arena.i32(eo), all_for, zero_division, context)
        var a = PythonObject(arena.f64(eo + 2))
        var r = PythonObject(arena.f64(eo + 4))
        var f = PythonObject(arena.f64(eo + 6))
        py_setitem(report, avg_names[i], py_call(b.dict, py_call(b.zip, headers, Python.tuple(a, r, f, total))))
        _ = py_call(avg_rows.append, Python.tuple(avg_names[i], a, r, f, total))
    if mx_truthy(args[6]):
        return report
    var widths = mx_list(py_call(b.map, b.len, names)) + Python.list(PythonObject(12), digits)
    var width = py_call(b.max, widths)
    var head_fmt = PythonObject("{:>{width}s}  {:>9} {:>9} {:>9} {:>9}")
    var out = family_call_kw(mx_attr(head_fmt, "format"), _fmt_kw(width, PythonObject(None)), PythonObject(""), headers[0],
                             headers[1], headers[2], headers[3]) + PythonObject("\n\n")
    var row_fmt = PythonObject("{:>{width}s}  {:>9.{digits}f} {:>9.{digits}f} {:>9.{digits}f} {:>9}\n")
    for i in range(n_rows):  # driver-loop: over the report's label rows
        var row = rows[i]
        out = out + family_call_kw(mx_attr(row_fmt, "format"), _fmt_kw(width, digits), row[0], row[1], row[2], row[3], row[4])
    out = out + PythonObject("\n")
    if micro_is_accuracy:
        var acc_fmt = PythonObject("{:>{width}s}  {:>9.{digits}} {:>9.{digits}} {:>9.{digits}f} {:>9}\n")
        out = out + family_call_kw(mx_attr(acc_fmt, "format"), _fmt_kw(width, digits), PythonObject("accuracy"), PythonObject(""),
                                   PythonObject(""), report["accuracy"], total)
    for i in range(mx_len(avg_rows)):  # small-loop(3: the report's average rows): at most three
        var row = avg_rows[i]
        out = out + family_call_kw(mx_attr(row_fmt, "format"), _fmt_kw(width, digits), row[0], row[1], row[2], row[3], row[4])
    return out


def _fmt_kw(width: PythonObject, digits: PythonObject) raises -> PythonObject:
    var keywords = Python.dict()
    py_setitem(keywords, "width", width)
    if digits is not None:
        py_setitem(keywords, "digits", digits)
    return keywords


def metrics_exp_cls_operation(operation: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """The classification metrics; each shell passes its arguments in its
    signature's order (see python/mojolearn/_expansion_metrics.py)."""
    if operation == "family_mexp_cls_prfs":
        return _prfs(args, context)
    if operation == "family_mexp_cls_fbeta":
        return _fbeta(args, context)
    if operation == "family_mexp_cls_jaccard":
        return _jaccard(args, context)
    if operation == "family_mexp_cls_balanced_accuracy":
        return _balanced_accuracy(args, context)
    if operation == "family_mexp_cls_matthews":
        return _matthews(args, context)
    if operation == "family_mexp_cls_mlcm":
        return _mlcm(args, context)
    if operation == "family_mexp_cls_hamming":
        return _hamming(args, context)
    if operation == "family_mexp_cls_zero_one":
        return _zero_one(args, context)
    if operation == "family_mexp_cls_kappa":
        return _kappa(args, context)
    if operation == "family_mexp_cls_likelihood_ratios":
        return _likelihood_ratios(args, context)
    if operation == "family_mexp_cls_report":
        return _report(args, context)
    raise Error("RuntimeError: unknown classification metric " + operation)
