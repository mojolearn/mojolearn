# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE METRICS LANE'S DOOR (docs/lanes/ALGORITHM_EXPANSION_PLAN.md item 1a).

Owned by the `metrics` expansion lane. `mojolearn/__init__.py` imports this
module and makes every name in `__all__` public as `mojolearn.<name>`. The
lane's functions belong to `mojolearn.metrics` (and, for the splitters,
`mojolearn.model_selection`), the way scikit-learn places them, so `__all__`
here is EMPTY on purpose and `_metrics_impl.py` re-exports the functions
below under `mojolearn.metrics`.

HOW THE FUNCTIONS COMPUTE. Every O(n) quantity comes out of ONE binding
entry, `x_metrics_run`, which runs a PROGRAM of units (x_metrics/common.mojo)
over one float32 arena: the GPU binding (`_mojolearn_x_metrics`) launches a
thread per unit, the host binding (`_mojolearn_x_metrics_host`, what
`_backend.binding` returns on a CPU-only install) runs the same units in a
loop. Python lays out the arena, lists the stages and reads the per-class /
per-column Float32 results back.

THE EPILOGUE (DEVIATION 6106). What is left after the O(n) folds is O(classes)
arithmetic on those Float32 results: ratios, averages, a square root. It runs
here, in IEEE binary64 with ONLY + - * / and sqrt, each correctly rounded on
every host Python runs on, in a fixed left-to-right order (and `_portable_math`
for the rare logarithm). That is bitwise the same on every machine by the
IEEE standard, and it is scikit-learn's own precision for the same step.
"""
import array
import ctypes
import numbers
import warnings

from . import _backend
from . import _portable_math as pmath
from ._array import Array
from ._buffer import as_f32_c, addr_ro, all_finite, empty
from ._labels import is_bool

__all__ = []

_BINDING = "_mojolearn_x_metrics"

#: op name -> id; x_metrics/units.mojo `run_unit` holds the same table.
_OPS = dict(group_sort=0, group_sum=1, pair_key=2)
_PARAMS = 14
_NONE = -1


def _binding(numeric_mode):
    if numeric_mode is not None and (
        not isinstance(numeric_mode, str)
        or numeric_mode.strip().lower() not in ("fast", "identical")
    ):
        raise ValueError("numeric_mode must be 'fast' or 'identical'")
    mode = (numeric_mode or _backend.default_mode()).strip().lower()
    return _backend.binding(_BINDING, mode)


class _Prog:
    """One program: an arena layout, the inputs copied into it, and stages."""

    def __init__(self):
        self.size = 0
        self._inputs = []
        self._stages = []
        self.arena = None

    def alloc(self, n):
        off = self.size
        self.size += max(int(n), 0)
        if self.size > 2147483647:
            raise ValueError("mojolearn metrics: the problem exceeds the Int32 arena bound")
        return off

    def put(self, arr):
        """A float32 C-contiguous Array (or anything as_f32_c takes) -> offset."""
        if not (isinstance(arr, Array) and arr.dtype == "<f4" and arr._has_order("C")):
            arr = as_f32_c(arr, ndim=None, name="input")[0]
        off = self.alloc(arr.size)
        self._inputs.append((off, arr))
        return off

    def put_i32(self, codes):
        """int32 values stored as their bits (read with ldi)."""
        if not (isinstance(codes, Array) and codes.dtype == "<i4" and codes._has_order("C")):
            codes = Array.from_list([int(c) for c in codes], "<i4")
        off = self.alloc(codes.size)
        self._inputs.append((off, codes))
        return off

    def stage(self, op, total, *params):
        if len(params) > _PARAMS:
            raise ValueError("x_metrics: too many stage parameters")
        self._stages.append([_OPS[op], int(total)] + [int(v) for v in params]
                            + [0] * (_PARAMS - len(params)))

    def run(self, numeric_mode):
        arena = array.array("f", bytes(4 * max(self.size, 1)))
        base = arena.buffer_info()[0]
        for off, arr in self._inputs:
            if arr.size:
                ctypes.memmove(base + 4 * off, addr_ro(arr, name="input"), 4 * arr.size)
        prog = array.array("i", [v for s in self._stages for v in s] or [0])
        _binding(numeric_mode).x_metrics_run(base, self.size, prog.buffer_info()[0], len(self._stages))
        self.arena = arena
        return self

    def floats(self, off, n):
        """Python floats (exact images of the Float32 results)."""
        return list(self.arena[off:off + n])

    def ints(self, off, n):
        store = array.array("i")
        store.frombytes(self.arena[off:off + n].tobytes())
        return list(store)

    def get(self, off, shape):
        shape = tuple(shape) if isinstance(shape, (tuple, list)) else (int(shape),)
        n = 1
        for s in shape:
            n *= s
        return Array._owned(self.arena[off:off + n], shape, "<f4", "C")


def _group(prog, key, n, m, *, values=_NONE, vstride=1, weights=_NONE, width=1):
    """Stable counting sort of rows by `key` (int32 offset, -1 = dropped),
    then one PairSum per (group, column). Returns (OFF, OUT) offsets."""
    off = prog.alloc(m + 1)
    order = prog.alloc(n)
    out = prog.alloc(m * width)
    prog.stage("group_sort", 1, key, n, m, off, order)
    if m * width:
        prog.stage("group_sum", m * width, off, order, values, vstride, weights, out, width)
    return off, out


# ---------------------------------------------------------------------------
# Shared validation
# ---------------------------------------------------------------------------

def _weights(sample_weight, n, caller):
    from ._metrics_impl import _sample_weight_f32
    return None if sample_weight is None else _sample_weight_f32(sample_weight, n, caller)


def _refuse_multilabel(y, name, caller):
    from .linear_model import _shape_of
    shape = _shape_of(y)
    if len(shape) == 2 and shape[1] != 1:
        raise NotImplementedError(
            f"mojolearn {caller}: {name} is 2-D; multilabel-indicator targets are NOT "
            "IMPLEMENTED (metrics/NOT_IMPLEMENTED.tsv), only binary and multiclass 1-D labels")


def _pair(y_true, y_pred, sample_weight, caller):
    """(true, pred, kind, present, w): single-label targets of one kind."""
    from ._metrics_impl import _classification_encoded, _label_set
    _refuse_multilabel(y_true, "y_true", caller)
    _refuse_multilabel(y_pred, "y_pred", caller)
    true, kind = _classification_encoded(y_true, "y_true")
    pred, pred_kind = _classification_encoded(y_pred, "y_pred")
    if len(true) != len(pred):
        raise ValueError(f"mojolearn {caller}: y_true and y_pred lengths differ")
    if kind != pred_kind:
        raise TypeError(f"mojolearn {caller}: y_true and y_pred must use the same label type")
    if len(true) > 2147483647:
        raise ValueError(f"mojolearn {caller}: at most INT32_MAX rows")
    w = _weights(sample_weight, len(true), caller)
    return true, pred, kind, sorted(_label_set(true) | _label_set(pred)), w


def _codes(labels_seq, order):
    from ._metrics_impl import _label_map
    index = {c: i for i, c in enumerate(order)}
    return _label_map(labels_seq, lambda v: index.get(v, -1))


class _Sums:
    """Per-label tp / pred / true sums and the total, over `order` (every
    label that can occur; the caller selects a prefix). Unweighted sums are
    exact integers (the group sizes); weighted ones are the Float32 PairSum
    of each group's weights, read as Python floats."""

    def __init__(self, true, pred, w, order, numeric_mode):
        n, L = len(true), len(order)
        yt, yp = _codes(true, order), _codes(pred, order)
        prog = _Prog()
        a = prog.put_i32(yt)
        b = prog.put_i32(yp)
        match = prog.alloc(n)
        prog.stage("pair_key", n, a, b, match, L, 1)
        W = _NONE if w is None else prog.put(w)
        groups = [_group(prog, k, n, L, weights=W) for k in (match, a, b)]
        total = None
        if w is not None:
            zero = prog.alloc(n)            # every row in group 0: the total weight
            prog.stage("pair_key", n, a, a, zero, 1, 2)
            groups.append(_group(prog, zero, n, 1, weights=W))
        prog.run(numeric_mode)
        if w is None:
            sums = []
            for off, _ in groups:
                o = prog.ints(off, L + 1)
                sums.append([o[i + 1] - o[i] for i in range(L)])
            total = n
        else:
            sums = [prog.floats(out, L) for _, out in groups[:3]]
            total = prog.floats(groups[3][1], 1)[0]
        self.tp, self.true, self.pred = sums
        self.total = total
        self.weighted = w is not None


def _undefined_warning(msg):
    try:
        from sklearn.exceptions import UndefinedMetricWarning
    except ImportError:
        UndefinedMetricWarning = RuntimeWarning
    warnings.warn(msg, UndefinedMetricWarning, stacklevel=3)


def _zero_division_value(zero_division):
    if isinstance(zero_division, str):
        if zero_division == "warn":
            return 0.0
    elif not is_bool(zero_division) and isinstance(zero_division, numbers.Real):
        if zero_division in (0, 1):
            return float(zero_division)
        if pmath.isnan(float(zero_division)):
            return float("nan")
    raise ValueError("zero_division must be 'warn', 0.0, 1.0 or np.nan")


def _divide(num, den, zero_division, what, warn_list):
    out = []
    bad = False
    zv = _zero_division_value(zero_division)
    for a, b in zip(num, den):
        if b == 0:
            out.append(zv)
            bad = True
        else:
            out.append(a / b)
    if bad and isinstance(zero_division, str) and warn_list is not None:
        warn_list.append(what)
    return out


def _nanaverage(values, weights=None):
    keep = [i for i, v in enumerate(values) if not pmath.isnan(v)]
    if not values or not keep:
        return float("nan")
    if weights is None:
        s = 0.0
        for i in keep:
            s += values[i]
        return s / len(keep)
    sw = 0.0
    for i in keep:
        sw += weights[i]
    if sw == 0:
        s = 0.0
        for i in keep:
            s += values[i]
        return s / len(keep)
    s = 0.0
    for i in keep:
        s += values[i] * weights[i]
    return s / sw


def _set_wise_labels(present, kind, average, labels, pos_label, caller):
    from ._metrics_impl import _classification_labels, _selected_labels
    averages = (None, "micro", "macro", "weighted", "samples", "binary")
    if average not in averages:
        raise ValueError("average has to be one of (None, 'micro', 'macro', 'weighted', 'samples', 'binary')")
    if average == "samples":
        raise NotImplementedError(
            f"mojolearn {caller}: average='samples' applies to multilabel targets, which are "
            "NOT IMPLEMENTED (metrics/NOT_IMPLEMENTED.tsv)")
    if average == "binary":
        if len(present) > 2:
            raise ValueError("Target is multiclass but average='binary'. Please choose another "
                             "average setting, one of [None, 'micro', 'macro', 'weighted'].")
        values, pkind = _classification_labels([pos_label], "pos_label")
        if pkind != kind:
            raise ValueError("pos_label must use the same type as the targets")
        if values[0] not in present and len(present) >= 2:
            raise ValueError(f"pos_label={pos_label} is not a valid label. It should be one of {present}")
        return [values[0]]
    if pos_label not in (None, 1):
        warnings.warn(f"Note that pos_label (set to {pos_label!r}) is ignored when average != 'binary' "
                      f"(got {average!r}). You may use labels=[pos_label] to specify a single positive class.",
                      UserWarning, stacklevel=3)
    return _selected_labels(labels, kind, present) if labels is not None else list(present)


def _label_order(labels, present):
    """`labels` then every present label not in it (sklearn's hstack with
    setdiff1d): the extra labels are counted and not reported."""
    chosen = set(labels)
    return list(labels) + [v for v in present if v not in chosen]


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def multilabel_confusion_matrix(y_true, y_pred, *, sample_weight=None, labels=None,
                                samplewise=False, numeric_mode=None):
    """Per-label one-vs-rest 2x2 confusion matrices, `[[tn, fp], [fn, tp]]`.

    scikit-learn 1.9 `multilabel_confusion_matrix` for binary and multiclass
    1-D targets (strings or integers). Unweighted counts are exact Int64;
    with sample_weight they are Float64 images of Float32 per-label PairSums
    (DEVIATION 6100). Multilabel-indicator input and samplewise=True are
    refused by name (metrics/NOT_IMPLEMENTED.tsv)."""
    if samplewise:
        raise NotImplementedError("mojolearn multilabel_confusion_matrix: samplewise=True applies to "
                                  "multilabel targets, which are NOT IMPLEMENTED")
    from ._metrics_impl import _selected_labels
    true, pred, kind, present, w = _pair(y_true, y_pred, sample_weight, "multilabel_confusion_matrix")
    chosen = list(present) if labels is None else _selected_labels(labels, kind, present)
    s = _Sums(true, pred, w, _label_order(chosen, present), numeric_mode)
    rows = []
    for i in range(len(chosen)):
        tp = s.tp[i]
        fp = s.pred[i] - tp
        fn = s.true[i] - tp
        tn = s.total - tp - fp - fn
        rows.extend([tn, fp, fn, tp])
    if s.weighted:
        return Array.from_list([float(v) for v in rows], "<f8").reshape((len(chosen), 2, 2))
    return Array.from_list([int(v) for v in rows], "<i8").reshape((len(chosen), 2, 2))


def precision_recall_fscore_support(y_true, y_pred, *, beta=1.0, labels=None, pos_label=1,
                                    average=None, warn_for=("precision", "recall", "f-score"),
                                    sample_weight=None, zero_division="warn", numeric_mode=None):
    """scikit-learn 1.9 `precision_recall_fscore_support` for binary and
    multiclass 1-D targets: per-label sums on the device (exact integers, or
    Float32 PairSums of sample_weight), the ratios and averages in the
    binary64 epilogue (DEVIATION 6106). zero_division is 'warn', 0.0, 1.0 or
    np.nan; average='samples' (multilabel) is refused by name."""
    if is_bool(beta) or not isinstance(beta, numbers.Real) or not beta >= 0:
        raise ValueError("beta should be >=0 in the F-beta score")
    _zero_division_value(zero_division)
    true, pred, kind, present, w = _pair(y_true, y_pred, sample_weight, "precision_recall_fscore_support")
    chosen = _set_wise_labels(present, kind, average, labels, pos_label, "precision_recall_fscore_support")
    s = _Sums(true, pred, w, _label_order(chosen, present), numeric_mode)
    k = len(chosen)
    tp, ps, ts = s.tp[:k], s.pred[:k], s.true[:k]
    if average == "micro":
        a = b = c = 0
        for i in range(k):
            a += tp[i]
            b += ps[i]
            c += ts[i]
        tp, ps, ts = [a], [b], [c]
    warned = []
    precision = _divide(tp, ps, zero_division, "precision", warned)
    recall = _divide(tp, ts, zero_division, "recall", warned)
    beta = float(beta)
    if pmath.isinf(beta):
        fscore = recall
    elif beta == 0:
        fscore = precision
    else:
        b2 = beta * beta
        fscore = _divide([(1 + b2) * v for v in tp], [b2 * t + p for t, p in zip(ts, ps)],
                         zero_division, "f-score", warned)
    for what in warned:
        if what in warn_for:
            _undefined_warning(f"{what.capitalize()} is ill-defined and being set to 0.0 in labels "
                               "with no predicted/true samples. Use `zero_division` parameter to "
                               "control this behavior.")
    if average is None:
        dtype = "<f8"
        support = (Array.from_list([float(v) for v in ts], "<f8") if s.weighted
                   else Array.from_list([int(v) for v in ts], "<i8"))
        return (Array.from_list(precision, dtype), Array.from_list(recall, dtype),
                Array.from_list(fscore, dtype), support)
    weights = ts if average == "weighted" else None
    return (_nanaverage(precision, weights), _nanaverage(recall, weights),
            _nanaverage(fscore, weights), None)


def fbeta_score(y_true, y_pred, *, beta, labels=None, pos_label=1, average="binary",
                sample_weight=None, zero_division="warn", numeric_mode=None):
    """scikit-learn 1.9 `fbeta_score` (binary / multiclass 1-D targets); see
    precision_recall_fscore_support."""
    _, _, f, _ = precision_recall_fscore_support(
        y_true, y_pred, beta=beta, labels=labels, pos_label=pos_label, average=average,
        warn_for=("f-score",), sample_weight=sample_weight, zero_division=zero_division,
        numeric_mode=numeric_mode)
    return f


def _prf_weighted(y_true, y_pred, labels, pos_label, average, sample_weight, zero_division,
                  numeric_mode, metric):
    """precision_score / recall_score / f1_score WITH sample_weight: the
    unweighted call keeps the original binding and its bits."""
    names = ("precision", "recall", "f-score")
    out = precision_recall_fscore_support(
        y_true, y_pred, beta=1.0, labels=labels, pos_label=pos_label, average=average,
        warn_for=(names[metric],), sample_weight=sample_weight, zero_division=zero_division,
        numeric_mode=numeric_mode)
    return out[metric]


def jaccard_score(y_true, y_pred, *, labels=None, pos_label=1, average="binary",
                  sample_weight=None, zero_division="warn", numeric_mode=None):
    """scikit-learn 1.9 `jaccard_score`, `tp / (tp + fp + fn)` per label,
    for binary and multiclass 1-D targets; see precision_recall_fscore_support
    for the sums and the epilogue."""
    _zero_division_value(zero_division)
    true, pred, kind, present, w = _pair(y_true, y_pred, sample_weight, "jaccard_score")
    chosen = _set_wise_labels(present, kind, average, labels, pos_label, "jaccard_score")
    s = _Sums(true, pred, w, _label_order(chosen, present), numeric_mode)
    k = len(chosen)
    num = list(s.tp[:k])
    den = [s.pred[i] + s.true[i] - s.tp[i] for i in range(k)]
    if average == "micro":
        a = b = 0
        for i in range(k):
            a += num[i]
            b += den[i]
        num, den = [a], [b]
    warned = []
    jac = _divide(num, den, zero_division, "jaccard", warned)
    if warned:
        _undefined_warning("Jaccard is ill-defined and being set to 0.0 in labels with no true or "
                           "predicted samples. Use `zero_division` parameter to control this behavior.")
    if average is None:
        return Array.from_list(jac, "<f8")
    weights = None
    if average == "weighted":
        weights = [s.true[i] for i in range(k)]
        if not any(weights):
            weights = None
    return _nanaverage(jac, weights) if weights is None else _average_plain(jac, weights)


def _average_plain(values, weights):
    s = sw = 0.0
    for v, w in zip(values, weights):
        s += v * w
        sw += w
    return s / sw


def balanced_accuracy_score(y_true, y_pred, *, sample_weight=None, adjusted=False, numeric_mode=None):
    """scikit-learn 1.9 `balanced_accuracy_score`: the mean per-class recall
    over the classes present in y_true (a class with no true weight is
    dropped with sklearn's warning), chance-adjusted when `adjusted`."""
    true, pred, kind, present, w = _pair(y_true, y_pred, sample_weight, "balanced_accuracy_score")
    s = _Sums(true, pred, w, present, numeric_mode)
    per_class = [s.tp[i] / s.true[i] for i in range(len(present)) if s.true[i] != 0]
    if len(per_class) != len(present):
        warnings.warn("y_pred contains classes not in y_true", stacklevel=2)
    if not per_class:
        return float("nan")
    score = 0.0
    for v in per_class:
        score += v
    score /= len(per_class)
    if adjusted:
        chance = 1 / len(per_class)
        score -= chance
        score /= 1 - chance
    return float(score)


def matthews_corrcoef(y_true, y_pred, *, sample_weight=None, numeric_mode=None):
    """scikit-learn 1.9 `matthews_corrcoef` (binary and multiclass): the
    covariance form over the per-class true / predicted sums and the trace."""
    true, pred, kind, present, w = _pair(y_true, y_pred, sample_weight, "matthews_corrcoef")
    s = _Sums(true, pred, w, present, numeric_mode)
    t_sum = [float(v) for v in s.true]
    p_sum = [float(v) for v in s.pred]
    n_correct = 0.0
    for v in s.tp:
        n_correct += v
    n = 0.0
    for v in p_sum:
        n += v
    tp_dot = pp_dot = tt_dot = 0.0
    for a, b in zip(t_sum, p_sum):
        tp_dot += a * b
        pp_dot += b * b
        tt_dot += a * a
    cov_ytyp = n_correct * n - tp_dot
    cov_ypyp = n * n - pp_dot
    cov_ytyt = n * n - tt_dot
    prod = cov_ypyp * cov_ytyt
    if prod == 0:
        return 0.0
    return float(cov_ytyp / pmath.sqrt(prod))


def _confusion(true, pred, w, order, numeric_mode):
    """The k x k (weighted) confusion matrix over `order`, rows true."""
    n, k = len(true), len(order)
    if k > 4096:
        raise ValueError("mojolearn metrics: at most 4096 labels in a confusion matrix")
    prog = _Prog()
    a = prog.put_i32(_codes(true, order))
    b = prog.put_i32(_codes(pred, order))
    key = prog.alloc(n)
    prog.stage("pair_key", n, a, b, key, k, 0)
    W = _NONE if w is None else prog.put(w)
    off, out = _group(prog, key, n, k * k, weights=W)
    prog.run(numeric_mode)
    if w is None:
        o = prog.ints(off, k * k + 1)
        return [o[i + 1] - o[i] for i in range(k * k)]
    return prog.floats(out, k * k)


def confusion_matrix_weighted(y_true, y_pred, labels, sample_weight, normalize, numeric_mode):
    """confusion_matrix with sample_weight (scikit-learn 1.9): Float64 cells,
    each the Float32 PairSum of its rows' weights; normalize in binary64."""
    from ._metrics_impl import _selected_labels, _label_set
    true, pred, kind, present, w = _pair(y_true, y_pred, sample_weight, "confusion_matrix")
    chosen = _selected_labels(labels, kind, present)
    if not set(chosen).intersection(_label_set(true)):
        raise ValueError("At least one label specified must be in y_true")
    k = len(chosen)
    cm = _confusion(true, pred, w, chosen, numeric_mode)
    cm = _normalize(cm, k, normalize)
    return Array.from_list([float(v) for v in cm], "<f8").reshape((k, k))


def _normalize(cm, k, normalize):
    if normalize is None:
        return cm
    out = list(cm)
    if normalize == "all":
        t = 0.0
        for v in cm:
            t += v
        return [v / t if t else 0.0 for v in cm]
    for i in range(k):
        t = 0.0
        for j in range(k):
            t += cm[i * k + j] if normalize == "true" else cm[j * k + i]
        for j in range(k):
            idx = i * k + j if normalize == "true" else j * k + i
            out[idx] = cm[idx] / t if t else 0.0
    return out


def cohen_kappa_score(y1, y2, *, labels=None, weights=None, sample_weight=None,
                      replace_undefined_by=float("nan"), numeric_mode=None):
    """scikit-learn 1.9 `cohen_kappa_score`: the (weighted) confusion matrix
    over `labels` on the device, the kappa in the binary64 epilogue;
    weights None, 'linear' or 'quadratic'."""
    from ._metrics_impl import _selected_labels, _label_set
    if weights not in (None, "linear", "quadratic"):
        raise ValueError("weights must be None, 'linear' or 'quadratic'")
    true, pred, kind, present, w = _pair(y1, y2, sample_weight, "cohen_kappa_score")
    chosen = _selected_labels(labels, kind, present)
    if not set(chosen).intersection(_label_set(true)):
        raise ValueError("At least one label in `labels` must be present in `y1` (even though "
                         "`cohen_kappa_score` is otherwise agnostic to the order of `y1` and `y2`).")
    k = len(chosen)
    cm = [float(v) for v in _confusion(true, pred, w, chosen, numeric_mode)]
    sum0 = [0.0] * k
    sum1 = [0.0] * k
    for i in range(k):
        for j in range(k):
            sum0[j] += cm[i * k + j]
            sum1[i] += cm[i * k + j]
    den = 0.0
    for v in sum0:
        den += v
    if den == 0:
        _undefined_warning("`y2` contains no labels that are present in both `y1` and `labels`.")
        return replace_undefined_by
    num_k = den_k = 0.0
    for i in range(k):
        for j in range(k):
            if weights is None:
                wm = 0.0 if i == j else 1.0
            elif weights == "linear":
                wm = float(abs(i - j))
            else:
                wm = float((i - j) * (i - j))
            num_k += wm * cm[i * k + j]
            den_k += wm * (sum0[i] * sum1[j] / den)
    if den_k == 0:
        _undefined_warning("`y1`, `y2` and `labels` have only one label in common.")
        return replace_undefined_by
    return float(1 - num_k / den_k)


def hamming_loss(y_true, y_pred, *, sample_weight=None, numeric_mode=None):
    """scikit-learn 1.9 `hamming_loss` for binary / multiclass 1-D targets:
    the (weighted) fraction of mismatched labels."""
    true, pred, kind, present, w = _pair(y_true, y_pred, sample_weight, "hamming_loss")
    s = _Sums(true, pred, w, present, numeric_mode)
    hit = 0.0
    for v in s.tp:
        hit += v
    return float((s.total - hit) / s.total)


def zero_one_loss(y_true, y_pred, *, normalize=True, sample_weight=None, numeric_mode=None):
    """scikit-learn 1.9 `zero_one_loss` for binary / multiclass 1-D targets."""
    if not is_bool(normalize):
        raise ValueError("normalize must be a bool")
    true, pred, kind, present, w = _pair(y_true, y_pred, sample_weight, "zero_one_loss")
    s = _Sums(true, pred, w, present, numeric_mode)
    hit = 0
    for v in s.tp:
        hit += v
    if normalize:
        return float((s.total - hit) / s.total)
    return (s.total - hit) if s.weighted else int(s.total - hit)


def accuracy_count(y_true, y_pred, sample_weight, numeric_mode):
    """accuracy_score(normalize=False): the (weighted) number of matches."""
    true, pred, kind, present, w = _pair(y_true, y_pred, sample_weight, "accuracy_score")
    s = _Sums(true, pred, w, present, numeric_mode)
    hit = 0
    for v in s.tp:
        hit += v
    return float(hit) if s.weighted else int(hit)


def class_likelihood_ratios(y_true, y_pred, *, labels=None, sample_weight=None,
                            replace_undefined_by=float("nan"), numeric_mode=None):
    """scikit-learn 1.9 `class_likelihood_ratios` (binary targets): LR+ and
    LR- from the 2 x 2 (weighted) confusion matrix; `replace_undefined_by`
    np.nan, 1.0 or {'LR+': v, 'LR-': v}."""
    from ._metrics_impl import _selected_labels
    true, pred, kind, present, w = _pair(y_true, y_pred, sample_weight, "class_likelihood_ratios")
    if len(present) > 2:
        raise ValueError("class_likelihood_ratios only supports binary classification problems, "
                         "got targets of type: multiclass")
    nan = float("nan")
    rub = replace_undefined_by
    if isinstance(rub, numbers.Real) and not is_bool(rub) and rub == 1.0:
        rub = {"LR+": 1.0, "LR-": 1.0}
    if isinstance(rub, dict):
        pos, neg = rub.get("LR+"), rub.get("LR-")
        if (set(rub) != {"LR+", "LR-"} or not isinstance(pos, numbers.Real)
                or not isinstance(neg, numbers.Real)
                or not (pos >= 1.0 or pmath.isnan(pos)) or not (0.0 <= neg <= 1.0 or pmath.isnan(neg))):
            raise ValueError("The dictionary passed as `replace_undefined_by` needs to be in the form "
                             "`{'LR+': `value_1`, 'LR-': `value_2`}`")
    elif not (isinstance(rub, float) and pmath.isnan(rub)):
        raise ValueError("replace_undefined_by must be np.nan, 1.0 or a dict")
    chosen = _selected_labels(labels, kind, present) if labels is not None else list(present)
    if len(chosen) == 1:
        chosen = chosen + [v for v in present if v not in chosen]
    k = len(chosen)
    cm = _confusion(true, pred, w, chosen, numeric_mode)
    if k != 2:
        raise ValueError("class_likelihood_ratios needs a 2 x 2 confusion matrix")
    tn, fp, fn, tp = (float(v) for v in cm)
    support_pos, support_neg = tp + fn, tn + fp
    if support_pos == 0:
        _undefined_warning("No samples of the positive class are present in `y_true`.")
        return nan, nan
    if fp == 0:
        _undefined_warning("`positive_likelihood_ratio` is ill-defined and set to `np.nan`.")
        lr_pos = rub if not isinstance(rub, dict) else rub["LR+"]
    else:
        lr_pos = (tp * support_neg) / (fp * support_pos)
    if tn == 0:
        _undefined_warning("`negative_likelihood_ratio` is ill-defined and set to `np.nan`.")
        lr_neg = rub if not isinstance(rub, dict) else rub["LR-"]
    else:
        lr_neg = (fn * support_neg) / (tn * support_pos)
    return float(lr_pos), float(lr_neg)


def classification_report(y_true, y_pred, *, labels=None, target_names=None, sample_weight=None,
                          digits=2, output_dict=False, zero_division="warn", numeric_mode=None):
    """scikit-learn 1.9 `classification_report` for binary / multiclass 1-D
    targets, from precision_recall_fscore_support."""
    from ._metrics_impl import _classification_encoded, _label_set
    true, _ = _classification_encoded(y_true, "y_true")
    pred, _ = _classification_encoded(y_pred, "y_pred")
    present = sorted(_label_set(true) | _label_set(pred))
    labels_given = labels is not None
    chosen = list(labels) if labels_given else present
    micro_is_accuracy = not labels_given or set(chosen) >= set(present)
    if target_names is not None and len(target_names) != len(chosen):
        raise ValueError(f"Number of classes, {len(chosen)}, does not match size of target_names, "
                         f"{len(target_names)}. Try specifying the labels parameter")
    names = [str(t) for t in target_names] if target_names is not None else [str(c) for c in chosen]
    headers = ["precision", "recall", "f1-score", "support"]
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=chosen, average=None,
                                                 sample_weight=sample_weight,
                                                 zero_division=zero_division, numeric_mode=numeric_mode)
    rows = list(zip(names, p.tolist(), r.tolist(), f.tolist(), s.tolist()))
    averages = (["micro avg"] if not micro_is_accuracy else []) + ["macro avg", "weighted avg"]
    report = {}
    for name, a, b, c, d in rows:
        report[name] = dict(zip(headers, (a, b, c, d)))
    total = 0
    for v in s.tolist():
        total += v
    avg_rows = []
    if micro_is_accuracy:
        acc = precision_recall_fscore_support(y_true, y_pred, labels=chosen, average="micro",
                                              sample_weight=sample_weight, zero_division=zero_division,
                                              numeric_mode=numeric_mode)[0]
        report["accuracy"] = acc
    for avg in averages:
        a, b, c, _ = precision_recall_fscore_support(y_true, y_pred, labels=chosen,
                                                     average=avg.split()[0],
                                                     sample_weight=sample_weight,
                                                     zero_division=zero_division, numeric_mode=numeric_mode)
        report[avg] = dict(zip(headers, (a, b, c, total)))
        avg_rows.append((avg, a, b, c, total))
    if output_dict:
        return report
    width = max([len(n) for n in names] + [len("weighted avg"), digits])
    head_fmt = "{:>{width}s} " + " {:>9}" * len(headers)
    out = head_fmt.format("", *headers, width=width) + "\n\n"
    row_fmt = "{:>{width}s} " + " {:>9.{digits}f}" * 3 + " {:>9}\n"
    for name, a, b, c, d in rows:
        out += row_fmt.format(name, a, b, c, d, width=width, digits=digits)
    out += "\n"
    if micro_is_accuracy:
        acc_fmt = "{:>{width}s} " + " {:>9.{digits}}" * 2 + " {:>9.{digits}f}" + " {:>9}\n"
        out += acc_fmt.format("accuracy", "", "", report["accuracy"], total, width=width, digits=digits)
    for avg, a, b, c, d in avg_rows:
        out += row_fmt.format(avg, a, b, c, d, width=width, digits=digits)
    return out
