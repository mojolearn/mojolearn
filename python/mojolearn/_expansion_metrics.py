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

THE EPILOGUE (DEVIATION 6106, narrowed 2026-09-28). What is left after the
O(n) folds is O(classes) arithmetic on those Float32 results: ratios,
averages, a square root. It runs here, in IEEE binary64 with ONLY + - * / and
sqrt, each correctly rounded on every host Python runs on, in a fixed
left-to-right order (and `_portable_math` for the rare logarithm), and only on
O(classes), O(outputs) or O(candidates x folds) scalars; `**`, `pow` and libm
transcendentals are not allowed. Any epilogue whose term count scales with
rows, curve points, contingency cells or k^2 * d runs in
x_metrics/epilogue.mojo under the same operation rules; its Python spelling
here is only the fallback and the MOJOLEARN_METRICS_EPILOGUE=python reference
arm. That is bitwise the same on every machine by the IEEE standard, and it
is scikit-learn's own precision for the same step.
"""
import array
import ctypes
import itertools
import math as _math
import numbers
import operator
import warnings

from . import _backend
from . import _portable_math as pmath
from ._array import Array
from ._buffer import as_f32_c, addr_ro, all_finite, empty
from ._labels import is_bool

__all__ = []

_BINDING = "_mojolearn_x_metrics"

#: op name -> id; x_metrics/units.mojo `run_unit` holds the same table.
_OPS = dict(group_sort=0, group_sum=1, pair_key=2, reg_term=3, col_sort=4, wpercentile=5, col_max=6, bin_curve=7, row_metric=8, row_centroid_dist=9, permute=10,
            fold_rows=36, rows64=41, strat_codes=45)
_PARAMS = 14
_NONE = -1


def _fsum(values):
    """`_portable_math.fsum` (the exact sum, rounded once to nearest/even),
    by CPython's `math.fsum` whenever that is finite: for finite binary64
    inputs `math.fsum` is that same correctly rounded sum (Shewchuk's exact
    partials; IEEE binary64 on every host Python runs on), so the bits are
    the same and a million terms cost milliseconds instead of seconds. A
    zero sum is +0.0, as the portable one returns; anything non-finite or an
    intermediate overflow goes to the portable sum, which decides it."""
    vals = values if isinstance(values, list) else list(values)
    try:
        s = _math.fsum(vals)
    except (OverflowError, ValueError):
        return pmath.fsum(vals)
    if s - s != 0:
        return pmath.fsum(vals)
    return s if s != 0 else 0.0


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
        #: the declared output ranges (`want`); None = the whole arena
        #: less the inputs and scratch (`_skip`), which never come back
        self._outs = None
        self._skip = []

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
        self._skip.append((off, off + arr.size))
        return off

    def scratch(self, n):
        """`alloc` for a slot only the device reads (a sort order, a key
        column): it never comes back (lane metrics-apple2)."""
        off = self.alloc(n)
        if n > 0:
            self._skip.append((off, off + int(n)))
        return off

    def put_i32(self, codes):
        """int32 values stored as their bits (read with ldi)."""
        if not (isinstance(codes, Array) and codes.dtype == "<i4" and codes._has_order("C")):
            codes = Array.from_list([int(c) for c in codes], "<i4")
        off = self.alloc(codes.size)
        self._inputs.append((off, codes))
        self._skip.append((off, off + codes.size))
        return off

    def stage(self, op, total, *params):
        if len(params) > _PARAMS:
            raise ValueError("x_metrics: too many stage parameters")
        self._stages.append([_OPS[op], int(total)] + [int(v) for v in params]
                            + [0] * (_PARAMS - len(params)))

    def run(self, numeric_mode):
        return _execute(self, numeric_mode)

    def want(self, off, n, count=None, mult=1):
        """Declare [off, off + n) an OUTPUT the caller reads (lane
        metrics-apple2). A program that declares any output brings back
        only its outputs from the device (the Apple GPU's download is its
        slowest link), and every read below refuses a word outside them,
        on every backend, so a missing declaration fails loudly. `count`
        (an Int32 slot the program writes, itself declared) bounds the
        range to its first mult * count words."""
        if self._outs is None:
            self._outs = []
        if n > 0:
            self._outs.append((int(off), int(off) + int(n), -1 if count is None else int(count), int(mult)))
        return off

    def _check(self, off, n):
        if n <= 0:
            return
        if self._outs is None:
            for lo, hi in self._skip:
                if off < hi and lo < off + n:
                    raise AssertionError(f"x_metrics: arena [{off}, {off + n}) was read but is an input "
                                         "or scratch slot, which never comes back")
            return
        for lo, hi, cn, mult in self._outs:
            if cn >= 0 and self.arena is not None:
                hi = lo + max(0, min(hi - lo, mult * self.ints(cn, 1)[0]))
            if lo <= off and off + n <= hi:
                return
        raise AssertionError(f"x_metrics: arena [{off}, {off + n}) was read but never declared an output")

    def _download(self):
        """The disjoint ascending [lo, hi) ranges the device brings back:
        the declared outputs, else the arena less the inputs and scratch;
        None = the whole arena."""
        if self._outs is not None:
            merged = []
            for lo, hi, cn, mult in sorted(o for o in self._outs if o[2] < 0):
                if merged and lo <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], hi)
                else:
                    merged.append([lo, hi, -1, 1])
            return merged + [list(o) for o in self._outs if o[2] >= 0]
        if not self._skip:
            return None
        merged = []
        at = 0
        for lo, hi in sorted(self._skip):
            if lo > at:
                merged.append([at, lo])
            at = max(at, hi)
        if at < self.size:
            merged.append([at, self.size])
        return [r + [-1, 1] for r in merged]

    def floats(self, off, n):
        """Python floats (exact images of the Float32 results)."""
        self._check(off, n)
        return list(self.arena[off:off + n])

    def ints(self, off, n):
        self._check(off, n)
        store = array.array("i")
        store.frombytes(self.arena[off:off + n].tobytes())
        return list(store)

    def words(self, off, n, code):
        """The 4 * n bytes of [off, off + n) as an array of `code` ("i",
        or "q" for the Int64 rows of `fold_rows` and `permute` wide)."""
        self._check(off, n)
        store = array.array(code)
        store.frombytes(memoryview(self.arena)[off:off + n].cast("B"))
        return store

    def get(self, off, shape):
        shape = tuple(shape) if isinstance(shape, (tuple, list)) else (int(shape),)
        n = 1
        for s in shape:
            n *= s
        self._check(off, n)
        return Array._owned(self.arena[off:off + n], shape, "<f4", "C")


def _execute(prog, numeric_mode):
    """Run a program on the x_metrics binding (GPU, or the host binding on
    a CPU-only install). A module-level function, not only a method, so the
    lane selector (tools/lane_select.py follows file-local functions, not
    classes) sees every caller reach `x_metrics_run`."""
    arena = array.array("f", bytes(4 * max(prog.size, 1)))
    base = arena.buffer_info()[0]
    for off, arr in prog._inputs:
        if arr.size:
            ctypes.memmove(base + 4 * off, addr_ro(arr, name="input"), 4 * arr.size)
    stages = array.array("i", [v for s in prog._stages for v in s] or [0])
    b = _binding(numeric_mode)
    merged = prog._download()
    run_out = getattr(b, "x_metrics_run_out", None) if merged is not None else None
    if run_out is not None:
        outs = array.array("i", [v for r in merged for v in r] or [0, 0, -1, 1])
        run_out(base, prog.size, stages.buffer_info()[0], len(prog._stages), outs.buffer_info()[0], len(merged))
    else:
        b.x_metrics_run(base, prog.size, stages.buffer_info()[0], len(prog._stages))
    prog.arena = arena
    return prog


def _group(prog, key, n, m, *, values=_NONE, vstride=1, weights=_NONE, width=1):
    """Stable counting sort of rows by `key` (int32 offset, -1 = dropped),
    then one PairSum per (group, column). Returns (OFF, OUT) offsets."""
    off = prog.alloc(m + 1)
    order = prog.scratch(n)
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
        match = prog.scratch(n)
        prog.stage("pair_key", n, a, b, match, L, 1)
        W = _NONE if w is None else prog.put(w)
        groups = [_group(prog, k, n, L, weights=W) for k in (match, a, b)]
        total = None
        if w is not None:
            zero = prog.scratch(n)          # every row in group 0: the total weight
            prog.stage("pair_key", n, a, a, zero, 1, 2)
            groups.append(_group(prog, zero, n, 1, weights=W))
        _execute(prog, numeric_mode)
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
        if chance == 1:
            return float("nan")     # numpy's 0 / 0 in scikit-learn, chosen by value here
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
    key = prog.scratch(n)
    prog.stage("pair_key", n, a, b, key, k, 0)
    W = _NONE if w is None else prog.put(w)
    off, out = _group(prog, key, n, k * k, weights=W)
    _execute(prog, numeric_mode)
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


# ---------------------------------------------------------------------------
# Regression (scikit-learn 1.9 sklearn/metrics/_regression.py)
# ---------------------------------------------------------------------------

_TERM = dict(sq=0, abs=1, sqlog=2, ape=3, pinball=4, tweedie=5, diff=6)


def _f32(x):
    """The Float32 nearest a binary64 value (round to nearest even)."""
    import struct
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


class _Reg:
    """Validated regression targets: (n, D) Float32, finite, same shape."""

    def __init__(self, y_true, y_pred, sample_weight, multioutput, caller, *, variance_ok=False):
        from ._metrics_impl import _shape_of, _is_float64
        from ._buffer import materialize_f32_lists
        arrays = []
        for name, v in (("y_true", y_true), ("y_pred", y_pred)):
            a = materialize_f32_lists(v, name)[0]
            if a.dtype != "<f4":
                raise TypeError(f"mojolearn {caller}: {name} must have dtype float32; cast explicitly before scoring")
            if a.ndim == 1:
                a = a.reshape((a.shape[0], 1))
            if a.ndim != 2 or a.shape[0] == 0 or a.shape[1] == 0:
                raise ValueError(f"mojolearn {caller}: {name} must be a nonempty (n,) or (n, n_outputs) array")
            a = as_f32_c(a, ndim=2, name=name)[0]
            if not all_finite(a):
                raise ValueError(f"mojolearn {caller}: {name} contains NaN or infinity")
            arrays.append(a)
        self.y, self.p = arrays
        if self.y.shape != self.p.shape:
            raise ValueError(f"mojolearn {caller}: y_true and y_pred have different shapes "
                             f"{self.y.shape} and {self.p.shape}")
        self.n, self.D = self.y.shape
        self.w = _weights(sample_weight, self.n, caller)
        allowed = ("raw_values", "uniform_average") + (("variance_weighted",) if variance_ok else ())
        if isinstance(multioutput, str):
            if multioutput not in allowed:
                raise ValueError(f"Allowed 'multioutput' string values are {allowed}. "
                                 f"You provided multioutput={multioutput!r}")
            self.mo = multioutput
        elif multioutput is None:
            self.mo = "uniform_average"
        else:
            vals = [float(v) for v in flatten_mo(multioutput)]
            if self.D == 1:
                raise ValueError("Custom weights are useful only in multi-output cases.")
            if len(vals) != self.D:
                raise ValueError("There must be equally many custom weights "
                                 f"({len(vals)}) as outputs ({self.D}).")
            self.mo = vals
        self.caller = caller

    def column_means(self, kinds, numeric_mode, *, pred_broadcast=None, scalar=None):
        """For each (kind, Y-source) in `kinds`: the per-column (weighted)
        mean of the term, binary64 from the Float32 sums (DEVIATION 6106)."""
        prog = _Prog()
        Y = prog.put(self.y)
        P = prog.put(self.p) if pred_broadcast is None else prog.put(
            Array.from_list([_f32(v) for v in pred_broadcast], "<f4"))
        S = prog.put(Array.from_list([_f32(0.0 if scalar is None else scalar)], "<f4"))
        W = _NONE if self.w is None else prog.put(self.w)
        n, D = self.n, self.D
        zero = prog.scratch(n)
        prog.stage("pair_key", n, 0, 0, zero, 1, 2)
        outs = []
        for kind in kinds:
            term = prog.scratch(n * D)
            prog.stage("reg_term", n * D, Y, P, term, D, _TERM[kind], S, 0 if pred_broadcast is None else 1)
            outs.append(_group(prog, zero, n, 1, values=term, vstride=D, weights=W, width=D)[1])
        sw_out = _group(prog, zero, n, 1, weights=W)[1] if self.w is not None else None
        _execute(prog, numeric_mode)
        den = float(n) if self.w is None else prog.floats(sw_out, 1)[0]
        return [[v / den for v in prog.floats(o, D)] for o in outs]

    def percentile(self, values_off_fn, rank, numeric_mode, *, average=True):
        """Per-column weighted percentile of the Float32 array a program
        stage writes (`values_off_fn(prog) -> offset`, n x D)."""
        prog = _Prog()
        V = values_off_fn(prog)
        n, D = self.n, self.D
        order = prog.alloc(n * D)
        prog.stage("col_sort", D, V, n, D, order)
        W = _NONE if self.w is None else prog.put(self.w)
        R = prog.put(Array.from_list([_f32(rank)], "<f4"))
        out = prog.want(prog.alloc(D), D)
        cdf = prog.alloc(n * D)
        for c in range(D):
            prog.want(cdf + c * n, 1)
        prog.stage("wpercentile", D, V, n, D, order, W, R, 1 if average else 0, out, cdf)
        _execute(prog, numeric_mode)
        flags = [prog.ints(cdf + c * n, 1)[0] for c in range(D)]
        vals = prog.floats(out, D)
        return [float("nan") if fl == -1 else v for v, fl in zip(vals, flags)]

    def average(self, errors):
        if self.mo == "raw_values":
            return Array.from_list([float(v) for v in errors], "<f8")
        if self.mo == "uniform_average":
            s = 0.0
            for v in errors:
                s += v
            return float(s / len(errors))
        s = sw = 0.0
        for v, w in zip(errors, self.mo):
            s += v * w
            sw += w
        return float(s / sw)


def flatten_mo(values):
    from ._labels import flatten_labels
    return flatten_labels(values)


def _mean_error(kind, y_true, y_pred, sample_weight, multioutput, numeric_mode, caller, *, root=False,
                scalar=None):
    r = _Reg(y_true, y_pred, sample_weight, multioutput, caller)
    errors = r.column_means([kind], numeric_mode, scalar=scalar)[0]
    if root:
        errors = [pmath.sqrt(v) for v in errors]
    return r.average(errors)


def regression_error_options(name, y_true, y_pred, sample_weight, multioutput, numeric_mode):
    """mean_squared_error / mean_absolute_error / root_mean_squared_error
    with sample_weight, multioutput or 2-D targets (lane/metrics); the 1-D
    unweighted call keeps its original kernel and bits."""
    kind = {"mean_squared_error": "sq", "mean_absolute_error": "abs", "root_mean_squared_error": "sq"}[name]
    return _mean_error(kind, y_true, y_pred, sample_weight, multioutput, numeric_mode, name,
                       root=name == "root_mean_squared_error")


def mean_squared_log_error(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average",
                           numeric_mode=None):
    """scikit-learn 1.9 `mean_squared_log_error`: the mean of
    `(log1p(y) - log1p(p))^2` (portable log1p, IDENTITY_PATHS row 51);
    values <= -1 are refused as scikit-learn refuses them."""
    _refuse_log_domain(y_true, y_pred, "Mean Squared Logarithmic Error")
    return _mean_error("sqlog", y_true, y_pred, sample_weight, multioutput, numeric_mode,
                       "mean_squared_log_error")


def root_mean_squared_log_error(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average",
                                numeric_mode=None):
    """scikit-learn 1.9 `root_mean_squared_log_error` (per-output root, then averaged)."""
    _refuse_log_domain(y_true, y_pred, "Root Mean Squared Logarithmic Error")
    return _mean_error("sqlog", y_true, y_pred, sample_weight, multioutput, numeric_mode,
                       "root_mean_squared_log_error", root=True)


def _refuse_log_domain(y_true, y_pred, what):
    from ._buffer import materialize_f32_lists
    for v in (y_true, y_pred):
        a = materialize_f32_lists(v, "input")[0]
        if a.size and a.min() <= -1:
            raise ValueError(f"{what} cannot be used when targets contain values less than or equal to -1.")


def mean_absolute_percentage_error(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average",
                                   numeric_mode=None):
    """scikit-learn 1.9 `mean_absolute_percentage_error`: `|p - y| / max(|y|,
    eps)` with numpy's float64 epsilon, averaged (not multiplied by 100)."""
    return _mean_error("ape", y_true, y_pred, sample_weight, multioutput, numeric_mode,
                       "mean_absolute_percentage_error")


def _check_alpha(alpha):
    if is_bool(alpha) or not isinstance(alpha, numbers.Real) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be a real number in [0, 1]")
    return float(alpha)


def mean_pinball_loss(y_true, y_pred, *, sample_weight=None, alpha=0.5, multioutput="uniform_average",
                      numeric_mode=None):
    """scikit-learn 1.9 `mean_pinball_loss` (alpha is rounded to Float32)."""
    return _mean_error("pinball", y_true, y_pred, sample_weight, multioutput, numeric_mode,
                       "mean_pinball_loss", scalar=_check_alpha(alpha))


def median_absolute_error(y_true, y_pred, *, multioutput="uniform_average", sample_weight=None,
                          numeric_mode=None):
    """scikit-learn 1.9 `median_absolute_error`: the per-output median of
    |y - p| (the weighted percentile at 50 with averaging when weighted, the
    same answer as numpy's median when not), from a stable device sort."""
    r = _Reg(y_true, y_pred, sample_weight, multioutput, "median_absolute_error")

    def terms(prog):
        Y, P = prog.put(r.y), prog.put(r.p)
        out = prog.alloc(r.n * r.D)
        prog.stage("reg_term", r.n * r.D, Y, P, out, r.D, _TERM["abs"], Y, 0)
        return out
    return r.average(r.percentile(terms, 50.0, numeric_mode))


def max_error(y_true, y_pred, *, numeric_mode=None):
    """scikit-learn 1.9 `max_error` (single output)."""
    r = _Reg(y_true, y_pred, None, "uniform_average", "max_error")
    if r.D != 1:
        raise ValueError("Multioutput not supported in max_error")
    prog = _Prog()
    Y, P = prog.put(r.y), prog.put(r.p)
    t = prog.scratch(r.n)
    prog.stage("reg_term", r.n, Y, P, t, 1, _TERM["abs"], Y, 0)
    out = prog.want(prog.alloc(1), 1)
    prog.stage("col_max", 1, t, r.n, 1, out)
    _execute(prog, numeric_mode)
    return float(prog.floats(out, 1)[0])


def _assemble(num, den, mo, force_finite):
    scores = []
    for a, b in zip(num, den):
        if not force_finite:
            if b != 0:
                scores.append(1 - a / b)
            elif a == 0:
                scores.append(float("nan"))
            else:
                scores.append(float("-inf"))
        elif b != 0 and a != 0:
            scores.append(1 - a / b)
        elif a != 0:
            scores.append(0.0)
        else:
            scores.append(1.0)
    if mo == "raw_values":
        return Array.from_list(scores, "<f8")
    if mo == "uniform_average":
        weights = None
    elif mo == "variance_weighted":
        weights = den if any(v != 0 for v in den) else None
    else:
        weights = mo
    if weights is None:
        s = 0.0
        for v in scores:
            s += v
        return float(s / len(scores))
    s = sw = 0.0
    for v, w in zip(scores, weights):
        s += v * w
        sw += w
    return float(s / sw)


def explained_variance_score(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average",
                             force_finite=True, numeric_mode=None):
    """scikit-learn 1.9 `explained_variance_score`: `1 - Var(y - p) / Var(y)`
    with (weighted) means, two device passes (the means, then the centered
    squares); force_finite=False returns scikit-learn's NaN / -inf by value."""
    r = _Reg(y_true, y_pred, sample_weight, multioutput, "explained_variance_score", variance_ok=True)
    diff_mean, y_mean = _diff_and_y_means(r, numeric_mode)
    num = _centered(r, "diff", diff_mean, numeric_mode)
    den = _centered(r, "y", y_mean, numeric_mode)
    return _assemble(num, den, r.mo, force_finite)


def _diff_and_y_means(r, numeric_mode):
    """Per-column (weighted) means of y - p and of y."""
    prog = _Prog()
    Y, P = prog.put(r.y), prog.put(r.p)
    W = _NONE if r.w is None else prog.put(r.w)
    n, D = r.n, r.D
    zero = prog.scratch(n)
    prog.stage("pair_key", n, 0, 0, zero, 1, 2)
    diff = prog.scratch(n * D)
    prog.stage("reg_term", n * D, Y, P, diff, D, _TERM["diff"], Y, 0)
    a = _group(prog, zero, n, 1, values=diff, vstride=D, weights=W, width=D)[1]
    b = _group(prog, zero, n, 1, values=Y, vstride=D, weights=W, width=D)[1]
    sw = _group(prog, zero, n, 1, weights=W)[1] if r.w is not None else None
    _execute(prog, numeric_mode)
    den = float(n) if r.w is None else prog.floats(sw, 1)[0]
    return [v / den for v in prog.floats(a, D)], [v / den for v in prog.floats(b, D)]


def _centered(r, source, means, numeric_mode, *, mean=True):
    """Per-column (weighted) mean (or sum) of (v - mean_c)^2, v = y - p or y."""
    prog = _Prog()
    Y, P = prog.put(r.y), prog.put(r.p)
    W = _NONE if r.w is None else prog.put(r.w)
    n, D = r.n, r.D
    zero = prog.scratch(n)
    prog.stage("pair_key", n, 0, 0, zero, 1, 2)
    if source == "diff":
        V = prog.scratch(n * D)
        prog.stage("reg_term", n * D, Y, P, V, D, _TERM["diff"], Y, 0)
    else:
        V = Y
    M = prog.put(Array.from_list([_f32(m) for m in means], "<f4"))
    sq = prog.scratch(n * D)
    prog.stage("reg_term", n * D, V, M, sq, D, _TERM["sq"], Y, 1)
    out = _group(prog, zero, n, 1, values=sq, vstride=D, weights=W, width=D)[1]
    sw = _group(prog, zero, n, 1, weights=W)[1] if r.w is not None else None
    _execute(prog, numeric_mode)
    if not mean:
        return prog.floats(out, D)
    den = float(n) if r.w is None else prog.floats(sw, 1)[0]
    return [v / den for v in prog.floats(out, D)]


def r2_score_options(y_true, y_pred, sample_weight, multioutput, force_finite, numeric_mode):
    """r2_score with multioutput, 2-D targets or force_finite=False
    (lane/metrics): `sum w (y - p)^2` over `sum w (y - avg)^2` per output;
    the 1-D default call keeps its original kernel and bits."""
    r = _Reg(y_true, y_pred, sample_weight, multioutput, "r2_score", variance_ok=True)
    if r.n < 2:
        _undefined_warning("R^2 score is not well-defined with less than two samples.")
        return float("nan")
    _, y_mean = _diff_and_y_means(r, numeric_mode)
    num = _centered_sse(r, numeric_mode)
    den = _centered(r, "y", y_mean, numeric_mode, mean=False)
    return _assemble(num, den, r.mo, force_finite)


def _centered_sse(r, numeric_mode):
    prog = _Prog()
    Y, P = prog.put(r.y), prog.put(r.p)
    W = _NONE if r.w is None else prog.put(r.w)
    n, D = r.n, r.D
    zero = prog.scratch(n)
    prog.stage("pair_key", n, 0, 0, zero, 1, 2)
    sq = prog.scratch(n * D)
    prog.stage("reg_term", n * D, Y, P, sq, D, _TERM["sq"], Y, 0)
    out = _group(prog, zero, n, 1, values=sq, vstride=D, weights=W, width=D)[1]
    _execute(prog, numeric_mode)
    return prog.floats(out, D)


def _tweedie_domain(r, power, caller):
    msg = f"Mean Tweedie deviance error with power={power} can only be used on "
    ymin, pmin = r.y.min(), r.p.min()
    if power < 0:
        if pmin <= 0:
            raise ValueError(msg + "strictly positive y_pred.")
    elif power == 0:
        pass
    elif 1 <= power < 2:
        if ymin < 0 or pmin <= 0:
            raise ValueError(msg + "non-negative y and strictly positive y_pred.")
    elif power >= 2:
        if ymin <= 0 or pmin <= 0:
            raise ValueError(msg + "strictly positive y and y_pred.")
    else:
        raise ValueError(f"mojolearn {caller}: power in (0, 1) is not a Tweedie distribution")


def mean_tweedie_deviance(y_true, y_pred, *, sample_weight=None, power=0, numeric_mode=None):
    """scikit-learn 1.9 `mean_tweedie_deviance` (single output; the power is
    rounded to Float32; pow/log are the portable row-12 functions)."""
    if is_bool(power) or not isinstance(power, numbers.Real):
        raise ValueError("power must be a real number")
    r = _Reg(y_true, y_pred, sample_weight, "uniform_average", "mean_tweedie_deviance")
    if r.D != 1:
        raise ValueError("Multioutput not supported in mean_tweedie_deviance")
    _tweedie_domain(r, power, "mean_tweedie_deviance")
    return r.column_means(["tweedie"], numeric_mode, scalar=float(power))[0][0]


def mean_poisson_deviance(y_true, y_pred, *, sample_weight=None, numeric_mode=None):
    """scikit-learn 1.9 `mean_poisson_deviance` (Tweedie power 1)."""
    return mean_tweedie_deviance(y_true, y_pred, sample_weight=sample_weight, power=1, numeric_mode=numeric_mode)


def mean_gamma_deviance(y_true, y_pred, *, sample_weight=None, numeric_mode=None):
    """scikit-learn 1.9 `mean_gamma_deviance` (Tweedie power 2)."""
    return mean_tweedie_deviance(y_true, y_pred, sample_weight=sample_weight, power=2, numeric_mode=numeric_mode)


def d2_tweedie_score(y_true, y_pred, *, sample_weight=None, power=0, numeric_mode=None):
    """scikit-learn 1.9 `d2_tweedie_score`: `1 - dev(y, p) / dev(y, avg y)`."""
    if is_bool(power) or not isinstance(power, numbers.Real):
        raise ValueError("power must be a real number")
    r = _Reg(y_true, y_pred, sample_weight, "uniform_average", "d2_tweedie_score")
    if r.D != 1:
        raise ValueError("Multioutput not supported in d2_tweedie_score")
    if r.n < 2:
        _undefined_warning("D^2 score is not well-defined with less than two samples.")
        return float("nan")
    _tweedie_domain(r, power, "d2_tweedie_score")
    num = r.column_means(["tweedie"], numeric_mode, scalar=float(power))[0][0]
    y_avg = r.column_means(["diff"], numeric_mode, pred_broadcast=[0.0])[0][0]
    den = r.column_means(["tweedie"], numeric_mode, pred_broadcast=[y_avg], scalar=float(power))[0][0]
    return float(1 - num / den)


def d2_pinball_score(y_true, y_pred, *, sample_weight=None, alpha=0.5, multioutput="uniform_average",
                     numeric_mode=None):
    """scikit-learn 1.9 `d2_pinball_score`: the pinball loss against the
    (weighted, averaged) alpha-quantile of y_true per output."""
    alpha = _check_alpha(alpha)
    r = _Reg(y_true, y_pred, sample_weight, multioutput, "d2_pinball_score")
    if r.n < 2:
        _undefined_warning("D^2 score is not well-defined with less than two samples.")
        return float("nan")
    num = r.column_means(["pinball"], numeric_mode, scalar=alpha)[0]
    quant = r.percentile(lambda prog: prog.put(r.y), alpha * 100, numeric_mode)
    den = r.column_means(["pinball"], numeric_mode, pred_broadcast=quant, scalar=alpha)[0]
    return _assemble(num, den, r.mo, True)


def d2_absolute_error_score(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average",
                            numeric_mode=None):
    """scikit-learn 1.9 `d2_absolute_error_score` (d2_pinball_score at alpha 0.5)."""
    return d2_pinball_score(y_true, y_pred, sample_weight=sample_weight, alpha=0.5,
                            multioutput=multioutput, numeric_mode=numeric_mode)


# ---------------------------------------------------------------------------
# Ranking and probabilistic scores (scikit-learn 1.9 _ranking.py, _classification.py)
# ---------------------------------------------------------------------------

_ROW = dict(topk=0, brier=1, logloss=2, hinge_bin=3, hinge_mc=4, dcg=5, coverage=6, lrap=7, rankloss=8,
            dcg_ignore_ties=9)


def _scores(y_score, n, caller, *, ndim=None, name="y_score"):
    from ._buffer import materialize_f32_lists
    a = materialize_f32_lists(y_score, name)[0]
    if a.dtype != "<f4":
        raise TypeError(f"mojolearn {caller}: {name} must have dtype float32; cast explicitly")
    if a.ndim == 2 and a.shape[1] == 1 and ndim == 1:
        a = a.reshape((a.shape[0],))
    if ndim is not None and a.ndim != ndim:
        raise ValueError(f"mojolearn {caller}: {name} must be {ndim}-D, got shape {a.shape}")
    if a.shape[0] != n:
        raise ValueError(f"mojolearn {caller}: {name} has {a.shape[0]} rows for {n} samples")
    a = as_f32_c(a, ndim=a.ndim, name=name)[0]
    if not all_finite(a):
        raise ValueError(f"mojolearn {caller}: {name} must be finite")
    return a


def _targets(y_true, caller):
    from ._metrics_impl import _classification_encoded, _label_set
    _refuse_multilabel(y_true, "y_true", caller)
    true, kind = _classification_encoded(y_true, "y_true")
    return true, kind, sorted(_label_set(true))


def _pos_label(pos_label, kind, classes, caller):
    from ._metrics_impl import _classification_labels
    if pos_label is None:
        if kind != "integer" or classes not in ([0], [1], [-1], [0, 1], [-1, 1]):
            raise ValueError(f"y_true takes value in {classes} and pos_label is not specified: either make "
                             "y_true take value in {0, 1} or {-1, 1} or pass pos_label explicitly.")
        return 1
    values, pk = _classification_labels([pos_label], "pos_label")
    if pk != kind:
        raise ValueError("pos_label must use the same type as y_true")
    return values[0]


class _Curve(tuple):
    """(fps, tps, thresholds); `keep` = the collinear-drop flags the device
    computed for an unweighted curve (bytes of 0/1 per slot), else None."""
    keep = None


class _DevCurve:
    """A curve left in its program's arena (lane metrics-apple2): the
    offsets of its fps, tps, thresholds and keep words (-1 = none; keep -2 =
    the device dropped the collinear points, every point is kept) and its
    point count `c`. `lists()` is the `_Curve` `_curves` returns; the
    epilogue helpers (`_auc_of`, `_ap_of`, `roc_curve`) read the words in
    place instead."""

    def __init__(self, prog, fps, tps, thr, keep, c, numeric_mode):
        self.prog, self.fps, self.tps, self.thr, self.keep, self.c = prog, fps, tps, thr, keep, c
        self.numeric_mode = numeric_mode

    def last(self):
        """(fps[c-1], tps[c-1]) as Python floats (c > 0)."""
        p = self.prog
        return p.floats(self.fps + self.c - 1, 1)[0], p.floats(self.tps + self.c - 1, 1)[0]

    def native(self, name):
        """The binding's epilogue entry `name`, or None (an older binary,
        MOJOLEARN_HOTPATH=python, or an empty curve)."""
        from ._buffer import hotpath_enabled
        if self.c <= 0 or not hotpath_enabled():
            return None
        fn = getattr(_binding(self.numeric_mode), name, None)
        if fn is not None:
            p = self.prog
            p._check(self.fps, self.c)
            p._check(self.tps, self.c)
            if self.keep >= 0:
                p._check(self.keep, self.c)
        return fn

    def addr(self):
        return self.prog.arena.buffer_info()[0]

    def lists(self):
        p, c = self.prog, self.c
        cur = _Curve((p.floats(self.fps, c), p.floats(self.tps, c),
                      p.floats(self.thr, c) if self.thr >= 0 else None))
        if self.keep >= 0:
            p._check(self.keep, c)
            lo = 4 * self.keep
            cur.keep = bytes(memoryview(p.arena).cast("B")[lo + (0 if _LITTLE else 3):lo + 4 * c:4])
        elif self.keep == -2:
            cur.keep = b"\x01" * c       # the device dropped the collinear points already
        return cur


def _epilogue(name, numeric_mode):
    """The binding's host epilogue entry `name` (x_metrics/epilogue.mojo,
    lane py-misc-metrics), or None: an older binary, MOJOLEARN_HOTPATH=python,
    or MOJOLEARN_METRICS_EPILOGUE=python (the reference arm the identity and
    timing jobs run beside the native one in the same build)."""
    import os
    from ._buffer import hotpath_enabled
    if not hotpath_enabled() or os.environ.get("MOJOLEARN_METRICS_EPILOGUE", "").strip().lower() == "python":
        return None
    return getattr(_binding(numeric_mode), name, None)


def _dev_epilogue(dev, name):
    """`_epilogue` for a `_DevCurve` whose fps, tps and thresholds words are
    all readable (declared outputs), or None."""
    if dev.c <= 0 or dev.thr < 0:
        return None
    fn = _epilogue(name, dev.numeric_mode)
    if fn is not None:
        p = dev.prog
        p._check(dev.fps, dev.c)
        p._check(dev.tps, dev.c)
        p._check(dev.thr, dev.c)
    return fn


def _f64_out(n):
    return array.array("d", bytes(8 * max(int(n), 1)))


def _f64_array(buf, m):
    del buf[m:]
    return Array._owned(buf, (m,), "<f8", "C")


def _i32_codes_addr(codes):
    """(address, keepalive) of int32 C-order codes, or (None, None)."""
    if isinstance(codes, Array) and codes.dtype == "<i4" and codes._has_order("C"):
        return addr_ro(codes, name="codes"), codes
    return None, None


def _f32_weights_addr(w):
    if w is None:
        return 0, None
    if isinstance(w, Array) and w.dtype == "<f4" and w._has_order("C"):
        return addr_ro(w, name="sample_weight"), w
    return None, None


def _f64_bits(x):
    import struct
    return struct.unpack("<q", struct.pack("<d", float(x)))[0]


def _auc_of(cur, max_fpr):
    """`_binary_auc` of a `_DevCurve`: the binding's host epilogue
    (x_metrics/epilogue.mojo binary_auc, the same binary64 operations)
    when both classes are present, else (or on any doubt) the Python."""
    fn = cur.native("x_metrics_curve_auc")
    if fn is not None:
        F, T = cur.last()
        if F > 0 and T > 0:
            try:
                return float(fn(cur.addr(), cur.fps, cur.tps, cur.keep, cur.c,
                                _f64_bits(-1.0 if max_fpr is None else max_fpr)))
            except Exception:
                pass
    L = cur.lists()
    return _binary_auc(L[0], L[1], max_fpr, L.keep)


def _ap_of(cur):
    """`_binary_ap` of a `_DevCurve` (x_metrics/epilogue.mojo binary_ap)."""
    fn = cur.native("x_metrics_curve_ap")
    if fn is not None:
        _, T = cur.last()
        if T != 0:
            try:
                return float(fn(cur.addr(), cur.fps, cur.tps, cur.c))
            except Exception:
                pass
    L = cur.lists()
    return _binary_ap(L[0], L[1])


def _curves(scores, flags, w, n, problems, numeric_mode, *, stride=1, thresholds=True, keep_flags=True,
            lazy=False, compact=False):
    """[(fps, tps, thresholds)] per problem, from the device sort and the
    cumulative counts (Python floats; unweighted counts are exact). An
    unweighted curve also carries `.keep` (lane metrics-apple). Only the
    curves come back from the device, each only up to its point count, and
    thresholds=False (the AUCs) leaves the thresholds on it: the third list
    is then None; keep_flags=False (the average precisions) computes no
    `.keep`. compact=True (lazy, unweighted, keep_flags) has the device drop
    the collinear points itself (bin_curve params 12, 13: the kept fps, tps,
    thresholds, and their count), so only the kept points come back; the
    `_DevCurve`s then carry keep = -2, every point kept (lane
    metrics-apple2)."""
    prog = _Prog()
    S = prog.put(scores)
    POS = prog.put_i32(flags)
    W = _NONE if w is None else prog.put(w)
    N = n * problems
    order = prog.scratch(N)
    flagged = w is None and keep_flags
    compact = compact and flagged and lazy
    cnt = prog.want(prog.alloc(problems), problems)
    fps = prog.alloc(N)
    tps = prog.alloc(N)
    thr = prog.alloc(N)
    keep = _NONE
    CF = CM = 0
    if compact:
        keep = prog.scratch(N)
        CF = prog.alloc(3 * N)
        CM = prog.want(prog.alloc(problems), problems)
        for t in range(problems):
            for b in range(3 if thresholds else 2):
                prog.want(CF + b * N + t * n, n, count=CM + t)
    else:
        if flagged:
            keep = prog.alloc(N)
        for t in range(problems):
            for b in (fps, tps) + ((thr,) if thresholds else ()) + ((keep,) if flagged else ()):
                prog.want(b + t * n, n, count=cnt + t)
    prog.stage("bin_curve", problems, S, stride, POS, W, n, order, fps, tps, thr, cnt,
               keep, 1 if flagged else 0, CF, CM)
    _execute(prog, numeric_mode)
    out = []
    counts = prog.ints(CM if compact else cnt, problems)
    if compact:
        return [_DevCurve(prog, CF + t * n, CF + N + t * n, CF + 2 * N + t * n if thresholds else _NONE,
                          -2, counts[t], numeric_mode) for t in range(problems)]
    view = memoryview(prog.arena).cast("B") if flagged else None
    for t in range(problems):
        c = counts[t]
        if lazy:
            out.append(_DevCurve(prog, fps + t * n, tps + t * n, thr + t * n if thresholds else _NONE,
                                 keep + t * n if flagged else _NONE, c, numeric_mode))
            continue
        cur = _Curve((prog.floats(fps + t * n, c), prog.floats(tps + t * n, c),
                      prog.floats(thr + t * n, c) if thresholds else None))
        if view is not None:
            prog._check(keep + t * n, c)
            lo = 4 * (keep + t * n)
            cur.keep = bytes(view[lo + (0 if _LITTLE else 3):lo + 4 * c:4])
        out.append(cur)
    return out


def _binary_curve(y_true, y_score, pos_label, sample_weight, numeric_mode, caller, lazy=False, compact=False):
    from ._metrics_impl import _label_map
    true, kind, classes = _targets(y_true, caller)
    if len(classes) > 2 and pos_label is None:
        raise ValueError("multiclass format is not supported")
    pos = _pos_label(pos_label, kind, classes, caller)
    n = len(true)
    s = _scores(y_score, n, caller, ndim=1)
    w = _weights(sample_weight, n, caller)
    flags = _label_map(true, lambda v: int(v == pos))
    return _curves(s, flags, w, n, 1, numeric_mode, lazy=lazy, compact=compact)[0], classes


def _drop_collinear(fps, tps, thr, keep=None):
    """Keep the first and last points and every point where either step
    changes: (f[i+1] - f[i]) != (f[i] - f[i-1]), the same binary64
    subtractions and compares, iterated in C (lane metrics-apple). `keep`,
    when given, is the device's flags for exactly these lists (an unweighted
    curve, `_Curve.keep`)."""
    if keep is not None and len(keep) == len(fps):
        return (list(itertools.compress(fps, keep)), list(itertools.compress(tps, keep)),
                list(itertools.compress(thr, keep)))
    df = list(map(operator.sub, fps[1:], fps[:-1]))
    dt = list(map(operator.sub, tps[1:], tps[:-1]))
    inner = map(operator.or_, map(operator.ne, df[1:], df[:-1]), map(operator.ne, dt[1:], dt[:-1]))
    keep = list(itertools.chain((True,), inner, (True,)))
    return (list(itertools.compress(fps, keep)), list(itertools.compress(tps, keep)),
            list(itertools.compress(thr, keep)))


def roc_curve(y_true, y_score, *, pos_label=None, sample_weight=None, drop_intermediate=True,
              numeric_mode=None):
    """scikit-learn 1.9 `roc_curve` for binary targets: fpr, tpr, thresholds
    (Float64; the first threshold is +inf). The sort and the cumulative
    counts run on the device; drop_intermediate removes collinear points."""
    dev, _ = _binary_curve(y_true, y_score, pos_label, sample_weight, numeric_mode, "roc_curve", lazy=True,
                           compact=bool(drop_intermediate))
    fn = dev.native("x_metrics_curve_roc")
    if fn is not None:
        # the three Float64 arrays straight from the arena words
        # (x_metrics/epilogue.mojo roc_arrays; lane metrics-apple2)
        F, T = dev.last()
        if F > 0 and T > 0:
            try:
                bufs = [array.array("d", bytes(8 * (dev.c + 1))) for _ in range(3)]
                m = int(fn(dev.addr(), (dev.fps, dev.tps, dev.thr, dev.keep), dev.c,
                           1 if drop_intermediate else 0, tuple(b.buffer_info()[0] for b in bufs)))
                for b in bufs:
                    del b[m:]
                return tuple(Array._owned(b, (m,), "<f8", "C") for b in bufs)
            except Exception:
                pass
    cur = dev.lists()
    fps, tps, thr = cur
    if drop_intermediate and len(fps) > 2:
        fps, tps, thr = _drop_collinear(fps, tps, thr, cur.keep)
    fps, tps, thr = [0.0] + fps, [0.0] + tps, [float("inf")] + thr
    if fps[-1] <= 0:
        _undefined_warning("No negative samples in y_true, false positive value should be meaningless")
        fpr = [float("nan")] * len(fps)
    else:
        fpr = list(map(operator.truediv, fps, itertools.repeat(fps[-1])))
    if tps[-1] <= 0:
        _undefined_warning("No positive samples in y_true, true positive value should be meaningless")
        tpr = [float("nan")] * len(tps)
    else:
        tpr = list(map(operator.truediv, tps, itertools.repeat(tps[-1])))
    return (Array.from_list(fpr, "<f8"), Array.from_list(tpr, "<f8"), Array.from_list(thr, "<f8"))


def precision_recall_curve_options(y_true, y_score, pos_label, sample_weight, drop_intermediate, numeric_mode):
    """precision_recall_curve with sample_weight or drop_intermediate=True
    (lane/metrics): Float64 outputs; the default call keeps its kernel."""
    dev, _ = _binary_curve(y_true, y_score, pos_label, sample_weight, numeric_mode, "precision_recall_curve",
                           lazy=True)
    fn = _dev_epilogue(dev, "x_metrics_curve_pr")
    if fn is not None and dev.last()[1] != 0:
        # the three Float64 arrays straight from the arena words
        # (x_metrics/epilogue.mojo pr_arrays; lane py-misc-metrics)
        try:
            bufs = [_f64_out(dev.c + 1) for _ in range(3)]
            m = int(fn(dev.addr(), (dev.fps, dev.tps, dev.thr), dev.c, 1 if drop_intermediate else 0,
                       tuple(b.buffer_info()[0] for b in bufs)))
            return _f64_array(bufs[0], m + 1), _f64_array(bufs[1], m + 1), _f64_array(bufs[2], m)
        except Exception:
            pass
    fps, tps, thr = dev.lists()
    if drop_intermediate and len(fps) > 2:
        keep = [0] + [i for i in range(1, len(fps) - 1)
                      if tps[i] != tps[i - 1] or tps[i + 1] != tps[i]] + [len(fps) - 1]
        fps, tps, thr = [fps[i] for i in keep], [tps[i] for i in keep], [thr[i] for i in keep]
    prec = [t / (t + f) if (t + f) != 0 else 0.0 for t, f in zip(tps, fps)]
    if tps[-1] == 0:
        warnings.warn("No positive class found in y_true, recall is set to one for all thresholds.",
                      UserWarning, stacklevel=3)
        rec = [1.0] * len(tps)
    else:
        rec = [t / tps[-1] for t in tps]
    return (Array.from_list(prec[::-1] + [1.0], "<f8"), Array.from_list(rec[::-1] + [0.0], "<f8"),
            Array.from_list(thr[::-1], "<f8"))


def det_curve(y_true, y_score, *, pos_label=None, sample_weight=None, drop_intermediate=False,
              numeric_mode=None):
    """scikit-learn 1.9 `det_curve`: fpr, fnr, thresholds."""
    dev, classes = _binary_curve(y_true, y_score, pos_label, sample_weight, numeric_mode, "det_curve", lazy=True)
    fn = _dev_epilogue(dev, "x_metrics_curve_det") if len(classes) == 2 else None
    if fn is not None:
        F, T = dev.last()
        if F != 0 and T != 0:
            # x_metrics/epilogue.mojo det_arrays (lane py-misc-metrics)
            try:
                bufs = [_f64_out(dev.c + 1) for _ in range(3)]
                m = int(fn(dev.addr(), (dev.fps, dev.tps, dev.thr), dev.c, 1 if drop_intermediate else 0,
                           tuple(b.buffer_info()[0] for b in bufs)))
                return tuple(_f64_array(b, m) for b in bufs)
            except Exception:
                pass
    fps, tps, thr = dev.lists()
    if drop_intermediate and len(fps) > 2:
        keep = [0] + [i for i in range(1, len(fps) - 1)
                      if tps[i] != tps[i - 1] or tps[i + 1] != tps[i]] + [len(fps) - 1]
        fps, tps, thr = [fps[i] for i in keep], [tps[i] for i in keep], [thr[i] for i in keep]
    tps, fps, thr = [0.0] + tps, [0.0] + fps, [float("inf")] + thr
    if len(classes) != 2:
        raise ValueError("Only one class is present in y_true. Detection error tradeoff curve is not "
                         "defined in that case.")
    fns = [tps[-1] - v for v in tps]
    p_count, n_count = tps[-1], fps[-1]
    import bisect
    right = bisect.bisect_right(fps, fps[0])
    first = right - 1 if right > 0 else 0
    last = bisect.bisect_left(tps, tps[-1]) + 1
    sl = slice(first, last)
    return (Array.from_list([v / n_count for v in fps[sl]][::-1], "<f8"),
            Array.from_list([v / p_count for v in fns[sl]][::-1], "<f8"),
            Array.from_list(thr[sl][::-1], "<f8"))


def _trapezoid(x, y):
    """fsum of (x[i] - x[i-1]) * (y[i] + y[i-1]) / 2, the same binary64
    operations per term, iterated in C (lane metrics-apple)."""
    terms = map(operator.truediv,
                map(operator.mul, map(operator.sub, x[1:], x[:-1]), map(operator.add, y[1:], y[:-1])),
                itertools.repeat(2))
    return _fsum(list(terms))


def auc(x, y):
    """scikit-learn 1.9 `auc`: the trapezoid rule over a monotonic x
    (host binary64, a correctly rounded `fsum`)."""
    xa, ya = _auc_f64(x), _auc_f64(y)
    if xa is not None and ya is not None and xa.size == ya.size and xa.size >= 2:
        fn = _epilogue("x_metrics_auc_xy", None)
        if fn is not None:
            # x_metrics/epilogue.mojo auc_xy (lane py-misc-metrics): the
            # same differences, direction and trapezoid terms, fsum-ed
            try:
                return float(fn(addr_ro(xa, name="x"), addr_ro(ya, name="y"), xa.size))
            except Exception:
                pass
    x = [float(v) for v in flatten_mo(x)]
    y = [float(v) for v in flatten_mo(y)]
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if len(x) < 2:
        raise ValueError(f"At least 2 points are needed to compute area under curve, but x.shape = ({len(x)},)")
    dx = [x[i] - x[i - 1] for i in range(1, len(x))]
    direction = 1
    if any(d < 0 for d in dx):
        if all(d <= 0 for d in dx):
            direction = -1
        else:
            raise ValueError(f"x is neither increasing nor decreasing : {x}.")
    return float(direction * _trapezoid(x, y))


def _auc_f64(v):
    """A 1-D float32 or float64 buffer as a C-order Float64 Array (float32
    widened exactly, as float() does), or None for anything else (lists,
    integers, n-D): those keep the Python conversion."""
    from ._buffer import view, typestr_of, _has_buffer, as_f64_c
    if isinstance(v, (list, tuple)) or not (isinstance(v, Array) or _has_buffer(v)):
        return None
    try:
        b = view(v, name="x")
        try:
            ok = b.ndim == 1 and typestr_of(b) in ("<f4", "<f8")
        finally:
            b.release()
        return as_f64_c(v, ndim=1, name="x")[0] if ok else None
    except Exception:
        return None


def _binary_auc(fps, tps, max_fpr, keep=None):
    if not fps or fps[-1] <= 0 or tps[-1] <= 0:
        _undefined_warning("Only one class is present in y_true. ROC AUC score is not defined in that case.")
        return float("nan")
    fps, tps, _ = _drop_collinear(fps, tps, fps, keep) if len(fps) > 2 else (fps, tps, fps)
    fpr = [0.0] + list(map(operator.truediv, fps, itertools.repeat(fps[-1])))
    tpr = [0.0] + list(map(operator.truediv, tps, itertools.repeat(tps[-1])))
    if max_fpr is None or max_fpr == 1:
        return float(_trapezoid(fpr, tpr))
    import bisect
    stop = bisect.bisect_right(fpr, max_fpr)
    x0, x1, y0, y1 = fpr[stop - 1], fpr[stop], tpr[stop - 1], tpr[stop]
    yi = y0 if x1 == x0 else y0 + (max_fpr - x0) * (y1 - y0) / (x1 - x0)
    part = _trapezoid(fpr[:stop] + [max_fpr], tpr[:stop] + [yi])
    min_area = 0.5 * max_fpr * max_fpr
    return float(0.5 * (1 + (part - min_area) / (max_fpr - min_area)))


def _binary_ap(fps, tps):
    if not tps or tps[-1] == 0:
        # sklearn: recall is set to one; the sum over diff(recall) is 0
        return 0.0
    # (r - r_prev) * (t / (t + f)) per point, r = t / T, in C (lane metrics-apple)
    rs = list(map(operator.truediv, tps, itertools.repeat(tps[-1])))
    terms = map(operator.mul, map(operator.sub, rs, itertools.chain((0.0,), rs[:-1])),
                map(operator.truediv, tps, map(operator.add, tps, fps)))
    return float(max(0.0, _fsum(list(terms))))


def _ovr(y_true, y_score, sample_weight, labels, caller, numeric_mode, keep_flags=True):
    """Binarized one-vs-rest problems over the class columns of y_score."""
    from ._metrics_impl import _label_map, _selected_labels
    true, kind, present = _targets(y_true, caller)
    n = len(true)
    s = _scores(y_score, n, caller, ndim=2)
    k = s.shape[1]
    classes = present if labels is None else _selected_labels(labels, kind, present)
    if labels is not None:
        if classes != sorted(classes):
            raise ValueError("Parameter 'labels' must be ordered")
        if set(present) - set(classes):
            raise ValueError("'y_true' contains labels not in parameter 'labels'")
    if len(classes) != k:
        raise ValueError("Number of classes in y_true not equal to the number of columns in 'y_score'")
    w = _weights(sample_weight, n, caller)
    index = {c: i for i, c in enumerate(classes)}
    codes = _label_map(true, lambda v: index[v])
    code_list = codes.tolist()
    if k <= 256:
        # class-major 0/1 flags as int32 words, built with bytes.translate
        # and a strided byte copy (little-endian '<i4'), and the unweighted
        # support by bytes.count: the same values (lane metrics-apple)
        cb = bytes(code_list)
        words = bytearray(4 * n * k)
        words[0::4] = b"".join(cb.translate(bytes(int(j == c) for j in range(256))) for c in range(k))
        store = array.array("i")
        store.frombytes(words)
        if not _LITTLE:
            store.byteswap()
        flags = Array._owned(store, (n * k,), "<i4", "C")
    else:
        flags = []
        for c in range(k):
            flags.extend(1 if v == c else 0 for v in code_list)
        flags = Array.from_list(flags, "<i4")
    curves = _curves(s, flags, w, n, k, numeric_mode, stride=k, thresholds=False, keep_flags=keep_flags,
                     lazy=True, compact=True)
    support = [0.0] * k
    if w is None:
        if k <= 256:
            support = [float(cb.count(c)) for c in range(k)]
        else:
            for v in code_list:
                support[v] += 1
    else:
        native = _class_sums_native(codes, w, k, numeric_mode)
        if native is not None:
            support = native
        else:
            for v, wt in zip(code_list, w.tolist()):
                support[v] += wt
    return curves, support, s, code_list, classes, w


_LITTLE = array.array("i", [1]).tobytes()[0] == 1


def _rows_sum_to_one(s, k, numeric_mode=None):
    """No row's correctly rounded sum is farther than 1e-8 + 1e-5 from 1
    (scikit-learn's check; the scores are finite float32). Each row's
    `math.fsum` is `_fsum`'s value (finite float32 terms never overflow
    binary64; a zero sum only differs in its sign, which |s - 1| drops),
    the rows iterated in C; |fl(s - 1)| is monotone on either side of 1,
    so the largest and smallest sums decide every row (lane metrics-apple)."""
    from ._buffer import hotpath_enabled
    n = s.size // k if k else 0
    fn = getattr(_binding(numeric_mode), "x_metrics_row_sum_range", None) if hotpath_enabled() else None
    if fn is not None and n > 0 and s.dtype == "<f4" and s._has_order("C"):
        # the same row fsums, in the binding (x_metrics/epilogue.mojo
        # row_sum_range; lane metrics-apple2)
        out = array.array("d", [0.0, 0.0])
        fn(addr_ro(s, name="y_score"), n, k, out.buffer_info()[0])
        tol = 1e-8 + 1e-5
        return not (abs(out[0] - 1) > tol or abs(out[1] - 1) > tol)
    flat = array.array("f")
    flat.frombytes(s.tobytes())
    if not _LITTLE:
        flat.byteswap()
    if not len(flat):
        return True
    sums = list(map(_math.fsum, zip(*[flat[c::k] for c in range(k)])))
    tol = 1e-8 + 1e-5
    return not (abs(max(sums) - 1) > tol or abs(min(sums) - 1) > tol)


def _average_scores(scores, support, average):
    if average is None:
        return Array.from_list(scores, "<f8")
    if average == "weighted":
        total = _fsum(support)
        return float(_fsum([a * b for a, b in zip(scores, support)]) / total) if total else 0.0
    return float(_fsum(scores) / len(scores))


def roc_auc_options(y_true, y_score, average, sample_weight, max_fpr, multi_class, labels, numeric_mode):
    """roc_auc_score with sample_weight, max_fpr < 1 or multiclass input
    (lane/metrics); the binary unweighted full-AUC call keeps its kernel.
    AUCs are the trapezoid rule over the device curve (host `fsum`)."""
    from ._metrics_impl import _label_map
    from ._buffer import materialize_f32_lists
    shape = materialize_f32_lists(y_score, "y_score")[0].shape
    true, kind, present = _targets(y_true, "roc_auc_score")
    multiclass = len(present) > 2 or (len(shape) == 2 and shape[1] > 2)
    if max_fpr is not None and (is_bool(max_fpr) or not isinstance(max_fpr, numbers.Real)
                                or not 0 < max_fpr <= 1):
        raise ValueError(f"Expected max_fpr in range (0, 1], got: {max_fpr!r}")
    if not multiclass:
        if average not in (None, "micro", "macro", "samples", "weighted"):
            raise ValueError("average has to be one of (None, 'micro', 'macro', 'weighted', 'samples')")
        if len(present) != 2:
            _undefined_warning("Only one class is present in y_true. ROC AUC score is not defined in that case.")
            return float("nan")
        n = len(true)
        s = _scores(y_score, n, "roc_auc_score", ndim=1)
        w = _weights(sample_weight, n, "roc_auc_score")
        flags = _label_map(true, lambda v: int(v == present[1]))
        cur = _curves(s, flags, w, n, 1, numeric_mode, thresholds=False, lazy=True, compact=True)[0]
        return _auc_of(cur, None if max_fpr == 1 else max_fpr)
    if max_fpr is not None and max_fpr != 1:
        raise ValueError("Partial AUC computation not available in multiclass setting, 'max_fpr' must be "
                         f"set to `None`, received `max_fpr={max_fpr}` instead")
    if multi_class not in ("ovo", "ovr"):
        raise ValueError("multi_class must be in ('ovo', 'ovr')")
    options = ("micro", "macro", "weighted", None) if multi_class == "ovr" else ("macro", "weighted", None)
    if average not in options:
        raise ValueError(f"average must be one of {options} for multiclass problems")
    if average is None and multi_class == "ovo":
        raise NotImplementedError("average=None is not implemented for multi_class='ovo'.")
    if multi_class == "ovo" and sample_weight is not None:
        raise ValueError("sample_weight is not supported for multiclass one-vs-one ROC AUC, "
                         "'sample_weight' must be None in this case.")
    s_check = _scores(y_score, len(true), "roc_auc_score", ndim=2)
    k = s_check.shape[1]
    if not _rows_sum_to_one(s_check, k, numeric_mode):
        raise ValueError("Target scores need to be probabilities for multiclass roc_auc, i.e. they "
                         "should sum up to 1.0 over classes")
    if multi_class == "ovr":
        curves, support, s, codes, classes, w = _ovr(y_true, y_score, sample_weight, labels, "roc_auc_score",
                                                     numeric_mode)
        if average == "micro":
            n = len(codes)
            flags = Array.from_list([1 if codes[r] == c else 0 for r in range(n) for c in range(k)], "<i4")
            wm = None if w is None else Array.from_list([x for x in w.tolist() for _ in range(k)], "<f4")
            cur = _curves(s.reshape((n * k,)), flags, wm, n * k, 1, numeric_mode, thresholds=False, lazy=True,
                          compact=True)[0]
            return _auc_of(cur, None)
        scores = [_auc_of(c, None) for c in curves]
        return _average_scores(scores, support, average)
    # one-vs-one (scikit-learn _average_multiclass_ovo_score)
    from ._metrics_impl import _selected_labels
    vals = s_check.tolist()
    classes = present if labels is None else _selected_labels(labels, kind, present)
    if len(classes) != k:
        raise ValueError("Number of classes in y_true not equal to the number of columns in 'y_score'")
    index = {c: i for i, c in enumerate(classes)}
    codes = _label_map(true, lambda v: index[v]).tolist()
    n = len(codes)
    pair_scores, prevalence = [], []
    for a in range(k):
        for b in range(a + 1, k):
            rows = [r for r in range(n) if codes[r] in (a, b)]
            prevalence.append(len(rows) / n)
            both = []
            for col, pos in ((a, a), (b, b)):
                sv = Array.from_list([vals[r][col] for r in rows], "<f4")
                fl = Array.from_list([1 if codes[r] == pos else 0 for r in rows], "<i4")
                cur = _curves(sv, fl, None, len(rows), 1, numeric_mode, thresholds=False, lazy=True,
                              compact=True)[0]
                both.append(_auc_of(cur, None))
            pair_scores.append((both[0] + both[1]) / 2)
    if average == "weighted":
        return float(_fsum([x * y for x, y in zip(pair_scores, prevalence)]) / _fsum(prevalence))
    return float(_fsum(pair_scores) / len(pair_scores))


def average_precision_score(y_true, y_score, *, average="macro", pos_label=1, sample_weight=None,
                            numeric_mode=None):
    """scikit-learn 1.9 `average_precision_score` (binary and multiclass
    one-vs-rest; multilabel-indicator targets are refused by name)."""
    from ._metrics_impl import _label_map
    if average not in (None, "micro", "macro", "weighted", "samples"):
        raise ValueError("average has to be one of (None, 'micro', 'macro', 'weighted', 'samples')")
    true, kind, present = _targets(y_true, "average_precision_score")
    if len(present) <= 2:
        if len(present) == 2 and pos_label not in present:
            raise ValueError(f"pos_label={pos_label} is not a valid label. It should be one of {present}")
        n = len(true)
        s = _scores(y_score, n, "average_precision_score", ndim=1)
        w = _weights(sample_weight, n, "average_precision_score")
        flags = _label_map(true, lambda v: int(v == pos_label))
        return _ap_of(_curves(s, flags, w, n, 1, numeric_mode, thresholds=False, keep_flags=False, lazy=True)[0])
    if pos_label != 1:
        raise ValueError("Parameter pos_label is fixed to 1 for multiclass y_true. Do not set pos_label "
                         "or set pos_label to 1.")
    if average == "samples":
        raise NotImplementedError("mojolearn average_precision_score: average='samples' applies to "
                                  "multilabel targets, which are NOT IMPLEMENTED")
    curves, support, s, codes, classes, w = _ovr(y_true, y_score, sample_weight, None,
                                                 "average_precision_score", numeric_mode, keep_flags=False)
    k = len(classes)
    if average == "micro":
        n = len(codes)
        flags = Array.from_list([1 if codes[r] == c else 0 for r in range(n) for c in range(k)], "<i4")
        wm = None if w is None else Array.from_list([x for x in w.tolist() for _ in range(k)], "<f4")
        return _ap_of(_curves(s.reshape((n * k,)), flags, wm, n * k, 1, numeric_mode, thresholds=False,
                              keep_flags=False, lazy=True)[0])
    return _average_scores([_ap_of(c) for c in curves], support, average)


def top_k_accuracy_score(y_true, y_score, *, k=2, normalize=True, sample_weight=None, labels=None,
                         numeric_mode=None):
    """scikit-learn 1.9 `top_k_accuracy_score`: a hit when the true class is
    among the k largest scores, ties ordered as numpy's stable argsort
    reversed (the larger class index first)."""
    from ._metrics_impl import _label_map, _selected_labels
    if is_bool(k) or not isinstance(k, numbers.Integral) or k < 1:
        raise ValueError("k must be a positive integer")
    true, kind, present = _targets(y_true, "top_k_accuracy_score")
    n = len(true)
    s = _scores(y_score, n, "top_k_accuracy_score")
    binary = len(present) <= 2 and not (labels is not None and len(labels) > 2)
    classes = present if labels is None else _selected_labels(labels, kind, present)
    if labels is not None:
        if classes != sorted(classes):
            raise ValueError("Parameter 'labels' must be ordered.")
        if set(present) - set(classes):
            raise ValueError("'y_true' contains labels not in parameter 'labels'.")
    n_score = s.shape[1] if s.ndim == 2 else 2
    if binary and s.ndim == 2 and s.shape[1] != 1:
        raise ValueError(f"`y_true` is binary while y_score is 2d with {s.shape[1]} classes. If `y_true` "
                         "does not contain all the labels, `labels` must be provided.")
    if len(classes) != n_score:
        raise ValueError(f"Number of classes in 'y_true' ({len(classes)}) not equal to the number of "
                         f"classes in 'y_score' ({n_score}).")
    if k >= len(classes):
        _undefined_warning(f"'k' ({k}) greater than or equal to 'n_classes' ({len(classes)}) will result "
                           "in a perfect score and is therefore meaningless.")
    w = _weights(sample_weight, n, "top_k_accuracy_score")
    index = {c: i for i, c in enumerate(classes)}
    codes = _label_map(true, lambda v: index[v])
    prog = _Prog()
    if binary:
        flat = s.reshape((n,)) if s.ndim == 2 else s
        if k == 1:
            thr = 0.5 if flat.min() >= 0 and flat.max() <= 1 else 0.0
            # two score columns [thr, s] make "s > thr" a top-1 hit of class 1
            # with the tie going to class 0 (scikit-learn: y_pred = s > thr)
            two = Array.from_list([v for x in flat.tolist() for v in (thr, x)], "<f4")
            S = prog.put(two)
            kk, cols = 1, 2
        else:
            S = prog.put(Array.from_list([0.0] * (2 * n), "<f4"))
            kk, cols = 2, 2
    else:
        S = prog.put(s)
        kk, cols = k, s.shape[1]
    Y = prog.put_i32(codes)
    hit = prog.scratch(n)
    prog.stage("row_metric", n, S, cols, Y, hit, _ROW["topk"], kk, 0)
    zero = prog.scratch(n)
    prog.stage("pair_key", n, 0, 0, zero, 1, 2)
    W = _NONE if w is None else prog.put(w)
    tot = _group(prog, zero, n, 1, values=hit, weights=W)[1]
    sw = _group(prog, zero, n, 1, weights=W)[1] if w is not None else None
    _execute(prog, numeric_mode)
    hits = prog.floats(tot, 1)[0]
    if not normalize:
        return float(hits)
    return float(hits / (n if w is None else prog.floats(sw, 1)[0]))


def _row_mean(S, cols, Y, n, kind, w, numeric_mode, *, K=0, D=None, prog=None, normalize=True):
    prog = prog or _Prog()
    Dt = _NONE if D is None else prog.put(Array.from_list([_f32(v) for v in D], "<f4"))
    out = prog.scratch(n)
    prog.stage("row_metric", n, S, cols, Y, out, _ROW[kind], K, Dt)
    zero = prog.scratch(n)
    prog.stage("pair_key", n, 0, 0, zero, 1, 2)
    W = _NONE if w is None else prog.put(w)
    tot = _group(prog, zero, n, 1, values=out, weights=W)[1]
    sw = _group(prog, zero, n, 1, weights=W)[1] if w is not None else None
    _execute(prog, numeric_mode)
    total = prog.floats(tot, 1)[0]
    if not normalize:
        return total
    return total / (n if w is None else prog.floats(sw, 1)[0])


def _proba(y_true, y_proba, labels, pos_label, caller):
    """(codes, P (n, k) Float32, k, binary): scikit-learn 1.9's validation of
    probabilistic predictions for a binary vector or a multiclass matrix."""
    from ._metrics_impl import _label_map, _selected_labels
    from ._buffer import materialize_f32_lists, _native
    true, kind, present = _targets(y_true, caller)
    n = len(true)
    a = materialize_f32_lists(y_proba, "y_proba")[0]
    if a.dtype != "<f4":
        raise TypeError(f"mojolearn {caller}: y_proba must have dtype float32; cast explicitly")
    binary = a.ndim == 1 or (a.ndim == 2 and a.shape[1] == 1)
    if binary:
        a = a.reshape((a.shape[0],))
        if len(present) > 2:
            raise ValueError("The type of the target inferred from y_true is multiclass but should be binary "
                             "according to the shape of y_prob.")
        if pos_label is None:
            try:
                pos = _pos_label(None, kind, present, caller)
            except ValueError:
                pos = present[-1]
        else:
            pos = _pos_label(pos_label, kind, present, caller)
        codes = _label_map(true, lambda v: int(v == pos))
        k = 2
    else:
        classes = present if labels is None else _selected_labels(labels, kind, present)
        if labels is not None and set(present) - set(classes):
            raise ValueError("y_true contains values not belonging to the passed labels")
        classes = sorted(classes)
        k = a.shape[1]
        if len(classes) != k:
            raise ValueError(f"y_true and y_proba contain different number of classes: {len(classes)} vs {k}")
        index = {c: i for i, c in enumerate(classes)}
        codes = _label_map(true, lambda v: index[v])
    if a.shape[0] != n:
        raise ValueError("y_true and y_proba have different numbers of rows")
    a = as_f32_c(a, ndim=a.ndim, name="y_proba")[0]
    packed = empty((n, 2), "<f4") if binary else None
    code = int(_native("probability_rows_f32")(addr_ro(a, name="y_proba"), addr_ro(packed, name="packed") if binary else 0,
                                                n, 1 if binary else k, int(binary)))
    if code:
        raise ValueError(f"mojolearn {caller}: y_proba " + {1: "must be finite", 2: "must lie in [0, 1]",
                         3: "rows must sum to one within sqrt(float32 eps)"}.get(code, "failed validation"))
    return codes, (packed if binary else a), k, binary


def _sq(v):
    """v * v: one correctly rounded product (DEVIATION 6106 allows + - * /
    and sqrt; `v ** 2` called libm pow, lane py-misc-metrics)."""
    return v * v


def _class_sums_native(codes, w, k, numeric_mode):
    """Per-class (weighted) counts, binary64 in row order, from the binding
    (x_metrics/epilogue.mojo class_sums, lane py-misc-metrics), or None."""
    fn = _epilogue("x_metrics_class_sums", numeric_mode)
    caddr, ckeep = _i32_codes_addr(codes)
    waddr, wkeep = _f32_weights_addr(w)
    if fn is None or caddr is None or waddr is None or k <= 0:
        return None
    out = _f64_out(k)
    try:
        fn(caddr, waddr, codes.size, k, out.buffer_info()[0])
    except Exception:
        return None
    return out.tolist()[:k]


def _class_weights(codes, w, k, numeric_mode=None):
    """Per-class (weighted) counts and the total, binary64 in row order."""
    per = _class_sums_native(codes, w, k, numeric_mode)
    if per is not None:
        return per, _fsum(per)
    cl = codes.tolist()
    per = [0.0] * k
    if w is None:
        for c in cl:
            per[c] += 1
    else:
        for c, x in zip(cl, w.tolist()):
            per[c] += x
    return per, _fsum(per)


def log_loss_options(y_true, y_pred, normalize, sample_weight, labels, numeric_mode):
    """log_loss with sample_weight (lane/metrics): the clipped `-log p_true`
    per row on the device, its weighted PairSum; the unweighted call keeps
    its kernel."""
    codes, P, k, _ = _proba(y_true, y_pred, labels, None, "log_loss")
    n = len(codes)
    w = _weights(sample_weight, n, "log_loss")
    prog = _Prog()
    S, Y = prog.put(P), prog.put_i32(codes)
    return float(_row_mean(S, k, Y, n, "logloss", w, numeric_mode, prog=prog, normalize=normalize))


def brier_score_loss(y_true, y_proba, *, sample_weight=None, pos_label=None, labels=None,
                     scale_by_half="auto", numeric_mode=None):
    """scikit-learn 1.9 `brier_score_loss` (binary vector or multiclass
    matrix): the mean of `sum_c (onehot - p)^2`, halved by default for the
    binary case."""
    codes, P, k, binary = _proba(y_true, y_proba, labels, pos_label, "brier_score_loss")
    n = len(codes)
    w = _weights(sample_weight, n, "brier_score_loss")
    prog = _Prog()
    S, Y = prog.put(P), prog.put_i32(codes)
    score = _row_mean(S, k, Y, n, "brier", w, numeric_mode, prog=prog)
    if scale_by_half == "auto":
        scale_by_half = binary or k < 3
    return float(score * 0.5 if scale_by_half else score)


def d2_log_loss_score(y_true, y_proba=None, *, sample_weight=None, labels=None, numeric_mode=None):
    """scikit-learn 1.9 `d2_log_loss_score`: one minus the log loss over the
    log loss of the (weighted) class frequencies."""
    codes, P, k, _ = _proba(y_true, y_proba, labels, None, "d2_log_loss_score")
    n = len(codes)
    if n < 2:
        _undefined_warning("D^2 score is not well-defined with less than two samples.")
        return float("nan")
    w = _weights(sample_weight, n, "d2_log_loss_score")
    prog = _Prog()
    S, Y = prog.put(P), prog.put_i32(codes)
    num = _row_mean(S, k, Y, n, "logloss", w, numeric_mode, prog=prog, normalize=False)
    per, total = _class_weights(codes, w, k, numeric_mode)
    eps = 1.1920928955078125e-07
    den = _fsum([wc * -pmath.log(min(max(wc / total, eps), 1 - eps)) for wc in per if wc])
    return float(1 - num / den)


def d2_brier_score(y_true, y_proba, *, sample_weight=None, pos_label=None, labels=None, numeric_mode=None):
    """scikit-learn 1.9 `d2_brier_score`: one minus the Brier score over the
    Brier score of the (weighted) class frequencies."""
    codes, P, k, _ = _proba(y_true, y_proba, labels, pos_label, "d2_brier_score")
    n = len(codes)
    if n < 2:
        _undefined_warning("D^2 score is not well-defined with less than two samples.")
        return float("nan")
    w = _weights(sample_weight, n, "d2_brier_score")
    prog = _Prog()
    S, Y = prog.put(P), prog.put_i32(codes)
    num = _row_mean(S, k, Y, n, "brier", w, numeric_mode, prog=prog)
    per, total = _class_weights(codes, w, k, numeric_mode)
    freq = [v / total for v in per]
    den = _fsum([per[c] * _fsum([_sq((1.0 if j == c else 0.0) - freq[j]) for j in range(k)])
                      for c in range(k)]) / total
    return float(1 - num / den)


def hinge_loss(y_true, pred_decision, *, labels=None, sample_weight=None, numeric_mode=None):
    """scikit-learn 1.9 `hinge_loss`: binary (labels mapped to -1 / +1, the
    larger class positive) or Crammer-Singer multiclass margins."""
    from ._metrics_impl import _label_map, _selected_labels
    true, kind, present = _targets(y_true, "hinge_loss")
    n = len(true)
    uniq = present if labels is None else sorted(_selected_labels(labels, kind, present))
    s = _scores(pred_decision, n, "hinge_loss", name="pred_decision")
    w = _weights(sample_weight, n, "hinge_loss")
    prog = _Prog()
    if len(uniq) > 2:
        if s.ndim != 2:
            raise ValueError("The shape of pred_decision cannot be 1d array with a multiclass target.")
        if s.shape[1] != len(uniq):
            raise ValueError("Please include all labels in y_true or pass labels as third argument"
                             if labels is None else
                             "The shape of pred_decision is not consistent with the number of classes.")
        index = {c: i for i, c in enumerate(uniq)}
        codes = _label_map(true, lambda v: index[v])
        S, Y = prog.put(s), prog.put_i32(codes)
        return float(_row_mean(S, s.shape[1], Y, n, "hinge_mc", w, numeric_mode, prog=prog))
    if s.ndim == 2:
        if s.shape[1] != 1:
            raise ValueError("pred_decision must be 1-D for a binary target")
        s = s.reshape((n,))
    pos = uniq[-1]
    codes = _label_map(true, lambda v: int(v == pos and len(uniq) == 2))
    S, Y = prog.put(s), prog.put_i32(codes)
    return float(_row_mean(S, 1, Y, n, "hinge_bin", w, numeric_mode, prog=prog))


def _relevance(y_true, y_score, caller, *, indicator):
    from ._buffer import materialize_f32_lists
    y = materialize_f32_lists(y_true, "y_true")[0]
    if y.dtype != "<f4":
        y = as_f32_c(y, ndim=y.ndim, name="y_true")[0]
    if y.ndim != 2:
        raise ValueError(f"mojolearn {caller}: y_true must be a 2-D (n_samples, n_labels) array")
    y = as_f32_c(y, ndim=2, name="y_true")[0]
    if not all_finite(y):
        raise ValueError(f"mojolearn {caller}: y_true must be finite")
    s = _scores(y_score, y.shape[0], caller, ndim=2)
    if s.shape != y.shape:
        raise ValueError("y_true and y_score have different shape")
    if indicator and any(v != 0.0 and v != 1.0 for v in y.reshape((y.size,)).tolist()):
        raise ValueError(f"{caller} requires a binary label indicator y_true")
    return y, s


def _dcg_discount(k_cols, log_base):
    return [1 / (pmath.log(i + 2) / pmath.log(log_base)) for i in range(k_cols)]


def dcg_score(y_true, y_score, *, k=None, log_base=2, sample_weight=None, ignore_ties=False,
              numeric_mode=None):
    """scikit-learn 1.9 `dcg_score`: tie-averaged gains by default;
    ignore_ties=True ranks tied scores by DESCENDING column index (numpy's
    `argsort(...)[::-1]` there, whose tie order its unstable sort leaves
    unspecified; this one is fixed)."""
    y, s = _relevance(y_true, y_score, "dcg_score", indicator=False)
    n, c = y.shape
    w = _weights(sample_weight, n, "dcg_score")
    prog = _Prog()
    S, Y = prog.put(s), prog.put(y)
    return float(_row_mean(S, c, Y, n, "dcg_ignore_ties" if ignore_ties else "dcg", w, numeric_mode,
                           K=0 if k is None else int(k),
                           D=_dcg_discount(c, log_base), prog=prog))


def ndcg_score(y_true, y_score, *, k=None, sample_weight=None, ignore_ties=False, numeric_mode=None):
    """scikit-learn 1.9 `ndcg_score`: each row's DCG over its ideal DCG
    (0 when the row has no relevant item)."""
    y, s = _relevance(y_true, y_score, "ndcg_score", indicator=False)
    n, c = y.shape
    if c <= 1:
        raise ValueError(f"Computing NDCG is only meaningful when there is more than 1 document. Got {c} instead.")
    if y.min() < 0:
        raise ValueError("ndcg_score should not be used on negative y_true values.")
    w = _weights(sample_weight, n, "ndcg_score")
    D = _dcg_discount(c, 2)
    K = 0 if k is None else int(k)
    prog = _Prog()
    S, Y = prog.put(s), prog.put(y)
    Dt = prog.put(Array.from_list([_f32(v) for v in D], "<f4"))
    gain = prog.alloc(n)
    ideal = prog.alloc(n)
    prog.stage("row_metric", n, S, c, Y, gain, _ROW["dcg_ignore_ties" if ignore_ties else "dcg"], K, Dt)
    prog.stage("row_metric", n, Y, c, Y, ideal, _ROW["dcg"], K, Dt)
    prog.want(gain, n)
    prog.want(ideal, n)
    _execute(prog, numeric_mode)
    fn = _epilogue("x_metrics_ndcg_mean", numeric_mode)
    waddr, keep = _f32_weights_addr(w)
    if fn is not None and waddr is not None:
        # x_metrics/epilogue.mojo ndcg_mean (lane py-misc-metrics): the same
        # ratios, products and fsums over the arena words
        prog._check(gain, n)
        prog._check(ideal, n)
        try:
            return float(fn(prog.arena.buffer_info()[0], gain, ideal, n, waddr))
        except Exception:
            pass
    g, i = prog.floats(gain, n), prog.floats(ideal, n)
    per = [a / b if b != 0 else 0.0 for a, b in zip(g, i)]
    if w is None:
        return float(_fsum(per) / n)
    wl = w.tolist()
    return float(_fsum([a * b for a, b in zip(per, wl)]) / _fsum(wl))


def _label_ranking(kind, y_true, y_score, sample_weight, numeric_mode, caller):
    y, s = _relevance(y_true, y_score, caller, indicator=True)
    n, c = y.shape
    w = _weights(sample_weight, n, caller)
    prog = _Prog()
    S, Y = prog.put(s), prog.put(y)
    return float(_row_mean(S, c, Y, n, kind, w, numeric_mode, prog=prog))


def coverage_error(y_true, y_score, *, sample_weight=None, numeric_mode=None):
    """scikit-learn 1.9 `coverage_error` (multilabel indicator y_true)."""
    return _label_ranking("coverage", y_true, y_score, sample_weight, numeric_mode, "coverage_error")


def label_ranking_average_precision_score(y_true, y_score, *, sample_weight=None, numeric_mode=None):
    """scikit-learn 1.9 `label_ranking_average_precision_score`."""
    return _label_ranking("lrap", y_true, y_score, sample_weight, numeric_mode,
                          "label_ranking_average_precision_score")


def label_ranking_loss(y_true, y_score, *, sample_weight=None, numeric_mode=None):
    """scikit-learn 1.9 `label_ranking_loss` (tied scores count as misordered)."""
    return _label_ranking("rankloss", y_true, y_score, sample_weight, numeric_mode, "label_ranking_loss")


# ---------------------------------------------------------------------------
# Clustering (scikit-learn 1.9 sklearn/metrics/cluster/)
# ---------------------------------------------------------------------------

def _clusterings(labels_true, labels_pred, caller):
    from ._metrics_impl import _classification_encoded, _label_set
    for v, name in ((labels_true, "labels_true"), (labels_pred, "labels_pred")):
        _refuse_multilabel(v, name, caller)
    a, ka = _classification_encoded(labels_true, "labels_true")
    b, kb = _classification_encoded(labels_pred, "labels_pred")
    if len(a) != len(b):
        raise ValueError(f"mojolearn {caller}: labels_true and labels_pred must have the same length")
    return a, b, sorted(_label_set(a)), sorted(_label_set(b))


def _contingency(a, b, ca, cb, numeric_mode):
    """Exact Int counts, rows = classes of `a`, columns = classes of `b`."""
    n, ka, kb = len(a), len(ca), len(cb)
    if ka * kb > 16777216:
        raise ValueError("mojolearn metrics: the contingency matrix exceeds 2^24 cells")
    prog = _Prog()
    A = prog.put_i32(_codes(a, ca))
    B = prog.put_i32(_codes(b, cb))
    key = prog.scratch(n)
    prog.stage("pair_key", n, A, B, key, max(ka, kb), 0)
    m = max(ka, kb) ** 2
    off, _ = _group(prog, key, n, m)
    _execute(prog, numeric_mode)
    o = prog.ints(off, m + 1)
    kk = max(ka, kb)
    return [[o[i * kk + j + 1] - o[i * kk + j] for j in range(kb)] for i in range(ka)]


def contingency_matrix(labels_true, labels_pred, *, eps=None, sparse=False, dtype="int64", numeric_mode=None):
    """scikit-learn 1.9 `contingency_matrix` (dense): exact counts from the
    device grouping, rows the sorted classes, columns the sorted clusters;
    `eps` adds a constant and makes the result Float64. sparse=True is
    NOT IMPLEMENTED (no sparse container in this package)."""
    if eps is not None and sparse:
        raise ValueError("Cannot set 'eps' when sparse=True")
    if sparse:
        raise NotImplementedError("mojolearn contingency_matrix: sparse=True is NOT IMPLEMENTED; this "
                                  "package has no sparse matrix type (metrics/NOT_IMPLEMENTED.tsv)")
    a, b, ca, cb = _clusterings(labels_true, labels_pred, "contingency_matrix")
    C = _contingency(a, b, ca, cb, numeric_mode)
    flat = [v for row in C for v in row]
    if eps is not None:
        return Array.from_list([v + float(eps) for v in flat], "<f8").reshape((len(ca), len(cb)))
    code = {"int64": "<i8", "int32": "<i4", "float64": "<f8", "float32": "<f4"}.get(
        dtype if isinstance(dtype, str) else getattr(dtype, "__name__", str(dtype)), "<i8")
    return Array.from_list([float(v) if code[1] == "f" else int(v) for v in flat], code).reshape((len(ca), len(cb)))


def pair_confusion_matrix(labels_true, labels_pred, *, numeric_mode=None):
    """scikit-learn 1.9 `pair_confusion_matrix`: the 2 x 2 Int64 pair counts
    from the exact contingency matrix (Python integers, no overflow)."""
    a, b, ca, cb = _clusterings(labels_true, labels_pred, "pair_confusion_matrix")
    C = _contingency(a, b, ca, cb, numeric_mode)
    n = len(a)
    n_c = [sum(row) for row in C]
    n_k = [sum(C[i][j] for i in range(len(C))) for j in range(len(cb))]
    sq = sum(v * v for row in C for v in row)
    c11 = sq - n
    c01 = sum(C[i][j] * n_k[j] for i in range(len(C)) for j in range(len(cb))) - sq
    c10 = sum(C[i][j] * n_c[i] for i in range(len(C)) for j in range(len(cb))) - sq
    c00 = n * n - c01 - c10 - sq
    return Array.from_list([c00, c01, c10, c11], "<i8").reshape((2, 2))


def _entropy_counts(counts):
    total = sum(counts)
    if total == 0:
        return 1.0
    lt = pmath.log(total)
    return -_fsum([(c / total) * (pmath.log(c) - lt) for c in counts if c])


def _mi_from_contingency(C, numeric_mode=None):
    fn = _epilogue("x_metrics_mi_contingency", numeric_mode)
    if fn is not None and C and C[0]:
        # x_metrics/epilogue.mojo mi_contingency (lane py-misc-metrics): the
        # same terms per nonzero cell, the two loop-invariant logs hoisted
        try:
            flat = array.array("q", itertools.chain.from_iterable(C))
            return float(fn(flat.buffer_info()[0], len(C), len(C[0])))
        except Exception:
            pass
    total = sum(v for row in C for v in row)
    pi = [sum(row) for row in C]
    pj = [sum(C[i][j] for i in range(len(C))) for j in range(len(C[0]))]
    if len(pi) == 1 or len(pj) == 1:
        return 0.0
    lt = pmath.log(total)
    terms = []
    for i, row in enumerate(C):
        for j, v in enumerate(row):
            if not v:
                continue
            nm = v / total
            log_outer = -pmath.log(pi[i] * pj[j]) + pmath.log(sum(pi)) + pmath.log(sum(pj))
            t = nm * (pmath.log(v) - lt) + nm * log_outer
            terms.append(0.0 if abs(t) < 2.220446049250313e-16 else t)
    return max(_fsum(terms), 0.0)


def _generalized_average(U, V, method):
    if method == "min":
        return min(U, V)
    if method == "geometric":
        return pmath.sqrt(U * V)
    if method == "arithmetic":
        return (U + V) / 2
    if method == "max":
        return max(U, V)
    raise ValueError("'average_method' must be 'min', 'geometric', 'arithmetic', or 'max'")


def normalized_mutual_info_score(labels_true, labels_pred, *, average_method="arithmetic", numeric_mode=None):
    """scikit-learn 1.9 `normalized_mutual_info_score`, in nats as scikit-learn
    (exact device contingency counts; the logarithms are the portable binary64
    ones of `_portable_math`, DEVIATION 6106)."""
    _generalized_average(1.0, 1.0, average_method)
    a, b, ca, cb = _clusterings(labels_true, labels_pred, "normalized_mutual_info_score")
    if len(ca) == len(cb) == 1 or len(ca) == len(cb) == 0:
        return 1.0
    C = _contingency(a, b, ca, cb, numeric_mode)
    mi = _mi_from_contingency(C, numeric_mode)
    if mi == 0:
        return 0.0
    ht = _entropy_counts([sum(r) for r in C])
    hp = _entropy_counts([sum(C[i][j] for i in range(len(C))) for j in range(len(cb))])
    return float(mi / _generalized_average(ht, hp, average_method))


def _expected_mi(a_counts, b_counts, n, numeric_mode=None):
    """E[MI] under the permutation model (Vinh, Epps and Bailey 2010), the sum
    scikit-learn's `expected_mutual_information` evaluates through gammaln.
    Here each hypergeometric pmf is built by its ratio recurrence from the
    mode and normalized by its own sum over the full support: no difference
    of large log-gamma values, portable binary64 log / exp only."""
    if len(a_counts) == 1 or len(b_counts) == 1:
        return 0.0
    # the same walks and terms in the binding's host binary64, the log the
    # C of mojolearn._portable_math.log (x_metrics/epilogue.mojo
    # expected_mi; lane metrics-apple2)
    from ._buffer import hotpath_enabled
    fn = getattr(_binding(numeric_mode), "x_metrics_expected_mi", None) if hotpath_enabled() else None
    if fn is not None and 0 < n < (1 << 31):
        A = array.array("q", a_counts)
        B = array.array("q", b_counts)
        try:
            return float(fn(A.buffer_info()[0], len(A), B.buffer_info()[0], len(B), n))
        except Exception:
            pass
    # Each walk away from the mode stops at the first u that is exactly 0:
    # every later u is that 0 times a finite ratio, so it adds nothing to z
    # (an exact sum) and its term is skipped as pr == 0 (the same bits as
    # walking the whole support; lane metrics-apple).
    emi_terms = []
    logs = {}

    def plog(v):
        r = logs.get(v)
        if r is None:
            r = logs[v] = pmath.log(v)
        return r
    for a in a_counts:
        la = plog(a)
        for b in b_counts:
            lb = plog(b)
            lo, hi = max(0, a + b - n), min(a, b)
            mode = min(max((a + 1) * (b + 1) // (n + 2), lo), hi)
            up = [1.0]
            x, v = mode, 1.0
            while x < hi:
                v = v * ((a - x) * (b - x)) / ((x + 1) * (n - a - b + x + 1))
                if v == 0:
                    break
                up.append(v)
                x += 1
            down = []
            x, v = mode, 1.0
            while x > lo:
                v = v * (x * (n - a - b + x)) / ((a - x + 1) * (b - x + 1))
                if v == 0:
                    break
                down.append(v)
                x -= 1
            z = _fsum(up + down)
            first = mode - len(down)
            for nij, u in zip(range(first, mode + len(up)), down[::-1] + up):
                if nij < 1:
                    continue
                pr = u / z
                if pr == 0:
                    continue
                emi_terms.append((nij / n) * (plog(n * nij) - la - lb) * pr)
    return _fsum(emi_terms)


def adjusted_mutual_info_score(labels_true, labels_pred, *, average_method="arithmetic", numeric_mode=None):
    """scikit-learn 1.9 `adjusted_mutual_info_score`: (MI - E[MI]) /
    (mean(H) - E[MI]) with scikit-learn's epsilon guards."""
    _generalized_average(1.0, 1.0, average_method)
    a, b, ca, cb = _clusterings(labels_true, labels_pred, "adjusted_mutual_info_score")
    if len(ca) == len(cb) == 1 or len(ca) == len(cb) == 0:
        return 1.0
    if len(ca) == 1 or len(cb) == 1:
        return 0.0
    C = _contingency(a, b, ca, cb, numeric_mode)
    n = len(a)
    mi = _mi_from_contingency(C, numeric_mode)
    rows = [sum(r) for r in C]
    cols = [sum(C[i][j] for i in range(len(C))) for j in range(len(cb))]
    emi = _expected_mi(rows, cols, n, numeric_mode)
    norm = _generalized_average(_entropy_counts(rows), _entropy_counts(cols), average_method)
    eps = 2.220446049250313e-16
    den = norm - emi
    den = min(den, -eps) if den < 0 else max(den, eps)
    num = mi - emi
    num = min(num, -eps) if num < 0 else max(num, eps)
    return float(num / den)


def _cluster_inputs(X, labels, caller):
    from ._metrics_impl import _classification_encoded, _label_set
    Xa = as_f32_c(X, ndim=2, name="X")[0]
    if not all_finite(Xa):
        raise ValueError(f"mojolearn {caller}: X contains NaN or infinity")
    lab, _ = _classification_encoded(labels, "labels")
    if len(lab) != Xa.shape[0]:
        raise ValueError(f"mojolearn {caller}: X and labels have different numbers of rows")
    classes = sorted(_label_set(lab))
    if not 1 < len(classes) < Xa.shape[0]:
        raise ValueError(f"Number of labels is {len(classes)}. Valid values are 2 to n_samples - 1 (inclusive)")
    return Xa, _codes(lab, classes), len(classes)


def _centroids(Xa, codes, k, numeric_mode):
    """(per-cluster sums (k x d), counts, global sum) from device PairSums."""
    n, d = Xa.shape
    prog = _Prog()
    X = prog.put(Xa)
    L = prog.put_i32(codes)
    off, sums = _group(prog, L, n, k, values=X, vstride=d, width=d)
    zero = prog.scratch(n)
    prog.stage("pair_key", n, 0, 0, zero, 1, 2)
    _, gsum = _group(prog, zero, n, 1, values=X, vstride=d, width=d)
    _execute(prog, numeric_mode)
    o = prog.ints(off, k + 1)
    counts = [o[i + 1] - o[i] for i in range(k)]
    return prog.floats(sums, k * d), counts, prog.floats(gsum, d)


class _Cents:
    """lane py-misc-metrics: `_centroids`' program left in place for the
    host epilogue (x_metrics/epilogue.mojo centroids_f32, ch_extra,
    db_score), which reads the per-cluster sums and the global sum words in
    the arena instead of Python lists of k * d floats."""

    def __init__(self, Xa, codes, k, numeric_mode):
        n, d = Xa.shape
        prog = _Prog()
        X = prog.put(Xa)
        L = prog.put_i32(codes)
        off, self.sums = _group(prog, L, n, k, values=X, vstride=d, width=d)
        zero = prog.scratch(n)
        prog.stage("pair_key", n, 0, 0, zero, 1, 2)
        _, self.gsum = _group(prog, zero, n, 1, values=X, vstride=d, width=d)
        _execute(prog, numeric_mode)
        o = prog.ints(off, k + 1)
        self.counts = [o[i + 1] - o[i] for i in range(k)]
        self.q = array.array("q", self.counts)
        prog._check(self.sums, k * d)
        prog._check(self.gsum, d)
        self.prog, self.k, self.d = prog, k, d

    def args(self):
        return self.prog.arena.buffer_info()[0], self.sums

    def f32(self, fn):
        """The Float32 centroid words `_row_dists` uploads."""
        k, d = self.k, self.d
        out = empty((k * d,), "<f4")
        fn(*self.args(), self.q.buffer_info()[0], k, d, addr_ro(out, name="centroids"))
        return out


def _cluster_native(Xa, codes, k, numeric_mode, name):
    """(_Cents, the centroid entry, the score entry) when the binding has
    them, else None."""
    cfn = _epilogue("x_metrics_centroids", numeric_mode)
    sfn = _epilogue(name, numeric_mode)
    if cfn is None or sfn is None:
        return None
    return _Cents(Xa, codes, k, numeric_mode), cfn, sfn


def _row_dists_f32(Xa, codes, C32, k, root, numeric_mode):
    """`_row_dists` given the Float32 centroid words."""
    n, d = Xa.shape
    prog = _Prog()
    X = prog.put(Xa)
    L = prog.put_i32(codes)
    C = prog.put(C32)
    out = prog.scratch(n)
    prog.stage("row_centroid_dist", n, X, d, L, C, out, 1 if root else 0)
    off, per = _group(prog, L, n, k, values=out)
    _execute(prog, numeric_mode)
    return prog.floats(per, k)


def _row_dists(Xa, codes, cents, root, numeric_mode):
    n, d = Xa.shape
    k = len(cents) // d
    prog = _Prog()
    X = prog.put(Xa)
    L = prog.put_i32(codes)
    C = prog.put(Array.from_list([_f32(v) for v in cents], "<f4"))
    out = prog.scratch(n)
    prog.stage("row_centroid_dist", n, X, d, L, C, out, 1 if root else 0)
    off, per = _group(prog, L, n, k, values=out)
    _execute(prog, numeric_mode)
    return prog.floats(per, k)


def calinski_harabasz_score(X, labels, *, numeric_mode=None):
    """scikit-learn 1.9 `calinski_harabasz_score`: the between- over the
    within-cluster dispersion, scaled by (n - k) / (k - 1)."""
    Xa, codes, k = _cluster_inputs(X, labels, "calinski_harabasz_score")
    n, d = Xa.shape
    nat = _cluster_native(Xa, codes, k, numeric_mode, "x_metrics_ch_extra")
    if nat is not None:
        cen, cfn, sfn = nat
        try:
            # x_metrics/epilogue.mojo ch_extra (lane py-misc-metrics)
            C32 = cen.f32(cfn)
            extra = float(sfn(*cen.args(), cen.gsum, cen.q.buffer_info()[0], k, d, n))
        except Exception:
            C32 = None
        if C32 is not None:
            intra = _fsum(_row_dists_f32(Xa, codes, C32, k, False, numeric_mode))
            return float(1.0 if intra == 0.0 else extra * (n - k) / (intra * (k - 1.0)))
    sums, counts, gsum = _centroids(Xa, codes, k, numeric_mode)
    cents = [sums[i * d + c] / counts[i] for i in range(k) for c in range(d)]
    mean = [v / n for v in gsum]
    extra = _fsum([counts[i] * _fsum([_sq(cents[i * d + c] - mean[c]) for c in range(d)])
                        for i in range(k)])
    intra = _fsum(_row_dists(Xa, codes, cents, False, numeric_mode))
    return float(1.0 if intra == 0.0 else extra * (n - k) / (intra * (k - 1.0)))


def davies_bouldin_score(X, labels, *, numeric_mode=None):
    """scikit-learn 1.9 `davies_bouldin_score`: the mean over clusters of the
    worst (s_i + s_j) / d(c_i, c_j)."""
    Xa, codes, k = _cluster_inputs(X, labels, "davies_bouldin_score")
    n, d = Xa.shape
    nat = _cluster_native(Xa, codes, k, numeric_mode, "x_metrics_db_score")
    if nat is not None:
        cen, cfn, sfn = nat
        try:
            C32 = cen.f32(cfn)
        except Exception:
            C32 = None
        if C32 is not None:
            per = array.array("d", _row_dists_f32(Xa, codes, C32, k, True, numeric_mode))
            try:
                # x_metrics/epilogue.mojo db_score (lane py-misc-metrics)
                return float(sfn(*cen.args(), cen.q.buffer_info()[0], k, d, per.buffer_info()[0]))
            except Exception:
                pass
    sums, counts, _ = _centroids(Xa, codes, k, numeric_mode)
    cents = [sums[i * d + c] / counts[i] for i in range(k) for c in range(d)]
    intra = [v / counts[i] for i, v in enumerate(_row_dists(Xa, codes, cents, True, numeric_mode))]
    dist = [[pmath.sqrt(_fsum([_sq(cents[i * d + c] - cents[j * d + c]) for c in range(d)]))
             for j in range(k)] for i in range(k)]
    close = lambda v: abs(v) <= 1e-8
    if all(close(v) for v in intra) or all(close(v) for row in dist for v in row):
        return 0.0
    scores = []
    for i in range(k):
        best = float("-inf")
        for j in range(k):
            den = dist[i][j] if dist[i][j] != 0 else float("inf")
            best = max(best, (intra[i] + intra[j]) / den)
        scores.append(best)
    return float(_fsum(scores) / k)


# ---------------------------------------------------------------------------
# The splitters' random draws (DEVIATION 6108; x_metrics/split.mojo)
# ---------------------------------------------------------------------------

_M64 = (1 << 64) - 1


def _mix64(x):
    """splitmix64's finalizer in exact Python integers."""
    x &= _M64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _M64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _M64
    return x ^ (x >> 31)


class CounterRng:
    """The splitters' random source: draw k of seed s is a device
    permutation keyed by `_mix64(s * GOLDEN + k)`. An int seed always
    gives the same sequence of draws; numpy RandomState / Generator objects
    are refused by name (their stream is not ours to reproduce)."""

    def __init__(self, random_state):
        import os
        if random_state is None:
            random_state = int.from_bytes(os.urandom(8), "little")
        if is_bool(random_state) or not isinstance(random_state, numbers.Integral) or random_state < 0:
            raise ValueError("mojolearn model_selection: random_state must be None or a non-negative int; "
                             "numpy RandomState / Generator objects are NOT IMPLEMENTED (their stream "
                             "is numpy's, x_metrics/split.mojo DEVIATION 6108)")
        self.seed = int(random_state) & _M64
        self.draws = 0
        self.binding = _BINDING

    def _salt(self):
        salt = _mix64(self.seed * 0x9E3779B97F4A7C15 + self.draws + 1)
        self.draws += 1
        return salt

    def permute_stage(self, prog, n):
        """The next draw as a `permute` stage of `prog`; returns its slot."""
        salt = self._salt()
        out = prog.alloc(max(n, 1))
        lo = salt & 0xFFFFFFFF
        hi = salt >> 32
        prog.stage("permute", 1, n, out, lo - (1 << 32) if lo >= 1 << 31 else lo,
                   hi - (1 << 32) if hi >= 1 << 31 else hi)
        return out

    def permutations(self, sizes, numeric_mode=None):
        """One permutation per size, drawn in order, in one device program."""
        prog = _Prog()
        outs = []
        for n in sizes:
            outs.append((prog.want(self.permute_stage(prog, n), n), n))
        if not outs:
            return []
        _execute(prog, numeric_mode)
        return [prog.ints(o, n) for o, n in outs]

    def permutation(self, n, numeric_mode=None):
        return self.permutations([n], numeric_mode)[0]

    def permutation_rows(self, sizes, numeric_mode=None):
        """`permutations` (the same draws) as array('q') rows: the device
        widens each to Int64 words (`rows64`), so no Python int is made
        per row (lane metrics-apple2)."""
        prog = _Prog()
        outs = []
        for n in sizes:
            o = self.permute_stage(prog, n)
            w = prog.want(prog.alloc(2 * n), 2 * n)
            if n:
                prog.stage("rows64", n, o, w)
            outs.append((w, n))
        if not outs:
            return []
        _execute(prog, numeric_mode)
        return [prog.words(w, 2 * n, "q") for w, n in outs]


#: the largest K-fold row table (2 words per row per fold) `fold_rows` builds
_FOLD_ROWS_BOUND = 1 << 27


def fold_rows(n, k, *, codes=None, rng=None, numeric_mode=None):
    """[(train, test)] Int64 row Arrays of a K-fold split, ascending, built
    by the `fold_rows` unit (lane metrics-apple2): `codes` = each row's
    test fold (bytes, values < 256), or `rng` = KFold's shuffle (the rows
    of fold f are the positions of fold f in the rng's next permutation).
    The rows come back as Int64 words, so no Python int is made per row.
    Returns None when the table would exceed `_FOLD_ROWS_BOUND` words (the
    caller keeps its own path)."""
    if n < 2 or k < 1 or 2 * n * k > _FOLD_ROWS_BOUND:
        return None
    prog = _Prog()
    if codes is None:
        order = rng.permute_stage(prog, n)
        code = prog.scratch(n)
    else:
        words = bytearray(4 * n)
        words[0 if _LITTLE else 3::4] = codes
        store = array.array("i")
        store.frombytes(words)
        code = prog.put_i32(Array._owned(store, (n,), "<i4", "C"))
        order = _NONE
    return _fold_rows_run(prog, n, k, code, order, numeric_mode)


def _fold_rows_run(prog, n, k, code, order, numeric_mode):
    out = prog.want(prog.alloc(2 * n * k), 2 * n * k)
    sz = prog.want(prog.alloc(k), k)
    prog.stage("fold_rows", 1, n, k, code, order, out, sz)
    _execute(prog, numeric_mode)
    sizes = prog.ints(sz, k)
    res = []
    for f in range(k):
        c = sizes[f]
        base = out + 2 * n * f
        test = prog.words(base, 2 * c, "q")
        train = prog.words(base + 2 * c, 2 * (n - c), "q")
        res.append((Array._owned(train, (n - c,), "<i8", "C"), Array._owned(test, (c,), "<i8", "C")))
    return res


def stratified_fold_rows(enc, counts, alloc, k, rng, numeric_mode=None):
    """StratifiedKFold's [(train, test)] in ONE device program (lane
    metrics-apple2): a stable group_sort of the first-seen class codes
    `enc`, each class's permutation (the rng's next draws, in class order,
    the draws `_fold_of_rows` makes) laid out at PB + the class's offset,
    `strat_codes` (each row's fold, x_metrics/split.mojo), then `fold_rows`.
    None when the table would exceed `_FOLD_ROWS_BOUND` words."""
    n, m = len(enc), len(counts)
    if n < 2 or k < 1 or 2 * n * k > _FOLD_ROWS_BOUND:
        return None
    prog = _Prog()
    ENC = prog.put_i32(Array._owned(array.array("i", enc), (n,), "<i4", "C"))
    OFF = prog.alloc(m + 1)
    ORD = prog.scratch(n)
    prog.stage("group_sort", 1, ENC, n, m, OFF, ORD)
    PB = _NONE
    if rng is not None:
        at = 0
        for c in range(m):
            o = rng.permute_stage(prog, counts[c])
            if c == 0:
                PB = o
            if o != PB + at:
                raise AssertionError("stratified_fold_rows: the class permutations are not contiguous")
            at += counts[c]
    cum = []
    for c in range(m):
        acc = 0
        cum.append(0)
        for f in range(k):
            acc += alloc[f][c]
            cum.append(acc)
    CUM = prog.put_i32(Array._owned(array.array("i", cum), (len(cum),), "<i4", "C"))
    code = prog.scratch(n)
    prog.stage("strat_codes", n, ENC, ORD, OFF, PB, CUM, k, code)
    return _fold_rows_run(prog, n, k, code, _NONE, numeric_mode)
