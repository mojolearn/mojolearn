# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE METRICS LANE'S DOOR.

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
x_metrics/epilogue.mojo under the same operation rules, the only route (lane
pyglue-sweep, 2026-10-03: the Python fallbacks and the
MOJOLEARN_METRICS_EPILOGUE=python reference arm are gone). That is bitwise the same on every machine by the IEEE standard, and it
is scikit-learn's own precision for the same step.
"""
import array
import itertools
from . import _portable_math as _math
import numbers
import operator
import warnings

from . import _backend
from . import _portable_math as pmath
from . import _arena_io
from ._array import Array
from ._buffer import as_f32_c, addr_ro, all_finite, empty
from ._labels import is_bool

__all__ = []

_BINDING = "_mojolearn_x_metrics"

#: op name -> id; x_metrics/units.mojo `run_unit` holds the same table.
_OPS = dict(group_sort=0, group_sum=1, pair_key=2, reg_term=3, col_sort=4, wpercentile=5, col_max=6, bin_curve=7, row_metric=8, row_centroid_dist=9, permute=10,
            fold_rows=36, rows64=41, strat_codes=45, curve_fold=46, onehot=52, rep_rows=53, pair_cols=54,
            # lane fam2-prep-metrics (x_metrics/cls_epi.mojo), both numeric modes
            # since lane cpu2-l7-metrics
            cls_epi=55,
            # lane cpu2-l7-metrics: the metric tails and scans on the device
            # (x_metrics/tail.mojo, cm_epi.mojo, reg_epi.mojo, rank_epi.mojo)
            off_diff=56, flag_scan=57, proba_rows=58, cm_epi=59, reg_epi=60, rank_epi=61, cl_epi=62,
            # lane cpu4-python: the curve arrays on the device (x_metrics/curve_out.mojo)
            curve_out=63, auc_xy=67,
            # lane cpu4-python: the contingency statistics, MI and expected MI
            # on the device (x_metrics/contingency.mojo)
            cont_stats=70,
            # lane py-runtime-b: StratifiedShuffleSplit's per-class draw counts
            # (x_metrics/split.mojo approx_mode_unit)
            approx_mode=81,
            # lane ne2-b-search-workers: permutation_test_score's within-group
            # permutation as one program (x_metrics/split.mojo group_take_unit, take_rows64_unit)
            group_take=85, take_rows64=86,
            # lane ne2-b-search-workers: one StratifiedShuffleSplit split as one
            # program (x_metrics/split.mojo permute_drawn_unit, strat_take_unit)
            permute_drawn=87, strat_take=88)
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


def _arena_view(arena):
    """A float32 memoryview over a program's arena."""
    return memoryview(arena)


def _i32_c(codes):
    """int32 C-order codes as an `Array` (no copy when they already are)."""
    if isinstance(codes, Array):
        if codes.dtype == "<i4" and codes._has_order("C"):
            return codes
        return codes.astype("<i4")._as_c()
    return Array.from_list(list(codes), "<i4")


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
        #: (MATCH, W, n) of a weighted `_Sums` program, for a tail that sums
        #: the rows itself (`_wsum_sf64`); None otherwise
        self.rows = None

    def alloc(self, n):
        off = self.size
        self.size += max(int(n), 0)
        if self.size > 2147483647:
            raise ValueError("mojolearn metrics: the problem exceeds the Int32 arena bound")
        return off

    def put(self, arr):
        """A float32 C-contiguous Array (or anything as_f32_c takes) -> offset.
        A float32 `_arena_io.DeviceRows` (a fold's rows, lane cpu4-misc) is
        kept as it is: `run` gathers it on the device, or copies its
        host rows when the binding cannot."""
        if isinstance(arr, _arena_io.DeviceRows) and arr.dtype != "<f4":
            arr = arr.materialize()
        if not (isinstance(arr, _arena_io.DeviceRows)
                or (isinstance(arr, Array) and arr.dtype == "<f4" and arr._has_order("C"))):
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
        codes = _i32_c(codes)
        off = self.alloc(codes.size)
        self._inputs.append((off, codes))
        self._skip.append((off, off + codes.size))
        return off

    def stage(self, op, total, *params):
        if len(params) > _PARAMS:
            raise ValueError("x_metrics: too many stage parameters")
        self._stages.append([_OPS[op], int(total)] + [int(v) for v in params]  # glue: one stage's parameter words
                            + [0] * (_PARAMS - len(params)))

    def run(self, numeric_mode):
        """Run the program on the x_metrics binding (GPU, or the host binding
        on a CPU-only install): one native call that owns the arena, the
        input copies, resident and device-gathered inputs, the range
        descriptors and the run (bindings/arena_prog_native.mojo)."""
        return _PROG_CALL(self, "x_metrics_run", (numeric_mode,), _PROG_CONTEXT)

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
            for lo, hi in self._skip:  # glue: the program's input and scratch ranges
                if off < hi and lo < off + n:
                    raise AssertionError(f"x_metrics: arena [{off}, {off + n}) was read but is an input "
                                         "or scratch slot, which never comes back")
            return
        for lo, hi, cn, mult in self._outs:  # glue: the program's declared output ranges
            if cn >= 0 and self.arena is not None:
                hi = lo + max(0, min(hi - lo, mult * self.ints(cn, 1)[0]))
            if lo <= off and off + n <= hi:
                return
        raise AssertionError(f"x_metrics: arena [{off}, {off + n}) was read but never declared an output")

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
        store.frombytes(_arena_view(self.arena)[off:off + n].cast("B"))
        return store

    def get(self, off, shape):
        shape = tuple(shape) if isinstance(shape, (tuple, list)) else (int(shape),)
        n = 1
        for s in shape:  # glue: the product of shape dims
            n *= s
        self._check(off, n)
        return Array._owned(self.arena[off:off + n], shape, "<f4", "C")


def _run(prog, numeric_mode):
    """`prog.run(numeric_mode)` as a module function, so a helper that takes
    a program as a parameter names its native call statically."""
    return _PROG_CALL(prog, "x_metrics_run", (numeric_mode,), _PROG_CONTEXT)


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


def _counts(prog, off, m):
    """The m group sizes of a `group_sort` OFF table as an Int32 slot the
    device fills (`off_diff`, x_metrics/tail.mojo); read with prog.ints."""
    out = prog.alloc(m)
    if m:
        prog.stage("off_diff", m, off, m, out)
    return out


#: flag_scan tests (x_metrics/tail.mojo)
_SCAN_LOG_DOMAIN, _SCAN_INDICATOR, _SCAN_NEGATIVE = 0, 1, 2


def _scan_flag(prog, src, n, mode):
    """One Int32 flag word the device sets to 1 when any of the n Float32
    values at `src` fails the test (`flag_scan`); 0 otherwise."""
    flag = prog.alloc(1)
    if n:
        prog.stage("flag_scan", n, src, n, mode, flag)
    return flag


def _put_proba(prog, a, n, k, binary):
    """The probabilities' check and layout in the program (`proba_rows`,
    x_metrics/tail.mojo): (S, FLAGS). S is the (n, k) Float32 matrix the
    metric reads: `a` itself, or for a binary column the `[1 - p, p]` rows
    the device packs. FLAGS: three Int32 words (non-finite, outside [0, 1],
    a row sum off one), read by `_proba_check` after the run."""
    A = prog.put(a)
    flags = prog.alloc(3)
    S = A
    if binary:
        S = prog.scratch(2 * n)
    if n:
        prog.stage("proba_rows", n, A, n, 1 if binary else k, 1 if binary else 0, S, flags)
    return S, flags


#: probability_rows codes 1..3, in the order the check reports them
_PROBA_REASONS = ("must be finite", "must lie in [0, 1]", "rows must sum to one within sqrt(float32 eps)")


def _proba_code(prog, flags):
    """0, or the first failed check (1 non-finite, 2 outside [0, 1], 3 a row sum)."""
    f = prog.ints(flags, 3)
    return 1 if f[0] else (2 if f[1] else (3 if f[2] else 0))


def _proba_check(prog, flags, caller):
    code = _proba_code(prog, flags)
    if code:
        raise ValueError(f"mojolearn {caller}: y_proba " + _PROBA_REASONS[code - 1])


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
    # Lane ne1-h-labels (N414, N438): the multilabel refusals, both sides'
    # reads and encodes, the length / kind / INT32 checks and the sorted
    # union in ONE native call (bindings/classical_labels.mojo
    # `labels_union_encode`); each side comes back as its sorted classes and
    # int32 codes, so `_codes` maps the k classes and gathers.
    from ._metrics_impl import _EncodedLabels
    from ._native_labels_bridge import labels_entry, LABELS_OP_UNION_ENCODE
    entry, context = labels_entry()
    true_classes, true_codes, pred_classes, pred_codes, kind, present = entry(
        LABELS_OP_UNION_ENCODE, (y_true, y_pred, caller), context)
    true = _EncodedLabels(true_classes, true_codes)
    pred = _EncodedLabels(pred_classes, pred_codes)
    w = _weights(sample_weight, len(true), caller)
    return true, pred, kind, present, w


def _codes(labels_seq, order):
    from ._metrics_impl import _label_map
    index = {c: i for i, c in enumerate(order)}  # glue: class index table
    return _label_map(labels_seq, lambda v: index.get(v, -1))


class UndefinedMetricWarning(UserWarning):
    """scikit-learn's `UndefinedMetricWarning` (a UserWarning): a metric that
    is ill-defined for the input and was set to a fallback value. Defined
    here so the metrics do not import scikit-learn."""


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

#: cm_epi kinds (x_metrics/cm_epi.mojo, op 59; lane cpu2-l7-metrics)
_CM_MLCM, _CM_MCC, _CM_COUNT, _CM_NORM, _CM_KAPPA, _CM_CLR, _CM_TOTAL, _CM_WSUM_LEAF, _CM_WSUM_FOLD = range(9)
#: rows per `_wsum_leaf` unit (x_metrics/cm_epi.mojo WSUM_BLOCK)
_WSUM_BLOCK = 256
#: the NORM modes: the 'all' total, then 'all', 'true', 'pred', none
_NORM_MODES = {"all": 1, "true": 2, "pred": 3, None: 4}
#: the kappa weight modes
_KAPPA_W = {None: 0, "linear": 1, "quadratic": 2}


def _wsum_sf64(prog, k):
    """Lane box-run-2-fix (2026-10-05): stage the weighted COUNT's hit and
    weight total as soft binary64 sums on the device (x_metrics/cm_epi.mojo
    `_wsum_leaf`: one unit per WSUM_BLOCK rows, then `_wsum_fold`: one stage
    per tree level, stride 1, 2, 4, ...), the same stages on every column.
    Returns the slot whose first 4 words are [hit, total]. Before, both were
    Float32 PairSums widened to binary64 (precision loss in score with
    sample_weight)."""
    match, W, n = prog.rows
    nb = max(1, -(-n // _WSUM_BLOCK))
    s = prog.scratch(4 * nb)
    prog.stage("cm_epi", nb, _CM_WSUM_LEAF, match, W, n, s, k)
    stride = 1
    while stride < nb:  # glue: one stage per fold level (log2 of the block count)
        prog.stage("cm_epi", -(-(nb - stride) // (2 * stride)), _CM_WSUM_FOLD, s, stride, nb)
        stride *= 2
    return s


# Engineering notes (moved from the docstring):
# Per-label one-vs-rest 2x2 confusion matrices, `[[tn, fp], [fn, tp]]`.
#
# scikit-learn 1.9 `multilabel_confusion_matrix` for binary and multiclass
# 1-D targets (strings or integers). Unweighted counts are exact Int64;
# with sample_weight they are Float64 images of Float32 per-label PairSums
# (DEVIATION 6100). Multilabel-indicator input and samplewise=True are
# refused by name (metrics/NOT_IMPLEMENTED.tsv).
def multilabel_confusion_matrix(y_true, y_pred, *, sample_weight=None, labels=None,
                                samplewise=False, numeric_mode=None):
    """Per-label one-vs-rest 2x2 confusion matrices, ``[[tn, fp], [fn, tp]]``.

    Follows scikit-learn's `multilabel_confusion_matrix` for binary and multiclass 1-D
    targets (strings or integers). Unweighted counts are exact integers; with
    `sample_weight` they are float64. Multilabel-indicator input and ``samplewise=True``
    raise an error.
    """
    return _NATIVE(None, _OP.family_mexp_cls_mlcm,
                   (y_true, y_pred, sample_weight, labels, samplewise, numeric_mode),
                   _NATIVE_CONTEXT)


# Engineering notes (moved from the docstring):
# scikit-learn 1.9 `precision_recall_fscore_support` for binary and
# multiclass 1-D targets: per-label sums on the device (exact integers, or
# Float32 PairSums of sample_weight), the ratios and averages in the
# binary64 epilogue (DEVIATION 6106). zero_division is 'warn', 0.0, 1.0 or
# np.nan; average='samples' (multilabel) is refused by name.
def precision_recall_fscore_support(y_true, y_pred, *, beta=1.0, labels=None, pos_label=1,
                                    average=None, warn_for=("precision", "recall", "f-score"),
                                    sample_weight=None, zero_division="warn", numeric_mode=None):
    """Precision, recall, F-score and support for each label.

    Follows scikit-learn's `precision_recall_fscore_support` for binary and multiclass
    1-D targets. `zero_division` is ``'warn'``, 0.0, 1.0 or ``np.nan``.
    ``average='samples'`` (multilabel) raises an error.
    """
    return _NATIVE(None, _OP.family_mexp_cls_prfs,
                   (y_true, y_pred, beta, labels, pos_label, average, warn_for, sample_weight, zero_division, numeric_mode),
                   _NATIVE_CONTEXT)


def fbeta_score(y_true, y_pred, *, beta, labels=None, pos_label=1, average="binary",
                sample_weight=None, zero_division="warn", numeric_mode=None):
    """scikit-learn 1.9 `fbeta_score` (binary / multiclass 1-D targets); see
    precision_recall_fscore_support."""
    return _NATIVE(None, _OP.family_mexp_cls_fbeta,
                   (y_true, y_pred, beta, labels, pos_label, average, sample_weight, zero_division, numeric_mode),
                   _NATIVE_CONTEXT)


def jaccard_score(y_true, y_pred, *, labels=None, pos_label=1, average="binary",
                  sample_weight=None, zero_division="warn", numeric_mode=None):
    """scikit-learn 1.9 `jaccard_score`, `tp / (tp + fp + fn)` per label,
    for binary and multiclass 1-D targets; see precision_recall_fscore_support
    for the sums and the epilogue."""
    return _NATIVE(None, _OP.family_mexp_cls_jaccard,
                   (y_true, y_pred, labels, pos_label, average, sample_weight, zero_division, numeric_mode),
                   _NATIVE_CONTEXT)


def balanced_accuracy_score(y_true, y_pred, *, sample_weight=None, adjusted=False, numeric_mode=None):
    """scikit-learn 1.9 `balanced_accuracy_score`: the mean per-class recall
    over the classes present in y_true (a class with no true weight is
    dropped with sklearn's warning), chance-adjusted when `adjusted`."""
    return _NATIVE(None, _OP.family_mexp_cls_balanced_accuracy,
                   (y_true, y_pred, sample_weight, adjusted, numeric_mode),
                   _NATIVE_CONTEXT)


def matthews_corrcoef(y_true, y_pred, *, sample_weight=None, numeric_mode=None):
    """scikit-learn 1.9 `matthews_corrcoef` (binary and multiclass): the
    covariance form over the per-class true / predicted sums and the trace."""
    return _NATIVE(None, _OP.family_mexp_cls_matthews, (y_true, y_pred, sample_weight, numeric_mode), _NATIVE_CONTEXT)


def cohen_kappa_score(y1, y2, *, labels=None, weights=None, sample_weight=None,
                      replace_undefined_by=float("nan"), numeric_mode=None):
    """scikit-learn 1.9 `cohen_kappa_score`: the (weighted) confusion matrix
    over `labels` on the device, the kappa in the binary64 epilogue;
    weights None, 'linear' or 'quadratic'."""
    return _NATIVE(None, _OP.family_mexp_cls_kappa,
                   (y1, y2, labels, weights, sample_weight, replace_undefined_by, numeric_mode),
                   _NATIVE_CONTEXT)


def hamming_loss(y_true, y_pred, *, sample_weight=None, numeric_mode=None):
    """scikit-learn 1.9 `hamming_loss` for binary / multiclass 1-D targets:
    the (weighted) fraction of mismatched labels."""
    return _NATIVE(None, _OP.family_mexp_cls_hamming, (y_true, y_pred, sample_weight, numeric_mode), _NATIVE_CONTEXT)


def zero_one_loss(y_true, y_pred, *, normalize=True, sample_weight=None, numeric_mode=None):
    """scikit-learn 1.9 `zero_one_loss` for binary / multiclass 1-D targets."""
    return _NATIVE(None, _OP.family_mexp_cls_zero_one,
                   (y_true, y_pred, sample_weight, normalize, numeric_mode),
                   _NATIVE_CONTEXT)


def accuracy_fraction(y_true, y_pred, sample_weight=None, numeric_mode=None):
    """sklearn's ClassifierMixin.score / accuracy_score over labels of any
    kind: the (weighted) match count over the row count or the weight total,
    both from the x_metrics binding's grouped sums (lane pyglue-sweep: the
    weight total is the program's PairSum, not a host sum; lane
    box-run-2-fix: the weighted hit and total are soft-f64 sums on the
    device, `_wsum_sf64`, no longer Float32 PairSums). Lane
    cgr4-py-compute: the estimators' `score` methods called this instead of
    a per-row Python comparison."""
    return _NATIVE(None, _OP.family_mcls_accuracy_fraction, (y_true, y_pred, sample_weight, numeric_mode),
                   _NATIVE_CONTEXT)


def class_likelihood_ratios(y_true, y_pred, *, labels=None, sample_weight=None,
                            replace_undefined_by=float("nan"), numeric_mode=None):
    """scikit-learn 1.9 `class_likelihood_ratios` (binary targets): LR+ and
    LR- from the 2 x 2 (weighted) confusion matrix; `replace_undefined_by`
    np.nan, 1.0 or {'LR+': v, 'LR-': v}."""
    return _NATIVE(None, _OP.family_mexp_cls_likelihood_ratios,
                   (y_true, y_pred, labels, sample_weight, replace_undefined_by, numeric_mode),
                   _NATIVE_CONTEXT)


def classification_report(y_true, y_pred, *, labels=None, target_names=None, sample_weight=None,
                          digits=2, output_dict=False, zero_division="warn", numeric_mode=None):
    """scikit-learn 1.9 `classification_report` for binary / multiclass 1-D
    targets, from precision_recall_fscore_support."""
    return _NATIVE(None, _OP.family_mexp_cls_report,
                   (y_true, y_pred, labels, target_names, sample_weight, digits, output_dict, zero_division, numeric_mode),
                   _NATIVE_CONTEXT)


# ---------------------------------------------------------------------------
# Regression (scikit-learn 1.9 sklearn/metrics/_regression.py)
# ---------------------------------------------------------------------------

_TERM = dict(sq=0, abs=1, sqlog=2, ape=3, pinball=4, tweedie=5, diff=6)


def _f32(x):
    """The Float32 nearest a binary64 value (round to nearest even)."""
    import struct
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


#: reg_epi (op 60, x_metrics/reg_epi.mojo): kinds, MEAN modes, AVG modes
_RE_MEAN, _RE_PCT, _RE_ASM, _RE_AVG, _RE_SIGN = 0, 1, 2, 3, 4
_RE_DIV, _RE_ROOT = 1, 2
_RE_UNIFORM, _RE_CUSTOM, _RE_VARIANCE = 1, 2, 3


class _Reg:
    """Validated regression targets: (n, D) Float32, finite, same shape."""

    def __init__(self, y_true, y_pred, sample_weight, multioutput, caller, *, variance_ok=False):
        from ._metrics_impl import _shape_of, _is_float64
        from ._buffer import materialize_f32_lists
        arrays = []
        for name, v in (("y_true", y_true), ("y_pred", y_pred)):  # glue: validates the two named arguments
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
            vals = [float(v) for v in flatten_mo(multioutput)]  # glue: converts the multioutput weights argument
            if self.D == 1:
                raise ValueError("Custom weights are useful only in multi-output cases.")
            if len(vals) != self.D:
                raise ValueError("There must be equally many custom weights "
                                 f"({len(vals)}) as outputs ({self.D}).")
            self.mo = vals
        self.caller = caller

    # Lane cpu2-l7-metrics (2026-10-04): every regression metric is ONE
    # program. The per-output tails (the means, roots, percentile flags, the
    # Float32 broadcast columns, the r2 / explained variance / d2 assembly,
    # the multioutput averages) are `reg_epi` stages (op 60,
    # x_metrics/reg_epi.mojo) in soft binary64 next to the folds that make
    # their inputs; only the final value(s) come back. DEVIATION 6106: the
    # same binary64 operations, in the same order, as the Python they replace.

    def program(self, scalar=None):
        """Start this metric's program: y, p, the kind's scalar (alpha or
        power, rounded to Float32), the weights, the one-group key and the
        weight total."""
        prog = self.prog = _Prog()
        self.Y, self.P = prog.put(self.y), prog.put(self.p)
        self.S = prog.put(Array.from_list([_f32(0.0 if scalar is None else scalar)], "<f4"))
        self.W = _NONE if self.w is None else prog.put(self.w)
        self.key = prog.scratch(self.n)
        prog.stage("pair_key", self.n, 0, 0, self.key, 1, 2)
        self.SW = _NONE if self.w is None else _group(prog, self.key, self.n, 1, weights=self.W)[1]
        return prog

    def term(self, kind, V=None, P=None, broadcast=False):
        """The n x D Float32 terms of `kind` (reg_term) of V (default y)
        against P (default p; D per-column values when broadcast)."""
        out = self.prog.scratch(self.n * self.D)
        self.prog.stage("reg_term", self.n * self.D, self.Y if V is None else V, self.P if P is None else P, out,
                        self.D, _TERM[kind], self.S, 1 if broadcast else 0)
        return out

    def sums(self, V):
        """The D per-column (weighted) Float32 PairSums of the n x D values at V."""
        return _group(self.prog, self.key, self.n, 1, values=V, vstride=self.D, weights=self.W, width=self.D)[1]

    def fin(self, sums, mode=_RE_DIV, *, f32=False):
        """reg_epi MEAN: D binary64 values from the Float32 sums, widened,
        over the weight total or n (_RE_DIV), rooted (_RE_ROOT); with f32,
        also (dst, d32) where d32 holds the Float32 nearest each, the
        broadcast column a later reg_term reads."""
        prog, D = self.prog, self.D
        dst = prog.alloc(2 * D)
        d32 = prog.scratch(D) if f32 else _NONE
        prog.stage("reg_epi", D, _RE_MEAN, D, sums, self.SW, self.n, dst, d32, mode)
        return (dst, d32) if f32 else dst

    def stage_mean(self, kind, mode=_RE_DIV, **kw):
        """The per-column (weighted) mean of the term, binary64 (DEVIATION 6106)."""
        return self.fin(self.sums(self.term(kind, **kw)), mode)

    def percentile(self, V, rank, *, average=True, f32=False):
        """Per-column weighted percentile of the n x D Float32 values at V
        (col_sort, wpercentile) as binary64, NaN for an all-zero weight
        column (reg_epi PCT); f32 as in `fin`."""
        prog, n, D = self.prog, self.n, self.D
        order = prog.scratch(n * D)
        prog.stage("col_sort", D, V, n, D, order)
        R = prog.put(Array.from_list([_f32(rank)], "<f4"))
        out = prog.scratch(D)
        cdf = prog.scratch(n * D)
        prog.stage("wpercentile", D, V, n, D, order, self.W, R, 1 if average else 0, out, cdf)
        dst = prog.alloc(2 * D)
        d32 = prog.scratch(D) if f32 else _NONE
        prog.stage("reg_epi", D, _RE_PCT, D, out, cdf, n, dst, d32)
        return (dst, d32) if f32 else dst

    def scan_sign(self, V):
        """Two flag words the device sets: a value < 0, a value <= 0 (reg_epi SIGN)."""
        flags = self.prog.alloc(2)
        self.prog.stage("reg_epi", self.n * self.D, _RE_SIGN, self.n * self.D, V, flags)
        return self.prog.want(flags, 2)

    def scan_log_domain(self):
        """One flag word: a y or p value <= -1 (flag_scan)."""
        N = self.n * self.D
        flag = _scan_flag(self.prog, self.Y, N, _SCAN_LOG_DOMAIN)
        self.prog.stage("flag_scan", N, self.P, N, _SCAN_LOG_DOMAIN, flag)
        return self.prog.want(flag, 1)

    def average(self, vals, *, nan_rule=False, variance=_NONE):
        """The multioutput answer of the D binary64 values at `vals`: the
        values themselves (raw_values), else one reg_epi AVG unit."""
        prog, D, mo = self.prog, self.D, self.mo
        if mo == "raw_values":
            return prog.want(vals, 2 * D)
        if mo == "uniform_average":
            mode, w = _RE_UNIFORM, _NONE
        elif mo == "variance_weighted":
            mode, w = _RE_VARIANCE, variance
        else:
            mode, w = _RE_CUSTOM, prog.put_i32(_words64(mo))
        dst = prog.alloc(2)
        prog.stage("reg_epi", 1, _RE_AVG, D, vals, mode, w, dst, 1 if nan_rule else 0)
        return prog.want(dst, 2)

    def result(self, out, numeric_mode, check=None):
        """Run the program, let `check(prog)` raise on its flags, then the
        answer: a float, or the per-output Float64 Array for raw_values."""
        _run(self.prog, numeric_mode)
        if check is not None:
            check(self.prog)
        if self.mo == "raw_values":
            return Array.from_list(list(self.prog.words(out, 2 * self.D, "d")), "<f8")
        return float(self.prog.words(out, 2, "d")[0])

    def scalar(self, out, numeric_mode, check=None):
        """Run the program; the one binary64 word pair at `out` (single output)."""
        self.prog.want(out, 2)
        _run(self.prog, numeric_mode)
        if check is not None:
            check(self.prog)
        return float(self.prog.words(out, 2, "d")[0])


def _words64(values):
    """binary64 values as their Int32 word pairs, low first (reg_epi ld64)."""
    store = array.array("i")
    store.frombytes(array.array("d", values).tobytes())
    return store


def flatten_mo(values):
    from ._labels import flatten_labels
    return flatten_labels(values)


def regression_error_options(name, y_true, y_pred, sample_weight, multioutput, numeric_mode):
    """mean_squared_error / mean_absolute_error / root_mean_squared_error
    with sample_weight, multioutput or 2-D targets (lane/metrics); the 1-D
    unweighted call keeps its original kernel and bits."""
    return _NATIVE(None, _OP.family_metric_regression_error,
                   (name, y_true, y_pred, sample_weight, multioutput, numeric_mode), _NATIVE_CONTEXT)


def mean_squared_log_error(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average",
                           numeric_mode=None):
    """scikit-learn 1.9 `mean_squared_log_error`: the mean of
    `(log1p(y) - log1p(p))^2` (portable log1p, IDENTITY_PATHS row 51);
    values <= -1 are refused as scikit-learn refuses them."""
    return _NATIVE(None, _OP.family_metric_msle, (y_true, y_pred, sample_weight, multioutput, numeric_mode),
                   _NATIVE_CONTEXT)


def root_mean_squared_log_error(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average",
                                numeric_mode=None):
    """scikit-learn 1.9 `root_mean_squared_log_error` (per-output root, then averaged)."""
    return _NATIVE(None, _OP.family_metric_rmsle, (y_true, y_pred, sample_weight, multioutput, numeric_mode),
                   _NATIVE_CONTEXT)


def mean_absolute_percentage_error(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average",
                                   numeric_mode=None):
    """scikit-learn 1.9 `mean_absolute_percentage_error`: `|p - y| / max(|y|,
    eps)` with numpy's float64 epsilon, averaged (not multiplied by 100)."""
    return _NATIVE(None, _OP.family_metric_mape, (y_true, y_pred, sample_weight, multioutput, numeric_mode),
                   _NATIVE_CONTEXT)


def mean_pinball_loss(y_true, y_pred, *, sample_weight=None, alpha=0.5, multioutput="uniform_average",
                      numeric_mode=None):
    """scikit-learn 1.9 `mean_pinball_loss` (alpha is rounded to Float32)."""
    return _NATIVE(None, _OP.family_metric_pinball, (y_true, y_pred, sample_weight, multioutput, numeric_mode, alpha),
                   _NATIVE_CONTEXT)


def median_absolute_error(y_true, y_pred, *, multioutput="uniform_average", sample_weight=None,
                          numeric_mode=None):
    """scikit-learn 1.9 `median_absolute_error`: the per-output median of
    ``|y - p|`` (the weighted percentile at 50 with averaging when weighted, the
    same answer as numpy's median when not), from a stable device sort."""
    return _NATIVE(None, _OP.family_metric_medae, (y_true, y_pred, sample_weight, multioutput, numeric_mode),
                   _NATIVE_CONTEXT)


def max_error(y_true, y_pred, *, numeric_mode=None):
    """scikit-learn 1.9 `max_error` (single output)."""
    return _NATIVE(None, _OP.family_metric_max_error, (y_true, y_pred, numeric_mode), _NATIVE_CONTEXT)


def _assemble(r, num, den, force_finite, *, average=True):
    """Stage the per-output scores `1 - num / den` (reg_epi ASM, the
    force_finite convention) and their multioutput average (reg_epi AVG
    with the NaN rule: a NaN score, or an infinite one with weight zero,
    answers NaN; variance_weighted weighs by `den` unless it is all zero)."""
    prog, D = r.prog, r.D
    scores = prog.alloc(2 * D)
    prog.stage("reg_epi", D, _RE_ASM, D, num, den, 1 if force_finite else 0, scores)
    return r.average(scores, nan_rule=True, variance=den) if average else scores


def explained_variance_score(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average",
                             force_finite=True, numeric_mode=None):
    """scikit-learn 1.9 `explained_variance_score`: `1 - Var(y - p) / Var(y)`
    with (weighted) means, one program (the means, then the centered
    squares); force_finite=False returns scikit-learn's NaN / -inf by value."""
    return _NATIVE(None, _OP.family_metric_explained_variance,
                   (y_true, y_pred, sample_weight, multioutput, force_finite, numeric_mode), _NATIVE_CONTEXT)


def _centered(r, V, M, *, mean=True):
    """Stage the per-column (weighted) mean (or the sum, widened) of
    (v - m_c)^2 over the n x D values at V and the Float32 column M:
    D binary64 values."""
    return r.fin(r.sums(r.term("sq", V=V, P=M, broadcast=True)), _RE_DIV if mean else 0)


def r2_score_options(y_true, y_pred, sample_weight, multioutput, force_finite, numeric_mode):
    """r2_score with multioutput, 2-D targets or force_finite=False
    (lane/metrics): `sum w (y - p)^2` over `sum w (y - avg)^2` per output;
    the 1-D default call keeps its original kernel and bits."""
    return _NATIVE(None, _OP.family_metric_r2_options,
                   (y_true, y_pred, sample_weight, multioutput, force_finite, numeric_mode), _NATIVE_CONTEXT)


def _r2_parts(r):
    """Stage SS_res and SS_tot per output (binary64 widenings of the Float32
    sums; the y mean rounded to Float32 in the program)."""
    y_mean = r.fin(r.sums(r.Y), f32=True)[1]
    return _centered_sse(r), _centered(r, r.Y, y_mean, mean=False)


def _centered_sse(r):
    return r.fin(r.sums(r.term("sq")), 0)


def _r2_sums_of(r, numeric_mode):
    """(SS_res, SS_tot) of a single-output `_Reg` from one program
    (linear_model._r2_sums)."""
    r.program()
    num, den = _r2_parts(r)
    words = r.prog.want(num, 2), r.prog.want(den, 2)
    _run(r.prog, numeric_mode)
    return float(r.prog.words(words[0], 2, "d")[0]), float(r.prog.words(words[1], 2, "d")[0])


def mean_tweedie_deviance(y_true, y_pred, *, sample_weight=None, power=0, numeric_mode=None):
    """scikit-learn 1.9 `mean_tweedie_deviance` (single output; the power is
    rounded to Float32; pow/log are the portable row-12 functions)."""
    return _NATIVE(None, _OP.family_metric_tweedie, (y_true, y_pred, sample_weight, numeric_mode, power),
                   _NATIVE_CONTEXT)


def mean_poisson_deviance(y_true, y_pred, *, sample_weight=None, numeric_mode=None):
    """scikit-learn 1.9 `mean_poisson_deviance` (Tweedie power 1)."""
    return _NATIVE(None, _OP.family_metric_poisson, (y_true, y_pred, sample_weight, numeric_mode), _NATIVE_CONTEXT)


def mean_gamma_deviance(y_true, y_pred, *, sample_weight=None, numeric_mode=None):
    """scikit-learn 1.9 `mean_gamma_deviance` (Tweedie power 2)."""
    return _NATIVE(None, _OP.family_metric_gamma, (y_true, y_pred, sample_weight, numeric_mode), _NATIVE_CONTEXT)


def d2_tweedie_score(y_true, y_pred, *, sample_weight=None, power=0, numeric_mode=None):
    """scikit-learn 1.9 `d2_tweedie_score`: `1 - dev(y, p) / dev(y, avg y)`
    in one program; a zero null deviance answers NaN (0/0) or -inf as
    numpy's float division does."""
    return _NATIVE(None, _OP.family_metric_d2_tweedie, (y_true, y_pred, sample_weight, numeric_mode, power),
                   _NATIVE_CONTEXT)


def d2_pinball_score(y_true, y_pred, *, sample_weight=None, alpha=0.5, multioutput="uniform_average",
                     numeric_mode=None):
    """scikit-learn 1.9 `d2_pinball_score`: the pinball loss against the
    (weighted, averaged) alpha-quantile of y_true per output, one program."""
    return _NATIVE(None, _OP.family_metric_d2_pinball, (y_true, y_pred, sample_weight, multioutput, numeric_mode, alpha),
                   _NATIVE_CONTEXT)


def d2_absolute_error_score(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average",
                            numeric_mode=None):
    """scikit-learn 1.9 `d2_absolute_error_score` (d2_pinball_score at alpha 0.5)."""
    return _NATIVE(None, _OP.family_metric_d2_absolute, (y_true, y_pred, sample_weight, multioutput, numeric_mode),
                   _NATIVE_CONTEXT)


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
    return true, kind, sorted(_label_set(true))  # glue: sorts the distinct class labels


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


#: words per problem of the curve fold (x_metrics/par.mojo CF_OUT) and its
#: points per chunk (x_metrics/plan.mojo CF_CHUNK)
_CF_OUT = 10
_CF_CHUNK = 1024


#: x_metrics/curve_out.mojo kinds and statuses
_CO_ROC, _CO_PR, _CO_DET = 0, 1, 2
_CO_EMPTY, _CO_ZERO_CLASS = 1, 2


# Engineering notes (moved from the docstring):
# scikit-learn 1.9 `roc_curve` for binary targets: fpr, tpr, thresholds
# (Float64; the first threshold is +inf). The sort, the cumulative
# counts, the collinear drop and the rates run on the device (lane
# cpu4-python: x_metrics/curve_out.mojo; an empty class's NaN rates
# there too).
def roc_curve(y_true, y_score, *, pos_label=None, sample_weight=None, drop_intermediate=True,
              numeric_mode=None):
    """Receiver operating characteristic curve for binary targets.

    Follows scikit-learn's `roc_curve`. Returns ``fpr``, ``tpr`` and ``thresholds`` as
    float64 arrays; the first threshold is ``+inf``. Computed on the GPU.
    """
    return _NATIVE(None, _OP.family_mexp_curve_roc,
                   (y_true, y_score, pos_label, sample_weight, drop_intermediate, numeric_mode),
                   _NATIVE_CONTEXT)


# Engineering notes (moved from the docstring):
# scikit-learn 1.9 `det_curve`: fpr, fnr, thresholds, formed on the
# device (lane cpu4-python: x_metrics/curve_out.mojo); a zero class
# weight raises as the Python division by zero did.
def det_curve(y_true, y_score, *, pos_label=None, sample_weight=None, drop_intermediate=False,
              numeric_mode=None):
    """Detection error tradeoff curve for binary targets.

    Follows scikit-learn's `det_curve`. Returns ``fpr``, ``fnr`` and ``thresholds``,
    computed on the GPU. A zero class weight raises ``ZeroDivisionError``.
    """
    return _NATIVE(None, _OP.family_mexp_curve_det,
                   (y_true, y_score, pos_label, sample_weight, drop_intermediate, numeric_mode),
                   _NATIVE_CONTEXT)


# Engineering notes (moved from the docstring):
# scikit-learn 1.9 `auc`: the trapezoid rule over a monotonic x, on
# the device (lane cpu4-python: x_metrics/curve_out.mojo auc_xy; the host
# column runs the same units): binary64 terms (x[i] - x[i-1]) * (y[i] +
# y[i-1]) / 2, summed as a float-float in fixed chunks (the host fsum
# left the route); a non-finite term's IEEE answer. A list or an integer
# buffer is converted to Float64 first.
def auc(x, y):
    """Area under a curve by the trapezoid rule.

    Follows scikit-learn's `auc`. `x` must be monotonic. Lists and integer arrays are
    converted to float64 first. Computed on the GPU.
    """
    return _NATIVE(None, _OP.family_mexp_curve_auc, (x, y), _NATIVE_CONTEXT)


#: rank_epi kinds of the ranking averages (x_metrics/rank_epi.mojo, lane cpu2-l7-metrics S3b)
_RANK_ROW_SUM, _RANK_CURVE_SCORE, _RANK_AVERAGE, _RANK_OVO_MASK, _RANK_OVO_PAIR = 5, 6, 7, 8, 9
_ROC_ONE_CLASS = "Only one class is present in y_true. ROC AUC score is not defined in that case."


_LITTLE = array.array("i", [1]).tobytes()[0] == 1


#: arena words one one-vs-one program may hold (its pairs' curve slots);
#: more pairs go to further programs
_OVO_WORDS = 1 << 26
_ROWS_MSG = ("Target scores need to be probabilities for multiclass roc_auc, i.e. they "
             "should sum up to 1.0 over classes")


def average_precision_score(y_true, y_score, *, average="macro", pos_label=1, sample_weight=None,
                            numeric_mode=None):
    """scikit-learn 1.9 `average_precision_score` (binary and multiclass
    one-vs-rest; multilabel-indicator targets are refused by name)."""
    return _NATIVE(None, _OP.family_mexp_curve_average_precision,
                   (y_true, y_score, pos_label, sample_weight, numeric_mode, average),
                   _NATIVE_CONTEXT)


def top_k_accuracy_score(y_true, y_score, *, k=2, normalize=True, sample_weight=None, labels=None,
                         numeric_mode=None):
    """scikit-learn 1.9 `top_k_accuracy_score`: a hit when the true class is
    among the k largest scores, ties ordered as numpy's stable argsort
    reversed (the larger class index first)."""
    return _NATIVE(None, _OP.family_mexp_curve_top_k,
                   (y_true, y_score, k, normalize, sample_weight, labels, numeric_mode),
                   _NATIVE_CONTEXT)


#: rank_epi kinds (x_metrics/rank_epi.mojo)
_RANK_MEAN, _RANK_NDCG_ROW, _RANK_D2_LOG, _RANK_D2_BRIER, _RANK_DCG_DISC = 0, 1, 2, 3, 4


def _proba(y_true, y_proba, labels, pos_label, caller):
    """(codes, a, k, binary): scikit-learn 1.9's validation of the shapes and
    labels of probabilistic predictions for a binary vector (`a` (n,)) or a
    multiclass matrix (`a` (n, k)); stage `_put_proba(prog, a, n, k,
    binary)` for the values' check and the (n, 2 or k) matrix."""
    from ._metrics_impl import _label_map, _selected_labels
    from ._buffer import materialize_f32_lists
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
        classes = sorted(classes)  # glue: sorts the distinct class labels
        k = a.shape[1]
        if len(classes) != k:
            raise ValueError(f"y_true and y_proba contain different number of classes: {len(classes)} vs {k}")
        index = {c: i for i, c in enumerate(classes)}  # glue: class index table
        codes = _label_map(true, lambda v: index[v])
    if a.shape[0] != n:
        raise ValueError("y_true and y_proba have different numbers of rows")
    a = as_f32_c(a, ndim=a.ndim, name="y_proba")[0]
    # the values' check (finite, in [0, 1], rows summing to one) and the
    # binary [1 - p, p] layout run in the caller's program (`_put_proba`)
    return codes, a, k, binary


def brier_score_loss(y_true, y_proba, *, sample_weight=None, pos_label=None, labels=None,
                     scale_by_half="auto", numeric_mode=None):
    """scikit-learn 1.9 `brier_score_loss` (binary vector or multiclass
    matrix): the mean of `sum_c (onehot - p)^2`, halved by default for the
    binary case."""
    return _NATIVE(None, _OP.family_mexp_rank_brier,
                   (y_true, y_proba, sample_weight, pos_label, labels, scale_by_half, numeric_mode),
                   _NATIVE_CONTEXT)


def d2_log_loss_score(y_true, y_proba=None, *, sample_weight=None, labels=None, numeric_mode=None):
    """scikit-learn 1.9 `d2_log_loss_score`: one minus the log loss over the
    log loss of the (weighted) class frequencies."""
    return _NATIVE(None, _OP.family_mexp_rank_d2_log_loss,
                   (y_true, y_proba, sample_weight, labels, None, numeric_mode),
                   _NATIVE_CONTEXT)


def d2_brier_score(y_true, y_proba, *, sample_weight=None, pos_label=None, labels=None, numeric_mode=None):
    """scikit-learn 1.9 `d2_brier_score`: one minus the Brier score over the
    Brier score of the (weighted) class frequencies."""
    return _NATIVE(None, _OP.family_mexp_rank_d2_brier,
                   (y_true, y_proba, sample_weight, labels, pos_label, numeric_mode),
                   _NATIVE_CONTEXT)


def hinge_loss(y_true, pred_decision, *, labels=None, sample_weight=None, numeric_mode=None):
    """scikit-learn 1.9 `hinge_loss`: binary (labels mapped to -1 / +1, the
    larger class positive) or Crammer-Singer multiclass margins."""
    return _NATIVE(None, _OP.family_mexp_rank_hinge,
                   (y_true, pred_decision, labels, sample_weight, numeric_mode),
                   _NATIVE_CONTEXT)


def dcg_score(y_true, y_score, *, k=None, log_base=2, sample_weight=None, ignore_ties=False,
              numeric_mode=None):
    """scikit-learn 1.9 `dcg_score`: tie-averaged gains by default;
    ignore_ties=True ranks tied scores by DESCENDING column index (numpy's
    `argsort(...)[::-1]` there, whose tie order its unstable sort leaves
    unspecified; this one is fixed)."""
    return _NATIVE(None, _OP.family_mexp_rank_dcg,
                   (y_true, y_score, k, log_base, sample_weight, ignore_ties, numeric_mode),
                   _NATIVE_CONTEXT)


# Engineering notes (moved from the docstring):
# scikit-learn 1.9 `ndcg_score`: each row's DCG over its ideal DCG
# (0 when the row has no relevant item), (weighted) averaged. The gains,
# the per-row ratios (`rank_epi` NDCG_ROW: the binary64 quotient narrowed
# to Float32), their (weighted) PairSum and the mean run in one device
# program (lane cpu2-l7-metrics: no host epilogue over the rows).
def ndcg_score(y_true, y_score, *, k=None, sample_weight=None, ignore_ties=False, numeric_mode=None):
    """Normalized discounted cumulative gain.

    Follows scikit-learn's `ndcg_score`: each row's DCG divided by its ideal DCG (0 when
    the row has no relevant item), averaged over rows, weighted when `sample_weight` is
    given. Computed on the GPU.
    """
    return _NATIVE(None, _OP.family_mexp_rank_ndcg,
                   (y_true, y_score, k, sample_weight, ignore_ties, numeric_mode),
                   _NATIVE_CONTEXT)


def coverage_error(y_true, y_score, *, sample_weight=None, numeric_mode=None):
    """scikit-learn 1.9 `coverage_error` (multilabel indicator y_true)."""
    return _NATIVE(None, _OP.family_mexp_rank_coverage, (y_true, y_score, sample_weight, numeric_mode), _NATIVE_CONTEXT)


def label_ranking_average_precision_score(y_true, y_score, *, sample_weight=None, numeric_mode=None):
    """scikit-learn 1.9 `label_ranking_average_precision_score`."""
    return _NATIVE(None, _OP.family_mexp_rank_lrap, (y_true, y_score, sample_weight, numeric_mode), _NATIVE_CONTEXT)


def label_ranking_loss(y_true, y_score, *, sample_weight=None, numeric_mode=None):
    """scikit-learn 1.9 `label_ranking_loss` (tied scores count as misordered)."""
    return _NATIVE(None, _OP.family_mexp_rank_loss, (y_true, y_score, sample_weight, numeric_mode), _NATIVE_CONTEXT)


# ---------------------------------------------------------------------------
# Clustering (scikit-learn 1.9 sklearn/metrics/cluster/)
# ---------------------------------------------------------------------------


def contingency_matrix(labels_true, labels_pred, *, eps=None, sparse=False, dtype="int64", numeric_mode=None):
    """scikit-learn 1.9 `contingency_matrix` (dense): exact counts from the
    device grouping, rows the sorted classes, columns the sorted clusters;
    `eps` adds a constant and makes the result Float64. The result is always
    dense; `sparse=True` raises `NotImplementedError`."""
    return _NATIVE(None, _OP.family_mexp_cluster_contingency,
                   (labels_true, labels_pred, eps, sparse, dtype, numeric_mode),
                   _NATIVE_CONTEXT)


def pair_confusion_matrix(labels_true, labels_pred, *, numeric_mode=None):
    """scikit-learn 1.9 `pair_confusion_matrix`: the 2 x 2 Int64 pair counts
    from the exact contingency matrix (exact Int64 in the binding: every
    pair count is below 2^62)."""
    return _NATIVE(None, _OP.family_mexp_cluster_pair_confusion,
                   (labels_true, labels_pred, numeric_mode),
                   _NATIVE_CONTEXT)


# Engineering notes (moved from the docstring):
# scikit-learn 1.9 `normalized_mutual_info_score`, in nats as scikit-learn
# (exact device contingency counts; the logarithms are the portable binary64
# ones of `_portable_math`, DEVIATION 6106).
def normalized_mutual_info_score(labels_true, labels_pred, *, average_method="arithmetic", numeric_mode=None):
    """Normalized mutual information between two clusterings.

    Follows scikit-learn's `normalized_mutual_info_score`, in nats.
    """
    return _NATIVE(None, _OP.family_mexp_cluster_nmi,
                   (labels_true, labels_pred, average_method, numeric_mode),
                   _NATIVE_CONTEXT)


def adjusted_mutual_info_score(labels_true, labels_pred, *, average_method="arithmetic", numeric_mode=None):
    """scikit-learn 1.9 `adjusted_mutual_info_score`: (MI - E[MI]) /
    (mean(H) - E[MI]) with scikit-learn's epsilon guards."""
    return _NATIVE(None, _OP.family_mexp_cluster_ami,
                   (labels_true, labels_pred, average_method, numeric_mode),
                   _NATIVE_CONTEXT)


def calinski_harabasz_score(X, labels, *, numeric_mode=None):
    """scikit-learn 1.9 `calinski_harabasz_score`: the between- over the
    within-cluster dispersion, scaled by (n - k) / (k - 1), in one device
    program."""
    return _NATIVE(None, _OP.family_mexp_cluster_calinski_harabasz, (X, labels, numeric_mode), _NATIVE_CONTEXT)


def davies_bouldin_score(X, labels, *, numeric_mode=None):
    """scikit-learn 1.9 `davies_bouldin_score`: the mean over clusters of the
    worst (s_i + s_j) / d(c_i, c_j), in one device program."""
    return _NATIVE(None, _OP.family_mexp_cluster_davies_bouldin, (X, labels, numeric_mode), _NATIVE_CONTEXT)


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
    _run(prog, numeric_mode)
    sizes = prog.ints(sz, k)
    res = []
    for f in range(k):  # glue: wraps each fold's result views
        c = sizes[f]
        base = out + 2 * n * f
        test = prog.words(base, 2 * c, "q")
        train = prog.words(base + 2 * c, 2 * (n - c), "q")
        res.append((Array._owned(train, (n - c,), "<i8", "C"), Array._owned(test, (c,), "<i8", "C")))
    return res


def regression_report(y_true, y_pred, *, sample_weight=None,
                      multioutput="uniform_average", force_finite=True,
                      numeric_mode="identical"):
    """Return mean squared error, mean absolute error and R² together."""
    return _NATIVE(None, _OP.family_metric_regression_report,
                   (y_true, y_pred, sample_weight, multioutput, force_finite, numeric_mode),
                   _NATIVE_CONTEXT)


def ranking_report(y_true, y_score, *, pos_label=None, sample_weight=None,
                   include_curves=False, numeric_mode="identical"):
    """Return binary ROC AUC, average precision and optional curves together."""
    return _NATIVE(None, _OP.family_mexp_curve_ranking_report,
                   (y_true, y_score, pos_label, sample_weight, include_curves, numeric_mode),
                   _NATIVE_CONTEXT)


# Import-time connection; `_Prog.run` hands each program off once to Mojo.
from ._native_arena_prog import connect as _connect_arena_prog
_PROG_CALL, _PROG_CONTEXT = _connect_arena_prog("x_metrics", _Prog)

# The compiled family owner (bindings/metrics_reg_native.mojo, reached from
# the classical door): the regression metrics above are one native call each.
# It builds the x_metrics program `_Reg` builds, with the fixed op and term
# codes below, and runs it through the shared program runner connected above.
from . import _native_family_bridge  # noqa: E402
_NATIVE, _NATIVE_CONTEXT = _native_family_bridge.connect()
# Lane ne1-h-labels: operations pass as integer codes from the one native
# table (bindings/native_operation_codes.mojo), read once at import.
from ._native_ops import codes as _operation_codes  # noqa: E402
_OP = _operation_codes()
_native_family_bridge.tables(family_metrics_ops=dict(_OPS), family_metrics_terms=dict(_TERM),
                             family_metrics_prog_context=_PROG_CONTEXT,
                             family_metrics_undefined=UndefinedMetricWarning, family_metrics_real=numbers.Real,
                             family_mexp_integral=numbers.Integral)

from ._native_owner_lifetime import connect_owner_context as _connect_metric_owners
_connect_metric_owners(_NATIVE_CONTEXT)
