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
            approx_mode=81)
_PARAMS = 14
_NONE = -1


def _sq(v):
    """v * v: one correctly rounded product. `v ** 2` calls the platform pow,
    which DEVIATION 6106 does not allow (it is not correctly rounded on every
    host). Lanes py-bugs and py-misc-metrics both made this change."""
    return v * v


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


class _OneHot:
    """0/1 flags the program forms itself from int32 `codes` (the onehot
    unit): layout 0 class-major (word c*n + r), layout 1 row-major (word
    r*k + c). Passed where `_curves` takes flags."""

    __slots__ = ("codes", "n", "k", "layout")

    def __init__(self, codes, n, k, layout):
        self.codes, self.n, self.k, self.layout = _i32_c(codes), int(n), int(k), int(layout)


class _RepRows:
    """Float32 weights `w` (n) each repeated over k words (the rep_rows
    unit). Passed where `_curves` takes weights."""

    __slots__ = ("w", "n", "k")

    def __init__(self, w, n, k):
        self.w, self.n, self.k = w, int(n), int(k)


def _put_flags(prog, flags):
    """The flags' arena offset: an `_OneHot` is formed by a stage on the
    device (or the host binding), anything else is copied in."""
    if isinstance(flags, _OneHot):
        C = prog.put_i32(flags.codes)
        out = prog.scratch(flags.n * flags.k)
        prog.stage("onehot", flags.n * flags.k, C, out, flags.n, flags.k, flags.layout)
        return out
    return prog.put_i32(flags)


def _put_weights(prog, w):
    """The weights' arena offset (_NONE for none): a `_RepRows` is repeated
    by a stage, anything else is copied in."""
    if w is None:
        return _NONE
    if isinstance(w, _RepRows):
        W = prog.put(w.w)
        out = prog.scratch(w.n * w.k)
        prog.stage("rep_rows", w.n * w.k, W, out, w.k)
        return out
    return prog.put(w)


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


def _flag_set(prog, flag):
    return prog.ints(flag, 1)[0] != 0


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
    return true, pred, kind, sorted(_label_set(true) | _label_set(pred)), w  # glue: sorts the distinct class labels


def _codes(labels_seq, order):
    from ._metrics_impl import _label_map
    index = {c: i for i, c in enumerate(order)}  # glue: class index table
    return _label_map(labels_seq, lambda v: index.get(v, -1))


def _epi_codes(average, zero_division):
    """(AVG, ZD) of x_metrics/cls_epi.mojo: 0 per label, 1 micro, 3 weighted,
    2 the plain mean (macro, binary); 0 / 1 / 2 = zero_division 0.0, 1.0, NaN."""
    zv = _zero_division_value(zero_division)
    return {None: 0, "micro": 1, "weighted": 3}.get(average, 2), (2 if zv != zv else int(zv))


class _Sums:
    """Per-label tp / pred / true sums and the total, over `order` (every
    label that can occur; the caller selects a prefix). Unweighted sums are
    exact integers (the group sizes, differenced on the device: `off_diff`);
    weighted ones are the Float32 PairSum of each group's weights, read as
    Python floats. `epi` (lane fam2-prep-metrics; kind, k, AVG, ZD, FMODE,
    beta^2) stages the set-wise epilogue in the same program (`cls_epi`,
    x_metrics/cls_epi.mojo: the ratios, micro sums and averages as binary64
    on the device, both numeric modes since lane cpu2-l7-metrics): `self.epi`
    = (flags, kept, scalars, per-label values), else None (a list of
    `epi` tuples stages one epilogue each and `self.epi` is the list of
    results). `self.prog` and
    `self.groups` (the three (OFF, OUT) groupings, then the total's) let a
    caller that staged its own tail (`tail`: a function of (prog, groups)
    that stages it and returns what to keep) read it back as `self.tail`."""

    def __init__(self, true, pred, w, order, numeric_mode, epi=None, tail=None):
        n, L = len(true), len(order)
        yt, yp = _codes(true, order), _codes(pred, order)
        prog = _Prog()
        a = prog.put_i32(yt)
        b = prog.put_i32(yp)
        match = prog.scratch(n)
        prog.stage("pair_key", n, a, b, match, L, 1)
        W = _NONE if w is None else prog.put(w)
        groups = [_group(prog, k, n, L, weights=W) for k in (match, a, b)]  # glue: the three count groupings
        if w is not None:
            zero = prog.scratch(n)          # every row in group 0: the total weight
            prog.stage("pair_key", n, a, a, zero, 1, 2)
            groups.append(_group(prog, zero, n, 1, weights=W))
        cnt = [_counts(prog, off, L) for off, _ in groups] if w is None else None  # glue: the three count slots
        eos = []
        for kind, k, avg, zd, fmode, b2 in ([epi] if isinstance(epi, tuple) else (epi or [])):  # glue: one cls_epi stage per requested average
            col = 0 if w is None else 1
            bo = 0
            if fmode == 2:
                bw = array.array("i")
                bw.frombytes(array.array("d", [1 + b2, b2]).tobytes())
                bo = prog.put_i32(list(bw))
            eo = prog.alloc(8 + 6 * max(L, 1))
            prog.stage("cls_epi", 1, groups[0][col], groups[1][col], groups[2][col], k, col, kind, avg, zd, eo,
                       fmode, bo)
            eos.append(eo)
        self.tail = None
        prog.rows = None if w is None else (match, W, n)
        keep = tail(prog, groups) if tail is not None else None
        _run(prog, numeric_mode)
        self.prog, self.groups = prog, groups
        res = []
        for eo in eos:  # glue: reads each staged epilogue's flags, scalars and values
            head = prog.ints(eo, 2)
            res.append((head[0], head[1], prog.words(eo + 2, 6, "d"), prog.words(eo + 8, 6 * max(L, 1), "d")))
        self.epi = (res[0] if res else None) if (epi is None or isinstance(epi, tuple)) else res
        if w is None:
            self.tp, self.true, self.pred = (prog.ints(c, L) for c in cnt)  # glue: reads the three count slots
            self.total = n
        else:
            self.tp, self.true, self.pred = (prog.floats(out, L) for _, out in groups[:3])  # glue: reads the weighted sums
            self.total = prog.floats(groups[3][1], 1)[0]
        self.weighted = w is not None
        if tail is not None:
            self.tail = keep


class UndefinedMetricWarning(UserWarning):
    """scikit-learn's `UndefinedMetricWarning` (a UserWarning): a metric that
    is ill-defined for the input and was set to a fallback value. Defined
    here so the metrics do not import scikit-learn."""


def _undefined_category():
    """The warning class to raise: scikit-learn's own when the caller has
    already imported `sklearn.exceptions` (so its filters and `pytest.warns`
    match), else `UndefinedMetricWarning` above. Never imports scikit-learn."""
    import sys
    sk = sys.modules.get("sklearn.exceptions")
    return getattr(sk, "UndefinedMetricWarning", UndefinedMetricWarning)


def _undefined_warning(msg):
    warnings.warn(msg, _undefined_category(), stacklevel=3)


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
    return list(labels) + [v for v in present if v not in chosen]  # glue: appends the present class labels


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


def _cm_stage(prog, groups, w, n, k, kind, size):
    """Stage a cm_epi tail over a `_Sums` program's groupings (`tail=`):
    the match / true / pred sums' OFF tables (unweighted) or PairSums
    (weighted), the first k labels; the total is the row count n or the
    total grouping's PairSum. Returns the `size`-word output slot."""
    col = 0 if w is None else 1
    out = prog.alloc(max(size, 1))
    tp, tr, pr = groups[0][col], groups[1][col], groups[2][col]
    tot = n if w is None else groups[3][1]
    if kind == _CM_COUNT and w is not None:
        # lane box-run-2-fix: the weighted COUNT reads soft-f64 [hit, total]
        # (`_wsum_sf64`), not the Float32 PairSums
        tot = _wsum_sf64(prog, k)
    if kind == _CM_MLCM:
        if k:
            prog.stage("cm_epi", k, kind, tp, tr, pr, k, col, tot, out)
    elif kind == _CM_MCC:
        prog.stage("cm_epi", 1, kind, tp, tr, pr, k, col, out)
    else:
        prog.stage("cm_epi", 1, kind, tp, k, col, tot, out)
    return out


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
    return _NATIVE(None, "family_mexp_cls_mlcm",
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
    return _NATIVE(None, "family_mexp_cls_prfs",
                   (y_true, y_pred, beta, labels, pos_label, average, warn_for, sample_weight, zero_division, numeric_mode),
                   _NATIVE_CONTEXT)


def fbeta_score(y_true, y_pred, *, beta, labels=None, pos_label=1, average="binary",
                sample_weight=None, zero_division="warn", numeric_mode=None):
    """scikit-learn 1.9 `fbeta_score` (binary / multiclass 1-D targets); see
    precision_recall_fscore_support."""
    return _NATIVE(None, "family_mexp_cls_fbeta",
                   (y_true, y_pred, beta, labels, pos_label, average, sample_weight, zero_division, numeric_mode),
                   _NATIVE_CONTEXT)


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
    return _NATIVE(None, "family_mexp_cls_jaccard",
                   (y_true, y_pred, labels, pos_label, average, sample_weight, zero_division, numeric_mode),
                   _NATIVE_CONTEXT)


def balanced_accuracy_score(y_true, y_pred, *, sample_weight=None, adjusted=False, numeric_mode=None):
    """scikit-learn 1.9 `balanced_accuracy_score`: the mean per-class recall
    over the classes present in y_true (a class with no true weight is
    dropped with sklearn's warning), chance-adjusted when `adjusted`."""
    return _NATIVE(None, "family_mexp_cls_balanced_accuracy",
                   (y_true, y_pred, sample_weight, adjusted, numeric_mode),
                   _NATIVE_CONTEXT)


def matthews_corrcoef(y_true, y_pred, *, sample_weight=None, numeric_mode=None):
    """scikit-learn 1.9 `matthews_corrcoef` (binary and multiclass): the
    covariance form over the per-class true / predicted sums and the trace."""
    return _NATIVE(None, "family_mexp_cls_matthews", (y_true, y_pred, sample_weight, numeric_mode), _NATIVE_CONTEXT)


def _confusion(true, pred, w, order, numeric_mode, tail):
    """The k x k (weighted) confusion matrix over `order`, rows true, made on
    the device and finished by `tail(prog, cells, col)` in the same program
    (a cm_epi stage, x_metrics/cm_epi.mojo): `cells` is the k*k groups' OFF
    table (col 0: exact counts, read by the unit as group sizes) or their
    Float32 PairSums (col 1). Returns (prog, what the tail returned)."""
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
    keep = tail(prog, off if w is None else out, 0 if w is None else 1)
    _run(prog, numeric_mode)
    return prog, keep


def confusion_matrix_weighted(y_true, y_pred, labels, sample_weight, normalize, numeric_mode):
    """confusion_matrix with sample_weight (scikit-learn 1.9): Float64 cells,
    each the Float32 PairSum of its rows' weights; normalize in binary64."""
    from ._metrics_impl import _selected_labels, _label_set
    true, pred, kind, present, w = _pair(y_true, y_pred, sample_weight, "confusion_matrix")
    chosen = _selected_labels(labels, kind, present)
    if not set(chosen).intersection(_label_set(true)):
        raise ValueError("At least one label specified must be in y_true")
    k = len(chosen)
    prog, out = _confusion(true, pred, w, chosen, numeric_mode,
                           lambda prog, cells, col: _normalize(prog, cells, col, k, normalize))
    return Array._owned(prog.words(out, 2 * k * k, "d"), (k, k), "<f8")


def _normalize(prog, cells, col, k, normalize):
    """Stage the normalization of the k x k cells (x_metrics/cm_epi.mojo,
    NORM: one unit per row ('true', 'all', none) or column ('pred'), each sum
    binary64 from 0.0 in ascending index, a zero sum giving 0.0 cells; 'all'
    first sums every cell in one unit). Returns the binary64 output slot."""
    out = prog.want(prog.alloc(2 * k * k), 2 * k * k)
    ts = prog.scratch(2)
    if normalize == "all":
        prog.stage("cm_epi", 1, _CM_NORM, cells, k, col, 0, out, ts)
    if k:
        prog.stage("cm_epi", k, _CM_NORM, cells, k, col, _NORM_MODES[normalize], out, ts)
    return out


def cohen_kappa_score(y1, y2, *, labels=None, weights=None, sample_weight=None,
                      replace_undefined_by=float("nan"), numeric_mode=None):
    """scikit-learn 1.9 `cohen_kappa_score`: the (weighted) confusion matrix
    over `labels` on the device, the kappa in the binary64 epilogue;
    weights None, 'linear' or 'quadratic'."""
    return _NATIVE(None, "family_mexp_cls_kappa",
                   (y1, y2, labels, weights, sample_weight, replace_undefined_by, numeric_mode),
                   _NATIVE_CONTEXT)


def hamming_loss(y_true, y_pred, *, sample_weight=None, numeric_mode=None):
    """scikit-learn 1.9 `hamming_loss` for binary / multiclass 1-D targets:
    the (weighted) fraction of mismatched labels."""
    return _NATIVE(None, "family_mexp_cls_hamming", (y_true, y_pred, sample_weight, numeric_mode), _NATIVE_CONTEXT)


def zero_one_loss(y_true, y_pred, *, normalize=True, sample_weight=None, numeric_mode=None):
    """scikit-learn 1.9 `zero_one_loss` for binary / multiclass 1-D targets."""
    return _NATIVE(None, "family_mexp_cls_zero_one",
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
    return _NATIVE(None, "family_mcls_accuracy_fraction", (y_true, y_pred, sample_weight, numeric_mode),
                   _NATIVE_CONTEXT)


def class_likelihood_ratios(y_true, y_pred, *, labels=None, sample_weight=None,
                            replace_undefined_by=float("nan"), numeric_mode=None):
    """scikit-learn 1.9 `class_likelihood_ratios` (binary targets): LR+ and
    LR- from the 2 x 2 (weighted) confusion matrix; `replace_undefined_by`
    np.nan, 1.0 or {'LR+': v, 'LR-': v}."""
    return _NATIVE(None, "family_mexp_cls_likelihood_ratios",
                   (y_true, y_pred, labels, sample_weight, replace_undefined_by, numeric_mode),
                   _NATIVE_CONTEXT)


def classification_report(y_true, y_pred, *, labels=None, target_names=None, sample_weight=None,
                          digits=2, output_dict=False, zero_division="warn", numeric_mode=None):
    """scikit-learn 1.9 `classification_report` for binary / multiclass 1-D
    targets, from precision_recall_fscore_support."""
    return _NATIVE(None, "family_mexp_cls_report",
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
    return _NATIVE(None, "family_metric_regression_error",
                   (name, y_true, y_pred, sample_weight, multioutput, numeric_mode), _NATIVE_CONTEXT)


def mean_squared_log_error(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average",
                           numeric_mode=None):
    """scikit-learn 1.9 `mean_squared_log_error`: the mean of
    `(log1p(y) - log1p(p))^2` (portable log1p, IDENTITY_PATHS row 51);
    values <= -1 are refused as scikit-learn refuses them."""
    return _NATIVE(None, "family_metric_msle", (y_true, y_pred, sample_weight, multioutput, numeric_mode),
                   _NATIVE_CONTEXT)


def root_mean_squared_log_error(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average",
                                numeric_mode=None):
    """scikit-learn 1.9 `root_mean_squared_log_error` (per-output root, then averaged)."""
    return _NATIVE(None, "family_metric_rmsle", (y_true, y_pred, sample_weight, multioutput, numeric_mode),
                   _NATIVE_CONTEXT)


def mean_absolute_percentage_error(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average",
                                   numeric_mode=None):
    """scikit-learn 1.9 `mean_absolute_percentage_error`: `|p - y| / max(|y|,
    eps)` with numpy's float64 epsilon, averaged (not multiplied by 100)."""
    return _NATIVE(None, "family_metric_mape", (y_true, y_pred, sample_weight, multioutput, numeric_mode),
                   _NATIVE_CONTEXT)


def mean_pinball_loss(y_true, y_pred, *, sample_weight=None, alpha=0.5, multioutput="uniform_average",
                      numeric_mode=None):
    """scikit-learn 1.9 `mean_pinball_loss` (alpha is rounded to Float32)."""
    return _NATIVE(None, "family_metric_pinball", (y_true, y_pred, sample_weight, multioutput, numeric_mode, alpha),
                   _NATIVE_CONTEXT)


def median_absolute_error(y_true, y_pred, *, multioutput="uniform_average", sample_weight=None,
                          numeric_mode=None):
    """scikit-learn 1.9 `median_absolute_error`: the per-output median of
    ``|y - p|`` (the weighted percentile at 50 with averaging when weighted, the
    same answer as numpy's median when not), from a stable device sort."""
    return _NATIVE(None, "family_metric_medae", (y_true, y_pred, sample_weight, multioutput, numeric_mode),
                   _NATIVE_CONTEXT)


def max_error(y_true, y_pred, *, numeric_mode=None):
    """scikit-learn 1.9 `max_error` (single output)."""
    return _NATIVE(None, "family_metric_max_error", (y_true, y_pred, numeric_mode), _NATIVE_CONTEXT)


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
    return _NATIVE(None, "family_metric_explained_variance",
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
    return _NATIVE(None, "family_metric_r2_options",
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
    return _NATIVE(None, "family_metric_tweedie", (y_true, y_pred, sample_weight, numeric_mode, power),
                   _NATIVE_CONTEXT)


def mean_poisson_deviance(y_true, y_pred, *, sample_weight=None, numeric_mode=None):
    """scikit-learn 1.9 `mean_poisson_deviance` (Tweedie power 1)."""
    return _NATIVE(None, "family_metric_poisson", (y_true, y_pred, sample_weight, numeric_mode), _NATIVE_CONTEXT)


def mean_gamma_deviance(y_true, y_pred, *, sample_weight=None, numeric_mode=None):
    """scikit-learn 1.9 `mean_gamma_deviance` (Tweedie power 2)."""
    return _NATIVE(None, "family_metric_gamma", (y_true, y_pred, sample_weight, numeric_mode), _NATIVE_CONTEXT)


def d2_tweedie_score(y_true, y_pred, *, sample_weight=None, power=0, numeric_mode=None):
    """scikit-learn 1.9 `d2_tweedie_score`: `1 - dev(y, p) / dev(y, avg y)`
    in one program; a zero null deviance answers NaN (0/0) or -inf as
    numpy's float division does."""
    return _NATIVE(None, "family_metric_d2_tweedie", (y_true, y_pred, sample_weight, numeric_mode, power),
                   _NATIVE_CONTEXT)


def d2_pinball_score(y_true, y_pred, *, sample_weight=None, alpha=0.5, multioutput="uniform_average",
                     numeric_mode=None):
    """scikit-learn 1.9 `d2_pinball_score`: the pinball loss against the
    (weighted, averaged) alpha-quantile of y_true per output, one program."""
    return _NATIVE(None, "family_metric_d2_pinball", (y_true, y_pred, sample_weight, multioutput, numeric_mode, alpha),
                   _NATIVE_CONTEXT)


def d2_absolute_error_score(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average",
                            numeric_mode=None):
    """scikit-learn 1.9 `d2_absolute_error_score` (d2_pinball_score at alpha 0.5)."""
    return _NATIVE(None, "family_metric_d2_absolute", (y_true, y_pred, sample_weight, multioutput, numeric_mode),
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

    def lists(self):
        p, c = self.prog, self.c
        cur = _Curve((p.floats(self.fps, c), p.floats(self.tps, c),
                      p.floats(self.thr, c) if self.thr >= 0 else None))
        if self.keep >= 0:
            p._check(self.keep, c)
            lo = 4 * self.keep
            cur.keep = bytes(_arena_view(p.arena).cast("B")[lo + (0 if _LITTLE else 3):lo + 4 * c:4])
        elif self.keep == -2:
            cur.keep = b"\x01" * c       # the device dropped the collinear points already
        return cur


def _f32_weights_addr(w):
    if w is None:
        return 0, None
    if isinstance(w, Array) and w.dtype == "<f4" and w._has_order("C"):
        return addr_ro(w, name="sample_weight"), w
    return None, None


def _auc_of(cur, max_fpr):
    """The ROC AUC of a curve
    when both classes are present, else (or on any doubt) the Python.
    A `_FoldCurve` (the device fold, lane cgr2-metrics-shap) scores itself."""
    if isinstance(cur, _FoldCurve):
        return cur.auc(max_fpr)
    raise TypeError("mojolearn metrics: an AUC is scored from the device fold (`_curves(..., fold='auc')`)")


def _curves(scores, flags, w, n, problems, numeric_mode, *, stride=1, thresholds=True, keep_flags=True,
            lazy=False, compact=False, fold=None, max_fpr=None, tail=None):
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
    metrics-apple2). Weighted curves are flagged and compacted too (lane
    cgr2-metrics-shap: exact two-sum steps on the device).

    fold="auc" (compact) or "ap" (lane cgr2-metrics-shap) adds the device
    curve fold (x_metrics/par.mojo curve_fold_unit) and brings back only its
    CF_OUT words per problem, no curve: the list holds `_FoldCurve`s.
    max_fpr (fold="auc") folds the partial AUC instead. tail (fold only):
    see `_fold_curves`."""
    if fold is not None:
        return _fold_curves(scores, flags, w, n, problems, numeric_mode, stride=stride, fold=fold,
                            max_fpr=max_fpr, tail=tail)
    if tail is not None:
        raise ValueError("mojolearn metrics: a curve tail needs the device fold")
    prog = _Prog()
    S = prog.put(scores)
    POS = _put_flags(prog, flags)
    W = _put_weights(prog, w)
    N = n * problems
    order = prog.scratch(N)
    flagged = bool(keep_flags)
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
        for t in range(problems):  # glue: declares each problem's output ranges
            for b in range(3 if thresholds else 2):  # glue: declares each curve output range
                prog.want(CF + b * N + t * n, n, count=CM + t)
    else:
        if flagged:
            keep = prog.alloc(N)
        for t in range(problems):  # glue: declares each problem's output ranges
            for b in (fps, tps) + ((thr,) if thresholds else ()) + ((keep,) if flagged else ()):  # glue: declares each curve output range
                prog.want(b + t * n, n, count=cnt + t)
    prog.stage("bin_curve", problems, S, stride, POS, W, n, order, fps, tps, thr, cnt,
               keep, 1 if flagged else 0, CF, CM)
    _run(prog, numeric_mode)
    counts = prog.ints(CM if compact else cnt, problems)
    if compact:
        return [_DevCurve(prog, CF + t * n, CF + N + t * n, CF + 2 * N + t * n if thresholds else _NONE,
                          -2, counts[t], numeric_mode) for t in range(problems)]  # glue: one result handle per problem
    # every caller is lazy (lane pyglue-sweep: the curve lists never come
    # back to Python whole; the epilogues read the arena words)
    return [_DevCurve(prog, fps + t * n, tps + t * n, thr + t * n if thresholds else _NONE,
                      keep + t * n if flagged else _NONE, counts[t], numeric_mode)
            for t in range(problems)]  # glue: one result handle per problem


#: words per problem of the curve fold (x_metrics/par.mojo CF_OUT) and its
#: points per chunk (x_metrics/plan.mojo CF_CHUNK)
_CF_OUT = 10
_CF_CHUNK = 1024


def _f32_split(x):
    """(hi, lo) Int32 bit words of the float-float x = hi + lo (hi the
    nearest float32, lo the nearest float32 of the remainder)."""
    import struct
    hi = struct.unpack("<f", struct.pack("<f", float(x)))[0]
    lo = float(x) - hi
    bits = struct.unpack("<ii", struct.pack("<ff", hi, lo))
    return bits[0], bits[1]


def _fold_stages(prog, S, stride, POS, W, n, problems, auc, max_fpr=None, out=None):
    """The curve (bin_curve) and its fold (curve_fold) as stages of `prog`
    over scores at S, flags at POS, weights at W (_NONE: none): the fold's
    CF_OUT words per problem at `out` (allocated and declared when None),
    returned. Shared by `_fold_curves` and the one-vs-one pairs."""
    N = n * problems
    order = prog.scratch(N)
    cnt = prog.scratch(problems)
    fps = prog.scratch(N)
    tps = prog.scratch(N)
    thr = prog.scratch(N)
    keep = _NONE
    CF = CM = 0
    if auc:
        keep = prog.scratch(N)
        CF = prog.scratch(3 * N)
        CM = prog.scratch(problems)
    prog.stage("bin_curve", problems, S, stride, POS, W, n, order, fps, tps, thr, cnt,
               keep, 1 if auc else 0, CF, CM)
    if out is None:
        out = prog.want(prog.alloc(_CF_OUT * problems), _CF_OUT * problems)
    mode, mh, ml = (1, 0, 0) if not auc else ((0, 0, 0) if max_fpr is None else (2,) + _f32_split(max_fpr))
    if auc:
        prog.stage("curve_fold", problems, n, CF, CF + N, CM, out, mode, mh, ml, _CF_CHUNK)
    else:
        prog.stage("curve_fold", problems, n, fps, tps, cnt, out, mode, mh, ml, _CF_CHUNK)
    return out


def _fold_curves(scores, flags, w, n, problems, numeric_mode, *, stride=1, fold="auc", max_fpr=None, tail=None):
    """`_curves` with the device curve fold: the curve stays on the device
    (compacted for the AUCs) and only the fold's words come back.

    tail (lane cpu2-l7-metrics S3b): a function `(prog, S, W, C, out, n,
    problems)` that stages more units in this program before it runs (S the
    scores, W the weights or _NONE, C the int32 codes of an `_OneHot` or
    _NONE, out the fold words); it keeps its own offsets."""
    prog = _Prog()
    S = prog.put(scores)
    C = _NONE
    if tail is not None and isinstance(flags, _OneHot):
        # `_put_flags`'s onehot stage, with the codes' offset kept for the tail
        C = prog.put_i32(flags.codes)
        POS = prog.scratch(flags.n * flags.k)
        prog.stage("onehot", flags.n * flags.k, C, POS, flags.n, flags.k, flags.layout)
    else:
        POS = _put_flags(prog, flags)
    W = _put_weights(prog, w)
    auc = fold == "auc"
    out = _fold_stages(prog, S, stride, POS, W, n, problems, auc, max_fpr)
    if tail is not None:
        tail(prog, S, W, C, out, n, problems)
    _run(prog, numeric_mode)
    holder = {}

    def redo():
        if "c" not in holder:
            holder["c"] = _curves(scores, flags, w, n, problems, numeric_mode, stride=stride, thresholds=False,
                                  keep_flags=auc, lazy=True, compact=auc)
        return holder["c"]

    return [_FoldCurve(prog, out + _CF_OUT * t, max_fpr, redo, t) for t in range(problems)]  # glue: one result handle per problem


class _FoldCurve:
    """A curve's device fold (lane cgr2-metrics-shap): the float-float sum
    and the few curve words x_metrics/par.mojo `_cf_finish` writes. `auc()`
    and `ap()` turn them into the score with a handful of binary64 scalar
    operations; an edge case (an empty class, a zero precision denominator,
    max_fpr past the last point) recomputes the curve for the Python rule."""

    def __init__(self, prog, off, max_fpr, redo, t):
        self.prog, self.off, self.max_fpr, self._redo, self.t = prog, off, max_fpr, redo, t

    def _words(self):
        f = self.prog.floats(self.off, _CF_OUT)
        i = self.prog.ints(self.off, _CF_OUT)
        return f[0] + f[1], f[2], f[3], i[4], f[5], f[6], f[7], f[8], i[9]

    def lists(self):
        return self._redo()[self.t].lists()

    def auc(self, max_fpr):
        s, F, T, k, a, b, u, v, c = self._words()
        if c <= 0 or not (F > 0 and T > 0):
            # the empty-class answer, without bringing the curve back
            _undefined_warning("Only one class is present in y_true. ROC AUC score is not defined in that case.")
            return float("nan")
        if max_fpr is None:
            return float(s / (2.0 * F * T))
        if k >= c:
            # max_fpr at or past the last point (max_fpr = 1: the last fpr is
            # F / F): the whole area, the device fold's word (lane py-runtime
            # round 3; the Python fsum rule over the curve is gone, new bits
            # for max_fpr = 1 only, the same on every column)
            return float(s / (2.0 * F * T))
        x0, x1, y0, y1 = a / F, b / F, u / T, v / T
        yi = y0 if x1 == x0 else y0 + (max_fpr - x0) * (y1 - y0) / (x1 - x0)
        part = s / (2.0 * F * T) + (max_fpr - x0) * (yi + y0) / 2.0
        min_area = 0.5 * max_fpr * max_fpr
        return float(0.5 * (1 + (part - min_area) / (max_fpr - min_area)))

    def ap(self):
        s, F, T, k, a, b, u, v, c = self._words()
        if c <= 0 or T == 0:
            return 0.0
        if k > 0:
            # a point with tps + fps == 0: the Python rule's t / (t + f) is a
            # float division by zero (lane py-runtime round 3: raised from the
            # fold's count word, no curve walk)
            raise ZeroDivisionError("float division by zero")
        return float(max(0.0, s / T))


#: x_metrics/curve_out.mojo kinds and statuses
_CO_ROC, _CO_PR, _CO_DET = 0, 1, 2
_CO_EMPTY, _CO_ZERO_CLASS = 1, 2


def _curve_out(y_true, y_score, pos_label, sample_weight, numeric_mode, caller, kind, drop):
    """One binary curve's three Float64 output arrays, formed on the device
    (lane cpu4-python: x_metrics/curve_out.mojo, the units the host column
    runs too) in the program that sorts and counts: bin_curve (compacted by
    its collinear drop for a dropping ROC), then curve_out. Only the arrays
    (up to their length) and six head words come back. Returns (arrays,
    length, F, T, classes): F, T the last fps and tps."""
    from ._metrics_impl import _label_map
    true, kind_y, classes = _targets(y_true, caller)
    if len(classes) > 2 and pos_label is None:
        raise ValueError("multiclass format is not supported")
    pos = _pos_label(pos_label, kind_y, classes, caller)
    n = len(true)
    s = _scores(y_score, n, caller, ndim=1)
    w = _weights(sample_weight, n, caller)
    flags = _label_map(true, lambda v: int(v == pos))
    if n <= 0:
        raise ValueError("mojolearn metrics: an empty curve has no thresholds")
    prog = _Prog()
    S = prog.put(s)
    POS = _put_flags(prog, flags)
    W = _put_weights(prog, w)
    order = prog.scratch(n)
    cnt = prog.scratch(1)
    fps = prog.scratch(n)
    tps = prog.scratch(n)
    thr = prog.scratch(n)
    roc_drop = kind == _CO_ROC and bool(drop)
    keep = _NONE
    CF = CM = 0
    if roc_drop:
        keep = prog.scratch(n)
        CF = prog.scratch(3 * n)
        CM = prog.scratch(1)
    prog.stage("bin_curve", 1, S, 1, POS, W, n, order, fps, tps, thr, cnt, keep, 1 if roc_drop else 0, CF, CM)
    LEN = prog.want(prog.alloc(6), 6)
    OUT = prog.alloc(6 * (n + 1))
    for b in range(3):  # glue: declares the three output arrays, each up to its length
        prog.want(OUT + 2 * b * (n + 1), 2 * (n + 1), count=LEN, mult=2)
    src = (CF, CF + n, CF + 2 * n, CM) if roc_drop else (fps, tps, thr, cnt)
    prog.stage("curve_out", 1, kind, n, *src, 1 if (drop and kind != _CO_ROC) else 0, OUT, LEN)
    _run(prog, numeric_mode)
    m, status = prog.ints(LEN, 2)
    if status == _CO_EMPTY:
        raise ValueError("mojolearn metrics: an empty curve has no thresholds")
    F, T = prog.words(LEN + 2, 4, "d")
    if status == _CO_ZERO_CLASS:
        return None, 0, F, T, classes
    # precision-recall's thresholds hold one word fewer than its precision and recall
    lens = (m, m, m - 1) if kind == _CO_PR else (m, m, m)
    arrays = tuple(Array._owned(prog.words(OUT + 2 * b * (n + 1), 2 * lens[b], "d"), (lens[b],), "<f8", "C")
                   for b in range(3))  # glue: wraps the three downloaded arrays
    return arrays, m, F, T, classes


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
    return _NATIVE(None, "family_mexp_curve_roc",
                   (y_true, y_score, pos_label, sample_weight, drop_intermediate, numeric_mode),
                   _NATIVE_CONTEXT)


def precision_recall_curve_options(y_true, y_score, pos_label, sample_weight, drop_intermediate, numeric_mode):
    """precision_recall_curve with sample_weight or drop_intermediate=True
    (lane/metrics): Float64 outputs, formed on the device (lane cpu4-python:
    x_metrics/curve_out.mojo; no positives' recall of one there too); the
    default call keeps its kernel."""
    arrays, _, _, T, _ = _curve_out(y_true, y_score, pos_label, sample_weight, numeric_mode,
                                    "precision_recall_curve", _CO_PR, drop_intermediate)
    if T == 0:
        warnings.warn("No positive class found in y_true, recall is set to one for all thresholds.",
                      UserWarning, stacklevel=3)
    return arrays


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
    return _NATIVE(None, "family_mexp_curve_det",
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
    return _NATIVE(None, "family_mexp_curve_auc", (x, y), _NATIVE_CONTEXT)


#: rank_epi kinds of the ranking averages (x_metrics/rank_epi.mojo, lane cpu2-l7-metrics S3b)
_RANK_ROW_SUM, _RANK_CURVE_SCORE, _RANK_AVERAGE, _RANK_OVO_MASK, _RANK_OVO_PAIR = 5, 6, 7, 8, 9
_ROC_ONE_CLASS = "Only one class is present in y_true. ROC AUC score is not defined in that case."


def _f64_bits(prog, vals):
    """Binary64 values as Int32 arena words (low first), an input."""
    bw = array.array("i")
    bw.frombytes(array.array("d", vals).tobytes())
    return prog.put_i32(bw)


def _stage_average(prog, SC, SUP, m, average):
    """`rank_epi` AVERAGE over m binary64 scores at SC (weights at SUP):
    the declared binary64 result's offset."""
    res = prog.want(prog.alloc(2), 2)
    prog.stage("rank_epi", 1, _RANK_AVERAGE, SC, SUP, m, 1 if average == "weighted" else 0, res)
    return res


class _OvrTail:
    """The one-vs-rest curve program's tail (`_fold_curves` tail=): the
    multiclass row-sum check (`rank_epi` ROW_SUM), the per-class supports
    (group sums over the codes: exact counts, or the Float32 PairSums of
    the weights), the per-class scores from the fold words (CURVE_SCORE) and
    their average (AVERAGE), all in the program that folds the curves. The
    host class_sums / row_sum_range walks and the Python averages are gone
    (lane cpu2-l7-metrics S3b)."""

    def __init__(self, fold, check_rows, average, numeric_mode):
        """average: None (per-class scores), "macro", "weighted", or
        "micro" (no per-class work: the row check only)."""
        self.fold, self.check_rows, self.average = fold, check_rows, average
        self.score = average != "micro"
        self.numeric_mode = numeric_mode
        self.prog = None

    def __call__(self, prog, S, W, C, out, n, k):
        self.n, self.k = n, k
        self.prog = prog
        if self.check_rows:
            self.rows = prog.want(prog.alloc(1), 1)
            if n:
                prog.stage("rank_epi", n, _RANK_ROW_SUM, S, n, k, self.rows)
        if not self.score:
            return
        weighted = W != _NONE
        off, sums = _group(prog, C, n, k, weights=W)
        self.sc = prog.want(prog.alloc(2 * k), 2 * k)
        self.fl = prog.want(prog.alloc(k), k)
        self.sup = prog.want(prog.alloc(2 * k), 2 * k)
        prog.stage("rank_epi", k, _RANK_CURVE_SCORE, out, k, 0 if self.fold == "auc" else 1, self.sc, self.fl,
                   sums if weighted else off, 1 if weighted else 0, self.sup)
        self.res = _stage_average(prog, self.sc, self.sup, k, self.average) if self.average is not None else _NONE

    def rows_ok(self):
        return not _flag_set(self.prog, self.rows)

    def result(self, curves):
        """The averaged score (binary64) or, average=None, the per-class
        Float64 array. A class whose AP needs the curve rule (a zero
        precision denominator) takes `_FoldCurve.ap`, and the average then
        runs in a second small program over the device words."""
        prog, k = self.prog, self.k
        fl = prog.ints(self.fl, k)
        if self.fold == "auc":
            for c in range(k):  # glue: one warning per class without both labels
                if fl[c] == 1:
                    _undefined_warning(_ROC_ONE_CLASS)
        edge = [c for c in range(k) if fl[c] == 2]  # glue: the classes whose AP takes the curve rule
        if not edge:
            if self.average is None:
                return Array.from_list(list(prog.words(self.sc, 2 * k, "d")), "<f8")
            return prog.words(self.res, 2, "d")[0]
        sc = list(prog.words(self.sc, 2 * k, "d"))
        for c in edge:  # glue: the curve rule's per-class scores
            sc[c] = curves[c].ap()
        if self.average is None:
            return Array.from_list(sc, "<f8")
        p2 = _Prog()
        SC = _f64_bits(p2, sc)
        SUP = p2.put_i32(prog.words(self.sup, 2 * k, "i"))
        res = _stage_average(p2, SC, SUP, k, self.average)
        _run(p2, self.numeric_mode)
        return p2.words(res, 2, "d")[0]


def _ovr(y_true, y_score, sample_weight, labels, caller, numeric_mode, keep_flags=True, fold=None, tail=None):
    """Binarized one-vs-rest problems over the class columns of y_score.
    `tail` (an `_OvrTail`, fold only) rides in the curve program."""
    from ._metrics_impl import _label_map, _selected_labels
    true, kind, present = _targets(y_true, caller)
    n = len(true)
    s = _scores(y_score, n, caller, ndim=2)
    k = s.shape[1]
    classes = present if labels is None else _selected_labels(labels, kind, present)
    if labels is not None:
        if classes != sorted(classes):  # glue: checks the class labels are sorted
            raise ValueError("Parameter 'labels' must be ordered")
        if set(present) - set(classes):
            raise ValueError("'y_true' contains labels not in parameter 'labels'")
    if len(classes) != k:
        raise ValueError("Number of classes in y_true not equal to the number of columns in 'y_score'")
    w = _weights(sample_weight, n, caller)
    index = {c: i for i, c in enumerate(classes)}  # glue: class index table
    codes = _label_map(true, lambda v: index[v])
    # the class-major flags formed by the onehot unit inside the curve
    # program, the supports, scores and average by the tail's units in the
    # same program; no n*k host walk (lane pyglue-sweep: the Python byte
    # layouts, the -D MOJOLEARN_PY2MOJO_core_OFF arm, are gone)
    codes = _i32_c(codes)
    curves = _curves(s, _OneHot(codes, n, k, 0), w, n, k, numeric_mode, stride=k, thresholds=False,
                     keep_flags=keep_flags, lazy=True, compact=True, fold=fold, tail=tail)
    return curves, tail, s, codes, classes, w


_LITTLE = array.array("i", [1]).tobytes()[0] == 1


def _micro_inputs(codes, w, n, k, numeric_mode=None):
    """The micro average's row-major one-hot flags (an `_OneHot`, word
    r * k + c is 1 exactly when row r's code is c) and its weights (a
    `_RepRows`, each of the n repeated k times, or None), both formed by
    stages of the curve program (x_metrics onehot and rep_rows units)."""
    codes = _i32_c(codes)
    if codes.size != n:
        raise ValueError("mojolearn metrics: micro-average codes do not match the scores")
    wm = None
    if w is not None:
        wa = w if isinstance(w, Array) and w.dtype == "<f4" else as_f32_c(w, ndim=1, name="sample_weight")[0]
        if wa.size != n:
            raise ValueError("mojolearn metrics: micro-average weights do not match the scores")
        wm = _RepRows(wa, n, k)
    return _OneHot(codes, n, k, 1), wm


#: arena words one one-vs-one program may hold (its pairs' curve slots);
#: more pairs go to further programs
_OVO_WORDS = 1 << 26
_ROWS_MSG = ("Target scores need to be probabilities for multiclass roc_auc, i.e. they "
             "should sum up to 1.0 over classes")


def _ovo_native(true, index, s, n, k, numeric_mode, average):
    """One-vs-one ROC AUC (scikit-learn _average_multiclass_ovo_score), on
    the device (lane cpu2-l7-metrics S3b): per program, the scores and codes
    go up once, the onehot unit forms every class's flags, `rank_epi`
    OVO_MASK marks each pair's rows as 0/1 curve weights (the curve drops
    the others: the host `x_metrics_ovo_pair` selection is gone), two curve
    folds per pair (column a with class a positive, column b with class b),
    CURVE_SCORE their AUCs, OVO_PAIR the pair score (mean of the two) and
    prevalence ((count_a + count_b) / n, the counts from a group over the
    codes: the host class_sums walk is gone), AVERAGE the macro or weighted
    mean. The first program also runs the multiclass row-sum check
    (ROW_SUM). Pairs whose curve slots exceed `_OVO_WORDS` go to further
    programs; their pair words then meet in one small AVERAGE program."""
    from ._metrics_impl import _label_map
    codes = _i32_c(_label_map(true, lambda v: index[v]))
    if codes.size != n:
        raise ValueError("mojolearn roc_auc_score: y_true and y_score lengths differ")
    P = k * (k - 1) // 2
    per_pair = 17 * n + 4 + 4 * _CF_OUT
    chunk = max(1, min(P, _OVO_WORDS // max(per_pair, 1)))
    pairs = [(a, b) for a in range(k) for b in range(a + 1, k)]  # glue: the stage parameters of each pair
    ps_words, prev_words = [], []
    g0 = 0
    while g0 < P:  # glue: one program per chunk of pairs
        m = min(chunk, P - g0)
        last_only = g0 == 0 and m == P
        prog = _Prog()
        S = prog.put(s)
        C = prog.put_i32(codes)
        OH = prog.scratch(n * k)
        if n:
            prog.stage("onehot", n * k, C, OH, n, k, 0)
        rows = _NONE
        if g0 == 0:
            rows = prog.want(prog.alloc(1), 1)
            if n:
                prog.stage("rank_epi", n, _RANK_ROW_SUM, S, n, k, rows)
        off, _ = _group(prog, C, n, k)
        prog.want(off + k, 1)
        CF = prog.scratch(2 * m * _CF_OUT)
        for t in range(m):  # glue: stages each pair's mask and its two curve folds
            a, b = pairs[g0 + t]
            W = prog.scratch(n)
            if n:
                prog.stage("rank_epi", n, _RANK_OVO_MASK, C, n, a, b, W)
            _fold_stages(prog, S + a, k, OH + a * n, W, n, 1, True, out=CF + 2 * t * _CF_OUT)
            _fold_stages(prog, S + b, k, OH + b * n, W, n, 1, True, out=CF + (2 * t + 1) * _CF_OUT)
        SC = prog.scratch(4 * m)
        FL = prog.want(prog.alloc(2 * m), 2 * m)
        prog.stage("rank_epi", 2 * m, _RANK_CURVE_SCORE, CF, 2 * m, 0, SC, FL, _NONE, 0, 0)
        if last_only:
            PS, PREV = prog.scratch(2 * m), prog.scratch(2 * m)
        else:
            PS, PREV = prog.want(prog.alloc(2 * m), 2 * m), prog.want(prog.alloc(2 * m), 2 * m)
        prog.stage("rank_epi", m, _RANK_OVO_PAIR, SC, off, k, n, g0, m, PS, PREV)
        res = _stage_average(prog, PS, PREV, P, average) if last_only else _NONE
        _run(prog, numeric_mode)
        if g0 == 0:
            if _flag_set(prog, rows):
                raise ValueError(_ROWS_MSG)
            if prog.ints(off + k, 1)[0] != n:
                raise ValueError("mojolearn roc_auc_score: a y_true label is not among the classes")
        for v in prog.ints(FL, 2 * m):  # glue: one warning per direction without both labels
            if v == 1:
                _undefined_warning(_ROC_ONE_CLASS)
        if last_only:
            return prog.words(res, 2, "d")[0]
        ps_words.extend(prog.words(PS, 2 * m, "i"))
        prev_words.extend(prog.words(PREV, 2 * m, "i"))
        g0 += m
    p2 = _Prog()
    res = _stage_average(p2, p2.put_i32(ps_words), p2.put_i32(prev_words), P, average)
    _run(p2, numeric_mode)
    return p2.words(res, 2, "d")[0]


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
        mf = None if max_fpr is None or max_fpr == 1 else max_fpr
        cur = _curves(s, flags, w, n, 1, numeric_mode, fold="auc", max_fpr=mf)[0]
        return _auc_of(cur, mf)
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
    # the multiclass row-sum check runs in the curve program that follows
    # (`rank_epi` ROW_SUM; the host row_sum_range walk is gone)
    if multi_class == "ovr":
        tail = _OvrTail("auc", True, average, numeric_mode)
        curves, tail, s, codes, classes, w = _ovr(y_true, y_score, sample_weight, labels, "roc_auc_score",
                                                  numeric_mode, fold="auc", tail=tail)
        if not tail.rows_ok():
            raise ValueError(_ROWS_MSG)
        if average == "micro":
            n = len(codes)
            flags, wm = _micro_inputs(codes, w, n, k, numeric_mode)
            cur = _curves(s.reshape((n * k,)), flags, wm, n * k, 1, numeric_mode, fold="auc")[0]
            return _auc_of(cur, None)
        return tail.result(curves)
    # one-vs-one (scikit-learn _average_multiclass_ovo_score)
    from ._metrics_impl import _selected_labels
    classes = present if labels is None else _selected_labels(labels, kind, present)
    if len(classes) != k:
        raise ValueError("Number of classes in y_true not equal to the number of columns in 'y_score'")
    index = {c: i for i, c in enumerate(classes)}  # glue: class index table
    return _ovo_native(true, index, s_check, len(true), k, numeric_mode, average)


def average_precision_score(y_true, y_score, *, average="macro", pos_label=1, sample_weight=None,
                            numeric_mode=None):
    """scikit-learn 1.9 `average_precision_score` (binary and multiclass
    one-vs-rest; multilabel-indicator targets are refused by name)."""
    return _NATIVE(None, "family_mexp_curve_average_precision",
                   (y_true, y_score, pos_label, sample_weight, numeric_mode, average),
                   _NATIVE_CONTEXT)


def top_k_accuracy_score(y_true, y_score, *, k=2, normalize=True, sample_weight=None, labels=None,
                         numeric_mode=None):
    """scikit-learn 1.9 `top_k_accuracy_score`: a hit when the true class is
    among the k largest scores, ties ordered as numpy's stable argsort
    reversed (the larger class index first)."""
    return _NATIVE(None, "family_mexp_curve_top_k",
                   (y_true, y_score, k, normalize, sample_weight, labels, numeric_mode),
                   _NATIVE_CONTEXT)


#: rank_epi kinds (x_metrics/rank_epi.mojo)
_RANK_MEAN, _RANK_NDCG_ROW, _RANK_D2_LOG, _RANK_D2_BRIER, _RANK_DCG_DISC = 0, 1, 2, 3, 4


def _f64_words(prog, v):
    """A binary64 scalar as two Int32 arena words (low first), an input."""
    bw = array.array("i")
    bw.frombytes(array.array("d", [float(v)]).tobytes())
    return prog.put_i32(list(bw))


def _row_fold(prog, S, cols, Y, n, kind, w, *, K=0, Dt=_NONE):
    """The row metric and its folds: (TOT, SW, W) offsets: the Float32
    PairSum of the n row values (weighted when w), the weight total (None
    unweighted) and the weights' slot."""
    out = prog.scratch(n)
    prog.stage("row_metric", n, S, cols, Y, out, _ROW[kind], K, Dt)
    zero = prog.scratch(n)
    prog.stage("pair_key", n, 0, 0, zero, 1, 2)
    W = _NONE if w is None else prog.put(w)
    tot = _group(prog, zero, n, 1, values=out, weights=W)[1]
    sw = _group(prog, zero, n, 1, weights=W)[1] if w is not None else None
    return tot, sw, W


def _row_mean(S, cols, Y, n, kind, w, numeric_mode, *, K=0, Dt=_NONE, prog=None, normalize=True, half=False,
              after=()):
    """The (weighted) mean of a row metric, binary64: the folds and the
    division (`rank_epi` MEAN) in one program. `after`: checks of the
    program's flag words, each a function of the program that raises."""
    prog = prog or _Prog()
    tot, sw, _ = _row_fold(prog, S, cols, Y, n, kind, w, K=K, Dt=Dt)
    res = prog.alloc(2)
    prog.stage("rank_epi", 1, _RANK_MEAN, tot, n, _NONE if sw is None else sw, int(bool(normalize)),
               int(bool(half)), res)
    _run(prog, numeric_mode)
    for check in after:  # glue: the program's flag checks
        check(prog)
    return prog.words(res, 2, "d")[0]


def _proba_after(flags, caller):
    return lambda prog: _proba_check(prog, flags, caller)


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


def log_loss_options(y_true, y_pred, normalize, sample_weight, labels, numeric_mode, caller="log_loss"):
    """log_loss with sample_weight (lane/metrics), and without it since lane
    cpu2-l7-metrics: the probabilities' check, the clipped `-log p_true` per
    row, its (weighted) PairSum and the division in one device program."""
    codes, a, k, binary = _proba(y_true, y_pred, labels, None, caller)
    n = len(codes)
    w = _weights(sample_weight, n, caller)
    prog = _Prog()
    S, flags = _put_proba(prog, a, n, k, binary)
    Y = prog.put_i32(codes)
    return float(_row_mean(S, k, Y, n, "logloss", w, numeric_mode, prog=prog, normalize=normalize,
                           after=(_proba_after(flags, caller),)))


def brier_score_loss(y_true, y_proba, *, sample_weight=None, pos_label=None, labels=None,
                     scale_by_half="auto", numeric_mode=None):
    """scikit-learn 1.9 `brier_score_loss` (binary vector or multiclass
    matrix): the mean of `sum_c (onehot - p)^2`, halved by default for the
    binary case."""
    return _NATIVE(None, "family_mexp_rank_brier",
                   (y_true, y_proba, sample_weight, pos_label, labels, scale_by_half, numeric_mode),
                   _NATIVE_CONTEXT)


def d2_log_loss_score(y_true, y_proba=None, *, sample_weight=None, labels=None, numeric_mode=None):
    """scikit-learn 1.9 `d2_log_loss_score`: one minus the log loss over the
    log loss of the (weighted) class frequencies."""
    return _NATIVE(None, "family_mexp_rank_d2_log_loss",
                   (y_true, y_proba, sample_weight, labels, None, numeric_mode),
                   _NATIVE_CONTEXT)


def d2_brier_score(y_true, y_proba, *, sample_weight=None, pos_label=None, labels=None, numeric_mode=None):
    """scikit-learn 1.9 `d2_brier_score`: one minus the Brier score over the
    Brier score of the (weighted) class frequencies."""
    return _NATIVE(None, "family_mexp_rank_d2_brier",
                   (y_true, y_proba, sample_weight, labels, pos_label, numeric_mode),
                   _NATIVE_CONTEXT)


def hinge_loss(y_true, pred_decision, *, labels=None, sample_weight=None, numeric_mode=None):
    """scikit-learn 1.9 `hinge_loss`: binary (labels mapped to -1 / +1, the
    larger class positive) or Crammer-Singer multiclass margins."""
    return _NATIVE(None, "family_mexp_rank_hinge",
                   (y_true, pred_decision, labels, sample_weight, numeric_mode),
                   _NATIVE_CONTEXT)


def dcg_score(y_true, y_score, *, k=None, log_base=2, sample_weight=None, ignore_ties=False,
              numeric_mode=None):
    """scikit-learn 1.9 `dcg_score`: tie-averaged gains by default;
    ignore_ties=True ranks tied scores by DESCENDING column index (numpy's
    `argsort(...)[::-1]` there, whose tie order its unstable sort leaves
    unspecified; this one is fixed)."""
    return _NATIVE(None, "family_mexp_rank_dcg",
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
    return _NATIVE(None, "family_mexp_rank_ndcg",
                   (y_true, y_score, k, sample_weight, ignore_ties, numeric_mode),
                   _NATIVE_CONTEXT)


def coverage_error(y_true, y_score, *, sample_weight=None, numeric_mode=None):
    """scikit-learn 1.9 `coverage_error` (multilabel indicator y_true)."""
    return _NATIVE(None, "family_mexp_rank_coverage", (y_true, y_score, sample_weight, numeric_mode), _NATIVE_CONTEXT)


def label_ranking_average_precision_score(y_true, y_score, *, sample_weight=None, numeric_mode=None):
    """scikit-learn 1.9 `label_ranking_average_precision_score`."""
    return _NATIVE(None, "family_mexp_rank_lrap", (y_true, y_score, sample_weight, numeric_mode), _NATIVE_CONTEXT)


def label_ranking_loss(y_true, y_score, *, sample_weight=None, numeric_mode=None):
    """scikit-learn 1.9 `label_ranking_loss` (tied scores count as misordered)."""
    return _NATIVE(None, "family_mexp_rank_loss", (y_true, y_score, sample_weight, numeric_mode), _NATIVE_CONTEXT)


# ---------------------------------------------------------------------------
# Clustering (scikit-learn 1.9 sklearn/metrics/cluster/)
# ---------------------------------------------------------------------------


def contingency_matrix(labels_true, labels_pred, *, eps=None, sparse=False, dtype="int64", numeric_mode=None):
    """scikit-learn 1.9 `contingency_matrix` (dense): exact counts from the
    device grouping, rows the sorted classes, columns the sorted clusters;
    `eps` adds a constant and makes the result Float64. The result is always
    dense; `sparse=True` raises `NotImplementedError`."""
    return _NATIVE(None, "family_mexp_cluster_contingency",
                   (labels_true, labels_pred, eps, sparse, dtype, numeric_mode),
                   _NATIVE_CONTEXT)


def pair_confusion_matrix(labels_true, labels_pred, *, numeric_mode=None):
    """scikit-learn 1.9 `pair_confusion_matrix`: the 2 x 2 Int64 pair counts
    from the exact contingency matrix (exact Int64 in the binding: every
    pair count is below 2^62)."""
    return _NATIVE(None, "family_mexp_cluster_pair_confusion",
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
    return _NATIVE(None, "family_mexp_cluster_nmi",
                   (labels_true, labels_pred, average_method, numeric_mode),
                   _NATIVE_CONTEXT)


def adjusted_mutual_info_score(labels_true, labels_pred, *, average_method="arithmetic", numeric_mode=None):
    """scikit-learn 1.9 `adjusted_mutual_info_score`: (MI - E[MI]) /
    (mean(H) - E[MI]) with scikit-learn's epsilon guards."""
    return _NATIVE(None, "family_mexp_cluster_ami",
                   (labels_true, labels_pred, average_method, numeric_mode),
                   _NATIVE_CONTEXT)


def calinski_harabasz_score(X, labels, *, numeric_mode=None):
    """scikit-learn 1.9 `calinski_harabasz_score`: the between- over the
    within-cluster dispersion, scaled by (n - k) / (k - 1), in one device
    program."""
    return _NATIVE(None, "family_mexp_cluster_calinski_harabasz", (X, labels, numeric_mode), _NATIVE_CONTEXT)


def davies_bouldin_score(X, labels, *, numeric_mode=None):
    """scikit-learn 1.9 `davies_bouldin_score`: the mean over clusters of the
    worst (s_i + s_j) / d(c_i, c_j), in one device program."""
    return _NATIVE(None, "family_mexp_cluster_davies_bouldin", (X, labels, numeric_mode), _NATIVE_CONTEXT)


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

    def approximate_modes(self, counts, n_train, n_test, numeric_mode=None):
        """StratifiedShuffleSplit's per-class train and test draw counts
        (lane py-runtime-b; it was Python's `model_selection._approximate_mode`
        twice): scikit-learn's `_approximate_mode` in exact integers, the
        train counts n_i over `counts`, then the test counts over counts -
        n_i, each in one `approx_mode` stage (x_metrics/split.mojo) of one
        device program. A tie group the remainders cut is broken by the
        next counter draw, so each call consumes the draws the stages report
        (0 or 1 each), the same draws in the same order as before. Returns
        (n_i, t_i) as lists of ints."""
        k = len(counts)
        prog = _Prog()
        C = prog.put_i32(counts)
        out = prog.want(prog.alloc(2 * k), 2 * k)
        used = prog.want(prog.alloc(2), 2)
        words = []
        for v in (self.seed & 0xFFFFFFFF, self.seed >> 32, self.draws & 0xFFFFFFFF, (self.draws >> 32) & 0xFFFFFFFF):  # glue: four parameter words
            words.append(v - (1 << 32) if v >= 1 << 31 else v)
        prog.stage("approx_mode", k, k, C, _NONE, n_train, out, _NONE, used, *words)
        prog.stage("approx_mode", k, k, C, out, n_test, out + k, used, used + 1, *words)
        _run(prog, numeric_mode)
        drew = prog.ints(used, 2)
        self.draws += drew[0] + drew[1]
        both = prog.ints(out, 2 * k)
        return both[:k], both[k:]

    def permutations(self, sizes, numeric_mode=None):
        """One permutation per size, drawn in order, in one device program."""
        prog = _Prog()
        outs = []
        for n in sizes:  # glue: one output slot per requested draw
            outs.append((prog.want(self.permute_stage(prog, n), n), n))
        if not outs:
            return []
        _run(prog, numeric_mode)
        return [prog.ints(o, n) for o, n in outs]  # glue: reads each requested draw's words

    def permutation(self, n, numeric_mode=None):
        return self.permutations([n], numeric_mode)[0]

    def permutation_rows(self, sizes, numeric_mode=None):
        """`permutations` (the same draws) as array('q') rows: the device
        widens each to Int64 words (`rows64`), so no Python int is made
        per row (lane metrics-apple2)."""
        prog = _Prog()
        outs = []
        for n in sizes:  # glue: one output slot per requested draw
            o = self.permute_stage(prog, n)
            w = prog.want(prog.alloc(2 * n), 2 * n)
            if n:
                prog.stage("rows64", n, o, w)
            outs.append((w, n))
        if not outs:
            return []
        _run(prog, numeric_mode)
        return [prog.words(w, 2 * n, "q") for w, n in outs]  # glue: reads each requested draw's words


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
    # an int32 Array of codes goes in as it is (lane metrics-apple3)
    ENC = prog.put_i32(enc if isinstance(enc, Array) and enc.dtype == "<i4" and enc.ndim == 1
                       else Array._owned(array.array("i", enc), (n,), "<i4", "C"))
    OFF = prog.alloc(m + 1)
    ORD = prog.scratch(n)
    prog.stage("group_sort", 1, ENC, n, m, OFF, ORD)
    PB = _NONE
    if rng is not None:
        at = 0
        for c in range(m):  # glue: one permutation stage per class
            o = rng.permute_stage(prog, counts[c])
            if c == 0:
                PB = o
            if o != PB + at:
                raise AssertionError("stratified_fold_rows: the class permutations are not contiguous")
            at += counts[c]
    cum = []
    for c in range(m):  # glue: fold by class offsets for the device code stage (m-sized: class count)
        acc = 0
        cum.append(0)
        for f in range(k):  # glue: fold by class offsets for the device code stage (k-sized: fold count)
            acc += alloc[f][c]
            cum.append(acc)
    CUM = prog.put_i32(Array._owned(array.array("i", cum), (len(cum),), "<i4", "C"))
    code = prog.scratch(n)
    prog.stage("strat_codes", n, ENC, ORD, OFF, PB, CUM, k, code)
    return _fold_rows_run(prog, n, k, code, _NONE, numeric_mode)


def regression_report(y_true, y_pred, *, sample_weight=None,
                      multioutput="uniform_average", force_finite=True,
                      numeric_mode="identical"):
    """One IDENTICAL report of MSE, MAE and R² using expansion metric contracts.

    C09 A shares target/prediction loads; B retains independent term stages.
    All three requested outputs, validation and synchronization are included.
    NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
    """
    if numeric_mode != "identical":
        raise ValueError("regression_report requires numeric_mode='identical'")
    r = _Reg(y_true, y_pred, sample_weight, multioutput, "regression_report")
    r.program()
    squared = r.term("sq")
    absolute = r.term("abs")
    squared_sums = r.sums(squared)
    mse = r.average(r.fin(squared_sums))
    mae = r.average(r.fin(r.sums(absolute)))
    y_mean = r.fin(r.sums(r.Y), f32=True)[1]
    denominator = _centered(r, r.Y, y_mean, mean=False)
    r2 = _assemble(r, r.fin(squared_sums, 0), denominator, force_finite)
    _run(r.prog, numeric_mode)
    def result(offset):
        if r.mo == "raw_values":
            return Array._owned(r.prog.words(offset, 2*r.D, "d"), (r.D,), "<f8", "C")
        return float(r.prog.words(offset, 2, "d")[0])
    if r.n < 2:
        _undefined_warning("R^2 score is not well-defined with less than two samples.")
        r2_value = float("nan")
    else:
        r2_value = result(r2)
    return {"mean_squared_error": result(mse), "mean_absolute_error": result(mae), "r2_score": r2_value}


def ranking_report(y_true, y_score, *, pos_label=None, sample_weight=None,
                   include_curves=False, numeric_mode="identical"):
    """Binary ROC AUC/AP and optional ROC/PR curves in one native program.

    C10 A reuses a stable score order and canonical prefixes. B executes
    both incumbent curve producers. Weighted ties keep their row order.
    NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
    """
    if numeric_mode != "identical":
        raise ValueError("ranking_report requires numeric_mode='identical'")
    from ._metrics_impl import _label_map
    true, kind, classes = _targets(y_true, "ranking_report")
    if len(classes) > 2:
        raise ValueError("ranking_report currently requires binary targets")
    pos = _pos_label(pos_label, kind, classes, "ranking_report")
    n = len(true)
    if n <= 0:
        raise ValueError("mojolearn metrics: an empty curve has no thresholds")
    scores = _scores(y_score, n, "ranking_report", ndim=1)
    weights = _weights(sample_weight, n, "ranking_report")
    flags = _label_map(true, lambda v: int(v == pos))
    prog = _Prog()
    S, POS, W = prog.put(scores), _put_flags(prog, flags), _put_weights(prog, weights)
    curves = []
    for _ in range(2):  # glue: two fixed requested reports, never rows
        order, fps, tps, thr = prog.scratch(n), prog.scratch(n), prog.scratch(n), prog.scratch(n)
        count = prog.scratch(1)
        curves.append((fps, tps, thr, count))
        prog.stage("bin_curve", 1, S, 1, POS, W, n, order, fps, tps, thr, count, _NONE, 0, 0, 0)
    scalar_outputs = []
    curve_outputs = []
    curve_totals = []
    for fold, (fps, tps, thr, count) in enumerate(curves):  # glue: fixed AUC/AP reports
        folded = prog.alloc(_CF_OUT)
        curve_totals.append(prog.want(folded+2, 2))
        result, status = prog.want(prog.alloc(2), 2), prog.want(prog.alloc(1), 1)
        prog.stage("curve_fold", 1, n, fps, tps, count, folded, fold, 0, 0, _CF_CHUNK)
        prog.stage("rank_epi", 1, _RANK_CURVE_SCORE, folded, 1, fold, result, status, _NONE, 0, 0)
        scalar_outputs.append((result, status))
        if include_curves:
            lengths = prog.want(prog.alloc(6), 6)
            output = prog.alloc(6*(n+1))
            for channel in range(3):  # glue: three native curve output arrays
                prog.want(output+2*channel*(n+1), 2*(n+1), count=lengths, mult=2)
            prog.stage("curve_out", 1, _CO_ROC if fold == 0 else _CO_PR, n,
                       fps, tps, thr, count, 0, output, lengths)
            curve_outputs.append((output, lengths))
    _run(prog, numeric_mode)
    auc_out, auc_flag = scalar_outputs[0]
    ap_out, ap_flag = scalar_outputs[1]
    if prog.ints(auc_flag, 1)[0] == 1:
        _undefined_warning("Only one class is present in y_true. ROC AUC score is not defined in that case.")
    if prog.ints(ap_flag, 1)[0] == 2:
        raise ZeroDivisionError("float division by zero")
    if prog.floats(curve_totals[1]+1, 1)[0] == 0:
        warnings.warn("No positive class found in y_true, recall is set to one for all thresholds.",
                      UserWarning, stacklevel=2)
    result = {"roc_auc": float(prog.words(auc_out, 2, "d")[0]),
              "average_precision": float(prog.words(ap_out, 2, "d")[0])}
    for kind, (output, lengths) in enumerate(curve_outputs):  # glue: wraps two fixed native results
        length = prog.ints(lengths, 1)[0]
        arrays = []
        for channel in range(3):  # glue: output shape/ownership only
            size = length-1 if kind == 1 and channel == 2 else length
            arrays.append(Array._owned(prog.words(output+2*channel*(n+1), 2*size, "d"),
                                       (size,), "<f8", "C"))
        result["roc_curve" if kind == 0 else "precision_recall_curve"] = tuple(arrays)
    return result


# Import-time connection; `_Prog.run` hands each program off once to Mojo.
from ._native_arena_prog import connect as _connect_arena_prog
_PROG_CALL, _PROG_CONTEXT = _connect_arena_prog("x_metrics", _Prog)

# The compiled family owner (bindings/metrics_reg_native.mojo, reached from
# the classical door): the regression metrics above are one native call each.
# It builds the x_metrics program `_Reg` builds, with the fixed op and term
# codes below, and runs it through the shared program runner connected above.
from . import _native_family_bridge  # noqa: E402
_NATIVE, _NATIVE_CONTEXT = _native_family_bridge.connect()
_native_family_bridge.tables(family_metrics_ops=dict(_OPS), family_metrics_terms=dict(_TERM),
                             family_metrics_prog_context=_PROG_CONTEXT,
                             family_metrics_undefined=UndefinedMetricWarning, family_metrics_real=numbers.Real,
                             family_mexp_integral=numbers.Integral)
