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
import ctypes
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
            off_diff=56, flag_scan=57, proba_rows=58, cm_epi=59, reg_epi=60, rank_epi=61, cl_epi=62)
_PARAMS = 14
_NONE = -1


#: `_portable_math.fsum` carries the `math.fsum` fast path this module
#: made first (lane py-shared moved it there, so every caller gains it; the
#: same bits as lane py-bugs' local `_fsum`: `math.fsum` when finite, +0.0
#: for a zero sum, the exact portable sum otherwise).
_fsum = pmath.fsum


def _sq(v):
    """v * v: one correctly rounded product. `v ** 2` calls the platform pow,
    which DEVIATION 6106 does not allow (it is not correctly rounded on every
    host). Lanes py-bugs and py-misc-metrics both made this change."""
    return v * v


def _optional_metrics_entry(binding, name):
    """Optional optimizations may be absent on an otherwise complete CPU facade."""
    binding.x_metrics_run  # Do not conceal a missing mandatory family binding.
    try:
        return getattr(binding, name)
    except (AttributeError, ImportError):
        return None


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


def _new_arena(size):
    return array.array("f", bytes(4 * max(size, 1)))


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


def _class_sums(codes, w, k, numeric_mode):
    """Per-class (weighted) counts, binary64 in row order, from the binding
    (x_metrics/epilogue.mojo class_sums); `codes` int32, `w` Float32 or None."""
    codes = _i32_c(codes)
    waddr, wkeep = _f32_weights_addr(w)
    if waddr is None:
        wkeep = as_f32_c(w, ndim=1, name="sample_weight")[0]
        waddr = addr_ro(wkeep, name="sample_weight")
    out = _f64_out(k)
    _binding(numeric_mode).x_metrics_class_sums(addr_ro(codes, name="codes"), waddr, codes.size, k,
                                                out.buffer_info()[0])
    del wkeep
    return out.tolist()[:k]


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

    def _download(self):
        """The disjoint ascending [lo, hi) ranges the device brings back:
        the declared outputs, else the arena less the inputs and scratch;
        None = the whole arena."""
        if self._outs is not None:
            merged = []
            for lo, hi, cn, mult in sorted(o for o in self._outs if o[2] < 0):  # glue: merges the declared output ranges
                if merged and lo <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], hi)
                else:
                    merged.append([lo, hi, -1, 1])
            return merged + [list(o) for o in self._outs if o[2] >= 0]  # glue: the counted output ranges
        if not self._skip:
            return None
        merged = []
        at = 0
        for lo, hi in sorted(self._skip):  # glue: merges the input and scratch ranges
            if lo > at:
                merged.append([at, lo])
            at = max(at, hi)
        if at < self.size:
            merged.append([at, self.size])
        return [r + [-1, 1] for r in merged]  # glue: marks the download ranges uncounted

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


def _execute(prog, numeric_mode):
    """Run a program on the x_metrics binding (GPU, or the host binding on
    a CPU-only install). A module-level function, not only a method, so the
    lane selector (tools/lane_select.py follows file-local functions, not
    classes) sees every caller reach `x_metrics_run`."""
    arena = _new_arena(prog.size)
    base = arena.buffer_info()[0]
    stages = array.array("i", [v for s in prog._stages for v in s] or [0])  # glue: packs the stage parameter words
    b = _binding(numeric_mode)
    merged = prog._download()
    run_ranges = _optional_metrics_entry(b, "x_metrics_run_ranges") if _arena_io.ranges_enabled() else None
    spans = []
    for off, arr in prog._inputs:  # glue: copies each program input once
        if not arr.size:
            continue
        cache = _arena_io.active_cache(b, "x_metrics", arr.size) if run_ranges is not None else None
        if cache is not None:
            # resident (lane py-shared): the device copies it from the store;
            # the host arena's words stay zero and are never read (`_check`)
            spans.append((off, off + arr.size, cache.id_of(arr)))
            continue
        ctypes.memmove(base + 4 * off, addr_ro(arr, name="input"), 4 * arr.size)
        spans.append((off, off + arr.size, -1))
    run_out = _optional_metrics_entry(b, "x_metrics_run_out") if merged is not None else None
    if run_ranges is not None:
        # the shared ranges runner (lane py-shared, core/arena_io.mojo): only
        # the inputs go up, only the outputs come back
        ins = _arena_io.input_ranges(spans)
        outs = [list(r) for r in merged] if merged is not None else [[0, prog.size, -1, 1]]  # glue: the download range descriptors
        ia, oa = _arena_io.pack_ins(ins), _arena_io.pack_outs(outs)
        run_ranges(base, stages.buffer_info()[0], (prog.size, len(prog._stages), len(ins), len(outs)),
                   ia.buffer_info()[0], oa.buffer_info()[0])
    elif run_out is not None:
        outs = array.array("i", [v for r in merged for v in r] or [0, 0, -1, 1])  # glue: packs the download range descriptors
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
    return true, pred, kind, sorted(_label_set(true) | _label_set(pred)), w


def _codes(labels_seq, order):
    from ._metrics_impl import _label_map
    index = {c: i for i, c in enumerate(order)}
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
        keep = tail(prog, groups) if tail is not None else None
        _execute(prog, numeric_mode)
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


import threading as _threading

#: classification_report's per-call memo (thread local; None outside it)
_REPORT = _threading.local()


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
    # classification_report asks for the same inputs four times: inside it
    # the encoded pair and the per-label sums are made once (lane
    # metrics-apple3; `_REPORT.memo` is None everywhere else)
    memo = getattr(_REPORT, "memo", None)
    same = (id(y_true), id(y_pred), id(sample_weight))
    hit = None if memo is None else memo.get(("pair", same))
    if hit is None:
        hit = _pair(y_true, y_pred, sample_weight, "precision_recall_fscore_support")
        if memo is not None:
            memo[("pair", same)] = hit
    true, pred, kind, present, w = hit
    chosen = _set_wise_labels(present, kind, average, labels, pos_label, "precision_recall_fscore_support")
    order = _label_order(chosen, present)
    k = len(chosen)
    fb = float(beta)
    fmode = 0 if pmath.isinf(fb) else (1 if fb == 0 else 2)
    if memo is None:
        # the epilogue runs in the sums' program (x_metrics/cls_epi.mojo)
        s = _Sums(true, pred, w, order, numeric_mode,
                  epi=(2, k) + _epi_codes(average, zero_division) + (fmode, fb * fb))
        res = s.epi
    else:
        # classification_report asks for the four averages of one input: one
        # program makes the sums and stages all four epilogues
        key = ("sums", same, tuple(order), numeric_mode, zero_division if isinstance(zero_division, str)
               else float(zero_division), fb)
        hit = memo.get(key)
        if hit is None:
            s = _Sums(true, pred, w, order, numeric_mode,
                      epi=[(2, k) + _epi_codes(a, zero_division) + (fmode, fb * fb) for a in _REPORT_AVERAGES])  # glue: the report's four averages
            hit = memo[key] = (s, dict(zip(_REPORT_AVERAGES, s.epi)))
        s, res = hit[0], hit[1][average]
    bad, _, scal, vals = res
    if isinstance(zero_division, str):
        for bit, what in ((1, "precision"), (2, "recall"), (4, "f-score")):  # glue: the three metric names
            if bad & bit and what in warn_for:
                _undefined_warning(f"{what.capitalize()} is ill-defined and being set to 0.0 in labels "
                                   "with no predicted/true samples. Use `zero_division` parameter to "
                                   "control this behavior.")
    if average is None:
        ts = s.true[:k]
        support = (Array.from_list([float(v) for v in ts], "<f8") if s.weighted  # glue: the k label supports
                   else Array.from_list([int(v) for v in ts], "<i8"))  # glue: the k label supports
        return (Array.from_list(list(vals[:k]), "<f8"), Array.from_list(list(vals[k:2 * k]), "<f8"),
                Array.from_list(list(vals[2 * k:3 * k]), "<f8"), support)
    return (scal[0], scal[1], scal[2], None)


#: the averages classification_report asks precision_recall_fscore_support for
_REPORT_AVERAGES = (None, "micro", "macro", "weighted")


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
    k = len(chosen)
    s = _Sums(true, pred, w, _label_order(chosen, present), numeric_mode,
              epi=(0, k) + _epi_codes(average, zero_division) + (0, 0.0))
    # the device epilogue (x_metrics/cls_epi.mojo): the binary64 ratios and averages
    bad, _, scal, vals = s.epi
    if bad and isinstance(zero_division, str):
        _undefined_warning("Jaccard is ill-defined and being set to 0.0 in labels with no true or "
                           "predicted samples. Use `zero_division` parameter to control this behavior.")
    if average is None:
        return Array.from_list(list(vals[:k]), "<f8")
    return scal[0]


def balanced_accuracy_score(y_true, y_pred, *, sample_weight=None, adjusted=False, numeric_mode=None):
    """scikit-learn 1.9 `balanced_accuracy_score`: the mean per-class recall
    over the classes present in y_true (a class with no true weight is
    dropped with sklearn's warning), chance-adjusted when `adjusted`."""
    true, pred, kind, present, w = _pair(y_true, y_pred, sample_weight, "balanced_accuracy_score")
    s = _Sums(true, pred, w, present, numeric_mode, epi=(1, len(present), 0, 1 if adjusted else 0, 0, 0.0))
    # the device epilogue (x_metrics/cls_epi.mojo): the mean recall over the kept
    # classes and the chance adjustment, binary64
    if s.epi[1] != len(present):
        warnings.warn("y_pred contains classes not in y_true", stacklevel=2)
    return float(s.epi[2][0])


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


def _accuracy_sums(y_true, y_pred, sample_weight, numeric_mode):
    """(the (weighted) number of matches, the row count or weight total)."""
    true, pred, kind, present, w = _pair(y_true, y_pred, sample_weight, "accuracy_score")
    s = _Sums(true, pred, w, present, numeric_mode)
    hit = 0
    for v in s.tp:
        hit += v
    return (float(hit) if s.weighted else int(hit)), s.total


def accuracy_count(y_true, y_pred, sample_weight, numeric_mode):
    """accuracy_score(normalize=False): the (weighted) number of matches."""
    return _accuracy_sums(y_true, y_pred, sample_weight, numeric_mode)[0]


def accuracy_fraction(y_true, y_pred, sample_weight=None, numeric_mode=None):
    """sklearn's ClassifierMixin.score / accuracy_score over labels of any
    kind: the (weighted) match count over the row count or the weight total,
    both from the x_metrics binding's grouped sums (lane pyglue-sweep: the
    weight total is the program's PairSum, not a host sum). Lane
    cgr4-py-compute: the estimators' `score` methods called this instead of
    a per-row Python comparison."""
    hit, total = _accuracy_sums(y_true, y_pred, sample_weight, numeric_mode)
    if sample_weight is None:
        return float(hit) / max(total, 1)
    return float(hit) / total


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
    names = [str(t) for t in target_names] if target_names is not None else [str(c) for c in chosen]  # glue: formats the report's row names
    headers = ["precision", "recall", "f1-score", "support"]
    if getattr(_REPORT, "memo", None) is None:
        _REPORT.memo = {}
        try:
            return _classification_report(y_true, y_pred, labels_given, chosen, present, names, headers,
                                          micro_is_accuracy, sample_weight, digits, output_dict,
                                          zero_division, numeric_mode)
        finally:
            _REPORT.memo = None
    return _classification_report(y_true, y_pred, labels_given, chosen, present, names, headers,
                                  micro_is_accuracy, sample_weight, digits, output_dict, zero_division,
                                  numeric_mode)


def _classification_report(y_true, y_pred, labels_given, chosen, present, names, headers, micro_is_accuracy,
                           sample_weight, digits, output_dict, zero_division, numeric_mode):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=chosen, average=None,
                                                 sample_weight=sample_weight,
                                                 zero_division=zero_division, numeric_mode=numeric_mode)
    rows = list(zip(names, p.tolist(), r.tolist(), f.tolist(), s.tolist()))
    averages = (["micro avg"] if not micro_is_accuracy else []) + ["macro avg", "weighted avg"]
    report = {}
    for name, a, b, c, d in rows:  # glue: formats the text report rows
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
    for avg in averages:  # glue: formats the report's average rows
        a, b, c, _ = precision_recall_fscore_support(y_true, y_pred, labels=chosen,
                                                     average=avg.split()[0],
                                                     sample_weight=sample_weight,
                                                     zero_division=zero_division, numeric_mode=numeric_mode)
        report[avg] = dict(zip(headers, (a, b, c, total)))
        avg_rows.append((avg, a, b, c, total))
    if output_dict:
        return report
    width = max([len(n) for n in names] + [len("weighted avg"), digits])  # glue: the report's text column width
    head_fmt = "{:>{width}s} " + " {:>9}" * len(headers)
    out = head_fmt.format("", *headers, width=width) + "\n\n"
    row_fmt = "{:>{width}s} " + " {:>9.{digits}f}" * 3 + " {:>9}\n"
    for name, a, b, c, d in rows:  # glue: formats the text report rows
        out += row_fmt.format(name, a, b, c, d, width=width, digits=digits)
    out += "\n"
    if micro_is_accuracy:
        acc_fmt = "{:>{width}s} " + " {:>9.{digits}}" * 2 + " {:>9.{digits}f}" + " {:>9}\n"
        out += acc_fmt.format("accuracy", "", "", report["accuracy"], total, width=width, digits=digits)
    for avg, a, b, c, d in avg_rows:  # glue: formats the report's average rows
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
        for kind in kinds:  # glue: one term stage per requested error kind
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
        # A constant target with force_finite=False can have score -inf
        # and variance weight zero. IEEE leaves the NaN sign from 0 * inf
        # implementation-dependent (Arm +NaN, x86 -NaN). This weighted
        # score is undefined: return the same explicit NaN as 0/0 above,
        # before performing the invalid operation. Do not drop the term.
        if _math.isnan(v) or (w == 0 and _math.isinf(v)):
            return float("nan")
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
        """The binding's epilogue entry `name`, its words checked readable."""
        fn = _epilogue(name, self.numeric_mode)
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
            cur.keep = bytes(_arena_view(p.arena).cast("B")[lo + (0 if _LITTLE else 3):lo + 4 * c:4])
        elif self.keep == -2:
            cur.keep = b"\x01" * c       # the device dropped the collinear points already
        return cur


def _epilogue(name, numeric_mode):
    """The binding's epilogue entry `name` (x_metrics/epilogue.mojo, lane
    py-misc-metrics). Every install's x_metrics binding carries it: the
    MOJOLEARN_METRICS_EPILOGUE=python and MOJOLEARN_HOTPATH=python reference
    arms are gone (lane pyglue-sweep: Python is glue only)."""
    return getattr(_binding(numeric_mode), name)


def _dev_epilogue(dev, name):
    """`_epilogue` for a `_DevCurve` with thresholds, its fps, tps and
    thresholds words checked readable (declared outputs)."""
    if dev.c <= 0 or dev.thr < 0:
        raise ValueError("mojolearn metrics: an empty curve has no thresholds")
    fn = _epilogue(name, dev.numeric_mode)
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


def _f32_weights_addr(w):
    if w is None:
        return 0, None
    if isinstance(w, Array) and w.dtype == "<f4" and w._has_order("C"):
        return addr_ro(w, name="sample_weight"), w
    return None, None


def _auc_of(cur, max_fpr):
    """`_binary_auc` of a curve
    when both classes are present, else (or on any doubt) the Python.
    A `_FoldCurve` (the device fold, lane cgr2-metrics-shap) scores itself."""
    if isinstance(cur, _FoldCurve):
        return cur.auc(max_fpr)
    L = cur.lists()
    return _binary_auc(L[0], L[1], max_fpr, L.keep)


def _ap_of(cur):
    """`_binary_ap` of a curve: a `_FoldCurve` (the device fold, lane
    cgr2-metrics-shap) scores itself."""
    if isinstance(cur, _FoldCurve):
        return cur.ap()
    L = cur.lists()
    return _binary_ap(L[0], L[1])


def _curves(scores, flags, w, n, problems, numeric_mode, *, stride=1, thresholds=True, keep_flags=True,
            lazy=False, compact=False, fold=None, max_fpr=None):
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
    max_fpr (fold="auc") folds the partial AUC instead."""
    if fold is not None:
        return _fold_curves(scores, flags, w, n, problems, numeric_mode, stride=stride, fold=fold,
                            max_fpr=max_fpr)
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
    _execute(prog, numeric_mode)
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


def _fold_curves(scores, flags, w, n, problems, numeric_mode, *, stride=1, fold="auc", max_fpr=None):
    """`_curves` with the device curve fold: the curve stays on the device
    (compacted for the AUCs) and only the fold's words come back."""
    prog = _Prog()
    S = prog.put(scores)
    POS = _put_flags(prog, flags)
    W = _put_weights(prog, w)
    N = n * problems
    order = prog.scratch(N)
    auc = fold == "auc"
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
    out = prog.want(prog.alloc(_CF_OUT * problems), _CF_OUT * problems)
    mode, mh, ml = (1, 0, 0) if not auc else ((0, 0, 0) if max_fpr is None else (2,) + _f32_split(max_fpr))
    if auc:
        prog.stage("curve_fold", problems, n, CF, CF + N, CM, out, mode, mh, ml, _CF_CHUNK)
    else:
        prog.stage("curve_fold", problems, n, fps, tps, cnt, out, mode, mh, ml, _CF_CHUNK)
    _execute(prog, numeric_mode)
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
            # `_binary_auc`'s empty-class answer, without bringing the curve back
            _undefined_warning("Only one class is present in y_true. ROC AUC score is not defined in that case.")
            return float("nan")
        if max_fpr is None:
            return float(s / (2.0 * F * T))
        if k >= c:
            # max_fpr at or past the last point: the Python rule
            L = self.lists()
            return _binary_auc(L[0], L[1], max_fpr, L.keep)
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
            L = self.lists()
            return _binary_ap(L[0], L[1])
        return float(max(0.0, s / T))


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
    # the three Float64 arrays straight from the arena words
    # (x_metrics/epilogue.mojo roc_arrays; lane metrics-apple2), an empty
    # class's NaN rates there too (lane pyglue-sweep)
    F, T = dev.last()
    if F <= 0:
        _undefined_warning("No negative samples in y_true, false positive value should be meaningless")
    if T <= 0:
        _undefined_warning("No positive samples in y_true, true positive value should be meaningless")
    fn = dev.native("x_metrics_curve_roc")
    bufs = [_f64_out(dev.c + 1) for _ in range(3)]  # glue: three output buffers
    m = int(fn(dev.addr(), (dev.fps, dev.tps, dev.thr, dev.keep), dev.c,
               1 if drop_intermediate else 0, tuple(b.buffer_info()[0] for b in bufs)))  # glue: three buffer addresses
    return tuple(_f64_array(b, m) for b in bufs)  # glue: three output arrays


def precision_recall_curve_options(y_true, y_score, pos_label, sample_weight, drop_intermediate, numeric_mode):
    """precision_recall_curve with sample_weight or drop_intermediate=True
    (lane/metrics): Float64 outputs; the default call keeps its kernel."""
    dev, _ = _binary_curve(y_true, y_score, pos_label, sample_weight, numeric_mode, "precision_recall_curve",
                           lazy=True)
    # the three Float64 arrays straight from the arena words
    # (x_metrics/epilogue.mojo pr_arrays; lane py-misc-metrics), no
    # positives' recall of one there too (lane pyglue-sweep)
    fn = _dev_epilogue(dev, "x_metrics_curve_pr")
    if dev.last()[1] == 0:
        warnings.warn("No positive class found in y_true, recall is set to one for all thresholds.",
                      UserWarning, stacklevel=3)
    bufs = [_f64_out(dev.c + 1) for _ in range(3)]  # glue: three output buffers
    m = int(fn(dev.addr(), (dev.fps, dev.tps, dev.thr), dev.c, 1 if drop_intermediate else 0,
               tuple(b.buffer_info()[0] for b in bufs)))  # glue: three buffer addresses
    return _f64_array(bufs[0], m + 1), _f64_array(bufs[1], m + 1), _f64_array(bufs[2], m)


def det_curve(y_true, y_score, *, pos_label=None, sample_weight=None, drop_intermediate=False,
              numeric_mode=None):
    """scikit-learn 1.9 `det_curve`: fpr, fnr, thresholds."""
    dev, classes = _binary_curve(y_true, y_score, pos_label, sample_weight, numeric_mode, "det_curve", lazy=True)
    if len(classes) != 2:
        raise ValueError("Only one class is present in y_true. Detection error tradeoff curve is not "
                         "defined in that case.")
    # x_metrics/epilogue.mojo det_arrays (lane py-misc-metrics); a zero class
    # weight raises there as the Python division by zero did
    fn = _dev_epilogue(dev, "x_metrics_curve_det")
    bufs = [_f64_out(dev.c + 1) for _ in range(3)]  # glue: three output buffers
    m = int(fn(dev.addr(), (dev.fps, dev.tps, dev.thr), dev.c, 1 if drop_intermediate else 0,
               tuple(b.buffer_info()[0] for b in bufs)))  # glue: three buffer addresses
    return tuple(_f64_array(b, m) for b in bufs)  # glue: three output arrays


def _trapezoid(x, y):
    """fsum of (x[i] - x[i-1]) * (y[i] + y[i-1]) / 2, the same binary64
    operations per term, iterated in C (lane metrics-apple)."""
    terms = map(operator.truediv,
                map(operator.mul, map(operator.sub, x[1:], x[:-1]), map(operator.add, y[1:], y[:-1])),
                itertools.repeat(2))
    return _fsum(list(terms))


def auc(x, y):
    """scikit-learn 1.9 `auc`: the trapezoid rule over a monotonic x (host
    binary64, a correctly rounded `fsum`; a non-finite term's IEEE sum), in
    the binding (x_metrics/epilogue.mojo auc_xy, lane py-misc-metrics) for
    every input: a list or an integer buffer is converted to Float64 first
    (lane apple-fast-py2mojo-core; the Python trapezoid route is gone, lane
    pyglue-sweep)."""
    xa, ya = _auc_f64(x), _auc_f64(y)
    if xa is None:
        xa = Array.from_list(flatten_mo(x), "<f8")
    if ya is None:
        ya = Array.from_list(flatten_mo(y), "<f8")
    if xa.size != ya.size:
        raise ValueError("x and y must have the same length")
    if xa.size < 2:
        raise ValueError(f"At least 2 points are needed to compute area under curve, but x.shape = ({xa.size},)")
    try:
        return float(_binding(None).x_metrics_auc_xy(addr_ro(xa, name="x"), addr_ro(ya, name="y"), xa.size))
    except Exception as exc:
        if "non-monotonic" in str(exc):
            raise ValueError(f"x is neither increasing nor decreasing : {xa.tolist()}.") from None
        raise


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


def _ovr(y_true, y_score, sample_weight, labels, caller, numeric_mode, keep_flags=True, fold=None):
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
    # the class-major flags formed by the onehot unit inside the curve
    # program, the support by the binding's class_sums (the same counts,
    # binary64 in row order); no n*k host walk (lane pyglue-sweep: the
    # Python byte layouts, the -D MOJOLEARN_PY2MOJO_core_OFF arm, are gone)
    codes = _i32_c(codes)
    curves = _curves(s, _OneHot(codes, n, k, 0), w, n, k, numeric_mode, stride=k, thresholds=False,
                     keep_flags=keep_flags, lazy=True, compact=True, fold=fold)
    return curves, _class_sums(codes, w, k, numeric_mode), s, codes, classes, w


_LITTLE = array.array("i", [1]).tobytes()[0] == 1


def _rows_sum_to_one(s, k, numeric_mode=None):
    """No row's correctly rounded sum is farther than 1e-8 + 1e-5 from 1
    (scikit-learn's check; the scores are finite float32): the row fsums in
    the binding (x_metrics/epilogue.mojo row_sum_range; lane metrics-apple2),
    whose largest and smallest decide every row since |fl(s - 1)| is monotone
    on either side of 1. The Python fallback is gone (lane
    apple-fast-py2mojo-core: every install's x_metrics binding carries
    `x_metrics_row_sum_range`)."""
    n = s.size // k if k else 0
    if n <= 0:
        return True
    if not (s.dtype == "<f4" and s._has_order("C")):
        s = as_f32_c(s, ndim=s.ndim, name="y_score")[0]
    out = array.array("d", [0.0, 0.0])
    _binding(numeric_mode).x_metrics_row_sum_range(addr_ro(s, name="y_score"), n, k, out.buffer_info()[0])
    tol = 1e-8 + 1e-5
    return not (abs(out[0] - 1) > tol or abs(out[1] - 1) > tol)


def _average_scores(scores, support, average):
    if average is None:
        return Array.from_list(scores, "<f8")
    if average == "weighted":
        total = _fsum(support)
        return float(_fsum([a * b for a, b in zip(scores, support)]) / total) if total else 0.0
    return float(_fsum(scores) / len(scores))


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


def _ovo_native(true, index, s, n, k, numeric_mode):
    """(pair_scores, prevalence) of one-vs-one ROC AUC with each pair's
    rows, scores and flags selected by the binding (`x_metrics_ovo_pair`,
    lane metrics-apple3): the rows of the pair in ascending order, their
    Float32 scores in the pair's two columns and the 0/1 flags, the words
    `[vals[r][col] for r in rows]` and `[1 if codes[r] == pos else 0 for r
    in rows]` would hold. The two curve programs and AUCs per pair are
    scikit-learn's definition. cpu-gpu-cleanup c-metrics-prep: the only
    path (no size threshold, no per-row Python selection)."""
    from ._metrics_impl import _label_map
    b = _binding(numeric_mode)
    pair = b.x_metrics_ovo_pair
    sums = b.x_metrics_class_sums
    if not s._has_order("C"):
        s = as_f32_c(s, ndim=2, name="y_score")[0]
    codes = _label_map(true, lambda v: index[v])
    if not isinstance(codes, Array) or codes.dtype != "<i4":
        codes = Array.from_list(list(codes.tolist() if isinstance(codes, Array) else codes), "<i4")
    if codes.size != n:
        raise ValueError("mojolearn roc_auc_score: y_true and y_score lengths differ")
    per = array.array("d", bytes(8 * k))
    sums(addr_ro(codes, name="codes"), 0, n, k, per.buffer_info()[0])
    counts = [int(v) for v in per]
    if sum(counts) != n:
        raise ValueError("mojolearn roc_auc_score: a y_true label is not among the classes")
    pair_scores, prevalence = [], []
    for a in range(k):
        for bcol in range(a + 1, k):
            m = counts[a] + counts[bcol]
            prevalence.append(m / n)
            if m < 1:
                sa, sb = Array.from_list([], "<f4"), Array.from_list([], "<f4")
                fa, fb = Array.from_list([], "<i4"), Array.from_list([], "<i4")
            else:
                sa, sb = empty((m,), "<f4"), empty((m,), "<f4")
                fa, fb = empty((m,), "<i4"), empty((m,), "<i4")
                got = int(pair(addr_ro(codes, name="codes"), addr_ro(s, name="y_score"), (n, k, a, bcol, m),
                               (sa._addr, sb._addr, fa._addr, fb._addr)))
                if got != m:
                    raise RuntimeError("mojolearn roc_auc_score: x_metrics_ovo_pair selected %d rows, "
                                       "counted %d" % (got, m))
            both = []
            for sv, fl in ((sa, fa), (sb, fb)):  # glue: the pair's two one-vs-one directions
                cur = _curves(sv, fl, None, m, 1, numeric_mode, fold="auc")[0]
                both.append(_auc_of(cur, None))
            pair_scores.append((both[0] + both[1]) / 2)
    return pair_scores, prevalence


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
    if not _rows_sum_to_one(s_check, k, numeric_mode):
        raise ValueError("Target scores need to be probabilities for multiclass roc_auc, i.e. they "
                         "should sum up to 1.0 over classes")
    if multi_class == "ovr":
        curves, support, s, codes, classes, w = _ovr(y_true, y_score, sample_weight, labels, "roc_auc_score",
                                                     numeric_mode, fold="auc")
        if average == "micro":
            n = len(codes)
            flags, wm = _micro_inputs(codes, w, n, k, numeric_mode)
            cur = _curves(s.reshape((n * k,)), flags, wm, n * k, 1, numeric_mode, fold="auc")[0]
            return _auc_of(cur, None)
        scores = [_auc_of(c, None) for c in curves]
        return _average_scores(scores, support, average)
    # one-vs-one (scikit-learn _average_multiclass_ovo_score)
    from ._metrics_impl import _selected_labels
    classes = present if labels is None else _selected_labels(labels, kind, present)
    if len(classes) != k:
        raise ValueError("Number of classes in y_true not equal to the number of columns in 'y_score'")
    index = {c: i for i, c in enumerate(classes)}
    pair_scores, prevalence = _ovo_native(true, index, s_check, len(true), k, numeric_mode)
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
        return _ap_of(_curves(s, flags, w, n, 1, numeric_mode, fold="ap")[0])
    if pos_label != 1:
        raise ValueError("Parameter pos_label is fixed to 1 for multiclass y_true. Do not set pos_label "
                         "or set pos_label to 1.")
    if average == "samples":
        raise NotImplementedError("mojolearn average_precision_score: average='samples' applies to "
                                  "multilabel targets, which are NOT IMPLEMENTED")
    curves, support, s, codes, classes, w = _ovr(y_true, y_score, sample_weight, None,
                                                 "average_precision_score", numeric_mode, keep_flags=False,
                                                 fold="ap")
    k = len(classes)
    if average == "micro":
        n = len(codes)
        flags, wm = _micro_inputs(codes, w, n, k, numeric_mode)
        return _ap_of(_curves(s.reshape((n * k,)), flags, wm, n * k, 1, numeric_mode, fold="ap")[0])
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
            # the columns laid out by the pair_cols unit (thr's bits, then s
            # bit for bit)
            import struct
            S1 = prog.put(flat)
            S = prog.scratch(2 * n)
            prog.stage("pair_cols", n, S1, S, struct.unpack("<i", struct.pack("<f", thr))[0])
            kk, cols = 1, 2
        else:
            # zero words: a scratch slot starts zero on every backend
            S = prog.scratch(2 * n)
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
    _execute(prog, numeric_mode)
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
        classes = sorted(classes)
        k = a.shape[1]
        if len(classes) != k:
            raise ValueError(f"y_true and y_proba contain different number of classes: {len(classes)} vs {k}")
        index = {c: i for i, c in enumerate(classes)}
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
    codes, a, k, binary = _proba(y_true, y_proba, labels, pos_label, "brier_score_loss")
    n = len(codes)
    w = _weights(sample_weight, n, "brier_score_loss")
    if scale_by_half == "auto":
        scale_by_half = binary or k < 3
    prog = _Prog()
    S, flags = _put_proba(prog, a, n, k, binary)
    Y = prog.put_i32(codes)
    return float(_row_mean(S, k, Y, n, "brier", w, numeric_mode, prog=prog, half=bool(scale_by_half),
                           after=(_proba_after(flags, "brier_score_loss"),)))


def _d2_proba(y_true, y_proba, sample_weight, labels, pos_label, numeric_mode, caller, kind):
    """d2_log_loss_score / d2_brier_score: the row metric's folds, the
    (weighted) class sums and `1 - num / den` (`rank_epi` D2_LOG / D2_BRIER:
    the class-frequency denominator in binary64) in one device program."""
    codes, a, k, binary = _proba(y_true, y_proba, labels, pos_label, caller)
    n = len(codes)
    if n < 2:
        _undefined_warning("D^2 score is not well-defined with less than two samples.")
        return float("nan")
    w = _weights(sample_weight, n, caller)
    prog = _Prog()
    S, flags = _put_proba(prog, a, n, k, binary)
    Y = prog.put_i32(codes)
    tot, sw, W = _row_fold(prog, S, k, Y, n, kind, w)
    off, per = _group(prog, Y, n, k, weights=W)
    PER, weighted = (off, 0) if w is None else (per, 1)
    res = prog.alloc(2)
    if kind == "logloss":
        prog.stage("rank_epi", 1, _RANK_D2_LOG, tot, PER, weighted, k, res)
    else:
        prog.stage("rank_epi", 1, _RANK_D2_BRIER, tot, n, _NONE if sw is None else sw, PER, weighted, k, res)
    _execute(prog, numeric_mode)
    _proba_check(prog, flags, caller)
    return float(prog.words(res, 2, "d")[0])


def d2_log_loss_score(y_true, y_proba=None, *, sample_weight=None, labels=None, numeric_mode=None):
    """scikit-learn 1.9 `d2_log_loss_score`: one minus the log loss over the
    log loss of the (weighted) class frequencies."""
    return _d2_proba(y_true, y_proba, sample_weight, labels, None, numeric_mode, "d2_log_loss_score", "logloss")


def d2_brier_score(y_true, y_proba, *, sample_weight=None, pos_label=None, labels=None, numeric_mode=None):
    """scikit-learn 1.9 `d2_brier_score`: one minus the Brier score over the
    Brier score of the (weighted) class frequencies."""
    return _d2_proba(y_true, y_proba, sample_weight, labels, pos_label, numeric_mode, "d2_brier_score", "brier")


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
    # `indicator`: the 0/1 test of y_true runs in the caller's program
    # (`_relevance_put`: flag_scan, x_metrics/tail.mojo)
    return y, s


def _relevance_put(prog, y, s, caller, indicator):
    """(S, Y, after): y_score and y_true in the program and, for an
    indicator y_true, the device 0/1 test and its check."""
    S, Y = prog.put(s), prog.put(y)
    if not indicator:
        return S, Y, ()
    flag = _scan_flag(prog, Y, y.size, _SCAN_INDICATOR)

    def check(prog):
        if _flag_set(prog, flag):
            raise ValueError(f"{caller} requires a binary label indicator y_true")
    return S, Y, (check,)


def _dcg_table(prog, c, log_base):
    """The c-word Float32 DCG discount table `1 / log_b(i + 2)` (binary64,
    narrowed), formed by the device (`rank_epi` DCG_DISC) in the program."""
    b = float(log_base)
    if not b > 0.0:
        raise ValueError("math domain error")
    if b == 1.0:
        raise ZeroDivisionError("float division by zero")
    B = _f64_words(prog, b)
    D = prog.scratch(c)
    if c:
        prog.stage("rank_epi", c, _RANK_DCG_DISC, c, B, D)
    return D


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
    S, Y, _ = _relevance_put(prog, y, s, "dcg_score", False)
    Dt = _dcg_table(prog, c, log_base)
    return float(_row_mean(S, c, Y, n, "dcg_ignore_ties" if ignore_ties else "dcg", w, numeric_mode,
                           K=0 if k is None else int(k), Dt=Dt, prog=prog))


def ndcg_score(y_true, y_score, *, k=None, sample_weight=None, ignore_ties=False, numeric_mode=None):
    """scikit-learn 1.9 `ndcg_score`: each row's DCG over its ideal DCG
    (0 when the row has no relevant item), (weighted) averaged. The gains,
    the per-row ratios (`rank_epi` NDCG_ROW: the binary64 quotient narrowed
    to Float32), their (weighted) PairSum and the mean run in one device
    program (lane cpu2-l7-metrics: no host epilogue over the rows)."""
    y, s = _relevance(y_true, y_score, "ndcg_score", indicator=False)
    n, c = y.shape
    if c <= 1:
        raise ValueError(f"Computing NDCG is only meaningful when there is more than 1 document. Got {c} instead.")
    w = _weights(sample_weight, n, "ndcg_score")
    K = 0 if k is None else int(k)
    prog = _Prog()
    S, Y, _ = _relevance_put(prog, y, s, "ndcg_score", False)
    neg = _scan_flag(prog, Y, y.size, _SCAN_NEGATIVE)
    Dt = _dcg_table(prog, c, 2)
    gain = prog.scratch(n)
    ideal = prog.scratch(n)
    ratio = prog.scratch(n)
    prog.stage("row_metric", n, S, c, Y, gain, _ROW["dcg_ignore_ties" if ignore_ties else "dcg"], K, Dt)
    prog.stage("row_metric", n, Y, c, Y, ideal, _ROW["dcg"], K, Dt)
    prog.stage("rank_epi", n, _RANK_NDCG_ROW, gain, ideal, n, ratio)
    zero = prog.scratch(n)
    prog.stage("pair_key", n, 0, 0, zero, 1, 2)
    W = _NONE if w is None else prog.put(w)
    tot = _group(prog, zero, n, 1, values=ratio, weights=W)[1]
    sw = _group(prog, zero, n, 1, weights=W)[1] if w is not None else None
    res = prog.alloc(2)
    prog.stage("rank_epi", 1, _RANK_MEAN, tot, n, _NONE if sw is None else sw, 1, 0, res)
    _execute(prog, numeric_mode)
    if _flag_set(prog, neg):
        raise ValueError("ndcg_score should not be used on negative y_true values.")
    return float(prog.words(res, 2, "d")[0])


def _label_ranking(kind, y_true, y_score, sample_weight, numeric_mode, caller):
    y, s = _relevance(y_true, y_score, caller, indicator=True)
    n, c = y.shape
    w = _weights(sample_weight, n, caller)
    prog = _Prog()
    S, Y, after = _relevance_put(prog, y, s, caller, True)
    return float(_row_mean(S, c, Y, n, kind, w, numeric_mode, prog=prog, after=after))


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
    for v, name in ((labels_true, "labels_true"), (labels_pred, "labels_pred")):  # glue: validates the two named arguments
        _refuse_multilabel(v, name, caller)
    a, ka = _classification_encoded(labels_true, "labels_true")
    b, kb = _classification_encoded(labels_pred, "labels_pred")
    if len(a) != len(b):
        raise ValueError(f"mojolearn {caller}: labels_true and labels_pred must have the same length")
    return a, b, sorted(_label_set(a)), sorted(_label_set(b))


class _Contingency:
    """The exact contingency counts of two clusterings and what the
    clustering metrics read from them, all formed by the binding
    (x_metrics/epilogue.mojo contingency_stats, lane pyglue-sweep): `C`
    (ka x kb Int64, rows = classes of `a`, columns = classes of `b`), `F`
    (C + eps as Float64, when eps is given), `rows` and `cols` (Int64
    sums), `pairs` (the 2 x 2 pair confusion counts, Int64) and `ent` (the
    row and column entropies, Float64)."""

    __slots__ = ("ka", "kb", "C", "F", "rows", "cols", "pairs", "ent")


def _contingency(a, b, ca, cb, numeric_mode, eps=None):
    n, ka, kb = len(a), len(ca), len(cb)
    if ka * kb > 16777216:
        raise ValueError("mojolearn metrics: the contingency matrix exceeds 2^24 cells")
    prog = _Prog()
    A = prog.put_i32(_codes(a, ca))
    B = prog.put_i32(_codes(b, cb))
    key = prog.scratch(n)
    kk = max(ka, kb)
    prog.stage("pair_key", n, A, B, key, kk, 0)
    m = kk * kk
    off, _ = _group(prog, key, n, m)
    _execute(prog, numeric_mode)
    o = prog.words(off, m + 1, "i")
    r = _Contingency()
    r.ka, r.kb = ka, kb
    r.C = empty((ka, kb), "<i8")
    r.F = empty((ka, kb), "<f8") if eps is not None else None
    r.rows, r.cols = empty((ka,), "<i8"), empty((kb,), "<i8")
    r.pairs, r.ent = empty((2, 2), "<i8"), empty((2,), "<f8")
    _binding(numeric_mode).x_metrics_contingency_stats(
        o.buffer_info()[0], (ka, kb, kk), float(eps) if eps is not None else 0.0,
        (r.C._addr, r.F._addr if r.F is not None else 0, r.rows._addr, r.cols._addr, r.pairs._addr,
         r.ent._addr))
    return r


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
    r = _contingency(a, b, ca, cb, numeric_mode, eps=eps)
    if eps is not None:
        return r.F
    code = {"int64": "<i8", "int32": "<i4", "float64": "<f8", "float32": "<f4"}.get(
        dtype if isinstance(dtype, str) else getattr(dtype, "__name__", str(dtype)), "<i8")
    return r.C if code == "<i8" else r.C.astype(code)


def pair_confusion_matrix(labels_true, labels_pred, *, numeric_mode=None):
    """scikit-learn 1.9 `pair_confusion_matrix`: the 2 x 2 Int64 pair counts
    from the exact contingency matrix (exact Int64 in the binding: every
    pair count is below 2^62)."""
    a, b, ca, cb = _clusterings(labels_true, labels_pred, "pair_confusion_matrix")
    return _contingency(a, b, ca, cb, numeric_mode).pairs


def _mi_from_contingency(r, numeric_mode=None):
    """MI of the counts in nats (x_metrics/epilogue.mojo mi_contingency, lane
    py-misc-metrics; the Python fallback is gone, lane pyglue-sweep: the
    counts of a clustering are never negative and total below 2^31)."""
    return float(_binding(numeric_mode).x_metrics_mi_contingency(r.C._addr, r.ka, r.kb))


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
    r = _contingency(a, b, ca, cb, numeric_mode)
    mi = _mi_from_contingency(r, numeric_mode)
    if mi == 0:
        return 0.0
    ht, hp = r.ent[0], r.ent[1]
    return float(mi / _generalized_average(ht, hp, average_method))


def _expected_mi(a_counts, b_counts, n, numeric_mode=None):
    """E[MI] under the permutation model (Vinh, Epps and Bailey 2010), the sum
    scikit-learn's `expected_mutual_information` evaluates through gammaln.
    Here each hypergeometric pmf is built by its ratio recurrence from the
    mode and normalized by its own sum over the full support: no difference
    of large log-gamma values, portable binary64 log / exp only."""
    if a_counts.size == 1 or b_counts.size == 1:
        return 0.0
    # the walks and terms in the binding's host binary64, the log the C of
    # mojolearn._portable_math.log (x_metrics/epilogue.mojo expected_mi; lane
    # metrics-apple2). The Python walk is gone (lane apple-fast-py2mojo-core):
    # every install's binding carries it, and its refusals (n outside
    # [1, 2**31), a non-finite term) cannot occur for a clustering's counts.
    return float(_binding(numeric_mode).x_metrics_expected_mi(a_counts._addr, a_counts.size, b_counts._addr,
                                                              b_counts.size, n))


def adjusted_mutual_info_score(labels_true, labels_pred, *, average_method="arithmetic", numeric_mode=None):
    """scikit-learn 1.9 `adjusted_mutual_info_score`: (MI - E[MI]) /
    (mean(H) - E[MI]) with scikit-learn's epsilon guards."""
    _generalized_average(1.0, 1.0, average_method)
    a, b, ca, cb = _clusterings(labels_true, labels_pred, "adjusted_mutual_info_score")
    if len(ca) == len(cb) == 1 or len(ca) == len(cb) == 0:
        return 1.0
    if len(ca) == 1 or len(cb) == 1:
        return 0.0
    r = _contingency(a, b, ca, cb, numeric_mode)
    n = len(a)
    mi = _mi_from_contingency(r, numeric_mode)
    emi = _expected_mi(r.rows, r.cols, n, numeric_mode)
    norm = _generalized_average(r.ent[0], r.ent[1], average_method)
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


class _Cents:
    """lane py-misc-metrics: the centroid program left in place for the
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
    """(_Cents, the centroid entry, the score entry): every install's
    binding carries both (the Python centroid fallback is gone, lane
    pyglue-sweep: a present cluster's count is never 0)."""
    return (_Cents(Xa, codes, k, numeric_mode), _epilogue("x_metrics_centroids", numeric_mode),
            _epilogue(name, numeric_mode))


def _row_dists_f32(Xa, codes, C32, k, root, numeric_mode):
    """Per-cluster sums of the row distances to the Float32 centroid words."""
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


def calinski_harabasz_score(X, labels, *, numeric_mode=None):
    """scikit-learn 1.9 `calinski_harabasz_score`: the between- over the
    within-cluster dispersion, scaled by (n - k) / (k - 1)
    (x_metrics/epilogue.mojo centroids_f32 and ch_extra, lane
    py-misc-metrics)."""
    Xa, codes, k = _cluster_inputs(X, labels, "calinski_harabasz_score")
    n, d = Xa.shape
    cen, cfn, sfn = _cluster_native(Xa, codes, k, numeric_mode, "x_metrics_ch_extra")
    C32 = cen.f32(cfn)
    extra = float(sfn(*cen.args(), cen.gsum, cen.q.buffer_info()[0], k, d, n))
    intra = _fsum(_row_dists_f32(Xa, codes, C32, k, False, numeric_mode))
    return float(1.0 if intra == 0.0 else extra * (n - k) / (intra * (k - 1.0)))


def davies_bouldin_score(X, labels, *, numeric_mode=None):
    """scikit-learn 1.9 `davies_bouldin_score`: the mean over clusters of the
    worst (s_i + s_j) / d(c_i, c_j) (x_metrics/epilogue.mojo centroids_f32
    and db_score, lane py-misc-metrics)."""
    Xa, codes, k = _cluster_inputs(X, labels, "davies_bouldin_score")
    n, d = Xa.shape
    cen, cfn, sfn = _cluster_native(Xa, codes, k, numeric_mode, "x_metrics_db_score")
    C32 = cen.f32(cfn)
    per = array.array("d", _row_dists_f32(Xa, codes, C32, k, True, numeric_mode))
    return float(sfn(*cen.args(), cen.q.buffer_info()[0], k, d, per.buffer_info()[0]))


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
        for n in sizes:  # glue: one output slot per requested draw
            outs.append((prog.want(self.permute_stage(prog, n), n), n))
        if not outs:
            return []
        _execute(prog, numeric_mode)
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
        _execute(prog, numeric_mode)
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
    _execute(prog, numeric_mode)
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
