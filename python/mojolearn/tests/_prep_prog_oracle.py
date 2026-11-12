# SPDX-License-Identifier: Apache-2.0
"""The former Python x_prep program builder and its staging helpers, kept as
the TEST ORACLE of the native prep owners (lane pyout-small-families,
2026-11-11).

Every prep public method is one native call: the native owners build the
program and the shared runner runs it (bindings/prep_classify_prog_door.mojo
`ClsProg`, bindings/prep_encode_native.mojo `_category_block` / `_codes`,
bindings/prep_select_native.mojo, bindings/prep_labels_native.mojo,
bindings/arena_prog_native_door.mojo). These Python copies had no product caller,
only the door tests that compare the native bits against them, so they left
`mojolearn/_expansion_prep.py` for here (tests never ship in the wheel).

Names not defined here resolve to `mojolearn._expansion_prep` (module
`__getattr__`), so a test can read every oracle name from this one module.
"""
import array
import ctypes
import os

from mojolearn import _arena_io
from mojolearn import _expansion_prep as _ep
from mojolearn._array import Array
from mojolearn._buffer import as_f32_c
from mojolearn._expansion_prep import (
    _IDN_CLASS_ONEPASS, _IDN_STATS_BLOCKED, _OPS, _PROG_CALL, _PROG_CONTEXT, _Scratch, _XB, _blocked, _idn_fam,
    _mode, _optional_prep_entry, _prep_binding)
from mojolearn._expansion_prep import _Prog as _ProgRecord


def __getattr__(name):
    return getattr(_ep, name)


_PARAMS = 14


_NONE = -1


def _native_helper(key):
    """The base binding's host helper `key` (`_buffer._native`)."""
    from ._buffer import _native
    return _native(key)


def _p2m_chunks(n, per=1):
    """(rows a chunk, chunks) for the chunk-parallel p2m units: at most
    _II_CH chunks, and at most 2**24 count words when each chunk keeps `per`."""
    ch = max(_II_CH, -(-n // _II_CH), -(-n * per // 2 ** 24), 1)
    return ch, max(1, -(-n // ch))


#: the smallest output (words) that `_Prog.output` keeps out of the arena
_OUT_MIN_WORDS = 2 ** 27


#: IterativeImputer(estimator=...): the most chunks of its row selection
#: (`ii_rcount` / `ii_rwrite`), so `uniq_scan` folds at most this many counts
_II_CH = 1024


#: lane prep-apple3: host-side changes, each with a name. A name in
#: _R3_DEFAULT is on; MOJOLEARN_XPREP_R3_ON / MOJOLEARN_XPREP_R3_OFF (names
#: joined by "+" or ",", or "all") switch names on and off for an A/B arm. None
#: of them moves a bit: they change where words live and which stages run.
#: All five are default since request 1790627886703 (M3 Ultra: every FAST and
#: IDENTICAL digest equal with them on, every case as fast or faster).
#:   imputer_nosort  SimpleImputer sorts only for median / most_frequent
#:   mapped          a large host arena or output is an anonymous mapping
#:                   (zero pages arrive when first written; nothing is
#:                   touched to allocate it)
#:   view            a large read is a view of the mapped block, not a copy
#:   work            blocks a stage writes whole and Python never reads
#:                   (the sorted columns, LDA's centered rows) are device
#:                   scratch
#:   te_arrays       TargetEncoder hands targets and folds over as arrays
_R3_NAMES = ("imputer_nosort", "mapped", "view", "work", "te_arrays")


_R3_DEFAULT = ("imputer_nosort", "mapped", "view", "work", "te_arrays")


#: the smallest read that is a view (the smallest mapped block, 2**18 words,
#: is bindings/arena_prog_native_door.mojo PREP_MAP_MIN_WORDS)
_VIEW_MIN_WORDS = 2 ** 20


def _col_stats(pr, xo, n, d, out, var=True):
    """col_stats' six rows of d into `out` (count, mean, variance, min, max,
    max |x|), in the blocked order when `_blocked()`; var=False leaves the
    variance row unwritten (its consumers here read the other rows)."""
    if not _blocked():
        pr.stage("col_stats", d, xo, n, d, out)
        return
    nb = (n + _XB - 1) // _XB
    part = pr.work(5 * nb * d)
    pr.stage("colb_part", nb * d, xo, n, d, part, nb)
    pr.stage("colb_fold", d, part, nb, d, out)
    if var:
        ss = pr.work(nb * d)
        pr.stage("colb_ss", nb * d, xo, n, d, out, ss, nb)
        pr.stage("colb_var", d, ss, nb, d, out)


def _r3(name):
    """Whether the lane prep-apple3 change `name` is on."""
    def names(var):
        v = [t.strip() for t in os.environ.get(var, "").replace("+", ",").split(",") if t.strip()]  # glue: parses an env var name list
        return set(_R3_NAMES) if "all" in v else set(v)
    off = names("MOJOLEARN_XPREP_R3_OFF")
    if name in off:
        return False
    return name in _R3_DEFAULT or name in names("MOJOLEARN_XPREP_R3_ON")


def _take(store, off, n, shape, code):
    """The n words of `store` at off as an Array of 4-byte `code` items
    ("f" or "i"): the whole array.array itself, a view of a large mapped
    block (`view`), else a copy."""
    dtype = "<f4" if code == "f" else "<i4"
    if n <= 0:
        return Array._owned(array.array(code), shape, dtype, "C")
    if isinstance(store, memoryview):
        part = store[off:off + n]
        if part.format != code:
            part = part.cast("B").cast(code)
        if n >= _VIEW_MIN_WORDS and _r3("view"):
            return Array._view_of(Array.from_buffer(part), shape, "C")
        out = array.array(code)
        out.frombytes(part.cast("B"))
        return Array._owned(out, shape, dtype, "C")
    if store.typecode == code:
        if off == 0 and n == len(store):
            return Array._owned(store, shape, dtype, "C")
        return Array._owned(store[off:off + n], shape, dtype, "C")
    out = array.array(code)
    out.frombytes(store[off:off + n].tobytes())
    return Array._owned(out, shape, dtype, "C")


class _Prog:
    """One program: an arena layout, the inputs copied into it, and stages."""

    def __init__(self):
        self.size = 0
        self.scratch_size = 0
        self.out_size = None
        self._inputs = []
        self._inout = []
        self._stages = []
        self.arena = None
        self._out = None
        self._out_at = None
        self._out_code = "f"
        #: callables run(self) right after the device returns (lane fam2-prep-metrics:
        #: refusals a program decides on the device, e.g. `_stage_partial_codes`)
        self._after = []

    def alloc(self, n):
        off = self.size
        self.size += max(int(n), 0)
        return off

    def scratch(self, n):
        """n device-only words that a stage writes before any stage reads them."""
        off = self.scratch_size
        self.scratch_size += max(int(n), 0)
        return _Scratch(off)

    def work(self, n):
        """n words that ONE stage writes whole before any stage reads them and
        that Python never reads (lane prep-apple3, `work`): device scratch,
        so they neither cross the bus nor touch host memory. Arena words
        when the change is off."""
        return self.scratch(n) if _r3("work") else self.alloc(n)

    def output(self, n, code="f"):
        """The program's output: n words that arrive zeroed (as `alloc`'s),
        read back only through `get` (code "f") or `get_i32` (code "i"), one
        per program. Below _OUT_MIN_WORDS it is plain arena words: measured
        on the M4 Pro, the region only pays for itself on very large outputs
        (taxi OneHotEncoder, 566M words: 1.68 -> 1.50 s; HIGGS, 88M words:
        0.52 -> 0.58 s)."""
        if int(n) < _OUT_MIN_WORDS:
            return self.alloc(n)
        if self.out_size is not None:
            raise ValueError("x_prep: one output region per program")
        self.out_size = max(int(n), 0)
        self._out_code = code
        return _Scratch(0, "o")

    def put(self, arr, inout=False):
        """A float32 C-contiguous Array (or anything as_f32_c takes) -> offset.
        inout (lane prep-apple3): a stage writes these words and Python reads
        them after the run, so they come back from the device (a plain input
        never does, and a read of one is refused). A `_arena_io.DeviceRows`
        (a fold's rows, lane cpu4-misc) is kept as it is: `run` gathers it
        on the device, or copies its host rows when the binding cannot."""
        if isinstance(arr, _arena_io.DeviceRows):
            if inout or arr.dtype != "<f4":
                arr = arr.materialize()
        if not (isinstance(arr, (Array, _arena_io.DeviceRows)) and arr.dtype == "<f4"
                and (isinstance(arr, _arena_io.DeviceRows) or arr._has_order("C"))):
            arr = as_f32_c(arr, ndim=None, name="input")[0]
        off = self.alloc(arr.size)
        self._inputs.append((off, arr, "f"))
        if inout and arr.size:
            self._inout.append((off, off + arr.size))
        return off

    def put_list(self, values, inout=False):
        flat = [float(v) for v in values] or [0.0]  # glue: packs a caller parameter list into the program buffer
        return self.put(Array._from_flat(flat, (len(flat),), "<f4"), inout=inout)

    def put_scalar(self, value):
        return self.put_list([value])

    def put_codes(self, codes):
        """int32 codes -> offset of their float values (an i2f stage)."""
        if not (isinstance(codes, Array) and codes.dtype == "<i4"):
            codes = Array.from_list([int(c) for c in codes], "<i4")  # glue: packs caller code list into program buffer
        bits = self.alloc(codes.size)
        self._inputs.append((bits, codes, "i"))
        out = self.alloc(codes.size)
        self.stage("i2f", codes.size, bits, out)
        return out

    def put_words(self, words):
        """An int32 Array's words as they are (no conversion stage) -> offset."""
        off = self.alloc(words.size)
        self._inputs.append((off, words, "i"))
        return off

    def put_ints(self, values):
        """int32 words (read by `ldi`) -> offset."""
        codes = Array.from_list([int(v) for v in values] or [0], "<i4")  # glue: packs caller int parameters into program buffer
        off = self.alloc(codes.size)
        self._inputs.append((off, codes, "i"))
        return off

    def stage(self, op, total, *params):
        if len(params) > _PARAMS:
            raise ValueError("x_prep: too many stage parameters")
        self._stages.append([_OPS[op], int(total)] + [v if isinstance(v, _Scratch) else int(v) for v in params]  # glue: builds one stage parameter record
                            + [0] * (_PARAMS - len(params)))

    def run(self, mode):
        """Run the program on the x_prep binding (GPU, or the host binding on
        a CPU-only install): one native call that owns the arena layout and
        stores, the input copies, resident, direct and device-gathered
        inputs, the range descriptors and the run
        (bindings/arena_prog_native_door.mojo). The program's refusals
        (`_after`) run before its first result is read."""
        # The runner admits only the product record class; the record shares
        # this builder's attribute dict, so the fields it fills land here.
        record = object.__new__(_ProgRecord)
        record.__dict__ = self.__dict__
        return _PROG_CALL(record, "x_prep_run", (mode,), _PROG_CONTEXT)

    def _refuse(self):
        """Run the program's post-run refusals once, before any result is
        read (lane fam2-prep-metrics: refusals a program decides on the
        device, e.g. `_stage_partial_codes`)."""
        hooks, self._after = self._after, []
        for fn in hooks:  # glue: the program's post-run refusal hooks
            fn(self)

    def _check(self, off, n):
        """An input's words never come back from the device (the ranges
        runner, lane py-shared): a read of one is refused on every backend,
        so a program that reads an input after a stage wrote it fails loudly
        instead of reading the host's stale copy."""
        for lo, hi in getattr(self, "_in_spans", ()):  # glue: walks the program span records
            if off < hi and lo < off + n:
                raise AssertionError(f"x_prep: arena [{off}, {off + n}) was read but is an input, "
                                     "which never comes back")

    def _read(self, off, shape, code):
        if self._after:
            self._refuse()
        shape = tuple(shape) if isinstance(shape, (tuple, list)) else (int(shape),)
        n = 1
        for s in shape:  # glue: walks the shape tuple entries
            n *= s
        if isinstance(off, _Scratch):
            if off.kind != "o":
                raise ValueError("x_prep: scratch words never reach the host")
            if self._out is not None:
                return _take(self._out, off.off, n, shape, code)
            off = self._out_at + off.off
        else:
            self._check(off, n)
        return _take(self.arena, off, n, shape, code)

    def get(self, off, shape):
        return self._read(off, shape, "f")

    def get_i32(self, off, shape):
        return self._read(off, shape, "i")

    def get_f64(self, off, shape):
        """An (rows, cols) block of float64 values written as word pairs (low, high) by `q64_softmax`:
        the 2 * prod(shape) words reinterpreted as bytes (glue: no arithmetic)."""
        rows, cols = shape
        words = self._read(off, (2 * rows * cols,), "i")
        out = array.array("d")
        out.frombytes(words.tobytes())
        return Array._owned(out, shape, "<f8", "C")

    def values(self, off, n):
        """Python floats of n arena entries (for integer bookkeeping)."""
        if self._after:
            self._refuse()
        self._check(off, n)
        return self.arena[off:off + n].tolist()


#: binding -> whether it exports `x_prep_fast_unique` (FAST + Apple default,
#: bindings/_mojolearn_x_prep.mojo X_PREP_FAST_UNIQUE; built with
#: -D MOJOLEARN_X_PREP_FAST_UNIQUE_OFF it does not), probed once per binding
_FAST_UNIQUE = {}


def _fast_on(name, mode):
    """Whether the switch `name` is on for the FAST tier (`mode` is the
    estimator's numeric mode, `_backend.default_mode`). UNIQUE is the
    binding's comptime default (a cached probe). Off, or on another tier,
    every route below is the old one."""
    if mode != "fast":
        return False
    if name == "UNIQUE":
        binding = _prep_binding(mode)
        key = id(binding)
        on = _FAST_UNIQUE.get(key)
        if on is None:
            on = _FAST_UNIQUE[key] = _optional_prep_entry(binding, "x_prep_fast_unique") is not None
        return on
    return False


def _x2d(X, name="X"):
    if isinstance(X, _arena_io.DeviceRows):
        # a fold's rows on the device (lane cpu4-misc): float32 2-D rows of
        # a C-contiguous base stay a handle for `_Prog.put`; anything else
        # takes its host rows and the usual conversion
        if X.dtype == "<f4" and X.ndim == 2:
            arr = X
        else:
            arr = as_f32_c(X.materialize(), ndim=2, name=name)[0]
    else:
        arr = as_f32_c(X, ndim=2, name=name)[0]
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"mojolearn: {name} must be a nonempty two-dimensional array")
    if arr.size > 2 ** 31 - 1:
        raise ValueError(f"mojolearn: {name} exceeds the native Int32 indexing bound")
    return arr


#: lane/apple-fast-gap-cls2 (x_prep/cat_cls2.mojo): the packed distinct
#: values' host region (words), the presence flags per column, flags per chunk
_CAT_CAP = 1 << 16


_CAT_R = 4096


_CAT_CH = 64


#: binding -> its `x_prep_cls2_cat` bits (0 when it has none), probed once
_CLS2_CAT = {}


def _cls2_cat(mode):
    """lane/apple-fast-gap-cls2: bit 1 PACK, bit 2 PRESENT on the FAST
    binding built with them (x_prep/cat_cls2.mojo), else 0."""
    if mode != "fast":
        return 0
    binding = _prep_binding(mode)
    key = id(binding)
    v = _CLS2_CAT.get(key)
    if v is None:
        fn = _optional_prep_entry(binding, "x_prep_cls2_cat")
        v = _CLS2_CAT[key] = int(fn()) if fn is not None else 0
    return v


def _fit_categories_cls2(mode, arr, bits):
    """`_fit_categories`' lists with the distinct values packed into a small
    host region (PACK) and, with PRESENT, found by presence flags instead of
    a sort. None (nothing kept) when a column is not small non-negative
    integers under PRESENT, or the values do not fit the region: the caller
    runs main's program."""
    n, d = arr.shape
    pr = _Prog()
    xo = pr.put(arr)
    co = pr.alloc(d)
    cap = min(n * d, _CAT_CAP)
    pk = pr.alloc(cap)
    bad = None
    if bits & 2:
        R, ch = _CAT_R, _CAT_CH
        nch = -(-R // ch)
        bad = pr.alloc(d)
        fl = pr.work(d * R)
        uo = pr.work(d * R)
        pr.stage("cat_zero", d * R, fl)
        pr.stage("cat_present", n * d, xo, n, d, R, fl, bad)
        for c in range(d):  # glue: stages one device scan per column (d-sized: feature count)
            cnt, off = pr.work(nch), pr.work(nch)
            pr.stage("pres_count", nch, fl + c * R, R, ch, cnt)
            pr.stage("uniq_scan", 1, cnt, nch, off, co + c)
            pr.stage("pres_write", nch, fl + c * R, R, ch, off, uo + c * R)
        pr.stage("cat_pack", d * R, uo, R, d, co, cap, pk)
    else:
        so = pr.work(n * d)
        uo = pr.work(n * d)
        pr.stage("sort_cols", d, xo, n, d, so, 1)
        ch = _label_chunk(n)
        nch = -(-n // ch)
        for c in range(d):  # glue: stages one device scan per column (d-sized: feature count)
            cnt, off = pr.work(nch), pr.work(nch)
            pr.stage("uniq_count", nch, so + c * n, n, ch, cnt)
            pr.stage("uniq_scan", 1, cnt, nch, off, co + c)
            pr.stage("uniq_write", nch, so + c * n, n, ch, off, uo + c * n)
        pr.stage("cat_pack", n * d, uo, n, d, co, cap, pk)
    pr.run(mode)
    if bad is not None and any(v != 0 for v in pr.get_i32(bad, (d,)).tolist()):  # glue: reads the per-column overflow flags the device set (d-sized: feature count)
        return None
    counts = [int(v) for v in pr.values(co, d)]  # glue: reads per-column category counts for slicing (d-sized: feature count)
    if sum(counts) > cap:  # glue: capacity check before slicing outputs (counts-sized: per-column category counts)
        return None
    out, o = [], 0
    for c in range(d):  # glue: slices the per-column category outputs (d-sized: feature count)
        out.append(pr.get(pk + o, counts[c]))
        o += counts[c]
    return out


def _fit_categories(mode, arr):
    """Per column, the sorted distinct values (-0.0 folded into 0.0), on the
    device: a sort per column and a run scan."""
    n, d = arr.shape
    cls2 = _cls2_cat(mode)
    if cls2 & 1 and _fast_on("UNIQUE", mode):
        got = _fit_categories_cls2(mode, arr, cls2)
        if got is not None:
            return got
    pr = _Prog()
    xo = pr.put(arr)
    so = pr.work(n * d)
    uo = pr.alloc(n * d)
    co = pr.alloc(d)
    pr.stage("sort_cols", d, xo, n, d, so, 1)
    if _fast_on("UNIQUE", mode):
        # lane/apple-fast-prep (2026-10-02), X_PREP_FAST_UNIQUE (FAST + Apple
        # default since the M3 A/B; -D MOJOLEARN_X_PREP_FAST_UNIQUE_OFF):
        # `unique_cols` (x_prep/prims.mojo unique_cols_unit) is ONE thread per
        # column walking every sorted row (taxi onehot / ordinal fit: 5
        # threads over 1M rows each, after a parallel sort). Here each column
        # takes the labels' chunked run scan (x_prep/labels.mojo uniq_count /
        # uniq_scan / uniq_write, ~sqrt(n) rows a thread): the same words
        # (`key` equality on the sorted column, the first word of every run,
        # the count as a float at co[c]).
        ch = _label_chunk(n)
        nch = -(-n // ch)
        for c in range(d):  # glue: stages one device scan per column (d-sized: feature count)
            cnt, off = pr.work(nch), pr.work(nch)
            pr.stage("uniq_count", nch, so + c * n, n, ch, cnt)
            pr.stage("uniq_scan", 1, cnt, nch, off, co + c)
            pr.stage("uniq_write", nch, so + c * n, n, ch, off, uo + c * n)
    else:
        pr.stage("unique_cols", d, so, n, d, uo, co)
    pr.run(mode)
    counts = [int(v) for v in pr.values(co, d)]  # glue: reads per-column category counts for slicing (d-sized: feature count)
    return [pr.get(uo + c * n, counts[c]) for c in range(d)]  # glue: slices the per-column category outputs (d-sized: feature count)


def _category_block(pr, categories):
    """Every column's categories in one (d, kmax) block. Returns (offset, kmax)."""
    kmax = max(c.size for c in categories)  # glue: largest category count for the program layout (categories-sized: per-column category arrays)
    block = [0.0] * (len(categories) * kmax)
    for j, cats in enumerate(categories):  # glue: stages per-column category tables (categories-sized: per-column category arrays)
        block[j * kmax:j * kmax + cats.size] = cats.tolist()
    return pr.put_list(block), kmax


def _codes(pr, arr, categories, *, device_only=False):
    """Stages that write each element's category index (or -1) and each
    column's unknown count. Returns (codes offset, unknown-count offset)."""
    n, d = arr.shape
    xo = pr.put(arr)
    uo, kmax = _category_block(pr, categories)
    co = pr.put_list([c.size for c in categories])  # glue: packs per-column category sizes (categories-sized: per-column category arrays)
    codes = pr.scratch(n * d) if device_only else pr.alloc(n * d)
    pr.stage("lookup", n * d, xo, n, d, uo, kmax, co, codes)
    if device_only:
        # TargetEncoder consumes the codes only; unknown counts are unused.
        return codes, None
    neg = pr.alloc(d)
    pr.stage("count_neg", d, codes, n, d, neg)
    return codes, neg


def _stage_folds(pr, n, n_folds, seed, shuffle, codes=_NONE, n_classes=0):
    """Stages TargetEncoder's cross-fit fold of every row (lane cpu2-l3-prep:
    on the device, every tier; the host entries x_prep_kfold_folds /
    x_prep_strat_folds walked a serial Fisher-Yates). Without `codes` (the
    float class codes' offset): KFold, c2_kfold. With them: StratifiedKFold,
    the rows grouped by class (p2m_ccount .. p2m_cwrite), each class's start
    in first-seen order (c2_strat_meta), its rows' folds (c2_strat_fold).
    Unshuffled, the reference's folds; shuffled, a seed-keyed permutation of
    the positions (of each class's fold block), integers only. Returns (the
    folds' float codes offset, the refusal flag offset or None: 1.0 when
    every class has fewer rows than n_folds)."""
    s = int(seed) & 0xFFFFFFFFFFFFFFFF
    fo = pr.alloc(max(n, 1))
    if codes == _NONE:
        pr.stage("c2_kfold", n, _seed_words(pr, s), n, n_folds, 1 if shuffle else 0, fo)
        return fo, None
    K = n_classes
    ch, nch = _p2m_chunks(n, K)
    cnt, tot = pr.alloc(nch * K), pr.alloc(K)
    pr.stage("p2m_ccount", nch, codes, n, K, ch, cnt)
    pr.stage("p2m_cscan", K, cnt, nch, K, tot)
    start, ro = pr.alloc(K), pr.alloc(n)
    pr.stage("p2m_cstart", 1, tot, K, start, _NONE)
    pr.stage("p2m_cwrite", nch, codes, n, K, ch, cnt, start, ro)
    meta, flag = pr.alloc(2 * K), pr.alloc(1)
    pr.stage("c2_strat_meta", K, start, tot, ro, K, meta)
    pr.stage("c2_strat_flag", 1, tot, K, n_folds, flag)
    pr.stage("c2_strat_fold", n, ro, start, tot, codes, meta, n_folds, _seed_words(pr, s), 1 if shuffle else 0, fo)
    return fo, flag


def _refuse_folds(pr, flag, n_folds):
    """After the run: the reference's refusal when every class is smaller
    than n_folds (one word read back)."""
    if flag is None:
        return

    def refuse(prog, flag=flag):
        if prog.values(flag, 1)[0] != 0:
            raise ValueError(f"mojolearn: n_splits={n_folds} cannot be greater than the number of members in "
                             "each class")
    pr._after.append(refuse)


def _native_folds(n, n_folds, seed, shuffle, codes=None, n_classes=0, as_array=False):
    """The folds `_stage_folds` gives TargetEncoder's fit, as int32 words
    (one program; `codes` a list or an int32 Array of class codes)."""
    pr = _Prog()
    co = pr.put_codes(codes) if codes is not None else _NONE
    fo, flag = _stage_folds(pr, n, n_folds, seed, shuffle, co, n_classes)
    _refuse_folds(pr, flag, n_folds)
    out = pr.alloc(max(n, 1))
    pr.stage("f2i", n, fo, out)
    pr.run(_mode())
    got = pr.get_i32(out, n)
    return got if as_array else got.tolist()


def _seed_words(pr, seed):
    """A uint64 seed as two int32 words (low, high) -> offset."""
    lo, hi = seed & 0xFFFFFFFF, (seed >> 32) & 0xFFFFFFFF
    return pr.put_ints([lo - (1 << 32) if lo >= 1 << 31 else lo, hi - (1 << 32) if hi >= 1 << 31 else hi])


#: `_cls`: the most partial words (blocks x classes x columns) a blocked
#: class_stats stage keeps; past it the serial stage (a function of the shape
#: only, so the device and the host column agree)
_CLS_BLOCK_WORDS = 2 ** 26


def _cs(pr, mode, xo, n, d, out, var=True):
    """A col_stats stage: x_prep/blocked.mojo's blocked order under
    IDN_STATS_BLOCKED (IDENTICAL; `_col_stats`), else one thread per column
    over every row. var=False: the caller reads no variance (the blocked
    order then skips that pass)."""
    if _blocked() and _idn_fam(mode) & _IDN_STATS_BLOCKED:
        _col_stats(pr, xo, n, d, out, var=var)
    else:
        pr.stage("col_stats", d, xo, n, d, out)


def _cls(pr, mode, xo, n, d, yo, K, cnt, mean, var, sums, *tail):
    """An unweighted class_stats stage: x_prep/blocked.mojo's blocked order
    under IDN_STATS_BLOCKED (IDENTICAL; `_class_stats`) while its partial
    tables stay within _CLS_BLOCK_WORDS, else one thread per (class, column)
    over every row (`tail`: the serial stage's trailing parameters)."""
    nb = (n + _XB - 1) // _XB
    if _blocked() and _idn_fam(mode) & _IDN_STATS_BLOCKED and nb * K * d <= _CLS_BLOCK_WORDS:
        _class_stats(pr, None, K * d, xo, n, d, yo, K, cnt, mean, var, sums, mode=mode)
    else:
        pr.stage("class_stats", K * d, xo, n, d, yo, K, cnt, mean, var, sums, *tail)


# Tried 2026-10-08 (C61 single-pass class statistics, csbm_part / csbm_fold / csbm_pool, x_prep ops 190-192;
# MOJOLEARN_CLASSICAL_C61_NB_CLASS_STATS=1|2 and MOJOLEARN_CLASSICAL_C61_DA_CLASS_STATS, run ge123e6f9): slower on
# both vendors, quality SAME. gaussian-nb arm 1 NV/AMD 1.58x/1.05x istella, 1.42x/1.08x taxi; arm 2 1.57x/1.02x,
# 1.37x/1.03x; lda-clf 1.38x/1.04x, 1.54x/1.11x. Deleted with the defines refused; recoverable at main bc10b8b56.
def _class_stats(pr, wo, total, xo, n, d, yo, K, cnt, mean, var, sums, mode=None):
    """class_stats, or its weighted form when a sample_weight offset is given;
    in x_prep/blocked.mojo's blocked order when `_blocked()` (offsets
    _NONE are not written). mode given and IDN_CLASS_ONEPASS (IDENTICAL):
    one unit per (block, column) walks X once for every class (csb1_part /
    csb1_ss; the same words as csb_part / csb_ss)."""
    if _blocked():
        nb = (n + _XB - 1) // _XB
        w = _NONE if wo is None else wo
        ps, pc, cn = pr.work(nb * K * d), pr.work(nb * K * d), pr.work(K * d)
        if var != _NONE and mean == _NONE:
            mean = pr.work(K * d)
        onepass = mode is not None and bool(_idn_fam(mode) & _IDN_CLASS_ONEPASS)
        if onepass:
            pr.stage("csb1_part", nb * d, xo, n, d, yo, K, ps, pc, nb, w, _NONE, _NONE)
        else:
            pr.stage("csb_part", nb * K * d, xo, n, d, yo, K, ps, pc, nb, w)
        pr.stage("csb_fold", K * d, ps, pc, nb, K, d, cn, cnt, mean, sums, w)
        if var != _NONE and onepass:
            pr.stage("csb1_ss", nb * d, xo, n, d, yo, K, mean, cn, ps, nb, w)
            pr.stage("csb_var", K * d, ps, nb, K, d, cn, var)
        elif var != _NONE:
            pr.stage("csb_ss", nb * K * d, xo, n, d, yo, K, mean, cn, ps, nb, w)
            pr.stage("csb_var", K * d, ps, nb, K, d, cn, var)
        return
    if wo is None:
        pr.stage("class_stats", total, xo, n, d, yo, K, cnt, mean, var, sums)
    else:
        pr.stage("class_stats_w", total, xo, n, d, yo, K, cnt, mean, var, sums, wo)


# lane neural-pass137: a numeric label BUFFER (an ndarray, an Array) takes the
# device from its own bytes. The route it replaces built Python objects for
# every label five times over (tolist, the float32 exactness test, the float
# list, the float32 Array, the type sets) in fit and again in transform, then
# ran two programs whose unique scan and unknown count were ONE thread
# walking every row. Here: one upload of the raw words, `lab_load` (the
# float32 word the old route would have built, NaN where float32 cannot hold
# the label, which sends the call back to the old route), the device sort, a
# chunked run scan, and for fit_transform the codes in the same program.
# The old route stays for labels no numeric buffer holds (str, bool, lists).
# The label encoders' routes live in bindings/prep_labels_native.mojo; this
# buffer view remains for preprocessing.py's integer counts.
_LABEL_KIND = {"<f4": (0, 1), "<i4": (1, 1), "<u4": (2, 1), "<i8": (3, 2), "<f8": (4, 2)}


class _LabelBuf:
    __slots__ = ("kind", "words", "n", "is_float", "_src")

    def __init__(self, kind, words, n, is_float, src):
        self.kind, self.words, self.n, self.is_float, self._src = kind, words, n, is_float, src


def _label_buffer(y):
    """A numeric label vector as raw int32 words (a view, no copy, of a
    contiguous buffer), or None: the old route (lists, str or bool labels,
    other dtypes, a matrix, an empty y; lane pyglue-numeric deleted the
    MOJOLEARN_XPREP_LABELS switch)."""
    if isinstance(y, (list, tuple, str, bytes)):
        return None
    from ._buffer import Buf, _has_buffer, _materialize, _raw_store_type, typestr_of
    if not isinstance(y, Array):
        if not _has_buffer(y):
            return None
        with Buf(y, name="y") as b:
            if typestr_of(b) == "|b1":
                return None
    try:
        arr, _ = _materialize(y, "y")
    except (TypeError, ValueError):
        return None
    spec = _LABEL_KIND.get(arr.dtype)
    if spec is None or arr.size == 0 or arr.ndim < 1 or arr.size != max(arr.shape):  # glue: shape check of the label argument
        return None
    kind, wpe = spec
    nw = arr.size * wpe
    if nw > 2 ** 30:
        return None
    store = _raw_store_type("i", nw).from_address(arr._addr)
    store._keep = arr
    words = Array._owned(store, (nw,), "<i4", "C")
    return _LabelBuf(kind, words, arr.size, arr.dtype in ("<f4", "<f8"), arr)


def _label_chunk(n):
    """Rows per chunk of the run scan and the unknown count (bookkeeping
    only: every chunking writes the same words)."""
    # exact integer square root by integer Newton steps (no platform math)
    m = int(n)
    r = m
    if m > 1:
        y = (r + 1) // 2
        while y < r:
            r, y = y, (y + m // y) // 2
    return max(1024, r + 1)


def _class_words(pr, out, n, ints):
    """The n 64-bit class words a program wrote (f2_code_gather, c2_add_i64), as an int64 or
    float64 Array (one byte copy, no Python object per label)."""
    words = pr.get_i32(out, 2 * n)
    store = array.array("q" if ints else "d")
    store.frombytes(ctypes.string_at(words._addr, 8 * n))
    return Array._owned(store, (n,), "<i8" if ints else "<f8", "C")
