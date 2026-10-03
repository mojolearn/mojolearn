# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE PREP LANE'S PUBLIC DOOR.

Owned by the `prep` expansion lane. `mojolearn/__init__.py` imports this
module and makes every name in `__all__` public as `mojolearn.<name>`; a name
that is already public is refused at import. Put the classes here, or import
them here from the lane's own modules. Two optional hooks route a SAVED model
to the CPU (`mojolearn.host_model`); `_classical_host.py` merges them when it is
first imported, after the package, so both may rely on every module existing:

  CLASSICAL_HOST_BASENAMES     {"_mojolearn_x_prep": "_mojolearn_x_prep_host"}
  def classical_host_formats() -> {"mojolearn-<format>-1": {"Estimator": HostEstimator}}
                               import `_classical_host` INSIDE it and subclass
                               its `_HostBound`; never import it at module level

HOW THE CLASSES COMPUTE. Every number these estimators produce comes out of
ONE binding entry, `x_prep_run`, which runs a PROGRAM of units
(x_prep/common.mojo) over one float32 arena: the GPU binding
(`_mojolearn_x_prep`) launches a thread per unit, the host binding
(`_mojolearn_x_prep_host`, what `_backend.binding` returns on a CPU-only
install) runs the same units in a loop. Python here only lays out the arena,
lists the stages and reads results back; the only arithmetic it does is
integer bookkeeping and IEEE basic operations on scalar parameters.

EXCEPTIONS, named (python_work_audit prep; each is host float64 Python today
and none is covered by a ledger row yet): KBinsDiscretizer's cumulative sample
weights and 53-bit uniform scaling (`_draw`-driven subsample); IterativeImputer's
`_truncnorm_host` (binary64 per missing entry, sample_posterior with a user
estimator only; its normal cdf / inverse cdf are the pinned `_portable_math`
twins since lane py-bugs, DEVIATION 6902, row 252) and the O(d^2) `_abs_corr` /
`_neighbours` normalisation (n_nearest_features). IterativeImputer with a
user `estimator` otherwise runs its data-sized plumbing as x_prep units
(x_prep/iterative.mojo `ii_rcount` .. `ii_scatter`: row selection, gathers,
the clipped float32 store; the stop is `ii_rowabs` / `ii_conv`).
"""
import array
import bisect
import copy
import ctypes
import inspect
import itertools
from . import _portable_math as math
import mmap
import numbers
import operator
import os
import time

from . import _backend
from . import _portable_math as _pm
from ._array import Array
from ._buffer import as_f32_c, addr_ro
from . import _labels
from . import _arena_io
from ._labels import flatten_labels, sorted_classes, label_kind

__all__ = ["f_classif", "f_regression", "r_regression", "chi2", "mutual_info_classif", "mutual_info_regression", "RobustScaler", "MaxAbsScaler", "OrdinalEncoder", "OneHotEncoder", "TargetEncoder", "SimpleImputer", "KBinsDiscretizer",
           "GaussianNB", "MultinomialNB", "BernoulliNB",
           "LinearDiscriminantAnalysis", "QuadraticDiscriminantAnalysis",
           "QuantileTransformer", "PowerTransformer", "Normalizer", "PolynomialFeatures", "SplineTransformer", "Binarizer", "LabelEncoder", "LabelBinarizer", "MultiLabelBinarizer", "IterativeImputer", "VarianceThreshold", "SelectKBest", "RFE", "ComplementNB", "CategoricalNB"]

_BINDING = "_mojolearn_x_prep"

#: op name -> id; x_prep/units.mojo `run_unit` holds the same table.
_OPS = dict(
    sort_cols=0, col_stats=1, quantile=2, affine=3, scale_params=4, unique_cols=5, mode_cols=6,
    lookup=7, count_neg=8, onehot=9, i2f=10, f2i=11, binarize=12, matmul=13, row_softmax=14,
    row_argmax=15, class_stats=16, center_rows=17, eigh=18, where_neg=19,
    te_global=20, te_enc=21, te_apply=22, mark_missing=23, fill=24, kbins_edges=25, kbins_codes=26,
    gnb_eps=27, gnb_params=28, gnb_jll=29, class_log_prior=30, mnb_params=31, bnb_params=32, cnb_params=33, cat_params=34, cat_jll=35,
    lda_prep=36, lda_w=37, lda_stage2=38, lda_stage3=39, qda_cov=40, qda_prep=41, qda_dec=42,
    qt_apply=43, pt_fit=44, pt_apply=45, std_params=46, normalize=47, poly=48, spline_knots=49, spline_apply=50, label_binarize=51, scatter_ones=52,
    ii_mean=53, ii_gram=54, ii_sub=55, ii_br=56, ii_predict=57, ii_snapshot=58, ii_conv=59, nan_mask=60, gather_cols=61, var_ptp=62, f_classif=63, f_regression=64, chi2=65,
    mi_colscale=66, mi_noise=67, mi_cc=68, mi_cd=69, mi_reduce=70, sqsum_cols=71, log=72, robust_uv=73,
    qt_inverse=74, pt_inverse=75, block_argmax=76, ord_inverse=77, cat_gather=78, where_code=79, kbins_inverse=80,
    da_shrink=81, da_pool=82, sym_fn=83, da_intercept=84, evr=85, class_stats_w=86,
    indicator=87, code_counts=88, remap_codes=89, add_arrays=90, gnb_merge=91, cat_counts=92, cat_flp=93,
    mi_dc=94, mi_dd=95, kbins_gw=96, kbins_wq=97, kbins_wkm=98, ii_sigma=99, ii_post=100,
    scaler_stats=101, std_scale=102, nan_keep=103, pt_init=104, pt_map=105, pt_fold=106, ii_rowabs=107, te_bucket=108, pt_log=109,
    pt_spts=110, pt_smap=111, pt_sfold=112, pt_sres=113, te_gather=114,
    te_hist=115, te_hsum=116, te_hstart=117, te_hscatter=118,
    lab_load=119, uniq_count=120, uniq_scan=121, uniq_write=122, chunk_neg=123,
    colb_part=124, colb_fold=125, colb_ss=126, colb_var=127, maxabs_fold=128, csb_part=129, csb_fold=130, csb_ss=131, csb_var=132, cat_hpart=133, cat_hfold=134,
    row_ones=135, ii_rcount=136, ii_rwrite=137, ii_gather=138, ii_scatter=139,
    hcat=140, colblock=141,
    # lane apple-fast-meta (x_prep/calib.mojo): only the FAST + Apple binding built with
    # -D MOJOLEARN_CALIB_GNB_FOLDS runs them (it exports x_prep_calib_folds); nothing stages
    # them otherwise
    cal_fold_part=142, cal_fold_scan=143, cal_fold_rank=144, cal_fold_assign=145, cal_lofo_merge=146,
    cal_eps_folds=147, cal_params_folds=148, cal_jll_folds=149, cal_platt_init=150, cal_platt_setup=151,
    cal_platt_part=152, cal_platt_step=153, cal_platt_ls_part=154, cal_platt_ls_pick=155, cal_sigmoid_avg=156,
)
_PARAMS = 14
_NONE = -1
#: x_prep/transform.mojo PT_EVALS (PT_ITERS + 2) and PT_STATE
_PT_EVALS = 50
_PT_STATE = 10


def _prep_binding(mode):
    return _backend.binding("_mojolearn_x_prep", mode)


def _optional_prep_entry(binding, name):
    """Probe optional exports without hiding a missing mandatory host implementation."""
    binding.x_prep_run  # Load/validate the family before interpreting a refusal as absence.
    try:
        return getattr(binding, name)
    except (AttributeError, ImportError):
        return None


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
#: the smallest block (words) that is mapped, and the smallest read that is a view
_MAP_MIN_WORDS = 2 ** 18
_VIEW_MIN_WORDS = 2 ** 20

#: lane gap-nb-maxabs-grp (2026-10-02), two A/B switches (unset or anything
#: but "0": on):
#:   MOJOLEARN_XPREP_DIRECT   an input of at least _DIRECT_MIN_WORDS words goes
#:                            up to the device FROM ITS OWN BUFFER (a store
#:                            slot for the length of the run, copied device to
#:                            device into the arena) instead of being copied
#:                            into the host arena first (at the board's 1M x 220
#:                            that copy was 880 MB of fresh pages a call).
#:                            Where a word travels moves no bit.
#:   MOJOLEARN_XPREP_BLOCKED  naive Bayes and MaxAbsScaler fold rows in
#:                            x_prep/blocked.mojo's blocks (one unit per block,
#:                            class and column, then the partials) instead of
#:                            col_stats / class_stats / cat_counts' one thread per
#:                            column over every row. Counts, minima, maxima and
#:                            max |x| keep their words (MaxAbsScaler, the
#:                            discrete models' counts on integer data, and
#:                            CategoricalNB unweighted are unchanged); sums,
#:                            means and variances take the blocked order for
#:                            n > _XB rows (GaussianNB's theta_, var_ and
#:                            epsilon_; non-integer feature counts).
_DIRECT_MIN_WORDS = 2 ** 20
#: x_prep/blocked.mojo XB: the rows a block unit folds
_XB = 2048
#: CategoricalNB's block histograms hold at most this many words; past it
#: the histogram block doubles (a function of the shape only)
_CAT_HIST_WORDS = 2 ** 26


def _switch(name):
    return os.environ.get(name, "1").strip() != "0"


def _blocked():
    return _switch("MOJOLEARN_XPREP_BLOCKED")


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


#: lane af-ptimpute (2026-10-03): binding -> the bits of `x_prep_ptimpute_flags`
#: (bindings/_mojolearn_x_prep.mojo, x_prep/fastpt.mojo PTIMPUTE_FLAGS: 1
#: PT_COLBATCH, 2 PT_SPEC, 4 PT_FUSED_TRANSFORM, 8 SI_ONEPASS, 16 PT_FOLD_NOX),
#: probed once per binding; 0 on a build without them (every other tier and
#: vendor, and the host). The device's own comptime switches do the fusing;
#: a program only shrinks the arena blocks the device no longer touches.
_PTIMPUTE_FLAGS = {}
#: PT_SPEC: the FAST search speculates this many golden steps a round (2^3 - 1
#: = 7 candidates, x_prep/fastpt.mojo PT_MAXM); no candidates' buffer exists,
#: so `_pt_spec_depth`'s 2^28-word cap does not apply. Fixed: no env read.
_PT_FAST_SPEC = 3


def _ptimpute_flags(mode):
    if mode != "fast":
        return 0
    binding = _prep_binding(mode)
    key = id(binding)
    v = _PTIMPUTE_FLAGS.get(key)
    if v is None:
        entry = _optional_prep_entry(binding, "x_prep_ptimpute_flags")
        v = _PTIMPUTE_FLAGS[key] = int(entry()) if entry is not None else 0
    return v


def _r3(name):
    """Whether the lane prep-apple3 change `name` is on."""
    def names(var):
        v = [t.strip() for t in os.environ.get(var, "").replace("+", ",").split(",") if t.strip()]
        return set(_R3_NAMES) if "all" in v else set(v)
    off = names("MOJOLEARN_XPREP_R3_OFF")
    if name in off:
        return False
    return name in _R3_DEFAULT or name in names("MOJOLEARN_XPREP_R3_ON")


def _zero_words(n, code="f"):
    """(store, address) of n zeroed 4-byte words: an array.array, or for a
    large block (`mapped`) a memoryview of an anonymous mapping."""
    n = max(int(n), 1)
    if n >= _MAP_MIN_WORDS and _r3("mapped"):
        m = mmap.mmap(-1, 4 * n)
        return memoryview(m).cast(code), ctypes.addressof(ctypes.c_char.from_buffer(m))
    a = array.array(code, bytes(4 * n))
    return a, a.buffer_info()[0]


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


class _Scratch:
    """An offset past a program's host arena (lane prep-apple2). Kind "s":
    DEVICE-ONLY scratch, words a stage writes before any stage reads them;
    kind "o": the program's one OUTPUT region, zeroed like arena words and
    read back by `get` into its own host buffer (no copy out of the arena).
    Both resolve when the program runs (host arena size, then the scratch,
    then the output) and never cross to the device from the host; the host
    binding, or MOJOLEARN_XPREP_OUT=0 for the output, gets them as plain
    arena words."""
    __slots__ = ("off", "kind")

    def __init__(self, off, kind="s"):
        self.off = off
        self.kind = kind

    def __add__(self, k):
        return _Scratch(self.off + int(k), self.kind)


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
        never does, and a read of one is refused)."""
        if not (isinstance(arr, Array) and arr.dtype == "<f4" and arr._has_order("C")):
            arr = as_f32_c(arr, ndim=None, name="input")[0]
        off = self.alloc(arr.size)
        self._inputs.append((off, arr, "f"))
        if inout and arr.size:
            self._inout.append((off, off + arr.size))
        return off

    def put_list(self, values, inout=False):
        flat = [float(v) for v in values] or [0.0]
        return self.put(Array._from_flat(flat, (len(flat),), "<f4"), inout=inout)

    def put_scalar(self, value):
        return self.put_list([value])

    def put_codes(self, codes):
        """int32 codes -> offset of their float values (an i2f stage)."""
        if not (isinstance(codes, Array) and codes.dtype == "<i4"):
            codes = Array.from_list([int(c) for c in codes], "<i4")
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
        codes = Array.from_list([int(v) for v in values] or [0], "<i4")
        off = self.alloc(codes.size)
        self._inputs.append((off, codes, "i"))
        return off

    def stage(self, op, total, *params):
        if len(params) > _PARAMS:
            raise ValueError("x_prep: too many stage parameters")
        self._stages.append([_OPS[op], int(total)] + [v if isinstance(v, _Scratch) else int(v) for v in params]
                            + [0] * (_PARAMS - len(params)))

    def run(self, mode):
        prof = os.environ.get("MOJOLEARN_XPREP_PROFILE", "0") == "1"
        t_run = time.perf_counter() if prof else 0.0
        binding = _prep_binding(mode)
        # Resolve the mandatory entry before probing optional GPU optimizations:
        # host facades raise ImportError (not AttributeError) for absent exports.
        # A missing host binary or mandatory implementation must still fail.
        run = binding.x_prep_run
        run_out = _optional_prep_entry(binding, "x_prep_run_out")
        H, sc, on = self.size, self.scratch_size, self.out_size or 0
        has_out = run_out is not None
        dev_out = has_out and on > 0 and os.environ.get("MOJOLEARN_XPREP_OUT", "1") != "0"
        run_scratch = _optional_prep_entry(binding, "x_prep_run_scratch") if sc > 0 and not has_out else None
        dev_scratch = sc > 0 and (has_out or run_scratch is not None)
        if H + sc + on > 2 ** 31 - 1:
            raise ValueError("x_prep: the program exceeds the native Int32 indexing bound")
        # layout: the host arena, then (host) the output unless the device keeps it,
        # then the scratch, then (device) the output
        ha = H + (0 if dev_out else on)
        sbase, obase = ha, (ha + sc if dev_out else H)
        host_words = ha + (0 if dev_scratch else sc)
        arena, base = _zero_words(host_words)
        t_alloc = time.perf_counter() if prof else 0.0
        run_ranges = (_optional_prep_entry(binding, "x_prep_run_ranges")
                      if (dev_scratch or sc == 0) and _arena_io.ranges_enabled() else None)
        spans = []
        direct = None
        for off, arr, _ in self._inputs:
            if not arr.size:
                continue
            inout = (off, off + arr.size) in self._inout
            cache = (_arena_io.active_cache(binding, "x_prep", arr.size)
                     if run_ranges is not None and not inout else None)
            if (cache is None and run_ranges is not None and not inout and arr.size >= _DIRECT_MIN_WORDS
                    and _switch("MOJOLEARN_XPREP_DIRECT")):
                # MOJOLEARN_XPREP_DIRECT: up from the input's own buffer into a
                # slot freed when the run ends (no copy into the host arena)
                if direct is None and _arena_io.DeviceCache.supports(binding, "x_prep"):
                    direct = _arena_io.DeviceCache(binding, "x_prep")
                cache = direct
            if cache is not None:
                # resident (lane py-shared): copied from the store on the device;
                # the host words stay zero and `_check` refuses a read of them
                spans.append((off, off + arr.size, cache.id_of(arr)))
                continue
            ctypes.memmove(base + 4 * off, addr_ro(arr, name="input"), 4 * arr.size)
            spans.append((off, off + arr.size, -1))
        self._in_spans = [(lo, hi) for lo, hi, _ in spans if (lo, hi) not in self._inout]
        prog = array.array("i", [(v.off + (sbase if v.kind == "s" else obase)) if isinstance(v, _Scratch) else v
                                 for s in self._stages for v in s] or [0])
        nst = len(self._stages)
        self._out, self._out_at = None, obase
        t_in = time.perf_counter() if prof else 0.0
        if run_ranges is not None:
            # the shared ranges runner (lane py-shared, core/arena_io.mojo):
            # the inputs go up, the rest of the host arena starts zero on the
            # device, and everything but the inputs comes back
            ins = _arena_io.input_ranges(spans)
            outs = _arena_io.output_ranges(_arena_io.complement(ins, ha) + [list(s) for s in self._inout])
            ia, oa = _arena_io.pack_ins(ins), _arena_io.pack_outs(outs)
            out, out_addr = _zero_words(on, self._out_code) if dev_out else (None, 0)
            try:
                run_ranges(base, prog.buffer_info()[0], out_addr,
                           (ha, sc if dev_scratch else 0, on if dev_out else 0, nst),
                           (ia.buffer_info()[0], len(ins), oa.buffer_info()[0], len(outs)))
            finally:
                if direct is not None:
                    direct.close()
            self._out = out
        elif dev_out:
            out, out_addr = _zero_words(on, self._out_code)
            run_out(base, prog.buffer_info()[0], out_addr, (ha, sc if dev_scratch else 0, on, nst))
            self._out = out
        elif dev_scratch:
            if has_out:
                run_out(base, prog.buffer_info()[0], 0, (ha, sc, 0, nst))
            else:
                run_scratch(base, ha, sc, prog.buffer_info()[0], nst)
        else:
            run(base, host_words, prog.buffer_info()[0], nst)
        self.arena = arena
        if prof:
            # one line per program (the device's XPPHASE lines come in the same order)
            inv = {v: k for k, v in _OPS.items()}
            print("XPPROG ops=" + "+".join(inv.get(st[0], str(st[0])) for st in self._stages)
                  + f" arena={ha} scratch={sc} out={on} dev_out={int(bool(dev_out))} ranges={int(run_ranges is not None)}"
                  + f" alloc_s={t_alloc - t_run:.4f} inputs_s={t_in - t_alloc:.4f}"
                  + f" call_s={time.perf_counter() - t_in:.4f}", flush=True)
        return self

    def _check(self, off, n):
        """An input's words never come back from the device (the ranges
        runner, lane py-shared): a read of one is refused on every backend,
        so a program that reads an input after a stage wrote it fails loudly
        instead of reading the host's stale copy."""
        for lo, hi in getattr(self, "_in_spans", ()):
            if off < hi and lo < off + n:
                raise AssertionError(f"x_prep: arena [{off}, {off + n}) was read but is an input, "
                                     "which never comes back")

    def _read(self, off, shape, code):
        shape = tuple(shape) if isinstance(shape, (tuple, list)) else (int(shape),)
        n = 1
        for s in shape:
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

    def values(self, off, n):
        """Python floats of n arena entries (for integer bookkeeping)."""
        self._check(off, n)
        return self.arena[off:off + n].tolist()


def _mode():
    return _backend.default_mode()


#: lane/apple-fast-prep (2026-10-02): the A/B switch
#: MOJOLEARN_X_PREP_FAST_NONEG=1 (opt-in; measured noise), read once at
#: import (never on a fit / transform path).
_X_PREP_FAST = frozenset(
    name for name in ("NONEG",) if os.environ.get("MOJOLEARN_X_PREP_FAST_" + name, "0") == "1")

#: binding -> whether it exports `x_prep_fast_unique` (FAST + Apple default,
#: bindings/_mojolearn_x_prep.mojo X_PREP_FAST_UNIQUE; built with
#: -D MOJOLEARN_X_PREP_FAST_UNIQUE_OFF it does not), probed once per binding
_FAST_UNIQUE = {}


def _fast_on(name, mode):
    """Whether the switch `name` is on for the FAST tier (`mode` is the
    estimator's numeric mode, `_backend.default_mode`). UNIQUE is the
    binding's comptime default (a cached probe); NONEG a set lookup. Off,
    or on another tier, every route below is the old one."""
    if mode != "fast":
        return False
    if name == "UNIQUE":
        binding = _prep_binding(mode)
        key = id(binding)
        on = _FAST_UNIQUE.get(key)
        if on is None:
            on = _FAST_UNIQUE[key] = _optional_prep_entry(binding, "x_prep_fast_unique") is not None
        return on
    return name in _X_PREP_FAST


def encode_labels(y):
    """(classes, int32 codes) under `_labels`' order rule: `_labels.encode_labels`,
    the native encoder (the base binding, or `_mojolearn_core_host` on a
    CPU-only install) with `sorted_classes(flatten_labels(y))` as its
    definition and fallback (lane py-shared; this module used to run the
    Python routine always)."""
    return _labels.encode_labels(y)


def decode_labels(classes, codes):
    """Codes back to labels: int classes an int64 Array, real classes a
    float64 Array, anything else a list: `_labels.decode_labels` (the native
    gather for int and float classes; lane py-shared)."""
    return _labels.decode_labels(classes, codes)


def _x2d(X, name="X"):
    arr = as_f32_c(X, ndim=2, name=name)[0]
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"mojolearn: {name} must be a nonempty two-dimensional array")
    if arr.size > 2 ** 31 - 1:
        raise ValueError(f"mojolearn: {name} exceeds the native Int32 indexing bound")
    return arr


class _PrepBase:
    """sklearn's parameter protocol and fit_transform for the lane's classes."""
    _parameters = ()

    def get_params(self, deep=True):
        return {name: getattr(self, name) for name in self._parameters}

    def set_params(self, **params):
        for k, v in params.items():
            if k not in self._parameters:
                raise ValueError(f"mojolearn: invalid parameter {k!r} for {type(self).__name__}")
            setattr(self, k, v)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X)

    def _check_fitted(self):
        if not hasattr(self, "n_features_in_"):
            raise RuntimeError(f"mojolearn: this {type(self).__name__} instance is not fitted yet")

    def _check_width(self, arr):
        if arr.shape[1] != self.n_features_in_:
            raise ValueError(f"mojolearn: X has {arr.shape[1]} features, {type(self).__name__} "
                             f"was fitted with {self.n_features_in_}")


def _affine(mode, X, center, scale):
    """(X - center) / scale on the device (either may be None)."""
    arr = _x2d(X)
    n, d = arr.shape
    pr = _Prog()
    xo = pr.put(arr)
    co = pr.put(center) if center is not None else _NONE
    so = pr.put(scale) if scale is not None else _NONE
    out = pr.output(n * d)
    pr.stage("affine", n * d, xo, n * d, d, co, so, out)
    return pr.run(mode).get(out, (n, d))


# ---------------------------------------------------------------- scalers
class RobustScaler(_PrepBase):
    """sklearn.preprocessing.RobustScaler: center by the median, scale by the
    quantile range (numpy's linear percentile over the non-NaN entries; NaN is
    ignored in fit and kept in transform). Float32 throughout; a scale below
    10 * float32 eps is one (`_handle_zeros_in_scale`); unit_variance divides
    the scale by norm.ppf(q_max) - norm.ppf(q_min) (Acklam, float32)."""
    _parameters = ("with_centering", "with_scaling", "quantile_range", "copy", "unit_variance")

    def __init__(self, *, with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), copy=True,
                 unit_variance=False):
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
        self.copy = copy
        self.unit_variance = unit_variance

    def fit(self, X, y=None):
        lo, hi = (float(v) for v in self.quantile_range)
        if not 0 <= lo <= hi <= 100:
            raise ValueError(f"mojolearn: invalid quantile range {self.quantile_range!r}")
        if self.unit_variance and not 0 < lo < hi < 100:
            raise ValueError("mojolearn: RobustScaler(unit_variance=True) needs 0 < q_min < q_max < 100 "
                             "(norm.ppf of 0 or 1 is infinite)")
        arr = _x2d(X)
        n, d = arr.shape
        mode = _mode()
        pr = _Prog()
        xo = pr.put(arr)
        so = pr.work(n * d)
        st = pr.alloc(6 * d)
        qf = pr.put_list([lo / 100.0, 0.5, hi / 100.0])
        q = pr.alloc(3 * d)
        center = pr.alloc(d)
        scale = pr.alloc(d)
        pr.stage("sort_cols", d, xo, n, d, so, 0)
        # the quantile stage reads the count row only: an exact integer in
        # the blocked order too (lane gap-prep2), so the same words
        _col_stats(pr, xo, n, d, st, var=False)
        pr.stage("quantile", 3 * d, so, n, d, qf, 3, q, st)
        pr.stage("scale_params", d, q, 3, d, center, scale, 0, 0, 2, 1)
        if self.unit_variance:
            pr.stage("robust_uv", d, scale, qf)
        pr.run(mode)
        self.center_ = pr.get(center, d) if self.with_centering else None
        self.scale_ = pr.get(scale, d) if self.with_scaling else None
        self.numeric_mode_, self.n_features_in_ = mode, d
        return self

    def transform(self, X):
        self._check_fitted()
        arr = _x2d(X)
        self._check_width(arr)
        return _affine(self.numeric_mode_, arr, self.center_, self.scale_)


class MaxAbsScaler(_PrepBase):
    """sklearn.preprocessing.MaxAbsScaler: X / max|X| per column over the
    non-NaN entries (NaN kept in transform); a zero column scales by one."""
    _parameters = ("copy",)

    def __init__(self, *, copy=True):
        self.copy = copy

    def fit(self, X, y=None):
        arr = _x2d(X)
        n, d = arr.shape
        mode = _mode()
        pr = _Prog()
        xo = pr.put(arr)
        scale = pr.alloc(d)
        if _blocked():
            # x_prep/blocked.mojo: one unit per (row block, column), then the
            # largest partial (a maximum is exact in any order: the same words)
            nb = (n + _XB - 1) // _XB
            part, ma = pr.work(5 * nb * d), pr.alloc(d)
            pr.stage("colb_part", nb * d, xo, n, d, part, nb)
            pr.stage("maxabs_fold", d, part, nb, d, ma, scale)
        else:
            st = pr.alloc(6 * d)
            ma = st + 5 * d
            pr.stage("col_stats", d, xo, n, d, st)
            pr.stage("scale_params", d, ma, 1, d, _NONE, scale, 1)
        pr.run(mode)
        self.max_abs_ = pr.get(ma, d)
        self.scale_ = pr.get(scale, d)
        self.numeric_mode_, self.n_features_in_, self.n_samples_seen_ = mode, d, n
        return self

    def transform(self, X):
        self._check_fitted()
        arr = _x2d(X)
        self._check_width(arr)
        return _affine(self.numeric_mode_, arr, None, self.scale_)


# ---------------------------------------------------------------- encoders
def _finite_2d(X, who):
    """NaN is a category like any other (the one quiet NaN word, sorted last,
    as numpy's unique does); OrdinalEncoder therefore codes NaN by its index
    rather than sklearn's `encoded_missing_value`."""
    return _x2d(X)


def _fit_categories(mode, arr):
    """Per column, the sorted distinct values (-0.0 folded into 0.0), on the
    device: a sort per column and a run scan."""
    n, d = arr.shape
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
        for c in range(d):
            cnt, off = pr.work(nch), pr.work(nch)
            pr.stage("uniq_count", nch, so + c * n, n, ch, cnt)
            pr.stage("uniq_scan", 1, cnt, nch, off, co + c)
            pr.stage("uniq_write", nch, so + c * n, n, ch, off, uo + c * n)
    else:
        pr.stage("unique_cols", d, so, n, d, uo, co)
    pr.run(mode)
    counts = [int(v) for v in pr.values(co, d)]
    return [pr.get(uo + c * n, counts[c]) for c in range(d)]


def _given_categories(categories, arr, mode, check_unknown, who):
    """categories=<list>: one list per column, numeric, sorted ascending with
    at most a NaN last (the reference refuses unsorted numeric categories),
    stored as float32. With check_unknown (handle_unknown='error') a training
    value outside its column's list is refused, as the reference's fit."""
    n, d = arr.shape
    if len(categories) != d:
        raise ValueError(f"mojolearn: {who} categories has {len(categories)} lists; X has {d} features")
    out = []
    for j, cats in enumerate(categories):
        vals = [float(v) for v in (cats.tolist() if hasattr(cats, "tolist") else cats)]
        f32 = array.array("f", vals)
        nums = [v for v in f32 if v == v]
        if len(nums) < len(f32) - 1 or (len(nums) < len(f32) and f32[-1] == f32[-1]):
            raise ValueError(f"mojolearn: {who} categories[{j}]: nan must be the last category")
        if nums != sorted(nums):
            raise ValueError(f"mojolearn: {who} unsorted categories are not supported for numerical categories")
        if any(a == b for a, b in zip(nums, nums[1:])):
            raise ValueError(f"mojolearn: {who} categories[{j}] has values equal in float32")
        if not f32:
            raise ValueError(f"mojolearn: {who} categories[{j}] is empty")
        canon = [0.0 if v == 0 else v for v in nums] + ([float("nan")] if len(nums) < len(f32) else [])
        out.append(Array.from_list(canon, "<f4"))
    if check_unknown:
        pr = _Prog()
        _codes_neg = _codes(pr, arr, out)
        pr.run(mode)
        bad = [j for j, v in enumerate(pr.values(_codes_neg[1], d)) if v > 0]
        if bad:
            raise ValueError(f"mojolearn: {who} found unknown categories in column(s) {bad} during fit")
    return out


def _category_block(pr, categories):
    """Every column's categories in one (d, kmax) block. Returns (offset, kmax)."""
    kmax = max(c.size for c in categories)
    block = [0.0] * (len(categories) * kmax)
    for j, cats in enumerate(categories):
        block[j * kmax:j * kmax + cats.size] = cats.tolist()
    return pr.put_list(block), kmax


def _codes(pr, arr, categories, neg=True):
    """Stages that write each element's category index (or -1) and each
    column's unknown count. Returns (codes offset, unknown-count offset).
    neg=False (lane apple-fast-prep, MOJOLEARN_X_PREP_FAST_NONEG=1): no
    count, the offset None. `count_neg` (x_prep/prims.mojo count_neg_unit)
    is ONE thread per column over every row; the encoders read it only under
    handle_unknown='error' / 'warn' or with a drop, so the board's
    OneHotEncoder(handle_unknown='ignore') and
    OrdinalEncoder(handle_unknown='use_encoded_value') never did."""
    n, d = arr.shape
    xo = pr.put(arr)
    uo, kmax = _category_block(pr, categories)
    co = pr.put_list([c.size for c in categories])
    codes = pr.alloc(n * d)
    pr.stage("lookup", n * d, xo, n, d, uo, kmax, co, codes)
    if not neg:
        return codes, None
    neg = pr.alloc(d)
    pr.stage("count_neg", d, codes, n, d, neg)
    return codes, neg


def _inverse_codes(pr, arr, categories, missing, emv, unknown, ncat=None, back=None):
    """OrdinalEncoder-style inverse: stages that turn codes back into category
    values. Returns (values offset, codes offset); a code of -2 is invalid,
    -1 unknown (the reference's None, written as NaN). With infrequent
    categories, `ncat` is each column's grouped cardinality and `back` the
    (MAP, MSTRIDE, NMAP) grouped -> category index table (the infrequent code
    maps to -3, written as NaN)."""
    n, d = arr.shape
    xo = pr.put(arr)
    mo = pr.put_list(missing)
    eo = pr.put_scalar(emv)
    uo = pr.put_scalar(0.0 if unknown is None else unknown)
    no = pr.put_list(ncat if ncat is not None else [c.size for c in categories])
    codes = pr.alloc(n * d)
    pr.stage("ord_inverse", n * d, xo, n, d, mo, eo, 0 if unknown is None else 1, uo, no, codes)
    if back is not None:
        codes = _remap(pr, codes, n, d, back, _NONE)
    return _gather_categories(pr, codes, n, d, categories), codes


def _check_infrequent_params(est):
    """The reference's parameter constraints; True when grouping is on."""
    mf, mc = est.min_frequency, est.max_categories
    if mf is not None:
        ok = (isinstance(mf, numbers.Integral) and not isinstance(mf, bool) and mf >= 1) or \
            (isinstance(mf, numbers.Real) and not isinstance(mf, numbers.Integral) and 0 < mf < 1)
        if not ok:
            raise ValueError(f"mojolearn: min_frequency must be an int >= 1 or a float in (0, 1), got {mf!r}")
    if mc is not None and not (isinstance(mc, numbers.Integral) and not isinstance(mc, bool) and mc >= 1):
        raise ValueError(f"mojolearn: max_categories must be an int >= 1, got {mc!r}")
    return mc is not None or mf is not None


def _category_counts(mode, arr, categories):
    """Per column, how many training rows hold each category (the device's
    lookup and a per-column count, integers)."""
    n, d = arr.shape
    pr = _Prog()
    codes, _neg = _codes(pr, arr, categories)
    kmax = max(c.size for c in categories)
    out = pr.alloc(d * kmax)
    pr.stage("code_counts", d, codes, n, d, kmax, out)
    pr.run(mode)
    flat = pr.get_i32(out, d * kmax).tolist()
    return [flat[j * kmax:j * kmax + c.size] for j, c in enumerate(categories)]


def _identify_infrequent(counts, n, min_frequency, max_categories):
    """sklearn `_identify_infrequent`: the sorted infrequent indices, or None.
    Integer counts; the fractional threshold is n * min_frequency in float64."""
    if min_frequency is None:
        mask = [False] * len(counts)
    elif isinstance(min_frequency, numbers.Integral):
        mask = [c < min_frequency for c in counts]
    else:
        lim = n * float(min_frequency)
        mask = [c < lim for c in counts]
    current = len(counts) - sum(mask) + 1
    if max_categories is not None and max_categories < current:
        keep = max_categories - 1
        if keep == 0:
            mask = [True] * len(counts)
        else:
            order = sorted(range(len(counts)), key=lambda i: counts[i])   # stable, as mergesort
            for i in order[:-keep]:
                mask[i] = True
    idx = [i for i, m in enumerate(mask) if m]
    return idx or None


def _fit_infrequent(est, mode, arr, ignore_missing):
    """Sets est._infrequent (per column: sorted infrequent indices or None)
    and est._grouping (per column: category index -> grouped code, or None),
    as the reference's `_fit_infrequent_category_mapping`. With
    ignore_missing (OrdinalEncoder) a trailing NaN category is left out of
    the grouping."""
    counts = _category_counts(mode, arr, est.categories_)
    n = arr.shape[0]
    est._infrequent, est._grouping = [], []
    for cats, cnt in zip(est.categories_, counts):
        if ignore_missing and cats.size and _is_nan_value(cats.tolist()[-1]):
            cnt = cnt[:-1]
        inf = _identify_infrequent(cnt, n, est.min_frequency, est.max_categories)
        est._infrequent.append(inf)
        if inf is None:
            est._grouping.append(None)
            continue
        infset = set(inf)
        nf = len(cnt) - len(inf)
        mapping, g = [], 0
        for i in range(len(cnt)):
            if i in infset:
                mapping.append(nf)
            else:
                mapping.append(g)
                g += 1
        est._grouping.append(mapping)
    est.infrequent_categories_ = [None if inf is None else Array.from_list([c.tolist()[i] for i in inf], "<f4")
                                  for c, inf in zip(est.categories_, est._infrequent)]


def _grouping_table(pr, grouping, inverse=False):
    """(MAP, MSTRIDE, NMAP) for remap_codes: category -> grouped code, or
    (inverse) grouped code -> category index with the infrequent code -> -3."""
    tables = []
    for g in grouping:
        if g is None:
            tables.append([])
        elif not inverse:
            tables.append(list(g))
        else:
            nf = max(g)
            back = [0] * (nf + 1)
            for i, v in enumerate(g):
                if v < nf:
                    back[v] = i
            back[nf] = -3
            tables.append(back)
    stride = max(1, max(len(t) for t in tables))
    flat = []
    for t in tables:
        flat.extend(t + [0] * (stride - len(t)))
    return pr.put_list(flat), stride, pr.put_list([len(t) for t in tables])


def _remap(pr, codes, n, d, table, neg):
    mo, stride, no = table
    out = pr.alloc(n * d)
    pr.stage("remap_codes", n * d, codes, n * d, d, mo, stride, no, neg, out)
    return out


def _gather_categories(pr, codes, n, d, categories):
    co, kmax = _category_block(pr, categories)
    out = pr.alloc(n * d)
    pr.stage("cat_gather", n * d, codes, n, d, co, kmax, out)
    return out


def _bad_codes(pr, codes, n, d, bad):
    """Row indices with a code equal to `bad`."""
    vals = pr.values(codes, n * d)
    return sorted({i // d for i, v in enumerate(vals) if v == bad})


def _block_argmax(pr, arr, widths, drops, check):
    """One code per (row, block): the block_argmax stage over a (n, sum(widths)) input."""
    n, W = arr.shape
    d = len(widths)
    xo = pr.put(arr)
    so = pr.put_list([sum(widths[:j]) for j in range(d)])
    wo = pr.put_list(widths)
    do = pr.put_list(drops) if drops is not None else _NONE
    codes = pr.alloc(n * d)
    pr.stage("block_argmax", n * d, xo, n, W, d, so, wo, do, 1 if check else 0, codes)
    return codes


def _raise_unknown(pr, neg, d, who):
    bad = [j for j, v in enumerate(pr.values(neg, d)) if v > 0]
    if bad:
        raise ValueError(f"mojolearn: {who} found unknown categories in column(s) {bad} during transform")


class OrdinalEncoder(_PrepBase):
    """sklearn.preprocessing.OrdinalEncoder over numeric columns: each value's
    index among its column's sorted distinct training values, as float32; a
    NaN seen in fit (the last category) is written as encoded_missing_value
    (NaN by default). handle_unknown 'error' or 'use_encoded_value'.
    inverse_transform maps codes back (an unknown_value row is NaN where the
    reference writes None: there is no object Array). categories='auto' or
    one sorted numeric list per column (as the reference). min_frequency /
    max_categories group infrequent categories into one code after the
    frequent ones (the reference's rule, a NaN category left out of it);
    inverse_transform writes that code as NaN (the reference's
    'infrequent_sklearn' string: there is no object Array), and
    unknown_value may not equal it (the reference lets it collide)."""
    _parameters = ("categories", "dtype", "handle_unknown", "unknown_value", "encoded_missing_value",
                   "min_frequency", "max_categories")

    def __init__(self, *, categories="auto", dtype=None, handle_unknown="error", unknown_value=None,
                 encoded_missing_value=float("nan"), min_frequency=None, max_categories=None):
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value
        self.encoded_missing_value = encoded_missing_value
        self.min_frequency = min_frequency
        self.max_categories = max_categories

    def fit(self, X, y=None):
        grouping = _check_infrequent_params(self)
        if self.handle_unknown not in ("error", "use_encoded_value"):
            raise ValueError(f"mojolearn: invalid handle_unknown {self.handle_unknown!r}")
        if self.handle_unknown == "use_encoded_value" and not isinstance(self.unknown_value, numbers.Real):
            raise TypeError("mojolearn: unknown_value must be a number when handle_unknown='use_encoded_value'")
        if not isinstance(self.encoded_missing_value, numbers.Real):
            raise TypeError("mojolearn: encoded_missing_value must be a number (or NaN)")
        arr = _finite_2d(X, "OrdinalEncoder")
        mode = _mode()
        self.categories_ = (_fit_categories(mode, arr) if _is_auto(self.categories) else
                            _given_categories(self.categories, arr, mode, self.handle_unknown == "error",
                                              "OrdinalEncoder"))
        self._missing = [c.size - 1 if c.size and _is_nan_value(c.tolist()[-1]) else -1 for c in self.categories_]
        self._infrequent = self._grouping = None
        if grouping:
            _fit_infrequent(self, mode, arr, True)
        cards = [c.size - (1 if m >= 0 else 0) for c, m in zip(self.categories_, self._missing)]
        if grouping:
            cards = [k if g is None else max(g) + 1 for k, g in zip(cards, self._grouping)]
        if self.handle_unknown == "use_encoded_value" and not _is_nan_value(self.unknown_value):
            if any(0 <= self.unknown_value < k for k in cards):
                raise ValueError(f"mojolearn: the used value for unknown_value {self.unknown_value} is one of the "
                                 "values already used for encoding the seen categories.")
        if any(m >= 0 for m in self._missing) and not _is_nan_value(self.encoded_missing_value):
            bad = [j for j, (k, m) in enumerate(zip(cards, self._missing))
                   if m >= 0 and 0 <= self.encoded_missing_value < k]
            if bad:
                raise ValueError(f"mojolearn: encoded_missing_value ({self.encoded_missing_value}) is already "
                                 f"used to encode a known category in features: {bad}")
        self.numeric_mode_, self.n_features_in_ = mode, arr.shape[1]
        return self

    def transform(self, X):
        self._check_fitted()
        arr = _finite_2d(X, "OrdinalEncoder")
        self._check_width(arr)
        n, d = arr.shape
        pr = _Prog()
        codes, neg = _codes(pr, arr, self.categories_,
                            neg=self.handle_unknown == "error" or not _fast_on("NONEG", self.numeric_mode_))
        out = codes
        if self._grouping is not None:
            out = _remap(pr, codes, n, d, _grouping_table(pr, self._grouping), _NONE)
        if self.handle_unknown == "use_encoded_value":
            val = pr.put_scalar(self.unknown_value)
            src, out = out, pr.alloc(n * d)
            pr.stage("where_neg", n * d, src, n * d, val, out)
        if any(m >= 0 for m in self._missing):
            src, out = out, pr.alloc(n * d)
            pr.stage("where_code", n * d, codes, n, d, pr.put_list(self._missing),
                     pr.put_scalar(self.encoded_missing_value), src, out)
        pr.run(self.numeric_mode_)
        if self.handle_unknown == "error":
            _raise_unknown(pr, neg, d, "OrdinalEncoder")
        return pr.get(out, (n, d))

    def inverse_transform(self, X):
        self._check_fitted()
        arr = _x2d(X)
        self._check_width(arr)
        n, d = arr.shape
        pr = _Prog()
        unknown = self.unknown_value if self.handle_unknown == "use_encoded_value" else None
        ncat = back = None
        if self._grouping is not None:
            ncat = [c.size if g is None else max(g) + 1 for c, g in zip(self.categories_, self._grouping)]
            back = _grouping_table(pr, self._grouping, inverse=True)
        out, codes = _inverse_codes(pr, arr, self.categories_, self._missing, self.encoded_missing_value, unknown,
                                    ncat, back)
        pr.run(self.numeric_mode_)
        bad = _bad_codes(pr, codes, n, d, -2)
        if bad:
            raise ValueError(f"mojolearn: rows {bad[:10]} hold codes that name no category")
        return pr.get(out, (n, d))


class OneHotEncoder(_PrepBase):
    """sklearn.preprocessing.OneHotEncoder over numeric columns, returned
    DENSE (float32) whatever `sparse_output` says: there is no sparse Array.
    drop None, 'first' or 'if_binary'; handle_unknown 'error' or 'ignore'
    (an unknown value is an all-zero block). inverse_transform is the
    reference's per-block argmax (an all-zero block is the dropped category,
    or unknown: an error for handle_unknown='error', NaN where the reference
    writes None for 'ignore'). categories='auto' or one sorted numeric list
    per column (as the reference). drop may also be one category per
    feature. min_frequency / max_categories group infrequent categories
    into one last column per feature (the reference's rule);
    handle_unknown 'infrequent_if_exist' and 'warn' send an unknown value to
    that column when the feature has one (else an all-zero block).
    inverse_transform writes the infrequent column as NaN (the reference's
    'infrequent_sklearn' string: there is no object Array)."""
    _parameters = ("categories", "drop", "sparse_output", "dtype", "handle_unknown", "min_frequency",
                   "max_categories", "feature_name_combiner")

    def __init__(self, *, categories="auto", drop=None, sparse_output=True, dtype=None, handle_unknown="error",
                 min_frequency=None, max_categories=None, feature_name_combiner="concat"):
        self.categories = categories
        self.drop = drop
        self.sparse_output = sparse_output
        self.dtype = dtype
        self.handle_unknown = handle_unknown
        self.min_frequency = min_frequency
        self.max_categories = max_categories
        self.feature_name_combiner = feature_name_combiner

    def fit(self, X, y=None):
        grouping = _check_infrequent_params(self)
        if self.handle_unknown not in ("error", "ignore", "infrequent_if_exist", "warn"):
            raise ValueError(f"mojolearn: OneHotEncoder handle_unknown={self.handle_unknown!r} is not valid")
        if isinstance(self.drop, str) and self.drop not in ("first", "if_binary"):
            raise ValueError("mojolearn: OneHotEncoder drop must be None, 'first', 'if_binary' or one category "
                             "per feature")
        arr = _finite_2d(X, "OneHotEncoder")
        mode = _mode()
        self.categories_ = (_fit_categories(mode, arr) if _is_auto(self.categories) else
                            _given_categories(self.categories, arr, mode, self.handle_unknown == "error",
                                              "OneHotEncoder"))
        self._infrequent = self._grouping = None
        if grouping:
            _fit_infrequent(self, mode, arr, False)
        self._set_drop_idx()
        self.numeric_mode_, self.n_features_in_ = mode, arr.shape[1]
        return self

    def _grouped_sizes(self):
        return [c.size if (self._grouping is None or g is None) else max(g) + 1
                for c, g in zip(self.categories_, self._grouping or [None] * len(self.categories_))]

    def _set_drop_idx(self):
        """The reference's `_set_drop_idx`: `_drop_after` in grouped codes,
        drop_idx_ in category indices."""
        sizes = self._grouped_sizes()
        grouping = self._grouping or [None] * len(sizes)
        if self.drop is None:
            after = None
        elif self.drop == "first":
            after = [0] * len(sizes)
        elif self.drop == "if_binary":
            after = [0 if k == 2 else None for k in sizes]
        else:
            vals = list(self.drop.tolist() if hasattr(self.drop, "tolist") else self.drop)
            if len(vals) != len(sizes):
                raise ValueError(f"mojolearn: `drop` should have length equal to the number of features "
                                 f"({len(sizes)}), got {len(vals)}")
            after, missing = [], []
            for j, (v, cats) in enumerate(zip(vals, self.categories_)):
                cl = cats.tolist()
                if _is_nan_value(v):
                    hit = [cats.size - 1] if cl and _is_nan_value(cl[-1]) else []
                else:
                    fv = array.array("f", [float(v)])[0]
                    hit = [i for i, c in enumerate(cl) if c == fv]
                if not hit:
                    missing.append((j, v))
                    continue
                i = hit[0]
                if grouping[j] is not None:
                    if i in self._infrequent[j]:
                        raise ValueError(f"mojolearn: Unable to drop category {cl[i]!r} from feature {j} "
                                         "because it is infrequent")
                    i = grouping[j][i]
                after.append(i)
            if missing:
                raise ValueError("mojolearn: The following categories were supposed to be dropped, but were not "
                                 "found in the training data.\n" + "\n".join(
                                     f"Category: {v}, Feature: {j}" for j, v in missing))
        self._drop_after = after
        if after is None:
            self.drop_idx_ = None
        else:
            self.drop_idx_ = [a if (a is None or g is None) else g.index(a) for a, g in zip(after, grouping)]

    def _widths(self):
        drops = self._drop_after or [None] * len(self.categories_)
        return [k - (0 if dr is None else 1) for k, dr in zip(self._grouped_sizes(), drops)], drops

    def _unknown_to(self):
        """Per column, the grouped code an unknown value takes: the infrequent
        one under 'infrequent_if_exist' / 'warn' when the column has it, else -1."""
        if self._grouping is None or self.handle_unknown not in ("infrequent_if_exist", "warn"):
            return None
        return [-1 if g is None else max(g) for g in self._grouping]

    def transform(self, X):
        self._check_fitted()
        arr = _finite_2d(X, "OneHotEncoder")
        self._check_width(arr)
        n, d = arr.shape
        widths, drops = self._widths()
        starts = [sum(widths[:j]) for j in range(d)]
        W = sum(widths)
        pr = _Prog()
        codes, neg = _codes(pr, arr, self.categories_,
                            neg=self.handle_unknown in ("error", "warn") or self.drop is not None
                            or not _fast_on("NONEG", self.numeric_mode_))
        if self._grouping is not None:
            unk = self._unknown_to()
            codes = _remap(pr, codes, n, d, _grouping_table(pr, self._grouping),
                           _NONE if unk is None else pr.put_list(unk))
        so = pr.put_list(starts)
        do = pr.put_list([-1 if dr is None else dr for dr in drops])
        out = pr.output(n * W)
        pr.stage("onehot", n * d, codes, n, d, so, do, W, out)
        pr.run(self.numeric_mode_)
        if self.handle_unknown == "error":
            _raise_unknown(pr, neg, d, "OneHotEncoder")
        elif self.handle_unknown == "warn" or (self.drop is not None and
                                               self.handle_unknown in ("ignore", "infrequent_if_exist")):
            bad = [j for j, v in enumerate(pr.values(neg, d)) if v > 0]
            if bad:
                import warnings
                where = ("encoded as the infrequent category" if self.handle_unknown != "ignore"
                         else "encoded as all zeros")
                warnings.warn(f"Found unknown categories in columns {bad} during transform. These unknown "
                              f"categories will be {where}.", UserWarning)
        return pr.get(out, (n, W))

    def inverse_transform(self, X):
        self._check_fitted()
        arr = _x2d(X)
        widths, drops = self._widths()
        n, W, d = arr.shape[0], arr.shape[1], len(widths)
        if W != sum(widths):
            raise ValueError(f"mojolearn: X has {W} columns, expected {sum(widths)}")
        pr = _Prog()
        codes = _block_argmax(pr, arr, widths, [-1 if dr is None else dr for dr in drops], True)
        grouped = codes
        if self._grouping is not None:
            codes = _remap(pr, codes, n, d, _grouping_table(pr, self._grouping, inverse=True), _NONE)
        out = _gather_categories(pr, codes, n, d, self.categories_)
        pr.run(self.numeric_mode_)
        # an all-zero block is unknown (NaN) under 'ignore', and under
        # 'infrequent_if_exist' / 'warn' for a column with no infrequent
        # category; anywhere else it cannot be inverted
        strict = [self.handle_unknown == "error" or (self.handle_unknown != "ignore" and self._infrequent is not None
                                                      and self._infrequent[j] is not None) for j in range(d)]
        vals = pr.values(grouped, n * d)
        bad = sorted({i // d for i, v in enumerate(vals) if v == -1 and strict[i % d]})
        if bad:
            raise ValueError(f"mojolearn: samples {bad[:10]} can not be inverted when drop=None and "
                             "handle_unknown='error' because they contain all zeros")
        return pr.get(out, (n, d))


# ---------------------------------------------------------------- target encoder
def _splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return state, z ^ (z >> 31)


def _kfold_assignment(n, n_folds, seed, shuffle=True):
    """Row -> fold for a shuffled K-fold: a Fisher-Yates permutation drawn
    from splitmix64(seed) in integer arithmetic (the same on every machine),
    then numpy KFold's split of the permuted order (the first n % k folds one
    row longer)."""
    perm = list(range(n))
    state = int(seed) & 0xFFFFFFFFFFFFFFFF
    for i in range(n - 1, 0, -1) if shuffle else ():
        state, z = _splitmix64(state)
        j = z % (i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    fold = [0] * n
    start = 0
    for k in range(n_folds):
        size = n // n_folds + (1 if k < n % n_folds else 0)
        for r in perm[start:start + size]:
            fold[r] = k
        start += size
    return fold


def _native_folds(n, n_folds, seed, shuffle, codes=None, n_classes=0, as_array=False):
    """The fold assignment through the binding's host entry (x_prep/folds.mojo,
    the same integers), or None when the binding has none. `codes` is a list
    or an int32 Array; as_array returns the folds as an int32 Array."""
    if os.environ.get("MOJOLEARN_XPREP_NATIVE_FOLDS", "1") == "0":
        return None
    b = _prep_binding(_mode())
    s = int(seed) & 0xFFFFFFFFFFFFFFFF
    halves = (s & 0xFFFFFFFF, s >> 32)
    out = array.array("i", bytes(4 * max(n, 1)))
    if codes is None:
        if _optional_prep_entry(b, "x_prep_kfold_folds") is None:
            return None
        b.x_prep_kfold_folds(out.buffer_info()[0], (n, n_folds, 1 if shuffle else 0), halves)
    else:
        if _optional_prep_entry(b, "x_prep_strat_folds") is None:
            return None
        if isinstance(codes, Array) and codes.dtype == "<i4" and codes._has_order("C") and codes.size == n:
            cod, cod_addr = codes, addr_ro(codes, name="codes")
        else:
            cod = array.array("i", codes)
            cod_addr = cod.buffer_info()[0]
        if b.x_prep_strat_folds(cod_addr, out.buffer_info()[0],
                                (n, n_classes, n_folds, 1 if shuffle else 0), halves) != 0:
            raise ValueError(f"mojolearn: n_splits={n_folds} cannot be greater than the number of members in each class")
    if as_array:
        return Array._owned(out, (len(out),), "<i4", "C")
    return out[:n].tolist()


def _stratified_assignment(codes, n_folds, seed, shuffle=True):
    """Row -> fold for numpy StratifiedKFold (`_make_test_folds`): classes
    renumbered by first appearance, each class's per-fold counts from the
    round robin over the sorted codes, a class's rows taking its fold
    indices in blocks; with shuffle each class's block is a Fisher-Yates
    permutation from one splitmix64(seed) stream (the reference draws
    numpy's)."""
    first = {}
    for c in codes:
        first.setdefault(c, len(first))
    enc = [first[c] for c in codes]
    K = len(first)
    counts = [0] * K
    for e in enc:
        counts[e] += 1
    if all(n_folds > k for k in counts):
        raise ValueError(f"mojolearn: n_splits={n_folds} cannot be greater than the number of members in each class")
    order = sorted(enc)
    alloc = [[0] * K for _ in range(n_folds)]
    for f in range(n_folds):
        for e in order[f::n_folds]:
            alloc[f][e] += 1
    rows = [[] for _ in range(K)]
    for i, e in enumerate(enc):
        rows[e].append(i)
    fold = [0] * len(codes)
    state = int(seed) & 0xFFFFFFFFFFFFFFFF
    for k in range(K):
        block = [f for f in range(n_folds) for _ in range(alloc[f][k])]
        if shuffle:
            for i in range(len(block) - 1, 0, -1):
                state, z = _splitmix64(state)
                j = z % (i + 1)
                block[i], block[j] = block[j], block[i]
        for r, f in zip(rows[k], block):
            fold[r] = f
    return fold


def _target_binary_codes(y, target_type):
    """(classes, int32 codes Array) of an explicit binary target, the label
    pass native (lane prep-apple3, `te_arrays`); None when `_target_kind`
    must run (any other target_type, or the change off)."""
    if target_type != "binary" or not _r3("te_arrays"):
        return None
    classes, codes = encode_labels(y)
    if not (isinstance(codes, Array) and codes.dtype == "<i4"):
        return None
    return classes, codes


def _target_kind(y, target_type):
    """(kind, classes, Y rows as a flat float list with T columns, T)."""
    labels = flatten_labels(y)
    if target_type == "continuous" or (target_type == "auto" and labels and all(
            isinstance(v, numbers.Real) and not isinstance(v, bool) for v in labels)
            and any(float(v) != int(float(v)) for v in labels)):
        return "continuous", None, [float(v) for v in labels], 1
    classes, codes = encode_labels(labels)
    codes = [int(c) for c in codes]
    if target_type == "binary" or (target_type == "auto" and len(classes) <= 2):
        return "binary", classes, [float(c) for c in codes], 1
    K = len(classes)
    flat = [0.0] * (len(codes) * K)
    for i, c in enumerate(codes):
        flat[i * K + c] = 1.0
    return "multiclass", classes, flat, K


class TargetEncoder(_PrepBase):
    """sklearn.preprocessing.TargetEncoder over numeric category columns:
    binary, continuous and multiclass targets, smooth 'auto' (empirical
    Bayes) or a float. `fit_transform` cross-fits over `cv` folds, as the
    reference: KFold for a continuous target, StratifiedKFold for a binary or
    multiclass one; the shuffle comes from `random_state` (splitmix64; the
    reference draws numpy's). `cv` may also be a splitter object (its
    `split(X, y)`) or an iterable of (train, test) index pairs, as the
    reference: the test folds must cover every row exactly once, and each
    fold's training rows must be every other row. categories='auto' or one
    sorted numeric list per column (as the encoders). Float32 throughout.
    A training value outside a given list is excluded from every category's
    statistics (it still counts in the target mean); the reference codes it
    as the first category, see x_prep/NOT_IMPLEMENTED.tsv."""
    _parameters = ("categories", "target_type", "smooth", "cv", "shuffle", "random_state")

    def __init__(self, categories="auto", target_type="auto", smooth="auto", cv=5, shuffle=True,
                 random_state=None):
        self.categories = categories
        self.target_type = target_type
        self.smooth = smooth
        self.cv = cv
        self.shuffle = shuffle
        self.random_state = random_state

    def _check(self):
        if not _is_auto(self.categories) and not isinstance(self.categories, (list, tuple)):
            raise ValueError("mojolearn: TargetEncoder categories must be 'auto' or a list of lists")
        if self.target_type not in ("auto", "binary", "continuous", "multiclass"):
            raise ValueError(f"mojolearn: invalid target_type {self.target_type!r}")
        if not (self.smooth == "auto" or (isinstance(self.smooth, numbers.Real) and self.smooth >= 0)):
            raise ValueError(f"mojolearn: invalid smooth {self.smooth!r}")
        if isinstance(self.cv, numbers.Integral) and self.cv < 2:
            raise ValueError("mojolearn: TargetEncoder cv must be an integer >= 2, a splitter or an iterable")

    def _run(self, arr, y, folds, n_folds, apply_rows_folds, binary=None):
        """binary (lane prep-apple3, `te_arrays`): (classes, int32 codes) of an
        explicit binary target, with `folds` an int32 Array: both cross as
        int32 words and become floats on the device (i2f), the same words the
        lists gave."""
        n, d = arr.shape
        if binary is None:
            kind, classes, yflat, T = _target_kind(y, self.target_type)
            rows = len(yflat)
        else:
            kind, classes, yflat, T = "binary", binary[0], None, 1
            rows = binary[1].size
        if rows != n * T:
            raise ValueError("mojolearn: X and y have different numbers of rows")
        mode = _mode()
        cats = (_fit_categories(mode, arr) if _is_auto(self.categories) else
                _given_categories(self.categories, arr, mode, False, "TargetEncoder"))
        cmax = max(c.size for c in cats)
        F = n_folds
        pr = _Prog()
        codes, _neg = _codes(pr, arr, cats)
        if binary is None:
            yo = pr.put_list(yflat)
            fo = pr.put_list(folds if folds is not None else [-1] * n)
        else:
            yo = pr.put_codes(binary[1])
            fo = pr.put_codes(folds)
        nco = pr.put_list([c.size for c in cats])
        meta = pr.alloc(2 * (F + 1) * T)
        smo = pr.put_scalar(-1.0 if self.smooth == "auto" else float(self.smooth))
        enc = pr.alloc((F + 1) * d * cmax * T)
        pr.stage("te_global", (F + 1) * T, yo, n, T, fo, meta)
        if _optional_prep_entry(_prep_binding(mode), "x_prep_host_column") is not None:
            # the host binding groups te_enc its own way (x_prep/host/target.mojo)
            pr.stage("te_enc", (F + 1) * d * cmax * T, codes, n, d, yo, T, fo, cmax, nco, meta, smo, enc)
        else:
            # each category's rows, ascending (te_bucket): te_enc walks one bucket, not every row
            bstart, brows = pr.alloc(d * (cmax + 1)), pr.alloc(n * d)
            if os.environ.get("MOJOLEARN_XPREP_TE_PBUCKET", "1") != "0":
                # the buckets by chunks in parallel (te_hist .. te_hscatter), te_bucket's START and
                # ROWS by construction. Default since lane prep-apple3 (M4, request 1790626651574:
                # FAST taxi 1.204 -> 0.961 s, IDENTICAL 1.292 -> 1.017 s, digests equal)
                ch = max(1, min(256, (n + 4095) // 4096))
                hh, tot = pr.scratch(d * ch * cmax), pr.scratch(d * cmax)
                pr.stage("te_hist", d * ch, codes, n, d, cmax, ch, hh)
                pr.stage("te_hsum", d * cmax, cmax, ch, hh, tot)
                pr.stage("te_hstart", d, cmax, bstart, tot)
                pr.stage("te_hscatter", d * ch, codes, n, d, cmax, ch, hh, bstart, brows)
            else:
                pr.stage("te_bucket", d, codes, n, d, cmax, bstart, brows)
            if os.environ.get("MOJOLEARN_XPREP_TE_GATHER", "1") != "0":
                # each bucket's folds and targets in bucket order (te_gather): te_enc streams them
                gb = pr.scratch(n * d * (1 + T))
                pr.stage("te_gather", n * d, brows, n, d, yo, T, fo, gb)
                pr.stage("te_enc", (F + 1) * d * cmax * T, codes, n, d, yo, T, fo, cmax, nco, meta, smo, enc,
                         bstart + 1, brows, gb + 1)
            else:
                pr.stage("te_enc", (F + 1) * d * cmax * T, codes, n, d, yo, T, fo, cmax, nco, meta, smo, enc,
                         bstart + 1, brows)
        out = _NONE
        if apply_rows_folds:
            out = pr.output(n * d * T)
            pr.stage("te_apply", n * d * T, codes, n, d, T, fo, enc, cmax, meta, F, out)
        pr.run(mode)
        self.categories_, self.target_type_, self.numeric_mode_, self.n_features_in_ = cats, kind, mode, d
        self.classes_ = classes
        self._T, self._cmax = T, cmax
        full = F * d * cmax * T
        self._enc = pr.get(enc + full, d * cmax * T)
        self._meta = pr.get(meta + 2 * F * T, 2 * T)
        self.encodings_ = [pr.get(enc + full + (j * cmax) * T, cats[j].size * T) for j in range(d)]
        means = pr.values(meta + 2 * F * T, 2 * T)[0::2]
        self.target_mean_ = pr.get(meta + 2 * F * T, 1) if T == 1 else Array.from_list(means, "<f4")
        return pr.get(out, (n, d * T)) if apply_rows_folds else None

    def fit(self, X, y):
        self._check()
        self._run(_x2d(X), y, None, 0, False)
        return self

    def _splitter_folds(self, X, y, n):
        """Row -> fold from a splitter object or (train, test) iterable."""
        splits = list(self.cv.split(X, y) if hasattr(self.cv, "split") else self.cv)
        idx = lambda a: [int(i) for i in (a.tolist() if hasattr(a, "tolist") else a)]
        fold = [-1] * n
        for k, (_train, test) in enumerate(splits):
            for i in idx(test):
                if not 0 <= i < n or fold[i] != -1:
                    fold = None
                    break
                fold[i] = k
            if fold is None:
                break
        if fold is None or -1 in fold or len(splits) < 1:
            raise ValueError("mojolearn: Validation indices from `cv` must cover each sample index exactly once "
                             "with no overlap. Pass a splitter with non-overlapping validation folds as `cv`.")
        sizes = [0] * len(splits)
        for k in fold:
            sizes[k] += 1
        for k, (train, _test) in enumerate(splits):
            # the training rows are every row outside fold k exactly once: as
            # many as there are, distinct, in range and none in fold k (the
            # folds already cover each row once)
            tr = idx(train)
            if (len(tr) != n - sizes[k] or len(set(tr)) != len(tr)
                    or any(not 0 <= i < n or fold[i] == k for i in tr)):
                raise NotImplementedError("mojolearn: TargetEncoder cv folds whose training rows are not every "
                                          "row outside the test fold are not implemented")
        return fold, len(splits)

    def fit_transform(self, X, y):
        self._check()
        arr = _x2d(X)
        n = arr.shape[0]
        if self.cv is not None and not isinstance(self.cv, numbers.Integral):
            folds, F = self._splitter_folds(X, y, n)
            return self._run(arr, y, folds, F, True)
        cv = 5 if self.cv is None else int(self.cv)
        if n < cv:
            raise ValueError(f"mojolearn: cv={cv} folds need at least {cv} rows")
        seed = 0 if self.random_state is None else int(self.random_state)
        binary = _target_binary_codes(y, self.target_type)
        if binary is not None:
            if binary[1].size != n:
                raise ValueError("mojolearn: X and y have different numbers of rows")
            folds = _native_folds(n, cv, seed, bool(self.shuffle), binary[1], len(binary[0]), as_array=True)
            if folds is not None:
                return self._run(arr, y, folds, cv, True, binary=binary)
        kind, _classes, _yflat, _T = _target_kind(y, self.target_type)
        if kind == "continuous":
            folds = _native_folds(n, cv, seed, bool(self.shuffle))
            if folds is None:
                folds = _kfold_assignment(n, cv, seed, bool(self.shuffle))
        else:
            labels = flatten_labels(y)
            if len(labels) != n:
                raise ValueError("mojolearn: X and y have different numbers of rows")
            classes, codes = encode_labels(labels)
            folds = _native_folds(n, cv, seed, bool(self.shuffle), codes.tolist(), len(classes))
            if folds is None:
                folds = _stratified_assignment(labels, cv, seed, bool(self.shuffle))
        return self._run(arr, y, folds, cv, True)

    def transform(self, X):
        self._check_fitted()
        arr = _x2d(X)
        self._check_width(arr)
        n, d = arr.shape
        T, cmax = self._T, self._cmax
        pr = _Prog()
        codes, _neg = _codes(pr, arr, self.categories_)
        enc = pr.put(self._enc)
        meta = pr.put(self._meta)
        out = pr.output(n * d * T)
        pr.stage("te_apply", n * d * T, codes, n, d, T, _NONE, enc, cmax, meta, 0, out)
        pr.run(self.numeric_mode_)
        return pr.get(out, (n, d * T))


# ---------------------------------------------------------------- imputer
def _is_auto(v):
    return isinstance(v, str) and v == "auto"


def _is_nan_value(v):
    return isinstance(v, float) and v != v


def _mark_missing(pr, xo, count, missing_values):
    """Stages that turn a numeric `missing_values` into NaN; returns the offset
    to read (the input itself when missing_values is NaN)."""
    if missing_values is None or _is_nan_value(missing_values):
        return xo
    val = pr.put_scalar(missing_values)
    out = pr.alloc(count)
    pr.stage("mark_missing", count, xo, count, val, out)
    return out


class SimpleImputer(_PrepBase):
    """sklearn.impute.SimpleImputer, numeric: strategy 'mean', 'median'
    (numpy's linear percentile of the non-missing entries), 'most_frequent'
    (the smallest on a tie) or 'constant'. An all-missing column is dropped
    from the output unless `keep_empty_features` (its statistic is NaN, as in
    the reference; with keep_empty_features it is 0, or fill_value).
    add_indicator appends MissingIndicator's columns (the features with a
    missing value in fit, 1.0 where missing). A callable strategy is the
    reference's: statistics_[j] = float(strategy(v)) over column j's
    non-missing float32 values v in row order (a 1-D Array); it runs in
    Python, on the host, and a NaN statistic drops the column unless
    keep_empty_features, as the reference. The fill itself runs on the
    device like every other strategy."""
    _parameters = ("missing_values", "strategy", "fill_value", "copy", "add_indicator", "keep_empty_features")

    def __init__(self, *, missing_values=float("nan"), strategy="mean", fill_value=None, copy=True,
                 add_indicator=False, keep_empty_features=False):
        self.missing_values = missing_values
        self.strategy = strategy
        self.fill_value = fill_value
        self.copy = copy
        self.add_indicator = add_indicator
        self.keep_empty_features = keep_empty_features

    def fit(self, X, y=None):
        if not callable(self.strategy) and self.strategy not in ("mean", "median", "most_frequent", "constant"):
            raise ValueError(f"mojolearn: SimpleImputer strategy {self.strategy!r} is not valid")
        if self.strategy == "constant" and self.fill_value is not None and \
                not isinstance(self.fill_value, numbers.Real):
            raise TypeError("mojolearn: SimpleImputer fill_value must be numeric")
        arr = _x2d(X)
        n, d = arr.shape
        mode = _mode()
        pr = _Prog()
        x_in = pr.put(arr)
        xo = _mark_missing(pr, x_in, n * d, self.missing_values)
        # only the median and the mode read the sorted columns (lane prep-apple3, `imputer_nosort`)
        sorts = self.strategy in ("median", "most_frequent") or not _r3("imputer_nosort")
        so = (pr.work(n * d) if self.strategy in ("median", "most_frequent") else pr.alloc(n * d)) if sorts else 0
        st = pr.alloc(6 * d)
        med = pr.alloc(d)
        mf = pr.alloc(d)
        half = pr.put_list([0.5])
        if sorts:
            pr.stage("sort_cols", d, xo, n, d, so, 0)
        pr.stage("col_stats", d, xo, n, d, st)
        if self.strategy == "median":
            pr.stage("quantile", d, so, n, d, half, 1, med, st)
        if self.strategy == "most_frequent":
            pr.stage("mode_cols", d, so, n, d, mf, _NONE)
        pr.run(mode)
        counts = [int(v) for v in pr.values(st, d)]
        empty = [c == 0 for c in counts]
        if callable(self.strategy):
            # missing_values NaN: the marked block is the input itself, which never comes back
            return self._fit_callable(arr if xo == x_in else pr.get(xo, (n, d)), counts, mode)
        if self.strategy == "constant":
            fv = 0.0 if self.fill_value is None else float(self.fill_value)
            stats = [fv] * d
            fill = list(stats)
        else:
            src = {"mean": st + d, "median": med, "most_frequent": mf}[self.strategy]
            stats = pr.values(src, d)
            fill = [0.0 if e else s for s, e in zip(stats, empty)]
        # the reference: an all-missing column's statistic is NaN and the
        # column is dropped, unless keep_empty_features (then 0, or fill_value)
        for j in range(d):
            if empty[j] and not self.keep_empty_features:
                stats[j] = float("nan")
            elif empty[j] and self.strategy != "constant":
                stats[j] = 0.0
        if self.strategy == "constant" or any(empty):
            self.statistics_ = Array.from_list(stats, "<f4")
            self._fill = Array.from_list(fill, "<f4")
        else:
            self.statistics_ = pr.get(src, d)
            self._fill = self.statistics_
        self._keep = [j for j in range(d) if self.keep_empty_features or not empty[j]]
        self._indicator = [j for j in range(d) if counts[j] < n] if self.add_indicator else []
        self.numeric_mode_, self.n_features_in_ = mode, d
        return self

    def _fit_callable(self, marked, counts, mode):
        """strategy=<callable>: the reference's `strategy(masked_X[:, j].compressed())`
        per column over the missing-marked X (NaN = missing)."""
        n, d = marked.shape
        cols = [list(c) for c in zip(*marked.tolist())]
        stats = []
        for j in range(d):
            v = Array.from_list([x for x in cols[j] if x == x], "<f4") if counts[j] else Array((0,), "<f4")
            stats.append(float(self.strategy(v)))
        self.statistics_ = Array.from_list(stats, "<f4")
        self._fill = self.statistics_
        self._keep = [j for j in range(d) if self.keep_empty_features or stats[j] == stats[j]]
        self._indicator = [j for j in range(d) if counts[j] < n] if self.add_indicator else []
        self.numeric_mode_, self.n_features_in_ = mode, d
        return self

    def transform(self, X):
        self._check_fitted()
        arr = _x2d(X)
        self._check_width(arr)
        n, d = arr.shape
        dout = len(self._keep)
        pr = _Prog()
        xo = _mark_missing(pr, pr.put(arr), n * d, self.missing_values)
        so = pr.put(self._fill)
        ko = pr.put_list(self._keep)
        out = pr.alloc(n * dout)
        pr.stage("fill", n * dout, xo, n, d, so, out, ko, dout)
        m = len(self._indicator)
        if m:
            io, mo = pr.put_list(self._indicator), pr.alloc(n * m)
            pr.stage("nan_mask", n * m, xo, n, d, io, m, mo)
            hc = pr.output(n * (dout + m))
            pr.stage("hcat", n * (dout + m), out, dout, mo, m, hc)
        pr.run(self.numeric_mode_)
        if not m:
            return pr.get(out, (n, dout))
        return pr.get(hc, (n, dout + m))


def join_column_blocks(parts, ranges, n, d, mode=None):
    """The (n, d) float32 matrix whose columns [start, end) are the C-order
    (n, end - start) Array parts[k] for ranges[k]: one `colblock` stage per
    part in one program (word copies, on the device on a GPU install)."""
    pr = _Prog()
    out = pr.output(n * d)
    for (start, end), part in zip(ranges, parts):
        w = end - start
        if w <= 0 or n <= 0:
            continue
        pr.stage("colblock", n * w, pr.put(part), w, out, d, start)
    pr.run(_mode() if mode is None else mode)
    return pr.get(out, (n, d))


# ---------------------------------------------------------------- discretizer
def _gather_rows(arr, rows):
    """A new float32 Array of the given rows of a C-order 2-D Array (a byte
    copy per row; no arithmetic)."""
    n, d = arr.shape
    out = Array((len(rows), d), "<f4")
    src, dst, rb = addr_ro(arr, name="X"), out._addr, 4 * d
    for k, r in enumerate(rows):
        ctypes.memmove(dst + k * rb, src + r * rb, rb)
    return out


class KBinsDiscretizer(_PrepBase):
    """sklearn.preprocessing.KBinsDiscretizer: strategy 'uniform', 'quantile'
    (every numpy quantile_method: 'averaged_inverted_cdf', the default,
    'inverted_cdf', 'closest_observation', 'interpolated_inverted_cdf',
    'hazen', 'weibull', 'linear', 'median_unbiased', 'normal_unbiased') or
    'kmeans' (1-D Lloyd from the uniform bin centres); encode 'onehot' (dense:
    there is no sparse Array), 'onehot-dense' or 'ordinal'. A constant column
    is one bin with edges (-inf, inf). Above `subsample` rows the fit uses a
    with-replacement resample drawn from `random_state` by splitmix64 (the
    reference draws numpy's). inverse_transform is the bin centres, as the
    reference's. sample_weight (nonnegative, not all zero) weighs the
    'quantile' edges (the reference's `_weighted_percentile` for
    'averaged_inverted_cdf' / 'inverted_cdf'; the other methods are refused,
    as the reference), the 'uniform' range (min / max over rows of nonzero
    weight) and the 'kmeans' Lloyd; above `subsample` rows it draws the
    resample instead (with replacement, weighted) and is then spent."""
    _parameters = ("n_bins", "encode", "strategy", "quantile_method", "dtype", "subsample", "random_state")

    def __init__(self, n_bins=5, *, encode="onehot", strategy="quantile", quantile_method="averaged_inverted_cdf",
                 dtype=None, subsample=200_000, random_state=None):
        self.n_bins = n_bins
        self.encode = encode
        self.strategy = strategy
        self.quantile_method = quantile_method
        self.dtype = dtype
        self.subsample = subsample
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        if self.encode not in ("onehot", "onehot-dense", "ordinal"):
            raise ValueError(f"mojolearn: invalid encode {self.encode!r}")
        strat = {"uniform": 0, "kmeans": 3}.get(self.strategy)
        if self.strategy == "quantile":
            strat = {"averaged_inverted_cdf": 1, "linear": 2, "inverted_cdf": 4, "closest_observation": 5,
                     "interpolated_inverted_cdf": 6, "hazen": 7, "weibull": 8, "median_unbiased": 9,
                     "normal_unbiased": 10}.get(self.quantile_method)
            if strat is None:
                raise ValueError(f"mojolearn: invalid quantile_method {self.quantile_method!r}")
        if strat is None:
            raise ValueError(f"mojolearn: invalid strategy {self.strategy!r}")
        arr = _x2d(X)
        n, d = arr.shape
        w = None
        if sample_weight is not None:
            w, wl = _check_weights(sample_weight, n, "KBinsDiscretizer")
        if self.subsample is not None and n > self.subsample:
            state = 0 if self.random_state is None else int(self.random_state)
            rows = []
            if w is None:
                for _ in range(int(self.subsample)):
                    state, z = _splitmix64(state)
                    rows.append(z % n)
            else:
                # the reference's weighted resample with replacement (its weights are then
                # spent): row = the first whose cumulative weight exceeds u * total, u a
                # 53-bit splitmix64 uniform; cumulative sums in Python float64
                cum, acc = [], 0.0
                for v in wl:
                    acc += v
                    cum.append(acc)
                for _ in range(int(self.subsample)):
                    state, z = _splitmix64(state)
                    rows.append(min(bisect.bisect_right(cum, (z >> 11) * 2.0 ** -53 * acc), n - 1))
                w = None
            arr = _gather_rows(arr, rows)
            n = arr.shape[0]
        nb = [int(self.n_bins)] * d if isinstance(self.n_bins, numbers.Integral) else [int(b) for b in self.n_bins]
        if len(nb) != d or min(nb) < 2:
            raise ValueError("mojolearn: n_bins must be >= 2 per feature")
        if w is not None and strat not in (0, 1, 3, 4):
            raise ValueError("mojolearn: When fitting with strategy='quantile' and sample weights, quantile_method "
                             "should either be set to 'averaged_inverted_cdf' or 'inverted_cdf', got "
                             f"quantile_method='{self.quantile_method}' instead.")
        nbmax = max(nb)
        mode = _mode()
        pr = _Prog()
        xo = pr.put(arr)
        so = pr.work(n * d) if w is None else pr.alloc(n * d)
        st = pr.alloc(6 * d)
        nbo = pr.put_list(nb)
        edges = pr.alloc(d * (nbmax + 1))
        ne = pr.alloc(d)
        lab = pr.alloc(n * d) if strat == 3 else 0
        cen = pr.alloc(d * nbmax) if strat == 3 else 0
        pr.stage("col_stats", d, xo, n, d, st)
        if w is None:
            pr.stage("sort_cols", d, xo, n, d, so, 0)
            pr.stage("kbins_edges", d, so, n, d, nbo, nbmax, strat, st, edges, ne, lab, cen)
        else:
            stw = st
            if strat in (0, 3):
                # the min / max over the rows of nonzero weight (the reference's nnz mask)
                nz = [i for i, v in enumerate(wl) if v > 0]
                stw = pr.alloc(6 * d)
                if len(nz) < n:
                    xz = pr.put(_gather_rows(arr, nz))
                    pr.stage("col_stats", d, xz, len(nz), d, stw)
                else:
                    stw = st
            if strat == 0:
                pr.stage("kbins_edges", d, so, n, d, nbo, nbmax, 0, stw, edges, ne, 0, 0)
            else:
                ug, ucnt = _weighted_groups(pr, arr, w, n, d)
                if strat == 3:
                    pr.stage("kbins_wkm", d, ug, n, d, ucnt, nbo, nbmax, st, stw, edges, cen, lab)
                else:
                    levels = [[i * (100.0 / b) for i in range(b)] + [100.0] for b in nb]
                    _weighted_levels(pr, ug, ucnt, n, d, levels, strat == 1, edges)
                pr.stage("kbins_edges", d, so, n, d, nbo, nbmax, 11, stw, edges, ne, 0, 0)
        pr.run(mode)
        counts = [int(v) for v in pr.values(ne, d)]
        self.bin_edges_ = [pr.get(edges + j * (nbmax + 1), counts[j]) for j in range(d)]
        self.n_bins_ = Array.from_list([c - 1 for c in counts], "<i8")
        self._edges = pr.get(edges, d * (nbmax + 1))
        self._ne = pr.get(ne, d)
        self._stride = nbmax + 1
        self.numeric_mode_, self.n_features_in_ = mode, d
        return self

    def transform(self, X):
        self._check_fitted()
        arr = _x2d(X)
        self._check_width(arr)
        n, d = arr.shape
        pr = _Prog()
        xo = pr.put(arr)
        eo = pr.put(self._edges)
        no = pr.put(self._ne)
        codes = pr.alloc(n * d)
        pr.stage("kbins_codes", n * d, xo, n, d, eo, self._stride, no, codes)
        if self.encode == "ordinal":
            pr.run(self.numeric_mode_)
            return pr.get(codes, (n, d))
        widths = [int(v) for v in self.n_bins_.tolist()]
        W = sum(widths)
        so = pr.put_list([sum(widths[:j]) for j in range(d)])
        out = pr.output(n * W)
        pr.stage("onehot", n * d, codes, n, d, so, _NONE, W, out)
        pr.run(self.numeric_mode_)
        return pr.get(out, (n, W))

    def inverse_transform(self, X):
        self._check_fitted()
        arr = _x2d(X)
        widths = [int(v) for v in self.n_bins_.tolist()]
        d = len(widths)
        n = arr.shape[0]
        pr = _Prog()
        if self.encode == "ordinal":
            self._check_width(arr)
            xo = pr.put(arr)
            codes = pr.alloc(n * d)
            pr.stage("ord_inverse", n * d, xo, n, d, pr.put_list([-1] * d), pr.put_scalar(0.0), 0,
                     pr.put_scalar(0.0), pr.put_list(widths), codes)
            bad_code, why = -2, "hold codes that name no bin"
        else:
            if arr.shape[1] != sum(widths):
                raise ValueError(f"mojolearn: X has {arr.shape[1]} columns, expected {sum(widths)}")
            codes = _block_argmax(pr, arr, widths, None, True)
            bad_code, why = -1, "can not be inverted because they contain all zeros"
        out = pr.output(n * d)
        pr.stage("kbins_inverse", n * d, codes, n, d, pr.put(self._edges), self._stride, out)
        pr.run(self.numeric_mode_)
        bad = _bad_codes(pr, codes, n, d, bad_code)
        if bad:
            raise ValueError(f"mojolearn: samples {bad[:10]} {why}")
        return pr.get(out, (n, d))


# ---------------------------------------------------------------- naive Bayes
class _Classifier(_PrepBase):
    """predict / predict_proba / predict_log_proba from a subclass's joint
    log likelihood stages (`_jll_stages`), normalised on the device."""

    def _encode_y(self, y, n):
        classes, codes = encode_labels(y)
        if codes.size != n:
            raise ValueError("mojolearn: X and y have different numbers of rows")
        self.classes_ = classes
        return codes

    def _scores(self, X, want):
        self._check_fitted()
        arr = _x2d(X)
        self._check_width(arr)
        n, d = arr.shape
        K = len(self.classes_)
        pr = _Prog()
        xo = pr.put(arr)
        chk = self._score_checks(pr, xo, n, d)
        # the joint log likelihood stays on the device unless it is the answer
        jll = pr.work(n * K) if want else pr.alloc(n * K)
        self._jll_stages(pr, xo, n, d, jll)
        lp = pr.alloc(n * K) if "log" in want else _NONE
        pp = pr.alloc(n * K) if "proba" in want else _NONE
        am = pr.alloc(n) if "predict" in want else _NONE
        if lp != _NONE or pp != _NONE:
            pr.stage("row_softmax", n, jll, n, K, lp, pp)
        if am != _NONE:
            pr.stage("row_argmax", n, jll, n, K, am)
        pr.run(self.numeric_mode_)
        self._score_refusals(pr, d, chk)
        return pr, n, K, dict(jll=jll, log=lp, proba=pp, predict=am)

    def _score_checks(self, pr, xo, n, d):
        """Stages a subclass adds to check its input in the scoring program
        (CategoricalNB); what it returns goes to `_score_refusals`."""
        return None

    def _score_refusals(self, pr, d, chk):
        """Raises on what `_score_checks` found, after the run."""

    def predict(self, X):
        pr, n, K, o = self._scores(X, ("predict",))
        return decode_labels(self.classes_, pr.get_i32(o["predict"], n))

    def predict_proba(self, X):
        pr, n, K, o = self._scores(X, ("proba",))
        return pr.get(o["proba"], (n, K))

    def predict_log_proba(self, X):
        pr, n, K, o = self._scores(X, ("log",))
        return pr.get(o["log"], (n, K))

    def predict_joint_log_proba(self, X):
        pr, n, K, o = self._scores(X, ())
        return pr.get(o["jll"], (n, K))

    def score(self, X, y):
        pred = self.predict(X)
        truth = list(y.tolist() if hasattr(y, "tolist") else y)
        pred = list(pred.tolist() if hasattr(pred, "tolist") else pred)
        return sum(1 for a, b in zip(pred, truth) if a == b) / max(len(truth), 1)


def _nb_weights(pr, sample_weight, n):
    """sample_weight -> its arena offset (None when not given)."""
    if sample_weight is None:
        return None
    w = [float(v) for v in (sample_weight.tolist() if hasattr(sample_weight, "tolist") else sample_weight)]
    if len(w) != n:
        raise ValueError(f"mojolearn: sample_weight has {len(w)} entries, expected {n}")
    if any(v != v or v in (float("inf"), float("-inf")) for v in w):
        raise ValueError("mojolearn: sample_weight must be finite")
    return pr.put_list(w)


def _class_stats(pr, wo, total, xo, n, d, yo, K, cnt, mean, var, sums):
    """class_stats, or its weighted form when a sample_weight offset is given;
    in x_prep/blocked.mojo's blocked order when `_blocked()` (offsets
    _NONE are not written)."""
    if _blocked():
        nb = (n + _XB - 1) // _XB
        w = _NONE if wo is None else wo
        ps, pc, cn = pr.work(nb * K * d), pr.work(nb * K * d), pr.work(K * d)
        if var != _NONE and mean == _NONE:
            mean = pr.work(K * d)
        pr.stage("csb_part", nb * K * d, xo, n, d, yo, K, ps, pc, nb, w)
        pr.stage("csb_fold", K * d, ps, pc, nb, K, d, cn, cnt, mean, sums, w)
        if var != _NONE:
            pr.stage("csb_ss", nb * K * d, xo, n, d, yo, K, mean, cn, ps, nb, w)
            pr.stage("csb_var", K * d, ps, nb, K, d, cn, var)
        return
    if wo is None:
        pr.stage("class_stats", total, xo, n, d, yo, K, cnt, mean, var, sums)
    else:
        pr.stage("class_stats_w", total, xo, n, d, yo, K, cnt, mean, var, sums, wo)


def _given_priors(values, K, who, check_sum=False):
    """A user prior list as floats, checked as the reference checks it."""
    vals = [float(v) for v in (values.tolist() if hasattr(values, "tolist") else values)]
    if len(vals) != K:
        raise ValueError(f"mojolearn: {who}: number of priors must match number of classes")
    if any(v < 0 for v in vals):
        raise ValueError(f"mojolearn: {who}: priors must be non-negative")
    if check_sum and abs(sum(vals) - 1.0) > 1e-8 * max(1.0, abs(sum(vals))) and \
            abs(sum(vals) - 1.0) > 1e-5:
        raise ValueError(f"mojolearn: {who}: the sum of the priors should be 1")
    return vals


def _partial_codes(est, y, classes, n):
    """sklearn `_check_partial_fit_first_call` and the batch's class codes:
    the first call (no classes_ yet) needs `classes`, later ones may repeat
    them only unchanged; a label outside classes_ is refused."""
    first = getattr(est, "classes_", None) is None
    if first and classes is None:
        raise ValueError("mojolearn: classes must be passed on the first call to partial_fit.")
    if classes is not None:
        cl, _ = sorted_classes(flatten_labels(classes))
        if not first and list(cl) != list(est.classes_):
            raise ValueError(f"mojolearn: `classes={cl}` is not the same as on last call to partial_fit, was: "
                             f"{est.classes_}")
        if first:
            est.classes_ = cl
    labels = flatten_labels(y)
    if len(labels) != n:
        raise ValueError("mojolearn: X and y have different numbers of rows")
    index = {c: i for i, c in enumerate(est.classes_)}
    bad = sorted({repr(v) for v in labels if v not in index})
    if bad:
        raise ValueError(f"mojolearn: The target label(s) {bad} in y do not exist in the initial classes "
                         f"{est.classes_}")
    return first, Array.from_list([index[v] for v in labels], "<i4")


def _copy_block(pr, src, rows, cols):
    """A bit-for-bit copy of a (rows, cols) block (gather_cols over every column)."""
    out = pr.alloc(rows * cols)
    pr.stage("gather_cols", rows * cols, src, rows, cols, pr.put_list(list(range(cols))), cols, out)
    return out


def _check_alpha(est):
    if not isinstance(est.alpha, numbers.Real) or not est.alpha > 0:
        raise NotImplementedError(f"mojolearn: {type(est).__name__} needs alpha > 0 "
                                  "(alpha = 0 makes log(0) terms)")


class GaussianNB(_Classifier):
    """sklearn.naive_bayes.GaussianNB (fit, predict, predict_proba,
    predict_log_proba): per-class mean and population variance plus
    var_smoothing * the largest feature variance, float32; `priors` as the
    reference checks them; sample_weight weights the class means, variances
    and counts (numpy `average`), as the reference. partial_fit merges each
    batch into the running counts, means and variances (the reference's
    `_update_mean_variance`, Chan's pairwise rule) and adds the batch's
    epsilon to the merged variance; the running variance is kept without
    epsilon (the reference subtracts the NEW batch's epsilon from a variance
    that holds the old one, see naive_bayes/NOT_IMPLEMENTED.tsv)."""
    _parameters = ("priors", "var_smoothing")

    def __init__(self, *, priors=None, var_smoothing=1e-9):
        self.priors = priors
        self.var_smoothing = var_smoothing

    def fit(self, X, y, sample_weight=None):
        arr = _x2d(X)
        n, d = arr.shape
        codes = self._encode_y(y, n)
        K = len(self.classes_)
        mode = _mode()
        pr = _Prog()
        xo = pr.put(arr)
        yo = pr.put_codes(codes)
        st = pr.alloc(6 * d)
        vs = pr.put_scalar(self.var_smoothing)
        eps = pr.alloc(1)
        cnt, theta, var, prior, const = pr.alloc(K), pr.alloc(K * d), pr.alloc(K * d), pr.alloc(K), pr.alloc(K)
        wo = _nb_weights(pr, sample_weight, n)
        _col_stats(pr, xo, n, d, st)
        pr.stage("gnb_eps", 1, st + 2 * d, d, eps, vs)
        _class_stats(pr, wo, K * d, xo, n, d, yo, K, cnt, theta, var, _NONE)
        raw = _copy_block(pr, var, K, d)
        given = _NONE
        if self.priors is not None:
            given = pr.put_list(_given_priors(self.priors, K, "GaussianNB", check_sum=True))
        elif wo is not None:
            # class_count_ / class_count_.sum() over the weighted counts
            given = pr.alloc(K)
            pr.stage("lda_prep", 1, cnt, theta, K, d, n, given, pr.alloc(d), 2, cnt)
        pr.stage("gnb_params", K, cnt, var, K, d, n, eps, prior, const, 1 if given != _NONE else 0,
                 given if given != _NONE else 0)
        pr.run(mode)
        self.theta_, self.var_ = pr.get(theta, (K, d)), pr.get(var, (K, d))
        self.class_count_, self.class_prior_ = pr.get(cnt, K), pr.get(prior, K)
        self.epsilon_ = pr.values(eps, 1)[0]
        self._const = pr.get(const, K)
        self._raw_var = pr.get(raw, (K, d))
        self.numeric_mode_, self.n_features_in_ = mode, d
        return self

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        arr = _x2d(X)
        n, d = arr.shape
        first, codes = _partial_codes(self, y, classes, n)
        K = len(self.classes_)
        if first:
            mode = _mode()
            zk, zkd = [0.0] * K, [0.0] * (K * d)
        else:
            self._check_width(arr)
            mode = self.numeric_mode_
            zk, zkd = None, None
        pr = _Prog()
        xo = pr.put(arr)
        yo = pr.put_codes(codes)
        st = pr.alloc(6 * d)
        vs = pr.put_scalar(self.var_smoothing)
        eps = pr.alloc(1)
        bc, bm, bv = pr.alloc(K), pr.alloc(K * d), pr.alloc(K * d)
        wo = _nb_weights(pr, sample_weight, n)
        _col_stats(pr, xo, n, d, st)
        pr.stage("gnb_eps", 1, st + 2 * d, d, eps, vs)
        _class_stats(pr, wo, K * d, xo, n, d, yo, K, bc, bm, bv, _NONE)
        oc = pr.put_list(zk) if first else pr.put(self.class_count_)
        om = pr.put_list(zkd) if first else pr.put(self.theta_)
        ov = pr.put_list(zkd) if first else pr.put(self._raw_var)
        cnt, theta, raw = pr.alloc(K), pr.alloc(K * d), pr.alloc(K * d)
        pr.stage("gnb_merge", K * d, oc, om, ov, bc, bm, bv, K, d, cnt, theta, raw)
        var = _copy_block(pr, raw, K, d)
        if self.priors is not None:
            given = pr.put_list(_given_priors(self.priors, K, "GaussianNB", check_sum=True))
        else:
            given = pr.alloc(K)
            pr.stage("lda_prep", 1, cnt, theta, K, d, n, given, pr.alloc(d), 2, cnt)
        prior, const = pr.alloc(K), pr.alloc(K)
        pr.stage("gnb_params", K, cnt, var, K, d, 1, eps, prior, const, 1, given)
        pr.run(mode)
        self.theta_, self.var_ = pr.get(theta, (K, d)), pr.get(var, (K, d))
        self.class_count_, self.class_prior_ = pr.get(cnt, K), pr.get(prior, K)
        self.epsilon_ = pr.values(eps, 1)[0]
        self._const, self._raw_var = pr.get(const, K), pr.get(raw, (K, d))
        self.numeric_mode_, self.n_features_in_ = mode, d
        return self

    def _jll_stages(self, pr, xo, n, d, out):
        K = len(self.classes_)
        th, va, co = pr.put(self.theta_), pr.put(self.var_), pr.put(self._const)
        pr.stage("gnb_jll", n * K, xo, n, d, th, va, co, K, out)


def _check_nonnegative(pr_values, who):
    if any(v < 0 for v in pr_values):
        raise ValueError(f"mojolearn: Negative values in data passed to {who}")


class _DiscreteNB(_Classifier):
    def _fit_counts(self, X, y, binarize=None, sample_weight=None):
        arr = _x2d(X)
        n, d = arr.shape
        codes = self._encode_y(y, n)
        K = len(self.classes_)
        mode = _mode()
        pr = _Prog()
        xo = pr.put(arr)
        wo = _nb_weights(pr, sample_weight, n)
        if binarize is not None:
            thr = pr.put_scalar(binarize)
            xb = pr.work(n * d)
            pr.stage("binarize", n * d, xo, n * d, thr, xb)
            xo = xb
        yo = pr.put_codes(codes)
        st = pr.alloc(6 * d)
        cnt, fc = pr.alloc(K), pr.alloc(K * d)
        clp = pr.alloc(K)
        _col_stats(pr, xo, n, d, st, var=False)
        _class_stats(pr, wo, K * d, xo, n, d, yo, K, cnt, _NONE, _NONE, fc)
        self._prior_stages(pr, K, cnt, clp)
        return pr, mode, n, d, K, st, cnt, fc, clp

    def _finish_counts(self, pr, mode, d, K, cnt, fc, clp):
        self.class_count_, self.feature_count_ = pr.get(cnt, K), pr.get(fc, (K, d))
        self.class_log_prior_ = pr.get(clp, K)
        self.numeric_mode_, self.n_features_in_ = mode, d

    def fit(self, X, y, sample_weight=None):
        _check_alpha(self)
        return self._params(*self._fit_counts(X, y, getattr(self, "binarize", None), sample_weight))

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """The reference's `_BaseDiscreteNB.partial_fit`: the batch's class and
        feature counts added to the running ones, then the log probabilities
        and the class log prior recomputed from the sums."""
        _check_alpha(self)
        arr = _x2d(X)
        n, d = arr.shape
        first, codes = _partial_codes(self, y, classes, n)
        K = len(self.classes_)
        if first:
            mode = _mode()
        else:
            self._check_width(arr)
            mode = self.numeric_mode_
        pr = _Prog()
        xo = pr.put(arr)
        wo = _nb_weights(pr, sample_weight, n)
        if getattr(self, "binarize", None) is not None:
            xb = pr.work(n * d)
            pr.stage("binarize", n * d, xo, n * d, pr.put_scalar(self.binarize), xb)
            xo = xb
        yo = pr.put_codes(codes)
        st = pr.alloc(6 * d)
        cnt, fc, clp = pr.alloc(K), pr.alloc(K * d), pr.alloc(K)
        _col_stats(pr, xo, n, d, st, var=False)
        if first:
            _class_stats(pr, wo, K * d, xo, n, d, yo, K, cnt, _NONE, _NONE, fc)
        else:
            bc, bf = pr.alloc(K), pr.alloc(K * d)
            _class_stats(pr, wo, K * d, xo, n, d, yo, K, bc, _NONE, _NONE, bf)
            pr.stage("add_arrays", K, pr.put(self.class_count_), bc, cnt)
            pr.stage("add_arrays", K * d, pr.put(self.feature_count_), bf, fc)
        self._prior_stages(pr, K, cnt, clp)
        return self._params(pr, mode, n, d, K, st, cnt, fc, clp)

    def _prior_stages(self, pr, K, cnt, clp):
        if getattr(self, "class_prior", None) is not None:
            po = pr.put_list(_given_priors(self.class_prior, K, type(self).__name__))
            pr.stage("log", K, po, clp)
        elif self.fit_prior:
            pr.stage("class_log_prior", K, cnt, K, clp)
        else:
            ones = pr.put_list([1.0] * K)
            pr.stage("class_log_prior", K, ones, K, clp)


class MultinomialNB(_DiscreteNB):
    """sklearn.naive_bayes.MultinomialNB, float32; alpha > 0 required.
    class_prior as given (its log); sample_weight weights the counts, as the
    reference; partial_fit adds each batch's counts, as the reference."""
    _parameters = ("alpha", "force_alpha", "fit_prior", "class_prior")

    def __init__(self, *, alpha=1.0, force_alpha=True, fit_prior=True, class_prior=None):
        self.alpha = alpha
        self.force_alpha = force_alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior

    def _params(self, pr, mode, n, d, K, st, cnt, fc, clp):
        a = pr.put_scalar(self.alpha)
        flp = pr.alloc(K * d)
        pr.stage("mnb_params", K, fc, K, d, a, flp)
        pr.run(mode)
        _check_nonnegative(pr.values(st + 3 * d, d), "MultinomialNB (input X)")
        self._finish_counts(pr, mode, d, K, cnt, fc, clp)
        self.feature_log_prob_ = pr.get(flp, (K, d))
        return self

    def _jll_stages(self, pr, xo, n, d, out):
        K = len(self.classes_)
        w, b = pr.put(self.feature_log_prob_), pr.put(self.class_log_prior_)
        pr.stage("matmul", n * K, xo, d, 1, w, 1, d, out, K, d, b, _NONE)


class BernoulliNB(_DiscreteNB):
    """sklearn.naive_bayes.BernoulliNB, float32 (X binarized at `binarize`
    unless it is None); alpha > 0 required; class_prior as given (its log);
    sample_weight weights the counts, as the reference; partial_fit adds each
    batch's counts, as the reference."""
    _parameters = ("alpha", "force_alpha", "binarize", "fit_prior", "class_prior")

    def __init__(self, *, alpha=1.0, force_alpha=True, binarize=0.0, fit_prior=True, class_prior=None):
        self.alpha = alpha
        self.force_alpha = force_alpha
        self.binarize = binarize
        self.fit_prior = fit_prior
        self.class_prior = class_prior

    def _params(self, pr, mode, n, d, K, st, cnt, fc, clp):
        a = pr.put_scalar(self.alpha)
        flp, w, bias = pr.alloc(K * d), pr.alloc(K * d), pr.alloc(K)
        pr.stage("bnb_params", K, fc, cnt, K, d, a, clp, flp, w, bias)
        pr.run(mode)
        self._finish_counts(pr, mode, d, K, cnt, fc, clp)
        self.feature_log_prob_ = pr.get(flp, (K, d))
        self._w, self._bias = pr.get(w, (K, d)), pr.get(bias, K)
        return self

    def _jll_stages(self, pr, xo, n, d, out):
        K = len(self.classes_)
        if self.binarize is not None:
            thr = pr.put_scalar(self.binarize)
            xb = pr.work(n * d)
            pr.stage("binarize", n * d, xo, n * d, thr, xb)
            xo = xb
        w, b = pr.put(self._w), pr.put(self._bias)
        pr.stage("matmul", n * K, xo, d, 1, w, 1, d, out, K, d, b, _NONE)


# ---------------------------------------------------------------- discriminant analysis
def _binary_difference(pr, src, rows, K, d_cols, out):
    """out[i, j] = src[i*? ...]: row 1 minus row 0 of a (2 x d_cols) block
    (rows=1) or column 1 minus column 0 of an (rows x 2) block, as one
    matmul with the vector [-1, 1] (-a + b is b - a exactly)."""
    w = pr.put_list([-1.0, 1.0])
    if rows == 1:
        pr.stage("matmul", d_cols, w, 0, 1, src, d_cols, 1, out, d_cols, 2, _NONE, _NONE)
    else:
        pr.stage("matmul", rows, src, K, 1, w, 1, 0, out, 1, 2, _NONE, _NONE)


def _class_counts(codes, K):
    counts = [0] * K
    for c in codes.tolist():
        counts[c] += 1
    return counts


def _shrinkage_value(shrinkage):
    """None, -1.0 for 'auto', or the constant in [0, 1]."""
    if shrinkage is None:
        return None
    if isinstance(shrinkage, str):
        if shrinkage != "auto":
            raise ValueError(f"mojolearn: invalid shrinkage {shrinkage!r}")
        return -1.0
    if not isinstance(shrinkage, numbers.Real) or not 0 <= shrinkage <= 1:
        raise ValueError(f"mojolearn: shrinkage must be 'auto' or a float in [0, 1], got {shrinkage!r}")
    return float(shrinkage)


def _estimator_covs(est, arr, codes, K, who):
    """covariance_estimator: the reference's `_cov(X_k, covariance_estimator=est)`
    for each class k (codes None: every row, one block). `est.fit` runs in
    Python on the class's float32 rows (a mojolearn Array); its covariance_
    is read as float32. Returns the (K, d, d) blocks as one flat list."""
    n, d = arr.shape
    code_list = [0] * n if codes is None else [int(c) for c in codes.tolist()]
    out = []
    for k in range(K):
        rows = [i for i, c in enumerate(code_list) if c == k]
        est.fit(_gather_rows(arr, rows))
        if not hasattr(est, "covariance_"):
            raise ValueError(f"mojolearn: {type(est).__name__} does not have a covariance_ attribute")
        cov = est.covariance_
        flat = flatten_labels(cov.tolist() if hasattr(cov, "tolist") else cov)
        if len(flat) != d * d:
            raise ValueError(f"mojolearn: {who}: covariance_ of {type(est).__name__} is not ({d}, {d})")
        out.extend(float(v) for v in flat)
    return out


def _check_estimator_shrinkage(est, shrinkage):
    if est is not None and shrinkage is not None and shrinkage != 0:
        raise ValueError("mojolearn: covariance_estimator and shrinkage parameters are not None. "
                         "Only one of the two can be set.")


def _lda_cov_blocks(pr, xo, n, d, yo, K, mean, var, cnt, shr, given=None):
    """K class covariances (divisor the class count), shrunk as the reference's
    `_cov(X_k, shrinkage)`, or the covariance_estimator's blocks as `given`.
    Returns the (K, d, d) offset."""
    if given is not None:
        return pr.put_list(given)
    cov = pr.alloc(K * d * d)
    pr.stage("qda_cov", K * d * d, xo, n, d, yo, mean, cnt, cov)
    if shr is not None:
        pr.stage("da_shrink", K, xo, n, d, yo, mean, var, cnt, cov, pr.put_scalar(shr), pr.alloc(K))
    return cov


def _lda_class_cov(pr, xo, n, d, yo, K, mean, cnt, priors, shr, var=_NONE, given=None):
    """The reference's `_class_cov`: sum_k priors_k _cov(X_k, shrinkage). Returns the (d, d) offset."""
    cov = _lda_cov_blocks(pr, xo, n, d, yo, K, mean, var, cnt, shr, given)
    sw = pr.alloc(d * d)
    pr.stage("da_pool", d * d, cov, K, d, priors, sw, _NONE, _NONE)
    return sw


def _lda_binary(pr, coef, inter, K, d):
    """The binary case's coef_[1] - coef_[0] and intercept difference."""
    cd, ci = pr.alloc(d), pr.alloc(1)
    if K == 2:
        _binary_difference(pr, coef, 1, K, d, cd)
        pr.stage("matmul", 1, pr.put_list([-1.0, 1.0]), 0, 1, inter, 1, 0, ci, 1, 2, _NONE, _NONE)
    return cd, ci


class LinearDiscriminantAnalysis(_Classifier):
    """sklearn.discriminant_analysis.LinearDiscriminantAnalysis, solver 'svd'
    (the default): the reference's two SVDs are symmetric eigendecompositions
    of the Gram matrices (cyclic Jacobi, x_prep/eigh.mojo), so `scalings_` and
    `transform` match the reference up to each component's sign and the
    decision function matches it outright. Solver 'lsqr' (the pooled class
    covariance, then `lstsq`'s minimum-norm solve through its
    eigendecomposition) and 'eigen' (the generalised eigenproblem Sb v = e Sw v
    as Sw^-1/2 Sb Sw^-1/2), both with shrinkage None, 'auto' (Ledoit-Wolf on
    standardised classes, as the reference) or a constant; store_covariance.
    Float32; priors as the reference takes them (renormalised when they do
    not sum to 1). covariance_estimator (solver 'lsqr' / 'eigen', as the
    reference): any object with `fit` and `covariance_`, fitted in Python on
    each class's rows (and, for 'eigen', on every row for the total scatter);
    its covariances enter the device solve as float32."""
    _parameters = ("solver", "shrinkage", "priors", "n_components", "store_covariance", "tol",
                   "covariance_estimator")

    def __init__(self, solver="svd", shrinkage=None, priors=None, n_components=None, store_covariance=False,
                 tol=1e-4, covariance_estimator=None):
        self.solver = solver
        self.shrinkage = shrinkage
        self.priors = priors
        self.n_components = n_components
        self.store_covariance = store_covariance
        self.tol = tol
        self.covariance_estimator = covariance_estimator

    def fit(self, X, y):
        if self.solver not in ("svd", "lsqr", "eigen"):
            raise ValueError(f"mojolearn: invalid solver {self.solver!r}")
        if self.solver == "svd" and self.shrinkage is not None:
            raise NotImplementedError("mojolearn: shrinkage not supported with 'svd' solver.")
        if self.covariance_estimator is not None and self.solver == "svd":
            raise ValueError("mojolearn: covariance estimator is not supported with svd solver. Try another solver")
        _check_estimator_shrinkage(self.covariance_estimator, self.shrinkage)
        shr = None if self.covariance_estimator is not None else _shrinkage_value(self.shrinkage)
        arr = _x2d(X)
        n, d = arr.shape
        codes = self._encode_y(y, n)
        K = len(self.classes_)
        if K < 2 or n <= K:
            raise ValueError("mojolearn: LinearDiscriminantAnalysis needs at least two classes and more "
                             "samples than classes")
        maxc = min(K - 1, d)
        if self.n_components is not None and self.n_components > maxc:
            raise ValueError("mojolearn: n_components cannot be larger than min(n_features, n_classes - 1)")
        mode = _mode()
        if self.solver != "svd":
            return self._fit_cov_solver(arr, codes, K, d, n, mode, shr)
        pr = _Prog()
        xo = pr.put(arr)
        yo = pr.put_codes(codes)
        cnt, mean, priors, xbar = pr.alloc(K), pr.alloc(K * d), pr.alloc(K), pr.alloc(d)
        z, stz, std, w, z2 = pr.work(n * d), pr.alloc(6 * d), pr.alloc(d), pr.alloc(d), pr.work(n * d)
        g, e1, v1 = pr.alloc(d * d), pr.alloc(d), pr.alloc(d * d)
        meta = pr.put_list([self.tol, 0.0, 0.0], inout=True)        # lda_stage2 writes the ranks into it
        scal1, g2, ms = pr.alloc(d * d), pr.alloc(d * d), pr.alloc(K * d)
        e2, v2 = pr.alloc(d), pr.alloc(d * d)
        scal, coef, inter, evr, tmp = pr.alloc(d * d), pr.alloc(K * d), pr.alloc(K), pr.alloc(d), pr.alloc(K * d)
        pr.stage("class_stats", K * d, xo, n, d, yo, K, cnt, mean, _NONE, _NONE)
        gflag, gofs = 0, 0
        if self.priors is not None:
            pv = _given_priors(self.priors, K, "LinearDiscriminantAnalysis")
            gflag, gofs = (2 if abs(sum(pv) - 1.0) > 1e-5 else 1), pr.put_list(pv)
        pr.stage("lda_prep", 1, cnt, mean, K, d, n, priors, xbar, gflag, gofs)
        pr.stage("center_rows", n * d, xo, n, d, mean, yo, _NONE, z)
        pr.stage("col_stats", d, z, n, d, stz)
        pr.stage("lda_w", d, stz + 2 * d, d, n, K, std, w)
        pr.stage("center_rows", n * d, xo, n, d, mean, yo, w, z2)
        pr.stage("matmul", d * d, z2, 1, d, z2, d, 1, g, d, n, _NONE, _NONE)
        pr.stage("eigh", 1, g, d, 0, e1, v1)
        pr.stage("lda_stage2", 1, e1, v1, std, mean, xbar, priors, K, d, n, meta, scal1, g2, ms)
        pr.stage("eigh", 1, g2, d, 0, e2, v2)
        pr.stage("lda_stage3", 1, e2, v2, scal1, mean, xbar, priors, K, d, meta, scal, coef, inter, evr, tmp)
        cd, ci = _lda_binary(pr, coef, inter, K, d)
        sw = _lda_class_cov(pr, xo, n, d, yo, K, mean, cnt, priors, None) if self.store_covariance else None
        pr.run(mode)
        if sw is not None:
            self.covariance_ = pr.get(sw, (d, d))
        rank2 = int(pr.values(meta + 2, 1)[0])
        self._rank = rank2
        self.means_, self.priors_, self.xbar_ = pr.get(mean, (K, d)), pr.get(priors, K), pr.get(xbar, d)
        full = pr.get(scal, (d, d))
        self._scal_full = full
        self.scalings_ = Array.from_list([row[:rank2] for row in full.tolist()], "<f4") if rank2 else \
            Array((d, 0), "<f4")
        self._coef, self._inter = pr.get(coef, (K, d)), pr.get(inter, K)
        if K == 2:
            self.coef_, self.intercept_ = pr.get(cd, (1, d)), pr.get(ci, 1)
        else:
            self.coef_, self.intercept_ = self._coef, self._inter
        self._max_components = maxc if self.n_components is None else int(self.n_components)
        self.explained_variance_ratio_ = pr.get(evr, self._max_components)
        self.numeric_mode_, self.n_features_in_ = mode, d
        return self

    def _fit_cov_solver(self, arr, codes, K, d, n, mode, shr):
        """solver 'lsqr' / 'eigen' (the reference's `_solve_lstsq` / `_solve_eigen`)."""
        pr = _Prog()
        xo = pr.put(arr)
        yo = pr.put_codes(codes)
        cnt, mean, priors, xbar = pr.alloc(K), pr.alloc(K * d), pr.alloc(K), pr.alloc(d)
        var = pr.alloc(K * d) if shr is not None else _NONE
        pr.stage("class_stats", K * d, xo, n, d, yo, K, cnt, mean, var, _NONE)
        gflag, gofs = 0, 0
        if self.priors is not None:
            pv = _given_priors(self.priors, K, "LinearDiscriminantAnalysis")
            gflag, gofs = (2 if abs(sum(pv) - 1.0) > 1e-5 else 1), pr.put_list(pv)
        pr.stage("lda_prep", 1, cnt, mean, K, d, n, priors, xbar, gflag, gofs)
        eigen = self.solver == "eigen"
        est = self.covariance_estimator
        tot = None
        if eigen:
            y0 = pr.put_list([0.0] * n)
            c1, m1 = pr.alloc(1), pr.alloc(d)
            v1 = pr.alloc(d) if shr is not None else _NONE
            pr.stage("class_stats", d, xo, n, d, y0, 1, c1, m1, v1, _NONE)
            gt = None if est is None else _estimator_covs(est, arr, None, 1, "LinearDiscriminantAnalysis")
            tot = _lda_cov_blocks(pr, xo, n, d, y0, 1, m1, v1, c1, shr, gt)
        gk = None if est is None else _estimator_covs(est, arr, codes, K, "LinearDiscriminantAnalysis")
        sw = _lda_class_cov(pr, xo, n, d, yo, K, mean, cnt, priors, shr, var, gk)
        sb = pr.alloc(d * d) if eigen else None
        if eigen:
            pr.stage("da_pool", d * d, sw, 1, d, pr.put_list([1.0]), pr.alloc(d * d), tot, sb)
        swc = pr.alloc(d * d)
        pr.stage("da_pool", d * d, sw, 1, d, pr.put_list([1.0]), swc, _NONE, _NONE)
        e, v = pr.alloc(d), pr.alloc(d * d)
        pr.stage("eigh", 1, swc, d, 0, e, v)
        coef, inter = pr.alloc(K * d), pr.alloc(K)
        mc = min(K - 1, d) if self.n_components is None else int(self.n_components)
        if not eigen:
            P = pr.alloc(d * d)
            pr.stage("sym_fn", d * d, e, v, d, 0, P)
            pr.stage("matmul", K * d, mean, d, 1, P, 1, d, coef, d, d, _NONE, _NONE)
        else:
            W, T, C = pr.alloc(d * d), pr.alloc(d * d), pr.alloc(d * d)
            pr.stage("sym_fn", d * d, e, v, d, 1, W)
            pr.stage("matmul", d * d, W, d, 1, sb, d, 1, T, d, d, _NONE, _NONE)
            pr.stage("matmul", d * d, T, d, 1, W, d, 1, C, d, d, _NONE, _NONE)
            e2, y2, ev, evr = pr.alloc(d), pr.alloc(d * d), pr.alloc(d * d), pr.alloc(d)
            pr.stage("eigh", 1, C, d, 0, e2, y2)
            pr.stage("matmul", d * d, W, d, 1, y2, d, 1, ev, d, d, _NONE, _NONE)
            pr.stage("evr", 1, e2, d, evr)
            t1 = pr.alloc(K * d)
            pr.stage("matmul", K * d, mean, d, 1, ev, d, 1, t1, d, d, _NONE, _NONE)
            pr.stage("matmul", K * d, t1, d, 1, ev, 1, d, coef, d, d, _NONE, _NONE)
        pr.stage("da_intercept", K, mean, coef, priors, d, inter)
        cd, ci = _lda_binary(pr, coef, inter, K, d)
        pr.run(mode)
        if eigen and min(pr.values(e, d)) <= 0:
            raise ValueError("mojolearn: the within-class covariance is not positive definite "
                             "(the reference's eigh(Sb, Sw) fails); set shrinkage")
        self.means_, self.priors_, self.xbar_ = pr.get(mean, (K, d)), pr.get(priors, K), pr.get(xbar, d)
        self.covariance_ = pr.get(sw, (d, d))
        self._coef, self._inter = pr.get(coef, (K, d)), pr.get(inter, K)
        if K == 2:
            self.coef_, self.intercept_ = pr.get(cd, (1, d)), pr.get(ci, 1)
        else:
            self.coef_, self.intercept_ = self._coef, self._inter
        self._max_components = mc
        if eigen:
            self.scalings_ = self._scal_full = pr.get(ev, (d, d))
            self._rank = d
            self.explained_variance_ratio_ = pr.get(evr, mc)
        self.numeric_mode_, self.n_features_in_ = mode, d
        return self

    def _jll_stages(self, pr, xo, n, d, out):
        K = len(self.classes_)
        c, b = pr.put(self._coef), pr.put(self._inter)
        pr.stage("matmul", n * K, xo, d, 1, c, 1, d, out, K, d, b, _NONE)

    def decision_function(self, X):
        pr, n, K, o = self._scores(X, ())
        if K == 2:
            return self._pair(X, o, pr, n)
        return pr.get(o["jll"], (n, K))

    def _pair(self, X, o, pr, n):
        arr = _x2d(X)
        q = _Prog()
        xo = q.put(arr)
        c, b = q.put(self.coef_), q.put(self.intercept_)
        out = q.alloc(n)
        q.stage("matmul", n, xo, arr.shape[1], 1, c, 1, 0, out, 1, arr.shape[1], b, _NONE)
        q.run(self.numeric_mode_)
        return q.get(out, n)

    def transform(self, X):
        self._check_fitted()
        arr = _x2d(X)
        self._check_width(arr)
        n, d = arr.shape
        if self.solver == "lsqr":
            raise NotImplementedError("mojolearn: transform not implemented for 'lsqr' solver (use 'svd' or 'eigen').")
        mc = min(self._max_components, self._rank)
        pr = _Prog()
        xo = pr.put(arr)
        xb, sc = pr.put(self.xbar_), pr.put(self._scal_full)
        cen, out = pr.alloc(n * d), pr.alloc(n * max(mc, 1))
        if self.solver == "eigen":
            cen = xo
        else:
            pr.stage("center_rows", n * d, xo, n, d, xb, _NONE, _NONE, cen)
        pr.stage("matmul", n * mc, cen, d, 1, sc, d, 1, out, mc, d, _NONE, _NONE)
        pr.run(self.numeric_mode_)
        return pr.get(out, (n, mc))

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)


class QuadraticDiscriminantAnalysis(_Classifier):
    """sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis (1.9): per
    class, the eigendecomposition of the class covariance (divisor n_k) stands
    in for the reference's SVD of the centred class rows (same S^2 / n_k,
    vectors up to sign). solver 'eigen' takes the reference's `_cov(X_k,
    shrinkage)` (None, 'auto' Ledoit-Wolf or a constant) and ignores
    reg_param, as the reference does. A class whose scalings are not all
    above `tol` is refused, as the reference refuses it. store_covariance
    keeps covariance_ (svd: V diag(scalings) V^T, as the reference forms it).
    Float32; priors as given. covariance_estimator (solver 'eigen' only, as
    the reference): any object with `fit` and `covariance_`, fitted in Python
    on each class's rows; its covariances enter the device eigh as float32."""
    _parameters = ("solver", "shrinkage", "priors", "reg_param", "store_covariance", "tol", "covariance_estimator")

    def __init__(self, *, solver="svd", shrinkage=None, priors=None, reg_param=0.0, store_covariance=False,
                 tol=1e-4, covariance_estimator=None):
        self.solver = solver
        self.shrinkage = shrinkage
        self.priors = priors
        self.reg_param = reg_param
        self.store_covariance = store_covariance
        self.tol = tol
        self.covariance_estimator = covariance_estimator

    def fit(self, X, y):
        if self.solver not in ("svd", "eigen"):
            raise ValueError(f"mojolearn: invalid solver {self.solver!r}")
        if self.solver == "svd" and self.shrinkage is not None:
            raise NotImplementedError("mojolearn: shrinkage not supported with 'svd' solver.")
        if self.covariance_estimator is not None and self.solver == "svd":
            raise ValueError("mojolearn: covariance_estimator is not supported with solver='svd'. "
                             "Try solver='eigen' instead.")
        _check_estimator_shrinkage(self.covariance_estimator, self.shrinkage)
        est = self.covariance_estimator
        shr = None if est is not None else _shrinkage_value(self.shrinkage)
        eigen = self.solver == "eigen"
        arr = _x2d(X)
        n, d = arr.shape
        codes = self._encode_y(y, n)
        K = len(self.classes_)
        if K < 2:
            raise ValueError("mojolearn: QuadraticDiscriminantAnalysis needs at least two classes")
        if min(_class_counts(codes, K)) < 2:
            raise ValueError("mojolearn: y has only 1 sample in a class, covariance is ill defined")
        mode = _mode()
        pr = _Prog()
        xo = pr.put(arr)
        yo = pr.put_codes(codes)
        cnt, mean, priors, xbar = pr.alloc(K), pr.alloc(K * d), pr.alloc(K), pr.alloc(d)
        cov, ev, evec = pr.alloc(K * d * d), pr.alloc(K * d), pr.alloc(K * d * d)
        reg = pr.put_scalar(0.0 if eigen else self.reg_param)
        rot, logc, s2 = pr.alloc(K * d * d), pr.alloc(K), pr.alloc(K * d)
        var = pr.alloc(K * d) if shr is not None else _NONE
        # the trailing 1: FAST keeps row-order class sums here (the tree sums did not pass
        # QuadraticDiscriminantAnalysis' paired quality check)
        pr.stage("class_stats", K * d, xo, n, d, yo, K, cnt, mean, var, _NONE, 1)
        gflag, gofs = 0, 0
        if self.priors is not None:
            gflag, gofs = 1, pr.put_list(_given_priors(self.priors, K, "QuadraticDiscriminantAnalysis"))
        pr.stage("lda_prep", 1, cnt, mean, K, d, n, priors, xbar, gflag, gofs)
        if est is not None:
            cov = pr.put_list(_estimator_covs(est, arr, codes, K, "QuadraticDiscriminantAnalysis"))
        else:
            pr.stage("qda_cov", K * d * d, xo, n, d, yo, mean, cnt, cov)
        if shr is not None:
            pr.stage("da_shrink", K, xo, n, d, yo, mean, var, cnt, cov, pr.put_scalar(shr), pr.alloc(K))
        keep = pr.alloc(K * d * d) if (self.store_covariance and eigen) else None
        if keep is not None:
            one = pr.put_list([1.0])
            for k in range(K):
                pr.stage("da_pool", d * d, cov + k * d * d, 1, d, one, keep + k * d * d, _NONE, _NONE)
        pr.stage("eigh", K, cov, d, d * d, ev, evec)
        pr.stage("qda_prep", K, ev, evec, K, d, reg, cnt, n, rot, logc, s2, gflag, gofs)
        if self.store_covariance and not eigen:
            keep = pr.alloc(K * d * d)
            pr.stage("sym_fn", K * d * d, s2, evec, d, 2, keep)
        pr.run(mode)
        s2v = pr.values(s2, K * d)
        for k in range(K):
            if sum(1 for v in s2v[k * d:(k + 1) * d] if v > self.tol) < d:
                raise ValueError(f"mojolearn: the covariance matrix of class {self.classes_[k]!r} is not full "
                                 f"rank. Increase the value of `{'shrinkage' if eigen else 'reg_param'}` to "
                                 "reduce the collinearity.")
        if keep is not None:
            self.covariance_ = [pr.get(keep + k * d * d, (d, d)) for k in range(K)]
        self.means_, self.priors_ = pr.get(mean, (K, d)), pr.get(priors, K)
        self.rotations_ = [pr.get(evec + k * d * d, (d, d)) for k in range(K)]
        self.scalings_ = [pr.get(s2 + k * d, d) for k in range(K)]
        self._rot, self._logc = pr.get(rot, K * d * d), pr.get(logc, K)
        self.numeric_mode_, self.n_features_in_ = mode, d
        return self

    def _jll_stages(self, pr, xo, n, d, out):
        K = len(self.classes_)
        m, r, lc = pr.put(self.means_), pr.put(self._rot), pr.put(self._logc)
        pr.stage("qda_dec", n * K, xo, n, d, m, r, lc, K, out)

    def decision_function(self, X):
        pr, n, K, o = self._scores(X, ())
        if K == 2:
            q = _Prog()
            src = q.put(pr.get(o["jll"], (n, K)))
            out = q.alloc(n)
            _binary_difference(q, src, n, K, 1, out)
            q.run(self.numeric_mode_)
            return q.get(out, n)
        return pr.get(o["jll"], (n, K))


# ---------------------------------------------------------------- additions: transformers
def _draw_without_replacement(n, k, seed):
    """k distinct rows of n, from a splitmix64 partial Fisher-Yates (integer
    arithmetic, the same on every machine); the reference draws numpy's."""
    perm = list(range(n))
    state = int(seed) & 0xFFFFFFFFFFFFFFFF
    for i in range(k):
        state, z = _splitmix64(state)
        j = i + z % (n - i)
        perm[i], perm[j] = perm[j], perm[i]
    return sorted(perm[:k])


class QuantileTransformer(_PrepBase):
    """sklearn.preprocessing.QuantileTransformer: per-column numpy linear
    percentiles of the non-NaN entries at n_quantiles evenly spaced
    references, then the reference's two-sided interpolation; output
    'uniform' or 'normal' (Acklam's inverse normal CDF, float32, clipped at
    the reference's +-5.1993). Above `subsample` rows the fit uses a
    without-replacement draw from `random_state` by splitmix64. NaN is kept.
    inverse_transform is the reference's (norm.cdf, Cephes ndtr in float32,
    for 'normal', then np.interp back onto the quantiles). Sparse input is
    refused."""
    _parameters = ("n_quantiles", "output_distribution", "ignore_implicit_zeros", "subsample", "random_state",
                   "copy")

    def __init__(self, *, n_quantiles=1000, output_distribution="uniform", ignore_implicit_zeros=False,
                 subsample=10_000, random_state=None, copy=True):
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
        self.ignore_implicit_zeros = ignore_implicit_zeros
        self.subsample = subsample
        self.random_state = random_state
        self.copy = copy

    def fit(self, X, y=None):
        if self.output_distribution not in ("uniform", "normal"):
            raise ValueError(f"mojolearn: invalid output_distribution {self.output_distribution!r}")
        arr = _x2d(X)
        n, d = arr.shape
        if self.subsample is not None and n > self.subsample:
            arr = _gather_rows(arr, _draw_without_replacement(
                n, int(self.subsample), 0 if self.random_state is None else int(self.random_state)))
            n = arr.shape[0]
        nq = max(1, min(int(self.n_quantiles), n))
        refs = [i / (nq - 1) if nq > 1 else 0.0 for i in range(nq)]
        mode = _mode()
        pr = _Prog()
        xo = pr.put(arr)
        # the references are an input no stage writes: the fitted attribute is
        # the array that went up (the same float32 words the arena held)
        refs_arr = Array._from_flat([float(v) for v in refs], (nq,), "<f4")
        so, st, qf = pr.work(n * d), pr.alloc(6 * d), pr.put(refs_arr)
        qo = pr.alloc(nq * d)
        pr.stage("sort_cols", d, xo, n, d, so, 0)
        pr.stage("col_stats", d, xo, n, d, st)
        pr.stage("quantile", nq * d, so, n, d, qf, nq, qo, st)
        pr.run(mode)
        self._q = pr.get(qo, nq * d)
        flat = pr.values(qo, nq * d)
        self.quantiles_ = Array.from_list([[flat[c * nq + j] for c in range(d)] for j in range(nq)], "<f4")
        self.references_ = refs_arr
        self.n_quantiles_, self.numeric_mode_, self.n_features_in_ = nq, mode, d
        return self

    def _apply(self, X, op):
        self._check_fitted()
        arr = _x2d(X)
        self._check_width(arr)
        n, d = arr.shape
        pr = _Prog()
        xo, qo, ro = pr.put(arr), pr.put(self._q), pr.put(self.references_)
        out = pr.output(n * d)
        pr.stage(op, n * d, xo, n, d, qo, self.n_quantiles_, ro,
                 1 if self.output_distribution == "normal" else 0, out)
        pr.run(self.numeric_mode_)
        return pr.get(out, (n, d))

    def transform(self, X):
        return self._apply(X, "qt_apply")

    def inverse_transform(self, X):
        return self._apply(X, "qt_inverse")


def _pt_spec_depth(n, d):
    """The device search's speculation depth (lane prep-apple2): each round
    evaluates the 2^S - 1 candidate points of the next S golden steps side by
    side. MOJOLEARN_XPREP_PT_SPEC = S (default 3: m4pro-b taxi 4.55 s staged,
    S=2 3.31, S=3 2.95, S=4 3.09; 0: one evaluation per fold, the staged
    search); the candidates' transforms (n*d words each) are capped at
    2^28 words. Every S gives the same lambdas. IDENTICAL only: FAST keeps the
    staged search, whose folds FAST runs as threadgroup trees
    (x_prep/fastred.mojo), already short."""
    try:
        s = int(os.environ.get("MOJOLEARN_XPREP_PT_SPEC", "3"))
    except ValueError:
        s = 3
    s = max(0, min(s, 6))
    while s > 1 and (2 ** s - 1) * n * d > 2 ** 28:
        s -= 1
    # the first round always evaluates the search's two opening points
    # (PowerTransformer.fit: m = 2 at k0 = 0), so the candidates' buffer holds
    # at least two transforms; when even two exceed the cap, the staged search
    # (lane gap-board-refusals: Istella 1,000,000 x 220 reached S = 1 with a
    # one-candidate buffer and the opening round wrote past it, an illegal
    # address on the L40S and the MI300X, 0.8.34 board)
    if s == 1 and 2 * n * d > 2 ** 28:
        s = 0
    return s


class PowerTransformer(_PrepBase):
    """sklearn.preprocessing.PowerTransformer: 'yeo-johnson' (default) or
    'box-cox' (strictly positive input), then StandardScaler when
    `standardize`. Each column's lambda maximises the reference's
    log-likelihood by a fixed-step golden-section search over [-8, 8]
    (the reference: scipy's Brent from the bracket (-2, 2)), float32. NaN is
    ignored in fit and kept. inverse_transform undoes the standardisation
    (x * scale + mean), then applies scipy's inv_boxcox or the reference's
    yeo-johnson inverse, as exp(log1p(lambda x) / lambda) in float32."""
    _parameters = ("method", "standardize", "copy")

    def __init__(self, method="yeo-johnson", *, standardize=True, copy=True):
        self.method = method
        self.standardize = standardize
        self.copy = copy

    def fit(self, X, y=None):
        if self.method not in ("yeo-johnson", "box-cox"):
            raise ValueError(f"mojolearn: invalid method {self.method!r}")
        arr = _x2d(X)
        n, d = arr.shape
        method = 1 if self.method == "box-cox" else 0
        mode = _mode()
        pr = _Prog()
        xo = pr.put(arr)
        st, lam = pr.alloc(6 * d), pr.alloc(d)
        pr.stage("col_stats", d, xo, n, d, st)
        if _optional_prep_entry(_prep_binding(mode), "x_prep_host_column") is not None:
            # the host binding: its own pt_fit (x_prep/host/power.mojo), the same words
            pr.stage("pt_fit", d, xo, n, d, method, st, lam)
        else:
            # the device: pt_fit_unit's golden-section search as stages (x_prep/transform.mojo),
            # each element's logarithm once, the transform of every element at once per
            # evaluation, then the column folds
            # lane af-ptimpute (FAST + Apple builds with the defines, `_ptimpute_flags`): the
            # device folds each evaluation as a row-tiled grid with the transform in registers
            # (x_prep/fastpt.mojo), so the T and LG blocks are never read or written: they shrink
            # to a word and `pt_log` is not staged. PT_SPEC (bit 2) runs that fold over the
            # speculated search's candidates; PT_COLBATCH (bit 1) over the staged search.
            flags = _ptimpute_flags(mode)
            tiled = bool(flags & 1)
            if flags & 2:
                spec = _PT_FAST_SPEC
            else:
                spec = _pt_spec_depth(n, d) if mode == "identical" else 0
            if spec:
                # the search speculated `spec` evaluations deep (transform.mojo pt_spts ..
                # pt_sres): the same points, values and decisions, fewer dependent folds
                mmax = max(2 ** spec - 1, 2)   # the opening round's two points (see _pt_spec_depth)
                # the candidates' transforms: contiguous per candidate (0) or one row's side by side (1)
                il = 0 if tiled else (1 if os.environ.get("MOJOLEARN_XPREP_PT_INTERLEAVE", "0") == "1" else 0)
                state, leval = pr.alloc(_PT_STATE * d), pr.alloc(d)
                spl, vals = pr.alloc(d * mmax), pr.alloc(d * mmax)
                if tiled:
                    lg, tv = pr.scratch(1), pr.scratch(1)
                else:
                    lg, tv = pr.scratch(n * d), pr.scratch(n * d * mmax)
                pr.stage("pt_init", d, method, st, d, lam, state, leval)
                if not tiled:
                    pr.stage("pt_log", n * d, xo, n, d, method, lg)
                k0 = 0
                while k0 <= _PT_EVALS - 2:
                    steps = 2 if k0 == 0 else min(spec, _PT_EVALS - 1 - k0)
                    m = 2 if k0 == 0 else 2 ** steps - 1
                    pr.stage("pt_spts", d, state, leval, spl, m, k0)
                    pr.stage("pt_smap", m * n * d, xo, n, d, method, spl, m, tv, lg, il)
                    pr.stage("pt_sfold", d * m, xo, n, d, method, tv, m, state, spl, vals, 1 if k0 == 0 else 0, il)
                    pr.stage("pt_sres", d, state, leval, m, vals, k0, steps, lam)
                    k0 += steps
            else:
                state, leval = pr.alloc(_PT_STATE * d), pr.alloc(d)
                tv, lg = (pr.alloc(1), pr.alloc(1)) if tiled else (pr.alloc(n * d), pr.alloc(n * d))
                pr.stage("pt_init", d, method, st, d, lam, state, leval)
                if not tiled:
                    pr.stage("pt_log", n * d, xo, n, d, method, lg)
                for k in range(_PT_EVALS):
                    # tiled: the device skips pt_map and fuses it into pt_fold (LG1 = 0 either way)
                    pr.stage("pt_map", n * d, xo, n, d, method, leval, tv, 0 if tiled else lg + 1)
                    pr.stage("pt_fold", d, xo, n, d, method, tv, k, state, leval, lam)
        mean, scale = pr.alloc(d), pr.alloc(d)
        if self.standardize:
            # PT_FUSED_TRANSFORM (bit 4, FAST + Apple): the device folds col_stats of the transform
            # straight from X (x_prep/fastpt.mojo cs_tile_kernel) and skips this col_stats stage, so
            # the transformed block is never written: a word
            fused = bool(_ptimpute_flags(mode) & 4)
            tx, st2 = pr.alloc(1) if fused else pr.alloc(n * d), pr.alloc(6 * d)
            pr.stage("pt_apply", n * d, xo, n, d, lam, method, _NONE, _NONE, tx)
            pr.stage("col_stats", d, tx, n, d, st2)
            pr.stage("std_params", d, st2, d, mean, scale)
        pr.run(mode)
        if method == 1 and any(v <= 0 for v in pr.values(st + 3 * d, d)):
            raise ValueError("mojolearn: The Box-Cox transformation can only be applied to strictly positive data")
        self.lambdas_ = pr.get(lam, d)
        self._mean = pr.get(mean, d) if self.standardize else None
        self._scale = pr.get(scale, d) if self.standardize else None
        self._method = method
        self.numeric_mode_, self.n_features_in_ = mode, d
        return self

    def _apply(self, X, op):
        self._check_fitted()
        arr = _x2d(X)
        self._check_width(arr)
        n, d = arr.shape
        pr = _Prog()
        xo, lo = pr.put(arr), pr.put(self.lambdas_)
        mo = pr.put(self._mean) if self.standardize else _NONE
        so = pr.put(self._scale) if self.standardize else _NONE
        out = pr.output(n * d)
        pr.stage(op, n * d, xo, n, d, lo, self._method, mo, so, out)
        pr.run(self.numeric_mode_)
        return pr.get(out, (n, d))

    def transform(self, X):
        return self._apply(X, "pt_apply")

    def inverse_transform(self, X):
        return self._apply(X, "pt_inverse")


class Normalizer(_PrepBase):
    """sklearn.preprocessing.Normalizer: each row divided by its 'l1', 'l2'
    or 'max' norm (columns summed in ascending order); a zero row is left
    as it is. Stateless."""
    _parameters = ("norm", "copy")

    def __init__(self, norm="l2", *, copy=True):
        self.norm = norm
        self.copy = copy

    def fit(self, X, y=None):
        if self.norm not in ("l1", "l2", "max"):
            raise ValueError(f"mojolearn: invalid norm {self.norm!r}")
        self.n_features_in_ = _x2d(X).shape[1]
        self.numeric_mode_ = _mode()
        return self

    def transform(self, X, copy=None):
        self._check_fitted()
        arr = _x2d(X)
        self._check_width(arr)
        n, d = arr.shape
        pr = _Prog()
        xo = pr.put(arr)
        out = pr.output(n * d)
        pr.stage("normalize", n, xo, n, d, {"l1": 0, "l2": 1, "max": 2}[self.norm], out)
        pr.run(self.numeric_mode_)
        return pr.get(out, (n, d))


class PolynomialFeatures(_PrepBase):
    """sklearn.preprocessing.PolynomialFeatures: the reference's column
    order (the bias, then combinations with replacement, or without when
    interaction_only, degree by degree); each output is the product of its
    input columns left to right on the device; order='F' returns the same
    values in a column-major (Fortran-ordered) array."""
    _parameters = ("degree", "interaction_only", "include_bias", "order")

    def __init__(self, degree=2, *, interaction_only=False, include_bias=True, order="C"):
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.order = order

    def _combos(self, d):
        from itertools import chain, combinations, combinations_with_replacement
        if isinstance(self.degree, numbers.Integral):
            lo, hi = 0, int(self.degree)
        else:
            lo, hi = (int(v) for v in self.degree)
        if hi < 0 or lo < 0 or lo > hi:
            raise ValueError(f"mojolearn: invalid degree {self.degree!r}")
        comb = combinations if self.interaction_only else combinations_with_replacement
        it = chain.from_iterable(comb(range(d), i) for i in range(max(1, lo), hi + 1))
        if self.include_bias:
            it = chain(comb(range(d), 0), it)
        return [tuple(c) for c in it]

    def fit(self, X, y=None):
        if self.order not in ("C", "F"):
            raise ValueError("mojolearn: PolynomialFeatures order must be 'C' or 'F'")
        d = _x2d(X).shape[1]
        self._terms = self._combos(d)
        self.n_features_in_, self.n_output_features_ = d, len(self._terms)
        self.powers_ = Array.from_list([[t.count(j) for j in range(d)] for t in self._terms] or [[0] * d], "<i8")
        self.numeric_mode_ = _mode()
        return self

    def transform(self, X):
        self._check_fitted()
        arr = _x2d(X)
        self._check_width(arr)
        n, d = arr.shape
        idx, start = [], [0]
        for t in self._terms:
            idx.extend(t)
            start.append(len(idx))
        nout = len(self._terms)
        pr = _Prog()
        xo, io, so = pr.put(arr), pr.put_list(idx or [0]), pr.put_list(start)
        # the output region measured slower here (m4-a: 0.452 -> 0.523 s): arena words
        out = pr.alloc(n * nout)
        pr.stage("poly", n * nout, xo, n, d, io, so, nout, out)
        pr.run(self.numeric_mode_)
        if self.order == "F":
            pr._check(out, n * nout)
            seg = pr.arena[out:out + n * nout]
            store = array.array("f")
            for j in range(nout):
                store.extend(seg[j::nout])
            return Array._owned(store, (n, nout), "<f4", "F")
        return pr.get(out, (n, nout))


def _f_order(pr, off, n, w):
    """An (n, w) arena block as a Fortran-ordered Array (the same words)."""
    pr._check(off, n * w)
    seg = pr.arena[off:off + n * w]
    store = array.array("f")
    for j in range(w):
        store.extend(seg[j::w])
    return Array._owned(store, (n, w), "<f4", "F")


def _check_weights(sample_weight, n, who):
    """sample_weight as float32 words (nonnegative, length n) and their list."""
    w = as_f32_c(sample_weight, ndim=1, name="sample_weight")[0]
    if w.size != n:
        raise ValueError(f"mojolearn: sample_weight has {w.size} entries; X has {n} rows")
    wl = w.tolist()
    if any(not v >= 0 for v in wl):
        raise ValueError(f"mojolearn: {who} sample_weight must be nonnegative")
    if not any(v > 0 for v in wl):
        raise ValueError(f"mojolearn: {who} sample_weight is all zero")
    return w, wl


def _weighted_groups(pr, arr, w, n, d):
    """Stages writing every column's distinct values (ascending, -0.0 folded
    into 0.0, NaN last) and their summed weights: UG[c*n :] and UG[n*d + c*n :],
    their count UCNT[c]. Returns (UG, UCNT)."""
    xo = pr.put(arr)
    so, ug, ucnt, codes = pr.work(n * d), pr.alloc(2 * n * d), pr.alloc(d), pr.alloc(n * d)
    pr.stage("sort_cols", d, xo, n, d, so, 1)
    pr.stage("unique_cols", d, so, n, d, ug, ucnt)
    pr.stage("lookup", n * d, xo, n, d, ug, n, ucnt, codes)
    pr.stage("kbins_gw", d, codes, n, d, pr.put(w), ug + n * d)
    return ug, ucnt


def _weighted_levels(pr, ug, ucnt, n, d, levels, average, out):
    """A stage of the reference's `_weighted_percentile` of every column at
    the percent `levels` (one list per column, padded to a common stride
    max + 1) into `out` (column c at c * stride), over `_weighted_groups`."""
    nb = [len(lv) - 1 for lv in levels]
    nbmax = max(nb)
    flat = []
    for lv in levels:
        flat += list(lv) + [0.0] * (nbmax + 1 - len(lv))
    pr.stage("kbins_wq", d, ug, n, d, ucnt, pr.put_list(nb), nbmax, pr.put_list(flat), int(average), out)


class SplineTransformer(_PrepBase):
    """sklearn.preprocessing.SplineTransformer: per feature, B-splines of
    `degree` on `n_knots` base knots ('uniform' over the training range,
    'quantile': numpy linear percentiles, or an explicit (n_knots,
    n_features) array), extended by `degree` knots at each end at the edge
    spacing, or wrapped for 'periodic'; extrapolation 'constant', 'continue',
    'linear', 'periodic' or 'error'; sample_weight weighs the knots (the
    reference's weighted percentile for 'quantile', the range over rows of
    nonzero weight for 'uniform'); handle_missing 'error' or 'zeros' (a NaN
    is left out of the knots and encodes as a zero block); order 'C' or 'F'.
    Dense float32 output (sparse_output is refused: there is no sparse
    Array)."""
    _parameters = ("n_knots", "degree", "knots", "extrapolation", "include_bias", "order", "handle_missing",
                   "sparse_output")
    _EXTRAP = {"constant": 0, "continue": 1, "error": 1, "linear": 2, "periodic": 3}

    def __init__(self, n_knots=5, degree=3, *, knots="uniform", extrapolation="constant", include_bias=True,
                 order="C", handle_missing="error", sparse_output=False):
        self.n_knots = n_knots
        self.degree = degree
        self.knots = knots
        self.extrapolation = extrapolation
        self.include_bias = include_bias
        self.order = order
        self.handle_missing = handle_missing
        self.sparse_output = sparse_output

    def _no_nan(self, pr, st, d, n):
        if self.handle_missing == "error" and any(int(v) != n for v in pr.values(st, d)):
            raise ValueError("mojolearn: Input X contains NaN values and `SplineTransformer` is configured to "
                             "error in this case (handle_missing='error'). To avoid this error, set "
                             "handle_missing='zeros' to encode missing values as splines with value 0 or ensure "
                             "no missing values in X.")

    def fit(self, X, y=None, sample_weight=None):
        if self.extrapolation not in self._EXTRAP:
            raise ValueError(f"mojolearn: invalid extrapolation {self.extrapolation!r}")
        if self.handle_missing not in ("error", "zeros"):
            raise ValueError(f"mojolearn: invalid handle_missing {self.handle_missing!r}")
        if self.order not in ("C", "F"):
            raise ValueError(f"mojolearn: invalid order {self.order!r}")
        if self.sparse_output:
            raise NotImplementedError("mojolearn: SplineTransformer sparse_output is not implemented "
                                      "(there is no sparse Array)")
        k = int(self.degree)
        if k < 0 or k > 7:
            raise ValueError("mojolearn: SplineTransformer needs 0 <= degree <= 7")
        arr = _x2d(X)
        n, d = arr.shape
        if n < 2:
            raise ValueError("mojolearn: SplineTransformer needs at least 2 samples")
        w = wl = None
        if sample_weight is not None:
            w, wl = _check_weights(sample_weight, n, "SplineTransformer")
        given = not isinstance(self.knots, str)
        if not given and self.knots not in ("uniform", "quantile"):
            raise ValueError(f"mojolearn: invalid knots {self.knots!r}")
        if given:
            rows = [[float(v) for v in (r.tolist() if hasattr(r, "tolist") else r)] for r in
                    (self.knots.tolist() if hasattr(self.knots, "tolist") else self.knots)]
            nk = len(rows)
            if nk < 2:
                raise ValueError("mojolearn: Number of knots, knots.shape[0], must be >= 2.")
            if any(len(r) != d for r in rows):
                raise ValueError("mojolearn: knots.shape[1] == n_features is violated.")
            cols = [list(array.array("f", [r[c] for r in rows])) for c in range(d)]
            if not all(b > a for col in cols for a, b in zip(col, col[1:])):
                raise ValueError("mojolearn: knots must be sorted without duplicates.")
        else:
            nk = int(self.n_knots)
            if nk < 2:
                raise ValueError("mojolearn: SplineTransformer needs n_knots >= 2")
        periodic = self.extrapolation == "periodic"
        if periodic and nk <= k:
            raise ValueError(f"mojolearn: Periodic splines require degree < n_knots. Got n_knots={nk} and "
                             f"degree={k}.")
        mode = _mode()
        pr = _Prog()
        xo = pr.put(arr)
        sorts = not given and self.knots == "quantile" and w is None
        so, st = (pr.work(n * d) if sorts else pr.alloc(n * d)), pr.alloc(6 * d)
        knots = pr.alloc(d * (nk + 2 * k))
        pr.stage("col_stats", d, xo, n, d, st)
        uniform, kst = 0, st
        if given:
            base = pr.put_list([v for col in cols for v in col])
        elif self.knots == "quantile":
            base = pr.alloc(d * nk)
            if w is None:
                qf = pr.put_list([i / (nk - 1) for i in range(nk)])
                pr.stage("sort_cols", d, xo, n, d, so, 0)
                pr.stage("quantile", d * nk, so, n, d, qf, nk, base, st)
            else:
                step = 1.0 / (nk - 1)
                lv = [100.0 * (i * step) for i in range(nk - 1)] + [100.0]
                ug, ucnt = _weighted_groups(pr, arr, w, n, d)
                _weighted_levels(pr, ug, ucnt, n, d, [lv] * d, False, base)
        else:
            base, uniform = pr.alloc(d * nk), 1
            if w is not None and any(v == 0 for v in wl):
                kst = pr.alloc(6 * d)
                nz = [i for i, v in enumerate(wl) if v > 0]
                pr.stage("col_stats", d, pr.put(_gather_rows(arr, nz)), len(nz), d, kst)
        pr.stage("spline_knots", d, base, nk, d, k, knots, uniform, kst, int(periodic))
        pr.run(mode)
        self._no_nan(pr, st, d, n)
        self._knots = pr.get(knots, d * (nk + 2 * k))
        flat = pr.values(knots, d * (nk + 2 * k))
        wd = nk + 2 * k
        self.bsplines_ = [Array.from_list(flat[c * wd:(c + 1) * wd], "<f4") for c in range(d)]
        self._lo = [flat[c * wd + k] for c in range(d)]
        self._hi = [flat[c * wd + k + nk - 1] for c in range(d)]
        self._nk, self._k = nk, k
        nspl = nk - 1 if periodic else nk + k - 1
        self.n_features_out_ = d * (nspl if self.include_bias else nspl - 1)
        self.numeric_mode_, self.n_features_in_ = mode, d
        return self

    def transform(self, X):
        self._check_fitted()
        arr = _x2d(X)
        self._check_width(arr)
        n, d = arr.shape
        W = self.n_features_out_
        pr = _Prog()
        xo, ko = pr.put(arr), pr.put(self._knots)
        out = pr.alloc(n * W) if self.order == "F" else pr.output(n * W)
        st = pr.alloc(6 * d)
        check = self.extrapolation == "error" or self.handle_missing == "error"
        if check:
            pr.stage("col_stats", d, xo, n, d, st)
        pr.stage("spline_apply", n * d, xo, n, d, ko, self._nk, self._k, self._EXTRAP[self.extrapolation], W,
                 1 if self.include_bias else 0, out)
        pr.run(self.numeric_mode_)
        if check:
            self._no_nan(pr, st, d, n)
        if self.extrapolation == "error":
            lo, hi = pr.values(st + 3 * d, d), pr.values(st + 4 * d, d)
            if any(a < b for a, b in zip(lo, self._lo)) or any(a > b for a, b in zip(hi, self._hi)):
                raise ValueError("mojolearn: X contains values beyond the limits of the knots")
        if self.order == "F":
            return _f_order(pr, out, n, W)
        return pr.get(out, (n, W))


class Binarizer(_PrepBase):
    """sklearn.preprocessing.Binarizer: 1 where X > threshold, else 0 (NaN is
    kept). Stateless."""
    _parameters = ("threshold", "copy")

    def __init__(self, *, threshold=0.0, copy=True):
        self.threshold = threshold
        self.copy = copy

    def fit(self, X, y=None):
        self.n_features_in_ = _x2d(X).shape[1]
        self.numeric_mode_ = _mode()
        return self

    def transform(self, X, copy=None):
        self._check_fitted()
        arr = _x2d(X)
        self._check_width(arr)
        n, d = arr.shape
        pr = _Prog()
        xo, th = pr.put(arr), pr.put_scalar(self.threshold)
        out = pr.alloc(n * d)
        pr.stage("binarize", n * d, xo, n * d, th, out)
        pr.run(self.numeric_mode_)
        return pr.get(out, (n, d))


# ---------------------------------------------------------------- label transformers
_F32_EXACT = 2 ** 24


_INT_FLOAT = frozenset((int, float))
_FLOAT_ONLY = frozenset((float,))
_INT_ONLY = frozenset((int,))


def _numeric_labels(values):
    """The labels as floats when every one is a real number that float32
    holds exactly (so the device's categories are the labels themselves),
    else None (str labels, ints beyond 2**24: the Python route)."""
    import struct
    kinds = set(map(type, values))
    if kinds and kinds <= _INT_FLOAT:
        # the whole list at C speed (lane prep-apple2): the same answer as the
        # loop below, which a list this test cannot settle still takes
        try:
            fl = list(map(float, values)) if kinds != _FLOAT_ONLY else list(values)
            if not any(map(math.isnan, fl)) and array.array("f", fl).tolist() == fl:
                return fl
        except OverflowError:
            pass
    out = []
    for v in values:
        if isinstance(v, bool) or not isinstance(v, numbers.Real):
            return None
        fv = float(v)
        if fv != fv or struct.unpack("f", struct.pack("f", fv))[0] != fv:
            return None
        out.append(fv)
    return out


def _label_classes(mode, values):
    """(classes list in the reference's order, device categories Array or
    None). Numeric labels: a device sort and run scan; else sorted()."""
    nums = _numeric_labels(values)
    if nums is None or not nums:
        classes, _ = sorted_classes(values)
        return classes, None
    cats = _fit_categories(mode, Array._from_flat(nums, (len(nums), 1), "<f4"))[0]
    ints = set(map(type, values)) == _INT_ONLY or all(isinstance(v, numbers.Integral) for v in values)
    classes = [int(c) if ints else float(c) for c in cats.tolist()]
    return classes, cats


def _label_codes(pr, values, cats):
    """Stages: each label's index among `cats` (or -1). Returns the codes
    offset and the unknown-count offset."""
    arr = Array._from_flat([float(v) for v in values], (len(values), 1), "<f4")
    return _codes(pr, arr, [cats])


def _classes_array(classes):
    kind = label_kind(classes)
    if kind == "int":
        return Array.from_list(classes, "<i8")
    if kind == "float":
        return Array.from_list(classes, "<f8")
    return list(classes)


def _multilabel_indicator(y):
    """y as a float32 (n, K) Array when it is the reference's
    `multilabel-indicator` (`type_of_target`): two-dimensional, more than one
    column, at most two distinct values, all integral. A 2-D y of more
    distinct values is `multiclass-multioutput`, refused as the reference
    refuses it; anything else (1-D, one column) is None."""
    shape = getattr(y, "shape", None)
    if shape is not None:
        if len(shape) != 2:
            return None
        rows = y.tolist()
    elif isinstance(y, (list, tuple)) and y and all(isinstance(r, (list, tuple)) for r in y):
        rows = [list(r) for r in y]
    else:
        return None
    if not rows or len(rows[0]) < 2:
        return None
    if any(len(r) != len(rows[0]) for r in rows):
        raise ValueError("mojolearn: y rows have different lengths")
    distinct = set()
    for r in rows:
        for v in r:
            if isinstance(v, bool) or not isinstance(v, numbers.Real) or v != v or float(v) != int(float(v)):
                raise ValueError("mojolearn: Multioutput target data is not supported with label binarization")
            distinct.add(float(v))
    if len(distinct) > 2:
        raise ValueError("mojolearn: Multioutput target data is not supported with label binarization")
    return _x2d(Array.from_list([[float(v) for v in r] for r in rows], "<f4"), "y")


# lane neural-pass137: a numeric label BUFFER (an ndarray, an Array) takes the
# device from its own bytes. The route it replaces built Python objects for
# every label five times over (tolist, the float32 exactness test, the float
# list, the float32 Array, the type sets) in fit and again in transform, then
# ran two programs whose unique scan and unknown count were ONE thread
# walking every row. Here: one upload of the raw words, `lab_load` (the
# float32 word the old route would have built, NaN where float32 cannot hold
# the label, which sends the call back to the old route), the device sort, a
# chunked run scan, and for fit_transform the codes in the same program.
# MOJOLEARN_XPREP_LABELS=0 is the old route (the A/B arm).
_LABEL_KIND = {"<f4": (0, 1), "<i4": (1, 1), "<u4": (2, 1), "<i8": (3, 2), "<f8": (4, 2)}


class _LabelBuf:
    __slots__ = ("kind", "words", "n", "is_float", "_src")

    def __init__(self, kind, words, n, is_float, src):
        self.kind, self.words, self.n, self.is_float, self._src = kind, words, n, is_float, src


def _label_buffer(y):
    """A numeric label vector as raw int32 words (a view, no copy, of a
    contiguous buffer), or None: the old route (lists, str or bool labels,
    other dtypes, a matrix, an empty y, MOJOLEARN_XPREP_LABELS=0)."""
    if os.environ.get("MOJOLEARN_XPREP_LABELS", "1") == "0":
        return None
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
    if spec is None or arr.size == 0 or arr.ndim < 1 or arr.size != max(arr.shape):
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
    return max(1024, int(n ** 0.5) + 1)


def _label_load(pr, lb):
    x = pr.work(lb.n)
    pr.stage("lab_load", lb.n, pr.put_words(lb.words), lb.kind, x)
    return x


def _label_unique(pr, x, n):
    """Stages: the sorted distinct words of x (sort_cols + a chunked run
    scan: `unique_cols`' words). Returns (U offset, count offset)."""
    s = pr.work(n)
    pr.stage("sort_cols", 1, x, n, 1, s, 1)
    ch = _label_chunk(n)
    nch = -(-n // ch)
    cnt, off = pr.work(nch), pr.work(nch)
    u, k = pr.alloc(n), pr.alloc(1)
    pr.stage("uniq_count", nch, s, n, ch, cnt)
    pr.stage("uniq_scan", 1, cnt, nch, off, k)
    pr.stage("uniq_write", nch, s, n, ch, off, u)
    return u, k


def _label_classes_of(pr, lb, u, k):
    """(classes, cats) read back after the run, or None when a label has no
    exact float32 word (NaN sorts last): the caller takes the old route."""
    K = int(pr.values(k, 1)[0])
    cats = pr.get(u, K)
    vals = cats.tolist()
    if not vals or vals[-1] != vals[-1]:
        return None
    classes = [float(c) for c in vals] if lb.is_float else [int(c) for c in vals]
    return classes, cats


def _label_fit_device(mode, lb, codes=False):
    """One program: (classes, cats, int32 codes Array or None), or None for
    the old route."""
    n = lb.n
    pr = _Prog()
    x = _label_load(pr, lb)
    u, k = _label_unique(pr, x, n)
    out = None
    if codes:
        c = pr.work(n)
        pr.stage("lookup", n, x, n, 1, u, n, k, c)
        out = pr.output(n, "i")
        pr.stage("f2i", n, c, out)
    pr.run(mode)
    got = _label_classes_of(pr, lb, u, k)
    if got is None:
        return None
    return got[0], got[1], (pr.get_i32(out, n) if codes else None)


def _label_codes_device(pr, lb, cats):
    """Stages: each label's index among `cats` (-1 when absent). Returns the
    float codes offset."""
    x = _label_load(pr, lb)
    K = cats.size
    c = pr.work(lb.n)
    pr.stage("lookup", lb.n, x, lb.n, 1, pr.put(cats), K, pr.put_scalar(K), c)
    return c


class LabelEncoder(_PrepBase):
    """sklearn.preprocessing.LabelEncoder: classes_ are the sorted distinct
    labels (numeric labels on the device: sort + run scan; str labels in
    Python), transform is each label's index (a device binary search),
    int32. An unseen label is refused, as the reference refuses it. A
    numeric label buffer crosses as its raw words (`_label_buffer`, lane
    neural-pass137); fit_transform is then one program."""
    _parameters = ()

    def fit(self, y):
        self.numeric_mode_ = _mode()
        lb = _label_buffer(y)
        got = _label_fit_device(self.numeric_mode_, lb) if lb is not None else None
        if got is not None:
            self._classes, self._cats = got[0], got[1]
        else:
            values = flatten_labels(y)
            self._classes, self._cats = _label_classes(self.numeric_mode_, values)
        self.classes_ = _classes_array(self._classes)
        return self

    def fit_transform(self, y):
        # a numeric buffer: classes and codes in ONE program (lane neural-pass137)
        lb = _label_buffer(y)
        if lb is not None:
            self.numeric_mode_ = _mode()
            got = _label_fit_device(self.numeric_mode_, lb, codes=True)
            if got is not None:
                self._classes, self._cats = got[0], got[1]
                self.classes_ = _classes_array(self._classes)
                return got[2]
        return self.fit(y).transform(y)

    def _check_fitted(self):
        if not hasattr(self, "_classes"):
            raise RuntimeError("mojolearn: this LabelEncoder instance is not fitted yet")

    def transform(self, y):
        self._check_fitted()
        lb = _label_buffer(y) if self._cats is not None else None
        if lb is not None:
            n = lb.n
            pr = _Prog()
            codes = _label_codes_device(pr, lb, self._cats)
            out = pr.output(n, "i")
            pr.stage("f2i", n, codes, out)
            ch = _label_chunk(n)
            nch = -(-n // ch)
            neg = pr.alloc(nch)
            pr.stage("chunk_neg", nch, codes, n, ch, neg)
            pr.run(self.numeric_mode_)
            if not any(pr.get_i32(neg, nch).tolist()):
                return pr.get_i32(out, n)
            # an unknown label: the old route names the refusal
        values = flatten_labels(y)
        if not values:
            return Array((0,), "<i4")
        if self._cats is None or _numeric_labels(values) is None:
            index = {c: i for i, c in enumerate(self._classes)}
            missing = [v for v in values if v not in index]
            if missing:
                raise ValueError(f"mojolearn: y contains previously unseen labels: {missing[:5]}")
            return Array.from_list([index[v] for v in values], "<i4")
        n = len(values)
        pr = _Prog()
        codes, neg = _label_codes(pr, values, self._cats)
        out = pr.alloc(n)
        pr.stage("f2i", n, codes, out)
        pr.run(self.numeric_mode_)
        if pr.values(neg, 1)[0] > 0:
            raise ValueError("mojolearn: y contains previously unseen labels")
        return pr.get_i32(out, n)

    def inverse_transform(self, y):
        self._check_fitted()
        codes = [int(c) for c in flatten_labels(y)]
        if any(c < 0 or c >= len(self._classes) for c in codes):
            raise ValueError("mojolearn: y contains previously unseen labels")
        return _classes_array([self._classes[c] for c in codes])


class LabelBinarizer(_PrepBase):
    """sklearn.preprocessing.LabelBinarizer for a single-label target: one
    int32 column per class (one column, the second class, for two classes;
    a NEG column for one), pos_label / neg_label; an unseen label is a NEG
    row. Numeric labels take the device route, str labels Python's.
    inverse_transform is the reference's (argmax for multiclass, the
    threshold for binary and multilabel). A multilabel indicator y (2-D,
    more than one column, at most two distinct integral values) is the
    reference's `multilabel-indicator`: classes_ are the column indices, a
    nonzero entry is pos_label and a zero neg_label. sparse_output is
    refused."""
    _parameters = ("neg_label", "pos_label", "sparse_output")

    def __init__(self, *, neg_label=0, pos_label=1, sparse_output=False):
        self.neg_label = neg_label
        self.pos_label = pos_label
        self.sparse_output = sparse_output

    def fit(self, y):
        if self.sparse_output:
            raise NotImplementedError("mojolearn: LabelBinarizer(sparse_output=True) is not implemented")
        if not (isinstance(self.neg_label, numbers.Integral) and isinstance(self.pos_label, numbers.Integral)
                and self.neg_label < self.pos_label):
            raise ValueError("mojolearn: neg_label must be an integer below pos_label")
        self.numeric_mode_ = _mode()
        ind = _multilabel_indicator(y)
        if ind is not None:
            self._classes, self._cats = list(range(ind.shape[1])), None
            self.classes_ = Array.from_list(self._classes, "<i8")
            self.y_type_ = "multilabel-indicator"
            return self
        lb = _label_buffer(y)
        got = _label_fit_device(self.numeric_mode_, lb) if lb is not None else None
        if got is not None:
            self._classes, self._cats = got[0], got[1]
        else:
            values = flatten_labels(y)
            self._classes, self._cats = _label_classes(self.numeric_mode_, values)
        self.classes_ = _classes_array(self._classes)
        self.y_type_ = "binary" if len(self._classes) <= 2 else "multiclass"
        return self

    def fit_transform(self, y):
        return self.fit(y).transform(y)

    def _check_fitted(self):
        if not hasattr(self, "_classes"):
            raise RuntimeError("mojolearn: this LabelBinarizer instance is not fitted yet")

    def transform(self, y):
        self._check_fitted()
        ind = _multilabel_indicator(y)
        if ind is not None or self.y_type_ == "multilabel-indicator":
            if self.y_type_ != "multilabel-indicator":
                raise ValueError("mojolearn: The object was not fitted with multilabel input.")
            if ind is None:
                raise ValueError("mojolearn: y is not a multilabel indicator; the binarizer was fitted with one")
            n, K = ind.shape
            if K != len(self._classes):
                raise ValueError(f"mojolearn: classes {self._classes} mismatch with the {K} label columns in y")
            pr = _Prog()
            out = pr.output(n * K, "i")
            pr.stage("indicator", n * K, pr.put(ind), n * K, 0, _NONE, int(self.neg_label), int(self.pos_label),
                     out)
            pr.run(self.numeric_mode_)
            return pr.get_i32(out, (n, K))
        K = len(self._classes)
        binary = K <= 2
        W = 1 if binary else K
        lb = _label_buffer(y) if self._cats is not None and K > 1 else None
        if lb is not None:
            # lane neural-pass137: the raw words up, the codes and the dense
            # output on the device, no Python object per label
            n = lb.n
            pr = _Prog()
            codes = _label_codes_device(pr, lb, self._cats)
            out = pr.output(n * W, "i")
            pr.stage("label_binarize", n * W, codes, n, K, 1 if binary else 0, int(self.neg_label),
                     int(self.pos_label), W, out)
            pr.run(self.numeric_mode_)
            return pr.get_i32(out, (n, W))
        values = flatten_labels(y)
        n = len(values)
        pr = _Prog()
        if self._cats is None or _numeric_labels(values) is None:
            index = {c: i for i, c in enumerate(self._classes)}
            codes = pr.put_list([index.get(v, -1) for v in values])
        else:
            codes, _neg = _label_codes(pr, values, self._cats)
        if K == 1:
            codes = pr.put_list([-1] * n)
        out = pr.output(n * W, "i")
        pr.stage("label_binarize", n * W, codes, n, K, 1 if binary else 0, int(self.neg_label),
                 int(self.pos_label), W, out)
        pr.run(self.numeric_mode_)
        return pr.get_i32(out, (n, W))

    def inverse_transform(self, Y, threshold=None):
        self._check_fitted()
        arr = _x2d(Y, "Y")
        n, W = arr.shape
        K = len(self._classes)
        pr = _Prog()
        if self.y_type_ == "multilabel-indicator":
            if W != K:
                raise ValueError(f"mojolearn: Y has {W} columns, expected {K}")
            if threshold is None:
                threshold = (self.pos_label + self.neg_label) / 2.0
            out = pr.alloc(n * W)
            pr.stage("indicator", n * W, pr.put(arr), n * W, 1, pr.put_scalar(threshold), 0, 1, out)
            pr.run(self.numeric_mode_)
            return pr.get_i32(out, (n, W))
        if self.y_type_ == "multiclass":
            if W != K:
                raise ValueError(f"mojolearn: Y has {W} columns, expected {K}")
            codes = _block_argmax(pr, arr, [K], None, False)
            pr.run(self.numeric_mode_)
            idx = [int(v) for v in pr.values(codes, n)]
        else:
            if W > 2:
                raise ValueError("mojolearn: output_type='binary', but y.shape = " + str((n, W)))
            if threshold is None:
                threshold = (self.pos_label + self.neg_label) / 2.0
            xo = pr.put(arr)
            out = pr.alloc(n * W)
            pr.stage("binarize", n * W, xo, n * W, pr.put_scalar(threshold), out)
            pr.run(self.numeric_mode_)
            # Read only output words, then select the last binary column.
            vals = pr.values(out, n * W)[W - 1::W]
            if K == 1:
                return _classes_array([self._classes[0]] * n)
            idx = [1 if v == 1.0 else 0 for v in vals]
        return _classes_array([self._classes[i] for i in idx])


#: lane gap-prep2 (2026-10-02): MultiLabelBinarizer's int label sets cross as
#: ONE int64 buffer (two C-speed passes over the sets) and the rest is the
#: device's: lab_load + the run scan for classes_ (LabelEncoder's route),
#: lookup for the codes and one unit per row (`row_ones`) for the indicator.
#: The Python route flattened the sets three times, converted every label to
#: a float twice and built a float row-owner vector on the host. The same
#: classes and the same int32 words. MOJOLEARN_MLB_DEVICE=0: the Python route.
_MLB_ROW_TYPES = frozenset((list, tuple, set, frozenset))


class _MLBFlat:
    __slots__ = ("n", "lb", "offs")

    def __init__(self, n, lb, offs):
        self.n, self.lb, self.offs = n, lb, offs


def _mlb_flat(y):
    """y's label sets as one int64 label buffer and int32 row offsets, or
    None (the Python route): rows that are not lists, tuples or sets, a
    label that is not a plain int (bool, float, str, numpy scalars), no
    label at all, or MOJOLEARN_MLB_DEVICE=0."""
    if os.environ.get("MOJOLEARN_MLB_DEVICE", "1").strip() == "0":
        return None
    rows = y if isinstance(y, (list, tuple)) else None
    if not rows or not set(map(type, rows)) <= _MLB_ROW_TYPES:
        return None
    if set(map(type, itertools.chain.from_iterable(rows))) != _INT_ONLY:
        return None
    try:
        flat = array.array("q", itertools.chain.from_iterable(rows))
    except OverflowError:
        return None
    if not flat or len(flat) > 2 ** 30:
        return None
    lb = _label_buffer(flat)
    if lb is None:
        return None
    offs = Array._owned(array.array("i", itertools.accumulate(map(len, rows), initial=0)), (len(rows) + 1,),
                        "<i4", "C")
    return _MLBFlat(len(rows), lb, offs)


class MultiLabelBinarizer(_PrepBase):
    """sklearn.preprocessing.MultiLabelBinarizer: classes_ the sorted union
    of every sample's labels (or `classes` as given, in that order), transform
    an int32 indicator matrix; an unseen label is ignored (the reference
    warns). Numeric labels take the device route (a sort, a binary search, a
    scatter of ones), str labels Python's. sparse_output is refused."""
    _parameters = ("classes", "sparse_output")

    def __init__(self, *, classes=None, sparse_output=False):
        self.classes = classes
        self.sparse_output = sparse_output

    def fit(self, y):
        if self.sparse_output:
            raise NotImplementedError("mojolearn: MultiLabelBinarizer(sparse_output=True) is not implemented")
        self.numeric_mode_ = _mode()
        if self.classes is not None:
            self._classes = list(self.classes)
            nums = _numeric_labels(self._classes)
            self._given = True
            self._cats = None
        elif not self._fit_flat(_mlb_flat(y)):
            flat = [v for row in y for v in row]
            self._classes, self._cats = _label_classes(self.numeric_mode_, flat) if flat else ([], None)
            self._given = False
        self.classes_ = _classes_array(self._classes)
        return self

    def _fit_flat(self, fl):
        """classes_ from the device route (`_mlb_flat`); False when it does
        not apply (the caller takes the Python route)."""
        if fl is None:
            return False
        got = _label_fit_device(self.numeric_mode_, fl.lb)
        if got is None:
            return False
        self._classes, self._cats = got[0], got[1]
        self._given = False
        return True

    def _transform_flat(self, fl):
        """The int32 indicator from the device route: one program."""
        n, K = fl.n, len(self._classes)
        pr = _Prog()
        codes = _label_codes_device(pr, fl.lb, self._cats)
        offs = pr.put_words(fl.offs)
        out = pr.output(n * max(K, 1), "i")
        pr.stage("row_ones", n, codes, offs, K, out)
        pr.run(self.numeric_mode_)
        return pr.get_i32(out, (n, K))

    def fit_transform(self, y):
        if self.classes is None and not self.sparse_output:
            fl = _mlb_flat(y)
            if fl is not None:
                self.numeric_mode_ = _mode()
                if self._fit_flat(fl):
                    self.classes_ = _classes_array(self._classes)
                    return self._transform_flat(fl)
        y = [list(row) for row in y]
        return self.fit(y).transform(y)

    def _check_fitted(self):
        if not hasattr(self, "_classes"):
            raise RuntimeError("mojolearn: this MultiLabelBinarizer instance is not fitted yet")

    def transform(self, y):
        self._check_fitted()
        if self._cats is not None and self._classes:
            fl = _mlb_flat(y)
            if fl is not None:
                return self._transform_flat(fl)
        rows = [list(r) for r in y]
        n, K = len(rows), len(self._classes)
        flat = [v for r in rows for v in r]
        owner = [i for i, r in enumerate(rows) for _ in r]
        pr = _Prog()
        if not flat:
            out = pr.alloc(n * max(K, 1))
            pr.run(self.numeric_mode_)
            return pr.get_i32(out, (n, K))
        if self._cats is None or _numeric_labels(flat) is None:
            index = {c: i for i, c in enumerate(self._classes)}
            codes = pr.put_list([index.get(v, -1) for v in flat])
        else:
            codes, _neg = _label_codes(pr, flat, self._cats)
        ro = pr.put_list(owner)
        out = pr.output(n * max(K, 1), "i")
        pr.stage("scatter_ones", len(flat), codes, ro, K, out)
        pr.run(self.numeric_mode_)
        return pr.get_i32(out, (n, K))

    def inverse_transform(self, yt):
        self._check_fitted()
        rows = yt.tolist() if hasattr(yt, "tolist") else list(yt)
        return [tuple(self._classes[j] for j, v in enumerate(r) if v) for r in rows]


# ---------------------------------------------------------------- iterative imputer
class IterativeImputer(_PrepBase):
    """sklearn.impute.IterativeImputer. With its default estimator
    (BayesianRidge, default priors, max_iter 300, tol 1e-3) the whole fit is
    one device program: initial fill by SimpleImputer(initial_strategy), then
    rounds over the features in imputation_order ('ascending' default,
    'descending', 'roman', 'arabic', or 'random': a fresh permutation each
    round), each feature regressed on the others (or on n_nearest_features of
    them, drawn without replacement with probability proportional to their
    absolute correlation with it) over its observed rows, its missing
    entries predicted and clipped to [min_value, max_value], or with
    sample_posterior drawn from the predictive normal truncated to them;
    stop when the matrix inf-norm of the change < tol * max|X_observed|.
    Another estimator runs the same rounds in Python over that estimator's
    fit / predict (predict(return_std=True) for sample_posterior).
    add_indicator appends MissingIndicator's columns. Every draw (the random
    order, the neighbours, the posterior samples) comes from random_state by
    splitmix64 (the reference draws numpy's; None is seed 0)."""
    _parameters = ("estimator", "missing_values", "sample_posterior", "max_iter", "tol", "n_nearest_features",
                   "initial_strategy", "fill_value", "imputation_order", "skip_complete", "min_value",
                   "max_value", "verbose", "random_state", "add_indicator", "keep_empty_features")

    def __init__(self, estimator=None, *, missing_values=float("nan"), sample_posterior=False, max_iter=10,
                 tol=1e-3, n_nearest_features=None, initial_strategy="mean", fill_value=None,
                 imputation_order="ascending", skip_complete=False, min_value=-float("inf"),
                 max_value=float("inf"), verbose=0, random_state=None, add_indicator=False,
                 keep_empty_features=False):
        self.estimator = estimator
        self.missing_values = missing_values
        self.sample_posterior = sample_posterior
        self.max_iter = max_iter
        self.tol = tol
        self.n_nearest_features = n_nearest_features
        self.initial_strategy = initial_strategy
        self.fill_value = fill_value
        self.imputation_order = imputation_order
        self.skip_complete = skip_complete
        self.min_value = min_value
        self.max_value = max_value
        self.verbose = verbose
        self.random_state = random_state
        self.add_indicator = add_indicator
        self.keep_empty_features = keep_empty_features

    def _refuse(self):
        if self.imputation_order not in ("ascending", "descending", "roman", "arabic", "random"):
            raise ValueError(f"mojolearn: invalid imputation_order {self.imputation_order!r}")
        nnf = self.n_nearest_features
        if nnf is not None and (not isinstance(nnf, numbers.Integral) or nnf < 1):
            raise ValueError(f"mojolearn: n_nearest_features must be an int >= 1; got {nnf!r}")
        if self.sample_posterior and self.estimator is not None and \
                "return_std" not in inspect.signature(self.estimator.predict).parameters:
            raise ValueError("mojolearn: If 'sample_posterior' is True, the estimator must support "
                             "'return_std' in its 'predict' method.")

    def _bounds(self, d):
        def per(v):
            vals = list(v) if isinstance(v, (list, tuple)) or hasattr(v, "tolist") else [v] * d
            vals = vals.tolist() if hasattr(vals, "tolist") else vals
            return [float(x) for x in vals]
        lo, hi = per(self.min_value), per(self.max_value)
        return [v for pair in zip(lo, hi) for v in pair]

    def _prepare(self, pr, arr, Xf, inout=False):
        """Arena: the filled block, its missing mask, the per-column bounds.
        inout: the program imputes the filled block in place and returns it."""
        n, d = arr.shape
        dk = len(self._keep)
        xo = _mark_missing(pr, pr.put(arr), n * d, self.missing_values)
        fo = pr.put(Xf, inout=inout)
        ko = pr.put_list(self._keep)
        mo = pr.alloc(n * dk)
        pr.stage("nan_mask", n * dk, xo, n, d, ko, dk, mo)
        bo = pr.put_list(self._bounds_k)
        return fo, mo, bo

    def _draw(self):
        """The next splitmix64 word of the instance's draw stream."""
        self._rng, z = _splitmix64(self._rng)
        return z

    def _orders(self, miss, dk, rounds):
        """The features each round imputes, in order: the reference's orders
        (a stable argsort of the missing counts; 'descending' that order
        reversed; 'random' a Fisher-Yates permutation per round of the
        candidates, every feature or with skip_complete those with missing
        entries). A feature with nothing missing is skipped whether or not
        skip_complete (the reference fits it and changes nothing)."""
        if self.imputation_order != "random":
            asc = sorted(range(dk), key=lambda j: miss[j])
            order = {"ascending": asc, "descending": asc[::-1], "roman": list(range(dk)),
                     "arabic": list(range(dk))[::-1]}[self.imputation_order]
            return [[j for j in order if miss[j] > 0]] * rounds
        cand = [j for j in range(dk) if miss[j] > 0] if self.skip_complete else list(range(dk))
        out = []
        for _ in range(rounds):
            perm = list(cand)
            for i in range(len(perm) - 1, 0, -1):
                k = self._draw() % (i + 1)
                perm[i], perm[k] = perm[k], perm[i]
            out.append([j for j in perm if miss[j] > 0])
        return out

    def _abs_corr(self, Xf, n, dk, mode):
        """The reference's `_get_abs_corr_mat` of the initially filled block:
        |corrcoef| (the centred Gram on the device, the d x d normalisation in
        Python float64), NaN -> 1e-6, clipped below at 1e-6, a zero diagonal,
        each column scaled to sum 1."""
        pr = _Prog()
        fo, mz = pr.put(Xf), pr.alloc(n * dk)
        means, cnt, g, flag = pr.alloc(dk), pr.alloc(1), pr.alloc(dk * dk), pr.alloc(1)
        pr.stage("ii_mean", dk, fo, n, dk, mz, 0, means, cnt, flag)
        pr.stage("ii_gram", dk * dk, fo, n, dk, mz, 0, means, g, flag)
        pr.run(mode)
        G = pr.values(g, dk * dk)
        m = [[0.0] * dk for _ in range(dk)]
        for a in range(dk):
            for b in range(dk):
                den = math.sqrt(G[a * dk + a] * G[b * dk + b]) if G[a * dk + a] > 0 and G[b * dk + b] > 0 else 0.0
                v = abs(G[a * dk + b] / den) if den > 0 else float("nan")
                v = min(v, 1.0) if v == v else 1e-6
                m[a][b] = 0.0 if a == b else max(v, 1e-6)
        for b in range(dk):
            col = _pm.nsum(m[a][b] for a in range(dk))
            if col > 0:
                for a in range(dk):
                    m[a][b] /= col
        return m

    def _neighbours(self, corr, j, dk):
        """n_nearest_features predictors of feature j, drawn without
        replacement with probability corr[:, j] (a 53-bit splitmix64 uniform
        against the cumulative weight of the columns not yet drawn)."""
        w = [corr[a][j] for a in range(dk)]
        chosen = set()
        for _ in range(int(self.n_nearest_features)):
            left = [a for a in range(dk) if a not in chosen and w[a] > 0]
            tot = _pm.nsum(w[a] for a in left)
            u = (self._draw() >> 11) * 2.0 ** -53 * tot
            pick, cum = left[-1], 0.0
            for a in left:
                cum += w[a]
                if cum > u:
                    pick = a
                    break
            chosen.add(pick)
        return sorted(chosen)

    def fit_transform(self, X, y=None):
        self._refuse()
        arr = _x2d(X)
        n, d = arr.shape
        mode = _mode()
        self._rng = 0 if self.random_state is None else int(self.random_state) & 0xFFFFFFFFFFFFFFFF
        self._post_step = 0
        self.initial_imputer_ = SimpleImputer(missing_values=self.missing_values, strategy=self.initial_strategy,
                                              fill_value=self.fill_value,
                                              keep_empty_features=self.keep_empty_features).fit(arr)
        Xf = self.initial_imputer_.transform(arr)
        self._keep = list(self.initial_imputer_._keep)
        dk = len(self._keep)
        bounds = self._bounds(d)
        self._bounds_k = [bounds[2 * c + h] for c in self._keep for h in (0, 1)]
        # missing counts per kept column, and the tolerance scale, from the device
        pr = _Prog()
        fo, mo, bo = self._prepare(pr, arr, Xf)
        xo = _mark_missing(pr, pr.put(arr), n * d, self.missing_values)
        st, stm = pr.alloc(6 * d), pr.alloc(6 * dk)
        pr.stage("col_stats", d, xo, n, d, st)
        pr.stage("col_stats", dk, mo, n, dk, stm)
        pr.run(mode)
        miss = [round(v * n) for v in pr.values(stm + dk, dk)]      # mean of the 0/1 mask
        scale = max([v for v in pr.values(st + 5 * d, d)] or [0.0])
        self._indicator = [j for j, c in enumerate(pr.values(st, d)) if int(c) < n] if self.add_indicator else []
        self.n_features_with_missing_ = sum(1 for m in miss if m > 0)
        self.numeric_mode_, self.n_features_in_ = mode, d
        rounds = int(self.max_iter)
        orders = self._orders(miss, dk, rounds)
        nnf = self.n_nearest_features
        corr = self._abs_corr(Xf, n, dk, mode) if nnf is not None and nnf < dk else None
        if self.estimator is not None:
            return self._with_indicator(arr, self._fit_user(arr, Xf, orders, corr, float(self.tol) * scale))
        pr = _Prog()
        fo, mo, bo = self._prepare(pr, arr, Xf, inout=True)
        tol = pr.put_scalar(float(self.tol) * scale)
        flag, niter = pr.alloc(1), pr.alloc(1)
        prev = pr.alloc(n * dk)
        cnt, g = pr.alloc(1), pr.alloc(dk * dk)
        p1 = max(dk - 1, 1)
        gs, eig, vec, w = pr.alloc(p1 * p1), pr.alloc(p1), pr.alloc(p1 * p1), pr.alloc(p1)
        seq, extra = [], []
        seed = self._rng & 0x7FFFFFFF
        # the reference checks convergence only without sample_posterior
        conv = not self.sample_posterior
        rowabs = None
        for r in range(rounds):
            if orders[r] and conv:
                pr.stage("ii_snapshot", n * dk, fo, prev, flag)
            for j in orders[r]:
                coef, inter, means = pr.alloc(dk), pr.alloc(1), pr.alloc(dk)
                nbl = self._neighbours(corr, j, dk) if corr is not None else None
                nb1 = pr.put_list([1 if a in nbl else 0 for a in range(dk)]) + 1 if nbl is not None else 0
                pp = len(nbl) if nbl is not None else dk - 1
                al = pr.alloc(2) if self.sample_posterior else -1
                seq.append((j, coef, inter))
                pr.stage("ii_mean", dk, fo, n, dk, mo, j, means, cnt, flag)
                pr.stage("ii_gram", dk * dk, fo, n, dk, mo, j, means, g, flag)
                if pp > 0:
                    pr.stage("ii_sub", 1, g, dk, j, gs, flag, nb1)
                    pr.stage("eigh", 1, gs, pp, 0, eig, vec, 1)  # q[5]=1: cyclic eigh_unit, not round-robin
                pr.stage("ii_br", 1, g, dk, j, eig, vec, means, cnt, coef, inter, flag, w, nb1, al + 1)
                if self.sample_posterior:
                    sig = pr.alloc(max(pp, 1) ** 2)
                    pr.stage("ii_sigma", pp * pp, eig, vec, pp, al, sig, flag)
                    key = pr.put_ints([seed, self._post_step])
                    self._post_step += 1
                    pr.stage("ii_post", n, fo, n, dk, mo, j, coef, inter, bo, flag, means, sig, al, nb1, key)
                    extra.append((nbl, sig, al, means, pp))
                else:
                    pr.stage("ii_predict", n, fo, n, dk, mo, j, coef, inter, bo, flag)
            if orders[r] and conv:
                if rowabs is None:
                    rowabs = pr.alloc(n)
                pr.stage("ii_rowabs", n, fo, prev, dk, rowabs, flag)
                pr.stage("ii_conv", 1, fo, prev, n * dk, tol, flag, niter, dk, rowabs + 1)
        pr.run(mode)
        any_order = any(orders)
        done = (int(pr.values(niter, 1)[0]) if conv else rounds) if any_order else 0
        self.n_iter_ = done if any_order else min(1, rounds)
        steps = sum(len(o) for o in orders[:done])
        self.imputation_sequence_ = [(j, pr.get(c, dk), pr.get(i, 1)) for j, c, i in seq[:steps]]
        self._posterior = [(nbl, pr.get(sg, max(pp, 1) ** 2), pr.get(al, 2), pr.get(mn, dk))
                           for nbl, sg, al, mn, pp in extra[:steps]]
        return self._with_indicator(arr, pr.get(fo, (n, dk)))

    def _with_indicator(self, arr, Xt):
        """Xt, then MissingIndicator's columns (add_indicator)."""
        m = len(self._indicator)
        if not m:
            return Xt
        n, d = arr.shape
        pr = _Prog()
        xo = _mark_missing(pr, pr.put(arr), n * d, self.missing_values)
        mo = pr.alloc(n * m)
        pr.stage("nan_mask", n * m, xo, n, d, pr.put_list(self._indicator), m, mo)
        dk = Xt.shape[1]
        to = pr.put(Xt)
        hc = pr.output(n * (dk + m))
        pr.stage("hcat", n * (dk + m), to, dk, mo, m, hc)
        pr.run(self.numeric_mode_)
        return pr.get(hc, (n, dk + m))

    def _fit_user(self, arr, Xf, orders, corr, tol):
        """estimator=<any>: the reference's rounds over copies of the
        estimator, its fit / predict on the float32 blocks. The data-sized
        plumbing runs as x_prep units (`_ii_take`, `_ii_put`; the device on a
        GPU install, the same units on the host on a CPU-only one): the
        observed / missing rows of feature j by a chunked count, scan and
        write, the estimator's blocks by a gather, the clipped predictions by
        a scatter, and the round's stop by `ii_rowabs` / `ii_conv`, the
        default estimator's own test (the matrix inf-norm of the round's
        change < tol, float32)."""
        n, d = arr.shape
        dk = len(self._keep)
        mode = _mode()
        mask = self._mask_arr(arr)
        Xt = as_f32_c(Xf, name="X")[0].copy()
        seq = []
        done = 0
        for r, order in enumerate(orders):
            check = not self.sample_posterior and bool(order)
            prev = Xt.copy() if check else None
            for j in order:
                nbl = self._neighbours(corr, j, dk) if corr is not None else [a for a in range(dk) if a != j]
                est = _clone(self.estimator)
                Xo, yo, Xm, rows, m = self._ii_take(Xt, mask, j, nbl, mode, fit=True)
                est.fit(Xo, yo)
                seq.append((j, nbl, est))
                Xt = self._ii_put(Xt, Xm, rows, m, j, est, mode)
            done = r + 1
            if check and self._ii_stop(Xt, prev, tol, mode):
                break
        self.n_iter_ = done if any(orders) else min(1, len(orders))
        self.imputation_sequence_ = seq
        return Xt

    def _ii_rows(self, pr, mo, n, dk, j, missing):
        """Stages: the ascending rows whose mask word for feature j is (or is
        not) zero, as int32 words, and their count (a float slot)."""
        ch = max(_II_CH, -(-n // _II_CH))
        nch = -(-n // ch)
        cnt, off, tot, rows = pr.alloc(nch), pr.alloc(nch), pr.alloc(1), pr.alloc(n)
        pr.stage("ii_rcount", nch, mo, n, dk, j, 1 if missing else 0, ch, cnt)
        pr.stage("uniq_scan", 1, cnt, nch, off, tot)
        pr.stage("ii_rwrite", nch, mo, n, dk, j, 1 if missing else 0, ch, off, rows)
        return rows, tot

    def _ii_take(self, Xt, mask, j, nbl, mode, fit):
        """One program: feature j's missing rows and the estimator's predict
        block Xt[mis][:, nbl] and, with fit, its training blocks
        Xt[obs][:, nbl] and Xt[obs, j]. Returns (Xo, yo, Xm, mis rows as int32
        words, m)."""
        n, dk = Xt.shape
        nc = len(nbl)
        pr = _Prog()
        xo, mo = pr.put(Xt), pr.put(mask)
        co = pr.put_list(nbl or [0])
        jo = pr.put_list([j])
        rm, tm = self._ii_rows(pr, mo, n, dk, j, True)
        gm = pr.alloc(n * nc)
        if nc:
            pr.stage("ii_gather", n * nc, xo, dk, rm, tm, co, nc, gm)
        if fit:
            ro, to = self._ii_rows(pr, mo, n, dk, j, False)
            go, gy = pr.alloc(n * nc), pr.alloc(n)
            if nc:
                pr.stage("ii_gather", n * nc, xo, dk, ro, to, co, nc, go)
            pr.stage("ii_gather", n, xo, dk, ro, to, jo, 1, gy)
        pr.run(mode)
        m = int(pr.values(tm, 1)[0])
        Xo = yo = None
        if fit:
            mob = int(pr.values(to, 1)[0])
            Xo, yo = self._ii_block(pr, go, gy, mob, nc)
        Xm, _ = self._ii_block(pr, gm, None, m, nc)
        rows = pr.get_i32(rm, (m,)) if m else None
        return Xo, yo, Xm, rows, m

    @staticmethod
    def _ii_block(pr, xo, yo, m, nc):
        """(X block (m, nc), y (m,)) as float32 Arrays; a 0-row block is
        `from_list([])`, a 0-column one its m empty rows, as before."""
        if m == 0:
            return Array.from_list([], "<f4"), (Array.from_list([], "<f4") if yo is not None else None)
        X = pr.get(xo, (m, nc)) if nc else Array.from_list([[] for _ in range(m)], "<f4")
        return X, (pr.get(yo, (m,)) if yo is not None else None)

    def _ii_put(self, Xt, Xm, rows, m, j, est, mode):
        """Predict on the missing rows' block, then (one program) the clipped
        float32 store into Xt[rows, j]. sample_posterior's truncated normal
        draw stays Python float64 on the pinned normal cdf / inverse cdf
        (DEVIATION 6902) and is stored unclipped."""
        if not m:
            return Xt
        n, dk = Xt.shape
        lo, hi = self._bounds_k[2 * j], self._bounds_k[2 * j + 1]
        if not self.sample_posterior:
            v = _pred_f64(est.predict(Xm))
            clip = 1
        else:
            mus, sig = est.predict(Xm, return_std=True)
            v = Array._owned(array.array("d", [self._truncnorm_host(float(a), float(s_), lo, hi)
                                               for a, s_ in zip(_as_list(mus), _as_list(sig))]),
                             (m,), "<f8", "C")
            clip = 0
        k = min(v.size, m)      # the reference's zip stops at the shorter
        if k == 0:
            return Xt
        if v.size != k:
            v = Array._owned(array.array("d", v.tolist()[:k]), (k,), "<f8", "C")
        pr = _Prog()
        xo = pr.put(Xt, inout=True)
        ro = pr.put_words(rows)
        vo = pr.put(v.astype("<f4"))
        bo = pr.put_list([lo, hi])
        pr.stage("ii_scatter", k, xo, dk, ro, j, vo, bo, clip)
        pr.run(mode)
        return pr.get(xo, (n, dk))

    def _ii_stop(self, Xt, prev, tol, mode):
        """The round's stop: the inf-norm of Xt - prev below tol."""
        n, dk = Xt.shape
        pr = _Prog()
        xo, po = pr.put(Xt), pr.put(prev)
        to = pr.put_scalar(tol)
        flag, niter, rowabs = pr.alloc(1), pr.alloc(1), pr.alloc(n)
        pr.stage("ii_rowabs", n, xo, po, dk, rowabs, flag)
        pr.stage("ii_conv", 1, xo, po, n * dk, to, flag, niter, dk, rowabs + 1)
        pr.run(mode)
        return pr.values(flag, 1)[0] != 0

    def _mask_arr(self, arr):
        """`_mask`'s words (1.0 missing, 0.0 observed) as the float32 (n, dk) Array."""
        n, d = arr.shape
        pr = _Prog()
        xo = _mark_missing(pr, pr.put(arr), n * d, self.missing_values)
        mo = pr.alloc(n * len(self._keep))
        pr.stage("nan_mask", n * len(self._keep), xo, n, d, pr.put_list(self._keep), len(self._keep), mo)
        pr.run(_mode())
        return as_f32_c(pr.get(mo, (n, len(self._keep))), name="mask")[0]

    def _truncnorm_host(self, mu, sigma, lo, hi):
        """`_impute_one_feature`'s rule in Python float64: mu beyond a bound
        -> the bound, sigma <= 0 -> mu, else inversion of the truncated normal
        at a 53-bit splitmix64 uniform. DEVIATION 6902: statistics.NormalDist's
        cdf and inv_cdf formulas on the pinned erfc / log (`_pm.normal_cdf`,
        `_pm.normal_inv_cdf`); NormalDist itself calls the platform erfc and
        a C accelerator a compiler may contract, so its bits vary by host."""
        if mu < lo:
            return lo
        if mu > hi:
            return hi
        if not sigma > 0:
            return mu
        pa = 0.0 if lo == -math.inf else _pm.normal_cdf((lo - mu) / sigma)
        pb = 1.0 if hi == math.inf else _pm.normal_cdf((hi - mu) / sigma)
        u = ((self._draw() >> 11) + 0.5) * 2.0 ** -53
        pu = pa + u * (pb - pa)
        if pu <= 0:
            return lo
        if pu >= 1:
            return hi
        return min(max(mu + sigma * _pm.normal_inv_cdf(pu), lo), hi)

    def fit(self, X, y=None):
        self.fit_transform(X)
        return self

    def transform(self, X):
        if not hasattr(self, "imputation_sequence_"):
            raise RuntimeError("mojolearn: this IterativeImputer instance is not fitted yet")
        arr = _x2d(X)
        self._check_width(arr)
        n, d = arr.shape
        dk = len(self._keep)
        Xf = self.initial_imputer_.transform(arr)
        if self.estimator is not None:
            mode = self.numeric_mode_
            mask = self._mask_arr(arr)
            Xt = as_f32_c(Xf, name="X")[0].copy()
            for j, nbl, est in self.imputation_sequence_:
                _, _, Xm, rows, m = self._ii_take(Xt, mask, j, nbl, mode, fit=False)
                Xt = self._ii_put(Xt, Xm, rows, m, j, est, mode)
            return self._with_indicator(arr, Xt)
        pr = _Prog()
        fo, mo, bo = self._prepare(pr, arr, Xf, inout=True)
        seed = self._rng & 0x7FFFFFFF
        for s, (j, coef, inter) in enumerate(self.imputation_sequence_):
            co, io = pr.put(coef), pr.put(inter)
            if self.sample_posterior:
                nbl, sig, al, means = self._posterior[s]
                nb1 = pr.put_list([1 if a in nbl else 0 for a in range(dk)]) + 1 if nbl is not None else 0
                key = pr.put_ints([seed, self._post_step])
                self._post_step += 1
                pr.stage("ii_post", n, fo, n, dk, mo, j, co, io, bo, _NONE, pr.put(means), pr.put(sig), pr.put(al),
                         nb1, key)
            else:
                pr.stage("ii_predict", n, fo, n, dk, mo, j, co, io, bo, _NONE)
        pr.run(self.numeric_mode_)
        return self._with_indicator(arr, pr.get(fo, (n, dk)))


def _clone(est):
    """A fresh unfitted copy (the reference's `clone`: the same class and
    get_params(deep=False))."""
    return type(est)(**est.get_params(deep=False)) if hasattr(est, "get_params") else copy.deepcopy(est)


def _as_list(v):
    return [float(x) for x in (v.tolist() if hasattr(v, "tolist") else v)]


def _pred_f64(v):
    """A user estimator's predictions as a float64 Array holding the values
    `_as_list` reads (`float(x)` per element): a float32 or float64 vector
    widened natively (exact), anything else through `_as_list`."""
    from ._buffer import _materialize, _has_buffer, as_f64_c
    if isinstance(v, Array) or (not isinstance(v, (list, tuple)) and _has_buffer(v)):
        a, _ = _materialize(v, "predict")
        if a.ndim == 1 and a.dtype in ("<f4", "<f8") and a.size:
            return as_f64_c(a, ndim=1, name="predict")[0]
    vals = _as_list(v)
    return Array._owned(array.array("d", vals), (len(vals),), "<f8", "C")


# ---------------------------------------------------------------- feature selection
class _SelectorMixin(_PrepBase):
    def get_support(self, indices=False):
        self._check_fitted()
        mask = list(self._mask)
        return [j for j, m in enumerate(mask) if m] if indices else mask

    def transform(self, X):
        self._check_fitted()
        arr = _x2d(X)
        self._check_width(arr)
        n, d = arr.shape
        keep = [j for j, m in enumerate(self._mask) if m]
        if not keep:
            raise ValueError("mojolearn: no features were selected")
        pr = _Prog()
        xo, ko = pr.put(arr), pr.put_list(keep)
        out = pr.alloc(n * len(keep))
        pr.stage("gather_cols", n * len(keep), xo, n, d, ko, len(keep), out)
        pr.run(self.numeric_mode_)
        return pr.get(out, (n, len(keep)))


class VarianceThreshold(_SelectorMixin):
    """sklearn.feature_selection.VarianceThreshold: population variance per
    column over the non-NaN entries (and, at threshold 0, min(variance,
    max - min), so a constant column is exactly 0); keeps the columns whose
    variance exceeds the threshold."""
    _parameters = ("threshold",)

    def __init__(self, threshold=0.0):
        self.threshold = threshold

    def fit(self, X, y=None):
        arr = _x2d(X)
        n, d = arr.shape
        mode = _mode()
        pr = _Prog()
        xo = pr.put(arr)
        st, var = pr.alloc(6 * d), pr.alloc(d)
        pr.stage("col_stats", d, xo, n, d, st)
        pr.stage("var_ptp", d, st, d, var, 1 if self.threshold == 0 else 0)
        pr.run(mode)
        self.variances_ = pr.get(var, d)
        self._mask = [v > self.threshold for v in pr.values(var, d)]
        if not any(self._mask):
            raise ValueError(f"mojolearn: No feature in X meets the variance threshold {self.threshold:.5f}")
        self.numeric_mode_, self.n_features_in_ = mode, d
        return self


def _scores_classif(X, y, kind):
    arr = _x2d(X)
    n, d = arr.shape
    classes, codes = encode_labels(y)
    if codes.size != n:
        raise ValueError("mojolearn: X and y have different numbers of rows")
    K = len(classes)
    pr = _Prog()
    xo, yo = pr.put(arr), pr.put_codes(codes)
    cnt, mean, sums = pr.alloc(K), pr.alloc(K * d), pr.alloc(K * d)
    sc, pv, st = pr.alloc(d), pr.alloc(d), pr.alloc(6 * d)
    pr.stage("class_stats", K * d, xo, n, d, yo, K, cnt, mean, _NONE, sums)
    if kind == "chi2":
        pr.stage("col_stats", d, xo, n, d, st)
        pr.stage("chi2", d, sums, K, d, cnt, n, sc, pv)
    else:
        pr.stage("f_classif", d, xo, n, d, yo, K, cnt, mean, sc, pv)
    pr.run(_mode())
    if kind == "chi2" and any(v < 0 for v in pr.values(st + 3 * d, d)):
        raise ValueError("mojolearn: Input X must be non-negative.")
    return pr.get(sc, d), pr.get(pv, d)


def f_classif(X, y):
    """sklearn.feature_selection.f_classif: the one-way ANOVA F of each
    feature against the classes, and its p-value (float32). As the
    reference: a constant feature (or a single class) scores NaN with p-value
    NaN, a feature constant within every class but not across them +inf with
    p-value 0."""
    return _scores_classif(X, y, "f")


def chi2(X, y):
    """sklearn.feature_selection.chi2 for non-negative X: chi-squared of the
    class-by-feature sums against their expectation, and its p-value; an
    all-zero feature is NaN with p-value NaN, as the reference."""
    return _scores_classif(X, y, "chi2")


def _pearson(X, y, center, force_finite):
    arr = _x2d(X)
    n, d = arr.shape
    yv = as_f32_c(y, ndim=1, name="y")[0]
    if yv.size != n:
        raise ValueError("mojolearn: X and y have different numbers of rows")
    pr = _Prog()
    xo, yo = pr.put(arr), pr.put(yv)
    sc, pv, co = pr.alloc(d), pr.alloc(d), pr.alloc(d)
    pr.stage("f_regression", d, xo, n, d, yo, 1 if center else 0, sc, pv, co, 1 if force_finite else 0)
    pr.run(_mode())
    return pr.get(sc, d), pr.get(pv, d), pr.get(co, d)


def f_regression(X, y, *, center=True, force_finite=True):
    """sklearn.feature_selection.f_regression: F of each feature's Pearson r
    with y, and its p-value. force_finite=True (the default) gives the
    reference's edge values (F 0, p 1 for a constant feature or target;
    float32 max, p 0 for |r| = 1); force_finite=False its raw NaN and +inf."""
    sc, pv, _ = _pearson(X, y, center, force_finite)
    return sc, pv


def r_regression(X, y, *, center=True, force_finite=True):
    """sklearn.feature_selection.r_regression: each feature's Pearson r with
    y (float32); a constant feature or target is 0 with force_finite, NaN
    without it."""
    return _pearson(X, y, center, force_finite)[2]


class SelectKBest(_SelectorMixin):
    """sklearn.feature_selection.SelectKBest: the k highest scores of
    `score_func` (this module's f_classif, chi2 or f_regression compute on
    the device; any other callable is called as is), ties broken as the
    reference's stable argsort does (the later column wins); k='all'."""
    _parameters = ("score_func", "k")

    def __init__(self, score_func=f_classif, *, k=10):
        self.score_func = score_func
        self.k = k

    def fit(self, X, y=None):
        arr = _x2d(X)
        d = arr.shape[1]
        out = self.score_func(arr, y)
        scores, pvals = out if isinstance(out, (tuple, list)) else (out, None)
        self.scores_, self.pvalues_ = scores, pvals
        vals = [float(v) for v in (scores.tolist() if hasattr(scores, "tolist") else scores)]
        vals = [(-float("inf") if v != v else v) for v in vals]
        if self.k == "all":
            self._mask = [True] * d
        else:
            k = int(self.k)
            if not 0 <= k <= d:
                raise ValueError(f"mojolearn: k should be 0 <= k <= n_features = {d}; got {k}")
            order = sorted(range(d), key=lambda j: vals[j])       # stable, ascending
            chosen = set(order[d - k:]) if k else set()
            self._mask = [j in chosen for j in range(d)]
        self.numeric_mode_, self.n_features_in_ = _mode(), d
        return self

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)


def _mi_discrete_mask(discrete_features, d):
    """The reference's discrete_features: 'auto' (dense X: none), a bool for
    every column, a bool mask of length d, or column indices (negative ones
    count from the end)."""
    if isinstance(discrete_features, str):
        if discrete_features != "auto":
            raise ValueError("mojolearn: Invalid string value for discrete_features.")
        return [False] * d
    if isinstance(discrete_features, bool) or type(discrete_features).__name__ == "bool_":
        return [bool(discrete_features)] * d
    vals = discrete_features.tolist() if hasattr(discrete_features, "tolist") else list(discrete_features)
    if isinstance(vals, bool):
        return [vals] * d
    if vals and all(isinstance(v, bool) for v in vals):
        if len(vals) != d:
            raise ValueError(f"mojolearn: discrete_features mask has {len(vals)} entries; X has {d} features")
        return list(vals)
    mask = [False] * d
    for v in vals:
        if isinstance(v, bool) or not isinstance(v, numbers.Integral):
            raise ValueError("mojolearn: discrete_features must be 'auto', a bool, a bool mask or indices")
        j = int(v)
        if not -d <= j < d:
            raise IndexError(f"mojolearn: discrete_features index {j} is out of bounds for {d} features")
        mask[j % d] = True
    return mask


#: lane/apple-fast-mi (2026-10-03): the A/B switch MOJOLEARN_MI_WORK=1 (opt-in,
#: FAST only, read once at import): `_mutual_info` keeps the noised columns, their
#: noise words and the per-point terms (3 n d words) as device-only scratch
#: (`_Prog.work`) instead of arena words that the host zeroes and the device
#: copies back (265 MB on Istella). Where a word lives moves no bit.
_MI_WORK = os.environ.get("MOJOLEARN_MI_WORK", "0") == "1"


def _plus1(off):
    """off + 1 for an arena offset or a `_Scratch` (the units' offset + 1 noise words)."""
    return _Scratch(off.off + 1, off.kind) if isinstance(off, _Scratch) else off + 1


def _mutual_info(X, y, discrete_target, discrete_features, n_neighbors, random_state):
    """The reference's `_estimate_mi`: continuous columns are scaled and
    noised (row-major over the continuous columns only, as the reference's
    draw of shape (n, n_continuous)), the 1e-10-scaled noise kept as a
    second word that breaks exact ties as the reference's float64 sum does
    (DEVIATION 5407); each column then takes the estimator
    its kinds name: Kraskov (continuous x, continuous y), Ross (one side
    discrete: the classes, or a discrete feature's categories against the
    noised target) or the contingency table (both discrete)."""
    k = int(n_neighbors)
    if not 1 <= k <= 32:
        raise NotImplementedError("mojolearn: mutual_info supports 1 <= n_neighbors <= 32")
    arr = _x2d(X)
    n, d = arr.shape
    mask = _mi_discrete_mask(discrete_features, d)
    cont = [j for j in range(d) if not mask[j]]
    disc = [j for j in range(d) if mask[j]]
    mode = _mode()
    seed = 0 if random_state is None else int(random_state) & 0x3FFFFFFF
    pr = _Prog()
    work = pr.work if (_MI_WORK and mode == "fast") else pr.alloc
    if discrete_target:
        classes, codes = encode_labels(y)
        if codes.size != n:
            raise ValueError("mojolearn: X and y have different numbers of rows")
        counts = _class_counts(codes, len(classes))
        if cont and max(counts) < 2:
            raise ValueError("mojolearn: mutual_info: every class has one sample (the reference's "
                             "neighbour search over the classes with more than one finds 0 samples)")
        yo, lc = pr.put_codes(codes), pr.put_list(counts)
    else:
        yv = as_f32_c(y, ndim=1, name="y")[0]
        if yv.size != n:
            raise ValueError("mojolearn: X and y have different numbers of rows")
    outc = outd = None
    if cont:
        dc = len(cont)
        xo = pr.put(arr if not disc else _gather(arr, cont, mode))
        st, sc, ma, z, zs = pr.alloc(6 * dc), pr.alloc(dc), pr.alloc(dc), work(n * dc), work(n * dc)
        pr.stage("col_stats", dc, xo, n, dc, st)
        pr.stage("mi_colscale", dc, xo, n, dc, st, sc, ma)
        pr.stage("mi_noise", n * dc, xo, n, dc, sc, ma, 2 * seed, z, _plus1(zs))
    if not discrete_target:
        yo = pr.put(yv)
        sty, scy, may, zy, zys = pr.alloc(6), pr.alloc(1), pr.alloc(1), work(n), work(n)
        pr.stage("col_stats", 1, yo, n, 1, sty)
        pr.stage("mi_colscale", 1, yo, n, 1, sty, scy, may)
        pr.stage("mi_noise", n, yo, n, 1, scy, may, 2 * seed + 1, zy, _plus1(zys))
    if cont:
        term, outc = work(n * dc), pr.alloc(dc)
        if discrete_target:
            used = sum(c for c in counts if c > 1)
            pr.stage("mi_cd", n * dc, z, n, dc, yo, lc, k, term, _plus1(zs))
            pr.stage("mi_reduce", dc, term, n, dc, 1, k, used, outc)
        else:
            pr.stage("mi_cc", n * dc, z, n, dc, zy, k, term, _plus1(zs), _plus1(zys))
            pr.stage("mi_reduce", dc, term, n, dc, 0, k, n, outc)
    if disc:
        dd = len(disc)
        xd = _gather(arr, disc, mode)
        cats = _fit_categories(mode, xd)
        kx = [c.size for c in cats]
        kmax = max(kx)
        if not discrete_target and n in kx:
            raise ValueError(f"mojolearn: mutual_info: discrete feature {disc[kx.index(n)]} has one sample per "
                             "value (the reference's neighbour search finds 0 samples)")
        xc, _neg = _codes(pr, xd, cats)
        outd = pr.alloc(dd)
        if discrete_target:
            ky = len(classes)
            stride = kmax * ky + kmax + ky
            tb = pr.alloc(dd * stride)
            pr.stage("mi_dd", dd, xc, n, dd, yo, ky, pr.put_list(kx), tb, stride, outd)
        else:
            cnti, cntf, term = pr.alloc(dd * kmax), pr.alloc(dd * kmax), work(n * dd)
            pr.stage("code_counts", dd, xc, n, dd, kmax, cnti)
            pr.stage("i2f", dd * kmax, cnti, cntf)
            pr.stage("mi_dc", n * dd, zy, n, dd, xc, cntf, kmax, k, term, _plus1(zys))
            pr.stage("mi_reduce", dd, term, n, dd, 2, k, 0, outd, cntf, kmax)
    pr.run(mode)
    if not disc:
        return pr.get(outc, d)
    vals = [0.0] * d
    for j, v in zip(cont, pr.values(outc, len(cont)) if cont else []):
        vals[j] = v
    for j, v in zip(disc, pr.values(outd, len(disc))):
        vals[j] = v
    return Array.from_list(vals, "<f4")


def mutual_info_classif(X, y, *, discrete_features="auto", n_neighbors=3, copy=True, random_state=None,
                        n_jobs=None):
    """sklearn.feature_selection.mutual_info_classif for dense X: Ross's
    k-NN estimator against the classes for a continuous feature, the
    contingency mutual information for a discrete one (discrete_features),
    float32, brute-force neighbour scans on the device. The tie-breaking
    noise is drawn from random_state by splitmix64 (the reference draws
    numpy's)."""
    return _mutual_info(X, y, True, discrete_features, n_neighbors, random_state)


def mutual_info_regression(X, y, *, discrete_features="auto", n_neighbors=3, copy=True, random_state=None,
                           n_jobs=None):
    """sklearn.feature_selection.mutual_info_regression for dense X: the
    Kraskov k-NN estimator for a continuous feature, Ross's estimator with
    the feature's categories as the classes for a discrete one
    (discrete_features), float32, brute-force neighbour scans on the
    device; noise from random_state by splitmix64."""
    return _mutual_info(X, y, False, discrete_features, n_neighbors, random_state)


def _gather(arr, cols, mode):
    n, d = arr.shape
    pr = _Prog()
    xo, ko = pr.put(arr), pr.put_list(cols)
    out = pr.alloc(n * len(cols))
    pr.stage("gather_cols", n * len(cols), xo, n, d, ko, len(cols), out)
    pr.run(mode)
    return pr.get(out, (n, len(cols)))


def _importances(est, mode, getter="auto"):
    """The squared importance of each column of a fitted estimator: coef_
    squared (summed over rows when 2-D) on the device, else
    feature_importances_ as given (a monotone stand-in for its square).
    A str getter (a dotted attribute path, as operator.attrgetter) or a
    callable picks the importances instead; they are squared (summed over
    rows when 2-D) on the device, as the reference's transform_func='square'."""
    if getter != "auto":
        coef = operator.attrgetter(getter)(est) if isinstance(getter, str) else getter(est)
    else:
        coef = getattr(est, "coef_", None)
    if coef is None and getter == "auto":
        imp = getattr(est, "feature_importances_", None)
        if imp is None:
            raise ValueError("mojolearn: RFE needs an estimator with coef_ or feature_importances_")
        return [float(v) for v in (imp.tolist() if hasattr(imp, "tolist") else imp)]
    c = as_f32_c(coef, ndim=None, name="coef_")[0]
    rows, d = (1, c.shape[0]) if c.ndim == 1 else c.shape
    pr = _Prog()
    co = pr.put(c)
    out = pr.alloc(d)
    pr.stage("sqsum_cols", d, co, rows, d, out)
    pr.run(mode)
    return pr.values(out, d)


class RFE(_SelectorMixin):
    """sklearn.feature_selection.RFE: fit, rank by squared coef_ (summed over
    classes; or feature_importances_), drop the `step` weakest, repeat.
    Ties are broken by a STABLE ascending sort (the lower column index is
    dropped first); the reference's quicksort leaves them unspecified.
    importance_getter 'auto', a dotted attribute path or a callable, as the
    reference."""
    _parameters = ("estimator", "n_features_to_select", "step", "verbose", "importance_getter")

    def __init__(self, estimator, *, n_features_to_select=None, step=1, verbose=0, importance_getter="auto"):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.verbose = verbose
        self.importance_getter = importance_getter

    def _clone(self):
        est = self.estimator
        return type(est)(**est.get_params()) if hasattr(est, "get_params") else est

    def fit(self, X, y, **fit_params):
        g = self.importance_getter
        if not (callable(g) or isinstance(g, str)):
            raise ValueError("mojolearn: RFE importance_getter must be 'auto', a str or a callable")
        arr = _x2d(X)
        n, d = arr.shape
        mode = _mode()
        nsel = self.n_features_to_select
        if nsel is None:
            nsel = d // 2
        elif isinstance(nsel, float) and 0 < nsel < 1:
            nsel = int(nsel * d)
        nsel = max(1, int(nsel))
        step = int(max(1, self.step * d)) if isinstance(self.step, float) and self.step < 1 else int(self.step)
        if step <= 0:
            raise ValueError("mojolearn: step must be > 0")
        support = [True] * d
        ranking = [1] * d
        while sum(support) > nsel:
            features = [j for j in range(d) if support[j]]
            est = self._clone().fit(_gather(arr, features, mode), y, **fit_params)
            imp = _importances(est, mode, self.importance_getter)
            ranks = sorted(range(len(features)), key=lambda r: imp[r])
            threshold = min(step, sum(support) - nsel)
            for r in ranks[:threshold]:
                support[features[r]] = False
            for j in range(d):
                if not support[j]:
                    ranking[j] += 1
        features = [j for j in range(d) if support[j]]
        self.estimator_ = self._clone().fit(_gather(arr, features, mode), y, **fit_params)
        self._mask, self.support_ = support, list(support)
        self.ranking_ = Array.from_list(ranking, "<i8")
        self.n_features_ = sum(support)
        self.numeric_mode_, self.n_features_in_ = mode, d
        return self

    def predict(self, X):
        return self.estimator_.predict(self.transform(X))

    def predict_proba(self, X):
        return self.estimator_.predict_proba(self.transform(X))

    def decision_function(self, X):
        return self.estimator_.decision_function(self.transform(X))

    def score(self, X, y):
        return self.estimator_.score(self.transform(X), y)


class ComplementNB(_DiscreteNB):
    """sklearn.naive_bayes.ComplementNB, float32: complement class feature
    counts, their log share (negated, or normalised when `norm`); the class
    prior enters only with a single class, as in the reference. alpha > 0
    required; class_prior as given (its log); sample_weight weights the counts, as the
    reference; partial_fit adds each batch's counts, as the reference."""
    _parameters = ("alpha", "force_alpha", "fit_prior", "class_prior", "norm")

    def __init__(self, *, alpha=1.0, force_alpha=True, fit_prior=True, class_prior=None, norm=False):
        self.alpha = alpha
        self.force_alpha = force_alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        self.norm = norm

    def _params(self, pr, mode, n, d, K, st, cnt, fc, clp):
        a = pr.put_scalar(self.alpha)
        flp = pr.alloc(K * d)
        pr.stage("cnb_params", K, fc, K, d, a, 1 if self.norm else 0, flp)
        pr.run(mode)
        _check_nonnegative(pr.values(st + 3 * d, d), "ComplementNB (input X)")
        self._finish_counts(pr, mode, d, K, cnt, fc, clp)
        self.feature_log_prob_ = pr.get(flp, (K, d))
        return self

    def _jll_stages(self, pr, xo, n, d, out):
        K = len(self.classes_)
        w = pr.put(self.feature_log_prob_)
        b = pr.put(self.class_log_prior_) if K == 1 else _NONE
        pr.stage("matmul", n * K, xo, d, 1, w, 1, d, out, K, d, b, _NONE)


class CategoricalNB(_DiscreteNB):
    """sklearn.naive_bayes.CategoricalNB, float32: X holds category indices
    0, 1, ...; per feature, class and category the smoothed log share of the
    class's rows. A category index outside the fitted range at predict time
    is refused, as the reference refuses it; class_prior as given (its log);
    min_categories floors n_categories_; sample_weight weights the counts,
    as the reference. partial_fit adds each batch's category counts to the
    running ones (category_count_, widened as new categories appear) and
    recomputes the log probabilities; n_categories_ is the counts' width
    (the reference narrows it to the last batch's, see
    naive_bayes/NOT_IMPLEMENTED.tsv)."""
    _parameters = ("alpha", "force_alpha", "fit_prior", "class_prior", "min_categories")

    def __init__(self, *, alpha=1.0, force_alpha=True, fit_prior=True, class_prior=None, min_categories=None):
        self.alpha = alpha
        self.force_alpha = force_alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        self.min_categories = min_categories

    def fit(self, X, y, sample_weight=None):
        _check_alpha(self)
        arr = _x2d(X)
        codes = self._encode_y(y, arr.shape[0])
        return self._cat_fit(arr, codes, sample_weight, _mode(), False)

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        _check_alpha(self)
        arr = _x2d(X)
        first, codes = _partial_codes(self, y, classes, arr.shape[0])
        if first:
            return self._cat_fit(arr, codes, sample_weight, _mode(), False)
        self._check_width(arr)
        return self._cat_fit(arr, codes, sample_weight, self.numeric_mode_, True)

    def _cat_fit(self, arr, codes, sample_weight, mode, merge):
        # the two programs share one device copy of X (and of the codes)
        with _arena_io.resident():
            return self._cat_fit_programs(arr, codes, sample_weight, mode, merge)

    def _cat_fit_programs(self, arr, codes, sample_weight, mode, merge):
        n, d = arr.shape
        K = len(self.classes_)
        pr = _Prog()
        xo, yo = pr.put(arr), pr.put_codes(codes)
        st, cnt, clp = pr.alloc(6 * d), pr.alloc(K), pr.alloc(K)
        wo = _nb_weights(pr, sample_weight, n)
        _col_stats(pr, xo, n, d, st, var=False)
        if merge:
            bc = pr.alloc(K)
            _class_stats(pr, wo, K, xo, n, 1, yo, K, bc, _NONE, _NONE, _NONE)
            pr.stage("add_arrays", K, pr.put(self.class_count_), bc, cnt)
        else:
            _class_stats(pr, wo, K, xo, n, 1, yo, K, cnt, _NONE, _NONE, _NONE)
        if self.class_prior is not None:
            pr.stage("log", K, pr.put_list(_given_priors(self.class_prior, K, "CategoricalNB")), clp)
        else:
            pr.stage("class_log_prior", K, cnt if self.fit_prior else pr.put_list([1.0] * K), K, clp)
        pr.run(mode)
        lo, hi = pr.values(st + 3 * d, d), pr.values(st + 4 * d, d)
        if any(v < 0 for v in lo):
            raise ValueError("mojolearn: Negative values in data passed to CategoricalNB (input X)")
        ncat = [int(v) + 1 for v in hi]
        if self.min_categories is not None:
            mc = self.min_categories
            mcs = [int(v) for v in (mc.tolist() if hasattr(mc, "tolist") else mc)] \
                if not isinstance(mc, numbers.Integral) else [int(mc)] * d
            if len(mcs) != d:
                raise ValueError(f"mojolearn: 'min_categories' should have shape ({d},) when an array-like "
                                 f"is provided. Got {len(mcs)} entries instead.")
            ncat = [max(a, b) for a, b in zip(ncat, mcs)]
        if merge:
            ncat = [max(a, b) for a, b in zip(ncat, self.n_categories_.tolist())]
        cmax = max(ncat)
        q = _Prog()
        xo, yo = q.put(arr), q.put_codes(codes)
        no, co, a = q.put_list(ncat), q.put(pr.get(cnt, K)), q.put_scalar(self.alpha)
        wq = _nb_weights(q, sample_weight, n)
        cc = q.alloc(d * K * cmax)
        if _blocked():
            # x_prep/blocked.mojo: a histogram per (row block, feature), then
            # each (feature, class, category) slot over the blocks (exact
            # integer counts: cat_counts' words)
            rows = _XB
            while ((n + rows - 1) // rows) * d * K * cmax > _CAT_HIST_WORDS and rows < 2 ** 22:
                rows *= 2
            nbh = (n + rows - 1) // rows
            hist = q.work(nbh * d * K * cmax)
            w = _NONE if wq is None else wq
            q.stage("cat_hpart", nbh * d, xo, n, d, yo, K, no, cmax, w, hist, rows)
            q.stage("cat_hfold", d * K * cmax, hist, nbh, d, K, no, cmax, w, cc)
        else:
            q.stage("cat_counts", d * K * cmax, xo, n, d, yo, K, no, cmax, _NONE if wq is None else wq, cc)
        if merge:
            old, oc = self._cc.tolist(), self._cmax
            pad = [0.0] * (d * K * cmax)
            for jk in range(d * K):
                pad[jk * cmax:jk * cmax + oc] = old[jk * oc:(jk + 1) * oc]
            src, cc = cc, q.alloc(d * K * cmax)
            q.stage("add_arrays", d * K * cmax, q.put_list(pad), src, cc)
        flp = q.alloc(d * K * cmax)
        q.stage("cat_flp", d * K * cmax, cc, K, no, cmax, co, a, flp)
        q.run(mode)
        self._cc = q.get(cc, d * K * cmax)
        # each (feature, class) row read once (the same values)
        ccv, flv = q.values(cc, d * K * cmax), q.values(flp, d * K * cmax)
        self.category_count_ = [Array.from_list(
            [ccv[(j * K + k) * cmax:(j * K + k) * cmax + ncat[j]] for k in range(K)], "<f4") for j in range(d)]
        self.n_categories_ = Array.from_list(ncat, "<i8")
        self._flp, self._cmax = q.get(flp, d * K * cmax), cmax
        self.feature_log_prob_ = [Array.from_list(
            [flv[(j * K + k) * cmax:(j * K + k) * cmax + ncat[j]] for k in range(K)], "<f4") for j in range(d)]
        self.class_count_, self.class_log_prior_ = pr.get(cnt, K), pr.get(clp, K)
        self.numeric_mode_, self.n_features_in_ = mode, d
        return self

    def _score_checks(self, pr, xo, n, d):
        """The input's column minima and maxima in the scoring program itself
        (one upload of X; cat_jll reads nothing outside the table for an index
        out of range, and the run is refused after it)."""
        st = pr.alloc(6 * d)
        _col_stats(pr, xo, n, d, st, var=False)
        return st

    def _score_refusals(self, pr, d, st):
        ncat = self.n_categories_.tolist()
        if any(v < 0 for v in pr.values(st + 3 * d, d)) or \
                any(int(v) >= c for v, c in zip(pr.values(st + 4 * d, d), ncat)):
            raise IndexError("mojolearn: CategoricalNB got a category index outside the fitted range")

    def _jll_stages(self, pr, xo, n, d, out):
        K = len(self.classes_)
        fo, co = pr.put(self._flp), pr.put(self.class_log_prior_)
        pr.stage("cat_jll", n * K, xo, n, d, fo, K, self._cmax, co, out)
