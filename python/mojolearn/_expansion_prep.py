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
truncated-normal posterior draw (in Mojo since lane py-runtime round 3:
bindings/normal_dist_helpers.mojo) and the O(d^2) `_abs_corr` /
`_neighbours` normalisation (n_nearest_features). IterativeImputer with a
user `estimator` otherwise runs its data-sized plumbing as x_prep units
(x_prep/iterative.mojo `ii_rcount` .. `ii_scatter`: row selection, gathers,
the clipped float32 store; the stop is `ii_rowabs` / `ii_conv`).
"""
import array
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
from ._buffer import as_f32_c, as_i32_c, addr_ro, addr as _addr_rw, _as_typed, empty, full
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
    centered_matmul=178, unique_inverse=177, sort_cols=0, col_stats=1, quantile=2, affine=3, scale_params=4, unique_cols=5, mode_cols=6,
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
    # lane apple-fast-gap-cls2 (x_prep/cat_cls2.mojo): only the FAST + Apple binding (default;
    # -D MOJOLEARN_X_PREP_FAST_CLS2_PACK_OFF has none) runs them (it exports x_prep_cls2_cat)
    cat_zero=157, cat_present=158, pres_count=159, pres_write=160, cat_pack=161,
    # lane idn-int-prep (x_prep/blocked.mojo): the IDENTICAL bindings (device and host column;
    # -D MOJOLEARN_IDN_NB_ONEPASS_OFF has none) run them (`_idn_int` bit 2); IDENTICAL also
    # compiles 157-160 (x_prep/label_fast.mojo IDN_LABEL)
    csb1_part=162, csb1_neg=163,
    # lane idn-all (x_prep/blocked.mojo `csr_dense_unit`): the IDENTICAL device binding
    # (`_idn_int` bit 8; -D MOJOLEARN_IDN_NB_CSR_DENSE_OFF has none) densifies a CSR input
    csr_dense=164,
    # lane fam-prep-metrics: the IDENTICAL bindings (device and host column) run them
    # (`_idn_fam` bit 2: csb1_ss, x_prep/blocked.mojo, -D MOJOLEARN_IDN_CLASS_ONEPASS_OFF has
    # none; bit 4: the blocked univariate scores, x_prep/select_blocked.mojo,
    # -D MOJOLEARN_IDN_SELECT_BLOCKED_OFF has none)
    csb1_ss=165, fcb_part=166, fcb_fin=167, frb_part1=168, frb_mean=169, frb_part2=170, frb_fin=171,
    # bit 8: PowerTransformer's blocked folds, x_prep/pt_blocked.mojo, -D MOJOLEARN_IDN_PT_BLOCKED_OFF has none
    ptb_part1=172, ptb_mean=173, ptb_part2=174, ptb_fin=175, ptb_step=176,
    # lane apple-fast-py2mojo-prep (x_prep/py2mojo.mojo, a range of its own): every binding
    # runs them (lane pyglue-numeric deleted the OFF arm and its Python loops)
    p2m_ccount=200, p2m_cscan=201, p2m_cstart=202, p2m_cwrite=203, p2m_rgather=204, p2m_smrows=205,
    p2m_sel_count=206, p2m_sel_scan=207, p2m_sel_write=208, p2m_transpose=209, p2m_rowflag=210,
    p2m_abscorr_cell=211, p2m_abscorr_norm=212,
    # lane fam2-prep-metrics (x_prep/fam2.mojo, a range of its own): every binding runs
    # 230-235 and 241 since lane cpu2-l3-prep (they were IDENTICAL-only)
    f2_wblk=230, f2_wscan=231, f2_wdraw=232, f2_perm_rows=233, f2_wpick=234, f2_clamp0=235,
    # the IDENTICAL tiled Gram (x_prep/gram_blocked.mojo): bit 16; gb_part_row is the bit 32 candidate
    gb_part=236, gb_part_row=237, gb_fold=238, qcb_part=239, qcb_fold=240,
    f2_code_gather=241,
    # lane cpu2-l3-prep (x_prep/cpu2_prep.mojo, in fam2's range): every binding, every tier
    c2_bin_code=242, c2_kfold=243, c2_strat_meta=244, c2_strat_flag=245, c2_strat_fold=246, c2_integral=247,
    c2_isum=248, c2_inf_m0=249, c2_inf_m1=250, c2_inf_map=251, c2_imp_stats=252, c2_key64=253, c2_topk=254,
    c2_gt=255, c2_rfe_step=256, c2_rfe_rank=257, c2_ii_miss=258, c2_ii_pos=259, c2_ii_ord=260, c2_ii_rand=261,
    c2_colmax=262, c2_grid=263, c2_nan_sub=264, c2_nan_rows=265, c2_cnt0=266, c2_mm_keep=267, c2_mm_merge=268,
    c2_add_i64=269, c2_nonfinite=270,
    # lane apple-fast-q-clf (x_prep/proba64.mojo, in the py2mojo range): staged only when the
    # binding exports x_prep_proba64 (FAST, not -D MOJOLEARN_PROBA64_QOLD)
    q64_softmax=213,
)
_PARAMS = 14
_NONE = -1
#: x_prep/transform.mojo PT_EVALS (PT_ITERS + 2) and PT_STATE
_PT_EVALS = 50
_PT_STATE = 10


def _native_helper(key):
    """The base binding's host helper `key` (`_buffer._native`)."""
    from ._buffer import _native
    return _native(key)


def _prep_binding(mode):
    return _backend.binding("_mojolearn_x_prep", mode)


def _optional_prep_entry(binding, name):
    """Probe optional exports without hiding a missing mandatory host implementation."""
    binding.x_prep_run  # Load/validate the family before interpreting a refusal as absence.
    try:
        return getattr(binding, name)
    except (AttributeError, ImportError):
        return None


def _p2m_chunks(n, per=1):
    """(rows a chunk, chunks) for the chunk-parallel p2m units: at most
    _II_CH chunks, and at most 2**24 count words when each chunk keeps `per`."""
    ch = max(_II_CH, -(-n // _II_CH), -(-n * per // 2 ** 24), 1)
    return ch, max(1, -(-n // ch))


def _p2m_class_rows(pr, co, n, K, rows=True):
    """Stages (p2m_ccount .. p2m_cwrite): the class counts of the float codes
    at `co` (int32 words, TOT) and, with rows, the rows grouped by class,
    ascending inside a class (int32 words, ROWS). Returns (ROWS or None, TOT)."""
    ch, nch = _p2m_chunks(n, K)
    cnt, tot = pr.alloc(nch * K), pr.alloc(K)
    pr.stage("p2m_ccount", nch, co, n, K, ch, cnt)
    pr.stage("p2m_cscan", K, cnt, nch, K, tot)
    if not rows:
        return None, tot
    start, ro = pr.alloc(K), pr.alloc(n)
    pr.stage("p2m_cstart", 1, tot, K, start, _NONE)
    pr.stage("p2m_cwrite", nch, co, n, K, ch, cnt, start, ro)
    return ro, tot


def _p2m_sel(pr, xo, n, d, mode, dst):
    """Stages (p2m_sel_count .. p2m_sel_write): per column j of the (n, d)
    block at xo, the selected rows ascending at dst + j*n (mode 0: the
    non-NaN words; mode 1: the indices of the positive rows). Returns the
    offset of the d counts (int32 words)."""
    ch, nch = _p2m_chunks(n)
    cnt, tot = pr.alloc(d * nch), pr.alloc(d)
    pr.stage("p2m_sel_count", d * nch, xo, n, d, ch, nch, mode, cnt)
    pr.stage("p2m_sel_scan", d, cnt, nch, tot)
    pr.stage("p2m_sel_write", d * nch, xo, n, d, ch, nch, mode, cnt, dst)
    return tot


def _p2m_positive_rows(pr, xo, wo, n, d, m):
    """Stages: the rows of the (n, d) block at xo whose weight (at wo) is
    positive, as a new (m, d) block (the old `_gather_rows(arr, nz)`; m is
    the count of positive weights). Returns its offset."""
    ro = pr.alloc(n)
    _p2m_sel(pr, wo, n, 1, 1, ro)
    xz = pr.alloc(m * d)
    if m:
        pr.stage("p2m_rgather", m * d, xo, d, ro, xz)
    return xz


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


#: lane/apple-fast-prep2: the QSELECT switch, read once at import (not on a fit path)
_X_PREP2_QSELECT = os.environ.get("MOJOLEARN_X_PREP_FAST_QSELECT") == "1"


def _prep2_qselect(mode, nq):
    """lane/apple-fast-prep2 (2026-10-02): MOJOLEARN_X_PREP_FAST_QSELECT=1 on
    the FAST tier of a Metal binding (every other build ignores it) takes
    SimpleImputer's median and RobustScaler's quantiles by a device radix
    select over the unsorted columns (x_prep/fastprep2.mojo qselect_device:
    the `quantile` stage with SELECT = 1 as its 8th parameter) instead of the
    full radix sort of every column (sort_cols + quantile, x_prep/dradix.mojo:
    four read + random-scatter passes over two n*d key blocks and the n*d
    sorted block). The same order statistics, so the same words. At most 4
    fractions per column (the select's histogram block carries two tasks
    per fraction)."""
    if nq > 4 or not _X_PREP2_QSELECT or str(mode).strip().lower() != "fast":
        return False
    b = _prep_binding(mode)
    if _optional_prep_entry(b, "x_prep_host_column") is not None:
        return False
    vendor = _optional_prep_entry(b, "x_prep_vendor")
    return vendor is not None and str(vendor()) == "metal"


def _r3(name):
    """Whether the lane prep-apple3 change `name` is on."""
    def names(var):
        v = [t.strip() for t in os.environ.get(var, "").replace("+", ",").split(",") if t.strip()]  # glue: parses an env var name list
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
        gathered = []
        for off, arr, _ in self._inputs:  # glue: walks the program input buffers once
            if not arr.size:
                continue
            if isinstance(arr, _arena_io.DeviceRows):
                # device-rows input (lane cpu4-misc): the fold's rows gathered
                # on the device out of the base X, put once into this binding's
                # store; a binding without the gather copies the host rows
                slot = None
                if run_ranges is not None:
                    if direct is None and _arena_io.DeviceCache.supports(binding, "x_prep"):
                        direct = _arena_io.DeviceCache(binding, "x_prep")
                    slot = _arena_io.rows_slot(binding, "x_prep", arr, direct)
                if slot is not None:
                    gathered.append(slot)
                    spans.append((off, off + arr.size, slot))
                    continue
                arr = arr.materialize()
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
        self._in_spans = [(lo, hi) for lo, hi, _ in spans if (lo, hi) not in self._inout]  # glue: walks the program span records
        prog = array.array("i", [(v.off + (sbase if v.kind == "s" else obase)) if isinstance(v, _Scratch) else v
                                 for s in self._stages for v in s] or [0])  # glue: flattens the stage parameter records
        nst = len(self._stages)
        self._out, self._out_at = None, obase
        t_in = time.perf_counter() if prof else 0.0
        if run_ranges is not None:
            # the shared ranges runner (lane py-shared, core/arena_io.mojo):
            # the inputs go up, the rest of the host arena starts zero on the
            # device, and everything but the inputs comes back
            ins = _arena_io.input_ranges(spans)
            outs = _arena_io.output_ranges(_arena_io.complement(ins, ha) + [list(s) for s in self._inout])  # glue: walks the program span records
            ia, oa = _arena_io.pack_ins(ins), _arena_io.pack_outs(outs)
            out, out_addr = _zero_words(on, self._out_code) if dev_out else (None, 0)
            try:
                run_ranges(base, prog.buffer_info()[0], out_addr,
                           (ha, sc if dev_scratch else 0, on if dev_out else 0, nst),
                           (ia.buffer_info()[0], len(ins), oa.buffer_info()[0], len(outs)))
            finally:
                for slot in gathered:  # glue: frees each fold-rows slot
                    binding.x_prep_dev_free(slot)
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
        for fn in self._after:  # glue: the program's post-run refusal hooks
            fn(self)
        if prof:
            # one line per program (the device's XPPHASE lines come in the same order)
            inv = {v: k for k, v in _OPS.items()}  # glue: debug print of stage names
            print("XPPROG ops=" + "+".join(inv.get(st[0], str(st[0])) for st in self._stages)  # glue: debug print of stage names
                  + f" arena={ha} scratch={sc} out={on} dev_out={int(bool(dev_out))} ranges={int(run_ranges is not None)}"
                  + f" alloc_s={t_alloc - t_run:.4f} inputs_s={t_in - t_alloc:.4f}"
                  + f" call_s={time.perf_counter() - t_in:.4f}", flush=True)
        return self

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
        self._check(off, n)
        return self.arena[off:off + n].tolist()


def _mode():
    return _backend.default_mode()


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


class _PrepBase:
    """sklearn's parameter protocol and fit_transform for the lane's classes."""
    _parameters = ()

    def get_params(self, deep=True):
        return {name: getattr(self, name) for name in self._parameters}  # glue: copies estimator keyword arguments

    def set_params(self, **params):
        for k, v in params.items():  # glue: copies estimator keyword arguments
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
        lo, hi = (float(v) for v in self.quantile_range)  # glue: unpacks the two quantile_range values
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
        qsel = _prep2_qselect(mode, 3)
        so = pr.work(n * d) if not qsel else 0
        st = pr.alloc(6 * d)
        qf = pr.put_list([lo / 100.0, 0.5, hi / 100.0])
        q = pr.alloc(3 * d)
        center = pr.alloc(d)
        scale = pr.alloc(d)
        if not qsel:
            pr.stage("sort_cols", d, xo, n, d, so, 0)
        # the quantile stage reads the count row only: an exact integer in
        # the blocked order too (lane gap-prep2), so the same words
        _col_stats(pr, xo, n, d, st, var=False)
        if qsel:
            # the three quantiles by radix select over the unsorted X (`_prep2_qselect`)
            pr.stage("quantile", 3 * d, xo, n, d, qf, 3, q, st, 1)
        else:
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
        # lane/apple-fast-prep3: `x_prep_maxabs_fit_direct` exists in the FAST
        # Apple build only (PREP3_MAXABS, default since M3 A/B
        # prep3-maxabs-istella-x-m3 121.7 -> 104.2 ms; -D
        # MOJOLEARN_PREP3_MAXABS_OFF reverts; x_prep/fastmaxabs.mojo):
        # X up once from its own buffer, the same max_abs_ and scale_ words
        direct = _optional_prep_entry(_prep_binding(mode), "x_prep_maxabs_fit_direct")
        if direct is not None:
            out = Array((2 * d,), "<f4")
            direct(addr_ro(arr, name="X"), out._addr, [n, d])
            self.max_abs_ = Array._from_flat(out.tolist()[:d], (d,), "<f4")
            self.scale_ = Array._from_flat(out.tolist()[d:], (d,), "<f4")
            self.numeric_mode_, self.n_features_in_, self.n_samples_seen_ = mode, d, n
            return self
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


def _given_categories(categories, arr, mode, check_unknown, who):
    """categories=<list>: one list per column, numeric, sorted ascending with
    at most a NaN last (the reference refuses unsorted numeric categories),
    stored as float32. With check_unknown (handle_unknown='error') a training
    value outside its column's list is refused, as the reference's fit."""
    n, d = arr.shape
    if len(categories) != d:
        raise ValueError(f"mojolearn: {who} categories has {len(categories)} lists; X has {d} features")
    out = []
    for j, cats in enumerate(categories):  # glue: validates the user categories argument (categories-sized: user category lists)
        vals = [float(v) for v in (cats.tolist() if hasattr(cats, "tolist") else cats)]  # glue: converts the user categories argument (cats-sized: one user category list)
        f32 = array.array("f", vals)
        nums = [v for v in f32 if v == v]  # glue: validates the user categories argument (f32-sized: one user category list)
        if len(nums) < len(f32) - 1 or (len(nums) < len(f32) and f32[-1] == f32[-1]):
            raise ValueError(f"mojolearn: {who} categories[{j}]: nan must be the last category")
        if nums != sorted(nums):  # glue: validates the user categories argument is sorted (nums-sized: one user category list)
            raise ValueError(f"mojolearn: {who} unsorted categories are not supported for numerical categories")
        if any(a == b for a, b in zip(nums, nums[1:])):  # glue: validates the user categories argument has no duplicates (nums-sized: one user category list)
            raise ValueError(f"mojolearn: {who} categories[{j}] has values equal in float32")
        if not f32:
            raise ValueError(f"mojolearn: {who} categories[{j}] is empty")
        canon = [0.0 if v == 0 else v for v in nums] + ([float("nan")] if len(nums) < len(f32) else [])  # glue: canonicalizes the user categories argument (nums-sized: one user category list)
        out.append(Array.from_list(canon, "<f4"))
    if check_unknown:
        pr = _Prog()
        _codes_neg = _codes(pr, arr, out)
        pr.run(mode)
        bad = [j for j, v in enumerate(pr.values(_codes_neg[1], d)) if v > 0]  # glue: reads the per-column unknown flags the device set (d-sized: feature count)
        if bad:
            raise ValueError(f"mojolearn: {who} found unknown categories in column(s) {bad} during fit")
    return out


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
    no = pr.put_list(ncat if ncat is not None else [c.size for c in categories])  # glue: packs per-column category sizes (categories-sized: per-column category arrays)
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


def _infrequent_threshold(n, min_frequency):
    """The integer count a frequent category reaches (THR of c2_inf_m0): an
    int min_frequency as is; a fraction f, ceil(n * f) in binary64 (count <
    n * f exactly when count < ceil(n * f)); none, 0. A scalar."""
    if min_frequency is None:
        return 0
    if isinstance(min_frequency, numbers.Integral):
        return int(min_frequency)
    return int(math.ceil(n * float(min_frequency)))


def _fit_infrequent(est, mode, arr, ignore_missing, inverse=None):
    """Sets est._infrequent (per column: sorted infrequent indices or None)
    and est._grouping (per column: category index -> grouped code, or None),
    as the reference's `_identify_infrequent` and
    `_fit_infrequent_category_mapping`. With ignore_missing (OrdinalEncoder)
    a trailing NaN category is left out of the grouping. Lane cpu2-l3-prep:
    one program, the counts (code_counts), the min_frequency mask
    (c2_inf_m0), the max_categories cut by a stable count rank (c2_inf_m1)
    and the grouped codes (c2_inf_map) on the device; the masks and codes
    come back as the fitted tables."""
    n, d = arr.shape
    cats_all = est.categories_
    ks = [c.size - 1 if (ignore_missing and c.size and _is_nan_value(c.tolist()[-1])) else c.size
          for c in cats_all]  # glue: one size per column
    pr = _Prog()
    if inverse is None:
        codes, _neg = _codes(pr, arr, cats_all)
    else:
        codes = pr.put(inverse)
    kmax = max(c.size for c in cats_all)  # glue: the widest column's category count (a shape)
    cnt, kc = pr.alloc(d * kmax), pr.put_list(ks)
    m0, m1, mp = pr.work(d * kmax), pr.alloc(d * kmax), pr.alloc(d * kmax)
    maxc = -1 if est.max_categories is None else int(est.max_categories)
    pr.stage("code_counts", d, codes, n, d, kmax, cnt)
    pr.stage("c2_inf_m0", d * kmax, cnt, kc, kmax, _infrequent_threshold(n, est.min_frequency), m0)
    pr.stage("c2_inf_m1", d * kmax, cnt, kc, kmax, m0, maxc, m1)
    pr.stage("c2_inf_map", d * kmax, kc, kmax, m1, mp)
    pr.run(mode)
    masks, maps = pr.get_i32(m1, d * kmax).tolist(), pr.get_i32(mp, d * kmax).tolist()
    est._infrequent, est._grouping = [], []
    for j, k in enumerate(ks):  # glue: the device's tables as the fitted per-column lists
        inf = [i for i in range(k) if masks[j * kmax + i]] or None  # glue: the device mask as the fitted index list
        est._infrequent.append(inf)
        est._grouping.append(None if inf is None else maps[j * kmax:j * kmax + k])
    est.infrequent_categories_ = [None if inf is None else Array.from_list([c.tolist()[i] for i in inf], "<f4")  # glue: slices fitted categories per column (inf-sized: infrequent category indices of one column)
                                  for c, inf in zip(est.categories_, est._infrequent)]  # glue: walks fitted per-column category lists (categories_-sized: fitted per-column category lists)


def _grouping_table(pr, grouping, inverse=False):
    """(MAP, MSTRIDE, NMAP) for remap_codes: category -> grouped code, or
    (inverse) grouped code -> category index with the infrequent code -> -3."""
    tables = []
    for g in grouping:  # glue: builds the category grouping table (grouping-sized: per-column category groupings)
        if g is None:
            tables.append([])
        elif not inverse:
            tables.append(list(g))
        else:
            nf = max(g)  # glue: largest group index of one column grouping (g-sized: one column grouping)
            back = [0] * (nf + 1)
            for i, v in enumerate(g):  # glue: inverts one column grouping table (g-sized: one column grouping)
                if v < nf:
                    back[v] = i
            back[nf] = -3
            tables.append(back)
    stride = max(1, max(len(t) for t in tables))  # glue: grouping table stride (tables-sized: per-column grouping tables)
    flat = []
    for t in tables:  # glue: pads the per-column grouping tables (tables-sized: per-column grouping tables)
        flat.extend(t + [0] * (stride - len(t)))
    return pr.put_list(flat), stride, pr.put_list([len(t) for t in tables])  # glue: packs the grouping table lengths (tables-sized: per-column grouping tables)


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


def _bad_rows_stages(pr, codes, n, d, bad, strict=_NONE):
    """Stages flagging each row with a code equal to `bad` (in a column whose
    `strict` word is nonzero, when given) and compacting the flagged rows
    ascending (p2m_rowflag, p2m_sel). Returns (ROWS, TOT): read them with
    `_bad_rows` after the run."""
    flags, rows = pr.alloc(n), pr.alloc(n)
    pr.stage("p2m_rowflag", n, codes, n, d, int(bad), strict, flags)
    return rows, _p2m_sel(pr, flags, n, 1, 1, rows)


def _bad_rows(pr, staged, limit=None):
    """The flagged rows `_bad_rows_stages` compacted (the first `limit`)."""
    rows, tot = staged
    k = int(pr.get_i32(tot, 1).tolist()[0])
    if k == 0:
        return []
    return pr.get_i32(rows, k if limit is None else min(k, limit)).tolist()


def _block_argmax(pr, arr, widths, drops, check):
    """One code per (row, block): the block_argmax stage over a (n, sum(widths)) input."""
    n, W = arr.shape
    d = len(widths)
    xo = pr.put(arr)
    so = pr.put_list([sum(widths[:j]) for j in range(d)])  # glue: prefix offsets of the per-column block widths (d-sized: feature count)
    wo = pr.put_list(widths)
    do = pr.put_list(drops) if drops is not None else _NONE
    codes = pr.alloc(n * d)
    pr.stage("block_argmax", n * d, xo, n, W, d, so, wo, do, 1 if check else 0, codes)
    return codes


def _raise_unknown(pr, neg, d, who):
    bad = [j for j, v in enumerate(pr.values(neg, d)) if v > 0]  # glue: reads the per-column unknown flags the device set (d-sized: feature count)
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
        if getattr(self, "_classical_capture_codes", False) and _is_auto(self.categories):
            self.categories_, self._classical_fit_codes = _fit_categories_with_codes(mode, arr)
        else:
            self.categories_ = (_fit_categories(mode, arr) if _is_auto(self.categories) else
                                _given_categories(self.categories, arr, mode, self.handle_unknown == "error",
                                                  "OrdinalEncoder"))
        self._missing = [c.size - 1 if c.size and _is_nan_value(c.tolist()[-1]) else -1 for c in self.categories_]  # glue: per-column NaN category slot (categories_-sized: fitted per-column category arrays)
        self._infrequent = self._grouping = None
        if grouping:
            _fit_infrequent(self, mode, arr, True, getattr(self, "_classical_fit_codes", None))
        cards = [c.size - (1 if m >= 0 else 0) for c, m in zip(self.categories_, self._missing)]  # glue: per-column category cardinalities (categories_-sized: fitted per-column category arrays)
        if grouping:
            cards = [k if g is None else max(g) + 1 for k, g in zip(cards, self._grouping)]  # glue: grouped per-column cardinalities (cards-sized: per-column category cardinalities)
        if self.handle_unknown == "use_encoded_value" and not _is_nan_value(self.unknown_value):
            if any(0 <= self.unknown_value < k for k in cards):  # glue: validates unknown_value against cardinalities (cards-sized: per-column category cardinalities)
                raise ValueError(f"mojolearn: the used value for unknown_value {self.unknown_value} is one of the "
                                 "values already used for encoding the seen categories.")
        if any(m >= 0 for m in self._missing) and not _is_nan_value(self.encoded_missing_value):  # glue: validates encoded_missing_value argument (_missing-sized: per-column NaN category slots)
            bad = [j for j, (k, m) in enumerate(zip(cards, self._missing))  # glue: validates encoded_missing_value argument (cards-sized: per-column category cardinalities)
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
        return self._transform_encoded(arr)

    def _transform_encoded(self, arr, inverse=None):
        n, d = arr.shape
        pr = _Prog()
        if inverse is None:
            codes, neg = _codes(pr, arr, self.categories_)
        else:
            codes, neg = pr.put(inverse), pr.alloc(d)
            pr.stage("count_neg", d, codes, n, d, neg)
        out = codes
        if self._grouping is not None:
            out = _remap(pr, codes, n, d, _grouping_table(pr, self._grouping), _NONE)
        if self.handle_unknown == "use_encoded_value":
            val = pr.put_scalar(self.unknown_value)
            src, out = out, pr.alloc(n * d)
            pr.stage("where_neg", n * d, src, n * d, val, out)
        if any(m >= 0 for m in self._missing):  # glue: checks for a NaN category slot (_missing-sized: per-column NaN category slots)
            src, out = out, pr.alloc(n * d)
            pr.stage("where_code", n * d, codes, n, d, pr.put_list(self._missing),
                     pr.put_scalar(self.encoded_missing_value), src, out)
        pr.run(self.numeric_mode_)
        if self.handle_unknown == "error":
            _raise_unknown(pr, neg, d, "OrdinalEncoder")
        return pr.get(out, (n, d))

    def fit_transform(self, X, y=None, **fit_params):
        # C08 explicitly owned fit-local inverse; never reused for a later
        # transform, mutated input, another fit or a TargetEncoder fold.
        # NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
        if not (_classical_shared(_mode()) & 1) or not _is_auto(self.categories):
            return super().fit_transform(X, y, **fit_params)
        self._classical_capture_codes = True
        try:
            self.fit(X, y, **fit_params)
            return self._transform_encoded(_finite_2d(X, "OrdinalEncoder"), self._classical_fit_codes)
        finally:
            self.__dict__.pop("_classical_capture_codes", None)
            self.__dict__.pop("_classical_fit_codes", None)

    def inverse_transform(self, X):
        self._check_fitted()
        arr = _x2d(X)
        self._check_width(arr)
        n, d = arr.shape
        pr = _Prog()
        unknown = self.unknown_value if self.handle_unknown == "use_encoded_value" else None
        ncat = back = None
        if self._grouping is not None:
            ncat = [c.size if g is None else max(g) + 1 for c, g in zip(self.categories_, self._grouping)]  # glue: per-column grouped category counts (categories_-sized: fitted per-column category arrays)
            back = _grouping_table(pr, self._grouping, inverse=True)
        out, codes = _inverse_codes(pr, arr, self.categories_, self._missing, self.encoded_missing_value, unknown,
                                    ncat, back)
        staged = _bad_rows_stages(pr, codes, n, d, -2)
        pr.run(self.numeric_mode_)
        bad = _bad_rows(pr, staged, 10)
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
        if getattr(self, "_classical_capture_codes", False) and _is_auto(self.categories):
            self.categories_, self._classical_fit_codes = _fit_categories_with_codes(mode, arr)
        else:
            self.categories_ = (_fit_categories(mode, arr) if _is_auto(self.categories) else
                                _given_categories(self.categories, arr, mode, self.handle_unknown == "error",
                                                  "OneHotEncoder"))
        self._infrequent = self._grouping = None
        if grouping:
            _fit_infrequent(self, mode, arr, False, getattr(self, "_classical_fit_codes", None))
        self._set_drop_idx()
        self.numeric_mode_, self.n_features_in_ = mode, arr.shape[1]
        return self

    def _grouped_sizes(self):
        return [c.size if (self._grouping is None or g is None) else max(g) + 1  # glue: per-column grouped category counts (c-sized: fitted per-column category arrays)
                for c, g in zip(self.categories_, self._grouping or [None] * len(self.categories_))]  # glue: per-column grouped category counts (categories_-sized: fitted per-column category arrays)

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
            after = [0 if k == 2 else None for k in sizes]  # glue: drop index for binary columns (sizes-sized: per-column category counts)
        else:
            vals = list(self.drop.tolist() if hasattr(self.drop, "tolist") else self.drop)
            if len(vals) != len(sizes):
                raise ValueError(f"mojolearn: `drop` should have length equal to the number of features "
                                 f"({len(sizes)}), got {len(vals)}")
            after, missing = [], []
            for j, (v, cats) in enumerate(zip(vals, self.categories_)):  # glue: validates the user drop argument (vals-sized: per-column drop arguments)
                cl = cats.tolist()
                if _is_nan_value(v):
                    hit = [cats.size - 1] if cl and _is_nan_value(cl[-1]) else []
                else:
                    fv = array.array("f", [float(v)])[0]
                    hit = [i for i, c in enumerate(cl) if c == fv]  # glue: finds the user drop category in its column (cl-sized: one column category list)
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
                                     f"Category: {v}, Feature: {j}" for j, v in missing))  # glue: formats the drop error message (missing-sized: unknown drop entries)
        self._drop_after = after
        if after is None:
            self.drop_idx_ = None
        else:
            self.drop_idx_ = [a if (a is None or g is None) else g.index(a) for a, g in zip(after, grouping)]  # glue: maps drop indices through the grouping (after-sized: per-column drop indices)

    def _widths(self):
        drops = self._drop_after or [None] * len(self.categories_)
        return [k - (0 if dr is None else 1) for k, dr in zip(self._grouped_sizes(), drops)], drops  # glue: per-column one-hot block widths (drops-sized: per-column drop indices)

    def _unknown_to(self):
        """Per column, the grouped code an unknown value takes: the infrequent
        one under 'infrequent_if_exist' / 'warn' when the column has it, else -1."""
        if self._grouping is None or self.handle_unknown not in ("infrequent_if_exist", "warn"):
            return None
        return [-1 if g is None else max(g) for g in self._grouping]  # glue: per-column infrequent slot (_grouping-sized: per-column category groupings)

    def transform(self, X):
        self._check_fitted()
        arr = _finite_2d(X, "OneHotEncoder")
        self._check_width(arr)
        return self._transform_encoded(arr)

    def _transform_encoded(self, arr, inverse=None):
        n, d = arr.shape
        widths, drops = self._widths()
        starts = [sum(widths[:j]) for j in range(d)]  # glue: prefix offsets of the per-column block widths (d-sized: feature count)
        W = sum(widths)  # glue: total one-hot output width (widths-sized: per-column block widths)
        pr = _Prog()
        if inverse is None:
            codes, neg = _codes(pr, arr, self.categories_)
        else:
            codes, neg = pr.put(inverse), pr.alloc(d)
            pr.stage("count_neg", d, codes, n, d, neg)
        if self._grouping is not None:
            unk = self._unknown_to()
            codes = _remap(pr, codes, n, d, _grouping_table(pr, self._grouping),
                           _NONE if unk is None else pr.put_list(unk))
        so = pr.put_list(starts)
        do = pr.put_list([-1 if dr is None else dr for dr in drops])  # glue: packs drop indices for the program (drops-sized: per-column drop indices)
        out = pr.output(n * W)
        pr.stage("onehot", n * d, codes, n, d, so, do, W, out)
        pr.run(self.numeric_mode_)
        if self.handle_unknown == "error":
            _raise_unknown(pr, neg, d, "OneHotEncoder")
        elif self.handle_unknown == "warn" or (self.drop is not None and
                                               self.handle_unknown in ("ignore", "infrequent_if_exist")):
            bad = [j for j, v in enumerate(pr.values(neg, d)) if v > 0]  # glue: reads the per-column unknown flags the device set (d-sized: feature count)
            if bad:
                import warnings
                where = ("encoded as the infrequent category" if self.handle_unknown != "ignore"
                         else "encoded as all zeros")
                warnings.warn(f"Found unknown categories in columns {bad} during transform. These unknown "
                              f"categories will be {where}.", UserWarning)
        return pr.get(out, (n, W))

    def fit_transform(self, X, y=None, **fit_params):
        # C08 keeps only invocation-owned inverse data, shared by category
        # counts and output emission. Future transforms always recode X.
        # NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
        if not (_classical_shared(_mode()) & 1) or not _is_auto(self.categories):
            return super().fit_transform(X, y, **fit_params)
        self._classical_capture_codes = True
        try:
            self.fit(X, y, **fit_params)
            return self._transform_encoded(_finite_2d(X, "OneHotEncoder"), self._classical_fit_codes)
        finally:
            self.__dict__.pop("_classical_capture_codes", None)
            self.__dict__.pop("_classical_fit_codes", None)

    def inverse_transform(self, X):
        self._check_fitted()
        arr = _x2d(X)
        widths, drops = self._widths()
        n, W, d = arr.shape[0], arr.shape[1], len(widths)
        if W != sum(widths):  # glue: validates the input width (widths-sized: per-column block widths)
            raise ValueError(f"mojolearn: X has {W} columns, expected {sum(widths)}")
        pr = _Prog()
        codes = _block_argmax(pr, arr, widths, [-1 if dr is None else dr for dr in drops], True)  # glue: packs drop indices for the program (drops-sized: per-column drop indices)
        grouped = codes
        if self._grouping is not None:
            codes = _remap(pr, codes, n, d, _grouping_table(pr, self._grouping, inverse=True), _NONE)
        out = _gather_categories(pr, codes, n, d, self.categories_)
        # an all-zero block is unknown (NaN) under 'ignore', and under
        # 'infrequent_if_exist' / 'warn' for a column with no infrequent
        # category; anywhere else it cannot be inverted (glue: d flags)
        strict = [1.0 if (self.handle_unknown == "error" or (self.handle_unknown != "ignore"
                                                               and self._infrequent is not None
                                                               and self._infrequent[j] is not None)) else 0.0
                  for j in range(d)]  # glue: slices the per-column decoded outputs (d-sized: feature count)
        staged = _bad_rows_stages(pr, grouped, n, d, -1, pr.put_list(strict))
        pr.run(self.numeric_mode_)
        bad = _bad_rows(pr, staged, 10)
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


def _numeric_target(y):
    """y as a flat numeric Array, or None for labels that are not numbers
    (strings, objects, bools)."""
    from ._buffer import _materialize
    try:
        arr, _ = _materialize(y, "y")
    except (TypeError, ValueError):
        return None
    if arr.dtype not in ("<f4", "<f8", "<i4", "<i8", "<u4", "<u1", "<i2", "<u2", "<i1"):
        return None
    return arr.reshape((arr.size,))


def _target_kind(y, target_type):
    """(kind, classes, targets, T): targets are a float32 Array (continuous)
    or the int32 class codes (binary, multiclass; the device makes their
    float and one-hot words). Lane cgr4-py-compute: the kind test (any
    non-integer value) is `_all_integral` (lane cpu2-l3-prep: on the
    device), not a per-label Python test, and no per-row target list is
    built."""
    num = _numeric_target(y)
    continuous = target_type == "continuous"
    if not continuous and target_type == "auto" and num is not None and num.size \
            and num.dtype in ("<f4", "<f8"):
        continuous = not _all_integral(num)
    if continuous:
        if num is None:
            raise ValueError("mojolearn: a continuous TargetEncoder target must be numeric")
        return "continuous", None, num.astype("<f4") if num.dtype != "<f4" else num, 1
    classes, codes = encode_labels(y if num is None else num)
    if not (isinstance(codes, Array) and codes.dtype == "<i4"):
        codes = as_i32_c(codes, ndim=1, name="codes")[0]
    if target_type == "binary" or (target_type == "auto" and len(classes) <= 2):
        return "binary", classes, codes, 1
    return "multiclass", classes, codes, len(classes)


def _all_integral(arr):
    """Whether every value of a float32 / float64 Array is finite and an
    integer (`reduce_stat`'s integral test), on the device: c2_integral by
    chunks, c2_isum, one word read back (lane cpu2-l3-prep)."""
    lb = _label_buffer(arr)
    if lb is None or lb.kind not in (0, 4):
        raise TypeError("mojolearn: the integral test takes a float32 or float64 vector")
    n = lb.n
    ch = max(4096, -(-n // 4096))
    nch = -(-n // ch)
    pr = _Prog()
    cnt, tot = pr.alloc(nch), pr.alloc(1)
    pr.stage("c2_integral", nch, pr.put_words(lb.words), lb.kind, n, ch, cnt)
    pr.stage("c2_isum", 1, cnt, nch, tot)
    pr.run(_mode())
    return int(pr.get_i32(tot, 1).tolist()[0]) == 0


def _target_arrays(y, target_type):
    """`_target_kind` without the per-label lists (lane apple-fast-py2mojo-prep):
    (kind, classes, target, T) with `target` the float32 Array of a
    continuous y, else the int32 class codes (binary: T = 1, the codes are
    the column; multiclass: T = K, the program expands them to the one-hot
    rows with a `label_binarize` stage). Only for a numeric VECTOR buffer
    with every value finite; None otherwise, and `_target_kind` answers."""
    from ._buffer import _has_buffer, _materialize, all_finite
    if isinstance(y, (list, tuple, str, bytes)) or not (isinstance(y, Array) or _has_buffer(y)):
        return None
    try:
        arr, _ = _materialize(y, "y")
    except (TypeError, ValueError):
        return None
    kind = arr.dtype.lstrip("<>|=")[:1]
    if arr.size == 0 or arr.size != max(arr.shape) or kind not in ("f", "i", "u"):  # glue: shape check of the target argument
        return None
    if kind == "f" and (arr.dtype not in ("<f4", "<f8") or not all_finite(arr)):
        return None
    # the reference's continuous test: some label is not an integer (`_all_integral`,
    # on the device; an integer dtype holds integers)
    cont = target_type == "continuous" or (
        target_type == "auto" and kind == "f" and not _all_integral(arr))
    if cont:
        vec = arr if arr.dtype == "<f4" and arr._has_order("C") else arr.astype("<f4")
        return "continuous", None, Array._view_of(vec, (vec.size,)), 1
    classes, codes = encode_labels(arr)
    if not (isinstance(codes, Array) and codes.dtype == "<i4"):
        return None
    if target_type == "binary" or (target_type == "auto" and len(classes) <= 2):
        return "binary", classes, codes, 1
    return "multiclass", classes, codes, len(classes)


_TARGET_SCRATCH = {}


def _target_scratch(mode):
    if mode != "fast":
        return False
    binding = _prep_binding(mode)
    key = id(binding)
    if key not in _TARGET_SCRATCH:
        fn = _optional_prep_entry(binding, "x_prep_target_scratch")
        _TARGET_SCRATCH[key] = bool(fn()) if fn is not None else False
    return _TARGET_SCRATCH[key]


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

    def _run(self, arr, y, folds, n_folds, apply_rows_folds, binary=None, target=None):
        """binary (lane prep-apple3, `te_arrays`): (classes, int32 codes) of an
        explicit binary target, with `folds` an int32 Array: both cross as
        int32 words and become floats on the device (i2f), the same words the
        lists gave. target (lane apple-fast-py2mojo-prep): `_target_arrays`'
        answer, the target's words as arrays (the one-hot rows built on the
        device), the same words `_target_kind`'s lists gave."""
        n, d = arr.shape
        mode = _mode()
        if binary is None and target is None:
            target = _target_arrays(y, self.target_type)
        if target is not None:
            kind, classes, tgt, T = target
        elif binary is None:
            kind, classes, tgt, T = _target_kind(y, self.target_type)
        else:
            kind, classes, tgt, T = "binary", binary[0], binary[1], 1
        rows = tgt.size * T
        if rows != n * T:
            raise ValueError("mojolearn: X and y have different numbers of rows")
        inverse = None
        if (_classical_shared(mode) & 1) and _is_auto(self.categories):
            # C08 reuses only unsupervised category IDs. Fold membership,
            # target statistics, and held-out target isolation are unchanged.
            # NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
            cats, inverse = _fit_categories_with_codes(mode, arr)
        else:
            cats = (_fit_categories(mode, arr) if _is_auto(self.categories) else
                    _given_categories(self.categories, arr, mode, False, "TargetEncoder"))
        cmax = max(c.size for c in cats)  # glue: largest category count for the program layout (cats-sized: per-column category arrays)
        F = n_folds
        pr = _Prog()
        scratch = _target_scratch(mode)
        if inverse is None:
            codes, _neg = _codes(pr, arr, cats, device_only=scratch)
        else:
            codes = pr.put(inverse)
        # the target's words as arrays (lane apple-fast-py2mojo-prep /
        # cgr4-py-compute): continuous float32, codes through i2f, the
        # one-hot rows built on the device
        if kind == "continuous":
            yo = pr.put(tgt)
        elif T == 1:
            yo = pr.put_codes(tgt)
        else:
            # the one-hot rows: word 1.0f (bits 0x3F800000) where row i's code is c, else 0.0f
            co, yo = pr.put_codes(tgt), pr.alloc(n * T)
            pr.stage("label_binarize", n * T, co, n, T, 0, 0, 0x3F800000, T, yo)
        if isinstance(folds, tuple):
            # cv an int: the folds on the device (`_stage_folds`): KFold for a continuous
            # target, else StratifiedKFold over the class codes
            _, seed, shuffle = folds
            codes_off = _NONE if kind == "continuous" else (yo if T == 1 else co)
            fo, flag = _stage_folds(pr, n, F, seed, shuffle, codes_off, len(classes) if classes is not None else 0)
            _refuse_folds(pr, flag, F)
        else:
            if folds is None:
                folds = full((max(n, 1),), -1, "<i4")
            elif not isinstance(folds, Array):
                store = array.array("i", folds)
                folds = Array._owned(store, (len(store),), "<i4", "C")
            fo = pr.put_codes(folds)
        nco = pr.put_list([c.size for c in cats])  # glue: packs per-column category sizes (cats-sized: per-column category arrays)
        meta = pr.alloc(2 * (F + 1) * T)
        smo = pr.put_scalar(-1.0 if self.smooth == "auto" else float(self.smooth))
        enc = pr.alloc((F + 1) * d * cmax * T)
        pr.stage("te_global", (F + 1) * T, yo, n, T, fo, meta)
        if _optional_prep_entry(_prep_binding(mode), "x_prep_host_column") is not None:
            # the host binding groups te_enc its own way (x_prep/host/target.mojo)
            pr.stage("te_enc", (F + 1) * d * cmax * T, codes, n, d, yo, T, fo, cmax, nco, meta, smo, enc)
        else:
            # each category's rows, ascending (te_bucket): te_enc walks one bucket, not every row
            if scratch:
                bstart, brows = pr.scratch(d * (cmax + 1)), pr.scratch(n * d)
                # te_gather also visits unused bucket tails (unknown categories).
                # Preserve alloc's initialized zero row indices in those tails.
                pr.stage("cat_zero", n * d, brows)
            else:
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
        base = F * d * cmax * T
        self._enc = pr.get(enc + base, d * cmax * T)
        self._meta = pr.get(meta + 2 * F * T, 2 * T)
        self.encodings_ = [pr.get(enc + base + (j * cmax) * T, cats[j].size * T) for j in range(d)]  # glue: slices the per-column encodings outputs (d-sized: feature count)
        means = pr.values(meta + 2 * F * T, 2 * T)[0::2]
        self.target_mean_ = pr.get(meta + 2 * F * T, 1) if T == 1 else Array.from_list(means, "<f4")
        return pr.get(out, (n, d * T)) if apply_rows_folds else None

    def fit(self, X, y):
        self._check()
        self._run(_x2d(X), y, None, 0, False)
        return self

    def _splitter_folds(self, X, y, n):
        """Row -> fold (an int32 Array) from a splitter object or (train,
        test) iterable, checked by the base binding's helpers (lane
        cgr4-py-compute: no per-row Python): the test folds cover every row
        exactly once, and each fold's training rows are every other row."""
        from ._buffer import as_index_i64
        splits = list(self.cv.split(X, y) if hasattr(self.cv, "split") else self.cv)
        cover = ("mojolearn: Validation indices from `cv` must cover each sample index exactly once "
                 "with no overlap. Pass a splitter with non-overlapping validation folds as `cv`.")
        if len(splits) < 1:
            raise ValueError(cover)
        fold = full((n,), -1, "<i4")
        assign = _native_helper("assign_fold_i64")  # cpu-route: input prep of a user cv splitter's index lists, before any fit
        trains = []
        for k, (train, test) in enumerate(splits):  # cpu-route: input prep of a user cv splitter index lists
            te = as_index_i64(test, name="test")
            if int(assign(addr_ro(te, name="test") if te.size else 0, te.size, n, k,
                          _addr_rw(fold, name="folds"))) != 0:
                raise ValueError(cover)
            trains.append(as_index_i64(train, name="train"))
        if fold.min() < 0:
            raise ValueError(cover)
        sizes = _class_counts(fold, len(splits))
        check = _native_helper("check_indices_i64")
        hits = _native_helper("count_fold_hits_i64")  # cpu-route: input prep of a user cv splitter's index lists, before any fit
        for k, tr in enumerate(trains):  # cpu-route: input prep of a user cv splitter index lists
            # every row outside fold k exactly once: as many as there are,
            # distinct, in range and none in fold k
            if (tr.size != n - sizes[k]
                    or tr.size and (int(check(addr_ro(tr, name="train"), tr.size, n)) != 0
                                    or int(hits(addr_ro(tr, name="train"), tr.size, addr_ro(fold, name="folds"),
                                                n, k)) != 0)):
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
        # the folds are staged in the fit's own program (`_stage_folds`, lane cpu2-l3-prep)
        folds = ("cv", seed, bool(self.shuffle))
        binary = _target_binary_codes(y, self.target_type)
        if binary is not None:
            if binary[1].size != n:
                raise ValueError("mojolearn: X and y have different numbers of rows")
            return self._run(arr, y, folds, cv, True, binary=binary)
        # the target as arrays (`_target_arrays`, lane apple-fast-py2mojo-prep;
        # else `_target_kind`'s arrays, lane cgr4-py-compute): no per-row Python either way
        target = _target_arrays(y, self.target_type)
        if target is None:
            target = _target_kind(y, self.target_type)
        if target[2].size != n:
            raise ValueError("mojolearn: X and y have different numbers of rows")
        return self._run(arr, y, folds, cv, True, target=target)

    def transform(self, X):
        self._check_fitted()
        arr = _x2d(X)
        self._check_width(arr)
        n, d = arr.shape
        T, cmax = self._T, self._cmax
        pr = _Prog()
        codes, _neg = _codes(pr, arr, self.categories_, device_only=_target_scratch(self.numeric_mode_))
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
        # only the median and the mode read the sorted columns (lane prep-apple3, `imputer_nosort`);
        # the median by radix select needs no sort (`_prep2_qselect`)
        qsel = self.strategy == "median" and _prep2_qselect(mode, 1)
        sorts = (self.strategy in ("median", "most_frequent") or not _r3("imputer_nosort")) and not qsel
        so = (pr.work(n * d) if self.strategy in ("median", "most_frequent") else pr.alloc(n * d)) if sorts else 0
        st = pr.alloc(6 * d)
        med = pr.alloc(d)
        mf = pr.alloc(d)
        half = pr.put_list([0.5])
        if sorts:
            pr.stage("sort_cols", d, xo, n, d, so, 0)
        _cs(pr, mode, xo, n, d, st, var=False)
        if self.strategy == "median" and qsel:
            pr.stage("quantile", d, xo, n, d, half, 1, med, st, 1)
        elif self.strategy == "median":
            pr.stage("quantile", d, so, n, d, half, 1, med, st)
        if self.strategy == "most_frequent":
            pr.stage("mode_cols", d, so, n, d, mf, _NONE)
        comp = None
        if callable(self.strategy):
            # each column's non-NaN words, ascending rows, at
            # comp + j*n (p2m_sel_*), in place of the Python transpose and filter
            comp = pr.alloc(n * d)
            ctot = _p2m_sel(pr, xo, n, d, 0, comp)
        # lane cpu2-l3-prep: statistics_ and the fill on the device (c2_imp_stats: an
        # all-missing column's statistic is NaN, or with keep_empty_features 0 / the
        # constant; its fill 0 / the constant), read back as the fitted vectors
        konst = self.strategy == "constant"
        src = so2 = fo2 = None
        if not callable(self.strategy):
            fv = pr.put_scalar(0.0 if self.fill_value is None else float(self.fill_value)) if konst else _NONE
            src = st + d if konst else {"mean": st + d, "median": med, "most_frequent": mf}[self.strategy]
            so2, fo2 = pr.alloc(d), pr.alloc(d)
            pr.stage("c2_imp_stats", d, st, src, d, 1 if self.keep_empty_features else 0, 1 if konst else 0, fv,
                     so2, fo2)
        pr.run(mode)
        counts = [int(v) for v in pr.values(st, d)]  # glue: reads per-column non-missing counts (d-sized: feature count)
        empty = [c == 0 for c in counts]  # glue: flags empty columns (counts-sized: per-column non-missing counts)
        if callable(self.strategy) and comp is not None:
            ncol = pr.get_i32(ctot, d).tolist()
            return self._fit_callable(None, counts, mode, [pr.get(comp + j * n, ncol[j]) for j in range(d)], n, d)  # glue: slices the per-column compacted outputs (d-sized: feature count)
        if konst or any(empty):
            self.statistics_ = pr.get(so2, d)
            self._fill = pr.get(fo2, d)
        else:
            self.statistics_ = pr.get(src, d)
            self._fill = self.statistics_
        self._keep = [j for j in range(d) if self.keep_empty_features or not empty[j]]  # glue: kept column index list (d-sized: feature count)
        self._indicator = [j for j in range(d) if counts[j] < n] if self.add_indicator else []  # glue: indicator column index list (d-sized: feature count)
        self.numeric_mode_, self.n_features_in_ = mode, d
        return self

    def _fit_callable(self, marked, counts, mode, kept, n=0, d=0):
        """strategy=<callable>: the reference's `strategy(masked_X[:, j].compressed())`
        per column over the missing-marked X (NaN = missing). kept (lane
        apple-fast-py2mojo-prep): the columns' non-NaN words already compacted
        by the program (`p2m_sel_*`, on the device), one float32 Array per
        column. Lane cpu3-python: the host compaction arm (`compact_notnan_f32`)
        was unreachable (the program always compacts for a callable) and is gone."""
        stats = []
        for j in range(d):  # cpu-route: calls the user strategy callable per column
            stats.append(float(self.strategy(kept[j])))
        self.statistics_ = Array.from_list(stats, "<f4")
        self._fill = self.statistics_
        self._keep = [j for j in range(d) if self.keep_empty_features or stats[j] == stats[j]]  # glue: kept column index list (d-sized: feature count)
        self._indicator = [j for j in range(d) if counts[j] < n] if self.add_indicator else []  # glue: indicator column index list (d-sized: feature count)
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
    for (start, end), part in zip(ranges, parts):  # glue: stages one copy per column block (parts-sized: column block parts)
        w = end - start
        if w <= 0 or n <= 0:
            continue
        pr.stage("colblock", n * w, pr.put(part), w, out, d, start)
    pr.run(_mode() if mode is None else mode)
    return pr.get(out, (n, d))


# ---------------------------------------------------------------- discretizer
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
        sub = None
        if self.subsample is not None and n > self.subsample and w is None:
            # the draws (p2m_smrows, splitmix64's closed form, one thread a draw) and the
            # gather (p2m_rgather) run in the fit's program below
            sub = ((0 if self.random_state is None else int(self.random_state)) & 0xFFFFFFFFFFFFFFFF,
                   int(self.subsample))
        elif self.subsample is not None and n > self.subsample:
            # the weighted resample with replacement is drawn in the fit's program below
            # (f2_wblk / f2_wscan / f2_wdraw, x_prep/fam2.mojo: blocked float32 weight sums,
            # one thread per draw), then gathered there; every tier (lane cpu2-l3-prep
            # deleted the host `weighted_draw_rows_i32` route)
            seed = (0 if self.random_state is None else int(self.random_state)) & 0xFFFFFFFFFFFFFFFF
            sub = ("w", seed, int(self.subsample), w)
            w = None
        nb = [int(self.n_bins)] * d if isinstance(self.n_bins, numbers.Integral) else [int(b) for b in self.n_bins]  # glue: converts the n_bins argument (nb-sized: per-column n_bins argument)
        if len(nb) != d or min(nb) < 2:  # glue: validates the n_bins argument (nb-sized: per-column n_bins argument)
            raise ValueError("mojolearn: n_bins must be >= 2 per feature")
        if w is not None and strat not in (0, 1, 3, 4):
            raise ValueError("mojolearn: When fitting with strategy='quantile' and sample weights, quantile_method "
                             "should either be set to 'averaged_inverted_cdf' or 'inverted_cdf', got "
                             f"quantile_method='{self.quantile_method}' instead.")
        nbmax = max(nb)  # glue: largest bin count for the program layout (nb-sized: per-column n_bins argument)
        mode = _mode()
        pr = _Prog()
        xo = pr.put(arr)
        if sub is not None and sub[0] == "w":
            _, seed, m, wsub = sub
            nbk = (n + _XB - 1) // _XB
            wo, bs, bp = pr.put(wsub), pr.work(nbk), pr.work(nbk)
            ro, xs = pr.alloc(m), pr.alloc(m * d)
            pr.stage("f2_wblk", nbk, wo, n, bs)
            pr.stage("f2_wscan", 1, bs, nbk, bp)
            pr.stage("f2_wdraw", m, wo, n, bs, bp, nbk, _seed_words(pr, seed), ro)
            pr.stage("p2m_rgather", m * d, xo, d, ro, xs)
            xo, n = xs, m
        elif sub is not None:
            seed, m = sub
            lo, hi = seed & 0xFFFFFFFF, seed >> 32
            so_seed = pr.put_ints([lo - (1 << 32) if lo >= 1 << 31 else lo, hi - (1 << 32) if hi >= 1 << 31 else hi])
            ro, xs = pr.alloc(m), pr.alloc(m * d)
            pr.stage("p2m_smrows", m, so_seed, n, ro)
            pr.stage("p2m_rgather", m * d, xo, d, ro, xs)
            xo, n = xs, m
        so = pr.work(n * d) if w is None else pr.alloc(n * d)
        st = pr.alloc(6 * d)
        nbo = pr.put_list(nb)
        edges = pr.alloc(d * (nbmax + 1))
        ne = pr.alloc(d)
        lab = pr.alloc(n * d) if strat == 3 else 0
        cen = pr.alloc(d * nbmax) if strat == 3 else 0
        _cs(pr, mode, xo, n, d, st)
        if w is None:
            pr.stage("sort_cols", d, xo, n, d, so, 0)
            pr.stage("kbins_edges", d, so, n, d, nbo, nbmax, strat, st, edges, ne, lab, cen)
        else:
            stw = st
            if strat in (0, 3):
                # the min / max over the rows of nonzero weight (the reference's nnz mask)
                stw = pr.alloc(6 * d)
                # the rows selected and gathered on the device (the weights are
                # validated nonnegative: v > 0 is v != 0)
                m = wl
                if m < n:
                    _cs(pr, mode, _p2m_positive_rows(pr, xo, pr.put(w), n, d, m), m, d, stw)
                else:
                    stw = st
            if strat == 0:
                pr.stage("kbins_edges", d, so, n, d, nbo, nbmax, 0, stw, edges, ne, 0, 0)
            else:
                ug, ucnt = _weighted_groups(pr, arr, w, n, d)
                if strat == 3:
                    pr.stage("kbins_wkm", d, ug, n, d, ucnt, nbo, nbmax, st, stw, edges, cen, lab)
                else:
                    # the percent levels i * (100 / b) on the device (c2_grid KIND 0)
                    _weighted_levels(pr, ug, ucnt, n, d, nb, 0, strat == 1, edges)
                pr.stage("kbins_edges", d, so, n, d, nbo, nbmax, 11, stw, edges, ne, 0, 0)
        pr.run(mode)
        counts = [int(v) for v in pr.values(ne, d)]  # glue: reads per-column edge counts (d-sized: feature count)
        self.bin_edges_ = [pr.get(edges + j * (nbmax + 1), counts[j]) for j in range(d)]  # glue: slices the per-column bin edges (d-sized: feature count)
        self.n_bins_ = Array.from_list([c - 1 for c in counts], "<i8")  # glue: per-column bin counts from edge counts (counts-sized: per-column edge counts)
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
        widths = [int(v) for v in self.n_bins_.tolist()]  # glue: reads fitted per-column bin counts (widths-sized: per-column bin counts)
        W = sum(widths)  # glue: total one-hot output width (widths-sized: per-column bin counts)
        so = pr.put_list([sum(widths[:j]) for j in range(d)])  # glue: prefix offsets of the per-column bin widths (d-sized: feature count)
        out = pr.output(n * W)
        pr.stage("onehot", n * d, codes, n, d, so, _NONE, W, out)
        pr.run(self.numeric_mode_)
        return pr.get(out, (n, W))

    def inverse_transform(self, X):
        self._check_fitted()
        arr = _x2d(X)
        widths = [int(v) for v in self.n_bins_.tolist()]  # glue: reads fitted per-column bin counts (widths-sized: per-column bin counts)
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
            if arr.shape[1] != sum(widths):  # glue: validates the input width (widths-sized: per-column bin counts)
                raise ValueError(f"mojolearn: X has {arr.shape[1]} columns, expected {sum(widths)}")
            codes = _block_argmax(pr, arr, widths, None, True)
            bad_code, why = -1, "can not be inverted because they contain all zeros"
        out = pr.output(n * d)
        pr.stage("kbins_inverse", n * d, codes, n, d, pr.put(self._edges), self._stride, out)
        staged = _bad_rows_stages(pr, codes, n, d, bad_code)
        pr.run(self.numeric_mode_)
        bad = _bad_rows(pr, staged, 10)
        if bad:
            raise ValueError(f"mojolearn: samples {bad[:10]} {why}")
        return pr.get(out, (n, d))


# ---------------------------------------------------------------- naive Bayes
#: mode -> whether its binding exports `x_prep_proba64` (lane apple-fast-q-clf)
_PROBA64 = {}


def _proba64_on(mode):
    """predict_proba as float64 (x_prep/proba64.mojo `q64_softmax_unit`): the
    FAST binding's default (QUALITY-FIX: a float32 probability saturates at
    exactly 1.0 past a ~16.6-nat gap; Istella log loss gaussian-nb 3.574 vs
    scikit-learn 3.417, bernoulli-nb 5.351 vs 4.279); a build with
    -D MOJOLEARN_PROBA64_QOLD (or IDENTICAL) has no export and keeps the
    float32 `row_softmax` probabilities."""
    key = str(mode)
    if key not in _PROBA64:
        try:
            _PROBA64[key] = _optional_prep_entry(_prep_binding(mode), "x_prep_proba64") is not None
        except Exception:  # noqa: BLE001  (no binding: the float32 route)
            _PROBA64[key] = False
    return _PROBA64[key]


class _Classifier(_PrepBase):
    """predict / predict_proba / predict_log_proba from a subclass's joint
    log likelihood stages (`_jll_stages`), normalised on the device."""
    #: lane cpu4-misc: fit and every scoring method take a
    #: `_arena_io.DeviceRows` (a cross-validation fold's X on the device):
    #: X reaches the arena only through `_x2d` -> `_Prog.put`, so the fold
    #: rows are gathered on the device and never come to the host. Every
    #: subclass (GaussianNB, MultinomialNB, BernoulliNB, ComplementNB,
    #: CategoricalNB, LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis)
    #: keeps to that; a user covariance_estimator takes host rows.
    _mojolearn_device_rows = True

    def _encode_y(self, y, n):
        classes, codes = encode_labels(y)
        if codes.size != n:
            raise ValueError("mojolearn: X and y have different numbers of rows")
        self.classes_ = classes
        return codes

    def _scores(self, X, want, csr=None):
        self._check_fitted()
        pr = _Prog()
        # lane idn-all: `csr` (the IDENTICAL CSR scoring's fallback, its
        # width already checked) is densified by the program on the device
        xo = _csr_dense_x(pr, csr, self.numeric_mode_) if csr is not None else None
        if xo is None:
            arr = _x2d(X.toarray() if csr is not None else X)
            self._check_width(arr)
            n, d = arr.shape
            xo = pr.put(arr)
        else:
            n, d = csr[3], csr[4]
        K = len(self.classes_)
        chk = self._score_checks(pr, xo, n, d)
        # the joint log likelihood stays on the device unless it is the answer
        jll = pr.work(n * K) if want else pr.alloc(n * K)
        self._jll_stages(pr, xo, n, d, jll)
        return self._score_tail(pr, n, d, K, jll, want, chk)

    def _score_tail(self, pr, n, d, K, jll, want, chk):
        """The scoring program after the joint log likelihood: its softmax
        and argmax stages, the run, the input refusals, the offsets."""
        lp = pr.alloc(n * K) if "log" in want else _NONE
        p64 = "proba" in want and _proba64_on(self.numeric_mode_)
        pp = pr.alloc((2 if p64 else 1) * n * K) if "proba" in want else _NONE
        am = pr.alloc(n) if "predict" in want else _NONE
        if p64:
            pr.stage("q64_softmax", n, jll, n, K, pp)
            if lp != _NONE:
                pr.stage("row_softmax", n, jll, n, K, lp, _NONE)
        elif lp != _NONE or pp != _NONE:
            pr.stage("row_softmax", n, jll, n, K, lp, pp)
        if am != _NONE:
            pr.stage("row_argmax", n, jll, n, K, am)
        pr.run(self.numeric_mode_)
        self._score_refusals(pr, d, chk)
        return pr, n, K, dict(jll=jll, log=lp, proba=pp, predict=am, proba64=p64)

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
        if o.get("proba64"):
            return pr.get_f64(o["proba"], (n, K))
        return pr.get(o["proba"], (n, K))

    def predict_log_proba(self, X):
        pr, n, K, o = self._scores(X, ("log",))
        return pr.get(o["log"], (n, K))

    def predict_joint_log_proba(self, X):
        pr, n, K, o = self._scores(X, ())
        return pr.get(o["jll"], (n, K))

    def score(self, X, y):
        from ._expansion_metrics import accuracy_fraction
        return accuracy_fraction(y, self.predict(X))


def _nb_weights(pr, sample_weight, n):
    """sample_weight -> its arena offset (None when not given)."""
    if sample_weight is None:
        return None
    w, _ = as_f32_c(sample_weight, ndim=1, name="sample_weight")
    if w.shape[0] != n:
        raise ValueError(f"mojolearn: sample_weight has {w.shape[0]} entries, expected {n}")
    from ._buffer import all_finite
    if n and not all_finite(w):
        raise ValueError("mojolearn: sample_weight must be finite")
    return pr.put(w)


#: binding -> its `x_prep_idn_int` bits (0 when it has none), probed once
_IDN_INT = {}
_IDN_NB_ONEPASS, _IDN_NB_CSR, _IDN_NB_CSR_DENSE = 2, 4, 8


def _idn_int(mode):
    """Lane idn-int-prep: the IDENTICAL binding's integer prep switches
    (bindings/_mojolearn_x_prep.mojo `idn_int_binding`: 1 IDN_LABEL, 2
    IDN_NB_ONEPASS, 4 IDN_NB_CSR, 8 IDN_NB_CSR_DENSE), 0 on another tier or a
    build with none."""
    if mode != "identical":
        return 0
    binding = _prep_binding(mode)
    key = id(binding)
    v = _IDN_INT.get(key)
    if v is None:
        fn = _optional_prep_entry(binding, "x_prep_idn_int")
        v = int(fn()) if fn is not None else 0
        _IDN_INT[key] = v
    return v


def _nb_onepass(mode):
    return _blocked() and bool(_idn_int(mode) & _IDN_NB_ONEPASS)


def _nb_counts(pr, wo, xo, n, d, yo, K, cnt, sums, thr=_NONE, neg=_NONE):
    """The discrete naive Bayes count pass as one unit per (block, column)
    (x_prep/blocked.mojo `csb1_part`, lane idn-int-prep): csb_part's
    partials of every class from one walk of X, then csb_fold. thr: the
    binarize threshold's offset (BernoulliNB; X is binarized in the unit).
    neg: d words that are -1 for a column holding a negative word, else 0
    (where the fit read the column minimum)."""
    nb = (n + _XB - 1) // _XB
    w = _NONE if wo is None else wo
    ps, pc, cn = pr.work(nb * K * d), pr.work(nb * K * d), pr.work(K * d)
    ng = pr.work(nb * d) if neg != _NONE else _NONE
    pr.stage("csb1_part", nb * d, xo, n, d, yo, K, ps, pc, nb, w, thr, ng)
    pr.stage("csb_fold", K * d, ps, pc, nb, K, d, cn, cnt, _NONE, sums, w)
    if neg != _NONE:
        pr.stage("csb1_neg", d, ng, nb, d, neg)


#: binding -> its `x_prep_idn_fam2` bits (0 when it has none), probed once
_IDN_FAM2 = {}
_F2_WDRAW, _F2_PERM_DRAW, _F2_WPICK, _F2_PARTIAL_CODES = 1, 2, 4, 8
_F2_GRAM, _F2_GRAM_ROWTILE, _F2_LABEL_INV = 16, 32, 64
#: the most partial words (blocks x cells) a blocked Gram keeps; past it the
#: block grows, then the one-thread-per-cell stage (a function of the shape
#: only, so the device and the host column agree)
_GRAM_WORDS = 2 ** 26


def _gram_rows(v, n, cells):
    """Rows per block of a blocked Gram over n rows with `cells` partial
    words a block (x_prep/gram_blocked.mojo): the binding's block length
    (`x_prep_idn_fam2` >> 16; 2048 unless a candidate arm is built), doubled
    until the partial table fits `_GRAM_WORDS`; 0 when it never does."""
    rows = (v >> 16) or _XB
    while ((n + rows - 1) // rows) * cells > _GRAM_WORDS and rows < 2 ** 22:  # glue: the block length doubles
        rows *= 2
    return rows if ((n + rows - 1) // rows) * cells <= _GRAM_WORDS else 0


def _gram(pr, mode, z, n, d, g):
    """Stages G = Z'Z of the (n, d) block at z into the (d, d) block g. Under
    IDENTICAL (`_idn_fam2` bit 16): block partials of the upper triangle, one
    thread per (block, a, b >= a) (bit 32, a candidate arm: one per
    (block, a)), then each cell folded over the blocks. Else `matmul`: one
    thread per cell over every row."""
    v = _idn_fam2(mode)
    rows = _gram_rows(v, n, d * d) if (_blocked() and v & _F2_GRAM) else 0
    if not rows:
        pr.stage("matmul", d * d, z, 1, d, z, d, 1, g, d, n, _NONE, _NONE)
        return
    nb = (n + rows - 1) // rows
    part = pr.work(nb * d * d)
    if v & _F2_GRAM_ROWTILE:
        pr.stage("gb_part_row", nb * d, z, n, d, part, nb, rows)
    else:
        pr.stage("gb_part", nb * d * d, z, n, d, part, nb, rows)
    pr.stage("gb_fold", d * d, part, nb, d, g)


def _qda_cov(pr, mode, xo, n, d, yo, K, mean, cnt, cov):
    """Stages the K class covariances (divisor the class count) into cov.
    Under IDENTICAL (`_idn_fam2` bit 16): block partials of every class from
    one walk per (block, a, b >= a), then each cell folded over the blocks
    (x_prep/gram_blocked.mojo). Else `qda_cov`: one thread per (class, a, b)
    over every row."""
    v = _idn_fam2(mode)
    rows = _gram_rows(v, n, K * d * d) if (_blocked() and v & _F2_GRAM) else 0
    if not rows:
        pr.stage("qda_cov", K * d * d, xo, n, d, yo, mean, cnt, cov)
        return
    nb = (n + rows - 1) // rows
    part = pr.work(nb * K * d * d)
    pr.stage("qcb_part", nb * d * d, xo, n, d, yo, K, mean, part, nb, rows)
    pr.stage("qcb_fold", K * d * d, part, nb, K, d, cnt, cov)


def _idn_fam2(mode):
    """Lane fam2-prep-metrics: the IDENTICAL binding's x_prep/fam2.mojo
    switches (bindings/_mojolearn_x_prep*.mojo `idn_fam2_binding`: 1
    IDN_WDRAW, 2 IDN_PERM_DRAW, 4 IDN_WPICK, 8 IDN_PARTIAL_CODES), 0 on
    another tier or a build with none. The device and the host column export
    the same bits, so both stage the same program."""
    if mode != "identical":
        return 0
    binding = _prep_binding(mode)
    key = id(binding)
    v = _IDN_FAM2.get(key)
    if v is None:
        fn = _optional_prep_entry(binding, "x_prep_idn_fam2")
        v = int(fn()) if fn is not None else 0
        _IDN_FAM2[key] = v
    return v


def _seed_words(pr, seed):
    """A uint64 seed as two int32 words (low, high) -> offset."""
    lo, hi = seed & 0xFFFFFFFF, (seed >> 32) & 0xFFFFFFFF
    return pr.put_ints([lo - (1 << 32) if lo >= 1 << 31 else lo, hi - (1 << 32) if hi >= 1 << 31 else hi])


#: binding -> its `x_prep_idn_fam` bits (0 when it has none), probed once
_IDN_FAM = {}
_IDN_STATS_BLOCKED, _IDN_CLASS_ONEPASS, _IDN_SELECT_BLOCKED, _IDN_PT_BLOCKED = 1, 2, 4, 8
#: `_cls`: the most partial words (blocks x classes x columns) a blocked
#: class_stats stage keeps; past it the serial stage (a function of the shape
#: only, so the device and the host column agree)
_CLS_BLOCK_WORDS = 2 ** 26


def _idn_fam(mode):
    """Lane fam-prep-metrics: the IDENTICAL binding's family switches
    (bindings/_mojolearn_x_prep*.mojo `idn_fam_binding`: 1 IDN_STATS_BLOCKED,
    2 IDN_CLASS_ONEPASS, 4 IDN_SELECT_BLOCKED, 8 IDN_PT_BLOCKED, 16
    IDN_RR_EIGH: informational), 0 on another tier or a build
    with none. The device and the host column export the same bits, so both
    stage the same program."""
    if mode != "identical":
        return 0
    binding = _prep_binding(mode)
    key = id(binding)
    v = _IDN_FAM.get(key)
    if v is None:
        fn = _optional_prep_entry(binding, "x_prep_idn_fam")
        v = int(fn()) if fn is not None else 0
        _IDN_FAM[key] = v
    return v


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


def _class_stats(pr, wo, total, xo, n, d, yo, K, cnt, mean, var, sums, mode=None):
    """class_stats, or its weighted form when a sample_weight offset is given;
    in x_prep/blocked.mojo's blocked order when `_blocked()` (offsets
    _NONE are not written). mode given and IDN_CLASS_ONEPASS (IDENTICAL):
    one unit per (block, column) walks X once for every class (csb1_part /
    csb1_ss; the same words as csb_part / csb_ss)."""
    # C55 explicit build control outranks incumbent blocked scheduling.
    # A's class-group serial profile is shared by host and all devices;
    # switching from a blocked incumbent is a versioned arithmetic change.
    # NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
    if _classical_shared(mode if mode is not None else _mode()) & 2:
        if wo is None:
            pr.stage("class_stats", total, xo, n, d, yo, K, cnt, mean, var, sums)
        else:
            pr.stage("class_stats_w", total, xo, n, d, yo, K, cnt, mean, var, sums, wo)
        return
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


def _given_priors(values, K, who, check_sum=False):
    """A user prior list as floats, checked as the reference checks it."""
    vals = [float(v) for v in (values.tolist() if hasattr(values, "tolist") else values)]  # glue: converts the user priors argument (vals-sized: user prior list)
    if len(vals) != K:
        raise ValueError(f"mojolearn: {who}: number of priors must match number of classes")
    if any(v < 0 for v in vals):  # glue: validates the user priors argument (vals-sized: user prior list)
        raise ValueError(f"mojolearn: {who}: priors must be non-negative")
    if (check_sum and abs(sum(vals) - 1.0) > 1e-8 * max(1.0, abs(sum(vals))) and  # glue: validates the user priors argument sums to one (vals-sized: user prior list)
            abs(sum(vals) - 1.0) > 1e-5):  # glue: validates the user priors argument sums to one (vals-sized: user prior list)
        raise ValueError(f"mojolearn: {who}: the sum of the priors should be 1")
    return vals


class _DevCodes:
    """A partial_fit batch's labels whose class codes the fit program makes
    on the device (lane fam2-prep-metrics, `_stage_partial_codes`): the
    numeric label buffer, the classes as sorted float32 categories, and
    `walk`, the Python route (the codes as an int32 Array; it names the
    labels of a refused batch)."""
    __slots__ = ("lb", "cats", "walk")

    def __init__(self, lb, cats, walk):
        self.lb, self.cats, self.walk = lb, cats, walk


def _partial_dev(est, y, n, walk):
    """`_DevCodes` when y is a numeric label buffer of n labels and every
    class is an int or float with an exact float32 word, strictly ascending
    (so `lookup`'s binary search over them is the dict's answer); else None."""
    cl = list(est.classes_)
    if not cl or any(type(c) not in (int, float) for c in cl):  # glue: the K class labels' types
        return None
    vals = [float(c) for c in cl]  # glue: the K class labels as floats
    cats = Array._from_flat(vals, (len(vals),), "<f4")
    ordered = all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))  # glue: the K classes ascend
    if cats.tolist() != vals or not ordered:
        return None
    lb = _label_buffer(y)
    if lb is None or lb.n != n:
        return None
    return _DevCodes(lb, cats, walk)


def _stage_partial_codes(pr, codes, mode):
    """The float class codes offset of a partial_fit batch. An int32 Array:
    `put_codes`. A `_DevCodes` (every tier, lane cpu2-l3-prep): staged in
    this program (lab_load, lookup among the classes, count_neg, f2_clamp0:
    a label outside classes_ counts and is clamped to class 0), and the
    program refuses the batch right after its run, before the caller reads a
    result, through the Python walk that names the labels."""
    if not isinstance(codes, _DevCodes):
        return pr.put_codes(codes)
    lb, cats = codes.lb, codes.cats
    K = cats.size
    x = _label_load(pr, lb)
    c, neg, yo = pr.work(lb.n), pr.alloc(1), pr.alloc(lb.n)
    pr.stage("lookup", lb.n, x, lb.n, 1, pr.put(cats), K, pr.put_scalar(K), c)
    pr.stage("count_neg", 1, c, lb.n, 1, neg)
    pr.stage("f2_clamp0", lb.n, c, yo)

    def refuse(prog, neg=neg, walk=codes.walk):
        if prog.values(neg, 1)[0] > 0:
            walk()
            raise ValueError("mojolearn: a target label in y does not exist in the initial classes")

    pr._after.append(refuse)
    return yo


def _partial_codes(est, y, classes, n, defer=False):
    """sklearn `_check_partial_fit_first_call` and the batch's class codes:
    the first call (no classes_ yet) needs `classes`, later ones may repeat
    them only unchanged; a label outside classes_ is refused. defer (lane
    fam2-prep-metrics): a numeric label buffer comes back as a `_DevCodes`
    for `_stage_partial_codes` (no Python walk over the labels)."""
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
    if defer:
        dev = _partial_dev(est, y, n, lambda: _partial_walk(est, y, n, first)[1])
        if dev is not None:
            return first, dev
    return _partial_walk(est, y, n, first)


def _partial_walk(est, y, n, first):
    """`_partial_codes`' Python route: labels of any kind (str, lists, a
    class list float32 cannot hold), and the refusal that names a batch's
    unknown labels."""
    labels = flatten_labels(y)
    if len(labels) != n:
        raise ValueError("mojolearn: X and y have different numbers of rows")
    index = {c: i for i, c in enumerate(est.classes_)}  # cpu-route: Python object labels of any kind, the explicit label input step
    bad = sorted({repr(v) for v in labels if v not in index})  # cpu-route: Python object labels of any kind, the explicit label input step
    if bad:
        raise ValueError(f"mojolearn: The target label(s) {bad} in y do not exist in the initial classes "
                         f"{est.classes_}")
    return first, Array.from_list([index[v] for v in labels], "<i4")  # cpu-route: Python object labels of any kind, the explicit label input step


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
        _class_stats(pr, wo, K * d, xo, n, d, yo, K, cnt, theta, var, _NONE, mode=mode)
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
        first, codes = _partial_codes(self, y, classes, n, defer=True)
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
        yo = _stage_partial_codes(pr, codes, mode)
        st = pr.alloc(6 * d)
        vs = pr.put_scalar(self.var_smoothing)
        eps = pr.alloc(1)
        bc, bm, bv = pr.alloc(K), pr.alloc(K * d), pr.alloc(K * d)
        wo = _nb_weights(pr, sample_weight, n)
        _col_stats(pr, xo, n, d, st)
        pr.stage("gnb_eps", 1, st + 2 * d, d, eps, vs)
        _class_stats(pr, wo, K * d, xo, n, d, yo, K, bc, bm, bv, _NONE, mode=mode)
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


def _csr_input(X):
    """(indptr, indices, data, n, d) of a scipy.sparse matrix as int32, int32
    and float32 C Arrays (its CSR form), or None for anything else."""
    if not (hasattr(X, "tocsr") and hasattr(X, "nnz") and hasattr(X, "shape")):
        return None
    X = X.tocsr()
    n, d = (int(v) for v in X.shape)  # glue: unpacks the two shape entries
    ip = _as_typed(X.indptr, "<i4", "C", 1, "indptr")[0]
    ix = _as_typed(X.indices, "<i4", "C", 1, "indices")[0]
    dv = _as_typed(X.data, "<f4", "C", 1, "data")[0]
    return ip, ix, dv, n, d


def _csr_dense_x(pr, csr, mode):
    """Lane idn-all: the dense n x d block of a CSR input, built ON THE
    DEVICE by the program itself (op csr_dense, x_prep/blocked.mojo): the
    CSR arrays go up, the block is device work words. Returns its offset, or
    None when the binding has no such stage (`_idn_int` bit 8 clear: the
    caller densifies on the host, the old form)."""
    if not _idn_int(mode) & _IDN_NB_CSR_DENSE:
        return None
    ip, ix, dv, n, d = csr
    if n * d > 2 ** 31 - 1:
        raise ValueError("mojolearn: X exceeds the native Int32 indexing bound")
    xo = pr.work(n * d)
    pr.stage("csr_dense", n, pr.put_words(ip), pr.put_words(ix), pr.put(dv), d, xo)
    return xo


def _check_nonnegative(pr_values, who):
    if any(v < 0 for v in pr_values):  # glue: raises on a negative per-column minimum (pr_values-sized: per-column minimum statistics)
        raise ValueError(f"mojolearn: Negative values in data passed to {who}")


class _DiscreteNB(_Classifier):
    #: MultinomialNB and ComplementNB take a CSR input on the FAST CSR path
    _csr_ok = False
    #: whether `_params` reads the column minimum row (the negative-input
    #: refusal of MultinomialNB / ComplementNB)
    _needs_min = False

    @classmethod
    def _nb_csr_ready(cls):
        """Whether this class fits a scipy.sparse CSR matrix without
        densifying it: FAST mode and the x_prep binding built on Apple with
        NB_TEXT_CSR (lane apple-fast-nb, default since the M3 A/B; it exports
        `x_prep_nb_csr_fit`). The bench hands such a build the text block as
        CSR. Also IDENTICAL on every vendor's device binding (lane
        idn-int-prep, IDN_NB_CSR, -D MOJOLEARN_IDN_NB_CSR_OFF). False
        everywhere else: FAST off Apple, the host column,
        -D MOJOLEARN_NB_TEXT_CSR_OFF."""
        if not cls._csr_ok:
            return False
        mode = _mode()
        try:
            if mode == "identical":
                # lane idn-int-prep: the IDENTICAL device binding on every
                # vendor (IDN_NB_CSR; the host column has no CSR entry and
                # takes the dense block: the same words)
                return bool(_idn_int(mode) & _IDN_NB_CSR)
            if mode != "fast":
                return False
            return _optional_prep_entry(_prep_binding("fast"), "x_prep_nb_csr_fit") is not None
        except Exception:
            return False

    def _csr_fast(self, X):
        """X's CSR parts when it is a scipy.sparse matrix and the FAST CSR
        path is on, else None (a dense input keeps main's program)."""
        if not type(self)._nb_csr_ready():
            return None
        return _csr_input(X)

    def _fit_counts_csr(self, csr, y):
        """`_fit_counts` on a CSR matrix (FAST + Apple + define only): the
        (class, feature) count table and the class counts from ONE upload of
        the CSR arrays (x_prep/fastnb_csr.mojo), then main's epilogue on them
        (the column-stats row is zero words, so `_params`' minimum check
        passes; a negative value is refused here from the kernel's flag)."""
        ip, ix, dv, n, d = csr
        codes = self._encode_y(y, n)
        K = len(self.classes_)
        mode = _mode()
        fit_csr = _optional_prep_entry(_prep_binding(mode), "x_prep_nb_csr_fit")
        fc = Array._from_flat([0.0] * (K * d), (K, d), "<f4")
        cnt = Array._from_flat([0.0] * K, (K,), "<f4")
        flag = Array.from_list([0], "<i4")
        fit_csr(addr_ro(ip, name="indptr"), addr_ro(ix, name="indices"), addr_ro(dv, name="data"),
                addr_ro(codes, name="y"), [n, d, K, dv.size],
                _addr_rw(fc, name="feature_count"), _addr_rw(cnt, name="class_count"), _addr_rw(flag, name="flag"))
        level = int(flag.tolist()[0])
        if mode == "identical":
            # x_prep/fastnb_csr.mojo: 2 a negative value; 1 the counts are not
            # exact integers below 2^24 (or the rows are not canonical), so
            # the caller runs the dense program
            if level == 1:
                return None
            level = 1 if level == 2 else 0
        if level != 0:
            raise ValueError(f"mojolearn: Negative values in data passed to {type(self).__name__} (input X)")
        pr = _Prog()
        st = pr.alloc(6 * d)
        z = pr.put_list([0.0] * (K * d))
        cnt_o, fc_o, clp = pr.alloc(K), pr.alloc(K * d), pr.alloc(K)
        pr.stage("add_arrays", K, pr.put(cnt), z, cnt_o)
        pr.stage("add_arrays", K * d, pr.put(fc), z, fc_o)
        self._prior_stages(pr, K, cnt_o, clp)
        return pr, mode, n, d, K, st, cnt_o, fc_o, clp

    def _csr_bias(self):
        """The class log prior the CSR scoring adds (None for none)."""
        return self.class_log_prior_

    def _scores(self, X, want):
        csr = self._csr_fast(X)
        if csr is None:
            return super()._scores(X, want)
        self._check_fitted()
        ip, ix, dv, n, d = csr
        if d != self.n_features_in_:
            raise ValueError(f"mojolearn: X has {d} features, but {type(self).__name__} was fitted with "
                             f"{self.n_features_in_}")
        K = len(self.classes_)
        jll_csr = _optional_prep_entry(_prep_binding(self.numeric_mode_), "x_prep_nb_csr_jll")
        jll_h = Array._from_flat([0.0] * (n * K), (n, K), "<f4")
        bias = self._csr_bias()
        args = (addr_ro(ip, name="indptr"), addr_ro(ix, name="indices"), addr_ro(dv, name="data"),
                addr_ro(self.feature_log_prob_, name="feature_log_prob_"),
                0 if bias is None else addr_ro(bias, name="class_log_prior_"),
                [n, d, K, dv.size], _addr_rw(jll_h, name="jll"))
        if self.numeric_mode_ == "identical":
            # lane idn-int-prep: the dense chain without its zero terms; a row
            # whose columns do not ascend strictly takes the dense program
            flag = Array.from_list([0], "<i4")
            jll_csr(*args, _addr_rw(flag, name="flag"))
            if int(flag.tolist()[0]) != 0:
                # lane idn-all: the dense program on a block the device
                # builds from the CSR arrays (no host densify mid-predict)
                return super()._scores(X, want, csr=csr)
        else:
            jll_csr(*args)
        pr = _Prog()
        z = pr.put_list([0.0] * (n * K))
        jll = pr.alloc(n * K)
        pr.stage("add_arrays", n * K, pr.put(jll_h), z, jll)
        return self._score_tail(pr, n, d, K, jll, want, None)

    def _fit_counts(self, X, y, binarize=None, sample_weight=None, csr=None):
        mode = _mode()
        pr = _Prog()
        # lane idn-all: `csr` (the IDENTICAL CSR route's fallback) is
        # densified by the program on the device, not by the host
        xo = _csr_dense_x(pr, csr, mode) if csr is not None else None
        if xo is None:
            arr = _x2d(X.toarray() if csr is not None else X)
            n, d = arr.shape
            xo = pr.put(arr)
        else:
            n, d = csr[3], csr[4]
        codes = self._encode_y(y, n)
        K = len(self.classes_)
        wo = _nb_weights(pr, sample_weight, n)
        if _nb_onepass(mode):
            # lane idn-int-prep: one unit per (block, column) counts every
            # class, binarizes (BernoulliNB) and flags a negative word; no
            # binarized copy and no column-stats pass. The same words.
            yo = pr.put_codes(codes)
            st = pr.alloc(6 * d)
            cnt, fc = pr.alloc(K), pr.alloc(K * d)
            clp = pr.alloc(K)
            _nb_counts(pr, wo, xo, n, d, yo, K, cnt, fc,
                       thr=_NONE if binarize is None else pr.put_scalar(binarize),
                       neg=st + 3 * d if self._needs_min else _NONE)
            self._prior_stages(pr, K, cnt, clp)
            return pr, mode, n, d, K, st, cnt, fc, clp
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
        csr = self._csr_fast(X) if sample_weight is None else None
        if csr is not None:
            got = self._fit_counts_csr(csr, y)
            if got is not None:
                return self._params(*got)
            # IDENTICAL only (`_fit_counts_csr`'s flag): the dense program,
            # its block built on the device from the CSR arrays (lane idn-all)
            return self._params(*self._fit_counts(X, y, getattr(self, "binarize", None), None, csr=csr))
        return self._params(*self._fit_counts(X, y, getattr(self, "binarize", None), sample_weight))

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """The reference's `_BaseDiscreteNB.partial_fit`: the batch's class and
        feature counts added to the running ones, then the log probabilities
        and the class log prior recomputed from the sums."""
        _check_alpha(self)
        arr = _x2d(X)
        n, d = arr.shape
        first, codes = _partial_codes(self, y, classes, n, defer=True)
        K = len(self.classes_)
        if first:
            mode = _mode()
        else:
            self._check_width(arr)
            mode = self.numeric_mode_
        pr = _Prog()
        xo = pr.put(arr)
        wo = _nb_weights(pr, sample_weight, n)
        onepass = _nb_onepass(mode)
        thr = _NONE
        if getattr(self, "binarize", None) is not None:
            if onepass:
                thr = pr.put_scalar(self.binarize)
            else:
                xb = pr.work(n * d)
                pr.stage("binarize", n * d, xo, n * d, pr.put_scalar(self.binarize), xb)
                xo = xb
        yo = _stage_partial_codes(pr, codes, mode)
        st = pr.alloc(6 * d)
        cnt, fc, clp = pr.alloc(K), pr.alloc(K * d), pr.alloc(K)
        if not onepass:
            _col_stats(pr, xo, n, d, st, var=False)
        if first:
            bc, bf = cnt, fc
        else:
            bc, bf = pr.alloc(K), pr.alloc(K * d)
        if onepass:
            # lane idn-int-prep (`_fit_counts`): the same words
            _nb_counts(pr, wo, xo, n, d, yo, K, bc, bf, thr=thr,
                       neg=st + 3 * d if self._needs_min else _NONE)
        else:
            _class_stats(pr, wo, K * d, xo, n, d, yo, K, bc, _NONE, _NONE, bf)
        if not first:
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
    _csr_ok = True
    _needs_min = True

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
    """The count of each class code 0 .. K-1 of an int32 codes Array: one
    program of p2m_ccount / p2m_cscan (lane apple-fast-py2mojo-prep)."""
    mode = _mode()
    n = int(codes.size)
    if n == 0 or K <= 0:
        return [0] * max(K, 0)
    pr = _Prog()
    _ro, tot = _p2m_class_rows(pr, pr.put_codes(codes), n, K, rows=False)
    pr.run(mode)
    return pr.get_i32(tot, K).tolist()


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
    if isinstance(arr, _arena_io.DeviceRows):
        # the user's estimator takes host rows (lane cpu4-misc)
        arr = arr.materialize()
    n, d = arr.shape
    mode = _mode()
    if codes is not None:
        # lane apple-fast-py2mojo-prep: the rows grouped by class on the device
        # (p2m_ccount .. p2m_cwrite, then p2m_rgather): class k's rows are one
        # contiguous block, ascending, the words the Python gather copied
        pr = _Prog()
        xo = pr.put(arr)
        ro, tot = _p2m_class_rows(pr, pr.put_codes(codes), n, K)
        xg = pr.alloc(n * d)
        pr.stage("p2m_rgather", n * d, xo, d, ro, xg)
        pr.run(mode)
        cnt = pr.get_i32(tot, K).tolist()
        starts = [0] + list(itertools.accumulate(cnt))[:-1]  # glue: row offsets of the K class blocks from the device class counts (K-sized)
        blocks = [pr.get(xg + starts[k] * d, (cnt[k], d)) for k in range(K)]  # glue: per-class row block views for the user estimator (K-sized: class count)
    else:
        blocks = [arr.copy()]
    out = []
    for k in range(K):  # cpu-route: fits the user covariance estimator once per class
        est.fit(blocks[k])
        if not hasattr(est, "covariance_"):
            raise ValueError(f"mojolearn: {type(est).__name__} does not have a covariance_ attribute")
        cov = est.covariance_
        flat = flatten_labels(cov.tolist() if hasattr(cov, "tolist") else cov)
        if len(flat) != d * d:
            raise ValueError(f"mojolearn: {who}: covariance_ of {type(est).__name__} is not ({d}, {d})")
        out.extend(float(v) for v in flat)  # cpu-route: reads the user covariance estimator output
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
    _qda_cov(pr, _mode(), xo, n, d, yo, K, mean, cnt, cov)
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
        _cls(pr, mode, xo, n, d, yo, K, cnt, mean, _NONE, _NONE)
        gflag, gofs = 0, 0
        if self.priors is not None:
            pv = _given_priors(self.priors, K, "LinearDiscriminantAnalysis")
            gflag, gofs = (2 if abs(sum(pv) - 1.0) > 1e-5 else 1), pr.put_list(pv)  # glue: validates the user priors argument sums to one (pv-sized: user prior list)
        pr.stage("lda_prep", 1, cnt, mean, K, d, n, priors, xbar, gflag, gofs)
        pr.stage("center_rows", n * d, xo, n, d, mean, yo, _NONE, z)
        _cs(pr, mode, z, n, d, stz)
        pr.stage("lda_w", d, stz + 2 * d, d, n, K, std, w)
        pr.stage("center_rows", n * d, xo, n, d, mean, yo, w, z2)
        _gram(pr, mode, z2, n, d, g)
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
        # the rank columns as one strided copy in Mojo (Array.__getitem__)
        self.scalings_ = full[:, :rank2] if rank2 else Array((d, 0), "<f4")
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
        _cls(pr, mode, xo, n, d, yo, K, cnt, mean, var, _NONE)
        gflag, gofs = 0, 0
        if self.priors is not None:
            pv = _given_priors(self.priors, K, "LinearDiscriminantAnalysis")
            gflag, gofs = (2 if abs(sum(pv) - 1.0) > 1e-5 else 1), pr.put_list(pv)  # glue: validates the user priors argument sums to one (pv-sized: user prior list)
        pr.stage("lda_prep", 1, cnt, mean, K, d, n, priors, xbar, gflag, gofs)
        eigen = self.solver == "eigen"
        est = self.covariance_estimator
        tot = None
        if eigen:
            y0 = pr.put_list([0.0] * n)
            c1, m1 = pr.alloc(1), pr.alloc(d)
            v1 = pr.alloc(d) if shr is not None else _NONE
            _cls(pr, mode, xo, n, d, y0, 1, c1, m1, v1, _NONE)
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
        if eigen and min(pr.values(e, d)) <= 0:  # glue: raises when the device reports a non-positive eigenvalue
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
        # C56: the incumbent binary answer is _pair's coefficient difference.
        # Build it once from the validated input, avoiding discarded K scores.
        # NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
        self._check_fitted()
        if len(self.classes_) == 2 and (_classical_shared(self.numeric_mode_) & 8):
            arr = _x2d(X)
            self._check_width(arr)
            n, d = arr.shape
            pr = _Prog()
            xo = pr.put(arr)
            chk = self._score_checks(pr, xo, n, d)
            coef, inter = pr.put(self.coef_), pr.put(self.intercept_)
            out = pr.alloc(n)
            pr.stage("matmul", n, xo, d, 1, coef, 1, 0, out, 1, d, inter, _NONE)
            pr.run(self.numeric_mode_)
            self._score_refusals(pr, d, chk)
            return pr.get(out, n)
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
        out = pr.alloc(n * max(mc, 1))
        if self.solver != "eigen" and (_classical_shared(self.numeric_mode_) & 4):
            # C04: the same center_rows -> matmul arithmetic in the consumer.
            # NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
            pr.stage("centered_matmul", n * mc, xo, n, d, xb, _NONE, _NONE,
                     sc, d, 1, out, mc, _NONE, _NONE)
        else:
            cen = xo if self.solver == "eigen" else pr.alloc(n * d)
            if self.solver != "eigen":
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
        mode = _mode()
        # lane apple-fast-py2mojo-prep: without a covariance_estimator the count check reads
        # the program's own class counts (class_stats `cnt`, exact integers) right after the
        # run, before anything else is read or raised; an estimator fits per class inside the
        # build, so its check stays first (`_class_counts`, a p2m program)
        late = est is None
        if not late and min(_class_counts(codes, K)) < 2:  # glue: raises when a class count from the binding is below two
            raise ValueError("mojolearn: y has only 1 sample in a class, covariance is ill defined")
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
        _cls(pr, mode, xo, n, d, yo, K, cnt, mean, var, _NONE, 1)
        gflag, gofs = 0, 0
        if self.priors is not None:
            gflag, gofs = 1, pr.put_list(_given_priors(self.priors, K, "QuadraticDiscriminantAnalysis"))
        pr.stage("lda_prep", 1, cnt, mean, K, d, n, priors, xbar, gflag, gofs)
        if est is not None:
            cov = pr.put_list(_estimator_covs(est, arr, codes, K, "QuadraticDiscriminantAnalysis"))
        else:
            _qda_cov(pr, mode, xo, n, d, yo, K, mean, cnt, cov)
        if shr is not None:
            pr.stage("da_shrink", K, xo, n, d, yo, mean, var, cnt, cov, pr.put_scalar(shr), pr.alloc(K))
        keep = pr.alloc(K * d * d) if (self.store_covariance and eigen) else None
        if keep is not None:
            one = pr.put_list([1.0])
            for k in range(K):  # glue: stages one pooled covariance copy per class (K-sized: class count)
                pr.stage("da_pool", d * d, cov + k * d * d, 1, d, one, keep + k * d * d, _NONE, _NONE)
        pr.stage("eigh", K, cov, d, d * d, ev, evec)
        pr.stage("qda_prep", K, ev, evec, K, d, reg, cnt, n, rot, logc, s2, gflag, gofs)
        if self.store_covariance and not eigen:
            keep = pr.alloc(K * d * d)
            pr.stage("sym_fn", K * d * d, s2, evec, d, 2, keep)
        pr.run(mode)
        if late and min(pr.values(cnt, K)) < 2:  # glue: raises when a class count from the binding is below two
            raise ValueError("mojolearn: y has only 1 sample in a class, covariance is ill defined")
        s2v = pr.values(s2, K * d)
        for k in range(K):  # glue: rank check per class before raising (K-sized: class count)
            if sum(1 for v in s2v[k * d:(k + 1) * d] if v > self.tol) < d:  # glue: counts device singular values above tol to raise (d-sized: feature count)
                raise ValueError(f"mojolearn: the covariance matrix of class {self.classes_[k]!r} is not full "
                                 f"rank. Increase the value of `{'shrinkage' if eigen else 'reg_param'}` to "
                                 "reduce the collinearity.")
        if keep is not None:
            self.covariance_ = [pr.get(keep + k * d * d, (d, d)) for k in range(K)]  # glue: per-class covariance views (K-sized: class count)
        self.means_, self.priors_ = pr.get(mean, (K, d)), pr.get(priors, K)
        self.rotations_ = [pr.get(evec + k * d * d, (d, d)) for k in range(K)]  # glue: per-class rotation views (K-sized: class count)
        self.scalings_ = [pr.get(s2 + k * d, d) for k in range(K)]  # glue: per-class scaling views (K-sized: class count)
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
        mode = _mode()
        perm = None
        if self.subsample is not None and n > self.subsample:
            # the rows are drawn without replacement in the program below (f2_perm_rows,
            # x_prep/fam2.mojo: a keyed permutation of [0, n), one thread per draw), then
            # gathered there; every tier (lane cpu2-l3-prep deleted the host
            # `draw_rows_without_replacement_i32` route)
            perm = ((0 if self.random_state is None else int(self.random_state)) & 0xFFFFFFFFFFFFFFFF,
                    int(self.subsample))
        n_all = n
        if perm is not None:
            n = perm[1]
        nq = max(1, min(int(self.n_quantiles), n))
        pr = _Prog()
        xo = pr.put(arr)
        if perm is not None:
            ro, xs = pr.alloc(n), pr.alloc(n * d)
            pr.stage("f2_perm_rows", n, _seed_words(pr, perm[0]), n_all, ro)
            pr.stage("p2m_rgather", n * d, xo, d, ro, xs)
            xo = xs
        # the references i / (nq - 1) (0 for one quantile) on the device, in binary64
        # rounded to float32 as the Python list was (c2_grid KIND 1, lane cpu2-l3-prep)
        so, st, qf = pr.work(n * d), pr.alloc(6 * d), _grid(pr, [nq - 1], nq, 1)
        qo = pr.alloc(nq * d)
        pr.stage("sort_cols", d, xo, n, d, so, 0)
        _cs(pr, mode, xo, n, d, st, var=False)
        pr.stage("quantile", nq * d, so, n, d, qf, nq, qo, st)
        # quantiles_ (nq, d): the (d, nq) block written column-major on the
        # device (p2m_transpose; lane pyglue-numeric: a Python transpose)
        qt = pr.alloc(nq * d)
        pr.stage("p2m_transpose", nq * d, qo, d, nq, qt)
        pr.run(mode)
        self._q = pr.get(qo, nq * d)
        self.quantiles_ = pr.get(qt, (nq, d))
        self.references_ = pr.get(qf, nq)
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
    side. MOJOLEARN_XPREP_PT_SPEC = S (default 3; 0: one evaluation per
    fold, the staged search). The rule is a size range, not a shape: S is
    lowered until the (2^S - 1) candidates' transforms (n*d words each) fit
    in 2^28 words, so every (n, d) gets the deepest speculation its buffer
    allows. The default 3 was measured on one row (m4pro-b, 1M x 11: staged
    4.55 s, S=2 3.31, S=3 2.95, S=4 3.09) and needs neighbor-shape
    validation (n*d around 2^28 / 7 and 2^28 / 3, d = 8..256). Every S gives the same lambdas. IDENTICAL only: FAST keeps the
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
        host = _optional_prep_entry(_prep_binding(mode), "x_prep_host_column") is not None
        # PT_SCORE_STABLE (bit 32, FAST+Apple default, rollback MOJOLEARN_PT_SCORE_STABLE_OFF): the device's centered coordinates; never on the host binding
        centered = bool(_ptimpute_flags(mode) & 32) and not host
        anchor, anchor_kind = (pr.alloc(d), pr.alloc(d)) if centered else (_NONE, _NONE)
        _cs(pr, mode, xo, n, d, st)
        # lane fam-prep-metrics, IDN_PT_BLOCKED (IDENTICAL, `_idn_fam` bit 8): every fold of the
        # search by row blocks (x_prep/pt_blocked.mojo). The host column then stages the SAME
        # program as the device (its pt_fit folds in row order), so both take the blocked order.
        fam_pt = _blocked() and bool(_idn_fam(mode) & _IDN_PT_BLOCKED)
        nb = (n + _XB - 1) // _XB
        if not fam_pt and host:
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
                pr.stage("pt_init", d, method, st, d, lam, state, leval, anchor, anchor_kind, int(self.standardize))
                if not tiled:
                    pr.stage("pt_log", n * d, xo, n, d, method, lg)
                if fam_pt:
                    bps, bpc, bss = pr.work(nb * d * mmax), pr.work(nb * d * mmax), pr.work(nb * d * mmax)
                    bpj, bmean, bcnt = pr.work(nb * d), pr.work(d * mmax), pr.work(d * mmax)
                k0 = 0
                while k0 <= _PT_EVALS - 2:
                    steps = 2 if k0 == 0 else min(spec, _PT_EVALS - 1 - k0)
                    m = 2 if k0 == 0 else 2 ** steps - 1
                    pr.stage("pt_spts", d, state, leval, spl, m, k0)
                    pr.stage("pt_smap", m * n * d, xo, n, d, method, spl, m, tv, lg, il)
                    if fam_pt:
                        first = 1 if k0 == 0 else 0
                        pr.stage("ptb_part1", nb * d * m, xo, n, d, method, tv, m, state, bps, bpc, bpj, nb, first, il)
                        pr.stage("ptb_mean", d * m, bps, bpc, bpj, nb, d, m, state, bmean, bcnt, first)
                        pr.stage("ptb_part2", nb * d * m, tv, n, d, m, state, bmean, bcnt, bss, nb, il)
                        pr.stage("ptb_fin", d * m, bss, nb, d, m, state, spl, vals, bcnt)
                    else:
                        pr.stage("pt_sfold", d * m, xo, n, d, method, tv, m, state, spl, vals, 1 if k0 == 0 else 0, il)
                    pr.stage("pt_sres", d, state, leval, m, vals, k0, steps, lam)
                    k0 += steps
            else:
                state, leval = pr.alloc(_PT_STATE * d), pr.alloc(d)
                tv, lg = (pr.alloc(1), pr.alloc(1)) if tiled else (pr.alloc(n * d), pr.alloc(n * d))
                pr.stage("pt_init", d, method, st, d, lam, state, leval, anchor, anchor_kind, int(self.standardize))
                if not tiled:
                    pr.stage("pt_log", n * d, xo, n, d, method, lg)
                if fam_pt:
                    bps, bpc, bss = pr.work(nb * d), pr.work(nb * d), pr.work(nb * d)
                    bpj, bmean, bcnt = pr.work(nb * d), pr.work(d), pr.work(d)
                for k in range(_PT_EVALS):  # glue: stages the fixed optimizer evaluations on the device (_PT_EVALS-sized: constant evaluation count)
                    # tiled: the device skips pt_map and fuses it into pt_fold (LG1 = 0 either way)
                    pr.stage("pt_map", n * d, xo, n, d, method, leval, tv, 0 if tiled else lg + 1)
                    if fam_pt:
                        # one candidate a column (M = 1, contiguous): the blocked fold, then pt_finish's step
                        first = 1 if k == 0 else 0
                        pr.stage("ptb_part1", nb * d, xo, n, d, method, tv, 1, state, bps, bpc, bpj, nb, first, 0)
                        pr.stage("ptb_mean", d, bps, bpc, bpj, nb, d, 1, state, bmean, bcnt, first)
                        pr.stage("ptb_part2", nb * d, tv, n, d, 1, state, bmean, bcnt, bss, nb, 0)
                        pr.stage("ptb_step", d, xo, n, d, method, tv, k, state, leval, lam, bss, nb, bcnt)
                    else:
                        pr.stage("pt_fold", d, xo, n, d, method, tv, k, state, leval, lam)
        mean, scale = pr.alloc(d), pr.alloc(d)
        if self.standardize:
            # PT_FUSED_TRANSFORM (bit 4, FAST + Apple): the device folds col_stats of the transform
            # straight from X (x_prep/fastpt.mojo cs_tile_kernel) and skips this col_stats stage, so
            # the transformed block is never written: a word
            fused = bool(_ptimpute_flags(mode) & 4)
            tx, st2 = pr.alloc(1) if fused else pr.alloc(n * d), pr.alloc(6 * d)
            pr.stage("pt_apply", n * d, xo, n, d, lam, method, _NONE, _NONE, tx, anchor, anchor_kind)
            _cs(pr, mode, tx, n, d, st2)
            pr.stage("std_params", d, st2, d, mean, scale)
        pr.run(mode)
        if method == 1 and any(v <= 0 for v in pr.values(st + 3 * d, d)):  # glue: raises on a non-positive per-column minimum (d-sized: feature count)
            raise ValueError("mojolearn: The Box-Cox transformation can only be applied to strictly positive data")
        self.lambdas_ = pr.get(lam, d)
        self._pt_anchor = pr.get(anchor, d) if centered and self.standardize else None
        self._pt_anchor_kind = pr.get(anchor_kind, d) if centered and self.standardize else None
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
        anchor = getattr(self, "_pt_anchor", None)
        ao = pr.put(anchor) if anchor is not None else _NONE
        ko = pr.put(self._pt_anchor_kind) if anchor is not None else _NONE
        pr.stage(op, n * d, xo, n, d, lo, self._method, mo, so, out, ao, ko)
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
            lo, hi = (int(v) for v in self.degree)  # glue: unpacks the two degree values
        if hi < 0 or lo < 0 or lo > hi:
            raise ValueError(f"mojolearn: invalid degree {self.degree!r}")
        comb = combinations if self.interaction_only else combinations_with_replacement
        it = chain.from_iterable(comb(range(d), i) for i in range(max(1, lo), hi + 1))  # glue: enumerates polynomial terms of feature indices (d-sized: feature count)
        if self.include_bias:
            it = chain(comb(range(d), 0), it)
        return [tuple(c) for c in it]  # glue: materializes the polynomial term list (it-sized: polynomial terms of feature indices)

    def fit(self, X, y=None):
        if self.order not in ("C", "F"):
            raise ValueError("mojolearn: PolynomialFeatures order must be 'C' or 'F'")
        d = _x2d(X).shape[1]
        self._terms = self._combos(d)
        self.n_features_in_, self.n_output_features_ = d, len(self._terms)
        self.powers_ = Array.from_list([[t.count(j) for j in range(d)] for t in self._terms] or [[0] * d], "<i8")  # glue: powers_ table of the polynomial terms (d-sized: feature count)
        self.numeric_mode_ = _mode()
        return self

    def transform(self, X):
        self._check_fitted()
        arr = _x2d(X)
        self._check_width(arr)
        n, d = arr.shape
        idx, start = [], [0]
        for t in self._terms:  # glue: flattens term index tables for the program (_terms-sized: polynomial terms of feature indices)
            idx.extend(t)
            start.append(len(idx))
        nout = len(self._terms)
        pr = _Prog()
        xo, io, so = pr.put(arr), pr.put_list(idx or [0]), pr.put_list(start)
        # the output region measured slower here (m4-a: 0.452 -> 0.523 s): arena words
        out = pr.alloc(n * nout)
        pr.stage("poly", n * nout, xo, n, d, io, so, nout, out)
        fo = _p2m_f_stage(pr, out, n, nout) if self.order == "F" else None
        pr.run(self.numeric_mode_)
        if fo is not None:
            return _p2m_f_get(pr, fo, n, nout)
        if self.order == "F":       # an empty block
            return Array._owned(array.array("f"), (n, nout), "<f4", "F")
        return pr.get(out, (n, nout))


def _p2m_f_stage(pr, off, n, w):
    """Lane apple-fast-py2mojo-prep: a p2m_transpose stage writing the (n, w)
    block at off column-major into the program's output; read it back with
    `_p2m_f_get`. None for an empty block."""
    if n * w <= 0:
        return None
    fo = pr.output(n * w)
    pr.stage("p2m_transpose", n * w, off, n, w, fo)
    return fo


def _p2m_f_get(pr, fo, n, w):
    """The column-major block `_p2m_f_stage` wrote, as an (n, w) Fortran-ordered Array."""
    return Array._view_of(pr.get(fo, (n * w,)), (n, w), "F")


def _check_weights(sample_weight, n, who):
    """sample_weight as float32 words (nonnegative, length n) and the count
    of its positive entries: one device program (p2m_sel_count / scan over
    the weights, col_stats for the minimum; lane pyglue-numeric: Python
    passes over the weight list)."""
    w = as_f32_c(sample_weight, ndim=1, name="sample_weight")[0]
    if w.size != n:
        raise ValueError(f"mojolearn: sample_weight has {w.size} entries; X has {n} rows")
    pr = _Prog()
    wo = pr.put(w)
    st = pr.alloc(6)
    _cs(pr, _mode(), wo, n, 1, st, var=False)
    pos = _p2m_sel(pr, wo, n, 1, 1, pr.alloc(n))
    pr.run(_mode())
    cnt, lo = pr.values(st, 1)[0], pr.values(st + 3, 1)[0]
    m = int(pr.get_i32(pos, 1).tolist()[0])
    if int(cnt) != n or lo < 0:            # a NaN is not counted by col_stats
        raise ValueError(f"mojolearn: {who} sample_weight must be nonnegative")
    if m == 0:
        raise ValueError(f"mojolearn: {who} sample_weight is all zero")
    return w, m


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


def _grid(pr, nb, w, kind):
    """Stages c2_grid (lane cpu2-l3-prep): one level row of stride w per entry
    b of nb (KIND 0: i * (100 / b), then 100; 1: i / b; 2: 100 * (i * (1 /
    b)), then 100; zeros past b), binary64 rounded to float32 as the Python
    lists were. Returns the (len(nb), w) block's offset."""
    out = pr.alloc(len(nb) * w)
    pr.stage("c2_grid", len(nb) * w, pr.put_list(nb), w, kind, out)
    return out


def _weighted_levels(pr, ug, ucnt, n, d, nb, kind, average, out):
    """A stage of the reference's `_weighted_percentile` of every column at
    the percent levels `_grid` writes for nb (one b per column, a common
    stride max + 1) into `out` (column c at c * stride), over
    `_weighted_groups`."""
    nbmax = max(nb)  # glue: the widest level row (a shape)
    lv = _grid(pr, nb, nbmax + 1, kind)
    pr.stage("kbins_wq", d, ug, n, d, ucnt, pr.put_list(nb), nbmax, lv, int(average), out)


def _spline_fused(mode):
    """lane apple-fast-gap-kapprox2: whether the binding is the FAST Apple
    build with -D MOJOLEARN_SPLINE_FAST_FUSED (it alone exports
    `x_prep_spline_fused`)."""
    if str(mode).strip().lower() != "fast":
        return False
    return _optional_prep_entry(_prep_binding(mode), "x_prep_spline_fused") is not None


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
        if self.handle_missing == "error" and any(int(v) != n for v in pr.values(st, d)):  # glue: raises on a NaN count from the device (d-sized: feature count)
            raise ValueError("mojolearn: Input X contains NaN values and `SplineTransformer` is configured to "
                             "error in this case (handle_missing='error'). To avoid this error, set "
                             "handle_missing='zeros' to encode missing values as splines with value 0 or ensure "
                             "no missing values in X.")

    def fit(self, X, y=None, sample_weight=None):
        self._fit(X, sample_weight, False)
        return self

    def fit_transform(self, X, y=None, sample_weight=None):
        # lane apple-fast-gap-kapprox2: with `x_prep_spline_fused` (the FAST
        # Apple build with -D MOJOLEARN_SPLINE_FAST_FUSED only) one program fits
        # and applies; 'error' extrapolation and order 'F' keep fit + transform.
        if self.extrapolation != "error" and self.order == "C" and _spline_fused(_mode()):
            return self._fit(X, sample_weight, True)
        return self.fit(X, y, sample_weight=sample_weight).transform(X)

    def _fit(self, X, sample_weight, fuse):
        """fit; with `fuse`, the same program also applies the splines to X
        and returns the output (fit_transform's words in one run)."""
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
            rows = [[float(v) for v in (r.tolist() if hasattr(r, "tolist") else r)] for r in  # glue: converts the user knots argument (r-sized: user knots argument rows)
                    (self.knots.tolist() if hasattr(self.knots, "tolist") else self.knots)]
            nk = len(rows)
            if nk < 2:
                raise ValueError("mojolearn: Number of knots, knots.shape[0], must be >= 2.")
            if any(len(r) != d for r in rows):  # glue: validates the user knots argument (rows-sized: user knots argument rows)
                raise ValueError("mojolearn: knots.shape[1] == n_features is violated.")
            cols = [list(array.array("f", [r[c] for r in rows])) for c in range(d)]  # glue: transposes the user knots argument (d-sized: feature count)
            if not all(b > a for col in cols for a, b in zip(col, col[1:])):  # glue: validates the user knots argument is sorted (cols-sized: user knots argument columns)
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
        # MOJOLEARN_SPLINE_FAST_FUSED: no n*d arena block when nothing sorts
        # (main's is never written and comes back from the device unread: 64 MB
        # at the board's 1M x 16), and the count / min / max rows by the blocked
        # units (one unit per row block and column) instead of one threadgroup
        # per column over every row. The same knots (counts, minima, maxima).
        fused = _spline_fused(mode)
        so = pr.work(n * d) if sorts else (None if fused else pr.alloc(n * d))
        st = pr.alloc(6 * d)
        knots = pr.alloc(d * (nk + 2 * k))
        if fused:
            _col_stats(pr, xo, n, d, st, var=False)
        else:
            _cs(pr, mode, xo, n, d, st, var=False)
        uniform, kst = 0, st
        if given:
            base = pr.put_list([v for col in cols for v in col])  # glue: packs the user knots argument (cols-sized: user knots argument columns)
        elif self.knots == "quantile":
            base = pr.alloc(d * nk)
            if w is None:
                qf = _grid(pr, [nk - 1], nk, 1)
                pr.stage("sort_cols", d, xo, n, d, so, 0)
                pr.stage("quantile", d * nk, so, n, d, qf, nk, base, st)
            else:
                ug, ucnt = _weighted_groups(pr, arr, w, n, d)
                _weighted_levels(pr, ug, ucnt, n, d, [nk - 1] * d, 2, False, base)
        else:
            base, uniform = pr.alloc(d * nk), 1
            if w is not None and wl < n:
                # the positive-weight rows on the device
                kst = pr.alloc(6 * d)
                _cs(pr, mode, _p2m_positive_rows(pr, xo, pr.put(w), n, d, wl), wl, d, kst, var=False)
        pr.stage("spline_knots", d, base, nk, d, k, knots, uniform, kst, int(periodic))
        nspl = nk - 1 if periodic else nk + k - 1
        W = d * (nspl if self.include_bias else nspl - 1)
        out = None
        if fuse:
            out = pr.output(n * W)
            pr.stage("spline_apply", n * d, xo, n, d, knots, nk, k, self._EXTRAP[self.extrapolation], W,
                     1 if self.include_bias else 0, out)
        pr.run(mode)
        self._no_nan(pr, st, d, n)
        self._knots = pr.get(knots, d * (nk + 2 * k))
        flat = pr.values(knots, d * (nk + 2 * k))
        wd = nk + 2 * k
        self.bsplines_ = [Array.from_list(flat[c * wd:(c + 1) * wd], "<f4") for c in range(d)]  # glue: per-column fitted spline knot views (d-sized: feature count)
        self._lo = [flat[c * wd + k] for c in range(d)]  # glue: per-column fitted lower knot (d-sized: feature count)
        self._hi = [flat[c * wd + k + nk - 1] for c in range(d)]  # glue: per-column fitted upper knot (d-sized: feature count)
        self._nk, self._k = nk, k
        self.n_features_out_ = W
        self.numeric_mode_, self.n_features_in_ = mode, d
        return pr.get(out, (n, W)) if fuse else self

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
            if _spline_fused(self.numeric_mode_):
                _col_stats(pr, xo, n, d, st, var=False)
            else:
                _cs(pr, self.numeric_mode_, xo, n, d, st, var=False)
        pr.stage("spline_apply", n * d, xo, n, d, ko, self._nk, self._k, self._EXTRAP[self.extrapolation], W,
                 1 if self.include_bias else 0, out)
        fo = _p2m_f_stage(pr, out, n, W) if self.order == "F" else None
        pr.run(self.numeric_mode_)
        if check:
            self._no_nan(pr, st, d, n)
        if self.extrapolation == "error":
            lo, hi = pr.values(st + 3 * d, d), pr.values(st + 4 * d, d)
            if any(a < b for a, b in zip(lo, self._lo)) or any(a > b for a, b in zip(hi, self._hi)):  # glue: raises when data leave the fitted knot range (lo-sized: per-column minimum statistics)
                raise ValueError("mojolearn: X contains values beyond the limits of the knots")
        if fo is not None:
            return _p2m_f_get(pr, fo, n, W)
        if self.order == "F":       # an empty block
            return Array._owned(array.array("f"), (n, W), "<f4", "F")
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
            if not any(map(_pm.isnan, fl)) and array.array("f", fl).tolist() == fl:
                return fl
        except OverflowError:
            pass
    out = []
    for v in values:  # cpu-route: Python list labels, the explicit label input step
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
    ints = set(map(type, values)) == _INT_ONLY or all(isinstance(v, numbers.Integral) for v in values)  # cpu-route: Python list labels kind test, the explicit label input step
    classes = [int(c) if ints else float(c) for c in cats.tolist()]  # glue: class list from the device categories (cats-sized: distinct class values)
    return classes, cats


def _label_codes(pr, values, cats):
    """Stages: each label's index among `cats` (or -1). Returns the codes
    offset and the unknown-count offset."""
    arr = Array._from_flat([float(v) for v in values], (len(values), 1), "<f4")  # cpu-route: Python list labels to a buffer, the explicit label input step
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
        return _multilabel_buffer(y, shape)
    elif isinstance(y, (list, tuple)) and y and all(isinstance(r, (list, tuple)) for r in y):  # cpu-route: Python list-of-lists y, the explicit label input step
        rows = [list(r) for r in y]  # cpu-route: Python list-of-lists y, the explicit label input step
    else:
        return None
    if not rows or len(rows[0]) < 2:
        return None
    if any(len(r) != len(rows[0]) for r in rows):  # cpu-route: Python list-of-lists y, the explicit label input step
        raise ValueError("mojolearn: y rows have different lengths")
    distinct = set()
    for r in rows:  # cpu-route: Python list-of-lists y, the explicit label input step
        for v in r:  # cpu-route: Python list-of-lists y, the explicit label input step
            if isinstance(v, bool) or not isinstance(v, numbers.Real) or v != v or float(v) != int(float(v)):
                raise ValueError("mojolearn: Multioutput target data is not supported with label binarization")
            distinct.add(float(v))
    if len(distinct) > 2:
        raise ValueError("mojolearn: Multioutput target data is not supported with label binarization")
    return _x2d(Array.from_list([[float(v) for v in r] for r in rows], "<f4"), "y")  # cpu-route: Python list-of-lists y, the explicit label input step


def _multilabel_buffer(y, shape):
    """`_multilabel_indicator` of a 2-D buffer (lane py-runtime round 2: no
    `tolist` walk): every value finite and integral (the core helper's
    integral test), at most two distinct values (native min, max and two
    equality counts), then float32 as the list route's `float(v)` made it."""
    if not shape[0] or shape[1] < 2:
        return None
    if "bool" in str(getattr(y, "dtype", "")):
        raise ValueError("mojolearn: Multioutput target data is not supported with label binarization")
    from ._array import _NATIVE_CODE, _REDUCE_INTEGRAL
    from ._buffer import as_f64_c
    Y, _ = as_f64_c(y, ndim=2, name="y")
    refuse = ValueError("mojolearn: Multioutput target data is not supported with label binarization")
    if not int(Y._native_reduce(_REDUCE_INTEGRAL)):
        raise refuse
    lo, hi = Y.min(), Y.max()
    hits = (Y == lo).sum() + ((Y == hi).sum() if hi != lo else 0)  # glue: native equality and native integer sums (Array helpers)
    if hits != Y.size:
        raise refuse
    return _x2d(Y.astype("<f4"), "y")


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
    classes = [float(c) for c in vals] if lb.is_float else [int(c) for c in vals]  # glue: class list from the device categories (vals-sized: distinct class values)
    return classes, cats


_LABEL_PRESENT = {}


def _label_present_enabled(mode):
    # FAST + Apple (LABEL_DIRECT) and, lane idn-int-prep, IDENTICAL on every
    # vendor and the host column (IDN_LABEL): the binding's export is the switch
    if mode not in ("fast", "identical"):
        return False
    binding = _prep_binding(mode)
    key = id(binding)
    if key not in _LABEL_PRESENT:
        fn = _optional_prep_entry(binding, "x_prep_label_present")
        _LABEL_PRESENT[key] = bool(fn()) if fn is not None else False
    return _LABEL_PRESENT[key]


def _label_fit_present(mode, lb, codes):
    """Exact small-integer distinct labels on GPU, avoiding an n-row sort
    and n-word download. A GPU BAD flag rejects values outside [0, 4096),
    including fractional/inexact labels. The caller then uses its unchanged
    general GPU sort. No host scan of the input or target values."""
    n, R, ch = lb.n, _CAT_R, _CAT_CH
    pr = _Prog()
    x = _label_load(pr, lb)
    fl, bad = pr.work(R), pr.alloc(1)
    u, k = pr.alloc(R), pr.alloc(1)
    nch = -(-R // ch)
    cnt, off = pr.work(nch), pr.work(nch)
    pr.stage("cat_zero", R, fl)
    pr.stage("cat_present", n, x, n, 1, R, fl, bad)
    pr.stage("pres_count", nch, fl, R, ch, cnt)
    pr.stage("uniq_scan", 1, cnt, nch, off, k)
    pr.stage("pres_write", nch, fl, R, ch, off, u)
    out = None
    if codes:
        c = pr.work(n)
        pr.stage("lookup", n, x, n, 1, u, R, k, c)
        out = pr.output(n, "i")
        pr.stage("f2i", n, c, out)
    pr.run(mode)
    if pr.get_i32(bad, 1).tolist()[0]:
        return None
    got = _label_classes_of(pr, lb, u, k)
    if got is None:
        return None
    return got[0], got[1], (pr.get_i32(out, n) if codes else None)


def _label_fit_device(mode, lb, codes=False):
    """One program: (classes, cats, int32 codes Array or None), or None for
    the old route."""
    if _label_present_enabled(mode):
        got = _label_fit_present(mode, lb, codes)
        if got is not None:
            return got
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
            index = {c: i for i, c in enumerate(self._classes)}  # cpu-route: str or object labels, the explicit label input step
            missing = [v for v in values if v not in index]  # cpu-route: str or object labels, the explicit label input step
            if missing:
                raise ValueError(f"mojolearn: y contains previously unseen labels: {missing[:5]}")
            return Array.from_list([index[v] for v in values], "<i4")  # cpu-route: str or object labels, the explicit label input step
        n = len(values)
        pr = _Prog()
        codes, neg = _label_codes(pr, values, self._cats)
        out = pr.alloc(n)
        pr.stage("f2i", n, codes, out)
        pr.run(self.numeric_mode_)
        if pr.values(neg, 1)[0] > 0:
            raise ValueError("mojolearn: y contains previously unseen labels")
        return pr.get_i32(out, n)

    def _inverse_device(self, y):
        """An integer or float code buffer over numeric classes, one program
        on every tier (lane fam2-prep-metrics; lane cpu2-l3-prep: FAST and
        float code buffers too): each code checked (an integer in [0, K)) and
        its class written as the int64 / float64 word returned
        (f2_code_gather, x_prep/fam2.mojo). None: str classes or a code list
        (the explicit input-prep route)."""
        if self._cats is None:
            return None
        lb = _label_buffer(y)
        if lb is None:
            return None
        n = lb.n
        ints = label_kind(self._classes) == "int"
        pr = _Prog()
        out, neg = _stage_class_gather(pr, _label_load(pr, lb), n, self._cats, ints)
        pr.run(self.numeric_mode_)
        if pr.values(neg, 1)[0] > 0:
            raise ValueError("mojolearn: y contains previously unseen labels")
        return _class_words(pr, out, n, ints)

    def inverse_transform(self, y):
        self._check_fitted()
        got = self._inverse_device(y)
        if got is not None:
            return got
        # str classes or a Python code list: the explicit input-prep route (G5)
        codes = [int(c) for c in flatten_labels(y)]  # cpu-route: str classes or a Python code list, the explicit label input step
        if any(c < 0 or c >= len(self._classes) for c in codes):  # cpu-route: str classes or a Python code list, the explicit label input step
            raise ValueError("mojolearn: y contains previously unseen labels")
        return _classes_array([self._classes[c] for c in codes])  # cpu-route: str classes or a Python code list, the explicit label input step


def _stage_class_gather(pr, codes, n, cats, ints):
    """Stages f2_code_gather (x_prep/fam2.mojo): the n float codes at `codes`
    to their classes among `cats` as 64-bit words (int64 when ints, else
    binary64), plus the count of codes that are not an integer in
    [0, cats.size). Returns (words offset, count offset)."""
    out, bad, neg = pr.alloc(2 * n), pr.work(n), pr.alloc(1)
    pr.stage("f2_code_gather", n, codes, cats.size, pr.put(cats), 0 if ints else 1, out, bad)
    pr.stage("count_neg", 1, bad, n, 1, neg)
    return out, neg


def _class_words(pr, out, n, ints):
    """The n 64-bit class words `_stage_class_gather` wrote, as an int64 or
    float64 Array (one byte copy, no Python object per label)."""
    words = pr.get_i32(out, 2 * n)
    store = array.array("q" if ints else "d")
    store.frombytes(ctypes.string_at(words._addr, 8 * n))
    return Array._owned(store, (n,), "<i8" if ints else "<f8", "C")


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
            index = {c: i for i, c in enumerate(self._classes)}  # cpu-route: str or object labels, the explicit label input step
            codes = pr.put_list([index.get(v, -1) for v in values])  # cpu-route: str or object labels, the explicit label input step
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
        else:
            if W > 2:
                raise ValueError("mojolearn: output_type='binary', but y.shape = " + str((n, W)))
            if threshold is None:
                threshold = (self.pos_label + self.neg_label) / 2.0
            # the last column against the threshold, one code per row (c2_bin_code)
            codes = pr.alloc(n)
            pr.stage("c2_bin_code", n, pr.put(arr), n, W, pr.put_scalar(threshold), codes)
        if self._cats is None:
            # str classes: the codes come back for the explicit object-label decode (G5)
            pr.run(self.numeric_mode_)
            idx = [int(v) for v in pr.values(codes, n)]  # cpu-route: str classes decode to Python objects, the explicit label output step
            if K == 1:
                idx = [0] * n
            return _classes_array([self._classes[i] for i in idx])  # cpu-route: str classes decode to Python objects, the explicit label output step
        # numeric classes (lane cpu2-l3-prep): each code's class gathered on the device
        # (f2_code_gather); one class: both codes name it
        ints = label_kind(self._classes) == "int"
        cats = self._cats if K > 1 else Array._from_flat(self._cats.tolist() * 2, (2,), "<f4")
        out, _neg = _stage_class_gather(pr, codes, n, cats, ints)
        pr.run(self.numeric_mode_)
        return _class_words(pr, out, n, ints)


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
    label at all (lane pyglue-numeric deleted the MOJOLEARN_MLB_DEVICE
    switch). glue: the walks below convert the caller's Python containers
    into one buffer, in C (map, chain, array)."""
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
    offs = Array._owned(array.array("i", itertools.accumulate(map(len, rows), initial=0)), (len(rows) + 1,),  # cpu-route: offsets of the user's Python label rows, the explicit label input step
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
            flat = [v for row in y for v in row]  # cpu-route: Python iterables of labels, the explicit label input step
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
        y = [list(row) for row in y]  # cpu-route: Python iterables of labels, the explicit label input step
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
        rows = [list(r) for r in y]  # cpu-route: Python list-of-lists y, the explicit label input step
        n, K = len(rows), len(self._classes)
        flat = [v for r in rows for v in r]  # cpu-route: Python iterables of labels, the explicit label input step
        owner = [i for i, r in enumerate(rows) for _ in r]  # cpu-route: Python iterables of labels, the explicit label input step
        pr = _Prog()
        if not flat:
            out = pr.alloc(n * max(K, 1))
            pr.run(self.numeric_mode_)
            return pr.get_i32(out, (n, K))
        if self._cats is None or _numeric_labels(flat) is None:
            index = {c: i for i, c in enumerate(self._classes)}  # cpu-route: str or object labels, the explicit label input step
            codes = pr.put_list([index.get(v, -1) for v in flat])  # cpu-route: Python iterables of labels, the explicit label input step
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
        return [tuple(self._classes[j] for j, v in enumerate(r) if v) for r in rows]  # cpu-route: builds the user list of Python label tuples (object output)


# ---------------------------------------------------------------- iterative imputer
class _NeighbourDraws:
    """IterativeImputer's n_nearest_features draws, one list per (round,
    feature) step in fit order. Every step is drawn up front in one device
    program (`_neighbours_device`, every tier: lane cpu2-l3-prep deleted the
    host route that read the |corr| matrix back and called the base binding's
    `weighted_pick_i32` per step); `next` then hands them out and moves the
    instance's splitmix64 state past the words the steps taken so far own
    (k per step: state + k * golden each)."""

    def __init__(self, imp, Xf, n, dk, mode, orders):
        self.imp, self.dk, self.at = imp, dk, 0
        js = [j for order in orders for j in order]  # glue: the (round, feature) step list
        self.k, self.start = int(imp.n_nearest_features), imp._rng
        self.lists = imp._neighbours_device(Xf, n, dk, mode, js, self.k) if js else []

    def next(self, j):
        out = self.lists[self.at]
        self.at += 1
        self.imp._rng = (self.start + self.at * self.k * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        return out


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
            return [float(x) for x in vals]  # glue: converts the min_value / max_value arguments (vals-sized: per-column bound arguments)
        lo, hi = per(self.min_value), per(self.max_value)
        return [v for pair in zip(lo, hi) for v in pair]  # glue: interleaves the bound arguments (lo-sized: per-column bound arguments)

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

    def _neighbours_device(self, Xf, n, dk, mode, js, k):
        """Lane fam2-prep-metrics (every tier since lane cpu2-l3-prep): every
        (round, feature) step's n_nearest_features draw in ONE program: the
        |corr| matrix (ii_mean, ii_gram, p2m_abscorr_*) stays on the device and one
        thread per step draws its k features from it (f2_wpick,
        x_prep/fam2.mojo; step c takes splitmix64 words c*k .. c*k + k - 1 of
        the instance's stream). Returns each step's ascending picks."""
        pr = _Prog()
        fo, mz = pr.put(Xf), pr.alloc(n * dk)
        means, cnt, g, flag = pr.alloc(dk), pr.alloc(1), pr.alloc(dk * dk), pr.alloc(1)
        pr.stage("ii_mean", dk, fo, n, dk, mz, 0, means, cnt, flag)
        pr.stage("ii_gram", dk * dk, fo, n, dk, mz, 0, means, g, flag)
        m = pr.alloc(dk * dk)
        pr.stage("p2m_abscorr_cell", dk * dk, g, dk, m)
        pr.stage("p2m_abscorr_norm", dk, m, dk)
        calls = len(js)
        fl, out = pr.work(calls * dk), pr.alloc(calls * (k + 1))
        pr.stage("f2_wpick", calls, m, dk, pr.put_ints(js), k, _seed_words(pr, self._rng), fl, out)
        pr.run(mode)
        words = pr.get_i32(out, calls * (k + 1)).tolist()
        lists = []
        for c in range(calls):  # glue: one feature list per step
            got = words[c * (k + 1) + k]
            if got < k:
                raise ValueError("mojolearn: IterativeImputer: no feature left to draw")
            lists.append(words[c * (k + 1):c * (k + 1) + got])
        return lists

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
        self._bounds_k = [bounds[2 * c + h] for c in self._keep for h in (0, 1)]  # glue: per-kept-column bound arguments (_keep-sized: kept feature indices)
        # missing counts per kept column, and the tolerance scale, from the device
        pr = _Prog()
        xo = _mark_missing(pr, pr.put(arr), n * d, self.missing_values)
        st = pr.alloc(6 * d)
        _cs(pr, mode, xo, n, d, st)
        rounds = int(self.max_iter)
        # lane cpu2-l3-prep: the missing counts (c2_ii_miss), each round's imputation
        # order (c2_ii_pos + c2_ii_ord, or c2_ii_rand: the reference's Fisher-Yates on the
        # instance's splitmix64 stream, one round a thread) and the tolerance scale
        # (c2_colmax) on the device; the orders come back as the fit's control lists
        R = max(rounds, 1)
        ordw, lens, sc = pr.alloc(max(R * dk, 1)), pr.alloc(R), pr.alloc(1)
        pr.stage("c2_colmax", 1, st + 5 * d, d, sc)
        if dk:
            missd = pr.work(dk)
            pr.stage("c2_ii_miss", dk, st, pr.put_ints(self._keep), n, missd)
            if self.imputation_order == "random":
                pr.stage("c2_ii_rand", R, missd, dk, 1 if self.skip_complete else 0, _seed_words(pr, self._rng),
                         ordw, lens)
            else:
                pos = pr.work(dk)
                mcode = {"ascending": 0, "descending": 1, "roman": 2, "arabic": 3}[self.imputation_order]
                pr.stage("c2_ii_pos", dk, missd, dk, mcode, pos)
                pr.stage("c2_ii_ord", dk, missd, dk, pos, R, ordw, lens)
        pr.run(mode)
        scale = pr.values(sc, 1)[0]
        self._indicator = [j for j, c in enumerate(pr.values(st, d)) if int(c) < n] if self.add_indicator else []  # glue: indicator column index list from device counts (d-sized: feature count)
        ln = pr.get_i32(lens, R).tolist() if dk else [0] * R
        ow = pr.get_i32(ordw, R * dk).tolist() if dk else []
        orders = [ow[r * dk:r * dk + ln[r]] for r in range(rounds)]  # glue: the device's orders as control lists
        self.n_features_with_missing_ = ln[0]
        if self.imputation_order == "random":
            # the draws the rounds took: (m - 1) a round over m candidates
            m = ln[0] if self.skip_complete else dk
            self._rng = (self._rng + rounds * max(m - 1, 0) * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        self.numeric_mode_, self.n_features_in_ = mode, d
        nnf = self.n_nearest_features
        corr = _NeighbourDraws(self, Xf, n, dk, mode, orders) if nnf is not None and nnf < dk else None
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
        for r in range(rounds):  # glue: drives the device imputation rounds (rounds-sized: imputation rounds)
            if orders[r] and conv:
                pr.stage("ii_snapshot", n * dk, fo, prev, flag)
            for j in orders[r]:  # glue: drives one device regression per feature (orders-sized: per-round feature orders)
                coef, inter, means = pr.alloc(dk), pr.alloc(1), pr.alloc(dk)
                nbl = corr.next(j) if corr is not None else None
                nb1 = pr.put_list([1 if a in nbl else 0 for a in range(dk)]) + 1 if nbl is not None else 0  # glue: neighbor-feature mask of one step (dk-sized: kept feature count)
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
        steps = sum(len(o) for o in orders[:done])  # glue: count of finished imputation steps (orders-sized: per-round feature orders)
        self.imputation_sequence_ = [(j, pr.get(c, dk), pr.get(i, 1)) for j, c, i in seq[:steps]]  # glue: fitted imputation sequence views (seq-sized: finished imputation steps)
        self._posterior = [(nbl, pr.get(sg, max(pp, 1) ** 2), pr.get(al, 2), pr.get(mn, dk))
                           for nbl, sg, al, mn, pp in extra[:steps]]  # glue: fitted imputation sequence views (extra-sized: finished imputation steps)
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
        for r, order in enumerate(orders):  # cpu-route: drives the user estimator per feature
            check = not self.sample_posterior and bool(order)
            prev = Xt.copy() if check else None
            for j in order:  # cpu-route: drives the user estimator per feature
                nbl = corr.next(j) if corr is not None else [a for a in range(dk) if a != j]  # glue: the other feature ids
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
        X = pr.get(xo, (m, nc)) if nc else Array((m, 0), "<f4")
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
            v = self._truncnorm_draws(mus, sig, lo, hi)
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

    def _truncnorm_draws(self, mus, sig, lo, hi):
        """`_impute_one_feature`'s rule per entry, in Mojo (the core helper
        `truncnorm_draws`, bindings/normal_dist_helpers.mojo; lane py-runtime
        round 3, it was Python float64 per entry): mu beyond a bound -> the
        bound, sigma <= 0 -> mu, else the inversion of the truncated normal
        at a 53-bit splitmix64 uniform on the pinned normal cdf / inverse cdf
        (DEVIATION 6902), the instance's stream advanced by the entries that
        draw. The reference's zip stops at the shorter of the two outputs."""
        from ._buffer import _native
        mu = array.array("d", mus.tolist() if hasattr(mus, "tolist") else mus)  # cpu-route: the user estimator's predict output, the explicit input step
        sd = array.array("d", sig.tolist() if hasattr(sig, "tolist") else sig)  # cpu-route: the user estimator's predict output, the explicit input step
        k = min(len(mu), len(sd))
        out = array.array("d", bytes(8 * k))
        st = int(self._rng)
        hi_lo = _native("truncnorm_draws")(mu.buffer_info()[0], sd.buffer_info()[0], k, [float(lo), float(hi)],
                                           [st >> 32, st & 0xFFFFFFFF], out.buffer_info()[0])
        self._rng = (int(hi_lo[0]) << 32) | int(hi_lo[1])
        return Array._owned(out, (k,), "<f8", "C")

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
            for j, nbl, est in self.imputation_sequence_:  # glue: drives one device regression per step (imputation_sequence_-sized: fitted imputation steps)
                _, _, Xm, rows, m = self._ii_take(Xt, mask, j, nbl, mode, fit=False)
                Xt = self._ii_put(Xt, Xm, rows, m, j, est, mode)
            return self._with_indicator(arr, Xt)
        pr = _Prog()
        fo, mo, bo = self._prepare(pr, arr, Xf, inout=True)
        seed = self._rng & 0x7FFFFFFF
        for s, (j, coef, inter) in enumerate(self.imputation_sequence_):  # glue: drives one device regression per step (imputation_sequence_-sized: fitted imputation steps)
            co, io = pr.put(coef), pr.put(inter)
            if self.sample_posterior:
                nbl, sig, al, means = self._posterior[s]
                nb1 = pr.put_list([1 if a in nbl else 0 for a in range(dk)]) + 1 if nbl is not None else 0  # glue: neighbor-feature mask of one step (dk-sized: kept feature count)
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
    return [float(x) for x in (v.tolist() if hasattr(v, "tolist") else v)]  # cpu-route: a user estimator's predict output as Python floats, the explicit input step


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


def _f64_words(pr, value):
    """A Python float as its binary64 word (two int32 words) -> offset."""
    w = array.array("i")
    w.frombytes(array.array("d", [float(value)]).tobytes())
    return pr.put_words(Array._owned(w, (2,), "<i4", "C"))


def _key_words(values, d):
    """(int32 words, c2_key64 KIND) of d values: a float32 / float64 / int32 /
    int64 buffer as its own words; anything else (a user score function's
    list) converted once to float64 (the explicit input-prep step)."""
    lb = _label_buffer(values)
    if lb is not None and lb.kind in (0, 1, 3, 4) and lb.n == d:
        return lb.words, lb.kind
    vals = values.tolist() if hasattr(values, "tolist") else values
    flat = array.array("d", [float(v) for v in vals])  # glue: a user callable's scores, converted once
    if len(flat) != d:
        raise ValueError(f"mojolearn: expected {d} scores, got {len(flat)}")
    w = array.array("i")
    w.frombytes(flat.tobytes())
    return Array._owned(w, (2 * d,), "<i4", "C"), 4


def _mask_list(pr, off, d):
    """A device 0.0 / 1.0 mask read back as the fitted bool list."""
    return [v != 0.0 for v in pr.values(off, d)]  # glue: the device's mask as the fitted list


# ---------------------------------------------------------------- feature selection
class _SelectorMixin(_PrepBase):
    def get_support(self, indices=False):
        self._check_fitted()
        mask = list(self._mask)
        return [j for j, m in enumerate(mask) if m] if indices else mask  # glue: support index list (mask-sized: feature support mask)

    def transform(self, X):
        self._check_fitted()
        arr = _x2d(X)
        self._check_width(arr)
        n, d = arr.shape
        keep = [j for j, m in enumerate(self._mask) if m]  # glue: kept column index list (_mask-sized: feature support mask)
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
        _cs(pr, mode, xo, n, d, st)
        pr.stage("var_ptp", d, st, d, var, 1 if self.threshold == 0 else 0)
        # lane cpu2-l3-prep: variance > threshold on the device, in binary64 (c2_key64, c2_gt)
        key, mask = pr.work(2 * d), pr.alloc(d)
        pr.stage("c2_key64", d, var, 0, key)
        pr.stage("c2_gt", d, key, _f64_words(pr, self.threshold), mask)
        pr.run(mode)
        self.variances_ = pr.get(var, d)
        self._mask = _mask_list(pr, mask, d)
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
    mode = _mode()
    _cls(pr, mode, xo, n, d, yo, K, cnt, mean, _NONE, sums)
    if kind == "chi2":
        _cs(pr, mode, xo, n, d, st, var=False)
        pr.stage("chi2", d, sums, K, d, cnt, n, sc, pv)
    elif _blocked() and _idn_fam(mode) & _IDN_SELECT_BLOCKED:
        # x_prep/select_blocked.mojo: the within-class squares by row blocks, then the score
        nb = (n + _XB - 1) // _XB
        ps = pr.work(nb * d)
        pr.stage("fcb_part", nb * d, xo, n, d, yo, mean, ps, nb)
        pr.stage("fcb_fin", d, ps, n, d, nb, K, cnt, mean, sc, pv)
    else:
        pr.stage("f_classif", d, xo, n, d, yo, K, cnt, mean, sc, pv)
    pr.run(mode)
    if kind == "chi2" and any(v < 0 for v in pr.values(st + 3 * d, d)):  # glue: raises on a negative per-column minimum (d-sized: feature count)
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
    mode = _mode()
    if _blocked() and _idn_fam(mode) & _IDN_SELECT_BLOCKED:
        # x_prep/select_blocked.mojo: the sums and the centred products by row blocks
        nb = (n + _XB - 1) // _XB
        mx = my = _NONE
        if center:
            psx, psy, mx, my = pr.work(nb * d), pr.work(nb), pr.work(d), pr.work(1)
            pr.stage("frb_part1", nb * d, xo, n, d, yo, psx, psy, nb)
            pr.stage("frb_mean", d, psx, psy, nb, d, n, mx, my)
        pxy, pxx, pyy = pr.work(nb * d), pr.work(nb * d), pr.work(nb)
        pr.stage("frb_part2", nb * d, xo, n, d, yo, mx, my, pxy, pxx, pyy, nb)
        pr.stage("frb_fin", d, pxy, pxx, pyy, nb, d, n, 1 if center else 0, sc, pv, co, 1 if force_finite else 0)
    else:
        pr.stage("f_regression", d, xo, n, d, yo, 1 if center else 0, sc, pv, co, 1 if force_finite else 0)
    pr.run(mode)
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
        if self.k == "all":
            self._mask = [True] * d
        else:
            k = int(self.k)
            if not 0 <= k <= d:
                raise ValueError(f"mojolearn: k should be 0 <= k <= n_features = {d}; got {k}")
            # lane cpu2-l3-prep: the top k on the device (c2_key64: binary64 ordered keys, a
            # NaN as -inf; c2_topk: stable ascending rank, the later of tied columns wins)
            pr = _Prog()
            key, mask = pr.work(2 * d), pr.alloc(d)
            words, kind = _key_words(scores, d)
            pr.stage("c2_key64", d, pr.put_words(words), kind, key)
            pr.stage("c2_topk", d, key, d, k, mask)
            pr.run(_mode())
            self._mask = _mask_list(pr, mask, d)
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
    if vals and all(isinstance(v, bool) for v in vals):  # glue: validates the discrete_features argument (vals-sized: user discrete_features argument)
        if len(vals) != d:
            raise ValueError(f"mojolearn: discrete_features mask has {len(vals)} entries; X has {d} features")
        return list(vals)
    mask = [False] * d
    for v in vals:  # glue: converts the discrete_features argument (vals-sized: user discrete_features argument)
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
    cont = [j for j in range(d) if not mask[j]]  # glue: continuous column index list (d-sized: feature count)
    disc = [j for j in range(d) if mask[j]]  # glue: discrete column index list (d-sized: feature count)
    mode = _mode()
    seed = 0 if random_state is None else int(random_state) & 0x3FFFFFFF
    pr = _Prog()
    work = pr.work if (_MI_WORK and mode == "fast") else pr.alloc
    if discrete_target:
        classes, codes = encode_labels(y)
        if codes.size != n:
            raise ValueError("mojolearn: X and y have different numbers of rows")
        counts = _class_counts(codes, len(classes))
        if cont and max(counts) < 2:  # glue: raises when every class has one sample (counts-sized: class counts from the binding)
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
        _cs(pr, mode, xo, n, dc, st)
        pr.stage("mi_colscale", dc, xo, n, dc, st, sc, ma)
        pr.stage("mi_noise", n * dc, xo, n, dc, sc, ma, 2 * seed, z, _plus1(zs))
    if not discrete_target:
        yo = pr.put(yv)
        sty, scy, may, zy, zys = pr.alloc(6), pr.alloc(1), pr.alloc(1), work(n), work(n)
        _cs(pr, mode, yo, n, 1, sty)
        pr.stage("mi_colscale", 1, yo, n, 1, sty, scy, may)
        pr.stage("mi_noise", n, yo, n, 1, scy, may, 2 * seed + 1, zy, _plus1(zys))
    if cont:
        term, outc = work(n * dc), pr.alloc(dc)
        if discrete_target:
            pr.stage("mi_cd", n * dc, z, n, dc, yo, lc, k, term, _plus1(zs))
            # the rows of the classes with more than one (KIND 3: summed on the device)
            pr.stage("mi_reduce", dc, term, n, dc, 3, k, 0, outc, lc, len(counts))
        else:
            pr.stage("mi_cc", n * dc, z, n, dc, zy, k, term, _plus1(zs), _plus1(zys))
            pr.stage("mi_reduce", dc, term, n, dc, 0, k, n, outc)
    if disc:
        dd = len(disc)
        xd = _gather(arr, disc, mode)
        cats = _fit_categories(mode, xd)
        kx = [c.size for c in cats]  # glue: per-column category counts (cats-sized: per-column category arrays)
        kmax = max(kx)  # glue: largest category count for the program layout (kx-sized: per-column category counts)
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
    for j, v in zip(cont, pr.values(outc, len(cont)) if cont else []):  # glue: scatters device MI values to their columns (cont-sized: continuous column indices)
        vals[j] = v
    for j, v in zip(disc, pr.values(outd, len(disc))):  # glue: scatters device MI values to their columns (disc-sized: discrete column indices)
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


def _stage_importance_keys(pr, est, m, getter="auto"):
    """Stages the ordered keys (c2_key64) of a fitted estimator's m column
    importances: coef_ squared (summed over rows when 2-D, sqsum_cols) on the
    device, else feature_importances_ as given (a monotone stand-in for its
    square). A str getter (a dotted attribute path, as operator.attrgetter)
    or a callable picks the importances instead; they are squared (summed
    over rows when 2-D) on the device, as the reference's
    transform_func='square'. Returns the keys' offset."""
    if getter != "auto":
        coef = operator.attrgetter(getter)(est) if isinstance(getter, str) else getter(est)
    else:
        coef = getattr(est, "coef_", None)
    key = pr.work(2 * m)
    if coef is None and getter == "auto":
        imp = getattr(est, "feature_importances_", None)
        if imp is None:
            raise ValueError("mojolearn: RFE needs an estimator with coef_ or feature_importances_")
        words, kind = _key_words(imp, m)
        pr.stage("c2_key64", m, pr.put_words(words), kind, key)
        return key
    c = as_f32_c(coef, ndim=None, name="coef_")[0]
    rows, d = (1, c.shape[0]) if c.ndim == 1 else c.shape
    if d != m:
        raise ValueError(f"mojolearn: RFE importances have {d} columns, expected {m}")
    out = pr.work(d)
    pr.stage("sqsum_cols", d, pr.put(c), rows, d, out)
    pr.stage("c2_key64", d, out, 0, key)
    return key


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
        ranking = full((d,), 1, "<f4")
        nsup = d
        while nsup > nsel:
            features = [j for j in range(d) if support[j]]  # glue: the supported column list (control)
            est = self._clone().fit(_gather(arr, features, mode), y, **fit_params)
            # lane cpu2-l3-prep: the ranking and elimination on the device (c2_rfe_step: the
            # `step` weakest by a stable ascending sort of the importance keys leave the
            # support; c2_rfe_rank: every unsupported column's ranking + 1)
            pr = _Prog()
            key = _stage_importance_keys(pr, est, len(features), self.importance_getter)
            sup = pr.put_list([1.0 if v else 0.0 for v in support], inout=True)  # glue: the support mask up
            rk = pr.put(ranking, inout=True)
            pr.stage("c2_rfe_step", len(features), key, pr.put_ints(features), len(features),
                     min(step, nsup - nsel), sup)
            pr.stage("c2_rfe_rank", d, sup, rk)
            pr.run(mode)
            support = _mask_list(pr, sup, d)
            ranking = pr.get(rk, d)
            nsup = len([1 for v in support if v])  # glue: the support size (control)
        features = [j for j in range(d) if support[j]]  # glue: support index list (d-sized: feature count)
        self.estimator_ = self._clone().fit(_gather(arr, features, mode), y, **fit_params)
        self._mask, self.support_ = support, list(support)
        self.ranking_ = ranking.astype("<i8")
        self.n_features_ = sum(support)  # glue: count of selected features (support-sized: feature support mask)
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
    _csr_ok = True
    _needs_min = True

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

    def _csr_bias(self):
        return self.class_log_prior_ if len(self.classes_) == 1 else None


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
        first, codes = _partial_codes(self, y, classes, arr.shape[0], defer=True)
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
        xo, yo = pr.put(arr), _stage_partial_codes(pr, codes, mode)
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
        if any(v < 0 for v in lo):  # glue: raises on a negative category value (lo-sized: per-column minimum statistics)
            raise ValueError("mojolearn: Negative values in data passed to CategoricalNB (input X)")
        ncat = [int(v) + 1 for v in hi]  # glue: per-column category counts for the program layout (hi-sized: per-column maximum statistics)
        if self.min_categories is not None:
            mc = self.min_categories
            mcs = ([int(v) for v in (mc.tolist() if hasattr(mc, "tolist") else mc)]  # glue: converts the min_categories argument (mc-sized: min_categories argument)
                   if not isinstance(mc, numbers.Integral) else [int(mc)] * d)
            if len(mcs) != d:
                raise ValueError(f"mojolearn: 'min_categories' should have shape ({d},) when an array-like "
                                 f"is provided. Got {len(mcs)} entries instead.")
            ncat = [max(a, b) for a, b in zip(ncat, mcs)]  # glue: applies the min_categories argument (ncat-sized: per-column category counts)
        if merge:
            ncat = [max(a, b) for a, b in zip(ncat, self.n_categories_.tolist())]  # glue: keeps fitted category counts on partial fit (ncat-sized: per-column category counts)
        cmax = max(ncat)  # glue: largest category count for the program layout (ncat-sized: per-column category counts)
        q = _Prog()
        xo, yo = q.put(arr), _stage_partial_codes(q, codes, mode)
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
            # the running counts re-strided to cmax on the device (colblock into
            # zeroed words)
            oc = self._cmax
            pad, src, cc = q.alloc(d * K * cmax), cc, q.alloc(d * K * cmax)
            if oc > 0:
                q.stage("colblock", d * K * oc, q.put(self._cc), oc, pad, cmax, 0)
            q.stage("add_arrays", d * K * cmax, pad, src, cc)
        flp = q.alloc(d * K * cmax)
        q.stage("cat_flp", d * K * cmax, cc, K, no, cmax, co, a, flp)
        q.run(mode)
        self._cc = q.get(cc, d * K * cmax)
        # each (feature, class) row read once (the same values)
        ccv, flv = q.values(cc, d * K * cmax), q.values(flp, d * K * cmax)
        self.category_count_ = [Array.from_list(
            [ccv[(j * K + k) * cmax:(j * K + k) * cmax + ncat[j]] for k in range(K)], "<f4") for j in range(d)]  # glue: per-column per-class count views (d-sized: feature count)
        self.n_categories_ = Array.from_list(ncat, "<i8")
        self._flp, self._cmax = q.get(flp, d * K * cmax), cmax
        self.feature_log_prob_ = [Array.from_list(
            [flv[(j * K + k) * cmax:(j * K + k) * cmax + ncat[j]] for k in range(K)], "<f4") for j in range(d)]  # glue: per-column per-class log-prob views (d-sized: feature count)
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
        if (any(v < 0 for v in pr.values(st + 3 * d, d)) or  # glue: raises on a negative category value (d-sized: feature count)
                any(int(v) >= c for v, c in zip(pr.values(st + 4 * d, d), ncat))):  # glue: raises on an unseen category value (ncat-sized: per-column category counts)
            raise IndexError("mojolearn: CategoricalNB got a category index outside the fitted range")

    def _jll_stages(self, pr, xo, n, d, out):
        K = len(self.classes_)
        fo, co = pr.put(self._flp), pr.put(self.class_log_prior_)
        pr.stage("cat_jll", n * K, xo, n, d, fo, K, self._cmax, co, out)



def _classical_shared(mode):
    """Compile-time admission only; no data processing or Python candidate arithmetic."""
    if mode != "identical":
        return 0
    entry = _optional_prep_entry(_prep_binding(mode), "x_prep_classical_shared")
    return int(entry()) if entry is not None else 0



def _fit_categories_with_codes(mode, arr):
    """C08 canonical categories and inverse from one owned native program."""
    n, d = arr.shape
    pr = _Prog()
    X, sorted_rows = pr.put(arr), pr.work(n*d)
    dictionary, counts, inverse = pr.alloc(n*d), pr.alloc(d), pr.alloc(n*d)
    pr.stage("sort_cols", d, X, n, d, sorted_rows, 1)
    pr.stage("unique_inverse", d, sorted_rows, n, d, dictionary, counts, X, inverse)
    pr.run(mode)
    sizes = [int(v) for v in pr.values(counts, d)]  # glue: per-feature output extents
    categories = [pr.get(dictionary+c*n, sizes[c]) for c in range(d)]  # glue: wraps dictionaries
    return categories, pr.get(inverse, (n, d))
