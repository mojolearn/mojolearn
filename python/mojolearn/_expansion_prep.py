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
import numbers
import operator
import os

from . import _backend
from . import _portable_math as _pm
from ._array import Array
from ._buffer import as_f32_c, as_i32_c, addr_ro, addr as _addr_rw, _as_typed, empty, full
from . import _labels
from . import _arena_io
from ._labels import flatten_labels, sorted_classes, label_kind
from ._mode import SklearnTagsMixin, available_if
from ._native_robust_bridge import native_method as _native_robust_method

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
    # TOMBSTONE: MOJOLEARN_IDN_LAB_UNIQUE_HIST, MOJOLEARN_IDN_LAB_CODES_DIRECT, MOJOLEARN_IDN_LAB_ONE_ENTRY (losers: label-encoder 2.1-2.6x, label-binarizer istella 1.15-1.39x slower, NV + AMD) deleted 2026-10-11 by de9ce7b39; code recoverable at de9ce7b39^.
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_LAB_UNIQUE_HIST_CODES_DIRECT_ONE_ENTRY.patch
    # lane apple-fast-q-clf (x_prep/proba64.mojo, in the py2mojo range): staged only when the
    # binding exports x_prep_proba64 (FAST, not -D MOJOLEARN_PROBA64_QOLD)
    q64_softmax=213,
)
_PARAMS = 14
_NONE = -1


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
#: the smallest read that is a view (the smallest mapped block, 2**18 words,
#: is bindings/arena_prog_native.mojo PREP_MAP_MIN_WORDS)
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
#: _DIRECT_MIN_WORDS (2**20 words) is bindings/arena_prog_native.mojo PREP_DIRECT_MIN_WORDS
#: x_prep/blocked.mojo XB: the rows a block unit folds
_XB = 2048
#: CategoricalNB's block histograms hold at most this many words; past it
#: the histogram block doubles (a function of the shape only)


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
#: PT_COLBATCH, 8 SI_ONEPASS, 32 PT_SCORE_STABLE),
#: probed once per binding; 0 on a build without them (every other tier and
#: vendor, and the host). The device's own comptime switches do the fusing;
#: a program only shrinks the arena blocks the device no longer touches.
_PTIMPUTE_FLAGS = {}
# TOMBSTONE: MOJOLEARN_PT_SPEC (DROP: +10% vs COLBATCH) deleted 2026-10-09 on lane/owed-deletions-D2: flag bit 2 and
# _PT_FAST_SPEC; code recoverable at b639a2bd2. Restore: git apply experiments/removed/MOJOLEARN_PT_SPEC.patch


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
        """Run the program on the x_prep binding (GPU, or the host binding on
        a CPU-only install): one native call that owns the arena layout and
        stores, the input copies, resident, direct and device-gathered
        inputs, the range descriptors and the run
        (bindings/arena_prog_native.mojo). The program's refusals
        (`_after`) run before its first result is read."""
        return _PROG_CALL(self, "x_prep_run", (mode,), _PROG_CONTEXT)

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


def _mode():
    return _backend.default_mode()
# TOMBSTONE: MOJOLEARN_X_PREP_FAST_NONEG (DROPPED-noise) deleted 2026-10-03 by 10a9ab8eb; code recoverable at 10a9ab8eb^.
# Restore: git apply experiments/removed/MOJOLEARN_X_PREP_FAST_NONEG.patch; record in docs/TOMBSTONES.md.


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


class _PrepBase(SklearnTagsMixin):
    """sklearn's parameter protocol, tags and fit_transform for the lane's classes."""
    _parameters = ()

    def get_params(self, deep=True):
        """Return the constructor parameters as a dict.

        Parameters
        ----------
        deep : bool, default True
            Accepted for scikit-learn compatibility.

        Returns
        -------
        dict
            Parameter names mapped to their values.
        """
        return {name: getattr(self, name) for name in self._parameters}  # glue: copies estimator keyword arguments

    def set_params(self, **params):
        """Set constructor parameters.

        An unknown parameter name raises ``ValueError``.

        Returns
        -------
        self
        """
        for k, v in params.items():  # glue: copies estimator keyword arguments
            if k not in self._parameters:
                raise ValueError(f"mojolearn: invalid parameter {k!r} for {type(self).__name__}")
            setattr(self, k, v)
        return self

    # Only a class with a `transform` has `fit_transform` (scikit-learn's
    # TransformerMixin): a classifier or detector on this base has neither.
    @available_if(lambda self: hasattr(self, "transform"))
    def fit_transform(self, X, y=None, **fit_params):
        """Fit to ``X``, then transform ``X``.

        Returns
        -------
        array
            The transformed data.
        """
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
# A scale below 10 * float32 eps is one (sklearn's `_handle_zeros_in_scale`);
# unit_variance divides the scale by norm.ppf(q_max) - norm.ppf(q_min), with
# norm.ppf by Acklam's approximation in float32.
class RobustScaler(_PrepBase):
    """Scale features by the median and the interquantile range.

    Follows scikit-learn's ``RobustScaler``. Each column is centered by its median and divided by the
    range between the ``quantile_range`` percentiles, computed with numpy's linear percentile over the
    non-NaN entries. NaN values are ignored during fit and kept as NaN by transform. A scale smaller than
    ten float32 epsilons is replaced by one. Input and output are float32.

    In `identical` mode, mojolearn returns the same bits on Apple, NVIDIA and AMD GPUs.

    Parameters
    ----------
    with_centering : bool, default True
        Subtract the median before scaling.
    with_scaling : bool, default True
        Divide by the interquantile range.
    quantile_range : tuple of float, default (25.0, 75.0)
        ``(q_min, q_max)`` percentiles, with ``0 <= q_min <= q_max <= 100``; otherwise fit raises
        ``ValueError``.
    copy : bool, default True
        Accepted for scikit-learn compatibility; transform always returns a new array.
    unit_variance : bool, default False
        Also divide the scale by ``norm.ppf(q_max) - norm.ppf(q_min)``, so that normally distributed
        data ends with unit variance. Requires ``0 < q_min < q_max < 100`` (``ValueError`` otherwise).

    Attributes
    ----------
    center_ : array of shape (n_features,) or None
        Per-column median, or None when ``with_centering=False``.
    scale_ : array of shape (n_features,) or None
        Per-column scale, or None when ``with_scaling=False``.
    ``n_features_in_`` : int
        Number of columns seen in fit.
    numeric_mode_ : str
        Numeric mode the estimator was fitted in; transform runs in the same mode.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> Xt = mojolearn.RobustScaler().fit_transform(X)
    """
    # scikit-learn tags, read by `_mode.sklearn_tags`
    _sklearn_input_tags = {"allow_nan": True}
    _parameters = ("with_centering", "with_scaling", "quantile_range", "copy", "unit_variance")

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _register_robust_scaler(cls, _ROBUST_CONTEXT)

    def __init__(self, *, with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), copy=True,
                 unit_variance=False):
        return _ROBUST_CALL(self, "init", (with_centering, with_scaling, quantile_range, copy, unit_variance), _ROBUST_CONTEXT)

    def get_params(self, deep=True):
        """Return this scaler's constructor parameters as a dictionary."""
        return _ROBUST_CALL(self, "get_params", (deep,), _ROBUST_CONTEXT)

    def set_params(self, **params):
        """Set constructor parameters and return this scaler."""
        return _ROBUST_CALL(self, "set_params", (params,), _ROBUST_CONTEXT)

    def __sklearn_tags__(self):
        return _ROBUST_CALL(self, "tags", (), _ROBUST_CONTEXT)

    def __sklearn_is_fitted__(self):
        return _ROBUST_CALL(self, "is_fitted", (), _ROBUST_CONTEXT)

    @_native_robust_method
    def fit(self, X, y=None):
        """Learn the feature medians and interquantile scales from X; return self."""
        return _ROBUST_CALL(self, "fit", (X, y), _ROBUST_CONTEXT)

    @_native_robust_method
    def fit_transform(self, X, y=None, **fit_params):
        """Fit on X and return a new float32 array of scaled features."""
        return _ROBUST_CALL(self, "fit_transform", (X, y, fit_params), _ROBUST_CONTEXT)

    @_native_robust_method
    def transform(self, X):
        """Return X centered and scaled by the learned feature statistics."""
        return _ROBUST_CALL(self, "transform", (X,), _ROBUST_CONTEXT)


class MaxAbsScaler(_PrepBase):
    """Scale each feature by its maximum absolute value.

    Follows scikit-learn's ``MaxAbsScaler``: each column is divided by the largest absolute value among
    its non-NaN entries, so the result lies in ``[-1, 1]``. NaN values are kept as NaN by transform. A
    column of zeros is scaled by one. Input and output are float32.

    In `identical` mode, mojolearn returns the same bits on Apple, NVIDIA and AMD GPUs.

    Parameters
    ----------
    copy : bool, default True
        Accepted for scikit-learn compatibility; transform always returns a new array.

    Attributes
    ----------
    max_abs_ : array of shape (n_features,)
        Per-column maximum absolute value.
    scale_ : array of shape (n_features,)
        Per-column divisor (``max_abs_``, with zeros replaced by one).
    ``n_features_in_`` : int
        Number of columns seen in fit.
    n_samples_seen_ : int
        Number of rows seen in fit.
    numeric_mode_ : str
        Numeric mode the estimator was fitted in; transform runs in the same mode.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).standard_normal((1000, 8)).astype(np.float32)
    >>> Xt = mojolearn.MaxAbsScaler().fit_transform(X)
    """
    # scikit-learn tags, read by `_mode.sklearn_tags`
    _sklearn_input_tags = {"allow_nan": True}
    _parameters = ("copy",)

    _mojolearn_native_public = True

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _register_maxabs_scaler(cls, _MAXABS_CONTEXT)

    def __init__(self, *, copy=True):
        return _MAXABS_CALL(self, "init", (copy,), _MAXABS_CONTEXT)

    def get_params(self, deep=True):
        """Return this scaler's constructor parameters as a dictionary."""
        return _MAXABS_CALL(self, "get_params", (deep,), _MAXABS_CONTEXT)

    def set_params(self, **params):
        """Set constructor parameters and return this scaler."""
        return _MAXABS_CALL(self, "set_params", (params,), _MAXABS_CONTEXT)

    def __sklearn_tags__(self):
        return _MAXABS_CALL(self, "tags", (), _MAXABS_CONTEXT)

    def __sklearn_is_fitted__(self):
        return _MAXABS_CALL(self, "is_fitted", (), _MAXABS_CONTEXT)

    def fit(self, X, y=None):
        """Learn per-feature absolute maxima from two-dimensional X; return self.

        y is accepted for pipeline compatibility. NaNs in X are ignored when
        learning each feature's maximum.
        """
        return _MAXABS_CALL(self, "fit", (X, y), _MAXABS_CONTEXT)

    def fit_transform(self, X, y=None, **fit_params):
        """Fit on X and return a new float32 array of scaled features."""
        return _MAXABS_CALL(self, "fit_transform", (X, y, fit_params), _MAXABS_CONTEXT)

    def transform(self, X):
        """Return X divided by the learned scales, preserving NaN entries."""
        return _MAXABS_CALL(self, "transform", (X,), _MAXABS_CONTEXT)


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
    """Encode numeric categorical features as integer codes.

    Follows scikit-learn's ``OrdinalEncoder`` for numeric columns. Each value is replaced by its index
    among its column's sorted distinct training values, returned as float32. NaN is a category: when a
    column holds NaN in fit, NaN sorts last and is written as ``encoded_missing_value``. Columns of
    strings or other objects are not supported.

    Parameters
    ----------
    categories : 'auto' or list of array-like, default 'auto'
        ``'auto'`` takes the categories from the training data. Otherwise one list per column, numeric,
        sorted ascending, without duplicates in float32 and with NaN (if present) last; anything else
        raises ``ValueError``.
    dtype : default None
        Accepted for scikit-learn compatibility; the output is always float32.
    handle_unknown : {'error', 'use_encoded_value'}, default 'error'
        ``'error'`` raises ``ValueError`` when transform (or fit, with given ``categories``) meets a
        value outside the categories. ``'use_encoded_value'`` writes ``unknown_value`` instead.
    unknown_value : float or None, default None
        Code for unknown values under ``handle_unknown='use_encoded_value'``; must be a number (NaN is
        allowed) and must not equal a code already used by a category (``ValueError``).
    encoded_missing_value : float, default NaN
        Code written for NaN. Must be a number and must not equal a code used by a known category.
    min_frequency : int or float, default None
        Categories seen fewer times than this (an int ``>= 1``, or a fraction of the rows in
        ``(0, 1)``) are grouped into one infrequent code placed after the frequent ones. The NaN
        category is never grouped.
    max_categories : int, default None
        Upper limit on output codes per column, counting the infrequent code; must be ``>= 1``.

    Attributes
    ----------
    categories_ : list of arrays
        Sorted categories of each column, float32, NaN last when present.
    ``n_features_in_`` : int
        Number of columns seen in fit.
    numeric_mode_ : str
        Numeric mode the estimator was fitted in; transform runs in the same mode.

    Notes
    -----
    ``inverse_transform`` maps codes back to categories. Where scikit-learn returns ``None`` (an unknown
    value) or the string ``'infrequent_sklearn'`` (the infrequent code), the result here is NaN, since
    the output is a float array. A code that names no category raises ``ValueError``. Unlike
    scikit-learn, ``unknown_value`` may not equal the infrequent code.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.array([[3.0, 10.0], [1.0, 20.0], [3.0, 10.0]], dtype=np.float32)
    >>> mojolearn.OrdinalEncoder().fit_transform(X)
    """
    # scikit-learn tags, read by `_mode.sklearn_tags`
    _sklearn_input_tags = {"allow_nan": True, "categorical": True}
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
        if not _c08_route(_mode(), 4) or not _is_auto(self.categories):
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
    """Encode numeric categorical features as one-hot columns.

    Follows scikit-learn's ``OneHotEncoder`` for numeric columns. Each column becomes a block of 0/1
    float32 columns, one per category (minus a dropped one, see ``drop``). The output is always a
    dense float32 array, whatever ``sparse_output`` says. NaN is a category and sorts last. Columns of
    strings or other objects are not supported.

    Parameters
    ----------
    categories : 'auto' or list of array-like, default 'auto'
        ``'auto'`` takes the categories from the training data. Otherwise one list per column, numeric,
        sorted ascending, without duplicates in float32 and with NaN (if present) last.
    drop : {None, 'first', 'if_binary'} or array-like, default None
        Category to leave out of each block: none, the first, the first of two-category columns only, or
        one given category per column. Dropping a category that is not in the training data, or one
        grouped as infrequent, raises ``ValueError``.
    sparse_output : bool, default True
        Accepted for scikit-learn compatibility; the output is always dense.
    dtype : default None
        Accepted for scikit-learn compatibility; the output is always float32.
    handle_unknown : {'error', 'ignore', 'infrequent_if_exist', 'warn'}, default 'error'
        ``'error'`` raises ``ValueError`` on a value outside the categories. ``'ignore'`` writes an
        all-zero block. ``'infrequent_if_exist'`` and ``'warn'`` send the value to the column's
        infrequent category when it has one, otherwise an all-zero block; ``'warn'`` (and ``'ignore'`` or
        ``'infrequent_if_exist'`` together with ``drop``) issues a ``UserWarning``.
    min_frequency : int or float, default None
        Categories seen fewer times than this (an int ``>= 1``, or a fraction of the rows in
        ``(0, 1)``) are grouped into one infrequent column, the last of the feature's block.
    max_categories : int, default None
        Upper limit on output columns per feature, counting the infrequent column; must be ``>= 1``.
    feature_name_combiner : str, default 'concat'
        Accepted for scikit-learn compatibility.

    Attributes
    ----------
    categories_ : list of arrays
        Sorted categories of each column, float32, NaN last when present.
    drop_idx_ : list or None
        Index into ``categories_[j]`` of the dropped category of each column (None for no drop).
    ``n_features_in_`` : int
        Number of columns seen in fit.
    numeric_mode_ : str
        Numeric mode the estimator was fitted in; transform runs in the same mode.

    Notes
    -----
    ``inverse_transform`` takes each block's largest entry. An all-zero block decodes to the dropped
    category; with no dropped category it decodes to NaN where scikit-learn would return ``None``
    (under ``handle_unknown='ignore'``, or under ``'infrequent_if_exist'`` or ``'warn'`` for a column
    without an infrequent category), and raises ``ValueError`` otherwise. The infrequent column
    decodes to NaN where scikit-learn returns ``'infrequent_sklearn'``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 0.0]], dtype=np.float32)
    >>> mojolearn.OneHotEncoder(drop="first").fit_transform(X)
    """
    # scikit-learn tags, read by `_mode.sklearn_tags`
    _sklearn_input_tags = {"allow_nan": True, "categorical": True}
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
        if not _c08_route(_mode(), 2) or not _is_auto(self.categories):
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


# The fold shuffle draws from random_state by splitmix64 (sklearn draws numpy's).
# A training value outside a given categories list: sklearn codes it as the first
# category, this class leaves it out of the statistics (x_prep/NOT_IMPLEMENTED.tsv).
class TargetEncoder(_PrepBase):
    """Encode numeric categorical features by the target mean of each category.

    Follows scikit-learn's ``TargetEncoder`` for numeric category columns. Each category is replaced by a
    shrunk mean of the target over the rows in that category. Binary, continuous and multiclass targets
    are supported; a multiclass target produces one output column per class for each feature.
    Computation is float32.

    ``fit_transform`` cross-fits: each row is encoded with statistics learned from the other folds, as in
    scikit-learn. ``transform`` (and ``fit`` followed by ``transform``) uses statistics from all the
    training rows, so ``fit(X, y).transform(X)`` differs from ``fit_transform(X, y)``.

    Parameters
    ----------
    categories : 'auto' or list of array-like, default 'auto'
        ``'auto'`` takes the categories from the training data. Otherwise one list per column, numeric and
        sorted ascending. A training value outside a given list is left out of every category's
        statistics but still counts toward the target mean; scikit-learn codes it as the first category
        instead.
    target_type : {'auto', 'binary', 'continuous', 'multiclass'}, default 'auto'
        Type of the target; ``'auto'`` infers it from ``y``.
    smooth : 'auto' or float, default 'auto'
        Shrinkage toward the target mean: ``'auto'`` uses an empirical Bayes estimate, a float ``>= 0``
        sets it directly.
    cv : int, splitter or iterable, default 5
        Folds for ``fit_transform``. An int ``>= 2`` uses KFold for a continuous target and
        StratifiedKFold otherwise. A splitter object (its ``split(X, y)``) or an iterable of
        ``(train, test)`` index pairs is also accepted: the test folds must cover every row exactly once
        (``ValueError`` otherwise), and each fold's training rows must be all the other rows
        (``NotImplementedError`` otherwise).
    shuffle : bool, default True
        Shuffle the rows before splitting into folds.
    random_state : int or None, default None
        Seed for the shuffle. None uses seed 0, so repeated fits give the same folds. The shuffle uses
        mojolearn's own generator, so folds differ from scikit-learn's for the same seed.

    Attributes
    ----------
    categories_ : list of arrays
        Sorted categories of each column, float32.
    encodings_ : list of arrays
        Per-column encoding of each category (times the number of classes for a multiclass target).
    target_type_ : str
        Target type used for the fit.
    target_mean_ : array
        Mean of the target (one value per class for a multiclass target).
    classes_ : array or None
        Class labels for a binary or multiclass target, None for a continuous one.
    ``n_features_in_`` : int
        Number of columns seen in fit.
    numeric_mode_ : str
        Numeric mode the estimator was fitted in; transform runs in the same mode.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.integers(0, 10, (1000, 3)).astype(np.float32)
    >>> y = rng.random(1000).astype(np.float32)
    >>> Xt = mojolearn.TargetEncoder().fit_transform(X, y)
    """
    # scikit-learn tags, read by `_mode.sklearn_tags`
    _sklearn_input_tags = {"allow_nan": True, "categorical": True}
    _sklearn_target_tags = {"required": True}
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
        if _c08_route(mode, 1) and _is_auto(self.categories):
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


# A callable strategy is sklearn's `strategy(masked_X[:, j].compressed())`: it
# runs in Python on the host, over each column's non-NaN float32
# values compacted on the device (p2m_sel_*); the fill runs on the device.
class SimpleImputer(_PrepBase):
    """Replace missing values with a per-column statistic.

    Follows scikit-learn's ``SimpleImputer`` for numeric data. Missing entries (``missing_values``) are
    replaced by each column's mean, median, most frequent value or a constant. Input and output are
    float32.

    A column with no observed values in fit is dropped from the output, unless ``keep_empty_features``
    is set; then it is filled with 0 (or with ``fill_value`` under ``strategy='constant'``).

    Parameters
    ----------
    missing_values : float, default NaN
        Value that marks a missing entry.
    strategy : {'mean', 'median', 'most_frequent', 'constant'} or callable, default 'mean'
        ``'median'`` uses numpy's linear percentile of the non-missing entries; ``'most_frequent'``
        picks the smallest value on a tie. A callable is called once per column, on the host, with the
        column's non-missing values as a 1-D float32 array, and must return the column's statistic; a
        NaN statistic drops the column unless ``keep_empty_features``. Any other value raises
        ``ValueError``.
    fill_value : float or None, default None
        Value used by ``strategy='constant'``; None means 0. Must be numeric (``TypeError`` otherwise).
    copy : bool, default True
        Accepted for scikit-learn compatibility; transform always returns a new array.
    add_indicator : bool, default False
        Append missing-indicator columns (1.0 where missing) for the features that had a missing value
        in fit, as scikit-learn's ``MissingIndicator``.
    keep_empty_features : bool, default False
        Keep columns that had no observed values in fit.

    Attributes
    ----------
    statistics_ : array of shape (n_features,)
        Fill value of each column (NaN for an all-missing column unless ``keep_empty_features``).
    ``n_features_in_`` : int
        Number of columns seen in fit.
    numeric_mode_ : str
        Numeric mode the estimator was fitted in; transform runs in the same mode.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.array([[1.0, np.nan], [3.0, 4.0], [np.nan, 8.0]], dtype=np.float32)
    >>> mojolearn.SimpleImputer(strategy="median").fit_transform(X)
    """
    # scikit-learn tags, read by `_mode.sklearn_tags`
    _sklearn_input_tags = {"allow_nan": True}
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
# Above `subsample` rows the with-replacement resample is drawn from
# random_state by splitmix64 (sklearn draws numpy's). Weighted 'quantile' edges
# follow sklearn's `_weighted_percentile`; with sample_weight above `subsample`
# rows the weights drive the resample and are then spent.
class KBinsDiscretizer(_PrepBase):
    """Bin continuous features into intervals.

    Follows scikit-learn's ``KBinsDiscretizer``. Each column is split into ``n_bins`` intervals and each
    value is replaced by its bin index or a one-hot block. One-hot output is always a dense float32 array.
    A constant column gets a single bin with edges ``(-inf, inf)``.

    Parameters
    ----------
    n_bins : int or array-like of int, default 5
        Number of bins, one value for all columns or one per column; each must be ``>= 2``.
    encode : {'onehot', 'onehot-dense', 'ordinal'}, default 'onehot'
        Output encoding. ``'onehot'`` returns a dense array, the same as ``'onehot-dense'``.
    strategy : {'uniform', 'quantile', 'kmeans'}, default 'quantile'
        ``'uniform'``: equal-width bins. ``'quantile'``: edges at the quantiles given by
        ``quantile_method``. ``'kmeans'``: 1-D k-means started from the uniform bin centers.
    quantile_method : str, default 'averaged_inverted_cdf'
        numpy quantile method for ``strategy='quantile'``: ``'averaged_inverted_cdf'``,
        ``'inverted_cdf'``, ``'closest_observation'``, ``'interpolated_inverted_cdf'``, ``'hazen'``,
        ``'weibull'``, ``'linear'``, ``'median_unbiased'`` or ``'normal_unbiased'``.
    dtype : default None
        Accepted for scikit-learn compatibility; the output is always float32.
    subsample : int or None, default 200000
        Above this many rows, fit uses a resample of ``subsample`` rows drawn with replacement. None
        uses every row.
    random_state : int or None, default None
        Seed for the resample. None uses seed 0. The draw uses mojolearn's own generator, so the rows
        differ from scikit-learn's for the same seed.

    Attributes
    ----------
    bin_edges_ : list of arrays
        Edges of each column's bins.
    n_bins_ : array of int64
        Number of bins of each column (fewer than requested for a constant column).
    ``n_features_in_`` : int
        Number of columns seen in fit.
    numeric_mode_ : str
        Numeric mode the estimator was fitted in; transform runs in the same mode.

    Notes
    -----
    ``fit`` accepts ``sample_weight`` (nonnegative, not all zero). It weights the quantile edges (only
    ``'averaged_inverted_cdf'`` and ``'inverted_cdf'`` are allowed with weights, as in scikit-learn;
    other methods raise ``ValueError``), the uniform range (minimum and maximum over rows of nonzero
    weight) and the k-means fit. Above ``subsample`` rows the weights instead drive a weighted resample.

    ``inverse_transform`` returns each bin's center. An all-zero one-hot block or a code that names no
    bin raises ``ValueError``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 4), dtype=np.float32)
    >>> Xt = mojolearn.KBinsDiscretizer(n_bins=4, encode="ordinal").fit_transform(X)
    """
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


class _Classifier(_PrepBase):
    """predict / predict_proba / predict_log_proba from a subclass's joint
    log likelihood stages (`_jll_stages`), normalised on the device."""
    _estimator_type = "classifier"
    #: lane cpu4-misc: fit and every scoring method take a
    #: `_arena_io.DeviceRows` (a cross-validation fold's X on the device):
    #: X reaches the arena only through `_x2d` -> `_Prog.put`, so the fold
    #: rows are gathered on the device and never come to the host. Every
    #: subclass (GaussianNB, MultinomialNB, BernoulliNB, ComplementNB,
    #: CategoricalNB, LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis)
    #: keeps to that; a user covariance_estimator takes host rows.
    _mojolearn_device_rows = True

    def predict(self, X):
        """Predict the class of each row.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Rows to classify.

        Returns
        -------
        array of shape (n_samples,)
            Predicted class labels.
        """
        return _CLS_CALL(self, "predict", (X,), _CLS_CONTEXT)

    def predict_proba(self, X):
        """Return class probabilities for each row.

        Probabilities are float32 in ``identical`` mode. In ``fast`` mode (Apple), they are float64, so that
        a very confident prediction does not round to exactly 1.0.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Rows to score.

        Returns
        -------
        array of shape (n_samples, n_classes)
            Probability of each class, in the order of ``classes_``.
        """
        return _CLS_CALL(self, "predict_proba", (X,), _CLS_CONTEXT)

    def predict_log_proba(self, X):
        """Return the log of the class probabilities for each row.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Rows to score.

        Returns
        -------
        array of shape (n_samples, n_classes)
            Log probability of each class, in the order of ``classes_``.
        """
        return _CLS_CALL(self, "predict_log_proba", (X,), _CLS_CONTEXT)

    def predict_joint_log_proba(self, X):
        """Return the unnormalized log score of each row and class.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Rows to score.

        Returns
        -------
        array of shape (n_samples, n_classes)
            Log scores that ``predict_log_proba`` normalizes, in the order of ``classes_``.
        """
        return _CLS_CALL(self, "predict_joint_log_proba", (X,), _CLS_CONTEXT)

    def score(self, X, y):
        """Return the mean accuracy on the given rows and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Rows to classify.
        y : array-like of shape (n_samples,)
            True labels.

        Returns
        -------
        float
            Fraction of rows classified correctly.
        """
        return _CLS_CALL(self, "score", (X, y), _CLS_CONTEXT)


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


#: TOMBSTONE: lane fg-knn-nb G5 (MOJOLEARN_XPREP_DEVICE_CODES, `_fit_device_codes`:
#: GaussianNB.fit classes and codes from the device unique_inverse) deleted
#: 2026-10-09 (lane postmerge-act-2): SLOWER, gaussian-nb istella NV 1.29x / AMD
#: 1.12x, taxi NV 1.70x / AMD 1.35x (nv n0668->n0631, amd a1066->a1091), accuracy
#: SAME, same digests; code recoverable at main 5c137b55e.


# partial_fit merges with sklearn's `_update_mean_variance` (Chan's pairwise
# rule) and adds the batch's epsilon to the merged variance; the running
# variance is kept without epsilon (sklearn subtracts the NEW batch's epsilon
# from a variance that holds the old one, see naive_bayes/NOT_IMPLEMENTED.tsv).
# sample_weight follows numpy `average`. float64 predict_proba is the FAST
# binding's `x_prep_proba64` export (`_proba64_on`).
class GaussianNB(_Classifier):
    """Gaussian naive Bayes classifier.

    Follows scikit-learn's ``GaussianNB``. Fit learns each class's per-feature mean and population
    variance, adds ``var_smoothing`` times the largest feature variance to every variance, and predicts
    with the Gaussian likelihood. Computation is float32.

    ``fit`` accepts ``sample_weight``, which weights the class means, variances and counts as in
    scikit-learn. ``partial_fit(X, y, classes=None, sample_weight=None)`` merges each batch into the
    running counts, means and variances; ``classes`` is required on the first call (``ValueError``
    otherwise) and may only be repeated unchanged afterwards.

    Parameters
    ----------
    priors : array-like of shape (n_classes,), default None
        Class prior probabilities; nonnegative and summing to one (``ValueError`` otherwise). None uses
        the class frequencies in the training data.
    var_smoothing : float, default 1e-9
        Fraction of the largest feature variance added to every variance.

    Attributes
    ----------
    classes_ : array of shape (n_classes,)
        Class labels.
    class_count_ : array of shape (n_classes,)
        Number (or weighted number) of training rows per class.
    class_prior_ : array of shape (n_classes,)
        Probability of each class.
    theta_ : array of shape (n_classes, n_features)
        Mean of each feature per class.
    var_ : array of shape (n_classes, n_features)
        Smoothed variance of each feature per class.
    epsilon_ : float
        Value added to the variances.
    ``n_features_in_`` : int
        Number of columns seen in fit.
    numeric_mode_ : str
        Numeric mode the estimator was fitted in; prediction runs in the same mode.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = (X[:, 0] > 0.5).astype(np.int64)
    >>> model = mojolearn.GaussianNB().fit(X, y)
    >>> model.predict(X[:5])
    """
    _parameters = ("priors", "var_smoothing")

    def __init__(self, *, priors=None, var_smoothing=1e-9):
        self.priors = priors
        self.var_smoothing = var_smoothing

    def fit(self, X, y, sample_weight=None):
        return _CLS_CALL(self, "fit", (X, y, sample_weight), _CLS_CONTEXT)

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        return _CLS_CALL(self, "partial_fit", (X, y, classes, sample_weight), _CLS_CONTEXT)


class _DiscreteNB(_Classifier):
    #: MultinomialNB and ComplementNB take a CSR input on the FAST CSR path
    _csr_ok = False

    def __sklearn_tags__(self):
        # scikit-learn's _BaseDiscreteNB tags: count models, not held to the
        # estimator checks' training-accuracy bar on dense Gaussian data
        tags = super().__sklearn_tags__()
        tags.classifier_tags.poor_score = True
        return tags
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

    def fit(self, X, y, sample_weight=None):
        return _CLS_CALL(self, "fit", (X, y, sample_weight), _CLS_CONTEXT)

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """Update the model with one batch of rows.

        The batch's class and feature counts are added to the running ones, then the log probabilities and
        the class log prior are recomputed from the sums, as in scikit-learn.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training rows.
        y : array-like of shape (n_samples,)
            Class labels.
        classes : array-like, default None
            Every class label. Required on the first call; later calls may pass it only unchanged.
        sample_weight : array-like of shape (n_samples,), default None
            Weights of the rows.

        Returns
        -------
        self
        """
        return _CLS_CALL(self, "partial_fit", (X, y, classes, sample_weight), _CLS_CONTEXT)


class MultinomialNB(_DiscreteNB):
    """Naive Bayes classifier for multinomial (count) features.

    Follows scikit-learn's ``MultinomialNB``. Fit sums the feature values of each class, smooths them by
    ``alpha`` and stores their log share; prediction scores each class by a dot product with those log
    probabilities plus the class log prior. ``X`` must be nonnegative (``ValueError`` otherwise).
    Computation is float32.

    ``fit`` accepts ``sample_weight``, which weights the counts. ``partial_fit(X, y, classes=None,
    sample_weight=None)`` adds each batch's counts to the running ones; ``classes`` is required on the
    first call.

    Parameters
    ----------
    alpha : float, default 1.0
        Additive smoothing. Must be greater than 0; ``alpha <= 0`` raises ``NotImplementedError``.
    force_alpha : bool, default True
        Accepted for scikit-learn compatibility.
    fit_prior : bool, default True
        Learn class priors from the data; when False, use a uniform prior.
    class_prior : array-like of shape (n_classes,), default None
        Given class priors (nonnegative); overrides ``fit_prior``.

    Attributes
    ----------
    classes_ : array of shape (n_classes,)
        Class labels.
    class_count_ : array of shape (n_classes,)
        Number (or weighted number) of training rows per class.
    class_log_prior_ : array of shape (n_classes,)
        Log prior of each class.
    feature_count_ : array of shape (n_classes, n_features)
        Sum of each feature per class.
    feature_log_prob_ : array of shape (n_classes, n_features)
        Smoothed log probability of each feature given the class.
    ``n_features_in_`` : int
        Number of columns seen in fit.
    numeric_mode_ : str
        Numeric mode the estimator was fitted in; prediction runs in the same mode.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.integers(0, 5, (1000, 20)).astype(np.float32)
    >>> y = rng.integers(0, 3, 1000)
    >>> model = mojolearn.MultinomialNB().fit(X, y)
    """
    # scikit-learn tags, read by `_mode.sklearn_tags`
    _sklearn_input_tags = {"positive_only": True}
    _parameters = ("alpha", "force_alpha", "fit_prior", "class_prior")
    _csr_ok = True
    _needs_min = True

    def __init__(self, *, alpha=1.0, force_alpha=True, fit_prior=True, class_prior=None):
        self.alpha = alpha
        self.force_alpha = force_alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior


class BernoulliNB(_DiscreteNB):
    """Naive Bayes classifier for binary features.

    Follows scikit-learn's ``BernoulliNB``. ``X`` is binarized at ``binarize`` (values above the threshold
    become 1), then fit counts how often each feature is 1 in each class, smooths the counts by ``alpha``
    and scores each class with the Bernoulli likelihood, including the features that are 0. Computation
    is float32.

    ``fit`` accepts ``sample_weight``, which weights the counts. ``partial_fit(X, y, classes=None,
    sample_weight=None)`` adds each batch's counts to the running ones; ``classes`` is required on the
    first call.

    Parameters
    ----------
    alpha : float, default 1.0
        Additive smoothing. Must be greater than 0; ``alpha <= 0`` raises ``NotImplementedError``.
    force_alpha : bool, default True
        Accepted for scikit-learn compatibility.
    binarize : float or None, default 0.0
        Threshold for binarizing ``X``; None means ``X`` is already binary.
    fit_prior : bool, default True
        Learn class priors from the data; when False, use a uniform prior.
    class_prior : array-like of shape (n_classes,), default None
        Given class priors (nonnegative); overrides ``fit_prior``.

    Attributes
    ----------
    classes_ : array of shape (n_classes,)
        Class labels.
    class_count_ : array of shape (n_classes,)
        Number (or weighted number) of training rows per class.
    class_log_prior_ : array of shape (n_classes,)
        Log prior of each class.
    feature_count_ : array of shape (n_classes, n_features)
        Number of rows per class in which each feature is 1.
    feature_log_prob_ : array of shape (n_classes, n_features)
        Smoothed log probability that each feature is 1 given the class.
    ``n_features_in_`` : int
        Number of columns seen in fit.
    numeric_mode_ : str
        Numeric mode the estimator was fitted in; prediction runs in the same mode.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.integers(0, 2, (1000, 20)).astype(np.float32)
    >>> y = rng.integers(0, 2, 1000)
    >>> model = mojolearn.BernoulliNB().fit(X, y)
    """
    _parameters = ("alpha", "force_alpha", "binarize", "fit_prior", "class_prior")

    def __init__(self, *, alpha=1.0, force_alpha=True, binarize=0.0, fit_prior=True, class_prior=None):
        self.alpha = alpha
        self.force_alpha = force_alpha
        self.binarize = binarize
        self.fit_prior = fit_prior
        self.class_prior = class_prior


# ---------------------------------------------------------------- discriminant analysis


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


# solver 'svd': sklearn's two SVDs are symmetric eigendecompositions of the
# Gram matrices (cyclic Jacobi, x_prep/eigh.mojo), so scalings_ and transform
# match sklearn up to each component's sign. 'lsqr' is the pooled class
# covariance then lstsq's minimum-norm solve through its eigendecomposition
# (sklearn's `_solve_lstsq`); 'eigen' solves Sb v = e Sw v as
# Sw^-1/2 Sb Sw^-1/2 (`_solve_eigen`). covariance_estimator runs in Python on
# host rows; its covariances enter the device solve as float32.
class LinearDiscriminantAnalysis(_Classifier):
    """Linear discriminant analysis classifier and supervised dimensionality reduction.

    Follows scikit-learn's ``LinearDiscriminantAnalysis``. It fits one Gaussian per class with a shared
    covariance, classifies by the resulting linear decision function, and ``transform`` projects onto the
    most discriminative directions. Computation is float32.

    With ``solver='svd'`` the projection directions (``scalings_`` and ``transform``) match
    scikit-learn's up to the sign of each component, and the decision function matches outright.

    Parameters
    ----------
    solver : {'svd', 'lsqr', 'eigen'}, default 'svd'
        ``'svd'`` needs no covariance matrix. ``'lsqr'`` solves with the pooled class covariance
        (minimum-norm least squares) and has no ``transform``. ``'eigen'`` solves the generalized
        eigenproblem between the between-class and within-class scatter; it raises ``ValueError`` when
        the within-class covariance is not positive definite.
    shrinkage : {None, 'auto'} or float, default None
        Covariance shrinkage for ``'lsqr'`` and ``'eigen'``: ``'auto'`` uses the Ledoit-Wolf estimate
        on standardized classes, a float in ``[0, 1]`` sets it directly. With ``solver='svd'`` it raises
        ``NotImplementedError``.
    priors : array-like of shape (n_classes,), default None
        Class priors (nonnegative); renormalized when they do not sum to one. None uses the class
        frequencies.
    n_components : int, default None
        Number of components kept by ``transform``, at most ``min(n_classes - 1, n_features)``
        (``ValueError`` otherwise). None keeps that maximum.
    store_covariance : bool, default False
        Store the pooled within-class covariance in ``covariance_`` (always stored by ``'lsqr'`` and
        ``'eigen'``).
    tol : float, default 1e-4
        Threshold below which singular values are treated as zero by ``'svd'``.
    covariance_estimator : object, default None
        An estimator with ``fit`` and a ``covariance_`` attribute, used instead of the empirical
        covariance with ``'lsqr'`` or ``'eigen'`` (``ValueError`` with ``'svd'`` or with ``shrinkage``
        set). It is fitted on the host on each class's float32 rows (and, for ``'eigen'``, on all rows).

    Attributes
    ----------
    classes_ : array of shape (n_classes,)
        Class labels.
    coef_ : array of shape (n_classes, n_features), or (1, n_features) for two classes
        Weights of the decision function.
    intercept_ : array of shape (n_classes,), or (1,) for two classes
        Intercepts of the decision function.
    means_ : array of shape (n_classes, n_features)
        Class means.
    priors_ : array of shape (n_classes,)
        Class priors used.
    xbar_ : array of shape (n_features,)
        Overall mean.
    scalings_ : array of shape (n_features, rank)
        Projection directions.
    explained_variance_ratio_ : array of shape (n_components,)
        Share of between-class variance explained by each component.
    covariance_ : array of shape (n_features, n_features)
        Pooled within-class covariance, when stored.
    ``n_features_in_`` : int
        Number of columns seen in fit.
    numeric_mode_ : str
        Numeric mode the estimator was fitted in; prediction runs in the same mode.

    Notes
    -----
    Fit raises ``ValueError`` with fewer than two classes or with no more rows than classes.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = rng.integers(0, 3, 1000)
    >>> lda = mojolearn.LinearDiscriminantAnalysis(n_components=2).fit(X, y)
    >>> Z = lda.transform(X)
    """
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
        return _CLS_CALL(self, "fit", (X, y), _CLS_CONTEXT)

    def decision_function(self, X):
        return _CLS_CALL(self, "decision_function", (X,), _CLS_CONTEXT)

    def transform(self, X):
        return _CLS_CALL(self, "transform", (X,), _CLS_CONTEXT)

    def fit_transform(self, X, y):
        return _CLS_CALL(self, "fit_transform", (X, y), _CLS_CONTEXT)


# Follows sklearn 1.9's QDA (solver, shrinkage, covariance_estimator). Per
# class, the eigendecomposition of the class covariance (divisor n_k) stands
# in for sklearn's SVD of the centered class rows (same S^2 / n_k, vectors up
# to sign). solver 'eigen' takes sklearn's `_cov(X_k, shrinkage)`.
# store_covariance under 'svd' is V diag(scalings) V^T, as sklearn forms it.
class QuadraticDiscriminantAnalysis(_Classifier):
    """Quadratic discriminant analysis classifier.

    Follows scikit-learn's ``QuadraticDiscriminantAnalysis``. It fits one Gaussian per class, each with
    its own covariance, and classifies by the resulting quadratic decision function. Computation is
    float32.

    Parameters
    ----------
    solver : {'svd', 'eigen'}, default 'svd'
        ``'svd'`` regularizes with ``reg_param``. ``'eigen'`` uses the empirical (optionally shrunk)
        class covariances and ignores ``reg_param``, as in scikit-learn.
    shrinkage : {None, 'auto'} or float, default None
        Covariance shrinkage for ``'eigen'``: ``'auto'`` uses the Ledoit-Wolf estimate, a float in
        ``[0, 1]`` sets it directly. With ``solver='svd'`` it raises ``NotImplementedError``.
    priors : array-like of shape (n_classes,), default None
        Class priors (nonnegative). None uses the class frequencies.
    reg_param : float, default 0.0
        Regularization of the class covariances under ``'svd'``.
    store_covariance : bool, default False
        Store each class covariance in ``covariance_``.
    tol : float, default 1e-4
        A class whose scalings are not all above ``tol`` is rejected with ``ValueError`` (its
        covariance is not full rank).
    covariance_estimator : object, default None
        An estimator with ``fit`` and a ``covariance_`` attribute, used instead of the empirical
        covariance with ``solver='eigen'`` (``ValueError`` with ``'svd'`` or with ``shrinkage`` set). It
        is fitted on the host on each class's float32 rows.

    Attributes
    ----------
    classes_ : array of shape (n_classes,)
        Class labels.
    means_ : array of shape (n_classes, n_features)
        Class means.
    priors_ : array of shape (n_classes,)
        Class priors used.
    rotations_ : list of arrays
        Per class, the (n_features, n_features) eigenvectors of its covariance.
    scalings_ : list of arrays
        Per class, the (n_features,) variances along those eigenvectors.
    covariance_ : list of arrays
        Per class covariance, when ``store_covariance=True``.
    ``n_features_in_`` : int
        Number of columns seen in fit.
    numeric_mode_ : str
        Numeric mode the estimator was fitted in; prediction runs in the same mode.

    Notes
    -----
    Fit raises ``ValueError`` with fewer than two classes or when a class has a single row.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 4), dtype=np.float32)
    >>> y = rng.integers(0, 2, 1000)
    >>> model = mojolearn.QuadraticDiscriminantAnalysis().fit(X, y)
    """
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
        return _CLS_CALL(self, "fit", (X, y), _CLS_CONTEXT)

    def decision_function(self, X):
        return _CLS_CALL(self, "decision_function", (X,), _CLS_CONTEXT)


# ---------------------------------------------------------------- additions: transformers
# Transform is sklearn's two-sided interpolation; 'normal' uses Acklam's
# inverse normal CDF in float32, clipped at sklearn's +-5.1993.
# inverse_transform is sklearn's (norm.cdf as Cephes ndtr in float32 for
# 'normal', then np.interp back onto the quantiles). The subsample is a
# without-replacement draw from random_state by splitmix64.
class QuantileTransformer(_PrepBase):
    """Map each feature to a uniform or normal distribution through its quantiles.

    Follows scikit-learn's ``QuantileTransformer``. Fit computes, per column, numpy's linear percentiles
    of the non-NaN values at ``n_quantiles`` evenly spaced references; transform maps values through
    them by interpolation. NaN values are kept as NaN. Input and output are float32. ``scipy.sparse``
    input is not supported.

    Parameters
    ----------
    n_quantiles : int, default 1000
        Number of quantiles; capped at the number of rows used in fit.
    output_distribution : {'uniform', 'normal'}, default 'uniform'
        Target distribution. ``'normal'`` output is clipped to about ``+-5.2``, as in scikit-learn.
    ignore_implicit_zeros : bool, default False
        Accepted for scikit-learn compatibility (it applies to sparse input only).
    subsample : int or None, default 10000
        Above this many rows, fit uses ``subsample`` rows drawn without replacement. None uses every row.
    random_state : int or None, default None
        Seed for the subsample. None uses seed 0. The draw uses mojolearn's own generator, so the rows
        differ from scikit-learn's for the same seed.
    copy : bool, default True
        Accepted for scikit-learn compatibility; transform always returns a new array.

    Attributes
    ----------
    ``n_quantiles_`` : int
        Number of quantiles used.
    quantiles_ : array of shape (``n_quantiles_``, n_features)
        Quantile values of each column.
    references_ : array of shape (``n_quantiles_``,)
        Quantile levels in ``[0, 1]``.
    ``n_features_in_`` : int
        Number of columns seen in fit.
    numeric_mode_ : str
        Numeric mode the estimator was fitted in; transform runs in the same mode.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).exponential(size=(1000, 4)).astype(np.float32)
    >>> Xt = mojolearn.QuantileTransformer(output_distribution="normal").fit_transform(X)
    """
    # scikit-learn tags, read by `_mode.sklearn_tags`
    _sklearn_input_tags = {"allow_nan": True}
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
        return _STATELESS_CALL(self, "fit", (X, y), _STATELESS_CONTEXT)

    def fit_transform(self, X, y=None, **fit_params):
        """Fit to ``X``, then transform ``X``.

        Returns
        -------
        array
            The transformed data.
        """
        return _STATELESS_CALL(self, "fit_transform", (X, y), _STATELESS_CONTEXT)

    def transform(self, X):
        return _STATELESS_CALL(self, "transform", (X,), _STATELESS_CONTEXT)

    def inverse_transform(self, X):
        return _STATELESS_CALL(self, "inverse_transform", (X,), _STATELESS_CONTEXT)


# Each column's lambda maximizes sklearn's log-likelihood by a fixed-step
# golden-section search over [-8, 8] (sklearn: scipy's Brent from the bracket
# (-2, 2)), float32. inverse_transform is x * scale + mean, then scipy's
# inv_boxcox or sklearn's yeo-johnson inverse as exp(log1p(lambda x) / lambda)
# in float32.
class PowerTransformer(_PrepBase):
    """Apply a power transform per feature to make the data more Gaussian.

    Follows scikit-learn's ``PowerTransformer``. Each column gets the Yeo-Johnson or Box-Cox transform
    with the lambda that maximizes the log-likelihood, then (by default) is standardized to zero mean and
    unit variance. Lambda is searched in ``[-8, 8]``, so fitted lambdas can differ slightly from
    scikit-learn's. NaN values are ignored in fit and kept as NaN. Input and output are float32.

    Parameters
    ----------
    method : {'yeo-johnson', 'box-cox'}, default 'yeo-johnson'
        ``'box-cox'`` requires strictly positive data (``ValueError`` otherwise).
    standardize : bool, default True
        Standardize the transformed columns.
    copy : bool, default True
        Accepted for scikit-learn compatibility; transform always returns a new array.

    Attributes
    ----------
    lambdas_ : array of shape (n_features,)
        Fitted lambda of each column.
    ``n_features_in_`` : int
        Number of columns seen in fit.
    numeric_mode_ : str
        Numeric mode the estimator was fitted in; transform runs in the same mode.

    Notes
    -----
    ``inverse_transform`` undoes the standardization, then the power transform.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).lognormal(size=(1000, 4)).astype(np.float32)
    >>> Xt = mojolearn.PowerTransformer().fit_transform(X)
    """
    # scikit-learn tags, read by `_mode.sklearn_tags`
    _sklearn_input_tags = {"allow_nan": True}
    _parameters = ("method", "standardize", "copy")

    def __init__(self, method="yeo-johnson", *, standardize=True, copy=True):
        self.method = method
        self.standardize = standardize
        self.copy = copy

    def fit(self, X, y=None):
        return _STATELESS_CALL(self, "fit", (X, y), _STATELESS_CONTEXT)

    def fit_transform(self, X, y=None, **fit_params):
        """Fit to ``X``, then transform ``X``.

        Returns
        -------
        array
            The transformed data.
        """
        return _STATELESS_CALL(self, "fit_transform", (X, y), _STATELESS_CONTEXT)

    def transform(self, X):
        return _STATELESS_CALL(self, "transform", (X,), _STATELESS_CONTEXT)

    def inverse_transform(self, X):
        return _STATELESS_CALL(self, "inverse_transform", (X,), _STATELESS_CONTEXT)


# Row norms sum the columns in ascending order.
class Normalizer(_PrepBase):
    """Scale each row to unit norm.

    Follows scikit-learn's ``Normalizer``. Each row is divided by its ``'l1'``, ``'l2'`` or ``'max'``
    norm; a row of zeros is left unchanged. The transformer is stateless: fit only records the number of
    columns. Input and output are float32.

    Parameters
    ----------
    norm : {'l1', 'l2', 'max'}, default 'l2'
        Norm used for each row.
    copy : bool, default True
        Accepted for scikit-learn compatibility; transform always returns a new array.

    Attributes
    ----------
    ``n_features_in_`` : int
        Number of columns seen in fit.
    numeric_mode_ : str
        Numeric mode at fit time; transform runs in the same mode.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> Xt = mojolearn.Normalizer(norm="l1").fit_transform(X)
    """
    _parameters = ("norm", "copy")

    def __init__(self, norm="l2", *, copy=True):
        self.norm = norm
        self.copy = copy

    def fit(self, X, y=None):
        return _STATELESS_CALL(self, "fit", (X, y), _STATELESS_CONTEXT)

    def transform(self, X, copy=None):
        return _STATELESS_CALL(self, "transform", (X,), _STATELESS_CONTEXT)

    def fit_transform(self, X, y=None, **fit_params):
        """Fit to ``X``, then transform ``X``.

        Returns
        -------
        array
            The transformed data.
        """
        return _STATELESS_CALL(self, "fit_transform", (X, y), _STATELESS_CONTEXT)


# Each output column is the product of its input columns left to right.
class PolynomialFeatures(_PrepBase):
    """Generate polynomial and interaction features.

    Follows scikit-learn's ``PolynomialFeatures``, with the same output column order: the bias column,
    then the products of input columns degree by degree (combinations with replacement, or without when
    ``interaction_only``). Input and output are float32.

    Parameters
    ----------
    degree : int or tuple of (int, int), default 2
        Largest degree, or ``(min_degree, max_degree)``. Negative values or ``min_degree > max_degree``
        raise ``ValueError``.
    interaction_only : bool, default False
        Only products of distinct columns (no powers of one column).
    include_bias : bool, default True
        Include the constant column of ones.
    order : {'C', 'F'}, default 'C'
        Memory order of the output array; ``'F'`` returns the same values column-major.

    Attributes
    ----------
    powers_ : array of shape (``n_output_features_``, ``n_features_in_``)
        Exponent of each input column in each output column.
    ``n_features_in_`` : int
        Number of columns seen in fit.
    ``n_output_features_`` : int
        Number of output columns.
    numeric_mode_ : str
        Numeric mode at fit time; transform runs in the same mode.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.arange(6, dtype=np.float32).reshape(3, 2)
    >>> mojolearn.PolynomialFeatures(degree=2).fit_transform(X)
    """
    _parameters = ("degree", "interaction_only", "include_bias", "order")

    def __init__(self, degree=2, *, interaction_only=False, include_bias=True, order="C"):
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.order = order

    def fit(self, X, y=None):
        return _STATELESS_CALL(self, "fit", (X, y), _STATELESS_CONTEXT)

    def transform(self, X):
        return _STATELESS_CALL(self, "transform", (X,), _STATELESS_CONTEXT)

    def fit_transform(self, X, y=None, **fit_params):
        """Fit to ``X``, then transform ``X``.

        Returns
        -------
        array
            The transformed data.
        """
        return _STATELESS_CALL(self, "fit_transform", (X, y), _STATELESS_CONTEXT)


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
    """Generate B-spline basis features for each feature.

    Follows scikit-learn's ``SplineTransformer``. Each column is expanded into B-splines of ``degree`` on
    ``n_knots`` base knots, extended by ``degree`` knots at each end at the edge spacing (or wrapped, for
    periodic splines). The output is always a dense float32 array.

    Parameters
    ----------
    n_knots : int, default 5
        Number of base knots, at least 2 (``ValueError`` otherwise). Ignored when ``knots`` is an array.
    degree : int, default 3
        Polynomial degree of the splines, from 0 to 7 (``ValueError`` otherwise).
    knots : {'uniform', 'quantile'} or array-like of shape (n_knots, n_features), default 'uniform'
        ``'uniform'`` spaces the knots evenly over the training range, ``'quantile'`` places them at
        numpy's linear percentiles, and an array gives them explicitly (sorted, without duplicates).
    extrapolation : {'constant', 'continue', 'linear', 'periodic', 'error'}, default 'constant'
        Behavior outside the knot range. ``'error'`` makes transform raise ``ValueError``.
        ``'periodic'`` requires ``degree < n_knots``.
    include_bias : bool, default True
        Keep the last spline of each feature; when False it is dropped.
    order : {'C', 'F'}, default 'C'
        Memory order of the output array.
    handle_missing : {'error', 'zeros'}, default 'error'
        ``'error'`` raises ``ValueError`` on NaN; ``'zeros'`` leaves NaN out of the knots and encodes it
        as a block of zeros.
    sparse_output : bool, default False
        Must be False; True raises ``NotImplementedError``.

    Attributes
    ----------
    bsplines_ : list of arrays
        Per feature, the full knot vector of its splines (scikit-learn stores ``BSpline`` objects here).
    ``n_features_in_`` : int
        Number of columns seen in fit.
    n_features_out_ : int
        Number of output columns.
    numeric_mode_ : str
        Numeric mode the estimator was fitted in; transform runs in the same mode.

    Notes
    -----
    ``fit`` and ``fit_transform`` accept ``sample_weight`` (nonnegative), which weights the knot
    placement: the weighted percentile for ``'quantile'``, and the range over rows of nonzero weight for
    ``'uniform'``. Fit needs at least two rows.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 3), dtype=np.float32)
    >>> Xt = mojolearn.SplineTransformer(n_knots=6, degree=3).fit_transform(X)
    """
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


    def _mojolearn_allows_nan(self):
        """handle_missing='zeros' takes NaN (its rows' spline features are 0)."""
        return self.handle_missing == "zeros"
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
            raise ValueError(f"Found array with {n} sample(s) (shape=({n}, {d})) while a minimum of 2 "
                             "is required by SplineTransformer.")
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
    """Set feature values to 0 or 1 by a threshold.

    Follows scikit-learn's ``Binarizer``: values greater than ``threshold`` become 1, the others 0; NaN
    is kept. The transformer is stateless: fit only records the number of columns. Input and output are
    float32.

    Parameters
    ----------
    threshold : float, default 0.0
        Values above this become 1.
    copy : bool, default True
        Accepted for scikit-learn compatibility; transform always returns a new array.

    Attributes
    ----------
    ``n_features_in_`` : int
        Number of columns seen in fit.
    numeric_mode_ : str
        Numeric mode at fit time; transform runs in the same mode.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.array([[-1.0, 0.5], [2.0, 0.0]], dtype=np.float32)
    >>> mojolearn.Binarizer(threshold=0.25).fit_transform(X)
    """
    _parameters = ("threshold", "copy")

    def __init__(self, *, threshold=0.0, copy=True):
        self.threshold = threshold
        self.copy = copy

    def fit(self, X, y=None):
        return _STATELESS_CALL(self, "fit", (X, y), _STATELESS_CONTEXT)

    def transform(self, X, copy=None):
        return _STATELESS_CALL(self, "transform", (X,), _STATELESS_CONTEXT)

    def fit_transform(self, X, y=None, **fit_params):
        """Fit to ``X``, then transform ``X``.

        Returns
        -------
        array
            The transformed data.
        """
        return _STATELESS_CALL(self, "fit_transform", (X, y), _STATELESS_CONTEXT)


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


# TOMBSTONE: MOJOLEARN_IDN_LAB_UNIQUE_HIST, MOJOLEARN_IDN_LAB_CODES_DIRECT, MOJOLEARN_IDN_LAB_ONE_ENTRY (losers: label-encoder 2.1-2.6x, label-binarizer istella 1.15-1.39x slower, NV + AMD) deleted 2026-10-11 by de9ce7b39; code recoverable at de9ce7b39^.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_LAB_UNIQUE_HIST_CODES_DIRECT_ONE_ENTRY.patch
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


# Numeric labels run on the device (sort + run scan for classes_, a binary
# search for transform); a numeric label buffer crosses as its raw words
# (`_label_buffer`), so fit_transform is one program. str labels take Python.
class LabelEncoder(_PrepBase):
    """Encode target labels as integers ``0`` to ``n_classes - 1``.

    Follows scikit-learn's ``LabelEncoder``. ``classes_`` are the sorted distinct labels and
    ``transform`` returns each label's index as int32. Numeric and string labels are supported. A label
    not seen in fit raises ``ValueError`` in ``transform`` and ``inverse_transform``.

    Attributes
    ----------
    classes_ : array or list
        Sorted distinct labels: an int64 array for integer labels, a float64 array for real labels,
        otherwise a list.
    numeric_mode_ : str
        Numeric mode at fit time; transform runs in the same mode.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> enc = mojolearn.LabelEncoder().fit(np.array([10, 30, 20, 10]))
    >>> enc.transform(np.array([30, 10]))
    >>> enc.inverse_transform(np.array([0, 2]))
    """
    # scikit-learn tags: transforms a 1-D label vector, not an X matrix
    _sklearn_input_tags = {"one_d_array": True, "two_d_array": False}
    _sklearn_target_tags = {"one_d_labels": True}
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
            native = _labels.str_labels("lookup", list(self._classes), values, False)
            if native is not None:
                codes, missing, count = native
                if count:
                    raise ValueError(f"mojolearn: y contains previously unseen labels: {missing}")
                return codes
            index = {c: i for i, c in enumerate(self._classes)}  # cpu-route: non-str object labels, the explicit label input step
            missing = [v for v in values if v not in index]  # cpu-route: non-str object labels, the explicit label input step
            if missing:
                raise ValueError(f"mojolearn: y contains previously unseen labels: {missing[:5]}")
            return Array.from_list([index[v] for v in values], "<i4")  # cpu-route: non-str object labels, the explicit label input step
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


# Numeric labels take the device route, str labels Python's.
class LabelBinarizer(_PrepBase):
    """Binarize labels in a one-vs-all fashion.

    Follows scikit-learn's ``LabelBinarizer``. ``transform`` returns an int32 matrix with one column per
    class holding ``pos_label`` for the row's class and ``neg_label`` elsewhere; with two classes it
    returns a single column (positive for the second class), and with one class a single column of
    ``neg_label``. A label not seen in fit becomes a row of ``neg_label``. Numeric and string labels are
    supported.

    A 2-D ``y`` with more than one column and at most two distinct integer values is treated as a
    multilabel indicator, as in scikit-learn: ``classes_`` are the column indices, and each nonzero
    entry becomes ``pos_label`` and each zero ``neg_label``.

    Parameters
    ----------
    neg_label : int, default 0
        Value for negative entries.
    pos_label : int, default 1
        Value for positive entries; must be greater than ``neg_label`` (``ValueError`` otherwise).
    sparse_output : bool, default False
        Must be False; True raises ``NotImplementedError``.

    Attributes
    ----------
    classes_ : array or list
        Class labels.
    y_type_ : str
        ``'binary'``, ``'multiclass'`` or ``'multilabel-indicator'``.
    numeric_mode_ : str
        Numeric mode at fit time; transform runs in the same mode.

    Notes
    -----
    ``inverse_transform(Y, threshold=None)`` takes the largest column for multiclass output and applies
    ``threshold`` (by default halfway between ``neg_label`` and ``pos_label``) for binary and multilabel
    output, as in scikit-learn.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> lb = mojolearn.LabelBinarizer().fit(np.array([1, 2, 6, 4, 2]))
    >>> lb.transform(np.array([1, 6]))
    """
    # scikit-learn tags: transforms a 1-D label vector, not an X matrix
    _sklearn_input_tags = {"one_d_array": True, "two_d_array": False}
    _sklearn_target_tags = {"one_d_labels": True}
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
            native = _labels.str_labels("lookup", list(self._classes), values, True) if values else None
            if native is not None:
                codes = pr.put(native[0])
            else:
                index = {c: i for i, c in enumerate(self._classes)}  # cpu-route: non-str object labels, the explicit label input step
                codes = pr.put_list([index.get(v, -1) for v in values])  # cpu-route: non-str object labels, the explicit label input step
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


# Numeric labels take the device route (a sort, a binary search, a scatter
# of ones), str labels Python's.
class MultiLabelBinarizer(_PrepBase):
    """Convert sets of labels to a binary indicator matrix.

    Follows scikit-learn's ``MultiLabelBinarizer``. ``y`` is an iterable of label collections, one per
    sample. ``transform`` returns an int32 matrix with a 1 where the sample has the class. A label not
    seen in fit is ignored (scikit-learn also warns). Numeric and string labels are supported.

    Parameters
    ----------
    classes : array-like, default None
        Class labels in the order of the output columns. None uses the sorted union of all labels in
        fit.
    sparse_output : bool, default False
        Must be False; True raises ``NotImplementedError``.

    Attributes
    ----------
    classes_ : array or list
        Class labels: an int64 array for integer labels, a float64 array for real labels, otherwise a
        list.
    numeric_mode_ : str
        Numeric mode at fit time; transform runs in the same mode.

    Notes
    -----
    ``inverse_transform`` returns a list with one tuple of labels per row.

    Examples
    --------
    >>> import mojolearn
    >>> mlb = mojolearn.MultiLabelBinarizer()
    >>> mlb.fit_transform([[1, 2], [3], [2, 3]])
    """
    # scikit-learn tags: transforms label collections, not an X matrix
    _sklearn_input_tags = {"two_d_array": False}
    _sklearn_target_tags = {"two_d_labels": True}
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
            native = _labels.str_labels("lookup", list(self._classes), flat, True)
            if native is not None:
                codes = pr.put(native[0])
            else:
                index = {c: i for i, c in enumerate(self._classes)}  # cpu-route: non-str object labels, the explicit label input step
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


# With the default estimator (BayesianRidge, default priors, max_iter 300, tol
# 1e-3) the whole fit is one device program; another estimator runs the same
# rounds in Python over its fit / predict (predict(return_std=True) for
# sample_posterior). The stop test is the matrix inf-norm of the round's change
# < tol * max|X_observed|. Every draw (random order, neighbours, posterior
# samples) comes from random_state by splitmix64 (sklearn draws numpy's; None
# is seed 0).
_INF = float("inf")


class IterativeImputer(_PrepBase):
    """Impute missing values by modeling each feature from the others, round by round.

    Follows scikit-learn's ``IterativeImputer``. Missing entries are first filled by a
    :class:`SimpleImputer` with ``initial_strategy``. Then, for up to ``max_iter`` rounds, each feature
    with missing values is regressed on the other features over its observed rows, and its missing
    entries are replaced by the predictions, clipped to ``[min_value, max_value]``. Fit stops early when
    the largest change in a round falls below ``tol`` times the largest absolute observed value. Input
    and output are float32.

    The default estimator is a Bayesian ridge regression (default priors, at most 300 iterations, tol
    1e-3), as in scikit-learn.

    Parameters
    ----------
    estimator : object, default None
        Regressor with ``fit`` and ``predict``, cloned for each step and run on the host. None uses the
        built-in Bayesian ridge regression.
    missing_values : float, default NaN
        Value that marks a missing entry.
    sample_posterior : bool, default False
        Draw each imputed value from the predictive normal distribution, truncated to
        ``[min_value, max_value]``, instead of using its mean. A custom ``estimator`` must then support
        ``predict(X, return_std=True)`` (``ValueError`` otherwise).
    max_iter : int, default 10
        Maximum number of imputation rounds.
    tol : float, default 1e-3
        Stopping tolerance, relative to the largest absolute observed value.
    n_nearest_features : int, default None
        Number of other features used to predict each feature, drawn without replacement with
        probability proportional to their absolute correlation with it. None uses all of them. Must be an
        int ``>= 1``.
    initial_strategy : {'mean', 'median', 'most_frequent', 'constant'}, default 'mean'
        Strategy of the initial :class:`SimpleImputer`.
    fill_value : float, default None
        Constant for ``initial_strategy='constant'``.
    imputation_order : {'ascending', 'descending', 'roman', 'arabic', 'random'}, default 'ascending'
        Order in which features are imputed in each round: by increasing or decreasing count of missing
        values, left to right, right to left, or a new random permutation each round.
    skip_complete : bool, default False
        Skip features with no missing values in fit.
    min_value : float or array-like of shape (n_features,), default -inf
        Lower bound for imputed values.
    max_value : float or array-like of shape (n_features,), default inf
        Upper bound for imputed values.
    verbose : int, default 0
        Accepted for scikit-learn compatibility.
    random_state : int or None, default None
        Seed for the random order, the feature draws and the posterior samples. None uses seed 0. Draws
        use mojolearn's own generator, so they differ from scikit-learn's for the same seed.
    add_indicator : bool, default False
        Append missing-indicator columns (1.0 where missing), as scikit-learn's ``MissingIndicator``.
    keep_empty_features : bool, default False
        Keep columns that had no observed values in fit (passed to the initial imputer).

    Attributes
    ----------
    initial_imputer_ : SimpleImputer
        Imputer used for the initial fill.
    imputation_sequence_ : list of tuples
        The fitted steps in the order they are applied.
    n_iter_ : int
        Number of rounds run.
    n_features_with_missing_ : int
        Number of features with missing values in fit.
    ``n_features_in_`` : int
        Number of columns seen in fit.
    numeric_mode_ : str
        Numeric mode the estimator was fitted in; transform runs in the same mode.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 5), dtype=np.float32)
    >>> X[rng.random(X.shape) < 0.1] = np.nan
    >>> Xt = mojolearn.IterativeImputer(max_iter=5).fit_transform(X)
    """
    # scikit-learn tags, read by `_mode.sklearn_tags`
    _sklearn_input_tags = {"allow_nan": True}
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
        mi = self.max_iter
        if isinstance(mi, bool) or not isinstance(mi, numbers.Integral) or mi < 0:
            raise ValueError("The 'max_iter' parameter of IterativeImputer must be an int in the range "
                             f"[0, inf). Got {mi!r} instead.")
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
        if scale == _INF:
            # the device's largest |observed value| (c2_colmax) is infinite
            # exactly when an observed entry is; scikit-learn refuses it
            raise ValueError("Input X contains infinity or a value too large for dtype('float64').")
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
        """Return the selected features.

        Parameters
        ----------
        indices : bool, default False
            Return the indices of the selected columns instead of a boolean mask.

        Returns
        -------
        list
            Boolean mask over the input columns, or the selected column indices.
        """
        self._check_fitted()
        mask = list(self._mask)
        return [j for j, m in enumerate(mask) if m] if indices else mask  # glue: support index list (mask-sized: feature support mask)

    def transform(self, X):
        """Keep only the selected columns of ``X``.

        Raises ``ValueError`` when no column is selected.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data with the columns seen in fit.

        Returns
        -------
        array of shape (n_samples, n_selected_features)
            The selected columns, float32.
        """
        return _STATELESS_CALL(self, "transform", (X,), _STATELESS_CONTEXT)


# At threshold 0 the variance is min(variance, max - min), so a constant
# column is exactly 0. variance > threshold is compared in binary64 on the
# device (c2_key64, c2_gt).
class VarianceThreshold(_SelectorMixin):
    """Feature selector that removes low-variance features.

    Follows scikit-learn's ``VarianceThreshold``. Fit computes each column's population variance over
    its non-NaN values and keeps the columns whose variance is greater than ``threshold``. With the
    default threshold of 0, a constant column has variance exactly 0 and is removed. Fit raises
    ``ValueError`` when no column passes.

    Parameters
    ----------
    threshold : float, default 0.0
        Columns with variance at or below this are removed.

    Attributes
    ----------
    variances_ : array of shape (n_features,)
        Variance of each column.
    ``n_features_in_`` : int
        Number of columns seen in fit.
    numeric_mode_ : str
        Numeric mode the estimator was fitted in; transform runs in the same mode.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.array([[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]], dtype=np.float32)
    >>> mojolearn.VarianceThreshold().fit_transform(X)
    """
    # scikit-learn tags, read by `_mode.sklearn_tags`
    _sklearn_input_tags = {"allow_nan": True}
    _parameters = ("threshold",)

    def __init__(self, threshold=0.0):
        self.threshold = threshold

    def fit(self, X, y=None):
        return _STATELESS_CALL(self, "fit", (X, y), _STATELESS_CONTEXT)

    def fit_transform(self, X, y=None, **fit_params):
        """Fit to ``X``, then keep only the selected columns of ``X``.

        Returns
        -------
        array
            The selected columns, float32.
        """
        return _STATELESS_CALL(self, "fit_transform", (X, y), _STATELESS_CONTEXT)


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
    """Compute the ANOVA F-value of each feature against the class labels.

    Follows scikit-learn's ``f_classif``. As in scikit-learn, a constant feature (or a single class)
    scores NaN with p-value NaN, and a feature that is constant within every class but not across them
    scores ``+inf`` with p-value 0. Computation is float32.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Class labels.

    Returns
    -------
    f_statistic : array of shape (n_features,)
        F-value of each feature.
    p_values : array of shape (n_features,)
        p-value of each F-value.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = rng.integers(0, 3, 1000)
    >>> F, p = mojolearn.f_classif(X, y)
    """
    return _scores_classif(X, y, "f")


def chi2(X, y):
    """Compute chi-squared statistics between each nonnegative feature and the class labels.

    Follows scikit-learn's ``chi2``: the per-class sums of each feature are compared with their
    expected values. ``X`` must be nonnegative (``ValueError`` otherwise). An all-zero feature scores NaN
    with p-value NaN, as in scikit-learn. Computation is float32.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Nonnegative feature matrix, such as counts or frequencies.
    y : array-like of shape (n_samples,)
        Class labels.

    Returns
    -------
    chi2 : array of shape (n_features,)
        Chi-squared statistic of each feature.
    p_values : array of shape (n_features,)
        p-value of each statistic.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.integers(0, 10, (1000, 8)).astype(np.float32)
    >>> y = rng.integers(0, 2, 1000)
    >>> stat, p = mojolearn.chi2(X, y)
    """
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
    """Compute the F-statistic of each feature's correlation with a continuous target.

    Follows scikit-learn's ``f_regression``: the F-statistic and p-value of each feature's Pearson
    correlation with ``y``. Computation is float32.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Target values.
    center : bool, default True
        Center ``X`` and ``y`` before computing the correlation.
    force_finite : bool, default True
        Replace non-finite results, as in scikit-learn: a constant feature or target gives F 0 and
        p-value 1, and a perfect correlation gives the largest float32 F and p-value 0. With False, the
        raw NaN and ``+inf`` values are returned.

    Returns
    -------
    f_statistic : array of shape (n_features,)
        F-statistic of each feature.
    p_values : array of shape (n_features,)
        p-value of each F-statistic.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = (2 * X[:, 0] + rng.random(1000)).astype(np.float32)
    >>> F, p = mojolearn.f_regression(X, y)
    """
    sc, pv, _ = _pearson(X, y, center, force_finite)
    return sc, pv


def r_regression(X, y, *, center=True, force_finite=True):
    """Compute the Pearson correlation of each feature with a continuous target.

    Follows scikit-learn's ``r_regression``. Computation is float32.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Target values.
    center : bool, default True
        Center ``X`` and ``y`` before computing the correlation.
    force_finite : bool, default True
        A constant feature or target gives a correlation of 0; with False it gives NaN.

    Returns
    -------
    correlation : array of shape (n_features,)
        Pearson correlation of each feature with ``y``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = X[:, 0] + rng.random(1000).astype(np.float32)
    >>> r = mojolearn.r_regression(X, y)
    """
    return _pearson(X, y, center, force_finite)[2]


class SelectKBest(_SelectorMixin):
    """Select the features with the ``k`` highest scores.

    Follows scikit-learn's ``SelectKBest``. Fit calls ``score_func(X, y)`` and keeps the ``k`` columns
    with the highest scores. NaN scores rank lowest. Among tied scores the later column is kept, as with
    scikit-learn's stable sort. ``score_func`` may be :func:`f_classif`, :func:`chi2`,
    :func:`f_regression`, or any callable returning scores or a ``(scores, pvalues)`` pair.

    Parameters
    ----------
    score_func : callable, default f_classif
        Function of ``(X, y)`` returning the scores.
    k : int or 'all', default 10
        Number of columns to keep, ``0 <= k <= n_features`` (``ValueError`` otherwise), or ``'all'``.

    Attributes
    ----------
    scores_ : array of shape (n_features,)
        Score of each column.
    pvalues_ : array of shape (n_features,) or None
        p-value of each column, when ``score_func`` returns them.
    ``n_features_in_`` : int
        Number of columns seen in fit.
    numeric_mode_ : str
        Numeric mode at fit time; transform runs in the same mode.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 20), dtype=np.float32)
    >>> y = (X[:, 3] > 0.5).astype(np.int64)
    >>> Xt = mojolearn.SelectKBest(mojolearn.f_classif, k=5).fit_transform(X, y)
    """
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


# Brute-force neighbour scans on the device; the tie-breaking noise is drawn
# from random_state by splitmix64 (sklearn draws numpy's). See _mutual_info.
def mutual_info_classif(X, y, *, discrete_features="auto", n_neighbors=3, copy=True, random_state=None,
                        n_jobs=None):
    """Estimate the mutual information between each feature and a discrete target.

    Follows scikit-learn's ``mutual_info_classif`` for dense ``X``. Continuous features use the k-nearest
    neighbor estimator of Ross (2014) against the classes; discrete features use the mutual information
    of the contingency table. As in scikit-learn, continuous features are scaled and a small noise is
    added to break ties. Computation is float32.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Class labels.
    discrete_features : 'auto', bool or array-like, default 'auto'
        Which features are discrete: ``'auto'`` (none, for dense ``X``), a bool for all features, a
        boolean mask, or column indices.
    n_neighbors : int, default 3
        Number of neighbors for the continuous estimator, from 1 to 32 (``NotImplementedError``
        otherwise).
    copy : bool, default True
        Accepted for scikit-learn compatibility.
    random_state : int or None, default None
        Seed for the tie-breaking noise. None uses seed 0. The noise uses mojolearn's own generator, so
        results differ slightly from scikit-learn's for the same seed.
    n_jobs : int, default None
        Accepted for scikit-learn compatibility.

    Returns
    -------
    mi : array of shape (n_features,)
        Estimated mutual information of each feature, in nats.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 5), dtype=np.float32)
    >>> y = (X[:, 0] > 0.5).astype(np.int64)
    >>> mi = mojolearn.mutual_info_classif(X, y, random_state=0)
    """
    return _mutual_info(X, y, True, discrete_features, n_neighbors, random_state)


# Brute-force neighbour scans on the device; the tie-breaking noise is drawn
# from random_state by splitmix64 (sklearn draws numpy's). See _mutual_info.
def mutual_info_regression(X, y, *, discrete_features="auto", n_neighbors=3, copy=True, random_state=None,
                           n_jobs=None):
    """Estimate the mutual information between each feature and a continuous target.

    Follows scikit-learn's ``mutual_info_regression`` for dense ``X``. Continuous features use the
    k-nearest neighbor estimator of Kraskov et al. (2004); discrete features use Ross's (2014) estimator
    with the feature's categories as the classes. As in scikit-learn, continuous variables are scaled and
    a small noise is added to break ties. Computation is float32.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Target values.
    discrete_features : 'auto', bool or array-like, default 'auto'
        Which features are discrete: ``'auto'`` (none, for dense ``X``), a bool for all features, a
        boolean mask, or column indices.
    n_neighbors : int, default 3
        Number of neighbors, from 1 to 32 (``NotImplementedError`` otherwise).
    copy : bool, default True
        Accepted for scikit-learn compatibility.
    random_state : int or None, default None
        Seed for the tie-breaking noise. None uses seed 0. The noise uses mojolearn's own generator, so
        results differ slightly from scikit-learn's for the same seed.
    n_jobs : int, default None
        Accepted for scikit-learn compatibility.

    Returns
    -------
    mi : array of shape (n_features,)
        Estimated mutual information of each feature, in nats.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 5), dtype=np.float32)
    >>> y = (X[:, 0] ** 2 + 0.1 * rng.random(1000)).astype(np.float32)
    >>> mi = mojolearn.mutual_info_regression(X, y, random_state=0)
    """
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
    """Feature ranking with recursive feature elimination.

    Follows scikit-learn's ``RFE``. The estimator is fitted on the current features, the ``step``
    weakest are removed, and the process repeats until ``n_features_to_select`` remain. Importance is the
    squared ``coef_`` (summed over rows when 2-D) or ``feature_importances_``. Ties are broken by a
    stable sort: of two equally weak columns, the lower index is removed first (scikit-learn leaves the
    order unspecified).

    The estimator is cloned with ``type(estimator)(**estimator.get_params())`` and fitted on float32
    data. ``predict``, ``predict_proba``, ``decision_function`` and ``score`` apply the final estimator
    to the selected columns.

    Parameters
    ----------
    estimator : object
        Estimator with ``fit`` that exposes ``coef_`` or ``feature_importances_`` (``ValueError``
        otherwise, with ``importance_getter='auto'``).
    n_features_to_select : int or float, default None
        Number of features to keep, or a fraction of them when a float in ``(0, 1)``. None keeps half.
    step : int or float, default 1
        Number of features removed per iteration, or a fraction of the features when a float in
        ``(0, 1)``. Must be greater than 0.
    verbose : int, default 0
        Accepted for scikit-learn compatibility.
    importance_getter : 'auto', str or callable, default 'auto'
        ``'auto'`` uses ``coef_`` or ``feature_importances_``; a dotted attribute path or a callable
        taking the estimator picks the importances instead (they are squared).

    Attributes
    ----------
    estimator_ : object
        Estimator fitted on the selected features.
    n_features_ : int
        Number of selected features.
    support_ : list of bool
        Mask of selected features.
    ranking_ : array of int64
        Rank of each feature; selected features have rank 1.
    ``n_features_in_`` : int
        Number of columns seen in fit.
    numeric_mode_ : str
        Numeric mode at fit time; transform runs in the same mode.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 10), dtype=np.float32)
    >>> y = (X[:, 0] + X[:, 1] > 1).astype(np.int64)
    >>> rfe = mojolearn.RFE(mojolearn.LinearDiscriminantAnalysis(), n_features_to_select=3).fit(X, y)
    >>> rfe.support_
    """
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
    """Complement naive Bayes classifier, suited to imbalanced text data.

    Follows scikit-learn's ``ComplementNB``. Fit learns, for each class, the smoothed feature counts of
    all the other classes and stores their negated log share (normalized when ``norm=True``). As in
    scikit-learn, the class prior is used only when there is a single class. ``X`` must be nonnegative
    (``ValueError`` otherwise). Computation is float32.

    ``fit`` accepts ``sample_weight``, which weights the counts. ``partial_fit(X, y, classes=None,
    sample_weight=None)`` adds each batch's counts to the running ones; ``classes`` is required on the
    first call.

    Parameters
    ----------
    alpha : float, default 1.0
        Additive smoothing. Must be greater than 0; ``alpha <= 0`` raises ``NotImplementedError``.
    force_alpha : bool, default True
        Accepted for scikit-learn compatibility.
    fit_prior : bool, default True
        Learn class priors from the data; when False, use a uniform prior.
    class_prior : array-like of shape (n_classes,), default None
        Given class priors (nonnegative); overrides ``fit_prior``.
    norm : bool, default False
        Normalize the weights of each class.

    Attributes
    ----------
    classes_ : array of shape (n_classes,)
        Class labels.
    class_count_ : array of shape (n_classes,)
        Number (or weighted number) of training rows per class.
    class_log_prior_ : array of shape (n_classes,)
        Log prior of each class.
    feature_count_ : array of shape (n_classes, n_features)
        Sum of each feature per class.
    feature_log_prob_ : array of shape (n_classes, n_features)
        Class weights computed from the complement counts.
    ``n_features_in_`` : int
        Number of columns seen in fit.
    numeric_mode_ : str
        Numeric mode the estimator was fitted in; prediction runs in the same mode.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.integers(0, 5, (1000, 20)).astype(np.float32)
    >>> y = rng.integers(0, 3, 1000)
    >>> model = mojolearn.ComplementNB().fit(X, y)
    """
    # scikit-learn tags, read by `_mode.sklearn_tags`
    _sklearn_input_tags = {"positive_only": True}
    _parameters = ("alpha", "force_alpha", "fit_prior", "class_prior", "norm")
    _csr_ok = True
    _needs_min = True

    def __init__(self, *, alpha=1.0, force_alpha=True, fit_prior=True, class_prior=None, norm=False):
        self.alpha = alpha
        self.force_alpha = force_alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        self.norm = norm


# partial_fit keeps n_categories_ as the accumulated counts' width; sklearn
# narrows it to the last batch's (naive_bayes/NOT_IMPLEMENTED.tsv).
class CategoricalNB(_DiscreteNB):
    """Naive Bayes classifier for categorical features.

    Follows scikit-learn's ``CategoricalNB``. Each column of ``X`` holds category indices ``0, 1, ...``.
    Fit counts each category per class and feature, smooths the counts by ``alpha`` and stores their log
    share of the class's rows. Negative values raise ``ValueError`` in fit; at prediction, a category
    index outside the fitted range raises ``IndexError``, as in scikit-learn. Computation is float32.

    ``fit`` accepts ``sample_weight``, which weights the counts. ``partial_fit(X, y, classes=None,
    sample_weight=None)`` adds each batch's category counts to the running ones, widening them as new
    categories appear; ``classes`` is required on the first call. After ``partial_fit``,
    ``n_categories_`` is the width of the accumulated counts (scikit-learn narrows it to the last
    batch's).

    Parameters
    ----------
    alpha : float, default 1.0
        Additive smoothing. Must be greater than 0; ``alpha <= 0`` raises ``NotImplementedError``.
    force_alpha : bool, default True
        Accepted for scikit-learn compatibility.
    fit_prior : bool, default True
        Learn class priors from the data; when False, use a uniform prior.
    class_prior : array-like of shape (n_classes,), default None
        Given class priors (nonnegative); overrides ``fit_prior``.
    min_categories : int or array-like of shape (n_features,), default None
        Minimum number of categories per feature; the count of categories is the larger of this and the
        largest index seen plus one.

    Attributes
    ----------
    classes_ : array of shape (n_classes,)
        Class labels.
    class_count_ : array of shape (n_classes,)
        Number (or weighted number) of training rows per class.
    class_log_prior_ : array of shape (n_classes,)
        Log prior of each class.
    category_count_ : list of arrays
        Per feature, an array of shape (n_classes, n_categories) of category counts.
    feature_log_prob_ : list of arrays
        Per feature, an array of shape (n_classes, n_categories) of smoothed log probabilities.
    n_categories_ : array of int64
        Number of categories of each feature.
    ``n_features_in_`` : int
        Number of columns seen in fit.
    numeric_mode_ : str
        Numeric mode the estimator was fitted in; prediction runs in the same mode.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.integers(0, 4, (1000, 6)).astype(np.float32)
    >>> y = rng.integers(0, 2, 1000)
    >>> model = mojolearn.CategoricalNB().fit(X, y)
    """
    # scikit-learn tags, read by `_mode.sklearn_tags`
    _sklearn_input_tags = {"positive_only": True, "categorical": True}
    _parameters = ("alpha", "force_alpha", "fit_prior", "class_prior", "min_categories")

    def __init__(self, *, alpha=1.0, force_alpha=True, fit_prior=True, class_prior=None, min_categories=None):
        self.alpha = alpha
        self.force_alpha = force_alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        self.min_categories = min_categories

    def fit(self, X, y, sample_weight=None):
        return _CLS_CALL(self, "fit", (X, y, sample_weight), _CLS_CONTEXT)

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        return _CLS_CALL(self, "partial_fit", (X, y, classes, sample_weight), _CLS_CONTEXT)


def _c08_route(mode, bit):
    """Compile-time admission of one C08 route (1 TargetEncoder, 2 OneHot
    fit_transform, 4 Ordinal fit_transform); no data processing."""
    if mode != "identical":
        return False
    entry = _optional_prep_entry(_prep_binding(mode), "x_prep_c08_routes")
    return bool(int(entry()) & bit) if entry is not None else False


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
    categories = [pr.get(dictionary+c*n, sizes[c]) for c in range(d)]  # small-loop(d: per-feature category dictionaries): wraps fitted metadata per feature
    return categories, pr.get(inverse, (n, d))


# Import-time metadata connection; public operations hand off once to Mojo.
from ._native_maxabs_bridge import connect as _connect_maxabs_scaler
from ._native_scaler_bridge import register_subclass as _register_maxabs_scaler
_MAXABS_CALL, _MAXABS_CONTEXT = _connect_maxabs_scaler(MaxAbsScaler)


# Import-time native metadata; the public operations hand off once to Mojo.
from ._native_robust_bridge import connect as _connect_robust_scaler
from ._native_scaler_bridge import register_subclass as _register_robust_scaler
_ROBUST_CALL, _ROBUST_CONTEXT = _connect_robust_scaler(RobustScaler, _X_PREP2_QSELECT)


# Import-time connection; `_Prog.run` hands each program off once to Mojo.
from ._native_arena_prog import connect as _connect_arena_prog
_PROG_CALL, _PROG_CONTEXT = _connect_arena_prog(
    "x_prep", _Prog, _Scratch, {v: k for k, v in _OPS.items()})  # glue: the stage names of the profiling line, once at import


# Import-time connection; Binarizer, Normalizer and PolynomialFeatures hand
# each public operation off once to Mojo (bindings/prep_stateless_native.mojo).
from ._native_prep_stateless_bridge import connect as _connect_stateless
_STATELESS_CALL, _STATELESS_CONTEXT = _connect_stateless(
    {Binarizer: "binarizer", Normalizer: "normalizer", PolynomialFeatures: "polynomial",
     VarianceThreshold: "variance_threshold", _SelectorMixin: "selector", QuantileTransformer: "quantile",
     PowerTransformer: "power"})
from ._native_prep_classify_bridge import connect as _connect_classify
_CLS_CALL, _CLS_CONTEXT = _connect_classify(
    {GaussianNB: "gnb", MultinomialNB: "mnb", BernoulliNB: "bnb", ComplementNB: "cnb", CategoricalNB: "cat",
     LinearDiscriminantAnalysis: "lda", QuadraticDiscriminantAnalysis: "qda"}, _PROG_CONTEXT)
