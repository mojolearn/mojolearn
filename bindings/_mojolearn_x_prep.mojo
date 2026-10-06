from experiments.classical_identical_ideas.shared_controls import C08_DICTIONARY, C55_CLASS_GROUP, C04_LOAD_CENTER, C56_LDA_INPUT
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE PREP LANE'S GPU BINDING (preprocessing additions, naive Bayes and
discriminant analysis). One entry runs a program of units on the device
(x_prep/common.mojo); the host binding runs the same units on the CPU."""
from x_prep.cat_cls2 import CAT_CLS2_PACK, CAT_CLS2_PRESENT
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder
from checks.vendor import COMPILED_VENDOR
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from x_prep.device import run_program_device, run_program_device_ranges, x_prep_ctx, X_PREP_STORE, X_PREP_POOL_ARENA
from x_prep.folds import I32P, kfold_folds, strat_folds
from x_prep.fastnb_csr import NB_TEXT_CSR, nb_csr_fit_py, nb_csr_jll_py, IDN_NB_CSR, nb_csr_fit_int_py, nb_csr_jll_chk_py
from x_prep.blocked import IDN_NB_ONEPASS, IDN_NB_CSR_DENSE
from x_prep.blocked import IDN_STATS_BLOCKED, IDN_CLASS_ONEPASS
from x_prep.select_blocked import IDN_SELECT_BLOCKED
from x_prep.pt_blocked import IDN_PT_BLOCKED
from x_prep.host.rr_eigh_host import IDN_RR_EIGH
from x_prep.fam2 import IDN_WDRAW, IDN_PERM_DRAW, IDN_WPICK, IDN_PARTIAL_CODES, IDN_LABEL_INV
from x_prep.gram_blocked import IDN_GRAM_BLOCKED, IDN_GRAM_ROWTILE, IDN_GRAM_ROWS
from x_prep.calib import CALIB_FOLDS, CAL_ST, CAL_LS
from x_prep.py2mojo import PY2MOJO_PREP
from x_prep.proba64 import PROBA64
from x_prep.prep3 import PREP3_MAXABS, PREP3_MAXABS_POOL
from x_prep.fastmaxabs import maxabs_fit_direct

from x_prep.fastpt import PTIMPUTE_FLAGS
from x_prep.label_fast import LABEL_PRESENT, IDN_LABEL


def run_binding(arena_addr: PythonObject, arena_len: PythonObject, prog_addr: PythonObject,
                stages: PythonObject) raises -> PythonObject:
    var fa = Int(py=arena_addr)
    var n = Int(py=arena_len)
    var qa = Int(py=prog_addr)
    var s = Int(py=stages)
    if fa == 0 or qa == 0 or n < 0 or s < 0:
        raise Error("x_prep: invalid program buffers")
    with GILReleased(Python()):
        run_program_device(fa, n, qa, s)
    return PythonObject(s)


def run_scratch_binding(arena_addr: PythonObject, arena_len: PythonObject, scratch_len: PythonObject,
                        prog_addr: PythonObject, stages: PythonObject) raises -> PythonObject:
    """x_prep_run with scratch_len device-only words after the arena (lane prep-apple2)."""
    var fa = Int(py=arena_addr)
    var n = Int(py=arena_len)
    var sc = Int(py=scratch_len)
    var qa = Int(py=prog_addr)
    var s = Int(py=stages)
    if fa == 0 or qa == 0 or n < 0 or sc < 0 or s < 0:
        raise Error("x_prep: invalid program buffers")
    with GILReleased(Python()):
        run_program_device(fa, n, qa, s, sc)
    return PythonObject(s)


def run_out_binding(arena_addr: PythonObject, prog_addr: PythonObject, out_addr: PythonObject,
                    sizes: PythonObject) raises -> PythonObject:
    """x_prep_run_scratch plus one OUTPUT region after the scratch, zeroed on
    the device and copied back into the host buffer at out_addr (lane
    prep-apple2). sizes = (arena_len, scratch_len, out_len, stages)."""
    var fa = Int(py=arena_addr)
    var qa = Int(py=prog_addr)
    var oa = Int(py=out_addr)
    var n = Int(py=sizes[0])
    var sc = Int(py=sizes[1])
    var on = Int(py=sizes[2])
    var s = Int(py=sizes[3])
    if fa == 0 or qa == 0 or n < 0 or sc < 0 or on < 0 or s < 0 or (on > 0 and oa == 0):
        raise Error("x_prep: invalid program buffers")
    with GILReleased(Python()):
        run_program_device(fa, n, qa, s, sc, oa, on)
    return PythonObject(s)


def run_ranges_binding(arena_addr: PythonObject, prog_addr: PythonObject, out_addr: PythonObject,
                       sizes: PythonObject, ranges: PythonObject) raises -> PythonObject:
    """x_prep_run_out whose host arena crosses by ranges (lane py-shared,
    core/arena_io.mojo). sizes = (arena_len, scratch_len, out_len, stages);
    ranges = (ins_addr, nins, outs_addr, nouts): Int32 triples [lo, hi, src]
    and quads [lo, hi, CNT, mult] inside the host arena."""
    var fa = Int(py=arena_addr)
    var qa = Int(py=prog_addr)
    var oa = Int(py=out_addr)
    var n = Int(py=sizes[0])
    var sc = Int(py=sizes[1])
    var on = Int(py=sizes[2])
    var s = Int(py=sizes[3])
    var ia = Int(py=ranges[0])
    var ni = Int(py=ranges[1])
    var ra = Int(py=ranges[2])
    var no = Int(py=ranges[3])
    if fa == 0 or qa == 0 or n < 0 or sc < 0 or on < 0 or s < 0 or (on > 0 and oa == 0) or ni < 0 or no < 0:
        raise Error("x_prep: invalid program buffers")
    with GILReleased(Python()):
        run_program_device_ranges(fa, n, qa, s, sc, oa, on, ia, ni, ra, no)
    return PythonObject(s)


def dev_put_binding(addr: PythonObject, n_words: PythonObject) raises -> PythonObject:
    """A resident copy of n_words host words (core/device_store.mojo); its id."""
    var a = Int(py=addr)
    var n = Int(py=n_words)
    var id: Int
    with GILReleased(Python()):
        id = X_PREP_STORE.get_or_create_ptr()[].put(x_prep_ctx(), a, n)
    return PythonObject(id)


def dev_free_binding(id: PythonObject) raises -> PythonObject:
    var i = Int(py=id)
    with GILReleased(Python()):
        X_PREP_STORE.get_or_create_ptr()[].free(x_prep_ctx(), i)
    return PythonObject(None)


def dev_take_rows_binding(
    src_id: PythonObject, row_words: PythonObject, idx_addr: PythonObject, n_idx: PythonObject
) raises -> PythonObject:
    """A new resident slot: rows `idx` (n_idx host int64 words) of slot
    src_id, gathered on the device (core/device_store.mojo `take_rows`,
    lane cpu4-misc device-rows input); its id."""
    var s = Int(py=src_id)
    var w = Int(py=row_words)
    var a = Int(py=idx_addr)
    var n = Int(py=n_idx)
    var id: Int
    with GILReleased(Python()):
        id = X_PREP_STORE.get_or_create_ptr()[].take_rows(x_prep_ctx(), s, w, a, n)
    return PythonObject(id)


def dev_live_binding() raises -> PythonObject:
    return PythonObject(X_PREP_STORE.get_or_create_ptr()[].live)


def _seed(v: PythonObject) raises -> UInt64:
    """(lo, hi) 32-bit halves -> the 64-bit seed."""
    return (UInt64(Int(py=v[1])) << 32) | UInt64(Int(py=v[0]))


def strat_folds_binding(codes_addr: PythonObject, out_addr: PythonObject, ints: PythonObject,
                        seed: PythonObject) raises -> PythonObject:
    """TargetEncoder's stratified fold assignment on the host (x_prep/folds.mojo).
    ints = (n, n_classes, n_folds, shuffle); seed = (lo, hi). Returns 0, or -1
    when every class has fewer rows than n_folds."""
    var ca = Int(py=codes_addr)
    var oa = Int(py=out_addr)
    var n = Int(py=ints[0])
    if ca == 0 or oa == 0 or n < 0:
        raise Error("x_prep: invalid fold buffers")
    var r = strat_folds(I32P(unsafe_from_address=ca), n, Int(py=ints[1]), Int(py=ints[2]), _seed(seed),
                        Int(py=ints[3]) != 0, I32P(unsafe_from_address=oa))
    return PythonObject(r)


def kfold_folds_binding(out_addr: PythonObject, ints: PythonObject, seed: PythonObject) raises -> PythonObject:
    """TargetEncoder's K-fold assignment on the host (x_prep/folds.mojo).
    ints = (n, n_folds, shuffle); seed = (lo, hi)."""
    var oa = Int(py=out_addr)
    var n = Int(py=ints[0])
    if oa == 0 or n < 0:
        raise Error("x_prep: invalid fold buffers")
    kfold_folds(n, Int(py=ints[1]), _seed(seed), Int(py=ints[2]) != 0, I32P(unsafe_from_address=oa))
    return PythonObject(0)


def maxabs_pool_binding() raises -> PythonObject:
    """1 when this binary's direct MaxAbsScaler fit uses pooled buffers
    (lane/apple-fast-w4-small PREP3_MAXABS_POOL): the quality pair's reach."""
    comptime if PREP3_MAXABS_POOL:
        return PythonObject(1)
    return PythonObject(0)


# lane/apple-fast-prep3 (FAST on Apple; registered under PREP3_MAXABS only, so
# Python's `_optional_prep_entry` probe finds it in no other build and takes
# main's route there).
def maxabs_fit_direct_binding(x_addr: PythonObject, out_addr: PythonObject, ints: PythonObject) raises -> PythonObject:
    """PREP3_MAXABS (FAST + Apple default, -D MOJOLEARN_PREP3_MAXABS_OFF
    reverts; M3 A/B prep3-maxabs-istella-x-m3 121.7 -> 104.2 ms): MaxAbsScaler's max_abs_ then
    scale_ (2 d words at out_addr) from the n x d float32 X at x_addr
    (x_prep/fastmaxabs.mojo). ints = (n, d). Returns 2 d."""
    var xa = Int(py=x_addr)
    var oa = Int(py=out_addr)
    var n = Int(py=ints[0])
    var d = Int(py=ints[1])
    if xa == 0 or oa == 0 or n <= 0 or d <= 0 or n > (2 ** 31 - 1) // d:
        raise Error("x_prep maxabs_fit_direct: invalid buffers or shape")
    with GILReleased(Python()):
        maxabs_fit_direct(xa, n, d, oa)
    return PythonObject(2 * d)


def numeric_mode_binding() raises -> PythonObject:
    return PythonObject(Int(GLOBAL_NUMERIC_MODE))


def vendor_binding() raises -> PythonObject:
    return PythonObject(String(COMPILED_VENDOR))


#: lane/apple-fast-prep (2026-10-02): the encoders' category fit takes the
#: chunked per-column run scan (uniq_count / uniq_scan / uniq_write, existing
#: units) in place of the one-thread-per-column `unique_cols` on the FAST
#: tier on Apple. DEFAULT since the M3 A/B (lane/apple-fast-prep 387211293,
#: n=1, output digests identical: onehot taxi 69.7 -> 30.4 ms, ordinal taxi
#: 67.2 -> 25.3 ms). It was the env switch MOJOLEARN_X_PREP_FAST_UNIQUE=1;
#: now `x_prep_fast_unique` is registered only when this is on, the Python
#: side probes it once per binding (no env read), and
#: `-D MOJOLEARN_X_PREP_FAST_UNIQUE_OFF` restores main's path. The old env
#: name stays harmless.
comptime X_PREP_FAST_UNIQUE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_X_PREP_FAST_UNIQUE_OFF"]()
)


#: lane apple-fast-gap-kapprox2 (2026-10-03), the FAST + Apple default since
#: the M3 A/B kap2-spl-fused-{istella,taxi} (spline istella 48.1 -> 11.6 ms,
#: taxi 34.2 -> 10.8 ms, output digests identical; `-D
#: MOJOLEARN_SPLINE_FAST_FUSED_OFF` reverts): MOJOLEARN_SPLINE_FAST_FUSED registers `x_prep_spline_fused`
#: in the FAST + Apple build only. SplineTransformer.fit then allocates no
#: n*d arena block when nothing sorts (main's is never written and comes back
#: from the device unread: 64 MB at the board's 1M x 16) and takes the count /
#: min / max rows from the blocked units; fit_transform runs ONE program (X up
#: once, stats, knots, apply). Same knots, same output words.
comptime SPLINE_FAST_FUSED = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_SPLINE_FAST_FUSED_OFF"]()
)


def spline_fused_binding() raises -> PythonObject:
    """Present only under SPLINE_FAST_FUSED (the Python probe)."""
    return PythonObject(1)


def fast_unique_binding() raises -> PythonObject:
    """Lane apple-fast-prep: present only under X_PREP_FAST_UNIQUE."""
    return PythonObject(1)


def cls2_cat_binding() raises -> PythonObject:
    """lane/apple-fast-gap-cls2 (x_prep/cat_cls2.mojo): bit 1 PACK, bit 2
    PRESENT; registered only under CAT_CLS2_PACK."""
    var f = 1
    comptime if CAT_CLS2_PRESENT:
        f |= 2
    return PythonObject(f)


def target_scratch_binding() raises -> PythonObject:
    return PythonObject(1)


def label_present_binding() raises -> PythonObject:
    return PythonObject(1)


def idn_int_binding() raises -> PythonObject:
    """Lane idn-int-prep (IDENTICAL): the bits of the integer prep switches
    this binding was built with, read by python/mojolearn/_expansion_prep.py
    `_idn_int` (1 IDN_LABEL, 2 IDN_NB_ONEPASS, 4 IDN_NB_CSR, 8
    IDN_NB_CSR_DENSE: op 164 densifies a CSR input on the device); registered
    only when one is on."""
    return PythonObject(
        (1 if IDN_LABEL else 0) | (2 if IDN_NB_ONEPASS else 0) | (4 if IDN_NB_CSR else 0)
        | (8 if IDN_NB_CSR_DENSE else 0)
    )


def pool_arena_binding() raises -> PythonObject:
    """Present only in a FAST Apple build with X_PREP_POOL_ARENA on (default; absent under -D MOJOLEARN_X_PREP_POOL_ARENA_OFF). A probe for checkers."""
    return PythonObject(1)


def ptimpute_flags_binding() raises -> PythonObject:
    """Lane af-ptimpute (FAST + Apple, each switch its own define, default off):
    the bits of x_prep/fastpt.mojo PTIMPUTE_FLAGS (1 PT_COLBATCH, 2 PT_SPEC,
    4 PT_FUSED_TRANSFORM, 8 SI_ONEPASS, 16 PT_FOLD_NOX); registered only
    when one is on, so the Python layer's probe is the switch."""
    return PythonObject(PTIMPUTE_FLAGS)


def calib_folds_binding() raises -> PythonObject:
    """Lane apple-fast-meta (FAST + Apple default, -D MOJOLEARN_CALIB_GNB_FOLDS_OFF turns it off):
    the CalibratedClassifierCV(GaussianNB) program's constants [words per
    Platt problem, line-search steps]; registered only when the ops exist."""
    var out = Python.list()
    out.append(PythonObject(CAL_ST))
    out.append(PythonObject(CAL_LS))
    return out


def proba64_binding() raises -> PythonObject:
    """Lane apple-fast-q-clf (x_prep/proba64.mojo): registered only under
    PROBA64 (FAST, not -D MOJOLEARN_PROBA64_QOLD); Python's probe for staging
    `q64_softmax` (float64 predict_proba)."""
    return PythonObject(1)


def py2mojo_binding() raises -> PythonObject:
    """Lane apple-fast-py2mojo-prep: present unless -D MOJOLEARN_PY2MOJO_prep_OFF
    (x_prep/py2mojo.mojo); Python then takes its old loops."""
    return PythonObject(1)


def classical_shared_binding() raises -> PythonObject:
    # NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
    return PythonObject(Int(C08_DICTIONARY) | (Int(C55_CLASS_GROUP) << 1) | (Int(C04_LOAD_CENTER) << 2) | (Int(C56_LDA_INPUT) << 3))


def idn_fam_binding() raises -> PythonObject:
    """Lane fam-prep-metrics (IDENTICAL, device and host column alike): the
    bits of the family switches this binding was built with, read by
    python/mojolearn/_expansion_prep.py `_idn_fam` (1 IDN_STATS_BLOCKED, 2
    IDN_CLASS_ONEPASS: op 165, 4 IDN_SELECT_BLOCKED: ops 166-171, 8
    IDN_PT_BLOCKED: ops 172-176, 16 IDN_RR_EIGH: informational, the eigh
    stage's order is the binding's own); registered only when one is on."""
    return PythonObject(
        (1 if IDN_STATS_BLOCKED else 0) | (2 if IDN_CLASS_ONEPASS else 0) | (4 if IDN_SELECT_BLOCKED else 0)
        | (8 if IDN_PT_BLOCKED else 0) | (16 if IDN_RR_EIGH else 0)
    )


def idn_fam2_binding() raises -> PythonObject:
    """Lane fam2-prep-metrics (IDENTICAL, device and host column alike): the
    bits of the x_prep/fam2.mojo switches this binding was built with, read
    by python/mojolearn/_expansion_prep.py `_idn_fam2` (1 IDN_WDRAW: ops
    230-232, 2 IDN_PERM_DRAW: op 233, 4 IDN_WPICK: op 234, 8
    IDN_PARTIAL_CODES: op 235, 16 IDN_GRAM_BLOCKED: ops 236, 238-240, 32
    IDN_GRAM_ROWTILE: op 237 (candidate), 64 IDN_LABEL_INV: op 241, bits 16 and up: the Gram's rows per
    block); registered only when one is on."""
    return PythonObject(
        (1 if IDN_WDRAW else 0) | (2 if IDN_PERM_DRAW else 0) | (4 if IDN_WPICK else 0)
        | (8 if IDN_PARTIAL_CODES else 0) | (16 if IDN_GRAM_BLOCKED else 0) | (32 if IDN_GRAM_ROWTILE else 0)
        | (64 if IDN_LABEL_INV else 0) | (IDN_GRAM_ROWS << 16)
    )


@export
def PyInit__mojolearn_x_prep() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_x_prep")
        m.def_function[run_binding]("x_prep_run")
        comptime if NB_TEXT_CSR:
            # lane apple-fast-nb: FAST + Apple default (M3 A/B 186.9 -> 35.4 ms), -D MOJOLEARN_NB_TEXT_CSR_OFF reverts (x_prep/fastnb_csr.mojo)
            m.def_function[nb_csr_fit_py]("x_prep_nb_csr_fit")
            m.def_function[nb_csr_jll_py]("x_prep_nb_csr_jll")
        comptime if IDN_NB_CSR:
            # lane idn-int-prep: IDENTICAL, every vendor, -D MOJOLEARN_IDN_NB_CSR_OFF reverts (x_prep/fastnb_csr.mojo)
            m.def_function[nb_csr_fit_int_py]("x_prep_nb_csr_fit")
            m.def_function[nb_csr_jll_chk_py]("x_prep_nb_csr_jll")
        comptime if IDN_LABEL or IDN_NB_ONEPASS or IDN_NB_CSR or IDN_NB_CSR_DENSE:
            m.def_function[idn_int_binding]("x_prep_idn_int")
        comptime if IDN_STATS_BLOCKED or IDN_CLASS_ONEPASS or IDN_SELECT_BLOCKED or IDN_PT_BLOCKED or IDN_RR_EIGH:
            # lane fam-prep-metrics
            m.def_function[idn_fam_binding]("x_prep_idn_fam")
            m.def_function[classical_shared_binding]("x_prep_classical_shared")
        comptime if IDN_WDRAW or IDN_PERM_DRAW or IDN_WPICK or IDN_PARTIAL_CODES or IDN_GRAM_BLOCKED or IDN_LABEL_INV:
            # lane fam2-prep-metrics: the fam2 switches (x_prep/fam2.mojo)
            m.def_function[idn_fam2_binding]("x_prep_idn_fam2")
        m.def_function[run_scratch_binding]("x_prep_run_scratch")
        m.def_function[run_out_binding]("x_prep_run_out")
        m.def_function[run_ranges_binding]("x_prep_run_ranges")
        m.def_function[dev_put_binding]("x_prep_dev_put")
        m.def_function[dev_free_binding]("x_prep_dev_free")
        m.def_function[dev_live_binding]("x_prep_dev_live")
        m.def_function[dev_take_rows_binding]("x_prep_dev_take_rows")
        m.def_function[strat_folds_binding]("x_prep_strat_folds")
        m.def_function[kfold_folds_binding]("x_prep_kfold_folds")
        comptime if PREP3_MAXABS:
            m.def_function[maxabs_fit_direct_binding]("x_prep_maxabs_fit_direct")
        m.def_function[maxabs_pool_binding]("x_prep_maxabs_pool")
        m.def_function[numeric_mode_binding]("x_prep_numeric_mode")
        m.def_function[vendor_binding]("x_prep_vendor")
        comptime if X_PREP_FAST_UNIQUE:
            m.def_function[fast_unique_binding]("x_prep_fast_unique")
        comptime if SPLINE_FAST_FUSED:
            m.def_function[spline_fused_binding]("x_prep_spline_fused")
        # FAST+Apple default promotion candidate, source 4d1ea20b2, M3 tags
        # gap26-target-current-{taxi,istella}: 270.948 -> 203.744 ms (-24.8%),
        # 250.424 -> 179.819 ms (-28.2%); both digests identical. Quality:
        # 108 exact fitted/output arrays plus independent smoothing oracle.
        # Recorded scan/cleanup/backup windows do not overlap these jobs.
        # Default/OFF builds are owed before main merge. _OFF restores counts/downloads; old opt-in is
        # harmless. See docs/apple-fast/ab/target-scratch.md.
        comptime if GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator() and not is_defined["MOJOLEARN_TARGET_SCRATCH_OFF"]():
            m.def_function[target_scratch_binding]("x_prep_target_scratch")
        comptime if LABEL_PRESENT:
            m.def_function[label_present_binding]("x_prep_label_present")
        comptime if X_PREP_POOL_ARENA:
            m.def_function[pool_arena_binding]("x_prep_pool_arena")
        comptime if CAT_CLS2_PACK:
            m.def_function[cls2_cat_binding]("x_prep_cls2_cat")
        comptime if CALIB_FOLDS:
            m.def_function[calib_folds_binding]("x_prep_calib_folds")
        comptime if PY2MOJO_PREP:
            m.def_function[py2mojo_binding]("x_prep_py2mojo")
        comptime if PROBA64:
            m.def_function[proba64_binding]("x_prep_proba64")
        comptime if PTIMPUTE_FLAGS != 0:
            m.def_function[ptimpute_flags_binding]("x_prep_ptimpute_flags")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_x_prep: ", e))
