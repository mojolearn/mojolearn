"""CPU binding for `_mojolearn_x_prep`. HOST ONLY: the same units as the
device, run in a loop on the caller's arena (x_prep/host/program.mojo), with
the GPU binding's export names and address contract."""
from experiments.classical_identical_ideas.shared_controls import C08_DICTIONARY, C55_CLASS_GROUP, C04_LDA, C56_LDA_INPUT
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632

from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder
from checks.kernel_matrix import COLUMN_CPU, TARGET_COLUMN, column_name
from checks.numerics import GLOBAL_NUMERIC_MODE
from x_prep.common import X_PREP_HOST_SABOTAGE
from x_prep.host.program import run_program_host
from x_prep.user_host import F32P, F64P, I32P, ii_rows, ii_gather, ii_scatter, ii_conv
from x_prep.folds import kfold_folds, strat_folds
from x_prep.py2mojo import PY2MOJO_PREP
from x_prep.label_fast import IDN_LABEL
from x_prep.blocked import IDN_NB_ONEPASS
from x_prep.blocked import IDN_STATS_BLOCKED, IDN_CLASS_ONEPASS
from x_prep.select_blocked import IDN_SELECT_BLOCKED
from x_prep.pt_blocked import IDN_PT_BLOCKED
from x_prep.host.rr_eigh_host import IDN_RR_EIGH
from x_prep.fam2 import IDN_WDRAW, IDN_PERM_DRAW, IDN_WPICK, IDN_PARTIAL_CODES, IDN_LABEL_INV
from x_prep.gram_blocked import IDN_GRAM_BLOCKED, IDN_GRAM_ROWTILE, IDN_GRAM_ROWS
from x_prep.proba64 import PROBA64


def run_binding(arena_addr: PythonObject, arena_len: PythonObject, prog_addr: PythonObject,
                stages: PythonObject) raises -> PythonObject:
    var fa = Int(py=arena_addr)
    var n = Int(py=arena_len)
    var qa = Int(py=prog_addr)
    var s = Int(py=stages)
    if fa == 0 or qa == 0 or n < 0 or s < 0:
        raise Error("x_prep: invalid program buffers")
    with GILReleased(Python()):
        run_program_host(fa, n, qa, s)
    return PythonObject(s)


def x_prep_host_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def x_prep_host_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def x_prep_host_column_binding() raises -> PythonObject:
    comptime assert TARGET_COLUMN == COLUMN_CPU, "x_prep host: pass -D MOJOLEARN_COLUMN_CPU"
    return PythonObject(column_name(TARGET_COLUMN))


def x_prep_host_sabotage_binding() raises -> PythonObject:
    return PythonObject(X_PREP_HOST_SABOTAGE)


def x_prep_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(Int(GLOBAL_NUMERIC_MODE))


def x_prep_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))



# lane py-misc-prep: IterativeImputer(estimator=...) host plumbing (x_prep/user_host.mojo),
# the same entries in the GPU binding and its host twin (host memory both ways).


def ii_rows_binding(mask_addr: PythonObject, rows_addr: PythonObject, ints: PythonObject) raises -> PythonObject:
    """ints = (n, dk, j, missing 0/1): rows[0:m] ascending; returns m."""
    var ma = Int(py=mask_addr)
    var ra = Int(py=rows_addr)
    var n = Int(py=ints[0])
    var dk = Int(py=ints[1])
    var j = Int(py=ints[2])
    if ma == 0 or ra == 0 or n < 0 or dk <= 0 or j < 0 or j >= dk:
        raise Error("x_prep ii_rows: invalid buffers or shape")
    return PythonObject(ii_rows(F32P(unsafe_from_address=ma), n, dk, j, Int(py=ints[3]) != 0,
                                I32P(unsafe_from_address=ra)))


def ii_gather_binding(x_addr: PythonObject, rows_addr: PythonObject, cols_addr: PythonObject, out_addr: PythonObject,
                      y_addr: PythonObject, ints: PythonObject) raises -> PythonObject:
    """ints = (dk, m, n_cols, j or -1): out = x[rows][:, cols], y = x[rows, j]."""
    var dk = Int(py=ints[0])
    var m = Int(py=ints[1])
    var nc = Int(py=ints[2])
    var j = Int(py=ints[3])
    var xa = Int(py=x_addr)
    var ra = Int(py=rows_addr)
    var ca = Int(py=cols_addr)
    var oa = Int(py=out_addr)
    var ya = Int(py=y_addr)
    if xa == 0 or ra == 0 or ca == 0 or oa == 0 or (j >= 0 and ya == 0) or dk <= 0 or m < 0 or nc < 0 or j >= dk:
        raise Error("x_prep ii_gather: invalid buffers or shape")
    var cp = I32P(unsafe_from_address=ca)
    for c in range(nc):
        if Int(cp[c]) < 0 or Int(cp[c]) >= dk:
            raise Error("x_prep ii_gather: a column is out of range")
    ii_gather(F32P(unsafe_from_address=xa), dk, I32P(unsafe_from_address=ra), m, cp, nc, j,
              F32P(unsafe_from_address=oa), F32P(unsafe_from_address=ya if j >= 0 else oa))
    return PythonObject(m)


def ii_scatter_binding(x_addr: PythonObject, rows_addr: PythonObject, v_addr: PythonObject, ints: PythonObject,
                       bounds: PythonObject) raises -> PythonObject:
    """ints = (dk, m, j, clip 0/1); bounds = (lo, hi) as Python floats."""
    var dk = Int(py=ints[0])
    var m = Int(py=ints[1])
    var j = Int(py=ints[2])
    var xa = Int(py=x_addr)
    var ra = Int(py=rows_addr)
    var va = Int(py=v_addr)
    if xa == 0 or ra == 0 or va == 0 or dk <= 0 or m < 0 or j < 0 or j >= dk:
        raise Error("x_prep ii_scatter: invalid buffers or shape")
    ii_scatter(F32P(unsafe_from_address=xa), dk, I32P(unsafe_from_address=ra), m, j, F64P(unsafe_from_address=va),
               Float64(py=bounds[0]), Float64(py=bounds[1]), Int(py=ints[3]) != 0)
    return PythonObject(m)


def ii_conv_binding(a_addr: PythonObject, b_addr: PythonObject, ints: PythonObject) raises -> PythonObject:
    """ints = (n, dk, compensated 0/1): the round's convergence measure (a Python float)."""
    var aa = Int(py=a_addr)
    var ba = Int(py=b_addr)
    var n = Int(py=ints[0])
    var dk = Int(py=ints[1])
    if aa == 0 or ba == 0 or n <= 0 or dk <= 0:
        raise Error("x_prep ii_conv: invalid buffers or shape")
    var r = Float64(0)
    with GILReleased(Python()):
        r = ii_conv(F32P(unsafe_from_address=aa), F32P(unsafe_from_address=ba), n, dk, Int(py=ints[2]) != 0)
    return PythonObject(r)


def _seed(v: PythonObject) raises -> UInt64:
    """(lo, hi) 32-bit halves -> the 64-bit seed."""
    return (UInt64(Int(py=v[1])) << 32) | UInt64(Int(py=v[0]))


def strat_folds_binding(codes_addr: PythonObject, out_addr: PythonObject, ints: PythonObject,
                        seed: PythonObject) raises -> PythonObject:
    """TargetEncoder's stratified fold assignment (x_prep/folds.mojo), the GPU
    binding's entry (lane apple-fast-py2mojo-prep: the CPU-only install ran
    Python's copy). ints = (n, n_classes, n_folds, shuffle); seed = (lo, hi)."""
    var ca = Int(py=codes_addr)
    var oa = Int(py=out_addr)
    var n = Int(py=ints[0])
    if ca == 0 or oa == 0 or n < 0:
        raise Error("x_prep: invalid fold buffers")
    var r = strat_folds(I32P(unsafe_from_address=ca), n, Int(py=ints[1]), Int(py=ints[2]), _seed(seed),
                        Int(py=ints[3]) != 0, I32P(unsafe_from_address=oa))
    return PythonObject(r)


def kfold_folds_binding(out_addr: PythonObject, ints: PythonObject, seed: PythonObject) raises -> PythonObject:
    """TargetEncoder's K-fold assignment (x_prep/folds.mojo). ints = (n, n_folds, shuffle)."""
    var oa = Int(py=out_addr)
    var n = Int(py=ints[0])
    if oa == 0 or n < 0:
        raise Error("x_prep: invalid fold buffers")
    kfold_folds(n, Int(py=ints[1]), _seed(seed), Int(py=ints[2]) != 0, I32P(unsafe_from_address=oa))
    return PythonObject(0)


def proba64_binding() raises -> PythonObject:
    """Lane apple-fast-q-clf (x_prep/proba64.mojo): registered only under
    PROBA64 (FAST, not -D MOJOLEARN_PROBA64_QOLD); Python's probe for staging
    `q64_softmax` (float64 predict_proba)."""
    return PythonObject(1)


def py2mojo_binding() raises -> PythonObject:
    """Lane apple-fast-py2mojo-prep: present unless -D MOJOLEARN_PY2MOJO_prep_OFF."""
    return PythonObject(1)


def label_present_binding() raises -> PythonObject:
    return PythonObject(1)


def idn_int_binding() raises -> PythonObject:
    """Lane idn-int-prep: bindings/_mojolearn_x_prep.mojo `idn_int_binding`'s
    bits for the host column (1 IDN_LABEL, 2 IDN_NB_ONEPASS; the CSR entry
    is the device's only)."""
    return PythonObject((1 if IDN_LABEL else 0) | (2 if IDN_NB_ONEPASS else 0))


def classical_shared_binding() raises -> PythonObject:
    # NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
    return PythonObject(Int(C08_DICTIONARY) | (Int(C55_CLASS_GROUP) << 1) | (Int(C04_LDA) << 2) | (Int(C56_LDA_INPUT) << 3))


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
def PyInit__mojolearn_x_prep_host() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_x_prep_host")
        m.def_function[x_prep_host_numeric_mode_binding]("x_prep_host_numeric_mode")
        m.def_function[x_prep_host_vendor_binding]("x_prep_host_vendor")
        m.def_function[x_prep_host_column_binding]("x_prep_host_column")
        m.def_function[x_prep_host_sabotage_binding]("x_prep_host_sabotage")
        m.def_function[run_binding]("x_prep_run")
        m.def_function[ii_rows_binding]("x_prep_ii_rows")
        m.def_function[ii_gather_binding]("x_prep_ii_gather")
        m.def_function[ii_scatter_binding]("x_prep_ii_scatter")
        m.def_function[ii_conv_binding]("x_prep_ii_conv")
        m.def_function[x_prep_numeric_mode_binding]("x_prep_numeric_mode")
        m.def_function[x_prep_vendor_binding]("x_prep_vendor")
        m.def_function[strat_folds_binding]("x_prep_strat_folds")
        m.def_function[kfold_folds_binding]("x_prep_kfold_folds")
        comptime if PY2MOJO_PREP:
            m.def_function[py2mojo_binding]("x_prep_py2mojo")
        # lane idn-int-prep: the host column runs the same programs as the
        # IDENTICAL device binding (x_prep/label_fast.mojo, x_prep/blocked.mojo)
        comptime if IDN_LABEL:
            m.def_function[label_present_binding]("x_prep_label_present")
        comptime if IDN_LABEL or IDN_NB_ONEPASS:
            m.def_function[idn_int_binding]("x_prep_idn_int")
        comptime if IDN_STATS_BLOCKED or IDN_CLASS_ONEPASS or IDN_SELECT_BLOCKED or IDN_PT_BLOCKED or IDN_RR_EIGH:
            # lane fam-prep-metrics
            m.def_function[idn_fam_binding]("x_prep_idn_fam")
            m.def_function[classical_shared_binding]("x_prep_classical_shared")
        comptime if IDN_WDRAW or IDN_PERM_DRAW or IDN_WPICK or IDN_PARTIAL_CODES or IDN_GRAM_BLOCKED or IDN_LABEL_INV:
            # lane fam2-prep-metrics: the fam2 switches (x_prep/fam2.mojo)
            m.def_function[idn_fam2_binding]("x_prep_idn_fam2")
        comptime if PROBA64:
            m.def_function[proba64_binding]("x_prep_proba64")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_x_prep_host: ", e))
