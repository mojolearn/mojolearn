# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU binding for `_mojolearn_x_prep`. HOST ONLY: the same units as the
device, run in a loop on the caller's arena (x_prep/host/program.mojo), with
the GPU binding's export names and address contract."""
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder
from checks.kernel_matrix import COLUMN_CPU, TARGET_COLUMN, column_name
from checks.numerics import GLOBAL_NUMERIC_MODE
from x_prep.common import X_PREP_HOST_SABOTAGE
from x_prep.host.program import run_program_host
from x_prep.user_host import F32P, F64P, I32P, ii_rows, ii_gather, ii_scatter, ii_conv


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
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_x_prep_host: ", e))
