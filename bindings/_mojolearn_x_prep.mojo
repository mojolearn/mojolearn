# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE PREP LANE'S GPU BINDING (preprocessing additions, naive Bayes and
discriminant analysis). One entry runs a program of units on the device
(x_prep/common.mojo); the host binding runs the same units on the CPU."""
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder
from checks.vendor import COMPILED_VENDOR
from checks.numerics import GLOBAL_NUMERIC_MODE
from x_prep.device import run_program_device, run_program_device_ranges, x_prep_ctx, X_PREP_STORE
from x_prep.folds import I32P, kfold_folds, strat_folds
from x_prep.user_host import F32P, F64P, ii_rows, ii_gather, ii_scatter, ii_conv
from x_prep.prep3 import PREP3_MAXABS, PREP3_SPLINE
from x_prep.fastmaxabs import maxabs_fit_direct


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


# lane/apple-fast-prep3 (FAST on Apple, each behind its own define; registered
# under the define only, so Python's `_optional_prep_entry` probe finds them in
# no other build and takes main's route there).


def maxabs_fit_direct_binding(x_addr: PythonObject, out_addr: PythonObject, ints: PythonObject) raises -> PythonObject:
    """-D MOJOLEARN_PREP3_MAXABS: MaxAbsScaler's max_abs_ then scale_ (2 d
    words at out_addr) from the n x d float32 X at x_addr (x_prep/fastmaxabs.mojo).
    ints = (n, d). Returns 2 d."""
    var xa = Int(py=x_addr)
    var oa = Int(py=out_addr)
    var n = Int(py=ints[0])
    var d = Int(py=ints[1])
    if xa == 0 or oa == 0 or n <= 0 or d <= 0 or n > (2 ** 31 - 1) // d:
        raise Error("x_prep maxabs_fit_direct: invalid buffers or shape")
    with GILReleased(Python()):
        maxabs_fit_direct(xa, n, d, oa)
    return PythonObject(2 * d)


def prep3_spline_binding() raises -> PythonObject:
    """-D MOJOLEARN_PREP3_SPLINE: present (1) in the build that takes
    SplineTransformer's prep3 route (python/mojolearn/_expansion_prep.py)."""
    return PythonObject(1)


def numeric_mode_binding() raises -> PythonObject:
    return PythonObject(Int(GLOBAL_NUMERIC_MODE))


def vendor_binding() raises -> PythonObject:
    return PythonObject(String(COMPILED_VENDOR))



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
def PyInit__mojolearn_x_prep() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_x_prep")
        m.def_function[run_binding]("x_prep_run")
        m.def_function[run_scratch_binding]("x_prep_run_scratch")
        m.def_function[run_out_binding]("x_prep_run_out")
        m.def_function[run_ranges_binding]("x_prep_run_ranges")
        m.def_function[dev_put_binding]("x_prep_dev_put")
        m.def_function[dev_free_binding]("x_prep_dev_free")
        m.def_function[dev_live_binding]("x_prep_dev_live")
        m.def_function[strat_folds_binding]("x_prep_strat_folds")
        m.def_function[kfold_folds_binding]("x_prep_kfold_folds")
        m.def_function[ii_rows_binding]("x_prep_ii_rows")
        m.def_function[ii_gather_binding]("x_prep_ii_gather")
        m.def_function[ii_scatter_binding]("x_prep_ii_scatter")
        m.def_function[ii_conv_binding]("x_prep_ii_conv")
        comptime if PREP3_MAXABS:
            m.def_function[maxabs_fit_direct_binding]("x_prep_maxabs_fit_direct")
        comptime if PREP3_SPLINE:
            m.def_function[prep3_spline_binding]("x_prep_prep3_spline")
        m.def_function[numeric_mode_binding]("x_prep_numeric_mode")
        m.def_function[vendor_binding]("x_prep_vendor")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_x_prep: ", e))
