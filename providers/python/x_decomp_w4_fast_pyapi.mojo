# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python shell of the w4 candidates (x_decomp/w4_fast.mojo): the doors
the GPU binding registers. Each door reads compile-time flags or unpacks ids
and integers, releases the GIL and calls one typed entry in
x_decomp/w4_fast.mojo; no data work happens here."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from providers.python.x_decomp_resident_pyapi import _id, _n
from x_decomp.w4_fast import LLE_FAST_DEV_LU, RSVD_FAST_DIRECT_IN, RSVD_IDN_DIRECT_IN, dev_lu_aux_run


def w4_flags_py() raises -> PythonObject:
    """Which w4 candidates this build compiled in (bit 1 LLE_FAST_DEV_LU,
    bit 2 RSVD_FAST_DIRECT_IN or RSVD_IDN_DIRECT_IN; bit 4, LLE_FAST_NULL_CANON,
    was deleted 2026-10-09)."""
    var f = 0
    comptime if LLE_FAST_DEV_LU:
        f |= 1
    comptime if RSVD_FAST_DIRECT_IN or RSVD_IDN_DIRECT_IN:
        f |= 2
    return PythonObject(f)


def dev_lu_aux_py(
    a: PythonObject,
    lu: PythonObject,
    pm: PythonObject,
    im: PythonObject,
    diag: PythonObject,
    st: PythonObject,
    p: PythonObject,
) raises -> PythonObject:
    """p = [n, clamp]. lu = the LU factor of device matrix a (a untouched;
    `launch_lu`, DevExec.lu's launches), then DevExec.lu_aux's launches on
    it: pm / im the swaps' row order and its inverse (n x 1), diag (1 x n),
    st = (max |u_ii|, zero pivots, negative pivots, swaps), and with clamp
    the pivots under eps max |u_jj| floored in lu. Waits once at the end
    (the scratch buffers are this call's). Returns n."""
    var n = _n(p, 0)
    var clamp = _n(p, 1)
    var ia = _id(a)
    var il = _id(lu)
    var ipm = _id(pm)
    var iim = _id(im)
    var idiag = _id(diag)
    var ist = _id(st)
    var res = 0
    with GILReleased(Python()):
        res = dev_lu_aux_run(ia, il, ipm, iim, idiag, ist, n, clamp)
    return PythonObject(res)
