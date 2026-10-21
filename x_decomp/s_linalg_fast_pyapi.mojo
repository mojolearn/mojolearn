# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python shell of x_decomp/s_linalg_fast.mojo: the route flags and the
resident LU, upload-scan and orthonormalize doors the GPU binding registers.
Each unpacks addresses, pool ids and integers, releases the GIL and calls
one typed `*_run` entry in x_decomp/s_linalg_fast.mojo; no data work happens
here."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from x_decomp.cells import F32Ptr
from x_decomp.resident_pyapi import _id, _n
from x_decomp.s_linalg_fast import (
    dev_lu_factor_run,
    dev_lu_solve_run,
    dev_orth_ws_run,
    dev_upload_scan_run,
    s_flags_run,
)


def s_flags_py() raises -> PythonObject:
    """Bit 1 RSVD_FAST_DEVSCAN, bit 2 DECOMP_FAST_ORTH_WS, bit 4
    LU_FAST_RESIDENT, bit 8 IDN_LU_RESIDENT_PAIR (0 when none)."""
    return PythonObject(s_flags_run())


def dev_lu_factor_py(
    a: PythonObject, lu: PythonObject, piv: PythonObject, info: PythonObject, ia: PythonObject,
    ilu: PythonObject, ipv: PythonObject, p: PythonObject,
) raises -> PythonObject:
    """p = [n]. Host a (n x n) into device matrix ia; -1 when a holds a NaN
    or infinity (nothing else done). Else device matrix ilu = the LU factor
    of a copy of it (`launch_lu`), device matrix ipv (n words) its int32
    pivots, and host lu (n x n float32), piv (n int32) and info (1 float32)
    their downloads; returns 0. Waits."""
    var n = _n(p, 0)
    var src = F32Ptr(unsafe_from_address=Int(py=a))
    var plu = F32Ptr(unsafe_from_address=Int(py=lu))
    var ppv = F32Ptr(unsafe_from_address=Int(py=piv))     # int32 words, moved as 4-byte floats
    var pinf = F32Ptr(unsafe_from_address=Int(py=info))
    var ida = _id(ia)
    var idl = _id(ilu)
    var idp = _id(ipv)
    var rc = 0
    with GILReleased(Python()):
        rc = dev_lu_factor_run(src, plu, ppv, pinf, ida, idl, idp, n)
    return PythonObject(rc)


def dev_lu_solve_py(
    ilu: PythonObject, ipv: PythonObject, ia: PythonObject, b: PythonObject, x: PythonObject, p: PythonObject
) raises -> PythonObject:
    """p = [n, nrhs, refine]. Host x (n x nrhs) = the solve of A x = b (host
    b, n x nrhs) from the device LU ilu / pivots ipv (`dev_lu_factor_py`'s),
    `launch_lu_solve` as DevExec.lu_solve runs it; with refine, one LU_QFIX
    step x += LU^-1 (b - A x), A the device matrix ia, the residual in
    float-float (`lu_resid_ff_kernel`, sliced as `lu_resid_py` slices it).
    Waits."""
    var n = _n(p, 0)
    var nrhs = _n(p, 1)
    var rf = _n(p, 2)
    var pb = F32Ptr(unsafe_from_address=Int(py=b))
    var px = F32Ptr(unsafe_from_address=Int(py=x))
    var il = _id(ilu)
    var ip = _id(ipv)
    var ida = _id(ia)
    var cells = 0
    with GILReleased(Python()):
        cells = dev_lu_solve_run(pb, px, il, ip, ida, n, nrhs, rf)
    return PythonObject(cells)


def dev_upload_scan_py(id: PythonObject, addr: PythonObject, n: PythonObject) raises -> PythonObject:
    """`dev_upload_py` (host floats into the device matrix `id`), then the
    first flat index of a NaN or infinity among those n device floats, or -1
    (one device read; waits)."""
    var cnt = Int(py=n)
    var src = F32Ptr(unsafe_from_address=Int(py=addr))
    var i = _id(id)
    var r = -1
    with GILReleased(Python()):
        r = dev_upload_scan_run(i, src, cnt)
    return PythonObject(r)


def dev_orth_ws_py(a: PythonObject, dst: PythonObject, p: PythonObject) raises -> PythonObject:
    """`dev_orth_py` (dst = the orthonormalized columns of a; a unchanged)
    with `orth_ws_on_device`."""
    var m = _n(p, 0)
    var l = _n(p, 1)
    var ia = _id(a)
    var io = _id(dst)
    var cells = 0
    with GILReleased(Python()):
        cells = dev_orth_ws_run(ia, io, m, l)
    return PythonObject(cells)
