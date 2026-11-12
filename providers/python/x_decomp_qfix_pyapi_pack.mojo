# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python shell of the FAST float-float residual (x_decomp/qfix.mojo): the
doors the binding registers. Each door reads a compile-time flag or unpacks
addresses and integers, releases the GIL and calls one typed entry in
x_decomp/qfix.mojo; no data work happens here."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from x_decomp.cells import F32Ptr
from x_decomp.qfix import LU_QFIX, TSVD_QFIX, lu_resid_run


def qfix_flags_py() raises -> PythonObject:
    """Bit 2 TSVD_QFIX, bit 4 LU_QFIX (0 in IDENTICAL); bit 1 (SVD_QFIX) was
    deleted 2026-10-09 and is always 0."""
    var f = 0
    comptime if TSVD_QFIX:
        f |= 2
    comptime if LU_QFIX:
        f |= 4
    return PythonObject(f)


def lu_resid_py(a: PythonObject, x: PythonObject, b: PythonObject, r: PythonObject, p: PythonObject) raises -> PythonObject:
    """p = [n, nrhs]: host r = b - a x (float-float residual, rounded once);
    a n x n, x / b / r n x nrhs, row major host float32 buffers."""
    var n = Int(py=p[0])
    var nrhs = Int(py=p[1])
    var pa = F32Ptr(unsafe_from_address=Int(py=a))
    var px = F32Ptr(unsafe_from_address=Int(py=x))
    var pb = F32Ptr(unsafe_from_address=Int(py=b))
    var pr = F32Ptr(unsafe_from_address=Int(py=r))
    var res = 0
    with GILReleased(Python()):
        res = lu_resid_run(pa, px, pb, pr, n, nrhs)
    return PythonObject(res)
