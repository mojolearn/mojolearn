# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The host column's Lanczos steps (lane cpu2-l8-decomp, 2026-10-04): the
twin of x_decomp/lanczos_dev.mojo `dev_lanczos_py` on host buffers, the
same products through HostExec's gemm (the words of the device's
launch_gemm) and the same scalar steps (x_decomp/lanczos_step.mojo), in the
same order. Registered as `x_decomp_lanczos` by the host binding."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from x_decomp.cells import F32Ptr
from x_decomp.host import HostExec
from x_decomp.lanczos_step import lz_alpha, lz_beta, lz_inv, lz_q, lz_w


def lanczos_host(pa: F32Ptr, pq: F32Ptr, pab: F32Ptr, n: Int, j0: Int, j1: Int, cap: Int) raises:
    """Steps j0 .. j1-1: pa n x n, pq (cap + 1) x n (rows 0 .. j0 hold
    q_0 .. q_j0), pab 2 cap floats (alphas, then betas)."""
    var scratch = List[Float32](length=3 * n + cap + 1, fill=Float32(0))
    var pw = F32Ptr(unsafe_from_address=Int(scratch.unsafe_ptr()))
    var pt = pw + n
    var pd = pw + 2 * n
    var pc = pw + 2 * n + 1
    for j in range(j0, j1):
        HostExec.gemm(pa, pq + j * n, pw, n, n, 1, False, False)
        for h in range(2):
            HostExec.gemm(pq, pw, pc, j + 1, n, 1, False, False)
            HostExec.gemm(pq, pc, pt, n, j + 1, 1, True, False)
            for i in range(n):
                pw.unsafe_store(i, lz_w(pw.unsafe_load(i), pt.unsafe_load(i)))
            pab.unsafe_store(j, lz_alpha(pab.unsafe_load(j), pc.unsafe_load(j), h != 0))
        HostExec.gemm(pw, pw, pd, 1, n, 1, True, False)
        var b = lz_beta(pd.unsafe_load(0))
        pab.unsafe_store(cap + j, b)
        var s = lz_inv(b, pab.unsafe_load(j))
        var qn = pq + (j + 1) * n
        for i in range(n):
            qn.unsafe_store(i, lz_q(pw.unsafe_load(i), s))
    _ = len(scratch)        # the pointers above read it: alive to here


def lanczos_py(a: PythonObject, q: PythonObject, ab: PythonObject, p: PythonObject) raises -> PythonObject:
    """`x_decomp_dev_lanczos` on host addresses; p = [n, j0, j1, cap]."""
    var n = Int(py=p[0])
    var j0 = Int(py=p[1])
    var j1 = Int(py=p[2])
    var cap = Int(py=p[3])
    if j1 > cap or j0 > j1 or j0 < 0 or n < 1 or (cap + 1) * n > 2147483647:
        raise Error("x_decomp: lanczos steps out of range")
    var pa = F32Ptr(unsafe_from_address=Int(py=a))
    var pq = F32Ptr(unsafe_from_address=Int(py=q))
    var pab = F32Ptr(unsafe_from_address=Int(py=ab))
    with GILReleased(Python()):
        lanczos_host(pa, pq, pab, n, j0, j1, cap)
    return PythonObject(j1 - j0)
