# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The decomp lane's Python shell: the PythonObject <-> typed-value doors that
`bindings/_mojolearn_x_decomp.mojo` (`*_py[DevExec]`) and
`bindings/_mojolearn_x_decomp_host.mojo` (`*_py[HostExec]`) register under the
same names. Each door unpacks addresses and integers, releases the GIL and
calls one typed entry in x_decomp/api.mojo or the kernel module it names; no
data work happens here. Every address is a host buffer the caller owns."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from x_decomp.api import (
    eigh_batch_run, eigh_run, ew_run, gemm_run, knn_select_run, lle_apply_run, lle_local_run, lu_solve_run, rand_run,
    sqdist_run, svd_run, trisolve_run,
)
from x_decomp.cells import F32Ptr, I32Ptr
from x_decomp.exec_trait import Exec
from x_decomp.moves import F64Ptr

def _f(addr: PythonObject) raises -> F32Ptr:
    var a = Int(py=addr)
    if a == 0:
        raise Error("x_decomp: null float32 buffer address")
    return F32Ptr(unsafe_from_address=a)


def _i(addr: PythonObject) raises -> I32Ptr:
    var a = Int(py=addr)
    if a == 0:
        raise Error("x_decomp: null int32 buffer address")
    return I32Ptr(unsafe_from_address=a)


def _n(p: PythonObject, i: Int) raises -> Int:
    var v = Int(py=p[i])
    if v < 0 or v > 2147483647:
        raise Error("x_decomp: dimension dst of range")
    return v


def _d(addr: PythonObject) raises -> F64Ptr:
    var a = Int(py=addr)
    if a == 0:
        raise Error("x_decomp: null float64 buffer address")
    return F64Ptr(unsafe_from_address=a)


def gemm_py[E: Exec](a: PythonObject, b: PythonObject, c: PythonObject, p: PythonObject) raises -> PythonObject:
    var m = _n(p, 0)
    var k = _n(p, 1)
    var n = _n(p, 2)
    var pa = _f(a)
    var pb = _f(b)
    var pc = _f(c)
    var ta = Int(py=p[3]) != 0
    var tb = Int(py=p[4]) != 0
    var r = 0
    with GILReleased(Python()):
        r = gemm_run[E](pa, pb, pc, m, k, n, ta, tb)
    return PythonObject(r)


def ew_py[E: Exec](
    a: PythonObject, b: PythonObject, c: PythonObject, dst: PythonObject, p: PythonObject, s: PythonObject
) raises -> PythonObject:
    # p = [op, count, d, lb, bm, lc, cm]
    var op = Int(py=p[0])
    var count = _n(p, 1)
    var d = _n(p, 2)
    var lb = _n(p, 3)
    var bm = Int(py=p[4])
    var lc = _n(p, 5)
    var cm = Int(py=p[6])
    var sv = Float32(Float64(py=s))
    var pa = _f(a)
    var pb = _f(b)
    var pc = _f(c)
    var po = _f(dst)
    with GILReleased(Python()):
        ew_run[E](op, pa, pb, lb, bm, pc, lc, cm, po, count, d, sv)
    return PythonObject(count)


def colsum_py[E: Exec](a: PythonObject, dst: PythonObject, p: PythonObject) raises -> PythonObject:
    var n = _n(p, 0)
    var d = _n(p, 1)
    var pa = _f(a)
    var po = _f(dst)
    with GILReleased(Python()):
        E.colsum(pa, po, n, d)
    return PythonObject(d)


def rowsum_py[E: Exec](a: PythonObject, dst: PythonObject, p: PythonObject) raises -> PythonObject:
    var n = _n(p, 0)
    var d = _n(p, 1)
    var pa = _f(a)
    var po = _f(dst)
    with GILReleased(Python()):
        E.rowsum(pa, po, n, d)
    return PythonObject(n)


def sqdist_py[E: Exec](a: PythonObject, b: PythonObject, dst: PythonObject, p: PythonObject) raises -> PythonObject:
    var na = _n(p, 0)
    var nb = _n(p, 1)
    var d = _n(p, 2)
    # optional: the distance kind (x_decomp/cells.mojo PD_*; 0 = squared
    # Euclidean) and the Minkowski p, a float rounded once to float32
    var kind = _n(p, 3) if len(p) > 3 else 0
    var pw = Float32(Float64(py=p[4])) if len(p) > 4 else Float32(2)
    var pa = _f(a)
    var pb = _f(b)
    var po = _f(dst)
    var r = 0
    with GILReleased(Python()):
        r = sqdist_run[E](pa, pb, po, na, nb, d, kind, pw)
    return PythonObject(r)


def rand_py[E: Exec](dst: PythonObject, p: PythonObject) raises -> PythonObject:
    var count = _n(p, 0)
    var seed = Int(py=p[1])
    var stream = Int(py=p[2])
    var kind = Int(py=p[3])
    var po = _f(dst)
    with GILReleased(Python()):
        rand_run[E](po, count, seed, stream, kind)
    return PythonObject(count)


def lu_py[E: Exec](a: PythonObject, piv: PythonObject, info: PythonObject, p: PythonObject) raises -> PythonObject:
    var n = _n(p, 0)
    var pa = _f(a)
    var pp = _i(piv)
    var pi = _f(info)
    with GILReleased(Python()):
        E.lu(pa, pp, pi, n)
    return PythonObject(n)


def trisolve_py[E: Exec](lu: PythonObject, idx: PythonObject, src: PythonObject, dst: PythonObject, p: PythonObject) raises -> PythonObject:
    """p = [n, nrhs, trans]: `trisolve_serial` (x_decomp/cells.mojo)."""
    var n = _n(p, 0)
    var nrhs = _n(p, 1)
    var trans = _n(p, 2)
    var pl = _f(lu)
    var pi = _f(idx)
    var ps = _f(src)
    var pd = _f(dst)
    with GILReleased(Python()):
        trisolve_run[E](pl, pi, ps, pd, n, nrhs, trans)
    return PythonObject(n)


def knn_select_py[E: Exec](dmat: PythonObject, dist: PythonObject, idx: PythonObject, p: PythonObject) raises -> PythonObject:
    """p = [n, m, k, exclude_self]: `knn_select_row` for every row."""
    var n = _n(p, 0)
    var m = _n(p, 1)
    var k = _n(p, 2)
    var ex = _n(p, 3)
    var pd = _f(dmat)
    var ps = _f(dist)
    var pi = _f(idx)
    with GILReleased(Python()):
        knn_select_run[E](pd, ps, pi, n, m, k, ex)
    return PythonObject(n)


def lu_solve_py[E: Exec](lu: PythonObject, piv: PythonObject, b: PythonObject, p: PythonObject) raises -> PythonObject:
    var n = _n(p, 0)
    var nrhs = _n(p, 1)
    var trans = _n(p, 2) if len(p) > 2 else 0
    var pl = _f(lu)
    var pp = _i(piv)
    var pb = _f(b)
    with GILReleased(Python()):
        lu_solve_run[E](pl, pp, pb, n, nrhs, trans)
    return PythonObject(n)


def chol_py[E: Exec](a: PythonObject, info: PythonObject, p: PythonObject) raises -> PythonObject:
    var n = _n(p, 0)
    var pa = _f(a)
    var pi = _f(info)
    with GILReleased(Python()):
        E.chol(pa, pi, n)
    return PythonObject(n)


def eigh_py[E: Exec](a: PythonObject, w: PythonObject, v: PythonObject, p: PythonObject) raises -> PythonObject:
    var n = _n(p, 0)
    # p[1] (optional): numpy's UPLO, 1 the lower triangle, 2 the upper, 0
    # (absent) the whole matrix
    var uplo = _n(p, 1) if len(p) > 1 else 0
    var pa = _f(a)
    var pw = _f(w)
    var pv = _f(v)
    eigh_run[E](pa, pw, pv, n, uplo)
    return PythonObject(n)


def lle_apply_py[E: Exec](
    wb: PythonObject, idx: PythonObject, emb: PythonObject, dst: PythonObject, p: PythonObject
) raises -> PythonObject:
    """p = [nq, nf, nn, nc]: LLE transform's out (nq x nc) = W E[idx]."""
    var nq = _n(p, 0)
    var nf = _n(p, 1)
    var nn = _n(p, 2)
    var nc = _n(p, 3)
    var pw = _f(wb)
    var pi = _f(idx)
    var pe = _f(emb)
    var po = _f(dst)
    with GILReleased(Python()):
        lle_apply_run[E](pw, pi, pe, po, nq, nf, nn, nc)
    return PythonObject(nq)


def lle_local_py[E: Exec](
    x: PythonObject, idx: PythonObject, b: PythonObject, p: PythonObject, f: PythonObject
) raises -> PythonObject:
    """LocallyLinearEmbedding's stacked factor (x_decomp/lle_local.mojo).
    p = [method (0 ltsa, 1 hessian, 2 modified), n, d, nn, nc]; f = [tol]
    (hessian_tol / modified_tol). idx (n x nn) the neighbor indices as exact
    floats; b zeroed, n nn x n (n (nn - 1 - nc) x n for hessian)."""
    var method = _n(p, 0)
    var n = _n(p, 1)
    var d = _n(p, 2)
    var nn = _n(p, 3)
    var nc = _n(p, 4)
    var tol = Float32(Float64(py=f[0]))
    var px = _f(x)
    var pi = _f(idx)
    var pb = _f(b)
    with GILReleased(Python()):
        lle_local_run[E](px, pi, pb, method, n, d, nn, nc, tol)
    return PythonObject(n)


def eigh_batch_py[E: Exec](a: PythonObject, w: PythonObject, v: PythonObject, p: PythonObject) raises -> PythonObject:
    """p = [batch, n]: `batch` n x n symmetric problems stacked in `a`; w
    (batch x n, ascending) and v (batch x n x n, vectors in columns), each
    the words `eigh` gives it (x_decomp/rr_batch.mojo)."""
    var batch = _n(p, 0)
    var n = _n(p, 1)
    var pa = _f(a)
    var pw = _f(w)
    var pv = _f(v)
    with GILReleased(Python()):
        eigh_batch_run[E](pa, pw, pv, batch, n)
    return PythonObject(batch)


def cd_rows_py[E: Exec](
    w: PythonObject, hht: PythonObject, xht: PythonObject, perm: PythonObject, viol: PythonObject, p: PythonObject
) raises -> PythonObject:
    var n = _n(p, 0)
    var k = _n(p, 1)
    var pw = _f(w)
    var ph = _f(hht)
    var px = _f(xht)
    var pp = _i(perm)
    var pv = _f(viol)
    with GILReleased(Python()):
        E.cd_rows(pw, ph, px, pp, pv, n, k)
    return PythonObject(n)


def orth_py[E: Exec](a: PythonObject, p: PythonObject) raises -> PythonObject:
    var m = _n(p, 0)
    var l = _n(p, 1)
    var pa = _f(a)
    with GILReleased(Python()):
        E.orth(pa, m, l)
    return PythonObject(l)


def orth_diag_py[E: Exec](a: PythonObject, diag: PythonObject, p: PythonObject) raises -> PythonObject:
    """`orth` in place, and diag (l floats) = the product of the two passes'
    R diagonals (lane neural-pass17)."""
    var m = _n(p, 0)
    var l = _n(p, 1)
    var pa = _f(a)
    var pd = _f(diag)
    with GILReleased(Python()):
        E.orth_diag(pa, m, l, pd)
    return PythonObject(l)


def svd_py[E: Exec](a: PythonObject, s: PythonObject, v: PythonObject, p: PythonObject) raises -> PythonObject:
    var m = _n(p, 0)
    var n = _n(p, 1)
    var pa = _f(a)
    var ps = _f(s)
    var pv = _f(v)
    svd_run[E](pa, ps, pv, m, n)
    return PythonObject(n)
