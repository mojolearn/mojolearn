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
    absmax_sign_run, lars_rows_run, lasso_rows_run, lda_rows_run, lu_aux_run, omp_rows_run, qr_r_run, r_signs_run,
    rand_gamma_run, rank_above_run, tsqr_r_run,
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


def lasso_rows_py[E: Exec](
    g: PythonObject, q: PythonObject, w: PythonObject, its: PythonObject, p: PythonObject, f: PythonObject
) raises -> PythonObject:
    # p = [n, k, max_iter, positive]; f = [alpha, tol]
    var n = _n(p, 0)
    var k = _n(p, 1)
    var max_iter = _n(p, 2)
    var positive = Int(py=p[3]) != 0
    var alpha = Float32(Float64(py=f[0]))
    var tol = Float32(Float64(py=f[1]))
    var pg = _f(g)
    var pq = _f(q)
    var pw = _f(w)
    var pi = _f(its)
    with GILReleased(Python()):
        lasso_rows_run[E](pg, pq, pw, pi, n, k, max_iter, positive, alpha, tol)
    return PythonObject(n)


def lu_aux_py[E: Exec](
    lu: PythonObject, piv: PythonObject, pm: PythonObject, im: PythonObject, diag: PythonObject,
    stats: PythonObject, p: PythonObject,
) raises -> PythonObject:
    """p = [n, clamp]: an LU factor's row order (pm), its inverse (im), its
    diagonal, (max |u_ii|, zero, negative pivots, swaps) and, with clamp,
    the tiny pivots floored in lu."""
    var n = _n(p, 0)
    var clamp = _n(p, 1)
    var pl = _f(lu)
    var pv = _i(piv)
    var p1 = _f(pm)
    var p2 = _f(im)
    var pd = _f(diag)
    var ps = _f(stats)
    with GILReleased(Python()):
        lu_aux_run[E](pl, pv, p1, p2, pd, ps, n, clamp)
    return PythonObject(n)


def lars_rows_py[E: Exec](
    g: PythonObject, q: PythonObject, w: PythonObject, na: PythonObject, p: PythonObject
) raises -> PythonObject:
    """p = [n, k, m, nnz]: sparse_encode 'lars' on the Gram G (k x k) and Q =
    X D^T (n x k), m the samples of each row's problem (x_decomp/cells.mojo
    `lars_row`, one thread a row)."""
    var n = _n(p, 0)
    var k = _n(p, 1)
    var m = _n(p, 2)
    var nnz = _n(p, 3)
    var pg = _f(g)
    var pq = _f(q)
    var pw = _f(w)
    var pn = _f(na)
    with GILReleased(Python()):
        lars_rows_run[E](pg, pq, pw, pn, n, k, m, nnz)
    return PythonObject(n)


def omp_rows_py[E: Exec](
    g: PythonObject, q: PythonObject, w: PythonObject, na: PythonObject, p: PythonObject
) raises -> PythonObject:
    var n = _n(p, 0)
    var k = _n(p, 1)
    var nnz = _n(p, 2)
    var pg = _f(g)
    var pq = _f(q)
    var pw = _f(w)
    var pn = _f(na)
    with GILReleased(Python()):
        omp_rows_run[E](pg, pq, pw, pn, n, k, nnz)
    return PythonObject(n)


def rand_gamma_py[E: Exec](dst: PythonObject, p: PythonObject, shape: PythonObject) raises -> PythonObject:
    var count = _n(p, 0)
    var seed = Int(py=p[1])
    var stream = Int(py=p[2])
    var a = Float32(Float64(py=shape))
    var po = _f(dst)
    with GILReleased(Python()):
        rand_gamma_run[E](po, count, seed, stream, a)
    return PythonObject(count)


def lda_rows_py[E: Exec](
    x: PythonObject, ew: PythonObject, d: PythonObject, e: PythonObject, its: PythonObject, p: PythonObject,
    f: PythonObject,
) raises -> PythonObject:
    # p = [n, k, v, max_iter]; f = [prior, tol]
    var n = _n(p, 0)
    var k = _n(p, 1)
    var v = _n(p, 2)
    var max_iter = _n(p, 3)
    var prior = Float32(Float64(py=f[0]))
    var tol = Float32(Float64(py=f[1]))
    var px = _f(x)
    var pw = _f(ew)
    var pd = _f(d)
    var pe = _f(e)
    var pi = _f(its)
    with GILReleased(Python()):
        lda_rows_run[E](px, pw, pd, pe, pi, n, k, v, max_iter, prior, tol)
    return PythonObject(n)


def dijkstra_rows_py[E: Exec](w: PythonObject, dist: PythonObject, reached: PythonObject, p: PythonObject) raises -> PythonObject:
    var n = _n(p, 0)
    var pw = _f(w)
    var pd = _f(dist)
    var pr = _f(reached)
    with GILReleased(Python()):
        E.dijkstra_rows(pw, pd, pr, n)
    return PythonObject(n)


def barycenter_rows_py[E: Exec](
    x: PythonObject, y: PythonObject, nbr: PythonObject, wt: PythonObject, flags: PythonObject, p: PythonObject,
    reg: PythonObject,
) raises -> PythonObject:
    # p = [n, ny, d, k]
    var n = _n(p, 0)
    var ny = _n(p, 1)
    var d = _n(p, 2)
    var k = _n(p, 3)
    var r = Float32(Float64(py=reg))
    var px = _f(x)
    var py_ = _f(y)
    var pn = _f(nbr)
    var pw = _f(wt)
    var pf = _f(flags)
    with GILReleased(Python()):
        E.barycenter_rows(px, py_, pn, pw, pf, n, ny, d, k, r)
    return PythonObject(n)


def als_rows_py[E: Exec](
    c: PythonObject, y: PythonObject, yty: PythonObject, x: PythonObject, flags: PythonObject, p: PythonObject,
    reg: PythonObject,
) raises -> PythonObject:
    var n = _n(p, 0)
    var m = _n(p, 1)
    var f = _n(p, 2)
    var r = Float32(Float64(py=reg))
    var pc = _f(c)
    var py_ = _f(y)
    var pg = _f(yty)
    var px = _f(x)
    var pf = _f(flags)
    with GILReleased(Python()):
        E.als_rows(pc, py_, pg, px, pf, n, m, f, r)
    return PythonObject(n)


def absmax_sign_py[E: Exec](a: PythonObject, dst: PythonObject, p: PythonObject) raises -> PythonObject:
    var n = _n(p, 0)
    var d = _n(p, 1)
    var by_col = Int(py=p[2]) != 0
    var pa = _f(a)
    var pd = _f(dst)
    var r = 0
    with GILReleased(Python()):
        r = absmax_sign_run[E](pa, pd, n, d, by_col)
    return PythonObject(r)


def qr_r_py[E: Exec](a: PythonObject, r: PythonObject, p: PythonObject) raises -> PythonObject:
    var m = _n(p, 0)
    var n = _n(p, 1)
    var pa = _f(a)
    var pr = _f(r)
    qr_r_run[E](pa, pr, m, n)
    return PythonObject(n)


def tsqr_r_py[E: Exec](a: PythonObject, b: PythonObject, r: PythonObject, p: PythonObject) raises -> PythonObject:
    """r (n x n, n = d + nrhs) = R of the blocked TSQR of [a | b] (a m x d,
    b m x nrhs, row major; b is read only when nrhs > 0). p = [m, d, nrhs,
    keep]: keep != 0 holds the factorization for `tsqr_q_py`."""
    var m = _n(p, 0)
    var d = _n(p, 1)
    var nrhs = _n(p, 2)
    var keep = Int(py=p[3]) != 0
    var pa = _f(a)
    var pb = _f(b)
    var pr = _f(r)
    var n = 0
    with GILReleased(Python()):
        n = tsqr_r_run[E](pa, pb, pr, m, d, nrhs, keep)
    return PythonObject(n)


def r_signs_py(r: PythonObject, c: PythonObject, p: PythonObject) raises -> PythonObject:
    """R's diagonal made non-negative for `linalg.qr` (`r_signs_run`). p = [n]."""
    var n = _n(p, 0)
    var pr = _f(r)
    var pc = _f(c)
    r_signs_run(pr, pc, n)
    return PythonObject(0)


def rank_above_py(sv: PythonObject, p: PythonObject, rtol: PythonObject) raises -> PythonObject:
    """How many singular values at sv are > 0 and > s[0] * rtol
    (`rank_above_run`). p = [n]."""
    var n = _n(p, 0)
    var a = Int(py=sv)
    var t = Float64(py=rtol)
    var r = rank_above_run(a, n, t)
    return PythonObject(r)
