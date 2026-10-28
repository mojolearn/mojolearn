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

from checks.numerics import GLOBAL_NUMERIC_MODE
from x_decomp.api import (
    eigh_run, ew_run, gemm_run, lle_apply_run, lle_local_run, lu_solve_run, rand_run,
    sqdist_run, svd_run, trisolve_run,
    absmax_sign_run, lars_rows_run, lasso_rows_run, lu_aux_run, omp_rows_run, r_signs_run,
    rank_above_run, tsqr_r_run,
    argmin_all_run, dsum_sq_run, gather_run, geqrf_run, idn_flags_run, lu_gesv_run, mcd_run,
    move_run, ols_tsqr_r_run, order_f_run, orgqr_run, pca_mle_rank_host_run, select_smallest_run,
    topn_desc_run, tsqr_q_run,
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


# TOMBSTONE: the dead x_decomp exports (registrations only where the body stays live in the decomp owners; see docs/TOMBSTONES.md) deleted 2026-10-26 by 25b5daca2; code recoverable at a88745022.
# Restore: git apply experiments/removed/dead_entries_x_decomp.patch


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


def idn_flags_py() raises -> PythonObject:
    """The IDENTICAL route bits (`idn_flags_run`)."""
    return PythonObject(idn_flags_run())


def ols_tsqr_r_py[E: Exec](
    a: PythonObject, b: PythonObject, r: PythonObject, mu: PythonObject, ymean: PythonObject, p: PythonObject
) raises -> PythonObject:
    """r ((d + 1) x (d + 1)) = R of the blocked TSQR of [a - mu | b - ymean]
    (a m x d, b m), mu (float32 [d]) and ymean (float64 [1]) written: X and
    y cross to the device once (Exec.ols_tsqr_factor). p = [m, d]."""
    var m = _n(p, 0)
    var d = _n(p, 1)
    var pa = _f(a)
    var pb = _f(b)
    var pr = _f(r)
    var pm = _f(mu)
    var ya = Int(py=ymean)
    var n = 0
    with GILReleased(Python()):
        n = ols_tsqr_r_run[E](pa, pb, pr, pm, ya, m, d)
    return PythonObject(n)


def lu_gesv_py[E: Exec](a: PythonObject, b: PythonObject, info: PythonObject, p: PythonObject) raises -> PythonObject:
    """b (n x nrhs) = the solution of a x = b through `lu` then `lu_solve`
    with the factor resident (Exec.lu_gesv); info (1) is `lu`'s. p = [n, nrhs]."""
    var n = _n(p, 0)
    var nrhs = _n(p, 1)
    var pa = _f(a)
    var pb = _f(b)
    var pi = _f(info)
    with GILReleased(Python()):
        lu_gesv_run[E](pa, pb, pi, n, nrhs)
    return PythonObject(n)


def tsqr_q_py[E: Exec](c: PythonObject, q: PythonObject, p: PythonObject) raises -> PythonObject:
    """q (m x k) = Q c (c n x k) for the factorization `tsqr_r_py` kept,
    which is then released; p = [m, n, k], k == 0 releases it only."""
    var m = _n(p, 0)
    var n = _n(p, 1)
    var k = _n(p, 2)
    var pc = _f(c)
    var pq = _f(q)
    with GILReleased(Python()):
        tsqr_q_run[E](pc, pq, m, n, k)
    return PythonObject(k)


def geqrf_py[E: Exec](a: PythonObject, tau: PythonObject, p: PythonObject) raises -> PythonObject:
    """In place: a (m x n, row major) becomes geqrf's factored form, tau
    (min(m, n)) its scalars."""
    var m = _n(p, 0)
    var n = _n(p, 1)
    var pa = _f(a)
    var pt = _f(tau)
    var r = 0
    with GILReleased(Python()):
        r = geqrf_run[E](pa, pt, m, n)
    return PythonObject(r)


def orgqr_py[E: Exec](h: PythonObject, tau: PythonObject, q: PythonObject, p: PythonObject) raises -> PythonObject:
    """q (m x qc) = the first qc columns of H_0 ... H_{kk-1} from geqrf's
    factored h (m x n) and tau (kk)."""
    var m = _n(p, 0)
    var n = _n(p, 1)
    var kk = _n(p, 2)
    var qc = _n(p, 3)
    var ph = _f(h)
    var pt = _f(tau)
    var pq = _f(q)
    with GILReleased(Python()):
        orgqr_run[E](ph, pt, pq, m, n, kk, qc)
    return PythonObject(qc)


def mcd_py[E: Exec](
    x: PythonObject, loc: PythonObject, cov: PythonObject, sup: PythonObject, dist: PythonObject,
    p: PythonObject,
) raises -> PythonObject:
    """MinCovDet's fast_mcd (x_decomp/mcd.mojo): x (n x d) in; location
    (d), covariance (d x d), support (n int32 0/1) and distances (n) out.
    p = [n, d, h, seed, n_sub, n_ss, h_sub, n_trials, n_m, h_m, n_best_m]."""
    var q = List[Int]()
    q.append(Int(py=p[0]))
    q.append(Int(py=p[1]))
    q.append(Int(py=p[2]))
    q.append(Int(py=p[3]))
    q.append(Int(py=p[4]))
    q.append(Int(py=p[5]))
    q.append(Int(py=p[6]))
    q.append(Int(py=p[7]))
    q.append(Int(py=p[8]))
    q.append(Int(py=p[9]))
    q.append(Int(py=p[10]))
    var n = q[0]
    var px = _f(x)
    var pl = _f(loc)
    var pc = _f(cov)
    var ps = _i(sup)
    var pd = _f(dist)
    with GILReleased(Python()):
        mcd_run[E](px, pl, pc, ps, pd, q)
    return PythonObject(n)


def gather_py(src: PythonObject, idx: PythonObject, m: PythonObject, dst: PythonObject) raises -> PythonObject:
    """dst[a] = src[idx[a]] for a < m (exact copies; x_decomp/moves.mojo)."""
    var n = Int(py=m)
    var ps = _f(src)
    var pi = _i(idx)
    var pd = _f(dst)
    with GILReleased(Python()):
        gather_run(ps, pi, n, pd)
    return PythonObject(n)


# ---- lane apple-fast-py2mojo-decomp (2026-10-03): host buffers, both bindings
def move_py(src: PythonObject, idx: PythonObject, dst: PythonObject, p: PythonObject) raises -> PythonObject:
    """x_decomp/moves.mojo `move` on host buffers; p = [op, count, a1, a2, a3,
    ist, ioff, nsrc, ndst, nidx], every index checked."""
    var op = Int(py=p[0])
    var count = _n(p, 1)
    var nsrc = _n(p, 7)
    var ndst = _n(p, 8)
    var ps = _f(src)
    var pi = _f(idx)
    var pd = _f(dst)
    var a1 = _n(p, 2)
    var a2 = _n(p, 3)
    var a3 = _n(p, 4)
    var ist = _n(p, 5)
    var ioff = _n(p, 6)
    var nidx = _n(p, 9)
    with GILReleased(Python()):
        move_run(op, ps, pi, pd, count, a1, a2, a3, ist, ioff, nsrc, ndst, nidx)
    return PythonObject(count)


def dsum_sq_py(x: PythonObject, m: PythonObject) raises -> PythonObject:
    """The binary64 sum of the squares of m float32 values in the device
    form's chunked order (`dsum_sq_host`): the host column."""
    var n = Int(py=m)
    var xa = Int(py=x)
    var t = Float64(0)
    with GILReleased(Python()):
        t = dsum_sq_run(xa, n)
    return PythonObject(t)


def order_f_py(x: PythonObject, m: PythonObject, dst: PythonObject) raises -> PythonObject:
    """The stable order of x (m float32 values) as exact floats; NaN refused."""
    var n = Int(py=m)
    order_f_run(Int(py=x), n, Int(py=dst))
    return PythonObject(n)


def select_smallest_py(x: PythonObject, p: PythonObject, sel: PythonObject, mask: PythonObject) raises -> PythonObject:
    """p = [m, h]: the h smallest of x by (value, index), ascending by index,
    as exact floats, and the int32 membership of every row."""
    var m = _n(p, 0)
    var h = _n(p, 1)
    select_smallest_run(Int(py=x), m, h, Int(py=sel), Int(py=mask))
    return PythonObject(h)


def argmin_all_py(x: PythonObject, m: PythonObject, dst: PythonObject) raises -> PythonObject:
    """The positions (exact floats) of every value equal to the minimum; their count."""
    var n = Int(py=m)
    return PythonObject(argmin_all_run(Int(py=x), n, Int(py=dst)))


def pca_mle_rank_host_py(sp: PythonObject, p: PythonObject) raises -> PythonObject:
    """The host column of `x_decomp_dev_pca_mle_rank` (x_decomp/pca_mle.mojo):
    sp = the address of d binary64 spectrum values, p = [d, n_samples]; the
    Minka MLE rank (0 when d < 2)."""
    var d = _n(p, 0)
    var n = _n(p, 1)
    return PythonObject(pca_mle_rank_host_run(Int(py=sp), d, n))


def topn_desc_py(x: PythonObject, p: PythonObject, skip: PythonObject, dst: PythonObject) raises -> PythonObject:
    """p = [m, n]: the n best positions by (-x, index), skipping the nonzero
    entries of the float32 row at `skip` (0: none); their count."""
    var m = _n(p, 0)
    var n = _n(p, 1)
    return PythonObject(topn_desc_run(Int(py=x), m, Int(py=skip), n, Int(py=dst)))


def numeric_mode_py() raises -> PythonObject:
    return PythonObject(Int(GLOBAL_NUMERIC_MODE))


def vendor_py[E: Exec]() raises -> PythonObject:
    return PythonObject(E.vendor())


# x_decomp_fast_defines (fast_defines_py) left with the dead x_decomp exports (lane dead-entries-2-0859); its
# body carried these two experiment tombstones:
# TOMBSTONE: MOJOLEARN_QR_FAST_DEV (DROPPED-slower) deleted 2026-10-09 on lane/owed-deletions-D1; code
# recoverable at b639a2bd2. Restore: git apply experiments/removed/MOJOLEARN_QR_FAST_DEV.patch.
# TOMBSTONE: MOJOLEARN_SVD_FAST_CHOLQR (DROPPED-slower) deleted 2026-10-09 on lane/owed-deletions-D1; code
# recoverable at b639a2bd2. Restore: git apply experiments/removed/MOJOLEARN_SVD_FAST_CHOLQR.patch.
