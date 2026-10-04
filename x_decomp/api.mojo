# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The decomp lane's Python entry points, written ONCE and instantiated for
both executors: `bindings/_mojolearn_x_decomp.mojo` registers `*_py[DevExec]`,
`bindings/_mojolearn_x_decomp_host.mojo` registers `*_py[HostExec]` under the
same names, so the GPU binding and the CPU host binding share the address
contract by construction. Every address is a host buffer the caller owns."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.sys.compile import is_defined

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from x_decomp.cells import F32Ptr, I32Ptr
from x_decomp.exec_trait import Exec
from x_decomp.kit import mat_from
from x_decomp.mcd import fast_mcd
from x_decomp.lda_online import lda_online_pass
from x_decomp.moves import (
    F64Ptr, PY2MOJO_DECOMP, accuracy, argmin_all, argsort_f32, dsum_sq, gather, iso_order, move_host, order_f, pca_mle_pa,
    pca_mle_terms, scatter, select_smallest, sign_labels, topn_desc, triu_nonzero,
)
from x_decomp.tsqr_core import TS_MAX_N


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


def gemm_py[E: Exec](a: PythonObject, b: PythonObject, c: PythonObject, p: PythonObject) raises -> PythonObject:
    var m = _n(p, 0)
    var k = _n(p, 1)
    var n = _n(p, 2)
    if m * n > 2147483647 or m * k > 2147483647 or k * n > 2147483647:
        raise Error("x_decomp: gemm exceeds the Int32 index bound")
    var pa = _f(a)
    var pb = _f(b)
    var pc = _f(c)
    var ta = Int(py=p[3]) != 0
    var tb = Int(py=p[4]) != 0
    with GILReleased(Python()):
        E.gemm(pa, pb, pc, m, k, n, ta, tb)
    return PythonObject(m * n)


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
    if d <= 0:
        raise Error("x_decomp: ew needs a positive row width")
    var sv = Float32(Float64(py=s))
    var pa = _f(a)
    var pb = _f(b)
    var pc = _f(c)
    var po = _f(dst)
    with GILReleased(Python()):
        E.ew(op, pa, pb, lb, bm, pc, lc, cm, po, count, d, sv)
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
    if na * nb > 2147483647:
        raise Error("x_decomp: sqdist exceeds the Int32 index bound")
    var pa = _f(a)
    var pb = _f(b)
    var po = _f(dst)
    with GILReleased(Python()):
        E.sqdist(pa, pb, po, na, nb, d, kind, pw)
    return PythonObject(na * nb)


def rand_py[E: Exec](dst: PythonObject, p: PythonObject) raises -> PythonObject:
    var count = _n(p, 0)
    var seed = UInt32(Int(py=p[1]) & 0xFFFFFFFF)
    var stream = UInt32(Int(py=p[2]) & 0xFFFFFFFF)
    var kind = Int(py=p[3])
    var po = _f(dst)
    with GILReleased(Python()):
        E.rand(po, count, seed, stream, kind)
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
    if n >= 1 << 24:
        raise Error("x_decomp: trisolve row numbers exceed float32's exact integers")
    var pl = _f(lu)
    var pi = _f(idx)
    var ps = _f(src)
    var pd = _f(dst)
    with GILReleased(Python()):
        E.trisolve(pl, pi, ps, pd, n, nrhs, trans)
    return PythonObject(n)


def knn_select_py[E: Exec](dmat: PythonObject, dist: PythonObject, idx: PythonObject, p: PythonObject) raises -> PythonObject:
    """p = [n, m, k, exclude_self]: `knn_select_row` for every row."""
    var n = _n(p, 0)
    var m = _n(p, 1)
    var k = _n(p, 2)
    var ex = _n(p, 3)
    if m >= 1 << 24:
        raise Error("x_decomp: knn_select columns exceed float32's exact integers")
    var pd = _f(dmat)
    var ps = _f(dist)
    var pi = _f(idx)
    with GILReleased(Python()):
        E.knn_select(pd, ps, pi, n, m, k, ex)
    return PythonObject(n)


def lu_solve_py[E: Exec](lu: PythonObject, piv: PythonObject, b: PythonObject, p: PythonObject) raises -> PythonObject:
    var n = _n(p, 0)
    var nrhs = _n(p, 1)
    var trans = _n(p, 2) if len(p) > 2 else 0
    if n >= 1 << 24:
        raise Error("x_decomp: lu_solve row numbers exceed float32's exact integers")
    var pl = _f(lu)
    var pp = _i(piv)
    var pb = _f(b)
    with GILReleased(Python()):
        E.lu_solve(pl, pp, pb, n, nrhs, trans)
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
    if n <= 0:
        raise Error("x_decomp: eigh needs n >= 1")
    # p[1] (optional): numpy's UPLO, 1 the lower triangle, 2 the upper, 0
    # (absent) the whole matrix; mirrored by the executor (on the device)
    var uplo = 0
    if len(p) > 1:
        uplo = _n(p, 1)
    if uplo < 0 or uplo > 2:
        raise Error("x_decomp: eigh uplo is 0, 1 (L) or 2 (U)")
    var pa = _f(a)
    var pw = _f(w)
    var pv = _f(v)
    E.eigh(pa, pw, pv, n, uplo)
    return PythonObject(n)


def lle_apply_py[E: Exec](
    wb: PythonObject, idx: PythonObject, emb: PythonObject, dst: PythonObject, p: PythonObject
) raises -> PythonObject:
    """p = [nq, nf, nn, nc]: LLE transform's out (nq x nc) = W E[idx]."""
    var nq = _n(p, 0)
    var nf = _n(p, 1)
    var nn = _n(p, 2)
    var nc = _n(p, 3)
    if nq * nc > 2147483647 or nf * nc > 2147483647 or nq * nn > 2147483647:
        raise Error("x_decomp: lle_apply exceeds the Int32 index bound")
    var pw = _f(wb)
    var pi = _f(idx)
    var pe = _f(emb)
    var po = _f(dst)
    with GILReleased(Python()):
        E.lle_apply(pw, pi, pe, po, nq, nf, nn, nc)
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
    if method < 0 or method > 2 or n < 1 or d < 1 or nn < 1 or nc < 1 or (method == 1 and nn - 1 - nc < 1):
        raise Error("x_decomp: lle_local needs method 0..2, n, d, n_neighbors, n_components >= 1 (hessian: n_neighbors > n_components + 1)")
    if n * nn * n > 2147483647 or n * nn * nn > 2147483647 or n >= 16777216:
        raise Error("x_decomp: lle_local exceeds the Int32 index bound")
    var px = _f(x)
    var pi = _f(idx)
    var pb = _f(b)
    with GILReleased(Python()):
        E.lle_local(px, pi, pb, method, n, d, nn, nc, tol)
    return PythonObject(n)


def eigh_batch_py[E: Exec](a: PythonObject, w: PythonObject, v: PythonObject, p: PythonObject) raises -> PythonObject:
    """p = [batch, n]: `batch` n x n symmetric problems stacked in `a`; w
    (batch x n, ascending) and v (batch x n x n, vectors in columns), each
    the words `eigh` gives it (x_decomp/rr_batch.mojo)."""
    var batch = _n(p, 0)
    var n = _n(p, 1)
    if n <= 0 or batch < 0:
        raise Error("x_decomp: eigh_batch needs n >= 1, batch >= 0")
    if batch * n * n > 2147483647:
        raise Error("x_decomp: eigh_batch exceeds the Int32 index bound")
    var pa = _f(a)
    var pw = _f(w)
    var pv = _f(v)
    with GILReleased(Python()):
        E.eigh_batch(pa, pw, pv, batch, n)
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
    if n <= 0 or m < n:
        raise Error("x_decomp: svd needs m >= n >= 1 (a tall matrix)")
    var pa = _f(a)
    var ps = _f(s)
    var pv = _f(v)
    E.svd(pa, m, n, ps, pv)
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
    var h = List[Float32](length=n * k if n * k > 0 else 1, fill=Float32(0))
    var ph = F32Ptr(unsafe_from_address=Int(h.unsafe_ptr()))
    with GILReleased(Python()):
        E.lasso_rows(pg, pq, pw, ph, pi, n, k, alpha, max_iter, tol, positive)
    _ = h^
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
    if n < 1 or n >= 16777216:
        raise Error("x_decomp: lu_aux needs 1 <= n < 2^24")
    var pl = _f(lu)
    var pv = _i(piv)
    var p1 = _f(pm)
    var p2 = _f(im)
    var pd = _f(diag)
    var ps = _f(stats)
    with GILReleased(Python()):
        E.lu_aux(pl, pv, p1, p2, pd, ps, n, clamp)
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
    if n * (k * k + 7 * k) > 2147483647:
        raise Error("x_decomp: lars_rows exceeds the Int32 index bound")
    var pg = _f(g)
    var pq = _f(q)
    var pw = _f(w)
    var pn = _f(na)
    with GILReleased(Python()):
        E.lars_rows(pg, pq, pw, pn, n, k, m, nnz)
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
    var per = k * k + 3 * k
    var s = List[Float32](length=n * per if n * per > 0 else 1, fill=Float32(0))
    var ps = F32Ptr(unsafe_from_address=Int(s.unsafe_ptr()))
    with GILReleased(Python()):
        E.omp_rows(pg, pq, pw, ps, pn, n, k, nnz)
    _ = s^
    return PythonObject(n)


def rand_gamma_py[E: Exec](dst: PythonObject, p: PythonObject, shape: PythonObject) raises -> PythonObject:
    var count = _n(p, 0)
    var seed = UInt32(Int(py=p[1]) & 0xFFFFFFFF)
    var stream = UInt32(Int(py=p[2]) & 0xFFFFFFFF)
    var a = Float32(Float64(py=shape))
    if not (a >= Float32(1)):
        raise Error("x_decomp: the gamma sampler takes shape >= 1")
    var po = _f(dst)
    with GILReleased(Python()):
        E.rand_gamma(po, count, seed, stream, a)
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
    var s = List[Float32](length=n * (v + k) if n > 0 else 1, fill=Float32(0))
    var ps = F32Ptr(unsafe_from_address=Int(s.unsafe_ptr()))
    with GILReleased(Python()):
        E.lda_rows(px, pw, pd, pe, ps, pi, n, k, v, prior, max_iter, tol)
    _ = s^
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
    with GILReleased(Python()):
        E.absmax_sign(pa, pd, n, d, by_col)
    return PythonObject(d if by_col else n)


def qr_r_py[E: Exec](a: PythonObject, r: PythonObject, p: PythonObject) raises -> PythonObject:
    var m = _n(p, 0)
    var n = _n(p, 1)
    if n <= 0 or m < n:
        raise Error("x_decomp: qr_r needs m >= n >= 1")
    var pa = _f(a)
    var pr = _f(r)
    E.qr_r(pa, m, n, pr)
    return PythonObject(n)


def tsqr_r_py[E: Exec](a: PythonObject, b: PythonObject, r: PythonObject, p: PythonObject) raises -> PythonObject:
    """r (n x n, n = d + nrhs) = R of the blocked TSQR of [a | b] (a m x d,
    b m x nrhs, row major; b is read only when nrhs > 0). p = [m, d, nrhs,
    keep]: keep != 0 holds the factorization for `tsqr_q_py`."""
    var m = _n(p, 0)
    var d = _n(p, 1)
    var nrhs = _n(p, 2)
    var keep = Int(py=p[3]) != 0
    var n = d + nrhs
    if d < 1 or n > TS_MAX_N or m < n:
        raise Error("x_decomp: tsqr_r needs 1 <= d, d + nrhs <= " + String(TS_MAX_N) + " and m >= d + nrhs")
    if m * n > 2147483647:
        raise Error("x_decomp: tsqr_r exceeds the Int32 index bound")
    var pa = _f(a)
    var pb = _f(b)
    var pr = _f(r)
    with GILReleased(Python()):
        E.tsqr_factor(pa, pb, pr, m, d, nrhs, keep)
    return PythonObject(n)


# lane idn-dense-linalg (2026-10-04): the one-entry routes Python takes when
# `idn_flags_py` says so (both bindings; the same words as the calls they
# replace). -D MOJOLEARN_IDN_OLS_ONE_ENTRY_OFF / -D MOJOLEARN_IDN_LU_GESV_OFF
# clear the bit, and Python keeps the old call sequence. IDENTICAL builds
# only: a FAST build's bits are 0 and its routes are as they were.
comptime _API_IDN = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
comptime IDN_OLS_ONE_ENTRY = _API_IDN and not (is_defined["MOJOLEARN_IDN_OLS_ONE_ENTRY_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]())
comptime IDN_LU_GESV = _API_IDN and not (is_defined["MOJOLEARN_IDN_LU_GESV_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]())


def idn_flags_py() raises -> PythonObject:
    """Bit 0: LinearRegression takes `ols_tsqr_r_py`; bit 1: solve takes
    `lu_gesv_py`."""
    var bits = 0
    comptime if IDN_OLS_ONE_ENTRY:
        bits |= 1
    comptime if IDN_LU_GESV:
        bits |= 2
    return PythonObject(bits)


def ols_tsqr_r_py[E: Exec](
    a: PythonObject, b: PythonObject, r: PythonObject, mu: PythonObject, ymean: PythonObject, p: PythonObject
) raises -> PythonObject:
    """r ((d + 1) x (d + 1)) = R of the blocked TSQR of [a - mu | b - ymean]
    (a m x d, b m), mu (float32 [d]) and ymean (float64 [1]) written: X and
    y cross to the device once (Exec.ols_tsqr_factor). p = [m, d]."""
    var m = _n(p, 0)
    var d = _n(p, 1)
    var n = d + 1
    if d < 1 or n > TS_MAX_N or m < n:
        raise Error("x_decomp: ols_tsqr_r needs 1 <= d, d + 1 <= " + String(TS_MAX_N) + " and m >= d + 1")
    if m * n > 2147483647:
        raise Error("x_decomp: ols_tsqr_r exceeds the Int32 index bound")
    var pa = _f(a)
    var pb = _f(b)
    var pr = _f(r)
    var pm = _f(mu)
    var ya = Int(py=ymean)
    if ya == 0:
        raise Error("x_decomp: null float64 buffer address")
    var py = MutPointer[UInt64, MutAnyOrigin](unsafe_from_address=ya)
    with GILReleased(Python()):
        E.ols_tsqr_factor(pa, pb, pr, pm, py, m, d)
    return PythonObject(n)


def lu_gesv_py[E: Exec](a: PythonObject, b: PythonObject, info: PythonObject, p: PythonObject) raises -> PythonObject:
    """b (n x nrhs) = the solution of a x = b through `lu` then `lu_solve`
    with the factor resident (Exec.lu_gesv); info (1) is `lu`'s. p = [n, nrhs]."""
    var n = _n(p, 0)
    var nrhs = _n(p, 1)
    if n >= 1 << 24:
        raise Error("x_decomp: lu_gesv row numbers exceed float32's exact integers")
    var pa = _f(a)
    var pb = _f(b)
    var pi = _f(info)
    with GILReleased(Python()):
        E.lu_gesv(pa, pb, pi, n, nrhs)
    return PythonObject(n)


def tsqr_q_py[E: Exec](c: PythonObject, q: PythonObject, p: PythonObject) raises -> PythonObject:
    """q (m x k) = Q c (c n x k) for the factorization `tsqr_r_py` kept,
    which is then released; p = [m, n, k], k == 0 releases it only."""
    var m = _n(p, 0)
    var n = _n(p, 1)
    var k = _n(p, 2)
    if k > 0 and m * k > 2147483647:
        raise Error("x_decomp: tsqr_q exceeds the Int32 index bound")
    var pc = _f(c)
    var pq = _f(q)
    with GILReleased(Python()):
        E.tsqr_apply(pc, pq, m, n, k)
    return PythonObject(k)


def geqrf_py[E: Exec](a: PythonObject, tau: PythonObject, p: PythonObject) raises -> PythonObject:
    """In place: a (m x n, row major) becomes geqrf's factored form, tau
    (min(m, n)) its scalars."""
    var m = _n(p, 0)
    var n = _n(p, 1)
    if m <= 0 or n <= 0:
        raise Error("x_decomp: geqrf needs m, n >= 1")
    var pa = _f(a)
    var pt = _f(tau)
    with GILReleased(Python()):
        E.geqrf(pa, pt, m, n)
    return PythonObject(m if m < n else n)


def orgqr_py[E: Exec](h: PythonObject, tau: PythonObject, q: PythonObject, p: PythonObject) raises -> PythonObject:
    """q (m x qc) = the first qc columns of H_0 ... H_{kk-1} from geqrf's
    factored h (m x n) and tau (kk)."""
    var m = _n(p, 0)
    var n = _n(p, 1)
    var kk = _n(p, 2)
    var qc = _n(p, 3)
    if m <= 0 or n <= 0 or kk > m or kk > n or qc > m or qc <= 0:
        raise Error("x_decomp: orgqr needs 1 <= qc <= m and kk <= min(m, n)")
    var ph = _f(h)
    var pt = _f(tau)
    var pq = _f(q)
    with GILReleased(Python()):
        E.orgqr(ph, pt, pq, m, n, kk, qc)
    return PythonObject(qc)


def als_cg_rows_py[E: Exec](
    c: PythonObject, y: PythonObject, yty: PythonObject, x: PythonObject, steps: PythonObject, p: PythonObject,
    reg: PythonObject,
) raises -> PythonObject:
    """implicit's conjugate-gradient half-sweep: x (n x f) is the start and
    the result."""
    var n = _n(p, 0)
    var m = _n(p, 1)
    var f = _n(p, 2)
    var cg = _n(p, 3)
    var r = Float32(Float64(py=reg))
    var pc = _f(c)
    var py_ = _f(y)
    var pg = _f(yty)
    var px = _f(x)
    var ps = _f(steps)
    with GILReleased(Python()):
        E.als_cg_rows(pc, py_, pg, px, ps, n, m, f, r, cg)
    return PythonObject(n)


def mcd_py[E: Exec](
    x: PythonObject, loc: PythonObject, cov: PythonObject, sup: PythonObject, dist: PythonObject,
    p: PythonObject,
) raises -> PythonObject:
    """MinCovDet's fast_mcd (x_decomp/mcd.mojo): x (n x d) in; location
    (d), covariance (d x d), support (n int32 0/1) and distances (n) out.
    p = [n, d, h, seed, n_sub, n_ss, h_sub, n_trials, n_m, h_m, n_best_m]."""
    var q = List[Int]()
    for i in range(11):
        q.append(Int(py=p[i]))
    var n = q[0]
    var d = q[1]
    if n < 1 or d < 2 or n * d > 2147483647 or q[2] < 1 or q[2] > n:
        raise Error("x_decomp: mcd needs n >= 1, d >= 2 and 1 <= h <= n")
    if n > 500 and (q[4] < 1 or q[4] * q[5] > n or q[8] > n or q[8] < 1 or q[10] < 1):
        raise Error("x_decomp: mcd subset plan out of range")
    var px = _f(x)
    var pl = _f(loc)
    var pc = _f(cov)
    var ps = _i(sup)
    var pd = _f(dist)
    with GILReleased(Python()):
        var X = mat_from(px, n, d)
        fast_mcd[E](X, q, pl, pc, ps, pd)
    return PythonObject(n)


def lda_online_py[E: Exec](
    x: PythonObject, comps: PythonObject, exp_dir: PythonObject, p: PythonObject, f: PythonObject
) raises -> PythonObject:
    """One online pass of LatentDirichletAllocation (x_decomp/lda_online.mojo)
    over x (n x v): comps and exp_dir (nc x v) updated in place.
    p = [n, v, nc, batch_size, max_doc_update_iter, seed, draw, n_batch_iter];
    f = [doc_topic_prior, topic_word_prior, learning_offset, learning_decay,
    mean_change_tol, total_samples]. Returns (draw, n_batch_iter)."""
    var n = _n(p, 0)
    var v = _n(p, 1)
    var nc = _n(p, 2)
    var bs = _n(p, 3)
    var mdi = _n(p, 4)
    var seed = Int(py=p[5])
    var draw = Int(py=p[6])
    var nbi = Int(py=p[7])
    if bs < 1 or v < 1 or nc < 1 or n * v > 2147483647 or nc * v > 2147483647:
        raise Error("x_decomp: lda_online shape out of range")
    var fv = List[Float64]()
    for i in range(6):
        fv.append(Float64(py=f[i]))
    var px = _f(x)
    var pc = _f(comps)
    var pe = _f(exp_dir)
    with GILReleased(Python()):
        var X = mat_from(px, n, v)
        var C = mat_from(pc, nc, v)
        var ED = mat_from(pe, nc, v)
        lda_online_pass[E](X, C, ED, bs, mdi, seed, draw, nbi, fv[0], fv[1], fv[2], fv[3], fv[4], fv[5])
        for i in range(nc * v):
            pc.unsafe_store(i, C.d[i])
            pe.unsafe_store(i, ED.d[i])
    return Python.tuple(draw, nbi)


def gather_py(src: PythonObject, idx: PythonObject, m: PythonObject, dst: PythonObject) raises -> PythonObject:
    """dst[a] = src[idx[a]] for a < m (exact copies; x_decomp/moves.mojo)."""
    var n = Int(py=m)
    if n < 0:
        raise Error("x_decomp: negative gather count")
    var ps = _f(src)
    var pi = _i(idx)
    var pd = _f(dst)
    with GILReleased(Python()):
        gather(ps, pi, n, pd)
    return PythonObject(n)


def scatter_py(dst: PythonObject, idx: PythonObject, m: PythonObject, src: PythonObject) raises -> PythonObject:
    """dst[idx[a]] = src[a] for a < m, in order (exact copies)."""
    var n = Int(py=m)
    if n < 0:
        raise Error("x_decomp: negative scatter count")
    var pd = _f(dst)
    var pi = _i(idx)
    var ps = _f(src)
    with GILReleased(Python()):
        scatter(pd, pi, n, ps)
    return PythonObject(n)


def triu_nonzero_py(dis: PythonObject, n: PythonObject, pos: PythonObject, mir: PythonObject) raises -> PythonObject:
    """The row-major positions of the nonzero strict upper triangle of an
    n x n matrix (and their mirrors); returns their count."""
    var nn = Int(py=n)
    if nn < 0 or nn * nn > 2147483647:
        raise Error("x_decomp: triu_nonzero exceeds the Int32 index bound")
    var pd = _f(dis)
    var pp = _i(pos)
    var pm = _i(mir)
    var m = 0
    with GILReleased(Python()):
        m = triu_nonzero(pd, nn, pp, pm)
    return PythonObject(m)


def argsort_f32_py(x: PythonObject, m: PythonObject, order: PythonObject) raises -> PythonObject:
    """The stable order of x (m float32 values); NaN refused."""
    var n = Int(py=m)
    var px = _f(x)
    var po = _i(order)
    argsort_f32(px, n, po)
    return PythonObject(n)


def iso_order_py(
    x: PythonObject, y: PythonObject, xorder: PythonObject, m: PythonObject, order: PythonObject
) raises -> PythonObject:
    """The stable order by (x, y) given the stable order by x; NaN refused."""
    var n = Int(py=m)
    var px = _f(x)
    var py_ = _f(y)
    var pxo = _i(xorder)
    var po = _i(order)
    iso_order(px, py_, pxo, n, po)
    return PythonObject(n)


# ---- lane apple-fast-py2mojo-decomp (2026-10-03): host buffers, both bindings
def py2mojo_py() raises -> PythonObject:
    """1: the Mojo data path; 0: built with -D MOJOLEARN_PY2MOJO_decomp_OFF (the A/B arm)."""
    return PythonObject(1 if PY2MOJO_DECOMP else 0)


def _d(addr: PythonObject) raises -> F64Ptr:
    var a = Int(py=addr)
    if a == 0:
        raise Error("x_decomp: null float64 buffer address")
    return F64Ptr(unsafe_from_address=a)


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
    if op == 0 and count > 0 and (count - 1) // max(a1, 1) * ist + ioff >= nidx:
        raise Error("x_decomp: take_rows reads past its index list")
    with GILReleased(Python()):
        move_host(op, ps, pi, pd, count, a1, a2, a3, ist, ioff, nsrc, ndst)
    return PythonObject(count)


def dsum_sq_py(x: PythonObject, m: PythonObject) raises -> PythonObject:
    """The in-order float64 sum of the squares of m float32 values."""
    var n = Int(py=m)
    if n <= 0:
        return PythonObject(Float64(0))
    var px = _f(x)
    var t = Float64(0)
    with GILReleased(Python()):
        t = dsum_sq(px, n)
    return PythonObject(t)


def order_f_py(x: PythonObject, m: PythonObject, dst: PythonObject) raises -> PythonObject:
    """The stable order of x (m float32 values) as exact floats; NaN refused."""
    var n = Int(py=m)
    if n > 0:
        order_f(_f(x), n, _f(dst))
    return PythonObject(n)


def select_smallest_py(x: PythonObject, p: PythonObject, sel: PythonObject, mask: PythonObject) raises -> PythonObject:
    """p = [m, h]: the h smallest of x by (value, index), ascending by index,
    as exact floats, and the int32 membership of every row."""
    var m = _n(p, 0)
    var h = _n(p, 1)
    if m > 0:
        select_smallest(_f(x), m, h, _f(sel), _i(mask))
    return PythonObject(h)


def argmin_all_py(x: PythonObject, m: PythonObject, dst: PythonObject) raises -> PythonObject:
    """The positions (exact floats) of every value equal to the minimum; their count."""
    var n = Int(py=m)
    if n <= 0:
        return PythonObject(0)
    return PythonObject(argmin_all(_f(x), n, _f(dst)))


def sign_labels_py(x: PythonObject, m: PythonObject, dst: PythonObject) raises -> PythonObject:
    """int32 1 where x >= 0, else -1."""
    var n = Int(py=m)
    if n > 0:
        sign_labels(_f(x), n, _i(dst))
    return PythonObject(n)


def accuracy_py(y: PythonObject, pred: PythonObject, w: PythonObject, m: PythonObject) raises -> PythonObject:
    """(matches, total): float64 labels against int32 predictions; w = 0
    for unweighted, else a float64 weight address (sums in order)."""
    var n = Int(py=m)
    var weighted = Int(py=w) != 0
    if n <= 0:
        return Python.tuple(Float64(0), Float64(0))
    var pw = _d(w) if weighted else _d(y)
    var r = accuracy(_d(y), _i(pred), pw, weighted, n)
    return Python.tuple(r[0], r[1])


def pca_mle_rank_terms_py(sp: PythonObject, p: PythonObject, v: PythonObject, dst: PythonObject) raises -> PythonObject:
    """p = [d, rank]: the float32 cross terms of Minka's rank `rank`; their count."""
    var d = _n(p, 0)
    var rank = _n(p, 1)
    if rank < 1 or rank >= d:
        raise Error("x_decomp: pca_mle rank out of range")
    return PythonObject(pca_mle_terms(_d(sp), d, rank, Float64(py=v), _f(dst)))


def pca_mle_pa_py(lt: PythonObject, m: PythonObject, logn: PythonObject) raises -> PythonObject:
    """sum over t of (t + logn) in order, float64."""
    var n = Int(py=m)
    if n <= 0:
        return PythonObject(Float64(0))
    return PythonObject(pca_mle_pa(_f(lt), n, Float64(py=logn)))


def topn_desc_py(x: PythonObject, p: PythonObject, skip: PythonObject, dst: PythonObject) raises -> PythonObject:
    """p = [m, n]: the n best positions by (-x, index), skipping the nonzero
    entries of the float32 row at `skip` (0: none); their count."""
    var m = _n(p, 0)
    var n = _n(p, 1)
    if m == 0 or n == 0:
        return PythonObject(0)
    var has = Int(py=skip) != 0
    var px = _f(x)
    var ps = _f(skip) if has else px
    return PythonObject(topn_desc(px, m, ps, has, n, _i(dst)))


def numeric_mode_py() raises -> PythonObject:
    return PythonObject(Int(GLOBAL_NUMERIC_MODE))


def vendor_py[E: Exec]() raises -> PythonObject:
    return PythonObject(E.vendor())
