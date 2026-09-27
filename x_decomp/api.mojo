# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The decomp lane's Python entry points, written ONCE and instantiated for
both executors: `bindings/_mojolearn_x_decomp.mojo` registers `*_py[DevExec]`,
`bindings/_mojolearn_x_decomp_host.mojo` registers `*_py[HostExec]` under the
same names, so the GPU binding and the CPU host binding share the address
contract by construction. Every address is a host buffer the caller owns."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from checks.numerics import GLOBAL_NUMERIC_MODE
from x_decomp.cells import F32Ptr, I32Ptr
from x_decomp.exec_trait import Exec


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
    if na * nb > 2147483647:
        raise Error("x_decomp: sqdist exceeds the Int32 index bound")
    var pa = _f(a)
    var pb = _f(b)
    var po = _f(dst)
    with GILReleased(Python()):
        E.sqdist(pa, pb, po, na, nb, d)
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


def lu_solve_py[E: Exec](lu: PythonObject, piv: PythonObject, b: PythonObject, p: PythonObject) raises -> PythonObject:
    var n = _n(p, 0)
    var nrhs = _n(p, 1)
    var trans = _n(p, 2) if len(p) > 2 else 0
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
    var pa = _f(a)
    var pw = _f(w)
    var pv = _f(v)
    E.eigh(pa, pw, pv, n)
    return PythonObject(n)


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


def numeric_mode_py() raises -> PythonObject:
    return PythonObject(Int(GLOBAL_NUMERIC_MODE))


def vendor_py[E: Exec]() raises -> PythonObject:
    return PythonObject(E.vendor())
