# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""LocallyLinearEmbedding's shift-invert subspace iteration in Mojo (lane
py-runtime-b, 2026-10-05): `_expansion_decomp._lle_iterate` (the F0 route
`_lle_smallest` takes) with `_lle_orth`, statement for statement on
`Kit[E]`: the same cells, triangular solves on the LU factor, Householder
QR (geqrf, orgqr), the SVD with `_Kit.svd`'s descending order, and Python's
float64 stopping tests in the same order. So the IDENTICAL words are the
Python driver's; x_decomp/lle_iter_dev.mojo runs the same on DKit.

Returns status 0 (settled), 1 (the null floor) or 2 (not settled; the
caller raises Python's RuntimeError with the last subspace change)."""
from std.math import inf, sqrt
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from x_decomp.api import _f, _n
from x_decomp.exec_trait import Exec
from x_decomp.fa_em import svd_desc
from x_decomp.kit import (
    Kit, Mat, OP_ABS, OP_ADDS, OP_MUL, OP_RECIP, OP_SELECT, OP_SQ, OP_SUB, mat_const, mat_from, mat_rows,
)
from x_decomp.select_ops import SEL_MAX, sel_fold


struct LleArgs(Copyable, Movable):
    var n: Int
    var n1: Int
    var nc: Int
    var p: Int
    var max_iter: Int
    var seed: Int
    var floor: Float64
    var sub_tol: Float64
    var stall_tol: Float64

    def __init__(out self, n: Int, n1: Int, nc: Int, p: Int, max_iter: Int, seed: Int, floor: Float64,
                 sub_tol: Float64, stall_tol: Float64):
        self.n = n
        self.n1 = n1
        self.nc = nc
        self.p = p
        self.max_iter = max_iter
        self.seed = seed
        self.floor = floor
        self.sub_tol = sub_tol
        self.stall_tol = stall_tol


def pad_zero_row(X: Mat) -> Mat:
    """`_Kit.pad_zero_row`: [X; 0]."""
    var out = Mat(X.r + 1, X.c)
    for i in range(X.n()):
        out.d[i] = X.d[i]
    return out^


def take_want(X: Mat, p: Int, nc: Int) -> Mat:
    """`X.take_cols(range(p - 1, p - 1 - nc, -1))`: exact copies."""
    var out = Mat(X.r, nc)
    for i in range(X.r):
        for j in range(nc):  # small-loop(nc: components): the wanted Ritz columns, descending
            out.d[i * nc + j] = X.d[i * X.c + p - 1 - j]
    return out^


def lle_orth[E: Exec](k: Kit[E], Z: Mat) raises -> Mat:
    """`_lle_orth`: columns scaled by 1 / their 1-norm (1 where that is not
    > 0), then Q of the Householder QR (geqrf, orgqr)."""
    var rc = k.ew1(OP_RECIP, k.colsum(k.ew1(OP_ABS, Z, 0.0)), 0.0)
    var sc = k.ew3(OP_SELECT, rc, rc, mat_const(1.0, 1, 1), 0.0)
    var h = k.ew2(OP_MUL, Z, sc)
    var kk = min(h.r, h.c)
    var tau = Mat(1, kk)
    E.geqrf(h.p(), tau.p(), h.r, h.c)
    var Q = Mat(h.r, Z.c)
    E.orgqr(h.p(), tau.p(), Q.p(), h.r, h.c, kk, Z.c)
    return Q^


def lle_iterate[E: Exec](
    F0: Mat, lu: Mat, pm: Mat, im: Mat, z: Mat, a: LleArgs, mut X: Mat, mut Y: Mat, mut S: Mat, mut e_last: Float64
) raises -> Int:
    var k = Kit[E]()
    var n = a.n
    var p = a.p
    X = lle_orth(k, k.ew1(OP_ADDS, k.rand(a.n1, p, a.seed, 0x11E, 0), -0.5))
    var prev = Mat(0, 0)
    var have_prev = False
    var e_prev = inf[DType.float64]()
    for it in range(max(1, a.max_iter)):
        var T = Mat(n, p)
        E.trisolve(lu.p(), im.p(), pad_zero_row(X).p(), T.p(), n, p, 1)
        T = lle_orth(k, k.ew2(OP_SUB, T, k.mm(z, k.mm(z, T, True, False), False, False)))
        var U = Mat(n, p)
        E.trisolve(lu.p(), pm.p(), T.p(), U.p(), n, p, 0)
        X = lle_orth(k, mat_rows(U, 0, a.n1))
        var B = k.mm(F0, pad_zero_row(X), False, False)
        var s = Mat(1, B.c)
        var v = Mat(B.c, B.c)
        E.svd(B.p(), B.r, B.c, s.p(), v.p())
        var Vt = Mat(0, 0)
        S = Mat(0, 0)
        svd_desc(s, v, S, Vt)
        X = k.mm(X, Vt, False, True)
        Y = take_want(X, p, a.nc)
        var sw = take_want(S, p, a.nc)
        if it >= 2 and Float64(sel_fold(SEL_MAX, sw.p(), 0, sw.n())) <= a.floor:
            return 1
        if have_prev:
            var Em = k.ew2(OP_SUB, Y, k.mm(prev, k.mm(prev, Y, True, False), False, False))
            var t = k.word(k.total(k.ew1(OP_SQ, Em, 0.0)))
            var e = sqrt(t if not (0.0 > t) else 0.0)
            if e <= a.sub_tol or (e <= a.stall_tol and e >= e_prev):
                return 0
            e_prev = e
        prev = Y.copy()
        have_prev = True
    e_last = e_prev
    return 2


def _lle_args(p: PythonObject, f: PythonObject) raises -> LleArgs:
    """p = [n, n1, nc, p, max_iter, seed]; f = [floor, sub_tol, stall_tol]."""
    return LleArgs(_n(p, 0), _n(p, 1), _n(p, 2), _n(p, 3), Int(py=p[4]), Int(py=p[5]), Float64(py=f[0]),
                   Float64(py=f[1]), Float64(py=f[2]))


def lle_iterate_py[E: Exec](
    f0: PythonObject, lu: PythonObject, pm: PythonObject, im: PythonObject, z: PythonObject, out: PythonObject,
    p: PythonObject, f: PythonObject,
) raises -> PythonObject:
    """out = [X (n1 x p), Y (n1 x nc), S (p)] addresses. Returns (status, e)."""
    var a = _lle_args(p, f)
    var n = a.n
    if a.n1 != n - 1 or a.p < a.nc or a.p > a.n1 or n * n > 2147483647:
        raise Error("x_decomp: lle iterate shape out of range")
    var pf = _f(f0)
    var pl = _f(lu)
    var ppm = _f(pm)
    var pim = _f(im)
    var pz = _f(z)
    var px = _f(out[0])
    var py_ = _f(out[1])
    var ps = _f(out[2])
    var st = 0
    var e = 0.0
    with GILReleased(Python()):
        var X = Mat(0, 0)
        var Y = Mat(0, 0)
        var S = Mat(0, 0)
        st = lle_iterate[E](mat_from(pf, n, n), mat_from(pl, n, n), mat_from(ppm, n, 1), mat_from(pim, n, 1),
                            mat_from(pz, n, 1), a, X, Y, S, e)
        for i in range(X.n()):
            px.unsafe_store(i, X.d[i])
        for i in range(Y.n()):
            py_.unsafe_store(i, Y.d[i])
        for i in range(S.n()):
            ps.unsafe_store(i, S.d[i])
    return Python.tuple(st, e)
