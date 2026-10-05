# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The PLS component loop in Mojo (lane py-runtime-b, 2026-10-05):
`_expansion_decomp._PLS.fit`'s `for _c in range(nc)` with `_power`
(NIPALS, modes A and B), `_thin_svd`, `_pinv` and `_svd_flip_1d`, statement
for statement on `Kit[E]`: the same cells, broadcast modes and float32
scalars, the SVD's descending order (`_Kit.svd`), the counts and tolerance
tests read as Python read them. So the IDENTICAL words are the Python
driver's; x_decomp/pls_dev.mojo is the same text on DKit (generated from
this file's driver section).

Status: the number of components fitted; a constant y residual stops the
loop there (the caller warns, as Python's StopIteration did)."""
from std.math import sqrt
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from x_decomp.api import _f, _n
from x_decomp.exec_trait import Exec
from x_decomp.kit import (
    Kit, Mat, OP_ABS, OP_ADDS, OP_DIV, OP_GTS, OP_LE, OP_MUL, OP_RECIP, OP_SCALE, OP_SELECT, OP_SQRT, OP_SUB,
    mat_from,
)

comptime PLS_F32_EPS: Float64 = 1.1920928955078125e-07


struct PlsArgs(Copyable, Movable):
    var nc: Int
    var max_iter: Int
    var tol: Float64
    var mode_b: Bool
    var norm_y: Bool
    var canonical: Bool
    var svd_algo: Bool

    def __init__(out self, nc: Int, max_iter: Int, tol: Float64, mode_b: Bool, norm_y: Bool, canonical: Bool,
                 svd_algo: Bool):
        self.nc = nc
        self.max_iter = max_iter
        self.tol = tol
        self.mode_b = mode_b
        self.norm_y = norm_y
        self.canonical = canonical
        self.svd_algo = svd_algo


def _pls_args(p: PythonObject, f: PythonObject) raises -> PlsArgs:
    """p = [n, px, q, nc, max_iter, mode_b, norm_y, canonical, svd_algo]; f = [tol]."""
    return PlsArgs(_n(p, 3), Int(py=p[4]), Float64(py=f[0]), Int(py=p[5]) != 0, Int(py=p[6]) != 0,
                   Int(py=p[7]) != 0, Int(py=p[8]) != 0)


# ---- driver (the same text on DKit in x_decomp/pls_dev.mojo)
def _f32(x: Float64) -> Float64:
    return Float64(Float32(x))


def thin_svd[E: Exec](mut k: Kit[E], X: Mat, nc: Int, u_based: Bool, mut U: Mat, mut S: Mat, mut Vt: Mat) raises:
    """`_thin_svd(k, X, nc, u_based)`."""
    var n = X.r
    var d = X.c
    var S0 = k.zeros(0, 0)
    var V0 = k.zeros(0, 0)
    if d <= n:
        k.svd(X, S0, V0)
        S = k.cols(S0, 0, nc)
        Vt = k.rows(V0, 0, nc)
        U = k.ew2(OP_DIV, k.mm(X, Vt, False, True), S)
    else:
        k.svd(k.t(X), S0, V0)
        S = k.cols(S0, 0, nc)
        var Ut = k.rows(V0, 0, nc)
        U = k.t(Ut)
        Vt = k.ew2(OP_DIV, k.mm(U, X, True, False), k.vec_t(k.copy(S)))
    if u_based:
        var sg = k.absmax_signs(U, True)
        U = k.ew2(OP_MUL, U, sg)
        Vt = k.ew2(OP_MUL, Vt, k.vec_t(sg^))
    else:
        var sg = k.absmax_signs(Vt, False)
        U = k.ew2(OP_MUL, U, k.vec_t(k.copy(sg)))
        Vt = k.ew2(OP_MUL, Vt, sg)


def pinv[E: Exec](mut k: Kit[E], A: Mat) raises -> Mat:
    """`_pinv(k, A)`: V diag(1/s) U^T, values at or below max(shape) eps s_max dropped."""
    var r = min(A.r, A.c)
    var U = k.zeros(0, 0)
    var S = k.zeros(0, 0)
    var Vt = k.zeros(0, 0)
    thin_svd(k, A, r, True, U, S, Vt)
    var cond = Float64(max(A.r, A.c)) * PLS_F32_EPS
    var cut = _f32(k.word(k.cols(S, 0, 1)) * cond) if r > 0 else 0.0
    var inv = k.ew1(OP_RECIP, k.ew3(OP_SELECT, S, S, k.zeros(1, 1), cut), 0.0)
    return k.mm(Vt, k.ew2(OP_MUL, U, inv), True, True)


def _dot[E: Exec](mut k: Kit[E], a: Mat, b: Mat) raises -> Mat:
    return k.mm(a, b, True, False)


def power[E: Exec](mut k: Kit[E], X: Mat, Y: Mat, a: PlsArgs, mut xw: Mat, mut yw: Mat, mut stop: Bool) raises -> Int:
    """`_PLS._power`: xw, yw set; stop when the y residual is constant."""
    var eps = PLS_F32_EPS
    var cnt = k.colsum(k.ew1(OP_GTS, k.ew1(OP_ABS, Y, 0.0), eps))
    if k.count_gt(cnt, 0.0) == 0:
        stop = True
        return 0
    var first = k.order_small(k.ew1(OP_ADDS, k.ew1(OP_SCALE, k.ew1(OP_GTS, cnt, 0.0), -1.0), 1.0))
    var y_score = k.take_col(Y, Int(first[0]))
    var Xp = k.zeros(0, 0)
    var Yp = k.zeros(0, 0)
    if a.mode_b:
        Xp = pinv(k, X)
        Yp = pinv(k, Y)
    var xw_old = k.zeros(0, 0)
    var have_old = False
    var it = 0
    for i in range(1, a.max_iter + 1):
        it = i
        if a.mode_b:
            xw = k.mm(Xp, y_score, False, False)
        else:
            xw = k.ew2(OP_DIV, k.mm(X, y_score, True, False), _dot(k, y_score, y_score))
        xw = k.ew2(OP_DIV, xw, k.ew1(OP_ADDS, k.ew1(OP_SQRT, _dot(k, xw, xw), 0.0), eps))
        var x_score = k.mm(X, xw, False, False)
        if a.mode_b:
            yw = k.mm(Yp, x_score, False, False)
        else:
            yw = k.ew2(OP_DIV, k.mm(Y, x_score, True, False), _dot(k, x_score, x_score))
        if a.norm_y:
            yw = k.ew2(OP_DIV, yw, k.ew1(OP_ADDS, k.ew1(OP_SQRT, _dot(k, yw, yw), 0.0), eps))
        y_score = k.ew2(OP_DIV, k.mm(Y, yw, False, False), k.ew1(OP_ADDS, _dot(k, yw, yw), eps))
        if Y.c == 1:
            break
        if have_old:
            var diff = k.ew2(OP_SUB, xw, xw_old)
            if k.word(_dot(k, diff, diff)) < a.tol:
                break
        else:
            var diff = k.ew1(OP_ADDS, xw, -100.0)
            if k.word(_dot(k, diff, diff)) < a.tol:
                break
        xw_old = k.copy(xw)
        have_old = True
    return it


def pls_components[E: Exec](
    mut k: Kit[E], var Xk: Mat, var Yk: Mat, a: PlsArgs, mut outs: List[Mat], mut its: List[Int]
) raises -> Int:
    """The component loop: outs = [xw, yw, xs, ys, xl, yl] per component,
    appended in order (6 per component); its the power iteration counts.
    Returns the number of components fitted."""
    var q = Yk.c
    var thr = 10 * PLS_F32_EPS
    var done = 0
    for _c in range(a.nc):  # small-loop(nc: components): one NIPALS component each, a chain of kit cells
        var live = k.colsum(k.ew2(OP_LE, k.ew1(OP_SCALE, k.ew1(OP_ABS, Yk, 0.0), -1.0), k.const(-thr, 1, 1)))
        if k.count_gt(live, 0.0) < q:
            Yk = k.ew2(OP_MUL, Yk, k.ew1(OP_GTS, live, 0.0))
        var xw = k.zeros(0, 0)
        var yw = k.zeros(0, 0)
        if a.svd_algo:
            var Cxy = k.mm(Xk, Yk, True, False)
            var U = k.zeros(0, 0)
            var S = k.zeros(0, 0)
            var Vt = k.zeros(0, 0)
            thin_svd(k, Cxy, 1, True, U, S, Vt)
            xw = k.cols(U, 0, 1)
            yw = k.vec_t(k.rows(Vt, 0, 1))
        else:
            var stop = False
            var it = power(k, Xk, Yk, a, xw, yw, stop)
            if stop:
                break
            its.append(it)
        var sg = k.absmax_signs(xw, True)
        xw = k.ew2(OP_MUL, xw, sg)
        yw = k.ew2(OP_MUL, yw, sg)
        var x_scores = k.mm(Xk, xw, False, False)
        var y_ss = k.const(1.0, 1, 1) if a.norm_y else _dot(k, yw, yw)
        var y_scores = k.ew2(OP_DIV, k.mm(Yk, yw, False, False), y_ss)
        var x_load = k.ew2(OP_DIV, k.mm(Xk, x_scores, True, False), _dot(k, x_scores, x_scores))
        Xk = k.ew2(OP_SUB, Xk, k.mm(x_scores, x_load, False, True))
        var y_load: Mat
        if a.canonical:
            y_load = k.ew2(OP_DIV, k.mm(Yk, y_scores, True, False), _dot(k, y_scores, y_scores))
            Yk = k.ew2(OP_SUB, Yk, k.mm(y_scores, y_load, False, True))
        else:
            y_load = k.ew2(OP_DIV, k.mm(Yk, x_scores, True, False), _dot(k, x_scores, x_scores))
            Yk = k.ew2(OP_SUB, Yk, k.mm(x_scores, y_load, False, True))
        outs.append(xw^)
        outs.append(yw^)
        outs.append(x_scores^)
        outs.append(y_scores^)
        outs.append(x_load^)
        outs.append(y_load^)
        done += 1
    return done
# ---- end driver


def pls_fit_py[E: Exec](
    x: PythonObject, y: PythonObject, outs: PythonObject, its: PythonObject, p: PythonObject, f: PythonObject
) raises -> PythonObject:
    """x (n x px) and y (n x q) the centred, scaled data; outs = six
    addresses of [xw px x nc, yw q x nc, xs n x nc, ys n x nc, xl px x nc,
    yl q x nc] (column c written for each fitted component); its (nc int32)
    the power iteration counts. Returns (fitted, iteration count entries)."""
    var n = _n(p, 0)
    var px = _n(p, 1)
    var q = _n(p, 2)
    var a = _pls_args(p, f)
    if n * px > 2147483647 or n * q > 2147483647 or a.nc < 1:
        raise Error("x_decomp: pls shape out of range")
    var pxa = _f(x)
    var pya = _f(y)
    var dst = List[Int]()
    for i in range(6):  # small-loop(i: six output addresses): Python parameter glue
        dst.append(Int(py=outs[i]))
    var pit = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=its))
    var done = 0
    var nits = 0
    with GILReleased(Python()):
        var k = Kit[E]()
        var res = List[Mat]()
        var il = List[Int]()
        done = pls_components(k, mat_from(pxa, n, px), mat_from(pya, n, q), a, res, il)
        for c in range(done):
            for t in range(6):
                var po = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=dst[t])
                ref v = res[c * 6 + t]
                for i in range(v.n()):
                    po.unsafe_store(i * a.nc + c, v.d[i])
        nits = len(il)
        for i in range(nits):
            pit.unsafe_store(i, Int32(il[i]))
    return Python.tuple(done, nits)
