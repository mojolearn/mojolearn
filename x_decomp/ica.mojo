# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""FastICA's fixed-point loops in Mojo (lane py-runtime-b, 2026-10-05):
`_par` (with `_sym_decorrelation`) and `_def` of `_expansion_decomp.FastICA`,
statement for statement on `Kit[E]` (x_decomp/kit.mojo): the same cells,
broadcast modes and float32 scalars, and Python's float64 scalars (the
eigenvalue floor, 1 / p, the norms' reciprocals, the convergence limits) in
Float64 in the same order. So the IDENTICAL words are the Python driver's on
every column. The GPU binding runs x_decomp/ica_dev.mojo, the same
statements on resident matrices."""
from std.math import sqrt
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from x_decomp.api import _f, _n
from x_decomp.exec_trait import Exec
from x_decomp.kit import (
    Kit, Mat, OP_ABS, OP_ADDS, OP_CUBE, OP_CUBEP, OP_EXPG, OP_EXPGP, OP_MAXS, OP_MUL, OP_ONEMSQ, OP_RECIP,
    OP_SCALE, OP_SQ, OP_SQRT, OP_SUB, OP_TANH, mat_from, mat_rows,
)
from x_decomp.select_ops import SEL_MAXABS

comptime ICA_F32_EPS: Float64 = 1.1920928955078125e-07
comptime ICA_FLT_MIN: Float64 = 1.1754943508222875e-38


def ica_norm[E: Exec](k: Kit[E], v: Mat) raises -> Float64:
    """`_norm`: sqrt(total(v^2)) read as a float."""
    return k.word(k.ew1(OP_SQRT, k.total(k.ew1(OP_SQ, v, 0.0)), 0.0))


def sym_decorrelation[E: Exec](k: Kit[E], W: Mat) raises -> Mat:
    """`_sym_decorrelation`: (W W^T)^(-1/2) W through eigh, the eigenvalues
    clipped at max(FLT_MIN, w_max * eps)."""
    var n = W.r
    var w = Mat(1, n)
    var u = Mat(n, n)
    k.eigh(k.mm(W, W, False, True), w, u)
    var top = Float64(w.d[w.c - 1]) * ICA_F32_EPS
    var floor = ICA_FLT_MIN
    if top > floor:
        floor = top
    w = k.ew1(OP_MAXS, w, floor)
    var ui = k.ew2(OP_MUL, u, k.ew1(OP_RECIP, k.ew1(OP_SQRT, w, 0.0), 0.0))
    return k.mm(k.mm(ui, u, False, True), W, False, False)


def ica_g[E: Exec](k: Kit[E], Y: Mat, fun: Int, alpha: Float64, mut gp: Mat) raises -> Mat:
    """`FastICA._g`: gx returned, gp = the row means of g'(Y).
    fun 0 logcosh (alpha), 1 exp, 2 cube."""
    var gx: Mat
    var g1: Mat
    if fun == 0:
        gx = k.ew1(OP_TANH, k.ew1(OP_SCALE, Y, alpha), 0.0)
        g1 = k.ew1(OP_SCALE, k.ew1(OP_ONEMSQ, gx, 0.0), alpha)
    elif fun == 1:
        gx = k.ew1(OP_EXPG, Y, 0.0)
        g1 = k.ew1(OP_EXPGP, Y, 0.0)
    else:
        gx = k.ew1(OP_CUBE, Y, 0.0)
        g1 = k.ew1(OP_CUBEP, Y, 0.0)
    gp = k.ew1(OP_SCALE, k.rowsum(g1), 1.0 / Float64(Y.c))
    return gx^


def ica_par[E: Exec](X1: Mat, mut W: Mat, fun: Int, alpha: Float64, max_iter: Int, tol: Float64) raises -> Int:
    """`FastICA._par`: W replaced; returns the iteration count."""
    var k = Kit[E]()
    W = sym_decorrelation(k, W)
    var p = X1.c
    var it = 0
    for i in range(1, max_iter + 1):
        it = i
        var gp = Mat(0, 0)
        var gx = ica_g(k, k.mm(W, X1, False, False), fun, alpha, gp)
        var W1 = sym_decorrelation(
            k, k.ew2(OP_SUB, k.ew1(OP_SCALE, k.mm(gx, X1, False, True), 1.0 / Float64(p)), k.ew2(OP_MUL, W, gp))
        )
        var dots = k.rowsum(k.ew2(OP_MUL, W1, W))
        var lim = k.word(k.reduce(k.ew1(OP_ADDS, k.ew1(OP_ABS, dots, 0.0), -1.0), SEL_MAXABS))
        W = W1^
        if lim < tol:
            break
    return it


def _recip_or0(v: Float64) -> Float64:
    """Python's `1.0 / v if v else 0.0`."""
    return 1.0 / v if v != 0.0 else 0.0


def ica_def[E: Exec](X1: Mat, mut W: Mat, fun: Int, alpha: Float64, max_iter: Int, tol: Float64) raises -> Int:
    """`FastICA._def`: the rows found one at a time into W (initially the
    w_init rows); returns the largest iteration count."""
    var k = Kit[E]()
    var nc = W.r
    var p = X1.c
    var Winit = W.copy()
    var best = 0
    var first = True
    for j in range(nc):  # small-loop(nc: components): one deflation row per component, each a device-sized loop
        var w = mat_rows(Winit, j, j + 1)
        var Wp = mat_rows(W, 0, j)
        if j > 0:
            w = k.ew2(OP_SUB, w, k.mm(k.mm(w, Wp, False, True), Wp, False, False))
        w = k.ew1(OP_SCALE, w, _recip_or0(ica_norm(k, w)))
        var it = 0
        for i in range(1, max_iter + 1):
            it = i
            var gp = Mat(0, 0)
            var gx = ica_g(k, k.mm(w, X1, False, False), fun, alpha, gp)
            var w1 = k.ew2(OP_SUB, k.ew1(OP_SCALE, k.mm(gx, X1, False, True), 1.0 / Float64(p)), k.ew2(OP_MUL, w, gp))
            if j > 0:
                w1 = k.ew2(OP_SUB, w1, k.mm(k.mm(w1, Wp, False, True), Wp, False, False))
            var nw = ica_norm(k, w1)
            w1 = k.ew1(OP_SCALE, w1, _recip_or0(nw))
            var lim = abs(abs(k.word(k.total(k.ew2(OP_MUL, w1, w)))) - 1)
            w = w1^
            if lim < tol:
                break
        if first or it > best:
            best = it
            first = False
        Kit[E].place_row(W, w, j)
    return best


def ica_solve_py[E: Exec](x1: PythonObject, w: PythonObject, p: PythonObject, f: PythonObject) raises -> PythonObject:
    """`FastICA._par` / `_def` over X1 (xr x xc): w (nc x nc, the initial
    unmixing matrix) replaced in place. p = [nc, xr, xc, algorithm (0
    parallel, 1 deflation), fun (0 logcosh, 1 exp, 2 cube), max_iter];
    f = [alpha, tol]. Returns the iteration count."""
    var nc = _n(p, 0)
    var xr = _n(p, 1)
    var xc = _n(p, 2)
    var algo = Int(py=p[3])
    var fun = Int(py=p[4])
    var max_iter = Int(py=p[5])
    if xr * xc > 2147483647 or nc * nc > 2147483647 or nc < 1:
        raise Error("x_decomp: fastica shape out of range")
    var alpha = Float64(py=f[0])
    var tol = Float64(py=f[1])
    var px = _f(x1)
    var pw = _f(w)
    var it = 0
    with GILReleased(Python()):
        var X1 = mat_from(px, xr, xc)
        var W = mat_from(pw, nc, nc)
        if algo == 0:
            it = ica_par[E](X1, W, fun, alpha, max_iter, tol)
        else:
            it = ica_def[E](X1, W, fun, alpha, max_iter, tol)
        for i in range(nc * nc):
            pw.unsafe_store(i, W.d[i])
    return PythonObject(it)
