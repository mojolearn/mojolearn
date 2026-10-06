# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""x_decomp/ica.mojo's FastICA loops on resident device matrices (lane
py-runtime-b): the same statements on `DKit` (x_decomp/kit_device.mojo),
the same cells on the same values (eigh: `DKit.eigh`, DevExec's solve), so
the same words as the host column. Each Python scalar read (the largest
eigenvalue, a norm, a convergence limit) is one word home where Python read
it. GPU binding only."""
from experiments.classical_identical_ideas.linear_controls import C27_COMPONENTS
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from x_decomp.api import _f, _n
from x_decomp.ica import ICA_F32_EPS, ICA_FLT_MIN, _recip_or0
from x_decomp.kit import (
    Mat, OP_ABS, OP_ADDS, OP_CUBE, OP_CUBEP, OP_EXPG, OP_EXPGP, OP_MAXS, OP_MUL, OP_ONEMSQ, OP_RECIP,
    OP_SCALE, OP_SQ, OP_SQRT, OP_SUB, OP_TANH, mat_from,
)
from x_decomp.kit_device import DKit, DMat
from x_decomp.select_ops import SEL_MAXABS


def ica_norm_dev(mut k: DKit, v: DMat) raises -> Float64:
    return k.word(k.ew1(OP_SQRT, k.total(k.ew1(OP_SQ, v, 0.0)), 0.0))


def sym_decorrelation_dev(mut k: DKit, W: DMat) raises -> DMat:
    var e = k.eigh(k.mm(W, W, False, True))
    var top = Float64(e.wh.d[e.wh.c - 1]) * ICA_F32_EPS
    var floor = ICA_FLT_MIN
    if top > floor:
        floor = top
    var w = k.ew1(OP_MAXS, e.wd, floor)
    var ui = k.ew2(OP_MUL, e.vd, k.ew1(OP_RECIP, k.ew1(OP_SQRT, w, 0.0), 0.0))
    return k.mm(k.mm(ui, e.vd, False, True), W, False, False)


def ica_g_dev(k: DKit, Y: DMat, fun: Int, alpha: Float64, mut gp: DMat) raises -> DMat:
    comptime if C27_COMPONENTS:
        var g1 = DMat(0, 0)
        var gx = k.classical_contrast(Y, fun, alpha, g1)
        gp = k.ew1(OP_SCALE, k.rowsum(g1), 1.0 / Float64(Y.c))
        return gx^
    var gx: DMat
    var g1: DMat
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


def ica_par_dev(mut k: DKit, X1: DMat, mut W: DMat, fun: Int, alpha: Float64, max_iter: Int, tol: Float64) raises -> Int:
    W = sym_decorrelation_dev(k, W)
    var p = X1.c
    var it = 0
    for i in range(1, max_iter + 1):
        it = i
        var gp = DMat(0, 0)
        var gx = ica_g_dev(k, k.mm(W, X1, False, False), fun, alpha, gp)
        var W1 = sym_decorrelation_dev(
            k, k.ew2(OP_SUB, k.ew1(OP_SCALE, k.mm(gx, X1, False, True), 1.0 / Float64(p)), k.ew2(OP_MUL, W, gp))
        )
        var dots = k.rowsum(k.ew2(OP_MUL, W1, W))
        var lim = k.word(k.reduce(k.ew1(OP_ADDS, k.ew1(OP_ABS, dots, 0.0), -1.0), SEL_MAXABS))
        W = W1^
        if lim < tol:
            break
    return it


def ica_def_dev(mut k: DKit, X1: DMat, mut W: DMat, fun: Int, alpha: Float64, max_iter: Int, tol: Float64) raises -> Int:
    var nc = W.r
    var p = X1.c
    var Winit = k.copy(W)
    var best = 0
    var first = True
    for j in range(nc):  # small-loop(nc: components): one deflation row per component, each a device-sized loop
        var w = k.rows(Winit, j, j + 1)
        var Wp = k.rows(W, 0, j)
        if j > 0:
            w = k.ew2(OP_SUB, w, k.mm(k.mm(w, Wp, False, True), Wp, False, False))
        w = k.ew1(OP_SCALE, w, _recip_or0(ica_norm_dev(k, w)))
        var it = 0
        for i in range(1, max_iter + 1):
            it = i
            var gp = DMat(0, 0)
            var gx = ica_g_dev(k, k.mm(w, X1, False, False), fun, alpha, gp)
            var w1 = k.ew2(OP_SUB, k.ew1(OP_SCALE, k.mm(gx, X1, False, True), 1.0 / Float64(p)), k.ew2(OP_MUL, w, gp))
            if j > 0:
                w1 = k.ew2(OP_SUB, w1, k.mm(k.mm(w1, Wp, False, True), Wp, False, False))
            var nw = ica_norm_dev(k, w1)
            w1 = k.ew1(OP_SCALE, w1, _recip_or0(nw))
            var lim = abs(abs(k.word(k.total(k.ew2(OP_MUL, w1, w)))) - 1)
            w = w1^
            if lim < tol:
                break
        if first or it > best:
            best = it
            first = False
        k.place_row(W, w, j)
    return best


def ica_solve_dev_py(x1: PythonObject, w: PythonObject, p: PythonObject, f: PythonObject) raises -> PythonObject:
    """`ica_solve_py` on the resident kit."""
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
        var k = DKit()
        var X1 = k.upload(mat_from(px, xr, xc))
        var W = k.upload(mat_from(pw, nc, nc))
        if algo == 0:
            it = ica_par_dev(k, X1, W, fun, alpha, max_iter, tol)
        else:
            it = ica_def_dev(k, X1, W, fun, alpha, max_iter, tol)
        var hw = k.get(W)
        k.sync()
        for i in range(nc * nc):
            pw.unsafe_store(i, hw.d[i])
    return PythonObject(it)
