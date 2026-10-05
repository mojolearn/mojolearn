# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""AlternatingLeastSquares's sweep loop in Mojo (lane py-runtime-b,
2026-10-05): `_expansion_decomp.AlternatingLeastSquares.fit`'s
`for _ in range(iterations)` and `_loss`, statement for statement on
`Kit[E]` (the host column: the item half-sweep on C^T, as the host kit ran
it): the same row solvers (`als_rows`, `als_cg_rows`), the same cells and
float32 scalars, the loss's float64 tail in Python's order. So the IDENTICAL
words are the Python driver's; x_decomp/als_dev.mojo runs the GPU binding's
form (C resident, the item half-sweep through strides)."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from x_decomp.api import _f, _n
from x_decomp.exec_trait import Exec
from x_decomp.kit import (
    Kit, Mat, OP_ABS, OP_ADD, OP_ADDS, OP_GTS, OP_MUL, OP_SCALE, OP_SELECT, OP_SQ, mat_from, mat_t,
)


struct AlsArgs(Copyable, Movable):
    var iterations: Int
    var use_cg: Bool
    var cg_steps: Int
    var calc_loss: Bool
    var reg: Float64

    def __init__(out self, iterations: Int, use_cg: Bool, cg_steps: Int, calc_loss: Bool, reg: Float64):
        self.iterations = iterations
        self.use_cg = use_cg
        self.cg_steps = cg_steps
        self.calc_loss = calc_loss
        self.reg = reg


def als_half[E: Exec](k: Kit[E], C: Mat, Y: Mat, reg: Float64) raises -> Mat:
    """`_Kit.als(C, Y, reg)` on the host column."""
    var YtY = k.mm(Y, Y, True, False)
    var X = Mat(C.r, Y.c)
    var flags = Mat(C.r, 1)
    if C.r > 0:
        E.als_rows(C.p(), Y.p(), YtY.p(), X.p(), flags.p(), C.r, C.c, Y.c, Float32(reg))
    return X^


def als_cg_half[E: Exec](k: Kit[E], C: Mat, Y: Mat, X0: Mat, reg: Float64, cg: Int) raises -> Mat:
    """`_Kit.als_cg(C, Y, X0, reg, cg)`."""
    var YtY = k.mm(Y, Y, True, False)
    var X = X0.copy()
    var steps = Mat(C.r, 1)
    if C.r > 0:
        E.als_cg_rows(C.p(), Y.p(), YtY.p(), X.p(), steps.p(), C.r, C.c, Y.c, Float32(reg), cg)
    return X^


def als_loss[E: Exec](k: Kit[E], C: Mat, X: Mat, Y: Mat, reg: Float64) raises -> Float64:
    """`AlternatingLeastSquares._loss`."""
    var P = k.mm(X, Y, False, True)
    var seen = k.ew1(OP_GTS, k.ew1(OP_ABS, C, 0.0), 0.0)
    var obs = k.ew2(OP_MUL, C, k.ew1(OP_SQ, k.ew1(OP_ADDS, k.ew1(OP_SCALE, P, -1.0), 1.0), 0.0))
    var term = k.ew3(OP_SELECT, seen, obs, k.ew1(OP_SQ, P, 0.0), 0.5)
    var regm = k.ew2(OP_ADD, k.total(k.ew1(OP_SQ, X, 0.0)), k.total(k.ew1(OP_SQ, Y, 0.0)))
    var tot = k.word(k.ew2(OP_ADD, k.total(term), k.ew1(OP_SCALE, regm, reg)))
    var nnz = k.word(k.total(seen))
    var conf = k.word(k.total(k.ew2(OP_MUL, C, seen)))
    return tot / (conf + (Float64(C.r * C.c) - nnz))


def als_fit[E: Exec](C: Mat, mut X: Mat, mut Y: Mat, a: AlsArgs, mut losses: List[Float64]) raises:
    var k = Kit[E]()
    var Ct = mat_t(C)
    for _ in range(a.iterations):
        if a.use_cg:
            X = als_cg_half(k, C, Y, X, a.reg, a.cg_steps)
            Y = als_cg_half(k, Ct, X, Y, a.reg, a.cg_steps)
        else:
            X = als_half(k, C, Y, a.reg)
            Y = als_half(k, Ct, X, a.reg)
        if a.calc_loss:
            losses.append(als_loss(k, C, X, Y, a.reg))


def _als_args(p: PythonObject, f: PythonObject) raises -> AlsArgs:
    """p = [n, m, f, iterations, use_cg, cg_steps, calc_loss]; f = [reg]."""
    return AlsArgs(Int(py=p[3]), Int(py=p[4]) != 0, Int(py=p[5]), Int(py=p[6]) != 0, Float64(py=f[0]))


def als_fit_py[E: Exec](
    c: PythonObject, x: PythonObject, y: PythonObject, ls: PythonObject, p: PythonObject, f: PythonObject
) raises -> PythonObject:
    """c (n x m) the confidences; x (n x f) and y (m x f) the starts,
    replaced; ls (iterations float64) the losses when asked. Returns the
    number of losses written."""
    var n = _n(p, 0)
    var m = _n(p, 1)
    var nf = _n(p, 2)
    if n * m > 2147483647 or n * nf > 2147483647 or m * nf > 2147483647:
        raise Error("x_decomp: als shape out of range")
    var a = _als_args(p, f)
    var pc = _f(c)
    var px = _f(x)
    var py_ = _f(y)
    var pl = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(py=ls))
    var cnt = 0
    with GILReleased(Python()):
        var X = mat_from(px, n, nf)
        var Y = mat_from(py_, m, nf)
        var losses = List[Float64]()
        als_fit[E](mat_from(pc, n, m), X, Y, a, losses)
        for i in range(n * nf):
            px.unsafe_store(i, X.d[i])
        for i in range(m * nf):
            py_.unsafe_store(i, Y.d[i])
        cnt = len(losses)
        for i in range(cnt):
            pl.unsafe_store(i, losses[i])
    return PythonObject(cnt)
