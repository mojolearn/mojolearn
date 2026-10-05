# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""x_decomp/als.mojo's sweep loop on resident device matrices (lane
py-runtime-b), as the GPU binding's kit ran it: the exact solve with C
resident and the item half-sweep through strides (`launch_als_rows`, su = 1,
si = items: no transpose), the CG route on C and C^T (`als_cg_kernel`), the
same cells for the loss with its three words home. GPU binding only."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from x_decomp.als import AlsArgs, _als_args
from x_decomp.api import _f, _n
from x_decomp.device import TPB, _blocks, als_cg_kernel, als_scratch, launch_als_rows
from x_decomp.kit import (
    Mat, OP_ABS, OP_ADD, OP_ADDS, OP_GTS, OP_MUL, OP_SCALE, OP_SELECT, OP_SQ, mat_from,
)
from x_decomp.kit_device import DKit, DMat


def als_half_dev(k: DKit, C: DMat, Y: DMat, reg: Float64, trans: Bool) raises -> DMat:
    """`_Kit.als(C, Y, reg, trans)` with C resident."""
    var n = C.c if trans else C.r
    var m = C.r if trans else C.c
    var f = Y.c
    var YtY = k.mm(Y, Y, True, False)
    var X = DMat(n, f)
    var flags = DMat(n, 1)
    var su = 1 if trans else C.c
    var si = C.c if trans else 1
    if n > 0:
        var ns = als_scratch(n, f)
        var S = DMat(max(ns, 1), 1)
        launch_als_rows(k.ctx, C.p(), Y.p(), YtY.p(), X.p(), S.p(), flags.p(), n, m, f, su, si, Float32(reg))
        k.ctx.synchronize()
    return X^


def als_cg_half_dev(k: DKit, C: DMat, Y: DMat, X0: DMat, reg: Float64, cg: Int) raises -> DMat:
    """`_Kit.als_cg(C, Y, X0, reg, cg)` (the CG kernel DevExec launches)."""
    var n = C.r
    var m = C.c
    var f = Y.c
    var YtY = k.mm(Y, Y, True, False)
    var X = k.copy(X0)
    if n > 0:
        var S = DMat(max(n * 3 * f, 1), 1)
        var steps = DMat(n, 1)
        k.ctx.enqueue_function[als_cg_kernel](
            C.p(), Y.p(), YtY.p(), X.p(), S.p(), steps.p(), Int32(n), Int32(m), Int32(f), Float32(reg), Int32(cg),
            grid_dim=_blocks(n), block_dim=TPB,
        )
        k.ctx.synchronize()
    return X^


def als_loss_dev(mut k: DKit, C: DMat, X: DMat, Y: DMat, reg: Float64) raises -> Float64:
    var P = k.mm(X, Y, False, True)
    var seen = k.ew1(OP_GTS, k.ew1(OP_ABS, C, 0.0), 0.0)
    var obs = k.ew2(OP_MUL, C, k.ew1(OP_SQ, k.ew1(OP_ADDS, k.ew1(OP_SCALE, P, -1.0), 1.0), 0.0))
    var term = k.ew3(OP_SELECT, seen, obs, k.ew1(OP_SQ, P, 0.0), 0.5)
    var regm = k.ew2(OP_ADD, k.total(k.ew1(OP_SQ, X, 0.0)), k.total(k.ew1(OP_SQ, Y, 0.0)))
    var tot = k.word(k.ew2(OP_ADD, k.total(term), k.ew1(OP_SCALE, regm, reg)))
    var nnz = k.word(k.total(seen))
    var conf = k.word(k.total(k.ew2(OP_MUL, C, seen)))
    return tot / (conf + (Float64(C.r * C.c) - nnz))


def als_fit_dev_py(
    c: PythonObject, x: PythonObject, y: PythonObject, ls: PythonObject, p: PythonObject, f: PythonObject
) raises -> PythonObject:
    """`als_fit_py` on the resident kit."""
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
        var k = DKit()
        var C = k.upload(mat_from(pc, n, m))
        var X = k.upload(mat_from(px, n, nf))
        var Y = k.upload(mat_from(py_, m, nf))
        var losses = List[Float64]()
        if a.use_cg:
            var Ct = k.t(C)
            for _ in range(a.iterations):
                X = als_cg_half_dev(k, C, Y, X, a.reg, a.cg_steps)
                Y = als_cg_half_dev(k, Ct, X, Y, a.reg, a.cg_steps)
                if a.calc_loss:
                    losses.append(als_loss_dev(k, C, X, Y, a.reg))
        else:
            for _ in range(a.iterations):
                X = als_half_dev(k, C, Y, a.reg, False)
                Y = als_half_dev(k, C, X, a.reg, True)
                if a.calc_loss:
                    losses.append(als_loss_dev(k, C, X, Y, a.reg))
        var hx = k.get(X)
        var hy = k.get(Y)
        k.sync()
        for i in range(n * nf):
            px.unsafe_store(i, hx.d[i])
        for i in range(m * nf):
            py_.unsafe_store(i, hy.d[i])
        cnt = len(losses)
        for i in range(cnt):
            pl.unsafe_store(i, losses[i])
    return PythonObject(cnt)
