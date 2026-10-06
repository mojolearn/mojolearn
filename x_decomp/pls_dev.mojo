# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""x_decomp/pls.mojo's component loop on resident device matrices (lane
py-runtime-b): the driver text of pls.mojo on `DKit` (the same cells and
reads; the SVD is DevExec's solve with its order home). GPU binding only."""
from std.python import Python, PythonObject
from experiments.classical_identical_ideas.linear_controls import C27_NORM_VECTOR
from x_decomp.cells import OP_CLASSICAL_NORM
from std.python._cpython import GILReleased
from x_decomp.api import _f, _n
from x_decomp.kit import (
    Mat, OP_ABS, OP_ADDS, OP_DIV, OP_GTS, OP_LE, OP_MUL, OP_RECIP, OP_SCALE, OP_SELECT, OP_SQRT, OP_SUB,
    mat_from,
)
from x_decomp.kit_device import DKit, DMat
from x_decomp.pls import PLS_F32_EPS, PlsArgs, _f32, _pls_args


def thin_svd_dev(mut k: DKit, X: DMat, nc: Int, u_based: Bool, mut U: DMat, mut S: DMat, mut Vt: DMat) raises:
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


def pinv_dev(mut k: DKit, A: DMat) raises -> DMat:
    """`_pinv(k, A)`: V diag(1/s) U^T, values at or below max(shape) eps s_max dropped."""
    var r = min(A.r, A.c)
    var U = k.zeros(0, 0)
    var S = k.zeros(0, 0)
    var Vt = k.zeros(0, 0)
    thin_svd_dev(k, A, r, True, U, S, Vt)
    var cond = Float64(max(A.r, A.c)) * PLS_F32_EPS
    var cut = _f32(k.word(k.cols(S, 0, 1)) * cond) if r > 0 else 0.0
    var inv = k.ew1(OP_RECIP, k.ew3(OP_SELECT, S, S, k.zeros(1, 1), cut), 0.0)
    return k.mm(Vt, k.ew2(OP_MUL, U, inv), True, True)


def _dot_dev(mut k: DKit, a: DMat, b: DMat) raises -> DMat:
    return k.mm(a, b, True, False)


def power_dev(mut k: DKit, X: DMat, Y: DMat, a: PlsArgs, mut xw: DMat, mut yw: DMat, mut stop: Bool) raises -> Int:
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
        Xp = pinv_dev(k, X)
        Yp = pinv_dev(k, Y)
    var xw_old = k.zeros(0, 0)
    var have_old = False
    var it = 0
    for i in range(1, a.max_iter + 1):
        it = i
        if a.mode_b:
            xw = k.mm(Xp, y_score, False, False)
        else:
            xw = k.ew2(OP_DIV, k.mm(X, y_score, True, False), _dot_dev(k, y_score, y_score))
        comptime if C27_NORM_VECTOR:
            xw = k.ew3(OP_CLASSICAL_NORM, xw, _dot_dev(k, xw, xw), k.zeros(1, 1), eps)
        else:
            xw = k.ew2(OP_DIV, xw, k.ew1(OP_ADDS, k.ew1(OP_SQRT, _dot_dev(k, xw, xw), 0.0), eps))
        var x_score = k.mm(X, xw, False, False)
        if a.mode_b:
            yw = k.mm(Yp, x_score, False, False)
        else:
            yw = k.ew2(OP_DIV, k.mm(Y, x_score, True, False), _dot_dev(k, x_score, x_score))
        if a.norm_y:
            comptime if C27_NORM_VECTOR:
                yw = k.ew3(OP_CLASSICAL_NORM, yw, _dot_dev(k, yw, yw), k.zeros(1, 1), eps)
            else:
                yw = k.ew2(OP_DIV, yw, k.ew1(OP_ADDS, k.ew1(OP_SQRT, _dot_dev(k, yw, yw), 0.0), eps))
        y_score = k.ew2(OP_DIV, k.mm(Y, yw, False, False), k.ew1(OP_ADDS, _dot_dev(k, yw, yw), eps))
        if Y.c == 1:
            break
        if have_old:
            var diff = k.ew2(OP_SUB, xw, xw_old)
            if k.word(_dot_dev(k, diff, diff)) < a.tol:
                break
        else:
            var diff = k.ew1(OP_ADDS, xw, -100.0)
            if k.word(_dot_dev(k, diff, diff)) < a.tol:
                break
        xw_old = k.copy(xw)
        have_old = True
    return it


def pls_components_dev(
    mut k: DKit, var Xk: DMat, var Yk: DMat, a: PlsArgs, mut outs: List[DMat], mut its: List[Int]
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
            thin_svd_dev(k, Cxy, 1, True, U, S, Vt)
            xw = k.cols(U, 0, 1)
            yw = k.vec_t(k.rows(Vt, 0, 1))
        else:
            var stop = False
            var it = power_dev(k, Xk, Yk, a, xw, yw, stop)
            if stop:
                break
            its.append(it)
        var sg = k.absmax_signs(xw, True)
        xw = k.ew2(OP_MUL, xw, sg)
        yw = k.ew2(OP_MUL, yw, sg)
        var x_scores = k.mm(Xk, xw, False, False)
        var y_ss = k.const(1.0, 1, 1) if a.norm_y else _dot_dev(k, yw, yw)
        var y_scores = k.ew2(OP_DIV, k.mm(Yk, yw, False, False), y_ss)
        var x_load = k.ew2(OP_DIV, k.mm(Xk, x_scores, True, False), _dot_dev(k, x_scores, x_scores))
        Xk = k.ew2(OP_SUB, Xk, k.mm(x_scores, x_load, False, True))
        var y_load: DMat
        if a.canonical:
            y_load = k.ew2(OP_DIV, k.mm(Yk, y_scores, True, False), _dot_dev(k, y_scores, y_scores))
            Yk = k.ew2(OP_SUB, Yk, k.mm(y_scores, y_load, False, True))
        else:
            y_load = k.ew2(OP_DIV, k.mm(Yk, x_scores, True, False), _dot_dev(k, x_scores, x_scores))
            Yk = k.ew2(OP_SUB, Yk, k.mm(x_scores, y_load, False, True))
        outs.append(xw^)
        outs.append(yw^)
        outs.append(x_scores^)
        outs.append(y_scores^)
        outs.append(x_load^)
        outs.append(y_load^)
        done += 1
    return done


def pls_fit_dev_py(
    x: PythonObject, y: PythonObject, outs: PythonObject, its: PythonObject, p: PythonObject, f: PythonObject
) raises -> PythonObject:
    """`pls_fit_py` on the resident kit."""
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
        var k = DKit()
        var res = List[DMat]()
        var il = List[Int]()
        done = pls_components_dev(k, k.upload(mat_from(pxa, n, px)), k.upload(mat_from(pya, n, q)), a, res, il)
        var hs = List[Mat]()
        for i in range(len(res)):  # small-loop(i: six vectors per component): their words home
            hs.append(k.get(res[i]))
        k.sync()
        for c in range(done):  # small-loop(done: components): column c of each output
            for t in range(6):  # small-loop(t: six outputs): exact copies
                var po = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=dst[t])
                ref v = hs[c * 6 + t]
                for i in range(v.n()):
                    po.unsafe_store(i * a.nc + c, v.d[i])
        nits = len(il)
        for i in range(nits):  # small-loop(nits: components): the iteration counts
            pit.unsafe_store(i, Int32(il[i]))
    return Python.tuple(done, nits)
