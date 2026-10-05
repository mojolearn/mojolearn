# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""NMF's solver loops in Mojo (lane py-runtime-b, 2026-10-05): `_mu`
(frobenius), `_mu_beta` (KL and IS) and `_cd` of
`_expansion_decomp.NMF`, statement for statement on `Kit[E]`
(x_decomp/kit.mojo): the same cells with the same operands, broadcast modes
and float32 scalars (each Python double rounded once, as the binding
boundary rounded it), and Python's float64 scalar tail (the error, the
convergence ratios, the violation sums) in Float64 in the same order. So the
IDENTICAL words are the Python driver's on every column. The GPU binding
runs x_decomp/nmf_dev.mojo, the same statements on resident matrices.

A zero first error (`err0`) made Python's convergence ratio raise
ZeroDivisionError; the drivers return NMF_ZERO_ERR0 and the caller raises it."""
from std.math import sqrt
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from x_decomp.api import _f, _n
from x_decomp.exec_trait import Exec
from x_decomp.kit import (
    Kit, Mat, OP_ABS, OP_ADD, OP_ADDS, OP_AXPY, OP_DIV, OP_GTS, OP_LOGS, OP_MAXS, OP_MUL, OP_MUZ, OP_RECIP,
    OP_SELECT, OP_SQDIFF, OP_SQRT, mat_const, mat_eye, mat_from, mat_t, mat_vec_t,
)

#: `_expansion_decomp._F32_EPS`
comptime NMF_F32_EPS: Float64 = 1.1920928955078125e-07
#: the `logs` floor of `_err` (FLT_MIN)
comptime NMF_LOG_FLOOR: Float64 = 1.1754943508222875e-38
#: the driver's return when err0 is 0 (Python's ZeroDivisionError)
comptime NMF_ZERO_ERR0 = -1


struct NmfArgs(Copyable, Movable):
    """The solver's Python scalars: beta 2 / 1 / 0, update_H, max_iter, tol,
    the regularization (l1W, l1H, l2W, l2H), shuffle and the seed."""
    var beta: Float64
    var update_h: Bool
    var max_iter: Int
    var tol: Float64
    var l1w: Float64
    var l1h: Float64
    var l2w: Float64
    var l2h: Float64
    var shuffle: Bool
    var seed: Int

    def __init__(out self, beta: Float64, update_h: Bool, max_iter: Int, tol: Float64, l1w: Float64,
                 l1h: Float64, l2w: Float64, l2h: Float64, shuffle: Bool, seed: Int):
        self.beta = beta
        self.update_h = update_h
        self.max_iter = max_iter
        self.tol = tol
        self.l1w = l1w
        self.l1h = l1h
        self.l2w = l2w
        self.l2h = l2h
        self.shuffle = shuffle
        self.seed = seed


def nmf_err[E: Exec](k: Kit[E], M: Mat, W: Mat, H: Mat, beta: Float64) raises -> Float64:
    """`NMF._err`: sklearn `_beta_divergence(..., square_root=True)`."""
    if beta == 2.0:
        return k.word(k.ew1(OP_SQRT, k.total(k.ew2(OP_SQDIFF, M, k.mm(W, H, False, False))), 0.0))
    var WH = k.ew1(OP_MAXS, k.mm(W, H, False, False), NMF_F32_EPS)
    var keep = k.ew1(OP_GTS, M, NMF_F32_EPS)
    var div = k.ew2(OP_DIV, M, WH)
    var res: Float64
    if beta == 1.0:
        var sum_wh = k.word(k.mm(k.colsum(W), k.rowsum(H), False, False))
        var xlog = k.word(k.total(k.ew2(OP_MUL, k.ew2(OP_MUL, M, k.ew1(OP_LOGS, div, NMF_LOG_FLOOR)), keep)))
        var xs = k.word(k.total(k.ew2(OP_MUL, M, keep)))
        res = xlog + sum_wh - xs
    else:
        var dsum = k.word(k.total(k.ew2(OP_MUL, div, keep)))
        var lsum = k.word(k.total(k.ew2(OP_MUL, k.ew1(OP_LOGS, div, NMF_LOG_FLOOR), keep)))
        res = dsum - Float64(M.r * M.c) - lsum
    return sqrt(2 * res) if res > 0 else 0.0


def _mu_ratio[E: Exec](k: Kit[E], M: Mat, W: Mat, H: Mat, beta: Float64, mut P: Mat) raises -> Mat:
    """`NMF._mu_ratio`: X / WH (KL) or X / WH^2 (IS) with WH clamped at
    EPSILON; P = WH^-1 for IS."""
    var WH = k.ew1(OP_MAXS, k.mm(W, H, False, False), NMF_F32_EPS)
    if beta == 1.0:
        return k.ew2(OP_DIV, M, WH)
    P = k.ew1(OP_RECIP, WH, 0.0)
    return k.ew2(OP_DIV, k.ew2(OP_DIV, M, WH), WH)


def nmf_mu[E: Exec](M: Mat, mut W: Mat, mut H: Mat, a: NmfArgs) raises -> Int:
    """`NMF._mu` (and `_mu_beta` for beta 1 / 0). Returns the iteration
    count, or NMF_ZERO_ERR0."""
    var k = Kit[E]()
    var beta = a.beta
    var err0 = nmf_err(k, M, W, H, beta)
    var prev = err0
    var it = 0
    var eps = mat_const(NMF_F32_EPS, 1, 1)
    var one = mat_const(1.0, 1, 1)
    for i in range(1, a.max_iter + 1):
        it = i
        if beta == 2.0:
            var num = k.mm(M, H, False, True)
            var den = k.mm(W, k.mm(H, H, False, True), False, False)
            if a.l1w > 0:
                den = k.ew1(OP_ADDS, den, a.l1w)
            if a.l2w > 0:
                den = k.ew2s(OP_AXPY, den, W, a.l2w)
            W = k.ew3(OP_MUZ, W, num, den, NMF_F32_EPS)
            if a.update_h:
                num = k.mm(W, M, True, False)
                den = k.mm(k.mm(W, W, True, False), H, False, False)
                if a.l1h > 0:
                    den = k.ew1(OP_ADDS, den, a.l1h)
                if a.l2h > 0:
                    den = k.ew2s(OP_AXPY, den, H, a.l2h)
                H = k.ew3(OP_MUZ, H, num, den, NMF_F32_EPS)
        else:
            var P = Mat(0, 0)
            var R = _mu_ratio(k, M, W, H, beta, P)
            var num = k.mm(R, H, False, True)
            var den: Mat
            if beta == 1.0:
                den = k.ew2(OP_ADD, Mat(W.r, W.c), mat_vec_t(k.rowsum(H)))
            else:
                den = k.mm(P, H, False, True)
            if a.l1w > 0:
                den = k.ew1(OP_ADDS, den, a.l1w)
            if a.l2w > 0:
                den = k.ew2s(OP_AXPY, den, W, a.l2w)
            den = k.ew3(OP_SELECT, k.ew1(OP_ABS, den, 0.0), den, eps, 0.0)
            var delta = k.ew2(OP_DIV, num, den)
            if beta == 0.0:
                delta = k.ew1(OP_SQRT, delta, 0.0)
            W = k.ew2(OP_MUL, W, delta)
            if a.update_h:
                R = _mu_ratio(k, M, W, H, beta, P)
                num = k.mm(W, R, True, False)
                if beta == 1.0:
                    var ws = k.colsum(W)
                    ws = k.ew3(OP_SELECT, k.ew1(OP_ABS, ws, 0.0), ws, one, 0.0)
                    den = k.ew2(OP_ADD, Mat(H.r, H.c), mat_vec_t(ws^))
                else:
                    den = k.mm(W, P, True, False)
                if a.l1h > 0:
                    den = k.ew1(OP_ADDS, den, a.l1h)
                if a.l2h > 0:
                    den = k.ew2s(OP_AXPY, den, H, a.l2h)
                den = k.ew3(OP_SELECT, k.ew1(OP_ABS, den, 0.0), den, eps, 0.0)
                delta = k.ew2(OP_DIV, num, den)
                if beta == 0.0:
                    delta = k.ew1(OP_SQRT, delta, 0.0)
                H = k.ew2(OP_MUL, H, delta)
        if a.tol > 0 and i % 10 == 0:
            var err = nmf_err(k, M, W, H, beta)
            if err0 == 0.0:
                return NMF_ZERO_ERR0
            if (prev - err) / err0 < a.tol:
                break
            prev = err
    return it


def _perm[E: Exec](k: Kit[E], kc: Int, a: NmfArgs, mut draws: Int) raises -> List[Int32]:
    """`NMF._perm`: the identity, or with shuffle a Philox draw's stable order."""
    if not a.shuffle:
        var p = List[Int32](length=max(kc, 1), fill=Int32(0))
        for j in range(kc):  # small-loop(kc: components): the identity coordinate order
            p[j] = Int32(j)
        return p^
    draws += 1
    return k.order_small(k.rand(1, kc, a.seed, 200 + draws, 0))


def _cd_side[E: Exec](k: Kit[E], M: Mat, mut W: Mat, Ht: Mat, l1: Float64, l2: Float64,
                      perm: List[Int32], trans: Bool) raises -> Float64:
    """`NMF._cd_side`: one half sweep of W against Ht (W in place)."""
    var HHt = k.mm(Ht, Ht, True, False)
    var XHt = k.mm(M, Ht, trans, False)
    if l2 != 0.0:
        HHt = k.ew2s(OP_AXPY, HHt, mat_eye(HHt.r), l2)
    if l1 != 0.0:
        XHt = k.ew1(OP_ADDS, XHt, -l1)
    return k.cd_rows(W, HHt, XHt, perm)


def nmf_cd[E: Exec](M: Mat, mut W: Mat, mut H: Mat, a: NmfArgs) raises -> Int:
    """`NMF._cd`: W and (update_H) H replaced. Returns the iteration count."""
    var k = Kit[E]()
    var draws = 0
    var Ht = mat_t(H)
    var v_init = 0.0
    var have_init = False
    var it = 0
    for i in range(1, a.max_iter + 1):
        it = i
        var viol = _cd_side(k, M, W, Ht, a.l1w, a.l2w, _perm(k, W.c, a, draws), False)
        if a.update_h:
            viol += _cd_side(k, M, Ht, W, a.l1h, a.l2h, _perm(k, W.c, a, draws), True)
        if not have_init:
            v_init = viol
            have_init = True
        if v_init == 0:
            break
        if viol / v_init <= a.tol:
            break
    if a.update_h:
        H = mat_t(Ht)
    return it


def _nmf_args(p: PythonObject, f: PythonObject) raises -> NmfArgs:
    """p = [n, d, nc, solver (0 cd, 1 mu), update_H, max_iter, shuffle,
    seed & 0xFFFFFFFF]; f = [beta, tol, l1W, l1H, l2W, l2H]."""
    var fv = List[Float64]()
    for i in range(6):  # small-loop(i: six float parameters): Python parameter words, not data
        fv.append(Float64(py=f[i]))
    if fv[0] != 2.0 and fv[0] != 1.0 and fv[0] != 0.0:
        raise Error("x_decomp: nmf beta is 2, 1 or 0")
    return NmfArgs(fv[0], Int(py=p[4]) != 0, Int(py=p[5]), fv[1], fv[2], fv[3], fv[4], fv[5],
                   Int(py=p[6]) != 0, Int(py=p[7]))


def nmf_solve_py[E: Exec](
    x: PythonObject, w: PythonObject, h: PythonObject, p: PythonObject, f: PythonObject
) raises -> PythonObject:
    """`NMF._cd` / `_mu` over x (n x d): w (n x nc) and h (nc x d) replaced
    in place (h only when update_H). Returns the iteration count, or
    NMF_ZERO_ERR0 (the caller raises ZeroDivisionError)."""
    var n = _n(p, 0)
    var d = _n(p, 1)
    var nc = _n(p, 2)
    var solver = Int(py=p[3])
    if n * d > 2147483647 or n * nc > 2147483647 or nc * d > 2147483647:
        raise Error("x_decomp: nmf shape out of range")
    var a = _nmf_args(p, f)
    if solver == 0 and a.beta != 2.0:
        raise Error("x_decomp: nmf cd takes beta 2")
    var px = _f(x)
    var pw = _f(w)
    var ph = _f(h)
    var it = 0
    with GILReleased(Python()):
        var X = mat_from(px, n, d)
        var W = mat_from(pw, n, nc)
        var H = mat_from(ph, nc, d)
        if solver == 0:
            it = nmf_cd[E](X, W, H, a)
        else:
            it = nmf_mu[E](X, W, H, a)
        for i in range(n * nc):
            pw.unsafe_store(i, W.d[i])
        for i in range(nc * d):
            ph.unsafe_store(i, H.d[i])
    return PythonObject(it)
