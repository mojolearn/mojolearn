# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""x_decomp/nmf.mojo's NMF solver loops on resident device matrices (lane
py-runtime-b): the same statements on `DKit` (x_decomp/kit_device.mojo),
the same cells on the same values, so the same words as the host column.
Each scalar Python read (an error, a violation sum, a shuffle order) is one
word home, where Python read it. GPU binding only."""
from experiments.classical_identical_ideas.linear_controls import C26_PRODUCTS, C26_UPDATE_FUSED
from x_decomp.cells import OP_CLASSICAL_MU_IS
from std.math import sqrt
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from x_decomp.api import _f, _n
from x_decomp.kit import (
    Mat, OP_ABS, OP_ADD, OP_ADDS, OP_AXPY, OP_DIV, OP_GTS, OP_LOGS, OP_MAXS, OP_MUL, OP_MUZ, OP_RECIP,
    OP_MINS, OP_SCALE, OP_SELECT, OP_SQ, OP_SQDIFF, OP_SQRT, mat_const, mat_eye, mat_from,
)
from x_decomp.kit_device import DKit, DMat
from x_decomp.nmf import NMF_F32_EPS, NMF_LOG_FLOOR, NMF_ZERO_ERR0, NmfArgs, _f32_prod, _nmf_args, _nrm_or0


def nmf_err_dev(mut k: DKit, M: DMat, W: DMat, H: DMat, beta: Float64) raises -> Float64:
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


def _mu_ratio_dev(mut k: DKit, M: DMat, W: DMat, H: DMat, beta: Float64, mut P: DMat) raises -> DMat:
    """`NMF._mu_ratio`: X / WH (KL) or X / WH^2 (IS) with WH clamped at
    EPSILON; P = WH^-1 for IS."""
    var WH = k.ew1(OP_MAXS, k.mm(W, H, False, False), NMF_F32_EPS)
    if beta == 1.0:
        return k.ew2(OP_DIV, M, WH)
    P = k.ew1(OP_RECIP, WH, 0.0)
    return k.ew2(OP_DIV, k.ew2(OP_DIV, M, WH), WH)


def nmf_mu_dev(M: DMat, mut W: DMat, mut H: DMat, a: NmfArgs) raises -> Int:
    """`NMF._mu` (and `_mu_beta` for beta 1 / 0). Returns the iteration
    count, or NMF_ZERO_ERR0."""
    var k = DKit()
    var beta = a.beta
    var err0 = nmf_err_dev(k, M, W, H, beta)
    var prev = err0
    var it = 0
    var eps = k.upload(mat_const(NMF_F32_EPS, 1, 1))
    var one = k.upload(mat_const(1.0, 1, 1))
    var frozen_num = DMat(0, 0)
    var frozen_gram = DMat(0, 0)
    comptime if C26_PRODUCTS:
        if not a.update_h and beta == 2.0:
            frozen_num = k.mm(M, H, False, True)
            frozen_gram = k.mm(H, H, False, True)
    for i in range(1, a.max_iter + 1):
        it = i
        if beta == 2.0:
            var num: DMat
            var den: DMat
            if C26_PRODUCTS and not a.update_h:
                num = k.copy(frozen_num)
                den = k.mm(W, frozen_gram, False, False)
            else:
                num = k.mm(M, H, False, True)
                den = k.mm(W, k.mm(H, H, False, True), False, False)
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
            var P = DMat(0, 0)
            var R = _mu_ratio_dev(k, M, W, H, beta, P)
            var num = k.mm(R, H, False, True)
            var den: DMat
            if beta == 1.0:
                den = k.ew2(OP_ADD, k.zeros(W.r, W.c), k.vec_t(k.rowsum(H)))
            else:
                den = k.mm(P, H, False, True)
            if a.l1w > 0:
                den = k.ew1(OP_ADDS, den, a.l1w)
            if a.l2w > 0:
                den = k.ew2s(OP_AXPY, den, W, a.l2w)
            comptime if C26_UPDATE_FUSED:
                W = k.ew3(OP_CLASSICAL_MU_IS if beta == 0.0 else OP_MUZ, W, num, den, NMF_F32_EPS)
            else:
                den = k.ew3(OP_SELECT, k.ew1(OP_ABS, den, 0.0), den, eps, 0.0)
                var delta = k.ew2(OP_DIV, num, den)
                if beta == 0.0:
                    delta = k.ew1(OP_SQRT, delta, 0.0)
                W = k.ew2(OP_MUL, W, delta)
            if a.update_h:
                R = _mu_ratio_dev(k, M, W, H, beta, P)
                num = k.mm(W, R, True, False)
                if beta == 1.0:
                    var ws = k.colsum(W)
                    ws = k.ew3(OP_SELECT, k.ew1(OP_ABS, ws, 0.0), ws, one, 0.0)
                    den = k.ew2(OP_ADD, k.zeros(H.r, H.c), k.vec_t(ws^))
                else:
                    den = k.mm(W, P, True, False)
                if a.l1h > 0:
                    den = k.ew1(OP_ADDS, den, a.l1h)
                if a.l2h > 0:
                    den = k.ew2s(OP_AXPY, den, H, a.l2h)
                comptime if C26_UPDATE_FUSED:
                    H = k.ew3(OP_CLASSICAL_MU_IS if beta == 0.0 else OP_MUZ, H, num, den, NMF_F32_EPS)
                else:
                    den = k.ew3(OP_SELECT, k.ew1(OP_ABS, den, 0.0), den, eps, 0.0)
                    var delta = k.ew2(OP_DIV, num, den)
                    if beta == 0.0:
                        delta = k.ew1(OP_SQRT, delta, 0.0)
                    H = k.ew2(OP_MUL, H, delta)
        if a.tol > 0 and i % 10 == 0:
            var err = nmf_err_dev(k, M, W, H, beta)
            if err0 == 0.0:
                return NMF_ZERO_ERR0
            if (prev - err) / err0 < a.tol:
                break
            prev = err
    return it


def _perm_dev(mut k: DKit, kc: Int, a: NmfArgs, mut draws: Int) raises -> List[Int32]:
    """`NMF._perm`: the identity, or with shuffle a Philox draw's stable order."""
    if not a.shuffle:
        var p = List[Int32](length=max(kc, 1), fill=Int32(0))
        for j in range(kc):  # small-loop(kc: components): the identity coordinate order
            p[j] = Int32(j)
        return p^
    draws += 1
    return k.order_small(k.rand(1, kc, a.seed, 200 + draws, 0))


def _cd_side_dev(mut k: DKit, M: DMat, mut W: DMat, Ht: DMat, l1: Float64, l2: Float64,
                      perm: List[Int32], trans: Bool) raises -> Float64:
    """`NMF._cd_side`: one half sweep of W against Ht (W in place)."""
    var HHt = k.mm(Ht, Ht, True, False)
    var XHt = k.mm(M, Ht, trans, False)
    if l2 != 0.0:
        HHt = k.ew2s(OP_AXPY, HHt, k.upload(mat_eye(HHt.r)), l2)
    if l1 != 0.0:
        XHt = k.ew1(OP_ADDS, XHt, -l1)
    return k.cd_rows(W, HHt, XHt, perm)


def nmf_cd_dev(M: DMat, mut W: DMat, mut H: DMat, a: NmfArgs) raises -> Int:
    """`NMF._cd`: W and (update_H) H replaced. Returns the iteration count."""
    var k = DKit()
    var draws = 0
    var Ht = k.t(H)
    var v_init = 0.0
    var have_init = False
    var it = 0
    var fixed_gram = DMat(0, 0)
    var fixed_cross = DMat(0, 0)
    comptime if C26_PRODUCTS:
        if not a.update_h:
            fixed_gram = k.mm(Ht, Ht, True, False)
            fixed_cross = k.mm(M, Ht, False, False)
            if a.l2w != 0.0:
                fixed_gram = k.ew2s(OP_AXPY, fixed_gram, k.upload(mat_eye(Ht.c)), a.l2w)
            if a.l1w != 0.0:
                fixed_cross = k.ew1(OP_ADDS, fixed_cross, -a.l1w)
    for i in range(1, a.max_iter + 1):
        it = i
        var perm = _perm_dev(k, W.c, a, draws)
        var viol = 0.0
        if C26_PRODUCTS and not a.update_h:
            viol = k.cd_rows(W, fixed_gram, fixed_cross, perm)
        else:
            viol = _cd_side_dev(k, M, W, Ht, a.l1w, a.l2w, perm, False)
        if a.update_h:
            viol += _cd_side_dev(k, M, Ht, W, a.l1h, a.l2h, _perm_dev(k, W.c, a, draws), True)
        if not have_init:
            v_init = viol
            have_init = True
        if v_init == 0:
            break
        if viol / v_init <= a.tol:
            break
    if a.update_h:
        H = k.t(Ht)
    return it


def nmf_solve_dev_py(
    x: PythonObject, w: PythonObject, h: PythonObject, p: PythonObject, f: PythonObject
) raises -> PythonObject:
    """`nmf_solve_py` on the resident kit."""
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
        var k = DKit()
        var X = k.upload(mat_from(px, n, d))
        var W = k.upload(mat_from(pw, n, nc))
        var H = k.upload(mat_from(ph, nc, d))
        if solver == 0:
            it = nmf_cd_dev(X, W, H, a)
        else:
            it = nmf_mu_dev(X, W, H, a)
        var hw = k.get(W)
        var hh = k.get(H)
        k.sync()
        for i in range(n * nc):
            pw.unsafe_store(i, hw.d[i])
        for i in range(nc * d):
            ph.unsafe_store(i, hh.d[i])
    return PythonObject(it)


def nmf_nndsvd_dev_py(
    u: PythonObject, s: PythonObject, vt: PythonObject, w: PythonObject, h: PythonObject, p: PythonObject
) raises -> PythonObject:
    """`nmf_nndsvd_py` on the resident kit (the same cells per component;
    U's column j taken as the row of U^T, the same words in an n x 1 shape)."""
    var n = _n(p, 0)
    var d = _n(p, 1)
    var r = _n(p, 2)
    var nc = _n(p, 3)
    if nc > r or n * r > 2147483647 or r * d > 2147483647 or n * nc > 2147483647:
        raise Error("x_decomp: nndsvd shape out of range")
    var pu = _f(u)
    var ps = _f(s)
    var pv = _f(vt)
    var pw = _f(w)
    var ph = _f(h)
    with GILReleased(Python()):
        var k = DKit()
        var Ut = k.t(k.upload(mat_from(pu, n, r)))
        var Sd = k.upload(mat_from(ps, r, 1))
        var V = k.upload(mat_from(pv, r, d))
        var Wt = k.zeros(nc, n)
        var H = k.zeros(nc, d)
        for j in range(nc):  # small-loop(nc: components): one factor pair per component, each a whole-matrix cell chain
            var x = k.vec_t(k.rows(Ut, j, j + 1))
            var y = k.rows(V, j, j + 1)
            var sj = k.rows(Sd, j, j + 1)
            var wc: DMat
            var hr: DMat
            if j == 0:
                var rr = k.ew1(OP_SQRT, sj, 0.0)
                wc = k.ew2(OP_MUL, k.ew1(OP_ABS, x, 0.0), rr)
                hr = k.ew2(OP_MUL, k.ew1(OP_ABS, y, 0.0), rr)
            else:
                var xp = k.ew1(OP_MAXS, x, 0.0)
                var yp = k.ew1(OP_MAXS, y, 0.0)
                var xn = k.ew1(OP_ABS, k.ew1(OP_MINS, x, 0.0), 0.0)
                var yn = k.ew1(OP_ABS, k.ew1(OP_MINS, y, 0.0), 0.0)
                var xpn = _vnorm_dev(k, xp)
                var ypn = _vnorm_dev(k, yp)
                var xnn = _vnorm_dev(k, xn)
                var ynn = _vnorm_dev(k, yn)
                var mp = _f32_prod(xpn, ypn)
                var mn = _f32_prod(xnn, ynn)
                var uu: DMat
                var vv: DMat
                var sigma: Float64
                if mp > mn:
                    uu = k.ew1(OP_SCALE, xp, _nrm_or0(xpn))
                    vv = k.ew1(OP_SCALE, yp, _nrm_or0(ypn))
                    sigma = mp
                else:
                    uu = k.ew1(OP_SCALE, xn, _nrm_or0(xnn))
                    vv = k.ew1(OP_SCALE, yn, _nrm_or0(ynn))
                    sigma = mn
                var lbd = k.ew1(OP_SQRT, k.ew1(OP_SCALE, sj, sigma), 0.0)
                wc = k.ew2(OP_MUL, uu, lbd)
                hr = k.ew2(OP_MUL, vv, lbd)
            k.place_row(Wt, wc, j)
            k.place_row(H, hr, j)
        var Wd = k.t(Wt)
        var hw = k.get(Wd)
        var hh = k.get(H)
        k.sync()
        _ = Wd^
        for i in range(n * nc):
            pw.unsafe_store(i, hw.d[i])
        for i in range(nc * d):
            ph.unsafe_store(i, hh.d[i])
    return PythonObject(nc)


def _vnorm_dev(mut k: DKit, v: DMat) raises -> Float64:
    return k.word(k.ew1(OP_SQRT, k.total(k.ew1(OP_SQ, v, 0.0)), 0.0))
