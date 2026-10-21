# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""x_decomp/lle_iter.mojo's subspace iteration on resident device matrices
(lane py-runtime-b): the same statements on `DKit` (the triangular solves
by `launch_trisolve` on the resident factor). The Householder QR and the
SVD take DevExec's host-address forms, as Python's kit called them (the
block comes home, the same launches run, the factor goes back up), and the
subspace change is one word home. GPU binding only."""
from std.math import inf, sqrt
from gemm.afn_apple_fast import AFN_GEMM_APPLE
from x_decomp.cells import F32Ptr, I32Ptr
from x_decomp.device import DevExec
from x_decomp.fa_em import svd_desc
from x_decomp.kit import Mat, OP_ABS, OP_ADDS, OP_MUL, OP_RECIP, OP_SELECT, OP_SQ, OP_SUB, mat_const, mat_from
from x_decomp.kit_device import DKit, DMat
from x_decomp.lle_iter import LleArgs, take_want
from x_decomp.select_ops import SEL_MAX, sel_fold


def lle_mm_dev(k: DKit, A: DMat, B: DMat, ta: Bool = False, tb: Bool = False) raises -> DMat:
    # Shift-invert amplifies error in the null-space projection, and Ritz
    # residuals/subspace comparisons approach float32's null floor. Splitting
    # these products into unordered MMA atomics makes unchanged operands
    # alternate convergence/refusal. Use existing ordered cells throughout
    # this iteration at every shape; solver, subspace size and limits stay.
    comptime if AFN_GEMM_APPLE:
        return k.mm_ordered(A, B, ta, tb)
    else:
        return k.mm(A, B, ta, tb)


def lle_orth_dev(mut k: DKit, Z: DMat) raises -> DMat:
    var rc = k.ew1(OP_RECIP, k.colsum(k.ew1(OP_ABS, Z, 0.0)), 0.0)
    var one = k.upload(mat_const(1.0, 1, 1))
    var sc = k.ew3(OP_SELECT, rc, rc, one, 0.0)
    var hd = k.ew2(OP_MUL, Z, sc)
    var h = k.get(hd)
    k.sync()
    var kk = min(h.r, h.c)
    var tau = Mat(1, kk)
    DevExec.geqrf(h.p(), tau.p(), h.r, h.c)
    var Q = Mat(h.r, Z.c)
    DevExec.orgqr(h.p(), tau.p(), Q.p(), h.r, h.c, kk, Z.c)
    return k.upload(Q)


def lle_iterate_dev(
    mut k: DKit, F0: DMat, lu: DMat, pm: DMat, im: DMat, z: DMat, a: LleArgs, mut X: DMat, mut Y: DMat,
    mut S: Mat, mut e_last: Float64,
) raises -> Int:
    var n = a.n
    var p = a.p
    X = lle_orth_dev(k, k.ew1(OP_ADDS, k.rand(a.n1, p, a.seed, 0x11E, 0), -0.5))
    var prev = DMat(0, 0)
    var have_prev = False
    var e_prev = inf[DType.float64]()
    for it in range(max(1, a.max_iter)):
        var T = k.trisolve(lu, im, k.pad_zero_row(X), 1)
        T = lle_orth_dev(k, k.ew2(OP_SUB, T, lle_mm_dev(k, z, lle_mm_dev(k, z, T, True, False), False, False)))
        var U = k.trisolve(lu, pm, T, 0)
        X = lle_orth_dev(k, k.rows(U, 0, a.n1))
        var B = lle_mm_dev(k, F0, k.pad_zero_row(X), False, False)
        var s = Mat(0, 0)
        var v = Mat(0, 0)
        k.svd_host(B, s, v)
        var Vt = Mat(0, 0)
        S = Mat(0, 0)
        svd_desc(s, v, S, Vt)
        X = lle_mm_dev(k, X, k.upload(Vt), False, True)
        var hx = k.get(X)
        k.sync()
        Y = k.upload(take_want(hx, p, a.nc))
        var sw = take_want(S, p, a.nc)
        if it >= 2 and Float64(sel_fold(SEL_MAX, sw.p(), 0, sw.n())) <= a.floor:
            return 1
        if have_prev:
            var Em = k.ew2(OP_SUB, Y, lle_mm_dev(k, prev, lle_mm_dev(k, prev, Y, True, False), False, False))
            var t = k.word(k.total(k.ew1(OP_SQ, Em, 0.0)))
            var e = sqrt(t if not (0.0 > t) else 0.0)
            if e <= a.sub_tol or (e <= a.stall_tol and e >= e_prev):
                return 0
            e_prev = e
        prev = k.copy(Y)
        have_prev = True
    e_last = e_prev
    return 2


def lle_iterate_dev_run(
    pf: F32Ptr, pl: F32Ptr, ppm: F32Ptr, pim: F32Ptr, pz: F32Ptr, px: F32Ptr, py_: F32Ptr, ps: F32Ptr, a: LleArgs,
    mut e: Float64,
) raises -> Int:
    """`lle_iterate_run` on the resident kit."""
    var n = a.n
    if a.n1 != n - 1 or a.p < a.nc or a.p > a.n1 or n * n > 2147483647:
        raise Error("x_decomp: lle iterate shape out of range")
    var k = DKit()
    var F = k.upload(mat_from(pf, n, n))
    var L = k.upload(mat_from(pl, n, n))
    var P = k.upload(mat_from(ppm, n, 1))
    var I = k.upload(mat_from(pim, n, 1))
    var Z = k.upload(mat_from(pz, n, 1))
    var X = DMat(0, 0)
    var Y = DMat(0, 0)
    var S = Mat(0, 0)
    var st = lle_iterate_dev(k, F, L, P, I, Z, a, X, Y, S, e)
    var hx = k.get(X)
    var hy = k.get(Y)
    k.sync()
    for i in range(hx.n()):
        px.unsafe_store(i, hx.d[i])
    for i in range(hy.n()):
        py_.unsafe_store(i, hy.d[i])
    for i in range(S.n()):
        ps.unsafe_store(i, S.d[i])
    return st


def lle_gemm_ordered_run(pa: F32Ptr, pb: F32Ptr, po: F32Ptr, m: Int, kk: Int, n: Int, ta: Bool, tb: Bool) raises:
    """LLE's near-null construction/products through the existing ordered GPU cells."""
    if m < 0 or kk < 0 or n < 0 or max(m * kk, max(kk * n, m * n)) > 2147483647:
        raise Error("x_decomp: LLE product shape out of range")
    var k = DKit()
    var A = k.upload(mat_from(pa, kk if ta else m, m if ta else kk))
    var B = k.upload(mat_from(pb, n if tb else kk, kk if tb else n))
    var C = k.mm_ordered(A, B, ta, tb)
    var host = k.get(C)
    k.sync()
    for j in range(host.n()):
        po.unsafe_store(j, host.d[j])


def lle_lu_ordered_run(pa: F32Ptr, pp: I32Ptr, pi: F32Ptr, n: Int) raises:
    """LLE's ill-conditioned spectral transform, scalar trailing updates on GPU."""
    if n < 0 or n * n > 2147483647:
        raise Error("x_decomp: LLE factor shape out of range")
    DevExec.lu_ordered(pa, pp, pi, n)
