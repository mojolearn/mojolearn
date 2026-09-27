# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Seams DEVIATION 5310-5313, 5316 and 5321 of the decomp lane, the one-thread-per-
row solvers: NMF's coordinate descent (5310), the Lasso on the Gram (5311),
OMP's atom choice and its tie (5312), LDA's document update (5313) and the
implicit-ALS row solve (5316).

    tools/with_identical_mode.sh pixi run mojo run -I . x_decomp/checks/rows_check.mojo
"""
from core.identity_trace import IdentityTrace
from x_decomp.checks.xd_oracles import (
    oracle_als,
    oracle_als_cg,
    oracle_cd,
    oracle_gemm,
    oracle_lasso,
    oracle_lda_step,
    o_div0,
    o_sqrt0,
)
from x_decomp.checks.seam_util import (
    count_diff_f32,
    iptr,
    positive_fixture,
    ptr,
    require_separates,
    same,
    seam_fixture,
    zeros,
)
from x_decomp.device import DevExec
from x_decomp.host import HostExec


def tame(var x: List[Float32]) -> List[Float32]:
    for t in range(len(x)):
        if abs(x[t]) > Float32(100):
            x[t] = x[t] * Float32(1e-6)
    return x^


def main() raises:
    var tr = IdentityTrace()
    tr.header("x_decomp rows_check (DEVIATIONS 5310-5313, 5316, 5321)")
    var info = zeros(1)
    # ---- 5310 NMF coordinate descent, one sweep
    var n = 37
    var k = 5
    var d = 9
    var X = positive_fixture(n, d, 1)
    var Ht = positive_fixture(d, k, 2)
    var W0 = positive_fixture(n, k, 3)
    var HHt = oracle_gemm(Ht, Ht, k, d, k, True, False)
    var XHt = oracle_gemm(X, Ht, n, d, k, False, False)
    var want = oracle_cd(W0, HHt, XHt, n, k)
    require_separates("5310 CD coordinate order", count_diff_f32(want, oracle_cd(W0, HHt, XHt, n, k, 1)))
    var perm = List[Int32]()
    for s in range(k):
        perm.append(Int32(s))
    var wd = W0.copy()
    var viol = zeros(n)
    DevExec.cd_rows(ptr(wd), ptr(HHt), ptr(XHt), iptr(perm), ptr(viol), n, k)
    same("5310 cd_rows device", count_diff_f32(wd, want))
    var wh = W0.copy()
    HostExec.cd_rows(ptr(wh), ptr(HHt), ptr(XHt), iptr(perm), ptr(viol), n, k)
    same("5310 cd_rows host", count_diff_f32(wh, want))
    tr.record_list_f32("x_decomp.cd_rows", wd)
    # ---- 5311 Lasso CD on the Gram, three sweeps
    var D = tame(seam_fixture(k, d, 4))
    var Xs = tame(seam_fixture(n, d, 5))
    var G = oracle_gemm(D, D, k, d, k, False, True)
    var Q = oracle_gemm(Xs, D, n, d, k, False, True)
    var L0 = zeros(n * k)
    var alpha = Float32(0.05)
    var wl = oracle_lasso(G, Q, L0, n, k, alpha, 3)
    require_separates("5311 Lasso coordinate order", count_diff_f32(wl, oracle_lasso(G, Q, L0, n, k, alpha, 3, 1)))
    var its = zeros(n)
    var ld = L0.copy()
    var hs = zeros(n * k)
    DevExec.lasso_rows(ptr(G), ptr(Q), ptr(ld), ptr(hs), ptr(its), n, k, alpha, 3, Float32(0), False)
    same("5311 lasso_rows device", count_diff_f32(ld, wl))
    var lh = L0.copy()
    HostExec.lasso_rows(ptr(G), ptr(Q), ptr(lh), ptr(hs), ptr(its), n, k, alpha, 3, Float32(0), False)
    same("5311 lasso_rows host", count_diff_f32(lh, wl))
    tr.record_list_f32("x_decomp.lasso_rows", ld)
    # ---- 5312 OMP, one atom: the argmax tie
    var Gi = zeros(k * k)
    for j in range(k):
        Gi[j * k + j] = Float32(4)
    var Qt = zeros(n * k)
    for i in range(n):
        for j in range(k):
            Qt[i * k + j] = Float32(Int((i + j) % 3)) - Float32(1)   # ties at |1|
    var wo = zeros(n * k)
    var wo_alt = zeros(n * k)
    for i in range(n):
        var lam = 0
        var lam_alt = 0
        var best = Float32(-1)
        var best_alt = Float32(-1)
        for j in range(k):
            var v = abs(Qt[i * k + j])
            if v > best:
                best = v
                lam = j
            if v >= best_alt:
                best_alt = v
                lam_alt = j
        var Lr = o_sqrt0(Gi[lam * k + lam])
        wo[i * k + lam] = o_div0(o_div0(Qt[i * k + lam], Lr), Lr)
        wo_alt[i * k + lam_alt] = o_div0(o_div0(Qt[i * k + lam_alt], Lr), Lr)
    require_separates("5312 OMP argmax tie", count_diff_f32(wo, wo_alt))
    var od = zeros(n * k)
    var sc = zeros(n * (k * k + 3 * k))
    var na = zeros(n)
    DevExec.omp_rows(ptr(Gi), ptr(Qt), ptr(od), ptr(sc), ptr(na), n, k, 1)
    same("5312 omp_rows device", count_diff_f32(od, wo))
    var oh = zeros(n * k)
    HostExec.omp_rows(ptr(Gi), ptr(Qt), ptr(oh), ptr(sc), ptr(na), n, k, 1)
    same("5312 omp_rows host", count_diff_f32(oh, wo))
    tr.record_list_f32("x_decomp.omp_rows", od)
    # ---- 5313 LDA: one document update
    var v = 14
    var C = positive_fixture(n, v, 6)
    var EW = positive_fixture(k, v, 7)
    var D0 = positive_fixture(n, k, 8)
    var E0 = positive_fixture(n, k, 9)
    for t in range(len(D0)):
        D0[t] = D0[t] + Float32(0.1)
        E0[t] = E0[t] + Float32(0.1)
    var wd2 = oracle_lda_step(C, EW, D0, E0, n, k, v, Float32(0.2))
    require_separates("5313 LDA word-fold order", count_diff_f32(wd2, oracle_lda_step(C, EW, D0, E0, n, k, v, Float32(0.2), 1)))
    var dd = D0.copy()
    var ed = E0.copy()
    var sl = zeros(n * (v + k))
    DevExec.lda_rows(ptr(C), ptr(EW), ptr(dd), ptr(ed), ptr(sl), ptr(its), n, k, v, Float32(0.2), 1, Float32(0))
    same("5313 lda_rows device", count_diff_f32(dd, wd2))
    var dh = D0.copy()
    var eh = E0.copy()
    HostExec.lda_rows(ptr(C), ptr(EW), ptr(dh), ptr(eh), ptr(sl), ptr(its), n, k, v, Float32(0.2), 1, Float32(0))
    same("5313 lda_rows host", count_diff_f32(dh, wd2))
    tr.record_list_f32("x_decomp.lda_rows", dd)
    # ---- 5316 ALS row solve
    var m = 23
    var f = 4
    var Cf = positive_fixture(n, m, 10)
    for t in range(len(Cf)):
        if t % 7 == 2:
            Cf[t] = -Cf[t]
    var Y = tame(seam_fixture(m, f, 11))
    var YtY = oracle_gemm(Y, Y, f, m, f, True, False)
    var wa = oracle_als(Cf, Y, n, m, f, Float32(0.1))
    require_separates("5316 ALS item order", count_diff_f32(wa, oracle_als(Cf, Y, n, m, f, Float32(0.1), 1)))
    var xa = zeros(n * f)
    var fl = zeros(n)
    DevExec.als_rows(ptr(Cf), ptr(Y), ptr(YtY), ptr(xa), ptr(fl), n, m, f, Float32(0.1))
    same("5316 als_rows device", count_diff_f32(xa, wa))
    var xh = zeros(n * f)
    HostExec.als_rows(ptr(Cf), ptr(Y), ptr(YtY), ptr(xh), ptr(fl), n, m, f, Float32(0.1))
    same("5316 als_rows host", count_diff_f32(xh, wa))
    tr.record_list_f32("x_decomp.als_rows", xa)
    # ---- 5321 ALS conjugate-gradient rows, from the exact solve's factors
    # perturbed (so the residual is not ~0) and from zero for three steps
    var x0 = wa.copy()
    for t in range(len(x0)):
        x0[t] = x0[t] * Float32(0.75) + Float32(0.01) * Float32(t % 5)
    var wc = oracle_als_cg(Cf, Y, x0, n, m, f, Float32(0.1), 3)
    require_separates("5321 ALS CG p.Ap order", count_diff_f32(wc, oracle_als_cg(Cf, Y, x0, n, m, f, Float32(0.1), 3, 1)))
    var cgd = x0.copy()
    var stp = zeros(n)
    DevExec.als_cg_rows(ptr(Cf), ptr(Y), ptr(YtY), ptr(cgd), ptr(stp), n, m, f, Float32(0.1), 3)
    same("5321 als_cg_rows device", count_diff_f32(cgd, wc))
    var cgh = x0.copy()
    HostExec.als_cg_rows(ptr(Cf), ptr(Y), ptr(YtY), ptr(cgh), ptr(stp), n, m, f, Float32(0.1), 3)
    same("5321 als_cg_rows host", count_diff_f32(cgh, wc))
    tr.record_list_f32("x_decomp.als_cg_rows", cgd)
    print("PASS x_decomp rows_check")
