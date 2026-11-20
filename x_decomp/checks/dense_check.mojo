# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Seams DEVIATION 5307-5309 and 5320 of the decomp lane: the LU pivot and
its tie (5307), the right-looking substitution folds of lu_solve and the Cholesky's (5308),
the two-pass orthonormalisation (5309: `orth_diag` in every mode) and the one
blocked TSQR pass that IDENTICAL `orth` has been since 2026-10-10
(RSVD_IDN_TSQR_ORTHO), and the Householder QR that keeps its reflectors with
its explicit Q (geqrf + orgqr, 5320).

    tools/with_identical_mode.sh pixi run mojo run -I . x_decomp/checks/dense_check.mojo
"""
from core.identity_trace import IdentityTrace
from x_decomp.checks.xd_oracles import (
    oracle_chol, oracle_geqrf, oracle_lu, oracle_lu_solve, oracle_lu_solve_t, oracle_orgqr, oracle_orth_diag,
    oracle_orth_tsqr,
)
from x_decomp.checks.seam_util import (
    count_diff_f32,
    count_diff_i32,
    iptr,
    lcg_unit,
    ptr,
    require_separates,
    same,
    seam_fixture,
    zeros,
)
from decomposition.host.pca_full_oracle import host_qr_factor
from x_decomp.qr_sliced import QS_ROWS
from x_decomp.device import DevExec
from x_decomp.host_qr import fast_qr_factor
from x_decomp.host import HostExec


def tie_matrix(n: Int) -> List[Float32]:
    """Small integers with EXACT pivot ties in every column (|a| = 2 twice),
    so the pivot row is the tie-break's choice, and a well-scaled rest."""
    var a = List[Float32](capacity=n * n)
    var st = UInt64(77)
    for i in range(n):
        for j in range(n):
            var u = lcg_unit(st)
            a.append(Float32(Int(u * Float32(5))) - Float32(2))
    for j in range(n):
        a[((j + 1) % n) * n + j] = Float32(2)
        a[((j + 3) % n) * n + j] = Float32(-2)
    return a^


def spd(n: Int) -> List[Float32]:
    var x = seam_fixture(n + 5, n, 11)
    for t in range(len(x)):
        if abs(x[t]) > Float32(100):
            x[t] = x[t] * Float32(1e-6)
    var g = List[Float32](length=n * n, fill=Float32(0))
    for i in range(n):
        for j in range(n):
            var acc = Float32(0)
            for r in range(n + 5):
                acc = acc + x[r * n + i] * x[r * n + j]
            g[i * n + j] = acc
        g[i * n + i] = g[i * n + i] + Float32(n)
    for i in range(n):
        for j in range(i + 1, n):
            g[j * n + i] = g[i * n + j]
    return g^


def run_checks() raises:
    var tr = IdentityTrace()
    tr.header("x_decomp dense_check (DEVIATIONS 5307-5309, 5320)")
    # ---- the host's fast Householder QR (x_decomp/host_qr.mojo) == the
    # replay of the device kernel it serves for (host_qr_factor), bit for bit:
    # one slice, many slices, column counts around the SIMD width
    var shapes = [37, 9, 5000, 13, 3000, 19, 700, 40, 64, 64]
    for sh in range(len(shapes) // 2):
        var qm = shapes[2 * sh]
        var qn = shapes[2 * sh + 1]
        var qa = seam_fixture(qm, qn, UInt64(60 + sh))
        var qa2 = qa.copy()
        var want_r = host_qr_factor(qa2, qm, qn)
        var got_r = fast_qr_factor(ptr(qa), qm, qn)
        same("host QR slices " + String(qm) + " x " + String(qn), count_diff_f32(got_r, want_r))
    # (2026-11-19, lane fx-gpu-checks-xfam) The two cases that held
    # x_decomp/host_jacobi.mojo (fast_one_sided_jacobi_svd, fast_jacobi_eigh)
    # to the serial oracles are gone with that module: no product path had
    # called it since the host column took the round-robin order
    # (x_decomp/rr_svd.mojo, rr_eigh; e07c50c348 and c2f2a2ae86, 2026-10-02),
    # and its SVD case compared a replay that skips insignificant rotations
    # with host_one_sided_jacobi_svd at its default skip_insignificant=False
    # (a switch since 1d5708ba86), so it could not pass. The host column's
    # SVD is held to the device below ("bounded device svd").
    # ---- the host LU's SIMD row eliminations (x_decomp/host_simd.mojo) at a
    # size past the vector width, against the oracle
    var ln = 100
    var la = seam_fixture(ln, ln, UInt64(90))
    for i in range(ln):
        la[i * ln + i] = la[i * ln + i] + Float32(3)
    var lw = oracle_lu(la, ln)
    var lh = la.copy()
    var lp = List[Int32](length=ln, fill=Int32(0))
    var linfo = zeros(1)
    HostExec.lu(ptr(lh), iptr(lp), ptr(linfo), ln)
    same("host LU factor n 100", count_diff_f32(lh, lw[0]))
    same("host LU pivots n 100", count_diff_i32(lp, lw[1]))
    # ---- 5307 the pivot
    var n = 9
    var a = tie_matrix(n)
    var got_o = oracle_lu(a, n)
    var alt_o = oracle_lu(a, n, 1)
    require_separates("5307 pivot tie (lowest row vs last)", count_diff_i32(got_o[1], alt_o[1]))
    var lu_d = a.copy()
    var piv_d = List[Int32](length=n, fill=Int32(0))
    var info = zeros(1)
    DevExec.lu(ptr(lu_d), iptr(piv_d), ptr(info), n)
    same("5307 lu pivots device", count_diff_i32(piv_d, got_o[1]))
    same("5307 lu factors device", count_diff_f32(lu_d, got_o[0]))
    var lu_h = a.copy()
    var piv_h = List[Int32](length=n, fill=Int32(0))
    HostExec.lu(ptr(lu_h), iptr(piv_h), ptr(info), n)
    same("5307 lu pivots host", count_diff_i32(piv_h, got_o[1]))
    same("5307 lu factors host", count_diff_f32(lu_h, got_o[0]))
    tr.record_list_f32("x_decomp.lu", lu_d)
    # ---- 5308 getrs and the Cholesky
    var nrhs = 5
    var bsrc = seam_fixture(n, nrhs, 13)
    for t in range(len(bsrc)):
        if abs(bsrc[t]) > Float32(100):
            bsrc[t] = bsrc[t] * Float32(1e-6)
    var want = oracle_lu_solve(got_o[0], got_o[1], bsrc, n, nrhs)
    require_separates("5308 lu_solve substitution order", count_diff_f32(want, oracle_lu_solve(got_o[0], got_o[1], bsrc, n, nrhs, 1)))
    var xd = bsrc.copy()
    DevExec.lu_solve(ptr(got_o[0]), iptr(got_o[1]), ptr(xd), n, nrhs)
    same("5308 lu_solve device", count_diff_f32(xd, want))
    var xh = bsrc.copy()
    HostExec.lu_solve(ptr(got_o[0]), iptr(got_o[1]), ptr(xh), n, nrhs)
    same("5308 lu_solve host", count_diff_f32(xh, want))
    tr.record_list_f32("x_decomp.lu_solve", xd)
    var want_t = oracle_lu_solve_t(got_o[0], got_o[1], bsrc, n, nrhs)
    require_separates("5308 lu_solve 'T' substitution order", count_diff_f32(want_t, oracle_lu_solve_t(got_o[0], got_o[1], bsrc, n, nrhs, 1)))
    var xtd = bsrc.copy()
    DevExec.lu_solve(ptr(got_o[0]), iptr(got_o[1]), ptr(xtd), n, nrhs, 1)
    same("5308 lu_solve trans device", count_diff_f32(xtd, want_t))
    var xth = bsrc.copy()
    HostExec.lu_solve(ptr(got_o[0]), iptr(got_o[1]), ptr(xth), n, nrhs, 1)
    same("5308 lu_solve trans host", count_diff_f32(xth, want_t))
    tr.record_list_f32("x_decomp.lu_solve_t", xtd)
    # past one diagonal block (device TRS_BLOCK 128): three blocks, the last
    # partial, so the feed kernel and the descending block order run; 'N'
    # and 'T', device == host == the oracle
    var bn = 300
    var bnr = 3
    var ba = seam_fixture(bn, bn, UInt64(91))
    for i in range(bn):
        ba[i * bn + i] = ba[i * bn + i] + Float32(6)
    var bw = oracle_lu(ba, bn)
    var bb = seam_fixture(bn, bnr, UInt64(92))
    for tt in range(2):
        var bwant = oracle_lu_solve(bw[0], bw[1], bb, bn, bnr) if tt == 0 else oracle_lu_solve_t(bw[0], bw[1], bb, bn, bnr)
        var bxd = bb.copy()
        DevExec.lu_solve(ptr(bw[0]), iptr(bw[1]), ptr(bxd), bn, bnr, tt)
        same("5308 lu_solve device n 300 trans " + String(tt), count_diff_f32(bxd, bwant))
        var bxh = bb.copy()
        HostExec.lu_solve(ptr(bw[0]), iptr(bw[1]), ptr(bxh), bn, bnr, tt)
        same("5308 lu_solve host n 300 trans " + String(tt), count_diff_f32(bxh, bwant))
    var ns = 11
    var g = spd(ns)
    var wl = oracle_chol(g, ns)
    require_separates("5308 Cholesky fold order", count_diff_f32(wl, oracle_chol(g, ns, 1)))
    var cd = g.copy()
    DevExec.chol(ptr(cd), ptr(info), ns)
    same("5308 chol device", count_diff_f32(cd, wl))
    var ch = g.copy()
    HostExec.chol(ptr(ch), ptr(info), ns)
    same("5308 chol host", count_diff_f32(ch, wl))
    tr.record_list_f32("x_decomp.chol", cd)
    # ---- 5309 MGS2
    var mm = 40
    var l = 6
    var q = seam_fixture(mm, l, 17)
    for t in range(mm):
        # nearly dependent columns: column 1 = column 0 + a small perturbation
        q[t * l + 1] = q[t * l + 0] + Float32(1e-3) * q[t * l + 2]
    for t in range(mm):
        # DEVIATION 5318: column 4 = 2 * column 3 + a perturbation 2^-20 of
        # column 5, numerically dependent at the 2^-16 guard
        q[t * l + 4] = Float32(2) * q[t * l + 3] + Float32(9.5367431640625e-07) * q[t * l + 5]
    # The two-pass route (qr_factor's R, the rank guard, trsm_row; twice):
    # `orth_diag` in every mode (linalg.svd's U), and `orth` itself in FAST
    # and for the shapes the TSQR does not take.
    var want2 = oracle_orth_diag(q, mm, l)
    require_separates("5309 orth passes (two vs one)", count_diff_f32(want2[0], oracle_orth_diag(q, mm, l, 1)[0]))
    require_separates("5318 orth rank guard", count_diff_f32(want2[0], oracle_orth_diag(q, mm, l, 2)[0]))
    var qd2 = q.copy()
    var dd2 = zeros(l)
    DevExec.orth_diag(ptr(qd2), mm, l, ptr(dd2))
    same("5309 orth_diag device", count_diff_f32(qd2, want2[0]))
    same("5309 orth_diag device R diagonals", count_diff_f32(dd2, want2[1]))
    var qh2 = q.copy()
    var dh2 = zeros(l)
    HostExec.orth_diag(ptr(qh2), mm, l, ptr(dh2))
    same("5309 orth_diag host", count_diff_f32(qh2, want2[0]))
    same("5309 orth_diag host R diagonals", count_diff_f32(dh2, want2[1]))
    # `orth`: under IDENTICAL one blocked TSQR pass with the rank-guarded
    # selection (x_decomp/tsqr_core.mojo RSVD_IDN_TSQR_ORTHO, the default
    # since 06865c6d84, 2026-10-10) on both columns. Until 2026-11-19 this
    # case held `orth` to the two-pass oracle above, which the promotion left
    # behind; it sat behind an earlier stale case and had never run. FAST
    # keeps the two passes and `same` makes no claim there.
    var wq = oracle_orth_tsqr(q, mm, l)
    require_separates("5309 orth TSQR chains (16 vs one)", count_diff_f32(wq, oracle_orth_tsqr(q, mm, l, 1)))
    require_separates("5318 orth TSQR rank guard", count_diff_f32(wq, oracle_orth_tsqr(q, mm, l, 2)))
    require_separates("5309 orth route (TSQR vs two passes)", count_diff_f32(wq, want2[0]))
    var qd = q.copy()
    DevExec.orth(ptr(qd), mm, l)
    var qh = q.copy()
    HostExec.orth(ptr(qh), mm, l)
    # the two columns against each other first, so a failure of the lines
    # after it is the oracle's restatement and not a column
    same("5309 orth device vs host replay", count_diff_f32(qd, qh))
    same("5309 orth device", count_diff_f32(qd, wq))
    same("5309 orth host", count_diff_f32(qh, wq))
    tr.record_list_f32("x_decomp.orth", qd)
    # The TSQR orth beyond what the oracle restates (three leaf blocks, the
    # combine tree, three panels in two pairs): the device against its host
    # replay, with a dependent and a zero column.
    var tm = 3 * 2048 + 37
    var tl = 40
    var tq = List[Float32](capacity=tm * tl)
    var tst = UInt64(5309) * UInt64(2654435761) + UInt64(1)
    for _ in range(tm * tl):
        tq.append(lcg_unit(tst) * Float32(4) - Float32(2))
    for t in range(tm):
        tq[t * tl + 7] = Float32(2) * tq[t * tl + 3]
        tq[t * tl + 21] = Float32(0)
    var tqh = tq.copy()
    HostExec.orth(ptr(tqh), tm, tl)
    var tqd = tq.copy()
    DevExec.orth(ptr(tqd), tm, tl)
    same("5309 orth device vs host replay " + String(tm) + " x " + String(tl), count_diff_f32(tqd, tqh))
    var tzero = 0
    for t in range(tm):
        if tqh[t * tl + 7] != Float32(0) or tqh[t * tl + 21] != Float32(0):
            tzero += 1
    if tzero != 0:
        raise Error("SEAM 5318 orth TSQR: a dependent column is not exactly zero in " + String(tzero) + " rows")
    print("  5318 orth TSQR dependent columns: exactly zero")
    # ---- 5320 geqrf + orgqr, a tall, a wide and a rank-deficient matrix,
    # and (lane hr-qr) a tall one of several slices (x_decomp/qr_sliced.mojo)
    for shape in range(4):
        var gm = 23 if shape == 0 or shape == 2 else (5 if shape == 1 else 3 * QS_ROWS + 37)
        var gn = 6 if shape != 1 else 9
        if shape == 3:
            gn = 5
        var ga = seam_fixture(gm, gn, UInt64(31 + shape))
        for t in range(len(ga)):
            if abs(ga[t]) > Float32(100):
                ga[t] = ga[t] * Float32(1e-6)
        if shape == 2:
            for t in range(gm):
                ga[t * gn + 3] = ga[t * gn + 1]          # a duplicated column
        var kk = gm if gm < gn else gn
        var gq = gm if shape != 3 else gn          # Q's columns (the slices' case: the thin Q)
        var want_f = oracle_geqrf(ga, gm, gn)
        require_separates("5320 geqrf fold order", count_diff_f32(want_f[0], oracle_geqrf(ga, gm, gn, 1)[0]))
        var hd = ga.copy()
        var td = zeros(kk)
        DevExec.geqrf(ptr(hd), ptr(td), gm, gn)
        same("5320 geqrf device h", count_diff_f32(hd, want_f[0]))
        same("5320 geqrf device tau", count_diff_f32(td, want_f[1]))
        var hh = ga.copy()
        var th = zeros(kk)
        HostExec.geqrf(ptr(hh), ptr(th), gm, gn)
        same("5320 geqrf host h", count_diff_f32(hh, want_f[0]))
        same("5320 geqrf host tau", count_diff_f32(th, want_f[1]))
        var want_q = oracle_orgqr(want_f[0], want_f[1], gm, gn, kk, gq)
        require_separates("5320 orgqr fold order", count_diff_f32(want_q, oracle_orgqr(want_f[0], want_f[1], gm, gn, kk, gq, 1)))
        var qgd = zeros(gm * gq)
        DevExec.orgqr(ptr(want_f[0]), ptr(want_f[1]), ptr(qgd), gm, gn, kk, gq)
        same("5320 orgqr device", count_diff_f32(qgd, want_q))
        var qgh = zeros(gm * gq)
        HostExec.orgqr(ptr(want_f[0]), ptr(want_f[1]), ptr(qgh), gm, gn, kk, gq)
        same("5320 orgqr host", count_diff_f32(qgh, want_q))
        tr.record_list_f32("x_decomp.geqrf", hd)
        tr.record_list_f32("x_decomp.orgqr", qgd)
    # ---- the device SVD BOUNDED IN WORK PER LAUNCH (lane/lle-timeout): the
    # QR in column ranges (x_decomp/qr_bounded.mojo), the round-robin Jacobi
    # one launch a round (x_decomp/rr_svd.mojo): == the host replay at the
    # default cut, and the same bits with the QR cut small; one slice
    # square, one slice tall, two TSQR slices
    var bshapes = [150, 150, 300, 150, 1200, 96]
    for sh in range(len(bshapes) // 2):
        var bm = bshapes[2 * sh]
        var bn = bshapes[2 * sh + 1]
        # a well-scaled fixture (seam_fixture's 1e-3 .. 1e8 columns exhaust
        # the sweep budget and HostExec.svd refuses them): uniform in (-2, 2]
        # from the seam LCG, with seam_fixture's tie and negated rows planted
        var ba = List[Float32](capacity=bm * bn)
        var bst = UInt64(95 + sh) * UInt64(2654435761) + UInt64(1)
        for _ in range(bm * bn):
            ba.append(lcg_unit(bst) * Float32(4) - Float32(2))
        for f in range(bn):
            ba[2 * bn + f] = ba[1 * bn + f] * Float32(0.5)
            ba[4 * bn + f] = -ba[3 * bn + f]
        var bh = ba.copy()
        var s_h = zeros(bn)
        var v_h = zeros(bn * bn)
        HostExec.svd(ptr(bh), bm, bn, ptr(s_h), ptr(v_h))
        # the input list's last named use was the pointer above: held to here
        _ = bh^
        var bd = ba.copy()
        var s_d = zeros(bn)
        var v_d = zeros(bn * bn)
        DevExec.svd(ptr(bd), bm, bn, ptr(s_d), ptr(v_d))
        _ = bd^
        var tag = String(bm) + " x " + String(bn)
        same("bounded device svd values " + tag, count_diff_f32(s_d, s_h))
        same("bounded device svd vectors " + tag, count_diff_f32(v_d, v_h))
        var bc = ba.copy()
        var s_c = zeros(bn)
        var v_c = zeros(bn * bn)
        DevExec.svd_cells(ptr(bc), bm, bn, ptr(s_c), ptr(v_c), 4096)
        _ = bc^
        same("bounded device svd values cut small " + tag, count_diff_f32(s_c, s_h))
        same("bounded device svd vectors cut small " + tag, count_diff_f32(v_c, v_h))
        tr.record_list_f32("x_decomp.svd_bounded", v_d)
    print("PASS x_decomp dense_check")
