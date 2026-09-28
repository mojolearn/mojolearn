# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Seams DEVIATION 5300-5306 and 5319 of the decomp lane: the folds (gemm 5300,
column/row sums 5301, squared distance 5302, the other distances 5319), the elementwise cell's zero
guards and clamps (5303, IDENTITY_PATHS Clause B), its flush of subnormal
operands (5304), its series transcendentals (digamma, lgamma: 5305) and the
counter-based draws (uniform, normal, Rademacher, Gamma: 5306).

    tools/with_identical_mode.sh pixi run mojo run -I . x_decomp/checks/fold_ew_check.mojo

Each seam's fixture is first shown to SEPARATE the pinned spelling from the
unpinned one (VACUOUS otherwise); then the device column (DevExec) and the
CPU column (HostExec) must each equal the host oracle bit for bit (FAST:
reported, no claim). With MOJOLEARN_IDENTITY_TRACE set, every seam's output
lands on the card."""
from std.memory import bitcast

from core.identity_trace import IdentityTrace
from x_decomp.cells import F32Ptr, FOLD_BLOCK
from x_decomp.checks.xd_oracles import (
    oracle_absmax_sign,
    oracle_colsum,
    oracle_ew,
    oracle_gamma,
    oracle_gemm,
    oracle_pdist,
    oracle_rand,
    oracle_rowsum,
    oracle_sqdist,
)
from x_decomp.checks.seam_util import count_diff_f32, ptr, require_separates, same, seam_fixture, zeros
from x_decomp.device import DevExec
from x_decomp.host import HostExec


def ew_inputs() -> List[Float32]:
    """x values for the elementwise cell: zeros of both signs, subnormals,
    the exp clamp's edges, the digamma/lgamma recurrence range, negatives."""
    var v = List[Float32]()
    v.append(Float32(0))
    v.append(Float32(-0.0))
    v.append(bitcast[DType.float32](UInt32(0x00000005)))
    v.append(bitcast[DType.float32](UInt32(0x80000007)))
    v.append(bitcast[DType.float32](UInt32(0x00400000)))
    v.append(Float32(100))
    v.append(Float32(-120))
    v.append(Float32(0.25))
    v.append(Float32(1.5))
    v.append(Float32(5.75))
    v.append(Float32(7.5))
    v.append(Float32(-3.5))
    v.append(Float32(1e-30))
    v.append(Float32(3e8))
    v.append(Float32(1))
    v.append(Float32(2))
    return v^


def main() raises:
    var tr = IdentityTrace()
    tr.header("x_decomp fold_ew_check (DEVIATIONS 5300-5306)")
    # ---- 5300 gemm, all four transposition arms
    var m = 9
    var k = 23
    var n = 7
    var a = seam_fixture(m, k, 1)
    var b = seam_fixture(k, n, 2)
    var at = seam_fixture(k, m, 3)
    var bt = seam_fixture(n, k, 4)
    for arm in range(4):
        var ta = arm == 1 or arm == 3
        var tb = arm >= 2
        var A = at.copy() if ta else a.copy()
        var B = bt.copy() if tb else b.copy()
        var want = oracle_gemm(A, B, m, k, n, ta, tb)
        if arm == 0:
            require_separates("5300 gemm fold order", count_diff_f32(want, oracle_gemm(A, B, m, k, n, ta, tb, 1)))
            require_separates("5300 gemm contraction", count_diff_f32(want, oracle_gemm(A, B, m, k, n, ta, tb, 2)))
        var dev = zeros(m * n)
        DevExec.gemm(ptr(A), ptr(B), ptr(dev), m, k, n, ta, tb)
        same("5300 gemm device arm " + String(arm), count_diff_f32(dev, want))
        var hst = zeros(m * n)
        HostExec.gemm(ptr(A), ptr(B), ptr(hst), m, k, n, ta, tb)
        same("5300 gemm host arm " + String(arm), count_diff_f32(hst, want))
        tr.record_list_f32("x_decomp.gemm." + String(arm), dev)
    # ---- 5300/5301 past FOLD_BLOCK: the blocked two-stage fold
    var kb = 9000
    var ab = seam_fixture(3, kb, 21)
    var bb2 = seam_fixture(kb, 2, 22)
    var wbk = oracle_gemm(ab, bb2, 3, kb, 2, False, False)
    require_separates("5300 blocked gemm vs one sequential fold", count_diff_f32(wbk, oracle_gemm(ab, bb2, 3, kb, 2, False, False, 3)))
    var dbk = zeros(6)
    DevExec.gemm(ptr(ab), ptr(bb2), ptr(dbk), 3, kb, 2, False, False)
    same("5300 blocked gemm device", count_diff_f32(dbk, wbk))
    var hbk = zeros(6)
    HostExec.gemm(ptr(ab), ptr(bb2), ptr(hbk), 3, kb, 2, False, False)
    same("5300 blocked gemm host", count_diff_f32(hbk, wbk))
    var cb = seam_fixture(kb, 3, 23)
    var wcb = oracle_colsum(cb, kb, 3)
    require_separates("5301 blocked colsum vs one sequential fold", count_diff_f32(wcb, oracle_colsum(cb, kb, 3, 3)))
    var dcb = zeros(3)
    DevExec.colsum(ptr(cb), ptr(dcb), kb, 3)
    same("5301 blocked colsum device", count_diff_f32(dcb, wcb))
    var hcb = zeros(3)
    HostExec.colsum(ptr(cb), ptr(hcb), kb, 3)
    same("5301 blocked colsum host", count_diff_f32(hcb, wcb))
    var rb = seam_fixture(2, kb, 24)
    var wrb = oracle_rowsum(rb, 2, kb)
    require_separates("5301 blocked rowsum vs one sequential fold", count_diff_f32(wrb, oracle_rowsum(rb, 2, kb, 3)))
    var drb = zeros(2)
    DevExec.rowsum(ptr(rb), ptr(drb), 2, kb)
    same("5301 blocked rowsum device", count_diff_f32(drb, wrb))
    var hrb = zeros(2)
    HostExec.rowsum(ptr(rb), ptr(hrb), 2, kb)
    same("5301 blocked rowsum host", count_diff_f32(hrb, wrb))
    tr.record_list_f32("x_decomp.gemm.blocked", dbk)
    tr.record_list_f32("x_decomp.colsum.blocked", dcb)
    # ---- 5301 column and row sums
    var rows = 41
    var cols = 12
    var x = seam_fixture(rows, cols, 5)
    var wc = oracle_colsum(x, rows, cols)
    var wr = oracle_rowsum(x, rows, cols)
    require_separates("5301 column-sum order", count_diff_f32(wc, oracle_colsum(x, rows, cols, 1)))
    var xr = seam_fixture(cols, rows, 6)
    var wr2 = oracle_rowsum(xr, cols, rows)
    require_separates("5301 row-sum order", count_diff_f32(wr2, oracle_rowsum(xr, cols, rows, 1)))
    var dc = zeros(cols)
    DevExec.colsum(ptr(x), ptr(dc), rows, cols)
    same("5301 colsum device", count_diff_f32(dc, wc))
    var hc = zeros(cols)
    HostExec.colsum(ptr(x), ptr(hc), rows, cols)
    same("5301 colsum host", count_diff_f32(hc, wc))
    var drs = zeros(cols)
    DevExec.rowsum(ptr(xr), ptr(drs), cols, rows)
    same("5301 rowsum device", count_diff_f32(drs, wr2))
    var hrs = zeros(cols)
    HostExec.rowsum(ptr(xr), ptr(hrs), cols, rows)
    same("5301 rowsum host", count_diff_f32(hrs, wr2))
    tr.record_list_f32("x_decomp.colsum", dc)
    # ---- 5304 the flush of an add's RESULT: two normal operands whose sum is subnormal
    var fz = List[Float32]()
    fz.append(Float32(1.5e-38))
    fz.append(Float32(2))
    fz.append(Float32(-1.4e-38))
    fz.append(Float32(1))
    fz.append(Float32(0))
    fz.append(Float32(-0.5))
    var wf = oracle_colsum(fz, 3, 2)
    require_separates("5304 subnormal partial sum flushed", count_diff_f32(wf, oracle_colsum(fz, 3, 2, 2)))
    var fdv = zeros(2)
    DevExec.colsum(ptr(fz), ptr(fdv), 3, 2)
    same("5304 flushed colsum device", count_diff_f32(fdv, wf))
    var fhs = zeros(2)
    HostExec.colsum(ptr(fz), ptr(fhs), 3, 2)
    same("5304 flushed colsum host", count_diff_f32(fhs, wf))
    tr.record_list_f32("x_decomp.rowsum", drs)
    # ---- 5302 squared distance
    var d = 13
    var qa = seam_fixture(17, d, 7)
    var qb = seam_fixture(11, d, 8)
    var ws = oracle_sqdist(qa, 17, qb, 11, d)
    require_separates("5302 sqdist fold order", count_diff_f32(ws, oracle_sqdist(qa, 17, qb, 11, d, 1)))
    require_separates("5302 sqdist contraction", count_diff_f32(ws, oracle_sqdist(qa, 17, qb, 11, d, 2)))
    var ds = zeros(17 * 11)
    DevExec.sqdist(ptr(qa), ptr(qb), ptr(ds), 17, 11, d)
    same("5302 sqdist device", count_diff_f32(ds, ws))
    var hs = zeros(17 * 11)
    HostExec.sqdist(ptr(qa), ptr(qb), ptr(hs), 17, 11, d)
    same("5302 sqdist host", count_diff_f32(hs, ws))
    tr.record_list_f32("x_decomp.sqdist", ds)
    # ---- 5319 the non-Euclidean distances (manhattan, chebyshev, minkowski 3, cosine)
    # minkowski's root exp(log(sum) / p) compresses a one-ulp fold difference,
    # so its fixture is wider (97 features) and its p 1.5
    var d3 = 97
    var qa3 = seam_fixture(17, d3, 21)
    var qb3 = seam_fixture(11, d3, 22)
    for kind in range(1, 5):
        var pw = Float32(1.5) if kind == 3 else Float32(3)
        var dd = d3 if kind == 3 else d
        var pa = qa3.copy() if kind == 3 else qa.copy()
        var pb = qb3.copy() if kind == 3 else qb.copy()
        var wp = oracle_pdist(pa, 17, pb, 11, dd, kind, pw)
        if kind != 2:   # a maximum has no fold order to separate
            require_separates("5319 pdist fold order kind " + String(kind), count_diff_f32(wp, oracle_pdist(pa, 17, pb, 11, dd, kind, pw, 1)))
        var dp = zeros(17 * 11)
        DevExec.sqdist(ptr(pa), ptr(pb), ptr(dp), 17, 11, dd, kind, pw)
        same("5319 pdist device kind " + String(kind), count_diff_f32(dp, wp))
        var hp = zeros(17 * 11)
        HostExec.sqdist(ptr(pa), ptr(pb), ptr(hp), 17, 11, dd, kind, pw)
        same("5319 pdist host kind " + String(kind), count_diff_f32(hp, wp))
        tr.record_list_f32("x_decomp.pdist." + String(kind), dp)
    # ---- 5303/5304/5305 the elementwise cell, every op code
    var xs = ew_inputs()
    var ne = len(xs)
    var ys = List[Float32]()
    var zs = List[Float32]()
    for t in range(ne):
        ys.append(xs[(t + 3) % ne])
        zs.append(xs[(t + 7) % ne])
    var ops: List[Int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 31, 33, 34, 35, 36, 37]
    var sep_guard = 0
    var sep_ftz = 0
    var sep_series = 0
    for op in ops:
        var s = Float32(0.5)
        if op == 10:
            s = Float32(0)
        var want = List[Float32]()
        for t in range(ne):
            var w = oracle_ew(op, xs[t], ys[t], zs[t], s)
            want.append(w)
            var g = oracle_ew(op, xs[t], ys[t], zs[t], s, 1)
            if bitcast[DType.uint32](g) != bitcast[DType.uint32](w):
                if op == 3 or op == 7 or op == 9 or op == 10 or op == 16 or op == 36:
                    sep_guard += 1
            if op == 0:
                var f = oracle_ew(op, xs[t], ys[t], zs[t], s, 2)
                if bitcast[DType.uint32](f) != bitcast[DType.uint32](w):
                    sep_ftz += 1
            if op == 24 or op == 37:
                var q = oracle_ew(op, xs[t], ys[t], zs[t], s, 3)
                if bitcast[DType.uint32](q) != bitcast[DType.uint32](w):
                    sep_series += 1
        var dev = zeros(ne)
        DevExec.ew(op, ptr(xs), ptr(ys), ne, 0, ptr(zs), ne, 0, ptr(dev), ne, ne, s)
        same("5303-5305 ew op " + String(op) + " device", count_diff_f32(dev, want))
        var hst = zeros(ne)
        HostExec.ew(op, ptr(xs), ptr(ys), ne, 0, ptr(zs), ne, 0, ptr(hst), ne, ne, s)
        same("5303-5305 ew op " + String(op) + " host", count_diff_f32(hst, want))
        tr.record_list_f32("x_decomp.ew." + String(op), dev)
    require_separates("5303 zero guards and clamps (Clause B)", sep_guard)
    require_separates("5304 subnormal flush of an add", sep_ftz)
    require_separates("5305 digamma/lgamma recurrence", sep_series)
    # ---- 5306 the draws
    var cnt = 300
    for kind in range(3):
        var want = List[Float32]()
        var altv = List[Float32]()
        for i in range(cnt):
            want.append(oracle_rand(i, UInt32(12345), UInt32(7), kind))
            altv.append(oracle_rand(i, UInt32(12345), UInt32(7), kind, 1))
        if kind < 2:
            require_separates("5306 draw mapping kind " + String(kind), count_diff_f32(want, altv))
        var dev = zeros(cnt)
        DevExec.rand(ptr(dev), cnt, UInt32(12345), UInt32(7), kind)
        same("5306 rand kind " + String(kind) + " device", count_diff_f32(dev, want))
        var hst = zeros(cnt)
        HostExec.rand(ptr(hst), cnt, UInt32(12345), UInt32(7), kind)
        same("5306 rand kind " + String(kind) + " host", count_diff_f32(hst, want))
        tr.record_list_f32("x_decomp.rand." + String(kind), dev)
    var gw = List[Float32]()
    var ga = List[Float32]()
    for i in range(cnt):
        gw.append(oracle_gamma(i, UInt32(99), UInt32(60), Float32(1.5)))
        ga.append(oracle_gamma(i, UInt32(99), UInt32(60), Float32(1.5), 1))
    require_separates("5306 gamma acceptance", count_diff_f32(gw, ga))
    var gd = zeros(cnt)
    DevExec.rand_gamma(ptr(gd), cnt, UInt32(99), UInt32(60), Float32(1.5))
    same("5306 gamma device", count_diff_f32(gd, gw))
    var gh = zeros(cnt)
    HostExec.rand_gamma(ptr(gh), cnt, UInt32(99), UInt32(60), Float32(1.5))
    same("5306 gamma host", count_diff_f32(gh, gw))
    tr.record_list_f32("x_decomp.gamma", gd)
    # ---- 5317 the sign of a vector: its largest-|.| entry, ties to the lower index
    var tv = seam_fixture(9, 6, 31)
    for c in range(6):
        tv[c] = Float32(1e9) if c % 2 == 0 else Float32(-1e9)
        tv[6 + c] = Float32(-1e9) if c % 2 == 0 else Float32(1e9)
    for by_col in range(2):
        var bc = by_col == 1
        var ws2 = oracle_absmax_sign(tv, 9, 6, bc)
        if bc:
            require_separates("5317 sign-flip tie", count_diff_f32(ws2, oracle_absmax_sign(tv, 9, 6, bc, 1)))
        var cnt2 = 6 if bc else 9
        var dsg = zeros(cnt2)
        DevExec.absmax_sign(ptr(tv), ptr(dsg), 9, 6, bc)
        same("5317 absmax sign device", count_diff_f32(dsg, ws2))
        var hsg = zeros(cnt2)
        HostExec.absmax_sign(ptr(tv), ptr(hsg), 9, 6, bc)
        same("5317 absmax sign host", count_diff_f32(hsg, ws2))
        tr.record_list_f32("x_decomp.absmax_sign." + String(by_col), dsg)
    # ---- 5317 across FOLD_BLOCK slices (the device's two-stage scan): the
    # largest |.| tied across slices goes to the LOWER slice, a strictly
    # larger one in a later slice wins, NaN never wins; 3 vectors of
    # 2 * FOLD_BLOCK + 9 entries, as columns and as rows
    var ln = 2 * FOLD_BLOCK + 9
    var lc = zeros(ln * 3)
    for i in range(ln * 3):
        lc[i] = Float32((i * 37) % 101 - 50) * Float32(0.001)
    lc[3 * 3 + 0] = Float32(1e9)
    lc[(FOLD_BLOCK + 2) * 3 + 0] = Float32(-1e9)
    lc[(ln - 1) * 3 + 0] = Float32(-1e9)
    lc[10 * 3 + 1] = Float32(5)
    lc[(FOLD_BLOCK + 100) * 3 + 1] = Float32(-7)
    lc[(2 * FOLD_BLOCK + 1) * 3 + 1] = Float32(7)
    lc[7 * 3 + 2] = Float32(0) / Float32(0)
    lc[(2 * FOLD_BLOCK + 4) * 3 + 2] = Float32(-3)
    var lr = zeros(ln * 3)
    for i in range(ln):
        for j in range(3):
            lr[j * ln + i] = lc[i * 3 + j]
    for by_col in range(2):
        var bc = by_col == 1
        var src = lc.copy() if bc else lr.copy()
        var nn = ln if bc else 3
        var dd = 3 if bc else ln
        var want = oracle_absmax_sign(src, nn, dd, bc)
        require_separates("5317 sign-flip tie across slices", count_diff_f32(want, oracle_absmax_sign(src, nn, dd, bc, 1)))
        var dl = zeros(3)
        DevExec.absmax_sign(ptr(src), ptr(dl), nn, dd, bc)
        same("5317 absmax sign across slices device", count_diff_f32(dl, want))
        var hl = zeros(3)
        HostExec.absmax_sign(ptr(src), ptr(hl), nn, dd, bc)
        same("5317 absmax sign across slices host", count_diff_f32(hl, want))
        tr.record_list_f32("x_decomp.absmax_sign_long." + String(by_col), dl)
    print("PASS x_decomp fold_ew_check")
