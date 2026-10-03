# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""FAST on Apple (lane/apple-fast-ldaqda): the discriminant-analysis stages
that run as one thread (or one thread a class) in naive_bayes/da.mojo, cut
into launches of one thread a CELL. Every cell is the unit's own chain (the
same operands, the same `add` / `mul` order), so the words are the units'.

PAR_STAGES (FAST + Apple default; off: -D MOJOLEARN_LDAQDA_PAR_STAGES_OFF):
  `lda_stage2` (op 38, t = 0): SCAL1 (d x d), MS (K x d) and G2 = MS'MS
    (d x d) on one thread; here the rank on one thread (d steps), then a
    thread a cell of each.
  `lda_stage3` (op 39, t = 0): SCAL = SCAL1 V2 (d^3 = 10.6 million steps at
    Istella's d = 220 on one thread), c_k, the intercepts and COEF; here the
    ranks and EVR on one thread (d steps), then a thread a cell of SCAL, TMP
    and COEF, a thread a class for the two d-step folds of the intercept.
  `qda_prep` (op 41, a thread a class): the d x d rotation R on the class's
    thread; here the scalings and log constant a class, R a thread a cell.
DEC_TILE (FAST + Apple default; off: -D MOJOLEARN_LDAQDA_DEC_TILE_OFF):
  `qda_dec` (op 42, t = i K + k): sum_r (sum_c (x_c - m_c) R[c, r])^2 with
    the row, the mean and R re-read from device memory d^2 times a unit;
    here a threadgroup takes 32 rows of one class, stages the centred rows
    and R in 32 x 32 shared tiles, every (row, r) chain folds c ascending in
    a register, then a thread a row folds r ascending: the unit's chains.
"""
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from x_prep.common import FP, IP, p, ld, st
from x_prep.prims import add, sub, mul, div, logf, sqrtf

comptime DA_TPB = 128
comptime DT = 32
"""qda_dec tile edge (rows, c and r)."""
comptime DT_TPB = 256


@always_inline
def _tid() -> Int:
    return Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)


# ---------------------------------------------------------------- lda_stage2
def lda2_rank_kernel(f: FP, q: IP):
    """META[1] = rank1 (the unit's first loop)."""
    if _tid() != 0:
        return
    var d = p(q, 7)
    var META = p(q, 9)
    var tol = ld(f, META)
    var rank = 0
    for r in range(d):
        var e = ld(f, p(q, 0) + r)
        if e < Float32(0):
            e = Float32(0)
        if sqrtf(e) > tol:
            rank += 1
    st(f, META + 1, Float32(rank))


def lda2_scal1_kernel(f: FP, q: IP):
    """t = c d + r: SCAL1[c, r]."""
    var t = _tid()
    var d = p(q, 7)
    if t >= d * d:
        return
    var c = t // d
    var r = t - c * d
    var rank = Int(ld(f, p(q, 9) + 1))
    var v = Float32(0)
    if r < rank:
        var e = ld(f, p(q, 0) + r)
        v = div(div(ld(f, p(q, 1) + c * d + r), ld(f, p(q, 2) + c)), sqrtf(e))
    st(f, p(q, 10) + c * d + r, v)


def lda2_ms_kernel(f: FP, q: IP):
    """t = k d + r: MS[k, r]."""
    var t = _tid()
    var K = p(q, 6)
    var d = p(q, 7)
    if t >= K * d:
        return
    var k = t // d
    var r = t - k * d
    var n = p(q, 8)
    var fac = Float32(1) if K == 1 else div(Float32(1), Float32(K - 1))
    var wk = sqrtf(mul(mul(Float32(n), ld(f, p(q, 5) + k)), fac))
    var s = Float32(0)
    for c in range(d):
        var cen = mul(wk, sub(ld(f, p(q, 3) + k * d + c), ld(f, p(q, 4) + c)))
        s = add(s, mul(cen, ld(f, p(q, 10) + c * d + r)))
    st(f, p(q, 12) + k * d + r, s)


def lda2_g2_kernel(f: FP, q: IP):
    """t = a d + b: G2[a, b] = sum_k MS[k, a] MS[k, b]."""
    var t = _tid()
    var K = p(q, 6)
    var d = p(q, 7)
    if t >= d * d:
        return
    var a = t // d
    var b = t - a * d
    var s = Float32(0)
    for k in range(K):
        s = add(s, mul(ld(f, p(q, 12) + k * d + a), ld(f, p(q, 12) + k * d + b)))
    st(f, p(q, 11) + a * d + b, s)


# ---------------------------------------------------------------- lda_stage3
def lda3_rank_kernel(f: FP, q: IP):
    """META[2] = rank2 and EVR (the unit's first two loops)."""
    if _tid() != 0:
        return
    var d = p(q, 7)
    var META = p(q, 8)
    var tol = ld(f, META)
    var s0 = Float32(0)
    var e0 = ld(f, p(q, 0))
    if e0 > Float32(0):
        s0 = sqrtf(e0)
    var rank2 = 0
    var tot = Float32(0)
    for r in range(d):
        var e = ld(f, p(q, 0) + r)
        if e < Float32(0):
            e = Float32(0)
        if sqrtf(e) > mul(tol, s0):
            rank2 += 1
        tot = add(tot, e)
    st(f, META + 2, Float32(rank2))
    for r in range(d):
        var e = ld(f, p(q, 0) + r)
        if e < Float32(0):
            e = Float32(0)
        st(f, p(q, 12) + r, div(e, tot) if tot > Float32(0) else Float32(0))


def lda3_scal_kernel(f: FP, q: IP):
    """t = c d + r: SCAL[c, r] = sum_r1 SCAL1[c, r1] V2[r1, r] (r < rank2)."""
    var t = _tid()
    var d = p(q, 7)
    if t >= d * d:
        return
    var c = t // d
    var r = t - c * d
    var rank2 = Int(ld(f, p(q, 8) + 2))
    var s = Float32(0)
    if r < rank2:
        for r1 in range(d):
            s = add(s, mul(ld(f, p(q, 2) + c * d + r1), ld(f, p(q, 1) + r1 * d + r)))
    st(f, p(q, 9) + c * d + r, s)


def lda3_tmp_kernel(f: FP, q: IP):
    """t = k d + r: TMP[k, r] = sum_c (MEAN[k, c] - XBAR[c]) SCAL[c, r]."""
    var t = _tid()
    var K = p(q, 6)
    var d = p(q, 7)
    if t >= K * d:
        return
    var k = t // d
    var r = t - k * d
    var s = Float32(0)
    for c in range(d):
        s = add(s, mul(sub(ld(f, p(q, 3) + k * d + c), ld(f, p(q, 4) + c)), ld(f, p(q, 9) + c * d + r)))
    st(f, p(q, 13) + k * d + r, s)


def lda3_inter_kernel(f: FP, q: IP):
    """t = k: INTER[k] = -0.5 |TMP_k|^2 + log prior_k (r ascending)."""
    var k = _tid()
    var K = p(q, 6)
    if k >= K:
        return
    var d = p(q, 7)
    var ss = Float32(0)
    for r in range(d):
        var s = ld(f, p(q, 13) + k * d + r)
        ss = add(ss, mul(s, s))
    st(f, p(q, 11) + k, add(mul(Float32(-0.5), ss), logf(ld(f, p(q, 5) + k))))


def lda3_coef_kernel(f: FP, q: IP):
    """t = k d + c: COEF[k, c] = sum_r TMP[k, r] SCAL[c, r]."""
    var t = _tid()
    var K = p(q, 6)
    var d = p(q, 7)
    if t >= K * d:
        return
    var k = t // d
    var c = t - k * d
    var s = Float32(0)
    for r in range(d):
        s = add(s, mul(ld(f, p(q, 13) + k * d + r), ld(f, p(q, 9) + c * d + r)))
    st(f, p(q, 10) + k * d + c, s)


def lda3_dot_kernel(f: FP, q: IP):
    """t = k: INTER[k] -= sum_c XBAR[c] COEF[k, c] (c ascending)."""
    var k = _tid()
    var K = p(q, 6)
    if k >= K:
        return
    var d = p(q, 7)
    var dot = Float32(0)
    for c in range(d):
        dot = add(dot, mul(ld(f, p(q, 4) + c), ld(f, p(q, 10) + k * d + c)))
    st(f, p(q, 11) + k, sub(ld(f, p(q, 11) + k), dot))


# ---------------------------------------------------------------- qda_prep
def qda_prep_scal_kernel(f: FP, q: IP):
    """t = k: S2 (floored) and LOGC[k], `qda_prep_unit` without R."""
    var k = _tid()
    var K = p(q, 2)
    if k >= K:
        return
    var d = p(q, 3)
    var reg = ld(f, p(q, 4))
    var E = p(q, 0) + k * d
    var smax = Float32(0)
    for r in range(d):
        var e = ld(f, E + r)
        if e < Float32(0):
            e = Float32(0)
        var s2 = add(mul(sub(Float32(1), reg), e), reg)
        st(f, p(q, 9) + k * d + r, s2)
        if s2 > smax:
            smax = s2
    var floor = mul(smax, Float32(1.1920929e-07))
    if smax <= Float32(0):
        floor = Float32(1)
    var sl = Float32(0)
    for r in range(d):
        var s2 = ld(f, p(q, 9) + k * d + r)
        if s2 < floor:
            s2 = floor
            st(f, p(q, 9) + k * d + r, s2)
        sl = add(sl, logf(s2))
    var prior = div(ld(f, p(q, 5) + k), Float32(p(q, 6)))
    if p(q, 10) != 0:
        prior = ld(f, p(q, 11) + k)
    st(f, p(q, 8) + k, sub(logf(prior), mul(Float32(0.5), sl)))


def qda_prep_rot_kernel(f: FP, q: IP):
    """t = (k d + c) d + r: R[k][c, r] = V[c, r] / sqrt(S2[k, r])."""
    var t = _tid()
    var K = p(q, 2)
    var d = p(q, 3)
    if t >= K * d * d:
        return
    var r = t % d
    var c = (t // d) % d
    var k = t // (d * d)
    var inv = div(Float32(1), sqrtf(ld(f, p(q, 9) + k * d + r)))
    st(f, p(q, 7) + k * d * d + c * d + r, mul(ld(f, p(q, 1) + k * d * d + c * d + r), inv))


# ---------------------------------------------------------------- qda_dec
def qda_dec_tile_kernel(f: FP, q: IP):
    """q = [X, n, d, MEAN, R, LOGC, K, OUT]; block g = rb K + k: rows rb DT ..
    of class k. Thread (lr, rg) = (tid / 8, tid % 8) keeps the chains of
    row lr at r = r0 + rg + 8 j, j < 4, folded c ascending; a thread a row
    then folds the r tile ascending into its norm."""
    var n = p(q, 1)
    var d = p(q, 2)
    var K = p(q, 6)
    var g = Int(block_idx.x)
    var k = g % K
    var i0 = (g // K) * DT
    var tid = Int(thread_idx.x)
    var lr = tid // 8
    var rg = tid - lr * 8
    var X = p(q, 0)
    var M = p(q, 3) + k * d
    var R = p(q, 4) + k * d * d
    var xs = stack_allocation[DT * DT, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var rs = stack_allocation[DT * DT, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var ss = stack_allocation[DT * DT, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var nrm = Float32(0)
    for r0 in range(0, d, DT):
        var a0 = Float32(0)
        var a1 = Float32(0)
        var a2 = Float32(0)
        var a3 = Float32(0)
        for c0 in range(0, d, DT):
            barrier()
            for u in range(DT * DT // DT_TPB):
                var e = tid + u * DT_TPB
                var hi = e // DT
                var lo = e - hi * DT
                var row = i0 + hi
                var xv = Float32(0)
                if row < n and c0 + lo < d:
                    xv = sub(ld(f, X + row * d + c0 + lo), ld(f, M + c0 + lo))
                xs[e] = xv
                var rv = Float32(0)
                if c0 + hi < d and r0 + lo < d:
                    rv = ld(f, R + (c0 + hi) * d + r0 + lo)
                rs[e] = rv
            barrier()
            var cl = min(DT, d - c0)
            for cc in range(cl):
                var xv = xs[lr * DT + cc]
                a0 = add(a0, mul(xv, rs[cc * DT + rg]))
                a1 = add(a1, mul(xv, rs[cc * DT + rg + 8]))
                a2 = add(a2, mul(xv, rs[cc * DT + rg + 16]))
                a3 = add(a3, mul(xv, rs[cc * DT + rg + 24]))
        ss[lr * DT + rg] = a0
        ss[lr * DT + rg + 8] = a1
        ss[lr * DT + rg + 16] = a2
        ss[lr * DT + rg + 24] = a3
        barrier()
        if tid < DT:
            var rl = min(DT, d - r0)
            for rr in range(rl):
                var s = ss[tid * DT + rr]
                nrm = add(nrm, mul(s, s))
        barrier()
    var row = i0 + tid
    if tid < DT and row < n:
        st(f, p(q, 7) + row * K + k, sub(ld(f, p(q, 5) + k), mul(Float32(0.5), nrm)))
