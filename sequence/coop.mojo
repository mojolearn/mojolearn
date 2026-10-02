# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""A LONG ONE-THREAD FOLD WITH A SIMDGROUP'S LOADS (lane/sequence-apple2,
2026-09-28; Apple only). The IDENTICAL reductions of this lane are one
thread's ascending chain (a 4M-float norm in LAMB and Adafactor, VAR's
million-term GEMM cells), and one thread could keep only a few loads in
flight: the fold waited on memory. Here the 32 lanes of a simdgroup load
the next 32 R elements together (coalesced), and EVERY lane then runs the
same chain over them in order, element k broadcast from its lane
(`shuffle_idx`): the same fmas on the same values in the same order as the
one-thread fold, so the bits are those of `sumsq_fold` / `gemm_dot`. Lane 0
stores. Launched by `DeviceExec` (`coop_kernel`, one simdgroup per cell)
for the ops below on an Apple GPU only; every other column (and the host)
runs the one-thread op."""
from std.gpu.primitives.warp import shuffle_idx

from sequence.ops import (
    FP,
    Args,
    OP_AF_ALPHA,
    OP_AF_BLK_SUMSQ,
    OP_AF_DENOM,
    OP_GEMM,
    OP_LAMB_RATIO,
    OP_SEG_SUMSQ,
    fma3,
    ld,
    st,
)
from sequence.adafactor import af_alpha_tail, af_denom_tail, lamb_ratio_tail, op_af_alpha, op_af_denom, op_lamb_ratio, op_seg_sumsq

#: the Apple simdgroup
comptime COOP_W = 32
#: blocks of COOP_W each lane loads ahead of the fold
comptime COOP_R = 8


@always_inline
def coop_sumsq(p: FP, start: Int, n: Int, lane: Int) -> Float32:
    """sum_{k < n} p[start + k]^2, k ascending, one fma per term, on every lane."""
    var acc = Float32(0.0)
    var k = 0
    while k + COOP_W * COOP_R <= n:
        var v = SIMD[DType.float32, COOP_R]()
        comptime for r in range(COOP_R):
            v[r] = ld(p, start + k + r * COOP_W + lane)
        comptime for r in range(COOP_R):
            comptime for j in range(COOP_W):
                var x = shuffle_idx(v[r], UInt32(j))
                acc = fma3(x, x, acc)
        k += COOP_W * COOP_R
    while k < n:
        var m = min(COOP_W, n - k)
        var x0 = ld(p, start + k + lane) if lane < m else Float32(0.0)
        for j in range(m):
            var x = shuffle_idx(x0, UInt32(j))
            acc = fma3(x, x, acc)
        k += COOP_W
    return acc


@always_inline
def coop_dot(pa: FP, abase: Int, sak: Int, pb: FP, bbase: Int, sbk: Int, K: Int, acc0: Float32,
             lane: Int) -> Float32:
    """gemm_dot's chain: acc0 + sum_k pa[abase + k sak] pb[k sbk + bbase], k
    ascending, on every lane."""
    var acc = acc0
    var k = 0
    while k + COOP_W * COOP_R <= K:
        var va = SIMD[DType.float32, COOP_R]()
        var vb = SIMD[DType.float32, COOP_R]()
        comptime for r in range(COOP_R):
            var kk = k + r * COOP_W + lane
            va[r] = ld(pa, abase + kk * sak)
            vb[r] = ld(pb, kk * sbk + bbase)
        comptime for r in range(COOP_R):
            comptime for j in range(COOP_W):
                acc = fma3(shuffle_idx(va[r], UInt32(j)), shuffle_idx(vb[r], UInt32(j)), acc)
        k += COOP_W * COOP_R
    while k < K:
        var m = min(COOP_W, K - k)
        var kk = k + lane
        var x0 = ld(pa, abase + kk * sak) if lane < m else Float32(0.0)
        var y0 = ld(pb, kk * sbk + bbase) if lane < m else Float32(0.0)
        for j in range(m):
            acc = fma3(shuffle_idx(x0, UInt32(j)), shuffle_idx(y0, UInt32(j)), acc)
        k += COOP_W
    return acc


@always_inline
def apply_coop[OP: Int](cell: Int, lane: Int, a: Args):
    """Cell `cell` of OP on one simdgroup. A FAST launch that carries
    partials (i2 / i0 / i1 set) runs the one-thread op on lane 0."""
    comptime if OP == OP_AF_ALPHA:
        if a.i2 > 0:
            if lane == 0:
                op_af_alpha(cell, a)
            return
        var ss = coop_sumsq(a.p0, 0, a.i0, lane)
        if lane == 0:
            af_alpha_tail(a, ss)
    elif OP == OP_AF_DENOM:
        if a.i2 > 0:
            if lane == 0:
                op_af_denom(cell, a)
            return
        var ss = coop_sumsq(a.p0, 0, a.i0, lane)
        if lane == 0:
            af_denom_tail(a, ss)
    elif OP == OP_AF_BLK_SUMSQ:
        # op_af_blk_sumsq's block chain on the simdgroup
        var lo = cell * a.i1
        var ss = coop_sumsq(a.p0, lo, min(a.i1, a.i0 - lo), lane)
        if lane == 0:
            st(a.p1, cell, ss)
    elif OP == OP_SEG_SUMSQ:
        if a.i0 != 0:
            if lane == 0:
                op_seg_sumsq(cell, a)
            return
        var s = Int(a.p1.unsafe_load(cell))
        var e = Int(a.p1.unsafe_load(cell + 1))
        var ss = coop_sumsq(a.p0, s, e - s, lane)
        if lane == 0:
            st(a.p2, cell, ss)
    elif OP == OP_LAMB_RATIO:
        if a.i1 != 0:
            if lane == 0:
                op_lamb_ratio(cell, a)
            return
        var s = Int(a.p2.unsafe_load(cell))
        var e = Int(a.p2.unsafe_load(cell + 1))
        var pss = coop_sumsq(a.p0, s, e - s, lane)
        var uss = coop_sumsq(a.p1, s, e - s, lane)
        if lane == 0:
            lamb_ratio_tail(a, cell, pss, uss)
    elif OP == OP_GEMM:
        # op_gemm's cell, the fold on the simdgroup
        var n_cols = a.i1
        var m = cell // n_cols
        var n = cell - m * n_cols
        var ci = m * a.i8 + n
        var acc = Float32(0.0)
        if a.i7 != 0:
            acc = ld(a.p2, ci)
        acc = coop_dot(a.p0, m * a.i3, a.i4, a.p1, n * a.i6, a.i5, a.i2, acc, lane)
        if lane == 0:
            st(a.p2, ci, acc)
