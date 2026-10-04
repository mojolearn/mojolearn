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
for the ops below on an Apple GPU, and (nr-small D1/D11, IDENTICAL,
`MOJOLEARN_IDN_SEQ_COOP_NVAMD`) on NVIDIA and AMD; the host runs the
one-thread op, the same chain."""
from std.gpu.primitives.id import lane_id
from std.gpu.primitives.warp import shuffle_idx
from checks.kernel_matrix import TARGET_COLUMN, lib_lane_width_for

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
from sequence.ops import OP_THETA
from sequence.ops import OP_LN_BWD_X, OP_LN_FWD, add, mul, sub
from sequence.layernorm import div as ln_div
from checks.numerics import ftz, identical_rsqrt
from sequence.theta_spec import THETA_SPEC, op_theta_spec
from sequence.ops import OP_AF_RMEAN, OP_AF_ROW
from sequence.adafactor import af_rmean_tail, af_row_tail
from sequence.adafactor import af_alpha_tail, af_denom_tail, lamb_ratio_tail, op_af_alpha, op_af_denom, op_lamb_ratio, op_seg_sumsq

#: the Apple simdgroup
comptime COOP_W = 32
#: blocks of COOP_W each lane loads ahead of the fold
comptime COOP_R = 8
#: nr-small D1/D11 (2026-10-04): on a 64-lane CDNA wavefront one wave holds
#: TWO cells of COOP_W lanes, and `shuffle_idx` names a PHYSICAL source
#: lane, so lane j of the upper cell must read physical lane 32 + j (the
#: `neighbors/impl/topk/logical_warp32.mojo` form, masked to the half).
#: Communication scope only: the fold is the same chain on the same values.
comptime COOP_ON64 = lib_lane_width_for[TARGET_COLUMN]() == 64


@always_inline
def coop_bcast(v: Float32, j: Int) -> Float32:
    """Element j of the cell's COOP_W lanes, on every lane of the cell."""
    comptime if COOP_ON64:
        var half = UInt32(lane_id()) & UInt32(0xffffffe0)
        var mask = UInt(0xffffffff) << UInt(half)
        return shuffle_idx(mask, v, half | UInt32(j))
    return shuffle_idx(v, UInt32(j))


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
                var x = coop_bcast(v[r], j)
                acc = fma3(x, x, acc)
        k += COOP_W * COOP_R
    while k < n:
        var m = min(COOP_W, n - k)
        var x0 = ld(p, start + k + lane) if lane < m else Float32(0.0)
        for j in range(m):
            var x = coop_bcast(x0, j)
            acc = fma3(x, x, acc)
        k += COOP_W
    return acc


@always_inline
def coop_sum(p: FP, start: Int, n: Int, lane: Int) -> Float32:
    """sum_{k < n} p[start + k], k ascending from +0.0, one add per term
    (`add`), on every lane (nr-small D11: op_af_rmean's chain)."""
    var acc = Float32(0.0)
    var k = 0
    while k < n:
        var m = min(COOP_W, n - k)
        var x0 = ld(p, start + k + lane) if lane < m else Float32(0.0)
        for j in range(m):
            acc = add(acc, coop_bcast(x0, j))
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
                acc = fma3(coop_bcast(va[r], j), coop_bcast(vb[r], j), acc)
        k += COOP_W * COOP_R
    while k < K:
        var m = min(COOP_W, K - k)
        var kk = k + lane
        var x0 = ld(pa, abase + kk * sak) if lane < m else Float32(0.0)
        var y0 = ld(pb, kk * sbk + bbase) if lane < m else Float32(0.0)
        for j in range(m):
            acc = fma3(coop_bcast(x0, j), coop_bcast(y0, j), acc)
        k += COOP_W
    return acc


@always_inline
def coop_ln_fwd(row: Int, lane: Int, a: Args):
    """nr-small D10: `op_ln_fwd`'s row on a simdgroup. The mean and variance
    chains are `_stats`'s (adds, then fmas of (x - mean)^2, c ascending, on
    every lane over broadcast words); the outputs are elementwise, lane c
    writes columns c, c + COOP_W, ... Same words as the one-thread row."""
    var D = a.i0
    var base = row * D
    var s = Float32(0.0)
    var k = 0
    while k < D:
        var m = min(COOP_W, D - k)
        var x0 = ld(a.p0, base + k + lane) if lane < m else Float32(0.0)
        for j in range(m):
            s = add(s, coop_bcast(x0, j))
        k += COOP_W
    var mean = ln_div(s, Float32(D))
    var q = Float32(0.0)
    k = 0
    while k < D:
        var m = min(COOP_W, D - k)
        var d0 = sub(ld(a.p0, base + k + lane), mean) if lane < m else Float32(0.0)
        for j in range(m):
            var d = coop_bcast(d0, j)
            q = fma3(d, d, q)
        k += COOP_W
    var rstd = ftz(identical_rsqrt(add(ln_div(q, Float32(D)), a.f0)))
    if lane == 0:
        st(a.p4, row, mean)
        st(a.p5, row, rstd)
    var c = lane
    while c < D:
        var y = mul(sub(ld(a.p0, base + c), mean), rstd)
        if a.i1 != 0:
            y = mul(y, ld(a.p1, c))
        if a.i2 != 0:
            y = add(y, ld(a.p2, c))
        st(a.p3, base + c, y)
        c += COOP_W


@always_inline
def coop_ln_bwd_x(row: Int, lane: Int, a: Args):
    """nr-small D10: `op_ln_bwd_x`'s row on a simdgroup: g and xhat of column
    c are computed by its lane (the one-thread row's expressions), broadcast,
    and the sum(g) / sum(g xhat) chains run c ascending on every lane; the
    outputs are elementwise. Same words as the one-thread row."""
    var D = a.i0
    var base = row * D
    var mean = ld(a.p4, row)
    var rstd = ld(a.p5, row)
    var sg = Float32(0.0)
    var sgx = Float32(0.0)
    var k = 0
    while k < D:
        var m = min(COOP_W, D - k)
        var g0 = Float32(0.0)
        var xh0 = Float32(0.0)
        if lane < m:
            var c = k + lane
            g0 = ld(a.p0, base + c)
            if a.i1 != 0:
                g0 = mul(g0, ld(a.p2, c))
            xh0 = mul(sub(ld(a.p1, base + c), mean), rstd)
        for j in range(m):
            var g = coop_bcast(g0, j)
            var xh = coop_bcast(xh0, j)
            sg = add(sg, g)
            sgx = fma3(g, xh, sgx)
        k += COOP_W
    var mg = ln_div(sg, Float32(D))
    var mgx = ln_div(sgx, Float32(D))
    var c = lane
    while c < D:
        var g = ld(a.p0, base + c)
        if a.i1 != 0:
            g = mul(g, ld(a.p2, c))
        var xh = mul(sub(ld(a.p1, base + c), mean), rstd)
        st(a.p3, base + c, mul(rstd, sub(sub(g, mg), mul(xh, mgx))))
        c += COOP_W


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
    elif THETA_SPEC and OP == OP_THETA:
        # lane/apple-fast-gap-tsa: theta with the Nelder-Mead candidates on
        # the simdgroup's lanes (sequence/theta_spec.mojo)
        op_theta_spec(cell, lane, a)
    elif OP == OP_AF_ROW:
        # nr-small D11: op_af_row's row chain on the simdgroup
        var ss = coop_sumsq(a.p0, cell * a.i0, a.i0, lane)
        if lane == 0:
            af_row_tail(a, cell, ss)
    elif OP == OP_AF_RMEAN:
        var s = coop_sum(a.p0, 0, a.i0, lane)
        if lane == 0:
            af_rmean_tail(a, s)
    elif OP == OP_LN_FWD:
        coop_ln_fwd(cell, lane, a)
    elif OP == OP_LN_BWD_X:
        coop_ln_bwd_x(cell, lane, a)
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
