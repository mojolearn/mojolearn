# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Serial folds fed by a SIMD group (lane prep-apple2).

A unit that folds a column in ascending row order on ONE thread waits a
memory latency every RUN rows: the next rows' loads come after the current
rows' adds in program order, so nothing overlaps them (43 ns a row measured
in PowerTransformer's folds on the M4; a shorter add chain, `ftz_chain`,
changed nothing). Here a SIMD group runs one unit: every lane loads its own
rows, SP blocks of WARP_SIZE rows at once (SP * WARP_SIZE rows in flight
instead of RUN), and every lane then folds ALL the rows in ascending order,
each word broadcast from the lane that loaded it (`shuffle_idx`). Every lane
computes the same chain of the same words in the same order as the
one-thread unit, so the result is its word; lane 0 stores it.
"""
from std.gpu import WARP_SIZE, block_idx, thread_idx
from std.gpu.primitives.warp import shuffle_idx
from checks.numerics import ftz
from x_prep.common import FP, IP, p, ld, raw, ldi, sti, st
from x_prep.prims import acc_add, sub, mul, div, logf
from x_prep.transform import PT_STATE, _pt_take1, _pt_take2

#: blocks of WARP_SIZE rows each lane holds between folds
comptime SP = 4


@always_inline
def _bc(v: Float32, u: Int) -> Float32:
    return shuffle_idx(v, UInt32(u))


def pt_sfold_simd_kernel(f: FP, q: IP):
    """`pt_sfold_unit` for unit t = block_idx.x (t = c*M + j) on one SIMD
    group (block_dim = WARP_SIZE): the same VALS[t] (and, FIRST, j == 0,
    the column's sum J and row count)."""
    var t = Int(block_idx.x)
    var lane = Int(thread_idx.x)
    var X = p(q, 0)
    var n = p(q, 1)
    var d = p(q, 2)
    var method = p(q, 3)
    var M = p(q, 5)
    var c = t // M
    var j = t % M
    var S = p(q, 6) + c * PT_STATE
    if raw(f, S + 7) != Float32(0):
        return
    var first = p(q, 9) != 0
    var nonan = not first and ldi(f, S + 9) == n
    var cnt = 0
    var sm = Float32(0)
    var sj = Float32(0)
    if not first:
        sj = raw(f, S + 8)
    var Tc = p(q, 4) + t * n
    var W = WARP_SIZE
    var i0 = 0
    while i0 < n:
        var tv = InlineArray[Float32, SP](fill=Float32(0))
        var xv = InlineArray[Float32, SP](fill=Float32(0))
        comptime for b in range(SP):
            var i = i0 + b * W + lane
            if i < n:
                tv[b] = raw(f, Tc + i)
                if first:
                    xv[b] = ld(f, X + i * d + c)
        comptime for b in range(SP):
            for u in range(W):
                var i = i0 + b * W + u
                if i >= n:
                    break
                var tu = _bc(tv[b], u)
                if first:
                    var xu = _bc(xv[b], u)
                    _pt_take1(xu, tu, method, True, cnt, sm, sj)
                elif nonan:
                    sm = acc_add(sm, tu)
                else:
                    _pt_take1(tu, tu, method, False, cnt, sm, sj)
        i0 += SP * W
    if nonan:
        cnt = n
    if first and j == 0 and lane == 0:
        f.unsafe_store(S + 8, sj)
        sti(f, S + 9, cnt)
    var ss = Float32(0)
    if cnt > 0:
        var mean = div(sm, Float32(cnt))
        var full = cnt == n and n > 0
        i0 = 0
        while i0 < n:
            var tv = InlineArray[Float32, SP](fill=Float32(0))
            comptime for b in range(SP):
                var i = i0 + b * W + lane
                if i < n:
                    tv[b] = raw(f, Tc + i)
            comptime for b in range(SP):
                for u in range(W):
                    var i = i0 + b * W + u
                    if i >= n:
                        break
                    var tu = _bc(tv[b], u)
                    if full:
                        var e = sub(tu, mean)
                        ss = acc_add(ss, mul(e, e))
                    else:
                        _pt_take2(tu, tu, mean, ss)
            i0 += SP * W
    if lane != 0:
        return
    var lam = raw(f, p(q, 7) + t)
    var val = Float32(0)
    if cnt > 0:
        var var_ = div(ss, Float32(cnt))
        val = sub(mul(mul(Float32(0.5), Float32(cnt)), logf(var_)), mul(sub(lam, Float32(1)), sj))
    f.unsafe_store(p(q, 8) + t, val)


def ii_mean_simd_kernel(f: FP, q: IP):
    """`ii_mean_unit` for column t = block_idx.x on one SIMD group: each lane
    loads its rows' mask and value, every lane folds the observed rows'
    values in ascending order. The same MEANS[t] (and CNT for t == 0)."""
    var t = Int(block_idx.x)
    var lane = Int(thread_idx.x)
    if ld(f, p(q, 7)) != Float32(0):
        return
    var n = p(q, 1)
    var d = p(q, 2)
    var j = p(q, 4)
    var Mk = p(q, 3)
    var X = p(q, 0)
    var s = Float32(0)
    var cnt = 0
    var W = WARP_SIZE
    var i0 = 0
    while i0 < n:
        var xv = InlineArray[Float32, SP](fill=Float32(0))
        var ok = InlineArray[Float32, SP](fill=Float32(0))
        comptime for b in range(SP):
            var i = i0 + b * W + lane
            if i < n and ld(f, Mk + i * d + j) == Float32(0):
                ok[b] = Float32(1)
                xv[b] = ld(f, X + i * d + t)
        comptime for b in range(SP):
            for u in range(W):
                if i0 + b * W + u >= n:
                    break
                var xu = _bc(xv[b], u)
                if _bc(ok[b], u) != Float32(0):
                    s = acc_add(s, xu)
                    cnt += 1
        i0 += SP * W
    if lane != 0:
        return
    st(f, p(q, 5) + t, div(s, Float32(cnt)) if cnt > 0 else Float32(0))
    if t == 0:
        st(f, p(q, 6), Float32(cnt))


def ii_gram_simd_kernel(f: FP, q: IP):
    """`ii_gram_unit` for t = block_idx.x (t = a*d + b) on one SIMD group:
    each lane forms its rows' centred products (the unit's own sub and mul,
    row by row), every lane folds the observed rows' products in ascending
    order. The same G[t]."""
    var t = Int(block_idx.x)
    var lane = Int(thread_idx.x)
    if ld(f, p(q, 7)) != Float32(0):
        return
    var n = p(q, 1)
    var d = p(q, 2)
    var j = p(q, 4)
    var Mk = p(q, 3)
    var X = p(q, 0)
    var a = t // d
    var bcol = t % d
    var ma = ld(f, p(q, 5) + a)
    var mb = ld(f, p(q, 5) + bcol)
    var s = Float32(0)
    var W = WARP_SIZE
    var i0 = 0
    while i0 < n:
        var pv = InlineArray[Float32, SP](fill=Float32(0))
        var ok = InlineArray[Float32, SP](fill=Float32(0))
        comptime for b in range(SP):
            var i = i0 + b * W + lane
            if i < n and ld(f, Mk + i * d + j) == Float32(0):
                ok[b] = Float32(1)
                pv[b] = mul(sub(ld(f, X + i * d + a), ma), sub(ld(f, X + i * d + bcol), mb))
        comptime for b in range(SP):
            for u in range(W):
                if i0 + b * W + u >= n:
                    break
                var pu = _bc(pv[b], u)
                if _bc(ok[b], u) != Float32(0):
                    s = acc_add(s, pu)
        i0 += SP * W
    if lane == 0:
        st(f, p(q, 6) + t, s)
