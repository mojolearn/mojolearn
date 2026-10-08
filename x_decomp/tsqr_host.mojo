# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE HOST REPLAY OF THE BLOCKED TSQR (lane neural-pass140, 2026-10-02):
x_decomp/tsqr_core.mojo's order in host loops, statement for statement the
device kernels of x_decomp/tsqr_device.mojo. No GPU import: the CPU binding
compiles this file, and it is the whole route on a CPU-only install.

Every chain is the device's: an inner product over a block's rows is TS_P
accumulators filled in ONE ascending pass (row i feeds accumulator i mod
TS_P, so each accumulator sees its rows ascending, which is what device
thread g does), folded by `ts_fold`. The blocks are independent and go over
host tasks (`host_parallelize`, the caller's floating-point environment); so
are the pairs of one tree level. Which task runs a block never changes what
it computes, so the bits are the same at every thread count.

THE FACTORED STATE. `ts_factor_host(..., keep=True)` keeps the factored
matrix, the panels' T factors, the tree tiles and the tree's tau in ONE
process slot until `ts_apply_host` (or `ts_free_host`) uses it; a new factor
replaces it."""
from std.ffi import _Global
from std.memory import memcpy

from checks.numerics import ftz, identical_div, identical_sqrt
from core.host_parallel import host_parallelize
from x_decomp.cells import F32Ptr
from x_decomp.tsqr_core import (
    TS_T2,
    TS_TPB,
    TS_TREE_ARITY,
    TS_TREE_PAR,
    TS_NB,
    TS_P,
    TS_W2,
    TS_WY_PAIR,
    ts_block_hi,
    ts_block_lo,
    ts_blocks,
    ts_fma,
    ts_fold,
    ts_pair_width,
    ts_pairs,
    ts_panels,
    ts_reflector,
    ts_scale,
    ts_tree_fold,
)

comptime _TT = TS_NB * TS_NB


struct _TsHost(Defaultable, Movable):
    var a: List[Float32]
    var t: List[Float32]
    var t2: List[Float32]
    var tiles: List[Float32]
    var tau: List[Float32]
    var m: Int
    var n: Int
    var live: Bool

    def __init__(out self):
        self.a = List[Float32]()
        self.t = List[Float32]()
        self.t2 = List[Float32]()
        self.tiles = List[Float32]()
        self.tau = List[Float32]()
        self.m = 0
        self.n = 0
        self.live = False


comptime TS_HOST_STATE = _Global[StorageType=_TsHost, name="MojoXDecompTsqrHost", init_fn=_TsHost.__init__]


def ts_free_host() raises:
    var st = TS_HOST_STATE.get_or_create_ptr()
    st[].a = List[Float32]()
    st[].t = List[Float32]()
    st[].t2 = List[Float32]()
    st[].tiles = List[Float32]()
    st[].tau = List[Float32]()
    st[].m = 0
    st[].n = 0
    st[].live = False


def _p(l: List[Float32]) -> F32Ptr:
    return F32Ptr(unsafe_from_address=Int(l.unsafe_ptr()))


def ts_dot_host(x: F32Ptr, ldx: Int, cx: Int, y: F32Ptr, ldy: Int, cy: Int, lo: Int, hi: Int) -> Float32:
    """sum over rows [lo, hi) of x[i, cx] y[i, cy]: TS_P chains (row i into
    chain i mod TS_P, ascending), then `ts_fold`."""
    var part = InlineArray[Float32, TS_P](fill=Float32(0.0))
    var g = lo % TS_P
    for i in range(lo, hi):
        part[g] = ts_fma(ftz(x.unsafe_load(i * ldx + cx)), ftz(y.unsafe_load(i * ldy + cy)), part[g])
        g += 1
        if g == TS_P:
            g = 0
    return ts_fold(part)


def ts_wy_host[transpose: Bool, W: Int = TS_NB](
    y: F32Ptr, ldy: Int, x: F32Ptr, ldx: Int, c_lo: Int, c_hi: Int, mb: Int, j0: Int, pw: Int,
    t: InlineArray[Float32, W * W],
):
    """Columns [c_lo, c_hi) of x (mb rows) times the block reflector of pw
    (<= W) reflectors with the W x W compact WY factor t: (I - Y T^T Y^T) x
    when `transpose` (the factorization), (I - Y T Y^T) x otherwise (Q C).
    Y is the columns j0 .. j0 + pw - 1 of y: unit at row j0 + p, the stored
    tail below it, zero above. W = TS_NB for one panel, TS_W2 for a pair
    (TS_WY_PAIR; the device's `_wy_chunk_kern` and `_wy_chunk_kern2`)."""
    for c in range(c_lo, c_hi):
        var z = InlineArray[Float32, W](fill=Float32(0.0))
        for p in range(pw):
            var tail = ts_dot_host(y, ldy, j0 + p, x, ldx, c, j0 + p + 1, mb)
            z[p] = ftz(ftz(x.unsafe_load((j0 + p) * ldx + c)) + tail)
        var w = InlineArray[Float32, W](fill=Float32(0.0))
        for p in range(pw):
            var acc = Float32(0.0)
            comptime if transpose:
                for q in range(p + 1):
                    acc = ts_fma(t[q * W + p], z[q], acc)
            else:
                for q in range(p, pw):
                    acc = ts_fma(t[p * W + q], z[q], acc)
            w[p] = acc
        for i in range(j0, mb):
            var acc = ftz(x.unsafe_load(i * ldx + c))
            var top = min(pw, i - j0 + 1)
            for p in range(top):
                var yv = Float32(1.0) if i == j0 + p else ftz(y.unsafe_load(i * ldy + j0 + p))
                acc = ts_fma(-w[p], yv, acc)
            x.unsafe_store(i * ldx + c, acc)


def ts_pair_t_host(
    blk: F32Ptr, mb: Int, n: Int, pa: Int, npan: Int, ta: InlineArray[Float32, _TT], tb: InlineArray[Float32, _TT],
    mut t2: InlineArray[Float32, TS_T2],
):
    """TS_WY_PAIR: the pair's T2 (TS_W2 x TS_W2) from panel pa's Ta and panel
    pa + 1's Tb (zeros when that panel does not exist): T2 = [[Ta, -Ta (G Tb)],
    [0, Tb]] with G = Ya^T Yb, G[p, q] = Ya[j0b + q, p] + sum over rows i >
    j0b + q of Ya[i, p] Yb[i, q] (Yb's unit row, then its stored tail: the
    TS_P chains of `ts_dot_host`). GT[k, q] folds kk = 0 .. pwb - 1 ascending,
    M[p, q] = Ta GT folds k = 0 .. TS_NB - 1 ascending, every entry from zero;
    the device's `ts_pair_t_kernel` runs the same loops."""
    var j0a = pa * TS_NB
    var j0b = j0a + TS_NB
    var has_b = pa + 1 < npan
    var pwb = min(TS_NB, n - j0b) if has_b else 0
    var g = InlineArray[Float32, _TT](fill=Float32(0.0))
    var gt = InlineArray[Float32, _TT](fill=Float32(0.0))
    var mm = InlineArray[Float32, _TT](fill=Float32(0.0))
    if has_b:
        for p in range(TS_NB):
            for q in range(pwb):
                var tail = ts_dot_host(blk, n, j0a + p, blk, n, j0b + q, j0b + q + 1, mb)
                g[p * TS_NB + q] = ftz(ftz(blk.unsafe_load((j0b + q) * n + j0a + p)) + tail)
        for p in range(TS_NB):
            for q in range(TS_NB):
                var acc = Float32(0.0)
                for kk in range(pwb):
                    acc = ts_fma(g[p * TS_NB + kk], tb[kk * TS_NB + q], acc)
                gt[p * TS_NB + q] = acc
        for p in range(TS_NB):
            for q in range(TS_NB):
                var acc = Float32(0.0)
                for k in range(TS_NB):
                    acc = ts_fma(ta[p * TS_NB + k], gt[k * TS_NB + q], acc)
                mm[p * TS_NB + q] = acc
    for e in range(TS_T2):
        var r = e // TS_W2
        var c = e - r * TS_W2
        var v = Float32(0.0)
        if r < TS_NB and c < TS_NB:
            v = ta[r * TS_NB + c]
        elif r >= TS_NB and c >= TS_NB:
            v = tb[(r - TS_NB) * TS_NB + (c - TS_NB)]
        elif r < TS_NB and c >= TS_NB and has_b:
            v = -mm[r * TS_NB + (c - TS_NB)]
        t2[e] = v


def ts_factor_block_host(blk: F32Ptr, mb: Int, n: Int, tst: F32Ptr, tst2: F32Ptr):
    """One leaf block (mb x n, leading dimension n) factored in place; its
    panels' T factors into tst (ts_panels(n) x TS_NB x TS_NB) and, under
    TS_WY_PAIR, its pairs' T2 into tst2 (ts_pairs(n) x TS_W2 x TS_W2)."""
    var npan = ts_panels(n)
    comptime if TS_WY_PAIR:
        for q in range(ts_pairs(n)):
            var pa = 2 * q
            var j0a = pa * TS_NB
            var ta = _ts_panel_host(blk, mb, n, pa, tst)
            var tb = InlineArray[Float32, _TT](fill=Float32(0.0))
            if pa + 1 < npan:
                # panel a's reflectors on panel b's columns only: the words
                # panel b sees before its own factorization
                var j0b = j0a + TS_NB
                var pwb = min(TS_NB, n - j0b)
                ts_wy_host[True, TS_NB](blk, n, blk, n, j0b, j0b + pwb, mb, j0a, TS_NB, ta)
                tb = _ts_panel_host(blk, mb, n, pa + 1, tst)
            var t2 = InlineArray[Float32, TS_T2](fill=Float32(0.0))
            ts_pair_t_host(blk, mb, n, pa, npan, ta, tb, t2)
            for e in range(TS_T2):
                tst2.unsafe_store(q * TS_T2 + e, t2[e])
            # (U) the columns right of the pair, one blocked update
            var pw2 = ts_pair_width(n, q)
            ts_wy_host[True, TS_W2](blk, n, blk, n, j0a + pw2, n, mb, j0a, pw2, t2)
    else:
        for pan in range(npan):
            var j0 = pan * TS_NB
            var pw = min(TS_NB, n - j0)
            var t = _ts_panel_host(blk, mb, n, pan, tst)
            # (U) the trailing columns, one blocked update
            ts_wy_host[True, TS_NB](blk, n, blk, n, j0 + pw, n, mb, j0, pw, t)


def _ts_panel_host(blk: F32Ptr, mb: Int, n: Int, pan: Int, tst: F32Ptr) -> InlineArray[Float32, _TT]:
    """Panel `pan` of a leaf block: (F) its reflectors and (T) its compact WY
    factor T, stored into tst's slot and returned. The trailing update is
    the caller's."""
    var j0 = pan * TS_NB
    var pw = min(TS_NB, n - j0)
    var t = InlineArray[Float32, _TT](fill=Float32(0.0))
    # (F) the panel's reflectors, one at a time, each applied to the
    # panel's columns to its right
    for p in range(pw):
        var j = j0 + p
        var sigma = ts_dot_host(blk, n, j, blk, n, j, j, mb)
        var normx = ftz(identical_sqrt(sigma))
        var ajj = ftz(blk.unsafe_load(j * n + j))
        var tau = Float32(0.0)
        if normx == Float32(0.0):
            blk.unsafe_store(j * n + j, Float32(0.0))
            for i in range(j + 1, mb):
                blk.unsafe_store(i * n + j, Float32(0.0))
        else:
            var rf = ts_reflector(ajj, normx)
            var u1 = rf[1]
            tau = rf[2]
            blk.unsafe_store(j * n + j, rf[0])
            for i in range(j + 1, mb):
                blk.unsafe_store(i * n + j, ftz(identical_div(ftz(blk.unsafe_load(i * n + j)), u1)))
        t[p * TS_NB + p] = tau
        if tau != Float32(0.0):
            for c in range(j + 1, j0 + pw):
                var ajc = ftz(blk.unsafe_load(j * n + c))
                var tail = ts_dot_host(blk, n, j, blk, n, c, j + 1, mb)
                var td = ts_scale(tau, ftz(ajc + tail))
                blk.unsafe_store(j * n + c, ftz(ajc - td))
                for i in range(j + 1, mb):
                    blk.unsafe_store(i * n + c, ts_fma(-td, ftz(blk.unsafe_load(i * n + j)), ftz(blk.unsafe_load(i * n + c))))
    # (T) larft, forward and columnwise: T[0:p, p] = -tau_p T[0:p, 0:p] (Y^T y_p)
    for p in range(1, pw):
        var gq = InlineArray[Float32, TS_NB](fill=Float32(0.0))
        for q in range(p):
            var tail = ts_dot_host(blk, n, j0 + q, blk, n, j0 + p, j0 + p + 1, mb)
            gq[q] = ftz(ftz(blk.unsafe_load((j0 + p) * n + j0 + q)) + tail)
        var ntau = -t[p * TS_NB + p]
        for q in range(p):
            var s = Float32(0.0)
            for kk in range(q, p):
                s = ts_fma(t[q * TS_NB + kk], gq[kk], s)
            t[q * TS_NB + p] = ts_scale(ntau, s)
    for e in range(_TT):
        tst.unsafe_store(pan * _TT + e, t[e])
    return t^


def ts_rtile_host(blk: F32Ptr, n: Int, tile: F32Ptr):
    for i in range(n):
        for c in range(n):
            tile.unsafe_store(i * n + c, ftz(blk.unsafe_load(i * n + c)) if c >= i else Float32(0.0))


def ts_combine_host(top: F32Ptr, bot: F32Ptr, taus: F32Ptr, n: Int):
    """The structured QR of [top; bot] (two n x n upper triangles): R into
    top, the reflectors' lower parts into bot's upper triangle, tau into
    taus[0, n)."""
    var lanes = List[Float32](length=TS_TPB, fill=Float32(0.0))
    for j in range(n):
        var alpha = ftz(top.unsafe_load(j * n + j))
        var s = Float32(0.0)
        comptime if TS_TREE_PAR:
            # the device's lane fold: lane t holds alpha^2 (t == 0) then rows
            # i == t (mod TS_TPB) ascending; the halving tree `ts_tree_fold`
            for t in range(TS_TPB):
                var acc = ts_fma(alpha, alpha, Float32(0.0)) if t == 0 else Float32(0.0)
                var i = t
                while i <= j:
                    var x = ftz(bot.unsafe_load(i * n + j))
                    acc = ts_fma(x, x, acc)
                    i += TS_TPB
                lanes[t] = acc
            ts_tree_fold(lanes)
            s = lanes[0]
        else:
            s = ts_fma(alpha, alpha, Float32(0.0))
            for i in range(j + 1):
                var x = ftz(bot.unsafe_load(i * n + j))
                s = ts_fma(x, x, s)
        var normx = ftz(identical_sqrt(s))
        var tau = Float32(0.0)
        if normx == Float32(0.0):
            top.unsafe_store(j * n + j, Float32(0.0))
            for i in range(j + 1):
                bot.unsafe_store(i * n + j, Float32(0.0))
        else:
            var rf = ts_reflector(alpha, normx)
            var u1 = rf[1]
            tau = rf[2]
            top.unsafe_store(j * n + j, rf[0])
            for i in range(j + 1):
                bot.unsafe_store(i * n + j, ftz(identical_div(ftz(bot.unsafe_load(i * n + j)), u1)))
        taus.unsafe_store(j, tau)
        if tau != Float32(0.0):
            for c in range(j + 1, n):
                var tail = Float32(0.0)
                for i in range(j + 1):
                    tail = ts_fma(ftz(bot.unsafe_load(i * n + j)), ftz(bot.unsafe_load(i * n + c)), tail)
                var tc = ftz(top.unsafe_load(j * n + c))
                var td = ts_scale(tau, ftz(tc + tail))
                top.unsafe_store(j * n + c, ftz(tc - td))
                for i in range(j + 1):
                    bot.unsafe_store(i * n + c, ts_fma(-td, ftz(bot.unsafe_load(i * n + j)), ftz(bot.unsafe_load(i * n + c))))


def ts_capply_host(xt: F32Ptr, xb: F32Ptr, v: F32Ptr, taus: F32Ptr, n: Int, k: Int):
    """[xt; xb] = H_0 ... H_{n-1} [xt; 0] for one tree node (n x k each)."""
    for c in range(k):
        for i in range(n):
            xb.unsafe_store(i * k + c, Float32(0.0))
        for jj in range(n):
            var j = n - 1 - jj
            var tau = taus.unsafe_load(j)
            if tau == Float32(0.0):
                continue
            var tail = Float32(0.0)
            for i in range(j + 1):
                tail = ts_fma(ftz(v.unsafe_load(i * n + j)), ftz(xb.unsafe_load(i * k + c)), tail)
            var xv = ftz(xt.unsafe_load(j * k + c))
            var td = ts_scale(tau, ftz(xv + tail))
            xt.unsafe_store(j * k + c, ftz(xv - td))
            for i in range(j + 1):
                xb.unsafe_store(i * k + c, ts_fma(-td, ftz(v.unsafe_load(i * n + j)), ftz(xb.unsafe_load(i * k + c))))


def ts_apply_block_host(blk: F32Ptr, n: Int, tst: F32Ptr, tst2: F32Ptr, x: F32Ptr, k: Int, mb: Int, cb: F32Ptr):
    """x (mb x k) = Q_block [cb; 0]: the panels (TS_WY_PAIR: the pairs) last
    to first."""
    for i in range(mb):
        for c in range(k):
            x.unsafe_store(i * k + c, cb.unsafe_load(i * k + c) if i < n else Float32(0.0))
    comptime if TS_WY_PAIR:
        var npair = ts_pairs(n)
        for qq in range(npair):
            var q = npair - 1 - qq
            var t2 = InlineArray[Float32, TS_T2](fill=Float32(0.0))
            for e in range(TS_T2):
                t2[e] = tst2.unsafe_load(q * TS_T2 + e)
            ts_wy_host[False, TS_W2](blk, n, x, k, 0, k, mb, 2 * q * TS_NB, ts_pair_width(n, q), t2)
    else:
        var npan = ts_panels(n)
        for pp in range(npan):
            var pan = npan - 1 - pp
            var j0 = pan * TS_NB
            var pw = min(TS_NB, n - j0)
            var t = InlineArray[Float32, _TT](fill=Float32(0.0))
            for e in range(_TT):
                t[e] = tst.unsafe_load(pan * _TT + e)
            ts_wy_host[False, TS_NB](blk, n, x, k, 0, k, mb, j0, pw, t)


def _combine_level_host(ptl: F32Ptr, ptau: F32Ptr, n: Int, nb: Int, s: Int):
    var pairs = (nb + TS_TREE_ARITY * s - 1) // (TS_TREE_ARITY * s)
    # C24 fixed higher-arity node: append children left to right, retaining
    # each child's reflector in that child's tile exactly as binary B does.
    for child in range(1, TS_TREE_ARITY):
        def _pair(t: Int) {imm ptl, imm ptau, imm n, imm nb, imm s, imm child}:
            var ia = TS_TREE_ARITY * s * t
            var ib = ia + child * s
            if ib < nb:
                ts_combine_host(ptl + ia * n * n, ptl + ib * n * n, ptau + ib * n, n)
        if pairs <= 1:
            _pair(0)
        else:
            host_parallelize(_pair, pairs)


def _capply_level_host(pcb: F32Ptr, ptl: F32Ptr, ptau: F32Ptr, n: Int, k: Int, nb: Int, s: Int):
    var pairs = (nb + TS_TREE_ARITY * s - 1) // (TS_TREE_ARITY * s)
    for offset in range(1, TS_TREE_ARITY):
        var child = TS_TREE_ARITY - offset
        def _pair(t: Int) {imm pcb, imm ptl, imm ptau, imm n, imm k, imm nb, imm s, imm child}:
            var ia = TS_TREE_ARITY * s * t
            var ib = ia + child * s
            if ib < nb:
                ts_capply_host(pcb + ia * n * k, pcb + ib * n * k, ptl + ib * n * n, ptau + ib * n, n, k)
        if pairs <= 1:
            _pair(0)
        else:
            host_parallelize(_pair, pairs)


def ts_factor_host(a: F32Ptr, bp: F32Ptr, r: F32Ptr, m: Int, d: Int, nrhs: Int, keep: Bool) raises:
    """R (n x n, n = d + nrhs) of [a | bp] (m x d and m x nrhs, row major)
    into r; with `keep` the factored state stays for `ts_apply_host`."""
    var n = d + nrhs
    ts_free_host()
    var st = TS_HOST_STATE.get_or_create_ptr()
    var nb = ts_blocks(m)
    var npan = ts_panels(n)
    var npair = ts_pairs(n)
    st[].a = List[Float32](length=m * n, fill=Float32(0.0))
    st[].t = List[Float32](length=nb * npan * _TT, fill=Float32(0.0))
    st[].t2 = List[Float32](length=(nb * npair * TS_T2) if TS_WY_PAIR else 1, fill=Float32(0.0))
    st[].tiles = List[Float32](length=nb * n * n, fill=Float32(0.0))
    st[].tau = List[Float32](length=nb * n, fill=Float32(0.0))
    var pa = _p(st[].a)
    var pt = _p(st[].t)
    var pt2 = _p(st[].t2)
    var ptl = _p(st[].tiles)
    var ptau = _p(st[].tau)
    if nrhs == 0:
        memcpy(dest=pa, src=a, count=m * n)
    else:
        for i in range(m):
            memcpy(dest=pa + i * n, src=a + i * d, count=d)
            memcpy(dest=pa + i * n + d, src=bp + i * nrhs, count=nrhs)

    def _blk(b: Int) {imm pa, imm pt, imm pt2, imm ptl, imm m, imm n, imm nb, imm npan, imm npair}:
        var lo = ts_block_lo(b)
        var hi = ts_block_hi(b, nb, m)
        ts_factor_block_host(pa + lo * n, hi - lo, n, pt + b * npan * _TT, pt2 + b * npair * TS_T2)
        ts_rtile_host(pa + lo * n, n, ptl + b * n * n)

    if nb <= 1:
        _blk(0)
    else:
        host_parallelize(_blk, nb)
    var s = 1
    while s < nb:
        _combine_level_host(ptl, ptau, n, nb, s)
        s *= TS_TREE_ARITY
    memcpy(dest=r, src=ptl, count=n * n)
    if keep:
        st[].m = m
        st[].n = n
        st[].live = True
    else:
        ts_free_host()


def ts_apply_host(c: F32Ptr, q: F32Ptr, m: Int, n: Int, k: Int) raises:
    """q (m x k) = Q c for the kept factorization of an m x n matrix (c is
    n x k), then the state is released."""
    var st = TS_HOST_STATE.get_or_create_ptr()
    if not st[].live or st[].m != m or st[].n != n:
        ts_free_host()
        raise Error("x_decomp tsqr: no kept factorization of this shape (tsqr_r with keep first)")
    var nb = ts_blocks(m)
    var npan = ts_panels(n)
    var npair = ts_pairs(n)
    var cbuf = List[Float32](length=nb * n * k, fill=Float32(0.0))
    var pcb = _p(cbuf)
    memcpy(dest=pcb, src=c, count=n * k)
    var pa = _p(st[].a)
    var pt = _p(st[].t)
    var pt2 = _p(st[].t2)
    var ptl = _p(st[].tiles)
    var ptau = _p(st[].tau)
    var strides = List[Int]()
    var s = 1
    while s < nb:
        strides.append(s)
        s *= TS_TREE_ARITY
    for li in range(len(strides)):
        _capply_level_host(pcb, ptl, ptau, n, k, nb, strides[len(strides) - 1 - li])

    def _blk(b: Int) {imm pa, imm pt, imm pt2, imm pcb, imm q, imm m, imm n, imm k, imm nb, imm npan, imm npair}:
        var lo = ts_block_lo(b)
        var hi = ts_block_hi(b, nb, m)
        ts_apply_block_host(
            pa + lo * n, n, pt + b * npan * _TT, pt2 + b * npair * TS_T2, q + lo * k, k, hi - lo, pcb + b * n * k
        )

    if nb <= 1:
        _blk(0)
    else:
        host_parallelize(_blk, nb)
    _ = cbuf^
    ts_free_host()
