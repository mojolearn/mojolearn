# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""LassoCV / ElasticNetCV, FAST on Apple: every row pass on the whole GPU
(lane/apple-fast-classical, 2026-10-02).

The team fit (x_linear/cd.mojo `enetcv_fit`) runs on ONE block: seven
centered Grams (the grid on all rows, one per fold, the refit), each from a
row list the lead thread writes alone, one thread per Gram cell walking a
million rows, and the held-out squared errors folded by the lead. On the M3
Ultra that is 3.2 s (taxi) and 74 s (Istella) against scikit-learn's 0.2
and 7.

Here the fold statistics come from TWO passes over the rows, on the grid:

  1. the column sums of [X, y] per chunk (8,192 rows inside one fold), then
     each fold's means;
  2. each fold's Gram of [X, y] centered at that fold's own means, as 32 x 32
     tiles per chunk, then summed per fold.

Every training set (all rows but fold f; all rows) is then a sum of fold
Grams moved to the set's means by the parallel-axis rule
G_S = sum_g [C_g + n_g (mu_g - m_S)(mu_g - m_S)'], which never subtracts
two large uncentered sums. The paths (one block per fold and l1_ratio,
x_linear/cd.mojo `enet_gram_cd`'s coordinate descent, warm starts and
duality gap with the block's threads sharing each sweep, `_cd_block`),
the held-out errors (a grid over the rows, one block a chunk), the choice
and the refit (one block, a thread per candidate, then `_cd_block`)
follow; one read back at the end.

Folds are scikit-learn's KFold(shuffle=False) from Python's `_kfold_ids`
(contiguous, the first n % k one row longer). The device checks every row's
id against those bounds; on a mismatch the caller runs the team fit.
FAST promises quality, not bits. `MOJOLEARN_X_LINEAR_ENETCV_FAST=0` is the
A/B arm (the team fit in the same build).
"""
from std.gpu import block_idx, thread_idx
from std.memory import bitcast, stack_allocation
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from x_linear.ops import FP, IP, ld, st, ldi, sti, fa, fs, fm, fd, fabs, fmax, fmad, fsign, i2f
from x_linear.team import team_barrier
from x_linear.cd import alpha_grid_value
from x_linear.witness import Witness, witness_end, WITNESS_TRIES, WITNESS_ABORT

# AFCL-L03: NEVER RUN — PENDING MEASUREMENT. Uncompiled/unverified, OFF.
# Evaluate two held-out rows together to reuse the path coefficient load;
# each row retains its full ascending feature chain and all folds/alphas.
comptime AFCL_L03 = (GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
                    and is_defined["MOJOLEARN_AFCL_L03"]())
comptime EF_TPB = 256
comptime EF_CH = 8192
"""Rows per chunk; a chunk never straddles a fold."""
comptime EF_TS = 32
"""Gram tile edge."""
comptime EF_RB = 32
"""Rows per shared-memory step of a Gram tile."""


def _nb(count: Int) -> Int:
    return max((count + EF_TPB - 1) // EF_TPB, 1)


@always_inline
def _aug(x: FP, y: FP, d: Int, i: Int, c: Int) -> Float32:
    """[X, y][i, c]."""
    if c < d:
        return ld(x, i * d + c)
    return ld(y, i)


@always_inline
def _pair(p: Int, nt: Int) -> Tuple[Int, Int]:
    """The p-th upper tile pair (tj <= tk), row-major."""
    var q = p
    var tj = 0
    while q >= nt - tj:
        q -= nt - tj
        tj += 1
    return (tj, tj + q)


# meta (int32): bnd F+1 | cf F+1 (each fold's first chunk) | ct 3 per chunk (start, count, fold)

def ef_check_kernel(y: FP, n: Int32, f_n: Int32, meta: IP, flag: IP, wf: IP, woff: Int32, nonce: Int32):
    """Every row's fold id (y[n + i]) against the KFold bounds."""
    var i = Int(block_idx.x) * EF_TPB + Int(thread_idx.x)
    var nn = Int(n)
    if i < nn:
        var f = 0
        while f < Int(f_n) - 1 and i >= ldi(meta, f + 1):
            f += 1
        if Int(ld(y, nn + i)) != f:
            sti(flag, 0, 1)
    witness_end(wf, woff, nonce)


def ef_sums_kernel(x: FP, y: FP, d_in: Int32, f_n: Int32, meta: IP, part_s: FP, wf: IP, woff: Int32,
                   nonce: Int32):
    """Chunk block_idx.x: the column sums of [X, y] over its rows."""
    var d = Int(d_in)
    var m = d + 1
    var ch = Int(block_idx.x)
    var co = 2 * (Int(f_n) + 1) + 3 * ch
    var lo = ldi(meta, co)
    var cnt = ldi(meta, co + 1)
    var tid = Int(thread_idx.x)
    var sh = stack_allocation[EF_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    if m <= EF_TPB:
        var g_n = EF_TPB // m
        var g = tid // m
        var c = tid - g * m
        var acc = Float32(0)
        if g < g_n:
            var r = g
            while r < cnt:
                acc += _aug(x, y, d, lo + r, c)
                r += g_n
        sh[tid] = acc
        barrier()
        if tid < m:
            var s = Float32(0)
            for gg in range(g_n):
                s += sh[gg * m + tid]
            st(part_s, ch * m + tid, s)
    else:
        for c in range(tid, m, EF_TPB):
            var acc = Float32(0)
            for r in range(cnt):
                acc += _aug(x, y, d, lo + r, c)
            st(part_s, ch * m + c, acc)
    witness_end(wf, woff, nonce)


def ef_means_kernel(part_s: FP, m_in: Int32, f_n: Int32, meta: IP, mu: FP, wf: IP, woff: Int32, nonce: Int32):
    """Thread (f, c): fold f's mean of column c."""
    var m = Int(m_in)
    var nfo = Int(f_n)
    var t = Int(block_idx.x) * EF_TPB + Int(thread_idx.x)
    if t < nfo * m:
        var f = t // m
        var c = t - f * m
        var s = Float32(0)
        for ch in range(ldi(meta, nfo + 1 + f), ldi(meta, nfo + 1 + f + 1)):
            s += ld(part_s, ch * m + c)
        st(mu, t, s / Float32(ldi(meta, f + 1) - ldi(meta, f)))
    witness_end(wf, woff, nonce)


def ef_gram_kernel(x: FP, y: FP, d_in: Int32, f_n: Int32, meta: IP, mu: FP, part_g: FP, npairs_in: Int32,
                   nt_in: Int32, wf: IP, woff: Int32, nonce: Int32):
    """Block b: tile pair b % npairs of chunk b // npairs, the chunk's rows
    of [X, y] centered at its fold's means; 4 cells a thread."""
    var d = Int(d_in)
    var m = d + 1
    var npairs = Int(npairs_in)
    var b = Int(block_idx.x)
    var p = b % npairs
    var ch = b // npairs
    var tjk = _pair(p, Int(nt_in))
    var j0 = tjk[0] * EF_TS
    var k0 = tjk[1] * EF_TS
    var co = 2 * (Int(f_n) + 1) + 3 * ch
    var lo = ldi(meta, co)
    var cnt = ldi(meta, co + 1)
    var f = ldi(meta, co + 2)
    var tid = Int(thread_idx.x)
    var sa = stack_allocation[EF_RB * EF_TS, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var sb = stack_allocation[EF_RB * EF_TS, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var r = tid // 8
    var c0 = (tid % 8) * 4
    var acc = SIMD[DType.float32, 4](0)
    var rb = 0
    while rb < cnt:
        comptime for u in range((EF_RB * EF_TS) // EF_TPB):
            var e = tid + u * EF_TPB
            var rr = e // EF_TS
            var cc = e - rr * EF_TS
            var row = rb + rr
            var va = Float32(0)
            var vb = Float32(0)
            if row < cnt:
                var ja = j0 + cc
                var kb = k0 + cc
                if ja < m:
                    va = _aug(x, y, d, lo + row, ja) - ld(mu, f * m + ja)
                if kb < m:
                    vb = _aug(x, y, d, lo + row, kb) - ld(mu, f * m + kb)
            sa[e] = va
            sb[e] = vb
        barrier()
        comptime for rr in range(EF_RB):
            var a = sa[rr * EF_TS + r]
            var bv = (sb + rr * EF_TS + c0).load[width=4]()
            acc += a * bv
        barrier()
        rb += EF_RB
    var o = (ch * npairs + p) * (EF_TS * EF_TS) + r * EF_TS + c0
    comptime for e in range(4):
        st(part_g, o + e, acc[e])
    witness_end(wf, woff, nonce)


def ef_gram_red_kernel(part_g: FP, m_in: Int32, f_n: Int32, meta: IP, npairs_in: Int32, nt_in: Int32, cg: FP,
                       wf: IP, woff: Int32, nonce: Int32):
    """Thread (f, pair, cell): fold f's centered Gram cell, both triangles."""
    var m = Int(m_in)
    var nfo = Int(f_n)
    var npairs = Int(npairs_in)
    comptime TT = EF_TS * EF_TS
    var t = Int(block_idx.x) * EF_TPB + Int(thread_idx.x)
    if t < nfo * npairs * TT:
        var f = t // (npairs * TT)
        var rest = t - f * npairs * TT
        var p = rest // TT
        var cell = rest - p * TT
        var tjk = _pair(p, Int(nt_in))
        var j = tjk[0] * EF_TS + cell // EF_TS
        var k = tjk[1] * EF_TS + cell % EF_TS
        if j < m and k < m:
            var s = Float32(0)
            for ch in range(ldi(meta, nfo + 1 + f), ldi(meta, nfo + 1 + f + 1)):
                s += ld(part_g, (ch * npairs + p) * TT + cell)
            st(cg, f * m * m + j * m + k, s)
            st(cg, f * m * m + k * m + j, s)
    witness_end(wf, woff, nonce)


@always_inline
def _ws_cells(d: Int) -> Int:
    """A set's workspace: xm d | G d*d | q d | Qw d | w d | ym, yn, rows."""
    return d * d + 4 * d + 3


def ef_sets_kernel(cg: FP, mu: FP, d_in: Int32, f_n: Int32, l_n: Int32, fi: Int32, meta: IP, ws: FP, wf: IP,
                   woff: Int32, nonce: Int32):
    """Thread (s, j, k): set s (all rows but fold s; s == F all rows),
    cell (j, k) of its Gram of [X, y] at its own means, into every l copy."""
    var d = Int(d_in)
    var m = d + 1
    var nfo = Int(f_n)
    var t = Int(block_idx.x) * EF_TPB + Int(thread_idx.x)
    if t < (nfo + 1) * m * m:
        var s = t // (m * m)
        var jk = t - s * m * m
        var j = jk // m
        var k = jk - j * m
        var ns = 0
        for g in range(nfo):
            if g != s:
                ns += ldi(meta, g + 1) - ldi(meta, g)
        var mj = Float32(0)
        var mk = Float32(0)
        if fi != 0:
            for g in range(nfo):
                if g != s:
                    var ng = Float32(ldi(meta, g + 1) - ldi(meta, g))
                    mj += ng * ld(mu, g * m + j)
                    mk += ng * ld(mu, g * m + k)
            mj = mj / Float32(ns)
            mk = mk / Float32(ns)
        var v = Float32(0)
        for g in range(nfo):
            if g != s:
                var ng = Float32(ldi(meta, g + 1) - ldi(meta, g))
                var dj = ld(mu, g * m + j) - mj
                var dk = ld(mu, g * m + k) - mk
                v += ld(cg, g * m * m + j * m + k) + ng * dj * dk
        var wsc = _ws_cells(d)
        for l in range(Int(l_n)):
            var base = (s * Int(l_n) + l) * wsc
            if j < d and k < d:
                st(ws, base + d + j * d + k, v)
            elif j < d and k == d:
                st(ws, base + d + d * d + j, v)
            elif j == d and k == d:
                st(ws, base + wsc - 2, v)
            if k == 0:
                if j < d:
                    st(ws, base + j, mj)
                    st(ws, base + d + d * d + d + j, Float32(0))
                    st(ws, base + d + d * d + 2 * d + j, Float32(0))
                else:
                    st(ws, base + wsc - 3, mj)
                    st(ws, base + wsc - 1, Float32(ns))
    witness_end(wf, woff, nonce)


@always_inline
def _gap_block(fw: FP, q: Int, qw: Int, w: Int, d: Int, ynorm2: Float32, l1: Float32, l2: Float32,
               positive: Bool) -> Float32:
    """x_linear/cd.mojo `_gap` with its sums over d as block folds; every
    thread returns the same value. Reads w and Qw after a device barrier."""
    var tid = Int(thread_idx.x)
    var sh = stack_allocation[4 * EF_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var wl2 = Float32(0)
    var qdw = Float32(0)
    var wqw = Float32(0)
    var wl1 = Float32(0)
    for j in range(tid, d, EF_TPB):
        var wj = ld(fw, w + j)
        wl2 = fmad(wj, wj, wl2)
        qdw = fmad(wj, ld(fw, q + j), qdw)
        wqw = fmad(wj, ld(fw, qw + j), wqw)
        wl1 = fa(wl1, fabs(wj))
    sh[tid] = wl2
    sh[EF_TPB + tid] = qdw
    sh[2 * EF_TPB + tid] = wqw
    sh[3 * EF_TPB + tid] = wl1
    barrier()
    var h = EF_TPB // 2
    while h > 0:
        if tid < h:
            comptime for r in range(4):
                sh[r * EF_TPB + tid] = fa(sh[r * EF_TPB + tid], sh[r * EF_TPB + tid + h])
        barrier()
        h //= 2
    wl2 = sh[0]
    qdw = sh[EF_TPB]
    wqw = sh[2 * EF_TPB]
    wl1 = sh[3 * EF_TPB]
    barrier()
    var r2 = fs(fa(ynorm2, wqw), fm(Float32(2), qdw))
    var ry = fs(ynorm2, qdw)
    var dn = Float32(0)
    for j in range(tid, d, EF_TPB):
        if l1 == 0:
            var a = fs(ld(fw, q + j), ld(fw, qw + j))
            dn = fmad(a, a, dn)
        else:
            var a = fs(fs(ld(fw, q + j), ld(fw, qw + j)), fm(l2, ld(fw, w + j)))
            dn = fmax(dn, a if positive else fabs(a))
    sh[tid] = dn
    barrier()
    h = EF_TPB // 2
    while h > 0:
        if tid < h:
            sh[tid] = fa(sh[tid], sh[tid + h]) if l1 == 0 else fmax(sh[tid], sh[tid + h])
        barrier()
        h //= 2
    var dual_norm = sh[0]
    barrier()
    if l1 == 0:
        if l2 == 0:
            return dual_norm
        var g = fs(fa(r2, fm(fm(Float32(0.5), l2), wl2)), ry)
        return fa(g, fm(fd(Float32(1), fm(Float32(2), l2)), dual_norm))
    var base = fa(r2, fm(l2, wl2))
    var primal = fa(fm(Float32(0.5), base), fm(l1, wl1))
    var scale = fd(l1, dual_norm) if dual_norm > l1 else Float32(1)
    var dual = fa(fm(fm(Float32(-0.5), fm(scale, scale)), base), fm(scale, ry))
    return fs(primal, dual)


@always_inline
def _cd_block(fw: FP, gg: Int, q: Int, qw: Int, w: Int, d: Int, ynorm2: Float32, l1: Float32, l2: Float32,
              max_iter: Int, tol: Float32, positive: Bool) -> Int:
    """x_linear/cd.mojo `enet_gram_cd` on one block: Qw = G w a thread per
    row, each coordinate's Qw update a thread per column, the gap's sums as
    block folds. The coordinate order, every Qw element's fmad chain and so
    w are the team routine's; only the gap's fold order differs. Every
    thread returns the sweeps run."""
    var tid = Int(thread_idx.x)
    var sd = stack_allocation[2, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    for j in range(tid, d, EF_TPB):
        var acc = Float32(0)
        for k in range(d):
            acc = fmad(ld(fw, gg + j * d + k), ld(fw, w + k), acc)
        st(fw, qw + j, acc)
    team_barrier()
    var tol_s = fm(tol, ynorm2)
    if _gap_block(fw, q, qw, w, d, ynorm2, l1, l2, positive) <= tol_s:
        return 0
    var w_max = Float32(0)
    var dw_max = Float32(0)
    for it in range(max_iter):
        w_max = Float32(0)
        dw_max = Float32(0)
        for j in range(d):
            var qjj = ld(fw, gg + j * d + j)
            if qjj == 0:
                continue
            if tid == 0:
                var wj = ld(fw, w + j)
                var t = fa(fs(ld(fw, q + j), ld(fw, qw + j)), fm(wj, qjj))
                var nw = fd(fm(fsign(t), fmax(fs(fabs(t), l1), Float32(0))), fa(qjj, l2))
                if positive and t < 0:
                    nw = Float32(0)
                st(fw, w + j, nw)
                sd[0] = fs(nw, wj)
                sd[1] = Float32(1) if nw != wj else Float32(0)
                var dw = fabs(fs(nw, wj))
                if dw > dw_max:
                    dw_max = dw
                if fabs(nw) > w_max:
                    w_max = fabs(nw)
            team_barrier()
            if sd[1] != 0:
                var delta = sd[0]
                for k in range(tid, d, EF_TPB):
                    st(fw, qw + k, fmad(delta, ld(fw, gg + j * d + k), ld(fw, qw + k)))
            team_barrier()
        if tid == 0:
            sd[1] = Float32(1) if (w_max == 0 or fd(dw_max, w_max) <= tol or it == max_iter - 1) else Float32(0)
        barrier()
        var check = sd[1] != 0
        barrier()
        if check:
            if _gap_block(fw, q, qw, w, d, ynorm2, l1, l2, positive) <= tol_s:
                return it + 1
    return max_iter


def ef_alphas_kernel(ws: FP, fp: FP, res: FP, d_in: Int32, n: Int32, f_n: Int32, l_n: Int32, a_n: Int32,
                     explicit: Int32, wf: IP, woff: Int32, nonce: Int32):
    """Block l: l1_ratio l's grid (x_linear/cd.mojo `enetcv_fit`'s, from all
    rows' X'y); max |X'y| a block fold, the grid a thread per alpha."""
    var d = Int(d_in)
    var ln = Int(l_n)
    var an = Int(a_n)
    var alphas = d + 4
    var l = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var sh = stack_allocation[EF_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var base = (Int(f_n) * ln) * _ws_cells(d)
    var q = base + d + d * d
    var eps = ld(fp, 0)
    var l1r = ld(fp, 2 + l)
    var qmax = Float32(0)
    for j in range(tid, d, EF_TPB):
        qmax = fmax(qmax, fabs(ld(ws, q + j)))
    sh[tid] = qmax
    barrier()
    var h = EF_TPB // 2
    while h > 0:
        if tid < h:
            sh[tid] = fmax(sh[tid], sh[tid + h])
        barrier()
        h //= 2
    qmax = sh[0]
    var amax = fd(qmax, fm(i2f(Int(n)), l1r))
    for k in range(tid, an, EF_TPB):
        var v: Float32
        if explicit != 0:
            v = ld(fp, 2 + ln + k)
        elif amax <= Float32(1e-6):
            v = Float32(1e-6)
        else:
            v = alpha_grid_value(amax, eps, k, an)
        st(res, alphas + l * an + k, v)
    witness_end(wf, woff, nonce)


def ef_path_kernel(ws: FP, fp: FP, res: FP, path: FP, d_in: Int32, l_n: Int32, a_n: Int32, max_iter: Int32,
                   positive: Int32, wf: IP, woff: Int32, nonce: Int32):
    """Block (f, l): fold f's warm-started path for l1_ratio l, the block's
    threads sharing each coordinate descent; each alpha's (w, b) into `path`."""
    var d = Int(d_in)
    var ln = Int(l_n)
    var an = Int(a_n)
    var b_id = Int(block_idx.x)
    var l = b_id % ln
    var tid = Int(thread_idx.x)
    var wsc = _ws_cells(d)
    var fw = ws + b_id * wsc
    var gg = d
    var q = gg + d * d
    var qw = q + d
    var w = qw + d
    var ym = ld(fw, wsc - 3)
    var yn = ld(fw, wsc - 2)
    var rows = ld(fw, wsc - 1)
    var l1r = ld(fp, 2 + l)
    var tol = ld(fp, 1)
    for k in range(an):
        var alpha = ld(res, d + 4 + l * an + k)
        var l1 = fm(fm(alpha, l1r), rows)
        var l2 = fm(fm(alpha, fs(Float32(1), l1r)), rows)
        _ = _cd_block(fw, gg, q, qw, w, d, yn, l1, l2, Int(max_iter), tol, positive != 0)
        var o = (b_id * an + k) * (d + 1)
        for j in range(tid, d, EF_TPB):
            st(path, o + j, ld(fw, w + j))
        if tid == 0:
            var b = ym
            for j in range(d):
                b = fs(b, fm(ld(fw, j), ld(fw, w + j)))
            st(path, o + d, b)
        team_barrier()
    witness_end(wf, woff, nonce)


def ef_mse_kernel(x: FP, y: FP, d_in: Int32, f_n: Int32, p_n: Int32, meta: IP, path: FP, part_m: FP,
                  wf: IP, woff: Int32, nonce: Int32):
    """Chunk block_idx.x (rows of one fold f, held out): each path p's sum of
    squared errors over the chunk, a block reduction a path."""
    var d = Int(d_in)
    var pn = Int(p_n)
    var ch = Int(block_idx.x)
    var co = 2 * (Int(f_n) + 1) + 3 * ch
    var lo = ldi(meta, co)
    var cnt = ldi(meta, co + 1)
    var f = ldi(meta, co + 2)
    var tid = Int(thread_idx.x)
    var sh = stack_allocation[EF_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    for p in range(pn):
        var o = (f * pn + p) * (d + 1)
        var acc = Float32(0)
        comptime if AFCL_L03:
            for r in range(tid, cnt, 2 * EF_TPB):
                var i = lo + r
                var i2 = i + EF_TPB
                var pr = ld(path, o + d)
                var pr2 = pr
                for j in range(d):
                    var w = ld(path, o + j)
                    pr += ld(x, i * d + j) * w
                    if r + EF_TPB < cnt:
                        pr2 += ld(x, i2 * d + j) * w
                var e = pr - ld(y, i)
                acc += e * e
                if r + EF_TPB < cnt:
                    var e2 = pr2 - ld(y, i2)
                    acc += e2 * e2
        else:
            for r in range(tid, cnt, EF_TPB):
                var i = lo + r
                var pr = ld(path, o + d)
                for j in range(d):
                    pr += ld(x, i * d + j) * ld(path, o + j)
                var e = pr - ld(y, i)
                acc += e * e
        sh[tid] = acc
        barrier()
        var h = EF_TPB // 2
        while h > 0:
            if tid < h:
                sh[tid] = sh[tid] + sh[tid + h]
            barrier()
            h //= 2
        if tid == 0:
            st(part_m, ch * pn + p, sh[0])
        barrier()
    witness_end(wf, woff, nonce)


def ef_mse_red_kernel(part_m: FP, f_n: Int32, p_n: Int32, meta: IP, res: FP, mse_off: Int32, wf: IP,
                      woff: Int32, nonce: Int32):
    """Thread (f, p): mse[p * F + f] = the fold's sum / its rows."""
    var nfo = Int(f_n)
    var pn = Int(p_n)
    var t = Int(block_idx.x) * EF_TPB + Int(thread_idx.x)
    if t < nfo * pn:
        var f = t // pn
        var p = t - f * pn
        var s = Float32(0)
        for ch in range(ldi(meta, nfo + 1 + f), ldi(meta, nfo + 1 + f + 1)):
            s += ld(part_m, ch * pn + p)
        var nte = ldi(meta, f + 1) - ldi(meta, f)
        st(res, Int(mse_off) + p * nfo + f, s / Float32(nte) if nte > 0 else Float32(0))
    witness_end(wf, woff, nonce)


def ef_final_kernel(ws: FP, fp: FP, res: FP, d_in: Int32, f_n: Int32, l_n: Int32, a_n: Int32,
                    max_iter: Int32, positive: Int32, fi: Int32, wf: IP, woff: Int32, nonce: Int32):
    """One block: the choice (the smallest mean over folds, first on a tie; a
    thread per (l, alpha), then a block fold on (mean, index)) and the refit
    on all rows from zero, the block sharing the coordinate descent
    (x_linear/cd.mojo `enetcv_fit`'s tail). The row count is the all-rows
    set's, from its workspace."""
    var d = Int(d_in)
    var nfo = Int(f_n)
    var ln = Int(l_n)
    var an = Int(a_n)
    var tid = Int(thread_idx.x)
    var alphas = d + 4
    var mse = alphas + ln * an
    var pn = ln * an
    var sv = stack_allocation[EF_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var si = stack_allocation[EF_TPB, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var inf = bitcast[DType.float32](UInt32(0x7F800000))
    var best = inf
    var best_p = pn
    for p in range(tid, pn, EF_TPB):
        var acc = Float32(0)
        for f in range(nfo):
            acc = fa(acc, ld(res, mse + p * nfo + f))
        var mv = fd(acc, i2f(nfo))
        if mv != mv:
            mv = inf
        if best_p == pn or mv < best:
            best = mv
            best_p = p
    sv[tid] = best
    si[tid] = Int32(best_p)
    barrier()
    var h = EF_TPB // 2
    while h > 0:
        if tid < h:
            var v2 = sv[tid + h]
            var i2 = si[tid + h]
            if v2 < sv[tid] or (v2 == sv[tid] and i2 < si[tid]):
                sv[tid] = v2
                si[tid] = i2
        barrier()
        h //= 2
    var bp = Int(si[0])
    if bp >= pn:
        bp = 0
    var best_l = bp // an
    var alpha = ld(res, alphas + bp)
    var l1r = ld(fp, 2 + best_l)
    var wsc = _ws_cells(d)
    var fw = ws + (nfo * ln + best_l) * wsc
    var gg = d
    var q = gg + d * d
    var qw = q + d
    var w = qw + d
    var nn = ld(fw, wsc - 1)
    var iters = _cd_block(fw, gg, q, qw, w, d, ld(fw, wsc - 2), fm(fm(alpha, l1r), nn),
                          fm(fm(alpha, fs(Float32(1), l1r)), nn), Int(max_iter), ld(fp, 1), positive != 0)
    for j in range(tid, d, EF_TPB):
        st(res, j, ld(fw, w + j))
    if tid == 0:
        var b = ld(fw, wsc - 3)
        for j in range(d):
            b = fs(b, fm(ld(fw, j), ld(fw, w + j)))
        st(res, d, b if fi != 0 else Float32(0))
        st(res, d + 1, alpha)
        st(res, d + 2, l1r)
        st(res, d + 3, i2f(iters))
    witness_end(wf, woff, nonce)


def enetcv_fast(
    mut ctx: DeviceContext, x: FP, n_x: Int, y: FP, n_y: Int, n: Int, d: Int,
    ip: List[Int32], fp: List[Float32], n_out: Int, res: FP,
) raises -> Bool:
    """The whole fit on the device; False (nothing written that matters) when
    the fold ids are not KFold's contiguous ones, so the caller runs the
    team fit."""
    var max_iter = Int(ip[0])
    var fi = Int(ip[1])
    var a_n = Int(ip[2])
    var f_n = Int(ip[3])
    var l_n = Int(ip[4])
    var explicit = Int(ip[5])
    var positive = Int(ip[6])
    if f_n < 2 or n < f_n or d < 1:
        return False
    var m = d + 1
    # KFold(shuffle=False): contiguous folds, the first n % k one row longer
    var meta = List[Int32]()
    var start = 0
    for f in range(f_n):  # small-loop(f_n: CV folds): fold start offsets for the launch plan, no data read
        meta.append(Int32(start))
        start += n // f_n + (1 if f < n % f_n else 0)
    meta.append(Int32(n))
    var nch = 0
    for f in range(f_n):  # small-loop(f_n: CV folds): chunk counts per fold for the launch plan, no data read
        meta.append(Int32(nch))
        var nf = Int(meta[f + 1]) - Int(meta[f])
        nch += (nf + EF_CH - 1) // EF_CH
    meta.append(Int32(nch))
    for f in range(f_n):  # small-loop(f_n: CV folds): emits the chunk descriptors, one per 8192 rows, plan words only, no data read
        var lo = Int(meta[f])
        var hi = Int(meta[f + 1])
        while lo < hi:
            var cnt = min(EF_CH, hi - lo)
            meta.append(Int32(lo))
            meta.append(Int32(cnt))
            meta.append(Int32(f))
            lo += cnt
    var nt = (m + EF_TS - 1) // EF_TS
    var npairs = nt * (nt + 1) // 2
    var p_n = l_n * a_n
    var wsc = _ws_cells(d)
    var mse_off = d + 4 + l_n * a_n

    var dx = ctx.enqueue_create_buffer[DType.float32](max(n_x, 1))
    var dy = ctx.enqueue_create_buffer[DType.float32](max(n_y, 1))
    var dfp = ctx.enqueue_create_buffer[DType.float32](max(len(fp), 1))
    var dmeta = ctx.enqueue_create_buffer[DType.int32](len(meta))
    var dflag = ctx.enqueue_create_buffer[DType.int32](1)
    var dps = ctx.enqueue_create_buffer[DType.float32](nch * m)
    var dmu = ctx.enqueue_create_buffer[DType.float32](f_n * m)
    var dpg = ctx.enqueue_create_buffer[DType.float32](nch * npairs * EF_TS * EF_TS)
    var dcg = ctx.enqueue_create_buffer[DType.float32](f_n * m * m)
    var dws = ctx.enqueue_create_buffer[DType.float32]((f_n + 1) * l_n * wsc)
    var dpath = ctx.enqueue_create_buffer[DType.float32](max(f_n * p_n * (d + 1), 1))
    var dpm = ctx.enqueue_create_buffer[DType.float32](max(nch * p_n, 1))
    var dres = ctx.enqueue_create_buffer[DType.float32](max(n_out, 1))
    ctx.enqueue_copy(dst_buf=dx, src_ptr=x)
    ctx.enqueue_copy(dst_buf=dy, src_ptr=y)
    var hfp = fp.copy()
    if len(hfp) > 0:
        ctx.enqueue_copy(dst_buf=dfp, src_ptr=hfp.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dmeta, src_ptr=meta.unsafe_ptr())
    var b_chk = _nb(n)
    var b_mu = _nb(f_n * m)
    var b_gram = npairs * nch
    var b_gred = _nb(f_n * npairs * EF_TS * EF_TS)
    var b_sets = _nb((f_n + 1) * m * m)
    var b_mred = _nb(f_n * p_n)
    var total = b_chk + nch + b_mu + b_gram + b_gred + b_sets + l_n + f_n * l_n + nch + b_mred + 1
    var wit = Witness(ctx, total)
    var flag = List[Int32](length=1, fill=Int32(0))
    var tries = 0
    while True:
        var nonce = wit.begin()
        var wf = wit.p()
        var wo = 0
        dflag.enqueue_fill(Int32(0))
        dres.enqueue_fill(Float32(0))
        ctx.enqueue_function[ef_check_kernel](
            dy.unsafe_ptr(), Int32(n), Int32(f_n), dmeta.unsafe_ptr(), dflag.unsafe_ptr(),
            wf, Int32(wo), nonce, grid_dim=b_chk, block_dim=EF_TPB)
        wo += b_chk
        ctx.enqueue_function[ef_sums_kernel](
            dx.unsafe_ptr(), dy.unsafe_ptr(), Int32(d), Int32(f_n), dmeta.unsafe_ptr(), dps.unsafe_ptr(),
            wf, Int32(wo), nonce, grid_dim=nch, block_dim=EF_TPB)
        wo += nch
        ctx.enqueue_function[ef_means_kernel](
            dps.unsafe_ptr(), Int32(m), Int32(f_n), dmeta.unsafe_ptr(), dmu.unsafe_ptr(),
            wf, Int32(wo), nonce, grid_dim=b_mu, block_dim=EF_TPB)
        wo += b_mu
        ctx.enqueue_function[ef_gram_kernel](
            dx.unsafe_ptr(), dy.unsafe_ptr(), Int32(d), Int32(f_n), dmeta.unsafe_ptr(), dmu.unsafe_ptr(),
            dpg.unsafe_ptr(), Int32(npairs), Int32(nt), wf, Int32(wo), nonce, grid_dim=b_gram, block_dim=EF_TPB)
        wo += b_gram
        ctx.enqueue_function[ef_gram_red_kernel](
            dpg.unsafe_ptr(), Int32(m), Int32(f_n), dmeta.unsafe_ptr(), Int32(npairs), Int32(nt), dcg.unsafe_ptr(),
            wf, Int32(wo), nonce, grid_dim=b_gred, block_dim=EF_TPB)
        wo += b_gred
        ctx.enqueue_function[ef_sets_kernel](
            dcg.unsafe_ptr(), dmu.unsafe_ptr(), Int32(d), Int32(f_n), Int32(l_n), Int32(fi), dmeta.unsafe_ptr(),
            dws.unsafe_ptr(), wf, Int32(wo), nonce, grid_dim=b_sets, block_dim=EF_TPB)
        wo += b_sets
        ctx.enqueue_function[ef_alphas_kernel](
            dws.unsafe_ptr(), dfp.unsafe_ptr(), dres.unsafe_ptr(), Int32(d), Int32(n), Int32(f_n), Int32(l_n),
            Int32(a_n), Int32(explicit), wf, Int32(wo), nonce, grid_dim=l_n, block_dim=EF_TPB)
        wo += l_n
        ctx.enqueue_function[ef_path_kernel](
            dws.unsafe_ptr(), dfp.unsafe_ptr(), dres.unsafe_ptr(), dpath.unsafe_ptr(), Int32(d), Int32(l_n),
            Int32(a_n), Int32(max_iter), Int32(positive), wf, Int32(wo), nonce, grid_dim=f_n * l_n, block_dim=EF_TPB)
        wo += f_n * l_n
        ctx.enqueue_function[ef_mse_kernel](
            dx.unsafe_ptr(), dy.unsafe_ptr(), Int32(d), Int32(f_n), Int32(p_n), dmeta.unsafe_ptr(),
            dpath.unsafe_ptr(), dpm.unsafe_ptr(), wf, Int32(wo), nonce, grid_dim=nch, block_dim=EF_TPB)
        wo += nch
        ctx.enqueue_function[ef_mse_red_kernel](
            dpm.unsafe_ptr(), Int32(f_n), Int32(p_n), dmeta.unsafe_ptr(), dres.unsafe_ptr(), Int32(mse_off),
            wf, Int32(wo), nonce, grid_dim=b_mred, block_dim=EF_TPB)
        wo += b_mred
        ctx.enqueue_function[ef_final_kernel](
            dws.unsafe_ptr(), dfp.unsafe_ptr(), dres.unsafe_ptr(), Int32(d), Int32(f_n), Int32(l_n),
            Int32(a_n), Int32(max_iter), Int32(positive), Int32(fi), wf, Int32(wo), nonce, grid_dim=1,
            block_dim=EF_TPB)
        wo += 1
        if n_out > 0:
            ctx.enqueue_copy(dst_ptr=res, src_buf=dres)
        ctx.enqueue_copy(dst_ptr=flag.unsafe_ptr(), src_buf=dflag)
        if wit.ok(ctx, total, "LassoCV/ElasticNetCV"):
            break
        tries += 1
        if tries >= WITNESS_TRIES:
            raise Error(WITNESS_ABORT)
    ctx.synchronize()
    var good = flag[0] == 0
    _ = meta^
    _ = hfp^
    _ = flag^
    _ = dx^
    _ = dy^
    _ = dfp^
    _ = dmeta^
    _ = dflag^
    _ = dps^
    _ = dmu^
    _ = dpg^
    _ = dcg^
    _ = dws^
    _ = dpath^
    _ = dpm^
    _ = dres^
    return good
