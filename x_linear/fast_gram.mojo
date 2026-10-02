# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The shared FAST grid Gram (lane/apple-fast-gram, 2026-10-02).

A d x d Gram of X, centered or not, plus X'Y for T targets, over a row
range, on the whole GPU: no one-block step, no host step. Generalised from
x_linear/enetcv_fast.mojo's `ef_sums_kernel` / `ef_gram_kernel` /
`ef_gram_red_kernel` (one fold = the whole range, so no parallel-axis move):

  1. `fg_sums_kernel`: the column sums of [X, Y] per chunk of FG_CH rows
     (one block a chunk);
  2. `fg_means_kernel`: the means (zeros without an intercept), a thread a
     column, into the caller's xm / ym words and the kernel's own mu;
  3. `fg_gram_kernel`: per (chunk, 32 x 32 tile pair) block, the centered
     tile product over the chunk's rows through shared memory, 4 cells a
     thread, into per-chunk partials; a class mask (`cls` >= 0: rows whose
     label is not `cls` contribute nothing) serves per-class covariances;
  4. `fg_red_kernel`: a thread a cell, the sum over the chunks, into G
     (both triangles) and X'Y (`xty[t * d + j]`); `fg_red_sym_kernel` is the
     same sum into one symmetric m x m block with an optional divisor
     (x_prep's class covariances and Grams).

The one-block forms these replace: `t_centered_gram` / `t_centered_xty`
(x_linear/tops.mojo: one thread per Gram cell walking every row on ONE
block), `xg_gram_kernel` (x_linear/device.mojo: the same chains on a grid
of d(d+1)/2 threads, one block at taxi's 16 features), `kf_cells_kernel`
(the k-fold RidgeCV Grams, likewise), and x_prep's `qda_cov` and the LDA
Gram `matmul` (one thread per cell walking every row). FAST promises
quality, not bits: the chunked partials are pairwise sums, never less
accurate than the serial chains. FAST only; IDENTICAL never compiles a
call to this file.
"""
from std.gpu import block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from x_linear.ops import FP, IP, ld, st

comptime FG_TPB = 256
comptime FG_CH = 8192
"""Rows per chunk when the Gram has 8 or more tile pairs."""
comptime FG_CH_SMALL = 2048
"""Rows per chunk below that (taxi's 16 features: one tile pair), so the
grid still holds hundreds of blocks."""
comptime FG_TS = 32
"""Gram tile edge."""
comptime FG_RB = 32
"""Rows per shared-memory step of a Gram tile."""
comptime FG_TT = FG_TS * FG_TS


@always_inline
def fg_blocks(count: Int) -> Int:
    return max((count + FG_TPB - 1) // FG_TPB, 1)


@always_inline
def fg_tiles(m: Int) -> Int:
    return (m + FG_TS - 1) // FG_TS


@always_inline
def fg_pairs(m: Int) -> Int:
    var nt = fg_tiles(m)
    return nt * (nt + 1) // 2


@always_inline
def fg_chunk_rows(m: Int) -> Int:
    return FG_CH if fg_pairs(m) >= 8 else FG_CH_SMALL


@always_inline
def fg_chunks(cnt: Int, m: Int) -> Int:
    var ch = fg_chunk_rows(m)
    return max((cnt + ch - 1) // ch, 1)


@always_inline
def fg_part_words(cnt: Int, m: Int) -> Int:
    """Words of per-chunk Gram partials a range of cnt rows needs."""
    return fg_chunks(cnt, m) * fg_pairs(m) * FG_TT


@always_inline
def _aug(x: FP, y: FP, d: Int, t_n: Int, i: Int, c: Int) -> Float32:
    """[X, Y][i, c] (Y row-major, t_n columns)."""
    if c < d:
        return ld(x, i * d + c)
    return ld(y, i * t_n + (c - d))


@always_inline
def _pair(p: Int, nt: Int) -> Tuple[Int, Int]:
    """The p-th upper tile pair (tj <= tk), row-major."""
    var q = p
    var tj = 0
    while q >= nt - tj:
        q -= nt - tj
        tj += 1
    return (tj, tj + q)


def fg_sums_kernel(x: FP, y: FP, d_in: Int32, t_in: Int32, lo_in: Int32, cnt_in: Int32, crows_in: Int32,
                   part_s: FP):
    """Chunk block_idx.x: the column sums of [X, Y] over its rows."""
    var d = Int(d_in)
    var t_n = Int(t_in)
    var m = d + t_n
    var ch = Int(block_idx.x)
    var crows = Int(crows_in)
    var lo = Int(lo_in) + ch * crows
    var cnt = min(crows, Int(lo_in) + Int(cnt_in) - lo)
    var tid = Int(thread_idx.x)
    var sh = stack_allocation[FG_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    if m <= FG_TPB:
        var g_n = FG_TPB // m
        var g = tid // m
        var c = tid - g * m
        var acc = Float32(0)
        if g < g_n:
            var r = g
            while r < cnt:
                acc += _aug(x, y, d, t_n, lo + r, c)
                r += g_n
        sh[tid] = acc
        barrier()
        if tid < m:
            var s = Float32(0)
            for gg in range(g_n):
                s += sh[gg * m + tid]
            st(part_s, ch * m + tid, s)
    else:
        for c in range(tid, m, FG_TPB):
            var acc = Float32(0)
            for r in range(cnt):
                acc += _aug(x, y, d, t_n, lo + r, c)
            st(part_s, ch * m + c, acc)


def fg_means_kernel(part_s: FP, nch_in: Int32, d_in: Int32, t_in: Int32, cnt_in: Int32, fi: Int32,
                    xm: FP, ym: FP, mu: FP):
    """Thread c: column c's mean over the range (zero without an intercept),
    into mu[c] and the caller's xm (c < d) or ym (c - d)."""
    var d = Int(d_in)
    var m = d + Int(t_in)
    var c = Int(block_idx.x) * FG_TPB + Int(thread_idx.x)
    if c < m:
        var s = Float32(0)
        if fi != 0:
            for ch in range(Int(nch_in)):
                s += ld(part_s, ch * m + c)
            s = s / Float32(Int(cnt_in))
        st(mu, c, s)
        if c < d:
            st(xm, c, s)
        else:
            st(ym, c - d, s)


def fg_gram_kernel(x: FP, y: FP, d_in: Int32, t_in: Int32, lo_in: Int32, cnt_in: Int32, crows_in: Int32,
                   mu: FP, use_mu: Int32, lab: FP, cls: Int32, part_g: FP, npairs_in: Int32, nt_in: Int32):
    """Block b: tile pair b % npairs of chunk b // npairs, the chunk's rows
    of [X, Y] centered at mu (use_mu != 0), rows whose label lab[i] is not
    cls skipped (cls >= 0); 4 cells a thread."""
    var d = Int(d_in)
    var t_n = Int(t_in)
    var m = d + t_n
    var npairs = Int(npairs_in)
    var b = Int(block_idx.x)
    var p = b % npairs
    var ch = b // npairs
    var tjk = _pair(p, Int(nt_in))
    var j0 = tjk[0] * FG_TS
    var k0 = tjk[1] * FG_TS
    var crows = Int(crows_in)
    var lo = Int(lo_in) + ch * crows
    var cnt = min(crows, Int(lo_in) + Int(cnt_in) - lo)
    var masked = Int(cls) >= 0
    var centered = use_mu != 0
    var tid = Int(thread_idx.x)
    var sa = stack_allocation[FG_RB * FG_TS, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var sb = stack_allocation[FG_RB * FG_TS, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var r = tid // 8
    var c0 = (tid % 8) * 4
    var acc = SIMD[DType.float32, 4](0)
    var rb = 0
    while rb < cnt:
        comptime for u in range((FG_RB * FG_TS) // FG_TPB):
            var e = tid + u * FG_TPB
            var rr = e // FG_TS
            var cc = e - rr * FG_TS
            var row = rb + rr
            var va = Float32(0)
            var vb = Float32(0)
            if row < cnt:
                var keep = True
                if masked:
                    keep = Int(ld(lab, lo + row)) == Int(cls)
                if keep:
                    var ja = j0 + cc
                    var kb = k0 + cc
                    if ja < m:
                        va = _aug(x, y, d, t_n, lo + row, ja)
                        if centered:
                            va = va - ld(mu, ja)
                    if kb < m:
                        vb = _aug(x, y, d, t_n, lo + row, kb)
                        if centered:
                            vb = vb - ld(mu, kb)
            sa[e] = va
            sb[e] = vb
        barrier()
        comptime for rr in range(FG_RB):
            var a = sa[rr * FG_TS + r]
            var bv = (sb + rr * FG_TS + c0).load[width=4]()
            acc += a * bv
        barrier()
        rb += FG_RB
    var o = (ch * npairs + p) * FG_TT + r * FG_TS + c0
    comptime for e in range(4):
        st(part_g, o + e, acc[e])


def fg_red_kernel(part_g: FP, nch_in: Int32, d_in: Int32, t_in: Int32, npairs_in: Int32, nt_in: Int32,
                  g: FP, xty: FP):
    """Thread (pair, cell): the cell's sum over the chunks into G (both
    triangles, j, k < d) or X'Y (xty[t * d + j], k = d + t); Y'Y dropped."""
    var d = Int(d_in)
    var m = d + Int(t_in)
    var npairs = Int(npairs_in)
    var t = Int(block_idx.x) * FG_TPB + Int(thread_idx.x)
    if t < npairs * FG_TT:
        var p = t // FG_TT
        var cell = t - p * FG_TT
        var tjk = _pair(p, Int(nt_in))
        var j = tjk[0] * FG_TS + cell // FG_TS
        var k = tjk[1] * FG_TS + cell % FG_TS
        if j < m and k < m:
            var s = Float32(0)
            for ch in range(Int(nch_in)):
                s += ld(part_g, (ch * npairs + p) * FG_TT + cell)
            if j < d and k < d:
                st(g, j * d + k, s)
                st(g, k * d + j, s)
            elif j < d:
                st(xty, (k - d) * d + j, s)
            elif k < d:
                st(xty, (j - d) * d + k, s)


def fg_red_sym_kernel(part_g: FP, nch_in: Int32, m_in: Int32, npairs_in: Int32, nt_in: Int32, out: FP,
                      div: FP, use_div: Int32):
    """Thread (pair, cell): the cell's sum over the chunks, divided by
    div[0] when use_div != 0, into out[j * m + k] and out[k * m + j]."""
    var m = Int(m_in)
    var npairs = Int(npairs_in)
    var t = Int(block_idx.x) * FG_TPB + Int(thread_idx.x)
    if t < npairs * FG_TT:
        var p = t // FG_TT
        var cell = t - p * FG_TT
        var tjk = _pair(p, Int(nt_in))
        var j = tjk[0] * FG_TS + cell // FG_TS
        var k = tjk[1] * FG_TS + cell % FG_TS
        if j < m and k < m:
            var s = Float32(0)
            for ch in range(Int(nch_in)):
                s += ld(part_g, (ch * npairs + p) * FG_TT + cell)
            if use_div != 0:
                s = s / ld(div, 0)
            st(out, j * m + k, s)
            st(out, k * m + j, s)


def fast_gram_into(
    mut ctx: DeviceContext, x: FP, y: FP, lo: Int, cnt: Int, d: Int, t_n: Int, fi: Bool,
    xm: FP, ym: FP, g: FP, xty: FP,
) raises:
    """Rows [lo, lo + cnt) of X (n x d, device) and Y (n x t_n, device): the
    means into xm (d) and ym (t_n) (zeros without an intercept), the Gram
    centered at them into g (d x d) and X'Y into xty (t_n x d), all device
    words. Waits for the launches, so its scratch dies here."""
    var m = d + t_n
    var nt = fg_tiles(m)
    var npairs = fg_pairs(m)
    var crows = fg_chunk_rows(m)
    var nch = fg_chunks(cnt, m)
    var dps = ctx.enqueue_create_buffer[DType.float32](nch * m)
    var dmu = ctx.enqueue_create_buffer[DType.float32](m)
    var dpg = ctx.enqueue_create_buffer[DType.float32](nch * npairs * FG_TT)
    ctx.enqueue_function[fg_sums_kernel](
        x, y, Int32(d), Int32(t_n), Int32(lo), Int32(cnt), Int32(crows), dps.unsafe_ptr(),
        grid_dim=nch, block_dim=FG_TPB)
    ctx.enqueue_function[fg_means_kernel](
        dps.unsafe_ptr(), Int32(nch), Int32(d), Int32(t_n), Int32(cnt), Int32(1 if fi else 0),
        xm, ym, dmu.unsafe_ptr(), grid_dim=fg_blocks(m), block_dim=FG_TPB)
    ctx.enqueue_function[fg_gram_kernel](
        x, y, Int32(d), Int32(t_n), Int32(lo), Int32(cnt), Int32(crows), dmu.unsafe_ptr(), Int32(1),
        x, Int32(-1), dpg.unsafe_ptr(), Int32(npairs), Int32(nt), grid_dim=nch * npairs, block_dim=FG_TPB)
    ctx.enqueue_function[fg_red_kernel](
        dpg.unsafe_ptr(), Int32(nch), Int32(d), Int32(t_n), Int32(npairs), Int32(nt), g, xty,
        grid_dim=fg_blocks(npairs * FG_TT), block_dim=FG_TPB)
    ctx.synchronize()
    _ = dps^
    _ = dmu^
    _ = dpg^


def fast_sym_gram_into(
    mut ctx: DeviceContext, x: FP, lo: Int, cnt: Int, d: Int, mu: FP, use_mu: Bool, lab: FP, cls: Int,
    part_g: FP, out: FP, div: FP, use_div: Bool,
) raises:
    """Rows [lo, lo + cnt) of X (n x d, device): sum_i (x_i - mu)(x_i - mu)'
    over the rows whose label lab[i] is cls (every row when cls < 0),
    uncentered when not use_mu, divided by div[0] when use_div, into out
    (d x d, both triangles). part_g: caller's scratch of at least
    `fg_part_words(cnt, d)` words. Enqueues only (no wait)."""
    var nt = fg_tiles(d)
    var npairs = fg_pairs(d)
    var crows = fg_chunk_rows(d)
    var nch = fg_chunks(cnt, d)
    ctx.enqueue_function[fg_gram_kernel](
        x, x, Int32(d), Int32(0), Int32(lo), Int32(cnt), Int32(crows), mu, Int32(1 if use_mu else 0),
        lab, Int32(cls), part_g, Int32(npairs), Int32(nt), grid_dim=nch * npairs, block_dim=FG_TPB)
    ctx.enqueue_function[fg_red_sym_kernel](
        part_g, Int32(nch), Int32(d), Int32(npairs), Int32(nt), out, div, Int32(1 if use_div else 0),
        grid_dim=fg_blocks(npairs * FG_TT), block_dim=FG_TPB)
