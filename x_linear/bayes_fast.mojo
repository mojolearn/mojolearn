# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""BayesianRidge / ARDRegression row statistics on the whole grid
(lane/apple-fast-kernel, 2026-10-02; FAST on Apple only, behind
`MOJOLEARN_KERNEL_FAST_BAYES_STATS=1`, read in x_linear/device.mojo).

Cause. The fit's row passes were (x_linear/device.mojo `xg_means_kernel`,
`xg_gram_kernel`; x_linear/bayes.mojo `t_col_means`, `t_mean`,
`t_centered_xty`, `_var`): one thread per column or per Gram cell, each a
SERIAL float32 chain over all n rows (`chain_cfmad`, one accumulator), the
means and X'y on ONE block of 256 threads, the target mean and variance on
the lead thread alone. At Istella (1M rows x 220 features) that is 24,310
chains of a million dependent fmads for the Gram and 220 strided chains of a
million rows on one block for X'y, and the quality cost is the larger one:
a serial float32 sum of a million terms carries a rounding error of order
n * eps (relative 6e-2 worst, sqrt(n) * eps ~ 6e-5 typical), so the Gram's
small eigenvalues (Istella's near-collinear columns) are roundoff and the
evidence iteration puts coefficient mass on noise directions
(board 0.8.34 Istella FAST r2 -4.2e4 against scikit-learn's float64 SVD
of X at -890).

Here: x_linear/enetcv_fast.mojo's grid pattern on the augmented matrix
[X, y]: per 8192-row chunk the column sums (`bf_sums_kernel`), the means
(`bf_means_kernel`), per (chunk, 32 x 32 tile pair) the centered partial
Gram (`bf_gram_kernel`, 4 cells a thread, shared-memory tiles), then one
thread per cell folding the chunk partials ascending (`bf_gram_red_kernel`)
into the layout `bayes_ridge_fit` / `ard_fit` read: xm at fw[0, d),
G at fw[gg, gg + d*d), X'y at fw[xty, xty + d), and three scalars at fw[so]:
the target mean used for centering, |y - ym|^2 and var(y). The partial sums
are 8192 terms long and the chunk fold 123 terms, so the rounding error is
of order log(n) * eps instead of n * eps, and every pass runs on every core.
The team kernel skips its own row passes when ip[6] == 1.
"""
from std.gpu import block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from x_linear.ops import FP, ld, st

comptime BF_TPB = 256
comptime BF_CH = 8192
"""Rows a chunk."""
comptime BF_TS = 32
"""Tile side (columns of [X, y])."""
comptime BF_RB = 32
"""Rows staged per shared-memory round."""


def bf_blocks(count: Int) -> Int:
    return max((count + BF_TPB - 1) // BF_TPB, 1)


def bf_chunks(n: Int) -> Int:
    return max((n + BF_CH - 1) // BF_CH, 1)


def bf_tiles(m: Int) -> Int:
    return (m + BF_TS - 1) // BF_TS


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


def bf_sums_kernel(x: FP, y: FP, n_in: Int32, d_in: Int32, part_s: FP):
    """Chunk block_idx.x: the column sums of [X, y] over its rows."""
    var n = Int(n_in)
    var d = Int(d_in)
    var m = d + 1
    var ch = Int(block_idx.x)
    var lo = ch * BF_CH
    var cnt = min(BF_CH, n - lo)
    var tid = Int(thread_idx.x)
    var sh = stack_allocation[BF_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    if m <= BF_TPB:
        var g_n = BF_TPB // m
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
        for c in range(tid, m, BF_TPB):
            var acc = Float32(0)
            for r in range(cnt):
                acc += _aug(x, y, d, lo + r, c)
            st(part_s, ch * m + c, acc)


def bf_means_kernel(part_s: FP, nch_in: Int32, n_in: Int32, d_in: Int32, fi: Int32, cen: FP, mu: FP, fw: FP):
    """Thread c: the mean of column c of [X, y] (mu, chunks ascending), the
    centering value cen[c] (the mean with an intercept, 0 without), and
    fw[c] = cen[c] for the feature columns (the team's xm)."""
    var d = Int(d_in)
    var m = d + 1
    var c = Int(block_idx.x) * BF_TPB + Int(thread_idx.x)
    if c < m:
        var s = Float32(0)
        for ch in range(Int(nch_in)):
            s += ld(part_s, ch * m + c)
        var v = s / Float32(Int(n_in))
        var cv = v if fi != 0 else Float32(0)
        st(mu, c, v)
        st(cen, c, cv)
        if c < d:
            st(fw, c, cv)


def bf_gram_kernel(x: FP, y: FP, n_in: Int32, d_in: Int32, cen: FP, part_g: FP, npairs_in: Int32, nt_in: Int32):
    """Block b: tile pair b % npairs of chunk b // npairs, the chunk's rows
    of [X, y] centered at cen; 4 cells a thread."""
    var n = Int(n_in)
    var d = Int(d_in)
    var m = d + 1
    var npairs = Int(npairs_in)
    var b = Int(block_idx.x)
    var p = b % npairs
    var ch = b // npairs
    var tjk = _pair(p, Int(nt_in))
    var j0 = tjk[0] * BF_TS
    var k0 = tjk[1] * BF_TS
    var lo = ch * BF_CH
    var cnt = min(BF_CH, n - lo)
    var tid = Int(thread_idx.x)
    var sa = stack_allocation[BF_RB * BF_TS, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var sb = stack_allocation[BF_RB * BF_TS, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var r = tid // 8
    var c0 = (tid % 8) * 4
    var acc = SIMD[DType.float32, 4](0)
    var rb = 0
    while rb < cnt:
        comptime for u in range((BF_RB * BF_TS) // BF_TPB):
            var e = tid + u * BF_TPB
            var rr = e // BF_TS
            var cc = e - rr * BF_TS
            var row = rb + rr
            var va = Float32(0)
            var vb = Float32(0)
            if row < cnt:
                var ja = j0 + cc
                var kb = k0 + cc
                if ja < m:
                    va = _aug(x, y, d, lo + row, ja) - ld(cen, ja)
                if kb < m:
                    vb = _aug(x, y, d, lo + row, kb) - ld(cen, kb)
            sa[e] = va
            sb[e] = vb
        barrier()
        comptime for rr in range(BF_RB):
            var a = sa[rr * BF_TS + r]
            var bv = (sb + rr * BF_TS + c0).load[width=4]()
            acc += a * bv
        barrier()
        rb += BF_RB
    var o = (ch * npairs + p) * (BF_TS * BF_TS) + r * BF_TS + c0
    comptime for e in range(4):
        st(part_g, o + e, acc[e])


def bf_gram_red_kernel(part_g: FP, nch_in: Int32, n_in: Int32, d_in: Int32, npairs_in: Int32, nt_in: Int32,
                       mu: FP, cen: FP, fw: FP, gg_in: Int32, xty_in: Int32, so_in: Int32):
    """Thread (pair, cell): the centered cell over the chunks ascending.
    Cells (j, k) with j, k < d go to both triangles of G; (j, d) is X'y_j;
    (d, d) is |y - cen_d|^2, stored at fw[so + 1] with the centering value
    at fw[so] and var(y) = |y - cen_d|^2 / n - (mu_d - cen_d)^2 at fw[so + 2]."""
    var d = Int(d_in)
    var m = d + 1
    var npairs = Int(npairs_in)
    comptime TT = BF_TS * BF_TS
    var t = Int(block_idx.x) * BF_TPB + Int(thread_idx.x)
    if t < npairs * TT:
        var p = t // TT
        var cell = t - p * TT
        var tjk = _pair(p, Int(nt_in))
        var j = tjk[0] * BF_TS + cell // BF_TS
        var k = tjk[1] * BF_TS + cell % BF_TS
        if j < m and k < m and j <= k:
            var s = Float32(0)
            for ch in range(Int(nch_in)):
                s += ld(part_g, (ch * npairs + p) * TT + cell)
            if k < d:
                st(fw, Int(gg_in) + j * d + k, s)
                st(fw, Int(gg_in) + k * d + j, s)
            elif j < d:
                st(fw, Int(xty_in) + j, s)
            else:
                var so = Int(so_in)
                var cd = ld(cen, d)
                var dm = ld(mu, d) - cd
                st(fw, so, cd)
                st(fw, so + 1, s)
                st(fw, so + 2, s / Float32(Int(n_in)) - dm * dm)


def bayes_fast_stats(
    ctx: DeviceContext, mut dx: DeviceBuffer[DType.float32], mut dy: DeviceBuffer[DType.float32],
    mut dfw: DeviceBuffer[DType.float32], n: Int, d: Int, fi: Int, gg: Int, xty: Int, so: Int,
) raises:
    """The four launches; dfw must already be zero-filled. Synchronizes
    before its scratch is released."""
    var m = d + 1
    var nch = bf_chunks(n)
    var nt = bf_tiles(m)
    var npairs = nt * (nt + 1) // 2
    var dps = ctx.enqueue_create_buffer[DType.float32](nch * m)
    var dmu = ctx.enqueue_create_buffer[DType.float32](m)
    var dcen = ctx.enqueue_create_buffer[DType.float32](m)
    var dpg = ctx.enqueue_create_buffer[DType.float32](nch * npairs * BF_TS * BF_TS)
    ctx.enqueue_function[bf_sums_kernel](
        dx.unsafe_ptr(), dy.unsafe_ptr(), Int32(n), Int32(d), dps.unsafe_ptr(),
        grid_dim=nch, block_dim=BF_TPB,
    )
    ctx.enqueue_function[bf_means_kernel](
        dps.unsafe_ptr(), Int32(nch), Int32(n), Int32(d), Int32(fi), dcen.unsafe_ptr(), dmu.unsafe_ptr(),
        dfw.unsafe_ptr(), grid_dim=bf_blocks(m), block_dim=BF_TPB,
    )
    ctx.enqueue_function[bf_gram_kernel](
        dx.unsafe_ptr(), dy.unsafe_ptr(), Int32(n), Int32(d), dcen.unsafe_ptr(), dpg.unsafe_ptr(),
        Int32(npairs), Int32(nt), grid_dim=npairs * nch, block_dim=BF_TPB,
    )
    ctx.enqueue_function[bf_gram_red_kernel](
        dpg.unsafe_ptr(), Int32(nch), Int32(n), Int32(d), Int32(npairs), Int32(nt), dmu.unsafe_ptr(),
        dcen.unsafe_ptr(), dfw.unsafe_ptr(), Int32(gg), Int32(xty), Int32(so),
        grid_dim=bf_blocks(npairs * BF_TS * BF_TS), block_dim=BF_TPB,
    )
    ctx.synchronize()
    _ = dps^
    _ = dmu^
    _ = dcen^
    _ = dpg^
