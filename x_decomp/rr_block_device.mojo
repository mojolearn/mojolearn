# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The block Jacobi eigh's device kernels (lane fam2-decomp, 2026-10-04): one
thread a cell, every cell the pinned step of x_decomp/rr_block.mojo (read its
header), which the host solver (`host_eigh_rb_sorted`) runs in the same order.
Every kernel launches ceil(count / PJ_TPB) blocks of PJ_TPB; no shared
memory. Each cell has one writer. The pivot-status kernel restores its own
eigenvalue cells in place before any outer matrix update; T, the second V
buffer and the pivot stack are separate buffers."""
from std.gpu import block_dim, block_idx, thread_idx

from checks.numerics import ftz
from x_decomp.cells import F32Ptr
from x_decomp.rr_block import (
    RB_W,
    rb_gather_cell,
    rb_pivot_shift,
    rb_hi,
    rb_idx,
    rb_is_pad,
    rb_left_cell,
    rb_lo,
    rb_rank_real,
    rb_row_dot,
)

comptime RbI32Ptr = MutPointer[Int32, MutAnyOrigin]


def rb_embed_kernel(src: F32Ptr, dst: F32Ptr, n_in: Int32, nn_in: Int32):
    """dst (N x N) = src (n x n) in the leading corner, zeros elsewhere."""
    var n = Int(n_in)
    var nn = Int(nn_in)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < nn * nn:
        var r = t // nn
        var c = t - r * nn
        var x = Float32(0.0)
        if r < n and c < n:
            x = src.unsafe_load(r * n + c)
        dst.unsafe_store(t, x)


def rb_gather_kernel(a: F32Ptr, p: F32Ptr, nn_in: Int32, m_in: Int32, round_in: Int32):
    """p[g] (RB_W x RB_W) = pair g's pivot problem of block round `round_in`
    (`rb_gather_cell`: exactly symmetric)."""
    var m = Int(m_in)
    var r = Int(round_in)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < (m // 2) * RB_W * RB_W:
        var g = t // (RB_W * RB_W)
        var rem = t - g * RB_W * RB_W
        var x = rem // RB_W
        var y = rem - x * RB_W
        p.unsafe_store(t, rb_gather_cell(a, Int(nn_in), rb_lo(m, r, g), rb_hi(m, r, g), x, y))


def rb_bad_kernel(info: F32Ptr, bad: F32Ptr, h_in: Int32, a: F32Ptr, wl: F32Ptr, nn_in: Int32, m_in: Int32, round_in: Int32):
    """The pivot solves' marks folded into two sticky flags (the driver reads
    them once a sweep): bad[0] = 1 when a problem did not converge, bad[1] = 1
    when its block did not run (info still -1). Every writer of a flag
    stores the same word."""
    var g = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if g < Int(h_in):
        var x = info.unsafe_load(2 * g)
        if x < Float32(0.0):
            bad.unsafe_store(1, Float32(1.0))
        elif x != Float32(1.0):
            bad.unsafe_store(0, Float32(1.0))
        else:
            # A is still the original matrix here: this kernel runs before
            # any outer left update. One writer per eigenvalue, same host sum.
            var shift = rb_pivot_shift(a, Int(nn_in), rb_lo(Int(m_in), Int(round_in), g), rb_hi(Int(m_in), Int(round_in), g))
            for i in range(RB_W):
                wl.unsafe_store(g * RB_W + i, ftz(wl.unsafe_load(g * RB_W + i) + shift))


def rb_right_kernel(a: F32Ptr, wv: F32Ptr, tb: F32Ptr, nn_in: Int32, m_in: Int32, round_in: Int32):
    """T = A W on the block pairs gi < gj, in pair coordinates: thread
    t = u N + v, u = gi RB_W + x, v = gj RB_W + lc."""
    var nn = Int(nn_in)
    var m = Int(m_in)
    var r = Int(round_in)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < nn * nn:
        var u = t // nn
        var v = t - u * nn
        var gi = u // RB_W
        var gj = v // RB_W
        if gi < gj:
            var row = rb_idx(rb_lo(m, r, gi), rb_hi(m, r, gi), u - gi * RB_W)
            tb.unsafe_store(
                t,
                rb_row_dot(a + row * nn, wv + gj * RB_W * RB_W, rb_lo(m, r, gj), rb_hi(m, r, gj), v - gj * RB_W),
            )


def rb_left_kernel(
    a: F32Ptr, tb: F32Ptr, wv: F32Ptr, wval: F32Ptr, nn_in: Int32, m_in: Int32, round_in: Int32
):
    """A = W^T T: thread (u, v) of pairs gi < gj stores its cell and the
    mirror; gi == gj stores the pivot's closed form, diag(w_g); gi > gj
    nothing (its cell is the mirror of another thread's)."""
    var nn = Int(nn_in)
    var m = Int(m_in)
    var r = Int(round_in)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < nn * nn:
        var u = t // nn
        var v = t - u * nn
        var gi = u // RB_W
        var gj = v // RB_W
        var lr = u - gi * RB_W
        var lc = v - gj * RB_W
        if gi <= gj:
            var ra = rb_idx(rb_lo(m, r, gi), rb_hi(m, r, gi), lr)
            var ca = rb_idx(rb_lo(m, r, gj), rb_hi(m, r, gj), lc)
            if gi < gj:
                var x = rb_left_cell(tb, wv + gi * RB_W * RB_W, nn, gi, lr, v)
                a.unsafe_store(ra * nn + ca, x)
                a.unsafe_store(ca * nn + ra, x)
            else:
                var d = Float32(0.0)
                if lr == lc:
                    d = wval.unsafe_load(gi * RB_W + lr)
                a.unsafe_store(ra * nn + ca, d)


def rb_v_kernel(v: F32Ptr, v2: F32Ptr, wv: F32Ptr, nn_in: Int32, m_in: Int32, round_in: Int32):
    """V2 = V W: thread t = k N + c (c = g RB_W + lc in pair coordinates)
    stores row k's cell of the pair's column lc."""
    var nn = Int(nn_in)
    var m = Int(m_in)
    var r = Int(round_in)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < nn * nn:
        var k = t // nn
        var c = t - k * nn
        var g = c // RB_W
        var lc = c - g * RB_W
        var lo = rb_lo(m, r, g)
        var hi = rb_hi(m, r, g)
        v2.unsafe_store(k * nn + rb_idx(lo, hi, lc), rb_row_dot(v + k * nn, wv + g * RB_W * RB_W, lo, hi, lc))


def rb_pad_kernel(v: F32Ptr, pad: F32Ptr, nn_in: Int32, n_in: Int32):
    """pad[c] = 1 for a pad column of the basis, else 0 (`rb_is_pad`)."""
    var nn = Int(nn_in)
    var c = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if c < nn:
        pad.unsafe_store(c, Float32(1.0) if rb_is_pad(v, nn, Int(n_in), c) else Float32(0.0))


def rb_rank_kernel(key: F32Ptr, pad: F32Ptr, pos: RbI32Ptr, nn_in: Int32, n_in: Int32):
    """pos[i] = real column i's ASCENDING position, n - 1 - `rb_rank_real`;
    -1 for a pad column."""
    var nn = Int(nn_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < nn:
        if pad.unsafe_load(i) != Float32(0.0):
            pos.unsafe_store(i, Int32(-1))
        else:
            pos.unsafe_store(i, Int32(Int(n_in) - 1 - rb_rank_real(key, pad, nn, i)))


def rb_scatter_kernel(
    key: F32Ptr, vecs: F32Ptr, pos: RbI32Ptr, nn_in: Int32, n_in: Int32, w_out: F32Ptr, v_out: F32Ptr
):
    """Thread t = r N + i, r < n: real column i of the N x N basis to column
    pos[i] of the n x n output; row 0's thread also moves the value."""
    var nn = Int(nn_in)
    var n = Int(n_in)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < n * nn:
        var r = t // nn
        var i = t - r * nn
        var c = Int(pos.unsafe_load(i))
        if c >= 0 and c < n:
            v_out.unsafe_store(r * n + c, vecs.unsafe_load(r * nn + i))
            if r == 0:
                w_out.unsafe_store(c, key.unsafe_load(i))
