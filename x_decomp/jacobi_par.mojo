# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""FAST on Apple (lane/decomp-apple3): the x_decomp kit's two Jacobi solvers
in the ROUND-ROBIN ordering, every rotation of a round on its own part of the
GPU. NOT an IDENTICAL path: `x_decomp/device.mojo` reaches these kernels only
in a FAST build for Metal, and IDENTICAL keeps the cyclic order its bits are
pinned to.

The cyclic kernels (`jacobi_eigh_kernel`, `jacobi_eigh2_kernel`,
`one_sided_jacobi_svd_kernel`, `one_sided_svd2_kernel`) run ONE block of 256
threads for the whole solve: n (n - 1) / 2 rotations a sweep, one after the
other, each behind a barrier. At n = 1500 that is 1.1 million serial
rotations a sweep (m4pro-b 1790619265077: eigh 46 to 99 s, svd 800 x 800 43 s,
LocallyLinearEmbedding 304 s) on one block while the rest of the GPU idles.

Here the n columns play a round-robin tournament (the circle method on
m = n or n + 1 players, the odd one a bye): m - 1 rounds a sweep, each round
m / 2 DISJOINT pairs, so the rotations of a round commute and run at once.

  one-sided SVD: ONE launch a round, a block a pair. The block folds the
    pair's 2 x 2 Gram (256 partials, a pairwise tree), takes the same
    relative test and the same `jacobi_rotation_cs` as the cyclic kernel and
    rotates the two columns of R and of V. R and V are kept transposed, so
    a column is a contiguous row. No device word crosses threads inside a
    launch: thread t owns elements t, t + 256, ... of the block's two rows.
  eigh: TWO launches a round. `eigh_par_cs_kernel` takes every pair's (c, s)
    from its three cells; `eigh_par_update_kernel` applies J^T A J by 2 x 2
    blocks: block (i, j) of pairs i < j is J_i^T B J_j, one thread, stored to
    the block and to its mirror (A stays exactly symmetric), the pair's own
    block is the closed form (a_pp - t a_pq, a_qq + t a_pq, 0), and V J by
    (row, pair). Every cell has one writer a launch and no reader but its
    writer.

Convergence is the cyclic kernels' own test (SVD: a sweep with no rotation
against tol sqrt(app) sqrt(aqq); eigh: sum of squared off-diagonal cells
against tol^2 ||A||_F^2), read on the host once a sweep. The host refuses
nothing here: a solve that does not converge in its budget returns False
and the caller runs the cyclic solver on the untouched input.
"""
from std.gpu import block_dim, block_idx, thread_idx
from std.math import sqrt
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from decomposition.checks.jacobi_eigh_device import jacobi_rotation_cs
from x_decomp.cells import F32Ptr
from checks.numerics import ftz, identical_mul_add, identical_mul, identical_div

comptime PJ_TPB = 256
"""Launch width of every kernel here (under the M2 Pro dispatch limit)."""


@always_inline
def pj_first(r: Int, b: Int, m: Int) -> Int:
    """One player of pair b in round r of the circle method on m players
    (m even, 0 <= r < m - 1, 0 <= b < m / 2): pair 0 is (r, m - 1), pair b
    is ((r + b) mod (m - 1), (r - b) mod (m - 1))."""
    if b == 0:
        return r
    return (r + b) % (m - 1)


@always_inline
def pj_second(r: Int, b: Int, m: Int) -> Int:
    if b == 0:
        return m - 1
    return (r + m - 1 - b) % (m - 1)


def pj_transpose_kernel(src: F32Ptr, dst: F32Ptr, n_in: Int32):
    """dst = src^T (n x n, row-major)."""
    var n = Int(n_in)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < n * n:
        var i = t // n
        var j = t - i * n
        dst.unsafe_store(j * n + i, src.unsafe_load(t))


def pj_identity_kernel(dst: F32Ptr, n_in: Int32):
    """dst = I (n x n)."""
    var n = Int(n_in)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < n * n:
        var i = t // n
        var j = t - i * n
        dst.unsafe_store(t, Float32(1.0) if i == j else Float32(0.0))


def svd_par_round_kernel(
    rt: F32Ptr, vt: F32Ptr, flags: F32Ptr, n_in: Int32, m_in: Int32, round_in: Int32, tol_in: Float32
):
    """Round `round_in` of the one-sided Jacobi: block b takes pair b. `rt`
    and `vt` are R^T and V^T (n x n); `flags` has 2 (m / 2) slots: slot b is
    set to 1 when pair b rotated (the host clears it before a sweep) and
    slot m / 2 + b takes round + 1 from EVERY block, the bye included, so
    the host can tell a dispatch Metal dropped (M2: a launch over the
    pipeline's thread limit is dropped with no error) from a sweep without
    rotations. Launch with m / 2 blocks of exactly PJ_TPB threads."""
    var n = Int(n_in)
    var m = Int(m_in)
    var b = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var a0 = pj_first(Int(round_in), b, m)
    var a1 = pj_second(Int(round_in), b, m)
    var p = min(a0, a1)
    var q = max(a0, a1)
    var slab = stack_allocation[3 * PJ_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var rot = stack_allocation[3, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    if tid == 0:
        flags.unsafe_store(m // 2 + b, Float32(Int(round_in) + 1))
    # q >= n is the bye of an odd n: uniform over the block
    if q < n:
        var lp = Float32(0.0)
        var lq = Float32(0.0)
        var lpq = Float32(0.0)
        var i = tid
        while i < n:
            var xp = rt.unsafe_load(p * n + i)
            var xq = rt.unsafe_load(q * n + i)
            lp = xp * xp + lp
            lq = xq * xq + lq
            lpq = xp * xq + lpq
            i += PJ_TPB
        slab[tid] = lp
        slab[PJ_TPB + tid] = lq
        slab[2 * PJ_TPB + tid] = lpq
        barrier()
        if tid == 0:
            var w = PJ_TPB // 2
            while w > 0:
                for t in range(w):
                    slab[t] = slab[t] + slab[t + w]
                    slab[PJ_TPB + t] = slab[PJ_TPB + t] + slab[PJ_TPB + t + w]
                    slab[2 * PJ_TPB + t] = slab[2 * PJ_TPB + t] + slab[2 * PJ_TPB + t + w]
                w = w // 2
            var app = slab[0]
            var aqq = slab[PJ_TPB]
            var apq = slab[2 * PJ_TPB]
            # two roots, as the cyclic kernel: the product of two squared
            # norms can overflow where this cannot
            var thresh = tol_in * (sqrt(app) * sqrt(aqq))
            var c0 = Float32(1.0)
            var s0 = Float32(0.0)
            var go = Float32(0.0)
            if abs(apq) > thresh:
                var cs = jacobi_rotation_cs(app, aqq, apq)
                c0 = cs[0]
                s0 = cs[1]
                go = Float32(1.0)
                flags.unsafe_store(b, Float32(1.0))
            rot[0] = c0
            rot[1] = s0
            rot[2] = go
        barrier()
        var c = rot[0]
        var s = rot[1]
        if rot[2] != Float32(0.0):
            var k = tid
            while k < n:
                var rp = rt.unsafe_load(p * n + k)
                var rq = rt.unsafe_load(q * n + k)
                var vp = vt.unsafe_load(p * n + k)
                var vq = vt.unsafe_load(q * n + k)
                rt.unsafe_store(p * n + k, c * rp - s * rq)
                rt.unsafe_store(q * n + k, s * rp + c * rq)
                vt.unsafe_store(p * n + k, c * vp - s * vq)
                vt.unsafe_store(q * n + k, s * vp + c * vq)
                k += PJ_TPB


def svd_par_norm_kernel(rt: F32Ptr, s_out: F32Ptr, n_in: Int32):
    """s_out[j] = the norm of row j of rt (column j of the rotated R): block
    j, 256 partials, the pairwise tree. Launch with n blocks of PJ_TPB."""
    var n = Int(n_in)
    var j = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var slab = stack_allocation[PJ_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var acc = Float32(0.0)
    if j < n:
        var i = tid
        while i < n:
            var x = rt.unsafe_load(j * n + i)
            acc = x * x + acc
            i += PJ_TPB
    slab[tid] = acc
    barrier()
    if tid == 0 and j < n:
        var w = PJ_TPB // 2
        while w > 0:
            for t in range(w):
                slab[t] = slab[t] + slab[t + w]
            w = w // 2
        s_out.unsafe_store(j, sqrt(slab[0]))


# ------------------------------------------------ the round-robin eigh as THE order (lane/neural-pass104)
# Andrew, 2026-10-01: the two-sided Jacobi in the round-robin ordering is the
# eigh of every column and of the host (the cyclic order stays behind
# `-D MOJOLEARN_XD_EIGH_CYCLIC=1`). Its arithmetic is pinned here as the
# cyclic solver's: every product `identical_mul`, every rotation step the
# host cyclic `_rot_sub_1` / `_rot_add_1` form (one fma, the other product
# flushed and negated), every result flushed, the (c, s) of
# `jacobi_rotation_cs`. A round's blocks, cells and V rows each have one
# writer and no other reader, so the host, running them one after another
# in place, computes the device's words.
@always_inline
def rr_sub(c: Float32, x: Float32, s: Float32, y: Float32) -> Float32:
    """c x - s y."""
    return ftz(identical_mul_add(c, x, -ftz(identical_mul(s, y))))


@always_inline
def rr_add(s: Float32, x: Float32, c: Float32, y: Float32) -> Float32:
    """s x + c y."""
    return ftz(identical_mul_add(s, x, ftz(identical_mul(c, y))))


@always_inline
def rr_cs(a: F32Ptr, n: Int, m: Int, r: Int, b: Int) -> SIMD[DType.float32, 2]:
    """Pair b of round r: its rotation, (1, 0) for the bye."""
    var a0 = pj_first(r, b, m)
    var a1 = pj_second(r, b, m)
    var p = min(a0, a1)
    var q = max(a0, a1)
    if q >= n:
        return SIMD[DType.float32, 2](Float32(1.0), Float32(0.0))
    return jacobi_rotation_cs(a.unsafe_load(p * n + p), a.unsafe_load(q * n + q), a.unsafe_load(p * n + q))


@always_inline
def rr_block(a: F32Ptr, cs: F32Ptr, n: Int, m: Int, r: Int, i: Int, j: Int):
    """Block (i, j), i <= j, of round r: J_i^T B J_j (and its mirror); the
    pair's own block in closed form."""
    var i0 = pj_first(r, i, m)
    var i1 = pj_second(r, i, m)
    var pi = min(i0, i1)
    var qi = max(i0, i1)
    var ci = cs.unsafe_load(2 * i)
    var si = cs.unsafe_load(2 * i + 1)
    if i == j:
        if qi < n:
            var app = a.unsafe_load(pi * n + pi)
            var aqq = a.unsafe_load(qi * n + qi)
            var apq = a.unsafe_load(pi * n + qi)
            var tt = ftz(identical_div(si, ci))
            var dlt = ftz(identical_mul(tt, apq))
            a.unsafe_store(pi * n + pi, ftz(app - dlt))
            a.unsafe_store(qi * n + qi, ftz(aqq + dlt))
            a.unsafe_store(pi * n + qi, Float32(0.0))
            a.unsafe_store(qi * n + pi, Float32(0.0))
        return
    var j0 = pj_first(r, j, m)
    var j1 = pj_second(r, j, m)
    var pj = min(j0, j1)
    var qj = max(j0, j1)
    var cj = cs.unsafe_load(2 * j)
    var sj = cs.unsafe_load(2 * j + 1)
    var vi = qi < n
    var vj = qj < n
    var b00 = a.unsafe_load(pi * n + pj)
    var b01 = Float32(0.0)
    var b10 = Float32(0.0)
    var b11 = Float32(0.0)
    if vj:
        b01 = a.unsafe_load(pi * n + qj)
    if vi:
        b10 = a.unsafe_load(qi * n + pj)
    if vi and vj:
        b11 = a.unsafe_load(qi * n + qj)
    var t00 = rr_sub(cj, b00, sj, b01)
    var t01 = rr_add(sj, b00, cj, b01)
    var t10 = rr_sub(cj, b10, sj, b11)
    var t11 = rr_add(sj, b10, cj, b11)
    var n00 = rr_sub(ci, t00, si, t10)
    var n01 = rr_sub(ci, t01, si, t11)
    var n10 = rr_add(si, t00, ci, t10)
    var n11 = rr_add(si, t01, ci, t11)
    a.unsafe_store(pi * n + pj, n00)
    a.unsafe_store(pj * n + pi, n00)
    if vj:
        a.unsafe_store(pi * n + qj, n01)
        a.unsafe_store(qj * n + pi, n01)
    if vi:
        a.unsafe_store(qi * n + pj, n10)
        a.unsafe_store(pj * n + qi, n10)
    if vi and vj:
        a.unsafe_store(qi * n + qj, n11)
        a.unsafe_store(qj * n + qi, n11)


@always_inline
def rr_vrow(v: F32Ptr, cs: F32Ptr, n: Int, m: Int, r: Int, k: Int, j: Int):
    """V = V J for row k, pair j of round r."""
    var j0 = pj_first(r, j, m)
    var j1 = pj_second(r, j, m)
    var pj = min(j0, j1)
    var qj = max(j0, j1)
    if qj < n:
        var cj = cs.unsafe_load(2 * j)
        var sj = cs.unsafe_load(2 * j + 1)
        var vkp = v.unsafe_load(k * n + pj)
        var vkq = v.unsafe_load(k * n + qj)
        v.unsafe_store(k * n + pj, rr_sub(cj, vkp, sj, vkq))
        v.unsafe_store(k * n + qj, rr_add(sj, vkp, cj, vkq))


@always_inline
def rr_row_off(a: F32Ptr, n: Int, k: Int) -> SIMD[DType.float32, 2]:
    """(row k's off-diagonal squares, j ascending; a_kk squared)."""
    var acc = Float32(0.0)
    for j in range(n):
        if j != k:
            var x = a.unsafe_load(k * n + j)
            acc = ftz(identical_mul_add(x, x, acc))
    var d = a.unsafe_load(k * n + k)
    return SIMD[DType.float32, 2](acc, ftz(identical_mul(d, d)))


def eigh_par_cs_kernel(a: F32Ptr, cs: F32Ptr, n_in: Int32, m_in: Int32, round_in: Int32):
    """cs[2 b], cs[2 b + 1] = the rotation of pair b in round `round_in`
    (`rr_cs`). One thread a pair."""
    var m = Int(m_in)
    var b = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if b < m // 2:
        var got = rr_cs(a, Int(n_in), m, Int(round_in), b)
        cs.unsafe_store(2 * b, got[0])
        cs.unsafe_store(2 * b + 1, got[1])


def eigh_par_update_kernel(a: F32Ptr, v: F32Ptr, cs: F32Ptr, n_in: Int32, m_in: Int32, round_in: Int32):
    """A = J^T A J and V = V J for round `round_in`: threads 0 .. h h - 1 the
    2 x 2 blocks (i, j), i <= j (`rr_block`), then V's (row, pair) (`rr_vrow`)."""
    var n = Int(n_in)
    var m = Int(m_in)
    var h = m // 2
    var r = Int(round_in)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < h * h:
        var i = t // h
        var j = t - i * h
        if i <= j:
            rr_block(a, cs, n, m, r, i, j)
    elif t < h * h + n * h:
        var u = t - h * h
        var k = u // h
        rr_vrow(v, cs, n, m, r, k, u - k * h)


def eigh_par_off_kernel(a: F32Ptr, dst: F32Ptr, n_in: Int32):
    """dst[k] = row k's off-diagonal squares, dst[n + k] = a_kk^2 (`rr_row_off`),
    dst[2 n + k] = a_kk. One thread a row; the host adds the first two in
    float64, rows ascending."""
    var n = Int(n_in)
    var k = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if k < n:
        var o = rr_row_off(a, n, k)
        dst.unsafe_store(k, o[0])
        dst.unsafe_store(n + k, o[1])
        dst.unsafe_store(2 * n + k, a.unsafe_load(k * n + k))
