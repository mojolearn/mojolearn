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

  eigh: TWO launches a round. `eigh_par_cs_kernel` takes every pair's (c, s)
    from its three cells; `eigh_par_update_kernel` applies J^T A J by 2 x 2
    blocks: block (i, j) of pairs i < j is J_i^T B J_j, one thread, stored to
    the block and to its mirror (A stays exactly symmetric), the pair's own
    block is the closed form (a_pp - t a_pq, a_qq + t a_pq, 0), and V J by
    (row, pair). Every cell has one writer a launch and no reader but its
    writer.

Convergence is the cyclic kernel's own test (sum of squared off-diagonal
cells against tol^2 ||A||_F^2), both sums folded on the device
(`eigh_par_off_fold_kernel`, the fixed order of `x_decomp/rr.mojo`
`rr_off_fold`) and three scalars read on the host once a sweep. A solve that
does not converge in its budget returns False and the caller runs the cyclic
solver on the untouched input. (cgfin-c-decomp deleted the one-sided SVD's
round-robin experiment, MOJOLEARN_XD_PJ_SVD_MIN, default never.)
"""
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.numerics import ftz
from x_decomp.cells import F32Ptr
from x_decomp.rr import RR_OFF_TPB, rr_cs, rr_block, rr_vrow, rr_row_off

comptime PJ_TPB = 256
"""Launch width of every kernel here (under the M2 Pro dispatch limit)."""


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
    dst[2 n + k] = a_kk. One thread a row; `eigh_par_off_fold_kernel` folds the
    first two."""
    var n = Int(n_in)
    var k = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if k < n:
        var o = rr_row_off(a, n, k)
        dst.unsafe_store(k, o[0])
        dst.unsafe_store(n + k, o[1])
        dst.unsafe_store(2 * n + k, a.unsafe_load(k * n + k))


def eigh_par_off_fold_kernel(src: F32Ptr, out: F32Ptr, n_in: Int32):
    """out[0] = the off-diagonal sum, out[1] = the diagonal sum of
    `eigh_par_off_kernel`'s rows (src[0, n), src[n, 2 n)), out[2] = the least
    of those 2 n cells (a -1 left there is a dispatch that did not run).
    `rr_off_fold`'s order: thread t adds rows t, t + RR_OFF_TPB, ...
    ascending, then the pairwise tree. ONE block of RR_OFF_TPB threads (n is
    the matrix order, not a row count)."""
    var n = Int(n_in)
    var tid = Int(thread_idx.x)
    var so = stack_allocation[RR_OFF_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var sd = stack_allocation[RR_OFF_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var sm = stack_allocation[RR_OFF_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var ao = Float32(0.0)
    var ad = Float32(0.0)
    var am = Float32(0.0)
    var k = tid
    while k < n:
        var o = src.unsafe_load(k)
        var d = src.unsafe_load(n + k)
        ao = ftz(ao + o)
        ad = ftz(ad + d)
        am = min(am, min(o, d))
        k += RR_OFF_TPB
    so[tid] = ao
    sd[tid] = ad
    sm[tid] = am
    barrier()
    var w = RR_OFF_TPB // 2
    while w > 0:
        if tid < w:
            so[tid] = ftz(so[tid] + so[tid + w])
            sd[tid] = ftz(sd[tid] + sd[tid + w])
            sm[tid] = min(sm[tid], sm[tid + w])
        barrier()
        w = w // 2
    if tid == 0:
        out.unsafe_store(0, so[0])
        out.unsafe_store(1, sd[0])
        out.unsafe_store(2, sm[0])
