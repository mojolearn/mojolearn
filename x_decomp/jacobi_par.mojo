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
(`eigh_par_off_part_kernel` then `eigh_par_off_fold_kernel`, the fixed order of `x_decomp/rr.mojo`
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


def eigh_par_off_part_kernel(a: F32Ptr, dst: F32Ptr, part: F32Ptr, n_in: Int32):
    """Thread t of block b takes row k = b RR_OFF_TPB + t: dst[2 n + k] = a_kk
    (the converged diagonal), and block b's pairwise tree over its rows'
    (`rr_row_off`) off-diagonal squares and a_kk^2 (rows past n add 0) goes to
    part[3 b], part[3 b + 1]; part[3 b + 2] = 0 marks the block ran (the
    caller fills -1). Launch ceil(n / RR_OFF_TPB) blocks of RR_OFF_TPB."""
    var n = Int(n_in)
    var b = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var k = b * RR_OFF_TPB + tid
    var so = stack_allocation[RR_OFF_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var sd = stack_allocation[RR_OFF_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var o = SIMD[DType.float32, 2](0.0, 0.0)
    if k < n:
        o = rr_row_off(a, n, k)
        dst.unsafe_store(2 * n + k, a.unsafe_load(k * n + k))
    so[tid] = o[0]
    sd[tid] = o[1]
    barrier()
    var w = RR_OFF_TPB // 2
    while w > 0:
        if tid < w:
            so[tid] = ftz(so[tid] + so[tid + w])
            sd[tid] = ftz(sd[tid] + sd[tid + w])
        barrier()
        w = w // 2
    if tid == 0:
        part.unsafe_store(3 * b, so[0])
        part.unsafe_store(3 * b + 1, sd[0])
        part.unsafe_store(3 * b + 2, Float32(0.0))


def eigh_par_off_fold_kernel(part: F32Ptr, out: F32Ptr, nb_in: Int32):
    """The tree past the blocks: thread t adds block partials t, t +
    RR_OFF_TPB, ... ascending, then the pairwise tree. out[0] = the
    off-diagonal sum, out[1] = the diagonal sum, out[2] = the least ran mark
    (-1: a block of `eigh_par_off_part_kernel` did not run). ONE block over
    the nb = ceil(n / RR_OFF_TPB) block partials (`rr_off_fold`'s order)."""
    var nb = Int(nb_in)
    var tid = Int(thread_idx.x)
    var so = stack_allocation[RR_OFF_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var sd = stack_allocation[RR_OFF_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var sm = stack_allocation[RR_OFF_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var ao = Float32(0.0)
    var ad = Float32(0.0)
    var am = Float32(0.0)
    var b = tid
    while b < nb:
        ao = ftz(ao + part.unsafe_load(3 * b))
        ad = ftz(ad + part.unsafe_load(3 * b + 1))
        am = min(am, part.unsafe_load(3 * b + 2))
        b += RR_OFF_TPB
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
