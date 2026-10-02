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

from max.gpu.host import DeviceBuffer, DeviceContext
from core.device_zero import enqueue_fill
from decomposition.checks.jacobi_eigh_device import jacobi_rotation_cs
from decomposition.impl.linalg.detail.pca import SIGNFLIP_TPB, sign_flip_kernel
from x_decomp.cells import F32Ptr
from x_decomp.rr import pj_first, pj_second, rr_cs, rr_block, rr_vrow, rr_row_off

comptime PJ_TPB = 256
"""Launch width of every kernel here (under the M2 Pro dispatch limit)."""
comptime PJ_SYNC_ROUNDS = 512
"""Rounds enqueued between two synchronize() calls (bounds the queue)."""


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


def _pj_grid(count: Int) -> Int:
    return (count + PJ_TPB - 1) // PJ_TPB if count > 0 else 1


def eigh_rr_on_device(
    ctx: DeviceContext,
    mut da: DeviceBuffer[DType.float32],
    n: Int,
    sweeps: Int,
    tol: Float32,
    mut diag: List[Float32],
    mut vecs: List[Float32],
) raises -> Tuple[Bool, Int]:
    """The round-robin two-sided Jacobi on `da` (n x n, symmetric, CONSUMED:
    its diagonal ends as the eigenvalues), the whole GPU a round: two launches
    a round, m - 1 rounds a sweep, the cyclic kernel's convergence test
    (sum of squared off-diagonal cells against tol^2 ||A||_F^2; row partials
    pinned in `rr_row_off`, added on the host in float64, rows ascending)
    read once before every sweep, then `sign_flip_kernel` on the basis.

    Returns (converged, sweeps run). Converged: `diag` holds the n diagonal
    cells and `vecs` the sign-flipped basis (vector i in COLUMN i), the pair
    `eigh_ascending` takes. Not converged in `sweeps` sweeps, or a solve that
    moved ||A||_F (J^T A J keeps it): False and nothing stored, so the caller
    runs the cyclic solver on its own copy of the input.

    `x_decomp/rr.mojo::host_eigh_rr` is this solve on the host, step for step
    (one writer a cell a round, so its serial order computes these words).

    Moved here from `x_decomp/device.mojo::DevExec._eigh_par` (lane/neural-
    pass144) so `linalg.eigh` (decomposition/linalg_public_device.mojo) runs
    the same driver: the same launches, the same order, the same bits."""
    var m = n + (n % 2)
    var h = m // 2
    var dv = ctx.enqueue_create_buffer[DType.float32](n * n)
    var dcs = ctx.enqueue_create_buffer[DType.float32](2 * h)
    var doff = ctx.enqueue_create_buffer[DType.float32](3 * n)
    var hoff = ctx.enqueue_create_host_buffer[DType.float32](3 * n)
    ctx.enqueue_function[pj_identity_kernel](dv.unsafe_ptr(), Int32(n), grid_dim=_pj_grid(n * n), block_dim=PJ_TPB)
    var tol2 = Float64(tol) * Float64(tol)
    var converged = False
    var executed = 0
    var fro_in = Float64(-1.0)
    var fro_now = Float64(0.0)
    for sweep in range(sweeps + 1):
        # a sum of squares is never negative: -1 left in the readback is
        # a dispatch that did not run
        enqueue_fill(ctx, doff, Float32(-1.0))
        ctx.enqueue_function[eigh_par_off_kernel](
            da.unsafe_ptr(), doff.unsafe_ptr(), Int32(n), grid_dim=_pj_grid(n), block_dim=PJ_TPB
        )
        ctx.enqueue_copy(dst_ptr=hoff.unsafe_ptr(), src_buf=doff)
        ctx.synchronize()
        var off = Float64(0.0)
        var dg = Float64(0.0)
        var ran = True
        for i in range(n):
            var o = Float64(hoff.unsafe_ptr().unsafe_load(i))
            var d2 = Float64(hoff.unsafe_ptr().unsafe_load(n + i))
            if o < 0.0 or d2 < 0.0:
                ran = False
            off += o
            dg += d2
        if not ran:
            break
        fro_now = off + dg
        if fro_in < 0.0:
            fro_in = fro_now
        if off <= tol2 * fro_now:
            converged = True
            break
        if sweep == sweeps:
            break
        executed += 1
        for rd in range(m - 1):
            ctx.enqueue_function[eigh_par_cs_kernel](
                da.unsafe_ptr(), dcs.unsafe_ptr(), Int32(n), Int32(m), Int32(rd),
                grid_dim=_pj_grid(h), block_dim=PJ_TPB,
            )
            ctx.enqueue_function[eigh_par_update_kernel](
                da.unsafe_ptr(), dv.unsafe_ptr(), dcs.unsafe_ptr(), Int32(n), Int32(m), Int32(rd),
                grid_dim=_pj_grid(h * h + n * h), block_dim=PJ_TPB,
            )
            if rd % PJ_SYNC_ROUNDS == PJ_SYNC_ROUNDS - 1:
                ctx.synchronize()
    # J^T A J keeps ||A||_F: a solve that moved it is not an answer
    if converged and not (abs(fro_now - fro_in) <= 1.0e-3 * fro_in):
        converged = False
    if converged:
        ctx.enqueue_function[sign_flip_kernel](
            dv.unsafe_ptr(), Int32(n), grid_dim=(n, 1, 1), block_dim=(SIGNFLIP_TPB, 1, 1)
        )
        var hv = ctx.enqueue_create_host_buffer[DType.float32](n * n)
        ctx.enqueue_copy(dst_ptr=hv.unsafe_ptr(), src_buf=dv)
        ctx.synchronize()
        # the last test's readback holds the diagonal of the converged A
        diag = List[Float32](capacity=n)
        for i in range(n):
            diag.append(hoff.unsafe_ptr().unsafe_load(2 * n + i))
        vecs = List[Float32](capacity=n * n)
        for i in range(n * n):
            vecs.append(hv.unsafe_ptr().unsafe_load(i))
        _ = hv^
    _ = dv^
    _ = dcs^
    _ = doff^
    _ = hoff^
    ctx.synchronize()
    return (converged, executed)
