# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""FAST on Apple (lane/apple-fast-decomp-linalg, 2026-10-02): the decomp
kit's geqrf / orgqr folds as GRID reductions. NOT an IDENTICAL path:
`x_decomp/device.mojo` reaches these kernels only under
`GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL` on the Apple column, and only
when `MOJOLEARN_QR_FAST_DEV=1` (host env, read at dispatch).

Cause: `DevExec.geqrf` / `DevExec.orgqr` run every fold of a column (the
reflector norm, and w = v^T a_j for each trailing column) as ONE thread's
chain, rows ascending, so the IDENTICAL bits hold; on Apple the kit then
takes `xd_qr_on_host` (x_decomp/qr_host.mojo), a HOST walk of the whole
factorization inside the board's qr cell (1,000,000 x d). Here the norm is
a scaled sum of squares over a grid of blocks (LAPACK dlassq's running
scale, combined pairwise), and the reflector products are row-chunk
partials (block b owns FQ_ROWS rows, thread t a column, so the loads of a
row are coalesced across the block) folded by `fold_kernel`. The scale
and update steps keep their elementwise grid kernels. Same algorithm
(unblocked Householder, dlarfg's sign), a different fold order.
"""
from std.gpu import block_dim, block_idx, thread_idx
from std.math import sqrt
from std.memory import stack_allocation
from std.os import getenv
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from x_decomp.cells import F32Ptr

#: threads per block of every kernel here
comptime FQ_TPB = 256
#: rows a reflector-product block folds (one partial per block and column)
comptime FQ_ROWS = 256
#: most blocks of the norm's first pass (the finish kernel folds their pairs)
comptime FQ_MAX_BLOCKS = 1024


def fast_qr_on() -> Bool:
    """`MOJOLEARN_QR_FAST_DEV=1` turns the grid folds on (default off)."""
    return String(getenv("MOJOLEARN_QR_FAST_DEV")) == "1"


def fq_dot_blocks(m: Int, k: Int) -> Int:
    """Blocks of a reflector-product launch for rows k .. m - 1."""
    var rows = m - k
    return (rows + FQ_ROWS - 1) // FQ_ROWS if rows > 0 else 1


def fq_head_blocks(m: Int) -> Int:
    """Blocks of the norm's first pass at the largest column height."""
    var b = (m + FQ_TPB - 1) // FQ_TPB
    if b < 1:
        b = 1
    if b > FQ_MAX_BLOCKS:
        b = FQ_MAX_BLOCKS
    return b


@always_inline
def _ssq_add(mut scale: Float32, mut ssq: Float32, v: Float32):
    """Running scaled sum of squares (dlassq): one more |v|."""
    if v > Float32(0):
        if v > scale:
            var r = scale / v
            ssq = Float32(1) + ssq * r * r
            scale = v
        else:
            var r = v / scale
            ssq = ssq + r * r


@always_inline
def _ssq_merge(mut scale: Float32, mut ssq: Float32, s2: Float32, q2: Float32):
    """(scale, ssq) += (s2, q2), another running pair."""
    if s2 > scale:
        var r = scale / s2
        ssq = q2 + ssq * r * r
        scale = s2
    elif s2 > Float32(0):
        var r = s2 / scale
        ssq = ssq + q2 * r * r


def fq_head_part_kernel(a: F32Ptr, part: F32Ptr, k: Int32, m: Int32, n: Int32, nblk: Int32):
    """Block b: the running (scale, ssq) of |a[i, k]| over the rows i > k it
    owns (i = k + 1 + b * FQ_TPB + tid, then strides of nblk * FQ_TPB),
    the block's pairs merged by a tree; part[2 b] = scale (the block's
    largest |x|), part[2 b + 1] = ssq. A block with no row stores (0, 0)."""
    var K = Int(k)
    var M = Int(m)
    var N = Int(n)
    var tid = Int(thread_idx.x)
    var b = Int(block_idx.x)
    var ss = stack_allocation[FQ_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var sq = stack_allocation[FQ_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var scale = Float32(0)
    var ssq = Float32(0)
    var i = K + 1 + b * FQ_TPB + tid
    var step = Int(nblk) * FQ_TPB
    while i < M:
        _ssq_add(scale, ssq, abs(a.unsafe_load(i * N + K)))
        i += step
    ss[tid] = scale
    sq[tid] = ssq
    barrier()
    var half = FQ_TPB // 2
    while half > 0:
        if tid < half:
            var s1 = ss[tid]
            var q1 = sq[tid]
            _ssq_merge(s1, q1, ss[tid + half], sq[tid + half])
            ss[tid] = s1
            sq[tid] = q1
        barrier()
        half //= 2
    if tid == 0:
        part.unsafe_store(2 * b, ss[0])
        part.unsafe_store(2 * b + 1, sq[0])


def fq_head_finish_kernel(
    a: F32Ptr, tau: F32Ptr, scal: F32Ptr, part: F32Ptr, k: Int32, m: Int32, n: Int32, nblk: Int32
):
    """`geqrf_head`'s tail on one thread from the blocks' pairs: xmax is the
    merged scale of the rows below the diagonal (0: tau = 0, the step
    skipped, as dlarfg); the norm then takes alpha in too. tau[k], scal =
    [alpha - beta, 1], beta on the diagonal."""
    if block_idx.x != 0 or thread_idx.x != 0:
        return
    var K = Int(k)
    var N = Int(n)
    var scale = Float32(0)
    var ssq = Float32(0)
    for b in range(Int(nblk)):
        _ssq_merge(scale, ssq, part.unsafe_load(2 * b), part.unsafe_load(2 * b + 1))
    var alpha = a.unsafe_load(K * N + K)
    if scale == Float32(0):
        tau.unsafe_store(K, Float32(0))
        scal.unsafe_store(0, Float32(1))
        scal.unsafe_store(1, Float32(0))
        return
    _ssq_add(scale, ssq, abs(alpha))
    var nrm = scale * sqrt(ssq)
    var beta = -nrm if alpha >= Float32(0) else nrm
    tau.unsafe_store(K, (beta - alpha) / beta)
    scal.unsafe_store(0, alpha - beta)
    scal.unsafe_store(1, Float32(1))
    a.unsafe_store(K * N + K, beta)


def fq_geqrf_dot_kernel(a: F32Ptr, scal: F32Ptr, part: F32Ptr, k: Int32, m: Int32, n: Int32):
    """part[b * n + j] = sum over block b's rows i (k + b * FQ_ROWS .. + FQ_ROWS,
    clipped at m) of v_i a[i, j], v_k = 1 and v_i = a[i, k] below, for
    every trailing column j > k (thread t: j = t, t + FQ_TPB, ...); 0 for
    j <= k and when the step does not act. The loads of a row are
    consecutive across the block's threads. `fold_kernel` sums the blocks."""
    var K = Int(k)
    var M = Int(m)
    var N = Int(n)
    var b = Int(block_idx.x)
    var r0 = K + b * FQ_ROWS
    var r1 = r0 + FQ_ROWS
    if r1 > M:
        r1 = M
    var acts = scal.unsafe_load(1) != Float32(0)
    var j = Int(thread_idx.x)
    while j < N:
        var acc = Float32(0)
        if j > K and acts:
            for i in range(r0, r1):
                var vi = Float32(1) if i == K else a.unsafe_load(i * N + K)
                acc = vi * a.unsafe_load(i * N + j) + acc
        part.unsafe_store(b * N + j, acc)
        j += FQ_TPB


def fq_orgqr_dot_kernel(h: F32Ptr, q: F32Ptr, part: F32Ptr, k: Int32, m: Int32, n: Int32, qc: Int32):
    """`orgqr_dot`'s partials: part[b * qc + j] = sum over block b's rows i
    (from k) of v_i q[i, j], v_k = 1 and v_i = h[i, k] below, every column j
    of Q. (A reflector with tau = 0 is skipped by the update kernel, so its
    w is never read.)"""
    var K = Int(k)
    var M = Int(m)
    var N = Int(n)
    var QC = Int(qc)
    var b = Int(block_idx.x)
    var r0 = K + b * FQ_ROWS
    var r1 = r0 + FQ_ROWS
    if r1 > M:
        r1 = M
    var j = Int(thread_idx.x)
    while j < QC:
        var acc = Float32(0)
        for i in range(r0, r1):
            var vi = Float32(1) if i == K else h.unsafe_load(i * N + K)
            acc = vi * q.unsafe_load(i * QC + j) + acc
        part.unsafe_store(b * QC + j, acc)
        j += FQ_TPB
