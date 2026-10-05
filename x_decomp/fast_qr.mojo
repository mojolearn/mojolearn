# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""FAST on Apple (lane/apple-fast-decomp-linalg, 2026-10-02): the decomp
kit's geqrf / orgqr folds as GRID reductions. NOT an IDENTICAL path:
`x_decomp/device.mojo` reaches these kernels only under
FAST on the Apple column, and only when built with `-D MOJOLEARN_QR_FAST_DEV`
(`QR_FAST_DEV` below; no env read). Recovered onto main by
lane/apple-fast-rec-decomp (2026-10-04).

Cause: `DevExec.geqrf` / `DevExec.orgqr` run every fold of a column (the
reflector norm, and w = v^T a_j for each trailing column) as ONE thread's
chain, rows ascending, so the IDENTICAL bits hold; on Apple the kit once
handed the whole factorization to a host walk (gone from main with lane
hr-qr: the A/B arm now races main's sliced device route and, from the
linalg door, the blocked TSQR of x_decomp/tsqr_device.mojo). Here the norm is
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
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from x_decomp.cells import F32Ptr, geqrf_scale_elem, geqrf_update_elem, orgqr_init_elem, orgqr_update_elem

#: threads per block of every kernel here
comptime FQ_TPB = 256
#: rows a reflector-product block folds (one partial per block and column)
comptime FQ_ROWS = 256
#: most blocks of the norm's first pass (the finish kernel folds their pairs)
comptime FQ_MAX_BLOCKS = 1024


#: Recovered candidate (lane/apple-fast-rec-decomp, 2026-10-04), default OFF,
#: FAST + Apple only. Source lane/apple-fast-decomp-linalg@74d52352b
#: (50d12a950). What it does: `DevExec.geqrf` / `DevExec.orgqr`
#: (x_decomp/device.mojo) run this file's grid folds (the reflector norm as a
#: scaled sum of squares over a grid plus a one-block pair fold; the
#: reflector products as row-chunk partials folded by `fold_kernel`) instead
#: of main's sliced route; linalg.qr takes the FAST kit's geqrf / orgqr ahead
#: of the blocked TSQR only when the FAST Metal binding reports the define
#: (`x_decomp_fast_defines`). Same algorithm (unblocked Householder, dlarfg's
#: sign), a different fold order. Known: queued once as dlin-qr-dev-istella
#: (lane/apple-fast-batch prebuilt arms); no recorded result or failure. It
#: was written against the host walk the Apple column once took (7.7 s on the
#: M3 Ultra), which main has since replaced by the sliced route and the TSQR
#: (qr taxi 46.6 ms on the FAST board), so a speed lead is unlikely. Fixed in
#: the port: the switch is this comptime define (with the FAST + Apple guard)
#: instead of a runtime `fast_qr_on()` read of the define. No dimension
#: window: FQ_ROWS / FQ_TPB are block sizes, FQ_MAX_BLOCKS caps the norm's
#: first pass so the one-block finish folds a bounded count.
comptime QR_FAST_DEV = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and is_defined["MOJOLEARN_QR_FAST_DEV"]()
)


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


def fq_head_finish_kernel(a: F32Ptr, tau: F32Ptr, scal: F32Ptr, part: F32Ptr, k: Int32, diag: Int32, nblk: Int32):
    """`geqrf_head`'s tail on ONE block of FQ_TPB threads: the blocks' (scale,
    ssq) pairs merged by a tree (thread t folds pairs t, t + FQ_TPB, ...,
    then the block's tree, as `fq_head_part_kernel`), and thread 0 finishes:
    xmax is the merged scale of the rows below the diagonal (0: tau = 0, the
    step skipped, as dlarfg); the norm then takes alpha = a[diag] in.
    tau[k], scal = [alpha - beta, 1], beta on the diagonal. `diag` is the
    diagonal's offset k * n + k, so the launch carries no row count."""
    var tid = Int(thread_idx.x)
    var ss = stack_allocation[FQ_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var sq = stack_allocation[FQ_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var scale = Float32(0)
    var ssq = Float32(0)
    var b = tid
    while b < Int(nblk):
        _ssq_merge(scale, ssq, part.unsafe_load(2 * b), part.unsafe_load(2 * b + 1))
        b += FQ_TPB
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
    if tid != 0:
        return
    var K = Int(k)
    var D = Int(diag)
    scale = ss[0]
    ssq = sq[0]
    var alpha = a.unsafe_load(D)
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
    a.unsafe_store(D, beta)


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


# The shipped one-column kernels `_geqrf_fast` / `_orgqr_fast` launch around
# the grid folds above, deleted from x_decomp/device.mojo with main's chain
# route (dc686f153); here unchanged, on the same x_decomp/cells.mojo cells.
def geqrf_scale_kernel(a: F32Ptr, scal: F32Ptr, k: Int32, m: Int32, n: Int32):
    var i = Int(k) + 1 + Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(m):
        geqrf_scale_elem(a, scal, Int(k), i, Int(n))


def geqrf_update_kernel(a: F32Ptr, tau: F32Ptr, scal: F32Ptr, w: F32Ptr, k: Int32, m: Int32, n: Int32):
    var cols = Int(n) - Int(k) - 1
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cols > 0 and t < (Int(m) - Int(k)) * cols:
        var i = Int(k) + t // cols
        var j = Int(k) + 1 + t % cols
        geqrf_update_elem(a, tau, scal, Int(k), i, j, Int(n), w.unsafe_load(j))


def orgqr_init_kernel(q: F32Ptr, m: Int32, qc: Int32):
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < Int(m) * Int(qc):
        orgqr_init_elem(q, t // Int(qc), t % Int(qc), Int(qc))


def orgqr_update_kernel(h: F32Ptr, tau: F32Ptr, q: F32Ptr, w: F32Ptr, k: Int32, m: Int32, n: Int32, qc: Int32):
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < (Int(m) - Int(k)) * Int(qc):
        var i = Int(k) + t // Int(qc)
        var j = t % Int(qc)
        orgqr_update_elem(h, tau, q, Int(k), i, j, Int(n), Int(qc), w.unsafe_load(j))
