# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""FAST, Apple: the L2 epsilon neighborhood with the pair products on the
simdgroup matrix unit, and the SAME adjacency as the unexpanded kernel.

WHY. `eps_unexp_neigh_kernel` sums `(x - y)^2` in scalar FMAs out of shared
memory, about 0.75 TFLOP/s-equivalent on the M4. The matrix unit computes
`q.x` for 8x8 blocks of pairs (`fast_mma_knn.mojo`'s technique).

EXACTNESS. The expanded form `|q|^2 + |x|^2 - 2 q.x` rounds differently
from the unexpanded sum, so on its own it could flip pairs lying within a
few ulps of `eps^2`. It is only used as a FILTER: with the data centered on
its mean (translation does not change distances; the norms shrink to the
data's spread) the expanded value is within
`MARGIN * (|q'|^2 + |x'|^2)` of the unexpanded one (the forward error of a
`d`-term dot product and of the norms is below `(2d + 4) u (|q'|^2 +
|x'|^2)`, centering and the unexpanded sum's own error add `(2d + 12) u`;
`MARGIN = 2^-15 = 512 u` covers `d <= 32` three times over). Pairs clearly
inside or clearly outside are decided by it; the rest are recomputed with
the unexpanded kernel's own arithmetic from the original rows, so every
adjacency byte and every degree equals the unexpanded kernel's.

Layout as `fast_mma_knn.mojo`: one simdgroup owns `8 * A` batch rows, the
block stages 128 dataset rows in shared memory (prefetched), each step
multiplies against `8 * B` of them. Each lane holds two columns of each 8x8
result; the four lanes sharing a row (lane bits 0 and 3) fold their degree
counts by shuffles at the end.

SCOPE. This serves `algorithm='brute'` only; the default ball-cover arm
(`rbc`) never reaches `eps_unexp_neighborhood`. M4, 10,000 x 200,000 x 16:
the unexpanded kernel 0.73-0.98 s, this one 0.10-0.16 s (measured
2026-09-25). Exactness: every adjacency byte and degree equal to the
unexpanded kernel's over 76 thresholds (d = 3, 5, 8, 13, 16, 30, data on a
1/16 grid so pairs sit exactly ON the threshold, and continuous data), and
a sweep of the threshold over +-4 ulps of 110 pairs' own distances (every
pair's flip point hit) with 0 differing bytes.

Gate: `FAST_MMA_EPS_ENABLED` (FAST, Apple, not `-D MOJOLEARN_DBSCAN_FAST_MMA_OFF`),
L2 only, `n_features <= 32`.
"""

from std.atomic import Atomic
from std.bit import count_trailing_zeros
from std.gpu import block_idx, thread_idx
from std.gpu.primitives.warp import shuffle_xor
from std.memory import bitcast, stack_allocation
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from neighbors.impl.detail.fast_mma_knn import _sg_load_t, _sg_mma

comptime FAST_MMA_EPS_ENABLED = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_DBSCAN_FAST_MMA_OFF"]()
)

comptime ME_SG = 8
comptime ME_TPB = ME_SG * 32
comptime ME_T = 128
comptime ME_MAX_D = 32
comptime ME_A = 2
comptime ME_B = 4
comptime ME_MARGIN = Float32(3.0517578125e-05)
"""2^-15: the filter's relative band, see the module docstring."""
comptime ME_NORM_CAP = Float32(1.0e30)
"""Largest centered squared norm the filter serves."""
comptime ME_PAD = Float32(1.0e30)
"""Norm of a padding row: far outside any threshold."""

comptime _M64 = SIMD[DType.float32, 64]


def fast_mma_eps_applies(n_features: Int) -> Bool:
    comptime if not FAST_MMA_EPS_ENABLED:
        return False
    return n_features >= 1 and n_features <= ME_MAX_D


def me_center_kernel(
    x: MutPointer[Float32, MutAnyOrigin],
    mean: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    k_in: Int32,
):
    """One block of 256: the column means (float32, any order: the center
    only has to be near the data, it never reaches the answer)."""
    var n = Int(n_in)
    var k = Int(k_in)
    var tid = Int(thread_idx.x)
    var f = tid % 32
    var part = tid // 32
    var part_sum = stack_allocation[256, Float32, address_space=AddressSpace.SHARED]()
    var s = Float32(0)
    if f < k:
        var r = part
        while r < n:
            s += x[r * k + f]
            r += 8
    part_sum[tid] = s
    barrier()
    if tid < k:
        var t = Float32(0)
        for p in range(8):
            t += part_sum[p * 32 + tid]
        mean[tid] = t / Float32(n)


def me_shift_kernel(
    x: MutPointer[Float32, MutAnyOrigin],
    mean: MutPointer[Float32, MutAnyOrigin],
    xc: MutPointer[Float32, MutAnyOrigin],
    nc: MutPointer[Float32, MutAnyOrigin],
    flag: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
    k_in: Int32,
):
    """`xc = x - mean` and `nc = |xc|^2`, one thread per row. A row whose
    centered norm is not finite or above `ME_NORM_CAP` raises `flag`: the
    filter's band would not bound its error, so the caller takes the
    unexpanded kernel instead (exponent bits, not a float compare: FAST may
    fold an infinity or NaN compare away)."""
    var n = Int(n_in)
    var k = Int(k_in)
    var r = Int(block_idx.x) * 256 + Int(thread_idx.x)
    if r >= n:
        return
    var s = Float32(0)
    for c in range(k):
        var v = x[r * k + c] - mean[c]
        xc[r * k + c] = v
        s += v * v
    nc[r] = s
    var eb = bitcast[DType.uint32](s) & 0x7F800000
    if eb == 0x7F800000 or s > ME_NORM_CAP:
        _ = Atomic.fetch_add(flag, Int32(1))


@always_inline
def _exact_l2(
    x: MutPointer[Float32, MutAnyOrigin], qi: Int, cj: Int, k: Int
) -> Float32:
    """The unexpanded kernel's per-pair arithmetic (`_eps_acc`, L2 arm,
    FAST: `acc = diff * diff + acc` in feature order)."""
    var acc = Float32(0)
    for c in range(k):
        var diff = x[qi * k + c] - x[cj * k + c]
        acc = diff * diff + acc
    return acc


def fast_mma_eps_kernel[D: Int, A: Int, B: Int](
    adj: MutPointer[UInt8, MutAnyOrigin],
    vd: MutPointer[Int32, MutAnyOrigin],
    xc: MutPointer[Float32, MutAnyOrigin],
    nc: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    start_in: Int32,
    m_in: Int32,
    n_in: Int32,
    k_in: Int32,
    thresh: Float32,
    slice_rows_in: Int32,
):
    """Rows `[q0, q0 + 8A)` of the batch (dataset rows `start + .`) against
    the dataset slice `block_idx.y`. `vd` zeroed for `m + 1` first."""
    comptime KS = D // 8
    comptime RB = 8 * B
    var start = Int(start_in)
    var m = Int(m_in)
    var n = Int(n_in)
    var k = Int(k_in)
    var tid = Int(thread_idx.x)
    var sg = tid // 32
    var lane = tid % 32
    var qd = lane // 4
    var frow = (qd & 4) + ((lane // 2) % 4)
    var fcol = (qd & 2) * 2 + (lane % 2) * 2
    var q0 = (Int(block_idx.x) * ME_SG + sg) * 8 * A

    var qf = InlineArray[_M64, A * KS](fill=_M64(0))
    var qn = SIMD[DType.float32, A](0)
    comptime for a in range(A):
        var q = q0 + 8 * a + frow
        if q < m:
            qn[a] = nc[start + q]
        comptime for kk in range(KS):
            var v = _M64(0)
            comptime for e in range(2):
                var c = 8 * kk + fcol + e
                if q < m and c < k:
                    v[e] = Float32(-2) * xc[(start + q) * k + c]
            qf[a * KS + kk] = v
    var cnt = SIMD[DType.int32, A](0)

    var lo = Int(block_idx.y) * Int(slice_rows_in)
    var hi = min(n, lo + Int(slice_rows_in))
    var tile = stack_allocation[
        ME_T * D, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var tnorm = stack_allocation[
        ME_T, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    comptime PER = (ME_T * D) // ME_TPB
    var pf = SIMD[DType.float32, PER](0)
    comptime for u in range(PER):
        var e = tid + u * ME_TPB
        var r = e // D
        var c = e - r * D
        if lo + r < hi and c < k:
            pf[u] = xc[(lo + r) * k + c]
    var base = lo
    while base < hi:
        var rows = min(ME_T, hi - base)
        comptime for u in range(PER):
            tile[tid + u * ME_TPB] = pf[u]
        if tid < ME_T:
            tnorm[tid] = nc[base + tid] if tid < rows else ME_PAD
        barrier()
        var nb = base + rows
        comptime for u in range(PER):
            var e = tid + u * ME_TPB
            var r = e // D
            var c = e - r * D
            var v = Float32(0)
            if nb + r < hi and c < k:
                v = xc[(nb + r) * k + c]
            pf[u] = v
        for cb in range(ME_T // RB):
            var xf = InlineArray[_M64, B * KS](fill=_M64(0))
            comptime for b in range(B):
                comptime for kk in range(KS):
                    xf[b * KS + kk] = _sg_load_t(
                        tile + (cb * RB + 8 * b) * D + 8 * kk, D
                    )
            # cv[a * 2B + 2b + e]: the expanded value minus |q|^2;
            # nv the matching |x|^2.
            var cv = SIMD[DType.float32, A * 2 * B](0)
            var nv = SIMD[DType.float32, 2 * B](0)
            comptime for b in range(B):
                var n2 = (tnorm + cb * RB + 8 * b + fcol).load[width=2]()
                nv[2 * b] = n2[0]
                nv[2 * b + 1] = n2[1]
                comptime for a in range(A):
                    var acc = _M64(0)
                    comptime for kk in range(KS):
                        acc = _sg_mma(qf[a * KS + kk], xf[b * KS + kk], acc)
                    cv[a * 2 * B + 2 * b] = n2[0] + acc[0]
                    cv[a * 2 * B + 2 * b + 1] = n2[1] + acc[1]
            var col0 = base + cb * RB + fcol
            comptime for a in range(A):
                var q = q0 + 8 * a + frow
                var m_in_ = UInt32(0)
                var m_unc = UInt32(0)
                comptime for j in range(2 * B):
                    var approx = cv[a * 2 * B + j] + qn[a]
                    var band = ME_MARGIN * (qn[a] + nv[j]) + Float32(1.0e-30)
                    if approx + band < thresh:
                        m_in_ |= UInt32(1 << j)
                    elif approx - band <= thresh:
                        m_unc |= UInt32(1 << j)
                # The rare path, written once: recompute the uncertain
                # pairs exactly, lowest column first.
                while m_unc != 0:
                    var j = Int(count_trailing_zeros(m_unc))
                    m_unc &= m_unc - 1
                    var cj = col0 + 8 * (j >> 1) + (j & 1)
                    if q < m and cj < hi:
                        if _exact_l2(x, start + q, cj, k) <= thresh:
                            m_in_ |= UInt32(1) << UInt32(j)
                if q < m:
                    comptime for j in range(2 * B):
                        var cj = col0 + 8 * (j >> 1) + (j & 1)
                        if cj < hi:
                            var bit = (m_in_ >> UInt32(j)) & 1
                            adj[q * n + cj] = UInt8(bit)
                            cnt[a] += Int32(bit)
        barrier()
        base += rows

    var tot = Int32(0)
    comptime for a in range(A):
        var c = cnt[a]
        c += shuffle_xor(c, UInt32(1))
        c += shuffle_xor(c, UInt32(8))
        var q = q0 + 8 * a + frow
        # lanes with bits 0 and 3 clear hold the folded row count once.
        if (lane & 9) == 0 and q < m:
            _ = Atomic.fetch_add(vd + q, c)
            tot += c
    tot += shuffle_xor(tot, UInt32(1))
    tot += shuffle_xor(tot, UInt32(2))
    tot += shuffle_xor(tot, UInt32(4))
    tot += shuffle_xor(tot, UInt32(8))
    tot += shuffle_xor(tot, UInt32(16))
    if lane == 0 and tot != 0:
        _ = Atomic.fetch_add(vd + m, tot)


def _launch_eps[D: Int](
    ctx: DeviceContext,
    mut adj: DeviceBuffer[DType.uint8],
    mut vd: DeviceBuffer[DType.int32],
    mut xc: DeviceBuffer[DType.float32],
    mut nc: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    start: Int,
    m: Int,
    n: Int,
    k: Int,
    thresh: Float32,
    slice_rows: Int,
    qblocks: Int,
    slices: Int,
) raises:
    ctx.enqueue_function[fast_mma_eps_kernel[D, ME_A, ME_B]](
        adj.unsafe_ptr(), vd.unsafe_ptr(), xc.unsafe_ptr(), nc.unsafe_ptr(),
        x.unsafe_ptr(), Int32(start), Int32(m), Int32(n), Int32(k), thresh,
        Int32(slice_rows),
        grid_dim=(qblocks, slices, 1), block_dim=(ME_TPB, 1, 1),
    )


def fast_mma_eps_neighborhood(
    ctx: DeviceContext,
    mut adj: DeviceBuffer[DType.uint8],
    mut vd: DeviceBuffer[DType.int32],
    mut x: DeviceBuffer[DType.float32],
    start_vertex_id: Int,
    m: Int,
    n: Int,
    k: Int,
    thresh: Float32,
) raises -> Bool:
    """Same contract as `eps_unexp_neighborhood[DBSCAN_METRIC_L2]`, or
    False (nothing written) when a row's norm is outside the filter's range
    and the caller must run the unexpanded kernel."""
    var mean = ctx.enqueue_create_buffer[DType.float32](ME_MAX_D)
    var xc = ctx.enqueue_create_buffer[DType.float32](n * k)
    var nc = ctx.enqueue_create_buffer[DType.float32](n)
    var flag = ctx.enqueue_create_buffer[DType.int32](1)
    var flag_h = ctx.enqueue_create_host_buffer[DType.int32](1)
    ctx.enqueue_memset(vd, Int32(0))
    ctx.enqueue_memset(flag, Int32(0))
    ctx.enqueue_function[me_center_kernel](
        x.unsafe_ptr(), mean.unsafe_ptr(), Int32(n), Int32(k),
        grid_dim=(1, 1, 1), block_dim=(256, 1, 1),
    )
    ctx.enqueue_function[me_shift_kernel](
        x.unsafe_ptr(), mean.unsafe_ptr(), xc.unsafe_ptr(), nc.unsafe_ptr(),
        flag.unsafe_ptr(), Int32(n), Int32(k),
        grid_dim=((n + 255) // 256, 1, 1), block_dim=(256, 1, 1),
    )
    ctx.enqueue_copy(dst_ptr=flag_h.unsafe_ptr(), src_buf=flag)
    ctx.synchronize()
    if flag_h.unsafe_ptr().unsafe_load(0) != 0:
        _ = mean^
        _ = xc^
        _ = nc^
        _ = flag^
        _ = flag_h^
        return False
    var QPB = ME_SG * 8 * ME_A
    var qblocks = (m + QPB - 1) // QPB
    var slices = 960 // qblocks
    if slices < 1:
        slices = 1
    if slices > 64:
        slices = 64
    var slice_rows = (n + slices - 1) // slices
    slice_rows = ((slice_rows + ME_T - 1) // ME_T) * ME_T
    slices = (n + slice_rows - 1) // slice_rows
    if k <= 8:
        _launch_eps[8](ctx, adj, vd, xc, nc, x, start_vertex_id, m, n, k, thresh, slice_rows, qblocks, slices)
    elif k <= 16:
        _launch_eps[16](ctx, adj, vd, xc, nc, x, start_vertex_id, m, n, k, thresh, slice_rows, qblocks, slices)
    else:
        _launch_eps[32](ctx, adj, vd, xc, nc, x, start_vertex_id, m, n, k, thresh, slice_rows, qblocks, slices)
    ctx.synchronize()
    _ = mean^
    _ = xc^
    _ = nc^
    _ = flag^
    _ = flag_h^
    return True
