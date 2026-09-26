# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""FAST, Apple: brute-force L2 k-NN whose dot products run on the Apple
GPU's simdgroup matrix unit (8x8x8 multiply-accumulate), with the top-k
selection fused behind it, so no distance matrix is ever written.

WHY. `fast_topk_knn` (one thread per two queries, each index row read from
shared memory and dotted in scalar FMAs) sits about 20x under the M4's FP32
rate: every pair costs a shared-memory row read per thread. Here one
simdgroup holds `8 * MQ_A` queries as matrix fragments (two floats per lane
per 8x8 block) and multiplies them against `8 * MQ_B` index rows per step
straight out of shared memory, so the 16 or 32 multiply-adds of a pair are a
share of one matrix instruction and the lane only compares the result.

HOW. Distances are the expanded `||x||^2 - 2 q.x` (+ `||q||^2` at the end,
clamped at 0), as the other FAST arms compute them; FAST promises quality,
not bits. Each lane holds two columns of each 8x8 result, so four lanes
share a query (lanes differing in bits 0 and 3); each keeps its own sorted
top-k for that query and the four lists meet in the merge kernel. The
filter threshold is the minimum of the four lanes' k-th distances (exchanged
once per shared tile): any candidate above it already has k better
candidates in some lane, so it cannot be in the answer. Ties order by index.
`grid.y` splits the index into slices so few queries still fill the GPU.

The matrix unit is reached through the AIR intrinsics Metal's own
`simdgroup_load` / `simdgroup_multiply_accumulate` compile to (declared as
external functions; the backend resolves them). A `simdgroup_float8x8` is a
`<64 x float>` whose elements 0 and 1 are the lane's two thread elements, at
row `(q & 4) + ((lane / 2) % 4)`, columns `(q & 2) * 2 + (lane % 2) * 2 + {0,1}`
with `q = lane / 4`.

Only for FAST builds on the Apple column; see `fast_mma_knn_applies`.
`-D MOJOLEARN_KNN_FAST_MMA_OFF` turns it off (the scalar fused arm, then the
tiled arm, take over).

WHAT MADE IT FAST (M4, 10,000 queries x 1,000,000 x 16, k = 5; measured
2026-09-25). The matrix unit alone did NOT help (2.2 s against the scalar
arm's 1.9 s): the cost was the selection code, a K-slot insertion unrolled
once per candidate position (32 copies per step) that overflowed the
instruction cache. Written ONCE per query block, behind a bit mask of the
step's passing candidates, it fell to 0.50 s; prefetching the next shared
tile into registers while the current one is multiplied took it to 0.22 s.
"""

from std.ffi import external_call
from std.gpu import block_idx, thread_idx
from std.gpu.primitives.warp import shuffle_xor
from std.math import sqrt
from std.bit import count_trailing_zeros
from std.memory import stack_allocation
from std.sys.compile import is_defined
from std.sys.info import _accelerator_arch, has_apple_gpu_accelerator
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from std.os import getenv
from std.time import perf_counter_ns

comptime MQ_SG = 8
"""Simdgroups per block."""
comptime MQ_TPB = MQ_SG * 32
comptime MQ_T = 128
"""Index rows per shared tile."""
comptime MQ_MAX_D = 32
comptime MQ_MAX_K = 32
comptime MQ_BIG = Float32(3.0e38)
"""Empty list slot and starting threshold (finite: FAST may fold
infinities away)."""
comptime MQ_PAD = Float32(3.3e38)
"""Norm of a padding row: above `MQ_BIG`, so never admitted."""

comptime FAST_MMA_KNN_ENABLED = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_KNN_FAST_MMA_OFF"]()
)

comptime _M64 = SIMD[DType.float32, 64]
comptime _V2 = SIMD[DType.int64, 2]


def fast_mma_knn_applies(n_features: Int, k: Int) -> Bool:
    """Whether this build and shape take the matrix-unit arm."""
    comptime if not FAST_MMA_KNN_ENABLED:
        return False
    return (
        n_features >= 1 and n_features <= MQ_MAX_D and k >= 1 and k <= MQ_MAX_K
    )


@always_inline
def _sg_load_t(
    p: UnsafePointer[Float32, MutUntrackedOrigin, address_space=AddressSpace.SHARED],
    stride: Int,
) -> _M64:
    """8x8 block B[c][j] = p[j * stride + c] (a transposed load of 8 rows).

    The intrinsic's signature depends on the AIR version the build targets:
    `--target-accelerator metal:1`..`metal:3` (what most binding scripts
    pass) emit the older AIR, whose load takes `(ptr, i64 stride, <2 x i64>
    origin, i1 transpose)`; the M4-native target (`metal:4`) takes `(ptr,
    <2 x i64> <stride, 8>, <2 x i64> element strides, <2 x i64> origin)`.
    The wrong one crashes Metal's backend compiler at pipeline creation
    (XPC_ERROR_CONNECTION_INTERRUPTED), measured on the M4 2026-09-25."""
    comptime arch = _accelerator_arch()
    comptime if "metal:1" in arch or "metal:2" in arch or "metal:3" in arch:
        return external_call["air.simdgroup_matrix_8x8_load.v64f32.p3f32", _M64](
            p, Int64(stride), _V2(0, 0), True
        )
    else:
        return external_call["air.simdgroup_matrix_8x8_load.v64f32.p3f32", _M64](
            p, _V2(Int64(stride), 8), _V2(Int64(stride), 1), _V2(0, 0)
        )


@always_inline
def _sg_mma(a: _M64, b: _M64, c: _M64) -> _M64:
    return external_call[
        "air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32",
        _M64,
    ](a, b, c)


@always_inline
def _worse(da: Float32, ia: UInt32, db: Float32, ib: UInt32) -> Bool:
    """(da, ia) sorts after (db, ib): larger distance, ties by larger index."""
    return da > db or (da == db and ia > ib)


def fast_mma_partial_kernel[D: Int, K: Int, A: Int, B: Int](
    queries: MutPointer[Float32, MutAnyOrigin],
    index: MutPointer[Float32, MutAnyOrigin],
    part_d: MutPointer[Float32, MutAnyOrigin],
    part_i: MutPointer[UInt32, MutAnyOrigin],
    n_queries_in: Int32,
    n_index_in: Int32,
    d_in: Int32,
    k_in: Int32,
    slice_rows_in: Int32,
):
    """One simdgroup: queries `[q0, q0 + 8A)`, index slice `block_idx.y`.
    Partials land as list `4 * slice + sub` of each query (`sub` = the
    lane's quarter), `k` sorted entries each."""
    comptime KS = D // 8
    comptime RB = 8 * B
    var nq = Int(n_queries_in)
    var ni = Int(n_index_in)
    var d = Int(d_in)
    var k = Int(k_in)
    var s = Int(block_idx.y)
    var tid = Int(thread_idx.x)
    var sg = tid // 32
    var lane = tid % 32
    var qd = lane // 4
    var frow = (qd & 4) + ((lane // 2) % 4)
    var fcol = (qd & 2) * 2 + (lane % 2) * 2
    var sub = (lane & 1) | (((lane >> 3) & 1) << 1)
    var q0 = (Int(block_idx.x) * MQ_SG + sg) * 8 * A

    # Query fragments: element (frow, fcol + e) of block (a, kk) is
    # query q0 + 8a + frow, feature 8kk + fcol + e (zero padded).
    var qf = InlineArray[_M64, A * KS](fill=_M64(0))
    comptime for a in range(A):
        var q = q0 + 8 * a + frow
        comptime for kk in range(KS):
            var v = _M64(0)
            comptime for e in range(2):
                var c = 8 * kk + fcol + e
                if q < nq and c < d:
                    v[e] = Float32(-2) * queries[unsafe_offset = q * d + c]
            qf[a * KS + kk] = v

    var bd = SIMD[DType.float32, K * A](MQ_BIG)
    var bi = SIMD[DType.uint32, K * A](0xFFFFFFFF)
    var worst = SIMD[DType.float32, A](MQ_BIG)
    var thr = SIMD[DType.float32, A](MQ_BIG)

    var lo = s * Int(slice_rows_in)
    var hi = min(ni, lo + Int(slice_rows_in))
    var tile = stack_allocation[
        MQ_T * D, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var tnorm = stack_allocation[
        MQ_T, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    comptime PER = (MQ_T * D) // MQ_TPB
    var pf = SIMD[DType.float32, PER](0)
    # Prefetch: the next tile's elements are read from device memory into
    # registers before the current tile is computed, hiding their latency.
    comptime for u in range(PER):
        var e = tid + u * MQ_TPB
        var r = e // D
        var c = e - r * D
        if lo + r < hi and c < d:
            pf[u] = index[unsafe_offset = (lo + r) * d + c]
    var base = lo
    while base < hi:
        var rows = min(MQ_T, hi - base)
        comptime for u in range(PER):
            tile[tid + u * MQ_TPB] = pf[u]
        barrier()
        if tid < MQ_T:
            var rr = (tile + tid * D).load[width=D]()
            var nn = (rr * rr).reduce_add()
            if tid >= rows:
                nn = MQ_PAD
            tnorm[tid] = nn
        barrier()
        var nb = base + rows
        comptime for u in range(PER):
            var e = tid + u * MQ_TPB
            var r = e // D
            var c = e - r * D
            var v = Float32(0)
            if nb + r < hi and c < d:
                v = index[unsafe_offset = (nb + r) * d + c]
            pf[u] = v
        for cb in range(MQ_T // RB):
            var xf = InlineArray[_M64, B * KS](fill=_M64(0))
            comptime for b in range(B):
                comptime for kk in range(KS):
                    xf[b * KS + kk] = _sg_load_t(
                        tile + (cb * RB + 8 * b) * D + 8 * kk, D
                    )
            # cv[a * 2B + 2b + e]: candidate (query a, column 8b + fcol + e).
            var cv = SIMD[DType.float32, A * 2 * B](0)
            comptime for b in range(B):
                var n2 = (tnorm + cb * RB + 8 * b + fcol).load[width=2]()
                comptime for a in range(A):
                    var acc = _M64(0)
                    comptime for kk in range(KS):
                        acc = _sg_mma(qf[a * KS + kk], xf[b * KS + kk], acc)
                    cv[a * 2 * B + 2 * b] = n2[0] + acc[0]
                    cv[a * 2 * B + 2 * b + 1] = n2[1] + acc[1]
            comptime for a in range(A):
                var m = UInt32(0)
                comptime for j in range(2 * B):
                    if cv[a * 2 * B + j] <= thr[a]:
                        m |= UInt32(1 << j)
                # The rare path, written once per query block: one
                # candidate at a time, lowest column first.
                while m != 0:
                    var j = Int(count_trailing_zeros(m))
                    m &= m - 1
                    var cd = Float32(0)
                    comptime for jj in range(2 * B):
                        if j == jj:
                            cd = cv[a * 2 * B + jj]
                    if cd <= thr[a]:
                        var ci = UInt32(base + cb * RB + 8 * (j >> 1) + fcol + (j & 1))
                        comptime for t in range(K):
                            if t < k and _worse(bd[a * K + t], bi[a * K + t], cd, ci):
                                var td = bd[a * K + t]
                                var ti = bi[a * K + t]
                                bd[a * K + t] = cd
                                bi[a * K + t] = ci
                                cd = td
                                ci = ti
                        comptime for t in range(K):
                            if t == k - 1:
                                worst[a] = bd[a * K + t]
                        thr[a] = min(thr[a], worst[a])
        # Share the threshold among the four lanes holding each query.
        # Two valid bounds on the query's final k-th distance: the smallest
        # lane k-th (that lane alone holds k better candidates), and the
        # largest lane j-th for j = ceil(k / 4) (the four lanes together
        # hold 4j >= k candidates at or below it).
        var jq = (k + 3) // 4
        comptime for a in range(A):
            var w = thr[a]
            w = min(w, shuffle_xor(w, UInt32(1)))
            w = min(w, shuffle_xor(w, UInt32(8)))
            var u = MQ_BIG
            comptime for t in range(K):
                if t == jq - 1:
                    u = bd[a * K + t]
            u = max(u, shuffle_xor(u, UInt32(1)))
            u = max(u, shuffle_xor(u, UInt32(8)))
            thr[a] = min(w, u)
        barrier()
        base += rows

    comptime for a in range(A):
        var q = q0 + 8 * a + frow
        if q < nq:
            var qn = Float32(0)
            for c in range(d):
                var v = queries[unsafe_offset = q * d + c]
                qn += v * v
            var o = ((4 * s + sub) * nq + q) * k
            comptime for t in range(K):
                if t < k:
                    part_d[unsafe_offset = o + t] = max(bd[a * K + t] + qn, Float32(0))
                    part_i[unsafe_offset = o + t] = bi[a * K + t]


def fast_mma_merge_kernel[K: Int](
    part_d: MutPointer[Float32, MutAnyOrigin],
    part_i: MutPointer[UInt32, MutAnyOrigin],
    out_d: MutPointer[Float32, MutAnyOrigin],
    out_i: MutPointer[UInt32, MutAnyOrigin],
    n_queries_in: Int32,
    k_in: Int32,
    n_lists_in: Int32,
    take_sqrt_in: Int32,
):
    var nq = Int(n_queries_in)
    var k = Int(k_in)
    var q = Int(block_idx.x) * 256 + Int(thread_idx.x)
    if q >= nq:
        return
    var bd = SIMD[DType.float32, K](MQ_BIG)
    var bi = SIMD[DType.uint32, K](0xFFFFFFFF)
    for s in range(Int(n_lists_in)):
        var o = (s * nq + q) * k
        for jj in range(k):
            var cd = part_d[unsafe_offset = o + jj]
            var ci = part_i[unsafe_offset = o + jj]
            comptime for j in range(K):
                if j < k and _worse(bd[j], bi[j], cd, ci):
                    var td = bd[j]
                    var ti = bi[j]
                    bd[j] = cd
                    bi[j] = ci
                    cd = td
                    ci = ti
    comptime for j in range(K):
        if j < k:
            var v = bd[j]
            if Int(take_sqrt_in) != 0:
                v = sqrt(v)
            out_d[unsafe_offset = q * k + j] = v
            out_i[unsafe_offset = q * k + j] = bi[j]


def _launch_partial[D: Int, K: Int, A: Int, B: Int](
    ctx: DeviceContext,
    mut queries: DeviceBuffer[DType.float32],
    mut index: DeviceBuffer[DType.float32],
    mut part_d: DeviceBuffer[DType.float32],
    mut part_i: DeviceBuffer[DType.uint32],
    n_queries: Int,
    n_index: Int,
    n_features: Int,
    k: Int,
    slice_rows: Int,
    qblocks: Int,
    slices: Int,
) raises:
    ctx.enqueue_function[fast_mma_partial_kernel[D, K, A, B]](
        queries.unsafe_ptr(), index.unsafe_ptr(), part_d.unsafe_ptr(),
        part_i.unsafe_ptr(), Int32(n_queries), Int32(n_index),
        Int32(n_features), Int32(k), Int32(slice_rows),
        grid_dim=(qblocks, slices, 1), block_dim=(MQ_TPB, 1, 1),
    )


def _launch_merge[K: Int](
    ctx: DeviceContext,
    mut part_d: DeviceBuffer[DType.float32],
    mut part_i: DeviceBuffer[DType.uint32],
    mut out_dist: DeviceBuffer[DType.float32],
    mut out_idx: DeviceBuffer[DType.uint32],
    n_queries: Int,
    k: Int,
    n_lists: Int,
    take_sqrt: Bool,
) raises:
    ctx.enqueue_function[fast_mma_merge_kernel[K]](
        part_d.unsafe_ptr(), part_i.unsafe_ptr(), out_dist.unsafe_ptr(),
        out_idx.unsafe_ptr(), Int32(n_queries), Int32(k), Int32(n_lists),
        Int32(1) if take_sqrt else Int32(0),
        grid_dim=((n_queries + 255) // 256, 1, 1), block_dim=(256, 1, 1),
    )


comptime MQ_A = 2
"""Query 8-blocks per simdgroup for k <= 8 (and `MQ_A16` for k <= 16; k up
to 32 takes 1 to keep the lists in registers)."""
comptime MQ_B = 4
"""Index 8-blocks per step."""
comptime MQ_A16 = 2


def fast_mma_knn(
    ctx: DeviceContext,
    mut queries: DeviceBuffer[DType.float32],
    mut index: DeviceBuffer[DType.float32],
    mut out_dist: DeviceBuffer[DType.float32],
    mut out_idx: DeviceBuffer[DType.uint32],
    n_queries: Int,
    n_index: Int,
    n_features: Int,
    k: Int,
    take_sqrt: Bool,
) raises:
    """Row-major queries and index; `out_*` are `n_queries x k`, each row
    ascending by (distance, index). Waits for its own partials."""
    var a_sel = MQ_A if (k <= 8) else (MQ_A16 if k <= 16 else 1)
    var QPB = MQ_SG * 8 * a_sel
    var qblocks = (n_queries + QPB - 1) // QPB
    var slices = 480 // qblocks
    if slices < 1:
        slices = 1
    if slices > 64:
        slices = 64
    var slice_rows = (n_index + slices - 1) // slices
    slice_rows = ((slice_rows + MQ_T - 1) // MQ_T) * MQ_T
    slices = (n_index + slice_rows - 1) // slice_rows
    var st_on = getenv("MOJOLEARN_STAGE_TIMES") == "1"
    var t0 = 0
    if st_on:
        ctx.synchronize()
        t0 = Int(perf_counter_ns())
    var part_d = ctx.enqueue_create_buffer[DType.float32](4 * slices * n_queries * k)
    var part_i = ctx.enqueue_create_buffer[DType.uint32](4 * slices * n_queries * k)
    if n_features <= 8:
        if k <= 8:
            _launch_partial[8, 8, MQ_A, MQ_B](ctx, queries, index, part_d, part_i, n_queries, n_index, n_features, k, slice_rows, qblocks, slices)
        elif k <= 16:
            _launch_partial[8, 16, MQ_A16, MQ_B](ctx, queries, index, part_d, part_i, n_queries, n_index, n_features, k, slice_rows, qblocks, slices)
        else:
            _launch_partial[8, 32, 1, MQ_B](ctx, queries, index, part_d, part_i, n_queries, n_index, n_features, k, slice_rows, qblocks, slices)
    elif n_features <= 16:
        if k <= 8:
            _launch_partial[16, 8, MQ_A, MQ_B](ctx, queries, index, part_d, part_i, n_queries, n_index, n_features, k, slice_rows, qblocks, slices)
        elif k <= 16:
            _launch_partial[16, 16, MQ_A16, MQ_B](ctx, queries, index, part_d, part_i, n_queries, n_index, n_features, k, slice_rows, qblocks, slices)
        else:
            _launch_partial[16, 32, 1, MQ_B](ctx, queries, index, part_d, part_i, n_queries, n_index, n_features, k, slice_rows, qblocks, slices)
    else:
        if k <= 8:
            _launch_partial[32, 8, MQ_A, MQ_B](ctx, queries, index, part_d, part_i, n_queries, n_index, n_features, k, slice_rows, qblocks, slices)
        elif k <= 16:
            _launch_partial[32, 16, MQ_A16, MQ_B](ctx, queries, index, part_d, part_i, n_queries, n_index, n_features, k, slice_rows, qblocks, slices)
        else:
            _launch_partial[32, 32, 1, MQ_B](ctx, queries, index, part_d, part_i, n_queries, n_index, n_features, k, slice_rows, qblocks, slices)
    if k <= 8:
        _launch_merge[8](ctx, part_d, part_i, out_dist, out_idx, n_queries, k, 4 * slices, take_sqrt)
    elif k <= 16:
        _launch_merge[16](ctx, part_d, part_i, out_dist, out_idx, n_queries, k, 4 * slices, take_sqrt)
    else:
        _launch_merge[32](ctx, part_d, part_i, out_dist, out_idx, n_queries, k, 4 * slices, take_sqrt)
    ctx.synchronize()
    if st_on:
        print("FAST_MMA_KNN ms=" + String((Int(perf_counter_ns()) - t0) // 1000000)
              + " slices=" + String(slices) + " qblocks=" + String(qblocks))
    _ = part_d^
    _ = part_i^
