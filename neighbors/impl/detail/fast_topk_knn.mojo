# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""FAST, Apple: brute-force L2 k-NN with the selection fused into the
distance loop, so no distance matrix is ever written.

WHY. The tiled arm materializes a `query_tile x n_index` distance tile and
selects from it: at 10,000 queries against a 1,000,000 x 16 index that is
about 80 GB of tile traffic (write then read) on the M4, 3.6 s for about
160 GFLOP of arithmetic. cuVS's own fused kernel (`fused_l2_knn`) did not
help as built on Apple (measured 2026-09-25, same 3.6-4.5 s either arm).

HOW. One thread owns one query (held in registers, `n_features <=
FKT_MAX_D`) and its running top-k (`k <= FKT_MAX_K`, sorted ascending by
(distance, index)). A block of `FKT_QB` threads streams one slice of the
index through shared memory `FKT_T` rows at a time; `grid.y` splits the
index into slices so a small query count still fills the machine, and
`fast_topk_merge_kernel` merges each query's per-slice lists. Distances are
the expanded `||x||^2 - 2 q.x + ||q||^2` (row norms computed once per
shared tile), as FAST's tiled arm computes them; FAST promises quality,
not bits. Ties order by index, so the result is deterministic.

Only for FAST builds on the Apple column (`fast_topk_knn_applies`); every
other build keeps the dispatch it had. `-D MOJOLEARN_KNN_FAST_TOPK_OFF`
turns it off.
"""

from std.gpu import block_idx, thread_idx
from std.math import sqrt, inf
from std.memory import stack_allocation
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from std.os import getenv
from std.time import perf_counter_ns

comptime FKT_QB = 256
comptime FKT_T = 64
comptime FKT_MAX_D = 32
comptime FKT_MAX_K = 16

comptime FAST_TOPK_KNN_ENABLED = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_KNN_FAST_TOPK_OFF"]()
)


def fast_topk_knn_applies(n_features: Int, k: Int) -> Bool:
    """Whether this build and shape take the fused top-k arm."""
    comptime if not FAST_TOPK_KNN_ENABLED:
        return False
    return n_features >= 1 and n_features <= FKT_MAX_D and k >= 1 and k <= FKT_MAX_K


@always_inline
def _worse(da: Float32, ia: UInt32, db: Float32, ib: UInt32) -> Bool:
    """(da, ia) sorts after (db, ib): larger distance, ties by larger index."""
    return da > db or (da == db and ia > ib)


comptime FKT_QPT = 2
"""Queries per thread: each loaded index row is reused across them."""


def fast_topk_partial_kernel[D: Int, K: Int](
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
    """Every register array is indexed at COMPILE time (queries and the
    shared tile zero-padded to `D` features, top-k lists sized to the `K`
    bucket), so nothing spills. Each thread owns `FKT_QPT` queries and reuses
    each shared row across them; the running worst distance per query is a
    scalar lane, so the common row costs one dot product and one compare per
    query."""
    comptime QPT = FKT_QPT
    var nq = Int(n_queries_in)
    var ni = Int(n_index_in)
    var d = Int(d_in)
    var k = Int(k_in)
    var s = Int(block_idx.y)
    var q0 = Int(block_idx.x) * FKT_QB * QPT + Int(thread_idx.x)
    var qv = SIMD[DType.float32, D * QPT](0)
    var qn = SIMD[DType.float32, QPT](0)
    comptime for j in range(QPT):
        var q = q0 + j * FKT_QB
        if q < nq:
            comptime for c in range(D):
                if c < d:
                    qv[j * D + c] = queries[unsafe_offset = q * d + c]
            var sq = Float32(0)
            comptime for c in range(D):
                sq += qv[j * D + c] * qv[j * D + c]
            qn[j] = sq
    var bd = SIMD[DType.float32, K * QPT](inf[DType.float32]())
    var bi = SIMD[DType.uint32, K * QPT](0xFFFFFFFF)
    var worst = SIMD[DType.float32, QPT](inf[DType.float32]())
    var lo = s * Int(slice_rows_in)
    var hi = min(ni, lo + Int(slice_rows_in))
    var tile = stack_allocation[
        FKT_T * D, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var tnorm = stack_allocation[
        FKT_T, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var base = lo
    while base < hi:
        var rows = min(FKT_T, hi - base)
        var e = Int(thread_idx.x)
        while e < FKT_T * D:
            var r = e // D
            var c = e - r * D
            var v = Float32(0)
            if r < rows and c < d:
                v = index[unsafe_offset = (base + r) * d + c]
            tile[e] = v
            e += FKT_QB
        barrier()
        if Int(thread_idx.x) < FKT_T:
            var rr = (tile + Int(thread_idx.x) * D).load[width=D]()
            tnorm[Int(thread_idx.x)] = (rr * rr).reduce_add()
        barrier()
        for r in range(rows):
            var row = (tile + r * D).load[width=D]()
            var rn = tnorm[r]
            comptime for j in range(QPT):
                var dot = Float32(0)
                comptime for c in range(D):
                    dot += row[c] * qv[j * D + c]
                # expanded form: ||x||^2 - 2 q.x (+ ||q||^2 at the end)
                var acc = rn - Float32(2) * dot
                if acc <= worst[j]:
                    var cd = acc
                    var ci = UInt32(base + r)
                    comptime for t in range(K):
                        if t < k and _worse(bd[j * K + t], bi[j * K + t], cd, ci):
                            var td = bd[j * K + t]
                            var ti = bi[j * K + t]
                            bd[j * K + t] = cd
                            bi[j * K + t] = ci
                            cd = td
                            ci = ti
                    comptime for t in range(K):
                        if t == k - 1:
                            worst[j] = bd[j * K + t]
        barrier()
        base += rows
    comptime for j in range(QPT):
        var q = q0 + j * FKT_QB
        if q < nq:
            var o = (s * nq + q) * k
            comptime for t in range(K):
                if t < k:
                    part_d[unsafe_offset = o + t] = max(bd[j * K + t] + qn[j], Float32(0))
                    part_i[unsafe_offset = o + t] = bi[j * K + t]


def fast_topk_merge_kernel(
    part_d: MutPointer[Float32, MutAnyOrigin],
    part_i: MutPointer[UInt32, MutAnyOrigin],
    out_d: MutPointer[Float32, MutAnyOrigin],
    out_i: MutPointer[UInt32, MutAnyOrigin],
    n_queries_in: Int32,
    k_in: Int32,
    n_slices_in: Int32,
    take_sqrt_in: Int32,
):
    var nq = Int(n_queries_in)
    var k = Int(k_in)
    var q = Int(block_idx.x) * FKT_QB + Int(thread_idx.x)
    if q >= nq:
        return
    var bd = SIMD[DType.float32, FKT_MAX_K](inf[DType.float32]())
    var bi = SIMD[DType.uint32, FKT_MAX_K](0xFFFFFFFF)
    for s in range(Int(n_slices_in)):
        var o = (s * nq + q) * k
        for jj in range(k):
            var cd = part_d[unsafe_offset = o + jj]
            var ci = part_i[unsafe_offset = o + jj]
            comptime for j in range(FKT_MAX_K):
                if j < k and _worse(bd[j], bi[j], cd, ci):
                    var td = bd[j]
                    var ti = bi[j]
                    bd[j] = cd
                    bi[j] = ci
                    cd = td
                    ci = ti
    comptime for j in range(FKT_MAX_K):
        if j < k:
            var v = bd[j]
            if Int(take_sqrt_in) != 0:
                v = sqrt(v)
            out_d[unsafe_offset = q * k + j] = v
            out_i[unsafe_offset = q * k + j] = bi[j]


def fast_topk_knn(
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
    """Row-major queries and index; `out_*` are `n_queries x k`. Waits for
    its own partials before returning."""
    var qblocks = (n_queries + FKT_QB - 1) // FKT_QB
    var slices = 320 // ((n_queries + FKT_QB * FKT_QPT - 1) // (FKT_QB * FKT_QPT))
    if slices < 1:
        slices = 1
    if slices > 64:
        slices = 64
    var slice_rows = (n_index + slices - 1) // slices
    slices = (n_index + slice_rows - 1) // slice_rows
    var st_on = getenv("MOJOLEARN_STAGE_TIMES") == "1"
    var t0 = 0
    if st_on:
        ctx.synchronize()
        t0 = Int(perf_counter_ns())
    var part_d = ctx.enqueue_create_buffer[DType.float32](slices * n_queries * k)
    var part_i = ctx.enqueue_create_buffer[DType.uint32](slices * n_queries * k)
    var pblocks = (n_queries + FKT_QB * FKT_QPT - 1) // (FKT_QB * FKT_QPT)
    if n_features <= 8 and k <= 8:
        ctx.enqueue_function[fast_topk_partial_kernel[8, 8]](
            queries.unsafe_ptr(), index.unsafe_ptr(), part_d.unsafe_ptr(),
            part_i.unsafe_ptr(), Int32(n_queries), Int32(n_index),
            Int32(n_features), Int32(k), Int32(slice_rows),
            grid_dim=(pblocks, slices, 1), block_dim=(FKT_QB, 1, 1),
        )
    elif n_features <= 8 and k <= 16:
        ctx.enqueue_function[fast_topk_partial_kernel[8, 16]](
            queries.unsafe_ptr(), index.unsafe_ptr(), part_d.unsafe_ptr(),
            part_i.unsafe_ptr(), Int32(n_queries), Int32(n_index),
            Int32(n_features), Int32(k), Int32(slice_rows),
            grid_dim=(pblocks, slices, 1), block_dim=(FKT_QB, 1, 1),
        )
    elif n_features <= 16 and k <= 8:
        ctx.enqueue_function[fast_topk_partial_kernel[16, 8]](
            queries.unsafe_ptr(), index.unsafe_ptr(), part_d.unsafe_ptr(),
            part_i.unsafe_ptr(), Int32(n_queries), Int32(n_index),
            Int32(n_features), Int32(k), Int32(slice_rows),
            grid_dim=(pblocks, slices, 1), block_dim=(FKT_QB, 1, 1),
        )
    elif n_features <= 16 and k <= 16:
        ctx.enqueue_function[fast_topk_partial_kernel[16, 16]](
            queries.unsafe_ptr(), index.unsafe_ptr(), part_d.unsafe_ptr(),
            part_i.unsafe_ptr(), Int32(n_queries), Int32(n_index),
            Int32(n_features), Int32(k), Int32(slice_rows),
            grid_dim=(pblocks, slices, 1), block_dim=(FKT_QB, 1, 1),
        )
    elif n_features <= 32 and k <= 8:
        ctx.enqueue_function[fast_topk_partial_kernel[32, 8]](
            queries.unsafe_ptr(), index.unsafe_ptr(), part_d.unsafe_ptr(),
            part_i.unsafe_ptr(), Int32(n_queries), Int32(n_index),
            Int32(n_features), Int32(k), Int32(slice_rows),
            grid_dim=(pblocks, slices, 1), block_dim=(FKT_QB, 1, 1),
        )
    elif n_features <= 32 and k <= 16:
        ctx.enqueue_function[fast_topk_partial_kernel[32, 16]](
            queries.unsafe_ptr(), index.unsafe_ptr(), part_d.unsafe_ptr(),
            part_i.unsafe_ptr(), Int32(n_queries), Int32(n_index),
            Int32(n_features), Int32(k), Int32(slice_rows),
            grid_dim=(pblocks, slices, 1), block_dim=(FKT_QB, 1, 1),
        )
    ctx.enqueue_function[fast_topk_merge_kernel](
        part_d.unsafe_ptr(),
        part_i.unsafe_ptr(),
        out_dist.unsafe_ptr(),
        out_idx.unsafe_ptr(),
        Int32(n_queries),
        Int32(k),
        Int32(slices),
        Int32(1) if take_sqrt else Int32(0),
        grid_dim=(qblocks, 1, 1),
        block_dim=(FKT_QB, 1, 1),
    )
    ctx.synchronize()
    if st_on:
        print("FAST_TOPK_KNN ms=" + String((Int(perf_counter_ns()) - t0) // 1000000)
              + " slices=" + String(slices) + " qblocks=" + String(qblocks))
    _ = part_d^
    _ = part_i^
