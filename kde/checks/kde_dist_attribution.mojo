# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Which part of the IDENTICAL euclidean cell costs the time (DEVIATION 2625).

A measuring tool, not a gate, and not a shipping path: the V1..V3 kernels
below drop `ftz` or the pinned `fma` and are NOT identity-safe; they exist
only to attribute the staged distance stage's time. T1 is a candidate
shipping kernel (a SIMD accumulator row in registers) and its log-kernel
matrix is hashed against the tiled pass's.

    pixi run mojo run -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 kde/checks/kde_dist_attribution.mojo
"""

from std.math import fma
from std.memory import bitcast, stack_allocation
from std.time import perf_counter_ns
from std.gpu import block_dim, block_idx, thread_idx

from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.numerics import (
    ftz,
    ftz_simd,
    identical_mul_add_simd,
    identical_sqrt,
    numeric_mode_name,
)
from kde.impl.distance.distance_ops import (
    DIST_L2_SQRT_UNEXPANDED,
    pairwise_unexpanded_kernel,
)
from neighbors.impl.distance.detail.distance_ops import l2_unexp_core
from kde.impl.neighbors.kernel_density import (
    KDE_KERNEL_GAUSSIAN,
    KDE_TILED_CELL,
    KDE_TILED_FEAT,
    KDE_TILED_TILE_FLOATS,
    compute_log_kernel,
    kde_lse_serial_sum_kernel,
    kde_lse_terms_kernel,
    kde_rowmax_reduce_kernel,
    kde_tiled_logk_kernel,
)


def _fixture(n: Int, d: Int, seed: UInt64) -> List[Float32]:
    var out = List[Float32](length=n * d, fill=Float32(0))
    var s = seed
    for i in range(n * d):
        s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var u = Float32(Int((s >> UInt64(40)) & UInt64(0xFFFFFF))) / Float32(16777216.0)
        out[i] = u * Float32(3.4) - Float32(1.7)
    return out^


def _upload(ctx: DeviceContext, values: List[Float32]) raises -> DeviceBuffer[DType.float32]:
    var n = len(values)
    var buf = ctx.enqueue_create_buffer[DType.float32](n)
    var host = ctx.enqueue_create_host_buffer[DType.float32](n)
    for i in range(n):
        host.unsafe_ptr().unsafe_store(i, values[i])
    ctx.enqueue_copy(dst_buf=buf, src_ptr=host.unsafe_ptr())
    ctx.synchronize()
    _ = host^
    return buf^


def _hash_dev(ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], n: Int) raises -> UInt64:
    var host = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=buf)
    ctx.synchronize()
    var h = UInt64(0xCBF29CE484222325)
    var p = host.unsafe_ptr()
    for i in range(n):
        h = (h ^ UInt64(bitcast[DType.uint32](p.unsafe_load(i)))) * UInt64(0x100000001B3)
    _ = host^
    return h


def _say(what: String, ms: Float64):
    print("KDE-ATTR " + what + " ms=" + String(ms))


# ---- one thread per cell, the staged shape ---------------------------------

def v1_fma_no_ftz_kernel(
    dist: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    k_in: Int32,
):
    var n = Int(n_in)
    var k = Int(k_in)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= Int(m_in) * n:
        return
    var i = idx // n
    var j = idx % n
    var acc = Float32(0.0)
    for f in range(k):
        var diff = x.unsafe_load(i * k + f) - y.unsafe_load(j * k + f)
        acc = fma(diff, diff, acc)
    dist.unsafe_store(idx, acc)


def v2_ftz_plain_muladd_kernel(
    dist: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    k_in: Int32,
):
    var n = Int(n_in)
    var k = Int(k_in)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= Int(m_in) * n:
        return
    var i = idx // n
    var j = idx % n
    var acc = Float32(0.0)
    for f in range(k):
        var diff = ftz(ftz(x.unsafe_load(i * k + f)) - ftz(y.unsafe_load(j * k + f)))
        acc = ftz(diff * diff + acc)
    dist.unsafe_store(idx, acc)


def v3_plain_kernel(
    dist: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    k_in: Int32,
):
    var n = Int(n_in)
    var k = Int(k_in)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= Int(m_in) * n:
        return
    var i = idx // n
    var j = idx % n
    var acc = Float32(0.0)
    for f in range(k):
        var diff = x.unsafe_load(i * k + f) - y.unsafe_load(j * k + f)
        acc += diff * diff
    dist.unsafe_store(idx, acc)


def v4_identical_no_sqrt_kernel(
    dist: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    k_in: Int32,
):
    var n = Int(n_in)
    var k = Int(k_in)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= Int(m_in) * n:
        return
    var i = idx // n
    var j = idx % n
    var acc = Float32(0.0)
    for f in range(k):
        acc = l2_unexp_core(acc, ftz(x.unsafe_load(i * k + f)), ftz(y.unsafe_load(j * k + f)))
    dist.unsafe_store(idx, acc)


# ---- T1: the tiled pass with a SIMD accumulator row --------------------------

def t1_simd_logk_kernel(
    logk: MutPointer[Float32, MutAnyOrigin],
    part_max: MutPointer[Float32, MutAnyOrigin],
    query: MutPointer[Float32, MutAnyOrigin],
    train: MutPointer[Float32, MutAnyOrigin],
    n_query_in: Int32,
    n_train_in: Int32,
    d_in: Int32,
    n_chunks_in: Int32,
    chunk_rows_in: Int32,
    bandwidth: Float32,
):
    """`kde_tiled_logk_kernel` for euclidean + gaussian, unweighted, with the
    64 cell accumulators as one SIMD value (lane for lane the same
    `ftz(x - y)` and `ftz(fma(diff, diff, acc))`)."""
    var n_query = Int(n_query_in)
    var n_train = Int(n_train_in)
    var d = Int(d_in)
    var n_chunks = Int(n_chunks_in)
    var chunk_rows = Int(chunk_rows_in)
    var tpb = Int(block_dim.x)
    var tid = Int(thread_idx.x)
    var q = Int(block_idx.x) * tpb + tid
    var chunk = Int(block_idx.y)
    var valid = q < n_query and chunk < n_chunks
    var j_begin = chunk * chunk_rows
    var j_end = j_begin + chunk_rows
    if j_end > n_train:
        j_end = n_train
    var tile = stack_allocation[
        KDE_TILED_TILE_FLOATS,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var neg_inf = bitcast[DType.float32](UInt32(0xFF800000))
    var m = neg_inf
    var qbase = q * d
    var obase = q * n_train
    var j_base = j_begin
    while j_base < j_end:
        var cells = j_end - j_base
        if cells > KDE_TILED_CELL:
            cells = KDE_TILED_CELL
        var acc = SIMD[DType.float32, 64](0.0)
        var f0 = 0
        while f0 < d:
            var feats = d - f0
            if feats > KDE_TILED_FEAT:
                feats = KDE_TILED_FEAT
            barrier()
            var idx = tid
            while idx < KDE_TILED_TILE_FLOATS:
                var feat = idx // KDE_TILED_CELL
                var cell = idx - feat * KDE_TILED_CELL
                var v = Float32(0.0)
                if cell < cells and feat < feats:
                    v = ftz(train.unsafe_load((j_base + cell) * d + f0 + feat))
                tile.unsafe_store(idx, v)
                idx += tpb
            barrier()
            if valid:
                for feat in range(feats):
                    var qv = SIMD[DType.float32, 64](ftz(query.unsafe_load(qbase + f0 + feat)))
                    var row = tile.unsafe_load[width=64](feat * KDE_TILED_CELL)
                    var diff = ftz_simd[64](qv - row)
                    acc = ftz_simd[64](identical_mul_add_simd[64](diff, diff, acc))
            f0 += KDE_TILED_FEAT
        if valid:
            for c in range(cells):
                var dist = ftz(identical_sqrt(acc[c]))
                var v = compute_log_kernel(ftz(dist), bandwidth, KDE_KERNEL_GAUSSIAN)
                logk.unsafe_store(obase + j_base + c, v)
                if v > m:
                    m = v
        j_base += KDE_TILED_CELL
    if valid:
        part_max.unsafe_store(q * n_chunks + chunk, m)


def main() raises:
    print("== kde/checks/kde_dist_attribution.mojo [" + numeric_mode_name() + "] ==")
    var n_train = 100000
    var d = 220
    var h = Float32(0.95)
    var ctx = DeviceContext()
    var train = _fixture(n_train, d, UInt64(7))
    var dtrain = _upload(ctx, train)

    # ---- staged-shape attribution at 500 queries (5e7 cells) ----
    var nq = 500
    var dq = _upload(ctx, _fixture(nq, d, UInt64(11)))
    var cells = nq * n_train
    var dist = ctx.enqueue_create_buffer[DType.float32](cells)
    ctx.synchronize()
    var grid = (cells + 255) // 256
    for rep in range(3):
        var t0 = perf_counter_ns()
        ctx.enqueue_function[pairwise_unexpanded_kernel](
            dist.unsafe_ptr(), dq.unsafe_ptr(), dtrain.unsafe_ptr(), Int32(nq), Int32(n_train), Int32(d),
            Int32(DIST_L2_SQRT_UNEXPANDED), grid_dim=(grid, 1, 1), block_dim=(256, 1, 1),
        )
        ctx.synchronize()
        _say("q500 rep=" + String(rep) + " v0_identical_with_sqrt", Float64(perf_counter_ns() - t0) / 1e6)
        t0 = perf_counter_ns()
        ctx.enqueue_function[v4_identical_no_sqrt_kernel](
            dist.unsafe_ptr(), dq.unsafe_ptr(), dtrain.unsafe_ptr(), Int32(nq), Int32(n_train), Int32(d),
            grid_dim=(grid, 1, 1), block_dim=(256, 1, 1),
        )
        ctx.synchronize()
        _say("q500 rep=" + String(rep) + " v4_identical_no_sqrt", Float64(perf_counter_ns() - t0) / 1e6)
        t0 = perf_counter_ns()
        ctx.enqueue_function[v1_fma_no_ftz_kernel](
            dist.unsafe_ptr(), dq.unsafe_ptr(), dtrain.unsafe_ptr(), Int32(nq), Int32(n_train), Int32(d),
            grid_dim=(grid, 1, 1), block_dim=(256, 1, 1),
        )
        ctx.synchronize()
        _say("q500 rep=" + String(rep) + " v1_fma_no_ftz", Float64(perf_counter_ns() - t0) / 1e6)
        t0 = perf_counter_ns()
        ctx.enqueue_function[v2_ftz_plain_muladd_kernel](
            dist.unsafe_ptr(), dq.unsafe_ptr(), dtrain.unsafe_ptr(), Int32(nq), Int32(n_train), Int32(d),
            grid_dim=(grid, 1, 1), block_dim=(256, 1, 1),
        )
        ctx.synchronize()
        _say("q500 rep=" + String(rep) + " v2_ftz_plain_muladd", Float64(perf_counter_ns() - t0) / 1e6)
        t0 = perf_counter_ns()
        ctx.enqueue_function[v3_plain_kernel](
            dist.unsafe_ptr(), dq.unsafe_ptr(), dtrain.unsafe_ptr(), Int32(nq), Int32(n_train), Int32(d),
            grid_dim=(grid, 1, 1), block_dim=(256, 1, 1),
        )
        ctx.synchronize()
        _say("q500 rep=" + String(rep) + " v3_plain", Float64(perf_counter_ns() - t0) / 1e6)
    _ = dist^
    _ = dq^

    # ---- the tiled pipeline step by step, and T1, at 2000 queries ----
    nq = 2000
    var dq2 = _upload(ctx, _fixture(nq, d, UInt64(11)))
    cells = nq * n_train
    var chunk_rows = 1024
    var n_chunks = (n_train + chunk_rows - 1) // chunk_rows
    var logk = ctx.enqueue_create_buffer[DType.float32](cells)
    var part = ctx.enqueue_create_buffer[DType.float32](nq * n_chunks)
    var rowmax = ctx.enqueue_create_buffer[DType.float32](nq)
    var lse = ctx.enqueue_create_buffer[DType.float32](nq)
    var logw = ctx.enqueue_create_buffer[DType.float32](1)
    ctx.synchronize()
    var qtpbs: List[Int] = [256, 512]
    for rep in range(3):
        for qt in qtpbs:
            var qtpb = qt
            var t0 = perf_counter_ns()
            ctx.enqueue_function[kde_tiled_logk_kernel](
                logk.unsafe_ptr(), part.unsafe_ptr(), dq2.unsafe_ptr(), dtrain.unsafe_ptr(), logw.unsafe_ptr(),
                Int32(nq), Int32(n_train), Int32(d), Int32(n_chunks), Int32(chunk_rows), Int32(0), h,
                Int32(KDE_KERNEL_GAUSSIAN), Int32(DIST_L2_SQRT_UNEXPANDED),
                grid_dim=((nq + qtpb - 1) // qtpb, n_chunks, 1), block_dim=(qtpb, 1, 1),
            )
            ctx.synchronize()
            _say("q2000 rep=" + String(rep) + " t0_tiled_logk_q" + String(qtpb), Float64(perf_counter_ns() - t0) / 1e6)
            var h_t0 = UInt64(0)
            if rep == 0:
                h_t0 = _hash_dev(ctx, logk, cells)
            t0 = perf_counter_ns()
            ctx.enqueue_function[t1_simd_logk_kernel](
                logk.unsafe_ptr(), part.unsafe_ptr(), dq2.unsafe_ptr(), dtrain.unsafe_ptr(),
                Int32(nq), Int32(n_train), Int32(d), Int32(n_chunks), Int32(chunk_rows), h,
                grid_dim=((nq + qtpb - 1) // qtpb, n_chunks, 1), block_dim=(qtpb, 1, 1),
            )
            ctx.synchronize()
            _say("q2000 rep=" + String(rep) + " t1_simd_logk_q" + String(qtpb), Float64(perf_counter_ns() - t0) / 1e6)
            if rep == 0:
                var h_t1 = _hash_dev(ctx, logk, cells)
                print("KDE-ATTR q2000 q" + String(qtpb) + " t1_logk_equals_t0=" + String(h_t1 == h_t0) + " t0=" + String(h_t0) + " t1=" + String(h_t1))
        var t0 = perf_counter_ns()
        ctx.enqueue_function[kde_rowmax_reduce_kernel](
            rowmax.unsafe_ptr(), part.unsafe_ptr(), Int32(nq), Int32(n_chunks),
            grid_dim=((nq + 127) // 128, 1, 1), block_dim=(128, 1, 1),
        )
        ctx.synchronize()
        _say("q2000 rep=" + String(rep) + " rowmax_reduce", Float64(perf_counter_ns() - t0) / 1e6)
        var etpbs: List[Int] = [256, 1024]
        t0 = perf_counter_ns()
        ctx.enqueue_function[kde_lse_terms_kernel](
            logk.unsafe_ptr(), rowmax.unsafe_ptr(), Int32(nq), Int32(n_train),
            grid_dim=((cells + 255) // 256, 1, 1), block_dim=(256, 1, 1),
        )
        ctx.synchronize()
        _say("q2000 rep=" + String(rep) + " lse_terms_tpb256", Float64(perf_counter_ns() - t0) / 1e6)
        t0 = perf_counter_ns()
        ctx.enqueue_function[kde_lse_serial_sum_kernel](
            logk.unsafe_ptr(), rowmax.unsafe_ptr(), lse.unsafe_ptr(), Int32(nq), Int32(n_train),
            grid_dim=((nq + 127) // 128, 1, 1), block_dim=(128, 1, 1),
        )
        ctx.synchronize()
        _say("q2000 rep=" + String(rep) + " lse_serial_sum_tpb128", Float64(perf_counter_ns() - t0) / 1e6)
        t0 = perf_counter_ns()
        ctx.enqueue_function[kde_lse_serial_sum_kernel](
            logk.unsafe_ptr(), rowmax.unsafe_ptr(), lse.unsafe_ptr(), Int32(nq), Int32(n_train),
            grid_dim=((nq + 15) // 16, 1, 1), block_dim=(16, 1, 1),
        )
        ctx.synchronize()
        _say("q2000 rep=" + String(rep) + " lse_serial_sum_tpb16", Float64(perf_counter_ns() - t0) / 1e6)
        _ = etpbs^
    _ = logk^
    _ = part^
    _ = rowmax^
    _ = lse^
    _ = logw^
    _ = dq2^
    _ = dtrain^
    _ = ctx^
