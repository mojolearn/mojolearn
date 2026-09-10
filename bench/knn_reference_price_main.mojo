# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""IDENTICAL k-NN timing at the cuML reference grid, one shape per process.

Shape from the environment: MOJOLEARN_KNN_REF_INDEX (index rows),
MOJOLEARN_KNN_REF_QUERIES, MOJOLEARN_KNN_REF_K, MOJOLEARN_KNN_REF_FEATURES
(default 32), MOJOLEARN_KNN_REF_ROUNDS (default 7). Inputs are the dyadic
mixer `bench/knn_smallk_dispatch_fixture._coordinate` (index salt 0, query
salt 593), which `tools/knn_cuml_reference.py` reproduces in numpy so the
two sides search the same bytes.

Two timed regions per round, both after two untimed warmups:
  request   the public `knn_search` call: host pointers in, host pointers
            out, transfers and the host order pass included;
  device    device-resident: index and queries already on the GPU, norms +
            search + synchronize, outputs left on the GPU.
Every round's request output must match the warmup bytes. With
MOJOLEARN_KNN_REF_DUMP set, the request indices are written there as
little-endian UInt32, row-major (queries x k), for the cuML script to
compare neighbour lists row by row.
"""
from max.gpu.host import DeviceContext
from std.math import isfinite
from std.memory import bitcast
from std.os import getenv
from std.time import perf_counter_ns

from bench.knn_smallk_dispatch_fixture import _coordinate
from checks.kernel_matrix import TARGET_COLUMN, column_name
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, numeric_mode_name
from neighbors.estimator import DEFAULT_QUERY_TILE, knn_search, plan_query_tile
from neighbors.impl.neighbors.detail.knn_brute_force import (
    EXPERIMENTAL_KNN_TRANSPOSE_IDENTICAL,
    EXPERIMENTAL_SMALLK_IDENTICAL,
    KNN_INDEX_TILE_IDENTICAL,
    KNN_METHOD_AUTO,
    KNN_PHASE_TIMERS,
    KNN_REGISTER_TILE_IDENTICAL,
    METRIC_FROM_IS_SQRT,
    brute_force_knn_impl,
    compute_norms_for_metric,
    identical_index_tile,
    resolve_metric,
)


def _env_int(name: String, default: Int) raises -> Int:
    var s = String(getenv(name))
    if s == "":
        return default
    return Int(atol(s))


def _median(var xs: List[Float64]) -> Float64:
    var n = len(xs)
    for i in range(1, n):
        var v = xs[i]
        var j = i - 1
        while j >= 0 and xs[j] > v:
            xs[j + 1] = xs[j]
            j -= 1
        xs[j + 1] = v
    if n % 2 == 1:
        return xs[n // 2]
    return (xs[n // 2 - 1] + xs[n // 2]) / 2.0


def main() raises:
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("knn reference price requires IDENTICAL mode")
    var n_index = _env_int("MOJOLEARN_KNN_REF_INDEX", 100000)
    var n_queries = _env_int("MOJOLEARN_KNN_REF_QUERIES", 1000)
    var k = _env_int("MOJOLEARN_KNN_REF_K", 10)
    var d = _env_int("MOJOLEARN_KNN_REF_FEATURES", 32)
    var rounds = _env_int("MOJOLEARN_KNN_REF_ROUNDS", 7)
    var dump = String(getenv("MOJOLEARN_KNN_REF_DUMP"))
    if n_index < k or n_queries < 1 or k < 1 or d < 1 or rounds < 1:
        raise Error("knn reference price: invalid shape")
    print(
        "KNN_REF_HEADER", "mode", numeric_mode_name(), "column",
        column_name(TARGET_COLUMN), "index", n_index, "queries", n_queries,
        "features", d, "k", k, "rounds", rounds, "fixture", "dyadic-v1",
        "selector", Int(EXPERIMENTAL_SMALLK_IDENTICAL),
        "transpose", Int(EXPERIMENTAL_KNN_TRANSPOSE_IDENTICAL),
        "register_tile", Int(KNN_REGISTER_TILE_IDENTICAL),
        "index_tile", KNN_INDEX_TILE_IDENTICAL,
    )
    with DeviceContext() as ctx:
        var index = ctx.enqueue_create_host_buffer[DType.float32](n_index * d)
        var queries = ctx.enqueue_create_host_buffer[DType.float32](n_queries * d)
        var distances = ctx.enqueue_create_host_buffer[DType.float32](n_queries * k)
        var indices = ctx.enqueue_create_host_buffer[DType.uint32](n_queries * k)
        ctx.synchronize()
        for row in range(n_index):
            for f in range(d):
                index.unsafe_ptr().unsafe_store(row * d + f, _coordinate(row, f, 0))
        for row in range(n_queries):
            for f in range(d):
                queries.unsafe_ptr().unsafe_store(row * d + f, _coordinate(row, f, 593))

        # ---- request region: the public call ------------------------------
        var expected_d = List[UInt32]()
        var expected_i = List[UInt32]()
        var request_ms = List[Float64]()
        for r in range(rounds + 2):
            for cell in range(n_queries * k):
                distances.unsafe_ptr().unsafe_store(cell, Float32(-12345))
                indices.unsafe_ptr().unsafe_store(cell, UInt32(4294967295))
            ctx.synchronize()
            var begin = perf_counter_ns()
            _ = knn_search(
                ctx, index.unsafe_ptr(), n_index, queries.unsafe_ptr(), n_queries,
                d, k, distances.unsafe_ptr(), indices.unsafe_ptr(),
                return_sqrt=True,
            )
            ctx.synchronize()
            var elapsed = Float64(perf_counter_ns() - begin) / 1000000.0
            for cell in range(n_queries * k):
                var distance = distances.unsafe_ptr().unsafe_load(cell)
                var neighbor = indices.unsafe_ptr().unsafe_load(cell)
                if not isfinite(distance) or distance < Float32(0) or neighbor >= UInt32(n_index):
                    raise Error("knn reference price: invalid or poisoned output")
                var raw = bitcast[DType.uint32](distance)
                if r == 0:
                    expected_d.append(raw)
                    expected_i.append(neighbor)
                elif raw != expected_d[cell] or neighbor != expected_i[cell]:
                    raise Error("knn reference price: request output bytes moved between rounds")
            if r >= 2:
                request_ms.append(elapsed)
                print("KNN_REF_ROUND", "request", r - 1, elapsed)
            else:
                print("KNN_REF_WARMUP", "request", r, elapsed)

        # ---- device region: everything but the transfers and the host sort --
        var query_tile = plan_query_tile(n_index, n_queries, DEFAULT_QUERY_TILE)
        var buf_len = n_index // 8
        if buf_len < k:
            buf_len = k
        var d_index = ctx.enqueue_create_buffer[DType.float32](n_index * d)
        var d_queries = ctx.enqueue_create_buffer[DType.float32](n_queries * d)
        var d_index_norm = ctx.enqueue_create_buffer[DType.float32](n_index)
        var d_query_norm = ctx.enqueue_create_buffer[DType.float32](n_queries)
        var d_dist_tile = ctx.enqueue_create_buffer[DType.float32](
            query_tile * identical_index_tile(n_index)
        )
        var d_buf_val = ctx.enqueue_create_buffer[DType.float32](query_tile * 2 * buf_len)
        var d_buf_idx = ctx.enqueue_create_buffer[DType.uint32](query_tile * 2 * buf_len)
        var d_out_dist = ctx.enqueue_create_buffer[DType.float32](n_queries * k)
        var d_out_idx = ctx.enqueue_create_buffer[DType.uint32](n_queries * k)
        var d_out_i32 = ctx.enqueue_create_buffer[DType.int32](n_queries * k)
        ctx.synchronize()
        ctx.enqueue_copy(dst_buf=d_index, src_ptr=index.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=d_queries, src_ptr=queries.unsafe_ptr())
        ctx.synchronize()
        var mtr = resolve_metric(METRIC_FROM_IS_SQRT, True)
        var device_ms = List[Float64]()
        var hd = ctx.enqueue_create_host_buffer[DType.float32](n_queries * k)
        var hi = ctx.enqueue_create_host_buffer[DType.uint32](n_queries * k)
        for r in range(rounds + 2):
            ctx.synchronize()
            var begin = perf_counter_ns()
            compute_norms_for_metric(ctx, d_index, d_index_norm, n_index, d, mtr)
            compute_norms_for_metric(ctx, d_queries, d_query_norm, n_queries, d, mtr)
            comptime if KNN_PHASE_TIMERS:
                # `-D MOJOLEARN_KNN_PHASE_TIMERS=1`: the norms are the one
                # launch class outside `tiled_brute_force_knn`, so they are
                # timed here; the tiled arm prints the other classes itself.
                ctx.synchronize()
                print(
                    "KNN_PHASE_TIMERS", "norms_ms",
                    Float64(perf_counter_ns() - begin) / 1000000.0,
                )
            brute_force_knn_impl(
                ctx, d_queries, d_query_norm, d_index, d_index_norm, d_dist_tile,
                d_buf_val, d_buf_idx, d_out_dist, d_out_idx, d_out_i32,
                n_queries, n_index, d, k, query_tile, buf_len, True,
                False, True, True, KNN_METHOD_AUTO, mtr, Float32(2.0),
            )
            ctx.synchronize()
            var elapsed = Float64(perf_counter_ns() - begin) / 1000000.0
            if r >= 2:
                device_ms.append(elapsed)
                print("KNN_REF_ROUND", "device", r - 1, elapsed)
            else:
                print("KNN_REF_WARMUP", "device", r, elapsed)
        # The device region's answer is the request's answer: under IDENTICAL
        # the selectors already write ascending (distance, index), so the
        # pre-sort device output equals the sorted request output bit for bit.
        ctx.enqueue_copy(dst_ptr=hd.unsafe_ptr(), src_buf=d_out_dist)
        ctx.enqueue_copy(dst_ptr=hi.unsafe_ptr(), src_buf=d_out_idx)
        ctx.synchronize()
        var device_mismatch = 0
        for cell in range(n_queries * k):
            if (
                bitcast[DType.uint32](hd.unsafe_ptr().unsafe_load(cell)) != expected_d[cell]
                or hi.unsafe_ptr().unsafe_load(cell) != expected_i[cell]
            ):
                device_mismatch += 1
        print("KNN_REF_DEVICE_VS_REQUEST", "mismatched_cells", device_mismatch)

        # ---- the fingerprint and the dump ---------------------------------
        var h = UInt64(1469598103934665603)
        var xd = UInt32(0)
        for cell in range(n_queries * k):
            h = (h ^ UInt64(expected_i[cell])) * UInt64(1099511628211)
            xd = xd ^ expected_d[cell]
        print("KNN_REF_FINGERPRINT", "index_fnv1a64", h, "distance_bits_xor", xd)
        for slot in range(k):
            print("KNN_REF_ROW0", slot, expected_i[slot], expected_d[slot])
        if dump != "":
            var bytes = List[UInt8]()
            for cell in range(n_queries * k):
                var v = expected_i[cell]
                bytes.append(UInt8(v & UInt32(255)))
                bytes.append(UInt8((v >> 8) & UInt32(255)))
                bytes.append(UInt8((v >> 16) & UInt32(255)))
                bytes.append(UInt8((v >> 24) & UInt32(255)))
            with open(dump, "w") as fh:
                fh.write_bytes(Span(bytes))
            print("KNN_REF_DUMP", dump, n_queries * k)

        # Optional exact A/B artifact outside both timed regions: interleaved
        # index and distance UInt32 words, little endian, all selected cells.
        var full_dump = String(getenv("MOJOLEARN_KNN_REF_DUMP_FULL"))
        if full_dump != "":
            var words = List[UInt8]()
            for cell in range(n_queries * k):
                for field in range(2):
                    var word = expected_i[cell] if field == 0 else expected_d[cell]
                    for byte in range(4):
                        words.append(UInt8((word >> UInt32(8 * byte)) & UInt32(255)))
            with open(full_dump, "w") as fh:
                fh.write_bytes(Span(words))
            print("KNN_REF_DUMP_FULL", full_dump, n_queries * k)

        print(
            "KNN_REF_RESULT", "index", n_index, "queries", n_queries, "k", k,
            "features", d, "request_median_ms", _median(request_ms.copy()),
            "device_median_ms", _median(device_ms.copy()), "rounds", rounds,
            "query_tile", query_tile, "index_tile", identical_index_tile(n_index),
        )
        _ = index^
        _ = queries^
        _ = distances^
        _ = indices^
        _ = hd^
        _ = hi^
        _ = d_index^
        _ = d_queries^
        _ = d_index_norm^
        _ = d_query_norm^
        _ = d_dist_tile^
        _ = d_buf_val^
        _ = d_buf_idx^
        _ = d_out_dist^
        _ = d_out_idx^
        _ = d_out_i32^
    print("KNN REFERENCE PRICE PASS")
