# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Main-run native public k-NN request timing, one sample per process.

Build IDENTICAL binaries with and without
-D MOJOLEARN_EXPERIMENTAL_SMALLK_IDENTICAL=1. Set
MOJOLEARN_SMALLK_PRICE_QUERIES to 32 (default), 128 or 1000.
Index rows=100000, features=32, K=10 are fixed, bounding the fixture.
Each process prepares deterministic dyadic host inputs, performs two warmups,
then times ONE public knn_search plus context synchronization. The public API
includes input uploads, search and output downloads. Host fixture preparation,
context construction, output validation/printing and warmups are excluded.
This is neither a Python-host request benchmark nor a CUDA/cuML comparison.

Main should rotate the two binaries for at least nine invocations per shape,
compare every PRICE_CELL record, and require completion markers before using
the PRICE_MS sample. This driver intentionally has no timing-round loop.
"""
from max.gpu.host import DeviceContext
from std.math import isfinite
from std.memory import bitcast
from std.os import getenv
from std.time import perf_counter_ns
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, numeric_mode_name
from neighbors.estimator import knn_search
from neighbors.impl.neighbors.detail.knn_brute_force import EXPERIMENTAL_SMALLK_IDENTICAL


def _coordinate(row: Int, feature: Int, salt: Int) -> Float32:
    # Same fixed integer mixer and dyadic conversion as dispatch_check.
    # The salts (index=0, query=593) completely specify this fixture's seed.
    var value = UInt32(row + 1) * UInt32(747796405) + UInt32(feature * 131 + salt)
    value = (value ^ (value >> 16)) * UInt32(2246822519)
    value = (value ^ (value >> 13)) * UInt32(3266489917)
    value = value ^ (value >> 16)
    return Float32(Int(value % UInt32(2048)) - 1024) / Float32(1024)


def run_price() raises:
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("public small-k request timing requires IDENTICAL mode")
    var query_env = String(getenv("MOJOLEARN_SMALLK_PRICE_QUERIES"))
    var n_queries = 32
    if query_env == "128":
        n_queries = 128
    elif query_env == "1000":
        n_queries = 1000
    elif query_env != "" and query_env != "32":
        raise Error("MOJOLEARN_SMALLK_PRICE_QUERIES must be 32, 128 or 1000")
    comptime N_INDEX = 100000
    comptime D = 32
    comptime K = 10
    print("SMALLK_PRICE", "experimental", Int(EXPERIMENTAL_SMALLK_IDENTICAL),
          "mode", numeric_mode_name(), "fixture", "dyadic-v1", "index", N_INDEX,
          "queries", n_queries, "features", D, "k", K,
          "index_salt", 0, "query_salt", 593, "requested_tile", 256,
          "warmups", 2, "timed_calls", 1,
          "scope", "native-public-upload-search-download-synchronize")
    with DeviceContext() as ctx:
        var index = ctx.enqueue_create_host_buffer[DType.float32](N_INDEX * D)
        var queries = ctx.enqueue_create_host_buffer[DType.float32](n_queries * D)
        var distances = ctx.enqueue_create_host_buffer[DType.float32](n_queries * K)
        var indices = ctx.enqueue_create_host_buffer[DType.uint32](n_queries * K)
        ctx.synchronize()
        for row in range(N_INDEX):
            for f in range(D):
                index.unsafe_ptr().unsafe_store(row * D + f, _coordinate(row, f, 0))
        for row in range(n_queries):
            for f in range(D):
                queries.unsafe_ptr().unsafe_store(row * D + f, _coordinate(row, f, 593))
        var expected_d = List[UInt32]()
        var expected_i = List[UInt32]()
        for warmup in range(2):
            for cell in range(n_queries * K):
                distances.unsafe_ptr().unsafe_store(cell, Float32(-12345))
                indices.unsafe_ptr().unsafe_store(cell, UInt32(4294967295))
            _ = knn_search(
                ctx, index.unsafe_ptr(), N_INDEX, queries.unsafe_ptr(), n_queries,
                D, K, distances.unsafe_ptr(), indices.unsafe_ptr(),
                return_sqrt=True, requested_query_tile=256,
            )
            ctx.synchronize()
            for cell in range(n_queries * K):
                var distance = distances.unsafe_ptr().unsafe_load(cell)
                var neighbor = indices.unsafe_ptr().unsafe_load(cell)
                if not isfinite(distance) or distance < Float32(0) or neighbor >= UInt32(N_INDEX):
                    raise Error("public k-NN warmup returned invalid or poisoned output")
                var raw = bitcast[DType.uint32](distance)
                if warmup == 0:
                    expected_d.append(raw)
                    expected_i.append(neighbor)
                elif raw != expected_d[cell] or neighbor != expected_i[cell]:
                    raise Error("public k-NN warmup output bytes changed")
        for cell in range(n_queries * K):
            distances.unsafe_ptr().unsafe_store(cell, Float32(-12345))
            indices.unsafe_ptr().unsafe_store(cell, UInt32(4294967295))
        ctx.synchronize()
        var begin = perf_counter_ns()
        var used_tile = knn_search(
            ctx, index.unsafe_ptr(), N_INDEX, queries.unsafe_ptr(), n_queries,
            D, K, distances.unsafe_ptr(), indices.unsafe_ptr(),
            return_sqrt=True, requested_query_tile=256,
        )
        ctx.synchronize()
        var elapsed = Float64(perf_counter_ns() - begin) / Float64(1000000)
        for cell in range(n_queries * K):
            var distance = distances.unsafe_ptr().unsafe_load(cell)
            var neighbor = indices.unsafe_ptr().unsafe_load(cell)
            if not isfinite(distance) or distance < Float32(0) or neighbor >= UInt32(N_INDEX):
                raise Error("public k-NN timed call returned invalid or poisoned output")
            var raw = bitcast[DType.uint32](distance)
            if raw != expected_d[cell] or neighbor != expected_i[cell]:
                raise Error("public k-NN timed output differs from warmup bytes")
            print("PRICE_CELL", "dyadic-v1", N_INDEX, n_queries, D, K, cell, raw, neighbor)
        print("PRICE_MS", "dyadic-v1", N_INDEX, n_queries, D, K, elapsed,
              "used_tile", used_tile)
        _ = index^
        _ = queries^
        _ = distances^
        _ = indices^
    print("KNN SMALL-K PUBLIC PRICE PASS", "queries", n_queries,
          "selected_pairs", n_queries * K)
