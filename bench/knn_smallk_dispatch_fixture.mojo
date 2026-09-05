# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Main-run public k-NN comparison driver; no timing or host search oracle.

Compile in IDENTICAL twice, with and without
-D MOJOLEARN_EXPERIMENTAL_SMALLK_IDENTICAL=1. Compare every DISPATCH_CELL
record between binaries and vendors; the activation header must differ
between baseline and experiment. Both call public knn_search with AUTO.
K8/10/16 reach the armed specialization; K4/15 remain radix fallbacks.
The second call changes query_tile from256 to128 and compares all bits,
covering nonzero output offsets and final partial tiles at257/1000 queries.
Maximum index1025, features8, querycount1000; contexts run sequentially.
"""
from max.gpu.host import DeviceContext
from std.math import isfinite
from std.memory import bitcast
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, numeric_mode_name
from neighbors.estimator import knn_search
from neighbors.checks.select_radix_identical import composite_key
from neighbors.impl.neighbors.detail.knn_brute_force import EXPERIMENTAL_SMALLK_IDENTICAL


def _coordinate(row: Int, feature: Int, salt: Int) -> Float32:
    # Integer mixing followed by an exact dyadic conversion. No host RNG,
    # transcendental, parallel reduction or vendor-dependent fixture inputs.
    var value = UInt32(row + 1) * UInt32(747796405) + UInt32(feature * 131 + salt)
    value = (value ^ (value >> 16)) * UInt32(2246822519)
    value = (value ^ (value >> 13)) * UInt32(3266489917)
    value = value ^ (value >> 16)
    return Float32(Int(value % UInt32(2048)) - 1024) / Float32(1024)


def _case(profile: Int, n_queries: Int, k: Int) raises:
    var n_index = 257 if profile == 0 else 1025
    comptime D = 8
    var name = String("dyadic") if profile == 0 else String("duplicates")
    with DeviceContext() as ctx:
        var index = ctx.enqueue_create_host_buffer[DType.float32](n_index * D)
        var queries = ctx.enqueue_create_host_buffer[DType.float32](n_queries * D)
        var distances = ctx.enqueue_create_host_buffer[DType.float32](n_queries * k)
        var indices = ctx.enqueue_create_host_buffer[DType.uint32](n_queries * k)
        ctx.synchronize()
        for row in range(n_index):
            var source = row if profile == 0 else row % 17
            for f in range(D):
                index.unsafe_ptr().unsafe_store(row * D + f, _coordinate(source, f, 0))
        for row in range(n_queries):
            var source = row if profile == 0 else (row * 7) % 17
            var salt = 593 if profile == 0 else 0
            for f in range(D):
                queries.unsafe_ptr().unsafe_store(row * D + f, _coordinate(source, f, salt))
        var expected_d = List[UInt32]()
        var expected_i = List[UInt32]()
        for repeat in range(2):
            for cell in range(n_queries * k):
                distances.unsafe_ptr().unsafe_store(cell, Float32(-12345))
                indices.unsafe_ptr().unsafe_store(cell, UInt32(4294967295))
            var requested_tile = 256 if repeat == 0 else 128
            var used_tile = knn_search(
                ctx, index.unsafe_ptr(), n_index, queries.unsafe_ptr(), n_queries,
                D, k, distances.unsafe_ptr(), indices.unsafe_ptr(),
                return_sqrt=True, requested_query_tile=requested_tile,
            )
            ctx.synchronize()
            print("DISPATCH_CALL", name, n_index, n_queries, D, k, repeat, "tile", used_tile)
            for cell in range(n_queries * k):
                var distance = distances.unsafe_ptr().unsafe_load(cell)
                var index_id = indices.unsafe_ptr().unsafe_load(cell)
                if not isfinite(distance) or distance < Float32(0) or index_id >= UInt32(n_index):
                    raise Error("public k-NN left poison or returned invalid output")
                var raw = bitcast[DType.uint32](distance)
                if cell % k > 0:
                    var before = composite_key(distances.unsafe_ptr().unsafe_load(cell - 1),
                                               indices.unsafe_ptr().unsafe_load(cell - 1), True)
                    if composite_key(distance, index_id, True) <= before:
                        raise Error("public k-NN output lost composite-key ordering")
                if repeat == 0:
                    expected_d.append(raw)
                    expected_i.append(index_id)
                    print("DISPATCH_CELL", name, n_index, n_queries, D, k, cell, raw, index_id)
                elif raw != expected_d[cell] or index_id != expected_i[cell]:
                    raise Error("public k-NN changed bytes across repeated query tiling")
        # Caller buffers remain input data, including the duplicated rows.
        for row in range(n_index):
            var source = row if profile == 0 else row % 17
            for f in range(D):
                if bitcast[DType.uint32](index.unsafe_ptr().unsafe_load(row * D + f)) != bitcast[DType.uint32](_coordinate(source, f, 0)):
                    raise Error("public k-NN mutated index input")
        for row in range(n_queries):
            var source = row if profile == 0 else (row * 7) % 17
            var salt = 593 if profile == 0 else 0
            for f in range(D):
                if bitcast[DType.uint32](queries.unsafe_ptr().unsafe_load(row * D + f)) != bitcast[DType.uint32](_coordinate(source, f, salt)):
                    raise Error("public k-NN mutated query input")
        _ = index^
        _ = queries^
        _ = distances^
        _ = indices^
    print("DISPATCH_FIXTURE_PASS", name, n_index, n_queries, D, k)


def run_check() raises:
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("public small-k dispatch gate requires IDENTICAL mode")
    print("SMALLK_DISPATCH", "experimental", Int(EXPERIMENTAL_SMALLK_IDENTICAL),
          "mode", numeric_mode_name(), "specialized_k", "8,10,16", "fallback_k", "4,15")
    for profile in range(2):
        for size in range(3):
            var queries = 1
            if size == 1:
                queries = 257
            elif size == 2:
                queries = 1000
            _case(profile, queries, 4)
            _case(profile, queries, 8)
            _case(profile, queries, 10)
            _case(profile, queries, 15)
            _case(profile, queries, 16)
    print("KNN SMALL-K PUBLIC DISPATCH PASS", "fixtures", 30, "selected_pairs", 133348)
