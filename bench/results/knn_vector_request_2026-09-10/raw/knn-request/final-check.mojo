# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Complete public-request scalar/vector trial, same process, alternating order.

Build with MOJOLEARN_KNN_VECTOR_REQUEST_CHECK=1 to exercise all three arms.
Both arms use the same public API, allocations and full-index stride.
Environment mutation and every-output comparison are outside timing.
Large promotion targets: 400000 index / 4000 queries / d32, k10 and k15.
Small or ragged fixtures qualify correctness only. Arm 0 scalar, arm 1 vector,
arm 2 actual scoped default. Explicit vector requests run only
when stride, partition offset and partition width permit aligned loads.
"""
from max.gpu.host import DeviceContext
from std.math import isfinite
from std.memory import bitcast
from std.os import getenv
from std.time import perf_counter_ns
from std.python import Python
from std.sys.compile import is_defined

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
    KNN_PREFLIGHT_METADATA,
    KNN_PREFLIGHT_METADATA_DEFAULT,
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
    comptime if not is_defined["MOJOLEARN_KNN_VECTOR_REQUEST_CHECK"]():
        raise Error("build with MOJOLEARN_KNN_VECTOR_REQUEST_CHECK=1")
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("knn reference price requires IDENTICAL mode")
    var n_index = _env_int("MOJOLEARN_KNN_REF_INDEX", 100000)
    var n_queries = _env_int("MOJOLEARN_KNN_REF_QUERIES", 1000)
    var k = _env_int("MOJOLEARN_KNN_REF_K", 10)
    var d = _env_int("MOJOLEARN_KNN_REF_FEATURES", 32)
    var rounds = _env_int("MOJOLEARN_KNN_REF_ROUNDS", 7)
    if n_index < k or n_queries < 1 or k < 1 or d < 1 or rounds < 1:
        raise Error("knn reference price: invalid shape")
    print(
        "KNN_REF_HEADER", "mode", numeric_mode_name(), "column",
        column_name(TARGET_COLUMN), "index", n_index, "queries", n_queries,
        "features", d, "k", k, "rounds", rounds, "fixture", "dyadic-v1",
        "selector", Int(EXPERIMENTAL_SMALLK_IDENTICAL),
        "transpose", Int(EXPERIMENTAL_KNN_TRANSPOSE_IDENTICAL),
        "register_tile", Int(KNN_REGISTER_TILE_IDENTICAL),
        "metadata_forced", Int(KNN_PREFLIGHT_METADATA),
        "metadata_default_capable", Int(KNN_PREFLIGHT_METADATA_DEFAULT),
        "index_tile", KNN_INDEX_TILE_IDENTICAL,
    )
    var os = Python.import_module("os")
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
            for turn in range(3):
                var arm = (r + turn) % 3
                os.environ["MOJOLEARN_KNN_VECTOR_TRIAL"] = String(arm)
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
                    if r == 0 and turn == 0:
                        expected_d.append(raw)
                        expected_i.append(neighbor)
                    elif raw != expected_d[cell] or neighbor != expected_i[cell]:
                        raise Error("knn reference price: request output bytes moved between rounds")
                if r >= 2:
                    request_ms.append(elapsed)
                    print("KNN_REF_ROUND", "request", r - 1, "arm", arm, elapsed)
                else:
                    print("KNN_REF_WARMUP", "request", r, "arm", arm, elapsed)

        print("FULL_OUTPUT_BITS_MATCH", n_queries * k, "per arm per round")

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

