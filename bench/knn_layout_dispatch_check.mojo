# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Main-run four-arm layout gate; no measurements or host search oracle.

The existing public dispatch suite covers K4/8/10/15/16, query counts
1/257/1000, duplicate/dyadic inputs and tile offsets. Additional cases here
exercise explicit L2Expanded despite return_sqrt=True, L2SqrtExpanded,
and noncandidate L1/Cosine fallback metrics at feature dimension17 and index
65 (transpose tails), query257 and K10. Compare DISPATCH_CELL and LAYOUT_CELL
records across baseline/selector-only/transpose-only/both and vendors. Each
additional case repeats with different query tiles and requires exact bits.
Compile in IDENTICAL with each combination of the two experimental defines;
LAYOUT_FLAGS must report the requested combination. This supplements rather
than replaces candidate component cancellation/subnormal/transpose-bit gates.
"""
from max.gpu.host import DeviceContext
from std.math import isfinite
from std.memory import bitcast
from bench.knn_smallk_dispatch_fixture import run_check, _coordinate
from checks.numerics import numeric_mode_name
from neighbors.estimator import knn_search
from neighbors.impl.neighbors.detail.knn_brute_force import EXPERIMENTAL_SMALLK_IDENTICAL, EXPERIMENTAL_KNN_TRANSPOSE_IDENTICAL
from neighbors.impl.distance.detail.distance_ops import DIST_L2_EXPANDED, DIST_L2_SQRT_EXPANDED, DIST_L1, DIST_COSINE_EXPANDED


def _metric_case(metric: Int) raises:
    comptime N = 65
    comptime Q = 257
    comptime D = 17
    comptime K = 10
    with DeviceContext() as ctx:
        var index = ctx.enqueue_create_host_buffer[DType.float32](N * D)
        var queries = ctx.enqueue_create_host_buffer[DType.float32](Q * D)
        var distances = ctx.enqueue_create_host_buffer[DType.float32](Q * K)
        var indices = ctx.enqueue_create_host_buffer[DType.uint32](Q * K)
        ctx.synchronize()
        for row in range(N):
            for f in range(D):
                index.unsafe_ptr().unsafe_store(row * D + f, _coordinate(row, f, 0))
        for row in range(Q):
            for f in range(D):
                queries.unsafe_ptr().unsafe_store(row * D + f, _coordinate(row, f, 593))
        var expected_d = List[UInt32]()
        var expected_i = List[UInt32]()
        for repeat in range(2):
            for cell in range(Q * K):
                distances.unsafe_ptr().unsafe_store(cell, Float32(-12345))
                indices.unsafe_ptr().unsafe_store(cell, UInt32(4294967295))
            _ = knn_search(
                ctx, index.unsafe_ptr(), N, queries.unsafe_ptr(), Q, D, K,
                distances.unsafe_ptr(), indices.unsafe_ptr(), return_sqrt=True,
                requested_query_tile=128 if repeat == 0 else 64, metric=metric,
            )
            ctx.synchronize()
            for cell in range(Q * K):
                var value = distances.unsafe_ptr().unsafe_load(cell)
                var neighbor = indices.unsafe_ptr().unsafe_load(cell)
                if not isfinite(value) or value == Float32(-12345) or neighbor >= UInt32(N):
                    raise Error("layout metric gate returned invalid output or poison")
                var bits = bitcast[DType.uint32](value)
                if repeat == 0:
                    expected_d.append(bits)
                    expected_i.append(neighbor)
                    print("LAYOUT_CELL", metric, N, Q, D, K, cell, bits, neighbor)
                elif bits != expected_d[cell] or neighbor != expected_i[cell]:
                    raise Error("layout metric output changed across query tiles")
        _ = index^
        _ = queries^
        _ = distances^
        _ = indices^


def main() raises:
    print("LAYOUT_FLAGS", "mode", numeric_mode_name(),
          "selector", Int(EXPERIMENTAL_SMALLK_IDENTICAL),
          "transpose", Int(EXPERIMENTAL_KNN_TRANSPOSE_IDENTICAL))
    run_check()
    _metric_case(DIST_L2_EXPANDED)
    _metric_case(DIST_L2_SQRT_EXPANDED)
    _metric_case(DIST_L1)
    _metric_case(DIST_COSINE_EXPANDED)
    print("KNN LAYOUT PUBLIC DISPATCH PASS", "metric_fixtures", 4,
          "additional_selected_pairs", 10280)
