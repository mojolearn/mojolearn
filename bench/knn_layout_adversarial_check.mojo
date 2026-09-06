# SPDX-License-Identifier: Apache-2.0
"""Four-arm IDENTICAL correctness only: non-dyadic, duplicate and offset data.

Compare every ADVERSARIAL_CELL across baseline/selector/transpose/both and
vendors. Each case also requires exact query-tile invariance and sorted keys.
No timing or host search is performed.
"""
from max.gpu.host import DeviceContext
from std.math import isfinite
from std.memory import bitcast
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, numeric_mode_name
from neighbors.estimator import knn_search
from neighbors.impl.neighbors.detail.knn_brute_force import EXPERIMENTAL_SMALLK_IDENTICAL, EXPERIMENTAL_KNN_TRANSPOSE_IDENTICAL
from neighbors.impl.distance.detail.distance_ops import DIST_L2_EXPANDED, DIST_L2_SQRT_EXPANDED, DIST_L1, DIST_COSINE_EXPANDED


def coordinate(row: Int, feature: Int, profile: Int, query: Bool) -> Float32:
    # Duplicates are intentional. Query shifts prevent an all-self fixture.
    var r = row % 37
    var numerator = (r * 43 + feature * 17 + (5 if query else 0)) % 101 - 50
    var value = Float32(numerator) / Float32(7 + feature % 5)
    if profile == 1:
        value = Float32(10000) + value / Float32(13)
    elif profile == 2:
        # Heterogeneous feature scales amplify conditioning differences.
        value = value * (Float32(0.0001) if feature % 3 == 0 else Float32(100))
    return value


def run_case(dimension: Int, profile: Int, metric: Int) raises:
    comptime N = 129
    comptime Q = 17
    comptime K = 10
    with DeviceContext() as ctx:
        var index = ctx.enqueue_create_host_buffer[DType.float32](N * dimension)
        var queries = ctx.enqueue_create_host_buffer[DType.float32](Q * dimension)
        var distances = ctx.enqueue_create_host_buffer[DType.float32](Q * K)
        var indices = ctx.enqueue_create_host_buffer[DType.uint32](Q * K)
        ctx.synchronize()
        for row in range(N):
            for feature in range(dimension):
                index.unsafe_ptr().unsafe_store(row * dimension + feature, coordinate(row, feature, profile, False))
        for row in range(Q):
            for feature in range(dimension):
                queries.unsafe_ptr().unsafe_store(row * dimension + feature, coordinate(row, feature, profile, True))
        var expected_d = List[UInt32]()
        var expected_i = List[UInt32]()
        for repeat in range(2):
            for cell in range(Q * K):
                distances.unsafe_ptr().unsafe_store(cell, Float32(-12345))
                indices.unsafe_ptr().unsafe_store(cell, UInt32(4294967295))
            _ = knn_search(ctx, index.unsafe_ptr(), N, queries.unsafe_ptr(), Q, dimension, K,
                           distances.unsafe_ptr(), indices.unsafe_ptr(),
                           requested_query_tile=8 if repeat == 0 else 16, metric=metric)
            ctx.synchronize()
            for cell in range(Q * K):
                var value = distances.unsafe_ptr().unsafe_load(cell)
                var neighbor = indices.unsafe_ptr().unsafe_load(cell)
                if not isfinite(value) or value == Float32(-12345) or neighbor >= UInt32(N):
                    raise Error("adversarial kNN invalid output")
                if cell % K > 0:
                    var previous = distances.unsafe_ptr().unsafe_load(cell - 1)
                    var previous_index = indices.unsafe_ptr().unsafe_load(cell - 1)
                    if value < previous or (value == previous and neighbor <= previous_index):
                        raise Error("adversarial kNN distance/index keys are not strictly ordered")
                var bits = bitcast[DType.uint32](value)
                if repeat == 0:
                    expected_d.append(bits)
                    expected_i.append(neighbor)
                    print("ADVERSARIAL_CELL", dimension, profile, metric, cell, bits, neighbor)
                elif expected_d[cell] != bits or expected_i[cell] != neighbor:
                    raise Error("adversarial kNN query-tile bits changed")
        _ = index^
        _ = queries^
        _ = distances^
        _ = indices^


def main() raises:
    if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("adversarial bitwise gate requires IDENTICAL")
    print("ADVERSARIAL_FLAGS", numeric_mode_name(), Int(EXPERIMENTAL_SMALLK_IDENTICAL), Int(EXPERIMENTAL_KNN_TRANSPOSE_IDENTICAL))
    var dimensions: List[Int] = [1, 3, 17, 33, 65]
    for dimension in dimensions:
        for profile in range(3):
            run_case(dimension, profile, DIST_L2_EXPANDED)
            run_case(dimension, profile, DIST_L2_SQRT_EXPANDED)
    run_case(17, 0, DIST_L1)
    run_case(17, 0, DIST_COSINE_EXPANDED)
    print("KNN ADVERSARIAL PASS", "cases", 32, "selected_pairs", 5440)
