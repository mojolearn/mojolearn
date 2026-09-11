# SPDX-License-Identifier: Apache-2.0
"""Exact output invariance when a request crosses the 512-query batch boundary."""
from max.gpu.host import DeviceContext
from std.memory import bitcast
from bench.knn_smallk_dispatch_fixture import _coordinate
from neighbors.estimator import knn_search, plan_query_tile, DEFAULT_QUERY_TILE, QUERY_TILE_512_CANDIDATE
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL


def _case(n: Int, q: Int, d: Int, k: Int) raises:
    with DeviceContext() as ctx:
        var index = ctx.enqueue_create_host_buffer[DType.float32](n * d)
        var queries = ctx.enqueue_create_host_buffer[DType.float32](q * d)
        var distances = ctx.enqueue_create_host_buffer[DType.float32](q * k)
        var indices = ctx.enqueue_create_host_buffer[DType.uint32](q * k)
        ctx.synchronize()
        for row in range(n):
            for f in range(d):
                # Repeated index rows create composite-key tie boundaries.
                index.unsafe_ptr().unsafe_store(row * d + f, _coordinate(row % 257, f, 0))
        for row in range(q):
            for f in range(d):
                queries.unsafe_ptr().unsafe_store(row * d + f, _coordinate(row, f, 593))
        var expected = List[UInt32]()
        for arm in range(2):
            var used = knn_search(
                ctx, index.unsafe_ptr(), n, queries.unsafe_ptr(), q, d, k,
                distances.unsafe_ptr(), indices.unsafe_ptr(),
                requested_query_tile=256 if arm == 0 else DEFAULT_QUERY_TILE,
            )
            ctx.synchronize()
            for cell in range(q * k):
                var ix = indices.unsafe_ptr().unsafe_load(cell)
                var bits = bitcast[DType.uint32](distances.unsafe_ptr().unsafe_load(cell))
                if arm == 0:
                    expected.append(ix)
                    expected.append(bits)
                elif ix != expected[2 * cell] or bits != expected[2 * cell + 1]:
                    raise Error("query batching changed an output word")
            print("QUERY_BATCH_CASE", n, q, d, k, arm, "used_tile", used)


def main() raises:
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("query batch gate requires IDENTICAL")
    # DEVIATION 2631: the query clamp goes last, so a default tile wider than
    # the request (4,096 against 4,000 queries) comes back as the query count.
    var bench_tile = DEFAULT_QUERY_TILE if DEFAULT_QUERY_TILE < 4000 else 4000
    if plan_query_tile(400000, 4000, DEFAULT_QUERY_TILE) != bench_tile:
        raise Error("default query batch unexpectedly shrank")
    if plan_query_tile(400000, 32, DEFAULT_QUERY_TILE) != 32:
        raise Error("query clamp changed")
    # Host-only policy checks: no large allocation. Above the measured
    # bound, the candidate must retain every old default shrink and floor.
    if plan_query_tile(400001, 4000, DEFAULT_QUERY_TILE) != 256:
        raise Error("unmeasured index escaped the historical batch cap")
    if plan_query_tile(1000000, 4000, DEFAULT_QUERY_TILE) != 128:
        raise Error("million-row index changed historical batch shrink")
    if plan_query_tile(100000000, 4000, DEFAULT_QUERY_TILE) != 32:
        raise Error("large-index historical floor changed")
    if plan_query_tile(100000000, 1, DEFAULT_QUERY_TILE) != 1:
        raise Error("large-index query clamp changed")
    # DEVIATION 2631: a request tile at or under the row's default keeps the
    # bounded-index budget; above a 512 default the historical cap applies.
    var expect_1024 = 1024 if DEFAULT_QUERY_TILE >= 1024 else 256
    if plan_query_tile(400000, 4000, 1024) != expect_1024:
        raise Error("explicit 1024 query tile left its budget rule")
    _case(513, 513, 17, 10)
    _case(65537, 513, 8, 15)
    # Crosses the 512, 2048 and 4096 batch boundaries and three column tiles.
    _case(140000, 4100, 11, 10)
    print("QUERY BATCH PASS", "enabled", QUERY_TILE_512_CANDIDATE, "default_tile", DEFAULT_QUERY_TILE, "cases", 3)
