# SPDX-License-Identifier: Apache-2.0
"""Fused IDENTICAL query tails and ties, with exact integer-distance oracle.

An AMD declaration on Apple exercises logical32 communication but is not a
physical CDNA wave64 run. Pure address/mask checks cover both virtual halves.
"""
from max.gpu.host import DeviceContext
from std.memory import bitcast
from std.sys.compile import is_defined
from checks.kernel_matrix import TARGET_COLUMN, COLUMN_AMD, column_is_simulated, column_name
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from neighbors.impl.detail.fused_l2_knn import fused_l2_knn
from neighbors.impl.topk.logical_warp32 import logical32_source_lane
from ivf.impl.neighbors.ivf_flat.ivf_flat_build import upload_f32


def check_case(ctx: DeviceContext, m: Int, k: Int) raises:
    comptime N = 97
    comptime D = 4
    var x = List[Float32]()
    var q = List[Float32]()
    var xn = List[Float32]()
    var qn = List[Float32]()
    for row in range(N):
        var norm = 0
        for f in range(D):
            var v = (row // 3 + f * 3) % 11
            x.append(Float32(v))
            norm += v * v
        xn.append(Float32(norm))
    for row in range(m):
        var norm = 0
        for f in range(D):
            var v = (row * 7 + f * 5) % 13
            q.append(Float32(v))
            norm += v * v
        qn.append(Float32(norm))
    var dx = upload_f32(ctx, x)
    var dq = upload_f32(ctx, q)
    var dxn = upload_f32(ctx, xn)
    var dqn = upload_f32(ctx, qn)
    var output = ctx.enqueue_create_buffer[DType.float32](m * k)
    var indices = ctx.enqueue_create_buffer[DType.uint32](m * k)
    var ho = ctx.enqueue_create_host_buffer[DType.float32](m * k)
    var hi = ctx.enqueue_create_host_buffer[DType.uint32](m * k)
    fused_l2_knn(ctx, dq, dqn, dx, dxn, output, indices, m, N, D, k, False)
    ctx.enqueue_copy(dst_ptr=ho.unsafe_ptr(), src_buf=output)
    ctx.enqueue_copy(dst_ptr=hi.unsafe_ptr(), src_buf=indices)
    ctx.synchronize()
    for row in range(m):
        var order = List[Int]()
        var distances = List[Int]()
        for col in range(N):
            var distance = 0
            for f in range(D):
                var delta = Int(q[row * D + f]) - Int(x[col * D + f])
                distance += delta * delta
            distances.append(distance)
            order.append(col)
            var pos = col
            while pos > 0 and distances[order[pos - 1]] > distance:
                order[pos] = order[pos - 1]
                pos -= 1
            order[pos] = col
        for rank in range(k):
            var cell = row * k + rank
            var want = order[rank]
            if hi.unsafe_ptr().unsafe_load(cell) != UInt32(want) or bitcast[DType.uint32](ho.unsafe_ptr().unsafe_load(cell)) != bitcast[DType.uint32](Float32(distances[want])):
                print("FUSED_LOGICAL32_FAIL", m, k, row, rank, hi.unsafe_ptr().unsafe_load(cell), want)
                raise Error("fused logical32 distance/tie mismatch")
    print("FUSED_LOGICAL32_CASE_PASS", m, k, "cells", m * k)


def main() raises:
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("fused logical32 gate requires IDENTICAL")
    comptime if is_defined["MOJOLEARN_REQUIRE_CDNA_TARGET"]():
        comptime assert TARGET_COLUMN == COLUMN_AMD and not column_is_simulated(), "requires actual CDNA compilation target"
    for physical in range(64):
        for source in range(32):
            if logical32_source_lane(UInt32(physical), UInt32(source)) != UInt32((physical // 32) * 32 + source):
                raise Error("logical32 broadcast address crosses a half-wave")
    print("FUSED_LOGICAL32_COLUMN", column_name(TARGET_COLUMN), "simulated", column_is_simulated())
    with DeviceContext() as ctx:
        var rows: List[Int] = [1, 9, 17]
        var sizes: List[Int] = [1, 8, 32, 64]
        for m in rows:
            for k in sizes:
                check_case(ctx, m, k)
    print("FUSED LOGICAL32 PASS", "cases", 12, "virtual_addresses", 2048)
