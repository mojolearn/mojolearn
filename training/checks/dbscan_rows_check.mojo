# SPDX-License-Identifier: Apache-2.0
"""Cloud-only complete neighborhood arrays, including CSR column ordering."""
from std.os import getenv, setenv
from max.gpu.host import DeviceContext, DeviceBuffer
from metrics.checks.device_io import upload_f32
from neighbors.impl.ball_cover.ball_cover import rbc_build_index, rbc_n_landmarks
from dbscan.impl.multi_gpu import (
    vertex_deg_dispatch, rbc_eps_nn_query_count,
    rbc_eps_nn_query_fill, rbc_eps_nn_query_max_k,
)


def equal[dt: DType](ctx: DeviceContext, mut a: DeviceBuffer[dt],
                     mut b: DeviceBuffer[dt], n: Int, label: String) raises:
    var x = ctx.enqueue_create_host_buffer[dt](n)
    var y = ctx.enqueue_create_host_buffer[dt](n)
    var av = a.create_sub_buffer[dt](0, n)
    var bv = b.create_sub_buffer[dt](0, n)
    ctx.enqueue_copy(dst_ptr=x.unsafe_ptr(), src_buf=av)
    ctx.enqueue_copy(dst_ptr=y.unsafe_ptr(), src_buf=bv)
    ctx.synchronize()
    for i in range(n):
        if x.unsafe_ptr()[i] != y.unsafe_ptr()[i]:
            raise Error(label + " differs at " + String(i))


def check(rows: Int, d: Int, eps: Float32) raises:
    var ctx = DeviceContext()
    var n = 37
    var landmarks = rbc_n_landmarks(n)
    var values = List[Float32]()
    for i in range(n * d):
        values.append(Float32((i * 17) % 113 - 56) / Float32(64))
    for f in range(d):
        values[3 * d + f] = values[2 * d + f]
    var x = upload_f32(ctx, values)
    var q = x.create_sub_buffer[DType.float32](2 * d, rows * d)
    var r = ctx.enqueue_create_buffer[DType.float32](landmarks*d)
    var xr = ctx.enqueue_create_buffer[DType.float32](n*d)
    var ids = ctx.enqueue_create_buffer[DType.int32](landmarks)
    var slots = ctx.enqueue_create_buffer[DType.int32](n)
    var slot_dist = ctx.enqueue_create_buffer[DType.float32](n)
    var nearest = ctx.enqueue_create_buffer[DType.int32](n)
    var nearest_dist = ctx.enqueue_create_buffer[DType.float32](n)
    var ip = ctx.enqueue_create_buffer[DType.int32](landmarks+1)
    var cols = ctx.enqueue_create_buffer[DType.int32](n)
    var dist = ctx.enqueue_create_buffer[DType.float32](n)
    var radius = ctx.enqueue_create_buffer[DType.float32](landmarks)
    var counts = ctx.enqueue_create_buffer[DType.int32](landmarks)
    var ia1 = ctx.enqueue_create_buffer[DType.int32](rows+1)
    var vd1 = ctx.enqueue_create_buffer[DType.int32](rows+1)
    var ja1 = ctx.enqueue_create_buffer[DType.int32](rows*n)
    var tmp1 = ctx.enqueue_create_buffer[DType.int32](rows*n)
    var scratch1 = ctx.enqueue_create_buffer[DType.int32](1)
    var adj1 = ctx.enqueue_create_buffer[DType.uint8](rows*n)
    var ia2 = ctx.enqueue_create_buffer[DType.int32](rows+1)
    var vd2 = ctx.enqueue_create_buffer[DType.int32](rows+1)
    var ja2 = ctx.enqueue_create_buffer[DType.int32](rows*n)
    var tmp2 = ctx.enqueue_create_buffer[DType.int32](rows*n)
    var scratch2 = ctx.enqueue_create_buffer[DType.int32](1)
    var adj2 = ctx.enqueue_create_buffer[DType.uint8](rows*n)
    ctx.synchronize()
    rbc_build_index(ctx, x, r, xr, ids, slots, slot_dist, nearest, nearest_dist,
                    ip, cols, dist, radius, counts, n, d, landmarks)
    for metric in range(2):
        _ = setenv("MOJOLEARN_DBSCAN_DEVICE_COUNT", "1", True)
        vertex_deg_dispatch(ctx, adj1, vd1, x, 2, rows, n, d, Float64(eps), metric)
        _ = setenv("MOJOLEARN_DBSCAN_DEVICE_COUNT", "2", True)
        vertex_deg_dispatch(ctx, adj2, vd2, x, 2, rows, n, d, Float64(eps), metric)
        equal(ctx, adj1, adj2, rows*n, "dense adjacency")
        equal(ctx, vd1, vd2, rows+1, "dense degrees")
    _ = setenv("MOJOLEARN_DBSCAN_DEVICE_COUNT", "1", True)
    var edges1 = rbc_eps_nn_query_count(ctx, xr, q, r, ip, cols, dist, radius,
                                       ia1, vd1, rows, d, landmarks, eps)
    _ = setenv("MOJOLEARN_DBSCAN_DEVICE_COUNT", "2", True)
    var edges2 = rbc_eps_nn_query_count(ctx, xr, q, r, ip, cols, dist, radius,
                                       ia2, vd2, rows, d, landmarks, eps)
    if edges1 != edges2:
        raise Error("RBC edge counts differ")
    equal(ctx, ia1, ia2, rows+1, "RBC count offsets")
    equal(ctx, vd1, vd2, rows+1, "RBC count degrees")
    _ = setenv("MOJOLEARN_DBSCAN_DEVICE_COUNT", "1", True)
    rbc_eps_nn_query_fill(ctx, xr, q, r, ip, cols, dist, radius,
                         ia1, ja1, rows, d, landmarks, eps)
    _ = setenv("MOJOLEARN_DBSCAN_DEVICE_COUNT", "2", True)
    rbc_eps_nn_query_fill(ctx, xr, q, r, ip, cols, dist, radius,
                         ia2, ja2, rows, d, landmarks, eps)
    equal(ctx, ja1, ja2, edges1, "RBC fill columns")
    _ = setenv("MOJOLEARN_DBSCAN_DEVICE_COUNT", "1", True)
    var longest1 = rbc_eps_nn_query_max_k(ctx, xr, q, r, ip, cols, dist, radius,
        ia1, ja1, vd1, tmp1, scratch1, rows, d, landmarks, eps, n)
    _ = setenv("MOJOLEARN_DBSCAN_DEVICE_COUNT", "2", True)
    var longest2 = rbc_eps_nn_query_max_k(ctx, xr, q, r, ip, cols, dist, radius,
        ia2, ja2, vd2, tmp2, scratch2, rows, d, landmarks, eps, n)
    if longest1 != longest2:
        raise Error("RBC maximum degrees differ")
    equal(ctx, ia1, ia2, rows+1, "RBC bounded offsets")
    equal(ctx, vd1, vd2, rows+1, "RBC bounded degrees")
    equal(ctx, ja1, ja2, edges1, "RBC bounded columns")
    print("PASS DBSCAN neighborhood arrays", rows, d, eps)
    _ = adj2^
    _ = scratch2^
    _ = tmp2^
    _ = ja2^
    _ = vd2^
    _ = ia2^
    _ = adj1^
    _ = scratch1^
    _ = tmp1^
    _ = ja1^
    _ = vd1^
    _ = ia1^
    _ = counts^
    _ = radius^
    _ = dist^
    _ = cols^
    _ = ip^
    _ = nearest_dist^
    _ = nearest^
    _ = slot_dist^
    _ = slots^
    _ = ids^
    _ = xr^
    _ = r^
    _ = q^
    _ = x^
    ctx.synchronize()


def main() raises:
    if String(getenv("RUNPOD_POD_ID")) == "":
        raise Error("cloud host required")
    var rows: List[Int] = [1, 3, 17]
    var features: List[Int] = [3, 129]
    for n in rows:
        for d in features:
            check(n, d, Float32(0.01))
            check(n, d, Float32(100.0))
