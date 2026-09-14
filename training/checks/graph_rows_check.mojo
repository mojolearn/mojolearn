# SPDX-License-Identifier: Apache-2.0
"""Cloud-only raw graph distance and KNN selection identity."""
from std.os import getenv
from max.gpu.host import DeviceContext, DeviceBuffer
from metrics.checks.device_io import upload_f32
from neighbors.impl.multi_gpu import parallel_knn_rows
from hierarchy.impl.cluster.detail.multi_gpu import pairwise_rows
from neighbors.checks.pinned_distance_tile import pinned_distance_tile_kernel
from neighbors.impl.detail.knn_brute_force import (
    brute_force_knn_impl, compute_norms_for_metric, tiled_distance_tile_cells,
    tiled_radix_scratch_len, KNN_METHOD_AUTO,
)


def equal[dt: DType](ctx: DeviceContext, mut a: DeviceBuffer[dt],
                     mut b: DeviceBuffer[dt], n: Int) raises:
    var x = ctx.enqueue_create_host_buffer[dt](n)
    var y = ctx.enqueue_create_host_buffer[dt](n)
    ctx.enqueue_copy(dst_ptr=x.unsafe_ptr(), src_buf=a)
    ctx.enqueue_copy(dst_ptr=y.unsafe_ptr(), src_buf=b)
    ctx.synchronize()
    for i in range(n):
        if x.unsafe_ptr()[i].bitcast[DType.uint32]() != y.unsafe_ptr()[i].bitcast[DType.uint32]():
            raise Error("graph raw bits differ at " + String(i))


def check(n: Int, d: Int) raises:
    var ctx = DeviceContext()
    var vals = List[Float32]()
    for i in range(n*d):
        vals.append(Float32((i*17)%113-56)/Float32(64))
    for f in range(d):
        vals[2*d+f] = vals[d+f]
    var x = upload_f32(ctx, vals)
    var q = x.create_sub_buffer[DType.float32](0, n*d)
    var xn = ctx.enqueue_create_buffer[DType.float32](n)
    var qn = ctx.enqueue_create_buffer[DType.float32](n)
    var a = ctx.enqueue_create_buffer[DType.float32](n*n)
    var b = ctx.enqueue_create_buffer[DType.float32](n*n)
    var k = min(n, 7)
    var qt = min(n, 5)
    var bl = tiled_radix_scratch_len(n, k)
    var tile = ctx.enqueue_create_buffer[DType.float32](tiled_distance_tile_cells(qt,n,d,k,0))
    var values = ctx.enqueue_create_buffer[DType.float32](qt*2*bl)
    var indices = ctx.enqueue_create_buffer[DType.uint32](qt*2*bl)
    var ad = ctx.enqueue_create_buffer[DType.float32](n*k)
    var bd = ctx.enqueue_create_buffer[DType.float32](n*k)
    var ai = ctx.enqueue_create_buffer[DType.uint32](n*k)
    var bi = ctx.enqueue_create_buffer[DType.uint32](n*k)
    var ai32 = ctx.enqueue_create_buffer[DType.int32](n*k)
    ctx.synchronize()
    # cuVS DistanceType 0 is L2Expanded; norm bytes are copied unchanged.
    compute_norms_for_metric(ctx, x, xn, n, d, 0)
    compute_norms_for_metric(ctx, q, qn, n, d, 0)
    for square_root in range(2):
        ctx.enqueue_function[pinned_distance_tile_kernel](
            a.unsafe_ptr(), x.unsafe_ptr(), q.unsafe_ptr(), xn.unsafe_ptr(), qn.unsafe_ptr(),
            Int32(n), Int32(n), Int32(d), Int32(square_root),
            grid_dim=((n*n+255)//256,1,1), block_dim=(256,1,1))
        pairwise_rows(ctx, x, xn, b, n, d, Int32(square_root), 256, 2)
        equal(ctx, a, b, n*n)
        brute_force_knn_impl(ctx, q, qn, x, xn, tile, values, indices, ad, ai, ai32,
            n,n,d,k,qt,bl,Bool(square_root),False,True,True,KNN_METHOD_AUTO)
        _ = parallel_knn_rows(ctx,q,qn,x,xn,bd,bi,n,n,d,k,qt,bl,
                             Bool(square_root),KNN_METHOD_AUTO,-1,Float32(2),2)
        equal(ctx, ad, bd, n*k)
        equal(ctx, ai, bi, n*k)
    print("PASS graph raw distance and selection bits", n, d)
    _ = ai32^
    _ = bi^
    _ = ai^
    _ = bd^
    _ = ad^
    _ = indices^
    _ = values^
    _ = tile^
    _ = b^
    _ = a^
    _ = qn^
    _ = xn^
    _ = q^
    _ = x^
    ctx.synchronize()


def main() raises:
    if String(getenv("RUNPOD_POD_ID")) == "":
        raise Error("cloud host required")
    var rows: List[Int] = [3, 17, 65]
    var features: List[Int] = [3, 129]
    for n in rows:
        for d in features:
            check(n,d)
