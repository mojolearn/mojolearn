from max.gpu.host import DeviceContext
from neighbors.mojo_only.knn_check import check_knn
from neighbors.ported.neighbors.detail.knn_brute_force import (
    compute_norms, tiled_brute_force_knn,
)

def main() raises:
    var ctx = DeviceContext()
    var N = 4096
    var Q = 4
    var F = 4
    var K = 8
    var buf_len = N // 8
    var index = ctx.enqueue_create_buffer[DType.float32](N * F)
    var queries = ctx.enqueue_create_buffer[DType.float32](Q * F)
    var index_norm = ctx.enqueue_create_buffer[DType.float32](N)
    var query_norm = ctx.enqueue_create_buffer[DType.float32](Q)
    var dist_tile = ctx.enqueue_create_buffer[DType.float32](Q * N)
    var bv = ctx.enqueue_create_buffer[DType.float32](Q * 2 * buf_len)
    var bi = ctx.enqueue_create_buffer[DType.uint32](Q * 2 * buf_len)
    var od = ctx.enqueue_create_buffer[DType.float32](Q * K)
    var oi = ctx.enqueue_create_buffer[DType.uint32](Q * K)
    ctx.synchronize()

    var hi = ctx.enqueue_create_host_buffer[DType.float32](N * F)
    for j in range(N):
        for f in range(F):
            var v = Float32(0.0)
            if f == 0:
                v = Float32(100 * j)
            hi.unsafe_ptr().unsafe_store(j * F + f, v)
    ctx.enqueue_copy(dst_buf=index, src_ptr=hi.unsafe_ptr())
    var hq = ctx.enqueue_create_host_buffer[DType.float32](Q * F)
    for i in range(Q):
        for f in range(F):
            var v = Float32(0.0)
            if f == 0:
                v = Float32(100.0) * (Float32(1000 + i) + Float32(0.3))
            hq.unsafe_ptr().unsafe_store(i * F + f, v)
    ctx.enqueue_copy(dst_buf=queries, src_ptr=hq.unsafe_ptr())
    ctx.synchronize()

    compute_norms(ctx, index, index_norm, N, F, False)
    compute_norms(ctx, queries, query_norm, Q, F, False)
    ctx.synchronize()
    tiled_brute_force_knn(ctx, queries, query_norm, index, index_norm,
        dist_tile, bv, bi, od, oi, Q, N, F, K, Q, buf_len, False)

    var ho = ctx.enqueue_create_host_buffer[DType.uint32](Q * K)
    var hd = ctx.enqueue_create_host_buffer[DType.float32](Q * K)
    ctx.enqueue_copy(dst_ptr=ho.unsafe_ptr(), src_buf=oi)
    ctx.enqueue_copy(dst_ptr=hd.unsafe_ptr(), src_buf=od)
    ctx.synchronize()
    for i in range(Q):
        print("query center", 1000 + i)
        for s in range(K):
            print("   slot", s, "idx", ho.unsafe_ptr().unsafe_load(i*K+s),
                  "dist", hd.unsafe_ptr().unsafe_load(i*K+s))
