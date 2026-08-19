"""Scaling curves. The measurement the fixed-size benchmark could not give.

Every number in `bench/results/` so far is at ONE size, and they are small:
DBSCAN at 4,000 points, k-NN at 20k index against 2k queries. Those are the
sizes where a GPU cannot win, because 16 million pairs is nothing and what
gets measured is launch overhead against Accelerate hitting AMX. mojotrees
already found this crossover once and put it near 150k rows.

So the question "what if people have a lot of data" is not a matter of
opinion, it is an untested regime, and it is the one this architecture exists
for. DBSCAN at 200k points is 40 BILLION pairs. That is where scikit-learn
stops being usable and where quadratic-on-a-GPU stops being a rounding error.

Prints `ARM <name>@<size> <ms>` so `run_bench.py` can alternate this process
with the scikit-learn one and compare per size.
"""

from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from dbscan.ported.dbscan.runner import dbscan_fit
from neighbors.ported.neighbors.detail.knn_brute_force import (
    compute_norms,
    tiled_brute_force_knn,
)


comptime REPEATS = 3


def _u01(row: Int, k: Int, salt: Int) -> Float64:
    var z = (
        UInt64(row) * 0x9E3779B97F4A7C15
        + UInt64(k + 1) * 0xBF58476D1CE4E5B9
        + UInt64(salt + 1) * 0x94D049BB133111EB
    )
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    z = z ^ (z >> 31)
    return Float64(z >> 11) * (1.0 / 9007199254740992.0)


def _knn_at(ctx: DeviceContext, n_index: Int, n_queries: Int) raises -> Int:
    var d = 32
    var k = 10
    var tile = 256
    var buf_len = max(n_index // 8, k)

    var index = ctx.enqueue_create_buffer[DType.float32](n_index * d)
    var queries = ctx.enqueue_create_buffer[DType.float32](n_queries * d)
    var inorm = ctx.enqueue_create_buffer[DType.float32](n_index)
    var qnorm = ctx.enqueue_create_buffer[DType.float32](n_queries)
    var dist = ctx.enqueue_create_buffer[DType.float32](tile * n_index)
    var bv = ctx.enqueue_create_buffer[DType.float32](tile * 2 * buf_len)
    var bi = ctx.enqueue_create_buffer[DType.uint32](tile * 2 * buf_len)
    var od = ctx.enqueue_create_buffer[DType.float32](n_queries * k)
    var oi = ctx.enqueue_create_buffer[DType.uint32](n_queries * k)
    var oi32 = ctx.enqueue_create_buffer[DType.int32](n_queries * k)
    ctx.synchronize()

    var hi = ctx.enqueue_create_host_buffer[DType.float32](n_index * d)
    for i in range(n_index):
        for f in range(d):
            hi.unsafe_ptr().unsafe_store(i * d + f, Float32(_u01(i, f, 1)))
    ctx.enqueue_copy(dst_buf=index, src_ptr=hi.unsafe_ptr())
    var hq = ctx.enqueue_create_host_buffer[DType.float32](n_queries * d)
    for i in range(n_queries):
        for f in range(d):
            hq.unsafe_ptr().unsafe_store(i * d + f, Float32(_u01(i, f, 2)))
    ctx.enqueue_copy(dst_buf=queries, src_ptr=hq.unsafe_ptr())
    ctx.synchronize()

    var best = 0
    for r in range(REPEATS):
        var t0 = perf_counter_ns()
        compute_norms(ctx, index, inorm, n_index, d, False)
        compute_norms(ctx, queries, qnorm, n_queries, d, False)
        tiled_brute_force_knn(
            ctx, queries, qnorm, index, inorm, dist, bv, bi, od, oi, oi32,
            n_queries, n_index, d, k, tile, buf_len, False,
        )
        ctx.synchronize()
        var dt = Int(perf_counter_ns() - t0)
        if r == 0 or dt < best:
            best = dt
        print("ARM knn@" + String(n_index) + " " + String(Float64(dt) / 1.0e6))
    return best


def _dbscan_at(ctx: DeviceContext, n: Int) raises:
    var d = 8
    var batch = 2048
    var eps = 0.30

    var x = ctx.enqueue_create_buffer[DType.float32](n * d)
    var xa = ctx.enqueue_create_buffer[DType.float32](n * d)
    var xn = ctx.enqueue_create_buffer[DType.float32](n)
    var xna = ctx.enqueue_create_buffer[DType.float32](n)
    var dist = ctx.enqueue_create_buffer[DType.float32](batch * n)
    var adj = ctx.enqueue_create_buffer[DType.uint8](batch * n)
    var vd = ctx.enqueue_create_buffer[DType.int32](n)
    var core = ctx.enqueue_create_buffer[DType.uint8](n)
    var ex = ctx.enqueue_create_buffer[DType.int32](n + 1)
    var ci = ctx.enqueue_create_buffer[DType.int32](n * 48)
    var lab = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.synchronize()

    var h = ctx.enqueue_create_host_buffer[DType.float32](n * d)
    for i in range(n):
        for f in range(d):
            h.unsafe_ptr().unsafe_store(i * d + f, Float32(_u01(i, f, 4) * 4.0))
    ctx.enqueue_copy(dst_buf=x, src_ptr=h.unsafe_ptr())
    ctx.synchronize()

    for _r in range(REPEATS):
        var t0 = perf_counter_ns()
        _ = dbscan_fit(
            ctx, x, xn, dist, adj, vd, core, ex, ci, lab, xa, xna,
            n, d, eps, 5, batch,
        )
        ctx.synchronize()
        print(
            "ARM dbscan@" + String(n) + " "
            + String(Float64(perf_counter_ns() - t0) / 1.0e6)
        )


def main() raises:
    var ctx = DeviceContext()
    # Unrolled: this Mojo has no variadic `List[Int](...)` constructor.
    _ = _knn_at(ctx, 20000, 2000)
    _ = _knn_at(ctx, 50000, 2000)
    _ = _knn_at(ctx, 100000, 2000)
    _ = _knn_at(ctx, 200000, 2000)
    _ = _knn_at(ctx, 400000, 2000)
    _dbscan_at(ctx, 4000)
    _dbscan_at(ctx, 16000)
    _dbscan_at(ctx, 50000)
    _dbscan_at(ctx, 100000)
    _dbscan_at(ctx, 200000)
