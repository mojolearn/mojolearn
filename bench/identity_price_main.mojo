"""What `IDENTICAL` costs, in milliseconds, for the three unsupervised paths.

`mojo_only/numerics.mojo` opens with the rule this file exists to satisfy:
**the cost is a measurement, not an argument.** The two modes differ by a
named set of rows, so "what does determinism cost" has an answer in seconds.

    tools/price_unsupervised_identity.sh

runs this main alternately in the two modes, process by process, and reports
per-arm medians. ALTERNATION IS NOT OPTIONAL and it is not tidiness: the M4's
GPU governor drifts up to 1.7x across a twenty-minute window
(`[[mojolearn-box-drifts]]`), and the two modes are two BINARIES, so they
cannot be interleaved inside one process the way `bench/` interleaves two
arms. Alternating whole processes is the closest available thing, and it is
why this main prints one line per arm rather than a summary.

THE ARMS, and why the tiled k-NN one is separated from the default:

    kmeans.fit        the Lloyd loop end to end. Pins here are the norm
                      fold, the contraction and the flushes -- all
                      arithmetic, no extra kernel -- so this arm is the
                      "pins only" price.
    knn.auto          the shipped default. Under IDENTICAL the arm is
                      pinned to cuVS's own dispatch and `grid_x` to 1
                      (DEVIATION 502), so this measures a SCHEDULING
                      change, not a kernel swap.
    knn.tiled         the arm k > 64 must take. Under IDENTICAL its
                      distances come from `pinned_distance_tile_kernel`
                      instead of the vendor matmul (DEVIATION 505), which
                      is the one place in this lane where identity buys a
                      genuinely slower kernel rather than a different
                      rounding. Priced separately BECAUSE it is the
                      expensive one; folding it into an average would hide
                      both numbers.
    dbscan.fit        brute-force eps neighbourhood plus propagation.
"""

from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from cluster.ported.cluster.detail.kmeans import kmeans_fit_main
from cluster.ported.cluster.kmeans_params import INIT_ARRAY, KMeansParams
from dbscan.ported.dbscan.dbscan import dbscan_fit_impl
from dbscan.ported.dbscan.runner import EPS_NN_BRUTE_FORCE
from mojo_only.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from neighbors.ported.neighbors.detail.knn_brute_force import (
    KNN_METHOD_AUTO,
    KNN_METHOD_TILED,
    brute_force_knn_impl,
    compute_norms,
)


comptime REPEATS = 3


def _mode() -> String:
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return String("IDENTICAL")
    return String("FAST")


def _u01(row: Int, k: Int, salt: Int) -> Float32:
    var z = (
        UInt64(row) * 0x9E3779B97F4A7C15
        + UInt64(k + 1) * 0xBF58476D1CE4E5B9
        + UInt64(salt + 1) * 0x94D049BB133111EB
    )
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    z = z ^ (z >> 31)
    return Float32(Int((z >> 40) & UInt64(0xFFFF))) / Float32(65536.0)


def _report(name: String, ms: Float64):
    print("PRICE", _mode(), name, ms)


def _kmeans_arm(ctx: DeviceContext) raises:
    var n = 100000
    var d = 32
    var k = 16

    var hx = ctx.enqueue_create_host_buffer[DType.float32](n * d)
    var hw = ctx.enqueue_create_host_buffer[DType.float32](n)
    var hc = ctx.enqueue_create_host_buffer[DType.float32](k * d)
    ctx.synchronize()
    for i in range(n):
        hw.unsafe_ptr().unsafe_store(i, Float32(1.0))
        for f in range(d):
            hx.unsafe_ptr().unsafe_store(
                i * d + f,
                Float32(i % k) * Float32(3.0) + _u01(i, f, 1),
            )
    for j in range(k):
        for f in range(d):
            hc.unsafe_ptr().unsafe_store(
                j * d + f, Float32(j) * Float32(3.0) + _u01(j, f, 77)
            )

    var x = ctx.enqueue_create_buffer[DType.float32](n * d)
    var w = ctx.enqueue_create_buffer[DType.float32](n)
    var cent = ctx.enqueue_create_buffer[DType.float32](k * d)
    var labels = ctx.enqueue_create_buffer[DType.uint32](n)
    ctx.synchronize()
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=w, src_ptr=hw.unsafe_ptr())
    ctx.synchronize()

    var params = KMeansParams.default()
    params.n_clusters = k
    params.init = INIT_ARRAY
    params.n_init = 1
    params.max_iter = 10
    # The smallest tolerance `validate()` accepts. The arm should do all
    # ten iterations rather than converging early -- a fit that stops on a
    # different iteration in the two modes is not a price, it is a
    # different amount of work, and the whole point is to compare equal
    # work. (`tol = 0` is refused: "invalid parameter (tol<=0)".)
    params.tol = 1.0e-12

    for _r in range(REPEATS):
        ctx.enqueue_copy(dst_buf=cent, src_ptr=hc.unsafe_ptr())
        ctx.synchronize()
        var t0 = perf_counter_ns()
        _ = kmeans_fit_main(
            ctx, x, w, cent, labels, params, n, d,
            Float32(4096.0), Float32(4096.0),
        )
        ctx.synchronize()
        _report(
            String("kmeans.fit"),
            Float64(perf_counter_ns() - t0) / 1.0e6,
        )


def _knn_arm(ctx: DeviceContext, method: Int, name: String) raises:
    var n_index = 20000
    var n_queries = 1000
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
    var hi = ctx.enqueue_create_host_buffer[DType.float32](n_index * d)
    var hq = ctx.enqueue_create_host_buffer[DType.float32](n_queries * d)
    ctx.synchronize()
    for i in range(n_index):
        for f in range(d):
            hi.unsafe_ptr().unsafe_store(i * d + f, _u01(i, f, 3))
    for i in range(n_queries):
        for f in range(d):
            hq.unsafe_ptr().unsafe_store(i * d + f, _u01(i, f, 5))
    ctx.enqueue_copy(dst_buf=index, src_ptr=hi.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=queries, src_ptr=hq.unsafe_ptr())
    ctx.synchronize()
    compute_norms(ctx, index, inorm, n_index, d, False)
    compute_norms(ctx, queries, qnorm, n_queries, d, False)
    ctx.synchronize()

    for _r in range(REPEATS):
        var t0 = perf_counter_ns()
        brute_force_knn_impl(
            ctx, queries, qnorm, index, inorm, dist, bv, bi, od, oi, oi32,
            n_queries, n_index, d, k, tile, buf_len, True, False, True, True,
            method,
        )
        ctx.synchronize()
        _report(name, Float64(perf_counter_ns() - t0) / 1.0e6)


def _dbscan_arm(ctx: DeviceContext) raises:
    var n = 20000
    var d = 8
    var hx = ctx.enqueue_create_host_buffer[DType.float32](n * d)
    ctx.synchronize()
    for i in range(n):
        for f in range(d):
            hx.unsafe_ptr().unsafe_store(
                i * d + f, Float32(i % 6) * Float32(5.0) + _u01(i, f, 9)
            )
    var x = ctx.enqueue_create_buffer[DType.float32](n * d)
    var labels = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.synchronize()
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
    ctx.synchronize()

    for _r in range(REPEATS):
        var t0 = perf_counter_ns()
        _ = dbscan_fit_impl(
            ctx, x, labels, n, d, 1.2, 8, 0, 200, EPS_NN_BRUTE_FORCE, False
        )
        ctx.synchronize()
        _report(String("dbscan.fit"), Float64(perf_counter_ns() - t0) / 1.0e6)


def main() raises:
    var ctx = DeviceContext()
    _kmeans_arm(ctx)
    _knn_arm(ctx, KNN_METHOD_AUTO, String("knn.auto"))
    _knn_arm(ctx, KNN_METHOD_TILED, String("knn.tiled"))
    _dbscan_arm(ctx)
