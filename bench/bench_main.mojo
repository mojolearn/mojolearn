"""Time our GPU implementations. One line per arm per repeat, parseable.

Prints `ARM <name> <milliseconds>` and nothing else on those lines, so
`bench/run_bench.py` can alternate this process with the scikit-learn one and
compare like for like.

WHAT IS TIMED AND WHAT IS NOT
-----------------------------
Setup, allocation and the host-to-device upload happen ONCE, outside the
loop. What is timed is the fit itself plus the `synchronize` that proves it
finished, because on this stack an un-synchronized timing measures how fast
we can enqueue and nothing else.

The arms interleave inside this process, and `run_bench.py` interleaves this
process with scikit-learn's. That is coarser than `mojo_only/
interleaved.mojo`'s within-process round robin, and it is the best available
across two languages. **This box has been measured drifting two- to
threefold between thermal windows**, so the alternation is not optional.
"""

from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from cluster.ported.cluster.kmeans import fit as kmeans_fit_api
from cluster.ported.cluster.kmeans_params import (
    INIT_ARRAY,
    KMeansParams,
    METRIC_L2_EXPANDED,
)
from dbscan.ported.dbscan.runner import dbscan_fit
from decomposition.ported.linalg.detail.pca import pca_fit
from glm.ported.linalg.detail.lstsq import lstsq_eig
from neighbors.ported.neighbors.detail.knn_brute_force import (
    compute_norms,
    tiled_brute_force_knn,
)
from mojo_only.fixed_point import choose_scale


comptime REPEATS = 5


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


def _emit(name: String, ns: Int) -> None:
    print("ARM " + name + " " + String(Float64(ns) / 1.0e6))


def main() raises:
    var ctx = DeviceContext()

    # ---- shapes, matched exactly in bench_sklearn.py --------------------
    var km_rows = 200000
    var km_cols = 32
    var km_k = 16
    var km_iter = 20

    var knn_index = 20000
    var knn_queries = 2000
    var knn_cols = 32
    var knn_k = 10
    var knn_tile = 256

    var pca_rows = 200000
    var pca_cols = 32
    var pca_comp = 8

    var db_rows = 4000
    var db_cols = 16

    var ols_rows = 500000
    var ols_cols = 32

    # ================= k-means setup ====================================
    var km_x = ctx.enqueue_create_buffer[DType.float32](km_rows * km_cols)
    var km_w = ctx.enqueue_create_buffer[DType.float32](km_rows)
    var km_c = ctx.enqueue_create_buffer[DType.float32](km_k * km_cols)
    var km_l = ctx.enqueue_create_buffer[DType.uint32](km_rows)
    ctx.synchronize()
    var hkx = ctx.enqueue_create_host_buffer[DType.float32](km_rows * km_cols)
    var hkw = ctx.enqueue_create_host_buffer[DType.float32](km_rows)
    var worst = Float64(0.0)
    for i in range(km_rows):
        hkw.unsafe_ptr().unsafe_store(i, Float32(1.0))
        for f in range(km_cols):
            hkx.unsafe_ptr().unsafe_store(
                i * km_cols + f, Float32(_u01(i, f, 0) * 10.0)
            )
    worst = Float64(km_rows) * 10.0
    ctx.enqueue_copy(dst_buf=km_x, src_ptr=hkx.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=km_w, src_ptr=hkw.unsafe_ptr())
    var hkc = ctx.enqueue_create_host_buffer[DType.float32](km_k * km_cols)
    for c in range(km_k):
        for f in range(km_cols):
            hkc.unsafe_ptr().unsafe_store(
                c * km_cols + f, Float32(_u01(c * 7919, f, 5) * 10.0)
            )
    ctx.enqueue_copy(dst_buf=km_c, src_ptr=hkc.unsafe_ptr())
    ctx.synchronize()
    var km_sum_scale = Float32(choose_scale(worst))
    var km_wt_scale = Float32(choose_scale(Float64(km_rows)))
    var km_p = KMeansParams.default()
    km_p.n_clusters = km_k
    km_p.init = INIT_ARRAY
    km_p.metric = METRIC_L2_EXPANDED
    km_p.n_init = 1
    km_p.max_iter = km_iter
    km_p.tol = 0.0000001

    # ================= k-NN setup =======================================
    var kn_idx = ctx.enqueue_create_buffer[DType.float32](
        knn_index * knn_cols
    )
    var kn_q = ctx.enqueue_create_buffer[DType.float32](
        knn_queries * knn_cols
    )
    var kn_in = ctx.enqueue_create_buffer[DType.float32](knn_index)
    var kn_qn = ctx.enqueue_create_buffer[DType.float32](knn_queries)
    var kn_dist = ctx.enqueue_create_buffer[DType.float32](
        knn_tile * knn_index
    )
    var kn_bl = knn_index // 8
    var kn_bv = ctx.enqueue_create_buffer[DType.float32](knn_tile * 2 * kn_bl)
    var kn_bi = ctx.enqueue_create_buffer[DType.uint32](knn_tile * 2 * kn_bl)
    var kn_od = ctx.enqueue_create_buffer[DType.float32](knn_queries * knn_k)
    var kn_oi = ctx.enqueue_create_buffer[DType.uint32](knn_queries * knn_k)
    ctx.synchronize()
    var hki = ctx.enqueue_create_host_buffer[DType.float32](
        knn_index * knn_cols
    )
    for i in range(knn_index):
        for f in range(knn_cols):
            hki.unsafe_ptr().unsafe_store(
                i * knn_cols + f, Float32(_u01(i, f, 1))
            )
    ctx.enqueue_copy(dst_buf=kn_idx, src_ptr=hki.unsafe_ptr())
    var hkq = ctx.enqueue_create_host_buffer[DType.float32](
        knn_queries * knn_cols
    )
    for i in range(knn_queries):
        for f in range(knn_cols):
            hkq.unsafe_ptr().unsafe_store(
                i * knn_cols + f, Float32(_u01(i, f, 2))
            )
    ctx.enqueue_copy(dst_buf=kn_q, src_ptr=hkq.unsafe_ptr())
    ctx.synchronize()

    # ================= PCA setup ========================================
    var pc_x = ctx.enqueue_create_buffer[DType.float32](pca_rows * pca_cols)
    var pc_mu = ctx.enqueue_create_buffer[DType.float32](pca_cols)
    var pc_cov = ctx.enqueue_create_buffer[DType.float32](pca_cols * pca_cols)
    var pc_xa = ctx.enqueue_create_buffer[DType.float32](pca_rows * pca_cols)
    var pc_xa2 = ctx.enqueue_create_buffer[DType.float32](pca_rows * pca_cols)
    ctx.synchronize()
    var hpx = ctx.enqueue_create_host_buffer[DType.float32](
        pca_rows * pca_cols
    )
    for i in range(pca_rows):
        for f in range(pca_cols):
            hpx.unsafe_ptr().unsafe_store(
                i * pca_cols + f, Float32(_u01(i, f, 3) * 4.0)
            )
    ctx.enqueue_copy(dst_buf=pc_x, src_ptr=hpx.unsafe_ptr())
    ctx.synchronize()

    # ================= DBSCAN setup =====================================
    var db_x = ctx.enqueue_create_buffer[DType.float32](db_rows * db_cols)
    var db_xa = ctx.enqueue_create_buffer[DType.float32](db_rows * db_cols)
    var db_xn = ctx.enqueue_create_buffer[DType.float32](db_rows)
    var db_xna = ctx.enqueue_create_buffer[DType.float32](db_rows)
    var db_d = ctx.enqueue_create_buffer[DType.float32](db_rows * db_rows)
    var db_adj = ctx.enqueue_create_buffer[DType.uint8](db_rows * db_rows)
    var db_vd = ctx.enqueue_create_buffer[DType.int32](db_rows)
    var db_core = ctx.enqueue_create_buffer[DType.uint8](db_rows)
    var db_ex = ctx.enqueue_create_buffer[DType.int32](db_rows + 1)
    var db_ci = ctx.enqueue_create_buffer[DType.int32](db_rows * db_rows)
    var db_lab = ctx.enqueue_create_buffer[DType.int32](db_rows)
    ctx.synchronize()
    var hdx = ctx.enqueue_create_host_buffer[DType.float32](db_rows * db_cols)
    for i in range(db_rows):
        for f in range(db_cols):
            hdx.unsafe_ptr().unsafe_store(
                i * db_cols + f, Float32(_u01(i, f, 4) * 2.0)
            )
    ctx.enqueue_copy(dst_buf=db_x, src_ptr=hdx.unsafe_ptr())
    ctx.synchronize()

    # ================= OLS setup ========================================
    var ol_a = ctx.enqueue_create_buffer[DType.float32](ols_rows * ols_cols)
    var ol_b = ctx.enqueue_create_buffer[DType.float32](ols_rows)
    var ol_w = ctx.enqueue_create_buffer[DType.float32](ols_cols)
    var ol_cov = ctx.enqueue_create_buffer[DType.float32](ols_cols * ols_cols)
    var ol_q = ctx.enqueue_create_buffer[DType.float32](ols_cols * ols_cols)
    var ol_qs = ctx.enqueue_create_buffer[DType.float32](ols_cols * ols_cols)
    var ol_s = ctx.enqueue_create_buffer[DType.float32](ols_cols)
    var ol_ab = ctx.enqueue_create_buffer[DType.float32](ols_cols)
    var ol_inv = ctx.enqueue_create_buffer[DType.float32](ols_cols * ols_cols)
    var ol_aa = ctx.enqueue_create_buffer[DType.float32](ols_rows * ols_cols)
    var ol_aa2 = ctx.enqueue_create_buffer[DType.float32](ols_rows * ols_cols)
    ctx.synchronize()
    var hoa = ctx.enqueue_create_host_buffer[DType.float32](
        ols_rows * ols_cols
    )
    var hob = ctx.enqueue_create_host_buffer[DType.float32](ols_rows)
    for i in range(ols_rows):
        var t = 0.0
        for f in range(ols_cols):
            var v = _u01(i, f, 6) - 0.5
            hoa.unsafe_ptr().unsafe_store(i * ols_cols + f, Float32(v))
            t += v * (1.0 + 0.1 * Float64(f))
        hob.unsafe_ptr().unsafe_store(i, Float32(t))
    ctx.enqueue_copy(dst_buf=ol_a, src_ptr=hoa.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=ol_b, src_ptr=hob.unsafe_ptr())
    ctx.synchronize()

    # ================= the interleaved loop =============================
    for _r in range(REPEATS):
        var t0 = perf_counter_ns()
        _ = kmeans_fit_api(
            ctx, km_x, km_w, km_c, km_l, km_p, km_rows, km_cols,
            km_sum_scale, km_wt_scale,
        )
        ctx.synchronize()
        _emit("kmeans", perf_counter_ns() - t0)

        t0 = perf_counter_ns()
        compute_norms(ctx, kn_idx, kn_in, knn_index, knn_cols, False)
        compute_norms(ctx, kn_q, kn_qn, knn_queries, knn_cols, False)
        tiled_brute_force_knn(
            ctx, kn_q, kn_qn, kn_idx, kn_in, kn_dist, kn_bv, kn_bi,
            kn_od, kn_oi, knn_queries, knn_index, knn_cols, knn_k,
            knn_tile, kn_bl, False,
        )
        ctx.synchronize()
        _emit("knn", perf_counter_ns() - t0)

        t0 = perf_counter_ns()
        _ = pca_fit(ctx, pc_x, pc_xa, pc_xa2, pc_mu, pc_cov, pca_rows, pca_cols, pca_comp)
        ctx.synchronize()
        _emit("pca", perf_counter_ns() - t0)

        t0 = perf_counter_ns()
        _ = dbscan_fit(
            ctx, db_x, db_xn, db_d, db_adj, db_vd, db_core, db_ex, db_ci,
            db_lab, db_xa, db_xna, db_rows, db_cols, 0.35, 5,
        )
        ctx.synchronize()
        _emit("dbscan", perf_counter_ns() - t0)

        t0 = perf_counter_ns()
        lstsq_eig(
            ctx, ol_a, ol_b, ol_w, ol_cov, ol_q, ol_qs, ol_s, ol_ab, ol_inv,
            ol_aa, ol_aa2, ols_rows, ols_cols,
        )
        ctx.synchronize()
        _emit("ols", perf_counter_ns() - t0)
