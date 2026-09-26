# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Lane kmeans-linear-speed stage probe (2026-09-17). Not a check and not a
# timing row: it splits ONE LLOYD ITERATION of our IDENTICAL k-means into its
# phases with a synchronize at every boundary, on a prepped benchmark block,
# and prints an FNV-1a digest of every phase's output so two builds of this
# probe (one per define set) can be compared cell for cell.
#
#   pixi run mojo build -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 \
#       cluster/tools/kmeans_linear_stage_probe_main.mojo -o kls_probe
#   ./kls_probe <prefix> <rows> <cols> <k> <iters> <base|blocked|sabotage>
#
# <prefix>_X.bin (float32 rows x cols) and <prefix>_init.bin (k x cols), the
# files bench/results/linear_cluster_istella_2026-09-11/probe_bins.py writes.
from std.memory import unsafe_memcpy
from std.sys import argv
from std.time import perf_counter_ns
from std.collections.string import atol
from max.gpu.host import DeviceBuffer, DeviceContext

from cluster.estimator import plan_sum_scale
from cluster.checks.reduce_by_key import (
    REDUCE_BY_KEY_TPB,
    SUM_MODE_SQDIFF,
    copy_f32_kernel,
    finalize_centroids_kernel,
    finish_sum_kernel,
    blocked_acc_table_cells,
    launch_accumulate_centroid_sums,
    launch_accumulate_centroid_sums_blocked,
    launch_accumulate_weight_per_cluster,
    launch_accumulate_weight_per_cluster_blocked,
    sum_partials_kernel,
    zero_i32_kernel,
)
from cluster.impl.detail.min_cluster_distance_compute import (
    compute_centroid_norms,
    min_cluster_and_distance_compute,
)
from cluster.impl.kmeans_params import METRIC_L2_EXPANDED
from checks.fixed_point import choose_scale
from core.device_zero import enqueue_fill
from core.row_norms import NORM_TPB, row_norm_kernel


def _load(path: String, n: Int) raises -> List[UInt8]:
    var f = open(path, "r")
    var b = f.read_bytes()
    f.close()
    if len(b) != n * 4:
        raise Error(path + " holds " + String(len(b)) + " bytes, expected " + String(n * 4))
    return b^


def _ms(t0: Int) -> Float64:
    return Float64(Int(perf_counter_ns()) - t0) / 1.0e6


def _now() -> Int:
    return Int(perf_counter_ns())


def _fnv(p: MutPointer[UInt8, MutUntrackedOrigin], n: Int) -> UInt64:
    var h = UInt64(0xCBF29CE484222325)
    for i in range(n):
        h = (h ^ UInt64(p.unsafe_load(i))) * UInt64(0x100000001B3)
    return h


def main() raises:
    var args = argv()
    if len(args) != 7:
        raise Error("usage: kls_probe <prefix> <rows> <cols> <k> <iters> <base|blocked|sabotage>")
    var arm = String(args[6])
    var prefix = String(args[1])
    var rows = Int(atol(args[2]))
    var cols = Int(atol(args[3]))
    var k = Int(atol(args[4]))
    var iters = Int(atol(args[5]))
    var cells = rows * cols
    var cd = k * cols
    var ctx = DeviceContext()

    var bx = _load(prefix + "_X.bin", cells)
    var bi = _load(prefix + "_init.bin", cd)
    var hx = ctx.enqueue_create_host_buffer[DType.float32](cells)
    var hi = ctx.enqueue_create_host_buffer[DType.float32](cd)
    var h_lab = ctx.enqueue_create_host_buffer[DType.uint32](rows)
    var h_md = ctx.enqueue_create_host_buffer[DType.float32](rows)
    var h_sums = ctx.enqueue_create_host_buffer[DType.int32](cd)
    var h_w = ctx.enqueue_create_host_buffer[DType.int32](k)
    var h_cen = ctx.enqueue_create_host_buffer[DType.float32](cd)
    var h_shift = ctx.enqueue_create_host_buffer[DType.float32](1)
    ctx.synchronize()
    unsafe_memcpy(dest=hx.unsafe_ptr().bitcast[UInt8](), src=bx.unsafe_ptr(), count=cells * 4)
    unsafe_memcpy(dest=hi.unsafe_ptr().bitcast[UInt8](), src=bi.unsafe_ptr(), count=cd * 4)
    _ = bx^
    _ = bi^

    for rep in range(1):
        var t0 = _now()
        var s = plan_sum_scale(hx.unsafe_ptr(), rows, cols)
        print("plan_sum_scale rep=" + String(rep) + " ms=" + String(_ms(t0)) + " scale=" + String(s))
    var sum_scale = Float32(plan_sum_scale(hx.unsafe_ptr(), rows, cols))
    var weight_scale = Float32(choose_scale(Float64(rows), rows))

    var x = ctx.enqueue_create_buffer[DType.float32](cells)
    var w = ctx.enqueue_create_buffer[DType.float32](rows)
    var cur = ctx.enqueue_create_buffer[DType.float32](cd)
    var nxt = ctx.enqueue_create_buffer[DType.float32](cd)
    var cnorm = ctx.enqueue_create_buffer[DType.float32](k)
    var lab = ctx.enqueue_create_buffer[DType.uint32](rows)
    var xn = ctx.enqueue_create_buffer[DType.float32](rows)
    var md = ctx.enqueue_create_buffer[DType.float32](rows)
    var dist = ctx.enqueue_create_buffer[DType.float32](1)
    var sums = ctx.enqueue_create_buffer[DType.int32](cd)
    var wsum = ctx.enqueue_create_buffer[DType.int32](k)
    var table = ctx.enqueue_create_buffer[DType.int32](blocked_acc_table_cells(rows, cols, k))
    var table_w = ctx.enqueue_create_buffer[DType.int32](blocked_acc_table_cells(rows, 1, k))
    var partials = ctx.enqueue_create_buffer[DType.float32](256)
    var d_shift = ctx.enqueue_create_buffer[DType.float32](1)
    ctx.synchronize()
    var t0 = _now()
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
    ctx.synchronize()
    print("upload_x_pinned ms=" + String(_ms(t0)))
    ctx.enqueue_copy(dst_buf=cur, src_ptr=hi.unsafe_ptr())
    enqueue_fill[DType.float32](ctx, w, Float32(1.0))
    t0 = _now()
    ctx.enqueue_function[row_norm_kernel](
        xn.unsafe_ptr(), x.unsafe_ptr(), Int32(cols), Int32(0),
        grid_dim=(rows, 1, 1), block_dim=(NORM_TPB, 1, 1),
    )
    ctx.synchronize()
    print("row_norm ms=" + String(_ms(t0)))

    for it in range(iters):
        t0 = _now()
        ctx.enqueue_function[zero_i32_kernel](
            sums.unsafe_ptr(), Int32(cd),
            grid_dim=((cd + 255) // 256, 1, 1), block_dim=(256, 1, 1),
        )
        ctx.enqueue_function[zero_i32_kernel](
            wsum.unsafe_ptr(), Int32(k),
            grid_dim=((k + 255) // 256, 1, 1), block_dim=(256, 1, 1),
        )
        compute_centroid_norms(ctx, cur, cnorm, k, cols, METRIC_L2_EXPANDED)
        ctx.synchronize()
        var t_zero = _ms(t0)
        t0 = _now()
        min_cluster_and_distance_compute(
            ctx, x, xn, cur, cnorm, dist, lab, md, rows, cols, k,
            METRIC_L2_EXPANDED, 0, 0,
        )
        ctx.synchronize()
        var t_assign = _ms(t0)
        t0 = _now()
        if arm == "blocked":
            launch_accumulate_centroid_sums_blocked(ctx, sums, table, x, lab, w, rows, cols, k, sum_scale)
        elif arm == "sabotage":
            launch_accumulate_centroid_sums_blocked[True](ctx, sums, table, x, lab, w, rows, cols, k, sum_scale)
        else:
            launch_accumulate_centroid_sums(ctx, sums, x, lab, w, rows, cols, k, sum_scale)
        ctx.synchronize()
        var t_sums = _ms(t0)
        t0 = _now()
        if arm == "base":
            launch_accumulate_weight_per_cluster(ctx, wsum, lab, w, rows, k, weight_scale)
        else:
            launch_accumulate_weight_per_cluster_blocked(ctx, wsum, table_w, lab, w, rows, k, weight_scale)
        ctx.synchronize()
        var t_w = _ms(t0)
        t0 = _now()
        ctx.enqueue_function[finalize_centroids_kernel](
            nxt.unsafe_ptr(), cur.unsafe_ptr(), sums.unsafe_ptr(), wsum.unsafe_ptr(),
            Int32(k), Int32(cols), sum_scale, weight_scale,
            grid_dim=((cd + 255) // 256, 1, 1), block_dim=(256, 1, 1),
        )
        var blocks = (cd + REDUCE_BY_KEY_TPB - 1) // REDUCE_BY_KEY_TPB
        if blocks > 256:
            blocks = 256
        ctx.enqueue_function[sum_partials_kernel](
            partials.unsafe_ptr(), cur.unsafe_ptr(), nxt.unsafe_ptr(),
            Int32(cd), Int32(SUM_MODE_SQDIFF),
            grid_dim=(blocks, 1, 1), block_dim=(REDUCE_BY_KEY_TPB, 1, 1),
        )
        ctx.enqueue_function[finish_sum_kernel](
            d_shift.unsafe_ptr(), partials.unsafe_ptr(), Int32(blocks),
            grid_dim=(1, 1, 1), block_dim=(REDUCE_BY_KEY_TPB, 1, 1),
        )
        ctx.enqueue_copy(dst_ptr=h_shift.unsafe_ptr(), src_buf=d_shift)
        ctx.enqueue_function[copy_f32_kernel](
            cur.unsafe_ptr(), nxt.unsafe_ptr(), Int32(cd),
            grid_dim=((cd + 255) // 256, 1, 1), block_dim=(256, 1, 1),
        )
        ctx.synchronize()
        var t_fin = _ms(t0)
        # Digests, outside every clock.
        ctx.enqueue_copy(dst_ptr=h_lab.unsafe_ptr(), src_buf=lab)
        ctx.enqueue_copy(dst_ptr=h_md.unsafe_ptr(), src_buf=md)
        ctx.enqueue_copy(dst_ptr=h_sums.unsafe_ptr(), src_buf=sums)
        ctx.enqueue_copy(dst_ptr=h_w.unsafe_ptr(), src_buf=wsum)
        ctx.enqueue_copy(dst_ptr=h_cen.unsafe_ptr(), src_buf=cur)
        ctx.synchronize()
        print(
            "iter=" + String(it)
            + " zero_norm_ms=" + String(t_zero)
            + " assign_ms=" + String(t_assign)
            + " sums_ms=" + String(t_sums)
            + " weights_ms=" + String(t_w)
            + " finalize_shift_ms=" + String(t_fin)
            + " | labels=" + String(_fnv(h_lab.unsafe_ptr().bitcast[UInt8](), rows * 4))
            + " min_dist=" + String(_fnv(h_md.unsafe_ptr().bitcast[UInt8](), rows * 4))
            + " sums_i32=" + String(_fnv(h_sums.unsafe_ptr().bitcast[UInt8](), cd * 4))
            + " weight_i32=" + String(_fnv(h_w.unsafe_ptr().bitcast[UInt8](), k * 4))
            + " centroids=" + String(_fnv(h_cen.unsafe_ptr().bitcast[UInt8](), cd * 4))
            + " shift=" + String(h_shift.unsafe_ptr().unsafe_load(0))
        )
    _ = hx^
    _ = hi^
    _ = x^
    _ = w^
    _ = xn^
    _ = md^
    _ = lab^
    _ = dist^
    _ = table^
    _ = table_w^
