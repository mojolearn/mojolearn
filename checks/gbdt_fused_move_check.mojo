# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Compare the fused Newton move/evaluation against the separate kernels.

Every output is compared bitwise, including cursor, derivative planes,
function-value partials and magnitude partials. Ragged rows, scattered bins,
nonuniform weights and repeated moves exercise indexing and accumulation.
Build separately with and without MOJOLEARN_NUMERIC_IDENTICAL. Timings are
kernel-pair microbenchmarks, not end-to-end fit speedups.
"""
from max.gpu.host import DeviceContext, DeviceBuffer
from std.memory import bitcast
from std.time import perf_counter_ns
from checks.numerics import numeric_mode_name
from gbdt.methods.kernel_add_model_value import (
    ABMV_BLOCK, ABMV_ELEMENTS, add_bin_model_value_kernel,
)
from gbdt.targets.kernel.pointwise_targets import (
    MSE_BLOCK_SIZE, OBJECTIVE_LOGLOSS, OBJECTIVE_CROSSENTROPY,
    OBJECTIVE_RMSE, OBJECTIVE_POISSON, OBJECTIVE_TWEEDIE,
    OBJECTIVE_LQ, OBJECTIVE_EXPECTILE, OBJECTIVE_HUBER,
    OBJECTIVE_QUANTILE, OBJECTIVE_MAE, OBJECTIVE_MAPE,
    OBJECTIVE_LOGLINQUANTILE, launch_approximate,
    launch_approximate_move_eval,
)


def same(ctx: DeviceContext, a: DeviceBuffer[DType.float32],
         b: DeviceBuffer[DType.float32], n: Int, label: String) raises:
    var ha = ctx.enqueue_create_host_buffer[DType.float32](n)
    var hb = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.enqueue_copy(dst_ptr=ha.unsafe_ptr(), src_buf=a)
    ctx.enqueue_copy(dst_ptr=hb.unsafe_ptr(), src_buf=b)
    ctx.synchronize()
    for i in range(n):
        if bitcast[DType.uint32](ha.unsafe_ptr().unsafe_load(i)) != bitcast[DType.uint32](hb.unsafe_ptr().unsafe_load(i)):
            raise Error(label + " differs at " + String(i))


def check[estimation: Bool](objective: Int, weighted: Bool,
                           n: Int = 4133, repeats: Int = 3) raises:
    var ctx = DeviceContext()
    var blocks = (n + MSE_BLOCK_SIZE - 1) // MSE_BLOCK_SIZE
    var alpha = Float32(1.5)
    if (objective == OBJECTIVE_QUANTILE or objective == OBJECTIVE_EXPECTILE
        or objective == OBJECTIVE_LOGLINQUANTILE):
        alpha = Float32(0.3)
    var target = ctx.enqueue_create_buffer[DType.float32](n)
    var weights = ctx.enqueue_create_buffer[DType.float32](n)
    var bins = ctx.enqueue_create_buffer[DType.uint32](n)
    var shift = ctx.enqueue_create_buffer[DType.float32](64)
    var cursor_a = ctx.enqueue_create_buffer[DType.float32](n)
    var cursor_b = ctx.enqueue_create_buffer[DType.float32](n)
    var stats_a = ctx.enqueue_create_buffer[DType.float32](2 * n)
    var stats_b = ctx.enqueue_create_buffer[DType.float32](2 * n)
    var fv_a = ctx.enqueue_create_buffer[DType.float32](blocks)
    var fv_b = ctx.enqueue_create_buffer[DType.float32](blocks)
    var mag_a = ctx.enqueue_create_buffer[DType.float32](2 * blocks)
    var mag_b = ctx.enqueue_create_buffer[DType.float32](2 * blocks)
    var ht = ctx.enqueue_create_host_buffer[DType.float32](n)
    var hw = ctx.enqueue_create_host_buffer[DType.float32](n)
    var hc = ctx.enqueue_create_host_buffer[DType.float32](n)
    var hb = ctx.enqueue_create_host_buffer[DType.uint32](n)
    var hs = ctx.enqueue_create_host_buffer[DType.float32](64)
    for i in range(n):
        ht.unsafe_ptr().unsafe_store(i, Float32((i * 17 + 3) % 113 + 1) / 128)
        hw.unsafe_ptr().unsafe_store(i, Float32((i * 23) % 19 + 1) / 16)
        hc.unsafe_ptr().unsafe_store(i, Float32((i * 31) % 101 - 50) / 64)
        hb.unsafe_ptr().unsafe_store(i, UInt32((i * 37 + i // 11) % 64))
    for i in range(64):
        hs.unsafe_ptr().unsafe_store(i, Float32((i * 13) % 31 - 15) / 1024)
    ctx.enqueue_copy(dst_buf=target, src_ptr=ht.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=weights, src_ptr=hw.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=bins, src_ptr=hb.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=shift, src_ptr=hs.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=cursor_a, src_ptr=hc.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=cursor_b, src_ptr=hc.unsafe_ptr())
    ctx.synchronize()
    var timings = List[Float64]()
    # Warm each kernel before timing; both arms see exactly repeats+1 moves.
    for fused in range(2):
        var started = perf_counter_ns()
        for rep in range(repeats + 1):
            if rep == 1:
                ctx.synchronize()
                started = perf_counter_ns()
            if fused == 0:
                ctx.enqueue_function[add_bin_model_value_kernel](
                    shift.unsafe_ptr(), bins.unsafe_ptr(), Int32(n),
                    Int32(1), Int32(n), cursor_a.unsafe_ptr(),
                    grid_dim=((n + ABMV_BLOCK * ABMV_ELEMENTS - 1) // (ABMV_BLOCK * ABMV_ELEMENTS), 1, 1),
                    block_dim=(ABMV_BLOCK, 1, 1),
                )
                launch_approximate[estimation](
                    ctx, objective, target, weights, Int32(n), cursor_a,
                    Int32(weighted), alpha, Float32(0.5), stats_a,
                    fv_a, Int32(1), mag_a, Int32(1), blocks,
                )
            else:
                launch_approximate_move_eval[estimation](
                    ctx, objective, shift, bins, target, weights, Int32(n),
                    cursor_b, Int32(weighted), alpha, Float32(0.5),
                    stats_b, fv_b, Int32(1), mag_b, Int32(1), blocks,
                )
        ctx.synchronize()
        timings.append(Float64(perf_counter_ns() - started) / 1.0e6 / Float64(repeats))
    same(ctx, cursor_a, cursor_b, n, "cursor")
    same(ctx, stats_a, stats_b, 2 * n, "derivatives")
    same(ctx, fv_a, fv_b, blocks, "function value")
    same(ctx, mag_a, mag_b, 2 * blocks, "magnitudes")
    print("equal", objective, "estimation", estimation, "weighted", weighted,
          "rows", n, "split_ms", timings[0], "fused_ms", timings[1])


def main() raises:
    print("numeric_mode", numeric_mode_name())
    var objectives: List[Int] = [OBJECTIVE_LOGLOSS, OBJECTIVE_CROSSENTROPY,
        OBJECTIVE_RMSE, OBJECTIVE_POISSON, OBJECTIVE_TWEEDIE, OBJECTIVE_LQ,
        OBJECTIVE_EXPECTILE, OBJECTIVE_HUBER, OBJECTIVE_QUANTILE,
        OBJECTIVE_MAE, OBJECTIVE_MAPE, OBJECTIVE_LOGLINQUANTILE]
    for objective in objectives:
        for weighted in range(2):
            check[True](objective, Bool(weighted))
            check[False](objective, Bool(weighted))
    for _ in range(5):
        check[True](OBJECTIVE_LOGLOSS, True, n=1000003, repeats=30)
    print("ALL FUSED MOVE ARMS GREEN")
