# SPDX-License-Identifier: Apache-2.0
"""lane/cnn-apple (2026-09-28): every EXECUTION plan of the pinned GEMM at the
x_cnn shapes (CNNClassifier (32,64) batch 256 on 3x32x32, the Conv2d N256
bench), on this device. Per (shape, plan): median ms per call (R calls queued
back to back, one synchronize, so the time is the GPU's, not the sync), and
the count of output words that differ from the SHIPPED dispatch
(`identical_gemm_into[False]`, what x_cnn/device.mojo runs) and from the x_cnn
pick (`device_gemm`'s forced split plans). Every plan is the same leaves and
the same fold (contract 6), so every mismatch count must read 0; a nonzero
one is a defect, not a trade. Timing only: synthetic operands.
    CNN_PLANS_ONLY=<name,...>   run only those rows"""
from std.memory import bitcast
from std.time import perf_counter_ns
from std.gpu import block_dim, block_idx, thread_idx
from std.os import getenv
from std.atomic import Atomic
from max.gpu.host import DeviceBuffer, DeviceContext
from gemm.checks.gemm_identical import (
    choose_gemm_plan, gemm_plan_name, identical_gemm_into, identical_gemm_with_plan,
    identical_gemm_workspace_floats, identical_gemm_workspace_max_floats, apple_mma_applies,
    PLAN_APPLE_MMA, PLAN_APPLE_MMA_SPLIT, PLAN_APPLE_MMA_SPLIT_BIG, PLAN_SPLIT_32_2X2, PLAN_SPLIT_16_1X1, PLAN_TUNED_32_2X2, PLAN_SPLIT_64_4X4, GEMM_PLAN_COUNT,
)
from gemm.checks.gemm_oracle import OP_NN, OP_NT, OP_TN, op_name


def fill_kernel(p: MutPointer[Float32, MutAnyOrigin], n: Int32, salt: UInt32):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n):
        var x = UInt32(i) * UInt32(1664525) + salt
        x = x ^ (x >> 13)
        x = x * UInt32(2246822519)
        p.unsafe_store(i, Float32(Int(x & UInt32(65535)) - 32767) / Float32(8192))


def mismatch_kernel(a: MutPointer[Float32, MutAnyOrigin], b: MutPointer[Float32, MutAnyOrigin], n: Int32,
                    count: MutPointer[Int32, MutAnyOrigin]):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n) and bitcast[DType.uint32](a.unsafe_load(i)) != bitcast[DType.uint32](b.unsafe_load(i)):
        _ = Atomic.fetch_add(count.unsafe_offset(0), Int32(1))


def cnn_pick(m: Int, n: Int, k: Int, op: Int) -> Int:
    """x_cnn/device.mojo `device_gemm`'s DEFAULT (the split plan tuned on the
    RTX 4090; -1 = the shipped dispatcher). On Apple IDENTICAL device_gemm
    measures its candidates per shape (`_apple_tuned_plan`)."""
    if op == OP_TN and m * n <= 65536:
        return PLAN_SPLIT_64_4X4 if (m >= 64 and n >= 64) else PLAN_SPLIT_32_2X2
    return -1


def run_plan(ctx: DeviceContext, mut c: DeviceBuffer[DType.float32], mut a: DeviceBuffer[DType.float32],
             mut b: DeviceBuffer[DType.float32], mut ws: DeviceBuffer[DType.float32], m: Int, n: Int, k: Int,
             op: Int, plan: Int) raises:
    if plan < 0:
        identical_gemm_into[False](ctx, c, a, b, ws, m, n, k, op)
    else:
        identical_gemm_with_plan(ctx, c, a, b, ws, m, n, k, op, plan)


def count_mismatch(ctx: DeviceContext, mut x: DeviceBuffer[DType.float32], mut y: DeviceBuffer[DType.float32],
                   total: Int) raises -> Int:
    var dm = ctx.enqueue_create_buffer[DType.int32](1)
    dm.enqueue_fill(Int32(0))
    ctx.enqueue_function[mismatch_kernel](x.unsafe_ptr(), y.unsafe_ptr(), Int32(total), dm.unsafe_ptr(),
        grid_dim=((total + 255) // 256, 1, 1), block_dim=(256, 1, 1))
    var hm = ctx.enqueue_create_host_buffer[DType.int32](1)
    ctx.enqueue_copy(dst_ptr=hm.unsafe_ptr(), src_buf=dm)
    ctx.synchronize()
    var r = Int(hm.unsafe_ptr().unsafe_load(0))
    _ = dm^
    _ = hm^
    return r


def time_plan(ctx: DeviceContext, mut c: DeviceBuffer[DType.float32], mut a: DeviceBuffer[DType.float32],
              mut b: DeviceBuffer[DType.float32], mut ws: DeviceBuffer[DType.float32], m: Int, n: Int, k: Int,
              op: Int, plan: Int) raises -> Float64:
    """Median ms per call over 5 samples of R back-to-back calls and one wait."""
    var t0 = perf_counter_ns()
    run_plan(ctx, c, a, b, ws, m, n, k, op, plan)
    ctx.synchronize()
    var one = Float64(perf_counter_ns() - t0) / 1e6
    var reps = 1 if one > 200.0 else (3 if one > 20.0 else 10)
    var xs = List[Float64]()
    for _ in range(5 if one < 500.0 else 1):
        var s = perf_counter_ns()
        for _ in range(reps):
            run_plan(ctx, c, a, b, ws, m, n, k, op, plan)
        ctx.synchronize()
        xs.append(Float64(perf_counter_ns() - s) / 1e6 / Float64(reps))
    # insertion sort, median
    for i in range(1, len(xs)):
        var j = i
        while j > 0 and xs[j - 1] > xs[j]:
            var t = xs[j]; xs[j] = xs[j - 1]; xs[j - 1] = t
            j -= 1
    return xs[len(xs) // 2]


def sweep(ctx: DeviceContext, name: String, m: Int, n: Int, k: Int, op: Int) raises:
    var only = String(getenv("CNN_PLANS_ONLY"))
    if only.byte_length() > 0 and (String(",") + only + ",").find(String(",") + name + ",") < 0:
        return
    var a_n = m * k
    var b_n = n * k
    var a = ctx.enqueue_create_buffer[DType.float32](a_n)
    var b = ctx.enqueue_create_buffer[DType.float32](b_n)
    var c = ctx.enqueue_create_buffer[DType.float32](m * n)
    var cref = ctx.enqueue_create_buffer[DType.float32](m * n)
    var wsn = identical_gemm_workspace_max_floats(m, n, k)
    for p in range(PLAN_APPLE_MMA_SPLIT_BIG + 1):
        var f = identical_gemm_workspace_floats(m, n, k, p)
        if f <= 64 * 1024 * 1024 and f > wsn:
            wsn = f
    var ws = ctx.enqueue_create_buffer[DType.float32](max(wsn, 1))
    ctx.enqueue_function[fill_kernel](a.unsafe_ptr(), Int32(a_n), UInt32(17),
        grid_dim=((a_n + 255) // 256, 1, 1), block_dim=(256, 1, 1))
    ctx.enqueue_function[fill_kernel](b.unsafe_ptr(), Int32(b_n), UInt32(31),
        grid_dim=((b_n + 255) // 256, 1, 1), block_dim=(256, 1, 1))
    run_plan(ctx, cref, a, b, ws, m, n, k, op, -1)
    ctx.synchronize()
    var pick = cnn_pick(m, n, k, op)
    var shipped = choose_gemm_plan(m, n, k)
    var t_ship = time_plan(ctx, c, a, b, ws, m, n, k, op, -1)
    var t_cnn = t_ship
    if pick >= 0:
        t_cnn = time_plan(ctx, c, a, b, ws, m, n, k, op, pick)
    print("CNN-GEMM", name, op_name(op), "m", m, "n", n, "k", k,
          "shipped", gemm_plan_name(shipped), String(t_ship), "ms",
          "x_cnn", gemm_plan_name(pick) if pick >= 0 else String("shipped"), String(t_cnn), "ms", flush=True)
    var best = t_cnn
    var best_name = String("x_cnn")
    for p in range(PLAN_APPLE_MMA_SPLIT_BIG + 1):
        if (p == PLAN_APPLE_MMA or p == PLAN_APPLE_MMA_SPLIT or p == PLAN_APPLE_MMA_SPLIT_BIG) and not apple_mma_applies(m, n, k):
            continue
        if identical_gemm_workspace_floats(m, n, k, p) > wsn:
            print("CNN-GEMM-PLAN", name, p, gemm_plan_name(p), "skip (workspace)", flush=True)
            continue
        c.enqueue_fill(Float32(0))
        var t = time_plan(ctx, c, a, b, ws, m, n, k, op, p)
        var mis = count_mismatch(ctx, c, cref, m * n)
        print("CNN-GEMM-PLAN", name, p, gemm_plan_name(p), String(t), "ms mismatches", mis, flush=True)
        if mis == 0 and t < best:
            best = t
            best_name = gemm_plan_name(p)
    print("CNN-GEMM-BEST", name, best_name, String(best), "ms vs x_cnn", String(t_cnn), "ms", flush=True)
    _ = a^; _ = b^; _ = c^; _ = cref^; _ = ws^


def main() raises:
    var ctx = DeviceContext()
    print("CNN-GEMM-DEVICE", ctx.name(), flush=True)
    # CNNClassifier (32,64), 3x32x32, k3 pad 1, pool 2, batch 256
    sweep(ctx, "t_b1_fwd", 262144, 32, 27, OP_NT)
    sweep(ctx, "t_b1_dw", 32, 27, 262144, OP_TN)
    sweep(ctx, "t_b1_db", 32, 1, 262144, OP_TN)
    sweep(ctx, "t_b2_fwd", 65536, 64, 288, OP_NT)
    sweep(ctx, "t_b2_dw", 64, 288, 65536, OP_TN)
    sweep(ctx, "t_b2_db", 64, 1, 65536, OP_TN)
    sweep(ctx, "t_b2_dx", 65536, 288, 64, OP_NN)
    sweep(ctx, "t_lin_fwd", 256, 10, 4096, OP_NT)
    sweep(ctx, "t_lin_dw", 10, 4096, 256, OP_TN)
    sweep(ctx, "t_lin_db", 10, 1, 256, OP_TN)
    sweep(ctx, "t_lin_dx", 256, 4096, 10, OP_NN)
    # Conv2d N256 bench shapes
    sweep(ctx, "c64_fwd", 262144, 64, 576, OP_NT)
    sweep(ctx, "c64_dw", 64, 576, 262144, OP_TN)
    sweep(ctx, "c64_dx", 262144, 576, 64, OP_NN)
    sweep(ctx, "c3_fwd", 262144, 64, 27, OP_NT)
    sweep(ctx, "c3_dw", 64, 27, 262144, OP_TN)
    sweep(ctx, "c128_fwd", 65536, 128, 576, OP_NT)
    sweep(ctx, "c128_dw", 128, 576, 65536, OP_TN)
    sweep(ctx, "c128_dx", 65536, 576, 128, OP_NN)
