# SPDX-License-Identifier: Apache-2.0
"""Interleaved old/new split tiles across the proposed dispatch domain."""
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from gemm.checks.gemm_tuned_probe import _fill, _poison, _digest
from gemm.checks.gemm_identical import (
    PLAN_SPLIT_64_4X4, PLAN_SPLIT_128_8X8, identical_gemm_with_plan,
    identical_gemm_workspace_floats, choose_gemm_plan,
)
from gemm.checks.gemm_oracle import OP_TN


def main() raises:
    var ms: List[Int] = [128, 128, 256, 256, 128, 512, 256, 512]
    var ns: List[Int] = [128, 256, 128, 256, 512, 128, 512, 256]
    var ks: List[Int] = [65536, 100003]
    var matches = 0
    for k in ks:
        for shape in range(len(ms)):
            var m = ms[shape]
            var n = ns[shape]
            var ctx = DeviceContext()
            var a = ctx.enqueue_create_buffer[DType.float32](m * k)
            var b = ctx.enqueue_create_buffer[DType.float32](n * k)
            var c = ctx.enqueue_create_buffer[DType.float32](m * n)
            var ws = ctx.enqueue_create_buffer[DType.float32](
                identical_gemm_workspace_floats(m, n, k, PLAN_SPLIT_128_8X8)
            )
            _fill(ctx, a, m * k, 17 + shape)
            _fill(ctx, b, n * k, 31 + shape)
            _poison(ctx, c, m * n)
            identical_gemm_with_plan(ctx, c, a, b, ws, m, n, k, OP_TN, PLAN_SPLIT_64_4X4)
            ctx.synchronize()
            var before = _digest(ctx, c, m * n)
            _poison(ctx, c, m * n)
            identical_gemm_with_plan(ctx, c, a, b, ws, m, n, k, OP_TN, PLAN_SPLIT_128_8X8)
            ctx.synchronize()
            var after = _digest(ctx, c, m * n)
            if before[1] != 0 or after[1] != 0 or before[0] != after[0]:
                raise Error("wide split tile moved bits or left poison")
            var old_ns = 0
            var new_ns = 0
            for round in range(7):
                # Alternate which plan goes first to reduce order bias.
                for arm in range(2):
                    var wide = (arm + round) % 2 == 1
                    var plan = PLAN_SPLIT_128_8X8 if wide else PLAN_SPLIT_64_4X4
                    var start = perf_counter_ns()
                    identical_gemm_with_plan(ctx, c, a, b, ws, m, n, k, OP_TN, plan)
                    ctx.synchronize()
                    var elapsed = perf_counter_ns() - start
                    if wide:
                        new_ns += elapsed
                    else:
                        old_ns += elapsed
            print("WIDE_SPLIT m=" + String(m) + " n=" + String(n) + " k=" + String(k)
                  + " bits=equal chosen=" + String(choose_gemm_plan(m, n, k))
                  + " old_ms=" + String(Float64(old_ns) / 7e6)
                  + " new_ms=" + String(Float64(new_ns) / 7e6)
                  + " ratio=" + String(Float64(new_ns) / Float64(old_ns)))
            matches += 1
            _ = a^
            _ = b^
            _ = c^
            _ = ws^
            _ = ctx^
    print("WIDE_SPLIT PASS " + String(matches) + " shapes")
