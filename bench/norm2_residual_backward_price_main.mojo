# SPDX-License-Identifier: Apache-2.0
"""Price the exact norm2-dx plus residual-fan-in fusion at GPT shapes."""

from std.os import getenv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from transformer.checks.transformer_backward import (
    BWD_TPB,
    _grid,
    bwd_add2_kernel,
    bwd_norm_dx_kernel,
)

comptime REPEATS = 9


def run_shape(ctx: DeviceContext, m: Int, dm: Int, fused: Bool) raises:
    var n = m * dm
    var dx = ctx.enqueue_create_buffer[DType.float32](n)
    var residual = ctx.enqueue_create_buffer[DType.float32](n)
    var branch = ctx.enqueue_create_buffer[DType.float32](n)
    var dprod = ctx.enqueue_create_buffer[DType.float32](n)
    var dh = ctx.enqueue_create_buffer[DType.float32](n)
    var dy = ctx.enqueue_create_buffer[DType.float32](n)
    var x = ctx.enqueue_create_buffer[DType.float32](n)
    var rstd = ctx.enqueue_create_buffer[DType.float32](m)
    var dv = ctx.enqueue_create_buffer[DType.float32](m)
    dx.enqueue_fill(Float32(0.0))
    residual.enqueue_fill(Float32(0.0))
    branch.enqueue_fill(Float32(0.125))
    dprod.enqueue_fill(Float32(0.0))
    dh.enqueue_fill(Float32(0.25))
    dy.enqueue_fill(Float32(0.375))
    x.enqueue_fill(Float32(0.5))
    rstd.enqueue_fill(Float32(0.75))
    dv.enqueue_fill(Float32(0.0625))
    ctx.synchronize()

    for r in range(REPEATS + 1):
        var t0 = perf_counter_ns()
        ctx.enqueue_function[bwd_norm_dx_kernel](
            dx.unsafe_ptr(), residual.unsafe_ptr(), branch.unsafe_ptr(),
            dprod.unsafe_ptr(), dh.unsafe_ptr(), dy.unsafe_ptr(),
            x.unsafe_ptr(), rstd.unsafe_ptr(), dv.unsafe_ptr(), Int32(m),
            Int32(dm), Int32(1 if fused else 0),
            grid_dim=(_grid(n), 1, 1), block_dim=(BWD_TPB, 1, 1),
        )
        if not fused:
            ctx.enqueue_function[bwd_add2_kernel](
                residual.unsafe_ptr(), dx.unsafe_ptr(), branch.unsafe_ptr(),
                Int32(n), grid_dim=(_grid(n), 1, 1),
                block_dim=(BWD_TPB, 1, 1),
            )
        ctx.synchronize()
        if r > 0:
            print(
                "PRICE", "fused" if fused else "split",
                String(m) + "x" + String(dm),
                Float64(perf_counter_ns() - t0) / 1.0e6,
            )
    _ = dx^
    _ = residual^
    _ = branch^
    _ = dprod^
    _ = dh^
    _ = dy^
    _ = x^
    _ = rstd^
    _ = dv^


def main() raises:
    var fused = getenv("MOJOLEARN_NORM2_RESIDUAL_ARM", "split") == "fused"
    var ctx = DeviceContext()
    run_shape(ctx, 2048, 768, fused)
    run_shape(ctx, 8192, 768, fused)
    run_shape(ctx, 32768, 768, fused)
