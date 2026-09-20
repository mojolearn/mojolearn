# SPDX-License-Identifier: Apache-2.0
"""Price the exact gate-product plus SiLU-backward fusion at GPT shapes."""

from std.time import perf_counter_ns
from max.gpu.host import DeviceBuffer, DeviceContext
from transformer.checks.transformer_backward import (
    BWD_TPB,
    _grid,
    bwd_mul2_kernel,
    bwd_mul2_silu_backward_kernel,
    bwd_silu_backward_kernel,
)

comptime REPEATS = 7


def _split(
    ctx: DeviceContext,
    mut dsi: DeviceBuffer[DType.float32],
    mut dup: DeviceBuffer[DType.float32],
    mut dg: DeviceBuffer[DType.float32],
    mut dy: DeviceBuffer[DType.float32],
    mut up: DeviceBuffer[DType.float32],
    mut silu: DeviceBuffer[DType.float32],
    mut gate: DeviceBuffer[DType.float32],
    n: Int,
) raises:
    ctx.enqueue_function[bwd_mul2_kernel](
        dsi.unsafe_ptr(), dup.unsafe_ptr(), dy.unsafe_ptr(), up.unsafe_ptr(),
        silu.unsafe_ptr(), Int32(n), grid_dim=(_grid(n), 1, 1),
        block_dim=(BWD_TPB, 1, 1),
    )
    ctx.enqueue_function[bwd_silu_backward_kernel](
        dg.unsafe_ptr(), dsi.unsafe_ptr(), gate.unsafe_ptr(),
        silu.unsafe_ptr(), Int32(n), grid_dim=(_grid(n), 1, 1),
        block_dim=(BWD_TPB, 1, 1),
    )


def _fused(
    ctx: DeviceContext,
    mut dsi: DeviceBuffer[DType.float32],
    mut dup: DeviceBuffer[DType.float32],
    mut dg: DeviceBuffer[DType.float32],
    mut dy: DeviceBuffer[DType.float32],
    mut up: DeviceBuffer[DType.float32],
    mut silu: DeviceBuffer[DType.float32],
    mut gate: DeviceBuffer[DType.float32],
    n: Int,
) raises:
    ctx.enqueue_function[bwd_mul2_silu_backward_kernel](
        dsi.unsafe_ptr(), dup.unsafe_ptr(), dg.unsafe_ptr(), dy.unsafe_ptr(),
        up.unsafe_ptr(), silu.unsafe_ptr(), gate.unsafe_ptr(), Int32(n),
        grid_dim=(_grid(n), 1, 1), block_dim=(BWD_TPB, 1, 1),
    )


def run_shape(ctx: DeviceContext, m: Int, it: Int) raises:
    var n = m * it
    var dsi = ctx.enqueue_create_buffer[DType.float32](n)
    var dup = ctx.enqueue_create_buffer[DType.float32](n)
    var dg = ctx.enqueue_create_buffer[DType.float32](n)
    var dy = ctx.enqueue_create_buffer[DType.float32](n)
    var up = ctx.enqueue_create_buffer[DType.float32](n)
    var silu = ctx.enqueue_create_buffer[DType.float32](n)
    var gate = ctx.enqueue_create_buffer[DType.float32](n)
    dsi.enqueue_fill(Float32(0.0))
    dup.enqueue_fill(Float32(0.0))
    dg.enqueue_fill(Float32(0.0))
    dy.enqueue_fill(Float32(0.125))
    up.enqueue_fill(Float32(0.25))
    silu.enqueue_fill(Float32(0.375))
    gate.enqueue_fill(Float32(0.5))
    ctx.synchronize()

    for r in range(REPEATS + 1):
        # Alternate whole-call order to avoid assigning drift to one arm.
        if r % 2 == 0:
            var t0 = perf_counter_ns()
            _split(ctx, dsi, dup, dg, dy, up, silu, gate, n)
            ctx.synchronize()
            var split_ns = perf_counter_ns() - t0
            t0 = perf_counter_ns()
            _fused(ctx, dsi, dup, dg, dy, up, silu, gate, n)
            ctx.synchronize()
            var fused_ns = perf_counter_ns() - t0
            if r > 0:
                print("PRICE", String(m) + "x" + String(it), "split", Float64(split_ns) / 1.0e6, "fused", Float64(fused_ns) / 1.0e6)
        else:
            var t0 = perf_counter_ns()
            _fused(ctx, dsi, dup, dg, dy, up, silu, gate, n)
            ctx.synchronize()
            var fused_ns = perf_counter_ns() - t0
            t0 = perf_counter_ns()
            _split(ctx, dsi, dup, dg, dy, up, silu, gate, n)
            ctx.synchronize()
            var split_ns = perf_counter_ns() - t0
            if r > 0:
                print("PRICE", String(m) + "x" + String(it), "split", Float64(split_ns) / 1.0e6, "fused", Float64(fused_ns) / 1.0e6)
    _ = dsi^
    _ = dup^
    _ = dg^
    _ = dy^
    _ = up^
    _ = silu^
    _ = gate^


def main() raises:
    var ctx = DeviceContext()
    run_shape(ctx, 2048, 3072)
    run_shape(ctx, 8192, 3072)
    run_shape(ctx, 32768, 3072)
