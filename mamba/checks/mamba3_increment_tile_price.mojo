# SPDX-License-Identifier: Apache-2.0
"""Interleaved device price for the two bit-equal Mamba-3 increment tiles."""
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from mamba.impl.ops.mamba3_siso import (
    m3_state_increment_shared_v_kernel,
    m3_state_increment_tiled_kernel,
)


def one_shape(ctx: DeviceContext, name: String, b: Int, t: Int, nh: Int) raises:
    comptime q = 64
    comptime pn = 64 * 128
    var nc = (t + q - 1) // q
    var k = ctx.enqueue_create_buffer[DType.float32](b * t * nh * 128)
    var v = ctx.enqueue_create_buffer[DType.float32](b * t * nh * 64)
    var decay = ctx.enqueue_create_buffer[DType.float32](b * nh * nc * (q + 1))
    var out = ctx.enqueue_create_buffer[DType.float32](b * nc * nh * pn)
    k.enqueue_fill(Float32(0.03125))
    v.enqueue_fill(Float32(-0.0625))
    decay.enqueue_fill(Float32(0.5))
    out.enqueue_fill(Float32(0.0))
    ctx.synchronize()
    for round in range(7):
        for pos in range(3):
            var arm = (round + pos) % 3
            var start = perf_counter_ns()
            if arm == 2:
                ctx.enqueue_function[m3_state_increment_tiled_kernel[16, 16]](
                    out.unsafe_ptr(), k.unsafe_ptr(), v.unsafe_ptr(), decay.unsafe_ptr(),
                    Int32(b), Int32(t), Int32(nh), Int32(nc), Int32(q),
                    grid_dim=(b * nc * nh * 32, 1, 1), block_dim=(256, 1, 1),
                )
            elif arm == 1:
                ctx.enqueue_function[m3_state_increment_tiled_kernel[8, 32]](
                    out.unsafe_ptr(), k.unsafe_ptr(), v.unsafe_ptr(), decay.unsafe_ptr(),
                    Int32(b), Int32(t), Int32(nh), Int32(nc), Int32(q),
                    grid_dim=(b * nc * nh * 32, 1, 1), block_dim=(256, 1, 1),
                )
            else:
                ctx.enqueue_function[m3_state_increment_shared_v_kernel](
                    out.unsafe_ptr(), k.unsafe_ptr(), v.unsafe_ptr(), decay.unsafe_ptr(),
                    Int32(b), Int32(t), Int32(nh), Int32(nc), Int32(q),
                    grid_dim=((b * nc * nh * pn + 255) // 256, 1, 1),
                    block_dim=(256, 1, 1),
                )
            ctx.synchronize()
            var elapsed = Float64(perf_counter_ns() - start) / 1.0e6
            var arm_name = "current_shared_v"
            if arm == 1:
                arm_name = "tiled_8x32"
            elif arm == 2:
                arm_name = "balanced_16x16"
            print("M3_INCREMENT_PRICE", name, "round", round, "arm", arm_name, "ms", elapsed)
    _ = k^
    _ = v^
    _ = decay^
    _ = out^


def main() raises:
    var ctx = DeviceContext()
    one_shape(ctx, "narrow.b8_l4096_d512", 8, 4096, 16)
    one_shape(ctx, "wide.b8_l1024_d2048", 8, 1024, 64)
