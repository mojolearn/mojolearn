"""Exact A/B price for the production bias epilogue followed by GELU."""
from std.memory import bitcast
from std.time import perf_counter_ns
from max.gpu.host import DeviceBuffer, DeviceContext
from transformer.impl.llama.modeling_llama import (
    LLAMA_TPB,
    add_bias_kernel,
    bias_gelu_kernel,
    gelu_kernel,
)

comptime M = 32768
comptime WIDTH = 2304
comptime N = M * WIDTH


def fill(ctx: DeviceContext, mut d: DeviceBuffer[DType.float32]) raises:
    var h = ctx.enqueue_create_host_buffer[DType.float32](N)
    ctx.synchronize()
    for i in range(N):
        var word = UInt32(0x3D800000) | UInt32((i * 2654435761 + 17) & 0x7FFFFF)
        h.unsafe_ptr().unsafe_store(i, bitcast[DType.float32](word))
    ctx.enqueue_copy(dst_buf=d, src_ptr=h.unsafe_ptr())
    ctx.synchronize()


def baseline(
    ctx: DeviceContext,
    mut out: DeviceBuffer[DType.float32],
    mut src: DeviceBuffer[DType.float32],
    mut bias: DeviceBuffer[DType.float32],
) raises:
    ctx.enqueue_function[add_bias_kernel](
        src.unsafe_ptr(),
        bias.unsafe_ptr(),
        Int32(N),
        Int32(WIDTH),
        grid_dim=((N + LLAMA_TPB - 1) // LLAMA_TPB, 1, 1),
        block_dim=(LLAMA_TPB, 1, 1),
    )
    ctx.enqueue_function[gelu_kernel](
        out.unsafe_ptr(),
        src.unsafe_ptr(),
        Int32(N),
        Int32(0),
        grid_dim=((N + LLAMA_TPB - 1) // LLAMA_TPB, 1, 1),
        block_dim=(LLAMA_TPB, 1, 1),
    )


def fused(
    ctx: DeviceContext,
    mut out: DeviceBuffer[DType.float32],
    mut src: DeviceBuffer[DType.float32],
    mut bias: DeviceBuffer[DType.float32],
) raises:
    ctx.enqueue_function[bias_gelu_kernel](
        out.unsafe_ptr(),
        src.unsafe_ptr(),
        bias.unsafe_ptr(),
        Int32(N),
        Int32(WIDTH),
        Int32(0),
        grid_dim=((N + LLAMA_TPB - 1) // LLAMA_TPB, 1, 1),
        block_dim=(LLAMA_TPB, 1, 1),
    )


def digest(
    ctx: DeviceContext, mut d: DeviceBuffer[DType.float32]
) raises -> UInt64:
    var h = ctx.enqueue_create_host_buffer[DType.float32](N)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=d)
    ctx.synchronize()
    var x = UInt64(0xCBF29CE484222325)
    for i in range(N):
        x = (
            x ^ UInt64(bitcast[DType.uint32](h.unsafe_ptr().unsafe_load(i)))
        ) * UInt64(0x100000001B3)
    return x


def main() raises:
    var ctx = DeviceContext()
    var pristine = ctx.enqueue_create_buffer[DType.float32](N)
    var s0 = ctx.enqueue_create_buffer[DType.float32](N)
    var s1 = ctx.enqueue_create_buffer[DType.float32](N)
    var o0 = ctx.enqueue_create_buffer[DType.float32](N)
    var o1 = ctx.enqueue_create_buffer[DType.float32](N)
    var bias = ctx.enqueue_create_buffer[DType.float32](WIDTH)
    ctx.synchronize()
    fill(ctx, pristine)
    var bh = ctx.enqueue_create_host_buffer[DType.float32](WIDTH)
    ctx.synchronize()
    for i in range(WIDTH):
        bh.unsafe_ptr().unsafe_store(i, Float32(i % 17 - 8) * Float32(0.001))
    ctx.enqueue_copy(dst_buf=bias, src_ptr=bh.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=s0, src_buf=pristine)
    ctx.enqueue_copy(dst_buf=s1, src_buf=pristine)
    ctx.synchronize()
    baseline(ctx, o0, s0, bias)
    fused(ctx, o1, s1, bias)
    ctx.synchronize()
    print(
        "BITS out0=",
        digest(ctx, o0),
        " out1=",
        digest(ctx, o1),
        " src0=",
        digest(ctx, s0),
        " src1=",
        digest(ctx, s1),
    )
    for r in range(9):
        ctx.enqueue_copy(dst_buf=s0, src_buf=pristine)
        ctx.enqueue_copy(dst_buf=s1, src_buf=pristine)
        ctx.synchronize()
        if r % 2 == 0:
            var t0 = perf_counter_ns()
            baseline(ctx, o0, s0, bias)
            ctx.synchronize()
            var t1 = perf_counter_ns()
            fused(ctx, o1, s1, bias)
            ctx.synchronize()
            var t2 = perf_counter_ns()
            print(
                "SAMPLE round=",
                r,
                " baseline_ns=",
                t1 - t0,
                " fused_ns=",
                t2 - t1,
            )
        else:
            var t0 = perf_counter_ns()
            fused(ctx, o1, s1, bias)
            ctx.synchronize()
            var t1 = perf_counter_ns()
            baseline(ctx, o0, s0, bias)
            ctx.synchronize()
            var t2 = perf_counter_ns()
            print(
                "SAMPLE round=",
                r,
                " baseline_ns=",
                t2 - t1,
                " fused_ns=",
                t1 - t0,
            )
