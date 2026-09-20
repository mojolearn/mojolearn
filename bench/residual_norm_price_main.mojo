"""Exact repeated-stage price for residual add followed by RMSNorm."""

from std.memory import bitcast
from std.sys.compile import is_defined
from std.time import perf_counter_ns
from max.gpu.host import DeviceBuffer, DeviceContext
from transformer.impl.llama.modeling_llama import (
    LLAMA_TPB,
    llama_rms_norm,
    residual_rms_norm_kernel,
)
from mamba.impl.modeling.modeling_mamba import residual_add_kernel

comptime M = (
    32768 if is_defined["MOJOLEARN_RESNORM_BATCH16"]() else
    8192 if is_defined["MOJOLEARN_RESNORM_BATCH4"]() else
    4096 if is_defined["MOJOLEARN_RESNORM_BATCH2"]() else
    2048
)
comptime DM = 768
comptime N = M * DM
comptime LAYERS = 12


def fill(ctx: DeviceContext, mut d: DeviceBuffer[DType.float32], salt: Int) raises:
    var h = ctx.enqueue_create_host_buffer[DType.float32](N)
    ctx.synchronize()
    for i in range(N):
        var word = UInt32(0x3D000000) | UInt32(((i + salt) * 2654435761) & 0x7FFFFF)
        h.unsafe_ptr().unsafe_store(i, bitcast[DType.float32](word))
    ctx.enqueue_copy(dst_buf=d, src_ptr=h.unsafe_ptr())
    ctx.synchronize()


def digest(ctx: DeviceContext, mut d: DeviceBuffer[DType.float32], n: Int) raises -> UInt64:
    var h = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=d)
    ctx.synchronize()
    var x = UInt64(0xCBF29CE484222325)
    for i in range(n):
        x = (x ^ UInt64(bitcast[DType.uint32](h.unsafe_ptr().unsafe_load(i)))) * UInt64(0x100000001B3)
    return x


def baseline(ctx: DeviceContext, mut r: DeviceBuffer[DType.float32], mut ss: DeviceBuffer[DType.float32], mut out: DeviceBuffer[DType.float32], mut a: DeviceBuffer[DType.float32], mut b: DeviceBuffer[DType.float32], mut w: DeviceBuffer[DType.float32]) raises:
    for _ in range(LAYERS):
        ctx.enqueue_function[residual_add_kernel](r.unsafe_ptr(), a.unsafe_ptr(), b.unsafe_ptr(), Int32(N), grid_dim=((N + LLAMA_TPB - 1) // LLAMA_TPB, 1, 1), block_dim=(LLAMA_TPB, 1, 1))
        llama_rms_norm(ctx, ss, out, r, w, M, DM, Float32(1e-6))


def fused(ctx: DeviceContext, mut r: DeviceBuffer[DType.float32], mut ss: DeviceBuffer[DType.float32], mut out: DeviceBuffer[DType.float32], mut a: DeviceBuffer[DType.float32], mut b: DeviceBuffer[DType.float32], mut w: DeviceBuffer[DType.float32]) raises:
    for _ in range(LAYERS):
        ctx.enqueue_function[residual_rms_norm_kernel](r.unsafe_ptr(), ss.unsafe_ptr(), out.unsafe_ptr(), a.unsafe_ptr(), b.unsafe_ptr(), w.unsafe_ptr(), Int32(M), Int32(DM), Float32(1e-6), grid_dim=((M + LLAMA_TPB - 1) // LLAMA_TPB, 1, 1), block_dim=(LLAMA_TPB, 1, 1))


def main() raises:
    var ctx = DeviceContext()
    var a = ctx.enqueue_create_buffer[DType.float32](N)
    var b = ctx.enqueue_create_buffer[DType.float32](N)
    var w = ctx.enqueue_create_buffer[DType.float32](DM)
    var r0 = ctx.enqueue_create_buffer[DType.float32](N)
    var r1 = ctx.enqueue_create_buffer[DType.float32](N)
    var s0 = ctx.enqueue_create_buffer[DType.float32](M)
    var s1 = ctx.enqueue_create_buffer[DType.float32](M)
    var o0 = ctx.enqueue_create_buffer[DType.float32](N)
    var o1 = ctx.enqueue_create_buffer[DType.float32](N)
    ctx.synchronize()
    fill(ctx, a, 17)
    fill(ctx, b, 41)
    var wh = ctx.enqueue_create_host_buffer[DType.float32](DM)
    ctx.synchronize()
    for i in range(DM):
        wh.unsafe_ptr().unsafe_store(i, Float32(1.0) + Float32(i % 11) * Float32(0.001))
    ctx.enqueue_copy(dst_buf=w, src_ptr=wh.unsafe_ptr())
    baseline(ctx, r0, s0, o0, a, b, w)
    fused(ctx, r1, s1, o1, a, b, w)
    ctx.synchronize()
    print("BITS residual=", digest(ctx, r0, N), "/", digest(ctx, r1, N), " sumsq=", digest(ctx, s0, M), "/", digest(ctx, s1, M), " out=", digest(ctx, o0, N), "/", digest(ctx, o1, N))
    for rep in range(11):
        var t0 = perf_counter_ns()
        baseline(ctx, r0, s0, o0, a, b, w)
        ctx.synchronize()
        var t1 = perf_counter_ns()
        fused(ctx, r1, s1, o1, a, b, w)
        ctx.synchronize()
        var t2 = perf_counter_ns()
        print("PRICE rep=", rep, " baseline_ms=", Float64(t1-t0)/1e6, " fused_ms=", Float64(t2-t1)/1e6)
