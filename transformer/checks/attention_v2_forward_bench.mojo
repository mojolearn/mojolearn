# SPDX-License-Identifier: Apache-2.0
"""Forward-only opt-in v2 timing and actual resident allocation witness."""
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from transformer.impl.llama.attention_v2 import enqueue_attention_v2_forward

def main() raises:
    comptime H = 12
    comptime L = 1024
    comptime HD = 64
    comptime R = H * L
    var ctx = DeviceContext()
    var q = ctx.enqueue_create_buffer[DType.float32](R * HD)
    var k = ctx.enqueue_create_buffer[DType.float32](H * L * HD)
    var v = ctx.enqueue_create_buffer[DType.float32](H * L * HD)
    var lo = ctx.enqueue_create_buffer[DType.int32](R)
    var hi = ctx.enqueue_create_buffer[DType.int32](R)
    var out = ctx.enqueue_create_buffer[DType.float32](R * HD)
    var m = ctx.enqueue_create_buffer[DType.float32](R)
    var z = ctx.enqueue_create_buffer[DType.float32](R)
    var hlo = ctx.enqueue_create_host_buffer[DType.int32](R)
    var hhi = ctx.enqueue_create_host_buffer[DType.int32](R)
    for r in range(R):
        hlo[r] = 0
        hhi[r] = Int32((r % L) + 1)
    ctx.enqueue_memset(q, Float32(0.03125))
    ctx.enqueue_memset(k, Float32(-0.0625))
    ctx.enqueue_memset(v, Float32(0.125))
    ctx.enqueue_copy(dst_buf=lo, src_buf=hlo)
    ctx.enqueue_copy(dst_buf=hi, src_buf=hhi)
    ctx.synchronize()
    var allocated = (R * HD + 2 * H * L * HD + 2 * R * HD + 2 * R) * 4 + 2 * R * 4
    for rep in range(4):
        var t0 = perf_counter_ns()
        enqueue_attention_v2_forward(ctx, q, k, v, lo, hi, out, m, z, R, L, HD, HD, L, Float32(0.125))
        ctx.synchronize()
        print("attention_v2 B1 H12 L1024 HD64 rep", rep, "ms", Float64(perf_counter_ns()-t0)/1e6, "resident_bytes", allocated, "quadratic_score_bytes", R*L*4)
    _ = q^; _ = k^; _ = v^; _ = lo^; _ = hi^; _ = out^; _ = m^; _ = z^; _ = hlo^; _ = hhi^
