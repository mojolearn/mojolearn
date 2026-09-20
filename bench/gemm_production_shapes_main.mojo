# SPDX-License-Identifier: Apache-2.0
"""Local IDENTICAL GEMM card for large transformer-training consumers."""
from std.memory import bitcast
from std.time import perf_counter_ns
from std.gpu import block_dim, block_idx, thread_idx
from std.os import getenv
from std.atomic import Atomic
from max.gpu.host import DeviceBuffer, DeviceContext
from gemm.checks.gemm_identical import (
    choose_gemm_plan, gemm_plan_name, identical_gemm_into,
    identical_gemm_with_plan,
    identical_gemm_workspace_max_floats,
)
from gemm.checks.gemm_oracle import OP_NT


def fill_kernel(p: MutPointer[Float32, MutAnyOrigin], n: Int32, salt: UInt32):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n):
        var x = UInt32(i) * UInt32(1664525) + salt
        p.unsafe_store(i, Float32(Int(x & UInt32(1023)) - 511) / Float32(1024))


def hash_kernel(p: MutPointer[Float32, MutAnyOrigin], n: Int32,
                result: MutPointer[UInt32, MutAnyOrigin]):
    # Diagnostic samples, not a commutative whole-buffer digest: fixed cells
    # cover every edge/tile and are stable enough to catch a wrong dispatch.
    if Int(thread_idx.x) == 0:
        var acc = UInt32(2166136261)
        var step = max(Int(n) // 4096, 1)
        var i = 0
        while i < Int(n):
            acc = (acc ^ bitcast[DType.uint32](p.unsafe_load(i))) * UInt32(16777619)
            i += step
        acc = (acc ^ bitcast[DType.uint32](p.unsafe_load(Int(n) - 1))) * UInt32(16777619)
        result.unsafe_store(0, acc)


def mismatch_kernel(a: MutPointer[Float32, MutAnyOrigin],
                    b: MutPointer[Float32, MutAnyOrigin], n: Int32,
                    count: MutPointer[Int32, MutAnyOrigin]):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n) and bitcast[DType.uint32](a.unsafe_load(i)) != bitcast[DType.uint32](b.unsafe_load(i)):
        _ = Atomic.fetch_add(count.unsafe_offset(0), Int32(1))


def run(ctx: DeviceContext, name: String, m: Int, n: Int, k: Int) raises:
    var forced = -1
    var forced_text = String(getenv("MOJOLEARN_PROD_GEMM_PLAN"))
    if forced_text.byte_length() > 0:
        forced = Int(forced_text)
    var a = ctx.enqueue_create_buffer[DType.float32](m * k)
    var b = ctx.enqueue_create_buffer[DType.float32](n * k)
    var c = ctx.enqueue_create_buffer[DType.float32](m * n)
    var cref = ctx.enqueue_create_buffer[DType.float32](m * n)
    var ws_n = identical_gemm_workspace_max_floats(m, n, k)
    var ws = ctx.enqueue_create_buffer[DType.float32](max(ws_n, 1))
    var hd = ctx.enqueue_create_host_buffer[DType.uint32](1)
    var hm = ctx.enqueue_create_host_buffer[DType.int32](1)
    hm.unsafe_ptr().unsafe_store(0, Int32(0))
    ctx.enqueue_function[fill_kernel](a.unsafe_ptr(), Int32(m * k), UInt32(17),
        grid_dim=((m * k + 255) // 256, 1, 1), block_dim=(256, 1, 1))
    ctx.enqueue_function[fill_kernel](b.unsafe_ptr(), Int32(n * k), UInt32(31),
        grid_dim=((n * k + 255) // 256, 1, 1), block_dim=(256, 1, 1))
    identical_gemm_with_plan(ctx, cref, a, b, ws, m, n, k, OP_NT, 10)
    if forced >= 0:
        identical_gemm_with_plan(ctx, c, a, b, ws, m, n, k, OP_NT, forced)
    else:
        identical_gemm_into(ctx, c, a, b, ws, m, n, k, OP_NT)
    ctx.synchronize()
    var samples = List[Int]()
    for rep in range(5):
        var t0 = perf_counter_ns()
        if forced >= 0:
            identical_gemm_with_plan(ctx, c, a, b, ws, m, n, k, OP_NT, forced)
        else:
            identical_gemm_into(ctx, c, a, b, ws, m, n, k, OP_NT)
        ctx.synchronize()
        samples.append(perf_counter_ns() - t0)
    var dh = ctx.enqueue_create_buffer[DType.uint32](1)
    var dm = ctx.enqueue_create_buffer[DType.int32](1)
    ctx.enqueue_copy(dst_buf=dm, src_ptr=hm.unsafe_ptr())
    ctx.enqueue_function[mismatch_kernel](cref.unsafe_ptr(), c.unsafe_ptr(), Int32(m*n), dm.unsafe_ptr(),
        grid_dim=((m*n + 255)//256, 1, 1), block_dim=(256, 1, 1))
    ctx.enqueue_function[hash_kernel](c.unsafe_ptr(), Int32(m * n), dh.unsafe_ptr(),
        grid_dim=(1, 1, 1), block_dim=(1, 1, 1))
    ctx.enqueue_copy(dst_ptr=hd.unsafe_ptr(), src_buf=dh)
    ctx.enqueue_copy(dst_ptr=hm.unsafe_ptr(), src_buf=dm)
    ctx.synchronize()
    print("PROD_GEMM", name, "m", m, "n", n, "k", k,
          "plan", gemm_plan_name(forced if forced >= 0 else choose_gemm_plan(m, n, k)),
          "workspace_floats", ws_n, "output_mib", Float64(m*n*4)/1048576.0,
          "samples_ns", samples, "hash", hd.unsafe_ptr().unsafe_load(0),
          "mismatches_vs_plan10", hm.unsafe_ptr().unsafe_load(0))
    _ = a^; _ = b^; _ = c^; _ = cref^; _ = ws^; _ = dh^; _ = dm^; _ = hd^; _ = hm^


def main() raises:
    var ctx = DeviceContext()
    print("PROD_GEMM_DEVICE", ctx.name())
    run(ctx, "qkv_32768x2304x768", 32768, 2304, 768)
    run(ctx, "mlp_32768x3072x768", 32768, 3072, 768)
    run(ctx, "head_chunk_32768x1024x768", 32768, 1024, 768)
