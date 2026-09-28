# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""What one enqueue costs on this stack, host side and end to end: a tiny
kernel launch, a small host-to-device copy, a small device-to-host copy, and
a synchronize with nothing queued. The GBDT Lossguide loop pays ~20 of these
per leaf split (trees-apple2 lead), so this prices each kind. Measurement
tooling only.

    pixi run mojo run -I . bench/speed/metal_enqueue_cost_main.mojo
"""

from max.gpu.host import DeviceContext
from std.gpu import block_idx, thread_idx
from std.time import perf_counter_ns


def touch_kernel(p: MutPointer[Int32, MutAnyOrigin], n: Int32):
    var i = Int(block_idx.x) * 64 + Int(thread_idx.x)
    if i < Int(n):
        p[unsafe_offset=i] = p[unsafe_offset=i] + 1


def main() raises:
    var ctx = DeviceContext()
    comptime N = 64
    comptime REPS = 2000
    var a = ctx.enqueue_create_buffer[DType.int32](N)
    var h = ctx.enqueue_create_host_buffer[DType.int32](N)
    for i in range(N):
        h.unsafe_ptr().unsafe_store(i, Int32(i))
    ctx.synchronize()

    for rnd in range(2):
        # kernel launches, host enqueue time, then the drain
        var t0 = perf_counter_ns()
        for _ in range(REPS):
            ctx.enqueue_function[touch_kernel](
                a.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
                Int32(N),
                grid_dim=1,
                block_dim=64,
            )
        var t1 = perf_counter_ns()
        ctx.synchronize()
        var t2 = perf_counter_ns()
        print("ENQ round", rnd, "launch host_us", Float64(t1 - t0) / Float64(REPS) / 1000.0, "total_us", Float64(t2 - t0) / Float64(REPS) / 1000.0)

        # small host-to-device copies (256 B)
        t0 = perf_counter_ns()
        for _ in range(REPS):
            ctx.enqueue_copy(dst_buf=a, src_ptr=h.unsafe_ptr())
        t1 = perf_counter_ns()
        ctx.synchronize()
        t2 = perf_counter_ns()
        print("ENQ round", rnd, "h2d host_us", Float64(t1 - t0) / Float64(REPS) / 1000.0, "total_us", Float64(t2 - t0) / Float64(REPS) / 1000.0)

        # launch + h2d interleaved (the Lossguide pattern)
        t0 = perf_counter_ns()
        for _ in range(REPS):
            ctx.enqueue_copy(dst_buf=a, src_ptr=h.unsafe_ptr())
            ctx.enqueue_function[touch_kernel](
                a.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
                Int32(N),
                grid_dim=1,
                block_dim=64,
            )
        t1 = perf_counter_ns()
        ctx.synchronize()
        t2 = perf_counter_ns()
        print("ENQ round", rnd, "h2d+launch host_us", Float64(t1 - t0) / Float64(REPS) / 1000.0, "total_us", Float64(t2 - t0) / Float64(REPS) / 1000.0)

        # one launch, one d2h copy, one synchronize (a host wait)
        t0 = perf_counter_ns()
        for _ in range(REPS // 4):
            ctx.enqueue_function[touch_kernel](
                a.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
                Int32(N),
                grid_dim=1,
                block_dim=64,
            )
            ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=a)
            ctx.synchronize()
        t1 = perf_counter_ns()
        print("ENQ round", rnd, "launch+d2h+sync us", Float64(t1 - t0) / Float64(REPS // 4) / 1000.0)

        # a synchronize with nothing queued
        t0 = perf_counter_ns()
        for _ in range(REPS):
            ctx.synchronize()
        t1 = perf_counter_ns()
        print("ENQ round", rnd, "empty sync us", Float64(t1 - t0) / Float64(REPS) / 1000.0)
