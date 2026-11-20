# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Launch cost model probe (lane/metal-launch-overhead).

Measures the HOST-SIDE cost of the four device operations the training
steps are made of, on whatever device `DeviceContext()` opens (Metal on the
Mac). Nothing here is a numerics check: the kernels write a constant and
the result is never compared. Run it alone on the GPU:

    bash mac_slot.sh metal pixi run mojo run -I . training/checks/metal_launch_probe.mojo

Arms, each timed over N operations (PROBE_N, default 1000), repeated
PROBE_REPEATS times (default 3); the line shape is

    probe <arm> rep <r> n <N> total_us <t> per_op_us <p>
    probe <arm> summary n <N> min_us <a> median_us <b>

  launch_only     N one-thread launches, ONE synchronize at the end
  launch_sync     N times (one-thread launch, synchronize)
  launch_d2h_sync N times (one-thread launch, 4 B device-to-host copy,
                  synchronize)
  launch_4096     N launches of 4096 threads, ONE synchronize at the end
  sync_only       N synchronizes on an empty queue
  d2h_only        N times (4 B device-to-host copy, synchronize)

`launch_only` versus `launch_4096` shows whether a launch's cost depends
on its grid; `launch_sync` minus `launch_only` is the price of a host
round trip; `sync_only` is that price on an idle queue.
"""

from max.gpu import block_dim, block_idx, thread_idx
from std.os import getenv
from std.time import perf_counter_ns
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

comptime PROBE_TPB = 128


def probe_write_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    value: Float32,
    n: Int32,
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n):
        return
    dst.unsafe_store(i, value)


def _env_int(name: String, default: Int) raises -> Int:
    var s = String(getenv(name))
    if s == "":
        return default
    return Int(s)


def _launch(ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], n: Int, value: Float32) raises:
    var blocks = (n + PROBE_TPB - 1) // PROBE_TPB
    ctx.enqueue_function[probe_write_kernel](
        buf.unsafe_ptr(), value, Int32(n),
        grid_dim=(blocks, 1, 1), block_dim=(PROBE_TPB, 1, 1),
    )


def _report(arm: String, rep: Int, n: Int, t0: Int, t1: Int) -> Float64:
    var total_us = Float64(t1 - t0) / 1000.0
    var per = total_us / Float64(n)
    print("probe", arm, "rep", rep, "n", n, "total_us", total_us, "per_op_us", per)
    return per


def _summary(arm: String, n: Int, mut samples: List[Float64]):
    # Insertion sort; the list has a handful of entries.
    for i in range(1, len(samples)):
        var j = i
        while j > 0 and samples[j - 1] > samples[j]:
            var tmp = samples[j - 1]
            samples[j - 1] = samples[j]
            samples[j] = tmp
            j -= 1
    var mid = len(samples) // 2
    var median = samples[mid]
    if len(samples) % 2 == 0:
        median = (samples[mid - 1] + samples[mid]) / 2.0
    print("probe", arm, "summary n", n, "min_us", samples[0], "median_us", median)


def main() raises:
    var n = _env_int("PROBE_N", 1000)
    var repeats = _env_int("PROBE_REPEATS", 3)
    var ctx = DeviceContext()
    print("probe device", ctx.name(), "api", ctx.api(), "n", n, "repeats", repeats)
    var one = ctx.enqueue_create_buffer[DType.float32](1)
    var big = ctx.enqueue_create_buffer[DType.float32](4096)
    var host = ctx.enqueue_create_host_buffer[DType.float32](1)
    ctx.synchronize()
    # Warm-up: compile both grid shapes and touch every path once.
    _launch(ctx, one, 1, Float32(1))
    _launch(ctx, big, 4096, Float32(1))
    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=one)
    ctx.synchronize()

    var s_launch_only = List[Float64]()
    var s_launch_sync = List[Float64]()
    var s_launch_d2h = List[Float64]()
    var s_launch_4096 = List[Float64]()
    var s_sync_only = List[Float64]()
    var s_d2h_only = List[Float64]()
    var checksum = Float32(0)
    for rep in range(repeats):
        # launch_only
        var t0 = Int(perf_counter_ns())
        for i in range(n):
            _launch(ctx, one, 1, Float32(i))
        ctx.synchronize()
        var t1 = Int(perf_counter_ns())
        s_launch_only.append(_report("launch_only", rep, n, t0, t1))

        # launch_sync
        t0 = Int(perf_counter_ns())
        for i in range(n):
            _launch(ctx, one, 1, Float32(i))
            ctx.synchronize()
        t1 = Int(perf_counter_ns())
        s_launch_sync.append(_report("launch_sync", rep, n, t0, t1))

        # launch_d2h_sync
        t0 = Int(perf_counter_ns())
        for i in range(n):
            _launch(ctx, one, 1, Float32(i))
            ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=one)
            ctx.synchronize()
            checksum += host.unsafe_ptr().unsafe_load(0)
        t1 = Int(perf_counter_ns())
        s_launch_d2h.append(_report("launch_d2h_sync", rep, n, t0, t1))

        # launch_4096
        t0 = Int(perf_counter_ns())
        for i in range(n):
            _launch(ctx, big, 4096, Float32(i))
        ctx.synchronize()
        t1 = Int(perf_counter_ns())
        s_launch_4096.append(_report("launch_4096", rep, n, t0, t1))

        # sync_only
        t0 = Int(perf_counter_ns())
        for _ in range(n):
            ctx.synchronize()
        t1 = Int(perf_counter_ns())
        s_sync_only.append(_report("sync_only", rep, n, t0, t1))

        # d2h_only
        t0 = Int(perf_counter_ns())
        for _ in range(n):
            ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=one)
            ctx.synchronize()
            checksum += host.unsafe_ptr().unsafe_load(0)
        t1 = Int(perf_counter_ns())
        s_d2h_only.append(_report("d2h_only", rep, n, t0, t1))

    # batch_k: K launches per synchronize. This is the lever the lane
    # exists to size. If a submit-and-wait round trip is a fixed price,
    # per-launch cost falls as 1/K and every removed synchronize is worth
    # one whole round trip.
    for rep in range(repeats):
        for kk in range(6):
            var k = 1 << kk
            var iters = n // k
            var b0 = Int(perf_counter_ns())
            for _ in range(iters):
                for j in range(k):
                    _launch(ctx, one, 1, Float32(j))
                ctx.synchronize()
            var b1 = Int(perf_counter_ns())
            var total_us = Float64(b1 - b0) / 1000.0
            print(
                "probe batch_k k", k, "rep", rep, "iters", iters,
                "launches", iters * k, "total_us", total_us,
                "per_launch_us", total_us / Float64(iters * k),
                "per_sync_us", total_us / Float64(iters),
            )

    _summary("launch_only", n, s_launch_only)
    _summary("launch_sync", n, s_launch_sync)
    _summary("launch_d2h_sync", n, s_launch_d2h)
    _summary("launch_4096", n, s_launch_4096)
    _summary("sync_only", n, s_sync_only)
    _summary("d2h_only", n, s_d2h_only)
    print("probe checksum", checksum)
    ctx.synchronize()
    _ = one^
    _ = big^
    _ = host^
