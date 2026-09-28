# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Transfer-path probe for the x_metrics device runner (lane metrics-apple).

Times, for arenas of 3M and 30M float32 words, the ways a call can move its
arena: device buffer <-> pageable host pointer (what the runner does), via a
pinned host buffer + memcpy, and a kernel working on the host buffer's
memory directly. Prints `XMCOPY <path> <words> <us>` (the best of 5).

    pixi run -e default mojo run -I . bench/x_metrics_copy_probe.mojo
"""
from std.gpu import block_idx, block_dim, thread_idx
from std.memory import memcpy
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

comptime FP = MutPointer[Float32, MutAnyOrigin]


def touch(f: FP, n: Int32):
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < Int(n):
        f.unsafe_store(t, f.unsafe_load(t) + Float32(1))


def best(ts: List[Int]) -> Int:
    var b = ts[0]
    for t in ts:
        b = min(b, t)
    return b // 1000


def main() raises:
    var ctx = DeviceContext()
    for n in [3_000_000, 30_000_000]:
        var host = List[Float32](length=n, fill=Float32(1))
        var hp = host.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var up = List[Int]()
        var down = List[Int]()
        var alloc = List[Int]()
        var pin_up = List[Int]()
        var pin_down = List[Int]()
        var direct = List[Int]()
        var kern = List[Int]()
        for _ in range(5):
            var t0 = perf_counter_ns()
            var d = ctx.enqueue_create_buffer[DType.float32](n)
            ctx.synchronize()
            var t1 = perf_counter_ns()
            ctx.enqueue_copy(dst_buf=d, src_ptr=hp)
            ctx.synchronize()
            var t2 = perf_counter_ns()
            ctx.enqueue_function[touch](d.unsafe_ptr(), Int32(n), grid_dim=(n + 255) // 256, block_dim=256)
            ctx.synchronize()
            var t3 = perf_counter_ns()
            ctx.enqueue_copy(dst_ptr=hp, src_buf=d)
            ctx.synchronize()
            var t4 = perf_counter_ns()
            alloc.append(t1 - t0)
            up.append(t2 - t1)
            kern.append(t3 - t2)
            down.append(t4 - t3)
            var h = ctx.enqueue_create_host_buffer[DType.float32](n)
            ctx.synchronize()
            var t5 = perf_counter_ns()
            memcpy(dest=h.unsafe_ptr(), src=hp, count=n)
            ctx.enqueue_copy(dst_buf=d, src_buf=h)
            ctx.synchronize()
            var t6 = perf_counter_ns()
            ctx.enqueue_copy(dst_buf=h, src_buf=d)
            ctx.synchronize()
            memcpy(dest=hp, src=h.unsafe_ptr(), count=n)
            var t7 = perf_counter_ns()
            pin_up.append(t6 - t5)
            pin_down.append(t7 - t6)
            # a kernel on the host buffer's memory (shared storage on Apple)
            var t8 = perf_counter_ns()
            memcpy(dest=h.unsafe_ptr(), src=hp, count=n)
            ctx.enqueue_function[touch](h.unsafe_ptr(), Int32(n), grid_dim=(n + 255) // 256, block_dim=256)
            ctx.synchronize()
            memcpy(dest=hp, src=h.unsafe_ptr(), count=n)
            var t9 = perf_counter_ns()
            direct.append(t9 - t8)
            _ = d^
            _ = h^
        print("XMCOPY alloc", n, best(alloc))
        print("XMCOPY upload_pageable", n, best(up))
        print("XMCOPY kernel_device", n, best(kern))
        print("XMCOPY download_pageable", n, best(down))
        print("XMCOPY upload_pinned+memcpy", n, best(pin_up))
        print("XMCOPY download_pinned+memcpy", n, best(pin_down))
        print("XMCOPY memcpy+kernel_on_host_buffer+memcpy", n, best(direct))
        print("XMCOPY check", host[0], host[n - 1])
        _ = len(host)
