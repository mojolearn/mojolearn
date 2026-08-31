# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext


def main() raises:
    var ctx = DeviceContext()
    var buf = ctx.enqueue_create_buffer[DType.uint32](1024)
    var host = ctx.enqueue_create_host_buffer[DType.uint32](1024)
    ctx.enqueue_copy(dst_buf=buf, src_ptr=host.unsafe_ptr())
    ctx.synchronize()

    # 54 drains is what a depth-6 tree does: 9 per level times 6 levels.
    for trial in range(3):
        var t0 = perf_counter_ns()
        for _ in range(54):
            ctx.synchronize()
        var bare = Float64(perf_counter_ns() - t0) / 1.0e6

        t0 = perf_counter_ns()
        for _ in range(54):
            ctx.enqueue_copy(dst_buf=buf, src_ptr=host.unsafe_ptr())
            ctx.synchronize()
        var with_copy = Float64(perf_counter_ns() - t0) / 1.0e6
        print("  trial", trial, ": 54 bare drains", bare,
              "ms, 54 copy+drain", with_copy, "ms")
