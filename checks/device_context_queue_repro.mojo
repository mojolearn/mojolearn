# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Minimal reproduction: does one DeviceContext leave a command queue behind?

Written 2026-09-15 for lane/metal-queue-leak, after the M4 accumulated
thousands of AGXCommandQueue objects, most of them naming no live process.
See docs/lanes/LANE_STATUS_lane-metal-queue-leak.md.

This is the SMALLEST program that answers the question, with no mojolearn
estimator in it: a loop that creates a DeviceContext, does one trivial buffer
fill on it, synchronizes, and drops it. Nothing here is vendor specific; it
runs the same on Metal, CUDA and HIP, and the counters are the host's.

    # one context per iteration (the shape every binding call has today)
    mojo build -I . checks/device_context_queue_repro.mojo -o /tmp/qrepro
    ITERS=200 /tmp/qrepro

    # the proposed workaround: ONE context reused for every iteration
    mojo build -I . -D ONE_CTX=1 checks/device_context_queue_repro.mojo -o /tmp/qrepro_one

Sample the queue count from another shell while it runs, and once after it
exits, with `tools/diag/metal_queue_leak.py --watch <pid>` (macOS), or
`ioclasscount AGXCommandQueue`. Three outcomes:

  flat during the loop                the runtime reuses one queue: not a leak
  grows, drops when the process ends  a per-process leak; reuse bounds it
  grows, survives the process         a kernel-side leak; reuse only slows it,
                                      and the report goes to Modular

`ITERS` is read at compile time (`-D ITERS=...` or the environment at build).
"""

from std.sys import env_get_int, has_accelerator, is_defined
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext

comptime ITERS = env_get_int["ITERS", 200]()
comptime ONE_CTX = is_defined["ONE_CTX"]()
comptime REPORT_EVERY = env_get_int["REPORT_EVERY", 20]()


def one_step(ctx: DeviceContext) raises:
    """One trivial piece of device work, the least a real call can do."""
    var buf = ctx.enqueue_create_buffer[DType.float32](1024)
    buf.enqueue_fill(1.0)
    ctx.synchronize()


def report(i: Int, t0: Int):
    if (i + 1) % REPORT_EVERY == 0 or i == 0:
        var ms = (perf_counter_ns() - t0) // 1_000_000
        print(
            "qrepro step=" + String(i + 1) + " elapsed_ms=" + String(ms),
            flush=True,
        )


def main() raises:
    comptime assert has_accelerator(), "device_context_queue_repro requires a GPU"
    var t0 = perf_counter_ns()
    print(
        "qrepro start iters=" + String(ITERS) + " one_ctx=" + String(ONE_CTX),
        flush=True,
    )
    comptime if ONE_CTX:
        var ctx = DeviceContext()
        for i in range(ITERS):
            one_step(ctx)
            report(i, t0)
    else:
        for i in range(ITERS):
            var ctx = DeviceContext()
            one_step(ctx)
            report(i, t0)
    var total_ms = (perf_counter_ns() - t0) // 1_000_000
    print("qrepro done total_ms=" + String(total_ms), flush=True)
