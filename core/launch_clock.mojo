# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The launch log's clock: `log_launch_ctx`, the `RF_LAUNCH_CLOCK` drain.

It lived in core/launch_log.mojo until 2026-09-25 and moved here unchanged.
It reads the same once-latched process global (`_LAUNCH_LOG`, whose state
struct keeps the clock fields so both functions share one latch), so only the
forest files that call it import this file, and an edit to the clock no
longer changes the source closure of every binding that logs a launch."""

from std.time import perf_counter_ns

from max.gpu.host import DeviceContext

from core.launch_log import _LAUNCH_LOG


def log_launch_ctx(ctx: DeviceContext, name: StringSlice) raises:
    """`log_launch`, plus the `RF_LAUNCH_CLOCK` drain when that is set."""
    var st = _LAUNCH_LOG.get_or_create_ptr()
    if not st[].enabled:
        return
    if not st[].clock:
        with open(st[].path, "a") as fh:
            fh.write(String(name) + "\n")
        return
    ctx.synchronize()
    var now = Int(perf_counter_ns())
    with open(st[].path, "a") as fh:
        if st[].last_name != "":
            fh.write(st[].last_name + "\t" + String(now - st[].last_ns) + "\n")
    st[].last_name = String(name)
    st[].last_ns = Int(perf_counter_ns())
