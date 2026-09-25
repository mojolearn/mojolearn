# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/amd-step-time-2 (2026-09-25): the launch-plus-synchronize stall
probe. The DigitalOcean MI325X showed, about every 250 ms of wall time, one
call (any kernel, or a host scan) about 100 ms slower than its siblings
(lane/amd-step-time-mi325x, README section MI325X). This times a loop of ONE
small kernel launch plus `synchronize()` for a fixed wall time and prints the
latency distribution and the spacing of the slow iterations, so the same
loop can be compared under the performance levels
(tools/amd_mi325x_perflevel_probe.sh). No numerics are judged here.

Environment: STALL_LABEL (printed), STALL_SECONDS (default 30),
STALL_SPIN (loop trips per thread in the kernel, default 0 = an almost empty
kernel; 20000 is about 1 ms), STALL_BLOCKS (default 1216).
Lines:
  STALL label=.. spin=.. iters=.. wall_s=.. median_us=.. p99_us=.. max_ms=..
        over5ms=.. over20ms=.. over50ms=.. over100ms=.. slow_ms_total=..
        slow_gap_median_ms=.. slow_gap_min_ms=..
  STALL_SLOW label=.. t_ms=.. lat_ms=..      (the first 40 slow iterations)
"slow" = more than 20 ms above the median.
"""
from std.gpu import block_idx, thread_idx
from std.os import getenv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from transformer.impl.llama.modeling_llama import _zeros


def spin_kernel(p: MutPointer[Float32, MutAnyOrigin], spin: Int32):
    var i = Int(block_idx.x) * 64 + Int(thread_idx.x)
    var x = p.unsafe_load(i)
    for _ in range(Int(spin)):
        x = x * Float32(0.9999999) + Float32(1.0e-7)
    p.unsafe_store(i, x)


def _env_int(name: String, dflt: Int) raises -> Int:
    var v = String(getenv(name))
    return dflt if v == "" else Int(v)


def _sort(mut v: List[Int]):
    # shell sort (the lists are up to a few hundred thousand entries)
    var gap = len(v) // 2
    while gap > 0:
        for i in range(gap, len(v)):
            var t = v[i]
            var j = i
            while j >= gap and v[j - gap] > t:
                v[j] = v[j - gap]
                j -= gap
            v[j] = t
        gap //= 2


def main() raises:
    var label = String(getenv("STALL_LABEL"))
    var seconds = _env_int("STALL_SECONDS", 30)
    var spin = _env_int("STALL_SPIN", 0)
    var blocks = _env_int("STALL_BLOCKS", 1216)
    var ctx = DeviceContext()
    var buf = _zeros(ctx, blocks * 64)
    # warm up (compile and first launches)
    for _ in range(20):
        ctx.enqueue_function[spin_kernel](buf.unsafe_ptr(), Int32(spin), grid_dim=(blocks, 1, 1), block_dim=(64, 1, 1))
        ctx.synchronize()
    var lat = List[Int]()
    var at = List[Int]()
    var t_start = Int(perf_counter_ns())
    var t_end = t_start + seconds * 1000000000
    while True:
        var t0 = Int(perf_counter_ns())
        if t0 >= t_end:
            break
        ctx.enqueue_function[spin_kernel](buf.unsafe_ptr(), Int32(spin), grid_dim=(blocks, 1, 1), block_dim=(64, 1, 1))
        ctx.synchronize()
        var t1 = Int(perf_counter_ns())
        lat.append(t1 - t0)
        at.append(t0 - t_start)
    var wall = Float64(Int(perf_counter_ns()) - t_start) / 1.0e9
    var n = len(lat)
    var srt = lat.copy()
    _sort(srt)
    var med = srt[n // 2]
    var p99 = srt[(n * 99) // 100]
    var mx = srt[n - 1]
    var o5 = 0
    var o20 = 0
    var o50 = 0
    var o100 = 0
    var slow_total = 0
    var slow_at = List[Int]()
    var printed = 0
    for i in range(n):
        var l = lat[i]
        if l > 5000000:
            o5 += 1
        if l > 20000000:
            o20 += 1
        if l > 50000000:
            o50 += 1
        if l > 100000000:
            o100 += 1
        if l - med > 20000000:
            slow_total += l - med
            slow_at.append(at[i])
            if printed < 40:
                print("STALL_SLOW label=" + label + " t_ms=" + String(Float64(at[i]) / 1.0e6)
                      + " lat_ms=" + String(Float64(l) / 1.0e6))
                printed += 1
    var gaps = List[Int]()
    for i in range(1, len(slow_at)):
        gaps.append(slow_at[i] - slow_at[i - 1])
    var gmed = Float64(0.0)
    var gmin = Float64(0.0)
    if len(gaps) > 0:
        _sort(gaps)
        gmed = Float64(gaps[len(gaps) // 2]) / 1.0e6
        gmin = Float64(gaps[0]) / 1.0e6
    print("STALL label=" + label + " spin=" + String(spin) + " iters=" + String(n) + " wall_s=" + String(wall)
          + " median_us=" + String(Float64(med) / 1.0e3) + " p99_us=" + String(Float64(p99) / 1.0e3)
          + " max_ms=" + String(Float64(mx) / 1.0e6) + " over5ms=" + String(o5) + " over20ms=" + String(o20)
          + " over50ms=" + String(o50) + " over100ms=" + String(o100)
          + " slow_ms_total=" + String(Float64(slow_total) / 1.0e6)
          + " slow_gap_median_ms=" + String(gmed) + " slow_gap_min_ms=" + String(gmin))
    _ = buf^
