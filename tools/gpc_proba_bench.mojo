# SPDX-License-Identifier: Apache-2.0
"""Focused timing and identity receipt for the host GPC probability epilogue."""

from std.memory import bitcast
from std.time import perf_counter_ns

from gaussian_process.host.gpc_steps import gpc_proba


def main() raises:
    comptime n = 32768
    var mean = List[Float32](capacity=n)
    var variance = List[Float32](capacity=n)
    for i in range(n):
        mean.append(Float32((i % 257) - 128) * Float32(0.0078125))
        variance.append(Float32(0.125) + Float32(i % 31) * Float32(0.03125))

    var warm = gpc_proba(mean, variance)
    var best = Int(1 << 62)
    var hash = UInt64(1469598103934665603)
    for _ in range(7):
        var t0 = perf_counter_ns()
        var out = gpc_proba(mean, variance)
        var elapsed = perf_counter_ns() - t0
        if elapsed < best:
            best = elapsed
        hash = UInt64(1469598103934665603)
        for i in range(n):
            hash = (hash ^ bitcast[DType.uint64](out[i])) * UInt64(1099511628211)
    print("GPC_PROBA_BENCH", n, "best_ns", best, "hash", hash, "warm", warm[0])
