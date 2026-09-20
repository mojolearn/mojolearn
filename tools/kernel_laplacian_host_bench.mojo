# SPDX-License-Identifier: Apache-2.0
"""Focused timing and identity receipt for the host Laplacian kernel matrix."""

from std.memory import bitcast
from std.time import perf_counter_ns

from kernel_methods.host.km_host_oracle import (
    KMH_KERNEL_LAPLACIAN,
    kmh_kernel_matrix,
)


def main() raises:
    comptime m = 512
    comptime n = 512
    comptime k = 32
    var a = List[Float32](capacity=m * k)
    var b = List[Float32](capacity=n * k)
    for i in range(m * k):
        a.append(Float32((i * 17) % 251) * Float32(0.00390625))
    for i in range(n * k):
        b.append(Float32((i * 29 + 7) % 251) * Float32(0.00390625))
    var warm = kmh_kernel_matrix(KMH_KERNEL_LAPLACIAN, 1, 0.25, 0.0, a, b, m, n, k)
    var best = Int(1 << 62)
    var hash = UInt64(1469598103934665603)
    for _ in range(5):
        var t0 = perf_counter_ns()
        var out = kmh_kernel_matrix(KMH_KERNEL_LAPLACIAN, 1, 0.25, 0.0, a, b, m, n, k)
        var elapsed = perf_counter_ns() - t0
        if elapsed < best:
            best = elapsed
        hash = UInt64(1469598103934665603)
        for i in range(m * n):
            hash = (hash ^ UInt64(bitcast[DType.uint32](out[i]))) * UInt64(1099511628211)
    print("KERNEL_LAPLACIAN_BENCH", m, n, k, "best_ns", best, "hash", hash, "warm", warm[0])
