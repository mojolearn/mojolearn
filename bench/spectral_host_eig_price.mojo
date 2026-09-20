# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Focused price for the host eigensolver used by spectral Lanczos restarts."""

from std.memory import bitcast
from std.os import getenv
from std.time import perf_counter_ns

from spectral.checks.symmetric_eig_host import symmetric_eig_host


def _repeats() raises -> Int:
    var value = String(getenv("MOJOLEARN_SPECTRAL_EIG_REPEATS"))
    return Int(atol(value)) if value != "" else 2000


def main() raises:
    var n = 20
    var repeats = _repeats()
    var hash = UInt64(1469598103934665603)
    var begin = perf_counter_ns()
    for rep in range(repeats):
        var matrix = List[Float32]()
        matrix.resize(n * n, Float32(0.0))
        for i in range(n):
            matrix[i * n + i] = Float32(i + 1) * Float32(0.03125)
            if i + 1 < n:
                var edge = Float32((i % 7) + 1) * Float32(0.00390625)
                matrix[i * n + i + 1] = edge
                matrix[(i + 1) * n + i] = edge
        # Model the dense arrow introduced by a thick restart.
        for i in range(4):
            var edge = Float32(i + 2) * Float32(0.001953125)
            matrix[4 * n + i] = edge
            matrix[i * n + 4] = edge
        var values = List[Float32]()
        var vectors = List[Float32]()
        _ = symmetric_eig_host[DType.float32](matrix, n, values, vectors)
        hash = (hash ^ UInt64(bitcast[DType.uint32](values[rep % n]))) * UInt64(
            1099511628211
        )
        hash = (
            hash ^ UInt64(bitcast[DType.uint32](vectors[(rep * 17) % (n * n)]))
        ) * UInt64(1099511628211)
    var elapsed = Float64(perf_counter_ns() - begin) / 1000000.0
    print(
        "SPECTRAL_HOST_EIG_PRICE",
        "repeats",
        repeats,
        "ms",
        elapsed,
        "hash",
        hash,
    )
