# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Exact and timing coverage for the production unweighted quantile path."""

from std.time import perf_counter_ns
from gbdt.metrics.sample_quantile import calculate_weighted_target_quantile


def main() raises:
    comptime N = 20000
    var y = List[Float32](capacity=N)
    var ones = List[Float32](capacity=N)
    for i in range(N):
        # Repeats exercise stable tie handling; the permutation avoids sorted input.
        y.append(Float32((i * 7919) % 1009 - 504) * Float32(0.125))
        ones.append(Float32(1.0))

    for alpha in [
        Float64(0.01),
        Float64(0.25),
        Float64(0.5),
        Float64(0.7),
        Float64(0.99),
    ]:
        var plain = calculate_weighted_target_quantile(
            y, List[Float32](), False, alpha, 1.0e-6
        )
        var explicit = calculate_weighted_target_quantile(
            y, ones, True, alpha, 1.0e-6
        )
        if plain != explicit:
            raise Error(
                "unweighted quantile differs from explicit unit weights"
            )

    # A repeatable production-sized measurement. Correctness above is the gate;
    # the timing line makes before/after comparison available without a GPU.
    var checksum = Float32(0.0)
    var t0 = perf_counter_ns()
    for _ in range(5):
        checksum += calculate_weighted_target_quantile(
            y, List[Float32](), False, 0.5, 1.0e-6
        )
    var elapsed_ms = Float64(perf_counter_ns() - t0) / 1.0e6
    var reference_checksum = Float32(0.0)
    t0 = perf_counter_ns()
    for _ in range(5):
        reference_checksum += calculate_weighted_target_quantile(
            y, ones, True, 0.5, 1.0e-6
        )
    var reference_ms = Float64(perf_counter_ns() - t0) / 1.0e6
    if checksum != reference_checksum:
        raise Error("timed quantile checksums differ")
    print(
        "sample quantile unweighted ms=",
        elapsed_ms,
        " explicit-unit-weight ms=",
        reference_ms,
        " checksum=",
        checksum,
    )
    print("sample quantile unweighted check: PASS")
