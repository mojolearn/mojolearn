# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The k-means fixed-point scale, ONE DEFINITION for the device and the host
column (lane cgfin-c-cluster, 2026-10-02).

`choose_scale` needs an upper bound on `sum over rows of abs(x[r, f])` for
the worst feature `f`. The bound is now formed by a FIXED-ORDER float32 fold
that the GPU runs and the host column restates add for add:

  1. chunk sums: rows split into chunks of `SUM_SCALE_CHUNK`; one thread per
     (chunk b, feature f) folds `ftz(abs(x[r, f]))` serially over its rows;
  2. per feature, one block of `SUM_SCALE_TPB` threads: thread t folds the
     chunk sums b = t, t + TPB, ... serially, then the halving tree
     `red[t] += red[t + step]`, step = TPB/2 .. 1 (`halving_block_sum`).

Every term is a non-negative float32 that is zero or normal (the input is
flushed), so every partial is zero or normal and every addition is the
correctly rounded IEEE one on every vendor: the column sums are the same
words on NVIDIA, AMD, Apple and the host. A non-finite input (bit test, not a
float compare, so no fast-math folding can hide it) or a float32 overflow
turns the column's sum into +inf, and `sum_scale_from_columns` refuses it by
name.

The float32 fold under-reads the exact sum by at most a factor `(1 - u)^h`
(`u = 2^-24`, `h` the fold height), so the bound handed to `choose_scale` is
`W * (1 + 4 h u) + 4 n 2^-126` (the second term covers the flushed
subnormals), formed with explicit fmas so no contraction choice can move it.
Old bits (the float64 host chain, the NVIDIA-only certificate, DEVIATION
3081) are gone: the scale can move one binade near a power-of-two boundary.
"""

from std.math import fma
from std.memory import bitcast

from checks.fixed_point import choose_scale
from checks.numerics import ftz

comptime SUM_SCALE_CHUNK = 2048
"""Rows one thread folds serially in stage 1."""
comptime SUM_SCALE_TPB = 256
"""Threads of the stage-2 block (one block per feature)."""
comptime SUM_SCALE_INF_BITS = UInt32(0x7F800000)


@always_inline
def sum_scale_n_chunks(n_samples: Int) -> Int:
    return (n_samples + SUM_SCALE_CHUNK - 1) // SUM_SCALE_CHUNK


@always_inline
def sum_scale_is_finite(v: Float32) -> Bool:
    """Finite by the exponent bits (no float compare)."""
    return (bitcast[DType.uint32](v) & SUM_SCALE_INF_BITS) != SUM_SCALE_INF_BITS


@always_inline
def sum_scale_term(v: Float32) -> Float32:
    """One stage-1 term: `abs`, subnormals flushed. A non-finite word stays
    non-finite (the caller tests its bits)."""
    return ftz(abs(v))


@always_inline
def sum_scale_inf() -> Float32:
    return bitcast[DType.float32](SUM_SCALE_INF_BITS)


def sum_scale_from_columns(cols: List[Float32], n_samples: Int) raises -> Float64:
    """The scale from the per-feature column sums of the fold above: the
    largest (a max, order-free), widened by the fold's error bound, then
    `choose_scale(bound, n_samples)`. A non-finite column is refused."""
    var worst = Float32(0.0)
    for f in range(len(cols)):
        var v = cols[f]
        if not sum_scale_is_finite(v):
            raise Error(
                "kmeans: feature " + String(f) + " has a non-finite value or a"
                " column magnitude past float32 range; refused by name"
            )
        if v > worst:
            worst = v
    if worst == Float32(0.0):
        return choose_scale(0.0, n_samples)
    var n_chunks = sum_scale_n_chunks(n_samples)
    var height = SUM_SCALE_CHUNK + (n_chunks + SUM_SCALE_TPB - 1) // SUM_SCALE_TPB + 8
    # 1 + 4 h 2^-24, and W * that + 4 n 2^-126, each one explicit fma.
    var grow = fma(Float64(4 * height), 5.9604644775390625e-08, 1.0)
    var flush = Float64(4 * n_samples) * 1.1754943508222875e-38
    var bound = fma(Float64(worst), grow, flush)
    return choose_scale(bound, n_samples)
