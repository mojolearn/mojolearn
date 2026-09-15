# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Fowlkes-Mallows index. cuML 26.08 ships no Fowlkes-Mallows entry
(`cpp/src/metrics/` has none and `cuml.metrics.cluster` exports none), so
this follows the scikit-learn reference `sklearn/metrics/cluster/_supervised.py`
`fowlkes_mallows_score` on the repository's integer contingency kernel:

    c  = contingency_matrix(labels_true, labels_pred).astype(np.int64)
    tk = np.dot(c.data, c.data) - n_samples
    pk = np.sum(c.sum(axis=0) ** 2) - n_samples     column (cluster) sums
    qk = np.sum(c.sum(axis=1) ** 2) - n_samples     row (class) sums
    return float(np.sqrt(tk / pk) * np.sqrt(tk / qk)) if tk != 0.0 else 0.0

The matrix is the device's (`contingency_matrix_host`, integer atomics,
exact and order-free on every vendor), at the caller's label range, which
the Python side remaps onto `[0, n_classes - 1]` exactly as it does for
mutual information. `tk`, `pk` and `qk` are Int64 sums of squares of Int32
counts (each at most `n^2 < 2^62`), so they are exact. What remains is two
Float64 divisions, two square roots and one multiply on the host, each
correctly rounded by IEEE 754 on every box (no transcendental; Float64
`std.math.sqrt` lowers to the hardware square root on arm64 and x86-64,
and the repository has no Float64 `portable_sqrt`), so the score is
identity-safe with no IDENTICAL arm. The integer-to-Float64
conversion of each operand before dividing is numpy's `int64 / int64`.

Unlike RAFT's ARI there is no `size < 2` or all-unique early return:
scikit-learn has none, and both cases reach `tk == 0` and return 0.0.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.math import sqrt

from metrics.impl.stats.detail.mutual_info_score import (
    col_sums,
    contingency_matrix_host,
    row_sums,
)


def fowlkes_mallows_from_contingency(
    c: List[Int32], k: Int, size: Int
) raises -> Float64:
    """The epilogue from a host copy of the `k * k` row-major matrix."""
    var n = Int64(size)
    var sum_c2 = Int64(0)
    for idx in range(k * k):
        var v = Int64(c[idx])
        sum_c2 += v * v
    var a = row_sums(c, k)
    var b = col_sums(c, k)
    var sum_b2 = Int64(0)
    var sum_a2 = Int64(0)
    for j in range(k):
        sum_b2 += b[j] * b[j]
    for i in range(k):
        sum_a2 += a[i] * a[i]
    var tk = sum_c2 - n
    var pk = sum_b2 - n
    var qk = sum_a2 - n
    if tk == 0:
        return 0.0
    var ft = Float64(tk)
    return sqrt(ft / Float64(pk)) * sqrt(ft / Float64(qk))


def fowlkes_mallows_score(
    ctx: DeviceContext,
    mut y: DeviceBuffer[DType.int32],
    mut y_hat: DeviceBuffer[DType.int32],
    n: Int,
    lower_class_range: Int32,
    upper_class_range: Int32,
) raises -> Float64:
    """`fowlkes_mallows_score(labels_true, labels_pred)` over labels already
    in `[lower_class_range, upper_class_range]`."""
    var k = Int(upper_class_range - lower_class_range + 1)
    var c = contingency_matrix_host(
        ctx, y, y_hat, n, lower_class_range, upper_class_range
    )
    return fowlkes_mallows_from_contingency(c, k, n)
