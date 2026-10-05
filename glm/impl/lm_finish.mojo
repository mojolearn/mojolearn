# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The d-sized finish of LinearRegression's and Ridge's centering (lane
py-runtime, 2026-10-05): the column means from the exact column sums, and the
intercept. Both ran in Python (`linear_model._column_means_f64`,
`_column_means`, `_vector_mean`, `_set_intercept`); Python is glue only, so
they are host Mojo here, shared word for word by the GPU binding
(bindings/_mojolearn_estimators.mojo) and the CPU column
(bindings/_mojolearn_estimators_host.mojo).

They stay on the host on every vendor: they are `cols` binary64 operations,
and Apple has no float64 on the device (glm/estimator.mojo's
ols_fit_resident_host makes the same choice for the same means).

SAME BITS AS THE PYTHON THEY REPLACE: the same IEEE binary64 operations in
the same order. mean_j = sum_j / rows; weighted, (mean_j * rows) / total
(cuML's weighted mean, the order `_column_means` used); the float32 mean is
ONE round-to-nearest-even of the binary64 mean (`_round_f32`). The intercept
is y_mean - fsum(xmean_j * coef_j): every product of two float32 values is
exact in binary64, and the sum is CPython's math.fsum (Shewchuk's exact
partials, rounded once; the same steps as x_metrics/epilogue.mojo's `fsum`,
written with bounded loops), including its NaN and infinity rules."""
from std.math import inf, isinf, isnan, nan


def _exact_sum(vals: List[Float64]) -> Float64:
    """CPython's math.fsum over finite values (Shewchuk's partials, rounded
    once; a zero sum is +0.0): x_metrics/epilogue.mojo's `fsum` with its
    `while` loops bounded by the partials count."""
    var p = List[Float64]()
    for k in range(len(vals)):  # small-loop(vals: one term per feature): the d products of one intercept
        var x = vals[k]
        var i = 0
        for j in range(len(p)):  # small-loop(p: exact partials): at most one partial per binary64 exponent band
            var y = p[j]
            if abs(x) < abs(y):
                var t = x
                x = y
                y = t
            var hi = x + y
            var yr = hi - x
            var lo = y - yr
            if lo != 0.0:
                p[i] = lo
                i += 1
            x = hi
        for _ in range(len(p) - i):  # small-loop(p: exact partials): drops the partials past the kept ones
            _ = p.pop()
        if x != 0.0:
            p.append(x)
    var n = len(p)
    var hi: Float64 = 0.0
    var lo: Float64 = 0.0
    if n > 0:
        n -= 1
        hi = p[n]
        for _ in range(len(p)):  # small-loop(p: exact partials): the final carry walk over the partials
            if n <= 0:
                break
            var x = hi
            n -= 1
            var y = p[n]
            hi = x + y
            var yr = hi - x
            lo = y - yr
            if lo != 0.0:
                break
        if n > 0 and ((lo < 0.0 and p[n - 1] < 0.0) or (lo > 0.0 and p[n - 1] > 0.0)):
            var y = lo * 2.0
            var x = hi + y
            var yr = x - hi
            if y == yr:
                hi = x
    if hi == 0.0:
        return 0.0
    return hi


def lm_means_finish(
    sums: MutPointer[Float64, MutUntrackedOrigin],
    mean64: MutPointer[Float64, MutUntrackedOrigin],
    mean32: MutPointer[Float32, MutUntrackedOrigin],
    cols: Int,
    rows: Int,
    weighted: Bool,
    total: Float64,
):
    """mean64[j] = sums[j] / rows (then * rows / total when weighted) and
    mean32[j] = that mean rounded once to float32."""
    var r = Float64(rows)
    for j in range(cols):  # small-loop(cols: feature count): one binary64 finish per column, Apple has no device float64
        var m = sums[j] / r
        if weighted:
            m = m * r / total
        mean64[j] = m
        mean32[j] = Float32(m)


def lm_intercept(
    xmean: MutPointer[Float32, MutUntrackedOrigin],
    coef: MutPointer[Float32, MutUntrackedOrigin],
    cols: Int,
    y_mean: Float64,
) raises -> Float64:
    """y_mean - math.fsum(float(xmean[j]) * float(coef[j]) for j)."""
    var terms = List[Float64](capacity=cols)
    var any_nan = False
    var pinf = False
    var ninf = False
    for j in range(cols):  # small-loop(cols: feature count): one exact binary64 product per feature
        var t = Float64(xmean[j]) * Float64(coef[j])
        if isnan(t):
            any_nan = True
        elif isinf(t) and t > 0.0:
            pinf = True
        elif isinf(t):
            ninf = True
        else:
            terms.append(t)
    # CPython's math.fsum special values: a NaN term gives NaN, +inf and
    # -inf together raise, one kind of infinity is the result
    var dot: Float64
    if any_nan:
        dot = nan[DType.float64]()
    elif pinf and ninf:
        raise Error("-inf + inf in fsum")
    elif pinf:
        dot = inf[DType.float64]()
    elif ninf:
        dot = -inf[DType.float64]()
    else:
        dot = _exact_sum(terms)
        if isinf(dot):
            raise Error("intermediate overflow in fsum")
    return y_mean - dot
