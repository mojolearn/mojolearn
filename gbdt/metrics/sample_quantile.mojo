# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The weighted sample quantile behind `boost_from_average` on the MAE,
Quantile and MAPE losses (lane/catboost-parity, 2026-09-19).

Reference: `catboost/libs/helpers/quantile.cpp` (`CalcSampleQuantile`, both
branches) and `libs/metrics/optimal_const_for_loss.h:69-116`
(`CalculateWeightedTargetQuantile` with its delta adjust,
`CalculateOptimalConstApproxForMAPE`), CatBoost `54a8143a`. HOST code, shared
by the device fit (`optimal_const_for_loss.calc_one_dimensional_optimum_const_
approx`) and the CPU host oracles, so the bias is the same bits on both.

TWO DEVIATIONS, stated. (1) The Quantile level reaches this module as the
loss descriptor's float `alpha` widened to double; theirs is the double the
params map parsed. They agree for levels a float holds exactly (0.5, 0.25,
...); for others (0.3) `total * alpha` can differ in its last bits and so
move a quantile that sits exactly on the boundary. (2) their binary-search branch splits each range with
`std::partition`, whose output order is the library's and not specified;
this uses a STABLE partition. The order only reaches the answer through the
double sums of the partition weights, which it can move by an ulp only for
non-integer weights of very different magnitudes; the unweighted fit sums
integers and is exact either way.
"""

from std.math import fma
from std.sys.compile import is_defined

comptime SAMPLE_QUANTILE_SABOTAGE = is_defined[
    "MOJOLEARN_SAMPLE_QUANTILE_SABOTAGE"
]()
"""THE NEGATIVE CONTROL of the quantile constant: `-D
MOJOLEARN_SAMPLE_QUANTILE_SABOTAGE=1` skips their delta adjust, so the
MAE / Quantile / MAPE starting point is the raw sample quantile. It moves
the bias of every `boost_from_average` fit of those losses (the
gbdt-bfa-quantile lane) and nothing else: RMSE, Logloss and CrossEntropy do
not reach this module."""

#: `DBL_EPSILON`
comptime SQ_DBL_EPSILON = 2.220446049250313e-16
#: `BINARY_SEARCH_ITERATIONS` (`quantile.cpp:22`)
comptime SQ_BINARY_SEARCH_ITERATIONS = 100


def _stable_sort_by_value(mut v: List[Float32], mut w: List[Float32]):
    """`StableSort(elements, value <)`: insertion into runs by a merge sort,
    ties keep their input order."""
    var n = len(v)
    var idx = List[Int](capacity=n)
    for i in range(n):
        idx.append(i)
    var tmp = List[Int](length=n, fill=0)
    var width = 1
    while width < n:
        var lo = 0
        while lo < n:
            var mid = min(lo + width, n)
            var hi = min(lo + 2 * width, n)
            var a = lo
            var b = mid
            var k = lo
            while a < mid and b < hi:
                # stable: take the LEFT run on a tie
                if v[idx[b]] < v[idx[a]]:
                    tmp[k] = idx[b]
                    b += 1
                else:
                    tmp[k] = idx[a]
                    a += 1
                k += 1
            while a < mid:
                tmp[k] = idx[a]
                a += 1
                k += 1
            while b < hi:
                tmp[k] = idx[b]
                b += 1
                k += 1
            lo = hi
        for i in range(n):
            idx[i] = tmp[i]
        width *= 2
    var sv = List[Float32](capacity=n)
    var sw = List[Float32](capacity=n)
    for i in range(n):
        sv.append(v[idx[i]])
        sw.append(w[idx[i]])
    v = sv^
    w = sw^


def _stable_sort_values(mut v: List[Float32]):
    """Stable value-only twin for the unweighted quantile path."""
    var n = len(v)
    var idx = List[Int](capacity=n)
    for i in range(n):
        idx.append(i)
    var tmp = List[Int](length=n, fill=0)
    var width = 1
    while width < n:
        var lo = 0
        while lo < n:
            var mid = min(lo + width, n)
            var hi = min(lo + 2 * width, n)
            var a = lo
            var b = mid
            var k = lo
            while a < mid and b < hi:
                if v[idx[b]] < v[idx[a]]:
                    tmp[k] = idx[b]
                    b += 1
                else:
                    tmp[k] = idx[a]
                    a += 1
                k += 1
            while a < mid:
                tmp[k] = idx[a]
                a += 1
                k += 1
            while b < hi:
                tmp[k] = idx[b]
                b += 1
                k += 1
            lo = hi
        for i in range(n):
            idx[i] = tmp[i]
        width *= 2
    var sv = List[Float32](capacity=n)
    for i in range(n):
        sv.append(v[idx[i]])
    v = sv^


def calc_sample_quantile(
    sample: List[Float32],
    weights: List[Float32],
    alpha: Float64,
    has_weights: Bool = True,
) -> Float64:
    """`CalcSampleQuantile` (`quantile.cpp:102-121`): 0 for an empty sample,
    the minimum at `alpha <= 0`, the linear search below 100 elements and
    the 100-step binary search otherwise. `weights` has one entry per
    sample (the caller passes ones for an unweighted pool)."""
    var n = len(sample)
    if n == 0:
        return 0.0
    if alpha <= 0:
        var mn = sample[0]
        for i in range(1, n):
            if sample[i] < mn:
                mn = sample[i]
        return Float64(mn)
    var total = Float64(n)
    if has_weights:
        total = Float64(0.0)
        for i in range(n):
            total += Float64(weights[i])
    # `total * alpha - eps` in ONE rounding: the default (contract=fast)
    # build fused the product into the subtraction (lane/explicit-fma-contract-proof)
    var need_floor = fma(total, alpha, -SQ_DBL_EPSILON)
    if n < 100:
        # `CalcSampleQuantileLinearSearch` (`:79-100`)
        var v = sample.copy()
        var w = List[Float32]()
        if has_weights:
            w = weights.copy()
            _stable_sort_by_value(v, w)
        else:
            _stable_sort_values(v)
        var acc = Float64(0.0)
        for i in range(n):
            acc += Float64(w[i]) if has_weights else 1.0
            if acc >= need_floor:
                return Float64(v[i])
        return Float64(v[n - 1])
    # `CalcSampleQuantileBinarySearch` (`:18-77`)
    var mn = sample[0]
    var mx = sample[0]
    for i in range(1, n):
        if sample[i] < mn:
            mn = sample[i]
        if mx < sample[i]:
            mx = sample[i]
    var l_q = Float64(mn) - SQ_DBL_EPSILON
    var r_q = Float64(mx)
    var ev = sample.copy()
    var ew = weights.copy() if has_weights else List[Float32]()
    var l = 0
    var r = n
    var collected = Float64(0.0)
    var tv = List[Float32](length=n, fill=Float32(0.0))
    var tw = List[Float32](
        length=n, fill=Float32(0.0)
    ) if has_weights else List[Float32]()
    for _ in range(SQ_BINARY_SEARCH_ITERATIONS):
        var q = (l_q + r_q) / 2
        # a STABLE partition of [l, r): `value <= q` first
        var k = l
        for i in range(l, r):
            if Float64(ev[i]) <= q:
                tv[k] = ev[i]
                if has_weights:
                    tw[k] = ew[i]
                k += 1
        var point = k
        for i in range(l, r):
            if not (Float64(ev[i]) <= q):
                tv[k] = ev[i]
                if has_weights:
                    tw[k] = ew[i]
                k += 1
        for i in range(l, r):
            ev[i] = tv[i]
            if has_weights:
                ew[i] = tw[i]
        var left_weight = Float64(point - l)
        if has_weights:
            left_weight = Float64(0.0)
            for i in range(l, point):
                left_weight += Float64(ew[i])
        if collected + left_weight < need_floor:
            l = point
            l_q = q
            collected += left_weight
        else:
            r = point
            r_q = q
    return r_q


def calculate_weighted_target_quantile(
    target: List[Float32],
    weights: List[Float32],
    has_weights: Bool,
    alpha: Float64,
    delta: Float64,
) -> Float32:
    """`CalculateWeightedTargetQuantile` (`optimal_const_for_loss.h:69-103`):
    the sample quantile, then the delta adjust away from the tie, returned
    through `inline float`."""
    var n = len(target)
    if n == 0:
        return Float32(0.0)
    var w = weights.copy() if has_weights else List[Float32]()
    var q = calc_sample_quantile(target, w, alpha, has_weights)
    comptime if SAMPLE_QUANTILE_SABOTAGE:
        return Float32(q)
    if delta > 0:
        var total = Float64(n)
        if has_weights:
            total = Float64(0.0)
            for i in range(n):
                total += Float64(weights[i])
        var less = Float64(0.0)
        var equal = Float64(0.0)
        for i in range(n):
            var wi = Float64(weights[i]) if has_weights else 1.0
            if Float64(target[i]) < q:
                less += wi
            elif Float64(target[i]) == q:
                equal += wi
        # both sides in ONE rounding, as the default build fused them (lane/explicit-fma-contract-proof)
        if fma(equal, alpha, less) >= fma(total, alpha, -SQ_DBL_EPSILON):
            q -= delta
        else:
            q += delta
    return Float32(q)


def calculate_optimal_const_approx_for_mape(
    target: List[Float32], weights: List[Float32], has_weights: Bool
) -> Float32:
    """`CalculateOptimalConstApproxForMAPE` (`optimal_const_for_loss.h:
    105-116`): the weighted median with each weight divided by
    `max(1, |target|)` in float."""
    var n = len(target)
    var w = List[Float32](capacity=n)
    for i in range(n):
        var wi = weights[i] if has_weights else Float32(1.0)
        w.append(wi / max(Float32(1.0), abs(target[i])))
    comptime if SAMPLE_QUANTILE_SABOTAGE:
        return Float32(calc_sample_quantile(target, w, 0.75))
    return Float32(calc_sample_quantile(target, w, 0.5))
