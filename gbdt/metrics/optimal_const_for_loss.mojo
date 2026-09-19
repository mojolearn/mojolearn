# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The optimum constant starting approx, per loss.

Reference: `catboost/libs/metrics/optimal_const_for_loss.h`. This is what
`boost_from_average` seeds the cursors with and what the model records as
its bias (`doc_parallel_boosting.h:174-182`, `:434`).

WHAT IS IMPLEMENTED AND WHAT IS NOT, by their switch
(`CalcOneDimensionalOptimumConstApprox`):

    RMSE                 IMPLEMENTED  (CalculateWeightedTargetAverage)
    Logloss/CrossEntropy IMPLEMENTED  (Logit of the weighted average)
    Quantile/MAE         IMPLEMENTED  (CalculateWeightedTargetQuantile over
                                  CalcSampleQuantile with their delta adjust,
                                  `gbdt/metrics/sample_quantile.mojo`,
                                  lane/catboost-parity)
    MAPE                 IMPLEMENTED  (CalculateOptimalConstApproxForMAPE)
    RMSPE / LogCosh / multi-dim
                         NOT YET (refused by name below, never approximated)

THE FLOAT32 TRUNCATION IS THEIRS AND IT IS LOAD-BEARING for bit parity:
`CalculateWeightedTargetAverage` accumulates in double and RETURNS FLOAT
(`inline float`, the narrowing at the return). For RMSE that float is the
answer, widened back to double by the `TMaybe<double>` return; for Logloss
`const double bestProbability = <that float>` widens BEFORE the Logit. A
implementation that kept the average in double end to end would be one ulp off their
bias on real data.

`Logit` is their `math_utils.h` `-log(1 / x - 1)`. HISTORY: the first implementation
took `std.math.log`, whose ~5e-8 error re-decides last bits (the
`checks/pointwise_target_check.mojo` finding); the recorded fix was the
host libm's `log` through `external_call`. NOW (DEVIATION 2262, 2026-09-08):
the library's own `portable_log64` (`checks/numerics.mojo` -- Cephes double
log through fma and basic ops only, measured within 2 ulp of libm by
`check-portable-log64`, the same bits on every host and every device), so
the identical path no longer depends on which C library the host links.
CONSEQUENCE, stated rather than hoped away: the bias's bits now come from
OUR log rather than the host's libm. CatBoost computes theirs with the
host's libm, so the recorded bias can differ from CatBoost's by the ULP
difference between the two logs at that one operand; `check-bfa-oracle`
demands `==` on the bits and may need re-baselining to a documented
1-ULP tolerance on that arm (its RMSE arms carry no log and are
unaffected). Operand order and spelling are theirs, unchanged.
"""

from checks.numerics import portable_log64

from gbdt.targets.kernel.pointwise_targets import (
    OBJECTIVE_CROSSENTROPY,
    OBJECTIVE_LOGLOSS,
    OBJECTIVE_MAE,
    OBJECTIVE_MAPE,
    OBJECTIVE_QUANTILE,
    OBJECTIVE_RMSE,
)
from gbdt.metrics.sample_quantile import (
    calculate_optimal_const_approx_for_mape,
    calculate_weighted_target_quantile,
)

#: their Quantile / MAE `delta` loss parameter's default
#: (`optimal_const_for_loss.h:198`), which this surface does not expose
comptime QUANTILE_CONST_DELTA = 1e-6


def calculate_weighted_target_average(
    target: List[Float32],
    weights: List[Float32],
    has_weights: Bool,
) raises -> Float32:
    """`NCB::CalculateWeightedTargetAverage`, including the float return.

    Their `weights.empty()` is `has_weights == False` here: the fit carries
    a ones buffer for the unweighted case, and their empty-weights branch
    is the same arithmetic with weight 1 -- but the BRANCH is kept, because
    `summaryWeight` is `target.size()` (an exact integer sum) on their
    empty branch and an accumulated float sum on the other, and those can
    differ in the last bit at scale.
    """
    var n = len(target)
    if n == 0:
        raise Error("optimal const approx: empty target")
    var summary_weight: Float64
    var target_sum = Float64(0.0)
    if not has_weights:
        summary_weight = Float64(n)
        for i in range(n):
            target_sum += Float64(target[i])
    else:
        if len(weights) != n:
            raise Error(
                "optimal const approx: " + String(len(weights))
                + " weights for " + String(n) + " targets"
            )
        summary_weight = Float64(0.0)
        for i in range(n):
            summary_weight += Float64(weights[i])
        for i in range(n):
            target_sum += Float64(target[i]) * Float64(weights[i])
    # their `return targetSum / summaryWeight;` through `inline float`
    return Float32(target_sum / summary_weight)


def calc_one_dimensional_optimum_const_approx(
    objective: Int,
    target: List[Float32],
    weights: List[Float32],
    has_weights: Bool,
    # the Quantile level from the loss params map, default 0.5
    # (`optimal_const_for_loss.h:196-197`); MAE's is 0.5
    alpha: Float64 = 0.5,
) raises -> Float64:
    """`NCB::CalcOneDimensionalOptimumConstApprox`'s implemented arms.

    Their unimplemented arms raise BY NAME rather than returning zero: a zero
    from this function is a valid answer (a centered target), so a silent
    fallback would be indistinguishable from arithmetic.
    """
    if objective == OBJECTIVE_RMSE:
        return Float64(
            calculate_weighted_target_average(target, weights, has_weights)
        )
    if objective == OBJECTIVE_LOGLOSS or objective == OBJECTIVE_CROSSENTROPY:
        # `const double bestProbability = CalculateWeightedTargetAverage(...)`
        # -- the float32 average widened, THEN their `Logit`.
        var best_probability = Float64(
            calculate_weighted_target_average(target, weights, has_weights)
        )
        if best_probability <= 0.0 or best_probability >= 1.0:
            # their Logit would return +-inf; CB_ENSUREs in the reference keep a
            # constant-label pool out of training before this is reached,
            # and an infinite cursor seed is a poisoned fit. Named here.
            raise Error(
                "boost_from_average: the weighted mean target is "
                + String(best_probability)
                + ", outside (0, 1); a one-class pool has no finite"
                " log-odds"
            )
        # their `Logit` is `-log(1 / x - 1)` (`math_utils.h:27-29`), NOT
        # `log(x / (1 - x))`: the two round differently and the
        # check-bfa-oracle differential measured the naive spelling ONE
        # ULP off CatBoost's bias on both Logloss fixtures. Their operand
        # order, kept exactly; the log is `portable_log64` since
        # DEVIATION 2262 (was the host libm through `external_call`) --
        # see the module docstring for the ULP consequence.
        return -portable_log64(1.0 / best_probability - 1.0)
    if objective == OBJECTIVE_QUANTILE or objective == OBJECTIVE_MAE:
        # their `inline float` return, widened back by `TMaybe<double>`
        return Float64(
            calculate_weighted_target_quantile(
                target, weights, has_weights,
                0.5 if objective == OBJECTIVE_MAE else alpha,
                QUANTILE_CONST_DELTA,
            )
        )
    if objective == OBJECTIVE_MAPE:
        return Float64(
            calculate_optimal_const_approx_for_mape(target, weights, has_weights)
        )
    raise Error(
        "boost_from_average is not implemented for this loss yet: RMSE,"
        " Logloss, CrossEntropy, Quantile, MAE and MAPE have"
        " CalcOptimumConstApprox arms here; the rest are refused by name"
        " rather than approximated."
    )
