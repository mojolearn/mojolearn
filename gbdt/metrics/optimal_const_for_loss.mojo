# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The optimum constant starting approx, per loss.

MIRRORS `catboost/libs/metrics/optimal_const_for_loss.h`. This is what
`boost_from_average` seeds the cursors with and what the model records as
its bias (`doc_parallel_boosting.h:174-182`, `:434`).

WHAT IS PORTED AND WHAT IS NOT, by their switch
(`CalcOneDimensionalOptimumConstApprox`):

    RMSE                 PORTED  (CalculateWeightedTargetAverage)
    Logloss/CrossEntropy PORTED  (Logit of the weighted average)
    Quantile/MAE         NOT YET (CalculateWeightedTargetQuantile needs
                                  CalcSampleQuantile, a weighted-quantile
                                  walk with their delta adjust -- refused
                                  by name below, never approximated)
    MAPE / RMSPE / LogCosh / multi-dim
                         NOT YET (same rule)

THE FLOAT32 TRUNCATION IS THEIRS AND IT IS LOAD-BEARING for bit parity:
`CalculateWeightedTargetAverage` accumulates in double and RETURNS FLOAT
(`inline float`, the narrowing at the return). For RMSE that float is the
answer, widened back to double by the `TMaybe<double>` return; for Logloss
`const double bestProbability = <that float>` widens BEFORE the Logit. A
port that kept the average in double end to end would be one ulp off their
bias on real data.

`Logit` is their `math_utils.h` log(x / (1 - x)), taken through libm's
`log` by `external_call` and NOT through `std.math.log`, whose ~5e-8 error
re-decides last bits (the `checks/pointwise_target_check.mojo` finding).
"""

from std.ffi import external_call

from gbdt.targets.kernel.pointwise_targets import (
    OBJECTIVE_CROSSENTROPY,
    OBJECTIVE_LOGLOSS,
    OBJECTIVE_RMSE,
)


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
) raises -> Float64:
    """`NCB::CalcOneDimensionalOptimumConstApprox`'s ported arms.

    Their unported arms raise BY NAME rather than returning zero: a zero
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
            # their Logit would return +-inf; CB_ENSUREs upstream keep a
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
        # ULP off CatBoost's bias on both Logloss fixtures. Same libm
        # `log`, their operand order.
        return -external_call["log", Float64](
            1.0 / best_probability - 1.0
        )
    raise Error(
        "boost_from_average is not ported for this loss yet: only RMSE,"
        " Logloss and CrossEntropy have CalcOptimumConstApprox arms here."
        " Their Quantile/MAE arm needs CalcSampleQuantile (a weighted"
        " quantile with a delta adjust) and is refused by name rather than"
        " approximated."
    )
