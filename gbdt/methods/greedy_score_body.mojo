# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The greedy subsets searcher's leaf accumulation (`calcer.AddLeaf(sum,
weight)` of `compute_scores.cu`), written once (AGENTS.md "One body per
operation").

GPU-free: no `max.gpu`, no `DeviceContext`, no kernel. The device's split
kernels (`gbdt/methods/greedy_subsets_searcher/kernel/compute_scores.mojo`:
`compute_optimal_splits_kernel` and the leafwise scan) and the CPU column's
greedy scorers (`gbdt/host/gbdt_oracle.mojo`: `_add_leaf_cosine` and the L2
candidate scan `_l2_gains_range`; `gbdt/host/gbdt_oracle_lossguide.mojo`
`add_leaf_l2`) call `greedy_add_leaf`, so both columns accumulate a
candidate's score with the same statements.

Lane gbdt-cpu-rest-4-sym (2026-11-23): before this module the body was
`compute_scores.mojo::_add_leaf`, and the host held two restatements of it
(the cosine arm in `gbdt_oracle.mojo::_add_leaf_cosine`, the L2 arm in
`gbdt_oracle_lossguide.mojo::add_leaf_l2`). Their arithmetic was the body's
statement for statement at `normalize=False, pin_mul_add=True`, so no bit is
expected to move on either column.
"""
from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    identical_mul_add,
)
from gbdt.options.catboost_options import (
    SCORE_FUNCTION_COSINE,
    SCORE_FUNCTION_NEWTON_COSINE,
)


@always_inline
def greedy_add_leaf[
    score_function: Int,
    normalize: Bool,
    pin_mul_add: Bool = (GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL),
](
    sum: Float32,
    weight: Float32,
    lambda_l2: Float32,
    mut score: Float32,
    mut denum_sqr: Float32,
):
    """`calcer.AddLeaf(sum, weight)` -- one score calcer's accumulation.

    Cosine and NewtonCosine accumulate `sum * mu` into `score` and
    `weight * mu * mu` into `denum_sqr`; L2 and NewtonL2 accumulate the
    leaf's `sum^2 / (weight + lambda)` into `score` and leave `denum_sqr`
    untouched. Under `pin_mul_add` (IDENTICAL builds) every product and sum
    is flushed and the two contractions are explicit fmas."""
    comptime cosine = (
        score_function == SCORE_FUNCTION_COSINE
        or score_function == SCORE_FUNCTION_NEWTON_COSINE
    )

    comptime if cosine:
        var lam = lambda_l2
        comptime if normalize:
            lam = ftz(lambda_l2 * weight)
        var mu = Float32(0.0)
        if weight > Float32(0.0):
            mu = sum / (weight + lam)

        comptime if pin_mul_add:
            mu = ftz(mu)
            score = ftz(identical_mul_add(sum, mu, score))
            denum_sqr = ftz(
                identical_mul_add(ftz(weight * mu), mu, denum_sqr)
            )
        else:
            score += sum * mu
            denum_sqr += weight * mu * mu
    else:
        if weight > Float32(1e-20):
            comptime if pin_mul_add:
                var num = ftz(sum * sum)
                score = ftz(score + ftz(num / (weight + lambda_l2)))
            else:
                score += (sum * sum) / (weight + lambda_l2)
