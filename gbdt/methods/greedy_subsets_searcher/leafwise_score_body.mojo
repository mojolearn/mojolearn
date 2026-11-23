# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The leafwise score statements, written ONCE (AGENTS.md "One body per
operation"; lane gbdt-cpu-rest-4-ns, 2026-11-23).

GPU-free: no `max.gpu`, no `DeviceContext`, no kernel. The device's score
kernels (`kernel/compute_scores.mojo`: `_leafwise_scan_part`, which
`compute_optimal_split_kernel` (Lossguide) and
`compute_optimal_splits_region_kernel` (Depthwise) run, and the symmetric
`compute_optimal_splits_kernel`) and the CPU non-symmetric driver
(`gbdt/host/gbdt_oracle_depthwise.mojo::_leafwise_gain`), the CPU symmetric
and multiclass searchers (`gbdt/host/gbdt_oracle.mojo::_add_leaf_cosine`)
call these, so a candidate is scored with the same statements on every
column:

- `child_hessian_below`: the child-Hessian rejection, by bits;
- `add_leaf_score`: `calcer.AddLeaf(sum, weight)` for the Cosine /
  NewtonCosine and L2 / NewtonL2 calcers (the old `_add_leaf`);
- `leafwise_finish_gain`: the Cosine normalization, the per-feature noise
  draw, the zero-part rule and the gain times the feature weight.

Before this module the host held its own copies of all three (the inline
bits test and tail in `_leafwise_gain`, `_add_leaf_cosine`, and
`gbdt_oracle_lossguide.add_leaf_l2`). Their statements were the kernel's
one for one (the pinned arm), so no bit is expected to move.
"""
from std.memory import bitcast

from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    identical_mul_add,
)
from gbdt.gpu_util.kernel.random_gen import advance_seed_k, next_normal_f
from gbdt.methods.score_calcer_body import FLOAT32_MAX, cosine_gain
from gbdt.options.catboost_options import (
    SCORE_FUNCTION_COSINE,
    SCORE_FUNCTION_NEWTON_COSINE,
)


@always_inline
def child_hessian_below(mass: Float32, threshold: Float32) -> Bool:
    """Compare nonnegative score-plane masses without subnormal FTZ.

    Sign clearing canonicalizes both signed zeros. The caller must clamp
    masses nonnegative and validate the enabled threshold before this call.
    """
    return (bitcast[DType.uint32](mass) & UInt32(0x7FFFFFFF)) < (bitcast[DType.uint32](threshold) & UInt32(0x7FFFFFFF))


@always_inline
def add_leaf_score[
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
    """`calcer.AddLeaf(sum, weight)` -- one score calcer's accumulation."""
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


@always_inline
def leafwise_finish_gain[
    score_function: Int
](
    score: Float32,
    denum_sqr: Float32,
    score_b: Float32,
    denum_sqr_b: Float32,
    to_zero_part_split: Bool,
    score_std_dev: Float32,
    global_seed: UInt64,
    feature_id: Int,
    feature_weight: Float32,
) -> Float32:
    """The tail of one leafwise candidate (`compute_scores.cu:406-472`):
    the Cosine normalization of the split and the parent scores, the noise
    draw per FEATURE (`advance_seed_k(seed + feature, 4)`, one normal, the
    pinned mul-add on both), the zero-part rule and the gain times the
    feature weight. The L2 calcers have no normalization and no noise."""
    comptime cosine = (
        score_function == SCORE_FUNCTION_COSINE
        or score_function == SCORE_FUNCTION_NEWTON_COSINE
    )
    var final_score = score
    var score_before = score_b
    comptime if cosine:
        final_score = cosine_gain(score, denum_sqr)
        score_before = cosine_gain(score_b, denum_sqr_b)

        if score_std_dev != Float32(0.0):
            var seed = advance_seed_k(global_seed + UInt64(feature_id), 4)
            var draw = next_normal_f(seed)
            var neg_draw = -draw[0]
            final_score = ftz(
                identical_mul_add(neg_draw, score_std_dev, final_score)
            )
            score_before = ftz(
                identical_mul_add(neg_draw, score_std_dev, score_before)
            )

    if to_zero_part_split:
        final_score = -FLOAT32_MAX
        score_before = -FLOAT32_MAX

    var gain = Float32(0.0)
    if not to_zero_part_split:
        gain = ftz(final_score - score_before)
    return ftz(gain * feature_weight)
