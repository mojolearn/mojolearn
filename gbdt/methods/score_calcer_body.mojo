# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CatBoost's five score calcers (`score_calcers.cuh`), written once (AGENTS.md
"One body per operation").

GPU-free: no `max.gpu`, no `DeviceContext`, no kernel. The device's
single-fold split kernel (`gbdt/methods/kernel/pointwise_scores.mojo`,
`find_optimal_split_single_fold_kernel`) and the CPU pointwise searcher's
candidate scorers (`gbdt/host/gbdt_oracle_ordered.mojo`, the plain L2 and
plain cosine arms of `_ordered_tree_structure_levels`) build the same
`ScoreCalcer` with `make_score_calcer` and call `next_feature`, `add_leaf`
(left then right, leaf by leaf) and `get_score`, so the two columns score a
candidate with the same statements.

Lane gbdt-one-body (2026-11-23): before this module the host held two
restatements (the plain L2 leaf sum inline in `_ord_score_blocks`, the
cosine chain and its noise draw in `_plain_cosine_range`). Their arithmetic
was the calcer's statement for statement (the L2 arm at meta exponent 1),
so no bit is expected to move on either column.
"""
from std.math import copysign, fma

from checks.numerics import (
    identical_log,
    identical_mul,
    identical_mul_add,
    identical_pow,
    identical_sqrt,
)
from gbdt.gpu_util.kernel.random_gen import advance_seed_k, next_normal_f
from gbdt.options.catboost_options import (
    SCORE_FUNCTION_COSINE,
    SCORE_FUNCTION_L2,
    SCORE_FUNCTION_LOO_L2,
    SCORE_FUNCTION_NEWTON_COSINE,
    SCORE_FUNCTION_NEWTON_L2,
    SCORE_FUNCTION_SAT_L2,
    SCORE_FUNCTION_SOLAR_L2,
)


#: `FLT_MAX`. Their "this candidate is unusable" sentinel (`pointwise_scores.cu:44-45`,
#: `score_calcers.cuh:161`). NOT negated -- see `pointwise_scores.mojo`'s sign section.
comptime FLOAT32_MAX = Float32(3.4028234663852886e38)


@fieldwise_init
struct ScoreCalcer[score_function: Int](Copyable, ImplicitlyCopyable, Movable):
    """The five calcers of `score_calcers.cuh`, as one comptime-tagged union.

    Their five classes share an interface -- `NextFeature(TCBinFeature)`,
    `AddLeaf(double, double)`, `GetScore()`, `Combine(other)` -- and
    `FindOptimalSplitSingleFoldImpl` is templated on which one it holds
    (`:220-222`). Mojo has no dynamic trait objects, so the template
    argument becomes a comptime `score_function` and every method is a
    `comptime if` over their five bodies, in their order.

    `Combine` is NOT implemented. Its only callers are the pairwise/multiclass
    reducers in `compute_scores.cu`, and this file's three kernels never
    call it -- each thread owns one candidate end to end and combines
    nothing.

    The fields are the union of all five classes' members. A calcer whose
    `score_function` does not use a field ignores it, exactly as their
    class simply does not declare it:

        TSolarScoreCalcer   Lambda (unused, `:35`), Score
        TL2ScoreCalcer      Score, Lambda, MetaExponent
        TLOOL2ScoreCalcer   Score
        TSatL2ScoreCalcer   Score
        TCosineScoreCalcer  Lambda, Normalize, ScoreStdDev, GlobalSeed,
                            FeatureId, Score, DenumSqr

    `TSolarScoreCalcer::Lambda` really is dead in their source: the
    constructor takes a `float` and names it nothing (`score_calcers.cuh:13`),
    and `Lambda = 0` is never read. `FindOptimalSplitPlain` still passes
    `l2` into it (`:487`). Copied, dead field and all.
    """

    var lambda_l2: Float32
    var meta_exponent: Float32
    var normalize: Bool
    var score_std_dev: Float32
    var global_seed: UInt64
    var feature_id: UInt32
    var score: Float32
    var denum_sqr: Float32

    @always_inline
    def next_feature(mut self, bf_feature_id: UInt32):
        """`NextFeature(TCBinFeature)` for all five.

        Four of them are `Score = 0` (`score_calcers.cuh:19, 50, 80, 111`).
        The cosine one also takes the feature id and seeds `DenumSqr`
        (`:146-150`):

            FeatureId = bf.FeatureId;  Score = 0;  DenumSqr = 1e-10f;

        THE 1e-10 SEED IS LOAD-BEARING: `GetScore`'s guard is
        `DenumSqr > 1e-15f`, so a feature that accumulated nothing still
        passes it and returns `-0/sqrt(1e-10)` = 0 rather than `FLT_MAX`.
        The dynamic cosine kernel seeds 1e-20 instead and therefore does
        NOT pass its guard. See finding 1 in the module docstring.
        """
        self.score = Float32(0.0)

        comptime if (
            Self.score_function == SCORE_FUNCTION_COSINE
            or Self.score_function == SCORE_FUNCTION_NEWTON_COSINE
        ):
            self.feature_id = bf_feature_id
            self.denum_sqr = Float32(1e-10)

    @always_inline
    def add_leaf(mut self, sum: Float32, weight: Float32):
        """`AddLeaf(double sum, double weight)`, all five bodies verbatim."""

        comptime if Self.score_function == SCORE_FUNCTION_SOLAR_L2:
            # `score_calcers.cuh:22-24`
            #   Score += (weight > 1e-20f
            #             ? (-sum * sum) * (1 + 2 * log(weight + 1.0)) / weight
            #             : 0);
            if weight > Float32(1e-20):
                # `1 + 2*log(w + 1)` is the fma the default build fused (lane/pinned-mul-contract-free)
                self.score += (
                    (-sum * sum)
                    * fma(Float32(2.0), identical_log(weight + Float32(1.0)), Float32(1.0))
                ) / weight

        comptime if (
            Self.score_function == SCORE_FUNCTION_L2
            or Self.score_function == SCORE_FUNCTION_NEWTON_L2
        ):
            # `score_calcers.cuh:53-56`
            #   double leafScore = (weight > 1e-20f
            #                       ? (-sum * sum) / (weight + Lambda) : 0);
            #   Score += (MetaExponent == 1.0f || leafScore == 0.0f)
            #            ? leafScore
            #            : std::copysign(pow(std::abs(leafScore) / weight,
            #                                MetaExponent), leafScore) * weight;
            var leaf_score = Float32(0.0)
            if weight > Float32(1e-20):
                leaf_score = (-sum * sum) / (weight + self.lambda_l2)
            if (
                self.meta_exponent == Float32(1.0)
                or leaf_score == Float32(0.0)
            ):
                self.score += leaf_score
            else:
                # `weight` cannot be 0 on this arm: `leafScore != 0` implies
                # the `weight > 1e-20f` branch above was taken.
                # pinned: LLVM leaves this product unfused (a plain PTX mul.f32
                # feeding the add.f32, which ptxas may fuse) (lane/pinned-mul-contract-free)
                self.score += identical_mul(
                    copysign(
                        identical_pow(abs(leaf_score) / weight, self.meta_exponent),
                        leaf_score,
                    ),
                    weight,
                )

        comptime if Self.score_function == SCORE_FUNCTION_LOO_L2:
            # `score_calcers.cuh:83-87`
            #   float adjust = weight > 1 ? weight / (weight - 1) : 0;
            #   adjust = adjust * adjust;
            #   Score += (weight > 0 ? adjust * (-sum * sum) / weight : 0);
            #
            # Note the guard mismatch, theirs: `adjust` is 0 for weight in
            # (0, 1], so the second guard's `weight > 0` arm contributes 0
            # there anyway. Copied as written.
            var adjust = Float32(0.0)
            if weight > Float32(1.0):
                adjust = weight / (weight - Float32(1.0))
            adjust = adjust * adjust
            if weight > Float32(0.0):
                self.score += adjust * (-sum * sum) / weight

        comptime if Self.score_function == SCORE_FUNCTION_SAT_L2:
            # `score_calcers.cuh:114-117`
            #   float adjust = weight > 2
            #       ? weight * (weight - 2) / (weight * weight - 3 * weight + 1)
            #       : 0;
            #   Score += (weight > 0 ? adjust * ((-sum * sum) / weight) : 0);
            #
            # THE DENOMINATOR HAS A REAL ROOT AT weight = (3+sqrt(5))/2 =
            # 2.618..., inside the `weight > 2` arm. Theirs divides by it.
            # Copied; a guard would be a fork.
            var adjust_s = Float32(0.0)
            if weight > Float32(2.0):
                # `w*w - 3*w` and the score accumulation are the fmas the
                # default build fused (lane/pinned-mul-contract-free)
                adjust_s = (
                    weight
                    * (weight - Float32(2.0))
                    / (
                        fma(weight, weight, Float32(-3.0) * weight)
                        + Float32(1.0)
                    )
                )
            if weight > Float32(0.0):
                self.score = fma(adjust_s, (-sum * sum) / weight, self.score)

        comptime if (
            Self.score_function == SCORE_FUNCTION_COSINE
            or Self.score_function == SCORE_FUNCTION_NEWTON_COSINE
        ):
            # `score_calcers.cuh:152-157`
            #   double lambda = Normalize ? Lambda * weight : Lambda;
            #   const double mu = weight > 0.0f ? (sum / (weight + lambda)) : 0;
            #   Score += sum * mu;
            #   DenumSqr += weight * mu * mu;
            var lam = self.lambda_l2
            if self.normalize:
                lam = self.lambda_l2 * weight
            var mu = Float32(0.0)
            if weight > Float32(0.0):
                mu = sum / (weight + lam)
            # both accumulates in ONE rounding each, the fusion every default
            # (contract=fast) build formed, written out (lane/explicit-fma-contract-proof, 2026-09-26)
            self.score = identical_mul_add(sum, mu, self.score)
            self.denum_sqr = identical_mul_add(weight * mu, mu, self.denum_sqr)

    @always_inline
    def get_score(self) -> Float32:
        """`GetScore()`.

        Four return `Score` unchanged (`score_calcers.cuh:26-28, 58-60,
        89-91, 119-121`). The cosine one (`:160-168`) normalizes and then
        adds the `random_strength` noise:

            float score = DenumSqr > 1e-15f ? -Score / sqrt(DenumSqr)
                                            : FLT_MAX;
            if (ScoreStdDev) {
                ui64 seed = GlobalSeed + FeatureId;
                AdvanceSeed(&seed, 4);
                score += NextNormal(&seed) * ScoreStdDev;
            }
            return score;

        THE NOISE IS DRAWN FROM THE FEATURE ID, NOT FROM THE THREAD. Every
        bin of one feature, on every device and in every block, gets the
        SAME normal draw, because the seed is `GlobalSeed + FeatureId` and
        nothing else. That is deliberate on their side: `random_strength`
        perturbs the ranking BETWEEN features and must not perturb the
        ranking between bins of one feature.

        `AdvanceSeed(&seed, 4)` is four whole advances BEFORE the draw, and
        `NextNormal` then consumes two more. An implementation that skips the four, or
        that draws normal-then-advance, produces a different tree.
        """
        var out = self.score

        comptime if (
            Self.score_function == SCORE_FUNCTION_COSINE
            or Self.score_function == SCORE_FUNCTION_NEWTON_COSINE
        ):
            if self.denum_sqr > Float32(1e-15):
                out = -self.score / identical_sqrt(self.denum_sqr)
            else:
                out = FLOAT32_MAX
            if self.score_std_dev != Float32(0.0):
                var seed = advance_seed_k(
                    self.global_seed + UInt64(self.feature_id), 4
                )
                var draw = next_normal_f(seed)
                # IDENTITY_PATHS row 9: the contraction PINNED (an explicit
                # fma under IDENTICAL), lane/catboost-parity -- `+=` of a
                # product is contracted on some backends and not on others
                out = identical_mul_add(draw[0], self.score_std_dev, out)

        return out


@always_inline
def add_dim_leaves[
    score_function: Int
](
    mut calcer: ScoreCalcer[score_function],
    weight_left: Float32,
    weight_right: Float32,
    sum_left: Float32,
    part_sum: Float32,
    mut total_left: Float32,
    mut total_part: Float32,
):
    """One approx dimension of one leaf of a multi-output candidate (lane
    gbdt-pw-multiclass, 2026-11-23): `add_leaf` left then right with the
    dimension's own sums and the leaf's one weight pair, and the dimension's
    sums into the totals the pinned class reads.

    The doc-parallel searcher's split score for a multi-output loss is the
    per-dimension score summed over every approx dimension, in the order the
    greedy multi-class scorer adds them (`compute_scores.mojo`
    `compute_optimal_splits_kernel`: per leaf, per dimension left then
    right, then the pinned class): one calcer accumulates every dimension
    of every leaf, so the cosine arm normalizes over all of them together.
    `sum_right = part_sum - sum_left` with no clamp, the single-fold
    kernel's own expression (finding 4). The device's
    `find_optimal_split_dims_kernel` and the CPU column's
    `_plain_dims_range` (gbdt/host/gbdt_oracle_ordered.mojo) call this body,
    so the two columns add the same terms in the same order."""
    var sum_right = part_sum - sum_left
    calcer.add_leaf(sum_left, weight_left)
    calcer.add_leaf(sum_right, weight_right)
    total_left += sum_left
    total_part += part_sum


@always_inline
def add_pinned_class_leaves[
    score_function: Int
](
    mut calcer: ScoreCalcer[score_function],
    weight_left: Float32,
    weight_right: Float32,
    total_left: Float32,
    total_part: Float32,
):
    """MultiClass's pinned last class (`MultiLogitOptimization`,
    `multiclass_targets.cpp:32-35`): its derivative is minus the sum of the
    other classes', so after the leaf's `add_dim_leaves` the leaf adds
    `-total` on each side, the greedy scorer's two extra `AddLeaf` calls
    (`compute_optimal_splits_kernel`'s `multiclass_optimization` arm).
    MultiClassOneVsAll pins no class and never calls this."""
    var total_right = total_part - total_left
    calcer.add_leaf(-total_left, weight_left)
    calcer.add_leaf(-total_right, weight_right)


@always_inline
def make_score_calcer[
    score_function: Int
](
    lambda_l2: Float32,
    meta_exponent: Float32,
    normalize: Bool,
    score_std_dev: Float32,
    global_seed: UInt64,
) -> ScoreCalcer[score_function]:
    """The five constructors of `score_calcers.cuh`, in one place.

    Their `FindOptimalSplitPlain` builds the calcer on the HOST and passes
    it by value (`:485-520`); DEVIATION 95 explains why the configuration
    crosses as scalars instead. This is the device-side other half.
    """
    return ScoreCalcer[score_function](
        lambda_l2,
        meta_exponent,
        normalize,
        score_std_dev,
        global_seed,
        UInt32(0),
        Float32(0.0),
        Float32(0.0),
    )
