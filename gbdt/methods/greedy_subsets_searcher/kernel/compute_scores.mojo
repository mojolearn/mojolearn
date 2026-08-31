# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Score every candidate split of a level and pick the best.

PORT OF `catboost/cuda/methods/greedy_subsets_searcher/kernel/
compute_scores.cu` at CatBoost `54a8143a`, with the score calcers from
`catboost/cuda/methods/kernel/score_calcers.cuh`. Transliterated. Do not
improve.

**The parallel axis is the BIN-FEATURE, and the leaf loop is SERIAL inside
the thread.** That is the whole shape:

    for (offset = blockIdx.x * BlockSize; offset < binFeatureCount; ...)
        binFeatureId = offset + tid
        for (i = 0; i < pCount; i++)          <- leaves, serially
            leafId = partIds[i]
            ...

An oblivious level scores ONE split across all its leaves, so a candidate's
score is a sum over leaves. Putting the candidate on the parallel axis makes
that sum thread-local: no cross-thread reduction over leaves at all, and the
per-leaf reads are a strided walk one thread owns. Putting the LEAF on the
parallel axis instead would force a reduction across threads for every
candidate.

The reason this shape is available to them and not, historically, to us: the
bin prefix scan already ran in its own kernel (`histogram_utils.scan`), so a
thread reads a leaf's cumulative sum at one bin directly. A kernel that
re-derives the running prefix per candidate must walk bins in order, which
forces the bin loop innermost, which pushes the leaf loop outward. **The scan
is what buys the parallel shape**, not a micro-optimization on top of it.

Their right-side arithmetic, copied: the right child is never accumulated,
only derived as `partStat - sumLeft`. One histogram, both sides.

=========================== SIGN OF THE SCORE ===========================
CatBoost's GPU scores are INVERSE: "in GPU catboost all scores are inverse,
lower is better" (`compute_scores.cu:374`). Every calcer of theirs returns a
negated objective and the argmax keeps the SMALLEST (`gain < bestGain`,
`:136-140`).

This port keeps the opposite sign -- larger is better -- because the host
side was written against it. Every calcer below is therefore theirs with
one negation folded in, and every comparison is flipped to match. The two
sentinels flip with it: their `FLT_MAX` "this candidate is unusable" becomes
`-FLOAT32_MAX` here, and their `bestGain = FLT_MAX` initial value becomes
`-FLOAT32_MAX`.
========================================================================

======================= DEVIATION: SCORE PRECISION =======================
THEIRS IS DOUBLE WHERE OURS IS FLOAT32. Three places, all of them theirs
by citation:

  * `const double* partStats` -- `compute_scores.cu:60`. Their per-leaf
    totals are a double buffer produced by `ComputePartitionStats` and
    all-reduced as double. Ours is `MutPointer[Float32]`.
  * `double partStat = __ldg(partStats + ...); float sumRight =
    static_cast<float>(partStat - sumLeft);` -- `compute_scores.cu:97-99`.
    The right child is derived by SUBTRACTION, which is the catastrophic
    cancellation step of this whole kernel: when a bin holds nearly all of a
    leaf's gradient, `partStat - sumLeft` cancels most of the significand.
    They do that subtraction in DOUBLE and narrow afterwards. We do it in
    Float32 throughout, so the cancelled result carries roughly half their
    significant bits.
  * `double Score; double DenumSqr;` in `TCosineScoreCalcer`, and
    `void AddLeaf(double sum, double weight)` in every calcer --
    `score_calcers.cuh:182-183, :152`. Their accumulators over
    `pCount * (statCount - 1) * 2` leaf terms are double. Ours are Float32,
    so the accumulation error grows with leaf count as well.

WHY: Metal has no fp64. There is no double on Apple GPUs at any cost, so
this cannot be closed by spending time or bandwidth -- it can only be closed
by giving up the Metal target or by a software double (which would be
roughly 10x the arithmetic on the hottest loop in the tree).

NOT FAKED. No Float64 appears here pretending to be theirs. The price is
recorded instead: split CHOICE can differ from CatBoost's on a level where
two candidates are within Float32 epsilon of each other after cancellation,
and the difference grows with `p_count` (leaves at the level) and with how
concentrated the gradient is in one bin.
=========================================================================
"""

from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.gpu.intrinsics import ldg
from std.math import sqrt
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.memory import stack_allocation

from gbdt.gpu_util.kernel.random_gen import advance_seed_k, next_normal_f
from gbdt.targets.kernel.pointwise_targets import pinned_block_sum
from original.kernel_matrix import partition_chunks_sm_for
from original.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    identical_mul_add,
    identical_sqrt,  # DEVIATION 258: row 10, NVIDIA sqrt is approximate
)
from gbdt.options.catboost_options import (
    SCORE_FUNCTION_COSINE,
    SCORE_FUNCTION_L2,
    SCORE_FUNCTION_NEWTON_COSINE,
    SCORE_FUNCTION_NEWTON_L2,
)


#: `compute_scores.cu:167`.
comptime SCORE_BLOCK_SIZE = 128

#: `FLT_MAX`. Their sentinel for "no usable candidate" (`compute_scores.cu:48`
#: and `score_calcers.cuh:161`). Negated at every use here, per the sign note
#: in the module docstring.
comptime FLOAT32_MAX = Float32(3.4028234663852886e38)


@always_inline
def _add_leaf[
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

    Factored out of the split loop because their `AddLeaf` is called from
    THREE places once MultiClass exists: the left side, the right side,
    and the pinned class's reconstructed contribution
    (`compute_scores.cu:101-102`, `:107-108`). Writing it three times is
    how the three drift apart.
    """
    comptime cosine = (
        score_function == SCORE_FUNCTION_COSINE
        or score_function == SCORE_FUNCTION_NEWTON_COSINE
    )

    comptime if cosine:
        # `TCosineScoreCalcer::AddLeaf` -- `score_calcers.cuh:152-157`:
        #     double lambda = Normalize ? Lambda * weight : Lambda;
        #     mu = weight > 0.0f ? sum / (weight + lambda) : 0.0f;
        #     Score += sum * mu;
        #     DenumSqr += weight * mu * mu;
        var lam = lambda_l2
        comptime if normalize:
            # DEVIATION 253 / IDENTITY_PATHS ROW 10: this product can
            # land denormal and becomes a divide operand two lines down;
            # stored through `ftz` (comptime no-op under FAST).
            # Unreachable today -- every launch takes the default
            # `normalize = False` -- pinned so the seam holds the day it
            # is not.
            lam = ftz(lambda_l2 * weight)
        var mu = Float32(0.0)
        if weight > Float32(0.0):
            mu = sum / (weight + lam)

        # ============ IDENTITY_PATHS ROW 9, MEASURED HERE ============
        # `Score += sum * mu` and `DenumSqr += weight * mu * mu` are the
        # ONLY contractible seams on the whole score path -- the L2 calcer
        # accumulates a quotient, which has no multiply to fuse into the
        # add. Row 9 says contraction is a codegen decision no runtime row
        # can reach.
        #
        # ON THIS KERNEL THE SEAM IS REAL AND THE PIN REACHES IT. Verified
        # in the emitted code, not inferred: `mojo build --emit asm` writes
        # a per-kernel AIR sidecar, and diffing a FAST build against an
        # IDENTICAL one turns `fmul contract` + `fadd contract` into
        # `call contract float @llvm.fma.f32`, 10 sites -> 22, exactly the
        # +12 that six pinned `_add_leaf` calls x two lines predicts. The
        # L2 kernel's module is byte-identical between the two builds,
        # which is the control.
        #
        # AND ON THIS SEAM METAL'S OWN FAST CODEGEN ALREADY FUSES.
        # `check-leafwise-scores` buckets every Cosine shape four ways and
        # the discriminating ones match the `fma` walk under FAST as well.
        # **That contradicts the generalization in IDENTITY_PATHS row 9**,
        # which records `check-ieee-arith` measuring Apple UNFUSED (fused 0
        # of 2^20). Both measurements can be right -- they are different
        # expressions in different kernels -- and the lesson is that
        # contraction must be measured PER SEAM, never inherited from a
        # probe. So the pin buys nothing on Apple here and everything on a
        # backend whose codegen chooses the other way.
        #
        # A SENTENCE THAT WAS HERE IS DELETED AS FALSE: "15 matched the
        # naive chain and 3 matched fma ... so MAX contracts on some
        # instantiations and not others". The 15 were TIES, counted as
        # naive by a three-way tally tested in the wrong order;
        # `host_best` reproduces the same 15/3 split against ITSELF on the
        # CPU with no device involved. See commit 97df3d8's message, which
        # carries the same error.
        #
        # `pin_mul_add` routes it through `numerics.identical_mul_add`,
        # which is `fma` under IDENTICAL and the naive chain under FAST.
        #
        # ======================== DEVIATION 253 ======================
        # THE DEFAULT NOW FOLLOWS THE MODE: `pin_mul_add` defaults to
        # true under IDENTICAL and false under FAST. Their source has
        # no such switch -- one CUDA build, contraction left to nvcc's
        # whim -- and the E1 campaign measured the whim disagreeing:
        # NVIDIA H100 first diverged at tree001.winners.scores (RMSE;
        # tree000 for Logloss) while Apple and AMD agreed bit for bit,
        # with every stage upstream of the score computation identical
        # on all three (E1 2026-08-23). This function IS the score
        # computation, and its unpinned mul-add chains were the seams;
        # the mode-derived default is what makes the pinned arm the arm
        # IDENTICAL compiles on the symmetric call sites below, which
        # pass no explicit `pin_mul_add`. Under FAST the default stays
        # false, so those sites compile the else arm character for
        # character as before and no lane's FAST numbers move. The
        # kernel-body ftz seams below share this deviation number.
        # =============================================================
        comptime if pin_mul_add:
            # IDENTITY_PATHS ROW 10 rides along on the pinned arm: `mu` is
            # a division in scoring (the row's own title), and the row's
            # checklist requires a pinned expression's INTERMEDIATES to be
            # stored through `ftz` -- an unflushed denormal intermediate
            # re-diverges the very seam row 9 just pinned. Every `ftz` is
            # a comptime no-op under FAST, so the FAST arm of this branch
            # is bitwise what it was.
            mu = ftz(mu)
            score = ftz(identical_mul_add(sum, mu, score))
            denum_sqr = ftz(
                identical_mul_add(ftz(weight * mu), mu, denum_sqr)
            )
        else:
            score += sum * mu
            denum_sqr += weight * mu * mu
    else:
        # `TL2ScoreCalcer::AddLeaf` at `MetaExponent == 1`:
        # `leafScore = (weight > 1e-20f ? (-sum*sum)/(weight+Lambda) : 0)`
        # -- `score_calcers.cuh:54`. Every calcer of theirs carries this
        # guard. Ours divided unconditionally: the weights are
        # `max(.., 0)` clamped but the SUMS are not, so a side whose
        # weight cancels to zero with a non-zero residual contributed
        # `sum^2 / lambda` here and 0 in theirs, and divides by zero
        # outright if lambda is ever 0.
        if weight > Float32(1e-20):
            comptime if pin_mul_add:
                # DEVIATION 253 / row 10: no contractible seam here (the
                # product feeds a divide, not an add), but `sum * sum`
                # squares a gradient and can land denormal, and so can
                # the quotient of a cancelled numerator; both stored
                # through `ftz` so an FTZ backend and a denormal-honoring
                # one accumulate the same terms. `weight + lambda_l2` is
                # a sum of nonnegatives from flushed producers and
                # cannot cancel into the denormal range. Same products,
                # same association as the else arm; comptime no-ops
                # under FAST.
                var num = ftz(sum * sum)
                score = ftz(score + ftz(num / (weight + lambda_l2)))
            else:
                score += (sum * sum) / (weight + lambda_l2)


def compute_optimal_splits_kernel[
    score_function: Int = SCORE_FUNCTION_COSINE,
    normalize: Bool = False,
](
    bf_skip: MutPointer[UInt8, MutAnyOrigin],
    bin_feature_count_in: Int32,
    bf_feature_id: MutPointer[UInt32, MutAnyOrigin],
    feature_weights: MutPointer[Float32, MutAnyOrigin],
    histograms: MutPointer[Float32, MutAnyOrigin],
    part_stats: MutPointer[Float32, MutAnyOrigin],
    stat_count_in: Int32,
    part_ids: MutPointer[UInt32, MutAnyOrigin],
    p_count_in: Int32,
    multiclass_optimization: Int32,
    lambda_l2: Float32,
    score_std_dev: Float32,
    global_seed: UInt64,
    out_score: MutPointer[Float32, MutAnyOrigin],
    out_bin: MutPointer[UInt32, MutAnyOrigin],
):
    """`ComputeOptimalSplits`, with the block argmax that follows it.

    One block reduces its candidates to a single best and writes one record.

    `bf_skip` and `bf_feature_id` are their `TCBinFeature` array split into
    parallel planes: `SkipInScoreCount` and `FeatureId`
    (`gpu_structures.h:17-21`). `BinId` is not needed here because the host
    resolves it from the bin-feature index.

    `feature_weights` is their `binFeaturesWeights`, indexed by FEATURE id
    and not by bin-feature id (`compute_scores.cu:136`). See the multiply
    below.

    `out_score` carries their `Gain`, not their `Score`: `Gain` is what their
    block argmax compares (`compute_scores.cu:30`) and what their host
    cross-block reduce compares (`TBestSplitProperties::operator<`,
    `gpu_structures.h:81`). Their `Score` field is carried alongside only for
    logging.
    """
    # Their `switch (scoreFunction)` pairs `Cosine` with `NewtonCosine` and
    # `L2` with `NewtonL2` onto ONE calcer each, and `default: throw
    # std::exception()` for the rest (`compute_scores.cu:201-219`). The
    # Newton spellings differ only in which derivative the caller put in the
    # stat planes, which is decided long before this kernel runs.
    comptime cosine = (
        score_function == SCORE_FUNCTION_COSINE
        or score_function == SCORE_FUNCTION_NEWTON_COSINE
    )
    comptime assert (
        cosine
        or score_function == SCORE_FUNCTION_L2
        or score_function == SCORE_FUNCTION_NEWTON_L2
    ), (
        "score_function has no calcer here; only Cosine, NewtonCosine, L2"
        " and NewtonL2 are ported"
    )

    var bin_feature_count = Int(bin_feature_count_in)
    var stat_count = Int(stat_count_in)
    var p_count = Int(p_count_in)
    var tid = Int(thread_idx.x)

    # `float bestGain = FLT_MAX` -- `compute_scores.cu:67-68`, negated for
    # our sign. It was 0.0, which under a maximizing comparison REJECTS a
    # candidate that scores exactly 0: an all-zero level then left
    # `best_bin` at the sentinel and the caller silently took bin 0. An
    # all-zero level is not exotic -- a constant feature, a leaf whose
    # gradient has already been spent, or an early level on a
    # perfectly-fit residual all produce it.
    var best_gain = -FLOAT32_MAX
    var best_bin = UInt32(0xFFFFFFFF)

    var offset = Int(block_idx.x) * SCORE_BLOCK_SIZE
    while offset < bin_feature_count:
        var bin_feature_id = offset + tid
        if bin_feature_id >= bin_feature_count:
            break
        if bf_skip.unsafe_load(bin_feature_id) != 0:
            offset += SCORE_BLOCK_SIZE * Int(grid_dim.x)
            continue

        # `calcer.NextFeature(bf[binFeatureId])` -- `compute_scores.cu:84`.
        # For `TL2ScoreCalcer` that is `Score = 0` (`score_calcers.cuh:50`);
        # for `TCosineScoreCalcer` it is `Score = 0; DenumSqr = 1e-10f`
        # (`score_calcers.cuh:146-149`). The 1e-10 seed is what keeps the
        # `DenumSqr > 1e-15f` guard below from firing on an empty calcer.
        var score = Float32(0.0)
        var denum_sqr = Float32(1e-10)

        # `TScoreCalcer beforeSplitCalcer = calcer` (`:85`). In THIS kernel
        # -- the oblivious one -- the copy is taken immediately after
        # `NextFeature` and is never fed a leaf, so its accumulators stay
        # at their `NextFeature` values and `GetScore()` on it is
        # `-0 / sqrt(1e-10)` = 0. Their leafwise kernels DO feed it
        # (`:341`), which is why the variable exists at all.
        #
        # It is NOT folded away here, and that sentence used to say it was
        # "NOT ported": with `ScoreStdDev` live, `beforeSplitCalcer` is no
        # longer zero -- it carries the same normal draw the after-calcer
        # does, and the subtraction at the bottom of the loop is what
        # cancels the noise. See the noise block below.

        # THE SERIAL LEAF LOOP. See the module docstring for why it is inside
        # the thread and not across threads.
        for i in range(p_count):
            # `const int leafId = __ldg(partIds + i)`
            # (`compute_scores.cu:88`). `__ldg` is a read-only non-coherent
            # load; `std.gpu.intrinsics.ldg` is its Mojo spelling. Note
            # `bf[binFeatureId]` above is NOT `__ldg` in their file
            # (`:80`, `:84`, `:136`) and is not one here.
            var leaf_id = Int(ldg(part_ids + i))
            var leaf_base = leaf_id * stat_count * bin_feature_count

            # stat 0 is the weight plane. `__ldg(histograms + ...)` and
            # `__ldg(partStats + leafId * statCount)`
            # (`compute_scores.cu:90-91`).
            var weight_left = max(
                ldg(histograms + (leaf_base + bin_feature_id)),
                Float32(0.0),
            )
            # DEVIATION 253 / IDENTITY_PATHS ROW 10: the subtraction can
            # cancel into the denormal range, and a denormal weight fed
            # to `sum / (weight + lambda)` diverges between Metal's FTZ
            # hardware and a denormal-honoring backend. Flushed at
            # derivation, as the leafwise scan below flushes its twin.
            # `weight_left` needs no flush: a load and a max are exact,
            # and the histogram's own store seam is the producer's
            # flush. Comptime no-op under FAST.
            var weight_right = ftz(
                max(
                    ldg(part_stats + leaf_id * stat_count) - weight_left,
                    Float32(0.0),
                )
            )

            # their `double totalSumLeft` / `double totalSumPart`
            # (`compute_scores.cu:93-94`). The width follows the same
            # recorded deviation as `partStat` above: theirs are double
            # and this port is float32 throughout the score path.
            var total_sum_left = Float32(0.0)
            var total_sum_part = Float32(0.0)

            for stat_id in range(1, stat_count):
                # `float sumLeft = __ldg(histograms + ...)` and
                # `double partStat = __ldg(partStats + ...)`
                # (`compute_scores.cu:96-97`). The DOUBLE is theirs and is
                # the deviation recorded in the module docstring; the `__ldg`
                # is not, and is ported here.
                var stat_slot = (
                    leaf_base
                    + stat_id * bin_feature_count
                    + bin_feature_id
                )
                var sum_left = ldg(histograms + stat_slot)
                var part_stat = ldg(
                    part_stats + (leaf_id * stat_count + stat_id)
                )
                # DEVIATION 253 / row 10: `partStat - sumLeft` is THE
                # cancellation step of this kernel (module docstring),
                # so it is the likeliest producer of a denormal on this
                # path. Flushed at derivation, as the leafwise scan
                # below flushes its twin; comptime no-op under FAST.
                var sum_right = ftz(part_stat - sum_left)

                # `calcer.AddLeaf(sumLeft, weightLeft)` then
                # `calcer.AddLeaf(sumRight, weightRight)`
                # -- `compute_scores.cu:101-102`.
                _add_leaf[score_function, normalize](
                    sum_left, weight_left, lambda_l2, score, denum_sqr
                )
                _add_leaf[score_function, normalize](
                    sum_right, weight_right, lambda_l2, score, denum_sqr
                )
                # `totalSumLeft += sumLeft` (`:103`). DEVIATION 253 /
                # row 10: sums of SIGNED gradients can cancel into the
                # denormal range, and both totals become `AddLeaf`
                # operands on the multiclass arm below. Stored through
                # `ftz`; comptime no-ops under FAST.
                total_sum_left = ftz(total_sum_left + sum_left)
                total_sum_part = ftz(total_sum_part + part_stat)

            # THE MULTICLASS ARM (`compute_scores.cu:105-110`).
            #
            #     if (multiclassOptimization) {
            #         double totalSumRight = totalSumPart - totalSumLeft;
            #         calcer.AddLeaf(-totalSumLeft, weightLeft);
            #         calcer.AddLeaf(-totalSumRight, weightRight);
            #     }
            #
            # ONE MORE LEAF CONTRIBUTION PER SIDE, and it is the PINNED
            # class's. The cursor carries `numClasses - 1` free approxes,
            # so the histogram has `numClasses - 1` gradient planes; the
            # missing one is not stored because the multinomial gradient
            # sums to zero over ALL numClasses, which makes it exactly
            # `-sum of the others`. The same identity the estimator's
            # `gradient[bin*rowSize + cursorDim] = -total` uses
            # (`pointwise_oracle.cpp:100`), applied on the search side.
            #
            # A port that skipped this would score every split as if the
            # last class did not exist, which is not a small error: for a
            # binary-ish 7-class problem the pinned class can carry most
            # of the mass.
            if multiclass_optimization != Int32(0):
                # DEVIATION 253 / row 10: one more cancelling
                # subtraction, flushed at derivation like `sum_right`.
                var total_sum_right = ftz(total_sum_part - total_sum_left)
                _add_leaf[score_function, normalize](
                    -total_sum_left, weight_left, lambda_l2,
                    score, denum_sqr,
                )
                _add_leaf[score_function, normalize](
                    -total_sum_right, weight_right, lambda_l2,
                    score, denum_sqr,
                )

        # `const ui32 featureId = bf[binFeatureId].FeatureId`
        # (`compute_scores.cu:136`), hoisted because the noise seed needs it
        # before the feature-weight multiply does.
        var feature_id = Int(bf_feature_id.unsafe_load(bin_feature_id))

        # `float score = calcer.GetScore()` -- `compute_scores.cu:131`.
        var final_score = score
        # `const float scoreBefore = beforeSplitCalcer.GetScore()`
        # (`:132`). `beforeSplitCalcer` is a copy of the calcer taken
        # immediately after `NextFeature` (`:85`) and THIS kernel never
        # feeds it a leaf, so its `Score` is 0 and its `DenumSqr` is the
        # 1e-10 seed -- which passes the `> 1e-15` guard, making its score
        # `-0 / sqrt(1e-10)`, i.e. zero, in our sign as in theirs. It is a
        # variable rather than a folded-away constant because the noise
        # below is added to it too.
        var score_before = Float32(0.0)
        comptime if cosine:
            # `DenumSqr > 1e-15f ? -Score / sqrt(DenumSqr) : FLT_MAX`
            # -- `score_calcers.cuh:161`. Both branches negated for our
            # sign, so their "unusable" FLT_MAX becomes -FLOAT32_MAX, which
            # loses every comparison here exactly as FLT_MAX loses every
            # comparison there.
            if denum_sqr > Float32(1e-15):
                # DEVIATION 253 / row 10 is TITLED "division and sqrt
                # in scoring": the quotient of a cancelled numerator can
                # land denormal, where Metal's FTZ hardware and a
                # denormal-honoring backend part ways. Flushed at the
                # derivation, as the leafwise twin below flushes;
                # comptime no-op under FAST.
                final_score = ftz(score / identical_sqrt(denum_sqr))
            else:
                final_score = -FLOAT32_MAX

            # ============= THE `random_strength` NOISE ==================
            # `score_calcers.cuh:162-166`, the ONLY calcer of the five that
            # has it:
            #
            #     if (ScoreStdDev) {
            #         ui64 seed = GlobalSeed + FeatureId;
            #         AdvanceSeed(&seed, 4);
            #         score += NextNormal(&seed) * ScoreStdDev;
            #     }
            #
            # Seeded by FEATURE ID, so every bin of one feature draws the
            # SAME normal on every device and in every block: the term
            # perturbs the ranking BETWEEN features and never between bins
            # of one feature. The four advances are part of the stream --
            # three, or a draw before the advance, is a different tree.
            #
            # ===== AND IN THIS KERNEL IT CANCELS. Do not "fix" that. =====
            # `GetScore()` runs on BOTH calcers, both carry the same
            # `GlobalSeed` and the same `FeatureId`, so `score` and
            # `scoreBefore` receive the SAME draw, and `gain = score -
            # scoreBefore` (`:134`) removes it. Algebraically the greedy
            # oblivious arm is noiseless no matter what `random_strength`
            # is; in float32 what survives is the rounding of
            # `fl(fl(s + n) - n)`, which is a perturbation of order
            # `ulp(n)` and not a redraw. The same holds in their two
            # leafwise kernels (`:370-375`, `:459-464`), where the before
            # calcer IS fed leaves but is still seeded identically.
            #
            # It is ported anyway, and the reason is not decoration: the
            # two calcers stop being seeded identically the moment anything
            # feeds them different features -- and a port that had dropped
            # the term would then be silently noiseless where CatBoost is
            # not. The arm where the noise DOES change the model is the
            # doc-parallel one (`kernel/pointwise_scores.cu:396-402`),
            # where `scoreBeforeSplit` is an unnoised HOST scalar carried
            # from the previous level.
            # ===========================================================
            if score_std_dev != Float32(0.0):
                var seed = advance_seed_k(
                    global_seed + UInt64(feature_id), 4
                )
                var draw = next_normal_f(seed)
                # ONE subtraction each, not addition: this port's cosine
                # score is theirs negated (see the module docstring), and
                # `-(theirs + n)` is `ours - n` exactly, float negation
                # being exact. Both calcers, because both of theirs run
                # this branch.
                #
                # IDENTITY_PATHS ROW 9: `score - draw * stdDev` is a
                # multiply feeding a subtract, and Mojo has been seen
                # contracting ACROSS a named intermediate, so a backend is
                # free to fuse one of the two subtractions and not the
                # other. Routed through `identical_mul_add(-draw, stdDev,
                # score)`: under FAST that is the same mul-then-add chain
                # (negation is exact, `x + (-n)` is `x - n` bit for bit),
                # under IDENTICAL it is one `fma` on every backend. Found
                # by the lossguide lane's audit, routed here by the
                # orchestrator 2026-08-22.
                var neg_draw = -draw[0]
                # DEVIATION 253 / row 10 rides along: each fma result is
                # an operand of the gain subtraction below, so it is
                # stored through `ftz` like every derived float on this
                # path. Comptime no-op under FAST.
                final_score = ftz(
                    identical_mul_add(
                        neg_draw, score_std_dev, final_score
                    )
                )
                score_before = ftz(
                    identical_mul_add(
                        neg_draw, score_std_dev, score_before
                    )
                )

        # `gain = score - scoreBefore`, then
        # `gain *= __ldg(binFeaturesWeights + featureId)` where
        # `featureId = bf[binFeatureId].FeatureId`
        # -- `compute_scores.cu:134-137`.
        #
        # Indexed by FEATURE, not by bin-feature: every bin of one feature
        # shares one weight. With numeric features only this is an exact
        # no-op, because `UpdateFeatureWeightsForBestSplits` fills 1.0 and
        # returns before touching anything when the CTR count is zero
        # (`update_feature_weights.cpp:14-22`). It is here so that
        # `model_size_reg` is not a divergence the day CTRs land.
        #
        # IDENTITY_PATHS ROW 10: `ftz` BEFORE the argmax compare, not only
        # at the store. Two candidates whose gains differ by a denormal
        # compare UNEQUAL on a denormal-honoring backend and EQUAL on
        # Metal's FTZ hardware, and the tie rule then picks different
        # bins; flushing at the store alone would leave the ranking
        # divergent. Comptime no-op under FAST.
        #
        # DEVIATION 253 extends the flush to the INTERMEDIATE: with a
        # non-unit feature weight, a denormal difference times a large
        # weight is NORMAL again -- Metal (operands flushed) gets zero
        # where a denormal-honoring backend gets a nonzero product, and
        # the outer `ftz` cannot un-diverge that. Row 10's checklist:
        # intermediates stored through `ftz`.
        var gain = ftz(
            ftz(final_score - score_before)
            * ldg(feature_weights + feature_id)
        )

        if gain > best_gain:
            best_gain = gain
            best_bin = UInt32(bin_feature_id)

        offset += SCORE_BLOCK_SIZE * Int(grid_dim.x)

    # Block argmax by shared tree reduction. `__syncthreads` only, no shuffle
    # (`compute_scores.cu:20-51`), so this one ports without deviation.
    var s_score = stack_allocation[
        SCORE_BLOCK_SIZE,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var s_bin = stack_allocation[
        SCORE_BLOCK_SIZE,
        Scalar[DType.uint32],
        address_space = AddressSpace.SHARED,
    ]()
    s_score[tid] = best_gain
    s_bin[tid] = best_bin
    barrier()

    var stride = SCORE_BLOCK_SIZE // 2
    while stride > 0:
        if tid < stride:
            # `gains[tid] > gains[tid+s] || (gains[tid] == gains[tid+s] &&
            #  indices[tid] > indices[tid+s])` -- `compute_scores.cu:30`.
            # On an exact tie the SMALLER bin-feature index wins. Ours kept
            # the lower `tid`, and a lower `tid` does not mean a lower bin:
            # thread 0 covers bins 0, 128, 256..., thread 5 covers 5, 133...
            # Ties are the common case, not the rare one -- constant
            # features, duplicated columns and small-integer gradient sums
            # all produce exact float equality -- so this made the split
            # depend on block geometry.
            var take = s_score[tid + stride] > s_score[tid]
            if s_score[tid + stride] == s_score[tid]:
                if s_bin[tid + stride] < s_bin[tid]:
                    take = True
            if take:
                s_score[tid] = s_score[tid + stride]
                s_bin[tid] = s_bin[tid + stride]
        barrier()
        stride //= 2

    if tid == 0:
        # ================= THE "NO SPLIT FOUND" POISON RECORD =============
        #     if (index != -1 && index < binFeatureCount) { ...normal... }
        #     else { result->FeatureId = static_cast<ui32>(-1);
        #            result->BinId    = static_cast<ui32>(-1);
        #            result->Score    = FLT_MAX;
        #            result->Gain     = FLT_MAX; }
        # -- `compute_scores.cu:40-50`.
        #
        # This is a POISON value, not a default. Their host asserts on it:
        #     CB_ENSURE(bestSplits.size() == 1 &&
        #               bestSplits[0].FeatureId != static_cast<ui32>(-1),
        #               "All splits have infinite score. Probably, numerical
        #                overflow occurs in loss function and/or split score
        #                calculation. Try increasing l2_leaf_reg, and/or
        #                decreasing learning_rate, etc.");
        # -- `greedy_search_helper.cpp:535-539`.
        #
        # The kernel side writes the record. THE CALLER MUST RAISE ON IT.
        # Clamping it to bin 0 turns a numerical-overflow diagnostic into a
        # silently wrong tree that splits on whatever feature happens to own
        # bin-feature 0.
        # ==================================================================
        var index = s_bin[0]
        # IDENTITY_PATHS ROW 10: the block-winner record is a
        # kernel-to-host seam (the host reduce reads it), flushed through
        # `ftz`. The gains were already flushed before the compare, so
        # this is the seam's own guarantee rather than a change of value;
        # the sentinel is a normal number and passes through untouched.
        if index != UInt32(0xFFFFFFFF) and Int(index) < bin_feature_count:
            out_score.unsafe_store(Int(block_idx.x), ftz(s_score[0]))
            out_bin.unsafe_store(Int(block_idx.x), index)
        else:
            out_score.unsafe_store(Int(block_idx.x), ftz(-FLOAT32_MAX))
            out_bin.unsafe_store(Int(block_idx.x), UInt32(0xFFFFFFFF))


#: `ComputeTargetVariance`'s block size (`compute_scores.cu:290`).
comptime TARGET_VARIANCE_BLOCK = 512


def compute_target_variance_kernel(
    stats: MutPointer[Float32, MutAnyOrigin],
    size_in: Int32,
    stat_count_in: Int32,
    stat_line_size_in: Int32,
    is_multiclass: Int32,
    partials: MutPointer[Float32, MutAnyOrigin],
):
    """`ComputeTargetVarianceImpl` (`compute_scores.cu:226-283`).

    The greedy arm's standard deviation, over `target.StatsToAggregate`:
    column 0 is the WEIGHT plane and columns 1.. are the der planes, laid
    out `stats[statId * statLineSize + row]`.

        while (i < size) {
            const float w = stats[i];
            if (w > 1e-15f) {
                float statSum = 0;
                for (ui32 statId = 1; statId < statCount; ++statId) {
                    const float wt = stats[i + statLineSize * statId];
                    weightedSum += wt;
                    weightedSum2 += wt * wt / w;   //cause we need sum w*t*t
                    statSum += wt;
                }
                if (isMulticlass) {
                    weightedSum += -statSum;
                    weightedSum2 += statSum * statSum / w;
                }
                totalWeight += w;
            }
            i += gridDim.x * BlockSize;
        }

    THE MULTICLASS ARM IS THE PINNED CLASS AGAIN. `StatsToAggregate` holds
    `numClasses - 1` der planes because the multinomial gradient sums to
    zero across all of them, so the missing plane is exactly `-statSum`;
    its variance contribution is `statSum^2 / w`. The same identity the
    score kernel's `multiclass_optimization` arm above uses.

    `weightedSum` (lane 0) is COMPUTED AND NEVER READ -- their host
    comments the `double sum = l2StatsCpu[0];` line out
    (`greedy_search_helper.cpp:374`). Ported anyway, because dropping a
    lane would silently change nothing until someone reads it.

    DEVIATION 138: theirs reduces in `cub::BlockReduce<double>` and ends in
    `TAtomicAdd<double>` on three slots. Metal has no fp64 and this
    repository does not accept a float atomic on the fit path
    (`deterministic_sum_lanes_kernel`'s docstring), so each block STORES
    three Float32 partials at its own slot and the fold is deterministic.
    Same three sums, wider error, same bits every run.
    """
    var size = Int(size_in)
    var stat_count = Int(stat_count_in)
    var stat_line_size = Int(stat_line_size_in)

    var i = TARGET_VARIANCE_BLOCK * Int(block_idx.x) + Int(thread_idx.x)

    var weighted_sum = Float32(0.0)
    var weighted_sum2 = Float32(0.0)
    var total_weight = Float32(0.0)

    # IDENTITY_PATHS ROW 10 on the accumulation chain: `wt * wt` squares a
    # gradient, so a |wt| under ~1e-19 lands the product in the denormal
    # range -- Metal's hardware flushes it, CUDA's default keeps it, and
    # the two partials part ways. Every op's result is stored through
    # `ftz` (the measured flush-operands/flush-result model, and flushed
    # results make later operand flushes redundant). Comptime no-ops
    # under FAST, so the FAST chain is character-for-character the ops it
    # was.
    while i < size:
        var w = stats.unsafe_load(i)
        if w > Float32(1e-15):
            var stat_sum = Float32(0.0)
            for stat_id in range(1, stat_count):
                var wt = stats.unsafe_load(i + stat_line_size * stat_id)
                weighted_sum = ftz(weighted_sum + wt)
                weighted_sum2 = ftz(
                    weighted_sum2 + ftz(ftz(wt * wt) / w)
                )
                stat_sum = ftz(stat_sum + wt)
            if is_multiclass != Int32(0):
                weighted_sum = ftz(weighted_sum + -stat_sum)
                weighted_sum2 = ftz(
                    weighted_sum2 + ftz(ftz(stat_sum * stat_sum) / w)
                )
            total_weight = ftz(total_weight + w)
        i += Int(grid_dim.x) * TARGET_VARIANCE_BLOCK

    # IDENTITY_PATHS row 8 (DEVIATION 251's family): the pinned-shape
    # fold, so the within-block sum does not follow AMD's 64-wide
    # wavefront. FAST arm IS `block.sum`, verbatim.
    var b_sum = pinned_block_sum[block_size=TARGET_VARIANCE_BLOCK](
        weighted_sum
    )
    var b_sum2 = pinned_block_sum[block_size=TARGET_VARIANCE_BLOCK](
        weighted_sum2
    )
    var b_weight = pinned_block_sum[block_size=TARGET_VARIANCE_BLOCK](
        total_weight
    )

    if thread_idx.x == 0:
        # row 10: the partials buffer is a kernel-to-kernel seam
        # (`deterministic_sum_lanes_kernel` reads it), flushed.
        var slot = 3 * Int(block_idx.x)
        partials.unsafe_store(slot, ftz(b_sum))
        partials.unsafe_store(slot + 1, ftz(b_sum2))
        partials.unsafe_store(slot + 2, ftz(b_weight))


def target_variance_blocks(size: Int, sm_count: Int) -> Int:
    """`min(4 * TArchProps::SMCount(), CeilDivide(size, blockSize))`
    (`compute_scores.cu:291`).

    **AND IT IS A NUMERIC ROW, PINNED HERE FOR THE SAME REASON
    `partition_stats_chunks` IS** (IDENTITY_PATHS row 7, DEVIATION 353;
    DEVIATION 252 is the doc-parallel twin, `random_score_helper.
    std_dev_blocks`, and this mirrors it): the kernel above strides by
    `grid_dim.x * blockSize`, so the block count decides WHICH rows land
    in WHICH float partial, and the machine's core count then decides the
    last bits of the greedy arm's score-noise std dev -- every noised
    score of the fit flows through it. Inert at this port's default
    `random_strength = 0`; CatBoost's default is 1.0, so the row must
    hold before that default is wired.

    Under `IDENTICAL` the `sm_count` fed to the formula comes from
    `kernel_matrix.partition_chunks_sm_for`, the SAME pin row 7 uses (32
    on every vendor, deliberately no real device's own number); under
    `FAST` it is the device's count, CatBoost's behavior unchanged. The
    pin is applied INSIDE this function -- the only place the formula
    lives -- so the launch and the `partials` buffer sizing in
    `compute_target_std_dev` cannot disagree, which is row 7's exact
    argument. `by_data` is pure f(size) and needs no pin.
    """
    comptime _identical = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var sm = partition_chunks_sm_for[_identical](sm_count)
    var by_data = (size + TARGET_VARIANCE_BLOCK - 1) // TARGET_VARIANCE_BLOCK
    var by_machine = 4 * sm
    if by_machine < 1:
        by_machine = 1
    if by_data < by_machine:
        return by_data if by_data > 0 else 1
    return by_machine


# =====================================================================
# THE LEAFWISE SCORERS -- `EGrowPolicy::Lossguide` and `EGrowPolicy::Region`
#
# Added 2026-08-22 by the LOSSGUIDE lane. Everything above this line is the
# SYMMETRIC arm and is not touched by it; `LOSSGUIDE.md` carries the lane
# boundary and the deviation block 300-349.
#
# `compute_scores.cu` holds THREE `__global__`s. The symmetric one above
# (`:56`) scores a candidate ONCE for the whole level, summing over every
# leaf. The other two score a candidate PER LEAF, because a non-symmetric
# tree gives each leaf its own split:
#
#   `ComputeOptimalSplitsRegion` (`:303-385`)  Depthwise and Region.
#       leaf ids come from a buffer:  partIds += blockIdx.y; partIds[0]
#   `ComputeOptimalSplit`        (`:393-475`)  Lossguide.
#       leaf ids are two SCALARS:  blockIdx.y == 0 ? partId : maybeSecondPartId
#
# ============================ DEVIATION 317 ============================
# THE TWO BODIES ARE THE SAME BODY. Diffed line for line: `:406-472` against
# `:316-382` is character-identical apart from the three lines that produce
# `thisPartId`. Theirs are two separate `__global__` templates because CUDA
# gives them no cheaper way to vary one load; ours is ONE `@always_inline`
# scan (`_leafwise_scan_part`) plus one `@always_inline` block argmax
# (`_leafwise_argmax_write`), and each policy's kernel is the four lines that
# pick a part id and call them.
#
# This is a deviation of SHAPE and it is priced at zero: the emitted code per
# kernel is the same instruction sequence their template instantiation emits,
# and the two arms cannot drift apart because there is one arm. It is
# recorded rather than assumed because the "identical bodies" claim is a
# claim about THEIR file, and a future CatBoost that edits one and not the
# other would break it silently. Re-diff `:303-475` when the pin moves.
#
# The Lossguide kernel is written here. **The Depthwise/Region kernel is the
# depthwise lane's and belongs at the foot of this block** -- it is four
# lines and both helpers are already exported for it.
# =======================================================================
#
# ================== WHAT THE SYMMETRIC ARM DOES NOT DO ==================
# Three things live in the leafwise bodies and in NEITHER the symmetric
# kernel above nor its port, so none of this is reachable by editing the
# arm above and none of it is redundant:
#
#   1. `beforeSplitCalcer` IS FED (`:341`, `:431`). The symmetric copy is
#      taken after `NextFeature` and never sees a leaf, so its score is a
#      constant zero and the port says so. Here it accumulates the PARENT
#      as one leaf -- `AddLeaf(partStat, partWeight)` -- so `gain = after -
#      before` is a real difference of two real scores, which is what makes
#      a per-leaf gain comparable ACROSS leaves. That comparability is the
#      whole of Lossguide: `FindBestLeafToSplit` argmins over it.
#
#   2. `toZeroPartSplit` (`:337-340`, `:427-430`). If either side's weight
#      falls under 1e-20 the candidate is not scored at all: both scores
#      become the sentinel and **the gain is forced to exactly 0**, not to
#      the sentinel. In their sign 0 beats every positive gain, so a
#      degenerate split still wins a leaf on which NOTHING improves. That
#      is not a bug to round off -- it is how a leaf with no usable split
#      still reports a defined `BestSplit`, which is what keeps
#      `FindBestLeafToSplit` from picking it while other leaves have real
#      gains, and what lets `IsTerminalLeaf` be the thing that stops the
#      tree instead. Ported exactly, sign-flipped once.
#
#   3. `bestScore = gain` (`:468`), where the symmetric kernel sets
#      `bestScore = score` (`:143`). **CHECKED, because Lossguide's leaf
#      argmin reads `BestSplit.Score` and not `BestSplit.Gain`**
#      (`greedy_search_helper.cpp:300`), while the cross-block reduce reads
#      `operator<` which is keyed on `Gain` (`gpu_structures.h:80-93`). If
#      the two fields differed on this path, this port's collapse of Score
#      and Gain into one number would silently change WHICH LEAF gets
#      split. They do not differ: on both leafwise kernels `bestScore` and
#      `bestGain` are assigned the same `gain`. The collapse is correct
#      here and would NOT have been correct on the symmetric arm.
# =======================================================================


#: `const int blockSize = 256` for BOTH leafwise launchers
#: (`compute_scores.cu:484`, `:557`), where the symmetric launcher uses 128
#: (`:167`, `SCORE_BLOCK_SIZE` above). Their number, not a tuning of ours.
#:
#: SCHEDULING, not numeric, and the argument is not "it is a block size":
#: every thread scores its own candidates with no cross-thread accumulation,
#: and the block argmax below is a total order (gain, then smaller bin id),
#: so no reduction shape can move the answer. `numerics.mojo`'s test is
#: whether a row changes the SEQUENCE of arithmetic, and this one cannot.
comptime LEAFWISE_SCORE_BLOCK_SIZE = 256


@always_inline
def _leafwise_scan_part[
    score_function: Int, normalize: Bool, block_size: Int
](
    this_part_id: Int,
    bf_skip: MutPointer[UInt8, MutAnyOrigin],
    bin_feature_count: Int,
    bf_feature_id: MutPointer[UInt32, MutAnyOrigin],
    feature_weights: MutPointer[Float32, MutAnyOrigin],
    histograms: MutPointer[Float32, MutAnyOrigin],
    part_stats: MutPointer[Float32, MutAnyOrigin],
    stat_count: Int,
    multiclass_optimization: Int32,
    lambda_l2: Float32,
    score_std_dev: Float32,
    global_seed: UInt64,
    mut best_gain: Float32,
    mut best_bin: UInt32,
):
    """One leaf's candidate scan -- `compute_scores.cu:406-472`, both copies.

    Their grid-stride loop over bin features, the two calcers, the skip
    test, the multiclass arm and the noise draw. Leaves `best_gain` and
    `best_bin` holding this thread's winner, in THIS port's sign (larger is
    better; see the module docstring's sign block).
    """
    comptime cosine = (
        score_function == SCORE_FUNCTION_COSINE
        or score_function == SCORE_FUNCTION_NEWTON_COSINE
    )
    comptime assert (
        cosine
        or score_function == SCORE_FUNCTION_L2
        or score_function == SCORE_FUNCTION_NEWTON_L2
    ), (
        "score_function has no calcer here; only Cosine, NewtonCosine, L2"
        " and NewtonL2 are ported"
    )

    var tid = Int(thread_idx.x)
    var leaf_base = this_part_id * stat_count * bin_feature_count

    var offset = Int(block_idx.x) * block_size
    while offset < bin_feature_count:
        var bin_feature_id = offset + tid
        if bin_feature_id >= bin_feature_count:
            break
        if bf_skip.unsafe_load(bin_feature_id) != 0:
            offset += block_size * Int(grid_dim.x)
            continue

        # `calcer.NextFeature(bf[binFeatureId]); TScoreCalcer
        # beforeSplitCalcer = calcer;` (`:424-425`). BOTH start from the
        # `NextFeature` seed, which is what makes their noise draws
        # identical and therefore cancelling.
        var score = Float32(0.0)
        var denum_sqr = Float32(1e-10)
        var score_b = Float32(0.0)
        var denum_sqr_b = Float32(1e-10)

        # `const double partWeight = __ldg(partStats + thisPartId *
        # statCount)` (`:427`), stat plane 0.
        var part_weight = ldg(part_stats + this_part_id * stat_count)
        var weight_left = max(
            ldg(histograms + (leaf_base + bin_feature_id)), Float32(0.0)
        )
        # IDENTITY_PATHS ROW 10: the subtraction can cancel into the
        # denormal range, and a denormal weight fed to `sum / (weight +
        # lambda)` diverges between Metal's FTZ hardware and a
        # denormal-honoring backend (lambda may be 0, so the denominator
        # is not guaranteed away from it). Flushed HERE, where the value
        # is derived, so every consumer sees one policy. Comptime no-op
        # under FAST.
        var weight_right = ftz(max(part_weight - weight_left, Float32(0.0)))

        # `bool toZeroPartSplit = false; if (weightLeft < 1e-20f ||
        # weightRight < 1e-20f) { toZeroPartSplit = true; }` (`:430-434`).
        var to_zero_part_split = (
            weight_left < Float32(1e-20) or weight_right < Float32(1e-20)
        )

        var total_sum_left = Float32(0.0)
        var total_sum_part = Float32(0.0)

        # ======================== DEVIATION 255 ======================
        # THE LEAFWISE ARM'S TWO REMAINING SCORE-PATH GAPS, closed with
        # DEVIATION 253's exact pattern from the symmetric arm above.
        # 253 reported them open when it pinned the symmetric body -- the
        # `total_sum_*` accumulations here ran unflushed (the symmetric
        # twins are flushed at their `totalSumLeft += sumLeft` lines and
        # at the multiclass `totalSumPart - totalSumLeft` derivation),
        # and the gain intermediate below went into the feature-weight
        # multiply unflushed (the symmetric arm flushes it BEFORE the
        # multiply, because a denormal difference times a large weight
        # is normal again and the outer flush cannot un-diverge that).
        # Sums of SIGNED gradients cancel into the denormal range, and
        # both totals become `AddLeaf` operands on the multiclass arm.
        # The noise fma results ride along, as they do on the symmetric
        # arm -- each is an operand of the gain subtraction. Every
        # `ftz` is a comptime no-op under FAST, so the FAST arm is
        # character for character the chain it was.
        # =============================================================
        for stat_id in range(1, stat_count):
            var stat_slot = (
                leaf_base + stat_id * bin_feature_count + bin_feature_id
            )
            var sum_left = ldg(histograms + stat_slot)
            var part_stat = ldg(
                part_stats + (this_part_id * stat_count + stat_id)
            )
            # DEVIATION 255 / row 10: flushed accumulation, the
            # symmetric arm's `total_sum_part` twin.
            total_sum_part = ftz(total_sum_part + part_stat)
            # row 10 again: `partStat - sumLeft` is THE cancellation step
            # of the kernel (module docstring), so it is the likeliest
            # producer of a denormal on this path. Flushed at derivation.
            var sum_right = ftz(part_stat - sum_left)

            # `calcer.AddLeaf(sumLeft, weightLeft); calcer.AddLeaf(sumRight,
            # weightRight); beforeSplitCalcer.AddLeaf(partStat, partWeight);`
            # (`:443-446`). The THIRD call is the one the symmetric arm has
            # no counterpart for.
            _add_leaf[score_function, normalize, True](
                sum_left, weight_left, lambda_l2, score, denum_sqr
            )
            _add_leaf[score_function, normalize, True](
                sum_right, weight_right, lambda_l2, score, denum_sqr
            )
            _add_leaf[score_function, normalize, True](
                part_stat, part_weight, lambda_l2, score_b, denum_sqr_b
            )
            # DEVIATION 255 / row 10: `totalSumLeft += sumLeft` (`:447`),
            # flushed like the symmetric arm's twin.
            total_sum_left = ftz(total_sum_left + sum_left)

        # `if (multiclassOptimization) { ... beforeSplitCalcer.AddLeaf(
        # -totalSumPart, partWeight); }` (`:448-454`). Same pinned-class
        # identity as the symmetric arm, plus the before-calcer's own term.
        if multiclass_optimization != Int32(0):
            # DEVIATION 255 / row 10: one more cancelling subtraction,
            # flushed at derivation like the symmetric arm's.
            var total_sum_right = ftz(total_sum_part - total_sum_left)
            _add_leaf[score_function, normalize, True](
                -total_sum_left, weight_left, lambda_l2, score, denum_sqr
            )
            _add_leaf[score_function, normalize, True](
                -total_sum_right, weight_right, lambda_l2, score, denum_sqr
            )
            _add_leaf[score_function, normalize, True](
                -total_sum_part, part_weight, lambda_l2,
                score_b, denum_sqr_b,
            )

        var feature_id = Int(bf_feature_id.unsafe_load(bin_feature_id))

        # `const float scoreAfter = !skip ? calcer.GetScore() : FLT_MAX;`
        # `const float scoreBefore = !skip ? beforeSplitCalcer.GetScore()
        #  : FLT_MAX;` (`:458-459`), both sentinels negated for our sign.
        var final_score = score
        var score_before = score_b
        comptime if cosine:
            # IDENTITY_PATHS ROW 10 IS TITLED "division and sqrt in
            # scoring", and these two quotients are that row: the quotient
            # of a cancelled numerator can land denormal, where Metal's
            # FTZ hardware and a denormal-honoring backend part ways.
            # Flushed at the derivation; comptime no-op under FAST.
            if denum_sqr > Float32(1e-15):
                final_score = ftz(score / identical_sqrt(denum_sqr))
            else:
                final_score = -FLOAT32_MAX
            if denum_sqr_b > Float32(1e-15):
                score_before = ftz(score_b / identical_sqrt(denum_sqr_b))
            else:
                score_before = -FLOAT32_MAX

            # `score_calcers.cuh:162-166`, run by BOTH calcers because
            # `GetScore()` is called on both. Same seed, same feature id,
            # same draw -- so it cancels in the difference exactly as it
            # does on the symmetric arm, and is ported for the same reason
            # (see the noise block above).
            if score_std_dev != Float32(0.0):
                var seed = advance_seed_k(
                    global_seed + UInt64(feature_id), 4
                )
                var draw = next_normal_f(seed)
                # IDENTITY_PATHS ROW 9, same seam as the symmetric arm's
                # noise block above: a multiply feeding a subtract is a
                # contraction seam, pinned to one `fma` under IDENTICAL;
                # the FAST arm is the same mul-then-add chain bit for bit.
                # DEVIATION 255 / row 10 rides along, as DEVIATION 253
                # does on the symmetric arm -- each fma result is an
                # operand of the gain subtraction below, so it is
                # stored through `ftz`. Comptime no-op under FAST.
                var neg_draw = -draw[0]
                final_score = ftz(
                    identical_mul_add(
                        neg_draw, score_std_dev, final_score
                    )
                )
                score_before = ftz(
                    identical_mul_add(
                        neg_draw, score_std_dev, score_before
                    )
                )

        if to_zero_part_split:
            final_score = -FLOAT32_MAX
            score_before = -FLOAT32_MAX

        # `float gain = !skip ? (scoreAfter - scoreBefore) : 0;` (`:463`)
        # then `gain *= __ldg(binFeaturesWeights + featureId)` (`:466`).
        #
        # THE ZERO IS NOT THE SENTINEL. Writing `-FLOAT32_MAX - -FLOAT32_MAX`
        # here would give NaN, and writing `-FLOAT32_MAX` would make a
        # degenerate candidate lose to everything -- both change which split
        # a hopeless leaf reports. Theirs is a literal 0 and so is this.
        var gain = Float32(0.0)
        if not to_zero_part_split:
            # DEVIATION 255 extends the flush to the INTERMEDIATE, the
            # symmetric arm's DEVIATION 253 argument verbatim: a
            # denormal difference times a large feature weight is
            # NORMAL again, so the outer `ftz` cannot un-diverge it.
            # The degenerate arm's literal 0 needs no flush.
            gain = ftz(final_score - score_before)
        # IDENTITY_PATHS ROW 10: flushed BEFORE the argmax compare, same
        # argument as the symmetric arm's gain -- flushing only at the
        # store would leave the in-block RANKING divergent on a
        # denormal-honoring backend. Comptime no-op under FAST.
        gain = ftz(gain * ldg(feature_weights + feature_id))

        if gain > best_gain:
            best_gain = gain
            best_bin = UInt32(bin_feature_id)

        offset += block_size * Int(grid_dim.x)


@always_inline
def _leafwise_argmax_write[
    block_size: Int
](
    best_gain: Float32,
    best_bin: UInt32,
    bin_feature_count: Int,
    out_slot: Int,
    out_score: MutPointer[Float32, MutAnyOrigin],
    out_bin: MutPointer[UInt32, MutAnyOrigin],
):
    """Their `ARGMAX()` macro (`compute_scores.cu:20-51`) for the leafwise
    kernels, with the same tie rule and the same poison record.

    `out_slot` is their `result += blockIdx.x + blockIdx.y * gridDim.x`
    (`:319`, `:406`): one record per (argmax block, score block) pair, which
    is the layout their host reduce walks (`greedy_search_helper.cpp:520-528`).

    **The symmetric kernel above inlines this same reduction rather than
    calling this helper.** That is duplication, recorded rather than fixed,
    because folding the symmetric arm onto this helper means editing another
    lane's live arm in a shared file.

    THE SENTENCE THAT WAS HERE IS DELETED AS FALSE. It said "the two are
    gated against each other in `original/leafwise_scores_check.mojo`".
    They are not: that file never launches `compute_optimal_splits_kernel`
    at all, so its G4 poison-record and G5 tie-rule gates exercise THIS copy
    only. A duplication audit found it, and the reason it survived is
    exactly the reason it was worth writing down -- a comment asserting a
    gate exists is as load-bearing as the gate, and nothing checks comments.

    **So the symmetric arm's tie rule and poison record are REACHED BUT
    UNGATED.** The two bodies were read line by line at the time of that
    audit and are behaviorally identical -- same `>` with the smaller-bin
    tie-break, same `(ui32)-1` poison record. That is a reading, not a
    measurement, and it is the honest status.
    """
    var tid = Int(thread_idx.x)
    var s_score = stack_allocation[
        block_size,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var s_bin = stack_allocation[
        block_size,
        Scalar[DType.uint32],
        address_space = AddressSpace.SHARED,
    ]()
    s_score[tid] = best_gain
    s_bin[tid] = best_bin
    barrier()

    var stride = block_size // 2
    while stride > 0:
        if tid < stride:
            # Their tie rule, sign-flipped once: on an exact tie the SMALLER
            # bin-feature index wins (`:30`).
            var take = s_score[tid + stride] > s_score[tid]
            if s_score[tid + stride] == s_score[tid]:
                if s_bin[tid + stride] < s_bin[tid]:
                    take = True
            if take:
                s_score[tid] = s_score[tid + stride]
                s_bin[tid] = s_bin[tid + stride]
        barrier()
        stride //= 2

    if tid == 0:
        var index = s_bin[0]
        # IDENTITY_PATHS ROW 10: kernel-to-host seam, flushed -- same
        # note as the symmetric argmax's store above.
        if index != UInt32(0xFFFFFFFF) and Int(index) < bin_feature_count:
            out_score.unsafe_store(out_slot, ftz(s_score[0]))
            out_bin.unsafe_store(out_slot, index)
        else:
            # The poison record (`:44-49`). THE CALLER MUST RAISE ON IT --
            # and on the Lossguide path the caller has one more duty the
            # symmetric path does not: a poisoned leaf must be left with an
            # UNDEFINED `BestSplit`, so `FindBestLeafToSplit` skips it
            # (`greedy_search_helper.cpp:299`) instead of splitting a leaf
            # whose score is a sentinel.
            out_score.unsafe_store(out_slot, ftz(-FLOAT32_MAX))
            out_bin.unsafe_store(out_slot, UInt32(0xFFFFFFFF))


def compute_optimal_split_kernel[
    score_function: Int = SCORE_FUNCTION_COSINE,
    normalize: Bool = False,
](
    bf_skip: MutPointer[UInt8, MutAnyOrigin],
    bin_feature_count_in: Int32,
    bf_feature_id: MutPointer[UInt32, MutAnyOrigin],
    feature_weights: MutPointer[Float32, MutAnyOrigin],
    histograms: MutPointer[Float32, MutAnyOrigin],
    part_stats: MutPointer[Float32, MutAnyOrigin],
    stat_count_in: Int32,
    part_id_in: Int32,
    maybe_second_part_id_in: Int32,
    multiclass_optimization: Int32,
    lambda_l2: Float32,
    score_std_dev: Float32,
    global_seed: UInt64,
    out_score: MutPointer[Float32, MutAnyOrigin],
    out_bin: MutPointer[UInt32, MutAnyOrigin],
):
    """`ComputeOptimalSplit` (`compute_scores.cu:393-475`) -- LOSSGUIDE.

    **Two leaves at most, and they arrive as scalars.** A Lossguide iteration
    splits exactly one leaf, which creates exactly two leaves without a
    `BestSplit`, and `SelectLeavesToVisit` returns exactly the leaves that
    lack one (`greedy_search_helper.cpp:697-710`). Their host asserts it:
    `CB_ENSURE(leavesToVisit.size() <= 2)` (`:511`). The root iteration
    returns one leaf, and their launcher then sets `numBlocks.y = partId ==
    maybeSecondPartId ? 1 : 2` (`:570`) -- so the CALLER passes the same id
    twice and launches one block row. Passing two equal ids with a two-row
    grid would score the same leaf twice and hand the host a duplicate.

    Grid is `(argmax_block_count, 1 or 2, 1)`, block is
    `LEAFWISE_SCORE_BLOCK_SIZE`.
    """
    var this_part_id = Int(part_id_in)
    if Int(block_idx.y) != 0:
        this_part_id = Int(maybe_second_part_id_in)

    var bin_feature_count = Int(bin_feature_count_in)
    var best_gain = -FLOAT32_MAX
    var best_bin = UInt32(0xFFFFFFFF)

    _leafwise_scan_part[
        score_function, normalize, LEAFWISE_SCORE_BLOCK_SIZE
    ](
        this_part_id,
        bf_skip,
        bin_feature_count,
        bf_feature_id,
        feature_weights,
        histograms,
        part_stats,
        Int(stat_count_in),
        multiclass_optimization,
        lambda_l2,
        score_std_dev,
        global_seed,
        best_gain,
        best_bin,
    )

    _leafwise_argmax_write[LEAFWISE_SCORE_BLOCK_SIZE](
        best_gain,
        best_bin,
        bin_feature_count,
        Int(block_idx.x) + Int(block_idx.y) * Int(grid_dim.x),
        out_score,
        out_bin,
    )


def compute_optimal_splits_region_kernel[
    score_function: Int = SCORE_FUNCTION_COSINE,
    normalize: Bool = False,
](
    bf_skip: MutPointer[UInt8, MutAnyOrigin],
    bin_feature_count_in: Int32,
    bf_feature_id: MutPointer[UInt32, MutAnyOrigin],
    feature_weights: MutPointer[Float32, MutAnyOrigin],
    histograms: MutPointer[Float32, MutAnyOrigin],
    part_stats: MutPointer[Float32, MutAnyOrigin],
    stat_count_in: Int32,
    part_ids: MutPointer[UInt32, MutAnyOrigin],
    multiclass_optimization: Int32,
    lambda_l2: Float32,
    score_std_dev: Float32,
    global_seed: UInt64,
    out_score: MutPointer[Float32, MutAnyOrigin],
    out_bin: MutPointer[UInt32, MutAnyOrigin],
):
    """`ComputeOptimalSplitsRegion` (`compute_scores.cu:303-385`) -- DEPTHWISE.

    Added by the DEPTHWISE lane at the foot of the lossguide lane's block,
    where DEVIATION 317 says it belongs. **Four lines and two calls**, because
    that deviation is exactly the finding that the two leafwise bodies are one
    body: `:316-382` is character-identical to `:406-472` apart from how
    `thisPartId` is produced.

    This one takes it from a BUFFER, one leaf per block row:

        result  += blockIdx.x + blockIdx.y * gridDim.x;
        partIds += blockIdx.y;
        const int thisPartId = partIds[0];

    where Lossguide's takes two scalars. That is the whole difference, and it
    is the reason `numScoreBlocks` can be `leavesToVisit.size()` here
    (`greedy_search_helper.cpp:428-432`) and is capped at 2 there
    (`:511`).

    **This is the FIRST version of this kernel in this file and not the
    second.** An earlier standalone copy of the body was written here before
    the lossguide lane landed its factoring; it has been deleted rather than
    left beside this one, and it carried a real defect that the factoring
    fixes: it used `SCORE_BLOCK_SIZE` (128), which is the SYMMETRIC launcher's
    block (`:167`). Both leafwise launchers use 256 (`:484`, `:557`). Their
    number, and it was ours to get wrong.

    Grid is `(argmax_block_count, leavesToVisit.size(), 1)`, block is
    `LEAFWISE_SCORE_BLOCK_SIZE`.
    """
    # `partIds += blockIdx.y; const int thisPartId = partIds[0];` (`:320-321`)
    var this_part_id = Int(part_ids.unsafe_load(Int(block_idx.y)))

    var bin_feature_count = Int(bin_feature_count_in)
    var best_gain = -FLOAT32_MAX
    var best_bin = UInt32(0xFFFFFFFF)

    _leafwise_scan_part[
        score_function, normalize, LEAFWISE_SCORE_BLOCK_SIZE
    ](
        this_part_id,
        bf_skip,
        bin_feature_count,
        bf_feature_id,
        feature_weights,
        histograms,
        part_stats,
        Int(stat_count_in),
        multiclass_optimization,
        lambda_l2,
        score_std_dev,
        global_seed,
        best_gain,
        best_bin,
    )

    _leafwise_argmax_write[LEAFWISE_SCORE_BLOCK_SIZE](
        best_gain,
        best_bin,
        bin_feature_count,
        Int(block_idx.x) + Int(block_idx.y) * Int(grid_dim.x),
        out_score,
        out_bin,
    )
