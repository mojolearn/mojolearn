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
from std.math import sqrt
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.memory import stack_allocation

from ported.options.catboost_options import (
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
    lambda_l2: Float32,
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

        # `TScoreCalcer beforeSplitCalcer = calcer` (`:85`) is NOT ported.
        # In THIS kernel -- the oblivious one -- the copy is taken
        # immediately after `NextFeature` and is never fed a leaf, so
        # `scoreBefore` is 0 for both calcers (`GetScore()` on an empty
        # cosine calcer is `-0 / sqrt(1e-10)`), and `gain = score - 0`.
        # Their leafwise kernels DO feed it (`:341`), which is why the
        # variable exists at all.

        # THE SERIAL LEAF LOOP. See the module docstring for why it is inside
        # the thread and not across threads.
        for i in range(p_count):
            var leaf_id = Int(part_ids.unsafe_load(i))
            var leaf_base = leaf_id * stat_count * bin_feature_count

            # stat 0 is the weight plane.
            var weight_left = max(
                histograms.unsafe_load(leaf_base + bin_feature_id),
                Float32(0.0),
            )
            var weight_right = max(
                part_stats.unsafe_load(leaf_id * stat_count) - weight_left,
                Float32(0.0),
            )

            for stat_id in range(1, stat_count):
                var sum_left = histograms.unsafe_load(
                    leaf_base + stat_id * bin_feature_count + bin_feature_id
                )
                var part_stat = part_stats.unsafe_load(
                    leaf_id * stat_count + stat_id
                )
                var sum_right = part_stat - sum_left

                # `calcer.AddLeaf(sumLeft, weightLeft)` then
                # `calcer.AddLeaf(sumRight, weightRight)`
                # -- `compute_scores.cu:101-102`.
                comptime if cosine:
                    # `TCosineScoreCalcer::AddLeaf` --
                    # `score_calcers.cuh:152-157`:
                    #     double lambda = Normalize ? Lambda * weight
                    #                               : Lambda;
                    #     mu = weight > 0.0f ? sum / (weight + lambda) : 0.0f;
                    #     Score += sum * mu;
                    #     DenumSqr += weight * mu * mu;
                    var lam_l = lambda_l2
                    comptime if normalize:
                        lam_l = lambda_l2 * weight_left
                    var mu_l = Float32(0.0)
                    if weight_left > Float32(0.0):
                        mu_l = sum_left / (weight_left + lam_l)
                    score += sum_left * mu_l
                    denum_sqr += weight_left * mu_l * mu_l

                    var lam_r = lambda_l2
                    comptime if normalize:
                        lam_r = lambda_l2 * weight_right
                    var mu_r = Float32(0.0)
                    if weight_right > Float32(0.0):
                        mu_r = sum_right / (weight_right + lam_r)
                    score += sum_right * mu_r
                    denum_sqr += weight_right * mu_r * mu_r
                else:
                    # `TL2ScoreCalcer::AddLeaf` at `MetaExponent == 1`:
                    # `leafScore = (weight > 1e-20f ? (-sum*sum)/
                    #  (weight+Lambda) : 0)` -- `score_calcers.cuh:54`.
                    # Every calcer of theirs carries this guard. Ours divided
                    # unconditionally. The weights are `max(.., 0)` clamped
                    # but the SUMS are not, so a side whose weight cancels to
                    # zero with a non-zero residual contributed
                    # `sum^2 / lambda` here and 0 in theirs, and divides by
                    # zero outright if lambda is ever 0.
                    if weight_left > Float32(1e-20):
                        score += (sum_left * sum_left) / (
                            weight_left + lambda_l2
                        )
                    if weight_right > Float32(1e-20):
                        score += (sum_right * sum_right) / (
                            weight_right + lambda_l2
                        )

        # `float score = calcer.GetScore()` -- `compute_scores.cu:131`.
        var final_score = score
        comptime if cosine:
            # `DenumSqr > 1e-15f ? -Score / sqrt(DenumSqr) : FLT_MAX`
            # -- `score_calcers.cuh:161`. Both branches negated for our
            # sign, so their "unusable" FLT_MAX becomes -FLOAT32_MAX, which
            # loses every comparison here exactly as FLT_MAX loses every
            # comparison there.
            #
            # The `ScoreStdDev` noise term that follows it
            # (`score_calcers.cuh:162-166`) is NOT ported: it is
            # `random_strength`, which `CatBoostOptions.check()` refuses.
            if denum_sqr > Float32(1e-15):
                final_score = score / sqrt(denum_sqr)
            else:
                final_score = -FLOAT32_MAX

        # `gain = score - scoreBefore`, then
        # `gain *= __ldg(binFeaturesWeights + featureId)` where
        # `featureId = bf[binFeatureId].FeatureId`
        # -- `compute_scores.cu:134-137`. `scoreBefore` is 0 here, see above.
        #
        # Indexed by FEATURE, not by bin-feature: every bin of one feature
        # shares one weight. With numeric features only this is an exact
        # no-op, because `UpdateFeatureWeightsForBestSplits` fills 1.0 and
        # returns before touching anything when the CTR count is zero
        # (`update_feature_weights.cpp:14-22`). It is here so that
        # `model_size_reg` is not a divergence the day CTRs land.
        var gain = final_score * feature_weights.unsafe_load(
            Int(bf_feature_id.unsafe_load(bin_feature_id))
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
        if index != UInt32(0xFFFFFFFF) and Int(index) < bin_feature_count:
            out_score.unsafe_store(Int(block_idx.x), s_score[0])
            out_bin.unsafe_store(Int(block_idx.x), index)
        else:
            out_score.unsafe_store(Int(block_idx.x), -FLOAT32_MAX)
            out_bin.unsafe_store(Int(block_idx.x), UInt32(0xFFFFFFFF))
