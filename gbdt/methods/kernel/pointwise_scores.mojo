"""CatBoost's POINTWISE split scorer and its five score calcers, ported.

PORT OF `catboost/cuda/methods/kernel/pointwise_scores.cu` (698 lines) and
`catboost/cuda/methods/kernel/score_calcers.cuh` (188 lines) at CatBoost
`54a8143a`. Transliterated. Do not improve.

WHICH FAMILY THIS IS
--------------------
The POINTWISE family, the one BOTH of CatBoost's oblivious tree searchers
share (`PORTING.md` 91 B): `TFeatureParallelObliviousTreeSearcher` and
`TDocParallelObliviousTreeSearcher`. It reads the histograms that
`pointwise_hist2*` produce and `split_properties_helpers.ScanHistogramsImpl`
scans.

It is NOT `greedy_subsets_searcher/kernel/compute_scores.cu`. Those are two
separate upstream files with two separate histogram layouts, and this port
shares no line with the other one. Where they DO overlap is
`score_calcers.cuh`, which upstream `#include`s into both
(`compute_scores.cu:9` and `pointwise_scores.cu:2`) -- so the calcer
arithmetic below is the same arithmetic our greedy port inlined, and the
two ports disagree on SIGN by design; see the next section.

THE HISTOGRAM LAYOUT, WHICH IS FIXED AND NOT NEGOTIABLE
-------------------------------------------------------
For bin-feature index `b` (which is `FirstFoldIndex[f] + fold` for feature
`f`, exactly as `split_properties_helpers.scan_pointwise_histograms_kernel`
writes it), partition `leaf`, fold `fold`, and stat `w`:

    binSums[((histogramOffset(leaf, fold) * binFeatureCount) + b) * 2 + w]

        w = 0  ->  WEIGHT
        w = 1  ->  TARGET

That is STAT-MINOR and WEIGHT-BEFORE-TARGET. Their spelling of the same
address is `const float* current = binSums + 2 * (i + tid);` followed by
`current[(size_t)binFeatureCount * helper.GetHistogramOffset(leaf, fold) * 2
+ {0,1}]` (`:87-104`, `:359-372`), and `TDirectHistLoader` (`:157-181`) is
the single-fold spelling of it.

THE GREEDY-SUBSETS FAMILY USES THE OPPOSITE CONVENTION ON BOTH COUNTS --
stat-MAJOR, and it carries the weight in the LAST stat slot rather than the
first. Copying an index expression across the two families transposes the
histogram, and a transposition moves no total, so a check that sums cannot
see it ([[uniform-test-data-hides-permutation]]).

THE SIGN IS THEIRS, AND IT IS THE OPPOSITE OF OUR GREEDY PORT
-------------------------------------------------------------
"in GPU catboost all scores are inverse, lower is better". Every calcer
here returns a NEGATED objective, the sentinel for "unusable" is `+FLT_MAX`
so that it loses every comparison, and the argmin keeps the SMALLEST gain
(`gain < bestGain`, `:120-124`).

`gbdt/methods/greedy_subsets_searcher/kernel/compute_scores.mojo` folds a
negation into every calcer and flips every comparison, because its host side
was written against "larger is better". THIS FILE DOES NOT. It has no host
side yet (see UNWIRED below), so there is nothing to accommodate, and
PORTING_RULES 0b says copy. **Whoever wires this up: the two files in this
repository disagree about the sign of a score, on purpose, and the
disagreement is upstream's own convention preserved here and inverted
there.**

TIE-BREAK. The block reduction is `gains[tid] > gains[tid + s] ||
(gains[tid] == gains[tid + s] && indices[tid] > indices[tid + s])`
(`:145-152`, `:291-298`, `:418-425`): smallest gain wins, and on an exact
tie the SMALLER BIN-FEATURE INDEX wins. Ties are the common case, not the
rare one -- constant features, duplicated columns and small-integer
gradient sums all produce exact float equality -- so a reduction that keeps
the lower `tid` instead makes the split depend on block geometry.

WHAT IS IN THIS PORT
--------------------
Everything in both files:

  `TSolarScoreCalcer`, `TL2ScoreCalcer` (both `MetaExponent` arms),
  `TLOOL2ScoreCalcer`, `TSatL2ScoreCalcer`, `TCosineScoreCalcer`
      -> `ScoreCalcer[score_function]`, one tagged union (PORTING_RULES 4).
  `ComputeSum<BLOCK_SIZE>`          -> `_compute_sum[block_size]`
  `FindOptimalSplitSolarImpl`       -> `find_optimal_split_solar_kernel`
  `TDirectHistLoader`,
  `TGatheredByLeavesHistLoader`     -> `hist_loader` comptime switch
  `FindOptimalSplitSingleFoldImpl`  -> `find_optimal_split_single_fold_kernel`
  `FindOptimalSplitCosineImpl`      -> `find_optimal_split_cosine_kernel`
  `FindOptimalSplitDynamic`         -> `find_optimal_split_dynamic`
  `FindOptimalSplitPlain`           -> `find_optimal_split_plain`
  `FindOptimalSplit`                -> `find_optimal_split`
  `GatherHistogramsByLeavesImpl`    -> `gather_histograms_by_leaves_kernel`
  `GatherHistogramByLeaves`         -> `gather_histogram_by_leaves`
  `PartitionUpdateImpl`             -> `partition_update_kernel`
  `UpdatePartitionProps`            -> `update_partition_props`

UNWIRED. Nothing in this repository calls any of it yet. The callers are
`pointwise_kernels.{h,cpp}` -> `pointwise_scores_calcer.h` ->
`oblivious_tree_doc_parallel_structure_searcher`, none of which are ported.
`UNWIRED.md` already carries the whole pointwise family; this file joins it.
Gated in isolation by `mojo_only/pointwise_scores_check.mojo`.

FOUR THINGS UPSTREAM DOES NOT AGREE WITH ITSELF ABOUT
------------------------------------------------------
All four are transcribed as written. They are recorded because a reader who
"fixes" any of them has forked the algorithm.

1. `denumSqr` is seeded `1e-20f` in `FindOptimalSplitCosineImpl` (`:344`)
   and `1e-10f` in `TCosineScoreCalcer::NextFeature`
   (`score_calcers.cuh:149`). The guard both are tested against is the same
   `> 1e-15f`, so the DYNAMIC cosine kernel can fall through to `FLT_MAX`
   on an all-empty feature while the SINGLE-FOLD one never does.

2. The `ScoreStdDev` noise enters at a different point in the two cosine
   paths. Dynamic (`:387-395`): `score *= catWeight`, THEN add noise.
   Single-fold, via `TCosineScoreCalcer::GetScore` (`score_calcers.cuh:
   160-168`) then `:271`: add noise, THEN `*= catWeight`. With a
   cat-feature weight other than 1 those are different distributions.

3. `FindOptimalSplitSolarImpl` (`:75-115`) does not compute
   `TSolarScoreCalcer`'s formula. The calcer is
   `-sum*sum * (1 + 2*log(w+1)) / w` per LEAF; the dynamic kernel is a
   held-out estimate, `-2*mu*sumTest + wTest*mu*mu` summed over fold pairs
   and then scaled by `(1 + 2*log(totalTestWeight + 1))` if that weight
   exceeds 2. Same name, different objective, and which one runs depends
   only on `foldCount == 1`.

4. The negative-weight clamp is inconsistent across the three kernels.
   Single-fold clamps the right WEIGHT and not the right SUM (`:260-263`).
   Dynamic cosine clamps the right weight on both fold halves (`:363`,
   `:369`). Dynamic solar clamps NOTHING (`:88`, `:94`). `binFeaturesWeightsCount`
   is a parameter of all three and is read by none of them.

   MEASURED 2026-08-21: the SINGLE-FOLD clamp is INERT. Every calcer
   guards its own body on the weight (`weight > 1e-20f` in Solar and L2,
   `weight > 0` in LOOL2 and SatL2, `weight > 0.0f` for `mu` in Cosine),
   and a negative weight fails all of those exactly as zero does; the one
   unguarded line, `DenumSqr += weight * mu * mu`, multiplies a `mu` that
   is already 0 on that branch. Deleting the `max` leaves every gate in
   `mojo_only/pointwise_scores_check.mojo` green with zero cells moved.
   It is kept because it is theirs. The dynamic cosine clamp on
   `weightTestRight` IS live -- its `mu` comes from a DIFFERENT fold -- and
   gate D4 moves 11 cells when it is removed.

A FIFTH, WHICH IS A REACHABILITY BUG IN THEIRS
-----------------------------------------------
`GatherHistogramByLeaves` (`:600-614`) launches `numBlocks.z =
(leafCount + blockSize - 1) / blockSize` and the kernel's only use of the z
axis is `threadIdx.z * BLOCK_SIZE` (`:571`) -- `threadIdx.z`, not
`blockIdx.z`. The launch is one-dimensional (`<<<numBlocks, blockSize>>>`
with `blockSize` an `int`), so `threadIdx.z` is 0 in every thread of every
block. At `leafCount > 1024` the extra z BLOCKS therefore recompute leaves
0..1023 and leaves 1024.. are never written. Transcribed as written,
including the dead term. It is unreachable in a default GPU fit
(`max_depth` 6, and 8 is their documented ceiling for the pointwise
searcher), and "fixing" it would be inventing.

==========================================================================
DEVIATION 94: NO FLOAT64 ANYWHERE. THEIRS ACCUMULATES IN DOUBLE.
==========================================================================
Metal has no fp64 at any cost, so every `double` in these two files is
`Float32` here. Their doubles, by citation:

  * `struct TPartitionStatistics { double Weight; double Sum; double Count; }`
    (`gpu_data/gpu_structures.h:113-116`) -- the per-partition totals every
    kernel here subtracts the left child from. Ours is a Float32 triple.
  * `void AddLeaf(double sum, double weight)` in all five calcers, and
    `double Score; double DenumSqr;` in `TCosineScoreCalcer`
    (`score_calcers.cuh:22, 53, 83, 114, 152, 182-183`).
  * `double scoreBeforeSplit`, `double l2`, `double scoreStdDev` as kernel
    ARGUMENTS (`:50`, `:325-326`).
  * `__shared__ volatile double localBuffer[BLOCK_SIZE]` and
    `Reduce<double, BLOCK_SIZE>` in `PartitionUpdateImpl` (`:637`, `:644`),
    which is where the partition totals are MADE. This one compounds: a
    Float32 partition total then feeds the Float32 subtraction above.

WHAT IT COSTS, named rather than hidden. The right child is derived as
`part.Sum - sumLeft` (`:263`, `:369`), which is the cancellation step of the
whole scorer: when one bin holds nearly all of a partition's gradient the
subtraction cancels most of the significand, and theirs cancels in double
and narrows afterwards while ours cancels in Float32 throughout. So a split
CHOICE can differ from CatBoost's whenever two candidates sit within Float32
epsilon of each other after cancellation, and the gap widens with `pCount`
and with `foldCount`.

NOT FAKED: no `Float64` appears in this file pretending to be theirs. The
one place Float64 is legitimate is a HOST oracle, and the gate uses one.

==========================================================================
DEVIATION 95: THEIR STRUCT POINTERS BECOME FLAT TYPED ARRAYS.
==========================================================================
A Mojo kernel argument cannot be a pointer to a non-trivial struct, and
PORTING_RULES 4 already records that `enqueue_function` refuses derived
pointers as aliasing. Their four struct arguments are therefore passed as
the flat arrays their C++ memory image already is:

  `const TCBinFeature* bf`      -> `MutPointer[UInt32]`, THREE words per
      entry: `[3b] = FeatureId`, `[3b+1] = BinId`, `[3b+2] =
      SkipInScoreCount`. `{ui32; ui32; bool;}` is 12 bytes with the bool in
      the low byte of the third word, so this is byte-identical, not a
      re-encoding.
  `const TPartitionStatistics* parts` -> `MutPointer[Float32]`, THREE per
      entry: Weight, Sum, Count. (Float32 by DEVIATION 94, not by this one.)
  `const TDataPartition* parts` -> `MutPointer[UInt32]`, TWO per entry:
      Offset, Size -- the same encoding `split_properties_helpers.mojo`
      already uses, and the `+ 1` for Size is load-bearing.
  `TBestSplitProperties* result` -> TWO pointers, `result_ids`
      (`MutPointer[UInt32]`, 2 per block: FeatureId, BinId) and
      `result_scores` (`MutPointer[Float32]`, 2 per block: Score, Gain).
      Their `{ui32; ui32; float; float;}` is one 16-byte record; splitting
      it by type rather than bitcasting through a UInt32 array keeps every
      store typed. `result += blockIdx.x` becomes `2 * blockIdx.x` into
      each.

Their `TScoreCalcer calcer` is also passed BY VALUE as a kernel argument
(`:250`, `:481`). A Mojo kernel cannot take a user struct by value, so the
calcer's CONFIGURATION crosses the launch boundary as scalars (`lambda_l2`,
`meta_exponent`, `normalize`, `score_std_dev`, `global_seed`) and the
calcer is constructed inside the kernel from them. Their host still makes
every decision that was host-side upstream -- in particular the
`MetaExponent` coin flip, which is `meta_exponent_draw` below and stays on
the host exactly as `:507` has it.

==========================================================================
DEVIATION 96: `StreamLoad` AND `__ldg` HAVE NO PORTABLE SPELLING.
==========================================================================
`ComputeSum` loads through `NKernel::StreamLoad` (`:27`), which is
`cub::ThreadLoad<cub::LOAD_CS>` -- a PTX `ld.global.cs`, a cache-streaming
hint that tells the L1 to evict the line early so a one-pass sum does not
displace the working set. `LdgWithFallback` is `cub::ThreadLoad<
cub::LOAD_LDG>`, the read-only data cache path. Both are NVIDIA cache-policy
PTX with no counterpart on Metal and no vendor-agnostic Mojo spelling.

  * `LdgWithFallback` / `__ldg` -> `std.gpu.intrinsics.ldg`, which is the
    Mojo spelling of the same intrinsic and lowers to a plain load where the
    target has none. Kept, because it is language-level.
  * `StreamLoad` -> a plain `unsafe_load`. THIS IS A HINT ONLY: `LOAD_CS`
    and a normal load return the same bytes, so the arithmetic is
    unchanged and only cache residency differs. Recorded rather than
    silently dropped because it is the one thing in `ComputeSum` that is
    not the sum.

The 16-wide manual unroll around it IS ported (`:23-32`), including their
comment's reason for it, because the unroll changes the ORDER of a float
summation and therefore the answer. Their `#pragma unroll` on the inner 16
is not expressible; the loop is written out as a Mojo `comptime for` over
16 iterations, which is the same unroll and the same order.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.gpu.intrinsics import ldg
from std.math import copysign

# DEVIATION 258 (row 10 sqrt on NVIDIA; row 12 log): both seam calls are
# the stdlib under FAST and the portable pair under IDENTICAL
from mojo_only.numerics import identical_log, identical_pow, identical_sqrt
from std.memory import stack_allocation

from gbdt.gpu_util.kernel.random_gen import (
    advance_seed_k,
    next_normal_f,
    next_uniform_f,
)
from gbdt.methods.kernel.split_properties_helpers import (
    PointwisePartOffsetsHelper,
)
from gbdt.options.catboost_options import (
    SCORE_FUNCTION_COSINE,
    SCORE_FUNCTION_L2,
    SCORE_FUNCTION_LOO_L2,
    SCORE_FUNCTION_NEWTON_COSINE,
    SCORE_FUNCTION_NEWTON_L2,
    SCORE_FUNCTION_SAT_L2,
    SCORE_FUNCTION_SOLAR_L2,
)


#: `const int blockSize = 128;` -- `:456`, `:479`. The score kernels' block.
comptime POINTWISE_SCORE_BLOCK = 128

#: `const int blockSize = 1024;` -- `:601`, `:685`. The gather and the
#: partition update run at 1024, which is a different constant and is not
#: interchangeable with the one above: `GatherHistogramsByLeavesImpl` divides
#: it by `leafCount` to get features per block.
comptime POINTWISE_WIDE_BLOCK = 1024

#: `FLT_MAX`. Their "this candidate is unusable" sentinel (`:44-45`,
#: `score_calcers.cuh:161`). NOT negated here -- see the sign section.
comptime FLOAT32_MAX = Float32(3.4028234663852886e38)

#: `TDirectHistLoader` (`:157-181`) / `TGatheredByLeavesHistLoader`
#: (`:184-215`). Their two `THistLoader` template arguments, as a tagged
#: union: Mojo has no dynamic trait objects and their host switches on this
#: anyway (`gatheredByLeaves`, `:547`).
comptime HIST_LOADER_DIRECT = 0
comptime HIST_LOADER_GATHERED_BY_LEAVES = 1


# ---------------------------------------------------------------------------
# `score_calcers.cuh`, all five classes
# ---------------------------------------------------------------------------


@fieldwise_init
struct ScoreCalcer[score_function: Int](Copyable, ImplicitlyCopyable, Movable):
    """The five calcers of `score_calcers.cuh`, as one comptime-tagged union.

    Their five classes share an interface -- `NextFeature(TCBinFeature)`,
    `AddLeaf(double, double)`, `GetScore()`, `Combine(other)` -- and
    `FindOptimalSplitSingleFoldImpl` is templated on which one it holds
    (`:220-222`). Mojo has no dynamic trait objects, so the template
    argument becomes a comptime `score_function` and every method is a
    `comptime if` over their five bodies, in their order.

    `Combine` is NOT ported. Its only callers are the pairwise/multiclass
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
                self.score += (
                    (-sum * sum)
                    * (Float32(1.0) + Float32(2.0) * identical_log(weight + Float32(1.0)))
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
                self.score += (
                    copysign(
                        identical_pow(abs(leaf_score) / weight, self.meta_exponent),
                        leaf_score,
                    )
                    * weight
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
                adjust_s = (
                    weight
                    * (weight - Float32(2.0))
                    / (
                        weight * weight
                        - Float32(3.0) * weight
                        + Float32(1.0)
                    )
                )
            if weight > Float32(0.0):
                self.score += adjust_s * ((-sum * sum) / weight)

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
            self.score += sum * mu
            self.denum_sqr += weight * mu * mu

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
        `NextNormal` then consumes two more. A port that skips the four, or
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
                out += draw[0] * self.score_std_dev

        return out


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


def meta_exponent_draw(
    seed: UInt64, meta_frequency: Float32, meta_l2_exponent: Float32
) -> Tuple[Float32, UInt64]:
    """`:507` -- the `MetaExponent` coin flip, on the HOST where theirs is.

        const float metaExponent = (NextUniform(&seed) >= metaFrequency
                                    ? 1.0f
                                    : static_cast<float>(metaL2Exponent));

    It happens ONCE PER CALL, not once per feature: every candidate in the
    launch shares the drawn exponent. And it ADVANCES the seed, but only
    on the `L2`/`NewtonL2` arm of their switch -- the cosine arm below it
    receives the UNADVANCED seed, because `FindOptimalSplitPlain` takes
    `ui64 seed` by value and the two arms are exclusive. Returning the
    advanced seed here keeps that visible instead of hiding it in a
    mutation.
    """
    var d = next_uniform_f(seed)
    if d[0] >= meta_frequency:
        return (Float32(1.0), d[1])
    return (meta_l2_exponent, d[1])


# ---------------------------------------------------------------------------
# `ComputeSum<BLOCK_SIZE>` (`:16-35`)
# ---------------------------------------------------------------------------


@always_inline
def _compute_sum[
    block_size: Int
](buffer: MutPointer[Float32, MutAnyOrigin], count: Int) -> Float32:
    """`ComputeSum<BLOCK_SIZE>` (`:16-35`), unroll and all.

        float sum = 0.f;
        const ui32 tid = threadIdx.x;
        ui32 i = tid;
        const int ITERS = 16;
        for (; i + (ITERS - 1) * BLOCK_SIZE < count;) {
            #pragma unroll
            for (int iter = 0; iter < ITERS; ++iter, i += BLOCK_SIZE) {
                sum += NKernel::StreamLoad(buffer + i);
            }
        }
        for (; i < count; i += BLOCK_SIZE) {
            sum += NKernel::StreamLoad(buffer + i);
        }

    Their own comment says the manual unroll is there because nvcc 11.4+
    refuses `#pragma unroll 16` on that loop shape. It is ported anyway,
    and not because of nvcc: the unroll fixes the ORDER in which one
    thread's strided elements are added, and float addition is not
    associative. Collapsing it into the tail loop alone would give a
    different sum on the same data.

    `StreamLoad` -> plain load, DEVIATION 96. Cache policy only.

    NOTE THE ENTRY CONDITION, `i + 15 * BLOCK_SIZE < count`. Below
    `15 * BLOCK_SIZE + 1` elements the unrolled loop never runs at all --
    at their `blockSize` of 1024 that is 15361 rows, so a partition
    smaller than that takes the tail loop exclusively. Both arms are
    reachable and the gate runs both.
    """
    var sum = Float32(0.0)
    var i = Int(thread_idx.x)
    while i + 15 * block_size < count:

        comptime for _iter in range(16):
            sum += buffer.unsafe_load(i)
            i += block_size

    while i < count:
        sum += buffer.unsafe_load(i)
        i += block_size
    return sum


# ---------------------------------------------------------------------------
# `FindOptimalSplitSolarImpl<BLOCK_SIZE>` (`:41-153`)
# ---------------------------------------------------------------------------


def find_optimal_split_solar_kernel[
    block_size: Int
](
    bf: MutPointer[UInt32, MutAnyOrigin],
    bin_feature_count_in: Int32,
    cat_features_weights: MutPointer[Float32, MutAnyOrigin],
    bin_features_weights: MutPointer[Float32, MutAnyOrigin],
    bin_features_weights_count_in: Int32,
    bin_sums: MutPointer[Float32, MutAnyOrigin],
    parts: MutPointer[Float32, MutAnyOrigin],
    p_count_in: Int32,
    fold_count_in: Int32,
    score_before: MutPointer[Float32, MutAnyOrigin],
    result_ids: MutPointer[UInt32, MutAnyOrigin],
    result_scores: MutPointer[Float32, MutAnyOrigin],
):
    """`FindOptimalSplitSolarImpl<BLOCK_SIZE>` (`:41-153`).

    THE ORDERED-BOOSTING SCORER. It runs only when `foldCount > 1`
    (`:546-556`), which is CatBoost's Ordered mode, and it is not
    `TSolarScoreCalcer` -- see finding 3 in the module docstring.

    The fold loop steps by TWO and the pair is (estimate, test):
    `fold` is the fold whose histogram ESTIMATES the leaf value `mu`, and
    `fold + 1` is the held-out fold the estimate is SCORED on (`:81-104`).
    That is the whole point of ordered boosting expressed as an index: a
    document's target never contributes to the estimate it is judged by.

    `weightEstimateRight = partLearn.Weight - weightEstimateLeft` with NO
    clamp (`:88`), unlike the other two kernels (finding 4).

    `score += leftTotalWeight > 2 ? leftScore * (1 + 2*log(leftTotalWeight+1))
    : 0` (`:113-114`) -- the `> 2` is a hard cut, not a smooth guard, so a
    leaf whose held-out weight is 2 or less contributes exactly nothing.
    """
    var bin_feature_count = Int(bin_feature_count_in)
    var p_count = Int(p_count_in)
    var fold_count = Int(fold_count_in)
    # DEVIATION 207: theirs is a host scalar; the blind level loop keeps
    # it on the device. Same word, uniform load.
    var score_before_split = score_before.unsafe_load(0)

    var best_score = FLOAT32_MAX
    var best_gain = FLOAT32_MAX
    var best_index = 0
    var tid = Int(thread_idx.x)

    var helper = PointwisePartOffsetsHelper(UInt32(fold_count))

    var i = Int(block_idx.x) * block_size
    while i < bin_feature_count:
        if i + tid >= bin_feature_count:
            break
        var b = i + tid
        # `bf[i + tid].SkipInScoreCount` (`:65`)
        if bf.unsafe_load(3 * b + 2) != UInt32(0):
            i += block_size * Int(grid_dim.x)
            continue

        # `const float* current = binSums + 2 * (i + tid);` (`:69`)
        var current = 2 * b

        var score = Float32(0.0)

        for leaf in range(p_count):
            var left_total_weight = Float32(0.0)
            var right_total_weight = Float32(0.0)
            var left_score = Float32(0.0)
            var right_score = Float32(0.0)

            var fold = 0
            while fold < fold_count:
                var learn_off = Int(
                    helper.data_partition_offset(UInt32(leaf), UInt32(fold))
                )
                var test_off = Int(
                    helper.data_partition_offset(UInt32(leaf), UInt32(fold + 1))
                )
                var part_learn_weight = ldg(parts.unsafe_offset(3 * learn_off + 0))
                var part_learn_sum = ldg(parts.unsafe_offset(3 * learn_off + 1))
                var part_test_weight = ldg(parts.unsafe_offset(3 * test_off + 0))
                var part_test_sum = ldg(parts.unsafe_offset(3 * test_off + 1))

                var h_learn = (
                    bin_feature_count
                    * Int(
                        helper.histogram_offset(UInt32(leaf), UInt32(fold))
                    )
                    * 2
                )
                var h_test = (
                    bin_feature_count
                    * Int(
                        helper.histogram_offset(UInt32(leaf), UInt32(fold + 1))
                    )
                    * 2
                )

                var weight_estimate_left = bin_sums.unsafe_load(
                    current + h_learn
                )
                var weight_estimate_right = (
                    part_learn_weight - weight_estimate_left
                )

                var sum_estimate_left = bin_sums.unsafe_load(
                    current + h_learn + 1
                )
                var sum_estimate_right = part_learn_sum - sum_estimate_left

                var weight_test_left = bin_sums.unsafe_load(current + h_test)
                var weight_test_right = part_test_weight - weight_test_left

                var sum_test_left = bin_sums.unsafe_load(current + h_test + 1)
                var sum_test_right = part_test_sum - sum_test_left

                # `:105-110`
                var mu_l = Float32(0.0)
                if weight_estimate_left > Float32(0.0):
                    mu_l = sum_estimate_left / (
                        weight_estimate_left + Float32(1e-15)
                    )
                left_score += (
                    Float32(-2.0) * mu_l * sum_test_left
                    + weight_test_left * mu_l * mu_l
                )
                left_total_weight += weight_test_left

                var mu_r = Float32(0.0)
                if weight_estimate_right > Float32(0.0):
                    mu_r = sum_estimate_right / (
                        weight_estimate_right + Float32(1e-15)
                    )
                right_total_weight += weight_test_right
                right_score += (
                    Float32(-2.0) * mu_r * sum_test_right
                    + weight_test_right * mu_r * mu_r
                )

                fold += 2

            if left_total_weight > Float32(2.0):
                score += left_score * (
                    Float32(1.0)
                    + Float32(2.0) * identical_log(left_total_weight + Float32(1.0))
                )
            if right_total_weight > Float32(2.0):
                score += right_score * (
                    Float32(1.0)
                    + Float32(2.0) * identical_log(right_total_weight + Float32(1.0))
                )

        # `:117-120`
        var feature_id = Int(bf.unsafe_load(3 * b))
        score *= ldg(cat_features_weights.unsafe_offset(feature_id))
        var gain = (score - score_before_split) * ldg(
            bin_features_weights.unsafe_offset(feature_id)
        )

        if gain < best_gain:
            best_score = score
            best_gain = gain
            best_index = b

        i += block_size * Int(grid_dim.x)

    _block_argmin_and_store[block_size](
        tid, best_score, best_gain, best_index, bf, bin_feature_count,
        result_ids, result_scores,
    )


# ---------------------------------------------------------------------------
# `FindOptimalSplitSingleFoldImpl<BLOCK_SIZE, THistLoader, TScoreCalcer>`
# (`:217-308`)
# ---------------------------------------------------------------------------


def find_optimal_split_single_fold_kernel[
    block_size: Int, hist_loader: Int, score_function: Int
](
    bf: MutPointer[UInt32, MutAnyOrigin],
    bin_feature_count_in: Int32,
    cat_features_weights: MutPointer[Float32, MutAnyOrigin],
    bin_features_weights: MutPointer[Float32, MutAnyOrigin],
    bin_features_weights_count_in: Int32,
    score_before: MutPointer[Float32, MutAnyOrigin],
    bin_sums: MutPointer[Float32, MutAnyOrigin],
    parts: MutPointer[Float32, MutAnyOrigin],
    p_count_in: Int32,
    lambda_l2: Float32,
    meta_exponent: Float32,
    normalize_in: Int32,
    score_std_dev: Float32,
    global_seed: UInt64,
    result_ids: MutPointer[UInt32, MutAnyOrigin],
    result_scores: MutPointer[Float32, MutAnyOrigin],
):
    """`FindOptimalSplitSingleFoldImpl` (`:217-308`), the PLAIN-boosting path.

    `TPointwisePartOffsetsHelper helper(1)` (`:230`): at one fold both
    offsets collapse to `partId`, which is why this kernel can hoist the
    fold loop out entirely and let a `THistLoader` own the addressing.

    THE TWO LOADERS ARE DIFFERENT LAYOUTS, not different index arithmetic
    on one layout:

      `TDirectHistLoader` (`:157-181`) reads the histogram as the
      `pointwise_hist2*` kernels WROTE it -- bin-feature major inside a
      partition, `BinSums[binFeatureCount * leaf * 2 + 2*b + {0,1}]`.

      `TGatheredByLeavesHistLoader` (`:184-215`) reads the TRANSPOSED copy
      that `GatherHistogramByLeaves` makes -- leaf-minor,
      `BinSums[2 * (b * leafCount + leaf) + {0,1}]`. Note their ctor names
      the argument `binFeatureId` and stores it in a member called
      `FeatureId`: it is the BIN-feature index, not the feature index, and
      reading it as the latter silently collapses every bin of a feature
      onto one row.

    The gather exists because at large `pCount` the direct layout strides
    `binFeatureCount * 2` floats per leaf, so one thread's `pCount` loads
    touch `pCount` different cache lines; gathered, they are contiguous.
    Which one runs is `gatheredByLeaves` at `:547`, a host switch, and both
    sides are gated.

    `weightRight = max(part.Weight - weightLeft, 0.0f)` but
    `sumRight = part.Sum - sumLeft` with no clamp (`:260-263`) -- finding 4.

    `score_before` is a DEVICE pointer where theirs is a host scalar
    (DEVIATION 207): the blind level loop keeps last level's winning score
    on the device, so the kernel loads it instead of receiving it. Every
    thread loads the same word; the value and the arithmetic are unchanged.
    """
    var bin_feature_count = Int(bin_feature_count_in)
    var p_count = Int(p_count_in)
    var score_before_split = score_before.unsafe_load(0)

    var best_score = FLOAT32_MAX
    var best_gain = FLOAT32_MAX
    var best_index = 0
    var tid = Int(thread_idx.x)

    var helper = PointwisePartOffsetsHelper(UInt32(1))

    var calcer = make_score_calcer[score_function](
        lambda_l2,
        meta_exponent,
        normalize_in != Int32(0),
        score_std_dev,
        global_seed,
    )

    var i = Int(block_idx.x) * block_size
    while i < bin_feature_count:
        if i + tid >= bin_feature_count:
            break
        var b = i + tid
        if bf.unsafe_load(3 * b + 2) != UInt32(0):
            i += block_size * Int(grid_dim.x)
            continue

        # `calcer.NextFeature(bf[i + tid])` (`:238`)
        calcer.next_feature(bf.unsafe_load(3 * b))

        for leaf in range(p_count):
            var part_off = Int(
                helper.data_partition_offset(UInt32(leaf), UInt32(0))
            )
            var part_weight = ldg(parts.unsafe_offset(3 * part_off + 0))
            var part_sum = ldg(parts.unsafe_offset(3 * part_off + 1))

            var weight_left = Float32(0.0)
            var sum_left = Float32(0.0)

            comptime if hist_loader == HIST_LOADER_DIRECT:
                # `TDirectHistLoader::LoadWeight/LoadSum` (`:170-177`)
                var base = (
                    2 * b
                    + bin_feature_count
                    * Int(helper.histogram_offset(UInt32(leaf), UInt32(0)))
                    * 2
                )
                weight_left = bin_sums.unsafe_load(base)
                sum_left = bin_sums.unsafe_load(base + 1)
            else:
                # `TGatheredByLeavesHistLoader::GetOffset` (`:198-200`)
                var off = 2 * (b * p_count + leaf)
                weight_left = bin_sums.unsafe_load(off)
                sum_left = bin_sums.unsafe_load(off + 1)

            var weight_right = max(part_weight - weight_left, Float32(0.0))
            var sum_right = part_sum - sum_left

            calcer.add_leaf(sum_left, weight_left)
            calcer.add_leaf(sum_right, weight_right)

        var score = calcer.get_score()

        var feature_id = Int(bf.unsafe_load(3 * b))
        score *= ldg(cat_features_weights.unsafe_offset(feature_id))
        var gain = score - score_before_split
        gain *= ldg(bin_features_weights.unsafe_offset(feature_id))

        if gain < best_gain:
            best_score = score
            best_gain = gain
            best_index = b

        i += block_size * Int(grid_dim.x)

    _block_argmin_and_store[block_size](
        tid, best_score, best_gain, best_index, bf, bin_feature_count,
        result_ids, result_scores,
    )


# ---------------------------------------------------------------------------
# `FindOptimalSplitCosineImpl<BLOCK_SIZE>` (`:314-437`)
# ---------------------------------------------------------------------------


def find_optimal_split_cosine_kernel[
    block_size: Int
](
    bf: MutPointer[UInt32, MutAnyOrigin],
    bin_feature_count_in: Int32,
    cat_features_weights: MutPointer[Float32, MutAnyOrigin],
    bin_features_weights: MutPointer[Float32, MutAnyOrigin],
    bin_features_weights_count_in: Int32,
    bin_sums: MutPointer[Float32, MutAnyOrigin],
    parts: MutPointer[Float32, MutAnyOrigin],
    p_count_in: Int32,
    fold_count_in: Int32,
    score_before: MutPointer[Float32, MutAnyOrigin],
    l2: Float32,
    normalize_in: Int32,
    score_std_dev: Float32,
    global_seed: UInt64,
    result_ids: MutPointer[UInt32, MutAnyOrigin],
    result_scores: MutPointer[Float32, MutAnyOrigin],
):
    """`FindOptimalSplitCosineImpl<BLOCK_SIZE>` (`:314-437`).

    The ordered-boosting cosine scorer, and CatBoost's DEFAULT score
    function (`oblivious_tree_options.cpp:22`) on the `foldCount > 1` side
    of the dispatch.

    Same (estimate, test) fold pairing as the solar kernel. What differs
    from `TCosineScoreCalcer` (finding 1 and 2 of the module docstring):

      * `denumSqr` starts at 1e-20f (`:344`), not 1e-10f, so an
        all-empty candidate DOES fall through to `FLT_MAX` here;
      * the noise is added AFTER `score *= catFeaturesWeights` (`:387-395`),
        where the calcer adds it before;
      * both `weightEstimateRight` and `weightTestRight` are clamped at 0
        (`:363`, `:369`) but neither sum is.

    `normalize` is a runtime kernel argument here, exactly as it is theirs
    (`:324`), rather than a comptime switch -- their host does not
    specialize on it and neither does this.
    """
    var bin_feature_count = Int(bin_feature_count_in)
    var p_count = Int(p_count_in)
    var fold_count = Int(fold_count_in)
    var normalize = normalize_in != Int32(0)
    # DEVIATION 207: theirs is a host scalar; the blind level loop keeps
    # it on the device. Same word, uniform load.
    var score_before_split = score_before.unsafe_load(0)

    var best_score = FLOAT32_MAX
    var best_gain = FLOAT32_MAX
    var best_index = 0
    var tid = Int(thread_idx.x)

    var helper = PointwisePartOffsetsHelper(UInt32(fold_count))

    var i = Int(block_idx.x) * block_size
    while i < bin_feature_count:
        if i + tid >= bin_feature_count:
            break
        var b = i + tid
        if bf.unsafe_load(3 * b + 2) != UInt32(0):
            i += block_size * Int(grid_dim.x)
            continue

        var score = Float32(0.0)
        var denum_sqr = Float32(1e-20)
        var current = 2 * b

        for leaf in range(p_count):
            var fold = 0
            while fold < fold_count:
                var learn_off = Int(
                    helper.data_partition_offset(UInt32(leaf), UInt32(fold))
                )
                var test_off = Int(
                    helper.data_partition_offset(UInt32(leaf), UInt32(fold + 1))
                )
                var part_learn_weight = ldg(parts.unsafe_offset(3 * learn_off + 0))
                var part_learn_sum = ldg(parts.unsafe_offset(3 * learn_off + 1))
                var part_test_weight = ldg(parts.unsafe_offset(3 * test_off + 0))
                var part_test_sum = ldg(parts.unsafe_offset(3 * test_off + 1))

                var h_learn = (
                    bin_feature_count
                    * Int(helper.histogram_offset(UInt32(leaf), UInt32(fold)))
                    * 2
                )
                var h_test = (
                    bin_feature_count
                    * Int(
                        helper.histogram_offset(UInt32(leaf), UInt32(fold + 1))
                    )
                    * 2
                )

                var weight_estimate_left = bin_sums.unsafe_load(
                    current + h_learn
                )
                var weight_estimate_right = max(
                    part_learn_weight - weight_estimate_left, Float32(0.0)
                )

                var sum_estimate_left = bin_sums.unsafe_load(
                    current + h_learn + 1
                )
                var sum_estimate_right = part_learn_sum - sum_estimate_left

                var weight_test_left = bin_sums.unsafe_load(current + h_test)
                var weight_test_right = max(
                    part_test_weight - weight_test_left, Float32(0.0)
                )

                var sum_test_left = bin_sums.unsafe_load(current + h_test + 1)
                var sum_test_right = part_test_sum - sum_test_left

                var lam_l = l2
                if normalize:
                    lam_l = l2 * weight_estimate_left
                var mu_l = Float32(0.0)
                if weight_estimate_left > Float32(0.0):
                    mu_l = sum_estimate_left / (weight_estimate_left + lam_l)
                score += sum_test_left * mu_l
                denum_sqr += weight_test_left * mu_l * mu_l

                var lam_r = l2
                if normalize:
                    lam_r = l2 * weight_estimate_right
                var mu_r = Float32(0.0)
                if weight_estimate_right > Float32(0.0):
                    mu_r = sum_estimate_right / (weight_estimate_right + lam_r)
                score += sum_test_right * mu_r
                denum_sqr += weight_test_right * mu_r * mu_r

                fold += 2

        # `:381`
        if denum_sqr > Float32(1e-15):
            score = -score / identical_sqrt(denum_sqr)
        else:
            score = FLOAT32_MAX

        var feature_id = Int(bf.unsafe_load(3 * b))
        score *= ldg(cat_features_weights.unsafe_offset(feature_id))

        var noisy_score = score
        if score_std_dev != Float32(0.0):
            var seed = advance_seed_k(
                global_seed + UInt64(UInt32(feature_id)), 4
            )
            var draw = next_normal_f(seed)
            noisy_score += draw[0] * score_std_dev

        var gain = (noisy_score - score_before_split) * ldg(
            bin_features_weights.unsafe_offset(feature_id)
        )
        if gain < best_gain:
            best_score = noisy_score
            best_gain = gain
            best_index = b

        i += block_size * Int(grid_dim.x)

    _block_argmin_and_store[block_size](
        tid, best_score, best_gain, best_index, bf, bin_feature_count,
        result_ids, result_scores,
    )


# ---------------------------------------------------------------------------
# the block argmin, which all three kernels end with, character for character
# ---------------------------------------------------------------------------


@always_inline
def _block_argmin_and_store[
    block_size: Int
](
    tid: Int,
    best_score: Float32,
    best_gain: Float32,
    best_index: Int,
    bf: MutPointer[UInt32, MutAnyOrigin],
    bin_feature_count: Int,
    result_ids: MutPointer[UInt32, MutAnyOrigin],
    result_scores: MutPointer[Float32, MutAnyOrigin],
):
    """`:135-152` == `:282-299` == `:409-426`, plus their writeback.

    Their three kernels end with the same twenty lines pasted three times.
    One copy here; the arithmetic is theirs, not a generalization of it.

        __shared__ float scores[BLOCK_SIZE];  scores[tid] = bestScore;
        __shared__ int   indices[BLOCK_SIZE]; indices[tid] = bestIndex;
        __shared__ float gains[BLOCK_SIZE];   gains[tid] = bestGain;
        __syncthreads();
        for (ui32 s = BLOCK_SIZE >> 1; s > 0; s >>= 1) {
            if (tid < s) {
                if (gains[tid] > gains[tid + s] ||
                    (gains[tid] == gains[tid + s] &&
                     indices[tid] > indices[tid + s])) {
                    scores[tid]  = scores[tid + s];
                    indices[tid] = indices[tid + s];
                    gains[tid]   = gains[tid + s];
                }
            }
            __syncthreads();
        }
        if (!tid) {
            const int index = indices[0];
            result->FeatureId = index < binFeatureCount ? bf[index].FeatureId : 0;
            result->BinId     = index < binFeatureCount ? bf[index].BinId : 0;
            result->Score     = scores[0];
            result->Gain      = gains[0];
        }

    THE BARRIER IS OUTSIDE THE `if (tid < s)`, which is what makes it legal
    and what makes it portable (`PORTING.md` 11 and 92: a threadgroup
    barrier reached by only some threads is undefined, and on Metal it does
    not merely warn). Their `ScanHistogramsImpl` in the same family gets
    this wrong; this loop gets it right, so it transliterates unchanged.

    ON A TIE THE SMALLER INDEX WINS. See the module docstring.

    THE `index < binFeatureCount` GUARD IS NOT DEAD in their code even
    though `bestIndex` starts at 0: it is how a launch with
    `binFeatureCount == 0` avoids reading `bf[0]`. Ported.
    """
    var s_scores = stack_allocation[
        block_size,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var s_indices = stack_allocation[
        block_size,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var s_gains = stack_allocation[
        block_size,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()

    s_scores[unsafe_offset=tid] = best_score
    s_indices[unsafe_offset=tid] = Int32(best_index)
    s_gains[unsafe_offset=tid] = best_gain
    barrier()

    var s = block_size >> 1
    while s > 0:
        if tid < s:
            var take = s_gains[unsafe_offset=tid] > s_gains[unsafe_offset=tid + s]
            if s_gains[unsafe_offset=tid] == s_gains[unsafe_offset=tid + s]:
                if s_indices[unsafe_offset=tid] > s_indices[unsafe_offset=tid + s]:
                    take = True
            if take:
                s_scores[unsafe_offset=tid] = s_scores[unsafe_offset=tid + s]
                s_indices[unsafe_offset=tid] = s_indices[unsafe_offset=tid + s]
                s_gains[unsafe_offset=tid] = s_gains[unsafe_offset=tid + s]
        barrier()
        s >>= 1

    if tid == 0:
        var index = Int(s_indices[unsafe_offset=0])
        var out = 2 * Int(block_idx.x)
        if index < bin_feature_count:
            result_ids.unsafe_store(out, bf.unsafe_load(3 * index))
            result_ids.unsafe_store(out + 1, bf.unsafe_load(3 * index + 1))
        else:
            result_ids.unsafe_store(out, UInt32(0))
            result_ids.unsafe_store(out + 1, UInt32(0))
        result_scores.unsafe_store(out, s_scores[unsafe_offset=0])
        result_scores.unsafe_store(out + 1, s_gains[unsafe_offset=0])


# ---------------------------------------------------------------------------
# `GatherHistogramsByLeavesImpl<BLOCK_SIZE, HIST_COUNT>` (`:551-580`)
# ---------------------------------------------------------------------------


def gather_histograms_by_leaves_kernel[
    block_size: Int, hist_count: Int
](
    bin_feature_count_in: Int32,
    histogram: MutPointer[Float32, MutAnyOrigin],
    hist_count_in: Int32,
    leaf_count_in: Int32,
    fold_count_in: Int32,
    result: MutPointer[Float32, MutAnyOrigin],
):
    """`GatherHistogramsByLeavesImpl<BLOCK_SIZE, HIST_COUNT>` (`:551-580`).

    The transpose that `TGatheredByLeavesHistLoader` reads. In:

        histogram[(featureId + binFeatureCount *
                   helper.GetHistogramOffset(leafId, foldId)) * HIST_COUNT
                  + histId]

    out:

        result[((featureId * leafCount + leafId) * foldCount + foldId)
               * HIST_COUNT + histId]

    So the leaf axis moves from OUTERMOST to second-innermost and the fold
    axis lands between leaf and stat. At `foldCount == 1` and
    `HIST_COUNT == 2` the output index is `2 * (featureId * leafCount +
    leafId) + histId`, which is exactly `TGatheredByLeavesHistLoader::
    GetOffset` -- the two files agree, and a gate that runs only at
    `foldCount == 1` cannot see the fold axis at all.

    `leafCount` MUST BE A POWER OF TWO: `threadIdx.x & (leafCount - 1)`
    (`:557`). Theirs is oblivious, so the leaves of a level always are.

    `hist_count_in` is their `histCount` parameter, which the kernel body
    never reads -- `HIST_COUNT` is the template argument and the runtime one
    is passed and ignored (`:554`, `:619`). Kept in the signature.

    Their `float leafVals[HIST_COUNT]` load-all-then-store-all is kept: it
    is a register array, and merging the two loops would change the memory
    schedule they chose.

    THE z AXIS IS DEAD -- see the module docstring's fifth finding. The
    `threadIdx.z * BLOCK_SIZE` term is transcribed and is always 0.
    """
    var bin_feature_count = Int(bin_feature_count_in)
    var leaf_count = Int(leaf_count_in)
    var fold_count = Int(fold_count_in)

    var features_per_block = (block_size + leaf_count - 1) // leaf_count
    var feature_id = (
        Int(block_idx.x) * features_per_block
        + Int(thread_idx.x) // leaf_count
    )
    var leaf_id = (Int(thread_idx.x) & (leaf_count - 1)) + Int(
        thread_idx.z
    ) * block_size

    var fold_id = Int(block_idx.y)
    var helper = PointwisePartOffsetsHelper(UInt32(grid_dim.y))

    if feature_id < bin_feature_count:
        var leaf_vals = InlineArray[Float32, hist_count](fill=Float32(0.0))

        comptime for hist_id in range(hist_count):
            var src = (
                feature_id
                + bin_feature_count
                * Int(
                    helper.histogram_offset(UInt32(leaf_id), UInt32(fold_id))
                )
            ) * hist_count + hist_id
            leaf_vals[hist_id] = ldg(histogram.unsafe_offset(src))

        comptime for hist_id2 in range(hist_count):
            var idx = (
                feature_id * leaf_count * fold_count
                + leaf_id * fold_count
                + fold_id
            ) * hist_count + hist_id2
            result.unsafe_store(idx, leaf_vals[hist_id2])


# ---------------------------------------------------------------------------
# `PartitionUpdateImpl<BLOCK_SIZE>` (`:623-679`)
# ---------------------------------------------------------------------------


def partition_update_kernel[
    block_size: Int
](
    target: MutPointer[Float32, MutAnyOrigin],
    weights: MutPointer[Float32, MutAnyOrigin],
    counts: MutPointer[Float32, MutAnyOrigin],
    have_target_in: Int32,
    have_weights_in: Int32,
    have_counts_in: Int32,
    parts: MutPointer[UInt32, MutAnyOrigin],
    part_stats: MutPointer[Float32, MutAnyOrigin],
):
    """`PartitionUpdateImpl<BLOCK_SIZE>` (`:623-679`).

    ONE BLOCK PER PARTITION. `parts += blockIdx.x; partStats += blockIdx.x;`
    then three block-wide sums over `[parts->Offset, parts->Offset + size)`,
    writing `Weight`, `Sum`, `Count`.

    THREE NULL CHECKS, and they are not symmetric (`:640`, `:658`, `:670`):

        weights == 0  ->  Weight = 0
        target  == 0  ->  Sum    = 0
        counts  == 0  ->  Count  = SIZE, not 0

    The third is the one that matters: with no per-document count column a
    partition's Count is its ROW COUNT, which is what every downstream
    leaf-value divisor expects. Ported as three explicit `have_*` flags
    because a Mojo kernel argument cannot be a null pointer -- passing a
    dummy buffer and a flag is the same test, spelled where the launcher
    can see it.

    `tmp = 0; __syncthreads();` between the three reductions (`:653-655`,
    `:665-667`) is theirs and is REQUIRED: all three sums share one shared
    buffer, so the barrier is what stops the second `localBuffer[tid] =`
    from racing the first `Reduce`'s reads. Both barriers are reached by
    every thread.

    `Reduce<double, BLOCK_SIZE>` (`kernel_helpers.cuh:123-137`) is inlined
    here rather than called: it is a plain tree reduce with the barrier
    outside the `if`, and it ends with `T result = data[0]; __syncthreads();
    return result;` -- that TRAILING barrier is also required, for the same
    reason. Float32 by DEVIATION 94.
    """
    var part = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var offset = Int(parts.unsafe_load(2 * part))
    var size = Int(parts.unsafe_load(2 * part + 1))

    var buf = stack_allocation[
        block_size,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()

    var tmp = Float32(0.0)

    if have_weights_in != Int32(0):
        buf[unsafe_offset=tid] = _compute_sum[block_size](weights.unsafe_offset(offset), size)
        barrier()
        tmp = _block_reduce_sum[block_size](buf, tid)

    if tid == 0:
        part_stats.unsafe_store(3 * part + 0, tmp)
    tmp = Float32(0.0)
    barrier()

    if have_target_in != Int32(0):
        buf[unsafe_offset=tid] = _compute_sum[block_size](target.unsafe_offset(offset), size)
        barrier()
        tmp = _block_reduce_sum[block_size](buf, tid)

    if tid == 0:
        part_stats.unsafe_store(3 * part + 1, tmp)

    tmp = Float32(0.0)
    barrier()

    if have_counts_in != Int32(0):
        buf[unsafe_offset=tid] = _compute_sum[block_size](counts.unsafe_offset(offset), size)
        barrier()
        tmp = _block_reduce_sum[block_size](buf, tid)
    else:
        tmp = Float32(size)

    if tid == 0:
        part_stats.unsafe_store(3 * part + 2, tmp)


@always_inline
def _block_reduce_sum[
    origin: MutOrigin, //, block_size: Int
](
    data: MutPointer[
        Scalar[DType.float32], origin, address_space = AddressSpace.SHARED
    ],
    tid: Int,
) -> Float32:
    """`Reduce<T, BLOCK_SIZE>` (`kernel_helpers.cuh:123-137`), Float32.

        for (int s = BLOCK_SIZE >> 1; s > 0; s >>= 1) {
            if (x < s) { data[x] += data[x + s]; }
            __syncthreads();
        }
        T result = data[0];
        __syncthreads();
        return result;

    Every thread returns `data[0]`, not just thread 0. Both barriers are
    outside the `if`.
    """
    var s = block_size >> 1
    while s > 0:
        if tid < s:
            data[unsafe_offset=tid] = data[unsafe_offset=tid] + data[unsafe_offset=tid + s]
        barrier()
        s >>= 1
    var result = data[unsafe_offset=0]
    barrier()
    return result


# ---------------------------------------------------------------------------
# the host side: `FindOptimalSplitDynamic`, `FindOptimalSplitPlain`,
# `FindOptimalSplit`, `GatherHistogramByLeaves`, `UpdatePartitionProps`
# ---------------------------------------------------------------------------


def find_optimal_split_dynamic(
    ctx: DeviceContext,
    mut binary_features: DeviceBuffer[DType.uint32],
    binary_feature_count: Int,
    mut cat_features_weights: DeviceBuffer[DType.float32],
    mut bin_features_weights: DeviceBuffer[DType.float32],
    binary_feature_weights_count: Int,
    mut splits: DeviceBuffer[DType.float32],
    mut parts: DeviceBuffer[DType.float32],
    p_count: Int,
    fold_count: Int,
    mut score_before: DeviceBuffer[DType.float32],
    mut result_ids: DeviceBuffer[DType.uint32],
    mut result_scores: DeviceBuffer[DType.float32],
    result_size: Int,
    score_function: Int,
    l2: Float32,
    normalize: Bool,
    score_std_dev: Float32,
    seed: UInt64,
) raises:
    """`FindOptimalSplitDynamic` (`:443-473`).

    Their switch, their two arms, their `default: throw std::exception()`:

        SolarL2                  -> FindOptimalSplitSolarImpl
        Cosine, NewtonCosine     -> FindOptimalSplitCosineImpl
        anything else            -> throw

    SO THE ORDERED PATH SUPPORTS THREE OF THE SEVEN SCORE FUNCTIONS AND
    NOT THE OTHER FOUR. `L2`, `NewtonL2`, `SatL2` and `LOOL2` reach a
    `throw` the moment `foldCount > 1`. That is upstream's own limit on
    ordered boosting and it is copied, not widened.

    `<<< resultSize, blockSize >>>`: `resultSize` BLOCKS, each producing one
    `TBestSplitProperties`. The host reduces the `resultSize` records
    afterwards (in `pointwise_scores_calcer.h`, not ported yet).
    """
    if score_function == SCORE_FUNCTION_SOLAR_L2:
        ctx.enqueue_function[
            find_optimal_split_solar_kernel[POINTWISE_SCORE_BLOCK]
        ](
            binary_features.unsafe_ptr(),
            Int32(binary_feature_count),
            cat_features_weights.unsafe_ptr(),
            bin_features_weights.unsafe_ptr(),
            Int32(binary_feature_weights_count),
            splits.unsafe_ptr(),
            parts.unsafe_ptr(),
            Int32(p_count),
            Int32(fold_count),
            score_before.unsafe_ptr(),
            result_ids.unsafe_ptr(),
            result_scores.unsafe_ptr(),
            grid_dim=(result_size, 1, 1),
            block_dim=(POINTWISE_SCORE_BLOCK, 1, 1),
        )
        return

    if (
        score_function == SCORE_FUNCTION_COSINE
        or score_function == SCORE_FUNCTION_NEWTON_COSINE
    ):
        ctx.enqueue_function[
            find_optimal_split_cosine_kernel[POINTWISE_SCORE_BLOCK]
        ](
            binary_features.unsafe_ptr(),
            Int32(binary_feature_count),
            cat_features_weights.unsafe_ptr(),
            bin_features_weights.unsafe_ptr(),
            Int32(binary_feature_weights_count),
            splits.unsafe_ptr(),
            parts.unsafe_ptr(),
            Int32(p_count),
            Int32(fold_count),
            score_before.unsafe_ptr(),
            l2,
            Int32(1) if normalize else Int32(0),
            score_std_dev,
            seed,
            result_ids.unsafe_ptr(),
            result_scores.unsafe_ptr(),
            grid_dim=(result_size, 1, 1),
            block_dim=(POINTWISE_SCORE_BLOCK, 1, 1),
        )
        return

    raise Error(
        "FindOptimalSplitDynamic: score function has no ordered-boosting"
        " kernel upstream; `pointwise_scores.cu:469` throws for everything"
        " but SolarL2, Cosine and NewtonCosine"
    )


def find_optimal_split_plain[
    hist_loader: Int
](
    ctx: DeviceContext,
    mut binary_features: DeviceBuffer[DType.uint32],
    binary_feature_count: Int,
    mut cat_features_weights: DeviceBuffer[DType.float32],
    mut bin_features_weights: DeviceBuffer[DType.float32],
    binary_feature_weights_count: Int,
    mut splits: DeviceBuffer[DType.float32],
    mut parts: DeviceBuffer[DType.float32],
    p_count: Int,
    mut score_before: DeviceBuffer[DType.float32],
    mut result_ids: DeviceBuffer[DType.uint32],
    mut result_scores: DeviceBuffer[DType.float32],
    result_size: Int,
    score_function: Int,
    l2: Float32,
    meta_l2_exponent: Float32,
    meta_frequency: Float32,
    normalize: Bool,
    score_std_dev: Float32,
    seed: UInt64,
) raises:
    """`FindOptimalSplitPlain<TLoader>` (`:475-527`), their five-arm switch.

        SolarL2              -> TSolarScoreCalcer(l2)
        SatL2                -> TSatL2ScoreCalcer(l2)
        LOOL2                -> TLOOL2ScoreCalcer(l2)
        L2, NewtonL2         -> TL2ScoreCalcer(l2, metaExponent)   <- draws
        Cosine, NewtonCosine -> TCosineScoreCalcer(l2, normalize,
                                                   scoreStdDev, seed)
        default              -> throw

    ONLY THE L2 ARM DRAWS. `metaExponent` is computed inside that case
    (`:507`) and the seed it advances is a by-value copy, so the cosine arm
    below receives the ORIGINAL seed. `meta_exponent_draw` keeps that
    explicit.

    `TSolarScoreCalcer`, `TSatL2ScoreCalcer` and `TLOOL2ScoreCalcer` all
    take `l2` and two of the three throw it away; that is theirs.
    """
    var meta_exponent = Float32(1.0)
    var calcer_seed = seed

    if (
        score_function == SCORE_FUNCTION_L2
        or score_function == SCORE_FUNCTION_NEWTON_L2
    ):
        var drawn = meta_exponent_draw(seed, meta_frequency, meta_l2_exponent)
        meta_exponent = drawn[0]

    if score_function == SCORE_FUNCTION_SOLAR_L2:
        _launch_single_fold[hist_loader, SCORE_FUNCTION_SOLAR_L2](
            ctx, binary_features, binary_feature_count, cat_features_weights,
            bin_features_weights, binary_feature_weights_count, splits, parts,
            p_count, score_before, l2, meta_exponent, normalize,
            score_std_dev, calcer_seed, result_ids, result_scores, result_size,
        )
    elif score_function == SCORE_FUNCTION_SAT_L2:
        _launch_single_fold[hist_loader, SCORE_FUNCTION_SAT_L2](
            ctx, binary_features, binary_feature_count, cat_features_weights,
            bin_features_weights, binary_feature_weights_count, splits, parts,
            p_count, score_before, l2, meta_exponent, normalize,
            score_std_dev, calcer_seed, result_ids, result_scores, result_size,
        )
    elif score_function == SCORE_FUNCTION_LOO_L2:
        _launch_single_fold[hist_loader, SCORE_FUNCTION_LOO_L2](
            ctx, binary_features, binary_feature_count, cat_features_weights,
            bin_features_weights, binary_feature_weights_count, splits, parts,
            p_count, score_before, l2, meta_exponent, normalize,
            score_std_dev, calcer_seed, result_ids, result_scores, result_size,
        )
    elif (
        score_function == SCORE_FUNCTION_L2
        or score_function == SCORE_FUNCTION_NEWTON_L2
    ):
        _launch_single_fold[hist_loader, SCORE_FUNCTION_L2](
            ctx, binary_features, binary_feature_count, cat_features_weights,
            bin_features_weights, binary_feature_weights_count, splits, parts,
            p_count, score_before, l2, meta_exponent, normalize,
            score_std_dev, calcer_seed, result_ids, result_scores, result_size,
        )
    elif (
        score_function == SCORE_FUNCTION_COSINE
        or score_function == SCORE_FUNCTION_NEWTON_COSINE
    ):
        _launch_single_fold[hist_loader, SCORE_FUNCTION_COSINE](
            ctx, binary_features, binary_feature_count, cat_features_weights,
            bin_features_weights, binary_feature_weights_count, splits, parts,
            p_count, score_before, l2, meta_exponent, normalize,
            score_std_dev, calcer_seed, result_ids, result_scores, result_size,
        )
    else:
        raise Error(
            "FindOptimalSplitPlain: unknown score function"
            " (`pointwise_scores.cu:521` throws)"
        )


def _launch_single_fold[
    hist_loader: Int, score_function: Int
](
    ctx: DeviceContext,
    mut binary_features: DeviceBuffer[DType.uint32],
    binary_feature_count: Int,
    mut cat_features_weights: DeviceBuffer[DType.float32],
    mut bin_features_weights: DeviceBuffer[DType.float32],
    binary_feature_weights_count: Int,
    mut splits: DeviceBuffer[DType.float32],
    mut parts: DeviceBuffer[DType.float32],
    p_count: Int,
    mut score_before: DeviceBuffer[DType.float32],
    l2: Float32,
    meta_exponent: Float32,
    normalize: Bool,
    score_std_dev: Float32,
    seed: UInt64,
    mut result_ids: DeviceBuffer[DType.uint32],
    mut result_scores: DeviceBuffer[DType.float32],
    result_size: Int,
) raises:
    """Their `RUN()` macro (`:480-481`), expanded once.

    A module-level function rather than a nested one: a closure here cannot
    capture a `DeviceContext`.
    """
    ctx.enqueue_function[
        find_optimal_split_single_fold_kernel[
            POINTWISE_SCORE_BLOCK, hist_loader, score_function
        ]
    ](
        binary_features.unsafe_ptr(),
        Int32(binary_feature_count),
        cat_features_weights.unsafe_ptr(),
        bin_features_weights.unsafe_ptr(),
        Int32(binary_feature_weights_count),
        score_before.unsafe_ptr(),
        splits.unsafe_ptr(),
        parts.unsafe_ptr(),
        Int32(p_count),
        l2,
        meta_exponent,
        Int32(1) if normalize else Int32(0),
        score_std_dev,
        seed,
        result_ids.unsafe_ptr(),
        result_scores.unsafe_ptr(),
        grid_dim=(result_size, 1, 1),
        block_dim=(POINTWISE_SCORE_BLOCK, 1, 1),
    )


def find_optimal_split(
    ctx: DeviceContext,
    mut binary_features: DeviceBuffer[DType.uint32],
    binary_feature_count: Int,
    mut cat_features_weights: DeviceBuffer[DType.float32],
    mut bin_features_weights: DeviceBuffer[DType.float32],
    binary_feature_weights_count: Int,
    mut splits: DeviceBuffer[DType.float32],
    mut parts: DeviceBuffer[DType.float32],
    p_count: Int,
    fold_count: Int,
    mut score_before: DeviceBuffer[DType.float32],
    mut result_ids: DeviceBuffer[DType.uint32],
    mut result_scores: DeviceBuffer[DType.float32],
    result_size: Int,
    score_function: Int,
    l2: Float32,
    meta_l2_exponent: Float32,
    meta_l2_frequency: Float32,
    normalize: Bool,
    score_std_dev: Float32,
    seed: UInt64,
    gathered_by_leaves: Bool,
) raises:
    """`FindOptimalSplit` (`:530-551`), THE DISPATCH.

    `score_before` is a ONE-FLOAT DEVICE BUFFER where theirs is a host
    scalar -- DEVIATION 207 (the blind level loop keeps last level's
    winning score on the device). A caller that still holds a host scalar
    stages it with `enqueue_fill` on a buffer it OWNS; a per-call
    temporary dies at `.unsafe_ptr()` (the last-use trap) and is not an
    option.

        if (foldCount == 1) {
            gatheredByLeaves ? Plain<TGatheredByLeavesHistLoader>
                             : Plain<TDirectHistLoader>
        } else {
            Dynamic
        }

    Three arms, every one of them exercised by the gate (PORTING_RULES 8:
    a non-default path is an unchecked path, and reach is per-branch).
    """
    if fold_count == 1:
        if gathered_by_leaves:
            find_optimal_split_plain[HIST_LOADER_GATHERED_BY_LEAVES](
                ctx, binary_features, binary_feature_count,
                cat_features_weights, bin_features_weights,
                binary_feature_weights_count, splits, parts, p_count,
                score_before, result_ids, result_scores, result_size,
                score_function, l2, meta_l2_exponent, meta_l2_frequency,
                normalize, score_std_dev, seed,
            )
        else:
            find_optimal_split_plain[HIST_LOADER_DIRECT](
                ctx, binary_features, binary_feature_count,
                cat_features_weights, bin_features_weights,
                binary_feature_weights_count, splits, parts, p_count,
                score_before, result_ids, result_scores, result_size,
                score_function, l2, meta_l2_exponent, meta_l2_frequency,
                normalize, score_std_dev, seed,
            )
    else:
        find_optimal_split_dynamic(
            ctx, binary_features, binary_feature_count, cat_features_weights,
            bin_features_weights, binary_feature_weights_count, splits, parts,
            p_count, fold_count, score_before, result_ids, result_scores,
            result_size, score_function, l2, normalize, score_std_dev, seed,
        )


def gather_histogram_by_leaves(
    ctx: DeviceContext,
    mut histogram: DeviceBuffer[DType.float32],
    bin_feature_count: Int,
    hist_count: Int,
    leaf_count: Int,
    fold_count: Int,
    mut result: DeviceBuffer[DType.float32],
) raises:
    """`GatherHistogramByLeaves` (`:583-621`), grid arithmetic included.

        const int blockSize = 1024;
        const int leavesInBlock = Min<int>(leafCount, blockSize);
        numBlocks.x = (binFeatureCount + (blockSize / leavesInBlock) - 1)
                      / (blockSize / leavesInBlock);
        numBlocks.y = foldCount;
        numBlocks.z = (leafCount + blockSize - 1) / blockSize;
        if (IsGridEmpty(numBlocks)) return;
        switch (histCount) { 1, 2, 4; default: CB_ENSURE_INTERNAL(false) }

    `numBlocks.z` is launched and never read -- module docstring, fifth
    finding. Kept so the launch shape matches theirs.

    `helper` inside the kernel is built from `gridDim.y`, so `foldCount`
    reaches the kernel TWICE: once as the grid's y extent, which decides
    the histogram offset, and once as an argument, which decides the output
    stride. A launcher that sets one and not the other is silently wrong.
    """
    var leaves_in_block = min(leaf_count, POINTWISE_WIDE_BLOCK)
    var per_block = POINTWISE_WIDE_BLOCK // leaves_in_block
    var nx = (bin_feature_count + per_block - 1) // per_block
    var ny = fold_count
    var nz = (leaf_count + POINTWISE_WIDE_BLOCK - 1) // POINTWISE_WIDE_BLOCK
    if nx == 0 or ny == 0 or nz == 0:
        return

    if hist_count == 1:
        ctx.enqueue_function[
            gather_histograms_by_leaves_kernel[POINTWISE_WIDE_BLOCK, 1]
        ](
            Int32(bin_feature_count), histogram.unsafe_ptr(),
            Int32(hist_count), Int32(leaf_count), Int32(fold_count),
            result.unsafe_ptr(),
            grid_dim=(nx, ny, nz),
            block_dim=(POINTWISE_WIDE_BLOCK, 1, 1),
        )
    elif hist_count == 2:
        ctx.enqueue_function[
            gather_histograms_by_leaves_kernel[POINTWISE_WIDE_BLOCK, 2]
        ](
            Int32(bin_feature_count), histogram.unsafe_ptr(),
            Int32(hist_count), Int32(leaf_count), Int32(fold_count),
            result.unsafe_ptr(),
            grid_dim=(nx, ny, nz),
            block_dim=(POINTWISE_WIDE_BLOCK, 1, 1),
        )
    elif hist_count == 4:
        ctx.enqueue_function[
            gather_histograms_by_leaves_kernel[POINTWISE_WIDE_BLOCK, 4]
        ](
            Int32(bin_feature_count), histogram.unsafe_ptr(),
            Int32(hist_count), Int32(leaf_count), Int32(fold_count),
            result.unsafe_ptr(),
            grid_dim=(nx, ny, nz),
            block_dim=(POINTWISE_WIDE_BLOCK, 1, 1),
        )
    else:
        raise Error(
            "histCount should be 1, 2, or 4, not " + String(hist_count)
        )


def update_partition_props(
    ctx: DeviceContext,
    mut target: DeviceBuffer[DType.float32],
    mut weights: DeviceBuffer[DType.float32],
    mut counts: DeviceBuffer[DType.float32],
    have_target: Bool,
    have_weights: Bool,
    have_counts: Bool,
    mut parts: DeviceBuffer[DType.uint32],
    mut part_stats: DeviceBuffer[DType.float32],
    parts_count: Int,
) raises:
    """`UpdatePartitionProps` (`:681-695`).

        const int blockSize = 1024;
        if (partsCount) {
            PartitionUpdateImpl<blockSize> <<< partsCount, blockSize >>> (...);
        }

    One block per partition, 1024 threads, and the `if (partsCount)` guard
    because a zero-block launch is an error. The three `have_*` flags stand
    in for their three null pointers (see the kernel docstring).
    """
    if parts_count == 0:
        return
    ctx.enqueue_function[partition_update_kernel[POINTWISE_WIDE_BLOCK]](
        target.unsafe_ptr(),
        weights.unsafe_ptr(),
        counts.unsafe_ptr(),
        Int32(1) if have_target else Int32(0),
        Int32(1) if have_weights else Int32(0),
        Int32(1) if have_counts else Int32(0),
        parts.unsafe_ptr(),
        part_stats.unsafe_ptr(),
        grid_dim=(parts_count, 1, 1),
        block_dim=(POINTWISE_WIDE_BLOCK, 1, 1),
    )
