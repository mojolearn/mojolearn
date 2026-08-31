# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""One score helper per feature policy, folded into one best split.

PORT OF `catboost/cuda/methods/pointwise_scores_calcer.h` at CatBoost
`54a8143a` -- `TScoreHelper` (which their `histograms_helper.h:353-419`
defines) and `TScoresCalcerOnCompressedDataSet`. Transliterated. Do not
improve.

This is the join between everything the pointwise family has built: a
`TScoreHelper` owns a `TComputeHistogramsHelper` (the full-pass state
machine) and a `TFindBestSplitsHelper` (the scorer), and the calcer owns ONE
HELPER PER POLICY and folds their answers.

    binary      32 one-bit features per block
    half-byte    8 features of up to 16 bins
    one-byte     4 features, dispatched again by bit width on the device

A policy with no features gets NO HELPER and therefore no launch
(`pointwise_scores_calcer.h:42,48,54` each guard on `GetGridSize(policy)`),
which is why an all-binary dataset costs one launch family and not three.

## THE FOLD DIRECTION IS NOT THE SAME IN BOTH PLACES, and it is theirs

`TakeBest(first, second)` is `first < second ? first : second` with a STRICT
comparator, so a full tie -- equal gain, equal feature id, equal bin id --
falls through to `second`. Which record that is differs between the two call
sites:

    pointwise_scores_calcer.h:100   TakeBest(helper->Read(), best)
                                    new candidate FIRST -> a tie keeps the
                                    INCUMBENT

    oblivious_tree_doc_parallel_    TakeBest(bestSplitProp, calcer->Read())
      structure_searcher.cpp:115    incumbent FIRST -> a tie takes the NEW
                                    one

Both are theirs. This file implements the first; the searcher implements the
second. Reversing either is invisible on any fixture whose candidates do not
tie exactly, and changes which feature a tie resolves to on one that does.

## THE SIGN CONVENTION (PORTING.md 94a)

`gbdt/methods/kernel/pointwise_scores.mojo` keeps CATBOOST's: `FLT_MAX`
sentinel, `gain < bestGain`, LOWER IS BETTER. The greedy-subsets scorer in
this same repository keeps the OPPOSITE, with one negation folded into its
host. Everything in this file and in the searcher above it is on CatBoost's
side of that, and a reader carrying the greedy family's intuition here will
read every comparison backwards.

DEVIATION 103: `TScoreHelper` upstream also owns a `TComputationStream` and
`requestStream` decides whether it gets its own. There are no streams on
Metal, so the field and the constructor argument are gone; every launch is
ordered on the one queue. Nothing else about the class changes.
"""

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from gbdt.gpu_data.compressed_index_builder import (
    CompressedIndexLayout,
)
from gbdt.gpu_data.feature_blocks import PolicyBlock
from gbdt.methods.helpers import TBestSplitProperties, take_best
from gbdt.methods.histograms_helper import (
    POLICY_BINARY,
    POLICY_HALF_BYTE,
    POLICY_ONE_BYTE,
    ComputeHistogramsHelper,
)
from gbdt.methods.kernel.pointwise_scores import find_optimal_split
from gbdt.methods.kernel.pointwise_split_resolve import (
    launch_pw_fold_winner,
    launch_pw_seed_sentinel,
)
from gbdt.methods.pointwise_kernels import FoldsHistogram, compute_hist2
from gbdt.methods.pointwise_optimization_subsets import TOptimizationSubsets
from gbdt.data.permutation import TRandom


def folds_histogram_for(folds: List[UInt32]) raises -> FoldsHistogram:
    """`TCpuGrid::ComputeFoldsHistogram` (`gpu_data/feature_layout.cpp:23`),
    on the host, for one policy's features.

        result.Counts[IntLog2(foldCount)]++   for every feature with folds > 0

    The one-byte fan-out needs it to size each bit width's multiplier, and
    `pointwise_kernels.cpp:57-60` reads it with the ranges `(4,5) (6,6)
    (7,7) (8,8)` -- so a 16-fold feature (IntLog2 == 4) counts toward the
    FIVE-bit kernel. `PORTING.md` 102a.
    """
    var h = FoldsHistogram()
    for i in range(len(folds)):
        var f = Int(folds[i])
        if f > 0:
            # `NCB::IntLog2` is `(ui32)ceil(log2(values))`
            # (`libs/helpers/math_utils.h:14-16`) -- CEILING, and the first
            # version of this function used floor. That is not a rounding
            # nicety, it is the whole dispatch:
            #
            #   a 100-fold feature has ceil 7 -> counted under (7,7), which
            #   launches the 7-bit kernel, whose device bound is (64, 128]
            #   and accepts it. Under FLOOR it is 6, which launches the
            #   6-bit kernel, whose bound is (32, 64] and REJECTS it -- and
            #   the 7-bit kernel never launches at all, because
            #   `if (featureCountForBits)` sees zero.
            #
            # So every one-byte feature whose fold count is not a power of
            # two goes unhistogrammed, SILENTLY: the tree still grows, on
            # whatever the other policies offer. Caught by
            # `mojo_only/pointwise_vs_greedy_check.mojo` on its first run.
            #
            # It is also why their one-byte ranges start at FOUR
            # (`PORTING.md` 102a): ceil(log2(16)) is 4, so a 16-fold
            # feature belongs to the 5-bit kernel, whose bound is (15, 32].
            var bit = 0
            while (1 << bit) < f:
                bit += 1
            h.counts[bit] += 1
    return h^


struct PolicyScoreHelper(Movable):
    """`TScoreHelper<TDocParallelLayout>` (`histograms_helper.h:353-419`).

    Owns the histogram buffer for ONE policy, the bin-feature table the
    scorer reads, and the state machine that decides full versus partial.
    """

    var policy: Int
    var feature_count: Int
    var bin_feature_count: Int
    """`GetBinFeatureCount(Policy)`: the policy's total fold count, and the
    `histLineSize` every kernel below strides by."""

    var weight_count: Int
    """`len(global_feature_ids)`. The score weights are indexed by FEATURE
    id while `bin_feature_count` above is a BIN-FEATURE count; the two were
    the same variable until 2026-08-31 and keeping both named is what stops
    them being confused again."""

    var folds_hist: FoldsHistogram
    var hist_helper: ComputeHistogramsHelper

    var d_offset: DeviceBuffer[DType.uint32]
    var d_first_fold: DeviceBuffer[DType.uint32]
    var d_folds: DeviceBuffer[DType.uint32]
    var d_one_hot: DeviceBuffer[DType.uint8]
    var d_bf: DeviceBuffer[DType.uint32]
    """`TCBinFeature[]`, 3 words each: FeatureId, BinId, SkipInScoreCount."""

    var d_hist: DeviceBuffer[DType.float32]
    var d_cat_w: DeviceBuffer[DType.float32]
    var d_bin_w: DeviceBuffer[DType.float32]
    var result_blocks: Int
    """`const ui64 blockCount = 32;` and
    `min(CeilDivide(histograms.Size(), 128), blockCount)`
    (`histograms_helper.h:205-208`). **It is the GRID**, and every block
    writes its OWN best record at `2 * blockIdx.x`, so `read_optimal_split`
    has to fold all of them.

    Reading only block 0 is not a partial answer that is usually right --
    it is an argmin over the first 128 bin features and nothing else, and
    it looks exactly like a working searcher that prefers low-numbered
    bins."""

    var d_result_ids: DeviceBuffer[DType.uint32]
    var d_result_scores: DeviceBuffer[DType.float32]
    var h_result_ids: HostBuffer[DType.uint32]
    var h_result_scores: HostBuffer[DType.float32]
    var d_score_scratch: DeviceBuffer[DType.float32]
    """One float. The score kernels read `scoreBeforeSplit` from the device
    (DEVIATION 207); a caller that still holds it as a host scalar stages
    it here with `enqueue_fill` -- owned by the helper because a per-call
    temporary dies at `.unsafe_ptr()` (the last-use trap)."""
    # `TScoreHelper`'s third constructor argument
    # (`histograms_helper.h:361-365`), handed to BOTH halves: it sizes the
    # histogram and becomes `gridDim.z`, and `foldCount == 1` IS the dispatch
    # between the plain and the dynamic scorer. Was a literal 1 at three
    # sites -- DEVIATION 126.
    var fold_count: Int

    def __init__(
        out self,
        ctx: DeviceContext,
        block: PolicyBlock,
        layout: CompressedIndexLayout,
        n_rows: Int,
        max_depth: Int,
        global_feature_ids: List[Int],
        fold_count: Int = 1,
    ) raises:
        """`CreateScoreHelper` (`pointwise_scores_calcer.h:11-27`) plus the
        host-side tables their `TCompressedDataSet` already holds.

        `feature_offset` is `TCFeature::Offset` -- the compressed-index
        COLUMN base in elements. This tree stores the index column-major
        with a stride of `n_rows`, so a feature's base is
        `layout.features[f].offset * n_rows`.

        **IT COMES FROM THE LAYOUT, PER FEATURE, and an earlier version
        computed it as `block.first_column + block.group_offset[i]`.**
        `PolicyBlock.group_offset` is 0 for EVERY feature -- it is not a
        group index -- so that formula returned the policy's FIRST column
        for all of them. A policy with more features than fit in one word
        (5 one-byte, 9 half-byte, 33 binary) had every feature past the
        first group reading the wrong column.

        It survived two gates. `check-pointwise-vs-greedy` and
        `check-fit-pointwise` both happened to plant their signal on
        features inside the first group of their policy, so the wrong
        column was never the one carrying the answer. Found by a THIRD
        fixture whose signal sat on the fifth one-byte feature; both gates
        now carry such a feature.
        """
        self.policy = block.policy
        self.fold_count = fold_count
        self.feature_count = block.count()
        self.folds_hist = folds_histogram_for(block.folds)

        var total = 0
        for i in range(self.feature_count):
            total += Int(block.folds[i])
        self.bin_feature_count = total

        self.hist_helper = ComputeHistogramsHelper(
            block.policy, fold_count, max_depth
        )

        var off = List[UInt32]()
        var first = List[UInt32]()
        var fol = List[UInt32]()
        var oh = List[UInt8]()
        var bf = List[UInt32]()
        for i in range(self.feature_count):
            # the column this feature's GROUP occupies, times the row stride
            off.append(
                UInt32(
                    Int(layout.features[block.feature_ids[i]].offset)
                    * n_rows
                )
            )
            first.append(block.fold_offset[i])
            fol.append(block.folds[i])
            # OFF THE LAYOUT, the same place the offset two lines above
            # comes from. This was a hardcoded `UInt8(0)` and every one-hot
            # feature therefore got PREFIX-SUMMED -- its equality
            # candidates scored as thresholds. See DEVIATION 114 for how a
            # gated skip stayed broken: both checks on it hand the flag
            # array in BY HAND, so no check ever built one through this
            # constructor.
            oh.append(
                UInt8(1) if layout.features[
                    block.feature_ids[i]
                ].one_hot_feature else UInt8(0)
            )
            var gid = UInt32(global_feature_ids[block.feature_ids[i]])
            for b in range(Int(block.folds[i])):
                bf.append(gid)
                bf.append(UInt32(b))
                bf.append(UInt32(0))  # SkipInScoreCount

        var n = self.feature_count
        self.d_offset = ctx.enqueue_create_buffer[DType.uint32](n)
        self.d_first_fold = ctx.enqueue_create_buffer[DType.uint32](n)
        self.d_folds = ctx.enqueue_create_buffer[DType.uint32](n)
        self.d_one_hot = ctx.enqueue_create_buffer[DType.uint8](n)
        ctx.enqueue_copy(dst_buf=self.d_offset, src_ptr=off.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=self.d_first_fold, src_ptr=first.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=self.d_folds, src_ptr=fol.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=self.d_one_hot, src_ptr=oh.unsafe_ptr())

        self.d_bf = ctx.enqueue_create_buffer[DType.uint32](3 * total)
        ctx.enqueue_copy(dst_buf=self.d_bf, src_ptr=bf.unsafe_ptr())

        # allocation is for the DEEPEST level, view is per level; the two
        # are different expressions and `histograms_helper` says why
        self.d_hist = ctx.enqueue_create_buffer[DType.float32](
            self.hist_helper.histogram_alloc_size(total)
        )
        ctx.enqueue_memset(self.d_hist, Float32(0.0))

        # THE WEIGHTS ARE INDEXED BY FEATURE ID, NOT BY BIN-FEATURE ID, so
        # they are sized by FEATURE COUNT. They were sized `total`, the
        # policy's bin-feature count, and `pointwise_scores.mojo:782-785`
        # reads `weights[feature_id]` where `feature_id` is the GLOBAL id
        # planted into `d_bf` twenty lines above. Two index spaces, two
        # sizes, and a device read past the end of a live buffer whenever a
        # policy's largest global id reaches its own bin-feature count.
        #
        # Upstream indexes by feature too (`compute_scores.cu:136`,
        # `binFeaturesWeights[featureId]`, transcribed at
        # `greedy_subsets_searcher/kernel/compute_scores.mojo:287-289`), and
        # the GREEDY arm in this tree sizes them `n_features`
        # (`greedy_search_helper.mojo:583, 1095, 2906`). Only the pointwise
        # arm got it wrong.
        #
        # WHY EVERY GATE STAYED GREEN, and it is DEVIATION 114's pattern for
        # the third time in this file. In every fixture in the tree the
        # binary features are the FIRST input columns, so a policy's largest
        # global id is one less than its own feature count and the read
        # lands in bounds by a hair: `check-fit-pointwise` has one binary
        # feature at gid 0 with total 1, `check-pointwise-vs-greedy` two at
        # gids 0..1 with total 2, `covbin_*` forty-three at gids 0..42 with
        # total 43. COVTYPE IS THE ONLY SHAPE WHERE THEY ARE NOT: its ten
        # one-byte features are gids 0..9 and its forty-three binary
        # features are gids 10..52, against a binary-policy total of 43, so
        # ten features read off the end. And `pointwise_scores_check.mojo`
        # cannot see it either, because it allocates these arrays itself at
        # `fx.n_features` and hands them in, so no check has ever built one
        # through this constructor.
        var n_feat = len(global_feature_ids)
        var ones = List[Float32]()
        for _ in range(n_feat):
            ones.append(1.0)
        self.weight_count = n_feat
        self.d_cat_w = ctx.enqueue_create_buffer[DType.float32](n_feat)
        self.d_bin_w = ctx.enqueue_create_buffer[DType.float32](n_feat)
        ctx.enqueue_copy(dst_buf=self.d_cat_w, src_ptr=ones.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=self.d_bin_w, src_ptr=ones.unsafe_ptr())

        var blocks_n = (total + 127) // 128
        if blocks_n > 32:
            blocks_n = 32
        if blocks_n < 1:
            blocks_n = 1
        self.result_blocks = blocks_n
        self.d_result_ids = ctx.enqueue_create_buffer[DType.uint32](
            2 * blocks_n
        )
        self.d_result_scores = ctx.enqueue_create_buffer[DType.float32](
            2 * blocks_n
        )
        self.h_result_ids = ctx.enqueue_create_host_buffer[DType.uint32](
            2 * blocks_n
        )
        self.h_result_scores = ctx.enqueue_create_host_buffer[
            DType.float32
        ](2 * blocks_n)
        self.d_score_scratch = ctx.enqueue_create_buffer[DType.float32](1)
        ctx.synchronize()

    def submit_compute(
        mut self,
        ctx: DeviceContext,
        mut subsets: TOptimizationSubsets,
        mut cindex: DeviceBuffer[DType.uint32],
        mut docs: DeviceBuffer[DType.uint32],
        n_rows: Int,
        sm_count: Int,
        fixed_scale: Float32,
    ) raises:
        """`TScoreHelper::SubmitCompute` (`histograms_helper.h:380-384`),
        which is `ComputeHistogramsHelper.Compute` and nothing else."""
        var plan = self.hist_helper.plan(Int(subsets.current_depth))
        if self.feature_count == 0:
            return
        # TWO SEPARATE BUFFERS, and that is not a convenience: the kernels
        # take `target` and `weight` on independent origins, and Mojo
        # refuses two views of one buffer at a launch. DEVIATION 97.2.
        var weight_p = subsets.gathered_weight.unsafe_ptr()
        var target_p = subsets.gathered_target.unsafe_ptr()
        var parts_p = subsets.partitions.unsafe_ptr()
        compute_hist2(
            ctx,
            self.policy,
            self.d_offset.unsafe_ptr(),
            self.d_first_fold.unsafe_ptr(),
            self.d_folds.unsafe_ptr(),
            self.d_one_hot.unsafe_ptr(),
            self.feature_count,
            0,
            self.bin_feature_count,
            cindex.unsafe_ptr(),
            # PLANE 1 is the weighted target and PLANE 0 the weight
            # (DEVIATION 97.3): `gathered` holds them `doc_count` apart in
            # `TPartitionStatistics{Weight, Sum}` order, which is the
            # REVERSE of `TL2Target`'s declaration order. Passing them the
            # other way round swaps every gradient with its weight, changes
            # no total, and destroys every split.
            target_p,
            weight_p,
            docs.unsafe_ptr(),
            n_rows,
            parts_p,
            plan.part_count,
            self.fold_count,
            self.d_hist.unsafe_ptr(),
            self.bin_feature_count,
            plan.build_from_scratch,
            self.folds_hist.copy(),
            sm_count,
            fixed_scale,
        )
        self.hist_helper.clear_from_scratch()

    def compute_optimal_split(
        mut self,
        ctx: DeviceContext,
        mut part_stats: DeviceBuffer[DType.float32],
        part_count: Int,
        score_before_split: Float32,
        score_function: Int,
        l2: Float32,
        score_std_dev: Float32,
        seed: UInt64,
    ) raises:
        """`TScoreHelper::ComputeOptimalSplit`
        (`histograms_helper.h:390-404`). The host-scalar form: stages the
        scalar into `d_score_scratch` (an enqueued fill, no sync) and
        launches -- kept for the OTHER searcher and the check library,
        which still carry the score on the host. The launch is spelled out
        rather than delegated to `compute_optimal_split_dev` because
        passing `self.d_score_scratch` mutably alongside `mut self` is the
        DEVIATION 97.2 aliasing refusal."""
        if self.feature_count == 0:
            return
        self.d_score_scratch.enqueue_fill(score_before_split)
        find_optimal_split(
            ctx,
            self.d_bf,
            self.bin_feature_count,
            self.d_cat_w,
            self.d_bin_w,
            self.weight_count,
            self.d_hist,
            part_stats,
            part_count,
            self.fold_count,
            self.d_score_scratch,
            self.d_result_ids,
            self.d_result_scores,
            self.result_blocks,
            score_function,
            l2,
            Float32(1.0),
            Float32(0.0),
            False,
            score_std_dev,
            seed,
            False,
        )

    def compute_optimal_split_dev(
        mut self,
        ctx: DeviceContext,
        mut part_stats: DeviceBuffer[DType.float32],
        part_count: Int,
        mut score_before: DeviceBuffer[DType.float32],
        score_function: Int,
        l2: Float32,
        score_std_dev: Float32,
        seed: UInt64,
    ) raises:
        """The device-score form (DEVIATION 207): `scoreBeforeSplit` is a
        one-float device buffer the blind level loop's pack kernel wrote,
        never read by the host. Same launches, same arithmetic."""
        if self.feature_count == 0:
            return
        find_optimal_split(
            ctx,
            self.d_bf,
            self.bin_feature_count,
            self.d_cat_w,
            self.d_bin_w,
            self.weight_count,
            self.d_hist,
            part_stats,
            part_count,
            self.fold_count,
            score_before,
            self.d_result_ids,
            self.d_result_scores,
            self.result_blocks,
            score_function,
            l2,
            Float32(1.0),
            Float32(0.0),
            False,
            score_std_dev,
            seed,
            False,
        )

    def read_optimal_split(
        mut self, ctx: DeviceContext
    ) raises -> TBestSplitProperties:
        """`TFindBestSplitsHelper::ReadOptimalSplit`
        (`histograms_helper.h:248`). The kernel already reduced to one
        record per block and the launch is one block, so this is a read."""
        if self.feature_count == 0:
            return TBestSplitProperties()
        ctx.enqueue_copy(dst_buf=self.h_result_ids, src_buf=self.d_result_ids)
        ctx.enqueue_copy(
            dst_buf=self.h_result_scores, src_buf=self.d_result_scores
        )
        ctx.synchronize()
        # `BestSplit(BestScores, Stream)` (`histograms_helper.h:250`): a fold
        # across the per-block records, not a read of the first one.
        var best = TBestSplitProperties()
        for b in range(self.result_blocks):
            best = take_best(
                TBestSplitProperties(
                    Int32(self.h_result_ids[2 * b]),
                    Int32(self.h_result_ids[2 * b + 1]),
                    self.h_result_scores[2 * b],
                    self.h_result_scores[2 * b + 1],
                ),
                best,
            )
        return best

    def reset_for_tree(mut self, ctx: DeviceContext) raises:
        """CONSTRUCTOR POSTCONDITIONS a new tree reads, for the pooled
        calcer: the histogram zeroed (`:263`'s memset) and the full-pass
        state machine at `CurrentBit = -1` / `BuildFromScratch = true`.

        Nothing else the constructor built moves between trees: the
        layout uploads (`d_offset`/`d_first_fold`/`d_folds`/`d_one_hot`/
        `d_bf`), the constant-ones fills (`d_cat_w`/`d_bin_w`) and the
        result scratch (overwritten by every `compute_optimal_split`,
        read only after it) are tree-invariant. The histogram memset is
        NOT an optimization to skip: the M > 1 writeback is an atomicAdd
        (`pointwise_hist2_one_byte_templ.mojo`'s writeback), so a
        from-scratch level adds onto whatever the buffer holds, and
        after a tree it holds the previous tree's SCANNED sums.
        """
        ctx.enqueue_memset(self.d_hist, Float32(0.0))
        self.hist_helper.reset()


struct ScoresCalcerOnCompressedDataSet(Movable):
    """`TScoresCalcerOnCompressedDataSet<TDocParallelLayout>`
    (`pointwise_scores_calcer.h:29-110`).

    A fan-out and a fold, and nothing else. It holds one helper per policy
    PRESENT in the dataset and forwards each of the three calls to all of
    them.
    """

    var helpers: List[PolicyScoreHelper]

    def __init__(
        out self,
        ctx: DeviceContext,
        blocks: List[PolicyBlock],
        layout: CompressedIndexLayout,
        n_rows: Int,
        max_depth: Int,
        global_feature_ids: List[Int],
        fold_count: Int = 1,
    ) raises:
        """Their constructor (`:35-59`): a helper per policy whose
        `GetGridSize` is non-zero. `blocks_for` already omits empty
        policies, so the guard is structural here rather than a test."""
        self.helpers = List[PolicyScoreHelper]()
        for i in range(len(blocks)):
            self.helpers.append(
                PolicyScoreHelper(
                    ctx, blocks[i], layout, n_rows, max_depth,
                    global_feature_ids, fold_count,
                )
            )

    def submit_compute(
        mut self,
        ctx: DeviceContext,
        mut subsets: TOptimizationSubsets,
        mut cindex: DeviceBuffer[DType.uint32],
        mut docs: DeviceBuffer[DType.uint32],
        n_rows: Int,
        sm_count: Int,
        fixed_scale: Float32,
    ) raises:
        """`SubmitCompute` (`:71-78`)."""
        for i in range(len(self.helpers)):
            self.helpers[i].submit_compute(
                ctx, subsets, cindex, docs, n_rows, sm_count, fixed_scale
            )

    def compute_optimal_split(
        mut self,
        ctx: DeviceContext,
        mut part_stats: DeviceBuffer[DType.float32],
        part_count: Int,
        score_before_split: Float32,
        score_function: Int,
        l2: Float32,
        score_std_dev: Float32,
        seed: UInt64,
    ) raises:
        """`ComputeOptimalSplit` (`:80-92`).

        THE SEED ADVANCES PER HELPER, not per call: theirs builds one
        `TRandom rand(seed)` and hands each helper `rand.NextUniformL()`, so
        the three policies get three different noise draws from one seed.
        Giving them all the same seed would correlate the `ScoreStdDev`
        noise across policies, which is invisible at `random_strength = 0`
        and biased everywhere else.
        """
        var rnd = TRandom(seed)
        for i in range(len(self.helpers)):
            self.helpers[i].compute_optimal_split(
                ctx,
                part_stats,
                part_count,
                score_before_split,
                score_function,
                l2,
                score_std_dev,
                rnd.next_uniform_l(),
            )

    def compute_optimal_split_dev(
        mut self,
        ctx: DeviceContext,
        mut part_stats: DeviceBuffer[DType.float32],
        part_count: Int,
        mut score_before: DeviceBuffer[DType.float32],
        score_function: Int,
        l2: Float32,
        score_std_dev: Float32,
        seed: UInt64,
    ) raises:
        """The device-score fan-out (DEVIATION 207). Identical to
        `compute_optimal_split` -- including the per-helper seed advance --
        except `scoreBeforeSplit` stays on the device."""
        var rnd = TRandom(seed)
        for i in range(len(self.helpers)):
            self.helpers[i].compute_optimal_split_dev(
                ctx,
                part_stats,
                part_count,
                score_before,
                score_function,
                l2,
                score_std_dev,
                rnd.next_uniform_l(),
            )

    def resolve_optimal_split(
        mut self,
        ctx: DeviceContext,
        mut best_ids: DeviceBuffer[DType.uint32],
        mut best_scores: DeviceBuffer[DType.float32],
    ) raises:
        """The device twin of `read_optimal_split` (DEVIATION 207): the same
        two nested folds, run by one-thread kernels instead of the host, so
        no drain. Fold order and tie rules are BIT-EQUAL to the host pair by
        construction: within a helper the per-block fold keeps the INCUMBENT
        on a full tie (challenger-first `take_best`, `read_optimal_split`'s
        loop), and across helpers the EARLIER policy wins a tie
        (`TakeBest(helper->Read(), best)`, `:94-105`). `is_first` seeds the
        incumbent slot with the undefined sentinel, so a dataset whose every
        helper is empty resolves to `FeatureId = (ui32)-1` exactly as the
        host fold returns the default record. `mojo_only/
        pointwise_resolve_check.mojo` plants the tie cases and compares this
        against the host fold record for record."""
        var first = True
        for i in range(len(self.helpers)):
            if self.helpers[i].feature_count == 0:
                continue
            launch_pw_fold_winner(
                ctx,
                self.helpers[i].d_result_ids,
                self.helpers[i].d_result_scores,
                self.helpers[i].result_blocks,
                first,
                best_ids,
                best_scores,
            )
            first = False
        if first:
            # no helper has features: the host fold would return the
            # sentinel default; write it so the pack kernel sees the same
            launch_pw_seed_sentinel(ctx, best_ids, best_scores)

    def read_optimal_split(
        mut self, ctx: DeviceContext
    ) raises -> TBestSplitProperties:
        """`ReadOptimalSplit` (`:94-105`).

        **THE FOLD IS `TakeBest(helper->Read(), best)`** -- the new
        candidate FIRST -- so a full tie keeps the INCUMBENT, which here
        means the policy that was folded EARLIER wins. The searcher above
        folds the other way. See the module docstring; both are theirs.
        """
        var best = TBestSplitProperties()
        for i in range(len(self.helpers)):
            best = take_best(self.helpers[i].read_optimal_split(ctx), best)
        return best

    def reset_for_tree(mut self, ctx: DeviceContext) raises:
        """The pooled calcer's per-tree reset: each helper back to its
        constructor postconditions. See `PolicyScoreHelper.reset_for_tree`
        for what that is and why the histogram memset is load-bearing.
        No drain: the memsets are enqueued and ordered before the next
        tree's launches by the queue, exactly like the constructor's own
        memset was."""
        for i in range(len(self.helpers)):
            self.helpers[i].reset_for_tree(ctx)
