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

from gbdt.gpu_data.feature_blocks import PolicyBlock
from gbdt.methods.helpers import TBestSplitProperties, take_best
from gbdt.methods.histograms_helper import (
    POLICY_BINARY,
    POLICY_HALF_BYTE,
    POLICY_ONE_BYTE,
    ComputeHistogramsHelper,
)
from gbdt.methods.kernel.pointwise_scores import find_optimal_split
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
            var bit = 0
            var v = f
            while v > 1:
                v >>= 1
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
    var d_result_ids: DeviceBuffer[DType.uint32]
    var d_result_scores: DeviceBuffer[DType.float32]
    var h_result_ids: HostBuffer[DType.uint32]
    var h_result_scores: HostBuffer[DType.float32]

    def __init__(
        out self,
        ctx: DeviceContext,
        block: PolicyBlock,
        n_rows: Int,
        max_depth: Int,
        global_feature_ids: List[Int],
    ) raises:
        """`CreateScoreHelper` (`pointwise_scores_calcer.h:11-27`) plus the
        host-side tables their `TCompressedDataSet` already holds.

        `feature_offset` is `TCFeature::Offset` -- the compressed-index
        COLUMN base in elements. This tree stores the index column-major
        with a stride of `n_rows` (`greedy_search_helper` passes `n_rows` as
        `bins_line_size`), so a group's base is `column * n_rows`, which is
        exactly what `cindex += feature->Offset` expects.
        """
        self.policy = block.policy
        self.feature_count = block.count()
        self.folds_hist = folds_histogram_for(block.folds)

        var total = 0
        for i in range(self.feature_count):
            total += Int(block.folds[i])
        self.bin_feature_count = total

        self.hist_helper = ComputeHistogramsHelper(block.policy, 1, max_depth)

        var off = List[UInt32]()
        var first = List[UInt32]()
        var fol = List[UInt32]()
        var oh = List[UInt8]()
        var bf = List[UInt32]()
        for i in range(self.feature_count):
            # the column this feature's GROUP occupies, times the row stride
            off.append(UInt32((block.first_column + Int(block.group_offset[i])) * n_rows))
            first.append(block.fold_offset[i])
            fol.append(block.folds[i])
            oh.append(UInt8(0))
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

        var ones = List[Float32]()
        for _ in range(total):
            ones.append(1.0)
        self.d_cat_w = ctx.enqueue_create_buffer[DType.float32](total)
        self.d_bin_w = ctx.enqueue_create_buffer[DType.float32](total)
        ctx.enqueue_copy(dst_buf=self.d_cat_w, src_ptr=ones.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=self.d_bin_w, src_ptr=ones.unsafe_ptr())

        self.d_result_ids = ctx.enqueue_create_buffer[DType.uint32](2)
        self.d_result_scores = ctx.enqueue_create_buffer[DType.float32](2)
        self.h_result_ids = ctx.enqueue_create_host_buffer[DType.uint32](2)
        self.h_result_scores = ctx.enqueue_create_host_buffer[DType.float32](2)
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
            1,
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
        (`histograms_helper.h:390-404`)."""
        if self.feature_count == 0:
            return
        find_optimal_split(
            ctx,
            self.d_bf,
            self.bin_feature_count,
            self.d_cat_w,
            self.d_bin_w,
            self.bin_feature_count,
            self.d_hist,
            part_stats,
            part_count,
            1,
            score_before_split,
            self.d_result_ids,
            self.d_result_scores,
            2,
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
        return TBestSplitProperties(
            Int32(self.h_result_ids[0]),
            Int32(self.h_result_ids[1]),
            self.h_result_scores[0],
            self.h_result_scores[1],
        )


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
        n_rows: Int,
        max_depth: Int,
        global_feature_ids: List[Int],
    ) raises:
        """Their constructor (`:35-59`): a helper per policy whose
        `GetGridSize` is non-zero. `blocks_for` already omits empty
        policies, so the guard is structural here rather than a test."""
        self.helpers = List[PolicyScoreHelper]()
        for i in range(len(blocks)):
            self.helpers.append(
                PolicyScoreHelper(
                    ctx, blocks[i], n_rows, max_depth, global_feature_ids
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
