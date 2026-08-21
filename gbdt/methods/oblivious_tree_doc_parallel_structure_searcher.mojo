"""The doc-parallel oblivious searcher: one split per level, all leaves at once.

PORT OF `catboost/cuda/methods/
oblivious_tree_doc_parallel_structure_searcher.{h,cpp}` at CatBoost
`54a8143a` -- `TDocParallelObliviousTreeSearcher::FitImpl`. Transliterated.
Do not improve.

**THIS IS THE FIRST CALLER OF THE POINTWISE FAMILY.** Everything under it --
six accumulators, three drivers, the host launch layer, the scorer, the
subsets, the histogram state machine -- was correct and reached by nothing
until this file existed (`UNWIRED.md`).

It is also the learner CatBoost runs for single-target symmetric trees at
`boosting_type=Plain` (`PORTING.md` 91 F), which is the arm every matched
benchmark in this repository pins CatBoost to. The other symmetric learner
in this tree, `greedy_subsets_searcher`, is the one CatBoost runs for
MULTICLASS symmetric trees.

## The level loop, which is short and does nothing clever

    for depth in 0 .. MaxDepth:
        gather the doc ids into partition order
        reduce the partition stats
        submit histograms for every policy
        score them, fold to one best split
        stop if the structure already has that split
        split the subsets, append to the structure

The whole of the oblivious constraint is in that shape: ONE split per level
applies to EVERY leaf, so the loop asks for one best split per depth and
never per leaf. It is what makes the histogram partial pass possible -- at
depth d there are exactly `1 << d` parts arranged in sibling pairs.

## THE STOPPING RULE IS `HasSplit`, NOT A SCORE THRESHOLD

    if (structure.HasSplit(bestSplit)) { leaves = ...; break; }

A level that re-proposes a split already in the tree has found nothing new,
and CatBoost treats that as the tree being finished rather than as an error.
It is checked BEFORE the split is applied, so the structure never contains a
duplicate.

## THE FOLD DIRECTION, and it is the opposite of the calcer's

    bestSplitProp = TakeBest(bestSplitProp, calcer->ReadOptimalSplit())

incumbent FIRST (`:115`, `:119`), so on a full tie the NEW candidate wins.
`TScoresCalcerOnCompressedDataSet::ReadOptimalSplit` folds the other way and
keeps the incumbent. Both are theirs; `gbdt/methods/helpers.take_best`
carries the argument.

DEVIATION 104: `ComputeWeakTarget`, the bootstrap and the leaf estimation are
NOT in this file. Theirs computes gradients, bootstraps, and estimates leaf
values inside `FitImpl`; ours takes the weak target already computed and
returns the STRUCTURE ONLY. The reason is that this repository's boosting
loop already owns all three for the greedy learner
(`doc_parallel_boosting.fit`), and duplicating them here would fork the
gradient path -- the one thing that must not differ between two learners
being compared. What this file returns is exactly the part that differs.

DEVIATION 105: `observations` must be the identity. Theirs gathers
`groupedByBinObservations = observations[subsets.Indices]` where
`observations` is the bootstrapped, filtered doc list; with no bootstrap
that is the identity and the gather collapses to `subsets.Indices` itself.
This file RAISES on a non-identity request rather than silently using the
wrong array, and the argument is there so the signature does not change when
bootstrap arrives.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from gbdt.gpu_data.feature_blocks import PolicyBlock, blocks_for
from gbdt.gpu_data.compressed_index_builder import CompressedIndexLayout
from gbdt.methods.helpers import TBestSplitProperties, take_best
from gbdt.methods.pointwise_optimization_subsets import (
    TL2Target,
    TOptimizationSubsets,
    create_subsets,
    split_subsets,
)
from gbdt.methods.pointwise_scores_calcer import ScoresCalcerOnCompressedDataSet
from gbdt.models.oblivious_model import TBinarySplit


def fit_oblivious_tree_structure(
    ctx: DeviceContext,
    layout: CompressedIndexLayout,
    n_rows: Int,
    max_depth: Int,
    mut cindex: DeviceBuffer[DType.uint32],
    var weights: DeviceBuffer[DType.float32],
    var weighted_target: DeviceBuffer[DType.float32],
    sm_count: Int,
    fixed_scale: Float32,
    score_function: Int,
    l2_leaf_reg: Float32 = Float32(3.0),
    score_std_dev: Float32 = Float32(0.0),
    seed: UInt64 = 0,
    one_hot: List[Bool] = List[Bool](),
    bootstrapped_observations: Bool = False,
) raises -> List[TBinarySplit]:
    """`TDocParallelObliviousTreeSearcher::FitImpl` (`:12-160`), the
    structure half.

    The weak target arrives as TWO buffers, which is `TL2Target` upstream
    and is forced here besides: the histogram kernels take `target` and
    `weight` on independent origins and Mojo refuses two views of one buffer
    at a launch (DEVIATION 97.2, found at exactly this wiring step).
    `run_tree_layout` takes the same two planes as ONE buffer, so a gate
    comparing the two searchers builds both forms from one source.
    """
    if bootstrapped_observations:
        raise Error(
            "DEVIATION 105: this searcher takes the identity observation"
            " list. Their `Gather(groupedByBinObservations, observations,"
            " subsets.Indices)` needs the bootstrapped doc list, and no"
            " bootstrap runs here yet; wire ComputeWeakTarget's filter"
            " before passing True."
        )

    var blocks = blocks_for(layout, n_rows)
    var global_ids = List[Int]()
    for f in range(len(layout.features)):
        global_ids.append(f)

    var target = TL2Target(weights^, weighted_target^, n_rows)
    var subsets = create_subsets(ctx, max_depth, target)
    var calcer = ScoresCalcerOnCompressedDataSet(
        ctx, blocks, n_rows, max_depth, global_ids
    )

    var structure = List[TBinarySplit]()
    var score_before_split = Float32(0.0)

    for depth in range(max_depth):
        # their `Gather(groupedByBinObservations, observations,
        # subsets.Indices)` (`:67`). At identity observations the gather IS
        # `subsets.Indices`; DEVIATION 105.
        var docs = subsets.indices.copy()
        calcer.submit_compute(
            ctx, subsets, cindex, docs, n_rows, sm_count, fixed_scale
        )
        var pstats = subsets.partition_stats.copy()
        calcer.compute_optimal_split(
            ctx,
            pstats,
            1 << depth,
            score_before_split,
            score_function,
            l2_leaf_reg,
            score_std_dev,
            seed,
        )

        # their fold (`:113-120`), incumbent FIRST so a tie takes the new
        var best = TBestSplitProperties()
        best = take_best(best, calcer.read_optimal_split(ctx))

        if best.feature_id < 0:
            raise Error(
                "best split is undefined at depth "
                + String(depth)
                + ": every candidate scored non-finite. Theirs raises the"
                " same way (`:122`)."
            )
        score_before_split = best.score

        var fid = Int(best.feature_id)
        ref cf = layout.features[fid]
        var is_one_hot = False
        if len(one_hot) == len(layout.features):
            is_one_hot = one_hot[fid]

        # `structure.HasSplit(bestSplit)` (`:134`), BEFORE applying it
        var seen = False
        for i in range(len(structure)):
            if (
                structure[i].feature_id == Int32(fid)
                and structure[i].bin_idx == best.bin_id
            ):
                seen = True
        if seen:
            break

        var docs2 = subsets.indices.copy()
        split_subsets(
            ctx,
            target,
            cindex,
            docs2,
            # `TCFeature::Offset` is an ELEMENT offset into the compressed
            # index -- `compressed_index[f_offset + doc]`. This tree's
            # layout stores `offset` as the COLUMN index and strides by
            # `n_rows`, so the multiply is the conversion. Passing the raw
            # column reads column 0's bits with this feature's shift and
            # mask: every document lands on one side, the partition comes
            # back [0, n] / [n, 0], and the level scores identically to its
            # parent -- so the searcher re-proposes the same split and
            # `HasSplit` stops the tree at depth 1. Which is exactly how
            # this was found.
            UInt32(Int(cf.offset) * n_rows),
            cf.mask,
            cf.shift,
            is_one_hot,
            UInt32(best.bin_id),
            subsets,
        )
        # `split_type`: TakeBin (one-hot equality) or TakeGreater
        structure.append(
            TBinarySplit(
                Int32(fid),
                best.bin_id,
                Int32(1) if is_one_hot else Int32(0),
            )
        )

    return structure^
