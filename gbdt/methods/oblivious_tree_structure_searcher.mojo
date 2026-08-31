# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The FEATURE-PARALLEL oblivious searcher, `SetTarget` arm, one device.

PORT OF `catboost/cuda/methods/oblivious_tree_structure_searcher.{h,cpp}` at
CatBoost `54a8143a` -- `TFeatureParallelObliviousTreeSearcher::Fit`
(`:46-306`), `::CreateSubsets` (`:29-44`) and
`TSubsetsHelper<NCudaLib::TMirrorMapping>::Split`
(`pointwise_optimization_subsets.h:74-93`). Transliterated. Do not improve.

RUNG 2. `NEXT_TWO.md` and `PORTING.md` 119 both priced this as "the fold
layout plus wiring, not a second searcher", on the strength of `PORTING.md`
91 B. **THAT PRICING IS WRONG AND THIS FILE IS WHY.** The correction is
below and it is the main result of this rung; the identity gate is the
second.

WHAT 91 B GETS RIGHT, AND THE THREE THINGS IT GETS WRONG
--------------------------------------------------------
91 B claims the two symmetric pointwise searchers "are one implementation
templated on the mapping, with two `TSubsetsHelper` specializations whose
`CreateSubsets` differ in exactly three lines". Read against their source:

**RIGHT.** The score path is genuinely shared and genuinely identical at one
device -- `TScoresCalcerOnCompressedDataSet`, `histograms_helper`,
`pointwise_kernels`, the whole `pointwise_hist2*` family, `TakeBest`,
`ToSplit`, the `HasSplit` stopping rule, `UpdateSubsetsStats`,
`ReorderBins`. Every one of those is reached from both `Fit` bodies with
arguments that coincide at `GetDeviceCount() == 1`. 91 A holds too: the
compressed index is bit-identical, so both searchers read the same bins.

**WRONG 1 -- `TSubsetsHelper<TMirrorMapping>` HAS NO `CreateSubsets`.** Look
at the specialization (`pointwise_optimization_subsets.h:72-105`): it
declares `Split` and two `CurrentPartsView` overloads and nothing else. Only
the Stripe specialization declares `CreateSubsets` (`:127-128`). The
feature-parallel `CreateSubsets` is a METHOD ON THE SEARCHER
(`oblivious_tree_structure_searcher.cpp:29`). There is no pair of
`CreateSubsets` implementations to differ in three lines.

**WRONG 2 -- THE TWO `Split`s ARE DIFFERENT CODE CALLING DIFFERENT
KERNELS.** This is the substantive one.

    Stripe   UpdateBinFromCompressedIndex(cindex, feature, bin, docsForBins,
                                          depth, Bins)
             ONE kernel, reading the compressed index at the gathered
             position (`pointwise_optimization_subsets.cpp:35-40`)

    Mirror   UpdateBins(Bins, nextLevelDocBins, docMap, CurrentDepth,
                        FoldBits)
             reads `docBins`, a SEPARATE per-document array that
             `TTreeUpdater` fills through a bit-packed intermediate
             (`pointwise_optimization_subsets.h:82`)

`docBins` has no counterpart on the doc-parallel path. Filling it is
`TTreeUpdater::AddSplit` -> `WriteCompressedSplit` ->
`UpdateBinFromCompressedBits`, which is `gbdt/methods/
oblivious_tree_bin_builder.mojo` and is ~350 lines of new port including two
kernels and a compression layout. So the feature-parallel arm runs THREE
kernels per level where the doc-parallel arm runs one, and the difference is
not a fold count.

**WRONG 3 -- THE `Split` CALL SITE DIFFERS, not just its body.** Their
feature-parallel `Fit` calls `treeUpdater.AddSplit(bestSplit)` (`:276`)
BEFORE `TSubsetsHelper<TMirrorMapping>::Split` (`:283`), and the bit
`AddSplit` writes is `BinarySplits.size()` -- exactly the `loadBit` the
following `Split` reads. Two statements, order-critical, present on one arm
only.

None of that falsifies 91 A, and none of it changes the ANSWER: at
`FoldBits == 0`, one device, one task and no CTR columns, the two chains
produce bit-identical `subsets.Bins` at every level and therefore the same
tree. That is what `original/feature_parallel_identity_check.mojo` gates.
What it falsifies is the PRICE. Rung 2 is not free.

WHY THEY BUILD `docBins` AT ALL
-------------------------------
Two reasons, both visible in their tree and neither about splitting:

1. `CacheBinsForModel(ScopedCache, FeaturesManager, DataSet, result,
   std::move(docBins))` (`:300-304`). `docBins` after the last level IS the
   leaf id of every document, in original document order, and the
   feature-parallel boosting loop's leaves estimator
   (`TObliviousTreeLeavesEstimator`) reads it. The doc-parallel searcher
   estimates its leaves inline from the partition stats it already has
   (`..._doc_parallel_...cpp:150-157`) and never needs a document-ordered
   array.
2. `TTreeUpdater` is also the tree-CTR path's tensor tracker
   (`CreateEmptyTensorTracker`, used at `:144`), and a tree CTR's tensor is
   "the splits already in this tree" -- which is what `BinarySplits` is.

This port returns `docBins` for the same reason (1) exists: it is the
model's leaf assignment and the identity gate checks it per document.

WHAT IS AND IS NOT HERE
-----------------------
DEVIATION 120 covers the four things their `Fit` does that this function
does not: `ComputeWeakTarget`, the bootstrap, the tree-CTR block, and the
per-level rebuild of `docIndices`. The first two mirror the doc-parallel
port's DEVIATION 104 exactly and for the same reason -- the boosting loop
owns the gradient path, and forking it is the one thing that must not differ
between two learners being compared.

The `AddTask` / `FoldBasedTasks` arm is NOT here. This is the `SetTarget`
arm, `FoldBits == 0`, which is what rung 2 is scoped to; the fold layout it
would need is `gbdt/methods/oblivious_tree_fold_tasks.mojo` and wiring it is
rung 3.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from gbdt.gpu_data.compressed_index_builder import CompressedIndexLayout
from gbdt.gpu_data.feature_blocks import blocks_for
from gbdt.gpu_util.kernel.fill import launch_make_sequence
from gbdt.gpu_util.kernel.radix_sort import launch_radix_sort_bins
from gbdt.gpu_util.kernel.transform import launch_gather_with_mask_u32
from gbdt.methods.helpers import TBestSplitProperties, take_best
from gbdt.methods.oblivious_tree_bin_builder import TTreeUpdater
from gbdt.methods.oblivious_tree_fold_tasks import plan_single_task_layout
from gbdt.methods.pointwise_kernels import update_fold_bins
from gbdt.methods.pointwise_optimization_subsets import (
    GATHER_NO_MASK,
    TL2Target,
    TOptimizationSubsets,
    create_subsets,
    update_subsets_stats,
)
from gbdt.methods.pointwise_scores_calcer import ScoresCalcerOnCompressedDataSet
from gbdt.models.oblivious_model import (
    BIN_SPLIT_TAKE_BIN,
    BIN_SPLIT_TAKE_GREATER,
    TBinarySplit,
)


def split_subsets_mirror(
    ctx: DeviceContext,
    mut source: TL2Target,
    mut next_level_doc_bins: DeviceBuffer[DType.uint32],
    mut doc_map: DeviceBuffer[DType.uint32],
    mut subsets: TOptimizationSubsets,
) raises:
    """`TSubsetsHelper<NCudaLib::TMirrorMapping>::Split`
    (`pointwise_optimization_subsets.h:74-93`), call for call.

        UpdateBins(subsets->Bins, nextLevelDocBins, docMap,
                   subsets->CurrentDepth, subsets->FoldBits);
        ReorderBins(subsets->Bins, subsets->Indices,
                    subsets->CurrentDepth + subsets->FoldBits, 1);
        ++subsets->CurrentDepth;
        UpdateSubsetsStats(sourceTarget, subsets);

    The last three statements are byte for byte the Stripe specialization's
    last three (`pointwise_optimization_subsets.cpp:42-50`). **THE FIRST IS
    NOT**, and it is the whole of WRONG 2 in the module docstring.

    `UpdateBins` here is `NKernel::UpdateFoldBins`
    (`methods/kernel/pointwise_hist2.cu:16-34`), already ported as
    `gbdt/methods/pointwise_kernels.update_fold_bins`:

        idx = docIndices[i];
        bit = (bins[idx] >> loadBit) & 1;
        dstBins[i] = dstBins[i] | (bit << (loadBit + foldBits));

    **NOTE THE TWO DIFFERENT BIT POSITIONS.** It reads bit `CurrentDepth` of
    `docBins` and writes bit `CurrentDepth + FoldBits` of `subsets.Bins`,
    because `docBins` carries no fold id -- it is one array per DOCUMENT and
    a document belongs to every fold it appears in -- while `subsets.Bins`
    packs the fold id in the low bits. At `FoldBits == 0` the two coincide,
    which is exactly why this rung can be an identity and rung 3 cannot.

    `docMap` is `observationIndices`, NOT `subsets.Indices`. It is
    `docIndices[subsets.Indices]` -- position in partition order to ORIGINAL
    document id -- and `docBins` is indexed by original document id. Passing
    `subsets.Indices` instead reads `docBins` at a position rather than at a
    document, which is the identity only while the permutation is.

    THE THREE STATEMENTS AFTER IT ARE THE STRIPE ARM'S AND EVERY WARNING ON
    THEM CARRIES OVER: `ReorderBins(..., offset, 1)` is one bit and must be
    STABLE, and `++CurrentDepth` happens BEFORE `UpdateSubsetsStats` so the
    reduce sees the new, twice-as-wide level. See
    `pointwise_optimization_subsets.split_subsets` for both.
    """
    var depth = subsets.current_depth + subsets.fold_bits
    if Int(depth) >= 32:
        raise Error(
            String("Split at depth ")
            + String(depth)
            + " would write bit "
            + String(depth)
            + " of a ui32 bin; CatBoost's ReorderBins asserts"
            " (offset + bits) <= 32 (cuda_util/sort.cpp:557)"
        )

    # `UpdateBins(subsets->Bins, nextLevelDocBins, docMap, CurrentDepth,
    #  FoldBits)` -- their Mirror-only first statement.
    update_fold_bins(
        ctx,
        subsets.bins.unsafe_ptr(),
        next_level_doc_bins.unsafe_ptr(),
        doc_map.unsafe_ptr(),
        subsets.doc_count,
        Int(subsets.current_depth),
        Int(subsets.fold_bits),
    )

    # `ReorderBins(subsets->Bins, subsets->Indices, depth, 1)`
    launch_radix_sort_bins(
        ctx,
        subsets.doc_count,
        Int(depth),
        Int(depth) + 1,
        subsets.bins,
        subsets.indices,
        subsets.tmp_bins,
        subsets.tmp_indices,
        subsets.scan_offsets,
        subsets.block_sums,
    )

    # `++subsets->CurrentDepth;`
    subsets.current_depth += 1

    # `UpdateSubsetsStats(sourceTarget, subsets);`
    update_subsets_stats(ctx, source, subsets)


def create_subsets_single_task(
    ctx: DeviceContext,
    max_depth: Int,
    mut source: TL2Target,
) raises -> TOptimizationSubsets:
    """`TFeatureParallelObliviousTreeSearcher::CreateSubsets` (`:29-44`),
    the `SetTarget` arm of the ternary.

        auto initParts = SingleTaskTarget == nullptr
                       ? WriteFoldBasedInitialBins(subsets.Bins)
                       : WriteSingleTaskInitialBins(subsets.Bins);
        subsets.Indices = TMirrorBuffer<ui32>::CopyMapping(subsets.Bins);
        subsets.CurrentDepth = 0;
        subsets.FoldCount = initParts.size();
        subsets.FoldBits  = NCB::IntLog2(subsets.FoldCount);
        MakeSequence(subsets.Indices);
        ui32 maxPartCount = 1 << (subsets.FoldBits + maxDepth);
        subsets.Partitions.Reset(...);
        subsets.PartitionStats.Reset(...);
        UpdateSubsetsStats(src, &subsets);

    `WriteSingleTaskInitialBins` (`:366-375`) is `FillBuffer(bins, 0u)` and
    one partition `{0, n}`, so `FoldCount == 1` and
    `FoldBits == IntLog2(1) == 0`. `plan_single_task_layout` is that
    function's host half and is reused rather than respelled.

    **THE ONE FIELD THAT DIFFERS FROM THE DOC-PARALLEL ARM IS `FoldCount`,
    AND IT IS INERT.** Stripe hardcodes `FoldCount = 0`
    (`pointwise_optimization_subsets.cpp:12`); Mirror computes
    `initParts.size() == 1`. Nothing downstream reads `FoldCount` on this
    path -- the score calcer takes its fold count as a constructor
    argument, where their feature-parallel `Fit` passes `subsets.FoldCount`
    (`:93`) and their doc-parallel one passes the literal `1` (`:40`), which
    are the same number. `FoldBits`, which IS read everywhere, is 0 on both.
    So the two arms' subsets are the same state with one differing
    unread counter, and that is the whole of the "three lines".
    """
    var layout = plan_single_task_layout(source.line_size)
    var subsets = create_subsets(
        ctx, max_depth, source, layout.fold_count, layout.fold_bits
    )
    # `WriteSingleTaskInitialBins`'s `FillBuffer(bins, 0u)` is already
    # `create_subsets`'s memset, and its single `{0, n}` partition is what a
    # zeroed bin array reduces to. No second fill, and no second
    # `UpdateSubsetsStats` with it -- unlike the FOLD arm, which has to
    # overwrite the bins after the fact (that arm's DEVIATION 125).
    return subsets^


def fit_feature_parallel_oblivious_tree_structure(
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
) raises -> Tuple[List[TBinarySplit], DeviceBuffer[DType.uint32]]:
    """`TFeatureParallelObliviousTreeSearcher::Fit` (`:46-306`), the
    `SetTarget` arm at one device.

    Returns their `TObliviousTreeStructure` AND `docBins` -- their
    `CacheBinsForModel(ScopedCache, FeaturesManager, DataSet, result,
    std::move(docBins))` (`:300-304`), which is the model's per-document
    leaf id in ORIGINAL document order. The doc-parallel searcher has no
    such array to return.

    The argument list is deliberately the doc-parallel searcher's, so the
    identity gate calls both with one set of values and a mismatch cannot be
    a mismatch of inputs.

    THERE IS NO SABOTAGE HOOK IN THIS FUNCTION and that is deliberate:
    `PORTING_RULES.md` 8 says a switch that outlives its measurement is a
    defect, and a permanently wired defect selector is one. The reach
    evidence for this file is five defects planted by EDITING it and re-run,
    tabulated in `original/feature_parallel_identity_check.mojo`'s
    docstring and in `PORTING.md` 120.
    """
    # `CB_ENSURE(FoldBasedTasks.size() || SingleTaskTarget);` (`:47`)
    if n_rows <= 0:
        raise Error(
            "Fit with no documents; their CB_ENSURE at `:47` requires a"
            " target to have been set"
        )

    var blocks = blocks_for(layout, n_rows)
    var global_ids = List[Int]()
    for f in range(len(layout.features)):
        global_ids.append(f)

    var target = TL2Target(weights^, weighted_target^, n_rows)

    # `TMirrorBuffer<ui32> docBins = CopyMapping(DataSet.GetIndices());`
    # (`:49`) and `TTreeUpdater treeUpdater(..., docBins);` (`:51-55`).
    # The updater OWNS docBins here, because their `TTreeUpdater` takes it
    # by reference and is the only writer.
    var tree_updater = TTreeUpdater(ctx, n_rows)

    # `TL2Target target = ComputeWeakTarget();` (`:57`) and the bootstrap
    # block (`:59-73`) -- DEVIATION 120, both arrive done.

    # `auto subsets = CreateSubsets(TreeConfig.MaxDepth, target);` (`:75`)
    var subsets = create_subsets_single_task(ctx, max_depth, target)

    # `auto observationIndices = CopyMapping(subsets.Indices);` (`:78`)
    var observation_indices = ctx.enqueue_create_buffer[DType.uint32](n_rows)

    # `MakeDocIndicesForSingleTask` (`:469-474`) is
    # `indices.Copy(SingleTaskTarget->GetTarget().GetIndices())`. Their
    # `MakeDocIndices` (`:476-498`) is called INSIDE the depth loop (`:124`)
    # and allocates a fresh buffer each level; the array it copies does not
    # change across levels, so it is hoisted here. DEVIATION 120: same
    # values, one allocation instead of `MaxDepth` of them.
    #
    # At permutation id 0 with no shuffle the target's indices are the
    # identity, which is the doc-parallel port's DEVIATION 105 assumption
    # holding on this arm too -- and unlike that arm the gather below is
    # still performed rather than collapsed, because `docBins` is indexed by
    # DOCUMENT and the collapse would only be legal for `subsets.Indices`.
    var doc_indices = ctx.enqueue_create_buffer[DType.uint32](n_rows)
    launch_make_sequence(ctx, UInt32(0), doc_indices, n_rows)

    # `featuresScoreCalcer = MakeHolder<...>(DataSet.GetFeatures(),
    #  TreeConfig, foldCount, true);` (`:90-95`), with
    # `foldCount = subsets.FoldCount` (`:83`) == 1.
    var calcer = ScoresCalcerOnCompressedDataSet(
        ctx, blocks, layout, n_rows, max_depth, global_ids
    )

    var structure = List[TBinarySplit]()
    var score_before_split = Float32(0.0)

    for depth in range(max_depth):
        # `MakeDocIndices(docIndices); Gather(observationIndices,
        #  docIndices, subsets.Indices);` (`:122-126`)
        launch_gather_with_mask_u32(
            ctx,
            observation_indices,
            doc_indices,
            subsets.indices,
            n_rows,
            GATHER_NO_MASK,
        )

        # the tree-CTR block (`:136-147`, `:201-255`) -- DEVIATION 120.

        # `featuresScoreCalcer->SubmitCompute(subsets, observationIndices);`
        # (`:158-159`). **THE SECOND ARGUMENT IS `observationIndices`, NOT
        # `subsets.Indices`** -- theirs, and it is the same array the
        # doc-parallel arm passes as `groupedByBinObservations` (`:82`).
        calcer.submit_compute(
            ctx,
            subsets,
            cindex,
            observation_indices,
            n_rows,
            sm_count,
            fixed_scale,
        )

        # `featuresScoreCalcer->ComputeOptimalSplit(partitionsStats, ...)`
        # (`:169-174`). `partitionsStats` is `subsets.PartitionStats` taken
        # at `:119`, BEFORE the level's work -- a mirror buffer needs no
        # all-reduce, which is the one line the doc-parallel arm has here
        # and this one does not (`..._doc_parallel_...cpp:70`).
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

        # `TBestSplitProperties bestSplitProp = {(ui32)-1, 0, inf, inf};`
        # then `TakeBest(bestSplitProp, calcer->ReadOptimalSplit())`
        # (`:188-195`) -- incumbent FIRST, so a full tie takes the NEW
        # candidate. The calcer's own fold goes the other way; both are
        # theirs (`NEXT_TWO.md` TRAPS).
        var best = TBestSplitProperties()
        best = take_best(best, calcer.read_optimal_split(ctx))

        # `scoreBeforeSplit = bestSplitProp.Score;` (`:257`) -- BEFORE the
        # CB_ENSURE, theirs.
        score_before_split = best.score

        # `CB_ENSURE(bestSplitProp.FeatureId != (ui32)-1, ...)` (`:259-261`)
        if best.feature_id < 0:
            raise Error(
                "best split is undefined at depth "
                + String(depth)
                + ": every candidate scored non-finite. Theirs raises the"
                " same way (`:259`)."
            )

        var fid = Int(best.feature_id)
        ref cf = layout.features[fid]
        var is_one_hot = False
        if len(one_hot) == len(layout.features):
            is_one_hot = one_hot[fid]

        # `bestSplit = ToSplit(FeaturesManager, bestSplitProp);` (`:263`).
        # `BIN_SPLIT_TAKE_BIN` is 0 and `BIN_SPLIT_TAKE_GREATER` is 1, the
        # opposite of what the names suggest (`oblivious_model.mojo:36-37`).
        var best_split = TBinarySplit(
            Int32(fid),
            best.bin_id,
            Int32(BIN_SPLIT_TAKE_BIN) if is_one_hot else Int32(
                BIN_SPLIT_TAKE_GREATER
            ),
        )

        # `if (result.HasSplit(bestSplit)) { break; }` (`:266-268`), BEFORE
        # the split is applied, so the structure never holds a duplicate.
        # Theirs breaks with no leaf estimation at all; the doc-parallel arm
        # estimates leaves first (`..._doc_parallel_...cpp:135`), which is
        # the DEVIATION 104 half neither port carries.
        var seen = False
        for i in range(len(structure)):
            if (
                structure[i].feature_id == best_split.feature_id
                and structure[i].bin_idx == best_split.bin_idx
            ):
                seen = True
        if seen:
            break

        # `treeUpdater.AddSplit(bestSplit);` (`:276`). **BEFORE THE SUBSETS
        # SPLIT AND THAT IS LOAD-BEARING**: it writes bit
        # `BinarySplits.size()` == `depth` of `docBins`, which is the exact
        # bit `split_subsets_mirror` reads as `loadBit` two statements
        # later.
        # `TCFeature::Offset` is an ELEMENT offset into the compressed
        # index; this tree's layout stores a COLUMN index and strides by
        # `n_rows`, so the multiply is the conversion. It is the same
        # conversion the doc-parallel searcher's `split_subsets` call makes,
        # and its comment records what dropping it looks like.
        var feature_offset = UInt32(Int(cf.offset) * n_rows)
        tree_updater.add_split(
            ctx,
            cindex,
            doc_indices,
            False,
            best_split,
            feature_offset,
            cf.mask,
            cf.shift,
            is_one_hot,
        )

        # `if ((depth + 1) != TreeConfig.MaxDepth) { TSubsetsHelper<Mirror>
        #  ::Split(target, docBins, observationIndices, &subsets); }`
        # (`:280-287`). The doc-parallel arm adds `|| needLeavesEstimation`
        # to this condition because it estimates leaves from the split it is
        # about to skip; DEVIATION 104 removed leaves from that port, so the
        # two conditions coincide.
        if (depth + 1) != max_depth:
            split_subsets_mirror(
                ctx,
                target,
                tree_updater.learn_bins,
                observation_indices,
                subsets,
            )

        # `result.Splits.push_back(bestSplit);` (`:294`)
        structure.append(best_split)

    # `CacheBinsForModel(ScopedCache, FeaturesManager, DataSet, result,
    #  std::move(docBins));` (`:300-304`)
    # `.copy()` is a HANDLE copy -- the same device allocation, the way
    # `subsets.indices.copy()` is used throughout this family. Moving the
    # field out of `tree_updater` instead is refused: Mojo will not destroy
    # a struct one of whose fields has already been moved from.
    var doc_bins = tree_updater.learn_bins.copy()
    return (structure^, doc_bins^)
