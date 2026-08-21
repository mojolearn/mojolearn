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

DEVIATION 127: **THIS FILE'S FOLD ARM HAS NO UPSTREAM COUNTERPART, and it
is here because rung 3 was wired before rung 2 landed.** Upstream's
doc-parallel `CreateSubsets` hard-codes `FoldCount = 0; FoldBits = 0`
(`pointwise_optimization_subsets.cpp:12-14`) and nothing can give it folds:
`TDocParallelObliviousTreeSearcher` is built by `TDocParallelObliviousTree`,
which `TBoosting` (`doc_parallel_boosting.h`) drives, and that is the PLAIN
learner (`PORTING.md` 91 F). Ordered boosting lives ONLY in
`TFeatureParallelObliviousTreeSearcher`, which is
`gbdt/methods/oblivious_tree_structure_searcher.mojo`.

So the `folds` parameter below is a DEVIATION to be DELETED, not a feature.
What is NOT a deviation is everything under it -- `create_fold_based_subsets`,
`make_fold_doc_indices`, the fold stripe, the histogram fold axis and the
dynamic scorer -- because both searchers share that stack (`PORTING.md` 91 B)
and it is ported from the feature-parallel side. Moving the arm is three
lines in the other searcher's `Fit`; until someone does, this one refuses at
DEVIATION 126 anyway and can never grow a tree.

DEVIATION 105: `observations` must be the identity. Theirs gathers
`groupedByBinObservations = observations[subsets.Indices]` where
`observations` is the bootstrapped, filtered doc list; with no bootstrap
that is the identity and the gather collapses to `subsets.Indices` itself.
This file RAISES on a non-identity request rather than silently using the
wrong array, and the argument is there so the signature does not change when
bootstrap arrives.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from gbdt.data.permutation import TRandom
from gbdt.gpu_data.feature_blocks import PolicyBlock, blocks_for
from gbdt.gpu_data.compressed_index_builder import CompressedIndexLayout
from gbdt.methods.helpers import TBestSplitProperties, take_best
from gbdt.methods.pointwise_optimization_subsets import (
    TL2Target,
    TOptimizationSubsets,
    create_subsets,
    reset_subsets,
    split_subsets,
)
from gbdt.methods.pointwise_scores_calcer import ScoresCalcerOnCompressedDataSet
from gbdt.gpu_util.kernel.transform import (
    launch_gather_with_mask_u32,
    launch_split_planes_f32,
)
from gbdt.methods.pointwise_optimization_subsets import GATHER_NO_MASK
from gbdt.methods.dynamic_boosting_folds import TFold
from gbdt.methods.kernel.pointwise_scores import (
    SCORE_FUNCTION_COSINE,
    SCORE_FUNCTION_NEWTON_COSINE,
    SCORE_FUNCTION_SOLAR_L2,
)
from gbdt.methods.oblivious_tree_fold_tasks import (
    FoldLayout,
    create_fold_based_subsets,
    fold_tasks_from_folds,
    make_fold_doc_indices,
    plan_fold_layout,
    plan_single_task_layout,
)
from gbdt.models.oblivious_model import (
    BIN_SPLIT_TAKE_BIN,
    BIN_SPLIT_TAKE_GREATER,
    TBinarySplit,
)


struct PointwiseTreeWorkspace(Movable):
    """DEVIATION 143: the pointwise searcher's CROSS-TREE POOL.

    CatBoost never rebuilds this state per tree -- its `TCudaManager`
    memory pool hands `CreateSubsets` and the score helpers recycled
    device memory, so their per-tree cost is a handful of fills. This
    port has no manager, and constructing fresh was measured at 17-26
    ms/tree (PREP_BILL step 21: ~17 buffer allocations in
    `create_subsets`, ~10 more plus four layout uploads AND a full
    drain per `PolicyScoreHelper`, per tree). Same shape as the greedy
    family's `TTreeWorkspace`, for the same reason: a POOL OF ONE,
    owned by the caller for the span of a fit, keyed on the shapes that
    size the buffers, rebuilt whole on any mismatch.

    A `List` of at most one element rather than an `Optional` for the
    reason `TTreeWorkspace` records: the caller declares an empty list,
    the first tree fills it, later trees hit the key check and reset
    instead.

    ONLY THE SINGLE-TASK ARM POOLS. `fold_count > 1` (ordered boosting)
    stores its fresh build here too -- one location for the `ref`
    bindings -- but its key can never hit (`fold_count_key > 1` fails
    the check), so every fold-arm tree constructs exactly as before.
    Pooling that arm needs `write_fold_based_initial_bins` replayed
    after the reset and a gate that holds it; neither exists yet, and
    an arm that silently skipped its fold seeding would be
    [[reached-but-inert]] in the worst way.

    The per-tree reset contract is CONSTRUCTOR POSTCONDITIONS
    (`reset_subsets` + `reset_for_tree`), held bit-exactly by
    `mojo_only/pointwise_pool_check.mojo` and end-to-end by the
    pooled-vs-fresh identity of whole fits.
    """

    var subsets: TOptimizationSubsets
    var calcer: ScoresCalcerOnCompressedDataSet
    var doc_count_key: Int
    var n_rows_key: Int
    var max_depth_key: Int
    var n_features_key: Int
    var fold_count_key: Int

    def __init__(
        out self,
        var subsets: TOptimizationSubsets,
        var calcer: ScoresCalcerOnCompressedDataSet,
        doc_count: Int,
        n_rows: Int,
        max_depth: Int,
        n_features: Int,
        fold_count: Int,
    ):
        self.subsets = subsets^
        self.calcer = calcer^
        self.doc_count_key = doc_count
        self.n_rows_key = n_rows
        self.max_depth_key = max_depth
        self.n_features_key = n_features
        self.fold_count_key = fold_count


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
    mut pool: List[PointwiseTreeWorkspace],
    l2_leaf_reg: Float32 = Float32(3.0),
    score_std_dev: Float32 = Float32(0.0),
    seed: UInt64 = 0,
    one_hot: List[Bool] = List[Bool](),
    bootstrapped_observations: Bool = False,
    folds: List[TFold] = List[TFold](),
    permutation: List[UInt32] = List[UInt32](),
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

    # ---------------------------------------------------------------
    # THE TERNARY (`oblivious_tree_structure_searcher.cpp:30-31`).
    #
    #     SingleTaskTarget == nullptr ? WriteFoldBasedInitialBins(...)
    #                                 : WriteSingleTaskInitialBins(...)
    #
    # An empty `folds` is `SetTarget` -- ONE task, `FoldCount = 1`,
    # `FoldBits = 0`, every document in bin 0. A non-empty one is `AddTask`
    # per fold -- N tasks, 2N alternating learn/test partitions,
    # `FoldCount = 2N`, `FoldBits = IntLog2(2N)`, and the fold id in the LOW
    # bits of every document's bin.
    # ---------------------------------------------------------------
    var fold_layout = plan_single_task_layout(n_rows)
    if len(folds) > 0:
        fold_layout = plan_fold_layout(fold_tasks_from_folds(folds))
    var fold_count = fold_layout.fold_count
    var doc_count = fold_layout.total_indices

    if fold_count > 1:
        # `FindOptimalSplitDynamic` (`pointwise_scores.cu:443-473`) has TWO
        # arms and a `default: throw std::exception()`. Four of the seven
        # score functions -- L2, NewtonL2, SatL2, LOOL2 -- have no
        # ordered-boosting kernel upstream at all. Refused HERE rather than
        # at the launch so the refusal names the option the caller set
        # instead of a kernel it never asked for, and so a tree is never
        # half-grown before it fires.
        if not (
            score_function == SCORE_FUNCTION_SOLAR_L2
            or score_function == SCORE_FUNCTION_COSINE
            or score_function == SCORE_FUNCTION_NEWTON_COSINE
        ):
            raise Error(
                "ordered boosting (fold_count "
                + String(fold_count)
                + ") supports only SolarL2, Cosine and NewtonCosine:"
                " `FindOptimalSplitDynamic` (`pointwise_scores.cu:469`)"
                " throws for every other score function. Got score"
                " function "
                + String(score_function)
            )
        if len(permutation) != 0 and len(permutation) != n_rows:
            raise Error(
                "permutation must be empty (identity) or have one entry"
                " per row"
            )

    var blocks = blocks_for(layout, n_rows)
    var global_ids = List[Int]()
    for f in range(len(layout.features)):
        global_ids.append(f)

    # `n_rows` is the compressed index's ROW STRIDE and `doc_count` is the
    # length of the CONCATENATED document array. They are the same number
    # at one task and different at N, because the fold estimate slices are
    # nested prefixes -- see `fold_tasks_from_folds`.
    var target = TL2Target(weights^, weighted_target^, doc_count)

    # ---- DEVIATION 143: reset the pooled pair, or build it -------------
    # The key check is the whole dispatch; see `PointwiseTreeWorkspace`
    # for why the fold arm can never hit.
    var pooled_hit = (
        len(pool) != 0
        and fold_count == 1
        and pool[0].fold_count_key == 1
        and pool[0].doc_count_key == doc_count
        and pool[0].n_rows_key == n_rows
        and pool[0].max_depth_key == max_depth
        and pool[0].n_features_key == len(layout.features)
    )
    if pooled_hit:
        reset_subsets(ctx, pool[0].subsets, target)
        pool[0].calcer.reset_for_tree(ctx)
    else:
        pool.clear()
        var subsets_new = create_subsets(ctx, max_depth, target) if (
            fold_count == 1
        ) else create_fold_based_subsets(
            ctx, max_depth, target, fold_layout
        )
        var calcer_new = ScoresCalcerOnCompressedDataSet(
            ctx, blocks, layout, n_rows, max_depth, global_ids
        )
        pool.append(
            PointwiseTreeWorkspace(
                subsets_new^, calcer_new^, doc_count, n_rows, max_depth,
                len(layout.features), fold_count,
            )
        )
    ref subsets = pool[0].subsets
    ref calcer = pool[0].calcer

    # DEVIATION 126, and it dissolves the moment the calcer carries folds.
    # `TScoreHelper` takes `foldCount` and hands it to BOTH halves
    # (`histograms_helper.h:361-365`): `TComputeHistogramsHelper` sizes the
    # histogram `(1 << MaxDepth) * FoldCount * binFeatures * 2` and passes
    # it to `ComputeHistogram2` as `gridDim.z`, and
    # `TFindBestSplitsHelper` passes it to `FindOptimalSplit`, whose
    # `foldCount == 1` test IS the dispatch between the plain and the
    # dynamic scorer. This tree's `PolicyScoreHelper` hard-codes 1 at all
    # three sites. Rather than grow a tree whose histograms are a fold
    # axis short, ask the calcer what it is carrying and refuse if it
    # disagrees with the layout.
    for i in range(len(calcer.helpers)):
        if calcer.helpers[i].hist_helper.fold_count != fold_count:
            raise Error(
                "DEVIATION 126: the fold layout has FoldCount "
                + String(fold_count)
                + " but `PolicyScoreHelper` was built at "
                + String(calcer.helpers[i].hist_helper.fold_count)
                + ". `TScoreHelper` takes foldCount"
                " (`histograms_helper.h:361`) and this port hard-codes 1"
                " at `pointwise_scores_calcer.mojo`\'s"
                " `ComputeHistogramsHelper(policy, 1, max_depth)`,"
                " `compute_hist2(..., plan.part_count, 1, ...)` and"
                " `find_optimal_split(..., part_count, 1, ...)`."
            )

    # `Gather(observationIndices, docIndices, subsets.Indices)` (`:120`,
    # and `MakeDocIndices` at `:485-505` builds `docIndices`). At ONE task
    # `docIndices` is the identity and the gather collapses to
    # `subsets.Indices` (DEVIATION 105); at N tasks it does not, and a
    # searcher that skipped it would read the compressed index at a
    # POSITION in the concatenated array instead of at a document id.
    var doc_ids_host = make_fold_doc_indices(folds, permutation) if (
        fold_count > 1
    ) else List[UInt32]()
    var d_doc_ids = ctx.enqueue_create_buffer[DType.uint32](
        len(doc_ids_host) if fold_count > 1 else 1
    )
    var d_observations = ctx.enqueue_create_buffer[DType.uint32](doc_count)
    if fold_count > 1:
        if len(doc_ids_host) != doc_count:
            raise Error(
                "MakeDocIndices produced "
                + String(len(doc_ids_host))
                + " ids for "
                + String(doc_count)
                + " concatenated documents"
            )
        ctx.enqueue_copy(
            dst_buf=d_doc_ids, src_ptr=doc_ids_host.unsafe_ptr()
        )
        ctx.synchronize()
        # keep the host list alive across the queue: a raw pointer does not
        # ([[mojo-buffer-freed-at-last-use]])
        _ = doc_ids_host[0]

    var structure = List[TBinarySplit]()
    var score_before_split = Float32(0.0)

    # `auto& random = objective.GetRandom()` (`:15`), drawn from ONCE PER
    # LEVEL at the two `ComputeOptimalSplit` call sites (`:86`, `:104`).
    #
    # THIS WAS A REAL BUG AND IT WAS INVISIBLE. The level loop below used to
    # hand `seed` itself to every level, so every level of a tree drew the
    # SAME per-feature normal -- the noise would have been a fixed
    # per-feature offset for the whole tree instead of a fresh draw per
    # level. Nothing caught it because no caller ever passed a non-zero
    # `score_std_dev`, which is exactly PORTING_RULES 8: a branch nothing
    # reaches is a branch nothing checks.
    #
    # DEVIATION 139: theirs is one `TGpuAwareRandom` for the whole fit and
    # this one is re-seeded per tree from the caller's `seed`.
    var level_rand = TRandom(seed)

    for depth in range(max_depth):
        # their `Gather(groupedByBinObservations, observations,
        # subsets.Indices)` (`:67`). At identity observations the gather IS
        # `subsets.Indices`; DEVIATION 105.
        var docs = subsets.indices.copy()
        if fold_count > 1:
            launch_gather_with_mask_u32(
                ctx,
                d_observations,
                d_doc_ids,
                docs,
                doc_count,
                GATHER_NO_MASK,
            )
            docs = d_observations.copy()
        calcer.submit_compute(
            ctx, subsets, cindex, docs, doc_count, sm_count, fixed_scale
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
            level_rand.next_uniform_l(),
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

        # their `Split(target, docBins, observationIndices, &subsets)`
        # (`oblivious_tree_structure_searcher.cpp:275-278`) -- the SAME
        # gathered array the histograms just read, because
        # `UpdateBinFromCompressedIndex` indexes the compressed index by
        # `docsForBins[i]` and not by `i`.
        var docs2 = d_observations.copy() if fold_count > 1 else (
            subsets.indices.copy()
        )
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
        # `split_type`, and the constants are NOT in the order the names
        # suggest: `BIN_SPLIT_TAKE_BIN` is 0 and `BIN_SPLIT_TAKE_GREATER`
        # is 1 (`oblivious_model.mojo:36-37`). Writing `1 if one_hot else 0`
        # makes every ORDINARY feature an equality test, which still grows a
        # well-formed tree of the right depth with the right splits and
        # partitions the rows completely differently -- 8 non-empty leaves
        # instead of 12. That is how this was found: identical structure,
        # different leaf values.
        structure.append(
            TBinarySplit(
                Int32(fid),
                best.bin_id,
                Int32(BIN_SPLIT_TAKE_BIN) if is_one_hot else Int32(
                    BIN_SPLIT_TAKE_GREATER
                ),
            )
        )

    return structure^


def split_stat_planes(
    ctx: DeviceContext,
    mut stats: DeviceBuffer[DType.float32],
    n_rows: Int,
) raises -> Tuple[
    DeviceBuffer[DType.float32], DeviceBuffer[DType.float32]
]:
    """Two columns of one buffer into two buffers, because they have to be.

    NO CATBOOST COUNTERPART -- upstream `TL2Target` is already two separate
    `TCudaBuffer<float>` and no split is needed. It exists because THIS tree
    carries the weak target as one two-plane buffer everywhere else
    (`greedy_search_helper`'s `stats`, plane 0 the weight and plane 1 the
    gradient), and the pointwise kernels cannot take two views of one
    buffer: they declare `target` and `weight` on independent origins and
    Mojo refuses the aliasing at `enqueue_function` itself (DEVIATION 97.2,
    `PORTING_RULES` 4).

    So this is a BRIDGE between two internal conventions, not a port of
    anything. It cost a full round trip through HOST memory when it was
    written -- `enqueue_copy` has no device-to-device form taking a source
    pointer -- which at 800k rows was 6.4 MB down and back per tree plus
    TWO drains the greedy arm never makes. `split_planes_f32_kernel` is
    the repair: the same split as one device launch, no host copy, no
    drain. What remains is two `n_rows` allocations per tree, which the
    pool does not yet own because `TL2Target` CONSUMES these buffers.

    It all disappears the moment the boosting loop carries the weak
    target as two buffers throughout, which is the right fix and is not
    attempted here because `stats` is read by the greedy searcher, the
    estimator and the bootstrap.
    """
    var w = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var t = ctx.enqueue_create_buffer[DType.float32](n_rows)
    launch_split_planes_f32(ctx, w, t, stats, n_rows, n_rows)
    return (w^, t^)
