# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The doc-parallel oblivious searcher: one split per level, all leaves at once.

Reference: `catboost/cuda/methods/
oblivious_tree_doc_parallel_structure_searcher.{h,cpp}` (CatBoost
`54a8143a`), `TDocParallelObliviousTreeSearcher::FitImpl`.

**THIS IS THE FIRST CALLER OF THE POINTWISE FAMILY.** Everything under it --
six accumulators, three drivers, the host launch layer, the scorer, the
subsets, the histogram state machine -- was correct and reached by nothing
until this file existed (`archive/plans/UNWIRED.md`).

It is also the learner CatBoost runs for single-target symmetric trees at
`boosting_type=Plain`, which is the arm every matched
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
learner. Ordered boosting lives ONLY in
`TFeatureParallelObliviousTreeSearcher`, which is
`gbdt/methods/oblivious_tree_structure_searcher.mojo`.

So the `folds` parameter below is a DEVIATION to be DELETED, not a feature.
What is NOT a deviation is everything under it -- `create_fold_based_subsets`,
`make_fold_doc_indices`, the fold stripe, the histogram fold axis and the
dynamic scorer -- because both searchers share that stack
and it is implemented from the feature-parallel side. Moving the arm is three
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

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from std.gpu import block_dim, block_idx, thread_idx
from std.os import getenv

from core.identity_trace import IdentityTrace
from gbdt.data.permutation import TRandom
from gbdt.methods.greedy_subsets_searcher.depthwise_stage_times import (
    StageTimes,
)
from gbdt.methods.histograms_helper import policy_name
from gbdt.gpu_data.feature_blocks import PolicyBlock, blocks_for
from gbdt.gpu_data.compressed_index_builder import CompressedIndexLayout
from gbdt.methods.pointwise_optimization_subsets import (
    TL2Target,
    TOptimizationSubsets,
    create_subsets,
    reset_subsets,
    split_subsets_from_desc,
    update_subsets_stats,
)
from gbdt.methods.kernel.pointwise_split_resolve import (
    PW_FUSED_SEARCH,
    PW_SENTINEL_ID,
    launch_pw_pack_winner,
    pw_resolve_pack_bins_kernel,
)
from gbdt.methods.pointwise_optimization_subsets import (
    SPLIT_BLOCK_SIZE as PW_SPLIT_BLOCK_SIZE,
    SPLIT_MAX_BLOCKS as PW_SPLIT_MAX_BLOCKS,
)
from gbdt.methods.pointwise_scores_calcer import ScoresCalcerOnCompressedDataSet
from gbdt.gpu_util.kernel.transform import (
    launch_gather_with_mask_u32,
    launch_split_planes_f32,
)
from gbdt.methods.pointwise_optimization_subsets import GATHER_NO_MASK
from gbdt.methods.dynamic_boosting_folds import TFold
from gbdt.methods.ordered_fast_switches import ORD_ALL
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
    write_fold_based_initial_bins,
)
from gbdt.models.oblivious_model import (
    BIN_SPLIT_TAKE_BIN,
    BIN_SPLIT_TAKE_GREATER,
    TBinarySplit,
)



# ================= lane/neural-pass116 (2026-10-02) =================
# THE FOLD-ORDER COMPRESSED INDEX. With more than one fold every level
# gathered `docs[i] = docIndices[subsets.Indices[i]]` and the histogram and
# split kernels read the column-major compressed index at those scattered
# document ids: four uncoalesced words per document per level over the ~2n
# concatenated documents (L40S, taxi 4.1M x 16, 20 Ordered trees:
# `pw.hist` 797 ms of 2,098). Here each permutation's index is gathered
# ONCE into the fold-concatenated order (column c, position p holds the
# word of document docIndices[p]), the calcer and the split table take the
# stride `doc_count`, and the levels read position `subsets.Indices[i]`
# directly: the same word for every (position, feature), in the same order,
# so the same histograms and splits. `MOJOLEARN_ORDERED_FOLD_INDEX=0`
# restores the per-level gather.
def ordered_fold_index() -> Bool:
    """OPT-IN (`MOJOLEARN_ORDERED_FOLD_INDEX=1`) since lane/neural-pass124:
    the histogram accumulators key their quantizer's dither on the row id
    they are handed (`hist2_dither(row)`), and on the fold-order index that
    id is the fold POSITION, not the document: where a stat is not exactly
    on the grid the dither moves the cell (peer's L40S, taxi 4.1M, 20
    Ordered trees: hash 72f4fbfc vs main's cc8d198d; the M4's 30K checks
    sat on the grid). Off, the levels gather document ids and the cells are
    main's. Keying the dither on docIndices[position] would make it a pure
    layout change; it measured 6% of pw.hist on the L40S."""
    return String(getenv("MOJOLEARN_ORDERED_FOLD_INDEX")) == "1"


def fold_cindex_gather_kernel(
    src: UnsafePointer[UInt32, MutAnyOrigin],
    ids: UnsafePointer[UInt32, MutAnyOrigin],
    dst: UnsafePointer[UInt32, MutAnyOrigin],
    n_rows: Int32,
    doc_count: Int32,
    n_cols: Int32,
):
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var dc = Int(doc_count)
    if t < Int(n_cols) * dc:
        var c = t // dc
        var p = t - c * dc
        dst[t] = src[c * Int(n_rows) + Int(ids[p])]


def fold_bins_from_table_kernel(
    starts: UnsafePointer[UInt32, MutAnyOrigin],
    bins: UnsafePointer[UInt32, MutAnyOrigin],
    n_parts: Int32,
    doc_count: Int32,
):
    """`ORD_ALL` fold bins in one launch (Apple FAST, wide data): `write_fold_based_initial_bins`'
    writes in one launch. `starts` holds the `n_parts` partition offsets
    of the fold layout in order (`FoldLayout.parts[p].offset`, learn then
    quality per fold, ascending) and `starts[n_parts] = doc_count`;
    position `p` gets the last partition `s` with `starts[s] <= p`, which
    is the one that contains it (an empty partition shares its start with
    the next one and is never the last such `s`). The same bins
    `enqueue_fill(view, UInt32(p))` wrote per partition."""
    var p = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if p < Int(doc_count):
        var lo = 0
        var hi = Int(n_parts) - 1
        while lo < hi:
            var mid = (lo + hi + 1) >> 1
            if Int(starts[mid]) <= p:
                lo = mid
            else:
                hi = mid - 1
        bins[p] = UInt32(lo)


def _cindex_columns(layout: CompressedIndexLayout) -> Int:
    var m = 0
    for f in range(len(layout.features)):
        m = max(m, Int(layout.features[f].offset) + 1)
    return m


struct PointwiseTreeWorkspace(Movable):
    """DEVIATION 143: the pointwise searcher's CROSS-TREE POOL.

    CatBoost never rebuilds this state per tree -- its `TCudaManager`
    memory pool hands `CreateSubsets` and the score helpers recycled
    device memory, so their per-tree cost is a handful of fills. This
    implementation has no manager, and constructing fresh was measured at 17-26
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

    THE FOLD ARM POOLS TOO (lane/ordered-speed, 2026-09-29). An Ordered
    fit's pool lives for that fit, whose fold layout is fixed, so a hit
    on (doc_count, fold_count, ...) is the same layout. The hit replays
    `create_fold_based_subsets`' state half in its order --
    `reset_subsets` at the fold counters (bins zeroed, sequences, one
    reduce), `write_fold_based_initial_bins`, the second reduce
    (DEVIATION 125) -- so the subsets hold what a fresh build holds.
    Fresh construction per tree plus the host `MakeDocIndices` over the
    ~2n concatenated documents were most of the Ordered structure
    search's host time at 4.1M rows. The fold doc ids are cached here
    too, per `permutation_id` (the caller's name for a permutation fixed
    for the fit); a pool rebuild drops them.

    The per-tree reset contract is CONSTRUCTOR POSTCONDITIONS
    (`reset_subsets` + `reset_for_tree`), held bit-exactly by
    `checks/pointwise_pool_check.mojo` and end-to-end by the
    pooled-vs-fresh identity of whole fits.
    """

    var subsets: TOptimizationSubsets
    var calcer: ScoresCalcerOnCompressedDataSet
    var doc_count_key: Int
    var n_rows_key: Int
    var max_depth_key: Int
    var n_features_key: Int
    var fold_count_key: Int
    #: the fold arm's `MakeDocIndices` per caller permutation id
    var doc_ids_keys: List[Int]
    var doc_ids: List[DeviceBuffer[DType.uint32]]
    # the compressed index in each permutation's fold order (one per
    # doc_ids entry; lane/neural-pass116, `ordered_fold_index`)
    var fold_cindex: List[DeviceBuffer[DType.uint32]]

    # ---- DEVIATION 207: the blind level loop's device state ----------
    # The winner fold slot (2 words + 2 floats), the per-level winner
    # records, the loop-carried scoreBeforeSplit, the packed split
    # descriptor, and the per-feature table the pack kernel reads --
    # everything the host loop used to touch between the read and the
    # split, now resident. `d_feat_table` is 4 words per feature
    # `(offset_elems, mask, shift, one_hot)`, the same values the host
    # lookup took from `layout.features[fid]` / `one_hot[fid]`, valid for
    # the pool's whole life because the pool is owned per fit and keyed on
    # (n_rows, n_features). The host pair rides the ONE drain per tree.
    var d_best_ids: DeviceBuffer[DType.uint32]
    var d_best_scores: DeviceBuffer[DType.float32]
    var d_winners_ids: DeviceBuffer[DType.uint32]
    var d_winners_scores: DeviceBuffer[DType.float32]
    var d_score_before: DeviceBuffer[DType.float32]
    var d_split_desc: DeviceBuffer[DType.uint32]
    var d_feat_table: DeviceBuffer[DType.uint32]
    var h_winners_ids: HostBuffer[DType.uint32]
    var h_winners_scores: HostBuffer[DType.float32]

    def __init__(
        out self,
        ctx: DeviceContext,
        var subsets: TOptimizationSubsets,
        var calcer: ScoresCalcerOnCompressedDataSet,
        layout: CompressedIndexLayout,
        one_hot: List[Bool],
        doc_count: Int,
        n_rows: Int,
        max_depth: Int,
        n_features: Int,
        fold_count: Int,
    ) raises:
        self.subsets = subsets^
        self.calcer = calcer^
        self.doc_count_key = doc_count
        self.n_rows_key = n_rows
        self.max_depth_key = max_depth
        self.n_features_key = n_features
        self.fold_count_key = fold_count
        self.doc_ids_keys = List[Int]()
        self.doc_ids = List[DeviceBuffer[DType.uint32]]()
        self.fold_cindex = List[DeviceBuffer[DType.uint32]]()

        self.d_best_ids = ctx.enqueue_create_buffer[DType.uint32](2)
        self.d_best_scores = ctx.enqueue_create_buffer[DType.float32](2)
        self.d_winners_ids = ctx.enqueue_create_buffer[DType.uint32](
            2 * max_depth
        )
        self.d_winners_scores = ctx.enqueue_create_buffer[DType.float32](
            2 * max_depth
        )
        self.d_score_before = ctx.enqueue_create_buffer[DType.float32](1)
        self.d_split_desc = ctx.enqueue_create_buffer[DType.uint32](5)
        self.d_feat_table = ctx.enqueue_create_buffer[DType.uint32](
            4 * n_features
        )
        self.h_winners_ids = ctx.enqueue_create_host_buffer[DType.uint32](
            2 * max_depth
        )
        self.h_winners_scores = ctx.enqueue_create_host_buffer[
            DType.float32
        ](2 * max_depth)

        # the same three values the host loop read from
        # `layout.features[fid]`, plus the same `one_hot[fid]` guard
        # (`len(one_hot) == len(layout.features)` or all-False)
        var table = List[UInt32]()
        var have_oh = len(one_hot) == n_features
        for f in range(n_features):
            table.append(UInt32(Int(layout.features[f].offset) * n_rows))
            table.append(UInt32(layout.features[f].mask))
            table.append(UInt32(layout.features[f].shift))
            table.append(
                UInt32(1) if (have_oh and one_hot[f]) else UInt32(0)
            )
        ctx.enqueue_copy(
            dst_buf=self.d_feat_table, src_ptr=table.unsafe_ptr()
        )
        ctx.synchronize()
        # keep the host list alive across the queue
        # ([[mojo-buffer-freed-at-last-use]])
        _ = table[0]


def _dd2(n: Int) -> String:
    """Two zero-padded digits, for depth components of trace tags."""
    if n < 10:
        return String("0") + String(n)
    return String(n)


def fit_oblivious_tree_structure_traced(
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
    mut trace: IdentityTrace,
    mut times: StageTimes,
    tree_tag: String,
    l2_leaf_reg: Float32 = Float32(3.0),
    score_std_dev: Float32 = Float32(0.0),
    seed: UInt64 = 0,
    one_hot: List[Bool] = List[Bool](),
    bootstrapped_observations: Bool = False,
    folds: List[TFold] = List[TFold](),
    permutation: List[UInt32] = List[UInt32](),
    permutation_id: Int = -1,
    fold_part_off: List[DeviceBuffer[DType.uint32]] = List[
        DeviceBuffer[DType.uint32]
    ](),
    obs_scratch: List[DeviceBuffer[DType.uint32]] = List[
        DeviceBuffer[DType.uint32]
    ](),
    ord_wide: Bool = False,
) raises -> List[TBinarySplit]:
    """`TDocParallelObliviousTreeSearcher::FitImpl` (`:12-160`), the
    structure half.

    `permutation_id` (fold arm only): a caller id for `permutation`, fixed
    for the span of `pool`; with it the fold doc ids are built once per id
    and kept in the pool. -1 builds them every call.

    `ord_wide`, `fold_part_off` and `obs_scratch` (the Apple FAST Ordered
    bundle, `ORD_ALL`; read only when it is compiled in and the caller
    passes `ord_wide = ord_all_on(n_features)`, every other build ignores
    them): the fold-order index as the default, one device table of the
    fold layout's partition starts plus `doc_count`
    (`fold_bins_from_table_kernel`), and one `doc_count`-long scratch for
    the per-level observation gather, both owned by the caller for the
    fit. `ord_wide = False` or empty lists take main's path.

    The weak target arrives as TWO buffers, which is `TL2Target` in the reference
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
        # ordered-boosting kernel in the reference at all. Refused HERE rather than
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

    var fold_order = False
    comptime if ORD_ALL:
        if ord_wide:
            # the fold-order index as the default (no env read a tree)
            fold_order = fold_count > 1
        else:
            fold_order = fold_count > 1 and ordered_fold_index()
    else:
        fold_order = fold_count > 1 and ordered_fold_index()
    var stride = doc_count if fold_order else n_rows
    var blocks = blocks_for(layout, stride)
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
        and pool[0].fold_count_key == fold_count
        and pool[0].doc_count_key == doc_count
        and pool[0].n_rows_key == stride
        and pool[0].max_depth_key == max_depth
        and pool[0].n_features_key == len(layout.features)
    )
    if pooled_hit:
        if fold_count == 1:
            reset_subsets(ctx, pool[0].subsets, target)
        else:
            # `create_fold_based_subsets`' state half, in its order
            reset_subsets(
                ctx, pool[0].subsets, target, fold_count,
                fold_layout.fold_bits,
            )
            comptime if ORD_ALL:
                if ord_wide and len(fold_part_off) > 0:
                    # every partition's bins in one launch from the
                    # caller's start table: no staging buffers, no drain
                    var starts = fold_part_off[0].copy()
                    ctx.enqueue_function[fold_bins_from_table_kernel](
                        starts.unsafe_ptr(),
                        pool[0].subsets.bins.unsafe_ptr(),
                        Int32(len(fold_layout.parts)),
                        Int32(doc_count),
                        grid_dim=max((doc_count + 255) // 256, 1),
                        block_dim=256,
                    )
                    _ = starts^
                else:
                    write_fold_based_initial_bins(
                        ctx, fold_layout, pool[0].subsets.bins
                    )
            else:
                write_fold_based_initial_bins(
                    ctx, fold_layout, pool[0].subsets.bins
                )
            update_subsets_stats(ctx, target, pool[0].subsets)
        pool[0].calcer.reset_for_tree(ctx)
    else:
        pool.clear()
        var subsets_new = create_subsets(ctx, max_depth, target) if (
            fold_count == 1
        ) else create_fold_based_subsets(
            ctx, max_depth, target, fold_layout, sm_count=sm_count
        )
        # DEVIATION 126, LIFTED 2026-09-03. This call omitted `fold_count`,
        # so the calcer's helpers were built at the default 1 while the
        # layout above was built at the fold count, and the consistency
        # check below raised on every fold arm. The constructor has always
        # accepted the argument and forwarded it to `PolicyScoreHelper`;
        # nothing was hard-coded and nothing needed rewriting. The subsets
        # branch immediately above already took the fold path.
        var calcer_new = ScoresCalcerOnCompressedDataSet(
            ctx, blocks, layout, stride, max_depth, global_ids, fold_count
        )
        pool.append(
            PointwiseTreeWorkspace(
                ctx, subsets_new^, calcer_new^, layout, one_hot, doc_count,
                stride, max_depth, len(layout.features), fold_count,
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
                " (`histograms_helper.h:361`) and this implementation hard-codes 1"
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
    var cached = -1
    if fold_count > 1 and permutation_id >= 0:
        for i in range(len(pool[0].doc_ids_keys)):
            if pool[0].doc_ids_keys[i] == permutation_id:
                cached = i
    var d_doc_ids: DeviceBuffer[DType.uint32]
    if cached >= 0:
        d_doc_ids = pool[0].doc_ids[cached].copy()
    else:
        var doc_ids_host = make_fold_doc_indices(folds, permutation) if (
            fold_count > 1
        ) else List[UInt32]()
        d_doc_ids = ctx.enqueue_create_buffer[DType.uint32](
            len(doc_ids_host) if fold_count > 1 else 1
        )
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
            # keep the host list alive across the queue: a raw pointer does
            # not ([[mojo-buffer-freed-at-last-use]])
            _ = doc_ids_host[0]
            if permutation_id >= 0:
                pool[0].doc_ids_keys.append(permutation_id)
                pool[0].doc_ids.append(d_doc_ids.copy())
    var d_fold_cindex: DeviceBuffer[DType.uint32]
    if fold_order and cached >= 0 and cached < len(pool[0].fold_cindex):
        d_fold_cindex = pool[0].fold_cindex[cached].copy()
    elif fold_order:
        var n_cols = _cindex_columns(layout)
        d_fold_cindex = ctx.enqueue_create_buffer[DType.uint32](
            max(n_cols * doc_count, 1)
        )
        ctx.enqueue_function[fold_cindex_gather_kernel](
            cindex.unsafe_ptr(),
            d_doc_ids.unsafe_ptr(),
            d_fold_cindex.unsafe_ptr(),
            Int32(n_rows),
            Int32(doc_count),
            Int32(n_cols),
            grid_dim=max((n_cols * doc_count + 255) // 256, 1),
            block_dim=256,
        )
        if permutation_id >= 0 and cached < 0:
            pool[0].fold_cindex.append(d_fold_cindex.copy())
    else:
        d_fold_cindex = ctx.enqueue_create_buffer[DType.uint32](1)
    var d_observations: DeviceBuffer[DType.uint32]
    comptime if ORD_ALL:
        # the caller's scratch when it hands one, one cell when the
        # fold-order index never gathers (both only when `ord_wide`)
        if ord_wide and len(obs_scratch) > 0 and not fold_order:
            d_observations = obs_scratch[0].copy()
        elif ord_wide and fold_order:
            d_observations = ctx.enqueue_create_buffer[DType.uint32](1)
        else:
            d_observations = ctx.enqueue_create_buffer[DType.uint32](
                doc_count
            )
    else:
        d_observations = ctx.enqueue_create_buffer[DType.uint32](doc_count)

    var structure = List[TBinarySplit]()

    # ================= DEVIATION 207 =================
    # ONE DRAIN PER TREE, NOT PER LEVEL -- the pointwise sibling of the
    # greedy family's DEVIATION 94, for the same price ledger: their
    # per-level `ReadOptimalSplit` is a ~5 us pinned read, this box's
    # drain is ~191 us plus a queue-empty bubble, and after DEVIATION 143
    # those `max_depth` waits were the arm's largest remaining host term
    # (PREP_BILL step 26: ~2-4 ms/tree). So the level loop below is
    # enqueued BLIND -- no host read anywhere in it. What made the
    # per-level read load-bearing, and what replaces each use:
    #
    # * the winner fold (`TakeBest` over blocks, then helpers) moves into
    #   `pw_fold_winner_kernel`, sequential, same nesting and tie rules;
    # * `score_before_split = best.score` becomes `d_score_before`,
    #   written by the pack kernel and loaded by the next level's score
    #   kernels (their host scalar, now a loop-carried device float);
    # * the `layout.features[fid]` / `one_hot[fid]` lookup becomes
    #   `d_feat_table`, and `split_subsets` consumes the packed
    #   descriptor (`split_subsets_from_desc`);
    # * the undefined-winner raise and the `HasSplit` stop move to the
    #   post-tree walk, which applies them IN LEVEL ORDER, so the first
    #   stop at level k discards levels k.. exactly as their loop would
    #   never have grown them. NOTHING AFTER THE LOOP READS `subsets`
    #   (the function returns the structure and the pool's next-tree
    #   reset rebuilds subset state from scratch), so unlike DEVIATION 94
    #   there is NO rollback: the blind extra levels' splits are simply
    #   discarded. A level that would have raised packs a well-formed
    #   (feature 0, bin 0) descriptor for the levels still in flight --
    #   see `pw_pack_winner_kernel` -- and the walk raises before reading
    #   anything a garbage level produced.
    # =================================================
    pool[0].d_score_before.enqueue_fill(Float32(0.0))

    # `auto& random = objective.GetRandom()` (`:15`), drawn from ONCE PER
    # LEVEL at the two `ComputeOptimalSplit` call sites (`:86`, `:104`).
    #
    # THIS WAS A REAL BUG AND IT WAS INVISIBLE. The level loop below used to
    # hand `seed` itself to every level, so every level of a tree drew the
    # SAME per-feature normal -- the noise would have been a fixed
    # per-feature offset for the whole tree instead of a fresh draw per
    # level. Nothing caught it because no caller ever passed a non-zero
    # `score_std_dev`, which is exactly CONTRIBUTING.md (Non-default paths): a branch nothing
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
        times.begin(ctx)
        if fold_count > 1 and not fold_order:
            launch_gather_with_mask_u32(
                ctx,
                d_observations,
                d_doc_ids,
                docs,
                doc_count,
                GATHER_NO_MASK,
            )
            docs = d_observations.copy()
        if fold_order:
            calcer.submit_compute(
                ctx, subsets, d_fold_cindex, docs, doc_count, sm_count,
                fixed_scale,
            )
        else:
            calcer.submit_compute(
                ctx, subsets, cindex, docs, doc_count, sm_count, fixed_scale
            )
        times.end(ctx, "pw.hist")

        # ---- identity checkpoint: this depth's REDUCED histograms ----
        # One record per policy present, over the LIVE VIEW only
        # (`histogram_view_size`, the parts-outer prefix the scorer
        # reads); the allocation's tail holds deeper levels' stale cells
        # (identity_trace rule 3). OUTSIDE the timed regions, and each
        # record drains -- a traced run is not a timing (rule 4).
        if trace.enabled:
            for hi in range(len(calcer.helpers)):
                if calcer.helpers[hi].feature_count == 0:
                    continue
                var view = calcer.helpers[hi].hist_helper.histogram_view_size(
                    depth, calcer.helpers[hi].bin_feature_count
                )
                trace.record_device(
                    ctx,
                    tree_tag + ".depth" + _dd2(depth) + ".hist."
                    + policy_name(calcer.helpers[hi].policy),
                    calcer.helpers[hi].d_hist,
                    count=view,
                )

        var pstats = subsets.partition_stats.copy()
        times.begin(ctx)
        calcer.compute_optimal_split_dev(
            ctx,
            pstats,
            1 << depth,
            pool[0].d_score_before,
            score_function,
            l2_leaf_reg,
            score_std_dev,
            level_rand.next_uniform_l(),
        )
        times.end(ctx, "pw.score")

        # their fold (`:113-120`) and the record's consumption, on the
        # device (DEVIATION 207); the raise and the `HasSplit` stop are in
        # the post-tree walk below
        # DEVIATION 3111: with the fused level the winner fold, the pack
        # and the bin update are ONE launch, enqueued below where the bin
        # update stood (`pw_resolve_pack_bins_kernel`).
        var live_helpers = 0
        for hi in range(len(calcer.helpers)):
            if calcer.helpers[hi].feature_count != 0:
                live_helpers += 1
        var fused_pw = PW_FUSED_SEARCH and live_helpers <= 3
        if not fused_pw:
            times.begin(ctx)
            calcer.resolve_optimal_split(
                ctx, pool[0].d_best_ids, pool[0].d_best_scores
            )
            launch_pw_pack_winner(
                ctx,
                pool[0].d_best_ids,
                pool[0].d_best_scores,
                depth,
                pool[0].d_winners_ids,
                pool[0].d_winners_scores,
                pool[0].d_score_before,
                pool[0].d_feat_table,
                len(layout.features),
                pool[0].d_split_desc,
            )
            times.end(ctx, "pw.winner")

        # their `Split(target, docBins, observationIndices, &subsets)`
        # (`oblivious_tree_structure_searcher.cpp:275-278`) -- the SAME
        # gathered array the histograms just read, because
        # `UpdateBinFromCompressedIndex` indexes the compressed index by
        # `docsForBins[i]` and not by `i`.
        var docs2 = d_observations.copy() if (
            fold_count > 1 and not fold_order
        ) else subsets.indices.copy()
        # `TCFeature::Offset` is an ELEMENT offset into the compressed
        # index and this tree's layout stores it as a COLUMN index strided
        # by `n_rows`; the conversion lives in `d_feat_table`'s build now
        # (`PointwiseTreeWorkspace.__init__`), where its history -- the raw
        # column reading column 0's bits and stopping every tree at depth
        # 1 -- is the reason the table stores `offset * n_rows`.
        times.begin(ctx)
        if fused_pw:
            comptime if PW_FUSED_SEARCH:
                var bin_depth = subsets.current_depth + subsets.fold_bits
                if Int(bin_depth) >= 32:
                    raise Error(
                        String("Split at depth ") + String(bin_depth)
                        + " would write bit " + String(bin_depth)
                        + " of a ui32 bin; CatBoost's ReorderBins asserts"
                        " (offset + bits) <= 32 (cuda_util/sort.cpp:557)"
                    )
                # the live helpers in calcer order, the fold order of
                # `resolve_optimal_split`; an absent slot folds 0 records
                var r_ids = List[MutPointer[UInt32, MutAnyOrigin]]()
                var r_scores = List[MutPointer[Float32, MutAnyOrigin]]()
                var r_n = List[Int]()
                for hi in range(len(calcer.helpers)):
                    if calcer.helpers[hi].feature_count == 0:
                        continue
                    r_ids.append(
                        rebind[MutPointer[UInt32, MutAnyOrigin]](
                            calcer.helpers[hi].d_result_ids.unsafe_ptr()
                        )
                    )
                    r_scores.append(
                        rebind[MutPointer[Float32, MutAnyOrigin]](
                            calcer.helpers[hi].d_result_scores.unsafe_ptr()
                        )
                    )
                    r_n.append(calcer.helpers[hi].result_blocks)
                var pad_ids = rebind[MutPointer[UInt32, MutAnyOrigin]](
                    pool[0].d_best_ids.unsafe_ptr()
                )
                var pad_scores = rebind[MutPointer[Float32, MutAnyOrigin]](
                    pool[0].d_best_scores.unsafe_ptr()
                )
                while len(r_n) < 3:
                    r_ids.append(pad_ids)
                    r_scores.append(pad_scores)
                    r_n.append(0)
                var split_ci = rebind[MutPointer[UInt32, MutAnyOrigin]](
                    cindex.unsafe_ptr()
                )
                if fold_order:
                    split_ci = rebind[MutPointer[UInt32, MutAnyOrigin]](
                        d_fold_cindex.unsafe_ptr()
                    )
                var num_blocks = (
                    subsets.doc_count + PW_SPLIT_BLOCK_SIZE - 1
                ) // PW_SPLIT_BLOCK_SIZE
                if num_blocks > PW_SPLIT_MAX_BLOCKS:
                    num_blocks = PW_SPLIT_MAX_BLOCKS
                if num_blocks < 1:
                    # the pack still runs on an empty doc list
                    num_blocks = 1
                ctx.enqueue_function[pw_resolve_pack_bins_kernel](
                    r_ids[0], r_scores[0], Int32(r_n[0]),
                    r_ids[1], r_scores[1], Int32(r_n[1]),
                    r_ids[2], r_scores[2], Int32(r_n[2]),
                    pad_ids, pad_scores,
                    Int32(depth),
                    pool[0].d_winners_ids.unsafe_ptr(),
                    pool[0].d_winners_scores.unsafe_ptr(),
                    pool[0].d_score_before.unsafe_ptr(),
                    pool[0].d_feat_table.unsafe_ptr(),
                    Int32(len(layout.features)),
                    pool[0].d_split_desc.unsafe_ptr(),
                    split_ci,
                    docs2.unsafe_ptr(),
                    Int32(subsets.doc_count),
                    UInt32(bin_depth),
                    subsets.bins.unsafe_ptr(),
                    grid_dim=(num_blocks, 1, 1),
                    block_dim=(PW_SPLIT_BLOCK_SIZE, 1, 1),
                )
        if fold_order:
            split_subsets_from_desc(
                ctx,
                target,
                d_fold_cindex,
                docs2,
                pool[0].d_split_desc,
                subsets,
                bins_done=fused_pw,
            )
        else:
            split_subsets_from_desc(
                ctx,
                target,
                cindex,
                docs2,
                pool[0].d_split_desc,
                subsets,
                bins_done=fused_pw,
            )
        times.end(ctx, "pw.split")

    # ---- THE ONE DRAIN OF THE TREE (DEVIATION 207) -------------------
    # Their per-level `ReadOptimalSplit`, folded into one: the winner
    # records hold every level and ride home behind everything the loop
    # enqueued.
    times.begin(ctx)
    ctx.enqueue_copy(
        dst_buf=pool[0].h_winners_ids, src_buf=pool[0].d_winners_ids
    )
    ctx.enqueue_copy(
        dst_buf=pool[0].h_winners_scores, src_buf=pool[0].d_winners_scores
    )
    ctx.synchronize()
    times.end(ctx, "pw.drain")

    # ---- identity checkpoint: the tree's winner records --------------
    # Every level's (feature, bin) and score pair, exactly as the one
    # drain brought them home; hashed from the HOST copies, so this adds
    # no device traffic of its own.
    trace.record_host(
        tree_tag + ".winners.ids",
        pool[0].h_winners_ids.unsafe_ptr(),
        2 * max_depth,
    )
    trace.record_host(
        tree_tag + ".winners.scores",
        pool[0].h_winners_scores.unsafe_ptr(),
        2 * max_depth,
    )

    # ============ THE GATES, POST-TREE ================================
    # The host loop applied these BEFORE each split; the walk applies them
    # in LEVEL ORDER, so the first stop at level k discards levels k..
    # exactly as the loop would never have grown them, and the returned
    # structure is unchanged record for record.
    for depth2 in range(max_depth):
        var fid_u = pool[0].h_winners_ids[2 * depth2]
        var bin_u = pool[0].h_winners_ids[2 * depth2 + 1]

        if fid_u == PW_SENTINEL_ID:
            raise Error(
                "best split is undefined at depth "
                + String(depth2)
                + ": every candidate scored non-finite. Theirs raises the"
                " same way (`:122`)."
            )

        var fid = Int(fid_u)
        var is_one_hot = False
        if len(one_hot) == len(layout.features):
            is_one_hot = one_hot[fid]

        # `structure.HasSplit(bestSplit)` (`:134`), BEFORE applying it
        var seen = False
        for i in range(len(structure)):
            if (
                structure[i].feature_id == Int32(fid)
                and structure[i].bin_idx == Int32(bin_u)
            ):
                seen = True
        if seen:
            break

        # `split_type`, and the constants are NOT in the order the names
        # suggest: `BIN_SPLIT_TAKE_BIN` is 0 and `BIN_SPLIT_TAKE_GREATER`
        # is 1 (`oblivious_model.mojo:36-37`). Writing `1 if one_hot else
        # 0` makes every ORDINARY feature an equality test, which still
        # grows a well-formed tree of the right depth with the right
        # splits and partitions the rows completely differently -- 8
        # non-empty leaves instead of 12. That is how this was found:
        # identical structure, different leaf values.
        structure.append(
            TBinarySplit(
                Int32(fid),
                Int32(bin_u),
                Int32(BIN_SPLIT_TAKE_BIN) if is_one_hot else Int32(
                    BIN_SPLIT_TAKE_GREATER
                ),
            )
        )

    return structure^


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
    """The un-instrumented entry: the exact pre-instrumentation signature,
    forwarding to `fit_oblivious_tree_structure_traced` with BOTH
    instruments off.

    The `StageTimes` is FORCE-DISABLED rather than env-constructed on
    purpose: an env-enabled timer here would pay a drain per stage per
    level for a table nobody reports (the fit-level table lives with the
    boosting loop, which calls the traced entry directly). Same for the
    trace: a per-tree `IdentityTrace()` would restart `seq` at 0 in a
    shared trace file and break the format's monotonic-seq contract.
    """
    var no_trace = IdentityTrace.disabled()
    var no_times = StageTimes()
    no_times.enabled = False
    return fit_oblivious_tree_structure_traced(
        ctx, layout, n_rows, max_depth, cindex,
        weights^, weighted_target^,
        sm_count, fixed_scale, score_function, pool,
        no_trace, no_times, String("tree"),
        l2_leaf_reg,
        score_std_dev=score_std_dev,
        seed=seed,
        one_hot=one_hot,
        bootstrapped_observations=bootstrapped_observations,
        folds=folds,
        permutation=permutation,
    )


def split_stat_planes(
    ctx: DeviceContext,
    mut stats: DeviceBuffer[DType.float32],
    n_rows: Int,
) raises -> Tuple[
    DeviceBuffer[DType.float32], DeviceBuffer[DType.float32]
]:
    """Two columns of one buffer into two buffers, because they have to be.

    NO CATBOOST COUNTERPART -- the reference's `TL2Target` is already two separate
    `TCudaBuffer<float>` and no split is needed. It exists because THIS tree
    carries the weak target as one two-plane buffer everywhere else
    (`greedy_search_helper`'s `stats`, plane 0 the weight and plane 1 the
    gradient), and the pointwise kernels cannot take two views of one
    buffer: they declare `target` and `weight` on independent origins and
    Mojo refuses the aliasing at `enqueue_function` itself (DEVIATION 97.2,
    `CONTRIBUTING.md`).

    So this is a BRIDGE between two internal conventions, not an implementation of
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
