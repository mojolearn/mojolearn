"""Grow one DEPTHWISE tree: every live leaf splits, each on its own feature.

PORT OF the `EGrowPolicy::Depthwise` arms of
`catboost/cuda/methods/greedy_subsets_searcher/greedy_search_helper.cpp`
(`:359`, `:428`, `:470`, `:546-553`, `:606`, `:685`), of
`TSplitPropertiesHelper::MakeSplit`'s MULTI-leaf arm
(`split_properties_helper.cpp:833-950`), and of
`TGreedyTreeLikeStructureSearcher<TNonSymmetricTree>::FitImpl`
(`structure_searcher_template.h:41-67`), at CatBoost `54a8143a`.
Transliterated. Do not improve.

WHY THIS IS A SEPARATE FILE FROM `greedy_search_helper.mojo`
-----------------------------------------------------------
CatBoost has ONE file for all four grow policies, because `TGreedySearchHelper`
is one class with `switch (Options.Policy)` inside three of its methods. This
port does not, and the reason is recorded rather than assumed:

* the SymmetricTree arm here is not a transliteration of that class at all --
  it is `run_tree_layout`, a specialization that folds the level's single
  winner onto the device (DEVIATION 94/95/207), enqueues every level blind
  and drains once per TREE. Nothing in that design survives the move to a
  per-leaf winner, so a `switch` would have been two programs in one function.
* three lanes are live in this checkout at once. One file with four policy
  arms is one file three sessions edit.

The FILE SPLIT IS THE ONLY DEVIATION IN THE STRUCTURE. Every branch below
cites the line of theirs it came from, and the Lossguide arms that belong to
the other lane are absent rather than stubbed.

THE FIVE THINGS DEPTHWISE DOES DIFFERENTLY, and they are all there is
--------------------------------------------------------------------
1. `numScoreBlocks = leavesToVisit.size()` instead of 1 (`:428-432`), so the
   score kernel is `ComputeOptimalSplitsRegion` with the leaf on grid y
   (`:470-488`) and the host reduces ONE WINNER PER LEAF (`:546-553`).
2. `MinLeafSize` goes LIVE: `IsTerminalLeaf`'s size test is guarded by
   `Options.Policy != EGrowPolicy::SymmetricTree` (`:685`), so this is the
   first lane in this port where `min_data_in_leaf` decides anything.
3. `UpdateFeatureWeightsForBestSplits` is NOT called -- it sits inside the
   SymmetricTree branch only (`:466`), so `model_size_reg` does nothing here.
4. The leaves are a LIST that grows unevenly. A level splits only the leaves
   that are non-terminal AND improving, so `leavesToSplit` is a subset and
   the new right children take ids `leavesCount + i` (`:861`).
5. The model is `TNonSymmetricTree`, rebuilt from the leaf PATHS by
   `model_builder.build_non_symmetric_tree`, not a split list.

Everything else -- the histogram build, the sibling subtraction, the scan,
the split-points chain, the stable partition, the reorder, the partition
stats -- is the SAME CODE the symmetric lane runs, called from here. That is
not an economy, it is the port's central claim about their design: the level
machinery is policy-blind, and `split_and_make_sequence_kernel` already takes
a per-leaf-slot `TCFeature` and bin because THEIR kernel does
(`split_properties_helper.cpp:856-880`), even though the symmetric caller
fills every slot with the same value.

THE CONTROL PLANE IS THEIRS, DELIBERATELY
-----------------------------------------
HOST_AND_DEVICE.md's rule two: cut our host waits down to THEIR count, not
below it, not yet. Theirs blocks the host TWICE per level -- `bestProps.Read`
(`greedy_search_helper.cpp:517`) and `RebuildLeavesSizes`
(`split_properties_helper.cpp:802`) -- and so does this loop, at exactly
those two points. The symmetric lane's blind-enqueue deviations bought ~13 ms
of a 50 ms tree AFTER the port had produced a number; this lane has no number
yet, and a control plane better than theirs cannot answer whether THEIR
design is fast on Metal.

NUMERIC IDENTITY
----------------
This lane adds NO new row to `IDENTITY_PATHS.md`. Every float that decides
anything here comes from machinery already enumerated there: the histogram
flush (rows 1-6), the partition-stats chunk count (row 7), the fixed-point
scale (row 8, open on both lanes). The two host reductions this file adds are
order-independent by construction rather than by pinning:

* the per-leaf cross-block argmin is a SEQUENTIAL fold in block order with a
  total order on `(gain, feature_id, bin_id)` (`best_split_properties_less`),
  so it cannot depend on how many blocks the machine chose to run;
* `build_necessary_histograms` groups siblings in LEAF-ID order where theirs
  walks a `THashMap` (`split_properties_helper.cpp:1306`). Hash order does
  not move bits -- the pairs are independent -- but id order is reproducible
  across builds and theirs is not, so this is a deviation that can only
  narrow the identity gap. Already recorded by the symmetric lane.
"""

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from max.gpu.host.device_attribute import DeviceAttribute

from mojo_only.fixed_point import choose_scale
from gbdt.gpu_lib.gpu_manager import TCudaManager
from gbdt.data.leaf_path import TLeafPath, split_leaf_path
from gbdt.data.permutation import TRandom
from gbdt.gpu_data.compressed_index_builder import (
    CompressedIndexLayout,
    build_layout,
)
from gbdt.gpu_data.feature_blocks import blocks_for
from gbdt.gpu_data.gpu_structures import CFeature
from gbdt.gpu_util.partitions_reduce import compute_partition_stats
from gbdt.methods.greedy_subsets_searcher.greedy_search_helper import (
    CFEATURE_BYTES,
    TTreeWorkspace,
    launch_histograms_for_blocks,
    resolve_split,
)
from gbdt.methods.greedy_subsets_searcher.kernel.compute_scores import (
    LEAFWISE_SCORE_BLOCK_SIZE,
    compute_optimal_splits_region_kernel,
)
from gbdt.methods.greedy_subsets_searcher.kernel.histogram_utils import (
    copy_histograms_kernel,
    scan_histograms_kernel,
    substract_histograms_kernel,
    zero_histograms_kernel,
)
from gbdt.methods.greedy_subsets_searcher.kernel.split_points import (
    SPLIT_BLOCK_SIZE,
    launch_reorder_in_leaves,
    launch_stable_partition,
    split_and_make_sequence_kernel,
    split_points_grid_x,
    update_partitions_after_split_kernel,
)
from gbdt.methods.greedy_subsets_searcher.model_builder import (
    build_non_symmetric_tree,
)
from gbdt.methods.greedy_subsets_searcher.points_subsets import (
    EHistogramsType,
    TBestSplitProperties,
    TLeaf,
)
from gbdt.methods.greedy_subsets_searcher.split_properties_helper import (
    HISTOGRAMS_CURRENT_PATH,
    HISTOGRAMS_PREVIOUS_PATH,
    HISTOGRAMS_ZEROES,
    LeafRecord,
    build_necessary_histograms,
    non_zero_leaves,
)
from gbdt.methods.greedy_subsets_searcher.structure_searcher_options import (
    TTreeStructureSearcherOptions,
)
from gbdt.methods.helpers import (
    SPLIT_VALUE_ONE,
    SPLIT_VALUE_ZERO,
    best_split_properties_less,
)
from gbdt.models.non_symmetric_tree import TNonSymmetricTree
from gbdt.models.oblivious_model import (
    BIN_SPLIT_TAKE_BIN,
    BIN_SPLIT_TAKE_GREATER,
    TBinarySplit,
)
from gbdt.options.catboost_options import (
    GROW_DEPTHWISE,
    SCORE_FUNCTION_COSINE,
    SCORE_FUNCTION_L2,
    SCORE_FUNCTION_NEWTON_L2,
)


struct TBinFeatureTable(Copyable, Movable):
    """Every bin-feature resolved to `(feature, bin)`, once per tree.

    Their `ToSplit(FeaturesManager, props)` (`methods/helpers.cpp:164-170`)
    does this resolution on the host, one candidate at a time, out of the
    features manager. `resolve_split` already does it here -- but it is an
    O(features) WALK per call, and the depthwise host reduce resolves
    `argmax_blocks * leavesToVisit.size()` records per level where the
    symmetric one resolved ONE per level.

    So the walk is done once for every bin-feature at the top of the fit and
    the level loop indexes it. That is a change of ALGORITHM on the host and
    therefore a deviation (PORTING.md 351), and it is bit-inert by
    construction: `resolve_split` is the function that fills the table, so
    the table cannot disagree with it. `mojo_only/depthwise_check.mojo`
    claim 1 asserts the two agree cell for cell anyway, because "cannot
    disagree by construction" is exactly the sentence this repository has
    been wrong about before.

    HOST_AND_DEVICE.md rule one holds: `len(feature)` is the total bin
    count, which scales with FEATURES times BORDERS and never with rows.
    """

    var feature: List[Int32]
    var bin: List[Int32]
    var one_hot: List[Bool]

    def __init__(out self, layout: CompressedIndexLayout) raises:
        self.feature = List[Int32]()
        self.bin = List[Int32]()
        self.one_hot = List[Bool]()
        for bf in range(layout.hist_cells):
            var choice = resolve_split(layout, bf)
            self.feature.append(Int32(choice.feature))
            self.bin.append(Int32(choice.bin))
            self.one_hot.append(
                layout.features[choice.feature].one_hot_feature
            )

    def to_split(self, bin_feature: Int) raises -> TBinarySplit:
        """Their `ToSplit` (`methods/helpers.cpp:164-170`).

        `SplitType` is `TakeBin` for a one-hot feature and `TakeGreater`
        otherwise, which is their `manager.IsCat(props.FeatureId)` test read
        off the layout instead of off the features manager -- the same
        substitution `run_tree_layout` makes at its own `ToSplit` site.
        """
        return TBinarySplit(
            self.feature[bin_feature],
            self.bin[bin_feature],
            Int32(
                BIN_SPLIT_TAKE_BIN
            ) if self.one_hot[bin_feature] else Int32(BIN_SPLIT_TAKE_GREATER),
        )


struct TDepthwiseWorkspace(Movable):
    """The buffers `TTreeWorkspace` does not have, and only those.

    `TTreeWorkspace` (the symmetric lane's pool, `greedy_search_helper.mojo`)
    already owns every large plane: histograms, the fixed-point accumulator,
    the partitions, the stat partials, the row flags and the reorder scratch,
    the compressed-index tables, the bin-feature planes and the scale. All of
    it is shape-keyed and reused verbatim here.

    Four things it cannot serve, all of them because the SYMMETRIC level has
    one winner and one dense id list where a depthwise level has neither:

    * `region_score` / `region_bin` -- their `bestProps`, sized
      `argmaxBlockCount * numScoreBlocks` (`greedy_search_helper.cpp:441`).
      The pool's `out_score` / `out_bin` are sized `argmaxBlockCount`,
      because `numScoreBlocks` is 1 for SymmetricTree.
    * `d_visit` -- their `leafIds` (`:472`), the leaves being scored.
    * `d_left` / `d_right` -- their `leftIdsGpu` / `rightIdsGpu` (`:895-896`),
      which for a symmetric level are the dense prefix and the dense prefix
      plus half, and so needed no buffer at all.
    * `h_part_stats` -- their `ReadReduce(currentPartStats)` (`:632`), the
      once-per-tree drain that becomes the leaf values.
    """

    var max_leaves_key: Int
    var stat_count_key: Int
    var argmax_blocks_key: Int

    var region_score: DeviceBuffer[DType.float32]
    var region_bin: DeviceBuffer[DType.uint32]
    var h_region_score: HostBuffer[DType.float32]
    var h_region_bin: HostBuffer[DType.uint32]
    var d_visit: DeviceBuffer[DType.uint32]
    var h_visit: HostBuffer[DType.uint32]
    var d_left: DeviceBuffer[DType.uint32]
    var h_left: HostBuffer[DType.uint32]
    var d_right: DeviceBuffer[DType.uint32]
    var h_right: HostBuffer[DType.uint32]
    var d_ids: DeviceBuffer[DType.uint32]
    var h_ids: HostBuffer[DType.uint32]
    var h_part_stats: HostBuffer[DType.float32]

    def __init__(
        out self,
        ctx: DeviceContext,
        max_leaves: Int,
        stat_count: Int,
        argmax_blocks: Int,
    ) raises:
        self.max_leaves_key = max_leaves
        self.stat_count_key = stat_count
        self.argmax_blocks_key = argmax_blocks
        var records = argmax_blocks * max_leaves
        self.region_score = ctx.enqueue_create_buffer[DType.float32](records)
        self.region_bin = ctx.enqueue_create_buffer[DType.uint32](records)
        self.h_region_score = ctx.enqueue_create_host_buffer[DType.float32](
            records
        )
        self.h_region_bin = ctx.enqueue_create_host_buffer[DType.uint32](
            records
        )
        self.d_visit = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
        self.h_visit = ctx.enqueue_create_host_buffer[DType.uint32](
            max_leaves
        )
        self.d_left = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
        self.h_left = ctx.enqueue_create_host_buffer[DType.uint32](max_leaves)
        self.d_right = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
        self.h_right = ctx.enqueue_create_host_buffer[DType.uint32](
            max_leaves
        )
        self.d_ids = ctx.enqueue_create_buffer[DType.uint32](max_leaves)
        self.h_ids = ctx.enqueue_create_host_buffer[DType.uint32](max_leaves)
        self.h_part_stats = ctx.enqueue_create_host_buffer[DType.float32](
            max_leaves * stat_count
        )


def is_terminal_leaf(
    leaf: TLeaf, options: TTreeStructureSearcherOptions
) raises -> Bool:
    """`TGreedySearchHelper::IsTerminalLeaf` (`greedy_search_helper.cpp:683`).

        const bool checkLeafSize = Options.Policy != EGrowPolicy::SymmetricTree;
        const bool flag = (checkLeafSize && leaf.Size <= Options.MinLeafSize)
                          || leaf.Path.GetDepth() >= Options.MaxDepth;

    **The size test is `<=`, not `<`.** `min_data_in_leaf = 1` therefore
    means a one-row leaf is TERMINAL, not that a one-row leaf is permitted.
    Getting that boundary backwards grows a tree one level deeper than
    CatBoost's on every branch that reaches a single row, and no leaf count
    or row conservation can see it.

    `checkLeafSize` is written out rather than folded away even though this
    file only ever runs with Depthwise, because the constant it folds to is
    a POLICY fact and the next reader should not have to know the policy to
    read the line.
    """
    var check_leaf_size = options.policy != 0  # EGrowPolicy::SymmetricTree
    if check_leaf_size and Float64(leaf.size) <= options.min_leaf_size:
        return True
    return leaf.get_depth() >= options.max_depth


def should_terminate(
    leaves: List[TLeaf], options: TTreeStructureSearcherOptions
) raises -> Bool:
    """`ShouldTerminate` (`greedy_search_helper.cpp:672-682`).

        if (leafCount >= Options.MaxLeaves) return true;
        ... return AreAllTerminal(subsets, allLeaves);

    `AreAllTerminal` is folded into the loop below; theirs builds an
    `Iota` vector and calls the helper, which is the same test.
    """
    if len(leaves) >= options.max_leaves:
        return True
    for i in range(len(leaves)):
        if not leaves[i].is_terminal:
            return False
    return True


def select_leaves_to_visit(leaves: List[TLeaf]) raises -> List[Int]:
    """`SelectLeavesToVisit` (`greedy_search_helper.cpp:697-710`).

        if (!leaf.IsTerminal) {
            if (leaf.BestSplit.Defined()) continue;
            leavesToVisit->push_back(leaf);
        }

    Note what the `Defined()` skip buys: a leaf whose histogram was NOT
    rebuilt this level still holds last level's best split, and is not
    re-scored. `BuildNecessaryHistograms` is what clears it, for exactly the
    leaves whose histograms it updated (`split_properties_helper.cpp:1367`).
    That coupling is the whole incremental mechanism and it is why the reset
    below lives in this file's histogram step and not next to the scoring.
    """
    var out = List[Int]()
    for leaf in range(len(leaves)):
        if not leaves[leaf].is_terminal:
            if leaves[leaf].best_split.defined:
                continue
            out.append(leaf)
    return out^


def select_leaves_to_split(leaves: List[TLeaf]) raises -> List[Int]:
    """`SelectLeavesToSplit`'s Depthwise arm (`greedy_search_helper.cpp:359`).

        CB_ENSURE(Options.Policy == SymmetricTree || Options.Policy == Depthwise);
        for (leaf ...) if (Leaves[leaf].BestSplit.Defined()
                           && Leaves[leaf].BestSplit.Score < 0)
            leavesToSplit->push_back(leaf);

    Depthwise and SymmetricTree share this branch exactly; Lossguide takes
    the single best leaf and Region takes the shallower of the last pair,
    and both of those belong to other lanes.

    **THE TEST IS ON `Score`, NOT ON `Gain`, and for this policy they are
    THE SAME NUMBER.** `ComputeOptimalSplitsRegion` writes the gain into
    both fields -- `if (gain < bestScore) { bestScore = gain; bestIndex =
    binFeatureId; bestGain = gain; }` (`compute_scores.cu:380-384`) --
    where the oblivious kernel writes the raw score into `Score` and the
    gain into `Gain` (`:132-140`). So porting the test as written is right
    here and would be WRONG if this branch were ever fed the oblivious
    kernel's records.

    ============ WHICH SIGN THIS FIELD IS IN, and it is THEIRS ============
    The KERNEL is sign-flipped (larger gain is better; see
    `kernel/compute_scores.mojo`). `TBestSplitProperties` IS NOT. It is a
    transliteration of their struct, its defaults are their defaults
    (`Gain = FLT_MAX` losing every comparison), and the only comparator over
    it -- `best_split_properties_less`, their `operator<` -- is keyed on
    their orientation. So `compute_optimal_splits` NEGATES the kernel's gain
    once, at the point where it builds the record, and everything downstream
    of that record is their code unaltered, this test included.

    Writing the kernel's sign into the struct and flipping the test instead
    LOOKS equivalent and is not: `best_split_properties_less` would then be
    reading a field in the opposite orientation from the one it was
    transcribed against, and would silently select the WORST candidate on
    every cross-block reduce. This function had `gain > 0` on its first
    run and grew a one-leaf tree, which is what that mistake looks like
    from the outside: no candidate ever passes, every leaf is marked
    terminal, and the fit returns a constant.
    ======================================================================
    """
    var out = List[Int]()
    for leaf in range(len(leaves)):
        if leaves[leaf].best_split.defined:
            # `BestSplit.Score < 0`, verbatim -- and on this kernel `Score`
            # and `Gain` are the same number (see above).
            if leaves[leaf].best_split.gain < Float32(0.0):
                out.append(leaf)
    return out^


def split_leaf(
    leaf: TLeaf, split: TBinarySplit, direction: Int
) raises -> TLeaf:
    """`SplitLeaf` (`split_properties_helper.cpp:786-798`), all five fields.

        newLeaf.Size = 0;
        newLeaf.Path = leaf.Path; newLeaf.Path.AddSplit(split, direction);
        if (leaf.HistogramsType == CurrentPath)
            newLeaf.HistogramsType = PreviousPath;
        newLeaf.BestSplit.Reset();

    **The `HistogramsType` transition is conditional and the default is
    `Zeroes`.** A child of a leaf whose histogram was current inherits
    `PreviousPath`, which is what makes it eligible for sibling subtraction
    next level -- its slot holds the PARENT's histogram, put there by the
    left child keeping the parent's slot and by `copy_histograms_kernel`
    filling the right child's. A child of a leaf whose histogram was NOT
    current falls through to `Zeroes` and is rebuilt. Collapsing the
    condition to an unconditional `PreviousPath` would subtract a histogram
    that was never written.

    `Size = 0` is theirs and is a placeholder: `RebuildLeavesSizes` fills it
    from the device right after the split kernel.
    """
    var child = TLeaf()
    child.size = 0
    child.path = split_leaf_path(leaf.path, split, direction)
    if leaf.histograms_type == EHistogramsType.CurrentPath:
        child.histograms_type = EHistogramsType.PreviousPath
    # else: stays at the constructor's Zeroes, theirs by omission
    return child^


def _leaf_records(
    leaves: List[TLeaf], parent_of: List[Int]
) raises -> List[LeafRecord]:
    """`BuildNecessaryHistograms`' view of the leaves.

    `path_id` is their `PreviousSplit(leaf.Path)` hash-map key
    (`split_properties_helper.cpp:1300`), replaced by the PARENT LEAF ID --
    the substitution `split_properties_helper.mojo` has carried since the
    symmetric lane wrote it. Two siblings share a parent id exactly when
    they share a parent path, so the equivalence classes are identical, and
    an integer key is reproducible where a hash-map walk is not.

    THE PARENT ID IS RECOVERED FROM THE PATH, not carried as a field. A
    leaf's parent is the leaf it was split from, which is the LEFT child's
    own id, and the left child keeps the parent's id (`MakeSplit`,
    `:861-862`). So the caller passes a `parent_of` list built at split
    time; this function does not guess it.
    """
    var out = List[LeafRecord]()
    for i in range(len(leaves)):
        var t = HISTOGRAMS_ZEROES
        if leaves[i].histograms_type == EHistogramsType.PreviousPath:
            t = HISTOGRAMS_PREVIOUS_PATH
        elif leaves[i].histograms_type == EHistogramsType.CurrentPath:
            t = HISTOGRAMS_CURRENT_PATH
        out.append(
            LeafRecord(
                UInt32(leaves[i].size),
                t,
                parent_of[i],
                leaves[i].is_terminal,
            )
        )
    return out^


def fit_depthwise_tree[
    hist2_smem_mode: Int = 0
](
    ctx: DeviceContext,
    n_rows: Int,
    fold_counts: List[Int],
    options: TTreeStructureSearcherOptions,
    mut cindex: DeviceBuffer[DType.uint32],
    mut stats: DeviceBuffer[DType.float32],
    mut row_index: DeviceBuffer[DType.uint32],
    weight_magnitude: Float32,
    gradient_magnitude: Float32,
    mut ws: List[TTreeWorkspace],
    mut dws: List[TDepthwiseWorkspace],
    one_hot: List[Bool] = List[Bool](),
    approx_dim: Int = 1,
    multiclass_optimization: Bool = False,
    random_seed: UInt64 = UInt64(0),
    # ---- A TEST KNOB, and the only reason it is a parameter ----
    # "A configuration that cannot be varied inside one process cannot be
    # measured here, so make it a parameter" -- RESUME.md, earned on this
    # box. THE CORE COUNT IS THE ONLY MACHINE-DEPENDENT INPUT this
    # algorithm has: it sizes every strided grid and it is what
    # `IDENTITY_PATHS.md` row 7 had to PIN when it turned out to be
    # feeding a float sum. Overriding it here lets one process ask the
    # cross-GPU question directly -- grow the same tree as a 10-core M4
    # and as a 108-SM A100 and compare bits -- instead of waiting for the
    # other machine. -1 means "read the device", which is every
    # non-test caller.
    sm_count_override: Int = -1,
) raises -> TNonSymmetricTree:
    """`TGreedyTreeLikeStructureSearcher<TNonSymmetricTree>::FitImpl`.

    Their whole tree, `structure_searcher_template.h:41-67`:

        TPointsSubsets subsets = searchHelper.CreateInitialSubsets(objective);
        while (true) {
            searchHelper.ComputeOptimalSplits(&subsets);
            if (!searchHelper.SplitLeaves(&subsets, &leaves, &weights, &values))
                break;
        }
        return BuildTreeLikeModel<TModel>(leaves, weights, values);

    ONE ITERATION IS ONE LEVEL under Depthwise, exactly as under
    SymmetricTree -- every non-terminal improving leaf splits at once. What
    differs is that "every leaf" is a SUBSET and each member takes its own
    split, so the iteration count is still bounded by `max_depth` but the
    leaf count is not `1 << depth`.

    `weight_magnitude` / `gradient_magnitude` are the sums of ABSOLUTE
    values, one per stat plane, and they are the safety argument for the
    fixed-point flush the `IDENTICAL` numeric mode puts in place of
    CatBoost's float atomic. Same contract, same words, as
    `run_tree_layout`: every partial sum the device forms is over a SUBSET
    of the rows, so bounding the full-dataset sum of magnitudes bounds every
    Int32 slot at every depth. See `mojo_only/fixed_point.mojo`.

    ================= DEVIATION 352 =================
    THE PARTITION STATS ARE RECOMPUTED, NOT UPDATED IN THE SPLIT.
    Their `TSplitPointsKernel` updates `subsets->PartitionStats` inside the
    split ("Update part stats", `split_properties_helper.cpp:918`), so their
    `ComputeOptimalSplits` finds them already correct. This port has never
    ported that half -- `run_tree_layout` calls `compute_partition_stats`
    at the top of every level instead -- and this lane does the same rather
    than growing a second mechanism. The cost is one extra reduction per
    level; the numbers are identical because it is the same reduction over
    the same rows with the same pinned chunk count (`IDENTITY_PATHS.md`
    row 7).
    ===============================================
    """
    if options.policy != GROW_DEPTHWISE:
        raise Error(
            "fit_depthwise_tree is EGrowPolicy::Depthwise only; call"
            " run_tree_layout for SymmetricTree"
        )
    options.check()

    var stat_count = 1 + approx_dim
    var max_leaves = options.max_leaves
    var max_depth = options.max_depth

    var layout = build_layout(fold_counts, one_hot)
    var blocks = blocks_for(layout, n_rows)
    var hist_cells_per_leaf = layout.hist_cells
    var sm_count = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
    if sm_count_override > 0:
        sm_count = sm_count_override

    # `argmaxBlockCount = Min(CeilDivide(binFeatureCountPerDevice, 256), 64)`
    # (`greedy_search_helper.cpp:439`).
    var argmax_blocks = (hist_cells_per_leaf + 255) // 256
    if argmax_blocks > 64:
        argmax_blocks = 64
    if argmax_blocks < 1:
        argmax_blocks = 1

    var wide = (n_rows + 255) // 256
    if wide > 256:
        wide = 256
    if wide < 1:
        wide = 1

    # ---- THE POOLS. The symmetric lane's, unchanged, plus this lane's ----
    if (
        len(ws) == 0
        or ws[0].n_rows_key != n_rows
        or ws[0].stat_count_key != stat_count
        or ws[0].max_leaves_key != max_leaves
        or ws[0].n_features_key != len(fold_counts)
        or ws[0].hist_cells_per_leaf_key != hist_cells_per_leaf
    ):
        ws.clear()
        ws.append(
            TTreeWorkspace(
                ctx, layout, blocks, n_rows, stat_count, max_depth
            )
        )
    if (
        len(dws) == 0
        or dws[0].max_leaves_key != max_leaves
        or dws[0].stat_count_key != stat_count
        or dws[0].argmax_blocks_key != argmax_blocks
    ):
        dws.clear()
        dws.append(
            TDepthwiseWorkspace(ctx, max_leaves, stat_count, argmax_blocks)
        )

    ref hist = ws[0].hist
    ref acc_i32 = ws[0].acc_i32
    ref block_hist = ws[0].block_hist
    ref dblocks = ws[0].dblocks
    ref p_off = ws[0].p_off
    ref p_sz = ws[0].p_sz
    ref hp_off = ws[0].hp_off
    ref hp_sz = ws[0].hp_sz
    ref h_off = ws[0].h_off
    ref h_sz = ws[0].h_sz
    ref part_stats = ws[0].part_stats
    ref stat_partials = ws[0].stat_partials
    ref flags = ws[0].flags
    ref seq = ws[0].seq
    ref gmap = ws[0].gmap
    ref sflags = ws[0].sflags
    ref new_index = ws[0].new_index
    ref new_stats = ws[0].new_stats
    ref chunk_zeros = ws[0].chunk_zeros
    ref chunk_offsets = ws[0].chunk_offsets
    ref leaf_zeros = ws[0].leaf_zeros
    ref skip = ws[0].skip
    ref bff = ws[0].bff
    ref ffw = ws[0].ffw
    ref flat_first = ws[0].flat_first
    ref flat_folds = ws[0].flat_folds
    ref flat_one_hot = ws[0].flat_one_hot
    ref sp_feats = ws[0].sp_feats
    ref sp_feats_h = ws[0].sp_feats_h
    ref sp_bins = ws[0].sp_bins
    ref sp_bins_h = ws[0].sp_bins_h
    ref dense_ids = ws[0].dense_ids

    ref region_score = dws[0].region_score
    ref region_bin = dws[0].region_bin
    ref h_region_score = dws[0].h_region_score
    ref h_region_bin = dws[0].h_region_bin
    ref d_visit = dws[0].d_visit
    ref h_visit = dws[0].h_visit
    ref d_left = dws[0].d_left
    ref h_left = dws[0].h_left
    ref d_right = dws[0].d_right
    ref h_right = dws[0].h_right
    ref d_ids = dws[0].d_ids
    ref h_ids = dws[0].h_ids
    ref h_part_stats = dws[0].h_part_stats

    var table = TBinFeatureTable(layout)

    # ================= CreateInitialSubsets =========================
    # `split_properties_helper.cpp:1043-1080`: zero the partitions, write
    # the root partition over every row, zero the stats and the histograms,
    # push ONE leaf, then `RebuildLeavesSizes`.
    for i in range(max_leaves):
        h_off.unsafe_ptr().unsafe_store(i, UInt32(0))
        h_sz.unsafe_ptr().unsafe_store(i, UInt32(0))
    h_sz.unsafe_ptr().unsafe_store(0, UInt32(n_rows))
    ctx.enqueue_copy(dst_buf=p_off, src_ptr=h_off.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=p_sz, src_ptr=h_sz.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=hp_off, src_ptr=h_off.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=hp_sz, src_ptr=h_sz.unsafe_ptr())
    ctx.enqueue_memset(hist, Float32(0.0))
    ctx.enqueue_memset(acc_i32, Int32(0))

    # The fixed-point scale, host-derived. DEVIATION 95's device derivation
    # is not wired here: it exists to remove the boosting loop's per-tree
    # magnitudes drain, and this lane has no boosting loop yet. Same
    # function, same bits (`choose_scale` is an exact integer search).
    var mag = Float64(weight_magnitude)
    if mag < 0.0:
        mag = -mag
    var gmag = Float64(gradient_magnitude)
    if gmag < 0.0:
        gmag = -gmag
    if gmag > mag:
        mag = gmag
    ws[0].h_scale.unsafe_ptr().unsafe_store(
        0, Float32(choose_scale(mag, n_rows))
    )
    ctx.enqueue_copy(
        dst_buf=ws[0].scale_dev, src_ptr=ws[0].h_scale.unsafe_ptr()
    )
    var fixed_scale = rebind[MutPointer[Float32, MutAnyOrigin]](
        ws[0].scale_dev.unsafe_ptr()
    )

    var leaves = List[TLeaf]()
    var root = TLeaf()
    root.size = n_rows
    leaves.append(root^)
    # the sibling-pairing key of `_leaf_records`; the root has no parent and
    # is `Zeroes`, so its value is never read.
    var parent_of = List[Int]()
    parent_of.append(0)

    var mgr = TCudaManager(ctx.copy(), sync_budget=-1)

    # their `TGpuAwareRandom`, drawn from ONCE PER `ComputeOptimalSplits`
    # call (`greedy_search_helper.cpp:487`), i.e. per LEVEL. Same per-tree
    # re-seed as `run_tree_layout` (DEVIATION 139).
    var level_rand = TRandom(random_seed)

    var result_paths = List[TLeafPath]()
    var result_weights = List[Float64]()
    var result_values = List[List[Float32]]()

    # Their `while (true)`. The bound is OURS: Depthwise splits every live
    # leaf at most once per iteration and `IsTerminalLeaf` stops at
    # `MaxDepth`, so `max_depth + 2` iterations is unreachable unless the
    # bookkeeping has broken. Their loop has no bound and would spin; this
    # raises with the state that made it spin.
    var iteration = 0
    while True:
        iteration += 1
        if iteration > max_depth + 2:
            raise Error(
                String("depthwise level loop did not terminate in ")
                + String(max_depth + 2)
                + " iterations; leaves="
                + String(len(leaves))
                + " (a leaf is neither terminal nor improving and its"
                " histogram is being rebuilt forever)"
            )

        # ================= ComputeOptimalSplits =====================
        # `greedy_search_helper.cpp:396`. One draw per level, before the
        # launch, regardless of which calcer the score function selects.
        var level_seed = level_rand.next_uniform_l()

        # --- SplitPropsHelper.BuildNecessaryHistograms(subsets) ---
        var records = _leaf_records(leaves, parent_of)
        var plan = build_necessary_histograms(records)
        var non_zero = non_zero_leaves(records, plan.compute_ids)

        if len(plan.compute_ids) > 0:
            # their `ZeroLeavesHistograms(zeroLeaves, subsets)`
            # (`:1350`), applied to EVERY compute slot rather than only the
            # empty ones. A computed slot is overwritten by the build
            # either way and an empty leaf's zeros ARE its histogram, so
            # this subsumes their split; it is the symmetric lane's choice
            # and is kept identical so the two cannot drift.
            for i in range(len(plan.compute_ids)):
                h_ids.unsafe_ptr().unsafe_store(i, plan.compute_ids[i])
            ctx.enqueue_copy(dst_buf=d_ids, src_ptr=h_ids.unsafe_ptr())
            ctx.enqueue_function[zero_histograms_kernel](
                d_ids.unsafe_ptr(),
                Int32(hist_cells_per_leaf),
                hist.unsafe_ptr(),
                grid_dim=(
                    (hist_cells_per_leaf + 255) // 256,
                    len(plan.compute_ids),
                    stat_count,
                ),
                block_dim=(256, 1, 1),
            )
            mgr.stream_kernel()

        if len(non_zero) > 0:
            # their `ComputeSplitProperties(loadPolicy, nonZeroComputeLeaves,
            # subsets)` (`:1347`). `ids` must be the NON-EMPTY set and the
            # call must not happen at all when it is empty -- their
            # `if (leavesToCompute.size() == 0) { return; }` (`:1089`).
            for i in range(len(non_zero)):
                h_ids.unsafe_ptr().unsafe_store(i, non_zero[i])
            ctx.enqueue_copy(dst_buf=d_ids, src_ptr=h_ids.unsafe_ptr())
            launch_histograms_for_blocks[hist2_smem_mode](
                ctx, dblocks, iteration - 1, len(non_zero), n_rows,
                stat_count, max_leaves, sm_count, fixed_scale,
                cindex, row_index, stats, p_off, p_sz, d_ids, dense_ids,
                hist, acc_i32, block_hist, hist_cells_per_leaf,
            )
            mgr.stream_kernel()

            # their `TScanHistogramsKernel`, over the BUILT set. A prefix
            # sum is linear, so the derived sibling needs no scan and an
            # all-zero slot scans to itself.
            ctx.enqueue_function[scan_histograms_kernel](
                d_ids.unsafe_ptr(),
                flat_first.unsafe_ptr(),
                flat_folds.unsafe_ptr(),
                flat_one_hot.unsafe_ptr(),
                Int32(len(fold_counts)),
                Int32(hist_cells_per_leaf),
                hist.unsafe_ptr(),
                grid_dim=(
                    (len(fold_counts) + 255) // 256,
                    len(non_zero),
                    stat_count,
                ),
                block_dim=(256, 1, 1),
            )
            mgr.stream_kernel()

        if len(plan.subtract_from) > 0:
            # their `SubstractHistograms(bigLeaves, smallLeaves, subsets)`
            # (`:1354`): `from - what`, in place, one launch for all pairs.
            for i in range(len(plan.subtract_from)):
                h_left.unsafe_ptr().unsafe_store(i, plan.subtract_from[i])
                h_right.unsafe_ptr().unsafe_store(i, plan.subtract_what[i])
            ctx.enqueue_copy(dst_buf=d_left, src_ptr=h_left.unsafe_ptr())
            ctx.enqueue_copy(dst_buf=d_right, src_ptr=h_right.unsafe_ptr())
            ctx.enqueue_function[substract_histograms_kernel](
                d_left.unsafe_ptr(),
                d_right.unsafe_ptr(),
                Int32(hist_cells_per_leaf),
                hist.unsafe_ptr(),
                grid_dim=(
                    (hist_cells_per_leaf + 255) // 256,
                    len(plan.subtract_from),
                    stat_count,
                ),
                block_dim=(256, 1, 1),
            )
            mgr.stream_kernel()

        # their `allUpdatedLeaves` loop (`:1359-1367`): computed leaves plus
        # derived big leaves become `CurrentPath`, AND THEIR BestSplit IS
        # RESET. The reset is what makes `SelectLeavesToVisit` re-score
        # exactly the leaves whose histogram moved.
        var updated = plan.updated_ids()
        for i in range(len(updated)):
            var id = Int(updated[i])
            leaves[id].histograms_type = EHistogramsType.CurrentPath
            leaves[id].best_split = TBestSplitProperties()

        # their `AllReduceThroughMaster(subsets->CurrentPartStats(), ...)`
        # (`:445`) over leaves `[0, leafCount)`. See DEVIATION 352.
        for i in range(len(leaves)):
            h_ids.unsafe_ptr().unsafe_store(i, UInt32(i))
        ctx.enqueue_copy(dst_buf=d_ids, src_ptr=h_ids.unsafe_ptr())
        compute_partition_stats(
            ctx, len(leaves), n_rows, stat_count, n_rows,
            d_ids, p_off, p_sz, stats, stat_partials, part_stats,
            sm_count=sm_count,
        )
        mgr.stream_kernel()

        var visit = select_leaves_to_visit(leaves)
        if len(visit) > 0:
            # `numScoreBlocks = leavesToVisit.size()` (`:428-432`), and the
            # kernel is `TComputeOptimalSplitsLeafwiseKernel` (`:470-488`).
            for i in range(len(visit)):
                h_visit.unsafe_ptr().unsafe_store(i, UInt32(visit[i]))
            ctx.enqueue_copy(dst_buf=d_visit, src_ptr=h_visit.unsafe_ptr())

            if (
                options.score_function == SCORE_FUNCTION_L2
                or options.score_function == SCORE_FUNCTION_NEWTON_L2
            ):
                ctx.enqueue_function[
                    compute_optimal_splits_region_kernel[SCORE_FUNCTION_L2]
                ](
                    skip.unsafe_ptr(),
                    Int32(hist_cells_per_leaf),
                    bff.unsafe_ptr(),
                    ffw.unsafe_ptr(),
                    hist.unsafe_ptr(),
                    part_stats.unsafe_ptr(),
                    Int32(stat_count),
                    d_visit.unsafe_ptr(),
                    Int32(1) if multiclass_optimization else Int32(0),
                    options.l2_reg,
                    # the L2 calcer has no noise term at all
                    # (`score_calcers.cuh:40-69`); the seed is still handed
                    # over so both arms consume the same stream.
                    Float32(0.0),
                    level_seed,
                    region_score.unsafe_ptr(),
                    region_bin.unsafe_ptr(),
                    grid_dim=(argmax_blocks, len(visit), 1),
                    block_dim=(LEAFWISE_SCORE_BLOCK_SIZE, 1, 1),
                )
            else:
                ctx.enqueue_function[
                    compute_optimal_splits_region_kernel[
                        SCORE_FUNCTION_COSINE
                    ]
                ](
                    skip.unsafe_ptr(),
                    Int32(hist_cells_per_leaf),
                    bff.unsafe_ptr(),
                    ffw.unsafe_ptr(),
                    hist.unsafe_ptr(),
                    part_stats.unsafe_ptr(),
                    Int32(stat_count),
                    d_visit.unsafe_ptr(),
                    Int32(1) if multiclass_optimization else Int32(0),
                    options.l2_reg,
                    options.random_strength,
                    level_seed,
                    region_score.unsafe_ptr(),
                    region_bin.unsafe_ptr(),
                    grid_dim=(argmax_blocks, len(visit), 1),
                    block_dim=(LEAFWISE_SCORE_BLOCK_SIZE, 1, 1),
                )
            mgr.stream_kernel()

            # ===== HOST WAIT ONE OF TWO: their `bestProps.Read(propsCpu)`
            # (`greedy_search_helper.cpp:517`). =====
            ctx.enqueue_copy(
                dst_ptr=h_region_score.unsafe_ptr(), src_buf=region_score
            )
            ctx.enqueue_copy(
                dst_ptr=h_region_bin.unsafe_ptr(), src_buf=region_bin
            )
            mgr.wait_complete()

            # their cross-block reduce, `:520-531`, ONE WINNER PER SCORE
            # BLOCK where the symmetric arm reduces to one for the level:
            #
            #     for (scoreBlockId ...) {
            #         blockProps = propsCpu.data() + scoreBlockId * argmaxBlockCount;
            #         for (i < argmaxBlockCount)
            #             if (blockProps[i] < bestSplits[scoreBlockId])
            #                 bestSplits[scoreBlockId] = blockProps[i];
            #     }
            #
            # `operator<` orders by GAIN then FeatureId (as ui32) then BinId
            # -- `best_split_properties_less`, already ported for the
            # doc-parallel searcher. A SEQUENTIAL fold under a total order,
            # so the block count cannot move the answer.
            #
            # THE RECORD LAYOUT IS TRANSPOSED FROM THEIRS. Their kernel
            # writes `result += blockIdx.x + blockIdx.y * gridDim.x`, i.e.
            # block-major within a score block; ours writes the same slot,
            # so `leaf i`'s records are `[i*argmax_blocks, (i+1)*argmax_blocks)`.
            for i in range(len(visit)):
                var best = TBestSplitProperties()
                for b in range(argmax_blocks):
                    var slot = i * argmax_blocks + b
                    var bf = h_region_bin.unsafe_ptr().unsafe_load(slot)
                    if bf == UInt32(0xFFFFFFFF):
                        # their poison record IS a default-constructed
                        # `TBestSplitProperties`, so skipping it and
                        # comparing against it are the same thing.
                        continue
                    if Int(bf) >= hist_cells_per_leaf:
                        raise Error(
                            String("score kernel returned bin-feature ")
                            + String(Int(bf))
                            + " outside the histogram's "
                            + String(hist_cells_per_leaf)
                            + " cells"
                        )
                    var our_gain = h_region_score.unsafe_ptr().unsafe_load(
                        slot
                    )
                    # THEIR sign, restored for the comparator: this port's
                    # gain is theirs negated (see `kernel/compute_scores`),
                    # and `best_split_properties_less` is a transcription of
                    # THEIR `operator<`.
                    var split = table.to_split(Int(bf))
                    var cand = TBestSplitProperties(
                        split.feature_id,
                        split.bin_idx,
                        -our_gain,
                        -our_gain,
                    )
                    if best_split_properties_less(cand, best):
                        best = cand
                # `subsets->Leaves[leafId].UpdateBestSplit(bestSplits[i])`
                # (`:551`). NOTE what is NOT here: the oblivious arm's
                # `CB_ENSURE(FeatureId != (ui32)-1, "All splits have
                # infinite score...")` (`:535`) is inside
                # `if (IsObliviousSplit())`. A non-oblivious leaf whose
                # every candidate was unusable simply keeps an UNDEFINED
                # best split and is never selected to split.
                leaves[visit[i]].update_best_split(best)

        # ===================== SplitLeaves ==========================
        # `greedy_search_helper.cpp:575`. `HaveFixedSplits` is absent: the
        # option that feeds it is refused by name (see
        # `structure_searcher_options.mojo`).
        var to_split = select_leaves_to_split(leaves)

        if len(to_split) > 0:
            # --- MakeSplit's multi-leaf arm, `split_properties_helper
            # .cpp:845-950`. `leftId = leavesToSplit[i]` keeps the parent's
            # partition slot and `rightId = leavesCount + i` is fresh.
            var leaves_count = len(leaves)
            var sp_bytes = sp_feats_h.unsafe_ptr()
            for i in range(len(to_split)):
                var left_id = to_split[i]
                var right_id = leaves_count + i
                var bs = leaves[left_id].best_split
                if not bs.defined:
                    raise Error(
                        String("Best split is undefined for leaf ")
                        + String(left_id)
                    )
                var split = TBinarySplit(
                    bs.feature_id,
                    bs.bin_id,
                    Int32(
                        BIN_SPLIT_TAKE_BIN
                    ) if layout.features[
                        Int(bs.feature_id)
                    ].one_hot_feature else Int32(BIN_SPLIT_TAKE_GREATER),
                )

                # their `splitsFeaturesBuilder.Add(DataSet.GetTCFeature(
                # splitFeature.FeatureId))` (`:875`) and
                # `splitBins.push_back(splitFeature.BinIdx)` (`:876`).
                # `CFeature` is one struct in the kernel, so the array is
                # raw bytes; see `make_split_features_buffers`.
                # ============ THE OFFSET IS IN ELEMENTS, NOT COLUMNS ============
                # `DataSet.GetTCFeature(featureId)` (`split_properties_helper
                # .cpp:875`) hands the split kernel a `TCFeature` whose
                # `Offset` is what `compressedIndex + feature.Offset +
                # loadIndex` indexes with (`split_points.cu:518`). In THIS
                # port's layout that is `column * n_rows`, and
                # `CompressedIndexLayout.features[f].offset` is the bare
                # COLUMN. The symmetric arm multiplies at the point it builds
                # its resolve table (`TTreeWorkspace`'s `bfr_off`,
                # `tf2.offset * UInt32(n_rows)`), and this is the same
                # multiply at this arm's equivalent point.
                #
                # PASSING THE BARE COLUMN DOES NOT CRASH AND DOES NOT LOOK
                # WRONG. It reads `cindex[column + row]` instead of
                # `cindex[column * n_rows + row]`, which is a real bin of a
                # real feature for almost every row, so the tree still grows,
                # still conserves rows, and still picks different features in
                # different leaves. The first thing that saw it was
                # `depthwise_check` claim 4 -- the apply kernel, which
                # computes the offset the other way -- at 852 rows in a bin
                # growth had put 243 in. A gate that only looked at the
                # partition would have called this green.
                var f = layout.features[Int(bs.feature_id)]
                var packed = CFeature(
                    f.offset * UInt32(n_rows),
                    f.mask,
                    f.shift,
                    f.first_fold_index,
                    f.folds,
                    f.one_hot_feature,
                )
                # `CFEATURE_BYTES` is `size_of[CFeature]()`, so indexing the
                # bitcast pointer by the slot IS their `splitsFeaturesBuilder`
                # writing element `i` of a `TCFeature` array; the byte buffer
                # exists only because Mojo device buffers are DType-shaped.
                var dst = sp_bytes.bitcast[CFeature]()
                dst[unsafe_offset=i] = packed
                sp_bins_h.unsafe_ptr().unsafe_store(i, UInt32(bs.bin_id))
                h_left.unsafe_ptr().unsafe_store(i, UInt32(left_id))
                h_right.unsafe_ptr().unsafe_store(i, UInt32(right_id))

                # `TLeaf leaf = subsets->Leaves[leftId];` -- the SNAPSHOT.
                # Both children are derived from the parent, so the parent
                # must be read before the left slot is overwritten.
                var parent = leaves[left_id].copy()
                var left = split_leaf(parent, split, SPLIT_VALUE_ZERO)
                var right = split_leaf(parent, split, SPLIT_VALUE_ONE)
                leaves[left_id] = left^
                leaves.append(right^)
                # the sibling key: both children's parent is the id the
                # left child kept.
                parent_of[left_id] = left_id
                parent_of.append(left_id)

            var n_split = len(to_split)
            ctx.enqueue_copy(dst_buf=sp_feats, src_ptr=sp_feats_h.unsafe_ptr())
            ctx.enqueue_copy(dst_buf=sp_bins, src_ptr=sp_bins_h.unsafe_ptr())
            ctx.enqueue_copy(dst_buf=d_left, src_ptr=h_left.unsafe_ptr())
            ctx.enqueue_copy(dst_buf=d_right, src_ptr=h_right.unsafe_ptr())

            # their `TSplitPointsKernel`, whose five steps are five calls
            # here (`split_points.cpp:64-136`): flag and sequence, stable
            # partition, segmented gather of the index and every stat
            # column, copy the histogram to the new leaf, update the
            # partitions. IDENTICAL CALLS TO THE SYMMETRIC LANE'S -- only
            # the id arrays differ.
            ctx.enqueue_function[split_and_make_sequence_kernel](
                cindex.unsafe_ptr(),
                row_index.unsafe_ptr(),
                p_off.unsafe_ptr(),
                p_sz.unsafe_ptr(),
                d_left.unsafe_ptr(),
                sp_feats.unsafe_ptr().bitcast[CFeature](),
                sp_bins.unsafe_ptr(),
                flags.unsafe_ptr(),
                seq.unsafe_ptr(),
                grid_dim=(
                    split_points_grid_x(n_split, sm_count), n_split, 1
                ),
                block_dim=(SPLIT_BLOCK_SIZE, 1, 1),
            )
            mgr.stream_kernel()

            launch_stable_partition(
                ctx, n_split, n_rows, d_left, p_off, p_sz, flags,
                chunk_zeros, chunk_offsets, leaf_zeros, gmap, sflags,
                sm_count=sm_count,
            )
            mgr.stream_kernel()

            var reorder_launches = launch_reorder_in_leaves(
                ctx, n_split, wide, n_rows, stat_count, n_rows,
                d_left, p_off, p_sz, stats, new_stats, row_index,
                new_index, gmap, sm_count=sm_count,
            )
            for _ in range(reorder_launches):
                mgr.stream_kernel()

            # their `CopyHistogram(LeafIdToSplit, RightLeafIdAfterSplit)`
            # (`split_points.cpp:326`). The left child kept the parent's
            # slot; this puts the same histogram in the right child's, so
            # both are `PreviousPath` and next level can pair them.
            ctx.enqueue_function[copy_histograms_kernel](
                d_left.unsafe_ptr(),
                d_right.unsafe_ptr(),
                Int32(stat_count),
                Int32(hist_cells_per_leaf),
                hist.unsafe_ptr(),
                grid_dim=(
                    (hist_cells_per_leaf * stat_count + 255) // 256,
                    n_split,
                    1,
                ),
                block_dim=(256, 1, 1),
            )
            mgr.stream_kernel()

            ctx.enqueue_function[update_partitions_after_split_kernel](
                d_left.unsafe_ptr(),
                d_right.unsafe_ptr(),
                Int32(n_split),
                sflags.unsafe_ptr(),
                p_off.unsafe_ptr(),
                p_sz.unsafe_ptr(),
                hp_off.unsafe_ptr(),
                hp_sz.unsafe_ptr(),
                grid_dim=(
                    split_points_grid_x(n_split, sm_count), n_split, 1
                ),
                block_dim=(512, 1, 1),
            )
            mgr.stream_kernel()

            # ===== HOST WAIT TWO OF TWO: `RebuildLeavesSizes`
            # (`split_properties_helper.cpp:800-812`). Theirs reads the
            # PINNED mirror with no copy; ours copies, for the reason in
            # `gpu_util/gpu_data/partitions.mojo`'s deviation block. =====
            ctx.enqueue_copy(dst_ptr=h_sz.unsafe_ptr(), src_buf=p_sz)
            mgr.wait_complete()
            for i in range(len(leaves)):
                leaves[i].size = Int(h_sz.unsafe_ptr().unsafe_load(i))

            # `MarkTerminal(leftIds, ...)` then `MarkTerminal(rightIds, ...)`
            # (`greedy_search_helper.cpp:617-618`), AFTER the sizes are
            # rebuilt -- `IsTerminalLeaf` reads `leaf.Size`.
            for i in range(n_split):
                var left_id = to_split[i]
                var right_id = leaves_count + i
                leaves[left_id].is_terminal = is_terminal_leaf(
                    leaves[left_id], options
                )
                leaves[right_id].is_terminal = is_terminal_leaf(
                    leaves[right_id], options
                )
        else:
            # `for (i ...) subsets.Leaves[i].IsTerminal = true;` (`:620-622`)
            for i in range(len(leaves)):
                leaves[i].is_terminal = True

        if should_terminate(leaves, options):
            # ============== the leaf values, `:625-650` ==============
            # `numStats` is their `PartitionStats.SingleObjectSize()`.
            # The partitions moved in the split above, so the stats are
            # recomputed here (DEVIATION 352) before being read.
            for i in range(len(leaves)):
                h_ids.unsafe_ptr().unsafe_store(i, UInt32(i))
            ctx.enqueue_copy(dst_buf=d_ids, src_ptr=h_ids.unsafe_ptr())
            compute_partition_stats(
                ctx, len(leaves), n_rows, stat_count, n_rows,
                d_ids, p_off, p_sz, stats, stat_partials, part_stats,
                sm_count=sm_count,
            )
            mgr.stream_kernel()
            ctx.enqueue_copy(
                dst_ptr=h_part_stats.unsafe_ptr(), src_buf=part_stats
            )
            mgr.wait_complete()

            var num_leaves = len(leaves)
            for leaf_id in range(num_leaves):
                var w = Float64(
                    h_part_stats.unsafe_ptr().unsafe_load(
                        leaf_id * stat_count
                    )
                )
                result_weights.append(w)

                var values = List[Float32]()
                var total_sum = Float64(0.0)
                for approx_id in range(stat_count - 1):
                    var v = Float32(0.0)
                    if w > 1e-20:
                        v = Float32(
                            Float64(
                                h_part_stats.unsafe_ptr().unsafe_load(
                                    leaf_id * stat_count + 1 + approx_id
                                )
                            )
                            / (w + Float64(options.l2_reg))
                        )
                    values.append(v)
                    total_sum += Float64(v)
                if multiclass_optimization:
                    # `for (approxId ...) resultValues[leafId][approxId]
                    #  += totalSum;` (`:644-646`)
                    for approx_id in range(stat_count - 1):
                        values[approx_id] += Float32(total_sum)
                result_values.append(values^)
                result_paths.append(leaves[leaf_id].path.copy())
            break

    # `return BuildTreeLikeModel<TModel>(leaves, leavesWeights, leavesValues)`
    return build_non_symmetric_tree(
        result_paths, result_weights, result_values
    )
