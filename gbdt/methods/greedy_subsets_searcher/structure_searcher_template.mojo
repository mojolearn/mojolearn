# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# This lane MIRRORS CatBoost (pinned commit 54a8143a). Per-file provenance is in gbdt/DERIVATION_MAP.tsv, in this file's own docstring, and in NOTICE; files with no CatBoost counterpart say so.
"""The level loop: grow one oblivious tree.

PORT OF `TGreedyTreeLikeStructureSearcher::FitImpl` in
`catboost/cuda/methods/greedy_subsets_searcher/structure_searcher_template.h`
and the two calls it makes into `greedy_search_helper.cpp`, at CatBoost
`54a8143a`. Transliterated. Do not improve.

Their whole tree is five lines:

    TPointsSubsets subsets = searchHelper.CreateInitialSubsets(objective);
    while (true) {
        searchHelper.ComputeOptimalSplits(&subsets);
        if (!searchHelper.SplitLeaves(&subsets, &leaves, &weights, &values))
            break;
    }

**Under `SymmetricTree` one iteration is one LEVEL, not one split**
(`greedy_search_helper.cpp:358-366`, `:534-545`), and `numScoreBlocks` is 1
(`:422-425`) because the whole level shares a single split. That is the
difference from a leaf-wise grower in one number, and it is why the launch
count does not depend on the leaf count.

WHAT THIS FILE IS AND IS NOT
----------------------------
It is the SEQUENCE: which kernels run per level, in what order, with what
data dependencies. It is not the device plumbing. Buffers, launch geometry
and the compressed-index builder do not exist yet, so `grow_tree` describes
the schedule and records what each step needs, and the driver that binds it
to real memory comes next.

Writing the sequence first is deliberate. The ordering constraint discovered
while porting `compute_scores` -- that the bin prefix scan must run in its
own kernel BEFORE scoring, or the score kernel loses its parallel shape --
is a property of this file, not of either kernel, and it is the kind of thing
that gets lost if the loop is written last.
"""

from gbdt.methods.greedy_subsets_searcher.split_properties_helper import (
    HISTOGRAMS_PREVIOUS_PATH,
    HISTOGRAMS_ZEROES,
    LeafRecord,
    LevelPlan,
    build_necessary_histograms,
)


#: One step of a level, in the order it must run.
comptime STEP_BUILD_HISTOGRAMS = 0
comptime STEP_SUBTRACT_SIBLINGS = 1
comptime STEP_SCAN_BINS = 2
comptime STEP_COMPUTE_SCORES = 3
comptime STEP_SPLIT_POINTS = 4
comptime STEP_REORDER = 5


def step_name(step: Int) -> String:
    if step == STEP_BUILD_HISTOGRAMS:
        return String("build histograms (smaller child only)")
    if step == STEP_SUBTRACT_SIBLINGS:
        return String("subtract siblings (one batched launch)")
    if step == STEP_SCAN_BINS:
        return String("scan bins (buys the score kernel its shape)")
    if step == STEP_COMPUTE_SCORES:
        return String("score candidates, leaf loop serial in-thread")
    if step == STEP_SPLIT_POINTS:
        return String("flag rows by side, write in-leaf sequence")
    return String("reorder each leaf range, stable")


@fieldwise_init
struct LevelSchedule(Copyable, Movable):
    """What one level costs, before any of it runs.

    Launch counts are the point: CatBoost's rule is that the number of
    kernels per level does not depend on the leaf count or on the dataset
    (`compute_by_blocks_helper.h:87-92`). Recording them here means a
    regression in that property is visible without a profiler.
    """

    var depth: Int
    var leaf_count: Int
    var histogram_builds: Int
    var subtraction_pairs: Int
    var launches: Int

    def launches_per_leaf(self) -> Float64:
        """Should FALL as the level widens. If it is flat, something in the
        schedule has become per-leaf and the design property is lost."""
        if self.leaf_count == 0:
            return 0.0
        return Float64(self.launches) / Float64(self.leaf_count)


def plan_level(
    leaves: List[LeafRecord], depth: Int, feature_blocks: Int
) raises -> LevelSchedule:
    """The schedule for one level, and the launch count it implies.

    `feature_blocks` is how many grouping policies are live (binary,
    half-byte, one-byte), since each needs its own histogram launch: that is
    CatBoost's `B` in `3B + 12`.

    The count, from the census in the CatBoost GPU study:
        B memsets + B histogram kernels + B writebacks
        + 1 scan + 1 subtraction + 1 score + 1 flag + ~4 gathers
        + 1 histogram copy + 1 boundary + 2 stat sums

    **This is the DESIGN's launch count and NOT what CatBoost actually pays.**
    On top of it they run one `cub::DeviceRadixSort::SortPairs` PER LEAF in a
    host loop (`split_points.cu:658-689`), which is 255 more for a depth-8
    tree and takes a real run from 168 to roughly 423. Their own comments
    call that wrong: "cub sucks for this, write proper segmented version" and
    "for oblivious trees we have overhead for launching kernel per leaf".

    The port substitutes a batched stable partition, so if it holds up we get
    the 168 without the 255. Worth knowing before comparing this number to a
    profile of real CatBoost, which will show the larger one.
    """
    var plan = build_necessary_histograms(leaves)
    var b = feature_blocks
    var launches = 3 * b + 12
    return LevelSchedule(
        depth,
        len(leaves),
        len(plan.compute_ids),
        len(plan.subtract_from),
        launches,
    )


def grow_tree_schedule(
    max_depth: Int, n_rows: UInt32, feature_blocks: Int
) raises -> List[LevelSchedule]:
    """`FitImpl`'s loop, as the schedule it produces for a balanced tree.

    Starts from one leaf holding every row (`CreateInitialSubsets`) and
    doubles per level, which is what `SymmetricTree` growth does: every leaf
    of the level takes the SAME split, so the level ends with `2^(d+1)`
    leaves and no leaf-wise frontier to track.

    A balanced tree is the honest shape to schedule against here because the
    launch count does not depend on the leaf sizes at all; only the histogram
    BUILD count does, and that is what `build_necessary_histograms` decides
    from the real sizes at run time.
    """
    var out = List[LevelSchedule]()
    var leaves = List[LeafRecord]()
    leaves.append(LeafRecord(n_rows, HISTOGRAMS_ZEROES, 0, False))

    for depth in range(max_depth):
        out.append(plan_level(leaves, depth, feature_blocks))

        # Split every leaf: the oblivious step. Each parent becomes two
        # children sharing a path id, and the halves are equal only because
        # this is the balanced schedule; the real sizes come from the device.
        var next = List[LeafRecord]()
        for i in range(len(leaves)):
            var half = leaves[i].size // 2
            var terminal = depth + 1 >= max_depth
            next.append(LeafRecord(half, HISTOGRAMS_ZEROES, i, terminal))
            next.append(
                LeafRecord(
                    leaves[i].size - half, HISTOGRAMS_ZEROES, i, terminal
                )
            )
        leaves = next^

    return out^
