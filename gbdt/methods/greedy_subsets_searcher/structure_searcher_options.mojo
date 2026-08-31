# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# This lane MIRRORS CatBoost (pinned commit 54a8143a). Per-file provenance is in the root DERIVATION_MAP.tsv, in this file's own docstring, and in NOTICE; files with no CatBoost counterpart say so.
"""The eleven knobs the structure searcher reads, and nothing else.

PORT OF `TTreeStructureSearcherOptions` in
`catboost/cuda/methods/greedy_subsets_searcher/structure_searcher_options.h`
at CatBoost `54a8143a`. Transliterated. Do not improve.

Theirs is a plain struct with defaults, built once per fit by
`CreateStructureSearcher` and then read by `TGreedySearchHelper` at every
level. It is worth having as its own type rather than eleven parameters
because THREE of its fields change meaning with `Policy`, and a struct is
where that can be said once:

* `MaxLeaves` is a real bound only under Lossguide; for the other three
  policies `catboost_options.cpp:993-1001` pins it to `1 << MaxDepth`, where
  it is reached exactly when the depth bound is.
* `MinLeafSize` is DEAD under SymmetricTree and LIVE under every other
  policy -- `IsTerminalLeaf` guards the size test with
  `Options.Policy != EGrowPolicy::SymmetricTree`
  (`greedy_search_helper.cpp:691`). This lane is the first to make it live.
* `ModelSizeReg` is read ONLY inside the SymmetricTree branch of
  `ComputeOptimalSplits` (`:447`, `UpdateFeatureWeightsForBestSplits`), so
  Depthwise and Lossguide never run it. It is carried anyway, because the
  field exists in their struct for all four policies and a missing field is
  a silent divergence the day a fourth policy reads it.

TWO FIELDS OF THEIRS ARE NOT HERE, both deliberately.
-----------------------------------------------------
`BootstrapOptions` (`TBootstrapConfig`) is not a field because in this port
the BOOSTING DRIVER bootstraps and hands the searcher stat planes that are
already sampled -- their `ComputeTarget` calls `objective.StochasticDer(
bootstrapConfig, ...)` INSIDE `CreateInitialSubsets`
(`greedy_search_helper.cpp:376-380`), and ours does it one level up. Adding
the field here would put a second, unread copy of the bootstrap
configuration in the tree. Recorded in `UNWIRED.md` with the call site that
would consume it if the split ever moves.

`FixedBinarySplits` is not here either. It is CatBoost's mechanism for
forcing the first `k` levels onto named binary features
(`greedy_search_helper.cpp:394`, `:470`, `:606`) and it is fed only by
`fixed_binary_splits`, an option this port refuses by name in
`catboost_options.check()`. A refused option must not grow a field that
reads as supported.
"""

from gbdt.options.catboost_options import (
    GROW_DEPTHWISE,
    GROW_LOSSGUIDE,
    GROW_SYMMETRIC,
    SCORE_FUNCTION_COSINE,
    SCORE_FUNCTION_L2,
    SCORE_FUNCTION_NEWTON_COSINE,
    SCORE_FUNCTION_NEWTON_L2,
    grow_policy_name,
)


@fieldwise_init
struct TTreeStructureSearcherOptions(Copyable, Movable):
    """`TTreeStructureSearcherOptions` (`structure_searcher_options.h:12`).

    Their defaults, exactly: Cosine, 64 leaves, depth 6, L2 3.0, model size
    reg 0.5, SymmetricTree, min leaf size 1, random strength 0.

    `random_strength` arrives ALREADY MULTIPLIED by the boosting loop's
    `CalcScoreModelLengthMult` -- their `CreateStructureSearcher` does
    `options.RandomStrength *= randomStrengthMult`
    (`greedy_subsets_searcher.h:73-76`) before the searcher ever sees it.
    Same contract as `run_tree_layout`'s parameter of the same name.
    """

    var score_function: Int
    """`ScoreFunction`, default `EScoreFunction::Cosine`."""

    var max_leaves: Int
    """`MaxLeaves`, default 64. See the module docstring: a real bound only
    under Lossguide."""

    var max_depth: Int
    """`MaxDepth`, default 6."""

    var l2_reg: Float32
    """`L2Reg`, default 3.0. Theirs is `double`; the whole score path in this
    port is Float32 for the Metal reason recorded in
    `kernel/compute_scores.mojo`."""

    var model_size_reg: Float32
    """`ModelSizeReg`, default 0.5. Read by the SymmetricTree branch only."""

    var policy: Int
    """`Policy`, default `EGrowPolicy::SymmetricTree`."""

    var min_leaf_size: Float64
    """`MinLeafSize`, default 1.

    THEIRS IS A DOUBLE AND THE COMPARISON IS `leaf.Size <= MinLeafSize`
    (`greedy_search_helper.cpp:693`), NOT `<`. So the default of 1 marks a
    one-row leaf terminal, and `min_data_in_leaf=1` means "a leaf of one row
    does not split" rather than "a leaf of one row is allowed". Getting that
    boundary wrong builds a tree one row deeper than CatBoost's on every
    unbalanced branch. The type stays Float64 because their comparison is
    against a double and an integer field would quietly change which side of
    a fractional bound a leaf falls on."""

    var random_strength: Float32
    """`RandomStrength`, default 0, already multiplied. Zero is their off
    switch."""

    var feature_weights: List[Float32]
    """`FeatureWeights`. One per FEATURE (not per bin-feature), multiplied
    into every candidate's gain (`compute_scores.cu:136-137`). Empty means
    the searcher fills 1.0 for every feature, which is what
    `UpdateFeatureWeightsForBestSplits` leaves them at when there are no
    CTRs (`update_feature_weights.cpp:14-22`)."""

    def __init__(out self):
        """Their aggregate defaults, field for field."""
        self.score_function = SCORE_FUNCTION_COSINE
        self.max_leaves = 64
        self.max_depth = 6
        self.l2_reg = Float32(3.0)
        self.model_size_reg = Float32(0.5)
        self.policy = GROW_SYMMETRIC
        self.min_leaf_size = Float64(1.0)
        self.random_strength = Float32(0.0)
        self.feature_weights = List[Float32]()

    def check(self) raises:
        """What this LANE can honor, refused by name rather than ignored.

        There is no counterpart of this in their struct -- theirs is a POD
        and the validation lives in `catboost_options.cpp`. It is here
        because a searcher option that is accepted and dropped is exactly
        the failure `PORTING_RULES.md` rule 3 exists to prevent, and the
        depthwise lane is the first caller that can reach a policy this
        struct does not implement.
        """
        # SymmetricTree, Depthwise and Lossguide each have a searcher in
        # this directory; `EGrowPolicy::Region` has none and no lane.
        #
        # THIS STRUCT DOES NOT DISPATCH. Each searcher raises on a policy
        # that is not its own -- `fit_depthwise_tree`'s first statement --
        # so accepting a policy here is not accepting it anywhere. What it
        # refuses is a policy with no implementation at all, which is the
        # only thing an options struct can honestly refuse.
        if (
            self.policy != GROW_SYMMETRIC
            and self.policy != GROW_DEPTHWISE
            and self.policy != GROW_LOSSGUIDE
        ):
            raise Error(
                String("grow_policy=")
                + grow_policy_name(self.policy)
                + " has no searcher in this port. SymmetricTree is"
                " gbdt/methods/greedy_subsets_searcher/"
                "greedy_search_helper.mojo, Depthwise is"
                " greedy_search_helper_depthwise.mojo, Lossguide is"
                " greedy_search_helper_lossguide.mojo; EGrowPolicy::Region"
                " is unported."
            )
        if self.max_depth < 1:
            raise Error(
                String("max_depth must be at least 1, got ")
                + String(self.max_depth)
            )
        # `catboost_options.cpp:993-1001`: every policy but Lossguide is
        # pinned to `1 << MaxDepth`, and a user value that differs is
        # REFUSED rather than clamped.
        if self.policy != GROW_LOSSGUIDE:
            var pinned = 1 << self.max_depth
            if self.max_leaves != pinned:
                raise Error(
                    String("max_leaves option works only with lossguide")
                    + " tree growing; for "
                    + grow_policy_name(self.policy)
                    + " CatBoost requires max_leaves == 1 << depth == "
                    + String(pinned)
                    + ", got "
                    + String(self.max_leaves)
                )
        # ============ THE SCORE FUNCTION, REFUSED BY NAME ============
        # Their launcher has a case per calcer and `default: { throw
        # std::exception(); }` (`compute_scores.cu:509-546`). This port has
        # calcers for two of the five: Cosine/NewtonCosine and L2/NewtonL2.
        #
        # THE DRIVER'S DISPATCH IS AN `if L2 else COSINE`, so SolarL2, SatL2
        # and LOOL2 -- all live constants in `catboost_options` -- fell into
        # the `else` and SILENTLY GOT THE COSINE CALCER. `catboost_options
        # .check()` refuses them, but no searcher calls that, so nothing
        # stood between a caller and a wrong score. Their `default: throw`
        # is what belongs here, and this is it. Found by an audit against
        # their source, 2026-08-22.
        if (
            self.score_function != SCORE_FUNCTION_COSINE
            and self.score_function != SCORE_FUNCTION_NEWTON_COSINE
            and self.score_function != SCORE_FUNCTION_L2
            and self.score_function != SCORE_FUNCTION_NEWTON_L2
        ):
            raise Error(
                String("score_function=")
                + String(self.score_function)
                + " has no calcer in this port; only Cosine, NewtonCosine,"
                " L2 and NewtonL2 are ported (SolarL2, SatL2 and LOOL2 are"
                " theirs at compute_scores.cu:509-546 and are not written)"
            )
        if self.min_leaf_size < Float64(0.0):
            raise Error(
                String("min_data_in_leaf must be non-negative, got ")
                + String(self.min_leaf_size)
            )
