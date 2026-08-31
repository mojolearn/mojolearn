# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The EXACT host oracle: `node_split_random`, transcribed, on our keyed draws.

    Verified by:  extratrees/checks/host_splitter_check.mojo

WHAT THIS FILE IS, said plainly because its directory does not say it. It is a
HOST ORACLE. It is a TRANSCRIPTION of scikit-learn `1.9.0` (`77def0e`)
`sklearn/tree/_splitter.pyx:507-720` (`node_split_random`) -- their branches, in
their order, with each branch citing the line it came from -- driven by this
lane's counter-based keyed draws instead of their sequential `our_rand_r`
stream (DEVIATION 130). Its JOB is to be the reference the device kernels of
`impl/decisiontree/batched_levelalgo/` are measured against, per node and per
candidate feature; in that comparison it is the authority, and
`extratrees/DERIVATION_MAP.tsv` carries its row (upstream `scikit-learn`,
`partial`) with the four branches of theirs that are not here. It sits under
`checks/` because of that ROLE and for no other reason: THE DIRECTORY NAME IS
NOT A PROVENANCE CLAIM, and it once read like one: this directory was called
`mojo_only/` and then `original/` before `checks/`, and under either of those
names the placement implied a file that is underived. This one is not, and
never was; `checks/` names the ROLE, which is the only claim the shelf makes.

THE SPEC, BRANCH BY BRANCH, AND WHERE EACH ONE WENT
---------------------------------------------------

| sklearn                | here                                              |
|------------------------|---------------------------------------------------|
| `:560` `_init_split`   | `Split()` (cuML's default ctor) + `best_index=-1`  |
| `:573-577` loop guard  | DEVIATION 151 -- the supplied `colids` IS the loop |
| `:591-603` Fisher-Yates| DEVIATION 131 -- cuML's sampler, supplied by caller|
| `:605` pick the feature| `colids[ci]`                                      |
| `:607-611` `find_min_max` | `node_feature_min_max` (`_partitioner.pyx:129-165`) |
| `:613-618` constant test  | `node_feature_is_constant`, all three arms     |
| `:619-626` constant skip  | record it and `continue`                       |
| `:628-629` swap into the drawn region | nothing -- no permutation state  |
| `:630`, `:639-651` missing| DEVIATION 136 -- a NaN is an ERROR, not a coin flip |
| `:632-637` `rand_uniform` | `draw_threshold` (RAFT PCG, keyed)             |
| `:653-654` `== max -> min`| inside `draw_threshold`, cited there           |
| `:656-659` partition      | DEVIATION 152 -- COUNT here, partition the winner once |
| `:661-662` n_left/n_right | `count`-relative, same arithmetic              |
| `:664-666` min_samples_leaf | a `continue`, NOT a redraw                   |
| `:671-672` reset/update   | the score pass IS the update                   |
| `:674-677` min_weight_leaf| DEVIATION 154 -- `sample_weight` unported      |
| `:679-689` monotonic_cst  | DEVIATION 154 -- `monotonic_cst` unported      |
| `:691` proxy              | `ProxyImpurityImprovement` (+ exact form, 144) |
| `:693` `>` first-wins     | DEVIATION 133/145/153 -- a total order         |
| `:694-700` missing_go_to_left | DEVIATION 136 -- the field does not exist  |
| `:705-710` re-partition   | DEVIATION 152 -- the caller partitions once    |
| `:712-721` children impurity + improvement | computed for the winner   |
| `:723-731` constant bookkeeping | DEVIATION 132 -- nothing is inherited    |

WHAT IT RETURNS AND WHY IT RETURNS SO MUCH
-------------------------------------------
A device kernel that produces the right `Split` for the wrong reason is the
failure mode this lane cannot afford, and a digest of the final answer cannot
see it. So `HostSplitResult` carries EVERY intermediate a device pass computes
-- per candidate: the range, the constant verdict, the raw and guarded
threshold, `n_left`/`n_right`, the left accumulators, and the score in both
the float and (classification) the exact-rational form -- so a device check
compares cell by cell rather than comparing one number.

TWO ENTRY POINTS, NOT ONE
--------------------------
`node_split_random_gini` and `node_split_random_mse`. sklearn has one function
because the criterion is a Python-level object it calls through; cuML has one
kernel template instantiated per objective (`builder_kernels.cuh`, one
`computeSplitClassificationKernel` / `computeSplitRegressionKernel` pair per
objective). Two concrete functions is cuML's shape, and it is also the shape
that keeps the classification path's exact-integer comparator (DEVIATION 144)
out of the regression path, where it does not exist.
"""

from extratrees.checks.pcg_rng import key_for
from extratrees.impl.decisiontree.batched_levelalgo.dataset import Dataset
from extratrees.impl.decisiontree.decisiontree import (
    CRITERION_ENTROPY,
    CRITERION_GINI,
)
from extratrees.impl.decisiontree.batched_levelalgo.objectives import (
    AggregateBin,
    CountBin,
    EntropyObjectiveFunction,
    GiniObjectiveFunction,
    GiniProxyExact,
    MSEObjectiveFunction,
    impurity_improvement,
)
from extratrees.impl.decisiontree.batched_levelalgo.split import (
    ET_TIE_BREAK_KEYED,
    Split,
    keyed_tie_wins,
    split_tie_salt_for,
)
from extratrees.impl.decisiontree.batched_levelalgo.kernels.builder_kernels import (
    NodeWorkItem,
)
from extratrees.impl.decisiontree.batched_levelalgo.kernels.builder_kernels_impl import (
    FeatureRange,
    draw_threshold,
    draw_threshold_raw,
    node_feature_is_constant,
    node_feature_min_max,
)


# ==========================================================================
# DEVIATION BLOCK 151 -- the supplied column list IS the search; sklearn's
#                        "keep drawing until one is non-constant" has no
#                        analogue and is not emulated
#
# THEIRS (`_splitter.pyx:573-577`), the loop guard, in full:
#
#     while (f_i > n_total_constants and
#            (n_visited_features < max_features or
#             n_visited_features <= n_found_constants + n_drawn_constants)):
#
#   Read carefully, that is TWO separate facts and `max_features` is neither
#   a cap nor a floor on its own:
#     (a) it stops early when every REMAINING feature is already known
#         constant (`f_i > n_total_constants`), so a node can visit fewer
#         than `max_features`;
#     (b) the second disjunct keeps drawing PAST `max_features` for as long
#         as every feature drawn so far was constant, so a node can visit
#         many more than `max_features` -- up to all `n_features` of them.
#   The guarantee it buys is: if ANY non-constant feature exists in the whole
#   column space, sklearn will find one and the node will split.
#
# OURS. This function receives `colids` -- the columns cuML's sampler already
#   chose for this node (DEVIATION 131) -- and evaluates EVERY one of them,
#   exactly once, in the order supplied (which must not matter; see 133/145).
#   There is no re-draw. If all of `colids` is constant over this node, the
#   node has no valid split and becomes a leaf.
#
# THE ANALOGUE, STATED: `len(colids)` plays `max_features`, and (a) is
#   subsumed -- visiting a constant column costs one range pass and no score
#   pass, which is what stopping early was for. Fact (b) has NO analogue and
#   is the part that is really gone.
#
# WHY IT CANNOT BE EMULATED HERE. The extension is a property of a SEQUENTIAL
#   sampler that still holds un-drawn features in a permutation it can keep
#   pulling from. cuML's samplers commit to `n_sampled_cols` ids per node up
#   front, on the device, before any range is known
#   (`builder_kernels.cuh:152`, `:246`, launched at `builder.cuh:427`), and
#   nothing in their pipeline can ask for more once a column turns out
#   constant. Re-invoking the sampler under a different key to get more
#   columns is not in either upstream, so it would be invention, and rule 0b
#   forbids it. DECLINED, with the price below.
#
# WHICH UPSTREAM THIS SIDES WITH. cuML's. Their `computeSplitKernel` scores
#   the sampled `colids` and nothing else; a node whose sampled columns yield
#   no valid split becomes a leaf via `split_not_valid`
#   (`builder_kernels.cuh:59-67`). So this is not a third behaviour invented
#   between two upstreams -- it is one upstream's behaviour where the two
#   disagree, chosen because the sampler is already theirs.
#
# PRICE, PAID IN TREES NOT IN CYCLES. A node all of whose SAMPLED columns are
#   constant becomes a leaf here where sklearn would keep drawing and could
#   still split. It compounds with DEVIATION 132 (we re-discover constants
#   per node instead of inheriting them, so our sampler keeps proposing
#   columns that are constant deep in the tree). On data with many constant
#   or near-constant columns our trees are therefore SHALLOWER than sklearn's
#   at the same `max_features`, and that is a quality difference, not a
#   rounding one. It is the reason the all-constant analytic fixture exists,
#   and the reason `HostSplitResult.n_constant` is reported rather than
#   discarded: a builder can count it and a user can see it.
# ==========================================================================


# ==========================================================================
# DEVIATION BLOCK 152 -- the candidate scan COUNTS; only the winner is
#                        partitioned
#
# THEIRS (`_splitter.pyx:656-659`). Every candidate calls
#   `partitioner.partition_samples(threshold, missing_go_to_left)`, which
#   REORDERS `samples[start:end]` in place by a two-pointer swap
#   (`_partitioner.pyx:217-246`) and returns the split position `pos`;
#   `n_left` is then `pos - start` (`:661`). The criterion accumulates over
#   the freshly partitioned array (`:671-672`). After the loop, if the LAST
#   candidate evaluated is not the winner, they partition once more to put
#   the array back into the winner's arrangement (`:705-710`).
#
# OURS. The scan makes ONE pass per candidate over the node's rows in
#   `row_ids` order, counting and accumulating, and MOVES NOTHING. The
#   winning split is handed to `partition_samples`
#   (`builder_kernels_impl.cuh:43-88`, already ported) by the caller, once.
#
# WHY. Their partition and their count are the same operation because they
#   partition first and read `pos` out of it. cuML's partition is the other
#   way round: `partitionSamples` takes `split.nLeft` as an INPUT
#   (`builder_kernels_impl.cuh:52`, `part = loffset + split.nLeft`), so it
#   cannot be used to discover the count. Something has to count first, and
#   in a parallel builder that something is the score pass, which is walking
#   the rows anyway. Their `:705-710` re-partition then has nothing to undo.
#
# PRICE, TWO PARTS, AND THE SECOND ONE IS REAL.
#   1. Row order inside a child is cuML's, not sklearn's. Already the case
#      and already recorded: DEVIATION 134.
#   2. THE ACCUMULATION ORDER DIFFERS. sklearn's criterion sums the left
#      child's labels in POST-PARTITION order; ours sums them in the node's
#      `row_ids` order, skipping the right-going rows. For classification
#      this changes nothing -- integer counts (DEVIATION 144). For
#      REGRESSION, floating-point addition is not associative, so the two
#      orders can give label sums that differ in the last bits, and the
#      device will use a third order again (a tree reduction). This is
#      exactly the ground DEVIATION 135 is open over, and it is why this
#      file takes its accumulator type as a PARAMETER and the check drives
#      it at `float64`: the oracle must not bake in an answer to an open
#      question.
# ==========================================================================


# ==========================================================================
# DEVIATION BLOCK 153 -- REGRESSION selects on sklearn's MSE proxy, and
#                        reports cuML's `GainPerSplit`
#
# THEIRS, and they are two different quantities, exactly as DEVIATION 144
#   found on the classification side.
#     - sklearn (`_splitter.pyx:691` over `_criterion.pyx:944-973`) selects
#       on `sum_L^2/n_L + sum_R^2/n_R`.
#     - cuML (`objectives.cuh:225-244`) selects on
#       `0.5/n * (sum_L^2/n_L + sum_R^2/n_R - S^2/n)`.
#   The map between them is affine with a positive slope `0.5/n` and an
#   offset `-S^2/n` that is constant within one node, so the ARGMAX is the
#   same. The VALUE is not, and neither is the set of pairs that tie in
#   float.
#
# OURS. The reduction key is sklearn's proxy; `Split.best_metric_val` carries
#   cuML's `GainPerSplit`, for reporting, for `min_impurity_decrease`
#   (`split_not_valid`, `builder_kernels.cuh:59-67`, whose threshold is
#   scaled to THEIR gain) and for `feature_importances_`.
#
# WHY THIS IS A DEVIATION AND NOT AN INSTANCE OF 145. DEVIATION 145 is argued
#   from an EXACT integer comparator existing for Gini; it does not, and
#   cannot while 135 is open, for MSE. The choice here is between two float
#   quantities with the same argmax and different rounding, and it is settled
#   the other way: by which upstream is the SPEC for the split rule. This
#   file transcribes `node_split_random`, so it compares what `:691`
#   compares.
#
# PRICE. On a near-tie the two forms can disagree, and then our chosen
#   feature is sklearn's rather than cuML's. `0.5/n * x - 0.5*S^2/n^2` is one
#   multiply and one subtract away from `x`, so the disagreement needs the
#   two candidates within ~1 ulp of each other after that map; unmeasured,
#   deliberately (no timing and no statistics are taken in this lane). Both
#   numbers are in `CandidateRecord`, so a check can count it whenever
#   someone wants the number.
# ==========================================================================


# ==========================================================================
# DEVIATION BLOCK 154 -- two of sklearn's four rejection branches are absent
#                        because the inputs they test do not exist here
#
# THEIRS. `node_split_random` rejects a candidate on FOUR tests, in this
#   order: `min_samples_leaf` (`:664-666`), `min_weight_leaf` (`:674-677`),
#   monotonicity (`:679-689`), and -- implicitly -- the `>` at `:693`.
#
# OURS. The first and the last are here. The middle two are not:
#     - `min_weight_leaf` compares `criterion.weighted_n_left/right` against
#       a threshold. With `sample_weight=None` -- the only case this port
#       supports, and the case DEVIATION 144's exact integer comparator
#       stands on -- `weighted_n_left == n_left`, and sklearn's own default
#       `min_weight_fraction_leaf=0.0` makes `min_weight_leaf` zero, so the
#       branch is unreachable in the configuration we support. `sample_weight`
#       is in `NOT_IMPLEMENTED.tsv`.
#     - `monotonic_cst` has no field in `DecisionTreeParams` (cuML has no
#       such parameter at all), so there is no way for a caller to ask for it.
#
# WHY NOT REFUSE BY NAME, the way `max_n_bins` is refused (DEVIATION 138)?
#   Because there is no name to refuse: neither parameter appears in this
#   port's parameter struct. A refusal needs a field to refuse.
#
# PRICE, AND IT IS A REAL GAP, NOT A ZERO. A user coming from sklearn who
#   sets `min_weight_fraction_leaf` or `monotonic_cst` gets no error from
#   THIS layer -- they get an error from the Python binding only if that
#   binding refuses the keyword, and no such binding exists yet. Whoever
#   writes it owes both names an explicit refusal, and this block is the
#   record of that debt.
# ==========================================================================


@fieldwise_init
struct CandidateRecord[dtype: DType](Copyable, Movable):
    """Everything one candidate feature produced, for a cell-by-cell check.

    One record per entry of `colids`, in the order supplied. A record exists
    even for a candidate that was skipped as constant or rejected by
    `min_samples_leaf` -- WHY it was skipped is the part a device check most
    needs to compare, and a missing record cannot be compared.
    """

    var colid: Int32
    """The feature, as supplied. Theirs is `current_split.feature`
    (`_splitter.pyx:605`)."""

    var extent: FeatureRange
    """`find_min_max`'s output, `_partitioner.pyx:129-165`. `n_missing` is
    always 0 here -- a nonzero value raises (DEVIATION 136)."""

    var is_constant: Bool
    """`_splitter.pyx:613-618`. When true, nothing below this line ran."""

    var drew_threshold: Bool
    """False exactly when `is_constant` -- the draw is downstream of the
    constant test in their order (`:618` `continue` precedes `:633`)."""

    var raw_threshold: Float32
    """The draw BEFORE sklearn's `:653-654` guard. Reported so a check can
    see the guard fire; `draw_threshold_raw` in `builder_kernels_impl.mojo`
    exists for the same reason."""

    var threshold: Float32
    """The draw after the guard. This is `current_split.threshold`."""

    var guard_fired: Bool
    """`raw_threshold == extent.max_value`, i.e. `:653` was true."""

    var n_left: Int32
    """`current_split.pos - start` (`:661`), reached by counting rather than
    by partitioning; DEVIATION 152."""

    var n_right: Int32
    """`end - current_split.pos` (`:662`)."""

    var rejected_min_samples_leaf: Bool
    """`:664-666`. A `continue`, NOT a redraw."""

    var scored: Bool
    """True when the candidate reached `:691` -- not constant, not rejected.
    Only a scored candidate can win."""

    var hist_left: List[CountBin]
    """Classification: the left child's class counts, `sum_left` in
    `_criterion.pyx`. Empty on the regression path."""

    var agg_left: AggregateBin[Self.dtype]
    """Regression: the left child's `(label_sum, count)`. Zero on the
    classification path."""

    var sq_sum_left: Scalar[Self.dtype]
    """Regression: the left child's sum of SQUARED labels. sklearn computes
    it only for the winner (`_criterion.pyx:997-1006`); it is accumulated for
    every candidate here because the score pass is already reading the label
    and a check wants it per cell. It takes no part in selection."""

    var proxy_float: Scalar[Self.dtype]
    """`criterion.proxy_impurity_improvement()`, `_splitter.pyx:691`. THE
    quantity their `:693` compares."""

    var proxy_exact: GiniProxyExact
    """Classification only: the same proxy as an exact rational (DEVIATION
    144). `valid=False` on the regression path and on rejected candidates."""

    var gain: Scalar[Self.dtype]
    """cuML's `GainPerSplit` (`objectives.cuh:52-83` / `:225-244`), which is
    what lands in `Split.best_metric_val`. NOT the selection key."""


@fieldwise_init
struct HostSplitResult[dtype: DType](Copyable, Movable):
    """The chosen split plus every intermediate that produced it."""

    var split: Split
    """The winner. `Split()` (colid `-1`) when nothing was valid, which is
    what `Split.is_valid` / `split.cuh:145` tests."""

    var found: Bool
    """Whether any candidate was scored and won. `split.is_valid()` agrees;
    both are kept because a device check should compare both."""

    var is_classification: Bool

    var node_rows: Int32
    """`end - start` (`:527-528`)."""

    var candidates: List[CandidateRecord[Self.dtype]]
    """One per supplied colid, in the supplied order."""

    var best_index: Int
    """Index into `candidates` of the winner, or `-1`. NOT a colid."""

    var hist_total: List[CountBin]
    """Classification: the node's class counts, `sum_total`."""

    var agg_total: AggregateBin[Self.dtype]
    """Regression: the node's `(label_sum, count)`."""

    var sq_sum_total: Scalar[Self.dtype]
    """Regression: `sq_sum_total`, `_criterion.pyx:938`."""

    var best_exact: GiniProxyExact
    """The winner's exact proxy (classification). Invalid when `found` is
    False."""

    var n_constant: Int32
    """How many of `colids` were constant over this node. DEVIATION 151's
    price is exactly the case where this equals `len(colids)`."""

    var n_visited: Int32
    """`len(colids)`. Their `n_visited_features` (`:556`, `:578`) counts the
    same thing under DEVIATION 151."""

    var n_tied_best: Int32
    """DEVIATION 463's exact-tie counter: how many SCORED candidates' selection
    key exactly ties the winner's (the winner itself included), so `>= 2`
    means this node's split was decided by the tie-break arm and not by the
    key. The key is the exact rational for classification and the float proxy
    for MSE. `0` when no candidate was scored. The exact-tie RATE a caller
    reports is `count(n_tied_best >= 2) / count(found)` over nodes."""

    var impurity_parent: Scalar[Self.dtype]
    """`criterion.node_impurity()` over the node's totals. Zero when no split
    was found."""

    var impurity_left: Scalar[Self.dtype]
    """`best_split.impurity_left`, `_splitter.pyx:712-716`."""

    var impurity_right: Scalar[Self.dtype]
    """`best_split.impurity_right`, same lines."""

    var improvement: Scalar[Self.dtype]
    """`best_split.improvement`, `_splitter.pyx:717-721`."""

    def candidate_for(self, colid: Int32) -> Int:
        """Index of the record for `colid`, or `-1`.

        A device check compares per (node, feature) and has no reason to know
        the order this host walked the columns in -- that order is precisely
        what must not matter (DEVIATION 130).
        """
        for i in range(len(self.candidates)):
            if self.candidates[i].colid == colid:
                return i
        return -1


def _empty_candidate[
    dtype: DType
](colid: Int32, extent: FeatureRange) -> CandidateRecord[dtype]:
    """A record for a candidate that never reached the draw (`:619-626`)."""
    return CandidateRecord[dtype](
        colid=colid,
        extent=extent,
        is_constant=True,
        drew_threshold=False,
        raw_threshold=Float32(0.0),
        threshold=Float32(0.0),
        guard_fired=False,
        n_left=Int32(0),
        n_right=Int32(0),
        rejected_min_samples_leaf=False,
        scored=False,
        hist_left=List[CountBin](),
        agg_left=AggregateBin[dtype](),
        sq_sum_left=Scalar[dtype](0),
        proxy_float=Scalar[dtype].MIN_FINITE,
        proxy_exact=GiniProxyExact(0, 0, Int64(0), False),
        gain=Scalar[dtype].MIN_FINITE,
    )


def _wins_on_total_order(cand: Split, best: Split, tie_salt: UInt32) -> Bool:
    """The tie order below the selection key, which the caller has already
    resolved (the exact rational for Gini, DEVIATION 145; sklearn's proxy for
    MSE, DEVIATION 153).

    DEVIATION 463: the shipping tie order is the keyed pseudorandom rank of
    `split.mojo::keyed_tie_wins` -- CALLED, not transcribed, so host and
    device cannot drift -- with `tie_salt = split_tie_salt_for(tree_id,
    node_id)` computed once per node by the callers above. sklearn's own tie
    semantics are first-in-a-uniformly-random-visit-order, i.e. uniform among
    the tied; the keyed rank reproduces the uniformity without depending on a
    visit order this parallel formulation does not have. The pre-463 arm --
    cuML `split.cuh:85`/`88`, greater `colid` then greater `quesval`, a
    systematic bias toward high column ids -- is kept below under
    `MOJOLEARN_ET_TIE_MAX_COLID` for the A/B.

    STILL NOT `Split.update`, and for DEVIATION 145's original reason:
    `update`'s first test is the `Float32` metric, and two candidates whose
    EXACT proxies differ can round to the same `Float32`; falling through it
    would pick by rounding noise.
    """
    comptime if ET_TIE_BREAK_KEYED:
        return keyed_tie_wins(tie_salt, cand, best)
    if cand.colid > best.colid:
        return True
    if cand.colid == best.colid:
        return cand.quesval > best.quesval
    return False


def _refuse_missing(colid: Int32, extent: FeatureRange) raises:
    """DEVIATION 136: a NaN is an error, not a coin flip.

    sklearn would take `:630` (`has_missing = n_missing != 0`), then draw a
    second random number at `:649` to send the missing rows left or right,
    then partition with `missing_go_to_left` (`:657-658`). None of that is
    ported. Refusing by name is `gbdt/`'s discipline and rule 3's: an
    unported path must be VISIBLE.
    """
    if extent.n_missing != 0:
        raise Error(
            "host_splitter: feature "
            + String(colid)
            + " has "
            + String(extent.n_missing)
            + " missing (NaN) values in this node. Missing values are"
            " refused, not randomized -- see DEVIATION 136 in"
            " extratrees/DEVIATIONS.md."
        )


def node_split_random_gini[
    dtype: DType
](
    dataset: Dataset,
    work_item: NodeWorkItem,
    colids: List[Int32],
    objective: GiniObjectiveFunction[dtype],
    seed: UInt64,
    tree_id: Int32,
    weighted_n_samples: Scalar[dtype] = Scalar[dtype](0),
    criterion: Int32 = CRITERION_GINI,
) raises -> HostSplitResult[dtype] where dtype.is_floating_point():
    """`_splitter.pyx:507-720` for CLASSIFICATION, on our keyed draws.

    `weighted_n_samples` is the WHOLE TREE's sample weight, which only
    `impurity_improvement` uses (`_criterion.pyx:165-199`) and which this
    function has no way to know. Pass it to get sklearn's tree-scaled
    `improvement`; leave it at 0 and the node's own row count is used, which
    makes `improvement` node-local. Nothing else in the result depends on it,
    and NOTHING in the selection does.

    `criterion` (DEVIATION 459) is `CRITERION_GINI` -- the exact rational
    selects, `objective` scores -- or `CRITERION_ENTROPY`, where an
    `EntropyObjectiveFunction` built from the same `nclasses` /
    `min_samples_leaf` computes cuML's float gain, `GainKeyExact` turns it
    into the `(key, 1)` pair, and the SAME `CompareProxyExact` orders it.
    The name keeps `_gini` because the loop is unchanged and every caller is
    unchanged; the objective is one branch at `:691`.
    """
    if criterion != CRITERION_GINI and criterion != CRITERION_ENTROPY:
        raise Error(
            "node_split_random_gini: criterion must be GINI or ENTROPY; got "
            + String(criterion)
        )
    var entropy_objective = EntropyObjectiveFunction[dtype](
        objective.nclasses, objective.min_samples_leaf
    )
    var begin = Int(work_item.instances.begin)
    var count = Int(work_item.instances.count)
    var nclasses = Int(objective.nclasses)
    var node_id = UInt32(Int(work_item.idx))

    # ------------------------------------------------------------------
    # `criterion.init` (`_criterion.pyx:302-370`): the node's totals, once.
    # sklearn recomputes `sum_total` at every `reset()`; the VALUE is the
    # node's class counts, so it is computed once here and the per-candidate
    # `sum_right` is recovered as `total - left` exactly as cuML recovers it
    # (`objectives.cuh:72-73`).
    # ------------------------------------------------------------------
    var hist_total = List[CountBin](length=nclasses, fill=CountBin(0))
    for p in range(begin, begin + count):
        var row = dataset.row_ids[unsafe_offset=p]
        var lab = dataset.label(row)
        if lab != lab:
            raise Error(
                "host_splitter: NaN label at row "
                + String(row)
                + "; labels are refused, not imputed (DEVIATION 136)."
            )
        var k = Int(lab)
        if Float32(k) != lab or k < 0 or k >= nclasses:
            raise Error(
                "host_splitter: label "
                + String(lab)
                + " at row "
                + String(row)
                + " is not an integer class id in [0, "
                + String(nclasses)
                + ")."
            )
        hist_total[k].x += 1

    # DEVIATION 216's companion, sklearn `_tree.pyx:240`: a PURE node --
    # `impurity <= EPSILON`, which for integer Gini counts is EXACTLY "one
    # class holds every row" -- is a LEAF before any candidate is drawn.
    # Under cuML's old `<=` gate pure nodes leafed through zero-gain
    # rejection, so this test is what keeps 216's accepted zero-gain splits
    # from cascading a pure region down to the depth cap. The returned
    # record is the nothing-was-valid shape (`Split()`, colid -1), which
    # `split_not_valid` rejects on the MIN_FINITE sentinel as always.
    for k in range(nclasses):
        if Int(hist_total[k].x) == count and count > 0:
            return HostSplitResult[dtype](
                split=Split(),
                found=False,
                is_classification=True,
                node_rows=Int32(count),
                candidates=List[CandidateRecord[dtype]](),
                best_index=-1,
                hist_total=hist_total^,
                agg_total=AggregateBin[dtype](),
                sq_sum_total=Scalar[dtype](0),
                best_exact=GiniProxyExact(0, 0, Int64(count), False),
                n_constant=Int32(0),
                n_visited=Int32(0),
                n_tied_best=Int32(0),
                impurity_parent=Scalar[dtype](0),
                impurity_left=Scalar[dtype](0),
                impurity_right=Scalar[dtype](0),
                improvement=Scalar[dtype](0),
            )

    # `:560` `_init_split(&best_split, end)`. Theirs seeds `pos = end` and
    # `improvement = -INFINITY`; cuML's default `Split` seeds `colid = -1`
    # and `best_metric_val = -max<DataT>()` (`split.cuh:54-59`), which is the
    # record this lane reduces over. DEVIATION 133.
    var best = Split()
    var best_exact = GiniProxyExact(0, 0, Int64(count), False)
    var best_index = -1
    var n_constant = 0

    # DEVIATION 463: the tie order's per-node rank key, the SAME value the
    # builder stages for the device reduction (`node_tie_salt`).
    var tie_salt = split_tie_salt_for(UInt32(Int(tree_id)), node_id)

    var records = List[CandidateRecord[dtype]]()

    # `:573-577` -> DEVIATION 151. The loop is over the supplied columns.
    for ci in range(len(colids)):
        var col = colids[ci]  # `:605` `current_split.feature = features[f_j]`

        # `:607-611` `partitioner.find_min_max(...)`.
        var extent = node_feature_min_max(dataset, work_item, col)
        _refuse_missing(col, extent)  # `:630`, `:639-651` -> DEVIATION 136

        # `:613-618` the constant test, all three arms.
        if node_feature_is_constant(extent, Int32(count)):
            # `:619-626`: not a candidate; `continue`.
            n_constant += 1
            records.append(_empty_candidate[dtype](col, extent))
            continue

        # `:628-629` swaps the feature into the drawn region of their
        # permutation. There is no permutation here; nothing to do.

        # `:632-637` the draw, `:653-654` the `== max -> min` guard. Both
        # live in `draw_threshold`, which cites them.
        var key = key_for(seed, UInt32(Int(tree_id)), node_id, UInt32(Int(col)))
        var raw = draw_threshold_raw(key, extent)
        var threshold = draw_threshold(key, extent)

        # `:656-662` -- COUNT, do not partition. DEVIATION 152.
        # `_partitioner.pyx:236-238`: `feature_values[p] <= threshold` goes
        # LEFT, and cuML agrees (`builder_kernels_impl.cuh:65-66`).
        var hist_left = List[CountBin](length=nclasses, fill=CountBin(0))
        var n_left = 0
        for p in range(begin, begin + count):
            var row = dataset.row_ids[unsafe_offset=p]
            var v = dataset.value(row, col)
            if v <= threshold:
                n_left += 1
                hist_left[Int(dataset.label(row))].x += 1
        var n_right = count - n_left

        var hl = hist_left.unsafe_ptr()
        var ht = hist_total.unsafe_ptr()

        # `:664-666` -- a `continue`, NOT a redraw.
        if (
            n_left < Int(objective.min_samples_leaf)
            or n_right < Int(objective.min_samples_leaf)
        ):
            records.append(
                CandidateRecord[dtype](
                    colid=col,
                    extent=extent,
                    is_constant=False,
                    drew_threshold=True,
                    raw_threshold=raw,
                    threshold=threshold,
                    guard_fired=raw == extent.max_value,
                    n_left=Int32(n_left),
                    n_right=Int32(n_right),
                    rejected_min_samples_leaf=True,
                    scored=False,
                    hist_left=hist_left.copy(),
                    agg_left=AggregateBin[dtype](),
                    sq_sum_left=Scalar[dtype](0),
                    proxy_float=Scalar[dtype].MIN_FINITE,
                    proxy_exact=GiniProxyExact(0, 0, Int64(count), False),
                    gain=Scalar[dtype].MIN_FINITE,
                )
            )
            _ = hist_left.unsafe_ptr()
            continue

        # `:674-677` min_weight_leaf and `:679-689` monotonic_cst -> 154.

        # `:691` `current_proxy_improvement = criterion.proxy_impurity_improvement()`
        var proxy_float: Scalar[dtype]
        var proxy_exact: GiniProxyExact
        var gain: Scalar[dtype]
        if criterion == CRITERION_ENTROPY:
            # DEVIATION 459: cuML's entropy gain is the metric AND the key.
            proxy_float = entropy_objective.ProxyImpurityImprovement(
                hl, ht, Int32(count), Int32(n_left)
            )
            gain = entropy_objective.GainPerSplit(
                hl, ht, Int32(count), Int32(n_left)
            )
            proxy_exact = entropy_objective.GainKeyExact(gain, Int32(count))
        else:
            proxy_float = objective.ProxyImpurityImprovement(
                hl, ht, Int32(count), Int32(n_left)
            )
            proxy_exact = objective.ProxyImpurityExact(
                hl, ht, Int32(count), Int32(n_left)
            )
            gain = objective.GainPerSplit(hl, ht, Int32(count), Int32(n_left))

        # `:693` `if current_proxy_improvement > best_proxy_improvement`.
        # Ours: DEVIATION 145 -- the EXACT comparator decides, and only on an
        # exact tie do cuML's remaining two arms (DEVIATION 133) run.
        var cand = Split(
            threshold, col, gain.cast[DType.float32](), Int32(n_left)
        )
        var take = False
        if best_index < 0:
            take = True
        else:
            var order = GiniObjectiveFunction[dtype].CompareProxyExact(
                proxy_exact, best_exact
            )
            if order > 0:
                take = True
            elif order == 0:
                take = _wins_on_total_order(cand, best, tie_salt)
        if take:
            best = cand
            best_exact = proxy_exact
            best_index = len(records)

        records.append(
            CandidateRecord[dtype](
                colid=col,
                extent=extent,
                is_constant=False,
                drew_threshold=True,
                raw_threshold=raw,
                threshold=threshold,
                guard_fired=raw == extent.max_value,
                n_left=Int32(n_left),
                n_right=Int32(n_right),
                rejected_min_samples_leaf=False,
                scored=True,
                hist_left=hist_left.copy(),
                agg_left=AggregateBin[dtype](),
                sq_sum_left=Scalar[dtype](0),
                proxy_float=proxy_float,
                proxy_exact=proxy_exact,
                gain=gain,
            )
        )
        _ = hist_left.unsafe_ptr()

    # ------------------------------------------------------------------
    # `:705-721`. Their `:705-710` re-partition is DEVIATION 152's business;
    # `:712-721` is the winner's impurity and improvement, which is a
    # different quantity from the proxy that chose it.
    # ------------------------------------------------------------------
    var impurity_parent = Scalar[dtype](0)
    var impurity_left = Scalar[dtype](0)
    var impurity_right = Scalar[dtype](0)
    var improvement = Scalar[dtype](0)
    if best_index >= 0:
        var wn = Scalar[dtype](count)
        var wl = Scalar[dtype](Int(best.n_left))
        var wr = wn - wl
        var htp = hist_total.unsafe_ptr()
        if criterion == CRITERION_ENTROPY:
            impurity_parent = entropy_objective.NodeImpurity(htp, wn)
            entropy_objective.ChildrenImpurity(
                records[best_index].hist_left.unsafe_ptr(),
                htp,
                wl,
                wr,
                impurity_left,
                impurity_right,
            )
        else:
            impurity_parent = objective.NodeImpurity(htp, wn)
            objective.ChildrenImpurity(
                records[best_index].hist_left.unsafe_ptr(),
                htp,
                wl,
                wr,
                impurity_left,
                impurity_right,
            )
        var wtotal = weighted_n_samples
        if not (wtotal > 0):
            wtotal = wn
        improvement = impurity_improvement[dtype](
            impurity_parent, impurity_left, impurity_right, wn, wtotal, wl, wr
        )
        _ = hist_total.unsafe_ptr()

    # DEVIATION 463's exact-tie tally: scored candidates whose exact key ties
    # the winner's, winner included. Order-independent (a property of the
    # candidate set, not of the fold), so a device check can recompute it.
    var n_tied_best = 0
    if best_index >= 0:
        for i in range(len(records)):
            if not records[i].scored:
                continue
            if (
                GiniObjectiveFunction[dtype].CompareProxyExact(
                    records[i].proxy_exact, best_exact
                )
                == 0
            ):
                n_tied_best += 1

    return HostSplitResult[dtype](
        split=best,
        found=best_index >= 0,
        is_classification=True,
        node_rows=Int32(count),
        candidates=records^,
        best_index=best_index,
        hist_total=hist_total^,
        agg_total=AggregateBin[dtype](),
        sq_sum_total=Scalar[dtype](0),
        best_exact=best_exact,
        n_constant=Int32(n_constant),
        n_visited=Int32(len(colids)),
        n_tied_best=Int32(n_tied_best),
        impurity_parent=impurity_parent,
        impurity_left=impurity_left,
        impurity_right=impurity_right,
        improvement=improvement,
    )


def node_split_random_mse[
    dtype: DType
](
    dataset: Dataset,
    work_item: NodeWorkItem,
    colids: List[Int32],
    objective: MSEObjectiveFunction[dtype],
    seed: UInt64,
    tree_id: Int32,
    weighted_n_samples: Scalar[dtype] = Scalar[dtype](0),
) raises -> HostSplitResult[dtype]:
    """`_splitter.pyx:507-720` for REGRESSION, on our keyed draws.

    Structurally identical to the Gini form above -- same branches, same
    order, same citations. The two differences are both stated as deviations:
    the accumulators are `(label_sum, count)` plus a sum of squares instead
    of class counts (DEVIATION 143's shape), and the selection key is
    sklearn's MSE proxy rather than cuML's gain (DEVIATION 153), with no
    exact-rational form because there is none while DEVIATION 135 is open.
    """
    var begin = Int(work_item.instances.begin)
    var count = Int(work_item.instances.count)
    var node_id = UInt32(Int(work_item.idx))

    # `criterion.init`: `sum_total`, `sq_sum_total` (`_criterion.pyx:751-800`;
    # `sq_sum_total` is accumulated at `:781` and `:794`).
    var agg_total = AggregateBin[dtype]()
    var sq_sum_total = Scalar[dtype](0)
    for p in range(begin, begin + count):
        var row = dataset.row_ids[unsafe_offset=p]
        var yv = dataset.label(row)
        if yv != yv:
            raise Error(
                "host_splitter: NaN target at row "
                + String(row)
                + "; refused, not imputed (DEVIATION 136)."
            )
        var y = yv.cast[dtype]()
        agg_total.label_sum += y
        agg_total.count += 1
        sq_sum_total += y * y

    var best = Split()
    var best_proxy = Scalar[dtype].MIN_FINITE
    var best_index = -1
    var n_constant = 0

    # DEVIATION 463, exactly as in the Gini form above.
    var tie_salt = split_tie_salt_for(UInt32(Int(tree_id)), node_id)

    var records = List[CandidateRecord[dtype]]()

    for ci in range(len(colids)):
        var col = colids[ci]  # `:605`

        var extent = node_feature_min_max(dataset, work_item, col)  # `:607-611`
        _refuse_missing(col, extent)  # DEVIATION 136

        if node_feature_is_constant(extent, Int32(count)):  # `:613-618`
            n_constant += 1
            records.append(_empty_candidate[dtype](col, extent))
            continue  # `:626`

        var key = key_for(seed, UInt32(Int(tree_id)), node_id, UInt32(Int(col)))
        var raw = draw_threshold_raw(key, extent)  # `:632-637`
        var threshold = draw_threshold(key, extent)  # + `:653-654`

        # `:656-662` -> DEVIATION 152: count and accumulate, move nothing.
        var agg_left = AggregateBin[dtype]()
        var sq_sum_left = Scalar[dtype](0)
        for p in range(begin, begin + count):
            var row = dataset.row_ids[unsafe_offset=p]
            var v = dataset.value(row, col)
            if v <= threshold:
                var y = dataset.label(row).cast[dtype]()
                agg_left.label_sum += y
                agg_left.count += 1
                sq_sum_left += y * y
        var n_left = Int(agg_left.count)
        var n_right = count - n_left

        if (
            n_left < Int(objective.min_samples_leaf)
            or n_right < Int(objective.min_samples_leaf)
        ):  # `:664-666`
            records.append(
                CandidateRecord[dtype](
                    colid=col,
                    extent=extent,
                    is_constant=False,
                    drew_threshold=True,
                    raw_threshold=raw,
                    threshold=threshold,
                    guard_fired=raw == extent.max_value,
                    n_left=Int32(n_left),
                    n_right=Int32(n_right),
                    rejected_min_samples_leaf=True,
                    scored=False,
                    hist_left=List[CountBin](),
                    agg_left=agg_left,
                    sq_sum_left=sq_sum_left,
                    proxy_float=Scalar[dtype].MIN_FINITE,
                    proxy_exact=GiniProxyExact(0, 0, Int64(count), False),
                    gain=Scalar[dtype].MIN_FINITE,
                )
            )
            continue

        # `:674-689` -> DEVIATION 154.

        var proxy_float = objective.ProxyImpurityImprovement(
            agg_left, agg_total, Int32(count), Int32(n_left)
        )  # `:691`
        var gain = objective.GainPerSplit(
            agg_left, agg_total, Int32(count), Int32(n_left)
        )

        # `:693`, with DEVIATION 153 for the first arm and DEVIATION 133 for
        # the other two.
        var cand = Split(
            threshold, col, gain.cast[DType.float32](), Int32(n_left)
        )
        var take = False
        if best_index < 0:
            take = True
        elif proxy_float > best_proxy:
            take = True
        elif proxy_float == best_proxy:
            take = _wins_on_total_order(cand, best, tie_salt)
        if take:
            best = cand
            best_proxy = proxy_float
            best_index = len(records)

        records.append(
            CandidateRecord[dtype](
                colid=col,
                extent=extent,
                is_constant=False,
                drew_threshold=True,
                raw_threshold=raw,
                threshold=threshold,
                guard_fired=raw == extent.max_value,
                n_left=Int32(n_left),
                n_right=Int32(n_right),
                rejected_min_samples_leaf=False,
                scored=True,
                hist_left=List[CountBin](),
                agg_left=agg_left,
                sq_sum_left=sq_sum_left,
                proxy_float=proxy_float,
                proxy_exact=GiniProxyExact(0, 0, Int64(count), False),
                gain=gain,
            )
        )

    # `:712-721`, for the winner only.
    var impurity_parent = Scalar[dtype](0)
    var impurity_left = Scalar[dtype](0)
    var impurity_right = Scalar[dtype](0)
    var improvement = Scalar[dtype](0)
    if best_index >= 0:
        var wn = Scalar[dtype](count)
        var wl = Scalar[dtype](Int(best.n_left))
        var wr = wn - wl
        impurity_parent = objective.NodeImpurity(sq_sum_total, agg_total, wn)
        objective.ChildrenImpurity(
            records[best_index].sq_sum_left,
            sq_sum_total,
            records[best_index].agg_left,
            agg_total,
            wl,
            wr,
            impurity_left,
            impurity_right,
        )
        var wtotal = weighted_n_samples
        if not (wtotal > 0):
            wtotal = wn
        improvement = impurity_improvement[dtype](
            impurity_parent, impurity_left, impurity_right, wn, wtotal, wl, wr
        )

    # DEVIATION 463's exact-tie tally, on the MSE key (the float proxy;
    # DEVIATION 153 says there is no exact rational here).
    var n_tied_best = 0
    if best_index >= 0:
        for i in range(len(records)):
            if not records[i].scored:
                continue
            if records[i].proxy_float == best_proxy:
                n_tied_best += 1

    return HostSplitResult[dtype](
        split=best,
        found=best_index >= 0,
        is_classification=False,
        node_rows=Int32(count),
        candidates=records^,
        best_index=best_index,
        hist_total=List[CountBin](),
        agg_total=agg_total,
        sq_sum_total=sq_sum_total,
        best_exact=GiniProxyExact(0, 0, Int64(count), False),
        n_constant=Int32(n_constant),
        n_visited=Int32(len(colids)),
        n_tied_best=Int32(n_tied_best),
        impurity_parent=impurity_parent,
        impurity_left=impurity_left,
        impurity_right=impurity_right,
        improvement=improvement,
    )
