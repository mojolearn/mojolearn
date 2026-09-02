# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The host control plane: the node queue that turns splits into a tree.

A PORT of cuML `cpp/src/decisiontree/batched-levelalgo/builder.cuh`, pinned at
`00094f7` in `~/CascadeProjects/upstream/cuml`:

| ours | theirs |
|---|---|
| `NodeQueue.__init__` | `builder.cuh:53-65` |
| `NodeQueue.has_work` | `builder.cuh:70` |
| `NodeQueue.pop` | `builder.cuh:72-81` |
| `NodeQueue.is_expandable` | `builder.cuh:83-89` |
| `NodeQueue.push` | `builder.cuh:91-134` |
| `max_nodes` | `builder.cuh:253-262` |

`PORTING_RULES.md` rule 2 is why this is a port and not an afterthought: the
control plane is code too, and cuML's frontier discipline -- batch, split,
push children, repeat -- is the algorithm's shape, not an implementation
detail we get to re-decide.

WHAT IS HERE AND WHAT IS NOT
-----------------------------
CORRECTED 2026-08-31. This section used to say `Builder` (`builder.cuh:140-601`)
was "NOT ported yet ... all of which need kernels this lane has not written",
and that `train()` was held in `train_loop_shape` as a docstring because a loop
calling a function that does not exist is a placeholder. The file falsifies
every clause: the kernels were written and enqueued, and most of the lines
below are the device driver that drives them.

A BARE `:N` BELOW IS A LINE OF THIS FILE; anything of theirs is qualified.

TRANSCRIBED: `NodeQueue` (`:178-402`), `max_nodes` (`:164`), and
`SetLeafPredictions` (`builder.cuh:556-599`) with the `leafKernel` it launches,
as `set_leaf_predictions_classification` (`:405`) and
`set_leaf_predictions_regression` (`:468`).

`Builder`'S DEVICE HALF IS HERE: `DeviceDataset` / `upload_dataset` (`:1189`,
`:1229`), `LevelWorkspace` / `make_level_workspace` (`:1381`, `:1508`),
`sample_features_for_device` (`:1286`) for the sampler launch, and `stage_batch`
/ `search_batch` / `search_batch_regression` (`:1825`, `:1908`, `:3030`) for
`doSplit`'s steps as device launches. The drivers are
`train_classification_device` (`:1259`), `train_forest_classification_device`
(`:2364`) and the regression pair (`:3442`, `:3478`), each with a resident
variant. `train_classification` (`:570`) and `train_regression` (`:723`) stay as
the HOST ORACLES those are checked against, which is not a CPU fallback
(`PORTING_RULES.md` 0b-ii).

ABSENT: their `allReduceHistograms`, `packedHistogramWorkspaceSize` and the
distributed flag, all multi-GPU only, and `MLCommon::TimerCPU` with
`tree->train_time` (`builder.cuh:377`, `:387`), declined as DEVIATION 303.

NOT THEIR SHAPE, which is a deviation and not a gap: DEVIATION 211 makes one
batch SPAN TREES -- a level cycle pops work from every in-flight tree's queue
into one launch, tree id riding per work item -- where cuML overlaps whole trees
with `#pragma omp parallel for num_threads(n_streams)` over CUDA streams
(`randomforest.cuh:336-341`). Metal has no streams, so their mechanism could not
be transcribed.

A SECOND UPSTREAM, ADDED 2026-09-01: sklearn's `BestFirstTreeBuilder`
------------------------------------------------------------------
This file now holds TWO growth modes, and cuML is the upstream of only one of
them. `DecisionTreeParams.max_leaf_nodes` selects sklearn's best-first
builder (`sklearn/tree/_tree.pyx:341-508`), which is a different BUILDER and
not a parameter of cuML's: a priority queue on impurity improvement, a budget
spent one expansion at a time, and the split search moved from pop time to
push time. DEVIATION BLOCKS 466 to 469 below carry it in full -- the mode,
the frontier key, the tie rule, and what the launch shape costs.

| ours | theirs (sklearn) |
|---|---|
| `FrontierRecord` | `_tree.pyx:341-357` |
| `bestfirst_before` | `_compare_records`, `:359-363` (ours is a TOTAL order; theirs is not) |
| `NodeQueue.bestfirst_admit` | `_add_to_frontier`, `:365-372` |
| `NodeQueue.bestfirst_pop` | `pop_heap` + `back` + `pop_back`, `:451-453` |
| `NodeQueue.bestfirst_budget_left` | `max_split_nodes`, `:424` and `:503` |
| `NodeQueue.bestfirst_expand` | the else arm of `build`, `:464-500` |
| `train_classification_bestfirst` / `train_regression_bestfirst` | `build`, `:392-508`, on the host |
| the `bestfirst` arm of `train_forest_*_device_timed` | the same loop, driving the same kernels |
| `frontier_key` | `Criterion.impurity_improvement`, `_criterion.pyx:165-199` |

THE DEFAULT IS UNTOUCHED BY ALL OF IT. `max_leaf_nodes == -1` is cuML's loop,
statement for statement, and every branch the mode added is written
`if bestfirst:` with the depth-wise arm unchanged inside the `else`.

THE ONE INVARIANT EVERYTHING DOWNSTREAM RESTS ON
-------------------------------------------------
Children are allocated as an ADJACENT PAIR and only the left index is stored
(`builder.cuh:112-129` pushes left then right; `flatnode.h:56` returns
`left_child_id + 1` for the right). So `sparsetree` and `node_instances_` grow
in lockstep and stay the same length -- which is what lets
`SetLeafPredictions` zip them (`builder.cuh:562-563` asserts exactly that).
"""

from extratrees.checks.host_splitter import (
    node_split_random_gini,
    node_split_random_mse,
)
from extratrees.impl.decisiontree.decisiontree import (
    CRITERION_END,
    CRITERION_ENTROPY,
    CRITERION_GINI,
    DecisionTreeParams,
    validity_check,
)
from extratrees.impl.decisiontree.flatnode import (
    SparseTreeNode,
    TreeMetaDataNode,
)
from extratrees.impl.decisiontree.batched_levelalgo.dataset import Dataset
from extratrees.impl.decisiontree.batched_levelalgo.objectives import (
    AggregateBin,
    CountBin,
    EntropyObjectiveFunction,
    GiniObjectiveFunction,
    MSEObjectiveFunction,
)
from extratrees.impl.decisiontree.batched_levelalgo.split import Split
from extratrees.impl.decisiontree.batched_levelalgo.kernels.builder_kernels import (
    FeatureSamplerPlan,
    InstanceRange,
    NodeWorkItem,
    SAMPLE_ALGO_L,
    WorkloadInfo,
    device_has_float64,
    plan_feature_sampling,
    sample_features,
    sample_features_device,
    sample_features_pertree,
    sampler_report_len,
    sampler_scratch_len,
    split_not_valid,
)
from extratrees.impl.decisiontree.batched_levelalgo.kernels.builder_kernels_impl import (
    node_feature_range_decode_kernel,
    node_feature_is_constant,
    node_nonconstant_flag_kernel,
    FeatureRange,
    WorkloadPlan,
    node_feature_min_max,
    LEAF_MAX_OUT_DEFAULT,
    LEAF_SAB_NONE,
    PARTITION_UNVISITED,
    PART_SAB_NONE,
    RANGE_SAB_NONE,
    SCORE_STATUS_SCORED,
    PHASE_SETUP_TPB,
    SEARCH_ROWS_PER_THREAD,
    build_workload_info,
    float_gain_key,
    leaf_kernel,
    node_split_kernel,
    node_feature_range_kernel,
    node_feature_score_finalize_kernel,
    node_feature_score_kernel,
    partition_samples,
    phase_setup_a_kernel,
    phase_setup_b_kernel,
)
from std.time import perf_counter_ns
from std.os import getenv

from core.device_liveness import assert_device_alive
from core.identity_trace import IdentityTrace
from extratrees.checks.host_splitter import HostSplitResult
from extratrees.checks.rescue import rescue_key, rescue_pick
from extratrees.impl.decisiontree.flatnode import SparseTreeNode
from extratrees.impl.decisiontree.batched_levelalgo.kernels.partition_multiblock import (
    PART_MB_SAB_NONE,
    partition_count_kernel,
    partition_scan_kernel,
    partition_scatter_kernel,
    partition_writeback_kernel,
)
from extratrees.impl.decisiontree.batched_levelalgo.split import (
    split_reduce_kernel,
    split_tie_count_kernel,
    split_tie_salt_for,
)
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from std.gpu import WARP_SIZE, block_dim, block_idx, grid_dim, thread_idx
from std.math import ceildiv, fma
from std.sys.compile import is_defined
from std.sys.info import size_of

from checks.numerics import ftz, identical_div, identical_mul
from core.philox import launch_uniform_int
from extratrees.checks.pcg_rng import row_sample_seed


def max_nodes(max_depth: Int32) -> Int:
    """`builder.cuh:253-262`, transcribed including their cliff.

    Theirs: a dense tree's node count for depth < 13, and a FIXED 8191 above
    that -- which is `2^13 - 1`, the dense count for depth 12, not a bound on
    anything. It is a starting reservation, not a cap: their `sparsetree` is a
    `std::vector` and grows past it. Ours is a `List` and does the same, so the
    number is a hint here exactly as it is there.
    """
    if max_depth < 13:
        return (1 << Int(max_depth + 1)) - 1
    return 8191


# ==========================================================================
# DEVIATION BLOCK 466 -- BEST-FIRST GROWTH: a SECOND GROWTH MODE, selected by
# sklearn's `max_leaf_nodes`, beside cuML's depth-wise default.
#
# THEIRS (sklearn, `_tree.pyx:374-508`, `BestFirstTreeBuilder`). Passing
#   `max_leaf_nodes` does not tighten a parameter, it selects a DIFFERENT
#   BUILDER. The frontier is a `vector[FrontierRecord]` kept as a heap
#   (`push_heap` / `pop_heap` around `_compare_records`, `:359-363`), the
#   budget is spent ONE POP AT A TIME (`max_split_nodes` at `:424`,
#   decremented at `:503`), and a node is SEARCHED WHEN IT IS ADDED to the
#   frontier (`_add_split_node`, `:562`, called at `:439`, `:509`, `:531`)
#   because its own gain is what orders it.
# THEIRS (cuML, `builder.cuh:72-134`). A FIFO deque popped `max_batch_size`
#   at a time, searched in one launch, pushed. Every node at a level is
#   expanded; nothing is ranked against anything.
# OURS. BOTH, selected by one field. `params.max_leaf_nodes == -1` is
#   cuML's loop, byte for byte the code that was here before this block was
#   written; anything else runs the best-first loop below. There is no third
#   behaviour and no blending: `max_leaf_nodes` NEVER maps onto cuML's
#   `max_leaves`, which stays a separate field with a separate meaning
#   (a cap on the breadth-first frontier that reorders nothing). Both may be
#   set; the tighter binds.
# WHY NOT REFUSE, which is what this lane did until 2026-09-01. The refusal
#   argued that the two upstreams grow different trees and that accepting
#   sklearn's name would be accepting cuML's algorithm under it. The first
#   half is true and the second half does not follow: the answer to two
#   upstreams disagreeing is to implement the one whose NAME the caller
#   typed, which is what this block does.
#
# THE SHAPE, so the next reader does not have to re-derive it from the loop.
# A depth-wise cycle is: pop a FIFO batch, search it, partition it, push it.
# A best-first cycle is the SAME THREE DEVICE STEPS IN THE SAME ORDER over
# two different memberships:
#
#     1. SEARCH  the nodes admitted since the last cycle -- the children the
#                last cycle's expansions created, or the roots on cycle 0.
#     2. ADMIT   each searched node whose split is VALID onto its own tree's
#                priority queue, keyed by DEVIATION 467's improvement.
#     3. POP     the single best node of each in-flight tree, if that tree's
#                leaf budget is not spent.
#     4. PARTITION those popped nodes -- their splits were computed in an
#                earlier cycle and are still correct, because a node's rows
#                are permuted only by its OWN partition or an ANCESTOR's,
#                and neither has happened while it sat on the frontier.
#     5. EXPAND  each popped node into a split node plus two leaf children,
#                and hand the expandable children to the next cycle's step 1.
#
# So the launches are unchanged, the workspace is unchanged, the kernels are
# unchanged, and DEVIATION 211's cross-tree batching survives intact. What
# changes is who is in each batch, and that is DEVIATION 469's cost.
#
# WHAT IS NOT PORTED FROM THEIR BUILDER, stated rather than left to be
# discovered. (a) Their frontier also carries records for nodes that are
# ALREADY leaves (`is_leaf = 1`, `improvement = 0.0`, `_tree.pyx:641-647`),
# popped later to be finalised. Ours does not, and the trees are the same:
# in this representation a node is CREATED as a leaf (`CreateLeafNode`) and
# only becomes a split node when it is expanded, so a leaf record on the
# frontier would pop, do nothing, and consume no budget -- their `is_leaf`
# arm at `:456-462` writes exactly the state ours never leaves. (b) Their
# `lower_bound` / `upper_bound` / `middle_value` members exist only for
# `monotonic_cst`, which is refused by name here (NOT_IMPLEMENTED.tsv).
# (c) Their `_add_split_node` computes the node VALUE at add time; this lane
# computes every leaf value in one launch at the end (`SetLeafPredictions`,
# DEVIATION 214) and that is unchanged.
#
# NOT LIFTED FROM THE LOSSGUIDE LANE, checked rather than assumed:
#   `gbdt/methods/greedy_subsets_searcher/greedy_search_helper_lossguide.mojo`
#   argmins over `TPointsSubsets` leaf scores on a HISTOGRAM tree, a
#   representation this histogram-free directory deliberately does not have.
#   It is a different learner's frontier, not a component.
# ==========================================================================


# ==========================================================================
# DEVIATION BLOCK 467 -- THE FRONTIER KEY is sklearn's `improvement`,
# reconstructed from cuML's gain rather than recomputed from impurities.
#
# THEIRS. `_compare_records` (`_tree.pyx:359-363`) orders on
#   `FrontierRecord.improvement`, which `_add_split_node` takes from
#   `SplitRecord.improvement`, which is
#   `Criterion.impurity_improvement` (`_criterion.pyx:165-199`):
#
#       (w_node / w_total) * (imp_parent - wR/w_node*imp_R - wL/w_node*imp_L)
#
#   The bracket is the node's own impurity DECREASE; the `w_node / w_total`
#   factor in front is what makes a big mediocre node outrank a small
#   excellent one, and it is therefore load bearing for the ORDER, not
#   cosmetic scaling.
# OURS. `Split.best_metric_val` is cuML's `GainPerSplit`, and this lane
#   already records (`objectives.mojo:61-72`) that cuML's gain IS that
#   bracket: `cuML_gain == parent_gini + sklearn_proxy / n`. So the whole
#   improvement is one multiply away from a number the reduction already
#   hands back, and the key is
#
#       key = ftz( identical_mul( ftz(count / total), ftz(gain) ) )
#
#   with `count` the node's row count and `total` the tree's sampled row
#   count -- `weighted_n_samples` with `sample_weight=None`, which is the
#   only case this port supports (NOT_IMPLEMENTED.tsv's `sample_weight`
#   row).
# WHY NOT RECOMPUTE THEIR EXPRESSION TERM FOR TERM: it needs
#   `imp_parent`, `imp_left` and `imp_right` as three separate Float32s at
#   the host, and the device reduction publishes none of them -- it
#   publishes the winning candidate's gain and, for Gini, DEVIATION 145's
#   exact rational. Adding three impurity fields to the readback to
#   re-derive a quantity already in it would be three more seams to pin for
#   no change in the order.
# WHY NOT ORDER ON THE EXACT RATIONAL, which is what DEVIATION 144/145 do
#   INSIDE a node. Their key is `num/den` with `-n` dropped, and `-n` is
#   constant only WITHIN one node; across nodes it is not, and restoring it
#   plus the `count/total` factor puts denominators at `n_total * n^2`,
#   which is past what the Int128 cross-multiply survives at the row counts
#   DEVIATION 218 admits. The exact rational stays what it always was, the
#   WITHIN-NODE selector; the ACROSS-NODE order is this float. The
#   consequence is stated and gated rather than hidden: two nodes whose true
#   improvements differ can round to the same `key`, and DEVIATION 468 is
#   what decides those.
# THE PIN. Every operation is a pinned primitive -- `identical_div`,
#   `identical_mul`, `ftz` -- so the key is bit-identical on Apple, NVIDIA
#   and AMD under IDENTICAL, and so is `>` on it. It is computed on the
#   HOST, and that is exactly why the pins are not optional: the host is a
#   different CPU on each vendor's box, and a plain `a / b * c` there is as
#   free to differ as a kernel is.
# NO NaN CAN REACH IT: `GainPerSplit` clamps its result at zero
#   (`objectives.mojo:555-556`, DEVIATION 217), `count` and `total` are
#   positive integers, so `key >= +0.0` and the comparison is a total order
#   on the floats that occur.
# PRICE: `Float32(Int(count))` rounds above 2^24 rows. It rounds the same
#   way on every vendor (IEEE round-to-nearest on an integer conversion is
#   not implementation defined), so it costs ORDER RESOLUTION at huge row
#   counts, never identity.
# ==========================================================================


# ==========================================================================
# DEVIATION BLOCK 468 -- THE TIE RULE. A float key needs one, sklearn does
# not have one, and we cannot inherit the one it does not have.
#
# THEIRS. `_compare_records` is `left.improvement < right.improvement` and
#   nothing else, handed to `std::push_heap` / `std::pop_heap`. Equal
#   improvements are therefore separated by the HEAP'S LAYOUT -- by the
#   insertion history and the sift order of a particular libstdc++ -- which
#   is not a rule, is not documented, and is not reproducible across a
#   standard-library change. It is the same shape of non-order DEVIATION 133
#   already recorded for their `>` on candidate splits.
# OURS. A STRICT TOTAL ORDER on the record's own fields, so the heap's
#   layout cannot be observed:
#
#       pop first the record with the GREATER `key`;
#       on an equal `key`, the SMALLER `tree_id`;
#       on an equal `tree_id`, the SMALLER `idx` (node id).
#
#   `(tree_id, idx)` is UNIQUE across the whole frontier -- one queue per
#   tree, and a node id is allocated once -- so the relation never ties, and
#   any correct heap yields one and only one pop sequence.
# WHY SMALLER `idx` AND NOT LARGER. Node ids are allocated in expansion
#   order, so "smaller id" is "admitted earlier": among equals the frontier
#   degrades to FIFO, which is the depth-wise builder's own rule. That makes
#   the two modes AGREE on a fixture where every improvement is equal, which
#   is a checkable statement and is checked, and it avoids DEVIATION 463's
#   scar -- "greater colid wins" was a systematic bias toward high column
#   ids that was paid in accuracy on covtype, and "greater node id wins"
#   would be the same mistake one level up, a systematic bias toward the
#   RIGHT and DEEPER side of the tree.
# WHY `tree_id` IS IN THE KEY AT ALL when each tree has its own frontier and
#   the arm can never fire: because DEVIATION 211's batch spans trees, the
#   ORDER OF THE MERGED BATCH is `(tree_id, idx)`, and writing the tie rule
#   over the same pair means the batch order and the frontier order are one
#   statement rather than two that can drift.
# PRICE: on an exact `key` tie we expand a different node than sklearn
#   would. That is unavoidable -- sklearn expands whichever one its heap
#   happens to surface -- and it is the price of having a rule at all.
# GATED, and the gate is shown capable of failing:
#   `BESTFIRST_SAB_TIE_MAX_IDX` flips the third arm and the pop order must
#   move; `BESTFIRST_SAB_FIFO` drops the key entirely and the tree must move.
# ==========================================================================


# ==========================================================================
# DEVIATION BLOCK 469 -- WHAT SEARCHING AT PUSH TIME COSTS. Stated in
# launches, because that is the currency this lane spends.
#
# DEPTH-WISE. One cycle covers a whole LEVEL of every in-flight tree. A
#   tree of depth d costs d cycles; the search batch is up to
#   `max_batch_size` nodes wide and, with DEVIATION 211, spans every tree in
#   the group.
# BEST-FIRST. One cycle expands AT MOST ONE NODE PER TREE, because the
#   choice of the second node depends on the first node's children being on
#   the frontier. A tree of L leaves costs L - 1 cycles. Per cycle the
#   PARTITION batch is at most `g` nodes (one per in-flight tree) and the
#   SEARCH batch is at most `2g` (the two children of each). `g` is the
#   group's tree count, so the batch width collapses from
#   `min(frontier, max_batch_size)` to `2g`.
# SO THE PRICE IS: (leaves - 1) / depth times as many cycles, each of them
#   `2g` nodes wide instead of up to `max_batch_size`. On a 100-tree forest
#   grown to 32 leaves that is 31 cycles of at most 200 nodes against about
#   6 cycles of up to 4096. DEVIATION 211 is what keeps this from being one
#   launch per node -- without the cross-tree batch, `g` would be 1 and a
#   best-first cycle would be a two-node launch. It is the reason this mode
#   is affordable here at all, and it is why `g` is CAPPED rather than
#   reduced: see the `max_batch_size / 2` clamp in the drivers.
# NO TIMING NUMBER IS ATTACHED TO ANY OF THIS AND NONE WILL BE UNTIL A
#   BENCH ARM MEASURES IT. The counts above are launch counts, which are
#   arithmetic; the seconds are not, and this lane has been wrong before
#   about which of the two it had.
# ONE MORE COST, in synchronizations rather than launches: the search batch
#   and the partition batch are DIFFERENT SETS in this mode, so `ws.h_items`
#   is re-staged between them on every cycle and DEVIATION 455's drain runs
#   every cycle rather than only on a rescue. Depth-wise pays it only when
#   DEVIATION 205's rescue fires.
# ==========================================================================


comptime BESTFIRST_SAB_NONE: Int32 = 0
"""No sabotage. The shipping value."""

comptime BESTFIRST_SAB_FIFO: Int32 = 1
"""Order the frontier by arrival instead of by improvement -- what a port
that kept cuML's deque and only added the leaf budget would build. The mode
becomes cuML's `max_leaves` under sklearn's name, so the TREE must move on
any fixture where the best node is not the oldest. DEVIATION 466's gate."""

comptime BESTFIRST_SAB_TIE_MAX_IDX: Int32 = 2
"""Invert the third arm of the tie rule: on an equal key, the GREATER node
id wins. DEVIATION 468's gate -- the pop order must move on a frontier
carrying two equal keys, and it must NOT move on one that carries none."""

comptime BESTFIRST_SAB_UNSCALED_KEY: Int32 = 3
"""Key on the raw gain, dropping DEVIATION 467's `count / total` factor --
which is what "just use best_metric_val" would build. The order must move
whenever two frontier nodes of different sizes are ranked against each
other. DEVIATION 467's gate."""

comptime BESTFIRST_SAB_NO_BUDGET: Int32 = 4
"""Ignore the leaf budget at pop time. The LEAF COUNT must move: this is the
arm that proves `max_leaf_nodes` is spent one pop at a time rather than
merely accepted."""


def frontier_key(gain: Float32, count: Int32, total: Int32) -> Float32:
    """sklearn's `improvement` for a split, as DEVIATION 467 derives it.

    `(count / total) * gain`, every operation a pinned primitive so the
    number and every comparison on it are bit-identical on Apple, NVIDIA and
    AMD under IDENTICAL. Computed on the HOST, where the pins matter for the
    same reason they matter in a kernel: the host CPU is a different CPU on
    each vendor's box.

    A degenerate node (`count <= 0`, or an empty tree) keys at `+0.0`; it
    cannot reach the frontier anyway, because a split over no rows is never
    valid.
    """
    if count <= 0 or total <= 0:
        return Float32(0.0)
    var q = identical_div(
        ftz(Float32(Int(count))), ftz(Float32(Int(total)))
    )
    return ftz(identical_mul(ftz(q), ftz(gain)))


@fieldwise_init
struct FrontierRecord(ImplicitlyCopyable, Movable):
    """One searched, splittable node waiting to be expanded.

    `FrontierRecord` (`_tree.pyx:341-357`), minus the members this port does
    not have a use for -- see DEVIATION BLOCK 466's "what is not ported".
    Theirs carries `start`/`end`/`pos` where ours carries the
    `NodeWorkItem`'s `InstanceRange` and the `Split`'s `n_left`, which are
    the same two numbers under different names.
    """

    var item: NodeWorkItem
    """The node, exactly as the search batch carried it."""

    var split: Split
    """Its split, found when it was ADMITTED. Still correct when it is
    popped: a node's rows are permuted only by its own partition or an
    ancestor's, and neither happens while it waits here."""

    var key: Float32
    """DEVIATION 467's improvement. The heap's first ordering arm."""

    var tree_id: Int32
    """The owning tree. DEVIATION 468's second arm, and DEVIATION 211's
    per-item tree id, which are deliberately the same field."""


def bestfirst_before(
    a: FrontierRecord, b: FrontierRecord, sabotage: Int32
) -> Bool:
    """Whether `a` is popped before `b`. DEVIATION 468's total order.

    Greater `key`; then smaller `tree_id`; then smaller `idx`. The last two
    are unique across the frontier, so this never returns False both ways
    for distinct records and the heap's layout is unobservable.
    """
    var ka = a.key
    var kb = b.key
    if sabotage == BESTFIRST_SAB_FIFO:
        # No key at all: arrival order, which for node ids allocated in
        # expansion order is exactly `idx` ascending.
        ka = Float32(0.0)
        kb = Float32(0.0)
    elif sabotage == BESTFIRST_SAB_UNSCALED_KEY:
        ka = a.split.best_metric_val
        kb = b.split.best_metric_val
    if ka > kb:
        return True
    if ka < kb:
        return False
    if a.tree_id != b.tree_id:
        return a.tree_id < b.tree_id
    if sabotage == BESTFIRST_SAB_TIE_MAX_IDX:
        return a.item.idx > b.item.idx
    return a.item.idx < b.item.idx


struct NodeQueue[dtype: DType](Movable):
    """Manages the iterative batched-level building of nodes on the host.

    `builder.cuh:44-135`. Their `std::deque<NodeWorkItem> work_items_` is a
    `List` plus a head cursor here: `pop_front` on a `List` is a shift, and
    their deque's only two operations are push-back and pop-front.
    """

    var params: DecisionTreeParams
    var tree: TreeMetaDataNode[Self.dtype]
    var node_instances: List[InstanceRange]
    """`std::vector<InstanceRange> node_instances_` (`builder.cuh:49`). One
    entry per node, SAME LENGTH as `tree.sparsetree`, always."""

    var work_items: List[NodeWorkItem]
    var head: Int
    """Index of the front of `work_items`; everything below it is popped."""

    var frontier: List[FrontierRecord]
    """DEVIATION 466's BEST-FIRST frontier, a binary max-heap under
    `bestfirst_before`. EMPTY AND UNTOUCHED unless `params.max_leaf_nodes`
    selects the mode, which is what makes the default fit bit-unchanged:
    `pop`, `push` and `is_expandable` never read it."""

    var total_rows: Int32
    """The tree's sampled row count -- sklearn's `weighted_n_samples` with
    `sample_weight=None`. DEVIATION 467's denominator. Recorded here because
    the frontier key needs it and nothing else in the queue did."""

    var bf_sabotage: Int32
    """`BESTFIRST_SAB_*`. Rule 8: the switch that selects a behaviour is a
    field a check can set, not a comment. `BESTFIRST_SAB_NONE` ships."""

    def __init__(
        out self,
        params: DecisionTreeParams,
        sampled_rows: Int32,
        num_outputs: Int32,
        treeid: Int32 = 0,
        row_base: Int32 = 0,
    ):
        """`builder.cuh:53-65`.

        The root is created as a LEAF holding every sampled row, and is pushed
        as work only if it is expandable -- so `max_depth == 0` yields a
        one-node tree with no work at all, which is their behaviour and is a
        case the check covers.

        `row_base` is DEVIATION 211's slot offset: the root's `InstanceRange`
        starts there instead of 0, because the batched forest trainer keeps
        every in-flight tree's rows in ONE device buffer and tree slot `s`
        owns `[s * n_rows, (s + 1) * n_rows)`. Every child range is carved
        out of its parent's, so one offset here makes every range this tree
        ever holds land in its own slot. The default keeps every single-tree
        caller exactly as it was.
        """
        self.params = params
        self.tree = TreeMetaDataNode[Self.dtype](
            treeid=treeid,
            depth_counter=0,
            leaf_counter=1,
            num_outputs=num_outputs,
            vector_leaf=List[Scalar[Self.dtype]](),
            sparsetree=List[SparseTreeNode[Self.dtype]](),
        )
        self.tree.sparsetree.append(
            SparseTreeNode[Self.dtype].CreateLeafNode(sampled_rows)
        )
        self.node_instances = List[InstanceRange]()
        self.node_instances.append(InstanceRange(row_base, sampled_rows))
        self.work_items = List[NodeWorkItem]()
        self.head = 0
        self.frontier = List[FrontierRecord]()
        self.total_rows = sampled_rows
        self.bf_sabotage = BESTFIRST_SAB_NONE
        if self.is_expandable(self.tree.sparsetree[0], 0):
            self.work_items.append(
                NodeWorkItem(0, 0, self.node_instances[0])
            )

    def get_tree(self) -> TreeMetaDataNode[Self.dtype]:
        """`builder.cuh:67`, `GetTree()`. Theirs hands back the `shared_ptr`
        it has been mutating; ours returns a copy, because Mojo will not let a
        field be moved out of a struct that is still alive and a reference
        would tie the tree's lifetime to the queue's."""
        return self.tree.copy()

    def has_work(self) -> Bool:
        """`builder.cuh:70`, `work_items_.size() > 0`."""
        return self.head < len(self.work_items)

    def pop(mut self) -> List[NodeWorkItem]:
        """`builder.cuh:72-81`: take up to `max_batch_size` from the front.

        Theirs reserves `min(max_batch_size, size)` and then drains in a
        `while`. The batch width is a SCHEDULING parameter: it decides how many
        nodes one kernel launch covers and must not change the tree. That
        property is checkable and is checked.
        """
        return self.pop_up_to(Int(self.params.max_batch_size))

    def pop_up_to(mut self, limit: Int) -> List[NodeWorkItem]:
        """`pop` with a caller-supplied bound BELOW `max_batch_size`.

        DEVIATION 211's forest trainer fills one merged batch from many
        queues, so each queue may only take what is left of the batch. The
        bound is still capped by `max_batch_size` -- the workspace is sized
        to it -- and a limit of that size IS `pop`, which delegates here.
        """
        var cap = Int(self.params.max_batch_size)
        if limit < cap:
            cap = limit
        var result = List[NodeWorkItem]()
        while self.head < len(self.work_items) and len(result) < cap:
            result.append(self.work_items[self.head])
            self.head += 1
        return result^

    def is_expandable(
        self, node: SparseTreeNode[Self.dtype], depth: Int32
    ) -> Bool:
        """`builder.cuh:83-89`, transcribed test for test, in their order.

        Note what is NOT here: no impurity test and no `min_samples_leaf`.
        Those live in `split_not_valid` and are applied to the SPLIT after it
        is found (`builder.cuh:99-103`), not to the node before. A node can be
        expandable and still end up a leaf.
        """
        if depth >= self.params.max_depth:
            return False
        if node.InstanceCount() < self.params.min_samples_split:
            return False
        if (
            self.params.max_leaves != -1
            and self.tree.leaf_counter >= self.params.max_leaves
        ):
            return False
        return True

    def push(
        mut self, work_items: List[NodeWorkItem], splits: List[Split]
    ) raises:
        """`builder.cuh:93-140`: turn a batch of splits into nodes and work.

        Transcribed in their order. ONE HALF of that order is load-bearing and
        the other half is not, and the difference was MEASURED rather than
        argued: the `max_leaves` test at `:106` sits AFTER the validity
        `continue` at `:101-104`, so an invalid split does not consume leaf
        budget -- sabotaging that turns `builder_check` red. The test and the
        `break` are ONE STATEMENT on `:106`; this paragraph said `:105` and
        `:106` as though they were two, and `:105` is blank. That `break` is
        EQUIVALENT to a `continue` here, because `leaf_counter` only
        ever increases inside this loop, so once the budget test is true it
        stays true for every remaining item in the batch. Replacing the break
        with a continue leaves the check green, and that is not a hole in the
        check: the two are the same function. Their `break` is a shortcut, not
        a semantic. Kept as theirs anyway (transcribe, do not tidy), and
        recorded here so nobody re-derives it.
        """
        if len(work_items) != len(splits):
            raise Error(
                "push: "
                + String(len(work_items))
                + " work items but "
                + String(len(splits))
                + " splits"
            )

        for i in range(len(work_items)):
            var split = splits[i]
            var item = work_items[i]
            var parent_range = self.node_instances[Int(item.idx)]

            # `:101-104`
            if split_not_valid(
                split,
                self.params.min_impurity_decrease,
                self.params.min_samples_leaf,
                parent_range.count,
            ):
                continue

            # `:106` -- a BREAK, not a continue.
            if (
                self.params.max_leaves != -1
                and self.tree.leaf_counter >= self.params.max_leaves
            ):
                break

            # `:108-115` -- the parent becomes a split node pointing at the
            # left child, which is the next slot about to be appended.
            var left_child_id = Int64(len(self.tree.sparsetree))
            self.tree.sparsetree[Int(item.idx)] = SparseTreeNode[
                Self.dtype
            ].CreateSplitNode(
                split.colid,
                Scalar[Self.dtype](split.quesval),
                Scalar[Self.dtype](split.best_metric_val),
                left_child_id,
                parent_range.count,
            )
            self.tree.leaf_counter += 1

            # `:116-124` -- left child.
            self.tree.sparsetree.append(
                SparseTreeNode[Self.dtype].CreateLeafNode(split.n_left)
            )
            self.node_instances.append(
                InstanceRange(parent_range.begin, split.n_left)
            )
            if self.is_expandable(
                self.tree.sparsetree[len(self.tree.sparsetree) - 1],
                item.depth + 1,
            ):
                self.work_items.append(
                    NodeWorkItem(
                        Int32(len(self.tree.sparsetree) - 1),
                        item.depth + 1,
                        self.node_instances[len(self.node_instances) - 1],
                    )
                )

            # `:126-133` -- right child.
            var n_right = parent_range.count - split.n_left
            self.tree.sparsetree.append(
                SparseTreeNode[Self.dtype].CreateLeafNode(n_right)
            )
            self.node_instances.append(
                InstanceRange(parent_range.begin + split.n_left, n_right)
            )
            if self.is_expandable(
                self.tree.sparsetree[len(self.tree.sparsetree) - 1],
                item.depth + 1,
            ):
                self.work_items.append(
                    NodeWorkItem(
                        Int32(len(self.tree.sparsetree) - 1),
                        item.depth + 1,
                        self.node_instances[len(self.node_instances) - 1],
                    )
                )

            # `:135-136`
            if item.depth + 1 > self.tree.depth_counter:
                self.tree.depth_counter = item.depth + 1

    # ======================================================================
    # DEVIATION 466's BEST-FIRST HALF. Everything below runs only when
    # `params.max_leaf_nodes != -1`; `pop`, `push` and `is_expandable` above
    # are untouched and are still the whole default fit.
    # ======================================================================

    def bestfirst_enabled(self) -> Bool:
        """Whether this queue grows best-first. The ONE test, so a caller
        cannot ask the question two slightly different ways."""
        return self.params.max_leaf_nodes != -1

    def bestfirst_budget_left(self) -> Bool:
        """`max_split_nodes > 0` (`_tree.pyx:424`, `:454`), in this port's
        counter.

        `tree.leaf_counter` IS the leaf count here -- it starts at 1 for the
        root and rises by exactly one per expansion (`push`, `:114` theirs)
        -- and sklearn's `max_split_nodes = max_leaf_nodes - 1` counts the
        same thing from the other end. So their `max_split_nodes <= 0` is
        our `leaf_counter >= max_leaf_nodes`, and stopping there yields
        EXACTLY `max_leaf_nodes` leaves whenever the frontier does not run
        dry first.
        """
        if self.bf_sabotage == BESTFIRST_SAB_NO_BUDGET:
            return True
        return self.tree.leaf_counter < self.params.max_leaf_nodes

    def bestfirst_can_pop(self) -> Bool:
        """Whether this tree contributes a node to the next expansion."""
        return len(self.frontier) > 0 and self.bestfirst_budget_left()

    def bestfirst_seed(mut self) -> List[NodeWorkItem]:
        """The root, as the first cycle's search batch.

        `__init__` already put the root on `work_items` if it is expandable,
        which is the depth-wise frontier; best-first takes it from there and
        leaves that list drained, so no node can be reached twice through the
        two frontiers.
        """
        var out = List[NodeWorkItem]()
        while self.head < len(self.work_items):
            out.append(self.work_items[self.head])
            self.head += 1
        return out^

    def _heap_swap(mut self, a: Int, b: Int):
        """One swap, through two LOCAL COPIES rather than two live element
        references. `FrontierRecord` is a handful of scalars, so the copies
        are free, and taking one mutable reference into a `List` at a time
        is the rule this file follows everywhere else."""
        var ra = self.frontier[a]
        var rb = self.frontier[b]
        self.frontier[a] = rb
        self.frontier[b] = ra

    def _heap_up(mut self, start: Int):
        """Sift `start` toward the root. `std::push_heap`'s half."""
        var i = start
        while i > 0:
            var parent = (i - 1) // 2
            var ri = self.frontier[i]
            var rp = self.frontier[parent]
            if not bestfirst_before(ri, rp, self.bf_sabotage):
                return
            self._heap_swap(i, parent)
            i = parent

    def _heap_down(mut self, start: Int):
        """Sift `start` toward the leaves. `std::pop_heap`'s half."""
        var i = start
        var n = len(self.frontier)
        while True:
            var l = 2 * i + 1
            var r = l + 1
            var best = i
            if l < n:
                var rl = self.frontier[l]
                var rb = self.frontier[best]
                if bestfirst_before(rl, rb, self.bf_sabotage):
                    best = l
            if r < n:
                var rr = self.frontier[r]
                var rb2 = self.frontier[best]
                if bestfirst_before(rr, rb2, self.bf_sabotage):
                    best = r
            if best == i:
                return
            self._heap_swap(i, best)
            i = best

    def bestfirst_admit(
        mut self, item: NodeWorkItem, split: Split, tree_id: Int32
    ) -> Bool:
        """`_add_to_frontier` (`_tree.pyx:365-372`), after the search.

        Returns whether the node was admitted. An INVALID split is not
        admitted, and that is where their `is_leaf` records go: in this
        representation the node is already a leaf and staying off the heap
        leaves it one (DEVIATION BLOCK 466, "what is not ported", (a)). The
        validity test is `split_not_valid`, the same call `push` makes and
        the same call `nodeSplitKernel` makes, so a node cannot be admitted
        under one rule and expanded under another.
        """
        if split_not_valid(
            split,
            self.params.min_impurity_decrease,
            self.params.min_samples_leaf,
            item.instances.count,
        ):
            return False
        self.frontier.append(
            FrontierRecord(
                item,
                split,
                frontier_key(
                    split.best_metric_val,
                    item.instances.count,
                    self.total_rows,
                ),
                tree_id,
            )
        )
        self._heap_up(len(self.frontier) - 1)
        return True

    def bestfirst_pop(mut self) raises -> FrontierRecord:
        """`pop_heap` + `frontier.back()` + `pop_back()` (`_tree.pyx:451-453`).

        The caller must have asked `bestfirst_can_pop` first; popping an
        empty frontier or a spent budget is a programming error here rather
        than a silently empty batch.
        """
        if len(self.frontier) == 0:
            raise Error("bestfirst_pop on an empty frontier")
        if not self.bestfirst_budget_left():
            raise Error(
                "bestfirst_pop with the leaf budget spent: "
                + String(self.tree.leaf_counter)
                + " leaves against max_leaf_nodes "
                + String(self.params.max_leaf_nodes)
            )
        var best = self.frontier[0]
        var last = len(self.frontier) - 1
        var moved = self.frontier[last]
        self.frontier[0] = moved
        _ = self.frontier.pop()
        if len(self.frontier) > 0:
            self._heap_down(0)
        return best

    def bestfirst_expand(
        mut self, item: NodeWorkItem, split: Split
    ) raises -> List[NodeWorkItem]:
        """ONE popped node into a split node plus two leaves.

        `push`'s body for a single item, with its two `continue`/`break`
        guards removed rather than duplicated: the validity test already ran
        at `bestfirst_admit`, and cuML's `max_leaves` break is replaced by
        `bestfirst_budget_left` at POP time, which is sklearn's placement
        (`_tree.pyx:454`) and not cuML's. Everything else -- the adjacent
        child pair, the left-index-only invariant, the `node_instances`
        lockstep, the depth counter -- is `push`'s code and this file's ONE
        INVARIANT paragraph applies to it unchanged.

        Returns the children that need searching. A child that is not
        expandable is left a leaf and never searched, exactly as in `push`.
        A child of a tree whose budget just went to zero is also not
        returned: that tree will never pop again, so searching it could not
        change the tree. That is an unobservable saving, not a rule, and it
        is written here rather than in the driver so both drivers get it.
        """
        var parent_range = self.node_instances[Int(item.idx)]
        var out = List[NodeWorkItem]()

        var left_child_id = Int64(len(self.tree.sparsetree))
        self.tree.sparsetree[Int(item.idx)] = SparseTreeNode[
            Self.dtype
        ].CreateSplitNode(
            split.colid,
            Scalar[Self.dtype](split.quesval),
            Scalar[Self.dtype](split.best_metric_val),
            left_child_id,
            parent_range.count,
        )
        self.tree.leaf_counter += 1

        self.tree.sparsetree.append(
            SparseTreeNode[Self.dtype].CreateLeafNode(split.n_left)
        )
        self.node_instances.append(
            InstanceRange(parent_range.begin, split.n_left)
        )
        var left_ok = self.is_expandable(
            self.tree.sparsetree[len(self.tree.sparsetree) - 1],
            item.depth + 1,
        )
        var left_item = NodeWorkItem(
            Int32(len(self.tree.sparsetree) - 1),
            item.depth + 1,
            self.node_instances[len(self.node_instances) - 1],
        )

        var n_right = parent_range.count - split.n_left
        self.tree.sparsetree.append(
            SparseTreeNode[Self.dtype].CreateLeafNode(n_right)
        )
        self.node_instances.append(
            InstanceRange(parent_range.begin + split.n_left, n_right)
        )
        var right_ok = self.is_expandable(
            self.tree.sparsetree[len(self.tree.sparsetree) - 1],
            item.depth + 1,
        )
        var right_item = NodeWorkItem(
            Int32(len(self.tree.sparsetree) - 1),
            item.depth + 1,
            self.node_instances[len(self.node_instances) - 1],
        )

        if item.depth + 1 > self.tree.depth_counter:
            self.tree.depth_counter = item.depth + 1

        if self.bestfirst_budget_left():
            if left_ok:
                out.append(left_item)
            if right_ok:
                out.append(right_item)
        return out^



def set_leaf_predictions_classification(
    dataset: Dataset,
    mut tree: TreeMetaDataNode[DType.float32],
    node_instances: List[InstanceRange],
) raises:
    """`builder.cuh:556-599` (`SetLeafPredictions`) plus the `leafKernel` it
    launches (`kernels/builder_kernels_impl.cuh:391-417`), for the
    classification objective.

    Their structure, which is the part that matters:

    * `vector_leaf` is sized `sparsetree.size() * num_outputs` and ZEROED
      (`builder.cuh:558`, `:582`), so every node gets a slot whether or not it
      is a leaf;
    * `leafKernel` runs ONE BLOCK PER NODE and returns immediately for a node
      that is not a leaf (`:403`), so an internal node's slot keeps the zeros;
    * the leaf's rows are read through `dataset.row_ids` over the node's own
      `InstanceRange` (`:409-412`), NOT over a contiguous row range -- this is
      why `SetLeafPredictions` asserts `sparsetree.size() ==
      instance_ranges.size()` (`builder.cuh:562-563`) and why the partition
      must have left `row_ids` in the state the ranges describe;
    * the per-class tally is a `CountBin` histogram of width `num_outputs`
      (their `IncrementHistogram(histogram, 1, 0, label)` -- note `n_bins = 1`
      and `bin = 0`, so it is a plain per-class counter, not a histogram over
      thresholds), and `SetLeafVector` turns it into probabilities
      (`objectives.cuh:97-107`).

    This is the HOST form; the device kernel lands beside `partition_samples`
    in `kernels/builder_kernels_impl.mojo` and is checked against this one.
    """
    var n_nodes = tree.num_nodes()
    if len(node_instances) != n_nodes:
        raise Error(
            "SetLeafPredictions: "
            + String(n_nodes)
            + " nodes but "
            + String(len(node_instances))
            + " instance ranges -- builder.cuh:562-563 asserts these are equal"
        )
    var k = Int(tree.num_outputs)

    # `builder.cuh:558` sizes it, `:582` zeroes it. Both, in that order.
    tree.vector_leaf = List[Float32](length=n_nodes * k, fill=Float32(0.0))

    var counts = List[CountBin](length=k, fill=CountBin())
    for node_id in range(n_nodes):
        # `builder_kernels_impl.cuh:403`, the early return.
        if not tree.sparsetree[node_id].IsLeaf():
            continue
        for c in range(k):
            counts[c] = CountBin()
        var rng = node_instances[node_id]
        for i in range(Int(rng.begin), Int(rng.begin) + Int(rng.count)):
            var row = Int(dataset.row_ids[unsafe_offset=i])
            var label = Int(dataset.labels[unsafe_offset=row])
            counts[label].x += 1
        GiniObjectiveFunction[DType.float32].SetLeafVector(
            Pointer(to=counts[0]),
            Int32(k),
            Pointer(to=tree.vector_leaf[node_id * k]),
        )


def set_leaf_predictions_regression(
    dataset: Dataset,
    mut tree: TreeMetaDataNode[DType.float32],
    node_instances: List[InstanceRange],
) raises:
    """The same pass for the MSE objective: the leaf value is the mean of its
    rows' labels (`objectives.cuh:259-264`).

    The accumulator is `AggregateBin[DType.float64]` here BECAUSE THIS IS THE
    HOST ORACLE and DEVIATION 135 -- what the device accumulates in -- is
    open. The device form must not silently inherit this choice.
    """
    var n_nodes = tree.num_nodes()
    if len(node_instances) != n_nodes:
        raise Error(
            "SetLeafPredictions: "
            + String(n_nodes)
            + " nodes but "
            + String(len(node_instances))
            + " instance ranges -- builder.cuh:562-563 asserts these are equal"
        )
    var k = Int(tree.num_outputs)
    if k != 1:
        raise Error(
            "regression leaves are one value per node; num_outputs is "
            + String(k)
        )

    tree.vector_leaf = List[Float32](length=n_nodes * k, fill=Float32(0.0))

    var acc = List[AggregateBin[DType.float64]](
        length=k, fill=AggregateBin[DType.float64]()
    )
    for node_id in range(n_nodes):
        if not tree.sparsetree[node_id].IsLeaf():
            continue
        for c in range(k):
            acc[c] = AggregateBin[DType.float64]()
        var rng = node_instances[node_id]
        for i in range(Int(rng.begin), Int(rng.begin) + Int(rng.count)):
            var row = Int(dataset.row_ids[unsafe_offset=i])
            acc[0].label_sum += Float64(dataset.labels[unsafe_offset=row])
            acc[0].count += 1
        var out = List[Float64](length=k, fill=Float64(0.0))
        MSEObjectiveFunction[DType.float64].SetLeafVector(
            Pointer(to=acc[0]), Int32(k), Pointer(to=out[0])
        )
        for c in range(k):
            tree.vector_leaf[node_id * k + c] = Float32(out[c])


def n_sampled_cols_for(params: DecisionTreeParams, n_cols: Int32) -> Int32:
    """`builder.cuh:222`: `max(1, IdxT(params.max_features * n_cols))`.

    Truncation, not rounding, and a floor of one. Transcribed rather than
    tidied: at `max_features = 0.3` on 3 columns theirs gives 1 candidate, not
    the 0.9 that rounds to 1 by coincidence.
    """
    var k = Int32(params.max_features * Float32(n_cols))
    return 1 if k < 1 else k


def _all_constant[
    dtype: DType
](result: HostSplitResult[dtype]) -> Bool:
    """Whether EVERY column this node sampled was constant on its rows.

    Not "no valid split was found": a non-constant column rejected by
    `min_samples_leaf` still counts as EVALUATED to sklearn
    (`_splitter.pyx:665-666` is a `continue`, and `n_visited - n_constant` has
    already been incremented), and once one non-constant feature is evaluated
    their loop stops at the budget. Keying the rescue on "no split" instead of
    "no non-constant column" would draw again in a case sklearn does not.
    """
    if len(result.candidates) == 0:
        return False
    for c in range(len(result.candidates)):
        if not result.candidates[c].is_constant:
            return False
    return True


def rescue_columns(
    dataset: Dataset, work_item: NodeWorkItem
) raises -> List[Int32]:
    """This node's non-constant columns, in ASCENDING column order.

    The order is part of DEVIATION 205's contract: `rescue_pick` returns an
    INDEX into this list, and the device kernel builds the same list in the
    same order, so the two paths land on the same column. Every column is
    tested, including the ones already sampled -- they were all constant, so
    they cannot appear here, and excluding them explicitly would be a second
    way of saying the same thing.
    """
    var out = List[Int32]()
    for col in range(Int(dataset.n)):
        var extent = node_feature_min_max(dataset, work_item, Int32(col))
        if not node_feature_is_constant(extent, work_item.instances.count):
            out.append(Int32(col))
    return out^


def train_classification(
    dataset: Dataset,
    params: DecisionTreeParams,
    tree_id: Int32,
    seed: UInt64,
    n_classes: Int32,
    rescue: Bool = True,
) raises -> TreeMetaDataNode[DType.float32]:
    """One ExtraTree, end to end. `Builder::train`, `builder.cuh:344-359`.

    Their loop, which this transcribes exactly::

        NodeQueue queue(params, maxNodes(), n_sampled_rows, num_outputs);
        while (queue.HasWork()) {
          auto work_items = queue.Pop();
          auto [splits_host_ptr, splits_count] = doSplit(work_items);
          queue.Push(work_items, splits_host_ptr);
        }
        auto tree = queue.GetTree();
        this->SetLeafPredictions(tree, queue.GetInstanceRanges());

    `doSplit` (`builder.cuh:379-475`) is inlined below in its HOST form: theirs
    samples features, launches `computeSplitKernel` per column block, then
    launches `nodeSplitKernel` which partitions, and this function does the same
    three steps in the same order with the host forms. They are the ORACLES the
    device kernels are checked against, not a gap -- the device form of the same
    three steps is `search_batch` (`:1908`), under `train_classification_device`
    (`:1259`).

    **WHAT IS LOAD-BEARING ABOUT THE ORDER, MEASURED RATHER THAN ASSUMED.**
    This docstring used to say that partitioning before `Push` was essential
    because `Push` computes the children's ranges from `split.n_left`
    (`builder.cuh:117-131`). A sabotage moving the partition to AFTER
    `queue.push` left `tree_check` green, so that claim was false and is
    deleted rather than annotated (rule 10). `Push` records only `(begin,
    count)`; the partition mutates `row_ids` and touches no range, so within
    one batch iteration the two commute.

    The real constraint is one step weaker and one step later: **every node in
    a batch must be partitioned before the NEXT `pop`**, because that is when
    its children become work items and start reading `row_ids` over the ranges
    `Push` recorded. Deferring the partitions past the loop is what breaks it,
    and `tree_check`'s pure-leaf assertions are what see it — the tree stays
    structurally perfect, every count conserves, and every piece-wise check
    stays green.

    Both orders inside the batch are therefore correct; theirs is kept
    (partition inside `doSplit`, `nodeSplitKernel`,
    `builder_kernels_impl.cuh:89-107`) because it is theirs.

    **AND THE VALIDITY GUARD AROUND THE PARTITION IS NOT OBSERVABLE EITHER**,
    which is also measured: partitioning an INVALID split was sabotaged in and
    the check stayed green. A partition only permutes rows inside the node's
    own range, and an invalid split leaves the node a leaf whose value depends
    on the SET of rows in that range and not their order. The guard is kept
    because `nodeSplitKernel` has it (`:100-104`), not because anything here
    can tell the difference.

    `dataset.row_ids` IS MUTATED. Theirs mutates it too — it is the array the
    whole frontier partitions in place.
    """
    if dataset.num_outputs != n_classes:
        raise Error(
            "dataset.num_outputs is "
            + String(dataset.num_outputs)
            + " but n_classes is "
            + String(n_classes)
        )
    validity_check(params)

    # DEVIATION 466: sklearn's `max_leaf_nodes` selects the OTHER BUILDER.
    # The dispatch is the FIRST thing after validation and the LAST thing
    # this function knows about the mode: everything below is cuML's loop,
    # unchanged, and a caller who did not ask for best-first gets the same
    # bits they got before this line existed.
    if params.max_leaf_nodes != -1:
        return train_classification_bestfirst(
            dataset, params, tree_id, seed, n_classes, rescue
        )

    var objective = GiniObjectiveFunction[DType.float32](
        n_classes, params.min_samples_leaf
    )
    # `decisiontree.cuh:253`: CRITERION_END resolves to GINI for
    # classification. ENTROPY rides through (DEVIATION 459).
    var criterion = params.split_criterion
    if criterion == CRITERION_END:
        criterion = CRITERION_GINI
    var k = n_sampled_cols_for(params, dataset.n)
    var queue = NodeQueue[DType.float32](
        params, dataset.n_sampled_rows, n_classes, tree_id
    )

    while queue.has_work():
        var work_items = queue.pop()

        # `builder.cuh:398-471` -- feature sampling, one row of `colids` per
        # work item. The plan is returned so a caller can say which sampler
        # ran, which rule 8 requires of a switch that selects a kernel.
        var colids = List[Int32](
            length=len(work_items) * Int(k), fill=Int32(0)
        )
        _ = sample_features(
            colids, work_items, tree_id, seed, Int(dataset.n), Int(k)
        )

        var splits = List[Split]()
        for i in range(len(work_items)):
            var item = work_items[i]
            var my_colids = List[Int32]()
            for c in range(Int(k)):
                my_colids.append(colids[i * Int(k) + c])

            # `computeSplitKernel`'s replacement, DEVIATION 137. The
            # criterion rides as `params.split_criterion`: Gini selects on
            # DEVIATION 144's exact rational, Entropy on cuML's float gain
            # through the same comparator (DEVIATION 459).
            var result = node_split_random_gini[DType.float32](
                dataset, item, my_colids, objective, seed, tree_id,
                criterion=criterion,
            )

            # DEVIATION 205, which closes 151. `_splitter.pyx:573-577` keeps
            # drawing past `max_features` for exactly as long as EVERY draw
            # has been constant, so a node whose whole sample was constant is
            # not a leaf to sklearn -- it evaluates one more, the first
            # non-constant feature in the remaining random order. That is
            # uniform over the node's non-constant columns, and `rescue_pick`
            # draws it. The rule lives in ONE place because the device path
            # must land on the same column.
            if rescue and _all_constant[DType.float32](result) and item.instances.count > 0:
                var nonconst = rescue_columns(dataset, item)
                if len(nonconst) > 0:
                    var u = rescue_pick(
                        rescue_key(seed, tree_id, UInt32(Int(item.idx))),
                        len(nonconst),
                    )
                    var one = List[Int32]()
                    one.append(nonconst[u])
                    result = node_split_random_gini[DType.float32](
                        dataset, item, one, objective, seed, tree_id,
                        criterion=criterion,
                    )
            splits.append(result.split)

            # `nodeSplitKernel`, `builder_kernels_impl.cuh:89-107`: check
            # validity, then partition. Theirs returns early on an invalid
            # split and leaves `row_ids` alone; so does this.
            if not split_not_valid(
                result.split,
                params.min_impurity_decrease,
                params.min_samples_leaf,
                item.instances.count,
            ):
                partition_samples(dataset, result.split, item)

        queue.push(work_items, splits)

    var tree = queue.get_tree()
    set_leaf_predictions_classification(dataset, tree, queue.node_instances)
    return tree^


def train_regression(
    dataset: Dataset,
    params: DecisionTreeParams,
    tree_id: Int32,
    seed: UInt64,
    rescue: Bool = True,
) raises -> TreeMetaDataNode[DType.float32]:
    """The same loop for MSE. See `train_classification` for the structure and
    for why the partition precedes the push."""
    if dataset.num_outputs != 1:
        raise Error(
            "regression wants one output; dataset.num_outputs is "
            + String(dataset.num_outputs)
        )
    validity_check(params)

    # DEVIATION 466, the regression half of the same dispatch.
    if params.max_leaf_nodes != -1:
        return train_regression_bestfirst(
            dataset, params, tree_id, seed, rescue
        )

    var objective = MSEObjectiveFunction[DType.float64](
        params.min_samples_leaf
    )
    var k = n_sampled_cols_for(params, dataset.n)
    var queue = NodeQueue[DType.float32](
        params, dataset.n_sampled_rows, 1, tree_id
    )

    while queue.has_work():
        var work_items = queue.pop()
        var colids = List[Int32](
            length=len(work_items) * Int(k), fill=Int32(0)
        )
        _ = sample_features(
            colids, work_items, tree_id, seed, Int(dataset.n), Int(k)
        )

        var splits = List[Split]()
        for i in range(len(work_items)):
            var item = work_items[i]
            var my_colids = List[Int32]()
            for c in range(Int(k)):
                my_colids.append(colids[i * Int(k) + c])

            var result = node_split_random_mse[DType.float64](
                dataset, item, my_colids, objective, seed, tree_id
            )

            # DEVIATION 205, the regression half. The clause
            # (`_splitter.pyx:573-577`) is in `node_split_random`, which
            # sklearn shares between both criteria -- it is not a
            # classification rule -- so a regression tree stops early for the
            # same reason and is fixed the same way. Same `rescue_columns`,
            # same `rescue_pick`, same key.
            if (
                rescue
                and _all_constant[DType.float64](result)
                and item.instances.count > 0
            ):
                var nonconst = rescue_columns(dataset, item)
                if len(nonconst) > 0:
                    var u = rescue_pick(
                        rescue_key(seed, tree_id, UInt32(Int(item.idx))),
                        len(nonconst),
                    )
                    var one = List[Int32]()
                    one.append(nonconst[u])
                    result = node_split_random_mse[DType.float64](
                        dataset, item, one, objective, seed, tree_id
                    )
            splits.append(result.split)

            if not split_not_valid(
                result.split,
                params.min_impurity_decrease,
                params.min_samples_leaf,
                item.instances.count,
            ):
                partition_samples(dataset, result.split, item)

        queue.push(work_items, splits)

    var tree = queue.get_tree()
    set_leaf_predictions_regression(dataset, tree, queue.node_instances)
    return tree^


# ==========================================================================
# DEVIATION 466's HOST ARM. The oracles the device best-first driver is
# checked against, in the same relation `train_classification` bears to
# `train_classification_device` (`PORTING_RULES.md` 0b-ii: an oracle, not a
# CPU fallback).
# ==========================================================================


def host_split_one_classification(
    dataset: Dataset,
    item: NodeWorkItem,
    params: DecisionTreeParams,
    objective: GiniObjectiveFunction[DType.float32],
    criterion: Int32,
    seed: UInt64,
    tree_id: Int32,
    k: Int32,
    rescue: Bool,
) raises -> Split:
    """One node's split search, DEVIATION 205's rescue included.

    Lifted verbatim out of `train_classification`'s inner loop so the two
    growth modes run THE SAME SEARCH rather than two copies of it. The
    sampler is keyed per work item (`sample_features_pertree`), so a
    one-item batch draws exactly the columns the same item would draw as
    member `i` of a wide batch; that is the property that lets best-first
    call this one node at a time without moving a single bit.
    """
    var colids = List[Int32](length=Int(k), fill=Int32(0))
    var one_item = List[NodeWorkItem]()
    one_item.append(item)
    _ = sample_features(
        colids, one_item, tree_id, seed, Int(dataset.n), Int(k)
    )
    var result = node_split_random_gini[DType.float32](
        dataset, item, colids, objective, seed, tree_id, criterion=criterion,
    )
    if rescue and _all_constant[DType.float32](result) and item.instances.count > 0:
        var nonconst = rescue_columns(dataset, item)
        if len(nonconst) > 0:
            var u = rescue_pick(
                rescue_key(seed, tree_id, UInt32(Int(item.idx))),
                len(nonconst),
            )
            var one = List[Int32]()
            one.append(nonconst[u])
            result = node_split_random_gini[DType.float32](
                dataset, item, one, objective, seed, tree_id,
                criterion=criterion,
            )
    return result.split


def host_split_one_regression(
    dataset: Dataset,
    item: NodeWorkItem,
    params: DecisionTreeParams,
    objective: MSEObjectiveFunction[DType.float64],
    seed: UInt64,
    tree_id: Int32,
    k: Int32,
    rescue: Bool,
) raises -> Split:
    """`host_split_one_classification`'s MSE twin, same argument."""
    var colids = List[Int32](length=Int(k), fill=Int32(0))
    var one_item = List[NodeWorkItem]()
    one_item.append(item)
    _ = sample_features(
        colids, one_item, tree_id, seed, Int(dataset.n), Int(k)
    )
    var result = node_split_random_mse[DType.float64](
        dataset, item, colids, objective, seed, tree_id
    )
    if (
        rescue
        and _all_constant[DType.float64](result)
        and item.instances.count > 0
    ):
        var nonconst = rescue_columns(dataset, item)
        if len(nonconst) > 0:
            var u = rescue_pick(
                rescue_key(seed, tree_id, UInt32(Int(item.idx))),
                len(nonconst),
            )
            var one = List[Int32]()
            one.append(nonconst[u])
            result = node_split_random_mse[DType.float64](
                dataset, item, one, objective, seed, tree_id
            )
    return result.split


def train_classification_bestfirst(
    dataset: Dataset,
    params: DecisionTreeParams,
    tree_id: Int32,
    seed: UInt64,
    n_classes: Int32,
    rescue: Bool = True,
    bf_sabotage: Int32 = BESTFIRST_SAB_NONE,
) raises -> TreeMetaDataNode[DType.float32]:
    """One ExtraTree grown BEST-FIRST. `BestFirstTreeBuilder::build`,
    `_tree.pyx:392-508`.

    Their loop, which this transcribes with the two structural changes
    DEVIATION BLOCK 466 states::

        rc = self._add_split_node(... root ...)
        if rc >= 0: _add_to_frontier(split_node_left, frontier)
        while not frontier.empty():
            pop_heap(...); record = frontier.back(); frontier.pop_back()
            is_leaf = (record.is_leaf or max_split_nodes <= 0)
            if is_leaf: <write the leaf>
            else:
                max_split_nodes -= 1
                self._add_split_node(... left ...)
                self._add_split_node(... right ...)
                _add_to_frontier(split_node_left, frontier)
                _add_to_frontier(split_node_right, frontier)

    THE ORDER INSIDE ONE EXPANSION IS LOAD BEARING AND IS NOT THEIRS'
    ORDER, because this port partitions where they index. Theirs never
    permutes anything: `_add_split_node` reads `samples[start:end]` and the
    parent's `node_split` already wrote `split.pos`. Ours must partition the
    parent's row range BEFORE either child's search reads it, so the
    expansion is EXPAND (which only computes ranges from `split.n_left`),
    then PARTITION, then SEARCH THE CHILDREN. Moving the partition after the
    children's search would search two children over an unpartitioned range,
    which is a silent wrong answer rather than a crash.

    `dataset.row_ids` IS MUTATED, as it is in `train_classification`.
    """
    if dataset.num_outputs != n_classes:
        raise Error(
            "dataset.num_outputs is "
            + String(dataset.num_outputs)
            + " but n_classes is "
            + String(n_classes)
        )
    validity_check(params)

    var objective = GiniObjectiveFunction[DType.float32](
        n_classes, params.min_samples_leaf
    )
    var criterion = params.split_criterion
    if criterion == CRITERION_END:
        criterion = CRITERION_GINI
    var k = n_sampled_cols_for(params, dataset.n)
    var queue = NodeQueue[DType.float32](
        params, dataset.n_sampled_rows, n_classes, tree_id
    )
    queue.bf_sabotage = bf_sabotage

    # `_tree.pyx:437-441`: the root is searched and admitted before the loop.
    var roots = queue.bestfirst_seed()
    for i in range(len(roots)):
        _ = queue.bestfirst_admit(
            roots[i],
            host_split_one_classification(
                dataset, roots[i], params, objective, criterion, seed,
                tree_id, k, rescue,
            ),
            tree_id,
        )

    while queue.bestfirst_can_pop():
        var rec = queue.bestfirst_pop()
        var kids = queue.bestfirst_expand(rec.item, rec.split)
        # `nodeSplitKernel`'s host form. The split was validated at admit,
        # so the guard `train_classification` carries here is already
        # discharged and is not repeated -- one rule, one place.
        partition_samples(dataset, rec.split, rec.item)
        for j in range(len(kids)):
            _ = queue.bestfirst_admit(
                kids[j],
                host_split_one_classification(
                    dataset, kids[j], params, objective, criterion, seed,
                    tree_id, k, rescue,
                ),
                tree_id,
            )

    var tree = queue.get_tree()
    set_leaf_predictions_classification(dataset, tree, queue.node_instances)
    return tree^


def train_regression_bestfirst(
    dataset: Dataset,
    params: DecisionTreeParams,
    tree_id: Int32,
    seed: UInt64,
    rescue: Bool = True,
    bf_sabotage: Int32 = BESTFIRST_SAB_NONE,
) raises -> TreeMetaDataNode[DType.float32]:
    """`train_classification_bestfirst` for MSE. Same loop, same order, same
    reason the partition sits between the expansion and the children."""
    if dataset.num_outputs != 1:
        raise Error(
            "regression wants one output; dataset.num_outputs is "
            + String(dataset.num_outputs)
        )
    validity_check(params)

    var objective = MSEObjectiveFunction[DType.float64](
        params.min_samples_leaf
    )
    var k = n_sampled_cols_for(params, dataset.n)
    var queue = NodeQueue[DType.float32](
        params, dataset.n_sampled_rows, 1, tree_id
    )
    queue.bf_sabotage = bf_sabotage

    var roots = queue.bestfirst_seed()
    for i in range(len(roots)):
        _ = queue.bestfirst_admit(
            roots[i],
            host_split_one_regression(
                dataset, roots[i], params, objective, seed, tree_id, k,
                rescue,
            ),
            tree_id,
        )

    while queue.bestfirst_can_pop():
        var rec = queue.bestfirst_pop()
        var kids = queue.bestfirst_expand(rec.item, rec.split)
        partition_samples(dataset, rec.split, rec.item)
        for j in range(len(kids)):
            _ = queue.bestfirst_admit(
                kids[j],
                host_split_one_regression(
                    dataset, kids[j], params, objective, seed, tree_id, k,
                    rescue,
                ),
                tree_id,
            )

    var tree = queue.get_tree()
    set_leaf_predictions_regression(dataset, tree, queue.node_instances)
    return tree^


# ==========================================================================
# THE DEVICE PATH
# ==========================================================================
# `Builder::doSplit` (`builder.cuh:379-494`) with its kernels enqueued, and
# `Builder::train`'s loop (`:344-359`) around it. Until this function existed
# every device kernel in this lane was UNWIRED -- built, checked per cell, and
# reached by nothing but its own check, which rule 3 says is not done.
#
# THEIR HOST/DEVICE SPLIT IS PORTED, NOT RE-DECIDED. cuML's node queue is a
# HOST structure: `doSplit` ends with `raft::update_host(h_splits, splits,
# work_items.size())` and a `sync_stream` (`:492-494`), and `Push` then runs on
# the host. So copying the batch's chosen splits back per level is theirs, not
# a shortcut here. What is on the device is what is on theirs: the range pass,
# the draw, the score pass and the split reduction.


def gain_per_split(
    acc_left: MutPointer[Int32, MutAnyOrigin],
    acc_total: MutPointer[Int32, MutAnyOrigin],
    base: Int,
    nclasses: Int,
    len_in: Int32,
    n_left_in: Int32,
    min_samples_leaf: Int32,
) -> Float32:
    """`GiniObjectiveFunction::GainPerSplit`, `objectives.cuh:52-83`, on device.

    Transcribed statement for statement, including the order the three terms
    accumulate into `gain` and including their `invLen`/`invLeft`/`invRight`
    reciprocals rather than divisions -- float division and
    multiply-by-reciprocal are different roundings, and this quantity feeds
    `split_not_valid`.

    The one shape difference is deviation 143's, already recorded: theirs
    indexes a prefix-summed histogram as `hist[n_bins*j + i]` for the left and
    `hist[n_bins*j + n_bins-1]` for the total, ours takes the left and total
    accumulators directly, because there is no bin dimension.

    ==================================================================
    DEVIATION BLOCK 183, SECOND FORM -- the gain is computed ON THE
    DEVICE, which is where cuML computes it.

    THE FIRST FIX WAS ON THE HOST AND IT WAS A RULE-2 VIOLATION.
    `PORTING_RULES.md` 2: "If they do something on the GPU in the control
    plane, we do it on the GPU. If they keep a decision on the device so
    the host never learns it, we keep it on the device." cuML computes
    `GainPerSplit` inside `computeSplitKernel` and the host never sees a
    per-candidate gain. The first closure of 183 copied `status`,
    `n_total` and `acc_total` back per level -- `n_cells * (2 +
    n_classes)` ints that their design never moves -- and formed the gain
    in `Float64` on the host. It was correct and it was the wrong shape,
    and the commit that introduced it argued "the cheaper fix moved less
    code", which optimises for the porter rather than for the port.

    THIS FORM: the gain is computed here, in `Float32`, from their
    expression, and travels with the candidate into the reduction. The
    three readbacks are gone; the only thing that crosses per level is
    the batch's chosen splits, which is what `builder.cuh:492-494` copies
    back anyway.

    A SIDE EFFECT WORTH NAMING: `best_metric_val` is now the same
    quantity computed the same way on both paths, so it stops being a
    field `device_tree_check` has to exclude for a reason it cannot
    check.
    ==================================================================
    """
    var length = Int(len_in)
    var n_left = Int(n_left_in)
    var n_right = length - n_left
    # `:61-63`, and note it is checked BEFORE the reciprocals are used.
    if n_left_in < min_samples_leaf or Int32(n_right) < min_samples_leaf:
        return Float32.MIN_FINITE
    var one = Float32(1.0)
    var inv_len = one / Float32(length)
    var inv_left = one / Float32(n_left)
    var inv_right = one / Float32(n_right)
    var gain = Float32(0.0)
    for j in range(nclasses):
        var lval_i = acc_left[unsafe_offset = base + j]
        var lval = Float32(Int(lval_i))
        gain = fma(lval * inv_left * lval, inv_len, gain)
        var total_sum = acc_total[unsafe_offset = base + j]
        var rval_i = total_sum - lval_i
        var rval = Float32(Int(rval_i))
        gain = fma(rval * inv_right * rval, inv_len, gain)
        var val = Float32(Int(lval_i + rval_i)) * inv_len
        gain = fma(-val, val, gain)
    # DEVIATION 453 (IDENTITY_PATHS row 10): the accumulated gain is the
    # one value in this function that cancellation can land in the
    # denormal band (every operand and every intermediate term is bounded
    # below by 1/n^2, normal at any legal n). Metal's arithmetic flushes
    # it to a signed zero; CUDA's default keeps it and the clamp below
    # then reads a different sign. Flushing the FINAL value under
    # IDENTICAL aligns the vendors to the Metal model; a denormal
    # difference at an INTERMEDIATE step is below half an ulp of every
    # later normal term and cannot survive into the result. Under FAST
    # `ftz` is a comptime no-op.
    gain = ftz(gain)
    # DEVIATION 217: the TRUE gain is provably non-negative (the within-
    # group sum of squares never exceeds the total: Gini and variance
    # decompositions alike), so a negative value HERE is pure float32
    # cancellation -- measured on year at node scale, where sums near 3e8
    # put the three ~1e5-magnitude terms' rounding at the size of the true
    # gain and a VALID winner evaluated at -0.027, which `split_not_valid`
    # then leafed (half a tree gone at one seed). cuML ships this defect;
    # sklearn evaluates in float64 and does not. The clamp is exact, not
    # cosmetic: it restores the sign the mathematics guarantees. All three
    # gain forms (this device one, the host Gini, the host MSE) clamp
    # identically or the arms would grow different trees.
    if gain < Float32(0.0):
        gain = Float32(0.0)
    return gain


def entropy_gain_per_split(
    acc_left: MutPointer[Int32, MutAnyOrigin],
    acc_total: MutPointer[Int32, MutAnyOrigin],
    base: Int,
    nclasses: Int,
    len_in: Int32,
    n_left_in: Int32,
    min_samples_leaf: Int32,
) -> Float32:
    """`EntropyObjectiveFunction::GainPerSplit`, `objectives.cuh:132-168`,
    on device. DEVIATION 459.

    NOT a second transcription: the kernel's Int32 accumulator slices are
    bitcast to `CountBin` (one `Int32` field, same layout) and handed to the
    ONE transcription in `objectives.mojo`, which the host oracle also calls.
    `gain_per_split` above carries its own Gini copy for DEVIATION 183's
    historical reason; entropy arrives after the rule and takes the single
    copy. The `log` inside is `identical_log` -- the stdlib under FAST,
    `portable_logf` under IDENTICAL -- and the 217 clamp is applied inside.
    """
    var objective = EntropyObjectiveFunction[DType.float32](
        Int32(nclasses), min_samples_leaf
    )
    return objective.GainPerSplit(
        acc_left.unsafe_offset(base).unsafe_bitcast[CountBin](),
        acc_total.unsafe_offset(base).unsafe_bitcast[CountBin](),
        len_in,
        n_left_in,
    )


def score_to_candidate_kernel(
    cand_quesval: MutPointer[Float32, MutAnyOrigin],
    cand_colid: MutPointer[Int32, MutAnyOrigin],
    cand_metric: MutPointer[Float32, MutAnyOrigin],
    cand_nleft: MutPointer[Int32, MutAnyOrigin],
    cand_num: MutPointer[Int64, MutAnyOrigin],
    cand_den: MutPointer[Int64, MutAnyOrigin],
    cand_valid: MutPointer[Int32, MutAnyOrigin],
    in_status: MutPointer[Int32, MutAnyOrigin],
    in_threshold: MutPointer[Float32, MutAnyOrigin],
    in_n_left: MutPointer[Int32, MutAnyOrigin],
    in_n_total: MutPointer[Int32, MutAnyOrigin],
    in_acc_left: MutPointer[Int32, MutAnyOrigin],
    in_acc_total: MutPointer[Int32, MutAnyOrigin],
    in_gini_num: MutPointer[Int64, MutAnyOrigin],
    in_gini_den: MutPointer[Int64, MutAnyOrigin],
    colids: MutPointer[Int32, MutAnyOrigin],
    n_cells_in: Int32,
    n_classes_in: Int32,
    min_samples_leaf_in: Int32,
    criterion_in: Int32,
):
    """Scored cells into reduction candidates, elementwise.

    `criterion_in` selects the classification objective (DEVIATION 459):
    `CRITERION_GINI` publishes the finalize kernel's exact rational and
    cuML's Gini gain as the metric; `CRITERION_ENTROPY` computes cuML's
    entropy gain HERE, publishes it as the metric AND as the key
    (`float_gain_key(gain)` / `1`), so the reduction orders on the float
    gain exactly as `Split::update` would. The regression path passes
    `CRITERION_MSE` and takes the first branch (its pair is DEVIATION
    189's exact MSE key, its metric the `n_classes = 1` gain as before).

    ==================================================================
    DEVIATION BLOCK 182 -- this kernel exists because 170 split their
    kernel in two, and it has no cuML counterpart

    THEIRS: `computeSplitKernel`'s elected last block scores the bins and
    hands the result straight to `sp.evalBestSplit(...)` in the same
    function (`builder_kernels_impl.cuh:328-340`) -- the candidate never
    exists as memory.

    OURS: DEVIATION 170 could not elect a last block (`threadfence` is
    NVIDIA-only), so the score pass ends at a kernel boundary and its
    output is a struct-of-arrays in global memory. Something must turn
    that into the reduction's input layout, and this is it.

    WHY IT IS ELEMENTWISE AND NOT FUSED INTO EITHER NEIGHBOUR: fusing it
    into the finalize kernel would make that kernel write two layouts of
    the same fact, and fusing it into the reduction would make the
    reduction read a layout it does not own. Both couple two ported files
    to each other through a shape neither upstream has.

    THE ONE PIECE OF POLICY IN IT: a cell whose status is not SCORED
    becomes the DEFAULT `Split` -- `colid = -1`, `best_metric_val =
    MIN_FINITE` -- with an INVALID exact key. That is `initSplit`'s value
    (`split.cuh:54-59`), so a non-scored cell loses to every scored one
    under `Split.update` and ties with other non-scored cells under
    `compare_exact_key`. A node all of whose candidates were constant
    therefore reduces to `colid == -1`, which `split_not_valid` rejects
    and `NodeQueue.push` turns into a leaf -- the same outcome the host
    path reaches by never producing a candidate at all.

    `best_metric_val` IS ZERO FOR A SCORED CELL, and that is DEVIATION 175
    surfacing here rather than a shortcut: the device does not compute
    cuML's float `GainPerSplit`. For classification it does not need to --
    DEVIATION 145 makes the exact rational the authority and the float a
    reporting quantity -- but it means the metric arm of `Split.update`'s
    tie-break is dead on this path and ties fall through to `colid`. The
    host path has real gains there. `device_tree_check` MEASURES whether
    that ever changes a tree rather than assuming it does not.
    ==================================================================
    """
    var n_cells = Int(n_cells_in)
    var n_classes = Int(n_classes_in)
    var min_samples_leaf = min_samples_leaf_in
    var entropy = criterion_in == CRITERION_ENTROPY
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while idx < n_cells:
        if in_status[unsafe_offset=idx] == SCORE_STATUS_SCORED:
            cand_quesval[unsafe_offset=idx] = in_threshold[unsafe_offset=idx]
            cand_colid[unsafe_offset=idx] = colids[unsafe_offset=idx]
            cand_nleft[unsafe_offset=idx] = in_n_left[unsafe_offset=idx]
            if entropy:
                # DEVIATION 459: the float gain is the metric AND the key.
                var g = entropy_gain_per_split(
                    in_acc_left,
                    in_acc_total,
                    idx * n_classes,
                    n_classes,
                    in_n_total[unsafe_offset=idx],
                    in_n_left[unsafe_offset=idx],
                    min_samples_leaf,
                )
                cand_metric[unsafe_offset=idx] = g
                cand_num[unsafe_offset=idx] = float_gain_key(g)
                cand_den[unsafe_offset=idx] = Int64(1)
            else:
                cand_metric[unsafe_offset=idx] = gain_per_split(
                    in_acc_left,
                    in_acc_total,
                    idx * n_classes,
                    n_classes,
                    in_n_total[unsafe_offset=idx],
                    in_n_left[unsafe_offset=idx],
                    min_samples_leaf,
                )
                cand_num[unsafe_offset=idx] = in_gini_num[unsafe_offset=idx]
                cand_den[unsafe_offset=idx] = in_gini_den[unsafe_offset=idx]
            cand_valid[unsafe_offset=idx] = Int32(1)
        else:
            cand_quesval[unsafe_offset=idx] = Float32.MIN_FINITE
            cand_colid[unsafe_offset=idx] = Int32(-1)
            cand_metric[unsafe_offset=idx] = Float32.MIN_FINITE
            cand_nleft[unsafe_offset=idx] = Int32(0)
            cand_num[unsafe_offset=idx] = Int64(0)
            cand_den[unsafe_offset=idx] = Int64(0)
            cand_valid[unsafe_offset=idx] = Int32(0)
        idx += stride


def row_ids_sequence_kernel(
    row_ids: MutPointer[Int32, MutAnyOrigin],
    n_rows_in: Int32,
):
    """`thrust::sequence(..., selected_rows->begin(), selected_rows->end())`,
    `randomforest.cuh:69` -- the `bootstrap == false` arm of `get_row_sample`.

    ==================================================================
    DEVIATION BLOCK 200 -- NOT a deviation any more, and the entry
    exists to record what it replaced.

    cuML fills the row list ON THE DEVICE: `get_row_sample`
    (`randomforest.cuh:50-72`) writes into a `rmm::device_uvector` and
    `fit` hands that straight to the builder (`:169`, `:186`). The host
    never materialises the permutation.

    THIS LANE BUILT IT AS A HOST `List` AND UPLOADED IT, once per tree.
    It could not be a wrong answer -- with `bootstrap=False` the value is
    the identity permutation and nothing is being decided -- but it is
    one `n_rows` H2D copy per tree their design does not have, and rule 2
    is about the SHAPE and not only about decisions. A mirror audit of
    `doSplit` and `fit` found exactly three such drifts: the gain
    computed host-side (deviation 183, fixed), the feature sampler
    running host-side (deviation 195), and this one.

    A grid-stride write-only map, which is what `thrust::sequence` is.
    ==================================================================
    """
    var n_rows = Int(n_rows_in)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while idx < n_rows:
        row_ids[unsafe_offset=idx] = Int32(idx)
        idx += stride


def row_ids_tiled_sequence_kernel(
    row_ids: MutPointer[Int32, MutAnyOrigin],
    total_in: Int32,
    n_rows_in: Int32,
):
    """`row_ids_sequence_kernel` once per tree SLOT, in one launch.

    DEVIATION 211: the batched forest trainer keeps every in-flight tree's
    row list in ONE buffer, slot `s` at `[s * n_rows, (s + 1) * n_rows)`, and
    every slot starts as the same identity permutation the single-tree kernel
    writes -- `bootstrap=False`, so nothing is being decided, exactly as in
    DEVIATION 200. `row_ids[i] = i mod n_rows`.
    """
    var total = Int(total_in)
    var n_rows = Int(n_rows_in)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while idx < total:
        row_ids[unsafe_offset=idx] = Int32(idx - (idx // n_rows) * n_rows)
        idx += stride


def fill_row_slots(
    ctx: DeviceContext,
    mut d_row_ids: DeviceBuffer[DType.int32],
    g: Int,
    slot_rows: Int32,
    n_rows: Int32,
    bootstrap: Bool,
    tree_ids: List[Int32],
    first: Int,
    seed: UInt64,
) raises:
    """`get_row_sample` (`randomforest.cuh:50-72`) for every tree SLOT of one
    group, on the device. DEVIATION 460.

    THEIRS, per tree: `rs = fnv1a32(fnv1a32(basis, seed), tree_id)`;
    `rng(rs, GenPhilox)`; `bootstrap ? rng.uniformInt(selected_rows, 0,
    n_rows) : thrust::sequence(selected_rows)`.

    OURS: the `bootstrap == false` arm is `row_ids_tiled_sequence_kernel`
    (DEVIATION 200/211, unchanged); the `bootstrap == true` arm is
    `core.philox.launch_uniform_int` -- the RF lane's port of RAFT's
    `uniformInt` under `GenPhilox` (its DEVIATION 184 geometry, its oracle)
    -- called ONCE PER SLOT on a sub-buffer view of that slot, seeded by
    `row_sample_seed(seed, tree_id)` (`pcg_rng.mojo`, the same fnv1a32
    chain with the RF lane's DEVIATION 400 high-half round). Slot `s` is
    `[s * slot_rows, (s + 1) * slot_rows)` where `slot_rows` is
    `n_sampled_rows` (sklearn's `max_samples`, None = `n_rows`), so a
    bootstrap slot is exactly `selected_rows.size()` wide. No synchronize:
    every kernel that reads the slot is queue-ordered behind the draw, as
    theirs is behind `uniformInt` on the stream.
    """
    var total_rows = g * Int(slot_rows)
    if not bootstrap:
        ctx.enqueue_function[row_ids_tiled_sequence_kernel](
            d_row_ids.unsafe_ptr(),
            Int32(total_rows),
            slot_rows,
            grid_dim=ceildiv(total_rows, 128),
            block_dim=128,
        )
        return
    for s in range(g):
        var slot = d_row_ids.create_sub_buffer[DType.int32](
            s * Int(slot_rows), Int(slot_rows)
        )
        launch_uniform_int(
            ctx,
            slot,
            Int(slot_rows),
            Int32(0),
            n_rows,
            UInt64(Int(row_sample_seed(seed, tree_ids[first + s]))),
        )
        _ = slot^
    _ = d_row_ids.unsafe_ptr()


@fieldwise_init
struct DeviceDataset(Movable):
    """The dataset, resident on the device for a whole FOREST.

    ==================================================================
    DEVIATION BLOCK 184 -- CLOSED. The dataset is uploaded once per FIT,
    not once per tree.

    THEIRS: cuML's `Dataset` holds device pointers for the whole fit
    (`dataset.h:22-38`); every tree reads one resident copy.

    WHAT OURS DID FOR ONE ROUND: `train_classification_device` was
    written as a whole-tree entry point with no forest above it, so it
    allocated and filled `d_data` and `d_labels` on entry. An
    `n_trees`-tree forest therefore uploaded the same IMMUTABLE matrix
    `n_trees` times -- `n_trees - 1` redundant copies of
    `4*n_rows*n_cols + 4*n_rows` bytes, plus that many redundant host
    staging fills and `synchronize()` points.

    IT COULD NEVER HAVE BEEN A WRONG ANSWER, only redundant traffic:
    the matrix is immutable and every tree uploaded identical bytes.
    That is why it was allowed to stand for a round rather than being
    rushed -- and why closing it needed no re-checking of any result.

    THE SPLIT. `upload_dataset` is the old prologue; `_resident` is the
    old body. `train_classification_device` survives as a two-line
    wrapper so that `device_tree_check`, which fits ONE tree, is
    untouched and still exercises the same code. `row_ids` stays
    per-tree and per-call, because it is the one input that differs
    between trees -- see deviation 185, which measures that its `mut`
    is currently vacuous and pins the fact that makes it so.
    ==================================================================
    """

    var d_data: DeviceBuffer[DType.float32]
    var d_labels: DeviceBuffer[DType.int32]
    var n_rows: Int32
    var n_cols: Int32
    var n_classes: Int32


def upload_dataset(
    ctx: DeviceContext,
    x_col_major: List[Float32],
    class_ids: List[Int32],
    n_rows: Int32,
    n_cols: Int32,
    n_classes: Int32,
) raises -> DeviceDataset:
    """Put the immutable half of the fit on the device, once. DEVIATION 184."""
    if len(x_col_major) != Int(n_rows) * Int(n_cols):
        raise Error("x_col_major must be n_rows * n_cols long, column major")
    if len(class_ids) != Int(n_rows):
        raise Error("class_ids must be n_rows long")
    var d_data = ctx.enqueue_create_buffer[DType.float32](len(x_col_major))
    var d_labels = ctx.enqueue_create_buffer[DType.int32](Int(n_rows))
    var h_data = ctx.enqueue_create_host_buffer[DType.float32](
        len(x_col_major)
    )
    var h_labels = ctx.enqueue_create_host_buffer[DType.int32](Int(n_rows))
    ctx.synchronize()
    for i in range(len(x_col_major)):
        h_data.unsafe_ptr().unsafe_store(i, x_col_major[i])
    for i in range(Int(n_rows)):
        h_labels.unsafe_ptr().unsafe_store(i, class_ids[i])
    ctx.enqueue_copy(dst_buf=d_data, src_ptr=h_data.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_labels, src_ptr=h_labels.unsafe_ptr())
    ctx.synchronize()
    return DeviceDataset(d_data^, d_labels^, n_rows, n_cols, n_classes)


def train_classification_device(
    ctx: DeviceContext,
    x_col_major: List[Float32],
    class_ids: List[Int32],
    mut row_ids: List[Int32],
    n_rows: Int32,
    n_cols: Int32,
    n_classes: Int32,
    params: DecisionTreeParams,
    tree_id: Int32,
    seed: UInt64,
) raises -> TreeMetaDataNode[DType.float32]:
    """One tree, uploading the dataset for itself. DEVIATION 184's wrapper.

    A forest should call `upload_dataset` once and then
    `train_classification_device_resident` per tree. This name is kept for the
    single-tree callers, and because keeping it means `device_tree_check` goes
    on exercising the same body rather than a copy of it.
    """
    var dataset = upload_dataset(
        ctx, x_col_major, class_ids, n_rows, n_cols, n_classes
    )
    return train_classification_device_resident(
        ctx, dataset, row_ids, params, tree_id, seed
    )


def sample_features_for_device[
    os: MutOrigin, orp: MutOrigin, ow: MutOrigin, ot: MutOrigin, //
](
    ctx: DeviceContext,
    mut d_colids_buf: DeviceBuffer[DType.int32],
    d_scratch: MutPointer[Int32, os],
    d_report: MutPointer[Int32, orp],
    d_work_items: MutPointer[NodeWorkItem, ow],
    d_tree_ids: MutPointer[Int32, ot],
    mut h_colids_stage: HostBuffer[DType.int32],
    work_items: List[NodeWorkItem],
    item_trees: List[Int32],
    seed: UInt64,
    n: Int,
    k: Int,
) raises -> FeatureSamplerPlan:
    """Sample this batch's features WHERE cuML SAMPLES THEM, with one arm that
    cannot run there and is named rather than hidden.

    ==================================================================
    DEVIATION BLOCK 201 -- the excess arm and the all-features arm run
    on the DEVICE; ALGORITHM L runs on the HOST, because Metal has no
    `double`.

    THEIRS: `builder.cuh:398-471` computes `n_parallel_samples` on the
    host and then launches one of three kernels. The `colids` array is
    produced on the device and the host never sees it.

    OURS, per arm:
      * `SAMPLE_EXCESS` and `SAMPLE_ALL_FEATURES` -- `sample_features_device`,
        which enqueues their kernel. Bit-identical to the host oracle over
        23,462 asserted slots.
      * `SAMPLE_ALGO_L` -- the HOST transcription, uploaded.

    WHY ALGORITHM L CANNOT RUN THERE, measured rather than assumed:
    cuML's algorithm L is a `double` algorithm in four places
    (`builder_kernels.cuh:291`, `:306` twice, `:313`) and Metal rejects
    `double` AT COMPILE TIME -- "function's return type 'double' is not
    supported", "llvm.fma.f64 has Metal-unsupported instructions". This
    is the same wall as no streams and no `threadfence`, and it is in the
    traps register.

    WHY NOT A FLOAT32 SUBSTITUTE: at the `k/n` their dispatch actually
    routes to this arm, `W` is about `1 - 1e-4`, so forming `1 - W` in
    `Float32` is catastrophic cancellation -- roughly 13 bits survive.
    That would be a DIFFERENT ALGORITHM wearing this one's name, on the
    one arm nobody would look at. Tracking `V = 1 - W` through `expm1f`
    was rejected for the opposite reason: it is numerically BETTER than
    cuML, and this is a port.

    WHY NOT REFUSE THE ARM: refusing would make the device path unusable
    whenever `k/n` is near 1 at large `n`, and the host transcription is
    not a guess -- it is the checked oracle the device kernels are
    verified against, cell for cell, over 1,063,780 cells. Running THEIR
    algorithm on the host is a placement difference; refusing to fit is a
    capability loss. The placement is reported in the returned plan, so a
    caller can see which arm ran and where.

    THE PRICE, stated: on a target without `double` the algo-L arm costs
    one `work_items_size * k` H2D copy per level that their design does
    not have. On CUDA and ROCm, where `double` exists, the same call
    takes the device kernel and the copy disappears -- the code is one
    source and the branch is a host-side capability query, never an
    `if apple` inside a kernel.
    ==================================================================
    """
    var plan = plan_feature_sampling(n, k)
    if plan.arm != SAMPLE_ALGO_L or device_has_float64():
        return sample_features_device(
            ctx,
            d_colids_buf.unsafe_ptr(),
            d_scratch,
            d_report,
            d_work_items,
            d_tree_ids,
            len(work_items),
            seed,
            n,
            k,
        )
    var host_colids = List[Int32](
        length=len(work_items) * k, fill=Int32(0)
    )
    _ = sample_features_pertree(
        host_colids, work_items, item_trees, seed, n, k
    )
    for i in range(len(host_colids)):
        h_colids_stage.unsafe_ptr().unsafe_store(i, host_colids[i])
    ctx.enqueue_copy(
        dst_buf=d_colids_buf, src_ptr=h_colids_stage.unsafe_ptr()
    )
    return plan


@fieldwise_init
struct LevelWorkspace(Movable):
    """Every per-level buffer, allocated ONCE, as `assignWorkspace` does.

    ==================================================================
    DEVIATION BLOCK -- DEVIATION 202. THE WORKSPACE IS ALLOCATED ONCE
    PER TREE, NOT ONCE PER LEVEL. (Since DEVIATION 211: once per GROUP
    of in-flight trees -- the forest trainers own it now.)

    THEIRS: cuML sizes the whole workspace up front from
    `params.max_batch_size` (`workspaceSize`, `builder.cuh:272-296`) and
    hands out pointers into one allocation (`assignWorkspace`,
    `builder.cuh:302-341`). Nothing in their level loop allocates. The
    sizes are capacities, not the current level's occupancy:
    `max_batch * n_sampled_cols` for `colids`, `max_batch` for `splits`
    and `d_work_items`, and `max_blocks_dimx` -- which is
    `1 + params.max_batch_size + dataset.n_sampled_rows / TPB_DEFAULT`
    (`builder.cuh:230`) -- for `workload_info`.

    WHAT OURS DID FOR SEVEN ROUNDS: allocated all 51 of them INSIDE the
    `while queue.has_work()` loop, sized to the current level. A
    depth-12 tree runs 13 levels, so a ten-tree forest performed about
    5,500 buffer creations where cuML performs ten.

    IT WAS NEVER A WRONG ANSWER, and that is why it survived: every
    buffer is explicitly initialised before use -- by
    `node_feature_range_init_kernel`, `node_feature_score_init_kernel`,
    `split_reduce_init_kernel`, or an `enqueue_memset` (since DEVIATION
    470, the shipped loop runs all six seeders as TWO fused launches --
    `phase_setup_a_kernel` / `_b_kernel` -- that call the same seed
    bodies) -- so a
    reused buffer and a fresh one are indistinguishable to every kernel
    that reads one. The identity checks could not see it and did not.

    WHAT IT COST, MEASURED. Time per level was flat in the amount of
    WORK: at 581,012 rows a level-iteration cost 24 ms at
    `max_features=5` and 32.5 ms at `max_features=54`, and total time
    tracked LEVEL COUNT almost exactly (5, 9 and 13 levels -> 662,
    1067, 1421 ms at four trees). A cost that does not move when the
    work moves is not the work.

    REUSE ACROSS LEVELS IS SAFE FOR THE HOST STAGING BUFFERS TOO, and
    that needed an argument rather than a hope: the copies out of them
    are asynchronous, so a staging buffer rewritten under an in-flight
    copy would corrupt it. Every level ends with a `synchronize()`
    before the splits are read back, so by the time the loop returns to
    the top, every copy issued by the previous level has completed.
    ==================================================================
    """

    var d_min: DeviceBuffer[DType.float32]
    var d_max: DeviceBuffer[DType.float32]
    var d_thresh: DeviceBuffer[DType.float32]
    var c_q: DeviceBuffer[DType.float32]
    var c_m: DeviceBuffer[DType.float32]
    var d_missing: DeviceBuffer[DType.int32]
    var d_merges: DeviceBuffer[DType.int32]
    var d_minkey: DeviceBuffer[DType.uint32]
    var d_maxkey: DeviceBuffer[DType.uint32]
    var d_nleft: DeviceBuffer[DType.int32]
    var d_ntotal: DeviceBuffer[DType.int32]
    var d_nblocks: DeviceBuffer[DType.int32]
    var d_status: DeviceBuffer[DType.int32]
    var c_c: DeviceBuffer[DType.int32]
    var c_l: DeviceBuffer[DType.int32]
    var c_v: DeviceBuffer[DType.int32]
    var d_colids: DeviceBuffer[DType.int32]
    var d_gnum: DeviceBuffer[DType.int64]
    var d_gden: DeviceBuffer[DType.int64]
    var c_nu: DeviceBuffer[DType.int64]
    var c_de: DeviceBuffer[DType.int64]
    var d_accl: DeviceBuffer[DType.int32]
    var d_acct: DeviceBuffer[DType.int32]
    var r_q: DeviceBuffer[DType.float32]
    var r_m: DeviceBuffer[DType.float32]
    var r_c: DeviceBuffer[DType.int32]
    var r_l: DeviceBuffer[DType.int32]
    var r_v: DeviceBuffer[DType.int32]
    var r_mg: DeviceBuffer[DType.int32]
    var r_nw: DeviceBuffer[DType.int32]
    var r_mx: DeviceBuffer[DType.int32]
    var d_tree: DeviceBuffer[DType.int32]
    var h_tree: HostBuffer[DType.int32]
    """DEVIATION 211: one tree id per work item in the staged batch, read by
    the score, finalize and sampler kernels as the tree component of every
    draw key. A single-tree batch stages one value repeated."""
    var d_tsalt: DeviceBuffer[DType.uint32]
    var h_tsalt: HostBuffer[DType.uint32]
    """DEVIATION 463: one tie-break rank salt per work item,
    `split_tie_salt_for(item_trees[i], work_items[i].idx)`, staged by
    `stage_batch` and read by `split_reduce_kernel` as `node_tie_salt`."""
    var d_ties: DeviceBuffer[DType.int32]
    var o_ties: HostBuffer[DType.int32]
    """DEVIATION 463's exact-tie counter cells (`split_tie_count_kernel`),
    written and read back only under `-D MOJOLEARN_ET_TIE_STATS=1`."""
    var d_nb: DeviceBuffer[DType.int32]
    var d_nc: DeviceBuffer[DType.int32]
    var d_iters: DeviceBuffer[DType.int32]
    var d_swaps: DeviceBuffer[DType.int32]
    var r_nu: DeviceBuffer[DType.int64]
    var r_de: DeviceBuffer[DType.int64]
    var d_samp_scratch: DeviceBuffer[DType.int32]
    var d_samp_report: DeviceBuffer[DType.int32]
    var d_items: DeviceBuffer[DType.uint8]
    var d_wl: DeviceBuffer[DType.uint8]
    var d_nonconst: DeviceBuffer[DType.int32]
    var h_nonconst: HostBuffer[DType.int32]
    var o_rmin: HostBuffer[DType.float32]
    var o_rmax: HostBuffer[DType.float32]
    var o_rmiss: HostBuffer[DType.int32]
    var d_blk_left: DeviceBuffer[DType.int32]
    var d_blk_off: DeviceBuffer[DType.int32]
    var d_blk_base: DeviceBuffer[DType.int32]
    var h_blk_base: HostBuffer[DType.int32]
    var d_row_alt: DeviceBuffer[DType.int32]
    var d_splits: DeviceBuffer[DType.uint8]
    var h_colids: HostBuffer[DType.int32]
    var h_items: HostBuffer[DType.uint8]
    var h_wl: HostBuffer[DType.uint8]
    var o_q: HostBuffer[DType.float32]
    var o_m: HostBuffer[DType.float32]
    var o_c: HostBuffer[DType.int32]
    var o_l: HostBuffer[DType.int32]
    var o_v: HostBuffer[DType.int32]
    var h_nb: HostBuffer[DType.int32]
    var h_nc: HostBuffer[DType.int32]
    var o_nu: HostBuffer[DType.int64]
    var o_de: HostBuffer[DType.int64]
    var h_splits: HostBuffer[DType.uint8]
    var cap_nodes: Int
    """The workspace's node CAPACITY (`max_batch`), not any batch's live
    count. DEVIATION 470's fused seeder covers `d_nonconst` and `r_mx` to
    this extent -- exactly what the two `enqueue_memset`s it replaced
    covered -- and DEVIATION 472's byte-compares run over it."""
    var cap_blocks: Int
    """The workload-info BLOCK capacity (`builder.cuh:230`'s bound), for
    DEVIATION 472's `d_wl` byte-compare extent."""
    var cap_report: Int
    """`sampler_report_len(cap_nodes)`: the full extent of `d_samp_report`,
    which DEVIATION 470's fused seeder covers on sampler cycles -- exactly
    what the `enqueue_memset` it replaced covered."""
    var s_items: HostBuffer[DType.uint8]
    var s_tree: HostBuffer[DType.int32]
    var s_tsalt: HostBuffer[DType.uint32]
    var s_wl: HostBuffer[DType.uint8]
    var s_nb: HostBuffer[DType.int32]
    var s_nc: HostBuffer[DType.int32]
    var s_blk_base: HostBuffer[DType.int32]
    """DEVIATION 472: one shadow per staged slot, holding the bytes LAST
    ENQUEUED to the slot's device buffer, over the FULL capacity extent the
    copy sends. `stage_batch` byte-compares against these and skips the
    `enqueue_copy` on equality; host-only, never read by the device."""
    var stage_valid: Bool
    """False until `stage_batch`'s first upload: pinned shadows arrive with
    arbitrary bytes, so the first stage always copies and snapshots."""


def make_level_workspace(
    ctx: DeviceContext,
    max_batch: Int,
    n_rows: Int32,
    n_cols: Int32,
    n_classes: Int32,
    k: Int,
    tpb: Int,
) raises -> LevelWorkspace:
    """`workspaceSize` + `assignWorkspace`, in one call.

    `blocks` is `builder.cuh:230` transcribed:
    `1 + params.max_batch_size + dataset.n_sampled_rows / TPB_DEFAULT`.
    That is the bound because `n_blocks_dimx` is
    `sum_i ceil(count_i / tpb)` over the batch, `sum_i count_i <= n_rows`,
    and each of at most `max_batch` nodes contributes at most one extra
    block from the ceiling.
    """
    var nodes = max_batch
    # THE CAPACITY IS `n_cols`, NOT `k`. DEVIATION 205's survey runs the same
    # range pass with EVERY column, so the widest batch this workspace has to
    # hold is `max_batch * n_cols`, not `max_batch * k`. Sizing it by `k`
    # worked on every fixture where `k * max_batch` happened to exceed the
    # survey's cells and would have written past the end where it did not.
    var k_cap = k if k > Int(n_cols) else Int(n_cols)
    var cells = nodes * k_cap
    var blocks = 1 + max_batch + Int(n_rows) // tpb
    var ws = LevelWorkspace(
        d_min=ctx.enqueue_create_buffer[DType.float32](cells),
        d_max=ctx.enqueue_create_buffer[DType.float32](cells),
        d_thresh=ctx.enqueue_create_buffer[DType.float32](cells),
        c_q=ctx.enqueue_create_buffer[DType.float32](cells),
        c_m=ctx.enqueue_create_buffer[DType.float32](cells),
        d_missing=ctx.enqueue_create_buffer[DType.int32](cells),
        d_merges=ctx.enqueue_create_buffer[DType.int32](cells),
        d_minkey=ctx.enqueue_create_buffer[DType.uint32](cells),
        d_maxkey=ctx.enqueue_create_buffer[DType.uint32](cells),
        d_nleft=ctx.enqueue_create_buffer[DType.int32](cells),
        d_ntotal=ctx.enqueue_create_buffer[DType.int32](cells),
        d_nblocks=ctx.enqueue_create_buffer[DType.int32](cells),
        d_status=ctx.enqueue_create_buffer[DType.int32](cells),
        c_c=ctx.enqueue_create_buffer[DType.int32](cells),
        c_l=ctx.enqueue_create_buffer[DType.int32](cells),
        c_v=ctx.enqueue_create_buffer[DType.int32](cells),
        d_colids=ctx.enqueue_create_buffer[DType.int32](cells),
        d_gnum=ctx.enqueue_create_buffer[DType.int64](cells),
        d_gden=ctx.enqueue_create_buffer[DType.int64](cells),
        c_nu=ctx.enqueue_create_buffer[DType.int64](cells),
        c_de=ctx.enqueue_create_buffer[DType.int64](cells),
        d_accl=ctx.enqueue_create_buffer[DType.int32](cells * Int(n_classes)),
        d_acct=ctx.enqueue_create_buffer[DType.int32](cells * Int(n_classes)),
        r_q=ctx.enqueue_create_buffer[DType.float32](nodes),
        r_m=ctx.enqueue_create_buffer[DType.float32](nodes),
        r_c=ctx.enqueue_create_buffer[DType.int32](nodes),
        r_l=ctx.enqueue_create_buffer[DType.int32](nodes),
        r_v=ctx.enqueue_create_buffer[DType.int32](nodes),
        r_mg=ctx.enqueue_create_buffer[DType.int32](nodes),
        r_nw=ctx.enqueue_create_buffer[DType.int32](nodes),
        r_mx=ctx.enqueue_create_buffer[DType.int32](nodes),
        d_tree=ctx.enqueue_create_buffer[DType.int32](nodes),
        h_tree=ctx.enqueue_create_host_buffer[DType.int32](nodes),
        d_tsalt=ctx.enqueue_create_buffer[DType.uint32](nodes),
        h_tsalt=ctx.enqueue_create_host_buffer[DType.uint32](nodes),
        d_ties=ctx.enqueue_create_buffer[DType.int32](nodes),
        o_ties=ctx.enqueue_create_host_buffer[DType.int32](nodes),
        d_nb=ctx.enqueue_create_buffer[DType.int32](nodes),
        d_nc=ctx.enqueue_create_buffer[DType.int32](nodes),
        d_iters=ctx.enqueue_create_buffer[DType.int32](nodes),
        d_swaps=ctx.enqueue_create_buffer[DType.int32](nodes),
        r_nu=ctx.enqueue_create_buffer[DType.int64](nodes),
        r_de=ctx.enqueue_create_buffer[DType.int64](nodes),
        d_samp_scratch=ctx.enqueue_create_buffer[DType.int32](sampler_scratch_len(nodes, Int(n_cols), k)),
        d_samp_report=ctx.enqueue_create_buffer[DType.int32](sampler_report_len(nodes)),
        d_items=ctx.enqueue_create_buffer[DType.uint8](nodes * size_of[NodeWorkItem]()),
        d_wl=ctx.enqueue_create_buffer[DType.uint8](blocks * size_of[WorkloadInfo]()),
        d_nonconst=ctx.enqueue_create_buffer[DType.int32](nodes),
        h_nonconst=ctx.enqueue_create_host_buffer[DType.int32](nodes),
        o_rmin=ctx.enqueue_create_host_buffer[DType.float32](cells),
        o_rmax=ctx.enqueue_create_host_buffer[DType.float32](cells),
        o_rmiss=ctx.enqueue_create_host_buffer[DType.int32](cells),
        d_blk_left=ctx.enqueue_create_buffer[DType.int32](blocks),
        d_blk_off=ctx.enqueue_create_buffer[DType.int32](blocks),
        d_blk_base=ctx.enqueue_create_buffer[DType.int32](nodes),
        h_blk_base=ctx.enqueue_create_host_buffer[DType.int32](nodes),
        d_row_alt=ctx.enqueue_create_buffer[DType.int32](Int(n_rows)),
        d_splits=ctx.enqueue_create_buffer[DType.uint8](nodes * size_of[Split]()),
        h_colids=ctx.enqueue_create_host_buffer[DType.int32](cells),
        h_items=ctx.enqueue_create_host_buffer[DType.uint8](nodes * size_of[NodeWorkItem]()),
        h_wl=ctx.enqueue_create_host_buffer[DType.uint8](blocks * size_of[WorkloadInfo]()),
        o_q=ctx.enqueue_create_host_buffer[DType.float32](nodes),
        o_m=ctx.enqueue_create_host_buffer[DType.float32](nodes),
        o_c=ctx.enqueue_create_host_buffer[DType.int32](nodes),
        o_l=ctx.enqueue_create_host_buffer[DType.int32](nodes),
        o_v=ctx.enqueue_create_host_buffer[DType.int32](nodes),
        h_nb=ctx.enqueue_create_host_buffer[DType.int32](nodes),
        h_nc=ctx.enqueue_create_host_buffer[DType.int32](nodes),
        o_nu=ctx.enqueue_create_host_buffer[DType.int64](nodes),
        o_de=ctx.enqueue_create_host_buffer[DType.int64](nodes),
        h_splits=ctx.enqueue_create_host_buffer[DType.uint8](nodes * size_of[Split]()),
        cap_nodes=nodes,
        cap_blocks=blocks,
        cap_report=sampler_report_len(nodes),
        s_items=ctx.enqueue_create_host_buffer[DType.uint8](nodes * size_of[NodeWorkItem]()),
        s_tree=ctx.enqueue_create_host_buffer[DType.int32](nodes),
        s_tsalt=ctx.enqueue_create_host_buffer[DType.uint32](nodes),
        s_wl=ctx.enqueue_create_host_buffer[DType.uint8](blocks * size_of[WorkloadInfo]()),
        s_nb=ctx.enqueue_create_host_buffer[DType.int32](nodes),
        s_nc=ctx.enqueue_create_host_buffer[DType.int32](nodes),
        s_blk_base=ctx.enqueue_create_host_buffer[DType.int32](nodes),
        stage_valid=False,
    )
    # DEVIATION 450: the workspace's ONE materialization drain. Every host
    # buffer above is created by an ENQUEUED op, and the level loop writes
    # them through raw pointers; this sync -- once per GROUP, in setup --
    # is what lets the per-cycle synchronizes that used to guard those
    # writes come out. The per-site arguments live in DEVIATIONS.md 450.
    ctx.synchronize()
    return ws^


comptime DEVICE_TPB = _device_tpb()
"""Threads per block for every device pass in this file.

One definition, because `build_workload_info` tiles the frontier by it and
DEVIATION 203's partition assumes that tiling: two copies that drifted would
put a block's rows somewhere its scatter does not write.

The value comes from `_device_tpb()`: 128 (cuML's `TPB_DEFAULT`) unless a
measurement arm overrides it with `-D MOJOLEARN_ET_TPB_256=1` or
`-D MOJOLEARN_ET_TPB_512=1` (tools/et_profile_leg.sh builds those arms
side by side with the default on a rented box).
"""


def _device_tpb() -> Int:
    """The block width of every frontier pass, chosen at compile time.

    ==================================================================
    DEVIATION 1943 -- the frontier block is 512 wide on a 64-lane
    wavefront, 128 (cuML's `TPB_DEFAULT`) on a 32-lane warp.

    `build_workload_info` gives each block exactly TPB rows, one row per
    thread, so the range, score and partition passes launch
    `n_rows / TPB` blocks per node per feature and each block does one
    compare per thread, six block reductions and a handful of atomics.
    At 128 that is a two-wavefront workgroup on CDNA, and the MI325X is
    bound by the workgroup dispatch rate rather than by the work:
    measured on higgs 1M x 28, 100 trees, depth 16 (leg
    2026-08-29_202227-mojolearn-e2-amd, `lanes/et_profile/`), the range
    pass took 8107 ms at TPB 128, 4614 ms at 256 and 2819 ms at 512; the
    score pass 8799 / 5186 / 3288 ms. The partition, reduce and leaf
    passes did not move. The same 128 is what the H100 and the M4 run at
    and neither shows the signature (the H100's whole fit is 4160 ms), so
    the row is keyed on the one compile-time fact that separates the
    vendors, `WARP_SIZE`, and is a no-op by construction where it is 32.

    THE BITS DO NOT DEPEND ON TPB. The range fold is in key space under
    IDENTICAL (DEVIATION 452) and an integer min/max across blocks
    (DEVIATION 204); the score pass sums integers through atomics
    (DEVIATION 135/171); the partition is stable by row index whatever
    the tiling (DEVIATION 203); the reduce runs one block per node at
    every `k <= TPB`. `partition_multiblock_check` already asserts the
    multi-block partition against the one-block oracle cell by cell, and
    phase 9's et-clf/et-reg stability lanes on the AMD leg are the gate
    that the identical hashes did not move.

    The defines are the measurement arms: `-D MOJOLEARN_ET_TPB_128`
    forces cuML's width on a 64-lane device for the A/B,
    `MOJOLEARN_ET_TPB_256` / `_512` widen a 32-lane device, and
    `MOJOLEARN_ET_TPB_1024` (added 2026-09-01 with DEVIATION 2020, and
    it is the sweep this entry's "Owed" paragraph already names: CDNA's
    maximum, never yet timed) widens either. 1024 is
    `MAX_THREADS_PER_BLOCK` on both Metal and CDNA (the traps register
    measured Apple's), and every static carve keyed on TPB stays under
    the 32 KiB shared floor at 1024: the partition's `2 * TPB` Int32
    carve is 8 KiB, `split_reduce_shared_bytes(1024, 64)` is 576 bytes.
    None of the defines is set by any build script. Note the interplay
    with DEVIATION 2020's `SEARCH_ROWS_PER_THREAD`: the search tile is
    `TPB * R`, so the two families multiply and an A/B sweeping both
    must alternate arms inside one window, not assume independence.
    ==================================================================
    """
    if is_defined["MOJOLEARN_ET_TPB_1024"]():
        return 1024
    if is_defined["MOJOLEARN_ET_TPB_512"]():
        return 512
    if is_defined["MOJOLEARN_ET_TPB_256"]():
        return 256
    if is_defined["MOJOLEARN_ET_TPB_128"]():
        return 128
    return 128 if WARP_SIZE <= 32 else 512

comptime DEVICE_MAX_ACC = _device_max_acc()
"""Widest per-cell class accumulator the score kernel's shared memory admits
(DEVIATION 172). One definition, for the same reason DEVICE_TPB is one.
32 unless a DEVIATION 2021 measurement arm narrows it -- see
`_device_max_acc()`."""


def _device_max_acc() -> Int:
    """The score pass's private-accumulator width, chosen at compile time.

    ==================================================================
    DEVIATION 2021 -- the per-thread accumulator arrays are sized by a
    comptime arm, because 32 slots price every fit for the widest fit

    WHERE THE 32 COMES FROM. DEVIATION 172 replaced cuML's dynamic
    shared-memory histogram with per-thread private arrays plus a block
    reduction, and sized them at a comptime 32 because
    `stack_allocation`'s slot count must be comptime (the same wall,
    from the other side, as the partition's static carve in DEVIATION
    176). cuML has no counterpart decision: their accumulator is the
    shared histogram, sized at runtime per launch
    (`builder_kernels_impl.cuh:235`, `extern __shared__`), so a binary
    fit never carries a 32-wide anything.

    WHAT THE 32 COSTS AT n_classes = 2. `node_feature_score_kernel`
    allocates TWO `stack_allocation[MAX_ACC, Int32]` arrays and zeroes
    all 2 * MAX_ACC slots per thread per block -- and because the row
    loop indexes them by the RUNTIME class id, the backend cannot keep a
    runtime-indexed array in registers: it lands in per-thread scratch
    (local memory on CUDA/HIP, thread stack on Metal), 256 bytes per
    thread, 128 KiB per 512-thread workgroup, ALL of it sized for 32
    classes when higgs has 2 and every regression fit has 1. Per-thread
    footprint is an OCCUPANCY divisor on every vendor, and the standing
    diagnosis of the Apple score pass is memory-LATENCY bound (DEVIATION
    208's probe) -- the regime where occupancy is the lever, because
    more resident blocks are what hide gather latency.

    THE ARMS. `-D MOJOLEARN_ET_MAX_ACC_4` / `_8` / `_16` narrow the
    comptime width; the default stays 32 and compiles the exact
    pre-2021 program. None is set by any build script. The bits cannot
    move: the arrays hold the SAME integers in the same slots and the
    unused tail was all zeros folded through integer sums -- removing a
    zero from an integer sum is the identity. The guard is already
    LOUD, not silent: the classification forest trainer raises by name
    when `n_classes > DEVICE_MAX_ACC` ("the device score kernel is built
    for at most ..."), so an arm too narrow for its dataset refuses
    the fit rather than mis-scoring it -- gate arms accordingly (the
    lane's fixtures fit `_8`; higgs2m and every regression fit `_4`).

    NOT taken instead: making the width a runtime kernel argument
    (impossible -- comptime slot count), or shrinking the DEFAULT
    (a default flips only on the orchestrator's measured bit-identical
    win, and a narrowed default would newly refuse legal 17-32-class
    fits; if the win is real the shipping shape is a small dispatch
    over two or three instantiations, which is a follow-up decision,
    not this arm). UNVERIFIED, RUN OWED: the A/B commands live in
    DEVIATIONS.md 2021 and PLAN.md.
    ==================================================================
    """
    if is_defined["MOJOLEARN_ET_MAX_ACC_4"]():
        return 4
    if is_defined["MOJOLEARN_ET_MAX_ACC_8"]():
        return 8
    if is_defined["MOJOLEARN_ET_MAX_ACC_16"]():
        return 16
    return 32

comptime FOREST_SAB_NONE = Int32(0)
comptime FOREST_SAB_SCALAR_TREE = Int32(1)
"""DEVIATION 211 sabotage: every item in a merged batch is staged with the
FIRST item's tree id, which is what a port that kept the per-launch scalar
would silently do. Every tree but the batch-first one must move."""
comptime FOREST_SAB_SHARED_ROW_BASE = Int32(2)
"""DEVIATION 211 sabotage: every in-flight tree's root range starts at slot
0, so their partitions overwrite each other. The forest must move -- this is
the gate watching that the slot offsets are what isolate the trees."""

comptime PHASE_SETUP = 0
comptime PHASE_STAGE = 1
comptime PHASE_RANGE = 2
comptime PHASE_SCORE = 3
comptime PHASE_REDUCE = 4
comptime PHASE_HOST_SPLITS = 5
comptime PHASE_PARTITION = 6
comptime PHASE_HOST_QUEUE = 7
comptime PHASE_LEAF = 8
comptime PHASE_HOST_PUSH = 9
comptime N_PHASES = 10


struct PhaseClock(Movable):
    """Per-phase wall time for one forest fit -- the lane's MICRO-STEP clock.

    DISABLED (the default and the only state any shipping caller uses) it is
    inert: `tick` does nothing, no synchronize is inserted, and the fit is
    byte-for-byte the untimed program. ENABLED, every phase boundary becomes
    `ctx.synchronize()` + a host clock read, which SERIALIZES the pipeline
    it measures -- the number is the duration of phases no longer allowed to
    overlap, which is a DIFFERENT PROGRAM (the RF lane's profiler states the
    same caution). That is why the clocked entry points are separate
    `*_timed` functions, why nothing on the fit path constructs an enabled
    clock, and why any report from this struct must print the clocked total
    NEXT TO an unclocked run of the same config: the gap between them is the
    measurement's own distortion, stated instead of hidden.

    Why it exists anyway: Apple Instruments gives dispatch durations but the
    stock template cannot name kernels (unnamed encoders), and DEVIATIONS
    212/213 were both chosen from whole-fit inference and both measured out
    as washes. Attribution has to come from somewhere; this is the exact
    per-phase form, priced honestly.
    """

    var enabled: Bool
    var ns: List[Int64]
    var last: Int64

    def __init__(out self, enabled: Bool = False):
        self.enabled = enabled
        self.ns = List[Int64](length=N_PHASES, fill=Int64(0))
        self.last = Int64(0)

    def mark(mut self, ctx: DeviceContext) raises:
        """Set the clock without charging any phase -- the fit's start."""
        if not self.enabled:
            return
        ctx.synchronize()
        self.last = Int64(perf_counter_ns())

    def tick(mut self, ctx: DeviceContext, phase: Int) raises:
        """Charge everything since the previous boundary to `phase`."""
        if not self.enabled:
            return
        ctx.synchronize()
        var now = Int64(perf_counter_ns())
        self.ns[phase] += now - self.last
        self.last = now

    def phase_name(self, phase: Int) -> String:
        if phase == PHASE_SETUP:
            return "setup (buffers, row fill, workspace)"
        if phase == PHASE_STAGE:
            return "stage + feature sampler"
        if phase == PHASE_RANGE:
            return "range pass (init+range+decode+nonconst)"
        if phase == PHASE_SCORE:
            return "score pass (init+score+finalize)"
        if phase == PHASE_REDUCE:
            return "candidate+reduce+splits readback"
        if phase == PHASE_HOST_SPLITS:
            return "host: split records"
        if phase == PHASE_PARTITION:
            return "partition (4 kernels)"
        if phase == PHASE_HOST_QUEUE:
            return "host: pop + batch assembly"
        if phase == PHASE_HOST_PUSH:
            return "host: queue push (children of the batch)"
        if phase == PHASE_LEAF:
            return "leaf pass"
        return "?"


comptime STAGE_TIMES_ENV = "MOJOLEARN_STAGE_TIMES"
"""Set `MOJOLEARN_STAGE_TIMES=1` and the shipping forest entry points run
their fit under an ENABLED `PhaseClock` and print stage -> seconds at fit
end. Read ONCE PER FIT, in the wrapper; unset (the shipping state), the
wrappers construct the same inert clock they always did and the fit is
byte-for-byte the untimed program."""


def stage_times_enabled() -> Bool:
    """The one place `STAGE_TIMES_ENV` is read: once, at fit entry."""
    return String(getenv(STAGE_TIMES_ENV)) == "1"


def print_stage_times(clock: PhaseClock, what: StringSlice) raises:
    """Stage -> seconds for one fit, printed at fit end. Inert clock: silent.

    The caution is `PhaseClock`'s, restated where the number lands: every
    phase boundary of an enabled clock is a `synchronize`, so these are the
    durations of phases FORBIDDEN TO OVERLAP -- a different program from the
    shipping fit. Read them as attribution, never as a benchmark; the gap to
    an unclocked run of the same config is the measurement's own distortion.
    """
    if not clock.enabled:
        return
    var total = Int64(0)
    for p in range(N_PHASES):
        total += clock.ns[p]
    print("MOJOLEARN_STAGE_TIMES [" + String(what) + "]")
    print("  (serialized-by-measurement; compare total to an untimed run)")
    for p in range(N_PHASES):
        print(
            "  ",
            clock.phase_name(p),
            "->",
            Float64(clock.ns[p]) / 1e9,
            "s",
        )
    print("  total ->", Float64(total) / 1e9, "s")


comptime FOREST_ROW_SLOT_CAP = 1 << 26
"""Ceiling on `in-flight trees * n_rows` row SLOTS one group of the batched
forest trainer (DEVIATION 211) may hold: 2^26 slots = 256 MB in `d_row_ids`
plus the same again in the partition's alternate buffer. Trees beyond the cap
run as further groups, sequentially. At 100,000 rows the cap admits 671 trees
in one group; at covtype's 581,012 it admits 115 -- a default-sized forest is
one group in both regimes. The bound also keeps every `InstanceRange.begin`
comfortably inside `Int32`."""


def _stage_upload_if_changed[
    dt: DType, //
](
    ctx: DeviceContext,
    mut dst: DeviceBuffer[dt],
    src: HostBuffer[dt],
    mut shadow: HostBuffer[dt],
    n: Int,
    seen_before: Bool,
    payload_slot: Bool,
) raises:
    """DEVIATION 472: enqueue one of `stage_batch`'s H2D copies ONLY when
    its bytes moved since the last enqueue.

    `shadow` holds the bytes last enqueued for this slot, over the SAME full
    capacity extent `enqueue_copy(dst_buf=...)` sends; on equality the
    device already holds this value (the queue is in-order and every kernel
    reading the buffer is enqueued after the copy that staged it), so
    re-sending is pure transfer waste. The comparison is over EXACT BYTES,
    never a semantic summary, so any change -- the rescue's `k` flip from
    `n_cols` to 1 included -- restages automatically with no bookkeeping.
    The fail-safe direction is DEVIATION 1917's (ensemble): a spurious
    mismatch (a stale pinned-tail byte, struct padding) costs one extra
    copy; a skip happens only on bytewise equality, so a changed value is
    never skipped. The snapshot taken here is faithful to what the copy will
    send because DEVIATION 450's invariant keeps `src` unrewritten until
    after the next required drain retires the copy.

    `payload_slot` marks the slots whose bytes are pure per-node DATA
    (`d_tree`, `d_tsalt`, `d_nb`, `d_nc`) as opposed to the loop's CONTROL
    and ADDRESSING state (`d_items`, `d_wl`, `d_blk_base`). It exists for
    the sabotage arm alone -- the shipped compare-and-skip treats every
    slot identically.

    `-D MOJOLEARN_ET_SAB_STAGE_SKIP_ALWAYS=1` (a measurement arm, never a
    gate) skips every re-upload after the first FOR THE PAYLOAD SLOTS ONLY:
    the merged forest freezes its first batch's tree ids and tie salts, so
    `device_batched_check`'s merged-vs-serial arms must go RED -- the
    check's `trees_mutually_differ >= 2` fixture guard exists precisely so
    a frozen tree id cannot hide. A REQUIRED-RED ARM MUST PROVABLY
    TERMINATE, and the first version of this arm did not: it froze all
    seven slots, and frozen `d_items`/`d_wl` are the batch's control state
    -- a later cycle's plan can have MORE workload blocks than the frozen
    prefix, at which point `d_wl` hands the kernels garbage entries (the
    first upload sends the pinned buffer's uninitialized tail) whose
    `node_id`s index `d_items` out of bounds; the 2026-09-01 gate run hung
    past 10 minutes with no output. Frozen `d_blk_base` has the same
    addressing hazard (stale bases plus live block counts can write
    `blk_off` out of bounds). So those three stay LIVE under the define:
    every device loop bound and every address derives from live control
    slots, and the arm terminates by the same argument as the clean run,
    while the frozen draw keys still move every tree after the batch-first
    one. Frozen `d_nb`/`d_nc` are in-bounds by construction (`i * k_old +
    k_old <= nodes * k_cap`), so they may freeze safely.
    """
    var sp = src.unsafe_ptr()
    var hp = shadow.unsafe_ptr()
    if seen_before:
        comptime if is_defined["MOJOLEARN_ET_SAB_STAGE_SKIP_ALWAYS"]():
            if payload_slot:
                return
        var same = True
        for i in range(n):
            if hp.unsafe_load(i) != sp.unsafe_load(i):
                same = False
                break
        if same:
            return
    for i in range(n):
        hp.unsafe_store(i, sp.unsafe_load(i))
    ctx.enqueue_copy(dst_buf=dst, src_ptr=src.unsafe_ptr())


def stage_batch(
    ctx: DeviceContext,
    mut ws: LevelWorkspace,
    work_items: List[NodeWorkItem],
    item_trees: List[Int32],
    plan: WorkloadPlan,
    k: Int,
) raises:
    """Put one batch's work items and workload map on the device.

    Extracted so it can run TWICE per level: once for the batch itself, and
    again after DEVIATION 205's rescue has pointed the same buffers at a
    SUB-batch. The partition reads `d_items` and `d_wl`, so a rescue that left
    the sub-batch there would partition the wrong ranges -- which is a silent
    wrong answer, not a crash, and is exactly the kind of thing an extracted
    function makes impossible to forget.
    """
    comptime TPB = DEVICE_TPB
    var n_nodes = len(work_items)
    ref h_items = ws.h_items
    ref h_wl = ws.h_wl
    ref h_nb = ws.h_nb
    ref h_nc = ws.h_nc

    if len(item_trees) != n_nodes:
        raise Error(
            "stage_batch: "
            + String(n_nodes)
            + " work items but "
            + String(len(item_trees))
            + " tree ids"
        )
    var items_ptr = h_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem]()
    for i in range(n_nodes):
        items_ptr[unsafe_offset=i] = work_items[i]
        # DEVIATION 211: the per-item tree id rides with the item.
        ws.h_tree.unsafe_ptr().unsafe_store(i, item_trees[i])
        # DEVIATION 463: the tie-break rank salt rides with it too, keyed on
        # the SAME (tree, node) the host oracle keys on.
        ws.h_tsalt.unsafe_ptr().unsafe_store(
            i,
            split_tie_salt_for(
                UInt32(Int(item_trees[i])), UInt32(Int(work_items[i].idx))
            ),
        )
    var wl_ptr = h_wl.unsafe_ptr().unsafe_bitcast[WorkloadInfo]()
    for i in range(plan.n_blocks_dimx):
        wl_ptr[unsafe_offset=i] = plan.info[i]
    var base_acc = 0
    for i in range(n_nodes):
        h_nb.unsafe_ptr().unsafe_store(i, Int32(i * Int(k)))
        h_nc.unsafe_ptr().unsafe_store(i, Int32(Int(k)))
        # Where node `i`'s blocks start in the flattened workload array.
        # `build_workload_info` lays them out contiguously in node order,
        # so this repeats the running sum it performs -- deviation 203's
        # scan pass needs that base and the device cannot derive it.
        ws.h_blk_base.unsafe_ptr().unsafe_store(i, Int32(base_acc))
        var nb_i = ceildiv(Int(work_items[i].instances.count), TPB)
        if nb_i < 1:
            nb_i = 1
        base_acc += nb_i

    # DEVIATION 472: each of the seven copies is byte-compared against the
    # bytes LAST ENQUEUED for its slot and skipped on equality (fail-safe
    # in DEVIATION 1917's direction: a spurious mismatch costs one copy, a
    # changed value is never skipped -- see `_stage_upload_if_changed`).
    # No slot is special-cased: `d_nb`/`d_nc` are byte-constant across a
    # group at fixed `k` (staged as `i * k` and `k`, never reading the
    # work items) and `d_tree` is constant while the frontier composition
    # is stable, so those collapse to one copy per group BY the compare,
    # not by bookkeeping; the rescue's `k` change restages them the same
    # way. The no-retry restage skip at the level loop's retry check
    # (DEVIATION 455's `elif len(retry) > 0`) is a different mechanism and
    # stays where it is.
    # `payload_slot` (the last argument) feeds ONLY the skip-always
    # sabotage arm: True for the pure per-node data slots, False for the
    # control/addressing slots the arm must keep live to terminate -- see
    # `_stage_upload_if_changed`. The shipped path ignores it.
    var seen = ws.stage_valid
    _stage_upload_if_changed(
        ctx,
        ws.d_items,
        ws.h_items,
        ws.s_items,
        ws.cap_nodes * size_of[NodeWorkItem](),
        seen,
        False,
    )
    _stage_upload_if_changed(
        ctx, ws.d_tree, ws.h_tree, ws.s_tree, ws.cap_nodes, seen, True
    )
    _stage_upload_if_changed(
        ctx, ws.d_tsalt, ws.h_tsalt, ws.s_tsalt, ws.cap_nodes, seen, True
    )
    _stage_upload_if_changed(
        ctx,
        ws.d_wl,
        ws.h_wl,
        ws.s_wl,
        ws.cap_blocks * size_of[WorkloadInfo](),
        seen,
        False,
    )
    _stage_upload_if_changed(
        ctx, ws.d_nb, ws.h_nb, ws.s_nb, ws.cap_nodes, seen, True
    )
    _stage_upload_if_changed(
        ctx, ws.d_nc, ws.h_nc, ws.s_nc, ws.cap_nodes, seen, True
    )
    _stage_upload_if_changed(
        ctx,
        ws.d_blk_base,
        ws.h_blk_base,
        ws.s_blk_base,
        ws.cap_nodes,
        seen,
        False,
    )
    ws.stage_valid = True
    # DEVIATION 450: no trailing synchronize. The copies above are queue-
    # ordered ahead of every kernel that reads their destinations, and the
    # `h_*` staging they read from is not rewritten until after the next
    # REQUIRED drain (the reduce readback, or the survey's) -- so the only
    # thing a sync here bought was one more per-cycle stall. cuML's
    # `doSplit` enqueues its `update_device` calls the same way and drains
    # ONCE, at `handle.sync_stream` (`builder.cuh:492-494`).

def search_batch(
    ctx: DeviceContext,
    mut ws: LevelWorkspace,
    mut dataset: DeviceDataset,
    mut d_row_ids: DeviceBuffer[DType.int32],
    work_items: List[NodeWorkItem],
    k: Int,
    params: DecisionTreeParams,
    n_classes: Int32,
    n_rows: Int32,
    n_cols: Int32,
    item_trees: List[Int32],
    seed: UInt64,
    use_sampler: Bool,
    host_colids: List[Int32],
    range_only: Bool,
    mut clock: PhaseClock,
) raises -> Tuple[
    List[Split], List[Int32], List[Float32], List[Float32], List[Int32]
]:
    """One batch through the split search: steps 2 to 8 of `doSplit`.

    DEVIATION 211: `item_trees` carries one tree id PER WORK ITEM, because
    the forest trainer merges every in-flight tree's frontier into one batch.
    Every draw was already keyed by `(seed, tree, node, col)`; the only thing
    that changed is where the tree component comes from.

    Extracted from the level loop so DEVIATION 205's rescue can run the SAME
    passes on a sub-batch instead of a second copy of the launch code. A copy
    drifts from its constant; this is the one copy.

    `use_sampler` selects deviation 201's device sampler (the normal path) or
    an upload of `host_colids` (the rescue, whose column the host chose).
    `range_only` returns after the range pass with the cells, which is the
    survey the rescue needs and nothing more.

    Returns `(splits, any_nonconstant_per_node, min, max, n_missing)`. The
    ranges are empty unless `range_only`.
    """
    comptime TPB = DEVICE_TPB
    comptime MAX_ACC = DEVICE_MAX_ACC
    var n_nodes = len(work_items)
    if n_nodes == 0:
        # DEVIATION 466: a best-first cycle can have NOTHING to search --
        # every node popped last cycle had two unexpandable children -- and
        # still have nodes left to pop. An empty batch is a well-formed
        # request for no work, not an error. The depth-wise loop breaks
        # before it can ever ask, so this arm belongs to best-first alone.
        return (
            List[Split](),
            List[Int32](),
            List[Float32](),
            List[Float32](),
            List[Int32](),
        )
    # --- 2. the ragged-batch flattening ------------------------------
    #
    # ==================================================================
    # DEVIATION BLOCK 2020 -- the search passes' tile is
    # `TPB * SEARCH_ROWS_PER_THREAD` rows per block, cuML's exact shape at
    # the shipped R = 1 and LightGBM's multi-row threads under the arms
    #
    # THEIRS (cuML). `updateWorkloadInfo` gives every block exactly TPB
    # rows (`builder.cuh:365-385` at the pin; v26.08.00 `builder.cuh:
    # 393-408` is unchanged), so the kernels' block-strided row loop
    # degenerates to at most ONE iteration per thread, and the two hot
    # passes launch `sum_i ceildiv(count_i, TPB) * k` workgroups per
    # cycle. DEVIATION 1943 measured what that costs where dispatch rate
    # is the bound: on a 64-lane device, halving the workgroup count
    # halved each pass, twice.
    #
    # THEIRS (LightGBM, the incumbent GPU precedent for the arm). Their
    # histogram kernel derives a REAL per-thread row count from the grid
    # (`cuda_histogram_constructor.cu:30-32`), sized by
    # `NUM_DATA_PER_THREAD = 400` (`cuda_histogram_constructor.hpp:22`)
    # with an occupancy FLOOR of 160 y-blocks so small leaves cannot
    # starve the device. Ours keeps its floor by construction: a node
    # never drops below one block per (node, feature), so R only merges
    # blocks that were siblings of the SAME node.
    #
    # OURS. The tile passed to `build_workload_info` -- and ONLY here and
    # in the regression twin; the partition keeps its own TPB tiling,
    # which DEVIATION 203's scatter assumes -- is widened by the comptime
    # `SEARCH_ROWS_PER_THREAD` (default 1: this line compiles the exact
    # pre-2020 program). BECAUSE the two plans now differ at R > 1, the
    # level loop's plain-cycle restage skip is no longer sound and the
    # partition restages `d_wl` -- the 2026-09-01 red round's finding;
    # see the DEVIATION 2020 restage leg at the `elif len(retry) > 0 or
    # SEARCH_ROWS_PER_THREAD > 1` sites and the ledger's RED ROUND
    # section. The kernels need no coverage change: their loop
    # was ALREADY `for i in tid, tid + TPB*num_blocks, ...`, complete and
    # disjoint at any block count. The BITS do not move because nothing
    # order-bearing is regrouped: the score pass sums integers through
    # atomics (135/171), the cross-block range merge is integer min/max
    # (204), the block fold is in key space under IDENTICAL (452), and
    # the one place a real per-thread fold appears at R > 1 -- the range
    # kernel's local min/max -- runs in key order under the flag (the
    # DEVIATION 2020 fold leg in `node_feature_range_kernel`), which is
    # grouping-free by total order and value-identical to R = 1.
    # Required-RED arm: `-D MOJOLEARN_ET_SAB_RPT_TAIL_DROP` (see
    # `SEARCH_SAB_RPT_TAIL_DROP`); it is a no-op at R = 1, which is
    # itself the witness that the arm targets the tiling.
    #
    # WHAT IT BUYS AND WHERE: workgroup count of the two hot launches
    # divides by R at the shallow levels (where DEVIATION 1943's
    # measured bound lives), and each surviving block amortizes its six
    # collectives and 3-7 publish atomics over R times the rows. The
    # gathers themselves do not shrink -- DEVIATION 212's lesson says to
    # presume the Apple column a wash and let the A/B say otherwise.
    # UNVERIFIED, RUN OWED: the A/B commands live in DEVIATIONS.md 2020
    # and PLAN.md.
    # ==================================================================
    var plan = build_workload_info(
        work_items, TPB * SEARCH_ROWS_PER_THREAD
    )
    var n_cells = n_nodes * Int(k)

    # --- per-batch device buffers ------------------------------------
    ref d_min = ws.d_min
    ref d_max = ws.d_max
    ref d_missing = ws.d_missing
    ref d_merges = ws.d_merges
    ref d_minkey = ws.d_minkey
    ref d_maxkey = ws.d_maxkey
    ref d_nleft = ws.d_nleft
    ref d_ntotal = ws.d_ntotal
    ref d_accl = ws.d_accl
    ref d_acct = ws.d_acct
    ref d_nblocks = ws.d_nblocks
    ref d_status = ws.d_status
    ref d_thresh = ws.d_thresh
    ref d_gnum = ws.d_gnum
    ref d_gden = ws.d_gden
    ref c_q = ws.c_q
    ref c_c = ws.c_c
    ref c_m = ws.c_m
    ref c_l = ws.c_l
    ref c_nu = ws.c_nu
    ref c_de = ws.c_de
    ref c_v = ws.c_v
    ref r_q = ws.r_q
    ref r_c = ws.r_c
    ref r_m = ws.r_m
    ref r_l = ws.r_l
    ref r_nu = ws.r_nu
    ref r_de = ws.r_de
    ref r_v = ws.r_v
    ref r_mg = ws.r_mg
    ref r_nw = ws.r_nw
    ref r_mx = ws.r_mx
    ref d_nb = ws.d_nb
    ref d_nc = ws.d_nc
    ref d_colids = ws.d_colids
    ref d_samp_scratch = ws.d_samp_scratch
    ref d_samp_report = ws.d_samp_report
    ref d_items = ws.d_items
    ref d_wl = ws.d_wl

    # One host staging buffer per copy: they are asynchronous, and a
    # shared one would be rewritten under an in-flight copy. The items and
    # workload staging moved into `stage_batch`; only `h_colids` is still
    # written here (the rescue's host-chosen columns).
    #
    # DEVIATION 450: no entry synchronize. Every path that enqueued a copy
    # READING an `h_*` staging buffer drained before returning to the
    # caller (the reduce readback's sync, or the survey's), so no such
    # copy can be in flight when this call rewrites the staging; the
    # workspace constructor's own drain covers the first write ever.
    ref h_colids = ws.h_colids

    stage_batch(ctx, ws, work_items, item_trees, plan, Int(k))

    # =================================================================
    # DEVIATION 470 -- TWO fused seeder launches replace this cycle's
    # SIX setup enqueues: half A carries the `d_samp_report` memset,
    # range init and the `d_nonconst` memset; half B the score init,
    # the `r_mx` memset and reduce init. TWO, NOT ONE: the one-kernel
    # form's 27 pointers + 7 scalars overran Metal's 31-entry binding
    # table (MAX's ABI binds scalars too) and died in the backend with
    # no source location -- do not re-fuse them. Hoisted HERE, after
    # `stage_batch` and before the sampler: all six are pure write-only
    # seeders over disjoint buffers, and nothing enqueued between each
    # one's old position and its first reader writes any of the seeded
    # buffers, so on the in-order queue the positions are equivalent to
    # every reader (the hoist is bit-inert). The capacity extents
    # (`cap_nodes`, `cap_report`) are exactly what the three memsets
    # covered -- seeding only the live batch leaves stale cells a later
    # larger batch reads (ensemble's 1916 lesson). A ZERO extent skips
    # a region: the rescue (`not use_sampler`) seeds no report, and the
    # survey (`range_only`) skips half B outright -- every B extent
    # would be zero, exactly its old behavior.
    # THE REFUSALS, restated from the scoping pass: NO seeder fuses into
    # its CONSUMER -- `node_nonconstant_flag_kernel` does cross-block
    # `Atomic.fetch_add`, and the range/score/reduce kernels accumulate
    # grid-wide into their seeded cells; Metal has no grid sync, so a
    # seed folded into any consumer could be read before every block
    # wrote it. The seeders fuse with each other and with nothing else.
    # Under the clock, the six seeders now bill to PHASE_STAGE instead
    # of their old phases; the timed program was always the serialized
    # one, and no seeded value moves.
    # =================================================================
    var setup_report = Int32(ws.cap_report) if use_sampler else Int32(0)
    var setup_a_extent = n_cells
    if Int(setup_report) > setup_a_extent:
        setup_a_extent = Int(setup_report)
    if ws.cap_nodes > setup_a_extent:
        setup_a_extent = ws.cap_nodes
    ctx.enqueue_function[phase_setup_a_kernel](
        d_samp_report.unsafe_ptr(),
        setup_report,
        d_min.unsafe_ptr(),
        d_max.unsafe_ptr(),
        d_minkey.unsafe_ptr(),
        d_maxkey.unsafe_ptr(),
        d_missing.unsafe_ptr(),
        d_merges.unsafe_ptr(),
        Int32(n_cells),
        ws.d_nonconst.unsafe_ptr(),
        Int32(ws.cap_nodes),
        grid_dim=ceildiv(setup_a_extent, PHASE_SETUP_TPB),
        block_dim=PHASE_SETUP_TPB,
    )
    if not range_only:
        var setup_acc = n_cells * Int(n_classes)
        var setup_b_extent = setup_acc
        if ws.cap_nodes > setup_b_extent:
            setup_b_extent = ws.cap_nodes
        ctx.enqueue_function[phase_setup_b_kernel](
            d_status.unsafe_ptr(),
            d_thresh.unsafe_ptr(),
            d_nleft.unsafe_ptr(),
            d_ntotal.unsafe_ptr(),
            d_gnum.unsafe_ptr(),
            d_gden.unsafe_ptr(),
            d_nblocks.unsafe_ptr(),
            d_accl.unsafe_ptr(),
            d_acct.unsafe_ptr(),
            Int32(n_cells),
            Int32(setup_acc),
            r_mx.unsafe_ptr(),
            Int32(ws.cap_nodes),
            r_q.unsafe_ptr(),
            r_c.unsafe_ptr(),
            r_m.unsafe_ptr(),
            r_l.unsafe_ptr(),
            r_nu.unsafe_ptr(),
            r_de.unsafe_ptr(),
            r_v.unsafe_ptr(),
            r_mg.unsafe_ptr(),
            r_nw.unsafe_ptr(),
            Int32(n_nodes),
            grid_dim=ceildiv(setup_b_extent, PHASE_SETUP_TPB),
            block_dim=PHASE_SETUP_TPB,
        )

    # --- 3. the range pass -------------------------------------------
    # --- feature sampling, WHERE cuML DOES IT (deviation 201), unless the
    # caller already chose the columns. DEVIATION 205's rescue does: its
    # column comes from the host, which is the only place the survey's cells
    # were read.
    if not use_sampler:
        if len(host_colids) != n_cells:
            raise Error(
                "host_colids must be n_nodes * k long; got "
                + String(len(host_colids))
                + " for "
                + String(n_cells)
            )
        for i in range(n_cells):
            h_colids.unsafe_ptr().unsafe_store(i, host_colids[i])
        # DEVIATION 450: no synchronize after this copy -- queue-ordered
        # ahead of the kernels reading `d_colids`, and `h_colids` is next
        # rewritten only after a required drain.
        ctx.enqueue_copy(dst_buf=d_colids, src_ptr=h_colids.unsafe_ptr())
    if use_sampler:
        # --- feature sampling, WHERE cuML DOES IT (deviation 201) --------
        # `d_report` is the DEVICE's own statement of which kernel ran;
        # DEVIATION 470's half-A seeder staged `SAMPLER_UNVISITED` into it
        # above, a value no kernel can produce.
        _ = sample_features_for_device(
            ctx,
            d_colids,
            d_samp_scratch.unsafe_ptr(),
            d_samp_report.unsafe_ptr(),
            d_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem](),
            ws.d_tree.unsafe_ptr(),
            h_colids,
            work_items,
            item_trees,
            seed,
            Int(n_cols),
            Int(k),
        )

    clock.tick(ctx, PHASE_STAGE)
    # DEVIATION 470: the range cells were seeded by fused half A above.
    ctx.enqueue_function[node_feature_range_kernel[TPB]](
        d_minkey.unsafe_ptr(),
        d_maxkey.unsafe_ptr(),
        d_missing.unsafe_ptr(),
        d_merges.unsafe_ptr(),
        dataset.d_data.unsafe_ptr(),
        d_row_ids.unsafe_ptr(),
        d_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem](),
        d_wl.unsafe_ptr().unsafe_bitcast[WorkloadInfo](),
        d_colids.unsafe_ptr(),
        n_rows,
        n_cols,
        Int32(k),
        Int32(0),
        grid_dim=(plan.n_blocks_dimx, Int(k), 1),
        block_dim=(TPB, 1, 1),
    )
    # DEVIATION 204: the merge produced order-preserving KEYS; this
    # turns them back into the `(min, max)` floats every later pass
    # reads, and applies the empty-cell sentinel.
    ctx.enqueue_function[node_feature_range_decode_kernel](
        d_min.unsafe_ptr(),
        d_max.unsafe_ptr(),
        d_minkey.unsafe_ptr(),
        d_maxkey.unsafe_ptr(),
        Int32(n_cells),
        Int32(RANGE_SAB_NONE),
        grid_dim=ceildiv(n_cells, 64),
        block_dim=64,
    )

    # --- 3b. DID ANY SAMPLED COLUMN VARY? (DEVIATION 205) -------------
    # One Int32 per node, not the 3 * n_cells the range cells would cost.
    # DEVIATION 470: `d_nonconst` was zeroed (over full capacity) by
    # fused half A above.
    ctx.enqueue_function[node_nonconstant_flag_kernel](
        ws.d_nonconst.unsafe_ptr(),
        d_min.unsafe_ptr(),
        d_max.unsafe_ptr(),
        d_missing.unsafe_ptr(),
        ws.d_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem](),
        Int32(n_cells),
        Int32(k),
        grid_dim=ceildiv(n_cells, 64),
        block_dim=64,
    )
    ctx.enqueue_copy(dst_buf=ws.h_nonconst, src_buf=ws.d_nonconst)
    # DEVIATION 450: the range pass drains only when the caller wants the
    # survey NOW. On the full path `h_nonconst` is not READ until after
    # the reduce readback's sync below, so its copy rides the queue and
    # this pass loses its per-cycle stall -- cuML's `doSplit` shape, which
    # drains ONCE per batch. `clock.tick` still syncs when the clock is
    # ENABLED: the timed program was always the serialized one.
    if range_only:
        ctx.enqueue_copy(dst_buf=ws.o_rmin, src_buf=d_min)
        ctx.enqueue_copy(dst_buf=ws.o_rmax, src_buf=d_max)
        ctx.enqueue_copy(dst_buf=ws.o_rmiss, src_buf=d_missing)
        ctx.synchronize()
    clock.tick(ctx, PHASE_RANGE)

    if range_only:
        # THE SURVEY. The rescue needs the cells themselves so the host can
        # pick a column; nothing else does, so nothing else pays for them.
        var any_nonconst = List[Int32](length=n_nodes, fill=Int32(0))
        for i in range(n_nodes):
            any_nonconst[i] = ws.h_nonconst.unsafe_ptr()[unsafe_offset=i]
        var rmin = List[Float32](length=n_cells, fill=Float32(0.0))
        var rmax = List[Float32](length=n_cells, fill=Float32(0.0))
        var rmiss = List[Int32](length=n_cells, fill=Int32(0))
        for i in range(n_cells):
            rmin[i] = ws.o_rmin.unsafe_ptr()[unsafe_offset=i]
            rmax[i] = ws.o_rmax.unsafe_ptr()[unsafe_offset=i]
            rmiss[i] = ws.o_rmiss.unsafe_ptr()[unsafe_offset=i]
        return (List[Split](), any_nonconst^, rmin^, rmax^, rmiss^)

    # --- 4. the draw and score pass ----------------------------------
    # DEVIATION 470: the score cells and class accumulators were seeded
    # by fused half B above (the survey skips half B and never gets here).
    ctx.enqueue_function[
        node_feature_score_kernel[TPB, MAX_ACC, True]
    ](
        d_nleft.unsafe_ptr(),
        d_ntotal.unsafe_ptr(),
        d_accl.unsafe_ptr(),
        d_acct.unsafe_ptr(),
        d_nblocks.unsafe_ptr(),
        d_min.unsafe_ptr(),
        d_max.unsafe_ptr(),
        d_missing.unsafe_ptr(),
        dataset.d_data.unsafe_ptr(),
        d_row_ids.unsafe_ptr(),
        dataset.d_labels.unsafe_ptr(),
        d_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem](),
        d_wl.unsafe_ptr().unsafe_bitcast[WorkloadInfo](),
        d_colids.unsafe_ptr(),
        ws.d_tree.unsafe_ptr(),
        n_rows,
        Int32(k),
        n_classes,
        seed,
        Int32(0),
        grid_dim=(plan.n_blocks_dimx, Int(k), 1),
        block_dim=(TPB, 1, 1),
    )
    ctx.enqueue_function[
        node_feature_score_finalize_kernel[MAX_ACC, True]
    ](
        d_status.unsafe_ptr(),
        d_thresh.unsafe_ptr(),
        d_gnum.unsafe_ptr(),
        d_gden.unsafe_ptr(),
        d_nleft.unsafe_ptr(),
        d_ntotal.unsafe_ptr(),
        d_accl.unsafe_ptr(),
        d_acct.unsafe_ptr(),
        d_min.unsafe_ptr(),
        d_max.unsafe_ptr(),
        d_missing.unsafe_ptr(),
        d_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem](),
        d_colids.unsafe_ptr(),
        ws.d_tree.unsafe_ptr(),
        Int32(n_cells),
        Int32(k),
        n_classes,
        seed,
        params.min_samples_leaf,
        Int32(0),
        grid_dim=ceildiv(n_cells, 64),
        block_dim=64,
    )

    # --- 5. scored cells into candidates (DEVIATION 182) -------------
    clock.tick(ctx, PHASE_SCORE)
    ctx.enqueue_function[score_to_candidate_kernel](
        c_q.unsafe_ptr(),
        c_c.unsafe_ptr(),
        c_m.unsafe_ptr(),
        c_l.unsafe_ptr(),
        c_nu.unsafe_ptr(),
        c_de.unsafe_ptr(),
        c_v.unsafe_ptr(),
        d_status.unsafe_ptr(),
        d_thresh.unsafe_ptr(),
        d_nleft.unsafe_ptr(),
        d_ntotal.unsafe_ptr(),
        d_accl.unsafe_ptr(),
        d_acct.unsafe_ptr(),
        d_gnum.unsafe_ptr(),
        d_gden.unsafe_ptr(),
        d_colids.unsafe_ptr(),
        Int32(n_cells),
        n_classes,
        params.min_samples_leaf,
        params.split_criterion,
        grid_dim=ceildiv(n_cells, 64),
        block_dim=64,
    )

    # --- 6. evalBestSplit's reduction ---------------------------------
    # DEVIATION 470: the reduce cells and the `r_mx` mutexes (over full
    # capacity) were seeded by fused half B above.
    var bpn = ceildiv(Int(k), TPB)
    if bpn < 1:
        bpn = 1
    ctx.enqueue_function[split_reduce_kernel[TPB]](
        r_q.unsafe_ptr(),
        r_c.unsafe_ptr(),
        r_m.unsafe_ptr(),
        r_l.unsafe_ptr(),
        r_nu.unsafe_ptr(),
        r_de.unsafe_ptr(),
        r_v.unsafe_ptr(),
        r_mg.unsafe_ptr(),
        r_nw.unsafe_ptr(),
        r_mx.unsafe_ptr(),
        c_q.unsafe_ptr(),
        c_c.unsafe_ptr(),
        c_m.unsafe_ptr(),
        c_l.unsafe_ptr(),
        c_nu.unsafe_ptr(),
        c_de.unsafe_ptr(),
        c_v.unsafe_ptr(),
        d_nb.unsafe_ptr(),
        d_nc.unsafe_ptr(),
        ws.d_tsalt.unsafe_ptr(),
        Int32(bpn),
        Int32(0),
        grid_dim=(bpn, n_nodes, 1),
        block_dim=(TPB, 1, 1),
    )
    # DEVIATION 463: the exact-tie counter, only when the build asks for it.
    comptime if is_defined["MOJOLEARN_ET_TIE_STATS"]():
        ctx.enqueue_function[split_tie_count_kernel](
            ws.d_ties.unsafe_ptr(),
            r_c.unsafe_ptr(),
            r_nu.unsafe_ptr(),
            r_de.unsafe_ptr(),
            r_v.unsafe_ptr(),
            c_nu.unsafe_ptr(),
            c_de.unsafe_ptr(),
            c_v.unsafe_ptr(),
            d_nb.unsafe_ptr(),
            d_nc.unsafe_ptr(),
            Int32(n_nodes),
            grid_dim=ceildiv(n_nodes, 64),
            block_dim=64,
        )
        ctx.enqueue_copy(dst_buf=ws.o_ties, src_buf=ws.d_ties)

    # --- 7. the splits come back to the host, as `:492-494` does ------
    # ONLY THE SPLITS CROSS, which is exactly what
    # `raft::update_host(h_splits, splits, work_items.size())` copies back
    # at `builder.cuh:492-494`. The gain travels WITH the candidate now
    # (DEVIATION 183, second form), so no per-level readback of the node
    # totals is needed and none happens.
    ref o_q = ws.o_q
    ref o_c = ws.o_c
    ref o_l = ws.o_l
    ref o_nu = ws.o_nu
    ref o_de = ws.o_de
    ref o_v = ws.o_v
    ref o_m = ws.o_m
    ctx.enqueue_copy(dst_buf=o_q, src_buf=r_q)
    ctx.enqueue_copy(dst_buf=o_c, src_buf=r_c)
    ctx.enqueue_copy(dst_buf=o_l, src_buf=r_l)
    ctx.enqueue_copy(dst_buf=o_nu, src_buf=r_nu)
    ctx.enqueue_copy(dst_buf=o_de, src_buf=r_de)
    ctx.enqueue_copy(dst_buf=o_v, src_buf=r_v)
    ctx.enqueue_copy(dst_buf=o_m, src_buf=r_m)
    ctx.synchronize()
    clock.tick(ctx, PHASE_REDUCE)

    # DEVIATION 450: the deferred `h_nonconst` read. Its copy was enqueued
    # in the range pass; the sync above is the batch's ONE drain, so the
    # values are complete here and nowhere earlier did the host need them.
    var any_nonconst = List[Int32](length=n_nodes, fill=Int32(0))
    for i in range(n_nodes):
        any_nonconst[i] = ws.h_nonconst.unsafe_ptr()[unsafe_offset=i]

    # --- 8. build the batch's splits on the host, as `:492-494` does --
    var splits = List[Split]()
    for i in range(n_nodes):
        var colid = o_c.unsafe_ptr()[unsafe_offset=i]
        var n_left = o_l.unsafe_ptr()[unsafe_offset=i]
        # The winning candidate's gain came off the DEVICE with it --
        # `r_m`, written by `score_to_candidate_kernel` and carried
        # through the reduction. `split_not_valid` below is unchanged.
        var metric = o_m.unsafe_ptr()[unsafe_offset=i]
        if o_v.unsafe_ptr()[unsafe_offset=i] == 0 or colid < 0:
            metric = Float32.MIN_FINITE
        splits.append(
            Split(
                o_q.unsafe_ptr()[unsafe_offset=i], colid, metric, n_left
            )
        )

    # DEVIATION 463: report the batch's exact-tie tally. The counter cells
    # rode the queue with the reduce readback, so the sync above completed
    # them; one line per batch, summed by whoever asked for the define.
    comptime if is_defined["MOJOLEARN_ET_TIE_STATS"]():
        var tie_decided = 0
        var tie_tied = 0
        for i in range(n_nodes):
            if o_c.unsafe_ptr()[unsafe_offset=i] >= 0:
                tie_decided += 1
                if ws.o_ties.unsafe_ptr()[unsafe_offset=i] >= Int32(2):
                    tie_tied += 1
        print("ET_TIE_STATS batch decided=", tie_decided, " tied=", tie_tied)

    clock.tick(ctx, PHASE_HOST_SPLITS)
    return (
        splits^,
        any_nonconst^,
        List[Float32](),
        List[Float32](),
        List[Int32](),
    )


def train_forest_classification_device(
    ctx: DeviceContext,
    mut dataset: DeviceDataset,
    params: DecisionTreeParams,
    tree_ids: List[Int32],
    seed: UInt64,
    sabotage: Int32 = FOREST_SAB_NONE,
    row_slot_cap: Int = FOREST_ROW_SLOT_CAP,
    bootstrap: Bool = False,
    n_sampled_rows: Int32 = 0,
    bf_sabotage: Int32 = BESTFIRST_SAB_NONE,
) raises -> List[TreeMetaDataNode[DType.float32]]:
    """The shipping entry point: an INERT clock, so no synchronize is ever
    added -- see `PhaseClock`. `_timed` below is the same function with the
    micro-step clock threaded; `bench/fit_once.mojo` calls it directly.
    `MOJOLEARN_STAGE_TIMES=1` (read once, here) enables the clock and prints
    stage -> seconds at fit end -- see `STAGE_TIMES_ENV`.

    `bootstrap` / `n_sampled_rows` (DEVIATION 460): with `bootstrap` each
    tree's row slot is a with-replacement sample of `n_sampled_rows` rows
    (0 = `n_rows`) drawn by `fill_row_slots`; without it the slot is the
    identity permutation and `n_sampled_rows` must be 0 or `n_rows`."""
    var clock = PhaseClock(stage_times_enabled())
    var out = train_forest_classification_device_timed(
        ctx, dataset, params, tree_ids, seed, sabotage, row_slot_cap, clock,
        bootstrap, n_sampled_rows, bf_sabotage,
    )
    # DEVIATION 2002: on a dead/saturated device (the 134 loaded window's
    # Metal context death) this loop's per-cycle split readbacks deliver
    # stale zeros, every node quietly becomes a leaf, and the fit returns
    # a well-formed forest of stumps with no error. The end-of-fit canary
    # (`core/device_liveness.mojo`) raises instead. One drain per FOREST
    # fit; `_timed` direct callers (bench, checks) are check-tier and
    # uncovered on purpose.
    assert_device_alive(ctx, "extratrees classification forest fit")
    print_stage_times(clock, "extratrees classification forest fit")
    return out^


def train_forest_classification_device_timed(
    ctx: DeviceContext,
    mut dataset: DeviceDataset,
    params: DecisionTreeParams,
    tree_ids: List[Int32],
    seed: UInt64,
    sabotage: Int32,
    row_slot_cap: Int,
    mut clock: PhaseClock,
    bootstrap: Bool = False,
    n_sampled_rows: Int32 = 0,
    bf_sabotage: Int32 = BESTFIRST_SAB_NONE,
) raises -> List[TreeMetaDataNode[DType.float32]]:
    """Every requested ExtraTree, with ONE merged frontier driving the GPU.

    ==================================================================
    DEVIATION BLOCK 211 -- THE BATCH SPANS TREES. cuML's cross-tree
    parallelism is a CUDA stream pool; ours is a wider grid.

    THEIRS. cuML overlaps trees with `#pragma omp parallel for
    num_threads(n_streams)` over the tree loop, one CUDA stream per
    OpenMP thread (`randomforest.cuh:336-341`), n_streams=4 shipped.
    Metal has no streams (`ctx.create_stream()` is unsupported -- the
    traps register), so their mechanism cannot be transcribed.

    OURS. The frontier batch itself spans trees. A `NodeWorkItem` never
    said which tree it belonged to -- the batch's tree id was a scalar
    kernel argument -- and NOTHING ELSE in the formulation is per-tree:
    `bootstrap=False` means every tree reads the SAME resident dataset
    (deviation 184), and every draw is a pure function of
    `(seed, tree_id, node_id, feature_id)` (deviation 130). So one
    level cycle pops work from EVERY tree's queue into one batch, the
    tree id rides per item (`item_trees`, staged in `ws.d_tree`), and
    the launches, readbacks and synchronize points that ran once per
    tree per level now run once per level for the whole forest.

    WHY THE TREES CANNOT MOVE, mechanism by mechanism:
      * feature draws / threshold draws / rescue picks: keyed per
        (tree, node); an item carries its own tree id to the kernel.
      * the score accumulation: integer atomics per (node, feature)
        cell (deviation 171); cells of different trees are different
        slots of the same launch, exactly as cells of different NODES
        already were.
      * the reduction: per node, over that node's own cells.
      * the partition: range-addressed. Tree slot `s` owns rows
        `[s * n_rows, (s + 1) * n_rows)` of ONE `d_row_ids` buffer
        (`row_ids_tiled_sequence_kernel`), every `InstanceRange` of its
        queue is carved from that slot (`NodeQueue`'s `row_base`), and
        the kernels never look outside an item's range.
      * the push: per queue, FIFO order preserved -- and the batch
        width was ALREADY a scheduling parameter that must not change
        the tree (`NodeQueue.pop`'s contract).

    GATED by `device_batched_check`: the merged forest against one-tree
    builds, node for node, plus BOTH sabotages above
    (`FOREST_SAB_SCALAR_TREE`, `FOREST_SAB_SHARED_ROW_BASE`) seen to
    move the forest -- so the gate watches the two mechanisms that
    isolate the trees, not just the totals.

    THE PRICE: `2 * 4 * min(n_trees, cap) * n_rows` bytes of row-id
    buffers (`FOREST_ROW_SLOT_CAP` bounds it), against launches, host
    round-trips and synchronize points divided by the number of
    in-flight trees. The workspace is one per GROUP now, not one per
    tree -- deviation 202 taken one level further.
    ==================================================================

    Per batch, the order is `train_classification_device_resident`'s old
    body, which was `Builder::train` (`builder.cuh:344-359`) around
    `doSplit` (`:379-494`): sample features, range pass, draw-and-score,
    reduce, splits back to the host, partition, push. That function is
    now a one-tree call of this one, so there is exactly ONE copy of the
    loop.
    """
    var n_rows = dataset.n_rows
    var n_cols = dataset.n_cols
    var n_classes = dataset.n_classes
    validity_check(params)

    comptime TPB = DEVICE_TPB
    comptime MAX_ACC = DEVICE_MAX_ACC
    if Int(n_classes) > MAX_ACC:
        raise Error(
            "the device score kernel is built for at most "
            + String(MAX_ACC)
            + " classes; got "
            + String(n_classes)
            + " (DEVIATION 172: shared sizing is comptime here)"
        )
    # DEVIATION 218: the 2^21-row refusal that stood here (DEVIATION 175,
    # first bound by a real request on full higgs) is LIFTED. Above
    # `SCORE_MAX_ROWS_EXACT` the published Gini pair is node-uniformly
    # SHIFTED (`classification_key_shift`) instead of refused --
    # `score_row_bound_ok` remains the exactness boundary's statement, and
    # everything at or under it is bit-for-bit unchanged.

    # --- identity trace (`core/identity_trace.mojo`) -- NOT A PORT --------
    # Stage checkpoints so a cross-backend bit difference has an ADDRESS.
    # `MOJOLEARN_IDENTITY_TRACE` is read ONCE, here, at fit entry; unset
    # (the shipping state) every `record_*` returns on one boolean test.
    # That file's four rules govern every checkpoint below -- in
    # particular rule 4: a traced run drains the queue per record and is
    # NEVER a timing.
    var trace = IdentityTrace()
    if trace.enabled:
        trace.header(
            String("extratrees.classification.forest n_rows=")
            + String(n_rows)
            + " n_cols="
            + String(n_cols)
            + " n_classes="
            + String(n_classes)
            + " n_trees="
            + String(len(tree_ids))
            + " seed="
            + String(seed)
        )
        # The post-upload boundary. ET fits the RAW resident matrix -- this
        # port has no quantile/binning stage, so deviation 184's resident
        # dataset is the corresponding stage output.
        trace.record_device(ctx, "dataset.data", dataset.d_data)
        trace.record_device(ctx, "dataset.labels", dataset.d_labels)

    var k = n_sampled_cols_for(params, n_cols)
    # DEVIATION 466: the growth mode, read ONCE per fit. Every branch on it
    # below is `if bestfirst:` with the depth-wise arm textually unchanged,
    # so a fit that did not ask for best-first executes the same statements
    # in the same order as it did before this mode existed.
    var bestfirst = params.max_leaf_nodes != -1
    if bestfirst and params.max_batch_size < 2:
        # DEVIATION 469's one hard edge, refused BY NAME rather than
        # silently overrunning the workspace. A best-first cycle searches
        # the TWO children of each expanded node, so the narrowest batch the
        # mode can run in is two; the group clamp below can cap `g` at one
        # tree but it cannot cap it below that. `max_batch_size` defaults to
        # 4096 and is a scheduling parameter, so this is reachable only by a
        # caller who set it to 1 on purpose.
        raise Error(
            "max_leaf_nodes needs max_batch_size >= 2: a best-first cycle"
            " searches both children of the node it expands, and a batch of"
            " one cannot hold them (DEVIATION 469). Got max_batch_size "
            + String(params.max_batch_size)
        )
    var out = List[TreeMetaDataNode[DType.float32]]()
    # `row_slot_cap` defaults to FOREST_ROW_SLOT_CAP; the batched check
    # passes a tiny cap to REACH the multi-group path at fixture sizes.
    # DEVIATION 460: the per-tree row SLOT is `n_sampled_rows` wide --
    # `selected_rows.size()` in `get_row_sample` -- which is `n_rows` unless
    # the caller bootstraps with sklearn's `max_samples`. The dataset stays
    # `n_rows` (M) wide; a slot's entries INDEX it.
    var slot_rows = n_rows
    if bootstrap:
        if n_sampled_rows > 0:
            slot_rows = n_sampled_rows
    elif n_sampled_rows != 0 and n_sampled_rows != n_rows:
        raise Error(
            "n_sampled_rows="
            + String(n_sampled_rows)
            + " without bootstrap: the identity permutation is n_rows wide"
            " (randomforest.cuh:69)"
        )
    var group_cap = row_slot_cap // Int(slot_rows)
    if group_cap < 1:
        group_cap = 1
    clock.mark(ctx)

    var gi = 0
    var first = 0
    while first < len(tree_ids):
        var g = len(tree_ids) - first
        if g > group_cap:
            g = group_cap
        if bestfirst:
            # DEVIATION 469: a best-first cycle searches at most TWO nodes
            # per in-flight tree (the children of the one node it expands),
            # so `2 * g` is the search batch's width and the workspace is
            # sized to `max_batch_size`. Capping `g` here is the only place
            # `max_batch_size` bounds anything in this mode, and it keeps
            # its contract: it is a SCHEDULING parameter, and lowering it
            # runs the same trees through narrower launches.
            var bf_cap = Int(params.max_batch_size) // 2
            if bf_cap < 1:
                bf_cap = 1
            if g > bf_cap:
                g = bf_cap
        var total_rows = g * Int(slot_rows)

        # ONE row-id buffer for the whole group, slot `s` holding tree
        # `tree_ids[first + s]`'s identity permutation (deviation 200,
        # tiled) or its bootstrap sample (DEVIATION 460). The partition
        # mutates each slot in place across levels.
        var d_row_ids = ctx.enqueue_create_buffer[DType.int32](total_rows)
        fill_row_slots(
            ctx, d_row_ids, g, slot_rows, n_rows, bootstrap, tree_ids,
            first, seed,
        )
        if trace.enabled and bootstrap:
            # DEVIATION 460: the drawn row sample is a stage a bit can move
            # at (the Philox draw, its stride, its range reduction), so it
            # is recorded BEFORE any level touches it. Algorithm position
            # only, never a machine property.
            trace.record_device(
                ctx, String("g") + String(gi) + ".bootstrap.rowids", d_row_ids
            )

        # THE WORKSPACE, ONCE PER GROUP (deviation 202, further). Its two
        # row-scaled pieces -- the workload bound and the partition's
        # alternate buffer -- are sized to the GROUP's rows.
        var ws = make_level_workspace(
            ctx,
            Int(params.max_batch_size),
            Int32(total_rows),
            n_cols,
            n_classes,
            Int(k),
            TPB,
        )

        var queues = List[NodeQueue[DType.float32]]()
        for s in range(g):
            var base = Int32(s) * slot_rows
            if sabotage == FOREST_SAB_SHARED_ROW_BASE:
                base = Int32(0)
            queues.append(
                NodeQueue[DType.float32](
                    params, slot_rows, n_classes, tree_ids[first + s], base
                )
            )
            queues[s].bf_sabotage = bf_sabotage

        clock.tick(ctx, PHASE_SETUP)
        # The trace's LEVEL-CYCLE counter: one merged frontier batch per
        # iteration. `gN.cM.` prefixes keep every tag unique within the
        # trace (the differ's alignment invariant) while naming a position
        # in the ALGORITHM -- group N, cycle M -- never a machine property.
        var cyc = 0
        # DEVIATION 466: the best-first carry between cycles -- the nodes
        # admitted-but-unsearched, and which queue each belongs to. Empty
        # and never read in depth-wise mode.
        var bf_pending = List[NodeWorkItem]()
        var bf_pending_q = List[Int]()
        if bestfirst:
            for s in range(g):
                var seeds = queues[s].bestfirst_seed()
                for i in range(len(seeds)):
                    bf_pending.append(seeds[i])
                    bf_pending_q.append(s)
        while True:
            # --- ONE merged batch across every queue with work ----------
            var work_items = List[NodeWorkItem]()
            var item_trees = List[Int32]()
            var seg_queue = List[Int]()
            var seg_start = List[Int]()
            var seg_count = List[Int]()
            # DEVIATION 466 step 1: the best-first SEARCH batch is the set
            # of nodes admitted since the last cycle -- the children the
            # last expansions created, or the roots on cycle 0 -- and not a
            # FIFO pop. `seg_*` stays empty; the best-first expansion is per
            # popped node and does not use it.
            if bestfirst:
                for i in range(len(bf_pending)):
                    work_items.append(bf_pending[i])
                    item_trees.append(tree_ids[first + bf_pending_q[i]])
            else:
                for s in range(g):
                    if len(work_items) >= Int(params.max_batch_size):
                        break
                    if not queues[s].has_work():
                        continue
                    var got = queues[s].pop_up_to(
                        Int(params.max_batch_size) - len(work_items)
                    )
                    if len(got) == 0:
                        continue
                    seg_queue.append(s)
                    seg_start.append(len(work_items))
                    seg_count.append(len(got))
                    for i in range(len(got)):
                        work_items.append(got[i])
                        item_trees.append(tree_ids[first + s])
            if len(work_items) == 0:
                # DEVIATION 466: an empty SEARCH batch does NOT end a
                # best-first fit. Both children of every node popped last
                # cycle can be unexpandable while frontiers still hold
                # splittable nodes, so the fit ends only when no tree can
                # pop -- `while not frontier.empty()` (`_tree.pyx:445`)
                # plus the budget test at `:454`.
                var bf_more = False
                if bestfirst:
                    for s in range(g):
                        if queues[s].bestfirst_can_pop():
                            bf_more = True
                if not bf_more:
                    clock.tick(ctx, PHASE_HOST_QUEUE)
                    break
            if sabotage == FOREST_SAB_SCALAR_TREE:
                for i in range(len(item_trees)):
                    item_trees[i] = item_trees[0]
            var n_nodes = len(work_items)

            clock.tick(ctx, PHASE_HOST_QUEUE)
            var found = search_batch(
                ctx,
                ws,
                dataset,
                d_row_ids,
                work_items,
                Int(k),
                params,
                n_classes,
                n_rows,
                n_cols,
                item_trees,
                seed,
                True,
                List[Int32](),
                False,
                clock,
            )
            var splits = found[0].copy()
            var any_nonconst = found[1].copy()

            var tag_pre = String("")
            if trace.enabled:
                # The tag is set whether or not there is anything to record
                # under it: DEVIATION 466's empty search batch still runs a
                # partition, and a cycle that fell back to the bare
                # "partition.rowids" tag would collide with the next one and
                # break the differ's uniqueness invariant.
                tag_pre = String("g") + String(gi) + ".c" + String(cyc) + "."
            # `n_nodes > 0`: an empty search batch has no cells to record,
            # and a zero-length record would break the differ's alignment
            # invariant rather than inform it.
            if trace.enabled and n_nodes > 0:
                # DEVIATION 454: the identity audit's hazard stages, in
                # PIPELINE order, so the cross-vendor differ bisects by
                # mechanism -- colids differ = the sampler (or its host
                # libm dispatch); ranges differ with colids equal = the
                # range fold; thresholds differ with ranges equal = the
                # draw; reduce differs with thresholds equal = score or
                # reduction. Each buffer is the batch's LOGICAL first
                # `n_cells` slots of a capacity-sized workspace buffer
                # (rule 3), recorded BEFORE deviation 205's rescue can
                # reuse the staging.
                var n_cells_t = n_nodes * Int(k)
                trace.record_device(
                    ctx, tag_pre + "colids", ws.d_colids, n_cells_t
                )
                trace.record_device(
                    ctx, tag_pre + "range.min", ws.d_min, n_cells_t
                )
                trace.record_device(
                    ctx, tag_pre + "range.max", ws.d_max, n_cells_t
                )
                trace.record_device(
                    ctx, tag_pre + "draw.thresh", ws.d_thresh, n_cells_t
                )
                # The level's REDUCED per-node winners, straight off the
                # `o_*` readback and BEFORE deviation 205's rescue can
                # reuse the staging. Rule 3: this is the logical reduced
                # buffer (one winner per node), never the per-cell
                # candidate scratch behind it.
                trace.record_host(
                    tag_pre + "reduce.colid", ws.o_c.unsafe_ptr(), n_nodes
                )
                trace.record_host(
                    tag_pre + "reduce.num", ws.o_nu.unsafe_ptr(), n_nodes
                )
                trace.record_host(
                    tag_pre + "reduce.den", ws.o_de.unsafe_ptr(), n_nodes
                )

            # --- DEVIATION 205: the nodes whose whole sample was constant.
            # Identical to the one-tree flow; the rescue key and the survey
            # simply use each item's OWN tree.
            var retry = List[Int]()
            for i in range(n_nodes):
                if any_nonconst[i] == 0 and work_items[i].instances.count > 0:
                    retry.append(i)

            if len(retry) > 0:
                var sub = List[NodeWorkItem]()
                var sub_trees = List[Int32]()
                for j in range(len(retry)):
                    sub.append(work_items[retry[j]])
                    sub_trees.append(item_trees[retry[j]])

                var ident = List[Int32](
                    length=len(sub) * Int(n_cols), fill=Int32(0)
                )
                for j in range(len(sub)):
                    for c in range(Int(n_cols)):
                        ident[j * Int(n_cols) + c] = Int32(c)
                var survey = search_batch(
                    ctx, ws, dataset, d_row_ids, sub, Int(n_cols), params,
                    n_classes, n_rows, n_cols, sub_trees, seed, False,
                    ident, True, clock,
                )

                var s_min = survey[2].copy()
                var s_max = survey[3].copy()
                var s_miss = survey[4].copy()

                var chosen_items = List[NodeWorkItem]()
                var chosen_trees = List[Int32]()
                var chosen_cols = List[Int32]()
                var chosen_slot = List[Int]()
                for j in range(len(sub)):
                    var nonconst = List[Int32]()
                    for c in range(Int(n_cols)):
                        var idx = j * Int(n_cols) + c
                        var extent = FeatureRange(
                            s_min[idx], s_max[idx], s_miss[idx]
                        )
                        if not node_feature_is_constant(
                            extent, sub[j].instances.count
                        ):
                            nonconst.append(Int32(c))
                    if len(nonconst) == 0:
                        continue
                    var u = rescue_pick(
                        rescue_key(
                            seed, sub_trees[j], UInt32(Int(sub[j].idx))
                        ),
                        len(nonconst),
                    )
                    chosen_items.append(sub[j])
                    chosen_trees.append(sub_trees[j])
                    chosen_cols.append(nonconst[u])
                    chosen_slot.append(retry[j])

                if len(chosen_items) > 0:
                    var res2 = search_batch(
                        ctx, ws, dataset, d_row_ids, chosen_items, 1, params,
                        n_classes, n_rows, n_cols, chosen_trees, seed, False,
                        chosen_cols, False, clock,
                    )
                    var rescued = res2[0].copy()
                    for t in range(len(chosen_slot)):
                        splits[chosen_slot[t]] = rescued[t]

            if trace.enabled and n_nodes > 0:
                # The SELECTED splits -- post-rescue. Under depth-wise
                # growth this is exactly what the partition and the queues
                # consume; under DEVIATION 466 it is what is ADMITTED to
                # the frontiers, and the partition consumes `part_splits`,
                # which are records admitted in an earlier cycle.
                var t_q = List[Float32]()
                var t_c = List[Int32]()
                var t_l = List[Int32]()
                var t_m = List[Float32]()
                for i in range(n_nodes):
                    t_q.append(splits[i].quesval)
                    t_c.append(splits[i].colid)
                    t_l.append(splits[i].n_left)
                    t_m.append(splits[i].best_metric_val)
                trace.record_list_f32(tag_pre + "split.thresh", t_q)
                trace.record_list_i32(tag_pre + "split.colid", t_c)
                trace.record_list_i32(tag_pre + "split.nleft", t_l)
                trace.record_list_f32(tag_pre + "split.gain", t_m)

            # --- DEVIATION 466 steps 2 and 3: ADMIT, then POP ------------
            # `part_*` is the PARTITION batch. In depth-wise mode it is the
            # search batch, item for item, which is what it always was; in
            # best-first it is the ONE node each tree pops, whose split was
            # found in an earlier cycle and is still correct because a
            # node's rows are permuted only by its own partition or an
            # ancestor's, and neither has happened while it waited.
            var part_items = List[NodeWorkItem]()
            var part_trees = List[Int32]()
            var part_splits = List[Split]()
            var part_queue = List[Int]()
            if bestfirst:
                for i in range(n_nodes):
                    _ = queues[bf_pending_q[i]].bestfirst_admit(
                        work_items[i], splits[i], item_trees[i]
                    )
                for s in range(g):
                    if not queues[s].bestfirst_can_pop():
                        continue
                    var rec = queues[s].bestfirst_pop()
                    part_items.append(rec.item)
                    part_trees.append(tree_ids[first + s])
                    part_splits.append(rec.split)
                    part_queue.append(s)
                if len(part_items) == 0:
                    # Every frontier is empty or every budget is spent.
                    clock.tick(ctx, PHASE_HOST_QUEUE)
                    break
            else:
                for i in range(n_nodes):
                    part_items.append(work_items[i])
                    part_trees.append(item_trees[i])
                    part_splits.append(splits[i])
            var n_part = len(part_items)

            var plan = build_workload_info(part_items, TPB)
            if bestfirst:
                # DEVIATION 469's synchronization price: the search batch
                # and the partition batch are DIFFERENT SETS here, so
                # `ws.h_items` is re-staged every cycle and DEVIATION 455's
                # drain runs every cycle rather than only when DEVIATION
                # 205's rescue fires.
                stage_batch(ctx, ws, part_items, part_trees, plan, Int(k))
                ctx.synchronize()
            elif len(retry) > 0 or SEARCH_ROWS_PER_THREAD > 1:
                # The sub-batches left THEIR work items on the device. The
                # partition below reads `d_items` and `d_wl`, so put this
                # batch's back.
                #
                # ==========================================================
                # DEVIATION 2020 (restage leg) -- THE DEFECT ITS FIRST CUT
                # SHIPPED, found by the 2026-09-01 gate round (every RPT>1
                # arm RED, root-up divergence). The plain no-rescue cycle
                # SKIPS this restage, and the skip's unstated premise was
                # that the partition's plan is BYTE-IDENTICAL to the plan
                # the search staged -- same work_items, same TPB tile -- so
                # `d_wl` already held it. 2020 widened the SEARCH tile to
                # `TPB * R` and broke exactly that premise: the partition's
                # grid is TPB-tiled (`plan` above, DEVIATION 203's scatter
                # contract) while `d_wl` still held the widened plan --
                # fewer entries, wrong `num_blocks`/`offset_blockid` -- so
                # every plain-path partition indexed stale WorkloadInfo and
                # corrupted `row_ids` from the first level on. So at R > 1
                # the plain cycle restages too (the comptime disjunct folds
                # away at R = 1, keeping the shipped program's exact
                # skip-and-no-drain shape). DEVIATION 472's byte-compare
                # keeps the restage cheap: on this path only `d_wl`'s bytes
                # differ, so only `d_wl` actually copies. The drain below is
                # DEVIATION 455's, and it is REQUIRED here for the same
                # measured reason: these copies land after the cycle's
                # reduce drain, and the next cycle's `stage_batch` rewrites
                # the same `h_*` staging with nothing in between to retire
                # them. That is one restage + one drain per plain cycle
                # under the arm -- a real price the A/B now measures instead
                # of a corruption the gates caught.
                # ==========================================================
                stage_batch(ctx, ws, work_items, item_trees, plan, Int(k))
                # DEVIATION 455: this re-stage's copies are the ONE set that
                # DEVIATION 450's invariant did not cover -- they are
                # enqueued AFTER the cycle's reduce drain, and the next
                # cycle's `stage_batch` rewrites the same `h_*` staging
                # with nothing in between to retire them. MEASURED as
                # run-to-run nondeterminism of the device fit on the
                # rescue-heavy fixture (rescue_check, shaped_constant_heavy:
                # device node counts 587/605 across two runs of identical
                # source). One drain, on the rescue path only (and, since
                # the 2020 restage leg, on every plain cycle at R > 1); the
                # R = 1 rescue-free cycle keeps 450's single-drain shape.
                ctx.synchronize()

            # --- the PARTITION, on the device (deviation 203) -------------
            # Range-addressed: every item's rows live in its own tree's
            # slot of `d_row_ids`, so nodes of different trees partition
            # side by side without seeing each other.
            #
            # DEVIATION 450: no synchronize before writing `h_splits`. The
            # last copy READING it was enqueued a full cycle ago and this
            # cycle's reduce readback drained the queue since; the copy
            # below is queue-ordered ahead of the partition kernels that
            # read `d_splits`.
            var splits_ptr = ws.h_splits.unsafe_ptr().unsafe_bitcast[Split]()
            for i in range(n_part):
                splits_ptr[unsafe_offset=i] = part_splits[i]
            ctx.enqueue_copy(
                dst_buf=ws.d_splits, src_ptr=ws.h_splits.unsafe_ptr()
            )
            ctx.enqueue_function[partition_count_kernel[TPB]](
                ws.d_blk_left.unsafe_ptr(),
                d_row_ids.unsafe_ptr(),
                dataset.d_data.unsafe_ptr(),
                ws.d_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem](),
                ws.d_wl.unsafe_ptr().unsafe_bitcast[WorkloadInfo](),
                ws.d_splits.unsafe_ptr().unsafe_bitcast[Split](),
                n_rows,
                params.min_impurity_decrease,
                params.min_samples_leaf,
                PART_MB_SAB_NONE,
                grid_dim=(plan.n_blocks_dimx, 1, 1),
                block_dim=(TPB, 1, 1),
            )
            ctx.enqueue_function[partition_scan_kernel[TPB]](
                ws.d_blk_off.unsafe_ptr(),
                ws.d_blk_left.unsafe_ptr(),
                ws.d_blk_base.unsafe_ptr(),
                ws.d_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem](),
                ws.d_splits.unsafe_ptr().unsafe_bitcast[Split](),
                params.min_impurity_decrease,
                params.min_samples_leaf,
                PART_MB_SAB_NONE,
                grid_dim=(n_part, 1, 1),
                block_dim=(TPB, 1, 1),
            )
            ctx.enqueue_function[partition_scatter_kernel[TPB]](
                ws.d_row_alt.unsafe_ptr(),
                d_row_ids.unsafe_ptr(),
                ws.d_blk_off.unsafe_ptr(),
                dataset.d_data.unsafe_ptr(),
                ws.d_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem](),
                ws.d_wl.unsafe_ptr().unsafe_bitcast[WorkloadInfo](),
                ws.d_splits.unsafe_ptr().unsafe_bitcast[Split](),
                n_rows,
                params.min_impurity_decrease,
                params.min_samples_leaf,
                PART_MB_SAB_NONE,
                grid_dim=(plan.n_blocks_dimx, 1, 1),
                block_dim=(TPB, 1, 1),
            )
            ctx.enqueue_function[partition_writeback_kernel[TPB]](
                d_row_ids.unsafe_ptr(),
                ws.d_row_alt.unsafe_ptr(),
                ws.d_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem](),
                ws.d_wl.unsafe_ptr().unsafe_bitcast[WorkloadInfo](),
                ws.d_splits.unsafe_ptr().unsafe_bitcast[Split](),
                params.min_impurity_decrease,
                params.min_samples_leaf,
                PART_MB_SAB_NONE,
                grid_dim=(plan.n_blocks_dimx, 1, 1),
                block_dim=(TPB, 1, 1),
            )
            # DEVIATION 450: no post-partition synchronize. Nothing on the
            # host reads the partition's output; the next cycle's kernels
            # are queue-ordered behind it, and the leaf pass drains before
            # its own host writes. `clock.tick` still serializes when the
            # clock is ENABLED, so the timed attribution is unchanged.
            clock.tick(ctx, PHASE_PARTITION)

            if trace.enabled:
                # The partition's output: the group's whole row-id buffer,
                # a permutation per tree slot. Logical content -- the same
                # bytes on any backend that partitioned identically.
                trace.record_device(
                    ctx, tag_pre + "partition.rowids", d_row_ids
                )
            cyc += 1

            # --- the push, PER QUEUE, in the order each was popped --------
            # --- DEVIATION 466 step 5, or cuML's Push --------------------
            if bestfirst:
                bf_pending.clear()
                bf_pending_q.clear()
                for t in range(n_part):
                    var kids = queues[part_queue[t]].bestfirst_expand(
                        part_items[t], part_splits[t]
                    )
                    for j in range(len(kids)):
                        bf_pending.append(kids[j])
                        bf_pending_q.append(part_queue[t])
            else:
                for t in range(len(seg_queue)):
                    var items_s = List[NodeWorkItem]()
                    var splits_s = List[Split]()
                    for j in range(seg_count[t]):
                        items_s.append(work_items[seg_start[t] + j])
                        splits_s.append(splits[seg_start[t] + j])
                    queues[seg_queue[t]].push(items_s, splits_s)
            clock.tick(ctx, PHASE_HOST_PUSH)

        # --- the LEAF VALUES, ONE launch for the whole group --------------
        # `SetLeafPredictions` (`builder.cuh:556-599`). DEVIATION 214: the
        # per-tree loop allocated seven buffers and synchronized once PER
        # TREE -- the per-level-allocation disease deviation 202 cured in the
        # level loop, still alive in the tail, and the phase clock priced it
        # at 8% of a 100-tree fit. The kernel was batch-ready all along
        # (deviation 180: slice pointers, shrink the grid -- here the grid
        # GROWS instead): every tree's ranges already point into its own
        # slot of the shared `d_row_ids`, and the kernel reads nothing
        # tree-relative but `left_child_id == -1`. So the group's trees are
        # concatenated and the tail is one allocation set, one launch, one
        # readback, one synchronize.
        var trees_g = List[TreeMetaDataNode[DType.float32]]()
        var leaf_base = List[Int]()
        var total_nodes = 0
        for s in range(g):
            leaf_base.append(total_nodes)
            total_nodes += len(queues[s].node_instances)
            trees_g.append(queues[s].get_tree())
        var k_out = Int(n_classes)
        var d_nodes = ctx.enqueue_create_buffer[DType.uint8](
            total_nodes * size_of[SparseTreeNode[DType.float32]]()
        )
        var d_ranges = ctx.enqueue_create_buffer[DType.uint8](
            total_nodes * size_of[InstanceRange]()
        )
        var d_leaves = ctx.enqueue_create_buffer[DType.float32](
            total_nodes * k_out
        )
        var d_visit = ctx.enqueue_create_buffer[DType.int32](total_nodes)
        var h_nodes = ctx.enqueue_create_host_buffer[DType.uint8](
            total_nodes * size_of[SparseTreeNode[DType.float32]]()
        )
        var h_ranges = ctx.enqueue_create_host_buffer[DType.uint8](
            total_nodes * size_of[InstanceRange]()
        )
        var h_leaves = ctx.enqueue_create_host_buffer[DType.float32](
            total_nodes * k_out
        )
        ctx.synchronize()
        var nodes_ptr = h_nodes.unsafe_ptr().unsafe_bitcast[
            SparseTreeNode[DType.float32]
        ]()
        var ranges_ptr = h_ranges.unsafe_ptr().unsafe_bitcast[
            InstanceRange
        ]()
        for s in range(g):
            for i in range(trees_g[s].num_nodes()):
                nodes_ptr[unsafe_offset = leaf_base[s] + i] = trees_g[
                    s
                ].sparsetree[i]
                ranges_ptr[unsafe_offset = leaf_base[s] + i] = queues[
                    s
                ].node_instances[i]
        ctx.enqueue_copy(dst_buf=d_nodes, src_ptr=h_nodes.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=d_ranges, src_ptr=h_ranges.unsafe_ptr())
        # `builder.cuh:582` memsets the leaf array before the launch, and an
        # internal node's ZERO IS ITS VALUE. DEVIATION 471: `zero_fill=True`
        # folds that memset and `d_visit`'s into the launch itself -- each
        # block zeroes its OWN node's `num_outputs` slot and visit cell
        # before the IsLeaf early return, and the grid is one block per
        # node over the whole concatenated buffer, so block-exclusive slot
        # ownership covers exactly what the two memsets covered.
        ctx.enqueue_function[
            leaf_kernel[TPB, LEAF_MAX_OUT_DEFAULT, True, zero_fill=True]
        ](
            d_leaves.unsafe_ptr(),
            d_visit.unsafe_ptr(),
            d_nodes.unsafe_ptr().unsafe_bitcast[
                SparseTreeNode[DType.float32]
            ](),
            d_ranges.unsafe_ptr().unsafe_bitcast[InstanceRange](),
            d_row_ids.unsafe_ptr(),
            dataset.d_labels.unsafe_ptr(),
            Int32(k_out),
            Float32(1.0),  # classification: no fixed-point rescale
            LEAF_SAB_NONE,
            grid_dim=(total_nodes, 1, 1),
            block_dim=(TPB, 1, 1),
        )
        ctx.enqueue_copy(dst_buf=h_leaves, src_buf=d_leaves)
        ctx.synchronize()
        for s in range(g):
            var n_s = trees_g[s].num_nodes()
            trees_g[s].vector_leaf = List[Float32](
                length=n_s * k_out, fill=Float32(0.0)
            )
            for i in range(n_s * k_out):
                trees_g[s].vector_leaf[i] = h_leaves.unsafe_ptr()[
                    unsafe_offset = leaf_base[s] * k_out + i
                ]
            out.append(trees_g[s].copy())
        if trace.enabled:
            # The leaf pass's output for the whole group: the values the
            # model returns, still concatenated across the group's trees.
            trace.record_device(
                ctx, String("g") + String(gi) + ".leaves", d_leaves
            )
        clock.tick(ctx, PHASE_LEAF)
        # Mojo frees a buffer at its LAST USE; these must outlive every
        # launch that read them, and every launch has synchronized above.
        _ = d_row_ids^
        _ = ws^
        gi += 1
        first += g

    return out^


def train_classification_device_resident(
    ctx: DeviceContext,
    mut dataset: DeviceDataset,
    mut row_ids: List[Int32],
    params: DecisionTreeParams,
    tree_id: Int32,
    seed: UInt64,
) raises -> TreeMetaDataNode[DType.float32]:
    """One ExtraTree with its split search on the GPU: a ONE-TREE call of
    `train_forest_classification_device`, which owns the only copy of the
    level loop (DEVIATION 211). Kept so single-tree callers -- and the
    checks that compare the merged forest against one-tree builds -- are
    untouched.

    `row_ids` is deviation 185's vacuous `mut`: the device path fills its
    own row list with a sequence kernel (deviation 200) and never reads the
    host copy. Kept in the signature so the two arms of the forest file
    stay the same loop.
    """
    _ = row_ids
    var ids = List[Int32]()
    ids.append(tree_id)
    var trees = train_forest_classification_device(
        ctx, dataset, params, ids, seed
    )
    return trees[0].copy()


def dataset_len_ok(
    x_col_major: List[Float32], n_rows: Int32, n_cols: Int32
) -> Bool:
    return len(x_col_major) == Int(n_rows) * Int(n_cols)


def search_batch_regression(
    ctx: DeviceContext,
    mut ws: LevelWorkspace,
    mut dataset: DeviceDataset,
    mut d_row_ids: DeviceBuffer[DType.int32],
    work_items: List[NodeWorkItem],
    k: Int,
    params: DecisionTreeParams,
    n_rows: Int32,
    n_cols: Int32,
    item_trees: List[Int32],
    seed: UInt64,
    use_sampler: Bool,
    host_colids: List[Int32],
    range_only: Bool,
    mut clock: PhaseClock,
) raises -> Tuple[
    List[Split], List[Int32], List[Float32], List[Float32], List[Int32]
]:
    """One batch through the REGRESSION split search.

    DEVIATION 211: `item_trees` is one tree id per work item -- see
    `search_batch`'s docstring; the two twins changed together.

    `search_batch`'s twin, and it exists for the same reason: DEVIATION 205's
    rescue has to run the SAME passes on a sub-batch, and a second copy of the
    launch code would drift. The two are not merged because the score pass is
    genuinely different -- fixed-point sums (DEVIATION 135) against class
    counts, and cuML's MSE gain against Gini (DEVIATION 189) -- and merging
    them would mean a runtime branch inside every launch rather than one
    function per objective, which is how cuML templates it
    (`builder.cuh:142`).
    """
    comptime TPB = DEVICE_TPB
    comptime MAX_ACC = DEVICE_MAX_ACC
    var n_nodes = len(work_items)
    if n_nodes == 0:
        # DEVIATION 466: a best-first cycle can have NOTHING to search --
        # every node popped last cycle had two unexpandable children -- and
        # still have nodes left to pop. An empty batch is a well-formed
        # request for no work, not an error. The depth-wise loop breaks
        # before it can ever ask, so this arm belongs to best-first alone.
        return (
            List[Split](),
            List[Int32](),
            List[Float32](),
            List[Float32](),
            List[Int32](),
        )
    # DEVIATION 2020: the search tile is `TPB * SEARCH_ROWS_PER_THREAD`
    # (default 1 = the exact pre-2020 program). The full block, with the
    # bit argument and the required-RED arm, is at the classification
    # twin's call site; the two twins must widen together or the two
    # objectives would launch different grids for the same frontier.
    var plan = build_workload_info(
        work_items, TPB * SEARCH_ROWS_PER_THREAD
    )
    var n_cells = n_nodes * Int(k)

    ref d_min = ws.d_min
    ref d_max = ws.d_max
    ref d_missing = ws.d_missing
    ref d_merges = ws.d_merges
    ref d_minkey = ws.d_minkey
    ref d_maxkey = ws.d_maxkey
    ref d_nleft = ws.d_nleft
    ref d_ntotal = ws.d_ntotal
    ref d_accl = ws.d_accl
    ref d_acct = ws.d_acct
    ref d_nblocks = ws.d_nblocks
    ref d_status = ws.d_status
    ref d_thresh = ws.d_thresh
    ref d_gnum = ws.d_gnum
    ref d_gden = ws.d_gden
    ref c_q = ws.c_q
    ref c_c = ws.c_c
    ref c_m = ws.c_m
    ref c_l = ws.c_l
    ref c_nu = ws.c_nu
    ref c_de = ws.c_de
    ref c_v = ws.c_v
    ref r_q = ws.r_q
    ref r_c = ws.r_c
    ref r_m = ws.r_m
    ref r_l = ws.r_l
    ref r_nu = ws.r_nu
    ref r_de = ws.r_de
    ref r_v = ws.r_v
    ref r_mg = ws.r_mg
    ref r_nw = ws.r_nw
    ref r_mx = ws.r_mx
    ref d_nb = ws.d_nb
    ref d_nc = ws.d_nc
    ref d_colids = ws.d_colids
    ref d_samp_scratch = ws.d_samp_scratch
    ref d_samp_report = ws.d_samp_report
    ref d_items = ws.d_items
    ref d_wl = ws.d_wl
    # DEVIATION 450: no entry synchronize -- see the classification twin.
    ref h_colids = ws.h_colids
    stage_batch(ctx, ws, work_items, item_trees, plan, Int(k))

    # DEVIATION 470: TWO fused seeder launches (halves A and B) replace
    # this cycle's six setup enqueues -- the full argument (bit-inert hoist
    # on the in-order queue, capacity extents for the three memset regions,
    # the survey skipping half B outright, the rescue's zero report extent,
    # the Metal 31-binding limit that forced the A/B split, and the refusal
    # list: no seeder fuses into its grid-accumulating consumer, Metal has
    # no grid sync) is at the classification twin's launch. The one twin
    # difference: the class-accumulator extent is `n_cells` (one output),
    # exactly what the old score-init launch passed here.
    var setup_report = Int32(ws.cap_report) if use_sampler else Int32(0)
    var setup_a_extent = n_cells
    if Int(setup_report) > setup_a_extent:
        setup_a_extent = Int(setup_report)
    if ws.cap_nodes > setup_a_extent:
        setup_a_extent = ws.cap_nodes
    ctx.enqueue_function[phase_setup_a_kernel](
        d_samp_report.unsafe_ptr(),
        setup_report,
        d_min.unsafe_ptr(),
        d_max.unsafe_ptr(),
        d_minkey.unsafe_ptr(),
        d_maxkey.unsafe_ptr(),
        d_missing.unsafe_ptr(),
        d_merges.unsafe_ptr(),
        Int32(n_cells),
        ws.d_nonconst.unsafe_ptr(),
        Int32(ws.cap_nodes),
        grid_dim=ceildiv(setup_a_extent, PHASE_SETUP_TPB),
        block_dim=PHASE_SETUP_TPB,
    )
    if not range_only:
        var setup_b_extent = n_cells
        if ws.cap_nodes > setup_b_extent:
            setup_b_extent = ws.cap_nodes
        ctx.enqueue_function[phase_setup_b_kernel](
            d_status.unsafe_ptr(),
            d_thresh.unsafe_ptr(),
            d_nleft.unsafe_ptr(),
            d_ntotal.unsafe_ptr(),
            d_gnum.unsafe_ptr(),
            d_gden.unsafe_ptr(),
            d_nblocks.unsafe_ptr(),
            d_accl.unsafe_ptr(),
            d_acct.unsafe_ptr(),
            Int32(n_cells),
            Int32(n_cells),
            r_mx.unsafe_ptr(),
            Int32(ws.cap_nodes),
            r_q.unsafe_ptr(),
            r_c.unsafe_ptr(),
            r_m.unsafe_ptr(),
            r_l.unsafe_ptr(),
            r_nu.unsafe_ptr(),
            r_de.unsafe_ptr(),
            r_v.unsafe_ptr(),
            r_mg.unsafe_ptr(),
            r_nw.unsafe_ptr(),
            Int32(n_nodes),
            grid_dim=ceildiv(setup_b_extent, PHASE_SETUP_TPB),
            block_dim=PHASE_SETUP_TPB,
        )

    if not use_sampler:
        # DEVIATION 205's rescue chose these columns on the host, from the
        # survey's own cells. Nothing draws here.
        if len(host_colids) != n_cells:
            raise Error(
                "host_colids must be n_nodes * k long; got "
                + String(len(host_colids))
                + " for "
                + String(n_cells)
            )
        for i in range(n_cells):
            h_colids.unsafe_ptr().unsafe_store(i, host_colids[i])
        # DEVIATION 450: no synchronize after this copy either -- it is
        # queue-ordered ahead of the kernels that read `d_colids`, and
        # `h_colids` is next rewritten only after a required drain.
        ctx.enqueue_copy(dst_buf=d_colids, src_ptr=h_colids.unsafe_ptr())
    else:
        # --- feature sampling, WHERE cuML DOES IT (deviation 201) --------
        # `d_report` is the DEVICE's own statement of which kernel ran;
        # DEVIATION 470's half-A seeder staged `SAMPLER_UNVISITED` into it
        # above, a value no kernel can produce.
        _ = sample_features_for_device(
            ctx,
            d_colids,
            d_samp_scratch.unsafe_ptr(),
            d_samp_report.unsafe_ptr(),
            d_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem](),
            ws.d_tree.unsafe_ptr(),
            h_colids,
            work_items,
            item_trees,
            seed,
            Int(n_cols),
            Int(k),
        )

    clock.tick(ctx, PHASE_STAGE)
    # DEVIATION 470: the range cells were seeded by fused half A above.
    ctx.enqueue_function[node_feature_range_kernel[TPB]](
        d_minkey.unsafe_ptr(),
        d_maxkey.unsafe_ptr(),
        d_missing.unsafe_ptr(),
        d_merges.unsafe_ptr(),
        dataset.d_data.unsafe_ptr(),
        d_row_ids.unsafe_ptr(),
        d_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem](),
        d_wl.unsafe_ptr().unsafe_bitcast[WorkloadInfo](),
        d_colids.unsafe_ptr(),
        n_rows,
        n_cols,
        Int32(k),
        Int32(0),
        grid_dim=(plan.n_blocks_dimx, Int(k), 1),
        block_dim=(TPB, 1, 1),
    )
    # DEVIATION 204: the merge produced order-preserving KEYS; this
    # turns them back into the `(min, max)` floats every later pass
    # reads, and applies the empty-cell sentinel.
    ctx.enqueue_function[node_feature_range_decode_kernel](
        d_min.unsafe_ptr(),
        d_max.unsafe_ptr(),
        d_minkey.unsafe_ptr(),
        d_maxkey.unsafe_ptr(),
        Int32(n_cells),
        Int32(RANGE_SAB_NONE),
        grid_dim=ceildiv(n_cells, 64),
        block_dim=64,
    )

    # --- DID ANY SAMPLED COLUMN VARY? (DEVIATION 205) -----------------
    # DEVIATION 470: `d_nonconst` was zeroed (over full capacity) by
    # fused half A above.
    ctx.enqueue_function[node_nonconstant_flag_kernel](
        ws.d_nonconst.unsafe_ptr(),
        d_min.unsafe_ptr(),
        d_max.unsafe_ptr(),
        d_missing.unsafe_ptr(),
        ws.d_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem](),
        Int32(n_cells),
        Int32(k),
        grid_dim=ceildiv(n_cells, 64),
        block_dim=64,
    )
    ctx.enqueue_copy(dst_buf=ws.h_nonconst, src_buf=ws.d_nonconst)
    # DEVIATION 450: drain only for the survey -- see the classification
    # twin's range pass.
    if range_only:
        ctx.enqueue_copy(dst_buf=ws.o_rmin, src_buf=d_min)
        ctx.enqueue_copy(dst_buf=ws.o_rmax, src_buf=d_max)
        ctx.enqueue_copy(dst_buf=ws.o_rmiss, src_buf=d_missing)
        ctx.synchronize()
    clock.tick(ctx, PHASE_RANGE)

    if range_only:
        var any_nonconst = List[Int32](length=n_nodes, fill=Int32(0))
        for i in range(n_nodes):
            any_nonconst[i] = ws.h_nonconst.unsafe_ptr()[unsafe_offset=i]
        var rmin = List[Float32](length=n_cells, fill=Float32(0.0))
        var rmax = List[Float32](length=n_cells, fill=Float32(0.0))
        var rmiss = List[Int32](length=n_cells, fill=Int32(0))
        for i in range(n_cells):
            rmin[i] = ws.o_rmin.unsafe_ptr()[unsafe_offset=i]
            rmax[i] = ws.o_rmax.unsafe_ptr()[unsafe_offset=i]
            rmiss[i] = ws.o_rmiss.unsafe_ptr()[unsafe_offset=i]
        return (List[Split](), any_nonconst^, rmin^, rmax^, rmiss^)
    # DEVIATION 470: the score cells and the one-output accumulators were
    # seeded by fused half B above (the survey skips half B and returned
    # already).
    ctx.enqueue_function[
        node_feature_score_kernel[TPB, MAX_ACC, False]
    ](
        d_nleft.unsafe_ptr(),
        d_ntotal.unsafe_ptr(),
        d_accl.unsafe_ptr(),
        d_acct.unsafe_ptr(),
        d_nblocks.unsafe_ptr(),
        d_min.unsafe_ptr(),
        d_max.unsafe_ptr(),
        d_missing.unsafe_ptr(),
        dataset.d_data.unsafe_ptr(),
        d_row_ids.unsafe_ptr(),
        dataset.d_labels.unsafe_ptr(),
        d_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem](),
        d_wl.unsafe_ptr().unsafe_bitcast[WorkloadInfo](),
        d_colids.unsafe_ptr(),
        ws.d_tree.unsafe_ptr(),
        n_rows,
        Int32(k),
        Int32(1),
        seed,
        Int32(0),
        grid_dim=(plan.n_blocks_dimx, Int(k), 1),
        block_dim=(TPB, 1, 1),
    )
    ctx.enqueue_function[
        node_feature_score_finalize_kernel[MAX_ACC, False]
    ](
        d_status.unsafe_ptr(),
        d_thresh.unsafe_ptr(),
        d_gnum.unsafe_ptr(),
        d_gden.unsafe_ptr(),
        d_nleft.unsafe_ptr(),
        d_ntotal.unsafe_ptr(),
        d_accl.unsafe_ptr(),
        d_acct.unsafe_ptr(),
        d_min.unsafe_ptr(),
        d_max.unsafe_ptr(),
        d_missing.unsafe_ptr(),
        d_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem](),
        d_colids.unsafe_ptr(),
        ws.d_tree.unsafe_ptr(),
        Int32(n_cells),
        Int32(k),
        Int32(1),
        seed,
        params.min_samples_leaf,
        Int32(0),
        grid_dim=ceildiv(n_cells, 64),
        block_dim=64,
    )
    # The regression candidate carries cuML's MSE gain as its exact key
    # (DEVIATION 189), so the conversion is the classification one with
    # `n_classes = 1`: the accumulator loop runs once and the metric is
    # the gain in SCALED units, which is monotone in the label's own units
    # and therefore orders identically.
    clock.tick(ctx, PHASE_SCORE)
    ctx.enqueue_function[score_to_candidate_kernel](
        c_q.unsafe_ptr(),
        c_c.unsafe_ptr(),
        c_m.unsafe_ptr(),
        c_l.unsafe_ptr(),
        c_nu.unsafe_ptr(),
        c_de.unsafe_ptr(),
        c_v.unsafe_ptr(),
        d_status.unsafe_ptr(),
        d_thresh.unsafe_ptr(),
        d_nleft.unsafe_ptr(),
        d_ntotal.unsafe_ptr(),
        d_accl.unsafe_ptr(),
        d_acct.unsafe_ptr(),
        d_gnum.unsafe_ptr(),
        d_gden.unsafe_ptr(),
        d_colids.unsafe_ptr(),
        Int32(n_cells),
        Int32(1),
        params.min_samples_leaf,
        params.split_criterion,
        grid_dim=ceildiv(n_cells, 64),
        block_dim=64,
    )
    # DEVIATION 470: the reduce cells and the `r_mx` mutexes (over full
    # capacity) were seeded by fused half B above.
    var bpn = ceildiv(Int(k), TPB)
    if bpn < 1:
        bpn = 1
    ctx.enqueue_function[split_reduce_kernel[TPB]](
        r_q.unsafe_ptr(),
        r_c.unsafe_ptr(),
        r_m.unsafe_ptr(),
        r_l.unsafe_ptr(),
        r_nu.unsafe_ptr(),
        r_de.unsafe_ptr(),
        r_v.unsafe_ptr(),
        r_mg.unsafe_ptr(),
        r_nw.unsafe_ptr(),
        r_mx.unsafe_ptr(),
        c_q.unsafe_ptr(),
        c_c.unsafe_ptr(),
        c_m.unsafe_ptr(),
        c_l.unsafe_ptr(),
        c_nu.unsafe_ptr(),
        c_de.unsafe_ptr(),
        c_v.unsafe_ptr(),
        d_nb.unsafe_ptr(),
        d_nc.unsafe_ptr(),
        ws.d_tsalt.unsafe_ptr(),
        Int32(bpn),
        Int32(0),
        grid_dim=(bpn, n_nodes, 1),
        block_dim=(TPB, 1, 1),
    )
    # DEVIATION 463: the exact-tie counter, only when the build asks for it.
    comptime if is_defined["MOJOLEARN_ET_TIE_STATS"]():
        ctx.enqueue_function[split_tie_count_kernel](
            ws.d_ties.unsafe_ptr(),
            r_c.unsafe_ptr(),
            r_nu.unsafe_ptr(),
            r_de.unsafe_ptr(),
            r_v.unsafe_ptr(),
            c_nu.unsafe_ptr(),
            c_de.unsafe_ptr(),
            c_v.unsafe_ptr(),
            d_nb.unsafe_ptr(),
            d_nc.unsafe_ptr(),
            Int32(n_nodes),
            grid_dim=ceildiv(n_nodes, 64),
            block_dim=64,
        )
        ctx.enqueue_copy(dst_buf=ws.o_ties, src_buf=ws.d_ties)

    ref o_q = ws.o_q
    ref o_c = ws.o_c
    ref o_l = ws.o_l
    ref o_v = ws.o_v
    ref o_m = ws.o_m
    ctx.enqueue_copy(dst_buf=o_q, src_buf=r_q)
    ctx.enqueue_copy(dst_buf=o_c, src_buf=r_c)
    ctx.enqueue_copy(dst_buf=o_l, src_buf=r_l)
    ctx.enqueue_copy(dst_buf=o_v, src_buf=r_v)
    ctx.enqueue_copy(dst_buf=o_m, src_buf=r_m)
    ctx.synchronize()
    clock.tick(ctx, PHASE_REDUCE)

    # DEVIATION 450: the deferred `h_nonconst` read -- see the twin.
    var any_nonconst = List[Int32](length=n_nodes, fill=Int32(0))
    for i in range(n_nodes):
        any_nonconst[i] = ws.h_nonconst.unsafe_ptr()[unsafe_offset=i]

    var splits = List[Split]()
    for i in range(n_nodes):
        var colid = o_c.unsafe_ptr()[unsafe_offset=i]
        var metric = o_m.unsafe_ptr()[unsafe_offset=i]
        if o_v.unsafe_ptr()[unsafe_offset=i] == 0 or colid < 0:
            metric = Float32.MIN_FINITE
        splits.append(
            Split(
                o_q.unsafe_ptr()[unsafe_offset=i],
                colid,
                metric,
                o_l.unsafe_ptr()[unsafe_offset=i],
            )
        )

    # DEVIATION 463: report the batch's exact-tie tally. The counter cells
    # rode the queue with the reduce readback, so the sync above completed
    # them; one line per batch, summed by whoever asked for the define.
    comptime if is_defined["MOJOLEARN_ET_TIE_STATS"]():
        var tie_decided = 0
        var tie_tied = 0
        for i in range(n_nodes):
            if o_c.unsafe_ptr()[unsafe_offset=i] >= 0:
                tie_decided += 1
                if ws.o_ties.unsafe_ptr()[unsafe_offset=i] >= Int32(2):
                    tie_tied += 1
        print("ET_TIE_STATS batch decided=", tie_decided, " tied=", tie_tied)

    clock.tick(ctx, PHASE_HOST_SPLITS)
    return (
        splits^,
        any_nonconst^,
        List[Float32](),
        List[Float32](),
        List[Int32](),
    )


def train_regression_device(
    ctx: DeviceContext,
    x_col_major: List[Float32],
    labels_q: List[Int32],
    scale: Float64,
    mut row_ids: List[Int32],
    n_rows: Int32,
    n_cols: Int32,
    params: DecisionTreeParams,
    tree_id: Int32,
    seed: UInt64,
) raises -> TreeMetaDataNode[DType.float32]:
    """One regression tree, uploading the dataset and building the workspace.

    DEVIATION 184's shape, which regression did not have: the upload and the
    workspace belong to the FIT, not to the tree, so `_resident` takes both
    and this two-line wrapper survives for callers that fit a single tree --
    `device_regression_check` is one, and it is untouched by the split.
    An `n_trees`-tree forest through here would upload the same immutable
    matrix `n_trees` times; `fit_regression_device` is the entry point that
    does not.
    """
    if len(x_col_major) != Int(n_rows) * Int(n_cols):
        raise Error("x_col_major must be n_rows * n_cols long, column major")
    if len(labels_q) != Int(n_rows):
        raise Error("labels_q must be n_rows long")
    validity_check(params)
    var dataset = upload_dataset(
        ctx, x_col_major, labels_q, n_rows, n_cols, 1
    )
    return train_regression_device_resident(
        ctx, dataset, scale, row_ids, n_rows, n_cols, params, tree_id,
        seed,
    )


def train_forest_regression_device(
    ctx: DeviceContext,
    mut dataset: DeviceDataset,
    scale: Float64,
    params: DecisionTreeParams,
    tree_ids: List[Int32],
    seed: UInt64,
    sabotage: Int32 = FOREST_SAB_NONE,
    row_slot_cap: Int = FOREST_ROW_SLOT_CAP,
    bootstrap: Bool = False,
    n_sampled_rows: Int32 = 0,
    bf_sabotage: Int32 = BESTFIRST_SAB_NONE,
) raises -> List[TreeMetaDataNode[DType.float32]]:
    """The shipping entry point: an INERT clock -- see `PhaseClock` and the
    classification twin above. `MOJOLEARN_STAGE_TIMES=1` (read once, here)
    enables the clock and prints stage -> seconds at fit end. `bootstrap` /
    `n_sampled_rows` as in the classification twin (DEVIATION 460)."""
    var clock = PhaseClock(stage_times_enabled())
    var out = train_forest_regression_device_timed(
        ctx, dataset, scale, params, tree_ids, seed, sabotage, row_slot_cap,
        clock, bootstrap, n_sampled_rows, bf_sabotage,
    )
    # DEVIATION 2002: same dead-device canary as the classification twin
    # directly above -- see that banner and `core/device_liveness.mojo`.
    assert_device_alive(ctx, "extratrees regression forest fit")
    print_stage_times(clock, "extratrees regression forest fit")
    return out^


def train_forest_regression_device_timed(
    ctx: DeviceContext,
    mut dataset: DeviceDataset,
    scale: Float64,
    params: DecisionTreeParams,
    tree_ids: List[Int32],
    seed: UInt64,
    sabotage: Int32,
    row_slot_cap: Int,
    mut clock: PhaseClock,
    bootstrap: Bool = False,
    n_sampled_rows: Int32 = 0,
    bf_sabotage: Int32 = BESTFIRST_SAB_NONE,
) raises -> List[TreeMetaDataNode[DType.float32]]:
    """`train_forest_classification_device`'s regression twin: the SAME
    merged-frontier forest loop (DEVIATION 211 -- read that block; it is not
    repeated here) with `search_batch_regression` in place of `search_batch`,
    the two kernels instantiated for `CLASSIFICATION = False`, and the leaf
    pass rescaling by `1 / scale` (deviation 179). The twins are not merged
    for `search_batch`'s own reason: the objectives are genuinely different
    functions, which is how cuML templates its builder (`builder.cuh:142`).

    `dataset.d_labels` holds labels ALREADY QUANTIZED by
    `fixed_point.choose_scale` / `quantize` (deviation 135); the scale is the
    forest's, chosen from the whole dataset, so every tree shares it and the
    resident labels serve every slot of the merged batch.

    One departure from the old one-tree body, recorded: the partition's
    splits now go through the WORKSPACE's `d_splits`/`h_splits` staging, as
    classification's always did, instead of allocating fresh buffers every
    level -- deviation 202's rule applied to the last per-level allocation
    this file still had.
    """
    var n_rows = dataset.n_rows
    var n_cols = dataset.n_cols
    validity_check(params)

    comptime TPB = DEVICE_TPB

    # --- identity trace -- NOT A PORT; see the classification twin --------
    var trace = IdentityTrace()
    if trace.enabled:
        trace.header(
            String("extratrees.regression.forest n_rows=")
            + String(n_rows)
            + " n_cols="
            + String(n_cols)
            + " n_trees="
            + String(len(tree_ids))
            + " seed="
            + String(seed)
        )
        # The post-quantize boundary: regression labels arrive ALREADY
        # fixed-point quantized (deviation 135), so the resident quantized
        # labels plus the forest's scale ARE this fit's binning output.
        trace.record_device(ctx, "dataset.data", dataset.d_data)
        trace.record_device(
            ctx, "dataset.labels.quantized", dataset.d_labels
        )
        # The scale, by its BITS (rule 1: never decimal text).
        var sc = List[Float64]()
        sc.append(scale)
        trace.record_host("dataset.scale", sc.unsafe_ptr(), 1)
        _ = sc^

    var k = n_sampled_cols_for(params, n_cols)
    # DEVIATION 466: the growth mode, read ONCE per fit. Every branch on it
    # below is `if bestfirst:` with the depth-wise arm textually unchanged,
    # so a fit that did not ask for best-first executes the same statements
    # in the same order as it did before this mode existed.
    var bestfirst = params.max_leaf_nodes != -1
    if bestfirst and params.max_batch_size < 2:
        # DEVIATION 469's one hard edge, refused BY NAME rather than
        # silently overrunning the workspace. A best-first cycle searches
        # the TWO children of each expanded node, so the narrowest batch the
        # mode can run in is two; the group clamp below can cap `g` at one
        # tree but it cannot cap it below that. `max_batch_size` defaults to
        # 4096 and is a scheduling parameter, so this is reachable only by a
        # caller who set it to 1 on purpose.
        raise Error(
            "max_leaf_nodes needs max_batch_size >= 2: a best-first cycle"
            " searches both children of the node it expands, and a batch of"
            " one cannot hold them (DEVIATION 469). Got max_batch_size "
            + String(params.max_batch_size)
        )
    var out = List[TreeMetaDataNode[DType.float32]]()
    # `row_slot_cap` defaults to FOREST_ROW_SLOT_CAP; the batched check
    # passes a tiny cap to REACH the multi-group path at fixture sizes.
    # DEVIATION 460: the per-tree row SLOT is `n_sampled_rows` wide --
    # `selected_rows.size()` in `get_row_sample` -- which is `n_rows` unless
    # the caller bootstraps with sklearn's `max_samples`. The dataset stays
    # `n_rows` (M) wide; a slot's entries INDEX it.
    var slot_rows = n_rows
    if bootstrap:
        if n_sampled_rows > 0:
            slot_rows = n_sampled_rows
    elif n_sampled_rows != 0 and n_sampled_rows != n_rows:
        raise Error(
            "n_sampled_rows="
            + String(n_sampled_rows)
            + " without bootstrap: the identity permutation is n_rows wide"
            " (randomforest.cuh:69)"
        )
    var group_cap = row_slot_cap // Int(slot_rows)
    if group_cap < 1:
        group_cap = 1
    clock.mark(ctx)

    var gi = 0
    var first = 0
    while first < len(tree_ids):
        var g = len(tree_ids) - first
        if g > group_cap:
            g = group_cap
        if bestfirst:
            # DEVIATION 469: a best-first cycle searches at most TWO nodes
            # per in-flight tree (the children of the one node it expands),
            # so `2 * g` is the search batch's width and the workspace is
            # sized to `max_batch_size`. Capping `g` here is the only place
            # `max_batch_size` bounds anything in this mode, and it keeps
            # its contract: it is a SCHEDULING parameter, and lowering it
            # runs the same trees through narrower launches.
            var bf_cap = Int(params.max_batch_size) // 2
            if bf_cap < 1:
                bf_cap = 1
            if g > bf_cap:
                g = bf_cap
        var total_rows = g * Int(slot_rows)

        var d_row_ids = ctx.enqueue_create_buffer[DType.int32](total_rows)
        # DEVIATION 460: identity permutation or bootstrap sample per slot,
        # exactly as the classification twin.
        fill_row_slots(
            ctx, d_row_ids, g, slot_rows, n_rows, bootstrap, tree_ids,
            first, seed,
        )
        if trace.enabled and bootstrap:
            trace.record_device(
                ctx, String("g") + String(gi) + ".bootstrap.rowids", d_row_ids
            )

        var ws = make_level_workspace(
            ctx,
            Int(params.max_batch_size),
            Int32(total_rows),
            n_cols,
            1,
            Int(k),
            TPB,
        )

        var queues = List[NodeQueue[DType.float32]]()
        for s in range(g):
            var base = Int32(s) * slot_rows
            if sabotage == FOREST_SAB_SHARED_ROW_BASE:
                base = Int32(0)
            queues.append(
                NodeQueue[DType.float32](
                    params, slot_rows, 1, tree_ids[first + s], base
                )
            )
            queues[s].bf_sabotage = bf_sabotage

        clock.tick(ctx, PHASE_SETUP)
        # The trace's level-cycle counter -- see the classification twin.
        var cyc = 0
        # DEVIATION 466: the best-first carry between cycles -- the nodes
        # admitted-but-unsearched, and which queue each belongs to. Empty
        # and never read in depth-wise mode.
        var bf_pending = List[NodeWorkItem]()
        var bf_pending_q = List[Int]()
        if bestfirst:
            for s in range(g):
                var seeds = queues[s].bestfirst_seed()
                for i in range(len(seeds)):
                    bf_pending.append(seeds[i])
                    bf_pending_q.append(s)
        while True:
            var work_items = List[NodeWorkItem]()
            var item_trees = List[Int32]()
            var seg_queue = List[Int]()
            var seg_start = List[Int]()
            var seg_count = List[Int]()
            # DEVIATION 466 step 1: the best-first SEARCH batch is the set
            # of nodes admitted since the last cycle -- the children the
            # last expansions created, or the roots on cycle 0 -- and not a
            # FIFO pop. `seg_*` stays empty; the best-first expansion is per
            # popped node and does not use it.
            if bestfirst:
                for i in range(len(bf_pending)):
                    work_items.append(bf_pending[i])
                    item_trees.append(tree_ids[first + bf_pending_q[i]])
            else:
                for s in range(g):
                    if len(work_items) >= Int(params.max_batch_size):
                        break
                    if not queues[s].has_work():
                        continue
                    var got = queues[s].pop_up_to(
                        Int(params.max_batch_size) - len(work_items)
                    )
                    if len(got) == 0:
                        continue
                    seg_queue.append(s)
                    seg_start.append(len(work_items))
                    seg_count.append(len(got))
                    for i in range(len(got)):
                        work_items.append(got[i])
                        item_trees.append(tree_ids[first + s])
            if len(work_items) == 0:
                # DEVIATION 466: an empty SEARCH batch does NOT end a
                # best-first fit. Both children of every node popped last
                # cycle can be unexpandable while frontiers still hold
                # splittable nodes, so the fit ends only when no tree can
                # pop -- `while not frontier.empty()` (`_tree.pyx:445`)
                # plus the budget test at `:454`.
                var bf_more = False
                if bestfirst:
                    for s in range(g):
                        if queues[s].bestfirst_can_pop():
                            bf_more = True
                if not bf_more:
                    clock.tick(ctx, PHASE_HOST_QUEUE)
                    break
            if sabotage == FOREST_SAB_SCALAR_TREE:
                for i in range(len(item_trees)):
                    item_trees[i] = item_trees[0]
            var n_nodes = len(work_items)

            clock.tick(ctx, PHASE_HOST_QUEUE)
            var rfound = search_batch_regression(
                ctx,
                ws,
                dataset,
                d_row_ids,
                work_items,
                Int(k),
                params,
                n_rows,
                n_cols,
                item_trees,
                seed,
                True,
                List[Int32](),
                False,
                clock,
            )
            var splits = rfound[0].copy()
            var any_nonconst = rfound[1].copy()

            var tag_pre = String("")
            if trace.enabled:
                # The tag is set whether or not there is anything to record
                # under it: DEVIATION 466's empty search batch still runs a
                # partition, and a cycle that fell back to the bare
                # "partition.rowids" tag would collide with the next one and
                # break the differ's uniqueness invariant.
                tag_pre = String("g") + String(gi) + ".c" + String(cyc) + "."
            # `n_nodes > 0`: an empty search batch has no cells to record,
            # and a zero-length record would break the differ's alignment
            # invariant rather than inform it.
            if trace.enabled and n_nodes > 0:
                # DEVIATION 454: the hazard stages in pipeline order --
                # see the classification twin for the bisection argument.
                var n_cells_t = n_nodes * Int(k)
                trace.record_device(
                    ctx, tag_pre + "colids", ws.d_colids, n_cells_t
                )
                trace.record_device(
                    ctx, tag_pre + "range.min", ws.d_min, n_cells_t
                )
                trace.record_device(
                    ctx, tag_pre + "range.max", ws.d_max, n_cells_t
                )
                trace.record_device(
                    ctx, tag_pre + "draw.thresh", ws.d_thresh, n_cells_t
                )
                # The level's REDUCED per-node winners (rule 3: the logical
                # reduced buffer). Regression's readback carries no exact
                # num/den pair -- the gain travels with the candidate
                # (deviation 183) -- so the winner's column and gain are
                # the reduced record.
                trace.record_host(
                    tag_pre + "reduce.colid", ws.o_c.unsafe_ptr(), n_nodes
                )
                trace.record_host(
                    tag_pre + "reduce.gain", ws.o_m.unsafe_ptr(), n_nodes
                )

            # --- DEVIATION 205, the regression half ----------------------
            var retry = List[Int]()
            for i in range(n_nodes):
                if any_nonconst[i] == 0 and work_items[i].instances.count > 0:
                    retry.append(i)

            if len(retry) > 0:
                var sub = List[NodeWorkItem]()
                var sub_trees = List[Int32]()
                for j in range(len(retry)):
                    sub.append(work_items[retry[j]])
                    sub_trees.append(item_trees[retry[j]])
                var ident = List[Int32](
                    length=len(sub) * Int(n_cols), fill=Int32(0)
                )
                for j in range(len(sub)):
                    for c in range(Int(n_cols)):
                        ident[j * Int(n_cols) + c] = Int32(c)
                var survey = search_batch_regression(
                    ctx, ws, dataset, d_row_ids, sub, Int(n_cols), params,
                    n_rows, n_cols, sub_trees, seed, False, ident, True, clock,
                )
                var s_min = survey[2].copy()
                var s_max = survey[3].copy()
                var s_miss = survey[4].copy()

                var chosen_items = List[NodeWorkItem]()
                var chosen_trees = List[Int32]()
                var chosen_cols = List[Int32]()
                var chosen_slot = List[Int]()
                for j in range(len(sub)):
                    var nonconst = List[Int32]()
                    for c in range(Int(n_cols)):
                        var idx = j * Int(n_cols) + c
                        var extent = FeatureRange(
                            s_min[idx], s_max[idx], s_miss[idx]
                        )
                        if not node_feature_is_constant(
                            extent, sub[j].instances.count
                        ):
                            nonconst.append(Int32(c))
                    if len(nonconst) == 0:
                        continue
                    var u = rescue_pick(
                        rescue_key(
                            seed, sub_trees[j], UInt32(Int(sub[j].idx))
                        ),
                        len(nonconst),
                    )
                    chosen_items.append(sub[j])
                    chosen_trees.append(sub_trees[j])
                    chosen_cols.append(nonconst[u])
                    chosen_slot.append(retry[j])

                if len(chosen_items) > 0:
                    var res2 = search_batch_regression(
                        ctx, ws, dataset, d_row_ids, chosen_items, 1, params,
                        n_rows, n_cols, chosen_trees, seed, False,
                        chosen_cols, False, clock,
                    )
                    var rescued = res2[0].copy()
                    for t in range(len(chosen_slot)):
                        splits[chosen_slot[t]] = rescued[t]

            if trace.enabled and n_nodes > 0:
                # The SELECTED splits -- post-rescue; see the twin, and
                # DEVIATION 466's note there about what this is under
                # best-first growth.
                var t_q = List[Float32]()
                var t_c = List[Int32]()
                var t_l = List[Int32]()
                var t_m = List[Float32]()
                for i in range(n_nodes):
                    t_q.append(splits[i].quesval)
                    t_c.append(splits[i].colid)
                    t_l.append(splits[i].n_left)
                    t_m.append(splits[i].best_metric_val)
                trace.record_list_f32(tag_pre + "split.thresh", t_q)
                trace.record_list_i32(tag_pre + "split.colid", t_c)
                trace.record_list_i32(tag_pre + "split.nleft", t_l)
                trace.record_list_f32(tag_pre + "split.gain", t_m)

            # --- DEVIATION 466 steps 2 and 3: ADMIT, then POP ------------
            # `part_*` is the PARTITION batch. In depth-wise mode it is the
            # search batch, item for item, which is what it always was; in
            # best-first it is the ONE node each tree pops, whose split was
            # found in an earlier cycle and is still correct because a
            # node's rows are permuted only by its own partition or an
            # ancestor's, and neither has happened while it waited.
            var part_items = List[NodeWorkItem]()
            var part_trees = List[Int32]()
            var part_splits = List[Split]()
            var part_queue = List[Int]()
            if bestfirst:
                for i in range(n_nodes):
                    _ = queues[bf_pending_q[i]].bestfirst_admit(
                        work_items[i], splits[i], item_trees[i]
                    )
                for s in range(g):
                    if not queues[s].bestfirst_can_pop():
                        continue
                    var rec = queues[s].bestfirst_pop()
                    part_items.append(rec.item)
                    part_trees.append(tree_ids[first + s])
                    part_splits.append(rec.split)
                    part_queue.append(s)
                if len(part_items) == 0:
                    # Every frontier is empty or every budget is spent.
                    clock.tick(ctx, PHASE_HOST_QUEUE)
                    break
            else:
                for i in range(n_nodes):
                    part_items.append(work_items[i])
                    part_trees.append(item_trees[i])
                    part_splits.append(splits[i])
            var n_part = len(part_items)

            var plan = build_workload_info(part_items, TPB)
            if bestfirst:
                # DEVIATION 469's synchronization price: the search batch
                # and the partition batch are DIFFERENT SETS here, so
                # `ws.h_items` is re-staged every cycle and DEVIATION 455's
                # drain runs every cycle rather than only when DEVIATION
                # 205's rescue fires.
                stage_batch(ctx, ws, part_items, part_trees, plan, Int(k))
                ctx.synchronize()
            elif len(retry) > 0 or SEARCH_ROWS_PER_THREAD > 1:
                # DEVIATION 2020 (restage leg) -- see the classification
                # twin for the full block: the plain cycle's restage skip
                # was sound only while the partition's TPB plan was
                # byte-identical to the plan the search staged, and the
                # widened search tile broke that premise; at R > 1 the
                # plain cycle restages `d_wl` (472's byte-compare keeps it
                # to the one changed slot) and pays 455's drain. The
                # comptime disjunct folds away at R = 1.
                stage_batch(ctx, ws, work_items, item_trees, plan, Int(k))
                # DEVIATION 455 -- see the classification twin: the
                # re-stage's copies must be drained before the next cycle
                # rewrites the staging.
                ctx.synchronize()

            # --- the PARTITION (deviation 203, the regression half) -------
            # DEVIATION 450: no pre-partition synchronize -- see the twin.
            var splits_ptr = ws.h_splits.unsafe_ptr().unsafe_bitcast[Split]()
            for i in range(n_part):
                splits_ptr[unsafe_offset=i] = part_splits[i]
            ctx.enqueue_copy(
                dst_buf=ws.d_splits, src_ptr=ws.h_splits.unsafe_ptr()
            )
            ctx.enqueue_function[partition_count_kernel[TPB]](
                ws.d_blk_left.unsafe_ptr(),
                d_row_ids.unsafe_ptr(),
                dataset.d_data.unsafe_ptr(),
                ws.d_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem](),
                ws.d_wl.unsafe_ptr().unsafe_bitcast[WorkloadInfo](),
                ws.d_splits.unsafe_ptr().unsafe_bitcast[Split](),
                n_rows,
                params.min_impurity_decrease,
                params.min_samples_leaf,
                PART_MB_SAB_NONE,
                grid_dim=(plan.n_blocks_dimx, 1, 1),
                block_dim=(TPB, 1, 1),
            )
            ctx.enqueue_function[partition_scan_kernel[TPB]](
                ws.d_blk_off.unsafe_ptr(),
                ws.d_blk_left.unsafe_ptr(),
                ws.d_blk_base.unsafe_ptr(),
                ws.d_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem](),
                ws.d_splits.unsafe_ptr().unsafe_bitcast[Split](),
                params.min_impurity_decrease,
                params.min_samples_leaf,
                PART_MB_SAB_NONE,
                grid_dim=(n_part, 1, 1),
                block_dim=(TPB, 1, 1),
            )
            ctx.enqueue_function[partition_scatter_kernel[TPB]](
                ws.d_row_alt.unsafe_ptr(),
                d_row_ids.unsafe_ptr(),
                ws.d_blk_off.unsafe_ptr(),
                dataset.d_data.unsafe_ptr(),
                ws.d_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem](),
                ws.d_wl.unsafe_ptr().unsafe_bitcast[WorkloadInfo](),
                ws.d_splits.unsafe_ptr().unsafe_bitcast[Split](),
                n_rows,
                params.min_impurity_decrease,
                params.min_samples_leaf,
                PART_MB_SAB_NONE,
                grid_dim=(plan.n_blocks_dimx, 1, 1),
                block_dim=(TPB, 1, 1),
            )
            ctx.enqueue_function[partition_writeback_kernel[TPB]](
                d_row_ids.unsafe_ptr(),
                ws.d_row_alt.unsafe_ptr(),
                ws.d_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem](),
                ws.d_wl.unsafe_ptr().unsafe_bitcast[WorkloadInfo](),
                ws.d_splits.unsafe_ptr().unsafe_bitcast[Split](),
                params.min_impurity_decrease,
                params.min_samples_leaf,
                PART_MB_SAB_NONE,
                grid_dim=(plan.n_blocks_dimx, 1, 1),
                block_dim=(TPB, 1, 1),
            )
            # DEVIATION 450: no post-partition synchronize -- see the twin.
            clock.tick(ctx, PHASE_PARTITION)

            if trace.enabled:
                # The partition's output permutation -- see the twin.
                trace.record_device(
                    ctx, tag_pre + "partition.rowids", d_row_ids
                )
            cyc += 1

            # --- DEVIATION 466 step 5, or cuML's Push --------------------
            if bestfirst:
                bf_pending.clear()
                bf_pending_q.clear()
                for t in range(n_part):
                    var kids = queues[part_queue[t]].bestfirst_expand(
                        part_items[t], part_splits[t]
                    )
                    for j in range(len(kids)):
                        bf_pending.append(kids[j])
                        bf_pending_q.append(part_queue[t])
            else:
                for t in range(len(seg_queue)):
                    var items_s = List[NodeWorkItem]()
                    var splits_s = List[Split]()
                    for j in range(seg_count[t]):
                        items_s.append(work_items[seg_start[t] + j])
                        splits_s.append(splits[seg_start[t] + j])
                    queues[seg_queue[t]].push(items_s, splits_s)
            clock.tick(ctx, PHASE_HOST_PUSH)

        # --- the LEAF VALUES, ONE launch for the whole group --------------
        # `SetLeafPredictions` (`builder.cuh:556-599`). DEVIATION 214: the
        # per-tree loop allocated seven buffers and synchronized once PER
        # TREE -- the per-level-allocation disease deviation 202 cured in the
        # level loop, still alive in the tail, and the phase clock priced it
        # at 8% of a 100-tree fit. The kernel was batch-ready all along
        # (deviation 180: slice pointers, shrink the grid -- here the grid
        # GROWS instead): every tree's ranges already point into its own
        # slot of the shared `d_row_ids`, and the kernel reads nothing
        # tree-relative but `left_child_id == -1`. So the group's trees are
        # concatenated and the tail is one allocation set, one launch, one
        # readback, one synchronize.
        var trees_g = List[TreeMetaDataNode[DType.float32]]()
        var leaf_base = List[Int]()
        var total_nodes = 0
        for s in range(g):
            leaf_base.append(total_nodes)
            total_nodes += len(queues[s].node_instances)
            trees_g.append(queues[s].get_tree())
        var k_out = 1
        var d_nodes = ctx.enqueue_create_buffer[DType.uint8](
            total_nodes * size_of[SparseTreeNode[DType.float32]]()
        )
        var d_ranges = ctx.enqueue_create_buffer[DType.uint8](
            total_nodes * size_of[InstanceRange]()
        )
        var d_leaves = ctx.enqueue_create_buffer[DType.float32](
            total_nodes * k_out
        )
        var d_visit = ctx.enqueue_create_buffer[DType.int32](total_nodes)
        var h_nodes = ctx.enqueue_create_host_buffer[DType.uint8](
            total_nodes * size_of[SparseTreeNode[DType.float32]]()
        )
        var h_ranges = ctx.enqueue_create_host_buffer[DType.uint8](
            total_nodes * size_of[InstanceRange]()
        )
        var h_leaves = ctx.enqueue_create_host_buffer[DType.float32](
            total_nodes * k_out
        )
        ctx.synchronize()
        var nodes_ptr = h_nodes.unsafe_ptr().unsafe_bitcast[
            SparseTreeNode[DType.float32]
        ]()
        var ranges_ptr = h_ranges.unsafe_ptr().unsafe_bitcast[
            InstanceRange
        ]()
        for s in range(g):
            for i in range(trees_g[s].num_nodes()):
                nodes_ptr[unsafe_offset = leaf_base[s] + i] = trees_g[
                    s
                ].sparsetree[i]
                ranges_ptr[unsafe_offset = leaf_base[s] + i] = queues[
                    s
                ].node_instances[i]
        ctx.enqueue_copy(dst_buf=d_nodes, src_ptr=h_nodes.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=d_ranges, src_ptr=h_ranges.unsafe_ptr())
        # `builder.cuh:582` memsets the leaf array before the launch, and an
        # internal node's ZERO IS ITS VALUE. DEVIATION 471: `zero_fill=True`
        # folds that memset and `d_visit`'s into the launch itself -- the
        # block-exclusive-ownership argument is at the classification twin.
        ctx.enqueue_function[
            leaf_kernel[TPB, LEAF_MAX_OUT_DEFAULT, False, zero_fill=True]
        ](
            d_leaves.unsafe_ptr(),
            d_visit.unsafe_ptr(),
            d_nodes.unsafe_ptr().unsafe_bitcast[
                SparseTreeNode[DType.float32]
            ](),
            d_ranges.unsafe_ptr().unsafe_bitcast[InstanceRange](),
            d_row_ids.unsafe_ptr(),
            dataset.d_labels.unsafe_ptr(),
            Int32(k_out),
            # `inv_scale` puts the fixed-point mean back into the
            # label's own units -- DEVIATION 179.
            Float32(1.0 / scale),
            LEAF_SAB_NONE,
            grid_dim=(total_nodes, 1, 1),
            block_dim=(TPB, 1, 1),
        )
        ctx.enqueue_copy(dst_buf=h_leaves, src_buf=d_leaves)
        ctx.synchronize()
        for s in range(g):
            var n_s = trees_g[s].num_nodes()
            trees_g[s].vector_leaf = List[Float32](
                length=n_s * k_out, fill=Float32(0.0)
            )
            for i in range(n_s * k_out):
                trees_g[s].vector_leaf[i] = h_leaves.unsafe_ptr()[
                    unsafe_offset = leaf_base[s] * k_out + i
                ]
            out.append(trees_g[s].copy())
        if trace.enabled:
            # The leaf pass's output for the whole group -- see the twin.
            trace.record_device(
                ctx, String("g") + String(gi) + ".leaves", d_leaves
            )
        clock.tick(ctx, PHASE_LEAF)
        _ = d_row_ids^
        _ = ws^
        gi += 1
        first += g

    return out^


def train_regression_device_resident(
    ctx: DeviceContext,
    mut dataset: DeviceDataset,
    scale: Float64,
    mut row_ids: List[Int32],
    n_rows: Int32,
    n_cols: Int32,
    params: DecisionTreeParams,
    tree_id: Int32,
    seed: UInt64,
) raises -> TreeMetaDataNode[DType.float32]:
    """One regression ExtraTree: a ONE-TREE call of
    `train_forest_regression_device`, which owns the only copy of the loop
    (DEVIATION 211). The old per-call `LevelWorkspace` argument is gone --
    the forest trainer builds its own, once per group -- and `row_ids` is
    deviation 185's vacuous `mut`, kept so the two arms stay the same loop.
    """
    _ = row_ids
    if dataset.n_rows != n_rows or dataset.n_cols != n_cols:
        raise Error(
            "train_regression_device_resident: the dataset on the device is "
            + String(dataset.n_rows)
            + "x"
            + String(dataset.n_cols)
            + " but the caller says "
            + String(n_rows)
            + "x"
            + String(n_cols)
        )
    var ids = List[Int32]()
    ids.append(tree_id)
    var trees = train_forest_regression_device(
        ctx, dataset, scale, params, ids, seed
    )
    return trees[0].copy()


def train_loop_shape() -> String:
    """`builder.cuh:344-359`, `Builder::train`, quoted rather than executed.

    Their loop is exactly:

        NodeQueue queue(params, maxNodes(), n_sampled_rows, num_outputs);
        while (queue.HasWork()) {
          auto work_items = queue.Pop();
          auto [splits_host_ptr, splits_count] = doSplit(work_items);
          queue.Push(work_items, splits_host_ptr);
        }
        auto tree = queue.GetTree();
        this->SetLeafPredictions(tree, queue.GetInstanceRanges());

    `doSplit` and `SetLeafPredictions` were device work this lane had not
    written when this quotation was added, so the shape was recorded instead of
    executed. They are written now: `train_classification` (`:570`) and
    `train_classification_device` (`:1259`) both run this loop, and
    `set_leaf_predictions_classification` (`:405`) is its last line. What is
    kept here is the verbatim quotation of their source that those trainers'
    docstrings cite -- a citation, no longer a placeholder.
    """
    return "see docstring"
