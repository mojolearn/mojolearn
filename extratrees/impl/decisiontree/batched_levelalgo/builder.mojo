# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""ExtraTrees host control plane and device drivers for breadth-first and best-first tree growth, implemented from pinned cuML and sklearn implementations."""

from std.collections import InlineArray
from core.tree_math import tree_log
from ensemble.tree_moments import balanced_mse_gain, et_leaf_moment_host
from ensemble.tree_identical_ideas import T02, T04, T05, T06, T07, T11, T11_LEVELS, T12, T12_BYTES, T13, T13_BYTES, T14, T14_EXACT, C48
from std.memory import bitcast, memcpy

from ensemble.instruments import StageTimes

from extratrees.impl.decisiontree.decisiontree import (
    CRITERION_END,
    CRITERION_ENTROPY,
    CRITERION_GAMMA,
    CRITERION_GINI,
    CRITERION_MSE,
    CRITERION_INVERSE_GAUSSIAN,
    CRITERION_POISSON,
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
    regression_deviance_gain,
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
    draw_threshold_device,
    range_key,
    range_unkey,
    RANGE_KEY_MIN_SEED,
    RANGE_KEY_MAX_SEED,
    classification_key_shift,
    regression_key,
    SCORE_SAB_NONE,
    T05_SMALL_NODE_ROWS,
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
    node_feature_score_host,
    ScoredCandidate,
    leaf_kernel,
    node_split_kernel,
    node_feature_range_kernel,
    node_feature_range_tiled_kernel,
    et_snap_code,
    ET_QSTRIDE,
    node_feature_score_reg_tiled_kernel,
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
from extratrees.checks.rescue import rescue_key, rescue_pick
from extratrees.impl.decisiontree.flatnode import SparseTreeNode
from extratrees.impl.decisiontree.batched_levelalgo.kernels.partition_multiblock import (
    PART_MB_SAB_NONE,
    partition_count_kernel,
    partition_scan_kernel,
    partition_scatter_kernel,
    partition_writeback_kernel,
)
from extratrees.impl.decisiontree.batched_levelalgo.kernels.et_loop_kernels import (
    ETL_HDR_WORDS,
    ETL_H_CUR,
    ETL_H_GCOUNT,
    ETL_H_HEAD,
    ETL_H_NODES,
    ETL_H_NSUB,
    ETL_H_OVERFLOW,
    ETL_H_POPS,
    ETL_H_ROWS,
    ETL_H_STAT_NODES,
    ETL_H_STAT_RESCUED,
    ETL_H_STAT_RETRY,
    ETL_H_TAIL,
    ETL_META_INTS,
    ETL_Q_INTS,
    ETL_SHARED_FITS,
    ETL_STAT_INTS,
    ETL_ST_DEPTH,
    ETL_ST_FRONT,
    ETL_ST_LEAVES,
    ETL_ST_NODES,
    ETL_TPB,
    etl_copy_splits_kernel,
    etl_dummy_item,
    etl_init_kernel,
    etl_map_kernel,
    etl_merge_kernel,
    etl_pop_kernel,
    etl_push_commit_kernel,
    etl_push_mark_kernel,
    etl_push_rank_kernel,
    etl_push_slot_kernel,
    etl_push_write_kernel,
    etl_retry_kernel,
    etl_scatter_kernel,
    etl_stage_kernel,
    etl_tree_base_kernel,
)
from extratrees.impl.decisiontree.batched_levelalgo.split import (
    ExactKey,
    SPLIT_SAB_NONE,
    SplitExact,
    split_reduce_kernel,
    split_tie_count_kernel,
    split_tie_salt_for,
)
from extratrees.checks.pcg_rng import key_for
from extratrees.checks.pcg_rng import PCGenerator, uniform_int_u32
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from std.gpu import WARP_SIZE, block_dim, block_idx, grid_dim, thread_idx
from std.math import ceildiv, fma
from std.atomic import Atomic
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from max.gpu.primitives.block import min as block_min
from max.gpu.primitives.block import max as block_max
from max.gpu.primitives.block import sum as block_sum
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator, size_of

from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_FAST,
    NUMERIC_IDENTICAL,
    ftz,
    identical_div,
    identical_mul,
)
from core.philox import launch_uniform_int
from core.abs_sum_blocked import device_abs_sum_blocked
from extratrees.checks.fixed_point import choose_scale
from ensemble.decisiontree.batched_levelalgo.quantiles import compute_quantiles
from extratrees.checks.pcg_rng import row_sample_seed


def max_nodes(max_depth: Int32) -> Int:
    """`builder.cuh:253-262`, including the reference's cliff.

    The reference: a dense tree's node count for depth < 13, and a FIXED 8191 above
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
#   argued that the two references grow different trees and that accepting
#   sklearn's name would be accepting cuML's algorithm under it. The first
#   half is true and the second half does not follow: the answer to two
#   references disagreeing is to implement the one whose NAME the caller
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
# WHAT IS NOT IMPLEMENTED FROM THEIR BUILDER, stated rather than left to be
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
#   only case this implementation supports (NOT_IMPLEMENTED.tsv's `sample_weight`
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
"""Order the frontier by arrival instead of by improvement -- what an implementation
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

    `FrontierRecord` (`_tree.pyx:341-357`), minus the members this implementation does
    not have a use for -- see DEVIATION BLOCK 466's "what is not implemented".
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

    @always_inline
    def __init__(out self, *, copy: Self):
        """Metal: an inlined field-by-field copy (see `InstanceRange` in
        `kernels/builder_kernels.mojo`): the synthesized out-of-line copy
        crashes Apple's Metal compiler on a whole-record device load."""
        self.item = copy.item
        self.split = copy.split
        self.key = copy.key
        self.tree_id = copy.tree_id


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
        """`builder.cuh:83-89`, test for test, in the reference order.

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
        """cpu4-forest: the HOST COLUMN body lives in `push_host` (the checker's
        host-column naming); this name is kept for its callers."""
        self.push_host(work_items, splits)

    def push_host(
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
        a semantic. Kept as the reference writes it anyway (do not tidy), and
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
        """`max_split_nodes > 0` (`_tree.pyx:424`, `:454`), in this implementation's
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
        leaves it one (DEVIATION BLOCK 466, "what is not implemented", (a)). The
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
    """cpu4-forest: the HOST COLUMN body lives in `set_leaf_predictions_classification_host` (the checker's
    host-column naming); this name is kept for its callers."""
    set_leaf_predictions_classification_host(dataset, tree, node_instances)


def set_leaf_predictions_classification_host(
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
    """cpu4-forest: the HOST COLUMN body lives in `set_leaf_predictions_regression_host` (the checker's
    host-column naming); this name is kept for its callers."""
    set_leaf_predictions_regression_host(dataset, tree, node_instances)


def set_leaf_predictions_regression_host(
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


def rescue_columns(
    dataset: Dataset, work_item: NodeWorkItem
) raises -> List[Int32]:
    """cpu4-forest: the HOST COLUMN body lives in `rescue_columns_host` (the checker's
    host-column naming); this name is kept for its callers."""
    return rescue_columns_host(dataset, work_item)


def rescue_columns_host(
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


# ==========================================================================
# THE HOST RESTATEMENT OF THE DEVICE TRAINER (the CPU training lane, phase
# 1, et-clf and et-reg, 2026-09-14).
#
# `train_classification` and `train_regression` above are sklearn's
# splitter on the host: `node_split_random_gini` orders candidates by the
# exact Gini rational and is held bit for bit to the device
# (`device_forest_check`), but `node_split_random_mse` orders by sklearn's
# FLOAT64 proxy (DEVIATION 153) where the device orders by cuML's MSE gain
# as an exact `Int64` rational over QUANTIZED labels (DEVIATION 189,
# `regression_key`: `(|S_L n_R - S_R n_L| >> j)^2 / (n_L n_R)`). The two
# orderings agree in exact arithmetic (the proxy is the gain plus a
# per-node constant) but not bit for bit: the device's node-uniform shift
# `j` (14 bits at 20,000 rows) turns near-ties into exact ties resolved by
# DEVIATION 463's keyed rank, and the float64 proxy separates them. A host
# fit that must reproduce the GPU columns' bytes on any input therefore
# restates the DEVICE's search, not sklearn's, and this block does so for
# both objectives from the device's own host oracles: per (node, feature)
# `node_feature_score_host` (the score kernel's sequential oracle,
# `builder_kernels_impl.mojo`; `regression_score_check` and
# `score_kernel_check` hold the kernels to it cell for cell), then the
# candidate exactly as `score_to_candidate_kernel` forms it (the metric
# from `gain_per_split` / `entropy_gain_per_split`, the key the oracle's
# rational or `float_gain_key` for entropy), then `SplitExact.update` in
# slot order (the reduction `split_reduce_kernel` runs; the order is total
# so the walk order is immaterial), then the readback's `MIN_FINITE` fix,
# `split_not_valid`, `partition_samples` and `NodeQueue.push` as the device
# loop applies them, DEVIATION 205's rescue keyed exactly as the device
# keys it, and the leaf pass as `leaf_kernel` computes it
# (`leaf_values_host`'s arithmetic over the quantized labels with the
# device's `inv_scale`). The tree structure the device returns and the
# leaf VALUES the device returns are both reproduced; the regressor's
# leaves are means of quantized labels, which the older
# `fit_extra_trees_regressor_reference` cannot give.
#
# Best-first growth (`max_leaf_nodes`, DEVIATION 466) was refused here by
# name until 2026-09-15; `train_tree_exact_bestfirst` below now restates
# the device's best-first cycle on the same exact search, and
# `train_tree_exact` dispatches to it.
# ==========================================================================


def _exact_candidate(
    dataset: Dataset,
    labels_q: MutPointer[Int32, MutAnyOrigin],
    item: NodeWorkItem,
    col: Int32,
    n_acc: Int,
    is_classification: Bool,
    criterion: Int32,
    min_samples_leaf: Int32,
    seed: UInt64,
    tree_id: Int32,
) -> SplitExact:
    """One (node, feature) cell as the device's reduction receives it:
    `node_feature_score_host` then `score_to_candidate_kernel`'s policy (a
    cell whose status is not SCORED is the default `Split` with the absent
    key; a scored cell carries cuML's float gain as the metric and the
    exact rational as the key, entropy's key being the sign-magnitude map
    of its float gain over `den = 1`, DEVIATION 459)."""
    var key = key_for(
        seed, UInt32(Int(tree_id)), UInt32(Int(item.idx)), UInt32(Int(col))
    )
    var cell: ScoredCandidate
    cell = node_feature_score_host(
        dataset.data.unsafe_origin_cast[MutAnyOrigin](),
        dataset.row_ids.unsafe_origin_cast[MutAnyOrigin](),
        labels_q,
        Int(dataset.m),
        Int(item.instances.begin),
        Int(item.instances.count),
        Int(col),
        node_feature_min_max(dataset, item, col),
        key,
        n_acc,
        is_classification,
        Int(min_samples_leaf),
        True,
    )
    if cell.status != SCORE_STATUS_SCORED:
        return SplitExact()
    var acc_left = cell.acc_left.copy()
    var acc_total = cell.acc_total.copy()
    var left_p = acc_left.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin]()
    var total_p = acc_total.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutAnyOrigin]()
    var metric: Float32
    var num: Int64
    var den: Int64
    if is_classification and criterion == CRITERION_ENTROPY:
        metric = entropy_gain_per_split(
            left_p, total_p, 0, n_acc, cell.n_total, cell.n_left, min_samples_leaf
        )
        num = float_gain_key(metric)
        den = Int64(1)
    elif not is_classification and (
        criterion == CRITERION_POISSON or criterion == CRITERION_GAMMA or criterion == CRITERION_INVERSE_GAUSSIAN
    ):
        # DEVIATION 5610, as `score_to_candidate_kernel` forms it.
        metric = regression_deviance_gain(acc_left[0], acc_total[0], cell.n_left, cell.n_total, criterion)
        num = float_gain_key(metric)
        den = Int64(1)
    else:
        metric = gain_per_split(
            left_p, total_p, 0, n_acc, cell.n_total, cell.n_left, min_samples_leaf
        )
        comptime if T14 and not T14_EXACT:
            if not is_classification and criterion == CRITERION_MSE:
                metric = max(Float32(0),ftz(Float32(2)*balanced_mse_gain(Float32(cell.n_total),Float32(cell.n_left),Float32(acc_total[0]),Float32(acc_left[0]))))
        num = cell.gini_num
        den = cell.gini_den
    _ = acc_left^
    _ = acc_total^
    return SplitExact(
        Split(cell.threshold, col, metric, cell.n_left), ExactKey(num, den, 1)
    )


def _exact_node_split(
    dataset: Dataset,
    labels_q: MutPointer[Int32, MutAnyOrigin],
    item: NodeWorkItem,
    colids: List[Int32],
    n_acc: Int,
    is_classification: Bool,
    criterion: Int32,
    min_samples_leaf: Int32,
    seed: UInt64,
    tree_id: Int32,
) -> Split:
    """The device's per-node winner: the candidates of `colids` reduced by
    `SplitExact.update` under `split_tie_salt_for(tree, node)`, then the
    readback's fix (`search_batch*`: an invalid key or a negative column
    reads `MIN_FINITE`, so `split_not_valid` rejects it)."""
    var tie_salt = split_tie_salt_for(UInt32(Int(tree_id)), UInt32(Int(item.idx)))
    var acc = SplitExact()
    for ci in range(len(colids)):
        var cand = _exact_candidate(
            dataset, labels_q, item, colids[ci], n_acc, is_classification,
            criterion, min_samples_leaf, seed, tree_id,
        )
        _ = acc.update(cand, SPLIT_SAB_NONE, tie_salt)
    var out = acc.split
    if acc.key.valid == 0 or out.colid < 0:
        out.best_metric_val = Float32.MIN_FINITE
    return out


def _exact_extent(
    dataset: Dataset, item: NodeWorkItem, col: Int32,
    is_classification: Bool,
) -> FeatureRange:
    """The range pass's cell: the float min and max (the binned
    `IDN_ET_BINNED` arm was deleted by lane trees-small, 2026-10-07)."""
    return node_feature_min_max(dataset, item, col)


def _exact_rescue_columns_host(
    dataset: Dataset, item: NodeWorkItem,
    is_classification: Bool,
) raises -> List[Int32]:
    """`rescue_columns` over `_exact_extent` (the device survey runs the
    same range pass the search runs)."""
    var out = List[Int32]()
    for col in range(Int(dataset.n)):
        var extent = _exact_extent(dataset, item, Int32(col), is_classification)
        if not node_feature_is_constant(extent, item.instances.count):
            out.append(Int32(col))
    return out^


def _exact_all_constant_host(
    dataset: Dataset, item: NodeWorkItem, colids: List[Int32],
    is_classification: Bool,
) -> Bool:
    """`node_nonconstant_flag_kernel`'s per-node answer: no sampled column
    varied on this node's rows."""
    for ci in range(len(colids)):
        var extent = _exact_extent(dataset, item, colids[ci], is_classification)
        if not node_feature_is_constant(extent, item.instances.count):
            return False
    return True


def set_leaf_predictions_exact_host(
    dataset: Dataset,
    labels_q: MutPointer[Int32, MutAnyOrigin],
    mut tree: TreeMetaDataNode[DType.float32],
    node_instances: List[InstanceRange],
    inv_scale: Float32,
    is_classification: Bool,
) raises:
    """`leaf_kernel`'s arithmetic on the host, per leaf (`leaf_values_host`
    is the same pass over pointers): integer class counts or the integer
    sum of the quantized labels, then `Float32(count_c) / Float32(total)`
    for a classifier and `ftz(Float32(sum) / Float32(seen) * inv_scale)`
    for a regressor. An internal node's slot keeps the zeros
    (`builder.cuh:582`'s memset, DEVIATION 471's zero_fill)."""
    var n_nodes = tree.num_nodes()
    if len(node_instances) != n_nodes:
        raise Error(
            "set_leaf_predictions_exact: "
            + String(n_nodes)
            + " nodes but "
            + String(len(node_instances))
            + " instance ranges"
        )
    var k = Int(tree.num_outputs)
    tree.vector_leaf = List[Float32](length=n_nodes * k, fill=Float32(0.0))
    for node_id in range(n_nodes):
        if not tree.sparsetree[node_id].IsLeaf():
            continue
        var rng = node_instances[node_id]
        comptime if T14 and not T14_EXACT:
            if not is_classification:
                tree.vector_leaf[node_id*k] = et_leaf_moment_host(dataset.row_ids,labels_q,Int(rng.begin),Int(rng.count),inv_scale)
                continue
        var acc = List[Int32](length=k, fill=Int32(0))
        var seen = Int32(0)
        for i in range(Int(rng.begin), Int(rng.begin) + Int(rng.count)):
            var row = Int(dataset.row_ids[unsafe_offset=i])
            var lab = Int(labels_q[unsafe_offset=row])
            if is_classification:
                if lab >= 0 and lab < k:
                    acc[lab] += Int32(1)
            else:
                acc[0] += Int32(lab)
            seen += 1
        var base = node_id * k
        if is_classification:
            var total = Int32(0)
            for c in range(k):
                total += acc[c]
            for c in range(k):
                tree.vector_leaf[base + c] = Float32(Int(acc[c])) / Float32(
                    Int(total)
                )
        else:
            for c in range(k):
                tree.vector_leaf[base + c] = ftz(
                    Float32(Int(acc[c])) / Float32(Int(seen)) * inv_scale
                )


def train_tree_exact(
    dataset: Dataset,
    labels_q: MutPointer[Int32, MutAnyOrigin],
    params: DecisionTreeParams,
    tree_id: Int32,
    seed: UInt64,
    is_classification: Bool,
    n_acc: Int,
    inv_scale: Float32,
) raises -> TreeMetaDataNode[DType.float32]:
    """cpu4-forest: the HOST COLUMN body lives in `train_tree_exact_host` (the checker's
    host-column naming); this name is kept for its callers."""
    return train_tree_exact_host(
        dataset, labels_q, params, tree_id, seed, is_classification, n_acc,
        inv_scale,
    )


def train_tree_exact_host(
    dataset: Dataset,
    labels_q: MutPointer[Int32, MutAnyOrigin],
    params: DecisionTreeParams,
    tree_id: Int32,
    seed: UInt64,
    is_classification: Bool,
    n_acc: Int,
    inv_scale: Float32,
) raises -> TreeMetaDataNode[DType.float32]:
    """One tree grown as the device grows it, on the host: the block comment
    above. `labels_q` is the device's label plane (class ids for a
    classifier, `quantize_labels_host`'s fixed point for a regressor; `n_acc`
    is the class count or 1; `inv_scale` is `Float32(1 / scale)` or 1)."""
    validity_check(params)
    if params.max_leaf_nodes != -1:
        # DEVIATION 466's growth mode, restated on the exact key
        # (et-clf-entropy-bestfirst, 2026-09-15): see the function below.
        return train_tree_exact_bestfirst(
            dataset, labels_q, params, tree_id, seed, is_classification,
            n_acc, inv_scale,
        )
    if Int(dataset.num_outputs) != n_acc:
        raise Error(
            "train_tree_exact: dataset.num_outputs is "
            + String(dataset.num_outputs)
            + " but n_acc is "
            + String(n_acc)
        )
    var k = n_sampled_cols_for(params, dataset.n)
    var queue = NodeQueue[DType.float32](
        params, dataset.n_sampled_rows, Int32(n_acc), tree_id
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
            var split = _exact_node_split(
                dataset, labels_q, item, my_colids, n_acc, is_classification,
                params.split_criterion, params.min_samples_leaf, seed, tree_id,
            )
            # DEVIATION 205 as the device loop keys it: every sampled column
            # constant on a non-empty node, then one non-constant column
            # picked by `rescue_pick` over the ascending list and searched
            # alone.
            if (
                item.instances.count > 0
                and _exact_all_constant_host(
                    dataset, item, my_colids, is_classification
                )
            ):
                var nonconst = _exact_rescue_columns_host(
                    dataset, item, is_classification
                )
                if len(nonconst) > 0:
                    var u = rescue_pick(
                        rescue_key(seed, tree_id, UInt32(Int(item.idx))),
                        len(nonconst),
                    )
                    var one = List[Int32]()
                    one.append(nonconst[u])
                    split = _exact_node_split(
                        dataset, labels_q, item, one, n_acc, is_classification,
                        params.split_criterion, params.min_samples_leaf, seed,
                        tree_id,
                    )
            splits.append(split)
            if not split_not_valid(
                split,
                params.min_impurity_decrease,
                params.min_samples_leaf,
                item.instances.count,
            ):
                partition_samples(dataset, split, item)
        queue.push(work_items, splits)
    var tree = queue.get_tree()
    set_leaf_predictions_exact_host(
        dataset, labels_q, tree, queue.node_instances, inv_scale, is_classification
    )
    return tree^


def _exact_search_one(
    dataset: Dataset,
    labels_q: MutPointer[Int32, MutAnyOrigin],
    item: NodeWorkItem,
    params: DecisionTreeParams,
    tree_id: Int32,
    seed: UInt64,
    is_classification: Bool,
    n_acc: Int,
    k: Int32,
) raises -> Split:
    """One node's search as `search_batch` answers it for member `i` of any
    batch, on the host: the column sample keyed by (seed, tree, node)
    (`sample_features` over a one-item batch draws the columns the same item
    draws in a wide one), `_exact_node_split`, then DEVIATION 205's rescue
    keyed as the device keys it. `train_tree_exact`'s inner loop, one item
    at a time."""
    var colids = List[Int32](length=Int(k), fill=Int32(0))
    var one_item = List[NodeWorkItem]()
    one_item.append(item)
    _ = sample_features(colids, one_item, tree_id, seed, Int(dataset.n), Int(k))
    var split = _exact_node_split(
        dataset, labels_q, item, colids, n_acc, is_classification,
        params.split_criterion, params.min_samples_leaf, seed, tree_id,
    )
    if item.instances.count > 0 and _exact_all_constant_host(
        dataset, item, colids, is_classification
    ):
        var nonconst = _exact_rescue_columns_host(
            dataset, item, is_classification
        )
        if len(nonconst) > 0:
            var u = rescue_pick(
                rescue_key(seed, tree_id, UInt32(Int(item.idx))), len(nonconst)
            )
            var one = List[Int32]()
            one.append(nonconst[u])
            split = _exact_node_split(
                dataset, labels_q, item, one, n_acc, is_classification,
                params.split_criterion, params.min_samples_leaf, seed, tree_id,
            )
    return split


def train_tree_exact_bestfirst(
    dataset: Dataset,
    labels_q: MutPointer[Int32, MutAnyOrigin],
    params: DecisionTreeParams,
    tree_id: Int32,
    seed: UInt64,
    is_classification: Bool,
    n_acc: Int,
    inv_scale: Float32,
) raises -> TreeMetaDataNode[DType.float32]:
    """One tree grown BEST-FIRST as the device grows it (DEVIATION 466), on
    the host and on the exact key (the CPU training lane,
    et-clf-entropy-bestfirst, 2026-09-15).

    The device driver (`train_forest_classification_device_timed` and its
    regression twin, the `if bestfirst:` arms) runs, per tree, a cycle of
    SEARCH the nodes admitted-but-unsearched (the roots on cycle 0, the
    children of the last expansion after), ADMIT each through
    `bestfirst_admit` in batch order, POP this tree's best record, PARTITION
    its row range (`partition_*_kernel`, a stable split of the range on
    `value <= quesval`, which `partition_samples` is on the host), then
    EXPAND it through `bestfirst_expand`, whose returned children are the
    next cycle's search. Every draw is keyed by (seed, tree, node, column),
    the frontier and its total order are per tree, and a node's rows move
    only under its own or an ancestor's partition, so other trees sharing
    the merged batch change nothing; this loop is that cycle for one tree.
    The search is `_exact_search_one` (the key the device's reduction
    orders by, not sklearn's splitter, which `train_classification_bestfirst`
    uses and which is why that oracle cannot be the CPU column) and the leaf
    pass is `set_leaf_predictions_exact`, as in `train_tree_exact`."""
    if Int(dataset.num_outputs) != n_acc:
        raise Error(
            "train_tree_exact_bestfirst: dataset.num_outputs is "
            + String(dataset.num_outputs)
            + " but n_acc is "
            + String(n_acc)
        )
    if params.max_batch_size < 2:
        # The device's refusal, in its words (DEVIATION 469).
        raise Error(
            "max_leaf_nodes needs max_batch_size >= 2: a best-first cycle"
            " searches both children of the node it expands, and a batch of"
            " one cannot hold them (DEVIATION 469). Got max_batch_size "
            + String(params.max_batch_size)
        )
    var k = n_sampled_cols_for(params, dataset.n)
    var queue = NodeQueue[DType.float32](
        params, dataset.n_sampled_rows, Int32(n_acc), tree_id
    )
    var pending = queue.bestfirst_seed()
    while True:
        for i in range(len(pending)):
            var split = _exact_search_one(
                dataset, labels_q, pending[i], params, tree_id, seed,
                is_classification, n_acc, k,
            )
            _ = queue.bestfirst_admit(pending[i], split, tree_id)
        if not queue.bestfirst_can_pop():
            break
        var rec = queue.bestfirst_pop()
        partition_samples(dataset, rec.split, rec.item)
        pending = queue.bestfirst_expand(rec.item, rec.split)
    var tree = queue.get_tree()
    set_leaf_predictions_exact_host(
        dataset, labels_q, tree, queue.node_instances, inv_scale, is_classification
    )
    return tree^


# ==========================================================================
# DEVIATION 466's HOST ARM. The oracles the device best-first driver is
# checked against, in the same relation `train_classification` bears to
# `train_classification_device` (`CONTRIBUTING.md` (Algorithms and references): an oracle, not a
# CPU fallback).
# ==========================================================================


# ==========================================================================
# THE DEVICE PATH
# ==========================================================================
# `Builder::doSplit` (`builder.cuh:379-494`) with its kernels enqueued, and
# `Builder::train`'s loop (`:344-359`) around it. Until this function existed
# every device kernel in this lane was UNWIRED -- built, checked per cell, and
# reached by nothing but its own check, which rule 3 says is not done.
#
# THEIR HOST/DEVICE SPLIT IS IMPLEMENTED, NOT RE-DECIDED. cuML's node queue is a
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

    Matches the reference, including the order the three terms
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
    `CONTRIBUTING.md`: "If they do something on the GPU in the control
    plane, we do it on the GPU. If they keep a decision on the device so
    the host never learns it, we keep it on the device." cuML computes
    `GainPerSplit` inside `computeSplitKernel` and the host never sees a
    per-candidate gain. The first closure of 183 copied `status`,
    `n_total` and `acc_total` back per level -- `n_cells * (2 +
    n_classes)` ints that the reference never moves -- and formed the gain
    in `Float64` on the host. It was correct and it was the wrong shape,
    and the commit that introduced it argued "the cheaper fix moved less
    code", which optimises for the author of the change rather than for the implementation.

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
    columns_per_node: Int32 = 1,
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
    reduction read a layout it does not own. Both couple two implemented files
    to each other through a shape neither reference has.

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
    # DEVIATION 5610: Poisson / Gamma / InverseGaussian take entropy's route.
    var deviance = criterion_in == CRITERION_POISSON or criterion_in == CRITERION_GAMMA or (
        criterion_in == CRITERION_INVERSE_GAUSSIAN
    )
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    var parent_terms = InlineArray[Float32,32](fill=Float32(0))
    comptime if T06:
        if entropy and n_classes <= 32:
            var node = idx
            var width = Int(columns_per_node)
            var start = node*width
            if start >= n_cells:
                return
            n_cells = min(n_cells,start+width)
            idx = start
            stride = 1
            for feature in range(start,n_cells):
                if in_status[unsafe_offset=feature] == SCORE_STATUS_SCORED:
                    var inv_len = ftz(Float32(1)/Float32(in_n_total[unsafe_offset=feature]))
                    for c in range(n_classes):
                        var count = in_acc_total[unsafe_offset=feature*n_classes+c]
                        if count != 0:
                            var value = ftz(Float32(count)*inv_len)
                            var term = ftz(value*tree_log(value))
                            parent_terms[c] = ftz(term/tree_log(Float32(2)))
                    break
    # T06 preserves the same class fold and exact parent statements; only the
    # per-node transform lifetime changes. Cached terms are never weighted.
    # NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
    while idx < n_cells:
        if in_status[unsafe_offset=idx] == SCORE_STATUS_SCORED:
            cand_quesval[unsafe_offset=idx] = in_threshold[unsafe_offset=idx]
            cand_colid[unsafe_offset=idx] = colids[unsafe_offset=idx]
            cand_nleft[unsafe_offset=idx] = in_n_left[unsafe_offset=idx]
            if entropy:
                # DEVIATION 459: the float gain is the metric AND the key.
                var g = Float32(0)
                if T06 and n_classes <= 32:
                    var objective = EntropyObjectiveFunction[DType.float32](Int32(n_classes),min_samples_leaf)
                    g = objective.GainPerSplitCached(
                        in_acc_left.unsafe_offset(idx*n_classes).unsafe_bitcast[CountBin](),
                        in_acc_total.unsafe_offset(idx*n_classes).unsafe_bitcast[CountBin](),
                        in_n_total[unsafe_offset=idx],in_n_left[unsafe_offset=idx],parent_terms,
                    )
                else:
                    g = entropy_gain_per_split(in_acc_left,in_acc_total,idx*n_classes,n_classes,
                        in_n_total[unsafe_offset=idx],in_n_left[unsafe_offset=idx],min_samples_leaf)
                cand_metric[unsafe_offset=idx] = g
                cand_num[unsafe_offset=idx] = float_gain_key(g)
                cand_den[unsafe_offset=idx] = Int64(1)
            elif deviance:
                # DEVIATION 5610: the deviance gain over the cell's scaled
                # label sums is the metric AND the key (`den = 1`).
                var g = regression_deviance_gain(
                    in_acc_left[unsafe_offset=idx],
                    in_acc_total[unsafe_offset=idx],
                    in_n_left[unsafe_offset=idx],
                    in_n_total[unsafe_offset=idx],
                    criterion_in,
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
                comptime if T14 and not T14_EXACT:
                    if criterion_in == CRITERION_MSE:
                        cand_metric[unsafe_offset=idx] = max(Float32(0),ftz(Float32(2)*balanced_mse_gain(
                            Float32(in_n_total[unsafe_offset=idx]),Float32(in_n_left[unsafe_offset=idx]),
                            Float32(in_acc_total[unsafe_offset=idx]),Float32(in_acc_left[unsafe_offset=idx]))))
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


comptime ET_SMALL_NODE_TPB = T05_SMALL_NODE_ROWS
"""T05 one row per lane, two exact class-count planes of at most 32 Int32s.

The capacity is a portable 128-thread block, not a dataset or tree-size gate.
Every admitted row stays in a register from its range read through threshold
scoring. Integer shared counts take 256 bytes; range/threshold scratch is
bounded independently of feature count. Larger nodes keep the incumbent path.
NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
"""


def small_node_raw_split_kernel[CLASSIFICATION: Bool](
    cand_quesval: MutPointer[Float32, MutAnyOrigin],
    cand_colid: MutPointer[Int32, MutAnyOrigin],
    cand_metric: MutPointer[Float32, MutAnyOrigin],
    cand_nleft: MutPointer[Int32, MutAnyOrigin],
    cand_num: MutPointer[Int64, MutAnyOrigin],
    cand_den: MutPointer[Int64, MutAnyOrigin],
    cand_valid: MutPointer[Int32, MutAnyOrigin],
    nonconstant: MutPointer[Int32, MutAnyOrigin],
    out_min: MutPointer[Float32, MutAnyOrigin],
    out_max: MutPointer[Float32, MutAnyOrigin],
    out_missing: MutPointer[Int32, MutAnyOrigin],
    out_draw: MutPointer[Float32, MutAnyOrigin],
    data: MutPointer[Float32, MutAnyOrigin],
    rows: MutPointer[Int32, MutAnyOrigin],
    labels: MutPointer[Int32, MutAnyOrigin],
    items: MutPointer[NodeWorkItem, MutAnyOrigin],
    columns: MutPointer[Int32, MutAnyOrigin],
    tree_ids: MutPointer[Int32, MutAnyOrigin],
    n_rows: Int32,
    n_columns_per_node: Int32,
    n_acc: Int32,
    seed: UInt64,
    min_samples_leaf: Int32,
    criterion: Int32,
):
    """T05/C45: one raw-value range, random draw, statistic and score block.

    Called after ordinary conversion, which initializes invalid small-node
    candidates from the deliberately empty separate range cells. This block
    replaces those candidates and supplies the ranges/draw consumed by tracing.
    Statistics remain shared: no intermediate class/moment arrays are written.
    Feature sampling and the following canonical split reducer are unchanged.
    NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
    """
    var nid = Int(block_idx.x)
    var feature = Int(block_idx.y)
    var tid = Int(thread_idx.x)
    var length = Int(items[unsafe_offset=nid].instances.count)
    if length <= 0 or length > ET_SMALL_NODE_TPB or n_acc < 1 or n_acc > 32:
        return
    var begin = Int(items[unsafe_offset=nid].instances.begin)
    var slot = nid * Int(n_columns_per_node) + feature
    var col = Int(columns[unsafe_offset=slot])
    if tid == 0:
        cand_quesval[unsafe_offset=slot] = Float32.MIN_FINITE
        cand_colid[unsafe_offset=slot] = Int32(-1)
        cand_metric[unsafe_offset=slot] = Float32.MIN_FINITE
        cand_nleft[unsafe_offset=slot] = Int32(0)
        cand_num[unsafe_offset=slot] = Int64(0)
        cand_den[unsafe_offset=slot] = Int64(0)
        cand_valid[unsafe_offset=slot] = Int32(0)

    # At most one row per lane, also at most one row per lane in the
    # incumbent range block. Its key-space min/max, missing sentinel and
    # signed-zero choices therefore need no new floating reduction graph.
    var row = 0
    var value = Float32(0)
    var lo_key = RANGE_KEY_MIN_SEED
    var hi_key = RANGE_KEY_MAX_SEED
    var missing = Int32(0)
    if tid < length:
        row = Int(rows[unsafe_offset=begin+tid])
        value = data[unsafe_offset=col*Int(n_rows)+row]
        if value != value:
            missing = Int32(1)
        else:
            lo_key = range_key(value)
            hi_key = lo_key
    var minimum = block_min[block_size=ET_SMALL_NODE_TPB](lo_key)
    barrier()
    var maximum = block_max[block_size=ET_SMALL_NODE_TPB](hi_key)
    barrier()
    var missing_total = block_sum[block_size=ET_SMALL_NODE_TPB](missing)
    barrier()
    var threshold = stack_allocation[1,Float32,address_space=AddressSpace.SHARED]()
    var active = stack_allocation[1,Int32,address_space=AddressSpace.SHARED]()
    if tid == 0:
        var lo = range_unkey(minimum)
        var hi = range_unkey(maximum)
        if lo > hi:
            lo = Float32(1)
            hi = Float32(-1)
        out_min[unsafe_offset=slot] = lo
        out_max[unsafe_offset=slot] = hi
        out_missing[unsafe_offset=slot] = missing_total
        var extent = FeatureRange(lo,hi,missing_total)
        var constant = node_feature_is_constant(extent,Int32(length))
        if not constant:
            _ = Atomic.fetch_add(nonconstant.unsafe_offset(nid),Int32(1))
        active[unsafe_offset=0] = Int32(0)
        threshold[unsafe_offset=0] = Float32(0)
        if missing_total == 0 and not constant:
            var key = key_for(seed,tree_ids[unsafe_offset=nid].cast[DType.uint32](),
                UInt32(Int(items[unsafe_offset=nid].idx)),UInt32(col))
            threshold[unsafe_offset=0] = draw_threshold_device(key,extent)
            out_draw[unsafe_offset=slot] = threshold[unsafe_offset=0]
            active[unsafe_offset=0] = Int32(1)
    barrier()
    if active[unsafe_offset=0] == 0:
        return

    var histogram = stack_allocation[64,Int32,address_space=AddressSpace.SHARED]()
    if tid < 64:
        histogram[unsafe_offset=tid] = Int32(0)
    barrier()
    var left = Int32(0)
    if tid < length:
        var label = labels[unsafe_offset=row]
        var goes_left = value <= threshold[unsafe_offset=0]
        left = Int32(1) if goes_left else Int32(0)
        comptime if CLASSIFICATION:
            if label >= 0 and label < n_acc:
                _ = Atomic.fetch_add(histogram.unsafe_offset(32+Int(label)),Int32(1))
                if goes_left:
                    _ = Atomic.fetch_add(histogram.unsafe_offset(Int(label)),Int32(1))
        else:
            _ = Atomic.fetch_add(histogram.unsafe_offset(32),label)
            if goes_left:
                _ = Atomic.fetch_add(histogram,label)
    var left_count = block_sum[block_size=ET_SMALL_NODE_TPB](left)
    barrier()
    if tid != 0:
        return
    var right_count = Int32(length)-left_count
    if left_count < min_samples_leaf or right_count < min_samples_leaf or left_count == 0 or right_count == 0:
        return
    comptime if CLASSIFICATION:
        for c in range(Int(n_acc)):
            if histogram[unsafe_offset=32+c] == Int32(length):
                return

    # Exactly node_feature_score_finalize_kernel's integer key, then the
    # same score helpers used by score_to_candidate_kernel. The canonical
    # feature/tie reduction stays a separate consumer of these winners.
    var numerator = Int64(0)
    var denominator = Int64(0)
    comptime if CLASSIFICATION:
        var sq_left = Int64(0)
        var sq_right = Int64(0)
        for c in range(Int(n_acc)):
            var lv = Int64(Int(histogram[unsafe_offset=c]))
            var rv = Int64(Int(histogram[unsafe_offset=32+c]-histogram[unsafe_offset=c]))
            sq_left += lv*lv
            sq_right += rv*rv
        var nl = Int64(Int(left_count))
        var nr = Int64(Int(right_count))
        var shift = Int64(classification_key_shift(length))
        numerator = (sq_left >> shift)*nr+(sq_right >> shift)*nl
        denominator = nl*nr
    else:
        if not regression_key(Int64(Int(histogram[unsafe_offset=0])),
            Int64(Int(histogram[unsafe_offset=32])),Int(left_count),
            Int(right_count),length,SCORE_SAB_NONE,numerator,denominator):
            return
    # The canonical helpers accept generic device pointers. Shared storage is
    # live through this finalization, after the block barrier above.
    var gain = Float32(0)
    if criterion == CRITERION_ENTROPY:
        # T06's parent cache belongs to the separate multi-feature score
        # pass. A fused feature already consumes its exact counts once.
        gain = entropy_gain_per_split(histogram.unsafe_address_space_cast[AddressSpace.GENERIC]().unsafe_origin_cast[MutAnyOrigin](),
            histogram.unsafe_offset(32).unsafe_address_space_cast[AddressSpace.GENERIC]().unsafe_origin_cast[MutAnyOrigin](),
            0,Int(n_acc),Int32(length),left_count,min_samples_leaf)
        numerator = float_gain_key(gain)
        denominator = Int64(1)
    elif criterion == CRITERION_POISSON or criterion == CRITERION_GAMMA or criterion == CRITERION_INVERSE_GAUSSIAN:
        gain = regression_deviance_gain(histogram[unsafe_offset=0],
            histogram[unsafe_offset=32],left_count,Int32(length),criterion)
        numerator = float_gain_key(gain)
        denominator = Int64(1)
    else:
        gain = gain_per_split(histogram.unsafe_address_space_cast[AddressSpace.GENERIC]().unsafe_origin_cast[MutAnyOrigin](),
            histogram.unsafe_offset(32).unsafe_address_space_cast[AddressSpace.GENERIC]().unsafe_origin_cast[MutAnyOrigin](),
            0,Int(n_acc),Int32(length),left_count,min_samples_leaf)
        comptime if T14 and not T14_EXACT and not CLASSIFICATION:
            if criterion == CRITERION_MSE:
                gain = max(Float32(0),ftz(Float32(2)*balanced_mse_gain(
                    Float32(length),Float32(left_count),
                    Float32(histogram[unsafe_offset=32]),Float32(histogram[unsafe_offset=0]))))
    cand_quesval[unsafe_offset=slot] = threshold[unsafe_offset=0]
    cand_colid[unsafe_offset=slot] = Int32(col)
    cand_metric[unsafe_offset=slot] = gain
    cand_nleft[unsafe_offset=slot] = left_count
    cand_num[unsafe_offset=slot] = numerator
    cand_den[unsafe_offset=slot] = denominator
    cand_valid[unsafe_offset=slot] = Int32(1)


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
    one `n_rows` H2D copy per tree the reference does not have, and rule 2
    is about the SHAPE and not only about decisions. A reference audit of
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
    `core.philox.launch_uniform_int` -- the RF lane's implementation of RAFT's
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
    # The ExtraTrees bootstrap-locality sort (MOJOLEARN_TREES_T09) was deleted
    # by lane/grid-prune (2026-10-07): unreachable on the board (et runs
    # bootstrap=False and returns above) and it put a host synchronize inside
    # the GPU fit. Recoverable at main ab554bb4a.
    _ = d_row_ids.unsafe_ptr()


def transpose_to_row_major_kernel(
    out_rm: MutPointer[Float32, MutAnyOrigin],
    in_cm: MutPointer[Float32, MutAnyOrigin],
    n_rows: Int32,
    n_cols: Int32,
):
    """`out_rm[r * n_cols + c] = in_cm[c * n_rows + r]`, one element per
    thread, writes coalesced. A pure move."""
    var o = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var nr = Int(n_rows)
    var nc = Int(n_cols)
    if o >= nr * nc:
        return
    var r = o // nc
    var c = o - r * nc
    out_rm[unsafe_offset=o] = in_cm[unsafe_offset = c * nr + r]


#: FAST on Apple, REGRESSION with the row-major tiled search on wide data:
#: the range and score passes read X as 16-bit codes over up to ET_BINS
#: quantile borders per feature (half the bytes per row). The range pass
#: publishes the borders of the node's lowest and highest codes, the
#: threshold is drawn in VALUE space from them as before, and every
#: threshold is snapped down to a border (`et_snap_code`): the score pass
#: compares codes against the border's code and `et_code_threshold_kernel`
#: stores the border itself, so the score, the partition and prediction
#: split the rows the same way (code(x) <= c  <=>  x <= q[c]). Measured on
#: the M4 at 1M rows: istellareg 86 -> 67 s, year 23.2 -> 18.0 s, RMSE equal
#: to float X (8-bit codes from 1024 sampled rows, or thresholds drawn in
#: code space, both moved RMSE). `-D MOJOLEARN_ET_BINNED_OFF` keeps float X.
#: The IDENTICAL arm of this search (`MOJOLEARN_IDN_ET_BINNED_U16`, fam2-forests
#: 2026-10-04) was a measured loser (2026-10-05 full-harness A/B: NVIDIA 1.064x,
#: AMD 1.025x slower, RMSE worse on Istella and Year; evidence
#: experiments/identical_speed/results/20261005/forest-et-decision/board.json)
#: and was DELETED with its host column (`HostBins`, host_binned.mojo) by lane
#: trees-small 2026-10-07 (docs/apple-fast/EXPERIMENTS.md row). FAST only now.
comptime ET_BINNED_REG = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_ET_BINNED_OFF"]()
)
comptime ET_BINS = ET_QSTRIDE
comptime ET_CODE = DType.uint16
#: FAST + Apple: LEGACY, default OFF: binning was taken only at n_cols >= 64, chosen
#: between taxi (16 columns, slower) and istella (220, faster). Removed as
#: benchmark-tuned on 2026-10-04: binning now applies at every width;
#: UNMEASURED.
comptime ET_BINNED_LEGACY_NARROW = is_defined["MOJOLEARN_LEGACY_NARROW_ET_BINNED"]()
comptime ET_BINNED_MIN_COLS = 64 if ET_BINNED_LEGACY_NARROW else 1
comptime ET_CODE_TILE = 8
"""Features per block for the code passes (M4 istellareg: 4 -> 53 s, 8 -> 49 s, 16 -> 67 s, 32 -> 79 s)."""


# THE DEVICE RESCUE (fam2-forests 2026-10-04; unconditional since cpu3-trees,
# in IDENTICAL and FAST, every vendor): DEVIATION 205's rescue column is
# picked ON THE DEVICE. `ident_colids_kernel` writes the survey's columns,
# `rescue_pick_kernel` runs the constant test over the survey's cells in
# ascending column order with `rescue_pick`'s keyed draw, and the rescued
# search reads its column from `d_colids`; the host reads back one Int32 per
# surveyed node (riding the rescued search's own drain). The rescued search
# covers every surveyed node (a node with no varying column searches column
# 0, constant on its rows by the survey, and its split is left as the first
# search found it). Same columns, same splits as the host column. The old
# host walk (download `3 * n_sub * n_cols` cells, pick on the host, upload
# the columns) and its `MOJOLEARN_IDN_ET_RESCUE_DEVICE_OFF` arm are removed.


def ident_colids_kernel(
    out_colids: MutPointer[Int32, MutAnyOrigin],
    n_cells: Int32,
    n_cols: Int32,
):
    """`out_colids[node * n_cols + c] = c`: the survey's column table."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_cells):
        return
    out_colids[unsafe_offset=i] = Int32(i % Int(n_cols))


def rescue_pick_kernel(
    out_pick: MutPointer[Int32, MutAnyOrigin],
    out_colids: MutPointer[Int32, MutAnyOrigin],
    in_min: MutPointer[Float32, MutAnyOrigin],
    in_max: MutPointer[Float32, MutAnyOrigin],
    in_n_missing: MutPointer[Int32, MutAnyOrigin],
    work_items: MutPointer[NodeWorkItem, MutAnyOrigin],
    tree_ids: MutPointer[Int32, MutAnyOrigin],
    n_sub: Int32,
    n_cols: Int32,
    seed: UInt64,
):
    """THE DEVICE RESCUE: one thread per surveyed node. Counts the
    node's non-constant columns over the survey's cells (`n_sub x n_cols`,
    ascending column order, `node_feature_is_constant`), draws
    `rescue_pick`'s index from `rescue_key(seed, tree, node)` and stores
    that column in `out_pick[node]` and `out_colids[node]` (the rescued
    search's `k = 1` column table). A node with no varying column stores
    -1 in `out_pick` and column 0 in `out_colids`."""
    var j = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if j >= Int(n_sub):
        return
    var nc = Int(n_cols)
    var rows = work_items[unsafe_offset=j].instances.count
    var count = 0
    for c in range(nc):
        var idx = j * nc + c
        var extent = FeatureRange(
            in_min[unsafe_offset=idx],
            in_max[unsafe_offset=idx],
            in_n_missing[unsafe_offset=idx],
        )
        if not node_feature_is_constant(extent, rows):
            count += 1
    if count == 0:
        out_pick[unsafe_offset=j] = Int32(-1)
        out_colids[unsafe_offset=j] = Int32(0)
        return
    var key = rescue_key(
        seed,
        tree_ids[unsafe_offset=j],
        UInt32(Int(work_items[unsafe_offset=j].idx)),
    )
    var gen = PCGenerator(key.seed, key.subsequence, UInt64(0))
    var u = Int(uniform_int_u32(gen, UInt32(0), UInt32(count)))
    var seen = 0
    var pick = Int32(0)
    for c in range(nc):
        var idx = j * nc + c
        var extent = FeatureRange(
            in_min[unsafe_offset=idx],
            in_max[unsafe_offset=idx],
            in_n_missing[unsafe_offset=idx],
        )
        if not node_feature_is_constant(extent, rows):
            if seen == u:
                pick = Int32(c)
            seen += 1
    out_pick[unsafe_offset=j] = pick
    out_colids[unsafe_offset=j] = pick


def et_code_threshold_kernel(
    q: MutPointer[Float32, MutAnyOrigin],
    c: MutPointer[Int32, MutAnyOrigin],
    quant: MutPointer[Float32, MutAnyOrigin],
    nbins: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_in):
        return
    var col = Int(c[i])
    if col < 0:
        return
    var t = et_snap_code(quant + col * ET_BINS, nbins[col], q[i])
    if t < 0:
        t = 0
    # The top code also holds every value above the sampled maximum
    # (`lower_bound_aspace` clamps), so `code <= top` is "every row": +inf.
    if t >= Int(nbins[col]) - 1:
        q[i] = Float32.MAX
        return
    q[i] = quant[col * ET_BINS + t]


def et_bin_rows_kernel(
    data: MutPointer[Float32, MutAnyOrigin],
    codes: MutPointer[Scalar[ET_CODE], MutAnyOrigin],
    quant: MutPointer[Float32, MutAnyOrigin],
    nbins: MutPointer[Int32, MutAnyOrigin],
    n_rows: Int32,
    n_cols: Int32,
):
    """Column-major X to row-major codes: `lower_bound` over the column's
    borders, clamped to the top code (RandomForest's `bin_dataset_kernel`,
    wider codes)."""
    var col = Int(block_idx.y)
    var nb = nbins[col]
    var qc = quant + col * ET_BINS
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_rows):
        return
    var v = data[col * Int(n_rows) + i]
    var lo = Int32(0)
    var hi = nb - 1
    while lo < hi:
        var mid = (lo + hi) // 2
        if qc[Int(mid)] >= v:
            hi = mid
        else:
            lo = mid + 1
    codes[i * Int(n_cols) + col] = Scalar[ET_CODE](Int(lo))


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
    var d_data_rm: DeviceBuffer[DType.float32]
    """A row-major copy of X for the FAST tiled search kernels, built by
    `ensure_row_major` only when a fit samples at least half the features;
    one element otherwise."""
    var has_rm: Bool
    var d_bins_rm: DeviceBuffer[ET_CODE]
    """ET_BINNED_REG: X as row-major 8-bit quantile codes (one element
    until `ensure_binned`)."""
    var d_quant: DeviceBuffer[DType.float32]
    """ET_BINNED_REG: the per-feature borders, `n_cols x ET_BINS`."""
    var d_nbins: DeviceBuffer[DType.int32]
    var has_bins: Bool

    @always_inline
    def bins_active(self) -> Bool:
        return self.has_bins

    def ensure_binned(mut self, ctx: DeviceContext) raises:
        """Build the row-major codes and their borders (RandomForest's
        `compute_quantiles`, then `et_bin_rows_kernel`): code c means
        `q[col][c-1] < x <= q[col][c]`, the top code also every larger x."""
        comptime if not ET_BINNED_REG:
            return
        if self.has_bins:
            return
        var nr = Int(self.n_rows)
        var nc = Int(self.n_cols)
        var qr = compute_quantiles(ctx, self.d_data, ET_BINS, nr, nc)
        self.d_quant = ctx.enqueue_create_buffer[DType.float32](nc * ET_BINS)
        self.d_nbins = ctx.enqueue_create_buffer[DType.int32](nc)
        ctx.enqueue_copy(
            dst_buf=self.d_quant,
            src_buf=qr.quantiles_array.create_sub_buffer[DType.float32](
                0, nc * ET_BINS
            ),
        )
        ctx.enqueue_copy(
            dst_buf=self.d_nbins,
            src_buf=qr.n_bins_array.create_sub_buffer[DType.int32](0, nc),
        )
        self.d_bins_rm = ctx.enqueue_create_buffer[ET_CODE](nr * nc)
        ctx.enqueue_function[et_bin_rows_kernel](
            self.d_data.unsafe_ptr(),
            self.d_bins_rm.unsafe_ptr(),
            self.d_quant.unsafe_ptr(),
            self.d_nbins.unsafe_ptr(),
            Int32(nr),
            Int32(nc),
            grid_dim=(ceildiv(nr, 256), nc, 1),
            block_dim=(256, 1, 1),
        )
        ctx.synchronize()
        _ = qr^
        self.has_bins = True

    def ensure_row_major(mut self, ctx: DeviceContext, k: Int) raises:
        """Build `d_data_rm` on the device when this build has a row-major
        consumer and the fit samples `2k >= n_cols` features: the tiled
        kernels read one row's sampled features from one row, which only
        saves traffic when most of the row is sampled (Apple M4, 1M rows:
        taxi and Istella-S regression 0.34 and 0.48, Istella-S
        classification at k = 15 of 220 slower, 1.81)."""
        comptime if not ET_RM_DATA:
            return
        if self.has_rm:
            return
        # ET_RM_NARROW: a row whose floats fit one cache line (ET_RM_LINE_BYTES)
        # also takes the row-major copy when a quarter of it is sampled.
        var narrow = False
        comptime if ET_RM_NARROW:
            comptime if ET_RM_NARROW_GENERAL:
                # FAST: no narrow term. A one-line row is still read whole
                # by the row-major copy (n_cols floats to use k), so the
                # byte rule is the wide one below, `2k >= n_cols`: at least
                # half the fetched bytes used. The quarter gate (`4k >=
                # n_cols`) landed on taxi's k = 4 of 16; removed as
                # benchmark-tuned on 2026-10-04, replacement UNMEASURED.
                narrow = False
            else:
                # IDENTICAL: main's line rule (lane/no-dim-idn,
                # ET_RM_LINE_BYTES) with the quarter gate.
                narrow = (
                    Int(self.n_cols) * 4 <= ET_RM_LINE_BYTES
                    and 4 * k >= Int(self.n_cols)
                )
        if 2 * k < Int(self.n_cols) and not narrow:
            return
        var nr = Int(self.n_rows)
        var nc = Int(self.n_cols)
        self.d_data_rm = ctx.enqueue_create_buffer[DType.float32](nr * nc)
        ctx.enqueue_function[transpose_to_row_major_kernel](
            self.d_data_rm.unsafe_ptr(),
            self.d_data.unsafe_ptr(),
            Int32(nr),
            Int32(nc),
            grid_dim=ceildiv(nr * nc, 256),
            block_dim=256,
        )
        self.has_rm = True

    def search_data_ptr(mut self) -> MutPointer[Float32, MutAnyOrigin]:
        """The matrix the range and score passes read."""
        comptime if ET_ROW_MAJOR:
            return self.d_data_rm.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        return self.d_data.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()


from ensemble.device_layout import upload_forest_x
from ensemble.device_finite import FOREST_DEVICE_FINITE, ForestFiniteScan


def upload_dataset(
    ctx: DeviceContext,
    x_col_major: List[Float32],
    class_ids: List[Int32],
    n_rows: Int32,
    n_cols: Int32,
    n_classes: Int32,
    x_addr: Int = 0,
    x_row_major: Bool = False,
    labels_on_device: Bool = False,
) raises -> DeviceDataset:
    """Put the immutable half of the fit on the device, once. DEVIATION 184.

    `labels_on_device` (cpu3-trees): `class_ids` is ignored and `d_labels`
    is allocated but not written; `upload_dataset_labels_f32` fills it with
    `class_ids_device_kernel`.

    A nonzero `x_addr` borrows the caller's float32 block through this
    synchronous upload (DEVIATION 2481); `x_row_major` says it is ROW-major
    (the caller's C-order block, untouched). cpu-gpu-cleanup t-forest: the
    borrowed bytes go to the device as they are and the column-major plane
    is written there (`ensemble/device_layout.mojo`), the same words the
    host pinned-stage transpose (DEVIATION 2637) staged. Only with `x_addr`."""
    if x_row_major and x_addr == 0:
        raise Error("upload_dataset: a row-major X must be a borrowed address")
    if x_addr == 0 and len(x_col_major) != Int(n_rows) * Int(n_cols):
        raise Error("x_col_major must be n_rows * n_cols long, column major")
    if not labels_on_device and len(class_ids) != Int(n_rows):
        raise Error("class_ids must be n_rows long")
    var count = Int(n_rows) * Int(n_cols)
    var boundary_times = StageTimes()
    var boundary_start = boundary_times.start()
    var d_data: DeviceBuffer[DType.float32]
    if x_addr != 0:
        var source = MutPointer[Float32, MutUntrackedOrigin](
            unsafe_from_address=x_addr
        )
        d_data = upload_forest_x(
            ctx, source, Int(n_rows), Int(n_cols), x_row_major
        )
    else:
        # A List caller (the checks): its column-major bytes are staged once.
        d_data = ctx.enqueue_create_buffer[DType.float32](count)
        var h_data = ctx.enqueue_create_host_buffer[DType.float32](count)
        ctx.synchronize()
        memcpy(dest=h_data.unsafe_ptr(), src=x_col_major.unsafe_ptr(), count=count)
        ctx.enqueue_copy(dst_buf=d_data, src_ptr=h_data.unsafe_ptr())
        ctx.synchronize()
        _ = h_data^
    var d_labels = ctx.enqueue_create_buffer[DType.int32](Int(n_rows))
    var h_labels = ctx.enqueue_create_host_buffer[DType.int32](
        1 if labels_on_device else Int(n_rows)
    )
    ctx.synchronize()
    if not labels_on_device:
        memcpy(
            dest=h_labels.unsafe_ptr(),
            src=class_ids.unsafe_ptr(),
            count=Int(n_rows),
        )
        ctx.enqueue_copy(dst_buf=d_labels, src_ptr=h_labels.unsafe_ptr())
    var d_data_rm = ctx.enqueue_create_buffer[DType.float32](1)
    # FAST on Apple (lane apple-fast-rfet-scan): the non-finite refusal is
    # this device scan of the uploaded X; the Python fit skips its host scan
    # (`trees_device_finite_scan`). See ensemble/device_finite.mojo.
    comptime if FOREST_DEVICE_FINITE:
        var fscan = ForestFiniteScan(ctx)
        fscan.enqueue(ctx, d_data, count)
        ctx.synchronize()
        fscan.refuse_if_bad("upload_dataset: ")
    else:
        ctx.synchronize()
    _ = h_labels^
    boundary_times.stop_host("boundary_dataset_upload", boundary_start)
    boundary_times.report()
    var d_bins_rm = ctx.enqueue_create_buffer[ET_CODE](1)
    var d_quant = ctx.enqueue_create_buffer[DType.float32](1)
    var d_nbins = ctx.enqueue_create_buffer[DType.int32](1)
    ctx.synchronize()
    return DeviceDataset(
        d_data^, d_labels^, n_rows, n_cols, n_classes, d_data_rm^, False,
        d_bins_rm^, d_quant^, d_nbins^, False,
    )


def class_ids_device_kernel(
    out_ids: MutPointer[Int32, MutAnyOrigin],
    labels: MutPointer[Float32, MutAnyOrigin],
    first_bad: MutPointer[Int32, MutAnyOrigin],
    n_rows: Int32,
    n_classes: Int32,
):
    """cpu3-trees: `class_ids_for` (`randomforest.mojo`) on the device, one
    thread per row. The id is the SAME truncation (`Int(label)`); a label
    whose truncation falls outside `[0, n_classes)` (or a NaN) stores 0 and
    lowers `first_bad[0]` to its row (an integer minimum: order-free)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_rows):
        return
    var v = labels[unsafe_offset=i]
    if v > Float32(-1.0) and v < Float32(Int(n_classes)):
        out_ids[unsafe_offset=i] = Int32(Int(v))
    else:
        out_ids[unsafe_offset=i] = Int32(0)
        _ = Atomic[DType.int32].min(first_bad, Int32(i))


def upload_dataset_labels_f32(
    ctx: DeviceContext,
    x_col_major: List[Float32],
    labels: List[Float32],
    n_rows: Int32,
    n_cols: Int32,
    n_classes: Int32,
    x_addr: Int = 0,
    x_row_major: Bool = False,
) raises -> DeviceDataset:
    """`upload_dataset` with the float labels converted to class ids ON THE
    DEVICE (cpu3-trees: the host `class_ids_for` row loop left the GPU fit).
    The labels cross the bus once as the caller's float32 words; the ids are
    `class_ids_device_kernel`'s, the same truncation as the host column's
    `class_ids_for`, so no bit moves. One Int32 comes back: the first row
    whose label is out of range, refused with `class_ids_for`'s message."""
    if len(labels) != Int(n_rows):
        raise Error(
            "labels must be n_rows long; got "
            + String(len(labels))
            + " for n_rows="
            + String(n_rows)
        )
    var dataset = upload_dataset(
        ctx, x_col_major, List[Int32](), n_rows, n_cols, n_classes,
        x_addr=x_addr, x_row_major=x_row_major, labels_on_device=True,
    )
    var n = Int(n_rows)
    var d_lf = ctx.enqueue_create_buffer[DType.float32](n)
    var h_lf = ctx.enqueue_create_host_buffer[DType.float32](n)
    var d_bad = ctx.enqueue_create_buffer[DType.int32](1)
    var h_bad = ctx.enqueue_create_host_buffer[DType.int32](1)
    ctx.synchronize()
    memcpy(dest=h_lf.unsafe_ptr(), src=labels.unsafe_ptr(), count=n)
    ctx.enqueue_copy(dst_buf=d_lf, src_ptr=h_lf.unsafe_ptr())
    ctx.enqueue_memset(d_bad, Int32(n))
    ctx.enqueue_function[class_ids_device_kernel](
        dataset.d_labels.unsafe_ptr(),
        d_lf.unsafe_ptr(),
        d_bad.unsafe_ptr(),
        n_rows,
        n_classes,
        grid_dim=ceildiv(n, 256),
        block_dim=256,
    )
    ctx.enqueue_copy(dst_buf=h_bad, src_buf=d_bad)
    ctx.synchronize()
    var bad = Int(h_bad.unsafe_ptr()[unsafe_offset=0])
    _ = d_lf^
    _ = h_lf^
    _ = d_bad^
    _ = h_bad^
    if bad < n:
        raise Error(
            "label "
            + String(labels[bad])
            + " at row "
            + String(bad)
            + " truncates to class id "
            + String(Int(labels[bad]))
            + ", which is outside [0, "
            + String(n_classes)
            + "). The device score kernel would index its shared"
            " accumulator out of bounds; refused by name."
        )
    return dataset^


def label_scale_exponent(scale: Float64) raises -> Int:
    """`e` with `scale == 2^e`: `choose_scale` returns a power of two
    (DEVIATION 135), read here from its binary64 exponent field."""
    var b = bitcast[DType.uint64](scale)
    if (b & UInt64(0x000FFFFFFFFFFFFF)) != UInt64(0) or scale <= 0.0:
        raise Error("label scale " + String(scale) + " is not a power of two")
    return Int((b >> UInt64(52)) & UInt64(0x7FF)) - 1023


def quantize_labels_device_kernel(
    out_q: MutPointer[Int32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    n_rows: Int32,
    scale_exp: Int32,
):
    """cpu3-trees: `quantize(Float64(y) * scale)` (DEVIATION 135) on the
    device, one thread per row, in integers. `scale` is `2^scale_exp`, so the
    binary64 product is exact and its truncation toward zero is the float's
    significand shifted by its exponent plus `scale_exp`, sign applied after:
    the same Int32 as the host column's multiply-and-truncate
    (`quantize_labels_host`), with no float operation on any vendor."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_rows):
        return
    var bits = bitcast[DType.uint32](y[unsafe_offset=i])
    var e = Int((bits >> UInt32(23)) & UInt32(0xFF))
    var sig = Int64(Int(bits & UInt32(0x7FFFFF)))
    if e == 0:
        e = 1
    else:
        sig = sig | Int64(0x800000)
    var sh = e - 150 + Int(scale_exp)
    var q = Int64(0)
    if sh >= 0:
        # the scale bounds every |y| * scale below 2^31 (DEVIATION 135), so a
        # valid label never shifts past 7 here; the cap only keeps the shift
        # defined
        if sh < 40:
            q = sig << Int64(sh)
    elif sh > -24:
        q = sig >> Int64(-sh)
    if (bits >> UInt32(31)) != UInt32(0):
        q = -q
    out_q[unsafe_offset=i] = q.cast[DType.int32]()


def upload_dataset_labels_quantized(
    ctx: DeviceContext,
    x_col_major: List[Float32],
    y: List[Float32],
    n_rows: Int32,
    n_cols: Int32,
    mut scale_out: Float64,
    x_addr: Int = 0,
    x_row_major: Bool = False,
) raises -> DeviceDataset:
    """`upload_dataset` for a regressor with the labels QUANTIZED ON THE
    DEVICE (cpu3-trees: the host `quantize_labels_host` row loops left the GPU
    fit). y crosses the bus once as the caller's float32 words; the scale's
    `sum |y|` is `core/abs_sum_blocked`'s fixed-order device sum (one word
    back; `quantize_labels_host` restates it with `host_abs_sum_blocked`),
    `choose_scale` runs on that one scalar, and `quantize_labels_device_kernel`
    writes the Int32 labels into the dataset. The scale is returned in
    `scale_out` for the leaf values."""
    if len(y) != Int(n_rows):
        raise Error(
            "labels must be n_rows long; got "
            + String(len(y))
            + " for n_rows="
            + String(n_rows)
        )
    var dataset = upload_dataset(
        ctx, x_col_major, List[Int32](), n_rows, n_cols, 1,
        x_addr=x_addr, x_row_major=x_row_major, labels_on_device=True,
    )
    var n = Int(n_rows)
    var d_y = ctx.enqueue_create_buffer[DType.float32](max(n, 1))
    var h_y = ctx.enqueue_create_host_buffer[DType.float32](max(n, 1))
    ctx.synchronize()
    memcpy(dest=h_y.unsafe_ptr(), src=y.unsafe_ptr(), count=n)
    ctx.enqueue_copy(dst_buf=d_y, src_ptr=h_y.unsafe_ptr())
    # reads back one word, so the upload above has landed after it
    var mag = device_abs_sum_blocked(ctx, d_y, n)
    var scale = choose_scale(mag, n)
    var e = label_scale_exponent(scale)
    if n > 0:
        ctx.enqueue_function[quantize_labels_device_kernel](
            dataset.d_labels.unsafe_ptr(),
            d_y.unsafe_ptr(),
            n_rows,
            Int32(e),
            grid_dim=ceildiv(n, 256),
            block_dim=256,
        )
    ctx.synchronize()
    _ = d_y^
    _ = h_y^
    scale_out = scale
    return dataset^


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
    var d_part_flags: DeviceBuffer[DType.uint8]
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

    `blocks` is `builder.cuh:230`:
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
        d_part_flags=ctx.enqueue_create_buffer[DType.uint8](
            Int(n_rows) if ET_PART_FLAGS else 1
        ),
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
measurement arm overrides it with `-D MOJOLEARN_ET_TPB_512=1`
(tools/et_profile_leg.sh builds those arms
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
    `MOJOLEARN_ET_TPB_512` widens a 32-lane device (`_256` was DROPPED
    on Apple: lane/apple-fast @ 269ffa57a), and
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
    # AFT F06: two 32-lane SIMD groups per block reduces the live score
    # reduction footprint. Keep range/partition work-map geometry coupled
    # through DEVICE_TPB, as the stable scatter consumes that same map.
    # This is a new 64-thread arm, not the rejected Apple 256-thread retry
    # recorded above. experiments/apple_fast_trees/IDEAS.md; no quality or
    # speed evidence; opt-in and uncompiled/unverified/unmeasured.
    if (
        GLOBAL_NUMERIC_MODE == NUMERIC_FAST
        and has_apple_gpu_accelerator()
        and is_defined["MOJOLEARN_AFT_F06"]()
    ):
        return 64
    if is_defined["MOJOLEARN_ET_TPB_1024"]():
        return 1024
    if is_defined["MOJOLEARN_ET_TPB_512"]():
        return 512
    if is_defined["MOJOLEARN_ET_TPB_128"]():
        return 128
    return 128 if WARP_SIZE <= 32 else 512

comptime DEVICE_MAX_ACC = _device_max_acc()
"""Widest per-cell class accumulator the score kernel's shared memory admits
(DEVIATION 172). One definition, for the same reason DEVICE_TPB is one.
32 unless a DEVIATION 2021 measurement arm narrows it. Normal classification
fits dispatch to 4/8/16/32-wide score kernels without narrowing this limit."""


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

    ORIGINAL MEASUREMENT ARMS. `-D MOJOLEARN_ET_MAX_ACC_4` / `_8` / `_16`
    force the comptime width. The supported default limit stays 32; `_32`
    forces the original width. None is set by build scripts. The bits cannot
    move: the arrays hold the SAME integers in the same slots and the
    unused tail was all zeros folded through integer sums -- removing a
    zero from an integer sum is the identity. The guard is already
    LOUD, not silent: the classification forest trainer raises by name
    when `n_classes > DEVICE_MAX_ACC` ("the device score kernel is built
    for at most ..."), so an arm too narrow for its dataset refuses
    the fit rather than mis-scoring it -- gate arms accordingly (the
    lane's fixtures fit `_8`; higgs2m and every regression fit `_4`).

    SHIPPING DISPATCH. Classification now selects 4, 8, 16 or 32 at runtime
    in `search_batch`, preserving the original 32-class support. The fixed
    arms above still override dispatch for measurements; `_32` explicitly
    selects the old full-width kernel for A/B identity and timing checks.
    Search tiling and block-size tuning remain independent opt-in arms.
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
FIRST item's tree id, which is what an implementation that kept the per-launch scalar
would silently do. Every tree but the batch-first one must move."""
comptime FOREST_SAB_SHARED_ROW_BASE = Int32(2)
"""DEVIATION 211 sabotage: every in-flight tree's root range starts at slot
0, so their partitions overwrite each other. The forest must move -- this is
the gate watching that the slot offsets are what isolate the trees."""

comptime ET_ROW_MAJOR = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and is_defined["MOJOLEARN_ET_ROW_MAJOR"]()
)

comptime ET_TILED_SEARCH_APPLE_DEFAULT = (
    has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_ET_TILED_SEARCH_OFF"]()
)
"""Apple FAST default for the two tiled search kernels below. Apple M4, 1M
rows, 100 trees, alternating processes, model hashes unchanged: taxireg
(k = 16 of 16) 20.7 -> 6.4 s (0.310), Istella-S regression (k = 220 of 220)
182.6 -> 87.6 s (0.480); classification (k = 4 of 16, 15 of 220) takes the
original kernels through `ensure_row_major`'s `2k >= n` gate, 0.991 both.
`-D MOJOLEARN_ET_TILED_SEARCH_OFF` turns both off on Apple."""

comptime ET_TILED_SEARCH_APPLE_IDENTICAL = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and ET_TILED_SEARCH_APPLE_DEFAULT
    and not (
        is_defined["MOJOLEARN_ET_TILED_SEARCH_IDENTICAL_OFF"]()
        or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
    )
)
"""Apple IDENTICAL (trees-apple2, 2026-09-28): the two tiled search kernels
under IDENTICAL too, bit-inert by construction. The range kernel folds in
range_key space under IDENTICAL (the one-feature kernel's IDENTICAL arm),
so the same min, max and NaN count; the regression score kernel publishes
the same integer counts and label sums (integers, any order). Same
`2k >= n_cols` gate (`ensure_row_major`), so only fits that sample at least
half the features (the regressors at max_features 1.0) take them. Never
binned codes: `ET_BINNED_REG` stays FAST. `-D
MOJOLEARN_ET_TILED_SEARCH_IDENTICAL_OFF` keeps the one-feature kernels."""

comptime IDN_ET_TILED_SEARCH = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not has_apple_gpu_accelerator()
    and not (
        is_defined["MOJOLEARN_IDN_ET_TILED_SEARCH_OFF"]()
        or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
    )
)
"""fam-forests (2026-10-04), IDENTICAL on NVIDIA and AMD: the two tiled
search kernels Apple IDENTICAL already takes (`ET_TILED_SEARCH_APPLE_IDENTICAL`
above, the same bit-inert argument: a key-space range fold, integer counts
and label sums). A fit that samples at least half its features (the
regressors at max_features 1.0) reads each row id and quantized label once
per `ET_FEATURE_TILE` features from a row-major X instead of once per
feature from a column gather. The host column is untouched (same cells).
`-D MOJOLEARN_IDN_ET_TILED_SEARCH_OFF` keeps the one-feature kernels."""

comptime ET_RANGE_TILED = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and (
        is_defined["MOJOLEARN_ET_RANGE_TILED"]()
        or ET_TILED_SEARCH_APPLE_DEFAULT
    )
) or ET_TILED_SEARCH_APPLE_IDENTICAL or IDN_ET_TILED_SEARCH
"""FAST experiment: the range pass reads a row-major X with up to
`ET_FEATURE_TILE` sampled features per block
(`node_feature_range_tiled_kernel`)."""

# AFCL-T09: eight sampled features per task halves the private range/score
# tile footprint and admits more independent tasks on Apple. Every selected
# feature keeps its RNG identity and full row set; this changes no threshold.
# NEVER RUN — PENDING MEASUREMENT; uncompiled/unverified; default OFF.
comptime AFCL_T09 = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and is_defined["MOJOLEARN_AFCL_T09"]()
)
comptime ET_FEATURE_TILE = 8 if AFCL_T09 else 16

comptime ET_SPLIT_REDUCE_ONE_BLOCK = GLOBAL_NUMERIC_MODE != NUMERIC_FAST
# T07: 128 lanes, all candidate columns in strided lanes; existing exact
# comparator and feature/node RNG salts retained. Same total winner graph.
# NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
comptime ET_REDUCE_TPB = 128 if T07 else DEVICE_TPB
"""DEVIATION 5611 (2026-09-27): under IDENTICAL every node's candidates are
reduced by ONE block (`blocks_per_node = 1`), so `split_reduce_kernel`
never merges two blocks through the node's device mutex. That merge reads
and writes the node's cells with PLAIN loads and stores inside the critical
section, which the M3 GPU does not make visible across threadgroups (the RF
builder lost candidates there; see `HIST_SPLIT_CANDIDATES_DEFAULT` in
`ensemble/.../builder_kernels_impl.mojo`). `SplitExact.update` is a total
order (exact key, then DEVIATION 463's keyed tie), so one block's grid-stride
fold picks the node's split that any arrival order of a correct merge picks:
the same bits on every other column. FAST keeps `ceildiv(k, TPB)` blocks."""

comptime ET_SCORE_TILED = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and (
        is_defined["MOJOLEARN_ET_SCORE_TILED"]()
        or ET_TILED_SEARCH_APPLE_DEFAULT
    )
) or ET_TILED_SEARCH_APPLE_IDENTICAL or IDN_ET_TILED_SEARCH
"""FAST experiment: the REGRESSION score pass reads a row-major X with up to
`ET_FEATURE_TILE` sampled features per block
(`node_feature_score_reg_tiled_kernel`)."""

comptime ET_RM_DATA = ET_ROW_MAJOR or ET_RANGE_TILED or ET_SCORE_TILED
"""FAST experiment: a row-major copy of X feeds the range and score passes,
whose grids put the feature slot on the fast axis so the blocks reading
one row chunk's features run together and share its cache lines."""

comptime ET_RM_NARROW = is_defined["MOJOLEARN_ET_RM_NARROW"]() or (
    has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_ET_RM_NARROW_OFF"]()
)
"""Apple, both modes (trees-apple2, 2026-09-28): `ensure_row_major` also
builds the row-major copy when one row's floats fit a 64-byte line
(`n_cols <= 16`) and the fit samples at least a quarter of the features,
so a classifier sampling k = 4 of 16 (taxi) takes the tiled range kernel.
The range kernel's cells are the same min/max/NaN counts either way.
M4 IDENTICAL (steward 1790610860810, always-on arm): ExtraTreesClassifier
taxi 3904 -> 3745 ms, same hash; RandomTreesEmbedding (k = 1) 469 -> 1086
ms, hence the quarter gate. `-D MOJOLEARN_ET_RM_NARROW_OFF` turns it off."""

comptime ET_RM_LINE_BYTES = (
    64
    if (
        is_defined["MOJOLEARN_ET_RM_LINE_HW_OFF"]()
        or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
    )
    else 128
)
"""lane/no-dim-idn (2026-10-04): the line `ET_RM_NARROW` compares a row
against. It was a literal 64 bytes, i.e. `n_cols <= 16`, exactly the board's
taxi width. It is now the cache line of the targets this arm runs on: 128
bytes on Apple GPUs, the NVIDIA L1 line and the CDNA3 L2 line alike, so
a row of up to 32 float32 columns is one line fetch however many of its
features are sampled, and the quarter gate (`4k >= n_cols`, from the
RandomTreesEmbedding k = 1 slowdown) still bounds the wasted bytes per line
at three quarters at any width. Bit-inert: the range cells are the same
either way. `-D MOJOLEARN_ET_RM_LINE_HW_OFF` (and `MOJOLEARN_IDN_ALL_OFF`)
restore 64 (A/B arm B)."""
comptime ET_RM_NARROW_GENERAL = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and not is_defined["MOJOLEARN_LEGACY_NARROW_ET_RM"]()
)
"""lane apple-fast-no-narrow-2 (2026-10-04): FAST drops the quarter gate
(see `ensure_row_major`); IDENTICAL keeps it, over main's ET_RM_LINE_BYTES line. `-D
MOJOLEARN_LEGACY_NARROW_ET_RM` restores the quarter gate in FAST."""


@always_inline
def search_grid(row_blocks: Int, k: Int) -> Tuple[Int, Int, Int]:
    """The range/score grid: (row blocks, features), swapped under
    ET_ROW_MAJOR."""
    comptime if ET_ROW_MAJOR:
        return (k, row_blocks, 1)
    return (row_blocks, k, 1)


# AFT F07: double the partition's rows/lane relative to the search work
# budget. At the shipped Apple geometry this is 4096 rows/block, reducing
# count/scan/scatter workgroup count without changing stable row order.
# The existing separate search/partition workload maps handle unequal tiles.
# experiments/apple_fast_trees/IDEAS.md; opt-in, no quality/speed evidence.
comptime AFT_F07 = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and is_defined["MOJOLEARN_AFT_F07"]()
)
comptime PART_ROWS_PER_THREAD = 2 * SEARCH_ROWS_PER_THREAD if AFT_F07 else (
    SEARCH_ROWS_PER_THREAD
    if (
        GLOBAL_NUMERIC_MODE == NUMERIC_FAST
        and (
            is_defined["MOJOLEARN_ET_PART_ROWS"]()
            or (
                has_apple_gpu_accelerator()
                and not is_defined["MOJOLEARN_ET_PART_ROWS_OFF"]()
            )
        )
    )
    or (
        # fam-forests (2026-10-04), `IDN_ET_PART_ROWS`: IDENTICAL on NVIDIA
        # and AMD. Those vendors search at 64 rows per thread and partitioned
        # at 1, so EVERY level cycle restaged `d_items` / `d_wl` and drained
        # the queue before the partition (the
        # `SEARCH_ROWS_PER_THREAD != PART_ROWS_PER_THREAD` arm of the level
        # loops); with equal tiles a plain cycle does neither. The ROWS > 1
        # arms of the four partition kernels are a stable partition (a
        # thread keeps its rows in order after the previous thread, blocks
        # in order), so `row_ids` is the same array as at ROWS == 1: no bit
        # moves and the host column is untouched.
        # `-D MOJOLEARN_IDN_ET_PART_ROWS_OFF` restores one row per thread.
        GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
        and not has_apple_gpu_accelerator()
        and not (
            is_defined["MOJOLEARN_IDN_ET_PART_ROWS_OFF"]()
            or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
        )
    )
    else 1
)
"""FAST: the partition's four kernels fold the search's rows per thread
too, on the SAME `TPB * R` workload tile the search staged, so a plain cycle
no longer restages `d_wl` and drains before partitioning, and each block
pays its scans once per `TPB * R` rows instead of per `TPB`. Row order
within a side stays stable by block and by thread; nothing downstream
reads it (DEVIATION 203). THE FAST APPLE DEFAULT since 2026-10-02 (M3 Ultra
board shapes, aft-ab-etpr: taxi 3.87 -> 3.04 s, Istella 4.37 -> 4.14 s, same
hashes); `-D MOJOLEARN_ET_PART_ROWS_OFF` is the A arm. Elsewhere opt-in
(`-D MOJOLEARN_ET_PART_ROWS=1`); Apple M4 1M rows was a wash."""

comptime ET_PART_FLAGS = (
    (GLOBAL_NUMERIC_MODE == NUMERIC_FAST or has_apple_gpu_accelerator())
    and PART_ROWS_PER_THREAD == 1
    and not is_defined["MOJOLEARN_ET_PART_FLAGS_OFF"]()
)
"""FAST, and IDENTICAL on Apple since 2026-09-28 (the same directions, so the
same partition: a data movement, bit-inert): the partition's count pass
stores each row's direction as one byte
(`LevelWorkspace.d_part_flags`) and the scatter pass reads it instead of
gathering the split column again. Same directions, same partition.
`-D MOJOLEARN_ET_PART_FLAGS_OFF` restores the second gather."""

comptime ET_STAGE_LIVE_PREFIX = (
    (
        GLOBAL_NUMERIC_MODE == NUMERIC_FAST
        or has_apple_gpu_accelerator()
        # fam-forests (2026-10-04), `IDN_ET_STAGE_LIVE`: IDENTICAL on NVIDIA
        # and AMD too. Only the live prefix is ever read, so no bit moves.
        # `-D MOJOLEARN_IDN_ET_STAGE_LIVE_OFF` restores full-capacity staging.
        or (
            GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
            and not (
                is_defined["MOJOLEARN_IDN_ET_STAGE_LIVE_OFF"]()
                or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
            )
        )
    )
    and not is_defined["MOJOLEARN_ET_STAGE_FULL_CAPACITY"]()
)
"""FAST, and IDENTICAL on Apple since 2026-09-28 (only the live prefix is
ever read, so moving less is bit-inert): `stage_batch` compares, snapshots
and uploads each staging
slot's LIVE prefix (the batch's nodes and workload blocks) instead of the
workspace's full capacity. At the Apple FAST batch width (32768 nodes, about
40k workload blocks at 1M rows) the full-capacity path moved every slot's
whole capacity each batch. `-D MOJOLEARN_ET_STAGE_FULL_CAPACITY` restores it."""

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
comptime PHASE_SEED = 10
comptime PHASE_SAMPLER = 11
comptime PHASE_STAGE_BATCH = 12
comptime N_PHASES = 13


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
        if phase == PHASE_SEED:
            return "stage_batch + fused seeders (of stage)"
        if phase == PHASE_SAMPLER:
            return "feature sampler (of stage)"
        if phase == PHASE_STAGE_BATCH:
            return "stage_batch alone (of stage)"
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


# AFT F08: permit 1 GiB total for the two Int32 row-slot planes, versus
# the default 512 MiB. More whole trees then expose independent nodes in
# one frontier; each tree retains its feature RNG and node growth policy.
# This is a byte budget, unrelated to any dataset dimension. Other per-tree
# scratch adds to this budget, so peak memory is a later qualification gate.
# experiments/apple_fast_trees/IDEAS.md; opt-in, no quality/speed evidence.
comptime FOREST_ROW_SLOT_CAP = (1 << 27) if (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and is_defined["MOJOLEARN_AFT_F08"]()
) else (1 << 26)
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
    live: Int = -1,
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
    comptime if ET_STAGE_LIVE_PREFIX:
        # FAST: compare, snapshot and send the LIVE prefix only. Every
        # kernel bound and address derives from the live counts (see the
        # sabotage paragraph above), so no kernel reads past `live`. The
        # FIRST upload of a slot still sends the full capacity below, so
        # the shadow equals the device over every byte from then on and a
        # later, longer prefix can never skip against unsent bytes.
        if seen_before and live >= 0 and live < n:
            if live == 0:
                return
            var lb = sp.bitcast[UInt8]()
            var lh = hp.bitcast[UInt8]()
            var bytes = live * size_of[Scalar[dt]]()
            if True:
                var same_live = True
                var j = 0
                while j + 16 <= bytes:
                    if lh.unsafe_load[width=16](j) != lb.unsafe_load[width=16](j):
                        same_live = False
                        break
                    j += 16
                if same_live:
                    while j < bytes:
                        if lh.unsafe_load(j) != lb.unsafe_load(j):
                            same_live = False
                            break
                        j += 1
                if same_live:
                    return
            memcpy(dest=hp, src=sp, count=live)
            var view = dst.create_sub_buffer[dt](0, live)
            ctx.enqueue_copy(dst_buf=view, src_ptr=sp)
            return
    if seen_before:
        comptime if is_defined["MOJOLEARN_ET_SAB_STAGE_SKIP_ALWAYS"]():
            if payload_slot:
                return
        var same = True
        # DEVIATION 2488: full-capacity byte equality, 16 bytes at a time.
        # The retained scalar build arm isolates this one mechanism in A/B.
        comptime if is_defined["MOJOLEARN_ET_SCALAR_STAGE_COMPARE"]():
            for i in range(n):  # small-loop(n: staging slot capacity of launch descriptors): A/B arm compares host staging bytes only
                if hp.unsafe_load(i) != sp.unsafe_load(i):
                    same = False
                    break
        else:
            var sb = sp.bitcast[UInt8]()
            var hb = hp.bitcast[UInt8]()
            var count = n * size_of[Scalar[dt]]()
            var i = 0
            while i + 16 <= count:
                if hb.unsafe_load[width=16](i) != sb.unsafe_load[width=16](i):
                    same = False
                    break
                i += 16
            if same:
                while i < count:
                    if hb.unsafe_load(i) != sb.unsafe_load(i):
                        same = False
                        break
                    i += 1
        if same:
            return
    comptime if is_defined["MOJOLEARN_ET_SCALAR_STAGE_COMPARE"]():
        for i in range(n):
            hp.unsafe_store(i, sp.unsafe_load(i))
    else:
        memcpy(dest=hp, src=sp, count=n)
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
    for i in range(n_nodes):  # small-loop(n_nodes: batch work-item descriptors): launch parameter list, batch capped by max_batch_size
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
    for i in range(plan.n_blocks_dimx):  # small-loop(plan.n_blocks_dimx: workload map entries): one launch descriptor per grid block
        wl_ptr[unsafe_offset=i] = plan.info[i]
    var base_acc = 0
    for i in range(n_nodes):  # small-loop(n_nodes: per-node block offsets): launch parameter list, batch capped by max_batch_size
        h_nb.unsafe_ptr().unsafe_store(i, Int32(i * Int(k)))
        h_nc.unsafe_ptr().unsafe_store(i, Int32(Int(k)))
        # Where node `i`'s blocks start in the flattened workload array.
        # `build_workload_info` lays them out contiguously in node order,
        # so this repeats the running sum it performs -- deviation 203's
        # scan pass needs that base and the device cannot derive it.
        ws.h_blk_base.unsafe_ptr().unsafe_store(i, Int32(base_acc))
        var nb_i = ceildiv(
            Int(work_items[i].instances.count), TPB * PART_ROWS_PER_THREAD
        )
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
        n_nodes * size_of[NodeWorkItem](),
    )
    _stage_upload_if_changed(
        ctx, ws.d_tree, ws.h_tree, ws.s_tree, ws.cap_nodes, seen, True, n_nodes
    )
    _stage_upload_if_changed(
        ctx, ws.d_tsalt, ws.h_tsalt, ws.s_tsalt, ws.cap_nodes, seen, True,
        n_nodes,
    )
    _stage_upload_if_changed(
        ctx,
        ws.d_wl,
        ws.h_wl,
        ws.s_wl,
        ws.cap_blocks * size_of[WorkloadInfo](),
        seen,
        False,
        plan.n_blocks_dimx * size_of[WorkloadInfo](),
    )
    _stage_upload_if_changed(
        ctx, ws.d_nb, ws.h_nb, ws.s_nb, ws.cap_nodes, seen, True, n_nodes
    )
    _stage_upload_if_changed(
        ctx, ws.d_nc, ws.h_nc, ws.s_nc, ws.cap_nodes, seen, True, n_nodes
    )
    _stage_upload_if_changed(
        ctx,
        ws.d_blk_base,
        ws.h_blk_base,
        ws.s_blk_base,
        ws.cap_nodes,
        seen,
        False,
        n_nodes,
    )
    ws.stage_valid = True
    # DEVIATION 450: no trailing synchronize. The copies above are queue-
    # ordered ahead of every kernel that reads their destinations, and the
    # `h_*` staging they read from is not rewritten until after the next
    # REQUIRED drain (the reduce readback, or the survey's) -- so the only
    # thing a sync here bought was one more per-cycle stall. cuML's
    # `doSplit` enqueues its `update_device` calls the same way and drains
    # ONCE, at `handle.sync_stream` (`builder.cuh:492-494`).

def _enqueue_classification_leaves[MAX_OUT: Int](
    ctx: DeviceContext,
    mut d_leaves: DeviceBuffer[DType.float32],
    mut d_visit: DeviceBuffer[DType.int32],
    mut d_nodes: DeviceBuffer[DType.uint8],
    mut d_ranges: DeviceBuffer[DType.uint8],
    mut d_row_ids: DeviceBuffer[DType.int32],
    mut dataset: DeviceDataset,
    k_out: Int,
    total_nodes: Int,
) raises:
    """Keep the established leaf kernel for <=16 classes; admit all 32.

    The old fixed width silently zeroed probabilities above 16 classes,
    even though split search admitted 32. The wider leaf kernel performs
    the same per-class integer counts and probability normalization.
    """
    ctx.enqueue_function[
        leaf_kernel[DEVICE_TPB, MAX_OUT, True, zero_fill=True]
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
        block_dim=(DEVICE_TPB, 1, 1),
    )


def _enqueue_classification_score[MAX_ACC: Int](
    ctx: DeviceContext,
    mut ws: LevelWorkspace,
    mut dataset: DeviceDataset,
    mut d_row_ids: DeviceBuffer[DType.int32],
    n_rows: Int32,
    k: Int,
    n_classes: Int32,
    seed: UInt64,
    params: DecisionTreeParams,
    n_cells: Int,
    n_blocks_dimx: Int,
) raises:
    """Launch class-sized integer scoring with the selected count storage.

    Global accumulators stay packed at the runtime class count. The kernel
    chooses private arrays or sharded shared counts; both retain exactly the
    same integers, row assignment, random draws and score finalization.
    """
    comptime TPB = DEVICE_TPB
    ctx.enqueue_function[
        node_feature_score_kernel[TPB, MAX_ACC, True, ET_ROW_MAJOR]
    ](
        ws.d_nleft.unsafe_ptr(),
        ws.d_ntotal.unsafe_ptr(),
        ws.d_accl.unsafe_ptr(),
        ws.d_acct.unsafe_ptr(),
        ws.d_nblocks.unsafe_ptr(),
        ws.d_min.unsafe_ptr(),
        ws.d_max.unsafe_ptr(),
        ws.d_missing.unsafe_ptr(),
        dataset.search_data_ptr(),
        d_row_ids.unsafe_ptr(),
        dataset.d_labels.unsafe_ptr(),
        ws.d_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem](),
        ws.d_wl.unsafe_ptr().unsafe_bitcast[WorkloadInfo](),
        ws.d_colids.unsafe_ptr(),
        ws.d_tree.unsafe_ptr(),
        n_rows,
        Int32(k),
        n_classes,
        seed,
        Int32(0),
        dataset.n_cols,
        grid_dim=search_grid(n_blocks_dimx, Int(k)),
        block_dim=(TPB, 1, 1),
    )
    ctx.enqueue_function[
        node_feature_score_finalize_kernel[MAX_ACC, True]
    ](
        ws.d_status.unsafe_ptr(),
        ws.d_thresh.unsafe_ptr(),
        ws.d_gnum.unsafe_ptr(),
        ws.d_gden.unsafe_ptr(),
        ws.d_nleft.unsafe_ptr(),
        ws.d_ntotal.unsafe_ptr(),
        ws.d_accl.unsafe_ptr(),
        ws.d_acct.unsafe_ptr(),
        ws.d_min.unsafe_ptr(),
        ws.d_max.unsafe_ptr(),
        ws.d_missing.unsafe_ptr(),
        ws.d_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem](),
        ws.d_colids.unsafe_ptr(),
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


def _enqueue_raw_range[FUSED_SMALL: Bool](
    ctx: DeviceContext,
    mut ws: LevelWorkspace,
    mut dataset: DeviceDataset,
    mut d_row_ids: DeviceBuffer[DType.int32],
    n_blocks: Int,
    columns_per_node: Int,
    tiled: Bool,
) raises:
    """Existing raw-value range launch with an explicit T05 node owner.

    Keeping the choice in a compile-time kernel argument preserves the existing
    raw kernel ABI for direct callers. Both variants use the same stored input
    layout and workload descriptors; only T05 skips its bounded fused nodes.
    NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
    """
    comptime TPB = DEVICE_TPB
    if tiled:
        ctx.enqueue_function[node_feature_range_tiled_kernel[TPB,ET_FEATURE_TILE,DType.float32,FUSED_SMALL]](
            ws.d_minkey.unsafe_ptr(),ws.d_maxkey.unsafe_ptr(),
            ws.d_missing.unsafe_ptr(),ws.d_merges.unsafe_ptr(),
            dataset.d_data_rm.unsafe_ptr(),d_row_ids.unsafe_ptr(),
            ws.d_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem](),
            ws.d_wl.unsafe_ptr().unsafe_bitcast[WorkloadInfo](),
            ws.d_colids.unsafe_ptr(),dataset.n_cols,Int32(columns_per_node),
            dataset.d_quant.unsafe_ptr(),
            grid_dim=(n_blocks,ceildiv(columns_per_node,ET_FEATURE_TILE),1),
            block_dim=(TPB,1,1),
        )
    else:
        ctx.enqueue_function[node_feature_range_kernel[TPB,ET_ROW_MAJOR,FUSED_SMALL]](
            ws.d_minkey.unsafe_ptr(),ws.d_maxkey.unsafe_ptr(),
            ws.d_missing.unsafe_ptr(),ws.d_merges.unsafe_ptr(),
            dataset.search_data_ptr(),d_row_ids.unsafe_ptr(),
            ws.d_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem](),
            ws.d_wl.unsafe_ptr().unsafe_bitcast[WorkloadInfo](),
            ws.d_colids.unsafe_ptr(),dataset.n_rows,dataset.n_cols,
            Int32(columns_per_node),Int32(0),
            grid_dim=search_grid(n_blocks,columns_per_node),block_dim=(TPB,1,1),
        )


def _enqueue_small_node_splits[CLASSIFICATION: Bool](
    ctx: DeviceContext,
    mut ws: LevelWorkspace,
    mut dataset: DeviceDataset,
    mut d_row_ids: DeviceBuffer[DType.int32],
    n_nodes: Int,
    columns_per_node: Int,
    n_acc: Int32,
    seed: UInt64,
    params: DecisionTreeParams,
) raises:
    """T05 production raw-value fusion before the existing winner reducer.

    Range kernels reserve the same <=ET_SMALL_NODE_TPB nodes. The fused launch
    reuses existing sampled columns, logical tree/node IDs and candidate slots.
    Ordinary range-only surveys and quantile-code routes never reserve nodes.
    NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
    """
    ctx.enqueue_function[small_node_raw_split_kernel[CLASSIFICATION]](
        ws.c_q.unsafe_ptr(),ws.c_c.unsafe_ptr(),ws.c_m.unsafe_ptr(),
        ws.c_l.unsafe_ptr(),ws.c_nu.unsafe_ptr(),ws.c_de.unsafe_ptr(),ws.c_v.unsafe_ptr(),
        ws.d_nonconst.unsafe_ptr(),ws.d_min.unsafe_ptr(),ws.d_max.unsafe_ptr(),
        ws.d_missing.unsafe_ptr(),ws.d_thresh.unsafe_ptr(),
        dataset.d_data.unsafe_ptr(),d_row_ids.unsafe_ptr(),dataset.d_labels.unsafe_ptr(),
        ws.d_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem](),
        ws.d_colids.unsafe_ptr(),ws.d_tree.unsafe_ptr(),
        dataset.n_rows,Int32(columns_per_node),n_acc,seed,
        params.min_samples_leaf,params.split_criterion,
        grid_dim=(n_nodes,columns_per_node),block_dim=ET_SMALL_NODE_TPB,
    )


def split_tie_tally_kernel(
    out_tally: MutPointer[Int32, MutAnyOrigin],
    r_c: MutPointer[Int32, MutAnyOrigin],
    ties: MutPointer[Int32, MutAnyOrigin],
    n_nodes: Int32,
):
    """`MOJOLEARN_ET_TIE_STATS` only (a measurement define): one thread per
    node adds to `out_tally[0]` when the node's reduce decided a column and
    to `out_tally[1]` when two or more candidates tied exactly. Integer
    atomics, so the two counts are exact in any order. Replaces the host
    walk over the per-node readback (cpu4-forest)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_nodes):
        return
    if r_c[unsafe_offset=i] >= 0:
        _ = Atomic.fetch_add(out_tally, Int32(1))
        if ties[unsafe_offset=i] >= Int32(2):
            _ = Atomic.fetch_add(out_tally.unsafe_offset(1), Int32(1))


def pack_splits_kernel(
    out_splits: MutPointer[Split, MutAnyOrigin],
    r_q: MutPointer[Float32, MutAnyOrigin],
    r_c: MutPointer[Int32, MutAnyOrigin],
    r_m: MutPointer[Float32, MutAnyOrigin],
    r_v: MutPointer[Int32, MutAnyOrigin],
    r_l: MutPointer[Int32, MutAnyOrigin],
    n_nodes: Int32,
):
    """cpu3-trees: one thread per node, the batch's `Split` exactly as the
    host loop built it (`builder.cuh:492-494`): the winner's threshold,
    column, gain and left count, with the gain forced to `MIN_FINITE` when
    the node has no valid candidate (`r_v == 0`) or no column. Pure moves
    and one select: no bit moves."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_nodes):
        return
    var colid = r_c[unsafe_offset=i]
    var metric = r_m[unsafe_offset=i]
    if r_v[unsafe_offset=i] == 0 or colid < 0:
        metric = Float32.MIN_FINITE
    out_splits[unsafe_offset=i] = Split(
        r_q[unsafe_offset=i], colid, metric, r_l[unsafe_offset=i]
    )


def search_batch_enqueue(
    ctx: DeviceContext,
    mut ws: LevelWorkspace,
    mut dataset: DeviceDataset,
    mut d_row_ids: DeviceBuffer[DType.int32],
    n_nodes: Int,
    n_blocks: Int,
    k: Int,
    params: DecisionTreeParams,
    n_classes: Int32,
    n_rows: Int32,
    n_cols: Int32,
    seed: UInt64,
    use_sampler: Bool,
    range_only: Bool,
    mut clock: PhaseClock,
) raises:
    """`search_batch`'s launch sequence on a batch that is ALREADY STAGED on
    the device (cpu4-forest): steps 2 to 8 of `doSplit` with no host list,
    no host staging and no readback. `n_nodes` items sit in `ws.d_items`
    (and `d_tree`, `d_tsalt`, `d_nb`, `d_nc`), `n_blocks` workload entries
    in `ws.d_wl`. The host-list wrapper stages with `stage_batch` and reads
    the splits back; the device level loop (`EtDeviceLoop`) stages with its
    own kernels, launches at proven bounds (dummy items own no block, map
    entries past the live total carry `nodeid == -1`) and reads nothing.
    Outputs stay on the device: `ws.d_splits` (packed), `ws.d_nonconst`,
    and the range cells (`d_min`, `d_max`, `d_missing`) for the survey.
    """
    comptime TPB = DEVICE_TPB
    if n_nodes == 0:
        return
    var n_cells = n_nodes * Int(k)
    # T05 exclusive small-node dispatch. Survey passes must populate every
    # range, while scored raw-value nodes may keep their statistics shared.
    # One row per fused lane bounds registers and preserves range key seams.
    # NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
    var fused_small_rows = Int32(ET_SMALL_NODE_TPB) if T05 and not range_only and n_classes <= 32 else Int32(0)

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
    # T04: every future larger frontier is initialized at its own extent.
    # Dummy nodes inside n_nodes remain initialized; capacity tails are unread.
    # NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
    var setup_nodes = n_nodes if T04 else ws.cap_nodes
    var setup_report = Int32(sampler_report_len(n_nodes) if T04 else ws.cap_report) if use_sampler else Int32(0)
    var setup_a_extent = n_cells
    if Int(setup_report) > setup_a_extent:
        setup_a_extent = Int(setup_report)
    if setup_nodes > setup_a_extent:
        setup_a_extent = setup_nodes
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
        Int32(setup_nodes),
        grid_dim=ceildiv(setup_a_extent, PHASE_SETUP_TPB),
        block_dim=PHASE_SETUP_TPB,
    )
    if not range_only:
        var setup_acc = n_cells * Int(n_classes)
        var setup_b_extent = setup_acc
        if setup_nodes > setup_b_extent:
            setup_b_extent = setup_nodes
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
            Int32(setup_nodes),
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

    clock.tick(ctx, PHASE_SEED)
    # --- 3. the range pass -------------------------------------------
    # --- feature sampling, WHERE cuML DOES IT (deviation 201), unless the
    # caller already chose the columns. DEVIATION 205's rescue does: its
    # column comes from `rescue_pick_kernel` (THE DEVICE RESCUE).
    # cpu3-trees: the rescue's columns are always the DEVICE's (the host
    # survey walk and its host column table are gone): the survey's identity
    # columns are written by `ident_colids_kernel`, the rescue's by
    # `rescue_pick_kernel` (already queued by the caller).
    var dev_colids = not use_sampler
    if range_only and not dev_colids:
        raise Error("the survey (range_only) runs on the device columns only")
    if dev_colids:
        if range_only:
            if Int(k) != Int(n_cols):
                raise Error(
                    "the device survey searches every column; got k = "
                    + String(Int(k))
                )
            ctx.enqueue_function[ident_colids_kernel](
                d_colids.unsafe_ptr(),
                Int32(n_cells),
                n_cols,
                grid_dim=ceildiv(n_cells, 256),
                block_dim=256,
            )
    if use_sampler:
        # --- feature sampling, WHERE cuML DOES IT (deviation 201) --------
        # cpu4-forest: the device sampler only. The batch's items are
        # staged on the device (by `stage_batch` or the device level loop),
        # so no host list exists to sample from; a target without float64
        # refuses the algo-L arm by name inside `sample_features_device`.
        _ = sample_features_device(
            ctx,
            d_colids.unsafe_ptr(),
            d_samp_scratch.unsafe_ptr(),
            d_samp_report.unsafe_ptr(),
            d_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem](),
            ws.d_tree.unsafe_ptr(),
            n_nodes,
            seed,
            Int(n_cols),
            Int(k),
        )

    clock.tick(ctx, PHASE_SAMPLER)
    # DEVIATION 470: the range cells were seeded by fused half A above.
    var tiled_range = False
    comptime if ET_RANGE_TILED:
        tiled_range = dataset.has_rm
    if fused_small_rows != 0:
        _enqueue_raw_range[True](ctx,ws,dataset,d_row_ids,n_blocks,k,tiled_range)
    else:
        _enqueue_raw_range[False](ctx,ws,dataset,d_row_ids,n_blocks,k,tiled_range)
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
    # DEVIATION 450: the range pass drains only when the caller wants the
    # survey NOW. On the full path `h_nonconst` is not READ until after
    # the reduce readback's sync below, so its copy rides the queue and
    # this pass loses its per-cycle stall -- cuML's `doSplit` shape, which
    # drains ONCE per batch. `clock.tick` still syncs when the clock is
    # ENABLED: the timed program was always the serialized one.
    clock.tick(ctx, PHASE_RANGE)

    if range_only:
        return

    # --- 4. the draw and score pass ----------------------------------
    # DEVIATION 470: the score cells and class accumulators were seeded
    # by fused half B above (the survey skips half B and never gets here).
    # Keep the fixed-width measurement arms, including a 32-wide baseline.
    # Normal fits retain all 32 supported classes through runtime dispatch.
    comptime FIXED_ACC = (
        is_defined["MOJOLEARN_ET_MAX_ACC_4"]()
        or is_defined["MOJOLEARN_ET_MAX_ACC_8"]()
        or is_defined["MOJOLEARN_ET_MAX_ACC_16"]()
        or is_defined["MOJOLEARN_ET_MAX_ACC_32"]()
    )
    comptime if FIXED_ACC:
        _enqueue_classification_score[DEVICE_MAX_ACC](
            ctx, ws, dataset, d_row_ids, n_rows, k, n_classes, seed,
            params, n_cells, n_blocks,
        )
    else:
        if n_classes <= 4:
            _enqueue_classification_score[4](
                ctx, ws, dataset, d_row_ids, n_rows, k, n_classes, seed,
                params, n_cells, n_blocks,
            )
        elif n_classes <= 8:
            _enqueue_classification_score[8](
                ctx, ws, dataset, d_row_ids, n_rows, k, n_classes, seed,
                params, n_cells, n_blocks,
            )
        elif n_classes <= 16:
            _enqueue_classification_score[16](
                ctx, ws, dataset, d_row_ids, n_rows, k, n_classes, seed,
                params, n_cells, n_blocks,
            )
        else:
            _enqueue_classification_score[32](
                ctx, ws, dataset, d_row_ids, n_rows, k, n_classes, seed,
                params, n_cells, n_blocks,
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
        Int32(k),
        grid_dim=ceildiv(n_cells, 64),
        block_dim=64,
    )

    if fused_small_rows != 0:
        _enqueue_small_node_splits[True](ctx,ws,dataset,d_row_ids,
            n_nodes,k,n_classes,seed,params)

    # --- 6. evalBestSplit's reduction ---------------------------------
    # DEVIATION 470: the reduce cells and the `r_mx` mutexes (over full
    # capacity) were seeded by fused half B above.
    var bpn = ceildiv(Int(k), ET_REDUCE_TPB)
    if bpn < 1 or ET_SPLIT_REDUCE_ONE_BLOCK:
        bpn = 1
    ctx.enqueue_function[split_reduce_kernel[ET_REDUCE_TPB]](
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
        block_dim=(ET_REDUCE_TPB, 1, 1),
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
        # cpu4-forest: the tally runs ON THE DEVICE (two integer counters),
        # so no per-node host walk reads the readback; one line per batch.
        var d_tally = ctx.enqueue_create_buffer[DType.int32](2)
        ctx.enqueue_memset(d_tally, Int32(0))
        ctx.enqueue_function[split_tie_tally_kernel](
            d_tally.unsafe_ptr(),
            r_c.unsafe_ptr(),
            ws.d_ties.unsafe_ptr(),
            Int32(n_nodes),
            grid_dim=ceildiv(n_nodes, 64),
            block_dim=64,
        )
        var h_tally = ctx.enqueue_create_host_buffer[DType.int32](2)
        ctx.enqueue_copy(dst_buf=h_tally, src_buf=d_tally)
        ctx.synchronize()
        print(
            "ET_TIE_STATS batch decided=",
            h_tally.unsafe_ptr()[unsafe_offset=0],
            " tied=",
            h_tally.unsafe_ptr()[unsafe_offset=1],
        )
        _ = d_tally^
        _ = h_tally^

    # --- 7. the splits come back to the host, as `:492-494` does ------
    # ONLY THE SPLITS CROSS, which is exactly what
    # `raft::update_host(h_splits, splits, work_items.size())` copies back
    # at `builder.cuh:492-494`. The gain travels WITH the candidate now
    # (DEVIATION 183, second form), so no per-level readback of the node
    # totals is needed and none happens.
    # cpu3-trees: the batch's `Split` records are packed ON THE DEVICE
    # (`pack_splits_kernel`: the invalid-candidate metric mask included)
    # and cross as one block, copied into the host scheduler's list whole.
    ctx.enqueue_function[pack_splits_kernel](
        ws.d_splits.unsafe_ptr().unsafe_bitcast[Split](),
        r_q.unsafe_ptr(),
        r_c.unsafe_ptr(),
        r_m.unsafe_ptr(),
        r_v.unsafe_ptr(),
        r_l.unsafe_ptr(),
        Int32(n_nodes),
        grid_dim=ceildiv(n_nodes, 64),
        block_dim=64,
    )


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
    if len(host_colids) != 0:
        raise Error(
            "host_colids: caller-chosen columns were the host rescue walk,"
            " removed (cpu3-trees); pass an empty list"
        )
    # --- 2. the ragged-batch flattening ------------------------------
    # Search tiles may cover multiple rows per thread. Coverage remains
    # complete and disjoint through the kernels' existing grid-stride loops;
    # score accumulation is integer, range merging is integer min/max, and
    # the local range fold uses total-order keys. Partitioning retains its
    # own TPB tile and therefore must restage its workload plan when R > 1.
    var plan = build_workload_info(
        work_items, TPB * SEARCH_ROWS_PER_THREAD
    )
    stage_batch(ctx, ws, work_items, item_trees, plan, Int(k))
    clock.tick(ctx, PHASE_STAGE_BATCH)
    search_batch_enqueue(
        ctx, ws, dataset, d_row_ids, n_nodes, plan.n_blocks_dimx, Int(k),
        params, n_classes, n_rows, n_cols, seed, use_sampler, range_only,
        clock,
    )
    ctx.enqueue_copy(dst_buf=ws.h_nonconst, src_buf=ws.d_nonconst)
    if range_only:
        return (
            List[Split](), List[Int32](), List[Float32](), List[Float32](),
            List[Int32](),
        )
    ctx.enqueue_copy(dst_buf=ws.h_splits, src_buf=ws.d_splits)
    ref o_c = ws.o_c
    ref o_nu = ws.o_nu
    ref o_de = ws.o_de
    ctx.enqueue_copy(dst_buf=o_c, src_buf=ws.r_c)
    ctx.enqueue_copy(dst_buf=o_nu, src_buf=ws.r_nu)
    ctx.enqueue_copy(dst_buf=o_de, src_buf=ws.r_de)
    ctx.synchronize()
    clock.tick(ctx, PHASE_REDUCE)

    # DEVIATION 450: the deferred `h_nonconst` read. Its copy was enqueued
    # in the range pass; the sync above is the batch's ONE drain, so the
    # values are complete here and nowhere earlier did the host need them.
    var any_nonconst = List[Int32](length=n_nodes, fill=Int32(0))
    memcpy(
        dest=any_nonconst.unsafe_ptr(),
        src=ws.h_nonconst.unsafe_ptr(),
        count=n_nodes,
    )

    # --- 8. the batch's splits, as `:492-494` hands them to the host --
    # (packed by `pack_splits_kernel` above; one block copy, no host loop)
    var splits = List[Split](length=n_nodes, fill=Split())
    memcpy(
        dest=splits.unsafe_ptr(),
        src=ws.h_splits.unsafe_ptr().unsafe_bitcast[Split](),
        count=n_nodes,
    )

    clock.tick(ctx, PHASE_HOST_SPLITS)
    return (
        splits^,
        any_nonconst^,
        List[Float32](),
        List[Float32](),
        List[Int32](),
    )


# =============================================================================
# cpu4-forest: THE LEVEL LOOP ON THE DEVICE, for the merged-frontier forest.
#
# The forest trainers below used to pop every batch on the host (one
# `NodeQueue` per in-flight tree), read the batch's splits back, build the
# DEVIATION 205 retry list, its sub-batch and the merge on the host, and push
# the children on the host -- a node list across the bus every level. The
# queue, the retry compaction, the rescue merge, the push and (best-first)
# the frontier are now device state (`kernels/et_loop_kernels.mojo` and the
# best-first kernels here); the host enqueues `ET_LOOP_K` batches, then
# drains ONE fixed-size header (`ETL_HDR_WORDS` scalar control words) and
# reads it, never a node list. The finished trees come back once per group,
# as the model (`EtDeviceLoop.download_trees`).
#
# SAME BITS: node ids, queue order per tree and every draw are the host
# queue's (see the kernel module's doc), so the trees equal what the host
# queue produced and the host column (`train_tree_exact*`) is unchanged. This
# is the default on every vendor in IDENTICAL and FAST; there is no arm that
# restores the host queue on the GPU route. `NodeQueue` stays as the host
# column's queue (`host_builder.mojo`, `train_tree_exact`) and the checks'.
# =============================================================================

comptime ET_LOOP_K = max(1, T11_LEVELS) if T11 else (
    1 if is_defined["MOJOLEARN_ET_DEVICE_LOOP_K1"]() else (
        2 if is_defined["MOJOLEARN_ET_DEVICE_LOOP_K2"]() else (
            8 if is_defined["MOJOLEARN_ET_DEVICE_LOOP_K8"]() else 4
        )
    )
)
"""Batches (best-first: cycles) enqueued per header drain. A scheduling
parameter: it moves no bit. `-D MOJOLEARN_ET_DEVICE_LOOP_K1/_K2/_K8`."""

comptime ETL_SRC_BATCH = 0
comptime ETL_SRC_SUB = 1
comptime ETL_SRC_PART = 2

comptime ET_BF_FRONTIER_BYTES = 1 << 28
"""Best-first: the device frontier budget per group (`g * f_cap` records).
The group width is capped to fit it -- a scheduling cap, like `group_cap`."""


def et_root_expandable(params: DecisionTreeParams, slot_rows: Int32) -> Int32:
    """`NodeQueue.is_expandable(root, 0)` with `leaf_counter == 1`."""
    if Int32(0) >= params.max_depth:
        return Int32(0)
    if slot_rows < params.min_samples_split:
        return Int32(0)
    if params.max_leaves != -1 and Int32(1) >= params.max_leaves:
        return Int32(0)
    return Int32(1)


def et_bf_frontier_cap(
    params: DecisionTreeParams, slot_rows: Int32, bf_sabotage: Int32
) -> Int:
    """Records one tree's frontier can hold: every record is a current leaf,
    and the leaf count never passes `max_leaf_nodes` (nor the tree's rows,
    each leaf holding at least one). Plus slack."""
    var cap = Int(slot_rows)
    if bf_sabotage != BESTFIRST_SAB_NO_BUDGET and Int(
        params.max_leaf_nodes
    ) < cap:
        cap = Int(params.max_leaf_nodes)
    if cap < 1:
        cap = 1
    return cap + 2


def et_bf_admit_kernel(
    hdr: MutPointer[Int32, MutAnyOrigin],
    b_items: MutPointer[NodeWorkItem, MutAnyOrigin],
    b_slot: MutPointer[Int32, MutAnyOrigin],
    b_gpos: MutPointer[Int32, MutAnyOrigin],
    b_splits: MutPointer[Split, MutAnyOrigin],
    slot_stat: MutPointer[Int32, MutAnyOrigin],
    slot_tree: MutPointer[Int32, MutAnyOrigin],
    f_rec: MutPointer[FrontierRecord, MutAnyOrigin],
    f_gpos: MutPointer[Int32, MutAnyOrigin],
    n_items: Int32,
    f_cap: Int32,
    slot_rows: Int32,
    min_impurity_decrease: Float32,
    min_samples_leaf: Int32,
):
    """`NodeQueue.bestfirst_admit` for every searched node, one thread each:
    a valid split joins its tree's frontier with `frontier_key`'s
    improvement. A record's POSITION in the frontier array is an atomic
    ticket, and is unobservable: the pop takes the maximum under
    `bestfirst_before`, a total order, which is what the host heap pops.
    Thread 0 also clears this cycle's pop count."""
    var j = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if j == 0:
        hdr[unsafe_offset=ETL_H_POPS] = Int32(0)
    if j >= Int(n_items):
        return
    var item = b_items[unsafe_offset=j]
    var count = item.instances.count
    if count <= Int32(0):
        return
    var sp = b_splits[unsafe_offset=j]
    var s = Int(b_slot[unsafe_offset=j])
    if split_not_valid(sp, min_impurity_decrease, min_samples_leaf, count):
        _ = Atomic.fetch_add(hdr.unsafe_offset(ETL_H_ROWS), -count)
        return
    var pos = Int(
        Atomic.fetch_add(
            slot_stat.unsafe_offset(s * ETL_STAT_INTS + ETL_ST_FRONT), Int32(1)
        )
    )
    if pos >= Int(f_cap):
        hdr[unsafe_offset=ETL_H_OVERFLOW] = Int32(4)
        return
    var at = s * Int(f_cap) + pos
    f_rec[unsafe_offset=at] = FrontierRecord(
        item,
        sp,
        frontier_key(sp.best_metric_val, count, slot_rows),
        slot_tree[unsafe_offset=s],
    )
    f_gpos[unsafe_offset=at] = b_gpos[unsafe_offset=j]


def et_bf_pop_kernel[
    TPB: Int
](
    hdr: MutPointer[Int32, MutAnyOrigin],
    slot_stat: MutPointer[Int32, MutAnyOrigin],
    f_rec: MutPointer[FrontierRecord, MutAnyOrigin],
    f_gpos: MutPointer[Int32, MutAnyOrigin],
    pt_items: MutPointer[NodeWorkItem, MutAnyOrigin],
    pt_slot: MutPointer[Int32, MutAnyOrigin],
    pt_gpos: MutPointer[Int32, MutAnyOrigin],
    pt_splits: MutPointer[Split, MutAnyOrigin],
    f_cap: Int32,
    max_leaf_nodes: Int32,
    bf_sabotage: Int32,
):
    """`NodeQueue.bestfirst_pop` for every tree, block `s` = tree slot `s`:
    if the tree can pop (`bestfirst_can_pop`), the frontier's first record
    under `bestfirst_before` (a block argmax over the records; the order is
    total, so the winner is the heap's), which leaves the frontier (the last
    record takes its place) and becomes partition slot `s`. A tree that
    cannot pop leaves a dummy there (count 0, invalid split)."""
    comptime assert TPB * 4 <= ETL_SHARED_FITS, "best-first pop page"
    var sh_best = stack_allocation[
        TPB, Scalar[DType.int32], address_space=AddressSpace.SHARED
    ]()
    var s = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var base = s * Int(f_cap)
    var fsz = Int(slot_stat[unsafe_offset = s * ETL_STAT_INTS + ETL_ST_FRONT])
    var leaves = slot_stat[unsafe_offset = s * ETL_STAT_INTS + ETL_ST_LEAVES]
    var budget = bf_sabotage == BESTFIRST_SAB_NO_BUDGET or leaves < max_leaf_nodes
    if fsz > Int(f_cap):
        fsz = Int(f_cap)
    if fsz <= 0 or not budget:
        if tid == 0:
            pt_items[unsafe_offset=s] = etl_dummy_item()
            pt_slot[unsafe_offset=s] = Int32(s)
            pt_gpos[unsafe_offset=s] = Int32(-1)
            pt_splits[unsafe_offset=s] = Split()
        return
    var best = -1
    var i = tid
    while i < fsz:
        if best < 0 or bestfirst_before(
            f_rec[unsafe_offset = base + i],
            f_rec[unsafe_offset = base + best],
            bf_sabotage,
        ):
            best = i
        i += TPB
    sh_best[unsafe_offset=tid] = Int32(best)
    barrier()
    var stride = TPB // 2
    while stride > 0:
        if tid < stride:
            var o = Int(sh_best[unsafe_offset = tid + stride])
            var m = Int(sh_best[unsafe_offset=tid])
            if o >= 0 and (
                m < 0
                or bestfirst_before(
                    f_rec[unsafe_offset = base + o],
                    f_rec[unsafe_offset = base + m],
                    bf_sabotage,
                )
            ):
                sh_best[unsafe_offset=tid] = Int32(o)
        barrier()
        stride //= 2
    if tid == 0:
        var b = Int(sh_best[unsafe_offset=0])
        var rec = f_rec[unsafe_offset = base + b]
        pt_items[unsafe_offset=s] = rec.item
        pt_slot[unsafe_offset=s] = Int32(s)
        pt_gpos[unsafe_offset=s] = f_gpos[unsafe_offset = base + b]
        pt_splits[unsafe_offset=s] = rec.split
        var last = fsz - 1
        f_rec[unsafe_offset = base + b] = f_rec[unsafe_offset = base + last]
        f_gpos[unsafe_offset = base + b] = f_gpos[unsafe_offset = base + last]
        slot_stat[unsafe_offset = s * ETL_STAT_INTS + ETL_ST_FRONT] = Int32(last)
        _ = Atomic.fetch_add(hdr.unsafe_offset(ETL_H_POPS), Int32(1))
        _ = Atomic.fetch_add(
            hdr.unsafe_offset(ETL_H_ROWS), -rec.item.instances.count
        )


def et_bf_expand_kernel(
    hdr: MutPointer[Int32, MutAnyOrigin],
    slot_stat: MutPointer[Int32, MutAnyOrigin],
    pt_items: MutPointer[NodeWorkItem, MutAnyOrigin],
    pt_gpos: MutPointer[Int32, MutAnyOrigin],
    pt_splits: MutPointer[Split, MutAnyOrigin],
    g_nodes: MutPointer[SparseTreeNode[DType.float32], MutAnyOrigin],
    g_meta: MutPointer[Int32, MutAnyOrigin],
    b_items: MutPointer[NodeWorkItem, MutAnyOrigin],
    b_slot: MutPointer[Int32, MutAnyOrigin],
    b_gpos: MutPointer[Int32, MutAnyOrigin],
    g_in: Int32,
    node_cap: Int32,
    max_depth: Int32,
    min_samples_split: Int32,
    max_leaves: Int32,
    max_leaf_nodes: Int32,
    bf_sabotage: Int32,
):
    """`NodeQueue.bestfirst_expand`, one thread per tree slot `s`: the popped
    node becomes a split node, its two children are appended as leaves with
    local ids `n_nodes, n_nodes + 1` (adjacent, left first), the leaf and
    depth counters move, and the expandable children of a tree with budget
    left become search slots `2 s`, `2 s + 1` of the next cycle (dummies
    otherwise). One writer per tree, so the per-tree counters need no
    atomics; arena records are atomic tickets (the scatter orders them)."""
    var s = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if s >= Int(g_in):
        return
    b_slot[unsafe_offset = 2 * s] = Int32(s)
    b_slot[unsafe_offset = 2 * s + 1] = Int32(s)
    b_items[unsafe_offset = 2 * s] = etl_dummy_item()
    b_items[unsafe_offset = 2 * s + 1] = etl_dummy_item()
    b_gpos[unsafe_offset = 2 * s] = Int32(-1)
    b_gpos[unsafe_offset = 2 * s + 1] = Int32(-1)
    var gpos = Int(pt_gpos[unsafe_offset=s])
    if gpos < 0:
        return
    var item = pt_items[unsafe_offset=s]
    var sp = pt_splits[unsafe_offset=s]
    var st = s * ETL_STAT_INTS
    var bn = slot_stat[unsafe_offset = st + ETL_ST_NODES]
    var leaves = slot_stat[unsafe_offset = st + ETL_ST_LEAVES] + Int32(1)
    var gl = Int(Atomic.fetch_add(hdr.unsafe_offset(ETL_H_NODES), Int32(2)))
    if gl + 2 > Int(node_cap):
        hdr[unsafe_offset=ETL_H_OVERFLOW] = Int32(1)
        return
    slot_stat[unsafe_offset = st + ETL_ST_NODES] = bn + Int32(2)
    slot_stat[unsafe_offset = st + ETL_ST_LEAVES] = leaves
    var d1 = item.depth + Int32(1)
    if d1 > slot_stat[unsafe_offset = st + ETL_ST_DEPTH]:
        slot_stat[unsafe_offset = st + ETL_ST_DEPTH] = d1
    var begin = item.instances.begin
    var count = item.instances.count
    var nl = sp.n_left
    var nr = count - nl
    g_nodes[unsafe_offset=gpos] = SparseTreeNode[
        DType.float32
    ].CreateSplitNode(
        sp.colid, sp.quesval, sp.best_metric_val, Int64(Int(bn)), count
    )
    g_nodes[unsafe_offset=gl] = SparseTreeNode[DType.float32].CreateLeafNode(
        nl
    )
    g_meta[unsafe_offset = gl * ETL_META_INTS + 0] = Int32(s)
    g_meta[unsafe_offset = gl * ETL_META_INTS + 1] = bn
    g_meta[unsafe_offset = gl * ETL_META_INTS + 2] = begin
    g_meta[unsafe_offset = gl * ETL_META_INTS + 3] = nl
    g_nodes[unsafe_offset = gl + 1] = SparseTreeNode[
        DType.float32
    ].CreateLeafNode(nr)
    g_meta[unsafe_offset = (gl + 1) * ETL_META_INTS + 0] = Int32(s)
    g_meta[unsafe_offset = (gl + 1) * ETL_META_INTS + 1] = bn + Int32(1)
    g_meta[unsafe_offset = (gl + 1) * ETL_META_INTS + 2] = begin + nl
    g_meta[unsafe_offset = (gl + 1) * ETL_META_INTS + 3] = nr
    var budget = (
        bf_sabotage == BESTFIRST_SAB_NO_BUDGET or leaves < max_leaf_nodes
    )
    if not budget:
        return
    var cuml_ok = max_leaves == Int32(-1) or leaves < max_leaves
    if d1 < max_depth and nl >= min_samples_split and cuml_ok:
        b_items[unsafe_offset = 2 * s] = NodeWorkItem(
            bn, d1, InstanceRange(begin, nl)
        )
        b_gpos[unsafe_offset = 2 * s] = Int32(gl)
        _ = Atomic.fetch_add(hdr.unsafe_offset(ETL_H_ROWS), nl)
    if d1 < max_depth and nr >= min_samples_split and cuml_ok:
        b_items[unsafe_offset = 2 * s + 1] = NodeWorkItem(
            bn + Int32(1), d1, InstanceRange(begin + nl, nr)
        )
        b_gpos[unsafe_offset = 2 * s + 1] = Int32(gl + 1)
        _ = Atomic.fetch_add(hdr.unsafe_offset(ETL_H_ROWS), nr)


struct EtDeviceLoop(Movable):
    """One group's device level loop: the header, the shared FIFO, the node
    arena, the per-tree counters, the batch / rescue / partition arrays and
    (best-first) the frontiers. See the block comment above."""

    var g: Int
    var max_batch: Int
    var n_chunks_cap: Int
    var node_cap: Int
    var queue_cap: Int
    var f_cap: Int
    var hdr: DeviceBuffer[DType.int32]
    var h_hdr: HostBuffer[DType.int32]
    var queue: DeviceBuffer[DType.int32]
    var g_nodes: DeviceBuffer[DType.uint8]
    var g_meta: DeviceBuffer[DType.int32]
    var slot_stat: DeviceBuffer[DType.int32]
    var slot_base: DeviceBuffer[DType.int32]
    var slot_tree: DeviceBuffer[DType.int32]
    var b_items: DeviceBuffer[DType.uint8]
    var b_slot: DeviceBuffer[DType.int32]
    var b_gpos: DeviceBuffer[DType.int32]
    var b_splits: DeviceBuffer[DType.uint8]
    var s_items: DeviceBuffer[DType.uint8]
    var s_slot: DeviceBuffer[DType.int32]
    var s_idx: DeviceBuffer[DType.int32]
    var d_pick: DeviceBuffer[DType.int32]
    var x_off: DeviceBuffer[DType.int32]
    var x_nb: DeviceBuffer[DType.int32]
    var x_large: DeviceBuffer[DType.int32]
    var p_rank: DeviceBuffer[DType.int32]
    var p_valid: DeviceBuffer[DType.int32]
    var p_left: DeviceBuffer[DType.int32]
    var p_kids: DeviceBuffer[DType.int32]
    var p_aoff: DeviceBuffer[DType.int32]
    var p_eoff: DeviceBuffer[DType.int32]
    var x_chunk: DeviceBuffer[DType.int32]
    var cnt: DeviceBuffer[DType.int32]
    var f_rec: DeviceBuffer[DType.uint8]
    var f_gpos: DeviceBuffer[DType.int32]
    var pt_items: DeviceBuffer[DType.uint8]
    var pt_slot: DeviceBuffer[DType.int32]
    var pt_gpos: DeviceBuffer[DType.int32]
    var pt_splits: DeviceBuffer[DType.uint8]
    var tree_base: DeviceBuffer[DType.int32]

    def __init__(
        out self, ctx: DeviceContext, g: Int, max_batch: Int, f_cap: Int
    ) raises:
        """Every buffer the loop needs, sized once per group. `f_cap` is the
        per-tree frontier capacity (0 for depth-wise growth). The arena and
        the FIFO start at `g + 2 * ET_LOOP_K * max_batch` and grow at
        drains (`ensure`)."""
        var nb = max_batch if max_batch > 2 * g else 2 * g
        if nb < 1:
            nb = 1
        var chunks = (nb + ETL_TPB - 1) // ETL_TPB
        var fcap = f_cap if f_cap > 0 else 1
        var cap0 = g + 2 * ET_LOOP_K * nb
        comptime if T13:
            # Group-owned node/queue arena reserves one bounded capacity for
            # reuse across all frontier batches and trees in this group.
            # NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
            var bytes_per_node = size_of[SparseTreeNode[DType.float32]]() + 4*ETL_META_INTS + 4*ETL_Q_INTS
            cap0 = max(cap0,T13_BYTES//max(1,bytes_per_node))
        self.g = g
        self.max_batch = nb
        self.n_chunks_cap = chunks
        self.node_cap = cap0
        self.queue_cap = cap0
        self.f_cap = f_cap
        self.hdr = ctx.enqueue_create_buffer[DType.int32](ETL_HDR_WORDS)
        self.h_hdr = ctx.enqueue_create_host_buffer[DType.int32](ETL_HDR_WORDS)
        self.queue = ctx.enqueue_create_buffer[DType.int32](cap0 * ETL_Q_INTS)
        self.g_nodes = ctx.enqueue_create_buffer[DType.uint8](
            cap0 * size_of[SparseTreeNode[DType.float32]]()
        )
        self.g_meta = ctx.enqueue_create_buffer[DType.int32](
            cap0 * ETL_META_INTS
        )
        self.slot_stat = ctx.enqueue_create_buffer[DType.int32](
            g * ETL_STAT_INTS
        )
        self.slot_base = ctx.enqueue_create_buffer[DType.int32](2 * g)
        self.slot_tree = ctx.enqueue_create_buffer[DType.int32](g)
        self.b_items = ctx.enqueue_create_buffer[DType.uint8](
            nb * size_of[NodeWorkItem]()
        )
        self.b_slot = ctx.enqueue_create_buffer[DType.int32](nb)
        self.b_gpos = ctx.enqueue_create_buffer[DType.int32](nb)
        self.b_splits = ctx.enqueue_create_buffer[DType.uint8](
            nb * size_of[Split]()
        )
        self.s_items = ctx.enqueue_create_buffer[DType.uint8](
            nb * size_of[NodeWorkItem]()
        )
        self.s_slot = ctx.enqueue_create_buffer[DType.int32](nb)
        self.s_idx = ctx.enqueue_create_buffer[DType.int32](nb)
        self.d_pick = ctx.enqueue_create_buffer[DType.int32](nb)
        self.x_off = ctx.enqueue_create_buffer[DType.int32](nb)
        self.x_nb = ctx.enqueue_create_buffer[DType.int32](nb)
        self.x_large = ctx.enqueue_create_buffer[DType.int32](nb)
        self.p_rank = ctx.enqueue_create_buffer[DType.int32](nb)
        self.p_valid = ctx.enqueue_create_buffer[DType.int32](nb)
        self.p_left = ctx.enqueue_create_buffer[DType.int32](nb)
        self.p_kids = ctx.enqueue_create_buffer[DType.int32](nb)
        self.p_aoff = ctx.enqueue_create_buffer[DType.int32](nb)
        self.p_eoff = ctx.enqueue_create_buffer[DType.int32](nb)
        self.x_chunk = ctx.enqueue_create_buffer[DType.int32](3 * chunks)
        self.cnt = ctx.enqueue_create_buffer[DType.int32](chunks * g)
        self.f_rec = ctx.enqueue_create_buffer[DType.uint8](
            g * fcap * size_of[FrontierRecord]()
        )
        self.f_gpos = ctx.enqueue_create_buffer[DType.int32](g * fcap)
        self.pt_items = ctx.enqueue_create_buffer[DType.uint8](
            g * size_of[NodeWorkItem]()
        )
        self.pt_slot = ctx.enqueue_create_buffer[DType.int32](g)
        self.pt_gpos = ctx.enqueue_create_buffer[DType.int32](g)
        self.pt_splits = ctx.enqueue_create_buffer[DType.uint8](
            g * size_of[Split]()
        )
        self.tree_base = ctx.enqueue_create_buffer[DType.int32](g + 1)

    def word(self, w: Int) -> Int:
        """Header word `w` as of the last drain."""
        return Int(self.h_hdr.unsafe_ptr()[unsafe_offset=w])

    def drain(mut self, ctx: DeviceContext) raises:
        """The loop's one readback: the header, then a synchronize."""
        ctx.enqueue_copy(dst_buf=self.h_hdr, src_buf=self.hdr)
        ctx.synchronize()
        var ov = self.word(ETL_H_OVERFLOW)
        if ov != 0:
            raise Error(
                "ET device level loop: header overflow code "
                + String(ov)
                + " (1 arena/FIFO capacity, 2 block-map bound, 3 pop bound,"
                " 4 best-first frontier capacity) -- a host bound bug"
            )

    def init_roots(
        mut self,
        ctx: DeviceContext,
        tree_ids: List[Int32],
        first: Int,
        slot_rows: Int32,
        shared_base: Int32,
        root_expandable: Int32,
        bestfirst: Bool,
    ) raises:
        """`NodeQueue.__init__` for the group: tree ids, roots, header."""
        var h_tree = ctx.enqueue_create_host_buffer[DType.int32](self.g)
        ctx.synchronize()
        memcpy(
            dest=h_tree.unsafe_ptr(),
            src=tree_ids.unsafe_ptr() + first,
            count=self.g,
        )
        ctx.enqueue_copy(dst_buf=self.slot_tree, src_ptr=h_tree.unsafe_ptr())
        ctx.enqueue_function[etl_init_kernel](
            self.hdr.unsafe_ptr(),
            self.queue.unsafe_ptr(),
            self.g_nodes.unsafe_ptr().unsafe_bitcast[
                SparseTreeNode[DType.float32]
            ](),
            self.g_meta.unsafe_ptr(),
            self.slot_stat.unsafe_ptr(),
            self.b_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem](),
            self.b_slot.unsafe_ptr(),
            self.b_gpos.unsafe_ptr(),
            Int32(self.g),
            slot_rows,
            shared_base,
            root_expandable,
            Int32(1) if bestfirst else Int32(0),
            grid_dim=ceildiv(self.g, ETL_TPB),
            block_dim=ETL_TPB,
        )
        self.drain(ctx)
        _ = h_tree^

    def ensure(mut self, ctx: DeviceContext, extra: Int) raises:
        """Call right after a drain. Grows the node arena so `extra` more
        records fit, and re-packs the FIFO's live stretch `[head, tail)` to
        the front of a buffer with `extra` more entries of room (its header
        words rewritten to match). Rare (doubling), so it synchronizes."""
        comptime NB = size_of[SparseTreeNode[DType.float32]]()
        var used = self.word(ETL_H_NODES)
        if used + extra > self.node_cap:
            var cap = 2 * self.node_cap
            if cap < used + extra:
                cap = used + extra
            var nn = ctx.enqueue_create_buffer[DType.uint8](cap * NB)
            var nm = ctx.enqueue_create_buffer[DType.int32](cap * ETL_META_INTS)
            if used > 0:
                var dn = nn.create_sub_buffer[DType.uint8](0, used * NB)
                var sn = self.g_nodes.create_sub_buffer[DType.uint8](
                    0, used * NB
                )
                ctx.enqueue_copy(dst_buf=dn, src_buf=sn)
                var dm = nm.create_sub_buffer[DType.int32](
                    0, used * ETL_META_INTS
                )
                var sm = self.g_meta.create_sub_buffer[DType.int32](
                    0, used * ETL_META_INTS
                )
                ctx.enqueue_copy(dst_buf=dm, src_buf=sm)
                ctx.synchronize()
                _ = dn^
                _ = sn^
                _ = dm^
                _ = sm^
            self.g_nodes = nn^
            self.g_meta = nm^
            self.node_cap = cap
        var head = self.word(ETL_H_HEAD)
        var tail = self.word(ETL_H_TAIL)
        if tail + extra > self.queue_cap:
            var live = tail - head
            var qcap = 2 * live
            if qcap < live + extra:
                qcap = live + extra
            var nq = ctx.enqueue_create_buffer[DType.int32](qcap * ETL_Q_INTS)
            if live > 0:
                var dq = nq.create_sub_buffer[DType.int32](0, live * ETL_Q_INTS)
                var sq = self.queue.create_sub_buffer[DType.int32](
                    head * ETL_Q_INTS, live * ETL_Q_INTS
                )
                ctx.enqueue_copy(dst_buf=dq, src_buf=sq)
                ctx.synchronize()
                _ = dq^
                _ = sq^
            self.queue = nq^
            self.queue_cap = qcap
            # The device header equals `h_hdr` here (drained, nothing queued
            # since): shift the two FIFO words and put the header back.
            self.h_hdr.unsafe_ptr()[unsafe_offset=ETL_H_HEAD] = Int32(0)
            self.h_hdr.unsafe_ptr()[unsafe_offset=ETL_H_TAIL] = Int32(live)
            ctx.enqueue_copy(dst_buf=self.hdr, src_ptr=self.h_hdr.unsafe_ptr())
            ctx.synchronize()

    def enqueue_pop(
        mut self,
        ctx: DeviceContext,
        n_bound: Int,
        bound_s: Int,
        tile_s: Int,
        bound_p: Int,
        tile_p: Int,
    ) raises:
        ctx.enqueue_function[etl_pop_kernel[ETL_TPB]](
            self.hdr.unsafe_ptr(),
            self.queue.unsafe_ptr(),
            self.b_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem](),
            self.b_slot.unsafe_ptr(),
            self.b_gpos.unsafe_ptr(),
            Int32(n_bound),
            Int32(bound_s),
            Int32(tile_s),
            Int32(bound_p),
            Int32(tile_p),
            grid_dim=1,
            block_dim=ETL_TPB,
        )

    def enqueue_stage(
        mut self,
        ctx: DeviceContext,
        mut ws: LevelWorkspace,
        src: Int,
        count_word: Int,
        n_bound: Int,
        k: Int,
        tile: Int,
        part_tile: Int,
        blocks_bound: Int,
        scalar_tree: Int32,
        search_phase: Bool = True,
    ) raises:
        """`stage_batch` from a device list (`ETL_SRC_*`) into the search
        workspace, then its block map. The host staging's shadow copies no
        longer describe the device buffers, so they are invalidated."""
        var src_items = (
            self.b_items.unsafe_ptr()
            .unsafe_bitcast[NodeWorkItem]()
            .unsafe_origin_cast[MutAnyOrigin]()
        )
        var src_slot = self.b_slot.unsafe_ptr().unsafe_origin_cast[
            MutAnyOrigin
        ]()
        if src == ETL_SRC_SUB:
            src_items = (
                self.s_items.unsafe_ptr()
                .unsafe_bitcast[NodeWorkItem]()
                .unsafe_origin_cast[MutAnyOrigin]()
            )
            src_slot = self.s_slot.unsafe_ptr().unsafe_origin_cast[
                MutAnyOrigin
            ]()
        elif src == ETL_SRC_PART:
            src_items = (
                self.pt_items.unsafe_ptr()
                .unsafe_bitcast[NodeWorkItem]()
                .unsafe_origin_cast[MutAnyOrigin]()
            )
            src_slot = self.pt_slot.unsafe_ptr().unsafe_origin_cast[
                MutAnyOrigin
            ]()
        ctx.enqueue_function[etl_stage_kernel[ETL_TPB]](
            self.hdr.unsafe_ptr(),
            Int32(count_word),
            src_items,
            src_slot,
            self.slot_tree.unsafe_ptr(),
            ws.d_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem](),
            ws.d_tree.unsafe_ptr(),
            ws.d_tsalt.unsafe_ptr(),
            ws.d_nb.unsafe_ptr(),
            ws.d_nc.unsafe_ptr(),
            ws.d_blk_base.unsafe_ptr(),
            self.x_off.unsafe_ptr(),
            self.x_nb.unsafe_ptr(),
            self.x_large.unsafe_ptr(),
            Int32(n_bound),
            Int32(k),
            Int32(tile),
            Int32(part_tile),
            Int32(blocks_bound),
            scalar_tree,
            Int32(1) if search_phase else Int32(0),
            grid_dim=1,
            block_dim=ETL_TPB,
        )
        ctx.enqueue_function[etl_map_kernel](
            self.hdr.unsafe_ptr(),
            self.x_off.unsafe_ptr(),
            self.x_nb.unsafe_ptr(),
            self.x_large.unsafe_ptr(),
            ws.d_wl.unsafe_ptr().unsafe_bitcast[WorkloadInfo](),
            Int32(n_bound),
            Int32(blocks_bound),
            grid_dim=ceildiv(blocks_bound, ETL_TPB),
            block_dim=ETL_TPB,
        )
        ws.stage_valid = False

    def enqueue_save_splits(
        mut self, ctx: DeviceContext, mut ws: LevelWorkspace, n_bound: Int
    ) raises:
        ctx.enqueue_function[etl_copy_splits_kernel](
            self.b_splits.unsafe_ptr().unsafe_bitcast[Split](),
            ws.d_splits.unsafe_ptr().unsafe_bitcast[Split](),
            Int32(n_bound),
            grid_dim=ceildiv(n_bound, ETL_TPB),
            block_dim=ETL_TPB,
        )

    def enqueue_retry(
        mut self, ctx: DeviceContext, mut ws: LevelWorkspace, n_bound: Int
    ) raises:
        ctx.enqueue_function[etl_retry_kernel[ETL_TPB]](
            self.hdr.unsafe_ptr(),
            self.b_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem](),
            self.b_slot.unsafe_ptr(),
            self.b_splits.unsafe_ptr().unsafe_bitcast[Split](),
            ws.d_nonconst.unsafe_ptr(),
            self.s_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem](),
            self.s_slot.unsafe_ptr(),
            self.s_idx.unsafe_ptr(),
            Int32(n_bound),
            grid_dim=1,
            block_dim=ETL_TPB,
        )

    def enqueue_merge(
        mut self, ctx: DeviceContext, mut ws: LevelWorkspace, n_bound: Int
    ) raises:
        ctx.enqueue_function[etl_merge_kernel](
            self.hdr.unsafe_ptr(),
            self.b_splits.unsafe_ptr().unsafe_bitcast[Split](),
            ws.d_splits.unsafe_ptr().unsafe_bitcast[Split](),
            self.d_pick.unsafe_ptr(),
            self.s_idx.unsafe_ptr(),
            Int32(n_bound),
            grid_dim=ceildiv(n_bound, ETL_TPB),
            block_dim=ETL_TPB,
        )

    def enqueue_push(
        mut self, ctx: DeviceContext, params: DecisionTreeParams, n_bound: Int
    ) raises:
        """`NodeQueue.push` for every tree of the batch: rank, per-tree
        offsets, marks, commit, write (see the kernels)."""
        var chunks = (n_bound + ETL_TPB - 1) // ETL_TPB
        ctx.enqueue_memset(self.cnt, Int32(0))
        ctx.enqueue_function[etl_push_rank_kernel[ETL_TPB]](
            self.hdr.unsafe_ptr(),
            self.b_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem](),
            self.b_slot.unsafe_ptr(),
            self.b_splits.unsafe_ptr().unsafe_bitcast[Split](),
            self.cnt.unsafe_ptr(),
            self.p_rank.unsafe_ptr(),
            self.p_valid.unsafe_ptr(),
            Int32(n_bound),
            Int32(self.g),
            params.min_impurity_decrease,
            params.min_samples_leaf,
            grid_dim=chunks,
            block_dim=ETL_TPB,
        )
        ctx.enqueue_function[etl_push_slot_kernel](
            self.cnt.unsafe_ptr(),
            self.slot_stat.unsafe_ptr(),
            self.slot_base.unsafe_ptr(),
            Int32(chunks),
            Int32(self.g),
            params.max_leaves,
            grid_dim=ceildiv(self.g, ETL_TPB),
            block_dim=ETL_TPB,
        )
        ctx.enqueue_function[etl_push_mark_kernel[ETL_TPB]](
            self.b_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem](),
            self.b_slot.unsafe_ptr(),
            self.b_splits.unsafe_ptr().unsafe_bitcast[Split](),
            self.cnt.unsafe_ptr(),
            self.p_rank.unsafe_ptr(),
            self.p_valid.unsafe_ptr(),
            self.slot_base.unsafe_ptr(),
            self.slot_stat.unsafe_ptr(),
            self.p_left.unsafe_ptr(),
            self.p_kids.unsafe_ptr(),
            self.p_aoff.unsafe_ptr(),
            self.p_eoff.unsafe_ptr(),
            self.x_chunk.unsafe_ptr(),
            Int32(n_bound),
            Int32(self.g),
            params.max_depth,
            params.min_samples_split,
            params.max_leaves,
            grid_dim=chunks,
            block_dim=ETL_TPB,
        )
        ctx.enqueue_function[etl_push_commit_kernel[ETL_TPB]](
            self.hdr.unsafe_ptr(),
            self.x_chunk.unsafe_ptr(),
            Int32(chunks),
            Int32(self.node_cap),
            Int32(self.queue_cap),
            grid_dim=1,
            block_dim=ETL_TPB,
        )
        ctx.enqueue_function[etl_push_write_kernel[ETL_TPB]](
            self.b_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem](),
            self.b_slot.unsafe_ptr(),
            self.b_gpos.unsafe_ptr(),
            self.b_splits.unsafe_ptr().unsafe_bitcast[Split](),
            self.p_left.unsafe_ptr(),
            self.p_kids.unsafe_ptr(),
            self.p_aoff.unsafe_ptr(),
            self.p_eoff.unsafe_ptr(),
            self.x_chunk.unsafe_ptr(),
            self.g_nodes.unsafe_ptr().unsafe_bitcast[
                SparseTreeNode[DType.float32]
            ](),
            self.g_meta.unsafe_ptr(),
            self.queue.unsafe_ptr(),
            Int32(n_bound),
            Int32(self.node_cap),
            Int32(self.queue_cap),
            grid_dim=chunks,
            block_dim=ETL_TPB,
        )

    def enqueue_bf_admit(
        mut self,
        ctx: DeviceContext,
        params: DecisionTreeParams,
        n_items: Int,
        slot_rows: Int32,
    ) raises:
        ctx.enqueue_function[et_bf_admit_kernel](
            self.hdr.unsafe_ptr(),
            self.b_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem](),
            self.b_slot.unsafe_ptr(),
            self.b_gpos.unsafe_ptr(),
            self.b_splits.unsafe_ptr().unsafe_bitcast[Split](),
            self.slot_stat.unsafe_ptr(),
            self.slot_tree.unsafe_ptr(),
            self.f_rec.unsafe_ptr().unsafe_bitcast[FrontierRecord](),
            self.f_gpos.unsafe_ptr(),
            Int32(n_items),
            Int32(self.f_cap),
            slot_rows,
            params.min_impurity_decrease,
            params.min_samples_leaf,
            grid_dim=ceildiv(n_items, ETL_TPB),
            block_dim=ETL_TPB,
        )

    def enqueue_bf_pop(
        mut self,
        ctx: DeviceContext,
        params: DecisionTreeParams,
        bf_sabotage: Int32,
    ) raises:
        ctx.enqueue_function[et_bf_pop_kernel[ETL_TPB]](
            self.hdr.unsafe_ptr(),
            self.slot_stat.unsafe_ptr(),
            self.f_rec.unsafe_ptr().unsafe_bitcast[FrontierRecord](),
            self.f_gpos.unsafe_ptr(),
            self.pt_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem](),
            self.pt_slot.unsafe_ptr(),
            self.pt_gpos.unsafe_ptr(),
            self.pt_splits.unsafe_ptr().unsafe_bitcast[Split](),
            Int32(self.f_cap),
            params.max_leaf_nodes,
            bf_sabotage,
            grid_dim=self.g,
            block_dim=ETL_TPB,
        )

    def enqueue_bf_expand(
        mut self,
        ctx: DeviceContext,
        params: DecisionTreeParams,
        bf_sabotage: Int32,
    ) raises:
        ctx.enqueue_function[et_bf_expand_kernel](
            self.hdr.unsafe_ptr(),
            self.slot_stat.unsafe_ptr(),
            self.pt_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem](),
            self.pt_gpos.unsafe_ptr(),
            self.pt_splits.unsafe_ptr().unsafe_bitcast[Split](),
            self.g_nodes.unsafe_ptr().unsafe_bitcast[
                SparseTreeNode[DType.float32]
            ](),
            self.g_meta.unsafe_ptr(),
            self.b_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem](),
            self.b_slot.unsafe_ptr(),
            self.b_gpos.unsafe_ptr(),
            Int32(self.g),
            Int32(self.node_cap),
            params.max_depth,
            params.min_samples_split,
            params.max_leaves,
            params.max_leaf_nodes,
            bf_sabotage,
            grid_dim=ceildiv(self.g, ETL_TPB),
            block_dim=ETL_TPB,
        )

    def enqueue_scatter(
        mut self,
        ctx: DeviceContext,
        mut d_nodes: DeviceBuffer[DType.uint8],
        mut d_ranges: DeviceBuffer[DType.uint8],
        total_nodes: Int,
    ) raises:
        """The finished arena, tree by tree in local-id order (call after the
        last drain): `tree_base`, then the scatter."""
        ctx.enqueue_function[etl_tree_base_kernel[ETL_TPB]](
            self.slot_stat.unsafe_ptr(),
            self.tree_base.unsafe_ptr(),
            Int32(self.g),
            grid_dim=1,
            block_dim=ETL_TPB,
        )
        if total_nodes > 0:
            ctx.enqueue_function[etl_scatter_kernel](
                self.g_nodes.unsafe_ptr().unsafe_bitcast[
                    SparseTreeNode[DType.float32]
                ](),
                self.g_meta.unsafe_ptr(),
                self.tree_base.unsafe_ptr(),
                d_nodes.unsafe_ptr().unsafe_bitcast[
                    SparseTreeNode[DType.float32]
                ](),
                d_ranges.unsafe_ptr().unsafe_bitcast[InstanceRange](),
                Int32(total_nodes),
                grid_dim=ceildiv(total_nodes, ETL_TPB),
                block_dim=ETL_TPB,
            )

    def download_trees(
        mut self,
        ctx: DeviceContext,
        mut d_nodes: DeviceBuffer[DType.uint8],
        mut d_leaves: DeviceBuffer[DType.float32],
        total_nodes: Int,
        k_out: Int,
        tree_ids: List[Int32],
        first: Int,
        mut trees_out: List[TreeMetaDataNode[DType.float32]],
    ) raises:
        """The group's model, once: per-tree counters and bases, the
        concatenated nodes and leaf values, one synchronize, then each tree's
        `sparsetree` and `vector_leaf` as two block copies."""
        var g = self.g
        var h_stat = ctx.enqueue_create_host_buffer[DType.int32](
            g * ETL_STAT_INTS
        )
        var h_base = ctx.enqueue_create_host_buffer[DType.int32](g + 1)
        var n_alloc = total_nodes if total_nodes > 0 else 1
        var h_nodes = ctx.enqueue_create_host_buffer[DType.uint8](
            n_alloc * size_of[SparseTreeNode[DType.float32]]()
        )
        var h_leaves = ctx.enqueue_create_host_buffer[DType.float32](
            n_alloc * k_out
        )
        ctx.enqueue_copy(dst_buf=h_stat, src_buf=self.slot_stat)
        ctx.enqueue_copy(dst_buf=h_base, src_buf=self.tree_base)
        if total_nodes > 0:
            ctx.enqueue_copy(dst_buf=h_nodes, src_buf=d_nodes)
            ctx.enqueue_copy(dst_buf=h_leaves, src_buf=d_leaves)
        ctx.synchronize()
        var sp = h_stat.unsafe_ptr()
        var bp = h_base.unsafe_ptr()
        var np = h_nodes.unsafe_ptr().unsafe_bitcast[
            SparseTreeNode[DType.float32]
        ]()
        var lp = h_leaves.unsafe_ptr()
        for s in range(g):  # small-loop(g: tree slots in this group): one output tree per slot as two block copies, g capped by group_cap
            var n_s = Int(sp[unsafe_offset = s * ETL_STAT_INTS + ETL_ST_NODES])
            var base = Int(bp[unsafe_offset=s])
            var nodes = List[SparseTreeNode[DType.float32]](
                length=n_s,
                fill=SparseTreeNode[DType.float32].CreateLeafNode(Int32(0)),
            )
            memcpy(dest=nodes.unsafe_ptr(), src=np + base, count=n_s)
            var leaves = List[Float32](length=n_s * k_out, fill=Float32(0.0))
            memcpy(
                dest=leaves.unsafe_ptr(),
                src=lp + base * k_out,
                count=n_s * k_out,
            )
            trees_out.append(
                TreeMetaDataNode[DType.float32](
                    treeid=tree_ids[first + s],
                    depth_counter=sp[
                        unsafe_offset = s * ETL_STAT_INTS + ETL_ST_DEPTH
                    ],
                    leaf_counter=sp[
                        unsafe_offset = s * ETL_STAT_INTS + ETL_ST_LEAVES
                    ],
                    num_outputs=Int32(k_out),
                    vector_leaf=leaves^,
                    sparsetree=nodes^,
                )
            )
        _ = h_stat^
        _ = h_base^
        _ = h_nodes^
        _ = h_leaves^


def _et_search[
    IS_CLF: Bool
](
    ctx: DeviceContext,
    mut ws: LevelWorkspace,
    mut dataset: DeviceDataset,
    mut d_row_ids: DeviceBuffer[DType.int32],
    n_nodes: Int,
    n_blocks: Int,
    k: Int,
    params: DecisionTreeParams,
    n_classes: Int32,
    n_rows: Int32,
    n_cols: Int32,
    seed: UInt64,
    use_sampler: Bool,
    range_only: Bool,
    mut clock: PhaseClock,
) raises:
    """The objective's device search on an already-staged batch."""
    comptime if IS_CLF:
        search_batch_enqueue(
            ctx, ws, dataset, d_row_ids, n_nodes, n_blocks, k, params,
            n_classes, n_rows, n_cols, seed, use_sampler, range_only, clock,
        )
    else:
        search_batch_regression_enqueue(
            ctx, ws, dataset, d_row_ids, n_nodes, n_blocks, k, params,
            n_rows, n_cols, seed, use_sampler, range_only, clock,
        )


def _et_enqueue_partition(
    ctx: DeviceContext,
    mut ws: LevelWorkspace,
    mut dataset: DeviceDataset,
    mut d_row_ids: DeviceBuffer[DType.int32],
    splits: MutPointer[Split, MutAnyOrigin],
    n_part: Int,
    n_blocks: Int,
    params: DecisionTreeParams,
    n_rows: Int32,
) raises:
    """THE PARTITION, on the device (deviation 203), over the batch staged
    with the partition tile: count, scan, scatter, write back. Range-
    addressed, so nodes of different trees partition side by side; `splits`
    is the batch's split per item (an invalid split leaves its node's rows
    alone, `_skip_node`)."""
    comptime TPB = DEVICE_TPB
    ctx.enqueue_function[
        partition_count_kernel[TPB, PART_ROWS_PER_THREAD, ET_PART_FLAGS]
    ](
        ws.d_blk_left.unsafe_ptr(),
        d_row_ids.unsafe_ptr(),
        dataset.d_data.unsafe_ptr(),
        ws.d_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem](),
        ws.d_wl.unsafe_ptr().unsafe_bitcast[WorkloadInfo](),
        splits,
        n_rows,
        params.min_impurity_decrease,
        params.min_samples_leaf,
        PART_MB_SAB_NONE,
        ws.d_part_flags.unsafe_ptr(),
        grid_dim=(n_blocks, 1, 1),
        block_dim=(TPB, 1, 1),
    )
    comptime if not C48:
        ctx.enqueue_function[partition_scan_kernel[TPB, PART_ROWS_PER_THREAD]](
            ws.d_blk_off.unsafe_ptr(),
            ws.d_blk_left.unsafe_ptr(),
            ws.d_blk_base.unsafe_ptr(),
            ws.d_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem](),
            splits,
            params.min_impurity_decrease,
            params.min_samples_leaf,
            PART_MB_SAB_NONE,
            grid_dim=(n_part, 1, 1),
            block_dim=(TPB, 1, 1),
        )
    ctx.enqueue_function[
        partition_scatter_kernel[TPB, PART_ROWS_PER_THREAD, ET_PART_FLAGS, C48]
    ](
        ws.d_row_alt.unsafe_ptr(),
        d_row_ids.unsafe_ptr(),
        ws.d_blk_left.unsafe_ptr() if C48 else ws.d_blk_off.unsafe_ptr(),
        dataset.d_data.unsafe_ptr(),
        ws.d_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem](),
        ws.d_wl.unsafe_ptr().unsafe_bitcast[WorkloadInfo](),
        splits,
        n_rows,
        params.min_impurity_decrease,
        params.min_samples_leaf,
        PART_MB_SAB_NONE,
        ws.d_part_flags.unsafe_ptr(),
        grid_dim=(n_blocks, 1, 1),
        block_dim=(TPB, 1, 1),
    )
    ctx.enqueue_function[partition_writeback_kernel[TPB, PART_ROWS_PER_THREAD]](
        d_row_ids.unsafe_ptr(),
        ws.d_row_alt.unsafe_ptr(),
        ws.d_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem](),
        ws.d_wl.unsafe_ptr().unsafe_bitcast[WorkloadInfo](),
        splits,
        params.min_impurity_decrease,
        params.min_samples_leaf,
        PART_MB_SAB_NONE,
        grid_dim=(n_blocks, 1, 1),
        block_dim=(TPB, 1, 1),
    )


def _et_rescue[
    IS_CLF: Bool
](
    ctx: DeviceContext,
    mut lp: EtDeviceLoop,
    mut ws: LevelWorkspace,
    mut dataset: DeviceDataset,
    mut d_row_ids: DeviceBuffer[DType.int32],
    params: DecisionTreeParams,
    n_classes: Int32,
    n_rows: Int32,
    n_cols: Int32,
    seed: UInt64,
    scalar_tree: Int32,
    n_bound: Int,
    bound_s: Int,
    tile_s: Int,
    tile_p: Int,
    mut clock: PhaseClock,
) raises:
    """DEVIATION 205 on the device (THE DEVICE RESCUE, unchanged in what it
    computes): the nodes whose sampled columns were all constant are
    compacted into a sub-batch, surveyed over every column, given a column
    by `rescue_pick_kernel`, searched with `k = 1`, and the rescued splits
    replace theirs where a column was found. Launched at the batch's bounds;
    an empty sub-batch is all dummies and does nothing."""
    lp.enqueue_retry(ctx, ws, n_bound)
    lp.enqueue_stage(
        ctx, ws, ETL_SRC_SUB, ETL_H_NSUB, n_bound, Int(n_cols), tile_s,
        tile_p, bound_s, scalar_tree,
    )
    _et_search[IS_CLF](
        ctx, ws, dataset, d_row_ids, n_bound, bound_s, Int(n_cols), params,
        n_classes, n_rows, n_cols, seed, False, True, clock,
    )
    ctx.enqueue_function[rescue_pick_kernel](
        lp.d_pick.unsafe_ptr(),
        ws.d_colids.unsafe_ptr(),
        ws.d_min.unsafe_ptr(),
        ws.d_max.unsafe_ptr(),
        ws.d_missing.unsafe_ptr(),
        ws.d_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem](),
        ws.d_tree.unsafe_ptr(),
        Int32(n_bound),
        n_cols,
        seed,
        grid_dim=ceildiv(n_bound, 64),
        block_dim=64,
    )
    lp.enqueue_stage(
        ctx, ws, ETL_SRC_SUB, ETL_H_NSUB, n_bound, 1, tile_s, tile_p,
        bound_s, scalar_tree,
    )
    _et_search[IS_CLF](
        ctx, ws, dataset, d_row_ids, n_bound, bound_s, 1, params, n_classes,
        n_rows, n_cols, seed, False, False, clock,
    )
    lp.enqueue_merge(ctx, ws, n_bound)


def _et_trace_search[
    IS_CLF: Bool
](
    ctx: DeviceContext,
    mut lp: EtDeviceLoop,
    mut ws: LevelWorkspace,
    mut trace: IdentityTrace,
    tag_pre: String,
    k: Int,
) raises:
    """DEVIATION 454's hazard stages for one batch, traced runs only (rule 4:
    a traced run drains per record and is never a timing). The live count
    comes from the header, so the records hold the batch's logical slots."""
    lp.drain(ctx)
    var n = lp.word(ETL_H_CUR)
    if n <= 0:
        return
    trace.record_device(ctx, tag_pre + "colids", ws.d_colids, n * k)
    trace.record_device(ctx, tag_pre + "range.min", ws.d_min, n * k)
    trace.record_device(ctx, tag_pre + "range.max", ws.d_max, n * k)
    trace.record_device(ctx, tag_pre + "draw.thresh", ws.d_thresh, n * k)
    trace.record_device(ctx, tag_pre + "reduce.colid", ws.r_c, n)
    comptime if IS_CLF:
        trace.record_device(ctx, tag_pre + "reduce.num", ws.r_nu, n)
        trace.record_device(ctx, tag_pre + "reduce.den", ws.r_de, n)
    else:
        trace.record_device(ctx, tag_pre + "reduce.gain", ws.r_m, n)


def _et_run_depthwise[
    IS_CLF: Bool
](
    ctx: DeviceContext,
    mut lp: EtDeviceLoop,
    mut ws: LevelWorkspace,
    mut dataset: DeviceDataset,
    mut d_row_ids: DeviceBuffer[DType.int32],
    params: DecisionTreeParams,
    k: Int,
    n_classes: Int32,
    n_rows: Int32,
    n_cols: Int32,
    seed: UInt64,
    sabotage: Int32,
    mut clock: PhaseClock,
    mut trace: IdentityTrace,
    gi: Int,
) raises -> Int:
    """`Builder::train`'s loop for the whole group, depth-wise: per drain,
    `ET_LOOP_K` batches of pop, stage, search, rescue, partition, push.

    Batch `j` after a drain holds at most `min(max_batch, Q * 2^j)` items
    (the FIFO at most doubles per batch) and at most `rows / min_samples_split`
    (queued items are disjoint ranges of at least `min_samples_split` rows);
    every batch's rows are a subset of the queued rows `R`, so its block map
    has at most `1 + n + R / tile` entries. Returns the batch count."""
    comptime TPB = DEVICE_TPB
    var tile_s = TPB * SEARCH_ROWS_PER_THREAD
    var tile_p = TPB * PART_ROWS_PER_THREAD
    var max_batch = Int(params.max_batch_size)
    var msp = Int(params.min_samples_split)
    if msp < 1:
        msp = 1
    var scalar = Int32(1) if sabotage == FOREST_SAB_SCALAR_TREE else Int32(0)
    var k_drain = 1 if trace.enabled else ET_LOOP_K
    var cyc = 0
    while True:
        var head = lp.word(ETL_H_HEAD)
        var tail = lp.word(ETL_H_TAIL)
        if head >= tail:
            break
        var rows = lp.word(ETL_H_ROWS)
        var item_cap = rows // msp
        if item_cap < 1:
            item_cap = 1
        var extra = 0
        var qb = tail - head
        for _ in range(k_drain):  # small-loop(k_drain: batches per drain): sums per-launch bounds, k_drain <= 8
            var nb = qb if qb < max_batch else max_batch
            if nb > item_cap:
                nb = item_cap
            extra += 2 * nb
            if qb < max_batch:
                qb *= 2
        lp.ensure(ctx, extra)
        qb = tail - head
        for _ in range(k_drain):  # small-loop(k_drain: batches per drain): enqueues one batch each, k_drain <= 8
            var nb = qb if qb < max_batch else max_batch
            if nb > item_cap:
                nb = item_cap
            if nb < 1:
                nb = 1
            var bound_s = 1 + nb + rows // tile_s
            if bound_s > ws.cap_blocks:
                bound_s = ws.cap_blocks
            var bound_p = 1 + nb + rows // tile_p
            if bound_p > ws.cap_blocks:
                bound_p = ws.cap_blocks
            var tag_pre = String("g") + String(gi) + ".c" + String(cyc) + "."
            lp.enqueue_pop(ctx, nb, bound_s, tile_s, bound_p, tile_p)
            clock.tick(ctx, PHASE_HOST_QUEUE)
            lp.enqueue_stage(
                ctx, ws, ETL_SRC_BATCH, ETL_H_CUR, nb, k, tile_s, tile_p,
                bound_s, scalar,
            )
            clock.tick(ctx, PHASE_STAGE_BATCH)
            _et_search[IS_CLF](
                ctx, ws, dataset, d_row_ids, nb, bound_s, k, params,
                n_classes, n_rows, n_cols, seed, True, False, clock,
            )
            lp.enqueue_save_splits(ctx, ws, nb)
            if trace.enabled:
                _et_trace_search[IS_CLF](ctx, lp, ws, trace, tag_pre, k)
            _et_rescue[IS_CLF](
                ctx, lp, ws, dataset, d_row_ids, params, n_classes, n_rows,
                n_cols, seed, scalar, nb, bound_s, tile_s, tile_p, clock,
            )
            if trace.enabled:
                lp.drain(ctx)
                var n_live = lp.word(ETL_H_CUR)
                if n_live > 0:
                    trace.record_device(
                        ctx,
                        tag_pre + "split.records",
                        lp.b_splits,
                        n_live * size_of[Split](),
                    )
            lp.enqueue_stage(
                ctx, ws, ETL_SRC_BATCH, ETL_H_CUR, nb, k, tile_p, tile_p,
                bound_p, scalar, search_phase=False,
            )
            _et_enqueue_partition(
                ctx, ws, dataset, d_row_ids,
                lp.b_splits.unsafe_ptr()
                .unsafe_bitcast[Split]()
                .unsafe_origin_cast[MutAnyOrigin](),
                nb, bound_p, params, n_rows,
            )
            clock.tick(ctx, PHASE_PARTITION)
            if trace.enabled:
                trace.record_device(
                    ctx, tag_pre + "partition.rowids", d_row_ids
                )
            lp.enqueue_push(ctx, params, nb)
            clock.tick(ctx, PHASE_HOST_PUSH)
            cyc += 1
            if qb < max_batch:
                qb *= 2
        lp.drain(ctx)
    return cyc


def _et_run_bestfirst[
    IS_CLF: Bool
](
    ctx: DeviceContext,
    mut lp: EtDeviceLoop,
    mut ws: LevelWorkspace,
    mut dataset: DeviceDataset,
    mut d_row_ids: DeviceBuffer[DType.int32],
    params: DecisionTreeParams,
    k: Int,
    n_classes: Int32,
    n_rows: Int32,
    n_cols: Int32,
    seed: UInt64,
    bf_sabotage: Int32,
    slot_rows: Int32,
    mut clock: PhaseClock,
    mut trace: IdentityTrace,
    gi: Int,
) raises -> Int:
    """DEVIATION 466's best-first growth for the whole group, on the device:
    per cycle, search the children the last expansions created (two search
    slots per tree), rescue, ADMIT the valid ones to their trees'
    frontiers, POP each tree's best record, partition the popped nodes and
    EXPAND them. The fit ends when a cycle pops nothing (no frontier with
    budget left), read from the header every `ET_LOOP_K` cycles; cycles
    enqueued after the end are inert. Returns the cycle count."""
    comptime TPB = DEVICE_TPB
    var tile_s = TPB * SEARCH_ROWS_PER_THREAD
    var tile_p = TPB * PART_ROWS_PER_THREAD
    var g = lp.g
    var n_s = 2 * g
    var k_drain = 1 if trace.enabled else ET_LOOP_K
    var cyc = 0
    while True:
        if cyc > 0 and lp.word(ETL_H_POPS) == 0:
            break
        var rows = lp.word(ETL_H_ROWS)
        if rows < 0:
            rows = 0
        lp.ensure(ctx, 2 * g * k_drain)
        var bound_s = 1 + n_s + rows // tile_s
        if bound_s > ws.cap_blocks:
            bound_s = ws.cap_blocks
        var bound_p = 1 + g + rows // tile_p
        if bound_p > ws.cap_blocks:
            bound_p = ws.cap_blocks
        for _ in range(k_drain):  # small-loop(k_drain: cycles per drain): enqueues one cycle each, k_drain <= 8
            var tag_pre = String("g") + String(gi) + ".c" + String(cyc) + "."
            lp.enqueue_stage(
                ctx, ws, ETL_SRC_BATCH, ETL_H_CUR, n_s, k, tile_s, tile_p,
                bound_s, Int32(0),
            )
            _et_search[IS_CLF](
                ctx, ws, dataset, d_row_ids, n_s, bound_s, k, params,
                n_classes, n_rows, n_cols, seed, True, False, clock,
            )
            lp.enqueue_save_splits(ctx, ws, n_s)
            if trace.enabled:
                _et_trace_search[IS_CLF](ctx, lp, ws, trace, tag_pre, k)
            _et_rescue[IS_CLF](
                ctx, lp, ws, dataset, d_row_ids, params, n_classes, n_rows,
                n_cols, seed, Int32(0), n_s, bound_s, tile_s, tile_p, clock,
            )
            lp.enqueue_bf_admit(ctx, params, n_s, slot_rows)
            lp.enqueue_bf_pop(ctx, params, bf_sabotage)
            clock.tick(ctx, PHASE_HOST_QUEUE)
            lp.enqueue_stage(
                ctx, ws, ETL_SRC_PART, ETL_H_GCOUNT, g, k, tile_p, tile_p,
                bound_p, Int32(0), search_phase=False,
            )
            _et_enqueue_partition(
                ctx, ws, dataset, d_row_ids,
                lp.pt_splits.unsafe_ptr()
                .unsafe_bitcast[Split]()
                .unsafe_origin_cast[MutAnyOrigin](),
                g, bound_p, params, n_rows,
            )
            clock.tick(ctx, PHASE_PARTITION)
            if trace.enabled:
                trace.record_device(
                    ctx, tag_pre + "partition.rowids", d_row_ids
                )
            lp.enqueue_bf_expand(ctx, params, bf_sabotage)
            clock.tick(ctx, PHASE_HOST_PUSH)
            cyc += 1
        lp.drain(ctx)
    return cyc


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
    traps register), so their mechanism cannot be reproduced.

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

    # --- identity trace (`core/identity_trace.mojo`) -- NO REFERENCE FILE --------
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
        # implementation has no quantile/binning stage, so deviation 184's resident
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
    comptime if T12:
        # Account for both row-id streams plus each tree's bounded queue;
        # immutable dataset remains shared. IDs/archive positions do not move.
        var per_tree_bytes = max(1, 8 * Int(slot_rows) + Int(params.max_batch_size)*32)
        group_cap = min(group_cap, max(1, T12_BYTES // per_tree_bytes))
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
            # cpu4-forest: the device frontiers hold `f_cap` records per tree;
            # the group is narrowed to fit `ET_BF_FRONTIER_BYTES` (a
            # scheduling cap: the trees do not depend on the grouping).
            var bf_rec = et_bf_frontier_cap(params, slot_rows, bf_sabotage) * (
                size_of[FrontierRecord]() + 4
            )
            var bf_mem_cap = ET_BF_FRONTIER_BYTES // bf_rec
            if bf_mem_cap < 1:
                bf_mem_cap = 1
            if g > bf_mem_cap:
                g = bf_mem_cap
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
        dataset.ensure_row_major(ctx, Int(k))
        var ws = make_level_workspace(
            ctx,
            Int(params.max_batch_size),
            Int32(total_rows),
            n_cols,
            n_classes,
            Int(k),
            TPB,
        )

        # cpu4-forest: THE LEVEL LOOP ON THE DEVICE (`EtDeviceLoop`, see the
        # block comment above it). The group's queue, retry list, rescue
        # merge, push and (best-first) frontiers live on the device; the
        # host enqueues `ET_LOOP_K` batches per drain and reads one header.
        var f_cap = 0
        if bestfirst:
            f_cap = et_bf_frontier_cap(params, slot_rows, bf_sabotage)
        var lp = EtDeviceLoop(ctx, g, Int(params.max_batch_size), f_cap)
        var shared_base = (
            Int32(1) if sabotage == FOREST_SAB_SHARED_ROW_BASE else Int32(0)
        )
        lp.init_roots(
            ctx,
            tree_ids,
            first,
            slot_rows,
            shared_base,
            et_root_expandable(params, slot_rows),
            bestfirst,
        )
        clock.tick(ctx, PHASE_SETUP)
        var cyc: Int
        if bestfirst:
            cyc = _et_run_bestfirst[True](
                ctx, lp, ws, dataset, d_row_ids, params, Int(k), n_classes,
                n_rows, n_cols, seed, bf_sabotage, slot_rows, clock, trace,
                gi,
            )
        else:
            cyc = _et_run_depthwise[True](
                ctx, lp, ws, dataset, d_row_ids, params, Int(k), n_classes,
                n_rows, n_cols, seed, sabotage, clock, trace, gi,
            )

        # --- the LEAF VALUES, ONE launch for the whole group --------------
        # `SetLeafPredictions` (`builder.cuh:556-599`), DEVIATION 214: the
        # group's trees concatenated, one allocation set, one launch. The
        # concatenation is built ON THE DEVICE now (`etl_scatter_kernel`:
        # each tree's nodes in local-id order, its ranges beside them), and
        # the model comes back once (`download_trees`).
        var k_out = Int(n_classes)
        var total_nodes = lp.word(ETL_H_NODES)
        var n_alloc = total_nodes if total_nodes > 0 else 1
        var d_nodes = ctx.enqueue_create_buffer[DType.uint8](
            n_alloc * size_of[SparseTreeNode[DType.float32]]()
        )
        var d_ranges = ctx.enqueue_create_buffer[DType.uint8](
            n_alloc * size_of[InstanceRange]()
        )
        var d_leaves = ctx.enqueue_create_buffer[DType.float32](
            n_alloc * k_out
        )
        var d_visit = ctx.enqueue_create_buffer[DType.int32](n_alloc)
        lp.enqueue_scatter(ctx, d_nodes, d_ranges, total_nodes)
        # `builder.cuh:582` memsets the leaf array before the launch, and an
        # internal node's ZERO IS ITS VALUE. DEVIATION 471: `zero_fill=True`
        # folds that memset and `d_visit`'s into the launch itself -- each
        # block zeroes its OWN node's `num_outputs` slot and visit cell
        # before the IsLeaf early return, and the grid is one block per
        # node over the whole concatenated buffer, so block-exclusive slot
        # ownership covers exactly what the two memsets covered.
        if k_out <= LEAF_MAX_OUT_DEFAULT:
            _enqueue_classification_leaves[LEAF_MAX_OUT_DEFAULT](
                ctx, d_leaves, d_visit, d_nodes, d_ranges, d_row_ids,
                dataset, k_out, total_nodes,
            )
        else:
            _enqueue_classification_leaves[32](
                ctx, d_leaves, d_visit, d_nodes, d_ranges, d_row_ids,
                dataset, k_out, total_nodes,
            )
        lp.download_trees(
            ctx, d_nodes, d_leaves, total_nodes, k_out, tree_ids, first, out
        )
        if trace.enabled:
            # The leaf pass's output for the whole group: the values the
            # model returns, still concatenated across the group's trees.
            trace.record_device(
                ctx, String("g") + String(gi) + ".leaves", d_leaves
            )
        clock.tick(ctx, PHASE_LEAF)
        # DEVIATION 2663's measurement define: what the group's level loop
        # did, from the device header's tallies.
        comptime if is_defined["MOJOLEARN_ET_CYCLE_STATS"]():
            print(
                "ET_CYCLE_STATS group=", gi, " trees=", g, " cycles=", cyc,
                " nodes=", lp.word(ETL_H_STAT_NODES),
                " survey_nodes=", lp.word(ETL_H_STAT_RETRY),
                " rescued=", lp.word(ETL_H_STAT_RESCUED),
                " max_batch=", params.max_batch_size,
            )
        else:
            _ = cyc
        # Mojo frees a buffer at its LAST USE; these must outlive every
        # launch that read them, and `download_trees` synchronized.
        _ = d_nodes^
        _ = d_ranges^
        _ = d_leaves^
        _ = d_visit^
        _ = lp^
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


def search_batch_regression_enqueue(
    ctx: DeviceContext,
    mut ws: LevelWorkspace,
    mut dataset: DeviceDataset,
    mut d_row_ids: DeviceBuffer[DType.int32],
    n_nodes: Int,
    n_blocks: Int,
    k: Int,
    params: DecisionTreeParams,
    n_rows: Int32,
    n_cols: Int32,
    seed: UInt64,
    use_sampler: Bool,
    range_only: Bool,
    mut clock: PhaseClock,
) raises:
    """`search_batch_regression`'s launch sequence on a batch that is ALREADY STAGED on
    the device (cpu4-forest): steps 2 to 8 of `doSplit` with no host list,
    no host staging and no readback. `n_nodes` items sit in `ws.d_items`
    (and `d_tree`, `d_tsalt`, `d_nb`, `d_nc`), `n_blocks` workload entries
    in `ws.d_wl`. The host-list wrapper stages with `stage_batch` and reads
    the splits back; the device level loop (`EtDeviceLoop`) stages with its
    own kernels, launches at proven bounds (dummy items own no block, map
    entries past the live total carry `nodeid == -1`) and reads nothing.
    Outputs stay on the device: `ws.d_splits` (packed), `ws.d_nonconst`,
    and the range cells (`d_min`, `d_max`, `d_missing`) for the survey.
    """
    comptime TPB = DEVICE_TPB
    comptime MAX_ACC = DEVICE_MAX_ACC
    if n_nodes == 0:
        return
    var n_cells = n_nodes * Int(k)
    # T05 operates on exact raw feature values. The optional incumbent code
    # representation has a separate threshold-snapping contract and keeps B.
    # NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
    var fused_small_rows = Int32(ET_SMALL_NODE_TPB) if T05 and not range_only and not dataset.bins_active() else Int32(0)

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

    # DEVIATION 470: TWO fused seeder launches (halves A and B) replace
    # this cycle's six setup enqueues -- the full argument (bit-inert hoist
    # on the in-order queue, capacity extents for the three memset regions,
    # the survey skipping half B outright, the rescue's zero report extent,
    # the Metal 31-binding limit that forced the A/B split, and the refusal
    # list: no seeder fuses into its grid-accumulating consumer, Metal has
    # no grid sync) is at the classification twin's launch. The one twin
    # difference: the class-accumulator extent is `n_cells` (one output),
    # exactly what the old score-init launch passed here.
    # T04: every future larger frontier is initialized at its own extent.
    # Dummy nodes inside n_nodes remain initialized; capacity tails are unread.
    # NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
    var setup_nodes = n_nodes if T04 else ws.cap_nodes
    var setup_report = Int32(sampler_report_len(n_nodes) if T04 else ws.cap_report) if use_sampler else Int32(0)
    var setup_a_extent = n_cells
    if Int(setup_report) > setup_a_extent:
        setup_a_extent = Int(setup_report)
    if setup_nodes > setup_a_extent:
        setup_a_extent = setup_nodes
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
        Int32(setup_nodes),
        grid_dim=ceildiv(setup_a_extent, PHASE_SETUP_TPB),
        block_dim=PHASE_SETUP_TPB,
    )
    if not range_only:
        var setup_b_extent = n_cells
        if setup_nodes > setup_b_extent:
            setup_b_extent = setup_nodes
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
            Int32(setup_nodes),
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

    # cpu3-trees: the rescue's columns are always the DEVICE's (the host
    # survey walk and its host column table are gone): the survey's identity
    # columns are written by `ident_colids_kernel`, the rescue's by
    # `rescue_pick_kernel` (already queued by the caller).
    var dev_colids = not use_sampler
    if range_only and not dev_colids:
        raise Error("the survey (range_only) runs on the device columns only")
    if dev_colids:
        if range_only:
            if Int(k) != Int(n_cols):
                raise Error(
                    "the device survey searches every column; got k = "
                    + String(Int(k))
                )
            ctx.enqueue_function[ident_colids_kernel](
                d_colids.unsafe_ptr(),
                Int32(n_cells),
                n_cols,
                grid_dim=ceildiv(n_cells, 256),
                block_dim=256,
            )
    else:
        # --- feature sampling, WHERE cuML DOES IT (deviation 201) --------
        # cpu4-forest: the device sampler only. The batch's items are
        # staged on the device (by `stage_batch` or the device level loop),
        # so no host list exists to sample from; a target without float64
        # refuses the algo-L arm by name inside `sample_features_device`.
        _ = sample_features_device(
            ctx,
            d_colids.unsafe_ptr(),
            d_samp_scratch.unsafe_ptr(),
            d_samp_report.unsafe_ptr(),
            d_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem](),
            ws.d_tree.unsafe_ptr(),
            n_nodes,
            seed,
            Int(n_cols),
            Int(k),
        )

    clock.tick(ctx, PHASE_STAGE)
    # DEVIATION 470: the range cells were seeded by fused half A above.
    var tiled_range = False
    comptime if ET_RANGE_TILED:
        tiled_range = dataset.has_rm
    if tiled_range and dataset.bins_active():
        ctx.enqueue_function[
            node_feature_range_tiled_kernel[TPB, ET_CODE_TILE, ET_CODE]
        ](
            d_minkey.unsafe_ptr(),
            d_maxkey.unsafe_ptr(),
            d_missing.unsafe_ptr(),
            d_merges.unsafe_ptr(),
            dataset.d_bins_rm.unsafe_ptr(),
            d_row_ids.unsafe_ptr(),
            d_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem](),
            d_wl.unsafe_ptr().unsafe_bitcast[WorkloadInfo](),
            d_colids.unsafe_ptr(),
            n_cols,
            Int32(k),
            dataset.d_quant.unsafe_ptr(),
            grid_dim=(n_blocks, ceildiv(Int(k), ET_CODE_TILE), 1),
            block_dim=(TPB, 1, 1),
        )
    elif fused_small_rows != 0:
        _enqueue_raw_range[True](ctx,ws,dataset,d_row_ids,n_blocks,k,tiled_range)
    else:
        _enqueue_raw_range[False](ctx,ws,dataset,d_row_ids,n_blocks,k,tiled_range)
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
    # DEVIATION 450: drain only for the survey -- see the classification
    # twin's range pass.
    clock.tick(ctx, PHASE_RANGE)

    if range_only:
        return
    # DEVIATION 470: the score cells and the one-output accumulators were
    # seeded by fused half B above (the survey skips half B and returned
    # already).
    var tiled_score = False
    comptime if ET_SCORE_TILED:
        tiled_score = dataset.has_rm
    if tiled_score and dataset.bins_active():
        ctx.enqueue_function[
            node_feature_score_reg_tiled_kernel[TPB, ET_CODE_TILE, ET_CODE]
        ](
            d_nleft.unsafe_ptr(),
            d_ntotal.unsafe_ptr(),
            d_accl.unsafe_ptr(),
            d_acct.unsafe_ptr(),
            d_nblocks.unsafe_ptr(),
            d_min.unsafe_ptr(),
            d_max.unsafe_ptr(),
            d_missing.unsafe_ptr(),
            dataset.d_bins_rm.unsafe_ptr(),
            d_row_ids.unsafe_ptr(),
            dataset.d_labels.unsafe_ptr(),
            d_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem](),
            d_wl.unsafe_ptr().unsafe_bitcast[WorkloadInfo](),
            d_colids.unsafe_ptr(),
            ws.d_tree.unsafe_ptr(),
            n_cols,
            Int32(k),
            seed,
            dataset.d_quant.unsafe_ptr(),
            dataset.d_nbins.unsafe_ptr(),
            grid_dim=(n_blocks, ceildiv(Int(k), ET_CODE_TILE), 1),
            block_dim=(TPB, 1, 1),
        )
    elif tiled_score:
        ctx.enqueue_function[
            node_feature_score_reg_tiled_kernel[TPB, ET_FEATURE_TILE]
        ](
            d_nleft.unsafe_ptr(),
            d_ntotal.unsafe_ptr(),
            d_accl.unsafe_ptr(),
            d_acct.unsafe_ptr(),
            d_nblocks.unsafe_ptr(),
            d_min.unsafe_ptr(),
            d_max.unsafe_ptr(),
            d_missing.unsafe_ptr(),
            dataset.d_data_rm.unsafe_ptr(),
            d_row_ids.unsafe_ptr(),
            dataset.d_labels.unsafe_ptr(),
            d_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem](),
            d_wl.unsafe_ptr().unsafe_bitcast[WorkloadInfo](),
            d_colids.unsafe_ptr(),
            ws.d_tree.unsafe_ptr(),
            n_cols,
            Int32(k),
            seed,
            dataset.d_quant.unsafe_ptr(),
            dataset.d_nbins.unsafe_ptr(),
            grid_dim=(n_blocks, ceildiv(Int(k), ET_FEATURE_TILE), 1),
            block_dim=(TPB, 1, 1),
        )
    else:
        ctx.enqueue_function[
            node_feature_score_kernel[TPB, MAX_ACC, False, ET_ROW_MAJOR]
        ](
            d_nleft.unsafe_ptr(),
            d_ntotal.unsafe_ptr(),
            d_accl.unsafe_ptr(),
            d_acct.unsafe_ptr(),
            d_nblocks.unsafe_ptr(),
            d_min.unsafe_ptr(),
            d_max.unsafe_ptr(),
            d_missing.unsafe_ptr(),
            dataset.search_data_ptr(),
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
            dataset.n_cols,
            grid_dim=search_grid(n_blocks, Int(k)),
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
        Int32(k),
        grid_dim=ceildiv(n_cells, 64),
        block_dim=64,
    )
    if fused_small_rows != 0:
        _enqueue_small_node_splits[False](ctx,ws,dataset,d_row_ids,
            n_nodes,k,Int32(1),seed,params)

    # DEVIATION 470: the reduce cells and the `r_mx` mutexes (over full
    # capacity) were seeded by fused half B above.
    var bpn = ceildiv(Int(k), ET_REDUCE_TPB)
    if bpn < 1 or ET_SPLIT_REDUCE_ONE_BLOCK:
        bpn = 1
    ctx.enqueue_function[split_reduce_kernel[ET_REDUCE_TPB]](
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
        block_dim=(ET_REDUCE_TPB, 1, 1),
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
        # cpu4-forest: the tally runs ON THE DEVICE (two integer counters),
        # so no per-node host walk reads the readback; one line per batch.
        var d_tally = ctx.enqueue_create_buffer[DType.int32](2)
        ctx.enqueue_memset(d_tally, Int32(0))
        ctx.enqueue_function[split_tie_tally_kernel](
            d_tally.unsafe_ptr(),
            r_c.unsafe_ptr(),
            ws.d_ties.unsafe_ptr(),
            Int32(n_nodes),
            grid_dim=ceildiv(n_nodes, 64),
            block_dim=64,
        )
        var h_tally = ctx.enqueue_create_host_buffer[DType.int32](2)
        ctx.enqueue_copy(dst_buf=h_tally, src_buf=d_tally)
        ctx.synchronize()
        print(
            "ET_TIE_STATS batch decided=",
            h_tally.unsafe_ptr()[unsafe_offset=0],
            " tied=",
            h_tally.unsafe_ptr()[unsafe_offset=1],
        )
        _ = d_tally^
        _ = h_tally^

    if dataset.bins_active():
        ctx.enqueue_function[et_code_threshold_kernel](
            r_q.unsafe_ptr(), r_c.unsafe_ptr(), dataset.d_quant.unsafe_ptr(),
            dataset.d_nbins.unsafe_ptr(),
            Int32(n_nodes), grid_dim=ceildiv(n_nodes, 64), block_dim=64,
        )
    # cpu3-trees: the batch's `Split` records are packed ON THE DEVICE
    # (`pack_splits_kernel`: the invalid-candidate metric mask included)
    # and cross as one block, copied into the host scheduler's list whole.
    ctx.enqueue_function[pack_splits_kernel](
        ws.d_splits.unsafe_ptr().unsafe_bitcast[Split](),
        r_q.unsafe_ptr(),
        r_c.unsafe_ptr(),
        r_m.unsafe_ptr(),
        r_v.unsafe_ptr(),
        r_l.unsafe_ptr(),
        Int32(n_nodes),
        grid_dim=ceildiv(n_nodes, 64),
        block_dim=64,
    )


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
    if len(host_colids) != 0:
        raise Error(
            "host_colids: caller-chosen columns were the host rescue walk,"
            " removed (cpu3-trees); pass an empty list"
        )
    # DEVIATION 2020: the search tile is `TPB * SEARCH_ROWS_PER_THREAD`
    # (default 1 = the exact pre-2020 program). The full block, with the
    # bit argument and the required-RED arm, is at the classification
    # twin's call site; the two twins must widen together or the two
    # objectives would launch different grids for the same frontier.
    var plan = build_workload_info(
        work_items, TPB * SEARCH_ROWS_PER_THREAD
    )
    stage_batch(ctx, ws, work_items, item_trees, plan, Int(k))
    search_batch_regression_enqueue(
        ctx, ws, dataset, d_row_ids, n_nodes, plan.n_blocks_dimx, Int(k),
        params, n_rows, n_cols, seed, use_sampler, range_only, clock,
    )
    ctx.enqueue_copy(dst_buf=ws.h_nonconst, src_buf=ws.d_nonconst)
    if range_only:
        return (
            List[Split](), List[Int32](), List[Float32](), List[Float32](),
            List[Int32](),
        )
    ctx.enqueue_copy(dst_buf=ws.h_splits, src_buf=ws.d_splits)
    ref o_c = ws.o_c
    ref o_m = ws.o_m
    ctx.enqueue_copy(dst_buf=o_c, src_buf=ws.r_c)
    ctx.enqueue_copy(dst_buf=o_m, src_buf=ws.r_m)
    ctx.synchronize()
    clock.tick(ctx, PHASE_REDUCE)

    # DEVIATION 450: the deferred `h_nonconst` read -- see the twin.
    var any_nonconst = List[Int32](length=n_nodes, fill=Int32(0))
    memcpy(
        dest=any_nonconst.unsafe_ptr(),
        src=ws.h_nonconst.unsafe_ptr(),
        count=n_nodes,
    )

    var splits = List[Split](length=n_nodes, fill=Split())
    memcpy(
        dest=splits.unsafe_ptr(),
        src=ws.h_splits.unsafe_ptr().unsafe_bitcast[Split](),
        count=n_nodes,
    )

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

    # --- identity trace -- NO REFERENCE FILE; see the classification twin --------
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
    comptime if T12:
        # Account for both row-id streams plus each tree's bounded queue;
        # immutable dataset remains shared. IDs/archive positions do not move.
        var per_tree_bytes = max(1, 8 * Int(slot_rows) + Int(params.max_batch_size)*32)
        group_cap = min(group_cap, max(1, T12_BYTES // per_tree_bytes))
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
            # cpu4-forest: the device frontiers hold `f_cap` records per tree;
            # the group is narrowed to fit `ET_BF_FRONTIER_BYTES` (a
            # scheduling cap: the trees do not depend on the grouping).
            var bf_rec = et_bf_frontier_cap(params, slot_rows, bf_sabotage) * (
                size_of[FrontierRecord]() + 4
            )
            var bf_mem_cap = ET_BF_FRONTIER_BYTES // bf_rec
            if bf_mem_cap < 1:
                bf_mem_cap = 1
            if g > bf_mem_cap:
                g = bf_mem_cap
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

        dataset.ensure_row_major(ctx, Int(k))
        comptime if ET_RANGE_TILED and ET_SCORE_TILED:
            if dataset.has_rm and Int(dataset.n_cols) >= ET_BINNED_MIN_COLS:
                dataset.ensure_binned(ctx)
        var ws = make_level_workspace(
            ctx,
            Int(params.max_batch_size),
            Int32(total_rows),
            n_cols,
            1,
            Int(k),
            TPB,
        )

        # cpu4-forest: THE LEVEL LOOP ON THE DEVICE (`EtDeviceLoop`, see the
        # block comment above it). The group's queue, retry list, rescue
        # merge, push and (best-first) frontiers live on the device; the
        # host enqueues `ET_LOOP_K` batches per drain and reads one header.
        var f_cap = 0
        if bestfirst:
            f_cap = et_bf_frontier_cap(params, slot_rows, bf_sabotage)
        var lp = EtDeviceLoop(ctx, g, Int(params.max_batch_size), f_cap)
        var shared_base = (
            Int32(1) if sabotage == FOREST_SAB_SHARED_ROW_BASE else Int32(0)
        )
        lp.init_roots(
            ctx,
            tree_ids,
            first,
            slot_rows,
            shared_base,
            et_root_expandable(params, slot_rows),
            bestfirst,
        )
        clock.tick(ctx, PHASE_SETUP)
        var cyc: Int
        if bestfirst:
            cyc = _et_run_bestfirst[False](
                ctx, lp, ws, dataset, d_row_ids, params, Int(k), Int32(1),
                n_rows, n_cols, seed, bf_sabotage, slot_rows, clock, trace,
                gi,
            )
        else:
            cyc = _et_run_depthwise[False](
                ctx, lp, ws, dataset, d_row_ids, params, Int(k), Int32(1),
                n_rows, n_cols, seed, sabotage, clock, trace, gi,
            )

        # --- the LEAF VALUES, ONE launch for the whole group --------------
        # `SetLeafPredictions` (`builder.cuh:556-599`), DEVIATION 214: the
        # group's trees concatenated, one allocation set, one launch. The
        # concatenation is built ON THE DEVICE now (`etl_scatter_kernel`:
        # each tree's nodes in local-id order, its ranges beside them), and
        # the model comes back once (`download_trees`).
        var k_out = 1
        var total_nodes = lp.word(ETL_H_NODES)
        var n_alloc = total_nodes if total_nodes > 0 else 1
        var d_nodes = ctx.enqueue_create_buffer[DType.uint8](
            n_alloc * size_of[SparseTreeNode[DType.float32]]()
        )
        var d_ranges = ctx.enqueue_create_buffer[DType.uint8](
            n_alloc * size_of[InstanceRange]()
        )
        var d_leaves = ctx.enqueue_create_buffer[DType.float32](
            n_alloc * k_out
        )
        var d_visit = ctx.enqueue_create_buffer[DType.int32](n_alloc)
        lp.enqueue_scatter(ctx, d_nodes, d_ranges, total_nodes)
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
        lp.download_trees(
            ctx, d_nodes, d_leaves, total_nodes, k_out, tree_ids, first, out
        )
        if trace.enabled:
            # The leaf pass's output for the whole group: the values the
            # model returns, still concatenated across the group's trees.
            trace.record_device(
                ctx, String("g") + String(gi) + ".leaves", d_leaves
            )
        clock.tick(ctx, PHASE_LEAF)
        # DEVIATION 2663's measurement define: what the group's level loop
        # did, from the device header's tallies.
        comptime if is_defined["MOJOLEARN_ET_CYCLE_STATS"]():
            print(
                "ET_CYCLE_STATS group=", gi, " trees=", g, " cycles=", cyc,
                " nodes=", lp.word(ETL_H_STAT_NODES),
                " survey_nodes=", lp.word(ETL_H_STAT_RETRY),
                " rescued=", lp.word(ETL_H_STAT_RESCUED),
                " max_batch=", params.max_batch_size,
            )
        else:
            _ = cyc
        # Mojo frees a buffer at its LAST USE; these must outlive every
        # launch that read them, and `download_trees` synchronized.
        _ = d_nodes^
        _ = d_ranges^
        _ = d_leaves^
        _ = d_visit^
        _ = lp^
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
