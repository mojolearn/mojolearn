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
`NodeQueue` is complete. `Builder` (`builder.cuh:140-601`) is NOT ported yet:
it is the device workspace, the feature sampler launch, `computeSplit` and
`SetLeafPredictions`, all of which need kernels this lane has not written. The
piece of `Builder` that is pure control flow -- `train()`, `builder.cuh:344-359`
-- is the four-line loop `while queue.HasWork(): pop, doSplit, push`, and it is
transcribed in `train_loop_shape` below as a docstring rather than as code,
because a loop calling a function that does not exist is not a port, it is a
placeholder pretending to be one (rule 3: an unported file is visible, a
mis-ported one is not).

THE ONE INVARIANT EVERYTHING DOWNSTREAM RESTS ON
-------------------------------------------------
Children are allocated as an ADJACENT PAIR and only the left index is stored
(`builder.cuh:112-129` pushes left then right; `flatnode.h:56` returns
`left_child_id + 1` for the right). So `sparsetree` and `node_instances_` grow
in lockstep and stay the same length -- which is what lets
`SetLeafPredictions` zip them (`builder.cuh:562-563` asserts exactly that).
"""

from extratrees.mojo_only.host_splitter import (
    node_split_random_gini,
    node_split_random_mse,
)
from extratrees.ported.decisiontree.decisiontree import (
    DecisionTreeParams,
    validity_check,
)
from extratrees.ported.decisiontree.flatnode import (
    SparseTreeNode,
    TreeMetaDataNode,
)
from extratrees.ported.decisiontree.batched_levelalgo.dataset import Dataset
from extratrees.ported.decisiontree.batched_levelalgo.objectives import (
    AggregateBin,
    CountBin,
    GiniObjectiveFunction,
    MSEObjectiveFunction,
)
from extratrees.ported.decisiontree.batched_levelalgo.split import Split
from extratrees.ported.decisiontree.batched_levelalgo.kernels.builder_kernels import (
    FeatureSamplerPlan,
    InstanceRange,
    NodeWorkItem,
    SAMPLE_ALGO_L,
    SAMPLER_UNVISITED,
    WorkloadInfo,
    device_has_float64,
    plan_feature_sampling,
    sample_features,
    sample_features_device,
    sampler_report_len,
    sampler_scratch_len,
    split_not_valid,
)
from extratrees.ported.decisiontree.batched_levelalgo.kernels.builder_kernels_impl import (
    node_feature_range_decode_kernel,
    LEAF_MAX_OUT_DEFAULT,
    LEAF_SAB_NONE,
    PARTITION_UNVISITED,
    PART_SAB_NONE,
    RANGE_SAB_NONE,
    SCORE_STATUS_SCORED,
    build_workload_info,
    leaf_kernel,
    node_split_kernel,
    node_feature_range_init_kernel,
    node_feature_range_kernel,
    node_feature_score_finalize_kernel,
    node_feature_score_init_kernel,
    node_feature_score_kernel,
    partition_samples,
    score_row_bound_ok,
)
from extratrees.ported.decisiontree.flatnode import SparseTreeNode
from extratrees.ported.decisiontree.batched_levelalgo.kernels.partition_multiblock import (
    PART_MB_SAB_NONE,
    partition_count_kernel,
    partition_scan_kernel,
    partition_scatter_kernel,
    partition_writeback_kernel,
)
from extratrees.ported.decisiontree.batched_levelalgo.split import (
    split_reduce_init_kernel,
    split_reduce_kernel,
)
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.math import ceildiv, fma
from std.sys.info import size_of


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

    def __init__(
        out self,
        params: DecisionTreeParams,
        sampled_rows: Int32,
        num_outputs: Int32,
        treeid: Int32 = 0,
    ):
        """`builder.cuh:53-65`.

        The root is created as a LEAF holding every sampled row, and is pushed
        as work only if it is expandable -- so `max_depth == 0` yields a
        one-node tree with no work at all, which is their behaviour and is a
        case the check covers.
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
        self.node_instances.append(InstanceRange(0, sampled_rows))
        self.work_items = List[NodeWorkItem]()
        self.head = 0
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
        var result = List[NodeWorkItem]()
        while (
            self.head < len(self.work_items)
            and len(result) < Int(self.params.max_batch_size)
        ):
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
        """`builder.cuh:91-134`: turn a batch of splits into nodes and work.

        Transcribed in their order. ONE HALF of that order is load-bearing and
        the other half is not, and the difference was MEASURED rather than
        argued: the `max_leaves` test at `:105` sits AFTER the validity
        `continue` at `:99`, so an invalid split does not consume leaf budget
        -- sabotaging that turns `builder_check` red. But their `break` at
        `:106` is EQUIVALENT to a `continue` here, because `leaf_counter` only
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

            # `:99-103`
            if split_not_valid(
                split,
                self.params.min_impurity_decrease,
                self.params.min_samples_leaf,
                parent_range.count,
            ):
                continue

            # `:105-106` -- a BREAK, not a continue.
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


def train_classification(
    dataset: Dataset,
    params: DecisionTreeParams,
    tree_id: Int32,
    seed: UInt64,
    n_classes: Int32,
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

    `doSplit` (`builder.cuh:379-475`) is inlined below because its device half
    is unported: theirs samples features, launches `computeSplitKernel` per
    column block, then launches `nodeSplitKernel` which partitions. Ours does
    the same three steps in the same order with the host forms, which are the
    oracles the kernels will be checked against.

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

    var objective = GiniObjectiveFunction[DType.float32](
        n_classes, params.min_samples_leaf
    )
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

            # `computeSplitKernel`'s replacement, DEVIATION 137.
            var result = node_split_random_gini[DType.float32](
                dataset, item, my_colids, objective, seed, tree_id
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
) raises -> TreeMetaDataNode[DType.float32]:
    """The same loop for MSE. See `train_classification` for the structure and
    for why the partition precedes the push."""
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
    return gain


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
):
    """Scored cells into reduction candidates, elementwise.

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
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while idx < n_cells:
        if in_status[unsafe_offset=idx] == SCORE_STATUS_SCORED:
            cand_quesval[unsafe_offset=idx] = in_threshold[unsafe_offset=idx]
            cand_colid[unsafe_offset=idx] = colids[unsafe_offset=idx]
            cand_metric[unsafe_offset=idx] = gain_per_split(
                in_acc_left,
                in_acc_total,
                idx * n_classes,
                n_classes,
                in_n_total[unsafe_offset=idx],
                in_n_left[unsafe_offset=idx],
                min_samples_leaf,
            )
            cand_nleft[unsafe_offset=idx] = in_n_left[unsafe_offset=idx]
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
    os: MutOrigin, orp: MutOrigin, ow: MutOrigin, //
](
    ctx: DeviceContext,
    mut d_colids_buf: DeviceBuffer[DType.int32],
    d_scratch: MutPointer[Int32, os],
    d_report: MutPointer[Int32, orp],
    d_work_items: MutPointer[NodeWorkItem, ow],
    mut h_colids_stage: HostBuffer[DType.int32],
    work_items: List[NodeWorkItem],
    tree_id: Int32,
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
            len(work_items),
            tree_id,
            seed,
            n,
            k,
        )
    var host_colids = List[Int32](
        length=len(work_items) * k, fill=Int32(0)
    )
    _ = sample_features(host_colids, work_items, tree_id, seed, n, k)
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
    PER TREE, NOT ONCE PER LEVEL.

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
    `split_reduce_init_kernel`, or an `enqueue_memset` -- so a reused
    buffer and a fresh one are indistinguishable to every kernel that
    reads one. The identity checks could not see it and did not.

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
    var cells = nodes * k
    var blocks = 1 + max_batch + Int(n_rows) // tpb
    return LevelWorkspace(
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
    )


def train_classification_device_resident(
    ctx: DeviceContext,
    mut dataset: DeviceDataset,
    mut row_ids: List[Int32],
    params: DecisionTreeParams,
    tree_id: Int32,
    seed: UInt64,
) raises -> TreeMetaDataNode[DType.float32]:
    """One ExtraTree with its split search on the GPU.

    `Builder::train` (`builder.cuh:344-359`) with `doSplit` (`:379-494`)
    enqueued. Per batch, in their order:

      1. host: sample features into `colids` (`:398-471`);
      2. host: `updateWorkloadInfo` flattens the ragged batch (`:365-385`);
      3. device: the range pass -- step 1 of DEVIATION 137;
      4. device: the draw and score pass -- steps 2-4;
      5. device: scored cells into reduction candidates (DEVIATION 182);
      6. device: `evalBestSplit`'s reduction (`split.cuh:107-152`);
      7. host: copy the batch's splits back, exactly as `:492-494` does;
      8. host: partition the winners and push the children.

    ROW_IDS IS RE-UPLOADED EVERY BATCH, because the partition is still on the
    host and it is the host copy that is authoritative between levels. That is
    a WIRING state, not a design: when the device partition lands, `row_ids`
    stays resident and this upload becomes the initial one only. It is called
    out here rather than left to be discovered because it is the difference
    between a device split search and a device tree build, and this function
    currently is the former.
    """
    var n_rows = dataset.n_rows
    var n_cols = dataset.n_cols
    var n_classes = dataset.n_classes
    validity_check(params)

    comptime TPB = 128
    comptime MAX_ACC = 32
    if Int(n_classes) > MAX_ACC:
        raise Error(
            "the device score kernel is built for at most "
            + String(MAX_ACC)
            + " classes; got "
            + String(n_classes)
            + " (DEVIATION 172: shared sizing is comptime here)"
        )
    if not score_row_bound_ok(Int(n_rows)):
        raise Error(
            "n_rows "
            + String(n_rows)
            + " exceeds the row count at which the published Int64 Gini"
            " numerator is exact; see SCORE_MAX_ROWS_EXACT and DEVIATION 175"
        )

    var k = n_sampled_cols_for(params, n_cols)
    var objective = GiniObjectiveFunction[DType.float32](
        n_classes, params.min_samples_leaf
    )
    var queue = NodeQueue[DType.float32](params, n_rows, n_classes, tree_id)

    # `d_data` and `d_labels` are resident (deviation 184). `d_row_ids` is
    # per-tree and is FILLED ON THE DEVICE by a sequence kernel, which is what
    # `get_row_sample` does (`randomforest.cuh:69`, `thrust::sequence`) --
    # deviation 200. No row list is uploaded.
    var d_row_ids = ctx.enqueue_create_buffer[DType.int32](Int(n_rows))
    ctx.enqueue_function[row_ids_sequence_kernel](
        d_row_ids.unsafe_ptr(),
        n_rows,
        grid_dim=ceildiv(Int(n_rows), 128),
        block_dim=128,
    )
    # `row_ids` is the caller's host list. The device path does not read it --
    # deviation 185 already measured that its `mut` never meant anything here,
    # and deviation 200 removed the one place it was read.
    _ = row_ids

    # THE WORKSPACE, ONCE, as `assignWorkspace` does (`builder.cuh:302-341`).
    # Deviation 202. Every buffer below is sized to the CAPACITY the batch
    # can reach, not to this level's occupancy, which is what lets one
    # allocation serve every level.
    var ws = make_level_workspace(
        ctx,
        Int(params.max_batch_size),
        n_rows,
        n_cols,
        n_classes,
        Int(k),
        TPB,
    )

    while queue.has_work():
        var work_items = queue.pop()
        var n_nodes = len(work_items)

        # --- 1. feature sampling, on the host, as theirs is -------------


        # --- 2. the ragged-batch flattening ------------------------------
        var plan = build_workload_info(work_items, TPB)
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
        # shared one would be rewritten under an in-flight copy.
        ref h_colids = ws.h_colids
        ref h_items = ws.h_items
        ref h_wl = ws.h_wl
        ref h_nb = ws.h_nb
        ref h_nc = ws.h_nc
        ctx.synchronize()

        var items_ptr = h_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem]()
        for i in range(n_nodes):
            items_ptr[unsafe_offset=i] = work_items[i]
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

        ctx.enqueue_copy(dst_buf=d_items, src_ptr=h_items.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=d_wl, src_ptr=h_wl.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=d_nb, src_ptr=h_nb.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=d_nc, src_ptr=h_nc.unsafe_ptr())
        ctx.enqueue_copy(
            dst_buf=ws.d_blk_base, src_ptr=ws.h_blk_base.unsafe_ptr()
        )
        ctx.synchronize()

        # --- 3. the range pass -------------------------------------------
        # --- feature sampling, WHERE cuML DOES IT (deviation 201) --------
        # `d_report` is the DEVICE's own statement of which kernel ran, so it
        # is seeded with a value no kernel can produce.
        ctx.enqueue_memset(d_samp_report, SAMPLER_UNVISITED)
        _ = sample_features_for_device(
            ctx,
            d_colids,
            d_samp_scratch.unsafe_ptr(),
            d_samp_report.unsafe_ptr(),
            d_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem](),
            h_colids,
            work_items,
            tree_id,
            seed,
            Int(n_cols),
            Int(k),
        )

        ctx.enqueue_function[node_feature_range_init_kernel](
            d_min.unsafe_ptr(),
            d_max.unsafe_ptr(),
            d_minkey.unsafe_ptr(),
            d_maxkey.unsafe_ptr(),
            d_missing.unsafe_ptr(),
            d_merges.unsafe_ptr(),
            Int32(n_cells),
            grid_dim=ceildiv(n_cells, 64),
            block_dim=64,
        )
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
            k,
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

        # --- 4. the draw and score pass ----------------------------------
        ctx.enqueue_function[node_feature_score_init_kernel](
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
            Int32(n_cells * Int(n_classes)),
            grid_dim=ceildiv(n_cells, 64),
            block_dim=64,
        )
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
            n_rows,
            k,
            n_classes,
            seed,
            tree_id,
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
            Int32(n_cells),
            k,
            n_classes,
            seed,
            tree_id,
            params.min_samples_leaf,
            Int32(0),
            grid_dim=ceildiv(n_cells, 64),
            block_dim=64,
        )

        # --- 5. scored cells into candidates (DEVIATION 182) -------------
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
            grid_dim=ceildiv(n_cells, 64),
            block_dim=64,
        )

        # --- 6. evalBestSplit's reduction ---------------------------------
        var bpn = ceildiv(Int(k), TPB)
        if bpn < 1:
            bpn = 1
        ctx.enqueue_memset(r_mx, Int32(0))
        ctx.enqueue_function[split_reduce_init_kernel](
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
            grid_dim=ceildiv(n_nodes, 64),
            block_dim=64,
        )
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
            Int32(bpn),
            Int32(0),
            grid_dim=(bpn, n_nodes, 1),
            block_dim=(TPB, 1, 1),
        )

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

        # --- 9. the PARTITION, on the device ------------------------------
        # `nodeSplitKernel` (`builder_kernels_impl.cuh:89-107`) and the
        # `partitionSamples` it calls. `row_ids` is mutated IN PLACE on the
        # device, which is what lets it stay resident: it is uploaded once,
        # above, and never round-trips again.
        ref d_splits = ws.d_splits
        ref d_iters = ws.d_iters
        ref d_swaps = ws.d_swaps
        ref h_splits = ws.h_splits
        ctx.synchronize()
        var splits_ptr = h_splits.unsafe_ptr().unsafe_bitcast[Split]()
        for i in range(n_nodes):
            splits_ptr[unsafe_offset=i] = splits[i]
        ctx.enqueue_copy(dst_buf=d_splits, src_ptr=h_splits.unsafe_ptr())
        # DEVIATION 203: three passes over the frontier's ROWS instead of
        # one block per node, on the same `WorkloadInfo` grid the split
        # search uses. `d_iters`/`d_swaps` are the one-block kernel's
        # report and this path does not produce them; that kernel stays in
        # the tree as the oracle `partition_multiblock_check` compares
        # against.
        ctx.enqueue_function[partition_count_kernel[TPB]](
            ws.d_blk_left.unsafe_ptr(),
            d_row_ids.unsafe_ptr(),
            dataset.d_data.unsafe_ptr(),
            d_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem](),
            d_wl.unsafe_ptr().unsafe_bitcast[WorkloadInfo](),
            d_splits.unsafe_ptr().unsafe_bitcast[Split](),
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
            d_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem](),
            d_splits.unsafe_ptr().unsafe_bitcast[Split](),
            params.min_impurity_decrease,
            params.min_samples_leaf,
            PART_MB_SAB_NONE,
            grid_dim=(n_nodes, 1, 1),
            block_dim=(TPB, 1, 1),
        )
        ctx.enqueue_function[partition_scatter_kernel[TPB]](
            ws.d_row_alt.unsafe_ptr(),
            d_row_ids.unsafe_ptr(),
            ws.d_blk_off.unsafe_ptr(),
            dataset.d_data.unsafe_ptr(),
            d_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem](),
            d_wl.unsafe_ptr().unsafe_bitcast[WorkloadInfo](),
            d_splits.unsafe_ptr().unsafe_bitcast[Split](),
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
            d_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem](),
            d_wl.unsafe_ptr().unsafe_bitcast[WorkloadInfo](),
            d_splits.unsafe_ptr().unsafe_bitcast[Split](),
            params.min_impurity_decrease,
            params.min_samples_leaf,
            PART_MB_SAB_NONE,
            grid_dim=(plan.n_blocks_dimx, 1, 1),
            block_dim=(TPB, 1, 1),
        )
        ctx.synchronize()
        _ = d_iters
        _ = d_swaps

        queue.push(work_items, splits)
        _ = objective

    # --- 10. the LEAF VALUES, on the device -------------------------------
    # `SetLeafPredictions` (`builder.cuh:556-599`) and the `leafKernel` it
    # launches. `row_ids` is still on the device, in the order the partitions
    # left it, so nothing is uploaded here -- only the tree and its ranges.
    var tree = queue.get_tree()
    var n_nodes_final = tree.num_nodes()
    var k_out = Int(tree.num_outputs)

    var d_nodes = ctx.enqueue_create_buffer[DType.uint8](
        n_nodes_final * size_of[SparseTreeNode[DType.float32]]()
    )
    var d_ranges = ctx.enqueue_create_buffer[DType.uint8](
        n_nodes_final * size_of[InstanceRange]()
    )
    var d_leaves = ctx.enqueue_create_buffer[DType.float32](
        n_nodes_final * k_out
    )
    var d_visit = ctx.enqueue_create_buffer[DType.int32](n_nodes_final)
    var h_nodes = ctx.enqueue_create_host_buffer[DType.uint8](
        n_nodes_final * size_of[SparseTreeNode[DType.float32]]()
    )
    var h_ranges = ctx.enqueue_create_host_buffer[DType.uint8](
        n_nodes_final * size_of[InstanceRange]()
    )
    var h_leaves = ctx.enqueue_create_host_buffer[DType.float32](
        n_nodes_final * k_out
    )
    ctx.synchronize()
    var nodes_ptr = h_nodes.unsafe_ptr().unsafe_bitcast[
        SparseTreeNode[DType.float32]
    ]()
    for i in range(n_nodes_final):
        nodes_ptr[unsafe_offset=i] = tree.sparsetree[i]
    var ranges_ptr = h_ranges.unsafe_ptr().unsafe_bitcast[InstanceRange]()
    for i in range(n_nodes_final):
        ranges_ptr[unsafe_offset=i] = queue.node_instances[i]
    ctx.enqueue_copy(dst_buf=d_nodes, src_ptr=h_nodes.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_ranges, src_ptr=h_ranges.unsafe_ptr())
    # `builder.cuh:582` memsets the leaf array before the launch, and an
    # internal node's ZERO IS ITS VALUE -- the kernel returns early for a
    # non-leaf and never writes those slots.
    ctx.enqueue_memset(d_leaves, Float32(0.0))
    ctx.enqueue_memset(d_visit, Int32(0))
    ctx.enqueue_function[
        leaf_kernel[TPB, LEAF_MAX_OUT_DEFAULT, True]
    ](
        d_leaves.unsafe_ptr(),
        d_visit.unsafe_ptr(),
        d_nodes.unsafe_ptr().unsafe_bitcast[SparseTreeNode[DType.float32]](),
        d_ranges.unsafe_ptr().unsafe_bitcast[InstanceRange](),
        d_row_ids.unsafe_ptr(),
        dataset.d_labels.unsafe_ptr(),
        Int32(k_out),
        Float32(1.0),  # classification: no fixed-point rescale
        LEAF_SAB_NONE,
        grid_dim=(n_nodes_final, 1, 1),
        block_dim=(TPB, 1, 1),
    )
    ctx.enqueue_copy(dst_buf=h_leaves, src_buf=d_leaves)
    ctx.synchronize()

    tree.vector_leaf = List[Float32](
        length=n_nodes_final * k_out, fill=Float32(0.0)
    )
    for i in range(n_nodes_final * k_out):
        tree.vector_leaf[i] = h_leaves.unsafe_ptr()[unsafe_offset=i]
    return tree^


def dataset_len_ok(
    x_col_major: List[Float32], n_rows: Int32, n_cols: Int32
) -> Bool:
    return len(x_col_major) == Int(n_rows) * Int(n_cols)


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
    """One regression ExtraTree with its split search on the GPU.

    The SAME loop as `train_classification_device_resident` -- `Builder::train`
    (`builder.cuh:344-359`) around `doSplit` (`:379-494`) -- with the two
    kernels instantiated for `CLASSIFICATION = False` and cuML's
    `MSEObjectiveFunction` in place of their Gini one. Their builder is a
    template over `ObjectiveT` for exactly this reason
    (`builder.cuh:142 template <typename ObjectiveT> struct Builder`), so
    swapping the objective and keeping the loop is their structure, not a
    parallel implementation.

    `labels_q` is the label vector ALREADY QUANTIZED by
    `fixed_point.choose_scale` / `quantize` -- deviation 135's ruling, and
    deviation 174's rule that the device sees only integers. `scale` is the
    multiplier that produced it, needed only to put leaf values back in the
    label's own units.

    THE REGRESSION KEY IS cuML's OWN MSE GAIN, not sklearn's proxy: deviation
    189 measured that sklearn's numerator is unpublishable in `Int64` above
    NINE ROWS and wraps negative, and that cuML's gain -- which is that proxy
    minus two node constants -- fits with no row cap at all. So the reduction
    ranks regression candidates by their quantity, which is the one this
    builder is templated on anyway.
    """
    if len(x_col_major) != Int(n_rows) * Int(n_cols):
        raise Error("x_col_major must be n_rows * n_cols long, column major")
    if len(labels_q) != Int(n_rows):
        raise Error("labels_q must be n_rows long")
    validity_check(params)

    comptime TPB = 128
    comptime MAX_ACC = 32

    var k = n_sampled_cols_for(params, n_cols)
    var queue = NodeQueue[DType.float32](params, n_rows, 1, tree_id)

    var dataset = upload_dataset(
        ctx, x_col_major, labels_q, n_rows, n_cols, 1
    )

    var d_row_ids = ctx.enqueue_create_buffer[DType.int32](Int(n_rows))
    ctx.enqueue_function[row_ids_sequence_kernel](
        d_row_ids.unsafe_ptr(),
        n_rows,
        grid_dim=ceildiv(Int(n_rows), 128),
        block_dim=128,
    )
    _ = row_ids

    while queue.has_work():
        var work_items = queue.pop()
        var n_nodes = len(work_items)

        var plan = build_workload_info(work_items, TPB)
        var n_cells = n_nodes * Int(k)

        var d_min = ctx.enqueue_create_buffer[DType.float32](n_cells)
        var d_max = ctx.enqueue_create_buffer[DType.float32](n_cells)
        var d_missing = ctx.enqueue_create_buffer[DType.int32](n_cells)
        var d_merges = ctx.enqueue_create_buffer[DType.int32](n_cells)
        var d_minkey = ctx.enqueue_create_buffer[DType.uint32](n_cells)
        var d_maxkey = ctx.enqueue_create_buffer[DType.uint32](n_cells)
        var d_nleft = ctx.enqueue_create_buffer[DType.int32](n_cells)
        var d_ntotal = ctx.enqueue_create_buffer[DType.int32](n_cells)
        var d_accl = ctx.enqueue_create_buffer[DType.int32](n_cells)
        var d_acct = ctx.enqueue_create_buffer[DType.int32](n_cells)
        var d_nblocks = ctx.enqueue_create_buffer[DType.int32](n_cells)
        var d_status = ctx.enqueue_create_buffer[DType.int32](n_cells)
        var d_thresh = ctx.enqueue_create_buffer[DType.float32](n_cells)
        var d_gnum = ctx.enqueue_create_buffer[DType.int64](n_cells)
        var d_gden = ctx.enqueue_create_buffer[DType.int64](n_cells)
        var c_q = ctx.enqueue_create_buffer[DType.float32](n_cells)
        var c_c = ctx.enqueue_create_buffer[DType.int32](n_cells)
        var c_m = ctx.enqueue_create_buffer[DType.float32](n_cells)
        var c_l = ctx.enqueue_create_buffer[DType.int32](n_cells)
        var c_nu = ctx.enqueue_create_buffer[DType.int64](n_cells)
        var c_de = ctx.enqueue_create_buffer[DType.int64](n_cells)
        var c_v = ctx.enqueue_create_buffer[DType.int32](n_cells)
        var r_q = ctx.enqueue_create_buffer[DType.float32](n_nodes)
        var r_c = ctx.enqueue_create_buffer[DType.int32](n_nodes)
        var r_m = ctx.enqueue_create_buffer[DType.float32](n_nodes)
        var r_l = ctx.enqueue_create_buffer[DType.int32](n_nodes)
        var r_nu = ctx.enqueue_create_buffer[DType.int64](n_nodes)
        var r_de = ctx.enqueue_create_buffer[DType.int64](n_nodes)
        var r_v = ctx.enqueue_create_buffer[DType.int32](n_nodes)
        var r_mg = ctx.enqueue_create_buffer[DType.int32](n_nodes)
        var r_nw = ctx.enqueue_create_buffer[DType.int32](n_nodes)
        var r_mx = ctx.enqueue_create_buffer[DType.int32](n_nodes)
        var d_nb = ctx.enqueue_create_buffer[DType.int32](n_nodes)
        var d_nc = ctx.enqueue_create_buffer[DType.int32](n_nodes)
        var d_colids = ctx.enqueue_create_buffer[DType.int32](n_cells)
        var d_samp_scratch = ctx.enqueue_create_buffer[DType.int32](
            sampler_scratch_len(n_nodes, Int(n_cols), Int(k))
        )
        var d_samp_report = ctx.enqueue_create_buffer[DType.int32](
            sampler_report_len(n_nodes)
        )
        var d_items = ctx.enqueue_create_buffer[DType.uint8](
            n_nodes * size_of[NodeWorkItem]()
        )
        var d_wl = ctx.enqueue_create_buffer[DType.uint8](
            plan.n_blocks_dimx * size_of[WorkloadInfo]()
        )
        var h_colids = ctx.enqueue_create_host_buffer[DType.int32](n_cells)
        var h_items = ctx.enqueue_create_host_buffer[DType.uint8](
            n_nodes * size_of[NodeWorkItem]()
        )
        var h_wl = ctx.enqueue_create_host_buffer[DType.uint8](
            plan.n_blocks_dimx * size_of[WorkloadInfo]()
        )
        var h_nb = ctx.enqueue_create_host_buffer[DType.int32](n_nodes)
        var h_nc = ctx.enqueue_create_host_buffer[DType.int32](n_nodes)
        ctx.synchronize()

        var items_ptr = h_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem]()
        for i in range(n_nodes):
            items_ptr[unsafe_offset=i] = work_items[i]
        var wl_ptr = h_wl.unsafe_ptr().unsafe_bitcast[WorkloadInfo]()
        for i in range(plan.n_blocks_dimx):
            wl_ptr[unsafe_offset=i] = plan.info[i]
        for i in range(n_nodes):
            h_nb.unsafe_ptr().unsafe_store(i, Int32(i * Int(k)))
            h_nc.unsafe_ptr().unsafe_store(i, Int32(Int(k)))
        ctx.enqueue_copy(dst_buf=d_items, src_ptr=h_items.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=d_wl, src_ptr=h_wl.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=d_nb, src_ptr=h_nb.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=d_nc, src_ptr=h_nc.unsafe_ptr())
        ctx.synchronize()

        # --- feature sampling, WHERE cuML DOES IT (deviation 201) --------
        # `d_report` is the DEVICE's own statement of which kernel ran, so it
        # is seeded with a value no kernel can produce.
        ctx.enqueue_memset(d_samp_report, SAMPLER_UNVISITED)
        _ = sample_features_for_device(
            ctx,
            d_colids,
            d_samp_scratch.unsafe_ptr(),
            d_samp_report.unsafe_ptr(),
            d_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem](),
            h_colids,
            work_items,
            tree_id,
            seed,
            Int(n_cols),
            Int(k),
        )

        ctx.enqueue_function[node_feature_range_init_kernel](
            d_min.unsafe_ptr(),
            d_max.unsafe_ptr(),
            d_minkey.unsafe_ptr(),
            d_maxkey.unsafe_ptr(),
            d_missing.unsafe_ptr(),
            d_merges.unsafe_ptr(),
            Int32(n_cells),
            grid_dim=ceildiv(n_cells, 64),
            block_dim=64,
        )
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
            k,
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
        ctx.enqueue_function[node_feature_score_init_kernel](
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
            grid_dim=ceildiv(n_cells, 64),
            block_dim=64,
        )
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
            n_rows,
            k,
            Int32(1),
            seed,
            tree_id,
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
            Int32(n_cells),
            k,
            Int32(1),
            seed,
            tree_id,
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
            grid_dim=ceildiv(n_cells, 64),
            block_dim=64,
        )
        var bpn = ceildiv(Int(k), TPB)
        if bpn < 1:
            bpn = 1
        ctx.enqueue_memset(r_mx, Int32(0))
        ctx.enqueue_function[split_reduce_init_kernel](
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
            grid_dim=ceildiv(n_nodes, 64),
            block_dim=64,
        )
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
            Int32(bpn),
            Int32(0),
            grid_dim=(bpn, n_nodes, 1),
            block_dim=(TPB, 1, 1),
        )

        var o_q = ctx.enqueue_create_host_buffer[DType.float32](n_nodes)
        var o_c = ctx.enqueue_create_host_buffer[DType.int32](n_nodes)
        var o_l = ctx.enqueue_create_host_buffer[DType.int32](n_nodes)
        var o_v = ctx.enqueue_create_host_buffer[DType.int32](n_nodes)
        var o_m = ctx.enqueue_create_host_buffer[DType.float32](n_nodes)
        ctx.enqueue_copy(dst_buf=o_q, src_buf=r_q)
        ctx.enqueue_copy(dst_buf=o_c, src_buf=r_c)
        ctx.enqueue_copy(dst_buf=o_l, src_buf=r_l)
        ctx.enqueue_copy(dst_buf=o_v, src_buf=r_v)
        ctx.enqueue_copy(dst_buf=o_m, src_buf=r_m)
        ctx.synchronize()

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

        var d_splits = ctx.enqueue_create_buffer[DType.uint8](
            n_nodes * size_of[Split]()
        )
        var d_iters = ctx.enqueue_create_buffer[DType.int32](n_nodes)
        var d_swaps = ctx.enqueue_create_buffer[DType.int32](n_nodes)
        var h_splits = ctx.enqueue_create_host_buffer[DType.uint8](
            n_nodes * size_of[Split]()
        )
        ctx.synchronize()
        var splits_ptr = h_splits.unsafe_ptr().unsafe_bitcast[Split]()
        for i in range(n_nodes):
            splits_ptr[unsafe_offset=i] = splits[i]
        ctx.enqueue_copy(dst_buf=d_splits, src_ptr=h_splits.unsafe_ptr())
        ctx.enqueue_memset(d_iters, PARTITION_UNVISITED)
        ctx.enqueue_function[node_split_kernel[TPB]](
            d_row_ids.unsafe_ptr(),
            d_iters.unsafe_ptr(),
            d_swaps.unsafe_ptr(),
            dataset.d_data.unsafe_ptr(),
            d_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem](),
            d_splits.unsafe_ptr().unsafe_bitcast[Split](),
            n_rows,
            params.min_impurity_decrease,
            params.min_samples_leaf,
            PART_SAB_NONE,
            grid_dim=(n_nodes, 1, 1),
            block_dim=(TPB, 1, 1),
        )
        ctx.synchronize()
        queue.push(work_items, splits)

    var tree = queue.get_tree()
    var n_nodes_final = tree.num_nodes()
    var d_nodes = ctx.enqueue_create_buffer[DType.uint8](
        n_nodes_final * size_of[SparseTreeNode[DType.float32]]()
    )
    var d_ranges = ctx.enqueue_create_buffer[DType.uint8](
        n_nodes_final * size_of[InstanceRange]()
    )
    var d_leaves = ctx.enqueue_create_buffer[DType.float32](n_nodes_final)
    var d_visit = ctx.enqueue_create_buffer[DType.int32](n_nodes_final)
    var h_nodes = ctx.enqueue_create_host_buffer[DType.uint8](
        n_nodes_final * size_of[SparseTreeNode[DType.float32]]()
    )
    var h_ranges = ctx.enqueue_create_host_buffer[DType.uint8](
        n_nodes_final * size_of[InstanceRange]()
    )
    var h_leaves = ctx.enqueue_create_host_buffer[DType.float32](
        n_nodes_final
    )
    ctx.synchronize()
    var nodes_ptr = h_nodes.unsafe_ptr().unsafe_bitcast[
        SparseTreeNode[DType.float32]
    ]()
    for i in range(n_nodes_final):
        nodes_ptr[unsafe_offset=i] = tree.sparsetree[i]
    var ranges_ptr = h_ranges.unsafe_ptr().unsafe_bitcast[InstanceRange]()
    for i in range(n_nodes_final):
        ranges_ptr[unsafe_offset=i] = queue.node_instances[i]
    ctx.enqueue_copy(dst_buf=d_nodes, src_ptr=h_nodes.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_ranges, src_ptr=h_ranges.unsafe_ptr())
    ctx.enqueue_memset(d_leaves, Float32(0.0))
    ctx.enqueue_memset(d_visit, Int32(0))
    # `inv_scale` puts the fixed-point mean back into the label's own units --
    # DEVIATION 179. This is the one place the scale is needed after
    # quantization.
    ctx.enqueue_function[leaf_kernel[TPB, LEAF_MAX_OUT_DEFAULT, False]](
        d_leaves.unsafe_ptr(),
        d_visit.unsafe_ptr(),
        d_nodes.unsafe_ptr().unsafe_bitcast[SparseTreeNode[DType.float32]](),
        d_ranges.unsafe_ptr().unsafe_bitcast[InstanceRange](),
        d_row_ids.unsafe_ptr(),
        dataset.d_labels.unsafe_ptr(),
        Int32(1),
        Float32(1.0 / scale),
        LEAF_SAB_NONE,
        grid_dim=(n_nodes_final, 1, 1),
        block_dim=(TPB, 1, 1),
    )
    ctx.enqueue_copy(dst_buf=h_leaves, src_buf=d_leaves)
    ctx.synchronize()
    tree.vector_leaf = List[Float32](
        length=n_nodes_final, fill=Float32(0.0)
    )
    for i in range(n_nodes_final):
        tree.vector_leaf[i] = h_leaves.unsafe_ptr()[unsafe_offset=i]
    return tree^


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

    `doSplit` and `SetLeafPredictions` are device work this lane has not
    written yet. Writing the loop now against stubs would produce a file that
    compiles, has a caller, and is wrong in a way nothing runs -- which is the
    exact failure `PORTING_RULES.md` rule 3 was written about. So the shape is
    recorded and the loop lands with the kernels.
    """
    return "see docstring"
