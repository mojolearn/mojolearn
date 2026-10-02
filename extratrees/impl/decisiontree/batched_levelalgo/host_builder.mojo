# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The ExtraTrees HOST ORACLE builders: sklearn's splitter on the host.

Moved out of `batched_levelalgo/builder.mojo` (cpu-gpu-cleanup t-forest)
because that module is the GPU trainer's and every GPU binding reaches it;
these builders call `extratrees/checks/host_splitter.mojo`'s host search,
which no GPU install may import. Nothing here changed: the functions are
the oracles the device kernels are checked against (`tree_check`,
`device_tree_check`, `bestfirst_check`, `rescue_check`, `forest_check`),
reached only from the checks and from
`extratrees/impl/randomforest/host_forest.mojo`'s reference fits.
"""

from extratrees.checks.host_splitter import (
    HostSplitResult,
    node_split_random_gini,
    node_split_random_mse,
)
from extratrees.checks.rescue import rescue_key, rescue_pick
from extratrees.impl.decisiontree.decisiontree import (
    CRITERION_END,
    CRITERION_GINI,
    DecisionTreeParams,
    validity_check,
)
from extratrees.impl.decisiontree.flatnode import TreeMetaDataNode
from extratrees.impl.decisiontree.batched_levelalgo.dataset import Dataset
from extratrees.impl.decisiontree.batched_levelalgo.objectives import (
    GiniObjectiveFunction,
    MSEObjectiveFunction,
)
from extratrees.impl.decisiontree.batched_levelalgo.split import Split
from extratrees.impl.decisiontree.batched_levelalgo.kernels.builder_kernels import (
    NodeWorkItem,
    sample_features,
    split_not_valid,
)
from extratrees.impl.decisiontree.batched_levelalgo.kernels.builder_kernels_impl import (
    partition_samples,
)
from extratrees.impl.decisiontree.batched_levelalgo.builder import (
    BESTFIRST_SAB_NONE,
    NodeQueue,
    n_sampled_cols_for,
    rescue_columns,
    set_leaf_predictions_classification,
    set_leaf_predictions_regression,
)


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


def train_classification(
    dataset: Dataset,
    params: DecisionTreeParams,
    tree_id: Int32,
    seed: UInt64,
    n_classes: Int32,
    rescue: Bool = True,
) raises -> TreeMetaDataNode[DType.float32]:
    """One ExtraTree, end to end. `Builder::train`, `builder.cuh:344-359`.

    The reference loop, which this matches exactly::

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

    The reference loop, which this matches except for the two structural changes
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
    ORDER, because this implementation partitions where they index. Theirs never
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
