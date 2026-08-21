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

from extratrees.ported.decisiontree.decisiontree import DecisionTreeParams
from extratrees.ported.decisiontree.flatnode import (
    SparseTreeNode,
    TreeMetaDataNode,
)
from extratrees.ported.decisiontree.batched_levelalgo.split import Split
from extratrees.ported.decisiontree.batched_levelalgo.kernels.builder_kernels import (
    InstanceRange,
    NodeWorkItem,
    split_not_valid,
)


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
