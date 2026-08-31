# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The node queue: structure invariants, and the batch width proved inert.

`NodeQueue` (`builder.cuh:44-135`) is the host control plane. Nothing about it
touches data, so it can be driven to completion by a SPLIT ORACLE: a pure
function from a node's (id, depth, range) to a split. That is the whole point
of the design here -- because the oracle depends only on the node and not on
which batch the node arrived in, the tree it produces must be identical for
every `max_batch_size`, and any dependence of the tree on the batch width is a
defect this check can see.

What is verified, per node rather than in aggregate:

1. `sparsetree` and `node_instances` stay the SAME LENGTH -- their
   `SetLeafPredictions` asserts this (`builder.cuh:562-563`) and zips them.
2. Children are an ADJACENT PAIR: `RightChildId() == LeftChildId() + 1`, and
   the two slots really are the two children.
3. The children's ranges TILE the parent's: `left.begin == parent.begin`,
   `right.begin == parent.begin + n_left`, and the counts sum. A tree that
   loses rows still looks like a tree.
4. `depth_counter` and `leaf_counter` equal what an INDEPENDENT recursive walk
   finds -- not what the incremental counters accumulated.
5. `is_expandable` is exercised on BOTH sides of each of its three tests
   (rule 8: a switch needs a named case per side).
6. An invalid split leaves the node a leaf AND does not consume leaf budget,
   and the `max_leaves` cutoff admits exactly the budgeted number of splits
   and not one more. NOT checked, because it is not checkable and saying so is
   the honest form: their `break` at `builder.cuh:106` versus a `continue` is
   a distinction without a difference, since `leaf_counter` only increases
   inside the loop and the budget test therefore stays true once true. A
   sabotage swapping break for continue leaves this check green, which is
   correct -- the two are the same function.
7. The tree is bit-identical across five batch widths.
"""

from std.testing import assert_equal, assert_true

from extratrees.ported.decisiontree.decisiontree import DecisionTreeParams
from extratrees.ported.decisiontree.flatnode import (
    SparseTreeNode,
    TreeMetaDataNode,
)
from extratrees.ported.decisiontree.batched_levelalgo.builder import (
    NodeQueue,
    max_nodes,
)
from extratrees.ported.decisiontree.batched_levelalgo.split import Split
from extratrees.ported.decisiontree.batched_levelalgo.kernels.builder_kernels import (
    InstanceRange,
    NodeWorkItem,
    split_not_valid,
)


def mix32(x: UInt32) -> UInt32:
    var h = x
    h ^= h >> 16
    h *= 0x85EBCA6B
    h ^= h >> 13
    h *= 0xC2B2AE35
    h ^= h >> 16
    return h


def oracle_split(item: NodeWorkItem) -> Split:
    """A split for a node, depending ONLY on the node.

    Hashed metric, colid and threshold so no two nodes agree, and an
    ASYMMETRIC left count so the tree is lopsided -- a 50/50 split makes
    `begin + n_left` and `begin + count/2` the same number and would hide an
    off-by-one in the right child's offset.

    A node with fewer than two rows gets a metric of 0.0, which
    `split_not_valid` rejects at the default `min_impurity_decrease` of 0.0.
    That is how this oracle produces leaves.
    """
    var h = mix32(UInt32(Int(item.idx)) ^ 0x5EED)
    var count = Int(item.instances.count)
    if count < 2:
        return Split(0.0, 0, 0.0, 0)
    var n_left = 1 + Int(h % UInt32(count - 1))
    return Split(
        Float32(Int(mix32(h) % 8192)) / 64.0,
        Int32(Int(mix32(h ^ 0x1234) % 37)),
        Float32(Int(h % 1000) + 1) / 10.0,
        Int32(n_left),
    )


def run_to_completion(
    params: DecisionTreeParams, n_rows: Int32
) raises -> TreeMetaDataNode[DType.float32]:
    """`builder.cuh:344-352`'s loop, with `doSplit` replaced by the oracle."""
    var queue = NodeQueue[DType.float32](params, n_rows, 1)
    while queue.has_work():
        var batch = queue.pop()
        var splits = List[Split]()
        for i in range(len(batch)):
            splits.append(oracle_split(batch[i]))
        queue.push(batch, splits)
    return queue.get_tree()


def walk_depth_and_leaves(
    tree: TreeMetaDataNode[DType.float32], node: Int, depth: Int32
) -> Tuple[Int32, Int32]:
    """An INDEPENDENT recursive walk: the deepest depth, and the leaf count.

    Recursive where the queue's counters are incremental, so a counter that
    drifted has nothing to hide behind.
    """
    if tree.sparsetree[node].IsLeaf():
        return (depth, Int32(1))
    var left = Int(tree.sparsetree[node].LeftChildId())
    var right = Int(tree.sparsetree[node].RightChildId())
    var a = walk_depth_and_leaves(tree, left, depth + 1)
    var b = walk_depth_and_leaves(tree, right, depth + 1)
    var deepest = a[0] if a[0] > b[0] else b[0]
    return (deepest, a[1] + b[1])


def check_structure(
    tree: TreeMetaDataNode[DType.float32],
    instances: List[InstanceRange],
    n_rows: Int32,
    label: String,
) raises -> Int:
    var cells = 0
    assert_equal(
        len(instances),
        tree.num_nodes(),
        label + ": sparsetree and node_instances must stay the same length",
    )
    cells += 1

    var total_leaf_rows = 0
    for i in range(tree.num_nodes()):
        var node = tree.sparsetree[i]
        assert_equal(
            node.InstanceCount(),
            instances[i].count,
            label + ": node " + String(i) + " count disagrees with its range",
        )
        cells += 1
        if node.IsLeaf():
            total_leaf_rows += Int(instances[i].count)
            cells += 1
            continue

        var left = Int(node.LeftChildId())
        var right = Int(node.RightChildId())
        assert_equal(
            right, left + 1, label + ": children must be an adjacent pair"
        )
        assert_true(
            left > i and right < tree.num_nodes(),
            label + ": child indices out of range",
        )
        # The children TILE the parent, exactly.
        assert_equal(
            instances[left].begin,
            instances[i].begin,
            label + ": left child must start where the parent starts",
        )
        assert_equal(
            instances[right].begin,
            instances[i].begin + instances[left].count,
            label + ": right child must start after the left child ends",
        )
        assert_equal(
            instances[left].count + instances[right].count,
            instances[i].count,
            label + ": children must account for every parent row",
        )
        assert_true(
            instances[left].count > 0 and instances[right].count > 0,
            label + ": a split that empties a child is not a split",
        )
        cells += 6

    # Every row lands in exactly one leaf.
    assert_equal(
        total_leaf_rows,
        Int(n_rows),
        label + ": the leaves must partition the input rows",
    )
    cells += 1

    # The incremental counters against the independent walk.
    var walked = walk_depth_and_leaves(tree, 0, 0)
    assert_equal(
        tree.depth_counter, walked[0], label + ": depth_counter drifted"
    )
    assert_equal(
        tree.leaf_counter, walked[1], label + ": leaf_counter drifted"
    )
    cells += 2
    return cells


def main() raises:
    var cells = 0

    # --- max_nodes, both sides of their depth-13 cliff --------------------
    assert_equal(max_nodes(0), 1)
    assert_equal(max_nodes(1), 3)
    assert_equal(max_nodes(12), 8191)
    assert_equal(max_nodes(13), 8191)
    assert_equal(max_nodes(30), 8191)
    cells += 5

    # --- a full build, and its structure ----------------------------------
    var p = DecisionTreeParams()
    p.max_depth = 6
    var tree = run_to_completion(p, 1000)
    var q = NodeQueue[DType.float32](p, 1000, 1)
    # rebuild alongside so the instance ranges are available for checking
    while q.has_work():
        var batch = q.pop()
        var splits = List[Split]()
        for i in range(len(batch)):
            splits.append(oracle_split(batch[i]))
        q.push(batch, splits)
    cells += check_structure(q.tree, q.node_instances, 1000, "depth 6")
    assert_true(
        q.tree.num_nodes() > 20,
        "a depth-6 build that produced almost no nodes is not exercising much",
    )
    print("depth-6 build: ", q.tree.num_nodes(), "nodes,", q.tree.leaf_counter, "leaves, depth", q.tree.depth_counter)

    # --- THE BATCH WIDTH IS INERT ----------------------------------------
    # The oracle depends only on the node, so the tree must not depend on how
    # the frontier was chopped up. This is the property that makes
    # `max_batch_size` a scheduling parameter rather than an algorithmic one.
    var widths = [Int32(1), Int32(2), Int32(3), Int32(7), Int32(4096)]
    var reference = run_to_completion(p, 1000)
    for w in widths:
        var pw = DecisionTreeParams()
        pw.max_depth = 6
        pw.max_batch_size = w
        var t = run_to_completion(pw, 1000)
        assert_equal(
            t.num_nodes(),
            reference.num_nodes(),
            "batch width changed the node count",
        )
        for i in range(t.num_nodes()):
            assert_true(
                t.sparsetree[i] == reference.sparsetree[i],
                "batch width "
                + String(w)
                + " changed node "
                + String(i),
            )
            cells += 1
    print("batch-width independence:", len(widths), "widths x", reference.num_nodes(), "nodes")

    # --- max_depth == 0: one node, no work, and it must not crash ---------
    var p0 = DecisionTreeParams()
    p0.max_depth = 0
    var q0 = NodeQueue[DType.float32](p0, 500, 1)
    assert_true(not q0.has_work(), "max_depth 0 must produce no work at all")
    assert_equal(q0.tree.num_nodes(), 1)
    assert_true(q0.tree.sparsetree[0].IsLeaf(), "the lone root must be a leaf")
    cells += 3

    # --- is_expandable, BOTH sides of all three tests ---------------------
    var pe = DecisionTreeParams()
    pe.max_depth = 3
    pe.min_samples_split = 10
    pe.max_leaves = 5
    var qe = NodeQueue[DType.float32](pe, 100, 1)
    var big = SparseTreeNode[DType.float32].CreateLeafNode(100)
    var small = SparseTreeNode[DType.float32].CreateLeafNode(9)
    assert_true(qe.is_expandable(big, 2), "depth 2 < max_depth 3: expandable")
    assert_true(
        not qe.is_expandable(big, 3), "depth == max_depth: NOT expandable"
    )
    assert_true(
        not qe.is_expandable(big, 4), "depth > max_depth: NOT expandable"
    )
    assert_true(
        qe.is_expandable(
            SparseTreeNode[DType.float32].CreateLeafNode(10), 0
        ),
        "count == min_samples_split: expandable, their test is <",
    )
    assert_true(
        not qe.is_expandable(small, 0),
        "count < min_samples_split: NOT expandable",
    )
    cells += 5
    # the max_leaves arm: raise the counter past the budget and watch it flip
    qe.tree.leaf_counter = 4
    assert_true(
        qe.is_expandable(big, 0), "leaf_counter below max_leaves: expandable"
    )
    qe.tree.leaf_counter = 5
    assert_true(
        not qe.is_expandable(big, 0), "leaf_counter == max_leaves: NOT"
    )
    var pu = DecisionTreeParams()
    pu.max_depth = 3
    var qu = NodeQueue[DType.float32](pu, 100, 1)
    qu.tree.leaf_counter = 100000
    assert_true(
        qu.is_expandable(big, 0),
        "max_leaves == -1 means unlimited, whatever the counter says",
    )
    cells += 3

    # --- the gain gate's BOUNDARY, both sides of it (DEVIATION 216) -------
    # A split whose gain exactly EQUALS min_impurity_decrease SPLITS: that is
    # sklearn's boundary, and at the default 0 it is what keeps zero-gain
    # refining splits alive on integer targets (this case pinned cuML's `<=`
    # -- "a rejected split must create no nodes" for gain == threshold --
    # until year's test MSE paid for it; see the 216 entry).
    var pi = DecisionTreeParams()
    pi.max_depth = 4
    var qi = NodeQueue[DType.float32](pi, 100, 1)
    var batch0 = qi.pop()
    var before_leaves = qi.tree.leaf_counter
    var before_nodes = qi.tree.num_nodes()
    var zero_gain = List[Split]()
    zero_gain.append(Split(1.0, 0, 0.0, 50))  # gain == min_impurity_decrease
    qi.push(batch0, zero_gain)
    assert_equal(
        qi.tree.num_nodes(),
        before_nodes + 2,
        "gain == min_impurity_decrease SPLITS (sklearn's boundary, 216)",
    )
    assert_equal(
        qi.tree.leaf_counter,
        before_leaves + 1,
        "the accepted boundary split consumes exactly one leaf of budget",
    )
    # And a split strictly BELOW the threshold is still rejected -- the
    # sentinel an invalid candidate carries is MIN_FINITE, which is the
    # rejection the fit path actually exercises.
    var qr2 = NodeQueue[DType.float32](pi, 100, 1)
    var batch1 = qr2.pop()
    var rb_leaves = qr2.tree.leaf_counter
    var rb_nodes = qr2.tree.num_nodes()
    var dud = List[Split]()
    dud.append(Split(1.0, 0, Float32.MIN_FINITE, 50))
    qr2.push(batch1, dud)
    assert_equal(
        qr2.tree.num_nodes(),
        rb_nodes,
        "a below-threshold split must create no nodes",
    )
    assert_equal(
        qr2.tree.leaf_counter,
        rb_leaves,
        "a rejected split must not consume leaf budget",
    )
    # `qi` took the ACCEPTED boundary split above (216), so its root is a
    # split node with two child work items; `qr2` took the rejected one and
    # must be untouched.
    assert_true(
        not qi.tree.sparsetree[0].IsLeaf(),
        "the boundary-accepted root must be a SPLIT node (216)",
    )
    assert_true(
        qr2.tree.sparsetree[0].IsLeaf(),
        "the rejected split's node must still be a leaf",
    )
    assert_true(not qr2.has_work(), "and no new work may be queued for it")
    cells += 5

    # --- the max_leaves cutoff admits exactly the budget, and no more ------
    # Two work items in one batch, both with valid splits, with room for
    # exactly one more split. The first takes it and the second is refused.
    # This does NOT distinguish their `break` from a `continue` -- nothing
    # can, see the header -- but it does pin the boundary: `>=` against `>`
    # in that test is one extra leaf, and a sabotage proves this case sees it.
    var pb = DecisionTreeParams()
    pb.max_depth = 8
    pb.max_leaves = 3
    var qb = NodeQueue[DType.float32](pb, 100, 1)
    var b1 = qb.pop()
    var s1 = List[Split]()
    s1.append(Split(1.0, 0, 5.0, 40))
    qb.push(b1, s1)  # leaf_counter 1 -> 2, two children queued
    assert_equal(qb.tree.num_nodes(), 3)
    var b2 = qb.pop()
    assert_equal(len(b2), 2, "both children should be queued as one batch")
    var s2 = List[Split]()
    s2.append(Split(1.0, 1, 5.0, 20))
    s2.append(Split(1.0, 2, 5.0, 30))
    qb.push(b2, s2)
    # budget: leaf_counter is 2, max_leaves 3, so the FIRST item splits
    # (counter -> 3) and the second hits the break.
    assert_equal(
        qb.tree.leaf_counter, 3, "exactly one more split fits in the budget"
    )
    assert_equal(
        qb.tree.num_nodes(), 5, "and it created exactly one pair of children"
    )
    cells += 4

    # --- push refuses a mismatched batch ----------------------------------
    var qm = NodeQueue[DType.float32](p, 100, 1)
    var bm = qm.pop()
    var refused = False
    try:
        qm.push(bm, List[Split]())
    except:
        refused = True
    assert_true(refused, "push must refuse a splits list of the wrong length")
    cells += 1

    print("builder: ", cells, "cells")
    print("builder_check: PASS")
