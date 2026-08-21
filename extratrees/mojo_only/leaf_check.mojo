"""Leaf predictions: per leaf, per class, against an independent tally.

`SetLeafPredictions` (`builder.cuh:556-599`) plus `leafKernel`
(`kernels/builder_kernels_impl.cuh:391-417`) is the pass that turns a built
tree into a model. It has three ways to be quietly wrong, and this file has a
case for each:

1. it reads the leaf's rows THROUGH `row_ids` over the node's own
   `InstanceRange`, so an implementation that walked `begin..begin+count` as
   raw row numbers would produce plausible probabilities from the wrong rows.
   The fixture therefore uses a SHUFFLED `row_ids`, which makes slot and row
   different numbers;
2. it must leave every INTERNAL node's slot zero (`:403`, the early return),
   and a pass that filled them would still predict correctly at every leaf;
3. the `vector_leaf` stride is `num_outputs` per node for ALL nodes, leaves and
   internal alike (`decisiontree.cuh:218` indexes by raw node id), so an
   implementation that packed only the leaves would be off by a growing
   offset -- and would still be right for node 0.

Labels are hashed so that no two leaves share a class distribution, per rule 8:
a fixture where every leaf has the same answer verifies the total and nothing
about placement.
"""

from std.testing import assert_equal, assert_true

from extratrees.ported.decisiontree.decisiontree import DecisionTreeParams
from extratrees.ported.decisiontree.flatnode import SparseTreeNode
from extratrees.ported.decisiontree.batched_levelalgo.builder import (
    NodeQueue,
    set_leaf_predictions_classification,
    set_leaf_predictions_regression,
)
from extratrees.ported.decisiontree.batched_levelalgo.dataset import Dataset
from extratrees.ported.decisiontree.batched_levelalgo.split import Split
from extratrees.ported.decisiontree.batched_levelalgo.kernels.builder_kernels import (
    InstanceRange,
    NodeWorkItem,
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
    """The same node-only split oracle `builder_check` uses: enough to build a
    lopsided tree whose ranges tile correctly, which is all this pass needs."""
    var h = mix32(UInt32(Int(item.idx)) ^ 0x5EED)
    var count = Int(item.instances.count)
    if count < 2:
        return Split(0.0, 0, 0.0, 0)
    var n_left = 1 + Int(h % UInt32(count - 1))
    return Split(1.0, Int32(Int(mix32(h) % 37)), 5.0, Int32(n_left))


def main() raises:
    var n_rows = 600
    var n_classes = 4

    # Hashed labels, scattered: no two leaves get the same distribution.
    var labels = List[Float32]()
    for r in range(n_rows):
        labels.append(Float32(Int(mix32(UInt32(r) ^ 0xC0DE) % UInt32(n_classes))))

    # A SHUFFLED row_ids, so a slot index and a row id are different numbers.
    var row_ids = List[Int32]()
    for r in range(n_rows):
        row_ids.append(Int32(r))
    for i in range(n_rows - 1, 0, -1):
        var j = Int(mix32(UInt32(i) ^ 0xBEE5) % UInt32(i + 1))
        var t = row_ids[i]
        row_ids[i] = row_ids[j]
        row_ids[j] = t

    var features = List[Float32](length=n_rows, fill=Float32(0.0))
    var dataset = Dataset(
        rebind[MutPointer[Float32, MutUntrackedOrigin]](features.unsafe_ptr()),
        rebind[MutPointer[Float32, MutUntrackedOrigin]](labels.unsafe_ptr()),
        Int32(n_rows),
        1,
        Int32(n_rows),
        1,
        rebind[MutPointer[Int32, MutUntrackedOrigin]](row_ids.unsafe_ptr()),
        Int32(n_classes),
    )

    # Build a real tree so the ranges tile the row space the way the builder
    # leaves them.
    var p = DecisionTreeParams()
    p.max_depth = 5
    var q = NodeQueue[DType.float32](p, Int32(n_rows), Int32(n_classes))
    while q.has_work():
        var batch = q.pop()
        var splits = List[Split]()
        for i in range(len(batch)):
            splits.append(oracle_split(batch[i]))
        q.push(batch, splits)
    var n_nodes = q.tree.num_nodes()
    print("tree:", n_nodes, "nodes,", q.tree.leaf_counter, "leaves")
    assert_true(n_nodes > 10, "a tree this small exercises too little")

    # ---------------- classification ------------------------------------
    set_leaf_predictions_classification(dataset, q.tree, q.node_instances)

    var cells = 0
    var leaves_seen = 0
    var rows_accounted = 0
    for node_id in range(n_nodes):
        var base = node_id * n_classes
        if not q.tree.sparsetree[node_id].IsLeaf():
            # every internal slot must still be zero
            for c in range(n_classes):
                assert_equal(
                    q.tree.vector_leaf[base + c],
                    Float32(0.0),
                    "internal node " + String(node_id) + " must keep a zero"
                    " leaf slot -- builder_kernels_impl.cuh:403 returns early",
                )
                cells += 1
            continue

        leaves_seen += 1
        var rng = q.node_instances[node_id]
        # INDEPENDENT tally: a plain loop, written differently, over the same
        # range, through row_ids.
        var want = List[Int](length=n_classes, fill=0)
        for i in range(Int(rng.begin), Int(rng.begin) + Int(rng.count)):
            want[Int(labels[Int(row_ids[i])])] += 1
        var total = 0
        for c in range(n_classes):
            total += want[c]
        assert_equal(
            total, Int(rng.count), "the tally must see every row in the range"
        )
        rows_accounted += total
        cells += 1

        var prob_sum = Float32(0.0)
        for c in range(n_classes):
            var got = q.tree.vector_leaf[base + c]
            var expect = Float32(want[c]) / Float32(total)
            assert_equal(
                got.to_bits(),
                expect.to_bits(),
                "leaf "
                + String(node_id)
                + " class "
                + String(c)
                + " probability differs",
            )
            prob_sum += got
            cells += 1
        # Their SetLeafVector divides counts by their own total, so the
        # probabilities sum to 1 by construction -- but only if the counts and
        # the total came from the SAME rows.
        assert_true(
            prob_sum > 0.9999 and prob_sum < 1.0001,
            "leaf " + String(node_id) + " probabilities must sum to 1",
        )
        cells += 1

    assert_equal(
        rows_accounted,
        n_rows,
        "the leaves' ranges must together account for every row",
    )
    # The fixture must actually distinguish leaves: if every leaf had the same
    # distribution, this check would verify the total and nothing else.
    var distinct = 0
    for a in range(n_nodes):
        if not q.tree.sparsetree[a].IsLeaf():
            continue
        var unique = True
        for b in range(a):
            if not q.tree.sparsetree[b].IsLeaf():
                continue
            var same = True
            for c in range(n_classes):
                if (
                    q.tree.vector_leaf[a * n_classes + c]
                    != q.tree.vector_leaf[b * n_classes + c]
                ):
                    same = False
            if same:
                unique = False
        if unique:
            distinct += 1
    print(
        "classification:",
        leaves_seen,
        "leaves,",
        distinct,
        "with a distribution no earlier leaf shares",
    )
    assert_true(
        distinct * 2 > leaves_seen,
        "most leaves must have DISTINCT distributions, or this fixture cannot"
        " tell a misplaced leaf from a correct one",
    )
    cells += 2

    # ---------------- regression -----------------------------------------
    var reg_labels = List[Float32]()
    for r in range(n_rows):
        reg_labels.append(
            Float32(Int(mix32(UInt32(r) ^ 0xFEED) % 8192)) / 64.0 - 64.0
        )
    var reg = Dataset(
        rebind[MutPointer[Float32, MutUntrackedOrigin]](features.unsafe_ptr()),
        rebind[MutPointer[Float32, MutUntrackedOrigin]](
            reg_labels.unsafe_ptr()
        ),
        Int32(n_rows),
        1,
        Int32(n_rows),
        1,
        rebind[MutPointer[Int32, MutUntrackedOrigin]](row_ids.unsafe_ptr()),
        1,
    )
    var qr = NodeQueue[DType.float32](p, Int32(n_rows), 1)
    while qr.has_work():
        var batch = qr.pop()
        var splits = List[Split]()
        for i in range(len(batch)):
            splits.append(oracle_split(batch[i]))
        qr.push(batch, splits)
    set_leaf_predictions_regression(reg, qr.tree, qr.node_instances)

    var reg_leaves = 0
    for node_id in range(qr.tree.num_nodes()):
        if not qr.tree.sparsetree[node_id].IsLeaf():
            assert_equal(
                qr.tree.vector_leaf[node_id],
                Float32(0.0),
                "internal regression slots must stay zero",
            )
            cells += 1
            continue
        reg_leaves += 1
        var rng = qr.node_instances[node_id]
        # Independent tally in float64, matching the oracle's accumulator.
        var acc = Float64(0.0)
        for i in range(Int(rng.begin), Int(rng.begin) + Int(rng.count)):
            acc += Float64(reg_labels[Int(row_ids[i])])
        var expect = Float32(acc / Float64(Int(rng.count)))
        assert_equal(
            qr.tree.vector_leaf[node_id].to_bits(),
            expect.to_bits(),
            "regression leaf " + String(node_id) + " is not its rows' mean",
        )
        cells += 1
    print("regression:", reg_leaves, "leaves, each the mean of its own rows")

    # A mismatched range list must be refused, not silently truncated.
    var refused = False
    try:
        set_leaf_predictions_classification(
            dataset, q.tree, List[InstanceRange]()
        )
    except:
        refused = True
    assert_true(
        refused,
        "a range list of the wrong length must be refused -- builder.cuh:562"
        " asserts it",
    )
    cells += 1

    # Keep the backing buffers alive: `Dataset`'s pointers are
    # `MutUntrackedOrigin` and Mojo frees a `List` after its last syntactic
    # use, which without this is the constructor call above.
    _ = labels.unsafe_ptr()
    _ = reg_labels.unsafe_ptr()
    _ = features.unsafe_ptr()
    _ = row_ids.unsafe_ptr()

    print("leaf: ", cells, "cells")
    print("leaf_check: PASS")
