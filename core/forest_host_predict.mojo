# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Sequential forest prediction on the host, for a box with no GPU.

HOST ONLY. Nothing here imports `max.gpu`, `std.gpu` or a `DeviceContext`,
and the GPU bindings do not import this file, so building it changes no byte
of theirs. It exists because the two forest loops the GPU bindings run for
`inference_engine="sequential"` live in modules that import `max.gpu.host` at
module level (`ensemble/randomforest.mojo:8`,
`extratrees/impl/randomforest/randomforest.mojo:100`), while the per-tree walks
they call are GPU-free. This module imports those walks as they are and
restates the loops around them.

WHAT IS REUSED, NOT RESTATED. `DecisionTree.predict` and its `predict_one`
(`ensemble/decisiontree/decisiontree.mojo:529-654`, with the DEVIATION 1942
feature flush) decide every bit of a RandomForest prediction, and
`predict_one_accumulate` (`extratrees/impl/decisiontree/flatnode.mojo:467-482`)
every bit of an ExtraTrees one. Both come from the files the GPU bindings
compile.

WHAT IS RESTATED, AND WHERE THE ORIGINAL IS. `rf_host_predict` MIRRORS
`RandomForest.predict_proba`, `ensemble/randomforest.mojo:1140-1161` (and the
regression branch of `RandomForest.predict`, `:1033-1074`, which is the same
loop read at output 0). `et_host_predict` MIRRORS `forest_vote`,
`extratrees/impl/randomforest/randomforest.mojo:516-547`. Each is three
statements. Zero a float32 vector, add every tree's leaf vector into it in
increasing tree order, divide each element by `Float32(n_trees)`. The tree
rebuilds MIRROR `_rebuild_trees` (`bindings/_mojolearn_rf.mojo:635-668`) and
`et_predict_binding`'s loop (`bindings/_mojolearn_trees.mojo:500-523`) and add
the bounds checks a file read off disk needs.

The restatement is a prediction until measured. tools/forest_host_gate.py is
the measurement, and the brief in docs/lanes/ records what it has shown.
"""
from std.sys.compile import is_defined

from ensemble.decisiontree.decisiontree import DecisionTree
from ensemble.decisiontree.decisiontree import TreeMetaDataNode as RfTree
from ensemble.flatnode import SparseTreeNode as RfNode
from extratrees.impl.decisiontree.flatnode import SparseTreeNode as EtNode
from extratrees.impl.decisiontree.flatnode import TreeMetaDataNode as EtTree
from extratrees.impl.decisiontree.flatnode import predict_one_accumulate


#: The gate's negative control. A build with this define divides the vote by
#: `n_trees + 1`, so every prediction of every fixture is wrong by
#: construction, and the gate must say so. Read back by
#: `forest_host_sabotage` and refused outside the gate by `_forest_host.py`.
comptime FOREST_HOST_SABOTAGE = is_defined["MOJOLEARN_FOREST_HOST_SABOTAGE"]()


def _divisor(n_trees: Int) -> Float32:
    """`Scalar[DType.float32](n_trees)` in `RandomForest.predict_proba`,
    `Float32(Int(forest.n_trees))` in `forest_vote`; the same value."""
    comptime if FOREST_HOST_SABOTAGE:
        return Float32(n_trees + 1)
    return Float32(n_trees)


def _tree_span(
    offsets_p: MutPointer[Int32, MutUntrackedOrigin],
    t: Int,
    n_nodes: Int,
) raises -> Int:
    """The node count of tree `t`, refusing an offsets member that is not a
    prefix scan inside `[0, n_nodes]` or an empty tree (which
    `DecisionTree.predict` would refuse one call later, `decisiontree.cuh:350`)."""
    var lo = Int(offsets_p[t])
    var hi = Int(offsets_p[t + 1])
    if lo < 0 or hi <= lo or hi > n_nodes:
        raise Error(
            "forest host: tree_offsets are not a prefix scan inside the node"
            " arrays (tree " + String(t) + " spans [" + String(lo) + ", "
            + String(hi) + ") of " + String(n_nodes) + " nodes)"
        )
    return lo


def _check_node(colid: Int32, left: Int32, local: Int, count: Int, n_cols: Int) raises:
    """A split node must test a real column and point at children that lie
    after it inside its own tree (every builder here allocates children
    after their parent), so the walk terminates and never reads past the
    tree. A leaf's `colid` is meaningless and is not checked."""
    if left == -1:
        return
    var l = Int(left)
    if l <= local or l + 1 >= count:
        raise Error(
            "forest host: node " + String(local) + " has left child "
            + String(l) + " in a tree of " + String(count) + " nodes"
        )
    var c = Int(colid)
    if c < 0 or c >= n_cols:
        raise Error(
            "forest host: node " + String(local) + " tests column "
            + String(c) + " of " + String(n_cols)
        )


def rf_host_trees(
    offsets_p: MutPointer[Int32, MutUntrackedOrigin],
    colid_p: MutPointer[Int32, MutUntrackedOrigin],
    quesval_p: MutPointer[Float32, MutUntrackedOrigin],
    left_p: MutPointer[Int32, MutUntrackedOrigin],
    leaves_p: MutPointer[Float32, MutUntrackedOrigin],
    n_trees: Int,
    n_nodes: Int,
    n_cols: Int,
    num_outputs: Int,
) raises -> List[RfTree[DType.float32]]:
    """MIRRORS `_rebuild_trees`, `bindings/_mojolearn_rf.mojo:635-668`.
    `instance_count`, `best_metric_val` and `train_time` are zero; the walk
    reads none of them."""
    var trees = List[RfTree[DType.float32]](capacity=n_trees)
    for t in range(n_trees):
        var lo = _tree_span(offsets_p, t, n_nodes)
        var count = Int(offsets_p[t + 1]) - lo
        var nodes = List[RfNode[DType.float32]](capacity=count)
        var vleaf = List[Float32](capacity=count * num_outputs)
        for i in range(lo, lo + count):
            _check_node(colid_p[i], left_p[i], i - lo, count, n_cols)
            nodes.append(
                RfNode[DType.float32](
                    colid_p[i], quesval_p[i], 0.0, Int64(left_p[i]), 0
                )
            )
            for k in range(num_outputs):
                vleaf.append(leaves_p[i * num_outputs + k])
        trees.append(
            RfTree[DType.float32](
                Int32(t), 0, 0, 0.0, vleaf^, nodes^, Int32(num_outputs)
            )
        )
    return trees^


def rf_host_predict(
    trees: List[RfTree[DType.float32]],
    rows: List[Float32],
    n_rows: Int,
    n_cols: Int,
    n_trees: Int,
    num_outputs: Int,
    mut out: List[Float32],
) raises:
    """MIRRORS `RandomForest.predict_proba`, `ensemble/randomforest.mojo:1140-1161`.

    `rows` is ROW-major, `n_rows * n_cols`. `out` receives
    `n_rows * num_outputs` values, the vote divided by `n_trees` and nothing
    else (the classifier's argmax is the Python layer's, as it is for the GPU
    binding; the regressor reads output 0 of a one-output vote, which is
    what `RandomForest.predict`'s REGRESSION branch does at `:1071-1074`).
    """
    if n_rows <= 0 or n_cols <= 0:
        raise Error("forest host: n_rows and n_cols must be positive")
    if n_trees <= 0 or len(trees) != n_trees:
        raise Error("forest host: the tree list does not hold n_trees trees")
    if len(rows) < n_rows * n_cols:
        raise Error("forest host: rows holds fewer than n_rows * n_cols values")
    if len(out) < n_rows * num_outputs:
        raise Error("forest host: out holds fewer than n_rows * num_outputs values")
    var divisor = _divisor(n_trees)
    for row_id in range(n_rows):
        # `randomforest.cuh:403`, zero-initialized once per row.
        var row_prediction = List[Float32]()
        for _ in range(num_outputs):
            row_prediction.append(0)
        # `:404-412`, one row at a time, every tree adds into it.
        for i in range(n_trees):
            DecisionTree.predict(
                trees[i],
                rows,
                1,
                n_cols,
                row_prediction,
                num_outputs,
                rows_offset=row_id * n_cols,
                preds_offset=0,
            )
        # `:414-416`, divide by n_trees, and stop.
        for k in range(num_outputs):
            out[row_id * num_outputs + k] = row_prediction[k] / divisor


def et_host_trees(
    offsets_p: MutPointer[Int32, MutUntrackedOrigin],
    colid_p: MutPointer[Int32, MutUntrackedOrigin],
    quesval_p: MutPointer[Float32, MutUntrackedOrigin],
    left_p: MutPointer[Int32, MutUntrackedOrigin],
    leaves_p: MutPointer[Float32, MutUntrackedOrigin],
    n_trees: Int,
    n_nodes: Int,
    n_cols: Int,
    num_outputs: Int,
) raises -> List[EtTree[DType.float32]]:
    """MIRRORS the rebuild in `et_predict_binding`,
    `bindings/_mojolearn_trees.mojo:500-523`."""
    var trees = List[EtTree[DType.float32]](capacity=n_trees)
    for t in range(n_trees):
        var lo = _tree_span(offsets_p, t, n_nodes)
        var count = Int(offsets_p[t + 1]) - lo
        var nodes = List[EtNode[DType.float32]](capacity=count)
        var vleaf = List[Float32](capacity=count * num_outputs)
        for i in range(lo, lo + count):
            _check_node(colid_p[i], left_p[i], i - lo, count, n_cols)
            nodes.append(
                EtNode[DType.float32](colid_p[i], quesval_p[i], 0.0, left_p[i], 0)
            )
            for k in range(num_outputs):
                vleaf.append(leaves_p[i * num_outputs + k])
        trees.append(
            EtTree[DType.float32](
                Int32(t), 0, 0, Int32(num_outputs), vleaf^, nodes^
            )
        )
    return trees^


def et_host_predict(
    trees: List[EtTree[DType.float32]],
    rows: List[Float32],
    n_rows: Int,
    n_cols: Int,
    n_trees: Int,
    num_outputs: Int,
    mut out: List[Float32],
) raises:
    """MIRRORS `forest_vote`, `extratrees/impl/randomforest/randomforest.mojo:516-547`,
    once per row, into `out` as `et_predict_binding` writes it (`:528-532`)."""
    if n_rows <= 0 or n_cols <= 0:
        raise Error("forest host: n_rows and n_cols must be positive")
    if n_trees <= 0 or len(trees) != n_trees:
        raise Error("forest host: the tree list does not hold n_trees trees")
    if len(rows) < n_rows * n_cols:
        raise Error("forest host: rows holds fewer than n_rows * n_cols values")
    if len(out) < n_rows * num_outputs:
        raise Error("forest host: out holds fewer than n_rows * num_outputs values")
    var divisor = _divisor(n_trees)
    for r in range(n_rows):
        # `std::vector<T> row_prediction(num_outputs)`, zero-initialized.
        var acc = List[Float32](length=num_outputs, fill=Float32(0.0))
        # `predict_one`'s `+=`, every tree in order (DEVIATION 147).
        for i in range(n_trees):
            predict_one_accumulate(
                rows,
                r * n_cols,
                trees[i],
                acc,
                0,
                num_outputs,
            )
        # `row_prediction[k] /= n_trees`.
        for k in range(num_outputs):
            out[r * num_outputs + k] = acc[k] / divisor
