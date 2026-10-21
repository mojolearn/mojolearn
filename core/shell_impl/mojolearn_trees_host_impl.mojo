# SPDX-License-Identifier: Apache-2.0
"""Typed model-array loops of the ExtraTrees host binding, moved word for word
from bindings/_mojolearn_trees_host.mojo."""
from core.shell_impl.forest_export_binding_impl import copy_forest_export_leaves
from extratrees.estimator import FitResult


def et_forest_lists(
    result: FitResult,
    mut offsets: List[Int64], mut colid: List[Int64], mut quesval: List[Float64],
    mut left_child: List[Int64], mut leaves: List[Float64],
):
    """The five model arrays `_forest_out` publishes, tree by tree."""
    var total = 0
    offsets.append(Int64(0))
    for t in range(len(result.forest.trees)):
        var n = result.forest.trees[t].num_nodes()
        total += n
        offsets.append(Int64(total))
        for i in range(n):
            var node = result.forest.trees[t].sparsetree[i]
            colid.append(Int64(Int(node.colid)))
            quesval.append(Float64(node.quesval))
            left_child.append(Int64(Int(node.left_child_id)))
        for i in range(len(result.forest.trees[t].vector_leaf)):
            leaves.append(Float64(result.forest.trees[t].vector_leaf[i]))


def et_export_into(
    model: FitResult,
    op: MutPointer[Int32, MutUntrackedOrigin],
    cp: MutPointer[Int32, MutUntrackedOrigin],
    tp: MutPointer[Float32, MutUntrackedOrigin],
    lp: MutPointer[Int32, MutUntrackedOrigin],
    leaves_addr: Int,
    outputs: Int,
) raises:
    """The retained forest into the caller's five validated arrays."""
    var total = 0
    op[0] = 0
    for t in range(len(model.forest.trees)):
        ref tree = model.forest.trees[t]
        for i in range(tree.num_nodes()):
            ref node = tree.sparsetree[i]
            cp[total + i] = node.colid
            tp[total + i] = node.quesval
            lp[total + i] = node.left_child_id
        copy_forest_export_leaves(
            tree.vector_leaf, leaves_addr, total * outputs
        )
        total += tree.num_nodes()
        op[t + 1] = Int32(total)
