# SPDX-License-Identifier: Apache-2.0
"""Typed tree flattening moved word for word from bindings/_mojolearn_rf.mojo.
The shell turns the flat lists into Python objects."""
from core.shell_impl.forest_export_binding_impl import copy_forest_export_leaves
from ensemble.decisiontree.decisiontree import TreeMetaDataNode


def rf_flatten_trees(
    trees: List[TreeMetaDataNode[DType.float32]],
    mut offsets: List[Int64], mut colid: List[Int64], mut quesval: List[Float64],
    mut left_child: List[Int64], mut leaves: List[Float64],
):
    """The retained same-fit diagnostic's five lists, tree by tree."""
    var total = 0
    offsets.append(Int64(0))
    for t in range(len(trees)):
        ref tree = trees[t]
        var n = len(tree.sparsetree)
        total += n
        offsets.append(Int64(total))
        for i in range(n):
            ref node = tree.sparsetree[i]
            colid.append(Int64(Int(node.ColumnId())))
            quesval.append(Float64(node.QueryValue()))
            left_child.append(Int64(Int(node.LeftChildId())))
        for i in range(len(tree.vector_leaf)):
            leaves.append(Float64(tree.vector_leaf[i]))


def rf_export_trees_into(
    trees: List[TreeMetaDataNode[DType.float32]],
    op: MutPointer[Int32, MutUntrackedOrigin],
    cp: MutPointer[Int32, MutUntrackedOrigin],
    tp: MutPointer[Float32, MutUntrackedOrigin],
    lp: MutPointer[Int32, MutUntrackedOrigin],
    leaves_addr: Int,
    outputs: Int,
) raises:
    """Writes the owned trees into the caller's validated export buffers."""
    var total = 0
    op[0] = 0
    for t in range(len(trees)):
        ref tree = trees[t]
        for i in range(len(tree.sparsetree)):
            ref node = tree.sparsetree[i]
            cp[total + i] = Int32(node.ColumnId())
            tp[total + i] = node.QueryValue()
            lp[total + i] = Int32(node.LeftChildId())
        copy_forest_export_leaves(tree.vector_leaf, leaves_addr,
                                  total * outputs)
        total += len(tree.sparsetree)
        op[t + 1] = Int32(total)
