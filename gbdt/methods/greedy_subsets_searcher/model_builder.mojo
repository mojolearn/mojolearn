# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Fold a list of leaf PATHS into a tree.

PORT OF `TFlatTreeBuilder` and `BuildTreeLikeModel<TNonSymmetricTree>` in
`catboost/cuda/methods/greedy_subsets_searcher/model_builder.cpp` at CatBoost
`54a8143a`. Transliterated. Do not improve.

The searcher hands back leaves as a flat `TVector<TLeafPath>` in LEAF-ID
order, plus a weight and a value vector per leaf. Nothing in that list says
which leaf is whose sibling; the paths do. This file rebuilds the tree from
them and flattens it into the pre-order `TTreeNode` array the apply kernel
walks.

WHY A TREE AND NOT AN INDEX MAP. A depthwise tree's leaf ids are assigned in
split order (`MakeSplit` gives the right child `leavesCount + i`,
`split_properties_helper.cpp:862`), so leaf id order is neither left-to-right
nor depth order. The model's bin numbering IS left-to-right, because that is
what `bin += node.LeftSubtree` produces. Sorting ids cannot get there; only
the paths can.

THREE OF THEIR SPECIALIZATIONS, ONE PORTED
------------------------------------------
Theirs has `BuildTreeLikeModel<TObliviousTreeModel>`,
`<TRegionModel>` and `<TNonSymmetricTree>`. Only the third is here.

* The OBLIVIOUS one is not needed: this port's symmetric lane never builds
  `TLeafPath`s to fold back (`run_tree_layout` returns the split list
  directly, which is what their specialization spends its body recovering
  -- `structure.Splits = leaves[0].Splits`). Porting it now would mean
  writing a function with no caller. `UNWIRED.md` records it.
* The REGION one belongs to `EGrowPolicy::Region`, which no lane owns.

`EDuplicateTerminalLeavesPolicy` IS ported, both arms, because their
non-symmetric call site passes `Exception` (`model_builder.cpp:288`) and the
`Combine` arm is what makes the enum meaningful. `Combine` is reached by
`checks/depthwise_model_check.mojo`'s duplicate-path claim, so neither arm
is inert.
"""

from gbdt.data.leaf_path import TLeafPath, split_equal
from gbdt.gpu_data.gpu_structures import TTreeNode
from gbdt.methods.helpers import SPLIT_VALUE_ONE, SPLIT_VALUE_ZERO
from gbdt.models.non_symmetric_tree import (
    TNonSymmetricTree,
    TNonSymmetricTreeStructure,
)
from gbdt.models.oblivious_model import TBinarySplit


#: Their `EDuplicateTerminalLeavesPolicy::Combine` (`model_builder.cpp:141`).
comptime DUPLICATE_LEAVES_COMBINE = 0

#: Their `EDuplicateTerminalLeavesPolicy::Exception`.
comptime DUPLICATE_LEAVES_EXCEPTION = 1


@fieldwise_init
struct _Node(Copyable, Movable):
    """One node of the intermediate tree their `TNode` holds by pointer.

    ================= DEVIATION BLOCK =================
    Theirs is `TSimpleSharedPtr<TNode>` with a `std::variant<TLeaf,
    TBinarySplit>` payload. Ours is an ARENA: every node lives in one
    `List[_Node]` and a child is an INDEX, with `-1` for their `nullptr`.

    Reason, and it is a Mojo one rather than a preference: a recursive
    struct needs indirection, and the reference-counted-pointer form makes
    the null test, the variant discriminant and the ownership all separate
    problems. An index arena makes `nullptr` a number the type system can
    hold and makes the flatten pass a plain loop over a list.

    THE SHAPE IS UNCHANGED. `is_terminal` is their variant discriminant,
    `split` is their `TBinarySplit` arm, `weight`/`values` are their `TLeaf`
    arm, and a node uses exactly one of the two -- which is asserted at every
    read, exactly where their `std::get` would have thrown.
    ===================================================
    """

    var is_terminal: Bool
    var split: TBinarySplit
    var weight: Float64
    var values: List[Float32]
    var left: Int
    var right: Int


struct TFlatTreeBuilder(Movable):
    """`TFlatTreeBuilder` (`model_builder.cpp:138`)."""

    var nodes: List[_Node]
    var root: Int
    var policy: Int

    def __init__(out self, policy: Int) raises:
        """Their `explicit TFlatTreeBuilder(EDuplicateTerminalLeavesPolicy)`.
        """
        if (
            policy != DUPLICATE_LEAVES_COMBINE
            and policy != DUPLICATE_LEAVES_EXCEPTION
        ):
            raise Error(
                String("EDuplicateTerminalLeavesPolicy is Combine(0) or")
                + " Exception(1), not "
                + String(policy)
            )
        self.nodes = List[_Node]()
        self.root = -1
        self.policy = policy

    def _child(self, parent: Int, is_right: Bool) -> Int:
        """Their `cursor`, dereferenced. `parent < 0` is the root slot."""
        if parent < 0:
            return self.root
        if is_right:
            return self.nodes[parent].right
        return self.nodes[parent].left

    def _set_child(mut self, parent: Int, is_right: Bool, child: Int):
        """Their `(*cursor) = new TNode(...)`."""
        if parent < 0:
            self.root = child
            return
        if is_right:
            self.nodes[parent].right = child
        else:
            self.nodes[parent].left = child

    def add(
        mut self, path: TLeafPath, values: List[Float32], weight: Float64
    ) raises:
        """Their `Add(path, values, weight)` (`model_builder.cpp:186-231`).

        Walks the path from the root, creating internal nodes as it goes and
        checking that an existing node carries the SAME split -- their
        `CB_ENSURE(!(*cursor)->IsTerminal() && (*cursor)->GetSplit() ==
        split, "Error: path is not from current tree.")`. That check is the
        one that catches a searcher which recorded a leaf's path wrong, and
        it fires before anything is flattened.
        """
        var parent = -1
        var is_right = False

        for i in range(path.get_depth()):
            var split = path.splits[i]
            var cursor = self._child(parent, is_right)

            if cursor < 0:
                var fresh = _Node(
                    False, split, Float64(0.0), List[Float32](), -1, -1
                )
                self.nodes.append(fresh^)
                cursor = len(self.nodes) - 1
                self._set_child(parent, is_right, cursor)
            else:
                if self.nodes[cursor].is_terminal or not split_equal(
                    self.nodes[cursor].split, split
                ):
                    raise Error("Error: path is not from current tree.")

            # `switch (direction) { Zero -> Left; One -> Right; }`
            var direction = path.directions[i]
            if direction != SPLIT_VALUE_ZERO and direction != SPLIT_VALUE_ONE:
                raise Error(
                    String("ESplitValue is Zero(0) or One(1), not ")
                    + String(direction)
                )
            parent = cursor
            is_right = direction == SPLIT_VALUE_ONE

        var cursor = self._child(parent, is_right)

        if cursor >= 0:
            # `CB_ENSURE((*cursor)->IsTerminal());`
            if not self.nodes[cursor].is_terminal:
                raise Error(
                    "Error: path ends on an internal node; two leaves of"
                    " this tree disagree about the tree's shape"
                )
            if self.policy == DUPLICATE_LEAVES_EXCEPTION:
                raise Error("Can't add terminal leaf twice")
            # `Combine`: their `leaf.Weight += weight` and elementwise
            # `leaf.Values[i] += values[i]` (`:222-227`).
            if len(self.nodes[cursor].values) != len(values):
                raise Error(
                    String("leaf value dimension changed: had ")
                    + String(len(self.nodes[cursor].values))
                    + ", got "
                    + String(len(values))
                )
            self.nodes[cursor].weight += weight
            for i in range(len(values)):
                self.nodes[cursor].values[i] += values[i]
        else:
            var leaf = _Node(
                True,
                TBinarySplit(Int32(0), Int32(0), Int32(0)),
                weight,
                values.copy(),
                -1,
                -1,
            )
            self.nodes.append(leaf^)
            self._set_child(parent, is_right, len(self.nodes) - 1)

    def build_flat(
        mut self,
        mut out_nodes: List[TTreeNode],
        mut out_split_types: List[Int32],
        mut out_values: List[Float32],
        mut out_weights: List[Float64],
    ) raises:
        """Their `BuildFlat` (`:233-242`), which clears then visits."""
        out_nodes.clear()
        out_split_types.clear()
        out_values.clear()
        out_weights.clear()
        _ = self._visit(
            self.root, out_nodes, out_split_types, out_values, out_weights
        )

    def _visit(
        self,
        cursor: Int,
        mut flat_nodes: List[TTreeNode],
        mut flat_split_types: List[Int32],
        mut leaves_values: List[Float32],
        mut weights: List[Float64],
    ) raises -> Int:
        """Their `Visit` (`:245-274`), returning the subtree's LEAF COUNT.

        Their pre-order emission is what makes `index + 1` the left child
        and `index + LeftSubtree` the right: the node is pushed BEFORE
        either recursion, and both subtree sizes are written back into the
        slot afterwards. Copied in that order, including the write-back.
        """
        # `CB_ENSURE(cursor, "Tree is empty (cursor is nullptr)");`
        if cursor < 0:
            raise Error("Tree is empty (cursor is nullptr)")

        if self.nodes[cursor].is_terminal:
            for i in range(len(self.nodes[cursor].values)):
                leaves_values.append(self.nodes[cursor].values[i])
            weights.append(self.nodes[cursor].weight)
            return 1

        var split = self.nodes[cursor].split
        flat_nodes.append(
            TTreeNode(
                _to_ui16(Int(split.feature_id), "feature id"),
                _to_ui16(Int(split.bin_idx), "bin"),
                UInt16(0),
                UInt16(0),
            )
        )
        flat_split_types.append(split.split_type)
        var idx = len(flat_nodes) - 1

        var left_subtree = self._visit(
            self.nodes[cursor].left,
            flat_nodes,
            flat_split_types,
            leaves_values,
            weights,
        )
        var right_subtree = self._visit(
            self.nodes[cursor].right,
            flat_nodes,
            flat_split_types,
            leaves_values,
            weights,
        )

        flat_nodes[idx].left_subtree = _to_ui16(left_subtree, "left subtree")
        flat_nodes[idx].right_subtree = _to_ui16(
            right_subtree, "right subtree"
        )
        return left_subtree + right_subtree


def _to_ui16(value: Int, what: String) raises -> UInt16:
    """`TTreeNode`'s fields are `ui16` and this is where that bites.

    Theirs narrows silently -- `node.FeatureId = split.FeatureId` from a
    `ui32`. A dataset with more than 65,535 features, or a tree wider than
    65,535 leaves on one side, would wrap and produce a model that applies
    cleanly and answers wrong. This raises instead, which is a DEVIATION
    (PORTING.md 350) and the only kind that is free: it cannot change any
    model their code would have built correctly.
    """
    if value < 0 or value > 65535:
        raise Error(
            String("TTreeNode.")
            + what
            + " does not fit in ui16: "
            + String(value)
            + " (CatBoost's TTreeNode is four ui16 fields,"
            " gpu_structures.h:167)"
        )
    return UInt16(value)


def build_non_symmetric_tree(
    leaves: List[TLeafPath],
    leaves_weight: List[Float64],
    leaves_values: List[List[Float32]],
) raises -> TNonSymmetricTree:
    """`BuildTreeLikeModel<TNonSymmetricTree>` (`model_builder.cpp:278-296`).

    Their three ensures, then the builder, then the flatten:

        CB_ENSURE(leaves.size(), "Error: empty region");
        CB_ENSURE(leaves.size() == leavesValues.size());
        CB_ENSURE(leaves.size() == leavesWeight.size());

    The policy is `Exception`: two identical paths in one tree means the
    searcher produced the same leaf twice, and combining them would hide it.
    """
    if len(leaves) == 0:
        raise Error("Error: empty region")
    if len(leaves) != len(leaves_values):
        raise Error(
            String("leaves.size()=")
            + String(len(leaves))
            + " != leavesValues.size()="
            + String(len(leaves_values))
        )
    if len(leaves) != len(leaves_weight):
        raise Error(
            String("leaves.size()=")
            + String(len(leaves))
            + " != leavesWeight.size()="
            + String(len(leaves_weight))
        )

    var builder = TFlatTreeBuilder(DUPLICATE_LEAVES_EXCEPTION)
    for leaf in range(len(leaves)):
        builder.add(leaves[leaf], leaves_values[leaf], leaves_weight[leaf])

    var structure = TNonSymmetricTreeStructure()
    var weights = List[Float64]()
    var values = List[Float32]()
    builder.build_flat(
        structure.nodes, structure.split_types, values, weights
    )
    var output_dim = len(leaves_values[0])
    return TNonSymmetricTree(structure^, values^, weights^, output_dim)
