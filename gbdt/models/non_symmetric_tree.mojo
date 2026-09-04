# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""A tree whose leaves do NOT all share one split list.

PORT OF `catboost/cuda/models/non_symmetric_tree.h` (and its one-function
`non_summetric_tree.cpp` -- their spelling of the basename, kept only in
this citation) at CatBoost `54a8143a`. Transliterated. Do not improve.

This is the model `EGrowPolicy::Depthwise` and `EGrowPolicy::Lossguide`
build, where `SymmetricTree` builds `TObliviousTreeModel`. The searcher is
the SAME searcher for all three -- `TGreedyTreeLikeStructureSearcher<TModel>`
is templated on the model and `BuildTreeLikeModel<TModel>` is the only thing
that differs at the end (`structure_searcher_template.h:66`). So the whole
distance between the symmetric lane and this one is: which score kernel runs
per level, which leaves get split, and which of these two structures the
paths are folded into.

THE FLAT LAYOUT, which is the only part worth stating twice
----------------------------------------------------------
`nodes` is a PRE-ORDER array. `left_subtree` and `right_subtree` are LEAF
COUNTS under each side, not node indices. From node `i`:

    left child's node  = i + 1                 (when left_subtree  != 1)
    right child's node = i + left_subtree      (when right_subtree != 1)
    a subtree of size 1 is a LEAF, not a node

and a leaf's bin index is the number of leaves to its left, which the apply
walk accumulates as `bin += node.left_subtree` every time it goes right
(`add_model_value.cu:378`). `leaves_count()` is `len(nodes) + 1` because a
binary tree with L leaves has L-1 internal nodes (`non_symmetric_tree.h:36`).

WHAT IS PORTED
--------------
`TNonSymmetricTreeStructure` (nodes, split types, `LeavesCount`, `VisitBins`)
and `TNonSymmetricTree` (structure, values, weights, dim). `GetHash`,
`Rescale`, `ShiftLeafValues`, `UpdateLeaves`, `UpdateWeights` and
`ComputeBins` are NOT written: the first is a hash-map key this port does not
need (see `data/leaf_path.mojo`), the next four are the boosting loop's and
this lane does not own the boosting driver, and `ComputeBins` is the DEVICE
apply, which lives with its kernel in `models/kernel/add_bin_values.mojo`
exactly as theirs lives in `add_model_value.cu`. Every one of them is
recorded in `archive/plans/UNWIRED.md` rather than written and left unreachable.
"""

from gbdt.data.leaf_path import TLeafPath
from gbdt.gpu_data.gpu_structures import TTreeNode
from gbdt.methods.helpers import SPLIT_VALUE_ONE, SPLIT_VALUE_ZERO
from gbdt.models.oblivious_model import TBinarySplit


@fieldwise_init
struct TVisitedLeaf(Copyable, Movable):
    """One `(path, bin)` pair their `VisitBins` visitor is called with.

    ================= DEVIATION BLOCK =================
    Theirs takes a VISITOR: `VisitBins(visitor)` calls `visitor(path, bin)`
    once per leaf and materializes nothing. Ours RETURNS THE LIST.

    The reason is not style. Their visitor is a C++ template parameter
    instantiated at every call site, and the two call sites in this lane
    (`model_builder`'s validation and the checks' leaf walk) both want the
    whole list anyway. A Mojo closure parameter would buy laziness that no
    caller uses and would make the walk's own state harder to read, which is
    the opposite of what a transliteration is for.

    THE COST IS BOUNDED AND STATED: one `TLeafPath` per leaf, so
    `O(leaves * depth)` host memory, on a host structure whose leaf count is
    `max_leaves` and whose depth is `max_depth`. Nothing row-scaled. That is
    inside archive/reference/HOST_AND_DEVICE.md's rule one.
    ===================================================
    """

    var path: TLeafPath
    var bin: Int


struct TNonSymmetricTreeStructure(Copyable, Movable):
    """`TNonSymmetricTreeStructure` (`non_symmetric_tree.h:10`).

    Their `SplitTypes` carries the comment "used only for conversion"
    (`:122`) and is a PARALLEL vector to `Nodes` rather than a member of
    `TTreeNode`, because `TTreeNode` is a device POD and `EBinSplitType` is
    not. Same split here, same reason: `gbdt/gpu_data/gpu_structures.mojo`'s
    `TTreeNode` is what the apply kernel reads.
    """

    var nodes: List[TTreeNode]
    var split_types: List[Int32]

    def __init__(out self):
        self.nodes = List[TTreeNode]()
        self.split_types = List[Int32]()

    def leaves_count(self) -> Int:
        """Their `LeavesCount()`, `Nodes.size() + 1` (`:35-37`).

        A SINGLE-LEAF TREE HAS ZERO NODES AND ONE LEAF, and their formula
        gives that without a special case. It is reachable here: a root that
        finds no improving split is a constant tree, which is what their
        `SplitLeaves` produces when `leavesToSplit` comes back empty
        (`greedy_search_helper.cpp:619-622`).
        """
        return len(self.nodes) + 1

    def __eq__(self, other: Self) -> Bool:
        """Their `operator==` (`:39-41`), both vectors."""
        if len(self.nodes) != len(other.nodes):
            return False
        if len(self.split_types) != len(other.split_types):
            return False
        for i in range(len(self.nodes)):
            if self.nodes[i] != other.nodes[i]:
                return False
        for i in range(len(self.split_types)):
            if self.split_types[i] != other.split_types[i]:
                return False
        return True

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    def visit_bins(self) raises -> List[TVisitedLeaf]:
        """`VisitBinsImpl` (`non_symmetric_tree.h:57-114`), the walk copied.

        Their iterative pre-order walk with an explicit node stack and an
        `unwind` flag, statement for statement. It is worth resisting the
        urge to rewrite it as recursion: the flag encodes WHICH CHILD we
        came back from, and getting that wrong produces a tree whose leaves
        are all reachable and whose PATHS are wrong, which no leaf count and
        no row conservation can see.

        The rule the flag implements: on returning to node `current` from
        child `prev`, the left child's node index is `current + 1` and the
        right child's is `current + left_subtree`. So
        `current + left_subtree != prev` means we came back from the LEFT
        and the right side is still owed; equality means the node is done.

        `binCursor` counts leaves in visit order, which IS the leaf
        numbering the apply kernel computes by accumulating `left_subtree`
        (`add_model_value.cu:378`). The two agreeing is not an assumption:
        `checks/depthwise_model_check.mojo` walks both and compares.
        """
        var out = List[TVisitedLeaf]()
        var current_path = TLeafPath()
        var bin_cursor = 0

        # `if (Nodes.empty()) { visitor(currentPath, binCursor); return; }`
        # -- the constant tree, one leaf, empty path.
        if len(self.nodes) == 0:
            out.append(TVisitedLeaf(current_path.copy(), bin_cursor))
            return out^

        var nodes = List[Int]()
        nodes.append(0)

        var unwind = False
        var prev = 0

        while len(nodes) > 0:
            var current = nodes[len(nodes) - 1]
            var node = self.nodes[current]
            # `CB_ENSURE(currentNode.LeftSubtree >= 1 &&
            #            currentNode.RightSubtree >= 1, ...)`
            if node.left_subtree < UInt16(1) or node.right_subtree < UInt16(1):
                raise Error("Left and/or right subtree is missing")

            if unwind:
                if current + Int(node.left_subtree) != prev:
                    _resize_path(current_path, len(nodes))
                    current_path.directions[len(current_path.directions) - 1] = (
                        SPLIT_VALUE_ONE
                    )

                    if node.right_subtree == UInt16(1):
                        out.append(TVisitedLeaf(current_path.copy(), bin_cursor))
                        bin_cursor += 1
                    else:
                        nodes.append(current + Int(node.left_subtree))
                        unwind = False
                        continue
                prev = current
                _ = nodes.pop()
            else:
                current_path.splits.append(
                    TBinarySplit(
                        Int32(node.feature_id),
                        Int32(node.bin),
                        self.split_types[current],
                    )
                )
                current_path.directions.append(SPLIT_VALUE_ZERO)

                if node.left_subtree != UInt16(1):
                    nodes.append(current + 1)
                else:
                    out.append(TVisitedLeaf(current_path.copy(), bin_cursor))
                    bin_cursor += 1
                    current_path.directions[
                        len(current_path.directions) - 1
                    ] = SPLIT_VALUE_ONE

                    if node.right_subtree == UInt16(1):
                        out.append(
                            TVisitedLeaf(current_path.copy(), bin_cursor)
                        )
                        bin_cursor += 1
                        unwind = True
                        prev = current
                        _ = nodes.pop()
                    else:
                        nodes.append(current + 1)

        return out^


def _resize_path(mut path: TLeafPath, size: Int) raises:
    """Their `Splits.resize(n)` / `Directions.resize(n)` (`:80-81`).

    ONLY EVER SHRINKS on this walk: the path grows one entry per node the
    non-unwind branch descends into, and the unwind branch resizes back to
    the CURRENT stack depth, which is never deeper. Growing would mean
    inventing a split, so this raises instead of default-constructing one --
    their `TVector::resize` would have silently appended a zeroed
    `TBinarySplit`, and a zeroed split is a real split on feature 0 bin 0.
    """
    if size > len(path.splits):
        raise Error(
            String("leaf-path resize would invent ")
            + String(size - len(path.splits))
            + " split(s); the walk is out of step with its stack"
        )
    while len(path.splits) > size:
        _ = path.splits.pop()
        _ = path.directions.pop()


struct TNonSymmetricTree(Copyable, Movable):
    """`TNonSymmetricTree` (`non_symmetric_tree.h:120`), their four members.

    `leaf_values` is `outputDim` floats per leaf, flat, leaf-major -- their
    `TConstArrayRef<float>(LeafValues.data() + bin * Dim, Dim)`
    (`:191`). `leaf_weights` is one per leaf. Both are indexed by the BIN
    the structure's walk assigns, not by the searcher's leaf id; the model
    builder is what re-indexes them.
    """

    var model_structure: TNonSymmetricTreeStructure
    var leaf_values: List[Float32]
    var leaf_weights: List[Float64]
    var dim: Int

    def __init__(
        out self,
        var model_structure: TNonSymmetricTreeStructure,
        var leaf_values: List[Float32],
        var leaf_weights: List[Float64],
        dim: Int,
    ):
        self.model_structure = model_structure^
        self.leaf_values = leaf_values^
        self.leaf_weights = leaf_weights^
        self.dim = dim

    def output_dim(self) -> Int:
        """Their `OutputDim()`."""
        return self.dim

    def bin_count(self) -> Int:
        """Their `BinCount()`, `ModelStructure.LeavesCount()`."""
        return self.model_structure.leaves_count()
