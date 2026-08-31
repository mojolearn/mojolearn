# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The flat tree node, the flat tree, and the host predict traversal.

PORT OF cuML at `00094f7` (`~/CascadeProjects/upstream/cuml`). Transliterated.
Do not improve.

| piece                                | cuML file:lines                                     |
|--------------------------------------|-----------------------------------------------------|
| `SparseTreeNode`                     | `cpp/include/cuml/tree/flatnode.h:33-77`            |
| ... fields                           | `flatnode.h:36-40`                                   |
| ... private ctor                     | `flatnode.h:41-49`                                   |
| ... accessors                        | `flatnode.h:52-57`                                   |
| ... `CreateSplitNode`                | `flatnode.h:59-64`                                   |
| ... `CreateLeafNode`                 | `flatnode.h:65-68`                                   |
| ... `IsLeaf`                         | `flatnode.h:69`                                      |
| ... `operator==`                     | `flatnode.h:70-76`                                   |
| `TreeMetaDataNode`                   | `cpp/include/cuml/tree/decisiontree.hpp:101-109`    |
| `predict` / `predict_all`            | `cpp/src/decisiontree/decisiontree.cuh:359-392`     |
| `predict_one` (THE traversal)        | `cpp/src/decisiontree/decisiontree.cuh:394-413`     |
| leaf slice `id * num_outputs`        | `decisiontree.cuh:411` and `decisiontree.cuh:218`   |
| argmax over the leaf vector          | `cpp/src/randomforest/randomforest.cuh:243-253`     |

**The right child is not stored.** `RightChildId() == left_child_id + 1`
(`flatnode.h:56`), i.e. children are always allocated as an adjacent pair. The
builder depends on it, `build_treelite_tree` depends on it
(`decisiontree.cuh:205-211` allocates `cleft` then `cright` back to back), and
nothing here may break it.

**Which side `<=` takes, answered from the files, not from memory.** cuML's
`predict_one` is `if (row[n.ColumnId()] <= n.QueryValue()) idx = n.LeftChildId()`
(`decisiontree.cuh:403-404`), and their treelite export writes the same
predicate as `tl::Operator::kLE` on the numerical test
(`decisiontree.cuh:214-215`). scikit-learn `1.9.0`'s `partition_samples` sends
`feature_value <= threshold` LEFT: `sklearn/tree/_partitioner.pyx:238`, inside
`DensePartitioner.partition_samples` at `:217-246`, and the same predicate
again at `:272` (`partition_samples_final`) and at `:488`
(`SparsePartitioner.partition_samples`), so all three of sklearn's partition
sites agree with each other. **cuML and sklearn agree.** `x <= quesval` goes
LEFT; there was nothing to choose between and no deviation was needed.

**The struct is POD on purpose.** Five scalar fields, no `String`, no `List`,
no heap, so a kernel can be handed `tree.sparsetree.unsafe_ptr()` and read it.
This repository's hard-won trap is that a WHOLE-STRUCT load inside a Metal
kernel kills the compiler, so every field also has a pointer-and-index reader
(`ColumnIdAt`, `QueryValueAt`, `LeftChildIdAt`, `InstanceCountAt`,
`IsLeafAt`) that touches exactly one field through the pointer. Device code
must use those and must never materialise a `SparseTreeNode` value.

**This substrate is DELIBERATELY DUPLICATED.** A parallel `ensemble/` lane is
writing its own flat node, its own tree container and its own predict
traversal from the same cuML files at the same pin. That is not an oversight
and it is not to be fixed by importing across the lane boundary. Rule 12 of
this repository says the predictor of integration pain is FILE CONVERGENCE,
not delegation: two lanes editing one file collide, two lanes editing two
files do not. So the two lanes duplicate rather than share, this file imports
NOTHING from `ensemble/`, `core/` or `gbdt/`, and deduplicating the two copies
into `core/` is a merge-time decision belonging to whoever runs the merge and
to neither lane.

Checked by `extratrees/original/flatnode_check.mojo`.
"""


comptime NODE_IS_LEAF: Int32 = -1
"""`left_child_id`'s sentinel. `flatnode.h:39` initialises the field to -1 and
`flatnode.h:69` tests it for -1; that is the whole leaf predicate."""


# ================================ DEVIATION BLOCK ======================
# 148. THE TEMPLATE SHAPE IS NARROWED: `LabelT` IS DROPPED AND `IdxT` IS
# FIXED TO `Int32`.
#
# **What theirs does.** `template <typename DataT, typename LabelT,
# typename IdxT = int> struct SparseTreeNode` (`flatnode.h:33`). Three
# parameters. `LabelT` appears in NONE of the five fields
# (`flatnode.h:36-40`) and in none of the accessors (`:52-57`); it exists
# only so that `TreeMetaDataNode<T, L>` (`decisiontree.hpp:101-109`) and the
# four typedefs at `decisiontree.hpp:141-144` can spell a matching node
# type. `IdxT` defaults to `int`, and both factories spell their return type
# `SparseTreeNode<DataT, LabelT>` (`flatnode.h:62` and `:67`) — dropping
# `IdxT`, so a caller who instantiated with a non-default `IdxT` gets a
# node of a DIFFERENT type back from `CreateSplitNode`. Nobody in the tree
# does, so it never fires upstream.
#
# **What ours does.** One parameter, `dtype`, carrying `DataT`. `LabelT` is
# not modelled at all. `IdxT` is `Int32` everywhere, spelled out rather than
# defaulted.
#
# **Why.** `LabelT` is a phantom parameter here, and a phantom parameter in
# Mojo would have to be written at every use site of a struct that never
# reads it. Fixing `IdxT` to `Int32` removes their factory bug by
# construction rather than reproducing a latent one: there is only one index
# type, so there is only one node type. `int` is 32-bit on every platform
# cuML builds for, so `Int32` is what `IdxT = int` means, not a narrowing.
#
# **Price.** A caller who wanted 64-bit node indices cannot have them. cuML
# cannot either — `left_child_id` is `IdxT` (`flatnode.h:39`), so their tree
# is already capped at 2^31 nodes no matter what they pass. The cap is
# theirs; we only stopped pretending it is configurable.
#
# **What is NOT a deviation, and is mirrored exactly.** Their accessors
# `LeftChildId()` and `RightChildId()` return `int64_t` (`flatnode.h:55-56`)
# while the FIELD is `IdxT`, and their constructor takes `int64_t
# left_child_id` (`flatnode.h:42`) and narrows it into the `IdxT` field
# (`flatnode.h:46`) with no check. Ours does the same: `Int32` field,
# `Int64` on the way in and on the way out. It is a real asymmetry in their
# header, it is load-bearing for nothing, and rule 0a says we transcribe it
# rather than tidy it.
# ======================================================================


@fieldwise_init
struct SparseTreeNode[dtype: DType](
    Copyable, Equatable, ImplicitlyCopyable, Movable
):
    """`SparseTreeNode`, `flatnode.h:33-77`. Five scalars, nothing else.

    Field for field, in their declaration order (`flatnode.h:36-40`), because
    the order is the in-memory layout a kernel reads through a pointer.
    """

    var colid: Int32
    """`IdxT colid = 0` (`flatnode.h:36`). The column the test reads."""

    var quesval: Scalar[Self.dtype]
    """`DataT quesval = DataT(0)` (`flatnode.h:37`). The threshold."""

    var best_metric_val: Scalar[Self.dtype]
    """`DataT best_metric_val = DataT(0)` (`flatnode.h:38`). Carried for
    reporting only; the traversal never reads it."""

    var left_child_id: Int32
    """`IdxT left_child_id = -1` (`flatnode.h:39`). -1 means leaf."""

    var instance_count: Int32
    """`IdxT instance_count = 0` (`flatnode.h:40`). Rows that reached here."""

    # ---- accessors, `flatnode.h:52-57` and `:69` ----------------------
    # Their names, not Mojo's naming convention. These are the transcribed
    # API and the call sites in `decisiontree.cuh` and `randomforest.cuh`
    # are transcribed against them.

    def ColumnId(self) -> Int32:
        """`flatnode.h:52`."""
        return self.colid

    def QueryValue(self) -> Scalar[Self.dtype]:
        """`flatnode.h:53`."""
        return self.quesval

    def BestMetric(self) -> Scalar[Self.dtype]:
        """`flatnode.h:54`."""
        return self.best_metric_val

    def LeftChildId(self) -> Int64:
        """`flatnode.h:55`. `int64_t` out of an `IdxT` field, theirs."""
        return Int64(self.left_child_id)

    def RightChildId(self) -> Int64:
        """`flatnode.h:56`. The right child is NOT stored; it is the left
        child's neighbour. Children are allocated as an adjacent pair."""
        return Int64(self.left_child_id) + 1

    def InstanceCount(self) -> Int32:
        """`flatnode.h:57`."""
        return self.instance_count

    def IsLeaf(self) -> Bool:
        """`flatnode.h:69`."""
        return self.left_child_id == NODE_IS_LEAF

    # ---- factories, `flatnode.h:59-68` --------------------------------

    @staticmethod
    def CreateSplitNode(
        colid: Int32,
        quesval: Scalar[Self.dtype],
        best_metric_val: Scalar[Self.dtype],
        left_child_id: Int64,
        instance_count: Int32,
    ) -> Self:
        """`flatnode.h:59-64`. `left_child_id` arrives as `int64_t` and is
        narrowed into the `IdxT` field unchecked, exactly as theirs does at
        `flatnode.h:46`."""
        return Self(
            colid,
            quesval,
            best_metric_val,
            Int32(left_child_id),
            instance_count,
        )

    @staticmethod
    def CreateLeafNode(instance_count: Int32) -> Self:
        """`flatnode.h:65-68`. `{0, 0, 0, -1, instance_count}`."""
        return Self(0, 0, 0, NODE_IS_LEAF, instance_count)

    # ---- `operator==`, `flatnode.h:70-76` -----------------------------

    def __eq__(self, other: Self) -> Bool:
        """`flatnode.h:70-76`. All five fields, `quesval` and
        `best_metric_val` by exact float equality, theirs."""
        return (
            self.colid == other.colid
            and self.quesval == other.quesval
            and self.best_metric_val == other.best_metric_val
            and self.left_child_id == other.left_child_id
            and self.instance_count == other.instance_count
        )

    def __ne__(self, other: Self) -> Bool:
        """Not in their header; C++ synthesises it. Mojo does not."""
        return not (self == other)

    # ---- field-by-field readers through a pointer ---------------------
    # NOT in cuML: their `predict_one` does `auto n = tree.sparsetree[idx]`
    # (`decisiontree.cuh:401`), a whole-struct load, which is fine in nvcc
    # and is the exact shape that kills the Metal compiler here. These are
    # the same reads, one field at a time, and they are what device code
    # must call. No deviation number: nothing about the ALGORITHM changes,
    # only which instruction loads the field.

    @staticmethod
    def ColumnIdAt[o: Origin](nodes: Pointer[Self, origin=o], i: Int) -> Int32:
        """`nodes[i].ColumnId()` without loading the struct."""
        return nodes.unsafe_offset(i)[].colid

    @staticmethod
    def QueryValueAt[
        o: Origin
    ](nodes: Pointer[Self, origin=o], i: Int) -> Scalar[Self.dtype]:
        """`nodes[i].QueryValue()` without loading the struct."""
        return nodes.unsafe_offset(i)[].quesval

    @staticmethod
    def BestMetricAt[
        o: Origin
    ](nodes: Pointer[Self, origin=o], i: Int) -> Scalar[Self.dtype]:
        """`nodes[i].BestMetric()` without loading the struct."""
        return nodes.unsafe_offset(i)[].best_metric_val

    @staticmethod
    def LeftChildIdAt[o: Origin](nodes: Pointer[Self, origin=o], i: Int) -> Int64:
        """`nodes[i].LeftChildId()` without loading the struct."""
        return Int64(nodes.unsafe_offset(i)[].left_child_id)

    @staticmethod
    def RightChildIdAt[
        o: Origin
    ](nodes: Pointer[Self, origin=o], i: Int) -> Int64:
        """`nodes[i].RightChildId()` without loading the struct."""
        return Int64(nodes.unsafe_offset(i)[].left_child_id) + 1

    @staticmethod
    def InstanceCountAt[
        o: Origin
    ](nodes: Pointer[Self, origin=o], i: Int) -> Int32:
        """`nodes[i].InstanceCount()` without loading the struct."""
        return nodes.unsafe_offset(i)[].instance_count

    @staticmethod
    def IsLeafAt[o: Origin](nodes: Pointer[Self, origin=o], i: Int) -> Bool:
        """`nodes[i].IsLeaf()` without loading the struct."""
        return nodes.unsafe_offset(i)[].left_child_id == NODE_IS_LEAF


# ================================ DEVIATION BLOCK ======================
# 146. `TreeMetaDataNode` IS REDUCED: `train_time` IS NOT PORTED.
#
# **What theirs does.** `decisiontree.hpp:101-109`:
#
#     struct TreeMetaDataNode {
#       int treeid;
#       int depth_counter;
#       int leaf_counter;
#       double train_time;
#       std::vector<T> vector_leaf;
#       std::vector<SparseTreeNode<T, L>> sparsetree;
#       int num_outputs;
#     };
#
# Seven members. `train_time` is a wall-clock seconds figure the builder
# stamps on the tree and the text/JSON dumps print.
#
# **What ours does.** Six: `treeid`, `depth_counter`, `leaf_counter`,
# `num_outputs`, `vector_leaf`, `sparsetree`. `train_time` is absent.
#
# **Why.** Timing is deferred in this lane by explicit instruction — no
# benchmark, no timing claim — so a `double` seconds field would be written
# by nothing and read by nothing, and rule 3 says an unported thing must be
# VISIBLE rather than present-but-dead. It is also the only member that
# could not survive a move to device memory (`gbdt/` has no `float64` on
# device at all), so a per-tree POD carrying it would have to shed it
# anyway. Restoring it is a one-line append whenever a builder needs it.
#
# **Price.** `get_tree_summary_text` / `get_tree_json`
# (`decisiontree.hpp:119`, `:139`) print a train time we cannot print. There
# is no such dump in this lane yet, so the price is currently zero and
# becomes one line when there is.
#
# **What is NOT a deviation.** `std::vector<T>` -> `List[Scalar[dtype]]` and
# `std::vector<SparseTreeNode>` -> `List[SparseTreeNode[dtype]]` are the
# same contiguous owning array in the two languages; `.unsafe_ptr()` on each
# is the pointer a kernel would take, and the LAYOUT is unchanged. The
# member ORDER differs (`num_outputs` is moved up beside the other `int`s
# instead of trailing the two vectors) because nothing indexes this struct
# by offset; the node array's layout is what matters and that is untouched.
# ======================================================================


@fieldwise_init
struct TreeMetaDataNode[dtype: DType](Copyable, Movable):
    """`TreeMetaDataNode`, `decisiontree.hpp:101-109`, minus `train_time`.

    THE LEAF LAYOUT IS THE POINT. `vector_leaf` is FLAT and is sized
    `num_nodes * num_outputs`, so node `i`'s leaf vector is the slice
    `vector_leaf[i * num_outputs ..< (i + 1) * num_outputs]`. cuML never
    names that slice: it inlines `tree.vector_leaf[idx * num_outputs + i]`
    in `predict_one` (`decisiontree.cuh:411`) and
    `rf_tree.vector_leaf.begin() + cuml_node_id * rf_tree.num_outputs` in
    the treelite export (`decisiontree.cuh:218`). Both are the same
    arithmetic and it is stated once here as `leaf_base`.

    Every node gets a slot, INCLUDING internal nodes whose slot is never
    read. That is theirs (`decisiontree.cuh:218` indexes by raw
    `cuml_node_id`, not by a leaf counter), and it is why `leaf_counter` is
    a separate member rather than a stride.
    """

    var treeid: Int32
    """`decisiontree.hpp:102`."""

    var depth_counter: Int32
    """`decisiontree.hpp:103`. Depth of the deepest node."""

    var leaf_counter: Int32
    """`decisiontree.hpp:104`. How many nodes are leaves."""

    var num_outputs: Int32
    """`decisiontree.hpp:108`. Class count for classification, 1 for
    regression. The `vector_leaf` stride."""

    var vector_leaf: List[Scalar[Self.dtype]]
    """`std::vector<T> vector_leaf` (`decisiontree.hpp:106`). Flat,
    `num_nodes * num_outputs` long."""

    var sparsetree: List[SparseTreeNode[Self.dtype]]
    """`std::vector<SparseTreeNode<T, L>> sparsetree`
    (`decisiontree.hpp:107`). Node 0 is the root
    (`decisiontree.cuh:400` starts at `idx = 0`)."""

    def num_nodes(self) -> Int:
        """Not in their struct; they call `tree.sparsetree.size()`
        directly (`decisiontree.cuh:374`)."""
        return len(self.sparsetree)

    def leaf_base(self, node_id: Int) -> Int:
        """Where node `node_id`'s leaf vector starts in `vector_leaf`.

        `node_id * num_outputs`. This is `decisiontree.cuh:411`'s
        `idx * num_outputs` and `decisiontree.cuh:218`'s
        `cuml_node_id * rf_tree.num_outputs`, named once instead of
        inlined twice.
        """
        return node_id * Int(self.num_outputs)

    def leaf_value(self, node_id: Int, output_id: Int) -> Scalar[Self.dtype]:
        """`vector_leaf[node_id * num_outputs + output_id]`,
        `decisiontree.cuh:411`."""
        return self.vector_leaf[self.leaf_base(node_id) + output_id]


# ---- the traversal, `decisiontree.cuh:394-413` ------------------------


def predict_leaf[
    dtype: DType
](
    tree: TreeMetaDataNode[dtype],
    row: List[Scalar[dtype]],
    row_offset: Int,
) -> Int:
    """The traversal from `predict_one` (`decisiontree.cuh:400-409`), with
    the leaf ACCUMULATION at `:410-412` left to the caller.

    Their loop, transcribed:

        std::size_t idx = 0;
        auto n          = tree.sparsetree[idx];
        while (!n.IsLeaf()) {
          if (row[n.ColumnId()] <= n.QueryValue()) {
            idx = n.LeftChildId();
          } else {
            idx = n.RightChildId();
          }
          n = tree.sparsetree[idx];
        }

    `<=` GOES LEFT (`decisiontree.cuh:403-404`), which is also what
    `tl::Operator::kLE` means in their treelite export
    (`decisiontree.cuh:214-215`) and what scikit-learn `1.9.0`'s
    `partition_samples` does (`sklearn/tree/_partitioner.pyx:236-240`).
    Both upstreams agree, so a boundary row whose feature value is EXACTLY
    `quesval` goes LEFT.

    Splitting the leaf lookup out of the loop is not a deviation: it is the
    same statement sequence with the accumulation moved to the caller, and
    it is what lets the check compare the LANDED NODE per row rather than
    only the value read out of it. A value comparison alone cannot tell a
    wrong leaf from a wrong `vector_leaf` stride.

    `row_offset` replaces their pointer arithmetic
    `&rows[row_id * n_cols]` (`decisiontree.cuh:390`).
    """
    var idx = 0
    var n = tree.sparsetree[idx]
    while not n.IsLeaf():
        if row[row_offset + Int(n.ColumnId())] <= n.QueryValue():
            idx = Int(n.LeftChildId())
        else:
            idx = Int(n.RightChildId())
        n = tree.sparsetree[idx]
    return idx


# ================================ DEVIATION BLOCK ======================
# 147. `predict_one` ACCUMULATES INTO THE OUTPUT AND NEVER ZEROES IT; WE
# MIRROR THAT AND ADD A ZEROING ENTRY POINT BESIDE IT.
#
# **What theirs does.** `decisiontree.cuh:410-412` is
#
#     for (int i = 0; i < num_outputs; i++) {
#       preds_out[i] += tree.vector_leaf[idx * num_outputs + i];
#     }
#
# `+=`, not `=`. `predict_all` (`decisiontree.cuh:382-392`) does not zero
# `preds` either. The reason is one level up: `RandomForest::predict`
# declares `std::vector<T> row_prediction(forest->trees[0]->num_outputs);`
# INSIDE the row loop (`randomforest.cuh:229`), which value-initialises to
# zero, then calls `DT::DecisionTree::predict` once per tree into that same
# buffer (`randomforest.cuh:231-237`) and divides by `n_trees` afterwards
# (`randomforest.cuh:240-242`). The accumulation across trees IS the
# forest sum, and the only thing that ever zeroes the buffer is
# `std::vector`'s constructor. Hand `predict_all` a dirty buffer and it
# silently adds to garbage.
#
# **What ours does.** `predict_one_accumulate` and `predict_all_accumulate`
# are theirs, byte for byte in behaviour, `+=` and no zeroing.
# `predict_all` (the name a caller reaches for first) zeroes `preds` and
# then calls the accumulating form.
#
# **Why.** The contract "output must already be zero" is invisible in the
# signature, it is documented nowhere in their header, and it is exactly
# the sort of thing this lane's checks cannot see: a check that allocates
# a fresh zero buffer passes under both behaviours. Keeping the faithful
# form under a name that SAYS it accumulates makes the contract legible;
# adding the zeroing form means a single-tree caller cannot get it wrong.
# The forest path must keep calling the accumulating form or the sum is
# lost, and that is why it was not simply changed.
#
# **Price.** Two names where cuML has one, and a lane that ports
# `RandomForest::predict` later must reach for the accumulating one. Stated
# here so that choice is made on purpose rather than by autocomplete.
# ======================================================================


def predict_one_accumulate[
    dtype: DType
](
    row: List[Scalar[dtype]],
    row_offset: Int,
    tree: TreeMetaDataNode[dtype],
    mut preds_out: List[Scalar[dtype]],
    preds_offset: Int,
    num_outputs: Int,
):
    """`predict_one`, `decisiontree.cuh:394-413`. ACCUMULATES (`+=`), and
    does NOT zero `preds_out`. See DEVIATION 147."""
    var idx = predict_leaf(tree, row, row_offset)
    var base = tree.leaf_base(idx)
    for i in range(num_outputs):
        preds_out[preds_offset + i] += tree.vector_leaf[base + i]


def predict_all_accumulate[
    dtype: DType
](
    tree: TreeMetaDataNode[dtype],
    rows: List[Scalar[dtype]],
    n_rows: Int,
    n_cols: Int,
    mut preds: List[Scalar[dtype]],
    num_outputs: Int,
):
    """`predict_all`, `decisiontree.cuh:382-392`. Row-major `rows`, one
    row of `n_cols`; `preds` is `n_rows * num_outputs`. ACCUMULATES; see
    DEVIATION 147."""
    for row_id in range(n_rows):
        predict_one_accumulate(
            rows,
            row_id * n_cols,
            tree,
            preds,
            row_id * num_outputs,
            num_outputs,
        )


def predict_all[
    dtype: DType
](
    tree: TreeMetaDataNode[dtype],
    rows: List[Scalar[dtype]],
    n_rows: Int,
    n_cols: Int,
    mut preds: List[Scalar[dtype]],
    num_outputs: Int,
) raises:
    """`predict_all` with `preds` zeroed first. DEVIATION 147.

    `DecisionTree::predict` (`decisiontree.cuh:359-379`) also asserts the
    tree is non-empty (`:374-376`); that assert is kept here because an
    empty `sparsetree` would index node 0 of nothing.
    """
    if tree.num_nodes() == 0:
        raise Error(
            "Cannot predict w/ empty tree, tree size 0"
        )
    for i in range(n_rows * num_outputs):
        preds[i] = 0
    predict_all_accumulate(tree, rows, n_rows, n_cols, preds, num_outputs)


def predict_regression[
    dtype: DType
](
    tree: TreeMetaDataNode[dtype],
    row: List[Scalar[dtype]],
    row_offset: Int,
) raises -> Scalar[dtype]:
    """The `num_outputs == 1` shape. `RandomForest::predict` takes
    `row_prediction[0]` for `RF_type::REGRESSION`
    (`randomforest.cuh:254-256`); for a single tree the division by
    `n_trees` (`:241`) is a division by one and is dropped."""
    if tree.num_outputs != 1:
        raise Error("predict_regression requires num_outputs == 1")
    var idx = predict_leaf(tree, row, row_offset)
    return tree.leaf_value(idx, 0)


def predict_class[
    dtype: DType
](
    tree: TreeMetaDataNode[dtype],
    row: List[Scalar[dtype]],
    row_offset: Int,
) -> Int:
    """Argmax over the leaf vector, `randomforest.cuh:243-253`.

    Their loop, transcribed:

        L best_class = 0;
        T best_prob  = 0.0;
        for (int k = 0; k < forest->trees[0]->num_outputs; k++) {
          if (row_prediction[k] > best_prob) {
            best_class = k;
            best_prob  = row_prediction[k];
          }
        }

    **HOW THE TIE IS BROKEN, since it was asked.** The comparison is
    STRICTLY greater (`randomforest.cuh:247`) against a running best that
    starts at class 0, so an exact tie between two classes keeps the one
    seen FIRST, i.e. the LOWEST class index wins. That is the whole rule;
    there is no secondary key.

    **And a second consequence of the same two lines that is easy to miss.**
    `best_prob` is initialised to `0.0`, NOT to `-infinity` and NOT to
    `row_prediction[0]` (`randomforest.cuh:245`). So a leaf whose scores are
    all `<= 0` — an all-zero leaf, or any leaf reached in a hypothetical
    signed-score tree — never enters the `if` at all and class 0 is
    returned by default. For cuML that is safe because leaf vectors are
    non-negative class probabilities; it is transcribed rather than
    hardened, and the check covers it with an all-zero leaf.

    The per-tree division by `n_trees` (`randomforest.cuh:240-242`) is
    dropped: it is a positive scalar applied to every class alike and
    cannot move an argmax, and for a single tree it is a division by one.
    """
    var idx = predict_leaf(tree, row, row_offset)
    var base = tree.leaf_base(idx)
    var best_class = 0
    var best_prob: Scalar[dtype] = 0.0
    for k in range(Int(tree.num_outputs)):
        if tree.vector_leaf[base + k] > best_prob:
            best_class = k
            best_prob = tree.vector_leaf[base + k]
    return best_class
