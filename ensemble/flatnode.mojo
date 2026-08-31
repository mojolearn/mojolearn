# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# This lane MIRRORS cuML's random forest. Per-file provenance is in this file's own docstring and in NOTICE.
"""The flat tree node every cuML forest is made of, and the only shape
inference walks.

MIRRORS `cpp/include/cuml/tree/flatnode.h` at rapidsai/cuml `v26.08.00`
(`265b9da6a0e75dbef071a3168398b993a5ff6f0e`), checked out read-only at
`~/CascadeProjects/upstream/cuml-v26.08.00`. Their `include/cuml/tree/`
prefix drops the same way `cpp/src/` does, so their 66-line file lands
here.

Their whole file is five private fields, one private constructor, seven
accessors, two static factories and an `operator==`. Every one of those
is transcribed below.

Three facts in that file decide what a traversal does, and all three are
easy to get wrong from memory:

  1. **A node is a leaf iff `left_child_id == -1`** (`flatnode.h:58`).
     Not a separate flag, not a sentinel column id -- the child index
     itself. `CreateLeafNode` (`flatnode.h:54-57`) is
     `{0, 0, 0, -1, instance_count}`, i.e. colid 0 and quesval 0 on a
     leaf are MEANINGLESS values that a correct walk never reads.

  2. **The siblings are adjacent**: `RightChildId()` is
     `left_child_id + 1` (`flatnode.h:45`). There is no right-child
     field. A tree is stored as one flat array in which every split
     node's two children sit next to each other, so a builder that
     allocates children in any other order produces a tree this walk
     silently mis-reads.

  3. **The default-constructed node is a LEAF**, because the default
     member initializers (`flatnode.h:25-29`) are
     `colid = 0, quesval = 0, best_metric_val = 0, left_child_id = -1,
     instance_count = 0`. That is what `std::vector<SparseTreeNode>`
     resize hands out, so `Self()` below reproduces it exactly rather
     than zeroing every field.

The comparison direction itself does NOT live in this file. It lives in
`decisiontree.cuh:379` (`row[n.ColumnId()] <= n.QueryValue()` goes
LEFT) and is transcribed in `decisiontree.mojo`, beside the walk, where
a reader will actually look for it.

================= DEVIATION BLOCK (whole file) =================
DEVIATION 116. Three spellings differ from their header. None of them
changes a value, and each is written down because a reader diffing this
file against `flatnode.h` will otherwise stop at it.

(a) THEIRS keeps all five fields `private:` (`flatnode.h:24`) and the
field-taking constructor private with them (`flatnode.h:30-38`), so the
only public way to make a node is `CreateSplitNode` / `CreateLeafNode`.
OURS cannot: Mojo 1.0 has no access-control keyword, so the
field-taking constructor is reachable and the fields are named with a
leading underscore to say what their header says with a keyword.
PRICE: zero on any output. The cost is one class of caller mistake --
constructing a node with a hand-picked `left_child_id` instead of going
through a factory -- that their compiler rejects and ours does not. The
accessor surface (`ColumnId`, `QueryValue`, `BestMetric`,
`LeftChildId`, `RightChildId`, `InstanceCount`, `IsLeaf`) is kept
complete and is what every consumer in this port uses, so the diff
surface against their tree is unchanged.

(b) THEIRS is `template <typename DataT, typename LabelT, typename
IdxT = int>` (`flatnode.h:22`). OURS carries only `dtype`. `LabelT` is
a PHANTOM parameter in their file: it appears in the template list and
inside the factories' return spelling `SparseTreeNode<DataT, LabelT>`
(`flatnode.h:51, 56`) and in NO field and NO method -- read all five
field declarations at `:25-29` and all seven accessors at `:41-46` to
confirm it. `IdxT` is instantiated as `int` everywhere in cuML's RF, so
it is Int32 here rather than a parameter.
PRICE: zero. A phantom type parameter has no runtime and no value
meaning; dropping it removes a parameter nobody can pass wrong. The
same phantom recurs on `TreeMetaDataNode<T, L>` and is recorded again
at its own site (DEVIATION 118).

(c) THEIR OWN WIDTHS ARE ASYMMETRIC and the asymmetry is transcribed,
not corrected. `left_child_id` is STORED as `IdxT` (= `int`, 32-bit,
`flatnode.h:28`), the private constructor RECEIVES it as `int64_t`
(`flatnode.h:31`) and so narrows on the way in, and `LeftChildId()` /
`RightChildId()` RETURN `int64_t` (`flatnode.h:44-45`) and so widen on
the way out. `ColumnId()` and `InstanceCount()` return `IdxT`
(`flatnode.h:41, 46`). Ours does the same: an Int32 field, an Int64
constructor argument that is narrowed explicitly, Int64 child
accessors, Int32 column/count accessors.
PRICE: zero, and keeping it is what has value. A port that "tidied"
these to one width would hide the fact that `RightChildId()` is
computed as `int64_t(int) + 1` and therefore cannot wrap where an Int32
`+ 1` could, and would hide the narrowing a builder passing a >2^31
child index would suffer identically on both sides.
=================================================================
"""


struct SparseTreeNode[dtype: DType](ImplicitlyCopyable, Movable):
    """`SparseTreeNode<DataT, LabelT, IdxT>`, `flatnode.h:22-65`.

    Field names are theirs with a leading underscore standing in for
    their `private:` (DEVIATION 116a). Read through the accessors.
    """

    # `flatnode.h:25` -- IdxT colid = 0;
    var _colid: Int32
    # `flatnode.h:26` -- DataT quesval = DataT(0);
    var _quesval: Scalar[Self.dtype]
    # `flatnode.h:27` -- DataT best_metric_val = DataT(0);
    var _best_metric_val: Scalar[Self.dtype]
    # `flatnode.h:28` -- IdxT left_child_id = -1;  (-1 IS the leaf test)
    var _left_child_id: Int32
    # `flatnode.h:29` -- IdxT instance_count = 0;
    var _instance_count: Int32

    def __init__(out self):
        """Their DEFAULT member initializers (`flatnode.h:25-29`).

        This is a LEAF, not a zeroed struct: `left_child_id` starts at
        -1. `std::vector<SparseTreeNode<T, L>> sparsetree` grows nodes
        in exactly this state, so any node the builder has allocated but
        not yet filled reads as a leaf with zero instances.
        """
        self._colid = 0
        self._quesval = 0
        self._best_metric_val = 0
        self._left_child_id = -1
        self._instance_count = 0

    def __init__(
        out self,
        colid: Int32,
        quesval: Scalar[Self.dtype],
        best_metric_val: Scalar[Self.dtype],
        left_child_id: Int64,
        instance_count: Int32,
    ):
        """Their PRIVATE field constructor (`flatnode.h:30-38`).

        Private in their header; see DEVIATION 116a for why it cannot be
        here. `left_child_id` arrives as Int64 and is narrowed to the
        Int32 field, which is their narrowing (`int64_t` parameter into
        an `IdxT` member), transcribed rather than widened away.
        """
        self._colid = colid
        self._quesval = quesval
        self._best_metric_val = best_metric_val
        self._left_child_id = Int32(left_child_id)
        self._instance_count = instance_count

    # ---- accessors, `flatnode.h:41-46` ----------------------------------

    @always_inline
    def ColumnId(self) -> Int32:
        """`flatnode.h:41`. Returns IdxT."""
        return self._colid

    @always_inline
    def QueryValue(self) -> Scalar[Self.dtype]:
        """`flatnode.h:42`. Returns DataT. The split threshold."""
        return self._quesval

    @always_inline
    def BestMetric(self) -> Scalar[Self.dtype]:
        """`flatnode.h:43`. Returns DataT. Feeds feature importances."""
        return self._best_metric_val

    @always_inline
    def LeftChildId(self) -> Int64:
        """`flatnode.h:44`. Returns int64_t from an IdxT field."""
        return Int64(Int(self._left_child_id))

    @always_inline
    def RightChildId(self) -> Int64:
        """`flatnode.h:45`. `left_child_id + 1`. THE SIBLING RULE.

        There is no right-child field. Siblings are adjacent in the flat
        array, and the arithmetic is done at int64_t width because the
        accessor's return type is int64_t.
        """
        return Int64(Int(self._left_child_id)) + 1

    @always_inline
    def InstanceCount(self) -> Int32:
        """`flatnode.h:46`. Returns IdxT."""
        return self._instance_count

    @always_inline
    def IsLeaf(self) -> Bool:
        """`flatnode.h:58`. `left_child_id == -1`. THE LEAF TEST.

        Compared at the FIELD's width (IdxT), which is what their
        `left_child_id == -1` does -- not against `LeftChildId()`.
        """
        return self._left_child_id == -1

    # ---- factories, `flatnode.h:48-57` ----------------------------------

    @staticmethod
    def CreateSplitNode(
        colid: Int32,
        quesval: Scalar[Self.dtype],
        best_metric_val: Scalar[Self.dtype],
        left_child_id: Int64,
        instance_count: Int32,
    ) -> Self:
        """`flatnode.h:48-53`."""
        return Self(
            colid, quesval, best_metric_val, left_child_id, instance_count
        )

    @staticmethod
    def CreateLeafNode(instance_count: Int32) -> Self:
        """`flatnode.h:54-57` -- `{0, 0, 0, -1, instance_count}`.

        colid 0 and quesval 0 on a leaf are placeholders their own
        factory writes; a correct walk stops before reading either.
        """
        return Self(0, 0, 0, -1, instance_count)

    # ---- equality, `flatnode.h:59-64` -----------------------------------

    def __eq__(self, other: Self) -> Bool:
        """`flatnode.h:59-64`. All five fields, including the two that
        are placeholders on a leaf -- so two leaves built by
        `CreateLeafNode` with equal instance counts compare equal, and a
        leaf whose placeholder fields were left over from a split does
        NOT. That is their behaviour."""
        return (
            (self._colid == other._colid)
            and (self._quesval == other._quesval)
            and (self._best_metric_val == other._best_metric_val)
            and (self._left_child_id == other._left_child_id)
            and (self._instance_count == other._instance_count)
        )

    def __ne__(self, other: Self) -> Bool:
        """Not in their file; C++ synthesizes it from `operator==`."""
        return not (self == other)
