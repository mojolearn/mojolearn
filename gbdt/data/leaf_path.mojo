# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# This lane MIRRORS CatBoost (pinned commit 54a8143a). Per-file provenance is in gbdt/DERIVATION_MAP.tsv, in this file's own docstring, and in NOTICE; files with no CatBoost counterpart say so.
"""The route from the root to one leaf, which is what a non-symmetric tree is.

PORT OF `catboost/cuda/data/leaf_path.h` at CatBoost `54a8143a`.
Transliterated. Do not improve.

An oblivious tree needs no such thing: every leaf of a level shares one split,
so `TObliviousTreeStructure` is a list of splits and a leaf is an index into
the bits. **A Lossguide or Depthwise tree has a different split per node**, so
the only way to say what a leaf IS, is to say how you got there. That is
`TLeafPath`, and it is the unit their whole non-symmetric pipeline passes
around: `TLeaf.Path` carries it while the tree grows
(`split_properties_helper.h`), `SplitLeaf` extends it by one
(`split_properties_helper.cpp:786-798`), `PreviousSplit` shortens it by one to
key a pair of siblings (`:1293`), and `BuildTreeLikeModel` is handed nothing
but a vector of them (`model_builder.cpp:11`).

## Two placements that differ from theirs, both recorded rather than hidden

1. **`ESplitValue` is NOT here.** Theirs is in `cuda/data/feature.h:25-28`
   beside `TBinarySplit`; this repository already carries it as
   `SPLIT_VALUE_ZERO` / `SPLIT_VALUE_ONE` in `gbdt/methods/helpers.mojo:108`,
   and `TBinarySplit` itself sits in `gbdt/models/oblivious_model.mojo`. So
   `feature.h` is already split across two of our files and this one does not
   make it three. A direction reaching a function here is a plain `Int` in
   their numbering; the names live with the constants.

2. **`GetHash` is NOT PORTED, deliberately.** Theirs is
   `MultiHash(VecCityHash(Splits), VecCityHash(Directions))` and its single
   consumer is `THashMap<TLeafPath, TVector<ui32>> rebuildLeaves`
   (`split_properties_helper.cpp:1293`), a grouping whose RESULT is decided by
   equality, not by the hash. A hash with a different distribution cannot move
   a bin, a split or a leaf value. Porting Yandex's `MultiHash` to buy a bucket
   order nothing reads would be inventing work. `paths_equal` below is the
   operation that grouping actually needs, and `split_properties_helper.mojo`
   already keys its own pairing on an integer parent id for the same reason.

   If a future reader needs a stable hash for a FILE FORMAT, that is a
   different requirement and must be ported properly at that point.
"""

from gbdt.models.oblivious_model import TBinarySplit


# =====================================================================
# `TBinarySplit`'s ORDER AND EQUALITY LIVE IN ONE PLACE, AND IT IS NOT HERE
#
# This file carried its own `split_less` / `split_equal` from `e1bfab9`
# until 2026-08-22. `gbdt/methods/batch_feature_tensor_builder.mojo:389`
# already had the same two functions, ported from the same
# `feature.h:50-64` -- AND THE TWO DISAGREED.
#
# Theirs compares through `_as_u32` because CatBoost's `TBinarySplit`
# fields are `ui32` and this port's are `Int32`; that is DEVIATION 116, and
# it is the same argument `helpers.best_split_properties_less` makes about
# the `(ui32)-1` sentinel -- read signed, an undefined split is the
# SMALLEST value there is; read unsigned, the largest. The copy deleted
# here compared `Int32` directly and so sorted the opposite way for any
# field at or above 2^31. Inert on a leaf path, which never holds an
# undefined split, and wrong anyway: a second port that silently reverts a
# numbered deviation is exactly what the deviation ledger exists to stop.
#
# A repo-wide duplication sweep found it, not a gate. Nothing compares two
# comparators.
#
# WHERE THEY BELONG: beside `TBinarySplit` in `models/oblivious_model.mojo`,
# which is what `feature.h` does -- the struct and its operators in one
# file. They are imported from `methods/` instead of moved because that
# file has eight call sites and belongs to another lane; the move is a
# merge-time job and this comment is the note for it.
# =====================================================================

from gbdt.methods.batch_feature_tensor_builder import (
    binary_split_equal as split_equal,
    binary_split_less as split_less,
)


struct TLeafPath(Copyable, Movable):
    """Their `TLeafPath` (`leaf_path.h:17-60`), both vectors, kept parallel.

    Two vectors rather than one vector of pairs, because theirs are two and
    because `PreviousSplit` resizes both by the same amount. The invariant
    `len(splits) == len(directions)` is theirs by construction, and every
    method here asserts it rather than assuming it -- an unequal pair is the
    shape a half-applied `AddSplit` leaves behind.
    """

    var splits: List[TBinarySplit]
    var directions: List[Int]

    def __init__(out self):
        self.splits = List[TBinarySplit]()
        self.directions = List[Int]()

    def get_depth(self) -> Int:
        """Their `GetDepth()` (`:20-22`), which is the split count.

        This is the number `IsTerminalLeaf` compares against `MaxDepth`
        (`greedy_search_helper.cpp:687`) and the number `FindMaxDepth` maxes
        over (`:311-317`), so it is load bearing twice per iteration."""
        return len(self.splits)

    def add_split(mut self, split: TBinarySplit, direction: Int):
        """Their `AddSplit` (`:24-27`). Appends to both, in step."""
        self.splits.append(split)
        self.directions.append(direction)

    def is_sorted(self) -> Bool:
        """Their `IsSorted()` (`:41-48`).

        Their test is `Splits[i] <= Splits[i-1]` returning false, so EQUAL
        adjacent splits count as unsorted -- their "sorted" means strictly
        increasing. Copied with the sense inverted once, not twice."""
        for i in range(1, len(self.splits)):
            if not split_less(self.splits[i - 1], self.splits[i]):
                return False
        return True

    def has_duplicates(self) -> Bool:
        """Their `HasDuplicates()` (`:50-57`).

        **ADJACENT ONLY**, exactly as theirs is. It is a post-sort check, not
        a set test, and calling it on an unsorted path answers a different
        question. `SortUniquePath` is the caller that makes it meaningful
        (`:106`)."""
        for i in range(1, len(self.splits)):
            if split_equal(self.splits[i - 1], self.splits[i]):
                return True
        return False


def paths_equal(a: TLeafPath, b: TLeafPath) -> Bool:
    """Their `TLeafPath::operator==` (`:30-32`), which ties both vectors.

    A free function rather than `__eq__` because the grouping that reads it
    (`PreviousSplit` keys, `model_builder`'s prefix test) reads better spelled
    out, and because `TLeafPath` carries `List`s that Mojo will not compare
    for us."""
    if len(a.splits) != len(b.splits):
        return False
    if len(a.directions) != len(b.directions):
        return False
    for i in range(len(a.splits)):
        if not split_equal(a.splits[i], b.splits[i]):
            return False
        if a.directions[i] != b.directions[i]:
            return False
    return True


def previous_split(path: TLeafPath) raises -> TLeafPath:
    """Their `PreviousSplit` (`:62-69`), the PARENT's path.

    Their `CB_ENSURE(size > 0, "Error: can't remove split")` is kept as a
    raise. It fires on the root, and the root has no parent -- a caller that
    reaches this with an empty path has lost track of which leaves are new,
    which is a bookkeeping bug and not a boundary case to swallow.

    This is the key their sibling pairing groups on
    (`split_properties_helper.cpp:1293`): two siblings differ only in the last
    direction, so dropping the last entry makes them collide, and that
    collision is the entire mechanism behind "build the smaller, derive the
    larger"."""
    var size = path.get_depth()
    if size == 0:
        raise Error("Error: can't remove split")
    var prev = TLeafPath()
    for i in range(size - 1):
        prev.add_split(path.splits[i], path.directions[i])
    return prev^


def split_leaf_path(
    path: TLeafPath, split: TBinarySplit, direction: Int
) -> TLeafPath:
    """The path half of their `SplitLeaf` (`split_properties_helper.cpp:786-798`).

    Their `SplitLeaf` also resets the child's `BestSplit` and demotes
    `CurrentPath` to `PreviousPath`; those live on `TLeaf` and stay with it.
    This is the piece that belongs to the path."""
    var child = path.copy()
    child.add_split(split, direction)
    return child^
