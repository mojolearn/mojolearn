# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# This lane MIRRORS CatBoost (pinned commit 54a8143a). Per-file provenance is in gbdt/DERIVATION_MAP.tsv, in this file's own docstring, and in NOTICE; files with no CatBoost counterpart say so.
"""Which histograms to BUILD and which to DERIVE, once per level.

PORT OF `TSplitPropertiesHelper::BuildNecessaryHistograms` in
`catboost/cuda/methods/greedy_subsets_searcher/split_properties_helper.cpp`
at CatBoost `54a8143a`. Transliterated. Do not improve.

This is the decision that halves a level's histogram work, and it is pure
host-side bookkeeping: no kernel, no device memory, just which leaf ids go
into which of three buckets.

    enum EHistogramsType { Zeroes, PreviousPath, CurrentPath };

Each leaf of the new level carries one of those. A leaf whose histogram is
still valid from the parent path needs nothing. A leaf marked `Zeroes` needs
a fresh build. And a PAIR of siblings needs exactly ONE build between them:
build the SMALLER, derive the larger as `parent - smaller`.

`if (firstLeaf.Size < secondLeaf.Size)` is the whole rule
(`split_properties_helper.cpp:1318`), and the bound it buys is that a level
never accumulates more than N/2 rows per feature however unbalanced the tree
is. The derivation itself is `histogram_utils.substract_histograms_kernel`,
one batched launch over every pair at once.

**THE TIE GOES TO THE RIGHT CHILD, and it is a real decision, not a
formality.** `ids[0]` is the LEFT child (`MakeSplit` gives it the existing,
lower id: `:861-862`, `:976-977`) and the comparison is a strict `<` on it,
so an equal-sized pair falls through to the `else` and the RIGHT child is the
one built. Equal sibling sizes are the common case on a binary feature -- 632
of 744 planned pairs on the balanced fixture in
`original/sibling_tiebreak_check.mojo` -- and the choice is NOT inert: the
subtraction runs in float32 on cells that have already been rounded out of
float32's exact-integer range, so swapping it moves histogram bits. PORTING.md
136.

**THIS HOST COPY IS NOT ON THE SHIPPED PATH.** DEVIATION 94 moved the choice
onto the device (`kernel/split_resolve.plan_level_kernel`); only `probe_main`
and the checks reach this function. It is kept in step with the device twin
so the two cannot drift.

**One case they handle that is easy to miss:** if BOTH siblings of a pair are
terminal, neither histogram will ever be read, so neither is built
(`:1326-1328`). Copied. Dropping it would be correct and would waste a build
per terminal pair at the last level, which is where the leaves are.

**A second case, and it is the one this port had wrong:** `computeLeaves` is
split ONE MORE TIME before anything launches, on `leaf.Size != 0`
(`:1342-1346`). An empty leaf's histogram is zeros by definition, so CatBoost
zeroes the slot instead of building it. See `non_zero_leaves` and
`zero_leaves` at the foot of this file.
"""


comptime HISTOGRAMS_ZEROES = 0
comptime HISTOGRAMS_PREVIOUS_PATH = 1
comptime HISTOGRAMS_CURRENT_PATH = 2


@fieldwise_init
struct LeafRecord(Copyable, ImplicitlyCopyable, Movable):
    """One leaf of the level, as the pairing rule needs to see it.

    `path_id` is their `PreviousSplit(leaf.Path)`, the PARENT's path, which
    is the key `THashMap<TLeafPath, TVector<ui32>> rebuildLeaves` groups on
    (`:1293`). Two siblings share it; that is the whole point of the key.
    """

    var size: UInt32
    var histograms_type: Int
    var path_id: Int
    var is_terminal: Bool


@fieldwise_init
struct LevelPlan(Copyable, Movable):
    """The output: what to launch this level.

    Their three vectors, same names, same meaning (`:1287-1291`):

    - `compute_ids`  = `computeLeaves`, accumulated from their rows
    - `subtract_what` = `smallLeaves`, a strict subset of `computeLeaves`
    - `subtract_from` = `bigLeaves`, derived as `parent - small` in place

    `updated_ids` is their `allUpdatedLeaves` (`:1359`), which is
    `computeLeaves` followed by `bigLeaves`, and is exactly the set that
    becomes `CurrentPath` once the level is built.
    """

    var compute_ids: List[UInt32]
    var subtract_from: List[UInt32]
    var subtract_what: List[UInt32]

    def builds_saved(self) -> Int:
        """How many accumulations the subtraction avoided. One per pair."""
        return len(self.subtract_from)

    def updated_ids(self) -> List[UInt32]:
        """Their `allUpdatedLeaves` (`:1359`)."""
        var out = List[UInt32]()
        for i in range(len(self.compute_ids)):
            out.append(self.compute_ids[i])
        for i in range(len(self.subtract_from)):
            out.append(self.subtract_from[i])
        return out^


def build_necessary_histograms(leaves: List[LeafRecord]) raises -> LevelPlan:
    """`BuildNecessaryHistograms`'s classification, copied.

    ==================== CORRECTED 2026-08-19 ====================
    The first port had this state machine BACKWARDS, and it was never
    wired, so nothing caught it. It skipped `PreviousPath` as "needs
    nothing" and paired up `Zeroes` leaves for subtraction. Theirs
    (`:1295-1304`) is the exact opposite:

        if (leaf.HistogramsType == EHistogramsType::PreviousPath) {
            auto prevPath = PreviousSplit(leaf.Path);
            rebuildLeaves[prevPath].push_back(i);
        } else if (leaf.HistogramsType == EHistogramsType::Zeroes) {
            computeLeaves.push_back(i);
        }

    A `PreviousPath` leaf is one whose slot ALREADY HOLDS ITS PARENT'S
    histogram, put there by `CopyHistogram` at split time
    (`split_points.cpp:326`). Those are precisely the leaves that can be
    paired and subtracted. A `Zeroes` leaf holds nothing and must be built.

    Wiring the old version would have silently subtracted the wrong
    histograms. See PORTING_RULES.md rule 3.
    ==============================================================
    """
    var compute_ids = List[UInt32]()
    var subtract_from = List[UInt32]()
    var subtract_what = List[UInt32]()

    var n = len(leaves)

    # `for (size_t i = 0; i < leaves.size(); ++i)` at `:1295`.
    # Zeroes goes straight to computeLeaves; PreviousPath is grouped by the
    # PARENT path so siblings meet.
    var grouped = List[Bool]()
    for _ in range(n):
        grouped.append(False)

    for i in range(n):
        if leaves[i].histograms_type == HISTOGRAMS_ZEROES:
            compute_ids.append(UInt32(i))

    # `for (auto& rebuildLeavesPair : rebuildLeaves)` at `:1306`.
    for i in range(n):
        if grouped[i]:
            continue
        if leaves[i].histograms_type != HISTOGRAMS_PREVIOUS_PATH:
            continue
        grouped[i] = True

        var sibling = -1
        for j in range(i + 1, n):
            if (
                not grouped[j]
                and leaves[j].histograms_type == HISTOGRAMS_PREVIOUS_PATH
                and leaves[j].path_id == leaves[i].path_id
            ):
                sibling = j
                break

        if sibling < 0:
            # `CB_ENSURE(subsets->Leaves[leafId].IsTerminal, ...)` at `:1311`.
            if not leaves[i].is_terminal:
                raise Error(
                    String("Error: this leaf should be terminal, id ")
                    + String(i)
                )
            continue

        grouped[sibling] = True

        # `:1318`, the rule the whole design rests on:
        #
        #     if (firstLeaf.Size < secondLeaf.Size) { small = ids[0]; ... }
        #     else                                  { small = ids[1]; ... }
        #
        # `ids` is pushed in ASCENDING leaf index (`:1300`), so `ids[0]` is
        # `i` and `ids[1]` is `sibling`, the strict `<` is on `i`, and ON AN
        # EXACT TIE THE `else` BRANCH FIRES AND `sibling` IS COMPUTED. This
        # port had it inverted from `409a16c` until 2026-08-21; PORTING.md
        # 136 has what that cost.
        var small = sibling
        var big = i
        if leaves[i].size < leaves[sibling].size:
            small = i
            big = sibling

        # `:1327`. Both terminal means neither histogram is ever read.
        if leaves[small].is_terminal and leaves[big].is_terminal:
            continue

        # `smallLeaves.push_back(smallLeafId);`
        # `computeLeaves.push_back(smallLeafId);`
        # `bigLeaves.push_back(bigLeafId);`  (`:1332-1334`)
        subtract_what.append(UInt32(small))
        compute_ids.append(UInt32(small))
        subtract_from.append(UInt32(big))

    return LevelPlan(compute_ids^, subtract_from^, subtract_what^)


def non_zero_leaves(
    leaves: List[LeafRecord], ids: List[UInt32]
) -> List[UInt32]:
    """`NonZeroLeaves`, copied (`split_properties_helper.cpp:762-773`).

        for (const auto leafId : leaves) {
            auto& leaf = subsets.Leaves.at(leafId);
            if (leaf.Size != 0) {
                result.push_back(leafId);
            }
        }

    THE POINT, so the next reader does not undo it. **An EMPTY leaf's
    histogram is all zeros by definition, so CatBoost never builds it.** It
    zeroes the slot and moves on. `BuildNecessaryHistograms` splits
    `computeLeaves` in two right before it launches anything (`:1342-1346`):

        auto nonZeroComputeLeaves = NonZeroLeaves(*subsets, computeLeaves);
        auto zeroLeaves           = ZeroLeaves(*subsets, computeLeaves);
        ComputeSplitProperties(loadPolicy, nonZeroComputeLeaves, subsets);
        ZeroLeavesHistograms(zeroLeaves, subsets);

    Feeding one undifferentiated list to the histogram builder costs a full
    accumulation over every row and every feature to produce a result that
    was known in advance. At depth 6 this port's own checks report 56 of 64
    leaves non-empty on one fixture and 46 of 64 on another, so that is 8 to
    18 wasted launches per level on the fixtures already in the tree, and
    far more on sparse data.

    `leaf.Size` has to be right for the partition to be safe. It is
    maintained by `RebuildLeavesSizes` (`:800-812`) after every split, and
    ours is filled from the same partition sizes at the same point, at the
    drain at the foot of the level loop in `run_tree_layout`.
    """
    var result = List[UInt32]()
    for i in range(len(ids)):
        if leaves[Int(ids[i])].size != UInt32(0):
            result.append(ids[i])
    return result^


def zero_leaves(leaves: List[LeafRecord], ids: List[UInt32]) -> List[UInt32]:
    """`ZeroLeaves`, copied (`split_properties_helper.cpp:775-784`).

    The complement of `non_zero_leaves` over the same list. Their two
    helpers are separate functions over the same input rather than one pass
    returning both, and this keeps that shape.
    """
    var result = List[UInt32]()
    for i in range(len(ids)):
        if leaves[Int(ids[i])].size == UInt32(0):
            result.append(ids[i])
    return result^
