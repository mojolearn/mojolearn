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

**One case they handle that is easy to miss:** if BOTH siblings of a pair are
terminal, neither histogram will ever be read, so neither is built
(`:1326-1328`). Copied. Dropping it would be correct and would waste a build
per terminal pair at the last level, which is where the leaves are.
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

        # `:1318`, the rule the whole design rests on.
        var small = i
        var big = sibling
        if leaves[sibling].size < leaves[i].size:
            small = sibling
            big = i

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
