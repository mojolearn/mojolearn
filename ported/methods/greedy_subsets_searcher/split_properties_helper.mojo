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
struct LeafRecord(Copyable, Movable):
    """One leaf of the level, as the pairing rule needs to see it.

    `path_id` stands in for `TLeafPath` as a hash key: two siblings share it,
    which is exactly what `THashMap<TLeafPath, TVector<ui32>> rebuildLeaves`
    is grouping on (`:1293`).
    """

    var size: UInt32
    var histograms_type: Int
    var path_id: Int
    var is_terminal: Bool


@fieldwise_init
struct LevelPlan(Copyable, Movable):
    """The output: what to launch this level.

    - `compute_ids`: leaves whose histogram is accumulated from their rows.
    - `subtract_from`, `subtract_what`: parallel arrays, one entry per pair,
      feeding the batched subtraction. `from - what` overwrites `from`.
    """

    var compute_ids: List[UInt32]
    var subtract_from: List[UInt32]
    var subtract_what: List[UInt32]

    def builds_saved(self) -> Int:
        """How many accumulations the subtraction avoided. One per pair."""
        return len(self.subtract_from)


def build_necessary_histograms(leaves: List[LeafRecord]) raises -> LevelPlan:
    """`BuildNecessaryHistograms`'s classification, copied.

    Their loop shape: first bucket every leaf by `HistogramsType`, grouping
    the rebuild candidates by path; then walk the groups and, for a group of
    two, pick the smaller to compute and record the pair for subtraction.
    """
    var compute_ids = List[UInt32]()
    var subtract_from = List[UInt32]()
    var subtract_what = List[UInt32]()

    # `for (size_t i = 0; i < leaves.size(); ++i)` at `:1295`. A
    # `PreviousPath` leaf needs nothing; a `Zeroes` leaf is a rebuild
    # candidate and is grouped by path.
    var n = len(leaves)
    var seen = List[Bool]()
    for _ in range(n):
        seen.append(False)

    for i in range(n):
        if seen[i]:
            continue
        if leaves[i].histograms_type == HISTOGRAMS_PREVIOUS_PATH:
            seen[i] = True
            continue
        if leaves[i].histograms_type != HISTOGRAMS_ZEROES:
            seen[i] = True
            continue

        # Find this leaf's sibling: the other member of its path group.
        var sibling = -1
        for j in range(i + 1, n):
            if (
                not seen[j]
                and leaves[j].histograms_type == HISTOGRAMS_ZEROES
                and leaves[j].path_id == leaves[i].path_id
            ):
                sibling = j
                break

        seen[i] = True
        if sibling < 0:
            # `if (ids.size() == 1)` at `:1309`: no sibling, build it.
            compute_ids.append(UInt32(i))
            continue

        seen[sibling] = True

        # `:1326-1328`. Both terminal means neither histogram is ever read.
        if leaves[i].is_terminal and leaves[sibling].is_terminal:
            continue

        # `:1318`, the rule the whole design rests on.
        var small = i
        var big = sibling
        if leaves[sibling].size < leaves[i].size:
            small = sibling
            big = i

        compute_ids.append(UInt32(small))
        # `parent - smaller = larger`, and the parent's histogram already
        # occupies the larger's slot, so `from` is the big leaf.
        subtract_from.append(UInt32(big))
        subtract_what.append(UInt32(small))

    return LevelPlan(compute_ids^, subtract_from^, subtract_what^)
