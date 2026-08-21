"""One gate over both halves of the pointwise port:
`gbdt/methods/pointwise_optimization_subsets.mojo` and
`gbdt/methods/helpers.mojo`.

WHAT IT IS STANDING IN FOR
---------------------------
Neither file has a caller. The searcher above them
(`oblivious_tree_doc_parallel_structure_searcher.{h,cpp}`) and the
`pointwise_hist2*` family below them are unported, so `PORTING_RULES.md` rule
3 applies with full force: `build_necessary_histograms` sat in this tree
"fully written, commented, tested by a probe, and with its state machine
exactly backwards" because nothing called it. This file is the caller until
there is one.

THE FIVE RULES THIS FIXTURE IS BUILT AROUND
--------------------------------------------
1. **PER CELL, NEVER A TOTAL.** Every gate below counts disagreeing CELLS --
   positions, partitions, (partition, stat) pairs, (plane, position) pairs --
   and names the first one. This repository has twice shipped a check that
   verified a sum and missed a permutation.

2. **HASHED AND DISTINCT.** Feature values, weights and targets are all
   splitmix64-derived per document, so no two documents look alike and a
   misplacement lands somewhere visible. Weights are integers in `[1, 63]`
   and targets integers in `[-63, 63]`; at 2003 documents the largest
   possible partition sum is 126,189, so every intermediate sum on the device
   and on the host is an exact Float32 integer and the comparison is EQUALITY
   with no tolerance anywhere.

3. **STABILITY IS GATED SEPARATELY FROM CORRECTNESS**, because
   `ReorderBins(bins, indices, depth, 1)` is a ONE-BIT sort whose whole job
   depends on ties keeping their order. Each level reports two independent
   numbers side by side:

       membership wrong: 0     <- which partition each document is in
       order wrong:      1442  <- where inside that partition it sits

   A sort that reverses each group scores 0 on the first and hundreds on the
   second, and every partition offset, partition size and partition stat stays
   BIT-IDENTICAL. That is the failure a naive check cannot see, and the
   sabotage section below demonstrates it rather than asserting it.

4. **EVERY MECHANISM IS MADE OBSERVABLE, not merely reached.** Five levels are
   run and three of them exist to hit a branch the other two cannot:

       L0  FEAT_A  bin 7    mixed, `>` predicate
       L1  FEAT_C  bin 3    mixed, ONE-HOT `==` predicate, second cindex
                            column (non-zero `feature.Offset`)
       L2  FEAT_B  bin 6    mixed, shift 8 inside the SAME word as FEAT_C
       L3  FEAT_A  bin 15   ALL LEFT -- no document goes right, so the upper
                            half of the level is empty and only
                            `UpdatePartitionSizes`'s trailing zero walk fills
                            it
       L4  FEAT_B  bin 0    ALL RIGHT -- every document goes right, so
                            partition 0 and its whole neighbourhood are empty
                            and `UpdatePartitionSizes`'s `i ? bins[i-1] : 0`
                            sentinel is the only thing that sizes them

   L3 and L4 are why the sentinel asymmetry between their offsets kernel
   (`UINT32_MAX`) and their sizes kernel (`0`) is checkable at all. A fixture
   where every level splits somewhere near the middle runs both walks zero
   times.

5. **THE LAYOUT IS THE ONE THE REST OF THE POINTWISE STACK ALREADY GATES.**
   `partitions[2 * p + 0]` = Offset, `[2 * p + 1]` = Size (`TDataPartition[]`
   reinterpreted); `partition_stats[3 * p + {0, 1, 2}]` = `{Weight, Sum,
   Count}` (`TPartitionStatistics`). Every read below is at the record
   stride, poison included -- a check that read a stride-2 record out of a
   stride-3 buffer would find plausible numbers in the wrong cells, which is
   the failure a layout change invites.

   Plane 2, `Count`, is gated against the HOST's document count and not
   against the device's own `partitions[2p+1]`, which would be comparing the
   port to itself. It is gated even though nothing reads it: their scorer
   touches `.Weight`/`.Sum` only (`methods/kernel/pointwise_scores.cu:89`,
   `:92`, `:260`, `:263`, `:359`, `:362`) and so does the ported one
   (`gbdt/methods/kernel/pointwise_scores.mojo:700-703`, `:884-885`,
   `:1011-1014`). A wrong plane 2 is the cheapest evidence that the record is
   being written at the wrong stride.

6. **THE TAIL IS POISONED.** Partitions and stats are allocated at
   `1 << maxDepth` slots and only `1 << (CurrentDepth + FoldBits)` of them are
   live. After every level the dead tail is filled with poison from the host
   and re-read after the NEXT level; anything that sized a launch from
   `max_part_count` instead of `current_part_count()` scribbles on it.

WHAT THIS FILE DELIBERATELY DOES NOT GATE
------------------------------------------
* **The DIGITS of a rendered border.** `split_condition_to_string` formats a
  float, and DEVIATION 99.3 records that Mojo's formatter and C++'s
  `ostream` disagree on how many. The expected strings below are built with
  the same formatter and an INDEPENDENTLY COMPUTED border index, so the gate
  is on which border and which comparator -- which is what a wrong nan arm
  gets wrong -- and not on the text of the number.
* **Float32 versus their `double` partition stats.** DEVIATION 97.4. The
  plants are chosen to make the question not arise; that is a sidestep, not
  an answer, and the deviation block says so.
* **The FoldBits path.** `fold_bits` is 0 on this specialization, so every
  `current_depth + fold_bits` below is observationally `current_depth`.
  Replacing one with the other is INERT here, and the sabotage log below
  reports that as a finding rather than hiding it.

SABOTAGE LOG -- run 2026-08-21, each break applied to the library, measured,
and reverted. Entries marked (re-run) were re-measured after the partition
record changed from parallel arrays to interleaved `TDataPartition[]` and the
stat record from stride 2 to stride 3
-----------------------------------------------------------------------------
A gate that cannot fail is not a gate, so every mechanism was broken on
purpose and the movement recorded. `CONTENT` and `ORDER` are the two counts
this file prints per level; the numbers are cells, not levels.

| # | what was broken | where | result |
|---|---|---|---|
| 1 | `bins[i] /= split << depth` becomes `bins[i] = split << depth` | `update_bins_from_compressed_index_kernel` | RED: 3294 CONTENT + 3190 ORDER. **L0 stayed green** -- at depth 0 there are no lower bits, so `=` and `/=` agree and a one-level fixture would have missed it entirely |
| 2 | `++CurrentDepth` moved AFTER `UpdateSubsetsStats` | `split_subsets` | RED at L0: the stats of the new level's upper half come back POISONED, and by L3 the partition array is unreadable |
| 3 (re-run x2) | `ReorderBins` made UNSTABLE -- every group reversed, same multiset | `reorder_one_bit_u32_kernel` (`gbdt/gpu_util/kernel/radix_sort.mojo`) | **THE HEADLINE, and UNCHANGED by either the layout move or the buffer split.** At L0: `CONTENT wrong 0 ( membership 0 offset 0 size 0 stats 0 count 0 tail 0 ) / ORDER wrong 5958 ( indices 2002 gathered 3956 [w 1966 t 1990] )`. Every partition offset, every partition size, every per-partition stat and every document's partition membership are BIT-IDENTICAL, and 5958 cells of order are wrong. The `[w .. t ..]` split is the proof that the tally still covers BOTH columns after they became separate allocations -- 1966 and 1990, not 3956 and 0 |
| 4 (re-run) | `GatherTarget` moves 1 column instead of 2 -- on two buffers this is "the second `Gather` call is deleted" | `update_subsets_stats` | RED, and **a different defect than it was on one buffer**: 1992 ORDER + 1-to-8 CONTENT per level, where the merged-buffer form gave 11952 ORDER + 31 CONTENT. Two things the new breakdown shows that the old number could not. (a) `[w 0 t 1992]` -- the damage is entirely in the gradient column and the weight column is untouched, which is what a dropped second call looks like and is NOT what a dropped second plane looked like. (b) 1992, not 2003: the un-gathered buffer comes back all zeros, so the 11 documents whose planted target is exactly 0 MATCH BY ACCIDENT. A boolean "did the gather work" would read those 11 cells as fine; a per-cell count prices them. The stat side moves too, `plane Sum : device 0.0 host 67.0` |
| 5 (re-run) | `UpdatePartitionDimensions` sized from `max_part_count` instead of `CurrentPartsView` | `update_subsets_stats` | RED: 196 cells, **all of them `tail`**, ORDER 0, everything else 0. Per level 60/56/48/32/0. L4 green because there `current_part_count() == max_part_count`. The count is the same as before the layout move for a reason worth stating: only the PARTITION records are clobbered (2 cells per dead slot), not the stat records, so widening the stat record from 2 to 3 does not widen this number |
| 6 (re-run) | `UpdatePartitionSizes`'s `i ? bins[i-1] : 0` sentinel changed to `UINT32_MAX` (the value the OFFSETS kernel uses) | `update_partition_sizes_kernel` | RED at **L4 only**: 33 CONTENT (membership 1, size 8, stats 16, count 8). L0-L3 green, because the sentinel is only observable when the first sorted bin is above 0 -- which is exactly why the L4 "ALL RIGHT" arm exists |
| 7 | every `current_depth + fold_bits` replaced by `current_depth` | `split_subsets`, `current_part_count` | **INERT -- 0 cells moved, and that is a finding, not a failure.** `FoldBits` is 0 on the `TStripeMapping` specialization, so the two spellings are the same number here. Nothing in this file can gate the fold path; only the unported `TMirrorMapping` specialization can, and until it lands the `+ fold_bits` spelling rests on their source (`pointwise_optimization_subsets.cpp:41`, `:46`) and not on a measurement |
| 8 (re-run) | `UpdatePartitionSizes` launched BEFORE `UpdatePartitionOffsets` | `launch_update_partition_dimensions` | RED at L0: partition 1 comes back as `[1012, +559040740)`. Sizes are a DIFFERENCE against the offsets, so the order is not cosmetic. **`CreateSubsets` at depth 0 stayed green** -- one partition whose offset is already 0 -- another branch a one-level fixture cannot reach |
| 9 | `ToSplit`'s categorical clamp "tidied" from `GetBinCount(f)` to `GetBinCount(f) - 1` | `to_split` | RED: `cat arm capped at 15 want 16` |
| 10 | `operator<` compares `FeatureId`/`BinId` as SIGNED `Int32` | `best_split_properties_less` | RED: an undefined candidate (`FeatureId == -1`) beat a real one on a gain tie, both argument orders |
| 11 | `operator<` orders by `Score` instead of `Gain` | `best_split_properties_less` | RED: 5 of the named cases, including the strictness one |
| 12 | `ENanMode::Min` reads `borders[BinIdx]` instead of `borders[BinIdx - 1]` | `split_condition_to_string` | RED: `Min, bin 2 gave >2.5 want >1.5` |
| 13 | `HasPermutationDependentSplit` ANDs its two predicates instead of nesting them | `has_permutation_dependent_split` | RED: a plain float column carrying a stale `IsPermutationDependent` answer tripped it |
| 14 | `MergeBits` drops the `y` half of its eighth term | `merge_bits` | RED: 32768 of 65536 pairs on the independent oracle AND 32768 on the `get_odd_bits` round trip; `get_even_bits` stayed at 0, which is the round trip proving it ties the three functions together and not to one shared transcription |
| 15 | `TPartitionStatistics` packed at stride 2 instead of 3 | `pack_partition_stats_kernel` | RED: 2/6/14/24/42 `stats` cells per level, growing with the partition count exactly as a stride error should. `count` stayed 0, because plane 2 was still written at the right place -- so the two halves of the record are independently placed and independently gated |
| 18 (NEW) | the two `compute_partition_stats` calls write EACH OTHER's output buffer | `update_subsets_stats` | RED: 2/4/8/16/32 `stats` cells, ORDER 0, everything else 0 -- a pure stat-PLACEMENT defect with the reduction itself perfectly correct. Only possible to write once the reduce split into two calls, and it is the failure that split invites: both calls are otherwise identical and differ only in which buffer they read and which they write. `plane Weight : device 67.0 host 64395.0` names it in one line, the gradient total sitting in the weight slot |
| 16 | `TDataPartition`'s two `ui32` swapped -- `PART_OFFSET = 1`, `PART_SIZE = 0` | the constants | **INERT ON THE FIRST ATTEMPT, AND THAT WAS A HOLE IN THIS FILE.** Every read here imported the library's own `PART_OFFSET`/`PART_SIZE`, so the swap moved writer and reader together and 0 cells changed. Fixed by `check_layout_contract`, which pins the record with `comptime assert` against LITERALS taken from the call sites that hardcode it. Re-run: **the check no longer COMPILES**, and the assertion names the constant and the file that disagrees. A contract is the one thing worth failing at build time rather than on a fixture |
| 17 | `TPartitionStatistics`'s `Weight` and `Sum` swapped | the constants | RED via `check_layout_contract`: does not compile, naming `TPartitionStatistics::Weight is first`. Added at the same time as 16 and for the same reason |

Four of the eighteen moved NOTHING where it mattered. 1 and 8 are inert at
depth 0; 7 is inert everywhere and says something true about `FoldBits`; **16
was inert because of a defect in THIS FILE, not in the port.** All four are
recorded rather than dropped. "Reached but inert" is the failure mode this
repository keeps re-learning, and 16 is its nastier cousin: a gate that reads
the layout through the library's own names cannot see the layout move.
"""

from max.gpu.host import DeviceContext, HostBuffer

from gbdt.gpu_util.kernel.transform import launch_gather_with_mask_u32
from gbdt.methods.greedy_subsets_searcher.points_subsets import (
    TBestSplitProperties,
)
from gbdt.methods.helpers import (
    SPLIT_VALUE_ONE,
    SPLIT_VALUE_ZERO,
    best_score_message,
    best_split_properties_less,
    get_even_bits,
    get_odd_bits,
    has_permutation_dependent_split,
    merge_bits,
    print_best_score,
    reverse_bits,
    split_condition_to_string,
    split_condition_to_string_value,
    take_best,
    take_best3,
    to_split,
)
from gbdt.gpu_util.kernel.fill import launch_make_sequence
from gbdt.methods.pointwise_optimization_subsets import (
    L2_PLANE_TARGET,
    L2_PLANE_WEIGHT,
    PART_OFFSET,
    PART_SIZE,
    PART_STAT_COUNT,
    PART_STAT_SUM,
    PART_STAT_WEIGHT,
    PARTITION_RECORD,
    PARTITION_STAT_STRIDE,
    POINTWISE_STAT_COUNT,
    TL2Target,
    TOptimizationSubsets,
    create_subsets,
    split_subsets,
)
from gbdt.models.oblivious_model import (
    BIN_SPLIT_TAKE_BIN,
    BIN_SPLIT_TAKE_GREATER,
    TBinarySplit,
    TObliviousTreeStructure,
)
from gbdt.options.data_processing_options import (
    NAN_MODE_FORBIDDEN,
    NAN_MODE_MAX,
    NAN_MODE_MIN,
)


#: Prime, and a ragged multiple of neither the 256-wide split block, the
#: 512-wide reorder block, nor the 512-wide stats block.
comptime N_DOCS = 2003

#: `1 << maxDepth` partition slots; five levels are run, so the last one is
#: exactly full and there is a live tail at every earlier level.
comptime MAX_DEPTH = 5

#: Written into the dead tail of `partitions_*` and `partition_stats` after
#: every level. Far outside any legal offset, size or stat.
comptime POISON_U32 = UInt32(0xDEADBEEF)
comptime POISON_F32 = Float32(-987654.0)

# --- the synthetic compressed index --------------------------------------
#
# Two `ui32` columns of `N_DOCS` words. `feature.Offset` selects the column,
# `feature.Shift` and `feature.Mask` select the bits inside the word, which
# is exactly how `TCFeature` addresses a real one
# (`gbdt/gpu_data/gpu_structures.mojo`, `grid_policy.mojo`).
#
#   column 0:  bits [0, 4)   FEAT_A, 16 bins, `>`
#   column 1:  bits [4, 7)   FEAT_C,  8 bins, one-hot `==`
#              bits [8, 12)  FEAT_B, 16 bins, `>`   -- SAME WORD as FEAT_C
#
# FEAT_B and FEAT_C sharing a word is the case where a wrong `mask << shift`
# silently reads the neighbour's bits, and it is the normal case in a real
# compressed index. FEAT_A sits in a different column so a dropped
# `feature.Offset` is visible too.

comptime FEAT_A_OFFSET = 0
comptime FEAT_A_SHIFT = UInt32(0)
comptime FEAT_A_MASK = UInt32(0xF)

comptime FEAT_B_OFFSET = N_DOCS
comptime FEAT_B_SHIFT = UInt32(8)
comptime FEAT_B_MASK = UInt32(0xF)

comptime FEAT_C_OFFSET = N_DOCS
comptime FEAT_C_SHIFT = UInt32(4)
comptime FEAT_C_MASK = UInt32(0x7)


def _mix(i: Int, salt: Int) -> UInt64:
    """splitmix64. Adjacent documents land nowhere near each other, which is
    what makes a one-position misplacement visible in the cell it lands in
    rather than only in a count."""
    var z = UInt64(i + 1) * 0x9E3779B97F4A7C15 + UInt64(
        salt + 1
    ) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _val_a(doc: Int) -> Int:
    """FEAT_A in `[0, 15]`. Reaches 0, so `> 0` is a real question."""
    return Int(_mix(doc, 11) % 16)


def _val_b(doc: Int) -> Int:
    """FEAT_B in `[1, 15]`. **NEVER 0**, which is what makes level 4
    (`FEAT_B > 0`) send EVERY document right and leave the whole lower half
    of the partition array empty."""
    return 1 + Int(_mix(doc, 22) % 15)


def _val_c(doc: Int) -> Int:
    """FEAT_C in `[0, 7]`, read with the one-hot `==` predicate."""
    return Int(_mix(doc, 33) % 8)


def _weight(doc: Int) -> Int:
    """`TPartitionStatistics::Weight`'s column, integer in `[1, 63]`.
    Strictly positive, which is what a weight column looks like and is the
    plane a broken reduction is likeliest to still get right."""
    return 1 + Int(_mix(doc, 44) % 63)


def _target(doc: Int) -> Int:
    """`TPartitionStatistics::Sum`'s column, integer in `[-63, 63]`. Straddles
    zero, so terms cancel and a dropped one does not simply shrink the
    total."""
    return Int(_mix(doc, 55) % 127) - 63


def _feature_val(doc: Int, feat: Int) -> Int:
    if feat == 0:
        return _val_a(doc)
    elif feat == 1:
        return _val_b(doc)
    else:
        return _val_c(doc)


def _goes_right(doc: Int, feat: Int, bin_idx: Int) -> Bool:
    """The host's copy of the kernel's predicate, written from the ALGORITHM
    (compare the feature's value against the bin) and not from the kernel's
    masked-word arithmetic, so the two cannot share a bug in the shift."""
    var v = _feature_val(doc, feat)
    if feat == 2:
        return v == bin_idx
    return v > bin_idx


struct Level(Copyable, ImplicitlyCopyable, Movable):
    var feat: Int
    var bin_idx: Int
    var name: String

    def __init__(out self, feat: Int, bin_idx: Int, var name: String):
        self.feat = feat
        self.bin_idx = bin_idx
        self.name = name^


def _levels() -> List[Level]:
    var out = List[Level]()
    out.append(Level(0, 7, String("FEAT_A > 7, mixed")))
    out.append(Level(2, 3, String("FEAT_C == 3, ONE-HOT, second column")))
    out.append(Level(1, 6, String("FEAT_B > 6, shift 8 in FEAT_C's word")))
    out.append(Level(0, 15, String("FEAT_A > 15, ALL LEFT")))
    out.append(Level(1, 0, String("FEAT_B > 0, ALL RIGHT")))
    return out^


# =========================================================================
# PART A -- `gbdt/methods/helpers.mojo`, host only.
# =========================================================================


def _bsp(feature_id: Int, bin_id: Int, gain: Float32) -> TBestSplitProperties:
    """A DEFINED candidate. `score` is set to the NEGATION of the gain so that
    any comparator that reads `Score` where theirs reads `Gain`
    (`gpu_structures.h:81`) orders the candidates backwards and is caught."""
    return TBestSplitProperties(
        Int32(feature_id), Int32(bin_id), -gain, gain
    )


def check_take_best() raises:
    """`TakeBest` (`helpers.h:11-23`) through
    `TBestSplitProperties::operator<` (`gpu_structures.h:80-93`).

    Four properties, each a NAMED case rather than a sweep, because each one
    is a different line of their comparator:

    1. gain decides, and `Score` does not;
    2. on a gain tie, the smaller `FeatureId` wins -- AS A `ui32`, so the
       `(ui32)-1` undefined sentinel LOSES every tie instead of winning it;
    3. on a `(gain, FeatureId)` tie, the smaller `BinId` wins;
    4. on a total tie the comparator is strict and the ternary is
       `first < second ? first : second`, so `TakeBest` falls through to its
       SECOND argument and `TakeBest(a, b, c)` -- left-associated at
       `helpers.h:21` -- keeps the LAST of three. This case is why the file
       is here: the first draft of BOTH the library docstring and this
       expectation said "first", and running it is what found that the
       ternary says otherwise.
    """
    var wrong = 0

    # 1. GAIN, not Score. `lo` has the better (smaller) gain and therefore the
    #    WORSE score under the negation above.
    var lo = _bsp(9, 1, Float32(-5.0))
    var hi = _bsp(2, 1, Float32(-1.0))
    if take_best(lo, hi).feature_id != Int32(9):
        wrong += 1
        print("      take_best did not order by gain (arg order lo, hi)")
    if take_best(hi, lo).feature_id != Int32(9):
        wrong += 1
        print("      take_best did not order by gain (arg order hi, lo)")

    # 2. GAIN TIE -> smaller FeatureId, unsigned.
    var f3 = _bsp(3, 0, Float32(-2.0))
    var f7 = _bsp(7, 0, Float32(-2.0))
    if take_best(f7, f3).feature_id != Int32(3):
        wrong += 1
        print("      take_best did not break a gain tie on FeatureId")

    #    THE SENTINEL. An undefined record is `FeatureId == (ui32)-1`, the
    #    LARGEST ui32 and the SMALLEST Int32. Their `Gain` also defaults to
    #    +inf, so the tie is only reachable when both gains are made equal --
    #    which is exactly what this case does, and it is the only way to see
    #    whether the comparison is signed.
    var undef = TBestSplitProperties()
    var undef_tied = TBestSplitProperties()
    undef_tied.gain = Float32(-2.0)
    if take_best(undef_tied, f3).feature_id != Int32(3):
        wrong += 1
        print(
            "      an UNDEFINED candidate (FeatureId = -1) beat a real one on"
            " a gain tie: the FeatureId comparison is signed, theirs is ui32"
        )
    if take_best(f3, undef_tied).feature_id != Int32(3):
        wrong += 1
        print("      the same, with the arguments the other way round")

    #    And with the DEFAULT undefined record -- gain +inf -- it must lose
    #    outright, which is the reachable case.
    if take_best(undef, f3).feature_id != Int32(3):
        wrong += 1
        print("      a default undefined candidate beat a real one")

    # 3. (gain, FeatureId) TIE -> smaller BinId.
    var b2 = _bsp(4, 2, Float32(-2.0))
    var b9 = _bsp(4, 9, Float32(-2.0))
    if take_best(b9, b2).bin_id != Int32(2):
        wrong += 1
        print("      take_best did not break a full tie on BinId")

    # 4. TOTAL TIE -> keep the FIRST argument. The two records are identical
    #    under the comparator and distinguishable only by a field it never
    #    reads, so this asks purely about `<` versus `<=`.
    var t1 = _bsp(5, 3, Float32(-2.0))
    var t2 = _bsp(5, 3, Float32(-2.0))
    t1.score = Float32(111.0)
    t2.score = Float32(222.0)
    if take_best(t1, t2).score != Float32(222.0):
        wrong += 1
        print(
            "      take_best did not fall through to its SECOND argument on a"
            " total tie: their ternary is `first < second ? first : second`"
            " and operator< is strict (gpu_structures.h:88)"
        )
    if take_best(t2, t1).score != Float32(111.0):
        wrong += 1
        print("      the same, with the arguments the other way round")

    #    take_best3 is LEFT-associated (`helpers.h:21`) and the ternary falls
    #    through to `second`, so on a three-way total tie the LAST survives.
    #    A right-associated fold would keep the middle one, which is a
    #    different answer from either end.
    var t3 = _bsp(5, 3, Float32(-2.0))
    t3.score = Float32(333.0)
    if take_best3(t1, t2, t3).score != Float32(333.0):
        wrong += 1
        print(
            "      take_best3 did not keep the LAST of three tied candidates;"
            " a right-associated fold would keep the middle one"
        )

    #    and it still finds the best when it is last.
    var best_last = _bsp(1, 1, Float32(-9.0))
    if take_best3(t1, t2, best_last).feature_id != Int32(1):
        wrong += 1
        print("      take_best3 missed a winner in its third argument")

    # The comparator itself, directly: strictness both ways.
    if best_split_properties_less(t1, t2) or best_split_properties_less(
        t2, t1
    ):
        wrong += 1
        print("      operator< says two identical records are less than each"
              " other")

    if wrong != 0:
        raise Error(
            String("TakeBest / operator< failed ") + String(wrong)
            + " of its named cases"
        )
    print("  TakeBest: gain order, ui32 sentinel, BinId tie, left-assoc: OK")


def _reverse_bits_oracle(u: Int, n_bits: Int) -> UInt32:
    """An INDEPENDENT reversal: read bit `k` of the input, write it at
    `n_bits - 1 - k`. No masks, no butterfly, no shared constants with
    `reverse_bits`, so the two can only agree by both being right."""
    var out = UInt32(0)
    for k in range(n_bits):
        if ((UInt32(u) >> UInt32(k)) & 1) == 1:
            out |= UInt32(1) << UInt32(n_bits - 1 - k)
    return out


def check_bit_helpers() raises:
    """`ReverseBits`, `GetEvenBits`, `GetOddBits` and `MergeBits`
    (`helpers.h:53-101`).

    `reverse_bits` is compared PER VALUE against an independent bit loop over
    every width from 1 to 16 and every value in `[0, 1 << width)` up to 12
    bits, then a hashed sample above that -- 20,000-odd distinct expected
    values, no two of which are the same.

    The other three are gated as a ROUND TRIP over the whole 8-bit square:

        get_even_bits(merge_bits(x, y)) == x
        get_odd_bits (merge_bits(x, y)) == y

    for all 65,536 pairs, plus `merge_bits` itself against an independent
    interleave. The round trip ties the three functions to each other, so a
    transcription error shared between two of them cannot pass; the
    independent oracle catches an error shared by all three.

    ALSO GATED: the EIGHT-BIT CEILING is theirs and is real. Their extractors
    stop at `(mask & 16384) >> 7`, so bits above 15 of the input are dropped.
    That is asserted rather than worked around.
    """
    var rev_wrong = 0
    var first_bad = String("")
    for n_bits in range(1, 17):
        var limit = 1 << n_bits
        if limit > 4096:
            limit = 4096
        for v in range(limit):
            var got = reverse_bits(v, n_bits)
            var want = _reverse_bits_oracle(v, n_bits)
            if got != want:
                rev_wrong += 1
                if first_bad.byte_length() == 0:
                    first_bad = (
                        String("reverse_bits(") + String(v) + ", "
                        + String(n_bits) + ") = " + String(got) + ", want "
                        + String(want)
                    )
        # above 12 bits, a hashed sample rather than the whole space
        if n_bits > 12:
            for s in range(2000):
                var v2 = Int(_mix(s, n_bits) % UInt64(1 << n_bits))
                if reverse_bits(v2, n_bits) != _reverse_bits_oracle(
                    v2, n_bits
                ):
                    rev_wrong += 1
    if rev_wrong != 0:
        raise Error(
            String("reverse_bits disagrees with an independent bit loop in ")
            + String(rev_wrong) + " cells; first: " + first_bad
        )

    var merge_wrong = 0
    var even_wrong = 0
    var odd_wrong = 0
    for x in range(256):
        for y in range(256):
            var m = merge_bits(x, y)
            # independent interleave
            var want = 0
            for k in range(8):
                if ((x >> k) & 1) == 1:
                    want |= 1 << (2 * k)
                if ((y >> k) & 1) == 1:
                    want |= 1 << (2 * k + 1)
            if m != want:
                merge_wrong += 1
            if get_even_bits(m) != x:
                even_wrong += 1
            if get_odd_bits(m) != y:
                odd_wrong += 1
    if merge_wrong != 0 or even_wrong != 0 or odd_wrong != 0:
        raise Error(
            String("bit interleave: merge wrong in ") + String(merge_wrong)
            + ", get_even_bits round trip wrong in " + String(even_wrong)
            + ", get_odd_bits round trip wrong in " + String(odd_wrong)
            + " of 65536 pairs"
        )

    # THEIR CEILING, asserted. Bit 16 of the input is above the last term of
    # both extractors (`(mask & 16384) >> 7` covers input bit 14 for even and
    # 15 for odd), so it must be DROPPED, not extracted.
    if get_even_bits(1 << 16) != 0:
        raise Error(
            "get_even_bits extracted input bit 16; theirs stops at bit 14"
            " (helpers.h:87) and this port must stop where theirs does"
        )
    if get_odd_bits(1 << 17) != 0:
        raise Error(
            "get_odd_bits extracted input bit 17; theirs stops at bit 15"
            " (helpers.h:74)"
        )
    print(
        "  bit helpers: reverse_bits vs an independent loop over 16 widths,"
        " merge/even/odd round trip over all 65536 pairs, ceiling held: OK"
    )


def check_to_split() raises:
    """`ToSplit` (`helpers.cpp:157-173`).

    The two caps ARE DIFFERENT EXPRESSIONS and the asymmetry is theirs:
    categorical clamps to `GetBinCount(f)`, float clamps to
    `GetBorders(f).size() - 1`. A port that "tidied" them into one would pass
    a check that only ran one arm.
    """
    var wrong = 0

    # float, under the cap
    var s = to_split(Int32(4), Int32(2), True, False, False, Int32(16), Int32(8))
    if (
        s.feature_id != Int32(4)
        or s.bin_idx != Int32(2)
        or s.split_type != Int32(BIN_SPLIT_TAKE_GREATER)
    ):
        wrong += 1
        print("      float arm passed through the wrong record")

    # float, ON the cap: 8 borders -> the largest admissible BinIdx is 7.
    var s2 = to_split(Int32(4), Int32(99), True, False, False, Int32(16), Int32(8))
    if s2.bin_idx != Int32(7):
        wrong += 1
        print(
            "      float arm capped at",
            s2.bin_idx,
            "want 7 (borders.size() - 1)",
        )

    # cat, ON the cap: their clamp is `GetBinCount(f)`, NOT `- 1`, so a
    # 16-bin categorical admits BinIdx 16.
    var s3 = to_split(Int32(5), Int32(99), True, False, True, Int32(16), Int32(8))
    if s3.split_type != Int32(BIN_SPLIT_TAKE_BIN):
        wrong += 1
        print("      cat arm did not select TakeBin")
    if s3.bin_idx != Int32(16):
        wrong += 1
        print(
            "      cat arm capped at",
            s3.bin_idx,
            "want 16: theirs is Min(GetBinCount(f), BinIdx) with NO -1"
            " (helpers.cpp:167)",
        )

    # under the cat cap, untouched
    var s4 = to_split(Int32(5), Int32(3), True, False, True, Int32(16), Int32(8))
    if s4.bin_idx != Int32(3):
        wrong += 1
        print("      cat arm moved a bin that was already inside the cap")

    # `CB_ENSURE(props.Defined())`
    var raised = False
    try:
        _ = to_split(Int32(-1), Int32(0), False, False, False, Int32(16), Int32(8))
    except:
        raised = True
    if not raised:
        wrong += 1
        print("      to_split accepted an UNDEFINED TBestSplitProperties")

    # the unported bundle arm raises rather than answering
    var raised2 = False
    try:
        _ = to_split(Int32(1), Int32(0), True, True, False, Int32(16), Int32(8))
    except:
        raised2 = True
    if not raised2:
        wrong += 1
        print("      to_split silently answered for a FEATURE BUNDLE")

    if wrong != 0:
        raise Error(String("ToSplit failed ") + String(wrong) + " cases")
    print("  ToSplit: both caps, the cat/float asymmetry, both raises: OK")


def check_split_condition_strings() raises:
    """Both `SplitConditionToString` overloads
    (`helpers.cpp:70-104`, `:106-140`) and `PrintBestScore` (`:142-155`).

    The expected strings are assembled from the SAME border list with an
    INDEPENDENTLY WRITTEN index, so what is gated is which border and which
    comparator -- not the digits, which DEVIATION 99.3 explains are Mojo's.

    The case that matters is `ENanMode::Min`: it reads `borders[BinIdx - 1]`
    and every other mode reads `borders[BinIdx]`. Bin 2 under `Min` and bin 2
    under `Max` must therefore name DIFFERENT borders, and with four distinct
    borders that is visible in the text.
    """
    var borders = List[Float32]()
    borders.append(Float32(0.5))
    borders.append(Float32(1.5))
    borders.append(Float32(2.5))
    borders.append(Float32(3.5))

    var greater = TBinarySplit(
        Int32(7), Int32(2), Int32(BIN_SPLIT_TAKE_GREATER)
    )
    var take_bin = TBinarySplit(Int32(7), Int32(2), Int32(BIN_SPLIT_TAKE_BIN))
    var wrong = 0

    def _want(got: String, want: String, what: String) raises -> Int:
        if got != want:
            print("      ", what, "gave", got, "want", want)
            return 1
        return 0

    wrong += _want(
        split_condition_to_string(take_bin, borders, NAN_MODE_FORBIDDEN),
        String("TakeBin"),
        String("TakeBin short-circuit"),
    )
    wrong += _want(
        split_condition_to_string(greater, borders, NAN_MODE_FORBIDDEN),
        String(">") + String(borders[2]),
        String("Forbidden, bin 2"),
    )
    # Min SHIFTS DOWN BY ONE: bin 2 names border 1, not border 2.
    wrong += _want(
        split_condition_to_string(greater, borders, NAN_MODE_MIN),
        String(">") + String(borders[1]),
        String("Min, bin 2 (must read borders[1])"),
    )
    wrong += _want(
        split_condition_to_string(
            TBinarySplit(Int32(7), Int32(0), Int32(BIN_SPLIT_TAKE_GREATER)),
            borders,
            NAN_MODE_MIN,
        ),
        String("== -inf (nan)"),
        String("Min, bin 0"),
    )
    # Max does NOT shift.
    wrong += _want(
        split_condition_to_string(greater, borders, NAN_MODE_MAX),
        String(">") + String(borders[2]),
        String("Max, bin 2 (must read borders[2])"),
    )
    wrong += _want(
        split_condition_to_string(
            TBinarySplit(Int32(7), Int32(4), Int32(BIN_SPLIT_TAKE_GREATER)),
            borders,
            NAN_MODE_MAX,
        ),
        String("== +inf (nan)"),
        String("Max, bin == borders.size()"),
    )

    # `CB_ENSURE(split.BinIdx == borders.size(), "Bin index is too large")`
    var raised = False
    try:
        _ = split_condition_to_string(
            TBinarySplit(Int32(7), Int32(5), Int32(BIN_SPLIT_TAKE_GREATER)),
            borders,
            NAN_MODE_MAX,
        )
    except:
        raised = True
    if not raised:
        wrong += 1
        print("      Max accepted a bin index past borders.size()")

    # The ESplitValue overload: every comparator flips, no index moves.
    wrong += _want(
        split_condition_to_string_value(
            take_bin, borders, NAN_MODE_FORBIDDEN, SPLIT_VALUE_ZERO
        ),
        String("SkipBin"),
        String("TakeBin, Zero"),
    )
    wrong += _want(
        split_condition_to_string_value(
            take_bin, borders, NAN_MODE_FORBIDDEN, SPLIT_VALUE_ONE
        ),
        String("TakeBin"),
        String("TakeBin, One"),
    )
    wrong += _want(
        split_condition_to_string_value(
            greater, borders, NAN_MODE_FORBIDDEN, SPLIT_VALUE_ZERO
        ),
        String("<=") + String(borders[2]),
        String("Forbidden, bin 2, Zero"),
    )
    wrong += _want(
        split_condition_to_string_value(
            greater, borders, NAN_MODE_MIN, SPLIT_VALUE_ZERO
        ),
        String("<=") + String(borders[1]),
        String("Min, bin 2, Zero (index must still shift)"),
    )
    wrong += _want(
        split_condition_to_string_value(
            TBinarySplit(Int32(7), Int32(0), Int32(BIN_SPLIT_TAKE_GREATER)),
            borders,
            NAN_MODE_MIN,
            SPLIT_VALUE_ZERO,
        ),
        String("!= -inf (nan)"),
        String("Min, bin 0, Zero"),
    )
    wrong += _want(
        split_condition_to_string_value(
            TBinarySplit(Int32(7), Int32(4), Int32(BIN_SPLIT_TAKE_GREATER)),
            borders,
            NAN_MODE_MAX,
            SPLIT_VALUE_ZERO,
        ),
        String("!= +inf (nan)"),
        String("Max, overflow bin, Zero"),
    )

    # `PrintBestScore`'s log entry, structure only.
    var msg = best_score_message(
        greater, borders, NAN_MODE_FORBIDDEN, Float64(-1.25), UInt32(3)
    )
    var want_msg = (
        String("Best split for depth 3: 7 / 2 (>")
        + String(borders[2])
        + ") with score "
        + String(Float64(-1.25))
    )
    wrong += _want(msg, want_msg, String("PrintBestScore log entry"))

    if wrong != 0:
        raise Error(
            String("SplitConditionToString / PrintBestScore failed ")
            + String(wrong) + " cases"
        )
    print(
        "  SplitConditionToString: 3 nan arms x 2 split values, the Min"
        " index shift, both CB_ENSUREs, the log entry: OK"
    )
    print("    sample:", msg)
    print_best_score(greater, borders, NAN_MODE_MIN, Float64(0.5), UInt32(1))


def check_has_permutation_dependent_split() raises:
    """`HasPermutationDependentSplit` (`helpers.cpp:60-68`).

    THE POINT IS THE NESTING. Their two predicates are nested, not and-ed: a
    feature that is not a CTR is never asked whether it is permutation
    dependent. The third case below is the one that separates a nested port
    from a flattened one -- a plain float column carrying a stale `True` in
    the second list must NOT trip the answer.
    """
    var structure = TObliviousTreeStructure()
    structure.splits.append(
        TBinarySplit(Int32(0), Int32(1), Int32(BIN_SPLIT_TAKE_GREATER))
    )
    structure.splits.append(
        TBinarySplit(Int32(2), Int32(1), Int32(BIN_SPLIT_TAKE_GREATER))
    )

    var is_ctr: List[Bool] = [False, False, True, True]

    # no CTR split is permutation dependent
    var dep_none: List[Bool] = [False, False, False, False]
    if has_permutation_dependent_split(structure, is_ctr, dep_none):
        raise Error("HasPermutationDependentSplit said yes with no dependent"
                    " CTR in the structure")

    # feature 2 IS in the structure and IS dependent
    var dep_two: List[Bool] = [False, False, True, False]
    if not has_permutation_dependent_split(structure, is_ctr, dep_two):
        raise Error("HasPermutationDependentSplit missed a dependent CTR that"
                    " is in the structure")

    # feature 3 is dependent but is NOT in the structure
    var dep_three: List[Bool] = [False, False, False, True]
    if has_permutation_dependent_split(structure, is_ctr, dep_three):
        raise Error("HasPermutationDependentSplit answered about a feature"
                    " that no split in the structure names")

    # THE NESTING CASE: feature 0 is NOT a CTR but carries True in the second
    # list. Theirs never asks the question, so the answer must be no.
    var dep_zero: List[Bool] = [True, False, False, False]
    if has_permutation_dependent_split(structure, is_ctr, dep_zero):
        raise Error(
            "HasPermutationDependentSplit read IsPermutationDependent for a"
            " feature that IsCtr says is not a CTR: theirs nests the two"
            " (helpers.cpp:62-65), it does not AND them"
        )
    print("  HasPermutationDependentSplit: 4 cases including the nesting: OK")


# =========================================================================
# PART B -- `gbdt/methods/pointwise_optimization_subsets.mojo`, on device.
# =========================================================================


def check_layout_contract() raises:
    """PIN THE RECORD LAYOUT AGAINST LITERALS, not against the library's own
    names.

    **THIS GATE EXISTS BECAUSE THE FIRST VERSION OF THIS FILE DID NOT HAVE
    IT AND A SABOTAGE WALKED STRAIGHT THROUGH.** Swapping `PART_OFFSET` and
    `PART_SIZE` in `pointwise_optimization_subsets.mojo` moved 0 cells,
    because every read below imported those same two constants and followed
    the library wherever it went. A check that spells the layout with the
    library's symbols is not checking the layout; it is checking that the
    library is self-consistent, which a swap preserves exactly.

    The layout is NOT this file's to define and NOT that file's either. It is
    the contract of two layers that landed first and hardcode it:

        gbdt/methods/kernel/split_properties_helpers.mojo:193,196
            partition_sizes.unsafe_load(Int(part_offset) * 2 + 1)
        gbdt/methods/kernel/pointwise_hist2_one_byte_templ.mojo:291-292
            partition.unsafe_load(2 * part + 1)   # size
            partition.unsafe_load(2 * part)       # offset
        gbdt/methods/kernel/pointwise_scores.mojo:700-703
            parts.unsafe_offset(3 * off + 0)      # Weight
            parts.unsafe_offset(3 * off + 1)      # Sum

    and behind those, `TDataPartition {ui32 Offset; ui32 Size;}`
    (`cuda_util/gpu_data/partitions.h`) and `TPartitionStatistics {double
    Weight; double Sum; double Count;}` (`gpu_data/gpu_structures.h:113-116`)
    -- field order, both of them. So the numbers below are written out, and
    every read in this file uses the literal rather than the symbol.
    """
    # COMPTIME, not runtime. These are `comptime` constants, so a runtime
    # comparison folds and the compiler warns that the branch is dead --
    # which is the compiler agreeing with the gate, but noisily, nine times a
    # run. `comptime assert` says the same thing once, at build time, and a
    # layout that has moved does not compile at all rather than failing on
    # the fixture. That is the stronger form for a CONTRACT: you cannot get
    # as far as running a check against a record shape the rest of the stack
    # cannot read.
    comptime assert PARTITION_RECORD == 2, (
        "sizeof(TDataPartition) is two ui32"
    )
    comptime assert PART_OFFSET == 0, (
        "TDataPartition declares Offset FIRST, and"
        " gbdt/methods/kernel/pointwise_hist2_one_byte_templ.mojo:292 reads"
        " it at `2 * part`"
    )
    comptime assert PART_SIZE == 1, (
        "gbdt/methods/kernel/split_properties_helpers.mojo:193 reads the"
        " partition size at `2 * p + 1` to pick the smaller sibling"
    )
    comptime assert PARTITION_STAT_STRIDE == 3, (
        "TPartitionStatistics is three doubles (gpu_structures.h:113-116) and"
        " gbdt/methods/kernel/pointwise_scores.mojo:700 indexes at that"
        " stride"
    )
    comptime assert PART_STAT_WEIGHT == 0, "TPartitionStatistics::Weight is first"
    comptime assert PART_STAT_SUM == 1, "TPartitionStatistics::Sum is second"
    comptime assert PART_STAT_COUNT == 2, "TPartitionStatistics::Count is third"
    comptime assert POINTWISE_STAT_COUNT == 2, (
        "TL2Target has two columns; Count is not one of them"
    )
    comptime assert L2_PLANE_WEIGHT == 0, (
        "stat plane 0 is the weight -- greedy_search_helper.mojo:244,"
        " `stat_count = 2  # [weight, gradient], their layout`"
    )
    comptime assert L2_PLANE_TARGET == 1, "stat plane 1 is the gradient"
    print(
        "  layout contract: partitions[2p + {0,1}] = {Offset, Size},"
        " part_stats[3p + {0,1,2}] = {Weight, Sum, Count}: OK"
    )


struct HostState(Movable):
    """The oracle's copy of `Bins` and `Indices`, maintained by a STABLE
    partition written from the definition and sharing nothing with the
    device's radix sort."""

    var bins: List[Int]
    var idx: List[Int]

    def __init__(out self, n: Int):
        self.bins = List[Int]()
        self.idx = List[Int]()
        for i in range(n):
            self.bins.append(0)
            self.idx.append(i)

    def apply(mut self, feat: Int, bin_idx: Int, depth: Int):
        """One `Split`: OR the bit in, then partition stably by it.

        The partition is `[everything with the bit clear, in order]` followed
        by `[everything with it set, in order]`, which IS the definition of a
        stable one-bit sort. It does no bit arithmetic beyond the one bit and
        has no notion of passes, blocks or scans, so it cannot share a defect
        with `launch_radix_sort_bins`.
        """
        var n = len(self.idx)
        for i in range(n):
            if _goes_right(self.idx[i], feat, bin_idx):
                self.bins[i] = self.bins[i] | (1 << depth)
        var nb = List[Int]()
        var ni = List[Int]()
        for i in range(n):
            if ((self.bins[i] >> depth) & 1) == 0:
                nb.append(self.bins[i])
                ni.append(self.idx[i])
        for i in range(n):
            if ((self.bins[i] >> depth) & 1) == 1:
                nb.append(self.bins[i])
                ni.append(self.idx[i])
        self.bins = nb^
        self.idx = ni^


def _poison_tail(
    ctx: DeviceContext, mut subsets: TOptimizationSubsets, part_count: Int
) raises:
    """Fill every partition RECORD and stat RECORD at or above `part_count`
    with poison. Re-read after the NEXT level; see rule 5 in the header.

    Both strides are the record stride, not 1: a poison written at the wrong
    stride would leave live cells poisoned and the gate would fail for the
    wrong reason, which is a way for a tail check to stop meaning anything.
    """
    var m = subsets.max_part_count
    if part_count >= m:
        return
    var n_tail = m - part_count
    var h_u = ctx.enqueue_create_host_buffer[DType.uint32](n_tail * 2)
    for i in range(n_tail * 2):
        h_u.unsafe_ptr().unsafe_store(i, POISON_U32)
    var h_f = ctx.enqueue_create_host_buffer[DType.float32](n_tail * 3)
    for i in range(n_tail * 3):
        h_f.unsafe_ptr().unsafe_store(i, POISON_F32)
    ctx.synchronize()
    ctx.enqueue_copy(
        dst_ptr=subsets.partitions.unsafe_ptr().unsafe_offset(part_count * 2),
        src_ptr=h_u.unsafe_ptr(),
        size=n_tail * 2,
    )
    ctx.enqueue_copy(
        dst_ptr=subsets.partition_stats.unsafe_ptr().unsafe_offset(
            part_count * 3
        ),
        src_ptr=h_f.unsafe_ptr(),
        size=n_tail * 3,
    )
    ctx.synchronize()


def _verify_level(
    ctx: DeviceContext,
    mut subsets: TOptimizationSubsets,
    host: HostState,
    label: String,
    mut order_wrong_out: Int,
    mut content_wrong_out: Int,
) raises:
    """Read everything back and compare PER CELL, splitting the verdict into
    the two independent halves rule 3 describes.

    ORDER (stability-sensitive):
      - `indices[i]` against the host's stable permutation, per position
      - `bins[i]` against the host's bins, per position
      - `gathered[plane * n + i]` against the planted stat of
        `host.idx[i]`, per (plane, position)

    CONTENT (stability-INSENSITIVE): everything a naive check would look at.
      - which partition each DOCUMENT is in, per document
      - `partitions_offset[p]` and `partitions_size[p]`, per partition,
        including every empty one
      - `partition_stats[p * 2 + s]`, per (partition, stat), exactly
      - the poisoned tail above `current_part_count()`, per cell

    A reversal inside every partition drives ORDER to hundreds and leaves
    CONTENT at zero. That is the whole reason the two counts are separate.
    """
    var n = subsets.doc_count
    var part_count = subsets.current_part_count()
    var m = subsets.max_part_count

    var g_idx = ctx.enqueue_create_host_buffer[DType.uint32](n)
    var g_bins = ctx.enqueue_create_host_buffer[DType.uint32](n)
    # TWO buffers now, and BOTH are read back. A buffer split is exactly the
    # sort of change that quietly halves a tally: if only `g_gat_w` were read
    # the gathered count would still be non-zero on most defects, still look
    # healthy, and would no longer cover the gradient column at all.
    var g_gat_w = ctx.enqueue_create_host_buffer[DType.float32](n)
    var g_gat_t = ctx.enqueue_create_host_buffer[DType.float32](n)
    var g_parts = ctx.enqueue_create_host_buffer[DType.uint32](m * 2)
    var g_stats = ctx.enqueue_create_host_buffer[DType.float32](m * 3)
    ctx.enqueue_copy(dst_ptr=g_idx.unsafe_ptr(), src_buf=subsets.indices)
    ctx.enqueue_copy(dst_ptr=g_bins.unsafe_ptr(), src_buf=subsets.bins)
    ctx.enqueue_copy(
        dst_ptr=g_gat_w.unsafe_ptr(), src_buf=subsets.gathered_weight
    )
    ctx.enqueue_copy(
        dst_ptr=g_gat_t.unsafe_ptr(), src_buf=subsets.gathered_target
    )
    ctx.enqueue_copy(dst_ptr=g_parts.unsafe_ptr(), src_buf=subsets.partitions)
    ctx.enqueue_copy(
        dst_ptr=g_stats.unsafe_ptr(), src_buf=subsets.partition_stats
    )
    ctx.synchronize()

    # ---- ORDER ----------------------------------------------------------
    var idx_wrong = 0
    var bins_wrong = 0
    var first_idx_bad = -1
    for i in range(n):
        if Int(g_idx.unsafe_ptr().unsafe_load(i)) != host.idx[i]:
            idx_wrong += 1
            if first_idx_bad < 0:
                first_idx_bad = i
        if Int(g_bins.unsafe_ptr().unsafe_load(i)) != host.bins[i]:
            bins_wrong += 1

    var gat_wrong = 0
    var gat_w_wrong = 0
    var gat_t_wrong = 0
    for i in range(n):
        var doc = host.idx[i]
        # Both columns, per position. Counted separately so a split that
        # loses one of them is visible as a HALVING rather than as a smaller
        # number of the same kind.
        var w = g_gat_w.unsafe_ptr().unsafe_load(i)
        var t = g_gat_t.unsafe_ptr().unsafe_load(i)
        if w != Float32(_weight(doc)):
            gat_wrong += 1
            gat_w_wrong += 1
        if t != Float32(_target(doc)):
            gat_wrong += 1
            gat_t_wrong += 1

    # ---- CONTENT --------------------------------------------------------
    # Which partition each DOCUMENT is in, per document, derived from the
    # DEVICE's own offsets/sizes/indices. Independent of order inside a
    # partition.
    var dev_part = List[Int]()
    for _ in range(n):
        dev_part.append(-1)
    var host_part = List[Int]()
    for _ in range(n):
        host_part.append(-1)
    for i in range(n):
        host_part[host.idx[i]] = host.bins[i]
    var covered = 0
    for p in range(part_count):
        var off = Int(g_parts.unsafe_ptr().unsafe_load(p * 2 + 0))
        var sz = Int(g_parts.unsafe_ptr().unsafe_load(p * 2 + 1))
        if off < 0 or sz < 0 or off + sz > n:
            raise Error(
                String(label) + ": partition " + String(p)
                + " is [" + String(off) + ", +" + String(sz)
                + ") which does not fit in " + String(n) + " documents"
            )
        for k in range(sz):
            var doc = Int(g_idx.unsafe_ptr().unsafe_load(off + k))
            if doc >= 0 and doc < n:
                dev_part[doc] = p
            covered += 1
    var membership_wrong = 0
    for d in range(n):
        if dev_part[d] != host_part[d]:
            membership_wrong += 1
    if covered != n:
        membership_wrong += 1
        print(
            "      the partitions cover", covered, "positions, not", n,
            "-- they overlap or leave a hole",
        )

    # Offsets and sizes, per partition, INCLUDING every empty one. The host
    # derives them by scanning its own bins, which is how a partition array
    # is defined, not how the kernels compute it.
    var want_size = List[Int]()
    for _ in range(m):
        want_size.append(0)
    for i in range(n):
        want_size[host.bins[i]] = want_size[host.bins[i]] + 1
    var want_off = List[Int]()
    var run = 0
    for p in range(m):
        want_off.append(run)
        if p < part_count:
            run += want_size[p]
    var off_wrong = 0
    var size_wrong = 0
    for p in range(part_count):
        if Int(g_parts.unsafe_ptr().unsafe_load(p * 2 + 0)) != want_off[p]:
            off_wrong += 1
        if Int(g_parts.unsafe_ptr().unsafe_load(p * 2 + 1)) != want_size[p]:
            size_wrong += 1

    # Per (partition, stat), exactly. Integer plants, so equality.
    var want_w = List[Float32]()
    var want_t = List[Float32]()
    for _ in range(m):
        want_w.append(Float32(0.0))
        want_t.append(Float32(0.0))
    for i in range(n):
        var b = host.bins[i]
        want_w[b] = want_w[b] + Float32(_weight(host.idx[i]))
        want_t[b] = want_t[b] + Float32(_target(host.idx[i]))
    var stat_wrong = 0
    var first_stat_bad = -1
    # WHICH plane moved. This used to be unrecorded and the print below
    # always showed the WEIGHT, so a Sum-only defect -- which is what
    # dropping one of the two Gathers now produces -- printed two identical
    # numbers and read as a harness bug.
    var first_stat_plane = 0
    for p in range(part_count):
        var gw = g_stats.unsafe_ptr().unsafe_load(p * 3 + 0)
        var gt = g_stats.unsafe_ptr().unsafe_load(p * 3 + 1)
        if gw != want_w[p]:
            stat_wrong += 1
            if first_stat_bad < 0:
                first_stat_bad = p
                first_stat_plane = 0
        if gt != want_t[p]:
            stat_wrong += 1
            if first_stat_bad < 0:
                first_stat_bad = p
                first_stat_plane = 1

    # PLANE 2, `TPartitionStatistics::Count`. Their `PartitionUpdateImpl`
    # sets it to `size` whenever `counts == nullptr`, which is always on this
    # path (`pointwise_kernels.h:240`), and NO SCORER READS IT -- theirs
    # touches `.Weight`/`.Sum` only, and so does the ported one
    # (`gbdt/methods/kernel/pointwise_scores.mojo:700-703`). It is gated
    # anyway, for two reasons: an unread plane that holds garbage is a plane
    # that will hold garbage on the day something does read it, and the
    # stride-3 packing is the thing that keeps every OTHER read aligned, so a
    # wrong plane 2 is evidence the record is being written at the wrong
    # stride. Compared against the host's own count, NOT against the device's
    # own `partitions_size`, which would be comparing the port to itself.
    var count_wrong = 0
    for p in range(part_count):
        var gc = g_stats.unsafe_ptr().unsafe_load(p * 3 + 2)
        if gc != Float32(want_size[p]):
            count_wrong += 1

    # The poisoned tail. Anything sized from `max_part_count` instead of
    # `current_part_count()` shows up here and nowhere else.
    var tail_wrong = 0
    for p in range(part_count, m):
        for k in range(2):
            if g_parts.unsafe_ptr().unsafe_load(p * 2 + k) != POISON_U32:
                tail_wrong += 1
        for k in range(3):
            if g_stats.unsafe_ptr().unsafe_load(p * 3 + k) != POISON_F32:
                tail_wrong += 1

    var order_wrong = idx_wrong + bins_wrong + gat_wrong
    var content_wrong = (
        membership_wrong
        + off_wrong
        + size_wrong
        + stat_wrong
        + count_wrong
        + tail_wrong
    )
    order_wrong_out = order_wrong
    content_wrong_out = content_wrong

    print(
        "    ", label,
        "| parts", part_count,
        "| CONTENT wrong", content_wrong,
        "( membership", membership_wrong,
        "offset", off_wrong,
        "size", size_wrong,
        "stats", stat_wrong,
        "count", count_wrong,
        "tail", tail_wrong, ")",
        "| ORDER wrong", order_wrong,
        "( indices", idx_wrong,
        "bins", bins_wrong,
        "gathered", gat_wrong,
        "[w", gat_w_wrong,
        "t", gat_t_wrong, "] )",
    )
    if first_idx_bad >= 0:
        print(
            "       first indices disagreement at position", first_idx_bad,
            ": device", g_idx.unsafe_ptr().unsafe_load(first_idx_bad),
            "host", host.idx[first_idx_bad],
        )
    if first_stat_bad >= 0:
        var plane_name = String("Weight") if first_stat_plane == 0 else (
            String("Sum")
        )
        var got_v = g_stats.unsafe_ptr().unsafe_load(
            first_stat_bad * 3 + first_stat_plane
        )
        var want_v = want_w[first_stat_bad] if first_stat_plane == 0 else (
            want_t[first_stat_bad]
        )
        print(
            "       first stat disagreement at partition", first_stat_bad,
            "plane", plane_name, ": device", got_v, "host", want_v,
        )


def check_subsets() raises:
    """`CreateSubsets` then five `Split`s, gated after every one.

    The compressed index is uploaded once and never permuted -- which is the
    invariant `gbdt/gpu_util/gpu_data/partitions.mojo` states about the real
    one -- and `docs_for_bins` is `subsets.indices` itself, because their
    caller builds it as `Gather(observations, subsets.Indices)` with
    `observations` the identity in this fixture
    (`oblivious_tree_doc_parallel_structure_searcher.cpp:65`). So a wrong
    permutation at level `d` propagates into level `d+1` rather than being
    quietly repaired.
    """
    var ctx = DeviceContext()
    var n = N_DOCS

    # ---- the compressed index, two columns ------------------------------
    var h_cindex = ctx.enqueue_create_host_buffer[DType.uint32](2 * n)
    for d in range(n):
        # column 0: FEAT_A at bits [0,4). FEAT_B is at bits [8,12) of column 1.
        h_cindex.unsafe_ptr().unsafe_store(d, UInt32(_val_a(d)))
        # column 1: FEAT_C at bits [4,7), FEAT_B at bits [8,12)
        h_cindex.unsafe_ptr().unsafe_store(
            n + d, UInt32((_val_c(d) << 4) | (_val_b(d) << 8))
        )
    var d_cindex = ctx.enqueue_create_buffer[DType.uint32](2 * n)
    ctx.enqueue_copy(dst_buf=d_cindex, src_ptr=h_cindex.unsafe_ptr())

    # ---- TL2Target, two columns, in ORIGINAL document order --------------
    # `TL2Target { WeightedTarget; Weights; }` -- two buffers, theirs.
    var h_w = ctx.enqueue_create_host_buffer[DType.float32](n)
    var h_t = ctx.enqueue_create_host_buffer[DType.float32](n)
    for d in range(n):
        h_w.unsafe_ptr().unsafe_store(d, Float32(_weight(d)))
        h_t.unsafe_ptr().unsafe_store(d, Float32(_target(d)))
    var d_w = ctx.enqueue_create_buffer[DType.float32](n)
    var d_t = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.enqueue_copy(dst_buf=d_w, src_ptr=h_w.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_t, src_ptr=h_t.unsafe_ptr())
    ctx.synchronize()

    # `groupedByBinObservations = TStripeBuffer<ui32>::CopyMapping(observations)`
    # (`oblivious_tree_doc_parallel_structure_searcher.cpp:27`) and the
    # `Gather(groupedByBinObservations, observations, subsets.Indices)` their
    # level loop opens with (`:65`). `observations` is the identity in this
    # fixture, so `groupedByBinObservations` ends up equal to `Indices` -- but
    # it is a SEPARATE buffer produced by a real gather, exactly as theirs is,
    # and not an alias of the one `Split` is about to permute.
    var d_obs = ctx.enqueue_create_buffer[DType.uint32](n)
    var d_docs = ctx.enqueue_create_buffer[DType.uint32](n)
    launch_make_sequence(ctx, UInt32(0), d_obs, n)
    ctx.synchronize()

    var source = TL2Target(d_w^, d_t^, n)
    var subsets = create_subsets(ctx, MAX_DEPTH, source)
    ctx.synchronize()

    var host = HostState(n)
    var total_order = 0
    var total_content = 0

    _poison_tail(ctx, subsets, subsets.current_part_count())
    var o = 0
    var c = 0
    _verify_level(
        ctx, subsets, host, String("CreateSubsets (depth 0)"), o, c
    )
    total_order += o
    total_content += c

    var levels = _levels()
    for li in range(len(levels)):
        var lv = levels[li]
        var depth = Int(subsets.current_depth + subsets.fold_bits)

        var f_off: UInt32
        var f_shift: UInt32
        var f_mask: UInt32
        var f_one_hot: Bool
        if lv.feat == 0:
            f_off = UInt32(FEAT_A_OFFSET)
            f_shift = FEAT_A_SHIFT
            f_mask = FEAT_A_MASK
            f_one_hot = False
        elif lv.feat == 1:
            f_off = UInt32(FEAT_B_OFFSET)
            f_shift = FEAT_B_SHIFT
            f_mask = FEAT_B_MASK
            f_one_hot = False
        else:
            f_off = UInt32(FEAT_C_OFFSET)
            f_shift = FEAT_C_SHIFT
            f_mask = FEAT_C_MASK
            f_one_hot = True

        # `Gather(groupedByBinObservations, observations, subsets.Indices)`
        # (`oblivious_tree_doc_parallel_structure_searcher.cpp:65`), run
        # BEFORE the split so `docsForBins` reflects the permutation the
        # PREVIOUS level left behind -- which is where their call sits too.
        launch_gather_with_mask_u32(
            ctx, d_docs, d_obs, subsets.indices, n, UInt32(0xFFFFFFFF)
        )
        split_subsets(
            ctx,
            source,
            d_cindex,
            d_docs,
            f_off,
            f_mask,
            f_shift,
            f_one_hot,
            UInt32(lv.bin_idx),
            subsets,
        )
        ctx.synchronize()
        host.apply(lv.feat, lv.bin_idx, depth)

        if Int(subsets.current_depth) != li + 1:
            raise Error(
                String("after level ") + String(li)
                + " CurrentDepth is " + String(subsets.current_depth)
                + ", want " + String(li + 1)
                + ": their ++CurrentDepth runs BEFORE UpdateSubsetsStats"
                " (pointwise_optimization_subsets.cpp:49-51)"
            )

        var o2 = 0
        var c2 = 0
        _verify_level(
            ctx,
            subsets,
            host,
            String("L") + String(li) + " " + lv.name,
            o2,
            c2,
        )
        total_order += o2
        total_content += c2
        _poison_tail(ctx, subsets, subsets.current_part_count())

    # REACH, per branch: L3 must have produced an empty upper half and L4 an
    # empty lower one. If the fixture stopped doing that -- a changed hash, a
    # changed document count -- the two walks it exists to exercise stop
    # being exercised and the check would go quietly green on less.
    var empty_after_l3 = 0
    for i in range(n):
        if ((host.bins[i] >> 3) & 1) == 1:
            empty_after_l3 += 1
    var left_after_l4 = 0
    for i in range(n):
        if ((host.bins[i] >> 4) & 1) == 0:
            left_after_l4 += 1
    if empty_after_l3 != 0:
        raise Error(
            String("L3 was supposed to send EVERY document left and sent ")
            + String(empty_after_l3)
            + " right; the trailing zero walk of UpdatePartitionSizes is no"
            " longer exercised"
        )
    if left_after_l4 != 0:
        raise Error(
            String("L4 was supposed to send EVERY document right and sent ")
            + String(left_after_l4)
            + " left; the `i ? bins[i-1] : 0` sentinel of"
            " UpdatePartitionSizes is no longer exercised"
        )

    if total_content != 0 or total_order != 0:
        raise Error(
            String("the pointwise subsets disagree with the host oracle: ")
            + String(total_content) + " CONTENT cells and "
            + String(total_order) + " ORDER cells over 6 levels"
        )
    print(
        "  CreateSubsets + 5 Splits: every position, every partition, every"
        " (partition, stat) cell and the poisoned tail agree"
    )


def main() raises:
    print("pointwise optimization subsets + methods/helpers")
    print(" A. helpers.mojo (host)")
    check_take_best()
    check_bit_helpers()
    check_to_split()
    check_split_condition_strings()
    check_has_permutation_dependent_split()
    check_layout_contract()
    print(" B. pointwise_optimization_subsets.mojo (device),",
          N_DOCS, "documents, max depth", MAX_DEPTH)
    check_subsets()
    print("both halves agree with the host oracle")
