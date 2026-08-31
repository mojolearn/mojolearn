# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The FOLD machinery of CatBoost's ORDERED boosting: `CreateFolds` and the
fold/permutation structures immediately around it.

PORT OF `catboost/cuda/methods/dynamic_boosting.h` at CatBoost `54a8143a`,
lines `:75-119` (`TFold`, `TFoldAndPermutationStorage`), `:177-185`
(`MinEstimationSize`), `:189-223` (`CreateFolds`) and `:283-289` / `:573-575`
(the permutation counts the fold grid is sized by). Supporting reads:
`catboost/cuda/gpu_data/samples_grouping.h:13-127` (`IQueriesGrouping` and its
two implementations), `catboost/cuda/utils/helpers.h:3-6` (`CeilDivide`),
`catboost/libs/helpers/math_utils.h:14-16` (`NCB::IntLog2`),
`catboost/cuda/cuda_lib/slice.h:9` (`TSlice`). Transliterated. Do not improve.

WHY THIS FILE IS ITS OWN THING, AND WHY IT NEEDS NO GPU
------------------------------------------------------
Ordered boosting is CatBoost's SHIPPED GPU DEFAULT for every non-multiclass
loss -- `catboost_options.cpp:802-806` sets `boostingType.SetDefault(Ordered)`
whenever the task is GPU, the loss is not a multiclass/multi-target one, and
the user did not say otherwise. It trains each tree on nested PREFIX FOLDS of
a permuted dataset: fold `k` estimates leaf values on `[0, L_k)` and is scored
on `[L_k, R_k)`, with the prefixes growing geometrically by
`fold_len_multiplier` (default 2.0, `boosting_options.cpp:11`) from
`min_fold_size` (default 100, `boosting_options.cpp:24`).

Everything that decides WHERE those boundaries fall is host integer
arithmetic over `sampleCount`, `min_fold_size`, `fold_len_multiplier` and the
sample grouping. No buffer, no kernel, no device. So it is a closed form and
it can be gated against an independently written one --
`original/dynamic_boosting_folds_check.mojo` does exactly that. That is the
whole reason this piece is worth landing on its own, ahead of the
`TDynamicBoosting` loop and the feature-parallel searcher that consume it.

WHAT THE FOLDS TURN INTO, AND WHY `fold_bits` IS THE LINK
---------------------------------------------------------
`PORTING.md` 91 B: the fold id occupies the LOW BITS of a document's bin and
the depth bits sit above it. The bridge is two lines of
`TFeatureParallelObliviousTreeSearcher::CreateSubsets`
(`oblivious_tree_structure_searcher.cpp:36-37`):

    subsets.FoldCount = initParts.size();
    subsets.FoldBits  = NCB::IntLog2(subsets.FoldCount);

and `initParts` comes from `WriteFoldBasedInitialBins` (`:338-364`), which
pushes TWO partitions per fold -- the estimate half at bin `2k` and the
quality-evaluate half at bin `2k + 1`. So

    FoldCount = 2 * folds.size()          (Ordered)
    FoldCount = 1                          (Plain, WriteSingleTaskInitialBins
                                            `:366-376`, one part, bins all 0)

`fold_count_for_folds` and `fold_bits_for_folds` below are those two lines and
nothing else. They are the numbers that flow into
`gbdt/methods/pointwise_optimization_subsets.mojo`'s `fold_count`/`fold_bits`
fields -- both 0/1 on the Plain path today -- and from there into
`PointwisePartOffsetsHelper` in
`gbdt/methods/kernel/split_properties_helpers.mojo`, where a DATA-PARTITION
offset rounds the fold axis up to `1 << fold_bits` and a HISTOGRAM offset
packs it tight at `fold_count`. Neither of those two files is touched here and
neither is duplicated here; this file only computes the two integers they take
as input, and the check reads the existing helper to show the two offsets
diverge at a real fold count.

WHAT THIS FILE DELIBERATELY DOES NOT PORT
-----------------------------------------
* `TDynamicBoosting::Fit` (`:234-540`), the boosting loop itself.
* `TFeatureParallelObliviousTreeSearcher` and `WriteFoldBasedInitialBins`'
  DEVICE half -- the `FillBuffer(learnBins, currentBin)` /
  `FillBuffer(testBins, currentBin + 1)` pair and the `bins.Reset(...)`
  around them (`:338-364`). Only the host arithmetic that says how many
  partitions there are survives, as `fold_count_for_folds`.
* `TFeatureParallelDataSet` / `TFeatureParallelDataSetsHolder`. `CreateFolds`'
  second overload (`:226-232`) exists only to pull `GetObjectCount()` and
  `GetSamplesGrouping()` out of one; callers here pass those two directly.
* `TQueriesGrouping`'s constructor proper (`samples_grouping.h:62-105`): the
  `TDataPermutation::FillGroupOrder` reordering, `QueryPairOffsets`,
  `FlatQueryPairs` and `QueryPairWeights`. `CreateFolds` reads exactly three
  members of the interface and none of the pair machinery. See DEVIATION 110.
* `TPermutationTarget` (`:79-95`), which is a vector of device targets with a
  `GetTarget(permutationId)` accessor. Nothing in it is fold arithmetic.

================================ DEVIATION 110 ========================
THEIRS: `IQueriesGrouping` (`samples_grouping.h:13-26`) is an abstract class
with five pure virtuals and two implementations, `TWithoutQueriesGrouping`
(`:28-58`) and `TQueriesGrouping` (`:60-...`). `CreateFolds` takes it by
`const&` and dispatches virtually.

OURS: one struct with a `kind` tag, built by `IQueriesGrouping.without_queries`
or `IQueriesGrouping.queries`, dispatching on the tag. `PORTING_RULES` rule 4
names this workaround: Mojo has no dynamic trait objects, and a tagged union
is what their worker switches on anyway. NOT ARITHMETIC -- every one of the
five accessors is transcribed branch for branch below.

Two members of `TQueriesGrouping` are NOT built: the pair vectors
(`FlatQueryPairs`, `QueryPairWeights`, `QueryPairOffsets`), which only
`GetQueryPairOffset` and the pairwise losses read, and the
`TDataPermutation::FillGroupOrder` shuffle, which decides in what ORDER groups
are laid out. Ours takes the already-ordered group sizes, which is the state
their constructor leaves behind. Consequence, stated rather than left to be
found: a caller cannot use this to permute groups, only to describe a grouping
that is already permuted.
======================================================================

================================ DEVIATION 111 ========================
THEIRS: `ui32` throughout `MinEstimationSize` and `CreateFolds`, `ui64` inside
`TSlice`, and one narrowing cast that matters --
`static_cast<ui32>(minEstimationSize * growthRate)` (`:211`, `:217`), a
`double` multiply truncated toward zero and then wrapped to 32 bits.

OURS: `Int` (signed, 64-bit) throughout, matching `gbdt/gpu_lib/slice.mojo`,
which already stores `TSlice` bounds as `Int` and declares that departure.
`Int(Float64)` truncates toward zero, so the truncation is the same; what is
NOT the same is the wrap. Theirs wraps at 2^32, which for
`minEstimationSize * growthRate >= 2^32` would produce a value SMALLER than
`sampleCount`, defeating the `Min` on the next line and running the loop on a
boundary that went backwards. Reaching it needs `sampleCount > 2^32 /
growth_rate` -- above `ui32`'s own range at `growthRate >= 1.0`, so their
`ui32 sampleCount` cannot actually get there and the wrap is unreachable in
their code too. Ours simply does not have it. NOT ARITHMETIC at any input
either tree admits.

`NCB::IntLog2` is `(ui32)ceil(log2(values))` in `double`
(`math_utils.h:14-16`, with their own TODO calling it slow). This file calls
`gbdt/ctrs/ctr_bins_builder.int_log2`, already in this tree, which computes it
as the BIT LENGTH of `values - 1` -- that identity for every `values >= 1`,
compared over every value in `[1, 1 << 20]` when it was written, and
re-compared here at the four powers of two `MinEstimationSize`'s threshold
actually lands on. It is REUSED rather than re-spelled. Note this is a
different function from the `1 << (ui32)ceil(log2((float)FoldCount))` inside
`PointwisePartOffsetsHelper.data_partition_offset`, which is a DEVICE
expression in `float` and is deliberately kept in float there.
======================================================================

================================ DEVIATION 112 ========================
THEIRS: `CreateFolds`' growth loop (`:215-222`) has no iteration bound. It
terminates because `NextQueryOffsetForLine` is strictly increasing below
`sampleCount` in both of their groupings, so `QualityEvaluateSamples.Right`
gains at least 1 per pass and is clamped at `sampleCount`.

OURS: the same loop with `max_iterations = sample_count + 2` and a `raise` if
it is exceeded. The bound is not a tuning knob -- it is the length of the
longest sequence their own termination argument permits, since the right edge
is a strictly increasing integer in `[1, sample_count]`.

MEASURED, AND THIS IS THE ONLY REASON IT EXISTS: the gate for this file has to
be able to FAIL, and the most direct sabotage of the fold series -- dropping
the `+ 1` from `TWithoutQueriesGrouping::NextQueryOffsetForLine` -- turns the
loop into `r <- floor(r * g)`, which at `g = 1.05, r = 1` is a fixed point.
Without the guard that sabotage HANGS the check instead of reddening it, and
a check that hangs proves nothing. With the guard it raises and the gate goes
red by name. Unsabotaged the guard has never fired. MEASURED fold counts at
the extremes the check sweeps: 4,899 folds at `n = 5,000, g = 1.0000001`
against a bound of 5,002; 78 at `g = 1.05`; 6 at the default `g = 2.0`. It has
never come within 100 passes of the bound.

Sabotage S1 in `original/dynamic_boosting_folds_check.mojo` is the measured
case: with the `+ 1` removed the guard raises at 502 passes over 499 samples
and the run goes red in 2 seconds. Sabotage S6 -- stepping the series from
`min_estimation` instead of the previous right edge -- trips it at 9 passes
over 6 samples, likewise a hang without it.
======================================================================
"""

from gbdt.ctrs.ctr_bins_builder import int_log2
from gbdt.gpu_lib.slice import TSlice


# ---------------------------------------------------------------------------
# `EBoostingType` (`private/libs/options/enums.h:48-51`), same two values in
# their order.  `Ordered` is 0 and `Plain` is 1, and the ordinal matters
# because their JSON option layer round-trips the enum by index.
# ---------------------------------------------------------------------------


struct EBoostingType(Copyable, ImplicitlyCopyable, Movable):
    """Their `EBoostingType` (`enums.h:48-51`)."""

    var value: Int32

    comptime Ordered = Self(0)
    comptime Plain = Self(1)

    def __init__(out self, value: Int32):
        self.value = value

    def __eq__(self, other: Self) -> Bool:
        return self.value == other.value

    def __ne__(self, other: Self) -> Bool:
        return self.value != other.value

    def to_string(self) -> String:
        if self == Self.Ordered:
            return "Ordered"
        return "Plain"


# ---------------------------------------------------------------------------
# `TFold` (`dynamic_boosting.h:75-78`).
# ---------------------------------------------------------------------------


struct TFold(Copyable, ImplicitlyCopyable, Movable):
    """Their `TFold` (`dynamic_boosting.h:75-78`), two slices and no methods.

        struct TFold {
            TSlice EstimateSamples;
            TSlice QualityEvaluateSamples;
        };

    `EstimateSamples` is the PREFIX the leaf values of this fold are estimated
    on; `QualityEvaluateSamples` is the block immediately after it that the
    structure is scored on. Ordered boosting's whole no-look-ahead property is
    that the second never overlaps the first
    (`dynamic_boosting.h:314-329` builds one learn target and one validate
    target per fold from exactly these two slices).
    """

    var estimate_samples: TSlice
    var quality_evaluate_samples: TSlice

    def __init__(out self, estimate: TSlice, quality_evaluate: TSlice):
        self.estimate_samples = estimate
        self.quality_evaluate_samples = quality_evaluate


# ---------------------------------------------------------------------------
# `IQueriesGrouping` (`gpu_data/samples_grouping.h:13-127`).  DEVIATION 110.
# ---------------------------------------------------------------------------

comptime GROUPING_WITHOUT_QUERIES = 0
"""`TWithoutQueriesGrouping` (`samples_grouping.h:28-58`): one document per
group, which is what every POINTWISE loss gets."""

comptime GROUPING_QUERIES = 1
"""`TQueriesGrouping` (`samples_grouping.h:60-...`): real groups, zero-based
group indices."""


struct IQueriesGrouping(Copyable, Movable):
    """Their `IQueriesGrouping` interface plus both implementations, as a
    tagged union.  DEVIATION 110.

    `CreateFolds` reads three of the five accessors -- `GetQueryCount`,
    `GetQueryOffset` and `NextQueryOffsetForLine`. `GetQueryId` is here
    because `NextQueryOffsetForLine` is written in terms of it in their
    grouped implementation (`samples_grouping.h:123-129`), and `GetQuerySize`
    because dropping one member of a five-member interface is how a port
    starts drifting.
    """

    var kind: Int
    var doc_count: Int
    var query_sizes: List[UInt32]
    var query_offsets: List[UInt32]
    var query_ids: List[UInt32]

    def __init__(
        out self,
        kind: Int,
        doc_count: Int,
        var query_sizes: List[UInt32],
        var query_offsets: List[UInt32],
        var query_ids: List[UInt32],
    ):
        self.kind = kind
        self.doc_count = doc_count
        self.query_sizes = query_sizes^
        self.query_offsets = query_offsets^
        self.query_ids = query_ids^

    @staticmethod
    def without_queries(doc_count: Int) -> Self:
        """`TWithoutQueriesGrouping(ui32 docCount)` (`samples_grouping.h
        :30-33`). Carries one integer; every accessor is a formula.
        """
        var empty_a = List[UInt32]()
        var empty_b = List[UInt32]()
        var empty_c = List[UInt32]()
        return Self(
            GROUPING_WITHOUT_QUERIES,
            doc_count,
            empty_a^,
            empty_b^,
            empty_c^,
        )

    @staticmethod
    def queries(group_sizes: List[UInt32]) raises -> Self:
        """The STATE `TQueriesGrouping`'s constructor leaves behind
        (`samples_grouping.h:62-105`), given groups already in their final
        order.

        Their loop is transcribed for the three vectors that survive
        DEVIATION 110:

            QuerySizes[groupId]   = groupInfo.GetSize();
            QueryOffsets[groupId] = groupStartIdx;
            QueryIds[groupStartIdx + j] = groupIdx;   for j in [0, size)
            groupStartIdx += groupInfo.GetSize();

        Their two `CB_ENSURE`s are transcribed with them: `"Error: empty
        group"` (`:80`) and `"Error: all groups have size 1"` (`:103`), the
        latter counting groups of size > 1 across the whole pool.
        """
        var query_sizes = List[UInt32]()
        var query_offsets = List[UInt32]()
        var query_ids = List[UInt32]()

        var at_least_two_doc_queries_count = 0
        var group_idx = 0
        var group_start_idx = 0

        for group_id in range(len(group_sizes)):
            var size = Int(group_sizes[group_id])
            if size == 0:
                raise Error("Error: empty group")
            query_sizes.append(UInt32(size))
            query_offsets.append(UInt32(group_start_idx))
            for _j in range(size):
                query_ids.append(UInt32(group_idx))
            if size > 1:
                at_least_two_doc_queries_count += 1
            group_start_idx += size
            group_idx += 1

        if at_least_two_doc_queries_count == 0:
            raise Error("Error: all groups have size 1")

        return Self(
            GROUPING_QUERIES,
            group_start_idx,
            query_sizes^,
            query_offsets^,
            query_ids^,
        )

    def get_query_count(self) -> Int:
        """`GetQueryCount()`. `DocCount` without queries
        (`samples_grouping.h:35-37`), `QuerySizes.size()` with
        (`:107-109`)."""
        if self.kind == GROUPING_WITHOUT_QUERIES:
            return self.doc_count
        return len(self.query_sizes)

    def get_query_offset(self, id: Int) -> Int:
        """`GetQueryOffset(ui32 id)`. `Min(id, DocCount)` without queries
        (`samples_grouping.h:39-41`); with queries, `QueryOffsets[id]` when
        `id` is in range and `QueryIds.size()` -- the DOC count, not the group
        count -- when it is not (`:111-113`)."""
        if self.kind == GROUPING_WITHOUT_QUERIES:
            return min(id, self.doc_count)
        if id < len(self.query_offsets):
            return Int(self.query_offsets[id])
        return len(self.query_ids)

    def get_query_size(self, id: Int) -> Int:
        """`GetQuerySize(ui32 id)`. Always 1 without queries
        (`samples_grouping.h:43-46`), `QuerySizes[id]` with (`:115-117`)."""
        if self.kind == GROUPING_WITHOUT_QUERIES:
            return 1
        return Int(self.query_sizes[id])

    def get_query_id(self, line: Int) -> Int:
        """`GetQueryId(size_t line)`. `Min(line, DocCount)` without queries
        (`samples_grouping.h:48-50`); with queries, `QueryIds[line]` in range
        and `QuerySizes.size()` -- the GROUP count -- out of it (`:119-121`).

        The two out-of-range answers are different quantities in the two
        implementations and in the two accessors above. Transcribed as
        written."""
        if self.kind == GROUPING_WITHOUT_QUERIES:
            return min(line, self.doc_count)
        if line < len(self.query_ids):
            return Int(self.query_ids[line])
        return len(self.query_sizes)

    def next_query_offset_for_line(self, line: Int) -> Int:
        """`NextQueryOffsetForLine(ui32 line)` -- THE off-by-one of this file.

        Without queries (`samples_grouping.h:52-54`):

            return Min<ui32>(line + 1, DocCount);

        so every fold boundary CreateFolds computes is one past the number it
        would otherwise be, right up to the clamp at `DocCount`. It is not a
        rounding step there; each document is its own group, so the offset of
        the next group after the group containing `line` is literally
        `line + 1`.

        With queries (`:123-129`):

            ui32 gid = GetQueryId(line);
            if (gid + 1 < QueryOffsets.size()) return QueryOffsets[gid + 1];
            return QueryIds.size();

        which snaps the boundary FORWARD to the start of the next group, so no
        fold ever splits a group. That is why `CreateFolds` calls this on
        every boundary it computes rather than using the geometric value
        directly.
        """
        if self.kind == GROUPING_WITHOUT_QUERIES:
            return min(line + 1, self.doc_count)
        var gid = self.get_query_id(line)
        if gid + 1 < len(self.query_offsets):
            return Int(self.query_offsets[gid + 1])
        return len(self.query_ids)


# ---------------------------------------------------------------------------
# `CeilDivide` (`cuda/utils/helpers.h:3-6`).
# ---------------------------------------------------------------------------


def ceil_divide(x: Int, y: Int) -> Int:
    """Their `CeilDivide` (`cuda/utils/helpers.h:3-6`), `(x + y - 1) / y`."""
    return (x + y - 1) // y


# ---------------------------------------------------------------------------
# `MinEstimationSize` (`dynamic_boosting.h:177-185`).
# ---------------------------------------------------------------------------

comptime MAX_FOLDS = 18
"""Their `const ui32 maxFolds = 18` (`dynamic_boosting.h:179`). It is a cap on
the fold COUNT, enforced by raising the first fold's size until the geometric
series fits under it."""


def min_estimation_size(doc_count: Int, min_fold_size: Int) -> Int:
    """Their `MinEstimationSize(ui32 docCount)` (`dynamic_boosting.h
    :177-185`), transcribed branch for branch:

        if (docCount < 500) {
            return 1;
        }
        const ui32 maxFolds = 18;
        const ui32 folds = NCB::IntLog2(NHelpers::CeilDivide(docCount,
                                                             Config.MinFoldSize));
        if (folds >= maxFolds) {
            return NHelpers::CeilDivide(docCount, 1 << maxFolds);
        }
        return Min<ui32>(Config.MinFoldSize, docCount / 50);

    Three arms, and the second is the one nothing ever runs by accident: it
    needs `ceil(docCount / min_fold_size) > 2^17`, i.e. `docCount >
    13,107,200` at the default `min_fold_size = 100`. Below that the answer is
    `min(min_fold_size, docCount / 50)`, which is `docCount / 50` until
    `docCount` reaches 5,000 and `min_fold_size` after.

    The `docCount < 500` arm is a CLIFF, not a taper: at 499 the first fold
    estimates on 1 document, at 500 on 10. That discontinuity is real in their
    code and is gated.

    `Config.MinFoldSize` is `min_fold_size` here rather than a field, because
    `TBoostingOptions` is not ported (`boosting_options.cpp:24` is its default,
    100, and it is a GPU-only option).
    """
    if doc_count < 500:
        return 1
    var folds = int_log2(ceil_divide(doc_count, min_fold_size))
    if folds >= MAX_FOLDS:
        return ceil_divide(doc_count, 1 << MAX_FOLDS)
    return min(min_fold_size, doc_count // 50)


# ---------------------------------------------------------------------------
# `CreateFolds` (`dynamic_boosting.h:189-223`).
# ---------------------------------------------------------------------------


def create_folds(
    sample_count: Int,
    growth_rate: Float64,
    samples_grouping: IQueriesGrouping,
    boosting_type: EBoostingType,
    min_fold_size: Int,
    dev_count: Int,
) raises -> List[TFold]:
    """Their `CreateFolds` (`dynamic_boosting.h:189-223`), transcribed.

    Their signature is `CreateFolds(ui32 sampleCount, double growthRate,
    const IQueriesGrouping&)`; `Config.BoostingType`, `Config.MinFoldSize` and
    `NCudaLib::GetCudaManager().GetDeviceCount()` are members and a global
    there, and are parameters here so that both sides of every branch can be
    reached from a check (`PORTING_RULES` rule 8). `growth_rate` is
    `Config.FoldLenMultiplier` at the one call site (`:594`).

    WHAT `growth_rate` ADMITS. `CB_ENSURE(growthRate > 1.0)` here (`:202`) and
    `CB_ENSURE(FoldLenMultiplier.Get() > 1.0f, "fold len multiplier should be
    greater than 1")` in the options (`boosting_options.cpp:64`). Any double
    strictly greater than 1.0, default 2.0. Not restricted to integers, not
    restricted to powers of two.

    THE MULTI-DEVICE ARM (`:194-198`). Skipped at one device, which is every
    run this repository makes. It raises the first fold to the offset of group
    `min(16 * devCount, queryCount / 2)` so that each device has several
    groups. Ported and parameterised so a check can run it; `dev_count` is
    `GetDeviceCount()`.

    THE PLAIN ARM (`:204-208`). One fold, both slices `[0, sampleCount)`. Note
    where it sits: AFTER all four `CB_ENSURE`s, so a Plain fit still refuses a
    pool of fewer than `4 * devCount` groups and still refuses
    `growthRate <= 1.0`. It is also the arm every fit in this tree takes
    today, since `TOptimizationSubsets` is built by the `TStripeMapping`
    specialization with `FoldCount = 0`.
    """
    var min_estimation = samples_grouping.next_query_offset_for_line(
        min_estimation_size(sample_count, min_fold_size)
    )
    # `const ui32 devCount = NCudaLib::GetCudaManager().GetDeviceCount();`
    # we should have at least several queries per devices
    if dev_count > 1:
        min_estimation = max(
            min_estimation,
            samples_grouping.get_query_offset(
                min(16 * dev_count, samples_grouping.get_query_count() // 2)
            ),
        )

    if samples_grouping.get_query_count() < 4 * dev_count:
        raise Error(
            "Error: pool has just "
            + String(samples_grouping.get_query_count())
            + " groups or docs, can't use #"
            + String(dev_count)
            + " GPUs to learn on such small pool"
        )
    if min_estimation == 0:
        raise Error("Error: min learn size should be positive")
    if not (growth_rate > 1.0):
        raise Error("Error: grow rate should be > 1.0")

    var folds = List[TFold]()
    if boosting_type == EBoostingType.Plain:
        folds.append(TFold(TSlice(0, sample_count), TSlice(0, sample_count)))
        return folds^

    var test_end = samples_grouping.next_query_offset_for_line(
        min(Int(Float64(min_estimation) * growth_rate), sample_count)
    )
    folds.append(
        TFold(
            TSlice(0, min_estimation),
            TSlice(min_estimation, test_end),
        )
    )

    # DEVIATION 112: `max_iterations` is ours. Their loop has no bound.
    var max_iterations = sample_count + 2
    var iterations = 0
    while folds[len(folds) - 1].quality_evaluate_samples.right < sample_count:
        iterations += 1
        if iterations > max_iterations:
            raise Error(
                "DEVIATION 112: CreateFolds ran "
                + String(iterations)
                + " growth passes over "
                + String(sample_count)
                + " samples; NextQueryOffsetForLine is not strictly"
                + " increasing below sampleCount"
            )
        var right = folds[len(folds) - 1].quality_evaluate_samples.right
        var learn_slice = TSlice(0, right)
        var end = samples_grouping.next_query_offset_for_line(
            min(Int(Float64(right) * growth_rate), sample_count)
        )
        var test_slice = TSlice(right, end)
        folds.append(TFold(learn_slice, test_slice))
    return folds^


# ---------------------------------------------------------------------------
# The two lines that turn a fold list into the subsets' fold axis.
# `oblivious_tree_structure_searcher.cpp:36-37`, whose `initParts` is
# `WriteFoldBasedInitialBins` (`:338-364`) or `WriteSingleTaskInitialBins`
# (`:366-376`).
# ---------------------------------------------------------------------------


def fold_count_for_folds(fold_list_size: Int, boosting_type: EBoostingType) -> Int:
    """`subsets.FoldCount = initParts.size()`
    (`oblivious_tree_structure_searcher.cpp:36`).

    Ordered: `WriteFoldBasedInitialBins` (`:346-362`) pushes TWO
    `TDataPartition`s per fold and advances `currentBin` by 2, so bin `2k` is
    fold `k`'s estimate half and bin `2k + 1` its quality-evaluate half.
    `initParts.size() == 2 * folds.size()`.

    Plain: `SingleTaskTarget` is set instead (`dynamic_boosting.h:305-313`
    calls `SetTarget`, not `AddTask`), so `CreateSubsets` takes
    `WriteSingleTaskInitialBins` (`:31-32`), which pushes ONE part and fills
    every bin with 0. `initParts.size() == 1`.

    The DEVICE half of both -- `bins.Reset(...)` and the `FillBuffer` calls --
    is not ported here. Only this count is, because it is what
    `TOptimizationSubsets.fold_count` and `fold_bits` are computed from.
    """
    if boosting_type == EBoostingType.Plain:
        return 1
    return 2 * fold_list_size


def fold_bits_for_folds(fold_list_size: Int, boosting_type: EBoostingType) -> Int:
    """`subsets.FoldBits = NCB::IntLog2(subsets.FoldCount)`
    (`oblivious_tree_structure_searcher.cpp:37`).

    This is the number the whole of `PORTING.md` 91 B is about: the fold id
    lives in bits `[0, FoldBits)` of a document's bin and the depth bits sit
    above it, which is why every index downstream reads
    `CurrentDepth + FoldBits`. `1 << FoldBits` is also the stripe
    `PointwisePartOffsetsHelper.data_partition_offset` rounds the fold axis up
    to (`kernel/split_properties_helpers.mojo:100-118`), and it is `>=`
    `FoldCount` with slack whenever `FoldCount` is not a power of two -- the
    usual case, since the fold count is `2 * ceil(log_g(n / m))`.

    Plain gives `IntLog2(1) == 0`, which is the `FoldBits = 0` that
    `pointwise_optimization_subsets.cpp:13` hard-codes for the `TStripeMapping`
    specialization. The two agree, and that agreement is a gate.
    """
    return int_log2(fold_count_for_folds(fold_list_size, boosting_type))


# ---------------------------------------------------------------------------
# `TFoldAndPermutationStorage` (`dynamic_boosting.h:97-119`).
# ---------------------------------------------------------------------------


struct TFoldAndPermutationStorage[TData: Copyable & Deinitable](Movable):
    """Their `TFoldAndPermutationStorage<TData>` (`dynamic_boosting.h
    :97-119`).

        TVector<TVector<TData>> FoldData;   // [permutation][fold]
        TData Estimation;                   // one extra, off the grid

    Used twice with two different `TData`: as `TCursor =
    TFoldAndPermutationStorage<TVec>` (`:233`), the per-fold approx cursors,
    and as `TFoldAndPermutationStorage<TWeakModel>` (`:361`), the per-fold
    weak models of one iteration. The shape is the same both times and it is
    the shape this file exists to size: `FoldData[p]` has one entry per fold
    of permutation `p`, `Estimation` has none.

    `Foreach` (`:114-118`) is NOT ported. It visits every fold entry and then
    `Estimation`, and its two call sites sweep DEVICE buffers
    (`dynamic_boosting_progress.h`'s save/load). There are no device buffers
    on this path; `permutation_count`, `fold_count_for_permutation` and `get`
    describe the same traversal for a host caller.
    """

    var fold_data: List[List[Self.TData]]
    var estimation: Self.TData

    def __init__(
        out self,
        var fold_data: List[List[Self.TData]],
        var estimation: Self.TData,
    ):
        """Their two-argument constructor (`dynamic_boosting.h:101-106`)."""
        self.fold_data = fold_data^
        self.estimation = estimation^

    def permutation_count(self) -> Int:
        """`FoldData.size()`, which `:576` resizes to
        `learnPermutationCount`."""
        return len(self.fold_data)

    def fold_count_for_permutation(self, permutation_id: Int) -> Int:
        """`FoldData[permutationId].size()`, which `:597` resizes to
        `folds.size()` for that permutation."""
        return len(self.fold_data[permutation_id])

    def get(self, permutation_id: Int, fold_id: Int) -> Self.TData:
        """Their `Get(ui32 permutationId, ui32 foldId)` (`:108-110`),
        `FoldData.at(permutationId).at(foldId)`. Theirs returns a reference
        and both `.at()` calls are bounds-checked; ours returns an explicit
        `.copy()` -- `List` is not implicitly copyable in Mojo 1.0 -- and
        `List.__getitem__` is the bounds check.
        """
        return self.fold_data[permutation_id][fold_id].copy()


# ---------------------------------------------------------------------------
# The permutation counts the fold grid is sized by.
# `dynamic_boosting.h:283-289` and `:573-575`.
# ---------------------------------------------------------------------------


def estimation_permutation(permutation_count: Int) -> Int:
    """`const ui32 estimationPermutation = permutationCount - 1;`
    (`dynamic_boosting.h:285`, and identically `:573`).

    The LAST permutation is the estimation permutation and is never used to
    search for structure. `Config.PermutationCount` defaults to 4
    (`boosting_options.cpp:14`), so this is 3.
    """
    return permutation_count - 1


def learn_permutation_count(permutation_count: Int) -> Int:
    """`const ui32 learnPermutationCount = estimationPermutation
    ? permutationCount - 1 : 1;  //fallback` (`dynamic_boosting.h:286`).

    `:574` spells the same value as `estimationPermutation ?
    estimationPermutation : 1`, which is equal because `estimationPermutation
    == permutationCount - 1`. Both spellings transcribed; this is the first.

    This is the number of rows `PermutationFolds` and `Cursor.FoldData` are
    resized to (`:576-578`), so `create_folds` is called exactly this many
    times per fit. At the default 4 it is 3.
    """
    var estimation = estimation_permutation(permutation_count)
    if estimation != 0:
        return permutation_count - 1
    return 1


def learn_permutation_id(random_value: Int, learn_permutation_count_in: Int) -> Int:
    """`dynamic_boosting.h:287-289`, transcribed AS WRITTEN:

        const ui32 learnPermutationId = learnPermutationCount > 1
            ? static_cast<const ui32>(Random.NextUniformL() %
                                      (learnPermutationCount - 1))
            : 0;

    THE MODULUS IS `learnPermutationCount - 1`, NOT `learnPermutationCount`.
    At the default `permutation_count = 4` that is `3 - 1 = 2`, so structure
    search only ever runs on permutations 0 and 1. Permutation 2 has folds
    built for it (`:582-604`), has cursors allocated for it, and IS used --
    every fit iteration estimates leaf values on all three (`:378-396`) and
    adds the model back to all three (`:447-465`). It is only the STRUCTURE
    SEARCH that never sees it.

    This is transcribed, not corrected. `PORTING_RULES` 0b: copy, do not
    improve. Whether the `- 1` is deliberate or a typo in their tree is not
    this port's question to answer, and a port that quietly used
    `% learnPermutationCount` would train a different model from CatBoost on
    the default configuration.
    """
    if learn_permutation_count_in > 1:
        return random_value % (learn_permutation_count_in - 1)
    return 0
