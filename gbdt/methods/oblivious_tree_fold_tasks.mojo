# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The FOLD side of the oblivious searcher: N tasks, 2N partitions.

PORT OF `catboost/cuda/methods/oblivious_tree_structure_searcher.{h,cpp}` at
CatBoost `54a8143a` -- `TOptimizationTask`, `AddTask`/`SetTarget`,
`WriteFoldBasedInitialBins` (`:338-364`), `WriteSingleTaskInitialBins`
(`:366-...`) and `ForeachOptimizationPartTask` (`:15-27`). Transliterated.
Do not improve.

**THIS IS RUNG 2, and rung 2 turned out not to be a second searcher.**

`NEXT_TWO.md` priced it as porting `TFeatureParallelObliviousTreeSearcher`,
713 lines, beside the doc-parallel one already here. Reading it says
otherwise: their searcher is ONE object with TWO modes on it
(`:88-100`) --

    SetTarget(target)              SingleTaskTarget   ONE task, the PLAIN arm
    AddTask(learnTarget, testTarget)  FoldBasedTasks  N pairs, the ORDERED arm

and `CreateSubsets` picks between them with a single ternary (`:30-31`):

    SingleTaskTarget == nullptr ? WriteFoldBasedInitialBins(subsets.Bins)
                                : WriteSingleTaskInitialBins(subsets.Bins)

Everything after that line -- the depth loop, the histograms, the scorer, the
`TakeBest` fold, the split -- is the same code for both arms. Combined with
`PORTING.md` 91 A (the two data layouts build a bit-identical compressed
index at device count 1) and 91 B (the two searchers share their entire
stack), what rung 2 actually costs is THIS FILE plus wiring it, not 713
lines.

## THE ENCODING, which is the whole thing

`WriteFoldBasedInitialBins` walks the tasks and, per task `k`:

    learn slice -> bin  2k
    test  slice -> bin  2k + 1
    parts.push_back({cursor, learn size});  cursor += learn size
    parts.push_back({cursor, test  size});  cursor += test  size

So N tasks give **2N partitions**, alternating learn and test, over ONE
concatenated document array. `FoldCount` is `parts.size()` = 2N and
`FoldBits` is `IntLog2(2N)` -- and the fold id therefore occupies the LOW
bits of a document's bin, with depth bits above it, which is why every
downstream index reads `CurrentDepth + FoldBits`.

**THE PAIRING IS WHY THE DYNAMIC SCORER STEPS FOLDS BY TWO.**
`find_optimal_split_solar_kernel` and the dynamic cosine one read
`(fold, fold + 1)` as `(estimate, test)`
(`gbdt/methods/kernel/pointwise_scores.mojo`, ported and gated before this
file existed). Fold `2k` is task `k`'s ESTIMATE half and `2k + 1` is its TEST
half. The two halves of ordered boosting meet exactly here: this file lays
the pairs out and that kernel consumes them.

## What this file is NOT

It does not create the folds -- `CreateFolds` (`dynamic_boosting.h:189-223`)
decides how many and how long, and is `gbdt/methods/dynamic_boosting_folds.mojo`.
It does not run a tree. It lays out the initial bins and partitions that a
multi-task tree grows from, and at one task with no test half it reduces to
what `create_subsets` already does.

DEVIATION 119: their `ForeachOptimizationPartTask` runs the per-task fills on
up to 8 CUDA STREAMS (`RunInStreams(tasks.size(), Min<ui32>(tasks.size(), 8),
...)`). There are no streams on Metal, so the walk is sequential. It changes
nothing: the cursor arithmetic is serial in their code too -- `cursor` and
`currentBin` are captured by reference and advanced in task order -- so the
streams only overlap the FILLS, never the layout.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from gbdt.gpu_util.gpu_data.partitions import DataPartition
from gbdt.methods.dynamic_boosting_folds import TFold
from gbdt.methods.pointwise_optimization_subsets import (
    TL2Target,
    TOptimizationSubsets,
    create_subsets,
    update_subsets_stats,
)


@fieldwise_init
struct FoldTask(Copyable, ImplicitlyCopyable, Movable):
    """`TOptimizationTask` (`oblivious_tree_structure_searcher.h:112`),
    reduced to what the bin layout needs.

    Theirs holds two `TTarget` objects; the layout only ever asks them for
    `GetTarget().GetIndices().GetObjectsSlice().Size()`, so this carries the
    two sizes. A task that owns targets belongs with the boosting loop that
    creates them.
    """

    var learn_size: Int
    var test_size: Int


@fieldwise_init
struct FoldLayout(Copyable, Movable):
    """What `WriteFoldBasedInitialBins` returns, plus what it wrote."""

    var parts: List[DataPartition]
    """`2 * len(tasks)` partitions, alternating LEARN then TEST."""

    var total_indices: Int
    """`GetTotalIndicesSize()`: the concatenated document count."""

    var fold_count: Int
    """`subsets.FoldCount = initParts.size()` (`:36`)."""

    var fold_bits: Int
    """`subsets.FoldBits = IntLog2(subsets.FoldCount)` (`:37`), and
    `IntLog2` is CEIL (`libs/helpers/math_utils.h:14-16`) -- the same
    ceiling that `PORTING.md` 107 records costing a day when it was read as
    floor."""


def int_log2_ceil(v: Int) -> Int:
    """`NCB::IntLog2`, `(ui32)ceil(log2(values))`."""
    var bit = 0
    while (1 << bit) < v:
        bit += 1
    return bit


def plan_fold_layout(tasks: List[FoldTask]) raises -> FoldLayout:
    """`WriteFoldBasedInitialBins` (`:338-364`), the HOST half.

    Returns the partition list and the counters; the device fill is
    `write_fold_based_initial_bins` below, which takes this and writes the
    bins. Split so the LAYOUT -- where every off-by-one lives -- is a pure
    function that gates without a GPU.
    """
    if len(tasks) == 0:
        raise Error(
            "WriteFoldBasedInitialBins with no tasks: their `AddTask` is"
            " called at least once before `Fit`, and `Fit` itself opens"
            " with CB_ENSURE(FoldBasedTasks.size() || SingleTaskTarget)"
        )
    var parts = List[DataPartition]()
    var cursor = 0
    for i in range(len(tasks)):
        # LEARN first, then TEST, and the order is load-bearing: the
        # dynamic scorer reads `(fold, fold + 1)` as `(estimate, test)`.
        parts.append(DataPartition(UInt32(cursor), UInt32(tasks[i].learn_size)))
        cursor += tasks[i].learn_size
        parts.append(DataPartition(UInt32(cursor), UInt32(tasks[i].test_size)))
        cursor += tasks[i].test_size
    var fc = len(parts)
    return FoldLayout(parts^, cursor, fc, int_log2_ceil(fc))


def plan_single_task_layout(n_rows: Int) raises -> FoldLayout:
    """`WriteSingleTaskInitialBins` (`:366-...`): every document in bin 0,
    ONE partition, and no test half.

    This is the PLAIN arm and it is what every fit in this repository takes
    today. `FoldCount` is 1 and `FoldBits` is 0, which is why
    `PointwisePartOffsetsHelper`'s two offsets coincide on the shipped path
    and the fold stripe costs nothing until this file's other function is
    used.
    """
    var parts = List[DataPartition]()
    parts.append(DataPartition(UInt32(0), UInt32(n_rows)))
    return FoldLayout(parts^, n_rows, 1, 0)


def write_fold_based_initial_bins(
    ctx: DeviceContext,
    layout: FoldLayout,
    mut bins: DeviceBuffer[DType.uint32],
) raises:
    """The device half: `FillBuffer(learnBins, currentBin)` /
    `FillBuffer(testBins, currentBin + 1)` per task (`:354-355`).

    One memset per partition rather than one per task, which is the same
    writes in the same places -- their two fills per task ARE two contiguous
    ranges, and the partition list already names them.
    """
    # **THE STAGING BUFFERS ARE HELD UNTIL AFTER `synchronize`, and that is
    # not tidiness.** Mojo frees a `DeviceBuffer` at its LAST USE
    # ([[mojo-buffer-freed-at-last-use]]), and a buffer handed to
    # `enqueue_copy` as `src_buf` is last used at the ENQUEUE, not at the
    # copy. Dropping it at the end of the loop body lets the next
    # iteration's `enqueue_create_buffer` land on the same block and the
    # next `enqueue_memset` overwrite the previous partition's copy source
    # while that copy is still pending. The failure is INTERMITTENT and it
    # looks like a partition array of zeros -- which is what it looked like
    # here, on the second run of a gate that had been green.
    var staging = List[DeviceBuffer[DType.uint32]]()
    for p in range(len(layout.parts)):
        ref part = layout.parts[p]
        if part.size == 0:
            continue
        # bin `p` is `currentBin` for an even p and `currentBin + 1` for an
        # odd one, because they advance `currentBin += 2` per task and this
        # list is two entries per task in the same order
        var view = ctx.enqueue_create_buffer[DType.uint32](Int(part.size))
        ctx.enqueue_memset(view, UInt32(p))
        ctx.enqueue_copy(
            dst_ptr=bins.unsafe_ptr().unsafe_offset(Int(part.offset)),
            src_buf=view,
        )
        staging.append(view^)
    ctx.synchronize()
    _ = staging^


# ---------------------------------------------------------------------------
# THE WIRING: turning a fold list into tasks, doc ids, and subsets.
# `dynamic_boosting.h:313-329` (AddTask), `MakeDocIndices`
# (`oblivious_tree_structure_searcher.cpp:485-505`) and `CreateSubsets`
# (`:29-43`).
# ---------------------------------------------------------------------------


def fold_tasks_from_folds(folds: List[TFold]) raises -> List[FoldTask]:
    """Their `AddTask` loop (`dynamic_boosting.h:313-329`), sizes only.

        for (ui32 foldId = 0; foldId < taskFolds.size(); ++foldId) {
            const auto& fold = taskFolds[foldId];
            auto learnTarget    = Create(taskTarget, fold.EstimateSamples, ...);
            auto validateTarget = Create(taskTarget, fold.QualityEvaluateSamples, ...);
            optimizer.AddTask(std::move(learnTarget), std::move(validateTarget));
        }

    ONE TASK PER FOLD, learn = `EstimateSamples`, test =
    `QualityEvaluateSamples`, in fold order. `plan_fold_layout` then turns
    N tasks into 2N partitions.

    **THE ESTIMATE SLICES ARE NESTED PREFIXES, so the concatenated document
    array is LONGER THAN THE DATASET and every document appears in several
    folds.** `CreateFolds` builds fold `k`'s `EstimateSamples` as
    `[0, right_{k-1})` (`dynamic_boosting.h:215-222`), so the total index
    size grows roughly as `sum_k right_k`, not as `n`. That is not a bug to
    be normalised away: it is what ordered boosting IS -- one copy of the
    document per fold, each carrying that fold's own cursor value.
    """
    var tasks = List[FoldTask]()
    for i in range(len(folds)):
        ref f = folds[i]
        var learn = f.estimate_samples.right - f.estimate_samples.left
        var test = (
            f.quality_evaluate_samples.right
            - f.quality_evaluate_samples.left
        )
        if learn < 0 or test < 0:
            raise Error(
                "fold " + String(i) + " has a negative slice; TSlice is"
                " half-open and Y_ASSERT(left <= right)"
            )
        tasks.append(FoldTask(learn, test))
    return tasks^


def make_fold_doc_indices(
    folds: List[TFold], permutation: List[UInt32] = List[UInt32]()
) raises -> List[UInt32]:
    """`TFeatureParallelObliviousTreeSearcher::MakeDocIndices`, fold arm
    (`oblivious_tree_structure_searcher.cpp:485-505`).

        indices.Reset(TMirrorMapping(GetTotalIndicesSize()));
        ForeachOptimizationPartTask(FoldBasedTasks, [&](learnSlice, testSlice, task, stream) {
            indices.SliceView(learnSlice).Copy(task.LearnTarget->GetTarget().GetIndices(), stream);
            indices.SliceView(testSlice).Copy(task.TestTarget->GetTarget().GetIndices(), stream);
        });

    A task's `GetIndices()` is the permuted document id array RESTRICTED to
    that task's slice of the learn permutation, so the concatenation is
    `permutation[fold.EstimateSamples] ++ permutation[fold.QualityEvaluateSamples]`
    for every fold in order. An empty `permutation` is the identity, which
    is `TDataPermutation` at permutation id 0 with no shuffle.

    **THIS IS THE ARRAY THE COMPRESSED INDEX IS READ THROUGH, and it is the
    one thing the doc-parallel searcher's DEVIATION 105 identity assumption
    stops being true for.** At one task `docs == subsets.Indices`; at N
    tasks `docs[i] == docIndices[subsets.Indices[i]]`, and the gather is not
    optional -- position `i` of the concatenated array and document id `i`
    are different numbers the moment a second fold exists.
    """
    var out = List[UInt32]()
    for i in range(len(folds)):
        ref f = folds[i]
        for p in range(f.estimate_samples.left, f.estimate_samples.right):
            out.append(
                UInt32(p) if len(permutation) == 0 else permutation[p]
            )
        for p in range(
            f.quality_evaluate_samples.left,
            f.quality_evaluate_samples.right,
        ):
            out.append(
                UInt32(p) if len(permutation) == 0 else permutation[p]
            )
    return out^


def create_fold_based_subsets(
    ctx: DeviceContext,
    max_depth: Int,
    mut source: TL2Target,
    layout: FoldLayout,
) raises -> TOptimizationSubsets:
    """`TFeatureParallelObliviousTreeSearcher::CreateSubsets`
    (`oblivious_tree_structure_searcher.cpp:29-43`), the FOLD arm.

        auto initParts = SingleTaskTarget == nullptr
                       ? WriteFoldBasedInitialBins(subsets.Bins)
                       : WriteSingleTaskInitialBins(subsets.Bins);
        subsets.Indices = TMirrorBuffer<ui32>::CopyMapping(subsets.Bins);
        subsets.CurrentDepth = 0;
        subsets.FoldCount = initParts.size();
        subsets.FoldBits  = NCB::IntLog2(subsets.FoldCount);
        MakeSequence(subsets.Indices);
        ui32 maxPartCount = 1 << (subsets.FoldBits + maxDepth);
        subsets.Partitions.Reset(...maxPartCount);
        subsets.PartitionStats.Reset(...maxPartCount);
        UpdateSubsetsStats(src, &subsets);

    Everything but the initial bin fill is already `create_subsets`, which
    takes `fold_count` and `fold_bits` for exactly this reason
    (`pointwise_optimization_subsets.mojo:1252`). This function is the
    ternary and the fill.

    **THE INDICES STAY THE IDENTITY AND THAT IS CORRECT.** Their
    `MakeSequence` runs AFTER the bins are written and no sort follows, so
    the document array is grouped by bin only because
    `WriteFoldBasedInitialBins` lays the partitions down in bin order --
    `parts[p]` is exactly `[cursor_p, cursor_p + size_p)` and bin `p` is
    written over that same range. `UpdatePartitionDimensions` then reads
    monotone bins and recovers the offsets it was just given. Write the
    test half of a task before its learn half and the bins stop being
    monotone, the partition scan returns garbage offsets, and every
    histogram reads the wrong span.

    DEVIATION 125: ONE EXTRA `UpdateSubsetsStats`. Theirs writes the bins
    BEFORE the single `UpdateSubsetsStats` at the end of `CreateSubsets`;
    ours calls `create_subsets` (which zeroes the bins and reduces once)
    and then overwrites the bins and reduces AGAIN. Same final state, one
    extra partition reduce over `1 << fold_bits` parts per TREE -- not per
    level. It is priced this way rather than fixed by splitting
    `create_subsets` in two because that file is shared with the
    doc-parallel arm and a second entry point into it would fork the
    allocation.
    """
    if layout.fold_count < 1:
        raise Error("FoldLayout.fold_count must be at least 1")
    if len(layout.parts) != layout.fold_count:
        raise Error(
            "FoldLayout is inconsistent: fold_count is "
            + String(layout.fold_count)
            + " but there are "
            + String(len(layout.parts))
            + " partitions; `subsets.FoldCount = initParts.size()`"
            " (`oblivious_tree_structure_searcher.cpp:36`)"
        )
    if layout.fold_bits != int_log2_ceil(layout.fold_count):
        raise Error(
            "FoldLayout.fold_bits is not IntLog2(fold_count); IntLog2 is"
            " CEIL (`libs/helpers/math_utils.h:14-16`)"
        )
    # THE SAME CLAIM SAID WITHOUT `int_log2_ceil`, and it is not
    # redundant: the line above compares the layout against the very
    # function that produced it, so a FLOOR `IntLog2` agrees with itself
    # and gets through. `PORTING.md` 107 already cost a day to that
    # reading. With FoldBits 3 and FoldCount 12 the fold stripe is 8, the
    # partition ids run past `1 << (FoldBits + maxDepth)`, and the failure
    # is an out-of-range partition write rather than a wrong answer -- the
    # sabotage HUNG instead of going red, which is exactly the shape
    # DEVIATION 112 exists to prevent.
    if layout.fold_count > (1 << layout.fold_bits):
        raise Error(
            "FoldCount "
            + String(layout.fold_count)
            + " does not fit in FoldBits "
            + String(layout.fold_bits)
            + ": the fold stripe `1 << FoldBits` is "
            + String(1 << layout.fold_bits)
            + " and every partition id is `part * stripe + fold`"
            " (`split_properties_helpers.cuh:78-81`), so a fold id at or"
            " above the stripe writes into the next leaf's slot."
            " `NCB::IntLog2` is CEIL."
        )
    if layout.total_indices != source.line_size:
        raise Error(
            "the weak target has "
            + String(source.line_size)
            + " entries but the fold layout concatenates "
            + String(layout.total_indices)
            + "; `ComputeWeakTarget`'s fold arm sizes"
            " WeightedTarget/Weights at `slices.back().Right`"
            " (`oblivious_tree_structure_searcher.cpp:400-401`)"
        )
    if layout.fold_bits + max_depth >= 32:
        raise Error(
            "1 << (FoldBits + maxDepth) does not fit a ui32 bin;"
            " ReorderBins asserts (offset + bits) <= 32"
        )

    var subsets = create_subsets(
        ctx, max_depth, source, layout.fold_count, layout.fold_bits
    )
    # the ternary's fold arm (`:30-31`)
    write_fold_based_initial_bins(ctx, layout, subsets.bins)
    # DEVIATION 125, above.
    update_subsets_stats(ctx, source, subsets)
    return subsets^
