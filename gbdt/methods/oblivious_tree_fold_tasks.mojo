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
    ctx.synchronize()
