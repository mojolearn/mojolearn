# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The offset arithmetic every POINTWISE histogram kernel shares.

PORT OF `catboost/cuda/methods/kernel/split_properties_helpers.cuh` at
CatBoost `54a8143a`. Transliterated. Do not improve.

This file belongs to CatBoost's OTHER histogram family. `PORTING.md` 91 B
lays out which is which; the short version is that CatBoost has three GPU
tree searchers and two histogram families:

    greedy_subsets_searcher/kernel/   ->  ported, drives our symmetric trees,
                                          and is what CatBoost runs for
                                          MULTICLASS symmetric trees
    methods/kernel/pointwise_hist2*   ->  THIS one, shared by BOTH of their
                                          oblivious searchers, and what
                                          CatBoost runs for single-target
                                          symmetric trees in either boosting
                                          mode

They are not configurations of each other. The greedy-subsets family is
organised around a SET of leaves carried in `TPointsSubsets`; this one is
organised around an oblivious LEVEL, where the leaves of a level are a
contiguous power-of-two block of partitions and the sibling of part `p` at
depth `d` is `p | (1 << d)`. That structural difference is what the offset
helpers below encode, and it is why the two families cannot share code even
though they compute the same sums.

WHAT IS AND IS NOT IN THIS PORT OF THE FILE
-------------------------------------------
Ported here: `TPointwisePartOffsetsHelper`, the host-side
`EstimateBlockPerFeatureMultiplier` and `HasOneHotFeatures`, `ELoadType`,
and `ScanHistogramsImpl`.

NOT ported, because they belong to the PAIRWISE family which this
repository does not build (`gbdt/UNPORTED.tsv` already carries that family):
`ConvertBlockToPart`, `GetPairwisePartIdToCalculate`,
`TCmpBinsWithoutOneHot`, `TCmpBinsWithOneHot`, `TCmpBinsOneByteTrait`.

Their device-side `GetMaxBinCount` and `HasOneHotFeatures` are deliberately
held until the kernels that call them land. Both reduce over exactly FOUR
shared-memory slots (`:31-40`, `:52-61`) while every thread in the block
writes one, so their contract depends on how many features the calling
kernel puts in a block -- and a helper ported without its call site is a
helper ported from a guess.
"""

from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.math import ceil, log2


# ---------------------------------------------------------------------------
# `ELoadType` (`:281-285`)
# ---------------------------------------------------------------------------

comptime LOAD_ONE_ELEMENT = 0
comptime LOAD_TWO_ELEMENTS = 1
comptime LOAD_FOUR_ELEMENTS = 2


# ---------------------------------------------------------------------------
# `TPointwisePartOffsetsHelper` (`:68-104`)
# ---------------------------------------------------------------------------


@fieldwise_init
struct PointwisePartOffsetsHelper(Copyable, ImplicitlyCopyable, Movable):
    """`TPointwisePartOffsetsHelper`, field for field: one `ui32 FoldCount`.

    THE TWO OFFSETS ARE NOT THE SAME FUNCTION, and that is the whole point
    of the struct. A histogram is indexed densely by `(part, fold)`:

        GetHistogramOffset(partId, foldId) = partId * FoldCount + foldId

    while a DATA PARTITION is indexed by `(part, fold)` with the fold axis
    rounded UP to a power of two:

        foldStripe = 1 << ceil(log2(FoldCount))
        GetDataPartitionOffset(partId, foldId) = partId * foldStripe + foldId

    The stripe exists because the fold id is packed into the LOW BITS of a
    document's bin (`PORTING.md` 91 B), so partitions have to be addressable
    by masking rather than by dividing. Histograms carry no such constraint
    and are packed tight. Reading either function for the other is a
    silent, data-dependent corruption whenever `FoldCount` is not already a
    power of two -- which, since their folds grow geometrically from
    `min_fold_size` by `fold_len_multiplier`, is the usual case.

    At `FoldCount == 1` -- Plain boosting, which is every fit this
    repository does today -- both reduce to `partId` and the distinction
    costs nothing. It is transcribed anyway because rung 3 turns it on.
    """

    var fold_count: UInt32

    @always_inline
    def histogram_offset(self, part_id: UInt32, fold_id: UInt32) -> UInt32:
        """`GetHistogramOffset` (`:74-76`)."""
        return part_id * self.fold_count + fold_id

    @always_inline
    def data_partition_offset(
        self, part_id: UInt32, fold_id: UInt32
    ) -> UInt32:
        """`GetDataPartitionOffset` (`:78-81`), including their float `log2`.

        Theirs is
            `1 << (ui32)ceil(log2((float)FoldCount))`
        and this is the same expression in the same precision. It is NOT
        rewritten as a bit trick: `ceil(log2(x))` on a float is not
        `32 - clz(x - 1)` at every input, and the two disagree exactly at
        the powers of two, where `log2` is the value most likely to land a
        hair below the integer it should be. Their expression is the
        contract; a faster one that agrees on the cases we happen to run is
        a different contract.
        """
        var fold_stripe = UInt32(
            1
        ) << UInt32(ceil(log2(Float32(self.fold_count))))
        return part_id * fold_stripe + fold_id


@fieldwise_init
struct ShiftedPointers(Copyable, ImplicitlyCopyable, Movable):
    """What `ShiftPartAndBinSumsPtr` (`:83-102`) does to its two arguments.

    Theirs takes both pointers by mutable reference and advances them.
    Mojo has no reference-to-pointer argument, so this returns the two
    OFFSETS instead and the caller adds them. Same arithmetic, same order;
    the only thing that changes is who does the addition.
    """

    var partition_offset: UInt32
    var bin_sums_offset: UInt32


@always_inline
def shift_part_and_bin_sums_ptr[
    origin: MutOrigin, //
](
    partition_sizes: MutPointer[UInt32, origin],
    fold_count: UInt32,
    block_y: UInt32,
    block_z: UInt32,
    grid_y: UInt32,
    total_feature_count: UInt32,
    full_pass: Bool,
    hist_count: UInt32 = 2,
) -> ShiftedPointers:
    """`ShiftPartAndBinSumsPtr` (`:83-102`), copied.

    THE `else` ARM IS THE WHOLE SIBLING-SUBTRACTION TRICK, expressed as
    addressing rather than as a separate kernel. On a partial pass the
    block looks at BOTH children of its pair --

        left  = GetDataPartitionOffset(blockIdx.y, blockIdx.z)
        right = GetDataPartitionOffset(gridDim.y | blockIdx.y, blockIdx.z)

    -- reads their sizes, and points the DATA at whichever is smaller while
    pointing the HISTOGRAM unconditionally at the RIGHT child's slot. So
    the kernel computes the cheaper child's histogram and files it under the
    right child's name, and a later subtraction fixes up the sign and the
    other sibling. `gridDim.y | blockIdx.y` rather than `+` is theirs, and
    it is exact rather than lucky: on a partial pass `gridDim.y` is the
    number of leaves at this depth, a power of two strictly greater than
    every `blockIdx.y`, so the OR sets the depth bit and nothing else.

    `partition_sizes` is the partitions array reinterpreted as `UInt32`,
    because a Mojo kernel argument cannot be a pointer to a non-trivial
    struct. Their `TDataPartition` is `{ui32 Offset; ui32 Size;}`, so part
    `p`'s SIZE is at `2 * p + 1` and its offset at `2 * p`. The `+ 1` is
    load-bearing and reading it off by one is not visible in half the
    cases: gate P3 in `mojo_only/pointwise_offsets_check.mojo` has one pair
    whose smaller child is on the left and one whose smaller child is on
    the right precisely because reading `Offset` instead of `Size` gets the
    first pair RIGHT -- offsets ascend with the part id, so they order the
    same way sizes do whenever the smaller child happens to come first.
    That is how this line was wrong when it was written and how the gate
    caught it.
    """
    var helper = PointwisePartOffsetsHelper(fold_count)
    var hist_line_size = hist_count * total_feature_count

    if full_pass:
        return ShiftedPointers(
            helper.data_partition_offset(block_y, block_z),
            helper.histogram_offset(block_y, block_z) * hist_line_size,
        )

    var left_part_offset = helper.data_partition_offset(block_y, block_z)
    var right_part_offset = helper.data_partition_offset(
        grid_y | block_y, block_z
    )
    var left_part_size = partition_sizes.unsafe_load(
        Int(left_part_offset) * 2 + 1
    )
    var right_part_size = partition_sizes.unsafe_load(
        Int(right_part_offset) * 2 + 1
    )

    var chosen = (
        left_part_offset if left_part_size
        < right_part_size else right_part_offset
    )
    return ShiftedPointers(
        chosen,
        hist_line_size * helper.histogram_offset(grid_y | block_y, block_z),
    )


# ---------------------------------------------------------------------------
# host-side helpers
# ---------------------------------------------------------------------------


def estimate_block_per_feature_multiplier(
    num_blocks_x: Int,
    num_blocks_y: Int,
    num_blocks_z: Int,
    ds_size: Int,
    sm_count: Int,
    limit: Int = 128,
) -> Int:
    """`EstimateBlockPerFeatureMultiplier` (`:15-23`), copied.

        int blocksPerSm = TArchProps::GetMajorVersion() < 5 ? 1 : 2;
        ui32 multiplier = 1;
        while ((numBlocks.x * numBlocks.y * min(numBlocks.z, 8) * multiplier
                  < TArchProps::SMCount() * blocksPerSm * 1.25) &&
               ((dsSize / multiplier) > 10000) && (multiplier < limit)) {
            multiplier *= 2;
        }

    This is how the pointwise family fills a machine that the feature axis
    alone cannot: when there are too few (feature block x leaf x fold)
    blocks to occupy the device, it splits the DOCUMENT axis `multiplier`
    ways and reduces afterwards. The greedy-subsets family solves the same
    problem with its `block_count`/`ComputeBlockHistograms` split, so the
    idea is familiar here even though the arithmetic is not shared.

    `blocksPerSm` is 2 on everything at or above compute capability 5, and
    this port takes that arm on every vendor -- the same modern-side choice
    `hist_2_one_byte_base.mojo` records for `Unroll` and `LoadSize`. It is a
    SCHEDULING constant: it changes how many document blocks are launched
    and therefore how many partial sums are reduced, which is a float
    summation ORDER. So it is pinned across vendors rather than tuned per
    vendor, exactly like the scan.

    GPU-AGNOSTIC: `sm_count` is threaded in rather than queried, the same
    way `partition_stats_chunks` takes it, because ONE number decided at the
    call site is auditable and a query buried in a helper is not.
    """
    var blocks_per_sm = 2
    var multiplier = 1
    var z = num_blocks_z if num_blocks_z < 8 else 8
    while (
        Float64(num_blocks_x * num_blocks_y * z * multiplier)
        < Float64(sm_count * blocks_per_sm) * 1.25
        and (ds_size // multiplier) > 10000
        and multiplier < limit
    ):
        multiplier *= 2
    return multiplier


def has_one_hot_features(
    one_hot_flags: List[Bool], feature_count: Int
) -> Bool:
    """`HasOneHotFeatures` (`:288-295`), the HOST overload."""
    for i in range(feature_count):
        if one_hot_flags[i]:
            return True
    return False


# ---------------------------------------------------------------------------
# `ScanHistogramsImpl` (`:108-181`)
# ---------------------------------------------------------------------------


def scan_pointwise_histograms_kernel(
    feature_first_fold_index: MutPointer[UInt32, MutAnyOrigin],
    feature_folds: MutPointer[UInt32, MutAnyOrigin],
    feature_one_hot: MutPointer[UInt8, MutAnyOrigin],
    feature_count_in: Int32,
    hist_line_size_in: Int32,
    hist_count_in: Int32,
    fold_count_in: Int32,
    histogram: MutPointer[Float32, MutAnyOrigin],
):
    """`ScanHistogramsImpl` (`:110-181`), with PORTING.md 8's substitution.

    Their kernel gives each feature a 32-lane warp, scans 32 bins at a time
    with `InclusiveScanInWarp` (`inplace_scan.cuh:166`) over a shared
    buffer, and carries lane 31's total into the next chunk.

    DEVIATION (PORTING.md 8, and 61 measured it): one thread per (feature,
    part, stat) scanning serially, no shared memory and no barrier. Two
    reasons, and the second is the binding one.

    First, it is the substitution this repository already made for the
    greedy-subsets scan (`histogram_utils.mojo:186`), and 61 measured that
    the serial shape costs nothing at these fold counts -- folds per feature
    is 255 at the very most and 1 for the binary features that dominate,
    while features, parts and stats are many and are what fill the machine.

    Second, and this is not a preference: **their loop puts `__syncthreads()`
    inside `if (featureId < featureCount)`** (`:149`, `:153`, `:158`,
    `:162`, `:168`). `featureId` is `blockIdx.x * featuresPerBlock +
    threadIdx.x / 32`, so within one block different warps carry different
    feature ids and the tail block has warps that fail the test. A block
    barrier reached by only some warps is undefined in CUDA and hangs on
    Metal. Transliterating it would be porting a bug onto a target that
    punishes it.

    ONE-HOT FEATURES ARE SKIPPED and that is theirs (`:126`): a one-hot
    bin is an EQUALITY test, so a running prefix across its bins is not a
    quantity that means anything.

    THE INDEXING IS THEIRS AND IT IS NOT THE GREEDY-SUBSETS INDEXING.
    Theirs is

        partId = TPointwisePartOffsetsHelper(gridDim.z)
                     .GetHistogramOffset(blockIdx.y, blockIdx.z)
        histogram += (partId * histLineSize + feature->FirstFoldIndex) * HIST_COUNT

    so the stat axis is the FASTEST-moving one -- bin `b` of feature `f` in
    part `p` at stat `h` is `((p * histLineSize) + firstFold + b) *
    HIST_COUNT + h`. The greedy-subsets histogram is the other way round,
    stat-major (`histogram_utils.mojo:246`). Getting this backwards
    transposes the histogram, which the totals cannot see: every cell is
    present, and a check that sums them passes
    ([[uniform-test-data-hides-permutation]] is exactly this failure).
    """
    var feature_count = Int(feature_count_in)
    var hist_line_size = Int(hist_line_size_in)
    var hist_count = Int(hist_count_in)

    var feature_id = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if feature_id >= feature_count:
        return

    # their `partId`, with `gridDim.z` as the fold count (`:120`)
    var helper = PointwisePartOffsetsHelper(UInt32(fold_count_in))
    var part_id = Int(
        helper.histogram_offset(UInt32(block_idx.y), UInt32(block_idx.z))
    )

    # their `skipFeature` (`:126`); the `Folds <= 1` half is ours by way of
    # the greedy-subsets scan and is free -- a one-bin prefix is the identity
    var folds = Int(feature_folds.unsafe_load(feature_id))
    if feature_one_hot.unsafe_load(feature_id) != UInt8(0) or folds <= 1:
        return

    var first_fold = Int(feature_first_fold_index.unsafe_load(feature_id))
    var base = (part_id * hist_line_size + first_fold) * hist_count

    for hist_id in range(hist_count):
        var running = Scalar[DType.float32](0.0)
        for b in range(folds):
            var at = base + b * hist_count + hist_id
            running += histogram.unsafe_load(at)
            histogram.unsafe_store(at, running)
