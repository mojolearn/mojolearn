# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# This lane MIRRORS CatBoost (pinned commit 54a8143a). Per-file provenance is in gbdt/DERIVATION_MAP.tsv, in this file's own docstring, and in NOTICE; files with no CatBoost counterpart say so.
"""The CTR elementwise kernels.

PORT OF `catboost/cuda/ctrs/kernel/ctr_calcers.cu` at CatBoost `54a8143a`.
Transliterated. Do not improve.

## WHY THIS FILE NEEDS NO KERNEL-MATRIX ROW

Every kernel below is elementwise or gather/scatter. Read the file and what
is NOT in it is the point:

* no `__shfl_*`, no `tiled_partition`, no cooperative groups -- so nothing
  hardcodes a 32-lane warp the way `SliceOffset()` does in the histogram
  family, and AMD's 64-wide wavefront changes nothing here;
* no `__shared__` at all -- so no threadgroup budget to query, and Apple's
  32 KB against NVIDIA's 48 KB does not reach this file;
* no atomics -- so no float-add ordering, and the answer does not depend on
  which block lands first.

`blockSize = 256` is theirs in all ten launchers and it is a SCHEDULING
constant with no vendor spread: it is not a warp multiple that has to
change, it is not sized from shared memory, and it changes no arithmetic.
`NonWeightedBinFreqCtrsImpl` additionally takes `elementsPerThreads = 4`
(`ctr_calcers.cu:118`), which is theirs and is likewise scheduling.

**So `original/kernel_matrix.mojo` gains nothing from this file, and that
is a result rather than an omission.** It is also the reason this block was
scoped as independent of the sort/scan lane: the kernels are the easy half,
and the system around them -- the segmented scan, the sort, the permutation
-- is the work.

## What is deliberately NOT ported here

The four groupwise-CTR kernels at the end of their file --
`ApplyGroupwiseCtrFix`, `MakeGroupStarts`, `FillBinIndices`,
`CreateFixedIndices` (`ctr_calcers.cu:300-429`). They run only when
`CtrHistoryUnit == ECtrHistoryUnit::Group`, which `SetCtrDefaults`
(`catboost_options.cpp:432-436`) sets only for a GROUPWISE loss, and no
ranking loss is ported. `THistoryBasedCtrCalcer::NeedFixForGroupwiseCtr()`
is `false` for every configuration this port can reach, so porting them
would add four kernels no caller reaches -- the exact defect
`PORTING_RULES.md` rule 3 names.

## The two spelling workarounds (`PORTING_RULES.md` rule 4)

1. **`writeIndices` and `map` are nullable in their signatures**
   (`ctr_calcers.cu:57`, `:280`) and select between `dst[Index(idx[i])]`
   and `dst[i]`. Mojo's `MutPointer` is non-nullable, so the null is
   carried as a separate `Int32` flag argument and the pointer is passed
   valid-but-unread. Same two branches, same order; only the spelling of
   "absent" changes.
2. **`FillBinarizedTargetsStatsImpl` is a `template <bool IS_BORDERS>`**
   whose value comes from a HOST bool (`FillBinarizedTargetsStats`'s
   `borders` argument, `:243-256`). Here it is a runtime `Int32` read
   inside the loop. Comptime specialization would be closer, but the
   branch is uniform across every thread in the grid and decides no
   arithmetic, so this is a scheduling difference and not a numeric one.

`StreamLoad`, `WriteThrough` and `LdgWithFallback` are CUB thread-load and
thread-store hints (`kernel_helpers.cuh:174-192`); Mojo 1.0 ships no
non-temporal load or store, so they are plain accesses here -- the same
deviation `fill.mojo` and `split_points.mojo` already record.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.memory import bitcast
from max.gpu.host import DeviceBuffer, DeviceContext

from gbdt.ctrs.index_wrapper import (
    CTR_SEGMENT_START_BIT,
    index_of,
    is_segment_start,
)


comptime CTR_BLOCK_SIZE = 256
"""`const ui32 blockSize = 256`, every launcher in `ctr_calcers.cu`."""

comptime CTR_DOCS_PER_THREAD = 4
"""`const ui32 elementsPerThreads = 4` (`ctr_calcers.cu:118`) and
`const int N = 4` (`:248`). Two different launchers, same constant."""


def _ceil_divide(x: Int, y: Int) -> Int:
    """Their `CeilDivide` (`kernel_helpers.cuh`)."""
    return (x + y - 1) // y


# --- GatherTrivialWeights (`ctr_calcers.cu:11-19`) -----------------------


def gather_trivial_weights_kernel(
    indices: MutPointer[UInt32, MutAnyOrigin],
    size_in: Int32,
    first_zero_index: UInt32,
    write_segment_start_float_mask: Int32,
    dst: MutPointer[Float32, MutAnyOrigin],
):
    """`GatherTrivialWeightsImpl`, copied.

    "Trivial weights" is their name for the learn/test split: every LEARN
    row weighs 1 and every TEST row weighs 0, and `firstZeroIndex` is the
    boundary (`ctr_helper.h:63-65` passes `CtrTargets.LearnSlice.Size()`).
    `dataset_helpers.cpp:29-32` builds the same vector on the host for the
    non-trivial path; this kernel is the version that never materializes
    it.

    The negation is not a sign, it is the SEGMENT-START FLAG moved from
    bit 31 of the index into the sign bit of the float, so that the
    segmented scan can read boundaries out of the value array alone. Note
    that it fires even when `val` is 0.0, giving -0.0, whose sign bit is
    set -- which is why every reader uses a BITWISE sign test and not
    `x < 0`.
    """
    var size = Int(size_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < size:
        var idx = indices.unsafe_load(i)
        var val = Float32(1.0) if index_of(idx) < first_zero_index else (
            Float32(0.0)
        )
        if write_segment_start_float_mask != Int32(0) and is_segment_start(
            idx
        ):
            dst.unsafe_store(i, -val)
        else:
            dst.unsafe_store(i, val)


def launch_gather_trivial_weights(
    ctx: DeviceContext,
    mut indices: DeviceBuffer[DType.uint32],
    size: Int,
    first_zero_index: UInt32,
    write_segment_start_float_mask: Bool,
    mut dst: DeviceBuffer[DType.float32],
) raises:
    """`GatherTrivialWeights` (`ctr_calcers.cu:21-32`)."""
    var num_blocks = _ceil_divide(size, CTR_BLOCK_SIZE)
    if num_blocks == 0:
        return
    ctx.enqueue_function[gather_trivial_weights_kernel](
        indices.unsafe_ptr(),
        Int32(size),
        first_zero_index,
        Int32(1) if write_segment_start_float_mask else Int32(0),
        dst.unsafe_ptr(),
        grid_dim=(num_blocks, 1, 1),
        block_dim=(CTR_BLOCK_SIZE, 1, 1),
    )


# --- WriteMask (`ctr_calcers.cu:35-42`) ----------------------------------


def write_mask_kernel(
    indices: MutPointer[UInt32, MutAnyOrigin],
    size_in: Int32,
    dst: MutPointer[Float32, MutAnyOrigin],
):
    """`WriteMaskImpl`, copied. In place: reads `dst[i]`, negates it at a
    segment start, writes it back. Their `WriteFloatMask` wrapper
    (`ctr_calcers.h:73`) calls this on ALREADY GATHERED weights, which is
    the non-trivial-weights counterpart of `GatherTrivialWeights`'s fused
    negate."""
    var size = Int(size_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < size:
        var idx = indices.unsafe_load(i)
        var val = dst.unsafe_load(i)
        if is_segment_start(idx):
            dst.unsafe_store(i, -val)
        else:
            dst.unsafe_store(i, val)


def launch_write_mask(
    ctx: DeviceContext,
    mut indices: DeviceBuffer[DType.uint32],
    size: Int,
    mut dst: DeviceBuffer[DType.float32],
) raises:
    """`WriteMask` (`ctr_calcers.cu:44-54`)."""
    var num_blocks = _ceil_divide(size, CTR_BLOCK_SIZE)
    if num_blocks == 0:
        return
    ctx.enqueue_function[write_mask_kernel](
        indices.unsafe_ptr(),
        Int32(size),
        dst.unsafe_ptr(),
        grid_dim=(num_blocks, 1, 1),
        block_dim=(CTR_BLOCK_SIZE, 1, 1),
    )


# --- WeightedBinFreqCtrs (`ctr_calcers.cu:56-67`) ------------------------


def weighted_bin_freq_ctrs_kernel(
    write_indices: MutPointer[UInt32, MutAnyOrigin],
    has_write_indices: Int32,
    bins: MutPointer[UInt32, MutAnyOrigin],
    bin_sums: MutPointer[Float32, MutAnyOrigin],
    total_weight: Float32,
    prior: Float32,
    prior_observations: Float32,
    dst: MutPointer[Float32, MutAnyOrigin],
    size_in: Int32,
):
    """`WeightedBinFreqCtrsImpl`, copied.

    **THIS IS THE DEFAULT FeatureFreq PATH, not the pure-freq one.**
    `TCalcCtrHelper::VisitEqualUpToPriorCtrs` sends a FeatureFreq config to
    `TCtrBinBuilder::VisitEqualUpToPriorFreqCtrs` (which calls
    `ComputeNonWeightedBinFreqCtr` below) only when
    `UseFullSetForCatFeatureStatCtrs()`, i.e. when
    `counter_calc_method == Full` (`binarizations_manager.h:290-292`).
    The shipped default is `SkipTest` (`cat_feature_options.cpp:233`), so
    the default dispatch lands HERE, on `TWeightedBinFreqCalcer`
    (`ctr_helper.h:96-111`).

    The two arms agree numerically on a fit with no test set: weights are
    1.0 on learn rows and 0.0 on test rows and `TotalWeight` is their sum
    (`dataset_helpers.cpp:24-39`), so `binSums[bin]` is the learn count and
    `totalWeight` is the learn row count. With no test set that is exactly
    `ComputeNonWeightedBinFreqCtr`'s `(count + prior) / (size +
    priorObservations)`. WITH a test set they differ, and the difference is
    the whole point of the option: `SkipTest` counts learn rows only,
    `Full` counts every row.
    """
    var size = Int(size_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < size:
        var dst_idx: Int
        if has_write_indices != Int32(0):
            dst_idx = Int(index_of(write_indices.unsafe_load(i)))
        else:
            dst_idx = i
        var bin = Int(bins.unsafe_load(i))
        dst.unsafe_store(
            dst_idx,
            (bin_sums.unsafe_load(bin) + prior)
            / (total_weight + prior_observations),
        )


def launch_compute_weighted_bin_freq_ctr(
    ctx: DeviceContext,
    mut write_indices: DeviceBuffer[DType.uint32],
    has_write_indices: Bool,
    mut bins: DeviceBuffer[DType.uint32],
    mut bin_sums: DeviceBuffer[DType.float32],
    total_weight: Float32,
    prior: Float32,
    prior_observations: Float32,
    mut dst: DeviceBuffer[DType.float32],
    size: Int,
) raises:
    """`ComputeWeightedBinFreqCtr` (`ctr_calcers.cu:104-115`)."""
    var num_blocks = _ceil_divide(size, CTR_BLOCK_SIZE)
    if num_blocks == 0:
        return
    ctx.enqueue_function[weighted_bin_freq_ctrs_kernel](
        write_indices.unsafe_ptr(),
        Int32(1) if has_write_indices else Int32(0),
        bins.unsafe_ptr(),
        bin_sums.unsafe_ptr(),
        total_weight,
        prior,
        prior_observations,
        dst.unsafe_ptr(),
        Int32(size),
        grid_dim=(num_blocks, 1, 1),
        block_dim=(CTR_BLOCK_SIZE, 1, 1),
    )


# --- NonWeightedBinFreqCtrs (`ctr_calcers.cu:69-101`) --------------------


def non_weighted_bin_freq_ctrs_kernel(
    write_indices: MutPointer[UInt32, MutAnyOrigin],
    has_write_indices: Int32,
    bins: MutPointer[UInt32, MutAnyOrigin],
    bin_offsets: MutPointer[UInt32, MutAnyOrigin],
    size_in: Int32,
    prior: Float32,
    prior_observations: Float32,
    dst: MutPointer[Float32, MutAnyOrigin],
):
    """`NonWeightedBinFreqCtrsImpl<256, 4>`, copied.

    THREE separate unrolled loops over `DOCS_PER_THREAD`, in their order:
    load the destination index and the bin, turn the bin into its segment
    LENGTH from the offsets table, then write. Their stride is `BLOCK_SIZE`
    between a thread's own elements, so the loads coalesce; a
    consecutive-per-thread rewrite would be the same answer with a
    different memory pattern and is exactly the kind of "improvement" the
    charter forbids.

    `nextBinOffset` guards with `bin < size`, which is their bound, not a
    typo for `bin < binCount`: the offsets array is allocated with one fake
    trailing bin (`ctr_calcers.h:317-320`) so the last real bin has a next
    offset to read.
    """
    var size = Int(size_in)
    var i = Int(block_idx.x) * CTR_BLOCK_SIZE * CTR_DOCS_PER_THREAD + Int(
        thread_idx.x
    )

    var dst_indices = InlineArray[Int32, CTR_DOCS_PER_THREAD](fill=-1)
    var bins_local = InlineArray[UInt32, CTR_DOCS_PER_THREAD](fill=0)

    for j in range(CTR_DOCS_PER_THREAD):
        var idx = i + CTR_BLOCK_SIZE * j
        if idx < size:
            if has_write_indices != Int32(0):
                dst_indices[j] = Int32(index_of(write_indices.unsafe_load(idx)))
            else:
                dst_indices[j] = Int32(idx)
            bins_local[j] = bins.unsafe_load(idx)

    for j in range(CTR_DOCS_PER_THREAD):
        var bin = Int(bins_local[j])
        var current_bin_offset = bin_offsets.unsafe_load(bin)
        var next_bin_offset: UInt32
        if bin < size:
            next_bin_offset = bin_offsets.unsafe_load(bin + 1)
        else:
            next_bin_offset = UInt32(size)
        bins_local[j] = next_bin_offset - current_bin_offset

    for j in range(CTR_DOCS_PER_THREAD):
        if dst_indices[j] != Int32(-1):
            dst.unsafe_store(
                Int(dst_indices[j]),
                (Float32(bins_local[j]) + prior)
                / (Float32(size) + prior_observations),
            )


def launch_compute_non_weighted_bin_freq_ctr(
    ctx: DeviceContext,
    mut write_indices: DeviceBuffer[DType.uint32],
    has_write_indices: Bool,
    mut bins: DeviceBuffer[DType.uint32],
    mut bin_offsets: DeviceBuffer[DType.uint32],
    size: Int,
    prior: Float32,
    prior_observations: Float32,
    mut dst: DeviceBuffer[DType.float32],
) raises:
    """`ComputeNonWeightedBinFreqCtr` (`ctr_calcers.cu:117-127`)."""
    var num_blocks = _ceil_divide(size, CTR_BLOCK_SIZE * CTR_DOCS_PER_THREAD)
    if num_blocks == 0:
        return
    ctx.enqueue_function[non_weighted_bin_freq_ctrs_kernel](
        write_indices.unsafe_ptr(),
        Int32(1) if has_write_indices else Int32(0),
        bins.unsafe_ptr(),
        bin_offsets.unsafe_ptr(),
        Int32(size),
        prior,
        prior_observations,
        dst.unsafe_ptr(),
        grid_dim=(num_blocks, 1, 1),
        block_dim=(CTR_BLOCK_SIZE, 1, 1),
    )


# --- UpdateBordersMask (`ctr_calcers.cu:130-147`) ------------------------


def update_borders_mask_kernel(
    bins: MutPointer[UInt32, MutAnyOrigin],
    prev_bins: MutPointer[UInt32, MutAnyOrigin],
    indices: MutPointer[UInt32, MutAnyOrigin],
    size_in: Int32,
):
    """`UpdateBordersMaskImpl`, copied.

    Three tests, short-circuited in their order, each only reached when the
    previous ones were false:

    1. the index ALREADY carries a segment start (a boundary from an
       earlier feature of the same tensor);
    2. `i == 0` or the sorted bin value changed;
    3. the PREVIOUS tensor's bin changed -- `prevBins` is indexed by the
       ORIGINAL row (`currentIndex.Index()`), not by the sorted position,
       which is what makes a combination of two cat features segment on
       the pair rather than on the newer feature alone.

    Test 3 reads `indices[i - 1]` and is guarded only by test 2's
    `i == 0`. That is safe because test 2 sets the mask at `i == 0` and
    short-circuits, and it is a real dependency: reordering the tests would
    read index -1.
    """
    var size = Int(size_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < size:
        var current_index = indices.unsafe_load(i)

        var mask = is_segment_start(current_index)
        if not mask:
            mask = i == 0 or bins.unsafe_load(i) != bins.unsafe_load(i - 1)
        if not mask:
            var prev_index = indices.unsafe_load(i - 1)
            var current_bin = prev_bins.unsafe_load(
                Int(index_of(current_index))
            )
            var prev_bin = prev_bins.unsafe_load(Int(index_of(prev_index)))
            mask = current_bin != prev_bin
        if mask:
            indices.unsafe_store(i, current_index | CTR_SEGMENT_START_BIT)
        else:
            indices.unsafe_store(i, current_index)


def launch_update_borders_mask(
    ctx: DeviceContext,
    mut bins: DeviceBuffer[DType.uint32],
    mut prev_bins: DeviceBuffer[DType.uint32],
    mut indices: DeviceBuffer[DType.uint32],
    size: Int,
) raises:
    """`UpdateBordersMask` (`ctr_calcers.cu:149-157`)."""
    var num_blocks = _ceil_divide(size, CTR_BLOCK_SIZE)
    if num_blocks == 0:
        return
    ctx.enqueue_function[update_borders_mask_kernel](
        bins.unsafe_ptr(),
        prev_bins.unsafe_ptr(),
        indices.unsafe_ptr(),
        Int32(size),
        grid_dim=(num_blocks, 1, 1),
        block_dim=(CTR_BLOCK_SIZE, 1, 1),
    )


# --- MergeBinsKernel (`ctr_calcers.cu:160-167`) --------------------------


def merge_bins_kernel(
    bins: MutPointer[UInt32, MutAnyOrigin],
    prev: MutPointer[UInt32, MutAnyOrigin],
    shift: UInt32,
    size_in: Int32,
):
    """`MergeBinsKernelImpl`, copied: `bins[i] = (bins[i] << shift) |
    prev[i]`.

    How a FEATURE TENSOR (a combination of cat features) gets one bin id:
    the new feature's bin is shifted above the bits the accumulated bins
    already occupy, `shift` being `IntLog2(uniqueValues)` of what came
    before. Unreached until tree CTRs land -- ported because it is four
    lines of their file and leaving a hole in a ported file is how a
    reader learns to distrust the whole file.
    """
    var size = Int(size_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < size:
        bins.unsafe_store(
            i, (bins.unsafe_load(i) << shift) | prev.unsafe_load(i)
        )


def launch_merge_bins(
    ctx: DeviceContext,
    mut bins: DeviceBuffer[DType.uint32],
    mut prev: DeviceBuffer[DType.uint32],
    shift: UInt32,
    size: Int,
) raises:
    """`MergeBinsKernel` (`ctr_calcers.cu:169-177`)."""
    var num_blocks = _ceil_divide(size, CTR_BLOCK_SIZE)
    if num_blocks == 0:
        return
    ctx.enqueue_function[merge_bins_kernel](
        bins.unsafe_ptr(),
        prev.unsafe_ptr(),
        shift,
        Int32(size),
        grid_dim=(num_blocks, 1, 1),
        block_dim=(CTR_BLOCK_SIZE, 1, 1),
    )


# --- ExtractBorderMasks (`ctr_calcers.cu:180-206`) -----------------------


def extract_border_masks_start_kernel(
    indices: MutPointer[UInt32, MutAnyOrigin],
    dst: MutPointer[UInt32, MutAnyOrigin],
    size_in: Int32,
):
    """`ExtractBorderMasksStartImpl`, copied."""
    var size = Int(size_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < size:
        var start = is_segment_start(indices.unsafe_load(i))
        dst.unsafe_store(i, UInt32(1) if start else UInt32(0))


def extract_border_masks_end_kernel(
    indices: MutPointer[UInt32, MutAnyOrigin],
    dst: MutPointer[UInt32, MutAnyOrigin],
    size_in: Int32,
):
    """`ExtractBorderMasksEndImpl`, copied.

    An END flag at `i` is the START flag at `i + 1`, and the LAST element
    is always an end. Both freq calcers call this one (`startSegment ==
    false`) and then EXCLUSIVE-scan it, which numbers the segments: the
    scan at position `i` counts the segments that closed strictly before
    `i`, which is `i`'s own segment index.
    """
    var size = Int(size_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < size:
        var is_end: Bool
        if (i + 1) < size:
            is_end = is_segment_start(indices.unsafe_load(i + 1))
        else:
            is_end = True
        dst.unsafe_store(i, UInt32(1) if is_end else UInt32(0))


def launch_extract_border_masks(
    ctx: DeviceContext,
    mut indices: DeviceBuffer[DType.uint32],
    mut dst: DeviceBuffer[DType.uint32],
    size: Int,
    start_segment: Bool,
) raises:
    """`ExtractBorderMasks` (`ctr_calcers.cu:208-219`)."""
    var num_blocks = _ceil_divide(size, CTR_BLOCK_SIZE)
    if num_blocks == 0:
        return
    if start_segment:
        ctx.enqueue_function[extract_border_masks_start_kernel](
            indices.unsafe_ptr(),
            dst.unsafe_ptr(),
            Int32(size),
            grid_dim=(num_blocks, 1, 1),
            block_dim=(CTR_BLOCK_SIZE, 1, 1),
        )
    else:
        ctx.enqueue_function[extract_border_masks_end_kernel](
            indices.unsafe_ptr(),
            dst.unsafe_ptr(),
            Int32(size),
            grid_dim=(num_blocks, 1, 1),
            block_dim=(CTR_BLOCK_SIZE, 1, 1),
        )


# --- FillBinarizedTargetsStats (`ctr_calcers.cu:222-256`) ----------------


def fill_binarized_targets_stats_kernel(
    binarized_targets: MutPointer[UInt8, MutAnyOrigin],
    sample_weight: MutPointer[Float32, MutAnyOrigin],
    dst: MutPointer[Float32, MutAnyOrigin],
    size_in: Int32,
    bin_index: UInt32,
    is_borders: Int32,
):
    """`FillBinarizedTargetsStatsImpl<IS_BORDERS, 4>`, copied.

    The numerator of an ordered target statistic, before the scan. Per row:

        |weight| * (IS_BORDERS ? target > binIndex : target == binIndex)

    then the WEIGHT'S SIGN BIT is re-applied to the result. That is the
    segment-start flag surviving one more stage, which is why the sign test
    is `ExtractSignBit` (`kernel_helpers.cuh:20-23`), a bit read, and not
    `weight < 0`: a segment start whose weight is 0.0 arrives as -0.0,
    where `-0.0 < 0` is FALSE and the flag would be dropped. A test row
    (weight 0) that starts a segment is exactly that case.

    `Borders` compares `>` and `Buckets` compares `==`; both walk the same
    binarized target, so the target grid is built once for the fit
    (`catboost_options.cpp:505` refuses a per-CTR one).
    """
    var size = Int(size_in)
    var i = (
        Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    ) * CTR_DOCS_PER_THREAD

    var local_samples = InlineArray[Float32, CTR_DOCS_PER_THREAD](fill=0)
    for k in range(CTR_DOCS_PER_THREAD):
        var idx = i + k
        var v = Float32(0.0)
        if idx < size:
            var weight = sample_weight.unsafe_load(idx)
            var target = binarized_targets.unsafe_load(idx)
            var hit: Bool
            if is_borders != Int32(0):
                hit = UInt32(target) > bin_index
            else:
                hit = UInt32(target) == bin_index
            var mag = weight if weight >= Float32(0.0) else -weight
            v = mag * (Float32(1.0) if hit else Float32(0.0))
            # `ExtractSignBit(weight)`: the BIT, so -0.0 counts
            if (bitcast[DType.uint32](weight) >> 31) != UInt32(0):
                v = -v
        local_samples[k] = v

    for k in range(CTR_DOCS_PER_THREAD):
        var idx = i + k
        if idx < size:
            dst.unsafe_store(idx, local_samples[k])


def launch_fill_binarized_targets_stats(
    ctx: DeviceContext,
    mut sample: DeviceBuffer[DType.uint8],
    mut sample_weights: DeviceBuffer[DType.float32],
    size: Int,
    mut sums: DeviceBuffer[DType.float32],
    bin_index: UInt32,
    borders: Bool,
) raises:
    """`FillBinarizedTargetsStats` (`ctr_calcers.cu:258-271`)."""
    var num_blocks = _ceil_divide(size, CTR_DOCS_PER_THREAD * CTR_BLOCK_SIZE)
    if num_blocks == 0:
        return
    ctx.enqueue_function[fill_binarized_targets_stats_kernel](
        sample.unsafe_ptr(),
        sample_weights.unsafe_ptr(),
        sums.unsafe_ptr(),
        Int32(size),
        bin_index,
        Int32(1) if borders else Int32(0),
        grid_dim=(num_blocks, 1, 1),
        block_dim=(CTR_BLOCK_SIZE, 1, 1),
    )


# --- MakeMeans (`ctr_calcers.cu:274-282`) --------------------------------


def make_means_kernel(
    sums: MutPointer[Float32, MutAnyOrigin],
    weights: MutPointer[Float32, MutAnyOrigin],
    size_in: Int32,
    sum_prior: Float32,
    weight_prior: Float32,
):
    """`MakeMeansImpl`, copied: the priors divide, in place.

        sums[tid] = (sums[tid] + sumPrior) / (weights[tid] + weightPrior)

    THE PRIOR FAN-OUT IS THIS LINE RUN THREE TIMES. `VisitCatFeatureCtr`
    scans once and then loops the configs (`ctr_calcers.h:141-150`),
    calling `DivideWithPriors` -- this kernel -- per prior. Three priors,
    three output columns, one scan.
    """
    var size = Int(size_in)
    var tid = Int(block_dim.x) * Int(block_idx.x) + Int(thread_idx.x)
    if tid < size:
        sums.unsafe_store(
            tid,
            (sums.unsafe_load(tid) + sum_prior)
            / (weights.unsafe_load(tid) + weight_prior),
        )


def launch_make_means(
    ctx: DeviceContext,
    mut sums: DeviceBuffer[DType.float32],
    mut weights: DeviceBuffer[DType.float32],
    size: Int,
    sum_prior: Float32,
    weight_prior: Float32,
) raises:
    """`MakeMeans` (`ctr_calcers.cu:284-293`)."""
    var num_blocks = _ceil_divide(size, CTR_BLOCK_SIZE)
    if num_blocks == 0:
        return
    ctx.enqueue_function[make_means_kernel](
        sums.unsafe_ptr(),
        weights.unsafe_ptr(),
        Int32(size),
        sum_prior,
        weight_prior,
        grid_dim=(num_blocks, 1, 1),
        block_dim=(CTR_BLOCK_SIZE, 1, 1),
    )


# --- MakeMeansAndScatter (`ctr_calcers.cu:295-307`) ----------------------


def make_means_and_scatter_kernel(
    sums: MutPointer[Float32, MutAnyOrigin],
    weights: MutPointer[Float32, MutAnyOrigin],
    size_in: Int32,
    sum_prior: Float32,
    weight_prior: Float32,
    map_ptr: MutPointer[UInt32, MutAnyOrigin],
    has_map: Int32,
    mask: UInt32,
    dst: MutPointer[Float32, MutAnyOrigin],
):
    """`MakeMeansAndScatterImpl`, copied. `MakeMeans` fused with the
    permutation write-back, so a scan that ran in SORTED-BY-BIN order lands
    back in row order without a second pass.

    The mask is passed in rather than taken from `TIndexWrapper::Index()`,
    and every call site passes `TCtrBinBuilder::GetMask()`, which is the
    same `0x3FFFFFFF`.
    """
    var size = Int(size_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < size:
        var m: Int
        if has_map != Int32(0):
            m = Int(map_ptr.unsafe_load(i) & mask)
        else:
            m = i
        dst.unsafe_store(
            m,
            (sums.unsafe_load(i) + sum_prior)
            / (weights.unsafe_load(i) + weight_prior),
        )


def launch_make_means_and_scatter(
    ctx: DeviceContext,
    mut sums: DeviceBuffer[DType.float32],
    mut weights: DeviceBuffer[DType.float32],
    size: Int,
    sum_prior: Float32,
    weight_prior: Float32,
    mut map_buf: DeviceBuffer[DType.uint32],
    has_map: Bool,
    mask: UInt32,
    mut dst: DeviceBuffer[DType.float32],
) raises:
    """`MakeMeansAndScatter` (`ctr_calcers.cu:309-322`)."""
    var num_blocks = _ceil_divide(size, CTR_BLOCK_SIZE)
    if num_blocks == 0:
        return
    ctx.enqueue_function[make_means_and_scatter_kernel](
        sums.unsafe_ptr(),
        weights.unsafe_ptr(),
        Int32(size),
        sum_prior,
        weight_prior,
        map_buf.unsafe_ptr(),
        Int32(1) if has_map else Int32(0),
        mask,
        dst.unsafe_ptr(),
        grid_dim=(num_blocks, 1, 1),
        block_dim=(CTR_BLOCK_SIZE, 1, 1),
    )
