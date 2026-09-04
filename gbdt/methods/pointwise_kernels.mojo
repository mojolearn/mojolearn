# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The POINTWISE host launch layer: grids, the multiplier ladder, the fan-out.

PORT OF, in one file because they are one call chain:

    `catboost/cuda/methods/pointwise_kernels.{h,cpp}`   the wrapper objects
    `catboost/cuda/methods/kernel/pointwise_hist2.cu`   `UpdateFoldBins`,
                                                        `UpdatePointwiseHistograms`,
                                                        `ScanPointwiseHistograms`
    `pointwise_hist2_one_byte_templ.cuh:221-270`        `ComputeHist2NonBinary`
    `pointwise_hist2_binary.cu:128-179`                 `ComputeHist2Binary`
    `pointwise_hist2_half_byte.cu:130-180`              `ComputeHist2HalfByte`
    `gpu_data/folds_histogram.h`                        `TFoldsHistogram`

at CatBoost `54a8143a`. Transliterated. Do not improve.

WHAT THIS FILE IS. The three drivers in `gbdt/methods/kernel/` know how to
turn a block into a histogram. Nothing yet decided HOW MANY BLOCKS, which
kernel gets which features, or what happens after the histogram is built.
That is all here, and it is four decisions:

    1. the grid          features per block is fixed per family (32 binary,
                         8 half-byte, 4 one-byte), leaves on `y`, folds on `z`
    2. the MULTIPLIER    when the feature axis cannot fill the machine,
                         split the DOCUMENT axis `M` ways and reduce with
                         atomics -- `EstimateBlockPerFeatureMultiplier`
    3. the fan-out       one call per POLICY, and for one-byte, one call per
                         BIT WIDTH; the kernels then partition the blocks
                         between themselves at runtime
    4. the tail          scan the histograms into prefix sums (every policy
                         but binary), then, on a partial pass, recover the
                         sibling by subtraction

THE ONE THING IN HERE THAT LOOKS LIKE A BUG AND IS NOT.
`ComputeHist2NonBinary` computes `numBlocks.x` TWICE, with two different
feature counts (`:240` and `:242`):

    numBlocks.x = (featureCountForBits + 3) / 4;         // sizes the estimate
    const ui32 multiplier = min(EstimateBlockPerFeatureMultiplier(numBlocks, size), 64);
    numBlocks.x = ((nbCount + 3) / 4);                   // sizes the LAUNCH
    numBlocks.x *= multiplier;

It is DELIBERATE and both values are load-bearing.

  * The LAUNCH must cover `nbCount`, every one-byte feature, because each
    bit-width kernel is launched over all of them and returns from the
    groups outside its range (`pointwise_hist2_one_byte_templ.mojo`'s
    step 3). A grid sized to `featureCountForBits` would not reach the
    later groups at all, and which groups those are is not knowable on the
    host -- the kernel decides it from `GetMaxBinCount` at runtime.
  * The ESTIMATE must be sized on `featureCountForBits`, the number of
    features this bit width will ACTUALLY do work for, because the
    multiplier answers "is there enough real work to fill the device". A
    grid of 500 blocks of which 8 do anything is not 500 blocks of work,
    and estimating on `nbCount` would return `multiplier == 1` for every
    width on a wide dataset and leave the document axis unsplit.

So the first assignment is a scratch value passed into an estimator that
takes a `dim3`, and the second is the grid. The consequence, which is also
theirs and is not accidental: a bit width owning few features gets a LARGE
multiplier and that multiplier is then applied to ALL `ceil(nbCount / 4)`
groups, most of which return immediately. Their kernels are cheap to refuse
-- one shared reduce over four `Folds` values and a bounds test -- and that
is the trade they made.

HALF OF IT IS UNOBSERVABLE FROM THE OUTPUT AND THAT IS WORTH SAYING PLAINLY.
Sizing the LAUNCH wrong is loud: sabotaged, it moves 2,796 of 3,686 cells in
gate F1 and leaves three of the four bit widths claiming nothing in F4.
Sizing the ESTIMATE wrong changes NO CELL -- both choices produce a grid
consistent with the `M` the kernel is instantiated for, so the histogram is
right either way and only the BLOCK COUNT differs. The one channel through
which a block count reaches the output is the writeback, which is a plain
store at `M == 1` and an atomicAdd above it; gate F2c seeds the histogram
with a non-zero pattern so that a store and an add are distinguishable, and
that gate -- and only that gate -- moves (2,004 of 3,090 cells). Without it,
half of the finding above would be a claim with nothing holding it up.

FOUR THINGS DIFFER FROM THEIRS AND ALL FOUR ARE DECLARED BELOW: deviations
100 (block sizes), 101 (`exit(1)`), 102 (the wrapper object), plus the scan
grid, which is an inherited consequence of `archive/reference/PORTING.md` 8 rather than a new
decision -- see `scan_pointwise_histograms`.

DEVIATION 100: THE BLOCK SIZES ARE THE KERNEL MATRIX'S, NOT THEIR LITERALS
--------------------------------------------------------------------------
THEIRS: `const int blockSize = 384;` for the one-byte family
        (`pointwise_hist2_one_byte_templ.cuh:238`) and
        `const int blockSize = 768;` for both small-bin families
        (`pointwise_hist2_binary.cu:141`, `pointwise_hist2_half_byte.cu:145`).

OURS:   `PW_HIST2_BLOCK` (256 on Apple, via `pw_hist2_block_size_for` --
        which also records the measured-negative doubled-block experiment)
        and `PW_HB_BLOCK` (512), READ FROM THE KERNEL FILES rather than
        restated here, so a launcher cannot drift from the kernel it
        launches.

MEASURED REASON: the accumulators are sized per thread -- 32 floats each for
the one-byte family, 16 for the small-bin one. At CatBoost's 768 the
small-bin accumulator wants 16 x 768 x 4 = 49,152 bytes of threadgroup
memory against Apple's 32,768 limit, and at 384 the one-byte accumulator
wants 32 x 384 x 4 = 49,152 against the same limit. The kernel matrix
resolves both; `pointwise_hist2_half_byte_template.mojo` records the
derivation and a `comptime assert` there refuses anything under 512, because
that family's `Reduce` folds its warp slices under `if (threadIdx.x < 512)`
and a smaller block would leave the top of the first slice unfolded.

WHAT IT DOES AND DOES NOT CHANGE. It does NOT change the grid: every
`numBlocks` expression below is in FEATURES and PARTS, never in threads, so
the block size appears in none of them. It does not change the multiplier:
`EstimateBlockPerFeatureMultiplier` counts blocks, not threads. It DOES
change how many blocks a scan launch needs (`ceil(featureCount / blockSize)`
moves with the divisor), and it changes the number of warp slices each
accumulator folds, which is a float summation ORDER inside the kernel and is
already recorded where the accumulators are.

DEVIATION 101: `exit(1)` AND `CB_ENSURE_INTERNAL` BECOME RAISED ERRORS
----------------------------------------------------------------------
THEIRS: the multiplier ladder ends `} else { exit(1); }`
        (`pointwise_hist2_one_byte_templ.cuh:266`, `_binary.cu:174`,
        `_half_byte.cu:175`) -- a bare process abort with no message. The
        `histCount` guards end `CB_ENSURE_INTERNAL(false, ...)`
        (`pointwise_hist2.cu:99`, `:129`), which throws.

OURS:   both raise, and the multiplier one names the offending value.

REASON: not a choice about behavior -- Mojo has no `exit` in a `def` that
already `raises`, and a library that kills the process instead of returning
an error cannot be gated. Both are UNREACHABLE by construction from inside
this file (`EstimateBlockPerFeatureMultiplier` only ever doubles from 1 and
is capped at 64, so it is always a power of two in [1, 64]; `histCount` is
passed as 2 by the only caller). They are transcribed because a guard that
is unreachable today is the one that catches tomorrow's caller, and
because `min(..., 64)` is what makes the seven enough:
`checks/pointwise_dispatch_check.mojo` gate F7 sweeps 245
configurations, finds 12 that return 128 before the clamp, and asserts
every clamped value is one of the seven.

DEVIATION 102: THE WRAPPER OBJECT BECOMES A FUNCTION
-----------------------------------------------------
THEIRS: `TComputeHist2Kernel : TStatelessKernel` holds thirteen members,
        declares `Y_SAVELOAD_DEFINE` over all of them, is registered in a
        global table as `REGISTER_KERNEL(0x420000, ...)`, and is dispatched
        to N devices by `LaunchKernels<TKernel>(targets.NonEmptyDevices(),
        ...)`.

OURS:   `compute_hist2(...)`, a function taking the same thirteen values as
        arguments, on one device.

REASON: three of theirs do not exist here and one is deliberate.
`Y_SAVELOAD_DEFINE` and `REGISTER_KERNEL` serialize a kernel invocation so
it can be sent to another PROCESS -- CatBoost's multi-host path. There is no
such path here and porting the table without it would be porting a name.
`TCudaBufferPtr<T>` carries a pointer and a size; ours are separate
arguments, the same substitution every ported kernel in this tree already
makes (`archive/reference/PORTING.md` 9). And `NonEmptyDevices()` is the multi-device fan-out,
which `archive/reference/PORTING.md` 91 A settled: at device count 1 the layouts coincide, and
one device is what this tree runs.

TWO SMALLER CONSEQUENCES OF THE SAME DECISION, both real ports of theirs:

  * `TFoldsHistogram` (`gpu_data/folds_histogram.h`) is ported HERE, not in
    `gbdt/gpu_data/`, because the one-byte fan-out cannot be written without
    it and this lane owns two files. It is 27 lines and it moves to
    `gbdt/gpu_data/folds_histogram.mojo` the moment anything else needs it.
  * `TComputeHist1Kernel` is NOT ported. `pointwise_hist1.cu` is dead in
    the upstream -- registered, wrapped, and called by nothing
    (`archive/reference/PORTING.md` 91 D, `gbdt/NOT_IMPLEMENTED.tsv`).

INHERITED, NOT NEW: the 8-bit path takes a `fixed_scale` their kernels have
no parameter for. That is DEVIATION 93 (Metal has no threadgroup float
atomic, so the 8-bit accumulator is Int32 fixed point); this layer only
threads the value through.
"""

from checks.kernel_matrix import (
    TARGET_COLUMN,
    pointwise_one_byte_fixed_for,
)
from checks.numerics import GLOBAL_NUMERIC_MODE as HIST_BUILD_MODE
from checks.numerics import NUMERIC_IDENTICAL
from max.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, grid_dim, thread_idx

from gbdt.gpu_data.grid_policy import (
    POLICY_BINARY,
    POLICY_HALF_BYTE,
    POLICY_ONE_BYTE,
)
from gbdt.methods.kernel.split_properties_helpers import (
    PointwisePartOffsetsHelper,
    estimate_block_per_feature_multiplier,
    scan_pointwise_histograms_kernel,
)
from gbdt.methods.kernel.pointwise_hist2_one_byte_5bit import (
    PW_HIST2_BLOCK,
    PW_HIST2_FLOAT_BLOCK,
)
from gbdt.methods.kernel.pointwise_hist2_half_byte_template import PW_HB_BLOCK
from gbdt.methods.kernel.pointwise_hist2_one_byte_templ import (
    compute_split_properties_nb_kernel,
)
from gbdt.methods.kernel.pointwise_hist2_binary import (
    compute_split_properties_b_kernel,
)
from gbdt.methods.kernel.pointwise_hist2_half_byte import (
    compute_split_properties_half_byte_kernel,
)


# ---------------------------------------------------------------------------
# constants (theirs, by file and line)
# ---------------------------------------------------------------------------

#: `numBlocks.x = (bCount + 31) / 32` (`pointwise_hist2_binary.cu:132`).
comptime PW_B_FEATURES_PER_BLOCK = 32

#: `numBlocks.x = (halfByteFeaturesCount + 7) / 8`
#: (`pointwise_hist2_half_byte.cu:139`).
comptime PW_HB_FEATURES_PER_BLOCK = 8

#: `numBlocks.x = (featureCountForBits + 3) / 4`
#: (`pointwise_hist2_one_byte_templ.cuh:240`).
comptime PW_NB_FEATURES_PER_BLOCK = 4

#: `min(EstimateBlockPerFeatureMultiplier(numBlocks, size), 64)`. The cap is
#: theirs and it is NOT `EstimateBlockPerFeatureMultiplier`'s own `limit`
#: argument, which defaults to 128 -- all three call sites clamp again to 64
#: on the way out because only seven instantiations exist.
comptime PW_MAX_MULTIPLIER = 64

#: `const int blockSize = 256;` for both elementwise kernels in
#: `pointwise_hist2.cu` (`:29`, `:81`). NOT a matrix row: neither kernel
#: allocates threadgroup memory, so DEVIATION 100 does not touch them and
#: their literal is portable as written.
comptime PW_UPDATE_BLOCK = 256

#: `const int scanBlockSize = 256;` (`pointwise_hist2.cu:107`).
comptime PW_SCAN_BLOCK = 256


@always_inline
def is_grid_empty(x: Int, y: Int, z: Int) -> Bool:
    """`IsGridEmpty` (`cuda_util/kernel/kernel_helpers.cuh:263-265`).

        return grid.x == 0 || grid.y == 0 || grid.z == 0;

    Honoured before EVERY launch below, exactly where theirs honours it.
    It is not defensive tidiness: `numBlocks.y` is `partCount / 2` on a
    partial pass, so a one-leaf level makes it zero, and `numBlocks.x` is
    zero whenever a policy or a bit width owns no features. CUDA tolerates a
    zero-extent launch; a Metal queue does not have to.
    """
    return x == 0 or y == 0 or z == 0


# ---------------------------------------------------------------------------
# `TFoldsHistogram` (`gpu_data/folds_histogram.h`)
# ---------------------------------------------------------------------------


struct FoldsHistogram(Copyable, Movable):
    """`TFoldsHistogram`, field for field: `std::array<ui32, 9> Counts`.

    HOW MANY FEATURES NEED EXACTLY `bit` BITS. Index `b` counts the features
    whose fold count fits in `b` bits and not in `b - 1`, for `b` in 0..8.
    It exists for one reason: `TComputeHist2Kernel::Run` needs
    `FeatureCountForBits(from, to)` to size the multiplier estimate for each
    of the four one-byte kernels, and nothing else in their tree reads it.

    THE FOUR RANGES ARE NOT UNIFORM and that is the file's whole content
    (`pointwise_kernels.cpp:57-60`):

        ComputeHist2NonBinary<5>   bits 4..5
        ComputeHist2NonBinary<6>   bit  6
        ComputeHist2NonBinary<7>   bit  7
        ComputeHist2NonBinary<8>   bit  8

    The 5-bit kernel claims bit FOUR as well, which is the host-side echo of
    `lowerBound = BITS > 5 ? upperBound / 2 : 15` in the kernel: a feature
    with 16 folds needs 5 bits to index but is claimed at 5, so the count
    that sizes the 5-bit estimate has to include the 16-fold features too.
    Read `4, 5` as `6, 6`'s sibling and the estimate under-counts.
    """

    var counts: InlineArray[UInt32, 9]
    """`std::array<ui32, 9> Counts`, the same fixed nine slots. An
    `InlineArray` rather than a `List` because theirs is a value type with a
    compile-time size and a `List` would make the struct non-copyable."""

    def __init__(out self):
        """`Counts.fill(0)`."""
        self.counts = InlineArray[UInt32, 9](fill=UInt32(0))

    def __init__(out self, counts: List[UInt32]) raises:
        """Fieldwise. `counts` must have nine entries, bits 0 through 8."""
        if len(counts) != 9:
            raise Error(
                "TFoldsHistogram: Counts is std::array<ui32, 9>, got "
                + String(len(counts))
                + " entries"
            )
        self.counts = InlineArray[UInt32, 9](fill=UInt32(0))
        for b in range(9):
            self.counts[b] = counts[b]

    def feature_count_for_bits(
        self, from_bit: Int, to_bit_inclusive: Int
    ) raises -> Int:
        """`FeatureCountForBits` (`folds_histogram.h:16-24`), copied.

            CB_ENSURE(toBitInclusive <= 8);
            CB_ENSURE(fromBit <= toBitInclusive);
            for (bit = fromBit; bit <= toBitInclusive; ++bit)
                count += Counts[bit];
        """
        if to_bit_inclusive > 8:
            raise Error(
                "FeatureCountForBits: toBitInclusive must be <= 8, got "
                + String(to_bit_inclusive)
            )
        if from_bit > to_bit_inclusive:
            raise Error(
                "FeatureCountForBits: fromBit "
                + String(from_bit)
                + " > toBitInclusive "
                + String(to_bit_inclusive)
            )
        var count = 0
        for bit in range(from_bit, to_bit_inclusive + 1):
            count += Int(self.counts[bit])
        return count


def folds_histogram_from_folds(folds: List[UInt32]) -> FoldsHistogram:
    """Bin a feature list by how many BITS its fold count needs.

    NOT a port of a function of theirs -- their histogram is filled while the
    grid is built (`feature_layout.cpp`), which this lane does not own -- but
    the same tally, and it is what a caller without a built grid needs. The
    rule is `Counts[bits_needed(folds)]`, where `bits_needed(n)` is the
    smallest `b` with `n <= (1 << b)`: a feature with 16 folds needs 4, with
    17 needs 5, with 256 needs 8.

    Reading this the other way -- `bits_needed = ceil(log2(folds + 1))` --
    shifts every power of two by one and moves the 16-fold features out of
    the 5-bit kernel's range, which is exactly the mistake the kernel's
    `lowerBound = ... : 15` exists to prevent. Gate F6 pins it.
    """
    var h = FoldsHistogram()
    for i in range(len(folds)):
        var n = Int(folds[i])
        var bits = 0
        while (1 << bits) < n:
            bits += 1
        if bits > 8:
            bits = 8
        h.counts[bits] += 1
    return h^


# ---------------------------------------------------------------------------
# `UpdateFoldBins` (`pointwise_hist2.cu:17-33`)
# ---------------------------------------------------------------------------


def update_bins_kernel(
    dst_bins: MutPointer[UInt32, MutAnyOrigin],
    bins: MutPointer[UInt32, MutAnyOrigin],
    doc_indices: MutPointer[UInt32, MutAnyOrigin],
    size_in: Int32,
    load_bit_in: Int32,
    fold_bits_in: Int32,
):
    """`UpdateBinsImpl` (`:17-25`), copied.

        const ui32 idx = LdgWithFallback(docIndices, i);
        const ui32 bit = (LdgWithFallback(bins, idx) >> loadBit) & 1;
        dstBins[i] = dstBins[i] | (bit << (loadBit + foldBits));

    ONE BIT OF THE SPLIT, OR-ed INTO THE DOCUMENT'S BIN. This is how the
    oblivious level advances: after a split is chosen, every document's leaf
    id gains one bit, and that bit is read out of the `bins` column the
    split names. `loadBit` selects the bit inside `bins`; `loadBit +
    foldBits` places it above the fold bits in the destination, which is the
    packing `archive/reference/PORTING.md` 91 B describes -- fold id in the LOW bits, depth
    bits above it.

    IT IS A GATHER AND AN OR, not a store. `dstBins[i] |= ...` keeps every
    bit already set, so calling it twice for the same `loadBit` is
    idempotent and calling it for successive bits accumulates the path.

    `LdgWithFallback` is `__ldg`, the read-only texture-cache load. It
    becomes a plain load: it is a cache HINT and moves no value.
    """
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(size_in):
        var idx = Int(doc_indices.unsafe_load(i))
        var bit = (bins.unsafe_load(idx) >> UInt32(load_bit_in)) & UInt32(1)
        var shift = UInt32(load_bit_in) + UInt32(fold_bits_in)
        dst_bins.unsafe_store(i, dst_bins.unsafe_load(i) | (bit << shift))


def update_fold_bins[
    o1: MutOrigin, o2: MutOrigin, o3: MutOrigin, //
](
    ctx: DeviceContext,
    dst_bins: MutPointer[UInt32, o1],
    bins: MutPointer[UInt32, o2],
    doc_indices: MutPointer[UInt32, o3],
    size: Int,
    load_bit: Int,
    fold_bits: Int,
) raises:
    """`UpdateFoldBins` (`:27-33`) and `TUpdateFoldBinsKernel::Run`
    (`pointwise_kernels.h:208-211`), copied.

        const ui32 blockSize = 256;
        const ui32 numBlocks = CeilDivide(size, blockSize);

    Theirs asserts `DstBins.Size() == DocIndices.Size()` in the wrapper
    before launching; that is the `CB_ENSURE` below.

    NO `IsGridEmpty` HERE and that is theirs: this is the one launcher in the
    family without the guard, because `CeilDivide(0, 256)` is 0 and their
    code simply launches it. Ours refuses instead -- a zero-extent launch is
    not portable -- and refusing is observationally identical because the
    kernel body is `if (i < size)`.
    """
    if size == 0:
        return
    var num_blocks = (size + PW_UPDATE_BLOCK - 1) // PW_UPDATE_BLOCK
    ctx.enqueue_function[update_bins_kernel](
        dst_bins,
        bins,
        doc_indices,
        Int32(size),
        Int32(load_bit),
        Int32(fold_bits),
        grid_dim=(num_blocks, 1, 1),
        block_dim=(PW_UPDATE_BLOCK, 1, 1),
    )


# ---------------------------------------------------------------------------
# `UpdatePointwiseHistograms` (`pointwise_hist2.cu:38-101`)
# ---------------------------------------------------------------------------


def update_pointwise_histograms_kernel[
    hist_count: Int
](
    histogram: MutPointer[Float32, MutAnyOrigin],
    first_bin_feature_in: Int32,
    features_count_in: Int32,
    # `TDataPartition*` as `{Offset, Size}` pairs of UInt32
    parts: MutPointer[UInt32, MutAnyOrigin],
    hist_line_size_in: Int32,
):
    """`UpdatePointwiseHistogramsImpl<HIST_COUNT>` (`:38-74`), copied.

    THE SIBLING SUBTRACTION, FINALLY DONE. `ShiftPartAndBinSumsPtr`'s `else`
    arm made the driver compute the CHEAPER child of each pair and file it
    under the RIGHT child's slot, leaving the LEFT slot holding the PARENT's
    histogram from the level before. This kernel resolves that:

        calcVal       = histogram[right]              the child that was built
        complementVal = histogram[left] - calcVal     parent - child
        histogram[left]  = isLeftCalculated ? calcVal : complementVal
        histogram[right] = isLeftCalculated ? complementVal : calcVal

    where `isLeftCalculated = leftPart.Size < rightPart.Size` re-derives, from
    the partition sizes, WHICH child the driver actually walked -- the same
    test the driver made, made again here rather than communicated. That is
    theirs and it is why nothing has to be passed between the two kernels.

    NOTE THE TWO DIFFERENT OFFSET FUNCTIONS AGAIN. `parts` is indexed with
    `GetDataPartitionOffset` (the fold axis rounded up to a power of two) and
    `histogram` with `GetHistogramOffset` (packed tight). At `FoldCount == 1`
    they coincide; above it, reading one for the other transposes the fixup
    onto the wrong pair. See `split_properties_helpers.mojo`.

    `blockIdx.y | gridDim.y` is theirs, and exact rather than lucky, for the
    reason recorded on `shift_part_and_bin_sums_ptr`: `gridDim.y` here is
    `partCount / 2`, a power of two strictly greater than every `blockIdx.y`.
    """
    var helper = PointwisePartOffsetsHelper(UInt32(grid_dim.z))

    var left_part_id = Int(
        helper.data_partition_offset(UInt32(block_idx.y), UInt32(block_idx.z))
    )
    var right_part_id = Int(
        helper.data_partition_offset(
            UInt32(block_idx.y) | UInt32(grid_dim.y), UInt32(block_idx.z)
        )
    )
    var first_bin_feature = Int(first_bin_feature_in)
    var bin_feature = (
        first_bin_feature
        + Int(block_idx.x) * Int(block_dim.x)
        + Int(thread_idx.x)
    )

    if bin_feature < first_bin_feature + Int(features_count_in):
        var left_size = parts.unsafe_load(2 * left_part_id + 1)
        var right_size = parts.unsafe_load(2 * right_part_id + 1)
        var is_left_calculated = left_size < right_size

        var hist_line_size = Int(hist_line_size_in)
        var left_offset = hist_count * (
            Int(
                helper.histogram_offset(
                    UInt32(block_idx.y), UInt32(block_idx.z)
                )
            )
            * hist_line_size
            + bin_feature
        )
        var right_offset = hist_count * (
            Int(
                helper.histogram_offset(
                    UInt32(block_idx.y) | UInt32(grid_dim.y),
                    UInt32(block_idx.z),
                )
            )
            * hist_line_size
            + bin_feature
        )

        # theirs reads BOTH planes before writing EITHER (`:60-64` then
        # `:67-71`), which matters because `left` and `right` alias when a
        # caller mis-sizes the grid; transcribed in the same two passes.
        var calc_val = InlineArray[Float32, hist_count](fill=Float32(0.0))
        var complement_val = InlineArray[Float32, hist_count](
            fill=Float32(0.0)
        )
        comptime for hist_id in range(hist_count):
            calc_val[hist_id] = histogram.unsafe_load(right_offset + hist_id)
            complement_val[hist_id] = (
                histogram.unsafe_load(left_offset + hist_id)
                - calc_val[hist_id]
            )

        comptime for hist_id in range(hist_count):
            histogram.unsafe_store(
                left_offset + hist_id,
                calc_val[hist_id] if is_left_calculated
                else complement_val[hist_id],
            )
            histogram.unsafe_store(
                right_offset + hist_id,
                complement_val[hist_id] if is_left_calculated
                else calc_val[hist_id],
            )


def update_pointwise_histograms[
    o1: MutOrigin, o2: MutOrigin, //
](
    ctx: DeviceContext,
    histograms: MutPointer[Float32, o1],
    first_bin_feature: Int,
    bin_feature_count: Int,
    part_count: Int,
    fold_count: Int,
    hist_count: Int,
    hist_line_size: Int,
    parts: MutPointer[UInt32, o2],
) raises:
    """`UpdatePointwiseHistograms` (`:76-101`), copied.

        const int blockSize = 256;
        numBlocks.x = (binFeatureCount + blockSize - 1) / blockSize;
        numBlocks.y = partCount / 2;
        numBlocks.z = foldCount;
        if (IsGridEmpty(numBlocks)) return;

    `numBlocks.y` IS `partCount / 2` UNCONDITIONALLY, with no `fullPass`
    term, because this kernel only ever runs on a partial pass -- one block
    per PAIR of leaves. Its caller guards it (`pointwise_kernels.cpp:82`).
    """
    var nx = (
        bin_feature_count + PW_UPDATE_BLOCK - 1
    ) // PW_UPDATE_BLOCK
    var ny = part_count // 2
    var nz = fold_count
    if is_grid_empty(nx, ny, nz):
        return

    if hist_count == 1:
        ctx.enqueue_function[update_pointwise_histograms_kernel[1]](
            histograms,
            Int32(first_bin_feature),
            Int32(bin_feature_count),
            parts,
            Int32(hist_line_size),
            grid_dim=(nx, ny, nz),
            block_dim=(PW_UPDATE_BLOCK, 1, 1),
        )
    elif hist_count == 2:
        ctx.enqueue_function[update_pointwise_histograms_kernel[2]](
            histograms,
            Int32(first_bin_feature),
            Int32(bin_feature_count),
            parts,
            Int32(hist_line_size),
            grid_dim=(nx, ny, nz),
            block_dim=(PW_UPDATE_BLOCK, 1, 1),
        )
    else:
        # DEVIATION 101: `CB_ENSURE_INTERNAL(false, ...)` (`:99`)
        raise Error(
            "UpdatePointwiseHistograms: histCount should be 1 or 2, not "
            + String(hist_count)
        )


# ---------------------------------------------------------------------------
# `ScanPointwiseHistograms` (`pointwise_hist2.cu:105-131`)
# ---------------------------------------------------------------------------


def scan_pointwise_histograms[
    o1: MutOrigin, o2: MutOrigin, o3: MutOrigin, o4: MutOrigin, //
](
    ctx: DeviceContext,
    feature_first_fold_index: MutPointer[UInt32, o1],
    feature_folds: MutPointer[UInt32, o2],
    feature_one_hot: MutPointer[UInt8, o3],
    feature_count: Int,
    part_count: Int,
    fold_count: Int,
    hist_line_size: Int,
    full_pass: Bool,
    hist_count: Int,
    bin_sums: MutPointer[Float32, o4],
) raises:
    """`ScanPointwiseHistograms` (`:105-131`), copied.

        const int histPartCount = (fullPass ? partCount : partCount / 2);
        scanBlocks.x = (featureCount * 32 + scanBlockSize - 1) / scanBlockSize;
        scanBlocks.y = histPartCount;
        scanBlocks.z = foldCount;
        if (IsGridEmpty(scanBlocks)) return;
        const int scanOffset = fullPass ? 0
            : ((partCount / 2) * histLineSize * histCount) * foldCount;

    THE `scanOffset` IS THE PARTIAL-PASS HALF OF THE BUFFER. On a partial
    pass the drivers wrote into the RIGHT children's slots, which begin at
    part `partCount / 2`; the scan is pointed there and never touches the
    left half, which still holds the parents. Theirs advances the pointer;
    ours advances it the same way with `unsafe_offset`.

    `grid.x` DIFFERS FROM THEIRS AND IT IS NOT A NEW DECISION. Theirs gives
    each feature a 32-lane WARP -- hence `featureCount * 32` threads -- and
    scans 32 bins at a time with `InclusiveScanInWarp`. `archive/reference/PORTING.md` 8
    replaced that with one THREAD per feature, for the reason recorded on
    `scan_pointwise_histograms_kernel`: their loop puts `__syncthreads()`
    inside `if (featureId < featureCount)`, which the tail block reaches
    with only some of its warps, and a partially-reached threadgroup barrier
    hangs on Metal. One thread per feature means `featureCount` threads, so
    `grid.x` loses the `* 32`. The grid is the LAST place that substitution
    shows up; every earlier consequence is already recorded at the kernel.

    `HIST_COUNT` is a template argument of theirs and a runtime argument of
    ours, for the same reason -- one thread walks both stat planes rather
    than two warps walking one each. The 1-or-2 guard is kept anyway, since
    it is the guard and not the templating that protects a caller.
    """
    if hist_count != 1 and hist_count != 2:
        # DEVIATION 101: `CB_ENSURE_INTERNAL(false, ...)` (`:129`)
        raise Error(
            "ScanPointwiseHistograms: histCount should be 1 or 2, not "
            + String(hist_count)
        )

    var hist_part_count = part_count if full_pass else part_count // 2
    var nx = (feature_count + PW_SCAN_BLOCK - 1) // PW_SCAN_BLOCK
    var ny = hist_part_count
    var nz = fold_count
    if is_grid_empty(nx, ny, nz):
        return

    var scan_offset = 0
    if not full_pass:
        scan_offset = (
            (part_count // 2) * hist_line_size * hist_count
        ) * fold_count

    ctx.enqueue_function[scan_pointwise_histograms_kernel](
        feature_first_fold_index,
        feature_folds,
        feature_one_hot,
        Int32(feature_count),
        Int32(hist_line_size),
        Int32(hist_count),
        Int32(fold_count),
        bin_sums.unsafe_offset(scan_offset),
        grid_dim=(nx, ny, nz),
        block_dim=(PW_SCAN_BLOCK, 1, 1),
    )


# ---------------------------------------------------------------------------
# `ComputeHist2NonBinary<Bits>` (`pointwise_hist2_one_byte_templ.cuh:221-270`)
# ---------------------------------------------------------------------------


def run_compute_hist2_non_binary_kernel[
    o1: MutOrigin,
    o2: MutOrigin,
    o3: MutOrigin,
    o4: MutOrigin,
    o5: MutOrigin,
    o6: MutOrigin,
    o7: MutOrigin,
    o8: MutOrigin,
    o9: MutOrigin, //,
    bits: Int,
    blocks_per_feature_count: Int,
](
    ctx: DeviceContext,
    feature_offset: MutPointer[UInt32, o1],
    feature_first_fold_index: MutPointer[UInt32, o2],
    feature_folds: MutPointer[UInt32, o3],
    nb_count: Int,
    cindex: MutPointer[UInt32, o4],
    target: MutPointer[Float32, o5],
    weight: MutPointer[Float32, o6],
    indices: MutPointer[UInt32, o7],
    partition: MutPointer[UInt32, o8],
    bin_sums: MutPointer[Float32, o9],
    bin_feature_count: Int,
    full_pass: Bool,
    fixed_scale: Float32,
    nx: Int,
    ny: Int,
    nz: Int,
) raises:
    """`RunComputeHist2NonBinaryKernel` (`:190-216`), copied.

    A `bool` selecting between two template instantiations, which is what
    theirs is and all that theirs is.

    THE BLOCK IS PER BIT WIDTH, matching the kernel's own geometry: the
    8-bit width takes the route-keyed block (`pw_hist2_block_size_for`),
    the float widths always take their dispatch's. Under their dispatch
    the two are the same value.
    """
    comptime nb_block = PW_HIST2_BLOCK if bits == 8 else (
        PW_HIST2_FLOAT_BLOCK
    )
    if full_pass:
        ctx.enqueue_function[
            compute_split_properties_nb_kernel[
                bits, True, blocks_per_feature_count
            ]
        ](
            feature_offset,
            feature_first_fold_index,
            feature_folds,
            Int32(nb_count),
            cindex,
            target,
            weight,
            indices,
            partition,
            bin_sums,
            Int32(bin_feature_count),
            fixed_scale,
            grid_dim=(nx, ny, nz),
            block_dim=(nb_block, 1, 1),
        )
    else:
        ctx.enqueue_function[
            compute_split_properties_nb_kernel[
                bits, False, blocks_per_feature_count
            ]
        ](
            feature_offset,
            feature_first_fold_index,
            feature_folds,
            Int32(nb_count),
            cindex,
            target,
            weight,
            indices,
            partition,
            bin_sums,
            Int32(bin_feature_count),
            fixed_scale,
            grid_dim=(nx, ny, nz),
            block_dim=(nb_block, 1, 1),
        )


def compute_hist2_non_binary[
    o1: MutOrigin,
    o2: MutOrigin,
    o3: MutOrigin,
    o4: MutOrigin,
    o5: MutOrigin,
    o6: MutOrigin,
    o7: MutOrigin,
    o8: MutOrigin,
    o9: MutOrigin, //,
    bits: Int,
](
    ctx: DeviceContext,
    feature_offset: MutPointer[UInt32, o1],
    feature_first_fold_index: MutPointer[UInt32, o2],
    feature_folds: MutPointer[UInt32, o3],
    nb_count: Int,
    cindex: MutPointer[UInt32, o4],
    target: MutPointer[Float32, o5],
    weight: MutPointer[Float32, o6],
    indices: MutPointer[UInt32, o7],
    size: Int,
    partition: MutPointer[UInt32, o8],
    part_count: Int,
    fold_count: Int,
    full_pass: Bool,
    hist_line_size: Int,
    bin_sums: MutPointer[Float32, o9],
    feature_count_for_bits: Int,
    sm_count: Int,
    fixed_scale: Float32,
) raises:
    """`ComputeHist2NonBinary<Bits>` (`:221-270`), copied line for line.

        if (featureCountForBits) {
            numBlocks.y = (fullPass ? partCount : partCount / 2);
            numBlocks.z = foldCount;
            numBlocks.x = (featureCountForBits + 3) / 4;
            multiplier = min(EstimateBlockPerFeatureMultiplier(numBlocks, size), 64);
            numBlocks.x = ((nbCount + 3) / 4);
            numBlocks.x *= multiplier;
            if (IsGridEmpty(numBlocks)) return;
            ... COMPUTE(1|2|4|8|16|32|64) else exit(1)
        }

    THE OUTER `if (featureCountForBits)` IS THE ONLY HOST-SIDE DISPATCH IN
    THE FAMILY. Everything else about which kernel handles which feature is
    decided on the device. All it does is skip the launch entirely for a bit
    width no feature needs -- and note it does NOT restrict the grid to
    those features, which is the double-`numBlocks.x` finding in the module
    docstring.

    `size` IS THE DOCUMENT COUNT, NOT THE FEATURE COUNT. It reaches
    `EstimateBlockPerFeatureMultiplier` as `dsSize` and gates the split at
    `(dsSize / multiplier) > 10000`: the document axis is never divided into
    pieces smaller than ~10k rows, so a small dataset gets `multiplier == 1`
    however empty the machine is. Theirs passes `Indices.Size()`
    (`pointwise_kernels.cpp:24`).
    """
    if feature_count_for_bits == 0:
        return

    var hist_part_count = part_count if full_pass else part_count // 2
    var ny = hist_part_count
    var nz = fold_count

    # `:240` -- sizes the ESTIMATE, from the features this width will serve
    var nx = (
        feature_count_for_bits + PW_NB_FEATURES_PER_BLOCK - 1
    ) // PW_NB_FEATURES_PER_BLOCK
    var multiplier = estimate_block_per_feature_multiplier(
        nx, ny, nz, size, sm_count
    )
    if multiplier > PW_MAX_MULTIPLIER:
        multiplier = PW_MAX_MULTIPLIER

    # `:242` -- sizes the LAUNCH, over EVERY one-byte feature
    nx = (
        nb_count + PW_NB_FEATURES_PER_BLOCK - 1
    ) // PW_NB_FEATURES_PER_BLOCK
    nx *= multiplier
    if is_grid_empty(nx, ny, nz):
        return

    if multiplier == 1:
        run_compute_hist2_non_binary_kernel[bits, 1](
            ctx, feature_offset, feature_first_fold_index, feature_folds,
            nb_count, cindex, target, weight, indices, partition, bin_sums,
            hist_line_size, full_pass, fixed_scale, nx, ny, nz,
        )
    elif multiplier == 2:
        run_compute_hist2_non_binary_kernel[bits, 2](
            ctx, feature_offset, feature_first_fold_index, feature_folds,
            nb_count, cindex, target, weight, indices, partition, bin_sums,
            hist_line_size, full_pass, fixed_scale, nx, ny, nz,
        )
    elif multiplier == 4:
        run_compute_hist2_non_binary_kernel[bits, 4](
            ctx, feature_offset, feature_first_fold_index, feature_folds,
            nb_count, cindex, target, weight, indices, partition, bin_sums,
            hist_line_size, full_pass, fixed_scale, nx, ny, nz,
        )
    elif multiplier == 8:
        run_compute_hist2_non_binary_kernel[bits, 8](
            ctx, feature_offset, feature_first_fold_index, feature_folds,
            nb_count, cindex, target, weight, indices, partition, bin_sums,
            hist_line_size, full_pass, fixed_scale, nx, ny, nz,
        )
    elif multiplier == 16:
        run_compute_hist2_non_binary_kernel[bits, 16](
            ctx, feature_offset, feature_first_fold_index, feature_folds,
            nb_count, cindex, target, weight, indices, partition, bin_sums,
            hist_line_size, full_pass, fixed_scale, nx, ny, nz,
        )
    elif multiplier == 32:
        run_compute_hist2_non_binary_kernel[bits, 32](
            ctx, feature_offset, feature_first_fold_index, feature_folds,
            nb_count, cindex, target, weight, indices, partition, bin_sums,
            hist_line_size, full_pass, fixed_scale, nx, ny, nz,
        )
    elif multiplier == 64:
        run_compute_hist2_non_binary_kernel[bits, 64](
            ctx, feature_offset, feature_first_fold_index, feature_folds,
            nb_count, cindex, target, weight, indices, partition, bin_sums,
            hist_line_size, full_pass, fixed_scale, nx, ny, nz,
        )
    else:
        # DEVIATION 101: theirs is `exit(1)` (`:266`)
        raise Error(
            "ComputeHist2NonBinary: no kernel instantiated for multiplier "
            + String(multiplier)
            + " (expected one of 1, 2, 4, 8, 16, 32, 64)"
        )


# ---------------------------------------------------------------------------
# `ComputeHist2Binary` (`pointwise_hist2_binary.cu:128-179`)
# ---------------------------------------------------------------------------


def run_compute_hist2_binary_kernel[
    o1: MutOrigin,
    o2: MutOrigin,
    o4: MutOrigin,
    o5: MutOrigin,
    o6: MutOrigin,
    o7: MutOrigin,
    o8: MutOrigin,
    o9: MutOrigin, //,
    blocks_per_feature_count: Int,
](
    ctx: DeviceContext,
    feature_offset: MutPointer[UInt32, o1],
    feature_first_fold_index: MutPointer[UInt32, o2],
    b_count: Int,
    cindex: MutPointer[UInt32, o4],
    target: MutPointer[Float32, o5],
    weight: MutPointer[Float32, o6],
    indices: MutPointer[UInt32, o7],
    partition: MutPointer[UInt32, o8],
    bin_sums: MutPointer[Float32, o9],
    total_feature_count: Int,
    full_pass: Bool,
    nx: Int,
    ny: Int,
    nz: Int,
) raises:
    """`RunComputeHist2BinaryKernel` (`:98-124`), copied."""
    if full_pass:
        ctx.enqueue_function[
            compute_split_properties_b_kernel[True, blocks_per_feature_count]
        ](
            feature_offset,
            feature_first_fold_index,
            Int32(b_count),
            cindex,
            target,
            weight,
            indices,
            partition,
            bin_sums,
            Int32(total_feature_count),
            grid_dim=(nx, ny, nz),
            block_dim=(PW_HB_BLOCK, 1, 1),
        )
    else:
        ctx.enqueue_function[
            compute_split_properties_b_kernel[False, blocks_per_feature_count]
        ](
            feature_offset,
            feature_first_fold_index,
            Int32(b_count),
            cindex,
            target,
            weight,
            indices,
            partition,
            bin_sums,
            Int32(total_feature_count),
            grid_dim=(nx, ny, nz),
            block_dim=(PW_HB_BLOCK, 1, 1),
        )


def compute_hist2_binary[
    o1: MutOrigin,
    o2: MutOrigin,
    o4: MutOrigin,
    o5: MutOrigin,
    o6: MutOrigin,
    o7: MutOrigin,
    o8: MutOrigin,
    o9: MutOrigin, //,
](
    ctx: DeviceContext,
    feature_offset: MutPointer[UInt32, o1],
    feature_first_fold_index: MutPointer[UInt32, o2],
    b_count: Int,
    cindex: MutPointer[UInt32, o4],
    target: MutPointer[Float32, o5],
    weight: MutPointer[Float32, o6],
    indices: MutPointer[UInt32, o7],
    size: Int,
    partition: MutPointer[UInt32, o8],
    parts_count: Int,
    fold_count: Int,
    full_pass: Bool,
    total_feature_count: Int,
    bin_sums: MutPointer[Float32, o9],
    sm_count: Int,
) raises:
    """`ComputeHist2Binary` (`:128-179`), copied line for line.

        numBlocks.x = (bCount + 31) / 32;
        const int histCount = fullPass ? partsCount : partsCount / 2;
        numBlocks.y = histCount; numBlocks.z = foldCount;
        const ui32 multiplier = min(EstimateBlockPerFeatureMultiplier(numBlocks, size), 64);
        numBlocks.x *= multiplier;
        if (IsGridEmpty(numBlocks)) return;
        if (bCount) { ... }

    THE ORDER OF THE LAST TWO GUARDS IS THEIRS AND IS NOT THE OBVIOUS ONE.
    `IsGridEmpty` comes FIRST and `if (bCount)` second, even though a zero
    `bCount` already makes `numBlocks.x` zero and the second test therefore
    unreachable. Kept in their order: reordering it would be an improvement,
    and improvements are how a port stops being one.

    UNLIKE THE ONE-BYTE LAUNCHER `numBlocks.x` IS COMPUTED ONCE. There is no
    second feature count here because there is no runtime bit-width
    partition -- every feature reaching this kernel is binary by
    construction, so the estimate and the launch see the same work.
    """
    var nx = (
        b_count + PW_B_FEATURES_PER_BLOCK - 1
    ) // PW_B_FEATURES_PER_BLOCK
    var hist_count = parts_count if full_pass else parts_count // 2
    var ny = hist_count
    var nz = fold_count

    var multiplier = estimate_block_per_feature_multiplier(
        nx, ny, nz, size, sm_count
    )
    if multiplier > PW_MAX_MULTIPLIER:
        multiplier = PW_MAX_MULTIPLIER
    nx *= multiplier
    if is_grid_empty(nx, ny, nz):
        return

    if b_count == 0:
        return

    if multiplier == 1:
        run_compute_hist2_binary_kernel[1](
            ctx, feature_offset, feature_first_fold_index, b_count, cindex,
            target, weight, indices, partition, bin_sums,
            total_feature_count, full_pass, nx, ny, nz,
        )
    elif multiplier == 2:
        run_compute_hist2_binary_kernel[2](
            ctx, feature_offset, feature_first_fold_index, b_count, cindex,
            target, weight, indices, partition, bin_sums,
            total_feature_count, full_pass, nx, ny, nz,
        )
    elif multiplier == 4:
        run_compute_hist2_binary_kernel[4](
            ctx, feature_offset, feature_first_fold_index, b_count, cindex,
            target, weight, indices, partition, bin_sums,
            total_feature_count, full_pass, nx, ny, nz,
        )
    elif multiplier == 8:
        run_compute_hist2_binary_kernel[8](
            ctx, feature_offset, feature_first_fold_index, b_count, cindex,
            target, weight, indices, partition, bin_sums,
            total_feature_count, full_pass, nx, ny, nz,
        )
    elif multiplier == 16:
        run_compute_hist2_binary_kernel[16](
            ctx, feature_offset, feature_first_fold_index, b_count, cindex,
            target, weight, indices, partition, bin_sums,
            total_feature_count, full_pass, nx, ny, nz,
        )
    elif multiplier == 32:
        run_compute_hist2_binary_kernel[32](
            ctx, feature_offset, feature_first_fold_index, b_count, cindex,
            target, weight, indices, partition, bin_sums,
            total_feature_count, full_pass, nx, ny, nz,
        )
    elif multiplier == 64:
        run_compute_hist2_binary_kernel[64](
            ctx, feature_offset, feature_first_fold_index, b_count, cindex,
            target, weight, indices, partition, bin_sums,
            total_feature_count, full_pass, nx, ny, nz,
        )
    else:
        # DEVIATION 101: theirs is `exit(1)` (`:174`)
        raise Error(
            "ComputeHist2Binary: no kernel instantiated for multiplier "
            + String(multiplier)
            + " (expected one of 1, 2, 4, 8, 16, 32, 64)"
        )


# ---------------------------------------------------------------------------
# `ComputeHist2HalfByte` (`pointwise_hist2_half_byte.cu:130-180`)
# ---------------------------------------------------------------------------


def run_compute_hist2_half_byte_kernel[
    o1: MutOrigin,
    o2: MutOrigin,
    o3: MutOrigin,
    o4: MutOrigin,
    o5: MutOrigin,
    o6: MutOrigin,
    o7: MutOrigin,
    o8: MutOrigin,
    o9: MutOrigin, //,
    blocks_per_feature_count: Int,
](
    ctx: DeviceContext,
    feature_offset: MutPointer[UInt32, o1],
    feature_first_fold_index: MutPointer[UInt32, o2],
    feature_folds: MutPointer[UInt32, o3],
    nb_count: Int,
    cindex: MutPointer[UInt32, o4],
    target: MutPointer[Float32, o5],
    weight: MutPointer[Float32, o6],
    indices: MutPointer[UInt32, o7],
    partition: MutPointer[UInt32, o8],
    bin_sums: MutPointer[Float32, o9],
    bin_feature_count: Int,
    full_pass: Bool,
    nx: Int,
    ny: Int,
    nz: Int,
) raises:
    """`RunComputeHist2HalfByteKernel` (`:99-127`), copied."""
    if full_pass:
        ctx.enqueue_function[
            compute_split_properties_half_byte_kernel[
                True, blocks_per_feature_count
            ]
        ](
            feature_offset,
            feature_first_fold_index,
            feature_folds,
            Int32(nb_count),
            cindex,
            target,
            weight,
            indices,
            partition,
            bin_sums,
            Int32(bin_feature_count),
            grid_dim=(nx, ny, nz),
            block_dim=(PW_HB_BLOCK, 1, 1),
        )
    else:
        ctx.enqueue_function[
            compute_split_properties_half_byte_kernel[
                False, blocks_per_feature_count
            ]
        ](
            feature_offset,
            feature_first_fold_index,
            feature_folds,
            Int32(nb_count),
            cindex,
            target,
            weight,
            indices,
            partition,
            bin_sums,
            Int32(bin_feature_count),
            grid_dim=(nx, ny, nz),
            block_dim=(PW_HB_BLOCK, 1, 1),
        )


def compute_hist2_half_byte[
    o1: MutOrigin,
    o2: MutOrigin,
    o3: MutOrigin,
    o4: MutOrigin,
    o5: MutOrigin,
    o6: MutOrigin,
    o7: MutOrigin,
    o8: MutOrigin,
    o9: MutOrigin, //,
](
    ctx: DeviceContext,
    feature_offset: MutPointer[UInt32, o1],
    feature_first_fold_index: MutPointer[UInt32, o2],
    feature_folds: MutPointer[UInt32, o3],
    half_byte_features_count: Int,
    cindex: MutPointer[UInt32, o4],
    target: MutPointer[Float32, o5],
    weight: MutPointer[Float32, o6],
    indices: MutPointer[UInt32, o7],
    size: Int,
    partition: MutPointer[UInt32, o8],
    parts_count: Int,
    fold_count: Int,
    full_pass: Bool,
    hist_line_size: Int,
    bin_sums: MutPointer[Float32, o9],
    sm_count: Int,
) raises:
    """`ComputeHist2HalfByte` (`:130-180`), copied line for line.

        numBlocks.x = (halfByteFeaturesCount + 7) / 8;
        const int histCount = fullPass ? partsCount : partsCount / 2;
        ... same shape as the binary launcher, /8 instead of /32
    """
    var nx = (
        half_byte_features_count + PW_HB_FEATURES_PER_BLOCK - 1
    ) // PW_HB_FEATURES_PER_BLOCK
    var hist_count = parts_count if full_pass else parts_count // 2
    var ny = hist_count
    var nz = fold_count

    var multiplier = estimate_block_per_feature_multiplier(
        nx, ny, nz, size, sm_count
    )
    if multiplier > PW_MAX_MULTIPLIER:
        multiplier = PW_MAX_MULTIPLIER
    nx *= multiplier
    if is_grid_empty(nx, ny, nz):
        return

    if half_byte_features_count == 0:
        return

    if multiplier == 1:
        run_compute_hist2_half_byte_kernel[1](
            ctx, feature_offset, feature_first_fold_index, feature_folds,
            half_byte_features_count, cindex, target, weight, indices,
            partition, bin_sums, hist_line_size, full_pass, nx, ny, nz,
        )
    elif multiplier == 2:
        run_compute_hist2_half_byte_kernel[2](
            ctx, feature_offset, feature_first_fold_index, feature_folds,
            half_byte_features_count, cindex, target, weight, indices,
            partition, bin_sums, hist_line_size, full_pass, nx, ny, nz,
        )
    elif multiplier == 4:
        run_compute_hist2_half_byte_kernel[4](
            ctx, feature_offset, feature_first_fold_index, feature_folds,
            half_byte_features_count, cindex, target, weight, indices,
            partition, bin_sums, hist_line_size, full_pass, nx, ny, nz,
        )
    elif multiplier == 8:
        run_compute_hist2_half_byte_kernel[8](
            ctx, feature_offset, feature_first_fold_index, feature_folds,
            half_byte_features_count, cindex, target, weight, indices,
            partition, bin_sums, hist_line_size, full_pass, nx, ny, nz,
        )
    elif multiplier == 16:
        run_compute_hist2_half_byte_kernel[16](
            ctx, feature_offset, feature_first_fold_index, feature_folds,
            half_byte_features_count, cindex, target, weight, indices,
            partition, bin_sums, hist_line_size, full_pass, nx, ny, nz,
        )
    elif multiplier == 32:
        run_compute_hist2_half_byte_kernel[32](
            ctx, feature_offset, feature_first_fold_index, feature_folds,
            half_byte_features_count, cindex, target, weight, indices,
            partition, bin_sums, hist_line_size, full_pass, nx, ny, nz,
        )
    elif multiplier == 64:
        run_compute_hist2_half_byte_kernel[64](
            ctx, feature_offset, feature_first_fold_index, feature_folds,
            half_byte_features_count, cindex, target, weight, indices,
            partition, bin_sums, hist_line_size, full_pass, nx, ny, nz,
        )
    else:
        # DEVIATION 101: theirs is `exit(1)` (`:175`)
        raise Error(
            "ComputeHist2HalfByte: no kernel instantiated for multiplier "
            + String(multiplier)
            + " (expected one of 1, 2, 4, 8, 16, 32, 64)"
        )


# ---------------------------------------------------------------------------
# `TComputeHist2Kernel::Run` (`pointwise_kernels.cpp:17-93`)
# ---------------------------------------------------------------------------


def compute_hist2[
    o1: MutOrigin,
    o2: MutOrigin,
    o3: MutOrigin,
    oh: MutOrigin,
    o4: MutOrigin,
    o5: MutOrigin,
    o6: MutOrigin,
    o7: MutOrigin,
    o8: MutOrigin,
    o9: MutOrigin, //,
](
    ctx: DeviceContext,
    policy: Int,
    feature_offset: MutPointer[UInt32, o1],
    feature_first_fold_index: MutPointer[UInt32, o2],
    feature_folds: MutPointer[UInt32, o3],
    feature_one_hot: MutPointer[UInt8, oh],
    feature_count: Int,
    bin_features_slice_left: Int,
    bin_features_slice_size: Int,
    cindex: MutPointer[UInt32, o4],
    target: MutPointer[Float32, o5],
    weight: MutPointer[Float32, o6],
    indices: MutPointer[UInt32, o7],
    size: Int,
    partition: MutPointer[UInt32, o8],
    part_count: Int,
    fold_count: Int,
    bin_sums: MutPointer[Float32, o9],
    hist_line_size: Int,
    full_pass: Bool,
    folds_hist: FoldsHistogram,
    sm_count: Int,
    fixed_scale: Float32,
) raises:
    """`TComputeHist2Kernel::Run` (`pointwise_kernels.cpp:17-93`), copied.

    THE TOP-LEVEL FAN-OUT, and it is three steps in a fixed order:

        1. switch on POLICY. Binary and half-byte are one call each;
           OneByteFeatures is FOUR calls, one per bit width, each given its
           own `FeatureCountForBits` and all four launched over every
           one-byte feature.
        2. SCAN, for every policy but binary (`:70`). Binary features have
           one fold, so their prefix sum is the identity and theirs skips
           the launch rather than doing nothing 32 times.
        3. on a PARTIAL pass only (`:82`), `UpdatePointwiseHistograms` to
           recover the sibling by subtraction.

    THE FOUR ONE-BYTE CALLS ARE NOT GUARDED BY EACH OTHER. They all run,
    into the same `binSums`, and the writes do not collide because each
    kernel refuses the blocks outside its range on the device. That is the
    design `pointwise_hist2_one_byte_templ.mojo` step 3 documents, and it is
    why the host does not need to know a single feature's fold count to
    dispatch correctly -- only how MANY features fall in each width, to size
    the multiplier.

    `BinFeaturesSlice` (their `TSlice`) reaches only the subtraction step,
    as `Left` and `Size` (`:84-85`). Their two constructors differ in
    exactly this: the whole-grid one sets it to `TSlice(0, binFeatureCount)`
    and the block one takes the caller's slice, which is how
    `ComputeBlockHistogram2` fixes up only the block it computed. Passed as
    two Ints here for the reason DEVIATION 102 gives.
    """
    if policy == POLICY_BINARY:
        compute_hist2_binary(
            ctx,
            feature_offset,
            feature_first_fold_index,
            feature_count,
            cindex,
            target,
            weight,
            indices,
            size,
            partition,
            part_count,
            fold_count,
            full_pass,
            hist_line_size,
            bin_sums,
            sm_count,
        )
    elif policy == POLICY_HALF_BYTE:
        compute_hist2_half_byte(
            ctx,
            feature_offset,
            feature_first_fold_index,
            feature_folds,
            feature_count,
            cindex,
            target,
            weight,
            indices,
            size,
            partition,
            part_count,
            fold_count,
            full_pass,
            hist_line_size,
            bin_sums,
            sm_count,
        )
    elif policy == POLICY_ONE_BYTE:
        # `DISPATCH_ONE_BYTE(ComputeHist2NonBinary, 4, 5)` and the three
        # that follow (`:57-60`). Note the 4 in the first: see
        # `FoldsHistogram`.
        # the Apple/IDENTICAL routing row sends every one-byte width
        # through the 8-bit fixed-point kernel (its `pw_bounds` widen to
        # match); the narrow float accumulators' turn-taking costs a full
        # threadgroup barrier per turn on a column with no warp barrier.
        # See `pointwise_one_byte_fixed_for`.
        comptime if pointwise_one_byte_fixed_for[
            TARGET_COLUMN, HIST_BUILD_MODE == NUMERIC_IDENTICAL
        ]():
            compute_hist2_non_binary[8](
                ctx, feature_offset, feature_first_fold_index,
                feature_folds,
                feature_count, cindex, target, weight, indices, size,
                partition,
                part_count, fold_count, full_pass, hist_line_size,
                bin_sums,
                folds_hist.feature_count_for_bits(4, 8), sm_count,
                fixed_scale,
            )
        else:
            compute_hist2_non_binary[5](
                ctx, feature_offset, feature_first_fold_index,
                feature_folds,
                feature_count, cindex, target, weight, indices, size,
                partition,
                part_count, fold_count, full_pass, hist_line_size,
                bin_sums,
                folds_hist.feature_count_for_bits(4, 5), sm_count,
                fixed_scale,
            )
            compute_hist2_non_binary[6](
                ctx, feature_offset, feature_first_fold_index,
                feature_folds,
                feature_count, cindex, target, weight, indices, size,
                partition,
                part_count, fold_count, full_pass, hist_line_size,
                bin_sums,
                folds_hist.feature_count_for_bits(6, 6), sm_count,
                fixed_scale,
            )
            compute_hist2_non_binary[7](
                ctx, feature_offset, feature_first_fold_index,
                feature_folds,
                feature_count, cindex, target, weight, indices, size,
                partition,
                part_count, fold_count, full_pass, hist_line_size,
                bin_sums,
                folds_hist.feature_count_for_bits(7, 7), sm_count,
                fixed_scale,
            )
            compute_hist2_non_binary[8](
                ctx, feature_offset, feature_first_fold_index,
                feature_folds,
                feature_count, cindex, target, weight, indices, size,
                partition,
                part_count, fold_count, full_pass, hist_line_size,
                bin_sums,
                folds_hist.feature_count_for_bits(8, 8), sm_count,
                fixed_scale,
            )
    else:
        # `CB_ENSURE(false, "Unexpected feature grouping policy")` (`:64`)
        raise Error(
            "ComputeHist2: unexpected feature grouping policy "
            + String(policy)
        )

    if policy != POLICY_BINARY:
        scan_pointwise_histograms(
            ctx,
            feature_first_fold_index,
            feature_folds,
            feature_one_hot,
            feature_count,
            part_count,
            fold_count,
            hist_line_size,
            full_pass,
            2,
            bin_sums,
        )

    if not full_pass:
        update_pointwise_histograms(
            ctx,
            bin_sums,
            bin_features_slice_left,
            bin_features_slice_size,
            part_count,
            fold_count,
            2,
            hist_line_size,
            partition,
        )
