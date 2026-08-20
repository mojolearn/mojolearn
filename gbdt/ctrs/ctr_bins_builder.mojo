"""`TCtrBinBuilder`: rows in bin order, with the segment boundaries marked.

MIRRORS `catboost/cuda/ctrs/ctr_bins_builder.h` at CatBoost `54a8143a`.

Every CTR in their design is a SEGMENTED reduction over rows grouped by
category, so before any calcer runs, something has to produce

    indices   the rows, ordered by (accumulated bin), each carrying a
              segment-start flag in bit 31 (`index_wrapper.mojo`)
    bins      the accumulated bin id at each SORTED position

and that is this class. `AddCompressedBins` folds one more cat feature into
the ordering (`:104-110`), which is how a feature TENSOR -- a combination --
is built up one feature at a time; a simple CTR calls it once.

Their `ProceedNewBins` (`:214-222`) is four steps and this file keeps them
in that order:

    ComputeCurrentBins(Indices, ...)  -> CurrentBins, the segment id of
                                         each ORIGINAL row under the
                                         ordering built so far
    GatherWithMask(Bins, rawBins, Indices, Mask)
    ReorderBins(Bins, Indices, 0, IntLog2(uniqueValues), ...)
    UpdateBordersMask(Bins, CurrentBins, Indices)

## TWO BUILDERS, AND WHICH ONE IS THE PORT

`TCtrBinBuilderGpu` at the bottom of this file IS the port: it runs
`GatherWithMask`, `ReorderBins`, `ComputeCurrentBins` and
`UpdateBordersMask` on the device, through the kernels their own code
calls. The `Gpu` suffix is OURS -- CatBoost has one class because it has no
host arm at all.

`TCtrBinBuilder` above it is the HOST reference. It computes the same
thing in host loops, it is what `mojo_only/ctr_check.mojo` gates the device
answer against cell by cell, and the FeatureFreq calcer still runs on it.
That half is what remains of `PORTING.md` deviation 52; the bin ordering
half is retired.

`_stable_sort_by_bin` stands in for `ReorderBins` in the host builder. It
has to be STABLE, and so does the device radix sort that replaced it in the
real one: the order of rows WITHIN a category is the CTR ESTIMATION
PERMUTATION, which is the history an ordered target statistic reads. An
unstable sort is invisible to any FeatureFreq check (counts do not care
about order) and silently randomizes every `Borders` value.

## THE PERMUTATION ENTERS HERE, AND NOWHERE ELSE

Their `SetIndices` (`:32-52`) is handed `ctrEstimationOrder`, a device
buffer that `doc_parallel_dataset_builder.cpp` fills with either
`MakeSequence` (the identity, for permutation-INDEPENDENT CTRs, `:206`) or
`ctrsEstimationPermutation.WriteOrder(...)` (a shuffle, for the dependent
ones, `:255`). Nothing else in the CTR block knows about permutations: the
gather reads through `Indices`, the stable sort preserves the tie order,
and the segmented scan sums "the rows of my bin that came before me" in
whatever order the initial `Indices` established. So both builders take an
ORDER rather than a row count, `order[i]` being the original row at
position `i` -- `gbdt/data/permutation.mojo` and `PORTING.md` 55.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from gbdt.ctrs.ctr import TCtrConfig, CTR_FEATURE_FREQ
from gbdt.ctrs.index_wrapper import (
    CTR_INDEX_MASK,
    index_of,
    is_segment_start,
    with_mask,
)
from gbdt.ctrs.kernel.ctr_calcers import (
    launch_extract_border_masks,
    launch_update_borders_mask,
)
from gbdt.gpu_util.kernel.radix_sort import launch_radix_sort_bins
from gbdt.gpu_util.kernel.scan import SCAN_BLOCK, launch_scan_vector_u32
from gbdt.gpu_util.kernel.transform import (
    launch_gather_with_mask_u32,
    launch_scatter_with_mask_u32,
)


def int_log2(unique_values: Int) -> Int:
    """`NCB::IntLog2(uniqueValues)` (`ctr_bins_builder.h:217`), which is
    `ReorderBins`' bit count.

    Theirs is `(ui32)ceil(log2(values))` (`libs/helpers/math_utils.h:14-16`,
    with their own TODO calling it slow). This is the same number computed
    as the BIT LENGTH of `values - 1`, which is that identity for every
    `values >= 1`, and which cannot land on the wrong side of an integer
    boundary the way `ceil(log2(x))` in float can. The two were compared over
    every `values` in `[1, 1 << 20]` while this was written.

    It is what a 16-category feature sorting 4 bits and a 17-category one
    sorting 5 comes from, so the 15/16/17 sweep in `ctr_check.mojo` is
    aimed straight at it.
    """
    var bits = 0
    var v = unique_values - 1
    while v > 0:
        bits += 1
        v >>= 1
    return bits


def _stable_sort_by_bin(
    mut bins: List[UInt32], mut indices: List[UInt32]
) raises:
    """Their `ReorderBins(Bins, Indices, 0, IntLog2(uniqueValues), ...)`
    (`ctr_bins_builder.h:218`), which is an LSD radix sort over the bin
    bits carrying `Indices` as the payload.

    Counting sort by bin value, host side. STABLE, as theirs is, and the
    stability is load bearing: within one category the surviving order is
    the learn permutation, which is the "history" an ordered target
    statistic scans.

    NOTE the flag: sorting moves the WHOLE index word including whatever
    segment-start bit an earlier feature set, which is theirs -- the
    payload is `Value()`, not `Index()`.
    """
    var n = len(bins)
    if n == 0:
        return
    var max_bin = 0
    for i in range(n):
        if Int(bins[i]) > max_bin:
            max_bin = Int(bins[i])

    var counts = List[Int]()
    for _ in range(max_bin + 2):
        counts.append(0)
    for i in range(n):
        counts[Int(bins[i]) + 1] += 1
    for b in range(1, max_bin + 2):
        counts[b] += counts[b - 1]

    var out_bins = List[UInt32]()
    var out_indices = List[UInt32]()
    for _ in range(n):
        out_bins.append(UInt32(0))
        out_indices.append(UInt32(0))
    for i in range(n):
        var b = Int(bins[i])
        var pos = counts[b]
        counts[b] = pos + 1
        out_bins[pos] = bins[i]
        out_indices[pos] = indices[i]
    bins = out_bins^
    indices = out_indices^


struct TCtrBinBuilder(Movable):
    """Their `TCtrBinBuilder<TMapping>`, the single-device, learn-only case.

    `TestSlice` is not carried. Their builder appends the test rows after
    the learn rows and offsets their indices (`:44-49`) so that one CTR
    pass covers both; this port has no test pool in `train()`, and adding
    an always-empty slice would be a field no branch reads.
    """

    var indices: List[UInt32]
    """Their `Indices`: row ids in bin order, segment start in bit 31."""

    var bins: List[UInt32]
    """Their `Bins`: the accumulated bin at each SORTED position."""

    var current_bins: List[UInt32]
    """Their `CurrentBins`: per ORIGINAL row, the segment id under the
    ordering built by the features added SO FAR. `UpdateBordersMask` reads
    it to keep a combination segmented on the whole tensor rather than on
    the newest feature."""

    var learn_size: Int
    """Their `LearnSlice.Size()`, which is `firstZeroIndex` for
    `GatherTrivialWeights`."""

    def __init__(out self, n_rows: Int):
        """Their `SetIndices` for the learn-only case (`:32-52`) followed
        by `Reset` (`:190-194`) over the IDENTITY order -- their
        `MakeSequence(ctrEstimationOrder)`
        (`doc_parallel_dataset_builder.cpp:206`), which is the order the
        permutation-INDEPENDENT CTRs are written from (`:229`). No flags
        set, so the whole dataset is ONE segment until a feature is
        added."""
        self.indices = List[UInt32]()
        self.bins = List[UInt32]()
        self.current_bins = List[UInt32]()
        self.learn_size = n_rows
        for r in range(n_rows):
            self.indices.append(UInt32(r))
            self.bins.append(UInt32(0))
            self.current_bins.append(UInt32(0))

    def __init__(out self, var order: List[UInt32]) raises:
        """The same `SetIndices`, over an arbitrary CTR ESTIMATION ORDER --
        their `ctrsEstimationPermutation.WriteOrder(ctrEstimationOrder)`
        (`doc_parallel_dataset_builder.cpp:255`).

        `order[i]` is the ORIGINAL row that sits at position `i`. The top
        bit is the segment-start flag and must be clear on the way in;
        their `WriteOrder` writes raw row ids for the same reason.
        """
        var n = len(order)
        self.learn_size = n
        self.bins = List[UInt32]()
        self.current_bins = List[UInt32]()
        for i in range(n):
            if (order[i] & ~CTR_INDEX_MASK) != UInt32(0):
                raise Error(
                    "ctr estimation order entry "
                    + String(i)
                    + " has bits above the 0x3FFFFFFF index mask set; the"
                    " order carries row ids only, the flag bits are"
                    " UpdateBordersMask's to write"
                )
            self.bins.append(UInt32(0))
            self.current_bins.append(UInt32(0))
        self.indices = order^

    @staticmethod
    def from_indices(var indices: List[UInt32]) -> Self:
        """Adopt a FINISHED index array -- flags already written -- so the
        host freq calcer can read an ordering the DEVICE builder produced.

        NOT THEIRS: their builder is one object and the calcers take
        `binBuilder.GetIndices()` from it. This is the seam
        `TCtrBinBuilderGpu.read_indices` writes into, and it exists only
        because the freq calcer has not moved to the device yet
        (`PORTING.md` 52). `Bins` and `CurrentBins` are left empty because
        no freq arm reads either: both walk `Indices` alone.
        """
        var b = Self(0)
        b.learn_size = len(indices)
        b.indices = indices^
        return b^

    def size(self) -> Int:
        return len(self.indices)

    def compute_current_bins(mut self):
        """Their static `ComputeCurrentBins` (`:134-142`).

            ExtractMask(indices, dst, false)   // END flags
            ScanVector(dst, tmp, false)        // EXCLUSIVE scan
            ScatterWithMask(dst, tmp, indices, Mask)

        The exclusive scan of END flags at position `i` counts the segments
        that closed strictly before `i`, which is `i`'s own segment index;
        the scatter moves it from sorted position to original row.
        """
        var n = len(self.indices)
        var running = UInt32(0)
        for i in range(n):
            # the scan is EXCLUSIVE, so the value written is the running
            # total BEFORE this position's own end flag
            self.current_bins[Int(index_of(self.indices[i]))] = running
            var is_end: Bool
            if i + 1 < n:
                is_end = is_segment_start(self.indices[i + 1])
            else:
                is_end = True
            if is_end:
                running += UInt32(1)

    def add_cat_feature_bins(
        mut self, cat_bins: List[UInt32], unique_values: Int
    ) raises:
        """Their `AddCompressedBins` -> `ProceedNewBins` (`:104-110`,
        `:224-231`), with the decompression step dropped because this port
        holds dense category codes rather than their packed `ui64` blocks.

        `unique_values` is their `FeaturesManager.GetBinCount(feature)` and
        is carried because `ReorderBins` sorts exactly `IntLog2(uniqueValues)`
        bits; the counting sort below derives the same range from the data,
        so the argument is validated rather than used.
        """
        if len(cat_bins) != len(self.indices):
            raise Error(
                "cat bins size "
                + String(len(cat_bins))
                + " does not match the builder's "
                + String(len(self.indices))
            )
        if unique_values <= 1:
            # their `CB_ENSURE(uniqueValues > 1, "Error: useless catFeature
            # found")` (`batch_binarized_ctr_calcer.cpp:150`)
            raise Error("Error: useless catFeature found")

        self.compute_current_bins()

        # `GatherWithMask(Bins, DecompressedTempBins, Indices, Mask)`
        for i in range(len(self.indices)):
            self.bins[i] = cat_bins[Int(index_of(self.indices[i]))]

        _stable_sort_by_bin(self.bins, self.indices)

        # `UpdateBordersMask(Bins, currentBins, Indices)`, the host twin of
        # `kernel/ctr_calcers.update_borders_mask_kernel`
        var n = len(self.indices)
        var new_indices = List[UInt32]()
        for i in range(n):
            var current_index = self.indices[i]
            var mask = is_segment_start(current_index)
            if not mask:
                mask = i == 0 or self.bins[i] != self.bins[i - 1]
            if not mask:
                var prev_index = self.indices[i - 1]
                mask = self.current_bins[Int(index_of(current_index))] != (
                    self.current_bins[Int(index_of(prev_index))]
                )
            new_indices.append(with_mask(current_index, mask))
        self.indices = new_indices^

    def segment_ids(self) -> List[UInt32]:
        """Their `ExtractMask(Indices, Tmp, false)` followed by
        `ScanVector(Tmp, Bins, false)`, the first two lines of BOTH freq
        calcers (`ctr_bins_builder.h:167-169`, `ctr_calcers.h:311-314`).

        Returns the segment index at each SORTED position.
        """
        var n = len(self.indices)
        var out = List[UInt32]()
        var running = UInt32(0)
        for i in range(n):
            out.append(running)
            var is_end: Bool
            if i + 1 < n:
                is_end = is_segment_start(self.indices[i + 1])
            else:
                is_end = True
            if is_end:
                running += UInt32(1)
        return out^

    def segment_offsets(self, segment_ids: List[UInt32]) -> List[UInt32]:
        """Their `UpdatePartitionOffsets(Bins, Tmp)`
        (`ctr_bins_builder.h:170`, kernel at
        `cuda_util/kernel/partitions.cu:81-107`).

        `offsets[b]` is the first sorted position holding segment `b`, and
        the array carries ONE FAKE TRAILING BIN so the last real segment
        has a next offset to subtract against (`ctr_calcers.h:317-320`
        spells that `+1` out).
        """
        var n = len(segment_ids)
        var n_segments = 0
        if n > 0:
            n_segments = Int(segment_ids[n - 1]) + 1
        var offsets = List[UInt32]()
        for _ in range(n_segments + 1):
            offsets.append(UInt32(n))
        var b = 0
        for i in range(n):
            while b <= Int(segment_ids[i]):
                offsets[b] = UInt32(i)
                b += 1
        while b <= n_segments:
            offsets[b] = UInt32(n)
            b += 1
        return offsets^

    def visit_equal_up_to_prior_freq_ctrs(
        self, ctr_configs: List[TCtrConfig]
    ) raises -> List[List[Float32]]:
        """Their `VisitEqualUpToPriorFreqCtrs` (`ctr_bins_builder.h:163-186`),
        the "pure freq, not weighted one, as a result it much faster" path.

        **THIS IS NOT THE DEFAULT DISPATCH.** `TCalcCtrHelper` reaches it
        only when `UseFullSetForCatFeatureStatCtrs()` is true, which is
        `counter_calc_method == Full` (`ctr_helper.h:96-104`,
        `binarizations_manager.h:290-292`); the shipped default is
        `SkipTest` (`cat_feature_options.cpp:233`) and lands on
        `TWeightedBinFreqCalcer` in `ctr_calcers.mojo` instead. On a fit
        with no test pool the two agree exactly -- see that file for why --
        and this arm exists so the option has both sides implemented rather
        than one side implemented and one side assumed.

        Returns one column per config, in config order, each indexed by
        ORIGINAL row.
        """
        var segment_ids = self.segment_ids()
        var offsets = self.segment_offsets(segment_ids)
        var size = len(self.indices)

        var out = List[List[Float32]]()
        for c in range(len(ctr_configs)):
            ref config = ctr_configs[c]
            if config.ctr_type != CTR_FEATURE_FREQ:
                raise Error(
                    "VisitEqualUpToPriorFreqCtrs takes FeatureFreq configs"
                    " only"
                )
            var prior = config.numerator_shift()
            var prior_observations = config.denumerator_shift()
            var column = List[Float32]()
            for _ in range(size):
                column.append(Float32(0.0))
            # `ComputeNonWeightedBinFreqCtr` (`kernel/ctr_calcers.cu:69`)
            for i in range(size):
                var bin = Int(segment_ids[i])
                var count = offsets[bin + 1] - offsets[bin]
                column[Int(index_of(self.indices[i]))] = (
                    Float32(count) + prior
                ) / (Float32(size) + prior_observations)
            out.append(column^)
        return out^


struct TCtrBinBuilderGpu(Movable):
    """`TCtrBinBuilder<TMapping>` ON THE DEVICE, which is the port.

    Every buffer below is theirs (`ctr_bins_builder.h:253-259`) and every
    launch is theirs. The `Gpu` suffix is ours and exists only because the
    host reference in this file already holds the unsuffixed name; see the
    file header.

    `TestSlice` is not carried, for the same reason the host builder does
    not carry it: `train()` has no test pool, and an always-empty slice is a
    field no branch reads.
    """

    var indices: DeviceBuffer[DType.uint32]
    """Their `Indices`: row ids in bin order, segment start in bit 31."""

    var bins: DeviceBuffer[DType.uint32]
    """Their `Bins`: the accumulated bin at each SORTED position."""

    var current_bins: DeviceBuffer[DType.uint32]
    """Their `CurrentBins`: per ORIGINAL row, the segment id under the
    ordering built so far."""

    var decompressed_temp_bins: DeviceBuffer[DType.uint32]
    """Their `DecompressedTempBins`. Holds the raw cat bins in ORIGINAL row
    order (their `Decompress` writes it; this port is handed dense codes and
    copies them in), and doubles as the radix sort's value ping-pong exactly
    as it does for them -- `:218` passes it as `ReorderBins`' second temp."""

    var tmp: DeviceBuffer[DType.uint32]
    """Their `Tmp`: the radix sort's key ping-pong, and the scan's output in
    `ComputeCurrentBins`."""

    var scan_block_sums: DeviceBuffer[DType.uint32]
    """`context.PartResults` for `ScanVector` (`cuda_util/scan.h:26-31`)."""

    var sort_offsets: DeviceBuffer[DType.int32]
    """`TRadixSortContext`'s scratch (`cuda_util/kernel/sort.cuh:20-33`),
    allocated by their `PrepareContext` and passed in here for the same
    reason: a kernel that allocates is a kernel that synchronizes."""

    var sort_block_sums: DeviceBuffer[DType.int32]

    var size: Int

    var learn_size: Int
    """Their `LearnSlice.Size()`, which is `firstZeroIndex` for
    `GatherTrivialWeights`."""

    def __init__(out self, ctx: DeviceContext, order: List[UInt32]) raises:
        """Their `SetIndices(learnIndices)` (`:32-52`) plus `Reset`
        (`:190-194`).

        `order` is `ctrEstimationOrder` as their builder receives it -- the
        identity for permutation-independent CTRs, a shuffle for the
        dependent ones. See the file header.
        """
        var n = len(order)
        if n == 0:
            raise Error("TCtrBinBuilderGpu: empty order")
        self.size = n
        self.learn_size = n

        var h = ctx.enqueue_create_host_buffer[DType.uint32](n)
        for i in range(n):
            if (order[i] & ~CTR_INDEX_MASK) != UInt32(0):
                raise Error(
                    "ctr estimation order entry "
                    + String(i)
                    + " has bits above the 0x3FFFFFFF index mask set; the"
                    " order carries row ids only, the flag bits are"
                    " UpdateBordersMask's to write"
                )
            h.unsafe_ptr().unsafe_store(i, order[i])

        self.indices = ctx.enqueue_create_buffer[DType.uint32](n)
        self.bins = ctx.enqueue_create_buffer[DType.uint32](n)
        self.current_bins = ctx.enqueue_create_buffer[DType.uint32](n)
        self.decompressed_temp_bins = ctx.enqueue_create_buffer[
            DType.uint32
        ](n)
        self.tmp = ctx.enqueue_create_buffer[DType.uint32](n)
        self.scan_block_sums = ctx.enqueue_create_buffer[DType.uint32](
            (n + SCAN_BLOCK - 1) // SCAN_BLOCK
        )
        # 512 is `REORDER_BLOCK`, the radix pass's block
        # (`reorder_one_bit.cu:35`): one aggregate per block of the pass.
        self.sort_offsets = ctx.enqueue_create_buffer[DType.int32](n)
        self.sort_block_sums = ctx.enqueue_create_buffer[DType.int32](
            (n + 512 - 1) // 512
        )

        ctx.enqueue_copy(dst_buf=self.indices, src_ptr=h.unsafe_ptr())
        ctx.enqueue_memset(self.bins, UInt32(0))
        ctx.enqueue_memset(self.current_bins, UInt32(0))
        ctx.synchronize()

    def compute_current_bins(mut self, ctx: DeviceContext) raises:
        """Their static `ComputeCurrentBins` (`:134-146`), launch for launch.

            ExtractMask(indices, dst, false)   // END flags
            ScanVector(dst, tmp, false)        // EXCLUSIVE
            ScatterWithMask(dst, tmp, indices, Mask)

        `dst` is `CurrentBins`; their `tmp` argument is `Bins`, passed by
        `ProceedNewBins` as `{ auto& tmp = Bins; ComputeCurrentBins(...); }`
        (`:207-210`) and legal because the very next launch overwrites
        `Bins` with the gather. That is why the scan's input and output are
        different buffers and why the scatter reads `Bins` while writing
        `CurrentBins`.
        """
        launch_extract_border_masks(
            ctx, self.indices, self.current_bins, self.size, False
        )
        launch_scan_vector_u32(
            ctx,
            self.size,
            False,
            self.current_bins,
            self.bins,
            self.scan_block_sums,
        )
        launch_scatter_with_mask_u32(
            ctx,
            self.current_bins,
            self.bins,
            self.indices,
            self.size,
            CTR_INDEX_MASK,
        )

    def add_cat_feature_bins(
        mut self,
        ctx: DeviceContext,
        cat_bins: List[UInt32],
        unique_values: Int,
    ) raises:
        """Their `AddCompressedBins` -> `ProceedNewBins` (`:104-110`,
        `:212-222`), with `Decompress` replaced by an upload because this
        port holds dense category codes rather than their packed `ui64`
        blocks.

            ComputeCurrentBins(Indices, Bins, CurrentBins)
            GatherWithMask(Bins, DecompressedTempBins, Indices, Mask)
            ReorderBins(Bins, Indices, 0, IntLog2(uniqueValues), Tmp,
                        DecompressedTempBins)
            UpdateBordersMask(Bins, CurrentBins, Indices)
        """
        if len(cat_bins) != self.size:
            raise Error(
                "cat bins size "
                + String(len(cat_bins))
                + " does not match the builder's "
                + String(self.size)
            )
        if unique_values <= 1:
            # their `CB_ENSURE(uniqueValues > 1, "Error: useless catFeature
            # found")` (`batch_binarized_ctr_calcer.cpp:141`)
            raise Error("Error: useless catFeature found")

        self.compute_current_bins(ctx)

        # their `AddLearnBins` -> `Decompress(compressedLearn, ...)`
        # (`:196-202`); dense codes need no decompression, so this is the
        # upload that puts the same bytes in the same buffer.
        var h = ctx.enqueue_create_host_buffer[DType.uint32](self.size)
        for i in range(self.size):
            if Int(cat_bins[i]) >= unique_values:
                raise Error(
                    "cat bin "
                    + String(cat_bins[i])
                    + " at row "
                    + String(i)
                    + " is outside 0.."
                    + String(unique_values - 1)
                )
            h.unsafe_ptr().unsafe_store(i, cat_bins[i])
        ctx.enqueue_copy(
            dst_buf=self.decompressed_temp_bins, src_ptr=h.unsafe_ptr()
        )
        ctx.synchronize()

        launch_gather_with_mask_u32(
            ctx,
            self.bins,
            self.decompressed_temp_bins,
            self.indices,
            self.size,
            CTR_INDEX_MASK,
        )

        # `ReorderBins(Bins, Indices, 0, newBits, Tmp,
        # DecompressedTempBins)`. Their `offset` is 0 and their `bits` is
        # `IntLog2(uniqueValues)`, so the sort touches exactly the bin bits
        # and never the index payload it carries.
        var new_bits = int_log2(unique_values)
        launch_radix_sort_bins(
            ctx,
            self.size,
            0,
            new_bits,
            self.bins,
            self.indices,
            self.tmp,
            self.decompressed_temp_bins,
            self.sort_offsets,
            self.sort_block_sums,
        )

        launch_update_borders_mask(
            ctx, self.bins, self.current_bins, self.indices, self.size
        )

    def read_indices(self, ctx: DeviceContext) raises -> List[UInt32]:
        """The whole `Indices` buffer, back on the host.

        NOT THEIRS -- their builder never leaves the device. It exists for
        two callers, and both are named here so this cannot quietly become a
        third: `mojo_only/ctr_device_check.mojo`, which compares the device
        ordering against the host one cell by cell, and the FeatureFreq
        calcer, which is the half of `PORTING.md` 52 that is still host
        side.
        """
        var h = ctx.enqueue_create_host_buffer[DType.uint32](self.size)
        ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=self.indices)
        ctx.synchronize()
        var out = List[UInt32]()
        for i in range(self.size):
            out.append(h.unsafe_ptr().unsafe_load(i))
        return out^
