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

## HOST, AND WHY THAT IS A DEVIATION RATHER THAN A DESIGN

**Theirs runs entirely on the device.** `ReorderBins` is their LSD radix
sort, `ScanVector` and `ScatterWithMask` are device kernels, and
`UpdateBordersMask` is `gbdt/ctrs/kernel/ctr_calcers.mojo`'s
`update_borders_mask_kernel`, which is already ported and enqueued by its
own check.

This file computes the same thing on the host, because the device radix
sort and segmented scan are being built by another lane and blocking on
them would have shipped nothing. **It is recorded as deviation 49 in
`PORTING.md`.** The device swap is a body replacement: the signatures, the
bit-31 packing, the segment definition and the call sites are all theirs,
so the sort lane replaces the loops without touching a caller.

`_stable_sort_by_bin` stands in for `ReorderBins`. It has to be STABLE:
their radix sort is, and the order of rows WITHIN a segment is the learn
permutation, which is what an ordered target statistic reads. An unstable
sort here would be invisible to a FeatureFreq check (counts do not care
about order) and would silently randomize every `Borders` value.
"""

from gbdt.ctrs.ctr import TCtrConfig, CTR_FEATURE_FREQ
from gbdt.ctrs.index_wrapper import index_of, is_segment_start, with_mask


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
        by `Reset` (`:190-194`): the identity permutation, no flags set,
        so the whole dataset is ONE segment until a feature is added."""
        self.indices = List[UInt32]()
        self.bins = List[UInt32]()
        self.current_bins = List[UInt32]()
        self.learn_size = n_rows
        for r in range(n_rows):
            self.indices.append(UInt32(r))
            self.bins.append(UInt32(0))
            self.current_bins.append(UInt32(0))

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
