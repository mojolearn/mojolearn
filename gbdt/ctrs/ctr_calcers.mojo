"""The two CTR calcers.

MIRRORS `catboost/cuda/ctrs/ctr_calcers.h` at CatBoost `54a8143a`:
`TWeightedBinFreqCalcer` (`:285-379`) and `THistoryBasedCtrCalcer`
(`:32-283`), which are the two halves of their GPU `simple_ctr` default.

## Which one runs, and when

`TCalcCtrHelper::VisitEqualUpToPriorCtrs` (`gpu_data/ctr_helper.h:71-129`)
dispatches on the CTR TYPE:

    IsCatFeatureStatisticCtr (FeatureFreq)
        counter_calc_method == Full     -> TCtrBinBuilder's pure-freq arm
        counter_calc_method == SkipTest -> TWeightedBinFreqCalcer   <-- default
    IsBinarizedTargetCtr (Borders, Buckets)
        -> THistoryBasedCtrCalcer
    else (FloatTargetMeanValue)
        -> THistoryBasedCtrCalcer::VisitFloatFeatureMeanCtrs

`SkipTest` is the shipped default (`cat_feature_options.cpp:233`), so the
frequency half of the GPU default lands on `TWeightedBinFreqCalcer` here.
`ctr_bins_builder.mojo` carries the other arm.

## WHICH HALF RUNS WHERE, AND WHY

`THistoryBasedCtrCalcerGpu` is the port of `THistoryBasedCtrCalcer`: it runs
`GatherTrivialWeights`, both segmented scans, `GetGatheredBinSample`,
`FillBinarizedTargetsStats` and `DivideWithPriors` on the device, through
the kernels their own code calls. It is what `train()` computes a `Borders`
column with.

`THistoryBasedCtrCalcer` above it -- no suffix -- is the HOST reference the
device answer is gated against in `mojo_only/ctr_device_check.mojo`, and it
is what `mojo_only/ctr_check.mojo` gates against an independent O(n^2)
tally. `TWeightedBinFreqCalcer` is still host side and that is what remains
of `PORTING.md` deviation 52.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.memory import bitcast

from gbdt.ctrs.ctr import (
    CTR_BORDERS,
    CTR_BUCKETS,
    CTR_FEATURE_FREQ,
    TCtrConfig,
    is_equal_up_to_prior_and_binarization,
)
from gbdt.ctrs.ctr_bins_builder import TCtrBinBuilder, TCtrBinBuilderGpu
from gbdt.ctrs.index_wrapper import (
    CTR_INDEX_MASK,
    index_of,
    is_segment_start,
)
from gbdt.ctrs.kernel.ctr_calcers import (
    launch_compute_weighted_bin_freq_ctr,
    launch_extract_border_masks,
    launch_fill_binarized_targets_stats,
    launch_gather_trivial_weights,
    launch_make_means_and_scatter,
)
from gbdt.gpu_util.kernel.partitions import (
    launch_update_partition_offsets,
)
from gbdt.gpu_util.kernel.scan import SCAN_BLOCK, launch_scan_vector_u32
from gbdt.gpu_util.kernel.segmented_reduce import (
    launch_segmented_reduce_sum,
)
from gbdt.gpu_util.kernel.segmented_scan import (
    SEG_SCAN_BLOCK,
    launch_segmented_scan_and_scatter_non_negative,
)
from gbdt.gpu_util.kernel.transform import (
    launch_gather_with_mask_f32,
    launch_gather_with_mask_u8,
)


def compute_simple_ctrs(
    cat_codes: List[UInt32],
    unique_values: Int,
    ctr_configs: List[TCtrConfig],
    binarized_target: List[UInt8],
    counter_calc_method_is_full: Bool,
) raises -> List[List[Float32]]:
    """One categorical feature in, one float column per config out.

    MIRRORS `TBatchedBinarizedCtrsCalcer::ComputeBinarizedCtrs`
    (`cuda/gpu_data/batch_binarized_ctr_calcer.cpp:30-116`), the part that
    is not dataset plumbing:

        binBuilder = BuildFeatureTensorBins(tensor, devId)
        ctrHelper  = TCalcCtrHelper(ctrTargets, binBuilder.GetIndices())
        ctrHelper->UseFullDataForCatFeatureStats(isFeatureFreqOnFullSet)
        for group in CreateGrouppedConfigs(...):
            ctrHelper->VisitEqualUpToPriorCtrs(group, ctrVisitor)

    Their grouping is `CreateEqualUpToPriorAndBinarizationCtrsGroupping`
    (`ctr.h:60-68`), which buckets configs by (Type, ParamId) so the
    expensive scan runs once per bucket and only the priors' divide
    repeats. That grouping is reproduced below rather than paraphrased,
    because it is what makes the three-prior fan-out cost one scan instead
    of three.

    ADDRESS NOTE: this driver's mirror address is under `gbdt/gpu_data/`,
    beside their `batch_binarized_ctr_calcer.cpp`. It lives here because
    this round owns `gbdt/ctrs/` and creating a second lane's file is a
    merge conflict, not a port. Move it when the lanes rejoin.
    """
    var n_rows = len(cat_codes)
    var builder = TCtrBinBuilder(n_rows)
    builder.add_cat_feature_bins(cat_codes, unique_values)

    var out = List[List[Float32]]()
    for _ in range(len(ctr_configs)):
        out.append(List[Float32]())

    # `CreateEqualUpToPriorAndBinarizationCtrsGroupping` (`ctr.h:60-68`),
    # keyed by (Type, ParamId) as `IsEqualUpToPriorAndBinarization` is.
    var done = List[Bool]()
    for _ in range(len(ctr_configs)):
        done.append(False)

    for i in range(len(ctr_configs)):
        if done[i]:
            continue
        var group = List[TCtrConfig]()
        var group_slots = List[Int]()
        for j in range(i, len(ctr_configs)):
            if not done[j] and is_equal_up_to_prior_and_binarization(
                ctr_configs[j], ctr_configs[i]
            ):
                done[j] = True
                group.append(ctr_configs[j])
                group_slots.append(j)

        var columns: List[List[Float32]]
        if group[0].ctr_type == CTR_FEATURE_FREQ:
            # `IsCatFeatureStatisticCtr` arm (`ctr_helper.h:95-111`)
            if counter_calc_method_is_full:
                columns = builder.visit_equal_up_to_prior_freq_ctrs(group)
            else:
                var calcer = TWeightedBinFreqCalcer.trivial(n_rows)
                columns = calcer.visit_equal_up_to_prior_freq_ctrs(
                    builder, group
                )
        elif group[0].ctr_type == CTR_BORDERS or (
            group[0].ctr_type == CTR_BUCKETS
        ):
            # `IsBinarizedTargetCtr` arm (`ctr_helper.h:112-119`)
            raise Error(
                "simple_ctr="
                + String("Borders" if group[0].ctr_type == CTR_BORDERS
                         else "Buckets")
                + " is PERMUTATION DEPENDENT (IsPermutationDependentCtrType,"
                " ctr_type.cpp:44-58) and cannot be written from this"
                " function, which is their permutation-INDEPENDENT"
                " writeCtrs call over the identity order"
                " (doc_parallel_dataset_builder.cpp:206 and :229). Running"
                " the ordered statistic in row order is a different and"
                " worse estimator, not a slower one. Use"
                " compute_simple_ctrs_gpu, which takes the CTR ESTIMATION"
                " PERMUTATION their :251-262 loop supplies"
            )
        else:
            raise Error(
                "ctr type " + String(group[0].ctr_type) + " has no calcer"
            )

        for k in range(len(group_slots)):
            out[group_slots[k]] = columns[k].copy()

    # `binarized_target` is consumed by the Borders arm above; it is taken
    # as an argument so the signature does not change the day that arm is
    # wired, and so a caller that forgot the target grid fails here rather
    # than silently producing frequency columns only.
    _ = binarized_target
    return out^


struct TWeightedBinFreqCalcer(Movable):
    """Their `TWeightedBinFreqCalcer<TMapping>` (`ctr_calcers.h:285-379`).

    `TotalWeight` is the sum of the LEARN weights only: `BuildCtrTarget`
    zeroes every test row's weight before summing
    (`dataset_helpers.cpp:29-39`), which is what makes this calcer the
    `SkipTest` one. With no test pool every weight is 1.0 and the total is
    the row count.
    """

    var weights: List[Float32]
    var total_weight: Float32

    def __init__(out self, var weights: List[Float32], total_weight: Float32):
        self.weights = weights^
        self.total_weight = total_weight

    @staticmethod
    def trivial(n_rows: Int) -> Self:
        """`BuildCtrTarget`'s trivial-weights case: 1.0 per learn row, and
        `TotalWeight` their sum. `TCtrTargets::IsTrivialWeights()` returns
        an unconditional `true` in their source (`ctr_helper.h:19-21`), so
        this is the only case their GPU learner actually builds."""
        var w = List[Float32]()
        for _ in range(n_rows):
            w.append(Float32(1.0))
        return Self(w^, Float32(n_rows))

    def visit_equal_up_to_prior_freq_ctrs(
        self, builder: TCtrBinBuilder, ctr_configs: List[TCtrConfig]
    ) raises -> List[List[Float32]]:
        """Their `VisitEqualUpToPriorFreqCtrs` (`ctr_calcers.h:307-341`),
        step for step:

            ExtractMask(indices, TempFlags, false)
            ScanVector(TempFlags, Bins, false)
            binCountWithFakeLastBin = ReadLast(Bins) + 2
            UpdatePartitionOffsets(Bins, SegmentStarts)
            GatherWithMask(Tmp, Weights, indices, Mask)
            SegmentedReduceVector(Tmp, SegmentStarts, BinWeights, Sum)
            for each config: ComputeWeightedBinFreqCtr(...)

        The final line is `weighted_bin_freq_ctrs_kernel`'s arithmetic:

            dst[Index(indices[i])] = (binSums[bins[i]] + prior)
                                     / (totalWeight + priorObservations)

        Their own comment on the loop -- "we don't calc several priors for
        featureFreq, btw this part could be optimized for several priors"
        (`:325`) -- is accurate: `GetDefaultPriors(FeatureFreq)` is a single
        `{0.0, 1}`, so the loop runs once under the defaults. It is written
        as a loop anyway because theirs is.
        """
        var size = builder.size()
        if len(self.weights) != size:
            raise Error(
                "weights size "
                + String(len(self.weights))
                + " does not match the bin builder's "
                + String(size)
            )

        var segment_ids = builder.segment_ids()
        var offsets = builder.segment_offsets(segment_ids)
        var n_segments = len(offsets) - 1

        # `GatherWithMask` + `SegmentedReduceVector(..., Sum)`
        var bin_weights = List[Float32]()
        for _ in range(n_segments):
            bin_weights.append(Float32(0.0))
        for i in range(size):
            bin_weights[Int(segment_ids[i])] += self.weights[
                Int(index_of(builder.indices[i]))
            ]

        var out = List[List[Float32]]()
        for c in range(len(ctr_configs)):
            ref config = ctr_configs[c]
            if config.ctr_type != CTR_FEATURE_FREQ:
                raise Error(
                    "TWeightedBinFreqCalcer takes FeatureFreq configs only;"
                    " got " + String(config.ctr_type)
                )
            var prior = config.numerator_shift()
            var prior_observations = config.denumerator_shift()
            var column = List[Float32]()
            for _ in range(size):
                column.append(Float32(0.0))
            for i in range(size):
                column[Int(index_of(builder.indices[i]))] = (
                    bin_weights[Int(segment_ids[i])] + prior
                ) / (self.total_weight + prior_observations)
            out.append(column^)
        return out^


def segmented_scan_and_scatter_non_negative_vector(
    src: List[Float32], indices: List[UInt32]
) raises -> List[Float32]:
    """THE SEAM. Their `SegmentedScanAndScatterNonNegativeVector`
    (`cuda_util/segmented_scan.h`), called three times by
    `THistoryBasedCtrCalcer` (`ctr_calcers.h:74`, `:92`, `:139`).

    An EXCLUSIVE scan that restarts at every segment boundary, where the
    boundary is the SIGN BIT of the value rather than a separate flag
    array, followed by a scatter through `Index(indices[i])` so the answer
    lands in original row order. "NonNegative" is the contract that makes
    the packing legal: the real values are >= 0, so a negative value can
    only mean "segment start", and `abs()` recovers it.

    **This host loop is the REFERENCE for the device scan, not a stand-in
    for a missing one.** The device version is
    `launch_segmented_scan_and_scatter_non_negative`
    (`gpu_util/kernel/segmented_scan.mojo`, deviation 49) and
    `THistoryBasedCtrCalcerGpu` below is what calls it;
    `mojo_only/ctr_device_check.mojo` compares the two cell by cell.

    The exclusive-and-restart semantics are what make the result an ORDERED
    statistic: position `i` receives the sum over the rows that PRECEDE it
    in the current segment, never its own value, so a row's CTR cannot see
    its own target.

    THEIR "EXCLUSIVE" IS AN INCLUSIVE SCAN WRITTEN ONE PLACE LATE.
    `TNonNegativeSegmentedScanOutputIterator::operator=`
    (`segmented_scan_helpers.cuh:62-76`) runs `cub::DeviceScan::
    InclusiveScan` and stores position `i`'s result at
    `Ptr[Index(indices[i + 1])]`, zero if `i + 1` starts a segment, with
    `FillBuffer(output, 0, size)` covering the first row that no store
    reaches (`scan.cu:57-59`). The loop below is that, unrolled.

    Two places carry the same flag and the two readers differ: the COMBINER
    (`TNonNegativeSegmentedSum`, `:11-22`) resets on `ExtractSignBit` of the
    VALUE, and the output iterator resets on bit 31 of the INDEX. They agree
    because every producer -- `GatherTrivialWeights`, `WriteMask`,
    `FillBinarizedTargetsStats` -- copies the index flag onto the value's
    sign. This reads the index, as the iterator does.
    """
    var n = len(src)
    if len(indices) != n:
        raise Error("segmented scan: src and indices differ in length")
    var out = List[Float32]()
    for _ in range(n):
        out.append(Float32(0.0))
    var running = Float32(0.0)
    for i in range(n):
        var v = src[i]
        # the sign bit IS the segment flag; `-0.0` counts, so the test is
        # on the index word, which carries the same flag
        if is_segment_start(indices[i]):
            running = Float32(0.0)
        out[Int(index_of(indices[i]))] = running
        running += v if v >= Float32(0.0) else -v
    return out^


struct THistoryBasedCtrCalcer(Movable):
    """Their `THistoryBasedCtrCalcer<TMapping>` (`ctr_calcers.h:32-283`),
    the ORDERED TARGET STATISTIC, which is the half of CatBoost's default
    that makes it a distinct algorithm.

    ## The value

    For a row `r` in category `c`, at target bin index `ParamId` and prior
    `(a, b)`:

        (sum over rows BEFORE r in the ctr permutation, same category, of
         [binarized target > ParamId]  + a)
        / (count of those rows + b)

    Three priors, three columns, one scan -- `VisitCatFeatureCtr` runs the
    statistic once and loops `DivideWithPriors` per config
    (`:141-150`).

    ## Construction, their `Reset(indices, firstZeroIndex, groupIds)`

        GatherTrivialWeights(GatheredWeightsWithMask, Indices,
                             firstZeroIndex, true)
        SegmentedScanAndScatterNonNegativeVector(GatheredWeightsWithMask,
                                                 Indices,
                                                 ScannedScatteredWeights,
                                                 false)

    so `ScannedScatteredWeights[r]` is the DENOMINATOR before the prior:
    how many rows precede `r` in its category. The `groupIds` fix is
    skipped -- `NeedFixForGroupwiseCtr()` is false for every configuration
    this port reaches, see `kernel/ctr_calcers.mojo`.

    ## THIS ONE IS THE HOST REFERENCE

    `THistoryBasedCtrCalcerGpu` at the bottom of this file is what `train()`
    runs. This class computes the same statistic in host loops and exists to
    be compared against it cell by cell
    (`mojo_only/ctr_device_check.mojo`), and to be compared itself against
    an independent O(n^2) tally (`mojo_only/ctr_check.mojo`). A host
    reference used to CHECK a device answer is not a CPU path
    (`PORTING_RULES.md` 0b-ii).

    `set_float_sample` / `VisitFloatFeatureMeanCtrs` (`:170-205`) is not
    ported: `FloatTargetMeanValue` is not in any default description.
    """

    var indices: List[UInt32]
    """Their `Indices`, sorted by bin, segment start in bit 31, taken in the
    CTR ESTIMATION PERMUTATION's order within each segment."""

    var gathered_weights_with_mask: List[Float32]
    """Their `GatheredWeightsWithMask`: 1.0 per learn row, negated at a
    segment start."""

    var scanned_scattered_weights: List[Float32]
    """Their `ScannedScatteredWeights`: per ORIGINAL row, the count of
    preceding same-category rows. The DENOMINATOR."""

    var binarized_sample: List[UInt8]
    """Their `BinarizedSample`, set by `SetBinarizedSample` from
    `CtrTargets.BinarizedTarget` (`ctr_helper.h:115-117`). Indexed by
    ORIGINAL row; `GetGatheredBinSample` gathers it into sorted order and
    caches that."""

    var first_zero_index: Int
    """Their `firstZeroIndex`, which is `CtrTargets.LearnSlice.Size()`."""

    def __init__(out self, builder: TCtrBinBuilder) raises:
        """Their trivial-weights constructor (`:52-63`) and the `Reset` it
        calls (`:87-100`)."""
        self.indices = builder.indices.copy()
        self.first_zero_index = builder.learn_size
        self.binarized_sample = List[UInt8]()
        self.gathered_weights_with_mask = List[Float32]()
        self.scanned_scattered_weights = List[Float32]()
        self.reset()

    def reset(mut self) raises:
        """`GatherTrivialWeights` then the segmented scan
        (`ctr_calcers.h:87-100`). The kernel twin of the first line is
        `kernel/ctr_calcers.gather_trivial_weights_kernel`."""
        var n = len(self.indices)
        self.gathered_weights_with_mask = List[Float32]()
        for i in range(n):
            var idx = self.indices[i]
            var val = Float32(1.0) if Int(index_of(idx)) < (
                self.first_zero_index
            ) else Float32(0.0)
            if is_segment_start(idx):
                self.gathered_weights_with_mask.append(-val)
            else:
                self.gathered_weights_with_mask.append(val)
        self.scanned_scattered_weights = (
            segmented_scan_and_scatter_non_negative_vector(
                self.gathered_weights_with_mask, self.indices
            )
        )

    def set_binarized_sample(mut self, var sample: List[UInt8]) raises:
        """`SetBinarizedSample` (`ctr_calcers.h:110-114`)."""
        if len(sample) != len(self.indices):
            raise Error(
                "binarized sample size "
                + String(len(sample))
                + " does not match the index size "
                + String(len(self.indices))
            )
        self.binarized_sample = sample^

    def has_binarized_target_sample(self) -> Bool:
        """`HasBinarizedTargetSample` (`ctr_calcers.h:102-104`)."""
        return len(self.binarized_sample) > 0

    def visit_cat_feature_ctr(
        self, ctr_configs: List[TCtrConfig]
    ) raises -> List[List[Float32]]:
        """`VisitCatFeatureCtr` (`ctr_calcers.h:121-153`), step for step.

            CB_ENSURE(referenceCtrConfig.Type == Borders || == Buckets)
            gatheredSample = GetGatheredBinSample()
            FillBinarizedTargetsStats(gatheredSample,
                                      GatheredWeightsWithMask, Dst,
                                      referenceCtrConfig.ParamId,
                                      referenceCtrConfig.Type)
            SegmentedScanAndScatterNonNegativeVector(Dst, Indices, Tmp)
            for each config:
                DivideWithPriors(Tmp, ScannedScatteredWeights,
                                 GetNumeratorShift, GetDenumeratorShift,
                                 Dst)

        ONE scan, N columns. The `IsEqualUpToPriorAndBinarization` assert on
        every config is theirs and is what makes that legal.
        """
        if len(ctr_configs) == 0:
            raise Error("VisitCatFeatureCtr called with no configs")
        if len(self.binarized_sample) != len(self.indices):
            raise Error(
                "VisitCatFeatureCtr needs a binarized target sample the"
                " size of the index"
            )
        ref reference = ctr_configs[0]
        if reference.ctr_type != CTR_BORDERS and (
            reference.ctr_type != CTR_BUCKETS
        ):
            raise Error(
                "VisitCatFeatureCtr takes Borders or Buckets configs only"
            )

        var n = len(self.indices)

        # `GetGatheredBinSample` (`:258-265`)
        var gathered_sample = List[UInt8]()
        for i in range(n):
            gathered_sample.append(
                self.binarized_sample[Int(index_of(self.indices[i]))]
            )

        # `FillBinarizedTargetsStats`, the host twin of
        # `kernel/ctr_calcers.fill_binarized_targets_stats_kernel`
        var bin_index = UInt32(reference.param_id)
        var is_borders = reference.ctr_type == CTR_BORDERS
        var dst = List[Float32]()
        for i in range(n):
            var weight = self.gathered_weights_with_mask[i]
            var target = UInt32(gathered_sample[i])
            var hit: Bool
            if is_borders:
                hit = target > bin_index
            else:
                hit = target == bin_index
            var mag = weight if weight >= Float32(0.0) else -weight
            var v = mag * (Float32(1.0) if hit else Float32(0.0))
            # `ExtractSignBit(weight)` (`kernel_helpers.cuh:20-23`) is a BIT
            # read, not `weight < 0`: a segment start whose weight is 0.0
            # arrives as -0.0, where `-0.0 < 0` is false and the flag would
            # be silently dropped.
            if (bitcast[DType.uint32](weight) >> 31) != UInt32(0):
                v = -v
            dst.append(v)

        var tmp = segmented_scan_and_scatter_non_negative_vector(
            dst, self.indices
        )

        var out = List[List[Float32]]()
        for c in range(len(ctr_configs)):
            ref config = ctr_configs[c]
            if not is_equal_up_to_prior_and_binarization(config, reference):
                raise Error(
                    "VisitCatFeatureCtr: every config in one call must be"
                    " equal up to prior and binarization"
                )
            var first_class_prior_count = config.numerator_shift()
            var total_prior_count = config.denumerator_shift()
            # `DivideWithPriors` == `MakeMeans`
            # (`kernel/ctr_calcers.make_means_kernel`)
            var column = List[Float32]()
            for r in range(n):
                column.append(
                    (tmp[r] + first_class_prior_count)
                    / (self.scanned_scattered_weights[r] + total_prior_count)
                )
            out.append(column^)
        return out^


struct THistoryBasedCtrCalcerGpu(Movable):
    """`THistoryBasedCtrCalcer<TMapping>` ON THE DEVICE, which is the port.

    Their class, their buffers (`ctr_calcers.h:265-283`), their launch
    order. The `Gpu` suffix is ours; CatBoost has one class because it has
    no host arm.

    Every step below is a kernel their own code calls, and none of them is
    new: `GatherTrivialWeights`, `FillBinarizedTargetsStats`,
    `MakeMeansAndScatter` and `GatherWithMask` were ported and gated by
    `mojo_only/ctr_kernels_check.mojo`, and
    `SegmentedScanAndScatterNonNegativeVector` by
    `pixi run check-segscan`. What this class adds is the wiring, which was
    the missing half.
    """

    var indices: DeviceBuffer[DType.uint32]
    """Their `Indices`, a const view of the bin builder's -- taken by
    `ConstCopyView()` in their constructor (`:42`). This port holds a copy
    of the bin builder instead of a view, so the buffer below is the
    builder's own and the builder must outlive the calcer."""

    var gathered_weights_with_mask: DeviceBuffer[DType.float32]
    var scanned_scattered_weights: DeviceBuffer[DType.float32]
    var dst: DeviceBuffer[DType.float32]
    var tmp: DeviceBuffer[DType.float32]
    var gathered_binarized_sample: DeviceBuffer[DType.uint8]
    var binarized_sample: DeviceBuffer[DType.uint8]

    var scan_scanned: DeviceBuffer[DType.float32]
    var scan_has_flag: DeviceBuffer[DType.uint8]
    var scan_block_sums: DeviceBuffer[DType.float32]
    var scan_block_flags: DeviceBuffer[DType.uint8]
    """`context.PartResults` for the segmented scan (`scan.h:26-31`), split
    across four buffers because our three-phase decoupling needs a value and
    a flag per element and per block where CUB keeps one opaque byte
    array."""

    var size: Int
    var first_zero_index: Int
    """Their `firstZeroIndex`, which is `CtrTargets.LearnSlice.Size()`."""

    var has_sample: Bool

    def __init__(
        out self, ctx: DeviceContext, builder: TCtrBinBuilderGpu
    ) raises:
        """Their trivial-weights constructor (`:52-63`) and the `Reset` it
        calls (`:85-99`)."""
        var n = builder.size
        self.size = n
        self.first_zero_index = builder.learn_size
        self.has_sample = False

        self.indices = builder.indices
        self.gathered_weights_with_mask = ctx.enqueue_create_buffer[
            DType.float32
        ](n)
        self.scanned_scattered_weights = ctx.enqueue_create_buffer[
            DType.float32
        ](n)
        self.dst = ctx.enqueue_create_buffer[DType.float32](n)
        self.tmp = ctx.enqueue_create_buffer[DType.float32](n)
        self.gathered_binarized_sample = ctx.enqueue_create_buffer[
            DType.uint8
        ](n)
        self.binarized_sample = ctx.enqueue_create_buffer[DType.uint8](n)

        var n_blocks = (n + SEG_SCAN_BLOCK - 1) // SEG_SCAN_BLOCK
        self.scan_scanned = ctx.enqueue_create_buffer[DType.float32](n)
        self.scan_has_flag = ctx.enqueue_create_buffer[DType.uint8](n)
        self.scan_block_sums = ctx.enqueue_create_buffer[DType.float32](
            n_blocks
        )
        self.scan_block_flags = ctx.enqueue_create_buffer[DType.uint8](
            n_blocks
        )

        self.reset(ctx)

    def reset(mut self, ctx: DeviceContext) raises:
        """`Reset(indices, firstZeroIndex, groupIds)` (`ctr_calcers.h:85-99`).

        The two launches that survive when `groupIds` is null:

            GatherTrivialWeights(GatheredWeightsWithMask, Indices,
                                 firstZeroIndex, true, Stream)
            SegmentedScanAndScatterNonNegativeVector(GatheredWeightsWithMask,
                                                     Indices,
                                                     ScannedScatteredWeights,
                                                     false, Stream)

        `ScannedScatteredWeights[r]` leaves holding the DENOMINATOR before
        the prior: how many rows precede `r` in its category, in the CTR
        estimation permutation's order. `NeedFixForGroupwiseCtr()` is false
        for every configuration this port reaches, so the `FixGroupwiseCtr`
        branch is unreachable rather than skipped -- see
        `kernel/ctr_calcers.mojo`.
        """
        launch_gather_trivial_weights(
            ctx,
            self.indices,
            self.size,
            UInt32(self.first_zero_index),
            True,
            self.gathered_weights_with_mask,
        )
        launch_segmented_scan_and_scatter_non_negative(
            ctx,
            self.size,
            False,
            self.gathered_weights_with_mask,
            self.indices,
            self.scanned_scattered_weights,
            self.scan_scanned,
            self.scan_has_flag,
            self.scan_block_sums,
            self.scan_block_flags,
        )
        self.has_sample = False

    def set_binarized_sample(
        mut self, ctx: DeviceContext, sample: List[UInt8]
    ) raises:
        """`SetBinarizedSample` (`ctr_calcers.h:108-112`).

        Theirs takes an already-device buffer; this uploads, because the
        target binarization runs on the host (`ctr_binarization.mojo`,
        their `BuildBinarizedTarget` is a host pass too --
        `gpu_data/dataset_helpers.cpp`).
        """
        if len(sample) != self.size:
            raise Error(
                "binarized sample size "
                + String(len(sample))
                + " does not match the index size "
                + String(self.size)
            )
        var h = ctx.enqueue_create_host_buffer[DType.uint8](self.size)
        for i in range(self.size):
            h.unsafe_ptr().unsafe_store(i, sample[i])
        ctx.enqueue_copy(
            dst_buf=self.binarized_sample, src_ptr=h.unsafe_ptr()
        )
        ctx.synchronize()
        self.has_sample = True

    def visit_cat_feature_ctr(
        mut self, ctx: DeviceContext, ctr_configs: List[TCtrConfig]
    ) raises -> List[List[Float32]]:
        """`VisitCatFeatureCtr` (`ctr_calcers.h:121-153`), launch for launch.

            gatheredSample = GetGatheredBinSample()
            FillBinarizedTargetsStats(gatheredSample,
                                      GatheredWeightsWithMask, Dst,
                                      referenceCtrConfig.ParamId,
                                      referenceCtrConfig.Type)
            SegmentedScanAndScatterNonNegativeVector(Dst, Indices, Tmp)
            for each config:
                DivideWithPriors(Tmp, ScannedScatteredWeights,
                                 GetNumeratorShift, GetDenumeratorShift,
                                 Dst)
                visitor(ctrConfig, Dst)

        ONE scan, N columns -- their `IsEqualUpToPriorAndBinarization`
        assert on every config is what makes that legal, and it is checked
        below as they check it.

        `DivideWithPriors`' five-argument overload is
        `TMakeMeanAndScatterKernel` with NO map (`ctr_kernels.h:553-561`),
        so the launch below passes `has_map=False` and the kernel's `m = i`
        branch runs. The four-argument in-place overload (`:534-539`) is a
        different function and is not the one this call site takes.
        """
        if len(ctr_configs) == 0:
            raise Error("VisitCatFeatureCtr called with no configs")
        if not self.has_sample:
            raise Error(
                "VisitCatFeatureCtr needs a binarized target sample the"
                " size of the index"
            )
        ref reference = ctr_configs[0]
        if reference.ctr_type != CTR_BORDERS and (
            reference.ctr_type != CTR_BUCKETS
        ):
            raise Error(
                "VisitCatFeatureCtr takes Borders or Buckets configs only"
            )

        # `GetGatheredBinSample` (`:258-265`)
        launch_gather_with_mask_u8(
            ctx,
            self.gathered_binarized_sample,
            self.binarized_sample,
            self.indices,
            self.size,
            CTR_INDEX_MASK,
        )

        launch_fill_binarized_targets_stats(
            ctx,
            self.gathered_binarized_sample,
            self.gathered_weights_with_mask,
            self.size,
            self.dst,
            UInt32(reference.param_id),
            reference.ctr_type == CTR_BORDERS,
        )

        launch_segmented_scan_and_scatter_non_negative(
            ctx,
            self.size,
            False,
            self.dst,
            self.indices,
            self.tmp,
            self.scan_scanned,
            self.scan_has_flag,
            self.scan_block_sums,
            self.scan_block_flags,
        )

        var out = List[List[Float32]]()
        var host = ctx.enqueue_create_host_buffer[DType.float32](self.size)
        for c in range(len(ctr_configs)):
            ref config = ctr_configs[c]
            if not is_equal_up_to_prior_and_binarization(config, reference):
                raise Error(
                    "VisitCatFeatureCtr: every config in one call must be"
                    " equal up to prior and binarization"
                )
            launch_make_means_and_scatter(
                ctx,
                self.tmp,
                self.scanned_scattered_weights,
                self.size,
                config.numerator_shift(),
                config.denumerator_shift(),
                self.indices,
                False,
                CTR_INDEX_MASK,
                self.dst,
            )
            # their `visitor(ctrConfig, constDst, Stream)`
            # (`ctr_calcers.h:149`), which for the batched calcer reads the
            # column back to the host to binarize it
            # (`batch_binarized_ctr_calcer.cpp:66-71`)
            ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=self.dst)
            ctx.synchronize()
            var column = List[Float32]()
            for r in range(self.size):
                column.append(host.unsafe_ptr().unsafe_load(r))
            out.append(column^)
        return out^


def compute_simple_ctrs_gpu(
    ctx: DeviceContext,
    cat_codes: List[UInt32],
    unique_values: Int,
    ctr_configs: List[TCtrConfig],
    binarized_target: List[UInt8],
    order: List[UInt32],
) raises -> List[List[Float32]]:
    """The PERMUTATION-DEPENDENT half of their two `writeCtrs` calls.

    MIRRORS the same `TBatchedBinarizedCtrsCalcer::ComputeBinarizedCtrs`
    driver `compute_simple_ctrs` above mirrors, with two differences that
    are theirs and not ours:

    * `order` is `ctrEstimationOrder` after
      `ctrsEstimationPermutation.WriteOrder(ctrEstimationOrder)`
      (`doc_parallel_dataset_builder.cpp:255`), where the host driver above
      gets it after `MakeSequence` (`:206`). Same function, two calls, two
      orders -- `:229` for the independent features and `:257` for the
      dependent ones.
    * every config here is a binarized-target one, so
      `VisitEqualUpToPriorCtrs` takes the `IsBinarizedTargetCtr` arm
      (`ctr_helper.h:112-119`) and nothing else.

    A FeatureFreq config handed to this function is refused rather than
    computed: it would be the right number from the wrong writer, and their
    `SplitByPermutationDependence` (`doc_parallel_dataset_builder.cpp:81`)
    cannot put one here.
    """
    var n = len(cat_codes)
    if len(order) != n:
        raise Error(
            "ctr estimation order has "
            + String(len(order))
            + " entries for "
            + String(n)
            + " rows"
        )
    if len(binarized_target) != n:
        raise Error(
            "binarized target has "
            + String(len(binarized_target))
            + " entries for "
            + String(n)
            + " rows; a Borders ctr is a statistic OF the target and"
            " cannot be computed without its grid"
        )
    for i in range(len(ctr_configs)):
        if ctr_configs[i].ctr_type == CTR_FEATURE_FREQ:
            raise Error(
                "compute_simple_ctrs_gpu is their permutation-DEPENDENT"
                " writeCtrs call; FeatureFreq is permutation independent"
                " (ctr_type.cpp:52-55) and belongs in compute_simple_ctrs,"
                " over the identity order"
            )

    var builder = TCtrBinBuilderGpu(ctx, order)
    builder.add_cat_feature_bins(ctx, cat_codes, unique_values)

    var out = List[List[Float32]]()
    for _ in range(len(ctr_configs)):
        out.append(List[Float32]())

    # `CreateEqualUpToPriorAndBinarizationCtrsGroupping` (`ctr.h:60-68`),
    # the same grouping the host driver above runs, so the expensive scan
    # runs once per bucket and only the priors' divide repeats.
    var done = List[Bool]()
    for _ in range(len(ctr_configs)):
        done.append(False)

    var calcer = THistoryBasedCtrCalcerGpu(ctx, builder)
    calcer.set_binarized_sample(ctx, binarized_target)

    for i in range(len(ctr_configs)):
        if done[i]:
            continue
        var group = List[TCtrConfig]()
        var group_slots = List[Int]()
        for j in range(i, len(ctr_configs)):
            if not done[j] and is_equal_up_to_prior_and_binarization(
                ctr_configs[j], ctr_configs[i]
            ):
                done[j] = True
                group.append(ctr_configs[j])
                group_slots.append(j)

        var columns = calcer.visit_cat_feature_ctr(ctx, group)
        for k in range(len(group_slots)):
            out[group_slots[k]] = columns[k].copy()

    return out^


struct TWeightedBinFreqCalcerGpu(Movable):
    """`TWeightedBinFreqCalcer<TMapping>` ON THE DEVICE, which is the port
    that retires the last line of `PORTING.md` deviation 52.

    Their class (`ctr_calcers.h:285-379`), their buffers, their launch
    order. The `Gpu` suffix is ours for the same reason
    `THistoryBasedCtrCalcerGpu`'s is: the host arm above is the gated
    REFERENCE, not a CPU path.

    Every launch in `visit_equal_up_to_prior_freq_ctrs` is one their own
    `VisitEqualUpToPriorFreqCtrs` makes (`:307-341`), and only two needed
    porting on the way: `UpdatePartitionOffsets`
    (`gpu_util/kernel/partitions.mojo`) and the Sum arm of
    `SegmentedReduceVector` (`gpu_util/kernel/segmented_reduce.mojo`,
    their cub call hand-written because MAX ships no segmented reduce --
    vendor check in that file's header).
    """

    var weights: DeviceBuffer[DType.float32]
    """Their `Weights` (`ConstCopyView` of the ctr target's); learn rows
    weigh 1.0 and any test tail 0.0, which is what makes this the
    `SkipTest` calcer."""

    var total_weight: Float32

    var temp_flags: DeviceBuffer[DType.uint32]
    """Their `TempFlags` (their TODO to shrink it to ui8 is theirs to
    keep; `ExtractMask` writes ui32 and both of us follow it)."""

    var bins: DeviceBuffer[DType.uint32]
    """Their `Bins`: segment id at each SORTED position."""

    var scan_block_sums: DeviceBuffer[DType.uint32]
    """`context.PartResults` for `ScanVector`, as the bin builder holds."""

    var tmp: DeviceBuffer[DType.float32]
    """Their `Tmp`: gathered weights, then each config's ctr column."""

    var size: Int

    def __init__(
        out self,
        ctx: DeviceContext,
        var weights: DeviceBuffer[DType.float32],
        total_weight: Float32,
        n_rows: Int,
    ) raises:
        self.weights = weights^
        self.total_weight = total_weight
        self.size = n_rows
        self.temp_flags = ctx.enqueue_create_buffer[DType.uint32](n_rows)
        self.bins = ctx.enqueue_create_buffer[DType.uint32](n_rows)
        self.scan_block_sums = ctx.enqueue_create_buffer[DType.uint32](
            (n_rows + SCAN_BLOCK - 1) // SCAN_BLOCK
        )
        self.tmp = ctx.enqueue_create_buffer[DType.float32](n_rows)

    @staticmethod
    def trivial(ctx: DeviceContext, n_rows: Int) raises -> Self:
        """The trivial-weights construction, mirroring the host
        `TWeightedBinFreqCalcer.trivial` above and their
        `IsTrivialWeights()` unconditional true (`ctr_helper.h:19-21`):
        1.0 per learn row, `TotalWeight` their sum."""
        var w = ctx.enqueue_create_buffer[DType.float32](n_rows)
        var h = ctx.enqueue_create_host_buffer[DType.float32](n_rows)
        for i in range(n_rows):
            h.unsafe_ptr().unsafe_store(i, Float32(1.0))
        ctx.enqueue_copy(dst_buf=w, src_ptr=h.unsafe_ptr())
        ctx.synchronize()
        return Self(ctx, w^, Float32(n_rows), n_rows)

    def visit_equal_up_to_prior_freq_ctrs(
        mut self,
        ctx: DeviceContext,
        mut builder: TCtrBinBuilderGpu,
        ctr_configs: List[TCtrConfig],
    ) raises -> List[List[Float32]]:
        """Their `VisitEqualUpToPriorFreqCtrs` (`ctr_calcers.h:307-341`),
        launch for launch:

            ExtractMask(indices, TempFlags, false)
            ScanVector(TempFlags, Bins, false)
            binCountWithFakeLastBin = ReadLast(Bins) + 2
            UpdatePartitionOffsets(Bins, SegmentStarts)
            GatherWithMask(Tmp, Weights, indices, Mask)
            SegmentedReduceVector(Tmp, SegmentStarts, BinWeights, Sum)
            for each config: ComputeWeightedBinFreqCtr(...)

        `SegmentStarts` and `BinWeights` are allocated per visit because
        their size is `ReadLast + 2`, known only here -- their `Reset`
        calls in the same place (`:317`, `:320`).

        Returns one HOST column per config, in config order, indexed by
        ORIGINAL row, exactly as the host reference does -- and on the
        trivial weights the two are bit-identical, because integer-valued
        float sums below 2^24 are exact in any reduction order.
        """
        if builder.size != self.size:
            raise Error(
                "bin builder holds "
                + String(builder.size)
                + " rows, calcer was built for "
                + String(self.size)
            )
        var n = self.size

        launch_extract_border_masks(
            ctx, builder.indices, self.temp_flags, n, False
        )
        launch_scan_vector_u32(
            ctx, n, False, self.temp_flags, self.bins, self.scan_block_sums
        )

        # ReadLast(Bins) + 2 (:315): one fake bin for the last segment end
        var h_bins = ctx.enqueue_create_host_buffer[DType.uint32](n)
        ctx.enqueue_copy(dst_ptr=h_bins.unsafe_ptr(), src_buf=self.bins)
        ctx.synchronize()
        var bin_count_with_fake = Int(
            h_bins.unsafe_ptr().unsafe_load(n - 1)
        ) + 2

        var segment_starts = ctx.enqueue_create_buffer[DType.uint32](
            bin_count_with_fake
        )
        launch_update_partition_offsets(
            ctx, segment_starts, bin_count_with_fake, self.bins, n
        )

        var bin_weights = ctx.enqueue_create_buffer[DType.float32](
            bin_count_with_fake - 1
        )
        launch_gather_with_mask_f32(
            ctx, self.tmp, self.weights, builder.indices, n, CTR_INDEX_MASK
        )
        launch_segmented_reduce_sum(
            ctx, self.tmp, segment_starts, bin_weights,
            bin_count_with_fake - 1,
        )

        var out = List[List[Float32]]()
        var dst = ctx.enqueue_create_buffer[DType.float32](n)
        var h_col = ctx.enqueue_create_host_buffer[DType.float32](n)
        for c in range(len(ctr_configs)):
            ref config = ctr_configs[c]
            if config.ctr_type != CTR_FEATURE_FREQ:
                raise Error(
                    "TWeightedBinFreqCalcerGpu takes FeatureFreq configs"
                    " only; got " + String(config.ctr_type)
                )
            launch_compute_weighted_bin_freq_ctr(
                ctx,
                builder.indices,
                True,
                self.bins,
                bin_weights,
                self.total_weight,
                config.numerator_shift(),
                config.denumerator_shift(),
                dst,
                n,
            )
            ctx.enqueue_copy(dst_ptr=h_col.unsafe_ptr(), src_buf=dst)
            ctx.synchronize()
            var column = List[Float32]()
            for i in range(n):
                column.append(h_col.unsafe_ptr().unsafe_load(i))
            out.append(column^)
        return out^


def compute_simple_ctrs_device(
    ctx: DeviceContext,
    cat_codes: List[UInt32],
    unique_values: Int,
    ctr_configs: List[TCtrConfig],
) raises -> List[List[Float32]]:
    """The permutation-INDEPENDENT half of their two `writeCtrs` calls, ON
    THE DEVICE -- the driver `compute_simple_ctrs` above runs on the host.

    Same grouping, same identity order (`MakeSequence(ctrEstimationOrder)`,
    `doc_parallel_dataset_builder.cpp:206`), same dispatch: a FeatureFreq
    group goes to `TWeightedBinFreqCalcerGpu` (the `SkipTest` default
    dispatch, `ctr_helper.h:96-111`); Borders/Buckets are refused with the
    same sentence the host driver refuses them with, because they are
    permutation DEPENDENT and belong to `compute_simple_ctrs_gpu`. The
    `counter_calc_method == Full` arm keeps its host implementation in
    `TCtrBinBuilder.visit_equal_up_to_prior_freq_ctrs`; it is opt-in, no
    benchmark reaches it, and with no test pool it is the same number.

    THE WIRING NOTE, for the lane that owns `gbdt/train.mojo`: `train()`'s
    independent-half call at its `compute_simple_ctrs(...)` site becomes
    `compute_simple_ctrs_device(ctx, ...)` -- a one-line change in a file
    this lane does not own, the same handoff shape as the ctr_quality
    harness rewire. Until then this driver is reached by
    `pixi run check-freq-ctr-device` only, and UNWIRED.md carries it.
    """
    var n = len(cat_codes)
    var order = List[UInt32]()
    for i in range(n):
        order.append(UInt32(i))
    var builder = TCtrBinBuilderGpu(ctx, order)
    builder.add_cat_feature_bins(ctx, cat_codes, unique_values)

    var out = List[List[Float32]]()
    for _ in range(len(ctr_configs)):
        out.append(List[Float32]())

    var done = List[Bool]()
    for _ in range(len(ctr_configs)):
        done.append(False)

    var calcer = TWeightedBinFreqCalcerGpu.trivial(ctx, n)

    for i in range(len(ctr_configs)):
        if done[i]:
            continue
        var group = List[TCtrConfig]()
        var group_slots = List[Int]()
        for j in range(i, len(ctr_configs)):
            if not done[j] and is_equal_up_to_prior_and_binarization(
                ctr_configs[j], ctr_configs[i]
            ):
                done[j] = True
                group.append(ctr_configs[j])
                group_slots.append(j)

        if group[0].ctr_type != CTR_FEATURE_FREQ:
            raise Error(
                "compute_simple_ctrs_device is the permutation-INDEPENDENT"
                " writer; ctr type "
                + String(group[0].ctr_type)
                + " is permutation dependent (or has no calcer) and belongs"
                " to compute_simple_ctrs_gpu"
            )
        var columns = calcer.visit_equal_up_to_prior_freq_ctrs(
            ctx, builder, group
        )
        for k in range(len(group_slots)):
            out[group_slots[k]] = columns[k].copy()

    return out^
