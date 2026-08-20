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

## HOST, AND THAT IS A DEVIATION

Both classes are device code in CatBoost. Here the freq calcer runs on the
host, and the history calcer's segmented scan does too, because the device
radix sort and segmented scan belong to another lane this round.
**Recorded as deviation 52 in `PORTING.md`.** The elementwise kernels these
two classes call ARE ported and enqueued -- `gbdt/ctrs/kernel/ctr_calcers.
mojo`, gated by `mojo_only/ctr_kernels_check.mojo` -- so what is missing is
the scan between them, not the arithmetic around it.
"""

from std.memory import bitcast

from gbdt.ctrs.ctr import (
    CTR_BORDERS,
    CTR_BUCKETS,
    CTR_FEATURE_FREQ,
    TCtrConfig,
    is_equal_up_to_prior_and_binarization,
)
from gbdt.ctrs.ctr_bins_builder import TCtrBinBuilder
from gbdt.ctrs.index_wrapper import index_of, is_segment_start


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
                + " is not wired into train(): THistoryBasedCtrCalcer is"
                " written and gated but needs two things this port does not"
                " have -- the device segmented scan and radix sort another"
                " lane is building, and the CTR ESTIMATION PERMUTATION"
                " (doc_parallel_dataset_builder.cpp:251-262,"
                " permutation_count default 4). Running the ordered"
                " statistic in row order would be a different estimator,"
                " not a slower one, so it raises instead"
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

    **This host loop is a stand-in for a DEVICE three-phase decoupled
    segmented scan that another lane is building** (`RECON_CTRS.md` step 3:
    the unsegmented version already exists in
    `gpu_util/kernel/reorder_one_bit.mojo` as
    `block_scan_flags_kernel` / `scan_block_sums_kernel` /
    `add_block_carry_kernel`, and the segmented one is that pattern with a
    carry that resets on a flag). Deviation 49.

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

    ## WHAT IS STILL MISSING, and it is not only the scan

    This class is NOT wired into `train()`, and the reason is a second
    seam that the sort/scan lane does not close. Their permutation-DEPENDENT
    CTRs are computed once PER PERMUTATION, over
    `ds.GetCtrsEstimationPermutation()`
    (`doc_parallel_dataset_builder.cpp:251-262`), with
    `permutation_count` defaulting to 4 (`boosting_options.cpp:14`); only
    the permutation-INDEPENDENT ones (FeatureFreq) use the identity order
    (`:204-206`). Feeding this calcer the identity order would not be a
    slower CatBoost, it would be a different and worse estimator -- on a
    dataset ordered by target, every row's statistic would read its own
    neighbourhood. So `train()` ships the FeatureFreq half and this class
    waits for the permutation machinery as well as for the device
    primitives.

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
