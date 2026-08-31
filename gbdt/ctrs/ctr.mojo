# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# This lane MIRRORS CatBoost (pinned commit 54a8143a). Per-file provenance is in gbdt/DERIVATION_MAP.tsv, in this file's own docstring, and in NOTICE; files with no CatBoost counterpart say so.
"""CTR types, configurations and priors.

MIRRORS `catboost/cuda/ctrs/ctr.h` plus the enum and the two predicate
tables it leans on, `catboost/private/libs/ctr_description/ctr_type.{h,cpp}`
and `catboost/private/libs/ctr_description/ctr_config.h`, at CatBoost
`54a8143a`.

A CTR turns a categorical column into a NUMERIC column the histogram
kernels can already split on. Their GPU `simple_ctr` default is TWO
calcers, and the difference between them is the whole design:

* `FeatureFreq` -- `count / (n + 1)`. Never looks at the target, so it is
  permutation INDEPENDENT and can be computed once, before training.
* `Borders` -- an ORDERED TARGET STATISTIC. For each row, the mean of the
  binarized target over the rows that PRECEDE it in the learn permutation
  and share its category. Permutation DEPENDENT, and the thing that makes
  CatBoost a distinct algorithm rather than a GBDT with frequency
  encoding.

## THE PRIOR FAN-OUT, which is why a `Borders` cat feature is THREE columns

`GetDefaultPriors` (`cat_feature_options.cpp:117-129`) returns

    Borders / Buckets / BinarizedTargetMeanValue -> {{0,1}, {0.5,1}, {1,1}}
    FeatureFreq / Counter                        -> {{0.0,1}}
    FloatTargetMeanValue                         -> {{0,1}}

and `CreateCtrConfigsFromDescription`
(`cuda/data/binarizations_manager.cpp:394-433`) emits ONE `TCtrConfig` per
(prior x target bin). So under the GPU defaults one categorical feature
becomes

    3 Borders columns  (three priors x one target bin)
  + 1 FeatureFreq column
  = 4 numeric columns

and a port that emits one column per cat feature is not slightly off, it is
missing two thirds of the information their learner splits on. Confirmed
against CatBoost's own resolved options: a CPU fit's `cat_feature_params`
prints `priors [[0,1],[0.5,1],[1,1]]` for the Borders description and
`[[0,1]]` for its counter.

`ParamId` is the target bin index. `numBins` is `TargetBorders.size()` for
Borders and `TargetBorders.size() + 1` for Buckets
(`binarizations_manager.cpp:415-419`), so at the GPU default of ONE target
border, Borders emits ParamId 0 only and the fan-out is the priors alone.

## The two shifts

`GetNumeratorShift` is `Prior[0]` and `GetDenumeratorShift` is `Prior[1]`
(`ctr.h:75-81`); every calcer in this directory divides
`(statistic + numerator) / (weight + denumerator)`.
"""


# --- ECtrType (`ctr_description/ctr_type.h:3-11`) ------------------------
#
# THEIR ORDER AND THEIR VALUES, so a number read out of one of their configs
# means the same thing here.

comptime CTR_BORDERS = 0
comptime CTR_BUCKETS = 1
comptime CTR_BINARIZED_TARGET_MEAN_VALUE = 2
comptime CTR_FLOAT_TARGET_MEAN_VALUE = 3
comptime CTR_COUNTER = 4
comptime CTR_FEATURE_FREQ = 5
comptime CTR_TYPES_COUNT = 6


def ctr_type_name(t: Int) -> String:
    if t == CTR_BORDERS:
        return String("Borders")
    if t == CTR_BUCKETS:
        return String("Buckets")
    if t == CTR_BINARIZED_TARGET_MEAN_VALUE:
        return String("BinarizedTargetMeanValue")
    if t == CTR_FLOAT_TARGET_MEAN_VALUE:
        return String("FloatTargetMeanValue")
    if t == CTR_COUNTER:
        return String("Counter")
    if t == CTR_FEATURE_FREQ:
        return String("FeatureFreq")
    return String("CtrTypesCount")


def need_target(t: Int) raises -> Bool:
    """`NeedTarget` (`ctr_type.cpp:8-24`)."""
    if (
        t == CTR_BUCKETS
        or t == CTR_BORDERS
        or t == CTR_BINARIZED_TARGET_MEAN_VALUE
        or t == CTR_FLOAT_TARGET_MEAN_VALUE
    ):
        return True
    if t == CTR_FEATURE_FREQ or t == CTR_COUNTER:
        return False
    raise Error("Unknown ctr type " + ctr_type_name(t))


def need_target_classifier(t: Int) raises -> Bool:
    """`NeedTargetClassifier` (`ctr_type.cpp:26-42`). This is the predicate
    that decides whether TARGET BINARIZATION has to run at all."""
    if t == CTR_FEATURE_FREQ or t == CTR_COUNTER or (
        t == CTR_FLOAT_TARGET_MEAN_VALUE
    ):
        return False
    if (
        t == CTR_BUCKETS
        or t == CTR_BORDERS
        or t == CTR_BINARIZED_TARGET_MEAN_VALUE
    ):
        return True
    raise Error("Unknown ctr type " + ctr_type_name(t))


def is_permutation_dependent_ctr_type(t: Int) raises -> Bool:
    """`IsPermutationDependentCtrType` (`ctr_type.cpp:44-58`).

    `Counter` and `FeatureFreq` are the only independent ones, which is
    exactly why FeatureFreq can be a one-shot host pass and Borders cannot.
    """
    if (
        t == CTR_BUCKETS
        or t == CTR_BORDERS
        or t == CTR_FLOAT_TARGET_MEAN_VALUE
        or t == CTR_BINARIZED_TARGET_MEAN_VALUE
    ):
        return True
    if t == CTR_COUNTER or t == CTR_FEATURE_FREQ:
        return False
    raise Error("Unknown ctr type " + ctr_type_name(t))


def is_binarized_target_ctr(t: Int) -> Bool:
    """`IsBinarizedTargetCtr` (`ctr.h:14-16`) -- the `THistoryBasedCtrCalcer`
    arm of `TCalcCtrHelper::VisitEqualUpToPriorCtrs`."""
    return t == CTR_BUCKETS or t == CTR_BORDERS


def is_float_target_ctr(t: Int) -> Bool:
    """`IsFloatTargetCtr` (`ctr.h:18-20`)."""
    return t == CTR_FLOAT_TARGET_MEAN_VALUE


def is_cat_feature_statistic_ctr(t: Int) -> Bool:
    """`IsCatFeatureStatisticCtr` (`ctr.h:22-24`) -- the bin-freq arm."""
    return t == CTR_FEATURE_FREQ


def is_borders_based_ctr(t: Int) -> Bool:
    """`IsBordersBasedCtr` (`ctr.h:26-28`)."""
    return t == CTR_BORDERS


def is_supported_ctr_type_gpu(t: Int) -> Bool:
    """`IsSupportedCtrType(ETaskType::GPU, ...)` (`restrictions.h:33-43`).

    Note what is NOT here: `Counter`. And note what is missing from the CPU
    arm of the same function (`:20-32`): `FeatureFreq`. The two task types
    do not implement the same set, which is the concrete reason a local
    CatBoost CPU run cannot be an oracle for our FeatureFreq values.
    """
    return (
        t == CTR_BORDERS
        or t == CTR_BUCKETS
        or t == CTR_FLOAT_TARGET_MEAN_VALUE
        or t == CTR_FEATURE_FREQ
    )


# --- TPrior (`cat_feature_options.h`) ------------------------------------


@fieldwise_init
struct TPrior(Copyable, ImplicitlyCopyable, Movable):
    """Their `TPrior`, which is a `std::pair<float, float>` of
    (numerator shift, denumerator shift)."""

    var numerator: Float32
    var denumerator: Float32


def get_default_priors(ctr_type: Int) raises -> List[TPrior]:
    """`NCatboostOptions::GetDefaultPriors`
    (`cat_feature_options.cpp:117-129`), branch for branch.

    THE THREE-PRIOR FAN-OUT LIVES HERE. It is one `switch` in their source
    and it is the reason a `Borders` cat feature produces three numeric
    columns.
    """
    var out = List[TPrior]()
    if (
        ctr_type == CTR_BORDERS
        or ctr_type == CTR_BUCKETS
        or ctr_type == CTR_BINARIZED_TARGET_MEAN_VALUE
    ):
        out.append(TPrior(Float32(0.0), Float32(1.0)))
        out.append(TPrior(Float32(0.5), Float32(1.0)))
        out.append(TPrior(Float32(1.0), Float32(1.0)))
        return out^
    if ctr_type == CTR_FEATURE_FREQ or ctr_type == CTR_COUNTER:
        out.append(TPrior(Float32(0.0), Float32(1.0)))
        return out^
    if ctr_type == CTR_FLOAT_TARGET_MEAN_VALUE:
        out.append(TPrior(Float32(0.0), Float32(1.0)))
        return out^
    raise Error("Unknown ctr type " + ctr_type_name(ctr_type))


# --- TCtrConfig (`ctr_description/ctr_config.h:18-41`) -------------------


@fieldwise_init
struct TCtrConfig(Copyable, ImplicitlyCopyable, Movable):
    """Their `NCB::TCtrConfig`. One of these is one OUTPUT COLUMN.

    `param_id` is the target bin index the statistic counts against; for
    `Borders` at one target border it is always 0, and `numCtrs` in
    `CreateCtrConfigsFromDescription` (`binarizations_manager.cpp:415`) is
    what would make it range.

    `ctr_binarization_config_id` indexes the manager's list of distinct
    binarization descriptions (`GetOrCreateCtrBinarizationId`,
    `:435-446`); it is carried so the config compares and hashes as theirs
    does, and so the border rule can be looked up per column.
    """

    var ctr_type: Int
    var prior: TPrior
    var param_id: Int
    var ctr_binarization_config_id: Int

    def numerator_shift(self) -> Float32:
        """`GetNumeratorShift` (`ctr.h:75-77`), which is `Prior.at(0)`."""
        return self.prior.numerator

    def denumerator_shift(self) -> Float32:
        """`GetDenumeratorShift` (`ctr.h:79-81`), which is `Prior.at(1)`."""
        return self.prior.denumerator


def is_equal_up_to_prior_and_binarization(
    left: TCtrConfig, right: TCtrConfig
) -> Bool:
    """`IsEqualUpToPriorAndBinarization` (`ctr.h:70-72`).

    This is the grouping key the calcers assert on: every config in one
    `Visit...` call shares a statistic, so the expensive scan runs ONCE and
    only the final divide is repeated per prior. Getting this wrong costs
    three scans instead of one and changes no answer, which is exactly the
    kind of deviation that never shows up in a check.
    """
    return left.param_id == right.param_id and left.ctr_type == right.ctr_type


def create_ctr_config_for_feature_freq(
    prior: Float32, unique_values: Int
) -> TCtrConfig:
    """`CreateCtrConfigForFeatureFreq` (`ctr.h:83-90`).

    NOT the simple-ctr default. Their default FeatureFreq prior comes from
    `GetDefaultPriors` and is `{0.0, 1}`; this constructor's
    `{prior, 0.5 * uniqueValues}` is used where a config is minted from a
    cardinality rather than from an option. Kept because it is theirs and
    because the two denominators are easy to confuse.
    """
    return TCtrConfig(
        CTR_FEATURE_FREQ,
        TPrior(prior, Float32(0.5) * Float32(unique_values)),
        0,
        0,
    )
