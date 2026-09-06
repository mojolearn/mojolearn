# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The apply-time CTR table: what a trained categorical model needs to
score a row it has never seen.

MIRRORS `catboost/libs/model/ctr_value_table.h` (`TCtrValueTable`), the
`TModelCtr` half of `catboost/libs/model/online_ctr.h:260-292`, and the
lookup loop of `TStaticCtrProvider::CalcCtrs`
(`libs/model/static_ctr_provider.cpp:63-71`) at CatBoost `54a8143a`.

## Why an applied model cannot do without it

A CTR value is a STATISTIC OF THE LEARN POOL. `train()` replaces a
high-cardinality categorical column with one float column per CTR config,
computed from the learn rows; a new row carries a category, not a
frequency, so scoring it means looking the category up in the table the
learn pool produced. Their model file carries exactly that under
`ctr_data.hash_map` (`model_export/json_model_helpers.cpp:440-482`), and
until this file existed `predict_floats` REFUSED a categorical model
rather than score it against a grid its rows were never mapped onto.

## Their decomposition, and it is not the obvious one

The table does NOT store the CTR value. It stores the raw statistic per
category and the model carries the priors beside it; the value is formed
at apply time by `TModelCtr::Calc` (`online_ctr.h:289-292`):

    float ctr = (countInClass + PriorNum) / (totalCount + PriorDenom);
    return (ctr + Shift) * Scale;

For `FeatureFreq` and `Counter` their provider passes
`Calc(ctrTotal[bucket], CounterDenominator)` (`static_ctr_provider.cpp:63-71`),
and `CounterDenominator` is set to the LEARN ROW COUNT for FeatureFreq
(`private/libs/algo/online_ctr.cpp:937-939`; for Counter it is instead the
largest bin count, `:934-936`). A category the learn pool never saw is not
an error and does not fall back to a neighbour -- it gets
`emptyVal = Calc(0, denominator)`, their own line, which for the GPU
default prior `{0, 1}` is exactly zero frequency.

## `Borders` IS THE SAME TABLE WITH ONE MORE AXIS, AND IT IS NOT ORDERED

The single most surprising fact about the apply side, and it was read off
their source rather than reasoned about: **a `Borders` model carries NO
permutation, NO scan and NO history.** Its `TCtrValueTable` is a
per-category HISTOGRAM OVER TARGET CLASSES computed on the WHOLE learn set
(`CalcFinalCtrsImpl`, `online_ctr.cpp:909-930`), stored as a flat
`int[uniqueCategories * TargetClassesCount]`. The ordered statistic that
makes CatBoost a distinct algorithm is a TRAINING-TIME device and stops at
the model file.

Their apply reads it in two branches (`static_ctr_provider.cpp:91-122`):

    TargetClassesCount == 2 (their GPU default, one target border):
        hist = &blob[bucket * 2]
        Calc(hist[1], hist[0] + hist[1])

    TargetClassesCount > 2:
        total = sum(hist[0 .. TargetBorderIdx])
        good  = sum(hist[TargetBorderIdx + 1 .. end]);  total += good
        Calc(good, total)

and an unseen category takes `Calc(0, 0)` -- **the prior alone**, which is
NOT FeatureFreq's `Calc(0, denominator)`. See `empty_value` for why that
difference is worth a dispatch rather than a shared line.

The numerator is the count of rows whose binarized target EXCEEDS
`TargetBorderIdx`, which is exactly the predicate the training-side kernel
tests per row (`target > binIndex`,
`ctrs/kernel/ctr_calcers.mojo::fill_binarized_targets_stats_kernel`, their
`FillBinarizedTargetsStatsImpl<IS_BORDERS>`). Same statistic, one over a
prefix of the permutation and one over everything.

Storing counts rather than values is what makes `Shift` and `Scale` mean
anything, and it is why this file carries four floats it currently never
varies: their CPU learner fills them from `CalcNormalization`
(`private/libs/algo/split.cpp:73-81`) to fold the ui8 rebinarization into
the CTR, and their GPU learner leaves them at the struct defaults
`Shift = 0, Scale = 1`. Ours are the GPU learner's, and a file that did
not carry them would silently become wrong the day the other path lands.

## The bit-exactness this buys, and it was the point

`TWeightedBinFreqCalcer::visit_equal_up_to_prior_freq_ctrs`
(`gbdt/ctrs/ctr_calcers.mojo`, their `ctr_calcers.h:307-341`) computes the
LEARN column as

    (binSums[bin] + prior) / (totalWeight + priorObservations)

with every weight 1.0, so `binSums[bin]` is the category's row count as a
Float32 and `totalWeight` is `Float32(n_rows)`. `Calc(count, n_rows)` is
the same three operations on the same three Float32 values, so a learn row
scored through this table lands on the SAME BITS the fit trained on --
which `checks/ctr_apply_check.mojo` asserts per row rather than
assuming. FeatureFreq is permutation-independent
(`ctr_type.cpp:44-58`), which is what makes that identity available at all.

**A `Borders` model has no such identity and must not be gated as though
it did.** Its learn column is the ordered statistic over the estimation
permutation; its apply column is the full-learn-set histogram above. The
two are different numbers on the same learn rows, in CatBoost as much as
here, so the gate for Borders is agreement with an INDEPENDENTLY COMPUTED
full-set tally per row, never agreement with the fit's own loss.

## The category IS the dense code here

Their table is keyed by a 64-bit hash of the raw categorical value, walked
through `TDenseIndexHashView` with a `NotFoundIndex` sentinel. `train()`
takes DENSE CODES `0..k-1` (it raises on a negative one), so the hash step
has no counterpart and the dense code indexes `counts` directly; out of
range is their `NotFoundIndex`, taking the same `emptyVal` branch. That is
a smaller structure doing the identical job, and it is recorded as a
deviation rather than left to be discovered.
"""

from gbdt.ctrs.ctr import (
    CTR_BORDERS,
    CTR_COUNTER,
    CTR_FEATURE_FREQ,
    TCtrConfig,
    ctr_type_name,
)
from std.math import isfinite


def dense_category_code(value: Float32, feature: Int, row: Int) raises -> Int:
    """Validate the dense-code contract shared by fit and apply.

    Converting first and checking only the resulting integer silently aliases
    1.5 with category 1 (and makes NaN/Inf conversion backend-dependent).
    Categorical inputs at this surface are exact, non-negative integer codes.
    An integer not observed during fitting remains valid at apply time and
    takes the CTR table's unseen-category value.
    """
    if not isfinite(value):
        raise Error(
            "categorical feature " + String(feature) + " row "
            + String(row) + " is not finite"
        )
    # Codes are stored as UInt32 by the CTR kernels. Check the exclusive
    # upper bound before conversion so a finite oversized Float32 cannot
    # wrap to an unrelated category.
    if value < Float32(0.0) or value >= Float32(4294967296.0):
        raise Error(
            "categorical feature " + String(feature) + " row "
            + String(row) + " is outside the UInt32 code range"
        )
    var code = Int(value)
    if Float32(code) != value:
        raise Error(
            "categorical feature " + String(feature) + " row "
            + String(row) + " must be an exact non-negative integer code; got "
            + String(value)
        )
    return code


def ctr_type_from_name(name: String) raises -> Int:
    """The inverse of `ctr_type_name`, for the model file. An unknown name
    RAISES: a reader that guessed would apply the wrong statistic."""
    for t in range(6):
        if ctr_type_name(t) == name:
            return t
    raise Error("unknown ctr type name '" + name + "'")


struct TCtrValueTable(Copyable, Movable):
    """Their `TCtrValueTable` plus the `TModelCtr` fields that turn its
    counts into a value. One of these per CTR COLUMN of the model.

    `column` is the position in the model's column space (the space
    `fold_counts`, `one_hot` and `borders` are indexed by) and
    `source_feature` is the position in the RAW INPUT space the caller
    hands to `predict_floats`. The two differ exactly because a
    high-cardinality categorical input is REPLACED by one column per CTR
    config (`binarizations_manager.cpp:106-115`), so one input feature can
    stand behind several columns.
    """

    var column: Int
    var source_feature: Int
    var ctr_type: Int
    var prior_num: Float32
    """`TModelCtr::PriorNum`, their `GetNumeratorShift` = `Prior[0]`."""
    var prior_denom: Float32
    """`TModelCtr::PriorDenom`, their `GetDenumeratorShift` = `Prior[1]`."""
    var shift: Float32
    """`TModelCtr::Shift`. Zero on their GPU learner's path."""
    var scale: Float32
    """`TModelCtr::Scale`. One on their GPU learner's path."""
    var counter_denominator: Int
    """`TCtrValueTable::CounterDenominator`; for FeatureFreq their
    `static_cast<int>(totalSampleCount)` (`online_ctr.cpp:937-939`).

    THEIR DEFAULT IS 0 AND A `Borders` TABLE NEVER LEAVES IT
    (`ctr_value_table.h:104`, and `CalcFinalCtrsImpl` sets it only on the
    `Counter`/`FeatureFreq` arm, `online_ctr.cpp:934-939`). Ours is 0 there
    too, and nothing on the Borders path reads it."""
    var target_classes_count: Int
    """`TCtrValueTable::TargetClassesCount` (`ctr_value_table.h:105`), which
    is `TTargetClassifier::GetClassesCount()` = `Borders.size() + 1`
    (`libs/model/target_classifier.h:32-34`). ZERO on a `FeatureFreq` or
    `Counter` table, where their default stands and the blob has one int
    per category; on a `Borders` table it is the SECOND AXIS of the blob,
    which their builder allocates as `leafCount * targetClassesCount`
    (`online_ctr.cpp:909-910`). At the GPU default
    `ctr_target_border_count = 1` it is 2."""
    var target_border_idx: Int
    """`TModelCtr::TargetBorderIdx` (`online_ctr.h:262`), the target bin
    this column counts ABOVE. It is `TCtrConfig::ParamId`, and at one
    target border it is always 0. A model field, not a table field: three
    columns over one histogram may differ in nothing else."""
    var counts: List[Int]
    """Their blob, `GetTypedArrayRefForBlobData<int>()`, indexed by the
    dense category code instead of by a hash bucket.

    ONE int per category for `FeatureFreq`/`Counter`. For `Borders` it is
    their `int[uniqueCategories * TargetClassesCount]`, ROW-MAJOR BY
    CATEGORY -- category `c`'s histogram is
    `counts[c * target_classes_count ..][.. target_classes_count]`, which
    is the addressing of `ctrIntArray.data() + targetClassesCount * elemId`
    (`online_ctr.cpp:927-929`)."""

    def __init__(
        out self,
        column: Int,
        source_feature: Int,
        ctr_type: Int,
        prior_num: Float32,
        prior_denom: Float32,
        shift: Float32,
        scale: Float32,
        counter_denominator: Int,
        target_classes_count: Int,
        target_border_idx: Int,
        var counts: List[Int],
    ):
        self.column = column
        self.source_feature = source_feature
        self.ctr_type = ctr_type
        self.prior_num = prior_num
        self.prior_denom = prior_denom
        self.shift = shift
        self.scale = scale
        self.counter_denominator = counter_denominator
        self.target_classes_count = target_classes_count
        self.target_border_idx = target_border_idx
        self.counts = counts^

    @no_inline
    def calc(self, count_in_class: Float32, total_count: Float32) -> Float32:
        """`TModelCtr::Calc` (`online_ctr.h:289-292`), verbatim.

        **`@no_inline` is not decoration.** Deviation 54 in `archive/reference/PORTING.md`
        measured Mojo contracting a multiply-then-add across an inlined
        numeric helper where clang at `-ffp-contract=on` contracts only
        within one source expression, which re-decided a tie on a
        dynamic-programming plateau. Their `Calc` is a function boundary
        and clang keeps its rounding; keeping ours a function boundary is
        what makes the learn-row bit-identity above a property rather than
        a hope.
        """
        var ctr = (count_in_class + self.prior_num) / (
            total_count + self.prior_denom
        )
        return (ctr + self.shift) * self.scale

    def empty_value(self) raises -> Float32:
        """The value a category the learn pool never saw takes. Not an
        error, not a neighbour, not a NaN -- and **NOT ONE FORMULA**, which
        is the trap this dispatch exists for.

        Their provider computes `emptyVal` once per CTR type, inside the
        arm (`static_ctr_provider.cpp:63-122`):

            Counter / FeatureFreq :65   emptyVal = ctr->Calc(0, denominator)
            Buckets              :77   emptyVal = ctr->Calc(0, 0)
            Borders (the else)   :95   emptyVal = ctr->Calc(0, 0)

        A `Borders` table has NO denominator (their `CounterDenominator`
        stays at its 0 default there), so `Calc(0, 0)` is the PRIOR ALONE:
        `PriorNum / PriorDenom`. At the {0.5, 1} and {1, 1} priors of their
        default fan-out that is 0.5 and 1.0, where FeatureFreq's
        `Calc(0, n)` at the same priors is 0.5/(n+1) and 1/(n+1) -- three
        orders of magnitude apart on a 4096-row pool. Reaching for the
        FeatureFreq formula here would be a silent scoring bug on exactly
        the rows a held-out set is made of.
        """
        if self.ctr_type == CTR_FEATURE_FREQ or (
            self.ctr_type == CTR_COUNTER
        ):
            return self.calc(
                Float32(0.0), Float32(self.counter_denominator)
            )
        if self.ctr_type == CTR_BORDERS:
            return self.calc(Float32(0.0), Float32(0.0))
        raise Error(
            "no apply-time table arithmetic is ported for ctr type "
            + ctr_type_name(self.ctr_type)
            + "; their provider's other arms read a TCtrMeanHistory blob"
            " (static_ctr_provider.cpp:53-62), which this format does not"
            " carry"
        )

    def categories(self) -> Int:
        """How many CATEGORIES the blob holds, which is not its length once
        the target-class axis exists."""
        if self.target_classes_count > 0:
            return len(self.counts) // self.target_classes_count
        return len(self.counts)

    def value_for(self, code: Int) raises -> Float32:
        """One row's CTR value, their `TStaticCtrProvider::CalcCtrs` inner
        loop (`static_ctr_provider.cpp:63-122`) transcribed arm for arm.

        Their bucket lookup is a hash walk guarded by `NotFoundIndex`; ours
        is a range test, because the key is the dense code (deviation 56).
        Everything after the lookup is theirs.
        """
        if self.ctr_type == CTR_FEATURE_FREQ or (
            self.ctr_type == CTR_COUNTER
        ):
            # `:63-71`
            if code < 0 or code >= len(self.counts):
                return self.empty_value()
            return self.calc(
                Float32(self.counts[code]),
                Float32(self.counter_denominator),
            )

        if self.ctr_type == CTR_BORDERS:
            # their `else` arm, `:91-122`
            var classes = self.target_classes_count
            if classes < 2:
                raise Error(
                    "a Borders table needs at least two target classes and"
                    " has " + String(classes)
                    + "; their TargetClassesCount is Borders.size() + 1"
                    " (libs/model/target_classifier.h:32-34)"
                )
            if self.target_border_idx < 0 or (
                self.target_border_idx >= classes
            ):
                raise Error(
                    "TargetBorderIdx " + String(self.target_border_idx)
                    + " is outside the " + String(classes)
                    + " target classes this table carries"
                )
            if code < 0 or code >= self.categories():
                return self.empty_value()
            var base = code * classes
            if classes > 2:
                # `:97-113`
                var total = 0
                var good = 0
                for class_id in range(self.target_border_idx + 1):
                    total += self.counts[base + class_id]
                for class_id in range(self.target_border_idx + 1, classes):
                    good += self.counts[base + class_id]
                total += good
                return self.calc(Float32(good), Float32(total))
            # `:115-121`, which is their GPU default: TargetClassesCount 2
            return self.calc(
                Float32(self.counts[base + 1]),
                Float32(self.counts[base] + self.counts[base + 1]),
            )

        raise Error(
            "no apply-time table arithmetic is ported for ctr type "
            + ctr_type_name(self.ctr_type)
            + "; TCatFeatureParams.check() admits only Borders and"
            " FeatureFreq (catboost_options.mojo), so a table of any other"
            " type is a corrupt file rather than a missing feature"
        )


def build_ctr_tables(
    cat_codes: List[UInt32],
    unique_values: Int,
    ctr_configs: List[TCtrConfig],
    binarized_target: List[UInt8],
    target_classes_count: Int,
    source_feature: Int,
    first_column: Int,
) raises -> List[TCtrValueTable]:
    """The apply-time tables for ONE categorical input feature, one per
    config, in the order `compute_simple_ctrs` emits its columns.

    This is their `CalcFinalCtrsImpl`
    (`private/libs/algo/online_ctr.cpp:875-939`) reduced to what a dense
    code needs -- their bucket id `elemId` is our code -- with both arms of
    its allocation and both arms of its accumulation loop:

        FeatureFreq   blob int[leafCount]                       :906
                      ++ctrIntArray[elemId]                     :922
                      CounterDenominator = totalSampleCount     :938

        Borders       TargetClassesCount = targetClassesCount   :909
                      blob int[leafCount * targetClassesCount]  :910
                      ++elem[targetClass[z]]                    :927-930

    **THE Borders TABLE IS NOT ORDERED.** It is a histogram over target
    classes on the WHOLE learn set: no permutation, no scan, no sort, no
    history. The CTR ESTIMATION PERMUTATION that
    `compute_simple_ctrs_gpu` runs is a TRAINING-TIME device and never
    reaches the model file. That claim was read off their source rather
    than inferred, and the consequence is measurable: the learn column a
    `Borders` fit trains on is the ORDERED statistic, the column an applied
    model reproduces is this UNORDERED one, and the two are different
    numbers on the same rows. `FeatureFreq` has no such gap, which is why
    its apply reproduces the fit bit for bit and Borders' cannot.

    **THREE PRIORS ARE THREE `TModelCtr` OVER ONE TABLE.** Their `ctr_data`
    is a map keyed by `TModelCtrBase`, and the three priors of their
    default Borders fan-out are three `TModelCtr` sharing one
    `TCtrValueTable`. The histogram below is therefore computed ONCE per
    categorical feature and handed to every Borders config; only the priors
    differ, and only at `Calc`. What each table then STORES is a copy,
    because this format keys a table by the model COLUMN it feeds -- see
    `archive/reference/PORTING.md` 58.
    """
    if unique_values <= 0:
        raise Error("a categorical feature with no categories has no table")
    var n_rows = len(cat_codes)

    # `++ctrIntArray[elemId]` (`online_ctr.cpp:922`), the FeatureFreq arm
    var counts = List[Int]()
    for _ in range(unique_values):
        counts.append(0)
    for r in range(n_rows):
        var c = Int(cat_codes[r])
        if c < 0 or c >= unique_values:
            raise Error(
                "category code " + String(c) + " is outside 0.."
                + String(unique_values - 1)
            )
        counts[c] += 1

    # `++elem[targetClass[z]]` (`online_ctr.cpp:927-930`), the Borders arm.
    # Built once and shared by every Borders config, and only when one is
    # present: a frequency-only fit has no binarized target at all.
    var wants_histogram = False
    for i in range(len(ctr_configs)):
        if ctr_configs[i].ctr_type == CTR_BORDERS:
            wants_histogram = True
    var histogram = List[Int]()
    if wants_histogram:
        if target_classes_count < 2:
            raise Error(
                "a Borders CTR table needs at least two target classes and"
                " was given " + String(target_classes_count)
                + "; their TargetClassesCount is the target classifier's"
                " Borders.size() + 1 (libs/model/target_classifier.h:32-34)"
            )
        if len(binarized_target) != n_rows:
            raise Error(
                "a Borders CTR table is a histogram OF the binarized"
                " target and got " + String(len(binarized_target))
                + " target classes for " + String(n_rows) + " rows"
            )
        for _ in range(unique_values * target_classes_count):
            histogram.append(0)
        for r in range(n_rows):
            var cls = Int(binarized_target[r])
            if cls < 0 or cls >= target_classes_count:
                raise Error(
                    "binarized target class " + String(cls) + " at row "
                    + String(r) + " is outside 0.."
                    + String(target_classes_count - 1)
                )
            histogram[Int(cat_codes[r]) * target_classes_count + cls] += 1

    var out = List[TCtrValueTable]()
    for i in range(len(ctr_configs)):
        ref cfg = ctr_configs[i]
        if cfg.ctr_type == CTR_FEATURE_FREQ:
            out.append(
                TCtrValueTable(
                    first_column + i,
                    source_feature,
                    cfg.ctr_type,
                    cfg.numerator_shift(),
                    cfg.denumerator_shift(),
                    # their GPU learner never touches Shift/Scale; the CPU
                    # learner's `CalcNormalization` is what fills them
                    Float32(0.0),
                    Float32(1.0),
                    n_rows,
                    0,
                    0,
                    counts.copy(),
                )
            )
        elif cfg.ctr_type == CTR_BORDERS:
            out.append(
                TCtrValueTable(
                    first_column + i,
                    source_feature,
                    cfg.ctr_type,
                    cfg.numerator_shift(),
                    cfg.denumerator_shift(),
                    Float32(0.0),
                    Float32(1.0),
                    # their CalcFinalCtrsImpl leaves CounterDenominator at
                    # its 0 default on this arm, and nothing reads it
                    0,
                    target_classes_count,
                    cfg.param_id,
                    histogram.copy(),
                )
            )
        else:
            raise Error(
                "no apply-time CTR table is ported for ctr type "
                + ctr_type_name(cfg.ctr_type)
                + "; TCatFeatureParams.check() admits only Borders and"
                " FeatureFreq, so this config never reached train()"
            )
    return out^


@fieldwise_init
struct CtrColumnPlan(Copyable, Movable):
    """How the model's COLUMN space maps back onto the RAW INPUT space.

    Reconstructed from the tables rather than stored a second time: the
    columns of one input feature are contiguous and in input order, which
    is the order `train()` appends them in, so walking the two spaces in
    lockstep recovers the map. `source_feature` on every table is then
    CHECKED against the walk instead of trusted -- a table naming the wrong
    input would otherwise read the wrong column of the caller's matrix and
    still produce numbers.
    """

    var source_of_column: List[Int]
    var table_of_column: List[Int]
    var n_input_features: Int


def column_plan(
    ctr_tables: List[TCtrValueTable], n_columns: Int
) raises -> CtrColumnPlan:
    var table_of_column = List[Int]()
    for _ in range(n_columns):
        table_of_column.append(-1)
    for i in range(len(ctr_tables)):
        var c = ctr_tables[i].column
        if c < 0 or c >= n_columns:
            raise Error(
                "a CTR table names column " + String(c) + " of "
                + String(n_columns)
            )
        if table_of_column[c] != -1:
            raise Error("two CTR tables name column " + String(c))
        table_of_column[c] = i

    var source_of_column = List[Int]()
    for _ in range(n_columns):
        source_of_column.append(-1)
    var c = 0
    var f = 0
    while c < n_columns:
        var ti = table_of_column[c]
        if ti < 0:
            source_of_column[c] = f
            c += 1
            f += 1
            continue
        var src = ctr_tables[ti].source_feature
        if src != f:
            raise Error(
                "CTR table for column " + String(c) + " names input feature "
                + String(src) + ", but the columns before it account for "
                + String(f) + " inputs"
            )
        while c < n_columns and table_of_column[c] >= 0 and (
            ctr_tables[table_of_column[c]].source_feature == src
        ):
            source_of_column[c] = f
            c += 1
        f += 1
    return CtrColumnPlan(source_of_column^, table_of_column^, f)


def expand_raw_columns(
    ctr_tables: List[TCtrValueTable],
    n_columns: Int,
    x_raw: List[Float32],
    n_rows: Int,
) raises -> List[Float32]:
    """RAW input columns in, MODEL columns out, colmajor both sides.

    Their `TStaticCtrProvider::CalcCtrs` writes one result column per
    needed CTR and the evaluator then quantizes it like any float feature;
    this is the same step for a dense-coded pool. A column with no table is
    copied through unchanged -- a numeric feature, or a one-hot categorical
    one whose codes ARE its bins.
    """
    var plan = column_plan(ctr_tables, n_columns)
    if len(x_raw) != n_rows * plan.n_input_features:
        raise Error(
            "expected " + String(plan.n_input_features)
            + " raw input columns x " + String(n_rows) + " rows = "
            + String(n_rows * plan.n_input_features) + " values, got "
            + String(len(x_raw))
        )
    var out = List[Float32]()
    for c in range(n_columns):
        var src = plan.source_of_column[c]
        var ti = plan.table_of_column[c]
        if ti < 0:
            for r in range(n_rows):
                out.append(x_raw[src * n_rows + r])
        else:
            ref table = ctr_tables[ti]
            for r in range(n_rows):
                out.append(
                    table.value_for(
                        dense_category_code(
                            x_raw[src * n_rows + r], src, r
                        )
                    )
                )
    return out^
