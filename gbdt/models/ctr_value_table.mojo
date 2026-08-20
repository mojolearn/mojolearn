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
which `mojo_only/ctr_apply_check.mojo` asserts per row rather than
assuming. FeatureFreq is permutation-independent
(`ctr_type.cpp:44-58`), which is what makes that identity available at
all; an ordered `Borders` statistic has no such property and this file
refuses to write one.

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
    CTR_FEATURE_FREQ,
    TCtrConfig,
    ctr_type_name,
)


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
    `static_cast<int>(totalSampleCount)` (`online_ctr.cpp:937-939`)."""
    var counts: List[Int]
    """Their blob, `GetTypedArrayRefForBlobData<int>()`, indexed by the
    dense category code instead of by a hash bucket."""

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
        self.counts = counts^

    @no_inline
    def calc(self, count_in_class: Float32, total_count: Float32) -> Float32:
        """`TModelCtr::Calc` (`online_ctr.h:289-292`), verbatim.

        **`@no_inline` is not decoration.** Deviation 54 in `PORTING.md`
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

    def empty_value(self) -> Float32:
        """Their `emptyVal = ctr->Calc(0, denominator)`
        (`static_ctr_provider.cpp:65`), the value a category the learn pool
        never saw takes. Not an error, not a neighbour, not a NaN."""
        return self.calc(Float32(0.0), Float32(self.counter_denominator))

    def value_for(self, code: Int) -> Float32:
        """One row's CTR value. Their loop is a hash lookup guarded by
        `NotFoundIndex` (`static_ctr_provider.cpp:66-70`); ours is a range
        test, because the key is the dense code."""
        if code < 0 or code >= len(self.counts):
            return self.empty_value()
        return self.calc(
            Float32(self.counts[code]), Float32(self.counter_denominator)
        )

    def categories(self) -> Int:
        return len(self.counts)


def build_feature_freq_tables(
    cat_codes: List[UInt32],
    unique_values: Int,
    ctr_configs: List[TCtrConfig],
    source_feature: Int,
    first_column: Int,
) raises -> List[TCtrValueTable]:
    """The apply-time tables for ONE categorical input feature, one per
    config, in the order `compute_simple_ctrs` emits its columns.

    This is their `CalcFinalCtrsImpl` for the `FeatureFreq` arm
    (`private/libs/algo/online_ctr.cpp:905-939`) reduced to what a dense
    code needs: `++ctrIntArray[elemId]` per row, then
    `CounterDenominator = totalSampleCount`. The learn-side column that
    the same counts reproduce is computed by
    `TWeightedBinFreqCalcer::visit_equal_up_to_prior_freq_ctrs`.
    """
    if unique_values <= 0:
        raise Error("a categorical feature with no categories has no table")
    var n_rows = len(cat_codes)
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

    var out = List[TCtrValueTable]()
    for i in range(len(ctr_configs)):
        ref cfg = ctr_configs[i]
        if cfg.ctr_type != CTR_FEATURE_FREQ:
            # NO TABLE FOR THIS CONFIG, and skipping is deliberate rather
            # than a shrug. This used to raise, on the reasoning that
            # `train()` could not produce a Borders column anyway; it can
            # now, so raising here would make every default categorical
            # fit die at the END of a successful training run.
            #
            # A skipped table is not a silent hole: `predict_floats`
            # refuses a model whose CTR columns have no table, so a Borders
            # model TRAINS and REFUSES TO SCORE, which is exactly what is
            # true of it. Writing a count table here instead would be the
            # dangerous option -- a wrong model rather than a missing one.
            #
            # What the real table needs is now known and is NOT ordered:
            # their `TCtrValueTable` for Borders is a per-category
            # histogram over TARGET CLASSES computed on the whole learn set
            # (`static_ctr_provider.cpp:255-283`), so no permutation, scan
            # or sort is involved. See RECON_CTRS.md.
            continue
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
                counts.copy(),
            )
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
                    table.value_for(Int(x_raw[src * n_rows + r]))
                )
    return out^
