# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The GradientBoosting fit with CTR categorical columns on the host, a
SECOND spelling of `gbdt/train.mojo::train`'s categorical arm on the
gbdt-categorical-ctr-tables lane (lane/trees-cpu, 2026-09-28).

HOST ONLY. Nothing here imports `max.gpu`, `std.gpu`, a `DeviceContext` or a
module that defines a kernel. The library imports are the GPU-free host
modules the device fit ITSELF runs on the host, unchanged: the CTR config
records and priors (`gbdt/ctrs/ctr.mojo`), the target grid and the CTR
column grids (`gbdt/ctrs/ctr_binarization.mojo`: `build_target_borders`,
`build_binarized_target`, `compute_ctr_borders`, all called on the calling
thread by `train` too), the permutations (`gbdt/data/permutation.mojo`),
`dense_category_code` and the apply-time tables `build_ctr_tables`
(`gbdt/models/ctr_value_table.mojo`, host code on the device path), and the
symmetric oracle (`gbdt/host/gbdt_oracle.mojo`) for the float grid, the
binarization and the boosting loop. The two CTR CALCERS are RESTATED below,
because `gbdt/ctrs/ctr_calcers.mojo` imports `max.gpu.host`.

THE CONFIGURATION THIS COVERS, by name (tools/identity_break.py
`gbdt-categorical-ctr-tables`: 20 depth-6 Logloss trees, `cat_features`
naming two columns above `one_hot_max_size` and one at it, the legacy
defaults `random_strength=0`, `bootstrap_type='No'`). The CTR options are
the GPU defaults `train` itself uses (`TCatFeatureParams.default()`,
`gbdt/options/catboost_options.mojo:1002-1033`; `gbdt_fit` never passes
others): simple CTRs Borders at the three default priors (ParamId 0, the
Uniform-15 grid) and FeatureFreq at prior (0, 1) (the MinEntropy-15 grid),
the target grid MinEntropy with one border, `one_hot_max_size` 2 and
`counter_calc_method` SkipTest. Refused by the binding by name: an eval set
with CTR columns, and every option the symmetric Logloss oracle refuses.

WHAT IS MIRRORED, IN THE ORDER THE FIT REACHES IT

  1. `train`'s CTR prelude (`gbdt/train.mojo:1336-1491`): the configs of
     `simple_ctr_configs` in slot order, split by permutation dependence
     (Borders dependent, FeatureFreq not); `has_permutation_features` and
     the Plain collapse to ONE permutation without them (`:1381-1434`);
     `permutation_count` (4 unset) and the estimation permutation (the last
     unless named); the target borders and the binarized target; one CTR
     order per permutation, `ctrs_estimation_permutation(n, p).fill_order()`.
  2. The column loop (`:1532-1676`): a raw column per non-categorical
     feature (one-hot when flagged), a dense-coded categorical column at or
     below `one_hot_max_size` becomes one one-hot column, one above it
     becomes one column per config in slot order: the FeatureFreq column
     from `TWeightedBinFreqCalcer` (`ctr_calcers.mojo:183-276`) over the
     identity order, the Borders columns from `THistoryBasedCtrCalcerGpu`
     (`ctr_calcers.mojo:544-798`) over EACH permutation's order, the model
     column taking the estimation permutation's values; then
     `build_ctr_tables` for the model.
  3. `_quantize_training_columns` (`:2345-2700`): the float columns through
     the symmetric oracle's `gbdt_host_grid` (the same per-column border
     search, `_calc_quantization_phase_b`, and the same shared subsample),
     one-hot columns `code + 0.5` below the largest code, a
     permutation-dependent CTR column `compute_ctr_borders` over
     PERMUTATION 0's values (`:2519-2532`), an independent one over its own,
     every non-float column `AsIs`.
  4. One compressed index per permutation (`:1745-1778`), the dependent
     columns from that permutation's values, the rest shared.
  5. `fit_with_test` over the permutations: `gbdt_host_boost`.
  6. `model_text` (`gbdt/models/model_text.mojo:374-670`): the
     `ctr_columns` header, `type ctr` / `type cat` feature records, the
     `ctr_table` and `ctr_entry` records in column order, the trees
     (`split_type take_bin` on a one-hot split) and the losses.

THE TWO CALCERS, AS ARITHMETIC. Every weight is 1.0 (`TCtrTargets::
IsTrivialWeights()` is unconditionally true, `ctr_helper.h:19-21`), so each
Float32 running sum of the device's segmented scans holds an INTEGER count,
exact below 2^24 rows; the one rounding is the final Float32 divide:

    FeatureFreq   (count(c) + prior_num) / (n + prior_denom)
    Borders       (before_hits(r) + prior_num) / (before(r) + prior_denom)

where `before(r)` counts the rows of `r`'s category that precede it in the
permutation's order (their exclusive scan over the stable sort by
category) and `before_hits(r)` those among them whose binarized target
exceeds ParamId. Integer counts, so the thread layout of the device scan
does not reach the bits and is not simulated; the Float32 conversion and
the add of the prior (0, 0.5 or 1, exact) precede the divide as theirs do.

THE NEGATIVE CONTROL. `-D MOJOLEARN_HOST_SABOTAGE=1` is the symmetric
oracle's (the Newton walker's regularizer), which this fit reaches through
`gbdt_host_boost`. The lane's end-to-end sabotage perturbs the ordered
statistic here (gbdt/checks/sabotage/ctr_ordered_cpu_only.patch).

The restatement is a prediction until measured. The four-column diff of
tools/identity_break.py on the gbdt-categorical-ctr-tables lane is the
measurement.
"""
from gbdt.ctrs.ctr import (
    CTR_BORDERS,
    CTR_FEATURE_FREQ,
    TCtrConfig,
    TPrior,
    ctr_type_name,
    get_default_priors,
)
from gbdt.ctrs.ctr_binarization import (
    BORDER_SELECTION_MIN_ENTROPY,
    BORDER_SELECTION_UNIFORM,
    TBinarizationOptions,
    build_binarized_target,
    build_target_borders,
    compute_ctr_borders,
    IDN_CTR_BORDERS_DEVICE,
    ctr_border_type_code,
)
from gbdt.options.data_processing_options import NAN_MODE_FORBIDDEN
from gbdt.data.permutation import (
    DEFAULT_PERMUTATION_COUNT,
    ctrs_estimation_permutation,
)
from gbdt.data.quantization import NAN_TREATMENT_AS_IS
from gbdt.gpu_data.compressed_index_builder import build_layout
from gbdt.host.gbdt_oracle import (
    GbdtHostFitWithEval,
    GbdtHostGrid,
    GbdtHostModel,
    GbdtHostParams,
    _binarize_columns,
    _nan_token,
    gbdt_f32_token,
    gbdt_f64_token,
    gbdt_host_boost,
    gbdt_host_grid,
)
from gbdt.host.gbdt_oracle_eval import GbdtHostEval
from gbdt.models.ctr_value_table import (
    TCtrValueTable,
    build_ctr_tables,
    dense_category_code,
)

#: `one_hot_max_size`, their GPU default (`catboost_options.mojo:1031`).
comptime GBDT_CTR_ONE_HOT_MAX_SIZE = 2
#: `TCatFeatureParams.default()`'s target grid border count
#: (`TBinarizationOptions(BORDER_SELECTION_MIN_ENTROPY, 1)`).
comptime GBDT_CTR_TARGET_BORDERS = 1
#: The CTR column grids' border count (`with_priors` and
#: `create_default_counter`, `catboost_options.mojo:877-913`).
comptime GBDT_CTR_GRID_BORDERS = 15

#: a column's kind in `gbdt_ctr_column_kinds`
comptime GBDT_COL_FLOAT = 0
comptime GBDT_COL_ONE_HOT = 1
comptime GBDT_COL_CTR = 2


def gbdt_ctr_column_kinds(
    flags: List[UInt32], x_colmajor: List[Float32], n_rows: Int, n_features: Int
) raises -> List[Int]:
    """`train`'s categorical dispatch (`gbdt/train.mojo:1532-1618`), per
    input feature: bit 0 of the flag categorical, bit 1 one-hot. A
    non-categorical feature is FLOAT (one-hot when flagged: the raw column
    under `column_one_hot`); a categorical one must be densely coded with
    at least two categories, and is ONE_HOT at or below `one_hot_max_size`,
    CTR above it."""
    var kinds = List[Int](length=n_features, fill=GBDT_COL_FLOAT)
    for f in range(n_features):
        var is_cat = (flags[f] & UInt32(1)) != UInt32(0)
        var flagged_one_hot = (flags[f] & UInt32(2)) != UInt32(0)
        if is_cat and flagged_one_hot:
            raise Error(
                "feature " + String(f) + " is in both cat_features and one_hot;"
                " cat_features makes the one-hot decision itself, from"
                " one_hot_max_size"
            )
        if not is_cat:
            kinds[f] = GBDT_COL_ONE_HOT if flagged_one_hot else GBDT_COL_FLOAT
            continue
        var unique_values = _dense_cardinality(x_colmajor, n_rows, f)
        kinds[f] = (
            GBDT_COL_ONE_HOT if unique_values <= GBDT_CTR_ONE_HOT_MAX_SIZE
            else GBDT_COL_CTR
        )
    return kinds^


def gbdt_has_ctr_columns(kinds: List[Int]) -> Bool:
    for f in range(len(kinds)):
        if kinds[f] == GBDT_COL_CTR:
            return True
    return False


def _dense_cardinality(
    x_colmajor: List[Float32], n_rows: Int, f: Int
) raises -> Int:
    """The dense-code validation of `train` (`:1567-1596`): cardinality is
    max + 1, every code below it present, at least two."""
    var maxc = 0
    var seen = List[Bool](length=1, fill=False)
    for r in range(n_rows):
        var c = dense_category_code(x_colmajor[f * n_rows + r], f, r)
        if c > maxc:
            maxc = c
            seen.resize(maxc + 1, False)
        seen[c] = True
    var unique_values = maxc + 1
    if unique_values <= 1:
        raise Error(
            "Error: useless catFeature found (feature " + String(f)
            + " has one category)"
        )
    for c in range(unique_values):
        if not seen[c]:
            raise Error(
                "cat_features column " + String(f) + " is not densely coded:"
                " category " + String(c) + " is absent from 0.." + String(maxc)
            )
    return unique_values


def gbdt_ctr_simple_configs() raises -> List[TCtrConfig]:
    """`TCatFeatureParams.default().simple_ctr_configs()`
    (`catboost_options.mojo:1070-1127`): description 0 Borders, every
    default prior (outer) by every target bin (inner, `numBins` =
    the option's border count); description 1 FeatureFreq at its one
    default prior, ParamId 0."""
    var out = List[TCtrConfig]()
    var bp = get_default_priors(CTR_BORDERS)
    for p in range(len(bp)):
        for i in range(GBDT_CTR_TARGET_BORDERS):
            out.append(TCtrConfig(CTR_BORDERS, bp[p], i, 0))
    var fp = get_default_priors(CTR_FEATURE_FREQ)
    for p in range(len(fp)):
        out.append(TCtrConfig(CTR_FEATURE_FREQ, fp[p], 0, 1))
    return out^


def _ctr_grid_for(config: TCtrConfig) -> TBinarizationOptions:
    """`ctr_binarization_for` (`catboost_options.mojo:1129-1146`) on the
    default descriptions: Borders Uniform 15, FeatureFreq MinEntropy 15."""
    if config.ctr_binarization_config_id == 0:
        return TBinarizationOptions(BORDER_SELECTION_UNIFORM, GBDT_CTR_GRID_BORDERS)
    return TBinarizationOptions(BORDER_SELECTION_MIN_ENTROPY, GBDT_CTR_GRID_BORDERS)


def _feature_freq_column(
    codes: List[UInt32], unique_values: Int, config: TCtrConfig
) -> List[Float32]:
    """`TWeightedBinFreqCalcer.trivial(n).visit_equal_up_to_prior_freq_ctrs`
    (`ctr_calcers.mojo:196-276`): the per-category Float32 sum of unit
    weights, then `(sum + prior) / (totalWeight + priorObservations)` with
    `totalWeight = Float32(n)`."""
    var n = len(codes)
    var bin_weights = List[Float32](length=unique_values, fill=Float32(0.0))
    for r in range(n):
        bin_weights[Int(codes[r])] += Float32(1.0)
    var total = Float32(n)
    var prior = config.numerator_shift()
    var prior_obs = config.denumerator_shift()
    var out = List[Float32](length=n, fill=Float32(0.0))
    for r in range(n):
        out[r] = (bin_weights[Int(codes[r])] + prior) / (total + prior_obs)
    return out^


def _ordered_target_columns(
    codes: List[UInt32],
    unique_values: Int,
    configs: List[TCtrConfig],
    binarized_target: List[UInt8],
    order: List[UInt32],
) raises -> List[List[Float32]]:
    """`compute_simple_ctrs_gpu` (`ctr_calcers.mojo:801-891`) for Borders
    configs: `TCtrBinBuilderGpu(order)` sorts the order STABLY by category,
    `GatherTrivialWeights` + the segmented scan give each row the count of
    its category's rows BEFORE it in that order, `FillBinarizedTargetsStats`
    + the second scan the count of those whose binarized target exceeds
    ParamId, and `MakeMeansAndScatter` divides with each config's prior
    (`(tmp + num) / (weights + denom)`)."""
    var n = len(codes)
    if len(order) != n:
        raise Error(
            "ctr estimation order has " + String(len(order)) + " entries for "
            + String(n) + " rows"
        )
    if len(binarized_target) != n:
        raise Error(
            "binarized target has " + String(len(binarized_target))
            + " entries for " + String(n) + " rows"
        )
    var out = List[List[Float32]]()
    for c in range(len(configs)):
        if configs[c].ctr_type != CTR_BORDERS:
            raise Error(
                "the ordered CTR restatement takes Borders configs only; got "
                + ctr_type_name(configs[c].ctr_type)
            )
        var before = List[Float32](length=unique_values, fill=Float32(0.0))
        var hits = List[Float32](length=unique_values, fill=Float32(0.0))
        var num = configs[c].numerator_shift()
        var denom = configs[c].denumerator_shift()
        var param = UInt32(configs[c].param_id)
        var column = List[Float32](length=n, fill=Float32(0.0))
        for i in range(n):
            var row = Int(order[i])
            var k = Int(codes[row])
            column[row] = (hits[k] + num) / (before[k] + denom)
            before[k] += Float32(1.0)
            if UInt32(binarized_target[row]) > param:
                hits[k] += Float32(1.0)
        out.append(column^)
    return out^


struct GbdtCtrHostFit(Movable):
    """The fit and what its model text needs beside the ensemble."""

    var fit: GbdtHostFitWithEval
    var one_hot: List[Bool]
    var tables: List[TCtrValueTable]
    var ctr_column_count: Int

    def __init__(
        out self,
        var fit: GbdtHostFitWithEval,
        var one_hot: List[Bool],
        var tables: List[TCtrValueTable],
        ctr_column_count: Int,
    ):
        self.fit = fit^
        self.one_hot = one_hot^
        self.tables = tables^
        self.ctr_column_count = ctr_column_count


def gbdt_ctr_host_fit(
    x_colmajor: List[Float32],
    y: List[Float32],
    n_rows: Int,
    n_features: Int,
    params: GbdtHostParams,
    flags: List[UInt32],
    permutation_count: Int,
    ctr_estimation_permutation_id: Int,
) raises -> GbdtCtrHostFit:
    """`train` with categorical columns on the covered configuration (see
    the module docstring for what that is and what mirrors what)."""
    if n_rows < 1 or n_features < 1:
        raise Error("train requires at least one row and one feature")
    if len(x_colmajor) != n_rows * n_features:
        raise Error("x_colmajor size mismatch")
    if len(y) != n_rows:
        raise Error("y size mismatch")
    if len(flags) != n_features:
        raise Error("cat flags must be one per feature")
    var kinds = gbdt_ctr_column_kinds(flags, x_colmajor, n_rows, n_features)

    # ---- the CTR prelude (`gbdt/train.mojo:1336-1491`) ----
    var configs = gbdt_ctr_simple_configs()
    var independent_slots = List[Int]()
    var dependent_slots = List[Int]()
    var independent_configs = List[TCtrConfig]()
    var dependent_configs = List[TCtrConfig]()
    for c in range(len(configs)):
        if configs[c].ctr_type == CTR_BORDERS:
            dependent_slots.append(c)
            dependent_configs.append(configs[c])
        else:
            independent_slots.append(c)
            independent_configs.append(configs[c])
    var has_permutation_features = (
        len(dependent_configs) > 0 and gbdt_has_ctr_columns(kinds)
    )
    var perm_count = permutation_count
    if perm_count == -1:
        perm_count = DEFAULT_PERMUTATION_COUNT
    if perm_count < 1:
        raise Error(
            "Permutation count should be positive, got " + String(perm_count)
        )
    if not has_permutation_features:
        perm_count = 1
    var est_perm = ctr_estimation_permutation_id
    if est_perm == -1:
        est_perm = perm_count - 1
    if est_perm < 0 or est_perm >= perm_count:
        raise Error(
            "ctr_estimation_permutation_id " + String(est_perm)
            + " is outside the " + String(perm_count)
            + " permutations this fit builds"
        )
    var binarized_target = List[UInt8]()
    var ctr_orders = List[List[UInt32]]()
    var target_classes_count = 0
    var any_cat = False
    for f in range(n_features):
        if (flags[f] & UInt32(1)) != UInt32(0):
            any_cat = True
    if len(dependent_configs) > 0 and any_cat:
        var target_borders = build_target_borders(
            y, TBinarizationOptions(
                BORDER_SELECTION_MIN_ENTROPY, GBDT_CTR_TARGET_BORDERS
            )
        )
        target_classes_count = len(target_borders) + 1
        binarized_target = build_binarized_target(y, target_borders)
        for p in range(perm_count):
            ctr_orders.append(ctrs_estimation_permutation(n_rows, p).fill_order())

    # ---- the column loop (`:1532-1676`) ----
    var columns = List[List[Float32]]()
    var column_kind = List[Int]()
    var column_ctr_grid = List[Int]()  # the config slot, or -1
    var dep_ordinal_of_column = List[Int]()
    var dep_by_perm = List[List[List[Float32]]]()
    for _ in range(perm_count):
        dep_by_perm.append(List[List[Float32]]())
    var tables = List[TCtrValueTable]()
    var ctr_column_count = 0
    var n_dep = 0
    for f in range(n_features):
        var col = List[Float32](capacity=n_rows)
        for r in range(n_rows):
            col.append(x_colmajor[f * n_rows + r])
        if kinds[f] != GBDT_COL_CTR:
            columns.append(col^)
            column_kind.append(kinds[f])
            column_ctr_grid.append(-1)
            dep_ordinal_of_column.append(-1)
            continue
        var codes = List[UInt32](capacity=n_rows)
        var maxc = 0
        for r in range(n_rows):
            var c = dense_category_code(col[r], f, r)
            if c > maxc:
                maxc = c
            codes.append(UInt32(c))
        var unique_values = maxc + 1
        var ctr_columns = List[List[Float32]]()
        for _ in range(len(configs)):
            ctr_columns.append(List[Float32]())
        for c in range(len(independent_configs)):
            ctr_columns[independent_slots[c]] = _feature_freq_column(
                codes, unique_values, independent_configs[c]
            )
        var dep_first_ordinal = n_dep
        if len(dependent_configs) > 0:
            for p in range(perm_count):
                var dep = _ordered_target_columns(
                    codes, unique_values, dependent_configs, binarized_target,
                    ctr_orders[p],
                )
                for c in range(len(dependent_slots)):
                    dep_by_perm[p].append(dep[c].copy())
                    if p == est_perm:
                        ctr_columns[dependent_slots[c]] = dep[c].copy()
            n_dep += len(dependent_slots)
        var base_col = len(columns)
        var ft = build_ctr_tables(
            codes, unique_values, configs, binarized_target,
            target_classes_count, f, base_col,
        )
        for c in range(len(ft)):
            tables.append(ft[c].copy())
        for c in range(len(ctr_columns)):
            columns.append(ctr_columns[c].copy())
            column_kind.append(GBDT_COL_CTR)
            column_ctr_grid.append(c)
            var ord = -1
            for d in range(len(dependent_slots)):
                if dependent_slots[d] == c:
                    ord = dep_first_ordinal + d
            dep_ordinal_of_column.append(ord)
            ctr_column_count += 1
    var n_columns = len(columns)

    # ---- `_quantize_training_columns` (`:2345-2700`) ----
    var float_cols = List[Int]()
    for c in range(n_columns):
        if column_kind[c] == GBDT_COL_FLOAT:
            float_cols.append(c)
    var borders = List[List[Float32]]()
    var fold_counts = List[Int]()
    var nan_treatment = List[Int]()
    var one_hot = List[Bool]()
    for c in range(n_columns):
        borders.append(List[Float32]())
        fold_counts.append(0)
        nan_treatment.append(NAN_TREATMENT_AS_IS)
        one_hot.append(column_kind[c] == GBDT_COL_ONE_HOT)
    if len(float_cols) > 0:
        var fx = List[Float32](capacity=len(float_cols) * n_rows)
        for k in range(len(float_cols)):
            for r in range(n_rows):
                fx.append(columns[float_cols[k]][r])
        var fg = gbdt_host_grid(
            fx, n_rows, len(float_cols), params.border_count,
            params.border_build_max_samples, params.random_seed,
            params.nan_mode, params.border_type,
        )
        for k in range(len(float_cols)):
            var c = float_cols[k]
            borders[c] = fg.borders[k].copy()
            fold_counts[c] = fg.fold_counts[k]
            nan_treatment[c] = fg.nan_treatment[k]
    for c in range(n_columns):
        if column_kind[c] == GBDT_COL_ONE_HOT:
            var maxc = 0
            for r in range(n_rows):
                var code = Int(columns[c][r])
                if code > maxc:
                    maxc = code
            if maxc > 254:
                raise Error(
                    "one-hot feature " + String(c)
                    + " has more than 255 categories"
                )
            var bs = List[Float32]()
            for code in range(maxc):
                bs.append(Float32(code) + Float32(0.5))
            fold_counts[c] = len(bs) + 1 if len(bs) > 0 else 0
            borders[c] = bs^
        elif column_kind[c] == GBDT_COL_CTR:
            var grid_desc = _ctr_grid_for(configs[column_ctr_grid[c]])
            var bs: List[Float32]
            comptime if IDN_CTR_BORDERS_DEVICE:
                # lane/fam2-gbdt F6: the device build's host restatement
                if dep_ordinal_of_column[c] >= 0:
                    bs = _ctr_borders_host_grid(
                        dep_by_perm[0][dep_ordinal_of_column[c]], grid_desc
                    )
                else:
                    bs = _ctr_borders_host_grid(columns[c], grid_desc)
            else:
                if dep_ordinal_of_column[c] >= 0:
                    # PERMUTATION 0'S VALUES decide a dependent column's grid
                    bs = compute_ctr_borders(
                        dep_by_perm[0][dep_ordinal_of_column[c]], grid_desc
                    )
                else:
                    bs = compute_ctr_borders(columns[c], grid_desc)
            fold_counts[c] = len(bs)
            borders[c] = bs^
    var grid = GbdtHostGrid(borders^, fold_counts^, nan_treatment^)
    var layout = build_layout(grid.fold_counts, one_hot)

    # ---- one compressed index per permutation (`:1745-1778`) ----
    var cindexes = List[List[UInt32]]()
    for p in range(perm_count):
        var flat = List[Float32](capacity=n_columns * n_rows)
        for c in range(n_columns):
            var ord = dep_ordinal_of_column[c]
            if ord >= 0:
                for r in range(n_rows):
                    flat.append(dep_by_perm[p][ord][r])
            else:
                for r in range(n_rows):
                    flat.append(columns[c][r])
        cindexes.append(_binarize_columns(flat, n_rows, n_columns, grid, layout))

    var fit = gbdt_host_boost(
        cindexes, est_perm, y, n_rows, n_columns, grid^, layout, params,
        -1, Float32(1.0), Float32(0.0), GbdtHostEval.none(), List[UInt32](),
    )
    return GbdtCtrHostFit(fit^, one_hot^, tables^, ctr_column_count)


def gbdt_ctr_host_model_text(r: GbdtCtrHostFit) raises -> String:
    """`model_text` (`gbdt/models/model_text.mojo:374-670`) for an oblivious
    one-dimensional model with CTR and one-hot columns and zero bias."""
    ref m = r.fit.model
    var n_columns = len(m.fold_counts)
    var table_of_column = List[Int](length=n_columns, fill=-1)
    for i in range(len(r.tables)):
        var c = r.tables[i].column
        if c < 0 or c >= n_columns:
            raise Error(
                "a CTR table names column " + String(c) + " of "
                + String(n_columns)
            )
        if table_of_column[c] != -1:
            raise Error("two CTR tables name column " + String(c))
        table_of_column[c] = i
    var out = String("")
    out += "# mojolearn model. One record per line, keyword first.\n"
    out += "# Every float is <decimal>/<IEEE-754 bits in hex>; the BITS are\n"
    out += "# what is loaded, because this toolchain's decimal formatter\n"
    out += "# loses one ULP on ~0.46% of float32 values (measured).\n"
    out += "# Format and CTR seam: gbdt/models/model_text.mojo.\n"
    out += String("format ") + String("mojolearn-model") + " " + String(2) + "\n"
    out += String("features ") + String(n_columns) + " " + String(len(r.one_hot)) + "\n"
    out += String("trees ") + String(m.n_trees()) + "\n"
    out += String("losses ") + String(len(m.losses)) + "\n"
    if r.ctr_column_count != 0:
        out += String("ctr_columns ") + String(r.ctr_column_count) + "\n"
    for f in range(n_columns):
        var kind = String("float")
        if table_of_column[f] >= 0:
            kind = String("ctr")
        elif r.one_hot[f]:
            kind = String("cat")
        var line = (
            String("feature ") + String(f) + " folds " + String(m.fold_counts[f])
            + " one_hot " + String(1 if r.one_hot[f] else 0)
            + " type " + kind
            + " nan " + _nan_token(m.nan_treatment[f])
            + " borders " + String(len(m.borders[f]))
        )
        for b in range(len(m.borders[f])):
            line += " " + gbdt_f32_token(m.borders[f][b])
        out += line + "\n"
    for f in range(n_columns):
        var ti = table_of_column[f]
        if ti < 0:
            continue
        ref tab = r.tables[ti]
        var classes = tab.target_classes_count
        var per_entry = classes if classes > 0 else 1
        if classes < 0:
            raise Error(
                "a CTR table for column " + String(f)
                + " declares a negative target class count"
            )
        if len(tab.counts) % per_entry != 0:
            raise Error(
                "the CTR table for column " + String(f) + " carries "
                + String(len(tab.counts)) + " counts, which is not a whole"
                " number of " + String(per_entry) + "-class histograms"
            )
        var entries = len(tab.counts) // per_entry
        out += (
            String("ctr_table ") + String(f)
            + " source " + String(tab.source_feature)
            + " type " + ctr_type_name(tab.ctr_type)
            + " prior_num " + gbdt_f32_token(tab.prior_num)
            + " prior_denom " + gbdt_f32_token(tab.prior_denom)
            + " shift " + gbdt_f32_token(tab.shift)
            + " scale " + gbdt_f32_token(tab.scale)
            + " denom " + String(tab.counter_denominator)
            + " classes " + String(classes)
            + " target_border " + String(tab.target_border_idx)
            + " entries " + String(entries) + "\n"
        )
        for c in range(entries):
            var line = String("ctr_entry ") + String(f) + " " + String(c)
            for k in range(per_entry):
                line += " " + String(tab.counts[c * per_entry + k])
            out += line + "\n"
    for t in range(m.n_trees()):
        var lo = m.tree_split_offsets[t]
        var depth = m.tree_split_offsets[t + 1] - lo
        var leaf_lo = m.tree_leaf_offsets[t]
        var n_values = 1 << depth
        if m.tree_leaf_offsets[t + 1] - leaf_lo != n_values:
            raise Error("tree " + String(t) + " has the wrong leaf count")
        out += String("tree ") + String(t) + " depth " + String(depth) + " dim 1 weights 0\n"
        for level in range(depth):
            var fid = m.split_features[lo + level]
            var line = (
                String("split ") + String(t) + " " + String(level) + " "
                + String(fid) + " " + String(m.split_bins[lo + level])
            )
            if r.one_hot[fid]:
                line += " split_type take_bin"
            out += line + "\n"
        for i in range(n_values):
            out += (
                String("leaf ") + String(t) + " " + String(i) + " "
                + gbdt_f32_token(m.leaf_values[leaf_lo + i]) + "\n"
            )
    for i in range(len(m.losses)):
        out += String("loss ") + String(i) + " " + gbdt_f64_token(m.losses[i]) + "\n"
    return out^


def _ctr_borders_host_grid(
    values: List[Float32], description: TBinarizationOptions
) raises -> List[Float32]:
    """`IDN_CTR_BORDERS_DEVICE`: `gbdt/train.mojo::_ctr_borders_device` on
    host memory: `gbdt_host_grid` (the device border build's restatement)
    over every row of the one column at NaN mode Forbidden, then the
    constant-feature 0.5."""
    var grid = gbdt_host_grid(
        values, len(values), 1, description.border_count, 0, UInt64(0),
        NAN_MODE_FORBIDDEN, ctr_border_type_code(description),
    )
    var bs = grid.borders[0].copy()
    if len(bs) == 0:
        bs.append(Float32(0.5))
    return bs^
