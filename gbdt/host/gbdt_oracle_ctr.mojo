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
`counter_calc_method` SkipTest. Held-out rows use the learn pool's final
CTR tables before quantization against the training grid. They never join
the CTR counts, target grid or permutation histories. The binding retains
the categorical arm's other option restrictions.

WHAT IS MIRRORED, IN THE ORDER THE FIT REACHES IT

  1. `train`'s CTR prelude (`gbdt/train.mojo:1336-1491`): the configs of
     `simple_ctr_configs` in slot order, split by permutation dependence
     (Borders dependent, FeatureFreq not); `has_permutation_features` and
     the Plain collapse to ONE permutation without them (`:1381-1434`);
     `permutation_count` (4 unset) and the estimation permutation (the last
     unless named); the target borders and the binarized target; one CTR
     order per permutation, `ctr_estimation_order_host(n, p)` (the device's
     keyed order, `gbdt/ctrs/ctr_order.mojo`).
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
  5. Held-out raw columns through the shared `expand_raw_columns`, as in
     `train`'s eval prelude, then `_binarize_columns` with the model grid
     and layout. `fit_with_test` over the permutations: `gbdt_host_boost`,
     including the held-out curve, detector and best-model shrink.
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
from gbdt.ctrs.ctr_order import learn_order_host
from gbdt.data.permutation import (
    DEFAULT_PERMUTATION_COUNT,
    ctrs_estimation_permutation,
)
from gbdt.ctrs.tree_ctr import (
    TREE_COL_FLOAT,
    TREE_COL_ONE_HOT,
    TREE_COL_SIMPLE_CTR,
    TreeCtrPlan,
)
from gbdt.data.quantization import NAN_TREATMENT_AS_IS, NAN_TREATMENT_AS_TRUE
from gbdt.gpu_data.compressed_index_builder import (
    CompressedIndexLayout,
    build_layout,
)
from core.host_parallel import host_parallelize
from core.host_predict_threads import HostF32Ptr, HostU32Ptr, host_predict_chunk
from gbdt.host.gbdt_oracle import (
    GbdtHostFitWithEval,
    GbdtHostGrid,
    GbdtHostModel,
    GbdtHostParams,
    GbdtIntPtr,
    _binarize_columns,
    _gbdt_units,
    _nan_token,
    gbdt_f32_token,
    gbdt_f64_token,
    gbdt_host_boost,
    gbdt_host_grid,
)
from gbdt.host.gbdt_oracle_eval import GbdtHostEval
from gbdt.host.gbdt_oracle_tree_ctr import GbdtTreeCtrHost
from gbdt.models.combination_map import TCombinationCtrs, combination_maps_text
from gbdt.models.ctr_value_table import (
    TCtrValueTable,
    build_ctr_tables,
    dense_category_code,
    expand_raw_columns,
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
#: a tree-CTR slot of the search layout (`gbdt_ctr_host_fit` only; never a
#: model column and never returned by `gbdt_ctr_column_kinds`)
comptime GBDT_COL_TREE_SLOT = 3


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
    max + 1, every code below it present, at least two.

    SERIAL BY NATURE (admission, once per fit and feature): the refusal
    names the FIRST bad row in row order (`dense_category_code` raises),
    and the presence set grows with the running maximum."""
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


#: the pointer a host task reads the binarized target through
comptime _CtrU8Ptr = MutPointer[UInt8, MutUntrackedOrigin]


def _ctr_copy_into(
    src: List[Float32], src_offset: Int, count: Int, mut dst: List[Float32],
    dst_offset: Int,
):
    """`dst[dst_offset + r] = src[src_offset + r]` for `count` rows (one
    column's values; a move per row), the rows in `_gbdt_units` contiguous
    ranges."""
    var units = _gbdt_units(count, count)
    var chunk = host_predict_chunk(count, units)
    var src_p = rebind[HostF32Ptr](src.unsafe_ptr())
    var dst_p = rebind[HostF32Ptr](dst.unsafe_ptr())

    def rows(unit: Int) {imm chunk, imm count, imm src_offset, imm dst_offset, imm src_p, imm dst_p}:
        for r in range(unit * chunk, min(count, (unit + 1) * chunk)):
            dst_p[dst_offset + r] = src_p[src_offset + r]

    host_parallelize(rows, units)
    _ = len(src)
    _ = len(dst)


def _ctr_max_code_range(col: HostF32Ptr, lo: Int, hi: Int) -> Int:
    var maxc = 0
    for r in range(lo, hi):
        var code = Int(col[r])
        if code > maxc:
            maxc = code
    return maxc


def _ctr_max_code(col: List[Float32]) -> Int:
    """The largest `Int(value)` of a column, 0 at least (an integer
    maximum: any grouping). One maximum a task over a contiguous row
    range, then the maximum of those."""
    var n = len(col)
    var units = _gbdt_units(n, 2 * n)
    var chunk = host_predict_chunk(n, units)
    var best = List[Int](length=max(units, 1), fill=0)
    var col_p = rebind[HostF32Ptr](col.unsafe_ptr())
    var best_p = rebind[GbdtIntPtr](best.unsafe_ptr())

    def rows(unit: Int) {imm chunk, imm n, imm col_p, imm best_p}:
        best_p[unit] = _ctr_max_code_range(
            col_p, min(n, unit * chunk), min(n, (unit + 1) * chunk)
        )

    host_parallelize(rows, units)
    var maxc = 0
    for u in range(len(best)):  # driver-loop: over the tasks' maxima
        if best[u] > maxc:
            maxc = best[u]
    _ = len(col)
    return maxc


def _ctr_codes_of(col: List[Float32], mut codes: List[UInt32]):
    """The dense codes of a CTR column, `UInt32(Int(value))` per row. The
    column was admitted by `gbdt_ctr_column_kinds` (`_dense_cardinality`
    ran `dense_category_code` on every row), so this is the conversion
    alone: a move per row, the rows in `_gbdt_units` contiguous ranges."""
    var n = len(codes)
    var units = _gbdt_units(n, 2 * n)
    var chunk = host_predict_chunk(n, units)
    var col_p = rebind[HostF32Ptr](col.unsafe_ptr())
    var codes_p = rebind[HostU32Ptr](codes.unsafe_ptr())

    def rows(unit: Int) {imm chunk, imm n, imm col_p, imm codes_p}:
        for r in range(unit * chunk, min(n, (unit + 1) * chunk)):
            codes_p[r] = UInt32(Int(col_p[r]))

    host_parallelize(rows, units)
    _ = len(col)
    _ = len(codes)


def _feature_freq_column(
    codes: List[UInt32], unique_values: Int, config: TCtrConfig
) -> List[Float32]:
    """`TWeightedBinFreqCalcerGpu.visit_equal_up_to_prior_freq_ctrs`
    (`ctr_calcers.mojo`): the per-category INTEGER row count, converted to
    Float32 once, then `(count + prior) / (totalWeight + priorObservations)`
    with `totalWeight = Float32(n)` (lane cpu4-gbdt: the device counts the
    segment length; below 2^24 rows this is the old Float32 running sum's
    word, above it the count stays exact).

    THE UNITS (lane gbdt-host-par). The counts are integers, exact in any
    grouping: task `c` counts its contiguous row range per category and
    the tasks' counts are added after the join (a task holds one count per
    category, so the task count is also bounded by `n / categories`). The
    quotient is one statement per row, in contiguous row ranges."""
    var n = len(codes)
    var cats = max(unique_values, 1)
    var units = _gbdt_units(n, 2 * n)
    units = max(1, min(units, n // cats))
    var chunk = host_predict_chunk(n, units)
    var task_counts = List[Int](length=units * cats, fill=0)
    var codes_p = rebind[HostU32Ptr](codes.unsafe_ptr())
    var counts_p = rebind[GbdtIntPtr](task_counts.unsafe_ptr())

    def count(unit: Int) {imm chunk, imm n, imm cats, imm codes_p, imm counts_p}:
        var base = unit * cats
        for r in range(unit * chunk, min(n, (unit + 1) * chunk)):
            var k = base + Int(codes_p[r])
            counts_p[k] = counts_p[k] + 1

    host_parallelize(count, units)
    for u in range(1, units):  # driver-loop: over the tasks' count planes, added onto the first
        for k in range(cats):
            task_counts[k] += task_counts[u * cats + k]
    var total = Float32(n)
    var prior = config.numerator_shift()
    var prior_obs = config.denumerator_shift()
    var column = List[Float32](length=n, fill=Float32(0.0))
    var column_p = rebind[HostF32Ptr](column.unsafe_ptr())
    var q_units = _gbdt_units(n, 4 * n)
    var q_chunk = host_predict_chunk(n, q_units)

    def quotient(unit: Int) {imm q_chunk, imm n, imm total, imm prior, imm prior_obs, imm codes_p, imm counts_p, imm column_p}:
        for r in range(unit * q_chunk, min(n, (unit + 1) * q_chunk)):
            column_p[r] = (Float32(counts_p[Int(codes_p[r])]) + prior) / (
                total + prior_obs
            )

    host_parallelize(quotient, q_units)
    _ = len(codes)
    _ = len(task_counts)
    _ = len(column)
    return column^


def _ordered_target_column_into(
    codes: HostU32Ptr,
    unique_values: Int,
    num: Float32,
    denom: Float32,
    param: UInt32,
    binarized_target: _CtrU8Ptr,
    order: HostU32Ptr,
    n: Int,
    column: HostF32Ptr,
):
    """One ordered Borders column over one estimation order. SERIAL BY
    NATURE: position `i` reads its category's counts of the rows BEFORE it
    in the order (the device's stable sort by category then segmented
    scan), so the positions of one column are a running prefix."""
    var before = List[Float32](length=unique_values, fill=Float32(0.0))
    var hits = List[Float32](length=unique_values, fill=Float32(0.0))
    for i in range(n):
        var row = Int(order[i])
        var k = Int(codes[row])
        column[row] = (hits[k] + num) / (before[k] + denom)
        before[k] += Float32(1.0)
        if UInt32(binarized_target[row]) > param:
            hits[k] += Float32(1.0)


def _ordered_target_columns(
    codes: List[UInt32],
    unique_values: Int,
    configs: List[TCtrConfig],
    binarized_target: List[UInt8],
    orders: List[List[UInt32]],
) raises -> List[List[Float32]]:
    """`compute_simple_ctrs_gpu` (`ctr_calcers.mojo:801-891`) for Borders
    configs over EVERY estimation order: `TCtrBinBuilderGpu(order)` sorts
    the order STABLY by category, `GatherTrivialWeights` + the segmented
    scan give each row the count of its category's rows BEFORE it in that
    order, `FillBinarizedTargetsStats` + the second scan the count of
    those whose binarized target exceeds ParamId, and `MakeMeansAndScatter`
    divides with each config's prior (`(tmp + num) / (weights + denom)`).

    Returns the columns as `[order * len(configs) + config]`.

    THE UNIT IS ONE COLUMN, a (permutation, config) pair: its counts and
    its values are its own, so the columns run on `_gbdt_units` tasks, a
    contiguous run of columns each. Inside a column the positions are a
    running prefix along the order and stay one chain
    (`_ordered_target_column_into`)."""
    var n = len(codes)
    var n_orders = len(orders)
    var n_configs = len(configs)
    for p in range(n_orders):  # driver-loop: over the permutations, their admission
        if len(orders[p]) != n:
            raise Error(
                "ctr estimation order has " + String(len(orders[p]))
                + " entries for " + String(n) + " rows"
            )
    if len(binarized_target) != n:
        raise Error(
            "binarized target has " + String(len(binarized_target))
            + " entries for " + String(n) + " rows"
        )
    var nums = List[Float32]()
    var denoms = List[Float32]()
    var params = List[Int]()
    for c in range(n_configs):  # driver-loop: over the configs, their admission and priors
        if configs[c].ctr_type != CTR_BORDERS:
            raise Error(
                "the ordered CTR restatement takes Borders configs only; got "
                + ctr_type_name(configs[c].ctr_type)
            )
        nums.append(configs[c].numerator_shift())
        denoms.append(configs[c].denumerator_shift())
        params.append(Int(configs[c].param_id))
    var total = n_orders * n_configs
    var columns = List[List[Float32]]()
    var column_at = List[Int]()
    for _ in range(total):  # driver-loop: over the columns, their storage
        columns.append(List[Float32](length=n, fill=Float32(0.0)))
    for k in range(total):  # driver-loop: over the columns, their addresses
        column_at.append(Int(columns[k].unsafe_ptr()))
    var order_at = List[Int]()
    for p in range(n_orders):  # driver-loop: over the permutations, their addresses
        order_at.append(Int(orders[p].unsafe_ptr()))
    var units = _gbdt_units(total, 6 * total * n)
    var chunk = host_predict_chunk(total, units)
    var codes_p = rebind[HostU32Ptr](codes.unsafe_ptr())
    var target_p = _CtrU8Ptr(unsafe_from_address=Int(binarized_target.unsafe_ptr()))
    var nums_p = rebind[HostF32Ptr](nums.unsafe_ptr())
    var denoms_p = rebind[HostF32Ptr](denoms.unsafe_ptr())
    var params_p = rebind[GbdtIntPtr](params.unsafe_ptr())
    var column_at_p = rebind[GbdtIntPtr](column_at.unsafe_ptr())
    var order_at_p = rebind[GbdtIntPtr](order_at.unsafe_ptr())

    def column_run(unit: Int) {imm chunk, imm total, imm n, imm n_configs, imm unique_values, imm codes_p, imm target_p, imm nums_p, imm denoms_p, imm params_p, imm column_at_p, imm order_at_p}:
        for k in range(unit * chunk, min(total, (unit + 1) * chunk)):
            var p = k // n_configs
            var c = k - p * n_configs
            _ordered_target_column_into(
                codes_p, unique_values, nums_p[c], denoms_p[c],
                UInt32(params_p[c]), target_p,
                HostU32Ptr(unsafe_from_address=order_at_p[p]), n,
                HostF32Ptr(unsafe_from_address=column_at_p[k]),
            )

    host_parallelize(column_run, units)
    _ = len(codes)
    _ = len(binarized_target)
    _ = len(orders)
    _ = len(nums)
    _ = len(denoms)
    _ = len(params)
    _ = len(column_at)
    _ = len(order_at)
    _ = len(columns)
    return columns^


struct GbdtCtrHostFit(Movable):
    """The fit and what its model text needs beside the ensemble."""

    var fit: GbdtHostFitWithEval
    var one_hot: List[Bool]
    var tables: List[TCtrValueTable]
    var ctr_column_count: Int
    #: the tree CTRs (feature combinations) the trees chose: the model's
    #: last columns; empty at `max_ctr_complexity` 1
    var combination_ctrs: TCombinationCtrs

    def __init__(
        out self,
        var fit: GbdtHostFitWithEval,
        var one_hot: List[Bool],
        var tables: List[TCtrValueTable],
        ctr_column_count: Int,
        var combination_ctrs: TCombinationCtrs,
    ):
        self.fit = fit^
        self.one_hot = one_hot^
        self.tables = tables^
        self.ctr_column_count = ctr_column_count
        self.combination_ctrs = combination_ctrs^


@fieldwise_init
struct GbdtCtrHostShape(Copyable, Movable):
    """What `gbdt_ctr_host_prelude` resolved beside the lists it fills."""

    #: the estimation permutation (the last unless named)
    var est_perm: Int
    #: the search layout's columns: the base columns, then the tree-CTR
    #: slots when `max_ctr_complexity` is above 1
    var n_columns: Int
    #: the columns before the tree-CTR slots (the model's base columns)
    var n_base_columns: Int
    var ctr_column_count: Int
    var layout: CompressedIndexLayout


def gbdt_ctr_host_prelude(
    x_colmajor: List[Float32],
    y: List[Float32],
    n_rows: Int,
    n_features: Int,
    params: GbdtHostParams,
    flags: List[UInt32],
    permutation_count: Int,
    ctr_estimation_permutation_id: Int,
    order_block: Int,
    eval_x: List[Float32],
    eval_rows: Int,
    max_ctr_complexity: Int,
    mut grid_out: GbdtHostGrid,
    mut cindexes_out: List[List[UInt32]],
    mut test_cindex_out: List[UInt32],
    mut one_hot_out: List[Bool],
    mut tables_out: List[TCtrValueTable],
    mut tree_state_out: List[GbdtTreeCtrHost],
) raises -> GbdtCtrHostShape:
    """Everything a categorical host fit does before its boosting loop
    (steps 1 to 4 of the module docstring, and the held-out index of step
    5): the column kinds, the CTR columns over each permutation's order,
    the apply-time tables, the grids, the layout and one compressed index
    per permutation. The Plain fit (`gbdt_ctr_host_fit`) and the Ordered
    fit (`gbdt/host/gbdt_oracle_ordered.mojo::gbdt_ordered_host_fit`) both
    call it, so the two share one spelling of the columns.

    `order_block` is the block size of the CTR estimation orders: 1 for a
    Plain fit, the fold permutation block for an Ordered one (`train`'s
    `ctr_order_block`). `eval_rows` 0 is no held-out set. The six `_out`
    arguments are filled (whatever they held is replaced).

    The loops here run over model structure (features, configs,
    permutations: driver loops, whose bodies append to the fit's column
    lists in order). Inside them the rows run across the tasks
    (`_gbdt_units`): a column's copy, its dense codes, its FeatureFreq
    counts and quotients, the one-hot maximum, the per-permutation flat
    planes; a feature's ordered Borders columns run one (permutation,
    config) column a task, because the running count along a permutation
    inside one ordered column is serial by nature; the orders themselves
    are written across the workers (`learn_order_host`)."""
    if max_ctr_complexity < 1:
        raise Error(
            "max_ctr_complexity must be at least 1; got "
            + String(max_ctr_complexity)
        )
    if n_rows < 1 or n_features < 1:
        raise Error("train requires at least one row and one feature")
    if len(x_colmajor) != n_rows * n_features:
        raise Error("x_colmajor size mismatch")
    if len(y) != n_rows:
        raise Error("y size mismatch")
    if len(flags) != n_features:
        raise Error("cat flags must be one per feature")
    if order_block < 1:
        raise Error("the CTR order block must be at least 1")
    if eval_rows < 0:
        raise Error("eval_set row count must not be negative")
    if eval_rows > 0 and len(eval_x) != eval_rows * n_features:
        raise Error("eval_set x_colmajor size mismatch")
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
            # the device's order (`gbdt/ctrs/ctr_order.mojo`, lane
            # cpu4-gbdt; seeded by lane cat-ctr): the load shuffle keyed
            # by `random_seed`, composed for every permutation but 0 with
            # its keyed bijection, the same words as
            # `launch_ctr_estimation_order` writes. `order_block` 1 is
            # the Plain CTR order (`ctr_estimation_order_host`'s words);
            # an Ordered fit passes its fold permutation block, so order
            # p IS its fold permutation p (`ordered_permutations`)
            ctr_orders.append(
                learn_order_host(n_rows, p, order_block, params.random_seed)
            )

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
    #: the raw input feature a float or one-hot column reads (-1 for a CTR
    #: column), and the CTR-bearing categorical features with their
    #: cardinalities and dense codes: read by the tree CTRs only
    var column_src = List[Int]()
    var tree_cat_features = List[Int]()
    var tree_cat_cards = List[Int]()
    var tree_cat_codes = List[List[UInt32]]()
    for f in range(n_features):  # driver-loop: over the input features (each appends its columns in order)
        var col = List[Float32](length=n_rows, fill=Float32(0.0))
        _ctr_copy_into(x_colmajor, f * n_rows, n_rows, col, 0)
        if kinds[f] != GBDT_COL_CTR:
            columns.append(col^)
            column_src.append(f)
            column_kind.append(kinds[f])
            column_ctr_grid.append(-1)
            dep_ordinal_of_column.append(-1)
            continue
        var codes = List[UInt32](length=n_rows, fill=UInt32(0))
        _ctr_codes_of(col, codes)
        var unique_values = _ctr_max_code(col) + 1
        var ctr_columns = List[List[Float32]]()
        for _ in range(len(configs)):
            ctr_columns.append(List[Float32]())
        for c in range(len(independent_configs)):
            ctr_columns[independent_slots[c]] = _feature_freq_column(
                codes, unique_values, independent_configs[c]
            )
        var dep_first_ordinal = n_dep
        if len(dependent_configs) > 0:
            # every (permutation, config) column of this feature at once,
            # one column a task
            var dep = _ordered_target_columns(
                codes, unique_values, dependent_configs, binarized_target,
                ctr_orders,
            )
            var n_dep_cfg = len(dependent_slots)
            for p in range(perm_count):  # driver-loop: over the permutations and configs, filing the columns
                for c in range(n_dep_cfg):
                    dep_by_perm[p].append(dep[p * n_dep_cfg + c].copy())
                    if p == est_perm:
                        ctr_columns[dependent_slots[c]] = dep[p * n_dep_cfg + c].copy()
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
            column_src.append(-1)
            ctr_column_count += 1
        if max_ctr_complexity > 1:
            tree_cat_features.append(f)
            tree_cat_cards.append(unique_values)
            tree_cat_codes.append(codes.copy())

    # TREE CTRS (`gbdt/train.mojo`, "TREE CTRS: the layout's slots"): after
    # every base column, `max_depth` persist slots and one candidate slot
    # per (CTR-bearing categorical feature, tree-CTR config), each a column
    # of zeros under a synthetic grid at the tree-CTR fold capacity. The
    # tree-CTR configs are `TCatFeatureParams.default().
    # combination_ctr_configs()`: the simple CTRs' own fan-out (Borders at
    # every default prior, FeatureFreq), every grid asked 15 borders.
    var n_base_columns = len(columns)
    var tree_configs = List[TCtrConfig]()
    if max_ctr_complexity > 1:
        tree_configs = configs.copy()
    var tree_on = len(tree_configs) > 0 and len(tree_cat_features) > 0
    var tree_fold_cap = GBDT_CTR_GRID_BORDERS
    if tree_on:
        var n_tree_slots = (
            params.max_depth + len(tree_cat_features) * len(tree_configs)
        )
        for _ in range(n_tree_slots):
            # no values are held for a slot: it is zero for every row
            columns.append(List[Float32]())
            column_src.append(-1)
            column_kind.append(GBDT_COL_TREE_SLOT)
            column_ctr_grid.append(-1)
            dep_ordinal_of_column.append(-1)
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
        var fx = List[Float32](length=len(float_cols) * n_rows, fill=Float32(0.0))
        for k in range(len(float_cols)):  # driver-loop: over the float columns, each one's rows across the tasks
            _ctr_copy_into(columns[float_cols[k]], 0, n_rows, fx, k * n_rows)
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
            var maxc = _ctr_max_code(columns[c])
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
        elif column_kind[c] == GBDT_COL_TREE_SLOT:
            # the fit writes bins into a slot, so its grid only has to give
            # it the fold capacity and send the initial zeros to bin 0
            var bs = List[Float32]()
            for b in range(tree_fold_cap):
                bs.append(Float32(b) + Float32(0.5))
            fold_counts[c] = len(bs)
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

    # TREE CTRS: the fit's state, one entry when on (`TreeCtrFit`'s
    # construction in `train`, on host lists)
    var tree_state = List[GbdtTreeCtrHost]()
    if tree_on:
        var tree_col_kind = List[Int]()
        var tree_col_cat = List[Int]()
        var tree_col_raw = List[Int]()
        var tree_col_borders = List[List[Float32]]()
        var tree_col_nan = List[Int]()
        for c in range(n_base_columns):
            if column_kind[c] == GBDT_COL_ONE_HOT:
                tree_col_kind.append(TREE_COL_ONE_HOT)
            elif column_kind[c] == GBDT_COL_CTR:
                tree_col_kind.append(TREE_COL_SIMPLE_CTR)
            else:
                tree_col_kind.append(TREE_COL_FLOAT)
            tree_col_cat.append(-1)
            tree_col_raw.append(column_src[c])
            tree_col_borders.append(grid.borders[c].copy())
            # the side of a `bin >` split a NaN lands on: the last bin
            # under AsTrue, bin 0 otherwise
            tree_col_nan.append(
                1 if grid.nan_treatment[c] == NAN_TREATMENT_AS_TRUE else 0
            )
        for t in range(len(tables)):
            tree_col_cat[tables[t].column] = tables[t].source_feature
        var tree_plan = TreeCtrPlan(
            tree_col_kind^, tree_col_cat^, tree_cat_features.copy(),
            tree_cat_cards.copy(), max_ctr_complexity,
        )
        var tree_border_counts = List[Int]()
        for _ in range(len(tree_configs)):
            tree_border_counts.append(GBDT_CTR_GRID_BORDERS)
        var tree_orders = List[List[UInt32]]()
        for p in range(len(ctr_orders)):
            tree_orders.append(ctr_orders[p].copy())
        var tree_test_x = List[Float32]()
        if eval_rows > 0:
            tree_test_x = eval_x.copy()
        else:
            # read through a pointer: never an empty list
            tree_test_x.append(Float32(0.0))
        tree_state.append(
            GbdtTreeCtrHost(
                tree_plan^, tree_configs.copy(), tree_border_counts^, n_rows,
                n_base_columns, params.max_depth, layout,
                tree_col_raw^, tree_col_borders^, tree_col_nan^,
                tree_cat_codes.copy(), tree_orders^, binarized_target.copy(),
                target_classes_count, perm_count,
                eval_rows if eval_rows > 0 else 0, n_features,
                tree_test_x^,
            )
        )

    # ---- one compressed index per permutation (`:1745-1778`) ----
    var cindexes = List[List[UInt32]]()
    for p in range(perm_count):  # driver-loop: over the permutations and columns, each column's rows across the tasks
        # a tree-CTR slot holds no values: its rows stay the fill's +0.0
        var flat = List[Float32](length=n_columns * n_rows, fill=Float32(0.0))
        for c in range(n_columns):
            var ord = dep_ordinal_of_column[c]
            if ord >= 0:
                _ctr_copy_into(dep_by_perm[p][ord], 0, n_rows, flat, c * n_rows)
            elif column_kind[c] != GBDT_COL_TREE_SLOT:
                _ctr_copy_into(columns[c], 0, n_rows, flat, c * n_rows)
        cindexes.append(_binarize_columns(flat, n_rows, n_columns, grid, layout))

    # Compiled 2026-10-28 at 2d5c5081a83e: full CPU GBDT host binding,
    # Mojo 1.0.0 (ed45d567), x86-64-v3, O0, 32 allocated CPUs/jobs, exit 0.
    # Runtime and identity remain pending; this is compilation evidence.
    # Receipt: plan gbdt-cpu-ctr-eval-20261028-compile (git history).
    # Match `train`'s eval prelude: use final learn-only CTR statistics,
    # not the ordered training columns or an eval-pool target histogram.
    # `expand_raw_columns` is the same host helper the GPU path calls; its
    # table lookups also implement the unseen-category prior. SkipTest is
    # the only counter mode this entry exposes. Supporting Full would
    # require admitting that option and carrying its counts explicitly.
    var test_cindex = List[UInt32]()
    if eval_rows > 0:
        # the base columns from the tables; a tree-CTR slot starts at zero
        # on the held-out rows too (the fit writes its bins)
        var eval_expanded = expand_raw_columns(
            tables, n_base_columns, eval_x, eval_rows
        )
        if tree_on:
            eval_expanded.resize(n_columns * eval_rows, Float32(0.0))
        test_cindex = _binarize_columns(
            eval_expanded, eval_rows, n_columns, grid, layout
        )
    grid_out = grid^
    cindexes_out = cindexes^
    test_cindex_out = test_cindex^
    one_hot_out = one_hot^
    tables_out = tables^
    tree_state_out = tree_state^
    return GbdtCtrHostShape(
        est_perm, n_columns, n_base_columns, ctr_column_count, layout^
    )



def gbdt_ctr_host_fit(
    x_colmajor: List[Float32],
    y: List[Float32],
    n_rows: Int,
    n_features: Int,
    params: GbdtHostParams,
    flags: List[UInt32],
    permutation_count: Int,
    ctr_estimation_permutation_id: Int,
    eval: GbdtHostEval = GbdtHostEval.none(),
    max_ctr_complexity: Int = 1,
) raises -> GbdtCtrHostFit:
    """`train` with categorical columns on the covered configuration (see
    the module docstring for what that is and what mirrors what).

    `max_ctr_complexity` above 1 grows tree CTRs (feature combinations),
    the host column of `gbdt/methods/tree_ctr_search.mojo`: the layout
    gains `max_depth` persist slots and one candidate slot per (CTR-bearing
    categorical feature, tree-CTR config), the boost loop takes its
    level-synchronized arm (`gbdt/host/gbdt_oracle.mojo::
    _tree_ctr_levels_host`), and the model's last columns are the
    combinations the trees chose. At 1 nothing here differs from the fit
    without them.

    The columns, grids, tables and per-permutation indexes are
    `gbdt_ctr_host_prelude`'s (which also checks the sizes of the learn
    arguments); this function is the Plain boosting loop over them and the
    model's tree-CTR tail."""
    if eval.n_rows < 0:
        raise Error("eval_set row count must not be negative")
    if eval.has_test():
        if len(eval.x_colmajor) != eval.n_rows * n_features:
            raise Error("eval_set x_colmajor size mismatch")
        if len(eval.y) != eval.n_rows:
            raise Error("eval_set y size mismatch")
    var grid = GbdtHostGrid(List[List[Float32]](), List[Int](), List[Int]())
    var cindexes = List[List[UInt32]]()
    var test_cindex = List[UInt32]()
    var one_hot = List[Bool]()
    var tables = List[TCtrValueTable]()
    var tree_state = List[GbdtTreeCtrHost]()
    # block 1: the Plain CTR orders. The held-out rows are read only when
    # there are some (`eval.n_rows` above 0).
    var shape = gbdt_ctr_host_prelude(
        x_colmajor, y, n_rows, n_features, params, flags, permutation_count,
        ctr_estimation_permutation_id, 1, eval.x_colmajor, eval.n_rows,
        max_ctr_complexity, grid, cindexes, test_cindex, one_hot, tables,
        tree_state,
    )
    var est_perm = shape.est_perm
    var n_columns = shape.n_columns
    var n_base_columns = shape.n_base_columns
    var ctr_column_count = shape.ctr_column_count
    var tree_on = len(tree_state) > 0
    var fit = gbdt_host_boost(
        cindexes, est_perm, y, n_rows, n_columns, grid^, shape.layout, params,
        -1, Float32(1.0), Float32(0.0), eval, test_cindex, tree_state,
    )

    # TREE CTRS: the model's columns are the base columns followed by one
    # column per combination the trees chose (its map, its table, the
    # borders it won under); the layout's slots do not reach the model, and
    # the trees' persist-slot splits are renumbered to those columns
    # (`train`'s own tail, on the flat host model).
    var combination_ctrs = TCombinationCtrs()
    if tree_on:
        var chosen = tree_state.pop()
        var section = chosen.assemble()
        var split_offsets = fit.model.tree_split_offsets.copy()
        chosen.rewrite_splits(
            split_offsets, fit.model.split_features, section.column_of
        )
        var m_folds = List[Int]()
        var m_one_hot = List[Bool]()
        var m_borders = List[List[Float32]]()
        var m_nan = List[Int]()
        for c in range(n_base_columns):
            m_folds.append(fit.model.fold_counts[c])
            m_one_hot.append(one_hot[c])
            m_borders.append(fit.model.borders[c].copy())
            m_nan.append(fit.model.nan_treatment[c])
        for i in range(len(section.borders)):
            m_folds.append(len(section.borders[i]))
            m_one_hot.append(False)
            m_borders.append(section.borders[i].copy())
            m_nan.append(NAN_TREATMENT_AS_IS)
        fit.model.fold_counts = m_folds^
        fit.model.borders = m_borders^
        fit.model.nan_treatment = m_nan^
        one_hot = m_one_hot^
        combination_ctrs = section.ctrs.copy()
    return GbdtCtrHostFit(
        fit^, one_hot^, tables^, ctr_column_count, combination_ctrs^
    )


def _ctr_table_records_host(
    keyword: String, f: Int, tab: TCtrValueTable
) raises -> String:
    """One table's header record under `keyword` (`ctr_table`, or
    `combination_table` for a tree CTR) and its `ctr_entry` records, for
    model column `f` (`model_text.mojo::_ctr_table_records`)."""
    var out = String("")
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
        keyword + " " + String(f)
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
    return out^


def gbdt_ctr_host_model_text(r: GbdtCtrHostFit) raises -> String:
    """`model_text` (`gbdt/models/model_text.mojo:374-670`) for an oblivious
    one-dimensional model with CTR and one-hot columns and zero bias."""
    return gbdt_ctr_host_model_text_of(
        r.fit.model, r.one_hot, r.tables, r.ctr_column_count,
        r.combination_ctrs,
    )


def gbdt_ctr_host_model_text_of(
    m: GbdtHostModel,
    one_hot: List[Bool],
    tables: List[TCtrValueTable],
    ctr_column_count: Int,
    combination_ctrs: TCombinationCtrs,
) raises -> String:
    """`gbdt_ctr_host_model_text` over the parts, for a fit that does not
    come as a `GbdtCtrHostFit` (the Ordered fit, which adds its `bias`
    record with `gbdt_text_with_bias`)."""
    var n_columns = len(m.fold_counts)
    var table_of_column = List[Int](length=n_columns, fill=-1)
    for i in range(len(tables)):
        var c = tables[i].column
        if c < 0 or c >= n_columns:
            raise Error(
                "a CTR table names column " + String(c) + " of "
                + String(n_columns)
            )
        if table_of_column[c] != -1:
            raise Error("two CTR tables name column " + String(c))
        table_of_column[c] = i
    # the tree-CTR columns (feature combinations): the model's last columns
    var n_comb = len(combination_ctrs.tables)
    if n_comb != 0:
        combination_ctrs.check(n_columns, n_columns)
    var first_comb = n_columns - n_comb
    var out = String("")
    out += "# mojolearn model. One record per line, keyword first.\n"
    out += "# Every float is <decimal>/<IEEE-754 bits in hex>; the BITS are\n"
    out += "# what is loaded, because this toolchain's decimal formatter\n"
    out += "# loses one ULP on ~0.46% of float32 values (measured).\n"
    out += "# Format and CTR seam: gbdt/models/model_text.mojo.\n"
    out += String("format ") + String("mojolearn-model") + " " + String(2) + "\n"
    out += String("features ") + String(n_columns) + " " + String(len(one_hot)) + "\n"
    out += String("trees ") + String(m.n_trees()) + "\n"
    out += String("losses ") + String(len(m.losses)) + "\n"
    if ctr_column_count != 0:
        out += String("ctr_columns ") + String(ctr_column_count) + "\n"
    for f in range(n_columns):
        var kind = String("float")
        if f >= first_comb:
            if one_hot[f] or table_of_column[f] >= 0:
                raise Error(
                    "tree-CTR column " + String(f) + " is also flagged"
                    " one-hot or carries a simple CTR table"
                )
            kind = String("tree_ctr")
        elif table_of_column[f] >= 0:
            kind = String("ctr")
        elif one_hot[f]:
            kind = String("cat")
        var line = (
            String("feature ") + String(f) + " folds " + String(m.fold_counts[f])
            + " one_hot " + String(1 if one_hot[f] else 0)
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
        out += _ctr_table_records_host(String("ctr_table"), f, tables[ti])
    # the tree CTRs: the maps, then one table per tree-CTR column in column
    # order (`model_text`'s order). A model without them writes nothing.
    if n_comb != 0:
        out += combination_maps_text(combination_ctrs.maps)
        for i in range(n_comb):
            out += _ctr_table_records_host(
                String("combination_table"), first_comb + i,
                combination_ctrs.tables[i],
            )
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
            if one_hot[fid]:
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
