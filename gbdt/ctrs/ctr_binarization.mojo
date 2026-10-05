# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Two grids a CTR needs, and NEITHER is the numeric feature grid.

Reference: three CatBoost sources (CatBoost `54a8143a`):

* `library/cpp/grid_creator/binarization.cpp:1262-1310`
  (`TUniformBinarizer::BestSplit`), the border rule a **Borders** CTR
  column is quantized with.
* `catboost/cuda/gpu_data/gpu_binarization_helpers.cpp:6-28`
  (`ComputeBorders`), which is the CTR-column entry point and carries the
  constant-feature hack.
* `catboost/cuda/gpu_data/dataset_helpers.cpp:137-151`
  (`BuildBinarizedTarget`) together with the target-border build at
  `catboost/cuda/train_lib/train.cpp:370-375`, which is TARGET
  binarization.

## WHY THIS FILE IS NOT `grid_creator/binarization.mojo`

It should be. `TUniformBinarizer` lives in the same reference file as
`GreedyLogSum` and `MinEntropy`, so `gbdt/grid_creator/binarization.mojo`
is its mirror address. It is here because that file is owned by another
lane in this round and a two-lane edit of one file is a merge conflict, not
an implementation. **Move `uniform_borders` there when the lanes rejoin**; nothing
else in this file belongs in `grid_creator/`.

## THE TWO GRIDS, AND THE ONE THAT WAS MIS-ATTRIBUTED

Their GPU resolves the simple-ctr defaults to two descriptions
(`SetCtrDefaults`, `catboost_options.cpp:449-452`):

    Borders      priors {0,1},{0.5,1},{1,1}   ctr_binarization  Uniform 15
    FeatureFreq  prior  {0.0,1}               ctr_binarization  MinEntropy 15

`Borders` takes the two-argument `TCtrDescription` constructor, whose
default is `TBinarizationOptions(EBorderSelectionType::Uniform, 15)`
(`cat_feature_options.cpp:167-170`). `FeatureFreq` is built by
`CreateDefaultCounter` (`catboost_options.cpp:392-415`), which passes
`MinEntropy` for `SimpleCtr` and `Median` for `TreeCtr` explicitly, and
`SetDefaultBinarizationsIfNeeded` (`:418-427`) re-applies MinEntropy to any
FeatureFreq description that left it unset -- it touches NOTHING else, so
Borders keeps Uniform.

So `Uniform, 15` is a real default on a real code path: **the Borders one.**
It is only the wrong path for FeatureFreq, and reading it as the CTR
binarizer for both is the mistake this repository has now made in both
directions. Verified against CatBoost's own resolved config, printed from a
1.2.10 CPU fit: `simple_ctrs[0].ctr_binarization = {border_count 15,
border_type Uniform}`, `target_binarization = {border_count 1, border_type
MinEntropy}`.

## Target binarization: MinEntropy with ONE border, and no per-CTR override

`TCatFeatureParams`'s constructor sets
`TargetBinarization("target_binarization", TBinarizationOptions(MinEntropy,
1))` (`cat_feature_options.cpp:230`), the plain-options layer spells its
border count `ctr_target_border_count`
(`plain_options_helper.cpp:421`), and the GPU REFUSES a per-CTR target
binarization outright: `CB_ENSURE(ctr.TargetBinarization.IsDefault(),
"Error: GPU doesn't not support target binarization per CTR description
currently. Please use ctr_target_border_count option instead")`
(`catboost_options.cpp:505`). One grid, for every CTR in the fit.

The borders themselves come from the same `TBordersBuilder` every feature
grid comes from (`train.cpp:370-375`), so target binarization at
`(MinEntropy, 1)` is exactly `best_split_min_entropy(y, 1)` -- the implementation
already in `grid_creator/binarization.mojo`, bit-exact to CatBoost over ten
budgets. Nothing new is computed here; what was missing was the CALL.

`TargetBinarization` also carries `DisableMaxSubsetSizeForBuildBordersOption()`
(`cat_feature_options.cpp:239`), so unlike a numeric feature grid the target
grid is built from ALL rows with no subsampling. That matters for us in
one direction only: it is the same all-rows rule `train.mojo` already uses.
"""

from gbdt.grid_creator.binarization import (
    BORDER_TYPE_MIN_ENTROPY,
    BORDER_TYPE_UNIFORM,
    best_split_min_entropy,
    binarize,
)
from std.sys.compile import is_defined

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

#: lane/fam2-gbdt F6 (IDENTICAL; CANDIDATE ARM, default OFF, enable with
#: `-D MOJOLEARN_IDN_GBDT_CTR_BORDERS_DEVICE` on the device AND the host
#: bindings): a simple-CTR column's grid is built on the device by
#: `device_float_borders` (the float columns' border pipeline: device sort,
#: device MinEntropy dynamic program or Uniform search, every row, NaN mode
#: Forbidden) instead of `compute_ctr_borders`' host scan / host sort of
#: every row per CTR column. The host column
#: (`gbdt/host/gbdt_oracle_ctr.mojo`) takes `gbdt_host_grid`, that build's
#: host restatement, under this same constant, so all four columns move
#: together. BITS MAY MOVE against `compute_ctr_borders` (the device build
#: flushes subnormals by bits and its Uniform is the float-feature Uniform),
#: which is why it is an arm to time and to compare on a categorical fixture
#: before it becomes a default. The two-level FeatureFreq tensor path
#: (`estimator.mojo`, `tensor_ctr_value_table.mojo`) is not touched.
#:
#: lane cpu4-gbdt (2026-10-04): THE ONLY ROUTE, every mode and vendor. The
#: GPU fit's simple-CTR columns are now resident on the device
#: (`gbdt/train.mojo`), so their grids come from the device build over the
#: resident column; `compute_ctr_borders`' host pass is no longer reachable
#: from the GPU fit and the candidate define is retired (no `_OFF`: the
#: owner's rule, no CPU data work in a GPU fit). The host column keeps
#: reading this constant and so takes `_ctr_borders_host_grid`, the device
#: build's restatement, with it. BITS MOVE against the old default
#: (`compute_ctr_borders`), on every column together.
comptime IDN_CTR_BORDERS_DEVICE = True


# --- EBorderSelectionType (`grid_creator/binarization.h:13-21`) ----------
#
# THEIR VALUES, which start at 1 and are not contiguous with anything.

comptime BORDER_SELECTION_MEDIAN = 1
comptime BORDER_SELECTION_GREEDY_LOG_SUM = 2
comptime BORDER_SELECTION_UNIFORM_AND_QUANTILES = 3
comptime BORDER_SELECTION_MIN_ENTROPY = 4
comptime BORDER_SELECTION_MAX_LOG_SUM = 5
comptime BORDER_SELECTION_UNIFORM = 6
comptime BORDER_SELECTION_GREEDY_MIN_ENTROPY = 7


def border_selection_name(t: Int) -> String:
    if t == BORDER_SELECTION_MEDIAN:
        return String("Median")
    if t == BORDER_SELECTION_GREEDY_LOG_SUM:
        return String("GreedyLogSum")
    if t == BORDER_SELECTION_UNIFORM_AND_QUANTILES:
        return String("UniformAndQuantiles")
    if t == BORDER_SELECTION_MIN_ENTROPY:
        return String("MinEntropy")
    if t == BORDER_SELECTION_MAX_LOG_SUM:
        return String("MaxLogSum")
    if t == BORDER_SELECTION_UNIFORM:
        return String("Uniform")
    return String("GreedyMinEntropy")


@fieldwise_init
struct TBinarizationOptions(Copyable, ImplicitlyCopyable, Movable):
    """`NCatboostOptions::TBinarizationOptions`, the two fields this implementation
    reaches. `NanMode` is `DisableNanModeOption()`d on both the CTR and the
    target descriptions (`cat_feature_options.cpp:219-223`, `:238`), so
    Forbidden is the only value either can hold and it is not carried."""

    var border_selection_type: Int
    var border_count: Int


def uniform_borders(
    values: List[Float32], max_borders_count: Int
) -> List[Float32]:
    """`TUniformBinarizer::BestSplit`
    (`library/cpp/grid_creator/binarization.cpp:1262-1310`), the arms this
    implementation reaches (no `DefaultValue`, no `initialBorders`, no
    `quantizedDefaultBinFraction`).

        currentValue = minValue + (i + 1) * (maxValue - minValue)
                                  / (maxBordersCount + 1)

    for `i` in `0..maxBordersCount-1`, inserted into a `THashSet<float>` so
    duplicates collapse. THE ARITHMETIC IS `double` AND THE STORE IS
    `float`, which is theirs literally: `currentValue` is declared `double`
    over `float` operands, and only the insert narrows. Computing it in
    Float32 throughout moves the last bits of every border.

    `minValue == maxValue` returns EMPTY (`:1282-1284`), which is a
    constant column; the caller's hack, not this function's, turns that
    into a single 0.5 border.
    """
    var out = List[Float32]()
    if len(values) == 0:
        return out^

    var min_value = values[0]
    var max_value = values[0]
    for i in range(1, len(values)):
        if values[i] < min_value:
            min_value = values[i]
        if values[i] > max_value:
            max_value = values[i]

    return uniform_borders_from_range(min_value, max_value, max_borders_count)


def uniform_borders_from_range(
    min_value: Float32, max_value: Float32, max_borders_count: Int
) -> List[Float32]:
    """`uniform_borders` after its min/max scan: the grid reads nothing of
    the column but its two extremes, so a caller that reduced them on the
    device (lane cpu4-gbdt, the two-level FeatureFreq fit) gets the same
    borders from the same arithmetic."""
    var out = List[Float32]()
    if min_value == max_value:
        return out^

    var lo = Float64(min_value)
    var hi = Float64(max_value)
    for i in range(max_borders_count):  # small-loop(max_borders_count: grid borders, the CTR description's count): border arithmetic, model parameters
        var current_value = lo + Float64(i + 1) * (hi - lo) / Float64(
            max_borders_count + 1
        )
        var b = Float32(current_value)
        # their `THashSet<float>`; the sequence is monotone, so the
        # previous element is the only possible duplicate
        if len(out) == 0 or out[len(out) - 1] != b:
            out.append(b)
    return out^


def ctr_border_type_code(description: TBinarizationOptions) raises -> Int:
    """`IDN_CTR_BORDERS_DEVICE`: the `BORDER_TYPE_*` code
    (`gbdt/grid_creator/binarization.mojo`) of a CTR grid's selection type,
    with `compute_ctr_borders`' refusal for the types the simple-ctr
    defaults never reach."""
    if description.border_selection_type == BORDER_SELECTION_MIN_ENTROPY:
        return BORDER_TYPE_MIN_ENTROPY
    if description.border_selection_type == BORDER_SELECTION_UNIFORM:
        return BORDER_TYPE_UNIFORM
    raise Error(
        "ctr_binarization border_type="
        + border_selection_name(description.border_selection_type)
        + " is not implemented; the GPU simple-ctr defaults are Uniform for"
        " Borders and MinEntropy for FeatureFreq"
        " (catboost_options.cpp:392-415, cat_feature_options.cpp:167-170)"
    )


def compute_ctr_borders(
    values: List[Float32], description: TBinarizationOptions
) raises -> List[Float32]:
    """`ComputeBorders` (`gpu_binarization_helpers.cpp:6-28`) reduced to the
    two selection types the GPU simple-ctr defaults reach.

    Theirs sorts on the device with `RadixSort` and reads the sorted column
    back before handing it to the host grid builder; both of our binarizers
    sort internally, so the sort is not repeated here. The `//hack to work
    with constant features` (`:22-27`) IS reproduced: an empty border list
    becomes a single 0.5, because a CTR column with one distinct value
    still has to occupy a feature slot with at least one bin boundary.

    `Median` is refused rather than approximated. It is the TREE-ctr
    FeatureFreq default (`CreateDefaultCounter`'s `TreeCtr` branch), and
    tree CTRs are not implemented, so a Median border here would be a value
    nothing produces.
    """
    var borders: List[Float32]
    if description.border_selection_type == BORDER_SELECTION_MIN_ENTROPY:
        borders = best_split_min_entropy(
            values.copy(), description.border_count
        )
    elif description.border_selection_type == BORDER_SELECTION_UNIFORM:
        borders = uniform_borders(values, description.border_count)
    else:
        raise Error(
            "ctr_binarization border_type="
            + border_selection_name(description.border_selection_type)
            + " is not implemented; the GPU simple-ctr defaults are Uniform for"
            " Borders and MinEntropy for FeatureFreq"
            " (catboost_options.cpp:392-415, cat_feature_options.cpp:167-170)"
        )
    if len(borders) == 0:
        # `//hack to work with constant features` (`:22-27`)
        borders.append(Float32(0.5))
    return borders^


def build_target_borders(
    y: List[Float32], description: TBinarizationOptions
) raises -> List[Float32]:
    """`featuresManager.SetTargetBorders(TBordersBuilder(...)(
    GetTargetBinarizationDescription()))` (`train.cpp:370-375`).

    At the GPU default the description is `(MinEntropy, 1)` and this is one
    call into the DP that `pixi run check-minentropy` already gates against
    CatBoost's own borders.
    """
    if description.border_selection_type != BORDER_SELECTION_MIN_ENTROPY:
        raise Error(
            "target_binarization border_type="
            + border_selection_name(description.border_selection_type)
            + " is not implemented; CatBoost's default is MinEntropy"
            " (cat_feature_options.cpp:230)"
        )
    if description.border_count < 1:
        raise Error(
            "Error: border count should be greater than 0. Got "
            + String(description.border_count)
        )
    return best_split_min_entropy(y.copy(), description.border_count)


def build_binarized_target(
    y: List[Float32], target_borders: List[Float32]
) -> List[UInt8]:
    """`BuildBinarizedTarget` (`dataset_helpers.cpp:137-151`).

    `NCB::BinarizeLine<ui8>(targets, ENanMode::Forbidden, borders)`, whose
    per-value body is `GetBinFromBorders` -- the count of borders the value
    STRICTLY exceeds (`quantization/utils.h:27-48`). That is the same
    convention `binarize` already implements for feature columns, so the
    call is shared rather than re-derived.

    Their `else` branch (no target binarization configured) fills zeros;
    here that case is the caller's, because a fit with no binarized-target
    CTR never calls this at all.
    """
    var out = List[UInt8]()
    for r in range(len(y)):
        out.append(UInt8(binarize(y[r], target_borders)))
    return out^
