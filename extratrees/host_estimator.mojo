# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The ExtraTrees HOST estimator entries: the independent host references
the checks hold the device to (`*_reference`) and the CPU-only install's
fits (`*_host_exact`, what `bindings/_mojolearn_trees_host.mojo` runs).

Moved out of `extratrees/estimator.mojo` (cpu-gpu-cleanup t-forest) because
that module is the GPU binding's and a GPU install must not import the host
trainers (`extratrees/impl/randomforest/host_forest.mojo`). Nothing here
changed: the plans are the same `classifier_plan` / `regressor_plan` both GPU
arms call, so every refusal is theirs.
"""

from extratrees.estimator import (
    ExtraTreesConfig,
    FitResult,
    classifier_plan,
    depth_cap_bound,
    regressor_plan,
)
from extratrees.impl.randomforest.randomforest import class_ids_for_host
from extratrees.checks.fixed_point import choose_scale, quantize
from core.abs_sum_blocked_host import host_abs_sum_blocked
from checks.numerics import ftz
from extratrees.impl.decisiontree.batched_levelalgo.builder import (
    weighted_label_scale_ratio,
)
from extratrees.impl.randomforest.host_forest import (
    fit_classification,
    fit_forest_exact,
    fit_regression,
)


def fit_extra_trees_classifier_reference(
    x_col_major: List[Float32],
    labels: List[Float32],
    n_rows: Int32,
    n_features: Int32,
    n_classes: Int32,
    config: ExtraTreesConfig,
) raises -> FitResult:
    """Host reference for independent checks; public fit is GPU-only."""
    var plan = classifier_plan(config, n_rows, n_features)
    var forest = fit_classification(
        x_col_major,
        labels,
        n_rows,
        n_features,
        n_classes,
        plan.params,
        plan.n_trees,
        config.random_state,
        plan.bootstrap,
        plan.n_sampled_rows,
    )
    var bound = depth_cap_bound(forest, plan)
    return FitResult(forest^, plan, bound)


def fit_extra_trees_regressor_reference(
    x_col_major: List[Float32],
    y: List[Float32],
    n_rows: Int32,
    n_features: Int32,
    config: ExtraTreesConfig,
) raises -> FitResult:
    """Host reference for independent checks; public fit is GPU-only."""
    var plan = regressor_plan(config, n_rows, n_features)
    var forest = fit_regression(
        x_col_major,
        y,
        n_rows,
        n_features,
        plan.params,
        plan.n_trees,
        config.random_state,
        plan.bootstrap,
        plan.n_sampled_rows,
    )
    var bound = depth_cap_bound(forest, plan)
    return FitResult(forest^, plan, bound)


@fieldwise_init
struct HostRowWeights(Movable):
    """lane sk-kernels: the host column's `DeviceDataset.set_row_weights`."""

    var weighted: Bool
    var q: List[Int32]
    var min_weight_units: Int64
    var scale: Float64
    """lane sk-final: the power of two `q` is quantized at (1.0 unweighted)."""


def host_quantize_row_weights(
    weights: List[Float32], n_rows: Int, min_weight_fraction: Float64
) raises -> HostRowWeights:
    """`DeviceDataset.set_row_weights` on the host: the same refusals in
    the same words, the same blocked binary64 total
    (`host_abs_sum_blocked`, `device_scan_weights`' order), all 1.0 an
    unweighted fit, the same power-of-two scale and `Int32(w * scale)`
    truncation, the same `min_weight_units`."""
    if len(weights) != n_rows:
        raise Error("sample weights must be n_rows long")
    var all_unit = True
    for i in range(n_rows):
        var v = weights[i]
        if not (v >= Float32(0) and v <= Float32(3.4028234663852886e38)):
            raise Error("sample weights must be finite and nonnegative")
        if v != Float32(1):
            all_unit = False
    var total = host_abs_sum_blocked(Int(weights.unsafe_ptr()), n_rows)
    if not (total > 0.0):
        raise Error("sample weights must have positive total")
    if all_unit:
        return HostRowWeights(False, List[Int32](), Int64(0), Float64(1.0))
    var scale = choose_scale(total, n_rows)
    if scale < Float64(1.1754943508222875e-38) or scale > Float64(1.7014118346046923e38):
        raise Error("sample weights exceed the Float32 fixed-point scale range")
    var fscale = Float32(scale)
    var q = List[Int32](capacity=n_rows)
    for i in range(n_rows):
        q.append(Int32(Int(weights[i] * fscale)))
    var mwu = Int64(0)
    if min_weight_fraction > 0.0:
        var units = min_weight_fraction * total * scale
        mwu = Int64(Int(units))
        if Float64(Int(mwu)) < units:
            mwu += 1
    return HostRowWeights(True, q^, mwu, scale)


def fit_extra_trees_classifier_host_exact(
    x_col_major: List[Float32],
    labels: List[Float32],
    n_rows: Int32,
    n_features: Int32,
    n_classes: Int32,
    config: ExtraTreesConfig,
    tree_start: Int = 0,
    weights: List[Float32] = List[Float32](),
) raises -> FitResult:
    """THE CPU COLUMN'S CLASSIFIER FIT (the CPU training lane, phase 1,
    et-clf, 2026-09-14): `fit_extra_trees_classifier_device` restated on
    the host, over `fit_forest_exact` (the block comment above
    `train_tree_exact` in `batched_levelalgo/builder.mojo`). The plan is
    `classifier_plan`, the same resolver both GPU arms call, so every
    refusal is theirs; the label plane is `class_ids_for_host`, the device's
    cast with its range refusal. Takes no DeviceContext; what
    `bindings/_mojolearn_trees_host.mojo` runs, and what
    `tools/identity_break.py --diff ... --require-columns 4` holds to the
    three GPU columns.

    lane sk-kernels: `weights` (empty: none) is the per-row weight
    (`host_quantize_row_weights`); all 1.0 fits unweighted with
    `min_weight_fraction_leaf` folded into `min_samples_leaf`."""
    var rw = HostRowWeights(False, List[Int32](), Int64(0), Float64(1.0))
    if len(weights) > 0:
        rw = host_quantize_row_weights(
            weights, Int(n_rows), config.min_weight_fraction_leaf
        )
    var plan = classifier_plan(
        config, n_rows, n_features, fold_min_weight=not rw.weighted
    )
    var class_ids = class_ids_for_host(labels, n_rows, n_classes)
    var forest = fit_forest_exact(
        x_col_major,
        labels,
        class_ids,
        n_rows,
        n_features,
        n_classes,
        plan.params,
        plan.n_trees,
        config.random_state,
        True,
        Float32(1.0),
        plan.bootstrap,
        plan.n_sampled_rows,
        tree_start,
        rw.q.copy(),
        rw.min_weight_units,
    )
    var bound = depth_cap_bound(forest, plan)
    return FitResult(forest^, plan, bound)


def fit_extra_trees_regressor_host_exact(
    x_col_major: List[Float32],
    y: List[Float32],
    n_rows: Int32,
    n_features: Int32,
    config: ExtraTreesConfig,
    tree_start: Int = 0,
    weights: List[Float32] = List[Float32](),
) raises -> FitResult:
    """THE CPU COLUMN'S REGRESSOR FIT (et-reg, 2026-09-14):
    `fit_extra_trees_regressor_device` restated on the host. The labels
    are QUANTIZED by `quantize_labels_host` exactly as the device arm quantizes
    them (DEVIATION 135), the search is the device's exact `Int64` MSE key
    over those integers (DEVIATION 189) and the leaves are means of the
    quantized labels rescaled by `Float32(1 / scale)` as `leaf_kernel`
    computes them (DEVIATION 179), so the leaf VALUES are the device's,
    where `fit_extra_trees_regressor_reference` returns Float64 means that
    differ from the device by up to one quantization step and cannot be
    the CPU column. Takes no DeviceContext.

    lane sk-final: `weights` (empty: none) is the per-row weight,
    `upload_dataset_labels_quantized_weighted` restated: the weights through
    `host_quantize_row_weights` (all 1.0 fits unweighted with
    `min_weight_fraction_leaf` folded into `min_samples_leaf`), the labels
    the search reads are `ftz(w * y)` quantized as `quantize_labels_host`
    quantizes y, and the leaf scale is `weighted_label_scale_ratio`."""
    var rw = HostRowWeights(False, List[Int32](), Int64(0), Float64(1.0))
    if len(weights) > 0:
        rw = host_quantize_row_weights(
            weights, Int(n_rows), config.min_weight_fraction_leaf
        )
    var plan = regressor_plan(
        config, n_rows, n_features, fold_min_weight=not rw.weighted
    )
    var ql: Tuple[List[Int32], Float64]
    var leaf_scale: Float64
    if rw.weighted:
        var z = List[Float32](capacity=Int(n_rows))
        for i in range(Int(n_rows)):
            # `et_weighted_labels_kernel`: one float32 product, flushed
            z.append(ftz(weights[i] * y[i]))
        ql = quantize_labels_host(z, n_rows)
        leaf_scale = weighted_label_scale_ratio(ql[1], rw.scale)
    else:
        ql = quantize_labels_host(y, n_rows)
        leaf_scale = ql[1]
    var forest = fit_forest_exact(
        x_col_major,
        y,
        ql[0],
        n_rows,
        n_features,
        Int32(1),
        plan.params,
        plan.n_trees,
        config.random_state,
        False,
        Float32(1.0 / leaf_scale),
        plan.bootstrap,
        plan.n_sampled_rows,
        tree_start,
        rw.q.copy(),
        rw.min_weight_units,
    )
    var bound = depth_cap_bound(forest, plan)
    return FitResult(forest^, plan, bound)


def quantize_labels_host(
    y: List[Float32], n_rows: Int32
) raises -> Tuple[List[Int32], Float64]:
    """The label vector in deviation 135's fixed point, and its scale.

    `choose_scale` takes the sum of magnitudes over the WHOLE label vector,
    because any node's rows are a subset of it -- 135's bound, which makes
    accumulator overflow impossible rather than unlikely. The same derivation
    `device_regression_check` uses. cpu3-trees: this is the HOST COLUMN
    (CPU fits, checks, benches); the GPU fit quantizes on the device
    (`upload_dataset_labels_quantized`) to the same Int32s and scale.
    """
    # cpu3-trees: the blocked fixed-order sum (`core/abs_sum_blocked`), the
    # same word the device fit (`upload_dataset_labels_quantized`) reads back
    if len(y) < Int(n_rows):
        raise Error("quantize_labels_host: y is shorter than n_rows")
    var mag = host_abs_sum_blocked(Int(y.unsafe_ptr()), Int(n_rows))
    var scale = choose_scale(mag, Int(n_rows))
    var q = List[Int32]()
    for r in range(Int(n_rows)):
        q.append(Int32(quantize(Float64(y[r]), scale)))
    return (q^, scale)
