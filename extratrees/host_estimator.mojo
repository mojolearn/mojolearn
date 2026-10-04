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
    quantize_labels_host,
    regressor_plan,
)
from extratrees.impl.randomforest.randomforest import class_ids_for
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


def fit_extra_trees_classifier_host_exact(
    x_col_major: List[Float32],
    labels: List[Float32],
    n_rows: Int32,
    n_features: Int32,
    n_classes: Int32,
    config: ExtraTreesConfig,
    tree_start: Int = 0,
) raises -> FitResult:
    """THE CPU COLUMN'S CLASSIFIER FIT (the CPU training lane, phase 1,
    et-clf, 2026-09-14): `fit_extra_trees_classifier_device` restated on
    the host, over `fit_forest_exact` (the block comment above
    `train_tree_exact` in `batched_levelalgo/builder.mojo`). The plan is
    `classifier_plan`, the same resolver both GPU arms call, so every
    refusal is theirs; the label plane is `class_ids_for`, the device's
    cast with its range refusal. Takes no DeviceContext; what
    `bindings/_mojolearn_trees_host.mojo` runs, and what
    `tools/identity_break.py --diff ... --require-columns 4` holds to the
    three GPU columns."""
    var plan = classifier_plan(config, n_rows, n_features)
    var class_ids = class_ids_for(labels, n_rows, n_classes)
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
    the CPU column. Takes no DeviceContext."""
    var plan = regressor_plan(config, n_rows, n_features)
    var ql = quantize_labels_host(y, n_rows)
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
        Float32(1.0 / ql[1]),
        plan.bootstrap,
        plan.n_sampled_rows,
        tree_start,
    )
    var bound = depth_cap_bound(forest, plan)
    return FitResult(forest^, plan, bound)
