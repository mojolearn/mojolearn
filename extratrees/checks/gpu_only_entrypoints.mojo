# SPDX-License-Identifier: Apache-2.0
"""Compile/reach check for the public GPU-only no-context ET entrypoints.

    tools/with_build_lock.sh pixi run mojo run -I . \
        extratrees/checks/gpu_only_entrypoints.mojo

Repeat with -D MOJOLEARN_NUMERIC_DETERMINISTIC=1 and
-D MOJOLEARN_NUMERIC_IDENTICAL=1. Eight rows, one tree per fit. The comparison
uses the existing context-taking GPU APIs; independent objective/RNG checks
own mathematical correctness. No host-reference training is called here.
"""
from std.testing import assert_equal, assert_true
from max.gpu.host import DeviceContext
from extratrees.estimator import (
    ExtraTreesConfig, FitResult,
    fit_extra_trees_classifier, fit_extra_trees_classifier_device,
    fit_extra_trees_regressor, fit_extra_trees_regressor_device,
)


def same_model(a: FitResult, b: FitResult) raises:
    assert_equal(len(a.forest.trees), 1)
    assert_equal(len(b.forest.trees), 1)
    assert_equal(a.depth_cap_bound, b.depth_cap_bound)
    for t in range(len(a.forest.trees)):
        assert_equal(a.forest.trees[t].num_outputs, b.forest.trees[t].num_outputs)
        assert_equal(a.forest.trees[t].depth_counter, b.forest.trees[t].depth_counter)
        assert_equal(a.forest.trees[t].leaf_counter, b.forest.trees[t].leaf_counter)
        assert_equal(len(a.forest.trees[t].sparsetree), len(b.forest.trees[t].sparsetree))
        assert_true(len(a.forest.trees[t].sparsetree) > 1)
        for i in range(len(a.forest.trees[t].sparsetree)):
            var x = a.forest.trees[t].sparsetree[i]
            var y = b.forest.trees[t].sparsetree[i]
            assert_equal(x.colid, y.colid)
            assert_equal(x.quesval.to_bits(), y.quesval.to_bits())
            assert_equal(x.best_metric_val.to_bits(), y.best_metric_val.to_bits())
            assert_equal(x.left_child_id, y.left_child_id)
            assert_equal(x.instance_count, y.instance_count)
        assert_equal(len(a.forest.trees[t].vector_leaf), len(b.forest.trees[t].vector_leaf))
        for i in range(len(a.forest.trees[t].vector_leaf)):
            assert_equal(a.forest.trees[t].vector_leaf[i].to_bits(),
                         b.forest.trees[t].vector_leaf[i].to_bits())


def main() raises:
    var x: List[Float32] = [0, 1, 2, 3, 4, 5, 6, 7]
    var labels: List[Float32] = [0, 0, 0, 0, 1, 1, 1, 1]
    var targets: List[Float32] = [0.125, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    var cfg = ExtraTreesConfig()
    cfg.n_estimators = 1
    cfg.max_depth = 2
    cfg.random_state = 19
    var classification = fit_extra_trees_classifier(x, labels, 8, 1, 2, cfg)
    var ctx = DeviceContext()
    var classification_device = fit_extra_trees_classifier_device(
        ctx, x, labels, 8, 1, 2, cfg
    )
    same_model(classification, classification_device)
    var regression_cfg = cfg.for_regression()
    var regression = fit_extra_trees_regressor(x, targets, 8, 1, regression_cfg)
    var regression_device = fit_extra_trees_regressor_device(
        ctx, x, targets, 8, 1, regression_cfg
    )
    same_model(regression, regression_device)
    print("PASS: both public ET fits match context-taking GPU model bits")
