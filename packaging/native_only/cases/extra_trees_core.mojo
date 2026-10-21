# SPDX-License-Identifier: Apache-2.0
# Native-only smoke helper: trained forests cross the same flat GPU prediction
# boundary as the package. Analytic expected outputs do not reuse tree walking.
from max.gpu.host import DeviceContext
from extratrees.estimator import (
    ExtraTreesConfig, MAX_FEATURES_ALL,
    fit_extra_trees_classifier_device, fit_extra_trees_regressor_device,
)
from extratrees.impl.randomforest.randomforest import Forest
from core.forest_inference import forest_predict_gpu


def predict_fitted(
    ctx: DeviceContext, forest: Forest, x: List[Float32], rows: Int,
    features: Int, outputs: Int,
) raises -> List[Float32]:
    var offsets = List[Int32]()
    var columns = List[Int32]()
    var thresholds = List[Float32]()
    var left = List[Int32]()
    var leaves = List[Float32]()
    offsets.append(0)
    var total = 0
    for t in range(len(forest.trees)):
        var n = forest.trees[t].num_nodes()
        if n < 3:
            raise Error("analytic forest must split its two distinct inputs")
        total += n
        offsets.append(Int32(total))
        for i in range(n):
            var node = forest.trees[t].sparsetree[i]
            columns.append(node.colid)
            thresholds.append(node.quesval)
            left.append(node.left_child_id)
        for i in range(len(forest.trees[t].vector_leaf)):
            leaves.append(forest.trees[t].vector_leaf[i])
    return forest_predict_gpu[False, False](
        ctx, offsets, columns, thresholds, left, leaves, x,
        rows, features, outputs,
    )


def check_extra_trees[REGRESSION: Bool]() raises:
    var ctx = DeviceContext()
    var rows = 64
    var features = 2
    var x = List[Float32]()
    var row_major = List[Float32]()
    var y = List[Float32]()
    # Repeated distinct inputs with a constant second feature guarantee that
    # any valid split of the signal feature separates both populations. This
    # has a closed-form prediction independent of the sampled threshold.
    for r in range(rows):
        var high = r % 2 != 0
        var value = Float32(10.0) if high else Float32(0.0)
        x.append(value)
        row_major.append(value)
        row_major.append(1.0)
        comptime if REGRESSION:
            y.append(Float32(-3.0) if high else Float32(2.0))
        else:
            y.append(Float32(1.0) if high else Float32(0.0))
    for _ in range(rows):
        x.append(1.0)
    var config = ExtraTreesConfig()
    comptime if REGRESSION:
        config = config.for_regression()
    config.n_estimators = 3
    config.max_depth = 3
    config.max_features_spec = MAX_FEATURES_ALL
    config.random_state = UInt64(0x7A31)
    comptime if REGRESSION:
        var fitted = fit_extra_trees_regressor_device(
            ctx, x, y, Int32(rows), Int32(features), config,
        )
        if len(fitted.forest.trees) != 3:
            raise Error("regression fit did not retain all three trees")
        var predicted = predict_fitted(ctx, fitted.forest, row_major, rows, features, 1)
        if len(predicted) != rows:
            raise Error("regression prediction shape differs")
        for r in range(rows):
            if predicted[r].to_bits() != y[r].to_bits():
                raise Error("regression prediction differs from analytic step")
    else:
        var fitted = fit_extra_trees_classifier_device(
            ctx, x, y, Int32(rows), Int32(features), Int32(2), config,
        )
        if len(fitted.forest.trees) != 3:
            raise Error("classification fit did not retain all three trees")
        var predicted = predict_fitted(ctx, fitted.forest, row_major, rows, features, 2)
        if len(predicted) != rows * 2:
            raise Error("classification probability shape differs")
        for r in range(rows):
            for c in range(2):
                var expected = Float32(1.0) if c == r % 2 else Float32(0.0)
                if predicted[2 * r + c].to_bits() != expected.to_bits():
                    raise Error("class probabilities differ from analytic partition")
    # Keep every borrowed input alive until fitting and prediction drain.
    _ = x^
    _ = y^
    _ = row_major^
