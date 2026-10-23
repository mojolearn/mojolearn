# SPDX-License-Identifier: Apache-2.0
# `score` of the tree ensembles that inherit scikit-learn's mixin answer
# (`_mode.ClassifierScoreMixin.score`: the accuracy of `predict(X)`;
# `_mode.RegressorScoreMixin.score`: the float32 R^2 of `predict(X)`), in one
# native call: the estimator's native predict (bindings/trees_public_boost.mojo)
# and the native accuracy / R^2 (bindings/trees_public_score.mojo), with no
# Python between the prediction and the metric. The metric kernels and their
# order are the ones the mixins called, so the score keeps its bits.
from std.python import Python, PythonObject
from bindings.py_balanced import py_get, py_fields
from bindings.trees_public_inputs import TreesMatrix, trees_f32
from bindings.trees_public_score import trees_accuracy, trees_r2
from bindings.trees_public_boost import boost_operation, boost_kind


def boost_score(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`score(X, y, sample_weight)` of Bagging*, AdaBoost* and DART*."""
    var X = args[0]
    var y = args[1]
    var sample_weight = args[2]
    # The mixins pass the estimator's own numeric_mode (None: the default).
    var mode = py_get(py_fields(model), "numeric_mode")
    var prediction = boost_operation(model, "trees_boost_predict", Python.tuple(X), context)
    if boost_kind(model, context).endswith("_classifier"):
        return PythonObject(trees_accuracy(y, prediction, sample_weight, mode, context))
    # `as_f32_c(y, ndim=None)` and `as_f32_c(prediction, ndim=None)`, then
    # `r2_score`'s 1-D path.
    var truth = trees_f32(y, -1, "C", "y", context)
    var pred = trees_f32(prediction, -1, "C", "prediction", context)
    var result = trees_r2(truth.keep, pred.keep, sample_weight, mode, context)
    _ = truth^
    _ = pred^
    return PythonObject(result)
