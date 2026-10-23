# SPDX-License-Identifier: Apache-2.0
# `score` of the tree ensembles that inherit scikit-learn's mixin answer
# (`_mode.ClassifierScoreMixin.score`: the accuracy of `predict(X)`;
# `_mode.RegressorScoreMixin.score`: the float32 R^2 of `predict(X)`), in one
# native call: the estimator's native predict (bindings/trees_public_boost.mojo)
# and the native accuracy / R^2 (bindings/trees_public_score.mojo), with no
# Python between the prediction and the metric. The metric kernels and their
# order are the ones the mixins called, so the score keeps its bits.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_get, py_fields, py_setitem, py_contains
from bindings.py_typed_call import py_call_typed
from bindings.trees_public_inputs import (
    TreesMatrix, trees_f32, trees_builtins, trees_table, trees_entry, trees_helper, trees_vector,
    trees_address, trees_cast, trees_all_finite,
)
from bindings.trees_public_score import (
    trees_accuracy, trees_r2, trees_shape_of, trees_sample_weight_f32, ScoreLabels, _classification_encoded,
    _metrics_mode,
)
from bindings.trees_public_gbdt import _adapter_fitted, adapter_codes, adapter_decision_function
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


# ---------------------------------------------------- the GBDT adapters


def _metrics_table(mode: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_metrics_impl._get_binding(numeric_mode)`: the metrics binding of the
    requested mode, refused when the binary reports another."""
    var selected = _metrics_mode(mode, context)
    var metrics = trees_table(context, selected, "_mojolearn_metrics")
    var read_mode = trees_entry(metrics, "metrics_numeric_mode")
    if read_mode is None:
        read_mode = trees_entry(metrics, "umap_numeric_mode")
    if read_mode is not None:
        var code = Int(py=py_call_typed(read_mode))
        var actual = String("fast") if code == 0 else (String("identical") if code == 1 else String("None"))
        if actual != selected:
            raise Error("RuntimeError: mojolearn metrics: requested " + selected + ", binary reports "
                        + actual + "; rebuild it")
    return metrics


def _codes_i32(codes: PythonObject, n: Int, context: PythonObject) raises -> PythonObject:
    """`_as_i32_1d` of the adapter's integer class codes: int32 as they
    are, int64 range-checked and narrowed by `cast_elements`."""
    var dtype = String(codes.dtype)
    if dtype == "<i4":
        return codes
    if dtype != "<i8":
        raise Error("mojolearn metrics: y_pred must be an integer label array, got " + dtype
                    + "; the ported kernels take int32 labels")
    if n:
        var low = py_call_typed(trees_helper(context, "reduce_stat"), PythonObject(trees_address(codes)),
                                PythonObject(3), PythonObject(n), PythonObject(0))
        var high = py_call_typed(trees_helper(context, "reduce_stat"), PythonObject(trees_address(codes)),
                                 PythonObject(3), PythonObject(n), PythonObject(1))
        if Bool(py=low < PythonObject(-2147483648)) or Bool(py=high > PythonObject(2147483647)):
            raise Error("mojolearn metrics: y_pred does not fit in int32; the implemented "
                        + "kernels are the int32 instantiation (the int64 overload of "
                        + "adjusted_rand_index is the same code at a wider type and is "
                        + "not instantiated, metrics/NOT_IMPLEMENTED.tsv)")
    var out = trees_vector(n, "<i4", context)
    trees_cast(trees_address(codes), "<i8", trees_address(out), "<i4", n, context)
    _ = out  # keep the Python object alive through the native call above
    return out


def _vocabulary_codes(target: ScoreLabels, classes: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_label_map(target, vocabulary.get(label, -1))`: each label's index in
    the fitted `classes_`, -1 for a label outside them; one lookup per class
    and the base binding's gather for encoded labels."""
    var b = trees_builtins()
    var vocabulary = Python.dict()
    for i in range(len(classes)):  # driver-loop: over the fitted classes (the binary vocabulary)
        py_setitem(vocabulary, classes[i], PythonObject(i))
    if target.encoded:
        var k = len(target.classes)
        var table = trees_vector(k, "<i4", context)
        var p = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=trees_address(table))
        _ = table  # keep the Python object alive through the native call above
        for i in range(k):  # driver-loop: over the distinct target classes
            p[i] = Int32(Int(py=py_call(vocabulary.get, target.classes[i], PythonObject(-1))))
            _ = table  # keep the Python object alive through the native call above
        var out = trees_vector(target.n, "<i4", context)
        _ = py_call_typed(trees_helper(context, "gather_i32"), PythonObject(trees_address(table)), PythonObject(k),
                          PythonObject(trees_address(target.codes)), PythonObject(target.n),
                          PythonObject(trees_address(out)))
        _ = table  # keep the Python object alive through the native call above
        _ = out  # keep the Python object alive through the native call above
        return out
    var n = len(target.values)
    var out = trees_vector(n, "<i4", context)
    var q = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=trees_address(out))
    _ = out  # keep the Python object alive through the native call above
    for i in range(n):  # cpu-route: Python object labels mapped per row, the explicit label input step
        q[i] = Int32(Int(py=py_call(vocabulary.get, target.values[i], PythonObject(-1))))
        _ = out  # keep the Python object alive through the native call above
    return out


def gbdt_adapter_score(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`GradientBoostingClassifier.score` (the accuracy of the class codes
    against `y` mapped through `classes_`) and `GradientBoostingRegressor.score`
    (the float32 R^2 of `predict`)."""
    var X = args[0]
    var y = args[1]
    var sample_weight = args[2]
    _ = _adapter_fitted(model, context)
    var fields = py_fields(model)
    var mode = fields["numeric_mode_"]
    if py_contains(fields, "classes_"):
        var target = _classification_encoded(y, "y", context)
        if not Bool(py=PythonObject(target.kind) == fields["_label_kind_"]):
            raise Error("TypeError: score labels must have the training label type")
        var encoded = _vocabulary_codes(target, fields["classes_"], context)
        var predicted = adapter_codes(model, X, context)
        var n = Int(py=encoded.size)
        var m = Int(py=predicted.size)
        if n != m:
            raise Error("mojolearn metrics: y_true has " + String(n) + " entries and "
                        + "y_pred has " + String(m) + "; they must be the same length")
        var yp = _codes_i32(predicted, m, context)
        var metrics = _metrics_table(mode, context)
        var result = PythonObject(None)
        if sample_weight is not None:
            var w = trees_sample_weight_f32(sample_weight, n, "accuracy_score", context)
            result = py_call_typed(metrics["accuracy_score_weighted"], PythonObject(trees_address(encoded)),
                                   PythonObject(trees_address(yp)), PythonObject(w.addr), Python.list(PythonObject(n)))
            _ = encoded  # keep the Python object alive through the native call above
            _ = w  # keep the Python object alive through the native call above
            _ = yp  # keep the Python object alive through the native call above
            _ = w^
        else:
            result = py_call_typed(metrics["accuracy_score"], PythonObject(trees_address(encoded)),
                                   PythonObject(trees_address(yp)), Python.list(PythonObject(n)))
            _ = encoded  # keep the Python object alive through the native call above
            _ = yp  # keep the Python object alive through the native call above
        return py_call(trees_builtins().float, result)
    # `as_f32_c(y, ndim=None)` then `_regression_target`.
    var target = trees_f32(y, -1, "C", "y", context)
    var shape = trees_shape_of(target.keep, context)
    if len(shape) != 1 or target.rows * target.cols == 0:
        raise Error("y must be nonempty one-dimensional targets")
    if not trees_all_finite(target.addr, target.rows, False, context):
        raise Error("y must be finite")
    _ = target  # keep the Python object alive through the native calls above
    var prediction = adapter_decision_function(model, X, context)
    var result = trees_r2(target.keep, prediction, sample_weight, mode, context)
    _ = target^
    return PythonObject(result)
