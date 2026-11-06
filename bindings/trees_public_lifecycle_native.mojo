# SPDX-License-Identifier: Apache-2.0
# Public RF/ET/DT validation, fitted state and output boundary. CV's existing
# trees_forest_* operations remain internal Array operations; public methods
# use this envelope instead of the Python _mode/_sk_validate decorators.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_contains, py_fields
from bindings.native_loaded_module import native_loaded_module
from bindings.classical_attributes import classical_check_layout
from bindings.classical_arrays import classical_is_array
from bindings.classical_pandas import classical_admit_value
from bindings.classical_validate import classical_check_x, classical_check_y, classical_check_width, classical_estimator_name
from bindings.classical_labels import labels_shape_of, _flatten_labels
from bindings.classical_exceptions import classical_require_fitted
from bindings.classifier_public_lifecycle import classifier_after_fit, _feature_names
from bindings.prep_lda_admission import lda_native_labels
from bindings.metrics_native_support import metric_warn
from providers.python.estimator_lifecycle_pyapi import estimator_outside as _outside, estimator_feature_check as _feature_check
from bindings.sequential_inputs import sequential_input
from bindings.trees_public_forest import forest_fit, forest_public_predict, forest_kind, forest_get_params, forest_check_fit_device
from bindings.trees_public_score import forest_score_public
from bindings.trees_public_inputs import trees_empty, trees_address, trees_cast, trees_f32
from bindings.trees_public_auxiliary import forest_auxiliary, forest_shape
from bindings.trees_public_state import forest_restore_state, tree_is_fitted
from bindings.trees_side_records import trees_admit_parameters


def _set_params(model: PythonObject, values: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    if py_type(values) is not b.dict:
        raise Error("TypeError: native forest parameter updates require an ordinary dictionary")
    if len(values) == 0:
        return model
    var params = forest_get_params(model, context)
    var names = py_call(b.list, values)
    var unknown = Python.list()
    for i in range(len(names)):  # driver-loop: parameter names before user-value storage
        if py_type(names[i]) is not b.str:
            raise Error("TypeError: native forest parameter names must be exact strings")
        if not py_contains(params, names[i]):
            py_call(unknown.append, names[i])
    if len(unknown):
        raise Error("Invalid parameter(s) " + String(py_call(b.repr, py_call(b.sorted, unknown))) + " for " + classical_estimator_name(model))
    py_call(params.update, values)
    var fields = py_fields(model)
    # Like reconstruction through the constructor, value checks stay at fit
    # and all prior fitted/cache state is removed only after binding succeeds.
    py_call(fields.clear)
    py_call(fields.update, params)
    py_setitem(fields, "_has_constructor_parameters", PythonObject(True))
    return model


def forest_public_operation(model: PythonObject, operation: String, args: PythonObject,
                            context: PythonObject) raises -> PythonObject:
    classical_check_layout(model, context)
    if operation == "trees_public_forest_setstate":
        return forest_restore_state(model, args[0], context)
    if operation == "trees_public_forest_is_fitted":
        return tree_is_fitted(model, True)
    if operation == "trees_public_forest_depth" or operation == "trees_public_forest_n_leaves":
        return forest_shape(model, operation, context)
    if operation == "trees_public_forest_set_params":
        return _set_params(model, args[0], context)
    if operation == "trees_public_forest_apply" or operation == "trees_public_forest_predict_with_leaves" or operation == "trees_public_forest_staged_predict" or operation == "trees_public_forest_auxiliary":
        var method = String(operation[byte=String("trees_public_forest_").byte_length():])
        # Retained e087692fd public baseline: these auxiliary APIs return
        # Array/tuple-of-Arrays even externally with NumPy loaded. Preserve
        # that boundary and Float32 means, including regressors.
        return forest_auxiliary(model, method, args, context)
    var b = native_loaded_module("builtins")
    var fields = py_fields(model)
    # Once per configuration, not per call (lane perf-trees-owed): the
    # identity record of the stored constructor arguments lives in the
    # native side table (bindings/trees_side_records.mojo).
    trees_admit_parameters(model, fields, py_type(model)._parameter_names, context)
    var fitting = operation == "trees_public_forest_fit"
    if fitting:
        # Parameter validation precedes input admission and conversion.
        forest_check_fit_device(model, context)
    classical_admit_value(args, context)
    var name = classical_estimator_name(model)
    if not fitting:
        classical_require_fitted(model, False, "score" if operation.endswith("score") else "predict", context)
    var x = classical_check_x(args[0], name, context)
    var kind = forest_kind(model, context)
    var classifier = kind.endswith("_classifier")
    var result = PythonObject(None)
    if not fitting:
        # classical_check_x returned C-order float32 X: view it once here and
        # hand that view to the predict (lane perf-trees-owed, 2026-11-06;
        # the forest re-viewed X and re-checked the fitted state and width).
        var matrix = trees_f32(x, 2, "C", "X", context)
        classical_check_width(matrix.cols, Int(py=fields["n_features_in_"]), name)
        _feature_check(model, fields, args[0], context)
        if operation == "trees_public_forest_predict":
            result = forest_public_predict(model, kind, matrix, False, False, context)
        elif operation == "trees_public_forest_proba":
            result = forest_public_predict(model, kind, matrix, True, False, context)
        elif operation == "trees_public_forest_score":
            return forest_score_public(model, kind, matrix, args[1], args[2], context)
        else:
            raise Error("unknown native public forest operation " + operation)
        _ = matrix^
        return _public_result(result, operation, classifier, context)
    var shape = labels_shape_of(x, context)
    var rows = shape[0]
    if fitting:
        var y = args[1]
        if classifier:
            lda_native_labels(y, context, True, name)
            var target_shape = labels_shape_of(y, context)
            if len(target_shape) == 0 or target_shape[0] != rows:
                raise Error("Found input variables with inconsistent numbers of samples")
        else:
            y = classical_check_y(y, rows, name, context)
        var depth_object = py_get(fields, "_mojolearn_fitting", PythonObject(0))
        if py_type(depth_object) is not b.int:
            raise Error("TypeError: native forest fit depth must be an exact integer")
        var depth = Int(py=depth_object)
        py_setitem(fields, "_mojolearn_fitting", PythonObject(depth + 1))
        try:
            result = forest_fit(model, Python.tuple(x, y, args[2], args[3], args[4]), context)
        finally:
            fields = py_fields(model)
            if depth > 0:
                py_setitem(fields, "_mojolearn_fitting", PythonObject(depth))
            else:
                py_call(fields.pop, "_mojolearn_fitting", PythonObject(None))
        if classifier:
            classifier_after_fit(fields, args[0], context)
        else:
            var feature_names = _feature_names(args[0], context)
            if feature_names is None:
                py_call(fields.pop, "feature_names_in_", PythonObject(None))
            else:
                py_setitem(fields, "feature_names_in_", feature_names)
            py_setitem(fields, "_mojolearn_fitted", PythonObject(True))
        return result
    raise Error("unknown native public forest operation " + operation)


def _public_result(var result: PythonObject, operation: String, classifier: Bool,
                   context: PythonObject) raises -> PythonObject:
    if classical_is_array(result, context) and py_get(context, "ndarray") is not None and _outside():
        if operation == "trees_public_forest_predict" and not classifier and String(result.dtype) == "<f4":
            var wide = trees_empty(result.shape, "<f8", context)
            trees_cast(trees_address(result), "<f4", trees_address(wide), "<f8", Int(py=result.size), context)
            _ = result
            result = wide
        return sequential_input(result, context)
    return result
