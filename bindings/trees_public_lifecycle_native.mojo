# SPDX-License-Identifier: Apache-2.0
# Public RF/ET/DT validation, fitted state and output boundary. CV's existing
# trees_forest_* operations remain internal Array operations; public methods
# use this envelope instead of the Python _mode/_sk_validate decorators.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_contains, py_fields
from bindings.classical_attributes import classical_check_layout
from bindings.classical_arrays import classical_is_array
from bindings.classical_pandas import classical_admit_value
from bindings.classical_validate import classical_check_x, classical_check_y, classical_check_width, classical_estimator_name
from bindings.classical_labels import labels_shape_of, _flatten_labels
from bindings.classical_exceptions import classical_require_fitted
from bindings.classifier_public_lifecycle import classifier_after_fit, _feature_names
from bindings.prep_lda_admission import lda_native_labels
from bindings.metrics_native_support import metric_warn
from bindings.native_caller_frame import native_caller_globals
from bindings.sequential_inputs import sequential_input
from bindings.trees_public_forest import forest_fit, forest_predict, forest_predict_proba, forest_kind, forest_get_params
from bindings.trees_public_score import forest_score
from bindings.trees_public_inputs import trees_empty, trees_address, trees_cast


def _outside() raises -> Bool:
    # One Python public wrapper hands over to this owner; inspect its caller
    # using CPython frame storage, preserving the established marker order.
    var globals = native_caller_globals(1)
    if globals is None:
        return True
    var name = py_get(globals, "__name__")
    var outside = True
    if py_type(name) is py_type(PythonObject("")):
        var module = String(name)
        outside = module != "mojolearn" and not module.startswith("mojolearn.") and not module.startswith("mojolearn_verify_all_")
    if py_get(globals, "__mojolearn_user_boundary__") is PythonObject(True):
        return True
    if py_get(globals, "__mojolearn_internal__") is PythonObject(True):
        return False
    return outside


def _feature_check(model: PythonObject, fields: PythonObject, source: PythonObject,
                   context: PythonObject) raises:
    var names = _feature_names(source, context)
    var fitted = py_get(fields, "feature_names_in_")
    var b = Python.import_module("builtins")
    var name = classical_estimator_name(model)
    if fitted is None:
        if names is not None:
            metric_warn("X has feature names, but " + name + " was fitted without feature names", b.UserWarning, context)
        return
    if names is None:
        metric_warn("X does not have valid feature names, but " + name + " was fitted with feature names", b.UserWarning, context)
        return
    classical_admit_value(fitted, context)
    var before = _flatten_labels(fitted, context)
    var after = _flatten_labels(names, context)
    classical_admit_value(before, context)
    classical_admit_value(after, context)
    if not Bool(py=before == after):
        raise Error("The feature names should match those that were passed during fit.")


def _set_params(model: PythonObject, values: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
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
    if operation == "trees_public_forest_set_params":
        return _set_params(model, args[0], context)
    var b = Python.import_module("builtins")
    var fields = py_fields(model)
    var names = py_type(model)._parameter_names
    for i in range(len(names)):  # driver-loop: stored constructor arguments before native conversions
        if py_contains(fields, names[i]):
            classical_admit_value(fields[names[i]], context)
    classical_admit_value(args, context)
    var name = classical_estimator_name(model)
    var fitting = operation == "trees_public_forest_fit"
    if not fitting:
        classical_require_fitted(model, False, "score" if operation.endswith("score") else "predict", context)
    var x = classical_check_x(args[0], name, context)
    var shape = labels_shape_of(x, context)
    var rows = shape[0]
    var cols = shape[1]
    var classifier = forest_kind(model, context).endswith("_classifier")
    if not fitting:
        classical_check_width(cols, Int(py=fields["n_features_in_"]), name)
        _feature_check(model, fields, args[0], context)
    var result = PythonObject(None)
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
    if operation == "trees_public_forest_predict":
        result = forest_predict(model, Python.tuple(x), context)
    elif operation == "trees_public_forest_proba":
        result = forest_predict_proba(model, Python.tuple(x), context)
    elif operation == "trees_public_forest_score":
        return forest_score(model, Python.tuple(x, args[1], args[2]), context)
    else:
        raise Error("unknown native public forest operation " + operation)
    if classical_is_array(result, context) and py_get(context, "ndarray") is not None and _outside():
        if operation == "trees_public_forest_predict" and not classifier and String(result.dtype) == "<f4":
            var wide = trees_empty(result.shape, "<f8", context)
            trees_cast(trees_address(result), "<f4", trees_address(wide), "<f8", Int(py=result.size), context)
            _ = result
            result = wide
        return sequential_input(result, context)
    return result
