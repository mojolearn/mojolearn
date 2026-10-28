# SPDX-License-Identifier: Apache-2.0
# Public GBDT adapter boundary. The compiled tree owners retain training and
# prediction; this envelope replaces _mode's Python validation, fit tracking,
# feature-name checks and output conversion. NaN X remains the learner's policy.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.mode_metadata_native import _fields
from bindings.params_metadata_native import _read, _resolved
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.classical_attributes import classical_check_layout
from bindings.classical_pandas import classical_admit_value
from bindings.classical_validate import classical_refuse_sparse, classical_check_y, classical_check_width, classical_estimator_name
from bindings.classical_labels import labels_shape_of, _flatten_labels
from bindings.classical_exceptions import classical_require_fitted
from bindings.classifier_public_lifecycle import classifier_after_fit, classifier_label_list, _feature_names
from bindings.prep_lda_admission import lda_native_labels
from bindings.parallel_public_result import parallel_public_result
from providers.python.estimator_lifecycle_pyapi import estimator_feature_check
from bindings.trees_constructor_native import tree_parameter_storage
from bindings.trees_public_state import _release_tree_values, tree_is_fitted
from bindings.trees_public_inputs import trees_view, trees_check_rank, trees_f32_view
from bindings.trees_public_gbdt_fit import adapter_classifier_fit, adapter_regressor_fit, _adapter_eval
from bindings.trees_public_gbdt import adapter_decision_function, adapter_predict, adapter_predict_proba
from bindings.trees_public_mixin_score import gbdt_adapter_score


def _context(context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var result = py_call(b.dict, context)
    var real = py_call(b.list, Python.tuple(b.int, b.float, b.bool))
    _ = py_call(real.extend, context["scalar_types"])
    # Existing numerical/label checks use isinstance. Concrete native type
    # tuples prevent numbers.Real/Integral ABC hooks from executing in Python.
    py_setitem(result, "trees_real_type", py_call(b.tuple, real))
    py_setitem(result, "trees_integral_type", context["integer_types"])
    return result


def _learner_state(learner: PythonObject, context: PythonObject) raises:
    var b = native_loaded_module("builtins")
    var cls = context["trees_gbdt_adapter_tables"]["learner"]
    if py_type(learner) is not cls:
        raise Error("TypeError: GBDT adapters require their connected native learner")
    classical_check_layout(learner, context["trees_state_layouts"][cls])
    var fields = _fields(learner)
    var keys = py_call(b.list, fields)
    tree_parameter_storage(learner, keys)
    for i in range(len(keys)):  # driver-loop: fitted learner metadata before scalar protocols or native calls
        var name = String(keys[i])
        var value = fields[keys[i]]
        if name == "_resident":
            _release_tree_values(Python.list(value), Python.list(), context)
            if value is not None:
                if py_type(value) is not b.tuple or len(value) != 4:
                    raise Error("TypeError: GBDT resident cache must be a native four-item tuple")
                if py_type(value[2]) is not b.str or py_type(value[3]) is not b.int:
                    raise Error("TypeError: GBDT resident cache requires native text and handle metadata")
        else:
            classical_admit_value(value, context)
    var text = py_get(fields, "model_")
    if text is not None and py_type(text) is not b.str:
        raise Error("TypeError: GBDT fitted model text must be an exact string")


def _adapter_state(model: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    classical_check_layout(model, context)
    var fields = _fields(model)
    var names = Python.tuple("numeric_mode", "numeric_mode_", "model_", "n_features_in_", "n_classes_",
                             "_label_kind_", "_mojolearn_fitting", "_mojolearn_fitted")
    tree_parameter_storage(model, py_call(b.list, names))
    for i in range(len(names)):  # driver-loop: native prediction and fitted-check scalar state
        if py_contains(fields, names[i]):
            classical_admit_value(fields[names[i]], context)
    if py_contains(fields, "classes_"):
        classical_admit_value(classifier_label_list(fields["classes_"], context), context)
    if py_contains(fields, "feature_names_in_"):
        lda_native_labels(fields["feature_names_in_"], context)
    var learner = py_get(fields, "_learner_")
    if learner is not None:
        _learner_state(learner, context)
    return fields


def gbdt_adapter_public_operation(model: PythonObject, operation: String, args: PythonObject,
                                 context: PythonObject) raises -> PythonObject:
    # The archive operations historically always refused, without interpreting
    # their path or inspecting fitted state. Keep those explicit public refusals.
    if operation == "trees_public_gbdt_adapter_save":
        raise Error("NotImplementedError: GBDT adapter archives are not implemented; pickle retains adapter state")
    if operation == "trees_public_gbdt_adapter_load":
        raise Error("NotImplementedError: Legacy raw GBDT archives have no adapter vocabulary/parameters")
    var b = native_loaded_module("builtins")
    var cls = py_type(model)
    if py_type(cls) is not b.type:
        raise Error("TypeError: native GBDT adapters require ordinary class metadata")
    var kind = py_get(context["trees_gbdt_adapter_public_classes"], cls)
    if kind is None:
        raise Error("TypeError: native GBDT lifecycle requires a connected adapter class")
    var native = _context(context)
    var fields = _adapter_state(model, native)
    if operation == "trees_public_gbdt_adapter_is_fitted":
        return tree_is_fitted(model, False)
    var fitting = operation.endswith("_fit")
    var method = String("fit")
    if not fitting:
        method = "score" if operation.endswith("_score") else ("decision_function" if operation.endswith("_decision") else ("predict_proba" if operation.endswith("_proba") else "predict"))
        classical_require_fitted(model, False, method, native)
    classical_admit_value(args, native)
    var name = classical_estimator_name(model)
    classical_refuse_sparse(args[0], name)
    # trees_view snapshots native input metadata and its owner once. The tree
    # conversion preserves NaN; classical_check_x's finite-only cast would
    # incorrectly discard the adapter's documented nan_mode behavior.
    var view = trees_view(args[0], "X", native)
    trees_check_rank(view, 2, "X")
    var x = trees_f32_view(view^, "C", native)
    var classifier = String(kind) == "classifier"
    if fitting:
        var params = _read(model, False, native["trees_gbdt_parameter_metadata"], Python.list())
        classical_admit_value(params, native)
        var loss = _resolved(cls, "_loss", PythonObject(None))
        classical_admit_value(loss, native)
        if py_type(loss) is not b.str:
            raise Error("TypeError: native GBDT adapter loss must be an exact string")
        var y = args[1]
        if classifier:
            lda_native_labels(y, native, True, name)
            var shape = labels_shape_of(y, native)
            if len(shape) == 0 or shape[0] != x.rows:
                raise Error("Found input variables with inconsistent numbers of samples")
            # Preserve the outer validator's column-target flattening (lists
            # stay two-dimensional and retain the adapter's single-label refusal).
            if len(shape) == 2 and shape[1] == 1 and py_type(y) is not b.list and py_type(y) is not b.tuple and py_type(y) is not b.memoryview:
                y = _flatten_labels(y, native)
            if args[3] is not None:
                var pair = _adapter_eval(args[3])
                lda_native_labels(pair[1], native)
        else:
            y = classical_check_y(y, x.rows, name, native)
        var saved = py_get(fields, "_mojolearn_fitting", PythonObject(0))
        if py_type(saved) is not b.int:
            raise Error("TypeError: native GBDT fit depth must be an exact integer")
        var depth = Int(py=saved)
        py_setitem(fields, "_mojolearn_fitting", PythonObject(depth + 1))
        var result = PythonObject(None)
        try:
            var inputs = Python.tuple(x.keep, y, args[2], args[3])
            result = adapter_classifier_fit(model, inputs, native) if classifier else adapter_regressor_fit(model, inputs, native)
        finally:
            if depth > 0:
                py_setitem(fields, "_mojolearn_fitting", PythonObject(depth))
            else:
                _ = py_call(fields.pop, "_mojolearn_fitting", PythonObject(None))
        if classifier:
            classifier_after_fit(fields, args[0], native)
        else:
            var names = _feature_names(args[0], native)
            if names is None:
                _ = py_call(fields.pop, "feature_names_in_", PythonObject(None))
            else:
                py_setitem(fields, "feature_names_in_", names)
            py_setitem(fields, "_mojolearn_fitted", PythonObject(True))
        _ = x
        return result
    if py_contains(fields, "n_features_in_"):
        classical_check_width(x.cols, Int(py=fields["n_features_in_"]), name)
    estimator_feature_check(model, fields, args[0], native)
    var result = PythonObject(None)
    if method == "score":
        if classifier:
            lda_native_labels(args[1], native)
        result = gbdt_adapter_score(model, Python.tuple(x.keep, args[1], args[2]), native)
    elif operation.endswith("_decision") or operation.endswith("_regression_predict"):
        result = adapter_decision_function(model, x.keep, native)
    elif operation.endswith("_proba"):
        result = adapter_predict_proba(model, x.keep, native)
    elif operation.endswith("_predict"):
        result = adapter_predict(model, x.keep, native)
    else:
        raise Error("unknown native GBDT adapter operation " + operation)
    _ = x
    return parallel_public_result(result, method, kind, native)
