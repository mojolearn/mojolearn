# SPDX-License-Identifier: Apache-2.0
# Native public boundary for Voting, Stacking, MultiOutput, OneVsRest and
# CalibratedClassifierCV. Internal trees_meta_* calls keep Array results;
# public doors preserve validation, fitted metadata and output conversion.
# 2026-10-25: Codex cybersecurity screening halted the delegated repair turn
# and directed it to Daybreak access. This interruption did not establish a
# Mojo defect: the observed problems were our inherited Python decorators
# and ClassLabels handling. The root agent continued the supported API repair.
# See docs/plans/NATIVE_SCREENING_AND_FALLBACKS_20261025.md. Andrew authorizes
# documented Python fallbacks if needed; this implementation introduces none
# and still requires integrated compilation and public-runtime validation.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_fields
from bindings.classical_validate import classical_check_x, classical_estimator_name
from bindings.classical_labels import labels_shape_of
from bindings.classical_exceptions import classical_require_fitted
from bindings.classifier_public_lifecycle import classifier_after_fit, _feature_names
from bindings.trees_public_lifecycle_native import _feature_check
from bindings.trees_public_meta import meta_kind, meta_operation
from bindings.prep_native_estimator import prep_estimator_admit, prep_estimator_call, prep_native_context
from bindings.msel_inputs import msel_admit, msel_input_count
from bindings.parallel_public_result import parallel_public_result


def meta_public_operation(model: PythonObject, operation: String, args: PythonObject,
                          context: PythonObject) raises -> PythonObject:
    # Admit the exact registered model and all argument objects before any
    # shape/label read. Unsupported Python input hooks must never be invoked.
    prep_estimator_admit(model, "native meta-estimator", context)
    _ = prep_estimator_call(model, "__admit__", Python.tuple(), context)
    var native = prep_native_context(context)
    msel_admit(args, native)
    var method = String("")
    if operation == "trees_public_meta_fit":
        method = "fit"
    elif operation == "trees_public_meta_fit_transform":
        method = "fit_transform"
    elif operation == "trees_public_meta_predict":
        method = "predict"
    elif operation == "trees_public_meta_predict_proba":
        method = "predict_proba"
    elif operation == "trees_public_meta_transform":
        method = "transform"
    elif operation == "trees_public_meta_score":
        method = "score"
    else:
        raise Error("unknown public meta-estimator operation " + operation)
    var fitting = method == "fit" or method == "fit_transform"
    if not fitting:
        classical_require_fitted(model, False, method, context)
    var name = classical_estimator_name(model)
    var x = classical_check_x(args[0], name, context)
    var fields = py_fields(model)
    var kind = meta_kind(model, context)
    if fitting:
        if args[1] is None:
            raise Error(name + " requires y to be passed, but the target y is None.")
        var shape = labels_shape_of(x, context)
        if msel_input_count(args[1], native) != shape[0]:
            raise Error("Found input variables with inconsistent numbers of samples")
    else:
        _feature_check(model, fields, args[0], context)
    var call_args = Python.tuple(x, args[1], args[2]) if fitting or method == "score" else Python.tuple(x)
    var result = PythonObject(None)
    if fitting:
        var b = native_loaded_module("builtins")
        var saved = py_get(fields, "_mojolearn_fitting", PythonObject(0))
        if py_type(saved) is not b.int:
            raise Error("TypeError: native meta-estimator fit depth must be an exact integer")
        var depth = Int(py=saved)
        py_setitem(fields, "_mojolearn_fitting", PythonObject(depth + 1))
        try:
            result = meta_operation(model, "trees_meta_" + method, call_args, context)
        finally:
            if depth:
                py_setitem(fields, "_mojolearn_fitting", PythonObject(depth))
            else:
                _ = py_call(fields.pop, "_mojolearn_fitting", PythonObject(None))
        if not kind.endswith("_regressor") and kind != "multioutput_classifier":
            classifier_after_fit(fields, args[0], context)
        else:
            # MultiOutputClassifier.classes_ is a list of per-output class
            # containers, not one flat ClassLabels instance.
            var names = _feature_names(args[0], context)
            if names is None:
                _ = py_call(fields.pop, "feature_names_in_", PythonObject(None))
            else:
                py_setitem(fields, "feature_names_in_", names)
            py_setitem(fields, "_mojolearn_fitted", PythonObject(True))
        if method == "fit":
            return result
    else:
        result = meta_operation(model, "trees_meta_" + method, call_args, context)
    var estimator_type = PythonObject("regressor" if kind.endswith("_regressor") else "classifier")
    return parallel_public_result(result, method, estimator_type, context)
