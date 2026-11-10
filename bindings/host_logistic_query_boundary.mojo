# SPDX-License-Identifier: Apache-2.0
"""One host Logistic admission shared by public inference and score."""
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_contains
from bindings.classical_pandas import classical_admit_value
from bindings.classical_exceptions import classical_require_fitted
from bindings.classical_caller import classical_caller_outside
from bindings.host_inference_scalars import host_inference_scalar
from bindings.host_inference_state import host_inference_state
from bindings.host_logistic_native import host_logistic_classes, host_logistic_inference


def host_logistic_operation(model: PythonObject, fields: PythonObject, operation: String,
    source: PythonObject, context: PythonObject, error_context: PythonObject,
    public: Bool) raises -> PythonObject:
    # Same field/class-label admission and refusal order as the former
    # Logistic branch of host_inference_operation. No Python method is called.
    var labels = PythonObject(None)
    if py_contains(fields, "classes_"):
        labels = host_logistic_classes(fields["classes_"], context)
    var b = native_loaded_module("builtins")
    var keys = py_call(b.list, fields.keys())
    for i in range(len(keys)):  # driver-loop: admitted public model fields
        if String(keys[i]) != "classes_":
            classical_admit_value(fields[keys[i]], context)
    host_inference_scalar(py_get(fields, "_mojolearn_fitted", False), context, True)
    host_inference_scalar(py_get(fields, "_mojolearn_fitting", False), context, True)
    if py_contains(fields, "n_features_in_"):
        host_inference_scalar(fields["n_features_in_"], context)
    if py_contains(fields, "fit_intercept"):
        host_inference_scalar(fields["fit_intercept"], context, True)
    var method = String("inverse_transform") if operation == "host_infer_inverse_transform" else (String("transform") if operation == "host_infer_transform" else String("predict"))
    if operation == "host_infer_decision_function":
        method = "decision_function"
    elif operation == "host_infer_predict_proba":
        method = "predict_proba"
    elif operation == "host_infer_predict_log_proba":
        method = "predict_log_proba"
    elif operation == "host_infer_score_samples":
        method = "score_samples"
    elif operation == "host_infer_score":
        method = "score"
    classical_require_fitted(model, False, method, error_context)
    var mode = py_get(fields, "numeric_mode")
    if mode is not None and (py_type(mode) is not b.str or String(mode) != "identical"):
        raise Error("explicit host models run identical mode only")
    return host_logistic_inference(model, fields, labels, operation, source, context, public)


def host_logistic_score_predict(model: PythonObject, source: PythonObject,
                                error_context: PythonObject) raises -> PythonObject:
    var context = error_context["host_inference_context"]
    # Keep the caller-frame/output-format rule: forwarding between compiled
    # entries never added a Python frame, so direct reuse has the same depth.
    var public = classical_caller_outside(error_context)
    var state = host_inference_state(model, context)
    if state[1] != "logistic":
        raise Error("TypeError: LogisticRegression score requires its connected native host class")
    return host_logistic_operation(model, state[0], "host_infer_predict", source,
        context, error_context, public)
