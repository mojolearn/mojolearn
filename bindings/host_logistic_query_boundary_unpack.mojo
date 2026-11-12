# SPDX-License-Identifier: Apache-2.0
"""One host Logistic admission shared by public inference and score."""
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_contains
from bindings.classical_pandas import classical_admit_value
from bindings.classical_exceptions_door import classical_require_fitted
from bindings.classical_caller_unpack import classical_caller_outside
from bindings.host_inference_scalars_door import host_inference_scalar
from bindings.host_inference_state import host_inference_state
from bindings.host_logistic_native import host_logistic_classes, host_logistic_inference


from bindings.host_logistic_query_boundary_pack import (
    host_logistic_operation,
)


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
