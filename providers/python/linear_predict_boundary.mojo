# SPDX-License-Identifier: Apache-2.0
"""Python UNPACK/PACK for Ridge and LinearRegression prediction.

Inputs and outputs stay pinned here. RUN receives only buffer addresses,
native dimensions/scalars, and the compiled descriptor endpoint.
"""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.py_balanced import py_get
from bindings.classical_arrays import classical_array, classical_address
from bindings.classical_validate import classical_check_x, classical_estimator_name
from bindings.admitted_input import admit_x
from bindings.sequential_description import sequential_fields
from core.linear_boundary_types import LINEAR_PREDICT_ABI
from core.linear_boundary_run import linear_predict_run
from core.linear_model_state import LinearModelState, linear_state_require_coef, linear_state_check_width, linear_state_check_coefficients, linear_state_predict_request
from providers.python.linear_state_boundary import LinearStateOwner, linear_state_owner
from providers.python.neural_descriptor_pyapi import NeuralDescriptorProvider, neural_descriptor_connect_factory_once


def _linear_predict_boundary(model: PythonObject, value: PythonObject,
    entries: PythonObject, context: PythonObject, wide: Bool) raises -> PythonObject:
    var fields = sequential_fields(model)
    var owner = linear_state_owner(model)
    var state = LinearModelState()
    if owner is not None:
        state = owner.downcast_value_ptr[LinearStateOwner]()[].state
    state.has_coef = py_get(fields, "coef_") is not None
    linear_state_require_coef(state)
    var est = classical_estimator_name(model)
    var x = classical_check_x(value, est, context, admitted=True)
    var rows = Int(py=x.shape[0])
    var cols = Int(py=x.shape[1])
    state.features = Int(py=fields["n_features_in_"])
    linear_state_check_width(state, cols, est)
    var coef = admit_x(fields["coef_"], 1, "coef_", context)
    state.coef_count = Int(py=coef.size)
    linear_state_check_coefficients(state, cols)
    state.coef = Int(py=classical_address(coef))
    state.intercept = Float64(py=fields["intercept_"])
    var dst = classical_array(rows, 0, context, wide)
    var request = linear_state_predict_request(state, Int(py=classical_address(x)),
        Int(py=classical_address(dst)), rows, cols, wide, est)
    var provider = neural_descriptor_connect_factory_once(
        entries["linear_predict_native_api_v1"], String(LINEAR_PREDICT_ABI), 1)
    var entry = provider.downcast_value_ptr[NeuralDescriptorProvider]()[].entry
    try:
        with GILReleased(Python()):
            linear_predict_run(entry, request)
    finally:
        # Raw addresses and the copied native function pointer do not retain
        # the admitted arrays or the extension image that owns the endpoint.
        _ = x
        _ = coef
        _ = dst
        _ = provider
        _ = owner
    return dst


def classical_linear_predict(model: PythonObject, value: PythonObject,
    entries: PythonObject, context: PythonObject) raises -> PythonObject:
    return _linear_predict_boundary(model, value, entries, context, False)


def classical_linear_predict_public(model: PythonObject, value: PythonObject,
    entries: PythonObject, context: PythonObject) raises -> PythonObject:
    return _linear_predict_boundary(model, value, entries, context, True)
