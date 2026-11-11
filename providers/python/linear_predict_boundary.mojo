# SPDX-License-Identifier: Apache-2.0
"""Python UNPACK/PACK for Ridge and LinearRegression prediction.

Inputs and outputs stay pinned here. RUN receives only buffer addresses,
native dimensions/scalars, and the compiled descriptor endpoint.
"""
from std.python import Python, PythonObject
from std.memory import ArcPointer
from std.python._cpython import GILReleased
from bindings.py_balanced import py_get
from bindings.classical_arrays import classical_array, classical_address
from bindings.classical_validate import classical_check_x_snapshot, classical_estimator_name
from bindings.admitted_input import admit_x
from bindings.sequential_description import sequential_fields
from core.linear_model_state import LinearModelState, linear_state_require_coef, linear_state_check_width, linear_state_check_coefficients
from providers.python.linear_state_boundary import LinearStateOwner, linear_state_owner, linear_state_refresh
from core.linear_registry import LINEAR_STATES, linear_lease_predict
from providers.python.linear_entry_boundary import linear_entry_connection


def _linear_predict_boundary(model: PythonObject, value: PythonObject,
    entries: PythonObject, context: PythonObject, wide: Bool) raises -> PythonObject:
    var fields = sequential_fields(model)
    var owner = linear_state_owner(model)
    var state = LinearModelState()
    var prior = Optional[ArcPointer[LinearModelState]]()
    if owner is not None:
        prior = LINEAR_STATES.acquire(owner.downcast_value_ptr[LinearStateOwner]()[].handle)
        state = prior.value()[].copy()
    var coefficient_source = py_get(fields, "coef_")
    state.has_coef = coefficient_source is not None
    linear_state_require_coef(state)
    # Capture public state while holding the GIL, before data preparation
    # releases it. A concurrent refit cannot mix old/new scalar owners.
    var feature_source = fields["n_features_in_"]
    var intercept_source = fields["intercept_"]
    var est = classical_estimator_name(model)
    var admitted_x = classical_check_x_snapshot(value, est, context, admitted=True)
    var x = admitted_x.array
    var rows = admitted_x.rows
    var cols = admitted_x.cols
    state.features = Int(py=feature_source)
    linear_state_check_width(state, cols, est)
    var coef = admit_x(coefficient_source, 1, "coef_", context)
    state.coef_count = Int(py=coef.size)
    linear_state_check_coefficients(state, cols)
    state.coef = Int(py=classical_address(coef))
    state.intercept = Float64(py=intercept_source)
    var dst = classical_array(rows, 0, context, wide)
    var native_call = linear_state_refresh(model, owner, state, coef, prior)
    var lease = native_call.lease.copy()
    var connection = linear_entry_connection(entries)
    var key = connection.key.copy()
    var destination = Int(py=classical_address(dst))
    var py = Python()
    try:
        with GILReleased(py):
            linear_lease_predict(key, lease, admitted_x.address, destination, rows, cols, wide, est)
    finally:
        # Raw addresses and the copied native function pointer do not retain
        # the admitted arrays or the extension image that owns the endpoint.
        _ = x
        _ = admitted_x
        _ = coef
        _ = dst
        _ = entries
        _ = connection
        _ = owner
        _ = lease
        _ = native_call
        _ = prior
    return dst


def classical_linear_predict(model: PythonObject, value: PythonObject,
    entries: PythonObject, context: PythonObject) raises -> PythonObject:
    return _linear_predict_boundary(model, value, entries, context, False)


def classical_linear_predict_public(model: PythonObject, value: PythonObject,
    entries: PythonObject, context: PythonObject) raises -> PythonObject:
    return _linear_predict_boundary(model, value, entries, context, True)
