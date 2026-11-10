# SPDX-License-Identifier: Apache-2.0
"""Python UNPACK/PACK for typed native Ridge fit.

Only public result arrays are Python-owned. Intermediate arrays, parameter
vectors, route decisions and solver launches belong to linear_ridge_run.
"""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.classical_arrays import classical_array, classical_address
from bindings.classical_metadata import classical_fit_replacements_admit
from providers.python.linear_parameter_boundary import classical_ridge_parameters
from bindings.classical_validate import classical_check_x, classical_check_y, classical_check_weights_input, classical_estimator_name
from bindings.py_balanced import py_setitem
from bindings.sequential_description import sequential_fields
from core.linear_fit_types import LINEAR_FIT_ABI, LINEAR_FIT_OPERATION_COUNT, RIDGE_FIT_ABI, LinearFitRequest, LinearFitResult
from core.linear_fit_run import linear_ridge_run
from providers.python.linear_state_boundary import linear_state_publish
from providers.python.neural_descriptor_pyapi import NeuralDescriptorProvider, neural_descriptor_connect_factory_once


def classical_ridge_fit(model: PythonObject, args: PythonObject,
    entries: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = sequential_fields(model)
    classical_fit_replacements_admit(fields, True, context)
    var est = classical_estimator_name(model)
    var x = classical_check_x(args[0], est, context, admitted=True)
    var rows = Int(py=x.shape[0])
    var cols = Int(py=x.shape[1])
    var y = classical_check_y(args[1], rows, est, context, admitted=True)
    classical_ridge_parameters(fields)
    var weights = classical_check_weights_input(args[2], rows, context, admitted=True)
    var coef = classical_array(cols, 0, context)
    var mean = classical_array(cols, 0, context)
    var request = LinearFitRequest(Int(py=classical_address(x)), Int(py=classical_address(y)),
        0 if weights is None else Int(py=classical_address(weights)),
        Int(py=classical_address(coef)), Int(py=classical_address(mean)), rows, cols,
        Bool(py=fields["fit_intercept"]), Float64(py=fields["alpha"]))
    var provider = neural_descriptor_connect_factory_once(
        entries["linear_fit_native_api_v1"], String(LINEAR_FIT_ABI), LINEAR_FIT_OPERATION_COUNT)
    var weighted_provider = neural_descriptor_connect_factory_once(
        entries["ridge_fit_native_api_v1"], String(RIDGE_FIT_ABI), 1)
    var entry = provider.downcast_value_ptr[NeuralDescriptorProvider]()[].entry
    var weighted_entry = weighted_provider.downcast_value_ptr[NeuralDescriptorProvider]()[].entry
    var result = LinearFitResult()
    try:
        with GILReleased(Python()):
            result = linear_ridge_run(entry, weighted_entry, request)
    finally:
        _ = x
        _ = y
        _ = weights
        _ = coef
        _ = mean
        _ = provider
        _ = weighted_provider
    py_setitem(fields, "coef_", coef)
    py_setitem(fields, "intercept_", result.intercept)
    py_setitem(fields, "_x_mean", mean)
    py_setitem(fields, "_y_mean", result.ymean)
    py_setitem(fields, "n_features_in_", cols)
    py_setitem(fields, "solver_", "eig")
    py_setitem(fields, "input_copied_", not (x is args[0]))
    py_setitem(fields, "_mojolearn_fitted", True)
    linear_state_publish(model, coef, mean, cols, result.intercept, result.ymean, not (x is args[0]))
    return model
