# SPDX-License-Identifier: Apache-2.0
"""Python UNPACK/PACK for typed native Ridge fit.

Only public result arrays are Python-owned. Intermediate arrays, parameter
vectors, route decisions and solver launches belong to linear_ridge_run.
"""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.classical_arrays import classical_array, classical_address
from providers.python.linear_metadata_boundary import classical_fit_replacements_admit
from providers.python.linear_parameter_boundary import linear_ridge_parameters_values
from bindings.classical_validate import classical_check_x_snapshot, classical_check_y, classical_check_weights_input, classical_estimator_name
from bindings.py_balanced import py_setitem
from bindings.sequential_description import sequential_fields
from core.linear_fit_types import LINEAR_FIT_ABI, LINEAR_FIT_OPERATION_COUNT, RIDGE_FIT_ABI, LinearFitRequest, LinearFitResult
from core.linear_fit_run import linear_ridge_run
from providers.python.linear_state_boundary import linear_state_publish
from providers.python.linear_entry_boundary import linear_entry_connection
from core.linear_entry_table import LINEAR_TABLE_ESTIMATORS, LINEAR_TABLE_X_LINEAR, linear_table_entry


def classical_ridge_fit(model: PythonObject, args: PythonObject,
    entries: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = sequential_fields(model)
    classical_fit_replacements_admit(fields, True, context)
    var alpha_source = fields["alpha"]
    var solver_source = fields["solver"]
    var normalize_source = fields["normalize"]
    var intercept_source = fields["fit_intercept"]
    var est = classical_estimator_name(model)
    var admitted_x = classical_check_x_snapshot(args[0], est, context, admitted=True)
    var x = admitted_x.array
    var rows = admitted_x.rows
    var cols = admitted_x.cols
    var y = classical_check_y(args[1], rows, est, context, admitted=True)
    var alpha = linear_ridge_parameters_values(alpha_source, solver_source, normalize_source)
    var intercept = Bool(py=intercept_source)
    var weights = classical_check_weights_input(args[2], rows, context, admitted=True)
    var coef = classical_array(cols, 0, context)
    var mean = classical_array(cols, 0, context)
    var request = LinearFitRequest(admitted_x.address, Int(py=classical_address(y)),
        0 if weights is None else Int(py=classical_address(weights)),
        Int(py=classical_address(coef)), Int(py=classical_address(mean)), rows, cols,
        intercept, alpha)
    var connection = linear_entry_connection(entries)
    var key = connection.key.copy()
    var entry = linear_table_entry[LINEAR_TABLE_ESTIMATORS](key, 0, String(LINEAR_FIT_ABI), LINEAR_FIT_OPERATION_COUNT)
    var weighted_connection = linear_entry_connection(entries, LINEAR_TABLE_X_LINEAR, True)
    if weighted_connection.key.mode != key.mode or weighted_connection.key.vendor != key.vendor:
        raise Error("native Ridge weighted execution key differs")
    var weighted_entry = linear_table_entry[LINEAR_TABLE_X_LINEAR](weighted_connection.key, 0, String(RIDGE_FIT_ABI), 1)
    var result = LinearFitResult()
    try:
        with GILReleased(Python()):
            result = linear_ridge_run(entry, weighted_entry, request)
    finally:
        _ = x
        _ = admitted_x
        _ = y
        _ = weights
        _ = coef
        _ = mean
        _ = entries
        _ = connection
        _ = weighted_connection
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
