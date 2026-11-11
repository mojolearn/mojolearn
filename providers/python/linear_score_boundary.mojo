# SPDX-License-Identifier: Apache-2.0
"""Unpack/pin R² inputs and pack the native scalar result."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.admitted_input import admit_x
from bindings.classical_arrays import classical_address
from bindings.classical_validate import classical_check_x_snapshot, classical_estimator_name
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_get, py_type
from bindings.sequential_description import sequential_fields
from core.linear_boundary_types import LINEAR_PREDICT_ABI
from core.linear_model_state import LinearModelState, linear_state_require_coef, linear_state_check_width, linear_state_check_coefficients
from core.linear_score_run import linear_score_run, linear_r2_run, linear_r2_check_target
from providers.python.neural_descriptor_pyapi import NeuralDescriptorProvider, neural_descriptor_connect_factory_once
from x_decomp.descriptor_schema import xd_vendor_code
from x_metrics.descriptor_schema import METRICS_PROGRAM_ABI, MP_OPERATION_COUNT

from providers.python.linear_entry_boundary import linear_entry_connection
from core.linear_entry_table import LINEAR_TABLE_ESTIMATORS, LINEAR_TABLE_METRICS, linear_table_entry


def _linear_metrics_provider(context: PythonObject) raises -> PythonObject:
    var mode = context["backend_fields"]["_DEFAULT_MODE"]
    var metrics = context["classical_metrics"][mode]
    if py_type(metrics) is native_loaded_module("builtins").str:
        raise Error("ImportError: " + String(metrics))
    return neural_descriptor_connect_factory_once(
        metrics["metrics_program_native_api_v1"], String(METRICS_PROGRAM_ABI), MP_OPERATION_COUNT)


def classical_linear_score(model: PythonObject, args: PythonObject,
    entries: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = sequential_fields(model)
    var state = LinearModelState()
    var coefficient_source = py_get(fields, "coef_")
    state.has_coef = coefficient_source is not None
    linear_state_require_coef(state)
    # Capture public state while holding the GIL, before data preparation
    # releases it. A concurrent refit cannot mix old/new scalar owners.
    var feature_source = fields["n_features_in_"]
    var intercept_source = fields["intercept_"]
    var est = classical_estimator_name(model)
    var admitted_x = classical_check_x_snapshot(args[0], est, context, admitted=True)
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
    var target = admit_x(args[1], 1, "y", context)
    var target_rows = Int(py=target.size)
    linear_r2_check_target(rows, target_rows)
    var target_address = Int(py=classical_address(target))
    var x_address = admitted_x.address
    var connection = linear_entry_connection(entries)
    var key = connection.key.copy()
    var predict = linear_table_entry[LINEAR_TABLE_ESTIMATORS](key, 1, String(LINEAR_PREDICT_ABI), 1)
    var metric_mode = context["backend_fields"]["_DEFAULT_MODE"]
    var metric_pins = context["classical_metrics"][metric_mode]
    if py_type(metric_pins) is native_loaded_module("builtins").str:
        raise Error("ImportError: " + String(metric_pins))
    var metric_connection = linear_entry_connection(metric_pins)
    var metric_key = metric_connection.key.copy()
    var metrics = linear_table_entry[LINEAR_TABLE_METRICS](metric_key, 0, String(METRICS_PROGRAM_ABI), MP_OPERATION_COUNT)
    var vendor = xd_vendor_code(String(py_get(context, "classical_metric_vendor", context["vendor"])))
    var mode = 1 if String(context["backend_fields"]["_DEFAULT_MODE"]) == "identical" else 0
    var result = Float64(0.0)
    try:
        with GILReleased(Python()):
            result = linear_score_run(predict, metrics, state, x_address, rows, cols,
                target_address, target_rows, vendor, mode, est)
    finally:
        _ = x
        _ = admitted_x
        _ = coef
        _ = target
        _ = entries
        _ = connection
        _ = metric_pins
        _ = metric_connection
    return PythonObject(result)


def classical_r2_score(prediction: PythonObject, target_source: PythonObject,
    context: PythonObject) raises -> PythonObject:
    var rows = Int(py=prediction.size)
    var target = admit_x(target_source, 1, "y", context)
    var target_rows = Int(py=target.size)
    linear_r2_check_target(rows, target_rows)
    var target_address = Int(py=classical_address(target))
    var prediction_address = Int(py=classical_address(prediction))
    var provider = _linear_metrics_provider(context)
    var entry = provider.downcast_value_ptr[NeuralDescriptorProvider]()[].entry
    var vendor = xd_vendor_code(String(py_get(context, "classical_metric_vendor", context["vendor"])))
    var mode = 1 if String(context["backend_fields"]["_DEFAULT_MODE"]) == "identical" else 0
    var result = Float64(0.0)
    try:
        with GILReleased(Python()):
            result = linear_r2_run(entry, prediction_address, rows, target_address, target_rows, vendor, mode)
    finally:
        _ = prediction
        _ = target
        _ = provider
    return PythonObject(result)
