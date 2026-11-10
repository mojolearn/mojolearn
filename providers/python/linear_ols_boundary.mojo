# SPDX-License-Identifier: Apache-2.0
"""Python input/output ownership for native LinearRegression fit."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.classical_arrays import classical_array, classical_address
from bindings.classical_metadata import classical_fit_replacements_admit
from bindings.classical_validate import classical_check_x, classical_check_y, classical_check_weights_input, classical_estimator_name
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_get, py_type, py_setitem
from bindings.sequential_description import sequential_fields
from core.linear_fit_types import LINEAR_FIT_ABI, LINEAR_FIT_OPERATION_COUNT, LinearFitRequest, LinearFitResult, LinearOlsPlan
from core.linear_ols_run import linear_ols_plan, linear_ols_run
from providers.python.linear_state_boundary import linear_state_publish
from providers.python.neural_descriptor_pyapi import NeuralDescriptorProvider, neural_descriptor_connect_factory_once
from x_decomp.descriptor_schema import XD_ABI_NAME, XD_OPERATION_COUNT, xd_vendor_code


def classical_ols_fit(model: PythonObject, args: PythonObject,
    entries: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = sequential_fields(model)
    classical_fit_replacements_admit(fields, False, context)
    var intercept = Bool(py=fields["fit_intercept"])
    var est = classical_estimator_name(model)
    var x = classical_check_x(args[0], est, context, admitted=True)
    var rows = Int(py=x.shape[0])
    var cols = Int(py=x.shape[1])
    var y = classical_check_y(args[1], rows, est, context, admitted=True)
    var provider = neural_descriptor_connect_factory_once(
        entries["linear_fit_native_api_v1"], String(LINEAR_FIT_ABI), LINEAR_FIT_OPERATION_COUNT)
    var entry = provider.downcast_value_ptr[NeuralDescriptorProvider]()[].entry
    var plan = LinearOlsPlan(0, False, 0)
    try:
        with GILReleased(Python()):
            plan = linear_ols_plan(entry, rows, cols, est)
    finally:
        _ = provider
    var weights = classical_check_weights_input(args[2], rows, context, admitted=True)
    var coef = classical_array(cols, 0, context)
    var mean = classical_array(cols, 0, context)
    var decomp_provider = PythonObject(None)
    var decomp_entry = entry
    if plan.tsqr:
        var decomp = py_get(entries, "decomp")
        if py_type(decomp) is not native_loaded_module("builtins").dict:
            raise Error("ImportError: native LinearRegression TSQR entries unavailable: " + String(decomp))
        decomp_provider = neural_descriptor_connect_factory_once(
            decomp["x_decomp_native_api_v1"], String(XD_ABI_NAME), XD_OPERATION_COUNT)
        decomp_entry = decomp_provider.downcast_value_ptr[NeuralDescriptorProvider]()[].entry
    var request = LinearFitRequest(Int(py=classical_address(x)), Int(py=classical_address(y)),
        0 if weights is None else Int(py=classical_address(weights)),
        Int(py=classical_address(coef)), Int(py=classical_address(mean)), rows, cols, intercept, 0.0)
    var vendor = xd_vendor_code(String(context["vendor"]))
    var result = LinearFitResult()
    try:
        with GILReleased(Python()):
            result = linear_ols_run(entry, decomp_entry, request, plan, vendor)
    finally:
        _ = x
        _ = y
        _ = weights
        _ = coef
        _ = mean
        _ = provider
        _ = decomp_provider
    py_setitem(fields, "coef_", coef)
    py_setitem(fields, "intercept_", result.intercept)
    py_setitem(fields, "_x_mean", mean)
    py_setitem(fields, "_y_mean", result.ymean)
    py_setitem(fields, "n_features_in_", cols)
    py_setitem(fields, "input_copied_", not (x is args[0]))
    py_setitem(fields, "_mojolearn_fitted", True)
    linear_state_publish(model, coef, mean, cols, result.intercept, result.ymean, not (x is args[0]))
    return model
