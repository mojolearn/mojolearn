# SPDX-License-Identifier: Apache-2.0
"""Unpack/pin R² inputs and pack the native scalar result."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.admitted_input_unpack import AdmittedX, admit_x, admit_f32
from core.linear_boundary_run import linear_predict_targets_run, linear_r2_targets
from providers.python.linear_predict_boundary_unpack import linear_intercepts_unpack
from bindings.classical_arrays_door import classical_address
from bindings.classical_validate_unpack import classical_check_x_snapshot, classical_estimator_name
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_get, py_type
from bindings.sequential_description_unpack import sequential_fields
from core.device_route import DeviceRoute
from core.linear_model_state import LinearModelState, linear_state_require_coef, linear_state_check_width, linear_state_check_coefficients
from core.linear_score_run import linear_score_run, linear_r2_run, linear_r2_check_target
from providers.python.neural_descriptor_pyapi_door import NeuralDescriptorProvider, neural_descriptor_connect_factory_once
from x_decomp.descriptor_schema import xd_vendor_code
from x_metrics.descriptor_schema import METRICS_PROGRAM_ABI, MP_OPERATION_COUNT

from providers.python.linear_entry_boundary_door import linear_entry_connection
from core.linear_entry_table import LINEAR_TABLE_METRICS, linear_table_entry


def _linear_metrics_provider(context: PythonObject) raises -> PythonObject:
    var mode = context["backend_fields"]["_DEFAULT_MODE"]
    var metrics = context["classical_metrics"][mode]
    if py_type(metrics) is native_loaded_module("builtins").str:
        raise Error("ImportError: " + String(metrics))
    return neural_descriptor_connect_factory_once(
        metrics["metrics_program_native_api_v1"], String(METRICS_PROGRAM_ABI), MP_OPERATION_COUNT)


def classical_linear_score(model: PythonObject, args: PythonObject,
    route: DeviceRoute, pins: PythonObject, context: PythonObject) raises -> PythonObject:
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
    # THE 2-D y (lane w3f3-k): a Ridge fitted on several targets (its 2-D
    # coef_, which the one-target admission refuses) scores scikit-learn's
    # r2_score(y, predict(X)) averaged over the targets. Decided on that
    # refusal only: the one-target score reads nothing more.
    var coef = PythonObject(None)
    try:
        coef = admit_x(coefficient_source, 1, "coef_", context)
    except error:
        if _two_d(coefficient_source):
            _ = x
            return _score_targets(model, fields, args, route, pins, context, admitted_x, est,
                                  Int(py=coefficient_source.shape[0]))
        raise error^
    state.coef_count = Int(py=coef.size)
    linear_state_check_coefficients(state, cols)
    state.coef = Int(py=classical_address(coef))
    state.intercept = Float64(py=intercept_source)
    var target = admit_x(args[1], 1, "y", context)
    var target_rows = Int(py=target.size)
    linear_r2_check_target(rows, target_rows)
    var target_address = Int(py=classical_address(target))
    var x_address = admitted_x.address
    # The estimator's resolved prediction entry (device table or CPU table).
    var predict = route.second.call
    # A CPU route resolved in a GPU installation holds its own CPU identical
    # metrics entry (providers/python/metrics_route_boundary_door.mojo): the
    # process's metrics table is the GPU one and is not read for it.
    var metrics = route.fifth.call
    var vendor = 0
    var mode = 1
    var metric_pins = PythonObject(None)
    var metric_owner = PythonObject(None)
    if not route.has_fifth:
        var metric_mode = context["backend_fields"]["_DEFAULT_MODE"]
        metric_pins = context["classical_metrics"][metric_mode]
        if py_type(metric_pins) is native_loaded_module("builtins").str:
            raise Error("ImportError: " + String(metric_pins))
        var metric_connection = linear_entry_connection(metric_pins, LINEAR_TABLE_METRICS)
        var metric_key = metric_connection.key.copy()
        metric_owner = metric_connection.images
        vendor = xd_vendor_code(String(py_get(context, "classical_metric_vendor", context["vendor"])))
        mode = 1 if String(context["backend_fields"]["_DEFAULT_MODE"]) == "identical" else 0
        # A canonical metrics owner can still belong to another selected route.
        # The explicit Host* classes may intentionally use the context's GPU
        # metrics, so compare against metric selection, not the estimator
        # connection key.
        if metric_key.vendor != vendor:
            raise Error("RuntimeError: native linear score metrics vendor differs from selected execution")
        if metric_key.mode != mode:
            raise Error("RuntimeError: native linear score metrics mode differs from selected execution")
        metrics = linear_table_entry[LINEAR_TABLE_METRICS](metric_key, 0, String(METRICS_PROGRAM_ABI), MP_OPERATION_COUNT)
    var result = Float64(0.0)
    var py = Python()
    try:
        with GILReleased(py):
            result = linear_score_run(predict, metrics, state, x_address, rows, cols,
                target_address, target_rows, vendor, mode, est)
    finally:
        _ = x
        _ = admitted_x
        _ = coef
        _ = target
        _ = pins
        _ = metric_pins
        _ = metric_owner
    return PythonObject(result)


def _two_d(value: PythonObject) -> Bool:
    """An array-like `coef_` of rank 2 (False for anything else)."""
    try:
        return Int(py=value.ndim) == 2
    except:
        return False


def _score_targets(model: PythonObject, fields: PythonObject, args: PythonObject, route: DeviceRoute,
    pins: PythonObject, context: PythonObject, admitted_x: AdmittedX, est: String, targets: Int) raises -> PythonObject:
    """`Ridge.score(X, Y)` of a multi-target fit (lane w3f3-k): the
    predictions (`linear_predict_targets_run`, as `predict`) against the rows
    x targets Y, `r2_score`'s uniform average (`linear_r2_targets`)."""
    var rows = admitted_x.rows
    var cols = admitted_x.cols
    var coef = admit_x(py_get(fields, "coef_"), 2, "coef_", context)
    var state = LinearModelState()
    state.coef_count = Int(py=coef.size)
    linear_state_check_coefficients(state, cols * targets)
    var intercepts = linear_intercepts_unpack(fields["intercept_"], targets, context)
    var target = admit_f32(args[1], 2, "y", context)
    linear_r2_check_target(rows, target.rows)
    if target.cols != targets:
        raise Error("ValueError: y has " + String(target.cols) + " targets, but " + est + " was fitted on "
                    + String(targets))
    var predictions = List[Float32](length=rows * targets, fill=Float32(0.0))
    var coef_address = Int(py=classical_address(coef))
    var predictions_address = Int(predictions.unsafe_ptr())
    var target_address = target.address
    var x_address = admitted_x.address
    var entry = route.second.call
    var result = Float64(0.0)
    var py = Python()
    try:
        with GILReleased(py):
            _ = linear_predict_targets_run(entry, x_address, coef_address, intercepts, predictions_address,
                                           rows, cols, targets)
            result = linear_r2_targets(target_address, predictions_address, rows, targets)
    finally:
        _ = admitted_x
        _ = coef
        _ = target
        _ = predictions
        _ = intercepts
        _ = pins
    return PythonObject(result)
