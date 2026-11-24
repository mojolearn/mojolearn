# SPDX-License-Identifier: Apache-2.0
"""Python UNPACK/PACK for typed native Ridge fit.

Only public result arrays are Python-owned. Intermediate arrays, parameter
vectors, route decisions and solver launches belong to linear_ridge_run.
"""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.classical_arrays_door import classical_array, classical_address
from providers.python.linear_metadata_boundary_door import classical_fit_replacements_admit
from providers.python.linear_parameter_boundary_unpack import linear_ridge_parameters_values
from bindings.classical_validate_unpack import classical_check_x_snapshot, classical_check_y, classical_check_weights_input, classical_estimator_name
from bindings.py_balanced import py_setitem
from bindings.sequential_description_unpack import sequential_fields
from core.linear_fit_types import LinearFitRequest, LinearFitResult
from core.linear_fit_run import linear_ridge_run, linear_ridge_targets_run
from bindings.admitted_input_unpack import AdmittedX, admit_f32, classical_view
from providers.python.linear_state_boundary_door import linear_state_publish
from core.device_route import DeviceRoute


def classical_ridge_fit(model: PythonObject, args: PythonObject,
    route: DeviceRoute, pins: PythonObject, context: PythonObject) raises -> PythonObject:
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
    # THE 2-D y (lane w3f3-k): scikit-learn's Ridge fits every column of a
    # rows x targets y; one column keeps the one-target fit (raveled).
    if args[1] is not None:
        var y_view = classical_view(args[1], context, "y", True)
        if Int(py=y_view.ndim) == 2 and Int(py=y_view.shape[1]) > 1:
            _ = y_view
            return _ridge_fit_targets(model, args, route, pins, context, fields, est, admitted_x, intercept_source,
                                      alpha_source, solver_source, normalize_source)
        _ = y_view
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
    # The estimator's resolved entries (core/linear_route.mojo): the device
    # table's or the CPU table's, read once per estimator. This fit admits no
    # connection owner and reads no table slot.
    if not route.has_third:
        raise Error("native Ridge weighted execution key differs")
    var entry = route.entry.call
    var weighted_entry = route.third.call
    var result = LinearFitResult()
    var py = Python()
    try:
        with GILReleased(py):
            result = linear_ridge_run(entry, weighted_entry, request)
    finally:
        _ = x
        _ = admitted_x
        _ = y
        _ = weights
        _ = coef
        _ = mean
        _ = pins
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


def _admit_targets(source: PythonObject, est: String, context: PythonObject) raises -> AdmittedX:
    """A rows x targets y as float32 words, with the one-target check's NaN
    words (bindings/admitted_input_unpack.mojo `admit_y`)."""
    try:
        var admitted = admit_f32(source, 2, "y", context, admitted=True)
        return admitted^
    except error:
        if String(error) == "Input y contains NaN or infinity":
            raise Error("Input y contains NaN or infinity (" + est + ").")
        raise error^


def _ridge_fit_targets(model: PythonObject, args: PythonObject, route: DeviceRoute, pins: PythonObject,
    context: PythonObject, fields: PythonObject, est: String, admitted_x: AdmittedX, intercept_source: PythonObject,
    alpha_source: PythonObject, solver_source: PythonObject, normalize_source: PythonObject) raises -> PythonObject:
    """`Ridge.fit(X, Y)` with a rows x targets Y (lane w3f3-k), as
    scikit-learn's: `coef_` targets x features, `intercept_` one float64 per
    target, predictions rows x targets. RUN: `linear_ridge_targets_run`, one
    factorization for every target. Y is admitted once (real, finite, as
    long as X: the target check's words)."""
    var x = admitted_x.array
    var rows = admitted_x.rows
    var cols = admitted_x.cols
    var admitted_y = _admit_targets(args[1], est, context)
    if admitted_y.rows != rows:
        raise Error("Found input variables with inconsistent numbers of samples: [" + String(rows) + ", "
                    + String(admitted_y.rows) + "]")
    var targets = admitted_y.cols
    var y = admitted_y.array
    var alpha = linear_ridge_parameters_values(alpha_source, solver_source, normalize_source)
    var intercept = Bool(py=intercept_source)
    var weights = classical_check_weights_input(args[2], rows, context, admitted=True)
    var coef = classical_array(targets, cols, context)
    var mean = classical_array(cols, 0, context)
    var intercepts = classical_array(targets, 0, context, True)
    var request = LinearFitRequest(admitted_x.address, admitted_y.address,
        0 if weights is None else Int(py=classical_address(weights)),
        Int(py=classical_address(coef)), Int(py=classical_address(mean)), rows, cols, intercept, alpha)
    if not route.has_third:
        raise Error("native Ridge weighted execution key differs")
    var weighted_entry = route.third.call
    var intercepts_address = Int(py=classical_address(intercepts))
    var py = Python()
    try:
        with GILReleased(py):
            linear_ridge_targets_run(weighted_entry, request, targets, intercepts_address)
    finally:
        _ = x
        _ = admitted_y
        _ = y
        _ = weights
        _ = coef
        _ = mean
        _ = intercepts
        _ = pins
    py_setitem(fields, "coef_", coef)
    py_setitem(fields, "intercept_", intercepts)
    py_setitem(fields, "_x_mean", mean)
    py_setitem(fields, "_y_mean", 0.0)
    py_setitem(fields, "n_features_in_", cols)
    py_setitem(fields, "solver_", "eig")
    py_setitem(fields, "input_copied_", not (x is args[0]))
    py_setitem(fields, "_mojolearn_fitted", True)
    linear_state_publish(model, coef, mean, cols, 0.0, 0.0, not (x is args[0]), targets)
    return model
