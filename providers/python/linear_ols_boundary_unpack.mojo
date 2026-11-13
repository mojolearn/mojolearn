# SPDX-License-Identifier: Apache-2.0
"""Python input/output ownership for native LinearRegression fit."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.classical_arrays_door import classical_array, classical_address
from providers.python.linear_metadata_boundary_door import classical_fit_replacements_admit
from bindings.admitted_input_run import admitted_finite_entry_refuse
from bindings.admitted_input_unpack import admit_x_unrecorded
from bindings.classical_validate_unpack import classical_check_y, classical_check_weights_input, classical_estimator_name
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_get, py_type, py_setitem
from bindings.sequential_description_unpack import sequential_fields
from core.linear_fit_types import LINEAR_FIT_ABI, LINEAR_FIT_OPERATION_COUNT, LinearFitRequest, LinearFitResult, LinearOlsPlan
from core.admitted_refusals import admitted_finite_refuse
from core.host_finite_scan import host_all_finite_f32
from core.linear_ols_run import linear_ols_plan, linear_ols_run
from core.native_descriptor_call import native_descriptor_call
from providers.python.linear_state_boundary_door import linear_state_publish
from providers.python.linear_entry_boundary_door import linear_entry_connection
from core.linear_entry_table import LINEAR_TABLE_ESTIMATORS, LINEAR_TABLE_DECOMP, linear_table_entry
from x_decomp.descriptor_schema import XD_ABI_NAME, XD_FLAGS, XD_OPERATION_COUNT, xd_vendor_code


def classical_ols_fit(model: PythonObject, args: PythonObject,
    entries: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = sequential_fields(model)
    classical_fit_replacements_admit(fields, False, context)
    var intercept = Bool(py=fields["fit_intercept"])
    var est = classical_estimator_name(model)
    # X's finite test is deferred to the one-entry centered TSQR
    # (x_decomp ols_tsqr_factor tests the words it uploads, lane
    # fable-perf-classical-0855); a borrowed X takes no host pass here. The
    # other routes (no intercept, weights, the resident or host OLS) read X
    # on the host, and X is scanned below once the route is known.
    var admitted_x = admit_x_unrecorded(args[0], est, context, admitted=True, defer_finite=True)
    var x = admitted_x.array
    var rows = admitted_x.rows
    var cols = admitted_x.cols
    var y = classical_check_y(args[1], rows, est, context, admitted=True)
    var connection = linear_entry_connection(entries)
    var key = connection.key.copy()
    var entry = linear_table_entry[LINEAR_TABLE_ESTIMATORS](key, 0, String(LINEAR_FIT_ABI), LINEAR_FIT_OPERATION_COUNT)
    var plan = LinearOlsPlan(0, False, 0)
    var py = Python()
    try:
        with GILReleased(py):
            plan = linear_ols_plan(entry, rows, cols, est)
    finally:
        _ = entries
        _ = connection
    var weights = classical_check_weights_input(args[2], rows, context, admitted=True)
    var coef = classical_array(cols, 0, context)
    var mean = classical_array(cols, 0, context)
    var decomp_pins = PythonObject(None)
    var decomp_entry = entry
    if plan.tsqr:
        var decomp = py_get(entries, "decomp")
        if py_type(decomp) is not native_loaded_module("builtins").dict:
            raise Error("ImportError: native LinearRegression TSQR entries unavailable: " + String(decomp))
        decomp_pins = decomp
        var decomp_connection = linear_entry_connection(decomp, LINEAR_TABLE_DECOMP)
        var decomp_key = decomp_connection.key.copy()
        decomp_pins = decomp_connection.images
        if decomp_key.mode != key.mode or decomp_key.vendor != key.vendor:
            raise Error("native LinearRegression TSQR execution key differs")
        decomp_entry = linear_table_entry[LINEAR_TABLE_DECOMP](decomp_key, 0, String(XD_ABI_NAME), XD_OPERATION_COUNT)
    var entry_tests_x = plan.tsqr and intercept and weights is None
    if entry_tests_x:
        try:
            with GILReleased(py):
                entry_tests_x = (native_descriptor_call(decomp_entry, XD_FLAGS, [], [], []) & 1) != 0
        finally:
            _ = entries
            _ = decomp_pins
    if not admitted_x.finite and not entry_tests_x:
        # parallel: finiteness scan across host cores (core/host_finite_scan.mojo)
        if not host_all_finite_f32(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=admitted_x.address), rows * cols):
            admitted_finite_refuse(1, est, "X")
    var request = LinearFitRequest(admitted_x.address, Int(py=classical_address(y)),
        0 if weights is None else Int(py=classical_address(weights)),
        Int(py=classical_address(coef)), Int(py=classical_address(mean)), rows, cols, intercept, 0.0)
    var vendor = xd_vendor_code(String(context["vendor"]))
    var result = LinearFitResult()
    try:
        with GILReleased(py):
            result = linear_ols_run(entry, decomp_entry, request, plan, vendor)
    except error:
        # the entry's own non-finite refusal in scikit-learn's wording
        admitted_finite_entry_refuse(error, est)
        raise error
    finally:
        _ = x
        _ = admitted_x
        _ = y
        _ = weights
        _ = coef
        _ = mean
        _ = entries
        _ = connection
        _ = decomp_pins
    py_setitem(fields, "coef_", coef)
    py_setitem(fields, "intercept_", result.intercept)
    py_setitem(fields, "_x_mean", mean)
    py_setitem(fields, "_y_mean", result.ymean)
    py_setitem(fields, "n_features_in_", cols)
    py_setitem(fields, "input_copied_", not (x is args[0]))
    py_setitem(fields, "_mojolearn_fitted", True)
    linear_state_publish(model, coef, mean, cols, result.intercept, result.ymean, not (x is args[0]))
    return model
