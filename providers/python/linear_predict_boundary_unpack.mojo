# SPDX-License-Identifier: Apache-2.0
"""Python UNPACK/PACK for Ridge and LinearRegression prediction.

Inputs and outputs stay pinned here. RUN receives only buffer addresses,
native dimensions/scalars, and the compiled descriptor endpoint.
"""
from std.python import Python, PythonObject
from std.memory import ArcPointer
from std.python._cpython import GILReleased
from bindings.py_balanced import py_get
from bindings.classical_arrays_door import classical_array, classical_address
from bindings.classical_validate_unpack import classical_estimator_name
from bindings.admitted_input_unpack import admit_x, admit_x_unrecorded, admit_c_view, admitted_dtype
from core.admitted_refusals import admitted_finite_refuse
from core.host_finite_scan import host_all_finite_f32
from bindings.sequential_description_unpack import sequential_fields
from core.linear_model_state import LinearModelState, linear_state_require_coef, linear_state_check_width, linear_state_check_coefficients
from providers.python.linear_state_boundary_door import LinearStateOwner, linear_state_owner, linear_state_refresh
from core.linear_registry import LINEAR_STATES, linear_lease_predict
from core.linear_boundary_run import linear_predict_targets_run
from core.device_route import DeviceRoute


def _linear_predict_boundary(model: PythonObject, value: PythonObject,
    route: DeviceRoute, pins: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = sequential_fields(model)
    var owner = linear_state_owner(model)
    var state = LinearModelState()
    var prior = Optional[ArcPointer[LinearModelState]]()
    if owner is not None:
        prior = LINEAR_STATES.acquire(owner.downcast_value_ptr[LinearStateOwner]()[].handle)
        state = prior.value()[].copy()
    if state.targets > 1:
        # THE 2-D y (lane w3f3-k): a Ridge fitted on several targets. One
        # integer compare on the one-target path, which is unchanged.
        _ = prior
        return linear_predict_targets_boundary(model, fields, value, route, pins, context, state.targets)
    var coefficient_source = py_get(fields, "coef_")
    state.has_coef = coefficient_source is not None
    linear_state_require_coef(state)
    # Capture public state while holding the GIL, before data preparation
    # releases it. A concurrent refit cannot mix old/new scalar owners.
    var feature_source = fields["n_features_in_"]
    var intercept_source = fields["intercept_"]
    var est = classical_estimator_name(model)
    # X's finite test is deferred to the prediction entry (lane
    # ols-regression, 2026-11-14): a borrowed float32 C-order X takes no host
    # pass here; the entry tests the words it uploads and returns the status
    # word at its one drain (core/p3_linear_impl.mojo ols_predict_device; the
    # host column scans in its entry). A cast X was tested in the cast pass
    # (`finite` True) and is not tested again. 0.8.55's predict scanned
    # nothing; the host scan here was the added pass (ols taxi infer L40S
    # 5.45 -> 9.30 ms against 0.8.55, paper board 80a7f8aca3).
    var admitted_x = admit_x_unrecorded(value, est, context, admitted=True, defer_finite=True)
    var x = admitted_x.array
    var rows = admitted_x.rows
    var cols = admitted_x.cols
    var test_x = not admitted_x.finite
    state.features = Int(py=feature_source)
    if test_x and cols != state.features:
        # scikit-learn refuses a non-finite X before the feature-count check;
        # only a refused call pays this scan.
        if not host_all_finite_f32(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=admitted_x.address),
                                   rows * cols):
            admitted_finite_refuse(1, est, "X")
    linear_state_check_width(state, cols, est)
    var coef = admit_x(coefficient_source, 1, "coef_", context)
    state.coef_count = Int(py=coef.size)
    linear_state_check_coefficients(state, cols)
    state.coef = Int(py=classical_address(coef))
    state.intercept = Float64(py=intercept_source)
    var dst = classical_array(rows, 0, context)
    var native_call = linear_state_refresh(model, owner, state, coef, prior)
    var lease = native_call.lease.copy()
    # The estimator's resolved prediction entry (device table or CPU table):
    # no connection admission and no table slot read per call.
    var entry = route.second.call
    var destination = Int(py=classical_address(dst))
    var py = Python()
    var status = 0
    try:
        with GILReleased(py):
            status = linear_lease_predict(entry, lease, admitted_x.address, destination, rows, cols, est, test_x)
    finally:
        # Raw addresses and the copied native function pointer do not retain
        # the admitted arrays or the extension image that owns the endpoint.
        _ = x
        _ = admitted_x
        _ = coef
        _ = dst
        _ = pins
        _ = owner
        _ = lease
        _ = native_call
        _ = prior
    admitted_finite_refuse(status, est, "X")
    return dst


def classical_linear_predict(model: PythonObject, value: PythonObject,
    route: DeviceRoute, pins: PythonObject, context: PythonObject) raises -> PythonObject:
    # One result for every caller (Andrew, 2026-11-22, lane predict-dtype):
    # the float32 predictions, public predict included.
    return _linear_predict_boundary(model, value, route, pins, context)


def linear_intercepts_unpack(source: PythonObject, targets: Int, context: PythonObject) raises -> List[Float64]:
    """UNPACK (lane w3f3-k): the public `intercept_` of a multi-target Ridge
    as `targets` float64 words, one admission of its array (a float64 view's
    words copied natively; another dtype is admitted as float32 words)."""
    var view = admit_c_view(source, "intercept_", context)
    if Int(py=view.size) != targets:
        raise Error("ValueError: intercept_ holds " + String(Int(py=view.size)) + " values, but this Ridge has "
                    + String(targets) + " targets")
    var words = List[Float64](length=targets, fill=0.0)
    if admitted_dtype(view, context) == "<f8":
        var src = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(view)))
        for t in range(targets):  # small-loop(the targets' intercepts)
            words[t] = src[t]
    else:
        var wide = admit_x(source, 1, "intercept_", context)
        var src = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(wide)))
        for t in range(targets):  # small-loop(the targets' intercepts)
            words[t] = Float64(src[t])
        _ = wide  # read through its address above
    _ = view  # read through its address above
    return words^


def linear_predict_targets_boundary(model: PythonObject, fields: PythonObject, value: PythonObject,
    route: DeviceRoute, pins: PythonObject, context: PythonObject, targets: Int) raises -> PythonObject:
    """`Ridge.predict(X)` of a multi-target fit (lane w3f3-k): rows x
    `targets` float32, `X @ coef_.T + intercept_`, as scikit-learn's. UNPACK
    reads the public `coef_` (targets x features) and `intercept_` (one per
    target) as the one-target predict reads its own, with X's finite test
    deferred to the entry as there; RUN is ONE call of the estimator's
    prediction entry with the T word (`linear_predict_targets_run`)."""
    var coefficient_source = py_get(fields, "coef_")
    var state = LinearModelState()
    state.has_coef = coefficient_source is not None
    linear_state_require_coef(state)
    var feature_source = fields["n_features_in_"]
    var intercept_source = fields["intercept_"]
    var est = classical_estimator_name(model)
    var admitted_x = admit_x_unrecorded(value, est, context, admitted=True, defer_finite=True)
    var x = admitted_x.array
    var rows = admitted_x.rows
    var cols = admitted_x.cols
    var test_x = not admitted_x.finite
    state.features = Int(py=feature_source)
    if test_x and cols != state.features:
        # scikit-learn refuses a non-finite X before the feature-count check
        if not host_all_finite_f32(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=admitted_x.address),
                                   rows * cols):
            admitted_finite_refuse(1, est, "X")
    linear_state_check_width(state, cols, est)
    var coef = admit_x(coefficient_source, 2, "coef_", context)
    state.coef_count = Int(py=coef.size)
    linear_state_check_coefficients(state, cols * targets)
    var intercepts = linear_intercepts_unpack(intercept_source, targets, context)
    var dst = classical_array(rows, targets, context)
    var coef_address = Int(py=classical_address(coef))
    var destination = Int(py=classical_address(dst))
    var entry = route.second.call
    var py = Python()
    var status = 0
    try:
        with GILReleased(py):
            status = linear_predict_targets_run(entry, admitted_x.address, coef_address, intercepts, destination,
                                                rows, cols, targets, test_x)
    finally:
        _ = x
        _ = admitted_x
        _ = coef
        _ = dst
        _ = intercepts
        _ = pins
    admitted_finite_refuse(status, est, "X")
    return dst
