# SPDX-License-Identifier: Apache-2.0
"""The member boundary of the Density family (DBSCAN, kind 1; KernelDensity,
kind 0; `_mojolearn_member = (MEMBER_FAMILY_DENSITY, kind)` on each class).
Recipe: MEMBER-ABI-V2.md; RUN: core/density_member.mojo.

KernelDensity is a member with fit, score_samples and score (no predict:
a grid search over its bandwidth scores it). Admit reads its own
validation (`_kde_validate`, bindings/kde_public_native_unpack.mojo) and
answers False for a kernel or metric outside the entry's tables (the class
refuses it by name) or a non-empty metric_params; its route is the
estimators binding's `kde_native_api_v1` entry. Its pack publishes what
`_kde_fit` publishes (a released resident cache, input_copied_, _x frozen,
n_features_in_, n_samples_fit_, _w frozen or None). A DBSCAN
with prediction_data=False has no predict either, and one with
metric='precomputed' takes words tied to the training rows: admit answers
False for both and the caller keeps the class's own doors.

  * `density_member_admit` (UNPACK, once per member): the estimator's own
    reads and refusals in its fit's order
    (bindings/dbscan_public_native_unpack.mojo `dbscan_fit_settings`,
    `dbscan_fit_budget`), its binding through the estimator's own binding
    route (`estimator_binding(model, "_mojolearn_estimators")`) and that
    binding's `dbscan_native_api_v1` entry as the route's `entry`, its
    capability word in `route.flags`.
  * `density_member_pack` (PACK): the attributes `_dbscan_fit` publishes
    (bindings/dbscan_public_native_pack.mojo: n_iter_, labels_,
    n_features_in_, n_samples_fit_, core_sample_indices_, components_ and
    _core_labels frozen, input_copied_, a fresh resident holder), through
    the same publication helper.
  * `density_member_state` (UNPACK): a member state from a fitted model.

SERIAL BY NATURE: one member's admission or publication (a few attribute
reads and block copies); a caller runs its members in parallel.
"""
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from bindings.admitted_family_door import admit_family
from bindings.classical_arrays_door import classical_address, classical_c_bytes
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.cluster_keys_unpack import cluster_keys
from bindings.dbscan_native_boundary_pack import dbscan_array, dbscan_publish, dbscan_fresh_holder
from bindings.dbscan_native_boundary_unpack import DBSCANConnection
from bindings.dbscan_public_native_unpack import dbscan_fit_settings, dbscan_fit_budget
from bindings.member_owner_unpack import MemberOwner
from bindings.native_buffer_address_unpack import native_buffer_address
from bindings.native_loaded_module_door import native_loaded_module
from bindings.p3nc_fitted_freeze_unpack import fitted_freeze
from bindings.py_balanced import py_get, py_fields, py_setitem
from bindings.public_lifecycle_owner_door import lifecycle_admit_fields
from cluster.dbscan_door_run import DBSCAN_METRIC_PRECOMPUTED
from core.dbscan_native_api import DBSCAN_MODE, DBSCAN_VENDOR, DBSCAN_NATIVE_COUNT
from core.kde_native_api import KDE_MODE, KDE_VENDOR, KDE_NATIVE_COUNT
from bindings.kde_native_boundary_unpack import KDEConnection
from bindings.kde_public_native_unpack import _kde_validate
from bindings.kde_public_native_pack import _kde_release
from bindings.family_common_unpack import family_is_str, family_str
from bindings.py_balanced import py_call
from core.device_route import DeviceRoute, DEVICE_CPU, DEVICE_GPU
from core.entry_table import EntryRef, entry_ref_word
from core.hostptr import f32_ptr
from core.member_abi import (
    MemberPlan, MemberCall, MemberMatrix, member_call, member_new, member_free,
    MEMBER_OP_EXPORT, MEMBER_OP_IMPORT, MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_FAMILY, MEMBER_IMPORT_DONE,
    MEMBER_F64, MEMBER_I32,
)
from providers.python.host_estimator_binding_pyapi_door import estimator_member_binding

comptime _KIND_DBSCAN = 1
comptime _KIND_KDE = 0
comptime _BLOCK_TRAIN = MEMBER_BLOCK_FAMILY + 4
comptime _BLOCK_WEIGHTS = MEMBER_BLOCK_FAMILY + 5


def _code(value: PythonObject, names: List[String]) raises -> Int:
    """The index of the exact string `value` in `names`, else -1."""
    if not family_is_str(value):
        return -1
    var text = family_str(value)
    for i in range(len(names)):  # driver-loop: at most nine option names
        if text == names[i]:
            return i
    return -1


def _kde_admit(model: PythonObject, fields: PythonObject, context: PythonObject, mut owner: MemberOwner) raises -> Bool:
    """`_kde_fit`'s validation, then the words; False where the class's own
    door must refuse or run the instance."""
    _kde_validate(fields, context)
    var kernels: List[String] = ["gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"]
    var metrics: List[String] = ["euclidean", "l2", "sqeuclidean", "l1", "cityblock", "manhattan", "chebyshev",
        "cosine", "minkowski"]
    var kernel = _code(py_get(fields, PythonObject("kernel"), PythonObject(None)), kernels)
    var metric = _code(py_get(fields, PythonObject("metric"), PythonObject(None)), metrics)
    if kernel < 0 or metric < 0:
        return False
    var params = py_get(fields, PythonObject("metric_params"), PythonObject(None))
    if params is not None and len(params) > 0:
        return False
    var bandwidth = Float64(py=py_call(native_loaded_module("builtins").float,
        py_get(fields, PythonObject("bandwidth"), PythonObject(None))))
    # An explicit Host* class (its own host-only word, lane mf-iterimp3b): the host image its own methods
    # bind (`estimator_member_binding`); any other class: the estimator's binding route, as before.
    var binding = estimator_member_binding(model, "_mojolearn_estimators", context)
    var connection = KDEConnection(binding)
    var mode = connection.driver.metadata(KDE_MODE)
    var vendor = connection.driver.metadata(KDE_VENDOR) - 1
    if vendor < 0:
        raise Error("ImportError: mojolearn: the estimators binding names no vendor")
    var route = DeviceRoute(DEVICE_CPU if vendor == 0 else DEVICE_GPU)
    route.mode = mode
    route.vendor = vendor
    route.entry = EntryRef(connection.driver.entry, entry_ref_word(mode, vendor, KDE_NATIVE_COUNT))
    route.flags = connection.capabilities
    owner.plan.route = route
    var words: List[Float64] = [Float64(_KIND_KDE), bandwidth, Float64(kernel), Float64(metric)]
    owner.plan.params = words^
    owner.pins = Python.tuple(binding, connection.provider)
    return True
comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]


def density_member_admit(model: PythonObject, fields: PythonObject, kind: Int, context: PythonObject,
    b: PythonObject, mut owner: MemberOwner) raises -> Bool:
    """UNPACK: fill `owner.plan.route`, `owner.plan.params` and `owner.pins`
    from the unfitted estimator. False where the class's own doors run the
    instance (see the module notes)."""
    if kind == _KIND_KDE:
        classical_connect_numpy(context)
        return _kde_admit(model, fields, context, owner)
    if kind != _KIND_DBSCAN:
        return False
    classical_connect_numpy(context)
    var builtins = native_loaded_module("builtins")
    var settings = dbscan_fit_settings(fields, builtins, context)
    if not settings.predicting or settings.metric == DBSCAN_METRIC_PRECOMPUTED:
        return False
    var budget = dbscan_fit_budget(fields, builtins, context)
    # An explicit Host* class (its own host-only word, lane mf-iterimp3b): the host image its own methods
    # bind (`estimator_member_binding`); any other class: the estimator's binding route, as before.
    var binding = estimator_member_binding(model, "_mojolearn_estimators", context)
    var connection = DBSCANConnection(binding)
    var mode = connection.driver.metadata(DBSCAN_MODE)
    # The entry's vendor word is 1 + the member vendor (0: no vendor).
    var vendor = connection.driver.metadata(DBSCAN_VENDOR) - 1
    if vendor < 0:
        raise Error("ImportError: mojolearn: the estimators binding names no vendor")
    var route = DeviceRoute(DEVICE_CPU if vendor == 0 else DEVICE_GPU)
    route.mode = mode
    route.vendor = vendor
    route.entry = EntryRef(connection.driver.entry, entry_ref_word(mode, vendor, DBSCAN_NATIVE_COUNT))
    route.flags = connection.capabilities
    owner.plan.route = route
    var words: List[Float64] = [Float64(_KIND_DBSCAN), settings.eps, Float64(settings.min_samples),
        Float64(settings.metric), Float64(settings.method), Float64(settings.cap), Float64(budget)]
    owner.plan.params = words^
    owner.pins = Python.tuple(binding, connection.provider)
    return True


def _export(plan: MemberPlan, mut call: MemberCall, block: Int) raises:
    call.block = block
    call.flags = 0
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)


def _i32_array(view: MemberMatrix, context: PythonObject) raises -> PythonObject:
    var count = view.rows * view.cols
    var array = dbscan_array(List[Int](length=1, fill=count), "<i4", context)
    if count > 0:
        unsafe_memcpy(dest=_I32P(unsafe_from_address=Int(py=classical_address(array))),
            src=_I32P(unsafe_from_address=view.data), count=count)
    return array


def _kde_pack(model: PythonObject, context: PythonObject, plan: MemberPlan, state: Int) raises:
    """`_kde_fit`'s publications, in its order, from the state's blocks."""
    var model_fields = py_fields(model)
    _kde_release(model_fields)
    lifecycle_admit_fields(model_fields)
    var call = MemberCall(plan.route, state)
    _export(plan, call, _BLOCK_TRAIN)
    var n = call.dst.rows
    var d = call.dst.cols
    var shape: List[Int] = [n, d]
    var x = dbscan_array(shape, "<f4", context)
    if n * d > 0:
        unsafe_memcpy(dest=f32_ptr(Int(py=classical_address(x))), src=f32_ptr(call.dst.data), count=n * d)
    _export(plan, call, _BLOCK_WEIGHTS)
    var weights = PythonObject(None)
    if call.dst.rows * call.dst.cols > 0:
        weights = dbscan_array(List[Int](length=1, fill=n), "<f4", context)
        unsafe_memcpy(dest=f32_ptr(Int(py=classical_address(weights))), src=f32_ptr(call.dst.data), count=n)
    py_setitem(model_fields, "input_copied_", PythonObject(False))
    _ = fitted_freeze(x, context)
    py_setitem(model_fields, "_x", x)
    py_setitem(model_fields, "n_features_in_", PythonObject(d))
    py_setitem(model_fields, "n_samples_fit_", PythonObject(n))
    if weights is not None:
        _ = fitted_freeze(weights, context)
    py_setitem(model_fields, "_w", weights)


def _kde_state(fields: PythonObject, context: PythonObject, plan: MemberPlan) raises -> Int:
    """A KernelDensity state from a fitted model: `_x`, `_w`, [rows, features]."""
    var source = py_get(fields, PythonObject("_x"), PythonObject(None))
    if source is None:
        raise Error("mojolearn KernelDensity: call fit() first")
    var x = admit_family(source, 2, "_x", context).array
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    var w_source = py_get(fields, PythonObject("_w"), PythonObject(None))
    var w = PythonObject(None)
    if w_source is not None:
        w = admit_family(w_source, 1, "_w", context).array
    var scalars: List[Float64] = [Float64(n), Float64(d)]
    var call = member_new(plan)
    try:
        call.block = _BLOCK_TRAIN
        call.x = MemberMatrix(Int(py=classical_address(x)) if n * d > 0 else 0, n, d)
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        if w is not None:
            call.block = _BLOCK_WEIGHTS
            call.x = MemberMatrix(Int(py=classical_address(w)) if n > 0 else 0, 1 if n > 0 else 0, n)
            _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = MEMBER_BLOCK_SCALARS
        call.x = MemberMatrix(Int(scalars.unsafe_ptr()), 1, 2, MEMBER_F64)
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    except error:
        member_free(plan, call)
        raise error
    _ = x
    _ = source
    _ = w
    _ = w_source
    _ = len(scalars)
    return call.state


def density_member_pack(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan, state: Int) raises:
    """PACK: write the fitted attributes of state `state` into `model`, as
    `_dbscan_fit` publishes them (same names, dtypes, shapes and order)."""
    classical_connect_numpy(context)
    if Int(plan.params[0]) == _KIND_KDE:
        _kde_pack(model, context, plan, state)
        return
    var model_fields = py_fields(model)
    lifecycle_admit_fields(model_fields)
    var nk = cluster_keys()
    var call = MemberCall(plan.route, state)
    _export(plan, call, MEMBER_BLOCK_SCALARS)
    var scalars = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call.dst.data)
    var n_iter = Int(scalars[0])
    var d = Int(scalars[1])
    _export(plan, call, MEMBER_BLOCK_FAMILY)
    var n = call.dst.cols
    var labels = _i32_array(call.dst, context)
    _export(plan, call, MEMBER_BLOCK_FAMILY + 1)
    var indices = _i32_array(call.dst, context)
    _export(plan, call, MEMBER_BLOCK_FAMILY + 2)
    var m = call.dst.rows
    var shape: List[Int] = [m, d]
    var components = dbscan_array(shape, "<f4", context)
    if m * d > 0:
        unsafe_memcpy(dest=f32_ptr(Int(py=classical_address(components))), src=f32_ptr(call.dst.data), count=m * d)
    _export(plan, call, MEMBER_BLOCK_FAMILY + 3)
    var core_labels = _i32_array(call.dst, context)
    # The member read the caller's X as it is: no input copy.
    py_setitem(model_fields, nk[].k_input_copied_, PythonObject(False))
    dbscan_publish(model_fields, "n_iter_", PythonObject(n_iter), context)
    dbscan_publish(model_fields, "labels_", labels, context)
    dbscan_publish(model_fields, "n_features_in_", PythonObject(d), context)
    dbscan_publish(model_fields, "n_samples_fit_", PythonObject(n), context)
    _ = fitted_freeze(indices, context)
    _ = fitted_freeze(components, context)
    _ = fitted_freeze(core_labels, context)
    dbscan_publish(model_fields, "core_sample_indices_", indices, context)
    dbscan_publish(model_fields, "components_", components, context)
    dbscan_publish(model_fields, "_core_labels", core_labels, context)
    # The class's next predict mints its resident core rows from the new arrays.
    dbscan_fresh_holder(model)


def density_member_state(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan) raises -> Int:
    """UNPACK: a new member state holding the core rows of a fitted DBSCAN
    (labels_ are not needed to predict and are not imported). The caller
    frees it (`member_free`)."""
    classical_connect_numpy(context)
    if Int(plan.params[0]) == _KIND_KDE:
        return _kde_state(fields, context, plan)
    var source = py_get(fields, PythonObject("components_"), PythonObject(None))
    if source is None:
        raise Error("NotFittedError: this member is not fitted yet")
    var components = admit_family(source, 2, "components_", context).array
    var m = Int(py=components.shape[0])
    var d = Int(py=components.shape[1])
    # The int32 vectors' raw C-order words, as the class's predict hands them over.
    var index_source = py_get(fields, PythonObject("core_sample_indices_"), PythonObject(None))
    var label_source = py_get(fields, PythonObject("_core_labels"), PythonObject(None))
    if index_source is None or label_source is None:
        raise Error("NotFittedError: this member is not fitted yet")
    var indices = classical_c_bytes(index_source, "core_sample_indices_", context)
    var core_labels = classical_c_bytes(label_source, "_core_labels", context)
    var n_iter = py_get(fields, PythonObject("n_iter_"), PythonObject(0))
    var scalars: List[Float64] = [Float64(py=n_iter), Float64(d)]
    var call = member_new(plan)
    try:
        call.block = MEMBER_BLOCK_FAMILY + 1
        call.x = MemberMatrix(native_buffer_address(indices) if m > 0 else 0, 1 if m > 0 else 0, m, MEMBER_I32)
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = MEMBER_BLOCK_FAMILY + 3
        call.x = MemberMatrix(native_buffer_address(core_labels) if m > 0 else 0, 1 if m > 0 else 0, m, MEMBER_I32)
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = MEMBER_BLOCK_FAMILY + 2
        call.x = MemberMatrix(Int(py=classical_address(components)) if m > 0 else 0, m, d)
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = MEMBER_BLOCK_SCALARS
        call.x = MemberMatrix(Int(scalars.unsafe_ptr()), 1, 2, MEMBER_F64)
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    except error:
        member_free(plan, call)
        raise error
    _ = components
    _ = source
    _ = indices
    _ = core_labels
    _ = index_source
    _ = label_source
    _ = len(scalars)
    return call.state
