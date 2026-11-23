# SPDX-License-Identifier: Apache-2.0
"""The member boundary of the Hierarchy family (AgglomerativeClustering, kind
0; `_mojolearn_member = (MEMBER_FAMILY_HIERARCHY, 0)` on the class). Recipe:
MEMBER-ABI-V2.md; RUN: core/hierarchy_member.mojo.

Only the form with a predict that fits on the class's solver route is a
member: prediction_data=True, single linkage, metric 'euclidean' or 'l2', no
connectivity matrix, no distance_threshold, compute_distances=False. Admit
answers False for every other instance (no predict, a connectivity matrix
or callable, or the x_cluster tree route) and the caller keeps the class's
own doors.

  * `hierarchy_member_admit` (UNPACK, once per member): the class's own
    validation (`_agglo_validate`, bindings/cfb_agglo_door.mojo) over the
    instance's parameters, its prediction_data refusal, `_fast_mst`'s
    environment read, the estimators binding through the estimator's own
    binding route (`estimator_binding(model, "_mojolearn_estimators")`, the
    DBSCAN labeled-reference entry -> `route.entry`, its capability word ->
    `route.flags`) and the solver binding the class's fit uses (the process
    default mode: `estimator_binding_routed(model, None, "_mojolearn_solver")`)
    -> `route.second`.
  * `hierarchy_member_pack` (PACK): what `cfb_agglo_operation` and
    `_agglo_legacy` publish (input_copied_ False: the member never copies
    X; _connectivity, _compute_distances, labels_, children_, n_clusters_,
    n_leaves_, n_boruvka_rounds_, n_connected_components_, n_features_in_,
    _fit_X, a fresh resident holder) through the same publication helper.
    The resident reference owner is minted by the class's first predict
    (`agglo_reference_owner`), not at pack.
  * `hierarchy_member_state` (UNPACK): a member state from a fitted model.

SERIAL BY NATURE: one member's admission or publication (a few attribute
reads and block copies); a caller runs its members in parallel.
"""
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from bindings.admitted_family_door import admit_family
from bindings.agglo_native_boundary_door import (
    AggloSolverConnection, agglo_array, agglo_publish, agglo_fresh_holder, agglo_integer_admitted,
)
from bindings.cfb_agglo_door import _agglo_validate, _fast_mst
from bindings.classical_arrays_door import classical_address, classical_c_bytes
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.cluster_fit_public_native_unpack import _attr
from bindings.dbscan_native_boundary_unpack import DBSCANConnection
from bindings.family_common_unpack import family_is_str
from bindings.member_owner_unpack import MemberOwner, member_host_only
from bindings.native_buffer_address_unpack import native_buffer_address
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_get, py_fields, py_type
from bindings.public_lifecycle_owner_door import lifecycle_admit_fields
from core.dbscan_native_api import DBSCAN_MODE, DBSCAN_VENDOR, DBSCAN_NATIVE_COUNT
from core.device_route import DeviceRoute, DEVICE_CPU, DEVICE_GPU
from core.entry_table import EntryRef, entry_ref_word
from core.hostptr import f32_ptr
from core.member_abi import (
    MemberPlan, MemberCall, MemberMatrix, member_call, member_new, member_free,
    MEMBER_OP_EXPORT, MEMBER_OP_IMPORT, MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_FAMILY, MEMBER_IMPORT_DONE,
    MEMBER_F64, MEMBER_I32,
)
from hierarchy.solver_descriptor_schema import HIERARCHY_SOLVER_OPERATIONS
from providers.python.compiled_binding_snapshot_pyapi_door import CompiledBindingSnapshot
from providers.python.estimator_binding_pyapi_door import estimator_binding_host, estimator_binding_routed
from providers.python.host_estimator_binding_pyapi_door import estimator_member_binding
from providers.python.hierarchy_settings_pyapi_unpack import AggloSettings

comptime _KIND_AGGLO = 0
comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]


def hierarchy_member_admit(model: PythonObject, fields: PythonObject, kind: Int, context: PythonObject,
    b: PythonObject, mut owner: MemberOwner) raises -> Bool:
    """UNPACK: fill `owner.plan.route`, `owner.plan.params` and `owner.pins`
    from the unfitted estimator. False where the class's own doors run the
    instance (see the module notes)."""
    if kind != _KIND_AGGLO:
        return False
    classical_connect_numpy(context)
    var connectivity = _attr(model, "connectivity")
    if connectivity is not None and not family_is_str(connectivity):
        return False
    var settings = AggloSettings(_attr(model, "numeric_mode"), _attr(model, "n_clusters"), _attr(model, "metric"),
        connectivity, _attr(model, "linkage"), _attr(model, "c"), _attr(model, "memory"),
        _attr(model, "compute_full_tree"), _attr(model, "distance_threshold"), _attr(model, "compute_distances"),
        _attr(model, "prediction_data"))
    var options = _agglo_validate(settings, context)
    var builtins = native_loaded_module("builtins")
    if py_type(settings.prediction_data) is not builtins.bool:
        raise Error("TypeError: mojolearn AgglomerativeClustering: prediction_data must be a bool, got "
            + String(py_type(settings.prediction_data).__name__))
    if not Bool(py=settings.prediction_data):
        return False
    options.prediction_data = True
    if not options.solver_route():
        return False
    var clusters = agglo_integer_admitted(settings.n_clusters)
    var fast = _fast_mst(0)
    # An explicit Host* class (its own host-only word, lane mf-iterimp3b): the host image its own methods
    # bind (`estimator_member_binding`); any other class: the estimator's binding route, as before.
    var binding = estimator_member_binding(model, "_mojolearn_estimators", context)
    var connection = DBSCANConnection(binding)
    var mode = connection.driver.metadata(DBSCAN_MODE)
    var vendor = connection.driver.metadata(DBSCAN_VENDOR) - 1
    if vendor < 0:
        raise Error("ImportError: mojolearn: the estimators binding names no vendor")
    var route = DeviceRoute(DEVICE_CPU if vendor == 0 else DEVICE_GPU)
    route.mode = mode
    route.vendor = vendor
    route.entry = EntryRef(connection.driver.entry, entry_ref_word(mode, vendor, DBSCAN_NATIVE_COUNT))
    route.flags = connection.capabilities
    # The solver keeps the process default mode it always ran in (requested None). An explicit Host* class
    # (lane mf-iterimp3b): the CPU identical solver beside its host estimators image, never the device one
    # (the route's vendor word above is the host image's).
    var solver_binding = (estimator_binding_host(model, "_mojolearn_solver", context) if member_host_only(model)
                          else estimator_binding_routed(model, PythonObject(None), "_mojolearn_solver", context))
    var solver = AggloSolverConnection(solver_binding)
    var solver_mode = solver.provider.downcast_value_ptr[CompiledBindingSnapshot]()[].mode
    route.second = EntryRef(solver.driver.entry, entry_ref_word(solver_mode, vendor, HIERARCHY_SOLVER_OPERATIONS))
    route.has_second = True
    owner.plan.route = route
    var words: List[Float64] = [Float64(_KIND_AGGLO), Float64(clusters), 1.0 if fast else 0.0]
    owner.plan.params = words^
    owner.pins = Python.tuple(binding, connection.provider, solver_binding, solver.provider)
    return True


def _export(plan: MemberPlan, mut call: MemberCall, block: Int) raises:
    call.block = block
    call.flags = 0
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)


def _i32_array(view: MemberMatrix, var shape: List[Int], context: PythonObject) raises -> PythonObject:
    var count = view.rows * view.cols
    var array = agglo_array(shape, "<i4", context)
    if count > 0:
        unsafe_memcpy(dest=_I32P(unsafe_from_address=Int(py=classical_address(array))),
            src=_I32P(unsafe_from_address=view.data), count=count)
    return array


def hierarchy_member_pack(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan, state: Int) raises:
    """PACK: write the fitted attributes of state `state` into `model`, as
    the class's solver-route fit publishes them (same names, dtypes, shapes
    and order)."""
    classical_connect_numpy(context)
    var model_fields = py_fields(model)
    lifecycle_admit_fields(model_fields)
    var call = MemberCall(plan.route, state)
    _export(plan, call, MEMBER_BLOCK_SCALARS)
    var scalars = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call.dst.data)
    var clusters = Int(scalars[0])
    var rounds = Int(scalars[1])
    var components = Int(scalars[2])
    _export(plan, call, MEMBER_BLOCK_FAMILY)
    var n = call.dst.cols
    var labels = _i32_array(call.dst, List[Int](length=1, fill=n), context)
    _export(plan, call, MEMBER_BLOCK_FAMILY + 1)
    var children_shape: List[Int] = [n - 1, 2]
    var children = _i32_array(call.dst, children_shape^, context)
    _export(plan, call, MEMBER_BLOCK_FAMILY + 2)
    var d = call.dst.cols
    var shape: List[Int] = [n, d]
    var fit_x = agglo_array(shape, "<f4", context)
    if n * d > 0:
        unsafe_memcpy(dest=f32_ptr(Int(py=classical_address(fit_x))), src=f32_ptr(call.dst.data), count=n * d)
    var connectivity = _attr(model, "connectivity")
    agglo_publish(model_fields, "input_copied_", PythonObject(False), context)
    agglo_publish(model_fields, "_connectivity",
        PythonObject("pairwise") if connectivity is None else connectivity, context)
    agglo_publish(model_fields, "_compute_distances", PythonObject(False), context)
    agglo_publish(model_fields, "labels_", labels, context)
    agglo_publish(model_fields, "children_", children, context)
    agglo_publish(model_fields, "n_clusters_", PythonObject(clusters), context)
    agglo_publish(model_fields, "n_leaves_", PythonObject(n), context)
    agglo_publish(model_fields, "n_boruvka_rounds_", PythonObject(rounds), context)
    agglo_publish(model_fields, "n_connected_components_", PythonObject(components), context)
    agglo_publish(model_fields, "n_features_in_", PythonObject(d), context)
    agglo_publish(model_fields, "_fit_X", fit_x, context)
    agglo_fresh_holder(model_fields)


def hierarchy_member_state(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan) raises -> Int:
    """UNPACK: a new member state holding the reference rows (`_fit_X`),
    labels_, children_ and the fitted scalars of a fitted model. The caller
    frees it (`member_free`)."""
    classical_connect_numpy(context)
    var source = py_get(fields, PythonObject("_fit_X"), PythonObject(None))
    var label_source = py_get(fields, PythonObject("labels_"), PythonObject(None))
    if source is None or label_source is None:
        raise Error("mojolearn AgglomerativeClustering.predict: prediction data was not stored. Fit with"
            + " AgglomerativeClustering(prediction_data=True)")
    var refs = admit_family(source, 2, "fitted reference rows", context).array
    var n = Int(py=refs.shape[0])
    var d = Int(py=refs.shape[1])
    # The int32 vectors' raw C-order words, as the class's predict hands them over.
    var labels = classical_c_bytes(label_source, "labels_", context)
    var child_source = py_get(fields, PythonObject("children_"), PythonObject(None))
    var children = PythonObject(None)
    var merges = 0
    if child_source is not None:
        children = classical_c_bytes(child_source, "children_", context)
        merges = len(children) // 8
    var names: List[String] = ["n_clusters_", "n_boruvka_rounds_", "n_connected_components_"]
    var scalars = List[Float64](length=3, fill=0.0)
    for i in range(3):  # driver-loop: three fitted scalar attributes
        var value = py_get(fields, PythonObject(names[i]), PythonObject(None))
        if value is not None:
            scalars[i] = Float64(py=value)
    var call = member_new(plan)
    try:
        call.block = MEMBER_BLOCK_FAMILY + 2
        call.x = MemberMatrix(Int(py=classical_address(refs)) if n * d > 0 else 0, n, d)
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = MEMBER_BLOCK_FAMILY
        call.x = MemberMatrix(native_buffer_address(labels) if n > 0 else 0, 1 if n > 0 else 0, n, MEMBER_I32)
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        if merges > 0:
            call.block = MEMBER_BLOCK_FAMILY + 1
            call.x = MemberMatrix(native_buffer_address(children), merges, 2, MEMBER_I32)
            _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = MEMBER_BLOCK_SCALARS
        call.x = MemberMatrix(Int(scalars.unsafe_ptr()), 1, 3, MEMBER_F64)
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    except error:
        member_free(plan, call)
        raise error
    _ = refs
    _ = source
    _ = labels
    _ = label_source
    _ = children
    _ = child_source
    _ = len(scalars)
    return call.state
