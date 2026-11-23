# SPDX-License-Identifier: Apache-2.0
"""The member boundary of the Hierarchy family (AgglomerativeClustering, kind
0; `_mojolearn_member = (MEMBER_FAMILY_HIERARCHY, 0)` on the class). Recipe:
MEMBER-ABI-V2.md; RUN: core/hierarchy_member.mojo.

Every setting the class's fit admits is a member (lane mf-clu6): both of
its numerical owners, the solver route and the x_cluster tree route (every
linkage and metric, a connectivity matrix, distance_threshold,
compute_distances), with prediction_data either way. A connectivity matrix
is read once here (dense or sparse, as `hierarchy_connectivity` reads it);
its refusals, a callable's included, are raised by the RUN at fit in
`_agglo_x`'s order.

  * `hierarchy_member_admit` (UNPACK, once per member): the class's own
    validation (`_agglo_validate`, bindings/cfb_agglo_door.mojo) over the
    instance's parameters, its prediction_data refusal, the estimators
    binding through the estimator's own binding route (the DBSCAN
    labeled-reference entry -> `route.entry`, its capability word ->
    `route.flags`), then on the solver route `_fast_mst`'s environment read
    and the solver binding (the process default mode) -> `route.second`, on
    the tree route the x_cluster binding -> `route.third` and, for cosine,
    the x_neighbors binding -> `route.fourth` (the estimator's mode, as
    `cfb_agglo_operation` selects them).
  * `hierarchy_member_pack` (PACK): what `_agglo_fit` and `_agglo_legacy` or
    `_agglo_x` publish, in that route's order (input_copied_ False: the
    member never copies X; _connectivity, _compute_distances, labels_,
    children_, n_clusters_, n_leaves_, n_boruvka_rounds_,
    n_connected_components_ and its warning, distances_ published or
    removed, n_features_in_, _fit_X or None, a fresh resident holder)
    through the same publication helper. The resident reference owner is
    minted by the class's first predict (`agglo_reference_owner`).
  * `hierarchy_member_state` (UNPACK): a member state from a fitted model.

SERIAL BY NATURE: one member's admission or publication (a few attribute
reads and block copies); a caller runs its members in parallel.
"""
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from bindings.admitted_family_door import admit_family
from bindings.agglo_native_boundary_door import (
    AggloSolverConnection, AggloClusterConnection, AggloZeroRowsConnection, agglo_array, agglo_publish,
    agglo_fresh_holder, agglo_integer_admitted, agglo_float_admitted,
)
from bindings.cluster_keys_unpack import cluster_keys
from bindings.host_sparse_native_door import host_sparse_format, _sparse_fields
from bindings.admitted_family_door import admit_family_meta
from bindings.metrics_native_support_door import metric_warn
from bindings.native_owner_lifetime_door import native_release_admit
from bindings.py_balanced import py_item
from cluster.cfb_agglo_run import agglo_checks_zero_rows, agglo_warns_components, agglo_publishes_distances
from core.agglo_native_state import AggloOptions
from core.cluster_member_words import member_words_append_f32, member_words_append_text
from hierarchy.cluster_descriptor_schema import HIERARCHY_CLUSTER_OPERATIONS
from providers.python.array_metadata_pyapi_door import ArrayPin
from providers.python.hierarchy_connectivity_pyapi_door import hierarchy_connectivity
from providers.python.x_neighbors_descriptor_pyapi_door import xn_admit
from x_neighbors.descriptor_schema import XN_OPERATION_COUNT
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


def _connectivity_words(source: PythonObject, context: PythonObject, mut words: List[Float64]) raises:
    """`hierarchy_connectivity`'s reads of a connectivity matrix, once, at
    UNPACK: its side, float32 words, edge count and edge mode are appended
    to `words` (NEW words 12-16 and the data behind them). X's row count is
    not known here, so the matrix's own side stands in for it; the shape
    check against X and any refusal its admission raised (a callable
    included: the class never calls one, its admission refuses it) are made
    at fit, where `_agglo_x` makes them (core/hierarchy_member.mojo)."""
    var side = 0
    var count = 0
    var edges = 0
    var mode = 0
    var refusal = String("")
    var data = List[Float64]()
    var nk = cluster_keys()
    try:
        var format = host_sparse_format(source)
        if format.byte_length() == 0:
            # The dense branch, as `hierarchy_connectivity` reads it.
            var pin = ArrayPin()
            var snapshot = admit_family_meta(source, 2, "connectivity", context, pin)
            side = snapshot.shape[0] if snapshot.shape[0] == snapshot.shape[1] else -1
            if side > 0:
                count = snapshot.count
                edges = snapshot.count
                mode = 1
                member_words_append_f32(data, snapshot.source.addr, count)
            _ = pin.keep
            _ = pin.owner
        else:
            # A sparse matrix's shape is read before its storage; a square
            # one is converted with its own side as the row count.
            var hint = 0
            try:
                var shape = py_item(_sparse_fields(source), nk[].k__shape)
                var first = Int(py=shape[0])
                hint = first if first == Int(py=shape[1]) else -1
            except:
                hint = 0
            if hint < 0:
                side = -1
            else:
                var aux = hierarchy_connectivity(source, hint, context)
                side = hint
                count = aux.count
                edges = aux.edges
                mode = aux.mode
                member_words_append_f32(data, aux.address, count)
                _ = aux^
    except error:
        refusal = String(error)
        side = 0
        count = 0
        edges = 0
        mode = 0
        data = List[Float64]()
    words.append(Float64(side))
    words.append(Float64(count))
    words.append(Float64(edges))
    words.append(Float64(mode))
    words.append(Float64(refusal.byte_length()))
    words.extend(data^)
    member_words_append_text(words, refusal)


def hierarchy_member_admit(model: PythonObject, fields: PythonObject, kind: Int, context: PythonObject,
    b: PythonObject, mut owner: MemberOwner) raises -> Bool:
    """UNPACK: fill `owner.plan.route`, `owner.plan.params` and `owner.pins`
    from the unfitted estimator. Every setting the class's fit admits is a
    member (lane mf-clu6): the answer is True or the class's own refusal."""
    if kind != _KIND_AGGLO:
        # Only kind 0 is marked (python/mojolearn/_hierarchy_impl.py); any
        # other mark is the member's own NEW refusal, raised here.
        raise Error("member ABI: hierarchy member kind " + String(kind) + " is not published")
    classical_connect_numpy(context)
    var connectivity = _attr(model, "connectivity")
    var settings = AggloSettings(_attr(model, "numeric_mode"), _attr(model, "n_clusters"), _attr(model, "metric"),
        connectivity, _attr(model, "linkage"), _attr(model, "c"), _attr(model, "memory"),
        _attr(model, "compute_full_tree"), _attr(model, "distance_threshold"), _attr(model, "compute_distances"),
        _attr(model, "prediction_data"))
    var options = _agglo_validate(settings, context)
    var builtins = native_loaded_module("builtins")
    if py_type(settings.prediction_data) is not builtins.bool:
        raise Error("TypeError: mojolearn AgglomerativeClustering: prediction_data must be a bool, got "
            + String(py_type(settings.prediction_data).__name__))
    options.prediction_data = Bool(py=settings.prediction_data)
    var solver_route = options.solver_route()
    var has_clusters = settings.n_clusters is not None
    var clusters = agglo_integer_admitted(settings.n_clusters) if has_clusters else 0
    var threshold = agglo_float_admitted(settings.distance_threshold) if options.has_threshold else Float64(0)
    # `_fast_mst` is read on the solver route only, as `_agglo_legacy` reads it.
    var fast = _fast_mst(0) if solver_route else False
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
    var host_only = member_host_only(model)
    # The images and providers the route's entries live in, pinned with the plan.
    var pin_a = PythonObject(None)
    var pin_b = PythonObject(None)
    var pin_c = PythonObject(None)
    var pin_d = PythonObject(None)
    if solver_route:
        # The solver keeps the process default mode it always ran in (requested None). An explicit Host*
        # class (lane mf-iterimp3b): the CPU identical solver beside its host estimators image, never the
        # device one (the route's vendor word above is the host image's).
        var solver_binding = (estimator_binding_host(model, "_mojolearn_solver", context) if host_only
                              else estimator_binding_routed(model, PythonObject(None), "_mojolearn_solver", context))
        var solver = AggloSolverConnection(solver_binding)
        var solver_mode = solver.provider.downcast_value_ptr[CompiledBindingSnapshot]()[].mode
        route.second = EntryRef(solver.driver.entry, entry_ref_word(solver_mode, vendor, HIERARCHY_SOLVER_OPERATIONS))
        route.has_second = True
        pin_a = solver_binding
        pin_b = solver.provider
    else:
        # The x_cluster tree route (`_agglo_x`): the estimator's x_cluster image in its numeric mode, and
        # for cosine its x_neighbors image (the zero-row test), as `cfb_agglo_operation` selects them.
        var cluster_binding = (estimator_binding_host(model, "_mojolearn_x_cluster", context) if host_only
                               else estimator_binding_routed(model, settings.numeric_mode, "_mojolearn_x_cluster",
                                   context))
        var cluster = AggloClusterConnection(cluster_binding)
        var cluster_mode = cluster.provider.downcast_value_ptr[CompiledBindingSnapshot]()[].mode
        route.third = EntryRef(cluster.driver.entry, entry_ref_word(cluster_mode, vendor, HIERARCHY_CLUSTER_OPERATIONS))
        route.has_third = True
        pin_a = cluster_binding
        pin_b = cluster.provider
        if agglo_checks_zero_rows(options.metric):
            var neighbors_binding = xn_admit(estimator_binding_host(model, "_mojolearn_x_neighbors", context)
                if host_only else estimator_binding_routed(model, settings.numeric_mode, "_mojolearn_x_neighbors",
                    context))
            var neighbors = AggloZeroRowsConnection(neighbors_binding)
            route.fourth = EntryRef(neighbors.driver.native.entry, entry_ref_word(mode, vendor, XN_OPERATION_COUNT))
            route.has_fourth = True
            pin_c = neighbors_binding
            pin_d = neighbors.provider
    owner.plan.route = route
    var words: List[Float64] = [Float64(_KIND_AGGLO), Float64(clusters), 1.0 if fast else 0.0,
        Float64(options.linkage), Float64(options.metric), 1.0 if options.constrained else 0.0,
        1.0 if options.has_threshold else 0.0, 1.0 if options.compute_distances else 0.0,
        1.0 if options.prediction_data else 0.0, Float64(options.full_tree), threshold,
        1.0 if has_clusters else 0.0]
    if options.constrained:
        _connectivity_words(connectivity, context, words)
    else:
        var empty: List[Float64] = [0.0, 0.0, 0.0, 0.0, 0.0]
        words.extend(empty^)
    owner.plan.params = words^
    owner.pins = Python.tuple(binding, connection.provider, pin_a, pin_b, pin_c, pin_d)
    return True


def _options(plan: MemberPlan) -> AggloOptions:
    """The class's settings as NEW words 3-9 carry them."""
    return AggloOptions(Int(plan.params[3]), Int(plan.params[4]), plan.params[5] != 0.0, plan.params[6] != 0.0,
        plan.params[7] != 0.0, plan.params[8] != 0.0, Int(plan.params[9]))


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
    the class's fit publishes them on the route it took (`_agglo_fit`, then
    `_agglo_legacy` or `_agglo_x`: same names, dtypes, shapes and order)."""
    classical_connect_numpy(context)
    var model_fields = py_fields(model)
    lifecycle_admit_fields(model_fields)
    var options = _options(plan)
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
    var children_shape: List[Int] = [call.dst.rows, 2]
    var children = _i32_array(call.dst, children_shape^, context)
    _export(plan, call, MEMBER_BLOCK_FAMILY + 2)
    var d = call.dst.cols
    var fit_x = PythonObject(None)
    if call.dst.rows > 0:
        var shape: List[Int] = [n, d]
        fit_x = agglo_array(shape, "<f4", context)
        if n * d > 0:
            unsafe_memcpy(dest=f32_ptr(Int(py=classical_address(fit_x))), src=f32_ptr(call.dst.data), count=n * d)
    _export(plan, call, MEMBER_BLOCK_FAMILY + 3)
    var merges = call.dst.cols
    var distances = agglo_array(List[Int](length=1, fill=merges), "<f4", context)
    if merges > 0:
        unsafe_memcpy(dest=f32_ptr(Int(py=classical_address(distances))), src=f32_ptr(call.dst.data), count=merges)
    var connectivity = _attr(model, "connectivity")
    agglo_publish(model_fields, "input_copied_", PythonObject(False), context)
    agglo_publish(model_fields, "_connectivity",
        PythonObject("pairwise") if connectivity is None else connectivity, context)
    agglo_publish(model_fields, "_compute_distances", PythonObject(options.compute_distances), context)
    if options.solver_route():
        agglo_publish(model_fields, "labels_", labels, context)
        agglo_publish(model_fields, "children_", children, context)
        agglo_publish(model_fields, "n_clusters_", PythonObject(clusters), context)
        agglo_publish(model_fields, "n_leaves_", PythonObject(n), context)
        agglo_publish(model_fields, "n_boruvka_rounds_", PythonObject(rounds), context)
        agglo_publish(model_fields, "n_connected_components_", PythonObject(components), context)
        agglo_publish(model_fields, "n_features_in_", PythonObject(d), context)
        agglo_publish(model_fields, "_fit_X", fit_x, context)
        agglo_fresh_holder(model_fields)
        return
    # `_agglo_x`'s publications, in its order.
    agglo_publish(model_fields, "children_", children, context)
    agglo_publish(model_fields, "n_leaves_", PythonObject(n), context)
    agglo_publish(model_fields, "n_connected_components_", PythonObject(components), context)
    if agglo_warns_components(options.constrained, components):
        metric_warn("the number of connected components of the connectivity matrix is " + String(components)
                    + " > 1. Completing it to avoid stopping the tree early.",
                    native_loaded_module("builtins").UserWarning, context)
    if agglo_publishes_distances(options):
        agglo_publish(model_fields, "distances_", distances, context)
    else:
        var nk = cluster_keys()
        lifecycle_admit_fields(model_fields)
        native_release_admit(py_get(model_fields, nk[].k_distances_), context)
        _ = py_call(model_fields.pop, PythonObject("distances_"), PythonObject(None))
    agglo_publish(model_fields, "n_clusters_", PythonObject(clusters), context)
    agglo_publish(model_fields, "labels_", labels, context)
    agglo_publish(model_fields, "n_boruvka_rounds_", PythonObject(rounds), context)
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
