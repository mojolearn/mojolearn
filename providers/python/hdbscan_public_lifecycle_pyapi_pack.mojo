# SPDX-License-Identifier: Apache-2.0
# HDBSCAN's public lifecycle: the operator of every public operation (UNPACK
# -> RUN -> PACK) and its PACK half: public output allocation, the per-model
# query record, and the returned objects. UNPACK is
# providers/python/hdbscan_public_lifecycle_pyapi_unpack.mojo; RUN is
# providers/python/hdbscan_public_lifecycle_run.mojo with the typed provider
# call of core/hdbscan_query_api.mojo (entered with the GIL released by
# bindings/hdbscan_query_boundary_unpack.mojo `hdbscan_query_run`); the fit is
# bindings/hdbscan_fit_boundary_pack.mojo `hdbscan_fit`.
from std.python import Python, PythonObject
from bindings.py_balanced import py_type, py_get, py_fields, py_item
from bindings.native_loaded_module_door import native_loaded_module
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.cluster_side_records_unpack import cluster_is_subclass, cluster_record_put, cluster_record_drop
from bindings.classical_validate_unpack import classical_check_x, classical_estimator_name
from bindings.native_array_view_pack import native_array_ndarray_view
from bindings.family_common_pack import family_array
from bindings.hdbscan_fit_boundary_unpack import _hdbscan_mode
from bindings.hdbscan_fit_boundary_pack import hdbscan_fit
from bindings.hdbscan_query_boundary_unpack import HdbscanQueryRecord, hdbscan_query_run, hdbscan_query_prepare
from providers.python.estimator_binding_pyapi_door import estimator_binding
from providers.python.estimator_lifecycle_pyapi_unpack import estimator_outside, estimator_fit_begin, estimator_fit_end, estimator_fit_complete
from providers.python.estimator_tabular_boundary_pyapi_unpack import _target, _refusal
from providers.python.hdbscan_public_lifecycle_run import _query_slot, hdbscan_query_kind, hdbscan_query_request
from providers.python.hdbscan_public_lifecycle_pyapi_unpack import (
    HdbscanQueryCall, hdbscan_initialize, hdbscan_query_unpack, hdbscan_query_unpack_storage,
    hdbscan_query_buffers, _admit, _record_key, _ADMIT_SLOT,
)
from bindings.operation_names_hdbscan import (
    HDBSCAN_OPN_FAMILY_HDBSCAN_PUBLIC_INITIALIZE, HDBSCAN_OPN_FAMILY_HDBSCAN_PUBLIC_BINDING,
    HDBSCAN_OPN_FAMILY_HDBSCAN_PUBLIC_FIT_PREDICT, HDBSCAN_OPN_FAMILY_HDBSCAN_PUBLIC_APPROXIMATE_PREDICT,
    HDBSCAN_OPN_FAMILY_HDBSCAN_PUBLIC_MEMBERSHIP_VECTOR, HDBSCAN_OPN_FAMILY_HDBSCAN_PUBLIC_ALL_POINTS_MEMBERSHIP_VECTORS)
from bindings.cluster_keys_unpack import cluster_keys
from providers.python.function_device_boundary_door import function_device_admit
from providers.python.device_route_boundary_door import device_request
from core.function_device_fitted import function_device_fitted


def _record(model: PythonObject, slot: String, key: PythonObject, binding: PythonObject,
            call: HdbscanQueryCall, handle: Int) raises:
    """Same weak-key/provider cache with typed counts, driver and named pins,
    and (lane vdf-misc4) the fitted snapshot `handle` the record now owns."""
    var record = HdbscanQueryRecord(binding, call.connection.value().provider,
        call.connection.value().driver.copy(), call.connection.value().host, call.dimensions(),
        call.raw, call.core, call.labels, call.parents, call.lambdas, call.deaths, call.clusters,
        call.children, call.indices, call.offsets, handle)
    cluster_record_put(slot, model, key, PythonObject(alloc=record^))


def _query(model: PythonObject, operation: Int, args: PythonObject, fields: PythonObject,
           binding: PythonObject, context: PythonObject, key: PythonObject) raises -> PythonObject:
    """approximate_predict, membership_vector or all_points_membership_vectors:
    UNPACK into `HdbscanQueryCall`, allocate the public output, build the
    typed request, one provider call, publish. Same refusals in the same
    order and the same allocations at the same stages as the single body
    this replaced (`_inference`)."""
    var approximate = operation == HDBSCAN_OPN_FAMILY_HDBSCAN_PUBLIC_APPROXIMATE_PREDICT
    var call = HdbscanQueryCall()
    hdbscan_query_unpack(call, model, operation, args, fields, binding, context, key)
    var output = family_array(Python.tuple(call.nq), "<i4", context) if approximate else family_array(Python.tuple(call.nq, call.selected), "<f4", context)
    if not approximate and (call.selected == 0 or call.nq == 0):
        return output
    hdbscan_query_unpack_storage(call, model, operation, fields, binding, context)
    var probabilities = PythonObject(None)
    if approximate:
        probabilities = family_array(Python.tuple(call.nq), "<f4", context)
    var buffers = hdbscan_query_buffers(call, operation, output, probabilities)
    # RUN: typed request, typed provider call.
    var request = hdbscan_query_request(call.dimensions(), buffers, call.nq, call.batch)
    # Lane vdf-misc4: on the device image the first query of a fitted model
    # takes its fitted arrays ONCE onto the device (a snapshot the model's
    # query record owns and releases); later queries read it in place.
    var handle = call.handle
    var minted = False
    if handle == 0 and not call.held:
        handle = hdbscan_query_prepare(call.connection.value(), request, hdbscan_query_kind(operation))
        minted = handle != 0
    try:
        hdbscan_query_run(call.connection.value(), request, hdbscan_query_kind(operation), handle)
    except problem:
        if minted:
            try:
                call.connection.value().driver.release(handle)
            except:
                pass
        raise problem^
    if not call.held:
        _record(model, _query_slot(operation), key, binding, call, handle)
    # Retain every input and output buffer owner through the final launch.
    _ = call^
    if approximate:
        return Python.tuple(output, probabilities)
    return output


# HDBSCAN has a device image and a CPU identical image (`_mojolearn_hdbscan`'s
# host route, the one HDBSCAN(device="cpu") runs).
comptime _HDBSCAN_HAS_GPU = True
comptime _HDBSCAN_HAS_CPU = True


def hdbscan_unknown_operation(operation: String) raises:
    """A `family_hdbscan_public_*` name the operation table does not hold
    (the shells pass table names only). The name chain had no refusal of
    its own (an unknown method ran the membership-vector query); the
    family door's unknown-operation text is raised instead."""
    raise Error("RuntimeError: unknown family operation " + operation)


def hdbscan_public_operation(model: PythonObject, operation: Int, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`operation` is the leaf code of bindings/operation_names_hdbscan.mojo
    (lane bw2-cluster: it was the name, sliced to a method String per call
    and matched by up to nine compares; the fit name and the three query
    slot names were concatenated per call)."""
    classical_connect_numpy(context)
    var b = native_loaded_module("builtins")
    if py_type(args) is not b.tuple:
        raise Error("TypeError: HDBSCAN arguments require their native tuple")
    # Passive subclasses share the connected native owner: CPython's MRO
    # test (bindings/cluster_side_records.mojo), not a Python-level walk.
    var nk = cluster_keys()
    if not cluster_is_subclass(py_type(model), py_item(context, nk[].k_hdbscan_public_class)):
        raise Error("TypeError: HDBSCAN execution requires its connected class")
    if operation == HDBSCAN_OPN_FAMILY_HDBSCAN_PUBLIC_INITIALIZE:
        return hdbscan_initialize(model, args, context)
    var fitting = operation > HDBSCAN_OPN_FAMILY_HDBSCAN_PUBLIC_BINDING and operation <= HDBSCAN_OPN_FAMILY_HDBSCAN_PUBLIC_FIT_PREDICT
    var query = operation >= HDBSCAN_OPN_FAMILY_HDBSCAN_PUBLIC_APPROXIMATE_PREDICT
    if query:
        # The query function's `device` (lane vdf-misc), its last argument
        # item, admitted first in the shared words. A query runs where the
        # fitted clusterer lives (its own route, below); an explicit word that
        # disagrees with that side is refused before any work
        # (core/function_device_fitted.mojo). None is one identity compare.
        var word = args[len(args) - 1]
        if word is not None:
            var name = String("approximate_predict")
            if operation == HDBSCAN_OPN_FAMILY_HDBSCAN_PUBLIC_MEMBERSHIP_VECTOR:
                name = "membership_vector"
            elif operation == HDBSCAN_OPN_FAMILY_HDBSCAN_PUBLIC_ALL_POINTS_MEMBERSHIP_VECTORS:
                name = "all_points_membership_vectors"
            var device = function_device_admit(word, name, _HDBSCAN_HAS_GPU, _HDBSCAN_HAS_CPU, context, b)
            function_device_fitted(device, device_request(py_fields(model), b), name,
                                   classical_estimator_name(model))
    var key = _record_key(model, context) if query else PythonObject(None)
    var recorded = _admit(model, context, fitting, key)
    var fields = py_fields(model)
    if fitting:
        # A fit replaces the fitted state: release the old records now.
        cluster_record_drop(_ADMIT_SLOT, model)
        cluster_record_drop(_query_slot(HDBSCAN_OPN_FAMILY_HDBSCAN_PUBLIC_APPROXIMATE_PREDICT), model)
        cluster_record_drop(_query_slot(HDBSCAN_OPN_FAMILY_HDBSCAN_PUBLIC_MEMBERSHIP_VECTOR), model)
        cluster_record_drop(_query_slot(HDBSCAN_OPN_FAMILY_HDBSCAN_PUBLIC_ALL_POINTS_MEMBERSHIP_VECTORS), model)
    var binding = estimator_binding(model, "_mojolearn_hdbscan", context, False)
    if operation == HDBSCAN_OPN_FAMILY_HDBSCAN_PUBLIC_BINDING:
        _hdbscan_mode(model, binding, context)
        return binding
    if not fitting:
        var result = _query(model, operation, args, fields, binding, context, key)
        if query and not recorded:
            cluster_record_put(_ADMIT_SLOT, model, key, PythonObject(True))
        return result
    var name = classical_estimator_name(model)
    var x = classical_check_x(args[0], name, context)
    var rows = Int(py=x.shape[0])
    var columns = Int(py=x.shape[1])
    _target(args[1], rows, name, context)
    var depth = estimator_fit_begin(fields)
    var result = model
    try:
        # The fit's own operator (it was reached through the cluster fit
        # door's code dispatch with a Python tuple built per call).
        var labels = hdbscan_fit(model, binding, x, context)
        if operation == HDBSCAN_OPN_FAMILY_HDBSCAN_PUBLIC_FIT_PREDICT:
            result = labels
    except problem:
        var refused = _refusal(problem, True, depth > 0, rows, columns, name)
        raise refused^
    finally:
        estimator_fit_end(model, depth)
    estimator_fit_complete(model, args[0], False, context)
    if estimator_outside() and py_type(result) is py_item(context, nk[].k_array_class) and py_get(context, nk[].k_ndarray) is not None:
        return native_array_ndarray_view(result, context)
    return result
