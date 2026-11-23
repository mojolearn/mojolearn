# SPDX-License-Identifier: Apache-2.0
"""The member boundary of the Neighbors family: KNeighborsClassifier (kind 0)
and KNeighborsRegressor (kind 1), `_mojolearn_member = (11, kind)` on each
class (MEMBER_FAMILY_NEIGHBORS; RUN: core/neighbors_member.mojo).
NearestNeighbors and RadiusNeighbors have no predict and are not members.

  * `neighbors_member_admit` (UNPACK, once per member): the estimator's own
    fit refusals (`_check_refusals`: algorithm, metric, p, weights), its
    resolved distance words, `n_neighbors` and `query_tile` as the six
    parameter words, and its route: the binding the estimator's own predict
    resolves (`_mojolearn` through `estimator_binding`, the CPU column for a
    CPU route) and that image's neighbors operations entry in the route's
    `entry` slot. The binding is the pin.
  * `neighbors_member_pack` (PACK): the fitted attributes the estimator's own
    fit writes (`_index`, `n_samples_fit_`, `n_features_in_`, `_y_cols`,
    `outputs_2d_`, the classifier's `_classes_list` from the native encoder,
    a fresh resident holder). A classifier packed with labels
    (`member_pack_labeled`) takes them here (`member_labels_take`): int32
    labels become the int32 target words (as the estimator's own fit keeps
    them, `_label_names` None); any other labels stay as `_label_names`
    over the code targets (predict decodes the voted codes, as for str
    labels). Without labels the targets are the codes.
  * `neighbors_member_state` (UNPACK): a member state from a fitted
    one-output estimator; the classifier's stored labels are encoded to codes
    by the native encoder (the class count is its class list's length).

SERIAL BY NATURE: one member's admission or publication (a few record reads
and block copies); a caller runs its members in parallel.
"""
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from bindings.admitted_input_unpack import admit_x, admit_c_view, admitted_dtype
from bindings.classical_arrays_door import classical_array, classical_address
from bindings.classical_labels_door import classical_encode_labels, classical_decode_labels, labels_cast, labels_reshape
from bindings.labels_descriptor_boundary_door import labels_scope
from bindings.family_common_pack import family_array, family_field
from bindings.member_owner_unpack import MemberOwner, member_labels_take, member_host_only
from bindings.neighbors_knn_native_door import _check_refusals, knn_dist_words, knn_ops_key, knn_release
from bindings.py_balanced import py_get, py_setitem
from core.device_route import DeviceRoute, DEVICE_CPU, DEVICE_GPU
from core.entry_table import EntryRef, entry_ref_word
from core.member_abi import (
    MemberPlan, MemberCall, MemberMatrix, member_call, member_new, member_free,
    MEMBER_OP_EXPORT, MEMBER_OP_IMPORT, MEMBER_IMPORT_DONE, MEMBER_I32,
)
from core.neighbors_entry_table import NB_OPERATION_COUNT, neighbors_ops_entry
from core.neighbors_member import NEIGHBORS_BLOCK_POINTS, NEIGHBORS_BLOCK_TARGETS
from providers.python.estimator_binding_pyapi_door import estimator_binding, estimator_binding_host

comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]


def neighbors_member_admit(model: PythonObject, fields: PythonObject, kind: Int, context: PythonObject,
    b: PythonObject, mut owner: MemberOwner) raises -> Bool:
    """UNPACK: fill `owner.plan.route`, `owner.plan.params` and `owner.pins`
    from the unfitted estimator."""
    if kind != 0 and kind != 1:
        raise Error("ValueError: mojolearn: unknown k-NN member kind " + String(kind))
    # The estimator's own refusals of its fit (by name, in its order).
    _ = _check_refusals(model, context)
    var dist = knn_dist_words(model, context)
    var k_source = family_field(fields, "n_neighbors")
    var tile_source = family_field(fields, "query_tile")
    var k = Int(py=k_source)
    var tile = Int(py=tile_source)
    # The image the estimator's own predict calls, and its operations entry.
    # An explicit HostKNeighbors* (its own host-only word, lane mf-cal3b): the core host image its own
    # fit and predict bind.
    var binding = (estimator_binding_host(model, "_mojolearn", context) if member_host_only(model)
                   else estimator_binding(model, "_mojolearn", context))
    var key = knn_ops_key(binding)
    var route = DeviceRoute(DEVICE_CPU if key.vendor == 0 else DEVICE_GPU)
    route.mode = key.mode
    route.vendor = key.vendor
    route.entry = EntryRef(neighbors_ops_entry(key), entry_ref_word(key.mode, key.vendor, NB_OPERATION_COUNT))
    owner.plan.route = route
    var words: List[Float64] = [Float64(kind), Float64(k), Float64(dist.metric), dist.metric_arg,
        Float64(dist.weights), Float64(tile)]
    owner.plan.params = words^
    owner.pins = binding
    return True


def neighbors_member_pack(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan, state: Int) raises:
    """PACK: write the fitted attributes of state `state` into `model`, as
    the estimator's own fit does."""
    var call = MemberCall(plan.route, state)
    call.block = NEIGHBORS_BLOCK_POINTS
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var rows = call.dst.rows
    var cols = call.dst.cols
    var index = classical_array(rows, cols, context)
    unsafe_memcpy(dest=_F32P(unsafe_from_address=Int(py=classical_address(index))),
        src=_F32P(unsafe_from_address=call.dst.data), count=rows * cols)
    call.block = NEIGHBORS_BLOCK_TARGETS
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    # A refit drops the device copy of the previous index, as the fit does.
    knn_release(model)
    py_setitem(fields, "_resident_holder", Python.dict())
    py_setitem(fields, "_index", index)
    py_setitem(fields, "n_samples_fit_", PythonObject(rows))
    py_setitem(fields, "n_features_in_", PythonObject(cols))
    py_setitem(fields, "outputs_2d_", PythonObject(False))
    if plan.kind == 0:
        # The caller's labels (member_pack_labeled), taken once: code c is labels[c].
        var labels = member_labels_take(fields, b)
        var flat = family_array(Python.tuple(rows), "<i4", context)
        var flat_address = Int(py=classical_address(flat))
        unsafe_memcpy(dest=_I32P(unsafe_from_address=flat_address),
            src=_I32P(unsafe_from_address=call.dst.data), count=rows)
        var names = PythonObject(None)
        if labels is not None:
            names = labels
            var label_view = admit_c_view(labels, "classes", context)
            if admitted_dtype(label_view, context) == "<i4":
                # int32 labels: the estimator's own fit keeps their values as
                # the int32 target words, `_label_names` None (_clf_fit).
                var decoded = admit_c_view(classical_decode_labels(labels, flat, labels_scope(model, context)),
                    "y", context)
                labels_cast(Int(py=classical_address(decoded)), admitted_dtype(decoded, context), flat_address,
                    "<i4", rows, context)
                _ = decoded  # keep the Python object alive through the native call above
                names = PythonObject(None)
            _ = label_view
        # Any other labels vote as their codes and predict decodes them
        # through `_label_names`, as the estimator's own fit does for str labels.
        py_setitem(fields, "_label_names", names)
        py_setitem(fields, "_y_cols", labels_reshape(flat, Python.tuple(1, rows), context))
        py_setitem(fields, "_classes_list", Python.list(classical_encode_labels(flat, context).classes))
    else:
        var yc = family_array(Python.tuple(1, rows), "<f4", context)
        unsafe_memcpy(dest=_F32P(unsafe_from_address=Int(py=classical_address(yc))),
            src=_F32P(unsafe_from_address=call.dst.data), count=rows)
        py_setitem(fields, "_y_cols", yc)


def neighbors_member_state(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan) raises -> Int:
    """UNPACK: a new member state holding the fitted attributes of `model`
    (one output). The caller frees it (`member_free`)."""
    var index_source = py_get(fields, "_index")
    var y_source = py_get(fields, "_y_cols")
    if index_source is None or y_source is None:
        raise Error("NotFittedError: this member is not fitted yet")
    if Int(py=y_source.shape[0]) != 1:
        raise Error("ValueError: mojolearn: a k-NN member has one output; this estimator was fitted on "
            + String(Int(py=y_source.shape[0])))
    var index = admit_x(index_source, 2, "_index", context)
    var rows = Int(py=index.shape[0])
    var cols = Int(py=index.shape[1])
    var targets = PythonObject(None)
    var classes = 0
    var flat = family_array(Python.tuple(rows), "<i4" if plan.kind == 0 else "<f4", context)
    var flat_address = Int(py=classical_address(flat))
    var y_address = Int(py=classical_address(y_source))
    if Int(py=y_source.size) != rows:
        raise Error("ValueError: mojolearn: the fitted targets and points differ in length")
    if plan.kind == 0:
        unsafe_memcpy(dest=_I32P(unsafe_from_address=flat_address), src=_I32P(unsafe_from_address=y_address),
            count=rows)
        var encoded = classical_encode_labels(flat, context)
        classes = len(encoded.classes)
        targets = encoded.codes
    else:
        unsafe_memcpy(dest=_F32P(unsafe_from_address=flat_address), src=_F32P(unsafe_from_address=y_address),
            count=rows)
        targets = flat
    _ = y_source
    var call = member_new(plan)
    try:
        call.block = NEIGHBORS_BLOCK_POINTS
        call.x = MemberMatrix(Int(py=classical_address(index)), rows, cols)
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = NEIGHBORS_BLOCK_TARGETS
        if plan.kind == 0:
            call.x = MemberMatrix(Int(py=classical_address(targets)), 1, rows, MEMBER_I32)
        else:
            call.x = MemberMatrix(Int(py=classical_address(targets)), 1, rows)
        call.classes = classes
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    except error:
        member_free(plan, call)
        raise error
    _ = index
    _ = targets
    _ = flat
    return call.state
