# SPDX-License-Identifier: Apache-2.0
"""The member boundary of the Neighbors family: KNeighborsClassifier (kind 0)
and KNeighborsRegressor (kind 1), `_mojolearn_member = (11, kind)` on each
class (MEMBER_FAMILY_NEIGHBORS; RUN: core/neighbors_member.mojo).
Lane mf-member8: NearestNeighbors (kind 2) and RadiusNeighbors (kind 3), fit
and export only (no predict): admitted with their fit's refusals
(`_check_refusals`; `radius_refusals`, which also writes `_metric_value` /
`_metric_arg` as the radius fit does), packed with `_index`, `n_samples_fit_`,
`n_features_in_` (NearestNeighbors: a fresh `_resident_holder` after the
release, as its fit does), imported from `_index`.

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
    estimator (one output or several, lane w3f3-k); the classifier's stored
    labels are encoded to codes by the native encoder, per output (the class
    count is its class list's length). A multi-output PACK writes
    `_y_cols` outputs x rows and `outputs_2d_` as the estimator's own fit.

SERIAL BY NATURE: one member's admission or publication (a few record reads
and block copies); a caller runs its members in parallel.
"""
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from bindings.admitted_input_unpack import admit_x, admit_c_view, admitted_dtype
from bindings.classical_arrays_door import classical_array, classical_address
from bindings.classical_labels_door import (
    classical_encode_labels, classical_decode_labels, labels_cast, labels_reshape, labels_label_kind,
)
from bindings.label_admission_native_pack import classifier_label_list
from bindings.labels_descriptor_boundary_door import labels_scope
from bindings.family_common_pack import family_array, family_field
from bindings.member_owner_unpack import MemberOwner, member_labels_take, member_host_only
from bindings.neighbors_knn_native_door import _check_refusals, _encode_column, knn_dist_words, knn_ops_key, knn_release
from bindings.py_balanced import py_call, py_get, py_setitem
from core.device_route import DeviceRoute, DEVICE_CPU, DEVICE_GPU
from core.entry_table import EntryRef, entry_ref_word
from core.member_abi import (
    MemberPlan, MemberCall, MemberMatrix, member_call, member_new, member_free,
    MEMBER_OP_EXPORT, MEMBER_OP_IMPORT, MEMBER_IMPORT_DONE, MEMBER_I32,
)
from core.neighbors_entry_table import NB_OPERATION_COUNT, neighbors_ops_entry
from core.neighbors_member import NEIGHBORS_BLOCK_POINTS, NEIGHBORS_BLOCK_TARGETS
from providers.python.estimator_binding_pyapi_door import estimator_binding, estimator_binding_host
from providers.python.radius_public_lifecycle_pyapi_door import radius_refusals

comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]


def neighbors_member_admit(model: PythonObject, fields: PythonObject, kind: Int, context: PythonObject,
    b: PythonObject, mut owner: MemberOwner) raises -> Bool:
    """UNPACK: fill `owner.plan.route`, `owner.plan.params` and `owner.pins`
    from the unfitted estimator."""
    if kind < 0 or kind > 3:
        raise Error("ValueError: mojolearn: unknown k-NN member kind " + String(kind))
    var metric_word = Float64(-1.0)
    var metric_arg = Float64(2.0)
    var weights_word = Float64(0.0)
    var k = 0
    var tile = 0
    if kind == 3:
        # RadiusNeighbors.fit's refusals (by name, in its order); they write
        # `_metric_value` / `_metric_arg` into the fields as its fit does.
        radius_refusals(model, context)
    elif kind == 2:
        # NearestNeighbors.fit's refusals; the resolved brute metric (-1 for
        # algorithm='rbc') decides its one input check at fit.
        metric_word = Float64(_check_refusals(model, context))
        k = Int(py=family_field(fields, "n_neighbors"))
        tile = Int(py=family_field(fields, "query_tile"))
    else:
        # The estimator's own refusals of its fit (by name, in its order).
        _ = _check_refusals(model, context)
        var dist = knn_dist_words(model, context)
        metric_word = Float64(dist.metric)
        metric_arg = dist.metric_arg
        weights_word = Float64(dist.weights)
        var k_source = family_field(fields, "n_neighbors")
        var tile_source = family_field(fields, "query_tile")
        k = Int(py=k_source)
        tile = Int(py=tile_source)
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
    var words: List[Float64] = [Float64(kind), Float64(k), metric_word, metric_arg, weights_word, Float64(tile)]
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
    if plan.kind == 2 or plan.kind == 3:
        # Lane mf-member8: `_nn_fit` / `radius_fit`'s publications (the
        # points, their two counts; NearestNeighbors releases the previous
        # device index and starts a fresh holder, as its fit and
        # `neighbors_public_finish` do).
        if plan.kind == 2:
            knn_release(model)
            py_setitem(fields, "_resident_holder", Python.dict())
        py_setitem(fields, "_index", index)
        py_setitem(fields, "n_samples_fit_", PythonObject(rows))
        py_setitem(fields, "n_features_in_", PythonObject(cols))
        return
    call.block = NEIGHBORS_BLOCK_TARGETS
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    # A refit drops the device copy of the previous index, as the fit does.
    knn_release(model)
    py_setitem(fields, "_resident_holder", Python.dict())
    py_setitem(fields, "_index", index)
    py_setitem(fields, "n_samples_fit_", PythonObject(rows))
    py_setitem(fields, "n_features_in_", PythonObject(cols))
    # THE 2-D y (lane w3f3-k): the targets block is outputs x rows.
    var outputs = call.dst.rows
    py_setitem(fields, "outputs_2d_", PythonObject(outputs > 1))
    if outputs > 1:
        _pack_outputs(model, fields, context, b, plan, call.dst.data, outputs, rows)
        return
    if plan.kind == 0:
        # The caller's labels (member_pack_labeled), taken once: code c is labels[c].
        # LANE w3f7-c: as a plain list. A classifier member's caller publishes
        # them as the connected ClassLabels (msel_member_units_door `_member_fit`:
        # `_public_labels`; a meta member's outputs likewise), a list SUBCLASS
        # the shared admission refuses in the Sequential words, so the view
        # below refused every refit PACK of a KNeighborsClassifier in a search
        # (test_mf_rows8d seed_as_given, the MultiOutputClassifier(KNN) search
        # row; verify lane public-scorer-factories). The 2-D path
        # (`_pack_outputs`) already copied each output's list this way
        # (69b6398f72); the estimator's own fit keeps `_label_names` an exact
        # list, which predict admits. Cause 2a6cf4325f (this view), exposed by
        # 7626d7be07 (a refused pack raises; it fell back before). One native
        # list copy per pack (classes-sized), only for ClassLabels.
        var labels = classifier_label_list(member_labels_take(fields, b), context)
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


def _pack_outputs(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan, data: Int, outputs: Int, rows: Int) raises:
    """PACK of a multi-output state (lane w3f3-k): `_y_cols` outputs x rows
    as the estimator's own fit keeps them (`_reg_fit`: the float32 targets;
    `_clf_fit`: the int32 labels, each output's codes decoded through the
    caller's labels of that output (`member_pack_labeled` hands one label
    list per output; without labels the codes are the labels), and
    `_classes_list` per output from the native encoder, `_label_names`
    None; labels of another kind keep the codes and `_label_names`, below).
    `data` is the exported targets block."""
    var count = outputs * rows
    if plan.kind != 0:
        var targets = family_array(Python.tuple(outputs, rows), "<f4", context)
        unsafe_memcpy(dest=_F32P(unsafe_from_address=Int(py=classical_address(targets))),
            src=_F32P(unsafe_from_address=data), count=count)
        py_setitem(fields, "_y_cols", targets)
        return
    var labels = member_labels_take(fields, b)
    var yc = family_array(Python.tuple(outputs, rows), "<i4", context)
    var yc_address = Int(py=classical_address(yc))
    unsafe_memcpy(dest=_I32P(unsafe_from_address=yc_address), src=_I32P(unsafe_from_address=data), count=count)
    var classes = Python.list()
    # Labels of any other kind than integers (str, bool, real; lane w3f3-k)
    # vote as their codes, and predict decodes them per output through
    # `_label_names` (one label list per output), as the 1-D fit does for
    # str labels (bindings/neighbors_knn_native_door.mojo `_clf_fit`,
    # `_decode_output_rows`). Each output's list is read once here.
    var names = PythonObject(None)
    if labels is not None:
        var lists = Python.list()
        var integral = True
        for j in range(outputs):  # driver-loop: over the outputs (their labels' kind, once per pack)
            var own = classifier_label_list(labels[j], context)
            if labels_label_kind(own, context) != "int":
                integral = False
            _ = py_call(lists.append, own)
        if not integral:
            names = lists
            labels = PythonObject(None)
    if labels is not None:
        var scope = labels_scope(model, context)
        for j in range(outputs):  # driver-loop: over the outputs (one decode each)
            var column = family_array(Python.tuple(rows), "<i4", context)
            var column_address = Int(py=classical_address(column))
            unsafe_memcpy(dest=_I32P(unsafe_from_address=column_address),
                src=_I32P(unsafe_from_address=yc_address + 4 * j * rows), count=rows)
            var own = labels[j]
            var decoded = admit_c_view(classical_decode_labels(own, column, scope), "y", context)
            labels_cast(Int(py=classical_address(decoded)), admitted_dtype(decoded, context),
                yc_address + 4 * j * rows, "<i4", rows, context)
            _ = decoded  # keep the Python object alive through the native call above
            _ = column  # keep the Python object alive through the native call above
    for j in range(outputs):  # driver-loop: over the outputs (one native encoding each)
        _ = py_call(classes.append, _encode_column(yc, j, rows, context))
    _ = yc  # keep the Python object alive through the native calls above
    py_setitem(fields, "_label_names", names)
    py_setitem(fields, "_y_cols", yc)
    py_setitem(fields, "_classes_list", classes)


def neighbors_member_state(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan) raises -> Int:
    """UNPACK: a new member state holding the fitted attributes of `model`
    (one output or several). The caller frees it (`member_free`)."""
    var index_source = py_get(fields, "_index")
    if plan.kind == 2 or plan.kind == 3:
        # Lane mf-member8: the points are the whole fitted state.
        if index_source is None:
            raise Error("NotFittedError: this member is not fitted yet")
        var points = admit_x(index_source, 2, "_index", context)
        var call = member_new(plan)
        try:
            call.block = NEIGHBORS_BLOCK_POINTS
            call.x = MemberMatrix(Int(py=classical_address(points)), Int(py=points.shape[0]), Int(py=points.shape[1]))
            call.flags = MEMBER_IMPORT_DONE
            _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        except error:
            member_free(plan, call)
            raise error
        _ = points
        return call.state
    var y_source = py_get(fields, "_y_cols")
    if index_source is None or y_source is None:
        raise Error("NotFittedError: this member is not fitted yet")
    # THE 2-D y (lane w3f3-k): one row of `_y_cols` per output.
    var outputs = Int(py=y_source.shape[0])
    var index = admit_x(index_source, 2, "_index", context)
    var rows = Int(py=index.shape[0])
    var cols = Int(py=index.shape[1])
    var targets = PythonObject(None)
    var classes = 0
    var counts = List[Int]()
    var flat = family_array(Python.tuple(rows), "<i4" if plan.kind == 0 else "<f4", context)
    var flat_address = Int(py=classical_address(flat))
    var y_address = Int(py=classical_address(y_source))
    if outputs < 1 or Int(py=y_source.size) != rows * outputs:
        raise Error("ValueError: mojolearn: the fitted targets and points differ in length")
    if plan.kind == 0 and outputs > 1:
        # Each output's stored labels encoded to codes by the native encoder,
        # its class count the length of its class list.
        targets = family_array(Python.tuple(outputs, rows), "<i4", context)
        var block_address = Int(py=classical_address(targets))
        for j in range(outputs):  # driver-loop: over the outputs (one native encoding each)
            unsafe_memcpy(dest=_I32P(unsafe_from_address=flat_address),
                src=_I32P(unsafe_from_address=y_address + 4 * j * rows), count=rows)
            var encoded = classical_encode_labels(flat, context)
            counts.append(len(encoded.classes))
            var codes = admit_c_view(encoded.codes, "y", context)
            labels_cast(Int(py=classical_address(codes)), admitted_dtype(codes, context),
                block_address + 4 * j * rows, "<i4", rows, context)
            _ = codes  # keep the Python object alive through the native call above
            _ = flat  # keep the Python object alive through the native call above
    elif plan.kind == 0:
        unsafe_memcpy(dest=_I32P(unsafe_from_address=flat_address), src=_I32P(unsafe_from_address=y_address),
            count=rows)
        var encoded = classical_encode_labels(flat, context)
        classes = len(encoded.classes)
        targets = encoded.codes
    elif outputs > 1:
        targets = family_array(Python.tuple(outputs, rows), "<f4", context)
        unsafe_memcpy(dest=_F32P(unsafe_from_address=Int(py=classical_address(targets))),
            src=_F32P(unsafe_from_address=y_address), count=rows * outputs)
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
            call.x = MemberMatrix(Int(py=classical_address(targets)), outputs, rows, MEMBER_I32)
        else:
            call.x = MemberMatrix(Int(py=classical_address(targets)), outputs, rows)
        call.classes = classes
        call.class_counts = Int(counts.unsafe_ptr()) if len(counts) > 0 else 0
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    except error:
        member_free(plan, call)
        raise error
    _ = counts^  # the class counts the import read
    _ = index
    _ = targets
    _ = flat
    return call.state
