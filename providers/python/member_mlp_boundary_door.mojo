# SPDX-License-Identifier: Apache-2.0
"""The member boundary of the Mlp family (MLPRegressor, MLPClassifier): the
family's place in `bindings/member_boundary_door.mojo` (recipe:
MEMBER-ABI-V2.md; RUN: core/mlp_member.mojo). Kind 0 is MLPRegressor,
kind 1 MLPClassifier (`_mojolearn_member = (MEMBER_FAMILY_MLP, kind)` on
each class). `context` is the sequence family's context (the
`_native_xseq_bridge` context the class's own doors receive).

  * `mlp_member_admit` (UNPACK, once per member): the estimator's own
    parameter checks (`_mlp_check`, `_mlp_hidden`), its route through the
    sequence resolver (`xseq_route`: a hit is one record read), the model and
    output-helper entries with their providers' retained workspaces in the
    route's first two slots, the R^2 entry of the route's metrics image in
    the third, and the NEW words of core/mlp_member.mojo. The seed word is
    the estimator's seeded stream (`XseqStream`): `random_state`, or ONE
    OS-entropy draw per admitted member when it is None (every state made
    from this plan then starts from that seed).
  * `mlp_member_pack` (PACK): the fitted attributes the estimator's own fit
    writes (`coefs_`, `intercepts_`, the curve and the counters), from the
    member state's blocks; `classes_` is the caller's PACK.
  * `mlp_member_state` (UNPACK): a member state from a fitted estimator.

SERIAL BY NATURE: one member's admission or publication (record reads and
one copy per fitted layer); a caller runs its members in parallel.
"""
from std.python import Python, PythonObject
from bindings.member_owner_unpack import MemberOwner
from bindings.native_loaded_module_door import NativeModules
from bindings.native_prepared_owner_unpack import prepared_drop
from bindings.py_balanced import py_get, py_item, py_setitem, py_call
from bindings.xseq_keys_unpack import xseq_keys
from bindings.xseq_native_common_door import (
    xseq_route, xseq_neural_provider, xseq_zeros, xseq_addr, xseq_curve_list, xseq_int, xseq_float, xseq_truth,
    XseqStream,
)
from bindings.xseq_neural_native_door import (
    _mlp_check, _mlp_hidden, _opt_real, _entry, _mlp_pack, _mlp_layer_copy, _MLP_PACKED_SLOT, xseq_readonly_view,
)
from bindings.xseq_score_native_door import _metrics_key
from providers.python.device_route_boundary_door import DeviceRouteOwner
from core.entry_table import EntryRef
from core.metrics_entry_table import metrics_r2_entry
from core.member_abi import (
    MemberPlan, MemberCall, MemberMatrix, member_call, member_new, member_free,
    MEMBER_OP_EXPORT, MEMBER_OP_IMPORT, MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_FAMILY, MEMBER_IMPORT_DONE, MEMBER_F64,
)
from sequence.xseq_neural_run import xseq_run_copy_f32, xseq_run_mlp_out_name


def mlp_member_admit(model: PythonObject, fields: PythonObject, kind: Int, context: PythonObject,
    b: PythonObject, mut owner: MemberOwner) raises -> Bool:
    """UNPACK: fill `owner.plan.route`, `owner.plan.params` and `owner.pins`
    from the unfitted estimator."""
    var nk = xseq_keys()
    _mlp_check(fields, context)
    var hidden = _mlp_hidden(fields, context)
    var held = xseq_route(model, fields, context)
    var model_provider = xseq_neural_provider(held.entries)
    var output_provider = xseq_neural_provider(held.output)
    var model_entry = _entry(model_provider)
    var output_entry = _entry(output_provider)
    # The device, tier and vendor words of the resolved image, as the route
    # record `xseq_route` keeps them; its glue entry is not a member slot.
    var route = held.owner.downcast_value_ptr[DeviceRouteOwner]()[].route
    route.entry = EntryRef(model_entry.call, model_entry.workspace)
    route.second = EntryRef(output_entry.call, output_entry.workspace)
    route.has_second = True
    if kind == 0:
        # The R^2 entry of the route's metrics image, read once here (the
        # estimator's own score door reads it per call). An installation
        # without that image leaves the slot empty: the member's score then
        # refuses in words and fit / predict are unaffected.
        try:
            var key = _metrics_key(held)
            route.third = EntryRef(metrics_r2_entry(key), 0)
            route.has_third = True
        except:
            route.has_third = False
    owner.plan.route = route
    var activation = xseq_int(py_item(context, nk[].k_xseq_mlp_activations)[py_item(fields, nk[].k_activation)])
    var solver = xseq_int(py_item(context, nk[].k_xseq_mlp_solvers)[py_item(fields, nk[].k_solver)])
    var rate = xseq_int(py_item(context, nk[].k_xseq_mlp_rates)[py_item(fields, nk[].k_learning_rate)])
    var requested = py_item(fields, nk[].k_batch_size)
    var batch = 0
    if not Bool(py=requested == PythonObject("auto")):
        batch = max(1, xseq_int(requested))
    var stream = XseqStream(py_get(fields, nk[].k_random_state), context)
    var words: List[Float64] = [
        Float64(kind), Float64(activation), Float64(solver), Float64(rate),
        _opt_real(py_item(fields, nk[].k_learning_rate_init)), _opt_real(py_item(fields, nk[].k_beta_1)),
        _opt_real(py_item(fields, nk[].k_beta_2)), _opt_real(py_item(fields, nk[].k_epsilon)),
        _opt_real(py_item(fields, nk[].k_momentum)), _opt_real(py_item(fields, nk[].k_power_t)),
        _opt_real(py_item(fields, nk[].k_alpha)), _opt_real(py_item(fields, nk[].k_tol)),
        1.0 if xseq_truth(py_item(fields, nk[].k_nesterovs_momentum)) else 0.0,
        1.0 if xseq_truth(py_item(fields, nk[].k_shuffle)) else 0.0,
        Float64(batch), Float64(xseq_int(py_item(fields, nk[].k_max_iter))),
        Float64(xseq_int(py_item(fields, nk[].k_n_iter_no_change))),
        Float64(Int(stream.seed & 0xFFFFFFFF)), Float64(Int(stream.seed >> 32)), Float64(len(hidden)),
    ]
    for l in range(len(hidden)):  # driver-loop: over hidden layer widths
        words.append(Float64(hidden[l]))
    owner.plan.params = words^
    # The route owner pins the images; the providers own the workspaces the
    # route's first two slots carry.
    owner.pins = Python.tuple(held.owner, model_provider, output_provider, held.metrics)
    return True


def mlp_member_pack(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan, state: Int) raises:
    """PACK: write the fitted attributes of state `state` into `model`, as
    the estimator's own fit does (the same names, so its own predict,
    predict_proba and score work on the result)."""
    var nk = xseq_keys()
    var call = MemberCall(plan.route, state)
    call.block = MEMBER_BLOCK_SCALARS
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var scalars = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call.dst.data)
    var n_iter = Int(scalars[0])
    var loss = scalars[1]
    var best = scalars[2]
    var t = Int(scalars[3])
    var out_act = Int(scalars[4])
    var outputs = Int(scalars[5])
    var features = Int(scalars[7])
    call.block = MEMBER_BLOCK_FAMILY + 1
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var widths = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call.dst.data)
    var sizes = List[Int]()
    for l in range(call.dst.cols):  # driver-loop: over network layers
        sizes.append(Int(widths[l]))
    call.block = MEMBER_BLOCK_FAMILY
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var total = call.dst.cols
    var params = xseq_zeros(Python.tuple(total), context)
    xseq_run_copy_f32(Int(py=xseq_addr(params)), call.dst.data, total)
    var coefs = Python.list()
    var intercepts = Python.list()
    var offset = 0
    for l in range(len(sizes) - 1):  # driver-loop: over network layers
        var fi = sizes[l]
        var fo = sizes[l + 1]
        _ = py_call(coefs.append, xseq_readonly_view(_mlp_layer_copy(params, offset, Python.tuple(fi, fo), fi * fo,
            context)))
        offset += fi * fo
        _ = py_call(intercepts.append, xseq_readonly_view(_mlp_layer_copy(params, offset, Python.tuple(fo), fo,
            context)))
        offset += fo
    _ = params
    call.block = MEMBER_BLOCK_FAMILY + 2
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var curve = xseq_zeros(Python.tuple(max(n_iter, 1)), context)
    xseq_run_copy_f32(Int(py=xseq_addr(curve)), call.dst.data, min(n_iter, call.dst.cols))
    var mods = NativeModules()
    prepared_drop(mods, _MLP_PACKED_SLOT, model)  # a refit publishes new coefs_ / intercepts_
    py_setitem(fields, nk[].k_n_outputs_, outputs)
    py_setitem(fields, nk[].k_n_layers_, len(sizes))
    py_setitem(fields, nk[].k_out_activation_, PythonObject(xseq_run_mlp_out_name(out_act)))
    py_setitem(fields, nk[].k_n_features_in_, features)
    if plan.kind == 1:
        py_setitem(fields, nk[].k__LOSS_CODE, 1 if outputs == 1 else 2)
    else:
        # The member's y is a (rows, outputs) view: one output predicts 1-D.
        py_setitem(fields, nk[].k__y_1d, outputs == 1)
    py_setitem(fields, nk[].k_coefs_, coefs)
    py_setitem(fields, nk[].k_intercepts_, intercepts)
    py_setitem(fields, nk[].k_n_iter_, n_iter)
    py_setitem(fields, nk[].k_loss_curve_, xseq_curve_list(curve, n_iter))
    py_setitem(fields, nk[].k_loss_, loss)
    py_setitem(fields, nk[].k_best_loss_, best)
    py_setitem(fields, nk[].k_t_, t)
    _ = curve


def mlp_member_state(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan) raises -> Int:
    """UNPACK: a new member state holding the fitted attributes of `model`.
    The caller frees it (`member_free`)."""
    var nk = xseq_keys()
    var coefs = py_get(fields, nk[].k_coefs_)
    if coefs is None:
        raise Error("NotFittedError: this member is not fitted yet")
    var intercepts = py_item(fields, nk[].k_intercepts_)
    var columns = xseq_int(py_item(fields, nk[].k_n_features_in_))
    # The estimator's own admission and packing of its fitted layers.
    var packed = _mlp_pack(coefs, intercepts, columns, context)
    var params = packed[0]
    var hidden = packed[1]
    var outputs = Int(py=packed[3])
    var widths: List[Float64] = [Float64(columns)]
    for l in range(len(hidden)):  # driver-loop: over hidden layer widths
        widths.append(Float64(Int(py=hidden[l])))
    widths.append(Float64(outputs))
    var out_act = xseq_int(py_item(context, nk[].k_xseq_mlp_outputs)[py_item(fields, nk[].k_out_activation_)])
    var classes = 0
    if plan.kind == 1:
        var labels = py_get(fields, nk[].k_classes_)
        classes = len(labels) if labels is not None else (2 if outputs == 1 else outputs)
    var n_iter = xseq_int(py_get(fields, nk[].k_n_iter_, PythonObject(0)))
    var scalars: List[Float64] = [
        Float64(n_iter), xseq_float(py_get(fields, nk[].k_loss_, PythonObject(0.0))),
        xseq_float(py_get(fields, nk[].k_best_loss_, PythonObject(0.0))),
        Float64(xseq_int(py_get(fields, nk[].k_t_, PythonObject(0)))), Float64(out_act), Float64(outputs),
        Float64(classes), Float64(columns),
    ]
    var call = member_new(plan)
    try:
        call.block = MEMBER_BLOCK_FAMILY
        call.x = MemberMatrix(Int(py=xseq_addr(params)), 1, Int(py=params.size))
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = MEMBER_BLOCK_FAMILY + 1
        call.x = MemberMatrix(Int(widths.unsafe_ptr()), 1, len(widths), MEMBER_F64)
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = MEMBER_BLOCK_SCALARS
        call.x = MemberMatrix(Int(scalars.unsafe_ptr()), 1, len(scalars), MEMBER_F64)
        call.classes = classes
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    except error:
        member_free(plan, call)
        raise error
    _ = params
    _ = packed
    _ = len(widths)
    _ = len(scalars)
    return call.state
