# SPDX-License-Identifier: Apache-2.0
"""The member boundary of the Sequence family (LSTM / GRU / RNN regressors,
kind 0, and classifiers, kind 1).

UNPACK / PACK only (GIL held, once per member estimator, never per fold):
  * `sequence_member_admit`: the estimator's own checks (`_rnn_cell`: the
    nonlinearity; `_rnn_optimizer`: the optimizer name and options, merged
    with PyTorch's defaults), its route through the sequence resolver
    (`xseq_route`), the model and output-helper entries with their
    providers' workspaces, the R^2 entry of the route's metrics image for a
    regressor, and the NEW words of core/sequence_member.mojo. A built-in
    `lr_schedule` (DATA: its class and settings) is admitted as the door's
    `opt_schedule` admits it and read once into the spec words; the member's
    fit runs its native series over the fold's own step count (lane
    mf-ens6). A schedule that is no built-in (a Python callback) is refused
    natively by `opt_schedule` in the door's words, which name the built-in
    native schedules.
  * `sequence_member_pack`: the fitted attributes the estimator's own fit
    writes (`params_` read-only, `loss_curve_`, `n_iter_`, `n_timesteps_`,
    `n_features_in_`, `n_outputs_`, `_y_1d` for a regressor); the stale
    params record is dropped. Labels: the caller's (`classes_`).
  * `sequence_member_state`: a state holding a fitted model's `params_`.
"""
from core.seq_member_rows import seq_rows_admit
from std.python import Python, PythonObject
from bindings.member_owner_unpack import MemberOwner
from bindings.native_loaded_module_door import NativeModules
from bindings.native_prepared_owner_unpack import prepared_drop
from bindings.py_balanced import py_get, py_item, py_setitem, py_type
from bindings.xseq_keys_unpack import xseq_keys
from bindings.xseq_native_common_door import (
    xseq_route, xseq_neural_provider, xseq_zeros, xseq_addr, xseq_int, xseq_truth, XseqStream,
)
from bindings.xseq_neural_native_door import (
    _attr, _entry, _opt_real, _rnn_cell, _rnn_optimizer, _RNN_PARAMS_SLOT, xseq_readonly_view,
)
from bindings.xseq_score_native_door import _metrics_key
from bindings.admitted_sequence_door import admit_nd_f32
from providers.python.xseq_public_lifecycle_pyapi_door import xseq_private_input
from providers.python.device_route_boundary_door import DeviceRouteOwner
from core.entry_table import EntryRef
from core.metrics_entry_table import metrics_r2_entry
from core.member_abi import (
    MemberPlan, MemberCall, MemberMatrix, member_call, member_new, member_free,
    MEMBER_OP_EXPORT, MEMBER_OP_IMPORT, MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_FAMILY, MEMBER_IMPORT_DONE, MEMBER_F64,
)
from sequence.xseq_neural_run import xseq_run_copy_f32
from bindings.optimizer_admission_door import opt_schedule
from bindings.optimizer_schedule_native_door import schedule_kind
from bindings.optimizer_public_door import schedule_spec_admitted


def sequence_member_admit(model: PythonObject, fields: PythonObject, kind: Int, context: PythonObject,
    b: PythonObject, mut owner: MemberOwner) raises -> Bool:
    """UNPACK: fill `owner.plan.route`, `owner.plan.params` and `owner.pins`
    from the unfitted estimator. False: a kind that is not the class's task."""
    var nk = xseq_keys()
    if Int(py=_attr(py_type(model), "_TASK")) != kind:
        return False
    # The door's `_rnn_rates` admission of a given schedule, read once.
    var lr_schedule = py_get(fields, nk[].k_lr_schedule)
    var sched_words = List[Float64](length=10, fill=0.0)
    if lr_schedule is not None:
        var optimizer_context = py_item(context, nk[].k_optimizer_context)
        opt_schedule(lr_schedule, optimizer_context)
        var spec = schedule_spec_admitted(lr_schedule)
        sched_words = [1.0, Float64(schedule_kind(lr_schedule, optimizer_context)), Float64(spec.kind),
            Float64(spec.a), Float64(spec.b), Float64(spec.c), spec.f0, spec.f1, spec.f2, spec.f3]
    var cell = _rnn_cell(model, fields)
    var optimizer = _rnn_optimizer(py_get(fields, nk[].k_optimizer), py_get(fields, nk[].k_optimizer_options), context)
    var held = xseq_route(model, fields, context)
    var model_provider = xseq_neural_provider(held.entries)
    var output_provider = xseq_neural_provider(held.output)
    var model_entry = _entry(model_provider)
    var output_entry = _entry(output_provider)
    var route = held.owner.downcast_value_ptr[DeviceRouteOwner]()[].route
    route.entry = EntryRef(model_entry.call, model_entry.workspace)
    route.second = EntryRef(output_entry.call, output_entry.workspace)
    route.has_second = True
    route.has_third = False
    if kind == 0:
        # The R^2 entry of the route's metrics image, read once here; without
        # that image the member's score refuses in words.
        try:
            var key = _metrics_key(held)
            route.third = EntryRef(metrics_r2_entry(key), 0)
            route.has_third = True
        except:
            route.has_third = False
    seq_rows_admit(route)
    owner.plan.route = route
    var random_state = py_get(fields, nk[].k_random_state)
    var fresh = random_state is None
    var seed = UInt64(0)
    if not fresh:
        seed = XseqStream(random_state, context).seed
    var words: List[Float64] = [
        Float64(kind), Float64(cell), Float64(xseq_int(py_item(fields, nk[].k_hidden_size))),
        Float64(xseq_int(py_item(fields, nk[].k_num_layers))), Float64(xseq_int(py_item(fields, nk[].k_batch_size))),
        Float64(xseq_int(py_item(fields, nk[].k_max_epochs))),
        1.0 if xseq_truth(py_item(fields, nk[].k_shuffle)) else 0.0,
        _opt_real(py_item(fields, nk[].k_learning_rate)), Float64(xseq_int(py_item(fields, nk[].k_predict_chunk))),
        Float64(optimizer.kind), Float64(optimizer.flags),
        optimizer.floats[0], optimizer.floats[1], optimizer.floats[2], optimizer.floats[3], optimizer.floats[4],
        optimizer.floats[5],
        Float64(Int(seed & 0xFFFFFFFF)), Float64(Int(seed >> 32)), 1.0 if fresh else 0.0,
    ]
    for i in range(len(sched_words)):  # small-loop(the ten schedule words)
        words.append(sched_words[i])
    owner.plan.params = words^
    owner.pins = Python.tuple(held.owner, model_provider, output_provider, held.metrics)
    return True


def sequence_member_pack(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan, state: Int) raises:
    """PACK: write the fitted attributes of state `state` into `model`, as
    the estimator's own fit does (its own predict, predict_proba and score
    work on the result)."""
    var nk = xseq_keys()
    var call = MemberCall(plan.route, state)
    call.block = MEMBER_BLOCK_SCALARS
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var scalars = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call.dst.data)
    var n_iter = Int(scalars[0])
    var steps_t = Int(scalars[1])
    var features = Int(scalars[2])
    var outputs = Int(scalars[3])
    call.block = MEMBER_BLOCK_FAMILY
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var params = xseq_zeros(Python.tuple(call.dst.cols), context)
    xseq_run_copy_f32(Int(py=xseq_addr(params)), call.dst.data, call.dst.cols)
    call.block = MEMBER_BLOCK_FAMILY + 1
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var losses = xseq_zeros(Python.tuple(call.dst.cols), context)
    xseq_run_copy_f32(Int(py=xseq_addr(losses)), call.dst.data, call.dst.cols)
    var mods = NativeModules()
    prepared_drop(mods, _RNN_PARAMS_SLOT, model)  # the new params_ is a new object
    py_setitem(fields, nk[].k_n_features_in_, features)
    py_setitem(fields, nk[].k_n_outputs_, outputs)
    if plan.kind == 0:
        # The member's y is a (rows, outputs) view: one output predicts 1-D.
        py_setitem(fields, nk[].k__y_1d, outputs == 1)
    py_setitem(fields, nk[].k_params_, xseq_readonly_view(params))
    py_setitem(fields, nk[].k_loss_curve_, losses)
    py_setitem(fields, nk[].k_n_iter_, n_iter)
    py_setitem(fields, nk[].k_n_timesteps_, steps_t)


def sequence_member_state(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan) raises -> Int:
    """UNPACK: a new member state holding the fitted `params_` of `model`.
    The caller frees it (`member_free`)."""
    var nk = xseq_keys()
    var source = py_get(fields, nk[].k_params_)
    if source is None:
        raise Error("NotFittedError: this member is not fitted yet")
    var params = admit_nd_f32(xseq_private_input(source, context, "params_"), "params_", False, context)
    var classes = 0
    if plan.kind == 1:
        classes = len(py_item(fields, nk[].k_classes_))
    var scalars: List[Float64] = [
        Float64(xseq_int(py_get(fields, nk[].k_n_iter_, PythonObject(0)))),
        Float64(xseq_int(py_get(fields, nk[].k_n_timesteps_, PythonObject(0)))),
        Float64(xseq_int(py_item(fields, nk[].k_n_features_in_))),
        Float64(xseq_int(py_item(fields, nk[].k_n_outputs_))), Float64(classes),
    ]
    var call = member_new(plan)
    try:
        call.block = MEMBER_BLOCK_FAMILY
        call.x = MemberMatrix(Int(py=xseq_addr(params)), 1, Int(py=params.size))
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
    _ = len(scalars)
    return call.state
