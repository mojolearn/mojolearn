# SPDX-License-Identifier: Apache-2.0
"""The member boundary of the Cnn family (CNNClassifier, kind 0).

UNPACK / PACK only (GIL held, once per member estimator, never per fold):
  * `cnn_member_admit`: the estimator's own admission of its settings
    (`_classifier_fields`, the checks `cnnclf_initialize` ran at
    construction), its route through the classifier's resolver
    (`_xcnn_route`: a hit is one record read) and the x_sequence model
    descriptor entry of that route's label provider, with the provider's
    retained workspace, in the route's first slot; the NEW words of
    core/cnn_member.mojo. The layout seed is `random_state` (0 when None,
    as `_build`); the training seed is the estimator's seeded stream
    (`XseqStream`) of an integer `random_state`, and when it is None the
    fresh-seed word is set and the member draws OS entropy at every fit.
  * `cnn_member_pack`: the fitted attributes the estimator's own fit writes
    (`cnnclf_fit`): the layer stack holding the trained arrays (read-only
    after training, as the door publishes them), `losses_`, `loss_curve_`,
    `n_features_in_`; the stale resident predict record is dropped. The
    labels are the caller's (`classes_`, written by the dispatcher).
  * `cnn_member_state`: a state holding a fitted classifier's learned arrays
    (no loss records: a state made this way exports empty loss blocks).
"""
from std.python import Python, PythonObject
from bindings.member_owner_unpack import MemberOwner
from bindings.native_loaded_module_door import NativeModules
from bindings.native_prepared_owner_unpack import prepared_drop
from bindings.py_balanced import py_item, py_call, py_setitem
from bindings.xcnn_keys_unpack import xcnn_keys
from bindings.xcnn_common_door import _xcnn_route, xcnn_addr, xcnn_readonly
from bindings.xcnn_classifier_native_door import (
    _b, _classifier_fields, _engine, _empty, _layout, _param_arrays, _PREDICT_SLOT, cnnclf_entry, cnnclf_publish_layers,
)
from bindings.xseq_native_common_door import XseqStream, xseq_neural_provider
from providers.python.device_route_boundary_door import DeviceRouteOwner
from core.entry_table import EntryRef
from core.seq_member_rows import seq_rows_admit
from core.member_abi import (
    MemberPlan, MemberCall, MemberMatrix, member_call, member_new, member_free,
    MEMBER_OP_EXPORT, MEMBER_OP_IMPORT, MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_FAMILY, MEMBER_IMPORT_DONE, MEMBER_F64,
)
from sequence.xseq_neural_run import xseq_run_copy_f32
from x_cnn.classifier_run import cnnclf_run_optimizer_word, cnnclf_run_features


def cnn_member_admit(model: PythonObject, fields: PythonObject, kind: Int, context: PythonObject,
    b: PythonObject, mut owner: MemberOwner) raises -> Bool:
    """UNPACK: fill `owner.plan.route`, `owner.plan.params` and `owner.pins`
    from the unfitted estimator."""
    if kind != 0:
        return False
    var nk = xcnn_keys()
    var own = _classifier_fields(model, context)
    var held = _xcnn_route(model, own, context)
    var labels = _engine(held.downcast_value_ptr[DeviceRouteOwner]()[].cls)
    var provider = xseq_neural_provider(labels)
    var entry = cnnclf_entry(provider)
    var route = held.downcast_value_ptr[DeviceRouteOwner]()[].route
    route.entry = EntryRef(entry.call, entry.workspace)
    # Lane vdf-scope-neural2: the GPU image's member store (THE RESIDENT ROUTE).
    seq_rows_admit(route)
    owner.plan.route = route
    var shape = py_item(own, nk[].k_input_shape)
    var channels = py_item(own, nk[].k_conv_channels)
    var rs = py_item(own, nk[].k_random_state)
    var fresh = rs is None
    var layout_seed = 0 if fresh else Int(py=py_call(_b().int, rs))
    var fit_seed = UInt64(0)
    if not fresh:
        fit_seed = XseqStream(rs, context).seed
    var betas = py_item(own, nk[].k_betas)
    var layout_bits = UInt64(layout_seed)
    var words: List[Float64] = [
        0.0, Float64(Int(py=shape[0])), Float64(Int(py=shape[1])), Float64(Int(py=shape[2])),
        Float64(Int(py=py_item(own, nk[].k_kernel_size))), Float64(Int(py=py_item(own, nk[].k_pool_size))),
        Float64(Int(py=py_item(own, nk[].k_batch_size))), Float64(Int(py=py_item(own, nk[].k_max_iter))),
        1.0 if Bool(py=py_item(own, nk[].k_shuffle)) else 0.0,
        Float64(cnnclf_run_optimizer_word(String(py=py_item(own, nk[].k_optimizer)))),
        1.0 if Bool(py=py_item(own, nk[].k_nesterov)) else 0.0,
        Float64(py=py_item(own, nk[].k_learning_rate)), Float64(py=py_item(own, nk[].k_momentum)),
        Float64(py=betas[0]), Float64(py=betas[1]), Float64(py=py_item(own, nk[].k_eps)),
        Float64(py=py_item(own, nk[].k_weight_decay)), Float64(py=py_item(own, nk[].k_dampening)),
        Float64(Int(layout_bits & 0xFFFFFFFF)), Float64(Int(layout_bits >> 32)),
        1.0 if fresh else 0.0,
        Float64(Int(fit_seed & 0xFFFFFFFF)), Float64(Int(fit_seed >> 32)), Float64(len(channels)),
    ]
    for i in range(len(channels)):  # driver-loop: over the conv blocks (model structure)
        words.append(Float64(Int(py=channels[i])))
    owner.plan.params = words^
    # The route owner pins the images; the provider owns the workspace the
    # route's first slot carries.
    owner.pins = Python.tuple(held, labels, provider)
    return True


def _f32_list(data: Int, count: Int, context: PythonObject) raises -> PythonObject:
    """A Python list of `count` float32 words (as the door's `.tolist()`)."""
    if count == 0:
        return _b().list()
    var buffer = _empty(Python.tuple(count), "float32", context)
    xseq_run_copy_f32(Int(py=xcnn_addr(buffer)), data, count)
    return py_call(buffer.tolist)


def cnn_member_pack(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan, state: Int) raises:
    """PACK: write the fitted attributes of state `state` into `model`, as
    the estimator's own fit does (the same names, so its own predict and
    predict_proba work on the result)."""
    var nk = xcnn_keys()
    var call = MemberCall(plan.route, state)
    call.block = MEMBER_BLOCK_SCALARS
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var scalars = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call.dst.data)
    var k = Int(scalars[0])
    var mods = NativeModules()
    prepared_drop(mods, _PREDICT_SLOT, model)  # a refit publishes new weights
    var layout = _layout(fields, k)
    var trained = Int(py=py_item(fields, nk[].k_max_iter)) > 0
    var arrays = Python.list()
    for i in range(len(layout.shapes)):  # driver-loop: over the learned arrays (model structure)
        var dims = Python.list()
        for j in range(len(layout.shapes[i])):  # small-loop(one array's dimensions)
            _ = py_call(dims.append, layout.shapes[i][j])
        var array = _empty(py_call(_b().tuple, dims), "float32", context)
        call.block = MEMBER_BLOCK_FAMILY + 2 + i
        _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
        if call.dst.cols != layout.counts[i]:
            raise Error("ValueError: the CNN member state does not have this model's layers")
        xseq_run_copy_f32(Int(py=xcnn_addr(array)), call.dst.data, layout.counts[i])
        # Trained words are published read-only (the resident predict
        # session keys on them), as the estimator's own fit publishes them.
        _ = py_call(arrays.append, xcnn_readonly(array, PythonObject(None), context) if trained else array)
    cnnclf_publish_layers(fields, layout, arrays, context)
    call.block = MEMBER_BLOCK_FAMILY + 1
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    py_setitem(fields, nk[].k_losses_, _f32_list(call.dst.data, call.dst.cols, context))
    call.block = MEMBER_BLOCK_FAMILY
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    py_setitem(fields, nk[].k_loss_curve_, _f32_list(call.dst.data, call.dst.cols, context))
    var shape = py_item(fields, nk[].k_input_shape)
    py_setitem(fields, nk[].k_n_features_in_, PythonObject(cnnclf_run_features(
        [Int(py=shape[0]), Int(py=shape[1]), Int(py=shape[2])])))


def cnn_member_state(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan) raises -> Int:
    """UNPACK: a new member state holding the fitted arrays of `model`.
    The caller frees it (`member_free`)."""
    var nk = xcnn_keys()
    var given = _classifier_fields(model, context, True)
    var classes = py_item(given, nk[].k_classes_)
    var k = len(classes)
    var arrays = _param_arrays(given, context)
    var shape = py_item(given, nk[].k_input_shape)
    var features = cnnclf_run_features([Int(py=shape[0]), Int(py=shape[1]), Int(py=shape[2])])
    var scalars: List[Float64] = [Float64(k), Float64(features), Float64(len(py_item(given, nk[].k_conv_channels))),
        Float64(len(arrays))]
    var call = member_new(plan)
    try:
        call.block = MEMBER_BLOCK_SCALARS
        call.x = MemberMatrix(Int(scalars.unsafe_ptr()), 1, len(scalars), MEMBER_F64)
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        for i in range(len(arrays)):  # driver-loop: over the learned arrays (model structure)
            call.block = MEMBER_BLOCK_FAMILY + 2 + i
            call.x = MemberMatrix(Int(py=xcnn_addr(arrays[i])), 1, Int(py=arrays[i].size))
            _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = 0
        call.x = MemberMatrix()
        call.classes = k
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    except error:
        member_free(plan, call)
        raise error
    _ = arrays
    _ = len(scalars)
    return call.state
