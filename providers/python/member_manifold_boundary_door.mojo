# SPDX-License-Identifier: Apache-2.0
"""The member boundary of the Manifold family (UNPACK / PACK of
core/manifold_member.mojo). Published: UMAP (kind 0). TSNE is family 42 and
not a member (no transform of new rows).

  * `manifold_member_admit` (UNPACK): the door's own parameter checks
    (bindings/umap_parameters_native_door.mojo `umap_parameters_fields`, its
    refusals in its order) on an operation-local copy of the context, as
    bindings/manifold_native_pack.mojo builds one; the door's mode word and
    route (`estimator_binding_routed(model, mode, "_mojolearn_metrics")`,
    the image the estimator's own fit calls, and `_umap_check_mode`'s key
    and refusal) -> `route.entry` = that image's UMAP operations entry,
    `route.mode` / `route.vendor` = the key's words; the 20 NEW words. The
    target settings are read without raising (the door checks them only
    when y is given): one the door refuses becomes target word -1 and a fit
    with y falls back to the door. False (the caller keeps the door): init
    other than 'spectral', an integer setting a Float64 word does not hold
    exactly (>= 2**53).
  * `manifold_member_pack` (PACK): the attributes `_umap_fit` and
    `manifold_finish` write: `embedding_`, `n_features_in_`,
    `input_copied_` (False: the member reads the caller's admitted view),
    `_transform_training`, `_transform_embedding` (its own copy, as the
    door's), `_transform_config` (the 15 validated words as the door's
    tuple), `_transform_mode`, `feature_names_in_` removed (a member's X
    has no names) and `_mojolearn_fitted`.
  * `manifold_member_state` (UNPACK): a state from `_transform_training` and
    `_transform_embedding`, after the door's refit refusal (settings or mode
    changed after fit).

SERIAL BY NATURE: one member's admission or publication (a few record reads
and two block copies); a caller runs its members in parallel.
"""
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from bindings.admitted_input_unpack import admit_x, admitted_storage_native
from bindings.classical_arrays_door import classical_address
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.member_owner_unpack import MemberOwner, member_host_only
from bindings.mani_umap_native_door import _umap_check_mode
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_get, py_item, py_setitem, py_type
from bindings.umap_parameters_native_door import (
    umap_parameters_fields, umap_scalar, umap_integer, umap_is_string, UMAP_I64_MAX,
)
from core.device_route import DeviceRoute, DEVICE_CPU, DEVICE_GPU
from core.entry_table import EntryRef, entry_ref_word
from core.manifold_member import MANIFOLD_KIND_UMAP, MF_BLOCK_ROWS, MF_BLOCK_EMBEDDING
from core.member_abi import (
    MemberPlan, MemberCall, MemberMatrix, member_call, member_new, member_free,
    MEMBER_OP_EXPORT, MEMBER_OP_IMPORT, MEMBER_IMPORT_DONE,
)
from core.neighbors_entry_table_umap import umap_ops_entry, UMAP_OPERATION_COUNT
from providers.python.estimator_binding_pyapi_door import estimator_binding_host, estimator_binding_routed
from x_ann.mani_umap_run import umap_mode_code, umap_check_target_weight, umap_target_neighbors

comptime _EXACT = 9007199254740992   # 2**53: the integers a Float64 word holds exactly


def _mode_word(fields: PythonObject, context: PythonObject, b: PythonObject) raises -> Tuple[PythonObject, String]:
    """(the settings' mode object, its word): `ManifoldSettings.mode` and
    the door's `_umap_mode`, in their words."""
    var default_mode = py_item(py_item(context, "backend_fields"), "_DEFAULT_MODE")
    var requested = py_get(fields, "numeric_mode")
    if requested is not None and py_type(requested) is not b.str:
        raise Error("numeric_mode must be fast, identical or None")
    var want = requested
    if not Bool(py=py_call(b.bool, want)):
        want = default_mode
    var word = String(py=py_call(py_call(want.strip).lower))
    var mode = default_mode if requested is None else requested
    return (mode, word)


def _target_words(fields: PythonObject, context: PythonObject, b: PythonObject) raises -> List[Float64]:
    """[target metric code, target_n_neighbors kept, target_weight]; code -1
    for a setting the door's `_target` refuses or a metric it does not
    implement (the fit with y then falls back and the door raises)."""
    var code = -1.0
    var kept = 0
    var weight = 0.5
    try:
        weight = umap_scalar(py_get(fields, "target_weight"), "target_weight", context)
        umap_check_target_weight(weight)
        var neighbors = Int(py=umap_integer(py_get(fields, "target_n_neighbors"), "target_n_neighbors", -1,
            UMAP_I64_MAX, context))
        kept = umap_target_neighbors(neighbors)
        var metric = py_get(fields, "target_metric")
        if kept < _EXACT and py_type(metric) is b.str:
            var name = String(py=metric)
            if name == "categorical":
                code = 0.0
            elif name == "l2" or name == "euclidean":
                code = 1.0
    except:
        code = -1.0
    if code < 0.0:
        kept = 0
        weight = 0.5
    var words: List[Float64] = [code, Float64(kept), weight]
    return words^


def manifold_member_admit(model: PythonObject, fields: PythonObject, kind: Int, context: PythonObject,
    b: PythonObject, mut owner: MemberOwner) raises -> Bool:
    """UNPACK: fill `owner.plan.route`, `owner.plan.params` and `owner.pins`
    from the unfitted estimator. False: the caller keeps the door."""
    if kind != MANIFOLD_KIND_UMAP:
        raise Error("ValueError: mojolearn: Manifold member kind " + String(kind) + " is not published")
    # Operation-local, as the door's (bindings/manifold_native_pack.mojo):
    # the NumPy connection and scalar snapshots never write the shared context.
    var local = py_call(b.dict, context)
    classical_connect_numpy(local)
    var validated = umap_parameters_fields(fields, local)
    var config = validated[0]
    var extras = validated[1]
    var init = py_get(fields, "init")
    if not umap_is_string(init) or String(py=init) != "spectral":
        return False
    var neighbors = Int(py=config[0])
    var epochs = Int(py=config[2])
    var seed = Int(py=config[7])
    if neighbors >= _EXACT or epochs >= _EXACT or seed >= _EXACT:
        return False
    var settings = _mode_word(fields, context, b)
    var mode_object = settings[0]
    var mode = settings[1]
    # An explicit HostUMAP (its own host-only word, lane mf-iterimp3b): the CPU identical metrics image its
    # own methods bind (`estimator_binding_host`, its refusal of a saved non-identical mode first), never the
    # device one; any other class: its binding route for the mode object, as before.
    var binding = (estimator_binding_host(model, "_mojolearn_metrics", local, False) if member_host_only(model)
                   else estimator_binding_routed(model, mode_object, "_mojolearn_metrics", local, False))
    var key = _umap_check_mode(binding, mode, "UMAP binary numeric mode disagrees with requested mode")
    var route = DeviceRoute(DEVICE_CPU if key.vendor == 0 else DEVICE_GPU)
    route.mode = key.mode
    route.vendor = key.vendor
    route.entry = EntryRef(umap_ops_entry(key), entry_ref_word(key.mode, key.vendor, UMAP_OPERATION_COUNT))
    owner.plan.route = route
    owner.pins = Python.tuple(binding)
    var target = _target_words(fields, local, b)
    var words: List[Float64] = [Float64(MANIFOLD_KIND_UMAP)]
    for i in range(11):  # small-loop(config: the door's eleven validated words)
        words.append(Float64(py=py_call(b.float, config[i])))
    for i in range(4):  # small-loop(extras: the door's four validated words)
        words.append(Float64(py=py_call(b.float, extras[i])))
    words.append(target[0])
    words.append(target[1])
    words.append(target[2])
    words.append(Float64(umap_mode_code(mode)))
    owner.plan.params = words^
    _ = local
    return True


def _config_tuple(plan: MemberPlan) raises -> PythonObject:
    """The door's `tuple(config) + tuple(extras)` from the NEW words 1 .. 15
    (ints where the door's are ints: the counts, the seed, the metric code)."""
    var items = Python.list()
    for i in range(1, 16):  # small-loop(config: the fifteen validated words)
        var word = plan.params[i]
        if i == 1 or i == 2 or i == 3 or i == 8 or i == 11 or i == 12:
            _ = py_call(items.append, PythonObject(Int(word)))
        else:
            _ = py_call(items.append, PythonObject(word))
    return py_call(native_loaded_module("builtins").tuple, items)


def _mode_name(plan: MemberPlan) -> String:
    return String("identical") if plan.params[19] == 1.0 else String("fast")


def _block_array(call: MemberCall, context: PythonObject) raises -> PythonObject:
    """A new float32 Array holding the exported block (rows x cols)."""
    var shape: List[Int] = [call.dst.rows, call.dst.cols]
    var array = admitted_storage_native(shape, "<f4", context, as_array=True)
    var count = call.dst.rows * call.dst.cols
    if count > 0:
        unsafe_memcpy(dest=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(array))),
            src=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=call.dst.data), count=count)
    return array


def manifold_member_pack(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan, state: Int) raises:
    """PACK: write the fitted attributes of state `state` into `model`."""
    var call = MemberCall(plan.route, state)
    call.block = MF_BLOCK_ROWS
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var features = call.dst.cols
    var training = _block_array(call, context)
    call.block = MF_BLOCK_EMBEDDING
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var embedding = _block_array(call, context)
    var frozen = _block_array(call, context)
    py_setitem(fields, "embedding_", embedding)
    py_setitem(fields, "n_features_in_", PythonObject(features))
    py_setitem(fields, "input_copied_", PythonObject(False))
    py_setitem(fields, "_transform_training", training)
    py_setitem(fields, "_transform_embedding", frozen)
    py_setitem(fields, "_transform_config", _config_tuple(plan))
    py_setitem(fields, "_transform_mode", PythonObject(_mode_name(plan)))
    _ = py_call(fields.pop, PythonObject("feature_names_in_"), PythonObject(None))
    py_setitem(fields, "_mojolearn_fitted", PythonObject(True))


def manifold_member_state(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan) raises -> Int:
    """UNPACK: a new member state holding the fitted attributes of `model`."""
    var training_source = py_get(fields, "_transform_training")
    var embedding_source = py_get(fields, "_transform_embedding")
    if training_source is None or embedding_source is None:
        raise Error("UMAP transform requires a successful fit")
    # The door's refit refusal: the settings and mode of this plan against
    # those the fit recorded.
    var op = native_loaded_module("operator")
    var fitted_config = py_get(fields, "_transform_config")
    var fitted_mode = py_get(fields, "_transform_mode")
    if (Bool(py=py_call(b.bool, py_call(op.ne, _config_tuple(plan), fitted_config)))
            or Bool(py=py_call(b.bool, py_call(op.ne, PythonObject(_mode_name(plan)), fitted_mode)))):
        raise Error("UMAP parameters or numeric mode changed after fit; refit before transform")
    var training = admit_x(training_source, 2, "training", context)
    var embedding = admit_x(embedding_source, 2, "embedding", context)
    var call = member_new(plan)
    try:
        call.block = MF_BLOCK_ROWS
        call.x = MemberMatrix(Int(py=classical_address(training)), Int(py=training.shape[0]),
            Int(py=training.shape[1]))
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = MF_BLOCK_EMBEDDING
        call.x = MemberMatrix(Int(py=classical_address(embedding)), Int(py=embedding.shape[0]),
            Int(py=embedding.shape[1]))
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    except error:
        member_free(plan, call)
        raise error
    _ = training
    _ = embedding
    return call.state
