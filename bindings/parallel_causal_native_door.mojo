# SPDX-License-Identifier: Apache-2.0
"""Native ownership for the existing sequential layer-partitioned CausalLM."""
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.torch_parameter_form_pack import torch_parameter_form
from bindings.native_error_class import error_without_class
from bindings.generation_buffers_door import GENERATION_BOUNDARY, generation_initialize_binding, generation_copy, generation_array, generation_array_snapshot, generation_address, py_ints
from bindings.causal_lm_construct_door import causal_model_read_weights_binding, _causal_model_prepare_snapshot, _model_construction_set, _model_options_snapshot, _model_weights_snapshot, _model_block, _model_bindings, _model_profile, _model_quantize15
from bindings.causal_lm_public_door import causal_model_embedding, causal_model_head, causal_lm_parameters_binding, _native_function, causal_entry_key
from bindings.neural_block_public_door import neural_block_owner, neural_block_allocate_binding, neural_block_forward_binding
from bindings.causal_model_config_door import config_load_binding, model_plan_for_binding, model_plan_snapshot, model_record_fields, model_dict, model_plain, model_sequence, model_string
from bindings.causal_operation_owner_door import causal_snapshot_fields, _causal_namespace, causal_model_capture, causal_model_publish
from bindings.mode_metadata_native_door import _fields
from bindings.native_loaded_module_door import native_loaded_module, NativeModules
from bindings.causal_checkpoint_door import checkpoint_open_binding, checkpoint_close_binding
from bindings.causal_exception_unpack import causal_borrowed_item, causal_set_exception
from bindings.native_json_door import native_path, native_path_exists
from bindings.native_process_pool_door import (
    process_pool_create, process_pool_start, process_pool_request, process_pool_close, process_worker_connect, process_pool_admit,
    process_pool_pids, process_worker_next, process_worker_value,
)
from bindings.native_process_wire_door import WireOutgoing, wire_unpack, wire_append, wire_key, wire_text, wire_send
from bindings.native_transport_door import transport_close
from bindings.py_balanced import py_call, py_contains, py_fields, py_get, py_setattr, py_setitem, py_type
from core.parallel_worker_request import (
    WorkerRequest, worker_request_decode, parallel_causal_op_code, WORKER_NO_HANDLE,
    PC_CONFIGURE, PC_BLOCK, PC_TENSORS, PC_EMBEDDING, PC_FINAL, PC_HEAD, PC_HEAD_INT15, PC_NORM, PC_ROUTE,
    PC_ALLOCATE, PC_RELEASE, PC_STATE, PC_PUBLISH_STATE, PC_RUN_LAYERS, PC_PARAMETERS, PC_RUN, PC_BLOCK_FIELDS,
)
from core.parallel_worker_frame import WIRE_RESULT, WIRE_ERROR, WIRE_CLOSE, WIRE_MAP, WIRE_LIST
from bindings.parallel_causal_native_run import (
    CAUSAL_DEVICE_INEXACT,
    causal_route_reads_devices, causal_route_plan, causal_route_name, causal_layer_map_admit,
    causal_closed_admit, causal_layer_format_admit, causal_layer_owner_admit,
    CausalRegistryState, causal_registry_usable, causal_registry_worker,
    causal_option_admit, causal_weight_format_admit, causal_block_index_admit, causal_allocation_index_admit,
    causal_worker_route, causal_worker_route_admit, causal_top_tensor_admit,
    causal_step_word, causal_run_request, causal_layers_request, causal_layers_count,
    causal_run_joins, causal_run_embeds, causal_run_final,
)
from core.shared_slot import SharedSlot
from std.collections import Dict
from bindings.parallel_keys_unpack import (
    parallel_keys,
    PARALLEL_CAUSAL_OPN_ALLOCATE_LAYER,
    PARALLEL_CAUSAL_OPN_BLOCK_ALLOCATE,
    PARALLEL_CAUSAL_OPN_BLOCK_CALL,
    PARALLEL_CAUSAL_OPN_BLOCK_FIELDS,
    PARALLEL_CAUSAL_OPN_BLOCK_INIT,
    PARALLEL_CAUSAL_OPN_BLOCK_RUN,
    PARALLEL_CAUSAL_OPN_CAPTURE_MODEL,
    PARALLEL_CAUSAL_OPN_CAPTURE_STATE,
    PARALLEL_CAUSAL_OPN_CLOSE,
    PARALLEL_CAUSAL_OPN_CLOSE_QUIET,
    PARALLEL_CAUSAL_OPN_ENTER,
    PARALLEL_CAUSAL_OPN_EXIT,
    PARALLEL_CAUSAL_OPN_INIT,
    PARALLEL_CAUSAL_OPN_INITIALIZE,
    PARALLEL_CAUSAL_OPN_LOAD,
    PARALLEL_CAUSAL_OPN_LOAD_WITH_WORKER,
    PARALLEL_CAUSAL_OPN_PARAMETERS,
    PARALLEL_CAUSAL_OPN_PRIMITIVE,
    PARALLEL_CAUSAL_OPN_PRIMITIVES_INIT,
    PARALLEL_CAUSAL_OPN_RELEASE_LAYERS,
    PARALLEL_CAUSAL_OPN_RPC,
    PARALLEL_CAUSAL_OPN_RUN,
    PARALLEL_CAUSAL_OPN_STATE_CLOSE,
    PARALLEL_CAUSAL_OPN_STATE_INIT,
    PARALLEL_CAUSAL_OPN_STATE_PUBLISH,
    PARALLEL_CAUSAL_OPN_STATE_SNAPSHOT,
    PARALLEL_CAUSAL_OPN_WORKER,
)
from bindings.py_keys import py_op_code
from bindings.py_balanced import py_item


struct ParallelCausalBoundary(Defaultable, Movable):
    var bi: PythonObject
    var context: PythonObject
    var classes: PythonObject
    var entry: PythonObject

    def __init__(out self):
        self.bi = PythonObject(None)
        self.context = PythonObject(None)
        self.classes = PythonObject(None)
        self.entry = PythonObject(None)


comptime PARALLEL_CAUSAL = SharedSlot[ParallelCausalBoundary, "MojolearnParallelCausalOwnerV1"]


def parallel_causal_initialize(mut mods: NativeModules, args: PythonObject) raises:
    var p = PARALLEL_CAUSAL.get_or_create_ptr()
    var bi = mods.module("builtins")
    model_sequence(mods, args)
    if len(args) != 3:
        raise Error("TypeError: parallel connection requires context, classes and native entry")
    model_dict(mods, args[0])
    model_dict(mods, args[1])
    var classes = Python.dict()
    for name in ["model", "block", "state", "primitives"]:  # small-loop(four import-connected record identities)
        var cls = args[1][name]
        if py_type(cls) is not bi.type:
            raise Error("TypeError: parallel connection requires ordinary class identities")
        if p[].classes is not None and p[].classes[name] is not cls:
            raise Error("TypeError: parallel record identities cannot be reconnected")
        py_setitem(classes, name, cls)
    if py_type(args[2]) is not py_type(bi.len):
        raise Error("ParallelCausalLM: execution router must be a compiled native entry")
    if py_type(args[2].__self__) is not mods.module("types").ModuleType or args[2].__self__ is bi:
        raise Error("ParallelCausalLM: execution router must belong to its native module")
    if p[].entry is not None:
        if p[].entry is not args[2]:
            raise Error("TypeError: parallel execution router cannot be reconnected")
        return
    p[].bi = bi
    p[].context = py_call(bi.dict, args[0])
    p[].classes = classes
    p[].entry = args[2]
    var nk = parallel_keys()
    _ = generation_initialize_binding(py_item(p[].context, nk[].k_generation_classes))


def _parallel_causal_fields(value: PythonObject) raises -> PythonObject:
    return _fields(value)


def _parallel_word(value: PythonObject) raises -> Int:
    if py_type(value) is not PARALLEL_CAUSAL.get_or_create_ptr()[].bi.int:
        raise Error("TypeError: parallel ownership requires exact integer metadata")
    return Int(py=value)


def _parallel_flag(value: PythonObject) raises -> Bool:
    if py_type(value) is not PARALLEL_CAUSAL.get_or_create_ptr()[].bi.bool:
        raise Error("TypeError: parallel ownership requires exact boolean metadata")
    return Bool(py=value)


def _parallel_model_fields(mut mods: NativeModules, owner: PythonObject) raises -> PythonObject:
    var nk = parallel_keys()
    return model_record_fields(mods, owner, py_item(PARALLEL_CAUSAL.get_or_create_ptr()[].classes, nk[].k_model),
        Python.tuple("layer_devices", "_worker_index", "_pool", "_pools", "_closed", "_failed", "_blocks", "kind",
            "weight_format", "_native_entries", "_native_vendor", "_native_numeric_mode", "route", "_head_int15", "_block_class", "_block_kwargs", "_prims",
            "plan", "config", "d_model", "vocab_size", "n_layers", "norm_eps", "numeric_profile", "device", "max_positions",
            "_embed", "_norm", "_head", "_runtime", "_parallel_entry", "unused_names", "_causal_entries_owner"))


def _parallel_block_fields(mut mods: NativeModules, block: PythonObject) raises -> PythonObject:
    var nk = parallel_keys()
    return model_record_fields(mods, block, py_item(PARALLEL_CAUSAL.get_or_create_ptr()[].classes, nk[].k_block),
        Python.tuple("model", "index", "weight_format"))


def _parallel_state_fields(mut mods: NativeModules, state: PythonObject) raises -> PythonObject:
    var nk = parallel_keys()
    return model_record_fields(mods, state, py_item(PARALLEL_CAUSAL.get_or_create_ptr()[].classes, nk[].k_state),
        Python.tuple("layer", "token", "_lifetime"))


struct ParallelCausalOwner(Movable, Writable):
    """One model's native owner, stored once in the instance dictionary as
    `_native`. `native` is the typed worker registry RUN decides on
    (bindings/parallel_causal_native_run.mojo): the device of each layer,
    the worker of each device, closed, failed. The Python objects are
    boundary pins the door admitted and built itself at construction: the
    layer-device tuple, the device-to-worker dictionary, the process pool's
    dictionary and its per-device aliases. An operation's captured namespace
    receives these pins as they are; nothing is re-admitted or copied per
    operation. `layer_devices`, `_worker_index`, `_pool`, `_pools`,
    `_closed` and `_failed` on the instance are mirrors for callers and for
    the shared model doors; this door writes them and never reads them."""
    var owner_id: Int
    var native: CausalRegistryState
    var devices: PythonObject
    var indices: PythonObject
    var pool: PythonObject
    var pools: PythonObject

    def __init__(out self, owner_id: Int, layer_devices: List[Int], devices: PythonObject,
                 indices: PythonObject, pool: PythonObject, pools: PythonObject):
        self.owner_id = owner_id
        self.native = CausalRegistryState(layer_devices)
        self.devices = devices
        self.indices = indices
        self.pool = pool
        self.pools = pools

    def write_to(self, mut writer: Some[Writer]):
        writer.write("ParallelCausalOwner(closed=", self.native.closed, ", failed=", self.native.failed, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def _causal_record_or_none(owner: PythonObject) raises -> PythonObject:
    """The model's native owner record, or None for an instance whose
    constructor has not built it yet. The instance dictionary is read
    through object's own lookup (no hook runs)."""
    var nk = parallel_keys()
    var record = py_get(py_fields(owner), nk[].k__native)
    if record is None:
        return record
    var owner_id = Int(py=py_call(PARALLEL_CAUSAL.get_or_create_ptr()[].bi.id, owner))
    try:
        var rec = record.downcast_value_ptr[ParallelCausalOwner]()
        if rec[].owner_id != owner_id:
            raise Error("foreign")
    except:
        raise Error("TypeError: parallel model storage does not belong to this model")
    return record


def _causal_record(owner: PythonObject) raises -> PythonObject:
    """The exact native owner record of a constructed model."""
    var record = _causal_record_or_none(owner)
    if record is None:
        raise Error("RuntimeError: ParallelCausalLM is not initialized")
    return record


def _causal_serving(owner: PythonObject) raises -> Bool:
    """Whether the model's workers can still be asked (not closed, not lost)."""
    var record = _causal_record(owner)
    var rec = record.downcast_value_ptr[ParallelCausalOwner]()
    return not rec[].native.closed and not rec[].native.failed


def _parallel_pool_admit(mut mods: NativeModules, pool: PythonObject) raises:
    """The pool handle is the pool door's native owner
    (bindings/native_process_pool_door.mojo `ProcessPoolOwner`): admitted by
    its exact native type. REMOVED 2026-11-21 (lane pools-frame): the exact
    dict test of the handle and the lookup of its `record` entry."""
    try:
        process_pool_admit(pool)
    except:
        raise Error("TypeError: parallel routing has an inconsistent process owner")


def _parallel_capture_model_fields(mut mods: NativeModules, owner: PythonObject, live: PythonObject) raises -> PythonObject:
    model_dict(mods, live)
    var result = Python.dict()
    var nk = parallel_keys()
    # The registry words come from the model's native owner: the pins it
    # admitted once at construction, as they are. REMOVED: the per-operation
    # re-admission of `layer_devices` (a tuple copy, the layer-map rule and
    # the route rule), the two exact-flag reads, the key lists and the
    # copies of the worker-index and pool-alias dictionaries.
    var record = _causal_record(owner)
    var rec = record.downcast_value_ptr[ParallelCausalOwner]()
    var devices = rec[].devices
    py_setitem(result, nk[].k_layer_devices, devices)
    py_setitem(result, nk[].k__closed, PythonObject(rec[].native.closed))
    py_setitem(result, nk[].k__failed, PythonObject(rec[].native.failed))
    py_setitem(result, nk[].k__pool, rec[].pool)
    py_setitem(result, nk[].k__worker_index, rec[].indices)
    py_setitem(result, nk[].k__pools, rec[].pools)
    var blocks = py_item(live, nk[].k__blocks)
    model_sequence(mods, blocks)
    var captured = Python.list()
    var seen = Python.dict()
    var bi = PARALLEL_CAUSAL.get_or_create_ptr()[].bi
    for i in range(len(blocks)):  # driver-loop: retain original block identities and capture each owner once
        var key = py_call(bi.id, blocks[i])
        var block = py_get(seen, key)
        if block is None:
            var source = _parallel_block_fields(mods, blocks[i])
            if py_item(source, nk[].k_model) is not owner:
                raise Error("ParallelCausalLM: remote block belongs to another model")
            var index = _parallel_word(py_item(source, nk[].k_index))
            causal_block_index_admit(index, len(devices))
            var fields = Python.dict()
            py_setitem(fields, nk[].k_model, owner)
            py_setitem(fields, nk[].k_index, index)
            py_setitem(fields, nk[].k_weight_format, model_string(mods, py_item(source, nk[].k_weight_format)))
            py_setitem(fields, nk[].k__causal_original, blocks[i])
            block = _causal_namespace(mods, fields)
            py_setitem(seen, key, block)
        _ = py_call(captured.append, block)
    py_setitem(result, nk[].k__native_parallel_blocks, captured)
    return result


def _parallel_capture_model(mut mods: NativeModules, owner: PythonObject) raises -> PythonObject:
    return _parallel_capture_model_fields(mods, owner, _parallel_model_fields(mods, owner))


def _parallel_capture_model(mut mods: NativeModules, owner: PythonObject, copied: PythonObject) raises -> PythonObject:
    _ = _parallel_model_fields(mods, owner)  # admit added parallel descriptors without recapturing values
    return _parallel_capture_model_fields(mods, owner, copied)


def _parallel_owner_snapshot(mut mods: NativeModules, owner: PythonObject) raises -> PythonObject:
    # Direct RemoteBlock/RemoteState methods enter here once. Shared CausalLM
    # operations supply their already captured model namespace instead.
    var live = _parallel_model_fields(mods, owner)
    var fields = _parallel_capture_model_fields(mods, owner, live)
    var nk = parallel_keys()
    py_setitem(fields, nk[].k_kind, model_string(mods, py_item(live, nk[].k_kind)))
    py_setitem(fields, nk[].k_weight_format, model_string(mods, py_item(live, nk[].k_weight_format)))
    for name in ["_native_vendor", "_native_numeric_mode"]:  # small-loop(two primitive layer construction metadata words)
        if py_contains(live, name):
            py_setitem(fields, name, model_plain(mods, live[name]))
    var entries = py_item(live, nk[].k__native_entries)
    model_dict(mods, entries)
    py_setitem(fields, nk[].k__native_entries, py_call(PARALLEL_CAUSAL.get_or_create_ptr()[].bi.dict, entries))
    py_setitem(fields, nk[].k__native_parallel_entries_live, entries)
    py_setitem(fields, nk[].k__causal_original, owner)
    py_setitem(fields, nk[].k__causal_live_fields, live)
    return _causal_namespace(mods, fields)


def _parallel_captured(mut mods: NativeModules, owner: PythonObject, captured: PythonObject) raises -> PythonObject:
    var fields = causal_snapshot_fields(mods, captured)
    model_dict(mods, fields)
    var nk = parallel_keys()
    if py_item(fields, nk[].k__causal_original) is not owner:
        raise Error("TypeError: parallel metadata belongs to another owner")
    var failure = py_get(fields, nk[].k__causal_admission_failure)
    if failure is not None:
        raise Error(model_string(mods, failure))
    return fields


def _parallel_capture_state_fields(mut mods: NativeModules, remote: PythonObject, live: PythonObject, model: PythonObject) raises -> PythonObject:
    var nk = parallel_keys()
    var layer = py_item(live, nk[].k_layer)
    var block = _parallel_find_block(mods, layer, model)
    var token = py_item(live, nk[].k_token)
    if token is not None:
        _ = _parallel_word(token)
    var lifetime = py_get(live, nk[].k__lifetime)
    if lifetime is not None:
        _ = lifetime.downcast_value_ptr[RemoteStateLifetime]()
    var result = Python.dict()
    py_setitem(result, nk[].k_layer, layer)
    py_setitem(result, nk[].k_token, token)
    py_setitem(result, nk[].k__lifetime, lifetime)
    py_setitem(result, nk[].k__causal_original, remote)
    py_setitem(result, nk[].k__native_state_identity, remote)
    py_setitem(result, nk[].k__causal_live_fields, live)
    py_setitem(result, nk[].k__causal_initial_buffers, Python.dict())
    py_setitem(result, nk[].k__native_parallel_model, model)
    py_setitem(result, nk[].k__native_parallel_block, block)
    return _causal_namespace(mods, result)


def _parallel_capture_state(mut mods: NativeModules, remote: PythonObject, model: PythonObject) raises -> PythonObject:
    return _parallel_capture_state_fields(mods, remote, _parallel_state_fields(mods, remote), model)


def _parallel_capture_state(mut mods: NativeModules, remote: PythonObject) raises -> PythonObject:
    var live = _parallel_state_fields(mods, remote)
    var nk = parallel_keys()
    var layer = _parallel_block_fields(mods, py_item(live, nk[].k_layer))
    return _parallel_capture_state_fields(mods, remote, live, _parallel_owner_snapshot(mods, py_item(layer, nk[].k_model)))


def _parallel_device_words(devices: PythonObject) raises -> List[Int]:
    """UNPACK: the device indices as words (CAUSAL_DEVICE_INEXACT for an
    entry that is not an exact integer)."""
    var bi = PARALLEL_CAUSAL.get_or_create_ptr()[].bi
    var words = List[Int]()
    for i in range(len(devices)):  # driver-loop: over explicit layer ownership metadata
        words.append(Int(py=devices[i]) if py_type(devices[i]) is bi.int else CAUSAL_DEVICE_INEXACT)
    return words^


def _parallel_causal_route(mut mods: NativeModules, devices: PythonObject) raises -> PythonObject:
    var p = PARALLEL_CAUSAL.get_or_create_ptr()
    model_sequence(mods, devices)
    var nk = parallel_keys()
    var vendor = String(py_item(p[].context, nk[].k_selected_vendor))
    # The rule reads the indices for Metal only; they are unpacked only then.
    var words = List[Int]()
    if causal_route_reads_devices(vendor):
        words = _parallel_device_words(devices)
    return PythonObject(causal_route_name(causal_route_plan(vendor, words)))


def _parallel_causal_map(mut mods: NativeModules, value: PythonObject, expected: Int = -1) raises -> PythonObject:
    var bi = PARALLEL_CAUSAL.get_or_create_ptr()[].bi
    model_sequence(mods, value)
    var devices = py_call(bi.tuple, value)
    causal_layer_map_admit(_parallel_device_words(devices), expected)
    _ = _parallel_causal_route(mods, devices)
    return devices


struct CausalWorkerRegistry(Movable):
    """The worker's own handles: a layer index names a block holder and a
    state token names a layer state, both valid only in this worker. A typed
    request carries the id; nothing about the parent's model object crosses.
    (Replaces the Python `world` dict and its "blocks"/"states"/"next_state"
    entries; lane parallel-workers-typed-rpc.)"""
    var plan: PythonObject
    var top: PythonObject
    var bindings: PythonObject
    var blocks: Dict[Int, PythonObject]
    var states: Dict[Int, PythonObject]
    var state_layers: Dict[Int, Int]
    var next_state: Int

    def __init__(out self):
        self.plan = PythonObject(None)
        self.top = PythonObject(None)
        self.bindings = PythonObject(None)
        self.blocks = Dict[Int, PythonObject]()
        self.states = Dict[Int, PythonObject]()
        self.state_layers = Dict[Int, Int]()
        self.next_state = 0


def _registry_block(registry: CausalWorkerRegistry, index: Int) raises -> PythonObject:
    if index not in registry.blocks:
        raise Error("ParallelCausalLM: layer not initialized in this worker")
    return registry.blocks.get(index, PythonObject(None))


def _registry_state(registry: CausalWorkerRegistry, token: Int, layer: Int = WORKER_NO_HANDLE) raises -> PythonObject:
    """The state `token` names; with `layer`, it must belong to that layer."""
    if token not in registry.states:
        raise Error("ParallelCausalLM: state released")
    if layer != WORKER_NO_HANDLE and registry.state_layers.get(token, WORKER_NO_HANDLE) != layer:
        raise Error("ParallelCausalLM: state belongs to another layer")
    return registry.states.get(token, PythonObject(None))


def _wire_true() -> WireOutgoing:
    var reply = WireOutgoing()
    reply.record.boolean(True)
    return reply^


def _wire_fields(fields: PythonObject, names: PythonObject) raises -> WireOutgoing:
    """The reply record of named fields, built field by field: no Python
    dictionary is made for the reply (lane parallel-payload-transport)."""
    var reply = WireOutgoing()
    reply.record.node(WIRE_MAP, len(names))
    for i in range(len(names)):  # driver-loop: over the reply's field names
        var name = names[i]
        var value = fields[name]
        wire_append(name, reply)
        wire_append(value, reply)
    return reply^


def _parallel_causal_worker_request(mut mods: NativeModules, mut registry: CausalWorkerRegistry, request: WorkerRequest,
                                    args: PythonObject) raises -> WireOutgoing:
    # RUN of one typed request: `request` is the decoded struct (operation
    # code, handle id, words); `args` is the payload PACKed from the frame's
    # typed record (None for handle-only operations). The reply is a typed
    # record (bindings/native_process_wire.mojo): the former dict replies
    # (block metadata, state, parameters, block_fields) are written field by
    # field into it.
    var p = PARALLEL_CAUSAL.get_or_create_ptr()
    var bi = p[].bi
    var context = p[].context
    var op = request.op
    var nk = parallel_keys()
    if op == PC_CONFIGURE:
        if registry.top is not None:
            raise Error("ParallelCausalLM: worker already configured")
        var connected = String(py_item(context, nk[].k_selected_vendor))
        var route = PythonObject(causal_route_name(causal_worker_route(connected)))
        var metadata = args[0]
        var plan_fields = args[1]
        var named = py_item(metadata, nk[].k_device)
        var named_exact = py_type(named) is bi.str
        causal_worker_route_admit(named_exact, String(named) if named_exact else String(""), connected)
        var plan = py_call(bi.object.__new__, py_item(py_item(context, nk[].k_config_context), nk[].k_plan_class))
        _ = py_call(_parallel_causal_fields(plan).update, plan_fields)
        var top = py_call(bi.object.__new__, py_item(context, nk[].k_model_class))
        _ = py_call(_parallel_causal_fields(top).update, metadata)
        py_setattr(top, "_embed", PythonObject(None))
        py_setattr(top, "_norm", PythonObject(None))
        py_setattr(top, "_head", PythonObject(None))
        py_setattr(top, "_head_int15", PythonObject(None))
        var bindings = _model_bindings(mods, route, String(plan.kind), context)
        var entries = bi.dict()
        # The worker's embedding and head reach the primitives and linalg
        # images through the CausalLM entry table (core/causal_lm_entry_table.mojo)
        # by the route's key; the five Python entry lookups that stood here
        # (embedding/rms_norm/linear forward, quantize_int15, gemm_int15) are gone.
        py_setitem(entries, nk[].k_entry_mode, py_item(bindings, nk[].k_entry_mode))
        py_setitem(entries, nk[].k_entry_vendor, py_item(bindings, nk[].k_entry_vendor))
        py_setattr(top, "_native_entries", entries)
        registry.plan = plan
        registry.top = top
        registry.bindings = bindings
        return _wire_true()
    var top = registry.top
    if top is None:
        raise Error("ParallelCausalLM: worker was not told its route")
    if op == PC_BLOCK:
        var index = request.handle
        if index in registry.blocks:
            raise Error("ParallelCausalLM: layer already initialized")
        var block = _model_block(mods, args, registry.plan, top.numeric_profile, top.device, registry.bindings, context)
        var holder = neural_block_owner(mods, block, context)
        registry.blocks[index] = holder
        var entries = holder._native_entries
        var reply = WireOutgoing()
        reply.record.node(WIRE_MAP, 5)
        wire_key(reply, "weight_format")
        wire_append(py_item(_parallel_causal_fields(block), nk[].k_weight_format), reply)
        wire_key(reply, "weight_names")
        wire_append(py_item(entries, nk[].k_weight_names), reply)
        wire_key(reply, "vendor")
        wire_append(holder._native_vendor, reply)
        wire_key(reply, "numeric_mode")
        wire_append(holder._native_numeric_mode, reply)
        wire_key(reply, "mamba1_profile")
        wire_append(py_get(entries, nk[].k_mamba1_profile, PythonObject(1)), reply)
        return reply^
    if op == PC_TENSORS:
        var fields = _parallel_causal_fields(top)
        var keys = py_call(bi.list, args)
        for i in range(len(keys)):  # driver-loop: over the embedding and final-head tensor owners
            var name = String(keys[i])
            causal_top_tensor_admit(name)
            py_setitem(fields, "_" + name, args[keys[i]])
        if py_contains(args, nk[].k_head) and String(top.numeric_profile) == "fixed15_v1":
            py_setattr(top, "_head_int15", _model_quantize15(mods, top._head, causal_entry_key(registry.bindings)))
        return wire_unpack(top._head_int15)
    if op == PC_EMBEDDING:
        var windowed_result12 = causal_model_embedding(top, args)
        mods.released()
        return wire_unpack(windowed_result12)
    if op == PC_FINAL or op == PC_HEAD or op == PC_HEAD_INT15:
        # Generation head admission reads only a captured native namespace
        # (generation_metadata_pyapi._generation_fields), never a live model
        # owner. Capture the worker's private top-level fields per operation,
        # as the single-process path captures its model.
        var captured_top = _causal_namespace(mods, _parallel_causal_fields(top))
        if op == PC_FINAL:
            var windowed_result11 = causal_model_head(captured_top, args)
            mods.released()
            return wire_unpack(windowed_result11)
        var windowed_result10 = causal_model_head(captured_top, args, False, op == PC_HEAD_INT15)
        mods.released()
        return wire_unpack(windowed_result10)
    if op == PC_NORM:
        var x = args[0]
        var dims = List[Int]()
        for i in range(len(x.shape)):  # driver-loop: over activation shape metadata
            dims.append(Int(py=x.shape[i]))
        var out = generation_array(dims)
        var keep = List[PythonObject]()
        var params = py_ints([Int(py=x.size) // Int(py=top.d_model), Int(py=top.d_model)])
        _ = py_call(params.append, args[1])
        _ = py_call(py_item(top._native_entries, nk[].k_rms_norm_forward), py_ints([generation_address(out, keep), generation_address(x, keep), generation_address(top._norm, keep)]), params)
        mods.released()
        _ = keep^
        return wire_unpack(out)
    if op == PC_ROUTE:
        if Bool(py=args != top.device):
            raise Error("ParallelCausalLM: worker already owns a different route")
        return _wire_true()
    if op == PC_ALLOCATE:
        # words: [batch, capacity or -1 for none]
        var holder = _registry_block(registry, request.handle)
        var capacity = PythonObject(None) if request.words[1] < 0 else PythonObject(request.words[1])
        var state = neural_block_allocate_binding(mods, holder._blocks[0], PythonObject(request.words[0]), capacity, context)
        var token = registry.next_state
        registry.next_state = token + 1
        registry.states[token] = state
        registry.state_layers[token] = request.handle
        var reply = WireOutgoing()
        reply.record.integer(token)
        return reply^
    if op == PC_RELEASE:
        # A token this worker no longer holds is a quiet success, as before.
        if request.handle in registry.states:
            _ = registry.states.pop(request.handle)
            _ = registry.state_layers.pop(request.handle)
        return _wire_true()
    if op == PC_STATE or op == PC_PUBLISH_STATE:
        var state = _registry_state(registry, request.handle)
        var fields = _parallel_causal_fields(state)
        if op == PC_STATE:
            return _wire_fields(fields, py_call(bi.list, fields))
        var supplied = args
        if Bool(py=py_call(bi.set, supplied) != py_call(bi.set, fields)):
            raise Error("ParallelCausalLM: restored state fields differ")
        var keys = py_call(bi.list, fields)
        for i in range(len(keys)):  # driver-loop: over saved remote state fields before atomic publication
            var old = fields[keys[i]]
            var value = supplied[keys[i]]
            if py_type(old) is py_item(context, nk[].k_array_class):
                if py_type(value) is not py_item(context, nk[].k_array_class) or Bool(py=value.shape != old.shape) or Bool(py=value.dtype != old.dtype):
                    raise Error("ParallelCausalLM: restored state tensor differs in shape or dtype")
            elif py_type(value) is not py_type(old):
                raise Error("ParallelCausalLM: restored state metadata type differs")
        _ = py_call(fields.update, supplied)
        return _wire_true()
    if op == PC_RUN_LAYERS:
        # Lane p3-msel-parallel-and-process-pool (audit B worst path 9): one
        # request runs this worker's consecutive layers for a token batch,
        # with the embedding when the batch starts here and the final head
        # when it ends here. words: [step, has ids, final, n, (layer index,
        # state token or -1) * n]; args: (x, ids or None). The same entries in
        # the same order as the per-layer requests: no bits change.
        var count = causal_layers_count(request.words)
        var step = PythonObject(request.words[0] != 0)
        var x = args[0]
        if request.words[1] != 0:
            x = causal_model_embedding(top, args[1])
            mods.released()
        for i in range(count):  # driver-loop: over this worker's consecutive layers
            var index = request.words[4 + 2 * i]
            var token = request.words[5 + 2 * i]
            var layer_holder = _registry_block(registry, index)
            var layer_state = PythonObject(None)
            if token != WORKER_NO_HANDLE:
                layer_state = _registry_state(registry, token, index)
            x = neural_block_forward_binding(mods, layer_holder._blocks[0], x, layer_state, step, PythonObject(None), context)
        if request.words[2] != 0:
            var head_top = _causal_namespace(mods, _parallel_causal_fields(top))
            x = causal_model_head(head_top, x)
            mods.released()
        return wire_unpack(x)
    var holder = _registry_block(registry, request.handle)
    var block = holder._blocks[0]
    if op == PC_PARAMETERS:
        var fields = _parallel_causal_fields(block)
        var names = py_item(holder._native_entries, nk[].k_weight_names)
        var weights = py_item(fields, nk[].k__w)
        var reply = WireOutgoing()
        reply.record.node(WIRE_MAP, len(names))
        for i in range(len(names)):  # driver-loop: over the original block parameter registry
            var name = names[i]
            var weight = weights[i]
            wire_append(name, reply)
            wire_append(weight, reply)
        return reply^
    if op == PC_RUN:
        # words: [step, state token or -1]; args: x
        var state = PythonObject(None)
        if request.words[1] != WORKER_NO_HANDLE:
            state = _registry_state(registry, request.words[1], request.handle)
        var y = neural_block_forward_binding(mods, block, args, state, PythonObject(request.words[0] != 0), PythonObject(None), context)
        return wire_unpack(y)
    if op == PC_BLOCK_FIELDS:
        var fields = _parallel_causal_fields(block)
        var present = bi.list()
        for key in [String("_w"), String("_wopt"), String("_int15"), String("_opts_tail"), String("n_heads"), String("n_kv_heads"), String("window"), String("dt_limit"), String("weight_format")]:  # driver-loop: over serializable parameter/fingerprint metadata
            if py_contains(fields, key):
                _ = py_call(present.append, PythonObject(key))
        return _wire_fields(fields, present)
    raise Error("ParallelCausalLM: unknown worker operation code " + String(op))


def parallel_causal_worker(mut mods: NativeModules, argv: PythonObject) raises:
    model_sequence(mods, argv)
    for i in range(len(argv)):  # cpu-route: admit native startup argument strings before worker connection
        _ = model_string(mods, argv[i])
    var socket = process_worker_connect(argv)
    mods.released()
    var registry = CausalWorkerRegistry()
    var no_request = List[Int64]()
    try:
        while True:  # driver-loop: over native remote model/layer/state requests
            # One typed frame per request (bindings/native_process_wire.mojo):
            # the request struct's words and its PACKed payload. REMOVED: the
            # codec envelope (JSON header, bytes request, structural payload)
            # and its decoder _parallel_causal_decode.
            # The typed receive every worker uses (`process_worker_next`:
            # the frame's request words and its record, no Python value). The
            # request is PACKed once, and only when it carries a record: the
            # operations whose arguments are all request words (allocate,
            # release, state, parameters, block fields) build no Python
            # value. REMOVED 2026-11-21 (lane pools-frame): `wire_receive`,
            # which PACKed every frame before its kind and words were read.
            var turn = process_worker_next(socket)
            mods.released()
            if not turn.open:
                break
            var sequence = turn.sequence
            # Outside the model-error handler, as the PACK always was: a
            # record that cannot be PACKed ends the worker.
            var payload = PythonObject(None)
            if turn.frame.record.node_count() > 0 or len(turn.frame.stores) > 0:
                payload = process_worker_value(turn)
            var reply = WireOutgoing()
            var failed = False
            try:
                var typed = worker_request_decode(Int(turn.request.unsafe_ptr()), len(turn.request) * 8)
                reply = _parallel_causal_worker_request(mods, registry, typed, payload)
            except error:
                # A model error is the reply; the worker and its registry live on.
                reply = wire_text(String(error))
                failed = True
            _ = turn^  # keep alive: worker_request_decode reads turn.request through an untracked pointer
            mods.released()
            wire_send(socket, WIRE_ERROR if failed else WIRE_RESULT, sequence, no_request, reply)
            mods.released()
    except error:
        mods.released()
        transport_close(socket)
        mods.released()
        raise error
    transport_close(socket)
    mods.released()


def _parallel_causal_usable(owner: PythonObject) raises:
    # A closed model or a lost worker state is a RuntimeError on every public
    # call (forward, step, parameters, ...); an invalid argument stays a
    # ValueError. One check, on the model's native owner, so the calls
    # cannot disagree.
    var record = _causal_record(owner)
    causal_registry_usable(record.downcast_value_ptr[ParallelCausalOwner]()[].native)


struct CausalRpc(Movable):
    """One worker request after UNPACK: the typed struct and the tensor or
    structural payload that rides beside it."""
    var request: WorkerRequest
    var payload: PythonObject

    def __init__(out self, var request: WorkerRequest, payload: PythonObject):
        self.request = request^
        self.payload = payload


def _parallel_causal_typed(mut mods: NativeModules, name: String, args: PythonObject) raises -> CausalRpc:
    """UNPACK of the private by-name entry points (`ParallelCausalLM._rpc`,
    `_RemoteBlock.call`, the primitives): the legacy argument shape of
    `name` read once into a typed request."""
    var op = parallel_causal_op_code(name)
    var request = WorkerRequest(op)
    var payload = args
    if op == PC_BLOCK:
        request.handle = _parallel_word(args[0])
        payload = args[1]
    elif op == PC_ALLOCATE:
        request.handle = _parallel_word(args[0])
        request.words.append(_parallel_word(args[1]))
        request.words.append(WORKER_NO_HANDLE if args[2] is None else _parallel_word(args[2]))
        payload = PythonObject(None)
    elif op == PC_RELEASE or op == PC_STATE or op == PC_PARAMETERS:
        request.handle = _parallel_word(args)
        payload = PythonObject(None)
    elif op == PC_BLOCK_FIELDS:
        request.handle = _parallel_word(args[0])
        payload = PythonObject(None)
    elif op == PC_PUBLISH_STATE:
        request.handle = _parallel_word(args[0])
        payload = args[1]
    elif op == PC_RUN:
        request.handle = _parallel_word(args[0])
        request.words.append(causal_step_word(model_string(mods, args[1])))
        request.words.append(WORKER_NO_HANDLE if args[3] is None else _parallel_word(args[3]))
        payload = args[2]
    elif op == PC_RUN_LAYERS:
        var items = args[0]
        var step_word = causal_step_word(model_string(mods, args[1]))
        var with_head = _parallel_flag(args[4])
        var pairs = List[Int]()
        for i in range(len(items)):  # driver-loop: over the layers of one worker's run
            pairs.append(_parallel_word(items[i][0]))
            pairs.append(WORKER_NO_HANDLE if items[i][1] is None else _parallel_word(items[i][1]))
        request = causal_layers_request(step_word, args[3] is not None, with_head, pairs)
        payload = Python.list(args[2], args[3])
    return CausalRpc(request^, payload)


def _parallel_causal_rpc(mut mods: NativeModules, owner: PythonObject, device: PythonObject, request: WorkerRequest, payload: PythonObject,
                         captured: PythonObject) raises -> PythonObject:
    # UNPACK once, before the pool is touched: a payload the wire refuses is
    # a caller error, not a lost worker.
    var outgoing = wire_unpack(payload)
    return _parallel_causal_send(mods, owner, device, request, outgoing, captured)


def _parallel_causal_send(mut mods: NativeModules, owner: PythonObject, device: PythonObject, request: WorkerRequest,
                          outgoing: WireOutgoing, captured: PythonObject) raises -> PythonObject:
    """One addressed typed request whose payload record is already built."""
    var fields = _parallel_captured(mods, owner, captured)
    var record = _causal_record(owner)
    var rec = record.downcast_value_ptr[ParallelCausalOwner]()
    causal_registry_usable(rec[].native)
    var nk = parallel_keys()
    var index = causal_registry_worker(rec[].native, _parallel_word(device))
    var pool = rec[].pool
    if pool is None:
        raise Error("RuntimeError: ParallelCausalLM has no process owner")
    var result = PythonObject(None)
    var wire = request.encode()
    try:
        result = process_pool_request(pool, index, wire, outgoing)
        mods.released()
    except error:
        mods.released()
        if len(process_pool_pids(pool)) == 0:
            rec[].native.failed = True
            # Mirrors: the operation's namespace and the instance field the
            # shared model doors and callers read.
            py_setitem(fields, nk[].k__failed, True)
            var live = py_item(fields, nk[].k__causal_live_fields)
            model_dict(mods, live)
            py_setitem(live, nk[].k__failed, True)
        raise error
    return result


def _parallel_causal_rpc(mut mods: NativeModules, owner: PythonObject, device: PythonObject, request: WorkerRequest, payload: PythonObject) raises -> PythonObject:
    return _parallel_causal_rpc(mods, owner, device, request, payload, _parallel_owner_snapshot(mods, owner))


def _parallel_causal_close(mut mods: NativeModules, owner: PythonObject) raises:
    # A constructor can fail before the whole model exists; close admits only
    # the lifecycle fields it consumes and still publishes partial reap progress.
    var nk = parallel_keys()
    var fields = model_record_fields(mods, owner, py_item(PARALLEL_CAUSAL.get_or_create_ptr()[].classes, nk[].k_model),
        Python.tuple("_closed", "_pool", "_pools"))
    py_setitem(fields, nk[].k__closed, True)
    var live = py_get(fields, nk[].k__pool)
    # A constructor that failed before the pool existed has no record.
    var record = _causal_record_or_none(owner)
    if record is not None:
        var rec = record.downcast_value_ptr[ParallelCausalOwner]()
        rec[].native.closed = True
        live = rec[].pool
        rec[].pools = Python.dict()
    if live is not None:
        _parallel_pool_admit(mods, live)
        try:
            process_pool_close(live)
            mods.released()
        except error:
            mods.released()
            raise error
    model_dict(mods, fields)
    py_setitem(fields, nk[].k__pools, Python.dict())


def _parallel_causal_block(mut mods: NativeModules, owner: PythonObject, index: Int, weights: PythonObject,
                           captured: PythonObject) raises -> PythonObject:
    var p = PARALLEL_CAUSAL.get_or_create_ptr()
    var fields = _parallel_captured(mods, owner, captured)
    var nk = parallel_keys()
    var metadata = _parallel_causal_rpc(mods, owner, py_item(fields, nk[].k_layer_devices)[index], WorkerRequest(PC_BLOCK, index), weights, captured)
    model_dict(mods, metadata)
    var format = model_string(mods, py_item(metadata, nk[].k_weight_format))
    causal_layer_format_admit(format, model_string(mods, py_item(fields, nk[].k_weight_format)))
    var block = py_call(p[].bi.object.__new__, py_item(p[].classes, nk[].k_block))
    var block_fields = _parallel_block_fields(mods, block)
    py_setitem(block_fields, nk[].k_model, owner)
    py_setitem(block_fields, nk[].k_index, index)
    py_setitem(block_fields, nk[].k_weight_format, format)
    var vendor = model_string(mods, py_item(metadata, nk[].k_vendor))
    var mode = _parallel_word(py_item(metadata, nk[].k_numeric_mode))
    if index == 0:
        var live = py_item(fields, nk[].k__causal_live_fields)
        model_dict(mods, live)
        py_setitem(fields, nk[].k__native_vendor, vendor)
        py_setitem(fields, nk[].k__native_numeric_mode, mode)
        py_setitem(live, nk[].k__native_vendor, vendor)
        py_setitem(live, nk[].k__native_numeric_mode, mode)
        var entries = py_item(fields, nk[].k__native_entries)
        var entries_live = py_item(fields, nk[].k__native_parallel_entries_live)
        model_dict(mods, entries_live)
        for name in ["weight_names", "mamba1_profile"]:  # small-loop(two worker-reported immutable entry metadata fields)
            var value = model_plain(mods, metadata[name])
            py_setitem(entries, name, value)
            py_setitem(entries_live, name, value)
    else:
        causal_layer_owner_admit(mode, vendor, _parallel_word(py_item(fields, nk[].k__native_numeric_mode)),
                                 model_string(mods, py_item(fields, nk[].k__native_vendor)))
    return block


def _parallel_causal_block(mut mods: NativeModules, owner: PythonObject, index: Int, weights: PythonObject) raises -> PythonObject:
    return _parallel_causal_block(mods, owner, index, weights, _parallel_owner_snapshot(mods, owner))


def _parallel_causal_init(mut mods: NativeModules, owner: PythonObject, input_plan: PythonObject, input_weights: PythonObject,
                         layer_devices: PythonObject, options: PythonObject, child_module: String) raises:
    var p = PARALLEL_CAUSAL.get_or_create_ptr()
    var bi = p[].bi
    var live = _parallel_model_fields(mods, owner)
    var fields = Python.dict()
    var plan = model_plan_snapshot(mods, input_plan)
    var nk = parallel_keys()
    var devices = _parallel_causal_map(mods, layer_devices, _parallel_word(py_item(plan, nk[].k_n_layers)))
    var route = _parallel_causal_route(mods, devices)
    model_sequence(mods, options)
    var ordinary = _model_options_snapshot(mods, Python.list(options[0], options[1], route, options[2]), p[].context)
    var weights = _model_weights_snapshot(mods, input_weights, p[].context)
    _model_construction_set(mods, live, fields, "_closed", PythonObject(False))
    _model_construction_set(mods, live, fields, "_failed", PythonObject(False))
    _model_construction_set(mods, live, fields, "_pool", PythonObject(None))
    _model_construction_set(mods, live, fields, "_pools", Python.dict())
    _model_construction_set(mods, live, fields, "layer_devices", devices)
    _model_construction_set(mods, live, fields, "route", route)
    # The shared preparer fills this schema-only private record directly;
    # never reconstruct execution metadata from the published model fields.
    _ = _causal_model_prepare_snapshot(mods, owner, input_plan, plan, weights, ordinary, p[].context, live, fields)
    var entries = Python.dict()
    py_setitem(entries, nk[].k_parallel_call, p[].entry)
    var entries_live = py_call(bi.dict, entries)
    py_setitem(live, nk[].k__native_entries, entries_live)
    py_setitem(fields, nk[].k__native_entries, entries)
    var blocks = Python.list()
    _model_construction_set(mods, live, fields, "_blocks", blocks)
    _model_construction_set(mods, live, fields, "_head_int15", PythonObject(None))
    _model_construction_set(mods, live, fields, "_block_class", py_item(p[].classes, nk[].k_block))
    _model_construction_set(mods, live, fields, "_block_kwargs", py_call(bi.dict, py_item(plan, nk[].k_block_options) if String(py_item(plan, nk[].k_kind)) == "transformer" else py_item(plan, nk[].k_block_kwargs)))
    var prims = py_call(bi.object.__new__, py_item(p[].classes, nk[].k_primitives))
    py_setitem(model_record_fields(mods, prims, py_item(p[].classes, nk[].k_primitives), Python.tuple("model")), nk[].k_model, owner)
    _model_construction_set(mods, live, fields, "_prims", prims)
    var unique = Python.list()
    var indices = Python.dict()
    for i in range(len(devices)):  # driver-loop: stable first-appearance physical worker ordering
        if not py_contains(indices, devices[i]):
            py_setitem(indices, devices[i], len(unique))
            _ = py_call(unique.append, devices[i])
    py_setitem(live, nk[].k__worker_index, py_call(bi.dict, indices))
    py_setitem(fields, nk[].k__worker_index, indices)
    var pool = process_pool_create(py_call(bi.tuple, unique), False, child_module)
    mods.released()
    # Requests and replies travel as typed frames, as every pool's do
    # (bindings/native_process_wire_door.mojo).
    _model_construction_set(mods, live, fields, "_pool", pool)
    var pools = Python.dict()
    for i in range(len(unique)):  # driver-loop: selected process ownership aliases
        py_setitem(pools, unique[i], pool)
    py_setitem(live, nk[].k__pools, py_call(bi.dict, pools))
    py_setitem(fields, nk[].k__pools, pools)
    # PACK, once per model: the native owner. Its pins are the door's own
    # admitted objects (the instance fields above are copies for callers).
    py_setitem(live, nk[].k__native, PythonObject(alloc=ParallelCausalOwner(
        Int(py=py_call(bi.id, owner)), _parallel_device_words(devices), devices, indices, pool, pools)))
    _ = py_call(fields.update, _parallel_capture_model_fields(mods, owner, fields))
    py_setitem(fields, nk[].k__causal_original, owner)
    py_setitem(fields, nk[].k__causal_live_fields, live)
    py_setitem(fields, nk[].k__native_parallel_entries_live, entries_live)
    var captured = _causal_namespace(mods, fields)
    try:
        process_pool_start(py_item(fields, nk[].k__pool))
        mods.released()
        # The configure record, written once for every worker: (model
        # metadata, block-construction plan) as typed fields. REMOVED: the
        # two Python dicts built only to be sent (and _parallel_causal_wire_plan).
        var configure = WireOutgoing()
        configure.record.node(WIRE_LIST, 2)
        configure.record.node(WIRE_MAP, 6)
        for key in [String("device"), String("kind"), String("d_model"), String("vocab_size"), String("norm_eps"), String("numeric_profile")]:  # small-loop(six captured model primitive metadata words)
            wire_key(configure, key)
            wire_append(fields[key], configure)
        configure.record.node(WIRE_MAP, 5)
        for key in [String("kind"), String("d_model"), String("n_heads"), String("block_options"), String("block_kwargs")]:  # small-loop(block-construction metadata)
            wire_key(configure, key)
            wire_append(py_get(plan, key), configure)
        for i in range(len(unique)):  # driver-loop: worker initialization before constructing layers
            _ = _parallel_causal_send(mods, owner, unique[i], WorkerRequest(PC_CONFIGURE), configure, captured)
        for i in range(_parallel_word(py_item(plan, nk[].k_n_layers))):  # driver-loop: each model layer is built only by its assigned worker
            _ = py_call(blocks.append, _parallel_causal_block(mods, owner, i, py_item(weights, nk[].k_layers)[i], captured))
            _model_construction_set(mods, live, fields, "_blocks", blocks)
        var first = WireOutgoing()
        first.record.node(WIRE_MAP, 1)
        wire_key(first, "embed")
        wire_append(py_item(fields, nk[].k__embed), first)
        _ = _parallel_causal_send(mods, owner, devices[0], WorkerRequest(PC_TENSORS), first, captured)
        var last = WireOutgoing()
        last.record.node(WIRE_MAP, 2)
        wire_key(last, "norm")
        wire_append(py_item(fields, nk[].k__norm), last)
        wire_key(last, "head")
        wire_append(py_item(fields, nk[].k__head), last)
        var packed = _parallel_causal_send(mods, owner, devices[len(devices) - 1], WorkerRequest(PC_TENSORS), last, captured)
        model_dict(mods, live)
        py_setitem(live, nk[].k__head_int15, packed)
    except error:
        mods.released()
        try:
            _parallel_causal_close(mods, owner)
        except cleanup_error:
            raise Error("RuntimeError: ParallelCausalLM initialization failed: " + String(error) + "; cleanup also failed: " + String(cleanup_error))
        raise error


def _parallel_causal_options(mut mods: NativeModules, kwargs: PythonObject) raises -> PythonObject:
    var bi = PARALLEL_CAUSAL.get_or_create_ptr()[].bi
    model_dict(mods, kwargs)
    var keys = py_call(bi.list, kwargs)
    for i in range(len(keys)):  # driver-loop: over constructor keyword admission
        var name = String(keys[i])
        causal_option_admit(name)
    var nk = parallel_keys()
    return Python.list(py_get(kwargs, nk[].k_weight_format, "float32"), py_get(kwargs, nk[].k_max_positions), py_get(kwargs, nk[].k_numeric_profile))


def _parallel_causal_load(mut mods: NativeModules, cls: PythonObject, path_value: PythonObject, layer_devices: PythonObject, options: PythonObject, child_module: String) raises -> PythonObject:
    var p = PARALLEL_CAUSAL.get_or_create_ptr()
    var context = p[].context
    var devices = _parallel_causal_map(mods, layer_devices)
    var route = _parallel_causal_route(mods, devices)
    model_sequence(mods, options)
    var checked = _model_options_snapshot(mods, Python.list(options[0], options[1], route, options[2]), context)
    var admitted_options = Python.list(checked[0], checked[1], checked[3])
    var format = model_string(mods, admitted_options[0])
    causal_weight_format_admit(format)
    var path = native_path(path_value)
    if not native_path_exists(path, True):
        mods.released()
        raise Error("FileNotFoundError: ParallelCausalLM.load requires a checkpoint directory")
    mods.released()
    var nk = parallel_keys()
    var config = config_load_binding(mods, py_item(py_item(context, nk[].k_config_context), nk[].k_config_class), path)
    var plan = model_plan_for_binding(mods, config, py_item(context, nk[].k_config_context))
    var plan_fields = model_plan_snapshot(mods, plan)
    _ = _parallel_causal_map(mods, devices, _parallel_word(py_item(plan_fields, nk[].k_n_layers)))
    _ = _model_profile(mods, admitted_options[2], model_string(mods, py_item(plan_fields, nk[].k_kind)), context)
    var ck = py_item(context, nk[].k_checkpoint_context)
    var checkpoint = checkpoint_open_binding(mods, py_item(ck, nk[].k_checkpoint_class), path, ck)
    var weights = PythonObject(None)
    try:
        weights = causal_model_read_weights_binding(mods, checkpoint, plan, admitted_options[0], route, context)
    except error:
        mods.released()
        _ = checkpoint_close_binding(mods, checkpoint)
        raise error
    _ = checkpoint_close_binding(mods, checkpoint)
    var owner = py_call(p[].bi.object.__new__, cls)
    _parallel_causal_init(mods, owner, plan, weights, devices, admitted_options, child_module)
    return owner


struct RemoteStateLifetime(Movable, Writable):
    """Releases a remote layer state's worker state when the parent-side
    state object is freed. It replaces a Python `__del__`, which ran inside
    whichever native call dropped the last reference (a state allocated and
    released within one `forward` or `generate`, a layer list replaced by
    `reset_state`). `released` is set by an explicit release, so the worker
    is asked once."""
    var layer: PythonObject
    var token: PythonObject
    var released: Bool

    def __init__(out self, layer: PythonObject, token: PythonObject):
        self.layer = layer
        self.token = token
        self.released = False

    def write_to(self, mut writer: Some[Writer]):
        writer.write("_RemoteStateLifetime(released=", self.released, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)

    def __deinit__(deinit self):
        if self.released or self.token is None:
            return
        try:
            # A finalizer is its own operation: it admits its modules afresh.
            var mods = NativeModules()
            _parallel_causal_release_token(mods, self.layer, self.token)
        except:
            # Quiet, as the former finalizer: a closed or lost pool has no
            # worker state left to release.
            pass


def _parallel_causal_release_token(mut mods: NativeModules, block: PythonObject, token: PythonObject) raises:
    var nk = parallel_keys()
    var original = py_item(_parallel_block_fields(mods, block), nk[].k_model)
    var captured = _parallel_owner_snapshot(mods, original)
    var fields = causal_snapshot_fields(mods, captured)
    if _causal_serving(original):
        var layer = causal_snapshot_fields(mods, _parallel_find_block(mods, block, captured))
        _ = _parallel_causal_rpc(mods, py_item(fields, nk[].k__causal_original), py_item(fields, nk[].k_layer_devices)[py_item(layer, nk[].k_index)],
                                WorkerRequest(PC_RELEASE, _parallel_word(token)), PythonObject(None), captured)


def _parallel_causal_attach_fields(fields: PythonObject, layer: PythonObject, token: PythonObject) raises:
    if token is not None:
        _ = _parallel_word(token)
    var nk = parallel_keys()
    py_setitem(fields, nk[].k_layer, layer)
    py_setitem(fields, nk[].k_token, token)
    py_setitem(fields, nk[].k__lifetime, PythonObject(alloc=RemoteStateLifetime(layer, token)))


def _parallel_causal_attach(mut mods: NativeModules, remote: PythonObject, layer: PythonObject, token: PythonObject) raises:
    var fields = _parallel_state_fields(mods, remote)
    _ = _parallel_block_fields(mods, layer)
    _parallel_causal_attach_fields(fields, layer, token)


def _parallel_causal_allocate(mut mods: NativeModules, owner: PythonObject, index: Int, batch: PythonObject, capacity: PythonObject,
                              captured: PythonObject) raises -> PythonObject:
    var p = PARALLEL_CAUSAL.get_or_create_ptr()
    var fields = _parallel_captured(mods, owner, captured)
    var nk = parallel_keys()
    var blocks = py_item(fields, nk[].k__native_parallel_blocks)
    causal_allocation_index_admit(index, len(blocks))
    var allocate = WorkerRequest(PC_ALLOCATE, index)
    allocate.words.append(_parallel_word(batch))
    allocate.words.append(WORKER_NO_HANDLE if capacity is None else _parallel_word(capacity))
    var result = py_call(p[].bi.object.__new__, py_item(p[].classes, nk[].k_state))
    var destination = _parallel_state_fields(mods, result)
    var original = py_item(causal_snapshot_fields(mods, blocks[index]), nk[].k__causal_original)
    var token = _parallel_causal_rpc(mods, owner, py_item(fields, nk[].k_layer_devices)[index], allocate, PythonObject(None), captured)
    _parallel_causal_attach_fields(destination, original, token)
    return result


def _parallel_causal_allocate(mut mods: NativeModules, owner: PythonObject, index: Int, batch: PythonObject, capacity: PythonObject) raises -> PythonObject:
    return _parallel_causal_allocate(mods, owner, index, batch, capacity, _parallel_owner_snapshot(mods, owner))


def _parallel_causal_release(mut mods: NativeModules, remote: PythonObject, captured: PythonObject) raises:
    var fields = _parallel_captured(mods, remote, captured)
    var nk = parallel_keys()
    var token = py_item(fields, nk[].k_token)
    if token is None:
        return
    var model = py_item(fields, nk[].k__native_parallel_model)
    var owner = causal_snapshot_fields(mods, model)
    var block = causal_snapshot_fields(mods, py_item(fields, nk[].k__native_parallel_block))
    if _causal_serving(py_item(owner, nk[].k__causal_original)):
        _ = _parallel_causal_rpc(mods, py_item(owner, nk[].k__causal_original), py_item(owner, nk[].k_layer_devices)[py_item(block, nk[].k_index)],
                                WorkerRequest(PC_RELEASE, _parallel_word(token)), PythonObject(None), model)
    # Preserve the old failure contract: a failed release keeps the token and
    # lifetime live, allowing an explicit retry or later native destruction.
    py_setitem(fields, nk[].k_token, PythonObject(None))
    var live = py_item(fields, nk[].k__causal_live_fields)
    model_dict(mods, live)
    py_setitem(live, nk[].k_token, PythonObject(None))
    var lifetime = py_item(fields, nk[].k__lifetime)
    if lifetime is not None:
        lifetime.downcast_value_ptr[RemoteStateLifetime]()[].released = True


def _parallel_causal_release(mut mods: NativeModules, remote: PythonObject) raises:
    _parallel_causal_release(mods, remote, _parallel_capture_state(mods, remote))


def _parallel_causal_snapshot(mut mods: NativeModules, remote: PythonObject, captured: PythonObject) raises -> PythonObject:
    var source = _parallel_captured(mods, remote, captured)
    var nk = parallel_keys()
    if py_item(source, nk[].k_token) is None:
        raise Error("ParallelCausalLM: state released")
    var p = PARALLEL_CAUSAL.get_or_create_ptr()
    var model = py_item(source, nk[].k__native_parallel_model)
    var owner = causal_snapshot_fields(mods, model)
    var block = causal_snapshot_fields(mods, py_item(source, nk[].k__native_parallel_block))
    var fields = _parallel_causal_rpc(mods, py_item(owner, nk[].k__causal_original), py_item(owner, nk[].k_layer_devices)[py_item(block, nk[].k_index)],
                                     WorkerRequest(PC_STATE, _parallel_word(py_item(source, nk[].k_token))), PythonObject(None), model)
    var state = py_call(p[].bi.object.__new__, py_item(p[].context, nk[].k_generation_classes)[2][py_item(owner, nk[].k_kind)])
    _ = py_call(_parallel_causal_fields(state).update, fields)
    return state


def _parallel_causal_snapshot(mut mods: NativeModules, remote: PythonObject) raises -> PythonObject:
    return _parallel_causal_snapshot(mods, remote, _parallel_capture_state(mods, remote))


def _parallel_causal_block_run(mut mods: NativeModules, block: PythonObject, method: PythonObject, x: PythonObject, state: PythonObject,
                               captured_model: PythonObject, captured_block: PythonObject,
                               captured_state: PythonObject) raises -> PythonObject:
    var method_name = model_string(mods, method)
    var owner = causal_snapshot_fields(mods, captured_model)
    var layer = _parallel_captured(mods, block, captured_block)
    var token = WORKER_NO_HANDLE
    var nk = parallel_keys()
    if state is not None:
        var fields = _parallel_captured(mods, state, captured_state)
        if py_item(fields, nk[].k_layer) is not block or py_item(fields, nk[].k_token) is None:
            raise Error("ParallelCausalLM: state belongs to another layer or has been released")
        token = _parallel_word(py_item(fields, nk[].k_token))
    var index = py_item(layer, nk[].k_index)
    var run = causal_run_request(_parallel_word(index), causal_step_word(method_name), token)
    return _parallel_causal_rpc(mods, py_item(owner, nk[].k__causal_original), py_item(owner, nk[].k_layer_devices)[index], run, x, captured_model)


def _parallel_find_block(mut mods: NativeModules, block: PythonObject, model: PythonObject) raises -> PythonObject:
    var owner = causal_snapshot_fields(mods, model)
    var nk = parallel_keys()
    var blocks = py_item(owner, nk[].k__native_parallel_blocks)
    for i in range(len(blocks)):  # driver-loop: private RemoteBlock entry preserves original owner identity
        if py_item(causal_snapshot_fields(mods, blocks[i]), nk[].k__causal_original) is block:
            return blocks[i]
    # Private RemoteBlock methods historically accept a separately retained
    # handle for an existing worker index. Preserve that handle's own identity;
    # it need not be the handle currently in model._blocks.
    var extra = py_get(owner, nk[].k__native_parallel_extra_blocks)
    if extra is None:
        extra = Python.list()
        py_setitem(owner, nk[].k__native_parallel_extra_blocks, extra)
    for i in range(len(extra)):  # driver-loop: deduplicate additional retained remote handles
        if py_item(causal_snapshot_fields(mods, extra[i]), nk[].k__causal_original) is block:
            return extra[i]
    var live = _parallel_block_fields(mods, block)
    if py_item(live, nk[].k_model) is not py_item(owner, nk[].k__causal_original):
        raise Error("ParallelCausalLM: remote block belongs to another model")
    var index = _parallel_word(py_item(live, nk[].k_index))
    causal_block_index_admit(index, len(py_item(owner, nk[].k_layer_devices)))
    var fields = Python.dict()
    py_setitem(fields, nk[].k_model, py_item(live, nk[].k_model))
    py_setitem(fields, nk[].k_index, index)
    py_setitem(fields, nk[].k_weight_format, model_string(mods, py_item(live, nk[].k_weight_format)))
    py_setitem(fields, nk[].k__causal_original, block)
    var captured = _causal_namespace(mods, fields)
    _ = py_call(extra.append, captured)
    return captured


def _parallel_causal_block_run(mut mods: NativeModules, block: PythonObject, method: PythonObject, x: PythonObject, state: PythonObject) raises -> PythonObject:
    var layer = _parallel_block_fields(mods, block)
    var nk = parallel_keys()
    var model = _parallel_owner_snapshot(mods, py_item(layer, nk[].k_model))
    var captured = PythonObject(None)
    if state is not None:
        captured = _parallel_capture_state(mods, state, model)
    return _parallel_causal_block_run(mods, block, method, x, state, model, _parallel_find_block(mods, block, model), captured)


@fieldwise_init
struct LayerItem(ImplicitlyCopyable, Movable):
    """One layer of a "run_layers" request: its index and its state token
    (WORKER_NO_HANDLE when it runs without a state)."""
    var index: Int
    var token: Int


def _parallel_layer_item(mut mods: NativeModules, captured_block: PythonObject, captured_state: PythonObject,
                         captured_model: PythonObject) raises -> LayerItem:
    """(layer index, state token) of one layer of a "run_layers" request,
    with the checks `_parallel_causal_block_run` makes for a "run"."""
    var nk = parallel_keys()
    var block = py_item(causal_snapshot_fields(mods, captured_block), nk[].k__causal_original)
    var state = py_item(causal_snapshot_fields(mods, captured_state), nk[].k__causal_original)
    _ = causal_snapshot_fields(mods, captured_model)
    var layer = _parallel_captured(mods, block, captured_block)
    var token = WORKER_NO_HANDLE
    if state is not None:
        var fields = _parallel_captured(mods, state, captured_state)
        if py_item(fields, nk[].k_layer) is not block or py_item(fields, nk[].k_token) is None:
            raise Error("ParallelCausalLM: state belongs to another layer or has been released")
        token = _parallel_word(py_item(fields, nk[].k_token))
    return LayerItem(_parallel_word(py_item(layer, nk[].k_index)), token)


def parallel_causal_call(mut mods: NativeModules, operation: Int, name: PythonObject, args: PythonObject) raises -> PythonObject:
    """`operation` is a `PARALLEL_CAUSAL_OPN_*` code (-1 for a name the table
    does not hold); `name` is the caller's exact str, read only for the
    unknown operation's text."""
    var p = PARALLEL_CAUSAL.get_or_create_ptr()
    var child = String("mojolearn._native_parallel_causal_worker")
    model_sequence(mods, args)
    var nk = parallel_keys()
    if operation == PARALLEL_CAUSAL_OPN_INITIALIZE:
        parallel_causal_initialize(mods, args)
    elif operation == PARALLEL_CAUSAL_OPN_WORKER:
        parallel_causal_worker(mods, args)
    elif operation == PARALLEL_CAUSAL_OPN_INIT:
        _parallel_causal_init(mods, args[0], args[1], args[2], args[3], _parallel_causal_options(mods, args[4]), child)
    elif operation == PARALLEL_CAUSAL_OPN_LOAD or operation == PARALLEL_CAUSAL_OPN_LOAD_WITH_WORKER:
        return _parallel_causal_load(mods, args[0], args[1], args[2], args[3], child if operation == PARALLEL_CAUSAL_OPN_LOAD else model_string(mods, args[4]))
    elif operation == PARALLEL_CAUSAL_OPN_RPC:
        var typed = _parallel_causal_typed(mods, model_string(mods, args[2]), args[3])
        return _parallel_causal_rpc(mods, args[0], args[1], typed.request, typed.payload)
    elif operation == PARALLEL_CAUSAL_OPN_CAPTURE_MODEL:
        return _parallel_capture_model(mods, args[0], args[1]) if len(args) > 1 else _parallel_capture_model(mods, args[0])
    elif operation == PARALLEL_CAUSAL_OPN_CAPTURE_STATE:
        return _parallel_capture_state(mods, args[0], args[1]) if len(args) > 1 else _parallel_capture_state(mods, args[0])
    elif operation == PARALLEL_CAUSAL_OPN_PARAMETERS:
        var model = causal_model_capture(mods, args[0], p[].context)
        _parallel_causal_usable(args[0])
        var result = PythonObject(None)
        try:
            result = causal_lm_parameters_binding(mods, model)
        except error:
            mods.released()
            causal_model_publish(mods, model)
            raise error
        causal_model_publish(mods, model)
        # args[1] (lane api-torch round 3): the torch result form of
        # bindings/torch_parameter_form_pack.mojo (0 the {name: array}
        # mapping, state_dict; 1 parameters(); 2 named_parameters()).
        return torch_parameter_form(result, Int(py=args[1]) if len(args) > 1 else 0)
    elif operation == PARALLEL_CAUSAL_OPN_BLOCK_FIELDS:
        var captured = args[2] if len(args) > 2 else _parallel_owner_snapshot(mods, args[0])
        var fields = _parallel_captured(mods, args[0], captured)
        var index = _parallel_word(args[1])
        return _parallel_causal_rpc(mods, args[0], py_item(fields, nk[].k_layer_devices)[index], WorkerRequest(PC_BLOCK_FIELDS, index), PythonObject(None), captured)
    elif operation == PARALLEL_CAUSAL_OPN_ALLOCATE_LAYER:
        var captured = args[4] if len(args) > 4 else _parallel_owner_snapshot(mods, args[0])
        return _parallel_causal_allocate(mods, args[0], _parallel_word(args[1]), args[2], args[3], captured)
    elif operation == PARALLEL_CAUSAL_OPN_BLOCK_ALLOCATE:
        var block = _parallel_block_fields(mods, args[0])
        return _parallel_causal_allocate(mods, py_item(block, nk[].k_model), _parallel_word(py_item(block, nk[].k_index)), args[1], args[2])
    elif operation == PARALLEL_CAUSAL_OPN_BLOCK_RUN:
        return _parallel_causal_block_run(mods, args[0], args[1], args[2], args[3])
    elif operation == PARALLEL_CAUSAL_OPN_BLOCK_CALL:
        var original = py_item(_parallel_block_fields(mods, args[0]), nk[].k_model)
        var captured = _parallel_owner_snapshot(mods, original)
        var fields = causal_snapshot_fields(mods, captured)
        var block = causal_snapshot_fields(mods, _parallel_find_block(mods, args[0], captured))
        var typed = _parallel_causal_typed(mods, model_string(mods, args[1]), args[2])
        return _parallel_causal_rpc(mods, py_item(fields, nk[].k__causal_original), py_item(fields, nk[].k_layer_devices)[py_item(block, nk[].k_index)], typed.request, typed.payload, captured)
    elif operation == PARALLEL_CAUSAL_OPN_BLOCK_INIT:
        var destination = _parallel_block_fields(mods, args[0])
        var created = _parallel_causal_block(mods, args[1], _parallel_word(args[2]), args[3])
        _ = py_call(destination.update, _parallel_block_fields(mods, created))
    # TOMBSTONE: operation "make_blocks" (dead code: its one caller, ParallelCausalLM._make_blocks, had no
    # caller) deleted 2026-11-09 by 57b86ba5ac; code recoverable at 57b86ba5ac^.
    # Restore: git apply experiments/removed/dead_code_p5_20261109.patch
    elif operation == PARALLEL_CAUSAL_OPN_STATE_INIT:
        _parallel_causal_attach(mods, args[0], args[1], args[2])
    elif operation == PARALLEL_CAUSAL_OPN_STATE_SNAPSHOT:
        return _parallel_causal_snapshot(mods, args[0], args[1]) if len(args) > 1 else _parallel_causal_snapshot(mods, args[0])
    elif operation == PARALLEL_CAUSAL_OPN_STATE_PUBLISH:
        var remote = args[0]
        var captured = args[2] if len(args) > 2 else _parallel_capture_state(mods, remote)
        var fields = _parallel_captured(mods, remote, captured)
        if py_item(fields, nk[].k_token) is None:
            raise Error("ParallelCausalLM: state released")
        var model = py_item(fields, nk[].k__native_parallel_model)
        var owner = causal_snapshot_fields(mods, model)
        var block = causal_snapshot_fields(mods, py_item(fields, nk[].k__native_parallel_block))
        return _parallel_causal_rpc(mods, py_item(owner, nk[].k__causal_original), py_item(owner, nk[].k_layer_devices)[py_item(block, nk[].k_index)],
                                    WorkerRequest(PC_PUBLISH_STATE, _parallel_word(py_item(fields, nk[].k_token))), _parallel_causal_fields(args[1]), model)
    elif operation == PARALLEL_CAUSAL_OPN_STATE_CLOSE:
        _parallel_causal_release(mods, args[0])
    elif operation == PARALLEL_CAUSAL_OPN_RELEASE_LAYERS:
        var owner = _parallel_captured(mods, args[0], args[2])
        var state = _parallel_captured(mods, args[1], args[3])
        var layers = py_item(state, nk[].k_layers)
        var failure = String()
        for i in range(len(layers)):  # driver-loop: over remote states owned by the caller's model state
            try:
                var fields = causal_snapshot_fields(mods, layers[i])
                _parallel_causal_release(mods, py_item(fields, nk[].k__causal_original), layers[i])
            except error:
                mods.released()
                if failure == "":
                    failure = String(error)
        if failure != "":
            raise Error(failure)
    elif operation == PARALLEL_CAUSAL_OPN_RUN:
        var owner = args[0]
        var ids = generation_array_snapshot(args[1])
        var length = Int(py=ids.shape[1])
        var state = args[2]
        var model_fields = _parallel_captured(mods, owner, args[4])
        var state_fields = _parallel_captured(mods, state, args[5])
        var devices = py_item(model_fields, nk[].k_layer_devices)
        var blocks = py_item(model_fields, nk[].k__native_parallel_blocks)
        var layers = py_item(state_fields, nk[].k_layers)
        # The legacy parallel API used each block's dedicated decode entry
        # after prefill. Preserve that route independently of ordinary CausalLM.
        var step = 1 if _parallel_flag(args[3]) else 0
        # Lane p3-msel-parallel-and-process-pool (audit B worst path 9: one
        # request per layer plus the embedding and the head, n_layers + 2 per
        # token batch): consecutive layers on one device go in ONE
        # "run_layers" request, with the embedding folded into the first and
        # the head into the last when they live on that device: one request
        # per device per token batch. REMOVED: the per-layer request loop.
        var nb = len(blocks)
        var embed_device = devices[0]
        var head_device = devices[len(devices) - 1]
        # The admitted layer map as words: the device runs are cut by word
        # compares (parallel_causal_native_run.mojo), the device object of a
        # run stays the pool's key.
        var embed_word = _parallel_word(embed_device)
        var head_word = _parallel_word(head_device)
        var x = PythonObject(None)
        var embedded = False
        if nb == 0:
            x = _parallel_causal_rpc(mods, owner, embed_device, WorkerRequest(PC_EMBEDDING), ids, args[4])
            embedded = True
        var i = 0
        var result = PythonObject(None)
        var finished = False
        while i < nb:  # driver-loop: over runs of consecutive layers on one device
            var items = List[Int]()
            # Optional, not a PythonObject read and reassigned in a loop (modular/modular#7546 pattern; runtime-fix-save-forest sweep).
            var run_device = Optional[PythonObject]()
            var run_word = 0
            var j = i
            while j < nb:  # driver-loop: over the layers of this device run
                var item = _parallel_layer_item(mods, blocks[j], layers[j], args[4])
                var layer_device = py_item(model_fields, nk[].k_layer_devices)[item.index]
                var layer_word = _parallel_word(layer_device)
                if not causal_run_joins(j - i, layer_word, run_word):
                    break
                run_device = layer_device
                run_word = layer_word
                items.append(item.index)
                items.append(item.token)
                j += 1
            # The run holds at least layer i (i < nb), so the device is set.
            var device = run_device.value()
            var first_ids = PythonObject(None)
            if not embedded:
                if causal_run_embeds(run_word, embed_word):
                    first_ids = ids
                else:
                    x = _parallel_causal_rpc(mods, owner, embed_device, WorkerRequest(PC_EMBEDDING), ids, args[4])
                embedded = True
            var last = causal_run_final(j, nb, run_word, head_word)
            var run = causal_layers_request(step, first_ids is not None, last, items)
            x = _parallel_causal_rpc(mods, owner, device, run, Python.list(x, first_ids), args[4])
            if last:
                result = x
                finished = True
            i = j
        if not finished:
            result = _parallel_causal_rpc(mods, owner, head_device, WorkerRequest(PC_FINAL), x, args[4])
        py_setitem(state_fields, nk[].k_positions, _parallel_word(py_item(state_fields, nk[].k_positions)) + length)
        return result
    elif operation == PARALLEL_CAUSAL_OPN_CLOSE or operation == PARALLEL_CAUSAL_OPN_CLOSE_QUIET or operation == PARALLEL_CAUSAL_OPN_EXIT:
        _parallel_causal_close(mods, args[0])
    elif operation == PARALLEL_CAUSAL_OPN_ENTER:
        _ = _parallel_model_fields(mods, args[0])
        var entered = _causal_record(args[0])
        causal_closed_admit(entered.downcast_value_ptr[ParallelCausalOwner]()[].native.closed)
        return args[0]
    elif operation == PARALLEL_CAUSAL_OPN_PRIMITIVES_INIT:
        var fields = model_record_fields(mods, args[0], py_item(p[].classes, nk[].k_primitives), Python.tuple("model"))
        _ = _parallel_model_fields(mods, args[1])
        py_setitem(fields, nk[].k_model, args[1])
    elif operation == PARALLEL_CAUSAL_OPN_PRIMITIVE:
        var fields = model_record_fields(mods, args[0], py_item(p[].classes, nk[].k_primitives), Python.tuple("model"))
        var owner = py_item(fields, nk[].k_model)
        var captured = _parallel_owner_snapshot(mods, owner)
        var devices = py_item(causal_snapshot_fields(mods, captured), nk[].k_layer_devices)
        var name = model_string(mods, args[1])
        var device = devices[0 if name == "embedding" else len(devices) - 1]
        var typed = _parallel_causal_typed(mods, name, args[2])
        return _parallel_causal_rpc(mods, owner, device, typed.request, typed.payload, captured)
    # The "resident" operation (ParallelCausalLM._generate_resident, always
    # None) is gone with the private name; `generate(..., resident=True)`
    # refuses a layer-worker model in bindings/causal_lm_public_door.mojo
    # (lane verify-open-sites, 2026-11-16).
    else:
        raise Error("ParallelCausalLM: unknown native operation " + String(name))
    return PythonObject(None)


def parallel_causal_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    var mods = NativeModules()
    var args = PythonObject(from_borrowed=args_ptr)
    var operation = -1
    try:
        model_sequence(mods, args)
        # model_string's exact-str refusal, without building the String: the
        # name is one lookup of the caller's own str in the operation table.
        if py_type(args[0]) is not mods.module("builtins").str:
            raise Error("TypeError: model metadata requires exact strings")
        var nk = parallel_keys()
        operation = py_op_code(nk[].ops_causal, args[0])
        var result = parallel_causal_call(mods, operation, args[0], args[1])
        return result.steal_data()
    except error:
        if operation == PARALLEL_CAUSAL_OPN_CLOSE_QUIET:
            var nil = PythonObject(None)
            return nil.steal_data()
        var message = String(error)
        if message.startswith("InterruptedError:"):
            return raise_python_exception(error_without_class(error, "InterruptedError"), ExceptionType("PyExc_InterruptedError"))
        if message.startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        if message.startswith("TypeError:"):
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        if message.startswith("FileNotFoundError:"):
            return raise_python_exception(error_without_class(error, "FileNotFoundError"), ExceptionType("PyExc_FileNotFoundError"))
        if message.startswith("NotImplementedError:"):
            return raise_python_exception(error_without_class(error, "NotImplementedError"), ExceptionType("PyExc_NotImplementedError"))
        if message.startswith("UnsupportedModel:"):
            try:
                ref cpy = Python().cpython()
                var context = cpy.PyTuple_GetItem(args_ptr, 2)
                var config = causal_borrowed_item(context, "config_context")
                return causal_set_exception(error_without_class(error, "UnsupportedModel"), causal_borrowed_item(config, "unsupported_class"))
            except nested:
                return raise_python_exception(nested)
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
