# SPDX-License-Identifier: Apache-2.0
"""Native ownership for the existing sequential layer-partitioned CausalLM."""
from std.ffi import _Global
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_error_class import error_without_class
from bindings.generation_buffers import GENERATION_BOUNDARY, generation_initialize_binding, generation_copy, generation_array, generation_array_snapshot, generation_address, py_ints
from bindings.causal_lm_construct import causal_model_read_weights_binding, _causal_model_prepare_snapshot, _model_options_snapshot, _model_weights_snapshot, _model_block, _model_bindings, _model_profile, _model_quantize15
from bindings.causal_lm_public import causal_model_embedding, causal_model_head, causal_lm_parameters_binding, _native_function
from bindings.neural_block_public import neural_block_owner, neural_block_allocate_binding, neural_block_forward_binding
from bindings.causal_model_config import config_load_binding, model_plan_for_binding, model_plan_snapshot, model_record_fields, model_dict, model_plain, model_sequence, model_string
from bindings.causal_operation_owner import causal_snapshot_fields, _causal_namespace, causal_model_capture, causal_model_publish
from bindings.mode_metadata_native import _fields
from bindings.native_loaded_module import native_loaded_module
from bindings.causal_checkpoint import checkpoint_open_binding, checkpoint_close_binding
from bindings.causal_exception import causal_borrowed_item, causal_set_exception
from bindings.native_json import native_path, native_path_exists
from bindings.native_process_pool import process_pool_create, process_pool_start, process_pool_call, process_pool_close, process_worker_connect, process_worker_next, process_send
from bindings.native_transport import transport_close, _transport_state
from bindings.py_balanced import py_call, py_contains, py_get, py_setattr, py_setitem, py_type


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


comptime PARALLEL_CAUSAL = _Global[
    StorageType=ParallelCausalBoundary, name="MojoParallelCausalOwner",
    init_fn=ParallelCausalBoundary.__init__,
]


def parallel_causal_initialize(args: PythonObject) raises:
    var p = PARALLEL_CAUSAL.get_or_create_ptr()
    var bi = native_loaded_module("builtins")
    model_sequence(args)
    if len(args) != 3:
        raise Error("TypeError: parallel connection requires context, classes and native entry")
    model_dict(args[0])
    model_dict(args[1])
    var classes = Python.dict()
    for name in ["model", "block", "state", "primitives"]:  # small-loop: retain the import-connected record identities
        var cls = args[1][name]
        if py_type(cls) is not bi.type:
            raise Error("TypeError: parallel connection requires ordinary class identities")
        if p[].classes is not None and p[].classes[name] is not cls:
            raise Error("TypeError: parallel record identities cannot be reconnected")
        py_setitem(classes, name, cls)
    if py_type(args[2]) is not py_type(bi.len):
        raise Error("ParallelCausalLM: execution router must be a compiled native entry")
    if py_type(args[2].__self__) is not native_loaded_module("types").ModuleType or args[2].__self__ is bi:
        raise Error("ParallelCausalLM: execution router must belong to its native module")
    if p[].entry is not None:
        if p[].entry is not args[2]:
            raise Error("TypeError: parallel execution router cannot be reconnected")
        return
    p[].bi = bi
    p[].context = py_call(bi.dict, args[0])
    p[].classes = classes
    p[].entry = args[2]
    _ = generation_initialize_binding(p[].context["generation_classes"])


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


def _parallel_model_fields(owner: PythonObject) raises -> PythonObject:
    return model_record_fields(owner, PARALLEL_CAUSAL.get_or_create_ptr()[].classes["model"],
        Python.tuple("layer_devices", "_worker_index", "_pool", "_pools", "_closed", "_failed", "_blocks", "kind",
            "weight_format", "_native_entries", "_native_vendor", "_native_numeric_mode", "route", "_head_int15", "_block_class", "_block_kwargs", "_prims"))


def _parallel_block_fields(block: PythonObject) raises -> PythonObject:
    return model_record_fields(block, PARALLEL_CAUSAL.get_or_create_ptr()[].classes["block"],
        Python.tuple("model", "index", "weight_format"))


def _parallel_state_fields(state: PythonObject) raises -> PythonObject:
    return model_record_fields(state, PARALLEL_CAUSAL.get_or_create_ptr()[].classes["state"],
        Python.tuple("layer", "token", "_lifetime"))


def _parallel_int_keys(value: PythonObject) raises -> PythonObject:
    var bi = PARALLEL_CAUSAL.get_or_create_ptr()[].bi
    if py_type(value) is not bi.dict:
        raise Error("TypeError: parallel routing requires exact dictionary storage")
    var keys = py_call(bi.list, value.keys())
    for i in range(len(keys)):  # driver-loop: admit keys before dictionary hashing or equality
        _ = _parallel_word(keys[i])
    return keys


def _parallel_pool_capture(pool: PythonObject) raises -> PythonObject:
    # Native process helpers mutate their resource record. Give them an admitted
    # operation-owned record, retain the socket capsules, and publish lifecycle
    # words back after every request, including failed transport/reap paths.
    model_dict(pool)
    var copy = Python.dict()
    var devices = pool["devices"]
    model_sequence(devices)
    for i in range(len(devices)):  # driver-loop: capture native worker device metadata
        _ = _parallel_word(devices[i])
    py_setitem(copy, "devices", py_call(PARALLEL_CAUSAL.get_or_create_ptr()[].bi.tuple, devices))
    py_setitem(copy, "cooperative", _parallel_flag(pool["cooperative"]))
    py_setitem(copy, "closed", _parallel_flag(pool["closed"]))
    py_setitem(copy, "sequence", _parallel_word(pool["sequence"]))
    py_setitem(copy, "module", model_string(pool["module"]))
    if py_contains(pool, "environment"):
        py_setitem(copy, "environment", model_plain(pool["environment"]))
    var workers = pool["workers"]
    model_sequence(workers)
    var captured = Python.list()
    for i in range(len(workers)):  # driver-loop: retain native sockets and snapshot each worker record
        var worker = workers[i]
        model_dict(worker)
        var item = Python.dict()
        py_setitem(item, "pid", _parallel_word(worker["pid"]))
        var socket = worker["socket"]
        if socket is not None:
            _ = _transport_state(socket, False)
        py_setitem(item, "socket", socket)
        var group = worker["devices"]
        model_sequence(group)
        for j in range(len(group)):  # small-loop: validate physical worker group indices
            _ = _parallel_word(group[j])
        py_setitem(item, "devices", py_call(PARALLEL_CAUSAL.get_or_create_ptr()[].bi.tuple, group))
        if py_contains(worker, "identity"):
            py_setitem(item, "identity", model_plain(worker["identity"]))
        _ = py_call(captured.append, item)
    py_setitem(copy, "workers", captured)
    return copy


def _parallel_capture_model_fields(owner: PythonObject, live: PythonObject) raises -> PythonObject:
    model_dict(live)
    var result = Python.dict()
    var devices = _parallel_causal_map(live["layer_devices"])
    py_setitem(result, "layer_devices", devices)
    py_setitem(result, "_closed", _parallel_flag(live["_closed"]))
    py_setitem(result, "_failed", _parallel_flag(live["_failed"]))
    var original_pool = live["_pool"]
    var pool = PythonObject(None)
    if original_pool is not None:
        pool = _parallel_pool_capture(original_pool)
    py_setitem(result, "_pool", pool)
    py_setitem(result, "_native_parallel_pool_live", original_pool)
    var indices = live["_worker_index"]
    var keys = _parallel_int_keys(indices)
    var copied = Python.dict()
    for i in range(len(keys)):  # driver-loop: copy addressed-worker routing words once
        py_setitem(copied, keys[i], _parallel_word(indices[keys[i]]))
    py_setitem(result, "_worker_index", copied)
    var pools = live["_pools"]
    keys = _parallel_int_keys(pools)
    copied = Python.dict()
    for i in range(len(keys)):  # driver-loop: preserve aliases to the one native process owner
        if pools[keys[i]] is not original_pool:
            raise Error("TypeError: parallel routing has an inconsistent process owner")
        py_setitem(copied, keys[i], pool)
    py_setitem(result, "_pools", copied)
    var blocks = live["_blocks"]
    model_sequence(blocks)
    var captured = Python.list()
    var seen = Python.dict()
    var bi = PARALLEL_CAUSAL.get_or_create_ptr()[].bi
    for i in range(len(blocks)):  # driver-loop: retain original block identities and capture each owner once
        var key = py_call(bi.id, blocks[i])
        var block = py_get(seen, key)
        if block is None:
            var source = _parallel_block_fields(blocks[i])
            if source["model"] is not owner:
                raise Error("ParallelCausalLM: remote block belongs to another model")
            var index = _parallel_word(source["index"])
            if index < 0 or index >= len(devices):
                raise Error("ParallelCausalLM: remote block index outside its layer map")
            var fields = Python.dict()
            py_setitem(fields, "model", owner)
            py_setitem(fields, "index", index)
            py_setitem(fields, "weight_format", model_string(source["weight_format"]))
            py_setitem(fields, "_causal_original", blocks[i])
            block = _causal_namespace(fields)
            py_setitem(seen, key, block)
        _ = py_call(captured.append, block)
    py_setitem(result, "_native_parallel_blocks", captured)
    return result


def _parallel_capture_model(owner: PythonObject) raises -> PythonObject:
    return _parallel_capture_model_fields(owner, _parallel_model_fields(owner))


def _parallel_capture_model(owner: PythonObject, copied: PythonObject) raises -> PythonObject:
    _ = _parallel_model_fields(owner)  # admit added parallel descriptors without recapturing values
    return _parallel_capture_model_fields(owner, copied)


def _parallel_owner_snapshot(owner: PythonObject) raises -> PythonObject:
    # Direct RemoteBlock/RemoteState methods enter here once. Shared CausalLM
    # operations supply their already captured model namespace instead.
    var live = _parallel_model_fields(owner)
    var fields = _parallel_capture_model_fields(owner, live)
    py_setitem(fields, "kind", model_string(live["kind"]))
    py_setitem(fields, "weight_format", model_string(live["weight_format"]))
    for name in ["_native_vendor", "_native_numeric_mode"]:  # small-loop: primitive layer construction metadata
        if py_contains(live, name):
            py_setitem(fields, name, model_plain(live[name]))
    var entries = live["_native_entries"]
    model_dict(entries)
    py_setitem(fields, "_native_entries", py_call(PARALLEL_CAUSAL.get_or_create_ptr()[].bi.dict, entries))
    py_setitem(fields, "_native_parallel_entries_live", entries)
    py_setitem(fields, "_causal_original", owner)
    py_setitem(fields, "_causal_live_fields", live)
    return _causal_namespace(fields)


def _parallel_captured(owner: PythonObject, captured: PythonObject) raises -> PythonObject:
    var fields = causal_snapshot_fields(captured)
    model_dict(fields)
    if fields["_causal_original"] is not owner:
        raise Error("TypeError: parallel metadata belongs to another owner")
    var failure = py_get(fields, "_causal_admission_failure")
    if failure is not None:
        raise Error(model_string(failure))
    return fields


def _parallel_capture_state_fields(remote: PythonObject, live: PythonObject, model: PythonObject) raises -> PythonObject:
    var layer = live["layer"]
    var block = _parallel_find_block(layer, model)
    var token = live["token"]
    if token is not None:
        _ = _parallel_word(token)
    var lifetime = py_get(live, "_lifetime")
    if lifetime is not None:
        _ = lifetime.downcast_value_ptr[RemoteStateLifetime]()
    var result = Python.dict()
    py_setitem(result, "layer", layer)
    py_setitem(result, "token", token)
    py_setitem(result, "_lifetime", lifetime)
    py_setitem(result, "_causal_original", remote)
    py_setitem(result, "_native_state_identity", remote)
    py_setitem(result, "_causal_live_fields", live)
    py_setitem(result, "_causal_initial_buffers", Python.dict())
    py_setitem(result, "_native_parallel_model", model)
    py_setitem(result, "_native_parallel_block", block)
    return _causal_namespace(result)


def _parallel_capture_state(remote: PythonObject, model: PythonObject) raises -> PythonObject:
    return _parallel_capture_state_fields(remote, _parallel_state_fields(remote), model)


def _parallel_capture_state(remote: PythonObject) raises -> PythonObject:
    var live = _parallel_state_fields(remote)
    var layer = _parallel_block_fields(live["layer"])
    return _parallel_capture_state_fields(remote, live, _parallel_owner_snapshot(layer["model"]))


def _parallel_causal_route(devices: PythonObject) raises -> PythonObject:
    var p = PARALLEL_CAUSAL.get_or_create_ptr()
    model_sequence(devices)
    var vendor = String(p[].context["selected_vendor"])
    if vendor == "metal":
        if len(devices) == 0:
            raise Error("Metal ParallelCausalLM requires every layer owner to be device 0")
        for i in range(len(devices)):  # driver-loop: over explicit layer ownership metadata
            if py_type(devices[i]) is not p[].bi.int or Int(py=devices[i]) != 0:
                raise Error("Metal ParallelCausalLM requires every layer owner to be device 0")
        return PythonObject("gpu")
    if vendor == "cuda" or vendor == "hip":
        return PythonObject("gpu")
    if vendor == "cpu":
        return PythonObject("cpu")
    raise Error("NotImplementedError: ParallelCausalLM requires CUDA or HIP device isolation, single-device Metal, or a CPU-only install")


def _parallel_causal_map(value: PythonObject, expected: Int = -1) raises -> PythonObject:
    var bi = PARALLEL_CAUSAL.get_or_create_ptr()[].bi
    model_sequence(value)
    var devices = py_call(bi.tuple, value)
    if len(devices) == 0 or (expected >= 0 and len(devices) != expected):
        raise Error("layer_devices requires one nonnegative device index per layer")
    for i in range(len(devices)):  # driver-loop: over layer-to-worker indices
        if py_type(devices[i]) is not bi.int or Int(py=devices[i]) < 0:
            raise Error("layer_devices requires one nonnegative device index per layer")
    _ = _parallel_causal_route(devices)
    return devices


def _parallel_causal_wire_plan(plan: PythonObject) raises -> PythonObject:
    var bi = PARALLEL_CAUSAL.get_or_create_ptr()[].bi
    var result = bi.dict()
    for key in [String("kind"), String("d_model"), String("n_heads"), String("block_options"), String("block_kwargs")]:  # driver-loop: over block-construction metadata
        py_setitem(result, key, py_get(plan, key))
    return result


def _parallel_causal_worker_request(world: PythonObject, operation: String, args: PythonObject) raises -> PythonObject:
    var p = PARALLEL_CAUSAL.get_or_create_ptr()
    var bi = p[].bi
    var context = p[].context
    if operation == "configure":
        if py_get(world, "top") is not None:
            raise Error("ParallelCausalLM: worker already configured")
        var route = PythonObject("cpu" if String(context["selected_vendor"]) == "cpu" else "gpu")
        if Bool(py=args[0]["device"] != route):
            raise Error("ParallelCausalLM: worker route differs from its connected backend")
        var plan = py_call(bi.object.__new__, context["config_context"]["plan_class"])
        _ = py_call(_parallel_causal_fields(plan).update, args[1])
        var top = py_call(bi.object.__new__, context["model_class"])
        _ = py_call(_parallel_causal_fields(top).update, args[0])
        py_setattr(top, "_embed", PythonObject(None))
        py_setattr(top, "_norm", PythonObject(None))
        py_setattr(top, "_head", PythonObject(None))
        py_setattr(top, "_head_int15", PythonObject(None))
        var bindings = _model_bindings(route, String(plan.kind), context)
        var entries = bi.dict()
        for key in [String("embedding_forward"), String("rms_norm_forward"), String("linear_forward")]:  # driver-loop: over the three existing model primitive entries
            py_setitem(entries, key, _native_function(bindings["primitives"], key))
        if String(top.numeric_profile) == "fixed15_v1":
            py_setitem(entries, "quantize15", _native_function(bindings["linalg"], "quantize_int15"))
            py_setitem(entries, "gemm15", _native_function(bindings["linalg"], "gemm_int15"))
        py_setattr(top, "_native_entries", entries)
        py_setitem(world, "plan", plan)
        py_setitem(world, "top", top)
        py_setitem(world, "bindings", bindings)
        py_setitem(world, "blocks", bi.dict())
        py_setitem(world, "states", bi.dict())
        py_setitem(world, "next_state", 0)
        return PythonObject(True)
    var top = py_get(world, "top")
    if top is None:
        raise Error("ParallelCausalLM: worker was not told its route")
    if operation == "block":
        var index = args[0]
        if py_contains(world["blocks"], index):
            raise Error("ParallelCausalLM: layer already initialized")
        var block = _model_block(args[1], world["plan"], top.numeric_profile, top.device, world["bindings"], context)
        var holder = neural_block_owner(block, context)
        py_setitem(world["blocks"], index, holder)
        var result = bi.dict()
        py_setitem(result, "weight_format", _parallel_causal_fields(block)["weight_format"])
        py_setitem(result, "weight_names", holder._native_entries["weight_names"])
        py_setitem(result, "vendor", holder._native_vendor)
        py_setitem(result, "numeric_mode", holder._native_numeric_mode)
        py_setitem(result, "mamba1_profile", py_get(holder._native_entries, "mamba1_profile", 1))
        return result
    if operation == "tensors":
        var fields = _parallel_causal_fields(top)
        var keys = py_call(bi.list, args)
        for i in range(len(keys)):  # driver-loop: over the embedding and final-head tensor owners
            var name = String(keys[i])
            if name != "embed" and name != "norm" and name != "head":
                raise Error("ParallelCausalLM: unknown top-level tensor")
            py_setitem(fields, "_" + name, args[keys[i]])
        if py_contains(args, "head") and String(top.numeric_profile) == "fixed15_v1":
            py_setattr(top, "_head_int15", _model_quantize15(top._head, world["bindings"]["linalg"]))
        return top._head_int15
    if operation == "embedding":
        return causal_model_embedding(top, args)
    if operation == "final":
        return causal_model_head(top, args)
    if operation == "head" or operation == "head_int15":
        return causal_model_head(top, args, False, operation == "head_int15")
    if operation == "norm":
        var x = args[0]
        var dims = List[Int]()
        for i in range(len(x.shape)):  # driver-loop: over activation shape metadata
            dims.append(Int(py=x.shape[i]))
        var out = generation_array(dims)
        var keep = List[PythonObject]()
        var params = py_ints([Int(py=x.size) // Int(py=top.d_model), Int(py=top.d_model)])
        _ = py_call(params.append, args[1])
        _ = py_call(top._native_entries["rms_norm_forward"], py_ints([generation_address(out, keep), generation_address(x, keep), generation_address(top._norm, keep)]), params)
        _ = keep^
        return out
    if operation == "route":
        if Bool(py=args != top.device):
            raise Error("ParallelCausalLM: worker already owns a different route")
        return PythonObject(True)
    if operation == "allocate":
        var holder = world["blocks"][args[0]]
        var state = neural_block_allocate_binding(holder._blocks[0], args[1], args[2], context)
        var token = Int(py=world["next_state"])
        py_setitem(world, "next_state", token + 1)
        py_setitem(world["states"], token, Python.list(args[0], state))
        return PythonObject(token)
    if operation == "release":
        _ = py_call(world["states"].pop, args, PythonObject(None))
        return PythonObject(True)
    if operation == "state" or operation == "publish_state":
        var token = args if operation == "state" else args[0]
        if not py_contains(world["states"], token):
            raise Error("ParallelCausalLM: state released")
        var state = world["states"][token][1]
        var fields = _parallel_causal_fields(state)
        if operation == "state":
            return py_call(bi.dict, fields)
        var supplied = args[1]
        if Bool(py=py_call(bi.set, supplied) != py_call(bi.set, fields)):
            raise Error("ParallelCausalLM: restored state fields differ")
        var keys = py_call(bi.list, fields)
        for i in range(len(keys)):  # driver-loop: over saved remote state fields before atomic publication
            var old = fields[keys[i]]
            var value = supplied[keys[i]]
            if py_type(old) is context["array_class"]:
                if py_type(value) is not context["array_class"] or Bool(py=value.shape != old.shape) or Bool(py=value.dtype != old.dtype):
                    raise Error("ParallelCausalLM: restored state tensor differs in shape or dtype")
            elif py_type(value) is not py_type(old):
                raise Error("ParallelCausalLM: restored state metadata type differs")
        _ = py_call(fields.update, supplied)
        return PythonObject(True)
    var holder = world["blocks"][args if operation == "parameters" else args[0]]
    var block = holder._blocks[0]
    if operation == "parameters":
        var fields = _parallel_causal_fields(block)
        var result = bi.dict()
        var names = holder._native_entries["weight_names"]
        for i in range(len(names)):  # driver-loop: over the original block parameter registry
            py_setitem(result, names[i], fields["_w"][i])
        return result
    if operation == "run":
        var state = PythonObject(None)
        if args[3] is not None:
            if not py_contains(world["states"], args[3]):
                raise Error("ParallelCausalLM: state released")
            var saved = world["states"][args[3]]
            if Bool(py=saved[0] != args[0]):
                raise Error("ParallelCausalLM: state belongs to another layer")
            state = saved[1]
        return neural_block_forward_binding(block, args[2], state, PythonObject(String(args[1]) == "step"), PythonObject(None), context)
    if operation == "block_fields":
        var fields = _parallel_causal_fields(block)
        var result = bi.dict()
        for key in [String("_w"), String("_wopt"), String("_int15"), String("_opts_tail"), String("n_heads"), String("n_kv_heads"), String("window"), String("dt_limit"), String("weight_format")]:  # driver-loop: over serializable parameter/fingerprint metadata
            if py_contains(fields, key):
                py_setitem(result, key, fields[key])
        return result
    raise Error("ParallelCausalLM: unknown worker operation " + operation)


def parallel_causal_worker(argv: PythonObject) raises:
    model_sequence(argv)
    for i in range(len(argv)):  # small-loop: worker startup accepts native argv words only
        _ = model_string(argv[i])
    var socket = process_worker_connect(argv)
    var world = PARALLEL_CAUSAL.get_or_create_ptr()[].bi.dict()
    try:
        while True:  # driver-loop: over native remote model/layer/state requests
            var request = process_worker_next(socket)
            if request is None:
                break
            var sequence = Int(py=request[0]["sequence"])
            try:
                var result = _parallel_causal_worker_request(world, String(request[1][0]), request[1][1])
                process_send(socket, "result", sequence, result)
            except error:
                process_send(socket, "error", sequence, PythonObject(String(error)))
    except error:
        transport_close(socket)
        raise error
    transport_close(socket)


def _parallel_causal_usable(fields: PythonObject) raises:
    # A closed model or a lost worker state is a RuntimeError on every public
    # call (forward, step, parameters, ...); an invalid argument stays a
    # ValueError. One check so the calls cannot disagree.
    if _parallel_flag(fields["_closed"]):
        raise Error("RuntimeError: ParallelCausalLM is closed")
    if _parallel_flag(fields["_failed"]):
        raise Error("RuntimeError: ParallelCausalLM worker state was lost; reload the model")


def _parallel_pool_publish(fields: PythonObject) raises:
    var pool = fields["_pool"]
    var live = fields["_native_parallel_pool_live"]
    if pool is None or live is None:
        return
    model_dict(live)
    # Do not expose this operation's mutable worker metadata through the live
    # pool: later RPCs can release the GIL. Only native socket capsule identity
    # is shared; the worker dictionaries/lists are detached publication copies.
    py_setitem(live, "workers", _parallel_pool_capture(pool)["workers"])
    for name in ["sequence", "closed"]:  # small-loop: publish resource progress even after a failed request
        py_setitem(live, name, pool[name])


def _parallel_causal_rpc(owner: PythonObject, device: PythonObject, operation: String, args: PythonObject,
                         captured: PythonObject) raises -> PythonObject:
    var fields = _parallel_captured(owner, captured)
    _parallel_causal_usable(fields)
    _ = _parallel_word(device)
    var index = _parallel_word(fields["_worker_index"][device])
    var pool = fields["_pool"]
    if pool is None:
        raise Error("RuntimeError: ParallelCausalLM has no process owner")
    var result = PythonObject(None)
    try:
        result = process_pool_call(pool, index, Python.list(PythonObject(operation), args))
    except error:
        _parallel_pool_publish(fields)
        if len(pool["workers"]) == 0:
            py_setitem(fields, "_failed", True)
            var live = fields["_causal_live_fields"]
            model_dict(live)
            py_setitem(live, "_failed", True)
        raise error
    _parallel_pool_publish(fields)
    return result


def _parallel_causal_rpc(owner: PythonObject, device: PythonObject, operation: String, args: PythonObject) raises -> PythonObject:
    return _parallel_causal_rpc(owner, device, operation, args, _parallel_owner_snapshot(owner))


def _parallel_causal_close(owner: PythonObject) raises:
    # A constructor can fail before the whole model exists; close admits only
    # the lifecycle fields it consumes and still publishes partial reap progress.
    var fields = model_record_fields(owner, PARALLEL_CAUSAL.get_or_create_ptr()[].classes["model"],
        Python.tuple("_closed", "_pool", "_pools"))
    py_setitem(fields, "_closed", True)
    var live = py_get(fields, "_pool")
    if live is not None:
        var pool = _parallel_pool_capture(live)
        var captured = Python.dict()
        py_setitem(captured, "_pool", pool)
        py_setitem(captured, "_native_parallel_pool_live", live)
        try:
            process_pool_close(pool)
        except error:
            _parallel_pool_publish(captured)
            raise error
        _parallel_pool_publish(captured)
    model_dict(fields)
    py_setitem(fields, "_pools", Python.dict())


def _parallel_causal_block(owner: PythonObject, index: Int, weights: PythonObject,
                           captured: PythonObject) raises -> PythonObject:
    var p = PARALLEL_CAUSAL.get_or_create_ptr()
    var fields = _parallel_captured(owner, captured)
    var metadata = _parallel_causal_rpc(owner, fields["layer_devices"][index], "block", Python.list(index, weights), captured)
    model_dict(metadata)
    var format = model_string(metadata["weight_format"])
    if format != model_string(fields["weight_format"]):
        raise Error("ParallelCausalLM: layer reports a different weight_format than requested")
    var block = py_call(p[].bi.object.__new__, p[].classes["block"])
    var block_fields = _parallel_block_fields(block)
    py_setitem(block_fields, "model", owner)
    py_setitem(block_fields, "index", index)
    py_setitem(block_fields, "weight_format", format)
    var vendor = model_string(metadata["vendor"])
    var mode = _parallel_word(metadata["numeric_mode"])
    if index == 0:
        var live = fields["_causal_live_fields"]
        model_dict(live)
        py_setitem(fields, "_native_vendor", vendor)
        py_setitem(fields, "_native_numeric_mode", mode)
        py_setitem(live, "_native_vendor", vendor)
        py_setitem(live, "_native_numeric_mode", mode)
        var entries = fields["_native_entries"]
        var entries_live = fields["_native_parallel_entries_live"]
        model_dict(entries_live)
        for name in ["weight_names", "mamba1_profile"]:  # small-loop: publish worker-reported immutable entry metadata
            var value = model_plain(metadata[name])
            py_setitem(entries, name, value)
            py_setitem(entries_live, name, value)
    elif mode != _parallel_word(fields["_native_numeric_mode"]) or vendor != model_string(fields["_native_vendor"]):
        raise Error("ParallelCausalLM: layer owners disagree on vendor or numeric mode")
    return block


def _parallel_causal_block(owner: PythonObject, index: Int, weights: PythonObject) raises -> PythonObject:
    return _parallel_causal_block(owner, index, weights, _parallel_owner_snapshot(owner))


def _parallel_causal_init(owner: PythonObject, input_plan: PythonObject, input_weights: PythonObject,
                         layer_devices: PythonObject, options: PythonObject, child_module: String) raises:
    var p = PARALLEL_CAUSAL.get_or_create_ptr()
    var bi = p[].bi
    var live = _parallel_model_fields(owner)
    var plan = model_plan_snapshot(input_plan)
    var devices = _parallel_causal_map(layer_devices, _parallel_word(plan["n_layers"]))
    var route = _parallel_causal_route(devices)
    model_sequence(options)
    var ordinary = _model_options_snapshot(Python.list(options[0], options[1], route, options[2]), p[].context)
    var weights = _model_weights_snapshot(input_weights, p[].context)
    py_setitem(live, "_closed", False)
    py_setitem(live, "_failed", False)
    py_setitem(live, "_pool", PythonObject(None))
    py_setitem(live, "_pools", Python.dict())
    py_setitem(live, "layer_devices", devices)
    py_setitem(live, "route", route)
    _ = _causal_model_prepare_snapshot(owner, input_plan, plan, weights, ordinary, p[].context)
    var entries = Python.dict()
    py_setitem(entries, "parallel_call", p[].entry)
    py_setitem(live, "_native_entries", entries)
    var blocks = Python.list()
    py_setitem(live, "_blocks", blocks)
    py_setitem(live, "_head_int15", PythonObject(None))
    py_setitem(live, "_block_class", p[].classes["block"])
    py_setitem(live, "_block_kwargs", py_call(bi.dict, plan["block_options"] if String(plan["kind"]) == "transformer" else plan["block_kwargs"]))
    var prims = py_call(bi.object.__new__, p[].classes["primitives"])
    py_setitem(model_record_fields(prims, p[].classes["primitives"], Python.tuple("model")), "model", owner)
    py_setitem(live, "_prims", prims)
    var unique = Python.list()
    var indices = Python.dict()
    for i in range(len(devices)):  # driver-loop: stable first-appearance physical worker ordering
        if not py_contains(indices, devices[i]):
            py_setitem(indices, devices[i], len(unique))
            _ = py_call(unique.append, devices[i])
    py_setitem(live, "_worker_index", indices)
    var pool = process_pool_create(py_call(bi.tuple, unique), False, child_module)
    py_setitem(live, "_pool", pool)
    var pools = Python.dict()
    for i in range(len(unique)):  # driver-loop: selected process ownership aliases
        py_setitem(pools, unique[i], pool)
    py_setitem(live, "_pools", pools)
    var fields = py_call(bi.dict, live)
    _ = py_call(fields.update, _parallel_capture_model_fields(owner, fields))
    py_setitem(fields, "_causal_original", owner)
    py_setitem(fields, "_causal_live_fields", live)
    py_setitem(fields, "_native_parallel_entries_live", entries)
    var captured = _causal_namespace(fields)
    try:
        process_pool_start(fields["_pool"])
        _parallel_pool_publish(fields)
        var metadata = Python.dict()
        for key in ["device", "kind", "d_model", "vocab_size", "norm_eps", "numeric_profile"]:  # small-loop: captured model primitive metadata
            py_setitem(metadata, key, fields[key])
        var wire_plan = _parallel_causal_wire_plan(plan)
        for i in range(len(unique)):  # driver-loop: worker initialization before constructing layers
            _ = _parallel_causal_rpc(owner, unique[i], "configure", Python.list(metadata, wire_plan), captured)
        for i in range(_parallel_word(plan["n_layers"])):  # driver-loop: each model layer is built only by its assigned worker
            _ = py_call(blocks.append, _parallel_causal_block(owner, i, weights["layers"][i], captured))
        var first = Python.dict()
        py_setitem(first, "embed", fields["_embed"])
        _ = _parallel_causal_rpc(owner, devices[0], "tensors", first, captured)
        var last = Python.dict()
        py_setitem(last, "norm", fields["_norm"])
        py_setitem(last, "head", fields["_head"])
        var packed = _parallel_causal_rpc(owner, devices[len(devices) - 1], "tensors", last, captured)
        model_dict(live)
        py_setitem(live, "_head_int15", packed)
    except error:
        _parallel_pool_publish(fields)
        try:
            _parallel_causal_close(owner)
        except cleanup_error:
            raise Error("RuntimeError: ParallelCausalLM initialization failed: " + String(error) + "; cleanup also failed: " + String(cleanup_error))
        raise error


def _parallel_causal_options(kwargs: PythonObject) raises -> PythonObject:
    var bi = PARALLEL_CAUSAL.get_or_create_ptr()[].bi
    model_dict(kwargs)
    var keys = py_call(bi.list, kwargs)
    for i in range(len(keys)):  # driver-loop: over constructor keyword admission
        var name = String(keys[i])
        if name != "weight_format" and name != "max_positions" and name != "numeric_profile":
            raise Error("TypeError: ParallelCausalLM: unexpected constructor keyword " + name)
    return Python.list(py_get(kwargs, "weight_format", "float32"), py_get(kwargs, "max_positions"), py_get(kwargs, "numeric_profile"))


def _parallel_causal_load(cls: PythonObject, path_value: PythonObject, layer_devices: PythonObject, options: PythonObject, child_module: String) raises -> PythonObject:
    var p = PARALLEL_CAUSAL.get_or_create_ptr()
    var context = p[].context
    var devices = _parallel_causal_map(layer_devices)
    var route = _parallel_causal_route(devices)
    model_sequence(options)
    var checked = _model_options_snapshot(Python.list(options[0], options[1], route, options[2]), context)
    var admitted_options = Python.list(checked[0], checked[1], checked[3])
    var format = model_string(admitted_options[0])
    if format != "float32" and format != "bfloat16" and format != "int8":
        raise Error("ParallelCausalLM: unsupported weight_format")
    var path = native_path(path_value)
    if not native_path_exists(path, True):
        raise Error("FileNotFoundError: ParallelCausalLM.load requires a checkpoint directory")
    var config = config_load_binding(context["config_context"]["config_class"], path)
    var plan = model_plan_for_binding(config, context["config_context"])
    var plan_fields = model_plan_snapshot(plan)
    _ = _parallel_causal_map(devices, _parallel_word(plan_fields["n_layers"]))
    _ = _model_profile(admitted_options[2], model_string(plan_fields["kind"]), context)
    var ck = context["checkpoint_context"]
    var checkpoint = checkpoint_open_binding(ck["checkpoint_class"], path, ck)
    var weights = PythonObject(None)
    try:
        weights = causal_model_read_weights_binding(checkpoint, plan, admitted_options[0], route, context)
    except error:
        _ = checkpoint_close_binding(checkpoint)
        raise error
    _ = checkpoint_close_binding(checkpoint)
    var owner = py_call(p[].bi.object.__new__, cls)
    _parallel_causal_init(owner, plan, weights, devices, admitted_options, child_module)
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
            _parallel_causal_release_token(self.layer, self.token)
        except:
            # Quiet, as the former finalizer: a closed or lost pool has no
            # worker state left to release.
            pass


def _parallel_causal_release_token(block: PythonObject, token: PythonObject) raises:
    var source = _parallel_block_fields(block)
    var captured = _parallel_owner_snapshot(source["model"])
    var fields = causal_snapshot_fields(captured)
    if not _parallel_flag(fields["_closed"]) and not _parallel_flag(fields["_failed"]):
        var index = _parallel_word(source["index"])
        _ = _parallel_causal_rpc(source["model"], fields["layer_devices"][index], "release", token, captured)


def _parallel_causal_attach_fields(fields: PythonObject, layer: PythonObject, token: PythonObject) raises:
    if token is not None:
        _ = _parallel_word(token)
    py_setitem(fields, "layer", layer)
    py_setitem(fields, "token", token)
    py_setitem(fields, "_lifetime", PythonObject(alloc=RemoteStateLifetime(layer, token)))


def _parallel_causal_attach(remote: PythonObject, layer: PythonObject, token: PythonObject) raises:
    var fields = _parallel_state_fields(remote)
    _ = _parallel_block_fields(layer)
    _parallel_causal_attach_fields(fields, layer, token)


def _parallel_causal_allocate(owner: PythonObject, index: Int, batch: PythonObject, capacity: PythonObject,
                              captured: PythonObject) raises -> PythonObject:
    var p = PARALLEL_CAUSAL.get_or_create_ptr()
    var fields = _parallel_captured(owner, captured)
    var blocks = fields["_native_parallel_blocks"]
    if index < 0 or index >= len(blocks):
        raise Error("ParallelCausalLM: layer allocation index outside model")
    _ = _parallel_word(batch)
    if capacity is not None:
        _ = _parallel_word(capacity)
    var result = py_call(p[].bi.object.__new__, p[].classes["state"])
    var destination = _parallel_state_fields(result)
    var original = causal_snapshot_fields(blocks[index])["_causal_original"]
    var token = _parallel_causal_rpc(owner, fields["layer_devices"][index], "allocate", Python.list(index, batch, capacity), captured)
    _parallel_causal_attach_fields(destination, original, token)
    return result


def _parallel_causal_allocate(owner: PythonObject, index: Int, batch: PythonObject, capacity: PythonObject) raises -> PythonObject:
    return _parallel_causal_allocate(owner, index, batch, capacity, _parallel_owner_snapshot(owner))


def _parallel_causal_release(remote: PythonObject, captured: PythonObject) raises:
    var fields = _parallel_captured(remote, captured)
    var token = fields["token"]
    if token is None:
        return
    var model = fields["_native_parallel_model"]
    var owner = causal_snapshot_fields(model)
    var block = causal_snapshot_fields(fields["_native_parallel_block"])
    if not _parallel_flag(owner["_closed"]) and not _parallel_flag(owner["_failed"]):
        _ = _parallel_causal_rpc(owner["_causal_original"], owner["layer_devices"][block["index"]], "release", token, model)
    # Preserve the old failure contract: a failed release keeps the token and
    # lifetime live, allowing an explicit retry or later native destruction.
    py_setitem(fields, "token", PythonObject(None))
    var live = fields["_causal_live_fields"]
    model_dict(live)
    py_setitem(live, "token", PythonObject(None))
    var lifetime = fields["_lifetime"]
    if lifetime is not None:
        lifetime.downcast_value_ptr[RemoteStateLifetime]()[].released = True


def _parallel_causal_release(remote: PythonObject) raises:
    _parallel_causal_release(remote, _parallel_capture_state(remote))


def _parallel_causal_snapshot(remote: PythonObject, captured: PythonObject) raises -> PythonObject:
    var source = _parallel_captured(remote, captured)
    if source["token"] is None:
        raise Error("ParallelCausalLM: state released")
    var p = PARALLEL_CAUSAL.get_or_create_ptr()
    var model = source["_native_parallel_model"]
    var owner = causal_snapshot_fields(model)
    var block = causal_snapshot_fields(source["_native_parallel_block"])
    var fields = _parallel_causal_rpc(owner["_causal_original"], owner["layer_devices"][block["index"]], "state", source["token"], model)
    var state = py_call(p[].bi.object.__new__, p[].context["generation_classes"][2][owner["kind"]])
    _ = py_call(_parallel_causal_fields(state).update, fields)
    return state


def _parallel_causal_snapshot(remote: PythonObject) raises -> PythonObject:
    return _parallel_causal_snapshot(remote, _parallel_capture_state(remote))


def _parallel_causal_block_run(block: PythonObject, method: PythonObject, x: PythonObject, state: PythonObject,
                               captured_model: PythonObject, captured_block: PythonObject,
                               captured_state: PythonObject) raises -> PythonObject:
    _ = model_string(method)
    var owner = causal_snapshot_fields(captured_model)
    var layer = _parallel_captured(block, captured_block)
    var token = PythonObject(None)
    if state is not None:
        var fields = _parallel_captured(state, captured_state)
        if fields["layer"] is not block or fields["token"] is None:
            raise Error("ParallelCausalLM: state belongs to another layer or has been released")
        token = fields["token"]
    var index = layer["index"]
    return _parallel_causal_rpc(owner["_causal_original"], owner["layer_devices"][index], "run", Python.list(index, method, x, token), captured_model)


def _parallel_find_block(block: PythonObject, model: PythonObject) raises -> PythonObject:
    var owner = causal_snapshot_fields(model)
    var blocks = owner["_native_parallel_blocks"]
    for i in range(len(blocks)):  # driver-loop: private RemoteBlock entry preserves original owner identity
        if causal_snapshot_fields(blocks[i])["_causal_original"] is block:
            return blocks[i]
    # Private RemoteBlock methods historically accept a separately retained
    # handle for an existing worker index. Preserve that handle's own identity;
    # it need not be the handle currently in model._blocks.
    var extra = py_get(owner, "_native_parallel_extra_blocks")
    if extra is None:
        extra = Python.list()
        py_setitem(owner, "_native_parallel_extra_blocks", extra)
    for i in range(len(extra)):  # driver-loop: deduplicate additional retained remote handles
        if causal_snapshot_fields(extra[i])["_causal_original"] is block:
            return extra[i]
    var live = _parallel_block_fields(block)
    if live["model"] is not owner["_causal_original"]:
        raise Error("ParallelCausalLM: remote block belongs to another model")
    var index = _parallel_word(live["index"])
    if index < 0 or index >= len(owner["layer_devices"]):
        raise Error("ParallelCausalLM: remote block index outside its layer map")
    var fields = Python.dict()
    py_setitem(fields, "model", live["model"])
    py_setitem(fields, "index", index)
    py_setitem(fields, "weight_format", model_string(live["weight_format"]))
    py_setitem(fields, "_causal_original", block)
    var captured = _causal_namespace(fields)
    _ = py_call(extra.append, captured)
    return captured


def _parallel_causal_block_run(block: PythonObject, method: PythonObject, x: PythonObject, state: PythonObject) raises -> PythonObject:
    var layer = _parallel_block_fields(block)
    var model = _parallel_owner_snapshot(layer["model"])
    var captured = PythonObject(None)
    if state is not None:
        captured = _parallel_capture_state(state, model)
    return _parallel_causal_block_run(block, method, x, state, model, _parallel_find_block(block, model), captured)


def parallel_causal_call(operation: String, args: PythonObject) raises -> PythonObject:
    var p = PARALLEL_CAUSAL.get_or_create_ptr()
    var child = String("mojolearn._native_parallel_causal_worker")
    model_sequence(args)
    if operation == "initialize":
        parallel_causal_initialize(args)
    elif operation == "worker":
        parallel_causal_worker(args)
    elif operation == "route":
        return _parallel_causal_route(args[0])
    elif operation == "init":
        _parallel_causal_init(args[0], args[1], args[2], args[3], _parallel_causal_options(args[4]), child)
    elif operation == "load" or operation == "load_with_worker":
        return _parallel_causal_load(args[0], args[1], args[2], args[3], child if operation == "load" else model_string(args[4]))
    elif operation == "rpc":
        return _parallel_causal_rpc(args[0], args[1], model_string(args[2]), args[3])
    elif operation == "capture_model":
        return _parallel_capture_model(args[0], args[1]) if len(args) > 1 else _parallel_capture_model(args[0])
    elif operation == "capture_state":
        return _parallel_capture_state(args[0], args[1]) if len(args) > 1 else _parallel_capture_state(args[0])
    elif operation == "parameters":
        var model = causal_model_capture(args[0], p[].context)
        _parallel_causal_usable(causal_snapshot_fields(model))
        var result = PythonObject(None)
        try:
            result = causal_lm_parameters_binding(model)
        except error:
            causal_model_publish(model)
            raise error
        causal_model_publish(model)
        return result
    elif operation == "block_fields":
        var captured = args[2] if len(args) > 2 else _parallel_owner_snapshot(args[0])
        var fields = _parallel_captured(args[0], captured)
        var index = _parallel_word(args[1])
        return _parallel_causal_rpc(args[0], fields["layer_devices"][index], "block_fields", Python.list(index), captured)
    elif operation == "allocate_layer":
        var captured = args[4] if len(args) > 4 else _parallel_owner_snapshot(args[0])
        return _parallel_causal_allocate(args[0], _parallel_word(args[1]), args[2], args[3], captured)
    elif operation == "block_allocate":
        var block = _parallel_block_fields(args[0])
        return _parallel_causal_allocate(block["model"], _parallel_word(block["index"]), args[1], args[2])
    elif operation == "block_run":
        return _parallel_causal_block_run(args[0], args[1], args[2], args[3])
    elif operation == "block_call":
        var block = _parallel_block_fields(args[0])
        var captured = _parallel_owner_snapshot(block["model"])
        var fields = causal_snapshot_fields(captured)
        return _parallel_causal_rpc(block["model"], fields["layer_devices"][_parallel_word(block["index"])], model_string(args[1]), args[2], captured)
    elif operation == "block_init":
        var destination = _parallel_block_fields(args[0])
        var created = _parallel_causal_block(args[1], _parallel_word(args[2]), args[3])
        _ = py_call(destination.update, _parallel_block_fields(created))
    elif operation == "make_blocks":
        model_sequence(args[1])
        var result = p[].bi.list()
        for i in range(len(args[1])):  # driver-loop: over legacy private layer-construction inputs
            _ = py_call(result.append, _parallel_causal_block(args[0], i, args[1][i]))
        return result
    elif operation == "state_init":
        _parallel_causal_attach(args[0], args[1], args[2])
    elif operation == "state_snapshot":
        return _parallel_causal_snapshot(args[0], args[1]) if len(args) > 1 else _parallel_causal_snapshot(args[0])
    elif operation == "state_publish":
        var remote = args[0]
        var captured = args[2] if len(args) > 2 else _parallel_capture_state(remote)
        var fields = _parallel_captured(remote, captured)
        if fields["token"] is None:
            raise Error("ParallelCausalLM: state released")
        var model = fields["_native_parallel_model"]
        var owner = causal_snapshot_fields(model)
        var block = causal_snapshot_fields(fields["_native_parallel_block"])
        return _parallel_causal_rpc(owner["_causal_original"], owner["layer_devices"][block["index"]], "publish_state", Python.list(fields["token"], _parallel_causal_fields(args[1])), model)
    elif operation == "state_close" or operation == "state_close_quiet":
        _parallel_causal_release(args[0])
    elif operation == "release_layers":
        var owner = _parallel_captured(args[0], args[2])
        var state = _parallel_captured(args[1], args[3])
        var layers = state["layers"]
        var failure = String()
        for i in range(len(layers)):  # driver-loop: over remote states owned by the caller's model state
            try:
                var fields = causal_snapshot_fields(layers[i])
                _parallel_causal_release(fields["_causal_original"], layers[i])
            except error:
                if failure == "":
                    failure = String(error)
        if failure != "":
            raise Error(failure)
    elif operation == "run":
        var owner = args[0]
        var ids = generation_array_snapshot(args[1])
        var length = Int(py=ids.shape[1])
        var state = args[2]
        var model_fields = _parallel_captured(owner, args[4])
        var state_fields = _parallel_captured(state, args[5])
        var devices = model_fields["layer_devices"]
        var blocks = model_fields["_native_parallel_blocks"]
        var layers = state_fields["layers"]
        var x = _parallel_causal_rpc(owner, devices[0], "embedding", ids, args[4])
        # The legacy parallel API used each block's dedicated decode entry
        # after prefill. Preserve that route independently of ordinary CausalLM.
        var method = PythonObject("step" if _parallel_flag(args[3]) else "forward")
        for i in range(len(blocks)):  # driver-loop: ordered persistent layer owners retain the dedicated remote decode route
            var block = causal_snapshot_fields(blocks[i])
            var layer = causal_snapshot_fields(layers[i])
            x = _parallel_causal_block_run(block["_causal_original"], method, x, layer["_causal_original"], args[4], blocks[i], layers[i])
        var result = _parallel_causal_rpc(owner, devices[len(devices) - 1], "final", x, args[4])
        py_setitem(state_fields, "positions", _parallel_word(state_fields["positions"]) + length)
        return result
    elif operation == "close" or operation == "close_quiet" or operation == "exit":
        _parallel_causal_close(args[0])
    elif operation == "enter":
        if _parallel_flag(_parallel_model_fields(args[0])["_closed"]):
            raise Error("RuntimeError: ParallelCausalLM is closed")
        return args[0]
    elif operation == "primitives_init":
        var fields = model_record_fields(args[0], p[].classes["primitives"], Python.tuple("model"))
        _ = _parallel_model_fields(args[1])
        py_setitem(fields, "model", args[1])
    elif operation == "primitive":
        var fields = model_record_fields(args[0], p[].classes["primitives"], Python.tuple("model"))
        var owner = fields["model"]
        var captured = _parallel_owner_snapshot(owner)
        var devices = causal_snapshot_fields(captured)["layer_devices"]
        var name = model_string(args[1])
        var device = devices[0 if name == "embedding" else len(devices) - 1]
        return _parallel_causal_rpc(owner, device, name, args[2], captured)
    elif operation == "resident":
        # Existing remote layer handles cannot enter the single-process
        # resident token loop. The shared native generation loop owns tokens.
        return PythonObject(None)
    else:
        raise Error("ParallelCausalLM: unknown native operation " + operation)
    return PythonObject(None)


def parallel_causal_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    var args = PythonObject(from_borrowed=args_ptr)
    var operation = String()
    try:
        model_sequence(args)
        operation = model_string(args[0])
        var result = parallel_causal_call(operation, args[1])
        return result.steal_data()
    except error:
        if operation == "close_quiet" or operation == "state_close_quiet":
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
