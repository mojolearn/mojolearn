# SPDX-License-Identifier: Apache-2.0
"""Native ownership for the existing sequential layer-partitioned CausalLM."""
from std.ffi import _Global
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.generation_buffers import GENERATION_BOUNDARY, generation_initialize_binding, generation_copy, generation_array, generation_address, py_ints
from bindings.causal_lm_construct import causal_model_prepare, causal_model_read_weights_binding, _model_block, _model_bindings, _model_profile, _model_quantize15
from bindings.causal_lm_public import causal_model_embedding, causal_model_head, causal_lm_parameters_binding, _native_function
from bindings.neural_block_public import neural_block_owner, neural_block_allocate_binding, neural_block_forward_binding
from bindings.causal_model_config import config_load_binding, model_plan_for_binding
from bindings.causal_checkpoint import checkpoint_open_binding, checkpoint_close_binding
from bindings.causal_exception import causal_borrowed_item, causal_set_exception
from bindings.native_json import native_path, native_path_exists
from bindings.native_process_pool import process_pool_create, process_pool_start, process_pool_call, process_pool_close, process_worker_connect, process_worker_next, process_send
from bindings.native_transport import transport_close


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
    p[].bi = Python.import_module("builtins")
    p[].context = args[0]
    p[].classes = args[1]
    p[].entry = args[2]
    if p[].bi.type(p[].entry) is not p[].bi.type(p[].bi.len):
        raise Error("ParallelCausalLM: execution router must be a compiled native entry")
    _ = generation_initialize_binding(p[].context["generation_classes"])


def _parallel_causal_fields(value: PythonObject) raises -> PythonObject:
    return PARALLEL_CAUSAL.get_or_create_ptr()[].bi.object.__getattribute__(value, "__dict__")


def _parallel_causal_route(devices: PythonObject) raises -> PythonObject:
    var p = PARALLEL_CAUSAL.get_or_create_ptr()
    var vendor = String(p[].context["selected_vendor"])
    if vendor == "metal":
        if len(devices) == 0:
            raise Error("Metal ParallelCausalLM requires every layer owner to be device 0")
        for i in range(len(devices)):  # driver-loop: over explicit layer ownership metadata
            if p[].bi.type(devices[i]) is not p[].bi.int or Int(py=devices[i]) != 0:
                raise Error("Metal ParallelCausalLM requires every layer owner to be device 0")
        return PythonObject("gpu")
    if vendor == "cuda" or vendor == "hip":
        return PythonObject("gpu")
    if vendor == "cpu":
        return PythonObject("cpu")
    raise Error("NotImplementedError: ParallelCausalLM requires CUDA or HIP device isolation, single-device Metal, or a CPU-only install")


def _parallel_causal_map(value: PythonObject, expected: Int = -1) raises -> PythonObject:
    var bi = PARALLEL_CAUSAL.get_or_create_ptr()[].bi
    var devices = bi.tuple(value)
    if len(devices) == 0 or (expected >= 0 and len(devices) != expected):
        raise Error("layer_devices requires one nonnegative device index per layer")
    for i in range(len(devices)):  # driver-loop: over layer-to-worker indices
        if bi.type(devices[i]) is not bi.int or Int(py=devices[i]) < 0:
            raise Error("layer_devices requires one nonnegative device index per layer")
    _ = _parallel_causal_route(devices)
    return devices


def _parallel_causal_wire_plan(plan: PythonObject) raises -> PythonObject:
    var bi = PARALLEL_CAUSAL.get_or_create_ptr()[].bi
    var result = bi.dict()
    for key in [String("kind"), String("d_model"), String("n_heads"), String("block_options"), String("block_kwargs")]:  # driver-loop: over block-construction metadata
        result[key] = bi.getattr(plan, key, PythonObject(None))
    return result


def _parallel_causal_worker_request(world: PythonObject, operation: String, args: PythonObject) raises -> PythonObject:
    var p = PARALLEL_CAUSAL.get_or_create_ptr()
    var bi = p[].bi
    var context = p[].context
    if operation == "configure":
        if world.get("top") is not None:
            raise Error("ParallelCausalLM: worker already configured")
        var route = PythonObject("cpu" if String(context["selected_vendor"]) == "cpu" else "gpu")
        if Bool(py=args[0]["device"] != route):
            raise Error("ParallelCausalLM: worker route differs from its connected backend")
        var plan = bi.object.__new__(context["config_context"]["plan_class"])
        _parallel_causal_fields(plan).update(args[1])
        var top = bi.object.__new__(context["model_class"])
        _parallel_causal_fields(top).update(args[0])
        top._embed = PythonObject(None)
        top._norm = PythonObject(None)
        top._head = PythonObject(None)
        top._head_int15 = PythonObject(None)
        var bindings = _model_bindings(route, String(plan.kind), context)
        var entries = bi.dict()
        for key in [String("embedding_forward"), String("rms_norm_forward"), String("linear_forward")]:  # driver-loop: over the three existing model primitive entries
            entries[key] = _native_function(bindings["primitives"], key)
        if String(top.numeric_profile) == "fixed15_v1":
            entries["quantize15"] = _native_function(bindings["linalg"], "quantize_int15")
            entries["gemm15"] = _native_function(bindings["linalg"], "gemm_int15")
        top._native_entries = entries
        world["plan"] = plan
        world["top"] = top
        world["bindings"] = bindings
        world["blocks"] = bi.dict()
        world["states"] = bi.dict()
        world["next_state"] = 0
        return PythonObject(True)
    var top = world.get("top")
    if top is None:
        raise Error("ParallelCausalLM: worker was not told its route")
    if operation == "block":
        var index = args[0]
        if Bool(py=world["blocks"].__contains__(index)):
            raise Error("ParallelCausalLM: layer already initialized")
        var block = _model_block(args[1], world["plan"], top.numeric_profile, top.device, world["bindings"], context)
        var holder = neural_block_owner(block, context)
        world["blocks"][index] = holder
        var result = bi.dict()
        result["weight_format"] = _parallel_causal_fields(block)["weight_format"]
        result["weight_names"] = holder._native_entries["weight_names"]
        result["vendor"] = holder._native_vendor
        result["numeric_mode"] = holder._native_numeric_mode
        result["mamba1_profile"] = holder._native_entries.get("mamba1_profile", 1)
        return result
    if operation == "tensors":
        var fields = _parallel_causal_fields(top)
        var keys = bi.list(args)
        for i in range(len(keys)):  # driver-loop: over the embedding and final-head tensor owners
            var name = String(keys[i])
            if name != "embed" and name != "norm" and name != "head":
                raise Error("ParallelCausalLM: unknown top-level tensor")
            fields["_" + name] = args[keys[i]]
        if Bool(py=args.__contains__("head")) and String(top.numeric_profile) == "fixed15_v1":
            top._head_int15 = _model_quantize15(top._head, world["bindings"]["linalg"])
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
        params.append(args[1])
        top._native_entries["rms_norm_forward"](py_ints([generation_address(out, keep), generation_address(x, keep), generation_address(top._norm, keep)]), params)
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
        world["next_state"] = token + 1
        world["states"][token] = Python.list(args[0], state)
        return PythonObject(token)
    if operation == "release":
        _ = world["states"].pop(args, PythonObject(None))
        return PythonObject(True)
    if operation == "state" or operation == "publish_state":
        var token = args if operation == "state" else args[0]
        if not Bool(py=world["states"].__contains__(token)):
            raise Error("ParallelCausalLM: state released")
        var state = world["states"][token][1]
        var fields = _parallel_causal_fields(state)
        if operation == "state":
            return bi.dict(fields)
        var supplied = args[1]
        if Bool(py=bi.set(supplied) != bi.set(fields)):
            raise Error("ParallelCausalLM: restored state fields differ")
        var keys = bi.list(fields)
        for i in range(len(keys)):  # driver-loop: over saved remote state fields before atomic publication
            var old = fields[keys[i]]
            var value = supplied[keys[i]]
            if bi.type(old) is context["array_class"]:
                if bi.type(value) is not context["array_class"] or Bool(py=value.shape != old.shape) or Bool(py=value.dtype != old.dtype):
                    raise Error("ParallelCausalLM: restored state tensor differs in shape or dtype")
            elif bi.type(value) is not bi.type(old):
                raise Error("ParallelCausalLM: restored state metadata type differs")
        fields.update(supplied)
        return PythonObject(True)
    var holder = world["blocks"][args if operation == "parameters" else args[0]]
    var block = holder._blocks[0]
    if operation == "parameters":
        var fields = _parallel_causal_fields(block)
        var result = bi.dict()
        var names = holder._native_entries["weight_names"]
        for i in range(len(names)):  # driver-loop: over the original block parameter registry
            result[names[i]] = fields["_w"][i]
        return result
    if operation == "run":
        var state = PythonObject(None)
        if args[3] is not None:
            if not Bool(py=world["states"].__contains__(args[3])):
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
            if Bool(py=fields.__contains__(key)):
                result[key] = fields[key]
        return result
    raise Error("ParallelCausalLM: unknown worker operation " + operation)


def parallel_causal_worker(argv: PythonObject) raises:
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


def _parallel_causal_rpc(owner: PythonObject, device: PythonObject, operation: String, args: PythonObject) raises -> PythonObject:
    if Bool(py=owner._closed):
        raise Error("RuntimeError: ParallelCausalLM is closed")
    if Bool(py=owner._failed):
        raise Error("RuntimeError: ParallelCausalLM worker state was lost; reload the model")
    var index = Int(py=owner._worker_index[device])
    try:
        return process_pool_call(owner._pool, index, Python.list(PythonObject(operation), args))
    except error:
        if len(owner._pool["workers"]) == 0:
            owner._failed = True
        raise error


def _parallel_causal_close(owner: PythonObject) raises:
    var fields = _parallel_causal_fields(owner)
    fields["_closed"] = True
    var pool = fields.get("_pool")
    if pool is not None:
        process_pool_close(pool)
    fields["_pools"] = PARALLEL_CAUSAL.get_or_create_ptr()[].bi.dict()


def _parallel_causal_block(owner: PythonObject, index: Int, weights: PythonObject) raises -> PythonObject:
    var p = PARALLEL_CAUSAL.get_or_create_ptr()
    var metadata = _parallel_causal_rpc(owner, owner.layer_devices[index], "block", Python.list(index, weights))
    if Bool(py=metadata["weight_format"] != owner.weight_format):
        raise Error("ParallelCausalLM: layer reports a different weight_format than requested")
    var block = p[].bi.object.__new__(p[].classes["block"])
    block.model = owner
    block.index = index
    block.weight_format = metadata["weight_format"]
    if index == 0:
        owner._native_vendor = metadata["vendor"]
        owner._native_numeric_mode = metadata["numeric_mode"]
        owner._native_entries["weight_names"] = metadata["weight_names"]
        owner._native_entries["mamba1_profile"] = metadata["mamba1_profile"]
    elif Bool(py=metadata["numeric_mode"] != owner._native_numeric_mode) or Bool(py=metadata["vendor"] != owner._native_vendor):
        raise Error("ParallelCausalLM: layer owners disagree on vendor or numeric mode")
    return block


def _parallel_causal_init(owner: PythonObject, plan: PythonObject, weights: PythonObject,
                         layer_devices: PythonObject, options: PythonObject, child_module: String) raises:
    var p = PARALLEL_CAUSAL.get_or_create_ptr()
    var bi = p[].bi
    var devices = _parallel_causal_map(layer_devices, Int(py=plan.n_layers))
    var route = _parallel_causal_route(devices)
    owner._closed = False
    owner._failed = False
    owner._pool = PythonObject(None)
    owner._pools = bi.dict()
    owner.layer_devices = devices
    owner.route = route
    var ordinary = Python.list(options[0], options[1], route, options[2])
    _ = causal_model_prepare(owner, plan, weights, ordinary, p[].context)
    owner._native_entries = bi.dict()
    owner._native_entries["parallel_call"] = p[].entry
    owner._blocks = bi.list()
    owner._head_int15 = PythonObject(None)
    owner._block_class = p[].classes["block"]
    owner._block_kwargs = bi.dict(plan.block_options if String(plan.kind) == "transformer" else plan.block_kwargs)
    owner._prims = bi.object.__new__(p[].classes["primitives"])
    owner._prims.model = owner
    var unique = bi.list()
    owner._worker_index = bi.dict()
    for i in range(len(devices)):  # driver-loop: over stable first-appearance physical worker ordering
        if not Bool(py=owner._worker_index.__contains__(devices[i])):
            owner._worker_index[devices[i]] = len(unique)
            unique.append(devices[i])
    owner._pool = process_pool_create(bi.tuple(unique), False, child_module)
    for i in range(len(unique)):  # driver-loop: over selected process ownership records
        owner._pools[unique[i]] = owner._pool
    try:
        process_pool_start(owner._pool)
        var metadata = bi.dict()
        for key in [String("device"), String("kind"), String("d_model"), String("vocab_size"), String("norm_eps"), String("numeric_profile")]:  # driver-loop: over model primitive metadata
            metadata[key] = bi.getattr(owner, key)
        var wire_plan = _parallel_causal_wire_plan(plan)
        for i in range(len(unique)):  # driver-loop: over worker initialization before any layer is built
            _ = _parallel_causal_rpc(owner, unique[i], "configure", Python.list(metadata, wire_plan))
        for i in range(Int(py=plan.n_layers)):  # driver-loop: over model layers, each built only by its assigned worker
            owner._blocks.append(_parallel_causal_block(owner, i, weights["layers"][i]))
        var first = bi.dict()
        first["embed"] = owner._embed
        _ = _parallel_causal_rpc(owner, devices[0], "tensors", first)
        var last = bi.dict()
        last["norm"] = owner._norm
        last["head"] = owner._head
        owner._head_int15 = _parallel_causal_rpc(owner, devices[len(devices) - 1], "tensors", last)
    except error:
        try:
            _parallel_causal_close(owner)
        except cleanup_error:
            raise Error("RuntimeError: ParallelCausalLM initialization failed: " + String(error) + "; cleanup also failed: " + String(cleanup_error))
        raise error


def _parallel_causal_options(kwargs: PythonObject) raises -> PythonObject:
    var bi = PARALLEL_CAUSAL.get_or_create_ptr()[].bi
    var keys = bi.list(kwargs)
    for i in range(len(keys)):  # driver-loop: over constructor keyword admission
        var name = String(keys[i])
        if name != "weight_format" and name != "max_positions" and name != "numeric_profile":
            raise Error("TypeError: ParallelCausalLM: unexpected constructor keyword " + name)
    return Python.list(kwargs.get("weight_format", "float32"), kwargs.get("max_positions"), kwargs.get("numeric_profile"))


def _parallel_causal_load(cls: PythonObject, path_value: PythonObject, layer_devices: PythonObject, options: PythonObject, child_module: String) raises -> PythonObject:
    var p = PARALLEL_CAUSAL.get_or_create_ptr()
    var context = p[].context
    var devices = _parallel_causal_map(layer_devices)
    var route = _parallel_causal_route(devices)
    var format = String(options[0])
    if format != "float32" and format != "bfloat16" and format != "int8":
        raise Error("ParallelCausalLM: unsupported weight_format")
    var path = native_path(path_value)
    if not native_path_exists(path, True):
        raise Error("FileNotFoundError: ParallelCausalLM.load requires a checkpoint directory")
    var config = config_load_binding(context["config_context"]["config_class"], path)
    var plan = model_plan_for_binding(config, context["config_context"])
    _ = _parallel_causal_map(devices, Int(py=plan.n_layers))
    _ = _model_profile(options[2], String(plan.kind), context)
    var ck = context["checkpoint_context"]
    var checkpoint = checkpoint_open_binding(ck["checkpoint_class"], path, ck)
    var weights = PythonObject(None)
    try:
        weights = causal_model_read_weights_binding(checkpoint, plan, options[0], route, context)
    except error:
        _ = checkpoint_close_binding(checkpoint)
        raise error
    _ = checkpoint_close_binding(checkpoint)
    var owner = p[].bi.object.__new__(cls)
    _parallel_causal_init(owner, plan, weights, devices, options, child_module)
    return owner


def _parallel_causal_allocate(owner: PythonObject, index: Int, batch: PythonObject, capacity: PythonObject) raises -> PythonObject:
    var p = PARALLEL_CAUSAL.get_or_create_ptr()
    var token = _parallel_causal_rpc(owner, owner.layer_devices[index], "allocate", Python.list(index, batch, capacity))
    var result = p[].bi.object.__new__(p[].classes["state"])
    result.layer = owner._blocks[index]
    result.token = token
    return result


def _parallel_causal_release(remote: PythonObject) raises:
    var fields = _parallel_causal_fields(remote)
    var token = fields.get("token")
    if token is None:
        return
    var block = fields["layer"]
    var owner = block.model
    if not Bool(py=owner._closed) and not Bool(py=owner._failed):
        _ = _parallel_causal_rpc(owner, owner.layer_devices[Int(py=block.index)], "release", token)
    fields["token"] = PythonObject(None)


def _parallel_causal_snapshot(remote: PythonObject) raises -> PythonObject:
    if remote.token is None:
        raise Error("ParallelCausalLM: state released")
    var p = PARALLEL_CAUSAL.get_or_create_ptr()
    var block = remote.layer
    var owner = block.model
    var fields = _parallel_causal_rpc(owner, owner.layer_devices[Int(py=block.index)], "state", remote.token)
    var state = p[].bi.object.__new__(p[].context["generation_classes"][2][owner.kind])
    _parallel_causal_fields(state).update(fields)
    return state


def _parallel_causal_block_run(block: PythonObject, method: PythonObject, x: PythonObject, state: PythonObject) raises -> PythonObject:
    if state is not None and (state.layer is not block or state.token is None):
        raise Error("ParallelCausalLM: state belongs to another layer or has been released")
    var owner = block.model
    var index = Int(py=block.index)
    return _parallel_causal_rpc(owner, owner.layer_devices[index], "run", Python.list(index, method, x, PythonObject(None) if state is None else state.token))


def parallel_causal_call(operation: String, args: PythonObject) raises -> PythonObject:
    var p = PARALLEL_CAUSAL.get_or_create_ptr()
    var child = String("mojolearn._native_parallel_causal_worker")
    if operation == "initialize":
        parallel_causal_initialize(args)
    elif operation == "worker":
        parallel_causal_worker(args)
    elif operation == "route":
        return _parallel_causal_route(args[0])
    elif operation == "init":
        _parallel_causal_init(args[0], args[1], args[2], args[3], _parallel_causal_options(args[4]), child)
    elif operation == "load" or operation == "load_with_worker":
        return _parallel_causal_load(args[0], args[1], args[2], args[3], child if operation == "load" else String(args[4]))
    elif operation == "rpc":
        return _parallel_causal_rpc(args[0], args[1], String(args[2]), args[3])
    elif operation == "parameters":
        return causal_lm_parameters_binding(args[0])
    elif operation == "block_fields":
        return _parallel_causal_rpc(args[0], args[0].layer_devices[Int(py=args[1])], "block_fields", Python.list(args[1]))
    elif operation == "allocate_layer":
        return _parallel_causal_allocate(args[0], Int(py=args[1]), args[2], args[3])
    elif operation == "block_allocate":
        return _parallel_causal_allocate(args[0].model, Int(py=args[0].index), args[1], args[2])
    elif operation == "block_run":
        return _parallel_causal_block_run(args[0], args[1], args[2], args[3])
    elif operation == "block_call":
        var block = args[0]
        return _parallel_causal_rpc(block.model, block.model.layer_devices[Int(py=block.index)], String(args[1]), args[2])
    elif operation == "block_init":
        var created = _parallel_causal_block(args[1], Int(py=args[2]), args[3])
        _parallel_causal_fields(args[0]).update(_parallel_causal_fields(created))
    elif operation == "make_blocks":
        var result = p[].bi.list()
        for i in range(len(args[1])):  # driver-loop: over legacy private layer-construction inputs
            result.append(_parallel_causal_block(args[0], i, args[1][i]))
        return result
    elif operation == "state_init":
        args[0].layer = args[1]
        args[0].token = args[2]
    elif operation == "state_snapshot":
        return _parallel_causal_snapshot(args[0])
    elif operation == "state_publish":
        var remote = args[0]
        if remote.token is None:
            raise Error("ParallelCausalLM: state released")
        var block = remote.layer
        return _parallel_causal_rpc(block.model, block.model.layer_devices[Int(py=block.index)], "publish_state", Python.list(remote.token, _parallel_causal_fields(args[1])))
    elif operation == "state_close" or operation == "state_close_quiet":
        _parallel_causal_release(args[0])
    elif operation == "release_layers":
        var layers = args[1].layers
        var failure = String()
        for i in range(len(layers)):  # driver-loop: over remote states owned by the caller's model state
            try:
                _parallel_causal_release(layers[i])
            except error:
                if failure == "":
                    failure = String(error)
        if failure != "":
            raise Error(failure)
    elif operation == "run":
        var owner = args[0]
        var ids = args[1]
        var state = args[2]
        var x = _parallel_causal_rpc(owner, owner.layer_devices[0], "embedding", ids)
        # The legacy parallel API used each block's dedicated decode entry
        # after prefill. Preserve that route independently of ordinary CausalLM.
        var method = PythonObject("step" if Bool(py=args[3]) else "forward")
        for i in range(len(owner._blocks)):  # driver-loop: over ordered persistent layer owners, with native activation transport
            x = _parallel_causal_block_run(owner._blocks[i], method, x, state.layers[i])
        var result = _parallel_causal_rpc(owner, owner.layer_devices[len(owner.layer_devices) - 1], "final", x)
        state.positions = Int(py=state.positions) + Int(py=ids.shape[1])
        return result
    elif operation == "close" or operation == "close_quiet" or operation == "exit":
        _parallel_causal_close(args[0])
    elif operation == "enter":
        if Bool(py=args[0]._closed):
            raise Error("RuntimeError: ParallelCausalLM is closed")
        return args[0]
    elif operation == "primitives_init":
        args[0].model = args[1]
    elif operation == "primitive":
        var owner = args[0].model
        var name = String(args[1])
        var device = owner.layer_devices[0 if name == "embedding" else len(owner.layer_devices) - 1]
        return _parallel_causal_rpc(owner, device, name, args[2])
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
        operation = String(args[0])
        var result = parallel_causal_call(operation, args[1])
        return result.steal_data()
    except error:
        if operation == "close_quiet" or operation == "state_close_quiet":
            var nil = PythonObject(None)
            return nil.steal_data()
        var message = String(error)
        if message.startswith("RuntimeError:"):
            return raise_python_exception(error, ExceptionType("PyExc_RuntimeError"))
        if message.startswith("TypeError:"):
            return raise_python_exception(error, ExceptionType.TypeError)
        if message.startswith("FileNotFoundError:"):
            return raise_python_exception(error, ExceptionType("PyExc_FileNotFoundError"))
        if message.startswith("NotImplementedError:"):
            return raise_python_exception(error, ExceptionType("PyExc_NotImplementedError"))
        if message.startswith("UnsupportedModel:"):
            try:
                ref cpy = Python().cpython()
                var context = cpy.PyTuple_GetItem(args_ptr, 2)
                var config = causal_borrowed_item(context, "config_context")
                return causal_set_exception(error, causal_borrowed_item(config, "unsupported_class"))
            except nested:
                return raise_python_exception(nested)
        return raise_python_exception(error, ExceptionType.ValueError)
