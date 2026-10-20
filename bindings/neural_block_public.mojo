# SPDX-License-Identifier: Apache-2.0
"""Native standalone block construction, state allocation and sequence execution."""
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from std.memory import memcpy
from std.os import getenv
from bindings.generation_buffers import generation_initialize_binding, generation_array, py_ints
from bindings.causal_lm_construct import _model_block, _model_profile, _model_bindings, _model_f32
from bindings.causal_lm_public import causal_lm_admit_binding, causal_layer_allocate, causal_layer_run, _addr, _native_function
from bindings.hostptr import f32_ptr


def neural_block_owner(block: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var fields = bi.object.__getattribute__(block, "__dict__")
    var kind = String(fields["_native_kind"])
    var route = fields["_native_route"]
    var mode = fields.get("numeric_mode")
    var bindings = _model_bindings(route, kind, context, mode)
    if kind == "transformer" and Bool(py=bi.hasattr(bindings[kind], "transformer_arithmetic_profile")):
        var next_profile = _native_function(bindings[kind], "transformer_arithmetic_profile")()
        var want_mode = context["backend"]._DEFAULT_MODE if mode is None else mode
        var saved = fields.get("_arithmetic_profile")
        if saved is not None and Bool(py=fields.get("_arithmetic_profile_mode") == want_mode) and not Bool(py=saved == next_profile):
            raise Error("neural block: saved arithmetic profile differs from binding")
        fields["_arithmetic_profile"] = next_profile
        fields["_arithmetic_profile_mode"] = want_mode
        fields["_arithmetic_profile_changed"] = _native_function(bindings[kind], "transformer_arithmetic_profile_changed")()
    var owner = bi.object.__new__(context["model_class"])
    owner.kind = PythonObject(kind)
    owner.device = route
    owner.d_model = fields["d_model"]
    owner.numeric_profile = fields.get("numeric_profile", PythonObject("fp32_v1"))
    owner._head_int15 = fields.get("_int15")
    owner._blocks = bi.list()
    owner._blocks.append(block)
    _ = causal_lm_admit_binding(owner, bindings[kind], bindings["primitives"], bindings["linalg"] if owner._head_int15 is not None else PythonObject(None))
    return owner


def neural_block_init_binding(block: PythonObject, weights: PythonObject, kind_value: PythonObject, options: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    _ = generation_initialize_binding(context["generation_classes"])
    var kind = String(kind_value)
    var cpu = Bool(py=bi.isinstance(block, context["block_classes"]["cpu"][kind]))
    var route = PythonObject("cpu" if cpu else "gpu")
    # The established GPU block class routes to native host arithmetic on a
    # CPU-only installation, just as the package binding selector does.
    if context["selected_vendor"] is None or String(context["selected_vendor"]) == "cpu":
        route = PythonObject("cpu")
    var plan = bi.object.__new__(context["config_context"]["plan_class"])
    plan.kind = kind_value
    var norm_name = "input_layernorm.weight" if kind == "transformer" else ("norm.weight" if kind == "mamba1" else "block_norm.weight")
    var norm = _model_f32(weights[norm_name], norm_name, context)
    if len(norm.shape) != 1 or Int(py=norm.shape[0]) < 1:
        raise Error("neural block: norm weight requires positive d_model")
    plan.d_model = norm.shape[0]
    var fields = bi.object.__getattribute__(block, "__dict__")
    var mode = options[1]
    fields["numeric_mode"] = mode
    plan.block_options = bi.dict()
    plan.block_kwargs = bi.dict()
    var profile = PythonObject("fp32_v1")
    if kind == "transformer":
        profile = _model_profile(options[2], "transformer_block", context)
        plan.n_heads = options[3]
        var names = context["block_option_names"]
        for i in range(len(names)):  # driver-loop: over transformer option keyword metadata
            plan.block_options[names[i]] = options[i + 4]
    elif kind == "mamba2":
        plan.block_kwargs["dt_limit"] = options[2]
    var bindings = _model_bindings(route, kind, context, mode)
    var created = _model_block(weights, plan, profile, route, bindings, context)
    fields.update(bi.object.__getattribute__(created, "__dict__"))
    fields["numeric_mode"] = mode
    fields["_native_kind"] = kind_value
    fields["_native_route"] = route
    return PythonObject(None)


def neural_block_allocate_binding(block: PythonObject, batch: PythonObject, capacity: PythonObject, context: PythonObject) raises -> PythonObject:
    _ = generation_initialize_binding(context["generation_classes"])
    var b = Int(py=batch)
    var cap = 1 if capacity is None else Int(py=capacity)
    if b < 1 or cap < 1:
        raise Error("neural block: batch_size and max_tokens must be positive")
    return causal_layer_allocate(neural_block_owner(block, context), 0, b, cap)


def _neural_input(value: PythonObject, d_model: Int, step: Bool, context: PythonObject) raises -> PythonObject:
    var x = _model_f32(value, "x", context)
    var bi = Python.import_module("builtins")
    if step and len(x.shape) == 2:
        var shaped = bi.object.__new__(context["array_class"])
        # A metadata view retains its native buffer owner; no Array callback.
        shaped._store = x._store
        shaped._base = x
        shaped._pin = x._pin
        shaped._mv = x._mv
        shaped._addr = x._addr
        shaped._readonly = x._readonly
        shaped.shape = bi.tuple(py_ints([Int(py=x.shape[0]), 1, Int(py=x.shape[1])]))
        shaped.dtype = x.dtype
        shaped.ndim = PythonObject(3)
        shaped.size = x.size
        shaped.nbytes = x.nbytes
        shaped.itemsize = x.itemsize
        shaped.order = x.order
        shaped.strides = bi.tuple(py_ints([d_model * 4, d_model * 4, 4]))
        x = shaped
    if len(x.shape) != 3 or Int(py=x.shape[0]) < 1 or Int(py=x.shape[1]) < 1 or Int(py=x.shape[2]) != d_model:
        raise Error("neural block: x must have shape (positive B, positive L, d_model)")
    if step and Int(py=x.shape[1]) != 1:
        raise Error("neural block: step requires one token per row")
    return x


def _neural_state(owner: PythonObject, state: PythonObject, b: Int, l: Int) raises:
    var bi = Python.import_module("builtins")
    var fields = bi.object.__getattribute__(owner._blocks[0], "__dict__")
    if String(owner.kind) == "transformer":
        if Int(py=state.batch_size) != b or Int(py=state.n_kv_heads) != Int(py=fields["n_kv_heads"]) or Int(py=state.head_dim) != Int(py=fields["head_dim"]) or Int(py=state.window) != Int(py=fields["window"]):
            raise Error("neural block: transformer state dimensions differ")
        var past = Int(py=state.cached_tokens)
        if past < 0 or past + l > Int(py=state.max_tokens) or past + l > Int(py=fields["max_positions"]):
            raise Error("neural block: sequence exceeds state capacity or max_positions")



def _neural_plain(owner: PythonObject, x: PythonObject, supplied_state: PythonObject, step: Bool) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var fields = bi.object.__getattribute__(owner._blocks[0], "__dict__")
    var b = Int(py=x.shape[0])
    var l = Int(py=x.shape[1])
    var d = Int(py=x.shape[2])
    var transformer = String(owner.kind) == "transformer"
    var e = owner._native_entries
    var reuse = transformer and Bool(py=e.__contains__("forward_session")) and String(getenv("MOJOLEARN_TRANSFORMER_LEGACY_SETUP")) != "1"
    var lock = fields.get("_runtime_lock") if transformer else PythonObject(None)
    var locked = transformer and (reuse or fields["_native_session"] is not None or fields["_int15"] is not None)
    if locked:
        lock.acquire()
    try:
        if transformer and fields["_native_session"] is not None and (not reuse or fields["_session_binding"] is not e["block_binding"]):
            _native_function(fields["_session_binding"], "transformer_session_close")(fields["_native_session"])
            fields["_native_session"] = PythonObject(None)
            fields["_session_binding"] = PythonObject(None)
        if transformer and supplied_state is None and not step and fields["_int15"] is None:
            var fresh_session = reuse and Bool(py=e.__contains__("transformer_session_forward_fresh")) and String(getenv("MOJOLEARN_TRANSFORMER_SESSION_FRESH")) != "0"
            if fresh_session or Bool(py=e.__contains__("transformer_forward_fresh")):
                var y = generation_array([b, l, d])
                var keep = List[PythonObject]()
                var addresses = py_ints([_addr(x, keep)])
                for i in range(len(fields["_w"])):  # driver-loop: over fixed transformer weight slots
                    addresses.append(_addr(fields["_w"][i], keep))
                var params = py_ints([b, l, d, Int(py=fields["n_heads"]), Int(py=fields["n_kv_heads"]), Int(py=fields["head_dim"]), Int(py=fields["intermediate"])])
                if fresh_session:
                    addresses.append(0)
                    addresses.append(0)
                    params.append(l)
                    params.append(0)
                addresses.append(_addr(y, keep))
                params.append(fields["window"])
                if Bool(py=fields["_extended"]):
                    for i in range(len(fields["_wopt"])):  # driver-loop: over optional transformer weight slots
                        addresses.append(_addr(fields["_wopt"][i], keep))
                    params.extend(fields["_opts_tail"])
                if fresh_session:
                    if fields["_native_session"] is None:
                        fields["_native_session"] = e["create"]()
                        fields["_session_binding"] = e["block_binding"]
                    e["transformer_session_forward_fresh"](fields["_native_session"], addresses, params)
                else:
                    e["transformer_forward_fresh"](addresses, params)
                if locked:
                    lock.release()
                _ = keep^
                return y
        var state = supplied_state
        if state is None:
            state = causal_layer_allocate(owner, 0, b, l)
        _neural_state(owner, state, b, l)
        var result = causal_layer_run(owner, 0, x, state, step)
        if locked:
            lock.release()
        return result
    except error:
        if locked:
            lock.release()
        if supplied_state is not None and String(owner.kind) == "mamba1":
            supplied_state._nn34_usable = PythonObject(False)
        raise error


def _neural_lengths(value: PythonObject, b: Int, l: Int, context: PythonObject) raises -> List[Int]:
    var bi = Python.import_module("builtins")
    var values = value
    if bi.type(value) is context["array_class"]:
        values = value._mv
        if len(value.shape) != 1:
            raise Error("neural block: lengths must be one-dimensional")
    elif bi.type(value) is not bi.list and bi.type(value) is not bi.tuple:
        values = bi.memoryview(value)
        if Int(py=values.ndim) != 1:
            raise Error("neural block: lengths must be one-dimensional")
    if len(values) != b:
        raise Error("neural block: one length is required per batch row")
    var result = List[Int]()
    for i in range(b):
        var v = values[i]
        if bi.type(v) is not bi.int:
            raise Error("TypeError: neural block lengths entries must be integers, not bool or float")
        var n = Int(py=v)
        if n < 1 or n > l:
            raise Error("neural block: lengths must be in [1, L]")
        result.append(n)
    return result^


def neural_block_forward_binding(block: PythonObject, input: PythonObject, state_value: PythonObject, step_value: PythonObject, lengths: PythonObject, context: PythonObject) raises -> PythonObject:
    _ = generation_initialize_binding(context["generation_classes"])
    var bi = Python.import_module("builtins")
    var owner = neural_block_owner(block, context)
    var d = Int(py=owner.d_model)
    var step = Bool(py=step_value)
    if step and state_value is None:
        raise Error("neural block: step requires a state")
    var x = _neural_input(input, d, step, context)
    var b = Int(py=x.shape[0])
    var l = Int(py=x.shape[1])
    if lengths is not None:
        if state_value is not None:
            raise Error("neural block: lengths cannot be combined with state")
        var sizes = _neural_lengths(lengths, b, l, context)
        var padded = generation_array([b, l, d])
        for row in range(b):
            memcpy(dest=f32_ptr(Int(py=padded._addr)) + row * l * d, src=f32_ptr(Int(py=x._addr)) + row * l * d, count=sizes[row] * d)
        var output = _neural_plain(owner, padded, PythonObject(None), False)
        var result = generation_array([b, l, d])
        for row in range(b):
            memcpy(dest=f32_ptr(Int(py=result._addr)) + row * l * d, src=f32_ptr(Int(py=output._addr)) + row * l * d, count=sizes[row] * d)
        return result
    return _neural_plain(owner, x, state_value, step)


def neural_block_c[operation: Int](py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=args_ptr)
        var result = PythonObject(None)
        comptime if operation == 0:
            result = neural_block_init_binding(args[0], args[1], args[2], args[3], args[4])
        elif operation == 1:
            result = neural_block_allocate_binding(args[0], args[1], args[2], args[3])
        else:
            result = neural_block_forward_binding(args[0], args[1], args[2], args[3], args[4], args[5])
        return result.steal_data()
    except error:
        if String(error).startswith("TypeError:"):
            return raise_python_exception(error, ExceptionType.TypeError)
        return raise_python_exception(error, ExceptionType.ValueError)
