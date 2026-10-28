# SPDX-License-Identifier: Apache-2.0
"""Native standalone block construction, state allocation and sequence execution."""
from std.python import Python, PythonObject
from core.neural_block_public_impl import neural_copy_row_prefixes
from std.python._cpython import PyObjectPtr, GILReleased
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_contains, py_get, py_setattr, py_setitem, py_type
from std.memory import memcpy
from std.os import getenv
from bindings.generation_buffers import generation_initialize_binding, generation_array, py_ints
from bindings.causal_lm_construct import _model_block, _model_profile, _model_bindings, _model_f32, _model_shape
from bindings.causal_lm_public import causal_lm_admit_binding, causal_layer_allocate, causal_layer_run, _addr, _native_function, causal_transformer_admin
from core.hostptr import f32_ptr
from core.host_helpers_impl import all_finite_f32
from core.native_descriptor_abi import TRANSFORMER_BLOCK_ABI_NAME, MAMBA_BLOCK_ABI_NAME
from providers.python.operation_lock_pyapi import OperationLockLease, operation_lock_create
from providers.python.neural_descriptor_pyapi import (
    neural_descriptor_reuse, neural_descriptor_execute, neural_descriptor_reserved_execute,
    neural_descriptor_handle,
)



def neural_block_owner(block: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var fields = py_call(bi.object.__getattribute__, block, "__dict__")
    var kind = String(fields["_native_kind"])
    var route = fields["_native_route"]
    var mode = py_get(fields, "numeric_mode")
    var bindings = _model_bindings(route, kind, context, mode)
    if kind == "transformer" and Bool(py=py_call(bi.hasattr, bindings[kind], "transformer_arithmetic_profile")):
        var next_profile = _native_function(bindings[kind], "transformer_arithmetic_profile")()
        var want_mode = context["backend"]._DEFAULT_MODE if mode is None else mode
        var saved = py_get(fields, "_arithmetic_profile")
        if saved is not None and Bool(py=py_get(fields, "_arithmetic_profile_mode") == want_mode) and not Bool(py=saved == next_profile):
            raise Error("neural block: saved arithmetic profile differs from binding")
        py_setitem(fields, "_arithmetic_profile", next_profile)
        py_setitem(fields, "_arithmetic_profile_mode", want_mode)
        py_setitem(fields, "_arithmetic_profile_changed", _native_function(bindings[kind], "transformer_arithmetic_profile_changed")())
    var owner = py_call(bi.object.__new__, context["model_class"])
    py_setattr(owner, "kind", PythonObject(kind))
    py_setattr(owner, "device", route)
    py_setattr(owner, "d_model", fields["d_model"])
    py_setattr(owner, "numeric_profile", py_get(fields, "numeric_profile", PythonObject("fp32_v1")))
    py_setattr(owner, "_head_int15", py_get(fields, "_int15"))
    py_setattr(owner, "_blocks", bi.list())
    _ = py_call(owner._blocks.append, block)
    _ = causal_lm_admit_binding(owner, bindings[kind], bindings["primitives"], bindings["linalg"] if owner._head_int15 is not None else PythonObject(None))
    return owner


def neural_block_init_binding(block: PythonObject, weights: PythonObject, kind_value: PythonObject, options: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    _ = generation_initialize_binding(context["generation_classes"])
    var kind = String(kind_value)
    var cpu = Bool(py=py_call(bi.isinstance, block, context["block_classes"]["cpu"][kind]))
    var route = PythonObject("cpu" if cpu else "gpu")
    # The established GPU block class routes to native host arithmetic on a
    # CPU-only installation, just as the package binding selector does.
    if context["selected_vendor"] is None or String(context["selected_vendor"]) == "cpu":
        route = PythonObject("cpu")
    var plan = py_call(bi.object.__new__, context["config_context"]["plan_class"])
    py_setattr(plan, "kind", kind_value)
    var norm_name = "input_layernorm.weight" if kind == "transformer" else ("norm.weight" if kind == "mamba1" else "block_norm.weight")
    var norm = _model_f32(weights[norm_name], norm_name, context)
    if len(norm.shape) != 1 or Int(py=norm.shape[0]) < 1:
        raise Error("neural block: norm weight requires positive d_model")
    py_setattr(plan, "d_model", norm.shape[0])
    var fields = py_call(bi.object.__getattribute__, block, "__dict__")
    var mode = options[1]
    py_setitem(fields, "numeric_mode", mode)
    py_setattr(plan, "block_options", bi.dict())
    py_setattr(plan, "block_kwargs", bi.dict())
    var profile = PythonObject("fp32_v1")
    if kind == "transformer":
        profile = _model_profile(options[2], "transformer_block", context)
        py_setattr(plan, "n_heads", options[3])
        var names = context["block_option_names"]
        for i in range(len(names)):  # driver-loop: over transformer option keyword metadata
            py_setitem(plan.block_options, names[i], options[i + 4])
    elif kind == "mamba2":
        py_setitem(plan.block_kwargs, "dt_limit", options[2])
    var bindings = _model_bindings(route, kind, context, mode)
    var created = _model_block(weights, plan, profile, route, bindings, context)
    _ = py_call(fields.update, py_call(bi.object.__getattribute__, created, "__dict__"))
    py_setitem(fields, "numeric_mode", mode)
    py_setitem(fields, "_native_kind", kind_value)
    py_setitem(fields, "_native_route", route)
    # Public parameter verification (2026-10-27): the constructor previously
    # retained only its ordered internal tensors, so inherited get_params
    # could not return the caller's weights and set_params could not rebuild.
    # Keep the original ordinary dictionary and its object identities.
    py_setitem(fields, "weights", weights)
    return PythonObject(None)


def neural_block_set_params_binding(block: PythonObject, updates: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    if py_type(updates) is not bi.dict:
        raise Error("TypeError: block parameters require an ordinary dictionary")
    if len(updates) == 0:
        return block
    var fields = py_call(bi.object.__getattribute__, block, "__dict__")
    var kind = String(fields["_native_kind"])
    var names = Python.list("weights", "numeric_mode")
    if kind == "mamba2":
        _ = py_call(names.append, "dt_limit")
    elif kind == "transformer":
        _ = py_call(names.append, "numeric_profile")
        _ = py_call(names.append, "n_heads")
        _ = py_call(names.extend, context["block_option_names"])
    var values = bi.dict()
    for i in range(len(names)):  # driver-loop: fixed constructor configuration, no Python callbacks
        py_setitem(values, names[i], fields[names[i]])
    var keys = py_call(bi.list, updates.keys())
    for i in range(len(keys)):  # driver-loop: validate all keys before constructing replacement
        if py_type(keys[i]) is not bi.str or not py_contains(values, keys[i]):
            raise Error("ValueError: Invalid parameter for neural block")
        py_setitem(values, keys[i], updates[keys[i]])
    var options = Python.list(PythonObject(None), values["numeric_mode"])
    if kind == "mamba2":
        _ = py_call(options.append, values["dt_limit"])
    elif kind == "transformer":
        _ = py_call(options.append, values["numeric_profile"])
        _ = py_call(options.append, values["n_heads"])
        var option_names = context["block_option_names"]
        for i in range(len(option_names)):  # driver-loop: native initializer's positional option schema
            _ = py_call(options.append, values[option_names[i]])
    var replacement = py_call(bi.object.__new__, py_type(block))
    _ = neural_block_init_binding(replacement, values["weights"], PythonObject(kind), options, context)
    # Preserve the established reset semantics and dictionary identity. A
    # rejected replacement leaves the live block's fields/session untouched.
    var fresh = py_call(bi.object.__getattribute__, replacement, "__dict__")
    _ = py_call(fields.clear)
    _ = py_call(fields.update, fresh)
    return block


def neural_block_allocate_binding(block: PythonObject, batch: PythonObject, capacity: PythonObject, context: PythonObject) raises -> PythonObject:
    _ = generation_initialize_binding(context["generation_classes"])
    # Integers only (anything with __index__): a str is not parsed and a
    # bool is not counted as 0 or 1.
    var bi = Python.import_module("builtins")
    if py_type(batch) is bi.bool or py_type(capacity) is bi.bool:
        raise Error("TypeError: neural block: batch_size and max_tokens must be integers")
    var b = 0
    var cap = 1
    try:
        var operator = Python.import_module("operator")
        b = Int(py=py_call(operator.index, batch))
        if capacity is not None:
            cap = Int(py=py_call(operator.index, capacity))
    except error:
        raise Error("TypeError: neural block: batch_size and max_tokens must be integers")
    if b < 1 or cap < 1:
        raise Error("neural block: batch_size and max_tokens must be positive")
    return causal_layer_allocate(neural_block_owner(block, context), 0, b, cap)


def _neural_input(value: PythonObject, d_model: Int, step: Bool, context: PythonObject) raises -> PythonObject:
    var x = _model_f32(value, "x", context)
    var bi = Python.import_module("builtins")
    if step and len(x.shape) == 2:
        var shaped = py_call(bi.object.__new__, context["array_class"])
        # A metadata view retains its native buffer owner; no Array callback.
        py_setattr(shaped, "_store", x._store)
        py_setattr(shaped, "_base", x)
        py_setattr(shaped, "_pin", x._pin)
        py_setattr(shaped, "_mv", x._mv)
        py_setattr(shaped, "_addr", x._addr)
        py_setattr(shaped, "_readonly", x._readonly)
        py_setattr(shaped, "shape", py_call(bi.tuple, py_ints([Int(py=x.shape[0]), 1, Int(py=x.shape[1])])))
        py_setattr(shaped, "dtype", x.dtype)
        py_setattr(shaped, "ndim", PythonObject(3))
        py_setattr(shaped, "size", x.size)
        py_setattr(shaped, "nbytes", x.nbytes)
        py_setattr(shaped, "itemsize", x.itemsize)
        py_setattr(shaped, "order", x.order)
        py_setattr(shaped, "strides", py_call(bi.tuple, py_ints([d_model * 4, d_model * 4, 4])))
        x = shaped
    if len(x.shape) != 3 or Int(py=x.shape[0]) < 1 or Int(py=x.shape[1]) < 1 or Int(py=x.shape[2]) != d_model:
        raise Error("neural block: x must be (B, L, d_model)" + (String(" -- or (B, d_model) for a step --") if step else String("")) +
                    " with positive B and L and the weights' d_model " + String(d_model) + ", got shape " + String(x.shape))
    if step and Int(py=x.shape[1]) != 1:
        raise Error("neural block: a decode step takes exactly one token per row, got L = " + String(x.shape[1]) + "; use forward() for prefill")
    return x


def _neural_state(owner: PythonObject, state: PythonObject, b: Int, l: Int) raises:
    var bi = Python.import_module("builtins")
    var fields = py_call(bi.object.__getattribute__, owner._blocks[0], "__dict__")
    if String(owner.kind) == "transformer":
        if Int(py=state.batch_size) != b or Int(py=state.n_kv_heads) != Int(py=fields["n_kv_heads"]) or Int(py=state.head_dim) != Int(py=fields["head_dim"]) or Int(py=state.window) != Int(py=fields["window"]):
            raise Error("neural block: transformer state dimensions differ")
        var past = Int(py=state.cached_tokens)
        if past < 0 or past + l > Int(py=state.max_tokens) or past + l > Int(py=fields["max_positions"]):
            raise Error("neural block: sequence exceeds state capacity or max_positions")



def _neural_plain(owner: PythonObject, x: PythonObject, supplied_state: PythonObject, step: Bool) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var fields = py_call(bi.object.__getattribute__, owner._blocks[0], "__dict__")
    var b = Int(py=x.shape[0])
    var l = Int(py=x.shape[1])
    var d = Int(py=x.shape[2])
    var transformer = String(owner.kind) == "transformer"
    # Admission refusal must not poison a valid state owned by a decode
    # session. Only a failure after entering execution can invalidate it.
    if supplied_state is not None and py_call(bi.getattr, supplied_state, "_resident_session", PythonObject(None)) is not None:
        raise Error("neural block: state is owned by a resident decode session")
    var e = owner._native_entries
    var reuse = transformer and py_contains(e, "forward_session") and String(getenv("MOJOLEARN_TRANSFORMER_LEGACY_SETUP")) != "1"
    var lock = py_get(fields, "_runtime_lock") if transformer else PythonObject(None)
    var lock_lease = OperationLockLease(lock)
    var locked = transformer and (reuse or fields["_native_session"] is not None or fields["_int15"] is not None)
    if locked:
        lock_lease.acquire()
    try:
        if transformer and fields["_native_session"] is not None and (not reuse or fields["_session_binding"] is not e["block_binding"]):
            causal_transformer_admin(fields["_session_binding"], fields["_native_session"], 13)
            py_setitem(fields, "_native_session", PythonObject(None))
            py_setitem(fields, "_session_binding", PythonObject(None))
        if transformer and supplied_state is None and not step and fields["_int15"] is None:
            var fresh_session = reuse and py_contains(e, "transformer_session_forward_fresh") and String(getenv("MOJOLEARN_TRANSFORMER_SESSION_FRESH")) != "0"
            if fresh_session or py_contains(e, "transformer_forward_fresh"):
                var y = generation_array([b, l, d])
                var keep = List[PythonObject]()
                var addresses: List[Int] = [_addr(x, keep)]
                var weights = fields["_w"]
                for i in range(len(weights)):  # driver-loop: admit transformer weight slots once
                    addresses.append(_addr(weights[i], keep))
                var params: List[Int] = [b, l, d, Int(py=fields["n_heads"]), Int(py=fields["n_kv_heads"]), Int(py=fields["head_dim"]), Int(py=fields["intermediate"])]
                if fresh_session:
                    addresses.append(0)
                    addresses.append(0)
                    params.append(l)
                    params.append(0)
                addresses.append(_addr(y, keep))
                params.append(Int(py=fields["window"]))
                if Bool(py=fields["_extended"]):
                    var optional = fields["_wopt"]
                    for i in range(len(optional)):  # driver-loop: admit optional transformer weight slots once
                        addresses.append(_addr(optional[i], keep))
                    var options = fields["_opts_tail"]
                    for i in range(len(options)):  # small-loop(block options integer span)
                        params.append(Int(py=options[i]))
                var native = py_get(e, "native_block_operations")
                if fresh_session:
                    if fields["_native_session"] is None:
                        py_setitem(fields, "_native_session", e["create"]())
                        py_setitem(fields, "_session_binding", e["block_binding"])
                    if native is not None:
                        var session_owner = fields["_native_session"]
                        var handle = Int(py=py_call(_native_function(e["block_binding"], "transformer_native_session_handle_v1"), session_owner))
                        _ = neural_descriptor_reserved_execute(native, session_owner, handle, 0, 1, 2, addresses, params, [])
                    else:
                        _ = py_call(e["transformer_session_forward_fresh"], fields["_native_session"], py_ints(addresses), py_ints(params))
                elif native is not None:
                    _ = neural_descriptor_execute(native, 4, addresses, params, [])
                else:
                    _ = py_call(e["transformer_forward_fresh"], py_ints(addresses), py_ints(params))
                if locked:
                    lock_lease.release()
                _ = keep^
                return y
        var state = supplied_state
        if state is None:
            state = causal_layer_allocate(owner, 0, b, l)
        _neural_state(owner, state, b, l)
        var result = causal_layer_run(owner, 0, x, state, step)
        if locked:
            lock_lease.release()
        return result
    except error:
        if locked:
            lock_lease.release()
        if supplied_state is not None and String(owner.kind) == "mamba1":
            py_setattr(supplied_state, "_nn34_usable", PythonObject(False))
        raise error


def _neural_lengths(value: PythonObject, b: Int, l: Int, context: PythonObject) raises -> List[Int]:
    var bi = Python.import_module("builtins")
    var values = value
    if py_type(value) is context["array_class"]:
        values = value._mv
        if len(value.shape) != 1:
            raise Error("neural block: lengths must be one-dimensional")
    elif py_type(value) is not bi.list and py_type(value) is not bi.tuple:
        values = py_call(bi.memoryview, value)
        if Int(py=values.ndim) != 1:
            raise Error("neural block: lengths must be one-dimensional")
    if len(values) != b:
        raise Error("neural block: one length is required per batch row")
    var result = List[Int]()
    for i in range(b):  # small-loop(one length argument word per batch row)
        var v = values[i]
        if py_type(v) is not bi.int:
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
        raise Error("neural block: state is required for step (a decode step continues a sequence; allocate_state(B) makes the fresh one)")
    var x = _neural_input(input, d, step, context)
    var b = Int(py=x.shape[0])
    var l = Int(py=x.shape[1])
    if lengths is not None:
        if state_value is not None:
            raise Error("neural block: lengths= (a ragged, right-padded batch) cannot carry a state; run the ragged prefill with state=None, or continue each sequence as its own batch")
        var sizes = _neural_lengths(lengths, b, l, context)
        var padded = generation_array([b, l, d])
        neural_copy_row_prefixes(Int(py=padded._addr), Int(py=x._addr), sizes, b, l, d)
        # `x` may be the only reference to a fresh Array converted from a NumPy
        # input; Mojo destroys a local at its last named use, which was the
        # `_addr` read above, so without this the copy read freed memory and
        # row 0's first floats came back as allocator bookkeeping.
        _ = x^
        var output = _neural_plain(owner, padded, PythonObject(None), False)
        var result = generation_array([b, l, d])
        neural_copy_row_prefixes(Int(py=result._addr), Int(py=output._addr), sizes, b, l, d)
        _ = output^  # the copy reads `output` by address
        _ = padded^
        return result
    return _neural_plain(owner, x, state_value, step)


def neural_training_entries(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var mode = py_get(fields, "numeric_mode")
    if mode is None:
        mode = context["backend"]._DEFAULT_MODE
    else:
        mode = mode.strip().lower()
    if String(mode) != "identical" and String(mode) != "fast":
        raise Error("NotImplementedError: neural backward requires identical or fast mode")
    var key = fields["_native_route"] + PythonObject(":") + mode
    var selected = py_get(context["training_bindings"], key)
    if selected is None:
        raise Error("NotImplementedError: native training bindings unavailable for " + String(key))
    var family = "transformer" if String(fields["_native_kind"]) == "transformer" else "mamba"
    var entries = selected[family]
    if py_type(entries) is bi.str:
        raise Error("RuntimeError: " + String(entries))
    var vendor = String(_native_function(entries, family + "_vendor")())
    var code = Int(py=_native_function(entries, family + "_numeric_mode")())
    if code != (1 if String(mode) == "identical" else 0):
        raise Error("RuntimeError: native training binding numeric mode differs from the block")
    if String(fields["_native_route"]) == "cpu":
        if vendor != "cpu":
            raise Error("RuntimeError: CPU backward requires native CPU kernels")
    elif vendor != "metal" and vendor != "cuda" and vendor != "hip":
        raise Error("RuntimeError: GPU backward requires native GPU kernels")
    if family == "transformer" and Bool(py=py_call(bi.hasattr, entries, "transformer_arithmetic_profile")):
        var profile = _native_function(entries, "transformer_arithmetic_profile")()
        var saved = py_get(fields, "_arithmetic_profile")
        if saved is not None and Bool(py=py_get(fields, "_arithmetic_profile_mode") == mode) and not Bool(py=saved == profile):
            raise Error("RuntimeError: saved transformer arithmetic profile differs from training binding")
        py_setitem(fields, "_arithmetic_profile", profile)
        py_setitem(fields, "_arithmetic_profile_mode", mode)
        py_setitem(fields, "_arithmetic_profile_changed", _native_function(entries, "transformer_arithmetic_profile_changed")())
    if family == "transformer":
        var native = PythonObject(None)
        if code == 1 and vendor != "cpu" and getenv("MOJOLEARN_NEURAL_NATIVE_DISPATCH", "0") == "1":
            native = neural_descriptor_reuse(py_get(fields, "_native_block_provider_cache"),
                _native_function(entries, "transformer_block_api_v1"), String(TRANSFORMER_BLOCK_ABI_NAME), 35)
        py_setitem(fields, "_native_block_provider_cache", native)
    else:
        var native = PythonObject(None)
        if code == 1 and vendor != "cpu" and getenv("MOJOLEARN_NEURAL_NATIVE_DISPATCH", "0") == "1":
            native = neural_descriptor_reuse(py_get(fields, "_native_mamba_block_provider_cache"),
                _native_function(entries, "mamba_block_api_v1"), String(MAMBA_BLOCK_ABI_NAME), 44)
        py_setitem(fields, "_native_mamba_block_provider_cache", native)
    return entries


def _neural_int_words(values: PythonObject) raises -> List[Int]:
    # Internal boundary vectors already contain admitted scalar/address words.
    var result = List[Int]()
    for i in range(len(values)):  # small-loop(public operation descriptor words)
        result.append(Int(py=values[i]))
    return result^


def neural_mamba_session_call(native: PythonObject, fields: PythonObject, entries: PythonObject,
                              operation: Int, addresses: List[Int], params: List[Int]) raises -> Int:
    # Operation admission only. The common provider pins this exact typed
    # session and compiled image and reserves/releases under the GIL.
    var session = fields["_prefill_session"]
    var handle = neural_descriptor_handle(native,
        _native_function(entries, "mamba_prefill_native_handle_v1"), session)
    return neural_descriptor_reserved_execute(native, session, handle, 0, 1, operation,
        addresses, params, [])


def neural_mamba3_restore(fields: PythonObject, entries: PythonObject, weights: PythonObject) raises:
    var bi = Python.import_module("builtins")
    if Bool(py=py_get(fields, "_owned_restore_pending", PythonObject(False))):
        if not Bool(py=py_call(bi.hasattr, entries, "mamba3_owned_weights_enabled")) or not Bool(py=_native_function(entries, "mamba3_owned_weights_enabled")()):
            raise Error("RuntimeError: Mamba3 owned weights require the compiled owned-weight entry")
        if py_get(fields, "_prefill_session") is None or py_get(fields, "_prefill_binding") is not entries.canonical:
            py_setitem(fields, "_prefill_session", _native_function(entries, "mamba3_prefill_session_create")())
            py_setitem(fields, "_prefill_binding", entries.canonical)
        var addresses = bi.list()
        for i in range(len(weights)):  # driver-loop: over the fixed owned-weight registry
            _ = py_call(addresses.append, weights[i]._addr)
        var native = py_get(fields, "_native_mamba_block_provider_cache")
        if native is not None:
            var generation = neural_mamba_session_call(native, fields, entries, 15,
                _neural_int_words(addresses), [Int(py=fields["d_model"])])
            py_setitem(fields, "_owned_generation", PythonObject(generation))
        else:
            py_setitem(fields, "_owned_generation", py_call(_native_function(entries, "mamba3_prefill_session_install_weights"), fields["_prefill_session"], addresses, fields["d_model"]))
        py_setitem(fields, "_owned_native_provider", native)
        py_setitem(fields, "_owned_restore_pending", False)
    if py_get(fields, "_owned_generation") is not None:
        if py_get(fields, "_prefill_binding") is not entries.canonical or getenv("MOJOLEARN_MAMBA3_LEGACY_SETUP", "0") == "1":
            raise Error("RuntimeError: Mamba3 owned snapshot requires its installed binding/setup")


def neural_block_backward_binding(block: PythonObject, input: PythonObject, grad_output: PythonObject, context: PythonObject) raises -> PythonObject:
    _ = generation_initialize_binding(context["generation_classes"])
    var bi = Python.import_module("builtins")
    var fields = py_call(bi.object.__getattribute__, block, "__dict__")
    var kind = String(fields["_native_kind"])
    var transformer = kind == "transformer"
    if transformer and (fields["_int15"] is not None or Bool(py=fields["_extended"])):
        var named = String("")
        for key in ["norm", "mlp", "qkv_bias", "o_bias", "norm_bias", "mlp_bias", "qk_norm", "rope_scaling"]:  # driver-loop: over the named block options
            if py_contains(fields, key):
                named += ", " + key + "=" + String(py_call(bi.repr, fields[key]))
        raise Error("NotImplementedError: TransformerBlock.backward requires default float32 block options (the default block options only); numeric_profile="
                    + String(fields["numeric_profile"]) + named)
    var entries = neural_training_entries(fields, context)
    var name = ("transformer" if transformer else kind) + "_backward"
    if not Bool(py=py_call(bi.hasattr, entries, name)):
        raise Error("RuntimeError: loaded native training binding lacks " + name)
    var d = Int(py=fields["d_model"])
    var x = _neural_input(input, d, False, context)
    var b = Int(py=x.shape[0])
    var l = Int(py=x.shape[1])
    var dy = _model_f32(grad_output, "grad_output", context)
    _model_shape(dy, [b, l, d], "grad_output")
    var weights = fields["_w"]
    if not transformer:
        var mode = py_get(fields, "numeric_mode")
        if mode is None:
            mode = context["backend"]._DEFAULT_MODE
        else:
            mode = mode.strip().lower()
        var grad_pointer = f32_ptr(Int(py=dy._addr))
        var grad_count = Int(py=dy.size)
        var finite = 0
        with GILReleased(Python()):
            finite = all_finite_f32(grad_pointer, grad_count)
        if finite != 1:
            raise Error("neural backward: grad_output must be finite")
        # Preserve the existing current-weight shape validation without
        # calling a Python block constructor from the operation.
        var plan = py_call(bi.object.__new__, context["config_context"]["plan_class"])
        py_setattr(plan, "kind", PythonObject(kind))
        py_setattr(plan, "d_model", PythonObject(d))
        py_setattr(plan, "block_kwargs", bi.dict())
        if kind == "mamba2":
            py_setitem(plan.block_kwargs, "dt_limit", fields["dt_limit"])
        var named = bi.dict()
        var names = block._W_NAMES
        for i in range(len(names)):  # driver-loop: over the fixed block weight registry
            py_setitem(named, names[i], weights[i])
        var checked = _model_block(named, plan, PythonObject("fp32_v1"), fields["_native_route"], _model_bindings(fields["_native_route"], kind, context, mode), context)
        weights = py_call(bi.object.__getattribute__, checked, "__dict__")["_w"]
    var keep = List[PythonObject]()
    var addresses = py_ints([_addr(x, keep)])
    var gradients = bi.list()
    _ = py_call(gradients.append, generation_array([b, l, d]))
    for i in range(len(weights)):  # driver-loop: over parameter buffers and their shapes
        _ = py_call(addresses.append, _addr(weights[i], keep))
        var shape = List[Int]()
        for j in range(len(weights[i].shape)):  # driver-loop: over tensor shape metadata
            shape.append(Int(py=weights[i].shape[j]))
        _ = py_call(gradients.append, generation_array(shape))
    _ = py_call(addresses.append, _addr(dy, keep))
    for i in range(len(gradients)):  # driver-loop: over gradient buffer addresses
        _ = py_call(addresses.append, _addr(gradients[i], keep))
    var params = py_ints([b, l, d])
    if transformer:
        for key in [String("n_heads"), String("n_kv_heads"), String("head_dim"), String("intermediate"), String("window")]:  # driver-loop: over transformer launch dimensions
            _ = py_call(params.append, fields[key])
        var lock = fields["_runtime_lock"]
        var lock_lease = OperationLockLease(lock)
        lock_lease.acquire()
        try:
            var session = Bool(py=py_call(bi.hasattr, entries, "transformer_session_backward")) and String(getenv("MOJOLEARN_TRANSFORMER_LEGACY_SETUP")) != "1"
            if session:
                if fields["_native_session"] is not None and fields["_session_binding"] is not entries.canonical:
                    causal_transformer_admin(fields["_session_binding"], fields["_native_session"], 13)
                    py_setitem(fields, "_native_session", PythonObject(None))
                    py_setitem(fields, "_session_binding", PythonObject(None))
                if fields["_native_session"] is None:
                    py_setitem(fields, "_native_session", _native_function(entries, "transformer_session_create")())
                    py_setitem(fields, "_session_binding", entries.canonical)
                var native = py_get(fields, "_native_block_provider_cache")
                if native is not None:
                    var session_owner = fields["_native_session"]
                    var handle = Int(py=py_call(_native_function(entries, "transformer_native_session_handle_v1"), session_owner))
                    _ = neural_descriptor_reserved_execute(native, session_owner, handle, 0, 1, 3,
                        _neural_int_words(addresses), _neural_int_words(params), [])
                else:
                    _ = py_call(_native_function(entries, "transformer_session_backward"), fields["_native_session"], addresses, params)
            else:
                if fields["_native_session"] is not None and Bool(py=py_call(bi.hasattr, entries, "transformer_session_invalidate_tape")):
                    causal_transformer_admin(entries, fields["_native_session"], 12)
                var native = py_get(fields, "_native_block_provider_cache")
                if native is not None:
                    _ = neural_descriptor_execute(native, 5, _neural_int_words(addresses), _neural_int_words(params), [])
                else:
                    _ = py_call(_native_function(entries, name), addresses, params)
            lock_lease.release()
        except error:
            lock_lease.release()
            raise error
    else:
        var native = py_get(fields, "_native_mamba_block_provider_cache")
        if kind == "mamba2":
            _ = py_call(params.extend, fields["dt_limit"])
        if kind == "mamba3":
            neural_mamba3_restore(fields, entries, weights)
            var generation = py_get(fields, "_owned_generation")
            var session = Bool(py=py_call(bi.hasattr, entries, "mamba3_prefill_session_backward")) and getenv("MOJOLEARN_MAMBA3_LEGACY_SETUP", "0") != "1"
            if generation is not None:
                var owned_addresses = py_ints([Int(py=addresses[0])])
                for i in range(10, len(addresses)):  # driver-loop: over input-gradient and result address slots
                    _ = py_call(owned_addresses.append, addresses[i])
                _ = py_call(params.append, generation)
                if native is not None:
                    _ = neural_mamba_session_call(native, fields, entries, 9,
                        _neural_int_words(owned_addresses), _neural_int_words(params))
                else:
                    _ = py_call(_native_function(entries, "mamba3_prefill_session_backward_owned"), fields["_prefill_session"], owned_addresses, params)
            elif session:
                if py_get(fields, "_prefill_session") is None or py_get(fields, "_prefill_binding") is not entries.canonical:
                    py_setitem(fields, "_prefill_session", _native_function(entries, "mamba3_prefill_session_create")())
                    py_setitem(fields, "_prefill_binding", entries.canonical)
                if native is not None:
                    _ = neural_mamba_session_call(native, fields, entries, 7,
                        _neural_int_words(addresses), _neural_int_words(params))
                else:
                    _ = py_call(_native_function(entries, "mamba3_prefill_session_backward"), fields["_prefill_session"], addresses, params)
            else:
                if native is not None:
                    _ = neural_descriptor_execute(native, 4, _neural_int_words(addresses), _neural_int_words(params), [])
                else:
                    _ = py_call(_native_function(entries, name), addresses, params)
        else:
            if native is not None:
                var words: List[Int] = [Int(py=params[0]), Int(py=params[1]), Int(py=params[2])]
                var floats = List[Float64]()
                if kind == "mamba2":
                    floats.append(Float64(py=params[3]))
                    floats.append(Float64(py=params[4]))
                _ = neural_descriptor_execute(native, 3 if kind == "mamba2" else 2,
                    _neural_int_words(addresses), words, floats)
            else:
                _ = py_call(_native_function(entries, name), addresses, params)
    var result = bi.dict()
    py_setitem(result, "x", gradients[0])
    var names = block._W_NAMES
    for i in range(len(names)):  # driver-loop: over named gradient slots
        py_setitem(result, names[i], gradients[i + 1])
    _ = keep^
    return result



def neural_block_getstate_binding(block: PythonObject, context: PythonObject) raises -> PythonObject:
    # Preserve arithmetic-profile admission before serialization. This native
    # admission uses the same binding contract as public forward, not the
    # Python _extension helper. Only ordinary model state crosses the boundary.
    var bi = Python.import_module("builtins")
    var fields = py_call(bi.object.__getattribute__, block, "__dict__")
    var mode = py_get(fields, "numeric_mode")
    if mode is None:
        mode = context["backend"]._DEFAULT_MODE
    var bindings = _model_bindings(fields["_native_route"], "transformer", context, mode)
    var entries = bindings["transformer"]
    var saved = py_get(fields, "_arithmetic_profile")
    var saved_mode = py_get(fields, "_arithmetic_profile_mode", PythonObject("identical") if saved is not None else PythonObject(None))
    if Bool(py=py_call(bi.hasattr, entries, "transformer_numeric_mode")):
        var code = Int(py=_native_function(entries, "transformer_numeric_mode")())
        if code != (1 if String(mode) == "identical" else 0):
            raise Error("RuntimeError: transformer serialized model numeric mode differs from binding")
    if Bool(py=py_call(bi.hasattr, entries, "transformer_arithmetic_profile")):
        var profile = _native_function(entries, "transformer_arithmetic_profile")()
        if saved is not None and Bool(py=saved_mode == mode) and not Bool(py=saved == profile):
            raise Error("transformer serialized model arithmetic profile differs from binding")
        py_setitem(fields, "_arithmetic_profile", profile)
        py_setitem(fields, "_arithmetic_profile_mode", mode)
        py_setitem(fields, "_arithmetic_profile_changed", _native_function(entries, "transformer_arithmetic_profile_changed")())
    elif saved is not None:
        raise Error("transformer: a tagged model requires a binding that reports its arithmetic profile")
    var result = py_call(bi.dict, fields)
    var transient_names = [String("_runtime_lock"), String("_native_session"), String("_session_binding"),
        String("_native_forward_owner"), String("_native_block_provider_cache"),
        String("_native_session_provider_cache"), String("_native_primitives_provider_cache")]
    for name in transient_names:  # driver-loop: over operation owners excluded from serialized model state
        _ = py_call(result.pop, name, PythonObject(None))
    return result


def neural_block_setstate_binding(block: PythonObject, state: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    if py_type(state) is not bi.dict:
        raise Error("TypeError: neural block restore requires an exact state dictionary")
    var fields = py_call(bi.object.__getattribute__, block, "__dict__")
    _ = py_call(fields.update, state)
    var mode = py_get(fields, "numeric_mode")
    if mode is None:
        mode = context["backend"]._DEFAULT_MODE
    if String(mode) == "identical" and getenv("MOJOLEARN_NEURAL_NATIVE_DISPATCH", "0") == "1":
        py_setitem(fields, "_runtime_lock", operation_lock_create())
    else:
        py_setitem(fields, "_runtime_lock", context["rlock"]())
    var session = PythonObject(None)
    var binding = PythonObject(None)
    if String(fields["_native_route"]) == "gpu" and String(mode) == "identical" and getenv("MOJOLEARN_NEURAL_NATIVE_DISPATCH", "0") == "1":
        var bindings = _model_bindings(fields["_native_route"], "transformer", context, mode)
        binding = bindings["transformer"]
        session = _native_function(binding, "transformer_session_create")()
    py_setitem(fields, "_native_session", session)
    py_setitem(fields, "_session_binding", binding)
    py_setitem(fields, "_native_forward_owner", PythonObject(None))
    py_setitem(fields, "_native_block_provider_cache", PythonObject(None))
    return PythonObject(None)

def neural_block_c[operation: Int](py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=args_ptr)
        var result = PythonObject(None)
        comptime if operation == 0:
            result = neural_block_init_binding(args[0], args[1], args[2], args[3], args[4])
        elif operation == 1:
            result = neural_block_allocate_binding(args[0], args[1], args[2], args[3])
        elif operation == 2:
            result = neural_block_forward_binding(args[0], args[1], args[2], args[3], args[4], args[5])
        elif operation == 4:
            result = neural_block_set_params_binding(args[0], args[1], args[2])
        elif operation == 5:
            result = neural_block_getstate_binding(args[0], args[1])
        elif operation == 6:
            result = neural_block_setstate_binding(args[0], args[1], args[2])
        else:
            result = neural_block_backward_binding(args[0], args[1], args[2], args[3])
        return result.steal_data()
    except error:
        if String(error).startswith("TypeError:"):
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        if String(error).startswith("NotImplementedError:"):
            return raise_python_exception(error_without_class(error, "NotImplementedError"), ExceptionType("PyExc_NotImplementedError"))
        if String(error).startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
