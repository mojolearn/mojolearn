# SPDX-License-Identifier: Apache-2.0
"""Native standalone block construction, state allocation and sequence execution."""
from std.python import Python, PythonObject
from core.neural_block_public_impl import neural_copy_row_prefixes
from std.python._cpython import PyObjectPtr, GILReleased
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_contains, py_get, py_setattr, py_setitem, py_type
from bindings.neural_block_admission import (
    neural_block_fields, neural_block_f32, neural_record_fields, neural_string_keys,
    neural_plain_value, neural_array_snapshot, neural_array_slots, neural_admit_weights,
    neural_native_scalar, neural_model_weights, neural_block_snapshot, neural_block_weight_names, neural_metadata_snapshot,
)
from bindings.mode_metadata_native import _fields
from bindings.native_loaded_module import native_loaded_module, NativeModules
from bindings.native_identity_key import NativeIdentityKey
from bindings.native_prepared_owner import prepared_get, prepared_put, prepared_drop, prepared_key_add, prepared_key_fields, prepared_key_same
from providers.python.borrowed_buffer_pyapi import admit_native_buffer_exporter
from std.memory import memcpy
from std.os import getenv
from bindings.generation_buffers import generation_initialize_binding, generation_array, py_ints
from bindings.causal_lm_construct import _model_block, _model_profile, _model_bindings, _model_f32, _model_shape, model_recurrent_weights_check
from bindings.causal_lm_public import causal_lm_admit_binding, causal_layer_allocate, causal_layer_run, causal_layer_allocate_fields, causal_layer_run_fields, _addr, _native_function, causal_transformer_admin, _native_has_entry, _native_entry_fields, _standalone_record_fields, causal_prepared_layer
from core.hostptr import f32_ptr
from core.host_helpers_impl import all_finite_f32
from core.native_descriptor_abi import TRANSFORMER_BLOCK_ABI_NAME, MAMBA_BLOCK_ABI_NAME, MAMBA_STATE_ABI_NAME
from core.native_mamba_workspace_abi import MAMBA2_WORKSPACE_CREATE, MAMBA2_WORKSPACE_CLOSE
from providers.python.operation_lock_pyapi import OperationLockLease, operation_lock_create
from providers.python.neural_descriptor_pyapi import (
    neural_descriptor_reuse, neural_descriptor_execute, neural_descriptor_reserved_execute,
    neural_descriptor_handle, neural_descriptor_adopt,
)



struct NativeBlockPlan(Movable, Writable):
    var context: PythonObject
    var entries: PythonObject
    var kind: String
    var d_model: Int
    var numeric_profile: PythonObject
    var packed: Bool
    var layer: PythonObject
    var state_scalars: PythonObject
    var state_buffers: PythonObject
    var state_fields: PythonObject
    var mamba2_workspace_provider: PythonObject
    var mamba2_workspace: PythonObject

    def __init__(out self, owner: PythonObject, layer: PythonObject) raises:
        self.context = owner._native_context
        self.entries = owner._native_entries
        self.kind = String(owner.kind)
        self.d_model = Int(py=owner.d_model)
        self.numeric_profile = owner.numeric_profile
        self.packed = owner._head_int15 is not None
        self.layer = layer
        self.mamba2_workspace_provider = py_get(self.entries, "mamba2_workspace_provider")
        self.mamba2_workspace = PythonObject(None)
        # Immutable schema belongs to the retained execution plan. Live class
        # hooks, scalar values and Array metadata are still admitted each call.
        self.state_scalars = Python.tuple("batch_size", "n_kv_heads", "head_dim", "max_tokens", "window", "cached_tokens",
            "arithmetic_profile", "_numeric_profile", "profile_id", "absolute_position", "_nn34_usable", "buffered_tokens")
        self.state_buffers = Python.tuple("k_cache", "v_cache", "conv_window", "h", "affine_boundary", "affine_a", "affine_b", "buffer_xbc", "buffer_dtraw")
        self.state_fields = self.state_scalars + self.state_buffers + Python.tuple("_resident_session")

    def write_to(self, mut writer: Some[Writer]):
        writer.write("NativeBlockPlan(native)")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


struct NativeBlockExecution(Movable):
    # Per-operation fields stay separate across GIL-release/reentrant calls.
    # Stable execution metadata lives in one retained native plan.
    var plan: PythonObject
    var fields: PythonObject

    def __init__(out self, plan: PythonObject, fields: PythonObject):
        self.plan = plan
        self.fields = fields


def _neural_default_mode(mut mods: NativeModules, context: PythonObject) raises -> PythonObject:
    var mode = _native_entry_fields(mods, context["backend"])["_DEFAULT_MODE"]
    if py_type(mode) is not mods.module("builtins").str:
        raise Error("TypeError: neural default numeric mode requires an exact native string")
    return mode


def _neural_block_owner_full(mut mods: NativeModules, block: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = mods.module("builtins")
    var fields = neural_block_snapshot(mods, block, context)
    var live = fields["_neural_live_fields"]
    var kind = String(fields["_native_kind"])
    var route = fields["_native_route"]
    var mode = py_get(fields, "numeric_mode")
    var bindings = _model_bindings(mods, route, kind, context, mode)
    if kind == "transformer" and _native_has_entry(mods, bindings[kind], "transformer_arithmetic_profile"):
        var next_profile = _native_function(mods, bindings[kind], "transformer_arithmetic_profile")()
        mods.released()
        var want_mode = _neural_default_mode(mods, context) if mode is None else mode
        var saved = py_get(fields, "_arithmetic_profile")
        if saved is not None and Bool(py=py_get(fields, "_arithmetic_profile_mode") == want_mode) and not Bool(py=saved == next_profile):
            raise Error("neural block: saved arithmetic profile differs from binding")
        py_setitem(fields, "_arithmetic_profile", next_profile)
        py_setitem(fields, "_arithmetic_profile_mode", want_mode)
        py_setitem(fields, "_arithmetic_profile_changed", PythonObject(Bool(py=_native_function(mods, bindings[kind], "transformer_arithmetic_profile_changed")())))
        mods.released()
    var owner = py_call(mods.module("types").SimpleNamespace)
    py_setattr(owner, "_native_block_fields", Python.list(fields))
    py_setattr(owner, "_native_block_live", live)
    py_setattr(owner, "_native_context", context)
    py_setattr(owner, "kind", PythonObject(kind))
    py_setattr(owner, "device", route)
    py_setattr(owner, "d_model", fields["d_model"])
    py_setattr(owner, "numeric_profile", py_get(fields, "numeric_profile", PythonObject("fp32_v1")))
    py_setattr(owner, "_head_int15", py_get(fields, "_int15"))
    py_setattr(owner, "_blocks", bi.list())
    _ = py_call(owner._blocks.append, block)
    _ = causal_lm_admit_binding(mods, owner, bindings[kind], bindings["primitives"], bindings["linalg"] if owner._head_int15 is not None else PythonObject(None))
    return owner


comptime _NEURAL_OWNER_SLOT = "neural_block_owner_v1"


def _neural_owner_runtime_names() -> List[String]:
    # Fields an operation itself writes back (sessions, provider caches,
    # arithmetic profile words, reports). They are not setup inputs: each
    # call copies their live values into its snapshot, as the full setup did.
    return [String("_arithmetic_profile"), String("_arithmetic_profile_mode"), String("_arithmetic_profile_changed"),
            String("_native_session"), String("_session_binding"), String("_native_forward_owner"), String("h_last_"),
            String("_native_block_provider_cache"), String("_native_mamba_block_provider_cache"), String("_native_mamba_state"),
            String("_native_session_provider_cache"), String("_native_primitives_provider_cache"),
            String("_native_mamba_provider_cache")]


def _neural_owner_env() -> String:
    # Environment switches read by causal_lm_admit_binding.
    return String(getenv("MOJOLEARN_GENERATION_NATIVE_SESSION")) + "|" + String(getenv("MOJOLEARN_NEURAL_NATIVE_DISPATCH"))


def _neural_owner_key(mut mods: NativeModules, block: PythonObject, live: PythonObject, context: PythonObject, mut binding_key: PythonObject, mut uses_default: Bool) raises -> PythonObject:
    var bi = mods.module("builtins")
    var key = NativeIdentityKey(mods)
    key.add_identity(py_type(block))
    key.add_identity(context)
    key.fields(live, _neural_owner_runtime_names())
    var route = py_get(live, "_native_route")
    var mode = py_get(live, "numeric_mode")
    var kind = py_get(live, "_native_kind")
    if py_type(route) is not bi.str or py_type(kind) is not bi.str or (mode is not None and py_type(mode) is not bi.str):
        return PythonObject(None)
    var mode_name = PythonObject("identical")
    uses_default = String(route) != "cpu" and mode is None
    if String(route) != "cpu":
        if uses_default:
            mode_name = _native_entry_fields(mods, context["backend"])["_DEFAULT_MODE"]
            key.add_identity(mode_name)
            if py_type(mode_name) is not bi.str:
                return PythonObject(None)
        else:
            mode_name = mode.strip().lower()
    var table = context["bindings"]
    if py_type(table) is not bi.dict:
        return PythonObject(None)
    key.add_identity(table)
    binding_key = route + PythonObject(":") + mode_name
    var entries = py_get(table, binding_key)
    if entries is None or py_type(entries) is not bi.dict:
        return PythonObject(None)
    key.add(entries)
    if String(kind) == "mamba2" and String(route) == "gpu":
        # A same-module capability replacement/addition/removal invalidates the
        # retained provider/resource too; module identity alone is insufficient.
        key.add_identity(py_get(_native_entry_fields(mods, entries["mamba2"]), "mamba2_workspace_api_v1"))
    key.watch_arrays(mods, context)
    if key.retains(block):
        return PythonObject(None)
    return PythonObject(alloc=key^)


def _neural_owner_matches(mut mods: NativeModules, key: PythonObject, block: PythonObject, live: PythonObject, context: PythonObject, binding_key: PythonObject, uses_default: Bool) raises -> Bool:
    ref native = key.downcast_value_ptr[NativeIdentityKey]()[]
    var offset = 0
    if not native.matches_identity(py_type(block), offset) or not native.matches_identity(context, offset) or not native.matches_fields(live, _neural_owner_runtime_names(), offset):
        return False
    if uses_default and not native.matches_identity(_native_entry_fields(mods, context["backend"])["_DEFAULT_MODE"], offset):
        return False
    var table = context["bindings"]
    if not native.matches_identity(table, offset):
        return False
    var entries = py_get(table, binding_key)
    if not native.matches(entries, offset):
        return False
    if String(live["_native_kind"]) == "mamba2" and String(live["_native_route"]) == "gpu":
        if not native.matches_identity(py_get(_native_entry_fields(mods, entries["mamba2"]), "mamba2_workspace_api_v1"), offset):
            return False
    return native.complete(offset) and native.arrays_match(mods, context)


def _neural_weights_retained(mut mods: NativeModules, snaps: PythonObject, originals: PythonObject, context: PythonObject) raises -> Bool:
    # True when every captured weight is a passive view of its original
    # Array (same storage), never a converted copy: in-place edits of the
    # original stay visible to a reused snapshot exactly as before.
    if len(snaps) != len(originals):
        return False
    for i in range(len(snaps)):  # driver-loop: over the block's weight slots, identity only
        var snap = snaps[i]
        var original = originals[i]
        if snap is None:
            if original is not None:
                return False
            continue
        if py_type(original) is not context["array_class"] or snap._base is not original:
            return False
    return True


def _neural_owner_reusable(mut mods: NativeModules, snapshot: PythonObject, live: PythonObject, context: PythonObject) raises -> Bool:
    for name in [String("_w"), String("_wopt")]:  # small-loop(weight slot lists)
        if py_contains(snapshot, name):
            if not _neural_weights_retained(mods, snapshot[name], snapshot[String("_neural_original") + name], context):
                return False
    var packed = py_get(live, "_int15")
    if packed is not None:
        var planes = snapshot["_int15"]
        for i in range(len(packed)):  # driver-loop: over packed weight planes, identity only
            if packed[i] is None or planes[i] is None:
                if packed[i] is not planes[i]:
                    return False
            elif not _neural_weights_retained(mods, planes[i], packed[i], context):
                return False
    return True


def neural_block_owner(mut mods: NativeModules, block: PythonObject, context: PythonObject) raises -> PythonObject:
    # The parallel worker keeps its existing owner ABI at one-time layer
    # construction. Repeated standalone execution uses the native plan below.
    return _neural_block_owner_full(mods, block, context)


def neural_block_execution(mut mods: NativeModules, block: PythonObject, context: PythonObject) raises -> NativeBlockExecution:
    var bi = mods.module("builtins")
    var live = neural_block_fields(mods, block, context)
    var env = _neural_owner_env()
    var record = prepared_get(mods, _NEURAL_OWNER_SLOT, block)
    var hit = False
    if record is not None and String(record[1]) == env:
        try:
            hit = _neural_owner_matches(mods, record[0], block, live, context, record[7], Bool(py=record[8]))
        except:
            hit = False
    if hit:
        # The remaining field dictionary is operation-local mutable state.
        # No identity list or SimpleNamespace owner is built on this path.
        var fields = py_call(bi.dict, live)
        _ = py_call(fields.update, record[3])
        py_setitem(fields, "_neural_live_fields", live)
        if Bool(py=record[4]):
            var mode = py_get(fields, "numeric_mode")
            var next_profile = record[5]
            var want_mode = _neural_default_mode(mods, context) if mode is None else mode
            var saved = py_get(fields, "_arithmetic_profile")
            if saved is not None and Bool(py=py_get(fields, "_arithmetic_profile_mode") == want_mode) and not Bool(py=saved == next_profile):
                raise Error("neural block: saved arithmetic profile differs from binding")
            py_setitem(fields, "_arithmetic_profile", next_profile)
            py_setitem(fields, "_arithmetic_profile_mode", want_mode)
            py_setitem(fields, "_arithmetic_profile_changed", record[6])
        return NativeBlockExecution(record[2], fields)
    var key = PythonObject(None)
    var binding_key = PythonObject(None)
    var uses_default = False
    try:
        key = _neural_owner_key(mods, block, live, context, binding_key, uses_default)
    except:
        key = PythonObject(None)
    var owner = _neural_block_owner_full(mods, block, context)
    var snapshot = owner._native_block_fields[0]
    var kind = String(snapshot["_native_kind"])
    var planes = kind == "transformer" and owner._head_int15 is not None and py_contains(owner._native_entries, "forward15")
    var layer = causal_prepared_layer(mods, snapshot, kind, planes)
    var plan = PythonObject(alloc=NativeBlockPlan(owner, layer))
    var reusable = key is not None and _neural_owner_reusable(mods, snapshot, live, context)
    if reusable:
        try:
            reusable = _neural_owner_matches(mods, key, block, live, context, binding_key, uses_default) and _neural_owner_env() == env
        except:
            reusable = False
    if not reusable:
        prepared_drop(mods, _NEURAL_OWNER_SLOT, block)
        return NativeBlockExecution(plan, snapshot)
    var derived = bi.dict()
    for name in [String("_neural_weight_names"), String("dt_limit"), String("_opts_tail"), String("rope_scaling"),  # small-loop(setup-derived snapshot words)
                 String("_neural_original_w"), String("_w"), String("_neural_original_wopt"), String("_wopt"), String("_int15")]:
        if py_contains(snapshot, name):
            py_setitem(derived, name, snapshot[name])
    var has_profile = kind == "transformer" and _native_has_entry(mods, context["bindings"][binding_key][kind], "transformer_arithmetic_profile")
    var next_profile = snapshot["_arithmetic_profile"] if has_profile else PythonObject(None)
    var changed = snapshot["_arithmetic_profile_changed"] if has_profile else PythonObject(False)
    prepared_put(mods, _NEURAL_OWNER_SLOT, block, Python.list(key, PythonObject(env), plan, derived, PythonObject(has_profile), next_profile, changed, binding_key, PythonObject(uses_default)))
    return NativeBlockExecution(plan, snapshot)


def neural_block_init_binding(mut mods: NativeModules, block: PythonObject, weights: PythonObject, kind_value: PythonObject, options: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = mods.module("builtins")
    _ = generation_initialize_binding(context["generation_classes"])
    var admitted_weights = neural_model_weights(mods, weights, context)
    neural_plain_value(mods, options, 0, context)
    var captured_options = neural_metadata_snapshot(mods, options)
    if py_type(kind_value) is not bi.str:
        raise Error("TypeError: neural block kind requires an exact string")
    var fields = neural_block_fields(mods, block, context, False)
    _ = neural_block_weight_names(mods, block)
    var kind = String(kind_value)
    ref cpy = Python().cpython()
    var cpu = cpy.lib.get_function[Int32]("PyType_IsSubtype")(py_type(block)._obj_ptr, context["block_classes"]["cpu"][kind]._obj_ptr) != 0
    var route = PythonObject("cpu" if cpu else "gpu")
    # The established GPU block class routes to native host arithmetic on a
    # CPU-only installation, just as the package binding selector does.
    if context["selected_vendor"] is None or String(context["selected_vendor"]) == "cpu":
        route = PythonObject("cpu")
    var plan = py_call(mods.module("types").SimpleNamespace)
    py_setattr(plan, "kind", kind_value)
    var norm_name = "input_layernorm.weight" if kind == "transformer" else ("norm.weight" if kind == "mamba1" else "block_norm.weight")
    var norm = _model_f32(admitted_weights[norm_name], norm_name, context)
    if len(norm.shape) != 1 or Int(py=norm.shape[0]) < 1:
        raise Error("neural block: norm weight requires positive d_model")
    py_setattr(plan, "d_model", norm.shape[0])
    var mode = captured_options[1]
    py_setitem(fields, "numeric_mode", mode)
    py_setattr(plan, "block_options", bi.dict())
    py_setattr(plan, "block_kwargs", bi.dict())
    var profile = PythonObject("fp32_v1")
    if kind == "transformer":
        profile = _model_profile(mods, captured_options[2], "transformer_block", context)
        py_setattr(plan, "n_heads", captured_options[3])
        var names = context["block_option_names"]
        for i in range(len(names)):  # driver-loop: over transformer option keyword metadata
            py_setitem(plan.block_options, names[i], captured_options[i + 4])
    elif kind == "mamba2":
        py_setitem(plan.block_kwargs, "dt_limit", captured_options[2])
    var bindings = _model_bindings(mods, route, kind, context, mode)
    # Preflight the selected storage class before shared construction can read
    # its dictionary or weight-name registry (CPU fallback may select a peer
    # class rather than the caller's class).
    var storage_probe = py_call(bi.object.__new__, context["block_classes"][route][kind])
    _ = neural_block_fields(mods, storage_probe, context, False)
    _ = neural_block_weight_names(mods, storage_probe)
    var created = _model_block(mods, admitted_weights, plan, profile, route, bindings, context)
    _ = py_call(fields.update, neural_block_fields(mods, created, context))
    py_setitem(fields, "numeric_mode", mode)
    py_setitem(fields, "_native_kind", kind_value)
    py_setitem(fields, "_native_route", route)
    # Public parameter verification (2026-10-27): the constructor previously
    # retained only its ordered internal tensors, so inherited get_params
    # could not return the caller's weights and set_params could not rebuild.
    # Keep the original ordinary dictionary and its object identities.
    py_setitem(fields, "weights", weights)
    return PythonObject(None)


def neural_block_set_params_binding(mut mods: NativeModules, block: PythonObject, updates: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = mods.module("builtins")
    if py_type(updates) is not bi.dict:
        raise Error("TypeError: block parameters require an ordinary dictionary")
    if len(updates) == 0:
        return block
    var fields = neural_block_fields(mods, block, context)
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
    _ = neural_block_init_binding(mods, replacement, values["weights"], PythonObject(kind), options, context)
    # Preserve the established reset semantics and dictionary identity. A
    # rejected replacement leaves the live block's fields/session untouched.
    var fresh = neural_block_fields(mods, replacement, context)
    _ = py_call(fields.clear)
    _ = py_call(fields.update, fresh)
    return block


def _neural_allocate(mut mods: NativeModules, owner: NativeBlockExecution, b: Int, cap: Int) raises -> PythonObject:
    ref plan = owner.plan.downcast_value_ptr[NativeBlockPlan]()[]
    return causal_layer_allocate_fields(mods, plan.kind, owner.fields, plan.entries, plan.numeric_profile, b, cap)


def _neural_layer_run(mut mods: NativeModules, owner: NativeBlockExecution, x: PythonObject, state: PythonObject, step: Bool) raises -> PythonObject:
    ref plan = owner.plan.downcast_value_ptr[NativeBlockPlan]()[]
    var workspace = PythonObject(None)
    if step and plan.kind == "mamba2" and plan.mamba2_workspace_provider is not None:
        if plan.mamba2_workspace is None:
            # Creation is native metadata allocation only, under the GIL. Thus
            # concurrent calls cannot publish competing owners into this plan.
            # Adoption closes the handle if boxing fails. Device allocation and
            # live-byte refresh happen inside the actual public step operation.
            var handle = neural_descriptor_execute(plan.mamba2_workspace_provider,
                MAMBA2_WORKSPACE_CREATE, [], [], [], False)
            if handle == 0:
                raise Error("Mamba2 public workspace: native create returned a null owner")
            plan.mamba2_workspace = neural_descriptor_adopt(plan.mamba2_workspace_provider,
                handle, MAMBA2_WORKSPACE_CLOSE, [])
        workspace = plan.mamba2_workspace
    var result = causal_layer_run_fields(mods, owner.fields, plan.entries, plan.kind, plan.d_model,
        plan.packed, plan.layer, x, state, step, workspace)
    _ = workspace  # pin the native resource/image through the complete call
    _ = owner
    return result


def neural_block_allocate_binding(mut mods: NativeModules, block: PythonObject, batch: PythonObject, capacity: PythonObject, context: PythonObject) raises -> PythonObject:
    _ = generation_initialize_binding(context["generation_classes"])
    var bi = mods.module("builtins")
    if not neural_native_scalar(mods, batch, context, True) or (capacity is not None and not neural_native_scalar(mods, capacity, context, True)):
        raise Error("TypeError: neural block: batch_size and max_tokens require native integers")
    var b = Int(py=batch)
    var cap = 1 if capacity is None else Int(py=capacity)
    if b < 1 or cap < 1:
        raise Error("neural block: batch_size and max_tokens must be positive")
    var owner = neural_block_execution(mods, block, context)
    var state = _neural_allocate(mods, owner, b, cap)
    _neural_publish_block(owner)
    return state


def neural_block_allocate_binding(block: PythonObject, batch: PythonObject, capacity: PythonObject, context: PythonObject) raises -> PythonObject:
    # Classical and other callers keep their per-call module checks.
    var mods = NativeModules()
    return neural_block_allocate_binding(mods, block, batch, capacity, context)


def _neural_input(mut mods: NativeModules, value: PythonObject, d_model: Int, step: Bool, context: PythonObject) raises -> PythonObject:
    var x = neural_block_f32(mods, value, "x", context)
    var bi = mods.module("builtins")
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


def _neural_input(value: PythonObject, d_model: Int, step: Bool, context: PythonObject) raises -> PythonObject:
    # Classical and other callers keep their per-call module checks.
    var mods = NativeModules()
    return _neural_input(mods, value, d_model, step, context)


comptime _NEURAL_STATE_SLOT = "neural_state_capture_v1"


def _neural_state_capture(mut mods: NativeModules, owner: NativeBlockExecution, state: PythonObject) raises -> PythonObject:
    var bi = mods.module("builtins")
    ref plan = owner.plan.downcast_value_ptr[NativeBlockPlan]()[]
    var context = plan.context
    var kind = plan.kind
    var cls = context["generation_classes"][2][kind]
    if py_type(state) is not cls:
        raise Error("TypeError: neural state requires the exact connected state class")
    var scalars = plan.state_scalars
    var buffers = plan.state_buffers
    # All three schema groups have the same descriptor refusal. Admit their
    # union once, avoiding repeated dictionary/MRO capture while still checking
    # every live field before any scalar or buffer access.
    var fields = neural_record_fields(mods, state, plan.state_fields)
    var ancestry = py_call(bi.type.__dict__["__mro__"].__get__, cls)
    from bindings.mode_metadata_native import _class_fields
    for i in range(len(ancestry)):  # small-loop(passive state access protocol)
        var row = _class_fields(mods, ancestry[i])
        if py_contains(row, "__getattr__"):
            raise Error("TypeError: neural state attribute hooks are not supported")
        for name in ["__getattribute__", "__setattr__"]:  # driver-loop: native admission metadata and retained owner registry
            var hook = py_get(row, PythonObject(name))
            if hook is not None and hook is not bi.object.__dict__[name]:
                raise Error("TypeError: neural state attribute hooks are not supported")
    var copied = py_call(bi.dict, fields)
    for i in range(len(scalars)):  # small-loop(capture native state words before conversion)
        if py_contains(fields, scalars[i]):
            neural_plain_value(mods, fields[scalars[i]], 0, context)
    # One metadata stamp covers this state's buffers; class admission is
    # checked once per state, not once for every cache/window tensor.
    var prepared = prepared_get(mods, _NEURAL_STATE_SLOT, state)
    var valid = False
    if prepared is not None:
        try:
            ref key = prepared[1].downcast_value_ptr[NativeIdentityKey]()[]
            var offset = 0
            valid = True
            for i in range(len(buffers)):  # small-loop(buffers: native state buffer names): compare owner identities
                if not key.matches_identity(py_get(fields, buffers[i]), offset):
                    valid = False
                    break
            valid = valid and key.complete(offset) and key.arrays_match(mods, context)
        except:
            valid = False
    var views = prepared[0] if valid else bi.dict()
    var stamp = PythonObject(None)
    if not valid:
        try:
            var key = NativeIdentityKey(mods)
            for i in range(len(buffers)):  # small-loop(buffers: native state buffer names): capture owner identities
                key.add_identity(py_get(fields, buffers[i]))
            key.watch_arrays(mods, context)
            if not key.retains(state):
                stamp = PythonObject(alloc=key^)
        except:
            stamp = PythonObject(None)
    for i in range(len(buffers)):  # small-loop(buffers: native state buffer names): retain stable state storage
        var value = py_get(fields, buffers[i])
        if value is not None:
            if py_type(value) is context["array_class"]:
                var view = views[buffers[i]] if valid else neural_array_snapshot(mods, value, context)
                py_setitem(copied, buffers[i], view)
                if not valid:
                    py_setitem(views, buffers[i], view)
            else:
                admit_native_buffer_exporter(value)
    if not valid:
        var stable = False
        if stamp is not None:
            try:
                ref key = stamp.downcast_value_ptr[NativeIdentityKey]()[]
                var offset = 0
                stable = True
                for i in range(len(buffers)):  # small-loop(buffers: native state buffer names): recheck after capture
                    if not key.matches_identity(py_get(fields, buffers[i]), offset):
                        stable = False
                        break
                stable = stable and key.complete(offset) and key.arrays_match(mods, context)
            except:
                stable = False
        if stable:
            prepared_put(mods, _NEURAL_STATE_SLOT, state, Python.list(views, stamp))
        else:
            prepared_drop(mods, _NEURAL_STATE_SLOT, state)
    var snapshot = py_call(bi.object.__new__, cls)
    _ = py_call(_fields(mods, snapshot).update, copied)
    return snapshot


def _neural_publish_state(state: PythonObject, captured: PythonObject) raises:
    if state is None:
        return
    var fields = _fields(state)
    var source = _fields(captured)
    for name in ["cached_tokens", "arithmetic_profile", "_numeric_profile", "absolute_position", "_nn34_usable", "buffered_tokens"]:  # driver-loop: native admission metadata and retained owner registry
        if py_contains(source, name):
            py_setitem(fields, name, source[name])


def _neural_publish_fields(captured: PythonObject) raises:
    var fields = captured["_neural_live_fields"]
    for name in ["_arithmetic_profile", "_arithmetic_profile_mode", "_arithmetic_profile_changed",  # driver-loop: native admission metadata and retained owner registry
                 "_native_session", "_session_binding", "_native_forward_owner", "h_last_",
                 "_native_block_provider_cache", "_native_mamba_block_provider_cache", "_native_mamba_state"]:
        if py_contains(captured, name):
            py_setitem(fields, name, captured[name])


def _neural_publish_block(owner: NativeBlockExecution) raises:
    _neural_publish_fields(owner.fields)


def _neural_state(mut mods: NativeModules, owner: NativeBlockExecution, state: PythonObject, b: Int, l: Int) raises:
    var bi = mods.module("builtins")
    var fields = owner.fields
    if owner.plan.downcast_value_ptr[NativeBlockPlan]()[].kind == "transformer":
        if Int(py=state.batch_size) != b or Int(py=state.n_kv_heads) != Int(py=fields["n_kv_heads"]) or Int(py=state.head_dim) != Int(py=fields["head_dim"]) or Int(py=state.window) != Int(py=fields["window"]):
            raise Error("neural block: transformer state dimensions differ")
        var past = Int(py=state.cached_tokens)
        if past < 0 or past + l > Int(py=state.max_tokens) or past + l > Int(py=fields["max_positions"]):
            raise Error("neural block: sequence exceeds state capacity or max_positions")



def _neural_plain(mut mods: NativeModules, owner: NativeBlockExecution, x: PythonObject, supplied_state: PythonObject, step: Bool) raises -> PythonObject:
    var bi = mods.module("builtins")
    var fields = owner.fields
    var b = Int(py=x.shape[0])
    var l = Int(py=x.shape[1])
    var d = Int(py=x.shape[2])
    var transformer = owner.plan.downcast_value_ptr[NativeBlockPlan]()[].kind == "transformer"
    # Admission refusal must not poison a valid state owned by a decode
    # session. Only a failure after entering execution can invalidate it.
    if supplied_state is not None and py_get(_fields(mods, supplied_state), "_resident_session") is not None:
        raise Error("neural block: state is owned by a resident decode session")
    var e = owner.plan.downcast_value_ptr[NativeBlockPlan]()[].entries
    var reuse = transformer and py_contains(e, "forward_session") and String(getenv("MOJOLEARN_TRANSFORMER_LEGACY_SETUP")) != "1"
    var lock = py_get(fields, "_runtime_lock") if transformer else PythonObject(None)
    var lock_lease = OperationLockLease(lock)
    var locked = transformer and (reuse or fields["_native_session"] is not None or fields["_int15"] is not None)
    if locked:
        lock_lease.acquire()
        mods.released()
    try:
        if transformer and fields["_native_session"] is not None and (not reuse or fields["_session_binding"] is not e["block_binding"]):
            causal_transformer_admin(mods, fields["_session_binding"], fields["_native_session"], 13)
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
                        mods.released()
                        py_setitem(fields, "_session_binding", e["block_binding"])
                    if native is not None:
                        var session_owner = fields["_native_session"]
                        var handle = Int(py=py_call(e["transformer_native_session_handle_v1"], session_owner))
                        mods.released()
                        _ = neural_descriptor_reserved_execute(native, session_owner, handle, 0, 1, 2, addresses, params, [])
                        mods.released()
                    else:
                        _ = py_call(e["transformer_session_forward_fresh"], fields["_native_session"], py_ints(addresses), py_ints(params))
                        mods.released()
                elif native is not None:
                    _ = neural_descriptor_execute(native, 4, addresses, params, [])
                    mods.released()
                else:
                    _ = py_call(e["transformer_forward_fresh"], py_ints(addresses), py_ints(params))
                    mods.released()
                if locked:
                    lock_lease.release()
                _ = keep^
                return y
        var state = supplied_state
        if state is None:
            state = _neural_allocate(mods, owner, b, l)
        _neural_state(mods, owner, state, b, l)
        var result = _neural_layer_run(mods, owner, x, state, step)
        if locked:
            lock_lease.release()
        return result
    except error:
        mods.released()
        if locked:
            lock_lease.release()
        if supplied_state is not None and owner.plan.downcast_value_ptr[NativeBlockPlan]()[].kind == "mamba1":
            py_setattr(supplied_state, "_nn34_usable", PythonObject(False))
        raise error


def _neural_lengths(mut mods: NativeModules, value: PythonObject, b: Int, l: Int, context: PythonObject) raises -> List[Int]:
    var bi = mods.module("builtins")
    var values = value
    if py_type(value) is context["array_class"]:
        var captured = neural_array_snapshot(mods, value, context)
        if len(captured.shape) != 1:
            raise Error("neural block: lengths must be one-dimensional")
        if Int(py=captured.shape[0]) != b:
            raise Error("neural block: one length is required per batch row")
        var code = String(captured.dtype)
        if code != "<i4" and code != "<i1" and code != "<u2":
            raise Error("TypeError: neural block lengths entries must be integers, not bool or float")
        var result = List[Int]()
        var address = Int(py=captured._addr)
        for i in range(b):  # small-loop(native integer length words from retained storage)
            var n = Int(Pointer[Int32, ImmUntrackedOrigin](unsafe_from_address=address)[i]) if code == "<i4" else (
                Int(Pointer[Int8, ImmUntrackedOrigin](unsafe_from_address=address)[i]) if code == "<i1" else
                Int(Pointer[UInt16, ImmUntrackedOrigin](unsafe_from_address=address)[i]))
            if n < 1 or n > l:
                raise Error("neural block: lengths must be in [1, L]")
            result.append(n)
        _ = captured^
        return result^
    elif py_type(value) is not bi.list and py_type(value) is not bi.tuple:
        admit_native_buffer_exporter(value)
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


def _neural_lengths(value: PythonObject, b: Int, l: Int, context: PythonObject) raises -> List[Int]:
    # Classical and other callers keep their per-call module checks.
    var mods = NativeModules()
    return _neural_lengths(mods, value, b, l, context)


def neural_block_forward_binding(mut mods: NativeModules, block: PythonObject, input: PythonObject, state_value: PythonObject, step_value: PythonObject, lengths: PythonObject, context: PythonObject) raises -> PythonObject:
    _ = generation_initialize_binding(context["generation_classes"])
    var bi = mods.module("builtins")
    var owner = neural_block_execution(mods, block, context)
    var d = owner.plan.downcast_value_ptr[NativeBlockPlan]()[].d_model
    if py_type(step_value) is not bi.bool:
        raise Error("TypeError: neural step flag requires a native Boolean")
    var step = Bool(py=step_value)
    if step and state_value is None:
        raise Error("neural block: state is required for step (a decode step continues a sequence; allocate_state(B) makes the fresh one)")
    var x = _neural_input(mods, input, d, step, context)
    var b = Int(py=x.shape[0])
    var l = Int(py=x.shape[1])
    if lengths is not None:
        if state_value is not None:
            raise Error("neural block: lengths= (a ragged, right-padded batch) cannot carry a state; run the ragged prefill with state=None, or continue each sequence as its own batch")
        var sizes = _neural_lengths(mods, lengths, b, l, context)
        var padded = generation_array([b, l, d])
        neural_copy_row_prefixes(Int(py=padded._addr), Int(py=x._addr), sizes, b, l, d)
        # `x` may be the only reference to a fresh Array converted from a NumPy
        # input; Mojo destroys a local at its last named use, which was the
        # `_addr` read above, so without this the copy read freed memory and
        # row 0's first floats came back as allocator bookkeeping.
        _ = x^
        var output = _neural_plain(mods, owner, padded, PythonObject(None), False)
        var result = generation_array([b, l, d])
        neural_copy_row_prefixes(Int(py=result._addr), Int(py=output._addr), sizes, b, l, d)
        _ = output^  # the copy reads `output` by address
        _ = padded^
        _neural_publish_block(owner)
        return result
    var captured = PythonObject(None) if state_value is None else _neural_state_capture(mods, owner, state_value)
    try:
        var result = _neural_plain(mods, owner, x, captured, step)
        _neural_publish_state(state_value, captured)
        _neural_publish_block(owner)
        return result
    except error:
        mods.released()
        _neural_publish_state(state_value, captured)
        _neural_publish_block(owner)
        raise error^


def neural_block_forward_binding(block: PythonObject, input: PythonObject, state_value: PythonObject, step_value: PythonObject, lengths: PythonObject, context: PythonObject) raises -> PythonObject:
    # Classical and other callers keep their per-call module checks.
    var mods = NativeModules()
    return neural_block_forward_binding(mods, block, input, state_value, step_value, lengths, context)


def neural_training_entries(mut mods: NativeModules, fields: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = mods.module("builtins")
    var mode = py_get(fields, "numeric_mode")
    neural_plain_value(mods, mode, 0, context)
    if mode is None:
        mode = _neural_default_mode(mods, context)
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
    var vendor = String(_native_function(mods, entries, family + "_vendor")())
    mods.released()
    var code = Int(py=_native_function(mods, entries, family + "_numeric_mode")())
    mods.released()
    if code != (1 if String(mode) == "identical" else 0):
        raise Error("RuntimeError: native training binding numeric mode differs from the block")
    if String(fields["_native_route"]) == "cpu":
        if vendor != "cpu":
            raise Error("RuntimeError: CPU backward requires native CPU kernels")
    elif vendor != "metal" and vendor != "cuda" and vendor != "hip":
        raise Error("RuntimeError: GPU backward requires native GPU kernels")
    if family == "transformer" and _native_has_entry(mods, entries, "transformer_arithmetic_profile"):
        var profile = _native_function(mods, entries, "transformer_arithmetic_profile")()
        mods.released()
        var saved = py_get(fields, "_arithmetic_profile")
        if saved is not None and Bool(py=py_get(fields, "_arithmetic_profile_mode") == mode) and not Bool(py=saved == profile):
            raise Error("RuntimeError: saved transformer arithmetic profile differs from training binding")
        py_setitem(fields, "_arithmetic_profile", profile)
        py_setitem(fields, "_arithmetic_profile_mode", mode)
        py_setitem(fields, "_arithmetic_profile_changed", PythonObject(Bool(py=_native_function(mods, entries, "transformer_arithmetic_profile_changed")())))
        mods.released()
    if family == "transformer":
        var native = PythonObject(None)
        if code == 1 and getenv("MOJOLEARN_NEURAL_NATIVE_DISPATCH", "0") == "1":
            native = neural_descriptor_reuse(py_get(fields, "_native_block_provider_cache"),
                _native_function(mods, entries, "transformer_block_api_v1"), String(TRANSFORMER_BLOCK_ABI_NAME), 36)
            mods.released()
        py_setitem(fields, "_native_block_provider_cache", native)
    else:
        var native = PythonObject(None)
        if code == 1 and getenv("MOJOLEARN_NEURAL_NATIVE_DISPATCH", "0") == "1":
            native = neural_descriptor_reuse(py_get(fields, "_native_mamba_block_provider_cache"),
                _native_function(mods, entries, "mamba_block_api_v1"), String(MAMBA_BLOCK_ABI_NAME), 44)
            mods.released()
        py_setitem(fields, "_native_mamba_block_provider_cache", native)
        var state_provider = PythonObject(None)
        if native is not None:
            state_provider = neural_descriptor_reuse(py_get(fields, "_native_mamba_state"),
                _native_function(mods, entries, "mamba_state_api_v1"), String(MAMBA_STATE_ABI_NAME), 6)
            mods.released()
        py_setitem(fields, "_native_mamba_state", state_provider)
    return entries


def neural_training_entries(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    # Classical and other callers keep their per-call module checks.
    var mods = NativeModules()
    return neural_training_entries(mods, fields, context)


def _neural_int_words(values: PythonObject) raises -> List[Int]:
    # Internal boundary vectors already contain admitted scalar/address words.
    var result = List[Int]()
    for i in range(len(values)):  # small-loop(public operation descriptor words)
        result.append(Int(py=values[i]))
    return result^


def neural_mamba_session_call(mut mods: NativeModules, native: PythonObject, fields: PythonObject, entries: PythonObject,
                              operation: Int, addresses: List[Int], params: List[Int]) raises -> Int:
    # Operation admission only. The common provider pins this exact typed
    # session and compiled image and reserves/releases under the GIL.
    var session = fields["_prefill_session"]
    var handle = neural_descriptor_handle(native,
        _native_function(mods, entries, "mamba_prefill_native_handle_v1"), session)
    mods.released()
    var windowed_result9 = neural_descriptor_reserved_execute(native, session, handle, 0, 1, operation,
        addresses, params, [])
    mods.released()
    return windowed_result9^


def neural_mamba_session_call(native: PythonObject, fields: PythonObject, entries: PythonObject,
                              operation: Int, addresses: List[Int], params: List[Int]) raises -> Int:
    # Classical and other callers keep their per-call module checks.
    var mods = NativeModules()
    return neural_mamba_session_call(mods, native, fields, entries, operation, addresses, params)


def neural_mamba3_restore(mut mods: NativeModules, fields: PythonObject, entries: PythonObject, weights: PythonObject) raises:
    var bi = mods.module("builtins")
    if Bool(py=py_get(fields, "_owned_restore_pending", PythonObject(False))):
        if not _native_has_entry(mods, entries, "mamba3_owned_weights_enabled") or not Bool(py=_native_function(mods, entries, "mamba3_owned_weights_enabled")()):
            mods.released()
            raise Error("RuntimeError: Mamba3 owned weights require the compiled owned-weight entry")
        mods.released()
        if py_get(fields, "_prefill_session") is None or py_get(fields, "_prefill_binding") is not entries.canonical:
            py_setitem(fields, "_prefill_session", _native_function(mods, entries, "mamba3_prefill_session_create")())
            mods.released()
            py_setitem(fields, "_prefill_binding", entries.canonical)
        var addresses = bi.list()
        for i in range(len(weights)):  # driver-loop: over the fixed owned-weight registry
            _ = py_call(addresses.append, weights[i]._addr)
        var native = py_get(fields, "_native_mamba_block_provider_cache")
        if native is not None:
            var generation = neural_mamba_session_call(mods, native, fields, entries, 15,
                _neural_int_words(addresses), [Int(py=fields["d_model"])])
            py_setitem(fields, "_owned_generation", PythonObject(generation))
        else:
            py_setitem(fields, "_owned_generation", py_call(_native_function(mods, entries, "mamba3_prefill_session_install_weights"), fields["_prefill_session"], addresses, fields["d_model"]))
            mods.released()
        py_setitem(fields, "_owned_native_provider", native)
        py_setitem(fields, "_owned_restore_pending", False)
    if py_get(fields, "_owned_generation") is not None:
        if py_get(fields, "_prefill_binding") is not entries.canonical or getenv("MOJOLEARN_MAMBA3_LEGACY_SETUP", "0") == "1":
            raise Error("RuntimeError: Mamba3 owned snapshot requires its installed binding/setup")


def neural_mamba3_restore(fields: PythonObject, entries: PythonObject, weights: PythonObject) raises:
    # Classical and other callers keep their per-call module checks.
    var mods = NativeModules()
    neural_mamba3_restore(mods, fields, entries, weights)


comptime _NEURAL_SNAPSHOT_SLOT = "neural_block_snapshot_v1"


def _neural_derived_words(mut mods: NativeModules, snapshot: PythonObject) raises -> PythonObject:
    # The snapshot words neural_block_snapshot computes from the stored
    # fields (weight views, metadata copies, weight names).
    var derived = mods.module("builtins").dict()
    for name in [String("_neural_weight_names"), String("dt_limit"), String("_opts_tail"), String("rope_scaling"),  # small-loop(setup-derived snapshot words)
                 String("_neural_original_w"), String("_w"), String("_neural_original_wopt"), String("_wopt"), String("_int15")]:
        if py_contains(snapshot, name):
            py_setitem(derived, name, snapshot[name])
    return derived


def _neural_snapshot_names() -> List[String]:
    return [String("_w"), String("_wopt"), String("_int15"), String("dt_limit"), String("_opts_tail"), String("rope_scaling")]


def _neural_snapshot_matches(mut mods: NativeModules, key: PythonObject, block: PythonObject, live: PythonObject, context: PythonObject) raises -> Bool:
    ref native = key.downcast_value_ptr[NativeIdentityKey]()[]
    var offset = 0
    if not native.matches_identity(py_type(block), offset) or not native.matches_identity(context, offset):
        return False
    for name in _neural_snapshot_names():  # small-loop(snapshot input field names)
        if not native.matches_identity(PythonObject(py_contains(live, name)), offset) or not native.matches(py_get(live, name), offset, 1):
            return False
    return native.complete(offset) and native.arrays_match(mods, context)


def neural_block_snapshot_prepared(mut mods: NativeModules, block: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = mods.module("builtins")
    var live = neural_block_fields(mods, block, context)
    var record = prepared_get(mods, _NEURAL_SNAPSHOT_SLOT, block)
    var hit = False
    if record is not None:
        try:
            hit = _neural_snapshot_matches(mods, record[0], block, live, context)
        except:
            hit = False
    if hit:
        var fields = py_call(bi.dict, live)
        _ = py_call(fields.update, record[1])
        py_setitem(fields, "_neural_live_fields", live)
        py_setitem(fields, "_neural_weight_names", neural_block_weight_names(mods, block))
        return fields
    var key = PythonObject(None)
    try:
        var native = NativeIdentityKey(mods)
        native.add_identity(py_type(block))
        native.add_identity(context)
        for name in _neural_snapshot_names():  # small-loop(snapshot input field names)
            native.add_identity(PythonObject(py_contains(live, name)))
            native.add(py_get(live, name), 1)
        native.watch_arrays(mods, context)
        if not native.retains(block):
            key = PythonObject(alloc=native^)
    except:
        key = PythonObject(None)
    var snapshot = neural_block_snapshot(mods, block, context)
    var reusable = key is not None and _neural_owner_reusable(mods, snapshot, live, context)
    if reusable:
        try:
            reusable = _neural_snapshot_matches(mods, key, block, live, context)
        except:
            reusable = False
    if reusable:
        prepared_put(mods, _NEURAL_SNAPSHOT_SLOT, block, Python.list(key, _neural_derived_words(mods, snapshot)))
    else:
        prepared_drop(mods, _NEURAL_SNAPSHOT_SLOT, block)
    return snapshot


def neural_block_backward_binding(mut mods: NativeModules, block: PythonObject, input: PythonObject, grad_output: PythonObject, context: PythonObject) raises -> PythonObject:
    _ = generation_initialize_binding(context["generation_classes"])
    var fields = neural_block_snapshot_prepared(mods, block, context)
    try:
        var result = _neural_backward_prepared(mods, block, input, grad_output, fields, context)
        _neural_publish_fields(fields)
        return result
    except error:
        mods.released()
        _neural_publish_fields(fields)
        raise error^


def neural_block_backward_binding(block: PythonObject, input: PythonObject, grad_output: PythonObject, context: PythonObject) raises -> PythonObject:
    # Classical and other callers keep their per-call module checks.
    var mods = NativeModules()
    return neural_block_backward_binding(mods, block, input, grad_output, context)


def _neural_backward_prepared(mut mods: NativeModules, block: PythonObject, input: PythonObject, grad_output: PythonObject, fields: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = mods.module("builtins")
    var kind = String(fields["_native_kind"])
    var transformer = kind == "transformer"
    if transformer and (fields["_int15"] is not None or Bool(py=fields["_extended"])):
        var named = String("")
        for key in ["norm", "mlp", "qkv_bias", "o_bias", "norm_bias", "mlp_bias", "qk_norm", "rope_scaling"]:  # driver-loop: over the named block options
            if py_contains(fields, key):
                named += ", " + key + "=" + String(py_call(bi.repr, fields[key]))
        raise Error("NotImplementedError: TransformerBlock.backward requires default float32 block options (the default block options only); numeric_profile="
                    + String(fields["numeric_profile"]) + named)
    var entries = neural_training_entries(mods, fields, context)
    var name = ("transformer" if transformer else kind) + "_backward"
    if not _native_has_entry(mods, entries, name):
        raise Error("RuntimeError: loaded native training binding lacks " + name)
    var d = Int(py=fields["d_model"])
    var x = _neural_input(mods, input, d, False, context)
    var b = Int(py=x.shape[0])
    var l = Int(py=x.shape[1])
    var dy = neural_block_f32(mods, grad_output, "grad_output", context)
    _model_shape(dy, [b, l, d], "grad_output")
    var weights = fields["_w"]
    if not transformer:
        var mode = py_get(fields, "numeric_mode")
        if mode is None:
            mode = _neural_default_mode(mods, context)
        else:
            mode = mode.strip().lower()
        var grad_pointer = f32_ptr(Int(py=dy._addr))
        var grad_count = Int(py=dy.size)
        var finite = 0
        with GILReleased(Python()):
            finite = all_finite_f32(grad_pointer, grad_count)
        mods.released()
        if finite != 1:
            raise Error("neural backward: grad_output must be finite")
        # Check the current weights and configuration directly: the same
        # admissions (connected binding, float32, key set, per-kind config
        # and shapes) the block constructor makes, without building a block.
        var plan = bi.dict()
        py_setitem(plan, "block_kwargs", bi.dict())
        if kind == "mamba2":
            py_setitem(plan["block_kwargs"], "dt_limit", fields["dt_limit"])
        var named = bi.dict()
        var names = fields["_neural_weight_names"]
        for i in range(len(names)):  # driver-loop: over the fixed block weight registry
            py_setitem(named, names[i], weights[i])
        var route = fields["_native_route"]
        weights = model_recurrent_weights_check(mods, named, kind, d, plan, route, _model_bindings(mods, route, kind, context, mode), context)
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
        mods.released()
        try:
            var session = _native_has_entry(mods, entries, "transformer_session_backward") and String(getenv("MOJOLEARN_TRANSFORMER_LEGACY_SETUP")) != "1"
            if session:
                if fields["_native_session"] is not None and fields["_session_binding"] is not entries.canonical:
                    causal_transformer_admin(mods, fields["_session_binding"], fields["_native_session"], 13)
                    py_setitem(fields, "_native_session", PythonObject(None))
                    py_setitem(fields, "_session_binding", PythonObject(None))
                if fields["_native_session"] is None:
                    py_setitem(fields, "_native_session", _native_function(mods, entries, "transformer_session_create")())
                    mods.released()
                    py_setitem(fields, "_session_binding", entries.canonical)
                var native = py_get(fields, "_native_block_provider_cache")
                if native is not None:
                    var session_owner = fields["_native_session"]
                    var handle = Int(py=py_call(_native_function(mods, entries, "transformer_native_session_handle_v1"), session_owner))
                    mods.released()
                    _ = neural_descriptor_reserved_execute(native, session_owner, handle, 0, 1, 3,
                        _neural_int_words(addresses), _neural_int_words(params), [])
                    mods.released()
                else:
                    _ = py_call(_native_function(mods, entries, "transformer_session_backward"), fields["_native_session"], addresses, params)
                    mods.released()
            else:
                if fields["_native_session"] is not None and _native_has_entry(mods, entries, "transformer_session_invalidate_tape"):
                    causal_transformer_admin(mods, entries, fields["_native_session"], 12)
                var native = py_get(fields, "_native_block_provider_cache")
                if native is not None:
                    _ = neural_descriptor_execute(native, 5, _neural_int_words(addresses), _neural_int_words(params), [])
                    mods.released()
                else:
                    _ = py_call(_native_function(mods, entries, name), addresses, params)
                    mods.released()
            lock_lease.release()
        except error:
            mods.released()
            lock_lease.release()
            raise error
    else:
        var native = py_get(fields, "_native_mamba_block_provider_cache")
        if kind == "mamba2":
            _ = py_call(params.extend, fields["dt_limit"])
        if kind == "mamba3":
            neural_mamba3_restore(mods, fields, entries, weights)
            var generation = py_get(fields, "_owned_generation")
            var session = _native_has_entry(mods, entries, "mamba3_prefill_session_backward") and getenv("MOJOLEARN_MAMBA3_LEGACY_SETUP", "0") != "1"
            if generation is not None:
                var owned_addresses = py_ints([Int(py=addresses[0])])
                for i in range(10, len(addresses)):  # driver-loop: over input-gradient and result address slots
                    _ = py_call(owned_addresses.append, addresses[i])
                _ = py_call(params.append, generation)
                if native is not None:
                    _ = neural_mamba_session_call(mods, native, fields, entries, 9,
                        _neural_int_words(owned_addresses), _neural_int_words(params))
                else:
                    _ = py_call(_native_function(mods, entries, "mamba3_prefill_session_backward_owned"), fields["_prefill_session"], owned_addresses, params)
                    mods.released()
            elif session:
                if py_get(fields, "_prefill_session") is None or py_get(fields, "_prefill_binding") is not entries.canonical:
                    py_setitem(fields, "_prefill_session", _native_function(mods, entries, "mamba3_prefill_session_create")())
                    mods.released()
                    py_setitem(fields, "_prefill_binding", entries.canonical)
                if native is not None:
                    _ = neural_mamba_session_call(mods, native, fields, entries, 7,
                        _neural_int_words(addresses), _neural_int_words(params))
                else:
                    _ = py_call(_native_function(mods, entries, "mamba3_prefill_session_backward"), fields["_prefill_session"], addresses, params)
                    mods.released()
            else:
                if native is not None:
                    _ = neural_descriptor_execute(native, 4, _neural_int_words(addresses), _neural_int_words(params), [])
                    mods.released()
                else:
                    _ = py_call(_native_function(mods, entries, name), addresses, params)
                    mods.released()
        else:
            if native is not None:
                var words: List[Int] = [Int(py=params[0]), Int(py=params[1]), Int(py=params[2])]
                var floats = List[Float64]()
                if kind == "mamba2":
                    floats.append(Float64(py=params[3]))
                    floats.append(Float64(py=params[4]))
                _ = neural_descriptor_execute(native, 3 if kind == "mamba2" else 2,
                    _neural_int_words(addresses), words, floats)
                mods.released()
            else:
                _ = py_call(_native_function(mods, entries, name), addresses, params)
                mods.released()
    var result = bi.dict()
    py_setitem(result, "x", gradients[0])
    var names = fields["_neural_weight_names"]
    for i in range(len(names)):  # driver-loop: over named gradient slots
        py_setitem(result, names[i], gradients[i + 1])
    _ = keep^
    return result



def neural_block_getstate_binding(mut mods: NativeModules, block: PythonObject, context: PythonObject) raises -> PythonObject:
    # Preserve arithmetic-profile admission before serialization. This native
    # admission uses the same binding contract as public forward, not the
    # Python _extension helper. Only ordinary model state crosses the boundary.
    var bi = mods.module("builtins")
    var fields = neural_block_fields(mods, block, context)
    var mode = py_get(fields, "numeric_mode")
    neural_plain_value(mods, mode, 0, context)
    if mode is None:
        mode = _native_entry_fields(mods, context["backend"])["_DEFAULT_MODE"]
    var bindings = _model_bindings(mods, fields["_native_route"], "transformer", context, mode)
    var entries = bindings["transformer"]
    var saved = py_get(fields, "_arithmetic_profile")
    var saved_mode = py_get(fields, "_arithmetic_profile_mode", PythonObject("identical") if saved is not None else PythonObject(None))
    if _native_has_entry(mods, entries, "transformer_numeric_mode"):
        var code = Int(py=_native_function(mods, entries, "transformer_numeric_mode")())
        mods.released()
        if code != (1 if String(mode) == "identical" else 0):
            raise Error("RuntimeError: transformer serialized model numeric mode differs from binding")
    if _native_has_entry(mods, entries, "transformer_arithmetic_profile"):
        var profile = _native_function(mods, entries, "transformer_arithmetic_profile")()
        mods.released()
        if saved is not None and Bool(py=saved_mode == mode) and not Bool(py=saved == profile):
            raise Error("transformer serialized model arithmetic profile differs from binding")
        py_setitem(fields, "_arithmetic_profile", profile)
        py_setitem(fields, "_arithmetic_profile_mode", mode)
        py_setitem(fields, "_arithmetic_profile_changed", PythonObject(Bool(py=_native_function(mods, entries, "transformer_arithmetic_profile_changed")())))
        mods.released()
    elif saved is not None:
        raise Error("transformer: a tagged model requires a binding that reports its arithmetic profile")
    var result = py_call(bi.dict, fields)
    var transient_names = [String("_runtime_lock"), String("_native_session"), String("_session_binding"),
        String("_native_forward_owner"), String("_native_block_provider_cache"),
        String("_native_session_provider_cache"), String("_native_primitives_provider_cache")]
    for name in transient_names:  # driver-loop: over operation owners excluded from serialized model state
        _ = py_call(result.pop, name, PythonObject(None))
    return result


def neural_block_setstate_binding(mut mods: NativeModules, block: PythonObject, state: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = mods.module("builtins")
    if py_type(state) is not bi.dict:
        raise Error("TypeError: neural block restore requires an exact state dictionary")
    neural_string_keys(mods, state)
    var fields = neural_block_fields(mods, block, context, False)
    var candidate = py_call(bi.object.__new__, py_type(block))
    var fresh = neural_block_fields(mods, candidate, context, False)
    _ = py_call(fresh.update, state)
    _ = neural_block_fields(mods, candidate, context)
    _ = py_call(fields.update, state)
    var mode = py_get(fields, "numeric_mode")
    neural_plain_value(mods, mode, 0, context)
    if mode is None:
        mode = _native_entry_fields(mods, context["backend"])["_DEFAULT_MODE"]
    if String(mode) == "identical" and getenv("MOJOLEARN_NEURAL_NATIVE_DISPATCH", "0") == "1":
        py_setitem(fields, "_runtime_lock", operation_lock_create())
        mods.released()
    else:
        py_setitem(fields, "_runtime_lock", context["rlock"]())
        mods.released()
    var session = PythonObject(None)
    var binding = PythonObject(None)
    if String(fields["_native_route"]) == "gpu" and String(mode) == "identical" and getenv("MOJOLEARN_NEURAL_NATIVE_DISPATCH", "0") == "1":
        var bindings = _model_bindings(mods, fields["_native_route"], "transformer", context, mode)
        binding = bindings["transformer"]
        session = _native_function(mods, binding, "transformer_session_create")()
        mods.released()
    py_setitem(fields, "_native_session", session)
    py_setitem(fields, "_session_binding", binding)
    py_setitem(fields, "_native_forward_owner", PythonObject(None))
    py_setitem(fields, "_native_block_provider_cache", PythonObject(None))
    return PythonObject(None)

def neural_block_c[operation: Int](py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    var mods = NativeModules()
    try:
        var args = PythonObject(from_borrowed=args_ptr)
        var result = PythonObject(None)
        comptime if operation == 0:
            result = neural_block_init_binding(mods, args[0], args[1], args[2], args[3], args[4])
        elif operation == 1:
            result = neural_block_allocate_binding(mods, args[0], args[1], args[2], args[3])
        elif operation == 2:
            result = neural_block_forward_binding(mods, args[0], args[1], args[2], args[3], args[4], args[5])
        elif operation == 4:
            result = neural_block_set_params_binding(mods, args[0], args[1], args[2])
        elif operation == 5:
            result = neural_block_getstate_binding(mods, args[0], args[1])
        elif operation == 6:
            result = neural_block_setstate_binding(mods, args[0], args[1], args[2])
        elif operation == 7:
            var bi = mods.module("builtins")
            var name = String(py_call(bi.type.__dict__["__name__"].__get__, py_type(args[0])))
            raise Error("NotImplementedError: mojolearn " + name + ".backward: inference is forward only; the backward pass runs on the GPU block class over a training binding")
        else:
            result = neural_block_backward_binding(mods, args[0], args[1], args[2], args[3])
        return result.steal_data()
    except error:
        if String(error).startswith("TypeError:"):
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        if String(error).startswith("NotImplementedError:"):
            return raise_python_exception(error_without_class(error, "NotImplementedError"), ExceptionType("PyExc_NotImplementedError"))
        if String(error).startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
