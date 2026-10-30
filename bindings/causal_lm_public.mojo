# SPDX-License-Identifier: Apache-2.0
"""Whole-model runtime owned by Mojo, over existing compiled kernel entries.

The selected CPU/GPU extensions are bound once at model construction. This
module never invokes model, block, Array or backend Python methods. State,
allocation, layer traversal, token traversal and selection are native. Calling
a compiled sibling extension through CPython's native function ABI changes no
arithmetic and uses no interpreter callback; targets are checked at admission.
"""
from providers.python.transformer_resident_pyapi import transformer_resident_decode_call, transformer_resident_words, transformer_resident_lm_open, transformer_resident_lm_close, transformer_resident_lm_run, transformer_resident_lm_read, transformer_resident_lm_sampling_read
from core.resident_sampling_read import ResidentSamplingRead
from core.resident_generation_read import ResidentGreedyRead, resident_greedy_publish, resident_greedy_outputs
from std.python import Python, PythonObject
from std.ffi import _Global
from std.python._cpython import PyObjectPtr, GILReleased
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_error_class import error_without_class
from bindings.native_loaded_module import native_loaded_module, NativeModules
from bindings.native_prepared_owner import prepared_namespace_admitted, prepared_namespace_admit
from bindings.py_balanced import py_call, py_contains, py_get, py_setattr, py_setitem, py_type
from bindings.py_typed_call import py_call_typed
from checks.numerics import NUMERIC_IDENTICAL
from core.native_descriptor_abi import TRANSFORMER_BLOCK_ABI_NAME, PRIMITIVE_ABI_NAME, MAMBA_STATE_ABI_NAME
from core.native_mamba_workspace_abi import MAMBA2_WORKSPACE_ABI_NAME, MAMBA2_WORKSPACE_OP_COUNT, MAMBA2_WORKSPACE_STEP
from providers.python.neural_descriptor_pyapi import neural_descriptor_reuse, neural_descriptor_execute, neural_descriptor_handle, neural_descriptor_reserved_execute, NativeDescriptorResource, neural_descriptor_resource_handle
from providers.python.generation_session_pyapi import (
    generation_reuse_native_provider, generation_connect_native_session,
    generation_native_session_matches, generation_native_forward,
)
from core.generation_metadata import generation_transformer_addresses
from transformer.native_generation_engine import NativeGenerationProgress, native_generation_read
from providers.python.generation_metadata_pyapi import (
    GenerationWeightSnapshot, GenerationLayerSnapshot, GenerationHeadSnapshot, GenerationOperationSnapshot, GenerationEngineAdmission, _generation_bool,
)
from std.memory import memcpy, memset_zero
from bindings.generation_buffers import (
    GENERATION_BOUNDARY, generation_array, generation_address, generation_copy, generation_mutable,
    generation_ids, generation_initialize_binding, generation_array_context, generation_array_snapshot, generation_buffer_view, py_ints,
)
from bindings.native_array_snapshot import native_array_slots, NativeArrayAdmission
from core.generation_policy_impl import GenerationSamplingScratch
from core.generation_kv_policy import GenerationReadControl
from core.hostptr import f32_ptr, i32_ptr
from core.sha256 import Sha256
from bindings.sha256_door import sha256_update, sha256_update_text
from std.math import isfinite
from std.os import getenv
from std.sys.compile import is_defined

from transformer.causal_lm_public_impl import (
    causal_lm_rows_done, causal_lm_join_resident, causal_lm_join_continuation,
    causal_lm_pick_step, causal_lm_tokens_row_major, causal_lm_drop_eos,
    causal_lm_utf8_cut, causal_lm_store_ids, causal_lm_flags_as_ints,
    causal_lm_restore_rows,
)


from bindings.tokenizer_public_native import tokenizer_vocab_encode, tokenizer_vocab_decode, tokenizer_vocab_fingerprint, tokenizer_decode_utf8
from bindings.mode_metadata_native import _fields as _native_record_fields, _truth
from bindings.neural_block_admission import neural_record_fields, neural_native_scalar, neural_plain_value, neural_string_keys, neural_block_weight_names, neural_metadata_snapshot, neural_block_fields
from bindings.causal_model_config import model_record_fields, model_plan_snapshot, model_dict, model_plain
from bindings.causal_operation_owner import causal_model_capture, causal_model_publish, causal_model_origin, causal_state_capture, causal_state_publish, causal_state_origin, causal_state_fields, causal_snapshot_fields, causal_model_cache_set, causal_block_cache_set, causal_checkpoint_capture, causal_pin_entries, causal_layer_capture, causal_layer_publish


struct CausalPublicConnection(Defaultable, Movable):
    var context: PythonObject

    def __init__(out self):
        self.context = PythonObject(None)


comptime CAUSAL_PUBLIC_CONNECTION = _Global[StorageType=CausalPublicConnection,
    name="MojoCausalPublicConnection", init_fn=CausalPublicConnection.__init__]


def causal_public_connect(context: PythonObject) raises:
    # Private import-connected metadata, retained natively for scalar admission.
    CAUSAL_PUBLIC_CONNECTION.get_or_create_ptr()[].context = context
    _ = generation_initialize_binding(context["generation_classes"])


def _causal_context() raises -> PythonObject:
    var context = CAUSAL_PUBLIC_CONNECTION.get_or_create_ptr()[].context
    if context is None:
        raise Error("RuntimeError: CausalLM requires its native construction context")
    return context


def _causal_context_for(mut mods: NativeModules, model: PythonObject) raises -> PythonObject:
    # Standalone neural blocks already supply an admitted private owner context.
    if py_type(model) is mods.module("types").SimpleNamespace:
        var fields = causal_snapshot_fields(mods, model)
        if not py_contains(fields, "_causal_original"):
            var context = py_get(fields, "_native_context")
            if context is not None:
                return context
    return _causal_context()


def _causal_state_fields(mut mods: NativeModules, state: PythonObject) raises -> PythonObject:
    return model_record_fields(mods, state, _causal_context()["generation_classes"][1],
        Python.tuple("batch_size", "max_tokens", "layers", "positions", "_owner", "_usable"))


def _causal_model_fields(mut mods: NativeModules, model: PythonObject) raises -> PythonObject:
    return model_record_fields(mods, model, _causal_context()["model_class"],
        Python.tuple("plan", "config", "kind", "d_model", "vocab_size", "n_layers", "norm_eps",
            "numeric_profile", "device", "weight_format", "max_positions", "_embed", "_norm", "_head",
            "_blocks", "_native_entries", "_native_vendor", "_native_numeric_mode", "_head_int15",
            "_closed", "_failed", "_runtime", "_parallel_entry", "_block_kwargs", "unused_names", "_block_class", "_prims", "_causal_entries_owner"))

comptime _LM_RESIDENT_GENERATE = is_defined["MOJOLEARN_IDN_LM_RESIDENT_GENERATE"]()
"""Experiment, default off (lane lm-resident-generate-20261028): a model
without a final norm (`final_norm=False`) is admitted to the resident
generate and forward routes (the session's head reads the stack output, as
the per-layer route's does; no float operation changes). Off, such models
take the per-layer route (`_run_snapshot`), as before."""


def _resident_norm_ok(model: PythonObject) raises -> Bool:
    """Whether the model's final norm lets it open a resident session."""
    comptime if _LM_RESIDENT_GENERATE:
        return True
    return model._norm is not None


def _standalone_record_fields(mut mods: NativeModules, model: PythonObject) raises -> PythonObject:
    # Standalone blocks use a private exact C namespace as their per-operation
    # owner. Public models still use the ordinary class dictionary admission.
    # SimpleNamespace.__dict__ is a native member descriptor, not the Python
    # class getset descriptor expected by the public-record helper.
    if py_type(model) is mods.module("types").SimpleNamespace:
        ref cpy = Python().cpython()
        var raw = cpy.lib.get_function[PyObjectPtr]("PyObject_GenericGetDict")(model._obj_ptr, PyObjectPtr())
        if not raw:
            raise cpy.unsafe_get_error()
        return PythonObject(from_owned=raw)
    return _native_record_fields(model)


def _causal_array_guard() raises:
    # Kernel/worker calls may release the GIL. Recheck descriptor storage
    # before consuming retained Array metadata afterwards; original numerical
    # metadata stays in its operation snapshot and is never read again.
    _ = native_array_slots(generation_array_context())


def _causal_array_guard(mut mods: NativeModules) raises:
    # The same descriptor re-check through the operation's admitted module
    # connections (re-checked after each GIL release window by the caller).
    _ = native_array_slots(mods, generation_array_context())


def _native_entry_fields(mut mods: NativeModules, entries: PythonObject) raises -> PythonObject:
    # Exact connected modules/namespaces use native storage, never getattr or
    # module __getattr__. Check keys before any string dictionary lookup.
    var types = mods.module("types")
    var bi = mods.module("builtins")
    ref cpy = Python().cpython()
    var fields = PythonObject(None)
    if py_type(entries) is types.ModuleType:
        var raw = cpy.lib.get_function[PyObjectPtr]("PyModule_GetDict")(entries._obj_ptr)
        if not raw:
            raise cpy.unsafe_get_error()
        fields = PythonObject(from_owned=cpy.Py_NewRef(raw))
        # A compiled binding module's namespace is admitted once: the key
        # scan below reruns only when the namespace dictionary or its size
        # changed since (an inserted key changes the size).
        if prepared_namespace_admitted(mods, entries, fields):
            return fields
        var module_keys = py_call(bi.list, fields.keys())
        for i in range(len(module_keys)):  # small-loop(module_keys: native binding record field names): admitted once per namespace
            if py_type(module_keys[i]) is not bi.str:
                raise Error("TypeError: native binding entries require ordinary string keys")
        prepared_namespace_admit(mods, entries, fields)
        return fields
    elif py_type(entries) is types.SimpleNamespace:
        var raw = cpy.lib.get_function[PyObjectPtr]("PyObject_GenericGetDict")(entries._obj_ptr, PyObjectPtr())
        if not raw:
            raise cpy.unsafe_get_error()
        fields = PythonObject(from_owned=raw)
    else:
        raise Error("TypeError: native binding entries require a connected module or namespace")
    var keys = py_call(bi.list, fields.keys())
    for i in range(len(keys)):  # small-loop(native binding record field names before keyed reads)
        if py_type(keys[i]) is not bi.str:
            raise Error("TypeError: native binding entries require ordinary string keys")
    return fields


def _native_has_entry(mut mods: NativeModules, entries: PythonObject, name: String) raises -> Bool:
    return py_contains(_native_entry_fields(mods, entries), name)


def _native_function(mut mods: NativeModules, module: PythonObject, name: String) raises -> PythonObject:
    var entry = py_get(_native_entry_fields(mods, module), name)
    var bi = mods.module("builtins")
    var types = mods.module("types")
    if py_type(entry) is not py_type(bi.len):
        raise Error("TypeError: CausalLM " + name + " must be a compiled native extension entry")
    if py_type(entry.__self__) is not types.ModuleType or String(entry.__name__) != name:
        raise Error("TypeError: CausalLM entry must retain its named compiled module function")
    if py_type(module) is types.ModuleType and entry.__self__ is not module:
        raise Error("TypeError: CausalLM entry belongs to another module")
    return entry


def _native_function(module: PythonObject, name: String) raises -> PythonObject:
    # Classical and other callers keep their per-call module checks.
    var mods = NativeModules()
    return _native_function(mods, module, name)


def causal_lm_admit_binding(mut mods: NativeModules,
    model: PythonObject, block_binding: PythonObject, primitives: PythonObject,
    int15_binding: PythonObject,
) raises -> PythonObject:
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    var entries = g[].bi.dict()
    var requested_gpu = String(model.device) == "gpu"
    py_setitem(entries, "cpu", PythonObject(not requested_gpu))
    var kind = String(model.kind)
    var block_prefix = ("transformer" if kind == "transformer" else "mamba") if requested_gpu else "neural_host"
    var primitive_prefix = "training" if requested_gpu else "neural_host"
    var vendor = String(_native_function(mods, block_binding, block_prefix + "_vendor")())
    mods.released()
    var primitive_vendor = String(_native_function(mods, primitives, primitive_prefix + "_vendor")())
    mods.released()
    var mode = Int(py=_native_function(mods, block_binding, block_prefix + "_numeric_mode")())
    mods.released()
    var primitive_mode = Int(py=_native_function(mods, primitives, primitive_prefix + "_numeric_mode")())
    mods.released()
    if requested_gpu:
        if vendor != "metal" and vendor != "cuda" and vendor != "hip":
            raise Error("CausalLM: requested GPU execution but native block reports " + vendor)
    elif vendor != "cpu":
        raise Error("CausalLM: requested CPU execution but native block reports " + vendor)
    if primitive_vendor != vendor or primitive_mode != mode:
        raise Error("CausalLM: native block and model primitives differ in vendor or numeric mode")
    if int15_binding is not None:
        var packed_vendor = String(_native_function(mods, int15_binding, "linalg_vendor")())
        mods.released()
        var packed_mode = Int(py=_native_function(mods, int15_binding, "linalg_numeric_mode")())
        mods.released()
        if packed_vendor != vendor or packed_mode != mode:
            raise Error("CausalLM: native packed arithmetic differs in vendor or numeric mode")
    py_setattr(model, "_native_vendor", PythonObject(vendor))
    py_setattr(model, "_native_numeric_mode", PythonObject(mode))
    var local_blocks = py_get(_standalone_record_fields(mods, model), "_native_block_fields")
    py_setitem(entries, "weight_names", local_blocks[0]["_neural_weight_names"] if local_blocks is not None else neural_block_weight_names(mods, model._blocks[0]))
    for name in [String("embedding_forward"), String("rms_norm_forward"), String("linear_forward")]:  # small-loop(primitive entry names)
        py_setitem(entries, name, _native_function(mods, primitives, name))
    if kind == "transformer":
        py_setitem(entries, "forward", _native_function(mods, block_binding, "transformer_forward"))
        py_setitem(entries, "decode", _native_function(mods, block_binding, "transformer_decode_step"))
        py_setitem(entries, "block_binding", block_binding)
        for name in [String("transformer_forward_fresh"), String("transformer_session_forward_fresh"), String("transformer_native_session_handle_v1")]:  # small-loop(optional entry names retained once at construction)
            if _native_has_entry(mods, block_binding, name):
                py_setitem(entries, name, _native_function(mods, block_binding, name))
        if _native_has_entry(mods, block_binding, "transformer_session_forward"):
            py_setitem(entries, "create", _native_function(mods, block_binding, "transformer_session_create"))
            py_setitem(entries, "forward_session", _native_function(mods, block_binding, "transformer_session_forward"))
            py_setitem(entries, "block_binding", block_binding)
        if model._head_int15 is not None:
            if _native_has_entry(mods, block_binding, "transformer_session_forward_int15"):
                py_setitem(entries, "create", _native_function(mods, block_binding, "transformer_session_create"))
                py_setitem(entries, "forward15", _native_function(mods, block_binding, "transformer_session_forward_int15"))
            else:
                py_setitem(entries, "forward15_host", _native_function(mods, block_binding, "transformer_forward_int15"))
            py_setitem(entries, "quantize15", _native_function(mods, int15_binding, "quantize_int15"))
            py_setitem(entries, "gemm15", _native_function(mods, int15_binding, "gemm_int15"))
        # Preserve the existing resident GPU route for fixed-length greedy
        # generation. No performance default changes with this migration.
        if _native_has_entry(mods, block_binding, "causal_lm_session_run"):
            py_setitem(entries, "resident_vendor", _native_function(mods, block_binding, "transformer_vendor")())
            mods.released()
            for name in [String("transformer_decode_session_create"), String("transformer_decode_session_open"),  # small-loop(resident entry names)
                         String("transformer_decode_session_close"), String("causal_lm_session_create"),
                         String("causal_lm_session_open"), String("causal_lm_session_run"), String("causal_lm_session_close")]:
                py_setitem(entries, name, _native_function(mods, block_binding, name))
            for name in [String("transformer_decode_session_load_state"), String("transformer_decode_session_export_state"),  # small-loop(optional resident entry names)
                         String("causal_lm_session_forward"), String("transformer_arithmetic_profile"),
                         String("transformer_arithmetic_profile_changed")]:
                if _native_has_entry(mods, block_binding, name):
                    py_setitem(entries, name, _native_function(mods, block_binding, name))
            if _native_has_entry(mods, block_binding, "causal_lm_resident_read_v1"):
                py_setitem(entries, "resident_read_v1", Int(py=_native_function(mods, block_binding, "causal_lm_resident_read_v1")()) == 1)
                mods.released()
            if _native_has_entry(mods, block_binding, "causal_lm_resident_sampling_v1"):
                py_setitem(entries, "resident_sampling_v1", Int(py=_native_function(mods, block_binding, "causal_lm_resident_sampling_v1")()) == 1)
                mods.released()
            if model._head_int15 is not None:
                py_setitem(entries, "transformer_decode_session_open_int15", _native_function(mods, block_binding, "transformer_decode_session_open_int15"))
                py_setitem(entries, "causal_lm_session_open_int15", _native_function(mods, block_binding, "causal_lm_session_open_int15"))
    elif kind == "mamba1":
        py_setitem(entries, "forward", _native_function(mods, block_binding, "mamba1_forward"))
        py_setitem(entries, "decode", _native_function(mods, block_binding, "mamba1_decode_step"))
        var profile = 1
        if _native_has_entry(mods, block_binding, "mamba1_profile"):
            profile = Int(py=_native_function(mods, block_binding, "mamba1_profile")())
            mods.released()
        py_setitem(entries, "mamba1_profile", PythonObject(profile))
    elif kind == "mamba2":
        py_setitem(entries, "forward", _native_function(mods, block_binding, "mamba2_forward"))
        py_setitem(entries, "decode", _native_function(mods, block_binding, "mamba2_decode_step"))
        if (requested_gpu and mode == NUMERIC_IDENTICAL and getenv("MOJOLEARN_NEURAL_NATIVE_DISPATCH", "1") == "1"
                and _native_has_entry(mods, block_binding, "mamba2_workspace_api_v1")):
            # Optional versioned public-step capability; older binding images
            # retain their existing public operation, never a private session.
            var workspace_provider = neural_descriptor_reuse(PythonObject(None),
                _native_function(mods, block_binding, "mamba2_workspace_api_v1"),
                String(MAMBA2_WORKSPACE_ABI_NAME), MAMBA2_WORKSPACE_OP_COUNT)
            mods.released()
            py_setitem(entries, "mamba2_workspace_provider", workspace_provider)
    else:
        raise Error("CausalLM: unsupported model kind " + kind)
    # Native ownership migration: resolve descriptor tables at admission by
    # default, so supported local operations enter the native execution engine.
    # Explicit MOJOLEARN_NEURAL_NATIVE_DISPATCH=0 retains the compatibility route.
    # This is an orchestration change, not a measured speedup/default tuning claim.
    if requested_gpu and kind == "transformer" and mode == NUMERIC_IDENTICAL and (String(getenv("MOJOLEARN_GENERATION_NATIVE_SESSION")) == "1" or String(getenv("MOJOLEARN_NEURAL_NATIVE_DISPATCH", "1")) == "1"):
        if not py_contains(entries, "forward_session"):
            raise Error("CausalLM: native session candidate requires transformer_session_forward")
        var owner_fields = neural_block_fields(mods, model._blocks[0], _causal_context_for(mods, model))
        var provider = generation_reuse_native_provider(py_get(owner_fields, "_native_session_provider_cache"), block_binding,
            _native_function(mods, block_binding, "transformer_native_api_v1"),
            _native_function(mods, block_binding, "transformer_native_session_handle_v1"))
        mods.released()
        py_setitem(owner_fields, "_native_session_provider_cache", provider)
        py_setitem(entries, "native_session_provider", provider)
    if mode == NUMERIC_IDENTICAL and String(getenv("MOJOLEARN_NEURAL_NATIVE_DISPATCH", "1")) == "1":
        var owner_fields = neural_block_fields(mods, model._blocks[0], _causal_context_for(mods, model))
        var primitive_provider = neural_descriptor_reuse(py_get(owner_fields, "_native_primitives_provider_cache"),
            _native_function(mods, primitives, "neural_primitives_api_v1"), String(PRIMITIVE_ABI_NAME), 29)
        mods.released()
        py_setitem(owner_fields, "_native_primitives_provider_cache", primitive_provider)
        py_setitem(entries, "native_primitives", primitive_provider)
        if kind == "transformer":
            var block_provider = neural_descriptor_reuse(py_get(owner_fields, "_native_block_provider_cache"),
                _native_function(mods, block_binding, "transformer_block_api_v1"), String(TRANSFORMER_BLOCK_ABI_NAME), 36)
            mods.released()
            py_setitem(owner_fields, "_native_block_provider_cache", block_provider)
            py_setitem(entries, "native_block_operations", block_provider)
        if kind == "mamba1" or kind == "mamba2":
            var mamba_provider = neural_descriptor_reuse(py_get(owner_fields, "_native_mamba_provider_cache"),
                _native_function(mods, block_binding, "mamba_state_api_v1"), String(MAMBA_STATE_ABI_NAME))
            mods.released()
            py_setitem(owner_fields, "_native_mamba_provider_cache", mamba_provider)
            py_setitem(entries, "native_mamba", mamba_provider)
    py_setattr(model, "_native_entries", entries)
    causal_pin_entries(mods, _standalone_record_fields(mods, model), entries)
    return PythonObject(None)


def _block(mut mods: NativeModules, model: PythonObject, index: Int) raises -> PythonObject:
    # PublicOutputsMixin.__getattribute__ is Python even for private metadata.
    # Access the object's plain dictionary with CPython's base implementation,
    # bypassing that overridden getter; every field below is stored metadata.
    # Standalone blocks provide one operation-local admission snapshot.
    var local = py_get(_standalone_record_fields(mods, model), "_native_block_fields")
    if local is not None:
        return local[index]
    var parallel = py_get(model._native_entries, "parallel_call")
    if parallel is not None:
        var windowed_result6 = py_call(parallel, "block_fields", Python.list(causal_model_origin(mods, model), index, model))
        mods.released()
        return windowed_result6^
    return neural_block_fields(mods, model._blocks[index], _causal_context_for(mods, model))


def _positions(model: PythonObject, state: PythonObject, length: Int) raises:
    var have = 0 if state is None else Int(py=state.positions)
    if length < 0 or have < 0:
        raise Error("CausalLM: negative sequence position")
    if model.max_positions is not None and have + length > Int(py=model.max_positions):
        raise Error("CausalLM: max_tokens or sequence exceeds max_positions")
    if state is not None and have + length > Int(py=state.max_tokens):
        raise Error("CausalLM: sequence exceeds allocated state capacity")


def causal_lm_parameters_binding(mut mods: NativeModules, model: PythonObject) raises -> PythonObject:
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    var result = g[].bi.dict()
    # The live arrays leave the model: a kept resident session set could miss
    # an in-place edit, so it is dropped and generation opens per call again.
    var model_fields = _standalone_record_fields(mods, model)
    causal_model_cache_set(mods, model, model_fields, "_parameters_exposed", True)
    causal_model_cache_set(mods, model, model_fields, "_resident_sessions", PythonObject(None))
    var p = model.plan
    var original_parameters = py_get(model_fields, "_causal_original_parameters", model_fields)
    py_setitem(result, p.embed_name, original_parameters["_embed"])
    if p.norm_name is not None:
        py_setitem(result, p.norm_name, original_parameters["_norm"])
    if p.head_name is not None:
        py_setitem(result, p.head_name, original_parameters["_head"])
    var weight_names = model._native_entries["weight_names"]
    # Same canonical optional slot order as the transformer native ABI.
    var optional_names = [String("q_proj.bias"), String("k_proj.bias"), String("v_proj.bias"),
                          String("o_proj.bias"), String("input_layernorm.bias"),
                          String("post_attention_layernorm.bias"), String("up_proj.bias"),
                          String("down_proj.bias"), String("gate_proj.bias"),
                          String("q_norm.weight"), String("k_norm.weight")]
    var keep = List[PythonObject]()
    for i in range(len(model._blocks)):  # driver-loop: over model layer metadata
        var names = g[].bi.dict()
        var slices = g[].bi.dict()
        for j in range(len(p._layer_map)):  # driver-loop: over checkpoint tensor mappings
            var entry = p._layer_map[j]
            py_setitem(names, entry[0], PythonObject(String(entry[1]).replace("{i}", String(i))))
            py_setitem(slices, entry[0], entry[2])
        var blk = _block(mods, model, i)
        var values = g[].bi.dict()
        var originals = g[].bi.dict()
        var original_w = py_get(blk, "_neural_original_w", blk["_w"])
        for j in range(min(len(weight_names), len(blk["_w"]))):  # driver-loop: over block weight slots
            if blk["_w"][j] is not None:
                py_setitem(values, weight_names[j], blk["_w"][j])
                py_setitem(originals, weight_names[j], original_w[j])
        if String(model.kind) == "transformer":
            var original_wopt = py_get(blk, "_neural_original_wopt", blk["_wopt"])
            for j in range(len(optional_names)):  # driver-loop: over optional block weight slots
                if blk["_wopt"][j] is not None:
                    py_setitem(values, optional_names[j], blk["_wopt"][j])
                    py_setitem(originals, optional_names[j], original_wopt[j])
        var keys = py_call(g[].bi.list, values)
        var fused = g[].bi.dict()
        for j in range(len(keys)):  # driver-loop: over checkpoint tensor names
            var key = keys[j]
            var name = py_get(names, key, PythonObject("layers." + String(i) + "." + String(key)))
            var rows = py_get(slices, key)
            if rows is None:
                py_setitem(result, name, originals[key])
            else:
                # A fused checkpoint tensor (Phi-3 QKV / gate-up) is split
                # for execution. Return every row under its checkpoint name.
                var piece = py_call(g[].bi.list, rows)
                _ = py_call(piece.append, values[key])
                _ = py_call(py_call(fused.setdefault, name, g[].bi.list()).append, piece)
        _causal_array_guard()
        var fused_names = py_call(g[].bi.list, fused)
        for j in range(len(fused_names)):  # driver-loop: over fused checkpoint tensor names
            var name = fused_names[j]
            var pieces = fused[name]
            var rows = 0
            var cols = Int(py=pieces[0][2].shape[1])
            for k in range(len(pieces)):  # driver-loop: over projection slice metadata
                rows = max(rows, Int(py=pieces[k][1]))
            var joined = generation_array([rows, cols])
            for k in range(len(pieces)):  # driver-loop: over contiguous projection buffers
                var begin = Int(py=pieces[k][0])
                var end = Int(py=pieces[k][1])
                if end > begin:
                    memcpy(dest=f32_ptr(Int(py=joined._addr)) + begin * cols, src=f32_ptr(_addr(pieces[k][2], keep)), count=(end - begin) * cols)
                    _ = joined  # keep the Python object alive through the native call above
            py_setitem(result, name, joined)
    _ = keep^
    return result


def causal_lm_parameters_binding(model: PythonObject) raises -> PythonObject:
    # Classical and other callers keep their per-call module checks.
    var mods = NativeModules()
    return causal_lm_parameters_binding(mods, model)


def causal_lm_blocks_binding(mut mods: NativeModules, model: PythonObject) raises -> PythonObject:
    var fields = _causal_model_fields(mods, model)
    var bi = mods.module("builtins")
    var blocks = fields["_blocks"]
    if py_type(blocks) is not bi.list and py_type(blocks) is not bi.tuple:
        raise Error("TypeError: CausalLM blocks require an ordinary owner sequence")
    return py_call(bi.tuple, blocks)


def causal_lm_blocks_binding(model: PythonObject) raises -> PythonObject:
    # Classical and other callers keep their per-call module checks.
    var mods = NativeModules()
    return causal_lm_blocks_binding(mods, model)


def causal_layer_allocate(mut mods: NativeModules, model: PythonObject, i: Int, b: Int, cap: Int) raises -> PythonObject:
    return causal_layer_allocate_fields(mods, String(model.kind), _block(mods, model, i), model._native_entries, model.numeric_profile, b, cap)


def causal_layer_allocate_fields(mut mods: NativeModules, kind: String, blk: PythonObject, entries: PythonObject, numeric_profile: PythonObject, b: Int, cap: Int) raises -> PythonObject:
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    var st = py_call(g[].bi.object.__new__, g[].layer_classes[kind])
    _ = model_record_fields(mods, st, g[].layer_classes[kind], Python.tuple("batch_size", "n_kv_heads", "head_dim", "max_tokens", "window", "k_cache", "v_cache", "cached_tokens", "_resident_session", "_numeric_profile", "conv_window", "h", "profile_id", "absolute_position", "_nn34_usable", "affine_boundary", "affine_a", "affine_b", "buffer_xbc", "buffer_dtraw", "buffered_tokens"))
    if kind == "transformer":
        var window = Int(py=blk["window"])
        var nkv = Int(py=blk["n_kv_heads"])
        var hd = Int(py=blk["head_dim"])
        var n = b * nkv * (cap if window == 0 else window) * hd
        py_setattr(st, "batch_size", PythonObject(b))
        py_setattr(st, "n_kv_heads", PythonObject(nkv))
        py_setattr(st, "head_dim", PythonObject(hd))
        py_setattr(st, "max_tokens", PythonObject(cap))
        py_setattr(st, "window", PythonObject(window))
        py_setattr(st, "k_cache", generation_array([n]))
        py_setattr(st, "v_cache", generation_array([n]))
        py_setattr(st, "cached_tokens", PythonObject(0))
        py_setattr(st, "_resident_session", PythonObject(None))
        py_setattr(st, "_numeric_profile", numeric_profile)
    elif kind == "mamba1":
        var di = Int(py=blk["d_inner"])
        var profile = Int(py=entries["mamba1_profile"])
        py_setattr(st, "conv_window", generation_array([b, di, 4]))
        py_setattr(st, "h", generation_array([b, di, 16]))
        py_setattr(st, "profile_id", PythonObject(profile))
        py_setattr(st, "absolute_position", PythonObject(0))
        py_setattr(st, "_nn34_usable", PythonObject(True))
        py_setattr(st, "affine_boundary", PythonObject(None))
        py_setattr(st, "affine_a", PythonObject(None))
        py_setattr(st, "affine_b", PythonObject(None))
        if profile == 2:
            py_setattr(st, "affine_boundary", generation_array([b, di, 16]))
            py_setattr(st, "affine_a", generation_array([b, di, 16, 6]))
            py_setattr(st, "affine_b", generation_array([b, di, 16, 6]))
    else:
        var cd = Int(py=blk["conv_dim"])
        var nh = Int(py=blk["nheads"])
        py_setattr(st, "conv_window", generation_array([b, cd, 4]))
        py_setattr(st, "h", generation_array([b, nh, 64, 128]))
        py_setattr(st, "buffer_xbc", generation_array([b, 256, cd]))
        py_setattr(st, "buffer_dtraw", generation_array([b, 256, nh]))
        py_setattr(st, "buffered_tokens", PythonObject(0))
    return st


def _public_int(mut mods: NativeModules, value: PythonObject, message: String) raises -> Int:
    # Exact connected native integer classes preserve NumPy integer support
    # without admitting arbitrary __index__ or conversion callbacks.
    if py_type(value) is mods.module("builtins").int:
        return Int(py=value)
    if not neural_native_scalar(mods, value, _causal_context(), True):
        raise Error("TypeError: " + message)
    return Int(py=value)


def causal_lm_allocate_binding(mut mods: NativeModules, model: PythonObject, batch: PythonObject, capacity: PythonObject) raises -> PythonObject:
    var b = _public_int(mods, batch, "CausalLM: batch_size and max_tokens must be integers")
    var cap = _public_int(mods, capacity, "CausalLM: batch_size and max_tokens must be integers")
    if b < 1 or cap < 1:
        raise Error("CausalLM: batch_size and max_tokens must be positive")
    _positions(model, PythonObject(None), cap)
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    var state = py_call(g[].bi.object.__new__, g[].state_cls)
    _ = model_record_fields(mods, state, g[].state_cls, Python.tuple("batch_size", "max_tokens", "positions", "_owner", "_usable", "layers"))
    py_setattr(state, "batch_size", PythonObject(b))
    py_setattr(state, "max_tokens", PythonObject(cap))
    py_setattr(state, "positions", PythonObject(0))
    py_setattr(state, "_owner", causal_model_origin(mods, model))
    py_setattr(state, "_usable", PythonObject(True))
    py_setattr(state, "layers", g[].bi.list())
    var kind = String(model.kind)
    var parallel = py_get(model._native_entries, "parallel_call")
    try:
        for i in range(len(model._blocks)):  # driver-loop: over model layer state owners
            _ = py_call(state.layers.append, causal_layer_allocate(mods, model, i, b, cap) if parallel is None else
                        py_call(parallel, "allocate_layer", Python.list(causal_model_origin(mods, model), i, b, cap, model)))
            mods.released()
    except error:
        mods.released()
        if parallel is not None:
            try:
                var captured = causal_state_capture(mods, state, model, _causal_context_for(mods, model))
                _ = py_call(parallel, "release_layers", Python.list(causal_model_origin(mods, model), state, model, captured))
                mods.released()
            except cleanup_error:
                raise Error("CausalLM: state allocation failed: " + String(error) + "; remote cleanup also failed: " + String(cleanup_error))
        raise error
    if py_type(model) is mods.module("types").SimpleNamespace and py_contains(causal_snapshot_fields(mods, model), "_causal_original"):
        return causal_state_capture(mods, state, model, _causal_context())
    return state


def causal_lm_allocate_binding(model: PythonObject, batch: PythonObject, capacity: PythonObject) raises -> PythonObject:
    # Classical and other callers keep their per-call module checks.
    var mods = NativeModules()
    return causal_lm_allocate_binding(mods, model, batch, capacity)


def _state(mut mods: NativeModules, model: PythonObject, state: PythonObject, b: Int, l: Int) raises:
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    var state_fields = causal_state_fields(mods, state, _causal_context_for(mods, model))
    if state_fields["_owner"] is not causal_model_origin(mods, model):
        raise Error("CausalLM: state belongs to another model")
    if not _truth(py_get(state_fields, "_usable", PythonObject(True))):
        raise Error("CausalLM: state lost after a failed call; reset or restore it")
    if Int(py=state.batch_size) != b or len(state.layers) != len(model._blocks):
        raise Error("CausalLM: state batch/layer counts disagree with model and input")
    _positions(model, state, l)


def causal_lm_reset_binding(mut mods: NativeModules, model: PythonObject, state: PythonObject) raises -> PythonObject:
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    var state_fields = causal_state_fields(mods, state, _causal_context_for(mods, model))
    if state_fields["_owner"] is not causal_model_origin(mods, model):
        raise Error("CausalLM: state belongs to another model")
    var fresh = causal_lm_allocate_binding(mods, model, state.batch_size, state.max_tokens)
    var parallel = py_get(model._native_entries, "parallel_call")
    if parallel is not None:
        try:
            _ = py_call(parallel, "release_layers", Python.list(causal_model_origin(mods, model), causal_state_origin(mods, state), model, state))
            mods.released()
        except error:
            mods.released()
            py_setattr(state, "_usable", PythonObject(False))
            try:
                _ = py_call(parallel, "release_layers", Python.list(causal_model_origin(mods, model), causal_state_origin(mods, fresh), model, fresh))
                mods.released()
            except cleanup_error:
                raise Error("CausalLM: state reset failed: " + String(error) + "; fresh-state cleanup also failed: " + String(cleanup_error))
            raise error
    py_setattr(state, "layers", fresh.layers)
    py_setattr(state, "positions", PythonObject(0))
    py_setattr(state, "_usable", PythonObject(True))
    return state


def causal_lm_reset_binding(model: PythonObject, state: PythonObject) raises -> PythonObject:
    # Classical and other callers keep their per-call module checks.
    var mods = NativeModules()
    return causal_lm_reset_binding(mods, model, state)


def _addr(value: PythonObject, mut keep: List[PythonObject]) raises -> Int:
    return generation_address(value, keep)


def _mutable(
    value: PythonObject, expected: List[Int], mut keep: List[PythonObject], mut copies: List[PythonObject],
    name: String = "",
) raises -> Int:
    return generation_mutable(value, expected, keep, copies, name)


def _writeback(mut copies: List[PythonObject]) raises:
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    for i in range(len(copies)):  # driver-loop: over copied state buffer handles
        var view = py_call(copies[i][0].cast, "B")
        var source = py_call(py_call(g[].bi.memoryview, copies[i][1]).cast, "B")
        py_setitem(view, py_call(g[].bi.slice, PythonObject(None)), source)
    copies.clear()


def _transformer_arg_words(
    model: PythonObject, blk: PythonObject, st: PythonObject, x: PythonObject, y: PythonObject,
    b: Int, l: Int, planes: Bool, mut keep: List[PythonObject],
) raises -> List[Int]:
    var addresses = List[Int]()
    addresses.append(_addr(x, keep))
    var weights = blk["_w"]
    for j in range(len(weights)):  # driver-loop: over block weight slots
        addresses.append(0 if planes and j >= 2 else _addr(weights[j], keep))
    addresses.append(_addr(st.k_cache, keep))
    addresses.append(_addr(st.v_cache, keep))
    addresses.append(_addr(y, keep))
    var optional = blk["_wopt"]
    for j in range(len(optional)):  # driver-loop: over optional block weight slots
        addresses.append(_addr(optional[j], keep))
    if planes:
        var packed = blk["_int15"]
        for j in range(len(packed)):  # driver-loop: over packed block weight slots
            var trip = packed[j]
            for k in range(3):  # small-loop(int15 plane triple)
                addresses.append(0 if trip is None else _addr(trip[k], keep))
    return addresses^


def _transformer_param_words(blk: PythonObject, st: PythonObject, b: Int, l: Int) raises -> List[Int]:
    var p: List[Int] = [b, l, Int(py=blk["d_model"]), Int(py=blk["n_heads"]), Int(py=blk["n_kv_heads"]),
                        Int(py=blk["head_dim"]), Int(py=blk["intermediate"]), Int(py=st.max_tokens),
                        Int(py=st.cached_tokens), Int(py=blk["window"])]
    var tail = blk["_opts_tail"]
    for j in range(len(tail)):  # driver-loop: read the block's admitted options tail
        p.append(Int(py=tail[j]))
    return p^


def _transformer_args(
    model: PythonObject, blk: PythonObject, st: PythonObject, x: PythonObject, y: PythonObject,
    b: Int, l: Int, planes: Bool, mut keep: List[PythonObject],
) raises -> PythonObject:
    return py_ints(_transformer_arg_words(model, blk, st, x, y, b, l, planes, keep))


def _transformer_params(blk: PythonObject, st: PythonObject, b: Int, l: Int) raises -> PythonObject:
    return py_ints(_transformer_param_words(blk, st, b, l))



def causal_transformer_admin(mut mods: NativeModules, binding: PythonObject, session: PythonObject, operation: Int) raises:
    # Boundary-only lifetime admission against the session's original image.
    # Administration intentionally holds the GIL, matching the raw public API.
    if String(getenv("MOJOLEARN_NEURAL_NATIVE_DISPATCH", "1")) == "1":
        var provider = neural_descriptor_reuse(PythonObject(None),
            _native_function(mods, binding, "transformer_block_api_v1"), String(TRANSFORMER_BLOCK_ABI_NAME), 36)
        mods.released()
        var handle = neural_descriptor_handle(provider,
            _native_function(mods, binding, "transformer_native_session_handle_v1"), session)
        mods.released()
        _ = neural_descriptor_execute(provider, operation, [], [handle], [], False)
        mods.released()
    else:
        var name = "transformer_session_invalidate_tape" if operation == 12 else "transformer_session_close"
        _ = py_call(_native_function(mods, binding, name), session)
        mods.released()

def causal_layer_run(mut mods: NativeModules, model: PythonObject, i: Int, x: PythonObject, supplied_state: PythonObject, step: Bool = False) raises -> PythonObject:
    # Standalone public block calls retain their existing ABI and admit anew.
    _causal_array_guard(mods)
    var shape = x.shape
    var b = Int(py=shape[0])
    var l = Int(py=shape[1])
    var d = Int(py=model.d_model)
    var kind = String(model.kind)
    var entries = model._native_entries
    var profile = kind == "transformer" and model._head_int15 is not None
    var planes = profile and py_contains(entries, "forward15")
    # A prepared standalone block owner carries the layer words and weight
    # views captured at its setup; reuse them and bind this call's snapshot.
    var prepared = PythonObject(None)
    if py_type(model) is mods.module("types").SimpleNamespace:
        prepared = py_get(_standalone_record_fields(mods, model), "_prepared_layer")
    if prepared is not None:
        var reused = prepared.downcast_value_ptr[PreparedGenerationLayer]()[].layer.copy()
        reused.rebind(_block(mods, model, i), kind)
        mods.released()
        return _causal_layer_run_prepared(mods, entries, kind, d, b, l, reused, profile, planes, x, supplied_state, step, PythonObject(None))
    var layer = GenerationLayerSnapshot(_block(mods, model, i), kind, planes)
    mods.released()
    return _causal_layer_run_prepared(mods, entries, kind, d, b, l, layer, profile, planes, x, supplied_state, step, PythonObject(None))


def causal_layer_run_fields(mut mods: NativeModules, fields: PythonObject, entries: PythonObject, kind: String, d: Int, packed: Bool, prepared: PythonObject, x: PythonObject, supplied_state: PythonObject, step: Bool = False, mamba_workspace: PythonObject = PythonObject(None)) raises -> PythonObject:
    # Standalone callers supply the retained native plan directly: no temporary
    # namespace or Python attribute dispatch to reconstruct execution metadata.
    _causal_array_guard(mods)
    var shape = x.shape
    var b = Int(py=shape[0])
    var l = Int(py=shape[1])
    var profile = kind == "transformer" and packed
    var planes = profile and py_contains(entries, "forward15")
    if prepared is not None:
        # Reuse the immutable native layer vectors directly. Only mutable
        # arithmetic-profile words and this operation's fields are rebound.
        model_dict(fields)
        var profile_id = PythonObject(None)
        var profile_changed = False
        if kind == "transformer":
            profile_id = model_plain(py_get(fields, "_arithmetic_profile", PythonObject(None)))
            profile_changed = _generation_bool(py_get(fields, "_arithmetic_profile_changed", PythonObject(False)))
        ref reused = prepared.downcast_value_ptr[PreparedGenerationLayer]()[].layer
        mods.released()
        return _causal_layer_run_bound(mods, entries, kind, d, b, l, reused, profile, planes, x, supplied_state, step, fields, profile_id, profile_changed, mamba_workspace)
    var layer = GenerationLayerSnapshot(fields, kind, planes)
    mods.released()
    return _causal_layer_run_prepared(mods, entries, kind, d, b, l, layer, profile, planes, x, supplied_state, step, mamba_workspace)


struct PreparedGenerationLayer(Movable, Writable):
    # Python-held owner of one standalone block's captured layer words and
    # weight views (registered by the tokenizer host binding).
    var layer: GenerationLayerSnapshot

    def __init__(out self, var layer: GenerationLayerSnapshot):
        self.layer = layer^

    def write_to(self, mut writer: Some[Writer]):
        writer.write("PreparedGenerationLayer(native)")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def causal_prepared_layer(mut mods: NativeModules, block_fields: PythonObject, kind: String, planes: Bool) raises -> PythonObject:
    # Capture a reusable layer for a prepared standalone owner. A capture
    # refusal is not raised here: the call then captures at its usual
    # position (causal_layer_run) and refuses there with the same error.
    try:
        return PythonObject(alloc=PreparedGenerationLayer(GenerationLayerSnapshot(block_fields, kind, planes)))
    except:
        mods.released()
        return PythonObject(None)


def _causal_layer_run_prepared(mut mods: NativeModules,
    e: PythonObject, kind: String, d: Int, b: Int, l: Int, layer: GenerationLayerSnapshot,
    head_profile: Bool, device15: Bool, x: PythonObject, supplied_state: PythonObject, step: Bool,
    mamba_workspace: PythonObject = PythonObject(None),
) raises -> PythonObject:
    return _causal_layer_run_bound(mods, e, kind, d, b, l, layer, head_profile, device15, x, supplied_state, step, layer.block, layer.profile, layer.profile_changed, mamba_workspace)


def _causal_layer_run_bound(mut mods: NativeModules,
    e: PythonObject, kind: String, d: Int, b: Int, l: Int, layer: GenerationLayerSnapshot,
    head_profile: Bool, device15: Bool, x: PythonObject, supplied_state: PythonObject, step: Bool,
    block_override: PythonObject, profile_override: PythonObject, profile_changed_override: Bool,
    mamba_workspace: PythonObject = PythonObject(None),
) raises -> PythonObject:
    var keep = List[PythonObject]()
    var copies = List[PythonObject]()
    var blk = block_override
    var st = supplied_state
    if py_type(st) is mods.module("types").SimpleNamespace:
        var failure = py_get(causal_snapshot_fields(mods, st), "_causal_admission_failure")
        if failure is not None:
            raise Error(String(failure))
    var y = generation_array([b, l, d])
    var addresses = PythonObject(None)
    var params = PythonObject(None)
    # every layer kind: a state a decode session holds resident is refused by name
    var gb = GENERATION_BOUNDARY.get_or_create_ptr()
    if py_call(gb[].bi.getattr, st, "_resident_session", PythonObject(None)) is not None:
        raise Error("CausalLM: layer state is owned by a decode session (resident in it); sync_state() or close it first")
    if kind == "transformer":
        var g = GENERATION_BOUNDARY.get_or_create_ptr()
        var old_profile = py_call(g[].bi.getattr, st, "arithmetic_profile", PythonObject(None))
        var profile_id = profile_override
        var profile_changed = profile_changed_override
        if old_profile is None and Int(py=st.cached_tokens) != 0 and profile_changed:
            raise Error("CausalLM: cached state requires arithmetic profile metadata")
        if old_profile is not None and profile_id is not None and not Bool(py=old_profile == profile_id):
            raise Error("CausalLM: state arithmetic profile differs")
        if profile_id is not None:
            py_setattr(st, "arithmetic_profile", profile_id)
        var capacity = Int(py=st.max_tokens)
        var cache_n = layer.words.cache_count(b, capacity)
        var p_words = layer.words.transformer_params(b, l, capacity, Int(py=st.cached_tokens))
        var a_words = generation_transformer_addresses(_addr(x, keep), layer.weights.resolve(keep),
            _addr(st.k_cache, keep), _addr(st.v_cache, keep), _addr(y, keep),
            layer.optional.resolve(keep), layer.packed.resolve(keep))
        a_words[10] = _mutable(st.k_cache, [cache_n], keep, copies, "k_cache")
        a_words[11] = _mutable(st.v_cache, [cache_n], keep, copies, "v_cache")
        var native_provider = py_get(e, "native_session_provider")
        var block_provider = py_get(e, "native_block_operations")
        if Bool(py=e["cpu"]) and block_provider is not None:
            py_setattr(st, "cached_tokens", PythonObject(neural_descriptor_execute(block_provider,
                34 if head_profile else 33, a_words, p_words, [])))
            mods.released()
        elif device15 and block_provider is not None:
            if blk["_native_session"] is None:
                causal_block_cache_set(mods, blk, "_native_session", e["create"]())
                mods.released()
                causal_block_cache_set(mods, blk, "_session_binding", e["block_binding"])
            var session = blk["_native_session"]
            var handle = neural_descriptor_handle(block_provider,
                e["transformer_native_session_handle_v1"], session)
            mods.released()
            py_setattr(st, "cached_tokens", PythonObject(neural_descriptor_reserved_execute(block_provider,
                session, handle, 0, 1, 35, a_words, p_words, [])))
            mods.released()
        elif native_provider is not None and not head_profile and String(getenv("MOJOLEARN_TRANSFORMER_LEGACY_SETUP")) != "1":
            if blk["_native_session"] is None:
                causal_block_cache_set(mods, blk, "_native_session", e["create"]())
                mods.released()
                causal_block_cache_set(mods, blk, "_session_binding", e["block_binding"])
            var session = blk["_native_session"]
            var linked = py_get(blk, "_native_forward_owner")
            if linked is None or not generation_native_session_matches(linked, native_provider, session):
                mods.released()
                linked = generation_connect_native_session(native_provider, session)
                mods.released()
                causal_block_cache_set(mods, blk, "_native_forward_owner", linked)
            mods.released()
            # Native vectors cross directly. No Python argument-list packing
            # or CPython callable dispatch occurs in this repeated layer call.
            py_setattr(st, "cached_tokens", PythonObject(generation_native_forward(linked, a_words, p_words)))
            mods.released()
        else:
            addresses = py_ints(a_words)
            params = py_ints(p_words)
            if device15:
                if blk["_native_session"] is None:
                    causal_block_cache_set(mods, blk, "_native_session", e["create"]())
                    mods.released()
                    causal_block_cache_set(mods, blk, "_session_binding", e["block_binding"])
                py_setattr(st, "cached_tokens", py_call(e["forward15"], blk["_native_session"], addresses, params))
                mods.released()
            elif head_profile:
                py_setattr(st, "cached_tokens", py_call(e["forward15_host"], addresses, params))
                mods.released()
            elif py_contains(e, "forward_session") and String(getenv("MOJOLEARN_TRANSFORMER_LEGACY_SETUP")) != "1":
                if blk["_native_session"] is None:
                    causal_block_cache_set(mods, blk, "_native_session", e["create"]())
                    mods.released()
                    causal_block_cache_set(mods, blk, "_session_binding", e["block_binding"])
                py_setattr(st, "cached_tokens", py_call(e["forward_session"], blk["_native_session"], addresses, params))
                mods.released()
            else:
                if step:
                    _ = py_call(params.pop, 1)
                py_setattr(st, "cached_tokens", py_call(e["decode" if step else "forward"], addresses, params))
                mods.released()
    else:
        if py_call(GENERATION_BOUNDARY.get_or_create_ptr()[].bi.getattr, st, "_resident_session", PythonObject(None)) is not None:
            raise Error("neural block: state is owned by a resident decode session")
        # Fixed public Mamba ABI words, plus the optional Mamba1 affine tail.
        # Reserve once so the input/weights/state/report vector does not grow
        # through multiple allocations during every public decode step.
        var workspace = mamba_workspace is not None
        if workspace and (kind != "mamba2" or not step):
            raise Error("Mamba2 public workspace requires its admitted public step")
        var m_addresses = List[Int](capacity=len(layer.weights.addresses) + 7 + Int(workspace))
        if workspace:
            var handle = neural_descriptor_resource_handle(mamba_workspace, [])
            if handle == 0:
                raise Error("Mamba2 public workspace: owner is closed")
            m_addresses.append(handle)
        m_addresses.append(_addr(x, keep))
        layer.weights.append_resolved(m_addresses, keep)
        if kind == "mamba1":
            var di = layer.words.inner
            m_addresses.append(_mutable(st.conv_window, [b, di, 4], keep, copies, "conv_window"))
            m_addresses.append(_mutable(st.h, [b, di, 16], keep, copies, "h"))
            m_addresses.append(_addr(y, keep))
            _ = y  # keep the Python object alive through the native call above
            var m_params: List[Int] = [b, d] if step else [b, l, d]
            var profile = Int(py=e["mamba1_profile"])
            if Int(py=st.profile_id) != profile or not Bool(py=st._nn34_usable):
                raise Error("CausalLM: incompatible or failed Mamba1 state")
            if profile == 2:
                m_addresses.append(_mutable(st.affine_boundary, [b, di, 16], keep, copies, "affine_boundary"))
                _ = y  # keep the Python object alive through the native call above
                m_addresses.append(_mutable(st.affine_a, [b, di, 16, 6], keep, copies, "affine_a"))
                _ = y  # keep the Python object alive through the native call above
                m_addresses.append(_mutable(st.affine_b, [b, di, 16, 6], keep, copies, "affine_b"))
                _ = y  # keep the Python object alive through the native call above
                m_params.append(Int(py=st.absolute_position))
            var native = py_get(e, "native_mamba")
            if native is not None and profile == 1:
                _ = neural_descriptor_execute(native, 1 if step else 0, m_addresses, m_params, [])
                mods.released()
            else:
                _ = py_call(e["decode" if step else "forward"], py_ints(m_addresses), py_ints(m_params))
                mods.released()
            _ = y  # keep the Python object alive through the native call above
            py_setattr(st, "absolute_position", PythonObject(Int(py=st.absolute_position) + l))
        else:
            var nh = layer.words.heads
            var cd = layer.words.conv
            m_addresses.append(_mutable(st.conv_window, [b, cd, 4], keep, copies, "conv_window"))
            _ = y  # keep the Python object alive through the native call above
            m_addresses.append(_mutable(st.h, [b, nh, 64, 128], keep, copies, "h"))
            _ = y  # keep the Python object alive through the native call above
            m_addresses.append(_mutable(st.buffer_xbc, [b, 256, cd], keep, copies, "buffer_xbc"))
            _ = y  # keep the Python object alive through the native call above
            m_addresses.append(_mutable(st.buffer_dtraw, [b, 256, nh], keep, copies, "buffer_dtraw"))
            _ = y  # keep the Python object alive through the native call above
            m_addresses.append(_addr(y, keep))
            _ = y  # keep the Python object alive through the native call above
            var last = generation_array([b, layer.words.heads, 64, 128])
            m_addresses.append(_addr(last, keep))
            _ = last  # keep the Python object alive through the native call above
            _ = y  # keep the Python object alive through the native call above
            var m_params: List[Int] = [b, d, Int(py=st.buffered_tokens)] if step else [b, l, d, Int(py=st.buffered_tokens)]
            var native = py_get(e, "native_mamba")
            if workspace:
                var resource = mamba_workspace.downcast_value_ptr[NativeDescriptorResource]()
                py_setattr(st, "buffered_tokens", PythonObject(neural_descriptor_execute(resource[].provider,
                    MAMBA2_WORKSPACE_STEP, m_addresses, m_params, [layer.words.dt_low, layer.words.dt_high])))
                mods.released()
            elif native is not None:
                py_setattr(st, "buffered_tokens", PythonObject(neural_descriptor_execute(native, 3 if step else 2,
                    m_addresses, m_params, [layer.words.dt_low, layer.words.dt_high])))
                mods.released()
            else:
                params = py_ints(m_params)
                _ = py_call(params.extend, Python.list(PythonObject(layer.words.dt_low), PythonObject(layer.words.dt_high)))
                py_setattr(st, "buffered_tokens", py_call(e["decode" if step else "forward"], py_ints(m_addresses), params))
                mods.released()
            _ = last  # keep the Python object alive through the native call above
            _ = y  # keep the Python object alive through the native call above
            causal_block_cache_set(mods, blk, "h_last_", last)
    _writeback(copies)
    _ = layer  # retain captured Array weights until every native read completes
    _ = mamba_workspace  # retain resource and compiled image across GIL release
    _ = keep^
    return y


def causal_model_embedding(model: PythonObject, ids: PythonObject) raises -> PythonObject:
    _causal_array_guard()
    var flat = Int(py=ids.ndim) == 1
    var b = 1 if flat else Int(py=ids.shape[0])
    var l = Int(py=ids.shape[0]) if flat else Int(py=ids.shape[1])
    var d = Int(py=model.d_model)
    var v = Int(py=model.vocab_size)
    var e = model._native_entries
    var weights = GenerationWeightSnapshot(Python.list(model._embed))
    return _causal_model_embedding_prepared(e, py_get(e, "native_primitives"), d, v, b, l, flat, weights, ids)


def _causal_model_embedding_prepared(
    e: PythonObject, native: PythonObject, d: Int, v: Int, b: Int, l: Int, flat: Bool,
    weights: GenerationWeightSnapshot, ids: PythonObject,
) raises -> PythonObject:
    var keep = List[PythonObject]()
    var x = generation_array([b, l, d])
    var weights_at = weights.resolve(keep)
    var a: List[Int] = [_addr(x, keep), weights_at[0], _addr(ids, keep)]
    if native is not None:
        _ = neural_descriptor_execute(native, 0, a, [b * l, v, d], [])
    else:
        _ = py_call(e["embedding_forward"], py_ints(a), py_ints([b * l, v, d]))
    _ = x  # keep the Python object alive through the native call above
    if flat:
        _causal_array_guard()
        var bi = GENERATION_BOUNDARY.get_or_create_ptr()[].bi
        py_setattr(x, "shape", py_call(bi.tuple, py_ints([l, d])))
        py_setattr(x, "strides", py_call(bi.tuple, py_ints([d * 4, 4])))
        py_setattr(x, "ndim", PythonObject(2))
    _ = weights
    _ = keep^
    return x


def causal_model_head(model: PythonObject, x: PythonObject, normalize: Bool = True, packed: Bool = True) raises -> PythonObject:
    _causal_array_guard()
    var flat = Int(py=x.ndim) == 2
    var b = 1 if flat else Int(py=x.shape[0])
    var l = Int(py=x.shape[0]) if flat else Int(py=x.shape[1])
    var d = Int(py=model.d_model)
    var v = Int(py=model.vocab_size)
    var head = GenerationHeadSnapshot(model, model._native_entries, d, v, normalize, packed)
    return _causal_model_head_prepared(head, x, b, l, flat, PythonObject(None))


def _causal_logits_output(b: Int, l: Int, v: Int, output: PythonObject) raises -> PythonObject:
    if output is None:
        return generation_array([b, l, v])
    # Private read-owned scratch has this admitted shape. Preserve the fresh
    # Array's zero seed even for optional legacy/packed binding routes; reuse
    # removes allocation and Array metadata construction, not initialization.
    var keep = List[PythonObject]()
    memset_zero(f32_ptr(_addr(output, keep)), b * l * v)
    _ = keep
    return output


def _causal_model_head_prepared(head: GenerationHeadSnapshot, x: PythonObject, b: Int, l: Int, flat: Bool, output: PythonObject) raises -> PythonObject:
    var e = head.entries
    var d = head.width
    var v = head.vocab
    var keep = List[PythonObject]()
    var weights_at = head.weights.resolve(keep)
    var hn = x
    var native = head.native
    if head.normalize:
        hn = generation_array([b, l, d])
        var a: List[Int] = [_addr(hn, keep), _addr(x, keep), weights_at[0]]
        if native is not None:
            _ = neural_descriptor_execute(native, 2, a, [b * l, d], [head.norm_eps])
        else:
            var np = py_ints([b * l, d])
            _ = py_call(np.append, PythonObject(head.norm_eps))
            _ = py_call(e["rms_norm_forward"], py_ints(a), np)
        _ = hn  # keep the Python object alive through the native call above
    var out = _causal_logits_output(b, l, v, output)
    if not head.packed:
        var a: List[Int] = [_addr(out, keep), _addr(hn, keep), weights_at[1]]
        if native is not None:
            _ = neural_descriptor_execute(native, 4, a, [b * l, v, d], [])
        else:
            _ = py_call(e["linear_forward"], py_ints(a), py_ints([b * l, v, d]))
        _ = out  # keep the Python object alive through the native call above
        _ = hn  # keep the Python object alive through the native call above
    else:
        var hi = generation_array([b * l, d], "b")
        var lo = generation_array([b * l, d], "b")
        var ex = generation_array([b * l], "i")
        _ = py_call(e["quantize15"], py_ints([_addr(hi, keep), _addr(lo, keep), _addr(ex, keep), _addr(hn, keep)]), py_ints([b * l, d]))
        _ = ex  # keep the Python object alive through the native call above
        _ = lo  # keep the Python object alive through the native call above
        _ = hi  # keep the Python object alive through the native call above
        _ = hn  # keep the Python object alive through the native call above
        e["gemm15"](py_ints([_addr(out, keep), _addr(hi, keep), _addr(lo, keep), _addr(ex, keep),
            weights_at[2], weights_at[3], weights_at[4]]), py_ints([b * l, v, d]))
        _ = ex  # keep the Python object alive through the native call above
        _ = lo  # keep the Python object alive through the native call above
        _ = hi  # keep the Python object alive through the native call above
        _ = out  # keep the Python object alive through the native call above
    if flat:
        _causal_array_guard()
        var bi = GENERATION_BOUNDARY.get_or_create_ptr()[].bi
        py_setattr(out, "shape", py_call(bi.tuple, py_ints([l, v])))
        py_setattr(out, "strides", py_call(bi.tuple, py_ints([v * 4, 4])))
        py_setattr(out, "ndim", PythonObject(2))
    _ = head
    _ = keep^
    return out


def _resident_profiles_ok(mut mods: NativeModules, model: PythonObject, state: PythonObject) raises -> Bool:
    # Profile tags are ordinary checkpoint metadata, including for the default
    # compiled arithmetic. Compare with the selected resident module, rather
    # than declining every tagged model or silently discarding a mismatch.
    var e = model._native_entries
    var selected = PythonObject(None)
    var changed = False
    if py_contains(e, "transformer_arithmetic_profile"):
        selected = e["transformer_arithmetic_profile"]()
        mods.released()
        if not py_contains(e, "transformer_arithmetic_profile_changed"):
            return False
        changed = Bool(py=e["transformer_arithmetic_profile_changed"]())
        mods.released()
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    for i in range(len(model._blocks)):  # driver-loop: captured profile admission once per public operation
        var blk = _block(mods, model, i)
        var profile = py_get(blk, "_arithmetic_profile", PythonObject(None))
        var st = state.layers[i]
        if py_type(st) is mods.module("types").SimpleNamespace and py_get(causal_snapshot_fields(mods, st), "_causal_admission_failure") is not None:
            return False
        if py_call(g[].bi.getattr, st, "_resident_session", PythonObject(None)) is not None:
            return False
        var old = py_call(g[].bi.getattr, st, "arithmetic_profile", PythonObject(None))
        if selected is None:
            if profile is not None or old is not None:
                return False
        elif profile is None or not Bool(py=profile == selected):
            return False
        if old is not None and profile is not None and not Bool(py=old == profile):
            return False
        if old is None and Int(py=st.cached_tokens) != 0 and (changed or
                _generation_bool(py_get(blk, "_arithmetic_profile_changed", PythonObject(False)))):
            # The ordinary layer path supplies its existing, ordered diagnostic.
            return False
    return True


def _resident_publish_profile(blk: PythonObject, state: PythonObject) raises:
    var profile = py_get(blk, "_arithmetic_profile", PythonObject(None))
    if profile is not None:
        py_setattr(state, "arithmetic_profile", profile)


def _resident_forward_ok(mut mods: NativeModules, model: PythonObject, state: PythonObject) raises -> Bool:
    """Whether `forward` runs as the one resident program
    (`causal_lm_session_forward`): a transformer stack with a norm, the
    resident entries, a head the resident session takes, matching arithmetic
    profile metadata and no layer state held by a decode session."""
    var e = model._native_entries
    if String(model.kind) != "transformer" or not _resident_norm_ok(model):
        return False
    for name in [String("causal_lm_session_forward"), String("transformer_decode_session_load_state"),  # small-loop(required resident entry names)
                 String("transformer_decode_session_export_state"), String("causal_lm_session_run")]:
        if not py_contains(e, name):
            return False
    if String(getenv("MOJOLEARN_TRANSFORMER_LEGACY_SETUP")) == "1" or String(getenv("MOJOLEARN_CAUSAL_LM_RESIDENT_FORWARD")) == "0":
        return False
    if model._head_int15 is not None and String(e["resident_vendor"]) != "cuda":
        return False
    return _resident_profiles_ok(mods, model, state)


def _resident_read_disjoint_state(mut mods: NativeModules, model: PythonObject, state: PythonObject) raises -> Bool:
    # The multi-token resident program skips the legacy export/reload between
    # tokens. Overlapping host KV spans would couple independent device caches
    # through that export order, including K/V aliases within one layer. Admit
    # only provably disjoint captured Array spans; unknown/foreign layouts keep
    # the existing per-token route and its original validation/error position.
    var starts = List[Int]()
    var sizes = List[Int]()
    var cls = GENERATION_BOUNDARY.get_or_create_ptr()[].array_cls
    try:
        for i in range(len(state.layers)):  # driver-loop: one read's retained layer state admission
            var fields = causal_snapshot_fields(mods, state.layers[i])
            for name in [String("k_cache"), String("v_cache")]:  # small-loop(two KV storage spans)
                var value = py_get(fields, name)
                if py_type(value) is not cls:
                    return False
                var view = generation_array_snapshot(value)
                if String(view.order) != "C" or String(view.dtype) != "<f4" or Bool(py=view._readonly):
                    return False
                var start = Int(py=view._addr)
                var size = Int(py=view.nbytes)
                if start <= 0 or size <= 0:
                    return False
                for j in range(len(starts)):  # driver-loop: native interval overlap, no Python element dispatch
                    # Differences avoid address+length overflow at the span end.
                    if (start <= starts[j] and starts[j] - start < size
                            or starts[j] < start and start - starts[j] < sizes[j]):
                        return False
                starts.append(start)
                sizes.append(size)
                _ = view
        var model_fields = _standalone_record_fields(mods, model)
        if (Bool(py=py_get(model_fields, "_parameters_exposed", PythonObject(False)))
                or Bool(py=py_get(model_fields, "_resident_busy", PythonObject(False)))):
            # The legacy route reopens private sessions each token under this
            # policy. A KV write into live weight storage would therefore feed
            # changed weights into its next forward. Read-wide weights cannot
            # preserve that coupling, even when all KV spans are disjoint.
            var weights = List[PythonObject]()
            weights.append(model._embed)
            weights.append(model._norm)
            var packed = model._head_int15 is not None
            if packed:
                for k in range(3):  # small-loop(consumed packed head planes)
                    weights.append(model._head_int15[k])
            else:
                weights.append(model._head)
            for i in range(len(model._blocks)):  # driver-loop: capture consumed weight spans once per read
                var block = _block(mods, model, i)
                var base = block["_w"]
                for j in range(len(base)):  # driver-loop: same weight slots as _transformer_arg_words
                    if not packed or j < 2:
                        weights.append(base[j])
                var optional = block["_wopt"]
                for j in range(len(optional)):  # driver-loop: consumed optional layer weight slots
                    weights.append(optional[j])
                if packed:
                    var planes = block["_int15"]
                    for j in range(len(planes)):  # driver-loop: consumed packed layer planes
                        if planes[j] is not None:
                            for k in range(3):  # small-loop(int15 plane triple)
                                weights.append(planes[j][k])
            for value in weights:  # driver-loop: compare captured live weight spans against KV storage
                if value is None:
                    continue
                if py_type(value) is not cls:
                    return False
                var view = generation_array_snapshot(value)
                if String(view.order) != "C":
                    return False
                var start = Int(py=view._addr)
                var size = Int(py=view.nbytes)
                if start <= 0 or size <= 0:
                    return False
                for j in range(len(starts)):  # driver-loop: native byte-span overlap test
                    if (start <= starts[j] and starts[j] - start < size
                            or starts[j] < start and start - starts[j] < sizes[j]):
                        return False
                _ = view
    except:
        # Admission is only a route decision; the existing forward diagnoses
        # malformed/released state at its original operation/layer position.
        mods.released()
        return False
    return True


def _resident_forward(mut mods: NativeModules, model: PythonObject, ids: PythonObject, state: PythonObject, output: PythonObject, prepared_key: PythonObject, mut read_plan: Optional[LegacyGenerationReadPlan]) raises -> PythonObject:
    """`(B, L, V)` logits of `forward` in one resident program: the kept (or
    newly opened) sessions take the state's caches, run the stack, and the
    advanced caches come back into the state."""
    var e = model._native_entries
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    var b = Int(py=ids.shape[0])
    var l = Int(py=ids.shape[1])
    var v = Int(py=model.vocab_size)
    var int15 = model._head_int15 is not None
    var keep = List[PythonObject]()
    var copies = List[PythonObject]()
    var held = Bool(read_plan) and read_plan.value().resident_active
    if held and not _legacy_resident_valid(read_plan.value()):
        # A supported parameters() call may expose/invalidate the model while
        # native execution releases the GIL. Do not reuse or republish its old
        # session cache: release this loan and resume the ordinary per-step path.
        read_plan.value().resident_reuse = False
        read_plan.value().resident_invalidated = True
        _legacy_resident_cancel(mods, model, read_plan.value())
        held = False
    var fields = read_plan.value().resident_fields if held else _standalone_record_fields(mods, model)
    var persist = held or (not Bool(py=py_get(fields, "_parameters_exposed", PythonObject(False)))
                   and not Bool(py=py_get(fields, "_resident_busy", PythonObject(False))))
    var key = _resident_key(mods, model, state, b, int15) if prepared_key is None else prepared_key
    var retain = persist and Bool(read_plan) and read_plan.value().resident_reuse
    var kept = py_get(fields, "_resident_sessions") if persist and not held else PythonObject(None)
    var lm = PythonObject(None)
    var sessions = PythonObject(None)
    var out = _causal_logits_output(b, l, v, output)
    if persist and not held:
        causal_model_cache_set(mods, model, fields, "_resident_busy", True)
    if retain:
        read_plan.value().resident_fields = fields
        read_plan.value().resident_active = True
    try:
        if held:
            # The preceding successful forward exported these exact cache
            # contents. The read owns the model lease and disjoint stream state;
            # no host reload or immutable-weight re-admission is needed here.
            lm = read_plan.value().resident_lm
            sessions = read_plan.value().resident_sessions
            read_plan.value().resident_uploads_skipped += 1
        elif kept is not None and Bool(py=kept[0] == key):
            lm = kept[1]
            sessions = kept[2]
            if retain:
                read_plan.value().resident_lm = lm
                read_plan.value().resident_sessions = sessions
            _resident_load_states(e, sessions, state, keep)
            mods.released()
        else:
            if kept is not None:
                if retain:
                    read_plan.value().resident_lm = kept[1]
                    read_plan.value().resident_sessions = kept[2]
                causal_model_cache_set(mods, model, fields, "_resident_sessions", PythonObject(None))
                _resident_close(e, kept[1], kept[2])
                mods.released()
                if retain:
                    read_plan.value().resident_lm = PythonObject(None)
                    read_plan.value().resident_sessions = PythonObject(None)
            var opened = _resident_open_all(mods, model, e, state, b, l, int15, keep)
            lm = opened[0]
            sessions = opened[1]
            if retain:
                read_plan.value().resident_lm = lm
                read_plan.value().resident_sessions = sessions
        if retain and not held:
            var live = py_get(fields, "_causal_live_fields", fields)
            read_plan.value().resident_cache = py_get(live, "_resident_sessions")
        var provider = py_get(e, "native_block_operations")
        var cached = PythonObject(None)
        if provider is not None:
            cached = PythonObject(transformer_resident_lm_run(provider, lm, sessions, False, [_addr(ids, keep), _addr(out, keep)], [l]))
            mods.released()
        else:
            cached = py_call(e["causal_lm_session_forward"], lm, sessions, py_ints([_addr(ids, keep), _addr(out, keep)]), py_ints([l]))
            mods.released()
        _ = out  # keep the Python object alive through the native call above
        for i in range(len(sessions)):  # driver-loop: over layer decode sessions
            var blk = _block(mods, model, i)
            var st = state.layers[i]
            var cap = Int(py=st.max_tokens) if Int(py=blk["window"]) == 0 else Int(py=blk["window"])
            var cache_n = b * Int(py=blk["n_kv_heads"]) * cap * Int(py=blk["head_dim"])
            var pk = _mutable(st.k_cache, [cache_n], keep, copies, "k_cache")
            var pv = _mutable(st.v_cache, [cache_n], keep, copies, "v_cache")
            var provider = py_get(e, "native_block_operations")
            if provider is not None:
                _ = transformer_resident_decode_call(provider, sessions[i], 22, [pk, pv], [])
                mods.released()
            else:
                _ = py_call(e["transformer_decode_session_export_state"], sessions[i], py_ints([pk, pv]))
                mods.released()
            _resident_publish_profile(blk, st)
            py_setattr(st, "cached_tokens", cached)
            _ = out  # keep the Python object alive through the native call above
        _writeback(copies)
    except error:
        mods.released()
        if retain:
            _legacy_resident_cancel(mods, model, read_plan.value())
            raise error
        if persist:
            causal_model_cache_set(mods, model, fields, "_resident_sessions", PythonObject(None))
            causal_model_cache_set(mods, model, fields, "_resident_busy", False)
        if lm is not None:
            _resident_close(e, lm, sessions)
            mods.released()
        raise error
    if retain:
        # Export/writeback above is still per successful forward, before host
        # selection. Only the upload and cache-owner publication are deferred.
        _ = read_plan
    elif persist:
        causal_model_cache_set(mods, model, fields, "_resident_sessions", Python.tuple(key, lm, sessions))
        causal_model_cache_set(mods, model, fields, "_resident_busy", False)
    else:
        _resident_close(e, lm, sessions)
        mods.released()
    _ = keep^
    return out


def _use_native_generation_engine(mut mods: NativeModules, model: PythonObject, state: PythonObject) raises -> Bool:
    # Select once at a public operation boundary, never in the token/layer
    # loop. No failed native admission/execution is caught and retried through
    # the object route: a selected engine keeps its original refusal semantics.
    var setting = String(getenv("MOJOLEARN_GENERATION_NATIVE_ENGINE", "auto"))
    if setting == "1":
        return True
    if setting != "auto":
        return False
    # Explicit dispatch opt-out keeps automatic selection on its supported
    # compatibility route. A forced engine still reports admission failures.
    if getenv("MOJOLEARN_NEURAL_NATIVE_DISPATCH", "1") != "1":
        return False
    var entries = model._native_entries
    if py_get(entries, "parallel_call") is not None or model._head_int15 is not None:
        return False
    if String(getenv("MOJOLEARN_TRANSFORMER_LEGACY_SETUP")) == "1":
        return False
    var kind = String(model.kind)
    if kind != "transformer" and kind != "mamba1" and kind != "mamba2":
        return False
    if py_get(entries, "native_primitives") is None or py_get(entries, "native_block_operations" if kind == "transformer" else "native_mamba") is None:
        return False
    if kind == "mamba1" and Int(py=entries["mamba1_profile"]) != 1:
        return False
    # Existing resident execution already owns a separate supported native
    # program. Its route, distributed execution and packed arithmetic remain
    # supported; they are explicit remaining migration scopes, not refusals.
    return not _resident_forward_ok(mods, model, state)


def _engine_admission(mut mods: NativeModules, model: PythonObject, state: PythonObject, batch: Int, length: Int) raises -> GenerationEngineAdmission:
    # Once selected, admission failures propagate. The boundary selector
    # handles unsupported routes before execution, without exception fallback.
    _state(mods, model, state, batch, length)
    try:
        var entries = model._native_entries
        if py_get(entries, "parallel_call") is not None:
            raise Error("CausalLM: native generation engine distributed execution is pending")
        if _resident_forward_ok(mods, model, state):
            raise Error("CausalLM: native generation engine does not replace the existing resident route")
        if getenv("MOJOLEARN_TRANSFORMER_LEGACY_SETUP", "0") == "1":
            raise Error("CausalLM: native generation engine requires the supported session setup")
        var blocks = List[PythonObject]()
        for i in range(len(model._blocks)):  # driver-loop: one operation's retained model owners
            blocks.append(_block(mods, model, i))
        var snapshot = GenerationOperationSnapshot(model, entries, blocks)
        mods.released()
        var windowed_result5 = GenerationEngineAdmission(snapshot, state, batch, length)
        mods.released()
        return windowed_result5^
    except problem:
        mods.released()
        py_setattr(state, "_usable", PythonObject(False))
        raise problem


def _run_engine(mut mods: NativeModules, model: PythonObject, ids: PythonObject, state: PythonObject) raises -> PythonObject:
    _causal_array_guard()
    var shape = ids.shape
    var batch = Int(py=shape[0])
    var length = Int(py=shape[1])
    var owner = _engine_admission(mods, model, state, batch, length)
    var keep = List[PythonObject]()
    var address = _addr(ids, keep)
    var logits = List[Float32]()
    try:
        # Common endpoints own per-call GIL release, lock acquisition and
        # session reservation. The native layer loop enters GIL-held.
        logits = owner.native.forward(address, length)
        mods.released()
    except problem:
        mods.released()
        owner.publish()
        raise problem
    owner.publish()
    var result = generation_array([batch, length, owner.native.vocab])
    memcpy(dest=f32_ptr(Int(py=result._addr)), src=logits.unsafe_ptr(), count=len(logits))
    _ = result
    _ = logits
    _ = owner
    _ = keep^
    return result


struct LegacyGenerationReadPlan(Movable):
    # One captured public read only. Public entry recaptures model/state on the
    # next call. Native execution changes positions/cache contents, not these
    # passive shape/route words. Parallel and uncaptured callers do not use it.
    var resident: Bool
    var batch: Int
    var limited: Bool
    var max_positions: Int
    var capacity: Int
    var resident_key: PythonObject
    var logits: PythonObject
    var logits_length: Int
    var resident_reuse: Bool
    var resident_uploads_skipped: Int
    var resident_invalidated: Bool
    var resident_active: Bool
    var resident_fields: PythonObject
    var resident_cache: PythonObject
    var resident_lm: PythonObject
    var resident_sessions: PythonObject

    def __init__(out self, model: PythonObject, state: PythonObject, batch: Int, resident: Bool) raises:
        self.resident = resident
        self.batch = batch
        self.limited = model.max_positions is not None
        self.max_positions = Int(py=model.max_positions) if self.limited else 0
        self.capacity = Int(py=state.max_tokens)
        self.resident_key = PythonObject(None)
        self.logits = PythonObject(None)
        self.logits_length = -1
        self.resident_reuse = False
        self.resident_uploads_skipped = 0
        self.resident_invalidated = False
        self.resident_active = False
        self.resident_fields = PythonObject(None)
        self.resident_cache = PythonObject(None)
        self.resident_lm = PythonObject(None)
        self.resident_sessions = PythonObject(None)

    def clear_resident(mut self):
        self.resident_active = False
        self.resident_fields = PythonObject(None)
        self.resident_cache = PythonObject(None)
        self.resident_lm = PythonObject(None)
        self.resident_sessions = PythonObject(None)

    def positions(self, state: PythonObject, batch: Int, length: Int) raises:
        # Position/capacity checks retain their per-step execution point, before
        # the forward try block (a refusal does not poison otherwise valid state).
        if batch != self.batch:
            raise Error("CausalLM: state batch/layer counts disagree with model and input")
        var have = Int(py=state.positions)
        if length < 0 or have < 0:
            raise Error("CausalLM: negative sequence position")
        if self.limited and have + length > self.max_positions:
            raise Error("CausalLM: max_tokens or sequence exceeds max_positions")
        if have + length > self.capacity:
            raise Error("CausalLM: sequence exceeds allocated state capacity")


def _legacy_resident_valid(mut plan: LegacyGenerationReadPlan) raises -> Bool:
    # This is an ownership invalidation check, not a scan of model data. A
    # per-operation snapshot does not own a concurrent public cache transition.
    var fields = plan.resident_fields
    var live = py_get(fields, "_causal_live_fields", fields)
    var exposed = py_get(live, "_parameters_exposed", PythonObject(False))
    if exposed is not PythonObject(False):
        # Exposure is monotonic. Preserve the live transition in this snapshot
        # too, so final publication cannot restore its earlier False value.
        py_setitem(fields, "_parameters_exposed", PythonObject(True))
        return False
    return py_get(live, "_resident_sessions") is plan.resident_cache


def _legacy_resident_cancel(mut mods: NativeModules, model: PythonObject, mut plan: LegacyGenerationReadPlan):
    # Failure cleanup must neither replace the operation error nor leave the
    # model busy. Pins remain live until every owner has had its close attempt.
    var fields = plan.resident_fields
    var lm = plan.resident_lm
    var sessions = plan.resident_sessions
    plan.resident_active = False
    try:
        causal_model_cache_set(mods, model, fields, "_resident_sessions", PythonObject(None))
    except:
        mods.released()
    try:
        _resident_read_cleanup(py_get(model._native_entries, "native_block_operations"), lm, sessions)
        mods.released()
    except:
        mods.released()
    try:
        causal_model_cache_set(mods, model, fields, "_resident_busy", False)
    except:
        mods.released()
    plan.clear_resident()


def _legacy_resident_finish(mut mods: NativeModules, model: PythonObject, mut read_plan: Optional[LegacyGenerationReadPlan]) raises:
    if not read_plan or not read_plan.value().resident_active:
        return
    ref plan = read_plan.value()
    try:
        if not _legacy_resident_valid(plan):
            _legacy_resident_cancel(mods, model, plan)
            return
        causal_model_cache_set(mods, model, plan.resident_fields, "_resident_sessions",
            Python.tuple(plan.resident_key, plan.resident_lm, plan.resident_sessions))
        causal_model_cache_set(mods, model, plan.resident_fields, "_resident_busy", False)
    except error:
        _legacy_resident_cancel(mods, model, plan)
        raise error
    plan.clear_resident()


def _legacy_read_captured(mut mods: NativeModules, model: PythonObject, state: PythonObject) raises -> Bool:
    var namespace = mods.module("types").SimpleNamespace
    return (py_type(model) is namespace and py_type(state) is namespace
            and py_contains(causal_snapshot_fields(mods, model), "_causal_original")
            and py_contains(causal_snapshot_fields(mods, state), "_causal_original"))


def _run(mut mods: NativeModules, model: PythonObject, ids: PythonObject, state: PythonObject, remote_step: Bool = False) raises -> PythonObject:
    if _use_native_generation_engine(mods, model, state):
        return _run_engine(mods, model, ids, state)
    var snapshot: Optional[GenerationOperationSnapshot] = None
    var read_plan: Optional[LegacyGenerationReadPlan] = None
    return _run_prepared(mods, model, ids, state, snapshot, read_plan, remote_step)


def _run_prepared(mut mods: NativeModules, model: PythonObject, ids: PythonObject, state: PythonObject,
                  mut snapshot: Optional[GenerationOperationSnapshot], mut read_plan: Optional[LegacyGenerationReadPlan], remote_step: Bool = False,
                  admitted: Bool = False, retain_resident: Bool = False) raises -> PythonObject:
    # `admitted`: the caller (the generation loop) already guarded the Array
    # class for this GIL-release window; do not repeat the full guard.
    if not admitted:
        _causal_array_guard()
    var shape = ids.shape
    var b = Int(py=shape[0])
    var l = Int(py=shape[1])
    if read_plan:
        read_plan.value().positions(state, b, l)
    else:
        _state(mods, model, state, b, l)
    var entries = model._native_entries
    var parallel = py_get(entries, "parallel_call")
    try:
        if parallel is not None:
            var windowed_result4 = py_call(parallel, "run", Python.list(causal_model_origin(mods, model), ids, causal_state_origin(mods, state), remote_step, model, state))
            mods.released()
            return windowed_result4^
        var resident = read_plan.value().resident if read_plan else _resident_forward_ok(mods, model, state)
        if not read_plan and admitted and _legacy_read_captured(mods, model, state):
            read_plan = LegacyGenerationReadPlan(model, state, b, resident)
            if resident and retain_resident and py_get(entries, "native_block_operations") is not None:
                var fields = _standalone_record_fields(mods, model)
                if (not Bool(py=py_get(fields, "_parameters_exposed", PythonObject(False)))
                        and not Bool(py=py_get(fields, "_resident_busy", PythonObject(False)))):
                    read_plan.value().resident_reuse = _resident_read_disjoint_state(mods, model, state)
        var output = PythonObject(None)
        var resident_key = PythonObject(None)
        if read_plan:
            ref plan = read_plan.value()
            # Private scratch never escapes as public forward logits. Prompt
            # and one-token extents each allocate only when their shape changes.
            if plan.logits_length == l:
                output = plan.logits
            if resident:
                if plan.resident_key is None:
                    plan.resident_key = _resident_key(mods, model, state, b, model._head_int15 is not None)
                resident_key = plan.resident_key
        if resident:
            var logits = _resident_forward(mods, model, ids, state, output, resident_key, read_plan)
            if read_plan:
                read_plan.value().logits = logits
                read_plan.value().logits_length = l
            py_setattr(state, "positions", PythonObject(Int(py=state.positions) + l))
            return logits
        if not snapshot:
            # Build only on the local nonresident route, after admission and
            # route checks. Parallel/resident execution retains its own owners.
            var blocks = List[PythonObject]()
            for i in range(len(model._blocks)):  # driver-loop: capture layer owners once per operation
                blocks.append(_block(mods, model, i))
            snapshot = GenerationOperationSnapshot(model, entries, blocks)
            mods.released()
        var out = _run_snapshot(mods, snapshot.value(), ids, state, b, l, output)
        if read_plan:
            read_plan.value().logits = out
            read_plan.value().logits_length = l
        py_setattr(state, "positions", PythonObject(Int(py=state.positions) + l))
        return out
    except error:
        mods.released()
        py_setattr(state, "_usable", PythonObject(False))
        raise error


def _run_snapshot(mut mods: NativeModules, snapshot: GenerationOperationSnapshot, ids: PythonObject, state: PythonObject,
                  b: Int, l: Int, output: PythonObject) raises -> PythonObject:
    var x = _causal_model_embedding_prepared(snapshot.entries, snapshot.native, snapshot.width, snapshot.vocab,
                                            b, l, False, snapshot.embedding, ids)
    mods.released()
    for i in range(len(snapshot.layers)):  # driver-loop: over retained native layer descriptions
        x = _causal_layer_run_prepared(mods, snapshot.entries, snapshot.kind, snapshot.width, b, l,
            snapshot.layers[i], snapshot.head.packed, snapshot.planes, x, state.layers[i], False, PythonObject(None))
    var out = _causal_model_head_prepared(snapshot.head, x, b, l, False, output)
    mods.released()
    _ = snapshot  # pins model and all captured weight owners until computation finishes
    return out


def causal_lm_forward_binding(mut mods: NativeModules, model: PythonObject, value: PythonObject, supplied_state: PythonObject, step: PythonObject) raises -> PythonObject:
    var state = supplied_state
    var one = Bool(py=step)
    var ids = generation_ids(value, Int(py=model.vocab_size), one)
    if state is None:
        if one:
            raise Error("CausalLM.step: state is required")
        state = causal_lm_allocate_binding(mods, model, ids.shape[0], ids.shape[1])
    var out = PythonObject(None)
    var parallel = py_get(model._native_entries, "parallel_call")
    try:
        out = _run(mods, model, ids, state, one)
    except error:
        mods.released()
        if supplied_state is None and parallel is not None:
            try:
                _ = py_call(parallel, "release_layers", Python.list(causal_model_origin(mods, model), causal_state_origin(mods, state), model, state))
                mods.released()
            except cleanup_error:
                raise Error("CausalLM: forward failed: " + String(error) + "; remote cleanup also failed: " + String(cleanup_error))
        raise error
    if supplied_state is None and parallel is not None:
        _ = py_call(parallel, "release_layers", Python.list(causal_model_origin(mods, model), causal_state_origin(mods, state), model, state))
        mods.released()
    if one:
        _causal_array_guard()
        var g = GENERATION_BOUNDARY.get_or_create_ptr()
        py_setattr(out, "shape", py_call(g[].bi.tuple, py_ints([Int(py=ids.shape[0]), Int(py=model.vocab_size)])))
        py_setattr(out, "strides", py_call(g[].bi.tuple, py_ints([Int(py=model.vocab_size) * 4, 4])))
        py_setattr(out, "ndim", PythonObject(2))
    return out


def causal_lm_forward_binding(model: PythonObject, value: PythonObject, supplied_state: PythonObject, step: PythonObject) raises -> PythonObject:
    # Classical and other callers keep their per-call module checks.
    var mods = NativeModules()
    return causal_lm_forward_binding(mods, model, value, supplied_state, step)


def _resident_close(entries: PythonObject, lm: PythonObject, sessions: PythonObject) raises:
    var provider = py_get(entries, "native_block_operations")
    if provider is not None:
        transformer_resident_lm_close(provider, lm)
    else:
        _ = py_call_typed(entries["causal_lm_session_close"], lm)
    for i in range(len(sessions)):  # driver-loop: over layer decode sessions
        var provider = py_get(entries, "native_block_operations")
        if provider is not None:
            _ = transformer_resident_decode_call(provider, sessions[i], 24, [], [])
        else:
            _ = py_call_typed(entries["transformer_decode_session_close"], sessions[i])


def _resident_open_all(mut mods: NativeModules,
    model: PythonObject, e: PythonObject, state: PythonObject, b: Int, l: Int, int15: Bool,
    mut keep: List[PythonObject],
) raises -> PythonObject:
    """Open the model's resident sessions on `state`: one decode session per
    layer (weights and the layer's caches copied to the device) and the
    model session (embedding, norm, head). Returns (lm, sessions)."""
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    var sessions = g[].bi.list()
    var lm = e["causal_lm_session_create"]()
    mods.released()
    try:
        for i in range(len(model._blocks)):  # driver-loop: over model layers
            var blk = _block(mods, model, i)
            var st = state.layers[i]
            var a = _transformer_args(model, blk, st, PythonObject(None), PythonObject(None), b, l, int15, keep)
            var open_a = g[].bi.list()
            for j in range(1, 12):  # driver-loop: over layer argument spans
                _ = py_call(open_a.append, a[j])
            for j in range(13, len(a)):  # driver-loop: over layer argument spans
                _ = py_call(open_a.append, a[j])
            var p = _transformer_params(blk, st, b, l)
            var open_p = g[].bi.list()
            for j in range(len(p)):  # driver-loop: over layer shape parameters
                if j != 1:
                    _ = py_call(open_p.append, p[j])
            var ss = e["transformer_decode_session_create"]()
            mods.released()
            _ = py_call(sessions.append, ss)
            var provider = py_get(e, "native_block_operations")
            if provider is not None:
                _ = transformer_resident_decode_call(provider, ss, 19 if int15 else 18,
                    transformer_resident_words(open_a), transformer_resident_words(open_p))
                mods.released()
            else:
                _ = py_call(e["transformer_decode_session_open_int15" if int15 else "transformer_decode_session_open"], ss, open_a, open_p)
                mods.released()
        # A model without a final norm passes 0 (admitted only under
        # MOJOLEARN_IDN_LM_RESIDENT_GENERATE; see `_resident_norm_ok`).
        var top = py_ints([_addr(model._embed, keep), 0 if model._norm is None else _addr(model._norm, keep)])
        if int15:
            for i in range(3):  # small-loop(int15 head plane triple)
                _ = py_call(top.append, _addr(model._head_int15[i], keep))
        else:
            _ = py_call(top.append, 0 if model._head is model._embed else _addr(model._head, keep))
        var tp = py_ints([b, Int(py=model.vocab_size), Int(py=model.d_model)])
        _ = py_call(tp.append, model.norm_eps)
        var provider = py_get(e, "native_block_operations")
        if provider is not None:
            transformer_resident_lm_open(provider, lm, int15, transformer_resident_words(top),
                [b, Int(py=model.vocab_size), Int(py=model.d_model)], Float64(py=model.norm_eps))
            mods.released()
        else:
            _ = py_call(e["causal_lm_session_open_int15" if int15 else "causal_lm_session_open"], lm, top, tp)
            mods.released()
    except error:
        mods.released()
        _resident_close(e, lm, sessions)
        mods.released()
        raise error
    return Python.tuple(lm, sessions)


def _resident_key(mut mods: NativeModules, model: PythonObject, state: PythonObject, b: Int, int15: Bool) raises -> PythonObject:
    """What a kept resident session set must match: the rows, every layer's
    cache capacity/profile, the head format and the entries it was opened with."""
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    var caps = g[].bi.list()
    for i in range(len(state.layers)):  # driver-loop: over model layers
        var profile = py_get(_block(mods, model, i), "_arithmetic_profile", PythonObject(None))
        _ = py_call(caps.append, Python.tuple(state.layers[i].max_tokens, profile))
    var identity = py_get(_standalone_record_fields(mods, model), "_causal_entries_identity", model._native_entries)
    return Python.tuple(b, py_call(g[].bi.tuple, caps), int15, py_call(g[].bi.id, identity))


def _resident_load_states(e: PythonObject, sessions: PythonObject, state: PythonObject, mut keep: List[PythonObject]) raises:
    """Each kept layer session takes this call's caches (the state half of
    an open; the weights stay where they are)."""
    for i in range(len(sessions)):  # driver-loop: over layer decode sessions
        var st = state.layers[i]
        var provider = py_get(e, "native_block_operations")
        var a: List[Int] = [_addr(st.k_cache, keep), _addr(st.v_cache, keep)]
        var p: List[Int] = [Int(py=st.cached_tokens)]
        if provider is not None:
            _ = transformer_resident_decode_call(provider, sessions[i], 23, a, p)
        else:
            _ = py_call(e["transformer_decode_session_load_state"], sessions[i], py_ints(a), py_ints(p))


def causal_lm_resident_binding(mut mods: NativeModules, model: PythonObject, value: PythonObject, maximum: PythonObject, last_logits: PythonObject, supplied_state: PythonObject) raises -> PythonObject:
    var e = model._native_entries
    if not py_contains(e, "causal_lm_session_run") or not _resident_norm_ok(model):
        return PythonObject(None)
    var int15 = model._head_int15 is not None
    if int15 and String(e["resident_vendor"]) != "cuda":
        return PythonObject(None)
    var ids = generation_ids(value, Int(py=model.vocab_size))
    var n = _public_int(mods, maximum, "generation: max_new_tokens must be an integer")
    if n == 0:
        return ids
    if n < 0:
        raise Error("generation: max_new_tokens must be nonnegative")
    var b = Int(py=ids.shape[0])
    var l = Int(py=ids.shape[1])
    var state = supplied_state
    if state is None:
        # This operation owns fresh, separately allocated caches, disjoint from
        # captured weights. Only externally supplied state needs span admission.
        state = causal_lm_allocate_binding(mods, model, PythonObject(b), PythonObject(l + n))
    elif n > 1 and not _resident_read_disjoint_state(mods, model, state):
        # Same optional-resident eligibility sentinel used above. Complete
        # generation then enters _read and keeps legacy alias coupling.
        return PythonObject(None)
    if not _resident_profiles_ok(mods, model, state):
        return PythonObject(None)
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    var keep = List[PythonObject]()
    var out = generation_array([b, n], "i")
    # The resident sessions persist under the model handle (lane
    # migrate-neural-2: every call re-uploaded every layer's weights). A kept
    # set is reused when its rows and capacities match; the caches of this
    # call's state are loaded into it. The model's weights never change after
    # construction; once `parameters()` has handed the live arrays out, the
    # sessions are opened per call again, so an in-place edit is observed as
    # before. A concurrent call, or a binding without load_state, opens its
    # own set.
    var fields = _standalone_record_fields(mods, model)
    var persist = (py_contains(e, "transformer_decode_session_load_state")
                   and not Bool(py=py_get(fields, "_parameters_exposed", PythonObject(False)))
                   and not Bool(py=py_get(fields, "_resident_busy", PythonObject(False))))
    var key = _resident_key(mods, model, state, b, int15)
    var kept = py_get(fields, "_resident_sessions") if persist else PythonObject(None)
    var lm = PythonObject(None)
    var sessions = PythonObject(None)
    if persist:
        causal_model_cache_set(mods, model, fields, "_resident_busy", True)
    try:
        if kept is not None and Bool(py=kept[0] == key):
            lm = kept[1]
            sessions = kept[2]
            _resident_load_states(e, sessions, state, keep)
            mods.released()
        else:
            if kept is not None:
                causal_model_cache_set(mods, model, fields, "_resident_sessions", PythonObject(None))
                _resident_close(e, kept[1], kept[2])
                mods.released()
            var opened = _resident_open_all(mods, model, e, state, b, l, int15, keep)
            lm = opened[0]
            sessions = opened[1]
        var provider = py_get(e, "native_block_operations")
        if provider is not None:
            _ = transformer_resident_lm_run(provider, lm, sessions, True,
                [_addr(ids, keep), _addr(out, keep), _addr(last_logits, keep)], [l, n, 0])
            mods.released()
        else:
            _ = py_call(e["causal_lm_session_run"], lm, sessions, py_ints([_addr(ids, keep), _addr(out, keep), _addr(last_logits, keep)]), py_ints([l, n, 0]))
            mods.released()
        _ = out  # keep the Python object alive through the native call above
        _ = ids  # keep the Python object alive through the native call above
    except error:
        mods.released()
        if persist:
            causal_model_cache_set(mods, model, fields, "_resident_sessions", PythonObject(None))
            causal_model_cache_set(mods, model, fields, "_resident_busy", False)
        if lm is not None:
            _resident_close(e, lm, sessions)
            mods.released()
        raise error
    if persist:
        causal_model_cache_set(mods, model, fields, "_resident_sessions", Python.tuple(key, lm, sessions))
        causal_model_cache_set(mods, model, fields, "_resident_busy", False)
    else:
        _resident_close(e, lm, sessions)
        mods.released()
    var result = generation_array([b, l + n], "i")
    var dst = i32_ptr(Int(py=result._addr))
    _ = result  # keep the Python object alive through the native call above
    var prompt = i32_ptr(Int(py=ids._addr))
    _ = ids  # keep the Python object alive through the native call above
    var generated = i32_ptr(Int(py=out._addr))
    _ = out  # keep the Python object alive through the native call above
    causal_lm_join_resident(dst, prompt, generated, b, l, n)
    _ = result  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
    _ = ids  # keep the Python object alive through the native call above
    _ = keep^
    return result


def causal_lm_resident_binding(model: PythonObject, value: PythonObject, maximum: PythonObject, last_logits: PythonObject, supplied_state: PythonObject) raises -> PythonObject:
    # Classical and other callers keep their per-call module checks.
    var mods = NativeModules()
    return causal_lm_resident_binding(mods, model, value, maximum, last_logits, supplied_state)


struct GenerationHandle(Movable, Writable):
    var model: PythonObject
    var state: PythonObject
    var prompt: PythonObject
    var current: PythonObject
    var tokenizer: PythonObject
    var pending_bytes: PythonObject
    var max_new: Int
    var produced: Int
    var greedy: Bool
    var sampling_policy: Int
    var temperature: Float32
    var top_k: Int
    var top_p: Float64
    var seed: UInt64
    var eos: List[Int]
    var pad: Int
    var lengths: List[Int]
    var finished: List[Bool]
    var busy: Bool
    var usable: Bool

    def __init__(out self, mut mods: NativeModules, model: PythonObject, prompt: PythonObject, maximum: Int) raises:
        self.model = model
        self.prompt = prompt
        self.current = prompt
        self.max_new = maximum
        self.produced = 0
        _causal_array_guard()
        var batch = Int(py=prompt.shape[0])
        var length = Int(py=prompt.shape[1])
        self.state = causal_lm_allocate_binding(mods, model, PythonObject(batch), PythonObject(length + maximum))
        self.greedy = True
        self.sampling_policy = 0
        self.temperature = Float32(1)
        self.top_k = 0
        self.top_p = Float64(1)
        self.seed = UInt64(0)
        self.eos = List[Int]()
        self.pad = 0
        self.lengths = List[Int](length=batch, fill=0)
        self.finished = List[Bool](length=batch, fill=False)
        self.busy = False
        self.usable = True
        self.tokenizer = PythonObject(None)
        self.pending_bytes = GENERATION_BOUNDARY.get_or_create_ptr()[].bi.bytearray()

    def write_to(self, mut writer: Some[Writer]):
        writer.write("GenerationHandle(produced=", self.produced, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)

    def done(self) -> Bool:
        if self.produced >= self.max_new:
            return True
        return causal_lm_rows_done(self.finished)


def _options(mut mods: NativeModules, mut s: GenerationHandle, options: PythonObject) raises:
    # The Python veneer forwards these scalar objects unchanged. Explicit
    # sampling requires an explicit seed; greedy consumes no RNG state.
    neural_plain_value(mods, options, 0, _causal_context())
    if py_type(options) is not mods.module("builtins").tuple and py_type(options) is not mods.module("builtins").list:
        raise Error("TypeError: generation options require an ordinary scalar tuple")
    if len(options) != 7:
        raise Error("generation: expected greedy, seed, temperature, top_k, top_p, eos_token_id, pad_token_id")
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    if py_type(options[0]) is not g[].bi.bool:
        raise Error("generation: greedy must be bool")
    s.greedy = Bool(py=options[0])
    # DEFAULT OFF: this explicit stream policy changes seeded output relative
    # to the legacy Float64 selector. Route admission never selects arithmetic.
    # Host fallback and resident device sampling share the same compensated
    # Float32 implementation. Cross-vendor identity/quality remain unqualified.
    s.sampling_policy = 1 if (not s.greedy and
        getenv("MOJOLEARN_EXPERIMENT_RESIDENT_SAMPLING_FP32", "0") == "1") else 0
    if not s.greedy and options[1] is None:
        raise Error("generation: greedy=True is the default decoding; greedy=False (sampling) requires an explicit seed")
    if options[1] is not None:
        if py_type(options[1]) is not g[].bi.int or Bool(py=options[1] < 0) or Bool(py=options[1] >= py_call(g[].bi.int, "18446744073709551616")):
            raise Error("generation: seed must be an integer in [0, 2**64)")
        var halves = py_call(g[].bi.divmod, options[1], py_call(g[].bi.int, "4294967296"))
        s.seed = (UInt64(Int(py=halves[0])) << 32) | UInt64(Int(py=halves[1]))
    s.temperature = Float32(Float64(py=options[2]))
    s.top_k = Int(py=options[3])
    s.top_p = Float64(py=options[4])
    var v = Int(py=s.model.vocab_size)
    if s.temperature <= 0 or not isfinite(s.temperature) or s.top_k < 0 or s.top_k > v or s.top_p <= 0 or s.top_p > 1 or not isfinite(s.top_p):
        raise Error("generation: invalid temperature, top_k or top_p")
    if options[5] is not None:
        var values = options[5]
        if py_type(values) is g[].bi.int:
            values = py_ints([Int(py=values)])
        for i in range(len(values)):  # small-loop(eos token id arguments)
            if py_type(values[i]) is not g[].bi.int:
                raise Error("generation: eos_token_id must be integer or sequence of integers")
            var id = Int(py=values[i])
            if id < 0 or id >= v:
                raise Error("generation: EOS id is outside vocabulary")
            s.eos.append(id)
    s.pad = (s.eos[0] if len(s.eos) else 0) if options[6] is None else Int(py=options[6])
    if s.pad < 0 or s.pad >= v:
        raise Error("generation: pad id is outside vocabulary")


def causal_lm_stream_binding(mut mods: NativeModules, model: PythonObject, value: PythonObject, maximum: PythonObject, options: PythonObject) raises -> PythonObject:
    var n = _public_int(mods, maximum, "generation: max_new_tokens must be an integer")
    if n < 0:
        raise Error("generation: max_new_tokens must be nonnegative")
    var prompt = generation_ids(value, Int(py=model.vocab_size))
    var handle = GenerationHandle(mods, model, prompt, n)
    try:
        _options(mods, handle, options)
    except error:
        mods.released()
        var parallel = py_get(model._native_entries, "parallel_call")
        if parallel is not None:
            try:
                _ = py_call(parallel, "release_layers", Python.list(causal_model_origin(mods, model), causal_state_origin(mods, handle.state), model, handle.state))
                mods.released()
            except cleanup_error:
                raise Error("generation: option validation failed: " + String(error) + "; remote cleanup also failed: " + String(cleanup_error))
        raise error
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    var stream = py_call(g[].bi.object.__new__, g[].stream_cls)
    _ = _causal_stream_fields(mods, stream)
    py_setattr(stream, "_native", PythonObject(alloc=handle^))
    py_setattr(stream, "_runtime", model._runtime)
    return stream


def causal_lm_stream_binding(model: PythonObject, value: PythonObject, maximum: PythonObject, options: PythonObject) raises -> PythonObject:
    # Classical and other callers keep their per-call module checks.
    var mods = NativeModules()
    return causal_lm_stream_binding(mods, model, value, maximum, options)


def _engine_publish_stream(mut s: GenerationHandle, progress: NativeGenerationProgress, batch: Int) raises:
    if progress.wrote > 0:
        var current = generation_array([batch, progress.length], "i")
        memcpy(dest=i32_ptr(Int(py=current._addr)), src=progress.current.unsafe_ptr(), count=len(progress.current))
        _ = current
        s.current = current
        s.produced = progress.produced
    _ = progress


def _read_engine(mut mods: NativeModules, mut s: GenerationHandle, capacity: Int) raises -> PythonObject:
    _causal_array_guard()
    var batch = len(s.finished)
    var length = Int(py=s.current.shape[1])
    var owner = _engine_admission(mods, s.model, s.state, batch, length)
    var keep = List[PythonObject]()
    var address = _addr(s.current, keep)
    var progress = NativeGenerationProgress(address, batch, length, s.produced, capacity)
    # The native engine already owns the same selector workspace. Carry the
    # stream policy into it; changing policy does not require a fallback route.
    progress.sampling.use_f32_policy = s.sampling_policy == 1
    try:
        native_generation_read(owner.native, progress, capacity, s.max_new, s.finished, s.lengths,
            s.eos, s.pad, s.greedy, s.temperature, s.top_k, s.top_p, s.seed)
    except problem:
        mods.released()
        owner.publish()
        _engine_publish_stream(s, progress, batch)
        raise problem
    owner.publish()
    _engine_publish_stream(s, progress, batch)
    var out = generation_array([batch, progress.wrote], "i")
    causal_lm_tokens_row_major(progress.tokens, i32_ptr(Int(py=out._addr)), batch, progress.wrote)
    _ = out
    _ = owner
    _ = keep^
    return out



def _resident_read_cleanup(provider: PythonObject, lm: PythonObject, sessions: PythonObject) raises:
    # Failure-only best effort: one failed close must not skip other owners.
    # The caller preserves its original operation error, including if cleanup
    # cannot inspect these already-admitted owners.
    if lm is not None:
        try:
            transformer_resident_lm_close(provider, lm)
        except cleanup:
            _ = cleanup
    if sessions is not None:
        for i in range(len(sessions)):  # driver-loop: release every owned layer after failure
            try:
                _ = transformer_resident_decode_call(provider, sessions[i], 24, [], [])
            except cleanup:
                _ = cleanup


def _resident_acquire_read(mut mods: NativeModules, model: PythonObject, state: PythonObject,
                           entries: PythonObject, fields: PythonObject, kept: PythonObject,
                           key: PythonObject, batch: Int, length: Int, int15: Bool,
                           mut keep: List[PythonObject], mut lm: PythonObject,
                           mut sessions: PythonObject, mut expected: PythonObject) raises:
    # Shared by greedy and sampled reads: each PUBLIC read reloads mutable host
    # KV, even when session identities/shape match. Residency removes only the
    # intra-read transfers. Pointer identity never proves unchanged contents.
    # Assign owners before a releasing call so caller failure cleanup retains
    # every old session if load/close fails. Numerical selection is not involved.
    if kept is not None and Bool(py=kept[0] == key):
        lm = kept[1]
        sessions = kept[2]
        _resident_load_states(entries, sessions, state, keep)
        mods.released()
    else:
        if kept is not None:
            lm = kept[1]
            sessions = kept[2]
            causal_model_cache_set(mods, model, fields, "_resident_sessions", PythonObject(None))
            expected = PythonObject(None)
            _resident_close(entries, lm, sessions)
            mods.released()
            lm = PythonObject(None)
            sessions = PythonObject(None)
        var opened = _resident_open_all(mods, model, entries, state, batch, length, int15, keep)
        lm = opened[0]
        sessions = opened[1]


def _read_resident(mut mods: NativeModules, mut s: GenerationHandle, capacity: Int,
                   mut admission: NativeArrayAdmission) raises -> PythonObject:
    # One provider boundary owns the complete token block. Native session
    # reservations hold all layers; no Python metadata, callbacks or provider
    # imports enter the GPU token loop. Mutable state is admitted each read.
    var batch = len(s.finished)
    var length = Int(py=s.current.shape[1])
    _state(mods, s.model, s.state, batch, length)
    var positions = Int(py=s.state.positions)
    var entries = s.model._native_entries
    var provider = entries["native_block_operations"]
    var int15 = s.model._head_int15 is not None
    var fields = _standalone_record_fields(mods, s.model)
    var persist = (not Bool(py=py_get(fields, "_parameters_exposed", PythonObject(False)))
                   and not Bool(py=py_get(fields, "_resident_busy", PythonObject(False))))
    var key = _resident_key(mods, s.model, s.state, batch, int15)
    var kept = py_get(fields, "_resident_sessions") if persist else PythonObject(None)
    var expected = kept
    var lm = PythonObject(None)
    var sessions = PythonObject(None)
    var keep = List[PythonObject]()
    var copies = List[PythonObject]()
    var pending = generation_array([batch, capacity], "i")
    # Heap-backed descriptor avoids the documented inline-address ABI issue;
    # its native lists, plus every host buffer, remain live through release.
    var progress = List[ResidentGreedyRead]()
    progress.append(ResidentGreedyRead(s.finished, s.lengths, s.eos, s.pad, positions,
        -1 if s.model.max_positions is None else Int(py=s.model.max_positions), Int(py=s.state.max_tokens)))
    var published = False
    if persist:
        causal_model_cache_set(mods, s.model, fields, "_resident_busy", True)
    try:
        _resident_acquire_read(mods, s.model, s.state, entries, fields, kept,
            key, batch, length, int15, keep, lm, sessions, expected)
        var cached = transformer_resident_lm_read(provider, lm, sessions,
            [_addr(s.current, keep), _addr(pending, keep), 0, Int(progress.unsafe_ptr())],
            [length, capacity, Int(py=s.state.layers[0].cached_tokens)])
        mods.released()
        # Commit diagnostics before export/result allocation can fail. Forward
        # positions include a completed forward whose later picker refused.
        s.produced += progress[0].wrote
        resident_greedy_publish(progress[0], s.finished, s.lengths)
        published = True
        py_setattr(s.state, "positions", PythonObject(positions + progress[0].consumed))
        admission.recheck(generation_array_context())
        # Export once per read, not once per token. The next public operation
        # may change or release its Array metadata, so no host address is cached.
        for i in range(len(sessions)):  # driver-loop: final owned layer-state publication
            var blk = _block(mods, s.model, i)
            var st = s.state.layers[i]
            var cap = Int(py=st.max_tokens) if Int(py=blk["window"]) == 0 else Int(py=blk["window"])
            var size = batch * Int(py=blk["n_kv_heads"]) * cap * Int(py=blk["head_dim"])
            var pk = _mutable(st.k_cache, [size], keep, copies, "k_cache")
            var pv = _mutable(st.v_cache, [size], keep, copies, "v_cache")
            _ = transformer_resident_decode_call(provider, sessions[i], 22, [pk, pv], [])
            mods.released()
            admission.recheck(generation_array_context())
            _resident_publish_profile(blk, st)
            py_setattr(st, "cached_tokens", PythonObject(cached))
        _writeback(copies)
        if persist:
            causal_model_cache_set(mods, s.model, fields, "_resident_sessions", Python.tuple(key, lm, sessions))
            causal_model_cache_set(mods, s.model, fields, "_resident_busy", False)
        else:
            _resident_close(entries, lm, sessions)
            mods.released()
        var wrote = progress[0].wrote
        var out = generation_array([batch, wrote], "i")
        var current = generation_array([batch, 1], "i")
        resident_greedy_outputs(i32_ptr(_addr(pending, keep)), i32_ptr(_addr(out, keep)),
            i32_ptr(_addr(current, keep)), batch, capacity, wrote)
        s.current = current
        return out
    except error:
        mods.released()
        if not published:
            # wrote counts only completed selections. A synchronized failure
            # readback also contains the failing step's earlier-row updates.
            s.produced += progress[0].wrote
            if progress[0].progress_ready:
                resident_greedy_publish(progress[0], s.finished, s.lengths)
        # Keep the first operation error even if metadata publication or any
        # release fails. No failed resident stream/cache becomes reusable.
        s.usable = False
        try:
            py_setattr(s.state, "_usable", PythonObject(False))
            py_setattr(s.state, "positions", PythonObject(positions + progress[0].consumed))
        except publication:
            _ = publication
        if persist:
            try:
                causal_model_cache_set(mods, s.model, fields, "_resident_sessions", PythonObject(None))
            except publication:
                _ = publication
            try:
                causal_model_cache_set(mods, s.model, fields, "_resident_busy", False)
            except publication:
                _ = publication
        try:
            _resident_read_cleanup(provider, lm, sessions)
            mods.released()
        except cleanup:
            _ = cleanup
        raise error
    finally:
        _ = keep
        _ = progress
        _ = pending
        _ = sessions
        _ = lm



def _sampling_read_progress(mut s: GenerationHandle, progress: ResidentGreedyRead,
                            pending_address: Int, current: PythonObject, current_address: Int,
                            capacity: Int, positions: Int, mut accounted: Bool) raises:
    # Selection counts advance only after a complete row set. The failing
    # selection may still carry earlier-row finished/length updates.
    if not accounted:
        s.produced += progress.wrote
        accounted = True
    if progress.progress_ready:
        resident_greedy_publish(progress, s.finished, s.lengths)
    if progress.progress_ready and progress.wrote > 0:
        var source = i32_ptr(pending_address)
        var target = i32_ptr(current_address)
        for row in range(len(s.finished)):  # driver-loop: final owned token publication
            target[row] = source[row * capacity + progress.wrote - 1]
        s.current = current
    py_setattr(s.state, "positions", PythonObject(positions + progress.consumed))


def _sampling_read_state(mut mods: NativeModules, mut s: GenerationHandle, cached: Int,
                         mut copies: List[PythonObject]) raises:
    for i in range(len(s.state.layers)):  # driver-loop: one final layer metadata publication
        _resident_publish_profile(_block(mods, s.model, i), s.state.layers[i])
        py_setattr(s.state.layers[i], "cached_tokens", PythonObject(cached))
    _writeback(copies)


def _sampling_read_cache_valid(fields: PythonObject, expected: PythonObject) raises -> Bool:
    var live = py_get(fields, "_causal_live_fields", fields)
    if py_get(live, "_parameters_exposed", PythonObject(False)) is not PythonObject(False):
        # A concurrent supported exposure must never be overwritten by this
        # operation's earlier private-parameter snapshot.
        py_setitem(fields, "_parameters_exposed", PythonObject(True))
        return False
    return py_get(live, "_resident_sessions") is expected


def _generation_route_witness(route: String, capacity: Int, produced: Int,
                              policy: GenerationReadControl, resident_forward: Bool = False,
                              eligible: Bool = False, skipped: Int = 0, invalidated: Bool = False):
    # Excluded-warmup diagnostic only, never a callback or per-token observer.
    # retained_kv now proves an actual skipped reload, not merely eligibility.
    if getenv("MOJOLEARN_EXPERIMENT_GENERATION_ROUTE_TRACE", "0") == "1":
        print("MOJOLEARN_GENERATION_ROUTE route=" + route + " capacity=" + String(capacity)
            + " produced=" + String(produced) + " retained_kv=" + String(Int(skipped > 0))
            + " requested_route=" + policy.route
            + " engine_requested=" + policy.engine_setting
            + " sampling_policy=" + String(policy.sampling_policy)
            + " greedy=" + String(Int(policy.greedy))
            + " kv_requested=" + String(Int(policy.kv.requested))
            + " kv_policy=" + policy.kv.status(route, resident_forward, eligible, skipped, invalidated)
            + " kv_uploads_skipped=" + String(skipped))


def _read_sampling_resident(mut mods: NativeModules, mut s: GenerationHandle, capacity: Int,
                            mut admission: NativeArrayAdmission) raises -> PythonObject:
    # DEFAULT OFF, UNMEASURED: one device sampling read, with the same policy
    # used by host fallback. Live KV plus two alternating committed snapshots
    # cost roughly 3x KV storage and one full cache copy per successful forward. This is not a
    # performance claim or permission to promote without vendor qualification.
    var batch = len(s.finished)
    var length = Int(py=s.current.shape[1])
    _state(mods, s.model, s.state, batch, length)
    var positions = Int(py=s.state.positions)
    var entries = s.model._native_entries
    var provider = entries["native_block_operations"]
    var int15 = s.model._head_int15 is not None
    var fields = _standalone_record_fields(mods, s.model)
    var persist = not Bool(py=py_get(fields, "_resident_busy", PythonObject(False)))
    var key = _resident_key(mods, s.model, s.state, batch, int15)
    var kept = py_get(fields, "_resident_sessions") if persist else PythonObject(None)
    var expected = kept
    var lm = PythonObject(None)
    var sessions = PythonObject(None)
    var keep = List[PythonObject]()
    var copies = List[PythonObject]()
    var pending = generation_array([batch, capacity], "i")
    var current = generation_array([batch, 1], "i")
    var pending_address = _addr(pending, keep)
    var current_address = _addr(current, keep)
    var keys = List[Int]()
    var values = List[Int]()
    # Pin and admit all recovery destinations before the GIL-release window.
    # No Python metadata lookup or deferred owner acquisition occurs on failure.
    for i in range(len(s.state.layers)):  # driver-loop: one read's KV destination admission
        var blk = _block(mods, s.model, i)
        var st = s.state.layers[i]
        var cap = Int(py=st.max_tokens) if Int(py=blk["window"]) == 0 else Int(py=blk["window"])
        var size = batch * Int(py=blk["n_kv_heads"]) * cap * Int(py=blk["head_dim"])
        keys.append(_mutable(st.k_cache, [size], keep, copies, "k_cache"))
        values.append(_mutable(st.v_cache, [size], keep, copies, "v_cache"))
    var progress = List[ResidentSamplingRead]()
    progress.append(ResidentSamplingRead(
        ResidentGreedyRead(s.finished, s.lengths, s.eos, s.pad, positions,
            -1 if s.model.max_positions is None else Int(py=s.model.max_positions), Int(py=s.state.max_tokens)),
        s.temperature, s.top_k, Float32(s.top_p), s.seed, s.produced,
        keys^, values^, Int(py=s.state.layers[0].cached_tokens)))
    var accounted = False
    var entered_native = False
    var native_completed = False
    if persist:
        causal_model_cache_set(mods, s.model, fields, "_resident_busy", True)
    try:
        _resident_acquire_read(mods, s.model, s.state, entries, fields, kept,
            key, batch, length, int15, keep, lm, sessions, expected)
        # Session opening/loading may release the GIL. Recheck the class
        # contract before preparing the final current-token span for execution.
        admission.recheck(generation_array_context())
        expected = py_get(py_get(fields, "_causal_live_fields", fields), "_resident_sessions")
        entered_native = True
        _ = transformer_resident_lm_sampling_read(provider, lm, sessions,
            [_addr(s.current, keep), pending_address, 0, Int(progress.unsafe_ptr())],
            [length, capacity, Int(py=s.state.layers[0].cached_tokens)])
        mods.released()
        native_completed = True
        _sampling_read_progress(s, progress[0].progress, pending_address, current,
            current_address, capacity, positions, accounted)
        admission.recheck(generation_array_context())
        if not progress[0].kv_ready:
            raise Error("generation: resident sampling KV publication did not complete")
        _sampling_read_state(mods, s, progress[0].cached_tokens, copies)
        if persist and _sampling_read_cache_valid(fields, expected):
            expected = Python.tuple(key, lm, sessions)
            causal_model_cache_set(mods, s.model, fields, "_resident_sessions", expected)
        else:
            if persist and py_get(py_get(fields, "_causal_live_fields", fields), "_resident_sessions") is expected:
                causal_model_cache_set(mods, s.model, fields, "_resident_sessions", PythonObject(None))
            _resident_close(entries, lm, sessions)
            mods.released()
        if persist:
            causal_model_cache_set(mods, s.model, fields, "_resident_busy", False)
        var wrote = progress[0].progress.wrote
        var out = generation_array([batch, wrote], "i")
        if wrote > 0:
            resident_greedy_outputs(i32_ptr(pending_address), i32_ptr(_addr(out, keep)),
                i32_ptr(current_address), batch, capacity, wrote)
        return out
    except error:
        mods.released()
        s.usable = False
        var state_valid = False
        try:
            # Failed native calls still require the post-GIL metadata recheck.
            # This checks class descriptors, not public instance identities.
            # Stream state is private to GenerationHandle; reads hold s.busy,
            # and checkpoints return copies rather than mutable state owners.
            _sampling_read_progress(s, progress[0].progress, pending_address, current,
                current_address, capacity, positions, accounted)
            admission.recheck(generation_array_context())
            if entered_native and progress[0].kv_ready:
                _sampling_read_state(mods, s, progress[0].cached_tokens, copies)
                state_valid = native_completed or not progress[0].failure_is_forward
        except publication:
            _ = publication
        if not state_valid:
            try:
                py_setattr(s.state, "_usable", PythonObject(False))
            except publication:
                _ = publication
        # A device/context loss can prevent committed-KV readback. Unlike the
        # old per-step host export, those bytes then cannot be recovered; mark
        # state unusable, never claim stale/partial KV is valid, retain first error.
        if persist:
            try:
                _ = _sampling_read_cache_valid(fields, expected)
                if py_get(py_get(fields, "_causal_live_fields", fields), "_resident_sessions") is expected:
                    causal_model_cache_set(mods, s.model, fields, "_resident_sessions", PythonObject(None))
                causal_model_cache_set(mods, s.model, fields, "_resident_busy", False)
            except publication:
                _ = publication
        try:
            _resident_read_cleanup(provider, lm, sessions)
            mods.released()
        except cleanup:
            _ = cleanup
        raise error
    finally:
        _ = keep
        _ = copies
        _ = progress
        _ = pending
        _ = current
        _ = sessions
        _ = lm


def _read(mut mods: NativeModules, mut s: GenerationHandle, count: Int, mut admission: Optional[NativeArrayAdmission]) raises -> PythonObject:
    # `admission` is set by the first decode step: ONE full Array guard per
    # call. Each forward may release the GIL, so each is followed by exactly
    # one post-release re-check against the admitted snapshot.
    if count < 1:
        raise Error("generation: read count must be positive")
    if not s.usable:
        raise Error("generation: stream lost after a failed call; restore a checkpoint")
    if s.busy:
        raise Error("generation: stream is busy")
    var b = len(s.finished)
    var v = Int(py=s.model.vocab_size)
    var capacity = min(count, s.max_new - s.produced)
    var read_control = GenerationReadControl(capacity, s.sampling_policy, s.greedy)
    s.busy = True
    var read_plan: Optional[LegacyGenerationReadPlan] = None
    var failed = False
    try:
        var legacy_resident = False
        if (read_control.route == "auto" or read_control.route == "resident") and capacity > 0 and not s.done() and s.greedy:
            var entries = s.model._native_entries
            if (Bool(py=py_get(entries, "resident_read_v1", PythonObject(False)))
                    and py_get(entries, "native_block_operations") is not None
                    and py_get(entries, "parallel_call") is None
                    and _resident_forward_ok(mods, s.model, s.state)):
                admission = NativeArrayAdmission(generation_array_context())
                # With only one possible forward there is no skipped
                # intermediate export/reload, so storage coupling is irrelevant.
                if capacity == 1 or _resident_read_disjoint_state(mods, s.model, s.state):
                    var result = _read_resident(mods, s, capacity, admission.value())
                    _generation_route_witness("resident_greedy", capacity, s.produced, read_control)
                    return result
                # Keep the original resident-forward host loop: another engine
                # does not establish the missing cross-layer alias contract.
                legacy_resident = True
        if (read_control.route == "auto" or read_control.route == "resident") and capacity > 0 and not s.done() and not s.greedy and s.sampling_policy == 1:
            var entries = s.model._native_entries
            var fields = _standalone_record_fields(mods, s.model)
            if (Bool(py=py_get(entries, "resident_sampling_v1", PythonObject(False)))
                    and py_get(entries, "native_block_operations") is not None
                    and py_get(entries, "parallel_call") is None
                    and not Bool(py=py_get(fields, "_parameters_exposed", PythonObject(False)))
                    and _resident_forward_ok(mods, s.model, s.state)):
                admission = NativeArrayAdmission(generation_array_context())
                # Alias refusal changes transport only. The fallback below
                # still selects the stream's checksum-bound FP32 policy.
                if capacity == 1 or _resident_read_disjoint_state(mods, s.model, s.state):
                    var result = _read_sampling_resident(mods, s, capacity, admission.value())
                    _generation_route_witness("resident_sampling", capacity, s.produced, read_control)
                    return result
                legacy_resident = True
        if capacity > 0 and not s.done() and read_control.route == "resident":
            raise Error("generation: explicit resident route requires compatible private disjoint state and greedy or FP32 sampling")
        if (capacity > 0 and not s.done() and not legacy_resident
                and (read_control.route == "native_engine" or (read_control.route == "auto"
                    and _use_native_generation_engine(mods, s.model, s.state)))):
            var result = _read_engine(mods, s, capacity)
            _generation_route_witness("native_engine", capacity, s.produced, read_control)
            return result
        # Only the legacy fallback needs host token staging. Resident/native
        # engines return their final native result without this unused buffer.
        var tokens = List[Int32](capacity=b * capacity)
        var wrote = 0
        var snapshot: Optional[GenerationOperationSnapshot] = None
        var token_a = PythonObject(None)
        var token_b = PythonObject(None)
        # Only this host-selection fallback needs scratch. Retain it across
        # tokens of this read; the sampler overwrites all used words and greedy
        # never sizes it. Resident/native routes have already returned above.
        var sampling = GenerationSamplingScratch()
        sampling.use_f32_policy = s.sampling_policy == 1
        # Upload retention is independent of greedy/Float64/FP32 selection.
        # The resident fast path already retains KV; only this existing legacy
        # forward path can remove a redundant upload, after ownership admission.
        var reuse_sampling_uploads = read_control.kv.may_retain()
        # Stream prompts/current tokens are copied by generation_ids; state is
        # allocated privately and resume/checkpoint copy its buffers. The busy
        # stream owns those bytes for this operation. This follows the resident
        # greedy read's ownership contract, not a promise for concurrent writes
        # through manually exposed internal storage or mutable model weights.
        while wrote < capacity and not s.done():  # driver-loop: over decode steps, one model forward each
            if not admission:
                admission = NativeArrayAdmission(generation_array_context())
            var l = Int(py=s.current.shape[1])
            var logits = _run_prepared(mods, s.model, s.current, s.state, snapshot, read_plan, s.produced > 0, True, reuse_sampling_uploads and not legacy_resident)
            # Post-release re-check for this forward's GIL-release window (not redundant).
            admission.value().recheck(generation_array_context())
            var raw = f32_ptr(Int(py=logits._addr))
            _ = logits  # keep the Python object alive through the native call above
            var next = PythonObject(None)
            if read_plan:
                # Alternate private destinations: a failed pick never overwrites
                # the stream's last committed current token. Prompt stays owned
                # separately, and public checkpoint/output paths copy the bytes.
                if wrote % 2 == 0:
                    if token_a is None:
                        token_a = generation_array([b, 1], "i")
                    next = token_a
                else:
                    if token_b is None:
                        token_b = generation_array([b, 1], "i")
                    next = token_b
            else:
                next = generation_array([b, 1], "i")
            var dst = i32_ptr(Int(py=next._addr))
            _ = next  # keep the Python object alive through the native call above
            causal_lm_pick_step(raw, dst, b, l, v, s.finished, s.lengths, s.eos, s.pad, s.greedy,
                                s.temperature, s.top_k, s.top_p, s.seed, s.produced, tokens, sampling)
            _ = next  # keep the Python object alive through the native call above
            _ = logits  # keep the Python object alive through the native call above
            # `raw` points into `logits`: keep it alive through the pick (Mojo
            # destroys a local at its last use, and nothing else owns it).
            _ = logits^
            s.current = next
            s.produced += 1
            wrote += 1
        var out = generation_array([b, wrote], "i")
        if wrote == 0:
            return out
        var dst = i32_ptr(Int(py=out._addr))
        _ = out  # keep the Python object alive through the native call above
        causal_lm_tokens_row_major(tokens, dst, b, wrote)
        _ = out  # keep the Python object alive through the native call above
        _generation_route_witness("legacy_forward", capacity, s.produced, read_control,
            Bool(read_plan) and read_plan.value().resident,
            Bool(read_plan) and read_plan.value().resident_reuse,
            read_plan.value().resident_uploads_skipped if read_plan else 0,
            Bool(read_plan) and read_plan.value().resident_invalidated)
        return out
    except error:
        mods.released()
        failed = True
        s.usable = False
        raise error
    finally:
        try:
            # A sampler/descriptor/result-allocation failure follows a
            # successful exported forward; its valid sessions may be retained.
            _legacy_resident_finish(mods, s.model, read_plan)
        except cleanup:
            if not failed:
                s.usable = False
                raise cleanup
        finally:
            s.busy = False


def _read(mut mods: NativeModules, mut s: GenerationHandle, count: Int) raises -> PythonObject:
    var admission: Optional[NativeArrayAdmission] = None
    return _read(mods, s, count, admission)


def _decode_chunk(mut s: GenerationHandle, ids: PythonObject) raises -> PythonObject:
    _causal_array_guard()
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    var n = Int(py=ids.shape[1])
    var raw = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=ids._addr))
    var values = py_ints(causal_lm_drop_eos(raw, n, s.eos))
    _ = py_call(s.pending_bytes.extend, tokenizer_vocab_decode(s.tokenizer, values, PythonObject(True), PythonObject("replace")))
    var size = len(s.pending_bytes)
    # The CPython array constructor copies the pending bytes to native storage.
    var pending = py_call(g[].arrays, "B", s.pending_bytes)
    var cut = causal_lm_utf8_cut(MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=pending.buffer_info()[0])), size, s.done())
    _ = pending  # keep the Python object alive through the native call above
    _ = pending^
    var prefix = s.pending_bytes[py_call(g[].bi.slice, PythonObject(0), PythonObject(cut))]
    var remaining = s.pending_bytes[py_call(g[].bi.slice, PythonObject(cut), PythonObject(None))]
    s.pending_bytes = remaining
    return tokenizer_decode_utf8(prefix, PythonObject("replace"))


def _causal_stream_fields(mut mods: NativeModules, stream: PythonObject) raises -> PythonObject:
    return model_record_fields(mods, stream, _causal_context()["generation_classes"][3], Python.tuple("_native", "_runtime"))


def causal_lm_read_binding(mut mods: NativeModules, stream: PythonObject, count: PythonObject) raises -> PythonObject:
    return causal_lm_read_binding(mods, stream, count, _causal_stream_fields(mods, stream)["_native"])


def causal_lm_read_binding(stream: PythonObject, count: PythonObject) raises -> PythonObject:
    # Classical and other callers keep their per-call module checks.
    var mods = NativeModules()
    return causal_lm_read_binding(mods, stream, count)


def causal_lm_read_binding(mut mods: NativeModules, stream: PythonObject, count: PythonObject, native: PythonObject) raises -> PythonObject:
    var owner = native.downcast_value_ptr[GenerationHandle]()
    var ids = _read(mods, owner[], _public_int(mods, count, "generation: max_tokens must be an integer"))
    if owner[].tokenizer is not None:
        var windowed_result3 = _decode_chunk(owner[], ids)
        mods.released()
        return windowed_result3^
    return ids


def causal_lm_read_binding(stream: PythonObject, count: PythonObject, native: PythonObject) raises -> PythonObject:
    # Classical and other callers keep their per-call module checks.
    var mods = NativeModules()
    return causal_lm_read_binding(mods, stream, count, native)


def causal_lm_info_binding(mut mods: NativeModules, stream: PythonObject) raises -> PythonObject:
    return causal_lm_info_binding(mods, stream, _causal_stream_fields(mods, stream)["_native"])


def causal_lm_info_binding(stream: PythonObject) raises -> PythonObject:
    # Classical and other callers keep their per-call module checks.
    var mods = NativeModules()
    return causal_lm_info_binding(mods, stream)


def causal_lm_info_binding(mut mods: NativeModules, stream: PythonObject, native: PythonObject) raises -> PythonObject:
    var s = native.downcast_value_ptr[GenerationHandle]()
    var result = GENERATION_BOUNDARY.get_or_create_ptr()[].bi.dict()
    py_setitem(result, "done", PythonObject(s[].done()))
    py_setitem(result, "generated_tokens", PythonObject(s[].produced))
    py_setitem(result, "lengths", py_ints(s[].lengths))
    py_setitem(result, "usable", PythonObject(s[].usable))
    return result


def causal_lm_info_binding(stream: PythonObject, native: PythonObject) raises -> PythonObject:
    # Classical and other callers keep their per-call module checks.
    var mods = NativeModules()
    return causal_lm_info_binding(mods, stream, native)


def _generate_complete(mut mods: NativeModules, model: PythonObject, value: PythonObject, maximum: PythonObject, mut s: GenerationHandle) raises -> PythonObject:
    if s.greedy and len(s.eos) == 0:
        var resident = causal_lm_resident_binding(mods, model, s.prompt, PythonObject(s.max_new), PythonObject(None), s.state)
        if resident is not None:
            return resident
    var admission: Optional[NativeArrayAdmission] = None
    var continuation = _read(mods, s, max(1, s.max_new), admission)
    # Per call, before the continuation is joined: the decode loop's admitted
    # snapshot makes this a cheap re-check; the engine route and an empty loop
    # leave no admission and keep the full guard.
    if admission:
        admission.value().recheck(generation_array_context())
    else:
        _causal_array_guard()
    var b = len(s.lengths)
    var l = Int(py=s.prompt.shape[1])
    var n = Int(py=continuation.shape[1])
    var result = generation_array([b, l + n], "i")
    var dest = i32_ptr(Int(py=result._addr))
    _ = result  # keep the Python object alive through the native call above
    var prompt = i32_ptr(Int(py=s.prompt._addr))
    var tail = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=continuation._addr))
    _ = continuation  # keep the Python object alive through the native call above
    causal_lm_join_continuation(dest, prompt, tail, b, l, n)
    _ = result  # keep the Python object alive through the native call above
    _ = continuation  # keep the Python object alive through the native call above
    _ = continuation^
    return result


def causal_lm_generate_binding(mut mods: NativeModules, model: PythonObject, value: PythonObject, maximum: PythonObject, options: PythonObject) raises -> PythonObject:
    var stream = causal_lm_stream_binding(mods, model, value, maximum, options)
    var s = _causal_stream_fields(mods, stream)["_native"].downcast_value_ptr[GenerationHandle]()
    var parallel = py_get(model._native_entries, "parallel_call")
    var result = PythonObject(None)
    try:
        result = _generate_complete(mods, model, value, maximum, s[])
    except error:
        mods.released()
        if parallel is not None:
            try:
                _ = py_call(parallel, "release_layers", Python.list(causal_model_origin(mods, model), causal_state_origin(mods, s[].state), model, s[].state))
                mods.released()
            except cleanup_error:
                raise Error("CausalLM: generation failed: " + String(error) + "; remote cleanup also failed: " + String(cleanup_error))
        raise error
    if parallel is not None:
        _ = py_call(parallel, "release_layers", Python.list(causal_model_origin(mods, model), causal_state_origin(mods, s[].state), model, s[].state))
        mods.released()
    return result


def causal_lm_generate_binding(model: PythonObject, value: PythonObject, maximum: PythonObject, options: PythonObject) raises -> PythonObject:
    # Classical and other callers keep their per-call module checks.
    var mods = NativeModules()
    return causal_lm_generate_binding(mods, model, value, maximum, options)


def causal_lm_text_stream_binding(mut mods: NativeModules, model: PythonObject, tokenizer: PythonObject, text: PythonObject, maximum: PythonObject, options: PythonObject) raises -> PythonObject:
    var ids = tokenizer_vocab_encode(tokenizer, text, PythonObject(True), PythonObject(False), PythonObject(False))
    mods.released()
    var prompt = generation_array([1, len(ids)], "i")
    var ptr = i32_ptr(Int(py=prompt._addr))
    _ = prompt  # keep the Python object alive through the native call above
    # The CPython array constructor unpacks the id list to native int64 storage.
    var packed = py_call(GENERATION_BOUNDARY.get_or_create_ptr()[].arrays, "q", ids)
    mods.released()
    causal_lm_store_ids(ptr, MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(py=packed.buffer_info()[0])), len(ids))
    _ = packed  # keep the Python object alive through the native call above
    _ = prompt  # keep the Python object alive through the native call above
    _ = packed^
    var stream = causal_lm_stream_binding(mods, model, prompt, maximum, options)
    var owner = _causal_stream_fields(mods, stream)["_native"].downcast_value_ptr[GenerationHandle]()
    owner[].tokenizer = tokenizer
    return stream


def causal_lm_text_stream_binding(model: PythonObject, tokenizer: PythonObject, text: PythonObject, maximum: PythonObject, options: PythonObject) raises -> PythonObject:
    # Classical and other callers keep their per-call module checks.
    var mods = NativeModules()
    return causal_lm_text_stream_binding(mods, model, tokenizer, text, maximum, options)


def causal_lm_generate_text_binding(mut mods: NativeModules, model: PythonObject, tokenizer: PythonObject, text: PythonObject, maximum: PythonObject, options: PythonObject) raises -> PythonObject:
    var stream = causal_lm_text_stream_binding(mods, model, tokenizer, text, maximum, options)
    var owner = _causal_stream_fields(mods, stream)["_native"].downcast_value_ptr[GenerationHandle]()
    var parallel = py_get(model._native_entries, "parallel_call")
    var result = PythonObject(None)
    try:
        var ids = _read(mods, owner[], max(1, owner[].max_new))
        result = _decode_chunk(owner[], ids)
        mods.released()
    except error:
        mods.released()
        if parallel is not None:
            try:
                _ = py_call(parallel, "release_layers", Python.list(causal_model_origin(mods, model), causal_state_origin(mods, owner[].state), model, owner[].state))
                mods.released()
            except cleanup_error:
                raise Error("CausalLM: text generation failed: " + String(error) + "; remote cleanup also failed: " + String(cleanup_error))
        raise error
    if parallel is not None:
        _ = py_call(parallel, "release_layers", Python.list(causal_model_origin(mods, model), causal_state_origin(mods, owner[].state), model, owner[].state))
        mods.released()
    return result


def causal_lm_generate_text_binding(model: PythonObject, tokenizer: PythonObject, text: PythonObject, maximum: PythonObject, options: PythonObject) raises -> PythonObject:
    # Classical and other callers keep their per-call module checks.
    var mods = NativeModules()
    return causal_lm_generate_text_binding(mods, model, tokenizer, text, maximum, options)


def _array_bytes(value: PythonObject) raises -> PythonObject:
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    if py_type(value) is g[].array_cls:
        var captured = generation_array_snapshot(value)
        return py_call(captured._mv.tobytes, "C")
    return py_call(generation_buffer_view(value).tobytes, "C")


def _hash() -> Sha256:
    return Sha256()


def _hash_scalar(mut hash: Sha256, value: PythonObject) raises:
    sha256_update_text(hash, value)
    hash.update_text(String("\n"))


def _fingerprint(mut mods: NativeModules, model: PythonObject) raises -> PythonObject:
    _causal_array_guard()
    var h = _hash()
    _hash_scalar(h, model.kind)
    _hash_scalar(h, model.numeric_profile)
    _hash_scalar(h, model.norm_eps)
    _hash_scalar(h, model.max_positions)
    _hash_scalar(h, model._embed.shape)
    sha256_update(h, _array_bytes(model._embed))
    if model._norm is not None:
        sha256_update(h, _array_bytes(model._norm))
    sha256_update(h, _array_bytes(model._head))
    for i in range(len(model._blocks)):  # driver-loop: over model layers
        var blk = _block(mods, model, i)
        _causal_array_guard()
        if String(model.kind) == "transformer":
            _hash_scalar(h, blk["_opts_tail"])
            _hash_scalar(h, blk["n_heads"])
            _hash_scalar(h, blk["n_kv_heads"])
            _hash_scalar(h, blk["window"])
            for j in range(len(blk["_wopt"])):  # driver-loop: over optional block weight slots
                if blk["_wopt"][j] is not None:
                    sha256_update(h, _array_bytes(blk["_wopt"][j]))
        elif String(model.kind) == "mamba2":
            _hash_scalar(h, blk["dt_limit"])
        else:
            _hash_scalar(h, model._native_entries["mamba1_profile"])
        for j in range(len(blk["_w"])):  # driver-loop: over block weight slots
            if blk["_w"][j] is not None:
                _hash_scalar(h, blk["_w"][j].shape)
                sha256_update(h, _array_bytes(blk["_w"][j]))
        if model._head_int15 is not None:
            for j in range(len(blk["_int15"])):  # driver-loop: over packed block weight slots
                if blk["_int15"][j] is not None:
                    for k in range(3):  # small-loop(int15 plane triple)
                        sha256_update(h, _array_bytes(blk["_int15"][j][k]))
    if model._head_int15 is not None:
        for k in range(3):  # small-loop(int15 head plane triple)
            sha256_update(h, _array_bytes(model._head_int15[k]))
    return PythonObject(h.hexdigest())


def _state_arrays(kind: String, profile: Int = 1) -> List[String]:
    if kind == "transformer":
        return [String("k_cache"), String("v_cache")]
    if kind == "mamba2":
        return [String("conv_window"), String("h"), String("buffer_xbc"), String("buffer_dtraw")]
    if profile == 2:
        return [String("conv_window"), String("h"), String("affine_boundary"), String("affine_a"), String("affine_b")]
    return [String("conv_window"), String("h")]


def _snapshot_hash(checkpoint: PythonObject) raises -> PythonObject:
    var h = _hash()
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    for name in [String("schema"), String("model"), String("options"), String("maximum"),  # small-loop(checkpoint scalar fields)
                 String("produced"), String("positions"), String("lengths"), String("finished"), String("text"), String("tokenizer")]:
        _hash_scalar(h, checkpoint[name])
    if String(checkpoint["schema"]) == "mojolearn.generation.v2":
        _hash_scalar(h, checkpoint["sampling_policy"])
    sha256_update(h, checkpoint["pending_bytes"])
    sha256_update(h, _array_bytes(checkpoint["prompt"]))
    sha256_update(h, _array_bytes(checkpoint["current"]))
    for i in range(len(checkpoint["layers"])):  # driver-loop: over checkpoint layers
        var layer = checkpoint["layers"][i]
        var names = py_call(g[].bi.sorted, layer.keys())
        for j in range(len(names)):  # driver-loop: over layer state field names
            _hash_scalar(h, names[j])
            var value = layer[names[j]]
            if py_type(value) is g[].array_cls:
                _hash_scalar(h, value.shape)
                _hash_scalar(h, value.dtype)
                sha256_update(h, _array_bytes(value))
            else:
                _hash_scalar(h, value)
    return PythonObject(h.hexdigest())


def causal_lm_snapshot_binding(mut mods: NativeModules, stream: PythonObject) raises -> PythonObject:
    return causal_lm_snapshot_binding(mods, stream, _causal_stream_fields(mods, stream)["_native"])


def causal_lm_snapshot_binding(stream: PythonObject) raises -> PythonObject:
    # Classical and other callers keep their per-call module checks.
    var mods = NativeModules()
    return causal_lm_snapshot_binding(mods, stream)


def causal_lm_snapshot_binding(mut mods: NativeModules, stream: PythonObject, native: PythonObject) raises -> PythonObject:
    var s = native.downcast_value_ptr[GenerationHandle]()
    if s[].busy or not s[].usable:
        raise Error("generation: cannot checkpoint a busy or failed stream")
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    var result = g[].bi.dict()
    py_setitem(result, "schema", PythonObject("mojolearn.generation.v2" if s[].sampling_policy == 1 else "mojolearn.generation.v1"))
    if s[].sampling_policy == 1:
        py_setitem(result, "sampling_policy", PythonObject("mojolearn.sampling.fp32-compensated.v1"))
    py_setitem(result, "model", _fingerprint(mods, s[].model))
    py_setitem(result, "maximum", PythonObject(s[].max_new))
    py_setitem(result, "produced", PythonObject(s[].produced))
    py_setitem(result, "positions", s[].state.positions)
    py_setitem(result, "prompt", generation_copy(s[].prompt))
    py_setitem(result, "current", generation_copy(s[].current))
    py_setitem(result, "lengths", py_ints(s[].lengths))
    py_setitem(result, "finished", py_call(g[].bi.list, py_call(g[].bi.map, g[].bi.bool, py_ints(causal_lm_flags_as_ints(s[].finished)))))
    py_setitem(result, "text", PythonObject(s[].tokenizer is not None))
    py_setitem(result, "tokenizer", PythonObject(None))
    if s[].tokenizer is not None:
        py_setitem(result, "tokenizer", tokenizer_vocab_fingerprint(s[].tokenizer))
        mods.released()
    py_setitem(result, "pending_bytes", py_call(g[].bi.bytes, s[].pending_bytes))
    var options = g[].bi.list()
    _ = py_call(options.append, s[].greedy)
    # Decimal conversion avoids truncating UInt64 through the signed Python
    # bridge. Every unsigned seed is representable by Python's native int.
    _ = py_call(options.append, py_call(g[].bi.int, String(s[].seed)))
    _ = py_call(options.append, Float64(s[].temperature))
    _ = py_call(options.append, s[].top_k)
    _ = py_call(options.append, s[].top_p)
    _ = py_call(options.append, py_ints(s[].eos))
    _ = py_call(options.append, s[].pad)
    py_setitem(result, "options", options)
    var layers = g[].bi.list()
    var kind = String(s[].model.kind)
    var profile = Int(py=s[].model._native_entries["mamba1_profile"]) if kind == "mamba1" else 1
    var arrays = _state_arrays(kind, profile)
    for i in range(len(s[].state.layers)):  # driver-loop: over model layer states
        var st = s[].state.layers[i]
        var parallel = py_get(s[].model._native_entries, "parallel_call")
        if parallel is not None:
            st = py_call(parallel, "state_snapshot", Python.list(causal_state_origin(mods, st), st))
            mods.released()
            st = causal_layer_capture(mods, st, s[].model.kind, _causal_context_for(mods, s[].model))
        var entry = g[].bi.dict()
        for j in range(len(arrays)):  # driver-loop: over layer state arrays
            py_setitem(entry, arrays[j], generation_copy(py_call(g[].bi.getattr, st, arrays[j])))
        if kind == "transformer":
            py_setitem(entry, "cached_tokens", st.cached_tokens)
        elif kind == "mamba1":
            py_setitem(entry, "absolute_position", st.absolute_position)
            py_setitem(entry, "profile_id", st.profile_id)
        else:
            py_setitem(entry, "buffered_tokens", st.buffered_tokens)
        _ = py_call(layers.append, entry)
    py_setitem(result, "layers", layers)
    py_setitem(result, "checksum", _snapshot_hash(result))
    return result


def causal_lm_snapshot_binding(stream: PythonObject, native: PythonObject) raises -> PythonObject:
    # Classical and other callers keep their per-call module checks.
    var mods = NativeModules()
    return causal_lm_snapshot_binding(mods, stream, native)


def causal_lm_resume_binding(mut mods: NativeModules, model: PythonObject, checkpoint: PythonObject, tokenizer: PythonObject) raises -> PythonObject:
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    var schema = String(checkpoint["schema"])
    if schema != "mojolearn.generation.v1" and schema != "mojolearn.generation.v2":
        raise Error("generation: unsupported checkpoint schema")
    var sampling_policy = 0
    if schema == "mojolearn.generation.v2":
        if not py_contains(checkpoint, "sampling_policy") or String(checkpoint["sampling_policy"]) != "mojolearn.sampling.fp32-compensated.v1":
            raise Error("generation: unsupported checkpoint sampling policy")
        sampling_policy = 1
    elif py_contains(checkpoint, "sampling_policy"):
        raise Error("generation: legacy checkpoint cannot declare a sampling policy")
    if String(checkpoint["model"]) != String(_fingerprint(mods, model)):
        raise Error("generation: checkpoint model weights/settings do not match")
    if String(checkpoint["checksum"]) != String(_snapshot_hash(checkpoint)):
        raise Error("generation: checkpoint checksum differs")
    if Bool(py=checkpoint["text"]) and tokenizer is None:
        raise Error("generation: a text checkpoint requires its tokenizer")
    if Bool(py=checkpoint["text"]):
        var fingerprint = tokenizer_vocab_fingerprint(tokenizer)
        mods.released()
        if not Bool(py=checkpoint["tokenizer"] == fingerprint):
            raise Error("generation: checkpoint tokenizer differs")
    var stream = causal_lm_stream_binding(mods, model, checkpoint["prompt"], checkpoint["maximum"], checkpoint["options"])
    var s = _causal_stream_fields(mods, stream)["_native"].downcast_value_ptr[GenerationHandle]()
    # Resume follows the checksum-bound policy, never the ambient toggle. This
    # also preserves v1's legacy policy when the experiment is enabled later.
    s[].sampling_policy = sampling_policy
    try:
        if sampling_policy == 1 and s[].greedy:
            raise Error("generation: sampled checkpoint policy requires greedy=False")
        s[].produced = Int(py=checkpoint["produced"])
        if s[].produced < 0 or s[].produced > s[].max_new:
            raise Error("generation: invalid checkpoint output position")
        s[].current = generation_ids(checkpoint["current"], Int(py=model.vocab_size))
        var b = len(s[].lengths)
        if Int(py=s[].current.shape[0]) != b or len(checkpoint["lengths"]) != b or len(checkpoint["finished"]) != b:
            raise Error("generation: checkpoint batch counts differ")
        if s[].produced > 0 and Int(py=s[].current.shape[1]) != 1:
            raise Error("generation: continuation checkpoint must hold one pending token per row")
        var position = 0 if s[].produced == 0 else Int(py=s[].prompt.shape[1]) + s[].produced - 1
        if Int(py=checkpoint["positions"]) != position:
            raise Error("generation: checkpoint cache position disagrees with output count")
        py_setattr(s[].state, "positions", PythonObject(position))
        # The CPython array constructor unpacks the per-row lists to native storage.
        var packed_lengths = py_call(g[].arrays, "q", checkpoint["lengths"])
        mods.released()
        var packed_finished = py_call(g[].arrays, "b", py_call(g[].bi.map, g[].bi.bool, checkpoint["finished"]))
        mods.released()
        causal_lm_restore_rows(
            MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(py=packed_lengths.buffer_info()[0])),
            MutPointer[Int8, MutUntrackedOrigin](unsafe_from_address=Int(py=packed_finished.buffer_info()[0])),
            b, s[].produced, s[].lengths, s[].finished,
        )
        _ = packed_lengths  # keep the Python object alive through the native call above
        _ = packed_finished  # keep the Python object alive through the native call above
        _ = packed_lengths^
        _ = packed_finished^
        if len(checkpoint["layers"]) != len(s[].state.layers):
            raise Error("generation: checkpoint layer count differs")
        var kind = String(model.kind)
        var profile = Int(py=model._native_entries["mamba1_profile"]) if kind == "mamba1" else 1
        var arrays = _state_arrays(kind, profile)
        for i in range(len(s[].state.layers)):  # driver-loop: over model layer states
            var dest = s[].state.layers[i]
            var parallel = py_get(model._native_entries, "parallel_call")
            if parallel is not None:
                dest = py_call(parallel, "state_snapshot", Python.list(causal_state_origin(mods, dest), dest))
                mods.released()
                dest = causal_layer_capture(mods, dest, model.kind, _causal_context_for(mods, model))
            var source = checkpoint["layers"][i]
            for j in range(len(arrays)):  # driver-loop: over layer state arrays
                var expected = py_call(g[].bi.getattr, dest, arrays[j])
                var supplied = source[arrays[j]]
                if py_type(supplied) is not g[].array_cls or String(supplied.dtype) != "<f4" or not Bool(py=supplied.shape == expected.shape):
                    raise Error("generation: checkpoint state array shape or dtype differs")
                _ = py_call(g[].bi.setattr, dest, arrays[j], generation_copy(supplied))
            if kind == "transformer":
                if Int(py=source["cached_tokens"]) != position:
                    raise Error("generation: checkpoint layer cache position differs")
                py_setattr(dest, "cached_tokens", PythonObject(position))
            elif kind == "mamba1":
                if Int(py=source["profile_id"]) != profile or Int(py=source["absolute_position"]) != position:
                    raise Error("generation: checkpoint Mamba profile/position differs")
                py_setattr(dest, "absolute_position", PythonObject(position))
            else:
                if Int(py=source["buffered_tokens"]) != position % 256:
                    raise Error("generation: checkpoint Mamba2 buffered length differs")
                py_setattr(dest, "buffered_tokens", source["buffered_tokens"])
            if parallel is not None:
                causal_layer_publish(mods, dest)
                _ = py_call(parallel, "state_publish", Python.list(causal_state_origin(mods, s[].state.layers[i]), causal_state_origin(mods, dest), s[].state.layers[i]))
                mods.released()
        s[].tokenizer = tokenizer if Bool(py=checkpoint["text"]) else PythonObject(None)
        s[].pending_bytes = py_call(g[].bi.bytearray, checkpoint["pending_bytes"])
        return stream
    except error:
        mods.released()
        # These caches belong only to this attempted restore. Existing live
        # streams retain their own worker tokens and are never touched.
        var parallel = py_get(model._native_entries, "parallel_call")
        if parallel is not None:
            try:
                _ = py_call(parallel, "release_layers", Python.list(causal_model_origin(mods, model), causal_state_origin(mods, s[].state), model, s[].state))
                mods.released()
            except cleanup_error:
                raise Error("generation: checkpoint restore failed: " + String(error) + "; remote cleanup also failed: " + String(cleanup_error))
        raise error


def causal_lm_resume_binding(model: PythonObject, checkpoint: PythonObject, tokenizer: PythonObject) raises -> PythonObject:
    # Classical and other callers keep their per-call module checks.
    var mods = NativeModules()
    return causal_lm_resume_binding(mods, model, checkpoint, tokenizer)


def causal_state_init_binding(mut mods: NativeModules, state: PythonObject, values: PythonObject, context: PythonObject) raises -> PythonObject:
    causal_public_connect(context)
    var fields = _causal_state_fields(mods, state)
    var bi = mods.module("builtins")
    for i in range(2):  # small-loop(public state dimension arguments)
        if py_type(values[i]) is not bi.str and not neural_native_scalar(mods, values[i], context):
            raise Error("TypeError: CausalLMState dimensions require native scalar values")
    if py_type(values[2]) is not bi.list and py_type(values[2]) is not bi.tuple:
        raise Error("TypeError: CausalLMState layers require an ordinary list or tuple")
    var batch = py_call(bi.int, values[0])
    var capacity = py_call(bi.int, values[1])
    var layers = py_call(bi.list, values[2])
    py_setitem(fields, "batch_size", batch)
    py_setitem(fields, "max_tokens", capacity)
    py_setitem(fields, "layers", layers)
    py_setitem(fields, "positions", PythonObject(0))
    py_setitem(fields, "_owner", values[3])
    py_setitem(fields, "_usable", PythonObject(True))
    return PythonObject(None)


def causal_state_repr_binding(mut mods: NativeModules, state: PythonObject) raises -> PythonObject:
    var fields = _causal_state_fields(mods, state)
    var bi = mods.module("builtins")
    for name in ["batch_size", "max_tokens", "positions"]:  # small-loop(displayed scalar state metadata)
        neural_plain_value(mods, fields[name], 0, _causal_context())
    var layers = fields["layers"]
    if py_type(layers) is not bi.list and py_type(layers) is not bi.tuple:
        raise Error("TypeError: CausalLMState layers require an ordinary list or tuple")
    return PythonObject("CausalLMState(batch_size=" + String(fields["batch_size"]) +
        ", max_tokens=" + String(fields["max_tokens"]) + ", layers=" + String(len(layers)) +
        ", positions=" + String(fields["positions"]) + ")")


def causal_lm_repr_binding(mut mods: NativeModules, model: PythonObject) raises -> PythonObject:
    var fields = _causal_model_fields(mods, model)
    var plan = model_plan_snapshot(mods, fields["plan"])
    var bi = mods.module("builtins")
    for name in ["n_layers", "d_model", "vocab_size", "weight_format", "device"]:  # small-loop(displayed native model metadata)
        neural_plain_value(mods, fields[name], 0, _causal_context())
    return PythonObject("CausalLM(model_type=" + String(py_call(bi.repr, plan["model_type"])) +
        ", layers=" + String(fields["n_layers"]) + ", d_model=" + String(fields["d_model"]) +
        ", vocab=" + String(fields["vocab_size"]) + ", weight_format=" + String(py_call(bi.repr, fields["weight_format"])) +
        ", device=" + String(py_call(bi.repr, fields["device"])) + ")")


def _causal_owner_unusable(mut mods: NativeModules, args_ptr: PyObjectPtr) -> Bool:
    # Error classification must not invoke a replaced property, __getattr__,
    # __dict__ descriptor or boolean hook on the rejected public owner.
    try:
        var args = PythonObject(from_borrowed=args_ptr)
        if len(args) == 0:
            return False
        var fields = neural_record_fields(mods, args[0], Python.tuple("_closed", "_failed"))
        return _truth(py_get(fields, "_closed")) or _truth(py_get(fields, "_failed"))
    except:
        mods.released()
        return False


def causal_public_c[operation: Int](py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    var mods = NativeModules()
    var model = PythonObject(None)
    var state = PythonObject(None)
    try:
        var args = PythonObject(from_borrowed=args_ptr)
        var result = PythonObject(None)
        var options = PythonObject(None)
        var checkpoint = PythonObject(None)
        var last_logits = PythonObject(None)
        var stream_native = PythonObject(None)
        comptime if operation == 1 or operation == 2 or operation == 3 or operation == 4 or operation == 7 or operation == 8 or operation == 9 or operation == 10 or operation == 12 or operation == 13:
            model = causal_model_capture(mods, args[0], _causal_context())
        comptime if operation == 2:
            state = causal_state_capture(mods, args[1], model, _causal_context())
        elif operation == 3:
            state = causal_state_capture(mods, args[2], model, _causal_context())
        elif operation == 10:
            state = causal_state_capture(mods, args[4], model, _causal_context())
            if args[3] is not None:
                last_logits = generation_array_snapshot(args[3]) if py_type(args[3]) is GENERATION_BOUNDARY.get_or_create_ptr()[].array_cls else generation_buffer_view(args[3])
        elif operation == 5 or operation == 6 or operation == 11:
            stream_native = _causal_stream_fields(mods, args[0])["_native"]
            comptime if operation == 5 or operation == 11:
                var stream = stream_native.downcast_value_ptr[GenerationHandle]()
                # Every public read/checkpoint captures current caller metadata
                # once; prior calls already published their state progress.
                model = causal_model_capture(mods, causal_model_origin(mods, stream[].model), _causal_context())
                state = causal_state_capture(mods, causal_state_origin(mods, stream[].state), model, _causal_context())
                stream[].model = model
                stream[].state = state
        comptime if operation == 4 or operation == 7 or operation == 8 or operation == 9:
            var raw_options = args[4] if operation == 8 or operation == 9 else args[3]
            neural_plain_value(mods, raw_options, 0, _causal_context())
            options = neural_metadata_snapshot(mods, raw_options)
        comptime if operation == 12:
            checkpoint = causal_checkpoint_capture(mods, args[1], _causal_context())
        comptime if operation == 0:
            result = causal_lm_admit_binding(mods, args[0], args[1], args[2], args[3])
        elif operation == 1:
            result = causal_lm_allocate_binding(mods, model, args[1], args[2])
            causal_state_publish(mods, result)
            result = causal_state_origin(mods, result)
        elif operation == 2:
            result = causal_lm_reset_binding(mods, model, state)
            result = causal_state_origin(mods, result)
        elif operation == 3:
            result = causal_lm_forward_binding(mods, model, args[1], state, args[3])
        elif operation == 4:
            result = causal_lm_stream_binding(mods, model, args[1], args[2], options)
        elif operation == 5:
            result = causal_lm_read_binding(mods, args[0], args[1], stream_native)
        elif operation == 6:
            result = causal_lm_info_binding(mods, args[0], stream_native)
        elif operation == 7:
            result = causal_lm_generate_binding(mods, model, args[1], args[2], options)
        elif operation == 8:
            result = causal_lm_text_stream_binding(mods, model, args[1], args[2], args[3], options)
        elif operation == 9:
            result = causal_lm_generate_text_binding(mods, model, args[1], args[2], args[3], options)
        elif operation == 10:
            result = causal_lm_resident_binding(mods, model, args[1], args[2], last_logits, state)
        elif operation == 11:
            result = causal_lm_snapshot_binding(mods, args[0], stream_native)
        elif operation == 12:
            result = causal_lm_resume_binding(mods, model, checkpoint, args[2])
        elif operation == 13:
            result = causal_lm_parameters_binding(mods, model)
        elif operation == 14:
            result = causal_lm_blocks_binding(mods, args[0])
        elif operation == 15:
            result = causal_state_init_binding(mods, args[0], args[1], args[2])
        elif operation == 16:
            result = causal_lm_repr_binding(mods, args[0])
        else:
            result = causal_state_repr_binding(mods, args[0])
        comptime if operation == 4 or operation == 8 or operation == 12:
            # Restored and newly allocated state owners must match the native
            # continuation before the next public operation recaptures them.
            var stream = _causal_stream_fields(mods, result)["_native"].downcast_value_ptr[GenerationHandle]()
            state = stream[].state
        causal_state_publish(mods, state)
        causal_model_publish(mods, model)
        return result.steal_data()
    except error:
        var message = String(error)
        try:
            causal_state_publish(mods, state)
            causal_model_publish(mods, model)
        except publication_error:
            message += "; native state publication also failed: " + String(publication_error)
            return raise_python_exception(Error(message), ExceptionType("PyExc_RuntimeError"))
        if message.startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        if message.startswith("TypeError:"):
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        if _causal_owner_unusable(mods, args_ptr):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
