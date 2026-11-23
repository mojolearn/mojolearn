# SPDX-License-Identifier: Apache-2.0
"""Whole-model runtime owned by Mojo, over existing compiled kernel entries.

The selected CPU/GPU extensions are bound once at model construction. This
module never invokes model, block, Array or backend Python methods. State,
allocation, layer traversal, token traversal and selection are native. Calling
a compiled sibling extension through CPython's native function ABI changes no
arithmetic and uses no interpreter callback; targets are checked at admission.
"""
from core.causal_lm_entry_table_sessions import (
    causal_sessions_ref, causal_sessions_call, causal_sessions_tables_admit, CAUSAL_SESSIONS_TRANSFORMER, CAUSAL_SESSIONS_MAMBA,
    CAUSAL_SESSION_DECODE_CREATE, CAUSAL_SESSION_DECODE_OPEN, CAUSAL_SESSION_DECODE_OPEN15,
    CAUSAL_SESSION_DECODE_EXPORT, CAUSAL_SESSION_DECODE_LOAD, CAUSAL_SESSION_LM_CREATE, CAUSAL_SESSION_LM_OPEN,
    CAUSAL_SESSION_LM_OPEN15, CAUSAL_SESSION_LM_RUN, CAUSAL_SESSION_LM_FORWARD, CAUSAL_SESSION_LM_CLOSE,
    CAUSAL_SESSION_PROGRAM_READ, CAUSAL_SESSION_PROGRAM_RELEASE,
)
from core.causal_session_handles import CausalSessionHandles
from core.entry_table import EntryRef
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr, GILReleased
from bindings.py_cpython_table import cpython_table
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_error_class import error_without_class
from bindings.native_loaded_module_door import native_loaded_module, NativeModules
from bindings.native_prepared_owner_unpack import prepared_namespace_admitted, prepared_namespace_admit
from bindings.py_balanced import py_call, py_contains, py_fields, py_get, py_item, py_setattr, py_setitem, py_type
from checks.numerics import NUMERIC_IDENTICAL
from core.native_descriptor_abi import TRANSFORMER_BLOCK_ABI_NAME, PRIMITIVE_ABI_NAME, MAMBA_STATE_ABI_NAME
from providers.python.neural_descriptor_pyapi_door import neural_descriptor_reuse
from providers.python.transformer_block_session_pyapi_door import (
    transformer_block_session_open, transformer_block_session_entry, transformer_block_session_handle,
    transformer_block_session_run_words,
)
from core.neural_entry_table import (
    NeuralEntryKey, neural_ref, neural_table_entry, NEURAL_TABLE_BLOCK_FORWARD,
    NEURAL_BLOCK_SESSION_FORWARD, NEURAL_BLOCK_SESSION_FORWARD_INT15,
)
from core.generation_metadata import generation_transformer_addresses
from providers.python.generation_metadata_pyapi_door import (
    GenerationWeightSnapshot, GenerationLayerSnapshot, GenerationHeadSnapshot, GenerationOperationSnapshot, GenerationCallLayers, GenerationEngineAdmission, _generation_bool,
    _generation_fields,
)
from core.causal_owner import (
    CausalModelWords, causal_positions_admit, causal_engine_candidate, causal_resident_candidate,
    CausalLayerStateWords, CausalBufferWords, causal_state_admit, causal_state_buffer_contiguous, causal_state_buffer_address,
    CAUSAL_STATE_K, CAUSAL_STATE_V, CAUSAL_STATE_CONV, CAUSAL_STATE_H, CAUSAL_STATE_AFFINE_BOUNDARY,
    CAUSAL_STATE_AFFINE_A, CAUSAL_STATE_AFFINE_B, CAUSAL_STATE_XBC, CAUSAL_STATE_DTRAW,
)
from bindings.causal_operation_owner_door import causal_model_owner, CausalModelOwner, causal_model_snapshot_keep
from bindings.causal_operation_owner_door import CausalStateOwner, causal_state_owner, causal_state_layer_unpack
from bindings.causal_keys_unpack import causal_keys
from core.causal_layer_run import (
    CausalLayerRoute, CausalLayerRequest, CAUSAL_LAYER_TRANSFORMER, CAUSAL_LAYER_MAMBA1, CAUSAL_LAYER_MAMBA2,
)
from autograd.block_step_ops import OPS_ATTN_STEP, OPS_M1_STEP, OPS_M2_STEP
from std.memory import unsafe_memcpy, unsafe_memset_zero
from core.generation_buffers_impl import generation_bytes_copy
from bindings.generation_buffers_door import (
    GENERATION_BOUNDARY, generation_array, generation_array_unfilled, generation_address, generation_weight_address, generation_copy, generation_mutable,
    generation_ids, generation_initialize_binding, generation_array_context, generation_array_snapshot, generation_buffer_view, py_ints,
)
from bindings.native_array_snapshot_door import native_array_slots, NativeArrayAdmission
from core.generation_policy_impl import GenerationSamplingScratch
from core.generation_kv_policy import GenerationReadControl, GENERATION_NATIVE_ENGINE_FORCE, GENERATION_NATIVE_ENGINE_OFF
from core.hostptr import f32_ptr, i32_ptr
from core.sha256 import Sha256
from core.native_archive_write import NativeArchiveMember, archive_owned, archive_text_bytes
from core.fitted_digest import fitted_digest_hex
from providers.python.borrowed_buffer_pyapi_unpack import admit_native_buffer_exporter, AdmittedBorrowedBuffer
from bindings.sha256_door import sha256_update, sha256_update_text
from std.math import isfinite
from std.sys.compile import is_defined

from transformer.causal_lm_public_impl import (
    causal_lm_rows_done, causal_lm_join_resident, causal_lm_join_continuation,
    causal_lm_pick_step, causal_lm_tokens_row_major, causal_lm_drop_eos,
    causal_lm_utf8_cut, causal_lm_store_ids, causal_lm_flags_as_ints,
    causal_lm_restore_rows,
)


from bindings.tokenizer_public_native_door import tokenizer_vocab_encode, tokenizer_vocab_decode, tokenizer_vocab_fingerprint, tokenizer_decode_utf8
from bindings.mode_metadata_native_door import _fields as _native_record_fields, _truth
from bindings.neural_block_admission_door import neural_record_fields, neural_native_scalar, neural_plain_value, neural_string_keys, neural_block_weight_names, neural_metadata_snapshot, neural_block_fields
from bindings.causal_model_config_door import model_record_fields, model_plan_snapshot, model_dict, model_plain
from bindings.causal_operation_owner_door import CAUSAL_ENTRY_CPU, CAUSAL_ENTRY_MODE, CAUSAL_ENTRY_VENDOR, CAUSAL_ENTRY_WEIGHT_NAMES, CAUSAL_ENTRY_BLOCK_SESSION, CAUSAL_ENTRY_FORWARD15, CAUSAL_ENTRY_RESIDENT, CAUSAL_ENTRY_RESIDENT_VENDOR, CAUSAL_ENTRY_MAMBA1_PROFILE, CAUSAL_ENTRY_PROGRAM_READ
from bindings.causal_operation_owner_door import causal_model_capture, causal_model_publish, causal_model_origin, causal_state_capture, causal_state_publish, causal_state_origin, causal_state_fields, causal_snapshot_fields, causal_model_cache_set, causal_block_cache_set, causal_checkpoint_capture, causal_pin_entries, causal_layer_capture, causal_layer_publish
from bindings.causal_operation_owner_door import CausalLease, causal_model_lease, causal_lease_publish, causal_lease_mirror
from core.causal_owner import (
    causal_lease_load, causal_lease_take, causal_lease_release, causal_lease_expose,
    CAUSAL_LEASE_TAKEN, CAUSAL_LEASE_EXPOSED,
)
from core.shared_slot import SharedSlot
from core.causal_lm_entry_table import (
    CausalEntryKey, causal_entry_vendor, causal_entry_call, causal_ref, causal_tables_admit,
    CAUSAL_TABLE_LINALG, CAUSAL_TABLE_PRIMITIVES, CAUSAL_TABLE_BLOCK, CAUSAL_TABLE_MAMBA,
    CAUSAL_LIN_GEMM_INT15, CAUSAL_LIN_QUANTIZE_INT15, CAUSAL_PRIM_EMBED_FORWARD, CAUSAL_PRIM_NORM_FORWARD,
    CAUSAL_PRIM_LINEAR_FORWARD, CAUSAL_BLOCK_HOST_FORWARD, CAUSAL_BLOCK_HOST_FORWARD_INT15,
    CAUSAL_MAMBA1_FORWARD, CAUSAL_MAMBA1_DECODE, CAUSAL_MAMBA2_FORWARD, CAUSAL_MAMBA2_DECODE,
)


# Lane envgate-neural-lm (2026-11-15): the CausalLM door's run-time switches
# are build constants with today's defaults; each was a getenv per public
# call, per layer of a forward, or per generate. B arms are RUN OWED
# (bench/results/paper_compare/owed-pre0867.json; docs/apple-fast/EXPERIMENTS.md).
#   -D MOJOLEARN_TRANSFORMER_LEGACY_SETUP (per-call layer setup through the
#   image's Python entries, no block session) ran the same
#   llama_decoder_layer_forward bodies as the block session:
# TOMBSTONE: MOJOLEARN_TRANSFORMER_LEGACY_SETUP (removed without measurement: plumbing, rule 2026-11-22) deleted 2026-11-23 by lane/ab-prep-transformer; code recoverable at 4d76a9e16e.
# Restore: git apply ~/mojolearn-evidence/experiments-removed/MOJOLEARN_TRANSFORMER_LEGACY_SETUP.patch; record in docs/TOMBSTONES.md.
#   -D MOJOLEARN_CAUSAL_LM_RESIDENT_FORWARD_OFF  no resident forward (was =0)
#   -D MOJOLEARN_EXPERIMENT_RESIDENT_SAMPLING_FP32  fp32 sampling stream (was =1)
#   -D MOJOLEARN_EXPERIMENT_GENERATION_ROUTE_TRACE  route witness print (was =1)
comptime CAUSAL_LM_RESIDENT_FORWARD = not is_defined["MOJOLEARN_CAUSAL_LM_RESIDENT_FORWARD_OFF"]()
comptime EXPERIMENT_RESIDENT_SAMPLING_FP32 = is_defined["MOJOLEARN_EXPERIMENT_RESIDENT_SAMPLING_FP32"]()
comptime EXPERIMENT_GENERATION_ROUTE_TRACE = is_defined["MOJOLEARN_EXPERIMENT_GENERATION_ROUTE_TRACE"]()


struct CausalPublicConnection(Defaultable, Movable):
    var context: PythonObject

    def __init__(out self):
        self.context = PythonObject(None)


comptime CAUSAL_PUBLIC_CONNECTION = SharedSlot[CausalPublicConnection, "MojolearnCausalPublicConnectionV1"]


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

# TOMBSTONE: MOJOLEARN_IDN_LM_RESIDENT_GENERATE (opt-in arm, no A/B recorded) deleted 2026-11-15 by lane/expdel2-02-base; code recoverable at 29fe2d5748.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_LM_RESIDENT_GENERATE.patch; record in docs/TOMBSTONES.md.
def _resident_norm_ok(model: PythonObject) raises -> Bool:
    """Whether the model's final norm lets it open a resident session (a
    model without one takes the per-layer route, `_run_snapshot`)."""
    return model._norm is not None


def _standalone_record_fields(mut mods: NativeModules, model: PythonObject) raises -> PythonObject:
    # Standalone blocks use a private exact C namespace as their per-operation
    # owner. Public models still use the ordinary class dictionary admission.
    # SimpleNamespace.__dict__ is a native member descriptor, not the Python
    # class getset descriptor expected by the public-record helper.
    if py_type(model) is mods.module("types").SimpleNamespace:
        ref cpy = Python().cpython()
        var raw = cpython_table()[].PyObject_GenericGetDict(model._obj_ptr, PyObjectPtr())
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
        var raw = cpython_table()[].PyModule_GetDict(entries._obj_ptr)
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
        var raw = cpython_table()[].PyObject_GenericGetDict(entries._obj_ptr, PyObjectPtr())
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


def causal_entry_key(entries: PythonObject) raises -> CausalEntryKey:
    # UNPACK: the admitted (tier, vendor) words a route's connection dict
    # (`_model_bindings`) or a model's entry dict (`causal_lm_admit_binding`)
    # carries for the CausalLM entry table (core/causal_lm_entry_table.mojo).
    return CausalEntryKey(Int(py=entries["entry_mode"]), Int(py=entries["entry_vendor"]))


def _causal_table_call(entry: EntryRef, operation: Int, addresses: List[Int],
    ints: List[Int], floats: List[Float64]) raises -> Int:
    # One descriptor call into a sibling image through the CausalLM entry
    # table: plain words, no CPython call. The GIL is released around the
    # numerical work as the former Python entries and provider calls released
    # it; a caller holding a NativeModules cache resets it afterwards
    # (mods.released()), as it did after those calls.
    var result = 0
    with GILReleased(Python()):
        result = causal_entry_call(entry, operation, addresses, ints, floats)
    return result


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


struct CausalAdmission(Movable):
    # The typed owner of one admission (lane dw-neural-models, 2026-11-21).
    # UNPACK fills the first six words from whichever record asks: a model's
    # construction record (`causal_lm_admit_binding`) or a standalone block's
    # snapshot (bindings/neural_block_public_door.mojo
    # `_neural_block_owner_typed`, which no longer builds a SimpleNamespace
    # owner to carry them). `causal_lm_admit` reads only these words and
    # answers the last four; no attribute of a Python owner is read or
    # written inside it.
    var gpu: Bool                  # the route word is "gpu"
    var kind: String
    var packed: Bool               # the head / block holds fixed15 planes
    var weight_names: PythonObject  # None: read from `first_block`'s class
    var first_block: PythonObject
    var context: PythonObject
    # Answers. `reported` is set once the images' vendor and mode agreed, so a
    # caller publishes them on a later failure exactly as before.
    var reported: Bool
    var vendor: String
    var mode: Int
    var entries: PythonObject

    def __init__(out self, gpu: Bool, kind: String, packed: Bool, weight_names: PythonObject,
                 first_block: PythonObject, context: PythonObject):
        self.gpu = gpu
        self.kind = kind.copy()
        self.packed = packed
        self.weight_names = weight_names
        self.first_block = first_block
        self.context = context
        self.reported = False
        self.vendor = String("")
        self.mode = 0
        self.entries = PythonObject(None)


def causal_lm_admit_binding(mut mods: NativeModules,
    model: PythonObject, block_binding: PythonObject, primitives: PythonObject,
    int15_binding: PythonObject,
) raises -> PythonObject:
    # The admission of an object owner (a model's construction record, the
    # parallel worker's layer holder): UNPACK its words, admit, PACK the
    # answers onto it. The reads and writes are the ones the single function
    # made on the owner.
    var local_blocks = py_get(_standalone_record_fields(mods, model), "_native_block_fields")
    var admission = CausalAdmission(String(model.device) == "gpu", String(model.kind), model._head_int15 is not None,
        local_blocks[0]["_neural_weight_names"] if local_blocks is not None else PythonObject(None),
        model._blocks[0], _causal_context_for(mods, model))
    try:
        causal_lm_admit(mods, admission, block_binding, primitives, int15_binding)
    except problem:
        if admission.reported:
            py_setattr(model, "_native_vendor", PythonObject(admission.vendor))
            py_setattr(model, "_native_numeric_mode", PythonObject(admission.mode))
        raise problem
    py_setattr(model, "_native_vendor", PythonObject(admission.vendor))
    py_setattr(model, "_native_numeric_mode", PythonObject(admission.mode))
    py_setattr(model, "_native_entries", admission.entries)
    causal_pin_entries(mods, _standalone_record_fields(mods, model), admission.entries)
    return PythonObject(None)


def causal_lm_admit(mut mods: NativeModules, mut admission: CausalAdmission,
    block_binding: PythonObject, primitives: PythonObject, int15_binding: PythonObject,
) raises:
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    var entries = g[].bi.dict()
    var requested_gpu = admission.gpu
    py_setitem(entries, CAUSAL_ENTRY_CPU, PythonObject(not requested_gpu))
    var kind = admission.kind.copy()
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
    admission.vendor = vendor.copy()
    admission.mode = mode
    admission.reported = True
    # The CausalLM entry table key (core/causal_lm_entry_table.mojo). The
    # embedding, norm and head projections, the int15 head, every CPU
    # Transformer layer and every Mamba layer call their image's descriptor
    # entry by it; the former Python entry lookups of embedding_forward,
    # rms_norm_forward, linear_forward, quantize_int15, gemm_int15,
    # transformer_forward_int15 (CPU) and the Mamba forward/decode are gone.
    var key = CausalEntryKey(mode, causal_entry_vendor(vendor))
    py_setitem(entries, CAUSAL_ENTRY_MODE, PythonObject(key.mode))
    py_setitem(entries, CAUSAL_ENTRY_VENDOR, PythonObject(key.vendor))
    causal_tables_admit(key, CAUSAL_TABLE_BLOCK if kind == "transformer" else (CAUSAL_TABLE_MAMBA if kind == "mamba1" or kind == "mamba2" else -1), int15_binding is not None)
    py_setitem(entries, CAUSAL_ENTRY_WEIGHT_NAMES, admission.weight_names if admission.weight_names is not None else neural_block_weight_names(mods, admission.first_block))
    if kind == "transformer":
        # The "forward" / "decode" entries (transformer_forward, transformer_decode_step) were read only by
        # the MOJOLEARN_TRANSFORMER_LEGACY_SETUP Python-entry arm.
        # TOMBSTONE: MOJOLEARN_TRANSFORMER_LEGACY_SETUP (removed without measurement: plumbing, rule 2026-11-22) deleted 2026-11-23 by lane/ab-prep-transformer; code recoverable at 4d76a9e16e.
        # Restore: git apply ~/mojolearn-evidence/experiments-removed/MOJOLEARN_TRANSFORMER_LEGACY_SETUP.patch; record in docs/TOMBSTONES.md.
        py_setitem(entries, "block_binding", block_binding)
        for name in [String("transformer_forward_fresh"), String("transformer_session_forward_fresh")]:  # small-loop(optional entry names retained once at construction)
            if _native_has_entry(mods, block_binding, name):
                py_setitem(entries, name, _native_function(mods, block_binding, name))
        # A GPU layer's session is a native registry handle of the block
        # image (core/transformer_block_session.mojo, lane
        # transformer-session-native): created, run (op 11, or op 35 under
        # fixed15_v1) and closed through the neural table's block entry for
        # this key. "block_session" and "forward15" are route flags; the
        # former Python entries (transformer_session_create,
        # transformer_session_forward, transformer_session_forward_int15,
        # transformer_native_session_handle_v1) are gone.
        if requested_gpu:
            _ = neural_table_entry[NEURAL_TABLE_BLOCK_FORWARD](NeuralEntryKey(key.mode, key.vendor))
            py_setitem(entries, CAUSAL_ENTRY_BLOCK_SESSION, PythonObject(True))
            if admission.packed:
                py_setitem(entries, CAUSAL_ENTRY_FORWARD15, PythonObject(True))
        # The resident GPU route for fixed-length greedy generation and the
        # resident forward: every session operation is the transformer
        # image's session entry (core/causal_lm_entry_table_sessions.mojo),
        # sessions are native registry handles. The former Python entry
        # lookups (transformer_decode_session_*, causal_lm_session_*,
        # transformer_vendor) are gone (lane causal-lm-sessions-native).
        if requested_gpu:
            causal_sessions_tables_admit(key, CAUSAL_SESSIONS_TRANSFORMER)
            py_setitem(entries, CAUSAL_ENTRY_RESIDENT, PythonObject(True))
            py_setitem(entries, CAUSAL_ENTRY_RESIDENT_VENDOR, PythonObject(vendor))
            for name in [String("transformer_arithmetic_profile"), String("transformer_arithmetic_profile_changed")]:  # small-loop(optional profile entry names)
                if _native_has_entry(mods, block_binding, name):
                    py_setitem(entries, name, _native_function(mods, block_binding, name))
    elif kind == "mamba1":
        var profile = 1
        if _native_has_entry(mods, block_binding, "mamba1_profile"):
            profile = Int(py=_native_function(mods, block_binding, "mamba1_profile")())
            mods.released()
        py_setitem(entries, CAUSAL_ENTRY_MAMBA1_PROFILE, PythonObject(profile))
    elif kind == "mamba2":
        pass
    else:
        raise Error("CausalLM: unsupported model kind " + kind)
    # Lane x45c: the decode program door (every GPU kind).
    # The program is the block image's session entry (CAUSAL_SESSIONS_TRANSFORMER
    # or CAUSAL_SESSIONS_MAMBA); "program_read" holds that family word.
    if requested_gpu:
        var program_family = CAUSAL_SESSIONS_TRANSFORMER if kind == "transformer" else CAUSAL_SESSIONS_MAMBA
        causal_sessions_tables_admit(key, program_family)
        py_setitem(entries, CAUSAL_ENTRY_PROGRAM_READ, PythonObject(program_family))
    # Native ownership migration: resolve descriptor tables at admission, so
    # supported local operations enter the native execution engine (the
    # per-admission getenv, then its build constant NEURAL_NATIVE_DISPATCH, are
    # deleted: native dispatch is the one form).
    # Lane transformer-session-native (2026-11-12): the generation session
    # provider (`native_session_provider`, `_native_session_provider_cache`,
    # MOJOLEARN_GENERATION_NATIVE_SESSION) is deleted: a layer's session
    # forward is the block table's op 11 on the session handle, the same
    # transformer_session_forward_native body with the same words.
    if mode == NUMERIC_IDENTICAL:
        var owner_fields = neural_block_fields(mods, admission.first_block, admission.context)
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
    admission.entries = entries


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


def _block_list(mut mods: NativeModules, model: PythonObject) raises -> List[PythonObject]:
    # One operation's admitted layer fields. Each `_block` is a full record
    # admission (~40 field reads plus the class ancestry walk); the resident
    # routes read the same layers for the profile check, the session key, the
    # open and the cache export, so they admit each layer once per operation
    # and pass this list down (the operation-snapshot pattern of `_run_prepared`).
    var blocks = List[PythonObject]()
    for i in range(len(model._blocks)):  # driver-loop: admit each layer's block fields once per operation
        blocks.append(_block(mods, model, i))
    return blocks^


def _block_at(mut mods: NativeModules, model: PythonObject, blocks: List[PythonObject], index: Int) raises -> PythonObject:
    # A list shorter than the layer index (never filled, or a partial profile
    # check) falls back to the ordinary per-layer admission.
    if index >= 0 and index < len(blocks):
        return blocks[index]
    return _block(mods, model, index)


def _positions(words: CausalModelWords, state: PythonObject, length: Int) raises:
    # UNPACK of the position check: the state's two words. The rule and the
    # model's limit are typed (core/causal_owner.mojo `causal_positions_admit`;
    # before: the model record's position limit read twice per call).
    if state is None:
        causal_positions_admit(words, False, 0, 0, length)
    else:
        causal_positions_admit(words, True, Int(py=state.positions), Int(py=state.max_tokens), length)


def _causal_weight_copy(mut mods: NativeModules, value: PythonObject) raises -> PythonObject:
    # state_dict() (lane fx-leftovers item 1, 2026-11-21): an independent copy
    # of one weight, so the model's own arrays never leave it and no exposure
    # mark is set. Cold API call (one per checkpoint tensor of state_dict).
    # An Array: one unfilled allocation and one byte copy of its C-order
    # storage. A packed bfloat16 / int8 record is returned as stored: the
    # model executes the float32 copy it unpacked at construction, so an
    # edit of the record never reaches a forward. A foreign C-contiguous
    # buffer (f / i / b) is copied into an Array of its shape.
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    var nk = causal_keys()
    var context = _causal_context()
    if value is None:
        return value
    var dims = List[Int]()
    if py_type(value) is g[].array_cls:
        var captured = generation_array_snapshot(value)
        if String(captured.order) != "C":
            raise Error("CausalLM: state_dict requires C-contiguous model weights")
        var dtype = String(captured.dtype)
        var code = String("f")
        if dtype == "<i4":
            code = "i"
        elif dtype == "<i1":
            code = "b"
        elif dtype != "<f4":
            raise Error("TypeError: CausalLM state_dict: unsupported weight dtype " + dtype)
        for i in range(len(captured.shape)):  # small-loop(array shape dimension words)
            dims.append(Int(py=captured.shape[i]))
        var copied = generation_array_unfilled(dims, code)
        generation_bytes_copy(Int(py=copied._addr), Int(py=captured._addr), Int(py=captured.nbytes))
        _ = captured  # keep the source storage alive through the byte copy above
        return copied
    if py_type(value) is py_item(context, nk[].k_bf16_class) or py_type(value) is py_item(context, nk[].k_int8_class):
        return value
    var view = generation_buffer_view(value)
    if not Bool(py=view.c_contiguous):
        raise Error("CausalLM: state_dict requires C-contiguous model weights")
    var format = String(view.format)
    if format != "f" and format != "i" and format != "b":
        raise Error("TypeError: CausalLM state_dict: unsupported weight buffer format " + format)
    for i in range(len(view.shape)):  # small-loop(buffer shape dimension words)
        dims.append(Int(py=view.shape[i]))
    var copied = generation_array_unfilled(dims, format)
    var destination = py_call(copied._mv.cast, "B")
    py_setitem(destination, py_call(g[].bi.slice, PythonObject(None)), py_call(view.cast, "B"))
    _ = view
    return copied


def _causal_weight_out(mut mods: NativeModules, value: PythonObject, copies: Bool) raises -> PythonObject:
    if copies:
        return _causal_weight_copy(mods, value)
    return value


def causal_lm_parameters_binding(mut mods: NativeModules, model: PythonObject) raises -> PythonObject:
    # parameters() / named_parameters(): the model's own arrays (exposure mark set).
    return causal_lm_parameters_binding(mods, model, False)


def causal_lm_parameters_binding(mut mods: NativeModules, model: PythonObject, copies: Bool) raises -> PythonObject:
    # `copies` (state_dict, lane fx-leftovers item 1, 2026-11-21): every array
    # is an independent copy and the model is NOT marked exposed, so the kept
    # resident session and decode program stay and a later forward reuses
    # them. Before, state_dict shared this entry with parameters(): it set
    # `_parameters_exposed` and dropped the resident sessions, so every
    # forward after a state_dict() opened a new session. parameters() keeps
    # its semantics (live arrays, mark set, sessions dropped).
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    var result = g[].bi.dict()
    var model_fields = _standalone_record_fields(mods, model)
    if not copies:
        # The live arrays leave the model: a kept resident session set could miss
        # an in-place edit, so it is dropped and generation opens per call again.
        # The exposure is one step on the live lease word. A free lease is
        # taken with it and its kept sessions dropped here; a running holder
        # sees the bit at its next test and at its release, and closes its own.
        if py_get(model._native_entries, "parallel_call") is not None:
            # A layer-partitioned model keeps no resident sessions in this
            # process and has no lease: the mark is its public attribute only
            # (ParallelCausalLM never reads it), written as before.
            causal_model_cache_set(mods, model, model_fields, "_parameters_exposed", True)
        else:
            var lease_held = causal_model_lease(mods, model)
            var lease = lease_held.downcast_value_ptr[CausalLease]()
            if causal_lease_expose(lease[].live):
                var dropped = lease[].kept
                lease[].kept = PythonObject(None)
                _ = causal_lease_release(lease[].live)
                _ = dropped
            causal_lease_publish(lease_held, py_get(model_fields, "_causal_live_fields", model_fields))
            _ = lease_held
        causal_model_cache_set(mods, model, model_fields, "_decode_program", PythonObject(None))
    var p = model.plan
    var original_parameters = py_get(model_fields, "_causal_original_parameters", model_fields)
    py_setitem(result, p.embed_name, _causal_weight_out(mods, original_parameters["_embed"], copies))
    if p.norm_name is not None:
        py_setitem(result, p.norm_name, _causal_weight_out(mods, original_parameters["_norm"], copies))
    if p.head_name is not None:
        py_setitem(result, p.head_name, _causal_weight_out(mods, original_parameters["_head"], copies))
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
                py_setitem(result, name, _causal_weight_out(mods, originals[key], copies))
            else:
                # A fused checkpoint tensor (Phi-3 QKV / gate-up) is split
                # (the joined array below is already a fresh copy)
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
                    unsafe_memcpy(dest=f32_ptr(Int(py=joined._addr)) + begin * cols, src=f32_ptr(_addr(pieces[k][2], keep)), count=(end - begin) * cols)
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
    var model_held = causal_model_owner(mods, model)
    _positions(model_held.downcast_value_ptr[CausalModelOwner]()[].words, PythonObject(None), cap)
    _ = model_held
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
    # A caller that holds no model words yet (once per read or operation).
    var model_held = causal_model_owner(mods, model)
    _state(mods, model_held.downcast_value_ptr[CausalModelOwner]()[].words, model, state, b, l)
    _ = model_held


def _state(mut mods: NativeModules, words: CausalModelWords, model: PythonObject, state: PythonObject, b: Int, l: Int) raises:
    var state_fields = causal_state_fields(mods, state, _causal_context_for(mods, model))
    var nk = causal_keys()
    var state_held = py_get(state_fields, nk[].k__causal_native_state)
    if state_held is not None:
        # The state owner's check (core/causal_owner.mojo `causal_state_admit`):
        # UNPACK writes the four words a user may edit; the model binding is
        # one generation word (the capture of this operation checked `_owner`
        # and bound the owner to this model generation) and the layer count is
        # the owner's. Before: the `_owner` field, the model origin and the
        # layer list length read from the records here again.
        var state_owner = state_held.downcast_value_ptr[CausalStateOwner]()
        state_owner[].words.usable = _truth(py_get(state_fields, "_usable", PythonObject(True)))
        state_owner[].words.batch = Int(py=state_fields["batch_size"])
        state_owner[].words.positions = Int(py=state_fields["positions"])
        state_owner[].words.capacity = Int(py=state_fields["max_tokens"])
        causal_state_admit(words, state_owner[].words, b, l)
        _ = state_held
        return
    if state_fields["_owner"] is not causal_model_origin(mods, model):
        raise Error("CausalLM: state belongs to another model")
    if not _truth(py_get(state_fields, "_usable", PythonObject(True))):
        raise Error("CausalLM: state lost after a failed call; reset or restore it")
    if Int(py=state.batch_size) != b or len(state.layers) != words.layers:
        raise Error("CausalLM: state batch/layer counts disagree with model and input")
    _positions(words, state, l)


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


def _weight_addr(value: PythonObject, mut keep: List[PythonObject]) raises -> Int:
    # Model weights only (read by every kernel, never written): a foreign
    # buffer is read in place instead of copied per call (row N200).
    return generation_weight_address(value, keep)


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
        addresses.append(0 if planes and j >= 2 else _weight_addr(weights[j], keep))
    addresses.append(_addr(st.k_cache, keep))
    addresses.append(_addr(st.v_cache, keep))
    addresses.append(_addr(y, keep))
    var optional = blk["_wopt"]
    for j in range(len(optional)):  # driver-loop: over optional block weight slots
        addresses.append(_weight_addr(optional[j], keep))
    if planes:
        var packed = blk["_int15"]
        for j in range(len(packed)):  # driver-loop: over packed block weight slots
            var trip = packed[j]
            for k in range(3):  # driver-loop: over int15 plane triple
                addresses.append(0 if trip is None else _weight_addr(trip[k], keep))
    return addresses^


def _transformer_param_words(blk: PythonObject, st: PythonObject, b: Int, l: Int) raises -> List[Int]:
    var p: List[Int] = [b, l, Int(py=blk["d_model"]), Int(py=blk["n_heads"]), Int(py=blk["n_kv_heads"]),
                        Int(py=blk["head_dim"]), Int(py=blk["intermediate"]), Int(py=st.max_tokens),
                        Int(py=st.cached_tokens), Int(py=blk["window"])]
    var tail = blk["_opts_tail"]
    for j in range(len(tail)):  # driver-loop: read the block's admitted options tail
        p.append(Int(py=tail[j]))
    return p^


def _transformer_params(blk: PythonObject, st: PythonObject, b: Int, l: Int) raises -> PythonObject:
    return py_ints(_transformer_param_words(blk, st, b, l))



# Lane transformer-session-native (2026-11-12): `causal_transformer_admin` (the
# session close through a provider capsule and the
# `transformer_native_session_handle_v1` / `transformer_session_close` Python
# entries) is deleted; a block session closes through its native owner
# (providers/python/transformer_block_session_pyapi.mojo).

# Lane dw-neural-models (2026-11-21): `causal_layer_run(model, i, ...)`, the
# layer run over a SimpleNamespace owner, is deleted: it had no caller (the
# block door runs `causal_layer_run_fields` with its plan's route, the
# whole-model path `_run_snapshot` with its snapshot's).


def causal_layer_run_fields(mut mods: NativeModules, fields: PythonObject, entries: PythonObject, route: CausalLayerRoute, kind: String, d: Int, packed: Bool, planes: Bool, prepared: PythonObject, x: PythonObject, supplied_state: PythonObject, step: Bool = False, stateless: Bool = False) raises -> PythonObject:
    # Standalone callers supply the retained native plan directly: no temporary
    # namespace or Python attribute dispatch to reconstruct execution metadata.
    # `route` and `planes` are the plan's (`NativeBlockPlan.layer_route`,
    # `.planes`), resolved once: this door reads no entry word by name.
    # `stateless` (lane fix-mamba12-forward-exec, `causal_layer_stateless_plan`):
    # a Mamba-1/2 device forward with no state object; `supplied_state` is
    # None and the call runs from a zero state with nothing written back.
    _causal_array_guard(mods)
    var shape = x.shape
    var b = Int(py=shape[0])
    var l = Int(py=shape[1])
    var profile = route.kind == CAUSAL_LAYER_TRANSFORMER and packed
    if prepared is not None:
        # Reuse the immutable native layer vectors directly. Only mutable
        # arithmetic-profile words and this operation's fields are rebound.
        model_dict(fields)
        var profile_id = PythonObject(None)
        var profile_changed = False
        if route.kind == CAUSAL_LAYER_TRANSFORMER:
            profile_id = model_plain(py_get(fields, "_arithmetic_profile", PythonObject(None)))
            profile_changed = _generation_bool(py_get(fields, "_arithmetic_profile_changed", PythonObject(False)))
        ref reused = prepared.downcast_value_ptr[PreparedGenerationLayer]()[].layer
        mods.released()
        return _causal_layer_run_bound(mods, route, entries, d, b, l, reused, profile, planes, x, supplied_state, step, fields, profile_id, profile_changed, stateless)
    var layer = GenerationLayerSnapshot(fields, kind, planes)
    mods.released()
    if stateless:
        return _causal_layer_run_bound(mods, route, entries, d, b, l, layer, profile, planes, x, supplied_state, step,
            layer.block, layer.profile, layer.profile_changed, True)
    return _causal_layer_run_prepared(mods, route, entries, d, b, l, layer, profile, planes, x, supplied_state, step)


def causal_layer_stateless_plan(kind: String, entries: PythonObject) raises -> Bool:
    """Lane fix-mamba12-forward-exec: whether a standalone layer FORWARD with
    no state runs stateless (`causal_layer_run_fields(stateless=True)`): a
    Mamba-1/2 forward on a device image (entry vendor not the host's 0),
    Mamba-1 on its plain profile. The device per-call program then zeroes
    its state tensor on the device and reads nothing back
    (mamba/impl/mamba_block_program.mojo `mamba_block_call_abi`), where the
    door used to allocate a host zero state (Mamba-2: about 2 MB at d_model
    384) that the call uploaded and downloaded. The host image reads every
    state address, so it keeps the allocated state.
    Decided ONCE, at plan construction (`NativeBlockPlan.stateless_forward`):
    the kind, the entry vendor and the Mamba-1 profile are fixed for a plan
    (hot-path audit 2026-11-13, BLOCK door:824/826: this read two entry
    words per forward call). A decode step never runs stateless; the caller
    tests `step` per call (a Bool)."""
    # TOMBSTONE: MOJOLEARN_IDN_MAMBA12_FORWARD_COPIES_OLD (old path, no A/B recorded) deleted 2026-11-15 by lane/expdel2-02-base; code recoverable at 2e9d66525c.
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_MAMBA12_FORWARD_COPIES_OLD.expdel2-02-base.patch; record in docs/TOMBSTONES.md.
    # RESTORED 2026-11-15 by lane/expdel2-restore-owed: owed B arm (owed-pre0867.json), under four weeks old; the deletion was premature.
    # TOMBSTONE: MOJOLEARN_IDN_MAMBA12_FORWARD_COPIES_OLD (removed without measurement: plumbing, rule 2026-11-22) deleted 2026-11-23 by lane/ab-prep-mamba; code recoverable at 4d76a9e16e.
    # Restore: git apply ~/mojolearn-evidence/experiments-removed/MOJOLEARN_IDN_MAMBA12_FORWARD_COPIES_OLD.ab-prep.patch; record in docs/TOMBSTONES.md.
    if kind != "mamba1" and kind != "mamba2":
        return False
    if Int(py=entries["entry_vendor"]) == 0:
        return False
    if kind == "mamba1" and Int(py=entries["mamba1_profile"]) == 2:
        return False
    return True


#: Lane fix-mamba-forward-doors (2026-11-14): what this switch decides is the
#: door a stateless Mamba-1/2 forward (`causal_layer_stateless_plan`) takes
#: from the neural door to its one table call. Default (A): the neural door
#: calls `causal_mamba_stateless_run` directly with the plan's prepared layer
#: and the Mamba table entry the plan resolved once (0.8.55 reached
#: `_mamba1_run` / `_mamba2_run` with one binding call): no per-call
#: `causal_layer_run_fields` / `_causal_layer_run_bound` prologue (the
#: fields dict check, the state-namespace and `_resident_session` getattr
#: probes of a None state, the (tier, vendor) dict reads and table read).
#: The same addresses and words reach the same table call: no bit moves.
# TOMBSTONE: MOJOLEARN_IDN_MAMBA12_FORWARD_DOOR_OLD (old path, no A/B recorded) deleted 2026-11-15 by lane/expdel2-02-base; code recoverable at f1c5239d24.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_MAMBA12_FORWARD_DOOR_OLD.expdel2-02-base.patch; record in docs/TOMBSTONES.md.
# RESTORED 2026-11-15 by lane/expdel2-restore-owed: owed B arm (owed-pre0867.json), under four weeks old; the deletion was premature.
# TOMBSTONE: MOJOLEARN_IDN_MAMBA12_FORWARD_DOOR_OLD (removed without measurement: plumbing, rule 2026-11-22; the four-hop neural door -> causal_layer_run_fields -> _causal_layer_run_bound -> table chain to the same table call) deleted 2026-11-23 by lane/ab-prep-mamba; code recoverable at 4d76a9e16e.
# Restore: git apply ~/mojolearn-evidence/experiments-removed/MOJOLEARN_IDN_MAMBA12_FORWARD_DOOR_OLD.ab-prep.patch; record in docs/TOMBSTONES.md.


def causal_mamba_entry(entries: PythonObject) raises -> EntryRef:
    """The Mamba state entry (CAUSAL_TABLE_MAMBA) of a block's admitted
    (tier, vendor) key: resolved once by the caller and held."""
    return causal_ref[CAUSAL_TABLE_MAMBA](causal_entry_key(entries))


def causal_mamba_stateless_run(mut mods: NativeModules, fields: PythonObject, entry: EntryRef, kind: String, d: Int,
    prepared: PythonObject, x: PythonObject) raises -> PythonObject:
    """Lane fix-mamba-forward-doors: the
    stateless Mamba-1/2 device forward (`causal_layer_stateless_plan`) as one
    table call through `entry` (held by the caller's plan) over the prepared
    layer: x, the weights, zero state addresses (the device program runs
    from a zero state and reads nothing back), y unfilled (written whole),
    and for Mamba-2 the h_last report into an unfilled array kept on the
    block's fields as `h_last_` -- the stateless arms of
    `_causal_layer_run_bound`, word for word."""
    _causal_array_guard(mods)
    var shape = x.shape
    var b = Int(py=shape[0])
    var l = Int(py=shape[1])
    ref layer = prepared.downcast_value_ptr[PreparedGenerationLayer]()[].layer
    mods.released()
    var keep = List[PythonObject]()
    var y = generation_array_unfilled([b, l, d])
    var m_addresses = List[Int](capacity=len(layer.weights.addresses) + 7)
    m_addresses.append(_addr(x, keep))
    layer.weights.append_resolved(m_addresses, keep)
    if kind == "mamba1":
        m_addresses.append(0)
        m_addresses.append(0)
        m_addresses.append(_addr(y, keep))
        var s_params: List[Int] = [b, l, d]
        _ = _causal_table_call(entry, CAUSAL_MAMBA1_FORWARD, m_addresses, s_params, [])
        mods.released()
    elif kind == "mamba2":
        m_addresses.append(0)
        m_addresses.append(0)
        m_addresses.append(0)
        m_addresses.append(0)
        m_addresses.append(_addr(y, keep))
        var s_last = generation_array_unfilled([b, layer.words.heads, 64, 128])
        m_addresses.append(_addr(s_last, keep))
        var s_params: List[Int] = [b, l, d, 0]
        _ = _causal_table_call(entry, CAUSAL_MAMBA2_FORWARD, m_addresses, s_params,
            [layer.words.dt_low, layer.words.dt_high])
        mods.released()
        _ = s_last  # keep the Python object alive through the native call above
        causal_block_cache_set(mods, fields, "h_last_", s_last)
    else:
        raise Error("CausalLM: a stateless layer call is a Mamba-1/2 forward with no state")
    _ = y  # keep the Python object alive through the native call above
    _ = prepared  # retain the captured weights until every native read completes
    _ = keep^
    return y


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
    # position (causal_layer_run_fields) and refuses there with the same error.
    try:
        return PythonObject(alloc=PreparedGenerationLayer(GenerationLayerSnapshot(block_fields, kind, planes)))
    except:
        mods.released()
        return PythonObject(None)


def _causal_layer_run_prepared(mut mods: NativeModules,
    route: CausalLayerRoute, e: PythonObject, d: Int, b: Int, l: Int, layer: GenerationLayerSnapshot,
    head_profile: Bool, device15: Bool, x: PythonObject, supplied_state: PythonObject, step: Bool,
) raises -> PythonObject:
    return _causal_layer_run_bound(mods, route, e, d, b, l, layer, head_profile, device15, x, supplied_state, step, layer.block, layer.profile, layer.profile_changed, False)


def _causal_layer_request_run(request: CausalLayerRequest) raises -> Int:
    # RUN of a layer call: typed words only (core/causal_layer_run.mojo). One
    # block-session operation (reserved and released around it with the GIL
    # held) or one CausalLM table call; the numerical work runs with the GIL
    # released. No Python object, entry dictionary or block field reaches
    # here. The caller keeps the session owner and every addressed buffer
    # alive, and resets its NativeModules cache afterwards.
    if request.session:
        return transformer_block_session_run_words(request.entry, request.handle, request.operation,
            request.addresses, request.words)
    return _causal_table_call(request.entry, request.operation, request.addresses, request.words, request.floats)


def _causal_layer_run_bound(mut mods: NativeModules,
    route: CausalLayerRoute, e: PythonObject, d: Int, b: Int, l: Int, layer: GenerationLayerSnapshot,
    head_profile: Bool, device15: Bool, x: PythonObject, supplied_state: PythonObject, step: Bool,
    block_override: PythonObject, profile_override: PythonObject, profile_changed_override: Bool,
    stateless: Bool = False,
) raises -> PythonObject:
    # Lane dw-neural-models (2026-11-21): one layer call as UNPACK / RUN /
    # PACK, for the block door and the whole-model path alike.
    #   UNPACK (below, to the RUN line): the state object, the weights and the
    #     block's session become a `CausalLayerRequest` (entry, operation,
    #     address and scalar words). Arm selection reads `route`, resolved
    #     once by the plan or the snapshot: no entry word is read by name per
    #     call (before: the `cpu`, `block_session` and `mamba1_profile` words
    #     per layer per call, and the (tier, vendor) key plus a table resolve
    #     per standalone call).
    #   RUN: `_causal_layer_request_run`, typed words only.
    #   PACK: the returned word onto the state, the Mamba-2 h_last report onto
    #     the block's fields, the copied state buffers back.
    # `e` is read only where a block session is opened (once per block).
    var keep = List[PythonObject]()
    var copies = List[PythonObject]()
    var blk = block_override
    var st = supplied_state
    if not route.bound:
        raise Error("CausalLM: layer route is not resolved")
    var transformer = route.kind == CAUSAL_LAYER_TRANSFORMER
    var mamba1 = route.kind == CAUSAL_LAYER_MAMBA1
    if stateless and (st is not None or step or (not mamba1 and route.kind != CAUSAL_LAYER_MAMBA2)):
        raise Error("CausalLM: a stateless layer call is a Mamba-1/2 forward with no state")
    if py_type(st) is mods.module("types").SimpleNamespace:
        var failure = py_get(causal_snapshot_fields(mods, st), "_causal_admission_failure")
        if failure is not None:
            raise Error(String(failure))
    # lane fix-mamba12-forward-exec: a Mamba-1/2 device call writes every
    # element of y (one download of the block output), so y is allocated
    # unfilled there; the host image and the Transformer keep the zeros.
    # Hot-path audit 2026-11-13 (BLOCK door:894): the Mamba table entry and
    # its vendor are resolved once, by the route's owner (`causal_layer_route`:
    # per block plan, per model snapshot); every Mamba arm calls through
    # `route.entry`.
    var device_mamba = route.device_mamba
    if stateless and not device_mamba:
        raise Error("CausalLM: a stateless layer call needs a device Mamba image")
    var y: PythonObject
    if device_mamba:
        y = generation_array_unfilled([b, l, d])
    else:
        y = generation_array([b, l, d])
    # every layer kind: a state a decode session holds resident is refused by name
    var gb = GENERATION_BOUNDARY.get_or_create_ptr()
    if py_call(gb[].bi.getattr, st, "_resident_session", PythonObject(None)) is not None:
        raise Error("CausalLM: layer state is owned by a decode session (resident in it); sync_state() or close it first")
    var request = CausalLayerRequest()
    # The block session owner of a device Transformer call and the Mamba-2
    # h_last report: Python-held buffers the request's words address, kept
    # alive here across RUN.
    var session = PythonObject(None)
    var last = PythonObject(None)
    if transformer:
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
        request.addresses = a_words^
        request.words = p_words^
        if route.cpu:
            # Every CPU layer, every tier: the neural host image's block entry
            # through the CausalLM entry table (op 34 under fixed15_v1, else
            # 33). The former Python-entry arms (transformer_forward,
            # transformer_decode_step at L = 1, transformer_forward_int15) ran
            # the same _run_forward with the same words.
            request.entry = route.entry
            request.operation = CAUSAL_BLOCK_HOST_FORWARD_INT15 if head_profile else CAUSAL_BLOCK_HOST_FORWARD
        elif device15 or route.block_session:
            # A GPU layer: the block's session (a native registry handle of its
            # image, opened on first use through the neural table) runs op 35
            # under fixed15_v1, else op 11, reserved with the GIL held and run
            # with it released (lane transformer-session-native). The former
            # three arms (the block provider capsule's op 35, the generation
            # session provider and the transformer_session_forward[_int15]
            # Python entries) ran these bodies with these words.
            session = blk["_native_session"]
            if session is None:
                session = transformer_block_session_open(neural_ref[NEURAL_TABLE_BLOCK_FORWARD](
                    NeuralEntryKey(Int(py=e["entry_mode"]), Int(py=e["entry_vendor"]))))
                causal_block_cache_set(mods, blk, "_native_session", session)
                mods.released()
            request.session = True
            request.entry = transformer_block_session_entry(session)
            request.handle = transformer_block_session_handle(session)
            request.operation = NEURAL_BLOCK_SESSION_FORWARD_INT15 if device15 else NEURAL_BLOCK_SESSION_FORWARD
        else:
            # A GPU admission always marks "block_session"; no other device route exists.
            # TOMBSTONE: MOJOLEARN_TRANSFORMER_LEGACY_SETUP (removed without measurement: plumbing, rule 2026-11-22) deleted 2026-11-23 by lane/ab-prep-transformer; code recoverable at 4d76a9e16e.
            # Restore: git apply ~/mojolearn-evidence/experiments-removed/MOJOLEARN_TRANSFORMER_LEGACY_SETUP.patch; record in docs/TOMBSTONES.md.
            raise Error("CausalLM: a device Transformer layer has no block session")
    else:
        # The `_resident_session` refusal above already covers every layer
        # kind; the second per-layer getattr that stood here never fired.
        # Fixed public Mamba ABI words, plus the optional Mamba1 affine tail.
        # Reserve once so the input/weights/state/report vector does not grow
        # through multiple allocations during every public decode step.
        # TOMBSTONE: the Mamba-2 public-step workspace (MAMBA2_WORKSPACE_STEP route) (moved onto the block step op, lane x5-mamba-on-executor) deleted 2026-11-09 by 08332f3665; code recoverable at 25e5650c14.
        # Restore: git apply experiments/removed/X5_MAMBA_PER_CALL_BODIES.patch
        # The Mamba image's state entry through the CausalLM entry table:
        # one route for every tier and profile (the provider-capsule and
        # Python-entry arms ran this same mamba_state operation; both
        # refuse the 17-word affine profile by the same arity rule).
        request.entry = route.entry
        var m_addresses = List[Int](capacity=len(layer.weights.addresses) + 7)
        m_addresses.append(_addr(x, keep))
        layer.weights.append_resolved(m_addresses, keep)
        if mamba1 and stateless:
            # lane fix-mamba12-forward-exec: no state object; every state
            # address 0 runs the device program from a zero state, nothing
            # read back (`causal_layer_stateless_plan`, plain profile only)
            m_addresses.append(0)
            m_addresses.append(0)
            m_addresses.append(_addr(y, keep))
            var s_params: List[Int] = [b, l, d]
            request.operation = CAUSAL_MAMBA1_FORWARD
            request.words = s_params^
        elif mamba1:
            var di = layer.words.inner
            m_addresses.append(_mutable(st.conv_window, [b, di, 4], keep, copies, "conv_window"))
            m_addresses.append(_mutable(st.h, [b, di, 16], keep, copies, "h"))
            m_addresses.append(_addr(y, keep))
            var m_params: List[Int] = [b, d] if step else [b, l, d]
            var profile = route.mamba1_profile
            if Int(py=st.profile_id) != profile or not Bool(py=st._nn34_usable):
                raise Error("CausalLM: incompatible or failed Mamba1 state")
            if profile == 2:
                m_addresses.append(_mutable(st.affine_boundary, [b, di, 16], keep, copies, "affine_boundary"))
                m_addresses.append(_mutable(st.affine_a, [b, di, 16, 6], keep, copies, "affine_a"))
                m_addresses.append(_mutable(st.affine_b, [b, di, 16, 6], keep, copies, "affine_b"))
                m_params.append(Int(py=st.absolute_position))
            request.operation = CAUSAL_MAMBA1_DECODE if step else CAUSAL_MAMBA1_FORWARD
            request.words = m_params^
        elif stateless:
            # lane fix-mamba12-forward-exec: the Mamba-2 stateless forward
            # (see the Mamba-1 arm above): buf_len 0, the h_last report into
            # an unfilled array (written whole by the report copy)
            m_addresses.append(0)
            m_addresses.append(0)
            m_addresses.append(0)
            m_addresses.append(0)
            m_addresses.append(_addr(y, keep))
            last = generation_array_unfilled([b, layer.words.heads, 64, 128])
            m_addresses.append(_addr(last, keep))
            var s_params: List[Int] = [b, l, d, 0]
            var s_floats: List[Float64] = [layer.words.dt_low, layer.words.dt_high]
            request.operation = CAUSAL_MAMBA2_FORWARD
            request.words = s_params^
            request.floats = s_floats^
        else:
            var nh = layer.words.heads
            var cd = layer.words.conv
            m_addresses.append(_mutable(st.conv_window, [b, cd, 4], keep, copies, "conv_window"))
            m_addresses.append(_mutable(st.h, [b, nh, 64, 128], keep, copies, "h"))
            m_addresses.append(_mutable(st.buffer_xbc, [b, 256, cd], keep, copies, "buffer_xbc"))
            m_addresses.append(_mutable(st.buffer_dtraw, [b, 256, nh], keep, copies, "buffer_dtraw"))
            m_addresses.append(_addr(y, keep))
            # lane fix-mamba12-forward-exec: written whole by the device
            # report copy; the host image keeps the zeros
            if device_mamba:
                last = generation_array_unfilled([b, layer.words.heads, 64, 128])
            else:
                last = generation_array([b, layer.words.heads, 64, 128])
            m_addresses.append(_addr(last, keep))
            var m_params: List[Int] = [b, d, Int(py=st.buffered_tokens)] if step else [b, l, d, Int(py=st.buffered_tokens)]
            var m_floats: List[Float64] = [layer.words.dt_low, layer.words.dt_high]
            request.operation = CAUSAL_MAMBA2_DECODE if step else CAUSAL_MAMBA2_FORWARD
            request.words = m_params^
            request.floats = m_floats^
        request.addresses = m_addresses^
    # RUN
    var result = _causal_layer_request_run(request)
    mods.released()
    # Every Python-held buffer the request addressed is named after the call:
    # Mojo frees a local at its last named use.
    _ = session
    _ = last
    _ = y
    # PACK
    if transformer:
        py_setattr(st, "cached_tokens", PythonObject(result))
    elif mamba1:
        if not stateless:
            py_setattr(st, "absolute_position", PythonObject(Int(py=st.absolute_position) + l))
    else:
        if not stateless:
            py_setattr(st, "buffered_tokens", PythonObject(result))
        causal_block_cache_set(mods, blk, "h_last_", last)
    _writeback(copies)
    _ = layer  # retain captured Array weights until every native read completes
    _ = request^
    _ = keep^
    return y


def _causal_layer_run_owned(mut mods: NativeModules,
    route: CausalLayerRoute, e: PythonObject, d: Int, b: Int, l: Int, layer: GenerationLayerSnapshot,
    head_profile: Bool, device15: Bool, x: PythonObject, fields: PythonObject, state: CausalLayerStateWords,
    blk: PythonObject, profile_id: PythonObject, profile_changed: Bool,
) raises -> PythonObject:
    # Lane dw-causal (2026-11-21): one layer of a whole-model forward on the
    # state owner's words. Same request, same words, same refusals as
    # `_causal_layer_run_bound` (which stays for the block door, for a step
    # or stateless block call, and for a state buffer that is not an Array).
    #   UNPACK: the layer's counter and profile words from the captured layer
    #     state's dictionary (`fields`, interned keys), the state buffers'
    #     addresses from `state` (the owner's words: typed shape and layout
    #     checks, no buffer object read), the weights from the model owner's
    #     snapshot, the block session.
    #   RUN: `_causal_layer_request_run`, typed words only.
    #   PACK: the counter word, the Mamba-2 h_last report. No write-back:
    #     every buffer is updated in place.
    # The caller holds the state owner's handle (the pins of the buffers)
    # and the model owner's (the pins of the weights) across this call.
    var keep = List[PythonObject]()
    var nk = causal_keys()
    if not route.bound:
        raise Error("CausalLM: layer route is not resolved")
    var transformer = route.kind == CAUSAL_LAYER_TRANSFORMER
    var mamba1 = route.kind == CAUSAL_LAYER_MAMBA1
    var device_mamba = route.device_mamba
    var y: PythonObject
    if device_mamba:
        y = generation_array_unfilled([b, l, d])
    else:
        y = generation_array([b, l, d])
    # every layer kind: a state a decode session holds resident is refused by name
    if py_get(fields, nk[].k__resident_session) is not None:
        raise Error("CausalLM: layer state is owned by a decode session (resident in it); sync_state() or close it first")
    var request = CausalLayerRequest()
    var session = PythonObject(None)
    var last = PythonObject(None)
    if transformer:
        var old_profile = py_get(fields, nk[].k_arithmetic_profile)
        var cached = Int(py=py_item(fields, nk[].k_cached_tokens))
        if old_profile is None and cached != 0 and profile_changed:
            raise Error("CausalLM: cached state requires arithmetic profile metadata")
        if old_profile is not None and profile_id is not None and not Bool(py=old_profile == profile_id):
            raise Error("CausalLM: state arithmetic profile differs")
        if profile_id is not None:
            py_setitem(fields, nk[].k_arithmetic_profile, profile_id)
        var capacity = Int(py=py_item(fields, nk[].k_max_tokens))
        var cache_n = layer.words.cache_count(b, capacity)
        var p_words = layer.words.transformer_params(b, l, capacity, cached)
        var x_address = _addr(x, keep)
        var weight_words = layer.weights.resolve(keep)
        causal_state_buffer_contiguous(state.buffers[CAUSAL_STATE_K])
        causal_state_buffer_contiguous(state.buffers[CAUSAL_STATE_V])
        var a_words = generation_transformer_addresses(x_address, weight_words,
            state.buffers[CAUSAL_STATE_K].address, state.buffers[CAUSAL_STATE_V].address, _addr(y, keep),
            layer.optional.resolve(keep), layer.packed.resolve(keep))
        var cache_shape: List[Int] = [cache_n]
        a_words[10] = causal_state_buffer_address(state.buffers[CAUSAL_STATE_K], cache_shape, "k_cache")
        a_words[11] = causal_state_buffer_address(state.buffers[CAUSAL_STATE_V], cache_shape, "v_cache")
        request.addresses = a_words^
        request.words = p_words^
        if route.cpu:
            request.entry = route.entry
            request.operation = CAUSAL_BLOCK_HOST_FORWARD_INT15 if head_profile else CAUSAL_BLOCK_HOST_FORWARD
        elif device15 or route.block_session:
            session = blk["_native_session"]
            if session is None:
                session = transformer_block_session_open(neural_ref[NEURAL_TABLE_BLOCK_FORWARD](
                    NeuralEntryKey(Int(py=e["entry_mode"]), Int(py=e["entry_vendor"]))))
                causal_block_cache_set(mods, blk, "_native_session", session)
                mods.released()
            request.session = True
            request.entry = transformer_block_session_entry(session)
            request.handle = transformer_block_session_handle(session)
            request.operation = NEURAL_BLOCK_SESSION_FORWARD_INT15 if device15 else NEURAL_BLOCK_SESSION_FORWARD
        else:
            # A GPU admission always marks "block_session"; no other device route exists.
            # TOMBSTONE: MOJOLEARN_TRANSFORMER_LEGACY_SETUP (removed without measurement: plumbing, rule 2026-11-22) deleted 2026-11-23 by lane/ab-prep-transformer; code recoverable at 4d76a9e16e.
            # Restore: git apply ~/mojolearn-evidence/experiments-removed/MOJOLEARN_TRANSFORMER_LEGACY_SETUP.patch; record in docs/TOMBSTONES.md.
            raise Error("CausalLM: a device Transformer layer has no block session")
    else:
        request.entry = route.entry
        var m_addresses = List[Int](capacity=len(layer.weights.addresses) + 7)
        m_addresses.append(_addr(x, keep))
        layer.weights.append_resolved(m_addresses, keep)
        if mamba1:
            var di = layer.words.inner
            m_addresses.append(causal_state_buffer_address(state.buffers[CAUSAL_STATE_CONV], [b, di, 4], "conv_window"))
            m_addresses.append(causal_state_buffer_address(state.buffers[CAUSAL_STATE_H], [b, di, 16], "h"))
            m_addresses.append(_addr(y, keep))
            var m_params: List[Int] = [b, l, d]
            var profile = route.mamba1_profile
            if Int(py=py_item(fields, nk[].k_profile_id)) != profile or not Bool(py=py_item(fields, nk[].k__nn34_usable)):
                raise Error("CausalLM: incompatible or failed Mamba1 state")
            if profile == 2:
                m_addresses.append(causal_state_buffer_address(state.buffers[CAUSAL_STATE_AFFINE_BOUNDARY], [b, di, 16], "affine_boundary"))
                m_addresses.append(causal_state_buffer_address(state.buffers[CAUSAL_STATE_AFFINE_A], [b, di, 16, 6], "affine_a"))
                m_addresses.append(causal_state_buffer_address(state.buffers[CAUSAL_STATE_AFFINE_B], [b, di, 16, 6], "affine_b"))
                m_params.append(Int(py=py_item(fields, nk[].k_absolute_position)))
            request.operation = CAUSAL_MAMBA1_FORWARD
            request.words = m_params^
        else:
            var nh = layer.words.heads
            var cd = layer.words.conv
            m_addresses.append(causal_state_buffer_address(state.buffers[CAUSAL_STATE_CONV], [b, cd, 4], "conv_window"))
            m_addresses.append(causal_state_buffer_address(state.buffers[CAUSAL_STATE_H], [b, nh, 64, 128], "h"))
            m_addresses.append(causal_state_buffer_address(state.buffers[CAUSAL_STATE_XBC], [b, 256, cd], "buffer_xbc"))
            m_addresses.append(causal_state_buffer_address(state.buffers[CAUSAL_STATE_DTRAW], [b, 256, nh], "buffer_dtraw"))
            m_addresses.append(_addr(y, keep))
            if device_mamba:
                last = generation_array_unfilled([b, nh, 64, 128])
            else:
                last = generation_array([b, nh, 64, 128])
            m_addresses.append(_addr(last, keep))
            var m2_params: List[Int] = [b, l, d, Int(py=py_item(fields, nk[].k_buffered_tokens))]
            var m2_floats: List[Float64] = [layer.words.dt_low, layer.words.dt_high]
            request.operation = CAUSAL_MAMBA2_FORWARD
            request.words = m2_params^
            request.floats = m2_floats^
        request.addresses = m_addresses^
    # RUN
    var result = _causal_layer_request_run(request)
    mods.released()
    _ = session
    _ = last
    _ = y
    # PACK
    if transformer:
        py_setitem(fields, nk[].k_cached_tokens, PythonObject(result))
    elif mamba1:
        py_setitem(fields, nk[].k_absolute_position, PythonObject(Int(py=py_item(fields, nk[].k_absolute_position)) + l))
    else:
        py_setitem(fields, nk[].k_buffered_tokens, PythonObject(result))
        causal_block_cache_set(mods, blk, "h_last_", last)
    _ = layer  # retain captured Array weights until every native read completes
    _ = request^
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
    return _causal_model_embedding_prepared(e, d, v, b, l, flat, weights, ids)


def _causal_model_embedding_prepared(
    e: PythonObject, d: Int, v: Int, b: Int, l: Int, flat: Bool,
    weights: GenerationWeightSnapshot, ids: PythonObject,
) raises -> PythonObject:
    var keep = List[PythonObject]()
    var x = generation_array([b, l, d])
    var weights_at = weights.resolve(keep)
    var a: List[Int] = [_addr(x, keep), weights_at[0], _addr(ids, keep)]
    # The primitives image's entry through the CausalLM entry table: one
    # route for every tier (the provider-capsule and Python-entry arms that
    # stood here ran this same operation 0).
    _ = _causal_table_call(causal_ref[CAUSAL_TABLE_PRIMITIVES](causal_entry_key(e)), CAUSAL_PRIM_EMBED_FORWARD, a, [b * l, v, d], [])
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
    unsafe_memset_zero(f32_ptr(_addr(output, keep)), b * l * v)
    _ = keep
    return output


def _causal_model_head_prepared(head: GenerationHeadSnapshot, x: PythonObject, b: Int, l: Int, flat: Bool, output: PythonObject) raises -> PythonObject:
    var e = head.entries
    var d = head.width
    var v = head.vocab
    var keep = List[PythonObject]()
    var weights_at = head.weights.resolve(keep)
    var hn = x
    # The primitives and linalg images' entries through the CausalLM entry
    # table (core/causal_lm_entry_table.mojo); the same operations the former
    # provider-capsule and Python-entry arms ran, with the same words.
    var key = causal_entry_key(e)
    # Resolved once for the head (lane entry-table-zero-alloc).
    var prim = causal_ref[CAUSAL_TABLE_PRIMITIVES](key)
    if head.normalize:
        hn = generation_array([b, l, d])
        var a: List[Int] = [_addr(hn, keep), _addr(x, keep), weights_at[0]]
        _ = _causal_table_call(prim, CAUSAL_PRIM_NORM_FORWARD, a, [b * l, d], [head.norm_eps])
        _ = hn  # keep the Python object alive through the native call above
    var out = _causal_logits_output(b, l, v, output)
    if not head.packed:
        var a: List[Int] = [_addr(out, keep), _addr(hn, keep), weights_at[1]]
        _ = _causal_table_call(prim, CAUSAL_PRIM_LINEAR_FORWARD, a, [b * l, v, d], [])
        _ = out  # keep the Python object alive through the native call above
        _ = hn  # keep the Python object alive through the native call above
    else:
        var hi = generation_array([b * l, d], "b")
        var lo = generation_array([b * l, d], "b")
        var ex = generation_array([b * l], "i")
        var lin = causal_ref[CAUSAL_TABLE_LINALG](key)
        _ = _causal_table_call(lin, CAUSAL_LIN_QUANTIZE_INT15,
            [_addr(hi, keep), _addr(lo, keep), _addr(ex, keep), _addr(hn, keep)], [b * l, d], [])
        _ = ex  # keep the Python object alive through the native call above
        _ = lo  # keep the Python object alive through the native call above
        _ = hi  # keep the Python object alive through the native call above
        _ = hn  # keep the Python object alive through the native call above
        _ = _causal_table_call(lin, CAUSAL_LIN_GEMM_INT15,
            [_addr(out, keep), _addr(hi, keep), _addr(lo, keep), _addr(ex, keep),
             weights_at[2], weights_at[3], weights_at[4]], [b * l, v, d], [])
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
    var blocks = List[PythonObject]()
    return _resident_profiles_ok(mods, model, state, blocks)


def _resident_profiles_ok(mut mods: NativeModules, model: PythonObject, state: PythonObject,
                          mut blocks: List[PythonObject]) raises -> Bool:
    # `blocks` receives each admitted layer, in order; on True it holds every
    # layer and the caller reuses it for the key, the open and the export.
    blocks.clear()
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
    var nk = causal_keys()
    var namespace = mods.module("types").SimpleNamespace
    var layer_states = state.layers
    for i in range(len(model._blocks)):  # driver-loop: captured profile admission once per public operation
        var blk = _block(mods, model, i)
        blocks.append(blk)
        var profile = py_get(blk, nk[].k__arithmetic_profile)
        var st = layer_states[i]
        # Continuation 4: a captured layer state is read from its dictionary
        # by interned key (before: three attribute reads with a name string
        # built each, per layer); any other layer state object by attribute.
        var captured = py_type(st) is namespace
        var sf = PythonObject(None)
        var session: PythonObject
        var old: PythonObject
        if captured:
            sf = causal_snapshot_fields(mods, st)
            if py_get(sf, nk[].k__causal_admission_failure) is not None:
                return False
            session = py_get(sf, nk[].k__resident_session)
            old = py_get(sf, nk[].k_arithmetic_profile)
        else:
            session = py_call(g[].bi.getattr, st, "_resident_session", PythonObject(None))
            old = py_call(g[].bi.getattr, st, "arithmetic_profile", PythonObject(None))
        if session is not None:
            return False
        if selected is None:
            if profile is not None or old is not None:
                return False
        elif profile is None or not Bool(py=profile == selected):
            return False
        if old is not None and profile is not None and not Bool(py=old == profile):
            return False
        if old is None and (Int(py=py_item(sf, nk[].k_cached_tokens)) if captured else Int(py=st.cached_tokens)) != 0 and (changed or
                _generation_bool(py_get(blk, "_arithmetic_profile_changed", PythonObject(False)))):
            # The ordinary layer path supplies its existing, ordered diagnostic.
            return False
    return True


def _resident_layer_publish(mut mods: NativeModules, blk: PythonObject, st: PythonObject, cached: PythonObject) raises:
    """PACK of one layer after a resident forward: the block's arithmetic
    profile (when it has one) and the cache counter `cached` onto the layer
    state. Continuation 4: a captured layer state is written by interned key
    into its dictionary (before: an attribute write with a name string built
    per word, per layer per token of a resident read); any other layer state
    object by attribute, as before."""
    var nk = causal_keys()
    var profile = py_get(blk, nk[].k__arithmetic_profile)
    if py_type(st) is mods.module("types").SimpleNamespace:
        var sf = causal_snapshot_fields(mods, st)
        if profile is not None:
            py_setitem(sf, nk[].k_arithmetic_profile, profile)
        py_setitem(sf, nk[].k_cached_tokens, cached)
        return
    if profile is not None:
        py_setattr(st, "arithmetic_profile", profile)
    py_setattr(st, "cached_tokens", cached)


def _resident_forward_ok(mut mods: NativeModules, words: CausalModelWords, model: PythonObject, state: PythonObject,
                         mut blocks: List[PythonObject]) raises -> Bool:
    """Whether `forward` runs as the one resident program
    (`causal_lm_session_forward`): a transformer stack with a norm, the
    resident entries, a head the resident session takes, matching arithmetic
    profile metadata and no layer state held by a decode session."""
    # The model half of the admission is typed (`causal_resident_candidate`:
    # the kind, the final norm, the resident entries and the packed head's
    # vendor, read before from the model record and its entry registry on
    # every public forward); the per-layer profile half reads the state.
    if not causal_resident_candidate(words):
        return False
    comptime if not CAUSAL_LM_RESIDENT_FORWARD:
        return False
    return _resident_profiles_ok(mods, model, state, blocks)


def _resident_forward(mut mods: NativeModules, words: CausalModelWords, model: PythonObject, ids: PythonObject, state: PythonObject, output: PythonObject, prepared_key: PythonObject, mut read_plan: Optional[LegacyGenerationReadPlan], admitted_blocks: List[PythonObject]) raises -> PythonObject:
    """`(B, L, V)` logits of `forward` in one resident program: the kept (or
    newly opened) sessions take the state's caches, run the stack, and the
    advanced caches come back into the state."""
    var e = model._native_entries
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    var b = Int(py=ids.shape[0])
    var l = Int(py=ids.shape[1])
    var v = words.vocab
    var int15 = words.head_packed
    var keep = List[PythonObject]()
    var held = Bool(read_plan) and read_plan.value().resident_active
    if held and not _legacy_resident_valid(read_plan.value()):
        # A supported parameters() call may expose/invalidate the model while
        # native execution releases the GIL. Do not reuse or republish its old
        # session cache: release this loan and resume the ordinary per-step path.
        read_plan.value().resident_reuse = False
        read_plan.value().resident_invalidated = True
        _legacy_resident_cancel(mods, model, read_plan.value())
        held = False
    # Each layer is admitted once per forward (once per read with a plan): the
    # key, the open and the export below all read this list.
    var blocks: List[PythonObject]
    if len(admitted_blocks) == words.layers:
        blocks = admitted_blocks.copy()
    elif Bool(read_plan) and len(read_plan.value().blocks) == words.layers:
        blocks = read_plan.value().blocks.copy()
    else:
        blocks = _block_list(mods, model)
    if Bool(read_plan) and len(read_plan.value().blocks) == 0:
        read_plan.value().blocks = blocks.copy()
    var key = _resident_key(mods, model, state, b, int15, blocks) if prepared_key is None else prepared_key
    var lm = PythonObject(None)
    var sessions = PythonObject(None)
    var out = _causal_logits_output(b, l, v, output)
    # The lease (lane causal-lease): a read that holds it keeps it; every
    # other forward takes the LIVE word here, in one compare-and-set, as the
    # last step before the sessions are touched. Nothing between the take
    # and the `try` below can raise. A forward that loses (another holder
    # has the kept sessions, or the model is exposed) holds no lease: it
    # opens private sessions, proceeds, and closes them at the end; it never
    # reads or writes the kept ones (orchestrator decision, 2026-11-25).
    var lease_held = PythonObject(None)
    if held:
        lease_held = read_plan.value().resident_lease
    else:
        lease_held = causal_model_lease(mods, model)
    var lease = lease_held.downcast_value_ptr[CausalLease]()
    var persist = held
    if not held:
        persist = causal_lease_take(lease[].live) == CAUSAL_LEASE_TAKEN
    var retain = persist and Bool(read_plan) and read_plan.value().resident_reuse
    var kept = PythonObject(None)
    if persist and not held:
        kept = lease[].kept
    if retain:
        read_plan.value().resident_lease = lease_held
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
            _resident_load_states(mods, e, lm, sessions, state, keep)
            mods.released()
        else:
            if kept is not None:
                if retain:
                    read_plan.value().resident_lm = kept[1]
                    read_plan.value().resident_sessions = kept[2]
                lease[].kept = PythonObject(None)
                _resident_close(kept[1])
                mods.released()
                if retain:
                    read_plan.value().resident_lm = PythonObject(None)
                    read_plan.value().resident_sessions = PythonObject(None)
            var opened = _resident_open_all(mods, model, e, state, b, l, int15, keep, blocks)
            lm = opened[0]
            sessions = opened[1]
            if retain:
                read_plan.value().resident_lm = lm
                read_plan.value().resident_sessions = sessions
        var cached = PythonObject(_resident_group(e, lm, CAUSAL_SESSION_LM_FORWARD, [_addr(ids, keep), _addr(out, keep)], [l]))
        mods.released()
        _ = out  # keep the Python object alive through the native call above
        if retain:
            # The read keeps these sessions for its next forward, which skips
            # the upload, and nothing reads the host caches between tokens of
            # one read (the stream is busy). Export the device KV once, when
            # the read publishes or cancels its sessions, instead of after
            # every token: the final host bytes are those of the last
            # per-token export (same layers, same order).
            read_plan.value().export_owed = True
            read_plan.value().export_state = state
            read_plan.value().export_batch = b
            var layer_states = state.layers
            for i in range(len(sessions)):  # driver-loop: over layer decode sessions
                _resident_layer_publish(mods, _block_at(mods, model, blocks, i), layer_states[i], cached)
        else:
            _resident_export_states(mods, model, e, lm, sessions, state, blocks, b, cached)
    except error:
        mods.released()
        if retain:
            _legacy_resident_cancel(mods, model, read_plan.value())
            raise error
        if persist:
            lease[].kept = PythonObject(None)
            _ = causal_lease_release(lease[].live)
        if lm is not None:
            _resident_close(lm)
            mods.released()
        raise error
    if retain:
        # The upload, the KV export and the cache-owner publication are
        # deferred to the read's finish (or cancel).
        _ = read_plan
    elif persist:
        lease[].kept = Python.tuple(key, lm, sessions)
        if causal_lease_release(lease[].live):
            # Exposed while this forward ran: nothing is kept. No taker can
            # reach the tuple (an exposed word is never taken).
            lease[].kept = PythonObject(None)
            _resident_close(lm)
            mods.released()
    else:
        _resident_close(lm)
        mods.released()
    _ = lease_held
    _ = keep^
    return out


def _use_native_generation_engine(mut mods: NativeModules, words: CausalModelWords, model: PythonObject, state: PythonObject,
                                  mut resident_verdict: Int, mut resident_blocks: List[PythonObject]) raises -> Bool:
    # `resident_verdict` receives 1/0 when the resident check ran (else stays
    # -1) and `resident_blocks` its admitted layers, so the caller decides the
    # route once per public call.
    # Select once at a public operation boundary, never in the token/layer
    # loop. No failed native admission/execution is caught and retried through
    # the object route: a selected engine keeps its original refusal semantics.
    # -D MOJOLEARN_GENERATION_NATIVE_ENGINE_FORCE / _OFF (was the run-time
    # MOJOLEARN_GENERATION_NATIVE_ENGINE=1 / 0, default auto).
    comptime if GENERATION_NATIVE_ENGINE_FORCE:
        return True
    comptime if GENERATION_NATIVE_ENGINE_OFF:
        return False
    # TOMBSTONE: MOJOLEARN_NEURAL_NATIVE_DISPATCH_OFF (removed without measurement: plumbing, rule 2026-11-22; the by-name CPython entry calls ran the same compiled entries with the same words) deleted 2026-11-23 by lane/ab-prep-neural; code recoverable at 4d76a9e16e.
    # Restore: git apply ~/mojolearn-evidence/experiments-removed/MOJOLEARN_NEURAL_NATIVE_DISPATCH_OFF.patch; record in docs/TOMBSTONES.md.
    # The model half is typed (`causal_engine_candidate`: a local fp32 model
    # of a known kind with its providers admitted; before, seven reads of the
    # model record and its entry registry per public forward).
    if not causal_engine_candidate(words):
        return False
    # Existing resident execution already owns a separate supported native
    # program. Its route, distributed execution and packed arithmetic remain
    # supported; they are explicit remaining migration scopes, not refusals.
    var resident = _resident_forward_ok(mods, words, model, state, resident_blocks)
    resident_verdict = 1 if resident else 0
    return not resident


def _owner_call_layers(mut mods: NativeModules, held: PythonObject, model: PythonObject, bind: Bool) raises -> GenerationCallLayers:
    """UNPACK of a local (engine or per-layer) forward's model side.

    Per public operation (once per read, never per token): this operation's
    block captures and their profile words (`GenerationCallLayers`).
    Once per model generation: the owner's weight snapshot, built here at
    the first local forward (the position the per-operation build had, so a
    refusal keeps its place and words) and kept by the owner. `bind`
    resolves the layer route for the per-layer run; the engine never runs
    the layers itself and never resolves it. Before lane dw-causal every
    public forward and every read rebuilt the whole snapshot (per layer
    about a dozen field reads and one storage pin per weight) and the engine
    then deep-copied it."""
    var owner = held.downcast_value_ptr[CausalModelOwner]()
    var blocks = List[PythonObject](capacity=owner[].words.layers)
    for i in range(owner[].words.layers):  # driver-loop: one operation's retained model owners
        blocks.append(_block(mods, model, i))
    if not owner[].snapshot:
        var built = GenerationOperationSnapshot(model, model._native_entries, blocks)
        causal_model_snapshot_keep(held, built^)
    if bind and not owner[].snapshot.value().layer_route.bound:
        # Idempotent (the same words from the same registry); written with
        # the GIL held, read only by operations that passed this line.
        owner[].snapshot.value().bind_layer_entry()
    var call = GenerationCallLayers(blocks, owner[].words.kind == CAUSAL_LAYER_TRANSFORMER)
    mods.released()
    return call^


def _engine_admission(mut mods: NativeModules, words: CausalModelWords, held: PythonObject, model: PythonObject, state: PythonObject, batch: Int, length: Int,
                      resident_refused: Bool = False) raises -> GenerationEngineAdmission:
    # `resident_refused`: the caller's route decision for this same call
    # already found the resident route unavailable; skip the repeat check.
    # Once selected, admission failures propagate. The boundary selector
    # handles unsupported routes before execution, without exception fallback.
    _state(mods, words, model, state, batch, length)
    try:
        if words.parallel:
            raise Error("CausalLM: native generation engine distributed execution is pending")
        if not resident_refused:
            var unused_blocks = List[PythonObject]()
            if _resident_forward_ok(mods, words, model, state, unused_blocks):
                raise Error("CausalLM: native generation engine does not replace the existing resident route")
        var call = _owner_call_layers(mods, held, model, False)
        var model_owner = held.downcast_value_ptr[CausalModelOwner]()
        var windowed_result5 = GenerationEngineAdmission(
            model_owner[].snapshot.value(), held, call^,
            words.limited, words.max_positions, state, batch, length)
        # The state half (lane dw-causal, continuation 4): layer by layer, in
        # layer order, on the state owner's words where they are current
        # (`causal_state_layer_unpack`: one pointer compare per buffer); the
        # object admission for any other layer. The admission holds the
        # owner's handle (the buffers' pins) for the whole call.
        var state_held = causal_state_owner(mods, state)
        windowed_result5.state_owner = state_held
        # The layer kind word of the model owner (the engine admission never
        # resolves the snapshot's layer route).
        var kind = words.kind
        var unowned = Optional[CausalLayerStateWords]()
        for i in range(windowed_result5.layer_count()):  # driver-loop: over the model's layers, once per engine forward
            var st = windowed_result5.layer_states[i]
            var st_fields = PythonObject(None)
            try:
                st_fields = causal_state_layer_unpack(mods, state_held, i, st, kind)
            except:
                # The object admission meets the same problem at this layer's
                # position and defers it there.
                st_fields = PythonObject(None)
            if st_fields is not None:
                var state_owner = state_held.downcast_value_ptr[CausalStateOwner]()
                windowed_result5.admit_layer(model_owner[].snapshot.value(), i, st, st_fields,
                    state_owner[].layers[state_owner[].slots[i]], True)
            else:
                if not unowned:
                    unowned = Optional(CausalLayerStateWords())
                windowed_result5.admit_layer(model_owner[].snapshot.value(), i, st, st_fields,
                    unowned.value(), False)
        _ = state_held
        mods.released()
        return windowed_result5^
    except problem:
        mods.released()
        py_setattr(state, "_usable", PythonObject(False))
        raise problem


def _run_engine(mut mods: NativeModules, words: CausalModelWords, held: PythonObject, model: PythonObject, ids: PythonObject, state: PythonObject,
                resident_refused: Bool = False) raises -> PythonObject:
    # The engine route of a forward, in three steps:
    #   UNPACK: `_engine_admission` (the state check on typed words, then the
    #     owner's weight views, this call's block captures, the state buffers
    #     and the endpoint as a `NativeGenerationModel`), the ids address and
    #     the result array.
    #   RUN: `owner.native.forward_to`: typed words and addresses only.
    #   PACK: `owner.publish()` (positions, counters, reports) and the result.
    _causal_array_guard()
    var shape = ids.shape
    var batch = Int(py=shape[0])
    var length = Int(py=shape[1])
    var owner = _engine_admission(mods, words, held, model, state, batch, length, resident_refused)
    var keep = List[PythonObject]()
    var address = _addr(ids, keep)
    # Row N199: allocate the result first and let the engine's head write
    # into it (formerly a List[Float32] result copied into a new Array).
    var result = PythonObject(None)
    try:
        result = generation_array([batch, length, owner.native.vocab])
        var out_address = Int(py=result._addr)
        # Common endpoints own per-call GIL release, lock acquisition and
        # session reservation. The native layer loop enters GIL-held.
        owner.native.forward_to(address, length, out_address, owner.native.batch * length * owner.native.vocab)
        mods.released()
    except problem:
        mods.released()
        owner.publish()
        raise problem
    owner.publish()
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
    # The handle of the model's lease (`CausalLease`) while this read holds
    # it (`resident_active`); None otherwise.
    var resident_lease: PythonObject
    var resident_lm: PythonObject
    var resident_sessions: PythonObject
    # The read's admitted layer fields (resident route), filled once by the
    # resident check or the first resident forward; cleared with the loan.
    var blocks: List[PythonObject]
    # A retained read defers the per-token KV export to its finish/cancel.
    var export_owed: Bool
    var export_state: PythonObject
    var export_batch: Int
    # The captured state's field dictionary (continuation 4): each token's
    # position check and position advance read and write it by interned key
    # (before: two attribute reads and one attribute write, each building
    # its name, per token).
    var state_fields: PythonObject

    def __init__(out self, mut mods: NativeModules, words: CausalModelWords, state: PythonObject, batch: Int, resident: Bool) raises:
        self.state_fields = causal_snapshot_fields(mods, state)
        self.resident = resident
        self.batch = batch
        self.limited = words.limited
        self.max_positions = words.max_positions if self.limited else 0
        self.capacity = Int(py=py_item(self.state_fields, causal_keys()[].k_max_tokens))
        self.resident_key = PythonObject(None)
        self.logits = PythonObject(None)
        self.logits_length = -1
        self.resident_reuse = False
        self.resident_uploads_skipped = 0
        self.resident_invalidated = False
        self.resident_active = False
        self.resident_lease = PythonObject(None)
        self.resident_lm = PythonObject(None)
        self.resident_sessions = PythonObject(None)
        self.blocks = List[PythonObject]()
        self.export_owed = False
        self.export_state = PythonObject(None)
        self.export_batch = 0

    def clear_resident(mut self):
        self.resident_active = False
        self.resident_lease = PythonObject(None)
        self.resident_lm = PythonObject(None)
        self.resident_sessions = PythonObject(None)
        self.blocks.clear()
        self.export_owed = False
        self.export_state = PythonObject(None)

    def positions(self, batch: Int, length: Int) raises:
        # Position/capacity checks retain their per-step execution point, before
        # the forward try block (a refusal does not poison otherwise valid state).
        if batch != self.batch:
            raise Error("CausalLM: state batch/layer counts disagree with model and input")
        var have = Int(py=py_item(self.state_fields, causal_keys()[].k_positions))
        if length < 0 or have < 0:
            raise Error("CausalLM: negative sequence position")
        if self.limited and have + length > self.max_positions:
            raise Error("CausalLM: max_tokens or sequence exceeds max_positions")
        if have + length > self.capacity:
            raise Error("CausalLM: sequence exceeds allocated state capacity")

    def advance(self, length: Int) raises:
        # PACK of one forward of the read: the state's position, by interned
        # key into the captured state's dictionary.
        var nk = causal_keys()
        py_setitem(self.state_fields, nk[].k_positions,
                   PythonObject(Int(py=py_item(self.state_fields, nk[].k_positions)) + length))


def _resident_export_states(mut mods: NativeModules, model: PythonObject, e: PythonObject, lm: PythonObject,
                            sessions: PythonObject, state: PythonObject, blocks: List[PythonObject], b: Int,
                            cached: PythonObject) raises:
    # Device KV of every layer session into the state's host caches (op 22).
    # `cached` None leaves cached_tokens and the profile as they are.
    # Continuation 4: a layer whose state the state owner's words describe
    # takes its cache addresses from them (typed checks, same words); any
    # other layer state keeps the object admission (`_mutable`).
    var keep = List[PythonObject]()
    var copies = List[PythonObject]()
    var nk = causal_keys()
    var state_held = causal_state_owner(mods, state)
    var layer_states = state.layers
    for i in range(len(sessions)):  # driver-loop: over layer decode sessions
        var blk = _block_at(mods, model, blocks, i)
        var st = layer_states[i]
        var sf = causal_state_layer_unpack(mods, state_held, i, st, CAUSAL_LAYER_TRANSFORMER)
        var window = Int(py=blk["window"])
        var pk = 0
        var pv = 0
        if sf is not None:
            var cap = Int(py=py_item(sf, nk[].k_max_tokens)) if window == 0 else window
            var cache_shape: List[Int] = [b * Int(py=blk["n_kv_heads"]) * cap * Int(py=blk["head_dim"])]
            var state_owner = state_held.downcast_value_ptr[CausalStateOwner]()
            pk = causal_state_buffer_address(state_owner[].layers[state_owner[].slots[i]].buffers[CAUSAL_STATE_K], cache_shape, "k_cache")
            pv = causal_state_buffer_address(state_owner[].layers[state_owner[].slots[i]].buffers[CAUSAL_STATE_V], cache_shape, "v_cache")
        else:
            var cap = Int(py=st.max_tokens) if window == 0 else window
            var cache_n = b * Int(py=blk["n_kv_heads"]) * cap * Int(py=blk["head_dim"])
            pk = _mutable(st.k_cache, [cache_n], keep, copies, "k_cache")
            pv = _mutable(st.v_cache, [cache_n], keep, copies, "v_cache")
        _ = _resident_layer_call(e, lm, i, CAUSAL_SESSION_DECODE_EXPORT, [pk, pv], List[Int]())
        mods.released()
        if cached is not None:
            _resident_layer_publish(mods, blk, st, cached)
    _writeback(copies)
    _ = state_held
    _ = keep^


def _legacy_resident_export(mut mods: NativeModules, model: PythonObject, mut plan: LegacyGenerationReadPlan) raises:
    # The one deferred KV export of a read that kept its sessions.
    if not plan.export_owed:
        return
    plan.export_owed = False
    var e = model._native_entries
    _resident_export_states(mods, model, e, plan.resident_lm, plan.resident_sessions,
                            plan.export_state, plan.blocks, plan.export_batch, PythonObject(None))


def _legacy_resident_valid(mut plan: LegacyGenerationReadPlan) raises -> Bool:
    # The holder's test of its loan, before each token and at finish: one
    # load of the live lease word. Only `parameters()` ends a loan from
    # outside (it sets the exposed bit; it never touches the kept sessions
    # of a running holder). Before lane causal-lease this read the live
    # record's `_parameters_exposed` and `_resident_sessions` (two dictionary
    # reads, each building its key) per token.
    var lease = plan.resident_lease.downcast_value_ptr[CausalLease]()
    return (causal_lease_load(lease[].live) & CAUSAL_LEASE_EXPOSED) == 0


def _legacy_resident_cancel(mut mods: NativeModules, model: PythonObject, mut plan: LegacyGenerationReadPlan):
    # Failure cleanup must neither replace the operation error nor leave the
    # model busy. Pins remain live until every owner has had its close attempt.
    var lease_held = plan.resident_lease
    var lm = plan.resident_lm
    var sessions = plan.resident_sessions
    plan.resident_active = False
    try:
        # Bring the deferred device KV home before the sessions close, so a
        # fresh open (or a checkpoint) sees the advanced caches.
        _legacy_resident_export(mods, model, plan)
    except:
        mods.released()
    # The order of the former publication: nothing kept, the sessions
    # closed, then the lease given back.
    try:
        if lease_held is not None:
            lease_held.downcast_value_ptr[CausalLease]()[].kept = PythonObject(None)
    except:
        mods.released()
    try:
        _resident_read_cleanup(lm)
        mods.released()
    except:
        mods.released()
    try:
        if lease_held is not None:
            _ = causal_lease_release(lease_held.downcast_value_ptr[CausalLease]()[].live)
    except:
        mods.released()
    _ = sessions
    plan.clear_resident()


def _legacy_resident_finish(mut mods: NativeModules, model: PythonObject, mut read_plan: Optional[LegacyGenerationReadPlan]) raises:
    if not read_plan or not read_plan.value().resident_active:
        return
    ref plan = read_plan.value()
    try:
        if not _legacy_resident_valid(plan):
            _legacy_resident_cancel(mods, model, plan)
            return
        _legacy_resident_export(mods, model, plan)
        var lease = plan.resident_lease.downcast_value_ptr[CausalLease]()
        lease[].kept = Python.tuple(plan.resident_key, plan.resident_lm, plan.resident_sessions)
        if causal_lease_release(lease[].live):
            # Exposed between the test above and the release: nothing is
            # kept (no taker can reach the tuple), the sessions close.
            lease[].kept = PythonObject(None)
            _resident_read_cleanup(plan.resident_lm)
            mods.released()
    except error:
        _legacy_resident_cancel(mods, model, plan)
        raise error
    plan.clear_resident()


def _legacy_read_captured(mut mods: NativeModules, model: PythonObject, state: PythonObject) raises -> Bool:
    var namespace = mods.module("types").SimpleNamespace
    return (py_type(model) is namespace and py_type(state) is namespace
            and py_contains(causal_snapshot_fields(mods, model), "_causal_original")
            and py_contains(causal_snapshot_fields(mods, state), "_causal_original"))


def _run(mut mods: NativeModules, words: CausalModelWords, held: PythonObject, model: PythonObject, ids: PythonObject, state: PythonObject, remote_step: Bool = False) raises -> PythonObject:
    # The route is decided once here, from the model owner's typed words and
    # the state; the resident verdict and its admitted layers are passed down
    # instead of re-running the check.
    var verdict = -1
    var resident_blocks = List[PythonObject]()
    if _use_native_generation_engine(mods, words, model, state, verdict, resident_blocks):
        return _run_engine(mods, words, held, model, ids, state, verdict == 0)
    var call_layers: Optional[GenerationCallLayers] = None
    var read_plan: Optional[LegacyGenerationReadPlan] = None
    # No state owner handle is held here: the per-layer route finds it once
    # (`_run_snapshot`), the other routes never need it.
    return _run_prepared(mods, words, held, model, ids, state, call_layers, read_plan, remote_step, False, False, verdict, resident_blocks,
                         PythonObject(None))


def _run_prepared(mut mods: NativeModules, words: CausalModelWords, held: PythonObject, model: PythonObject, ids: PythonObject, state: PythonObject,
                  mut call_layers: Optional[GenerationCallLayers], mut read_plan: Optional[LegacyGenerationReadPlan], remote_step: Bool,
                  admitted: Bool, retain_resident: Bool, resident_verdict: Int,
                  resident_blocks: List[PythonObject], state_held: PythonObject) raises -> PythonObject:
    # `admitted`: the caller (the generation loop) already guarded the Array
    # class for this GIL-release window; do not repeat the full guard.
    # `state_held`: the state's native owner when the caller holds it for a
    # whole read (`GenerationHandle.state_held`), else None.
    if not admitted:
        _causal_array_guard()
    var shape = ids.shape
    var b = Int(py=shape[0])
    var l = Int(py=shape[1])
    if read_plan:
        read_plan.value().positions(b, l)
    else:
        _state(mods, words, model, state, b, l)
    try:
        if words.parallel:
            # The parallel door's own operation: it receives the public
            # origins and the captures, and runs its workers.
            var parallel = py_get(model._native_entries, "parallel_call")
            var windowed_result4 = py_call(parallel, "run", Python.list(causal_model_origin(mods, model), ids, causal_state_origin(mods, state), remote_step, model, state))
            mods.released()
            return windowed_result4^
        # Layer fields admitted by the resident check; the resident forward
        # (and, with a read plan, every later forward of this read) reuses them.
        var admitted_blocks = List[PythonObject]()
        var resident: Bool
        if read_plan:
            resident = read_plan.value().resident
        elif resident_verdict >= 0:
            # Decided by the caller for this same call (no state change since).
            resident = resident_verdict == 1
            admitted_blocks = resident_blocks.copy()
        else:
            resident = _resident_forward_ok(mods, words, model, state, admitted_blocks)
        if not read_plan and admitted and _legacy_read_captured(mods, model, state):
            read_plan = LegacyGenerationReadPlan(mods, words, state, b, resident)
            if resident:
                read_plan.value().blocks = admitted_blocks.copy()
            if resident and retain_resident and words.block_operations:
                # Once per read: only a free, unexposed lease can be kept
                # across the read's tokens (the live word; the take itself
                # is the first forward's).
                var lease_held = causal_model_lease(mods, model)
                if causal_lease_load(lease_held.downcast_value_ptr[CausalLease]()[].live) == 0:
                    read_plan.value().resident_reuse = _resident_read_disjoint_state(mods, model, state)
                _ = lease_held
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
                    if len(plan.blocks) != words.layers:
                        plan.blocks = _block_list(mods, model)
                    plan.resident_key = _resident_key(mods, model, state, b, words.head_packed, plan.blocks)
                resident_key = plan.resident_key
        if resident:
            var logits = _resident_forward(mods, words, model, ids, state, output, resident_key, read_plan, admitted_blocks)
            if read_plan:
                read_plan.value().logits = logits
                read_plan.value().logits_length = l
                read_plan.value().advance(l)
            else:
                py_setattr(state, "positions", PythonObject(Int(py=state.positions) + l))
            return logits
        if not call_layers:
            # Only on the local nonresident route, after admission and route
            # checks (parallel / resident execution retains its own owners):
            # once per operation this operation's block captures; once per
            # model generation the owner's weight snapshot.
            call_layers = _owner_call_layers(mods, held, model, True)
        var out = _run_snapshot(mods, held.downcast_value_ptr[CausalModelOwner]()[].snapshot.value(),
                                call_layers.value(), ids, state, b, l, output, state_held)
        if read_plan:
            read_plan.value().logits = out
            read_plan.value().logits_length = l
            read_plan.value().advance(l)
        else:
            py_setattr(state, "positions", PythonObject(Int(py=state.positions) + l))
        return out
    except error:
        mods.released()
        py_setattr(state, "_usable", PythonObject(False))
        raise error


def _run_snapshot(mut mods: NativeModules, snapshot: GenerationOperationSnapshot, call: GenerationCallLayers,
                  ids: PythonObject, state: PythonObject,
                  b: Int, l: Int, output: PythonObject, held_state: PythonObject) raises -> PythonObject:
    # `snapshot` is the model owner's (weights and layer words of this model
    # generation; the caller holds the owner's handle across this call);
    # `call` is this operation's block captures and profile words.
    var x = _causal_model_embedding_prepared(snapshot.entries, snapshot.width, snapshot.vocab,
                                            b, l, False, snapshot.embedding, ids)
    mods.released()
    # The state's native owner (None for an uncaptured state): the read's
    # handle when the caller holds one (`held_state`: no lookup per token),
    # else one read per forward. Layer UNPACK stays interleaved with layer
    # RUN: two layers may share one state object, so layer i's counter is
    # read after layer i-1 ran.
    var state_held = held_state
    if state_held is None:
        state_held = causal_state_owner(mods, state)
    var layer_states = state.layers
    for i in range(len(snapshot.layers)):  # driver-loop: over retained native layer descriptions
        var st = layer_states[i]
        var st_fields = causal_state_layer_unpack(mods, state_held, i, st, snapshot.layer_route.kind)
        if st_fields is not None:
            var state_owner = state_held.downcast_value_ptr[CausalStateOwner]()
            x = _causal_layer_run_owned(mods, snapshot.layer_route, snapshot.entries, snapshot.width, b, l,
                snapshot.layers[i], snapshot.head.packed, snapshot.planes, x, st_fields,
                state_owner[].layers[state_owner[].slots[i]], call.blocks[i], call.profiles[i], call.changed[i])
        else:
            x = _causal_layer_run_bound(mods, snapshot.layer_route, snapshot.entries, snapshot.width, b, l,
                snapshot.layers[i], snapshot.head.packed, snapshot.planes, x, st, False,
                call.blocks[i], call.profiles[i], call.changed[i], False)
    _ = state_held
    var out = _causal_model_head_prepared(snapshot.head, x, b, l, False, output)
    mods.released()
    return out


def causal_lm_forward_binding(mut mods: NativeModules, model: PythonObject, value: PythonObject, supplied_state: PythonObject, step: PythonObject) raises -> PythonObject:
    # Lane dw-causal (2026-11-21): forward and step as UNPACK / RUN / PACK.
    #   UNPACK: the model's native owner (one read of the capture; its typed
    #     words decide the route and check the state: the decision reads no
    #     model field and no entry registry word by name), the ids, the state.
    #     Still read from the capture inside each route's own UNPACK and
    #     PACK: the weights, the layer states and the resident session lease
    #     (owner stages S2 and S3 of the lane's design).
    #   RUN, typed on every local route (`_run`): the engine's
    #     `NativeGenerationModel.forward_to`, the resident program's one
    #     session-table call, or per layer `_causal_layer_request_run` between
    #     the embedding and head table calls. The parallel route is the
    #     parallel door's own operation.
    #   PACK: the route's publication (positions, counters, reports) and, for
    #     a step, the (B, V) shape of the result.
    var model_held = causal_model_owner(mods, model)
    ref owner = model_held.downcast_value_ptr[CausalModelOwner]()[]
    var state = supplied_state
    var one = Bool(py=step)
    var ids = generation_ids(value, owner.words.vocab, one)
    if state is None:
        if one:
            raise Error("CausalLM.step: state is required")
        state = causal_lm_allocate_binding(mods, model, ids.shape[0], ids.shape[1])
    var out = PythonObject(None)
    var parallel = PythonObject(None)
    if owner.words.parallel:
        parallel = py_get(model._native_entries, "parallel_call")
    try:
        out = _run(mods, owner.words, model_held, model, ids, state, one)
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
        py_setattr(out, "shape", py_call(g[].bi.tuple, py_ints([Int(py=ids.shape[0]), owner.words.vocab])))
        py_setattr(out, "strides", py_call(g[].bi.tuple, py_ints([owner.words.vocab * 4, 4])))
        py_setattr(out, "ndim", PythonObject(2))
    _ = model_held
    return out


def causal_lm_forward_binding(model: PythonObject, value: PythonObject, supplied_state: PythonObject, step: PythonObject) raises -> PythonObject:
    # Classical and other callers keep their per-call module checks.
    var mods = NativeModules()
    return causal_lm_forward_binding(mods, model, value, supplied_state, step)


def _session_call_ref(entry: EntryRef, operation: Int, addresses: List[Int], words: List[Int],
    floats: List[Float64]) raises -> Int:
    # One session-table call (core/causal_lm_entry_table_sessions.mojo)
    # through a resolved session entry: plain words, no CPython call. The GIL
    # is released around the device work as the former Python session entries
    # released it.
    var result = 0
    with GILReleased(Python()):
        result = causal_sessions_call(entry, operation, addresses, words, floats)
    return result


def _session_call(e: PythonObject, family: Int, operation: Int, addresses: List[Int], words: List[Int],
    floats: List[Float64]) raises -> Int:
    # A one-off session call by the model's admitted key (resolved here); the
    # per-step calls go through their session owner's resolved entry.
    return _session_call_ref(causal_sessions_ref(family, causal_entry_key(e)), operation, addresses, words, floats)


def _resident_layer_call(e: PythonObject, lm: PythonObject, index: Int, operation: Int, addresses: List[Int],
    scalars: List[Int]) raises -> Int:
    # One layer decode session's operation, by its handle in the owner.
    var owner = lm.downcast_value_ptr[CausalSessionHandles]()
    var words: List[Int] = [owner[].layers[index]]
    for i in range(len(scalars)):  # driver-loop: admitted session scalar span
        words.append(scalars[i])
    var result = _session_call_ref(owner[].entry, operation, addresses, words, List[Float64]())
    _ = lm
    return result


def _resident_group(e: PythonObject, lm: PythonObject, operation: Int, addresses: List[Int],
    scalars: List[Int]) raises -> Int:
    # The model session and every layer session in one grouped call
    # (LM_RUN, LM_FORWARD): words [model, n, layer handles..., scalars...].
    var owner = lm.downcast_value_ptr[CausalSessionHandles]()
    var words: List[Int] = [owner[].primary, len(owner[].layers)]
    for i in range(len(owner[].layers)):  # driver-loop: over layer decode sessions
        words.append(owner[].layers[i])
    for i in range(len(scalars)):  # driver-loop: admitted session scalar span
        words.append(scalars[i])
    var result = _session_call_ref(owner[].entry, operation, addresses, words, List[Float64]())
    _ = lm
    return result


# TOMBSTONE: transformer_resident_lm_read / transformer_resident_lm_sampling_read (REFACTORED: CausalLM reads run as the decode program, transformer/causal_lm_program.mojo) deleted 2026-11-09 by lane/x45c-causal-generate-program-decode; code recoverable at 5b5490960d.
# Restore: git apply experiments/removed/CAUSAL_GENERATE_RESIDENT_READS.patch; record in docs/TOMBSTONES.md.
# (Moved here from providers/python/transformer_resident_pyapi.mojo, deleted
# by lane causal-lm-sessions-native when its provider arms became session-table calls.)


def _resident_close(lm: PythonObject) raises:
    # The handle owner closes the model session, then every layer session.
    if lm is not None:
        lm.downcast_value_ptr[CausalSessionHandles]()[].close()


def _resident_open_all(mut mods: NativeModules,
    model: PythonObject, e: PythonObject, state: PythonObject, b: Int, l: Int, int15: Bool,
    mut keep: List[PythonObject],
) raises -> PythonObject:
    var blocks = List[PythonObject]()
    return _resident_open_all(mods, model, e, state, b, l, int15, keep, blocks)


def _resident_open_all(mut mods: NativeModules,
    model: PythonObject, e: PythonObject, state: PythonObject, b: Int, l: Int, int15: Bool,
    mut keep: List[PythonObject], blocks: List[PythonObject],
) raises -> PythonObject:
    """Open the model's resident sessions on `state`: one decode session per
    layer (weights and the layer's caches copied to the device) and the
    model session (embedding, norm, head). Returns (lm, sessions): `lm` the
    native handle owner (core/causal_session_handles.mojo), `sessions` its
    layer handle words as a list (the resident cache record's shape)."""
    var lm = PythonObject(alloc=CausalSessionHandles(causal_entry_key(e), CAUSAL_SESSIONS_TRANSFORMER,
        _session_call(e, CAUSAL_SESSIONS_TRANSFORMER, CAUSAL_SESSION_LM_CREATE, List[Int](), List[Int](), List[Float64]()),
        CAUSAL_SESSION_LM_CLOSE))
    mods.released()
    var owner = lm.downcast_value_ptr[CausalSessionHandles]()
    try:
        for i in range(len(model._blocks)):  # driver-loop: over model layers
            var blk = _block_at(mods, model, blocks, i)
            var st = state.layers[i]
            var a = _transformer_arg_words(model, blk, st, PythonObject(None), PythonObject(None), b, l, int15, keep)
            var open_a = List[Int]()
            for j in range(1, 12):  # driver-loop: over layer argument spans
                open_a.append(a[j])
            for j in range(13, len(a)):  # driver-loop: over layer argument spans
                open_a.append(a[j])
            var ss = _session_call_ref(owner[].entry, CAUSAL_SESSION_DECODE_CREATE, List[Int](), List[Int](), List[Float64]())
            owner[].layers.append(ss)
            var p = _transformer_param_words(blk, st, b, l)
            var open_p: List[Int] = [ss]
            for j in range(len(p)):  # driver-loop: over layer shape parameters
                if j != 1:
                    open_p.append(p[j])
            _ = _session_call_ref(owner[].entry, CAUSAL_SESSION_DECODE_OPEN15 if int15 else CAUSAL_SESSION_DECODE_OPEN,
                open_a, open_p, List[Float64]())
            mods.released()
        # A model without a final norm passes 0 (admitted only under
        # MOJOLEARN_IDN_LM_RESIDENT_GENERATE; see `_resident_norm_ok`).
        var top: List[Int] = [_weight_addr(model._embed, keep), 0 if model._norm is None else _weight_addr(model._norm, keep)]
        if int15:
            for i in range(3):  # driver-loop: over int15 head plane triple
                top.append(_weight_addr(model._head_int15[i], keep))
        else:
            top.append(0 if model._head is model._embed else _weight_addr(model._head, keep))
        var tp: List[Int] = [owner[].primary, b, Int(py=model.vocab_size), Int(py=model.d_model)]
        _ = _session_call_ref(owner[].entry, CAUSAL_SESSION_LM_OPEN15 if int15 else CAUSAL_SESSION_LM_OPEN,
            top, tp, [Float64(py=model.norm_eps)])
        mods.released()
    except error:
        mods.released()
        _resident_read_cleanup(lm)
        mods.released()
        raise error
    return Python.tuple(lm, py_ints(owner[].layers))


def _resident_key(mut mods: NativeModules, model: PythonObject, state: PythonObject, b: Int, int15: Bool) raises -> PythonObject:
    var blocks = List[PythonObject]()
    return _resident_key(mods, model, state, b, int15, blocks)


def _resident_key(mut mods: NativeModules, model: PythonObject, state: PythonObject, b: Int, int15: Bool,
                  blocks: List[PythonObject]) raises -> PythonObject:
    """What a kept resident session set must match: the rows, every layer's
    cache capacity/profile, the head format and the entries it was opened with."""
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    var caps = g[].bi.list()
    for i in range(len(state.layers)):  # driver-loop: over model layers
        var profile = py_get(_block_at(mods, model, blocks, i), "_arithmetic_profile", PythonObject(None))
        _ = py_call(caps.append, Python.tuple(state.layers[i].max_tokens, profile))
    var identity = py_get(_standalone_record_fields(mods, model), "_causal_entries_identity", model._native_entries)
    return Python.tuple(b, py_call(g[].bi.tuple, caps), int15, py_call(g[].bi.id, identity))


def _resident_cache_address(buffer: CausalBufferWords) raises -> Int:
    # `_addr` on owner words: the layout check, then the address (0 when the
    # state has no buffer).
    causal_state_buffer_contiguous(buffer)
    return buffer.address if buffer.present else 0


def _resident_load_states(mut mods: NativeModules, e: PythonObject, lm: PythonObject, sessions: PythonObject, state: PythonObject, mut keep: List[PythonObject]) raises:
    """Each kept layer session takes this call's caches (the state half of
    an open; the weights stay where they are). Continuation 4: from the
    state owner's words where they are current for the layer (the same
    layout check and address as `_addr`; no buffer object read), else from
    the buffer objects."""
    var nk = causal_keys()
    var state_held = causal_state_owner(mods, state)
    var layer_states = state.layers
    for i in range(len(sessions)):  # driver-loop: over layer decode sessions
        var st = layer_states[i]
        var sf = causal_state_layer_unpack(mods, state_held, i, st, CAUSAL_LAYER_TRANSFORMER)
        var a = List[Int](capacity=2)
        var cached = 0
        if sf is not None:
            var state_owner = state_held.downcast_value_ptr[CausalStateOwner]()
            var slot = state_owner[].slots[i]
            a.append(_resident_cache_address(state_owner[].layers[slot].buffers[CAUSAL_STATE_K]))
            a.append(_resident_cache_address(state_owner[].layers[slot].buffers[CAUSAL_STATE_V]))
            cached = Int(py=py_item(sf, nk[].k_cached_tokens))
        else:
            a.append(_addr(st.k_cache, keep))
            a.append(_addr(st.v_cache, keep))
            cached = Int(py=st.cached_tokens)
        var p: List[Int] = [cached]
        _ = _resident_layer_call(e, lm, i, CAUSAL_SESSION_DECODE_LOAD, a, p)
    _ = state_held


def causal_lm_resident_binding(mut mods: NativeModules, model: PythonObject, value: PythonObject, maximum: PythonObject, last_logits: PythonObject, supplied_state: PythonObject) raises -> PythonObject:
    var e = model._native_entries
    if not py_contains(e, "resident") or not _resident_norm_ok(model):
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
    var blocks = List[PythonObject]()
    if not _resident_profiles_ok(mods, model, state, blocks):
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
    var lease_held = causal_model_lease(mods, model)
    var lease = lease_held.downcast_value_ptr[CausalLease]()
    var key = _resident_key(mods, model, state, b, int15, blocks)
    var lm = PythonObject(None)
    var sessions = PythonObject(None)
    # The take: one compare-and-set on the live word, the last step before
    # the sessions are touched (nothing between it and the `try` raises). A
    # call that loses holds no lease: private sessions, closed at the end.
    var persist = causal_lease_take(lease[].live) == CAUSAL_LEASE_TAKEN
    var kept = PythonObject(None)
    if persist:
        kept = lease[].kept
    try:
        if kept is not None and Bool(py=kept[0] == key):
            lm = kept[1]
            sessions = kept[2]
            _resident_load_states(mods, e, lm, sessions, state, keep)
            mods.released()
        else:
            if kept is not None:
                lease[].kept = PythonObject(None)
                _resident_close(kept[1])
                mods.released()
            var opened = _resident_open_all(mods, model, e, state, b, l, int15, keep, blocks)
            lm = opened[0]
            sessions = opened[1]
        _ = _resident_group(e, lm, CAUSAL_SESSION_LM_RUN, [_addr(ids, keep), _addr(out, keep), _addr(last_logits, keep)], [l, n, 0])
        mods.released()
        _ = out  # keep the Python object alive through the native call above
        _ = ids  # keep the Python object alive through the native call above
    except error:
        mods.released()
        if persist:
            lease[].kept = PythonObject(None)
            _ = causal_lease_release(lease[].live)
        if lm is not None:
            _resident_close(lm)
            mods.released()
        raise error
    if persist:
        lease[].kept = Python.tuple(key, lm, sessions)
        if causal_lease_release(lease[].live):
            # Exposed while this call ran: nothing is kept.
            lease[].kept = PythonObject(None)
            _resident_close(lm)
            mods.released()
    else:
        _resident_close(lm)
        mods.released()
    _ = lease_held
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


def _generate_resident_requested(args: PythonObject) raises -> Bool:
    # `CausalLM.generate(..., resident=...)`: the fifth argument of the public
    # generate call (lane verify-open-sites, 2026-11-16; was the private
    # `CausalLM._generate_resident`). One tuple read and one type identity per
    # generate call, against the builtins resolved once in GENERATION_BOUNDARY.
    if len(args) < 5:
        return False
    var flag = args[4]
    if py_type(flag) is not GENERATION_BOUNDARY.get_or_create_ptr()[].bi.bool:
        raise Error("TypeError: generation: resident must be bool")
    return Bool(py=flag)


def causal_lm_generate_resident_binding(mut mods: NativeModules, model: PythonObject, value: PythonObject, maximum: PythonObject, options: PythonObject) raises -> PythonObject:
    """`generate(..., resident=True)`: the resident decode sessions of
    `causal_lm_resident_binding`, fresh caches, no last-logits output. The
    route decodes greedily to `max_new_tokens`; sampling and EOS stopping
    are refused rather than ignored, and a model without resident sessions
    (CPU, a profile or norm the sessions do not cover, layer workers) is
    refused, never routed elsewhere."""
    if py_get(model._native_entries, "parallel_call") is not None:
        raise Error("generation: resident=True runs in one process; ParallelCausalLM layers decode in their workers")
    var bi = GENERATION_BOUNDARY.get_or_create_ptr()[].bi
    if len(options) != 7 or py_type(options[0]) is not bi.bool or not Bool(py=options[0]) or options[5] is not None:
        raise Error("generation: resident=True decodes greedily to max_new_tokens; it takes greedy=True and no eos_token_id")
    var result = causal_lm_resident_binding(mods, model, value, maximum, PythonObject(None), PythonObject(None))
    if result is None:
        raise Error("generation: resident=True needs a GPU model whose numeric profile and norm have resident decode sessions")
    return result


struct GenerationHandle(Movable, Writable):
    # `model` / `state`: this read's captures (refreshed by each public read
    # and checkpoint, `_causal_public_execute`). `model_held` / `state_held`
    # (lane dw-causal, continuation 4): their native owners
    # (`CausalModelOwner`, `CausalStateOwner`), taken once per read at the
    # same point; every forward of the read reads its typed words through
    # these handles instead of finding the owners again per token. None
    # until the first read (a stream's state is allocated, not captured).
    var model: PythonObject
    var state: PythonObject
    var model_held: PythonObject
    var state_held: PythonObject
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
        self.model_held = PythonObject(None)
        self.state_held = PythonObject(None)
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
        EXPERIMENT_RESIDENT_SAMPLING_FP32) else 0
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
        for i in range(len(values)):  # driver-loop: over eos token id arguments
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


def _generation_route_witness(route: String, capacity: Int, produced: Int,
                              policy: GenerationReadControl, resident_forward: Bool = False,
                              eligible: Bool = False, skipped: Int = 0, invalidated: Bool = False):
    # Excluded-warmup diagnostic only, never a callback or per-token observer.
    # retained_kv now proves an actual skipped reload, not merely eligibility.
    comptime if EXPERIMENT_GENERATION_ROUTE_TRACE:
        print("MOJOLEARN_GENERATION_ROUTE route=" + route + " capacity=" + String(capacity)
            + " produced=" + String(produced) + " retained_kv=" + String(Int(skipped > 0))
            + " requested_route=" + policy.route
            + " engine_requested=" + policy.engine_setting
            + " sampling_policy=" + String(policy.sampling_policy)
            + " greedy=" + String(Int(policy.greedy))
            + " kv_requested=" + String(Int(policy.kv.requested))
            + " kv_policy=" + policy.kv.status(route, resident_forward, eligible, skipped, invalidated)
            + " kv_uploads_skipped=" + String(skipped))


# ---------------------------------------------------------------- the decode program
# Lane x45c-causal-generate-program-decode (2026-11-09): every GPU read of a
# float32 CausalLM (transformer, Mamba-1 or Mamba-2 layers; greedy or the FP32
# sampling policy) is ONE `Program.decode` over ONE operation session
# (transformer/causal_lm_program.mojo, door `causal_lm_program_read_v1` of the
# model's GPU block binding): the embedding, every layer's step op, the final
# norm, the head and the pick are session ops; the layers' state is state
# tensors of that session; the token loop is the executor's bounded loop.
# This shell only admits the Python model and state and flattens them into
# the door's descriptor. It replaces the resident greedy read, the resident
# sampling read and the native generation engine's token x layer loop.


def _program_state_fields(st: PythonObject) raises -> PythonObject:
    var names = Python.tuple(
        "_resident_session", "arithmetic_profile", "cached_tokens", "max_tokens", "k_cache", "v_cache", "conv_window",
        "h", "profile_id", "_nn34_usable", "absolute_position", "buffer_xbc", "buffer_dtraw", "buffered_tokens",
    )
    return _generation_fields(st, names, True)


def _program_refusal(s: GenerationHandle, words: CausalModelWords, blocks: List[PythonObject]) raises -> String:
    """Why this read cannot run as the decode program ("" when it can).
    Continuation 4: decided on the model owner's words (before: the entry
    registry, the head, the kind, the norm and its eps read by name per
    read)."""
    if not s.greedy and s.sampling_policy != 1:
        return "the Float64 sampling policy runs on the host"
    if words.program_read < 0:
        return "no GPU decode program in this install"
    if words.parallel:
        return "distributed execution"
    if words.head_packed:
        return "the fixed15_v1 head"
    if words.kind != CAUSAL_LAYER_TRANSFORMER and words.kind != CAUSAL_LAYER_MAMBA1 and words.kind != CAUSAL_LAYER_MAMBA2:
        # The refusal names the kind (read only on this refusal).
        return "layer kind " + String(s.model.kind)
    if words.kind == CAUSAL_LAYER_MAMBA1 and words.mamba1_profile != 1:
        return "the Mamba-1 profile"
    if len(blocks) == 0:
        return "a model without layers"
    if words.norm and not (words.norm_eps > 0):
        return "a final norm with eps 0"
    return ""


def _program_state_pieces(kind: Int, blk: PythonObject, fields: PythonObject, b: Int, cap: Int,
                          mut keep: List[PythonObject], mut copies: List[PythonObject],
                          owned: Bool, words: CausalLayerStateWords) raises -> List[Int]:
    """The layer's state arrays (step-op order), admitted writable. `owned`:
    from the state owner's words of this layer (typed checks, same words;
    no buffer object read), else from the buffer objects."""
    var out = List[Int]()
    if kind == CAUSAL_LAYER_TRANSFORMER:
        var window = Int(py=blk["window"])
        var size = b * Int(py=blk["n_kv_heads"]) * (cap if window == 0 else window) * Int(py=blk["head_dim"])
        var cache_shape: List[Int] = [size]
        if owned:
            out.append(causal_state_buffer_address(words.buffers[CAUSAL_STATE_K], cache_shape, "k_cache"))
            out.append(causal_state_buffer_address(words.buffers[CAUSAL_STATE_V], cache_shape, "v_cache"))
        else:
            out.append(_mutable(fields["k_cache"], cache_shape, keep, copies, "k_cache"))
            out.append(_mutable(fields["v_cache"], cache_shape, keep, copies, "v_cache"))
    elif kind == CAUSAL_LAYER_MAMBA1:
        var inner = Int(py=blk["d_inner"])
        var conv_shape: List[Int] = [b, inner, 4]
        var h_shape: List[Int] = [b, inner, 16]
        if owned:
            out.append(causal_state_buffer_address(words.buffers[CAUSAL_STATE_CONV], conv_shape, "conv_window"))
            out.append(causal_state_buffer_address(words.buffers[CAUSAL_STATE_H], h_shape, "h"))
        else:
            out.append(_mutable(fields["conv_window"], conv_shape, keep, copies, "conv_window"))
            out.append(_mutable(fields["h"], h_shape, keep, copies, "h"))
    else:
        var heads = Int(py=blk["nheads"])
        var conv = Int(py=blk["conv_dim"])
        var conv_shape: List[Int] = [b, conv, 4]
        var h_shape: List[Int] = [b, heads, 64, 128]
        var xbc_shape: List[Int] = [b, 256, conv]
        var dt_shape: List[Int] = [b, 256, heads]
        if owned:
            out.append(causal_state_buffer_address(words.buffers[CAUSAL_STATE_CONV], conv_shape, "conv_window"))
            out.append(causal_state_buffer_address(words.buffers[CAUSAL_STATE_H], h_shape, "h"))
            out.append(causal_state_buffer_address(words.buffers[CAUSAL_STATE_XBC], xbc_shape, "buffer_xbc"))
            out.append(causal_state_buffer_address(words.buffers[CAUSAL_STATE_DTRAW], dt_shape, "buffer_dtraw"))
        else:
            out.append(_mutable(fields["conv_window"], conv_shape, keep, copies, "conv_window"))
            out.append(_mutable(fields["h"], h_shape, keep, copies, "h"))
            out.append(_mutable(fields["buffer_xbc"], xbc_shape, keep, copies, "buffer_xbc"))
            out.append(_mutable(fields["buffer_dtraw"], dt_shape, keep, copies, "buffer_dtraw"))
    return out^


def _read_program(mut mods: NativeModules, mut s: GenerationHandle, capacity: Int,
                  mut admission: NativeArrayAdmission, blocks: List[PythonObject]) raises -> PythonObject:
    # Continuation 4: the model's words and the state's owner come from the
    # stream's handles (taken at this read's capture); the layer states'
    # buffers are admitted from the owner's words where they are current.
    var b = len(s.finished)
    var length = Int(py=s.current.shape[1])
    var model_held = s.model_held
    if model_held is None:
        model_held = causal_model_owner(mods, s.model)
    ref mw = model_held.downcast_value_ptr[CausalModelOwner]()[].words
    _state(mods, mw, s.model, s.state, b, length)
    var kind = mw.kind
    var positions = Int(py=s.state.positions)
    var cap = Int(py=s.state.max_tokens)
    # Forward f (from 1) ends at positions + length + f - 1: the read runs at
    # most `allowed` forwards; the next one is refused below, as the per-token
    # loop refused it at that forward (`_positions`).
    var limit = cap
    if mw.limited:
        limit = min(limit, mw.max_positions)
    var n = min(capacity, limit - positions - length + 1)
    var keep = List[PythonObject]()
    var copies = List[PythonObject]()
    var mf = _generation_fields(s.model, Python.tuple("_embed", "_norm", "_head"))
    var v = mw.vocab
    var d = mw.width
    var pn = _weight_addr(mf["_norm"], keep)
    var tokens = generation_array([max(1, n) * b], "i")
    var fin = generation_array([b], "i")
    var lens = generation_array([b], "i")
    var fin_p = i32_ptr(_addr(fin, keep))
    var lens_p = i32_ptr(_addr(lens, keep))
    for r in range(b):  # driver-loop: over batch rows (row flags, no data)
        fin_p[r] = Int32(1 if s.finished[r] else 0)
        lens_p[r] = Int32(s.lengths[r])
    var a: List[Int] = [_weight_addr(mf["_embed"], keep), pn, _weight_addr(mf["_head"], keep),
                        _addr(s.current, keep), _addr(tokens, keep), Int(fin_p), Int(lens_p)]
    for i in range(len(s.eos)):  # driver-loop: over eos: the stream's end ids: argument words
        a.append(s.eos[i])
    var f: List[Float64] = [mw.norm_eps if pn != 0 else 0.0, Float64(s.temperature), s.top_p]
    var sts = List[PythonObject]()
    var reports = List[PythonObject]()
    var nk = causal_keys()
    var state_held = s.state_held
    if state_held is None:
        state_held = causal_state_owner(mods, s.state)
    var layer_states = s.state.layers
    var unowned = Optional[CausalLayerStateWords]()
    for i in range(len(blocks)):  # driver-loop: over the model's layers (weight and state admission)
        var blk = blocks[i]
        var layer_state = layer_states[i]
        # The captured layer dictionary when the owner's words are current
        # for it (one pointer compare per buffer), else the field admission.
        var st = causal_state_layer_unpack(mods, state_held, i, layer_state, kind)
        var owned = st is not None
        if not owned:
            st = _program_state_fields(layer_state)
        if py_get(st, nk[].k__resident_session) is not None:
            raise Error("CausalLM: layer state is owned by a decode session (resident in it); sync_state() or close it first")
        var w = blk["_w"]
        var weights = List[Int]()
        for j in range(len(w)):  # driver-loop: over the block's weight slots
            weights.append(_weight_addr(w[j], keep))
        var ints = List[Int]()
        var words = List[Int]()
        var code = OPS_ATTN_STEP
        if kind == CAUSAL_LAYER_TRANSFORMER:
            var optional = blk["_wopt"]
            for j in range(len(optional)):  # driver-loop: over the block's optional weight slots
                weights.append(_weight_addr(optional[j], keep))
            ints = [b, Int(py=blk["d_model"]), Int(py=blk["n_heads"]), Int(py=blk["n_kv_heads"]),
                    Int(py=blk["head_dim"]), Int(py=blk["intermediate"]), cap, Int(py=blk["window"])]
            var tail = blk["_opts_tail"]
            for j in range(len(tail)):  # driver-loop: over the block's option words
                ints.append(Int(py=tail[j]))
            var cached = Int(py=py_item(st, nk[].k_cached_tokens))
            # The arithmetic profile contract of the per-layer route: a cache
            # written under another profile is refused, the block's profile
            # is published onto the state after the read.
            var old_profile = model_plain(py_get(st, nk[].k_arithmetic_profile))
            var profile = model_plain(py_get(blk, "_arithmetic_profile", PythonObject(None)))
            var changed = _generation_bool(py_get(blk, "_arithmetic_profile_changed", PythonObject(False)))
            if old_profile is None and cached != 0 and changed:
                raise Error("CausalLM: cached state requires arithmetic profile metadata")
            if old_profile is not None and profile is not None and not Bool(py=old_profile == profile):
                raise Error("CausalLM: state arithmetic profile differs")
            words.append(cached)
        else:
            ints = [b, d]
            if kind == CAUSAL_LAYER_MAMBA1:
                code = OPS_M1_STEP
                if Int(py=py_item(st, nk[].k_profile_id)) != 1 or not _generation_bool(py_item(st, nk[].k__nn34_usable)):
                    raise Error("CausalLM: incompatible or failed Mamba1 state profile")
            else:
                code = OPS_M2_STEP
                var limits = blk["dt_limit"]
                f.append(Float64(py=limits[0]))
                f.append(Float64(py=limits[1]))
                words.append(Int(py=py_item(st, nk[].k_buffered_tokens)))
        var pieces: List[Int]
        if owned:
            var state_owner = state_held.downcast_value_ptr[CausalStateOwner]()
            pieces = _program_state_pieces(kind, blk, st, b, cap, keep, copies, True,
                state_owner[].layers[state_owner[].slots[i]])
        else:
            if not unowned:
                unowned = Optional(CausalLayerStateWords())
            pieces = _program_state_pieces(kind, blk, st, b, cap, keep, copies, False, unowned.value())
        a.append(code)
        a.append(len(ints))
        a.extend(ints^)
        a.append(len(weights))
        a.extend(weights^)
        a.append(len(pieces))
        a.extend(pieces^)
        a.append(len(words))
        a.extend(words^)
        if kind == CAUSAL_LAYER_MAMBA2:
            var report = generation_array([b, Int(py=blk["nheads"]), 64, 128])
            a.append(1)
            a.append(_addr(report, keep))
            reports.append(report)
        else:
            a.append(0)
            reports.append(PythonObject(None))
        sts.append(st)
    var p: List[Int] = [b, v, d, len(blocks), len(s.eos), s.pad, 1 if s.greedy else 0, s.top_k,
                        Int(s.seed & UInt64(0xFFFFFFFF)), Int(s.seed >> 32), length, max(0, n), s.produced]
    var fields = _standalone_record_fields(mods, s.model)
    # The decode program is kept only while the model is not exposed: the
    # live lease word, once per read.
    var lease_held = causal_model_lease(mods, s.model)
    var persist = (causal_lease_load(lease_held.downcast_value_ptr[CausalLease]()[].live) & CAUSAL_LEASE_EXPOSED) == 0
    _ = lease_held
    # The model's decode program is a native registry handle in its block
    # image (core/causal_lm_entry_table_sessions.mojo PROGRAM_READ); the
    # model keeps its thin owner (core/causal_session_handles.mojo).
    var entries = s.model._native_entries
    var family = mw.program_read
    var owner = py_get(fields, "_decode_program") if persist else PythonObject(None)
    var held = 0 if owner is None else owner.downcast_value_ptr[CausalSessionHandles]()[].primary
    var reply_words = 5 + 2 * len(blocks)
    var reply = List[Int](length=reply_words, fill=0)
    var ret = List[Int]()
    try:
        var read_words: List[Int] = [held, Int(reply.unsafe_ptr()), reply_words]
        for i in range(len(p)):  # driver-loop: over descriptor words (argument packing)
            read_words.append(p[i])
        # A held program's owner already carries its resolved session entry
        # (resolved once at its construction); only the first read of a model,
        # with no owner yet, resolves it by the admitted key.
        var served: Int
        if owner is not None and owner.downcast_value_ptr[CausalSessionHandles]()[].family == family:
            served = _session_call_ref(owner.downcast_value_ptr[CausalSessionHandles]()[].entry,
                CAUSAL_SESSION_PROGRAM_READ, a, read_words, f)
        else:
            served = _session_call(entries, family, CAUSAL_SESSION_PROGRAM_READ, a, read_words, f)
        mods.released()
        if served != held:
            # A new program serves these weights: its owner replaces the held
            # one (whose collection releases the old program), or, when the
            # program does not persist, dies here and releases it at once.
            var fresh = PythonObject(alloc=CausalSessionHandles(causal_entry_key(entries), family, served,
                CAUSAL_SESSION_PROGRAM_RELEASE))
            if persist:
                causal_model_cache_set(mods, s.model, fields, "_decode_program", fresh)
        for i in range(reply[0]):  # driver-loop: over result words (no data)
            ret.append(reply[1 + i])
        _ = reply^
    except error:
        mods.released()
        s.usable = False
        try:
            py_setattr(s.state, "_usable", PythonObject(False))
            if persist:
                causal_model_cache_set(mods, s.model, fields, "_decode_program", PythonObject(None))
        except publication:
            _ = publication
        raise error
    admission.recheck(generation_array_context())
    var wrote = ret[0]
    var forwards = ret[1]
    var bad = ret[2]
    var alive = ret[3] != 0
    var at = 4
    for i in range(len(blocks)):  # driver-loop: over the model's layers (state publication)
        # PACK by interned key (before: a key string built per word).
        if kind == CAUSAL_LAYER_TRANSFORMER:
            py_setitem(sts[i], nk[].k_cached_tokens, PythonObject(ret[at]))
            at += 1
            var profile = model_plain(py_get(blocks[i], nk[].k__arithmetic_profile))
            if profile is not None:
                py_setitem(sts[i], nk[].k_arithmetic_profile, profile)
        elif kind == CAUSAL_LAYER_MAMBA2:
            py_setitem(sts[i], nk[].k_buffered_tokens, PythonObject(ret[at]))
            at += 1
            py_setitem(blocks[i], "h_last_", reports[i])
        else:
            py_setitem(sts[i], nk[].k_absolute_position, PythonObject(Int(py=py_item(sts[i], nk[].k_absolute_position)) + length + forwards - 1))
    _writeback(copies)
    if forwards > 0:
        py_setattr(s.state, "positions", PythonObject(positions + length + forwards - 1))
    for r in range(b):  # driver-loop: over batch rows (row flags, no data)
        s.finished[r] = fin_p[r] != 0
        s.lengths[r] = Int(lens_p[r])
    s.produced += wrote
    var out = generation_array([b, wrote], "i")
    if wrote > 0:
        var toks = List[Int32](capacity=wrote * b)
        var tp = i32_ptr(_addr(tokens, keep))
        for k in range(wrote * b):  # driver-loop: over the committed token record (id words, no data)
            toks.append(tp[k])
        causal_lm_tokens_row_major(toks, i32_ptr(_addr(out, keep)), b, wrote)
        var current = generation_array([b, 1], "i")
        var cur = i32_ptr(_addr(current, keep))
        for r in range(b):  # driver-loop: over batch rows (the last committed ids, no data)
            cur[r] = tp[(wrote - 1) * b + r]
        s.current = current
    var failure = String("")
    if bad >= 0:
        failure = "generation: non-finite logit at token " + String(bad)
    elif alive and wrote < capacity and n < capacity:
        if mw.limited and positions + length + forwards > mw.max_positions:
            failure = "CausalLM: max_tokens or sequence exceeds max_positions"
        else:
            failure = "CausalLM: sequence exceeds allocated state capacity"
    if failure != "":
        s.usable = False
        raise Error(failure)
    _ = keep^
    _ = state_held
    _ = model_held
    return out


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
            for name in [String("k_cache"), String("v_cache")]:  # driver-loop: over two KV storage spans
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
        var lease_held = causal_model_lease(mods, model)
        var lease_word = causal_lease_load(lease_held.downcast_value_ptr[CausalLease]()[].live)
        _ = lease_held
        if lease_word != 0:
            # The legacy route reopens private sessions each token under this
            # policy. A KV write into live weight storage would therefore feed
            # changed weights into its next forward. Read-wide weights cannot
            # preserve that coupling, even when all KV spans are disjoint.
            var weights = List[PythonObject]()
            weights.append(model._embed)
            weights.append(model._norm)
            var packed = model._head_int15 is not None
            if packed:
                for k in range(3):  # driver-loop: over consumed packed head planes
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
                            for k in range(3):  # driver-loop: over int15 plane triple
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


def _resident_read_cleanup(lm: PythonObject) raises:
    # Failure-only best effort: the caller preserves its original operation
    # error. A handle a refused close keeps is released when the owner is
    # collected (core/causal_session_handles.mojo).
    if lm is not None:
        try:
            lm.downcast_value_ptr[CausalSessionHandles]()[].close()
        except cleanup:
            _ = cleanup


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
        # Lane x45c: the decode program is the GPU route of every read
        # (`_read_program`); the resident greedy read, the resident sampling
        # read and the native generation engine read are deleted. The host
        # loop below stays for what the program does not serve (a CPU install,
        # distributed execution, the fixed15_v1 head, the Float64 sampler).
        if capacity > 0 and not s.done() and read_control.route != "legacy":
            var blocks = _block_list(mods, s.model)
            var refusal_held = s.model_held
            if refusal_held is None:
                refusal_held = causal_model_owner(mods, s.model)
            var refusal = _program_refusal(s, refusal_held.downcast_value_ptr[CausalModelOwner]()[].words, blocks)
            _ = refusal_held
            if refusal == "":
                admission = NativeArrayAdmission(generation_array_context())
                var result = _read_program(mods, s, capacity, admission.value(), blocks)
                _generation_route_witness("program_decode", capacity, s.produced, read_control)
                return result
            if read_control.route != "auto":
                raise Error("generation: the decode program route does not serve " + refusal)
        # Only the legacy fallback needs host token staging. Resident/native
        # engines return their final native result without this unused buffer.
        var tokens = List[Int32](capacity=b * capacity)
        var wrote = 0
        var call_layers: Optional[GenerationCallLayers] = None
        # Optional, not a PythonObject read and reassigned in a loop (modular/modular#7546 pattern; runtime-fix-save-forest sweep).
        var token_a = Optional[PythonObject]()
        var token_b = Optional[PythonObject]()
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
        # Lane dw-causal: the model's native owner, once per read; every
        # token's forward reads its typed words through this handle (before:
        # the entry registry and its `parallel_call` word read per token).
        # Continuation 4: the stream holds both owners for the read (taken at
        # the read's capture); a handle without them (an internal caller)
        # finds the model owner here, once.
        var model_held = s.model_held
        if model_held is None:
            model_held = causal_model_owner(mods, s.model)
        ref model_owner = model_held.downcast_value_ptr[CausalModelOwner]()[]
        while wrote < capacity and not s.done():  # driver-loop: over decode steps, one model forward each
            if not admission:
                admission = NativeArrayAdmission(generation_array_context())
            var l = Int(py=s.current.shape[1])
            var logits = _run_prepared(mods, model_owner.words, model_held, s.model, s.current, s.state, call_layers, read_plan, s.produced > 0, True, reuse_sampling_uploads and not legacy_resident, -1, List[PythonObject](), s.state_held)
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
                    if not token_a:
                        token_a = Optional[PythonObject](generation_array([b, 1], "i"))
                    next = token_a.value()
                else:
                    if not token_b:
                        token_b = Optional[PythonObject](generation_array([b, 1], "i"))
                    next = token_b.value()
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
    return model_record_fields(mods, stream, _causal_context()["generation_classes"][3], Python.tuple("_native", "_runtime", "_native_public_busy"))


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
    # TOMBSTONE: the resident greedy shortcut of generate (REFACTORED: generate reads the decode program like every stream) deleted 2026-11-09 by lane/x45c-causal-generate-program-decode; code recoverable at 5b5490960d.
    # Restore: git apply experiments/removed/CAUSAL_GENERATE_RESIDENT_READS.patch; record in docs/TOMBSTONES.md.
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


def _hash_array(mut hash: Sha256, value: PythonObject) raises:
    # The same bytes in the same order as `_array_bytes` (tobytes("C")),
    # hashed straight from the pinned buffer: no bytes object and no
    # array('B') copy per tensor. A strided buffer is packed once.
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    var source = generation_array_snapshot(value)._mv if py_type(value) is g[].array_cls else generation_buffer_view(value)
    admit_native_buffer_exporter(source)
    var pin = AdmittedBorrowedBuffer(source)
    try:
        var layout = pin.layout.value().copy()
        var n = layout.count * layout.itemsize
        if n > 0 and layout.contiguous():
            hash.update(layout.address, n)
        elif n > 0:
            var packed = List[UInt8](length=n, fill=0)
            layout.copy_to(Int(packed.unsafe_ptr()))
            hash.update(Int(packed.unsafe_ptr()), n)
            _ = packed^
    except error:
        pin.release()
        raise error
    pin.release()
    _ = source


def _hash() -> Sha256:
    return Sha256()


def _hash_scalar(mut hash: Sha256, value: PythonObject) raises:
    sha256_update_text(hash, value)
    hash.update_text(String("\n"))


def _fingerprint(mut mods: NativeModules, model: PythonObject) raises -> PythonObject:
    """The checkpoint fingerprint of a captured model: the hash of its settings
    and every weight, taken again at every snapshot and every resume (once per
    call, off the token path). It is never kept: the weights are the caller's
    Arrays and can be edited in place without changing the model's identity
    key, so a kept word could accept a checkpoint the edited weights no longer
    match (fx-audit 21)."""
    return _fingerprint_hash(mods, model)


def _fingerprint_hash(mut mods: NativeModules, model: PythonObject) raises -> PythonObject:
    # The checkpoint fingerprint and `CausalLM.digest_` are ONE walk
    # (`_causal_digest_members`) and one hash (core/fitted_digest.mojo:
    # chunks across the host cores, the interpreter lock released). Before
    # 2026-11-25 this was a second walk: one serial SHA-256 chain over every
    # weight with the lock held. The fingerprint's bits changed with it.
    return PythonObject(_causal_digest_hex(mods, model))


def _digest_scalar_member(mut members: List[NativeArchiveMember], name: String, value: PythonObject) raises:
    """A metadata scalar of the model (a kind word, a size, a shape tuple) as
    the UTF-8 text of `str(value)`: the text the fingerprint always hashed."""
    var text = String(py_call(native_loaded_module("builtins").str, value))
    var raw = archive_text_bytes(text)
    var count = len(raw)
    members.append(archive_owned(name, String("U"), List[Int](length=1, fill=count), 1, raw^))


def _digest_array_member(mut members: List[NativeArchiveMember], mut pins: List[AdmittedBorrowedBuffer],
                         mut sources: List[PythonObject], name: String, value: PythonObject) raises:
    """A weight array as (name, element format, shape, bytes in C order),
    read in place from its pinned buffer: no bytes object and no copy of a
    contiguous tensor (a strided one is packed once, in the hash's parallel
    prepare). The pin and its exporter are kept by the caller to the end of
    the hash."""
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    var source = generation_array_snapshot(value)._mv if py_type(value) is g[].array_cls else generation_buffer_view(value)
    admit_native_buffer_exporter(source)
    var pin = AdmittedBorrowedBuffer(source)
    var layout = pin.layout.value().copy()
    var element = String(pin.format)
    members.append(NativeArchiveMember(name, element, layout.shape.copy(), layout.strides.copy(), layout.address,
                                       layout.count * layout.itemsize, layout.itemsize, List[UInt8]()))
    pins.append(pin^)
    sources.append(source)


def _causal_digest_members(mut mods: NativeModules, model: PythonObject, mut members: List[NativeArchiveMember],
                           mut pins: List[AdmittedBorrowedBuffer], mut sources: List[PythonObject]) raises:
    """The fitted state of a captured causal model, described once: its kind
    and profile words, the embedding, the final norm, the head, then every
    block's options and weights (and the int15 planes of a packed model), in
    model order. Nothing of the device, the vendor or the session is read."""
    _digest_scalar_member(members, String("kind"), model.kind)
    _digest_scalar_member(members, String("numeric_profile"), model.numeric_profile)
    _digest_scalar_member(members, String("norm_eps"), model.norm_eps)
    _digest_scalar_member(members, String("max_positions"), model.max_positions)
    _digest_array_member(members, pins, sources, String("embed"), model._embed)
    if model._norm is not None:
        _digest_array_member(members, pins, sources, String("norm"), model._norm)
    _digest_array_member(members, pins, sources, String("head"), model._head)
    var kind = String(model.kind)
    for i in range(len(model._blocks)):  # driver-loop: over model layers
        var blk = _block(mods, model, i)
        _causal_array_guard()
        var at = String("block") + String(i) + "."
        if kind == "transformer":
            _digest_scalar_member(members, at + "opts_tail", blk["_opts_tail"])
            _digest_scalar_member(members, at + "n_heads", blk["n_heads"])
            _digest_scalar_member(members, at + "n_kv_heads", blk["n_kv_heads"])
            _digest_scalar_member(members, at + "window", blk["window"])
            for j in range(len(blk["_wopt"])):  # driver-loop: over optional block weight slots
                if blk["_wopt"][j] is not None:
                    _digest_array_member(members, pins, sources, at + "wopt" + String(j), blk["_wopt"][j])
        elif kind == "mamba2":
            _digest_scalar_member(members, at + "dt_limit", blk["dt_limit"])
        else:
            _digest_scalar_member(members, at + "mamba1_profile", model._native_entries["mamba1_profile"])
        for j in range(len(blk["_w"])):  # driver-loop: over block weight slots
            if blk["_w"][j] is not None:
                _digest_array_member(members, pins, sources, at + "w" + String(j), blk["_w"][j])
        if model._head_int15 is not None:
            for j in range(len(blk["_int15"])):  # driver-loop: over packed block weight slots
                if blk["_int15"][j] is not None:
                    for k in range(3):  # small-loop(int15 plane triple)
                        _digest_array_member(members, pins, sources, at + "int15_" + String(j) + "_" + String(k),
                                             blk["_int15"][j][k])
    if model._head_int15 is not None:
        for k in range(3):  # small-loop(int15 head plane triple)
            _digest_array_member(members, pins, sources, String("head_int15_") + String(k), model._head_int15[k])


def _causal_digest_hex(mut mods: NativeModules, model: PythonObject) raises -> String:
    """The hash of a captured causal model as it is now (64 lowercase hex
    digits): UNPACK describes every field once, RUN hashes the typed members
    across the host cores with the interpreter lock released. Nothing is
    kept. The class word is the public class's on every device and route."""
    _causal_array_guard()
    var members = List[NativeArchiveMember]()
    var pins = List[AdmittedBorrowedBuffer]()
    var sources = List[PythonObject]()
    var text = String("")
    try:
        _causal_digest_members(mods, model, members, pins, sources)
        with GILReleased(Python()):
            text = fitted_digest_hex(String("CausalLM"), members)
    finally:
        for i in range(len(pins)):  # driver-loop: over the model's pinned weight buffers (release)
            pins[i].release()
    _ = sources
    _ = members
    return text


def causal_lm_digest_binding(mut mods: NativeModules, model: PythonObject) raises -> PythonObject:
    """`CausalLM.digest_`: hashed at every read, never the owner's kept
    checkpoint fingerprint (an in-place write into a weight changes no
    generation word). A `fast` model has no digest.

    A ParallelCausalLM (one worker process per layer group; its blocks hold
    a worker index, not the weights) is refused in words (fx-audit 34): this
    walk reads every block's weights from the model's own fields, and
    hashing the workers' copies would need a gather the class does not do.
    One instance-dictionary probe, at a digest read only."""
    if py_contains(py_fields(model), "layer_devices"):
        raise Error("AttributeError: ParallelCausalLM has no digest_: its layers live in worker processes; "
                    + "load the same checkpoint with CausalLM.load and read its digest_")
    var mode = model._native_numeric_mode
    if mode is not None and String(mode) == "fast":
        raise Error("AttributeError: digest_ exists in identical mode only: fast mode promises no identical "
                    + "bits, so a fast model has no digest")
    return PythonObject(_causal_digest_hex(mods, model))


def _state_arrays(kind: String, profile: Int = 1) -> List[String]:
    if kind == "transformer":
        return [String("k_cache"), String("v_cache")]
    if kind == "mamba2":
        return [String("conv_window"), String("h"), String("buffer_xbc"), String("buffer_dtraw")]
    if profile == 2:
        return [String("conv_window"), String("h"), String("affine_boundary"), String("affine_a"), String("affine_b")]
    return [String("conv_window"), String("h")]


def _state_array_slot(name: String) -> Int:
    # The owner's buffer slot (`CAUSAL_STATE_*`) of a checkpoint state array.
    if name == "k_cache":
        return CAUSAL_STATE_K
    if name == "v_cache":
        return CAUSAL_STATE_V
    if name == "conv_window":
        return CAUSAL_STATE_CONV
    if name == "h":
        return CAUSAL_STATE_H
    if name == "affine_boundary":
        return CAUSAL_STATE_AFFINE_BOUNDARY
    if name == "affine_a":
        return CAUSAL_STATE_AFFINE_A
    if name == "affine_b":
        return CAUSAL_STATE_AFFINE_B
    if name == "buffer_xbc":
        return CAUSAL_STATE_XBC
    return CAUSAL_STATE_DTRAW


def _state_owner_copy(buffer: CausalBufferWords) raises -> PythonObject:
    """A new Array with the bytes of a state buffer the owner admitted as
    writable C-contiguous float32 (the copy `generation_copy` makes of it:
    one unfilled allocation, one copy). None for any other buffer: the
    caller then copies it from the object."""
    if not buffer.present or not buffer.contiguous or not buffer.writable:
        return PythonObject(None)
    var out = generation_array_unfilled(buffer.shape, "f")
    generation_bytes_copy(Int(py=out._addr), buffer.address, buffer.nbytes)
    return out


def _snapshot_hash(checkpoint: PythonObject) raises -> PythonObject:
    var h = _hash()
    var g = GENERATION_BOUNDARY.get_or_create_ptr()
    for name in [String("schema"), String("model"), String("options"), String("maximum"),  # small-loop(checkpoint scalar fields)
                 String("produced"), String("positions"), String("lengths"), String("finished"), String("text"), String("tokenizer")]:
        _hash_scalar(h, checkpoint[name])
    if String(checkpoint["schema"]) == "mojolearn.generation.v2":
        _hash_scalar(h, checkpoint["sampling_policy"])
    sha256_update(h, checkpoint["pending_bytes"])
    _hash_array(h, checkpoint["prompt"])
    _hash_array(h, checkpoint["current"])
    for i in range(len(checkpoint["layers"])):  # driver-loop: over checkpoint layers
        var layer = checkpoint["layers"][i]
        var names = py_call(g[].bi.sorted, layer.keys())
        for j in range(len(names)):  # driver-loop: over layer state field names
            _hash_scalar(h, names[j])
            var value = layer[names[j]]
            if py_type(value) is g[].array_cls:
                _hash_scalar(h, value.shape)
                _hash_scalar(h, value.dtype)
                _hash_array(h, value)
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
    # Continuation 4 (design 1.3): the layer states are copied from the state
    # owner's words where they are current (each buffer: one allocation and
    # one copy from the owner's address, the same bytes as `generation_copy`;
    # counters by interned key); any other layer (a parallel model's remote
    # state, a buffer the owner does not hold as writable C-contiguous float32)
    # by the object route. Same keys, same bytes, same checksum.
    var layers = g[].bi.list()
    var model_held = s[].model_held
    if model_held is None:
        model_held = causal_model_owner(mods, s[].model)
    ref mw = model_held.downcast_value_ptr[CausalModelOwner]()[].words
    var kind = String(s[].model.kind)
    var profile = mw.mamba1_profile if mw.kind == CAUSAL_LAYER_MAMBA1 else 1
    var arrays = _state_arrays(kind, profile)
    var nk = causal_keys()
    var parallel = py_get(s[].model._native_entries, "parallel_call") if mw.parallel else PythonObject(None)
    var state_held = s[].state_held
    if state_held is None and parallel is None:
        state_held = causal_state_owner(mods, s[].state)
    var layer_states = s[].state.layers
    for i in range(len(layer_states)):  # driver-loop: over model layer states
        var st = layer_states[i]
        var sf = PythonObject(None)
        if parallel is not None:
            st = py_call(parallel, "state_snapshot", Python.list(causal_state_origin(mods, st), st))
            mods.released()
            st = causal_layer_capture(mods, st, s[].model.kind, _causal_context_for(mods, s[].model))
        else:
            sf = causal_state_layer_unpack(mods, state_held, i, st, mw.kind)
        var entry = g[].bi.dict()
        if sf is not None:
            var state_owner = state_held.downcast_value_ptr[CausalStateOwner]()
            var slot = state_owner[].slots[i]
            for j in range(len(arrays)):  # driver-loop: over layer state arrays
                var index = _state_array_slot(arrays[j])
                var copied = _state_owner_copy(state_owner[].layers[slot].buffers[index])
                if copied is None:
                    copied = generation_copy(py_call(g[].bi.getattr, st, arrays[j]))
                py_setitem(entry, arrays[j], copied)
            if mw.kind == CAUSAL_LAYER_TRANSFORMER:
                py_setitem(entry, "cached_tokens", py_item(sf, nk[].k_cached_tokens))
            elif mw.kind == CAUSAL_LAYER_MAMBA1:
                py_setitem(entry, "absolute_position", py_item(sf, nk[].k_absolute_position))
                py_setitem(entry, "profile_id", py_item(sf, nk[].k_profile_id))
            else:
                py_setitem(entry, "buffered_tokens", py_item(sf, nk[].k_buffered_tokens))
        else:
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
    _ = state_held
    _ = model_held
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
        var parallel = py_get(model._native_entries, "parallel_call")
        for i in range(len(s[].state.layers)):  # driver-loop: over model layer states
            var dest = s[].state.layers[i]
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
                if parallel is None:
                    # Continuation 4 (design 1.3): the bytes go into the fresh
                    # Arrays this resume allocated (one copy, the bytes of
                    # `generation_copy`); before, a second Array was allocated
                    # and replaced the fresh one. Nothing is shared with the
                    # checkpoint afterwards.
                    var captured = generation_array_snapshot(supplied)
                    generation_bytes_copy(Int(py=expected._addr), Int(py=captured._addr), Int(py=captured.nbytes))
                    _ = captured
                    _ = expected
                else:
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


def _causal_public_execute[operation: Int](py_self: PyObjectPtr, args_ptr: PyObjectPtr, stream_fields: PythonObject) -> PyObjectPtr:
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
        comptime if operation == 1 or operation == 2 or operation == 3 or operation == 4 or operation == 7 or operation == 8 or operation == 9 or operation == 10 or operation == 12 or operation == 13 or operation == 18 or operation == 19:
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
            stream_native = stream_fields["_native"]
            comptime if operation == 5 or operation == 11:
                var stream = stream_native.downcast_value_ptr[GenerationHandle]()
                # Every public read/checkpoint captures current caller metadata
                # once; prior calls already published their state progress.
                model = causal_model_capture(mods, causal_model_origin(mods, stream[].model), _causal_context())
                state = causal_state_capture(mods, causal_state_origin(mods, stream[].state), model, _causal_context())
                stream[].model = model
                stream[].state = state
                # The owners of these captures, held by the stream for this
                # read: one interned read each, once per public read.
                stream[].model_held = causal_model_owner(mods, model)
                stream[].state_held = causal_state_owner(mods, state)
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
            if _generate_resident_requested(args):
                result = causal_lm_generate_resident_binding(mods, model, args[1], args[2], options)
            else:
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
        elif operation == 18:
            # state_dict(): copies, no exposure mark (lane fx-leftovers item 1).
            result = causal_lm_parameters_binding(mods, model, True)
        elif operation == 19:
            # digest_: the fitted state hashed now (one walk with the checkpoint fingerprint).
            result = causal_lm_digest_binding(mods, model)
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
        # PACK: the lease's read-only mirror, once per public operation.
        causal_lease_mirror(mods, model)
        return result.steal_data()
    except error:
        var message = String(error)
        try:
            causal_state_publish(mods, state)
            causal_model_publish(mods, model)
            causal_lease_mirror(mods, model)
        except publication_error:
            message += "; native state publication also failed: " + String(publication_error)
            return raise_python_exception(Error(message), ExceptionType("PyExc_RuntimeError"))
        if message.startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        if message.startswith("TypeError:"):
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        comptime if operation == 19:
            # A refused `digest_` is a missing attribute (`hasattr` is False).
            if message.startswith("AttributeError:"):
                return raise_python_exception(error_without_class(error, "AttributeError"), ExceptionType("PyExc_AttributeError"))
        if _causal_owner_unusable(mods, args_ptr):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)


def causal_public_c[operation: Int](py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    comptime if operation != 5 and operation != 6 and operation != 11:
        return _causal_public_execute[operation](py_self, args_ptr, PythonObject(None))
    else:
        # Claim the original stream while holding the GIL, BEFORE borrowing its
        # GenerationHandle or capturing model/state metadata. Capture itself can
        # release the GIL; a late _read busy check cannot protect those owners.
        # Exact admitted dictionary/string/boolean operations execute no Python
        # callbacks. The claim spans read, info, and checkpoint publication and
        # text conversion; independent streams retain independent claims.
        var mods = NativeModules()
        var fields = PythonObject(None)
        var busy_name = PythonObject(None)
        var claimed = False
        var result = PythonObject(None)
        try:
            var args = PythonObject(from_borrowed=args_ptr)
            busy_name = PythonObject("_native_public_busy")
            fields = _causal_stream_fields(mods, args[0])
            if py_get(fields, busy_name, PythonObject(False)) is not PythonObject(False):
                raise Error("generation: stream is busy")
            py_setitem(fields, busy_name, PythonObject(True))
            claimed = True
            var raw = _causal_public_execute[operation](py_self, args_ptr, fields)
            if raw:
                result = PythonObject(from_owned=raw)
            # On an inner ABI error, this exact-dict update preserves its active
            # CPython exception. On success, retain the owned result until claim
            # cleanup succeeds so a cleanup error cannot leak a stolen result.
            py_setitem(fields, busy_name, PythonObject(False))
            claimed = False
            if not raw:
                return raw
            return result.steal_data()
        except error:
            var message = String(error)
            if claimed:
                try:
                    py_setitem(fields, busy_name, PythonObject(False))
                except cleanup:
                    return raise_python_exception(Error(message + "; stream claim cleanup failed: " + String(cleanup)), ExceptionType("PyExc_RuntimeError"))
            if message.startswith("TypeError:"):
                return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
            if message.startswith("RuntimeError:"):
                return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
            return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
