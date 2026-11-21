# SPDX-License-Identifier: Apache-2.0
"""Native standalone block construction, state allocation and sequence execution."""
from core.mode_answer import MODE_FAMILY_NEURAL, answer_mode_word, word_runs_identical
from bindings.mode_answer_door import answer_mode_object
from bindings.native_class_admission_door import class_record, class_admit
from std.python import Python, PythonObject
from core.neural_block_public_impl import neural_copy_row_prefixes
from std.python._cpython import PyObjectPtr, GILReleased
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_contains, py_get, py_setattr, py_setitem, py_type
from bindings.neural_block_admission_door import (
    neural_block_fields, admit_block_f32, BLOCK_READ, BLOCK_OPERATION, neural_record_fields, neural_string_keys,
    neural_plain_value, neural_array_snapshot, neural_array_slots, neural_admit_weights,
    neural_native_scalar, neural_model_weights, neural_block_snapshot, neural_block_weight_names, neural_metadata_snapshot,
)
from bindings.mode_metadata_native_door import _fields, _class_fields
from bindings.native_loaded_module_door import native_loaded_module, NativeModules
from bindings.native_identity_key_door import NativeIdentityKey
from bindings.native_prepared_owner_unpack import prepared_get, prepared_put, prepared_drop, prepared_key_add, prepared_key_fields, prepared_key_same
from providers.python.borrowed_buffer_pyapi_unpack import admit_native_buffer_exporter
from bindings.boundary_read_door import buffer_view, view_int_at, read_exact_int_items
from core.boundary_layout import BOUNDARY_KIND_FLOAT, BOUNDARY_KIND_BOOL
from std.memory import unsafe_memcpy
from std.sys.compile import is_defined
from bindings.generation_buffers_door import generation_initialize_binding, generation_array, py_ints
from bindings.causal_lm_construct_door import _model_block, _model_profile, _model_bindings, _model_f32, _model_shape, model_recurrent_weights_check
from bindings.causal_lm_public_door import causal_lm_admit_binding, causal_lm_admit, CausalAdmission, causal_layer_allocate, causal_layer_allocate_fields, causal_layer_run_fields, causal_layer_stateless_plan, _addr, _weight_addr, _native_function, _native_has_entry, _native_entry_fields, _standalone_record_fields, causal_prepared_layer
from core.hostptr import f32_ptr
from core.host_helpers_impl import all_finite_f32
from core.native_descriptor_abi import TRANSFORMER_BLOCK_ABI_NAME, MAMBA_BLOCK_ABI_NAME, MAMBA_STATE_ABI_NAME
from providers.python.operation_lock_pyapi_door import OperationLockLease, operation_lock_create
from bindings.py_model_records import bump_model_generation
from providers.python.neural_descriptor_pyapi_door import (
    neural_descriptor_reuse, NEURAL_NATIVE_DISPATCH,
)
from providers.python.transformer_block_session_pyapi_door import (
    transformer_block_session_open, transformer_block_session_handle, transformer_block_session_serves,
    transformer_block_session_close,
)
from providers.python.mamba3_prefill_session_pyapi_door import (
    mamba3_prefill_caps, mamba3_prefill_session_ensure, mamba3_prefill_session_serves, mamba3_prefill_session_run,
)
from core.mamba_state_entry_table import Mamba3StateRoute
from bindings.causal_lm_public_door import _causal_array_guard
from bindings.causal_lm_public_door import causal_mamba_entry, causal_mamba_stateless_run
from autograd.block_step_ops import IDN_MAMBA12_FORWARD_DOOR_OLD
from core.entry_table import EntryRef, entry_ref_unresolved, entry_ref_call
from core.causal_layer_run import CausalLayerRoute
from providers.python.generation_metadata_pyapi_door import causal_layer_route
from core.device_route import DEVICE_CPU, DEVICE_GPU, device_name, device_route_plan, device_cpu_absent
from providers.python.device_route_boundary_door import (
    DeviceRouteOwner, device_unpack, device_route_keep, device_route_owner, device_route_value,
)
from core.neural_entry_table import (
    NeuralEntryKey, neural_entry_vendor, neural_entry_call, neural_ref,
    NEURAL_TABLE_BLOCK_FORWARD, NEURAL_TABLE_BLOCK_TRAIN, NEURAL_TABLE_MAMBA_TRAIN,
    NEURAL_BLOCK_RESERVE, NEURAL_BLOCK_RELEASE, NEURAL_BLOCK_SESSION_FORWARD_FRESH,
    NEURAL_BLOCK_SESSION_BACKWARD, NEURAL_BLOCK_FORWARD_FRESH, NEURAL_BLOCK_BACKWARD,
    NEURAL_MAMBA1_BACKWARD, NEURAL_MAMBA2_BACKWARD,
    NEURAL_MAMBA3_BACKWARD, NEURAL_MAMBA3_SESSION_BACKWARD, NEURAL_MAMBA3_OWNED_BACKWARD,
    NEURAL_MAMBA3_OWNED_INSTALL, MAMBA3_CAP_SESSION_PREFILL, MAMBA3_CAP_OWNED_WEIGHTS,
)
# Lane ne3-c-blocks-embedding (row N115): the one record table (lane
# ne1-e-records contract, bindings/py_model_records.mojo).
from bindings.py_model_records import (
    RECORDS_PREPARED, ContentRecord, content_value, records_stamped_get, records_stamped_put,
)



# Lane envgate-neural-lm (2026-11-15): the block doors' setup switches are
# build constants with today's defaults; each was a getenv per public call.
# B arms RUN OWED (bench/results/paper_compare/owed-pre0867.json,
# docs/apple-fast/EXPERIMENTS.md): -D MOJOLEARN_TRANSFORMER_LEGACY_SETUP
# (was =1), -D MOJOLEARN_TRANSFORMER_SESSION_FRESH_OFF (was =0),
# -D MOJOLEARN_MAMBA3_LEGACY_SETUP (was =1) lost A/B ab3 (NVIDIA L40S, wheel
# d652a4b0a1, 2026-11-18, job nvc-ab-20261118T182558Z-cb6e02, digests equal):
# arm/base 1.007 geomean over 5 mamba3/samba cells (0.97-1.07), noise.
# TOMBSTONE: MOJOLEARN_MAMBA3_LEGACY_SETUP (LOSER: Mamba3 per-call setup without prefill sessions, arm/base geomean 1.007 over 5 cells, noise, NV L40S ab3 2026-11-18) deleted 2026-11-18 by lane/fx-verdicts2; code recoverable at 85c36e324f.
# Restore: git apply experiments/removed/MOJOLEARN_MAMBA3_LEGACY_SETUP.patch; record in docs/TOMBSTONES.md.
comptime TRANSFORMER_LEGACY_SETUP = is_defined["MOJOLEARN_TRANSFORMER_LEGACY_SETUP"]()
comptime TRANSFORMER_SESSION_FRESH = not is_defined["MOJOLEARN_TRANSFORMER_SESSION_FRESH_OFF"]()


struct NeuralBlockOwner(Movable):
    # The typed owner of a standalone block's plan build (lane
    # dw-neural-models, 2026-11-21): what the plan and its prepared layer
    # need from the setup snapshot and from admission. It replaces the
    # SimpleNamespace owner that was built, written attribute by attribute
    # and read back at every owner-record miss.
    var fields: PythonObject           # the setup snapshot of this operation
    var context: PythonObject
    var kind: String
    var d_model: Int
    var numeric_profile: PythonObject
    var packed: Bool                   # the block holds fixed15 planes
    var entries: PythonObject          # the admitted entry dictionary
    var vendor: String                 # what the admitted image reported
    var mode: Int

    def __init__(out self, fields: PythonObject, context: PythonObject, kind: String, d_model: Int,
                 numeric_profile: PythonObject, packed: Bool, entries: PythonObject, vendor: String, mode: Int):
        self.fields = fields
        self.context = context
        self.kind = kind.copy()
        self.d_model = d_model
        self.numeric_profile = numeric_profile
        self.packed = packed
        self.entries = entries
        self.vendor = vendor.copy()
        self.mode = mode


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
    # The forward entry (core/neural_entry_table.mojo) of the block image
    # this plan's forward calls, keyed by the tier and vendor
    # causal_lm_admit_binding admitted for it and resolved once, here, at
    # plan construction (lane entry-table-zero-alloc): every forward is one
    # indirect call through it.
    var forward_entry: EntryRef
    # Whether a forward with no state runs stateless (lane
    # fix-mamba12-forward-exec, `causal_layer_stateless_plan`): kind, entry
    # vendor and Mamba-1 profile are fixed for a plan, so it is decided here
    # once (hot-path audit 2026-11-13: it read two entry words per call).
    var stateless_forward: Bool
    # Lane fix-mamba-forward-doors (IDN_MAMBA12_FORWARD_DOOR_OLD,
    # bindings/causal_lm_public_door.mojo): the Mamba state entry a stateless
    # forward calls, resolved at the plan's first stateless forward and held
    # (word 0 until then).
    var mamba_entry: EntryRef
    # What the admitted block image itself reported (causal_lm_admit_binding):
    # its vendor word and numeric-mode code. `vendor_used` and
    # `numeric_mode_used` of a block answer from here, so a `device="cpu"`
    # block in a GPU installation says `cpu` (lane dw-neural-models).
    var vendor: String
    var mode: Int
    # Lane dw-neural-models (2026-11-21): the words of the entry dictionary a
    # forward read by name on every call, fixed for a plan and read once
    # here: whether the packed (fixed15) planes run (`forward15`), whether the
    # route has a block session (`block_session`: a GPU Transformer route)
    # and which fresh-forward entries its image has.
    var planes: Bool
    var block_session: Bool
    var session_fresh: Bool
    var forward_fresh: Bool
    # The layer run's typed route (core/causal_layer_run.mojo): the CausalLM
    # table entry and the host/device and Mamba-1 profile words. Resolved at
    # the plan's first layer run (`_neural_layer_run`) and held; unbound
    # until then, as `mamba_entry` is, so a plan whose kind never runs a
    # CausalLM layer (Mamba-3) resolves nothing.
    var layer_route: CausalLayerRoute

    def __init__(out self, owner: NeuralBlockOwner, layer: PythonObject) raises:
        self.context = owner.context
        self.entries = owner.entries
        self.kind = owner.kind.copy()
        self.vendor = owner.vendor.copy()
        self.mode = owner.mode
        # Only a Transformer block's forward calls the forward table.
        self.forward_entry = entry_ref_unresolved()
        if self.kind == "transformer":
            self.forward_entry = neural_ref[NEURAL_TABLE_BLOCK_FORWARD](
                NeuralEntryKey(owner.mode, neural_entry_vendor(owner.vendor)))
        self.d_model = owner.d_model
        self.numeric_profile = owner.numeric_profile
        self.packed = owner.packed
        self.planes = self.kind == "transformer" and self.packed and py_contains(self.entries, "forward15")
        self.block_session = py_contains(self.entries, "block_session")
        self.session_fresh = py_contains(self.entries, "transformer_session_forward_fresh")
        self.forward_fresh = py_contains(self.entries, "transformer_forward_fresh")
        self.layer_route = CausalLayerRoute()
        self.layer = layer
        self.stateless_forward = causal_layer_stateless_plan(self.kind, self.entries)
        self.mamba_entry = entry_ref_unresolved()
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


def _neural_table_call(entry: EntryRef, operation: Int, addresses: List[Int],
    ints: List[Int], floats: List[Float64]) raises -> Int:
    # One descriptor call into a sibling image through the neural entry table
    # (core/neural_entry_table.mojo): plain words, no CPython call. The GIL is
    # released around the numerical work as the former Python entries and
    # provider calls released it; a caller holding a NativeModules cache
    # resets it afterwards (mods.released()), as it did after those calls.
    var result = 0
    with GILReleased(Python()):
        result = neural_entry_call(entry, operation, addresses, ints, floats)
    return result


def _neural_table_reserved(entry: EntryRef, handle: Int, reserve: Int, release: Int,
    operation: Int, addresses: List[Int], ints: List[Int], floats: List[Float64]) raises -> Int:
    # A session operation: reserve and release run with the GIL held (the
    # image's busy/closed admission, as the provider ran them), the operation
    # with it released. The session handle leads the operation's words. The
    # caller pins the session owner until this returns.
    # The reserve/release requests are fixed one-word envelopes on this frame
    # (no heap); the operation's words are sized once (handle + scalars).
    var none: Array[Int, 0] = []
    var slot: Array[Int, 1] = [handle]
    var no_floats: Array[Float64, 0] = []
    var words = List[Int](capacity=1 + len(ints))
    words.append(handle)
    for i in range(len(ints)):  # driver-loop: over argument words (fixed per operation, not data); small-loop(session operation scalar words)
        words.append(ints[i])
    _ = entry_ref_call(entry, reserve, none, slot, no_floats)
    var result = 0
    try:
        with GILReleased(Python()):
            result = neural_entry_call(entry, operation, addresses, words, floats)
    except problem:
        _ = entry_ref_call(entry, release, none, slot, no_floats)
        raise problem
    _ = entry_ref_call(entry, release, none, slot, no_floats)
    _ = words^
    return result


# The Mamba-3 prefill session's operations run through its owner
# (providers/python/mamba3_prefill_session_pyapi.mojo, lane
# mamba3-prefill-handle): the former `_neural_session_handle` factory call
# (`mamba_prefill_native_handle_v1`) and `_neural_mamba_session_table` are gone.


# Lane loops-trees-ab (2026-11-11; perf audit 20261111 section 5): this
# file's Transformer forward/backward, Mamba1/2/3 backward and Mamba3 owned
# install call the neural entry table.
# TOMBSTONE: MOJOLEARN_NEURAL_TABLE_ROUTE_OFF (old path, no A/B recorded) deleted 2026-11-15 by lane/expdel2-02-base; code recoverable at 0ace134aa9.
# Restore: git apply experiments/removed/MOJOLEARN_NEURAL_TABLE_ROUTE_OFF.patch; record in docs/TOMBSTONES.md.


def _neural_default_mode(mut mods: NativeModules, context: PythonObject) raises -> PythonObject:
    var mode = _native_entry_fields(mods, context["backend"])["_DEFAULT_MODE"]
    if py_type(mode) is not mods.module("builtins").str:
        raise Error("TypeError: neural default numeric mode requires an exact native string")
    return mode


def _neural_block_owner_setup(mut mods: NativeModules, block: PythonObject, context: PythonObject, mut bindings: PythonObject) raises -> PythonObject:
    # The setup both owner forms share: the operation snapshot, the route's
    # binding connections (answered in `bindings`) and the arithmetic profile
    # words of the block image.
    var fields = neural_block_snapshot(mods, block, context)
    var kind = String(fields["_native_kind"])
    var route = fields["_native_route"]
    var mode = py_get(fields, "numeric_mode")
    bindings = _model_bindings(mods, route, kind, context, mode)
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
    return fields


def _neural_block_owner_typed(mut mods: NativeModules, block: PythonObject, context: PythonObject) raises -> NeuralBlockOwner:
    # The owner of a standalone block's plan build (`neural_block_execution`
    # at an owner-record miss): typed words, no SimpleNamespace, no attribute
    # written or read back. Admission answers into `CausalAdmission`; the
    # plan retains the entry dictionary itself, so the namespace form's entry
    # pin (`causal_pin_entries`, read only by a model capture) is not built.
    var bindings = PythonObject(None)
    var fields = _neural_block_owner_setup(mods, block, context, bindings)
    var kind = String(fields["_native_kind"])
    var int15 = py_get(fields, "_int15")
    var packed = int15 is not None
    var admission = CausalAdmission(String(fields["_native_route"]) == "gpu", kind, packed,
        fields["_neural_weight_names"], block, context)
    causal_lm_admit(mods, admission, bindings[kind], bindings["primitives"], bindings["linalg"] if packed else PythonObject(None))
    return NeuralBlockOwner(fields, context, kind, Int(py=fields["d_model"]),
        py_get(fields, "numeric_profile", PythonObject("fp32_v1")), packed,
        admission.entries, admission.vendor, admission.mode)


def _neural_block_owner_full(mut mods: NativeModules, block: PythonObject, context: PythonObject) raises -> PythonObject:
    # The object owner: only the parallel worker's layer holder
    # (`neural_block_owner`, one build per layer at worker construction),
    # which keeps the holder and reads its fields. A standalone block's plan
    # build takes `_neural_block_owner_typed`.
    var bi = mods.module("builtins")
    var bindings = PythonObject(None)
    var fields = _neural_block_owner_setup(mods, block, context, bindings)
    var kind = String(fields["_native_kind"])
    var route = fields["_native_route"]
    var owner = py_call(mods.module("types").SimpleNamespace)
    py_setattr(owner, "_native_block_fields", Python.list(fields))
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
            String("_native_session"), String("h_last_"),
            String("_native_block_provider_cache"), String("_native_mamba_block_provider_cache"), String("_native_mamba_state"),
            String("_native_primitives_provider_cache"),
            String("_native_mamba_provider_cache")]


# `_neural_owner_env` (one getenv of MOJOLEARN_NEURAL_NATIVE_DISPATCH per
# standalone block call, twice on a miss) and the owner record's env word are
# gone: the dispatch is the build constant NEURAL_NATIVE_DISPATCH
# (providers/python/neural_descriptor_pyapi_door.mojo; lane
# flag-native-dispatch-gate). Record: [key, plan, derived, has_profile,
# next_profile, changed, binding_key, uses_default].


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
            # lane fx-fast-mask: the record's binding key names the set that
            # runs (NEURAL answer); built once per owner record, not per call.
            var mode_word = String(py=mode_name)
            answer_mode_object(mode_name, mode_word, MODE_FAMILY_NEURAL)
    var table = context["bindings"]
    if py_type(table) is not bi.dict:
        return PythonObject(None)
    key.add_identity(table)
    binding_key = route + PythonObject(":") + mode_name
    var entries = py_get(table, binding_key)
    if entries is None or py_type(entries) is not bi.dict:
        return PythonObject(None)
    key.add(entries)
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


def _neural_block_route(mut mods: NativeModules, block: PythonObject, kind: String, device: PythonObject,
    mode: PythonObject, context: PythonObject) raises -> String:
    # The one place a block's `device` is turned into its route (constructor,
    # `set_params` through the constructor, and an owner record's miss).
    # None is this installation's device; the `*Inference` classes are the
    # CPU side by class; an unknown value, an explicit 'gpu' in a CPU-only
    # process and 'cpu' beside an explicit fast mode are refused in words
    # by the one rule (core/device_route.mojo `device_route_plan`).
    var bi = mods.module("builtins")
    var code = device_unpack(device, bi)
    ref cpy = Python().cpython()
    var cpu_class = cpy.lib.get_function[Int32]("PyType_IsSubtype")(py_type(block)._obj_ptr, context["block_classes"]["cpu"][kind]._obj_ptr) != 0
    var selected = context["selected_vendor"]
    var process_vendor = 1
    if selected is None or String(selected) == "cpu":
        process_vendor = 0
    var fast = False
    if mode is not None and py_type(mode) is bi.str:
        fast = String(py=mode.strip().lower()) == "fast"
    if cpu_class:
        return String("cpu")
    return device_name(device_route_plan(code, process_vendor, 0 if fast else 1).device)


def _neural_route_publish(block: PythonObject, route: PythonObject, context: PythonObject) raises:
    # The block's resolved device on the ONE route owner
    # (providers/python/device_route_boundary_door.mojo), written where the
    # device is read: construction, `set_params`, restore. The owner lives in
    # the weak model-record table, not in the instance dictionary, so it is
    # the block's own even when its fields were handed over from another
    # block. `_native_route` stays the per-block plan word every door reads.
    var owner = DeviceRouteOwner(DEVICE_CPU if String(route) == "cpu" else DEVICE_GPU)
    owner.resolved = True
    owner.context = context
    _ = device_route_keep(block, owner^)


def _neural_device_reconcile(mut mods: NativeModules, block: PythonObject, live: PythonObject, context: PythonObject) raises:
    # Owner record miss only, and no read of the `device` parameter: the
    # block's route owner (written at construction) is compared with the
    # plan word of its fields. They differ only when the fields were handed
    # over from the same block on the other device (`check` copies every
    # non-parameter field into the twin it constructed); the block is then
    # rebuilt from its own constructor parameters, as `set_params` rebuilds
    # it. Cost at a miss: one record read and a word compare.
    var bi = mods.module("builtins")
    var route = py_get(live, "_native_route")
    if py_type(route) is not bi.str:
        return
    var held = device_route_owner(block)
    if held is None:
        # No owner (a block built by a whole model, or one that takes no
        # weak reference): its fields are its own.
        return
    if device_name(device_route_value(held).device) == String(route):
        return
    _ = bump_model_generation(block)
    _neural_block_rebuild(mods, block, live, bi.dict(), context)


def neural_block_owner(mut mods: NativeModules, block: PythonObject, context: PythonObject) raises -> PythonObject:
    # The parallel worker keeps its existing owner ABI at one-time layer
    # construction. Repeated standalone execution uses the native plan below.
    return _neural_block_owner_full(mods, block, context)


def neural_block_execution(mut mods: NativeModules, block: PythonObject, context: PythonObject) raises -> NativeBlockExecution:
    var bi = mods.module("builtins")
    var live = neural_block_fields(mods, block, context)
    var record = prepared_get(mods, _NEURAL_OWNER_SLOT, block)
    var hit = False
    if record is not None:
        try:
            hit = _neural_owner_matches(mods, record[0], block, live, context, record[6], Bool(py=record[7]))
        except:
            hit = False
    if hit:
        # The remaining field dictionary is operation-local mutable state.
        # No identity list or SimpleNamespace owner is built on this path.
        var fields = py_call(bi.dict, live)
        _ = py_call(fields.update, record[2])
        py_setitem(fields, "_neural_live_fields", live)
        if Bool(py=record[3]):
            var mode = py_get(fields, "numeric_mode")
            var next_profile = record[4]
            var want_mode = _neural_default_mode(mods, context) if mode is None else mode
            var saved = py_get(fields, "_arithmetic_profile")
            if saved is not None and Bool(py=py_get(fields, "_arithmetic_profile_mode") == want_mode) and not Bool(py=saved == next_profile):
                raise Error("neural block: saved arithmetic profile differs from binding")
            py_setitem(fields, "_arithmetic_profile", next_profile)
            py_setitem(fields, "_arithmetic_profile_mode", want_mode)
            py_setitem(fields, "_arithmetic_profile_changed", record[5])
        return NativeBlockExecution(record[1], fields)
    _neural_device_reconcile(mods, block, live, context)
    var key = PythonObject(None)
    var binding_key = PythonObject(None)
    var uses_default = False
    try:
        key = _neural_owner_key(mods, block, live, context, binding_key, uses_default)
    except:
        key = PythonObject(None)
    var owner = _neural_block_owner_typed(mods, block, context)
    var snapshot = owner.fields
    var kind = owner.kind.copy()
    var planes = kind == "transformer" and owner.packed and py_contains(owner.entries, "forward15")
    var layer = causal_prepared_layer(mods, snapshot, kind, planes)
    var plan = PythonObject(alloc=NativeBlockPlan(owner, layer))
    var reusable = key is not None and _neural_owner_reusable(mods, snapshot, live, context)
    if reusable:
        try:
            reusable = _neural_owner_matches(mods, key, block, live, context, binding_key, uses_default)
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
    prepared_put(mods, _NEURAL_OWNER_SLOT, block, Python.list(key, plan, derived, PythonObject(has_profile), next_profile, changed, binding_key, PythonObject(uses_default)))
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
    # `device` (options[0]) is read here, once, into the block's route; every
    # later operation runs the owner record built for that route and none
    # asks which device the process has. None is this installation's device:
    # the established GPU block class routes to native host arithmetic on a
    # CPU-only installation, just as the package binding selector does.
    var route = PythonObject(_neural_block_route(mods, block, kind, captured_options[0], captured_options[1], context))
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
    # The constructor parameter as given (get_params, clone, pickle).
    py_setitem(fields, "device", captured_options[0])
    _neural_route_publish(block, route, context)
    # Public parameter verification (2026-10-27): the constructor previously
    # retained only its ordered internal tensors, so inherited get_params
    # could not return the caller's weights and set_params could not rebuild.
    # Keep the original ordinary dictionary and its object identities.
    py_setitem(fields, "weights", weights)
    return PythonObject(None)


def neural_block_set_params_binding(mut mods: NativeModules, block: PythonObject, updates: PythonObject, context: PythonObject) raises -> PythonObject:
    # Generation word (bindings/py_model_records.mojo, lane ne1-e-records):
    # every stamped record of this object misses from here on.
    _ = bump_model_generation(block)
    var bi = mods.module("builtins")
    if py_type(updates) is not bi.dict:
        raise Error("TypeError: block parameters require an ordinary dictionary")
    if len(updates) == 0:
        return block
    var fields = neural_block_fields(mods, block, context)
    _neural_block_rebuild(mods, block, fields, updates, context)
    return block


# ---- PyTorch names (lane api-torch, 2026-11-21; docs/api/TORCH_CONVENTIONS.md) ----
# `state_dict`, `parameters`, `named_parameters`, `load_state_dict` and `to`
# of a standalone block. Each is one call over state the block already holds
# (its constructor `weights` dictionary, the one `get_params` returns) or the
# existing rebuild through the constructor (`set_params`); nothing is copied
# and no route is resolved here. Metadata only: no independent units to split.

def neural_block_weights_binding(mut mods: NativeModules, block: PythonObject, form: Int, context: PythonObject) raises -> PythonObject:
    # form 0: `state_dict()` -> a new {name: array} over the same arrays;
    # 1: `parameters()` -> [array, ...]; 2: `named_parameters()` -> [(name, array), ...].
    var bi = mods.module("builtins")
    var fields = neural_block_fields(mods, block, context)
    var weights = py_get(fields, "weights")
    if weights is None:
        raise Error("RuntimeError: neural block: this block holds no weights of its own (a whole model built it); read the weights from that model")
    if form == 0:
        return py_call(bi.dict, weights)
    if form == 1:
        return py_call(bi.list, py_call(bi.dict.values, weights))
    return py_call(bi.list, py_call(bi.dict.items, weights))


def neural_block_load_state_dict_binding(mut mods: NativeModules, block: PythonObject, state: PythonObject, strict: PythonObject,
    context: PythonObject) raises -> PythonObject:
    # `load_state_dict(state, strict=True)`: the `set_params(weights=state)`
    # rebuild, so every name and shape is admitted by the constructor before
    # the live block changes. A block is built from one complete weight set:
    # torch's `strict=False` (keep the unlisted weights) has no meaning here.
    var bi = mods.module("builtins")
    if py_type(strict) is not bi.bool or not Bool(py=strict):
        raise Error("ValueError: neural block: load_state_dict needs strict=True: the block is rebuilt from one complete set of weights (call state_dict(), update it, and load the whole dictionary)")
    if py_type(state) is not bi.dict:
        raise Error("TypeError: neural block: load_state_dict requires an ordinary dictionary of float32 arrays")
    var updates = bi.dict()
    py_setitem(updates, "weights", state)
    return neural_block_set_params_binding(mods, block, updates, context)


def _neural_float32_word(mut mods: NativeModules, value: PythonObject) raises -> Bool:
    # True for `torch.float32`, NumPy's float32 dtype or the word "float32";
    # False for any other dtype. Exact-type admission: a str, or a dtype
    # class named `dtype` (torch.dtype; NumPy's `dtype[float32]` or, from
    # NumPy 2, `Float32DType`: C types whose str is their name); nothing
    # else is read.
    var bi = mods.module("builtins")
    var kind = py_type(value)
    if kind is bi.str:
        var word = String(value)
        return word == "float32" or word == "torch.float32"
    var kind_name = String(py_call(bi.type.__dict__["__name__"].__get__, kind))
    if kind_name != "dtype" and not kind_name.startswith("dtype[") and not kind_name.endswith("DType"):
        raise Error("TypeError: neural block: dtype must be torch.float32, numpy.float32's dtype or 'float32'")
    var word = String(py_call(bi.str, value))
    return word == "float32" or word == "torch.float32"


def neural_block_to_binding(mut mods: NativeModules, block: PythonObject, args: PythonObject, kwargs: PythonObject,
    context: PythonObject) raises -> PythonObject:
    # `to(device=None, dtype=None)` as torch spells it: a str positional is the
    # device, a dtype positional is the dtype. A new device rebuilds the block
    # on that route through the constructor (as `set_params(device=...)`); the
    # device word is admitted there by the one rule (core/device_route.mojo).
    # Blocks are float32 only, so the dtype is admitted or refused, never
    # converted.
    var bi = mods.module("builtins")
    if py_type(args) is not bi.tuple or py_type(kwargs) is not bi.dict:
        raise Error("TypeError: neural block: to() requires ordinary positional and keyword arguments")
    if len(args) > 2:
        raise Error("TypeError: neural block: to() takes a device and a dtype")
    var device = PythonObject(None)
    var dtype = PythonObject(None)
    var has_device = False
    for i in range(len(args)):  # driver-loop: over at most two positional arguments
        var item = args[i]
        if item is None:
            continue
        if py_type(item) is bi.str and String(item) != "float32" and String(item) != "torch.float32":
            if has_device:
                raise Error("TypeError: neural block: to() got two devices")
            device = item
            has_device = True
        else:
            dtype = item
    var keys = py_call(bi.list, kwargs)
    for i in range(len(keys)):  # driver-loop: over the keyword names given
        var key = keys[i]
        if py_type(key) is not bi.str:
            raise Error("TypeError: neural block: to() keywords must be names")
        var name = String(key)
        if name == "device":
            if has_device:
                raise Error("TypeError: neural block: to() got two devices")
            device = kwargs[key]
            has_device = device is not None
        elif name == "dtype":
            dtype = kwargs[key]
        elif name == "non_blocking" or name == "copy":
            # Accepted for PyTorch code that passes them: a rebuild is
            # synchronous and never shares the old block's device storage.
            continue
        else:
            raise Error("TypeError: neural block: to() got an unexpected keyword argument '" + name + "'")
    if dtype is not None and not _neural_float32_word(mods, dtype):
        raise Error("ValueError: neural block: the blocks run float32 only (identical mode defines its bits in float32), got dtype " + String(py_call(bi.repr, dtype)))
    if not has_device:
        return block
    var fields = neural_block_fields(mods, block, context)
    var held = py_get(fields, "device")
    if held is not None and py_type(held) is bi.str and String(held) == String(device):
        return block
    # The `*Inference` classes are the CPU side by class (`_neural_block_route`):
    # a GPU word would rebuild them on the CPU again, so it is refused.
    ref cpy = Python().cpython()
    var kind = fields["_native_kind"]
    var cpu_class = cpy.lib.get_function[Int32]("PyType_IsSubtype")(py_type(block)._obj_ptr, context["block_classes"]["cpu"][kind]._obj_ptr) != 0
    if cpu_class and (py_type(device) is not bi.str or String(device) != "cpu"):
        raise Error("ValueError: neural block: " + String(py_call(bi.type.__dict__["__name__"].__get__, py_type(block)))
            + " runs on the CPU only; build the GPU block class to run on the GPU")
    var updates = bi.dict()
    py_setitem(updates, "device", device)
    return neural_block_set_params_binding(mods, block, updates, context)


def _neural_block_rebuild(mut mods: NativeModules, block: PythonObject, fields: PythonObject, updates: PythonObject, context: PythonObject) raises:
    # The block rebuilt through its constructor door from its stored
    # constructor parameters with `updates` applied (`set_params`; the device
    # reconcile passes no update). A refused replacement leaves the live
    # block untouched.
    var bi = mods.module("builtins")
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
    # A block stored before `device` existed has no such field: None, this
    # installation's device.
    py_setitem(values, "device", py_get(fields, "device"))
    var keys = py_call(bi.list, updates.keys())
    for i in range(len(keys)):  # driver-loop: validate all keys before constructing replacement
        if py_type(keys[i]) is not bi.str or not py_contains(values, keys[i]):
            raise Error("ValueError: Invalid parameter for neural block")
        py_setitem(values, keys[i], updates[keys[i]])
    var options = Python.list(values["device"], values["numeric_mode"])
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
    _neural_route_publish(block, fresh["_native_route"], context)


def _neural_allocate(mut mods: NativeModules, owner: NativeBlockExecution, b: Int, cap: Int) raises -> PythonObject:
    ref plan = owner.plan.downcast_value_ptr[NativeBlockPlan]()[]
    return causal_layer_allocate_fields(mods, plan.kind, owner.fields, plan.entries, plan.numeric_profile, b, cap)


def _neural_layer_run(mut mods: NativeModules, owner: NativeBlockExecution, x: PythonObject, state: PythonObject, step: Bool, stateless: Bool = False) raises -> PythonObject:
    # TOMBSTONE: the Mamba-2 public-step workspace owner (Mamba2PublicWorkspace, mamba2_workspace_api_v1) (moved onto the block step op, lane x5-mamba-on-executor) deleted 2026-11-09 by 08332f3665; code recoverable at 25e5650c14.
    # Restore: git apply experiments/removed/X5_MAMBA_PER_CALL_BODIES.patch
    # Every Mamba step and forward reaches the cached block program through
    # the one native Mamba descriptor (mamba/native_state_dispatch.mojo).
    # The call is UNPACK / RUN / PACK inside `_causal_layer_run_bound`
    # (bindings/causal_lm_public_door.mojo): RUN takes a typed request; the
    # operation's fields dictionary serves UNPACK and PACK only.
    ref plan = owner.plan.downcast_value_ptr[NativeBlockPlan]()[]
    if not plan.layer_route.bound:
        plan.layer_route = causal_layer_route(plan.kind, plan.entries)
    var result = causal_layer_run_fields(mods, owner.fields, plan.entries, plan.layer_route, plan.kind, plan.d_model,
        plan.packed, plan.planes, plan.layer, x, state, step, stateless)
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
    # 0.8.66 regression C5: x is only read; a C-contiguous float32 export is
    # read in place (admit_block_f32 BLOCK_READ), not copied per call; any
    # other x is one copy into uninitialised storage.
    var x = admit_block_f32(mods, value, "x", context, BLOCK_READ)
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


def _neural_input_rows(mut mods: NativeModules, value: PythonObject, d_model: Int, step: Bool, context: PythonObject,
                       mut rows: Int, mut length: Int) raises -> PythonObject:
    # `_neural_input` for a caller that reads only x's address and its (B, L)
    # (the Transformer decode session's per-token door; lane
    # restore-neural-call-paths 2026-11-13): a (B, d_model) step input is
    # admitted as is, B rows of one token in the same C-order bytes, without
    # the (B, 1, d_model) metadata view `_neural_input` builds (a new Array
    # object, fourteen attribute writes and two tuples per token). The same
    # admission (admit_block_f32 BLOCK_READ) and the same refusals with the
    # same messages; `rows` and `length` receive B and L.
    var x = admit_block_f32(mods, value, "x", context, BLOCK_READ)
    if step and len(x.shape) == 2:
        var b2 = Int(py=x.shape[0])
        var d2 = Int(py=x.shape[1])
        if b2 < 1 or d2 != d_model:
            raise Error("neural block: x must be (B, L, d_model) -- or (B, d_model) for a step -- with positive B and L"
                        + " and the weights' d_model " + String(d_model) + ", got shape (" + String(b2) + ", 1, "
                        + String(d2) + ")")
        rows = b2
        length = 1
        return x
    if len(x.shape) != 3 or Int(py=x.shape[0]) < 1 or Int(py=x.shape[1]) < 1 or Int(py=x.shape[2]) != d_model:
        raise Error("neural block: x must be (B, L, d_model)" + (String(" -- or (B, d_model) for a step --") if step else String("")) +
                    " with positive B and L and the weights' d_model " + String(d_model) + ", got shape " + String(x.shape))
    if step and Int(py=x.shape[1]) != 1:
        raise Error("neural block: a decode step takes exactly one token per row, got L = " + String(x.shape[1]) + "; use forward() for prefill")
    rows = Int(py=x.shape[0])
    length = Int(py=x.shape[1])
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
    # The attribute-hook refusal reads only the state class rows: kept per
    # class (native_class_admission; re-run when the class or an ancestor
    # changes, every call on Python < 3.12), not an MRO walk per step (lane
    # p3-neural-blocks).
    if class_record(mods, _NEURAL_STATE_HOOKS_SLOT, cls) is None:
        var ancestry = py_call(bi.type.__dict__["__mro__"].__get__, cls)
        for i in range(len(ancestry)):  # driver-loop: over the state class MRO (class structure, not data); small-loop(passive state access protocol)
            var row = _class_fields(mods, ancestry[i])
            if py_contains(row, "__getattr__"):
                raise Error("TypeError: neural state attribute hooks are not supported")
            for name in ["__getattribute__", "__setattr__"]:  # driver-loop: native admission metadata and retained owner registry
                var hook = py_get(row, PythonObject(name))
                if hook is not None and hook is not bi.object.__dict__[name]:
                    raise Error("TypeError: neural state attribute hooks are not supported")
        class_admit(mods, _NEURAL_STATE_HOOKS_SLOT, cls, PythonObject(True))
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


comptime _NEURAL_STATE_HOOKS_SLOT = "neural_state_hooks_class_v1"


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
                 "_native_session", "h_last_",
                 "_native_block_provider_cache", "_native_mamba_block_provider_cache", "_native_mamba_state"]:
        if py_contains(captured, name):
            py_setitem(fields, name, captured[name])


def _neural_publish_block(owner: NativeBlockExecution) raises:
    _neural_publish_fields(owner.fields)


def _neural_state(mut mods: NativeModules, owner: NativeBlockExecution, state: PythonObject, b: Int, l: Int) raises:
    var bi = mods.module("builtins")
    var fields = owner.fields
    if owner.plan.downcast_value_ptr[NativeBlockPlan]()[].kind == "transformer":
        # Preserve the original short-circuit order while naming the first
        # incompatible field. Admission must finish before state execution.
        if Int(py=state.batch_size) != b:
            raise Error("neural block: transformer state batch_size differs from input")
        if Int(py=state.n_kv_heads) != Int(py=fields["n_kv_heads"]):
            raise Error("neural block: transformer state n_kv_heads differs from block")
        if Int(py=state.head_dim) != Int(py=fields["head_dim"]):
            raise Error("neural block: transformer state head_dim differs from block")
        if Int(py=state.window) != Int(py=fields["window"]):
            raise Error("neural block: transformer state window differs from block")
        var past = Int(py=state.cached_tokens)
        if past < 0:
            raise Error("neural block: transformer state cached_tokens must be nonnegative")
        if past + l > Int(py=state.max_tokens) or past + l > Int(py=fields["max_positions"]):
            raise Error("neural block: cached_tokens plus sequence exceeds state capacity or max_positions")



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
    # Plan words, read once at plan construction (they were three by-name
    # reads of the entry dictionary on every forward).
    var route_session = owner.plan.downcast_value_ptr[NativeBlockPlan]()[].block_session
    var entry_session_fresh = owner.plan.downcast_value_ptr[NativeBlockPlan]()[].session_fresh
    var entry_forward_fresh = owner.plan.downcast_value_ptr[NativeBlockPlan]()[].forward_fresh
    # TOMBSTONE: MOJOLEARN_TRANSFORMER_LEGACY_SETUP / MOJOLEARN_TRANSFORMER_SESSION_FRESH (runtime switches, SUPERSEDED by the block session) deleted 2026-11-09 by lane x4-transformer-on-executor; code recoverable at f59d913b0b.
    # Restore: git apply experiments/removed/MOJOLEARN_TRANSFORMER_FRESH_PREFILL.patch
    # restored 2026-11-09 by lane r1-restore-measured-winners (the switches came back with the fresh prefill arm;
    # default unchanged: the block's session).
    # The forward image's table entry (core/neural_entry_table.mojo), resolved
    # once at plan construction; a GPU block's session is a native registry
    # handle of that image (core/transformer_block_session.mojo, lane
    # transformer-session-native): "block_session" marks a GPU Transformer
    # route (the former "forward_session" Python entry's presence).
    var forward = owner.plan.downcast_value_ptr[NativeBlockPlan]()[].forward_entry
    var reuse = transformer and route_session and not TRANSFORMER_LEGACY_SETUP
    var lock = py_get(fields, "_runtime_lock") if transformer else PythonObject(None)
    var lock_lease = OperationLockLease(lock)
    var locked = transformer and (reuse or fields["_native_session"] is not None or fields["_int15"] is not None)
    if locked:
        lock_lease.acquire()
        mods.released()
    try:
        if transformer and fields["_native_session"] is not None and (not reuse or not transformer_block_session_serves(fields["_native_session"], forward)):
            transformer_block_session_close(fields["_native_session"])
            py_setitem(fields, "_native_session", PythonObject(None))
        if transformer and supplied_state is None and not step and fields["_int15"] is None:
            var fresh_session = reuse and entry_session_fresh and TRANSFORMER_SESSION_FRESH
            if fresh_session or entry_forward_fresh:
                var y = generation_array([b, l, d])
                var keep = List[PythonObject]()
                var addresses: List[Int] = [_addr(x, keep)]
                var weights = fields["_w"]
                for i in range(len(weights)):  # driver-loop: admit transformer weight slots once
                    # The one weight door, as Mamba (core/weight_epoch.mojo): the weight is frozen read-only and
                    # the epoch bumps on a new or reassigned weight, so the block session's address + epoch key
                    # (transformer/attn_step_session.mojo refresh_weights) is exact. Before (smoke 2026-11-15):
                    # `_addr` snapshotted an Array (an in-place edit kept the stale device copy, a NaN weight was
                    # not refused) and copied a foreign buffer whole on every call (a new address, so a re-upload
                    # per call).
                    addresses.append(_weight_addr(weights[i], keep))
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
                        addresses.append(_weight_addr(optional[i], keep))  # the one weight door (above)
                    var options = fields["_opts_tail"]
                    for i in range(len(options)):  # driver-loop: over the fixed block-options tail (argument words, not data); small-loop(block options integer span)
                        params.append(Int(py=options[i]))
                # The forward image's table entry (core/neural_entry_table.mojo):
                # op 2 on the block's session, op 4 ephemeral. The former Python
                # entries (transformer_session_forward_fresh,
                # transformer_forward_fresh) and the env-gated provider capsule
                # arms ran the same bodies with these words.
                if fresh_session:
                    if fields["_native_session"] is None:
                        py_setitem(fields, "_native_session", transformer_block_session_open(forward))
                    var session_owner = fields["_native_session"]
                    var handle = transformer_block_session_handle(session_owner)
                    _ = _neural_table_reserved(forward, handle, NEURAL_BLOCK_RESERVE,
                        NEURAL_BLOCK_RELEASE, NEURAL_BLOCK_SESSION_FORWARD_FRESH, addresses, params, List[Float64]())
                    mods.released()
                    _ = session_owner^
                else:
                    _ = _neural_table_call(forward, NEURAL_BLOCK_FORWARD_FRESH,
                        addresses, params, List[Float64]())
                    mods.released()
                if locked:
                    lock_lease.release()
                _ = keep^
                return y
        # lane fix-mamba12-forward-exec: a Mamba-1/2 device forward with no
        # state runs stateless (no host zero state allocated, uploaded and
        # read back per call; decided once per plan:
        # `NativeBlockPlan.stateless_forward`)
        if supplied_state is None and not step:
            ref nplan = owner.plan.downcast_value_ptr[NativeBlockPlan]()[]
            if nplan.stateless_forward:
                # TOMBSTONE: MOJOLEARN_IDN_MAMBA12_FORWARD_DOOR_OLD (old path, no A/B recorded) deleted 2026-11-15 by lane/expdel2-02-base; code recoverable at f1c5239d24.
                # Restore: git apply experiments/removed/MOJOLEARN_IDN_MAMBA12_FORWARD_DOOR_OLD.expdel2-02-base.patch; record in docs/TOMBSTONES.md.
                # RESTORED 2026-11-15 by lane/expdel2-restore-owed: owed B arm (owed-pre0867.json), under four weeks old; the deletion was premature.
                comptime if not IDN_MAMBA12_FORWARD_DOOR_OLD:
                    # lane fix-mamba-forward-doors: one direct call with the
                    # plan's prepared layer and held Mamba entry (the
                    # causal_layer_run_fields hop is the B arm)
                    if nplan.layer is not None:
                        if nplan.mamba_entry.word == 0:
                            nplan.mamba_entry = causal_mamba_entry(nplan.entries)
                        var direct = causal_mamba_stateless_run(mods, owner.fields, nplan.mamba_entry, nplan.kind,
                            nplan.d_model, nplan.layer, x)
                        if locked:
                            lock_lease.release()
                        return direct
                var fresh_result = _neural_layer_run(mods, owner, x, PythonObject(None), step, True)
                if locked:
                    lock_lease.release()
                return fresh_result
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
    if py_type(value) is context["array_class"]:
        var captured = neural_array_snapshot(mods, value, context)
        if len(captured.shape) != 1:
            raise Error("neural block: lengths must be one-dimensional")
        if Int(py=captured.shape[0]) != b:
            raise Error("neural block: one length is required per batch row")
        var code = String(captured.dtype)
        if code != "<i4" and code != "<i1" and code != "<u2":
            raise Error("TypeError: neural block lengths entries must be integers, not bool or float")
        var result = List[Int](length=b, fill=0)
        var address = Int(py=captured._addr)
        # Native reads of the retained storage (no CPython call per row).
        for i in range(b):
            var n = Int(Pointer[Int32, ImmUntrackedOrigin](unsafe_from_address=address)[i]) if code == "<i4" else (
                Int(Pointer[Int8, ImmUntrackedOrigin](unsafe_from_address=address)[i]) if code == "<i1" else
                Int(Pointer[UInt16, ImmUntrackedOrigin](unsafe_from_address=address)[i]))
            if n < 1 or n > l:
                raise Error("neural block: lengths must be in [1, L]")
            result[i] = n
        _ = captured^
        return result^
    elif py_type(value) is not bi.list and py_type(value) is not bi.tuple:
        # One buffer export (the former memoryview + one boxed index read
        # per batch row, lane loops-neural 2026-11-11), then native reads
        # through the view's stride. A float or bool format refuses with the
        # text the per-row `int` check raised.
        admit_native_buffer_exporter(value)
        var view = buffer_view(value)
        if view.ndim != 1:
            raise Error("neural block: lengths must be one-dimensional")
        if view.shape[0] != b:
            raise Error("neural block: one length is required per batch row")
        if b > 0 and (view.kind == BOUNDARY_KIND_FLOAT or view.kind == BOUNDARY_KIND_BOOL):
            raise Error("TypeError: neural block lengths entries must be integers, not bool or float")
        var viewed = List[Int](length=b, fill=0)
        for i in range(b):
            var n = view_int_at(view.layout, view.kind, i)
            if n < 1 or n > l:
                raise Error("neural block: lengths must be in [1, L]")
            viewed[i] = n
        view.release()
        return viewed^
    if len(value) != b:
        raise Error("neural block: one length is required per batch row")
    # An exact list or tuple of Python ints: one borrowed slot read and one
    # PyLong_AsLongLong per batch row (no boxed index, no item reference,
    # no `__getitem__`); a Python list has no buffer to read once.
    var is_tuple = py_type(value) is bi.tuple
    # perf-audit-3: the bulk form resolves the process tables once per
    # call; the per-item form looked two global slots up per batch row.
    return read_exact_int_items(value, b, is_tuple,
        "TypeError: neural block lengths entries must be integers, not bool or float",
        1, l, "neural block: lengths must be in [1, L]")


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


struct NeuralTrainingSelection(Movable):
    # The connected training entries and the neural entry table key
    # (core/neural_entry_table.mojo) of the image they belong to.
    var entries: PythonObject
    var key: NeuralEntryKey

    def __init__(out self, entries: PythonObject, key: NeuralEntryKey):
        self.entries = entries
        self.key = key.copy()


def neural_training_entries(mut mods: NativeModules, fields: PythonObject, context: PythonObject) raises -> PythonObject:
    var selection = neural_training_selection(mods, fields, context)
    return selection.entries


def neural_training_selection(mut mods: NativeModules, fields: PythonObject, context: PythonObject) raises -> NeuralTrainingSelection:
    var bi = mods.module("builtins")
    var mode = py_get(fields, "numeric_mode")
    neural_plain_value(mods, mode, 0, context)
    if mode is None:
        mode = _neural_default_mode(mods, context)
    elif String(py=mode) != "identical" and String(py=mode) != "fast":
        # Stored modes are already normalized; only another spelling pays
        # the two Python string calls (lane p3-neural-blocks).
        mode = mode.strip().lower()
    var mode_text = String(py=mode)
    if mode_text != "identical" and mode_text != "fast":
        raise Error("NotImplementedError: neural backward requires identical or fast mode")
    answer_mode_word(mode_text, MODE_FAMILY_NEURAL)
    var key = PythonObject(String(py=fields["_native_route"]) + ":" + mode_text)
    var selected = py_get(context["training_bindings"], key)
    if selected is None:
        if String(py=fields["_native_route"]) == "cpu":
            # A CPU-routed block in a GPU installation (device='cpu'): the
            # host training images are the CPU-only installation's and are
            # not loaded beside their GPU siblings (failure path only).
            # Goes when the host training images load beside the GPU set
            # (the duplicate type registration, lane dw-core round 2).
            device_cpu_absent(String("backward"), String("the host training images are not loaded beside the GPU set in this installation"))
        raise Error("NotImplementedError: native training bindings unavailable for " + String(key))
    var family = "transformer" if String(fields["_native_kind"]) == "transformer" else "mamba"
    var entries = selected[family]
    if py_type(entries) is bi.str:
        raise Error("RuntimeError: " + String(entries))
    # A compiled binding's vendor and numeric-mode answers are fixed for the
    # module object: ask them once per binding object (weakref-guarded
    # prepared slot, keyed by the entries object and the family), not on
    # every forward/backward/step of every block.
    # The transformer arithmetic profile and its changed flag are answers of
    # the compiled module too: they join the cached verdict (lane
    # p3-neural-blocks; they were two binding crossings per step).
    var vendor: String
    var code: Int
    var profile = PythonObject(None)
    var profile_changed = PythonObject(None)
    var verdict = PythonObject(None)
    try:
        verdict = prepared_get(mods, String(_NEURAL_TRAINING_VERDICT_SLOT) + String(family), entries)
    except:
        verdict = PythonObject(None)
    if verdict is not None and len(verdict) == 4:
        vendor = String(py=verdict[0])
        code = Int(py=verdict[1])
        profile = verdict[2]
        profile_changed = verdict[3]
    else:
        vendor = String(_native_function(mods, entries, family + "_vendor")())
        mods.released()
        code = Int(py=_native_function(mods, entries, family + "_numeric_mode")())
        mods.released()
        if family == "transformer" and _native_has_entry(mods, entries, "transformer_arithmetic_profile"):
            profile = _native_function(mods, entries, "transformer_arithmetic_profile")()
            mods.released()
            profile_changed = PythonObject(Bool(py=_native_function(mods, entries, "transformer_arithmetic_profile_changed")()))
            mods.released()
        try:
            prepared_put(mods, String(_NEURAL_TRAINING_VERDICT_SLOT) + String(family), entries,
                         Python.tuple(PythonObject(vendor), PythonObject(code), profile, profile_changed))
        except:
            # An entries object without weak references keeps the per-call
            # queries; the verdict is never cached for it.
            pass
    if code != (1 if mode_text == "identical" else 0):  # lane fx-fast-mask: the answered word (it read the raw one)
        raise Error("RuntimeError: native training binding numeric mode differs from the block")
    var entry_key = NeuralEntryKey(code, neural_entry_vendor(vendor))
    if String(fields["_native_route"]) == "cpu":
        if vendor != "cpu":
            raise Error("RuntimeError: CPU backward requires native CPU kernels")
    elif vendor != "metal" and vendor != "cuda" and vendor != "hip":
        raise Error("RuntimeError: GPU backward requires native GPU kernels")
    if family == "transformer" and profile_changed is not None:
        var saved = py_get(fields, "_arithmetic_profile")
        if saved is not None and Bool(py=py_get(fields, "_arithmetic_profile_mode") == mode) and not Bool(py=saved == profile):
            raise Error("RuntimeError: saved transformer arithmetic profile differs from training binding")
        py_setitem(fields, "_arithmetic_profile", profile)
        py_setitem(fields, "_arithmetic_profile_mode", mode)
        py_setitem(fields, "_arithmetic_profile_changed", profile_changed)
    # The provider-capsule caches below no longer serve this file's forward,
    # backward or Mamba3 restore (they call the neural entry table by `entry_key`).
    # They remain for their unconverted readers: bindings/neural_prefill_native.mojo,
    # bindings/samba_inference_native.mojo, providers/python/samba_execution_pyapi.mojo
    # and bindings/mamba_public_native.mojo.
    if family == "transformer":
        var native = PythonObject(None)
        if code == 1 and NEURAL_NATIVE_DISPATCH:
            native = neural_descriptor_reuse(py_get(fields, "_native_block_provider_cache"),
                _native_function(mods, entries, "transformer_block_api_v1"), String(TRANSFORMER_BLOCK_ABI_NAME), 36)
            mods.released()
        py_setitem(fields, "_native_block_provider_cache", native)
    else:
        var native = PythonObject(None)
        if code == 1 and NEURAL_NATIVE_DISPATCH:
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
    return NeuralTrainingSelection(entries, entry_key)


def neural_training_entries(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    # Classical and other callers keep their per-call module checks.
    var mods = NativeModules()
    return neural_training_entries(mods, fields, context)


def neural_training_selection(fields: PythonObject, context: PythonObject) raises -> NeuralTrainingSelection:
    # The connected training entries and their neural table key, for callers
    # without a NativeModules cache (the Samba executor).
    var mods = NativeModules()
    return neural_training_selection(mods, fields, context)


def _neural_int_words(values: PythonObject) raises -> List[Int]:
    # Internal boundary vectors already contain admitted scalar/address words.
    var result = List[Int]()
    for i in range(len(values)):  # driver-loop: over address and parameter argument words (fixed per operation, not data); small-loop(public operation descriptor words)
        result.append(Int(py=values[i]))
    return result^


def neural_mamba3_restore(mut mods: NativeModules, fields: PythonObject, entries: PythonObject, weights: PythonObject,
                          route: Mamba3StateRoute) raises:
    # `route`: the block's Mamba-3 key (its training selection), which names
    # the training image holding its prefill session (route.session_ref()).
    # `entries` is unused (its pre-table B arm was deleted 2026-11-15); kept for the callers.
    if Bool(py=py_get(fields, "_owned_restore_pending", PythonObject(False))):
        var entry = route.session_ref()
        if (mamba3_prefill_caps(entry) & MAMBA3_CAP_OWNED_WEIGHTS) == 0:
            raise Error("RuntimeError: Mamba3 owned weights require the compiled owned-weight entry")
        var session = mamba3_prefill_session_ensure(fields, entry)
        var addresses = List[Int]()
        for i in range(len(weights)):  # driver-loop: over the fixed owned-weight registry
            addresses.append(Int(py=weights[i]._addr))
        # Owned snapshot install (op 15) on the block's session handle; the
        # former Python entry mamba3_prefill_session_install_weights and the
        # env-gated provider arm ran the same install with these words.
        var generation: Int
        generation = mamba3_prefill_session_run(session, NEURAL_MAMBA3_OWNED_INSTALL,
            addresses, [Int(py=fields["d_model"])])
        mods.released()
        _ = weights
        _ = session^
        py_setitem(fields, "_owned_generation", PythonObject(generation))
        py_setitem(fields, "_owned_restore_pending", False)
    if py_get(fields, "_owned_generation") is not None:
        if (not route.has_sessions()
                or not mamba3_prefill_session_serves(py_get(fields, "_prefill_session"), route.session_ref())):
            raise Error("RuntimeError: Mamba3 owned snapshot requires its installed binding/setup")


def neural_mamba3_restore(fields: PythonObject, entries: PythonObject, weights: PythonObject,
                          route: Mamba3StateRoute) raises:
    # Classical and other callers keep their per-call module checks.
    var mods = NativeModules()
    neural_mamba3_restore(mods, fields, entries, weights, route)


comptime _NEURAL_SNAPSHOT_SLOT = "neural_block_snapshot_v1"
comptime _NEURAL_TRAINING_VERDICT_SLOT = "neural_training_verdict_v1:"


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
    for name in _neural_snapshot_names():  # driver-loop: over the fixed snapshot field names (model structure, not data); small-loop(snapshot input field names)
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
        for name in _neural_snapshot_names():  # driver-loop: over the fixed snapshot field names (model structure, not data); small-loop(snapshot input field names)
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
    var training = neural_training_selection(mods, fields, context)
    var entries = training.entries
    var name = ("transformer" if transformer else kind) + "_backward"
    if not _native_has_entry(mods, entries, name):
        raise Error("RuntimeError: loaded native training binding lacks " + name)
    var d = Int(py=fields["d_model"])
    var x = _neural_input(mods, input, d, False, context)
    var b = Int(py=x.shape[0])
    var l = Int(py=x.shape[1])
    var dy = admit_block_f32(mods, grad_output, "grad_output", context, BLOCK_OPERATION)
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
        # Row N115: admitted once per weight identity. The verdict is kept
        # on the block's record (generation word plus the block's own weight
        # list, item by item) with the admitted tensors; a later step whose
        # snapshot holds the very same tensor objects reuses it. Any
        # set_params, load, attribute assignment or replaced list item
        # misses and the full admission runs again. Only native Arrays the
        # snapshot passes through unchanged are reused (a foreign buffer is
        # copied per call and never matches).
        var record_name = String("neural_block_backward_weights:") + kind + ":" + String(mode)
        var kept = records_stamped_get(RECORDS_PREPARED, block, record_name)
        var reused = False
        if kept is not None and len(kept[0]) == len(weights):
            reused = True
            for i in range(len(weights)):  # driver-loop: over the fixed block weight registry (identity compare)
                if not (kept[0][i] is weights[i]):
                    reused = False
                    break
        if reused:
            # The descriptor recheck the full admission makes after the
            # GIL release above.
            _causal_array_guard()
            weights = kept[1]
        else:
            var plan = bi.dict()
            py_setitem(plan, "block_kwargs", bi.dict())
            if kind == "mamba2":
                py_setitem(plan["block_kwargs"], "dt_limit", fields["dt_limit"])
            var named = bi.dict()
            var names = fields["_neural_weight_names"]
            for i in range(len(names)):  # driver-loop: over the fixed block weight registry
                py_setitem(named, names[i], weights[i])
            var route = fields["_native_route"]
            var snapshot_weights = py_call(bi.list, weights)
            weights = model_recurrent_weights_check(mods, named, kind, d, plan, route, _model_bindings(mods, route, kind, context, mode), context)
            var same_objects = False
            if py_contains(fields, "_neural_original_w") and py_contains(fields, "_neural_live_fields"):
                var originals = fields["_neural_original_w"]
                same_objects = len(snapshot_weights) == len(originals)
                if same_objects:
                    for i in range(len(snapshot_weights)):  # driver-loop: over the fixed block weight registry (identity compare)
                        if not (snapshot_weights[i] is originals[i]):
                            same_objects = False
                            break
            if same_objects:
                var content = ContentRecord()
                var live = fields["_neural_live_fields"]
                content_value(content, live["_w"], 0, 1, True)
                if kind == "mamba2" and py_contains(live, "dt_limit"):
                    content_value(content, live["dt_limit"], 0, 2, True)
                _ = records_stamped_put(RECORDS_PREPARED, block, record_name, content^,
                                        Python.tuple(snapshot_weights, weights))
    var keep = List[PythonObject]()
    # Plain launch words for the neural entry table (core/neural_entry_table.mojo):
    # the former Python address/parameter lists, their appends and the
    # per-call int conversions are gone.
    var addresses: List[Int] = [_addr(x, keep)]
    var gradients = bi.list()
    _ = py_call(gradients.append, generation_array([b, l, d]))
    for i in range(len(weights)):  # driver-loop: over parameter buffers and their shapes
        # Integration (integ/pre-0867-w5, lane x56d): a Mamba block's weights
        # go through the binding's one weight door (generation_weight_address:
        # frozen read-only, the weight epoch bumped on a new or copied
        # weight), so the cached block program keys them by address + epoch
        # (core/weight_epoch.mojo) and an unchanged weight is NOT re-uploaded
        # per backward call. The transformer's weights take the same door
        # (smoke 2026-11-15: its separate `_addr` route snapshotted Arrays
        # and copied foreign weights per call).
        addresses.append(_weight_addr(weights[i], keep))
        var shape = List[Int]()
        var weight_shape = weights[i].shape  # read once, not once per axis
        for j in range(len(weight_shape)):  # driver-loop: over tensor shape metadata
            shape.append(Int(py=weight_shape[j]))
        _ = py_call(gradients.append, generation_array(shape))
    addresses.append(_addr(dy, keep))
    for i in range(len(gradients)):  # driver-loop: over gradient buffer addresses
        addresses.append(_addr(gradients[i], keep))
    var params: List[Int] = [b, l, d]
    var table_key = training.key.copy()
    # The training entry, resolved once per backward (lane
    # entry-table-zero-alloc): every table call below goes through it.
    var train_entry = neural_ref[NEURAL_TABLE_BLOCK_TRAIN](table_key) if transformer else neural_ref[NEURAL_TABLE_MAMBA_TRAIN](table_key)
    if transformer:
        for name_key in [String("n_heads"), String("n_kv_heads"), String("head_dim"), String("intermediate"), String("window")]:  # driver-loop: over transformer launch dimensions
            params.append(Int(py=fields[name_key]))
        var lock = fields["_runtime_lock"]
        var lock_lease = OperationLockLease(lock)
        lock_lease.acquire()
        mods.released()
        try:
            var session = _native_has_entry(mods, entries, "transformer_session_backward") and not TRANSFORMER_LEGACY_SETUP
            if session:
                # The block's session is a registry handle of the training
                # image (the GPU image publishes one entry for forward and
                # training, so a forward's session serves the backward).
                if fields["_native_session"] is not None and not transformer_block_session_serves(fields["_native_session"], train_entry):
                    transformer_block_session_close(fields["_native_session"])
                    py_setitem(fields, "_native_session", PythonObject(None))
                if fields["_native_session"] is None:
                    py_setitem(fields, "_native_session", transformer_block_session_open(train_entry))
                # Session backward (op 3) through the training image's table
                # entry. The former Python entry transformer_session_backward
                # and the env-gated provider capsule arm ran the same
                # _transformer_run_session_backward with these words.
                var session_owner = fields["_native_session"]
                var handle = transformer_block_session_handle(session_owner)
                _ = _neural_table_reserved(train_entry, handle, NEURAL_BLOCK_RESERVE,
                    NEURAL_BLOCK_RELEASE, NEURAL_BLOCK_SESSION_BACKWARD, addresses, params, List[Float64]())
                mods.released()
                _ = session_owner^
            else:
                # TOMBSTONE: TRANSFORMER_FORWARD_TAPE_API (DELETED: superseded by the attention step op and the tape rule, lane x6) deleted 2026-11-09 by e49d0ee90d; code recoverable at 4f41e3d67d.
                # Restore: git apply experiments/removed/TRANSFORMER_FORWARD_TAPE_API.patch
                # (the session tape's invalidation before a per-call backward.)
                # Ephemeral backward (op 5): the former transformer_backward entry.
                _ = _neural_table_call(train_entry, NEURAL_BLOCK_BACKWARD, addresses, params, List[Float64]())
                mods.released()
            lock_lease.release()
        except error:
            mods.released()
            lock_lease.release()
            raise error
    else:
        # Mamba backward through the training image's table entry (ops 2/3/4,
        # 7 session, 9 owned). The former Python entries (mambaN_backward,
        # mamba3_prefill_session_backward[_owned]) and the env-gated provider
        # capsule arms ran the same bodies with these words; Mamba2's two
        # dt_limit scalars travel as the float span, as the capsule arm sent them.
        var floats = List[Float64]()
        if kind == "mamba2":
            var limit = fields["dt_limit"]
            floats.append(Float64(py=limit[0]))
            floats.append(Float64(py=limit[1]))
        if kind == "mamba3":
            var route = Mamba3StateRoute(table_key.mode, table_key.vendor, table_key.vendor == 0)
            neural_mamba3_restore(mods, fields, entries, weights, route)
            var generation = py_get(fields, "_owned_generation")
            var session = (mamba3_prefill_caps(train_entry) & MAMBA3_CAP_SESSION_PREFILL) != 0
            if generation is not None:
                var owned_addresses: List[Int] = [addresses[0]]
                for i in range(10, len(addresses)):  # driver-loop: over input-gradient and result address slots
                    owned_addresses.append(addresses[i])
                params.append(Int(py=generation))
                var owner = fields["_prefill_session"]
                _ = mamba3_prefill_session_run(owner, NEURAL_MAMBA3_OWNED_BACKWARD, owned_addresses, params)
                mods.released()
                _ = owner^
            elif session:
                # The block's session: a native registry handle of the training
                # image (core/mamba3_prefill_session.mojo, lane
                # mamba3-prefill-handle); the former Python entry
                # mamba3_prefill_session_create and the _prefill_binding check.
                var owner = mamba3_prefill_session_ensure(fields, train_entry)
                _ = mamba3_prefill_session_run(owner, NEURAL_MAMBA3_SESSION_BACKWARD, addresses, params)
                mods.released()
                _ = owner^
            else:
                _ = _neural_table_call(train_entry, NEURAL_MAMBA3_BACKWARD, addresses, params, List[Float64]())
                mods.released()
        else:
            _ = _neural_table_call(train_entry,
                NEURAL_MAMBA2_BACKWARD if kind == "mamba2" else NEURAL_MAMBA1_BACKWARD, addresses, params, floats)
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
        if code != (1 if word_runs_identical(String(mode), MODE_FAMILY_NEURAL) else 0):  # lane fx-fast-mask
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
    var transient_names = [String("_runtime_lock"), String("_native_session"), String("_native_block_provider_cache"),
        String("_native_primitives_provider_cache")]
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
    var runs_identical = word_runs_identical(String(mode), MODE_FAMILY_NEURAL)  # lane fx-fast-mask: an answered `fast` restores the identical route
    if runs_identical and NEURAL_NATIVE_DISPATCH:
        py_setitem(fields, "_runtime_lock", operation_lock_create())
        mods.released()
    else:
        py_setitem(fields, "_runtime_lock", context["rlock"]())
        mods.released()
    var session = PythonObject(None)
    if String(fields["_native_route"]) == "gpu" and runs_identical and NEURAL_NATIVE_DISPATCH:
        # A native registry handle of the block's GPU image (lane
        # transformer-session-native; the former `transformer_session_create`).
        var bindings = _model_bindings(mods, fields["_native_route"], "transformer", context, mode)
        session = transformer_block_session_open(neural_ref[NEURAL_TABLE_BLOCK_FORWARD](
            NeuralEntryKey(Int(py=bindings["entry_mode"]), Int(py=bindings["entry_vendor"]))))
    py_setitem(fields, "_native_session", session)
    py_setitem(fields, "_native_block_provider_cache", PythonObject(None))
    _neural_route_publish(block, fields["_native_route"], context)
    return PythonObject(None)

def neural_block_used_binding(mut mods: NativeModules, block: PythonObject, vendor: Bool, context: PythonObject) raises -> PythonObject:
    # `vendor_used` / `numeric_mode_used` of a block: the words the admitted
    # image of ITS route reported, held by the retained plan. The class-level
    # read-back (bindings/mode_metadata_native_door.mojo) names the class's
    # GPU binding whatever the block's device, so the block classes answer
    # here (metadata; one owner-record read, no kernel).
    _ = generation_initialize_binding(context["generation_classes"])
    var owner = neural_block_execution(mods, block, context)
    var result = PythonObject(None)
    if vendor:
        result = PythonObject(owner.plan.downcast_value_ptr[NativeBlockPlan]()[].vendor.copy())
    else:
        result = PythonObject("identical" if owner.plan.downcast_value_ptr[NativeBlockPlan]()[].mode == 1 else "fast")
    _neural_publish_block(owner)
    return result


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
        elif operation == 8:
            result = neural_block_used_binding(mods, args[0], True, args[1])
        elif operation == 9:
            result = neural_block_used_binding(mods, args[0], False, args[1])
        elif operation == 10:
            result = neural_block_weights_binding(mods, args[0], 0, args[1])
        elif operation == 11:
            result = neural_block_weights_binding(mods, args[0], 1, args[1])
        elif operation == 12:
            result = neural_block_weights_binding(mods, args[0], 2, args[1])
        elif operation == 13:
            result = neural_block_load_state_dict_binding(mods, args[0], args[1], args[2], args[3])
        elif operation == 14:
            result = neural_block_to_binding(mods, args[0], args[1], args[2], args[3])
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
