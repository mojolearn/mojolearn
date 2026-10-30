# SPDX-License-Identifier: Apache-2.0
# Stable whole-operation model/state metadata. Original records are retained
# only for identity and native dictionary publication, never runtime attributes.
from std.python import Python, PythonObject
from std.collections import Dict
from std.ffi import _Global
from bindings.native_identity_key import NativeIdentityKey
from std.python._cpython import PyObjectPtr
from bindings.native_loaded_module import native_loaded_module, NativeModules
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.causal_model_config import model_record_fields, model_plan_snapshot
from bindings.neural_block_admission import neural_plain_value, neural_string_keys, neural_metadata_snapshot, neural_block_snapshot, neural_native_scalar, neural_block_fields
from bindings.mode_metadata_native import _fields
from bindings.generation_buffers import generation_array_snapshot, generation_buffer_view
from bindings.native_prepared_owner import prepared_get, prepared_put, prepared_drop


struct CausalFieldNames(Defaultable, Movable):
    # Immutable interface schema only: no model, state or mutable user data.
    # Retain the exact Python names once because record admission consumes
    # Python string keys; rebuilding them in every layer/token adds no work.
    var model: PythonObject
    var layer_scalars: PythonObject
    var layer_buffers: PythonObject
    var layer_all: PythonObject
    var resident: PythonObject

    def __init__(out self):
        self.model = PythonObject(None)
        self.layer_scalars = PythonObject(None)
        self.layer_buffers = PythonObject(None)
        self.layer_all = PythonObject(None)
        self.resident = PythonObject(None)


comptime CAUSAL_FIELD_NAMES = _Global[StorageType=CausalFieldNames,
    name="MojolearnCausalFieldNamesV1", init_fn=CausalFieldNames.__init__]


def causal_snapshot_fields(mut mods: NativeModules, value: PythonObject) raises -> PythonObject:
    if py_type(value) is not mods.module("types").SimpleNamespace:
        raise Error("TypeError: causal operation metadata requires its native namespace")
    ref cpy = Python().cpython()
    var raw = cpy.lib.get_function[PyObjectPtr]("PyObject_GenericGetDict")(value._obj_ptr, PyObjectPtr())
    if not raw:
        raise cpy.unsafe_get_error()
    return PythonObject(from_owned=raw)


def _causal_namespace(mut mods: NativeModules, fields: PythonObject) raises -> PythonObject:
    var value = py_call(mods.module("types").SimpleNamespace)
    _ = py_call(causal_snapshot_fields(mods, value).update, fields)
    return value


def _causal_native_plain(mut mods: NativeModules, value: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = mods.module("builtins")
    var cls = py_type(value)
    if value is None or cls is bi.str or cls is bi.int or cls is bi.float or cls is bi.bool:
        return value
    if cls is bi.list or cls is bi.tuple:
        var result = Python.list()
        for i in range(len(value)):  # driver-loop: over already admitted scalar metadata containers
            _ = py_call(result.append, _causal_native_plain(mods, value[i], context))
        return py_call(bi.tuple, result) if cls is bi.tuple else result
    if cls is bi.dict:
        neural_string_keys(mods, value)
        var result = Python.dict()
        var keys = py_call(bi.list, value.keys())
        for i in range(len(keys)):  # driver-loop: over captured passive metadata dictionary fields
            py_setitem(result, keys[i], _causal_native_plain(mods, value[keys[i]], context))
        return result
    if neural_native_scalar(mods, value, context, True):
        return py_call(bi.int, value)
    if neural_native_scalar(mods, value, context):
        return py_call(bi.float, value)
    raise Error("TypeError: causal metadata requires native scalar/container values")


def _causal_buffer(value: PythonObject, context: PythonObject) raises -> PythonObject:
    if value is None:
        return value
    if py_type(value) is context["array_class"]:
        return generation_array_snapshot(value)
    return generation_buffer_view(value)


def causal_model_origin(mut mods: NativeModules, model: PythonObject) raises -> PythonObject:
    if py_type(model) is mods.module("types").SimpleNamespace:
        return py_get(causal_snapshot_fields(mods, model), "_causal_original", model)
    return model


def causal_state_origin(mut mods: NativeModules, state: PythonObject) raises -> PythonObject:
    if py_type(state) is mods.module("types").SimpleNamespace:
        return py_get(causal_snapshot_fields(mods, state), "_causal_original", state)
    return state


def causal_state_fields(mut mods: NativeModules, state: PythonObject, context: PythonObject) raises -> PythonObject:
    if py_type(state) is mods.module("types").SimpleNamespace:
        var fields = causal_snapshot_fields(mods, state)
        if not py_contains(fields, "_causal_original"):
            raise Error("TypeError: causal state requires its retained native owner")
        return fields
    return model_record_fields(mods, state, context["generation_classes"][1],
        Python.tuple("batch_size", "max_tokens", "positions", "layers", "_owner", "_usable"))


struct CausalExecutionEntries(Movable, Writable):
    var entries: PythonObject

    def __init__(out self, mut mods: NativeModules, entries: PythonObject) raises:
        self.entries = py_call(mods.module("builtins").dict, entries)

    def write_to(self, mut writer: Some[Writer]):
        writer.write("CausalExecutionEntries(native)")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def causal_pin_entries(mut mods: NativeModules, fields: PythonObject, entries: PythonObject) raises:
    neural_string_keys(mods, fields)
    neural_string_keys(mods, entries)
    py_setitem(fields, "_causal_entries_owner", PythonObject(alloc=CausalExecutionEntries(mods, entries)))


def _causal_entries(mut mods: NativeModules, entries: PythonObject, pinned: PythonObject, context: PythonObject) raises -> PythonObject:
    neural_string_keys(mods, entries)
    var bi = mods.module("builtins")
    var types = mods.module("types")
    var original = PythonObject(None)
    if pinned is not None:
        original = pinned.downcast_value_ptr[CausalExecutionEntries]()[].entries
        if len(original) != len(entries):
            raise Error("TypeError: CausalLM native execution registry changed after admission")
    elif py_get(entries, "parallel_call") is not None:
        ref cpy = Python().cpython()
        var runtime = context["runtime"]
        if py_type(runtime) is not types.ModuleType:
            raise Error("TypeError: parallel execution requires its connected native runtime")
        var raw = cpy.lib.get_function[PyObjectPtr]("PyModule_GetDict")(runtime._obj_ptr)
        if not raw:
            raise cpy.unsafe_get_error()
        var connected = PythonObject(from_owned=cpy.Py_NewRef(raw))
        neural_string_keys(mods, connected)
        if entries["parallel_call"] is not connected["parallel_causal_call"]:
            raise Error("TypeError: parallel execution entry changed after native connection")
    else:
        raise Error("TypeError: CausalLM execution requires its constructed native entry owner")
    var result = Python.dict()
    var names = py_call(bi.list, entries.keys())
    for i in range(len(names)):  # driver-loop: over capture the admitted execution registry once
        var name = String(names[i])
        var item = entries[names[i]]
        if original is not None and (not py_contains(original, names[i]) or item is not original[names[i]]):
            raise Error("TypeError: CausalLM native execution entry changed after admission")
        if (name == "native_primitives" or name == "native_block_operations" or name == "native_mamba"
                or name == "block_binding" or name == "native_session_provider"
                or name == "mamba2_workspace_provider"):
            # Typed provider/lifetime admission occurs at its native boundary;
            # these values are retained, never called or traversed as Python.
            py_setitem(result, names[i], item)
        elif name == "resident_read_v1" or name == "resident_sampling_v1":
            # Native capability metadata, not an executable registry callback.
            # The constructed-owner identity check above still forbids edits.
            if py_type(item) is not bi.bool:
                raise Error("TypeError: CausalLM resident capability must be bool")
            py_setitem(result, names[i], item)
        elif (name == "cpu" or name == "weight_names" or name == "resident_vendor"
                or name == "mamba1_profile" or name == "native_enabled"):
            neural_plain_value(mods, item, 0, context)
            py_setitem(result, names[i], neural_metadata_snapshot(mods, item))
        else:
            if py_type(item) is not py_type(bi.len) or py_type(item.__self__) is not types.ModuleType:
                raise Error("TypeError: CausalLM execution entries must remain native compiled functions")
            if item.__self__ is bi:
                raise Error("TypeError: CausalLM execution cannot substitute builtin callbacks")
            py_setitem(result, names[i], item)
    return result


def _causal_model_capture_full(mut mods: NativeModules, model: PythonObject, context: PythonObject) raises -> PythonObject:
    var names = _causal_model_names()
    var live = model_record_fields(mods, model, context["model_class"], names)
    var bi = mods.module("builtins")
    var fields = Python.dict()
    for i in range(len(names)):  # driver-loop: over retain only fields consumed by this operation
        if py_contains(live, names[i]):
            py_setitem(fields, names[i], live[names[i]])
    py_setitem(fields, "_causal_original", model)
    py_setitem(fields, "_causal_live_fields", live)
    for name in ["kind", "d_model", "vocab_size", "n_layers", "norm_eps", "numeric_profile", "device", "weight_format", "max_positions", "_native_vendor", "_native_numeric_mode", "_parameters_exposed", "_resident_busy", "_closed", "_failed"]:  # driver-loop: over captured model scalar metadata
        if py_contains(fields, name):
            neural_plain_value(mods, fields[name], 0, context)
            py_setitem(fields, name, _causal_native_plain(mods, fields[name], context))
    py_setitem(fields, "plan", _causal_namespace(mods, model_plan_snapshot(mods, fields["plan"])))
    py_setitem(fields, "_runtime", context["runtime"])
    py_setitem(fields, "_causal_entries_identity", fields["_native_entries"])
    var entries = _causal_entries(mods, fields["_native_entries"], py_get(fields, "_causal_entries_owner"), context)
    py_setitem(fields, "_native_entries", entries)
    var buffers = Python.dict()
    var original_parameters = Python.dict()
    for name in ["_embed", "_norm", "_head"]:  # small-loop(retained model head buffers, preserving tied-weight identity)
        var original = fields[name]
        py_setitem(original_parameters, name, original)
        var identity = py_call(bi.id, original)
        var captured = py_get(buffers, identity)
        if captured is None and original is not None:
            captured = _causal_buffer(original, context)
            py_setitem(buffers, identity, captured)
        py_setitem(fields, name, captured)
    py_setitem(fields, "_causal_original_parameters", original_parameters)
    var packed = py_get(fields, "_head_int15")
    if packed is not None:
        if (py_type(packed) is not bi.list and py_type(packed) is not bi.tuple) or len(packed) != 3:
            raise Error("TypeError: CausalLM packed head requires three native planes")
        var copied = Python.list()
        for i in range(3):  # small-loop(retained int15 head planes)
            _ = py_call(copied.append, _causal_buffer(packed[i], context))
        py_setitem(fields, "_head_int15", copied)
    var blocks = fields["_blocks"]
    if py_type(blocks) is not bi.list and py_type(blocks) is not bi.tuple:
        raise Error("TypeError: CausalLM blocks require an ordinary owner sequence")
    py_setitem(fields, "_blocks", py_call(bi.list, blocks))
    var parallel = py_get(entries, "parallel_call")
    if parallel is not None:
        var routing = py_call(parallel, "capture_model", Python.list(model, fields))
        mods.released()
        neural_string_keys(mods, routing)
        _ = py_call(fields.update, routing)
    else:
        var captured = Python.list()
        var seen = Dict[Int, PythonObject]()
        for i in range(len(blocks)):  # driver-loop: over capture unique block owners before execution
            var identity = Int(blocks[i]._obj_ptr)
            var block = seen.get(identity, PythonObject(None))
            if block is None:
                block = neural_block_snapshot(mods, blocks[i], context)
                for name in ["d_model", "d_inner", "nheads", "conv_dim", "d_in_proj", "dt_rank", "n_heads", "n_kv_heads", "head_dim", "intermediate", "window", "max_positions", "_opts_tail", "_extended", "dt_limit", "numeric_profile", "_arithmetic_profile", "_arithmetic_profile_changed"]:  # small-loop(normalize captured block launch words)
                    if py_contains(block, name):
                        py_setitem(block, name, _causal_native_plain(mods, block[name], context))
                seen[identity] = block
            _ = py_call(captured.append, block)
        py_setitem(fields, "_native_block_fields", captured)
    var resident = py_get(fields, "_resident_sessions")
    if resident is not None:
        if (py_type(resident) is not bi.tuple and py_type(resident) is not bi.list) or len(resident) != 3:
            raise Error("TypeError: CausalLM resident cache requires its native owner record")
        neural_plain_value(mods, resident[0], 0, context)
        if py_type(resident[2]) is not bi.list and py_type(resident[2]) is not bi.tuple:
            raise Error("TypeError: CausalLM resident sessions require native owner slots")
        py_setitem(fields, "_resident_sessions", Python.tuple(neural_metadata_snapshot(mods, resident[0]), resident[1], py_call(bi.list, resident[2])))
    return _causal_namespace(mods, fields)


comptime _CAUSAL_MODEL_SLOT = "causal_model_capture_v1"


def _causal_model_names() raises -> PythonObject:
    var names = CAUSAL_FIELD_NAMES.get_or_create_ptr()
    if names[].model is None:
        names[].model = Python.tuple("plan", "config", "kind", "d_model", "vocab_size", "n_layers", "norm_eps",
        "numeric_profile", "device", "weight_format", "max_positions", "_embed", "_norm", "_head", "_blocks",
        "_native_entries", "_native_vendor", "_native_numeric_mode", "_head_int15", "_runtime",
        "_parameters_exposed", "_resident_busy", "_resident_sessions", "_closed", "_failed", "layer_devices", "_worker_index", "_pool", "_pools", "_causal_entries_owner")
    return names[].model


def _causal_model_runtime_names() -> List[String]:
    # Model fields a call itself updates (lifecycle words, resident sessions):
    # read and normalized from the live model on every call, as before.
    return [String("_parameters_exposed"), String("_resident_busy"), String("_resident_sessions"),
            String("_closed"), String("_failed")]


def _causal_block_runtime_names() -> List[String]:
    # Block fields an operation writes back (sessions, provider caches,
    # arithmetic profile words, reports): taken live on every call.
    return [String("_arithmetic_profile"), String("_arithmetic_profile_mode"), String("_arithmetic_profile_changed"),
            String("_native_session"), String("_session_binding"), String("_native_forward_owner"), String("h_last_"),
            String("_native_block_provider_cache"), String("_native_mamba_block_provider_cache"), String("_native_mamba_state"),
            String("_native_session_provider_cache"), String("_native_primitives_provider_cache"),
            String("_native_mamba_provider_cache")]


def _causal_model_key(mut mods: NativeModules, model: PythonObject, live: PythonObject, context: PythonObject) raises -> PythonObject:
    # Everything the full capture reads: the model's class and stored
    # fields (weights, plan, entries, block list) and each block's class and
    # stored fields. Runtime-owned words are excluded (read live per call).
    var bi = mods.module("builtins")
    var key = NativeIdentityKey(mods)
    key.add(py_type(model))
    key.add_identity(context)
    key.fields(live, _causal_model_runtime_names())
    var blocks = py_get(live, "_blocks")
    if blocks is None or (py_type(blocks) is not bi.list and py_type(blocks) is not bi.tuple):
        return PythonObject(None)
    for i in range(len(blocks)):  # driver-loop: over each model layer's identity and setup inputs once
        key.add(py_type(blocks[i]))
        key.fields(_fields(mods, blocks[i]), _causal_block_runtime_names())
    key.watch_arrays(mods, context)
    # A self-referential field must not make the cache retain its weak key.
    if key.retains(model):
        return PythonObject(None)
    return PythonObject(alloc=key^)


def _causal_model_matches(mut mods: NativeModules, saved: PythonObject, model: PythonObject, live: PythonObject, context: PythonObject) raises -> Bool:
    # Walk live storage against retained native identities. No Python key
    # list, dictionary items list, or integer object is built on a cache hit.
    var key = saved.downcast_value_ptr[NativeIdentityKey]()
    var offset = 0
    if not key[].matches(py_type(model), offset) or not key[].matches_identity(context, offset):
        return False
    if not key[].matches_fields(live, _causal_model_runtime_names(), offset):
        return False
    var bi = mods.module("builtins")
    var blocks = py_get(live, "_blocks")
    if blocks is None or (py_type(blocks) is not bi.list and py_type(blocks) is not bi.tuple):
        return False
    for i in range(len(blocks)):  # driver-loop: over model layer setup inputs compared in place
        if not key[].matches(py_type(blocks[i]), offset):
            return False
        if not key[].matches_fields(_fields(mods, blocks[i]), _causal_block_runtime_names(), offset):
            return False
    return key[].complete(offset) and key[].arrays_match(mods, context)


def _causal_view_retained(mut mods: NativeModules, view: PythonObject, original: PythonObject, context: PythonObject) raises -> Bool:
    # True when `view` is a passive view of the Array `original` (same
    # storage), never a converted copy or a foreign exporter.
    if view is None or original is None:
        return view is None and original is None
    return py_type(original) is context["array_class"] and view._base is original


def _causal_block_retained(mut mods: NativeModules, snap: PythonObject, live: PythonObject, context: PythonObject) raises -> Bool:
    for name in [String("_w"), String("_wopt")]:  # small-loop(weight slot lists)
        if py_contains(snap, name):
            var views = snap[name]
            var originals = snap[String("_neural_original") + name]
            if len(views) != len(originals):
                return False
            for i in range(len(views)):  # driver-loop: over the block's weight slots, identity only
                if not _causal_view_retained(mods, views[i], originals[i], context):
                    return False
    var packed = py_get(live, "_int15")
    if packed is not None:
        var planes = snap["_int15"]
        for i in range(len(packed)):  # driver-loop: over packed weight planes, identity only
            if packed[i] is None or planes[i] is None:
                if packed[i] is not planes[i]:
                    return False
                continue
            for j in range(len(packed[i])):  # small-loop(packed: weight plane slots): identity only, no hooks
                if not _causal_view_retained(mods, planes[i][j], packed[i][j], context):
                    return False
    return True


def _causal_runtime_word(mut mods: NativeModules, fields: PythonObject, live: PythonObject, name: String, context: PythonObject) raises:
    # The full capture's per-name copy and normalization of one runtime word.
    if py_contains(live, name):
        py_setitem(fields, name, live[name])
        if name != "_resident_sessions":
            neural_plain_value(mods, fields[name], 0, context)
            py_setitem(fields, name, _causal_native_plain(mods, fields[name], context))
    else:
        _ = py_call(fields.pop, name, PythonObject(None))


def causal_model_capture(mut mods: NativeModules, model: PythonObject, context: PythonObject) raises -> PythonObject:
    # Prepared model capture: the full capture (_causal_model_capture_full:
    # every block snapshot, head buffers, plan, entries) runs on first use and
    # whenever anything it read changed; otherwise its result is reused and
    # only runtime words, entries and block fields are taken live. Record
    # admission runs first on every call, so a refusal keeps its type,
    # message and position.
    var bi = mods.module("builtins")
    var live = model_record_fields(mods, model, context["model_class"], _causal_model_names())
    var record = prepared_get(mods, _CAUSAL_MODEL_SLOT, model)
    var matches = False
    if record is not None:
        try:
            matches = _causal_model_matches(mods, record[0], model, live, context)
        except:
            matches = False
    if matches:
        var fields = py_call(bi.dict, record[1])
        py_setitem(fields, "_causal_original", model)
        py_setitem(fields, "_causal_live_fields", live)
        # The private original-parameter mapping is read-only after capture;
        # the shallow template copy already retains it without another dict.
        var runtime_names = _causal_model_runtime_names()
        for name in runtime_names:  # small-loop(runtime-owned model words)
            _causal_runtime_word(mods, fields, live, name, context)
        py_setitem(fields, "_causal_entries_identity", live["_native_entries"])
        py_setitem(fields, "_native_entries", _causal_entries(mods, live["_native_entries"], py_get(live, "_causal_entries_owner"), context))
        var blocks = live["_blocks"]
        py_setitem(fields, "_blocks", py_call(bi.list, blocks))
        var derived = record[2]
        var captured = Python.list()
        var seen = Dict[Int, PythonObject]()
        for i in range(len(blocks)):  # driver-loop: over unique block owners, admitted per call
            var identity = Int(blocks[i]._obj_ptr)
            var block = seen.get(identity, PythonObject(None))
            if block is None:
                var block_live = neural_block_fields(mods, blocks[i], context)
                block = py_call(bi.dict, block_live)
                _ = py_call(block.update, derived[i])
                py_setitem(block, "_neural_live_fields", block_live)
                for name in ["_arithmetic_profile", "_arithmetic_profile_changed"]:  # small-loop(runtime launch words normalized per call)
                    if py_contains(block, name):
                        py_setitem(block, name, _causal_native_plain(mods, block[name], context))
                seen[identity] = block
            _ = py_call(captured.append, block)
        py_setitem(fields, "_native_block_fields", captured)
        var resident = py_get(fields, "_resident_sessions")
        if resident is not None:
            if (py_type(resident) is not bi.tuple and py_type(resident) is not bi.list) or len(resident) != 3:
                raise Error("TypeError: CausalLM resident cache requires its native owner record")
            neural_plain_value(mods, resident[0], 0, context)
            if py_type(resident[2]) is not bi.list and py_type(resident[2]) is not bi.tuple:
                raise Error("TypeError: CausalLM resident sessions require native owner slots")
            py_setitem(fields, "_resident_sessions", Python.tuple(neural_metadata_snapshot(mods, resident[0]), resident[1], py_call(bi.list, resident[2])))
        return _causal_namespace(mods, fields)
    var key = PythonObject(None)
    try:
        key = _causal_model_key(mods, model, live, context)
    except:
        key = PythonObject(None)
    var result = _causal_model_capture_full(mods, model, context)
    if key is None:
        return result
    var fields = causal_snapshot_fields(mods, result)
    var entries = fields["_native_entries"]
    if py_get(entries, "parallel_call") is not None:
        return result
    for name in ["_embed", "_norm", "_head"]:  # small-loop(head buffers)
        if not _causal_view_retained(mods, fields[name], fields["_causal_original_parameters"][name], context):
            return result
    var packed = py_get(live, "_head_int15")
    if packed is not None:
        for i in range(3):  # small-loop(int15 head planes)
            if not _causal_view_retained(mods, fields["_head_int15"][i], packed[i], context):
                return result
    var derived = Python.list()
    var blocks = live["_blocks"]
    var captured = fields["_native_block_fields"]
    for i in range(len(blocks)):  # driver-loop: over unique block owners, setup-derived words
        var snap = captured[i]
        var block_live = snap["_neural_live_fields"]
        if not _causal_block_retained(mods, snap, block_live, context):
            return result
        var words = bi.dict()
        var items = py_call(bi.list, py_call(snap.items))
        for j in range(len(items)):  # driver-loop: over the captured block words, identity only
            var name = items[j][0]
            if String(name) == "_neural_live_fields" or String(name) == "_arithmetic_profile" or String(name) == "_arithmetic_profile_changed":
                continue
            if not py_contains(block_live, name) or block_live[name] is not items[j][1]:
                py_setitem(words, name, items[j][1])
        _ = py_call(derived.append, words)
    var unchanged = False
    try:
        unchanged = _causal_model_matches(mods, key, model, live, context)
    except:
        unchanged = False
    if not unchanged:
        prepared_drop(mods, _CAUSAL_MODEL_SLOT, model)
        return result
    var template = py_call(bi.dict, fields)
    for name in ["_native_block_fields", "_causal_live_fields", "_causal_original"]:  # small-loop(per-call capture slots)
        _ = py_call(template.pop, name, PythonObject(None))
    prepared_put(mods, _CAUSAL_MODEL_SLOT, model, Python.list(key, template, derived))
    return result


def causal_model_publish(mut mods: NativeModules, model: PythonObject) raises:
    if model is None or py_type(model) is not mods.module("types").SimpleNamespace:
        return
    var fields = causal_snapshot_fields(mods, model)
    var live = py_get(fields, "_causal_live_fields")
    if live is None:
        return
    neural_string_keys(mods, live)
    # Lifecycle transitions already publish through causal_model_cache_set at
    # the operation that owns them, before any GIL release. A captured value is
    # not a pending write: another read may have cleared busy, replaced its
    # sessions, or exposed parameters since this snapshot was admitted. Replaying
    # the snapshot here restored busy=True after overlapping private reads.
    # Preserve live-field admission, but never replay another operation's state.
    var blocks = py_get(fields, "_native_block_fields")
    if blocks is not None:
        for i in range(len(blocks)):  # driver-loop: validate retained live block owners without replay
            var block = blocks[i]
            var original = block["_neural_live_fields"]
            neural_string_keys(mods, original)
            # The causal forward writes these lifecycle/report fields through
            # causal_block_cache_set. Replaying an untouched block snapshot can
            # likewise replace a newer owner or report from another operation.


comptime _CAUSAL_LAYER_SLOT = "causal_layer_buffers_v1"


def causal_layer_capture(mut mods: NativeModules, state: PythonObject, kind: PythonObject, context: PythonObject) raises -> PythonObject:
    var names = CAUSAL_FIELD_NAMES.get_or_create_ptr()
    if names[].layer_scalars is None:
        var scalars = Python.tuple("batch_size", "n_kv_heads", "head_dim", "max_tokens", "window", "cached_tokens",
        "arithmetic_profile", "_numeric_profile", "profile_id", "absolute_position", "_nn34_usable", "buffered_tokens")
        names[].layer_buffers = Python.tuple("k_cache", "v_cache", "conv_window", "h", "affine_boundary", "affine_a", "affine_b", "buffer_xbc", "buffer_dtraw")
        names[].resident = Python.tuple("_resident_session")
        names[].layer_all = scalars + names[].layer_buffers + names[].resident
        names[].layer_scalars = scalars
    var scalar_names = names[].layer_scalars
    var buffers = names[].layer_buffers
    # Admit the union once: repeating the same class/namespace walk for
    # scalar, buffer and session groups adds no protection between reads.
    var live = model_record_fields(mods, state, context["generation_classes"][2][kind], names[].layer_all)
    var bi = mods.module("builtins")
    var fields = Python.dict()
    for i in range(len(scalar_names)):  # driver-loop: over retain declared layer state words
        if py_contains(live, scalar_names[i]):
            py_setitem(fields, scalar_names[i], live[scalar_names[i]])
    for i in range(len(buffers)):  # driver-loop: over retain declared layer state buffers
        if py_contains(live, buffers[i]):
            py_setitem(fields, buffers[i], live[buffers[i]])
    if py_contains(live, "_resident_session"):
        py_setitem(fields, "_resident_session", live["_resident_session"])
    for i in range(len(scalar_names)):  # driver-loop: over capture layer state scalars before conversion
        if py_contains(fields, scalar_names[i]):
            neural_plain_value(mods, fields[scalar_names[i]], 0, context)
            py_setitem(fields, scalar_names[i], _causal_native_plain(mods, fields[scalar_names[i]], context))
    var original_buffers = Python.dict()
    # Prepared state views: an Array buffer still stored under the same name
    # reuses the passive view captured for it on an earlier call (same
    # storage); a new or replaced buffer, or a foreign exporter, is captured
    # as before.
    var record = prepared_get(mods, _CAUSAL_LAYER_SLOT, state)
    var prepared = PythonObject(None)
    if record is not None and len(record) == 2:
        try:
            if record[1].downcast_value_ptr[NativeIdentityKey]()[].arrays_match(mods, context):
                prepared = record[0]
        except:
            prepared = PythonObject(None)
    # A single Array class/slot stamp covers every buffer in this state.
    # Check original identities before taking views, so replacement triggers
    # a fresh combined stamp rather than nine repeated class-namespace scans.
    if prepared is not None:
        for i in range(len(buffers)):  # driver-loop: over current state buffer identities
            var value = py_get(fields, buffers[i])
            var hit = py_get(prepared, buffers[i])
            if value is not None and py_type(value) is context["array_class"]:
                if hit is None or hit[0] is not value:
                    prepared = PythonObject(None)
                    break
            elif hit is not None:
                prepared = PythonObject(None)
                break
    var stamp = NativeIdentityKey(mods)
    var stamped = False
    if prepared is None:
        try:
            for i in range(len(buffers)):  # driver-loop: over state buffers retained in the native metadata stamp
                var value = py_get(fields, buffers[i])
                if value is not None and py_type(value) is context["array_class"]:
                    stamp.add_identity(value)
            stamp.watch_arrays(mods, context)
            stamped = True
        except:
            stamped = False
    var views = bi.dict() if prepared is None else prepared
    for i in range(len(buffers)):  # driver-loop: over retain each mutable state buffer exporter
        if py_contains(fields, buffers[i]):
            var value = fields[buffers[i]]
            var captured = PythonObject(None)
            if value is not None and py_type(value) is context["array_class"]:
                var hit = PythonObject(None) if prepared is None else py_get(prepared, buffers[i])
                if hit is not None:
                    captured = hit[1]
                else:
                    captured = _causal_buffer(value, context)
                    py_setitem(views, buffers[i], Python.list(value, captured))
            else:
                captured = _causal_buffer(value, context)
            py_setitem(original_buffers, buffers[i], captured)
            py_setitem(fields, buffers[i], captured)
    if prepared is None and stamped:
        var unchanged = False
        try:
            unchanged = stamp.arrays_match(mods, context)
        except:
            unchanged = False
        if unchanged and not stamp.retains(state):
            prepared_put(mods, _CAUSAL_LAYER_SLOT, state, Python.list(views, PythonObject(alloc=stamp^)))
    py_setitem(fields, "_causal_original", state)
    py_setitem(fields, "_native_state_identity", state)
    py_setitem(fields, "_causal_live_fields", live)
    py_setitem(fields, "_causal_initial_buffers", original_buffers)
    return _causal_namespace(mods, fields)


def causal_state_capture(mut mods: NativeModules, state: PythonObject, model: PythonObject, context: PythonObject) raises -> PythonObject:
    if state is None:
        return state
    var live = model_record_fields(mods, state, context["generation_classes"][1],
        Python.tuple("batch_size", "max_tokens", "positions", "layers", "_owner", "_usable"))
    var bi = mods.module("builtins")
    var fields = Python.dict()
    for name in ["batch_size", "max_tokens", "positions", "_usable", "_owner", "layers"]:  # small-loop(retain only outer-state schema fields)
        if py_contains(live, name):
            py_setitem(fields, name, live[name])
    for name in ["batch_size", "max_tokens", "positions", "_usable"]:  # small-loop(capture outer decode-state words)
        if py_contains(fields, name):
            neural_plain_value(mods, fields[name], 0, context)
            py_setitem(fields, name, _causal_native_plain(mods, fields[name], context))
    if fields["_owner"] is not causal_model_origin(mods, model):
        raise Error("CausalLM: state belongs to another model")
    var layers = fields["layers"]
    if py_type(layers) is not bi.list and py_type(layers) is not bi.tuple:
        raise Error("TypeError: CausalLM state layers require an ordinary sequence")
    var parallel = py_get(model._native_entries, "parallel_call")
    var captured = Python.list()
    var seen = Dict[Int, PythonObject]()
    for i in range(len(layers)):  # driver-loop: over capture each unique state identity once
        var identity = Int(layers[i]._obj_ptr)
        var layer = seen.get(identity, PythonObject(None))
        if layer is None:
            try:
                if parallel is not None:
                    layer = py_call(parallel, "capture_state", Python.list(layers[i], model))
                    mods.released()
                else:
                    layer = causal_layer_capture(mods, layers[i], model.kind, context)
            except problem:
                mods.released()
                # Retain the original execution position of invalid layer
                # state. Native execution raises here only when it reaches
                # this layer, preserving earlier partial progress.
                var failed = Python.dict()
                py_setitem(failed, "_causal_original", layers[i])
                py_setitem(failed, "_native_state_identity", layers[i])
                py_setitem(failed, "_causal_admission_failure", PythonObject(String(problem)))
                py_setitem(failed, "_causal_live_fields", PythonObject(None))
                layer = _causal_namespace(mods, failed)
            seen[identity] = layer
        _ = py_call(captured.append, layer)
    py_setitem(fields, "layers", captured)
    py_setitem(fields, "_causal_initial_layers", captured)
    py_setitem(fields, "_causal_original", state)
    py_setitem(fields, "_native_state_identity", state)
    py_setitem(fields, "_causal_live_fields", live)
    return _causal_namespace(mods, fields)


def causal_layer_publish(mut mods: NativeModules, value: PythonObject) raises:
    if py_type(value) is not mods.module("types").SimpleNamespace:
        return
    var layer = causal_snapshot_fields(mods, value)
    var target = layer["_causal_live_fields"]
    if target is None:
        return
    neural_string_keys(mods, target)
    var initial = py_get(layer, "_causal_initial_buffers", Python.dict())
    for name in ["cached_tokens", "arithmetic_profile", "_numeric_profile", "profile_id", "absolute_position", "_nn34_usable", "buffered_tokens", "_resident_session", "k_cache", "v_cache", "conv_window", "h", "affine_boundary", "affine_a", "affine_b", "buffer_xbc", "buffer_dtraw", "token", "_lifetime"]:  # driver-loop: over consumed state fields only
        if not py_contains(layer, name):
            continue
        if py_contains(initial, name) and layer[name] is initial[name]:
            continue
        py_setitem(target, name, layer[name])


def causal_state_publish(mut mods: NativeModules, state: PythonObject) raises:
    if state is None or py_type(state) is not mods.module("types").SimpleNamespace:
        return
    var fields = causal_snapshot_fields(mods, state)
    var live = fields["_causal_live_fields"]
    neural_string_keys(mods, live)
    for name in ["positions", "_usable"]:  # small-loop(success/failure publication)
        if py_contains(fields, name):
            py_setitem(live, name, fields[name])
    var layers = fields["layers"]
    var bi = mods.module("builtins")
    if layers is not fields["_causal_initial_layers"]:
        # reset_state installed freshly allocated state owners.
        var published = Python.list()
        for i in range(len(layers)):  # driver-loop: over reset publishes fresh public state owners
            causal_layer_publish(mods, layers[i])
            _ = py_call(published.append, causal_state_origin(mods, layers[i]))
        py_setitem(live, "layers", published)
        return
    var seen = Dict[Int, Bool]()
    for i in range(len(layers)):  # driver-loop: over publish one result per original aliased state
        var layer = causal_snapshot_fields(mods, layers[i])
        var identity = Int(layer["_causal_original"]._obj_ptr)
        if identity in seen:
            continue
        seen[identity] = True
        causal_layer_publish(mods, layers[i])


def causal_model_cache_set(mut mods: NativeModules, model: PythonObject, fields: PythonObject, name: String, value: PythonObject) raises:
    # Busy and close/install transitions must be visible before GIL release,
    # just as they were when execution mutated the public dictionary directly.
    py_setitem(fields, name, value)
    var live = py_get(fields, "_causal_live_fields")
    if live is not None:
        neural_string_keys(mods, live)
        py_setitem(live, name, value)


def causal_block_cache_set(mut mods: NativeModules, fields: PythonObject, name: String, value: PythonObject) raises:
    py_setitem(fields, name, value)
    var live = py_get(fields, "_neural_live_fields")
    if live is not None:
        neural_string_keys(mods, live)
        py_setitem(live, name, value)


def causal_checkpoint_capture(mut mods: NativeModules, checkpoint: PythonObject, context: PythonObject) raises -> PythonObject:
    neural_string_keys(mods, checkpoint)
    var bi = mods.module("builtins")
    var result = Python.dict()
    var keys = py_call(bi.list, checkpoint.keys())
    for i in range(len(keys)):  # driver-loop: over capture public checkpoint fields before checksum or restoration
        var name = String(keys[i])
        var value = checkpoint[keys[i]]
        if name == "prompt" or name == "current":
            py_setitem(result, keys[i], _causal_buffer(value, context))
        elif name == "pending_bytes":
            if py_type(value) is not bi.bytes and py_type(value) is not bi.bytearray:
                raise Error("TypeError: generation checkpoint pending bytes require native byte storage")
            py_setitem(result, keys[i], py_call(bi.bytes, value))
        elif name == "layers":
            if py_type(value) is not bi.list and py_type(value) is not bi.tuple:
                raise Error("TypeError: generation checkpoint layers require native records")
            var layers = Python.list()
            for j in range(len(value)):  # driver-loop: over capture every saved layer record
                neural_string_keys(mods, value[j])
                var fields = Python.dict()
                var names = py_call(bi.list, value[j].keys())
                for k in range(len(names)):  # driver-loop: over captured checkpoint scalars and state bytes
                    var item = value[j][names[k]]
                    if py_type(item) is context["array_class"]:
                        item = generation_array_snapshot(item)
                    else:
                        neural_plain_value(mods, item, 0, context)
                        item = neural_metadata_snapshot(mods, item)
                    py_setitem(fields, names[k], item)
                _ = py_call(layers.append, fields)
            py_setitem(result, keys[i], layers)
        else:
            neural_plain_value(mods, value, 0, context)
            py_setitem(result, keys[i], neural_metadata_snapshot(mods, value))
    return result
