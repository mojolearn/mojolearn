# SPDX-License-Identifier: Apache-2.0
# Stable whole-operation model/state metadata. Original records are retained
# only for identity and native dictionary publication, never runtime attributes.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.causal_model_config import model_record_fields, model_plan_snapshot
from bindings.neural_block_admission import neural_plain_value, neural_string_keys, neural_metadata_snapshot, neural_block_snapshot, neural_native_scalar
from bindings.generation_buffers import generation_array_snapshot, generation_buffer_view


def causal_snapshot_fields(value: PythonObject) raises -> PythonObject:
    if py_type(value) is not native_loaded_module("types").SimpleNamespace:
        raise Error("TypeError: causal operation metadata requires its native namespace")
    ref cpy = Python().cpython()
    var raw = cpy.lib.get_function[PyObjectPtr]("PyObject_GenericGetDict")(value._obj_ptr, PyObjectPtr())
    if not raw:
        raise cpy.unsafe_get_error()
    return PythonObject(from_owned=raw)


def _causal_namespace(fields: PythonObject) raises -> PythonObject:
    var value = py_call(native_loaded_module("types").SimpleNamespace)
    _ = py_call(causal_snapshot_fields(value).update, fields)
    return value


def _causal_native_plain(value: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = native_loaded_module("builtins")
    var cls = py_type(value)
    if value is None or cls is bi.str or cls is bi.int or cls is bi.float or cls is bi.bool:
        return value
    if cls is bi.list or cls is bi.tuple:
        var result = Python.list()
        for i in range(len(value)):  # driver-loop: over already admitted scalar metadata containers
            _ = py_call(result.append, _causal_native_plain(value[i], context))
        return py_call(bi.tuple, result) if cls is bi.tuple else result
    if cls is bi.dict:
        neural_string_keys(value)
        var result = Python.dict()
        var keys = py_call(bi.list, value.keys())
        for i in range(len(keys)):  # driver-loop: over captured passive metadata dictionary fields
            py_setitem(result, keys[i], _causal_native_plain(value[keys[i]], context))
        return result
    if neural_native_scalar(value, context, True):
        return py_call(bi.int, value)
    if neural_native_scalar(value, context):
        return py_call(bi.float, value)
    raise Error("TypeError: causal metadata requires native scalar/container values")


def _causal_buffer(value: PythonObject, context: PythonObject) raises -> PythonObject:
    if value is None:
        return value
    if py_type(value) is context["array_class"]:
        return generation_array_snapshot(value)
    return generation_buffer_view(value)


def causal_model_origin(model: PythonObject) raises -> PythonObject:
    if py_type(model) is native_loaded_module("types").SimpleNamespace:
        return py_get(causal_snapshot_fields(model), "_causal_original", model)
    return model


def causal_state_origin(state: PythonObject) raises -> PythonObject:
    if py_type(state) is native_loaded_module("types").SimpleNamespace:
        return py_get(causal_snapshot_fields(state), "_causal_original", state)
    return state


def causal_state_fields(state: PythonObject, context: PythonObject) raises -> PythonObject:
    if py_type(state) is native_loaded_module("types").SimpleNamespace:
        var fields = causal_snapshot_fields(state)
        if not py_contains(fields, "_causal_original"):
            raise Error("TypeError: causal state requires its retained native owner")
        return fields
    return model_record_fields(state, context["generation_classes"][1],
        Python.tuple("batch_size", "max_tokens", "positions", "layers", "_owner", "_usable"))


struct CausalExecutionEntries(Movable, Writable):
    var entries: PythonObject

    def __init__(out self, entries: PythonObject) raises:
        self.entries = py_call(native_loaded_module("builtins").dict, entries)

    def write_to(self, mut writer: Some[Writer]):
        writer.write("CausalExecutionEntries(native)")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def causal_pin_entries(fields: PythonObject, entries: PythonObject) raises:
    neural_string_keys(fields)
    neural_string_keys(entries)
    py_setitem(fields, "_causal_entries_owner", PythonObject(alloc=CausalExecutionEntries(entries)))


def _causal_entries(entries: PythonObject, pinned: PythonObject, context: PythonObject) raises -> PythonObject:
    neural_string_keys(entries)
    var bi = native_loaded_module("builtins")
    var types = native_loaded_module("types")
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
        neural_string_keys(connected)
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
                or name == "block_binding" or name == "native_session_provider"):
            # Typed provider/lifetime admission occurs at its native boundary;
            # these values are retained, never called or traversed as Python.
            py_setitem(result, names[i], item)
        elif (name == "cpu" or name == "weight_names" or name == "resident_vendor"
                or name == "mamba1_profile" or name == "native_enabled"):
            neural_plain_value(item, 0, context)
            py_setitem(result, names[i], neural_metadata_snapshot(item))
        else:
            if py_type(item) is not py_type(bi.len) or py_type(item.__self__) is not types.ModuleType:
                raise Error("TypeError: CausalLM execution entries must remain native compiled functions")
            if item.__self__ is bi:
                raise Error("TypeError: CausalLM execution cannot substitute builtin callbacks")
            py_setitem(result, names[i], item)
    return result


def causal_model_capture(model: PythonObject, context: PythonObject) raises -> PythonObject:
    var names = Python.tuple("plan", "config", "kind", "d_model", "vocab_size", "n_layers", "norm_eps",
        "numeric_profile", "device", "weight_format", "max_positions", "_embed", "_norm", "_head", "_blocks",
        "_native_entries", "_native_vendor", "_native_numeric_mode", "_head_int15", "_runtime",
        "_parameters_exposed", "_resident_busy", "_resident_sessions", "_closed", "_failed", "layer_devices", "_worker_index", "_pool", "_pools", "_causal_entries_owner")
    var live = model_record_fields(model, context["model_class"], names)
    var bi = native_loaded_module("builtins")
    var fields = Python.dict()
    for i in range(len(names)):  # driver-loop: over retain only fields consumed by this operation
        if py_contains(live, names[i]):
            py_setitem(fields, names[i], live[names[i]])
    py_setitem(fields, "_causal_original", model)
    py_setitem(fields, "_causal_live_fields", live)
    for name in ["kind", "d_model", "vocab_size", "n_layers", "norm_eps", "numeric_profile", "device", "weight_format", "max_positions", "_native_vendor", "_native_numeric_mode", "_parameters_exposed", "_resident_busy", "_closed", "_failed"]:  # driver-loop: over captured model scalar metadata
        if py_contains(fields, name):
            neural_plain_value(fields[name], 0, context)
            py_setitem(fields, name, _causal_native_plain(fields[name], context))
    py_setitem(fields, "plan", _causal_namespace(model_plan_snapshot(fields["plan"])))
    py_setitem(fields, "_runtime", context["runtime"])
    py_setitem(fields, "_causal_entries_identity", fields["_native_entries"])
    var entries = _causal_entries(fields["_native_entries"], py_get(fields, "_causal_entries_owner"), context)
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
        neural_string_keys(routing)
        _ = py_call(fields.update, routing)
    else:
        var captured = Python.list()
        var seen = Python.dict()
        for i in range(len(blocks)):  # driver-loop: over capture unique block owners before execution
            var identity = py_call(bi.id, blocks[i])
            var block = py_get(seen, identity)
            if block is None:
                block = neural_block_snapshot(blocks[i], context)
                for name in ["d_model", "d_inner", "nheads", "conv_dim", "d_in_proj", "dt_rank", "n_heads", "n_kv_heads", "head_dim", "intermediate", "window", "max_positions", "_opts_tail", "_extended", "dt_limit", "numeric_profile", "_arithmetic_profile", "_arithmetic_profile_changed"]:  # small-loop(normalize captured block launch words)
                    if py_contains(block, name):
                        py_setitem(block, name, _causal_native_plain(block[name], context))
                py_setitem(seen, identity, block)
            _ = py_call(captured.append, block)
        py_setitem(fields, "_native_block_fields", captured)
    var resident = py_get(fields, "_resident_sessions")
    if resident is not None:
        if (py_type(resident) is not bi.tuple and py_type(resident) is not bi.list) or len(resident) != 3:
            raise Error("TypeError: CausalLM resident cache requires its native owner record")
        neural_plain_value(resident[0], 0, context)
        if py_type(resident[2]) is not bi.list and py_type(resident[2]) is not bi.tuple:
            raise Error("TypeError: CausalLM resident sessions require native owner slots")
        py_setitem(fields, "_resident_sessions", Python.tuple(neural_metadata_snapshot(resident[0]), resident[1], py_call(bi.list, resident[2])))
    return _causal_namespace(fields)


def causal_model_publish(model: PythonObject) raises:
    if model is None or py_type(model) is not native_loaded_module("types").SimpleNamespace:
        return
    var fields = causal_snapshot_fields(model)
    var live = py_get(fields, "_causal_live_fields")
    if live is None:
        return
    neural_string_keys(live)
    for name in ["_parameters_exposed", "_resident_busy", "_resident_sessions"]:  # small-loop(native lifecycle updates only)
        if py_contains(fields, name):
            py_setitem(live, name, fields[name])
    var blocks = py_get(fields, "_native_block_fields")
    if blocks is not None:
        for i in range(len(blocks)):  # driver-loop: over publish admitted block session ownership
            var block = blocks[i]
            var original = block["_neural_live_fields"]
            neural_string_keys(original)
            for name in ["_native_session", "_session_binding", "_native_forward_owner", "h_last_"]:  # small-loop(block session lifecycle and report fields)
                if py_contains(block, name):
                    py_setitem(original, name, block[name])


def causal_layer_capture(state: PythonObject, kind: PythonObject, context: PythonObject) raises -> PythonObject:
    var scalar_names = Python.tuple("batch_size", "n_kv_heads", "head_dim", "max_tokens", "window", "cached_tokens",
        "arithmetic_profile", "_numeric_profile", "profile_id", "absolute_position", "_nn34_usable", "buffered_tokens")
    var buffers = Python.tuple("k_cache", "v_cache", "conv_window", "h", "affine_boundary", "affine_a", "affine_b", "buffer_xbc", "buffer_dtraw")
    var live = model_record_fields(state, context["generation_classes"][2][kind], scalar_names)
    _ = model_record_fields(state, context["generation_classes"][2][kind], buffers)
    _ = model_record_fields(state, context["generation_classes"][2][kind], Python.tuple("_resident_session"))
    var bi = native_loaded_module("builtins")
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
            neural_plain_value(fields[scalar_names[i]], 0, context)
            py_setitem(fields, scalar_names[i], _causal_native_plain(fields[scalar_names[i]], context))
    var original_buffers = Python.dict()
    for i in range(len(buffers)):  # driver-loop: over retain each mutable state buffer exporter
        if py_contains(fields, buffers[i]):
            var captured = _causal_buffer(fields[buffers[i]], context)
            py_setitem(original_buffers, buffers[i], captured)
            py_setitem(fields, buffers[i], captured)
    py_setitem(fields, "_causal_original", state)
    py_setitem(fields, "_native_state_identity", state)
    py_setitem(fields, "_causal_live_fields", live)
    py_setitem(fields, "_causal_initial_buffers", original_buffers)
    return _causal_namespace(fields)


def causal_state_capture(state: PythonObject, model: PythonObject, context: PythonObject) raises -> PythonObject:
    if state is None:
        return state
    var live = model_record_fields(state, context["generation_classes"][1],
        Python.tuple("batch_size", "max_tokens", "positions", "layers", "_owner", "_usable"))
    var bi = native_loaded_module("builtins")
    var fields = Python.dict()
    for name in ["batch_size", "max_tokens", "positions", "_usable", "_owner", "layers"]:  # small-loop(retain only outer-state schema fields)
        if py_contains(live, name):
            py_setitem(fields, name, live[name])
    for name in ["batch_size", "max_tokens", "positions", "_usable"]:  # small-loop(capture outer decode-state words)
        if py_contains(fields, name):
            neural_plain_value(fields[name], 0, context)
            py_setitem(fields, name, _causal_native_plain(fields[name], context))
    if fields["_owner"] is not causal_model_origin(model):
        raise Error("CausalLM: state belongs to another model")
    var layers = fields["layers"]
    if py_type(layers) is not bi.list and py_type(layers) is not bi.tuple:
        raise Error("TypeError: CausalLM state layers require an ordinary sequence")
    var parallel = py_get(model._native_entries, "parallel_call")
    var captured = Python.list()
    var seen = Python.dict()
    for i in range(len(layers)):  # driver-loop: over capture each unique state identity once
        var identity = py_call(bi.id, layers[i])
        var layer = py_get(seen, identity)
        if layer is None:
            try:
                if parallel is not None:
                    layer = py_call(parallel, "capture_state", Python.list(layers[i], model))
                else:
                    layer = causal_layer_capture(layers[i], model.kind, context)
            except problem:
                # Retain the original execution position of invalid layer
                # state. Native execution raises here only when it reaches
                # this layer, preserving earlier partial progress.
                var failed = Python.dict()
                py_setitem(failed, "_causal_original", layers[i])
                py_setitem(failed, "_native_state_identity", layers[i])
                py_setitem(failed, "_causal_admission_failure", PythonObject(String(problem)))
                py_setitem(failed, "_causal_live_fields", PythonObject(None))
                layer = _causal_namespace(failed)
            py_setitem(seen, identity, layer)
        _ = py_call(captured.append, layer)
    py_setitem(fields, "layers", captured)
    py_setitem(fields, "_causal_initial_layers", captured)
    py_setitem(fields, "_causal_original", state)
    py_setitem(fields, "_native_state_identity", state)
    py_setitem(fields, "_causal_live_fields", live)
    return _causal_namespace(fields)


def causal_layer_publish(value: PythonObject) raises:
    if py_type(value) is not native_loaded_module("types").SimpleNamespace:
        return
    var layer = causal_snapshot_fields(value)
    var target = layer["_causal_live_fields"]
    if target is None:
        return
    neural_string_keys(target)
    var initial = py_get(layer, "_causal_initial_buffers", Python.dict())
    for name in ["cached_tokens", "arithmetic_profile", "_numeric_profile", "profile_id", "absolute_position", "_nn34_usable", "buffered_tokens", "_resident_session", "k_cache", "v_cache", "conv_window", "h", "affine_boundary", "affine_a", "affine_b", "buffer_xbc", "buffer_dtraw", "token", "_lifetime"]:  # driver-loop: over consumed state fields only
        if not py_contains(layer, name):
            continue
        if py_contains(initial, name) and layer[name] is initial[name]:
            continue
        py_setitem(target, name, layer[name])


def causal_state_publish(state: PythonObject) raises:
    if state is None or py_type(state) is not native_loaded_module("types").SimpleNamespace:
        return
    var fields = causal_snapshot_fields(state)
    var live = fields["_causal_live_fields"]
    neural_string_keys(live)
    for name in ["positions", "_usable"]:  # small-loop(success/failure publication)
        if py_contains(fields, name):
            py_setitem(live, name, fields[name])
    var layers = fields["layers"]
    var bi = native_loaded_module("builtins")
    if layers is not fields["_causal_initial_layers"]:
        # reset_state installed freshly allocated state owners.
        var published = Python.list()
        for i in range(len(layers)):  # driver-loop: over reset publishes fresh public state owners
            causal_layer_publish(layers[i])
            _ = py_call(published.append, causal_state_origin(layers[i]))
        py_setitem(live, "layers", published)
        return
    var seen = Python.dict()
    for i in range(len(layers)):  # driver-loop: over publish one result per original aliased state
        var layer = causal_snapshot_fields(layers[i])
        var identity = py_call(bi.id, layer["_causal_original"])
        if py_contains(seen, identity):
            continue
        py_setitem(seen, identity, PythonObject(True))
        causal_layer_publish(layers[i])


def causal_model_cache_set(model: PythonObject, fields: PythonObject, name: String, value: PythonObject) raises:
    # Busy and close/install transitions must be visible before GIL release,
    # just as they were when execution mutated the public dictionary directly.
    py_setitem(fields, name, value)
    var live = py_get(fields, "_causal_live_fields")
    if live is not None:
        neural_string_keys(live)
        py_setitem(live, name, value)


def causal_block_cache_set(fields: PythonObject, name: String, value: PythonObject) raises:
    py_setitem(fields, name, value)
    var live = py_get(fields, "_neural_live_fields")
    if live is not None:
        neural_string_keys(live)
        py_setitem(live, name, value)


def causal_checkpoint_capture(checkpoint: PythonObject, context: PythonObject) raises -> PythonObject:
    neural_string_keys(checkpoint)
    var bi = native_loaded_module("builtins")
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
                neural_string_keys(value[j])
                var fields = Python.dict()
                var names = py_call(bi.list, value[j].keys())
                for k in range(len(names)):  # driver-loop: over captured checkpoint scalars and state bytes
                    var item = value[j][names[k]]
                    if py_type(item) is context["array_class"]:
                        item = generation_array_snapshot(item)
                    else:
                        neural_plain_value(item, 0, context)
                        item = neural_metadata_snapshot(item)
                    py_setitem(fields, names[k], item)
                _ = py_call(layers.append, fields)
            py_setitem(result, keys[i], layers)
        else:
            neural_plain_value(value, 0, context)
            py_setitem(result, keys[i], neural_metadata_snapshot(value))
    return result
