# SPDX-License-Identifier: Apache-2.0
"""Native configuration admission for the supported causal model families."""
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_contains, py_get, py_setattr, py_setitem, py_type
from bindings.causal_exception_unpack import causal_borrowed_item, causal_set_exception
from bindings.native_json_door import native_json_read, native_path, native_path_exists
from bindings.native_loaded_module_door import native_loaded_module, NativeModules
from bindings.native_instance_item_pack import native_instance_item
from bindings.mode_metadata_native_door import _fields, _class_fields
from bindings.native_class_admission_door import class_record, class_admit
from core.shared_slot import SharedSlot
from bindings.causal_keys_unpack import causal_keys
from bindings.py_balanced import py_item
from bindings.causal_model_config_run import (
    CausalConfigValue, CausalConfigRequest, CausalConfigPlan, CausalConfigTables, CausalFixedField, CausalLayerEntry,
    causal_config_plan, causal_layer_weights, causal_checkpoint_names,
    CONFIG_NULL, CONFIG_BOOL, CONFIG_INT, CONFIG_FLOAT, CONFIG_STR, CONFIG_OTHER,
)


comptime _RECORD_FIELDS_SLOT = "causal_model_record_fields"


struct ModelConfigConnection(Defaultable, Movable):
    var context: PythonObject
    # The layer tables and byte-LM fixed fields, typed once at connect for RUN.
    var tables: CausalConfigTables

    def __init__(out self):
        self.context = PythonObject(None)
        self.tables = CausalConfigTables()


comptime MODEL_CONFIG_CONNECTION = SharedSlot[ModelConfigConnection, "MojolearnModelConfigConnectionV1"]


def model_string(mut mods: NativeModules, value: PythonObject) raises -> String:
    if py_type(value) is not mods.module("builtins").str:
        raise Error("TypeError: model metadata requires exact strings")
    return String(value)


def model_string(value: PythonObject) raises -> String:
    # Classical and other callers keep their per-call module checks.
    var mods = NativeModules()
    return model_string(mods, value)


def model_integer(mut mods: NativeModules, value: PythonObject) raises -> Int:
    if py_type(value) is not mods.module("builtins").int:
        raise Error("TypeError: model metadata requires exact integers; scalar hooks are unsupported")
    return Int(py=value)


def model_integer(value: PythonObject) raises -> Int:
    # Classical and other callers keep their per-call module checks.
    var mods = NativeModules()
    return model_integer(mods, value)


def model_sequence(mut mods: NativeModules, value: PythonObject) raises:
    var bi = mods.module("builtins")
    if py_type(value) is not bi.list and py_type(value) is not bi.tuple:
        raise Error("TypeError: model metadata requires exact list or tuple storage")


def model_sequence(value: PythonObject) raises:
    # Classical and other callers keep their per-call module checks.
    var mods = NativeModules()
    model_sequence(mods, value)


def model_dict(mut mods: NativeModules, value: PythonObject) raises:
    var bi = mods.module("builtins")
    if py_type(value) is not bi.dict:
        raise Error("TypeError: model metadata requires an exact dictionary")
    # Admission must inspect current keys, including same-size substitutions.
    # Iterate the exact dictionary in place instead of materializing keys on
    # every model capture. PyDict_Next does not invoke hashing/equality hooks.
    ref cpy = Python().cpython()
    var pos = Int(0)
    var key = PyObjectPtr()
    var item = PyObjectPtr()
    var string_type = bi.str
    # Resolved once per call, not per key (main's compiled form, native_identity_key.mojo).
    var dict_next = cpy.lib.get_function[Int32]("PyDict_Next")
    while dict_next(value._obj_ptr, MutPointer(to=pos), MutPointer(to=key), MutPointer(to=item)):  # driver-loop: over current exact dictionary keys
        if py_type(PythonObject(from_borrowed=key)) is not string_type:
            raise Error("TypeError: model metadata requires exact strings")


def model_dict(value: PythonObject) raises:
    # Classical and other callers keep their per-call module checks.
    var mods = NativeModules()
    model_dict(mods, value)


def model_plain(mut mods: NativeModules, value: PythonObject, depth: Int = 0) raises -> PythonObject:
    # Snapshot mutable JSON metadata once; later validation/conversion cannot
    # reach user containers, scalar hooks or descriptor protocols.
    if depth > 128:
        raise Error("TypeError: model metadata nesting exceeds 128")
    var bi = mods.module("builtins")
    var cls = py_type(value)
    if value is None or cls is bi.str or cls is bi.int or cls is bi.float or cls is bi.bool:
        return value
    if cls is bi.list or cls is bi.tuple:
        var result = Python.list()
        for i in range(len(value)):  # driver-loop: capture passive model configuration sequence
            _ = py_call(result.append, model_plain(mods, value[i], depth + 1))
        return py_call(bi.tuple, result) if cls is bi.tuple else result
    model_dict(mods, value)
    var result = Python.dict()
    var keys = py_call(bi.list, value.keys())
    for i in range(len(keys)):  # driver-loop: capture passive model configuration dictionary
        py_setitem(result, keys[i], model_plain(mods, native_instance_item(value, keys[i]), depth + 1))
    return result


def model_plain(value: PythonObject, depth: Int = 0) raises -> PythonObject:
    # Classical and other callers keep their per-call module checks.
    var mods = NativeModules()
    return model_plain(mods, value, depth)


def model_record_fields(mut mods: NativeModules, value: PythonObject, cls: PythonObject, names: PythonObject) raises -> PythonObject:
    var bi = mods.module("builtins")
    if py_type(cls) is not bi.type or py_type(py_type(value)) is not bi.type:
        raise Error("TypeError: model records require ordinary connected class metadata")
    # The ancestry walk below depends only on (type(value), cls, names). Keep
    # its verdict per class under the type-watcher record: any change to the
    # class or an ancestor bumps the epoch and forces the full walk again.
    # The payload holds cls by address only (it is in the key's MRO, so it
    # stays alive while the record can hit) and the names tuple by value.
    var record = class_record(mods, _RECORD_FIELDS_SLOT, py_type(value))
    if record is not None and py_type(names) is bi.tuple and Int(py=record[0]) == Int(cls._obj_ptr) and Bool(py=record[1] == names):
        var cached_fields = _fields(mods, value)
        model_dict(mods, cached_fields)
        return cached_fields
    var nk = causal_keys()
    var ancestry = py_call(py_item(bi.type.__dict__, nk[].k___mro__).__get__, py_type(value))
    var admitted = False
    for i in range(len(ancestry)):  # driver-loop: inspect model record descriptors before storage reads
        admitted = admitted or ancestry[i] is cls
        var row = _class_fields(mods, ancestry[i])
        if py_contains(row, nk[].k___getattr__):
            raise Error("TypeError: model records cannot invoke Python attribute hooks")
        for hook in [String("__getattribute__"), String("__setattr__")]:  # small-loop(native record access hooks)
            if py_contains(row, hook) and row[hook] is not bi.object.__dict__[hook]:
                raise Error("TypeError: model records cannot invoke Python attribute hooks")
        for j in range(len(names)):  # small-loop(fixed consumed model field names)
            if py_contains(row, names[j]):
                raise Error("TypeError: model state fields cannot be replaced by class descriptors")
    if not admitted:
        raise Error("TypeError: model operation requires its connected record class")
    if py_type(names) is bi.tuple:
        class_admit(mods, _RECORD_FIELDS_SLOT, py_type(value), Python.tuple(PythonObject(Int(cls._obj_ptr)), names))
    var fields = _fields(mods, value)
    model_dict(mods, fields)
    return fields


def _model_context() raises -> PythonObject:
    var result = MODEL_CONFIG_CONNECTION.get_or_create_ptr()[].context
    if result is None:
        raise Error("RuntimeError: model configuration requires import-time connection")
    return result


def _config_fields(mut mods: NativeModules, value: PythonObject) raises -> PythonObject:
    var nk = causal_keys()
    return model_record_fields(mods, value, py_item(_model_context(), nk[].k_config_class), Python.tuple("_d", "path"))


def _plan_fields(mut mods: NativeModules, value: PythonObject) raises -> PythonObject:
    var nk = causal_keys()
    return model_record_fields(mods, value, py_item(_model_context(), nk[].k_plan_class),
        Python.tuple("_layer_map", "embed_name", "norm_name", "head_name", "n_layers", "model_type", "kind", "d_model", "vocab_size", "n_heads", "n_kv_heads", "head_dim", "intermediate", "tie_embeddings", "norm_eps", "max_position_embeddings", "block_options", "block_kwargs", "config", "family"))


def model_plan_snapshot(mut mods: NativeModules, value: PythonObject) raises -> PythonObject:
    """Captured plan dictionary for complete native CausalLM construction."""
    var fields = _plan_fields(mods, value)
    var result = Python.dict()
    var names = py_call(mods.module("builtins").list, fields.keys())
    var nk = causal_keys()
    for i in range(len(names)):  # driver-loop: capture every passive plan member before model construction
        var name = model_string(mods, names[i])
        var item = native_instance_item(fields, names[i])
        if name == "config":
            var source = _config_fields(mods, item)
            var captured = py_call(mods.module("builtins").object.__new__, py_item(_model_context(), nk[].k_config_class))
            _ = config_init_binding(mods, captured, native_instance_item(source, nk[].k__d), native_instance_item(source, nk[].k_path))
            py_setitem(result, names[i], captured)
        elif name == "family":
            model_sequence(mods, item)
            if len(item) != 3:
                raise Error("TypeError: model family metadata requires its three-word row")
            _ = model_string(mods, item[1])
            _ = model_string(mods, item[2])
            # Historical first item is an opaque planner function identity.
            # Native consumers dispatch by kind; they never call that item.
            py_setitem(result, names[i], Python.tuple(item[0], item[1], item[2]))
        else:
            py_setitem(result, names[i], model_plain(mods, item))
    return result


def model_config_connect(mut mods: NativeModules, context: PythonObject) raises -> PythonObject:
    model_dict(mods, context)
    var copy = Python.dict()
    var bi = mods.module("builtins")
    for name in [String("config_class"), String("plan_class"), String("unsupported_class")]:  # small-loop(import-connected model types)
        var cls = native_instance_item(context, PythonObject(name))
        if py_type(cls) is not bi.type:
            raise Error("TypeError: model connection requires ordinary classes")
        py_setitem(copy, PythonObject(name), cls)
    py_setitem(copy, PythonObject("defaults"), model_plain(mods, native_instance_item(context, PythonObject("defaults"))))
    # The layer tables and the byte-LM fixed fields are read only by the
    # planner's RUN: typed here once, never kept as Python copies.
    var tables = CausalConfigTables()
    var table_names = [String("llama_layer"), String("qkv_bias"), String("qk_norm"), String("mamba1_layer"), String("mamba2_layer"), String("byte_layer")]
    for t in range(len(table_names)):  # small-loop(the six static layer tables, CONFIG_TABLE_* order)
        tables.tables[t] = causal_layer_map_unpack(mods, model_plain(mods, native_instance_item(context, PythonObject(table_names[t]))))
    var fixed = model_plain(mods, native_instance_item(context, PythonObject("byte_fixed")))
    model_dict(mods, fixed)
    var fixed_names = py_call(bi.list, fixed.keys())
    for i in range(len(fixed_names)):  # small-loop(the fixed byte-LM config fields)
        tables.byte_fixed.append(CausalFixedField(String(fixed_names[i]), _config_value(mods, fixed, fixed_names[i])))
    var nk = causal_keys()
    # Family rows retain their historical function identity as public metadata;
    # planning dispatch below never calls those functions.
    var families = native_instance_item(context, nk[].k_families)
    model_dict(mods, families)
    py_setitem(copy, nk[].k_families, py_call(bi.dict, families))
    py_setitem(copy, nk[].k_byte_row, native_instance_item(context, nk[].k_byte_row))
    var state = MODEL_CONFIG_CONNECTION.get_or_create_ptr()
    if state[].context is not None:
        raise Error("TypeError: model configuration is already connected")
    state[].context = copy
    state[].tables = tables^
    return PythonObject(None)


def _model_error_class(mut mods: NativeModules) raises -> PythonObject:
    var nk = causal_keys()
    var cls = py_item(_model_context(), nk[].k_unsupported_class)
    var bi = mods.module("builtins")
    var bases = py_call(py_item(bi.type.__dict__, nk[].k___bases__).__get__, cls)
    if len(bases) != 1 or bases[0] is not bi.ValueError:
        raise Error("TypeError: UnsupportedModel must retain native ValueError behavior")
    var row = _class_fields(mods, cls)
    for name in ["__new__", "__init__", "__str__", "__repr__", "__getattribute__", "__getattr__", "__setattr__"]:  # small-loop(exception normalization hooks)
        if py_contains(row, name):
            raise Error("TypeError: UnsupportedModel cannot invoke Python exception hooks")
    return cls


def config_init_binding(mut mods: NativeModules, value: PythonObject, data: PythonObject, path: PythonObject) raises -> PythonObject:
    var fields = _config_fields(mods, value)
    model_dict(mods, data)
    var captured = model_plain(mods, data)
    var filename = PythonObject(None) if path is None else native_path(path)
    var nk = causal_keys()
    py_setitem(fields, nk[].k__d, captured)
    py_setitem(fields, nk[].k_path, filename)
    return PythonObject(None)


def config_load_binding(mut mods: NativeModules, cls: PythonObject, input_path: PythonObject) raises -> PythonObject:
    var nk = causal_keys()
    if cls is not py_item(_model_context(), nk[].k_config_class):
        raise Error("TypeError: config factory requires its connected class")
    var path = native_path(input_path)
    if native_path_exists(path, True):
        mods.released()
        path = py_call(path.rstrip, "/") + PythonObject("/config.json")
    mods.released()
    if not native_path_exists(path):
        mods.released()
        raise Error("FileNotFoundError: config.json does not exist")
    mods.released()
    var value = py_call(mods.module("builtins").object.__new__, cls)
    _ = config_init_binding(mods, value, native_json_read(path), path)
    mods.released()
    return value


def config_access_binding(mut mods: NativeModules, config: PythonObject, operation: PythonObject, key: PythonObject, default: PythonObject) raises -> PythonObject:
    var op = model_integer(mods, operation)
    var fields = _config_fields(mods, config)
    var nk = causal_keys()
    var data = model_plain(mods, native_instance_item(fields, nk[].k__d))
    model_dict(mods, data)
    if op == 0:
        var mt = py_get(data, nk[].k_model_type)
        if py_type(mt) is not mods.module("builtins").str:
            raise Error("model config has no string model_type")
        return mt
    if op >= 1 and op <= 3:
        _ = model_string(mods, key)
    if op == 1:
        return py_get(data, key, default)
    if op == 2:
        if not py_contains(data, key):
            raise Error("KeyError: " + String(py_call(mods.module("builtins").repr, key)))
        return native_instance_item(data, key)
    if op == 3:
        return py_contains(data, key)
    if op == 4:
        return data.keys()
    if op != 5:
        raise Error("ValueError: unknown model config access operation")
    return data


def model_plan_init_binding(mut mods: NativeModules, plan: PythonObject, input: PythonObject) raises -> PythonObject:
    var fields = _plan_fields(mods, plan)
    model_dict(mods, input)
    var names = py_call(mods.module("builtins").list, input.keys())
    for i in range(len(names)):  # driver-loop: preserve caller plan fields without invoking setattr descriptors
        py_setitem(fields, names[i], native_instance_item(input, names[i]))
    return PythonObject(None)


def causal_layer_map_unpack(mut mods: NativeModules, entries: PythonObject) raises -> List[CausalLayerEntry]:
    """UNPACK: an already captured (model_plain) layer map as typed entries."""
    model_sequence(mods, entries)
    var result = List[CausalLayerEntry](capacity=len(entries))
    for j in range(len(entries)):  # driver-loop: admit each layer map entry once
        var entry = entries[j]
        model_sequence(mods, entry)
        if len(entry) != 3:
            raise Error("ValueError: model layer map requires name, checkpoint name and row slice")
        var key = model_string(mods, entry[0])
        var name = model_string(mods, entry[1])
        var rows = entry[2]
        var begin = 0
        var end = 0
        if rows is not None:
            model_sequence(mods, rows)
            if len(rows) != 2:
                raise Error("ValueError: model weight row slice requires two offsets")
            begin = model_integer(mods, rows[0])
            end = model_integer(mods, rows[1])
        result.append(CausalLayerEntry(key^, name^, rows is not None, begin, end))
    return result^


def causal_layer_map_pack(entries: List[CausalLayerEntry]) raises -> PythonObject:
    """PACK: typed layer entries as the plan's list of `(key, name, rows_or_None)` tuples."""
    var out = Python.list()
    for ref entry in entries:  # driver-loop: publish each layer map entry
        var rows = Python.tuple(PythonObject(entry.begin), PythonObject(entry.end)) if entry.has_rows else PythonObject(None)
        _ = py_call(out.append, Python.tuple(PythonObject(entry.key), PythonObject(entry.name), rows))
    return out


def causal_plan_layer_map(mut mods: NativeModules, fields: PythonObject) raises -> List[CausalLayerEntry]:
    """UNPACK: the plan fields' layer map, captured and typed once."""
    var nk = causal_keys()
    return causal_layer_map_unpack(mods, model_plain(mods, native_instance_item(fields, nk[].k__layer_map)))


def _model_layer_weights(mut mods: NativeModules, fields: PythonObject, layer: Int) raises -> PythonObject:
    return causal_layer_map_pack(causal_layer_weights(causal_plan_layer_map(mods, fields), layer))


def model_layer_weights_binding(mut mods: NativeModules, plan: PythonObject, layer: PythonObject) raises -> PythonObject:
    return _model_layer_weights(mods, _plan_fields(mods, plan), model_integer(mods, layer))


def model_checkpoint_names_snapshot(mut mods: NativeModules, fields: PythonObject) raises -> PythonObject:
    # UNPACK: the three outer names, the layer count and the layer map, read
    # once; RUN: `causal_checkpoint_names` (typed dedupe in insertion order);
    # PACK: the list of names.
    model_dict(mods, fields)
    var outer = List[String]()
    for field in [String("embed_name"), String("norm_name"), String("head_name")]:  # small-loop(three outer checkpoint weight names)
        var name = native_instance_item(fields, PythonObject(field))
        if name is not None:
            outer.append(model_string(mods, name))
    var nk = causal_keys()
    var layers = model_integer(mods, native_instance_item(fields, nk[].k_n_layers))
    var entries = causal_plan_layer_map(mods, fields)
    var names = causal_checkpoint_names(outer, entries, layers)
    var out = Python.list()
    for ref name in names:  # driver-loop: publish the plan's checkpoint names
        _ = py_call(out.append, PythonObject(name))
    return out


def model_checkpoint_names_binding(mut mods: NativeModules, plan: PythonObject) raises -> PythonObject:
    return model_checkpoint_names_snapshot(mods, model_plan_snapshot(mods, plan))


def model_config_repr(mut mods: NativeModules, value: PythonObject, plan_value: PythonObject) raises -> PythonObject:
    var bi = mods.module("builtins")
    if py_type(plan_value) is not bi.bool:
        raise Error("TypeError: model repr kind requires an exact boolean")
    var nk = causal_keys()
    if not Bool(py=plan_value):
        var fields = _config_fields(mods, value)
        var data = model_plain(mods, native_instance_item(fields, nk[].k__d))
        var path = native_instance_item(fields, nk[].k_path)
        path = PythonObject(None) if path is None else native_path(path)
        return PythonObject("HFConfig(model_type=" + String(py_call(bi.repr, py_get(data, nk[].k_model_type))) + ", path=" + String(py_call(bi.repr, path)) + ")")
    var fields = _plan_fields(mods, value)
    var mt = model_string(mods, native_instance_item(fields, nk[].k_model_type))
    var kind = model_string(mods, native_instance_item(fields, nk[].k_kind))
    var layers = model_integer(mods, native_instance_item(fields, nk[].k_n_layers))
    var dm = model_integer(mods, native_instance_item(fields, nk[].k_d_model))
    var vocab = model_integer(mods, native_instance_item(fields, nk[].k_vocab_size))
    return PythonObject("ModelPlan(" + String(py_call(bi.repr, PythonObject(mt))) + ", kind=" + String(py_call(bi.repr, PythonObject(kind))) + ", layers=" + String(layers) + ", d_model=" + String(dm) + ", vocab=" + String(vocab) + ")")


def _config_value(mut mods: NativeModules, data: PythonObject, key: PythonObject) raises -> CausalConfigValue:
    # UNPACK of one captured (model_plain) configuration field: exact JSON
    # type, value and Python truth. Plain values carry no Python hooks.
    if not py_contains(data, key):
        return CausalConfigValue()
    var v = native_instance_item(data, key)
    if v is None:
        return CausalConfigValue(CONFIG_NULL, 0, 0.0, String(), False)
    var bi = mods.module("builtins")
    var cls = py_type(v)
    if cls is bi.bool:
        var b = Bool(py=v)
        return CausalConfigValue(CONFIG_BOOL, Int(b), Float64(Int(b)), String(), b)
    if cls is bi.int:
        var i = Int(py=v)
        return CausalConfigValue(CONFIG_INT, i, Float64(i), String(), i != 0)
    if cls is bi.float:
        var f = Float64(py=v)
        return CausalConfigValue(CONFIG_FLOAT, 0, f, String(), f != 0)
    if cls is bi.str:
        var s = String(v)
        var truthy = s.byte_length() > 0
        return CausalConfigValue(CONFIG_STR, 0, 0.0, s^, truthy)
    return CausalConfigValue(CONFIG_OTHER, 0, 0.0, String(), Bool(py=v))


def _config_request(mut mods: NativeModules, source: PythonObject, data: PythonObject, mt: String, byte_lm: Bool) raises -> CausalConfigRequest:
    # UNPACK: every field the planner consumes, read once from `source`
    # (the configuration, or the byte-LM format's renamed HF view).
    var r = CausalConfigRequest()
    r.model_type = mt
    r.byte_lm = byte_lm
    var names = [String("num_local_experts"), String("num_experts"), String("num_experts_per_tok"), String("n_routed_experts"),
                 String("rope_scaling"), String("rope_type"), String("hidden_act"), String("use_sliding_window"),
                 String("hidden_size"), String("num_attention_heads"), String("num_key_value_heads"), String("head_dim"),
                 String("intermediate_size"), String("num_hidden_layers"), String("vocab_size"), String("rms_norm_eps"),
                 String("rope_theta"), String("sliding_window"), String("attention_bias"), String("mlp_bias"),
                 String("attention_dropout"), String("tie_word_embeddings"), String("max_position_embeddings"),
                 String("state_size"), String("conv_kernel"), String("expand"), String("use_bias"), String("use_conv_bias"),
                 String("n_groups"), String("chunk_size"), String("rms_norm"), String("norm_before_gate"), String("num_heads"),
                 String("time_step_rank"), String("layer_norm_epsilon"), String("original_max_position_embeddings"),
                 String("attn_logit_softcapping"), String("final_logit_softcapping"), String("query_pre_attn_scalar")]
    for ref name in names:  # small-loop(the fixed consumed configuration field names)
        r.fields[name] = _config_value(mods, source, PythonObject(name))
    var nk = causal_keys()
    r.hidden_activation_key = py_contains(data, nk[].k_hidden_activation)
    var types = py_get(source, nk[].k_layer_types)
    if types is not None:
        r.layer_kinds = len(py_call(mods.module("builtins").set, types))
    if mt == "mamba2":
        var limit = py_get(source, nk[].k_time_step_limit)
        if limit is not None:
            var bi = mods.module("builtins")
            r.dt_limit_given = True
            r.dt_lo = Float64(py=py_call(bi.float, limit[0]))
            r.dt_hi = Float64(py=py_call(bi.float, limit[1]))
    if byte_lm:
        ref tables = MODEL_CONFIG_CONNECTION.get_or_create_ptr()[].tables
        for ref fixed in tables.byte_fixed:  # small-loop(the fixed byte-LM config fields)
            r.fixed_got[fixed.name] = _config_value(mods, data, PythonObject(fixed.name))
    return r^


def _config_refusal(mut mods: NativeModules, source: PythonObject, plan: CausalConfigPlan) raises -> Error:
    # PACK of a refusal by name: the message quotes the field's Python repr.
    var bi = mods.module("builtins")
    var nk = causal_keys()
    var why = plan.refuse_why
    if plan.refuse_suffix_field.byte_length() > 0:
        why += String(py_call(bi.repr, py_get(source, plan.refuse_suffix_field))) + ")"
    return Error("model_type " + String(py_call(bi.repr, py_get(source, nk[].k_model_type, PythonObject("mojolearn-byte-lm")))) + ": " + plan.refuse_field + "=" + String(py_call(bi.repr, py_get(source, plan.refuse_field))) + " is not honored: " + why)


def _config_plan_fields(mut mods: NativeModules, plan: CausalConfigPlan, context: PythonObject) raises -> PythonObject:
    # PACK: the plan words as the ModelPlan fields (same keys, same order).
    var bi = mods.module("builtins")
    var nk = causal_keys()
    var fields = bi.dict()
    py_setitem(fields, nk[].k_kind, PythonObject(plan.kind))
    py_setitem(fields, nk[].k_n_layers, PythonObject(plan.n_layers))
    py_setitem(fields, nk[].k_d_model, PythonObject(plan.d_model))
    var kwargs = bi.dict()
    var options = bi.dict()
    if plan.transformer:
        py_setitem(fields, nk[].k_n_heads, PythonObject(plan.n_heads))
        py_setitem(fields, nk[].k_n_kv_heads, PythonObject(plan.n_kv_heads))
        py_setitem(fields, nk[].k_head_dim, PythonObject(plan.head_dim))
        py_setitem(fields, nk[].k_intermediate, PythonObject(plan.intermediate))
        options = py_call(bi.dict, py_item(context, nk[].k_defaults))
        py_setitem(options, nk[].k_n_kv_heads, PythonObject(plan.opt_n_kv_heads) if plan.opt_n_kv_heads >= 0 else PythonObject(None))
        py_setitem(options, nk[].k_head_dim, PythonObject(plan.opt_head_dim) if plan.opt_head_dim >= 0 else PythonObject(None))
        py_setitem(options, nk[].k_window, PythonObject(plan.window))
        py_setitem(options, nk[].k_rope_theta, PythonObject(plan.rope_theta))
        py_setitem(options, nk[].k_norm_eps, PythonObject(plan.norm_eps))
        py_setitem(options, nk[].k_qkv_bias, PythonObject(plan.qkv_bias))
        py_setitem(options, nk[].k_o_bias, PythonObject(False))
        py_setitem(options, nk[].k_mlp_bias, PythonObject(plan.mlp_bias))
        py_setitem(options, nk[].k_qk_norm, PythonObject(plan.qk_norm))
    elif plan.has_dt_limit:
        py_setitem(kwargs, nk[].k_dt_limit, Python.tuple(PythonObject(plan.dt_lo), PythonObject(plan.dt_hi)))
    py_setitem(fields, nk[].k_vocab_size, PythonObject(plan.vocab_size))
    py_setitem(fields, nk[].k_tie_embeddings, PythonObject(plan.tied))
    py_setitem(fields, nk[].k_norm_eps, PythonObject(plan.norm_eps))
    py_setitem(fields, nk[].k_max_position_embeddings, PythonObject(plan.max_positions) if plan.transformer else PythonObject(None))
    py_setitem(fields, nk[].k_block_options, options)
    py_setitem(fields, nk[].k_block_kwargs, kwargs)
    py_setitem(fields, nk[].k__layer_map, causal_layer_map_pack(plan.layer_map))
    py_setitem(fields, nk[].k_embed_name, PythonObject(plan.embed_name))
    py_setitem(fields, nk[].k_norm_name, PythonObject(plan.norm_name) if plan.has_norm_name else PythonObject(None))
    py_setitem(fields, nk[].k_head_name, PythonObject(plan.head_name) if plan.has_head_name else PythonObject(None))
    return fields


def model_plan_for_binding(mut mods: NativeModules, input: PythonObject, supplied_context: PythonObject) raises -> PythonObject:
    # UNPACK: the configuration record, its model_type and family row, and the
    # consumed fields (`_config_request`). RUN: `causal_config_plan`
    # (bindings/causal_model_config_run.mojo), typed. PACK: the ModelPlan.
    # The GIL is held through RUN: it is a few dozen comparisons once per
    # model construction, less than a release / reacquire pair.
    var bi = mods.module("builtins")
    var context = _model_context()
    var config = input
    var nk = causal_keys()
    if py_type(input) is bi.dict:
        config = py_call(bi.object.__new__, py_item(context, nk[].k_config_class))
        _ = config_init_binding(mods, config, input, PythonObject(None))
    var fields_owner = _config_fields(mods, config)
    var data = model_plain(mods, native_instance_item(fields_owner, nk[].k__d))
    model_dict(mods, data)
    var mt_value = py_get(data, nk[].k_model_type)
    var byte_lm = mt_value is None and Bool(py=py_get(data, nk[].k_format) == PythonObject("mojolearn.byte-lm"))
    if not byte_lm and py_type(mt_value) is not bi.str:
        raise Error("model config has no string model_type")
    var mt = "mojolearn-byte-lm" if byte_lm else String(mt_value)
    var row = py_item(context, nk[].k_byte_row) if byte_lm else py_get(py_item(context, nk[].k_families), mt)
    if row is None:
        raise Error("model_type '" + mt + "' is not in the option matrix")
    var source = data
    if byte_lm:
        # The byte-LM format's fields under their HF names (the planner and
        # its refusal messages read this view).
        source = bi.dict()
        var targets = [String("hidden_size"), String("num_attention_heads"), String("num_key_value_heads"), String("head_dim"), String("intermediate_size"), String("num_hidden_layers"), String("vocab_size"), String("rms_norm_eps"), String("rope_theta"), String("max_position_embeddings")]
        var sources = [String("d_model"), String("n_heads"), String("n_kv"), String("head_dim"), String("intermediate"), String("n_layers"), String("vocab_size"), String("rms_norm_eps"), String("rope_theta"), String("max_position_embeddings")]
        for i in range(len(targets)):  # small-loop(the fixed config field mapping)
            var v = py_get(data, sources[i])
            if v is not None:
                py_setitem(source, targets[i], v)
        py_setitem(source, nk[].k_tie_word_embeddings, PythonObject(False))
    var request = _config_request(mods, source, data, mt, byte_lm)
    ref tables = MODEL_CONFIG_CONNECTION.get_or_create_ptr()[].tables
    var plan = causal_config_plan(request, tables)
    if plan.refused:
        var refusal_573 = _config_refusal(mods, source, plan)
        raise refusal_573^
    var fields = _config_plan_fields(mods, plan, context)
    py_setitem(fields, nk[].k_model_type, PythonObject(mt))
    py_setitem(fields, nk[].k_config, config)
    py_setitem(fields, nk[].k_family, row)
    var result = py_call(bi.object.__new__, py_item(context, nk[].k_plan_class))
    _ = model_plan_init_binding(mods, result, fields)
    return result


def model_config_c[operation: Int](py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    var mods = NativeModules()
    var args = PythonObject(from_borrowed=args_ptr)
    try:
        var result = PythonObject(None)
        comptime if operation == 0:
            result = config_init_binding(mods, args[0], args[1], args[2])
        elif operation == 1:
            result = config_load_binding(mods, args[0], args[1])
        elif operation == 2:
            result = config_access_binding(mods, args[0], args[1], args[2], args[3])
        elif operation == 3:
            result = model_plan_init_binding(mods, args[0], args[1])
        elif operation == 4:
            result = model_layer_weights_binding(mods, args[0], args[1])
        elif operation == 5:
            result = model_checkpoint_names_binding(mods, args[0])
        elif operation == 6:
            result = model_plan_for_binding(mods, args[0], args[1])
        elif operation == 7:
            result = model_config_connect(mods, args[0])
        elif operation == 8:
            result = model_config_repr(mods, args[0], args[1])
        return result.steal_data()
    except error:
        var message = String(error)
        if message.startswith("TypeError:"):
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        if message.startswith("FileNotFoundError:"):
            return raise_python_exception(error_without_class(error, "FileNotFoundError"), ExceptionType("PyExc_FileNotFoundError"))
        if message.startswith("KeyError:"):
            return raise_python_exception(error_without_class(error, "KeyError"), ExceptionType("PyExc_KeyError"))
        comptime if operation == 6 or operation == 2:
            try:
                var cls = _model_error_class(mods)
                return causal_set_exception(error, cls._obj_ptr)
            except nested:
                return raise_python_exception(nested)
        else:
            return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
