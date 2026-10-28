# SPDX-License-Identifier: Apache-2.0
"""Native configuration admission for the supported causal model families."""
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from std.ffi import _Global
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_contains, py_get, py_setattr, py_setitem, py_type
from bindings.causal_exception import causal_borrowed_item, causal_set_exception
from bindings.native_json import native_json_read, native_path, native_path_exists
from bindings.native_loaded_module import native_loaded_module, NativeModules
from bindings.native_instance_item import native_instance_item
from bindings.mode_metadata_native import _fields, _class_fields


struct ModelConfigConnection(Defaultable, Movable):
    var context: PythonObject

    def __init__(out self):
        self.context = PythonObject(None)


comptime MODEL_CONFIG_CONNECTION = _Global[StorageType=ModelConfigConnection,
    name="MojolearnModelConfigConnection", init_fn=ModelConfigConnection.__init__]


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
    var keys = py_call(bi.list, value.keys())
    for i in range(len(keys)):  # driver-loop: admit every key before keyed metadata access
        _ = model_string(mods, keys[i])


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
    var ancestry = py_call(bi.type.__dict__["__mro__"].__get__, py_type(value))
    var admitted = False
    for i in range(len(ancestry)):  # driver-loop: inspect model record descriptors before storage reads
        admitted = admitted or ancestry[i] is cls
        var row = _class_fields(mods, ancestry[i])
        if py_contains(row, "__getattr__"):
            raise Error("TypeError: model records cannot invoke Python attribute hooks")
        for hook in [String("__getattribute__"), String("__setattr__")]:  # small-loop(native record access hooks)
            if py_contains(row, hook) and row[hook] is not bi.object.__dict__[hook]:
                raise Error("TypeError: model records cannot invoke Python attribute hooks")
        for j in range(len(names)):  # small-loop(fixed consumed model field names)
            if py_contains(row, names[j]):
                raise Error("TypeError: model state fields cannot be replaced by class descriptors")
    if not admitted:
        raise Error("TypeError: model operation requires its connected record class")
    var fields = _fields(mods, value)
    model_dict(mods, fields)
    return fields


def _model_context() raises -> PythonObject:
    var result = MODEL_CONFIG_CONNECTION.get_or_create_ptr()[].context
    if result is None:
        raise Error("RuntimeError: model configuration requires import-time connection")
    return result


def _config_fields(mut mods: NativeModules, value: PythonObject) raises -> PythonObject:
    return model_record_fields(mods, value, _model_context()["config_class"], Python.tuple("_d", "path"))


def _plan_fields(mut mods: NativeModules, value: PythonObject) raises -> PythonObject:
    return model_record_fields(mods, value, _model_context()["plan_class"],
        Python.tuple("_layer_map", "embed_name", "norm_name", "head_name", "n_layers", "model_type", "kind", "d_model", "vocab_size", "n_heads", "n_kv_heads", "head_dim", "intermediate", "tie_embeddings", "norm_eps", "max_position_embeddings", "block_options", "block_kwargs", "config", "family"))


def model_plan_snapshot(mut mods: NativeModules, value: PythonObject) raises -> PythonObject:
    """Captured plan dictionary for complete native CausalLM construction."""
    var fields = _plan_fields(mods, value)
    var result = Python.dict()
    var names = py_call(mods.module("builtins").list, fields.keys())
    for i in range(len(names)):  # driver-loop: capture every passive plan member before model construction
        var name = model_string(mods, names[i])
        var item = native_instance_item(fields, names[i])
        if name == "config":
            var source = _config_fields(mods, item)
            var captured = py_call(mods.module("builtins").object.__new__, _model_context()["config_class"])
            _ = config_init_binding(mods, captured, native_instance_item(source, PythonObject("_d")), native_instance_item(source, PythonObject("path")))
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
    for name in [String("defaults"), String("llama_layer"), String("qkv_bias"), String("qk_norm"), String("mamba1_layer"), String("mamba2_layer"), String("byte_layer"), String("byte_fixed")]:  # small-loop(static option matrix data)
        py_setitem(copy, PythonObject(name), model_plain(mods, native_instance_item(context, PythonObject(name))))
    # Family rows retain their historical function identity as public metadata;
    # planning dispatch below never calls those functions.
    var families = native_instance_item(context, PythonObject("families"))
    model_dict(mods, families)
    py_setitem(copy, "families", py_call(bi.dict, families))
    py_setitem(copy, "byte_row", native_instance_item(context, PythonObject("byte_row")))
    var state = MODEL_CONFIG_CONNECTION.get_or_create_ptr()
    if state[].context is not None:
        raise Error("TypeError: model configuration is already connected")
    state[].context = copy
    return PythonObject(None)


def _model_error_class(mut mods: NativeModules) raises -> PythonObject:
    var cls = _model_context()["unsupported_class"]
    var bi = mods.module("builtins")
    var bases = py_call(bi.type.__dict__["__bases__"].__get__, cls)
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
    py_setitem(fields, "_d", captured)
    py_setitem(fields, "path", filename)
    return PythonObject(None)


def config_load_binding(mut mods: NativeModules, cls: PythonObject, input_path: PythonObject) raises -> PythonObject:
    if cls is not _model_context()["config_class"]:
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
    var data = model_plain(mods, native_instance_item(fields, PythonObject("_d")))
    model_dict(mods, data)
    if op == 0:
        var mt = py_get(data, "model_type")
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


def _model_layer_weights(mut mods: NativeModules, fields: PythonObject, layer: Int) raises -> PythonObject:
    var bi = mods.module("builtins")
    var entries = model_plain(mods, native_instance_item(fields, PythonObject("_layer_map")))
    model_sequence(mods, entries)
    var out = Python.list()
    for j in range(len(entries)):  # driver-loop: expand admitted checkpoint names for one model layer
        var entry = entries[j]
        model_sequence(mods, entry)
        if len(entry) != 3:
            raise Error("ValueError: model layer map requires name, checkpoint name and row slice")
        var key = model_string(mods, entry[0])
        var name = model_string(mods, entry[1]).replace("{i}", String(layer))
        var rows = entry[2]
        if rows is not None:
            model_sequence(mods, rows)
            if len(rows) != 2:
                raise Error("ValueError: model weight row slice requires two offsets")
            _ = model_integer(mods, rows[0])
            _ = model_integer(mods, rows[1])
        _ = py_call(out.append, Python.tuple(key, name, rows))
    return out


def model_layer_weights_binding(mut mods: NativeModules, plan: PythonObject, layer: PythonObject) raises -> PythonObject:
    return _model_layer_weights(mods, _plan_fields(mods, plan), model_integer(mods, layer))


def model_checkpoint_names_snapshot(mut mods: NativeModules, fields: PythonObject) raises -> PythonObject:
    model_dict(mods, fields)
    var names = Python.list()
    for field in [String("embed_name"), String("norm_name"), String("head_name")]:  # small-loop(three outer checkpoint weight names)
        var name = native_instance_item(fields, PythonObject(field))
        if name is not None:
            _ = model_string(mods, name)
            _ = py_call(names.append, name)
    var layers = model_integer(mods, native_instance_item(fields, PythonObject("n_layers")))
    # Capture the mutable layer map once for the whole enumeration.
    var captured = Python.dict()
    py_setitem(captured, "_layer_map", model_plain(mods, native_instance_item(fields, PythonObject("_layer_map"))))
    for i in range(layers):  # driver-loop: enumerate complete model checkpoint names
        var layer = _model_layer_weights(mods, captured, i)
        for j in range(len(layer)):  # driver-loop: deduplicate native checkpoint names in insertion order
            var name = layer[j][1]
            if not py_contains(names, name):
                _ = py_call(names.append, name)
    return names


def model_checkpoint_names_binding(mut mods: NativeModules, plan: PythonObject) raises -> PythonObject:
    return model_checkpoint_names_snapshot(mods, model_plan_snapshot(mods, plan))


def model_config_repr(mut mods: NativeModules, value: PythonObject, plan_value: PythonObject) raises -> PythonObject:
    var bi = mods.module("builtins")
    if py_type(plan_value) is not bi.bool:
        raise Error("TypeError: model repr kind requires an exact boolean")
    if not Bool(py=plan_value):
        var fields = _config_fields(mods, value)
        var data = model_plain(mods, native_instance_item(fields, PythonObject("_d")))
        var path = native_instance_item(fields, PythonObject("path"))
        path = PythonObject(None) if path is None else native_path(path)
        return PythonObject("HFConfig(model_type=" + String(py_call(bi.repr, py_get(data, "model_type"))) + ", path=" + String(py_call(bi.repr, path)) + ")")
    var fields = _plan_fields(mods, value)
    var mt = model_string(mods, native_instance_item(fields, PythonObject("model_type")))
    var kind = model_string(mods, native_instance_item(fields, PythonObject("kind")))
    var layers = model_integer(mods, native_instance_item(fields, PythonObject("n_layers")))
    var dm = model_integer(mods, native_instance_item(fields, PythonObject("d_model")))
    var vocab = model_integer(mods, native_instance_item(fields, PythonObject("vocab_size")))
    return PythonObject("ModelPlan(" + String(py_call(bi.repr, PythonObject(mt))) + ", kind=" + String(py_call(bi.repr, PythonObject(kind))) + ", layers=" + String(layers) + ", d_model=" + String(dm) + ", vocab=" + String(vocab) + ")")


def _config_int(mut mods: NativeModules, data: PythonObject, field: String, default: PythonObject, required: Bool = True) raises -> PythonObject:
    var v = py_get(data, field, default)
    var bi = mods.module("builtins")
    if v is None and not required:
        return v
    if v is None:
        raise Error("model config: " + field + ": the field is required and the config lacks it")
    if py_type(v) is not bi.int:
        raise Error("model config: " + field + ": an integer is required")
    return v


def _config_int(mut mods: NativeModules, data: PythonObject, field: String) raises -> PythonObject:
    return _config_int(mods, data, field, PythonObject(None), True)


def _config_refuse(mut mods: NativeModules, data: PythonObject, field: String, why: String) raises:
    var bi = mods.module("builtins")
    raise Error("model_type " + String(py_call(bi.repr, py_get(data, "model_type", PythonObject("mojolearn-byte-lm")))) + ": " + field + "=" + String(py_call(bi.repr, py_get(data, field))) + " is not honored: " + why)


def _config_number(mut mods: NativeModules, data: PythonObject, field: String, default: Float64) raises -> PythonObject:
    var v = py_get(data, field, PythonObject(default))
    var bi = mods.module("builtins")
    if py_type(v) is not bi.int and py_type(v) is not bi.float:
        raise Error("model config: " + field + " requires a number")
    return py_call(bi.float, v)


def _config_fixed(mut mods: NativeModules, data: PythonObject, names: List[String], values: List[Int]) raises:
    for i in range(len(names)):  # small-loop(fixed config field words)
        var got = py_get(data, names[i], PythonObject(values[i]))
        if not Bool(py=got == PythonObject(values[i])):
            _config_refuse(mods, data, names[i], "fixed at " + String(values[i]))


def _config_transformer(mut mods: NativeModules, data: PythonObject, mt: String, context: PythonObject) raises -> PythonObject:
    var bi = mods.module("builtins")
    for field in [String("num_local_experts"), String("num_experts"), String("num_experts_per_tok"), String("n_routed_experts")]:  # small-loop(the four mixture-of-experts field names)
        var v = py_get(data, field)
        if v is not None and not Bool(py=v == PythonObject(0)) and not Bool(py=v == PythonObject(1)):
            _config_refuse(mods, data, field, "mixture of experts")
    if py_get(data, "rope_scaling") is not None:
        _config_refuse(mods, data, "rope_scaling", "rope type default only")
    var rt = py_get(data, "rope_type", PythonObject("default"))
    if rt is not None and String(rt) != "default":
        _config_refuse(mods, data, "rope_type", "default only")
    var act = String(py_get(data, "hidden_act", PythonObject("silu")))
    if act != "silu" and act != "swish":
        _config_refuse(mods, data, "hidden_act", "requires silu/swish SwiGLU")
    if Bool(py=py_get(data, "use_sliding_window", PythonObject(False))):
        _config_refuse(mods, data, "use_sliding_window", "sliding-window-alternating layers (max_window_layers=" + String(py_call(bi.repr, py_get(data, "max_window_layers"))) + ")")
    var types = py_get(data, "layer_types")
    if types is not None and len(py_call(bi.set, types)) > 1:
        raise Error("model config: layer_types requests alternating attention kinds")
    var dm = Int(py=_config_int(mods, data, "hidden_size"))
    var nh = Int(py=_config_int(mods, data, "num_attention_heads"))
    var nk = Int(py=_config_int(mods, data, "num_key_value_heads", PythonObject(nh)))
    if dm < 1 or nh < 1 or nk < 1:
        raise Error("model config: hidden_size and attention head counts must be positive")
    var hd_field = _config_int(mods, data, "head_dim", PythonObject(None), False)
    var hd = dm // nh if hd_field is None else Int(py=hd_field)
    if nh * hd != dm:
        raise Error("model config: head_dim * num_attention_heads must equal hidden_size")
    if nh % nk != 0:
        _config_refuse(mods, data, "num_key_value_heads", "num_attention_heads is not a multiple of it (must divide num_attention_heads)")
    if hd % 2 != 0:
        raise Error("model config: head_dim must be even")
    var inter = _config_int(mods, data, "intermediate_size")
    var layers = _config_int(mods, data, "num_hidden_layers")
    var vocab = _config_int(mods, data, "vocab_size")
    var eps = _config_number(mods, data, "rms_norm_eps", 1e-6)
    var theta = _config_number(mods, data, "rope_theta", 10000.0)
    var window = py_get(data, "sliding_window")
    if window is None:
        window = PythonObject(0)
    if py_type(window) is not bi.int or Int(py=window) < 0:
        raise Error("model config: sliding_window requires an integer width or null")
    var bias = Bool(py=py_get(data, "attention_bias", PythonObject(mt == "qwen2")))
    var mlp_bias = Bool(py=py_get(data, "mlp_bias", PythonObject(False)))
    var dropout = py_get(data, "attention_dropout", PythonObject(0.0))
    if dropout is not None and Float64(py=dropout) != 0:
        raise Error("model config: attention_dropout must be zero: inference has no dropout")
    var tied = Bool(py=py_get(data, "tie_word_embeddings", PythonObject(False)))
    var maxpos = _config_int(mods, data, "max_position_embeddings", PythonObject(8192))
    var options = py_call(bi.dict, context["defaults"])
    py_setitem(options, "n_kv_heads", PythonObject(nk) if nk != nh else PythonObject(None))
    py_setitem(options, "head_dim", PythonObject(hd) if hd != dm // nh else PythonObject(None))
    py_setitem(options, "window", window)
    py_setitem(options, "rope_theta", theta)
    py_setitem(options, "norm_eps", eps)
    py_setitem(options, "qkv_bias", PythonObject(bias))
    py_setitem(options, "o_bias", PythonObject(False))
    py_setitem(options, "mlp_bias", PythonObject(mlp_bias))
    py_setitem(options, "qk_norm", PythonObject(mt == "qwen3"))
    var layer_map = py_call(bi.list, context["llama_layer"])
    if bias:
        _ = py_call(layer_map.extend, context["qkv_bias"])
    if mt == "qwen3":
        _ = py_call(layer_map.extend, context["qk_norm"])
    var fields = bi.dict()
    py_setitem(fields, "kind", PythonObject("transformer"))
    py_setitem(fields, "n_layers", layers)
    py_setitem(fields, "d_model", PythonObject(dm))
    py_setitem(fields, "n_heads", PythonObject(nh))
    py_setitem(fields, "n_kv_heads", PythonObject(nk))
    py_setitem(fields, "head_dim", PythonObject(hd))
    py_setitem(fields, "intermediate", inter)
    py_setitem(fields, "vocab_size", vocab)
    py_setitem(fields, "tie_embeddings", PythonObject(tied))
    py_setitem(fields, "norm_eps", eps)
    py_setitem(fields, "max_position_embeddings", maxpos)
    py_setitem(fields, "block_options", options)
    py_setitem(fields, "block_kwargs", bi.dict())
    py_setitem(fields, "_layer_map", layer_map)
    py_setitem(fields, "embed_name", PythonObject("model.embed_tokens.weight"))
    py_setitem(fields, "norm_name", PythonObject("model.norm.weight"))
    py_setitem(fields, "head_name", PythonObject(None) if tied else PythonObject("lm_head.weight"))
    return fields


def _config_mamba(mut mods: NativeModules, data: PythonObject, mt: String, context: PythonObject) raises -> PythonObject:
    var bi = mods.module("builtins")
    var dm = Int(py=_config_int(mods, data, "hidden_size"))
    var layers = _config_int(mods, data, "num_hidden_layers")
    var vocab = _config_int(mods, data, "vocab_size")
    var two = mt == "mamba2"
    _config_fixed(mods, data, ["state_size", "conv_kernel", "expand", "use_bias", "use_conv_bias"],
                  [128 if two else 16, 4, 2, 0, 1])
    var kwargs = bi.dict()
    if two:
        _config_fixed(mods, data, ["head_dim", "n_groups", "chunk_size", "rms_norm"], [64, 1, 256, 1])
        if Bool(py=py_get(data, "norm_before_gate", PythonObject(False))):
            raise Error("model config: norm_before_gate must be false")
        if dm % 32 != 0:
            _config_refuse(mods, data, "hidden_size", "must be a multiple of 32")
        var nh = py_get(data, "num_heads")
        if nh is not None and not Bool(py=nh == PythonObject(2 * dm // 64)):
            raise Error("model config: num_heads must equal expand*hidden_size/head_dim")
        var limit = py_get(data, "time_step_limit")
        var bounds = bi.list()
        if limit is None:
            _ = py_call(bounds.append, 0.0)
            _ = py_call(bounds.append, py_call(bi.float, "inf"))
        else:
            _ = py_call(bounds.append, py_call(bi.float, limit[0]))
            _ = py_call(bounds.append, py_call(bi.float, limit[1]))
        py_setitem(kwargs, "dt_limit", py_call(bi.tuple, bounds))
    else:
        var rank = py_get(data, "time_step_rank", PythonObject("auto"))
        if not Bool(py=rank == PythonObject("auto")) and not Bool(py=rank == PythonObject((dm + 15) // 16)):
            raise Error("model config: time_step_rank must be auto or ceil(hidden_size/16)")
        if String(py_get(data, "hidden_act", PythonObject("silu"))) != "silu":
            raise Error("model config: hidden_act requires silu")
    var eps = _config_number(mods, data, "layer_norm_epsilon", 1e-5)
    if Float64(py=eps) != 1e-5:
        _config_refuse(mods, data, "layer_norm_epsilon", "fixed at 1e-5")
    var tied = Bool(py=py_get(data, "tie_word_embeddings", PythonObject(True)))
    var fields = bi.dict()
    py_setitem(fields, "kind", PythonObject("mamba2" if two else "mamba1"))
    py_setitem(fields, "n_layers", layers)
    py_setitem(fields, "d_model", PythonObject(dm))
    py_setitem(fields, "vocab_size", vocab)
    py_setitem(fields, "tie_embeddings", PythonObject(tied))
    py_setitem(fields, "norm_eps", eps)
    py_setitem(fields, "max_position_embeddings", PythonObject(None))
    py_setitem(fields, "block_options", bi.dict())
    py_setitem(fields, "block_kwargs", kwargs)
    py_setitem(fields, "_layer_map", py_call(bi.list, context["mamba2_layer" if two else "mamba1_layer"]))
    py_setitem(fields, "embed_name", PythonObject("backbone.embeddings.weight"))
    py_setitem(fields, "norm_name", PythonObject("backbone.norm_f.weight"))
    py_setitem(fields, "head_name", PythonObject(None) if tied else PythonObject("lm_head.weight"))
    return fields


def _config_phi3(mut mods: NativeModules, mut fields: PythonObject, data: PythonObject) raises:
    var bi = mods.module("builtins")
    var original = py_get(data, "original_max_position_embeddings")
    if original is not None and not Bool(py=original == fields["max_position_embeddings"]):
        raise Error("model config: original_max_position_embeddings requests longrope extended context")
    var qr = Int(py=fields["n_heads"]) * Int(py=fields["head_dim"])
    var kr = Int(py=fields["n_kv_heads"]) * Int(py=fields["head_dim"])
    var it = Int(py=fields["intermediate"])
    var layers = fields["_layer_map"]
    var mapped = bi.list()
    for i in range(len(layers)):  # driver-loop: over model layer descriptors
        var item = py_call(bi.list, layers[i])
        var key = String(item[0])
        var begin = -1
        var end = -1
        if key == "q_proj.weight" or key == "k_proj.weight" or key == "v_proj.weight":
            py_setitem(item, 1, PythonObject("model.layers.{i}.self_attn.qkv_proj.weight"))
            begin = 0 if key == "q_proj.weight" else (qr if key == "k_proj.weight" else qr + kr)
            end = qr if key == "q_proj.weight" else (qr + kr if key == "k_proj.weight" else qr + 2 * kr)
        elif key == "gate_proj.weight" or key == "up_proj.weight":
            py_setitem(item, 1, PythonObject("model.layers.{i}.mlp.gate_up_proj.weight"))
            begin = 0 if key == "gate_proj.weight" else it
            end = it if key == "gate_proj.weight" else 2 * it
        if begin >= 0:
            var bounds = bi.list()
            _ = py_call(bounds.append, begin)
            _ = py_call(bounds.append, end)
            py_setitem(item, 2, py_call(bi.tuple, bounds))
        _ = py_call(mapped.append, py_call(bi.tuple, item))
    py_setitem(fields, "_layer_map", mapped)


def model_plan_for_binding(mut mods: NativeModules, input: PythonObject, supplied_context: PythonObject) raises -> PythonObject:
    var bi = mods.module("builtins")
    var context = _model_context()
    var config = input
    if py_type(input) is bi.dict:
        config = py_call(bi.object.__new__, context["config_class"])
        _ = config_init_binding(mods, config, input, PythonObject(None))
    var fields_owner = _config_fields(mods, config)
    var data = model_plain(mods, native_instance_item(fields_owner, PythonObject("_d")))
    model_dict(mods, data)
    var mt_value = py_get(data, "model_type")
    var byte_lm = mt_value is None and Bool(py=py_get(data, "format") == PythonObject("mojolearn.byte-lm"))
    if not byte_lm and py_type(mt_value) is not bi.str:
        raise Error("model config has no string model_type")
    var mt = "mojolearn-byte-lm" if byte_lm else String(mt_value)
    var row = context["byte_row"] if byte_lm else py_get(context["families"], mt)
    if row is None:
        raise Error("model_type '" + mt + "' is not in the option matrix")
    if mt == "gemma2":
        for field in [String("attn_logit_softcapping"), String("final_logit_softcapping"), String("query_pre_attn_scalar")]:  # small-loop(the three fixed gemma2 field names)
            if py_get(data, field) is not None:
                _config_refuse(mods, data, field, "unsupported for gemma2")
    if mt == "gemma" or mt == "gemma2":
        var field = "hidden_activation" if py_contains(data, "hidden_activation") else "hidden_act"
        _config_refuse(mods, data, field, "unsupported GeGLU, RMSNorm offset and embedding scale")
    var fields = PythonObject(None)
    if byte_lm:
        var fixed = context["byte_fixed"]
        var names = py_call(bi.list, fixed)
        for i in range(len(names)):  # small-loop(the fixed byte-LM config fields)
            var name = names[i]
            if py_contains(data, name) and not Bool(py=data[name] == fixed[name]):
                raise Error("model config: byte-LM " + String(name) + " differs from required value")
        var hf = bi.dict()
        var targets = [String("hidden_size"), String("num_attention_heads"), String("num_key_value_heads"), String("head_dim"), String("intermediate_size"), String("num_hidden_layers"), String("vocab_size"), String("rms_norm_eps"), String("rope_theta"), String("max_position_embeddings")]
        var sources = [String("d_model"), String("n_heads"), String("n_kv"), String("head_dim"), String("intermediate"), String("n_layers"), String("vocab_size"), String("rms_norm_eps"), String("rope_theta"), String("max_position_embeddings")]
        for i in range(len(targets)):  # small-loop(the fixed config field mapping)
            var v = py_get(data, sources[i])
            if v is not None:
                py_setitem(hf, targets[i], v)
        py_setitem(hf, "tie_word_embeddings", PythonObject(False))
        fields = _config_transformer(mods, hf, mt, context)
        py_setitem(fields, "_layer_map", py_call(bi.list, context["byte_layer"]))
        py_setitem(fields, "embed_name", PythonObject("embed"))
        py_setitem(fields, "norm_name", PythonObject(None))
        py_setitem(fields, "head_name", PythonObject("lm_head"))
    elif mt == "mamba" or mt == "mamba2":
        fields = _config_mamba(mods, data, mt, context)
    else:
        fields = _config_transformer(mods, data, mt, context)
        if mt == "phi3":
            _config_phi3(mods, fields, data)
    py_setitem(fields, "model_type", PythonObject(mt))
    py_setitem(fields, "config", config)
    py_setitem(fields, "family", row)
    var plan = py_call(bi.object.__new__, context["plan_class"])
    _ = model_plan_init_binding(mods, plan, fields)
    return plan


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
