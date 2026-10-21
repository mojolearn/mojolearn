# SPDX-License-Identifier: Apache-2.0
"""Native configuration admission for the supported causal model families."""
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_contains, py_get, py_setattr, py_setitem, py_type
from bindings.causal_exception import causal_borrowed_item, causal_set_exception
from bindings.native_json import native_json_read, native_path, native_path_exists


def config_init_binding(value: PythonObject, data: PythonObject, path: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    if py_type(data) is not bi.dict:
        raise Error("TypeError: HFConfig requires a JSON object")
    py_setattr(value, "_d", py_call(bi.dict, data))
    py_setattr(value, "path", path)
    return PythonObject(None)


def config_load_binding(cls: PythonObject, input_path: PythonObject) raises -> PythonObject:
    var path = native_path(input_path)
    if native_path_exists(path, True):
        path = py_call(path.rstrip, "/") + PythonObject("/config.json")
    if not native_path_exists(path):
        raise Error("FileNotFoundError: config.json does not exist")
    var value = py_call(Python.import_module("builtins").object.__new__, cls)
    _ = config_init_binding(value, native_json_read(path), path)
    return value


def config_access_binding(config: PythonObject, operation: PythonObject, key: PythonObject, default: PythonObject) raises -> PythonObject:
    var op = Int(py=operation)
    if op == 0:
        var mt = py_get(config._d, "model_type")
        if py_type(mt) is not Python.import_module("builtins").str:
            raise Error("model config has no string model_type")
        return mt
    if op == 1:
        return py_get(config._d, key, default)
    if op == 2:
        if not py_contains(config._d, key):
            raise Error("KeyError: " + String(py_call(Python.import_module("builtins").repr, key)))
        return config._d[key]
    if op == 3:
        return py_contains(config._d, key)
    if op == 4:
        return config._d.keys()
    return py_call(Python.import_module("builtins").dict, config._d)


def model_plan_init_binding(plan: PythonObject, fields: PythonObject) raises -> PythonObject:
    _ = py_call(plan.__dict__.update, fields)
    return PythonObject(None)


def model_layer_weights_binding(plan: PythonObject, layer: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var out = bi.list()
    for j in range(len(plan._layer_map)):  # driver-loop: over the model layer map entries
        var entry = plan._layer_map[j]
        var item = bi.list()
        _ = py_call(item.append, entry[0])
        _ = py_call(item.append, PythonObject(String(entry[1]).replace("{i}", String(Int(py=layer)))))
        _ = py_call(item.append, entry[2])
        _ = py_call(out.append, py_call(bi.tuple, item))
    return out


def model_checkpoint_names_binding(plan: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var names = bi.list()
    _ = py_call(names.append, plan.embed_name)
    if plan.norm_name is not None:
        _ = py_call(names.append, plan.norm_name)
    if plan.head_name is not None:
        _ = py_call(names.append, plan.head_name)
    for i in range(Int(py=plan.n_layers)):  # driver-loop: over model layers
        var layer = model_layer_weights_binding(plan, PythonObject(i))
        for j in range(len(layer)):  # driver-loop: over one layer's weight entries
            var name = layer[j][1]
            if not py_contains(names, name):
                _ = py_call(names.append, name)
    return names


def _config_int(data: PythonObject, field: String, default: PythonObject, required: Bool = True) raises -> PythonObject:
    var v = py_get(data, field, default)
    var bi = Python.import_module("builtins")
    if v is None and not required:
        return v
    if v is None:
        raise Error("model config: " + field + ": the field is required and the config lacks it")
    if py_type(v) is not bi.int:
        raise Error("model config: " + field + ": an integer is required")
    return v


def _config_int(data: PythonObject, field: String) raises -> PythonObject:
    return _config_int(data, field, PythonObject(None), True)


def _config_refuse(data: PythonObject, field: String, why: String) raises:
    var bi = Python.import_module("builtins")
    raise Error("model_type " + String(py_call(bi.repr, py_get(data, "model_type", PythonObject("mojolearn-byte-lm")))) + ": " + field + "=" + String(py_call(bi.repr, py_get(data, field))) + " is not honored: " + why)


def _config_number(data: PythonObject, field: String, default: Float64) raises -> PythonObject:
    var v = py_get(data, field, PythonObject(default))
    var bi = Python.import_module("builtins")
    if py_type(v) is not bi.int and py_type(v) is not bi.float:
        raise Error("model config: " + field + " requires a number")
    return py_call(bi.float, v)


def _config_fixed(data: PythonObject, names: List[String], values: List[Int]) raises:
    for i in range(len(names)):  # small-loop(fixed config field words)
        var got = py_get(data, names[i], PythonObject(values[i]))
        if not Bool(py=got == PythonObject(values[i])):
            _config_refuse(data, names[i], "fixed at " + String(values[i]))


def _config_transformer(data: PythonObject, mt: String, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    for field in [String("num_local_experts"), String("num_experts"), String("num_experts_per_tok"), String("n_routed_experts")]:  # small-loop(the four mixture-of-experts field names)
        var v = py_get(data, field)
        if v is not None and not Bool(py=v == PythonObject(0)) and not Bool(py=v == PythonObject(1)):
            _config_refuse(data, field, "mixture of experts")
    if py_get(data, "rope_scaling") is not None:
        _config_refuse(data, "rope_scaling", "rope type default only")
    var rt = py_get(data, "rope_type", PythonObject("default"))
    if rt is not None and String(rt) != "default":
        _config_refuse(data, "rope_type", "default only")
    var act = String(py_get(data, "hidden_act", PythonObject("silu")))
    if act != "silu" and act != "swish":
        _config_refuse(data, "hidden_act", "requires silu/swish SwiGLU")
    if Bool(py=py_get(data, "use_sliding_window", PythonObject(False))):
        _config_refuse(data, "use_sliding_window", "sliding-window-alternating layers (max_window_layers=" + String(py_call(bi.repr, py_get(data, "max_window_layers"))) + ")")
    var types = py_get(data, "layer_types")
    if types is not None and len(py_call(bi.set, types)) > 1:
        raise Error("model config: layer_types requests alternating attention kinds")
    var dm = Int(py=_config_int(data, "hidden_size"))
    var nh = Int(py=_config_int(data, "num_attention_heads"))
    var nk = Int(py=_config_int(data, "num_key_value_heads", PythonObject(nh)))
    if dm < 1 or nh < 1 or nk < 1:
        raise Error("model config: hidden_size and attention head counts must be positive")
    var hd_field = _config_int(data, "head_dim", PythonObject(None), False)
    var hd = dm // nh if hd_field is None else Int(py=hd_field)
    if nh * hd != dm:
        raise Error("model config: head_dim * num_attention_heads must equal hidden_size")
    if nh % nk != 0:
        _config_refuse(data, "num_key_value_heads", "num_attention_heads is not a multiple of it (must divide num_attention_heads)")
    if hd % 2 != 0:
        raise Error("model config: head_dim must be even")
    var inter = _config_int(data, "intermediate_size")
    var layers = _config_int(data, "num_hidden_layers")
    var vocab = _config_int(data, "vocab_size")
    var eps = _config_number(data, "rms_norm_eps", 1e-6)
    var theta = _config_number(data, "rope_theta", 10000.0)
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
    var maxpos = _config_int(data, "max_position_embeddings", PythonObject(8192))
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


def _config_mamba(data: PythonObject, mt: String, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var dm = Int(py=_config_int(data, "hidden_size"))
    var layers = _config_int(data, "num_hidden_layers")
    var vocab = _config_int(data, "vocab_size")
    var two = mt == "mamba2"
    _config_fixed(data, ["state_size", "conv_kernel", "expand", "use_bias", "use_conv_bias"],
                  [128 if two else 16, 4, 2, 0, 1])
    var kwargs = bi.dict()
    if two:
        _config_fixed(data, ["head_dim", "n_groups", "chunk_size", "rms_norm"], [64, 1, 256, 1])
        if Bool(py=py_get(data, "norm_before_gate", PythonObject(False))):
            raise Error("model config: norm_before_gate must be false")
        if dm % 32 != 0:
            _config_refuse(data, "hidden_size", "must be a multiple of 32")
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
    var eps = _config_number(data, "layer_norm_epsilon", 1e-5)
    if Float64(py=eps) != 1e-5:
        _config_refuse(data, "layer_norm_epsilon", "fixed at 1e-5")
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


def _config_phi3(mut fields: PythonObject, data: PythonObject) raises:
    var bi = Python.import_module("builtins")
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


def model_plan_for_binding(input: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var config = input
    if py_type(input) is bi.dict:
        config = py_call(bi.object.__new__, context["config_class"])
        _ = config_init_binding(config, input, PythonObject(None))
    var data = config._d
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
                _config_refuse(data, field, "unsupported for gemma2")
    if mt == "gemma" or mt == "gemma2":
        var field = "hidden_activation" if py_contains(data, "hidden_activation") else "hidden_act"
        _config_refuse(data, field, "unsupported GeGLU, RMSNorm offset and embedding scale")
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
        fields = _config_transformer(hf, mt, context)
        py_setitem(fields, "_layer_map", py_call(bi.list, context["byte_layer"]))
        py_setitem(fields, "embed_name", PythonObject("embed"))
        py_setitem(fields, "norm_name", PythonObject(None))
        py_setitem(fields, "head_name", PythonObject("lm_head"))
    elif mt == "mamba" or mt == "mamba2":
        fields = _config_mamba(data, mt, context)
    else:
        fields = _config_transformer(data, mt, context)
        if mt == "phi3":
            _config_phi3(fields, data)
    py_setitem(fields, "model_type", PythonObject(mt))
    py_setitem(fields, "config", config)
    py_setitem(fields, "family", row)
    var plan = py_call(bi.object.__new__, context["plan_class"])
    _ = model_plan_init_binding(plan, fields)
    return plan


def model_config_c[operation: Int](py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    var args = PythonObject(from_borrowed=args_ptr)
    try:
        var result = PythonObject(None)
        comptime if operation == 0:
            result = config_init_binding(args[0], args[1], args[2])
        elif operation == 1:
            result = config_load_binding(args[0], args[1])
        elif operation == 2:
            result = config_access_binding(args[0], args[1], args[2], args[3])
        elif operation == 3:
            result = model_plan_init_binding(args[0], args[1])
        elif operation == 4:
            result = model_layer_weights_binding(args[0], args[1])
        elif operation == 5:
            result = model_checkpoint_names_binding(args[0])
        elif operation == 6:
            result = model_plan_for_binding(args[0], args[1])
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
                ref cpy = Python().cpython()
                var cls = PyObjectPtr()
                comptime if operation == 6:
                    cls = causal_borrowed_item(cpy.PyTuple_GetItem(args_ptr, 1), "unsupported_class")
                else:
                    cls = cpy.PyTuple_GetItem(args_ptr, 4)
                return causal_set_exception(error, cls)
            except nested:
                return raise_python_exception(nested)
        else:
            return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
