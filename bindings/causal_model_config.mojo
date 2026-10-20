# SPDX-License-Identifier: Apache-2.0
"""Native configuration admission for the supported causal model families."""
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.causal_exception import causal_borrowed_item, causal_set_exception
from bindings.native_json import native_json_read, native_path, native_path_exists


def config_init_binding(value: PythonObject, data: PythonObject, path: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    if bi.type(data) is not bi.dict:
        raise Error("TypeError: HFConfig requires a JSON object")
    value._d = bi.dict(data)
    value.path = path
    return PythonObject(None)


def config_load_binding(cls: PythonObject, input_path: PythonObject) raises -> PythonObject:
    var path = native_path(input_path)
    if native_path_exists(path, True):
        path = path.rstrip("/") + PythonObject("/config.json")
    if not native_path_exists(path):
        raise Error("FileNotFoundError: config.json does not exist")
    var value = Python.import_module("builtins").object.__new__(cls)
    _ = config_init_binding(value, native_json_read(path), path)
    return value


def config_access_binding(config: PythonObject, operation: PythonObject, key: PythonObject, default: PythonObject) raises -> PythonObject:
    var op = Int(py=operation)
    if op == 0:
        var mt = config._d.get("model_type")
        if Python.import_module("builtins").type(mt) is not Python.import_module("builtins").str:
            raise Error("model config has no string model_type")
        return mt
    if op == 1:
        return config._d.get(key, default)
    if op == 2:
        if not Bool(py=config._d.__contains__(key)):
            raise Error("KeyError: " + String(Python.import_module("builtins").repr(key)))
        return config._d[key]
    if op == 3:
        return config._d.__contains__(key)
    if op == 4:
        return config._d.keys()
    return Python.import_module("builtins").dict(config._d)


def model_plan_init_binding(plan: PythonObject, fields: PythonObject) raises -> PythonObject:
    plan.__dict__.update(fields)
    return PythonObject(None)


def model_layer_weights_binding(plan: PythonObject, layer: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var out = bi.list()
    for j in range(len(plan._layer_map)):
        var entry = plan._layer_map[j]
        var item = bi.list()
        item.append(entry[0])
        item.append(PythonObject(String(entry[1]).replace("{i}", String(Int(py=layer)))))
        item.append(entry[2])
        out.append(bi.tuple(item))
    return out


def model_checkpoint_names_binding(plan: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var names = bi.list()
    names.append(plan.embed_name)
    if plan.norm_name is not None:
        names.append(plan.norm_name)
    if plan.head_name is not None:
        names.append(plan.head_name)
    for i in range(Int(py=plan.n_layers)):
        var layer = model_layer_weights_binding(plan, PythonObject(i))
        for j in range(len(layer)):
            var name = layer[j][1]
            if not Bool(py=names.__contains__(name)):
                names.append(name)
    return names


def _config_int(data: PythonObject, field: String, default: PythonObject, required: Bool = True) raises -> PythonObject:
    var v = data.get(field, default)
    var bi = Python.import_module("builtins")
    if v is None and not required:
        return v
    if bi.type(v) is not bi.int:
        raise Error("model config: " + field + " requires an integer")
    return v


def _config_int(data: PythonObject, field: String) raises -> PythonObject:
    return _config_int(data, field, PythonObject(None), True)


def _config_refuse(data: PythonObject, field: String, why: String) raises:
    var bi = Python.import_module("builtins")
    raise Error("model_type " + String(bi.repr(data.get("model_type", PythonObject("mojolearn-byte-lm")))) + ": " + field + "=" + String(bi.repr(data.get(field))) + " is not honored: " + why)


def _config_number(data: PythonObject, field: String, default: Float64) raises -> PythonObject:
    var v = data.get(field, PythonObject(default))
    var bi = Python.import_module("builtins")
    if bi.type(v) is not bi.int and bi.type(v) is not bi.float:
        raise Error("model config: " + field + " requires a number")
    return bi.float(v)


def _config_fixed(data: PythonObject, names: List[String], values: List[Int]) raises:
    for i in range(len(names)):
        var got = data.get(names[i], PythonObject(values[i]))
        if not Bool(py=got == PythonObject(values[i])):
            _config_refuse(data, names[i], "fixed at " + String(values[i]))


def _config_transformer(data: PythonObject, mt: String, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    for field in [String("num_local_experts"), String("num_experts"), String("num_experts_per_tok"), String("n_routed_experts")]:
        var v = data.get(field)
        if v is not None and not Bool(py=v == PythonObject(0)) and not Bool(py=v == PythonObject(1)):
            _config_refuse(data, field, "mixture of experts")
    if data.get("rope_scaling") is not None:
        _config_refuse(data, "rope_scaling", "rope type default only")
    var rt = data.get("rope_type", PythonObject("default"))
    if rt is not None and String(rt) != "default":
        _config_refuse(data, "rope_type", "default only")
    var act = String(data.get("hidden_act", PythonObject("silu")))
    if act != "silu" and act != "swish":
        _config_refuse(data, "hidden_act", "requires silu/swish SwiGLU")
    if Bool(py=data.get("use_sliding_window", PythonObject(False))):
        _config_refuse(data, "use_sliding_window", "alternating layers (max_window_layers=" + String(bi.repr(data.get("max_window_layers"))) + ")")
    var types = data.get("layer_types")
    if types is not None and len(bi.set(types)) > 1:
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
        _config_refuse(data, "num_key_value_heads", "must divide num_attention_heads")
    if hd % 2 != 0:
        raise Error("model config: head_dim must be even")
    var inter = _config_int(data, "intermediate_size")
    var layers = _config_int(data, "num_hidden_layers")
    var vocab = _config_int(data, "vocab_size")
    var eps = _config_number(data, "rms_norm_eps", 1e-6)
    var theta = _config_number(data, "rope_theta", 10000.0)
    var window = data.get("sliding_window")
    if window is None:
        window = PythonObject(0)
    if bi.type(window) is not bi.int or Int(py=window) < 0:
        raise Error("model config: sliding_window requires an integer width or null")
    var bias = Bool(py=data.get("attention_bias", PythonObject(mt == "qwen2")))
    var mlp_bias = Bool(py=data.get("mlp_bias", PythonObject(False)))
    var dropout = data.get("attention_dropout", PythonObject(0.0))
    if dropout is not None and Float64(py=dropout) != 0:
        raise Error("model config: attention_dropout must be zero for inference")
    var tied = Bool(py=data.get("tie_word_embeddings", PythonObject(False)))
    var maxpos = _config_int(data, "max_position_embeddings", PythonObject(8192))
    var options = bi.dict(context["defaults"])
    options["n_kv_heads"] = PythonObject(nk) if nk != nh else PythonObject(None)
    options["head_dim"] = PythonObject(hd) if hd != dm // nh else PythonObject(None)
    options["window"] = window
    options["rope_theta"] = theta
    options["norm_eps"] = eps
    options["qkv_bias"] = PythonObject(bias)
    options["o_bias"] = PythonObject(False)
    options["mlp_bias"] = PythonObject(mlp_bias)
    options["qk_norm"] = PythonObject(mt == "qwen3")
    var layer_map = bi.list(context["llama_layer"])
    if bias:
        layer_map.extend(context["qkv_bias"])
    if mt == "qwen3":
        layer_map.extend(context["qk_norm"])
    var fields = bi.dict()
    fields["kind"] = PythonObject("transformer")
    fields["n_layers"] = layers
    fields["d_model"] = PythonObject(dm)
    fields["n_heads"] = PythonObject(nh)
    fields["n_kv_heads"] = PythonObject(nk)
    fields["head_dim"] = PythonObject(hd)
    fields["intermediate"] = inter
    fields["vocab_size"] = vocab
    fields["tie_embeddings"] = PythonObject(tied)
    fields["norm_eps"] = eps
    fields["max_position_embeddings"] = maxpos
    fields["block_options"] = options
    fields["block_kwargs"] = bi.dict()
    fields["_layer_map"] = layer_map
    fields["embed_name"] = PythonObject("model.embed_tokens.weight")
    fields["norm_name"] = PythonObject("model.norm.weight")
    fields["head_name"] = PythonObject(None) if tied else PythonObject("lm_head.weight")
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
        if Bool(py=data.get("norm_before_gate", PythonObject(False))):
            raise Error("model config: norm_before_gate must be false")
        if dm % 32 != 0:
            _config_refuse(data, "hidden_size", "must be a multiple of 32")
        var nh = data.get("num_heads")
        if nh is not None and not Bool(py=nh == PythonObject(2 * dm // 64)):
            raise Error("model config: num_heads must equal expand*hidden_size/head_dim")
        var limit = data.get("time_step_limit")
        var bounds = bi.list()
        if limit is None:
            bounds.append(0.0)
            bounds.append(bi.float("inf"))
        else:
            bounds.append(bi.float(limit[0]))
            bounds.append(bi.float(limit[1]))
        kwargs["dt_limit"] = bi.tuple(bounds)
    else:
        var rank = data.get("time_step_rank", PythonObject("auto"))
        if not Bool(py=rank == PythonObject("auto")) and not Bool(py=rank == PythonObject((dm + 15) // 16)):
            raise Error("model config: time_step_rank must be auto or ceil(hidden_size/16)")
        if String(data.get("hidden_act", PythonObject("silu"))) != "silu":
            raise Error("model config: hidden_act requires silu")
    var eps = _config_number(data, "layer_norm_epsilon", 1e-5)
    if Float64(py=eps) != 1e-5:
        _config_refuse(data, "layer_norm_epsilon", "fixed at 1e-5")
    var tied = Bool(py=data.get("tie_word_embeddings", PythonObject(True)))
    var fields = bi.dict()
    fields["kind"] = PythonObject("mamba2" if two else "mamba1")
    fields["n_layers"] = layers
    fields["d_model"] = PythonObject(dm)
    fields["vocab_size"] = vocab
    fields["tie_embeddings"] = PythonObject(tied)
    fields["norm_eps"] = eps
    fields["max_position_embeddings"] = PythonObject(None)
    fields["block_options"] = bi.dict()
    fields["block_kwargs"] = kwargs
    fields["_layer_map"] = bi.list(context["mamba2_layer" if two else "mamba1_layer"])
    fields["embed_name"] = PythonObject("backbone.embeddings.weight")
    fields["norm_name"] = PythonObject("backbone.norm_f.weight")
    fields["head_name"] = PythonObject(None) if tied else PythonObject("lm_head.weight")
    return fields


def _config_phi3(mut fields: PythonObject, data: PythonObject) raises:
    var bi = Python.import_module("builtins")
    var original = data.get("original_max_position_embeddings")
    if original is not None and not Bool(py=original == fields["max_position_embeddings"]):
        raise Error("model config: original_max_position_embeddings requests longrope extended context")
    var qr = Int(py=fields["n_heads"]) * Int(py=fields["head_dim"])
    var kr = Int(py=fields["n_kv_heads"]) * Int(py=fields["head_dim"])
    var it = Int(py=fields["intermediate"])
    var layers = fields["_layer_map"]
    var mapped = bi.list()
    for i in range(len(layers)):
        var item = bi.list(layers[i])
        var key = String(item[0])
        var begin = -1
        var end = -1
        if key == "q_proj.weight" or key == "k_proj.weight" or key == "v_proj.weight":
            item[1] = PythonObject("model.layers.{i}.self_attn.qkv_proj.weight")
            begin = 0 if key == "q_proj.weight" else (qr if key == "k_proj.weight" else qr + kr)
            end = qr if key == "q_proj.weight" else (qr + kr if key == "k_proj.weight" else qr + 2 * kr)
        elif key == "gate_proj.weight" or key == "up_proj.weight":
            item[1] = PythonObject("model.layers.{i}.mlp.gate_up_proj.weight")
            begin = 0 if key == "gate_proj.weight" else it
            end = it if key == "gate_proj.weight" else 2 * it
        if begin >= 0:
            var bounds = bi.list()
            bounds.append(begin)
            bounds.append(end)
            item[2] = bi.tuple(bounds)
        mapped.append(bi.tuple(item))
    fields["_layer_map"] = mapped


def model_plan_for_binding(input: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var config = input
    if bi.type(input) is bi.dict:
        config = bi.object.__new__(context["config_class"])
        _ = config_init_binding(config, input, PythonObject(None))
    var data = config._d
    var mt_value = data.get("model_type")
    var byte_lm = mt_value is None and Bool(py=data.get("format") == PythonObject("mojolearn.byte-lm"))
    if not byte_lm and bi.type(mt_value) is not bi.str:
        raise Error("model config has no string model_type")
    var mt = "mojolearn-byte-lm" if byte_lm else String(mt_value)
    var row = context["byte_row"] if byte_lm else context["families"].get(mt)
    if row is None:
        raise Error("model_type '" + mt + "' is not in the option matrix")
    if mt == "gemma2":
        for field in [String("attn_logit_softcapping"), String("final_logit_softcapping"), String("query_pre_attn_scalar")]:
            if data.get(field) is not None:
                _config_refuse(data, field, "unsupported for gemma2")
    if mt == "gemma" or mt == "gemma2":
        var field = "hidden_activation" if Bool(py=data.__contains__("hidden_activation")) else "hidden_act"
        _config_refuse(data, field, "unsupported GeGLU, RMSNorm offset and embedding scale")
    var fields = PythonObject(None)
    if byte_lm:
        var fixed = context["byte_fixed"]
        var names = bi.list(fixed)
        for i in range(len(names)):
            var name = names[i]
            if Bool(py=data.__contains__(name)) and not Bool(py=data[name] == fixed[name]):
                raise Error("model config: byte-LM " + String(name) + " differs from required value")
        var hf = bi.dict()
        var targets = [String("hidden_size"), String("num_attention_heads"), String("num_key_value_heads"), String("head_dim"), String("intermediate_size"), String("num_hidden_layers"), String("vocab_size"), String("rms_norm_eps"), String("rope_theta"), String("max_position_embeddings")]
        var sources = [String("d_model"), String("n_heads"), String("n_kv"), String("head_dim"), String("intermediate"), String("n_layers"), String("vocab_size"), String("rms_norm_eps"), String("rope_theta"), String("max_position_embeddings")]
        for i in range(len(targets)):
            var v = data.get(sources[i])
            if v is not None:
                hf[targets[i]] = v
        hf["tie_word_embeddings"] = PythonObject(False)
        fields = _config_transformer(hf, mt, context)
        fields["_layer_map"] = bi.list(context["byte_layer"])
        fields["embed_name"] = PythonObject("embed")
        fields["norm_name"] = PythonObject(None)
        fields["head_name"] = PythonObject("lm_head")
    elif mt == "mamba" or mt == "mamba2":
        fields = _config_mamba(data, mt, context)
    else:
        fields = _config_transformer(data, mt, context)
        if mt == "phi3":
            _config_phi3(fields, data)
    fields["model_type"] = PythonObject(mt)
    fields["config"] = config
    fields["family"] = row
    var plan = bi.object.__new__(context["plan_class"])
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
            return raise_python_exception(error, ExceptionType.TypeError)
        if message.startswith("FileNotFoundError:"):
            return raise_python_exception(error, ExceptionType("PyExc_FileNotFoundError"))
        if message.startswith("KeyError:"):
            return raise_python_exception(error, ExceptionType("PyExc_KeyError"))
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
            return raise_python_exception(error, ExceptionType.ValueError)
