# SPDX-License-Identifier: Apache-2.0
"""Native checkpoint-to-model ownership over the existing compiled kernels.

Module connection metadata is supplied once by package import. Runtime route,
profile, layout, shape admission, packing and layer construction live here.
"""
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from std.math import isfinite
from std.memory import memcpy, bitcast
from std.os import getenv
from bindings.generation_buffers import GENERATION_BOUNDARY, generation_initialize_binding, generation_array, generation_address, py_ints
from bindings.causal_lm_public import causal_lm_admit_binding, _native_function
from bindings.causal_model_config import config_load_binding, model_plan_for_binding, model_checkpoint_names_binding, model_layer_weights_binding
from bindings.causal_checkpoint import checkpoint_array, checkpoint_open_binding, checkpoint_index_binding, checkpoint_read_binding, checkpoint_close_binding
from bindings.native_json import native_path, native_path_exists
from bindings.hostptr import f32_ptr


def _model_shape(value: PythonObject, shape: List[Int], name: String) raises:
    if len(value.shape) != len(shape):
        raise Error("CausalLM: " + name + " has the wrong rank")
    for i in range(len(shape)):  # driver-loop: over tensor shape metadata
        if Int(py=value.shape[i]) != shape[i]:
            raise Error("CausalLM: " + name + " has the wrong shape")


def _model_f32(value: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    if bi.type(value) is context["array_class"]:
        if String(value.dtype) != "<f4":
            raise Error("TypeError: CausalLM tensor " + name + " must be float32")
        if String(value.order) == "C":
            return value
        var dims = List[Int]()
        for i in range(len(value.shape)):  # driver-loop: over tensor shape metadata
            dims.append(Int(py=value.shape[i]))
        var result = generation_array(dims)
        var source = f32_ptr(Int(py=value._addr))
        var target = f32_ptr(Int(py=result._addr))
        for i in range(Int(py=value.size)):
            var remaining = i
            var source_index = 0
            for j in range(len(dims)):
                var axis = len(dims) - 1 - j
                var coordinate = remaining % dims[axis]
                remaining //= dims[axis]
                source_index += coordinate * Int(py=value.strides[axis]) // 4
            target[i] = source[source_index]
        return result
    var view = bi.memoryview(value)
    var format = String(view.format)
    if format != "f" and format != "<f" and format != "=f" and format != "@f":
        raise Error("TypeError: CausalLM tensor " + name + " must be float32")
    return checkpoint_array(view.tobytes(), view.shape, "f", context["array_class"])


def _model_native_buffer(value: PythonObject, code: String, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    if bi.type(value) is context["array_class"]:
        var dtype = "<u2" if code == "H" else ("<i1" if code == "b" else "<i4")
        if String(value.dtype) != dtype or String(value.order) != "C":
            raise Error("TypeError: CausalLM packed storage requires matching dtype and C layout")
        return value
    var view = bi.memoryview(value)
    if String(view.format) != code:
        raise Error("TypeError: CausalLM packed storage has the wrong dtype")
    return checkpoint_array(view.tobytes(), view.shape, code, context["array_class"])


def _model_pack(value: PythonObject, format: String, linalg: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    if format == "float32":
        return value
    if len(value.shape) != 2:
        raise Error("CausalLM: packed projection weights require two dimensions")
    var rows = Int(py=value.shape[0])
    var cols = Int(py=value.shape[1])
    var packed = bi.object.__new__(context["bf16_class"] if format == "bfloat16" else context["int8_class"])
    packed.shape = value.shape
    if format == "bfloat16":
        var bits = checkpoint_array(bi.bytes(rows * cols * 2), value.shape, "H", context["array_class"])
        _native_function(linalg, "to_bf16")(bits._addr, value._addr, py_ints([rows * cols]))
        packed.bits = bits
    else:
        var codes = generation_array([rows, cols], "b")
        var exponents = generation_array([rows], "i")
        _native_function(linalg, "quantize_int8")(codes._addr, exponents._addr, value._addr, py_ints([rows, cols]))
        packed.codes = codes
        packed.exponents = exponents
    return packed


def _model_unpack(value: PythonObject, linalg: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var bf16 = bi.type(value) is context["bf16_class"]
    var int8 = bi.type(value) is context["int8_class"]
    if not bf16 and not int8:
        return _model_f32(value, "weight", context)
    if len(value.shape) != 2:
        raise Error("CausalLM: packed projection weights require two dimensions")
    var rows = Int(py=value.shape[0])
    var cols = Int(py=value.shape[1])
    var out = generation_array([rows, cols])
    if bf16:
        var bits = _model_native_buffer(value.bits, "H", context)
        _model_shape(bits, [rows, cols], "bfloat16 bits")
        _native_function(linalg, "from_bf16")(out._addr, bits._addr, py_ints([rows * cols]))
    else:
        var codes = _model_native_buffer(value.codes, "b", context)
        var exponents = _model_native_buffer(value.exponents, "i", context)
        _model_shape(codes, [rows, cols], "int8 codes")
        _model_shape(exponents, [rows], "int8 exponents")
        _native_function(linalg, "dequantize_int8")(out._addr, codes._addr, exponents._addr, py_ints([rows, cols]))
    return out


def _model_quantize15(value: PythonObject, linalg: PythonObject) raises -> PythonObject:
    var rows = Int(py=value.shape[0])
    var cols = Int(py=value.shape[1])
    var hi = generation_array([rows, cols], "b")
    var lo = generation_array([rows, cols], "b")
    var exponents = generation_array([rows], "i")
    _native_function(linalg, "quantize_int15")(py_ints([Int(py=hi._addr), Int(py=lo._addr), Int(py=exponents._addr), Int(py=value._addr)]), py_ints([rows, cols]))
    var out = Python.import_module("builtins").list()
    out.append(hi)
    out.append(lo)
    out.append(exponents)
    return Python.import_module("builtins").tuple(out)


def _model_profile(requested: PythonObject, family: String, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var profiles = context["profiles"]
    var defaults = context["profile_defaults"]
    var selected = requested
    if selected is None:
        selected = defaults.get("inference")
        if selected is None:
            var env = getenv("MOJOLEARN_NUMERIC_PROFILE", "")
            selected = PythonObject(env).strip()
            if len(selected) == 0:
                selected = PythonObject("fp32_v1")
    if bi.type(selected) is not bi.str:
        raise Error("TypeError: numeric_profile must be a profile name")
    selected = selected.strip()
    if not Bool(py=profiles.__contains__(selected)):
        var reason = context["rejected_profiles"].get(selected)
        raise Error("numeric_profile=" + String(selected) + " is not offered: " + (String(reason) if reason is not None else String("not a registered numeric profile")))
    var row = profiles[selected]
    if not Bool(py=row["inference"]):
        raise Error("NotImplementedError: numeric_profile=" + String(selected) + " is unavailable for inference")
    if requested is None and defaults.get("inference") is None:
        defaults["inference"] = selected
    var computes = row["computes"]
    var covered = computes is None or Bool(py=computes.__contains__(family))
    if requested is None:
        var reaches = row["default_for"]
        if not covered or (reaches is not None and not Bool(py=reaches.__contains__(family))):
            return PythonObject("fp32_v1")
    elif not covered:
        raise Error("NotImplementedError: numeric_profile=" + String(selected) + " does not compute the " + family + " family")
    return selected


def _model_route(device: PythonObject, context: PythonObject) raises -> PythonObject:
    var route = String(device)
    if route == "auto":
        route = "cpu" if context["selected_vendor"] is None or String(context["selected_vendor"]) == "cpu" else "gpu"
    if route != "cpu" and route != "gpu":
        raise Error("CausalLM: device must be 'auto', 'cpu' or 'gpu'")
    return PythonObject(route)


def _model_bindings(route: PythonObject, kind: String, context: PythonObject, requested_mode: PythonObject) raises -> PythonObject:
    var mode = PythonObject("identical") if String(route) == "cpu" else (context["backend"]._DEFAULT_MODE if requested_mode is None else requested_mode.strip().lower())
    var key = route + PythonObject(":") + mode
    var entries = context["bindings"].get(key)
    if entries is None:
        raise Error("CausalLM: native binding connection is unavailable for " + String(key))
    var block = entries[kind]
    if Python.import_module("builtins").type(block) is Python.import_module("builtins").str:
        raise Error("CausalLM: " + String(block))
    var prefix = ("transformer" if kind == "transformer" else "mamba") if String(route) == "gpu" else "neural_host"
    var vendor = String(_native_function(block, prefix + "_vendor")())
    var mode_code = Int(py=_native_function(block, prefix + "_numeric_mode")())
    if String(route) == "cpu" and vendor != "cpu":
        raise Error("CausalLM: CPU route requires native CPU kernels")
    if String(route) == "gpu" and vendor != "metal" and vendor != "cuda" and vendor != "hip":
        raise Error("CausalLM: GPU route requires native GPU kernels")
    if mode_code != (1 if String(mode) == "identical" else 0):
        raise Error("CausalLM: binding numeric mode differs from requested mode")
    var arithmetic = entries["linalg"]
    if Python.import_module("builtins").type(arithmetic) is not Python.import_module("builtins").str:
        if String(_native_function(arithmetic, "linalg_vendor")()) != vendor or Int(py=_native_function(arithmetic, "linalg_numeric_mode")()) != mode_code:
            raise Error("CausalLM: packed weight binding vendor or mode differs from model")
    return entries


def _model_bindings(route: PythonObject, kind: String, context: PythonObject) raises -> PythonObject:
    return _model_bindings(route, kind, context, PythonObject(None))


def _model_bits(value: PythonObject, name: String) raises -> Int:
    var number = Float64(py=Python.import_module("builtins").float(value))
    if not isfinite(number) or number <= 0:
        raise Error("CausalLM: " + name + " must be finite and positive")
    var rounded = Float32(number)
    if not isfinite(rounded):
        raise Error("CausalLM: " + name + " exceeds float32 range")
    return Int(bitcast[DType.uint32](rounded))


def _model_block(weights: PythonObject, plan: PythonObject, profile: PythonObject, route: PythonObject, bindings: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var kind = String(plan.kind)
    var cls = context["block_classes"][route][kind]
    var block = bi.object.__new__(cls)
    var fields = bi.object.__getattribute__(block, "__dict__")
    fields["numeric_mode"] = PythonObject("identical") if String(route) == "cpu" else context["backend"]._DEFAULT_MODE
    var names = cls._W_NAMES
    var w = bi.dict()
    var keys = bi.list(weights)
    var format = "float32"
    for i in range(len(keys)):  # driver-loop: over the layer's named tensors
        var v = weights[keys[i]]
        var tag = "bfloat16" if bi.type(v) is context["bf16_class"] else ("int8" if bi.type(v) is context["int8_class"] else "float32")
        if tag != "float32":
            if format != "float32" and format != tag:
                raise Error("CausalLM: mixed packed weight formats")
            format = tag
        w[keys[i]] = _model_unpack(v, bindings["linalg"], context)
    fields["weight_format"] = PythonObject(format)
    fields["_native_kind"] = plan.kind
    fields["_native_route"] = route
    var dm = Int(py=plan.d_model)
    var base = bi.list()
    var optional = bi.list()
    var expected = bi.set()
    var shapes = bi.list()
    var gated = True
    var it = 0
    if kind == "transformer":
        var options = bi.dict(context["config_context"]["defaults"])
        options.update(plan.block_options)
        var nh = Int(py=plan.n_heads)
        if nh < 1:
            raise Error("CausalLM: n_heads must be positive")
        var nkv = nh if options["n_kv_heads"] is None else Int(py=options["n_kv_heads"])
        var hd = dm // nh if options["head_dim"] is None else Int(py=options["head_dim"])
        if nh < 1 or nkv < 1 or hd < 1 or dm != nh * hd:
            raise Error("CausalLM: d_model must equal positive n_heads * head_dim")
        var mlp = String(options["mlp"])
        var norm = String(options["norm"])
        var nk = 0 if norm == "rmsnorm" else (1 if norm == "layernorm" else (2 if norm == "rmsnorm_offset" else -1))
        var mk = 0 if mlp == "swiglu" else (1 if mlp == "gelu" else (2 if mlp == "gelu_tanh" else (3 if mlp == "geglu" else (4 if mlp == "geglu_tanh" else -1))))
        if nk < 0 or mk < 0:
            raise Error("CausalLM: unsupported norm or mlp kind")
        gated = mk == 0 or mk == 3 or mk == 4
        var source = w["gate_proj.weight" if gated else "up_proj.weight"]
        if len(source.shape) != 2 or Int(py=source.shape[0]) < 1:
            raise Error("CausalLM: projection must have positive intermediate size")
        it = Int(py=source.shape[0])
        var tail = List[Int](length=17, fill=0)
        tail[0] = _model_bits(options["rope_theta"], "rope_theta")
        tail[2] = 0x3f800000
        tail[3] = 0x3f800000
        tail[4] = 0x40800000
        tail[5] = 8192
        var scaling = options["rope_scaling"]
        if scaling is not None:
            var sk = String(scaling.get("type", scaling.get("rope_type")))
            if sk != "linear" and sk != "llama3":
                raise Error("CausalLM: unsupported rope_scaling type")
            tail[1] = 1 if sk == "linear" else 2
            tail[2] = _model_bits(scaling["factor"], "rope_scaling factor")
            if sk == "llama3":
                tail[3] = _model_bits(scaling["low_freq_factor"], "low_freq_factor")
                tail[4] = _model_bits(scaling["high_freq_factor"], "high_freq_factor")
                tail[5] = Int(py=scaling["original_max_position_embeddings"])
                if Float64(py=scaling["high_freq_factor"]) <= Float64(py=scaling["low_freq_factor"]) or tail[5] < 1:
                    raise Error("CausalLM: invalid llama3 rope_scaling bounds")
        var rd = hd if options["rope_dim"] is None else Int(py=options["rope_dim"])
        if rd < 1 or rd > hd or rd % 2 != 0:
            raise Error("CausalLM: rope_dim must be positive even and no larger than head_dim")
        tail[6] = 0 if rd == hd else rd
        tail[7] = Int(py=options["max_positions"])
        tail[8] = Int(Bool(py=options["qkv_bias"]))
        tail[9] = Int(Bool(py=options["o_bias"]))
        tail[10] = nk
        tail[11] = _model_bits(options["norm_eps"], "norm_eps")
        tail[12] = Int(Bool(py=options["norm_bias"]))
        tail[13] = mk
        tail[14] = Int(Bool(py=options["mlp_bias"]))
        tail[15] = Int(Bool(py=options["qk_norm"]))
        tail[16] = 0 if options["attn_softcap"] is None else _model_bits(options["attn_softcap"], "attn_softcap")
        if tail[7] < 1 or (tail[12] != 0 and nk != 1) or (tail[15] != 0 and nk == 1):
            raise Error("CausalLM: invalid max_positions, norm_bias or qk_norm combination")
        var window = Int(py=options["window"])
        if window < 0:
            raise Error("CausalLM: sliding window must be nonnegative")
        fields.update(options)
        fields["d_model"] = PythonObject(dm)
        fields["n_heads"] = PythonObject(nh)
        fields["n_kv_heads"] = PythonObject(nkv)
        fields["head_dim"] = PythonObject(hd)
        fields["intermediate"] = PythonObject(it)
        fields["rope_dim"] = PythonObject(rd)
        fields["_opts_tail"] = py_ints(tail)
        var default_tail = [0x461c4000, 0, 0x3f800000, 0x3f800000, 0x40800000, 8192, 0, 8192, 0, 0, 0, 0x358637bd, 0, 0, 0, 0, 0]
        var extended = False
        for i in range(17):  # driver-loop: over the fixed block options record
            extended = extended or tail[i] != default_tail[i]
        fields["_extended"] = PythonObject(extended)
        fields["_gated"] = PythonObject(gated)
        fields["_runtime_lock"] = context["rlock"]()
        fields["_native_session"] = PythonObject(None)
        fields["_session_binding"] = PythonObject(None)
        fields["_arithmetic_profile"] = PythonObject(None)
        if Bool(py=bi.hasattr(bindings[kind], "transformer_arithmetic_profile")):
            fields["_arithmetic_profile"] = _native_function(bindings[kind], "transformer_arithmetic_profile")()
            fields["_arithmetic_profile_mode"] = fields["numeric_mode"]
            fields["_arithmetic_profile_changed"] = _native_function(bindings[kind], "transformer_arithmetic_profile_changed")()
        fields["numeric_profile"] = profile
        shapes.append(py_ints([dm]))
        shapes.append(py_ints([dm]))
        shapes.append(py_ints([nh * hd, dm]))
        shapes.append(py_ints([nkv * hd, dm]))
        shapes.append(py_ints([nkv * hd, dm]))
        shapes.append(py_ints([dm, nh * hd]))
        shapes.append(py_ints([it, dm]))
        shapes.append(py_ints([it, dm]))
        shapes.append(py_ints([dm, it]))
        var flags = [tail[8], tail[8], tail[8], tail[9], tail[12], tail[12], tail[14], tail[14], tail[14] if gated else 0, tail[15], tail[15]]
        var widths = [nh * hd, nkv * hd, nkv * hd, dm, dm, dm, it, dm, it, hd, hd]
        var flag_dict = bi.dict()
        for i in range(11):  # driver-loop: over optional block tensors
            var name = context["optional_names"][i]
            flag_dict[name] = PythonObject(flags[i] != 0)
            var present = Bool(py=w.__contains__(name))
            if present != (flags[i] != 0):
                raise Error("CausalLM: " + String(name) + " presence does not match its option")
            if present:
                _model_shape(w[name], [widths[i]], String(name))
                optional.append(w[name])
                expected.add(name)
            else:
                optional.append(PythonObject(None))
        fields["_opts_flags"] = flag_dict
        fields["_wopt"] = optional
    else:
        var di = 2 * dm
        fields["d_model"] = PythonObject(dm)
        fields["d_inner"] = PythonObject(di)
        if kind == "mamba1":
            var rank = (dm + 15) // 16
            fields["dt_rank"] = PythonObject(rank)
            shapes.append(py_ints([dm]))
            shapes.append(py_ints([2 * di, dm]))
            shapes.append(py_ints([di, 1, 4]))
            shapes.append(py_ints([di]))
            shapes.append(py_ints([rank + 32, di]))
            shapes.append(py_ints([di, rank]))
            shapes.append(py_ints([di]))
            shapes.append(py_ints([di, 16]))
            shapes.append(py_ints([di]))
            shapes.append(py_ints([dm, di]))
        else:
            if dm % 32 != 0:
                raise Error("CausalLM: Mamba2 d_model must be a multiple of 32")
            var nh = di // 64
            var cd = di + 256
            fields["nheads"] = PythonObject(nh)
            fields["conv_dim"] = PythonObject(cd)
            var defaults = bi.list()
            defaults.append(0.0)
            defaults.append(bi.float("inf"))
            var limit = plan.block_kwargs.get("dt_limit", defaults)
            var pair = bi.list()
            pair.append(bi.float(limit[0]))
            pair.append(bi.float(limit[1]))
            fields["dt_limit"] = bi.tuple(pair)
            shapes.append(py_ints([dm]))
            shapes.append(py_ints([2 * di + 256 + nh, dm]))
            shapes.append(py_ints([cd, 1, 4]))
            shapes.append(py_ints([cd]))
            shapes.append(py_ints([nh]))
            shapes.append(py_ints([nh]))
            shapes.append(py_ints([nh]))
            shapes.append(py_ints([di]))
            shapes.append(py_ints([dm, di]))
    for i in range(len(names)):  # driver-loop: over fixed block weight slots
        var name = names[i]
        if kind == "transformer" and i == 6 and not gated:
            base.append(PythonObject(None))
            continue
        if not Bool(py=w.__contains__(name)):
            raise Error("CausalLM: missing block tensor " + String(name))
        var value = w[name]
        var shape = List[Int]()
        for j in range(len(shapes[i])):  # driver-loop: over tensor shape metadata
            shape.append(Int(py=shapes[i][j]))
        if kind != "transformer" and i == 2 and len(value.shape) == 2:
            shape = [shape[0], 4]
        _model_shape(value, shape, String(name))
        base.append(value)
        expected.add(name)
    if len(expected) != len(w):
        raise Error("CausalLM: unexpected block tensor names")
    fields["_w"] = base
    fields["_int15"] = PythonObject(None)
    if String(profile) == "fixed15_v1":
        var planes = bi.list()
        for i in range(2, 9):  # driver-loop: over the seven transformer projections
            planes.append(PythonObject(None) if base[i] is None else _model_quantize15(base[i], bindings["linalg"]))
        fields["_int15"] = planes
    return block


def causal_model_init_binding(model: PythonObject, plan: PythonObject, weights: PythonObject, options: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    _ = generation_initialize_binding(context["generation_classes"])
    var format = String(options[0])
    if format != "float32" and format != "bfloat16" and format != "int8":
        raise Error("CausalLM: unsupported weight_format " + format)
    var route = _model_route(options[2], context)
    var profile = _model_profile(options[3], String(plan.kind), context)
    var bindings = _model_bindings(route, String(plan.kind), context)
    model._runtime = context["runtime"]
    model.plan = plan
    model.config = plan.config
    model.kind = plan.kind
    model.d_model = plan.d_model
    model.vocab_size = plan.vocab_size
    model.n_layers = plan.n_layers
    model.norm_eps = plan.norm_eps
    model.numeric_profile = profile
    model.device = route
    model.weight_format = options[0]
    model.unused_names = bi.list(weights.get("unused", bi.tuple()))
    var dm = Int(py=plan.d_model)
    var vocab = Int(py=plan.vocab_size)
    if dm < 1 or vocab < 1 or Int(py=plan.n_layers) < 1:
        raise Error("CausalLM: model dimensions and layer count must be positive")
    var positions = options[1]
    if String(plan.kind) == "transformer":
        var config_cap = 8192 if plan.max_position_embeddings is None else Int(py=plan.max_position_embeddings)
        var cap = min(config_cap, 8192)
        if positions is None:
            positions = PythonObject(cap)
        var mp = Int(py=bi.int(positions))
        if mp < 1:
            raise Error("CausalLM: max_positions must be positive")
        if mp > config_cap or mp > 8192:
            raise Error("UnsupportedModel: max_positions exceeds max_position_embeddings or transformer ceiling 8192")
        positions = PythonObject(mp)
    elif positions is not None:
        positions = bi.int(positions)
    model.max_positions = positions
    model._embed = _model_f32(weights["embed"], String(plan.embed_name), context)
    _model_shape(model._embed, [vocab, dm], String(plan.embed_name))
    model._norm = PythonObject(None)
    if plan.norm_name is not None:
        model._norm = _model_f32(weights["norm"], String(plan.norm_name), context)
        _model_shape(model._norm, [dm], String(plan.norm_name))
    model._head = model._embed if plan.head_name is None else _model_f32(weights["head"], String(plan.head_name), context)
    _model_shape(model._head, [vocab, dm], "head")
    for value in [model._embed, model._norm, model._head]:  # driver-loop: over the three top-level tensors
        if value is not None:
            var ptr = f32_ptr(Int(py=value._addr))
            for i in range(Int(py=value.size)):
                if not isfinite(ptr[i]):
                    raise Error("CausalLM: top-level weight is not finite")
    var layers = weights["layers"]
    if len(layers) != Int(py=plan.n_layers):
        raise Error("CausalLM: layer dict count differs from num_hidden_layers")
    model._blocks = bi.list()
    for i in range(len(layers)):  # driver-loop: over the model layer definitions
        var block = _model_block(layers[i], plan, profile, route, bindings, context)
        var fields = bi.object.__getattribute__(block, "__dict__")
        if String(fields["weight_format"]) != format:
            raise Error("CausalLM: layer reports a different weight_format than requested")
        model._blocks.append(block)
    model._block_class = context["block_classes"][route][plan.kind]
    model._block_kwargs = bi.dict(plan.block_options if String(plan.kind) == "transformer" else plan.block_kwargs)
    if String(plan.kind) == "transformer":
        model._block_kwargs["n_heads"] = plan.n_heads
        model._block_kwargs["numeric_profile"] = profile
    model._prims = bi.object.__new__(context["primitive_classes"][route])
    if String(route) == "gpu":
        model._prims._T = context["training_module"]
    else:
        model._prims._ext = bindings["primitives"]
    model._head_int15 = PythonObject(None)
    if String(profile) == "fixed15_v1":
        model._head_int15 = _model_quantize15(model._head, bindings["linalg"])
    _ = causal_lm_admit_binding(model, bindings[plan.kind], bindings["primitives"], bindings["linalg"] if model._head_int15 is not None else PythonObject(None))
    return PythonObject(None)


def causal_model_read_weights_binding(checkpoint: PythonObject, plan: PythonObject, format: PythonObject, route: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    _ = generation_initialize_binding(context["generation_classes"])
    var ck = context["checkpoint_context"]
    var infos = checkpoint_index_binding(checkpoint, ck)
    var need = model_checkpoint_names_binding(plan)
    for i in range(len(need)):  # driver-loop: over checkpoint tensor names
        if not Bool(py=infos.__contains__(need[i])):
            raise Error("UnsupportedModel: checkpoint lacks tensor " + String(need[i]))
    var out = bi.dict()
    out["embed"] = checkpoint_read_binding(checkpoint, plan.embed_name, PythonObject("widen"), ck)
    if plan.norm_name is not None:
        out["norm"] = checkpoint_read_binding(checkpoint, plan.norm_name, PythonObject("widen"), ck)
    if plan.head_name is not None:
        out["head"] = checkpoint_read_binding(checkpoint, plan.head_name, PythonObject("widen"), ck)
    var bindings = _model_bindings(route, String(plan.kind), context)
    var projections = context["projections"][plan.kind]
    var layers = bi.list()
    for i in range(Int(py=plan.n_layers)):  # driver-loop: over checkpoint layers
        var weights = bi.dict()
        var mapping = model_layer_weights_binding(plan, PythonObject(i))
        for j in range(len(mapping)):  # driver-loop: over named layer tensor entries
            var entry = mapping[j]
            var key = entry[0]
            var name = entry[1]
            var rows = entry[2]
            var packed = String(format) != "float32" and Bool(py=projections.__contains__(key))
            if packed and String(format) == "bfloat16" and String(infos[name].dtype) == "BF16" and rows is None and len(infos[name].shape) == 2:
                weights[key] = checkpoint_read_binding(checkpoint, name, PythonObject("bits"), ck)
                continue
            var value = _model_f32(checkpoint_read_binding(checkpoint, name, PythonObject("widen"), ck), String(name), context)
            if rows is not None:
                var begin = Int(py=rows[0])
                var end = Int(py=rows[1])
                if len(value.shape) != 2 or begin < 0 or end < begin or end > Int(py=value.shape[0]):
                    raise Error("UnsupportedModel: checkpoint tensor " + String(name) + " is too small for projection row slice")
                var cols = Int(py=value.shape[1])
                var sliced = generation_array([end - begin, cols])
                if end > begin:
                    memcpy(dest=f32_ptr(Int(py=sliced._addr)), src=f32_ptr(Int(py=value._addr)) + begin * cols, count=(end - begin) * cols)
                value = sliced
            weights[key] = _model_pack(value, String(format), bindings["linalg"], context) if packed else value
        layers.append(weights)
    out["layers"] = layers
    var unused = bi.list()
    var names = bi.list(infos)
    for i in range(len(names)):  # driver-loop: over checkpoint metadata names
        if not Bool(py=need.__contains__(names[i])):
            unused.append(names[i])
    out["unused"] = unused
    return out


def causal_model_load_binding(cls: PythonObject, path_value: PythonObject, options: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var path = native_path(path_value)
    if not native_path_exists(path, True):
        raise Error("FileNotFoundError: CausalLM.load requires a checkpoint directory")
    var config = config_load_binding(context["config_context"]["config_class"], path)
    var plan = model_plan_for_binding(config, context["config_context"])
    _ = _model_profile(options[3], String(plan.kind), context)
    var route = _model_route(options[2], context)
    var ck = context["checkpoint_context"]
    var checkpoint = checkpoint_open_binding(ck["checkpoint_class"], path, ck)
    var weights = PythonObject(None)
    try:
        weights = causal_model_read_weights_binding(checkpoint, plan, options[0], route, context)
    except error:
        _ = checkpoint_close_binding(checkpoint)
        raise error
    _ = checkpoint_close_binding(checkpoint)
    var model = bi.object.__new__(cls)
    _ = causal_model_init_binding(model, plan, weights, options, context)
    return model


def causal_model_c[operation: Int](py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    var args = PythonObject(from_borrowed=args_ptr)
    try:
        var result = PythonObject(None)
        comptime if operation == 0:
            result = causal_model_init_binding(args[0], args[1], args[2], args[3], args[4])
        elif operation == 1:
            result = causal_model_load_binding(args[0], args[1], args[2], args[3])
        else:
            result = causal_model_read_weights_binding(args[0], args[1], args[2], _model_route(PythonObject("auto"), args[3]), args[3])
        return result.steal_data()
    except error:
        var message = String(error)
        if message.startswith("TypeError:"):
            return raise_python_exception(error, ExceptionType.TypeError)
        if message.startswith("FileNotFoundError:"):
            return raise_python_exception(error, ExceptionType("PyExc_FileNotFoundError"))
        if message.startswith("NotImplementedError:"):
            return raise_python_exception(error, ExceptionType("PyExc_NotImplementedError"))
        if message.startswith("UnsupportedModel:"):
            try:
                var context = args[4] if operation == 0 else args[3]
                var cls = context["config_context"]["unsupported_class"]
                var ptr = cls.steal_data()
                ref cpy = Python().cpython()
                cpy.PyErr_SetString(ptr, message.as_c_string_slice().unsafe_ptr().as_unsafe_any_origin())
                cpy.Py_DecRef(ptr)
                return PyObjectPtr()
            except nested:
                return raise_python_exception(nested)
        return raise_python_exception(error, ExceptionType.ValueError)
