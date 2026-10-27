# SPDX-License-Identifier: Apache-2.0
"""Native checkpoint-to-model ownership over the existing compiled kernels.

Module connection metadata is supplied once by package import. Runtime route,
profile, layout, shape admission, packing and layer construction live here.
"""
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_contains, py_get, py_setattr, py_setitem, py_type
from bindings.causal_exception import causal_borrowed_item, causal_set_exception
from std.math import isfinite
from std.memory import memcpy, bitcast
from std.os import getenv
from bindings.generation_buffers import GENERATION_BOUNDARY, generation_initialize_binding, generation_array, generation_address, py_ints
from bindings.causal_lm_public import causal_lm_admit_binding, _native_function
from bindings.causal_model_config import config_load_binding, model_plan_for_binding, model_checkpoint_names_binding, model_layer_weights_binding
from bindings.causal_checkpoint import checkpoint_array, checkpoint_open_binding, checkpoint_index_binding, checkpoint_read_binding, checkpoint_close_binding
from bindings.native_json import native_path, native_path_exists
from core.native_descriptor_abi import MAMBA_STATE_ABI_NAME
from providers.python.neural_descriptor_pyapi import neural_descriptor_cached
from core.hostptr import f32_ptr
from transformer.causal_lm_construct_impl import causal_lm_gather_f32, causal_lm_check_finite


def _model_shape(value: PythonObject, shape: List[Int], name: String) raises:
    if len(value.shape) != len(shape):
        raise Error("CausalLM: " + name + " has the wrong rank")
    for i in range(len(shape)):  # driver-loop: over tensor shape metadata
        if Int(py=value.shape[i]) != shape[i]:
            raise Error("CausalLM: " + name + " has the wrong shape")


def _model_dtype_refusal(name: String, code: String) -> String:
    var label = String(code)
    if code == "<f8" or code == "d" or code == "<d" or code == "=d" or code == "@d":
        label = "float64"
    elif code == "<f2" or code == "e" or code == "<e" or code == "=e" or code == "@e":
        label = "float16"
    elif code == "<u2" or code == "H":
        label = "uint16 (bfloat16 bits)"
    return ("TypeError: CausalLM tensor " + name + " has dtype " + label + "; float32 only: bfloat16, "
            + "float16 and float64 are refused by name rather than converted. Convert to float32.")


def _model_f32(value: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    if py_type(value) is context["array_class"]:
        if String(value.dtype) != "<f4":
            raise Error(_model_dtype_refusal(name, String(value.dtype)))
        if String(value.order) == "C":
            return value
        var dims = List[Int]()
        var strides = List[Int]()
        for i in range(len(value.shape)):  # driver-loop: over tensor shape metadata
            dims.append(Int(py=value.shape[i]))
            strides.append(Int(py=value.strides[i]))
        var result = generation_array(dims)
        var source = f32_ptr(Int(py=value._addr))
        var target = f32_ptr(Int(py=result._addr))
        _ = result  # keep the Python object alive through the native call above
        causal_lm_gather_f32(source, target, Int(py=value.size), dims, strides)
        _ = result  # keep the Python object alive through the native call above
        return result
    var view = py_call(bi.memoryview, value)
    var format = String(view.format)
    if format != "f" and format != "<f" and format != "=f" and format != "@f":
        raise Error(_model_dtype_refusal(name, format))
    return checkpoint_array(view.tobytes(), view.shape, "f", context["array_class"])


def _model_native_buffer(value: PythonObject, code: String, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    if py_type(value) is context["array_class"]:
        var dtype = "<u2" if code == "H" else ("<i1" if code == "b" else "<i4")
        if String(value.dtype) != dtype or String(value.order) != "C":
            raise Error("TypeError: CausalLM packed storage requires matching dtype and C layout")
        return value
    var view = py_call(bi.memoryview, value)
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
    var packed = py_call(bi.object.__new__, context["bf16_class"] if format == "bfloat16" else context["int8_class"])
    py_setattr(packed, "shape", value.shape)
    if format == "bfloat16":
        var bits = checkpoint_array(py_call(bi.bytes, rows * cols * 2), value.shape, "H", context["array_class"])
        _ = py_call(_native_function(linalg, "to_bf16"), bits._addr, value._addr, py_ints([rows * cols]))
        _ = bits  # keep the Python object alive through the native call above
        py_setattr(packed, "bits", bits)
    else:
        var codes = generation_array([rows, cols], "b")
        var exponents = generation_array([rows], "i")
        _ = py_call(_native_function(linalg, "quantize_int8"), codes._addr, exponents._addr, value._addr, py_ints([rows, cols]))
        _ = exponents  # keep the Python object alive through the native call above
        _ = codes  # keep the Python object alive through the native call above
        py_setattr(packed, "codes", codes)
        py_setattr(packed, "exponents", exponents)
    return packed


def _model_unpack(value: PythonObject, linalg: PythonObject, context: PythonObject, name: String = "weight") raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var bf16 = py_type(value) is context["bf16_class"]
    var int8 = py_type(value) is context["int8_class"]
    if not bf16 and not int8:
        return _model_f32(value, name, context)
    if len(value.shape) != 2:
        raise Error("CausalLM: packed projection weights require two dimensions")
    var rows = Int(py=value.shape[0])
    var cols = Int(py=value.shape[1])
    var out = generation_array([rows, cols])
    if bf16:
        var bits = _model_native_buffer(value.bits, "H", context)
        _model_shape(bits, [rows, cols], "bfloat16 bits")
        _ = py_call(_native_function(linalg, "from_bf16"), out._addr, bits._addr, py_ints([rows * cols]))
        _ = bits  # keep the Python object alive through the native call above
        _ = out  # keep the Python object alive through the native call above
        _ = bits^  # read by address above
    else:
        var codes = _model_native_buffer(value.codes, "b", context)
        var exponents = _model_native_buffer(value.exponents, "i", context)
        _model_shape(codes, [rows, cols], "int8 codes")
        _model_shape(exponents, [rows], "int8 exponents")
        _ = py_call(_native_function(linalg, "dequantize_int8"), out._addr, codes._addr, exponents._addr, py_ints([rows, cols]))
        _ = exponents  # keep the Python object alive through the native call above
        _ = codes  # keep the Python object alive through the native call above
        _ = out  # keep the Python object alive through the native call above
        _ = codes^  # read by address above
        _ = exponents^
    return out


def _model_quantize15(value: PythonObject, linalg: PythonObject) raises -> PythonObject:
    var rows = Int(py=value.shape[0])
    var cols = Int(py=value.shape[1])
    var hi = generation_array([rows, cols], "b")
    var lo = generation_array([rows, cols], "b")
    var exponents = generation_array([rows], "i")
    _ = py_call(_native_function(linalg, "quantize_int15"), py_ints([Int(py=hi._addr), Int(py=lo._addr), Int(py=exponents._addr), Int(py=value._addr)]), py_ints([rows, cols]))
    _ = exponents  # keep the Python object alive through the native call above
    _ = lo  # keep the Python object alive through the native call above
    _ = hi  # keep the Python object alive through the native call above
    var out = Python.import_module("builtins").list()
    _ = py_call(out.append, hi)
    _ = py_call(out.append, lo)
    _ = py_call(out.append, exponents)
    return py_call(Python.import_module("builtins").tuple, out)


def _model_profile(requested: PythonObject, family: String, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var profiles = context["profiles"]
    var defaults = context["profile_defaults"]
    var selected = requested
    if selected is None:
        selected = py_get(defaults, "inference")
        if selected is None:
            var env = getenv("MOJOLEARN_NUMERIC_PROFILE", "")
            selected = PythonObject(env).strip()
            if len(selected) == 0:
                selected = PythonObject("fp32_v1")
    if py_type(selected) is not bi.str:
        raise Error("TypeError: numeric_profile must be a profile name")
    selected = selected.strip()
    if not py_contains(profiles, selected):
        var reason = py_get(context["rejected_profiles"], selected)
        raise Error("numeric_profile=" + String(selected) + " is not offered: " + (String(reason) if reason is not None else String("not a registered numeric profile")))
    var row = profiles[selected]
    if not Bool(py=row["inference"]):
        raise Error("NotImplementedError: numeric_profile=" + String(selected) + " is unavailable for inference")
    if requested is None and py_get(defaults, "inference") is None:
        py_setitem(defaults, "inference", selected)
    var computes = row["computes"]
    var covered = computes is None or py_contains(computes, family)
    if requested is None:
        var reaches = row["default_for"]
        if not covered or (reaches is not None and not py_contains(reaches, family)):
            return PythonObject("fp32_v1")
    elif not covered:
        raise Error("NotImplementedError: numeric_profile=" + String(selected) + " does not compute the " + family + " family")
    return selected


def _model_profile_named(requested: PythonObject, context: PythonObject) raises:
    """The family-free half of `_model_profile` for a NAMED profile: an
    unknown name or one unavailable for inference is refused before any
    path is read."""
    var bi = Python.import_module("builtins")
    if py_type(requested) is not bi.str:
        raise Error("TypeError: numeric_profile must be a profile name")
    var selected = requested.strip()
    if not py_contains(context["profiles"], selected):
        var reason = py_get(context["rejected_profiles"], selected)
        raise Error("numeric_profile=" + String(selected) + " is not offered: "
                    + (String(reason) if reason is not None else String("not a registered numeric profile")))
    if not Bool(py=context["profiles"][selected]["inference"]):
        raise Error("NotImplementedError: numeric_profile=" + String(selected) + " is unavailable for inference")


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
    var entries = py_get(context["bindings"], key)
    if entries is None:
        raise Error("CausalLM: native binding connection is unavailable for " + String(key))
    var block = entries[kind]
    if py_type(block) is Python.import_module("builtins").str:
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
    if py_type(arithmetic) is not Python.import_module("builtins").str:
        if String(_native_function(arithmetic, "linalg_vendor")()) != vendor or Int(py=_native_function(arithmetic, "linalg_numeric_mode")()) != mode_code:
            raise Error("CausalLM: packed weight binding vendor or mode differs from model")
    return entries


def _model_bindings(route: PythonObject, kind: String, context: PythonObject) raises -> PythonObject:
    return _model_bindings(route, kind, context, PythonObject(None))


def _model_bits(value: PythonObject, name: String) raises -> Int:
    var number = Float64(py=py_call(Python.import_module("builtins").float, value))
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
    var block = py_call(bi.object.__new__, cls)
    var fields = py_call(bi.object.__getattribute__, block, "__dict__")
    py_setitem(fields, "numeric_mode", PythonObject("identical") if String(route) == "cpu" else context["backend"]._DEFAULT_MODE)
    var names = cls._W_NAMES
    var w = bi.dict()
    var keys = py_call(bi.list, weights)
    var format = "float32"
    for i in range(len(keys)):  # driver-loop: over the layer's named tensors
        var v = weights[keys[i]]
        var tag = "bfloat16" if py_type(v) is context["bf16_class"] else ("int8" if py_type(v) is context["int8_class"] else "float32")
        if tag != "float32":
            if format != "float32" and format != tag:
                raise Error("CausalLM: mixed packed weight formats")
            format = tag
        py_setitem(w, keys[i], _model_unpack(v, bindings["linalg"], context, String(keys[i])))
    py_setitem(fields, "weight_format", PythonObject(format))
    py_setitem(fields, "_native_kind", plan.kind)
    py_setitem(fields, "_native_route", route)
    var dm = Int(py=plan.d_model)
    var base = bi.list()
    var optional = bi.list()
    var expected = bi.set()
    var shapes = bi.list()
    var gated = True
    var it = 0
    if kind == "transformer":
        var options = py_call(bi.dict, context["config_context"]["defaults"])
        _ = py_call(options.update, plan.block_options)
        var nh = Int(py=plan.n_heads)
        if nh < 1:
            raise Error("CausalLM: n_heads must be positive")
        var nkv = nh if options["n_kv_heads"] is None else Int(py=options["n_kv_heads"])
        var hd = dm // nh if options["head_dim"] is None else Int(py=options["head_dim"])
        if nh < 1 or nkv < 1 or hd < 1 or dm != nh * hd:
            raise Error("CausalLM: d_model must equal n_heads * head_dim, all positive")
        var mlp = String(options["mlp"])
        var norm = String(options["norm"])
        var nk = 0 if norm == "rmsnorm" else (1 if norm == "layernorm" else (2 if norm == "rmsnorm_offset" else -1))
        var mk = 0 if mlp == "swiglu" else (1 if mlp == "gelu" else (2 if mlp == "gelu_tanh" else (3 if mlp == "geglu" else (4 if mlp == "geglu_tanh" else -1))))
        if nk < 0:
            raise Error("CausalLM: unsupported norm=" + String(py_call(bi.repr, options["norm"]))
                        + "; norm is 'rmsnorm', 'layernorm' or 'rmsnorm_offset'")
        if mk < 0:
            raise Error("CausalLM: unsupported mlp=" + String(py_call(bi.repr, options["mlp"]))
                        + "; mlp is 'swiglu', 'gelu', 'gelu_tanh', 'geglu' or 'geglu_tanh'")
        gated = mk == 0 or mk == 3 or mk == 4
        var has_gate = py_contains(w, "gate_proj.weight")
        if gated and not has_gate:
            raise Error("CausalLM: gate_proj.weight is missing; mlp=" + String(py_call(bi.repr, options["mlp"])) + " is gated")
        if not gated and has_gate:
            raise Error("CausalLM: gate_proj.weight is present but mlp=" + String(py_call(bi.repr, options["mlp"]))
                        + " has no gate")
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
            var sk = String(py_get(scaling, "type", py_get(scaling, "rope_type")))
            if sk != "linear" and sk != "llama3":
                raise Error("CausalLM: unsupported rope_scaling type " + String(py_call(bi.repr, PythonObject(sk)))
                            + "; the types are 'linear' and 'llama3'")
            tail[1] = 1 if sk == "linear" else 2
            tail[2] = _model_bits(scaling["factor"], "rope_scaling factor")
            if sk == "llama3":
                tail[3] = _model_bits(scaling["low_freq_factor"], "low_freq_factor")
                tail[4] = _model_bits(scaling["high_freq_factor"], "high_freq_factor")
                tail[5] = Int(py=scaling["original_max_position_embeddings"])
                if Float64(py=scaling["high_freq_factor"]) <= Float64(py=scaling["low_freq_factor"]) or tail[5] < 1:
                    raise Error("CausalLM: llama3 rope_scaling needs high_freq_factor > low_freq_factor "
                                + "and a positive original_max_position_embeddings")
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
        if tail[7] < 1:
            raise Error("CausalLM: max_positions must be positive")
        if tail[12] != 0 and nk != 1:
            raise Error("CausalLM: norm_bias=True needs norm='layernorm'")
        if tail[15] != 0 and nk == 1:
            raise Error("CausalLM: qk_norm=True with norm='layernorm' is not supported (qk_norm is an RMSNorm)")
        var window = Int(py=options["window"])
        if window < 0:
            raise Error("CausalLM: sliding window must be nonnegative")
        _ = py_call(fields.update, options)
        py_setitem(fields, "d_model", PythonObject(dm))
        py_setitem(fields, "n_heads", PythonObject(nh))
        py_setitem(fields, "n_kv_heads", PythonObject(nkv))
        py_setitem(fields, "head_dim", PythonObject(hd))
        py_setitem(fields, "intermediate", PythonObject(it))
        py_setitem(fields, "rope_dim", PythonObject(rd))
        py_setitem(fields, "_opts_tail", py_ints(tail))
        var default_tail = [0x461c4000, 0, 0x3f800000, 0x3f800000, 0x40800000, 8192, 0, 8192, 0, 0, 0, 0x358637bd, 0, 0, 0, 0, 0]
        var extended = False
        for i in range(17):  # driver-loop: over the fixed block options record
            extended = extended or tail[i] != default_tail[i]
        py_setitem(fields, "_extended", PythonObject(extended))
        py_setitem(fields, "_gated", PythonObject(gated))
        py_setitem(fields, "_runtime_lock", context["rlock"]())
        py_setitem(fields, "_native_session", PythonObject(None))
        py_setitem(fields, "_session_binding", PythonObject(None))
        py_setitem(fields, "_arithmetic_profile", PythonObject(None))
        if Bool(py=py_call(bi.hasattr, bindings[kind], "transformer_arithmetic_profile")):
            py_setitem(fields, "_arithmetic_profile", _native_function(bindings[kind], "transformer_arithmetic_profile")())
            py_setitem(fields, "_arithmetic_profile_mode", fields["numeric_mode"])
            py_setitem(fields, "_arithmetic_profile_changed", _native_function(bindings[kind], "transformer_arithmetic_profile_changed")())
        py_setitem(fields, "numeric_profile", profile)
        _ = py_call(shapes.append, py_ints([dm]))
        _ = py_call(shapes.append, py_ints([dm]))
        _ = py_call(shapes.append, py_ints([nh * hd, dm]))
        _ = py_call(shapes.append, py_ints([nkv * hd, dm]))
        _ = py_call(shapes.append, py_ints([nkv * hd, dm]))
        _ = py_call(shapes.append, py_ints([dm, nh * hd]))
        _ = py_call(shapes.append, py_ints([it, dm]))
        _ = py_call(shapes.append, py_ints([it, dm]))
        _ = py_call(shapes.append, py_ints([dm, it]))
        var flags = [tail[8], tail[8], tail[8], tail[9], tail[12], tail[12], tail[14], tail[14], tail[14] if gated else 0, tail[15], tail[15]]
        var widths = [nh * hd, nkv * hd, nkv * hd, dm, dm, dm, it, dm, it, hd, hd]
        var flag_dict = bi.dict()
        for i in range(11):  # driver-loop: over optional block tensors
            var name = context["optional_names"][i]
            py_setitem(flag_dict, name, PythonObject(flags[i] != 0))
            var present = py_contains(w, name)
            if present and flags[i] == 0:
                raise Error("CausalLM: " + String(name) + " is present but its option is off")
            if not present and flags[i] != 0:
                raise Error("CausalLM: " + String(name) + " is required by its option but missing")
            if present:
                _model_shape(w[name], [widths[i]], String(name))
                _ = py_call(optional.append, w[name])
                _ = py_call(expected.add, name)
            else:
                _ = py_call(optional.append, PythonObject(None))
        py_setitem(fields, "_opts_flags", flag_dict)
        py_setitem(fields, "_wopt", optional)
    else:
        var di = 2 * dm
        py_setitem(fields, "d_model", PythonObject(dm))
        py_setitem(fields, "d_inner", PythonObject(di))
        if kind == "mamba1":
            var rank = (dm + 15) // 16
            py_setitem(fields, "dt_rank", PythonObject(rank))
            _ = py_call(shapes.append, py_ints([dm]))
            _ = py_call(shapes.append, py_ints([2 * di, dm]))
            _ = py_call(shapes.append, py_ints([di, 1, 4]))
            _ = py_call(shapes.append, py_ints([di]))
            _ = py_call(shapes.append, py_ints([rank + 32, di]))
            _ = py_call(shapes.append, py_ints([di, rank]))
            _ = py_call(shapes.append, py_ints([di]))
            _ = py_call(shapes.append, py_ints([di, 16]))
            _ = py_call(shapes.append, py_ints([di]))
            _ = py_call(shapes.append, py_ints([dm, di]))
        elif kind == "mamba2":
            if dm % 32 != 0:
                raise Error("CausalLM: Mamba2 d_model must be a multiple of 32")
            var nh = di // 64
            var cd = di + 256
            py_setitem(fields, "nheads", PythonObject(nh))
            py_setitem(fields, "conv_dim", PythonObject(cd))
            var defaults = bi.list()
            _ = py_call(defaults.append, 0.0)
            _ = py_call(defaults.append, py_call(bi.float, "inf"))
            var limit = py_get(plan.block_kwargs, "dt_limit", defaults)
            var pair = bi.list()
            _ = py_call(pair.append, py_call(bi.float, limit[0]))
            _ = py_call(pair.append, py_call(bi.float, limit[1]))
            py_setitem(fields, "dt_limit", py_call(bi.tuple, pair))
            _ = py_call(shapes.append, py_ints([dm]))
            _ = py_call(shapes.append, py_ints([2 * di + 256 + nh, dm]))
            _ = py_call(shapes.append, py_ints([cd, 1, 4]))
            _ = py_call(shapes.append, py_ints([cd]))
            _ = py_call(shapes.append, py_ints([nh]))
            _ = py_call(shapes.append, py_ints([nh]))
            _ = py_call(shapes.append, py_ints([nh]))
            _ = py_call(shapes.append, py_ints([di]))
            _ = py_call(shapes.append, py_ints([dm, di]))
        elif kind == "mamba3":
            if dm % 32 != 0:
                raise Error("Mamba3Block: d_model must be a multiple of 32")
            var nh = di // 64
            var dip = 2 * di + 256 + 3 * nh + 32
            py_setitem(fields, "nheads", PythonObject(nh))
            py_setitem(fields, "d_in_proj", PythonObject(dip))
            _ = py_call(shapes.append, py_ints([dm]))
            _ = py_call(shapes.append, py_ints([dip, dm]))
            _ = py_call(shapes.append, py_ints([nh]))
            _ = py_call(shapes.append, py_ints([128]))
            _ = py_call(shapes.append, py_ints([128]))
            _ = py_call(shapes.append, py_ints([nh, 128]))
            _ = py_call(shapes.append, py_ints([nh, 128]))
            _ = py_call(shapes.append, py_ints([nh]))
            _ = py_call(shapes.append, py_ints([dm, di]))
        else:
            raise Error("neural block: unsupported block kind " + kind)
    for i in range(len(names)):  # small-loop(block weight name words)
        if kind == "transformer" and i == 6 and not gated:
            continue
        if not Bool(py=w.__contains__(names[i])):
            raise Error("CausalLM: weight dict mismatch; missing block tensor " + String(names[i]) + "; the exact key set is " + String(names))
    for i in range(len(names)):  # driver-loop: over fixed block weight slots
        var name = names[i]
        if kind == "transformer" and i == 6 and not gated:
            _ = py_call(base.append, PythonObject(None))
            continue
        if not py_contains(w, name):
            raise Error("CausalLM: missing block tensor " + String(name))
        var value = w[name]
        var shape = List[Int]()
        for j in range(len(shapes[i])):  # driver-loop: over tensor shape metadata
            shape.append(Int(py=shapes[i][j]))
        if (kind == "mamba1" or kind == "mamba2") and i == 2 and len(value.shape) == 2:
            shape = [shape[0], 4]
        _model_shape(value, shape, String(name))
        _ = py_call(base.append, value)
        _ = py_call(expected.add, name)
    if len(expected) != len(w):
        raise Error("CausalLM: unexpected block tensor names")
    py_setitem(fields, "_w", base)
    py_setitem(fields, "_int15", PythonObject(None))
    if String(profile) == "fixed15_v1":
        var planes = bi.list()
        for i in range(2, 9):  # driver-loop: over the seven transformer projections
            _ = py_call(planes.append, PythonObject(None) if base[i] is None else _model_quantize15(base[i], bindings["linalg"]))
        py_setitem(fields, "_int15", planes)
    if kind == "mamba3":
        _ = neural_descriptor_cached(fields, bindings[kind], "_native_mamba_state", "mamba_state_api_v1",
            String(MAMBA_STATE_ABI_NAME), String(route) == "gpu")
    return block


def causal_model_prepare(model: PythonObject, plan: PythonObject, weights: PythonObject, options: PythonObject, context: PythonObject) raises -> PythonObject:
    # Shared native admission and host metadata ownership. Parallel models
    # construct their layer owners in the assigned child processes afterward.
    var bi = Python.import_module("builtins")
    _ = generation_initialize_binding(context["generation_classes"])
    var format = String(options[0])
    if format != "float32" and format != "bfloat16" and format != "int8":
        raise Error("CausalLM: weight_format must be one of float32, bfloat16 or int8; unsupported weight_format " + format)
    var route = _model_route(options[2], context)
    var profile = _model_profile(options[3], String(plan.kind), context)
    var bindings = _model_bindings(route, String(plan.kind), context)
    py_setattr(model, "_runtime", context["runtime"])
    py_setattr(model, "plan", plan)
    py_setattr(model, "config", plan.config)
    py_setattr(model, "kind", plan.kind)
    py_setattr(model, "d_model", plan.d_model)
    py_setattr(model, "vocab_size", plan.vocab_size)
    py_setattr(model, "n_layers", plan.n_layers)
    py_setattr(model, "norm_eps", plan.norm_eps)
    py_setattr(model, "numeric_profile", profile)
    py_setattr(model, "device", route)
    py_setattr(model, "weight_format", options[0])
    py_setattr(model, "unused_names", py_call(bi.list, py_get(weights, "unused", bi.tuple())))
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
        var mp = Int(py=py_call(bi.int, positions))
        if mp < 1:
            raise Error("CausalLM: max_positions must be positive")
        if mp > config_cap or mp > 8192:
            raise Error("UnsupportedModel: max_positions exceeds max_position_embeddings or transformer ceiling 8192")
        positions = PythonObject(mp)
    elif positions is not None:
        positions = py_call(bi.int, positions)
    py_setattr(model, "max_positions", positions)
    py_setattr(model, "_embed", _model_f32(weights["embed"], String(plan.embed_name), context))
    if not Bool(py=model._embed.shape == py_call(bi.tuple, py_ints([vocab, dm]))):
        raise Error("UnsupportedModel: model_type " + String(plan.model_type) + ": vocab_size=" + String(vocab) + " and hidden_size=" + String(dm) + " but " + String(plan.embed_name) + " has shape " + String(py_call(bi.str, model._embed.shape)))
    py_setattr(model, "_norm", PythonObject(None))
    if plan.norm_name is not None:
        py_setattr(model, "_norm", _model_f32(weights["norm"], String(plan.norm_name), context))
        if not Bool(py=model._norm.shape == py_call(bi.tuple, py_ints([dm]))):
            raise Error("UnsupportedModel: tensor " + String(plan.norm_name) + " has shape " + String(py_call(bi.str, model._norm.shape)) + ", want hidden_size=" + String(dm))
    py_setattr(model, "_head", model._embed if plan.head_name is None else _model_f32(weights["head"], String(plan.head_name), context))
    if not Bool(py=model._head.shape == py_call(bi.tuple, py_ints([vocab, dm]))):
        raise Error("UnsupportedModel: tensor " + String(plan.head_name) + " has shape " + String(py_call(bi.str, model._head.shape)) + ", want vocab_size=" + String(vocab) + " and hidden_size=" + String(dm))
    for value in [model._embed, model._norm, model._head]:  # driver-loop: over the three top-level tensors
        if value is not None:
            var ptr = f32_ptr(Int(py=value._addr))
            causal_lm_check_finite(ptr, Int(py=value.size))
    var layers = weights["layers"]
    if len(layers) != Int(py=plan.n_layers):
        raise Error("CausalLM: layer dict count differs from num_hidden_layers")
    return bindings


def causal_model_init_binding(model: PythonObject, plan: PythonObject, weights: PythonObject, options: PythonObject, context: PythonObject) raises -> PythonObject:
    var bindings = causal_model_prepare(model, plan, weights, options, context)
    var bi = Python.import_module("builtins")
    var format = String(options[0])
    var route = model.device
    var profile = model.numeric_profile
    var layers = weights["layers"]
    py_setattr(model, "_blocks", bi.list())
    for i in range(len(layers)):  # driver-loop: over the model layer definitions
        var block = _model_block(layers[i], plan, profile, route, bindings, context)
        var fields = py_call(bi.object.__getattribute__, block, "__dict__")
        if String(fields["weight_format"]) != format:
            raise Error("CausalLM: layer reports a different weight_format than requested")
        _ = py_call(model._blocks.append, block)
    py_setattr(model, "_block_class", context["block_classes"][route][plan.kind])
    py_setattr(model, "_block_kwargs", py_call(bi.dict, plan.block_options if String(plan.kind) == "transformer" else plan.block_kwargs))
    if String(plan.kind) == "transformer":
        py_setitem(model._block_kwargs, "n_heads", plan.n_heads)
        py_setitem(model._block_kwargs, "numeric_profile", profile)
    py_setattr(model, "_prims", py_call(bi.object.__new__, context["primitive_classes"][route]))
    if String(route) == "gpu":
        py_setattr(model._prims, "_T", context["training_module"])
    else:
        py_setattr(model._prims, "_ext", bindings["primitives"])
    py_setattr(model, "_head_int15", PythonObject(None))
    if String(profile) == "fixed15_v1":
        py_setattr(model, "_head_int15", _model_quantize15(model._head, bindings["linalg"]))
    _ = causal_lm_admit_binding(model, bindings[plan.kind], bindings["primitives"], bindings["linalg"] if model._head_int15 is not None else PythonObject(None))
    return PythonObject(None)


def causal_model_read_weights_binding(checkpoint: PythonObject, plan: PythonObject, format: PythonObject, route: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    _ = generation_initialize_binding(context["generation_classes"])
    var ck = context["checkpoint_context"]
    var infos = checkpoint_index_binding(checkpoint, ck)
    var need = model_checkpoint_names_binding(plan)
    for i in range(len(need)):  # driver-loop: over checkpoint tensor names
        if not py_contains(infos, need[i]):
            raise Error("UnsupportedModel: checkpoint lacks tensor " + String(need[i]))
    var out = bi.dict()
    py_setitem(out, "embed", checkpoint_read_binding(checkpoint, plan.embed_name, PythonObject("widen"), ck))
    if plan.norm_name is not None:
        py_setitem(out, "norm", checkpoint_read_binding(checkpoint, plan.norm_name, PythonObject("widen"), ck))
    if plan.head_name is not None:
        py_setitem(out, "head", checkpoint_read_binding(checkpoint, plan.head_name, PythonObject("widen"), ck))
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
            var packed = String(format) != "float32" and py_contains(projections, key)
            if packed and String(format) == "bfloat16" and String(infos[name].dtype) == "BF16" and rows is None and len(infos[name].shape) == 2:
                py_setitem(weights, key, checkpoint_read_binding(checkpoint, name, PythonObject("bits"), ck))
                continue
            var value = _model_f32(checkpoint_read_binding(checkpoint, name, PythonObject("widen"), ck), String(name), context)
            if rows is not None:
                var begin = Int(py=rows[0])
                var end = Int(py=rows[1])
                if len(value.shape) != 2 or begin < 0 or end < begin or end > Int(py=value.shape[0]):
                    raise Error("UnsupportedModel: checkpoint tensor " + String(name) + " is too small for projection row slice")
                var cols = Int(py=value.shape[1])
                var sliced = generation_array([end - begin, cols])
                # `whole` owns the fused tensor's storage through the memcpy:
                # Mojo destroys a local at its last use, which for `value`
                # was the `value._addr` read in the memcpy's arguments, so the
                # array was freed before the copy read it (phi3's fused
                # qkv_proj / gate_up_proj; the allocator's free-list words
                # landed in the first rows of the q and gate slices and moved
                # hf-causal-lm's `every` part from run to run once the
                # balanced calls stopped leaking a reference to it).
                var whole = value
                if end > begin:
                    memcpy(dest=f32_ptr(Int(py=sliced._addr)), src=f32_ptr(Int(py=whole._addr)) + begin * cols, count=(end - begin) * cols)
                    _ = whole  # keep the Python object alive through the native call above
                    _ = sliced  # keep the Python object alive through the native call above
                _ = whole^  # read by address above
                value = sliced
            py_setitem(weights, key, _model_pack(value, String(format), bindings["linalg"], context) if packed else value)
        _ = py_call(layers.append, weights)
    py_setitem(out, "layers", layers)
    var unused = bi.list()
    var names = py_call(bi.list, infos)
    for i in range(len(names)):  # driver-loop: over checkpoint metadata names
        if not py_contains(need, names[i]):
            _ = py_call(unused.append, names[i])
    py_setitem(out, "unused", unused)
    return out


def causal_model_load_binding(cls: PythonObject, path_value: PythonObject, options: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var format = String(options[0])
    if format != "float32" and format != "bfloat16" and format != "int8":
        raise Error("CausalLM: weight_format must be one of float32, bfloat16 or int8; unsupported weight_format " + format)
    var route = _model_route(options[2], context)
    if options[3] is not None:
        _model_profile_named(options[3], context)
    var path = native_path(path_value)
    if not native_path_exists(path, True):
        raise Error("FileNotFoundError: CausalLM.load requires a checkpoint directory")
    var config = config_load_binding(context["config_context"]["config_class"], path)
    var plan = model_plan_for_binding(config, context["config_context"])
    _ = _model_profile(options[3], String(plan.kind), context)
    var ck = context["checkpoint_context"]
    var checkpoint = checkpoint_open_binding(ck["checkpoint_class"], path, ck)
    var weights = PythonObject(None)
    try:
        weights = causal_model_read_weights_binding(checkpoint, plan, options[0], route, context)
    except error:
        _ = checkpoint_close_binding(checkpoint)
        raise error
    _ = checkpoint_close_binding(checkpoint)
    var model = py_call(bi.object.__new__, cls)
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
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        if message.startswith("FileNotFoundError:"):
            return raise_python_exception(error_without_class(error, "FileNotFoundError"), ExceptionType("PyExc_FileNotFoundError"))
        if message.startswith("NotImplementedError:"):
            return raise_python_exception(error_without_class(error, "NotImplementedError"), ExceptionType("PyExc_NotImplementedError"))
        if message.startswith("UnsupportedModel:"):
            try:
                ref cpy = Python().cpython()
                var context = cpy.PyTuple_GetItem(args_ptr, 4 if operation == 0 else 3)
                var config = causal_borrowed_item(context, "config_context")
                var cls = causal_borrowed_item(config, "unsupported_class")
                return causal_set_exception(error_without_class(error, "UnsupportedModel"), cls)
            except nested:
                return raise_python_exception(nested)
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
