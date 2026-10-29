# SPDX-License-Identifier: Apache-2.0
"""Native checkpoint-to-model ownership over the existing compiled kernels.

Module connection metadata is supplied once by package import. Runtime route,
profile, layout, shape admission, packing and layer construction live here.
"""
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module, NativeModules
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_contains, py_get, py_setattr, py_setitem, py_type
from bindings.causal_exception import causal_borrowed_item, causal_set_exception
from std.math import isfinite
from std.memory import memcpy, bitcast
from std.os import getenv
from providers.python.operation_lock_pyapi import operation_lock_create
from bindings.generation_buffers import GENERATION_BOUNDARY, generation_initialize_binding, generation_array, generation_array_snapshot, generation_address, py_ints
from bindings.causal_lm_public import causal_lm_admit_binding, causal_public_connect, _causal_model_fields, _native_function, _native_entry_fields, _native_has_entry, _causal_array_guard
from bindings.causal_operation_owner import _causal_namespace, causal_snapshot_fields
from bindings.causal_model_config import config_load_binding, model_plan_for_binding, model_plan_snapshot, model_checkpoint_names_snapshot, _model_layer_weights, model_plain, model_record_fields
from bindings.causal_checkpoint import checkpoint_array, checkpoint_open_binding, checkpoint_index_binding, checkpoint_read_binding, checkpoint_close_binding, _info_snapshot
from bindings.native_json import native_path, native_path_exists
from bindings.mode_metadata_native import _fields, _class_fields
from bindings.neural_block_admission import neural_block_f32, neural_array_snapshot, neural_array_slots, neural_record_fields, neural_native_scalar, neural_plain_value, neural_string_keys, neural_metadata_snapshot, neural_model_weights, neural_block_fields, neural_block_weight_names
from providers.python.borrowed_buffer_pyapi import AdmittedBorrowedBuffer, admit_native_buffer_exporter
from core.native_descriptor_abi import MAMBA_STATE_ABI_NAME
from providers.python.neural_descriptor_pyapi import neural_descriptor_cached
from core.hostptr import f32_ptr
from transformer.causal_lm_construct_impl import causal_lm_gather_f32, causal_lm_check_finite


def _model_shape(value: PythonObject, shape: List[Int], name: String) raises:
    _causal_array_guard()
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
    # Shared native admission captures the original Array slots/export once,
    # retains their owner and preserves C-view versus F/foreign-copy behavior.
    return neural_block_f32(value, name, context)


def _model_native_buffer(mut mods: NativeModules, value: PythonObject, code: String, context: PythonObject) raises -> PythonObject:
    if py_type(value) is context["array_class"]:
        var captured = neural_array_snapshot(value, context)
        var dtype = "<u2" if code == "H" else ("<i1" if code == "b" else "<i4")
        if String(captured.dtype) != dtype or String(captured.order) != "C":
            raise Error("TypeError: CausalLM packed storage requires matching dtype and C layout")
        return captured
    admit_native_buffer_exporter(value)
    var pin = AdmittedBorrowedBuffer(value)
    if pin.format != code:
        raise Error("TypeError: CausalLM packed storage has the wrong dtype")
    var layout = pin.layout.value().copy()
    var size = 2 if code == "H" else (1 if code == "b" else 4)
    if layout.itemsize != size:
        raise Error("TypeError: CausalLM packed storage has the wrong item size")
    _ = neural_array_slots(context)
    var bi = mods.module("builtins")
    var result = checkpoint_array(mods, py_call(bi.bytes, layout.count * size),
        py_ints(layout.shape), code, context["array_class"])
    layout.copy_to(Int(py=result._addr))
    _ = pin^
    return result


def _model_native_buffer(value: PythonObject, code: String, context: PythonObject) raises -> PythonObject:
    # Classical and other callers keep their per-call module checks.
    var mods = NativeModules()
    return _model_native_buffer(mods, value, code, context)


def _model_packed_fields(mut mods: NativeModules, value: PythonObject, context: PythonObject) raises -> PythonObject:
    var cls = py_type(value)
    if cls is not context["bf16_class"] and cls is not context["int8_class"]:
        raise Error("TypeError: packed weights require their exact connected class")
    var names = Python.tuple("shape", "bits", "codes", "exponents")
    var fields = model_record_fields(mods, value, cls, names)
    var shape = fields["shape"]
    var bi = mods.module("builtins")
    if (py_type(shape) is not bi.tuple and py_type(shape) is not bi.list) or len(shape) != 2:
        raise Error("TypeError: packed projection shape requires two native dimensions")
    var dims = List[Int]()
    var count = 1
    for i in range(2):  # small-loop(packed projection metadata captured before storage)
        if py_type(shape[i]) is not bi.int:
            raise Error("TypeError: packed projection dimensions require native integers")
        var n = Int(py=shape[i])
        if n < 0 or (n != 0 and count > 2147483647 // n):
            raise Error("CausalLM: packed projection dimensions exceed native indexing")
        count *= n
        dims.append(n)
    var result = py_call(bi.dict, fields)
    py_setitem(result, "shape", py_call(bi.tuple, py_ints(dims)))
    return result


def _model_pack(mut mods: NativeModules, value: PythonObject, format: String, linalg: PythonObject, context: PythonObject) raises -> PythonObject:
    _causal_array_guard()
    var bi = mods.module("builtins")
    if format == "float32":
        return value
    if len(value.shape) != 2:
        raise Error("CausalLM: packed projection weights require two dimensions")
    var rows = Int(py=value.shape[0])
    var cols = Int(py=value.shape[1])
    var cls = context["bf16_class"] if format == "bfloat16" else context["int8_class"]
    var packed = py_call(bi.object.__new__, cls)
    var fields = model_record_fields(mods, packed, cls, Python.tuple("shape", "bits", "codes", "exponents"))
    py_setitem(fields, "shape", value.shape)
    if format == "bfloat16":
        var bits = checkpoint_array(mods, py_call(bi.bytes, rows * cols * 2), value.shape, "H", context["array_class"])
        _ = py_call(_native_function(mods, linalg, "to_bf16"), bits._addr, value._addr, py_ints([rows * cols]))
        mods.released()
        _ = bits  # keep the Python object alive through the native call above
        py_setitem(fields, "bits", bits)
    else:
        var codes = generation_array([rows, cols], "b")
        var exponents = generation_array([rows], "i")
        _ = py_call(_native_function(mods, linalg, "quantize_int8"), codes._addr, exponents._addr, value._addr, py_ints([rows, cols]))
        mods.released()
        _ = exponents  # keep the Python object alive through the native call above
        _ = codes  # keep the Python object alive through the native call above
        py_setitem(fields, "codes", codes)
        py_setitem(fields, "exponents", exponents)
    return packed


def _model_unpack(mut mods: NativeModules, value: PythonObject, linalg: PythonObject, context: PythonObject, name: String = "weight") raises -> PythonObject:
    var bi = mods.module("builtins")
    var bf16 = py_type(value) is context["bf16_class"]
    var int8 = py_type(value) is context["int8_class"]
    if not bf16 and not int8:
        return _model_f32(value, name, context)
    var fields = _model_packed_fields(mods, value, context)
    var rows = Int(py=fields["shape"][0])
    var cols = Int(py=fields["shape"][1])
    var out = generation_array([rows, cols])
    if bf16:
        var bits = _model_native_buffer(mods, fields["bits"], "H", context)
        _model_shape(bits, [rows, cols], "bfloat16 bits")
        _ = py_call(_native_function(mods, linalg, "from_bf16"), out._addr, bits._addr, py_ints([rows * cols]))
        mods.released()
        _ = bits  # keep the Python object alive through the native call above
        _ = out  # keep the Python object alive through the native call above
        _ = bits^  # read by address above
    else:
        var codes = _model_native_buffer(mods, fields["codes"], "b", context)
        var exponents = _model_native_buffer(mods, fields["exponents"], "i", context)
        _model_shape(codes, [rows, cols], "int8 codes")
        _model_shape(exponents, [rows], "int8 exponents")
        _ = py_call(_native_function(mods, linalg, "dequantize_int8"), out._addr, codes._addr, exponents._addr, py_ints([rows, cols]))
        mods.released()
        _ = exponents  # keep the Python object alive through the native call above
        _ = codes  # keep the Python object alive through the native call above
        _ = out  # keep the Python object alive through the native call above
        _ = codes^  # read by address above
        _ = exponents^
    return out


def _model_unpack(value: PythonObject, linalg: PythonObject, context: PythonObject, name: String = "weight") raises -> PythonObject:
    # Classical and other callers keep their per-call module checks.
    var mods = NativeModules()
    return _model_unpack(mods, value, linalg, context, name)


def _model_quantize15(mut mods: NativeModules, value: PythonObject, linalg: PythonObject) raises -> PythonObject:
    _causal_array_guard()
    var rows = Int(py=value.shape[0])
    var cols = Int(py=value.shape[1])
    var hi = generation_array([rows, cols], "b")
    var lo = generation_array([rows, cols], "b")
    var exponents = generation_array([rows], "i")
    _ = py_call(_native_function(mods, linalg, "quantize_int15"), py_ints([Int(py=hi._addr), Int(py=lo._addr), Int(py=exponents._addr), Int(py=value._addr)]), py_ints([rows, cols]))
    mods.released()
    _ = exponents  # keep the Python object alive through the native call above
    _ = lo  # keep the Python object alive through the native call above
    _ = hi  # keep the Python object alive through the native call above
    var out = mods.module("builtins").list()
    _ = py_call(out.append, hi)
    _ = py_call(out.append, lo)
    _ = py_call(out.append, exponents)
    return py_call(mods.module("builtins").tuple, out)


def _model_profile(mut mods: NativeModules, requested: PythonObject, family: String, context: PythonObject) raises -> PythonObject:
    var bi = mods.module("builtins")
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


def _model_profile(requested: PythonObject, family: String, context: PythonObject) raises -> PythonObject:
    # Classical and other callers keep their per-call module checks.
    var mods = NativeModules()
    return _model_profile(mods, requested, family, context)


def _model_profile_named(mut mods: NativeModules, requested: PythonObject, context: PythonObject) raises:
    """The family-free half of `_model_profile` for a NAMED profile: an
    unknown name or one unavailable for inference is refused before any
    path is read."""
    var bi = mods.module("builtins")
    if py_type(requested) is not bi.str:
        raise Error("TypeError: numeric_profile must be a profile name")
    var selected = requested.strip()
    if not py_contains(context["profiles"], selected):
        var reason = py_get(context["rejected_profiles"], selected)
        raise Error("numeric_profile=" + String(selected) + " is not offered: "
                    + (String(reason) if reason is not None else String("not a registered numeric profile")))
    if not Bool(py=context["profiles"][selected]["inference"]):
        raise Error("NotImplementedError: numeric_profile=" + String(selected) + " is unavailable for inference")


def _model_route(mut mods: NativeModules, device: PythonObject, context: PythonObject) raises -> PythonObject:
    if py_type(device) is not mods.module("builtins").str:
        raise Error("TypeError: CausalLM device requires an ordinary route name")
    var route = String(device)
    if route == "auto":
        route = "cpu" if context["selected_vendor"] is None or String(context["selected_vendor"]) == "cpu" else "gpu"
    if route != "cpu" and route != "gpu":
        raise Error("CausalLM: device must be 'auto', 'cpu' or 'gpu'")
    return PythonObject(route)


def _model_bindings(mut mods: NativeModules, route: PythonObject, kind: String, context: PythonObject, requested_mode: PythonObject) raises -> PythonObject:
    var bi = mods.module("builtins")
    if requested_mode is not None and py_type(requested_mode) is not bi.str:
        raise Error("TypeError: CausalLM numeric mode requires an ordinary string")
    var mode = PythonObject("identical") if String(route) == "cpu" else (_native_entry_fields(mods, context["backend"])["_DEFAULT_MODE"] if requested_mode is None else requested_mode.strip().lower())
    if py_type(mode) is not bi.str:
        raise Error("TypeError: CausalLM connected numeric mode requires an ordinary string")
    var key = route + PythonObject(":") + mode
    var entries = py_get(context["bindings"], key)
    if entries is None:
        raise Error("CausalLM: native binding connection is unavailable for " + String(key))
    var block = entries[kind]
    if py_type(block) is mods.module("builtins").str:
        raise Error("CausalLM: " + String(block))
    var prefix = ("transformer" if kind == "transformer" else "mamba") if String(route) == "gpu" else "neural_host"
    var vendor = String(_native_function(mods, block, prefix + "_vendor")())
    mods.released()
    var mode_code = Int(py=_native_function(mods, block, prefix + "_numeric_mode")())
    mods.released()
    if String(route) == "cpu" and vendor != "cpu":
        raise Error("CausalLM: CPU route requires native CPU kernels")
    if String(route) == "gpu" and vendor != "metal" and vendor != "cuda" and vendor != "hip":
        raise Error("CausalLM: GPU route requires native GPU kernels")
    if mode_code != (1 if String(mode) == "identical" else 0):
        raise Error("CausalLM: binding numeric mode differs from requested mode")
    var arithmetic = entries["linalg"]
    if py_type(arithmetic) is not mods.module("builtins").str:
        if String(_native_function(mods, arithmetic, "linalg_vendor")()) != vendor or Int(py=_native_function(mods, arithmetic, "linalg_numeric_mode")()) != mode_code:
            mods.released()
            raise Error("CausalLM: packed weight binding vendor or mode differs from model")
        mods.released()
    return entries


def _model_bindings(route: PythonObject, kind: String, context: PythonObject, requested_mode: PythonObject) raises -> PythonObject:
    # Classical and other callers keep their per-call module checks.
    var mods = NativeModules()
    return _model_bindings(mods, route, kind, context, requested_mode)


def _model_bindings(mut mods: NativeModules, route: PythonObject, kind: String, context: PythonObject) raises -> PythonObject:
    return _model_bindings(mods, route, kind, context, PythonObject(None))


def _model_bindings(route: PythonObject, kind: String, context: PythonObject) raises -> PythonObject:
    # Classical and other callers keep their per-call module checks.
    var mods = NativeModules()
    return _model_bindings(mods, route, kind, context)


def _model_bits(mut mods: NativeModules, value: PythonObject, name: String, context: PythonObject) raises -> Int:
    var bi = mods.module("builtins")
    if py_type(value) is not bi.str and not neural_native_scalar(mods, value, context):
        raise Error("TypeError: CausalLM " + name + " requires an ordinary numeric scalar")
    var number = Float64(py=py_call(bi.float, value))
    if not isfinite(number) or number <= 0:
        raise Error("CausalLM: " + name + " must be finite and positive")
    var rounded = Float32(number)
    if not isfinite(rounded):
        raise Error("CausalLM: " + name + " exceeds float32 range")
    return Int(bitcast[DType.uint32](rounded))


def _model_block_plan(mut mods: NativeModules, value: PythonObject, context: PythonObject) raises -> PythonObject:
    # Standalone blocks supply a private namespace; whole models pass their
    # operation snapshot dictionary. Both routes avoid plan attribute hooks.
    var bi = mods.module("builtins")
    var names = Python.tuple("kind", "d_model", "n_heads", "block_options", "block_kwargs")
    var source = PythonObject(None)
    if py_type(value) is bi.dict:
        neural_string_keys(mods, value)
        source = value
    elif py_type(value) is mods.module("types").SimpleNamespace:
        source = _native_entry_fields(mods, value)
    else:
        source = model_record_fields(mods, value, context["config_context"]["plan_class"], names)
    var result = Python.dict()
    for i in range(len(names)):  # small-loop(only the plan fields consumed by one block)
        if py_contains(source, names[i]):
            neural_plain_value(mods, source[names[i]], 0, context)
            py_setitem(result, names[i], neural_metadata_snapshot(mods, source[names[i]]))
    return result


def _model_options_snapshot(mut mods: NativeModules, options: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = mods.module("builtins")
    if (py_type(options) is not bi.tuple and py_type(options) is not bi.list) or len(options) != 4:
        raise Error("TypeError: CausalLM constructor requires its four native options")
    neural_plain_value(mods, options, 0, context)
    var captured = neural_metadata_snapshot(mods, options)
    if py_type(captured[0]) is not bi.str or py_type(captured[2]) is not bi.str:
        raise Error("TypeError: CausalLM weight format and device require ordinary strings")
    if captured[1] is not None and py_type(captured[1]) is not bi.str and not neural_native_scalar(mods, captured[1], context):
        raise Error("TypeError: CausalLM maximum positions require a native scalar")
    return captured


def _model_weights_snapshot(mut mods: NativeModules, weights: PythonObject, context: PythonObject) raises -> PythonObject:
    # Capture all caller tensor/exporter metadata before constructor packing or
    # finite-value kernels can release the GIL. Prepared snapshots retain owners.
    neural_string_keys(mods, weights)
    var bi = mods.module("builtins")
    var result = Python.dict()
    for name in ["embed", "norm", "head"]:  # small-loop(top-level model weights)
        if py_contains(weights, name):
            py_setitem(result, name, _model_f32(weights[name], name, context))
    var layers = weights["layers"]
    if py_type(layers) is not bi.list and py_type(layers) is not bi.tuple:
        raise Error("TypeError: CausalLM layers require an ordinary weight-map sequence")
    var copied = Python.list()
    for i in range(len(layers)):  # driver-loop: capture each complete layer weight map
        _ = py_call(copied.append, neural_model_weights(mods, layers[i], context))
    py_setitem(result, "layers", copied)
    var unused = py_get(weights, "unused", Python.tuple())
    neural_plain_value(mods, unused, 0, context)
    if py_type(unused) is not bi.list and py_type(unused) is not bi.tuple:
        raise Error("TypeError: CausalLM unused tensor names require an ordinary sequence")
    py_setitem(result, "unused", neural_metadata_snapshot(mods, unused))
    return result


def _model_recurrent_shapes(mut mods: NativeModules, kind: String, dm: Int, plan: PythonObject, fields: PythonObject) raises -> PythonObject:
    # Per-kind recurrent block configuration and current-weight shapes. Shared
    # by block construction and the recurrent backward weight check so both
    # admit exactly the same configuration, in the same order.
    var bi = mods.module("builtins")
    var shapes = bi.list()
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
        var limit = py_get(plan["block_kwargs"], "dt_limit", defaults)
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
    return shapes


def _model_block(mut mods: NativeModules, weights: PythonObject, input_plan: PythonObject, profile: PythonObject, route: PythonObject, bindings: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = mods.module("builtins")
    var plan = _model_block_plan(mods, input_plan, context)
    neural_string_keys(mods, weights)
    var kind = String(plan["kind"])
    var cls = context["block_classes"][route][kind]
    var block = py_call(bi.object.__new__, cls)
    var fields = neural_block_fields(mods, block, context, False)
    py_setitem(fields, "numeric_mode", PythonObject("identical") if String(route) == "cpu" else _native_entry_fields(mods, context["backend"])["_DEFAULT_MODE"])
    var names = neural_block_weight_names(mods, block)
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
        py_setitem(w, keys[i], _model_unpack(mods, v, bindings["linalg"], context, String(keys[i])))
    _causal_array_guard()
    py_setitem(fields, "weight_format", PythonObject(format))
    py_setitem(fields, "_native_kind", plan["kind"])
    py_setitem(fields, "_native_route", route)
    var dm = Int(py=plan["d_model"])
    var base = bi.list()
    var optional = bi.list()
    var expected = bi.set()
    var shapes = bi.list()
    var gated = True
    var it = 0
    if kind == "transformer":
        var options = py_call(bi.dict, context["config_context"]["defaults"])
        _ = py_call(options.update, plan["block_options"])
        var nh = Int(py=plan["n_heads"])
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
        tail[0] = _model_bits(mods, options["rope_theta"], "rope_theta", context)
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
            tail[2] = _model_bits(mods, scaling["factor"], "rope_scaling factor", context)
            if sk == "llama3":
                tail[3] = _model_bits(mods, scaling["low_freq_factor"], "low_freq_factor", context)
                tail[4] = _model_bits(mods, scaling["high_freq_factor"], "high_freq_factor", context)
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
        tail[11] = _model_bits(mods, options["norm_eps"], "norm_eps", context)
        tail[12] = Int(Bool(py=options["norm_bias"]))
        tail[13] = mk
        tail[14] = Int(Bool(py=options["mlp_bias"]))
        tail[15] = Int(Bool(py=options["qk_norm"]))
        tail[16] = 0 if options["attn_softcap"] is None else _model_bits(mods, options["attn_softcap"], "attn_softcap", context)
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
        if String(fields["numeric_mode"]) == "identical" and getenv("MOJOLEARN_NEURAL_NATIVE_DISPATCH", "1") == "1":
            py_setitem(fields, "_runtime_lock", operation_lock_create())
            mods.released()
        else:
            py_setitem(fields, "_runtime_lock", context["rlock"]())
            mods.released()
        var session = PythonObject(None)
        var session_binding = PythonObject(None)
        if String(route) == "gpu" and String(fields["numeric_mode"]) == "identical" and getenv("MOJOLEARN_NEURAL_NATIVE_DISPATCH", "1") == "1":
            if _native_has_entry(mods, bindings[kind], "transformer_session_create"):
                session = _native_function(mods, bindings[kind], "transformer_session_create")()
                mods.released()
                session_binding = bindings[kind]
        py_setitem(fields, "_native_session", session)
        py_setitem(fields, "_session_binding", session_binding)
        py_setitem(fields, "_arithmetic_profile", PythonObject(None))
        if _native_has_entry(mods, bindings[kind], "transformer_arithmetic_profile"):
            py_setitem(fields, "_arithmetic_profile", _native_function(mods, bindings[kind], "transformer_arithmetic_profile")())
            mods.released()
            py_setitem(fields, "_arithmetic_profile_mode", fields["numeric_mode"])
            py_setitem(fields, "_arithmetic_profile_changed", PythonObject(Bool(py=_native_function(mods, bindings[kind], "transformer_arithmetic_profile_changed")())))
            mods.released()
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
        shapes = _model_recurrent_shapes(mods, kind, dm, plan, fields)
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
            _ = py_call(planes.append, PythonObject(None) if base[i] is None else _model_quantize15(mods, base[i], bindings["linalg"]))
        py_setitem(fields, "_int15", planes)
    if kind == "mamba3":
        if String(route) == "gpu" and String(fields["numeric_mode"]) == "identical" and getenv("MOJOLEARN_NEURAL_NATIVE_DISPATCH", "0") == "1":
            if _native_has_entry(mods, bindings[kind], "mamba3_prefill_session_create"):
                py_setitem(fields, "_prefill_session", _native_function(mods, bindings[kind], "mamba3_prefill_session_create")())
                mods.released()
                py_setitem(fields, "_prefill_binding", bindings[kind])
        _ = neural_descriptor_cached(fields, bindings[kind], "_native_mamba_state", "mamba_state_api_v1",
            String(MAMBA_STATE_ABI_NAME), String(route) == "gpu")
        mods.released()
    return block


def _model_class_weight_names(mut mods: NativeModules, cls: PythonObject) raises -> PythonObject:
    # The weight registry a fresh instance of `cls` reports: the same lookup
    # and checks as neural_block_weight_names, read from the class ancestry.
    var bi = mods.module("builtins")
    var names = PythonObject(None)
    var ancestry = py_call(bi.type.__dict__["__mro__"].__get__, cls)
    for i in range(len(ancestry)):  # driver-loop: native admission metadata and retained owner registry
        names = py_get(_class_fields(ancestry[i]), "_W_NAMES")
        if names is not None:
            break
    if py_type(names) is not bi.tuple and py_type(names) is not bi.list:
        raise Error("TypeError: neural weight registry requires native names")
    for i in range(len(names)):  # driver-loop: native admission metadata and retained owner registry
        if py_type(names[i]) is not bi.str:
            raise Error("TypeError: neural weight names require exact strings")
    return py_call(bi.tuple, names)


def model_recurrent_weights_check(mut mods: NativeModules, weights: PythonObject, kind: String, dm: Int, plan: PythonObject, route: PythonObject, bindings: PythonObject, context: PythonObject) raises -> PythonObject:
    # Current-weight admission for a recurrent (Mamba) block backward without
    # constructing a block. It performs the admissions _model_block makes over
    # the operation's weight snapshot, in the same order and with the same
    # refusals: connected class registry, per-tensor float32 admission, the
    # Array descriptor recheck after the caller's GIL release, per-kind
    # configuration, exact key set and shapes. `weights` maps the block's
    # weight names to its snapshot tensors; `plan` carries block_kwargs.
    # Returns the admitted tensors in registry order.
    var bi = mods.module("builtins")
    neural_string_keys(mods, weights)
    var names = _model_class_weight_names(mods, context["block_classes"][route][kind])
    var w = bi.dict()
    var keys = py_call(bi.list, weights)
    for i in range(len(keys)):  # driver-loop: over the layer's named tensors
        var v = weights[keys[i]]
        # Snapshot tensors are already admitted float32 C Arrays owned by this
        # operation; anything else takes the constructor's unpack admission.
        if py_type(v) is not context["array_class"]:
            v = _model_unpack(mods, v, bindings["linalg"], context, String(keys[i]))
        py_setitem(w, keys[i], v)
    # The one Array descriptor recheck after the grad_output finite scan
    # released the GIL; kept exactly once for that release window.
    _causal_array_guard()
    var shapes = _model_recurrent_shapes(mods, kind, dm, plan, bi.dict())
    for i in range(len(names)):  # small-loop(block weight name words)
        if not Bool(py=w.__contains__(names[i])):
            raise Error("CausalLM: weight dict mismatch; missing block tensor " + String(names[i]) + "; the exact key set is " + String(names))
    var base = bi.list()
    var expected = bi.set()
    for i in range(len(names)):  # driver-loop: over fixed block weight slots
        var name = names[i]
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
    return base


def _model_block(weights: PythonObject, input_plan: PythonObject, profile: PythonObject, route: PythonObject, bindings: PythonObject, context: PythonObject) raises -> PythonObject:
    # Classical and other callers keep their per-call module checks.
    var mods = NativeModules()
    return _model_block(mods, weights, input_plan, profile, route, bindings, context)


def _model_construction_set(mut mods: NativeModules, live: PythonObject, captured: PythonObject, name: String, value: PythonObject) raises:
    # The caller-visible record keeps its incremental construction updates.
    # Execution only reads the separate retained record after native calls.
    neural_string_keys(mods, live)
    py_setitem(captured, name, value)
    var published = value
    var bi = mods.module("builtins")
    if (name == "_embed" or name == "_norm" or name == "_head") and value is not None:
        # Caller-visible Array slots must not be the metadata traversed by
        # subsequent constructor kernels. Retain the same live storage and
        # preserve the tied head/embedding object's public identity.
        var buffers = py_get(captured, "_construction_published_buffers")
        if buffers is None:
            buffers = Python.dict()
            py_setitem(captured, "_construction_published_buffers", buffers)
        var identity = py_call(bi.id, value)
        published = py_get(buffers, identity)
        if published is None:
            published = generation_array_snapshot(value)
            py_setitem(buffers, identity, published)
    elif name == "_blocks" or name == "unused_names":
        published = py_call(bi.list, value)
    elif name == "_block_kwargs":
        published = py_call(bi.dict, value)
    py_setitem(live, name, published)


def _model_construction_admission(mut mods: NativeModules, live: PythonObject, admitted: PythonObject) raises:
    neural_string_keys(mods, live)
    for name in ["_native_vendor", "_native_numeric_mode", "_native_entries", "_causal_entries_owner"]:  # small-loop(native admission fields published on success and error)
        if py_contains(admitted, name):
            py_setitem(live, name, admitted[name])


def causal_model_prepare(mut mods: NativeModules, model: PythonObject, input_plan: PythonObject, input_weights: PythonObject, input_options: PythonObject, context: PythonObject) raises -> PythonObject:
    causal_public_connect(context)
    var live = _causal_model_fields(mods, model)
    var captured = Python.dict()
    var plan = model_plan_snapshot(mods, input_plan)
    var options = _model_options_snapshot(mods, input_options, context)
    var weights = _model_weights_snapshot(mods, input_weights, context)
    return _causal_model_prepare_snapshot(mods, model, input_plan, plan, weights, options, context, live, captured)


def _causal_model_prepare_snapshot(mut mods: NativeModules, model: PythonObject, input_plan: PythonObject, plan: PythonObject, weights: PythonObject, options: PythonObject, context: PythonObject) raises -> PythonObject:
    # Preserve the parallel constructor's six-argument native entry contract.
    var live = _causal_model_fields(mods, model)
    var captured = Python.dict()
    return _causal_model_prepare_snapshot(mods, model, input_plan, plan, weights, options, context, live, captured)


def _causal_model_prepare_snapshot(mut mods: NativeModules, model: PythonObject, input_plan: PythonObject, plan: PythonObject, weights: PythonObject, options: PythonObject, context: PythonObject, live: PythonObject, captured: PythonObject) raises -> PythonObject:
    # Shared native admission and host metadata ownership. Parallel models
    # construct their layer owners in the assigned child processes afterward.
    var bi = mods.module("builtins")
    causal_public_connect(context)
    neural_string_keys(mods, live)
    neural_string_keys(mods, captured)
    var format = String(options[0])
    if format != "float32" and format != "bfloat16" and format != "int8":
        raise Error("CausalLM: weight_format must be one of float32, bfloat16 or int8; unsupported weight_format " + format)
    var route = _model_route(mods, options[2], context)
    var profile = _model_profile(mods, options[3], String(plan["kind"]), context)
    var bindings = _model_bindings(mods, route, String(plan["kind"]), context)
    _model_construction_set(mods, live, captured, "_runtime", context["runtime"])
    _model_construction_set(mods, live, captured, "plan", input_plan)
    _model_construction_set(mods, live, captured, "config", plan["config"])
    _model_construction_set(mods, live, captured, "kind", plan["kind"])
    _model_construction_set(mods, live, captured, "d_model", plan["d_model"])
    _model_construction_set(mods, live, captured, "vocab_size", plan["vocab_size"])
    _model_construction_set(mods, live, captured, "n_layers", plan["n_layers"])
    _model_construction_set(mods, live, captured, "norm_eps", plan["norm_eps"])
    _model_construction_set(mods, live, captured, "numeric_profile", profile)
    _model_construction_set(mods, live, captured, "device", route)
    _model_construction_set(mods, live, captured, "weight_format", options[0])
    _model_construction_set(mods, live, captured, "unused_names", py_call(bi.list, py_get(weights, "unused", bi.tuple())))
    var dm = Int(py=plan["d_model"])
    var vocab = Int(py=plan["vocab_size"])
    if dm < 1 or vocab < 1 or Int(py=plan["n_layers"]) < 1:
        raise Error("CausalLM: model dimensions and layer count must be positive")
    var positions = options[1]
    if String(plan["kind"]) == "transformer":
        var config_cap = 8192 if plan["max_position_embeddings"] is None else Int(py=plan["max_position_embeddings"])
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
    _model_construction_set(mods, live, captured, "max_positions", positions)
    _model_construction_set(mods, live, captured, "_embed", _model_f32(weights["embed"], String(plan["embed_name"]), context))
    _causal_array_guard()
    if not Bool(py=captured["_embed"].shape == py_call(bi.tuple, py_ints([vocab, dm]))):
        raise Error("UnsupportedModel: model_type " + String(plan["model_type"]) + ": vocab_size=" + String(vocab) + " and hidden_size=" + String(dm) + " but " + String(plan["embed_name"]) + " has shape " + String(py_call(bi.str, captured["_embed"].shape)))
    _model_construction_set(mods, live, captured, "_norm", PythonObject(None))
    if plan["norm_name"] is not None:
        _model_construction_set(mods, live, captured, "_norm", _model_f32(weights["norm"], String(plan["norm_name"]), context))
        _causal_array_guard()
        if not Bool(py=captured["_norm"].shape == py_call(bi.tuple, py_ints([dm]))):
            raise Error("UnsupportedModel: tensor " + String(plan["norm_name"]) + " has shape " + String(py_call(bi.str, captured["_norm"].shape)) + ", want hidden_size=" + String(dm))
    _model_construction_set(mods, live, captured, "_head", captured["_embed"] if plan["head_name"] is None else _model_f32(weights["head"], String(plan["head_name"]), context))
    _causal_array_guard()
    if not Bool(py=captured["_head"].shape == py_call(bi.tuple, py_ints([vocab, dm]))):
        raise Error("UnsupportedModel: tensor " + String(plan["head_name"]) + " has shape " + String(py_call(bi.str, captured["_head"].shape)) + ", want vocab_size=" + String(vocab) + " and hidden_size=" + String(dm))
    for value in [captured["_embed"], captured["_norm"], captured["_head"]]:  # driver-loop: over the three top-level tensors
        if value is not None:
            _causal_array_guard()
            var ptr = f32_ptr(Int(py=value._addr))
            causal_lm_check_finite(ptr, Int(py=value.size))
    var layers = weights["layers"]
    if len(layers) != Int(py=plan["n_layers"]):
        raise Error("CausalLM: layer dict count differs from num_hidden_layers")
    return bindings


def causal_model_init_binding(mut mods: NativeModules, model: PythonObject, input_plan: PythonObject, input_weights: PythonObject, input_options: PythonObject, context: PythonObject) raises -> PythonObject:
    causal_public_connect(context)
    var live = _causal_model_fields(mods, model)
    var captured = Python.dict()
    var plan = model_plan_snapshot(mods, input_plan)
    var options = _model_options_snapshot(mods, input_options, context)
    var weights = _model_weights_snapshot(mods, input_weights, context)
    var bindings = _causal_model_prepare_snapshot(mods, model, input_plan, plan, weights, options, context, live, captured)
    var bi = mods.module("builtins")
    var format = String(options[0])
    var route = captured["device"]
    var profile = captured["numeric_profile"]
    var layers = weights["layers"]
    _model_construction_set(mods, live, captured, "_blocks", bi.list())
    for i in range(len(layers)):  # driver-loop: over the model layer definitions
        var block = _model_block(mods, layers[i], plan, profile, route, bindings, context)
        var fields = neural_block_fields(mods, block, context)
        if String(fields["weight_format"]) != format:
            raise Error("CausalLM: layer reports a different weight_format than requested")
        _ = py_call(captured["_blocks"].append, block)
        _model_construction_set(mods, live, captured, "_blocks", captured["_blocks"])
    _model_construction_set(mods, live, captured, "_block_class", context["block_classes"][route][plan["kind"]])
    _model_construction_set(mods, live, captured, "_block_kwargs", py_call(bi.dict, plan["block_options"] if String(plan["kind"]) == "transformer" else plan["block_kwargs"]))
    if String(plan["kind"]) == "transformer":
        py_setitem(captured["_block_kwargs"], "n_heads", plan["n_heads"])
        py_setitem(captured["_block_kwargs"], "numeric_profile", profile)
        _model_construction_set(mods, live, captured, "_block_kwargs", captured["_block_kwargs"])
    var primitive_class = context["primitive_classes"][route]
    var primitive = py_call(bi.object.__new__, primitive_class)
    var primitive_fields = model_record_fields(mods, primitive, primitive_class, Python.tuple("_T", "_ext"))
    if String(route) == "gpu":
        py_setitem(primitive_fields, "_T", context["training_module"])
    else:
        py_setitem(primitive_fields, "_ext", bindings["primitives"])
    _model_construction_set(mods, live, captured, "_prims", primitive)
    _model_construction_set(mods, live, captured, "_head_int15", PythonObject(None))
    if String(profile) == "fixed15_v1":
        _model_construction_set(mods, live, captured, "_head_int15", _model_quantize15(mods, captured["_head"], bindings["linalg"]))
    # Admission may call sibling native entries. Give it the private record;
    # a replaced public model descriptor cannot run after those calls.
    py_setitem(captured, "_native_context", context)
    var admission = _causal_namespace(mods, captured)
    var admitted = causal_snapshot_fields(mods, admission)
    try:
        _ = causal_lm_admit_binding(mods, admission, bindings[plan["kind"]], bindings["primitives"], bindings["linalg"] if captured["_head_int15"] is not None else PythonObject(None))
    except problem:
        mods.released()
        _model_construction_admission(mods, live, admitted)
        raise problem
    _model_construction_admission(mods, live, admitted)
    return PythonObject(None)


def causal_model_read_weights_binding(mut mods: NativeModules, checkpoint: PythonObject, input_plan: PythonObject, format: PythonObject, route: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = mods.module("builtins")
    causal_public_connect(context)
    var plan = model_plan_snapshot(mods, input_plan)
    if py_type(format) is not bi.str:
        raise Error("TypeError: CausalLM weight format requires an ordinary string")
    var ck = context["checkpoint_context"]
    var infos = checkpoint_index_binding(mods, checkpoint, ck)
    var captured_infos = Python.dict()
    var info_names = py_call(bi.list, infos.keys())
    for i in range(len(info_names)):  # driver-loop: retain checkpoint shape/dtype metadata before execution
        py_setitem(captured_infos, info_names[i], _info_snapshot(mods, infos[info_names[i]]))
    var need = model_checkpoint_names_snapshot(mods, plan)
    for i in range(len(need)):  # driver-loop: over checkpoint tensor names
        if not py_contains(infos, need[i]):
            raise Error("UnsupportedModel: checkpoint lacks tensor " + String(need[i]))
    var out = bi.dict()
    py_setitem(out, "embed", checkpoint_read_binding(mods, checkpoint, plan["embed_name"], PythonObject("widen"), ck))
    if plan["norm_name"] is not None:
        py_setitem(out, "norm", checkpoint_read_binding(mods, checkpoint, plan["norm_name"], PythonObject("widen"), ck))
    if plan["head_name"] is not None:
        py_setitem(out, "head", checkpoint_read_binding(mods, checkpoint, plan["head_name"], PythonObject("widen"), ck))
    var bindings = _model_bindings(mods, route, String(plan["kind"]), context)
    var projections = context["projections"][plan["kind"]]
    var layers = bi.list()
    for i in range(Int(py=plan["n_layers"])):  # driver-loop: over checkpoint layers
        var weights = bi.dict()
        var mapping = _model_layer_weights(mods, plan, i)
        for j in range(len(mapping)):  # driver-loop: over named layer tensor entries
            var entry = mapping[j]
            var key = entry[0]
            var name = entry[1]
            var rows = entry[2]
            var packed = String(format) != "float32" and py_contains(projections, key)
            if packed and String(format) == "bfloat16" and String(captured_infos[name]["dtype"]) == "BF16" and rows is None and len(captured_infos[name]["shape"]) == 2:
                py_setitem(weights, key, checkpoint_read_binding(mods, checkpoint, name, PythonObject("bits"), ck))
                continue
            var value = _model_f32(checkpoint_read_binding(mods, checkpoint, name, PythonObject("widen"), ck), String(name), context)
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
            py_setitem(weights, key, _model_pack(mods, value, String(format), bindings["linalg"], context) if packed else value)
        _ = py_call(layers.append, weights)
    py_setitem(out, "layers", layers)
    var unused = bi.list()
    var names = py_call(bi.list, infos)
    for i in range(len(names)):  # driver-loop: over checkpoint metadata names
        if not py_contains(need, names[i]):
            _ = py_call(unused.append, names[i])
    py_setitem(out, "unused", unused)
    return out


def causal_model_load_binding(mut mods: NativeModules, cls: PythonObject, path_value: PythonObject, input_options: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = mods.module("builtins")
    var options = _model_options_snapshot(mods, input_options, context)
    var format = String(options[0])
    if format != "float32" and format != "bfloat16" and format != "int8":
        raise Error("CausalLM: weight_format must be one of float32, bfloat16 or int8; unsupported weight_format " + format)
    var route = _model_route(mods, options[2], context)
    if options[3] is not None:
        _model_profile_named(mods, options[3], context)
    var path = native_path(path_value)
    if not native_path_exists(path, True):
        mods.released()
        raise Error("FileNotFoundError: CausalLM.load requires a checkpoint directory")
    mods.released()
    var config = config_load_binding(mods, context["config_context"]["config_class"], path)
    var plan = model_plan_for_binding(mods, config, context["config_context"])
    var plan_fields = model_plan_snapshot(mods, plan)
    _ = _model_profile(mods, options[3], String(plan_fields["kind"]), context)
    var ck = context["checkpoint_context"]
    var checkpoint = checkpoint_open_binding(mods, ck["checkpoint_class"], path, ck)
    var weights = PythonObject(None)
    try:
        weights = causal_model_read_weights_binding(mods, checkpoint, plan, options[0], route, context)
    except error:
        mods.released()
        _ = checkpoint_close_binding(mods, checkpoint)
        raise error
    _ = checkpoint_close_binding(mods, checkpoint)
    var model = py_call(bi.object.__new__, cls)
    _ = causal_model_init_binding(mods, model, plan, weights, options, context)
    return model


def causal_model_c[operation: Int](py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    var mods = NativeModules()
    var args = PythonObject(from_borrowed=args_ptr)
    try:
        var result = PythonObject(None)
        comptime if operation == 0:
            result = causal_model_init_binding(mods, args[0], args[1], args[2], args[3], args[4])
        elif operation == 1:
            result = causal_model_load_binding(mods, args[0], args[1], args[2], args[3])
        else:
            result = causal_model_read_weights_binding(mods, args[0], args[1], args[2], _model_route(mods, PythonObject("auto"), args[3]), args[3])
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
