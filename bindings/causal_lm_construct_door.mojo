# SPDX-License-Identifier: Apache-2.0
"""Native checkpoint-to-model ownership over the existing compiled kernels.

Module connection metadata is supplied once by package import. Runtime route,
profile, layout, shape admission, packing and layer construction live here.
"""
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module, NativeModules
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_contains, py_get, py_setattr, py_setitem, py_type
from bindings.causal_exception_unpack import causal_borrowed_item, causal_set_exception
from std.math import isfinite
from std.memory import unsafe_memcpy, bitcast
from std.os import getenv
from providers.python.operation_lock_pyapi_door import operation_lock_create
from bindings.generation_buffers_door import GENERATION_BOUNDARY, generation_initialize_binding, generation_array, generation_array_snapshot, generation_address, py_ints
from bindings.causal_lm_public_door import causal_lm_admit_binding, causal_public_connect, _causal_model_fields, _native_function, _native_entry_fields, _native_has_entry, _causal_array_guard, causal_entry_key
from bindings.causal_operation_owner_door import _causal_namespace, causal_snapshot_fields
from bindings.causal_model_config_door import config_load_binding, model_plan_for_binding, model_plan_snapshot, model_checkpoint_names_snapshot, _model_layer_weights, model_plain, model_record_fields
from bindings.causal_checkpoint_door import checkpoint_array, checkpoint_zeros, checkpoint_open_binding, checkpoint_index_binding, checkpoint_read_binding, checkpoint_close_binding, _info_snapshot
from bindings.native_json_door import native_path, native_path_exists
from bindings.mode_metadata_native_door import _fields, _class_fields
from bindings.neural_block_admission_door import admit_block_f32, BLOCK_OPERATION, neural_array_snapshot, neural_array_slots, neural_record_fields, neural_native_scalar, neural_plain_value, neural_string_keys, neural_metadata_snapshot, neural_model_weights, neural_block_fields, neural_block_weight_names
from providers.python.borrowed_buffer_pyapi_unpack import AdmittedBorrowedBuffer, admit_native_buffer_exporter
from core.native_descriptor_abi import MAMBA_STATE_ABI_NAME
from providers.python.neural_descriptor_pyapi_door import neural_descriptor_cached, NEURAL_NATIVE_DISPATCH
from core.hostptr import f32_ptr
from transformer.causal_lm_construct_impl import causal_lm_gather_f32, causal_lm_check_finite
from std.python._cpython import GILReleased
from core.mode_answer import MODE_FAMILY_NEURAL
from bindings.mode_answer_door import answer_mode_object
from core.neural_entry_table import NeuralEntryKey, neural_ref, NEURAL_TABLE_BLOCK_FORWARD, NEURAL_TABLE_MAMBA_TRAIN, MAMBA3_CAP_SESSION_PREFILL
from providers.python.mamba3_prefill_session_pyapi_door import mamba3_prefill_caps, mamba3_prefill_session_open
from providers.python.transformer_block_session_pyapi_door import transformer_block_session_open
from core.causal_lm_entry_table import (
    CausalEntryKey, causal_entry_vendor, causal_entry_call, causal_ref, causal_table_entry, CAUSAL_TABLE_LINALG,
    CAUSAL_LIN_TO_BF16, CAUSAL_LIN_QUANTIZE_INT8, CAUSAL_LIN_FROM_BF16, CAUSAL_LIN_DEQUANTIZE_INT8,
    CAUSAL_LIN_QUANTIZE_INT15,
)
from bindings.causal_keys_unpack import causal_keys
from bindings.py_balanced import py_item


def _model_shape(value: PythonObject, shape: List[Int], name: String) raises:
    _causal_array_guard()
    if len(value.shape) != len(shape):
        raise Error("CausalLM: " + name + " has the wrong rank")
    for i in range(len(shape)):  # driver-loop: over tensor shape metadata
        if Int(py=value.shape[i]) != shape[i]:
            raise Error("CausalLM: " + name + " has the wrong shape")


def _model_f32(value: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    # Shared native admission captures the original Array slots/export once,
    # retains their owner and preserves C-view versus F/foreign-copy behavior.
    return admit_block_f32(value, name, context, BLOCK_OPERATION)


def _model_native_buffer(mut mods: NativeModules, value: PythonObject, code: String, context: PythonObject) raises -> PythonObject:
    var nk = causal_keys()
    if py_type(value) is py_item(context, nk[].k_array_class):
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
    var result = checkpoint_zeros(mods, layout.count, py_ints(layout.shape), code, py_item(context, nk[].k_array_class))
    layout.copy_to(Int(py=result._addr))
    _ = pin^
    return result


def _model_native_buffer(value: PythonObject, code: String, context: PythonObject) raises -> PythonObject:
    # Classical and other callers keep their per-call module checks.
    var mods = NativeModules()
    return _model_native_buffer(mods, value, code, context)


def _model_packed_fields(mut mods: NativeModules, value: PythonObject, context: PythonObject) raises -> PythonObject:
    var cls = py_type(value)
    var nk = causal_keys()
    if cls is not py_item(context, nk[].k_bf16_class) and cls is not py_item(context, nk[].k_int8_class):
        raise Error("TypeError: packed weights require their exact connected class")
    var names = Python.tuple("shape", "bits", "codes", "exponents")
    var fields = model_record_fields(mods, value, cls, names)
    var shape = py_item(fields, nk[].k_shape)
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
    py_setitem(result, nk[].k_shape, py_call(bi.tuple, py_ints(dims)))
    return result


def _model_linalg(mut mods: NativeModules, key: CausalEntryKey, operation: Int, addresses: List[Int], params: List[Int]) raises:
    # One call of the linalg image's descriptor entry through the CausalLM
    # entry table (core/causal_lm_entry_table.mojo), replacing the former
    # CPython call of its Python entry (same `*_native` function, same words).
    # The GIL is released around the numerical work as that entry released it;
    # the module admission cache is reset after the release window.
    var floats = List[Float64]()
    var lin = causal_ref[CAUSAL_TABLE_LINALG](key)
    try:
        with GILReleased(Python()):
            _ = causal_entry_call(lin, operation, addresses, params, floats)
    finally:
        mods.released()
    _ = floats


def _model_pack(mut mods: NativeModules, value: PythonObject, format: String, key: CausalEntryKey, context: PythonObject) raises -> PythonObject:
    _causal_array_guard()
    var bi = mods.module("builtins")
    if format == "float32":
        return value
    if len(value.shape) != 2:
        raise Error("CausalLM: packed projection weights require two dimensions")
    var rows = Int(py=value.shape[0])
    var cols = Int(py=value.shape[1])
    var nk = causal_keys()
    var cls = py_item(context, nk[].k_bf16_class) if format == "bfloat16" else py_item(context, nk[].k_int8_class)
    var packed = py_call(bi.object.__new__, cls)
    var fields = model_record_fields(mods, packed, cls, Python.tuple("shape", "bits", "codes", "exponents"))
    py_setitem(fields, nk[].k_shape, value.shape)
    if format == "bfloat16":
        var bits = checkpoint_zeros(mods, rows * cols, value.shape, "H", py_item(context, nk[].k_array_class))
        _model_linalg(mods, key, CAUSAL_LIN_TO_BF16, [Int(py=bits._addr), Int(py=value._addr)], [rows * cols])
        _ = bits  # keep the Python object alive through the native call above
        py_setitem(fields, nk[].k_bits, bits)
    else:
        var codes = generation_array([rows, cols], "b")
        var exponents = generation_array([rows], "i")
        _model_linalg(mods, key, CAUSAL_LIN_QUANTIZE_INT8, [Int(py=codes._addr), Int(py=exponents._addr), Int(py=value._addr)], [rows, cols])
        _ = exponents  # keep the Python object alive through the native call above
        _ = codes  # keep the Python object alive through the native call above
        py_setitem(fields, nk[].k_codes, codes)
        py_setitem(fields, nk[].k_exponents, exponents)
    return packed


def _model_unpack(mut mods: NativeModules, value: PythonObject, key: CausalEntryKey, context: PythonObject, name: String = "weight") raises -> PythonObject:
    var bi = mods.module("builtins")
    var nk = causal_keys()
    var bf16 = py_type(value) is py_item(context, nk[].k_bf16_class)
    var int8 = py_type(value) is py_item(context, nk[].k_int8_class)
    if not bf16 and not int8:
        return _model_f32(value, name, context)
    var fields = _model_packed_fields(mods, value, context)
    var rows = Int(py=py_item(fields, nk[].k_shape)[0])
    var cols = Int(py=py_item(fields, nk[].k_shape)[1])
    var out = generation_array([rows, cols])
    if bf16:
        var bits = _model_native_buffer(mods, py_item(fields, nk[].k_bits), "H", context)
        _model_shape(bits, [rows, cols], "bfloat16 bits")
        _model_linalg(mods, key, CAUSAL_LIN_FROM_BF16, [Int(py=out._addr), Int(py=bits._addr)], [rows * cols])
        _ = bits  # keep the Python object alive through the native call above
        _ = out  # keep the Python object alive through the native call above
        _ = bits^  # read by address above
    else:
        var codes = _model_native_buffer(mods, py_item(fields, nk[].k_codes), "b", context)
        var exponents = _model_native_buffer(mods, py_item(fields, nk[].k_exponents), "i", context)
        _model_shape(codes, [rows, cols], "int8 codes")
        _model_shape(exponents, [rows], "int8 exponents")
        _model_linalg(mods, key, CAUSAL_LIN_DEQUANTIZE_INT8, [Int(py=out._addr), Int(py=codes._addr), Int(py=exponents._addr)], [rows, cols])
        _ = exponents  # keep the Python object alive through the native call above
        _ = codes  # keep the Python object alive through the native call above
        _ = out  # keep the Python object alive through the native call above
        _ = codes^  # read by address above
        _ = exponents^
    return out


def _model_unpack(value: PythonObject, key: CausalEntryKey, context: PythonObject, name: String = "weight") raises -> PythonObject:
    # Classical and other callers keep their per-call module checks.
    var mods = NativeModules()
    return _model_unpack(mods, value, key, context, name)


def _model_quantize15(mut mods: NativeModules, value: PythonObject, key: CausalEntryKey) raises -> PythonObject:
    _causal_array_guard()
    var rows = Int(py=value.shape[0])
    var cols = Int(py=value.shape[1])
    var hi = generation_array([rows, cols], "b")
    var lo = generation_array([rows, cols], "b")
    var exponents = generation_array([rows], "i")
    _model_linalg(mods, key, CAUSAL_LIN_QUANTIZE_INT15, [Int(py=hi._addr), Int(py=lo._addr), Int(py=exponents._addr), Int(py=value._addr)], [rows, cols])
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
    var nk = causal_keys()
    var profiles = py_item(context, nk[].k_profiles)
    var defaults = py_item(context, nk[].k_profile_defaults)
    var selected = requested
    if selected is None:
        selected = py_get(defaults, nk[].k_inference)
        if selected is None:
            var env = getenv("MOJOLEARN_NUMERIC_PROFILE", "")
            selected = PythonObject(env).strip()
            if len(selected) == 0:
                selected = PythonObject("fp32_v1")
    if py_type(selected) is not bi.str:
        raise Error("TypeError: numeric_profile must be a profile name")
    selected = selected.strip()
    if not py_contains(profiles, selected):
        var reason = py_get(py_item(context, nk[].k_rejected_profiles), selected)
        raise Error("numeric_profile=" + String(selected) + " is not offered: " + (String(reason) if reason is not None else String("not a registered numeric profile")))
    var row = profiles[selected]
    if not Bool(py=py_item(row, nk[].k_inference)):
        raise Error("NotImplementedError: numeric_profile=" + String(selected) + " is unavailable for inference")
    if requested is None and py_get(defaults, nk[].k_inference) is None:
        py_setitem(defaults, nk[].k_inference, selected)
    var computes = py_item(row, nk[].k_computes)
    var covered = computes is None or py_contains(computes, family)
    if requested is None:
        var reaches = py_item(row, nk[].k_default_for)
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
    var nk = causal_keys()
    if not py_contains(py_item(context, nk[].k_profiles), selected):
        var reason = py_get(py_item(context, nk[].k_rejected_profiles), selected)
        raise Error("numeric_profile=" + String(selected) + " is not offered: "
                    + (String(reason) if reason is not None else String("not a registered numeric profile")))
    if not Bool(py=py_item(py_item(context, nk[].k_profiles)[selected], nk[].k_inference)):
        raise Error("NotImplementedError: numeric_profile=" + String(selected) + " is unavailable for inference")


def _model_route(mut mods: NativeModules, device: PythonObject, context: PythonObject) raises -> PythonObject:
    if py_type(device) is not mods.module("builtins").str:
        raise Error("TypeError: CausalLM device requires an ordinary route name")
    var route = String(device)
    if route == "auto":
        var nk = causal_keys()
        route = "cpu" if py_item(context, nk[].k_selected_vendor) is None or String(py_item(context, nk[].k_selected_vendor)) == "cpu" else "gpu"
    if route != "cpu" and route != "gpu":
        raise Error("CausalLM: device must be 'auto', 'cpu' or 'gpu'")
    return PythonObject(route)


def _model_bindings(mut mods: NativeModules, route: PythonObject, kind: String, context: PythonObject, requested_mode: PythonObject) raises -> PythonObject:
    var bi = mods.module("builtins")
    if requested_mode is not None and py_type(requested_mode) is not bi.str:
        raise Error("TypeError: CausalLM numeric mode requires an ordinary string")
    var nk = causal_keys()
    var mode = PythonObject("identical") if String(route) == "cpu" else (py_item(_native_entry_fields(mods, py_item(context, nk[].k_backend)), nk[].k__DEFAULT_MODE) if requested_mode is None else requested_mode.strip().lower())
    if py_type(mode) is not bi.str:
        raise Error("TypeError: CausalLM connected numeric mode requires an ordinary string")
    # lane fx-fast-mask: a requested `fast` that the identical set answers
    # (NEURAL) selects the identical binding; the mode check below then
    # compares that set with its binary.
    var mode_word = String(py=mode)
    answer_mode_object(mode, mode_word, MODE_FAMILY_NEURAL)
    var key = route + PythonObject(":") + mode
    var entries = py_get(py_item(context, nk[].k_bindings), key)
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
    # The CausalLM entry table key of this route (core/causal_lm_entry_table.mojo):
    # every linalg, primitive, block and Mamba call of the model reads its
    # image's entry by these two words.
    var entry_key = CausalEntryKey(mode_code, causal_entry_vendor(vendor))
    py_setitem(entries, nk[].k_entry_mode, PythonObject(entry_key.mode))
    py_setitem(entries, nk[].k_entry_vendor, PythonObject(entry_key.vendor))
    var arithmetic = py_item(entries, nk[].k_linalg)
    if py_type(arithmetic) is not mods.module("builtins").str:
        # A linalg publication under the model's (tier, vendor) key is the
        # vendor/mode agreement the former two Python metadata calls checked.
        try:
            _ = causal_table_entry[CAUSAL_TABLE_LINALG](entry_key)
        except problem:
            raise Error("CausalLM: packed weight binding vendor or mode differs from model (" + String(problem) + ")")
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
        var nk = causal_keys()
        source = model_record_fields(mods, value, py_item(py_item(context, nk[].k_config_context), nk[].k_plan_class), names)
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
    var nk = causal_keys()
    var layers = py_item(weights, nk[].k_layers)
    if py_type(layers) is not bi.list and py_type(layers) is not bi.tuple:
        raise Error("TypeError: CausalLM layers require an ordinary weight-map sequence")
    var copied = Python.list()
    for i in range(len(layers)):  # driver-loop: capture each complete layer weight map
        _ = py_call(copied.append, neural_model_weights(mods, layers[i], context))
    py_setitem(result, nk[].k_layers, copied)
    var unused = py_get(weights, nk[].k_unused, Python.tuple())
    neural_plain_value(mods, unused, 0, context)
    if py_type(unused) is not bi.list and py_type(unused) is not bi.tuple:
        raise Error("TypeError: CausalLM unused tensor names require an ordinary sequence")
    py_setitem(result, nk[].k_unused, neural_metadata_snapshot(mods, unused))
    return result


def _model_recurrent_shapes(mut mods: NativeModules, kind: String, dm: Int, plan: PythonObject, fields: PythonObject) raises -> PythonObject:
    # Per-kind recurrent block configuration and current-weight shapes. Shared
    # by block construction and the recurrent backward weight check so both
    # admit exactly the same configuration, in the same order.
    var bi = mods.module("builtins")
    var shapes = bi.list()
    var di = 2 * dm
    var nk = causal_keys()
    py_setitem(fields, nk[].k_d_model, PythonObject(dm))
    py_setitem(fields, nk[].k_d_inner, PythonObject(di))
    if kind == "mamba1":
        var rank = (dm + 15) // 16
        py_setitem(fields, nk[].k_dt_rank, PythonObject(rank))
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
        py_setitem(fields, nk[].k_nheads, PythonObject(nh))
        py_setitem(fields, nk[].k_conv_dim, PythonObject(cd))
        var defaults = bi.list()
        _ = py_call(defaults.append, 0.0)
        _ = py_call(defaults.append, py_call(bi.float, "inf"))
        var limit = py_get(py_item(plan, nk[].k_block_kwargs), nk[].k_dt_limit, defaults)
        var pair = bi.list()
        _ = py_call(pair.append, py_call(bi.float, limit[0]))
        _ = py_call(pair.append, py_call(bi.float, limit[1]))
        py_setitem(fields, nk[].k_dt_limit, py_call(bi.tuple, pair))
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
        py_setitem(fields, nk[].k_nheads, PythonObject(nh))
        py_setitem(fields, nk[].k_d_in_proj, PythonObject(dip))
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
    var nk = causal_keys()
    var kind = String(py_item(plan, nk[].k_kind))
    var cls = py_item(context, nk[].k_block_classes)[route][kind]
    var block = py_call(bi.object.__new__, cls)
    var fields = neural_block_fields(mods, block, context, False)
    py_setitem(fields, nk[].k_numeric_mode, PythonObject("identical") if String(route) == "cpu" else py_item(_native_entry_fields(mods, py_item(context, nk[].k_backend)), nk[].k__DEFAULT_MODE))
    var names = neural_block_weight_names(mods, block)
    var w = bi.dict()
    var keys = py_call(bi.list, weights)
    var format = "float32"
    # The route's (tier, vendor) table key, read once for every tensor.
    var table_key = causal_entry_key(bindings)
    for i in range(len(keys)):  # driver-loop: over the layer's named tensors
        var v = weights[keys[i]]
        var tag = "bfloat16" if py_type(v) is py_item(context, nk[].k_bf16_class) else ("int8" if py_type(v) is py_item(context, nk[].k_int8_class) else "float32")
        if tag != "float32":
            if format != "float32" and format != tag:
                raise Error("CausalLM: mixed packed weight formats")
            format = tag
        py_setitem(w, keys[i], _model_unpack(mods, v, table_key, context, String(keys[i])))
    _causal_array_guard()
    py_setitem(fields, nk[].k_weight_format, PythonObject(format))
    py_setitem(fields, nk[].k__native_kind, py_item(plan, nk[].k_kind))
    py_setitem(fields, nk[].k__native_route, route)
    var dm = Int(py=py_item(plan, nk[].k_d_model))
    var base = bi.list()
    var optional = bi.list()
    var expected = bi.set()
    var shapes = bi.list()
    var gated = True
    var it = 0
    if kind == "transformer":
        var options = py_call(bi.dict, py_item(py_item(context, nk[].k_config_context), nk[].k_defaults))
        _ = py_call(options.update, py_item(plan, nk[].k_block_options))
        var nh = Int(py=py_item(plan, nk[].k_n_heads))
        if nh < 1:
            raise Error("CausalLM: n_heads must be positive")
        var nkv = nh if py_item(options, nk[].k_n_kv_heads) is None else Int(py=py_item(options, nk[].k_n_kv_heads))
        var hd = dm // nh if py_item(options, nk[].k_head_dim) is None else Int(py=py_item(options, nk[].k_head_dim))
        if nh < 1 or nkv < 1 or hd < 1 or dm != nh * hd:
            raise Error("CausalLM: d_model must equal n_heads * head_dim, all positive")
        var mlp = String(py_item(options, nk[].k_mlp))
        var norm = String(py_item(options, nk[].k_norm))
        var norm_code = 0 if norm == "rmsnorm" else (1 if norm == "layernorm" else (2 if norm == "rmsnorm_offset" else -1))
        var mk = 0 if mlp == "swiglu" else (1 if mlp == "gelu" else (2 if mlp == "gelu_tanh" else (3 if mlp == "geglu" else (4 if mlp == "geglu_tanh" else -1))))
        if norm_code < 0:
            raise Error("CausalLM: unsupported norm=" + String(py_call(bi.repr, py_item(options, nk[].k_norm)))
                        + "; norm is 'rmsnorm', 'layernorm' or 'rmsnorm_offset'")
        if mk < 0:
            raise Error("CausalLM: unsupported mlp=" + String(py_call(bi.repr, py_item(options, nk[].k_mlp)))
                        + "; mlp is 'swiglu', 'gelu', 'gelu_tanh', 'geglu' or 'geglu_tanh'")
        gated = mk == 0 or mk == 3 or mk == 4
        var has_gate = py_contains(w, "gate_proj.weight")
        if gated and not has_gate:
            raise Error("CausalLM: gate_proj.weight is missing; mlp=" + String(py_call(bi.repr, py_item(options, nk[].k_mlp))) + " is gated")
        if not gated and has_gate:
            raise Error("CausalLM: gate_proj.weight is present but mlp=" + String(py_call(bi.repr, py_item(options, nk[].k_mlp)))
                        + " has no gate")
        var source = w["gate_proj.weight" if gated else "up_proj.weight"]
        if len(source.shape) != 2 or Int(py=source.shape[0]) < 1:
            raise Error("CausalLM: projection must have positive intermediate size")
        it = Int(py=source.shape[0])
        var tail = List[Int](length=17, fill=0)
        tail[0] = _model_bits(mods, py_item(options, nk[].k_rope_theta), "rope_theta", context)
        tail[2] = 0x3f800000
        tail[3] = 0x3f800000
        tail[4] = 0x40800000
        tail[5] = 8192
        var scaling = py_item(options, nk[].k_rope_scaling)
        if scaling is not None:
            var sk = String(py_get(scaling, nk[].k_type, py_get(scaling, nk[].k_rope_type)))
            if sk != "linear" and sk != "llama3":
                raise Error("CausalLM: unsupported rope_scaling type " + String(py_call(bi.repr, PythonObject(sk)))
                            + "; the types are 'linear' and 'llama3'")
            tail[1] = 1 if sk == "linear" else 2
            tail[2] = _model_bits(mods, py_item(scaling, nk[].k_factor), "rope_scaling factor", context)
            if sk == "llama3":
                tail[3] = _model_bits(mods, py_item(scaling, nk[].k_low_freq_factor), "low_freq_factor", context)
                tail[4] = _model_bits(mods, py_item(scaling, nk[].k_high_freq_factor), "high_freq_factor", context)
                tail[5] = Int(py=py_item(scaling, nk[].k_original_max_position_embeddings))
                if Float64(py=py_item(scaling, nk[].k_high_freq_factor)) <= Float64(py=py_item(scaling, nk[].k_low_freq_factor)) or tail[5] < 1:
                    raise Error("CausalLM: llama3 rope_scaling needs high_freq_factor > low_freq_factor "
                                + "and a positive original_max_position_embeddings")
        var rd = hd if py_item(options, nk[].k_rope_dim) is None else Int(py=py_item(options, nk[].k_rope_dim))
        if rd < 1 or rd > hd or rd % 2 != 0:
            raise Error("CausalLM: rope_dim must be positive even and no larger than head_dim")
        tail[6] = 0 if rd == hd else rd
        tail[7] = Int(py=py_item(options, nk[].k_max_positions))
        tail[8] = Int(Bool(py=py_item(options, nk[].k_qkv_bias)))
        tail[9] = Int(Bool(py=py_item(options, nk[].k_o_bias)))
        tail[10] = norm_code
        tail[11] = _model_bits(mods, py_item(options, nk[].k_norm_eps), "norm_eps", context)
        tail[12] = Int(Bool(py=py_item(options, nk[].k_norm_bias)))
        tail[13] = mk
        tail[14] = Int(Bool(py=py_item(options, nk[].k_mlp_bias)))
        tail[15] = Int(Bool(py=py_item(options, nk[].k_qk_norm)))
        tail[16] = 0 if py_item(options, nk[].k_attn_softcap) is None else _model_bits(mods, py_item(options, nk[].k_attn_softcap), "attn_softcap", context)
        if tail[7] < 1:
            raise Error("CausalLM: max_positions must be positive")
        if tail[12] != 0 and norm_code != 1:
            raise Error("CausalLM: norm_bias=True needs norm='layernorm'")
        if tail[15] != 0 and norm_code == 1:
            raise Error("CausalLM: qk_norm=True with norm='layernorm' is not supported (qk_norm is an RMSNorm)")
        var window = Int(py=py_item(options, nk[].k_window))
        if window < 0:
            raise Error("CausalLM: sliding window must be nonnegative")
        _ = py_call(fields.update, options)
        py_setitem(fields, nk[].k_d_model, PythonObject(dm))
        py_setitem(fields, nk[].k_n_heads, PythonObject(nh))
        py_setitem(fields, nk[].k_n_kv_heads, PythonObject(nkv))
        py_setitem(fields, nk[].k_head_dim, PythonObject(hd))
        py_setitem(fields, nk[].k_intermediate, PythonObject(it))
        py_setitem(fields, nk[].k_rope_dim, PythonObject(rd))
        py_setitem(fields, nk[].k__opts_tail, py_ints(tail))
        var default_tail = [0x461c4000, 0, 0x3f800000, 0x3f800000, 0x40800000, 8192, 0, 8192, 0, 0, 0, 0x358637bd, 0, 0, 0, 0, 0]
        var extended = False
        for i in range(17):  # driver-loop: over the fixed block options record
            extended = extended or tail[i] != default_tail[i]
        py_setitem(fields, nk[].k__extended, PythonObject(extended))
        py_setitem(fields, nk[].k__gated, PythonObject(gated))
        if String(py_item(fields, nk[].k_numeric_mode)) == "identical" and NEURAL_NATIVE_DISPATCH:
            py_setitem(fields, nk[].k__runtime_lock, operation_lock_create())
            mods.released()
        else:
            py_setitem(fields, nk[].k__runtime_lock, py_item(context, nk[].k_rlock)())
            mods.released()
        var session = PythonObject(None)
        if String(route) == "gpu" and String(py_item(fields, nk[].k_numeric_mode)) == "identical" and NEURAL_NATIVE_DISPATCH:
            # The block's session: a native registry handle of its GPU
            # Transformer image, created through the block entry of the neural
            # table (core/transformer_block_session.mojo; lane
            # transformer-session-native). Replaces the former Python entry
            # `transformer_session_create` and the `_session_binding` module
            # identity the owner's resolved entry now carries.
            session = transformer_block_session_open(neural_ref[NEURAL_TABLE_BLOCK_FORWARD](
                NeuralEntryKey(Int(py=py_item(bindings, nk[].k_entry_mode)), Int(py=py_item(bindings, nk[].k_entry_vendor)))))
        py_setitem(fields, nk[].k__native_session, session)
        py_setitem(fields, nk[].k__arithmetic_profile, PythonObject(None))
        if _native_has_entry(mods, bindings[kind], "transformer_arithmetic_profile"):
            py_setitem(fields, nk[].k__arithmetic_profile, _native_function(mods, bindings[kind], "transformer_arithmetic_profile")())
            mods.released()
            py_setitem(fields, nk[].k__arithmetic_profile_mode, py_item(fields, nk[].k_numeric_mode))
            py_setitem(fields, nk[].k__arithmetic_profile_changed, PythonObject(Bool(py=_native_function(mods, bindings[kind], "transformer_arithmetic_profile_changed")())))
            mods.released()
        py_setitem(fields, nk[].k_numeric_profile, profile)
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
            var name = py_item(context, nk[].k_optional_names)[i]
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
        py_setitem(fields, nk[].k__opts_flags, flag_dict)
        py_setitem(fields, nk[].k__wopt, optional)
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
    py_setitem(fields, nk[].k__w, base)
    py_setitem(fields, nk[].k__int15, PythonObject(None))
    if String(profile) == "fixed15_v1":
        var planes = bi.list()
        var quantize_key = causal_entry_key(bindings)
        for i in range(2, 9):  # driver-loop: over the seven transformer projections
            _ = py_call(planes.append, PythonObject(None) if base[i] is None else _model_quantize15(mods, base[i], quantize_key))
        py_setitem(fields, nk[].k__int15, planes)
    if kind == "mamba3":
        if String(route) == "gpu" and String(py_item(fields, nk[].k_numeric_mode)) == "identical" and NEURAL_NATIVE_DISPATCH:
            # The block's prefill session: a native registry handle of its
            # GPU Mamba image (core/mamba3_prefill_session.mojo, lane
            # mamba3-prefill-handle; the former Python entry
            # mamba3_prefill_session_create and the _prefill_binding field).
            var session_entry = neural_ref[NEURAL_TABLE_MAMBA_TRAIN](
                NeuralEntryKey(Int(py=py_item(bindings, nk[].k_entry_mode)), Int(py=py_item(bindings, nk[].k_entry_vendor))))
            if (mamba3_prefill_caps(session_entry) & MAMBA3_CAP_SESSION_PREFILL) != 0:
                py_setitem(fields, nk[].k__prefill_session, mamba3_prefill_session_open(session_entry))
        _ = neural_descriptor_cached(fields, bindings[kind], "_native_mamba_state", "mamba_state_api_v1",
            String(MAMBA_STATE_ABI_NAME), String(route) == "gpu")
        mods.released()
    return block


def _model_class_weight_names(mut mods: NativeModules, cls: PythonObject) raises -> PythonObject:
    # The weight registry a fresh instance of `cls` reports: the same lookup
    # and checks as neural_block_weight_names, read from the class ancestry.
    var bi = mods.module("builtins")
    var names = PythonObject(None)
    var nk = causal_keys()
    var ancestry = py_call(py_item(bi.type.__dict__, nk[].k___mro__).__get__, cls)
    for i in range(len(ancestry)):  # driver-loop: native admission metadata and retained owner registry
        names = py_get(_class_fields(ancestry[i]), nk[].k__W_NAMES)
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
    var nk = causal_keys()
    var names = _model_class_weight_names(mods, py_item(context, nk[].k_block_classes)[route][kind])
    var w = bi.dict()
    var keys = py_call(bi.list, weights)
    for i in range(len(keys)):  # driver-loop: over the layer's named tensors
        var v = weights[keys[i]]
        # Snapshot tensors are already admitted float32 C Arrays owned by this
        # operation; anything else takes the constructor's unpack admission.
        if py_type(v) is not py_item(context, nk[].k_array_class):
            v = _model_unpack(mods, v, causal_entry_key(bindings), context, String(keys[i]))
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
        var nk = causal_keys()
        # Caller-visible Array slots must not be the metadata traversed by
        # subsequent constructor kernels. Retain the same live storage and
        # preserve the tied head/embedding object's public identity.
        var buffers = py_get(captured, nk[].k__construction_published_buffers)
        if buffers is None:
            buffers = Python.dict()
            py_setitem(captured, nk[].k__construction_published_buffers, buffers)
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
    var nk = causal_keys()
    var profile = _model_profile(mods, options[3], String(py_item(plan, nk[].k_kind)), context)
    var bindings = _model_bindings(mods, route, String(py_item(plan, nk[].k_kind)), context)
    _model_construction_set(mods, live, captured, "_runtime", py_item(context, nk[].k_runtime))
    _model_construction_set(mods, live, captured, "plan", input_plan)
    _model_construction_set(mods, live, captured, "config", py_item(plan, nk[].k_config))
    _model_construction_set(mods, live, captured, "kind", py_item(plan, nk[].k_kind))
    _model_construction_set(mods, live, captured, "d_model", py_item(plan, nk[].k_d_model))
    _model_construction_set(mods, live, captured, "vocab_size", py_item(plan, nk[].k_vocab_size))
    _model_construction_set(mods, live, captured, "n_layers", py_item(plan, nk[].k_n_layers))
    _model_construction_set(mods, live, captured, "norm_eps", py_item(plan, nk[].k_norm_eps))
    _model_construction_set(mods, live, captured, "numeric_profile", profile)
    _model_construction_set(mods, live, captured, "device", route)
    _model_construction_set(mods, live, captured, "weight_format", options[0])
    _model_construction_set(mods, live, captured, "unused_names", py_call(bi.list, py_get(weights, nk[].k_unused, bi.tuple())))
    var dm = Int(py=py_item(plan, nk[].k_d_model))
    var vocab = Int(py=py_item(plan, nk[].k_vocab_size))
    if dm < 1 or vocab < 1 or Int(py=py_item(plan, nk[].k_n_layers)) < 1:
        raise Error("CausalLM: model dimensions and layer count must be positive")
    var positions = options[1]
    if String(py_item(plan, nk[].k_kind)) == "transformer":
        var config_cap = 8192 if py_item(plan, nk[].k_max_position_embeddings) is None else Int(py=py_item(plan, nk[].k_max_position_embeddings))
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
    _model_construction_set(mods, live, captured, "_embed", _model_f32(py_item(weights, nk[].k_embed), String(py_item(plan, nk[].k_embed_name)), context))
    _causal_array_guard()
    if not Bool(py=py_item(captured, nk[].k__embed).shape == py_call(bi.tuple, py_ints([vocab, dm]))):
        raise Error("UnsupportedModel: model_type " + String(py_item(plan, nk[].k_model_type)) + ": vocab_size=" + String(vocab) + " and hidden_size=" + String(dm) + " but " + String(py_item(plan, nk[].k_embed_name)) + " has shape " + String(py_call(bi.str, py_item(captured, nk[].k__embed).shape)))
    _model_construction_set(mods, live, captured, "_norm", PythonObject(None))
    if py_item(plan, nk[].k_norm_name) is not None:
        _model_construction_set(mods, live, captured, "_norm", _model_f32(py_item(weights, nk[].k_norm), String(py_item(plan, nk[].k_norm_name)), context))
        _causal_array_guard()
        if not Bool(py=py_item(captured, nk[].k__norm).shape == py_call(bi.tuple, py_ints([dm]))):
            raise Error("UnsupportedModel: tensor " + String(py_item(plan, nk[].k_norm_name)) + " has shape " + String(py_call(bi.str, py_item(captured, nk[].k__norm).shape)) + ", want hidden_size=" + String(dm))
    _model_construction_set(mods, live, captured, "_head", py_item(captured, nk[].k__embed) if py_item(plan, nk[].k_head_name) is None else _model_f32(py_item(weights, nk[].k_head), String(py_item(plan, nk[].k_head_name)), context))
    _causal_array_guard()
    if not Bool(py=py_item(captured, nk[].k__head).shape == py_call(bi.tuple, py_ints([vocab, dm]))):
        raise Error("UnsupportedModel: tensor " + String(py_item(plan, nk[].k_head_name)) + " has shape " + String(py_call(bi.str, py_item(captured, nk[].k__head).shape)) + ", want vocab_size=" + String(vocab) + " and hidden_size=" + String(dm))
    for index in range(3):  # driver-loop: over the three top-level tensors
        var value = py_item(captured, nk[].k__embed) if index == 0 else (py_item(captured, nk[].k__norm) if index == 1 else py_item(captured, nk[].k__head))
        # Tied embeddings: the head is the embedding Array; scan it once.
        if index == 2 and value is py_item(captured, nk[].k__embed):
            continue
        if value is not None:
            _causal_array_guard()
            var ptr = f32_ptr(Int(py=value._addr))
            causal_lm_check_finite(ptr, Int(py=value.size))
    var layers = py_item(weights, nk[].k_layers)
    if len(layers) != Int(py=py_item(plan, nk[].k_n_layers)):
        raise Error("CausalLM: layer dict count differs from num_hidden_layers")
    return bindings


def causal_model_init_binding(mut mods: NativeModules, model: PythonObject, input_plan: PythonObject, input_weights: PythonObject, input_options: PythonObject, context: PythonObject) raises -> PythonObject:
    causal_public_connect(context)
    var plan = model_plan_snapshot(mods, input_plan)
    return _causal_model_init(mods, model, input_plan, plan, input_weights, input_options, context)


def _causal_model_init(mut mods: NativeModules, model: PythonObject, input_plan: PythonObject, plan: PythonObject, input_weights: PythonObject, input_options: PythonObject, context: PythonObject) raises -> PythonObject:
    # `plan` is model_plan_snapshot(input_plan), taken once by the caller
    # (row N203: CausalLM.load passes the snapshot it already holds).
    causal_public_connect(context)
    var live = _causal_model_fields(mods, model)
    var captured = Python.dict()
    var options = _model_options_snapshot(mods, input_options, context)
    var weights = _model_weights_snapshot(mods, input_weights, context)
    var bindings = _causal_model_prepare_snapshot(mods, model, input_plan, plan, weights, options, context, live, captured)
    var bi = mods.module("builtins")
    var format = String(options[0])
    var nk = causal_keys()
    var route = py_item(captured, nk[].k_device)
    var profile = py_item(captured, nk[].k_numeric_profile)
    var layers = py_item(weights, nk[].k_layers)
    _model_construction_set(mods, live, captured, "_blocks", bi.list())
    for i in range(len(layers)):  # driver-loop: over the model layer definitions
        var block = _model_block(mods, layers[i], plan, profile, route, bindings, context)
        var fields = neural_block_fields(mods, block, context)
        if String(py_item(fields, nk[].k_weight_format)) != format:
            raise Error("CausalLM: layer reports a different weight_format than requested")
        _ = py_call(py_item(captured, nk[].k__blocks).append, block)
        _model_construction_set(mods, live, captured, "_blocks", py_item(captured, nk[].k__blocks))
    _model_construction_set(mods, live, captured, "_block_class", py_item(context, nk[].k_block_classes)[route][py_item(plan, nk[].k_kind)])
    _model_construction_set(mods, live, captured, "_block_kwargs", py_call(bi.dict, py_item(plan, nk[].k_block_options) if String(py_item(plan, nk[].k_kind)) == "transformer" else py_item(plan, nk[].k_block_kwargs)))
    if String(py_item(plan, nk[].k_kind)) == "transformer":
        py_setitem(py_item(captured, nk[].k__block_kwargs), nk[].k_n_heads, py_item(plan, nk[].k_n_heads))
        py_setitem(py_item(captured, nk[].k__block_kwargs), nk[].k_numeric_profile, profile)
        _model_construction_set(mods, live, captured, "_block_kwargs", py_item(captured, nk[].k__block_kwargs))
    var primitive_class = py_item(context, nk[].k_primitive_classes)[route]
    var primitive = py_call(bi.object.__new__, primitive_class)
    var primitive_fields = model_record_fields(mods, primitive, primitive_class, Python.tuple("_T", "_ext"))
    if String(route) == "gpu":
        py_setitem(primitive_fields, nk[].k__T, py_item(context, nk[].k_training_module))
    else:
        py_setitem(primitive_fields, nk[].k__ext, py_item(bindings, nk[].k_primitives))
    _model_construction_set(mods, live, captured, "_prims", primitive)
    _model_construction_set(mods, live, captured, "_head_int15", PythonObject(None))
    if String(profile) == "fixed15_v1":
        _model_construction_set(mods, live, captured, "_head_int15", _model_quantize15(mods, py_item(captured, nk[].k__head), causal_entry_key(bindings)))
    # Admission may call sibling native entries. Give it the private record;
    # a replaced public model descriptor cannot run after those calls.
    py_setitem(captured, nk[].k__native_context, context)
    var admission = _causal_namespace(mods, captured)
    var admitted = causal_snapshot_fields(mods, admission)
    try:
        _ = causal_lm_admit_binding(mods, admission, bindings[py_item(plan, nk[].k_kind)], py_item(bindings, nk[].k_primitives), py_item(bindings, nk[].k_linalg) if py_item(captured, nk[].k__head_int15) is not None else PythonObject(None))
    except problem:
        mods.released()
        _model_construction_admission(mods, live, admitted)
        raise problem
    _model_construction_admission(mods, live, admitted)
    return PythonObject(None)


def causal_model_read_weights_binding(mut mods: NativeModules, checkpoint: PythonObject, input_plan: PythonObject, format: PythonObject, route: PythonObject, context: PythonObject) raises -> PythonObject:
    causal_public_connect(context)
    var plan = model_plan_snapshot(mods, input_plan)
    return _causal_model_read_weights(mods, checkpoint, plan, format, route, context)


def _causal_model_read_weights(mut mods: NativeModules, checkpoint: PythonObject, plan: PythonObject, format: PythonObject, route: PythonObject, context: PythonObject) raises -> PythonObject:
    # `plan` is the caller's model_plan_snapshot (row N203).
    var bi = mods.module("builtins")
    causal_public_connect(context)
    if py_type(format) is not bi.str:
        raise Error("TypeError: CausalLM weight format requires an ordinary string")
    var nk = causal_keys()
    var ck = py_item(context, nk[].k_checkpoint_context)
    var infos = checkpoint_index_binding(mods, checkpoint, ck)
    var captured_infos = Python.dict()
    # Row N203: the captured shape/dtype metadata is read only by the
    # bfloat16 packed-projection test below; other formats skip the capture.
    if String(format) == "bfloat16":
        var info_names = py_call(bi.list, infos.keys())
        for i in range(len(info_names)):  # driver-loop: retain checkpoint shape/dtype metadata before execution
            py_setitem(captured_infos, info_names[i], _info_snapshot(mods, infos[info_names[i]]))
    var need = model_checkpoint_names_snapshot(mods, plan)
    for i in range(len(need)):  # driver-loop: over checkpoint tensor names
        if not py_contains(infos, need[i]):
            raise Error("UnsupportedModel: checkpoint lacks tensor " + String(need[i]))
    var out = bi.dict()
    py_setitem(out, nk[].k_embed, checkpoint_read_binding(mods, checkpoint, py_item(plan, nk[].k_embed_name), PythonObject("widen"), ck))
    if py_item(plan, nk[].k_norm_name) is not None:
        py_setitem(out, nk[].k_norm, checkpoint_read_binding(mods, checkpoint, py_item(plan, nk[].k_norm_name), PythonObject("widen"), ck))
    if py_item(plan, nk[].k_head_name) is not None:
        py_setitem(out, nk[].k_head, checkpoint_read_binding(mods, checkpoint, py_item(plan, nk[].k_head_name), PythonObject("widen"), ck))
    var bindings = _model_bindings(mods, route, String(py_item(plan, nk[].k_kind)), context)
    # The route's (tier, vendor) table key, read once for every packed tensor.
    var pack_key = causal_entry_key(bindings)
    var projections = py_item(context, nk[].k_projections)[py_item(plan, nk[].k_kind)]
    var layers = bi.list()
    for i in range(Int(py=py_item(plan, nk[].k_n_layers))):  # driver-loop: over checkpoint layers
        var weights = bi.dict()
        var mapping = _model_layer_weights(mods, plan, i)
        for j in range(len(mapping)):  # driver-loop: over named layer tensor entries
            var entry = mapping[j]
            var key = entry[0]
            var name = entry[1]
            var rows = entry[2]
            var packed = String(format) != "float32" and py_contains(projections, key)
            if packed and String(format) == "bfloat16" and String(py_item(captured_infos[name], nk[].k_dtype)) == "BF16" and rows is None and len(py_item(captured_infos[name], nk[].k_shape)) == 2:
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
                    unsafe_memcpy(dest=f32_ptr(Int(py=sliced._addr)), src=f32_ptr(Int(py=whole._addr)) + begin * cols, count=(end - begin) * cols)
                    _ = whole  # keep the Python object alive through the native call above
                    _ = sliced  # keep the Python object alive through the native call above
                _ = whole^  # read by address above
                value = sliced
            py_setitem(weights, key, _model_pack(mods, value, String(format), pack_key, context) if packed else value)
        _ = py_call(layers.append, weights)
    py_setitem(out, nk[].k_layers, layers)
    var unused = bi.list()
    var names = py_call(bi.list, infos)
    for i in range(len(names)):  # driver-loop: over checkpoint metadata names
        if not py_contains(need, names[i]):
            _ = py_call(unused.append, names[i])
    py_setitem(out, nk[].k_unused, unused)
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
    var nk = causal_keys()
    var config = config_load_binding(mods, py_item(py_item(context, nk[].k_config_context), nk[].k_config_class), path)
    var plan = model_plan_for_binding(mods, config, py_item(context, nk[].k_config_context))
    var plan_fields = model_plan_snapshot(mods, plan)
    _ = _model_profile(mods, options[3], String(py_item(plan_fields, nk[].k_kind)), context)
    var ck = py_item(context, nk[].k_checkpoint_context)
    var checkpoint = checkpoint_open_binding(mods, py_item(ck, nk[].k_checkpoint_class), path, ck)
    var weights = PythonObject(None)
    try:
        # Row N203: the one plan snapshot of this load serves the weight
        # read and the init (formerly rebuilt in each).
        weights = _causal_model_read_weights(mods, checkpoint, plan_fields, options[0], route, context)
    except error:
        mods.released()
        _ = checkpoint_close_binding(mods, checkpoint)
        raise error
    _ = checkpoint_close_binding(mods, checkpoint)
    var model = py_call(bi.object.__new__, cls)
    _ = _causal_model_init(mods, model, plan, plan_fields, weights, options, context)
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
