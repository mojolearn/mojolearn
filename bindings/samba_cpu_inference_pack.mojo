# SPDX-License-Identifier: Apache-2.0
"""Native ownership of the existing CPU-only SambaInference surface."""
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.generation_buffers_door import generation_initialize_binding, generation_copy, py_ints
from bindings.samba_state_native_door import samba_fields, _samba_registry
from bindings.samba_config_native_door import samba_config_from_dict, samba_block_shapes
from bindings.samba_checkpoint_native_pack import samba_checkpoint_read
from bindings.causal_lm_construct_door import _model_bindings, _model_block, _model_unpack, _model_profile, _model_shape
from bindings.causal_lm_public_door import _native_function, causal_entry_key
from bindings.causal_checkpoint_door import checkpoint_array
from bindings.py_balanced import py_call, py_get, py_setattr, py_setitem, py_type
from bindings.samba_keys_unpack import samba_keys
from bindings.py_balanced import py_item


def samba_inference_construct(model: PythonObject, config: PythonObject, weights: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = samba_keys()
    _ = generation_initialize_binding(py_item(py_item(context, nk[].k_neural), nk[].k_generation_classes))
    var bi = native_loaded_module("builtins")
    if not Bool(py=py_call(bi.isinstance, config, py_item(context, nk[].k_config_class))):
        raise Error("TypeError: SambaInference: config must be a SambaConfig")
    if py_type(weights) is not bi.dict:
        raise Error("TypeError: SambaInference: weights must be a dict of native buffers")
    var nc = py_item(context, nk[].k_neural)
    var bindings = _model_bindings(PythonObject("cpu"), "mamba3", nc, PythonObject("identical"))
    var registry = _samba_registry(config)
    var names = bi.set()
    for i in range(len(registry)):  # driver-loop: over admitted model registry metadata
        _ = py_call(names.add, py_item(registry[i], nk[].k_name))
    if Bool(py=py_call(bi.set, weights) != names):
        raise Error("SambaInference: weight dict mismatch: missing or unknown registry names")
    var arrays = bi.dict()
    var format = "float32"
    # The route's (tier, vendor) key is read once, not once per parameter.
    var entry_key = causal_entry_key(bindings)
    for i in range(len(registry)):  # driver-loop: over copied CPU inference parameter owners
        var item = registry[i]
        var name = py_item(item, nk[].k_name)
        var value = weights[name]
        var tag = "bfloat16" if py_type(value) is py_item(nc, nk[].k_bf16_class) else ("int8" if py_type(value) is py_item(nc, nk[].k_int8_class) else "float32")
        if tag != "float32":
            if format != "float32" and format != tag:
                raise Error("SambaInference: weights mix packed formats")
            format = tag
        var unpacked = _model_unpack(value, entry_key, nc)
        var shape = List[Int]()
        for axis in range(len(py_item(item, nk[].k_shape))):  # driver-loop: over tensor shape metadata
            shape.append(Int(py=py_item(item, nk[].k_shape)[axis]))
        _model_shape(unpacked, shape, String(name))
        if Int(py=py_call(py_item(context, nk[].k_all_finite_f32), unpacked._addr, unpacked.size)) != 1:
            raise Error("SambaInference: nonfinite weight " + String(name))
        py_setitem(arrays, name, generation_copy(unpacked))
    var fields = samba_fields(model)
    py_setitem(fields, nk[].k_config, config)
    py_setitem(fields, nk[].k_weight_format, format)
    py_setitem(fields, nk[].k__w, arrays)
    py_setitem(fields, nk[].k_arrays, arrays)
    py_setitem(fields, nk[].k_numeric_mode, "identical")
    py_setitem(fields, nk[].k__native_cpu_inference, True)
    var blocks = bi.list()
    var cfg = samba_fields(config)
    for i in range(len(py_item(cfg, nk[].k_layers))):  # driver-loop: over CPU block execution owners
        var kind = "mamba3" if String(py_item(cfg, nk[].k_layers)[i]) == "mamba3" else "transformer"
        var layer = bi.dict()
        var shapes = samba_block_shapes(config, String(py_item(cfg, nk[].k_layers)[i]))
        for j in range(len(shapes)):  # driver-loop: over one block's parameter registry
            var name = shapes[j][0]
            py_setitem(layer, name, arrays["layers." + String(i) + "." + String(name)])
        var plan = py_call(bi.object.__new__, py_item(py_item(nc, nk[].k_config_context), nk[].k_plan_class))
        py_setattr(plan, "kind", kind)
        py_setattr(plan, "d_model", py_item(cfg, nk[].k_d_model))
        py_setattr(plan, "block_options", bi.dict())
        py_setattr(plan, "block_kwargs", bi.dict())
        var profile = PythonObject("fp32_v1")
        if kind == "transformer":
            py_setattr(plan, "n_heads", py_item(cfg, nk[].k_n_heads))
            py_setitem(plan.block_options, nk[].k_n_kv_heads, py_item(cfg, nk[].k_n_kv_heads))
            py_setitem(plan.block_options, nk[].k_head_dim, py_item(cfg, nk[].k_head_dim))
            py_setitem(plan.block_options, nk[].k_norm_eps, PythonObject(1e-6))
            profile = _model_profile(PythonObject(None), "transformer", nc)
        _ = py_call(blocks.append, _model_block(layer, plan, profile, PythonObject("cpu"), bindings, nc))
    py_setitem(fields, nk[].k__blocks, blocks)
    return PythonObject(None)


def samba_inference_parameters(model: PythonObject) raises -> PythonObject:
    var nk = samba_keys()
    var arrays = py_item(samba_fields(model), nk[].k__w)
    var bi = native_loaded_module("builtins")
    var keys = py_call(bi.list, arrays)
    var result = bi.dict()
    for i in range(len(keys)):  # driver-loop: over independent public weight copies
        var name = keys[i]
        py_setitem(result, name, generation_copy(arrays[name]))
    return result


def samba_inference_checkpoint(cls: PythonObject, path: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = samba_keys()
    var state = samba_checkpoint_read(path, context)
    var bi = native_loaded_module("builtins")
    if String(py_get(state, nk[].k_schema)) != "mojolearn.samba-stack-state.v1" or String(py_get(state, nk[].k_profile)) != "mojolearn.samba-stack.fp32.v1":
        raise Error("SambaInference: checkpoint schema/profile mismatch")
    var config = samba_config_from_dict(py_item(context, nk[].k_config_class), py_item(state, nk[].k_config))
    var registry = _samba_registry(config)
    if Bool(py=registry != py_item(state, nk[].k_registry)):
        raise Error("SambaInference: checkpoint registry differs from config")
    var flat = py_item(state, nk[].k_parameters)
    var raw = flat._mv
    var weights = bi.dict()
    for i in range(len(registry)):  # driver-loop: over verified checkpoint parameter slices
        var item = registry[i]
        var view = raw[py_call(bi.slice, py_item(item, nk[].k_offset), py_item(item, nk[].k_offset) + py_item(item, nk[].k_size))]
        # checkpoint_array owns the copied bytes and shape; no Array constructor
        # or reshape callback is invoked by the loader.
        py_setitem(weights, py_item(item, nk[].k_name), checkpoint_array(view.tobytes(), py_item(item, nk[].k_shape), "f", py_item(py_item(context, nk[].k_neural), nk[].k_array_class)))
    var model = py_call(bi.object.__new__, cls)
    _ = samba_inference_construct(model, config, weights, context)
    return model


def samba_inference_repr(model: PythonObject) raises -> PythonObject:
    var nk = samba_keys()
    var bi = native_loaded_module("builtins")
    var config = samba_fields(py_item(samba_fields(model), nk[].k_config))
    return PythonObject("SambaInference(layers=" + String(py_call(bi.repr, py_call(bi.list, py_item(config, nk[].k_layers)))) + ", d_model=" + String(py_item(config, nk[].k_d_model)) + ", vocab=" + String(py_item(config, nk[].k_vocab)) + ")")
