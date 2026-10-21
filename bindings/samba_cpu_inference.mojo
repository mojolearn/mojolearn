# SPDX-License-Identifier: Apache-2.0
"""Native ownership of the existing CPU-only SambaInference surface."""
from std.python import Python, PythonObject
from bindings.generation_buffers import generation_initialize_binding, generation_copy, py_ints
from bindings.samba_state_native import samba_fields, _samba_registry
from bindings.samba_config_native import samba_config_from_dict, samba_block_shapes
from bindings.samba_checkpoint_native import samba_checkpoint_read
from bindings.causal_lm_construct import _model_bindings, _model_block, _model_unpack, _model_profile, _model_shape
from bindings.causal_lm_public import _native_function
from bindings.causal_checkpoint import checkpoint_array


def samba_inference_construct(model: PythonObject, config: PythonObject, weights: PythonObject, context: PythonObject) raises -> PythonObject:
    _ = generation_initialize_binding(context["neural"]["generation_classes"])
    var bi = Python.import_module("builtins")
    if not Bool(py=bi.isinstance(config, context["config_class"])):
        raise Error("TypeError: SambaInference: config must be a SambaConfig")
    if bi.type(weights) is not bi.dict:
        raise Error("TypeError: SambaInference: weights must be a dict of native buffers")
    var nc = context["neural"]
    var bindings = _model_bindings(PythonObject("cpu"), "mamba3", nc, PythonObject("identical"))
    var registry = _samba_registry(config)
    var names = bi.set()
    for i in range(len(registry)):  # driver-loop: over admitted model registry metadata
        names.add(registry[i]["name"])
    if Bool(py=bi.set(weights) != names):
        raise Error("SambaInference: weight dict has missing or unknown registry names")
    var arrays = bi.dict()
    var format = "float32"
    for i in range(len(registry)):  # driver-loop: over copied CPU inference parameter owners
        var item = registry[i]
        var name = item["name"]
        var value = weights[name]
        var tag = "bfloat16" if bi.type(value) is nc["bf16_class"] else ("int8" if bi.type(value) is nc["int8_class"] else "float32")
        if tag != "float32":
            if format != "float32" and format != tag:
                raise Error("SambaInference: weights mix packed formats")
            format = tag
        var unpacked = _model_unpack(value, bindings["linalg"], nc)
        var shape = List[Int]()
        for axis in range(len(item["shape"])):  # driver-loop: over tensor shape metadata
            shape.append(Int(py=item["shape"][axis]))
        _model_shape(unpacked, shape, String(name))
        if Int(py=context["all_finite_f32"](unpacked._addr, unpacked.size)) != 1:
            raise Error("SambaInference: nonfinite weight " + String(name))
        arrays[name] = generation_copy(unpacked)
    var fields = samba_fields(model)
    fields["config"] = config
    fields["weight_format"] = format
    fields["_w"] = arrays
    fields["arrays"] = arrays
    fields["numeric_mode"] = "identical"
    fields["_native_cpu_inference"] = True
    var blocks = bi.list()
    var cfg = samba_fields(config)
    for i in range(len(cfg["layers"])):  # driver-loop: over CPU block execution owners
        var kind = "mamba3" if String(cfg["layers"][i]) == "mamba3" else "transformer"
        var layer = bi.dict()
        var shapes = samba_block_shapes(config, String(cfg["layers"][i]))
        for j in range(len(shapes)):  # driver-loop: over one block's parameter registry
            var name = shapes[j][0]
            layer[name] = arrays["layers." + String(i) + "." + String(name)]
        var plan = bi.object.__new__(nc["config_context"]["plan_class"])
        plan.kind = kind
        plan.d_model = cfg["d_model"]
        plan.block_options = bi.dict()
        plan.block_kwargs = bi.dict()
        var profile = PythonObject("fp32_v1")
        if kind == "transformer":
            plan.n_heads = cfg["n_heads"]
            plan.block_options["n_kv_heads"] = cfg["n_kv_heads"]
            plan.block_options["head_dim"] = cfg["head_dim"]
            plan.block_options["norm_eps"] = PythonObject(1e-6)
            profile = _model_profile(PythonObject(None), "transformer", nc)
        blocks.append(_model_block(layer, plan, profile, PythonObject("cpu"), bindings, nc))
    fields["_blocks"] = blocks
    return PythonObject(None)


def samba_inference_parameters(model: PythonObject) raises -> PythonObject:
    var arrays = samba_fields(model)["_w"]
    var bi = Python.import_module("builtins")
    var keys = bi.list(arrays)
    var result = bi.dict()
    for i in range(len(keys)):  # driver-loop: over independent public weight copies
        var name = keys[i]
        result[name] = generation_copy(arrays[name])
    return result


def samba_inference_checkpoint(cls: PythonObject, path: PythonObject, context: PythonObject) raises -> PythonObject:
    var state = samba_checkpoint_read(path, context)
    var bi = Python.import_module("builtins")
    if String(state.get("schema")) != "mojolearn.samba-stack-state.v1" or String(state.get("profile")) != "mojolearn.samba-stack.fp32.v1":
        raise Error("SambaInference: checkpoint schema/profile mismatch")
    var config = samba_config_from_dict(context["config_class"], state["config"])
    var registry = _samba_registry(config)
    if Bool(py=registry != state["registry"]):
        raise Error("SambaInference: checkpoint registry differs from config")
    var flat = state["parameters"]
    var raw = flat._mv
    var weights = bi.dict()
    for i in range(len(registry)):  # driver-loop: over verified checkpoint parameter slices
        var item = registry[i]
        var view = raw[bi.slice(item["offset"], item["offset"] + item["size"])]
        # checkpoint_array owns the copied bytes and shape; no Array constructor
        # or reshape callback is invoked by the loader.
        weights[item["name"]] = checkpoint_array(view.tobytes(), item["shape"], "f", context["neural"]["array_class"])
    var model = bi.object.__new__(cls)
    _ = samba_inference_construct(model, config, weights, context)
    return model


def samba_inference_repr(model: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var config = samba_fields(samba_fields(model)["config"])
    return PythonObject("SambaInference(layers=" + String(bi.repr(bi.list(config["layers"]))) + ", d_model=" + String(config["d_model"]) + ", vocab=" + String(config["vocab"]) + ")")
