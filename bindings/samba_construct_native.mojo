# SPDX-License-Identifier: Apache-2.0
"""Native Samba registry, explicit-weight and seeded constructor ownership."""
from std.python import Python, PythonObject
from std.math import sqrt
from std.memory import memcpy
from std.os import getenv
from bindings.generation_buffers import generation_initialize_binding, generation_array, py_ints
from bindings.generator_state_native import generator_init, generator_fields, generator_validate, generator_next
from bindings.samba_state_native import samba_fields, samba_entries, samba_view, _samba_registry, samba_build_blocks, samba_schedule_snapshot
from bindings.causal_lm_construct import _model_f32, _model_shape
from bindings.causal_lm_public import _native_function


def samba_load_weights(model: PythonObject, weights: PythonObject, context: PythonObject) raises -> PythonObject:
    _ = generation_initialize_binding(context["neural"]["generation_classes"])
    var bi = Python.import_module("builtins")
    var fields = samba_fields(model)
    var names = fields["names"]
    if Bool(py=bi.set(names) != bi.set(weights)):
        raise Error("SambaStack: weight dict mismatch; missing or unknown registry names")
    for i in range(len(names)):  # driver-loop: over ordered named parameter buffers
        var name = names[i]
        var value = _model_f32(weights[name], String(name), context["neural"])
        var dims = List[Int]()
        var shape = fields["shapes"][name]
        for axis in range(len(shape)):  # driver-loop: over tensor shape metadata
            dims.append(Int(py=shape[axis]))
        _model_shape(value, dims, String(name))
        if Int(py=context["all_finite_f32"](value._addr, value.size)) != 1:
            raise Error("SambaStack: " + String(name) + " is not finite")
        var target = fields["arrays"][name]
        memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=target._addr)),
               src=Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=Int(py=value._addr)), count=Int(py=value.nbytes))
        _ = value^
    return PythonObject(None)


def samba_parameters(model: PythonObject) raises -> PythonObject:
    var fields = samba_fields(model)
    var result = Python.dict()
    for i in range(len(fields["names"])):  # driver-loop: over the public named parameter registry
        var name = fields["names"][i]
        result[name] = fields["arrays"][name]
    return result


def _samba_initialize_parameters(model: PythonObject, context: PythonObject) raises:
    var fields = samba_fields(model)
    var generator = fields["generator"]
    var rng = generator_fields(generator)
    _ = generator_validate(rng["seed"], rng["counter"])
    var entries = samba_entries(context, rng["numeric_mode"])
    var seed = rng["seed"]
    var low = Int(py=seed & PythonObject(0xffffffff))
    var high = Int(py=(seed >> PythonObject(32)) & PythonObject(0xffffffff))
    # Keep the legacy float64 sequence (including sqrt(5), then gain), rather
    # than simplifying the bound algebraically. Both supported host baselines
    # use correctly rounded hardware sqrt, as portable_math.c does.
    var a = sqrt(Float64(5.0))
    var square = a * a
    var gain = sqrt(Float64(2.0) / (Float64(1.0) + square))
    var unused = generation_array([1])
    for i in range(len(fields["names"])):  # driver-loop: over initializer calls in the saved registry order
        var name = String(fields["names"][i])
        var output = fields["arrays"][fields["names"][i]]
        var kind = 0
        var first = Float32(0.0)
        var second = Float32(0.0)
        var random = True
        if name == "embed.weight" or name == "lm_head.weight":
            kind = 1
            second = Float32(0.02)
        elif name.endswith(".weight") and len(output.shape) == 2:
            var bound = gain * sqrt(Float64(3.0) / Float64(Int(py=output.shape[1])))
            first = Float32(-bound)
            var high_bound = Float32(bound)
            second = Float32(Float64(high_bound) - Float64(first))
        elif name.endswith(".dt_bias"):
            first = -4.0
            second = 2.0
        else:
            random = False
            var dst = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=output._addr))
            for j in range(Int(py=output.size)):
                dst[j] = 1.0
        if random:
            var stream = generator_next(generator)
            var plan = py_ints([Int(py=output.size), 0, low, high, Int(py=stream), kind])
            plan.append(Float64(first))
            plan.append(Float64(second))
            _native_function(entries, "neural_rng")(Python.list(output._addr, unused._addr), plan)
    _ = unused^



def samba_construct(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    _ = generation_initialize_binding(context["neural"]["generation_classes"])
    var bi = Python.import_module("builtins")
    var config = args[0]
    if not Bool(py=bi.isinstance(config, context["config_class"])):
        raise Error("TypeError: SambaStack: config must be a SambaConfig")
    var weights = args[1]
    var generator = args[2]
    if (weights is None) == (generator is None):
        raise Error("SambaStack: pass exactly one of weights= or generator=")
    var mode = args[10]
    _ = samba_entries(context, mode)
    var accumulation = Int(py=bi.int(args[9]))
    if accumulation < 1 or (accumulation & (accumulation - 1)) != 0:
        raise Error("SambaStack: accumulation_steps must be a power of two >= 1")
    _ = samba_schedule_snapshot(args[7], context)
    var registry = _samba_registry(config)
    var last = registry[len(registry) - 1]
    var n = Int(py=last["offset"]) + Int(py=last["size"])
    var fields = samba_fields(model)
    fields["config"] = config
    fields["numeric_mode"] = mode
    fields["n_total"] = n
    fields["flat"] = generation_array([n])
    fields["names"] = Python.list()
    fields["shapes"] = Python.dict()
    fields["offsets"] = Python.list()
    fields["arrays"] = Python.dict()
    var params = Python.list()
    for i in range(len(registry)):  # driver-loop: over flat storage views in registry order
        var item = registry[i]
        var dims = List[Int]()
        for axis in range(len(item["shape"])):  # driver-loop: over tensor shape metadata
            dims.append(Int(py=item["shape"][axis]))
        var value = samba_view(fields["flat"], Int(py=item["offset"]), dims, context)
        fields["names"].append(item["name"])
        fields["shapes"][item["name"]] = bi.tuple(item["shape"])
        fields["offsets"].append(item["offset"])
        fields["arrays"][item["name"]] = value
        params.append(value)
    fields["offsets"].append(n)
    if generator is None:
        generator = bi.object.__new__(context["generator_class"])
        _ = generator_init(generator, PythonObject(0), mode)
    fields["generator"] = generator
    if weights is None:
        _samba_initialize_parameters(model, context)
    else:
        _ = samba_load_weights(model, weights, context)
    fields["max_norm"] = PythonObject(None) if args[8] is None else bi.float(args[8])
    var optimizer = bi.object.__new__(context["optimizer_class"])
    var opt = samba_fields(optimizer)
    opt["_where"] = "AdamW"
    opt["_res"] = PythonObject(None)
    opt["_res_binding"] = PythonObject(None)
    opt["_host_fresh"] = True
    opt["_resident_wanted"] = PythonObject(None)
    var lazy = getenv("MOJOLEARN_OPT_LAZY_MOMENTS", "1") != "0"
    opt["_pristine"] = lazy
    opt["_m"] = PythonObject(None) if lazy else generation_array([n])
    opt["_v"] = PythonObject(None) if lazy else generation_array([n])
    opt["params"] = params
    opt["n_total"] = n
    opt["accumulation_steps"] = accumulation
    opt["numeric_mode"] = mode
    opt["lr"] = bi.float(args[3])
    opt["betas"] = Python.tuple(bi.float(args[4][0]), bi.float(args[4][1]))
    opt["eps"] = bi.float(args[5])
    opt["weight_decay"] = bi.float(args[6])
    opt["lr_schedule"] = args[7]
    opt["maximize"] = False
    opt["t"] = 0
    opt["packed_"] = PythonObject(None)
    opt["total_norm_"] = PythonObject(None)
    opt["clip_coef_"] = PythonObject(None)
    opt["buf_initialized"] = generation_array([len(registry)], "i")
    var offsets = generation_array([len(registry) + 1], "i")
    var ptr = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=offsets._addr))
    for i in range(len(registry) + 1):  # driver-loop: over registry offset metadata
        ptr[i] = Int32(Int(py=fields["offsets"][i]))
    opt["offsets"] = offsets
    fields["optimizer"] = optimizer
    samba_build_blocks(model, context)
    fields["last_"] = PythonObject(None)
    return PythonObject(None)
