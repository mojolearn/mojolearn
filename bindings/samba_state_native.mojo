# SPDX-License-Identifier: Apache-2.0
"""Native Samba snapshots and publication over the existing ordered flat state.

Only supported built-in schedules are admitted by this private distributed
bridge. The legacy public callback interface remains separate runtime debt.
"""
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from std.memory import memcpy
from std.os import getenv
from bindings.generation_buffers import generation_initialize_binding, generation_array, generation_copy, py_ints
from bindings.causal_lm_construct import _model_f32, _model_shape, _model_block, _model_bindings, _model_native_buffer
from bindings.causal_lm_public import _native_function
from bindings.samba_config_native import samba_config_dict, samba_config_from_dict, samba_registry, samba_block_shapes


def samba_fields(value: PythonObject) raises -> PythonObject:
    return Python.import_module("builtins").object.__getattribute__(value, "__dict__")


def samba_entries(context: PythonObject, mode: PythonObject) raises -> PythonObject:
    var entries = context["entries"]
    var selected = context["mode"] if mode is None else mode
    if Bool(py=selected != context["mode"]):
        raise Error("Samba: requested numeric mode differs from the connected native mode")
    var want = 1 if String(selected) == "identical" else 0
    if Int(py=_native_function(entries, "training_numeric_mode")()) != want or Bool(py=_native_function(entries, "training_vendor")() != context["vendor"]):
        raise Error("Samba: native training mode/vendor admission mismatch")
    return entries


def samba_experiment(entries: PythonObject) raises -> PythonObject:
    if Bool(py=Python.import_module("builtins").hasattr(entries, "training_experiment_profile")):
        return _native_function(entries, "training_experiment_profile")()
    return PythonObject("baseline")


def samba_view(flat: PythonObject, offset: Int, dims: List[Int], context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var n = 1
    for d in dims:  # driver-loop: over tensor shape metadata
        if d < 0 or (d != 0 and n > 2147483647 // d):
            raise Error("Samba: tensor shape exceeds signed 32-bit indexing")
        n *= d
    if offset < 0 or offset + n > Int(py=flat.size):
        raise Error("Samba: tensor view exceeds flat parameter storage")
    var result = bi.object.__new__(context["neural"]["array_class"])
    result._store = flat._store
    result._base = flat
    result._pin = flat._pin
    result._mv = flat._mv[bi.slice(offset, offset + n)]
    result._addr = PythonObject(Int(py=flat._addr) + 4 * offset)
    result._readonly = PythonObject(False)
    result.shape = bi.tuple(py_ints(dims))
    result.dtype = PythonObject("<f4")
    result.ndim = PythonObject(len(dims))
    result.size = PythonObject(n)
    result.nbytes = PythonObject(4 * n)
    result.itemsize = PythonObject(4)
    result.order = PythonObject("C")
    var strides = bi.list()
    var stride = 4
    for reverse in range(len(dims)):  # driver-loop: over tensor stride metadata
        strides.insert(0, stride)
        stride *= dims[len(dims) - reverse - 1]
    result.strides = bi.tuple(strides)
    return result


def samba_schedule_snapshot(schedule: PythonObject, context: PythonObject) raises -> PythonObject:
    if schedule is None:
        return PythonObject(None)
    var bi = Python.import_module("builtins")
    var kinds = bi.list(context["schedule_classes"])
    var kind = String()
    for i in range(len(kinds)):  # driver-loop: over supported built-in schedule classes
        if bi.type(schedule) is context["schedule_classes"][kinds[i]]:
            kind = String(kinds[i])
    if kind == "":
        raise Error("NotImplementedError: native Samba state requires a built-in learning-rate schedule; custom callback interfaces remain on the legacy public surface")
    var fields = samba_fields(schedule)
    var result = bi.dict()
    result["kind"] = kind
    for name in [String("peak_lr"), String("min_lr"), String("warmup_steps"), String("total_steps")]:  # driver-loop: over schedule metadata fields
        result[name] = fields[name]
    return result


def samba_schedule_rate(config: PythonObject, step: Int, context: PythonObject) raises -> PythonObject:
    var kind = String(config["kind"])
    var code = 0 if kind == "constant" else (1 if kind == "linear" else 2)
    if kind != "constant" and kind != "linear" and kind != "cosine":
        raise Error("Samba: unsupported saved learning-rate schedule")
    var total = -1 if config["total_steps"] is None else Int(py=config["total_steps"])
    var scalars = Python.list(config["peak_lr"], config["min_lr"])
    var result = context["exact_schedule"](py_ints([code, Int(py=config["warmup_steps"]), total, step]), scalars)
    if Int(py=result[0]) != 0:
        raise Error("Samba: native exact learning-rate schedule refused saved configuration, status " + String(Int(py=result[0])))
    return result[1]


def samba_schedule_from_state(config: PythonObject, context: PythonObject) raises -> PythonObject:
    if config is None:
        return PythonObject(None)
    var bi = Python.import_module("builtins")
    var kind = String(config["kind"])
    if not Bool(py=context["schedule_classes"].__contains__(kind)):
        raise Error("Samba: unsupported saved learning-rate schedule")
    var result = bi.object.__new__(context["schedule_classes"][kind])
    result.peak_lr = PythonObject(Float64(Float32(Float64(py=bi.float(config["peak_lr"])))))
    result.min_lr = PythonObject(Float64(Float32(Float64(py=bi.float(config["min_lr"])))))
    result.warmup_steps = bi.int(config["warmup_steps"])
    result.total_steps = PythonObject(None) if config["total_steps"] is None else bi.int(config["total_steps"])
    var warmup = Int(py=result.warmup_steps)
    var tail = warmup + 1 if result.total_steps is None else max(Int(py=result.total_steps), warmup + 1)
    result._fast = PythonObject(getenv("MOJOLEARN_LR_TABLE", "1") != "0" and getenv("MOJOLEARN_LR_EXACT_ONLY", "0") != "1")
    result._tail_t = PythonObject(tail)
    result._tail_v = samba_schedule_rate(config, tail, context)
    result._tb = PythonObject(-1)
    result._tbv = PythonObject(None)
    return result


def _samba_registry(config: PythonObject) raises -> PythonObject:
    var source = samba_registry(config)
    var result = Python.list()
    var offset = 0
    for i in range(len(source)):  # driver-loop: over the ordered parameter registry
        var n = 1
        for j in range(len(source[i][1])):  # driver-loop: over tensor shape metadata
            var d = Int(py=source[i][1][j])
            if d < 1 or n > 2147483647 // d:
                raise Error("Samba: invalid tensor shape")
            n *= d
        if offset > 2147483647 - n:
            raise Error("Samba: flat parameter registry exceeds signed 32-bit indexing")
        var item = Python.dict()
        item["name"] = source[i][0]
        item["shape"] = Python.import_module("builtins").list(source[i][1])
        item["offset"] = offset
        item["size"] = n
        result.append(item)
        offset += n
    return result


def samba_snapshot(model: PythonObject, copy_arrays: Bool, context: PythonObject) raises -> PythonObject:
    _ = generation_initialize_binding(context["neural"]["generation_classes"])
    var bi = Python.import_module("builtins")
    var fields = samba_fields(model)
    var entries = samba_entries(context, fields["numeric_mode"])
    var opt = samba_fields(fields["optimizer"])
    var n = Int(py=fields["n_total"])
    if opt["_m"] is None:
        opt["_m"] = generation_array([n])
    if opt["_v"] is None:
        opt["_v"] = generation_array([n])
    var m = _model_f32(opt["_m"], "exp_avg", context["neural"])
    var v = _model_f32(opt["_v"], "exp_avg_sq", context["neural"])
    _model_shape(m, [n], "exp_avg")
    _model_shape(v, [n], "exp_avg_sq")
    if opt["_res"] is not None and not Bool(py=opt["_host_fresh"]):
        _native_function(entries, "optimizer_resident_download")(opt["_res"], m._addr, v._addr, n)
        # A caller may assign writable foreign buffers to the public moment
        # properties. Keep those aliases and write native download bytes back.
        if m is not opt["_m"]:
            var m_view = bi.memoryview(opt["_m"])
            m_view[bi.slice(PythonObject(None))] = m._mv
        if v is not opt["_v"]:
            var v_view = bi.memoryview(opt["_v"])
            v_view[bi.slice(PythonObject(None))] = v._mv
        opt["_host_fresh"] = True
    var flags = _model_native_buffer(opt["buf_initialized"], "i", context["neural"])
    opt["_pristine"] = False
    var result = bi.dict()
    result["schema"] = "mojolearn.samba-stack-state.v1"
    result["profile"] = "mojolearn.samba-stack.fp32.v1"
    result["experiment_profile"] = samba_experiment(entries)
    result["numeric_mode"] = context["mode"]
    result["config"] = samba_config_dict(fields["config"])
    result["registry"] = _samba_registry(fields["config"])
    result["parameters"] = generation_copy(fields["flat"]) if copy_arrays else fields["flat"]
    result["exp_avg"] = generation_copy(m) if copy_arrays else opt["_m"]
    result["exp_avg_sq"] = generation_copy(v) if copy_arrays else opt["_v"]
    result["buf_initialized"] = generation_copy(flags) if copy_arrays else opt["buf_initialized"]
    result["t"] = opt["t"]
    var optimizer = bi.dict()
    optimizer["kind"] = "adamw"
    optimizer["lr"] = opt["lr"]
    optimizer["beta1"] = opt["betas"][0]
    optimizer["beta2"] = opt["betas"][1]
    optimizer["eps"] = opt["eps"]
    optimizer["weight_decay"] = opt["weight_decay"]
    optimizer["max_norm"] = fields["max_norm"]
    optimizer["accumulation_steps"] = opt["accumulation_steps"]
    result["optimizer"] = optimizer
    result["schedule"] = samba_schedule_snapshot(opt["lr_schedule"], context)
    var generator = samba_fields(fields["generator"])
    var rng = bi.dict()
    rng["seed"] = generator["seed"]
    rng["counter"] = generator["counter"]
    result["rng"] = rng
    return result


def samba_validate_state(state: PythonObject, context: PythonObject) raises -> PythonObject:
    _ = generation_initialize_binding(context["neural"]["generation_classes"])
    var bi = Python.import_module("builtins")
    var entries = samba_entries(context, state["numeric_mode"])
    if String(state.get("schema")) != "mojolearn.samba-stack-state.v1" or String(state.get("profile")) != "mojolearn.samba-stack.fp32.v1":
        raise Error("Samba: state schema/profile mismatch")
    if Bool(py=state.get("experiment_profile", PythonObject("baseline")) != samba_experiment(entries)):
        raise Error("Samba: experiment numerical profile mismatch")
    if String(state.get("numeric_profile", PythonObject("fp32_v1"))) != "fp32_v1":
        raise Error("Samba: state numeric profile must be fp32_v1")
    var config = samba_config_from_dict(context["config_class"], state["config"])
    var registry = _samba_registry(config)
    if Bool(py=registry != state["registry"]):
        raise Error("Samba: saved parameter registry differs from configuration")
    var n = Int(py=registry[len(registry) - 1]["offset"]) + Int(py=registry[len(registry) - 1]["size"])
    var result = bi.dict(state)
    for name in [String("parameters"), String("exp_avg"), String("exp_avg_sq")]:  # driver-loop: over the three floating training-state buffers
        var value = _model_f32(state[name], name, context["neural"])
        _model_shape(value, [n], name)
        result[name] = generation_copy(value)
    var flags = state["buf_initialized"]
    var own = bi.type(flags) is context["neural"]["array_class"]
    var view = flags._mv if own else bi.memoryview(flags)
    var shape = flags.shape if own else view.shape
    if len(shape) != 1 or Int(py=shape[0]) != len(registry) or (String(view.format) != "i" and String(view.format) != "<i" and String(view.format) != "=i"):
        raise Error("Samba: optimizer flags must be one int32 per registered tensor")
    var copied = generation_array([len(registry)], "i")
    copied._mv[bi.slice(PythonObject(None))] = view
    result["buf_initialized"] = copied
    var step = Int(py=state["t"])
    if step < 0 or step > 2147483647:
        raise Error("Samba: optimizer step exceeds signed 32-bit range")
    var accumulation = Int(py=state["optimizer"]["accumulation_steps"])
    if accumulation < 1 or (accumulation & (accumulation - 1)) != 0:
        raise Error("Samba: accumulation_steps must be a positive power of two")
    if String(state["optimizer"]["kind"]) != "adamw":
        raise Error("Samba: optimizer kind must be adamw")
    var counter = Int(py=state["rng"]["counter"])
    if counter < 0 or counter > 4294967296:
        raise Error("Samba: RNG counter must be in [0, 2^32], including exhaustion")
    if Bool(py=state["rng"]["seed"] < 0) or Bool(py=state["rng"]["seed"] >= bi.int("18446744073709551616")):
        raise Error("Samba: RNG seed must be a 64-bit unsigned integer")
    _ = samba_schedule_from_state(state["schedule"], context)
    result["config"] = samba_config_dict(config)
    result["registry"] = registry
    return result


def samba_publish(model: PythonObject, state: PythonObject, context: PythonObject) raises -> PythonObject:
    var checked = samba_validate_state(state, context)
    var fields = samba_fields(model)
    if Bool(py=samba_config_dict(fields["config"]) != checked["config"]):
        raise Error("Samba: state config differs from this stack")
    var opt = samba_fields(fields["optimizer"])
    var oc = checked["optimizer"]
    if Bool(py=opt["accumulation_steps"] != oc["accumulation_steps"]):
        raise Error("Samba: saved accumulation_steps differs from this stack")
    var schedule = samba_schedule_from_state(checked["schedule"], context)
    var flat = fields["flat"]
    var source = checked["parameters"]
    var dst = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=flat._addr))
    var src = Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=Int(py=source._addr))
    memcpy(dest=dst, src=src, count=Int(py=flat.nbytes))
    opt["_m"] = checked["exp_avg"]
    opt["_v"] = checked["exp_avg_sq"]
    opt["_host_fresh"] = True
    opt["_pristine"] = False
    if opt["_res"] is not None:
        var entries = samba_entries(context, fields["numeric_mode"])
        _native_function(entries, "optimizer_resident_upload")(opt["_res"], opt["_m"]._addr, opt["_v"]._addr, fields["n_total"])
    opt["buf_initialized"] = checked["buf_initialized"]
    opt["t"] = checked["t"]
    opt["lr"] = oc["lr"]
    opt["betas"] = Python.tuple(oc["beta1"], oc["beta2"])
    opt["eps"] = oc["eps"]
    opt["weight_decay"] = oc["weight_decay"]
    opt["lr_schedule"] = schedule
    fields["max_norm"] = oc["max_norm"]
    var generator = samba_fields(fields["generator"])
    generator["seed"] = checked["rng"]["seed"]
    generator["counter"] = checked["rng"]["counter"]
    return model


def samba_from_state(state: PythonObject, context: PythonObject) raises -> PythonObject:
    var checked = samba_validate_state(state, context)
    var bi = Python.import_module("builtins")
    var model = bi.object.__new__(context["model_class"])
    var config = samba_config_from_dict(context["config_class"], checked["config"])
    model.config = config
    model.numeric_mode = checked["numeric_mode"]
    model.flat = checked["parameters"]
    model.n_total = model.flat.size
    model.names = bi.list()
    model.shapes = bi.dict()
    model.offsets = bi.list()
    model.arrays = bi.dict()
    var registry = checked["registry"]
    var params = bi.list()
    for i in range(len(registry)):  # driver-loop: over registry views retaining the flat parameter owner
        var item = registry[i]
        var dims = List[Int]()
        for j in range(len(item["shape"])):  # driver-loop: over tensor shape metadata
            dims.append(Int(py=item["shape"][j]))
        model.names.append(item["name"])
        model.shapes[item["name"]] = bi.tuple(item["shape"])
        model.offsets.append(item["offset"])
        var value = samba_view(model.flat, Int(py=item["offset"]), dims, context)
        model.arrays[item["name"]] = value
        params.append(value)
    model.offsets.append(model.n_total)
    var generator = bi.object.__new__(context["generator_class"])
    var gen_fields = samba_fields(generator)
    gen_fields["seed"] = checked["rng"]["seed"]
    gen_fields["counter"] = checked["rng"]["counter"]
    gen_fields["numeric_mode"] = checked["numeric_mode"]
    model.generator = generator
    model.max_norm = checked["optimizer"]["max_norm"]
    var optimizer = bi.object.__new__(context["optimizer_class"])
    var opt = samba_fields(optimizer)
    opt["_where"] = "AdamW"
    opt["_res"] = PythonObject(None)
    opt["_res_binding"] = PythonObject(None)
    opt["_host_fresh"] = True
    opt["_resident_wanted"] = PythonObject(None)
    opt["_pristine"] = False
    opt["params"] = params
    opt["n_total"] = model.n_total
    opt["accumulation_steps"] = checked["optimizer"]["accumulation_steps"]
    opt["numeric_mode"] = checked["numeric_mode"]
    opt["maximize"] = False
    opt["packed_"] = PythonObject(None)
    opt["total_norm_"] = PythonObject(None)
    opt["clip_coef_"] = PythonObject(None)
    var offsets = generation_array([len(registry) + 1], "i")
    var offset_ptr = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=offsets._addr))
    for i in range(len(registry) + 1):  # driver-loop: over registry offset metadata
        offset_ptr[i] = Int32(Int(py=model.offsets[i]))
    opt["offsets"] = offsets
    model.optimizer = optimizer
    model._blocks = bi.list()
    var route = PythonObject("cpu" if context["vendor"] is None or String(context["vendor"]) == "cpu" else "gpu")
    var nc = context["neural"]
    for i in range(len(config.layers)):  # driver-loop: over configured model layers
        var layer_kind = String(config.layers[i])
        var kind = "mamba3" if layer_kind == "mamba3" else "transformer"
        var weights = bi.dict()
        var names = samba_block_shapes(config, layer_kind)
        for j in range(len(names)):  # driver-loop: over the layer's named parameter views
            weights[names[j][0]] = model.arrays["layers." + String(i) + "." + String(names[j][0])]
        var plan = bi.object.__new__(nc["config_context"]["plan_class"])
        plan.kind = PythonObject(kind)
        plan.d_model = config.d_model
        plan.block_options = bi.dict()
        plan.block_kwargs = bi.dict()
        if kind == "transformer":
            plan.n_heads = config.n_heads
            plan.block_options["n_kv_heads"] = config.n_kv_heads
            plan.block_options["head_dim"] = config.head_dim
            # Samba's attention blocks use TransformerBlock's 1e-6 default.
            # The checkpoint model interface defaults to 1e-5 instead, and
            # SambaConfig.norm_eps controls only the final stack normalization.
            plan.block_options["norm_eps"] = PythonObject(1e-6)
        var bindings = _model_bindings(route, kind, nc, model.numeric_mode)
        var block = _model_block(weights, plan, PythonObject("fp32_v1"), route, bindings, nc)
        var block_fields = samba_fields(block)
        block_fields["numeric_mode"] = model.numeric_mode
        if kind == "mamba3":
            block_fields["_discard_reports"] = True
        model._blocks.append(block)
    model.last_ = PythonObject(None)
    return samba_publish(model, checked, context)


def samba_apply_gradients_from_state(state: PythonObject, gradients: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var entries = samba_entries(context, state["numeric_mode"])
    if Int(py=bi.int(getenv("MOJOLEARN_OPTIMIZER_DEVICE_COUNT", "1"))) > 1:
        for name in [String("optimizer_parallel_available"), String("accumulate_parallel_available"), String("clip_parallel_available")]:  # driver-loop: over cooperative native capability metadata
            if not Bool(py=bi.hasattr(entries, name)) or Int(py=_native_function(entries, name)()) != 1:
                raise Error("RuntimeError: native Samba cooperative update is unavailable: " + name)
    var result = samba_validate_state(state, context)
    var registry = result["registry"]
    if len(gradients) != len(registry):
        raise Error("Samba: gradients must match the ordered parameter registry")
    var step = Int(py=result["t"])
    if step >= 2147483647:
        raise Error("Samba: optimizer step exceeds signed 32-bit range")
    var config = bi.dict(result["optimizer"])
    if result["schedule"] is not None:
        config["lr"] = samba_schedule_rate(result["schedule"], step + 1, context)
    var max_norm = 0.0 if config["max_norm"] is None else Float64(py=config["max_norm"])
    if config["max_norm"] is not None and not (max_norm > 0.0):
        raise Error("Samba: max_norm must be positive or None")
    var n = Int(py=result["parameters"].size)
    var flat_gradient = generation_array([n])
    var offsets = generation_array([len(registry) + 1], "i")
    var gp = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=flat_gradient._addr))
    var op = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=offsets._addr))
    for i in range(len(registry)):  # driver-loop: over registry-ordered gradient buffers
        var item = registry[i]
        var dims = List[Int]()
        for axis in range(len(item["shape"])):  # driver-loop: over tensor shape metadata
            dims.append(Int(py=item["shape"][axis]))
        var value = _model_f32(gradients[i], "gradient", context["neural"])
        _model_shape(value, dims, "gradient")
        var offset = Int(py=item["offset"])
        op[i] = Int32(offset)
        var src = Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=Int(py=value._addr))
        memcpy(dest=gp + offset, src=src, count=Int(py=item["size"]))
        _ = value^
    op[len(registry)] = Int32(n)
    var plan = py_ints([len(registry), 2, step + 1, 0])
    for key in [String("lr"), String("beta1"), String("beta2"), String("eps"), String("weight_decay")]:  # driver-loop: over optimizer scalar launch metadata
        plan.append(config[key])
    plan.append(0.0)
    plan.append(0.0)
    plan.append(max_norm)
    plan.append(0)
    var info = generation_array([3])
    var written = _native_function(entries, "optimizer_step")(
        result["parameters"]._addr, flat_gradient._addr, result["exp_avg"]._addr,
        result["exp_avg_sq"]._addr, offsets._addr, result["buf_initialized"]._addr,
        info._addr, plan)
    if Int(py=written) != n:
        raise Error("RuntimeError: Samba optimizer returned an incomplete update")
    result["optimizer"] = config
    result["t"] = PythonObject(step + 1)
    result["rng"] = bi.dict(result["rng"])
    _ = flat_gradient^
    _ = offsets^
    _ = info^
    return result
