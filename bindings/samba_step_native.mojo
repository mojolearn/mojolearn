# SPDX-License-Identifier: Apache-2.0
"""Native standard Samba step with the existing resident and accumulation ABIs.

This private entry does not invoke arbitrary caller schedule callbacks. The
legacy public custom-callback interface remains distinct runtime debt.
"""
from std.python import Python, PythonObject
from std.memory import memcpy
from std.os import getenv
from bindings.generation_buffers import generation_initialize_binding, generation_array, py_ints
from bindings.generator_state_native import generator_next
from bindings.samba_state_native import samba_fields, samba_entries, samba_schedule_snapshot, samba_schedule_rate
from bindings.samba_training_native import samba_ids, samba_gradients, _samba_count
from bindings.causal_lm_construct import _model_f32, _model_shape
from bindings.causal_lm_public import _native_function


def _samba_moments(opt: PythonObject, context: PythonObject) raises -> PythonObject:
    var n = Int(py=opt["n_total"])
    var result = Python.list()
    for name in [String("_m"), String("_v")]:  # driver-loop: over the optimizer's two moment owners
        if opt[name] is None:
            opt[name] = generation_array([n])
        var value = _model_f32(opt[name], name, context["neural"])
        _model_shape(value, [n], name)
        result.append(value)
    return result


def _samba_open_optimizer(opt: PythonObject, entries: PythonObject, context: PythonObject) raises:
    if opt["_res"] is not None or opt["_resident_wanted"] is PythonObject(False):
        return
    var bi = Python.import_module("builtins")
    var enabled = getenv("MOJOLEARN_OPTIMIZER_RESIDENT", "1") != "0" and getenv("MOJOLEARN_OPTIMIZER_DEVICE_COUNT", "1") == "1"
    for name in [String("optimizer_resident_open"), String("optimizer_resident_close"), String("optimizer_resident_download"), String("optimizer_resident_upload"), String("optimizer_resident_step")]:  # driver-loop: over the existing resident optimizer ABI
        enabled = enabled and Bool(py=bi.hasattr(entries, name))
    if not enabled:
        opt["_resident_wanted"] = False
        return
    opt["_res_binding"] = entries
    opt["_res"] = _native_function(entries, "optimizer_resident_open")(opt["n_total"])
    if not Bool(py=opt["_pristine"]):
        var moments = _samba_moments(opt, context)
        _native_function(entries, "optimizer_resident_upload")(opt["_res"], moments[0]._addr, moments[1]._addr, opt["n_total"])
        _ = moments^
    opt["_host_fresh"] = True


def _samba_step_plan(fields: PythonObject, opt: PythonObject, context: PythonObject) raises -> PythonObject:
    var step = Int(py=opt["t"])
    if step < 0 or step >= 2147483647:
        raise Error("Samba: optimizer step exceeds signed 32-bit indexing")
    var schedule = samba_schedule_snapshot(opt["lr_schedule"], context)
    if schedule is not None:
        opt["lr"] = samba_schedule_rate(schedule, step + 1, context)
    var max_norm = 0.0 if fields["max_norm"] is None else Float64(py=fields["max_norm"])
    if fields["max_norm"] is not None and not (max_norm > 0.0):
        raise Error("Samba: max_norm must be positive or None")
    opt["t"] = step + 1
    var plan = py_ints([len(opt["params"]), 2, step + 1, 0])
    plan.append(opt["lr"])
    plan.append(opt["betas"][0])
    plan.append(opt["betas"][1])
    plan.append(opt["eps"])
    plan.append(opt["weight_decay"])
    plan.append(0.0)
    plan.append(0.0)
    plan.append(max_norm)
    plan.append(1 if Bool(py=opt.get("maximize", PythonObject(False))) else 0)
    return plan


def _samba_step_finish(fields: PythonObject, opt: PythonObject, info: PythonObject, loss: PythonObject) raises -> PythonObject:
    var ptr = Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=Int(py=info._addr))
    opt["total_norm_"] = PythonObject(Float64(ptr[1])) if ptr[0] != 0.0 else PythonObject(None)
    opt["clip_coef_"] = PythonObject(Float64(ptr[2])) if ptr[0] != 0.0 else PythonObject(None)
    _ = info^
    opt["lr_"] = PythonObject(Float64(Float32(Float64(py=opt["lr"]))))
    var result = Python.dict()
    result["loss"] = loss
    result["lr"] = opt["lr_"]
    result["step"] = opt["t"]
    result["total_norm"] = opt["total_norm_"]
    fields["last_"] = result
    return Python.import_module("builtins").dict(result)


def _samba_resident_step(fields: PythonObject, opt: PythonObject, entries: PythonObject, ids: PythonObject, targets: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var config = samba_fields(fields["config"])
    if Float64(py=config["dropout"]) != 0.0:
        return PythonObject(None)
    var enabled = True
    for name in [String("samba_resident_enabled"), String("samba_resident_open"), String("samba_resident_forward"), String("samba_resident_step")]:  # driver-loop: over the existing resident Samba ABI
        enabled = enabled and Bool(py=bi.hasattr(entries, name))
    if not enabled or not Bool(py=_native_function(entries, "samba_resident_enabled")()):
        return PythonObject(None)
    if Bool(py=bi.hasattr(entries, "training_chunked_lm_head_enabled")) and Bool(py=_native_function(entries, "training_chunked_lm_head_enabled")()):
        return PythonObject(None)
    _samba_open_optimizer(opt, entries, context)
    if opt["_res"] is None:
        return PythonObject(None)
    var session = fields.get("_resident_handle")
    if session is None:
        var kinds = Python.list()
        for i in range(len(config["layers"])):  # driver-loop: over model layer kind metadata
            kinds.append(0 if String(config["layers"][i]) == "mamba3" else 1)
        var params = Python.list(config["vocab"], config["d_model"])
        for name in [String("n_heads"), String("n_kv_heads"), String("head_dim"), String("intermediate")]:  # driver-loop: over resident model shape metadata
            params.append(PythonObject(0) if config[name] is None else config[name])
        params.append(1 if Bool(py=config["tie_embeddings"]) else 0)
        params.append(config["norm_eps"])
        params.append(fields["n_total"])
        session = _native_function(entries, "samba_resident_open")(kinds, params, fields["flat"]._addr)
        fields["_resident_handle"] = session
    var plan = _samba_step_plan(fields, opt, context)
    var args = Python.list(ids.shape[0], ids.shape[1])
    args.extend(plan)
    var info = generation_array([3])
    var loss = generation_array([1])
    _native_function(entries, "samba_resident_step")(session,
        Python.list(fields["flat"]._addr, ids._addr, targets._addr, opt["offsets"]._addr,
                    opt["buf_initialized"]._addr, info._addr, loss._addr), args, opt["_res"])
    opt["_host_fresh"] = False
    var scalar = PythonObject(Float64(Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=Int(py=loss._addr))[0]))
    _ = loss^
    return _samba_step_finish(fields, opt, info, scalar)


def samba_train_step(model: PythonObject, inputs: PythonObject, targets: PythonObject, context: PythonObject) raises -> PythonObject:
    _ = generation_initialize_binding(context["neural"]["generation_classes"])
    var bi = Python.import_module("builtins")
    var fields = samba_fields(model)
    var config = samba_fields(fields["config"])
    var opt = samba_fields(fields["optimizer"])
    var entries = samba_entries(context, fields["numeric_mode"])
    # Admit the built-in schedule before consuming a stream or changing state.
    _ = samba_schedule_snapshot(opt["lr_schedule"], context)
    if Bool(py=opt.get("_p_dev", PythonObject(False))):
        raise Error("NotImplementedError: private native Samba step does not yet own externally transferred optimizer parameter buffers")
    if len(opt["params"]) != len(fields["names"]):
        raise Error("Samba: optimizer parameter registry has changed")
    for i in range(len(fields["names"])):  # driver-loop: over live optimizer parameter metadata
        var name = fields["names"][i]
        var value = opt["params"][i]
        if bi.type(value) is not context["neural"]["array_class"]:
            raise Error("NotImplementedError: private native Samba step requires its original flat parameter views")
        if Bool(py=value.shape != fields["shapes"][name]) or String(value.dtype) != "<f4" or String(value.order) != "C":
            raise Error("Samba: optimizer parameter shape or layout has changed")
        if Int(py=value._addr) != Int(py=fields["flat"]._addr) + 4 * Int(py=fields["offsets"][i]):
            raise Error("NotImplementedError: private native Samba step requires its original flat parameter views")
    var ids = samba_ids(inputs, Int(py=config["vocab"]), False, context)
    var y = samba_ids(targets, Int(py=config["vocab"]), True, context)
    if Bool(py=ids.shape != y.shape):
        raise Error("Samba: targets must match inputs' shape")
    var b = Int(py=ids.shape[0])
    var length = Int(py=ids.shape[1])
    var a = Int(py=opt["accumulation_steps"])
    if a < 1 or (a & (a - 1)) != 0 or b % a != 0:
        raise Error("Samba: batch rows must be divisible by power-of-two accumulation_steps")
    if a > 1 and not Bool(py=_native_function(entries, "accumulation_is_aligned")(py_ints([b * length, a]))):
        raise Error("Samba: MISALIGNED microbatch split; optimizer contract clause 9.2")
    if a == 1:
        var resident = _samba_resident_step(fields, opt, entries, ids, y, context)
        if resident is not None:
            return resident
    var count = _samba_count(y, context)
    var stream = generator_next(fields["generator"]) if Float64(py=config["dropout"]) > 0.0 else PythonObject(None)
    var rows = b // a
    var n = Int(py=fields["n_total"])
    var stacked = generation_array([a, n])
    var losses = generation_array([a])
    var loss_ptr = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=losses._addr))
    var gradient_ptr = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=stacked._addr))
    for k in range(a):  # driver-loop: over aligned microbatch gradient operations
        var xpart = generation_array([rows, length], "i")
        var ypart = generation_array([rows, length], "i")
        for which in range(2):  # driver-loop: over input and target buffer slices
            var source = ids if which == 0 else y
            var output = xpart if which == 0 else ypart
            memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=output._addr)),
                   src=Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=Int(py=source._addr)) + k * rows * length * 4,
                   count=rows * length * 4)
            _ = source^
            _ = output^
        var result = samba_gradients(model, xpart, ypart, PythonObject(count), stream, k * rows * length, context)
        loss_ptr[k] = Float32(Float64(py=result[0]))
        var gradients = result[1]
        if len(gradients) != len(fields["names"]):
            raise Error("Samba: gradient registry is incomplete")
        for j in range(len(gradients)):  # driver-loop: over the ordered parameter gradients
            var value = gradients[j]
            memcpy(dest=gradient_ptr + (k * n + Int(py=fields["offsets"][j])) * 4,
                   src=Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=Int(py=value._addr)), count=Int(py=value.nbytes))
        _ = gradients^
    var gradient = stacked
    var loss = PythonObject(Float64(loss_ptr[0]))
    if a > 1:
        gradient = generation_array([n])
        _native_function(entries, "accumulate")(Python.list(gradient._addr, stacked._addr), py_ints([n, a, b * length]))
        var total = generation_array([1])
        _native_function(entries, "accumulate")(Python.list(total._addr, losses._addr), py_ints([1, a, -1]))
        loss = PythonObject(Float64(Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=Int(py=total._addr))[0]))
        _ = total^
    _ = losses^
    _ = stacked^
    var plan = _samba_step_plan(fields, opt, context)
    opt["packed_"] = True
    var info = generation_array([3])
    _samba_open_optimizer(opt, entries, context)
    if opt["_res"] is not None:
        if Int(py=plan[12]) != 0 and Bool(py=bi.hasattr(entries, "optimizer_maximize_dev")):
            _native_function(entries, "optimizer_resident_step_io")(fields["flat"]._addr, gradient._addr, opt["offsets"]._addr,
                opt["buf_initialized"]._addr, info._addr, plan, Python.list(opt["_res"], PythonObject(15)))
        else:
            _native_function(entries, "optimizer_resident_step")(fields["flat"]._addr, gradient._addr, opt["offsets"]._addr,
                opt["buf_initialized"]._addr, info._addr, plan, opt["_res"])
        opt["_host_fresh"] = False
    else:
        var moments = _samba_moments(opt, context)
        _native_function(entries, "optimizer_step")(fields["flat"]._addr, gradient._addr, moments[0]._addr, moments[1]._addr,
            opt["offsets"]._addr, opt["buf_initialized"]._addr, info._addr, plan)
        for i in range(2):  # driver-loop: over caller-owned moment aliases
            var original = opt["_m" if i == 0 else "_v"]
            if moments[i] is not original:
                var view = bi.memoryview(original)
                view[bi.slice(PythonObject(None))] = moments[i]._mv
        _ = moments^
    _ = gradient^
    return _samba_step_finish(fields, opt, info, loss)
