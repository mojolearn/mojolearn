# SPDX-License-Identifier: Apache-2.0
"""Native standard Samba step with the existing resident and accumulation ABIs.

This public execution owner accepts native built-in schedule configurations;
arbitrary Python callbacks are refused before state mutation.
"""
from providers.python.samba_execution_pyapi import samba_provider_accumulated_gradients
from providers.python.samba_resident_pyapi import samba_resident_provider, samba_resident_session, samba_resident_native_step, samba_optimizer_execute
from providers.python.neural_descriptor_pyapi import NeuralDescriptorProvider, neural_descriptor_execute
from training.checks.optimizer_contract import microbatch_split_is_identical
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from std.memory import memcpy
from std.os import getenv
from bindings.generation_buffers import generation_initialize_binding, generation_array, py_ints
from bindings.generator_state_native import generator_next
from bindings.samba_state_native import samba_fields, samba_entries, samba_schedule_snapshot, samba_schedule_rate, samba_moment_view
from bindings.samba_training_native import samba_ids, samba_gradients, _samba_count
from bindings.causal_lm_construct import _model_f32, _model_shape
from bindings.causal_lm_public import _native_function
from bindings.py_balanced import py_call, py_get, py_setitem, py_type


def _samba_moments(opt: PythonObject, context: PythonObject) raises -> PythonObject:
    var n = Int(py=opt["n_total"])
    var result = Python.list()
    for name in [String("_m"), String("_v")]:  # driver-loop: over the optimizer's two moment owners
        if opt[name] is None:
            py_setitem(opt, name, generation_array([n]))
        var value = _model_f32(opt[name], name, context["neural"])
        _model_shape(value, [n], name)
        _ = py_call(result.append, value)
    return result


def _samba_open_optimizer(opt: PythonObject, entries: PythonObject, context: PythonObject, provider: PythonObject) raises:
    if opt["_res"] is not None or opt["_resident_wanted"] is PythonObject(False):
        return
    var bi = native_loaded_module("builtins")
    var enabled = getenv("MOJOLEARN_OPTIMIZER_RESIDENT", "1") != "0" and getenv("MOJOLEARN_OPTIMIZER_DEVICE_COUNT", "1") == "1"
    for name in [String("optimizer_resident_open"), String("optimizer_resident_close"), String("optimizer_resident_download"), String("optimizer_resident_upload"), String("optimizer_resident_step")]:  # driver-loop: over the existing resident optimizer ABI
        enabled = enabled and Bool(py=py_call(bi.hasattr, entries, name))
    if not enabled:
        py_setitem(opt, "_resident_wanted", False)
        return
    var open_slot = _native_function(entries, "optimizer_resident_open")
    # `entries` is a connection namespace, not the native registry owner.
    # The verified built-in entry retains the module required by optimizer
    # state downloads/uploads and eventual resident-slot cleanup.
    var owner = open_slot.__self__
    if py_type(owner) is not context["admission"]["module_type"]:
        raise Error("TypeError: Samba resident optimizer requires its owning native extension")
    py_setitem(opt, "_res_binding", owner)
    if provider is None:
        py_setitem(opt, "_res", py_call(open_slot, opt["n_total"]))
    else:
        if provider.downcast_value_ptr[NeuralDescriptorProvider]()[].module is not owner:
            raise Error("Samba optimizer provider belongs to another binding")
        py_setitem(opt, "_res", PythonObject(neural_descriptor_execute(provider, 20, [], [Int(py=opt["n_total"])], [])))
        py_setitem(opt, "_resident_native_provider", provider)
    if not Bool(py=opt["_pristine"]):
        var moments = _samba_moments(opt, context)
        if provider is None:
            _ = py_call(_native_function(entries, "optimizer_resident_upload"), opt["_res"], moments[0]._addr, moments[1]._addr, opt["n_total"])
        else:
            _ = neural_descriptor_execute(provider, 23, [Int(py=moments[0]._addr), Int(py=moments[1]._addr)],
                [Int(py=opt["_res"]), Int(py=opt["n_total"])], [])
        _ = moments^
    py_setitem(opt, "_host_fresh", True)


def _samba_step_plan(fields: PythonObject, opt: PythonObject, context: PythonObject) raises -> PythonObject:
    var step = Int(py=opt["t"])
    if step < 0 or step >= 2147483647:
        raise Error("Samba: optimizer step exceeds signed 32-bit indexing")
    var schedule = samba_schedule_snapshot(opt["lr_schedule"], context)
    if schedule is not None:
        py_setitem(opt, "lr", samba_schedule_rate(schedule, step + 1, context))
    var max_norm = 0.0 if fields["max_norm"] is None else Float64(py=fields["max_norm"])
    if fields["max_norm"] is not None and not (max_norm > 0.0):
        raise Error("Samba: max_norm must be positive or None")
    py_setitem(opt, "t", step + 1)
    var plan = py_ints([len(opt["params"]), 2, step + 1, 0])
    _ = py_call(plan.append, opt["lr"])
    _ = py_call(plan.append, opt["betas"][0])
    _ = py_call(plan.append, opt["betas"][1])
    _ = py_call(plan.append, opt["eps"])
    _ = py_call(plan.append, opt["weight_decay"])
    _ = py_call(plan.append, 0.0)
    _ = py_call(plan.append, 0.0)
    _ = py_call(plan.append, max_norm)
    _ = py_call(plan.append, 1 if Bool(py=py_get(opt, "maximize", PythonObject(False))) else 0)
    return plan


def _samba_step_finish(fields: PythonObject, opt: PythonObject, info: PythonObject, loss: PythonObject) raises -> PythonObject:
    var ptr = Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=Int(py=info._addr))
    py_setitem(opt, "total_norm_", PythonObject(Float64(ptr[1])) if ptr[0] != 0.0 else PythonObject(None))
    py_setitem(opt, "clip_coef_", PythonObject(Float64(ptr[2])) if ptr[0] != 0.0 else PythonObject(None))
    # `info` is borrowed from the caller for this entire function; it cannot
    # be consumed here, and the borrow retains storage through the reads.
    py_setitem(opt, "lr_", PythonObject(Float64(Float32(Float64(py=opt["lr"])))))
    var result = Python.dict()
    py_setitem(result, "loss", loss)
    py_setitem(result, "lr", opt["lr_"])
    py_setitem(result, "step", opt["t"])
    py_setitem(result, "total_norm", opt["total_norm_"])
    py_setitem(fields, "last_", result)
    return py_call(native_loaded_module("builtins").dict, result)


def _samba_resident_step(fields: PythonObject, opt: PythonObject, entries: PythonObject, ids: PythonObject, targets: PythonObject, context: PythonObject,
                         accumulation: Int = 1, total_count: Int = 0) raises -> PythonObject:
    var bi = native_loaded_module("builtins")
    var config = samba_fields(fields["config"])
    if Float64(py=config["dropout"]) != 0.0:
        return PythonObject(None)
    var provider = samba_resident_provider(fields, entries)
    var enabled = True
    for name in [String("samba_resident_enabled"), String("samba_resident_open"), String("samba_resident_forward"), String("samba_resident_step")]:  # driver-loop: over the existing resident Samba ABI
        enabled = enabled and Bool(py=py_call(bi.hasattr, entries, name))
    if not enabled:
        return PythonObject(None)
    if (provider is None and not Bool(py=_native_function(entries, "samba_resident_enabled")())) or (provider is not None and neural_descriptor_execute(provider, 24, [], [], [], False) == 0):
        return PythonObject(None)
    if accumulation > 1 and not Bool(py=py_call(bi.hasattr, entries, "samba_resident_accumulated_step")):
        return PythonObject(None)
    if provider is None and Bool(py=py_call(bi.hasattr, entries, "training_chunked_lm_head_enabled")) and Bool(py=_native_function(entries, "training_chunked_lm_head_enabled")()):
        return PythonObject(None)
    _samba_open_optimizer(opt, entries, context, provider)
    if opt["_res"] is None:
        return PythonObject(None)
    var session = py_get(fields, "_resident_handle")
    if provider is not None:
        session = samba_resident_session(fields, config, provider)
    elif session is None:
        var kinds = Python.list()
        for i in range(len(config["layers"])):  # driver-loop: over model layer kind metadata
            _ = py_call(kinds.append, 0 if String(config["layers"][i]) == "mamba3" else 1)
        var params = Python.list(config["vocab"], config["d_model"])
        for name in [String("n_heads"), String("n_kv_heads"), String("head_dim"), String("intermediate")]:  # driver-loop: over resident model shape metadata
            _ = py_call(params.append, PythonObject(0) if config[name] is None else config[name])
        _ = py_call(params.append, 1 if Bool(py=config["tie_embeddings"]) else 0)
        _ = py_call(params.append, config["norm_eps"])
        _ = py_call(params.append, fields["n_total"])
        session = py_call(_native_function(entries, "samba_resident_open"), kinds, params, fields["flat"]._addr)
        py_setitem(fields, "_resident_handle", session)
    var plan = _samba_step_plan(fields, opt, context)
    var args = Python.list(ids.shape[0], ids.shape[1])
    if accumulation > 1:
        # the microbatch walk as one resident program: the a gradients are
        # combined on the device and the optimizer steps there
        _ = py_call(args.append, accumulation)
        _ = py_call(args.append, total_count)
    _ = py_call(args.extend, plan)
    var info = generation_array([3])
    var loss = generation_array([1])
    if provider is not None:
        samba_resident_native_step(fields, config, provider, opt, ids, targets, info, loss, plan, accumulation, total_count)
    else:
        _ = py_call(_native_function(entries, "samba_resident_accumulated_step" if accumulation > 1 else "samba_resident_step"), session,
            Python.list(fields["flat"]._addr, ids._addr, targets._addr, opt["offsets"]._addr,
                        opt["buf_initialized"]._addr, info._addr, loss._addr), args, opt["_res"])
    _ = loss  # keep the Python object alive through the native call above
    _ = info  # keep the Python object alive through the native call above
    py_setitem(opt, "_host_fresh", False)
    var scalar = PythonObject(Float64(Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=Int(py=loss._addr))[0]))
    _ = loss  # keep the Python object alive through the native call above
    _ = loss^
    return _samba_step_finish(fields, opt, info, scalar)



def _samba_parameter_pack(fields: PythonObject, opt: PythonObject, context: PythonObject) raises -> PythonObject:
    # Parameters may have been rebound after construction. Preserve the
    # optimizer's writable C-buffer interface instead of assuming flat views.
    var bi = native_loaded_module("builtins")
    var params = opt["params"]
    if len(params) != len(fields["names"]):
        raise Error("Samba: optimizer parameter registry has changed")
    var views = bi.list()
    var original_flat = True
    for i in range(len(params)):  # driver-loop: over caller-owned optimizer parameter buffers
        var value = params[i]
        var view = PythonObject(None)
        var shape = PythonObject(None)
        if py_type(value) is context["neural"]["array_class"]:
            if String(value.dtype) != "<f4":
                raise Error("TypeError: Samba optimizer parameters must be float32")
            if String(value.order) != "C" or Bool(py=value._readonly):
                raise Error("Samba optimizer parameters must be C-contiguous and writable")
            view = value._mv
            shape = value.shape
        else:
            view = py_call(bi.memoryview, value)
            var fmt = String(view.format)
            if fmt != "f" and fmt != "<f" and fmt != "=f" and fmt != "@f":
                raise Error("TypeError: Samba optimizer parameters must be float32")
            if not Bool(py=view.c_contiguous) or Bool(py=view.readonly):
                raise Error("Samba optimizer parameters must be C-contiguous and writable")
            shape = view.shape
        if Bool(py=shape != fields["shapes"][fields["names"][i]]):
            raise Error("Samba optimizer parameter shape differs from its gradient")
        var raw = py_call(view.cast, "B")
        var begin = 4 * Int(py=fields["offsets"][i])
        var end = 4 * Int(py=fields["offsets"][i + 1])
        if len(raw) != end - begin:
            raise Error("Samba optimizer parameter size differs from its registry")
        original_flat = original_flat and py_type(value) is context["neural"]["array_class"]
        if original_flat:
            original_flat = Int(py=value._addr) == Int(py=fields["flat"]._addr) + begin
            _ = value  # keep the Python object alive through the native call above
        _ = py_call(views.append, raw)
    if original_flat:
        return Python.tuple(fields["flat"], bi.list())
    var flat = generation_array([Int(py=fields["n_total"])])
    var destination = py_call(flat._mv.cast, "B")
    for i in range(len(views)):  # driver-loop: over replacement parameter buffers
        var begin = 4 * Int(py=fields["offsets"][i])
        var end = 4 * Int(py=fields["offsets"][i + 1])
        py_setitem(destination, py_call(bi.slice, begin, end), views[i])
    return Python.tuple(flat, views)


def _samba_parameter_publish(flat: PythonObject, views: PythonObject, fields: PythonObject) raises:
    var bi = native_loaded_module("builtins")
    var raw = py_call(flat._mv.cast, "B")
    for i in range(len(views)):  # driver-loop: over caller-owned parameter write-back buffers
        var begin = 4 * Int(py=fields["offsets"][i])
        var end = 4 * Int(py=fields["offsets"][i + 1])
        var target = views[i]
        py_setitem(target, py_call(bi.slice, PythonObject(None)), raw[py_call(bi.slice, begin, end)])

def samba_train_step(model: PythonObject, inputs: PythonObject, targets: PythonObject, context: PythonObject) raises -> PythonObject:
    _ = generation_initialize_binding(context["neural"]["generation_classes"])
    var bi = native_loaded_module("builtins")
    var fields = samba_fields(model)
    var config = samba_fields(fields["config"])
    var opt = samba_fields(fields["optimizer"])
    var entries = samba_entries(context, fields["numeric_mode"])
    # Admit the built-in schedule before consuming a stream or changing state.
    _ = samba_schedule_snapshot(opt["lr_schedule"], context)
    var ids = samba_ids(inputs, Int(py=config["vocab"]), False, context)
    var y = samba_ids(targets, Int(py=config["vocab"]), True, context)
    if Bool(py=ids.shape != y.shape):
        raise Error("Samba: targets must match inputs' shape")
    var b = Int(py=ids.shape[0])
    var length = Int(py=ids.shape[1])
    var a = Int(py=opt["accumulation_steps"])
    if a < 1 or (a & (a - 1)) != 0 or b % a != 0:
        raise Error("Samba: batch rows must be divisible by power-of-two accumulation_steps")
    if a > 1 and not microbatch_split_is_identical(b * length, a):
        raise Error("Samba: MISALIGNED microbatch split; optimizer contract clause 9.2")
    if a == 1:
        var resident = _samba_resident_step(fields, opt, entries, ids, y, context)
        if resident is not None:
            return resident
    var count = _samba_count(y, context)
    if a > 1:
        var resident = _samba_resident_step(fields, opt, entries, ids, y, context, a, count)
        if resident is not None:
            return resident
    var provider = samba_resident_provider(fields, entries)
    var stream = generator_next(fields["generator"]) if Float64(py=config["dropout"]) > 0.0 else PythonObject(None)
    var completed = samba_provider_accumulated_gradients(fields, config, entries, provider, ids, y, stream, a, count, context["neural"])
    var gradient = PythonObject(None)
    var loss = PythonObject(None)
    if completed is not None:
        loss = completed[0]
        gradient = completed[1]
    else:
        var rows = b // a
        var n = Int(py=fields["n_total"])
        var stacked = generation_array([a, n])
        var losses = generation_array([a])
        var loss_ptr = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=losses._addr))
        _ = losses  # keep the Python object alive through the native call above
        var gradient_ptr = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=stacked._addr))
        _ = stacked  # keep the Python object alive through the native call above
        for k in range(a):  # driver-loop: over aligned microbatch gradient operations
            var xpart = generation_array([rows, length], "i")
            var ypart = generation_array([rows, length], "i")
            for which in range(2):  # driver-loop: over input and target buffer slices
                var source = ids if which == 0 else y
                var output = xpart if which == 0 else ypart
                memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=output._addr)),
                       src=Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=Int(py=source._addr)) + k * rows * length * 4,
                       count=rows * length * 4)
                _ = output  # keep the Python object alive through the native call above
                _ = source  # keep the Python object alive through the native call above
                _ = source^
                _ = output^
            var result = samba_gradients(model, xpart, ypart, PythonObject(count), stream, k * rows * length, context)
            loss_ptr[k] = Float32(Float64(py=result[0]))
            _ = losses  # keep the Python object alive through the native call above
            var gradients = result[1]
            if len(gradients) != len(fields["names"]):
                raise Error("Samba: gradient registry is incomplete")
            for j in range(len(gradients)):  # driver-loop: over the ordered parameter gradients
                var value = gradients[j]
                memcpy(dest=gradient_ptr + (k * n + Int(py=fields["offsets"][j])) * 4,
                       src=Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=Int(py=value._addr)), count=Int(py=value.nbytes))
                _ = stacked  # keep the Python object alive through the native call above
                _ = value  # keep the Python object alive through the native call above
            _ = gradients^
        gradient = stacked
        loss = PythonObject(Float64(loss_ptr[0]))
        _ = losses  # keep the Python object alive through the native call above
        if a > 1:
            gradient = generation_array([n])
            if provider is not None:
                _ = neural_descriptor_execute(provider, 9, [Int(py=gradient._addr), Int(py=stacked._addr)], [n, a, b * length], [])
            else:
                _ = py_call(_native_function(entries, "accumulate"), Python.list(gradient._addr, stacked._addr), py_ints([n, a, b * length]))
            _ = gradient  # keep the Python object alive through the native call above
            _ = stacked  # keep the Python object alive through the native call above
            var total = generation_array([1])
            if provider is not None:
                _ = neural_descriptor_execute(provider, 9, [Int(py=total._addr), Int(py=losses._addr)], [1, a, -1], [])
            else:
                _ = py_call(_native_function(entries, "accumulate"), Python.list(total._addr, losses._addr), py_ints([1, a, -1]))
            _ = total  # keep the Python object alive through the native call above
            _ = losses  # keep the Python object alive through the native call above
            loss = PythonObject(Float64(Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=Int(py=total._addr))[0]))
            _ = total  # keep the Python object alive through the native call above
            _ = total^
        _ = losses^
        _ = stacked^
    var device_parameters = Bool(py=py_get(opt, "_p_dev", PythonObject(False)))
    var parameter_owner = PythonObject(None)
    var parameter_buffer = fields["flat"]
    if not device_parameters:
        parameter_owner = _samba_parameter_pack(fields, opt, context)
        parameter_buffer = parameter_owner[0]
    var plan = _samba_step_plan(fields, opt, context)
    if not device_parameters:
        py_setitem(opt, "packed_", True)
    var info = generation_array([3])
    _samba_open_optimizer(opt, entries, context, provider)
    if device_parameters:
        if opt["_res"] is None or not Bool(py=py_call(bi.hasattr, entries, "optimizer_resident_step_io")):
            raise Error("RuntimeError: Samba optimizer device parameters require their native resident owner")
        # Preserve step_device semantics: forward reads the existing host
        # weights, while the optimizer updates its device parameters. Host
        # aliases remain stale until the caller requests params_to_host().
        var io = 2 if fields["max_norm"] is None else 10
        if provider is not None:
            _ = samba_optimizer_execute(provider, 27, [Int(py=info._addr), Int(py=gradient._addr), Int(py=opt["offsets"]._addr),
                Int(py=opt["buf_initialized"]._addr), Int(py=info._addr)], plan, Int(py=opt["_res"]), io)
        else:
            _ = py_call(_native_function(entries, "optimizer_resident_step_io"), info._addr, gradient._addr, opt["offsets"]._addr,
                opt["buf_initialized"]._addr, info._addr, plan, Python.list(opt["_res"], PythonObject(io)))
        _ = gradient  # keep the Python object alive through the native call above
        _ = info  # keep the Python object alive through the native call above
        py_setitem(opt, "_host_fresh", False)
        py_setitem(opt, "_g_dev", True)
    elif opt["_res"] is not None:
        if Int(py=plan[12]) != 0 and Bool(py=py_call(bi.hasattr, entries, "optimizer_maximize_dev")):
            if provider is not None:
                _ = samba_optimizer_execute(provider, 27, [Int(py=parameter_buffer._addr), Int(py=gradient._addr), Int(py=opt["offsets"]._addr),
                    Int(py=opt["buf_initialized"]._addr), Int(py=info._addr)], plan, Int(py=opt["_res"]), 15)
            else:
                _ = py_call(_native_function(entries, "optimizer_resident_step_io"), parameter_buffer._addr, gradient._addr, opt["offsets"]._addr,
                    opt["buf_initialized"]._addr, info._addr, plan, Python.list(opt["_res"], PythonObject(15)))
            _ = parameter_buffer  # keep the Python object alive through the native call above
            _ = gradient  # keep the Python object alive through the native call above
            _ = info  # keep the Python object alive through the native call above
        else:
            if provider is not None:
                _ = samba_optimizer_execute(provider, 26, [Int(py=parameter_buffer._addr), Int(py=gradient._addr), Int(py=opt["offsets"]._addr),
                    Int(py=opt["buf_initialized"]._addr), Int(py=info._addr)], plan, Int(py=opt["_res"]), 0)
            else:
                _ = py_call(_native_function(entries, "optimizer_resident_step"), parameter_buffer._addr, gradient._addr, opt["offsets"]._addr,
                    opt["buf_initialized"]._addr, info._addr, plan, opt["_res"])
            _ = parameter_buffer  # keep the Python object alive through the native call above
            _ = gradient  # keep the Python object alive through the native call above
            _ = info  # keep the Python object alive through the native call above
        py_setitem(opt, "_host_fresh", False)
    else:
        var moments = _samba_moments(opt, context)
        if provider is not None:
            _ = samba_optimizer_execute(provider, 25, [Int(py=parameter_buffer._addr), Int(py=gradient._addr), Int(py=moments[0]._addr),
                Int(py=moments[1]._addr), Int(py=opt["offsets"]._addr), Int(py=opt["buf_initialized"]._addr), Int(py=info._addr)], plan, 0, 0)
        else:
            _ = py_call(_native_function(entries, "optimizer_step"), parameter_buffer._addr, gradient._addr, moments[0]._addr, moments[1]._addr,
                opt["offsets"]._addr, opt["buf_initialized"]._addr, info._addr, plan)
        _ = parameter_buffer  # keep the Python object alive through the native call above
        _ = gradient  # keep the Python object alive through the native call above
        _ = info  # keep the Python object alive through the native call above
        for i in range(2):  # driver-loop: over caller-owned moment aliases
            var original = opt["_m" if i == 0 else "_v"]
            if moments[i] is not original:
                var view = samba_moment_view(original, context)
                py_setitem(view, py_call(bi.slice, PythonObject(None)), moments[i]._mv)
        _ = moments^
    if not device_parameters:
        _samba_parameter_publish(parameter_buffer, parameter_owner[1], fields)
    _ = parameter_owner^
    _ = parameter_buffer^
    _ = gradient^
    return _samba_step_finish(fields, opt, info, loss)
