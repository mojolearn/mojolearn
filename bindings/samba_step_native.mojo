# SPDX-License-Identifier: Apache-2.0
"""Native standard Samba step with the existing resident and accumulation ABIs.

This public execution owner accepts native built-in schedule configurations;
arbitrary Python callbacks are refused before state mutation.
"""
from providers.python.samba_execution_pyapi import samba_provider_accumulated_gradients
from providers.python.samba_resident_pyapi import samba_resident_provider, samba_resident_session, samba_resident_native_step
from providers.python.neural_descriptor_pyapi import NeuralDescriptorProvider, neural_descriptor_execute
from training.checks.optimizer_contract import microbatch_split_is_identical
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.optimizer_owner_boundary import OptimizerOwner, OptimizerOwnerLease
from bindings.optimizer_descriptor_boundary import optimizer_driver
from bindings.samba_optimizer_boundary import samba_optimizer_provider, samba_optimizer_primitives, samba_optimizer_execute, samba_optimizer_transfer
from core.samba_optimizer_plan import SambaOptimizerPlan
from training.optimizer_public_common import OptHyper
from bindings.native_loaded_module import native_loaded_module, NativeModules
from bindings.p3_samba_caps import samba_caps, SAMBA_CAP_RESIDENT_ABI, SAMBA_CAP_RESIDENT_ON, SAMBA_CAP_ACCUMULATED, SAMBA_CAP_CHUNKED
from std.memory import unsafe_memcpy
from std.os import getenv
from bindings.generation_buffers import generation_initialize_binding, generation_array, py_ints
from bindings.generator_state_native import generator_next
from bindings.samba_state_native import samba_fields, samba_entries, samba_schedule_snapshot, samba_schedule_rate, samba_moment_view, samba_optimizer_device_plan
from core.device_plan import DevicePlan
from bindings.samba_training_native import samba_ids, samba_gradients_admitted, _samba_count
from bindings.causal_lm_construct import _model_f32, _model_shape
from bindings.causal_lm_public import _native_function
from bindings.py_balanced import py_call, py_get, py_setitem, py_type


@fieldwise_init
struct SambaMomentOwners(Movable):
    var first: PythonObject
    var second: PythonObject


def _samba_moments(opt: PythonObject, context: PythonObject) raises -> SambaMomentOwners:
    var n = Int(py=opt["n_total"])
    if opt["_m"] is None:
        py_setitem(opt, "_m", generation_array([n]))
    var first = _model_f32(opt["_m"], "_m", context["neural"])
    _model_shape(first, [n], "_m")
    if opt["_v"] is None:
        py_setitem(opt, "_v", generation_array([n]))
    var second = _model_f32(opt["_v"], "_v", context["neural"])
    _model_shape(second, [n], "_v")
    return SambaMomentOwners(first, second)


def _samba_open_optimizer(opt: PythonObject, mut record: OptimizerOwner, entries: PythonObject,
    context: PythonObject, model_provider: PythonObject, device_plan: DevicePlan) raises:
    var provider = samba_optimizer_provider(entries)
    if record.control.handle >= 0:
        if record.provider.downcast_value_ptr[NeuralDescriptorProvider]()[].module is not provider.downcast_value_ptr[NeuralDescriptorProvider]()[].module:
            raise Error("Samba optimizer provider belongs to another binding")
        return
    if opt["_resident_wanted"] is PythonObject(False):
        return
    var driver = optimizer_driver(provider)
    var enabled = not driver.host and getenv("MOJOLEARN_OPTIMIZER_RESIDENT", "1") != "0" and device_plan.setting == "1"
    if not enabled:
        py_setitem(opt, "_resident_wanted", False)
        return
    if model_provider is not None and model_provider.downcast_value_ptr[NeuralDescriptorProvider]()[].module is not provider.downcast_value_ptr[NeuralDescriptorProvider]()[].module:
        raise Error("Samba optimizer provider belongs to another binding")
    var handle = -1
    var count = Int(py=opt["n_total"])
    with GILReleased(Python()):
        handle = driver.open(count)
    # Attach immediately: a failed moment admission/upload still has one owner
    # responsible for the slot. No raw handle escapes without an image pin.
    record.attach(handle, driver, provider)
    if not record.control.pristine:
        try:
            var moments = _samba_moments(opt, context)
            samba_optimizer_transfer(record, Int(py=moments.first._addr), Int(py=moments.second._addr), count, True)
            _ = moments
        except problem:
            record.close()
            raise problem^
    record.control.host_fresh = True
    _ = provider


def _samba_step_plan(fields: PythonObject, opt: PythonObject, context: PythonObject, schedule: PythonObject) raises -> SambaOptimizerPlan:
    # `schedule` is the snapshot samba_train_step admitted at its start; the
    # step never changes opt["lr_schedule"], so it is not taken a second time.
    var step = Int(py=opt["t"])
    if step < 0 or step >= 2147483647:
        raise Error("Samba: optimizer step exceeds signed 32-bit indexing")
    if schedule is not None:
        py_setitem(opt, "lr", samba_schedule_rate(schedule, step + 1, context))
    var max_norm = 0.0 if fields["max_norm"] is None else Float64(py=fields["max_norm"])
    if fields["max_norm"] is not None and not (max_norm > 0.0):
        raise Error("Samba: max_norm must be positive or None")
    py_setitem(opt, "t", step + 1)
    var betas = opt["betas"]
    var hyper = OptHyper(2, 0, Bool(py=py_get(opt, "maximize", False)),
        Float64(py=opt["lr"]), Float64(py=betas[0]), Float64(py=betas[1]),
        Float64(py=opt["eps"]), Float64(py=opt["weight_decay"]), 0.0, 0.0, max_norm)
    return SambaOptimizerPlan(len(opt["params"]), step + 1, hyper)


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


def _samba_resident_step(fields: PythonObject, opt: PythonObject, mut record: OptimizerOwner, entries: PythonObject, ids: PythonObject, targets: PythonObject, context: PythonObject,
                         device_plan: DevicePlan, schedule: PythonObject, accumulation: Int = 1, total_count: Int = 0) raises -> PythonObject:
    var config = samba_fields(fields["config"])
    if Float64(py=config["dropout"]) != 0.0:
        return PythonObject(None)
    var provider = samba_resident_provider(fields, entries)
    # The resident ABI and the compile-time switches of one connected binding
    # are fixed: read them from the per-binding capability record
    # (bindings/p3_samba_caps.mojo, weakly keyed by the binding module). The
    # former record was keyed on the `entries` SimpleNamespace, which cannot
    # be weakly referenced, so it never stored and every step re-probed.
    var caps = samba_caps(entries)
    if provider is None and not (Bool(py=caps[SAMBA_CAP_RESIDENT_ABI]) and Bool(py=caps[SAMBA_CAP_RESIDENT_ON])):
        return PythonObject(None)
    if provider is not None and (not Bool(py=caps[SAMBA_CAP_RESIDENT_ABI]) or neural_descriptor_execute(provider, 24, [], [], [], False) == 0):
        return PythonObject(None)
    if accumulation > 1 and not Bool(py=caps[SAMBA_CAP_ACCUMULATED]):
        return PythonObject(None)
    if provider is None and Bool(py=caps[SAMBA_CAP_CHUNKED]):
        return PythonObject(None)
    _samba_open_optimizer(opt, record, entries, context, provider, device_plan)
    if record.control.handle < 0:
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
    var plan = _samba_step_plan(fields, opt, context, schedule)
    var info = generation_array([3])
    var loss = generation_array([1])
    var optimizer_provider = provider if provider is not None else samba_optimizer_primitives(entries)
    samba_resident_native_step(fields, config, optimizer_provider, opt, ids, targets, info, loss,
        plan, record.control.handle, accumulation, total_count)
    _ = optimizer_provider
    _ = loss  # keep the Python object alive through the native call above
    _ = info  # keep the Python object alive through the native call above
    record.control.host_fresh = False
    record.control.pristine = False
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
            # Lane x56d: the model's own registry views are read-only to
            # callers (bindings/samba_state_native.mojo `samba_view`); the
            # native step writes them by address. A rebound parameter is
            # written back through its buffer, so it must be writable.
            var own = Int(py=value._addr) == Int(py=fields["flat"]._addr) + 4 * Int(py=fields["offsets"][i])
            if String(value.order) != "C" or (Bool(py=value._readonly) and not own):
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
    var fields = samba_fields(model)
    var optimizer = fields["optimizer"]
    var lease = OptimizerOwnerLease(optimizer)
    var record = lease.value.downcast_value_ptr[OptimizerOwner]()
    try:
        return _samba_train_step_admitted(model, inputs, targets, context, record[])
    finally:
        _ = lease
        _ = optimizer


def _samba_train_step_admitted(model: PythonObject, inputs: PythonObject, targets: PythonObject,
    context: PythonObject, mut record: OptimizerOwner) raises -> PythonObject:
    # The optimizer device plan, read from the environment once per step.
    var device_plan = samba_optimizer_device_plan()
    _ = generation_initialize_binding(context["neural"]["generation_classes"])
    var bi = native_loaded_module("builtins")
    var fields = samba_fields(model)
    var config = samba_fields(fields["config"])
    var opt = samba_fields(fields["optimizer"])
    var entries = samba_entries(context, fields["numeric_mode"])
    # Admit the built-in schedule before consuming a stream or changing state.
    var schedule = samba_schedule_snapshot(opt["lr_schedule"], context)
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
        var resident = _samba_resident_step(fields, opt, record, entries, ids, y, context, device_plan, schedule)
        if resident is not None:
            return resident
    var count = _samba_count(y, context)
    if a > 1:
        var resident = _samba_resident_step(fields, opt, record, entries, ids, y, context, device_plan, schedule, a, count)
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
                unsafe_memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=output._addr)),
                       src=Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=Int(py=source._addr)) + k * rows * length * 4,
                       count=rows * length * 4)
                _ = output  # keep the Python object alive through the native call above
                _ = source  # keep the Python object alive through the native call above
                _ = source^
                _ = output^
            var result = samba_gradients_admitted(model, xpart, ypart, PythonObject(count), stream, k * rows * length, context)
            loss_ptr[k] = Float32(Float64(py=result[0]))
            _ = losses  # keep the Python object alive through the native call above
            var gradients = result[1]
            if len(gradients) != len(fields["names"]):
                raise Error("Samba: gradient registry is incomplete")
            for j in range(len(gradients)):  # driver-loop: over the ordered parameter gradients
                var value = gradients[j]
                unsafe_memcpy(dest=gradient_ptr + (k * n + Int(py=fields["offsets"][j])) * 4,
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
    var device_parameters = record.control.parameters_device
    var parameter_owner = PythonObject(None)
    var parameter_buffer = fields["flat"]
    if not device_parameters:
        parameter_owner = _samba_parameter_pack(fields, opt, context)
        parameter_buffer = parameter_owner[0]
    var plan = _samba_step_plan(fields, opt, context, schedule)
    if not device_parameters:
        py_setitem(opt, "packed_", True)
    var info = generation_array([3])
    _samba_open_optimizer(opt, record, entries, context, provider, device_plan)
    var optimizer_provider = samba_optimizer_primitives(entries)
    var offsets = opt["offsets"]
    var initialized = opt["buf_initialized"]
    if device_parameters:
        if record.control.handle < 0 or not Bool(py=context["optimizer_step_io"]):
            raise Error("RuntimeError: Samba optimizer device parameters require their native resident owner")
        # Preserve step_device semantics and the existing upload/download mask.
        var io = 2 if fields["max_norm"] is None else 10
        _ = samba_optimizer_execute(optimizer_provider, 27,
            [Int(py=info._addr), Int(py=gradient._addr), Int(py=offsets._addr),
             Int(py=initialized._addr), Int(py=info._addr)], plan, record.control.handle, io)
        record.control.host_fresh = False
        record.control.pristine = False
        record.control.gradients_device = True
    elif record.control.handle >= 0:
        var io = 15 if plan.hyper.maximize and Bool(py=context["optimizer_maximize_device"]) else 0
        _ = samba_optimizer_execute(optimizer_provider, 27 if io else 26,
            [Int(py=parameter_buffer._addr), Int(py=gradient._addr), Int(py=offsets._addr),
             Int(py=initialized._addr), Int(py=info._addr)], plan, record.control.handle, io)
        record.control.host_fresh = False
        record.control.pristine = False
    else:
        var moments = _samba_moments(opt, context)
        _ = samba_optimizer_execute(optimizer_provider, 25,
            [Int(py=parameter_buffer._addr), Int(py=gradient._addr), Int(py=moments.first._addr),
             Int(py=moments.second._addr), Int(py=offsets._addr),
             Int(py=initialized._addr), Int(py=info._addr)], plan, 0, 0)
        record.control.pristine = False
        for i in range(2):  # driver-loop: the two writable public moment owners
            var original = opt["_m" if i == 0 else "_v"]
            var completed = moments.first if i == 0 else moments.second
            if completed is not original:
                var view = samba_moment_view(original, context)
                py_setitem(view, py_call(bi.slice, PythonObject(None)), completed._mv)
        _ = moments
    _ = optimizer_provider
    _ = offsets
    _ = initialized
    _ = info
    if not device_parameters:
        _samba_parameter_publish(parameter_buffer, parameter_owner[1], fields)
    _ = parameter_owner^
    _ = parameter_buffer^
    _ = gradient^
    return _samba_step_finish(fields, opt, info, loss)
