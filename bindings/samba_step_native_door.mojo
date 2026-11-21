# SPDX-License-Identifier: Apache-2.0
"""Native standard Samba step with the existing resident and accumulation ABIs.

This public execution owner accepts native built-in schedule configurations;
arbitrary Python callbacks are refused before state mutation.

UNPACK and PACK of one train step. RUN is training/samba_step_run.mojo (the
checks, the plan, the microbatch slices, the stacked gradient, the aligned
accumulation, the report words) and core/samba_optimizer_plan.mojo (the
optimizer entry call): typed words, no Python object.
"""
from core.weight_epoch import registry_epoch_bump
from providers.python.samba_execution_pyapi_door import samba_provider_accumulated_gradients
from providers.python.samba_resident_pyapi_door import samba_resident_provider, samba_resident_native_step
from providers.python.neural_descriptor_pyapi_door import (
    NeuralDescriptorProvider, neural_descriptor_execute, neural_descriptor_provider_id,
)
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.optimizer_owner_boundary_unpack import OptimizerOwner, OptimizerOwnerLease
from bindings.optimizer_descriptor_boundary_door import optimizer_driver
from bindings.samba_optimizer_boundary_unpack import samba_optimizer_provider, samba_optimizer_primitives, samba_optimizer_execute, samba_optimizer_transfer
from core.samba_optimizer_plan import SambaOptimizerPlan
from bindings.native_loaded_module_door import native_loaded_module, NativeModules
from bindings.p3_samba_caps_door import samba_caps, SAMBA_CAP_RESIDENT_ABI, SAMBA_CAP_RESIDENT_ON, SAMBA_CAP_ACCUMULATED, SAMBA_CAP_CHUNKED
from bindings.generation_buffers_door import generation_initialize_binding, generation_array
from bindings.generator_state_native_pack import generator_next
from bindings.samba_state_native_door import samba_fields, samba_schedule_snapshot, samba_schedule_rate, samba_moment_view
from bindings.samba_route_unpack import samba_route
from training.samba_state_run import samba_optimizer_device_plan, samba_max_norm_check
# RUN: typed pieces, no Python import.
from training.samba_step_run import (
    samba_step_index_check, samba_step_plan_run, samba_step_split_check, samba_registry_count_check,
    samba_microbatch_copy_run, samba_gradient_stack_run, samba_loss_store, samba_float_read,
    samba_accumulate_admit, samba_accumulate_run, samba_step_io_device, samba_step_io_host, samba_step_clipped,
    samba_step_rate_reported, samba_parameter_format_check, samba_parameter_extent_check,
)
from core.device_plan import DevicePlan
from bindings.samba_training_native_door import samba_ids, samba_gradients_admitted, _samba_count
from bindings.causal_lm_construct_door import _model_f32, _model_shape
from bindings.py_balanced import py_call, py_get, py_setitem, py_type
from bindings.samba_keys_unpack import samba_keys
from bindings.py_balanced import py_item


@fieldwise_init
struct SambaMomentOwners(Movable):
    var first: PythonObject
    var second: PythonObject


def _samba_moments(opt: PythonObject, context: PythonObject) raises -> SambaMomentOwners:
    var nk = samba_keys()
    var n = Int(py=py_item(opt, nk[].k_n_total))
    if py_item(opt, nk[].k__m) is None:
        py_setitem(opt, nk[].k__m, generation_array([n]))
    var first = _model_f32(py_item(opt, nk[].k__m), "_m", py_item(context, nk[].k_neural))
    _model_shape(first, [n], "_m")
    if py_item(opt, nk[].k__v) is None:
        py_setitem(opt, nk[].k__v, generation_array([n]))
    var second = _model_f32(py_item(opt, nk[].k__v), "_v", py_item(context, nk[].k_neural))
    _model_shape(second, [n], "_v")
    return SambaMomentOwners(first, second)


def _samba_open_optimizer(opt: PythonObject, mut record: OptimizerOwner, entries: PythonObject,
    context: PythonObject, model_provider: PythonObject, device_plan: DevicePlan) raises:
    var nk = samba_keys()
    var provider = samba_optimizer_provider(entries)
    if record.control.handle >= 0:
        if record.provider.downcast_value_ptr[NeuralDescriptorProvider]()[].module is not provider.downcast_value_ptr[NeuralDescriptorProvider]()[].module:
            raise Error("Samba optimizer provider belongs to another binding")
        return
    if py_item(opt, nk[].k__resident_wanted) is PythonObject(False):
        return
    var driver = optimizer_driver(provider)
    # The resident optimizer is the one route (MOJOLEARN_OPTIMIZER_RESIDENT_OFF
    # lost A/B ab3, deleted 2026-11-18 by lane/fx-verdicts2).
    var enabled = not driver.host and device_plan.setting == "1"
    if not enabled:
        py_setitem(opt, nk[].k__resident_wanted, False)
        return
    if model_provider is not None and model_provider.downcast_value_ptr[NeuralDescriptorProvider]()[].module is not provider.downcast_value_ptr[NeuralDescriptorProvider]()[].module:
        raise Error("Samba optimizer provider belongs to another binding")
    var handle = -1
    var count = Int(py=py_item(opt, nk[].k_n_total))
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
    var nk = samba_keys()
    # `schedule` is the snapshot samba_train_step admitted at its start; the
    # step never changes opt["lr_schedule"], so it is not taken a second time.
    # UNPACK the optimizer's scalars (once per step); the checks and the plan
    # are RUN. `max_norm` is read once (three reads until 2026-11-21).
    var step = Int(py=py_item(opt, nk[].k_t))
    samba_step_index_check(step)
    if schedule is not None:
        py_setitem(opt, nk[].k_lr, samba_schedule_rate(schedule, step + 1, context))
    var limit = py_item(fields, nk[].k_max_norm)
    var given = limit is not None
    var max_norm = 0.0
    if given:
        max_norm = Float64(py=limit)
    samba_max_norm_check(given, max_norm)
    py_setitem(opt, nk[].k_t, step + 1)
    var betas = py_item(opt, nk[].k_betas)
    return samba_step_plan_run(len(py_item(opt, nk[].k_params)), step + 1,
        Bool(py=py_get(opt, nk[].k_maximize, False)), Float64(py=py_item(opt, nk[].k_lr)),
        Float64(py=betas[0]), Float64(py=betas[1]), Float64(py=py_item(opt, nk[].k_eps)),
        Float64(py=py_item(opt, nk[].k_weight_decay)), max_norm)


def _samba_step_finish(fields: PythonObject, opt: PythonObject, info: PythonObject, loss: PythonObject) raises -> PythonObject:
    var nk = samba_keys()
    # PACK of the step report: info is [flag, total_norm, clip_coef].
    var address = Int(py=info._addr)
    var clipped = samba_step_clipped(address)
    py_setitem(opt, nk[].k_total_norm_, PythonObject(samba_float_read(address, 1)) if clipped else PythonObject(None))
    py_setitem(opt, nk[].k_clip_coef_, PythonObject(samba_float_read(address, 2)) if clipped else PythonObject(None))
    # `info` is borrowed from the caller for this entire function; it cannot
    # be consumed here, and the borrow retains storage through the reads.
    py_setitem(opt, nk[].k_lr_, PythonObject(samba_step_rate_reported(Float64(py=py_item(opt, nk[].k_lr)))))
    var result = Python.dict()
    py_setitem(result, nk[].k_loss, loss)
    py_setitem(result, nk[].k_lr, py_item(opt, nk[].k_lr_))
    py_setitem(result, nk[].k_step, py_item(opt, nk[].k_t))
    py_setitem(result, nk[].k_total_norm, py_item(opt, nk[].k_total_norm_))
    py_setitem(fields, nk[].k_last_, result)
    return py_call(native_loaded_module("builtins").dict, result)


def _samba_resident_step(fields: PythonObject, opt: PythonObject, mut record: OptimizerOwner, entries: PythonObject, ids: PythonObject, targets: PythonObject, context: PythonObject,
                         device_plan: DevicePlan, schedule: PythonObject, accumulation: Int = 1, total_count: Int = 0) raises -> PythonObject:
    var nk = samba_keys()
    var config = samba_fields(py_item(fields, nk[].k_config))
    if Float64(py=py_item(config, nk[].k_dropout)) != 0.0:
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
    # The resident session is the primitive entry's native owner (op 12,
    # training/native_resident_dispatch.mojo), opened by
    # samba_resident_native_step below on whichever primitive provider the
    # stack connected. Lane samba-sessions-native (2026-11-12) deleted the
    # second, Python-object session (`samba_resident_open`, a
    # `_SambaResidentSession` held in `_resident_handle` and adopted through
    # `samba_resident_native_handle_v1`): the same SambaResidentSession, the
    # same constructor words.
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
    var scalar = PythonObject(samba_float_read(Int(py=loss._addr), 0))
    _ = loss  # keep the Python object alive through the native call above
    _ = loss^
    return _samba_step_finish(fields, opt, info, scalar)



def _samba_parameter_pack(fields: PythonObject, opt: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = samba_keys()
    # Parameters may have been rebound after construction. Preserve the
    # optimizer's writable C-buffer interface instead of assuming flat views.
    var bi = native_loaded_module("builtins")
    var params = py_item(opt, nk[].k_params)
    if len(params) != len(py_item(fields, nk[].k_names)):
        raise Error("Samba: optimizer parameter registry has changed")
    var views = bi.list()
    var original_flat = True
    for i in range(len(params)):  # driver-loop: over caller-owned optimizer parameter buffers
        var value = params[i]
        var view = PythonObject(None)
        var shape = PythonObject(None)
        if py_type(value) is py_item(py_item(context, nk[].k_neural), nk[].k_array_class):
            if String(value.dtype) != "<f4":
                raise Error("TypeError: Samba optimizer parameters must be float32")
            # Lane x56d: the model's own registry views are read-only to
            # callers (bindings/samba_state_native.mojo `samba_view`); the
            # native step writes them by address. A rebound parameter is
            # written back through its buffer, so it must be writable.
            var own = Int(py=value._addr) == Int(py=py_item(fields, nk[].k_flat)._addr) + 4 * Int(py=py_item(fields, nk[].k_offsets)[i])
            if String(value.order) != "C" or (Bool(py=value._readonly) and not own):
                raise Error("Samba optimizer parameters must be C-contiguous and writable")
            view = value._mv
            shape = value.shape
        else:
            view = py_call(bi.memoryview, value)
            samba_parameter_format_check(String(view.format))
            if not Bool(py=view.c_contiguous) or Bool(py=view.readonly):
                raise Error("Samba optimizer parameters must be C-contiguous and writable")
            shape = view.shape
        if Bool(py=shape != py_item(fields, nk[].k_shapes)[py_item(fields, nk[].k_names)[i]]):
            raise Error("Samba optimizer parameter shape differs from its gradient")
        var raw = py_call(view.cast, "B")
        var begin = 4 * Int(py=py_item(fields, nk[].k_offsets)[i])
        var end = 4 * Int(py=py_item(fields, nk[].k_offsets)[i + 1])
        samba_parameter_extent_check(len(raw), begin, end)
        original_flat = original_flat and py_type(value) is py_item(py_item(context, nk[].k_neural), nk[].k_array_class)
        if original_flat:
            original_flat = Int(py=value._addr) == Int(py=py_item(fields, nk[].k_flat)._addr) + begin
            _ = value  # keep the Python object alive through the native call above
        _ = py_call(views.append, raw)
    if original_flat:
        return Python.tuple(py_item(fields, nk[].k_flat), bi.list())
    var flat = generation_array([Int(py=py_item(fields, nk[].k_n_total))])
    var destination = py_call(flat._mv.cast, "B")
    for i in range(len(views)):  # driver-loop: over replacement parameter buffers
        var begin = 4 * Int(py=py_item(fields, nk[].k_offsets)[i])
        var end = 4 * Int(py=py_item(fields, nk[].k_offsets)[i + 1])
        py_setitem(destination, py_call(bi.slice, begin, end), views[i])
    return Python.tuple(flat, views)


def _samba_parameter_publish(flat: PythonObject, views: PythonObject, fields: PythonObject) raises:
    var nk = samba_keys()
    var bi = native_loaded_module("builtins")
    var raw = py_call(flat._mv.cast, "B")
    for i in range(len(views)):  # driver-loop: over caller-owned parameter write-back buffers
        var begin = 4 * Int(py=py_item(fields, nk[].k_offsets)[i])
        var end = 4 * Int(py=py_item(fields, nk[].k_offsets)[i + 1])
        var target = views[i]
        py_setitem(target, py_call(bi.slice, PythonObject(None)), raw[py_call(bi.slice, begin, end)])

def samba_train_step(model: PythonObject, inputs: PythonObject, targets: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = samba_keys()
    var fields = samba_fields(model)
    var optimizer = py_item(fields, nk[].k_optimizer)
    var lease = OptimizerOwnerLease(optimizer)
    var record = lease.value.downcast_value_ptr[OptimizerOwner]()
    try:
        return _samba_train_step_admitted(model, inputs, targets, context, record[])
    finally:
        _ = lease
        _ = optimizer


def _samba_train_step_admitted(model: PythonObject, inputs: PythonObject, targets: PythonObject,
    context: PythonObject, mut record: OptimizerOwner) raises -> PythonObject:
    var nk = samba_keys()
    # The optimizer device plan, read from the environment once per step.
    var device_plan = samba_optimizer_device_plan()
    _ = generation_initialize_binding(py_item(py_item(context, nk[].k_neural), nk[].k_generation_classes))
    var bi = native_loaded_module("builtins")
    var fields = samba_fields(model)
    var config = samba_fields(py_item(fields, nk[].k_config))
    var opt = samba_fields(py_item(fields, nk[].k_optimizer))
    # The stack's one route (bindings/samba_route_unpack.mojo): the entries
    # of its device, resolved once; the resident optimizer opens on them.
    var route = samba_route(model, fields, context)
    var entries = route.entries
    # Admit the built-in schedule before consuming a stream or changing state.
    var schedule = samba_schedule_snapshot(py_item(opt, nk[].k_lr_schedule), context)
    var ids = samba_ids(inputs, Int(py=py_item(config, nk[].k_vocab)), False, context)
    var y = samba_ids(targets, Int(py=py_item(config, nk[].k_vocab)), True, context)
    if Bool(py=ids.shape != y.shape):
        raise Error("Samba: targets must match inputs' shape")
    var b = Int(py=ids.shape[0])
    var length = Int(py=ids.shape[1])
    var a = Int(py=py_item(opt, nk[].k_accumulation_steps))
    samba_step_split_check(a, b, length)
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
    # The provider id, unpacked once for the per-parameter calls below; the
    # provider stays pinned by its use after them.
    var provider_id = 0
    if provider is not None:
        provider_id = neural_descriptor_provider_id(provider)
    var stream = generator_next(py_item(fields, nk[].k_generator)) if Float64(py=py_item(config, nk[].k_dropout)) > 0.0 else PythonObject(None)
    var completed = samba_provider_accumulated_gradients(fields, config, entries, provider, ids, y, stream, a, count, py_item(context, nk[].k_neural))
    var gradient = PythonObject(None)
    var loss = PythonObject(None)
    if completed is not None:
        loss = completed[0]
        gradient = completed[1]
    else:
        # The per-call route (no resident owner, or a refused resident
        # step): one gradient operation per aligned microbatch, stacked, then
        # one aligned reduction. The slices, the stack and the reduction are
        # RUN (training/samba_step_run.mojo).
        var rows = b // a
        var n = Int(py=py_item(fields, nk[].k_n_total))
        var stacked = generation_array([a, n])
        var losses = generation_array([a])
        var losses_addr = Int(py=losses._addr)
        var stacked_addr = Int(py=stacked._addr)
        var ids_addr = Int(py=ids._addr)
        var y_addr = Int(py=y._addr)
        # Serial by nature: each microbatch runs the stack's blocks, whose
        # tape records the block door holds one operation at a time.
        for k in range(a):  # driver-loop: over aligned microbatch gradient operations
            var xpart = generation_array([rows, length], "i")
            var ypart = generation_array([rows, length], "i")
            samba_microbatch_copy_run(Int(py=xpart._addr), Int(py=ypart._addr), ids_addr, y_addr, k, rows * length)
            var result = samba_gradients_admitted(model, xpart, ypart, PythonObject(count), stream, k * rows * length, context)
            samba_loss_store(losses_addr, k, Float64(py=result[0]))
            var gradients = result[1]
            samba_registry_count_check(len(gradients), len(py_item(fields, nk[].k_names)))
            for j in range(len(gradients)):  # driver-loop: over the ordered parameter gradients
                var value = gradients[j]
                samba_gradient_stack_run(stacked_addr, k, n, Int(py=py_item(fields, nk[].k_offsets)[j]),
                                         Int(py=value._addr), Int(py=value.nbytes))
                _ = value  # keep the Python object alive through the copy above
            _ = gradients^
        _ = ids  # the slices above read both by address
        _ = y
        gradient = stacked
        loss = PythonObject(samba_float_read(losses_addr, 0))
        if a > 1:
            gradient = generation_array([n])
            var total = generation_array([1])
            var gradient_addr = Int(py=gradient._addr)
            var total_addr = Int(py=total._addr)
            var primitives = route.primitives()
            # Primitive op 9 (the former `accumulate` entry) for the gradient
            # and the loss under ONE GIL release (two until 2026-11-21).
            samba_accumulate_admit(provider_id)
            with GILReleased(Python()):
                samba_accumulate_run(primitives, provider_id, gradient_addr, stacked_addr, n, total_addr, losses_addr,
                                     a, b * length)
            loss = PythonObject(samba_float_read(total_addr, 0))
            _ = total^
        _ = losses^
        _ = stacked^
    var device_parameters = record.control.parameters_device
    var parameter_owner = PythonObject(None)
    var parameter_buffer = py_item(fields, nk[].k_flat)
    var registry = parameter_buffer  # the model's flat registry this step writes
    if not device_parameters:
        parameter_owner = _samba_parameter_pack(fields, opt, context)
        parameter_buffer = parameter_owner[0]
    var plan = _samba_step_plan(fields, opt, context, schedule)
    if not device_parameters:
        py_setitem(opt, nk[].k_packed_, True)
    var info = generation_array([3])
    _samba_open_optimizer(opt, record, entries, context, provider, device_plan)
    var optimizer_provider = samba_optimizer_primitives(entries)
    var offsets = py_item(opt, nk[].k_offsets)
    var initialized = py_item(opt, nk[].k_buf_initialized)
    if device_parameters:
        if record.control.handle < 0 or not Bool(py=py_item(context, nk[].k_optimizer_step_io)):
            raise Error("RuntimeError: Samba optimizer device parameters require their native resident owner")
        # Preserve step_device semantics and the existing upload/download mask.
        var io = samba_step_io_device(py_item(fields, nk[].k_max_norm) is not None)
        _ = samba_optimizer_execute(optimizer_provider, 27,
            [Int(py=info._addr), Int(py=gradient._addr), Int(py=offsets._addr),
             Int(py=initialized._addr), Int(py=info._addr)], plan, record.control.handle, io)
        record.control.host_fresh = False
        record.control.pristine = False
        record.control.gradients_device = True
    elif record.control.handle >= 0:
        var io = samba_step_io_host(plan.hyper.maximize,
            plan.hyper.maximize and Bool(py=py_item(context, nk[].k_optimizer_maximize_device)))
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
    # The written registry's own epoch (core/weight_epoch.mojo): stamped by
    # the step from the registry object it already holds (one slot read),
    # where the public door read the model's fields, `flat` and its address
    # twice per call (hot-path audit 2026-11-13, BLOCK pack:69/72). Lane
    # fix-wave2-leftovers: the address comes from the optimizer's native
    # owner, which read it once for this registry (no `_addr` read per step).
    registry_epoch_bump(record.registry_address(registry))
    _ = registry^
    return _samba_step_finish(fields, opt, info, loss)
