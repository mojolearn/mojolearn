# SPDX-License-Identifier: Apache-2.0
"""Native Samba snapshots and publication over the existing ordered flat state.

Built-in schedules are native configurations. Python callback schedules are
refused before execution under the native-only public runtime contract.
"""
from core.weight_epoch import registry_epoch_bump
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.p3nc_fitted_freeze_unpack import fitted_freeze_any
from bindings.neural_block_admission_door import neural_array_snapshot
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from core.native_descriptor_abi import PRIMITIVE_ABI_NAME
from providers.python.neural_descriptor_pyapi_door import neural_descriptor_cached
from std.memory import unsafe_memcpy
from bindings.py_balanced import py_call, py_contains, py_get, py_setattr, py_setitem, py_type
from std.python._cpython import GILReleased
from core.entry_table import EntryRef, entry_ref_call
from core.samba_entry_table import SAMBA_PRIM_OPTIMIZER_STEP
from bindings.generation_buffers_door import generation_initialize_binding, generation_array, generation_copy, py_ints
from bindings.causal_lm_construct_door import _model_f32, _model_shape, _model_block, _model_bindings, _model_native_buffer
from bindings.causal_lm_public_door import _native_function
from bindings.samba_route_unpack import samba_route, samba_route_process, samba_route_publish, samba_device_admit
from bindings.samba_config_native_door import samba_config_dict, samba_config_from_dict, samba_registry, samba_block_shapes
from bindings.optimizer_owner_boundary_unpack import OptimizerOwner, OptimizerOwnerLease
from bindings.optimizer_owner_boundary_pack import optimizer_owner_new
from bindings.samba_optimizer_boundary_unpack import samba_optimizer_transfer
# RUN (lane rs-neural): typed pieces, no Python import.
from training.samba_state_run import samba_optimizer_device_plan
from bindings.samba_keys_unpack import samba_keys
from bindings.py_balanced import py_item


def samba_fields(value: PythonObject) raises -> PythonObject:
    return py_call(native_loaded_module("builtins").object.__getattribute__, value, "__dict__")


def samba_primitive(entry: EntryRef, op: Int, addresses: List[Int], ints: List[Int],
                    floats: List[Float64]) raises -> Int:
    """One synchronous neural primitive call through the Samba entry the
    caller resolved once (`samba_ref`), with the GIL released, as the former
    Python entries ran it. The caller keeps every addressed buffer alive past
    the call."""
    var result = 0
    with GILReleased(Python()):
        result = entry_ref_call(entry, op, addresses, ints, floats)
    return result


def samba_experiment(entries: PythonObject) raises -> PythonObject:
    if Bool(py=py_call(native_loaded_module("builtins").hasattr, entries, "training_experiment_profile")):
        return _native_function(entries, "training_experiment_profile")()
    return PythonObject("baseline")


def samba_view(flat: PythonObject, offset: Int, dims: List[Int], context: PythonObject) raises -> PythonObject:
    var nk = samba_keys()
    var bi = native_loaded_module("builtins")
    var n = 1
    for d in dims:  # driver-loop: over tensor shape metadata
        if d < 0 or (d != 0 and n > 2147483647 // d):
            raise Error("Samba: tensor shape exceeds signed 32-bit indexing")
        n *= d
    if offset < 0 or offset + n > Int(py=flat.size):
        raise Error("Samba: tensor view exceeds flat parameter storage")
    var result = py_call(bi.object.__new__, py_item(py_item(context, nk[].k_neural), nk[].k_array_class))
    py_setattr(result, "_store", flat._store)
    py_setattr(result, "_base", flat)
    py_setattr(result, "_pin", flat._pin)
    py_setattr(result, "_mv", flat._mv[py_call(bi.slice, offset, offset + n)])
    py_setattr(result, "_addr", PythonObject(Int(py=flat._addr) + 4 * offset))
    # Lane x56d (the weights contract every neural family follows, decided
    # 2026-11-09): a weight a kernel reads is read-only. The public
    # parameters are read-only views of the registry (NumPy honours the flag
    # through the array interface); native operations write the registry
    # by address and move the weight epoch. Assign new arrays to change
    # weights (`load_weights`, a checkpoint load).
    py_setattr(result, "_readonly", PythonObject(True))
    py_setattr(result, "shape", py_call(bi.tuple, py_ints(dims)))
    py_setattr(result, "dtype", PythonObject("<f4"))
    py_setattr(result, "ndim", PythonObject(len(dims)))
    py_setattr(result, "size", PythonObject(n))
    py_setattr(result, "nbytes", PythonObject(4 * n))
    py_setattr(result, "itemsize", PythonObject(4))
    py_setattr(result, "order", PythonObject("C"))
    var strides = bi.list()
    var stride = 4
    for reverse in range(len(dims)):  # driver-loop: over tensor stride metadata
        _ = py_call(strides.insert, 0, stride)
        stride *= dims[len(dims) - reverse - 1]
    py_setattr(result, "strides", py_call(bi.tuple, strides))
    return result


def samba_schedule_snapshot(schedule: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = samba_keys()
    if schedule is None:
        return PythonObject(None)
    var bi = native_loaded_module("builtins")
    # The three built-in kinds are fixed (python/mojolearn/_native_samba_bridge.py
    # schedule_classes); compare the exact type against each, no list(dict).
    var classes = py_item(context, nk[].k_schedule_classes)
    var schedule_type = py_type(schedule)
    var kind = String()
    for name in [String("constant"), String("linear"), String("cosine")]:  # driver-loop: over the three built-in schedule kinds
        if schedule_type is classes[name]:
            kind = String(name)
    if kind == "":
        raise Error("NotImplementedError: Samba requires ConstantLR, WarmupLinearLR or WarmupCosineLR; Python schedule callbacks cannot execute in the native runtime")
    var fields = samba_fields(schedule)
    var result = bi.dict()
    py_setitem(result, nk[].k_kind, kind)
    for name in [String("peak_lr"), String("min_lr"), String("warmup_steps"), String("total_steps")]:  # driver-loop: over schedule metadata fields
        py_setitem(result, name, fields[name])
    return result


def samba_schedule_rate(config: PythonObject, step: Int, context: PythonObject) raises -> PythonObject:
    var nk = samba_keys()
    var kind = String(py_item(config, nk[].k_kind))
    var code = 0 if kind == "constant" else (1 if kind == "linear" else 2)
    if kind != "constant" and kind != "linear" and kind != "cosine":
        raise Error("Samba: unsupported saved learning-rate schedule")
    var total = -1 if py_item(config, nk[].k_total_steps) is None else Int(py=py_item(config, nk[].k_total_steps))
    var scalars = Python.list(py_item(config, nk[].k_peak_lr), py_item(config, nk[].k_min_lr))
    var result = py_call(py_item(context, nk[].k_exact_schedule), py_ints([code, Int(py=py_item(config, nk[].k_warmup_steps)), total, step]), scalars)
    if Int(py=result[0]) != 0:
        raise Error("Samba: native exact learning-rate schedule refused saved configuration, status " + String(Int(py=result[0])))
    return result[1]


def samba_schedule_from_state(config: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = samba_keys()
    if config is None:
        return PythonObject(None)
    var bi = native_loaded_module("builtins")
    var kind = String(py_item(config, nk[].k_kind))
    if not py_contains(py_item(context, nk[].k_schedule_classes), kind):
        raise Error("Samba: unsupported saved learning-rate schedule")
    var result = py_call(bi.object.__new__, py_item(context, nk[].k_schedule_classes)[kind])
    # fb6-smoke (2026-11-17): the settings go into the instance dictionary
    # through the optimizer owner's native `schedule_setstate` (admission and
    # ONE rate-block bind), not nine attribute sets through the schedule's
    # __setattr__ (a Python frame and a block rebuild per set). The old
    # `_fast`, `_tail_t`, `_tail_v`, `_tb`, `_tbv` table fields are gone:
    # nothing has read them since the rate block replaced the Python table.
    var state = bi.dict()
    py_setitem(state, nk[].k_peak_lr, PythonObject(Float64(Float32(Float64(py=py_call(bi.float, py_item(config, nk[].k_peak_lr)))))))
    py_setitem(state, nk[].k_min_lr, PythonObject(Float64(Float32(Float64(py=py_call(bi.float, py_item(config, nk[].k_min_lr)))))))
    py_setitem(state, nk[].k_warmup_steps, py_call(bi.int, py_item(config, nk[].k_warmup_steps)))
    py_setitem(state, nk[].k_total_steps, PythonObject(None) if py_item(config, nk[].k_total_steps) is None else py_call(bi.int, py_item(config, nk[].k_total_steps)))
    _ = py_call(py_item(context, nk[].k_schedule_native), "schedule_setstate", result, Python.tuple(state), None, py_item(context, nk[].k_schedule_context))
    return result


def _samba_registry(config: PythonObject) raises -> PythonObject:
    var nk = samba_keys()
    var source = samba_registry(config)
    var result = Python.list()
    var list_type = native_loaded_module("builtins").list
    var offset = 0
    for i in range(len(source)):  # driver-loop: over the ordered parameter registry
        var entry = source[i]
        var shape = entry[1]
        var n = 1
        for j in range(len(shape)):  # driver-loop: over tensor shape metadata
            var d = Int(py=shape[j])
            if d < 1 or n > 2147483647 // d:
                raise Error("Samba: invalid tensor shape")
            n *= d
        if offset > 2147483647 - n:
            raise Error("Samba: flat parameter registry exceeds signed 32-bit indexing")
        var item = Python.dict()
        py_setitem(item, nk[].k_name, entry[0])
        py_setitem(item, nk[].k_shape, py_call(list_type, shape))
        py_setitem(item, nk[].k_offset, offset)
        py_setitem(item, nk[].k_size, n)
        _ = py_call(result.append, item)
        offset += n
    return result


def samba_moment_view(value: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = samba_keys()
    # Native Array has retained storage even on CPython versions before the
    # Python-level buffer protocol. Capture its admitted slots once; foreign
    # writable buffers keep their original exporter and alias writeback.
    if py_type(value) is py_item(py_item(context, nk[].k_neural), nk[].k_array_class):
        var captured = neural_array_snapshot(value, py_item(context, nk[].k_neural))
        var view = captured._mv
        _ = captured
        return view
    return py_call(native_loaded_module("builtins").memoryview, value)


def samba_snapshot(model: PythonObject, copy_arrays: Bool, context: PythonObject) raises -> PythonObject:
    var nk = samba_keys()
    _ = generation_initialize_binding(py_item(py_item(context, nk[].k_neural), nk[].k_generation_classes))
    var bi = native_loaded_module("builtins")
    var fields = samba_fields(model)
    var route = samba_route(model, fields, context)
    var opt = samba_fields(py_item(fields, nk[].k_optimizer))
    var lease = OptimizerOwnerLease(py_item(fields, nk[].k_optimizer))
    var owner = lease.value.downcast_value_ptr[OptimizerOwner]()
    var n = Int(py=py_item(fields, nk[].k_n_total))
    if py_item(opt, nk[].k__m) is None:
        py_setitem(opt, nk[].k__m, generation_array([n]))
    if py_item(opt, nk[].k__v) is None:
        py_setitem(opt, nk[].k__v, generation_array([n]))
    var m = _model_f32(py_item(opt, nk[].k__m), "exp_avg", py_item(context, nk[].k_neural))
    var v = _model_f32(py_item(opt, nk[].k__v), "exp_avg_sq", py_item(context, nk[].k_neural))
    _model_shape(m, [n], "exp_avg")
    _model_shape(v, [n], "exp_avg_sq")
    if owner[].control.handle >= 0 and not owner[].control.host_fresh:
        samba_optimizer_transfer(owner[], Int(py=m._addr), Int(py=v._addr), n, False)
        # A caller may assign writable foreign buffers to the public moment
        # properties. Keep those aliases and write native download bytes back.
        if m is not py_item(opt, nk[].k__m):
            var m_view = samba_moment_view(py_item(opt, nk[].k__m), context)
            py_setitem(m_view, py_call(bi.slice, PythonObject(None)), m._mv)
        if v is not py_item(opt, nk[].k__v):
            var v_view = samba_moment_view(py_item(opt, nk[].k__v), context)
            py_setitem(v_view, py_call(bi.slice, PythonObject(None)), v._mv)
        owner[].control.host_fresh = True
    var flags = _model_native_buffer(py_item(opt, nk[].k_buf_initialized), "i", py_item(context, nk[].k_neural))
    owner[].control.pristine = False
    var result = bi.dict()
    py_setitem(result, nk[].k_schema, "mojolearn.samba-stack-state.v1")
    py_setitem(result, nk[].k_profile, "mojolearn.samba-stack.fp32.v1")
    py_setitem(result, nk[].k_experiment_profile, samba_experiment(route.entries))
    py_setitem(result, nk[].k_numeric_mode, route.mode)
    py_setitem(result, nk[].k_config, samba_config_dict(py_item(fields, nk[].k_config)))
    py_setitem(result, nk[].k_registry, _samba_registry(py_item(fields, nk[].k_config)))
    py_setitem(result, nk[].k_parameters, generation_copy(py_item(fields, nk[].k_flat)) if copy_arrays else py_item(fields, nk[].k_flat))
    py_setitem(result, nk[].k_exp_avg, generation_copy(m) if copy_arrays else py_item(opt, nk[].k__m))
    py_setitem(result, nk[].k_exp_avg_sq, generation_copy(v) if copy_arrays else py_item(opt, nk[].k__v))
    py_setitem(result, nk[].k_buf_initialized, generation_copy(flags) if copy_arrays else py_item(opt, nk[].k_buf_initialized))
    py_setitem(result, nk[].k_t, py_item(opt, nk[].k_t))
    var optimizer = bi.dict()
    py_setitem(optimizer, nk[].k_kind, "adamw")
    py_setitem(optimizer, nk[].k_lr, py_item(opt, nk[].k_lr))
    py_setitem(optimizer, nk[].k_beta1, py_item(opt, nk[].k_betas)[0])
    py_setitem(optimizer, nk[].k_beta2, py_item(opt, nk[].k_betas)[1])
    py_setitem(optimizer, nk[].k_eps, py_item(opt, nk[].k_eps))
    py_setitem(optimizer, nk[].k_weight_decay, py_item(opt, nk[].k_weight_decay))
    py_setitem(optimizer, nk[].k_max_norm, py_item(fields, nk[].k_max_norm))
    py_setitem(optimizer, nk[].k_accumulation_steps, py_item(opt, nk[].k_accumulation_steps))
    py_setitem(result, nk[].k_optimizer, optimizer)
    py_setitem(result, nk[].k_schedule, samba_schedule_snapshot(py_item(opt, nk[].k_lr_schedule), context))
    var generator = samba_fields(py_item(fields, nk[].k_generator))
    var rng = bi.dict()
    py_setitem(rng, nk[].k_seed, py_item(generator, nk[].k_seed))
    py_setitem(rng, nk[].k_counter, py_item(generator, nk[].k_counter))
    py_setitem(result, nk[].k_rng, rng)
    _ = lease
    return result


def samba_validate_state(state: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = samba_keys()
    _ = generation_initialize_binding(py_item(py_item(context, nk[].k_neural), nk[].k_generation_classes))
    var bi = native_loaded_module("builtins")
    # A state has no stack: the process route, admitted for its mode word.
    var process = samba_route_process(py_item(state, nk[].k_numeric_mode), context)
    var entries = process.entries
    if String(py_get(state, nk[].k_schema)) != "mojolearn.samba-stack-state.v1" or String(py_get(state, nk[].k_profile)) != "mojolearn.samba-stack.fp32.v1":
        raise Error("Samba: state schema/profile mismatch")
    if Bool(py=py_get(state, nk[].k_experiment_profile, PythonObject("baseline")) != samba_experiment(entries)):
        raise Error("Samba: experiment numerical profile mismatch")
    var saved_profile = String(py_get(state, nk[].k_numeric_profile, PythonObject("fp32_v1")))
    if saved_profile != "fp32_v1":
        raise Error("Samba: state numeric profile must be fp32_v1, got " + saved_profile)
    var config = samba_config_from_dict(py_item(context, nk[].k_config_class), py_item(state, nk[].k_config))
    var registry = _samba_registry(config)
    if Bool(py=registry != py_item(state, nk[].k_registry)):
        raise Error("Samba: saved parameter registry differs from configuration")
    var n = Int(py=py_item(registry[len(registry) - 1], nk[].k_offset)) + Int(py=py_item(registry[len(registry) - 1], nk[].k_size))
    var result = py_call(bi.dict, state)
    for name in [String("parameters"), String("exp_avg"), String("exp_avg_sq")]:  # driver-loop: over the three floating training-state buffers
        var value = _model_f32(state[name], name, py_item(context, nk[].k_neural))
        _model_shape(value, [n], name)
        py_setitem(result, name, generation_copy(value))
    var flags = py_item(state, nk[].k_buf_initialized)
    var own = py_type(flags) is py_item(py_item(context, nk[].k_neural), nk[].k_array_class)
    var view = flags._mv if own else py_call(bi.memoryview, flags)
    var shape = flags.shape if own else view.shape
    if len(shape) != 1 or Int(py=shape[0]) != len(registry) or (String(view.format) != "i" and String(view.format) != "<i" and String(view.format) != "=i"):
        raise Error("Samba: optimizer flags must be one int32 per registered tensor")
    var copied = generation_array([len(registry)], "i")
    py_setitem(copied._mv, py_call(bi.slice, PythonObject(None)), view)
    py_setitem(result, nk[].k_buf_initialized, copied)
    var step = Int(py=py_item(state, nk[].k_t))
    if step < 0 or step > 2147483647:
        raise Error("Samba: optimizer step exceeds signed 32-bit range")
    var accumulation = Int(py=py_item(py_item(state, nk[].k_optimizer), nk[].k_accumulation_steps))
    if accumulation < 1 or (accumulation & (accumulation - 1)) != 0:
        raise Error("Samba: accumulation_steps must be a positive power of two")
    if String(py_item(py_item(state, nk[].k_optimizer), nk[].k_kind)) != "adamw":
        raise Error("Samba: optimizer kind must be adamw")
    var counter = Int(py=py_item(py_item(state, nk[].k_rng), nk[].k_counter))
    if counter < 0 or counter > 4294967296:
        raise Error("Samba: RNG counter must be in [0, 2^32], including exhaustion")
    if Bool(py=py_item(py_item(state, nk[].k_rng), nk[].k_seed) < 0) or Bool(py=py_item(py_item(state, nk[].k_rng), nk[].k_seed) >= py_call(bi.int, "18446744073709551616")):
        raise Error("Samba: RNG seed must be a 64-bit unsigned integer")
    _ = samba_schedule_from_state(py_item(state, nk[].k_schedule), context)
    py_setitem(result, nk[].k_config, samba_config_dict(config))
    py_setitem(result, nk[].k_registry, registry)
    return result


def samba_publish(model: PythonObject, state: PythonObject, context: PythonObject) raises -> PythonObject:
    return _samba_publish_checked(model, samba_validate_state(state, context), context)


def _samba_publish_checked(model: PythonObject, checked: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = samba_keys()
    # `checked` is a samba_validate_state result (its arrays are already
    # private copies): samba_from_state publishes it without validating and
    # copying the three state arrays a second time (row N166).
    var fields = samba_fields(model)
    if Bool(py=samba_config_dict(py_item(fields, nk[].k_config)) != py_item(checked, nk[].k_config)):
        raise Error("Samba: state config differs from this stack")
    var opt = samba_fields(py_item(fields, nk[].k_optimizer))
    var oc = py_item(checked, nk[].k_optimizer)
    if Bool(py=py_item(opt, nk[].k_accumulation_steps) != py_item(oc, nk[].k_accumulation_steps)):
        raise Error("Samba: saved accumulation_steps differs from this stack")
    var schedule = samba_schedule_from_state(py_item(checked, nk[].k_schedule), context)
    var lease = OptimizerOwnerLease(py_item(fields, nk[].k_optimizer))
    var owner = lease.value.downcast_value_ptr[OptimizerOwner]()
    var flat = py_item(fields, nk[].k_flat)
    var source = py_item(checked, nk[].k_parameters)
    var flat_addr = Int(py=flat._addr)
    if flat is not source:
        var dst = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=flat_addr)
        var src = Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=Int(py=source._addr))
        unsafe_memcpy(dest=dst, src=src, count=Int(py=flat.nbytes))
    # The written registry's own epoch (core/weight_epoch.mojo; hot-path
    # audit 2026-11-13: stamped here from the address this write holds, not
    # by the public door re-reading `flat` per call)
    registry_epoch_bump(flat_addr)
    var first = py_item(checked, nk[].k_exp_avg)
    var second = py_item(checked, nk[].k_exp_avg_sq)
    if owner[].control.handle >= 0:
        samba_optimizer_transfer(owner[], Int(py=first._addr), Int(py=second._addr), Int(py=py_item(fields, nk[].k_n_total)), True)
    # Publish moment owners only after upload succeeds, as public AdamW does.
    py_setitem(opt, nk[].k__m, first)
    py_setitem(opt, nk[].k__v, second)
    owner[].control.host_fresh = True
    owner[].control.pristine = False
    py_setitem(opt, nk[].k_buf_initialized, py_item(checked, nk[].k_buf_initialized))
    py_setitem(opt, nk[].k_t, py_item(checked, nk[].k_t))
    py_setitem(opt, nk[].k_lr, py_item(oc, nk[].k_lr))
    py_setitem(opt, nk[].k_betas, Python.tuple(py_item(oc, nk[].k_beta1), py_item(oc, nk[].k_beta2)))
    py_setitem(opt, nk[].k_eps, py_item(oc, nk[].k_eps))
    py_setitem(opt, nk[].k_weight_decay, py_item(oc, nk[].k_weight_decay))
    py_setitem(opt, nk[].k_lr_schedule, schedule)
    py_setitem(fields, nk[].k_max_norm, py_item(oc, nk[].k_max_norm))
    var generator = samba_fields(py_item(fields, nk[].k_generator))
    py_setitem(generator, nk[].k_seed, py_item(py_item(checked, nk[].k_rng), nk[].k_seed))
    py_setitem(generator, nk[].k_counter, py_item(py_item(checked, nk[].k_rng), nk[].k_counter))
    _ = first
    _ = second
    _ = lease
    return model


def samba_build_blocks(model: PythonObject, context: PythonObject) raises:
    var nk = samba_keys()
    var bi = native_loaded_module("builtins")
    var fields = samba_fields(model)
    var config = py_item(fields, nk[].k_config)
    py_setattr(model, "_blocks", bi.list())
    # The stack's one route (bindings/samba_route_unpack.mojo), resolved here
    # at construction / restoration: its blocks are built on the route's
    # device word and its primitives are the route's image.
    var resolved = samba_route_publish(model, fields, context)
    var route = resolved.blocks
    var nc = py_item(context, nk[].k_neural)
    # Immutable ABI connection occurs once at model construction/restoration.
    _ = neural_descriptor_cached(fields, resolved.entries,
        "_native_primitives", "neural_primitives_api_v1", String(PRIMITIVE_ABI_NAME), True)
    # Samba preserves its training block interfaces on CPU-only installs too.
    # The execution route still selects the installed native CPU kernels; only
    # the Python-facing class differs from checkpoint inference construction.
    var block_context = py_call(bi.dict, nc)
    var classes = py_call(bi.dict, py_item(nc, nk[].k_block_classes))
    py_setitem(classes, route, py_item(py_item(nc, nk[].k_block_classes), nk[].k_gpu))
    py_setitem(block_context, nk[].k_block_classes, classes)
    for i in range(len(config.layers)):  # driver-loop: over configured model layers
        var layer_kind = String(config.layers[i])
        var kind = "mamba3" if layer_kind == "mamba3" else "transformer"
        var weights = bi.dict()
        var names = samba_block_shapes(config, layer_kind)
        for j in range(len(names)):  # driver-loop: over the layer's named parameter views
            py_setitem(weights, names[j][0], model.arrays["layers." + String(i) + "." + String(names[j][0])])
        var plan = py_call(bi.object.__new__, py_item(py_item(nc, nk[].k_config_context), nk[].k_plan_class))
        py_setattr(plan, "kind", PythonObject(kind))
        py_setattr(plan, "d_model", config.d_model)
        py_setattr(plan, "block_options", bi.dict())
        py_setattr(plan, "block_kwargs", bi.dict())
        if kind == "transformer":
            py_setattr(plan, "n_heads", config.n_heads)
            py_setitem(plan.block_options, nk[].k_n_kv_heads, config.n_kv_heads)
            py_setitem(plan.block_options, nk[].k_head_dim, config.head_dim)
            # Samba's attention blocks use TransformerBlock's 1e-6 default.
            # The checkpoint model interface defaults to 1e-5 instead, and
            # SambaConfig.norm_eps controls only the final stack normalization.
            py_setitem(plan.block_options, nk[].k_norm_eps, PythonObject(1e-6))
        var bindings = _model_bindings(route, kind, nc, model.numeric_mode)
        var block = _model_block(weights, plan, PythonObject("fp32_v1"), route, bindings, block_context)
        var block_fields = samba_fields(block)
        py_setitem(block_fields, nk[].k_numeric_mode, model.numeric_mode)
        if kind == "mamba3":
            py_setitem(block_fields, nk[].k__discard_reports, True)
        _ = py_call(model._blocks.append, block)


def samba_from_state(state: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = samba_keys()
    return samba_from_state(state, py_item(context, nk[].k_model_class), PythonObject(None), context)


def samba_from_state(state: PythonObject, cls: PythonObject, device: PythonObject, context: PythonObject) raises -> PythonObject:
    """A stack of class `cls` from a state, on `device` (None: this
    installation's device). The device is refused before the state is
    validated; the stack's route is resolved once, by `samba_build_blocks`."""
    var nk = samba_keys()
    samba_device_admit(device, context)
    var checked = samba_validate_state(state, context)
    var bi = native_loaded_module("builtins")
    var model = py_call(bi.object.__new__, cls)
    var config = samba_config_from_dict(py_item(context, nk[].k_config_class), py_item(checked, nk[].k_config))
    py_setattr(model, "config", config)
    py_setattr(model, "device", device)
    py_setattr(model, "numeric_mode", py_item(checked, nk[].k_numeric_mode))
    py_setattr(model, "flat", py_item(checked, nk[].k_parameters))
    # lane x56d: the registry a kernel reads is read-only (see samba_view)
    _ = fitted_freeze_any(model.flat)
    py_setattr(model, "n_total", model.flat.size)
    py_setattr(model, "names", bi.list())
    py_setattr(model, "shapes", bi.dict())
    py_setattr(model, "offsets", bi.list())
    py_setattr(model, "arrays", bi.dict())
    var registry = py_item(checked, nk[].k_registry)
    var params = bi.list()
    for i in range(len(registry)):  # driver-loop: over registry views retaining the flat parameter owner
        var item = registry[i]
        var dims = List[Int]()
        for j in range(len(py_item(item, nk[].k_shape))):  # driver-loop: over tensor shape metadata
            dims.append(Int(py=py_item(item, nk[].k_shape)[j]))
        _ = py_call(model.names.append, py_item(item, nk[].k_name))
        py_setitem(model.shapes, py_item(item, nk[].k_name), py_call(bi.tuple, py_item(item, nk[].k_shape)))
        _ = py_call(model.offsets.append, py_item(item, nk[].k_offset))
        var value = samba_view(model.flat, Int(py=py_item(item, nk[].k_offset)), dims, context)
        py_setitem(model.arrays, py_item(item, nk[].k_name), value)
        _ = py_call(params.append, value)
    _ = py_call(model.offsets.append, model.n_total)
    var generator = py_call(bi.object.__new__, py_item(context, nk[].k_generator_class))
    var gen_fields = samba_fields(generator)
    py_setitem(gen_fields, nk[].k_seed, py_item(py_item(checked, nk[].k_rng), nk[].k_seed))
    py_setitem(gen_fields, nk[].k_counter, py_item(py_item(checked, nk[].k_rng), nk[].k_counter))
    py_setitem(gen_fields, nk[].k_numeric_mode, py_item(checked, nk[].k_numeric_mode))
    py_setattr(model, "generator", generator)
    py_setattr(model, "max_norm", py_item(py_item(checked, nk[].k_optimizer), nk[].k_max_norm))
    var optimizer = py_call(bi.object.__new__, py_item(context, nk[].k_optimizer_class))
    var opt = samba_fields(optimizer)
    py_setitem(opt, nk[].k__resident_wanted, PythonObject(None))
    py_setitem(opt, nk[].k__native_optimizer_owner, optimizer_owner_new(optimizer, 2, False))
    py_setitem(opt, nk[].k_params, params)
    py_setitem(opt, nk[].k_n_total, model.n_total)
    py_setitem(opt, nk[].k_accumulation_steps, py_item(py_item(checked, nk[].k_optimizer), nk[].k_accumulation_steps))
    py_setitem(opt, nk[].k_numeric_mode, py_item(checked, nk[].k_numeric_mode))
    py_setitem(opt, nk[].k_maximize, False)
    py_setitem(opt, nk[].k_packed_, PythonObject(None))
    py_setitem(opt, nk[].k_total_norm_, PythonObject(None))
    py_setitem(opt, nk[].k_clip_coef_, PythonObject(None))
    var offsets = generation_array([len(registry) + 1], "i")
    var offset_ptr = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=offsets._addr))
    for i in range(len(registry) + 1):  # driver-loop: over registry offset metadata
        offset_ptr[i] = Int32(Int(py=model.offsets[i]))
    py_setitem(opt, nk[].k_offsets, offsets)
    py_setattr(model, "optimizer", optimizer)
    samba_build_blocks(model, context)
    py_setattr(model, "last_", PythonObject(None))
    return _samba_publish_checked(model, checked, context)


def samba_apply_gradients_from_state(state: PythonObject, gradients: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = samba_keys()
    var bi = native_loaded_module("builtins")
    var device_plan = samba_optimizer_device_plan()
    # A state has no stack: the process route, resolved once for the call.
    var route = samba_route_process(py_item(state, nk[].k_numeric_mode), context)
    var entries = route.entries
    if Int(py=py_call(bi.int, device_plan.setting)) > 1:
        for name in [String("optimizer_parallel_available"), String("accumulate_parallel_available"), String("clip_parallel_available")]:  # driver-loop: over cooperative native capability metadata
            if not Bool(py=py_call(bi.hasattr, entries, name)) or Int(py=_native_function(entries, name)()) != 1:
                raise Error("RuntimeError: native Samba cooperative update is unavailable: " + name)
    var result = samba_validate_state(state, context)
    var registry = py_item(result, nk[].k_registry)
    if len(gradients) != len(registry):
        raise Error("Samba: gradients must match the ordered parameter registry")
    var step = Int(py=py_item(result, nk[].k_t))
    if step >= 2147483647:
        raise Error("Samba: optimizer step exceeds signed 32-bit range")
    var config = py_call(bi.dict, py_item(result, nk[].k_optimizer))
    if py_item(result, nk[].k_schedule) is not None:
        py_setitem(config, nk[].k_lr, samba_schedule_rate(py_item(result, nk[].k_schedule), step + 1, context))
    var max_norm = 0.0 if py_item(config, nk[].k_max_norm) is None else Float64(py=py_item(config, nk[].k_max_norm))
    if py_item(config, nk[].k_max_norm) is not None and not (max_norm > 0.0):
        raise Error("Samba: max_norm must be positive or None")
    var n = Int(py=py_item(result, nk[].k_parameters).size)
    var flat_gradient = generation_array([n])
    var offsets = generation_array([len(registry) + 1], "i")
    var gp = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=flat_gradient._addr))
    var op = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=offsets._addr))
    for i in range(len(registry)):  # driver-loop: over registry-ordered gradient buffers
        var item = registry[i]
        var dims = List[Int]()
        for axis in range(len(py_item(item, nk[].k_shape))):  # driver-loop: over tensor shape metadata
            dims.append(Int(py=py_item(item, nk[].k_shape)[axis]))
        var value = _model_f32(gradients[i], "gradient", py_item(context, nk[].k_neural))
        _model_shape(value, dims, "gradient")
        var offset = Int(py=py_item(item, nk[].k_offset))
        op[i] = Int32(offset)
        var src = Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=Int(py=value._addr))
        unsafe_memcpy(dest=gp + offset, src=src, count=Int(py=py_item(item, nk[].k_size)))
        _ = value^
    op[len(registry)] = Int32(n)
    # The per-call AdamW step (primitive op 25, the former `optimizer_step`
    # entry's words): ints [J, kind, t, nesterov, maximize], floats [lr, beta1,
    # beta2, eps, weight_decay, momentum, dampening, max_norm].
    var ints: List[Int] = [len(registry), 2, step + 1, 0, 0]
    var floats = List[Float64]()
    for key in [String("lr"), String("beta1"), String("beta2"), String("eps"), String("weight_decay")]:  # driver-loop: over optimizer scalar launch metadata
        floats.append(Float64(py=config[key]))
    floats.append(0.0)
    floats.append(0.0)
    floats.append(max_norm)
    var info = generation_array([3])
    var addresses: List[Int] = [Int(py=py_item(result, nk[].k_parameters)._addr), Int(py=flat_gradient._addr),
        Int(py=py_item(result, nk[].k_exp_avg)._addr), Int(py=py_item(result, nk[].k_exp_avg_sq)._addr), Int(py=offsets._addr),
        Int(py=py_item(result, nk[].k_buf_initialized)._addr), Int(py=info._addr)]
    var written = samba_primitive(route.primitives(),
        SAMBA_PRIM_OPTIMIZER_STEP, addresses, ints, floats)
    _ = info  # keep the Python owner alive through the native call above
    _ = flat_gradient
    _ = offsets
    if written != n:
        raise Error("RuntimeError: Samba optimizer returned an incomplete update")
    py_setitem(result, nk[].k_optimizer, config)
    py_setitem(result, nk[].k_t, PythonObject(step + 1))
    py_setitem(result, nk[].k_rng, py_call(bi.dict, py_item(result, nk[].k_rng)))
    _ = flat_gradient^
    _ = offsets^
    _ = info^
    return result
