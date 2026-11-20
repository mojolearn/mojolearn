# SPDX-License-Identifier: Apache-2.0
"""Schedule UNPACK/PACK; typed plans own validation and native execution."""
from std.memory import bitcast
from std.python import Python, PythonObject
from std.python._cpython import GILReleased, PyObjectPtr
from bindings.py_cpython_table import cpython_table
from bindings.optimizer_real_unpack import optimizer_real
from bindings.optimizer_public_words_unpack import stash_python_error
from bindings.py_balanced import py_call, py_type, py_fields, py_setitem
from bindings.optimizer_admission_door import opt_scalar, opt_value, opt_schedule, opt_exact
from bindings.optimizer_public_door import schedule_spec_admitted
from bindings.lr_schedule_block_door import schedule_bind_block
from bindings.optimizer_scalar_boundary_unpack import optimizer_integer, optimizer_truth
from bindings.native_array_storage_door import native_array_unfilled
from bindings.native_buffer_owner_pack import native_storage_allocate
from bindings.native_buffer_address_unpack import native_buffer_address
from bindings.py_lists import py_list_f32
from sequence.lr_table import LrSpec, F32P
from training.optimizer_schedule_plan import (
    optimizer_warmup_schedule, optimizer_decay_schedule, optimizer_cycle_schedule,
    optimizer_schedule_config_kind, optimizer_schedule_config_name, optimizer_schedule_evaluation,
)
from bindings.optimizer_keys_unpack import optimizer_keys
from bindings.optimizer_keys_unpack import (
    OPTIMIZER_OPN_SCHEDULE_AT,
    OPTIMIZER_OPN_SCHEDULE_BITS,
    OPTIMIZER_OPN_SCHEDULE_CONFIG,
    OPTIMIZER_OPN_SCHEDULE_CONSTRUCT,
    OPTIMIZER_OPN_SCHEDULE_DELATTR,
    OPTIMIZER_OPN_SCHEDULE_FROM_CONFIG,
    OPTIMIZER_OPN_SCHEDULE_GETSTATE,
    OPTIMIZER_OPN_SCHEDULE_LIST,
    OPTIMIZER_OPN_SCHEDULE_SETATTR,
    OPTIMIZER_OPN_SCHEDULE_SETSTATE,
    OPTIMIZER_OPN_SCHEDULE_SPEC,
    OPTIMIZER_OPN_SCHEDULE_VALUES,
)
from bindings.py_balanced import py_item


def schedule_kind(owner: PythonObject, context: PythonObject) raises -> Int:
    var nk = optimizer_keys()
    var classes = py_item(context, nk[].k_schedules)
    for i in range(len(classes)):  # driver-loop: over built-in schedule class identities
        if py_type(owner) is classes[i]:
            return i
    raise Error("TypeError: schedule requires a built-in native configuration; Python callbacks cannot execute")


def _sched_real(value: PythonObject, label: String) raises -> Float64:
    # Keep the public no-text-parsing conversion, after exact scalar admission.
    var real: Float64
    try:
        real = Float64(py=value)
    except:
        raise Error("TypeError: " + label + " must be a real number")
    # an integer above 2^53 rounds once to float32 (bindings/optimizer_real_unpack.mojo)
    return optimizer_real(value, real)


def _sched_int(value: PythonObject, label: String, context: PythonObject) raises -> Int:
    # PyNumber_Index is the cached native index slot, with no operator lookup.
    # The caller has admitted exact scalar types before this hook-capable API.
    var nk = optimizer_keys()
    if py_type(value) is py_item(context, nk[].k_scalar_types)[1]:
        raise Error("TypeError: " + label + " must be an integer")
    var table = cpython_table()
    var pointer = table[].PyNumber_Index(value._obj_ptr)
    if not pointer:
        table[].PyErr_Clear()
        raise Error("TypeError: " + label + " must be an integer")
    var number = PythonObject(from_owned=pointer)
    try:
        return Int(py=number)
    except:
        raise Error("TypeError: " + label + " must be an integer")


def _warmup_unpack(kind: Int, peak_value: PythonObject, warmup_value: PythonObject,
    total_value: PythonObject, low_value: PythonObject, label: String,
    context: PythonObject) raises -> LrSpec:
    var peak = _sched_real(peak_value, label + "peak_lr")
    var low = 0.0 if kind == 0 else _sched_real(low_value, label + "min_lr")
    var warmup = _sched_int(warmup_value, label + "warmup_steps", context)
    var has_total = kind != 0 and total_value is not None
    var total = _sched_int(total_value, label + "total_steps", context) if has_total else -1
    return optimizer_warmup_schedule(kind, peak, low, warmup, total, has_total)


def _warmup_pack(owner: PythonObject, spec: LrSpec) raises:
    var fields = py_fields(owner)
    var nk = optimizer_keys()
    py_setitem(fields, nk[].k_peak_lr, PythonObject(spec.f0))
    py_setitem(fields, nk[].k_min_lr, PythonObject(spec.f1))
    py_setitem(fields, nk[].k_warmup_steps, PythonObject(spec.a))
    py_setitem(fields, nk[].k_total_steps, PythonObject(None) if spec.b < 0 else PythonObject(spec.b))


def schedule_construct(owner: PythonObject, values: PythonObject, context: PythonObject) raises -> PythonObject:
    opt_scalar(values, context)
    var kind = schedule_kind(owner, context)
    var fields = py_fields(owner)
    var nk = optimizer_keys()
    var label = String(py=py_call(py_item(context, nk[].k_type_name), py_type(owner))) + ": "
    if kind < 3:
        var spec = _warmup_unpack(kind, values[0], values[1], values[2], values[3], label, context)
        _warmup_pack(owner, spec)
    elif kind < 5:
        var base = _sched_real(values[0], label + "base_lr")
        var step = 1 if kind == 4 else _sched_int(values[1], label + "step_size", context)
        var gamma = _sched_real(values[2], label + "gamma")
        var spec = optimizer_decay_schedule(kind, base, step, gamma, label)
        py_setitem(fields, nk[].k_base_lr, PythonObject(spec.f0))
        py_setitem(fields, nk[].k_gamma, PythonObject(spec.f1))
        if kind == 3:
            py_setitem(fields, nk[].k_step_size, PythonObject(spec.a))
            py_setitem(fields, nk[].k__step, PythonObject(spec.a))
    else:
        var maximum = _sched_real(values[0], label + "max_lr")
        var total = _sched_int(values[1], label + "total_steps", context)
        var pct = _sched_real(values[2], label + "pct_start")
        var anneal = String(values[3]) if py_type(values[3]) is py_item(context, nk[].k_scalar_types)[5] else String()
        var divide = _sched_real(values[4], label + "div_factor")
        var final_divide = _sched_real(values[5], label + "final_div_factor")
        var spec = optimizer_cycle_schedule(maximum, total, pct, anneal, divide, final_divide)
        py_setitem(fields, nk[].k_max_lr, PythonObject(spec.f0))
        py_setitem(fields, nk[].k_total_steps, PythonObject(spec.c))
        py_setitem(fields, nk[].k_pct_start, PythonObject(spec.f3))
        py_setitem(fields, nk[].k_anneal_strategy, values[3])
        py_setitem(fields, nk[].k_div_factor, PythonObject(spec.f1))
        py_setitem(fields, nk[].k_final_div_factor, PythonObject(spec.f2))
        py_setitem(fields, nk[].k_three_phase, PythonObject(optimizer_truth(values[6], context)))
    # lane fb2-lr (2026-11-17): `lr_at` / `bits_at` are the native block's
    # METH_VARARGS | METH_KEYWORDS methods, bound here once (bindings/lr_schedule_block_door.mojo).
    schedule_bind_block(owner, kind, context)
    return PythonObject(None)


def _schedule_arguments(owner: PythonObject, kind: Int) raises -> PythonObject:
    """The constructor arguments the stored settings stand for, in
    `schedule_construct`'s order for `kind`."""
    var fields = py_fields(owner)
    var nk = optimizer_keys()
    if kind < 3:
        return Python.tuple(py_item(fields, nk[].k_peak_lr), py_item(fields, nk[].k_warmup_steps), py_item(fields, nk[].k_total_steps), py_item(fields, nk[].k_min_lr))
    if kind < 5:
        return Python.tuple(py_item(fields, nk[].k_base_lr), py_item(fields, nk[].k_step_size) if kind == 3 else PythonObject(1), py_item(fields, nk[].k_gamma))
    return Python.tuple(py_item(fields, nk[].k_max_lr), py_item(fields, nk[].k_total_steps), py_item(fields, nk[].k_pct_start), py_item(fields, nk[].k_anneal_strategy),
                        py_item(fields, nk[].k_div_factor), py_item(fields, nk[].k_final_div_factor), py_item(fields, nk[].k_three_phase))


def _schedule_rebind(owner: PythonObject, name: PythonObject, context: PythonObject) raises:
    """After an edit or a deletion of attribute `name`: rebuild the native
    rate block from the stored settings, or drop the bound block methods when
    the admission refuses them (the class methods then re-read and refuse at
    the call). An edit of `lr_at` / `bits_at` themselves rebuilds nothing."""
    var word = String(py=name)
    if word == "lr_at" or word == "bits_at":
        return
    try:
        opt_schedule(owner, context)
        # fb6-smoke (2026-11-17): the edited settings go through the
        # constructor's own unpack and pack (float32 rates, validated ints,
        # StepLR's `_step`), then one bind, so an edited schedule is the
        # schedule a fresh constructor call builds on every reader (lr_at,
        # bits_at, lr_values, schedule_at, config, pickle). The raw stored
        # double gave lr_values' warmup steps one ulp off a fresh schedule.
        _ = schedule_construct(owner, _schedule_arguments(owner, schedule_kind(owner, context)), context)
    except:
        var edited = py_fields(owner)
        var nk = optimizer_keys()
        _ = edited.pop(nk[].k_lr_at, PythonObject(None))
        _ = edited.pop(nk[].k_bits_at, PythonObject(None))


def schedule_run(operation: Int, owner: PythonObject, values: PythonObject, context: PythonObject) raises -> PythonObject:
    """A schedule operation by its code (bindings/optimizer_keys_unpack.mojo).
    A code that is no schedule operation (an unknown `schedule_*` name, -1)
    ends where an unknown name ended before: the rate at the given step."""
    if operation == OPTIMIZER_OPN_SCHEDULE_CONSTRUCT:
        return schedule_construct(owner, values, context)
    var nk = optimizer_keys()
    if operation == OPTIMIZER_OPN_SCHEDULE_FROM_CONFIG:
        var cfg = values[0]
        opt_value(cfg, context)
        if not opt_exact(py_type(cfg), py_item(context, nk[].k_mapping_types)):
            raise Error("TypeError: schedule configuration requires a native dictionary")
        var name = py_item(cfg, nk[].k_kind)
        opt_scalar(name, context)
        var kind = optimizer_schedule_config_kind(String(name) if py_type(name) is py_item(context, nk[].k_scalar_types)[5] else String())
        var peak = py_item(cfg, nk[].k_peak_lr)
        var warmup = py_item(cfg, nk[].k_warmup_steps)
        var total = py_item(cfg, nk[].k_total_steps)
        var low = py_item(cfg, nk[].k_min_lr)
        # Admit the four external values without a new Python tuple envelope.
        opt_scalar(peak, context, 1)
        opt_scalar(warmup, context, 1)
        opt_scalar(total, context, 1)
        opt_scalar(low, context, 1)
        var cls = py_item(context, nk[].k_schedules)[kind]
        var label = String(py=py_call(py_item(context, nk[].k_type_name), cls)) + ": "
        var spec = _warmup_unpack(kind, peak, warmup, total, low, label, context)
        var result = native_storage_allocate(cls)
        _warmup_pack(result, spec)
        schedule_bind_block(result, kind, context)
        return result
    if operation == OPTIMIZER_OPN_SCHEDULE_GETSTATE:
        # Pickle and copy state: the stored settings only. The bound block
        # methods are rebuilt by `schedule_setstate`, never serialized.
        opt_schedule(owner, context)
        var stored = py_fields(owner)
        var copied = cpython_table()[].PyDict_Copy(stored._obj_ptr)
        if not copied:
            raise Error("RuntimeError: schedule state copy failed")
        var state = PythonObject(from_owned=copied)
        _ = state.pop(nk[].k_lr_at, PythonObject(None))
        _ = state.pop(nk[].k_bits_at, PythonObject(None))
        return state
    if operation == OPTIMIZER_OPN_SCHEDULE_SETATTR:
        # `sched.<name> = value` (lane fb2-lr, 2026-11-17): the instance
        # dictionary takes the value as object.__setattr__ stored it, then the
        # rate block is rebuilt from the edited settings, so `lr_at` follows
        # every edit as the per-call re-read did. Settings the admission
        # refuses drop the bound block methods instead; the class methods
        # then re-read and refuse at the call, as before. One native call per
        # edit; nothing is added to `lr_at`.
        var name = values[0]
        if py_type(name) is not py_item(context, nk[].k_scalar_types)[5]:
            raise Error("TypeError: attribute name must be string, not '"
                        + String(py=py_call(py_item(context, nk[].k_type_name), py_type(name))) + "'")
        var edited = py_fields(owner)
        py_setitem(edited, name, values[1])
        _schedule_rebind(owner, name, context)
        return PythonObject(None)
    if operation == OPTIMIZER_OPN_SCHEDULE_DELATTR:
        # `del sched.<name>` (lane fb4-audit, 2026-11-17): object.__delattr__'s
        # own deletion and refusal (PyObject_GenericSetAttr with no value),
        # then the same rebuild as an edit. Without it a deleted setting left
        # the old bound block answering `lr_at` from the settings it no longer
        # has; now the class methods re-read and refuse at the call, as the
        # per-call re-read did. One native call per deletion.
        var name = values[0]
        if py_type(name) is not py_item(context, nk[].k_scalar_types)[5]:
            raise Error("TypeError: attribute name must be string, not '"
                        + String(py=py_call(py_item(context, nk[].k_type_name), py_type(name))) + "'")
        if cpython_table()[].PyObject_GenericSetAttr(owner._obj_ptr, name._obj_ptr, PyObjectPtr()) != 0:
            var failure = stash_python_error()
            raise failure^
        _schedule_rebind(owner, name, context)
        return PythonObject(None)
    if operation == OPTIMIZER_OPN_SCHEDULE_SETSTATE:
        var restored = values[0]
        if py_type(restored) is not py_item(context, nk[].k_container_types)[2]:
            raise Error("TypeError: schedule state requires a native dictionary")
        var target = py_fields(owner)
        _ = target.update(restored)
        opt_schedule(owner, context)
        schedule_bind_block(owner, schedule_kind(owner, context), context)
        return PythonObject(None)
    opt_schedule(owner, context)
    var kind = schedule_kind(owner, context)
    var spec = schedule_spec_admitted(owner)
    if operation == OPTIMIZER_OPN_SCHEDULE_SPEC:
        # These containers are the documented public _lr_spec result only.
        return Python.tuple(Python.list(spec.kind, spec.a, spec.b, spec.c),
                            Python.list(spec.f0, spec.f1, spec.f2, spec.f3))
    if operation == OPTIMIZER_OPN_SCHEDULE_CONFIG:
        var fields = py_fields(owner)
        return Python.dict(kind=optimizer_schedule_config_name(kind), peak_lr=py_item(fields, nk[].k_peak_lr), min_lr=py_item(fields, nk[].k_min_lr),
                           warmup_steps=py_item(fields, nk[].k_warmup_steps), total_steps=py_item(fields, nk[].k_total_steps))
    opt_scalar(values, context)
    var requested = optimizer_integer(values[0], context)
    var series = operation == OPTIMIZER_OPN_SCHEDULE_VALUES or operation == OPTIMIZER_OPN_SCHEDULE_LIST
    var plan = optimizer_schedule_evaluation(kind, spec, requested, series)
    if series:
        if operation == OPTIMIZER_OPN_SCHEDULE_LIST:
            var rates = List[Float32](length=plan.count, fill=0)
            with GILReleased(Python()):
                plan.fill(rates.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]())
            return py_list_f32(rates)
        # lane fix-output-zero-fill (2026-11-14): unfilled; plan.fill writes all plan.count rates.
        var result = native_array_unfilled("f", plan.count)
        # Fresh unpublished native storage has no foreign resize owner; pin
        # the array locally until the unchanged native fill returns.
        var address = native_buffer_address(result)
        with GILReleased(Python()):
            plan.fill(F32P(unsafe_from_address=address))
        _ = result
        return result
    var stored = List[Float32](length=1, fill=0)
    with GILReleased(Python()):
        plan.fill(stored.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]())
    if operation == OPTIMIZER_OPN_SCHEDULE_BITS:
        return PythonObject(Int(bitcast[DType.uint32](stored[0])))
    return PythonObject(Float64(stored[0]))
