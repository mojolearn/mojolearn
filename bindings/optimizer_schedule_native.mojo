# SPDX-License-Identifier: Apache-2.0
"""Schedule UNPACK/PACK; typed plans own validation and native execution."""
from std.memory import bitcast
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.cpython_table import cpython_table
from bindings.py_balanced import py_call, py_type, py_fields, py_setitem
from bindings.optimizer_admission import opt_scalar, opt_value, opt_schedule, opt_exact
from bindings.optimizer_public_door import schedule_spec_admitted
from bindings.optimizer_scalar_boundary import optimizer_integer, optimizer_truth
from bindings.native_array_storage import native_array_zeros
from bindings.native_buffer_owner import native_storage_allocate
from bindings.native_buffer_address import native_buffer_address
from bindings.py_lists import py_list_f32
from sequence.lr_table import LrSpec, F32P
from training.optimizer_schedule_plan import (
    optimizer_warmup_schedule, optimizer_decay_schedule, optimizer_cycle_schedule,
    optimizer_schedule_config_kind, optimizer_schedule_config_name, optimizer_schedule_evaluation,
)


def schedule_kind(owner: PythonObject, context: PythonObject) raises -> Int:
    var classes = context["schedules"]
    for i in range(len(classes)):  # driver-loop: over built-in schedule class identities
        if py_type(owner) is classes[i]:
            return i
    raise Error("TypeError: schedule requires a built-in native configuration; Python callbacks cannot execute")


def _sched_real(value: PythonObject, label: String) raises -> Float64:
    # Keep the public no-text-parsing conversion, after exact scalar admission.
    try:
        return Float64(py=value)
    except:
        raise Error("TypeError: " + label + " must be a real number")


def _sched_int(value: PythonObject, label: String, context: PythonObject) raises -> Int:
    # PyNumber_Index is the cached native index slot, with no operator lookup.
    # The caller has admitted exact scalar types before this hook-capable API.
    if py_type(value) is context["scalar_types"][1]:
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
    py_setitem(fields, "peak_lr", PythonObject(spec.f0))
    py_setitem(fields, "min_lr", PythonObject(spec.f1))
    py_setitem(fields, "warmup_steps", PythonObject(spec.a))
    py_setitem(fields, "total_steps", PythonObject(None) if spec.b < 0 else PythonObject(spec.b))


def schedule_construct(owner: PythonObject, values: PythonObject, context: PythonObject) raises -> PythonObject:
    opt_scalar(values, context)
    var kind = schedule_kind(owner, context)
    var fields = py_fields(owner)
    var label = String(py=py_call(context["type_name"], py_type(owner))) + ": "
    if kind < 3:
        var spec = _warmup_unpack(kind, values[0], values[1], values[2], values[3], label, context)
        _warmup_pack(owner, spec)
    elif kind < 5:
        var base = _sched_real(values[0], label + "base_lr")
        var step = 1 if kind == 4 else _sched_int(values[1], label + "step_size", context)
        var gamma = _sched_real(values[2], label + "gamma")
        var spec = optimizer_decay_schedule(kind, base, step, gamma, label)
        py_setitem(fields, "base_lr", PythonObject(spec.f0))
        py_setitem(fields, "gamma", PythonObject(spec.f1))
        if kind == 3:
            py_setitem(fields, "step_size", PythonObject(spec.a))
            py_setitem(fields, "_step", PythonObject(spec.a))
    else:
        var maximum = _sched_real(values[0], label + "max_lr")
        var total = _sched_int(values[1], label + "total_steps", context)
        var pct = _sched_real(values[2], label + "pct_start")
        var anneal = String(values[3]) if py_type(values[3]) is context["scalar_types"][5] else String()
        var divide = _sched_real(values[4], label + "div_factor")
        var final_divide = _sched_real(values[5], label + "final_div_factor")
        var spec = optimizer_cycle_schedule(maximum, total, pct, anneal, divide, final_divide)
        py_setitem(fields, "max_lr", PythonObject(spec.f0))
        py_setitem(fields, "total_steps", PythonObject(spec.c))
        py_setitem(fields, "pct_start", PythonObject(spec.f3))
        py_setitem(fields, "anneal_strategy", values[3])
        py_setitem(fields, "div_factor", PythonObject(spec.f1))
        py_setitem(fields, "final_div_factor", PythonObject(spec.f2))
        py_setitem(fields, "three_phase", PythonObject(optimizer_truth(values[6], context)))
    return PythonObject(None)


def schedule_operation(operation: String, owner: PythonObject, values: PythonObject, context: PythonObject) raises -> PythonObject:
    if operation == "schedule_construct":
        return schedule_construct(owner, values, context)
    if operation == "schedule_from_config":
        var cfg = values[0]
        opt_value(cfg, context)
        if not opt_exact(py_type(cfg), context["mapping_types"]):
            raise Error("TypeError: schedule configuration requires a native dictionary")
        var name = cfg["kind"]
        opt_scalar(name, context)
        var kind = optimizer_schedule_config_kind(String(name) if py_type(name) is context["scalar_types"][5] else String())
        var peak = cfg["peak_lr"]
        var warmup = cfg["warmup_steps"]
        var total = cfg["total_steps"]
        var low = cfg["min_lr"]
        # Admit the four external values without a new Python tuple envelope.
        opt_scalar(peak, context, 1)
        opt_scalar(warmup, context, 1)
        opt_scalar(total, context, 1)
        opt_scalar(low, context, 1)
        var cls = context["schedules"][kind]
        var label = String(py=py_call(context["type_name"], cls)) + ": "
        var spec = _warmup_unpack(kind, peak, warmup, total, low, label, context)
        var result = native_storage_allocate(cls)
        _warmup_pack(result, spec)
        return result
    opt_schedule(owner, context)
    var kind = schedule_kind(owner, context)
    var spec = schedule_spec_admitted(owner)
    if operation == "schedule_spec":
        # These containers are the documented public _lr_spec result only.
        return Python.tuple(Python.list(spec.kind, spec.a, spec.b, spec.c),
                            Python.list(spec.f0, spec.f1, spec.f2, spec.f3))
    if operation == "schedule_config":
        var fields = py_fields(owner)
        return Python.dict(kind=optimizer_schedule_config_name(kind), peak_lr=fields["peak_lr"], min_lr=fields["min_lr"],
                           warmup_steps=fields["warmup_steps"], total_steps=fields["total_steps"])
    opt_scalar(values, context)
    var requested = optimizer_integer(values[0], context)
    var series = operation == "schedule_values" or operation == "schedule_list"
    var plan = optimizer_schedule_evaluation(kind, spec, requested, series)
    if series:
        if operation == "schedule_list":
            var rates = List[Float32](length=plan.count, fill=0)
            with GILReleased(Python()):
                plan.fill(rates.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]())
            return py_list_f32(rates)
        var result = native_array_zeros("f", plan.count)
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
    if operation == "schedule_bits":
        return PythonObject(Int(bitcast[DType.uint32](stored[0])))
    return PythonObject(Float64(stored[0]))
