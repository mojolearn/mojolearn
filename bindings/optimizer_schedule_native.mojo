# SPDX-License-Identifier: Apache-2.0
# Built-in public schedule data and scalar evaluation, without Python accessors.
from std.memory import bitcast
from std.math import isfinite
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_fields, py_setitem
from bindings.optimizer_admission import opt_scalar, opt_value, opt_schedule, opt_exact
from bindings.optimizer_public_door import schedule_spec


def schedule_kind(owner: PythonObject, context: PythonObject) raises -> Int:
    var classes = context["schedules"]
    for i in range(len(classes)):  # driver-loop: over built-in schedule class identities
        if py_type(owner) is classes[i]:
            return i
    raise Error("TypeError: schedule requires a built-in native configuration; Python callbacks cannot execute")


def schedule_construct(owner: PythonObject, values: PythonObject, context: PythonObject) raises -> PythonObject:
    opt_scalar(values, context)
    var kind = schedule_kind(owner, context)
    var fields = py_fields(owner)
    var b = Python.import_module("builtins")
    if kind < 3:
        var peak = Float64(Float32(Float64(py=py_call(b.float, values[0]))))
        var low = 0.0 if kind == 0 else Float64(Float32(Float64(py=py_call(b.float, values[3]))))
        var warmup = Int(py=py_call(b.int, values[1]))
        var total = PythonObject(None) if kind == 0 or values[2] is None else py_call(b.int, values[2])
        if not isfinite(peak) or peak < 0:
            raise Error("ValueError: peak_lr must be finite and >= 0")
        if not isfinite(low) or low < 0:
            raise Error("ValueError: min_lr must be finite and >= 0")
        if warmup < 0:
            raise Error("ValueError: warmup_steps must be >= 0")
        if total is not None and Int(py=total) < max(1, warmup):
            raise Error("ValueError: total_steps must be >= max(1, warmup_steps)")
        py_setitem(fields, "peak_lr", PythonObject(peak))
        py_setitem(fields, "min_lr", PythonObject(low))
        py_setitem(fields, "warmup_steps", PythonObject(warmup))
        py_setitem(fields, "total_steps", total)
    elif kind < 5:
        var step = 1 if kind == 4 else Int(py=py_call(b.int, values[1]))
        if step < 1:
            raise Error("ValueError: StepLR: step_size must be >= 1")
        py_setitem(fields, "base_lr", py_call(b.float, values[0]))
        py_setitem(fields, "gamma", py_call(b.float, values[2]))
        if kind == 3:
            py_setitem(fields, "step_size", PythonObject(step))
            py_setitem(fields, "_step", PythonObject(step))
    else:
        var maximum = Float64(py=py_call(b.float, values[0]))
        var total = Int(py=py_call(b.int, values[1]))
        var pct = Float64(py=py_call(b.float, values[2]))
        var anneal = values[3]
        var divide = Float64(py=py_call(b.float, values[4]))
        var final_divide = Float64(py=py_call(b.float, values[5]))
        if total < 1:
            raise Error("ValueError: OneCycleLR: total_steps must be >= 1")
        if not (pct >= 0.0 and pct <= 1.0):
            raise Error("ValueError: OneCycleLR: pct_start must lie in [0, 1]")
        if py_type(anneal) is not b.str or (String(anneal) != "cos" and String(anneal) != "linear"):
            raise Error("ValueError: OneCycleLR: anneal_strategy must be 'cos' or 'linear'")
        if not isfinite(maximum) or not isfinite(divide) or not isfinite(final_divide):
            raise Error("ValueError: OneCycleLR: max_lr and division factors must be finite")
        if divide == 0 or final_divide == 0:
            raise Error("ZeroDivisionError: OneCycleLR: division factors must be nonzero")
        py_setitem(fields, "max_lr", PythonObject(maximum))
        py_setitem(fields, "total_steps", PythonObject(total))
        py_setitem(fields, "pct_start", PythonObject(pct))
        py_setitem(fields, "anneal_strategy", anneal)
        py_setitem(fields, "div_factor", PythonObject(divide))
        py_setitem(fields, "final_div_factor", PythonObject(final_divide))
        py_setitem(fields, "three_phase", py_call(b.bool, values[6]))
    return PythonObject(None)


def schedule_operation(operation: String, owner: PythonObject, values: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    if operation == "schedule_construct":
        return schedule_construct(owner, values, context)
    if operation == "schedule_from_config":
        var cfg = values[0]
        opt_value(cfg, context)
        if not opt_exact(py_type(cfg), context["mapping_types"]):
            raise Error("TypeError: schedule configuration requires a native dictionary")
        var name = cfg["kind"]
        opt_scalar(name, context)
        var names = Python.tuple("constant", "linear", "cosine")
        for i in range(3):  # driver-loop: over three warmup schedule configuration names
            if name == names[i]:
                var result = py_call(b.object.__new__, context["schedules"][i])
                _ = schedule_construct(result, Python.tuple(cfg["peak_lr"], cfg["warmup_steps"], cfg["total_steps"], cfg["min_lr"]), context)
                return result
        raise Error("KeyError: unknown native schedule kind")
    opt_schedule(owner, context)
    var kind = schedule_kind(owner, context)
    var spec = schedule_spec(owner)
    var ip = Python.list(spec.kind, spec.a, spec.b, spec.c)
    var fp = Python.list(spec.f0, spec.f1, spec.f2, spec.f3)
    if operation == "schedule_spec":
        return Python.tuple(ip, fp)
    if operation == "schedule_config":
        var fields = py_fields(owner)
        var names = Python.tuple("constant", "linear", "cosine")
        return Python.dict(kind=names[kind], peak_lr=fields["peak_lr"], min_lr=fields["min_lr"],
                           warmup_steps=fields["warmup_steps"], total_steps=fields["total_steps"])
    opt_scalar(values, context)
    var style = 0 if kind < 3 else (1 if kind < 5 else 2)
    var entries = context["schedule_entries"]
    if operation == "schedule_values" or operation == "schedule_list":
        var result = py_call(entries["lr_schedule_series"], ip, fp, values[0], style)
        return py_call(result.tolist) if operation == "schedule_list" else result
    var value = py_call(entries["lr_schedule_at"], ip, fp, values[0], style)
    if operation == "schedule_bits":
        return PythonObject(Int(bitcast[DType.uint32](Float32(Float64(py=value)))))
    return value
