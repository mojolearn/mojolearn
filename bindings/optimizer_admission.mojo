# SPDX-License-Identifier: Apache-2.0
# Optimizer admission runs before native code encounters any Python hook.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_fields


def opt_exact(cls: PythonObject, classes: PythonObject) raises -> Bool:
    for i in range(len(classes)):  # driver-loop: over fixed admitted native ABI types
        if cls is classes[i]:
            return True
    return False


def opt_numpy(cls: PythonObject, context: PythonObject) raises -> Bool:
    var module = py_get(context["modules"], "numpy")
    if module is None or py_type(module) is not context["module_type"]:
        return False
    if (Int(py=py_call(context["type_flags"], cls)) & 512) != 0:
        return False
    var fields = py_fields(module)
    var names = context["numpy_names"]
    for i in range(len(names)):  # driver-loop: over fixed native NumPy buffer/scalar types
        if cls is py_get(fields, names[i]):
            return True
    return False


def opt_guards(context: PythonObject) raises:
    var guards = context["guards"]
    for i in range(len(guards)):  # driver-loop: over registered public native object layouts
        var guard = guards[i]
        if py_call(context["type_mro"], guard[0]) is not guard[1]:
            raise Error("TypeError: optimizer native owner inheritance has changed")
        var fields = guard[2]
        var pairs = guard[3]
        for j in range(len(pairs)):  # driver-loop: over retained native descriptors and attribute hooks
            if py_get(fields, pairs[j][0]) is not pairs[j][1]:
                raise Error("TypeError: optimizer native owner has a replaced descriptor or Python attribute hook")


def opt_scalar(value: PythonObject, context: PythonObject, depth: Int = 0) raises:
    if depth > 64:
        raise Error("TypeError: optimizer scalar metadata exceeds the native nesting limit")
    var cls = py_type(value)
    if opt_exact(cls, context["scalar_types"]):
        return
    if opt_numpy(cls, context):
        var numpy = py_fields(context["modules"]["numpy"])
        if cls is not numpy["ndarray"]:
            return
    if cls is context["container_types"][1] or cls is context["container_types"][0]:
        for i in range(len(value)):  # driver-loop: over scalar shape/configuration metadata
            opt_scalar(value[i], context, depth + 1)
        return
    raise Error("TypeError: optimizer scalar arguments require native scalar values without Python conversion hooks")


def opt_schedule(value: PythonObject, context: PythonObject) raises:
    if value is None:
        return
    if not opt_exact(py_type(value), context["schedules"]):
        raise Error("TypeError: optimizer lr_schedule requires a built-in native schedule (ConstantLR, WarmupLinearLR, WarmupCosineLR, StepLR, ExponentialLR or OneCycleLR); Python learning-rate callbacks cannot execute in the native runtime")
    var fields = py_fields(value)
    var classes = context["schedules"]
    var index = 0
    for i in range(len(classes)):  # driver-loop: over connected built-in schedule types
        if py_type(value) is classes[i]:
            index = i
    if index < 3:
        for name in ["peak_lr", "min_lr", "warmup_steps", "total_steps"]:  # driver-loop: over stored warmup schedule configuration
            opt_scalar(fields[PythonObject(name)], context)
    elif index < 5:
        opt_scalar(fields["base_lr"], context)
        opt_scalar(fields["gamma"], context)
        if index == 3:
            opt_scalar(fields["_step"], context)
    else:
        for name in ["max_lr", "total_steps", "pct_start", "anneal_strategy", "div_factor", "final_div_factor", "three_phase"]:  # driver-loop: over stored OneCycle schedule configuration
            opt_scalar(fields[PythonObject(name)], context)


def opt_value(value: PythonObject, context: PythonObject, depth: Int = 0) raises:
    if depth > 64:
        raise Error("TypeError: optimizer input nesting exceeds the native limit")
    var b = Python.import_module("builtins")
    var cls = py_type(value)
    if opt_exact(cls, context["scalar_types"]) or opt_exact(cls, context["buffer_types"]) or opt_numpy(cls, context):
        return
    if cls is context["container_types"][0] or cls is context["container_types"][1]:
        for i in range(len(value)):  # driver-loop: over caller's native tensor/argument containers
            opt_value(value[i], context, depth + 1)
        return
    if opt_exact(cls, context["mapping_types"]):
        var keys = py_call(context["container_types"][0], value)
        for i in range(len(keys)):  # driver-loop: over validate names before keyed native dictionary reads
            if py_type(keys[i]) is not b.str:
                raise Error("TypeError: optimizer state requires native string keys")
            opt_value(value[keys[i]], context, depth + 1)
        return
    if cls is context["container_types"][5]:
        var owner = value.obj
        if owner is None or opt_exact(py_type(owner), context["buffer_types"]) or opt_numpy(py_type(owner), context):
            return
        raise Error("TypeError: optimizer memoryview requires a native buffer owner")
    if cls is context["array_class"]:
        var getters = context["array_getters"]
        for name in ["shape", "dtype", "order", "_addr", "_readonly", "ndim", "size", "nbytes", "itemsize", "strides"]:  # driver-loop: over fixed native Array metadata slots
            opt_scalar(py_call(getters[PythonObject(name)], value), context, depth + 1)
        var view = py_call(getters["_mv"], value)
        if py_type(view) is not context["container_types"][5]:
            raise Error("TypeError: optimizer Array requires native memoryview storage")
        opt_value(view, context, depth + 1)
        return
    if cls is context["device_class"]:
        var fields = py_fields(value)
        for name in ["h", "size", "shape"]:  # driver-loop: over fixed native device-handle metadata
            opt_scalar(fields[PythonObject(name)], context, depth + 1)
        return
    raise Error("TypeError: optimizer requires native buffers and lists/tuples; Python protocols and callbacks cannot execute")


def opt_owner(owner: PythonObject, context: PythonObject) raises:
    if not opt_exact(py_type(owner), context["owners"]):
        raise Error("TypeError: optimizer requires an exact SGD, Adam or AdamW native owner")
    var fields = py_fields(owner)
    var b = Python.import_module("builtins")
    var keys = py_call(context["container_types"][0], fields)
    for i in range(len(keys)):  # driver-loop: over stored field names before native keyed lookup
        if py_type(keys[i]) is not b.str:
            raise Error("TypeError: optimizer metadata requires native string keys")
    for name in ["_where", "_KIND", "_res", "_res_host", "_resident_wanted", "_host_fresh", "_pristine", "_p_dev", "_g_dev", "n_total", "t", "numeric_mode", "lr", "momentum", "dampening", "weight_decay", "nesterov", "maximize", "betas", "eps", "accumulation_steps"]:  # driver-loop: over fixed optimizer scalar configuration/state
        opt_scalar(py_get(fields, PythonObject(name)), context)
    opt_schedule(py_get(fields, "lr_schedule"), context)
    for name in ["params", "offsets", "buf_initialized", "_m", "_v"]:  # driver-loop: over five parameter/state buffer collections
        opt_value(py_get(fields, PythonObject(name)), context)
