# SPDX-License-Identifier: Apache-2.0
# Optimizer admission runs before native code encounters any Python hook.
from bindings.native_loaded_module import NativeModules
from bindings.native_class_admission import dict_keys_admitted, dict_keys_admit
from bindings.p3_numpy_types import numpy_admitted_type
from bindings.p3_guard_records import guard_row_admitted, guard_row_admit
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.cpython_table import cpython_table
from bindings.py_balanced import py_call, py_type, py_get, py_fields


def opt_exact(cls: PythonObject, classes: PythonObject) raises -> Bool:
    for i in range(len(classes)):  # driver-loop: over fixed admitted native ABI types
        if cls is classes[i]:
            return True
    return False


def opt_numpy(cls: PythonObject, context: PythonObject) raises -> Bool:
    # Static NumPy type identities resolved once per NumPy module
    # (bindings/p3_numpy_types.mojo), not a flags call, a module dictionary
    # read and up to 16 name lookups per admitted value.
    return numpy_admitted_type(cls, context)


# Class-watcher slot family of a passed optimizer layout guard row
# (bindings/p3_guard_records.mojo): a row is re-checked only after its class
# or an ancestor was modified, instead of on every door call.
comptime _OPT_GUARD_SLOT = "p3_opt_layout_guard_v1_"


def opt_guards(context: PythonObject) raises:
    var guards = context["guards"]
    var mods = NativeModules()
    for i in range(len(guards)):  # driver-loop: over registered public native object layouts
        var guard = guards[i]
        if guard_row_admitted(mods, _OPT_GUARD_SLOT, i, guard[0], guard):
            continue
        if py_call(context["type_mro"], guard[0]) is not guard[1]:
            raise Error("TypeError: optimizer native owner inheritance has changed")
        var fields = guard[2]
        var pairs = guard[3]
        for j in range(len(pairs)):  # driver-loop: over retained native descriptors and attribute hooks
            if py_get(fields, pairs[j][0]) is not pairs[j][1]:
                raise Error("TypeError: optimizer native owner has a replaced descriptor or Python attribute hook")
        guard_row_admit(mods, _OPT_GUARD_SLOT, i, guard[0], guard)


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
    var cls = py_type(value)
    if opt_exact(cls, context["scalar_types"]) or opt_exact(cls, context["buffer_types"]) or opt_numpy(cls, context):
        return
    if cls is context["container_types"][0] or cls is context["container_types"][1]:
        for i in range(len(value)):  # driver-loop: over caller's native tensor/argument containers
            opt_value(value[i], context, depth + 1)
        return
    if opt_exact(cls, context["mapping_types"]):
        var table = cpython_table()
        if cls is context["container_types"][2]:
            var position = List[Int](length=1, fill=0)
            var pair = List[PyObjectPtr](length=2, fill=PyObjectPtr())
            while table[].PyDict_Next(value._obj_ptr, Int(position.unsafe_ptr()),
                    Int(pair.unsafe_ptr()), Int(pair.unsafe_ptr() + 1)):  # driver-loop: exact external dictionary entries
                var key = PythonObject(from_borrowed=pair[0])
                if py_type(key) is not context["scalar_types"][5]:
                    raise Error("TypeError: optimizer state requires native string keys")
                opt_value(PythonObject(from_borrowed=pair[1]), context, depth + 1)
        else:
            # Exact OrderedDict has its own order after move_to_end. Keep that
            # admission/refusal order without materializing a Python key list.
            ref cpy = Python().cpython()
            var pointer = table[].PyObject_GetIter(value._obj_ptr)
            if not pointer:
                raise cpy.unsafe_get_error()
            var iterator = PythonObject(from_owned=pointer)
            while True:  # driver-loop: exact external OrderedDict keys, under the GIL
                var item = table[].PyIter_Next(iterator._obj_ptr)
                if not item:
                    if table[].PyErr_Occurred():
                        raise cpy.unsafe_get_error()
                    break
                var key = PythonObject(from_owned=item)
                if py_type(key) is not context["scalar_types"][5]:
                    raise Error("TypeError: optimizer state requires native string keys")
                opt_value(value[key], context, depth + 1)
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


def _opt_owner_keys(owner: PythonObject, context: PythonObject) raises -> PythonObject:
    """`opt_owner`'s first half: the exact owner class and native string
    field names (so a keyed lookup runs no key `__eq__`). Returns the
    instance dictionary."""
    if not opt_exact(py_type(owner), context["owners"]):
        raise Error("TypeError: optimizer requires an exact SGD, Adam or AdamW native owner")
    var fields = py_fields(owner)
    # Key admission kept per dictionary under the dict watcher
    # (bindings/native_class_admission.mojo); Python 3.10/3.11 scan per call.
    if not dict_keys_admitted(fields):
        var table = cpython_table()
        var position = List[Int](length=1, fill=0)
        var pair = List[PyObjectPtr](length=2, fill=PyObjectPtr())
        while table[].PyDict_Next(fields._obj_ptr, Int(position.unsafe_ptr()),
                Int(pair.unsafe_ptr()), Int(pair.unsafe_ptr() + 1)):  # driver-loop: exact instance dictionary names
            if py_type(PythonObject(from_borrowed=pair[0])) is not context["scalar_types"][5]:
                raise Error("TypeError: optimizer metadata requires native string keys")
        dict_keys_admit(fields)
    return fields


def opt_owner_mode(owner: PythonObject, context: PythonObject) raises:
    """The part of `opt_owner` a router needs before it hands the owner to a
    compiled door that runs the full `opt_owner` itself as its first step
    (`door_step`, `door_state`, `door_zero` in
    bindings/optimizer_public_door.mojo): the exact owner class, native
    string field names and the `numeric_mode` word the router reads to pick
    the binding. The door's admission is the one that walks every field and
    buffer collection, so the owner is walked once per call, not twice; the
    first refusal and its message are the same, since nothing else reads the
    owner between the two."""
    var fields = _opt_owner_keys(owner, context)
    opt_scalar(py_get(fields, "numeric_mode"), context)


def opt_owner(owner: PythonObject, context: PythonObject) raises:
    opt_owner(owner, context, False)


def opt_owner(owner: PythonObject, context: PythonObject, defer_params: Bool) raises:
    """With `defer_params`, the `params` entry is admitted as an exact
    list/tuple only: the caller admits each element itself before it reads it
    (`door_step` reads every host parameter through `tensor_view`, which runs
    the same element admission, and admits the whole list with `opt_value`
    when the parameters are device-resident). Each tensor is admitted once
    per step instead of twice."""
    var fields = _opt_owner_keys(owner, context)
    for name in ["_where", "_KIND", "_res", "_res_host", "_resident_wanted", "_host_fresh", "_pristine", "_p_dev", "_g_dev", "n_total", "t", "numeric_mode", "lr", "momentum", "dampening", "weight_decay", "nesterov", "maximize", "betas", "eps", "accumulation_steps"]:  # driver-loop: over fixed optimizer scalar configuration/state
        opt_scalar(py_get(fields, PythonObject(name)), context)
    opt_schedule(py_get(fields, "lr_schedule"), context)
    var params = py_get(fields, "params")
    if defer_params:
        var cls = py_type(params)
        if not (cls is context["container_types"][0] or cls is context["container_types"][1]):
            opt_value(params, context)  # the full walk raises its usual refusal
    else:
        opt_value(params, context)
    for name in ["offsets", "buf_initialized", "_m", "_v"]:  # driver-loop: over four parameter/state buffer collections
        opt_value(py_get(fields, PythonObject(name)), context)
