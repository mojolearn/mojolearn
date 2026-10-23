# SPDX-License-Identifier: Apache-2.0
# Exact native search/container storage. No Python attribute or subclass hooks.
from std.python import Python, PythonObject
from bindings.py_slice_native import py_slice, py_slice_from
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.msel_inputs import msel_input, msel_admit
from bindings.arena_materialize_native import _array


def msel_search_plan(model: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var cls = py_type(model)
    if py_type(cls) is not b.type:
        raise Error("TypeError: native search refuses Python metaclass hooks")
    var plan = py_get(context["family_msel_search_layouts"], cls)
    if plan is None:
        raise Error("TypeError: use a connected native search or parameter container; Python subclass hooks are not supported")
    var mro = py_call(b.type.__dict__["__mro__"].__get__, cls)
    if len(mro) != len(plan["mro"]):
        raise Error("TypeError: native search ancestry changed")
    for k in range(len(mro)):  # driver-loop: retained native class ancestry
        if mro[k] is not plan["mro"][k]:
            raise Error("TypeError: native search ancestry changed")
    var hooks = plan["hooks"]
    for k in range(len(hooks)):  # driver-loop: class storage/method identity admission
        var row = py_call(b.type.__dict__["__dict__"].__get__, hooks[k][0])
        if py_get(row, hooks[k][1]) is not hooks[k][2]:
            raise Error("TypeError: native search class hooks changed")
    return plan


def msel_search_fields(model: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var plan = msel_search_plan(model, context)
    var fields = py_call(plan["fields_get"], model, py_type(model))
    var keys = py_call(b.list, fields.keys())
    for k in range(len(keys)):  # driver-loop: stored field names before dictionary lookup
        if py_type(keys[k]) is not b.str:
            raise Error("TypeError: native search state requires exact string keys")
        if py_contains(plan["protected"], keys[k]):
            raise Error("TypeError: native search instance overrides a method or storage hook")
    return fields


def msel_search_initialize(model: PythonObject, args: PythonObject,
                           context: PythonObject) raises -> PythonObject:
    var plan = msel_search_plan(model, context)
    var fields = msel_search_fields(model, context)
    var names = plan["parameters"]
    if len(args) != len(names):
        raise Error("TypeError: native search constructor argument count disagrees with its schema")
    for k in range(len(names)):  # driver-loop: fixed constructor assignments, no callbacks or input coercion
        py_setitem(fields, names[k], args[k])
    return PythonObject(None)


def msel_parameter_values(value: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    msel_admit(value, context)
    var cls = py_type(value)
    if cls is b.str or cls is b.bytes:
        raise Error("TypeError: parameter values require a native sequence, not a string")
    if cls is context["array_class"]:
        var packed = msel_input(value, "parameter values", context)
        var rank = Int(py=packed.ndim)
        if rank == 0:
            raise Error("TypeError: iteration over a 0-d Array")
        if rank == 1:
            return py_call(b.list, packed._mv)
        var shape = py_slice_from(packed.shape, 1)
        var n = Int(py=packed.shape[0])
        var step = 1
        for k in range(len(shape)):  # driver-loop: native row view extents
            step *= Int(py=shape[k])
        var rows = Python.list()
        for k in range(n):  # driver-loop: parameter row views retain their original packed owner
            var view = py_slice(packed._mv, k * step, (k + 1) * step)
            var row = _array(shape, String(packed.dtype), view, packed._store, packed._pin,
                             context["family_msel_resident_context"], packed)
            _ = py_call(rows.append, row)
        return rows
    if (cls is b.list or cls is b.tuple or cls is b.range or cls is b.set or cls is b.frozenset
            or cls is b.dict or cls is b.memoryview or cls is context["array_type"]
            or cls is py_get(context, "ndarray")):
        # Exact built-in / immutable extension iterator slots only. ndarray
        # iteration yields native scalar/row views, without NumPy Python code.
        return py_call(b.list, value)
    raise Error("TypeError: parameter values require native containers or buffers; Python iterator/distribution callbacks are not supported")
