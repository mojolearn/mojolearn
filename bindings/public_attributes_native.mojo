# SPDX-License-Identifier: Apache-2.0
# Native public attribute resolution and zero-copy fitted-output boundary.
# No generic object getter, property invocation, Array.__array__, or Python
# caller-frame inspection is part of the operation.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_loaded_module import native_loaded_module
from bindings.native_caller_frame import native_frame_globals
from bindings.mode_metadata_native import _fields, _class_fields, _truth
from bindings.classical_attributes import _type_mro, _classical_descriptor_get, classical_connect_numpy
from bindings.native_array_view import native_array_ndarray_view as _array_view
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.native_error_class import error_without_class


def _string_keys(fields: PythonObject) raises:
    var keys = py_call(py_type(Python.list()), fields.keys())
    for i in range(len(keys)):  # driver-loop: refuse dictionary equality hooks before native keyed reads
        if py_type(keys[i]) is not py_type(PythonObject("")):
            raise Error("TypeError: native attribute metadata requires exact string keys")


def _outside() raises -> Bool:
    ref cpy = Python().cpython()
    var pointer = cpy.lib.get_function[PyObjectPtr]("PyEval_GetFrame")()
    if not pointer:
        return True
    var frame = PythonObject(from_borrowed=pointer)
    while True:  # driver-loop: skip only the old public wrapper frames
        var fields = native_frame_globals(frame)
        _string_keys(fields)
        var module = py_get(fields, "__name__")
        var skip = False
        if py_type(module) is py_type(PythonObject("")) and String(module) == "mojolearn._mode":
            pointer = cpy.lib.get_function[PyObjectPtr]("PyFrame_GetCode")(frame._obj_ptr)
            if not pointer:
                raise cpy.unsafe_get_error()
            var code = PythonObject(from_owned=pointer)
            var name = String(code.co_name)
            skip = name == "public" or name == "guarded" or name == "tracked" or name == "load"
        if not skip:
            if py_get(fields, "__mojolearn_user_boundary__") is PythonObject(True):
                return True
            if py_get(fields, "__mojolearn_internal__") is PythonObject(True):
                return False
            if not _truth(module):
                module = PythonObject("")
            if py_type(module) is not py_type(PythonObject("")):
                raise Error("TypeError: native output caller module name must be a string")
            var name = String(module)
            return name != "mojolearn" and not name.startswith("mojolearn.") and not name.startswith("mojolearn_verify_all_")
        pointer = cpy.lib.get_function[PyObjectPtr]("PyFrame_GetBack")(frame._obj_ptr)
        _ = frame
        if not pointer:
            if cpy.lib.get_function[PyObjectPtr]("PyErr_Occurred")():
                raise cpy.unsafe_get_error()
            return True
        frame = PythonObject(from_owned=pointer)


def _plan(model: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var cls = py_type(model)
    if py_type(cls) is not b.type:
        raise Error("TypeError: native attributes refuse Python metaclass protocols")
    var mro = _type_mro(cls)
    var plan = PythonObject(None)
    var getter = context["missing"]
    for i in range(len(mro)):  # driver-loop: validate native slot and passive subclass ancestry
        if py_type(mro[i]) is not b.type:
            raise Error("TypeError: native attributes refuse Python metaclass protocols")
        var namespace = _class_fields(mro[i])
        if py_contains(namespace, "__getattr__"):
            raise Error("TypeError: native attributes refuse Python __getattr__ hooks")
        if getter is context["missing"] and py_contains(namespace, "__getattribute__"):
            getter = namespace["__getattribute__"]
        if plan is None:
            plan = py_get(context["plans"], mro[i])
    if plan is None or getter is not context["descriptor"]:
        raise Error("TypeError: native attribute class has no connected slot")
    return plan


def _has_slot(cls: PythonObject, name: String) raises -> Bool:
    var ancestry = _type_mro(cls)
    for i in range(len(ancestry)):  # driver-loop: descriptor metadata without foreign key equality hooks
        if py_contains(_class_fields(ancestry[i]), name):
            return True
    return False


def _member(model: PythonObject, name: PythonObject) raises -> PythonObject:
    var mro = _type_mro(py_type(model))
    for i in range(len(mro)):  # driver-loop: exact native attribute resolution
        var namespace = _class_fields(mro[i])
        if py_contains(namespace, name):
            return Python.tuple(True, namespace[name])
    return Python.tuple(False, None)


def _bound_function(function: PythonObject, model: PythonObject) raises -> PythonObject:
    # Exact Python function binding is a native CPython operation. A function
    # can have an instance attribute named __get__; never consult that mutable
    # attribute or invoke it as a callback while binding a public method.
    if py_type(function) is not native_loaded_module("types").FunctionType:
        raise Error("TypeError: native method binding requires an exact function")
    ref cpy = Python().cpython()
    var pointer = cpy.lib.get_function[PyObjectPtr]("PyMethod_New")(function._obj_ptr, model._obj_ptr)
    if not pointer:
        raise cpy.unsafe_get_error()
    return PythonObject(from_owned=pointer)


def _property(model: PythonObject, name: PythonObject, kind: String, fields: PythonObject,
              plan: PythonObject, context: PythonObject) raises -> PythonObject:
    if kind == "cholesky_logdet":
        var info = PythonObject(None)
        try:
            info = _lookup(model, PythonObject("info_"), fields, plan, context)
        except error:
            if String(error).startswith("AttributeError:"):
                raise Error("AttributeError: mojolearn Cholesky: logdet_ is set by fit")
            raise error^
        var b = native_loaded_module("builtins")
        if py_type(info) is not b.int and py_type(info) is not b.bool and py_type(info) is not b.float:
            raise Error("TypeError: Cholesky info_ must be native numeric status")
        if _truth(info):
            raise Error("ValueError: mojolearn Cholesky: the factorization failed (info=" + String(py_call(b.str, info)) + "), so there is no determinant to report (DEVIATION 1634)")
        return _lookup(model, PythonObject("_logdet"), fields, plan, context)
    if kind == "stl_config":
        var out = Python.dict()
        var names = Python.tuple("period", "seasonal", "seasonal_deg", "seasonal_jump", "trend", "trend_deg", "trend_jump", "low_pass", "low_pass_deg", "low_pass_jump", "robust")
        for i in range(len(names)):  # driver-loop: supported STL configuration fields
            py_setitem(out, names[i], _lookup(model, names[i], fields, plan, context))
        return out
    raise Error("TypeError: unconnected native property schema")


def _lookup(model: PythonObject, name: PythonObject, fields: PythonObject,
            plan: PythonObject, context: PythonObject) raises -> PythonObject:
    var found = _member(model, name)
    var present = Bool(py=found[0])
    var member = found[1]
    var cls = py_type(model)
    var member_type = py_type(member)
    var descriptor = present and _has_slot(member_type, "__get__")
    var data = descriptor and (_has_slot(member_type, "__set__") or _has_slot(member_type, "__delete__"))
    var binding = py_get(plan["bindings"], name)
    var known = binding is not None and member is binding[0]
    if data:
        var property_plan = py_get(plan["properties"], name)
        if property_plan is not None and member is property_plan[0]:
            return _property(model, name, String(property_plan[1]), fields, plan, context)
        if not known:
            raise Error("TypeError: Python attribute properties/descriptors require a native owner")
        return _classical_descriptor_get(binding, model, cls, plan)
    if py_contains(fields, name):
        return fields[name]
    if not present:
        raise Error("AttributeError: native model has no attribute '" + String(name) + "'")
    if descriptor:
        # Creating a bound method does not execute its body. Preserve ordinary
        # subclass functions and later import-time native setter connections.
        # Use CPython's method constructor, not mutable function.__get__.
        if member_type is native_loaded_module("types").FunctionType:
            return _bound_function(member, model)
        if not known:
            raise Error("TypeError: Python attribute descriptor requires a native owner")
        return _classical_descriptor_get(binding, model, cls, plan)
    return member


def public_attribute_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=args_ptr)
        var context = args[0]
        var model = args[1]
        var name = args[2]
        if py_type(name) is not py_type(PythonObject("")):
            raise Error("TypeError: native model attribute name must be an exact string")
        var plan = _plan(model, context)
        var fields = _fields(model)
        var result = _lookup(model, name, fields, plan, context)
        var text = String(name)
        if (text.byte_length() > 0 and text.endswith("_") and not text.startswith("_")
                and py_type(result) is context["array_class"]):
            classical_connect_numpy(context)
            if py_get(context, "ndarray") is not None and _outside():
                result = _array_view(result, context)
        return result.steal_data()
    except error:
        if String(error).startswith("AttributeError:"):
            return raise_python_exception(error_without_class(error, "AttributeError"), ExceptionType("PyExc_AttributeError"))
        if String(error).startswith("ValueError:"):
            return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
        return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
