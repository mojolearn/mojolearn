# SPDX-License-Identifier: Apache-2.0
# The nineteen linear constructors assign existing references only. Public
# arguments are bound here; no Python initializer, signature binder, descriptor
# or caller protocol runs. Reinitialization preserves fitted/user fields and
# preserves an existing numeric_mode when the supplied mode is None, matching
# NumericModeMixin's original post-constructor assignment rule.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_loaded_module import native_loaded_module
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.mode_metadata_native import _fields, _class_fields
from bindings.params_setter_native import _setter_plan, _release_admit
from bindings.classical_attributes import classical_connect_numpy
from bindings.optimizer_public_words import optimizer_python_error, PYTHON_ERROR_PENDING


def _initialize(model: PythonObject, positional: PythonObject, keywords: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    if py_type(positional) is not b.tuple or py_type(keywords) is not b.dict:
        raise Error("TypeError: native constructor arguments require an ordinary tuple and dict")
    if not Bool(py=context["ready"]):
        raise Error("RuntimeError: native linear constructor schemas are not connected")
    var fields = _fields(model)
    _ = _setter_plan(model, fields, context["setters"])
    var plan = py_get(context["plans"], py_type(model))
    if plan is None:
        raise Error("TypeError: native linear constructor requires an exact connected class")
    var hooks = plan["field_hooks"]
    for i in range(len(hooks)):  # driver-loop: reject changed parameter descriptors without invoking them
        if py_get(_class_fields(hooks[i][0]), hooks[i][1]) is not hooks[i][2]:
            raise Error("TypeError: native constructor parameter descriptors changed after connection")
    var keys = py_call(b.list, keywords.keys())
    for i in range(len(keys)):  # driver-loop: reject user-key hashing/equality before argument lookup
        if py_type(keys[i]) is not b.str:
            raise Error("TypeError: native constructor keyword names must be exact strings")
    var names = plan["names"]
    var allowed = plan["positional"]
    var defaults = plan["defaults"]
    var label = String(plan["label"])
    # Python's old mode wrapper removed numeric_mode before binding the
    # estimator's original signature; it is still keyword-only here.
    var values = Python.dict()
    for i in range(min(len(positional), len(allowed))):  # driver-loop: native positional argument binding
        py_setitem(values, allowed[i], positional[i])
    for i in range(len(keys)):  # driver-loop: duplicate and unexpected keywords before any state mutation
        var name = keys[i]
        if not py_contains(names, name):
            raise Error("TypeError: " + label + "() got an unexpected keyword argument '" + String(name) + "'")
        if py_contains(values, name):
            raise Error("TypeError: " + label + "() got multiple values for argument '" + String(name) + "'")
        py_setitem(values, name, keywords[name])
    if len(positional) > len(allowed):
        var required = 1  # the already bound self still appears in Python's error counts
        for i in range(len(allowed)):  # driver-loop: exact constructor positional arity metadata
            if not py_contains(defaults, allowed[i]):
                required += 1
        var maximum = len(allowed) + 1
        var expected = String(maximum)
        if required != maximum:
            expected = "from " + String(required) + " to " + String(maximum)
        var noun = String(" positional arguments")
        if maximum == 1:
            noun = " positional argument"
        var supplied = String(len(positional) + 1)
        var kwonly = 0
        for i in range(len(keys)):  # driver-loop: numeric_mode was stripped by the original wrapper
            if String(keys[i]) != "numeric_mode" and not py_contains(allowed, keys[i]):
                kwonly += 1
        if kwonly > 0:
            var suffix = String(" keyword-only argument")
            if kwonly != 1:
                suffix += "s"
            supplied += " positional arguments (and " + String(kwonly) + suffix + ")"
        raise Error("TypeError: " + label + "() takes " + expected + noun + " but " + supplied + " were given")
    for i in range(len(names)):  # driver-loop: borrow the import-time defaults without copies/conversions
        var name = names[i]
        if not py_contains(values, name):
            if not py_contains(defaults, name):
                raise Error("TypeError: " + label + "() missing required argument '" + String(name) + "'")
            py_setitem(values, name, defaults[name])
    if values["numeric_mode"] is None and py_contains(fields, "numeric_mode"):
        py_setitem(values, "numeric_mode", fields["numeric_mode"])
    var seen = Python.dict()
    var ownership_connected = False
    for i in range(len(names)):  # driver-loop: all overwritten owners must have callback-free destruction before commit
        var name = names[i]
        if py_contains(fields, name) and fields[name] is not values[name]:
            if not ownership_connected:
                classical_connect_numpy(context["setters"]["ownership"])
                ownership_connected = True
            _release_admit(fields[name], context["setters"], seen)
    var order = plan["order"]
    for i in range(len(order)):  # driver-loop: original field assignment order; fitted and unrelated fields remain intact
        py_setitem(fields, order[i], values[order[i]])
    return PythonObject(None)


def linear_constructor_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=args_ptr)
        if len(args) != 4:
            raise Error("TypeError: native constructor expects model, positional arguments, keywords and context")
        var result = _initialize(args[0], args[1], args[2], args[3])
        return result.steal_data()
    except error:
        if String(error) == PYTHON_ERROR_PENDING:
            return optimizer_python_error(error)
        if String(error).startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        if String(error).startswith("AttributeError:"):
            return raise_python_exception(error_without_class(error, "AttributeError"), ExceptionType("PyExc_AttributeError"))
        return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
