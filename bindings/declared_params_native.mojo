# SPDX-License-Identifier: Apache-2.0
# Shallow parameter mutation preserves keyword order and fitted state. This is
# deliberately separate from the constructor/reset semantics of ParamsMixin.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_loaded_module import native_loaded_module
from bindings.native_error_class import error_without_class
from bindings.native_instance_item import native_instance_item
from bindings.native_owner_lifetime import native_release_admit, native_release_weakrefs
from bindings.classical_attributes import classical_connect_numpy
from bindings.mode_metadata_native import _fields
from bindings.params_metadata_native import _plan, _connected_plan, _resolved
from bindings.params_setter_native import _constructor_layout
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.optimizer_public_words import optimizer_python_error, PYTHON_ERROR_PENDING


def _assignment_plan(model: PythonObject, fields: PythonObject, context: PythonObject) raises -> PythonObject:
    var metadata = _plan(model, fields, context["parameter_metadata"])
    var cls = py_type(model)
    var plan = _connected_plan(cls, context)
    if plan is None:
        raise Error("TypeError: use a connected native declared-parameter estimator")
    var pinned = plan["pinned"]
    for i in range(len(pinned)):  # driver-loop: fixed setter/storage identities, including passive subclasses
        if _resolved(cls, pinned[i][0], context["missing"]) is not pinned[i][1]:
            raise Error("TypeError: native parameter assignment refuses changed setters or storage descriptors")
    for name in ["set_params", "__setattr__", "_parameters"]:  # driver-loop: instance protocols cannot redirect native assignment
        if py_contains(fields, PythonObject(name)):
            raise Error("TypeError: native parameter assignment refuses instance protocol overrides")
    return plan


def _release(value: PythonObject, context: PythonObject) raises:
    var b = native_loaded_module("builtins")
    var todo = Python.list()
    _ = py_call(todo.append, value)
    var seen = Python.dict()
    var retained = context["retained_owners"]
    var cursor = 0
    classical_connect_numpy(context["ownership"])
    while cursor < len(todo):  # driver-loop: cycle-safe native ownership traversal before each replacement
        var item = todo[cursor]
        cursor += 1
        var identity = PythonObject(Int(item._obj_ptr))
        if py_contains(seen, identity):
            continue
        py_setitem(seen, identity, item)
        var pinned = False
        for i in range(len(retained)):  # driver-loop: independently retained native modules, classes and fixed score functions
            if item is retained[i]:
                pinned = True
                break
        if pinned:
            continue
        native_release_weakrefs(item, context["ownership"])
        var cls = py_type(item)
        if cls is b.object:
            continue
        if cls is b.list or cls is b.tuple:
            for i in range(len(item)):  # driver-loop: exact container owners, without element protocols
                _ = py_call(todo.append, item[i])
        elif cls is b.dict:
            var pairs = py_call(b.list, item.items())
            for i in range(len(pairs)):  # driver-loop: no user-key hashing or equality when following dictionary owners
                _ = py_call(todo.append, pairs[i][0])
                _ = py_call(todo.append, pairs[i][1])
        elif py_type(cls) is b.type and py_contains(context["release_plans"], cls):
            var plan = context["release_plans"][cls]
            var fields = _fields(item)
            _constructor_layout(item, fields, plan["constructor"])
            if _resolved(cls, PythonObject("__del__"), context["missing"]) is not plan["finalizer"]:
                raise Error("TypeError: native parameter assignment refuses changed model finalizers")
            _ = py_call(todo.append, fields)
        else:
            native_release_admit(item, context["ownership"])


def _assign(model: PythonObject, keywords: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    if py_type(keywords) is not b.dict:
        raise Error("TypeError: native parameter keywords require an ordinary dict")
    # The old loop does nothing for empty keywords, even for an incomplete
    # model; do not turn that identity/no-op into constructor validation.
    if len(keywords) == 0:
        return model
    if not Bool(py=context["ready"]):
        raise Error("RuntimeError: native declared-parameter plans are not connected")
    var fields = _fields(model)
    var plan = _assignment_plan(model, fields, context)
    var names = plan["names"]
    var cls = py_type(model)
    # Exact dict.items materializes references in insertion order without
    # hashing supplied keys. Validate each key immediately before its write:
    # an invalid later keyword must not roll back earlier valid assignments.
    var pairs = py_call(b.list, keywords.items())
    for i in range(len(pairs)):  # driver-loop: ordered shallow keyword assignments, never nested updates
        var name = pairs[i][0]
        if py_type(name) is not b.str:
            raise Error("TypeError: native parameter names must be exact strings")
        if not Bool(py=py_call(names.__contains__, name)):
            if String(plan["style"]) == "prep":
                raise Error("ValueError: mojolearn: invalid parameter " + String(py_call(b.repr, name)) + " for " + String(cls.__name__))
            raise Error("ValueError: Invalid " + String(cls.__name__) + " parameter: " + String(name))
        var descriptor = _resolved(cls, name, context["missing"])
        if descriptor is not context["missing"]:
            var descriptor_type = py_type(descriptor)
            if py_type(descriptor_type) is not b.type or _resolved(descriptor_type, PythonObject("__set__"), context["missing"]) is not context["missing"]:
                raise Error("TypeError: native parameter assignment cannot invoke Python parameter descriptors")
        var value = pairs[i][1]
        if py_contains(fields, name):
            var previous = native_instance_item(fields, name)
            if previous is not value:
                _release(previous, context)
        py_setitem(fields, name, value)
    return model


def declared_params_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=args_ptr)
        if len(args) != 3:
            raise Error("TypeError: declared parameter assignment expects model, keywords and context")
        var result = _assign(args[0], args[1], args[2])
        return result.steal_data()
    except error:
        if String(error) == PYTHON_ERROR_PENDING:
            return optimizer_python_error(error)
        if String(error).startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        if String(error).startswith("ValueError:"):
            return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
        if String(error).startswith("AttributeError:"):
            return raise_python_exception(error_without_class(error, "AttributeError"), ExceptionType("PyExc_AttributeError"))
        return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
