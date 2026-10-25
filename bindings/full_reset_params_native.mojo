# SPDX-License-Identifier: Apache-2.0
# ParamsMixin full-reset is separate from the nineteen linear setters' fitted-
# field-only reset. Constructor event schemas borrow inputs without cloning or
# calling Python constructors. Every old owner that will actually be dropped is
# admitted before clearing the existing dictionary, whose identity is preserved.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_loaded_module import native_loaded_module
from bindings.native_error_class import error_without_class
from bindings.native_owner_lifetime import native_release_weakrefs
from bindings.mode_metadata_native import _fields, _class_fields
from bindings.params_metadata_native import _read, _connected_plan, _resolved
from bindings.linear_constructor_native import _constructor_plan
from bindings.params_setter_native import _set as _linear_set, _constructor_layout
from bindings.declared_params_native import _assign as _declared_set, _release as _declared_release
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.optimizer_public_words import stash_python_error, optimizer_python_error, PYTHON_ERROR_PENDING


def _plan(model: PythonObject, fields: PythonObject, context: PythonObject) raises -> PythonObject:
    var plan = _constructor_plan(model, fields, context)
    var names = plan["field_names"]
    var cls = py_type(model)
    for i in range(len(names)):  # driver-loop: fixed extra fields must reject subclass descriptors too
        var value = _resolved(cls, names[i], context["missing"])
        if value is context["missing"]:
            continue
        var kind = py_type(value)
        if (_resolved(kind, "__set__", context["missing"]) is not context["missing"]
                or _resolved(kind, "__delete__", context["missing"]) is not context["missing"]):
            raise Error("TypeError: native parameter reset cannot invoke field descriptor callbacks")
    return plan


def _release_owned(values: PythonObject, retained: PythonObject, context: PythonObject) raises:
    var b = native_loaded_module("builtins")
    var todo = py_call(b.list, values)
    var seen = Python.dict()
    var cursor = 0
    while cursor < len(todo):  # driver-loop: cycle-safe ownership admission before the full dictionary reset
        var item = todo[cursor]
        cursor += 1
        var identity = PythonObject(Int(item._obj_ptr))
        if py_contains(seen, identity):
            continue
        py_setitem(seen, identity, item)
        var kept = False
        for i in range(len(retained)):  # driver-loop: replacement fields already retain these identical owners
            if item is retained[i]:
                kept = True
                break
        if kept:
            continue
        var cls = py_type(item)
        if cls is b.object:
            # Exact object() has no instance dictionary, slots, weakrefs or
            # destructor callback. Parameter sentinels remain valid owners.
            continue
        if cls is b.type and py_contains(context["plans"], item):
            # Import-time schema keys independently retain these exact classes.
            continue
        native_release_weakrefs(item, context["declared"]["ownership"])
        if cls is b.list or cls is b.tuple:
            for i in range(len(item)):  # driver-loop: ordinary container elements, no input protocols
                _ = py_call(todo.append, item[i])
        elif cls is b.dict:
            var pairs = py_call(b.list, item.items())
            for i in range(len(pairs)):  # driver-loop: visit key/value owners without hashing caller keys
                _ = py_call(todo.append, pairs[i][0])
                _ = py_call(todo.append, pairs[i][1])
        elif py_type(cls) is b.type and _connected_plan(cls, context) is not None:
            var fields = _fields(item)
            var plan = _plan(item, fields, context)
            if not Bool(py=plan["release_allowed"]):
                raise Error("TypeError: native parameter reset cannot execute Python finalizers")
            var ancestry = py_call(b.type.__dict__["__mro__"].__get__, cls)
            for i in range(len(ancestry)):  # driver-loop: refuse unenumerated slot owners before destruction
                var slots = py_get(_class_fields(ancestry[i]), PythonObject("__slots__"))
                if slots is not None and py_type(slots) is not b.tuple and py_type(slots) is not b.list and py_type(slots) is not b.str:
                    raise Error("TypeError: native parameter reset cannot invoke slot metadata protocols")
                if slots is not None and len(slots) > 0:
                    raise Error("TypeError: native parameter reset requires supported native ownership for slotted values")
            _ = py_call(todo.append, fields)
        else:
            # Existing admission covers native buffers, NumPy ownership,
            # connected built-in models and retained native module handles.
            _declared_release(item, context["declared"])


def _fresh_fields(model: PythonObject, values: PythonObject, plan: PythonObject,
                  context: PythonObject) raises -> PythonObject:
    var fresh = Python.dict()
    var events = plan["events"]
    for i in range(len(events)):  # driver-loop: original constructor assignment and wrapper order
        var event = events[i]
        var name = event[0]
        var value = event[2]
        if String(event[1]) == "parameter":
            value = values[event[2]]
        if Bool(py=event[3]) and value is None:
            # A replacement is fresh: the old instance mode is irrelevant.
            # Nested NumericModeMixin wrappers preserve already-written mode,
            # and ordinary inherited class metadata satisfies old hasattr().
            if (py_contains(fresh, name)
                    or _resolved(py_type(model), name, context["missing"]) is not context["missing"]):
                continue
        py_setitem(fresh, name, value)
    return fresh


def _reset(model: PythonObject, params: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var fields = _fields(model)
    var plan = _plan(model, fields, context)
    var values = _read(model, False, context["parameter_metadata"], Python.list())
    var pairs = py_call(b.list, params.items())
    var nested = Python.dict()
    for i in range(len(pairs)):  # driver-loop: validate every root before any nested or parent mutation
        var key = pairs[i][0]
        if py_type(key) is not b.str:
            raise Error("TypeError: native parameter names must be exact strings")
        var parts = py_call(key.partition, "__")
        var name = parts[0]
        if not py_contains(values, name):
            raise Error("ValueError: Invalid parameter " + String(py_call(b.repr, name)) + " for estimator "
                        + String(py_type(model).__name__) + ". Valid parameters are: "
                        + String(py_call(b.repr, py_call(b.sorted, values))) + ".")
        if String(parts[1]) == "":
            py_setitem(values, name, pairs[i][1])
        else:
            if not py_contains(nested, name):
                py_setitem(nested, name, Python.dict())
            py_setitem(nested[name], parts[2], pairs[i][1])
    var names = py_call(b.list, nested.keys())
    for i in range(len(names)):  # driver-loop: earlier successful nested changes remain if a later child refuses
        _ = _set(values[names[i]], nested[names[i]], context)
    var fresh = _fresh_fields(model, values, plan, context)
    var previous = py_call(b.list, fields.values())
    var retained = py_call(b.list, fresh.values())
    _release_owned(previous, retained, context)
    # Native dictionary operations preserve the public dictionary's identity.
    # All removed references are retained in previous until admission completes.
    _ = py_call(fields.clear)
    _ = py_call(fields.update, fresh)
    return model


def _dispatch(model: PythonObject, params: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    if py_type(params) is not b.dict:
        raise Error("TypeError: native parameter keywords require an ordinary dict")
    if len(params) == 0:
        return model
    if not Bool(py=context["ready"]):
        raise Error("RuntimeError: native full-reset constructor schemas are not connected")
    var cls = py_type(model)
    if py_type(cls) is not b.type:
        raise Error("TypeError: native parameter reset cannot invoke Python metaclass hooks")
    if _connected_plan(cls, context) is not None:
        return _reset(model, params, context)
    if _connected_plan(cls, context["linear"]) is not None:
        return _linear_set(model, params, context["linear"])
    if _connected_plan(cls, context["declared"]) is not None:
        return _declared_set(model, params, context["declared"])
    var native = py_get(context["native_setters"], cls)
    if native is not None:
        var fields = _fields(model)
        _constructor_layout(model, fields, native["layout"])
        # Forward the exact existing native ABI, never a Python public method
        # or constructor. Export identity alone does not qualify its transitive
        # owner release: the public nested-lifetime tests remain mandatory.
        ref cpy = Python().cpython()
        var entry = native["entry"]
        var args = Python.tuple(model, native["operation"], Python.tuple(params), native["context"])
        var result = cpy.PyObject_Call(entry._obj_ptr, args._obj_ptr, PyObjectPtr())
        if not result:
            var failure = stash_python_error()
            raise failure^
        return PythonObject(from_owned=result)
    if _resolved(cls, "set_params", context["missing"]) is context["missing"]:
        raise Error("AttributeError: '" + String(cls.__name__) + "' object has no attribute 'set_params'")
    raise Error("TypeError: Python parameter callbacks are not accepted; use a connected native setter")


def _set(model: PythonObject, params: PythonObject, context: PythonObject) raises -> PythonObject:
    ref cpy = Python().cpython()
    var label = String(" while setting estimator parameters")
    if cpy.lib.get_function[Int32]("Py_EnterRecursiveCall")(label.as_c_string_slice().unsafe_ptr()) != 0:
        var failure = stash_python_error()
        raise failure^
    try:
        var result = _dispatch(model, params, context)
        cpy.lib.get_function[NoneType]("Py_LeaveRecursiveCall")()
        return result
    except error:
        cpy.lib.get_function[NoneType]("Py_LeaveRecursiveCall")()
        raise error^


def full_reset_params_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=args_ptr)
        if len(args) != 3:
            raise Error("TypeError: full-reset parameters expect model, keywords and native context")
        var result = _set(args[0], args[1], args[2])
        return result.steal_data()
    except error:
        if String(error) == PYTHON_ERROR_PENDING:
            return optimizer_python_error(error)
        if String(error).startswith("RecursionError:"):
            return raise_python_exception(error_without_class(error, "RecursionError"), ExceptionType("PyExc_RecursionError"))
        if String(error).startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        if String(error).startswith("ValueError:"):
            return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
        if String(error).startswith("AttributeError:"):
            return raise_python_exception(error_without_class(error, "AttributeError"), ExceptionType("PyExc_AttributeError"))
        return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
