# SPDX-License-Identifier: Apache-2.0
# Public set_params for connected field-only constructors. Reuse their native
# plans without cloning parameters or invoking Python estimator methods.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_loaded_module import native_loaded_module
from bindings.native_buffer_owner import native_storage_allocate
from bindings.native_owner_lifetime import native_release_admit, native_release_weakrefs
from bindings.classical_attributes import classical_connect_numpy
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.mode_metadata_native import _fields, _class_fields
from bindings.params_metadata_native import _read, _resolved
from bindings.optimizer_public_words import stash_python_error, optimizer_python_error, PYTHON_ERROR_PENDING


def _setter_plan(model: PythonObject, fields: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var cls = py_type(model)
    if py_type(cls) is not b.type:
        raise Error("TypeError: native parameter setters cannot invoke Python metaclass hooks")
    var owner = py_get(context["plans"], cls)
    if owner is None:
        var setter = _resolved(cls, "set_params", context["missing"])
        if setter is context["missing"]:
            raise Error("AttributeError: parameter value has no set_params")
        raise Error("TypeError: Python parameter callbacks are not accepted; use a connected native parameter plan")
    var plan = owner["constructor"]
    var hooks = plan["hooks"]
    for i in range(len(hooks)):  # driver-loop: exact connected constructor/method/storage identities
        if py_get(_class_fields(hooks[i][0]), hooks[i][1]) is not hooks[i][2]:
            raise Error("TypeError: native parameter constructor or storage hooks changed after connection")
    var ancestry = py_call(b.type.__dict__["__mro__"].__get__, cls)
    if len(ancestry) != len(plan["mro"]):
        raise Error("TypeError: native parameter class ancestry changed")
    for i in range(len(ancestry)):  # driver-loop: fixed class ancestry
        if ancestry[i] is not plan["mro"][i]:
            raise Error("TypeError: native parameter class ancestry changed")
    var protected = plan["protected"]
    for i in range(len(protected)):  # driver-loop: instance protocols are never product callbacks
        if py_contains(fields, protected[i]):
            raise Error("TypeError: native parameter setters refuse instance protocol overrides")
    return owner


def _release_admit(value: PythonObject, context: PythonObject, seen: PythonObject) raises:
    var b = native_loaded_module("builtins")
    var todo = Python.list()
    _ = py_call(todo.append, value)
    var cursor = 0
    while cursor < len(todo):  # driver-loop: iterative ownership walk, including deeply nested/cyclic native data
        var item = todo[cursor]
        cursor += 1
        # builtin id() emits an audit event and can run a Python audit hook.
        # Native pointer identity provides the same cycle key without it.
        var identity = PythonObject(Int(item._obj_ptr))
        if py_contains(seen, identity):
            continue
        # Exact builtin identity integers never invoke an object's hash/eq.
        # Retaining each owner also postpones decrefs until admission finishes.
        py_setitem(seen, identity, item)
        var kind = py_type(item)
        native_release_weakrefs(item, context["ownership"])
        var finalizer = _resolved(kind, "__del__", context["missing"])
        if finalizer is not context["missing"]:
            raise Error("TypeError: native parameter reset cannot execute Python finalizers")
        if kind is b.list or kind is b.tuple:
            for i in range(len(item)):  # driver-loop: ordinary container ownership before references are dropped
                _ = py_call(todo.append, item[i])
        elif kind is b.dict:
            var keys = py_call(b.list, item.keys())
            var values = py_call(b.list, item.values())
            for i in range(len(keys)):  # driver-loop: no user-key hashing or item protocol
                _ = py_call(todo.append, keys[i])
                _ = py_call(todo.append, values[i])
        elif py_get(context["plans"], kind) is not None:
            # The connected plan retains this class and pins its ordinary
            # storage/lifecycle hooks before following estimator ownership.
            var fields = _fields(item)
            _ = _setter_plan(item, fields, context)
            _ = py_call(todo.append, fields)
        else:
            # Shared admission covers native arrays, NumPy dtype metadata and
            # object-array owners; arbitrary Python lifecycle objects refuse.
            native_release_admit(item, context["ownership"])


def _set_inner(model: PythonObject, params: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    if py_type(params) is not b.dict:
        raise Error("TypeError: native parameter keywords require an ordinary dict")
    # The established empty-update contract is an exact identity/no-op.
    if len(params) == 0:
        return model
    if not Bool(py=context["ready"]):
        raise Error("RuntimeError: native parameter constructor plans are not connected")
    var fields = _fields(model)
    var owner = _setter_plan(model, fields, context)
    var values = _read(model, False, context["parameter_metadata"], Python.list())
    var keys = py_call(b.list, params.keys())
    var nested = Python.dict()
    for i in range(len(keys)):  # driver-loop: validate every root name before any nested or parent mutation
        var key = keys[i]
        if py_type(key) is not b.str:
            raise Error("TypeError: native parameter names must be exact strings")
        var parts = py_call(key.partition, "__")
        var name = parts[0]
        if not py_contains(values, name):
            raise Error("ValueError: Invalid parameter " + String(py_call(b.repr, name)) + " for estimator "
                        + String(owner["class_name"]) + ". Valid parameters are: "
                        + String(py_call(b.repr, py_call(b.sorted, values))) + ".")
        if String(parts[1]) == "":
            py_setitem(values, name, params[key])
        else:
            if not py_contains(nested, name):
                py_setitem(nested, name, Python.dict())
            py_setitem(nested[name], parts[2], params[key])
    var names = py_call(b.list, nested.keys())
    for i in range(len(names)):  # driver-loop: nested updates retain the original child object and argument order
        _ = _set(values[names[i]], nested[names[i]], context)
    var constructor = owner["constructor"]
    var parameters = constructor["parameters"]
    var replacement = native_storage_allocate(py_type(model))
    var fresh = _fields(replacement)
    for i in range(len(parameters)):  # driver-loop: native field-only constructor plan, borrowing every parameter
        py_setitem(fresh, parameters[i], values[parameters[i]])
    var fitted = owner["fitted_fields"]
    var seen = Python.dict()
    classical_connect_numpy(context["ownership"])
    for i in range(len(fitted)):  # driver-loop: prove removed owned references cannot run Python finalizers
        if py_contains(fields, fitted[i]):
            _release_admit(fields[fitted[i]], context, seen)
    for i in range(len(parameters)):  # driver-loop: changed constructor references before committing reset
        if py_contains(fields, parameters[i]) and fields[parameters[i]] is not fresh[parameters[i]]:
            _release_admit(fields[parameters[i]], context, seen)
    for i in range(len(fitted)):  # driver-loop: remove only native-owned fitted fields; preserve unrelated user data
        if py_contains(fields, fitted[i]):
            _ = py_call(fields.pop, fitted[i])
    for i in range(len(parameters)):  # driver-loop: commit the constructor state to the same estimator object
        py_setitem(fields, parameters[i], fresh[parameters[i]])
    py_setitem(fields, "_mojolearn_params_reset", fitted)
    return model


def _set(model: PythonObject, params: PythonObject, context: PythonObject) raises -> PythonObject:
    ref cpy = Python().cpython()
    var label = String(" while setting estimator parameters")
    if cpy.lib.get_function[Int32]("Py_EnterRecursiveCall")(label.as_c_string_slice().unsafe_ptr()) != 0:
        var failure = stash_python_error()
        raise failure^
    try:
        var result = _set_inner(model, params, context)
        cpy.lib.get_function[NoneType]("Py_LeaveRecursiveCall")()
        return result
    except error:
        cpy.lib.get_function[NoneType]("Py_LeaveRecursiveCall")()
        raise error^


def params_setter_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=args_ptr)
        if len(args) != 3:
            raise Error("TypeError: parameter setter expects model, keywords and native context")
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
