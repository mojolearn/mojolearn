# SPDX-License-Identifier: Apache-2.0
# Shared fitted-state and compiled-binary metadata. Import time connects the
# exact read-back entries; a public operation never calls Python _bind,
# default_mode, getattr hooks, or a Python truth/string conversion protocol.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_loaded_module import native_loaded_module
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_type, py_get, py_contains
from bindings.optimizer_public_words import stash_python_error, optimizer_python_error, PYTHON_ERROR_PENDING


def _native_read(entry: PythonObject) raises -> PythonObject:
    ref cpy = Python().cpython()
    if py_type(entry) is not py_type(native_loaded_module("builtins").len):
        raise Error("TypeError: binary metadata read-back must be a compiled native entry")
    var args = Python.tuple()
    var value = cpy.PyObject_Call(entry._obj_ptr, args._obj_ptr, PyObjectPtr())
    if not value:
        var failure = stash_python_error()
        raise failure^
    return PythonObject(from_owned=value)


def _fields(model: PythonObject) raises -> PythonObject:
    ref cpy = Python().cpython()
    var b = native_loaded_module("builtins")
    var cls = py_type(model)
    if py_type(cls) is not b.type:
        raise Error("TypeError: native metadata refuses Python metaclass protocols")
    var ancestry = py_call(b.type.__dict__["__mro__"].__get__, cls)
    for i in range(len(ancestry)):  # driver-loop: refuse dictionary protocol overrides before storage access
        var namespace = _class_fields(ancestry[i])
        if py_contains(namespace, "__dict__"):
            var descriptor = namespace["__dict__"]
            if (py_type(descriptor) is not py_type(b.type.__dict__["__dict__"])
                    or descriptor.__objclass__ is not ancestry[i] or String(descriptor.__name__) != "__dict__"):
                raise Error("TypeError: native metadata refuses Python dictionary properties; use ordinary instance storage")
            break
    # This documented accessor reads the actual native instance dictionary;
    # GenericGetAttr("__dict__") could invoke a user-defined property.
    var raw = cpy.lib.get_function[PyObjectPtr]("PyObject_GenericGetDict")(model._obj_ptr, PyObjectPtr())
    if not raw:
        raise cpy.unsafe_get_error()
    var fields = PythonObject(from_owned=raw)
    if py_type(fields) is not b.dict:
        raise Error("TypeError: native metadata requires an ordinary instance dictionary")
    var keys = py_call(b.list, fields.keys())
    for i in range(len(keys)):  # driver-loop: dictionary keys before any keyed read
        if py_type(keys[i]) is not b.str:
            raise Error("TypeError: native metadata requires exact string state keys")
    return fields


def _truth(value: PythonObject) raises -> Bool:
    var b = native_loaded_module("builtins")
    var kind = py_type(value)
    if value is None:
        return False
    if (kind is b.bool or kind is b.int or kind is b.float or kind is b.str
            or kind is b.bytes or kind is b.list or kind is b.tuple or kind is b.dict):
        return Bool(py=value)
    raise Error("TypeError: native fitted-state flags cannot invoke Python boolean protocols")


def _fitted(model: PythonObject) raises -> PythonObject:
    var fields = _fields(model)
    if _truth(py_get(fields, "_mojolearn_fitted")) or _truth(py_get(fields, "_mojolearn_fitting")):
        return PythonObject(True)
    var names = py_call(native_loaded_module("builtins").list, fields.keys())
    for i in range(len(names)):  # driver-loop: fitted public attribute names, including legacy models
        var name = String(names[i])
        if name.endswith("_") and not name.startswith("_") and fields[names[i]] is not None:
            return PythonObject(True)
    return PythonObject(False)


def _class_fields(cls: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var fields = py_call(b.type.__dict__["__dict__"].__get__, cls)
    var names = py_call(b.list, fields.keys())
    for i in range(len(names)):  # driver-loop: native class namespace keys before metadata reads
        if py_type(names[i]) is not b.str:
            raise Error("TypeError: native metadata requires exact string class keys")
    return fields


def _plan(model: PythonObject, state: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var cls = py_type(model)
    if py_type(cls) is not b.type:
        raise Error("TypeError: native metadata refuses Python metaclass protocols")
    if py_contains(state, "_bind") or py_contains(state, "_BINDING"):
        raise Error("TypeError: native binary metadata requires its import-connected class binding; Python binding overrides are not accepted")
    var ancestry = py_call(b.type.__dict__["__mro__"].__get__, cls)
    var binding = PythonObject(None)
    var bind = PythonObject(None)
    var plan = PythonObject(None)
    for i in range(len(ancestry)):  # driver-loop: resolve the connected native route through ordinary subclasses
        if py_type(ancestry[i]) is not b.type:
            raise Error("TypeError: native metadata refuses Python metaclass protocols")
        var fields = _class_fields(ancestry[i])
        if binding is None and py_contains(fields, "_BINDING"):
            binding = fields["_BINDING"]
        if bind is None and py_contains(fields, "_bind"):
            bind = fields["_bind"]
        if plan is None:
            plan = py_get(context["plans"], ancestry[i])
    if plan is None:
        raise Error("RuntimeError: this estimator's native binary metadata route was not connected")
    if py_type(binding) is not b.str or binding != plan["binding"] or bind is not plan["bind"]:
        raise Error("TypeError: estimator binding metadata changed after native connection")
    return plan


def _mode(value: PythonObject) raises -> String:
    if py_type(value) is not native_loaded_module("builtins").str:
        raise Error("ValueError: numeric_mode must be fast, identical or None")
    var result = String(py_call(py_call(value.strip).lower))
    if result != "identical" and result != "fast":
        raise Error("ValueError: numeric_mode must be fast, identical or None")
    return result


def _configuration_value(model: PythonObject, fields: PythonObject, name: String) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var cls = py_type(model)
    var ancestry = py_call(b.type.__dict__["__mro__"].__get__, cls)
    var inherited = PythonObject(None)
    for i in range(len(ancestry)):  # driver-loop: preserve plain class defaults without invoking Python properties
        var namespace = _class_fields(ancestry[i])
        if py_contains(namespace, PythonObject(name)):
            inherited = namespace[name]
            var kind = py_type(inherited)
            if kind is native_loaded_module("types").MemberDescriptorType:
                if inherited.__objclass__ is not ancestry[i] or String(inherited.__name__) != name:
                    raise Error("TypeError: native mode metadata refuses a foreign slot descriptor")
                ref cpy = Python().cpython()
                var getter = inherited.__get__
                var args = Python.tuple(model, cls)
                var value = cpy.PyObject_Call(getter._obj_ptr, args._obj_ptr, PyObjectPtr())
                if not value:
                    if cpy.lib.get_function[Int32]("PyErr_ExceptionMatches")(b.AttributeError._obj_ptr) != 0:
                        cpy.lib.get_function[NoneType]("PyErr_Clear")()
                        return PythonObject(None)
                    var failure = stash_python_error()
                    raise failure^
                return PythonObject(from_owned=value)
            var value_ancestry = py_call(b.type.__dict__["__mro__"].__get__, kind)
            for j in range(len(value_ancestry)):  # driver-loop: refuse descriptor hooks before attribute access
                if py_contains(_class_fields(value_ancestry[j]), "__get__"):
                    raise Error("TypeError: native mode metadata cannot invoke Python properties; store " + name + " as a native value")
            break
    if py_contains(fields, PythonObject(name)):
        return fields[name]
    return inherited


def _readback(model: PythonObject, operation: String, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var fields = _fields(model)
    var plan = _plan(model, fields, context)
    var requested = _configuration_value(model, fields, "numeric_mode")
    var route = String(plan["route"])
    var selected = String()
    if route == "host" or route == "host_gp":
        if requested is not None and (py_type(requested) is not b.str or String(requested) != "identical"):
            raise Error("ValueError: this host estimator runs identical only")
        selected = "identical"
    else:
        if requested is None:
            var defaults = context["backend_fields"]
            if py_type(defaults) is not b.dict:
                raise Error("TypeError: native default mode requires its connected module dictionary")
            var names = py_call(b.list, defaults.keys())
            for i in range(len(names)):  # driver-loop: reject foreign equality hooks before reading mutable process metadata
                if py_type(names[i]) is not b.str:
                    raise Error("TypeError: native default mode requires exact string module keys")
            selected = _mode(defaults["_DEFAULT_MODE"])
        else:
            selected = _mode(requested)
        if route == "forest":
            var captured = _configuration_value(model, fields, "_fit_numeric_mode")
            if captured is not None:
                var locked = _mode(captured)
                if requested is not None and selected != locked:
                    raise Error("ValueError: this forest was fitted with numeric_mode=" + locked
                                + "; refit or use set_params before changing numeric_mode")
                selected = locked
    var connection = plan["modes"][selected]
    var error = py_get(connection, "error")
    if error is not None:
        # Preserve the exact refusal recorded at import without invoking an
        # exception constructor, __str__, or an optional Python adapter.
        ref cpy = Python().cpython()
        var kind = py_type(error)
        cpy.lib.get_function[NoneType]("PyErr_SetObject")(kind._obj_ptr, error._obj_ptr)
        var failure = stash_python_error()
        raise failure^
    var mode_entry = connection["mode"]
    var mode_value = connection["legacy_mode"]
    if mode_entry is not None:
        if py_type(mode_entry) is not py_type(b.len):
            raise Error("TypeError: binary numeric mode read-back must be a compiled native entry")
        var code = _native_read(mode_entry)
        if py_type(code) is not b.int or not py_contains(context["mode_codes"], code):
            raise Error("RuntimeError: compiled binary returned an unknown numeric mode")
        mode_value = context["mode_codes"][code]
        if String(mode_value) != selected:
            raise Error("RuntimeError: compiled binary numeric mode does not match the selected mode")
    if operation == "numeric_mode_used":
        return mode_value
    var entry = connection["vendor"]
    if entry is None:
        return PythonObject(None)
    if py_type(entry) is not py_type(b.len):
        raise Error("TypeError: binary vendor read-back must be a compiled native entry")
    var value = _native_read(entry)
    var kind = py_type(value)
    if not (kind is b.str or kind is b.int or kind is b.float or kind is b.bool or value is None):
        raise Error("TypeError: binary vendor metadata cannot invoke Python string protocols")
    return py_call(b.str, value)


def mode_metadata_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=args_ptr)
        if len(args) != 3 or py_type(args[1]) is not py_type(PythonObject("")):
            raise Error("TypeError: mode metadata expects model, operation and native context")
        var operation = String(args[1])
        var result = PythonObject(None)
        if operation == "is_fitted":
            result = _fitted(args[0])
        elif operation == "numeric_mode_used" or operation == "vendor_used":
            result = _readback(args[0], operation, args[2])
        else:
            raise Error("ValueError: unknown native mode metadata operation")
        return result.steal_data()
    except error:
        if String(error) == PYTHON_ERROR_PENDING:
            return optimizer_python_error(error)
        if String(error).startswith("ValueError:"):
            return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
        if String(error).startswith("ImportError:"):
            return raise_python_exception(error_without_class(error, "ImportError"), ExceptionType("PyExc_ImportError"))
        if String(error).startswith("ModuleNotFoundError:"):
            return raise_python_exception(error_without_class(error, "ModuleNotFoundError"), ExceptionType("PyExc_ModuleNotFoundError"))
        if String(error).startswith("AttributeError:"):
            return raise_python_exception(error_without_class(error, "AttributeError"), ExceptionType("PyExc_AttributeError"))
        if String(error).startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
