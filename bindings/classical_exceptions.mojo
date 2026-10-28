# SPDX-License-Identifier: Apache-2.0
# Native NotFittedError class selection preserves the existing optional sklearn
# relationship. No imports of sklearn or calls to Python exception factories.
from bindings.native_loaded_module import native_loaded_module
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_contains
from bindings.py_slice_native import py_slice
from bindings.classical_metadata import classical_is_fitted
from bindings.mode_metadata_native import _class_fields


def _exception_class(cls: PythonObject, first: PythonObject, second: PythonObject) raises:
    var bi = native_loaded_module("builtins")
    if py_type(cls) is not bi.type:
        raise Error("TypeError: NotFittedError requires a native metaclass")
    var bases = py_call(bi.type.__dict__["__bases__"].__get__, cls)
    if len(bases) != 2 or bases[0] is not first or bases[1] is not second:
        raise Error("TypeError: NotFittedError must retain its connected exception bases")
    # The class dictionary can change after import, including through a
    # non-string key whose equality hook would run during a keyed lookup.
    # Admit its exact keys before checking construction and attribute hooks.
    var fields = _class_fields(cls)
    var hooks = Python.tuple("__new__", "__init__", "__init_subclass__", "__getattribute__",
                             "__getattr__", "__setattr__", "__delattr__", "__del__", "args")
    for i in range(len(hooks)):  # driver-loop: fixed exception construction and storage hooks
        if py_contains(fields, hooks[i]):
            raise Error("TypeError: NotFittedError may not install Python runtime hooks")


def classical_not_fitted_class(context: PythonObject) raises -> PythonObject:
    var bi = native_loaded_module("builtins")
    var own = context["not_fitted_class"]
    _exception_class(own, bi.ValueError, bi.AttributeError)
    var module = py_get(native_loaded_module("sys").modules, "sklearn.exceptions")
    if module is None:
        py_setitem(context, "active_not_fitted_class", own)
        return own
    if py_type(module) is not context["native_module_type"]:
        raise Error("TypeError: fitted-state checks require the standard sklearn exceptions module")
    var fields = py_call(bi.object.__getattribute__, module, "__dict__")
    var names = py_call(bi.list, fields.keys())
    for i in range(len(names)):  # driver-loop: admit module names before dictionary lookup
        if py_type(names[i]) is not bi.str:
            raise Error("TypeError: sklearn exception metadata requires exact string keys")
    var base = py_get(fields, "NotFittedError")
    if base is None or base is own:
        py_setitem(context, "active_not_fitted_class", own)
        return own
    _exception_class(base, bi.ValueError, bi.AttributeError)
    var bridges = context["not_fitted_bridges"]
    if py_type(bridges) is not bi.dict:
        raise Error("TypeError: NotFittedError bridges require ordinary native dictionary storage")
    var keys = py_call(bi.list, bridges.keys())
    for i in range(len(keys)):  # driver-loop: bridge keys before native dictionary lookup
        if py_type(keys[i]) is not bi.type:
            raise Error("TypeError: NotFittedError bridge keys require native metaclasses")
    var bridge = py_get(bridges, base)
    if bridge is None:
        if (py_type(context["not_fitted_reduce"]) is not context["not_fitted_function_type"]
                or py_type(context["not_fitted_doc"]) is not bi.str):
            raise Error("TypeError: NotFittedError bridge metadata must retain its connected native layout")
        var namespace = Python.dict()
        py_setitem(namespace, "__module__", "mojolearn.exceptions")
        py_setitem(namespace, "__doc__", context["not_fitted_doc"])
        py_setitem(namespace, "__reduce__", context["not_fitted_reduce"])
        bridge = py_call(bi.type, "NotFittedError", Python.tuple(own, base), namespace)
        py_setitem(bridges, base, bridge)
    _exception_class(bridge, own, base)
    py_setitem(context, "active_not_fitted_class", bridge)
    return bridge


def classical_require_fitted(model: PythonObject, ridge: Bool, operation: String,
                             context: PythonObject) raises:
    if not Bool(py=classical_is_fitted(model)):
        _ = classical_not_fitted_class(context)
        raise Error("NotFittedError: This " + String(py_type(model).__name__) +
                    " instance is not fitted yet. Call 'fit' with appropriate arguments before using " + operation + ".")


def classical_error_restore(args: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = native_loaded_module("builtins")
    if len(args) != 1 or py_type(args[0]) is not bi.str:
        raise Error("TypeError: NotFittedError restoration requires a native message string")
    var cls = classical_not_fitted_class(context)
    return py_call(cls, args[0])


def classical_error_reduce(error: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = native_loaded_module("builtins")
    var cls = py_type(error)
    var own = context["not_fitted_class"]
    _exception_class(own, bi.ValueError, bi.AttributeError)
    var valid = cls is own
    var bridges = context["not_fitted_bridges"]
    if py_type(bridges) is not bi.dict:
        raise Error("TypeError: NotFittedError bridges require ordinary native dictionary storage")
    var keys = py_call(bi.list, bridges.keys())
    for k in range(len(keys)):  # driver-loop: admit all registry keys before any keyed read
        if py_type(keys[k]) is not bi.type:
            raise Error("TypeError: NotFittedError bridge keys require native metaclasses")
    for k in range(len(keys)):  # driver-loop: over native-owned fitted-error bridge identities
        if cls is bridges[keys[k]]:
            _exception_class(keys[k], bi.ValueError, bi.AttributeError)
            _exception_class(cls, own, keys[k])
            valid = True
    if not valid:
        raise Error("TypeError: error serialization requires a connected NotFittedError")
    var args = py_call(bi.BaseException.__dict__["args"].__get__, error)
    return Python.tuple(context["not_fitted_restore"], py_slice(args, 0, 1))
