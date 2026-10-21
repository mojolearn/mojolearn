# SPDX-License-Identifier: Apache-2.0
# Native NotFittedError class selection preserves the existing optional sklearn
# relationship. No imports of sklearn or calls to Python exception factories.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_contains
from bindings.classical_metadata import classical_is_fitted


def classical_not_fitted_class(context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var own = context["not_fitted_class"]
    var module = py_get(Python.import_module("sys").modules, "sklearn.exceptions")
    if module is None:
        py_setitem(context, "active_not_fitted_class", own)
        return own
    if py_type(module) is not context["native_module_type"]:
        raise Error("TypeError: fitted-state checks require the standard sklearn exceptions module")
    var fields = py_call(bi.object.__getattribute__, module, "__dict__")
    var base = py_get(fields, "NotFittedError")
    if base is None or base is own:
        py_setitem(context, "active_not_fitted_class", own)
        return own
    if py_type(base) is not bi.type:
        raise Error("TypeError: sklearn NotFittedError requires a native metaclass")
    var class_fields = py_call(bi.type.__dict__["__dict__"].__get__, base)
    var bases = py_call(bi.type.__dict__["__bases__"].__get__, base)
    if bases != Python.tuple(bi.ValueError, bi.AttributeError):
        raise Error("TypeError: sklearn NotFittedError must retain its standard native exception bases")
    var hooks = Python.tuple("__new__", "__init__", "__init_subclass__", "__getattribute__", "__setattr__", "__del__")
    for k in range(len(hooks)):  # driver-loop: over exception class hooks that could enter Python
        if py_contains(class_fields, hooks[k]):
            raise Error("TypeError: sklearn NotFittedError may not install Python runtime hooks")
    var bridges = context["not_fitted_bridges"]
    var bridge = py_get(bridges, base)
    if bridge is None:
        var namespace = Python.dict()
        py_setitem(namespace, "__module__", "mojolearn.exceptions")
        py_setitem(namespace, "__doc__", context["not_fitted_doc"])
        py_setitem(namespace, "__reduce__", context["not_fitted_reduce"])
        bridge = py_call(bi.type, "NotFittedError", Python.tuple(own, base), namespace)
        py_setitem(bridges, base, bridge)
    py_setitem(context, "active_not_fitted_class", bridge)
    return bridge


def classical_require_fitted(model: PythonObject, ridge: Bool, operation: String,
                             context: PythonObject) raises:
    if not Bool(py=classical_is_fitted(model)):
        _ = classical_not_fitted_class(context)
        raise Error("NotFittedError: This " + String(context["ridge_name" if ridge else "linear_name"]) +
                    " instance is not fitted yet. Call 'fit' with appropriate arguments before using " + operation + ".")


def classical_error_restore(args: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    if len(args) != 1 or py_type(args[0]) is not bi.str:
        raise Error("TypeError: NotFittedError restoration requires a native message string")
    var cls = classical_not_fitted_class(context)
    return py_call(cls, args[0])


def classical_error_reduce(error: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var cls = py_type(error)
    var valid = cls is context["not_fitted_class"]
    var classes = py_call(bi.list, context["not_fitted_bridges"].values())
    for k in range(len(classes)):  # driver-loop: over native-owned fitted-error bridge class identities
        valid = valid or cls is classes[k]
    if not valid:
        raise Error("TypeError: error serialization requires a connected NotFittedError")
    var args = py_call(bi.object.__getattribute__, error, "args")
    return Python.tuple(context["not_fitted_restore"], args[0:1])
