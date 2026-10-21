# SPDX-License-Identifier: Apache-2.0
# Native NotFittedError class selection preserves the existing optional sklearn
# relationship. No imports of sklearn or calls to Python exception factories.
from std.python import Python, PythonObject
from bindings.classical_metadata import classical_is_fitted


def classical_not_fitted_class(context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var own = context["not_fitted_class"]
    var module = Python.import_module("sys").modules.get("sklearn.exceptions")
    if module is None:
        context["active_not_fitted_class"] = own
        return own
    if bi.type(module) is not context["native_module_type"]:
        raise Error("TypeError: fitted-state checks require the standard sklearn exceptions module")
    var fields = bi.object.__getattribute__(module, "__dict__")
    var base = fields.get("NotFittedError")
    if base is None or base is own:
        context["active_not_fitted_class"] = own
        return own
    if bi.type(base) is not bi.type:
        raise Error("TypeError: sklearn NotFittedError requires a native metaclass")
    var class_fields = bi.type.__dict__["__dict__"].__get__(base)
    var bases = bi.type.__dict__["__bases__"].__get__(base)
    if bases != Python.tuple(bi.ValueError, bi.AttributeError):
        raise Error("TypeError: sklearn NotFittedError must retain its standard native exception bases")
    var hooks = Python.tuple("__new__", "__init__", "__init_subclass__", "__getattribute__", "__setattr__", "__del__")
    for k in range(len(hooks)):  # driver-loop: over exception class hooks that could enter Python
        if Bool(py=class_fields.__contains__(hooks[k])):
            raise Error("TypeError: sklearn NotFittedError may not install Python runtime hooks")
    var bridges = context["not_fitted_bridges"]
    var bridge = bridges.get(base)
    if bridge is None:
        var namespace = Python.dict()
        namespace["__module__"] = "mojolearn.exceptions"
        namespace["__doc__"] = context["not_fitted_doc"]
        namespace["__reduce__"] = context["not_fitted_reduce"]
        bridge = bi.type("NotFittedError", Python.tuple(own, base), namespace)
        bridges[base] = bridge
    context["active_not_fitted_class"] = bridge
    return bridge


def classical_require_fitted(model: PythonObject, ridge: Bool, operation: String,
                             context: PythonObject) raises:
    if not Bool(py=classical_is_fitted(model)):
        _ = classical_not_fitted_class(context)
        raise Error("NotFittedError: This " + ("Ridge" if ridge else "LinearRegression") +
                    " instance is not fitted yet. Call 'fit' with appropriate arguments before using " + operation + ".")


def classical_error_restore(args: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    if len(args) != 1 or bi.type(args[0]) is not bi.str:
        raise Error("TypeError: NotFittedError restoration requires a native message string")
    var cls = classical_not_fitted_class(context)
    return cls(args[0])


def classical_error_reduce(error: PythonObject, context: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var cls = bi.type(error)
    var valid = cls is context["not_fitted_class"]
    var classes = bi.list(context["not_fitted_bridges"].values())
    for k in range(len(classes)):  # driver-loop: over native-owned fitted-error bridge class identities
        valid = valid or cls is classes[k]
    if not valid:
        raise Error("TypeError: error serialization requires a connected NotFittedError")
    var args = bi.object.__getattribute__(error, "args")
    return Python.tuple(context["not_fitted_restore"], args[0:1])
