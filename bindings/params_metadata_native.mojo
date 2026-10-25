# SPDX-License-Identifier: Apache-2.0
# Native get_params metadata. Parameter values are borrowed references, never
# cloned, converted, or passed through Python getter/descriptor protocols.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_loaded_module import native_loaded_module
from bindings.native_error_class import error_without_class
from bindings.native_instance_item import native_instance_item
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.mode_metadata_native import _fields, _class_fields, _truth
from bindings.optimizer_public_words import stash_python_error, optimizer_python_error, PYTHON_ERROR_PENDING


def _resolved(cls: PythonObject, name: PythonObject, missing: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var ancestry = py_call(b.type.__dict__["__mro__"].__get__, cls)
    for i in range(len(ancestry)):  # driver-loop: raw class namespaces, without descriptor calls
        var fields = _class_fields(ancestry[i])
        if py_contains(fields, name):
            return fields[name]
    return missing


def _connected_plan(cls: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var ancestry = py_call(b.type.__dict__["__mro__"].__get__, cls)
    for i in range(len(ancestry)):  # driver-loop: find the connected base without calling a Python subclass protocol
        var plan = py_get(context["plans"], ancestry[i])
        if plan is not None:
            return plan
    return PythonObject(None)


def _loaded_data_class(module_name: String, name: String) raises -> PythonObject:
    var sys = native_loaded_module("sys")
    var b = native_loaded_module("builtins")
    var module = py_get(sys.modules, PythonObject(module_name))
    if module is None or py_type(module) is not py_type(sys):
        return PythonObject(None)
    ref cpy = Python().cpython()
    var raw = cpy.lib.get_function[PyObjectPtr]("PyModule_GetDict")(module._obj_ptr)
    if not raw:
        raise cpy.unsafe_get_error()
    var fields = PythonObject(from_borrowed=raw)
    var names = py_call(b.list, fields.keys())
    for i in range(len(names)):  # driver-loop: native module metadata before a keyed read
        if py_type(names[i]) is not b.str:
            raise Error("TypeError: native data metadata requires exact string module keys")
    var cls = py_get(fields, PythonObject(name))
    return cls if cls is not None and py_type(cls) is b.type else PythonObject(None)


def _plan(model: PythonObject, fields: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var cls = py_type(model)
    if py_type(cls) is not b.type:
        raise Error("TypeError: native parameters refuse Python metaclass protocols")
    var plan = _connected_plan(cls, context)
    if plan is None:
        if py_contains(context["pending"], cls):
            raise Error("RuntimeError: " + String(context["pending"][cls]))
        raise Error("TypeError: Python parameter callbacks are not accepted; use a connected native estimator")
    var pinned = plan["pinned"]
    for i in range(len(pinned)):  # driver-loop: verify connected method/parameter descriptor identities
        if _resolved(cls, pinned[i][0], context["missing"]) is not pinned[i][1]:
            raise Error("TypeError: native parameter methods or descriptors changed after connection")
    for name in ["get_params", "_get_param_names", "__getattribute__", "__getattr__"]:  # driver-loop: refuse instance protocol overrides before lookup
        if py_contains(fields, PythonObject(name)):
            raise Error("TypeError: native parameters cannot invoke instance protocol overrides")
    return plan


def _value(model: PythonObject, fields: PythonObject, name: PythonObject,
           context: PythonObject, missing_error: String) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var cls = py_type(model)
    var value = _resolved(cls, name, context["missing"])
    if value is not context["missing"]:
        var kind = py_type(value)
        if kind is native_loaded_module("types").MemberDescriptorType:
            # Exact CPython member descriptors read native slot storage only.
            var getter = value.__get__
            var args = Python.tuple(model, cls)
            ref cpy = Python().cpython()
            var raw = cpy.PyObject_Call(getter._obj_ptr, args._obj_ptr, PyObjectPtr())
            if not raw:
                var failure = stash_python_error()
                raise failure^
            return PythonObject(from_owned=raw)
        if _resolved(kind, PythonObject("__get__"), context["missing"]) is not context["missing"]:
            raise Error("TypeError: native parameters cannot invoke Python properties; store constructor parameters as native fields")
    if py_contains(fields, name):
        # See native_instance_item for the retained Mojo subscript failure.
        return native_instance_item(fields, name)
    if value is not context["missing"]:
        return value
    raise Error(missing_error + ": constructor parameter is not stored under its own name: " + String(name))


def _nested(value: PythonObject, context: PythonObject) raises -> Bool:
    var b = native_loaded_module("builtins")
    var kind = py_type(value)
    # A class is a parameter value, never an estimator instance to expand.
    ref cpy = Python().cpython()
    if cpy.lib.get_function[Int32]("PyType_IsSubtype")(kind._obj_ptr, b.type._obj_ptr) != 0:
        return False
    var connected = _connected_plan(kind, context) is not None
    var getter = _resolved(kind, PythonObject("get_params"), context["missing"])
    if (getter is not context["missing"] and not connected
            and _resolved(py_type(getter), "__get__", context["missing"]) is not context["missing"]):
        raise Error("TypeError: native parameters cannot invoke unconnected parameter descriptors")
    var missing_hook = _resolved(kind, PythonObject("__getattr__"), context["missing"])
    if missing_hook is not context["missing"] and not connected:
        # Exact pandas storage objects are data parameters. Their column/index
        # fallback is not an estimator protocol: keep the object without
        # evaluating Python label lookup or interpreting a data cell as a
        # callback. An actual class/instance get_params override is still
        # checked above/below, before any invocation.
        var series = _loaded_data_class("pandas.core.series", "Series")
        if ((kind is not _loaded_data_class("pandas.core.frame", "DataFrame") and kind is not series) or series is None
                or _resolved(series, "__call__", context["missing"]) is not context["missing"]):
            raise Error("TypeError: native parameters cannot probe Python attribute hooks")
    var lookup = _resolved(kind, PythonObject("__getattribute__"), context["missing"])
    if (lookup is not context["missing"]
            and py_type(lookup) is not py_type(b.object.__dict__["__getattribute__"])
            and not connected):
        raise Error("TypeError: native parameters cannot probe Python attribute hooks")
    # A plain object's instance get_params override must not silently become
    # a native protocol. Scalar/array native types without dictionaries never
    # receive arbitrary Python attribute lookup here.
    var dictionary = _resolved(kind, PythonObject("__dict__"), context["missing"])
    if dictionary is not context["missing"]:
        var fields = _fields(value)
        if py_contains(fields, "get_params"):
            getter = fields["get_params"]
    if getter is context["missing"] or getter is None:
        return False
    return cpy.lib.get_function[Int32]("PyCallable_Check")(getter._obj_ptr) != 0


def _read_inner(model: PythonObject, deep: Bool, context: PythonObject,
          ancestors: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    for i in range(len(ancestors)):  # driver-loop: preserve cycle refusal without unbounded native recursion
        if model is ancestors[i]:
            raise Error("RecursionError: cyclic estimator parameter graph")
    var path = py_call(b.list, ancestors)
    _ = py_call(path.append, model)
    var fields = _fields(model)
    var plan = _plan(model, fields, context)
    var kind = String(plan["kind"])
    var out = Python.dict()
    if kind == "splitter":
        var names = py_call(b.list, fields.keys())
        for i in range(len(names)):  # driver-loop: splitter's public configuration fields
            if not String(names[i]).startswith("_"):
                py_setitem(out, names[i], fields[names[i]])
        return out
    if kind == "repeated":
        py_setitem(out, "n_repeats", _value(model, fields, "n_repeats", context, "AttributeError"))
        py_setitem(out, "random_state", _value(model, fields, "random_state", context, "AttributeError"))
        var values = _value(model, fields, "cvargs", context, "AttributeError")
        if py_type(values) is not b.dict:
            raise Error("TypeError: native repeated-splitter options require an ordinary dict")
        var names = py_call(b.list, values.keys())
        for i in range(len(names)):  # driver-loop: native keyword storage, preserving references and insertion order
            if py_type(names[i]) is not b.str:
                raise Error("TypeError: native repeated-splitter option names must be strings")
            if py_contains(out, names[i]):
                raise Error("TypeError: repeated splitter has a duplicate constructor parameter")
            py_setitem(out, names[i], values[names[i]])
        return out
    if kind == "forest":
        var has = py_get(fields, "_has_constructor_parameters")
        if has is None:
            has = _resolved(py_type(model), "_has_constructor_parameters", PythonObject(False))
        if not _truth(has):
            raise Error("Exception: This inference-only forest archive has no constructor parameters; construct a new estimator for cloning/refitting")
    if kind == "gbdt" and _resolved(py_type(model), "__init__", context["missing"]) is not plan["gbdt_init"]:
        raise Error("TypeError: GBDT adapter subclasses with custom constructors need their own parameter protocol")
    var names = plan["names"]
    var expand = deep and Bool(py=plan["recurse"])
    for i in range(len(names)):  # driver-loop: connected parameter schema in its established order
        var name = names[i]
        var field = PythonObject("_resident_wanted") if kind == "optimizer" and String(name) == "resident" else name
        var value = PythonObject(None)
        if Bool(py=plan["fields_only"]):
            if not py_contains(fields, field):
                raise Error(String(plan["missing_error"]) + ": " + String(field))
            value = fields[field]
        else:
            value = _value(model, fields, field, context, String(plan["missing_error"]))
        if expand and _nested(value, context):
            var nested = _read(value, True, context, path)
            var keys = py_call(b.list, nested.keys())
            for j in range(len(keys)):  # driver-loop: flatten supported nested estimator names, retaining the same values
                py_setitem(out, PythonObject(String(name) + "__" + String(keys[j])), nested[keys[j]])
        py_setitem(out, name, value)
    return out


def _read(model: PythonObject, deep: Bool, context: PythonObject,
          ancestors: PythonObject) raises -> PythonObject:
    # The documented extension recursion guard follows the caller's actual
    # interpreter limit; a long parameter graph cannot overflow a native stack.
    ref cpy = Python().cpython()
    var label = String(" while reading estimator parameters")
    if cpy.lib.get_function[Int32]("Py_EnterRecursiveCall")(label.as_c_string_slice().unsafe_ptr()) != 0:
        var failure = stash_python_error()
        raise failure^
    try:
        var result = _read_inner(model, deep, context, ancestors)
        cpy.lib.get_function[NoneType]("Py_LeaveRecursiveCall")()
        return result
    except error:
        cpy.lib.get_function[NoneType]("Py_LeaveRecursiveCall")()
        raise error^


def params_metadata_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=args_ptr)
        if len(args) != 3:
            raise Error("TypeError: parameter metadata expects model, deep and native context")
        var result = _read(args[0], _truth(args[1]), args[2], Python.list())
        return result.steal_data()
    except error:
        if String(error) == PYTHON_ERROR_PENDING:
            return optimizer_python_error(error)
        if String(error).startswith("RecursionError:"):
            return raise_python_exception(error_without_class(error, "RecursionError"), ExceptionType("PyExc_RecursionError"))
        if String(error).startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        if String(error).startswith("AttributeError:"):
            return raise_python_exception(error_without_class(error, "AttributeError"), ExceptionType("PyExc_AttributeError"))
        if String(error).startswith("KeyError:"):
            return raise_python_exception(error_without_class(error, "KeyError"), ExceptionType("PyExc_KeyError"))
        if String(error).startswith("Exception:"):
            return raise_python_exception(error_without_class(error, "Exception"), ExceptionType("PyExc_Exception"))
        return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
