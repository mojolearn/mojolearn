# SPDX-License-Identifier: Apache-2.0
# Native public model attribute lookup. Only connected function/native
# descriptors may bind; arbitrary Python properties/hooks are never called.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.py_balanced import py_call, py_type, py_get, py_contains
from bindings.native_error_class import error_without_class
from bindings.sequential_admission import sequential_admit_value
from bindings.sequential_inputs import sequential_input


def _type_fields(cls: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    return py_call(b.type.__dict__["__dict__"].__get__, cls)


def _type_mro(cls: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    return py_call(b.type.__dict__["__mro__"].__get__, cls)


def _descriptor_slot(cls: PythonObject, name: String) raises -> Bool:
    var mro = _type_mro(cls)
    for k in range(len(mro)):  # driver-loop: native descriptor class metadata
        if py_contains(_type_fields(mro[k]), name):
            return True
    return False


def _public_plan(model: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var cls = model if py_type(model) is b.type else py_type(model)
    var layouts = py_get(context, "classical_public_layouts")
    if layouts is None:
        raise Error("RuntimeError: native public model metadata was not connected")
    var plan = py_get(layouts, cls)
    if plan is None:
        raise Error("TypeError: native public model requires an exact connected class")
    return plan


def classical_check_layout(model: PythonObject, context: PythonObject) raises:
    # Existing private-owner callers have no installer metadata. Once public
    # installation is requested, the complete layout is mandatory and checked.
    if py_get(context, "classical_public_layouts") is None:
        if Bool(py=py_get(context, "classical_public_installed", False)):
            raise Error("RuntimeError: native public model metadata was removed")
        return
    var b = Python.import_module("builtins")
    var cls = model if py_type(model) is b.type else py_type(model)
    var plan = _public_plan(model, context)
    var mro = _type_mro(cls)
    var connected = plan["mro"]
    if len(mro) != len(connected):
        raise Error("TypeError: native public model class layout changed")
    var found_fields = False
    for k in range(len(mro)):  # driver-loop: exact connected class ancestry
        if mro[k] is not connected[k]:
            raise Error("TypeError: native public model class ancestry changed")
        if py_contains(_type_fields(mro[k]), "__getattr__"):
            # CPython otherwise invokes this hook after AttributeError from
            # our native getter. Refuse while we still control the boundary.
            raise Error("TypeError: Python model __getattr__ hooks are not accepted")
        if py_contains(_type_fields(mro[k]), "__del__"):
            raise Error("TypeError: Python model finalizer hooks are not accepted")
        if not found_fields:
            var descriptor = py_get(_type_fields(mro[k]), "__dict__")
            if descriptor is not None:
                if descriptor is not plan["fields_descriptor"]:
                    raise Error("TypeError: Python model dictionary descriptors are not accepted")
                found_fields = True
    if not found_fields:
        raise Error("TypeError: native public model dictionary descriptor is missing")
    if py_type(model) is not b.type:
        var fields = py_call(plan["fields_get"], model, cls)
        var keys = py_call(b.list, fields.keys())
        for k in range(len(keys)):  # driver-loop: native instance attribute names
            if py_type(keys[k]) is not b.str:
                # Refuse before a dict lookup can compare a malicious key
                # against one of the native implementation's string names.
                raise Error("TypeError: native model dictionary requires exact string keys")


def classical_getattribute(model: PythonObject, name: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    if py_type(name) is not b.str:
        raise Error("TypeError: native model attribute name must be an exact string")
    classical_check_layout(model, context)
    var plan = _public_plan(model, context)
    var cls = py_type(model)
    var member = PythonObject(None)
    var present = False
    var mro = plan["mro"]
    for k in range(len(mro)):  # driver-loop: native class attribute lookup order
        var fields = _type_fields(mro[k])
        if py_contains(fields, name):
            member = fields[name]
            present = True
            break
    var binding = py_get(plan["bindings"], name)
    var known = binding is not None and member is binding[0]
    var member_type = py_type(member)
    var descriptor = present and _descriptor_slot(member_type, "__get__")
    var data_descriptor = descriptor and (_descriptor_slot(member_type, "__set__") or _descriptor_slot(member_type, "__delete__"))
    if data_descriptor:
        if not known:
            raise Error("TypeError: Python model property/descriptor hooks are not accepted")
        return py_call(binding[1], model, cls)
    var fields = py_call(plan["fields_get"], model, cls)
    if py_contains(fields, name):
        var value = fields[name]
        if (len(name) > 0 and Bool(py=py_call(name.endswith, "_")) and not Bool(py=py_call(name.startswith, "_"))
                and py_type(value) is context["array_class"] and py_get(context, "ndarray") is not None):
            # Native slot getters + ndarray C type wrap the same owner/storage.
            # Never invoke Array.__array__, properties, or output Python code.
            # Without NumPy the Array itself is the fitted attribute.
            sequential_admit_value(value, context)
            return sequential_input(value, context)
        return value
    if not present:
        raise Error("AttributeError: native model has no attribute '" + String(name) + "'")
    if descriptor:
        if not known:
            raise Error("TypeError: unconnected Python model descriptors are not accepted")
        return py_call(binding[1], model, cls)
    return member


def classical_connect_numpy(context: PythonObject) raises:
    """Connect the NumPy admission entries the first time a classical entry
    runs with NumPy loaded after a NumPy-free connection (NumPy imported
    after mojolearn). An ndarray exists only once NumPy is loaded, so the
    loaded module decides; NumPy is never imported here."""
    if py_get(context, "ndarray") is not None:
        return
    var connect = py_get(context, "classical_connect_numpy")
    if connect is None:
        return
    if py_get(context["modules"], "numpy") is not None:
        _ = py_call(connect, context)


def _attribute_context(model: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var host = py_get(context, "classical_host_context")
    if host is not None and py_contains(host["classical_kinds"], py_type(model)):
        return host
    return context


def classical_attribute_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=args_ptr)
        var model = args[1]
        var context = _attribute_context(model, args[0])
        classical_connect_numpy(context)
        var result = classical_getattribute(model, args[2], context)
        return result.steal_data()
    except error:
        if String(error).startswith("AttributeError:"):
            return raise_python_exception(error_without_class(error, "AttributeError"), ExceptionType("PyExc_AttributeError"))
        if String(error).startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)


def classical_attribute_descriptor_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        ref cpy = Python().cpython()
        var function = cpy.PyTuple_GetItem(args_ptr, 0)
        if not function:
            raise cpy.unsafe_get_error()
        var value = PythonObject(from_borrowed=function)
        var b = Python.import_module("builtins")
        var types = Python.import_module("types")
        if py_type(value) is not types.MethodType or py_type(value.__func__) is not types.BuiltinFunctionType:
            raise Error("native attribute descriptor requires a bound compiled entry")
        # Documented CPython API, present throughout package Python >=3.10.
        # Borrows `function`; owns one new reference returned directly to the
        # C caller. No stdlib patch, raw type-slot edit or refcount adjustment.
        # https://docs.python.org/3.10/c-api/method.html#c.PyInstanceMethod_New
        # Resolve against std.python's already selected interpreter library.
        # OwnedDLHandle.get_function retains the library borrow through the
        # call; an extension must not assume CPython symbols link globally.
        var make_descriptor = cpy.lib.get_function[PyObjectPtr]("PyInstanceMethod_New")
        var descriptor = make_descriptor(function)
        if not descriptor:
            raise cpy.unsafe_get_error()
        return descriptor
    except error:
        return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
