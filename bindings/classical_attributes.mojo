# SPDX-License-Identifier: Apache-2.0
# Native public model attribute lookup. Only connected function/native
# descriptors may bind; arbitrary Python properties/hooks are never called.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_fields, py_setitem
from bindings.native_error_class import error_without_class
from bindings.sequential_admission import sequential_admit_value
from bindings.sequential_inputs import sequential_input
from bindings.native_numpy_metadata import native_numpy_type, native_numpy_module_fields


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
    if py_type(cls) is not py_type(py_type(PythonObject(0))):
        raise Error("TypeError: native public attributes cannot invoke Python metaclass hooks")
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
    var found_finalizer = False
    for k in range(len(mro)):  # driver-loop: exact connected class ancestry
        if mro[k] is not connected[k]:
            raise Error("TypeError: native public model class ancestry changed")
        if py_contains(_type_fields(mro[k]), "__getattr__"):
            # CPython otherwise invokes this hook after AttributeError from
            # our native getter. Refuse while we still control the boundary.
            raise Error("TypeError: Python model __getattr__ hooks are not accepted")
        if not found_finalizer and py_contains(_type_fields(mro[k]), "__del__"):
            # Explicit CPU archive classes replace device finalizers with a
            # connected native descriptor. Only the first resolved slot can
            # run; a shadowed GPU-base Python finalizer is never invoked.
            var native_finalizer = py_get(context, "host_archive_finalizer_descriptor")
            if native_finalizer is None or _type_fields(mro[k])["__del__"] is not native_finalizer:
                raise Error("TypeError: Python model finalizer hooks are not accepted")
            found_finalizer = True
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
        var keys = py_call(py_type(Python.list()), fields.keys())
        for k in range(len(keys)):  # driver-loop: native instance attribute names
            if py_type(keys[k]) is not b.str:
                # Refuse before a dict lookup can compare a malicious key
                # against one of the native implementation's string names.
                raise Error("TypeError: native model dictionary requires exact string keys")


def _classical_descriptor_get(binding: PythonObject, model: PythonObject, cls: PythonObject,
                              plan: PythonObject) raises -> PythonObject:
    # py_call's generic Mojo error keeps the text, not the CPython class.
    # An unset native member must remain AttributeError for getattr/hasattr.
    # Resolve all metadata/functions before the call can set an exception.
    ref cpy = Python().cpython()
    var getter = binding[1]
    var envelope = Python.tuple(model, cls)
    var attribute_error = plan["attribute_error"]
    var matches = cpy.lib.get_function[Int32]("PyErr_ExceptionMatches")
    var result = cpy.PyObject_Call(getter._obj_ptr, envelope._obj_ptr, PyObjectPtr())
    if not result:
        var missing = matches(attribute_error._obj_ptr) != 0
        var error = cpy.unsafe_get_error()
        if missing:
            raise Error("AttributeError: " + String(error))
        raise error^
    return PythonObject(from_owned=result)


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
        return _classical_descriptor_get(binding, model, cls, plan)
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
        return _classical_descriptor_get(binding, model, cls, plan)
    return member


def classical_connect_numpy(context: PythonObject) raises:
    """Connect loaded NumPy's native types without a Python setup callback."""
    if py_get(context, "ndarray") is not None:
        return
    var numpy = py_get(context["modules"], "numpy")
    if numpy is None:
        return
    if py_type(numpy) is not context["native_module_type"]:
        raise Error("TypeError: classical NumPy support requires the native NumPy module")
    # Read the loaded module dictionary directly; module __getattr__ is a
    # Python hook. All dtype construction below uses NumPy's C dtype type.
    var fields = native_numpy_module_fields(numpy)
    # C-created builtin values supply stable constructor/type identities.
    # These names may be replaced in builtins after package import; no
    # replacement becomes a callback or changes scalar admission metadata.
    var list_type = py_type(Python.list())
    var tuple_type = py_type(Python.tuple())
    var int_type = py_type(PythonObject(0))
    var float_type = py_type(PythonObject(0.0))
    var bool_type = py_type(PythonObject(True))
    # Check both factory identities before invoking or publishing either.
    # A late NumPy import must not turn first-call metadata setup into a call
    # to an arbitrary Python replacement of numpy.dtype or numpy.ndarray.
    var dtype_class = native_numpy_type("dtype", context)
    var ndarray = native_numpy_type("ndarray", context)
    var numeric = Python.dict()
    var classes = Python.dict()
    var scalars = py_call(list_type)
    var integers = py_call(list_type, Python.tuple(int_type))
    var kinds = Python.tuple("u", "i", "f", "b")
    var sizes = Python.tuple(1, 2, 4, 8)
    for kind in range(4):  # driver-loop: fixed native scalar ABI metadata
        for index in range(4):  # driver-loop: supported scalar widths
            var size = Int(py=sizes[index])
            if (kind == 2 and size == 1) or (kind == 3 and size != 1):
                continue
            var code = "?" if kind == 3 else String(kinds[kind]) + String(size)
            var dtype = py_call(dtype_class, code)
            py_setitem(numeric, Python.tuple(kinds[kind], size), dtype)
            py_setitem(classes, Python.tuple(kind, size), dtype)
            _ = py_call(scalars.append, dtype.type)
            if kind < 2:
                _ = py_call(integers.append, dtype.type)
    var values = py_call(list_type, context["native_value_types"])
    var owners = py_call(list_type, context["buffer_owner_types"])
    _ = py_call(values.append, ndarray)
    _ = py_call(owners.append, ndarray)
    var archive = Python.dict()
    var archive_codes = Python.tuple("<f4", "<f8", "<i4", "<i8", "<u4", "<u1", "<f2",
                                     "=f4", "=f8", "=i4", "=i8", "=u4", "|u1", "|b1")
    for i in range(len(archive_codes)):  # driver-loop: supported NPY header dtype metadata
        var code = archive_codes[i]
        # The archive's Boolean payload is represented by byte storage.
        var dtype = py_call(dtype_class, "u1" if String(code) == "|b1" else String(code))
        py_setitem(archive, code, dtype)
    var update = Python.dict(
        ndarray=ndarray, scalar_type=fields["generic"], integer=fields["integer"],
        scalar_types=py_call(tuple_type, scalars), integer_types=py_call(tuple_type, integers),
        native_value_types=py_call(tuple_type, values), buffer_owner_types=py_call(tuple_type, owners),
        numeric_dtypes=numeric, class_dtypes=classes, archive_npy_dtypes=archive,
        float_types=Python.tuple(float_type, fields["floating"]),
        int_types=Python.tuple(int_type, fields["integer"]), bool_types=Python.tuple(bool_type, fields["bool_"]),
        float32=py_call(dtype_class, "f4"), float64=py_call(dtype_class, "f8"),
        int32=py_call(dtype_class, "i4"), int64=py_call(dtype_class, "i8"),
        uint64=py_call(dtype_class, "u8"), bool=py_call(dtype_class, "?"))
    # Publish together after setup succeeds. Only admission metadata changes:
    # explicit Host* vendor and binding entries retain their CPU ownership.
    _ = py_call(context.update, update)


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
