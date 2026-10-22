# SPDX-License-Identifier: Apache-2.0
# Native ownership of DeviceRows metadata, without invoking buffer protocols.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.py_balanced import py_call, py_type, py_get, py_setitem
from bindings.native_error_class import error_without_class
from bindings.classical_attributes import classical_connect_numpy
from bindings.native_owner_lifetime import native_release_admit, native_release_weakrefs
from bindings.arena_array_metadata import arena_array_metadata
from bindings.arena_materialize_native import arena_materialize
from bindings.arena_cache_stats import arena_cache_stats
from bindings.arena_cache_get import arena_cache_get
from bindings.optimizer_public_words import optimizer_python_error, PYTHON_ERROR_PENDING


def _rows_metadata_fields(value: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = context["arena_native_types"]
    var cls = py_type(value)
    if py_type(cls) is not b["type"]:
        raise Error("TypeError: DeviceRows requires a native metadata class")
    var ancestry = py_call(b["type"].__dict__["__mro__"].__get__,cls)
    var expected = context["maxabs_rows_class"]
    var descriptor = context["maxabs_rows_dict"]
    if len(ancestry) < 2 or ancestry[len(ancestry)-2] is not expected or ancestry[len(ancestry)-1] is not b["object"]:
        raise Error("TypeError: DeviceRows requires its ordinary class or a subclass without mixed-in protocols")
    for i in range(len(ancestry)-1):  # driver-loop: admit ordinary subclass layouts without invoking their descriptors
        var mapping = py_call(b["type"].__dict__["__dict__"].__get__,ancestry[i])
        var keys = py_call(b["list"],mapping)
        for j in range(len(keys)):  # driver-loop: class keys precede lookup that could invoke application equality
            if py_type(keys[j]) is not b["str"]:
                raise Error("TypeError: DeviceRows class keys must be native strings")
        if py_get(mapping,"__del__") is not None or py_get(mapping,"__getattribute__") is not None:
            raise Error("TypeError: DeviceRows cannot invoke attribute or lifecycle hooks")
        var dictionary = py_get(mapping,"__dict__")
        if dictionary is not None and dictionary is not descriptor:
            raise Error("TypeError: DeviceRows requires its native dictionary descriptor")
    var fields = py_call(descriptor.__get__,value,expected)
    if py_type(fields) is not b["dict"]:
        raise Error("TypeError: DeviceRows requires a native metadata dictionary")
    var keys = py_call(b["list"],fields)
    for i in range(len(keys)):  # driver-loop: reject metadata key protocols before field lookup or replacement
        if py_type(keys[i]) is not b["str"]:
            raise Error("TypeError: DeviceRows metadata keys must be native strings")
    return fields


def _rows_initialize(model: PythonObject, args: PythonObject, context: PythonObject) raises:
    var fields = _rows_metadata_fields(model,context)
    var base = arena_array_metadata(args[0],context)
    var indices = arena_array_metadata(args[1],context)
    var base_shape = base["shape"]
    if len(base_shape) == 0:
        raise Error("IndexError: tuple index out of range")
    var shape = Python.list(indices["size"])
    var size = Int(py=indices["size"])
    for i in range(1,len(base_shape)):  # driver-loop: shape of the selected rows, preserving all trailing dimensions
        var extent = Int(py=base_shape[i])
        if size > 0 and extent > 9223372036854775807//size:
            raise Error("ValueError: DeviceRows shape exceeds native indexing bounds")
        size *= extent
        _ = py_call(shape.append,base_shape[i])
    var first = Int(py=base_shape[0])
    # Keep the existing opaque take field without invoking it. Its explicit
    # materialization operation owns callback admission separately.
    var proposed = Python.dict(base=args[0],indices=args[1],_take=args[2],_host=PythonObject(None),
        shape=py_call(context["arena_native_types"]["tuple"],shape),dtype=base["dtype"],
        itemsize=base["itemsize"],ndim=base["ndim"],size=size,
        row_words=Int(py=base["size"])//first if first != 0 else 0)
    var names = py_call(context["arena_native_types"]["list"],proposed)
    for i in range(len(names)):  # driver-loop: admit every replaced owner before the first state mutation
        var previous = py_get(fields,names[i])
        var replacement = proposed[names[i]]
        if previous is replacement:
            continue
        if py_type(previous) is context["owner_method_class"] and py_type(replacement) is context["owner_method_class"] and previous.__func__ is replacement.__func__ and previous.__self__ is replacement.__self__:
            # Renewed access to the same ordinary bound method retains both
            # owners. Only the old method object's weak references can run.
            native_release_weakrefs(previous,context)
        else:
            native_release_admit(previous,context)
    for i in range(len(names)):  # driver-loop: publish native constructor metadata after lifetime admission succeeds
        py_setitem(fields,names[i],proposed[names[i]])


def arena_rows_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        var model = call[0]
        var operation = String(call[1])
        var args = call[2]
        var context = call[3]
        var result = PythonObject(None)
        if operation == "cache_len" or operation == "resident_stats":
            result = arena_cache_stats(model,operation,args,context)
        elif operation == "cache_get":
            result = arena_cache_get(model,args,context)
        else:
            classical_connect_numpy(context)
        if operation == "init":
            _rows_initialize(model,args,context)
        elif operation == "materialize":
            result = arena_materialize(_rows_metadata_fields(model,context),context)
        elif operation == "len":
            var shape = _rows_metadata_fields(model,context)["shape"]
            var b = context["arena_native_types"]
            if py_type(shape) is not b["tuple"] or len(shape) == 0 or py_type(shape[0]) is not b["int"]:
                raise Error("TypeError: DeviceRows length requires native shape metadata")
            result = shape[0]
        elif operation != "cache_len" and operation != "resident_stats" and operation != "cache_get":
            raise Error("ValueError: unknown native DeviceRows metadata operation")
        return result.steal_data()
    except error:
        if String(error) == PYTHON_ERROR_PENDING or String(error).startswith("OverflowError:"):
            return optimizer_python_error(error)
        if String(error).startswith("TypeError:"):
            return raise_python_exception(error_without_class(error,"TypeError"),ExceptionType.TypeError)
        if String(error).startswith("IndexError:"):
            return raise_python_exception(error_without_class(error,"IndexError"),ExceptionType("PyExc_IndexError"))
        if String(error).startswith("AttributeError:"):
            return raise_python_exception(error_without_class(error,"AttributeError"),ExceptionType("PyExc_AttributeError"))
        return raise_python_exception(error_without_class(error,"ValueError"),ExceptionType.ValueError)
