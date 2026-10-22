# SPDX-License-Identifier: Apache-2.0
# Native ownership of DeviceRows metadata, without invoking buffer protocols.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.py_balanced import py_call, py_type, py_get, py_setitem
from bindings.native_error_class import error_without_class
from bindings.classical_attributes import classical_connect_numpy
from bindings.native_numpy_metadata import native_numpy_dtype


def _rows_metadata_fields(value: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var cls = py_type(value)
    if py_type(cls) is not b.type:
        raise Error("TypeError: DeviceRows requires a native metadata class")
    var ancestry = py_call(b.type.__dict__["__mro__"].__get__,cls)
    var expected = context["maxabs_rows_class"]
    var descriptor = context["maxabs_rows_dict"]
    if len(ancestry) < 2 or ancestry[len(ancestry)-2] is not expected or ancestry[len(ancestry)-1] is not b.object:
        raise Error("TypeError: DeviceRows requires its ordinary class or a subclass without mixed-in protocols")
    for i in range(len(ancestry)-1):  # driver-loop: admit ordinary subclass layouts without invoking their descriptors
        var mapping = py_call(b.type.__dict__["__dict__"].__get__,ancestry[i])
        var keys = py_call(b.list,mapping)
        for j in range(len(keys)):  # driver-loop: class keys precede lookup that could invoke application equality
            if py_type(keys[j]) is not b.str:
                raise Error("TypeError: DeviceRows class keys must be native strings")
        if py_get(mapping,"__del__") is not None or py_get(mapping,"__getattribute__") is not None:
            raise Error("TypeError: DeviceRows cannot invoke attribute or lifecycle hooks")
        var dictionary = py_get(mapping,"__dict__")
        if dictionary is not None and dictionary is not descriptor:
            raise Error("TypeError: DeviceRows requires its native dictionary descriptor")
    var fields = py_call(descriptor.__get__,value,expected)
    if py_type(fields) is not b.dict:
        raise Error("TypeError: DeviceRows requires a native metadata dictionary")
    var keys = py_call(b.list,fields)
    for i in range(len(keys)):  # driver-loop: reject metadata key protocols before field lookup or replacement
        if py_type(keys[i]) is not b.str:
            raise Error("TypeError: DeviceRows metadata keys must be native strings")
    return fields


def _array_metadata(value: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var cls = py_type(value)
    var metadata = Python.dict()
    var names = Python.tuple("shape", "size", "dtype", "itemsize", "ndim")
    if cls is context["array_class"]:
        var ancestry = py_call(b.type.__dict__["__mro__"].__get__,cls)
        if len(ancestry) != 2 or ancestry[1] is not b.object:
            raise Error("TypeError: DeviceRows Array metadata cannot inherit lifecycle hooks")
        var mapping = py_call(b.type.__dict__["__dict__"].__get__,cls)
        var keys = py_call(b.list,mapping)
        for i in range(len(keys)):  # driver-loop: reject application key equality before class metadata lookups
            if py_type(keys[i]) is not b.str:
                raise Error("TypeError: DeviceRows requires native Array class keys")
        if py_get(mapping,"__del__") is not None or py_get(mapping,"__getattribute__") is not None:
            raise Error("TypeError: DeviceRows cannot invoke Array lifecycle or attribute hooks")
        for i in range(len(names)):  # driver-loop: retained native slots own the constructor's metadata reads
            var descriptor = context["arena_array_slots"][names[i]]
            if py_get(mapping,names[i]) is not descriptor:
                raise Error("TypeError: DeviceRows requires the connected native Array slots")
            py_setitem(metadata,names[i],py_call(descriptor.__get__,value,cls))
        if py_type(metadata["dtype"]) is not b.str:
            raise Error("TypeError: DeviceRows Array dtype must be a native string")
    elif cls is py_get(context,"ndarray"):
        # An exact ndarray exposes these fields through NumPy's native
        # immutable descriptors; subclasses and their properties never enter.
        py_setitem(metadata,"shape",value.shape)
        py_setitem(metadata,"size",value.size)
        py_setitem(metadata,"dtype",value.dtype)
        py_setitem(metadata,"itemsize",value.itemsize)
        py_setitem(metadata,"ndim",value.ndim)
    else:
        raise Error("TypeError: DeviceRows requires ordinary Array or ndarray metadata; Python buffer protocols are not supported")
    var shape = metadata["shape"]
    if py_type(shape) is not b.tuple or py_type(metadata["size"]) is not b.int or py_type(metadata["itemsize"]) is not b.int or py_type(metadata["ndim"]) is not b.int:
        raise Error("TypeError: DeviceRows requires native shape and extent metadata")
    var count = 1
    for i in range(len(shape)):  # driver-loop: construct and validate rank metadata without invoking shape protocols
        if py_type(shape[i]) is not b.int:
            raise Error("TypeError: DeviceRows dimensions must be native integers")
        var extent = Int(py=shape[i])
        if extent < 0 or (count > 0 and extent > 9223372036854775807//count):
            raise Error("ValueError: DeviceRows shape exceeds native indexing bounds")
        count *= extent
    if Int(py=metadata["size"]) != count or Int(py=metadata["ndim"]) != len(shape) or Int(py=metadata["itemsize"]) < 1:
        raise Error("ValueError: DeviceRows metadata extents disagree")
    return metadata


def _released_weakrefs(value: PythonObject, context: PythonObject) raises:
    var refs = py_call(context["arena_getweakrefs"],value)
    for i in range(len(refs)):  # driver-loop: releasing an otherwise native owner must not dispatch application callbacks
        if py_type(refs[i]) is not context["arena_weakref_class"] or py_call(context["arena_weakref_callback"].__get__,refs[i]) is not None:
            raise Error("TypeError: DeviceRows reinitialization cannot release weak-reference callbacks")


def _released_dtype(value: PythonObject, context: PythonObject, depth: Int) raises -> Bool:
    if py_get(context,"ndarray") is None:
        return False
    var classes = py_get(context,"arena_dtype_classes")
    if classes is None:
        classes = Python.list()
        var scalars = Python.list()
        var factory = native_numpy_dtype(context)
        for code in ["?", "i1", "u1", "i2", "u2", "i4", "u4", "i8", "u8", "f2", "f4", "f8", "g", "c8", "c16", "G", "O", "S", "U", "V", "M8", "m8"]:  # driver-loop: fixed immutable NumPy dtype classes before their retained owners are inspected
            var dtype = py_call(factory,code)
            _ = py_call(classes.append,py_type(dtype))
            if code != "O" and code != "V":
                _ = py_call(scalars.append,dtype.type)
        py_setitem(context,"arena_dtype_classes",classes)
        py_setitem(context,"arena_dtype_scalars",scalars)
    for i in range(len(classes)):  # driver-loop: admit exact native dtype classes before inspecting their C descriptors
        if py_type(value) is classes[i]:
            # A structured/subarray dtype and its native metadata do not
            # imply a Python callback. Admit their actual retained owners,
            # including field titles and metadata values, without hashing
            # application-controlled dictionary keys.
            for mapping in [value.metadata, value.fields]:  # driver-loop: NumPy-owned dictionary proxies
                if mapping is not None:
                    var pairs = py_call(Python.import_module("builtins").list,mapping.items())
                    for j in range(len(pairs)):  # driver-loop: native dtype key/value ownership, no key lookups
                        _released_owner(pairs[j][0],context,depth+1)
                        _released_owner(pairs[j][1],context,depth+1)
            _released_owner(value.names,context,depth+1)
            _released_owner(value.subdtype,context,depth+1)
            return True
    return False


def _released_owner(value: PythonObject, context: PythonObject, depth: Int = 0) raises:
    # Reinitialization decrements the previous fields' references. Admit the
    # complete owners that could be destroyed before replacing any field.
    # Keeping an identical value is handled by the caller and drops no owner.
    var b = Python.import_module("builtins")
    var cls = py_type(value)
    if value is None or cls is b.str or cls is b.bytes or cls is b.bytearray or cls is b.bool or cls is b.int or cls is b.float or cls is b.complex:
        return
    if depth > 64:
        raise Error("TypeError: DeviceRows reinitialization owner nesting exceeds the native bound")
    _released_weakrefs(value,context)
    if cls is b.tuple or cls is b.list:
        for i in range(len(value)):  # driver-loop: old metadata containers may own application finalizers transitively
            _released_owner(value[i],context,depth+1)
        return
    if cls is b.dict:
        var pairs = py_call(b.list,value.items())
        for i in range(len(pairs)):  # driver-loop: inspect dictionary owners without key equality lookups
            _released_owner(pairs[i][0],context,depth+1)
            _released_owner(pairs[i][1],context,depth+1)
        return
    if cls is context["array_class"]:
        _ = _array_metadata(value,context)
        var mapping = py_call(b.type.__dict__["__dict__"].__get__,cls)
        var owners = py_call(b.list,context["arena_array_slots"])
        for i in range(len(owners)):  # driver-loop: all Array slots own references, including otherwise unused metadata
            var descriptor = context["arena_array_slots"][owners[i]]
            if py_get(mapping,owners[i]) is not descriptor:
                raise Error("TypeError: DeviceRows reinitialization requires native Array owner slots")
            _released_owner(py_call(descriptor.__get__,value,cls),context,depth+1)
        return
    if cls is b.memoryview:
        _released_owner(value.obj,context,depth+1)
        return
    if cls is context["array_type"] or cls is context["arena_mmap_class"]:
        return
    if cls is py_get(context,"ndarray"):
        var dtype = value.dtype
        if not _released_dtype(dtype,context,depth+1):
            raise Error("TypeError: DeviceRows reinitialization requires an admitted native dtype")
        if Bool(py=dtype.hasobject):
            # ndarray.item is NumPy's immutable native descriptor. It returns
            # object references (or a structured tuple), never their protocols;
            # the ndarray retains them throughout this admission pass.
            for i in range(Int(py=value.size)):  # driver-loop: object-buffer ownership must be checked before release
                _released_owner(py_call(value.item,i),context,depth+1)
        _released_owner(value.base,context,depth+1)
        return
    if _released_dtype(value,context,depth):
        return
    var scalars = py_get(context,"arena_dtype_scalars")
    if scalars is not None:
        for i in range(len(scalars)):  # driver-loop: exact primitive NumPy scalars own no application objects
            if cls is scalars[i]:
                return
    raise Error("TypeError: DeviceRows reinitialization cannot release Python-owned metadata; retain the same owner or release it in application code first")


def _rows_initialize(model: PythonObject, args: PythonObject, context: PythonObject) raises:
    var fields = _rows_metadata_fields(model,context)
    var base = _array_metadata(args[0],context)
    var indices = _array_metadata(args[1],context)
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
        shape=py_call(Python.import_module("builtins").tuple,shape),dtype=base["dtype"],
        itemsize=base["itemsize"],ndim=base["ndim"],size=size,
        row_words=Int(py=base["size"])//first if first != 0 else 0)
    var names = py_call(Python.import_module("builtins").list,proposed)
    for i in range(len(names)):  # driver-loop: admit every replaced owner before the first state mutation
        var previous = py_get(fields,names[i])
        var replacement = proposed[names[i]]
        if previous is replacement:
            continue
        if py_type(previous) is context["arena_method_class"] and py_type(replacement) is context["arena_method_class"] and previous.__func__ is replacement.__func__ and previous.__self__ is replacement.__self__:
            # Renewed access to the same ordinary bound method retains both
            # owners. Only the old method object's weak references can run.
            _released_weakrefs(previous,context)
        else:
            _released_owner(previous,context)
    for i in range(len(names)):  # driver-loop: publish native constructor metadata after lifetime admission succeeds
        py_setitem(fields,names[i],proposed[names[i]])


def arena_rows_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        var model = call[0]
        var operation = String(call[1])
        var args = call[2]
        var context = call[3]
        classical_connect_numpy(context)
        var result = PythonObject(None)
        if operation == "init":
            _rows_initialize(model,args,context)
        elif operation == "len":
            var shape = _rows_metadata_fields(model,context)["shape"]
            var b = Python.import_module("builtins")
            if py_type(shape) is not b.tuple or len(shape) == 0 or py_type(shape[0]) is not b.int:
                raise Error("TypeError: DeviceRows length requires native shape metadata")
            result = shape[0]
        else:
            raise Error("ValueError: unknown native DeviceRows metadata operation")
        return result.steal_data()
    except error:
        if String(error).startswith("TypeError:"):
            return raise_python_exception(error_without_class(error,"TypeError"),ExceptionType.TypeError)
        if String(error).startswith("IndexError:"):
            return raise_python_exception(error_without_class(error,"IndexError"),ExceptionType("PyExc_IndexError"))
        return raise_python_exception(error_without_class(error,"ValueError"),ExceptionType.ValueError)
