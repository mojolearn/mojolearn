# SPDX-License-Identifier: Apache-2.0
# Admit retained object ownership before a native operation releases it.
# Import-time metadata comes from _native_owner_lifetime.connect_owner_context;
# callers connect already-loaded NumPy before admission and preserve all old
# fields until every replaced owner passes. Identical retained owners need no
# release admission. This helper never mutates or frees the admitted value.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_type, py_get, py_setitem
from bindings.native_numpy_metadata import native_numpy_dtype


def native_release_weakrefs(value: PythonObject, context: PythonObject) raises:
    var refs = py_call(context["owner_getweakrefs"],value)
    for i in range(len(refs)):  # driver-loop: releasing an otherwise native owner must not dispatch application callbacks
        if py_type(refs[i]) is not context["owner_weakref_class"] or py_call(context["owner_weakref_callback"].__get__,refs[i]) is not None:
            raise Error("TypeError: native owner reinitialization cannot release weak-reference callbacks")


def _released_dtype(value: PythonObject, context: PythonObject, depth: Int) raises -> Bool:
    if py_get(context,"ndarray") is None:
        return False
    var classes = py_get(context,"owner_dtype_classes")
    if classes is None:
        classes = Python.list()
        var scalars = Python.list()
        var factory = native_numpy_dtype(context)
        for code in ["?", "i1", "u1", "i2", "u2", "i4", "u4", "i8", "u8", "f2", "f4", "f8", "g", "c8", "c16", "G", "O", "S", "U", "V", "M8", "m8"]:  # driver-loop: fixed immutable NumPy dtype classes before their retained owners are inspected
            var dtype = py_call(factory,code)
            _ = py_call(classes.append,py_type(dtype))
            if code != "O" and code != "V":
                _ = py_call(scalars.append,dtype.type)
        py_setitem(context,"owner_dtype_classes",classes)
        py_setitem(context,"owner_dtype_scalars",scalars)
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
                        native_release_admit(pairs[j][0],context,depth+1)
                        native_release_admit(pairs[j][1],context,depth+1)
            native_release_admit(value.names,context,depth+1)
            native_release_admit(value.subdtype,context,depth+1)
            return True
    return False


def _native_item(value: PythonObject, index: PythonObject) raises -> PythonObject:
    # Callers admit the exact ndarray (or its native flat iterator), dtype
    # and native integer/field-name index before this public CPython API.
    ref cpy = Python().cpython()
    var result = cpy.lib.get_function[PyObjectPtr]("PyObject_GetItem")(value._obj_ptr,index._obj_ptr)
    if not result:
        raise cpy.unsafe_get_error()
    return PythonObject(from_owned=result)


def _released_object_elements(value: PythonObject, context: PythonObject, depth: Int) raises:
    # Follow field views without following their .base back to the array:
    # dtype ownership and the original buffer owner are checked separately.
    var dtype = value.dtype
    if dtype.fields is not None:
        var names = dtype.names
        for i in range(len(names)):  # driver-loop: structured fields retain object owners independently of native numeric bytes
            var field = _native_item(value,names[i])
            _released_object_elements(field,context,depth+1)
    elif Bool(py=dtype.hasobject):
        var flattened = value.flat
        for i in range(Int(py=value.size)):  # driver-loop: object-array indexing returns the retained object reference, without conversion
            native_release_admit(_native_item(flattened,PythonObject(i)),context,depth+1)


def native_release_admit(value: PythonObject, context: PythonObject, depth: Int = 0) raises:
    # Reinitialization decrements the previous fields' references. Admit the
    # complete owners that could be destroyed before replacing any field.
    # Keeping an identical value is handled by the caller and drops no owner.
    var b = Python.import_module("builtins")
    var cls = py_type(value)
    if value is None or cls is b.str or cls is b.bytes or cls is b.bytearray or cls is b.bool or cls is b.int or cls is b.float or cls is b.complex:
        return
    if depth > 64:
        raise Error("TypeError: native owner reinitialization owner nesting exceeds the native bound")
    native_release_weakrefs(value,context)
    if cls is b.tuple or cls is b.list:
        for i in range(len(value)):  # driver-loop: old metadata containers may own application finalizers transitively
            native_release_admit(value[i],context,depth+1)
        return
    if cls is b.dict:
        var pairs = py_call(b.list,value.items())
        for i in range(len(pairs)):  # driver-loop: inspect dictionary owners without key equality lookups
            native_release_admit(pairs[i][0],context,depth+1)
            native_release_admit(pairs[i][1],context,depth+1)
        return
    if cls is context["array_class"]:
        var ancestry = py_call(b.type.__dict__["__mro__"].__get__,cls)
        if len(ancestry) != 2 or ancestry[1] is not b.object:
            raise Error("TypeError: native owner reinitialization cannot inherit Array lifecycle hooks")
        var mapping = py_call(b.type.__dict__["__dict__"].__get__,cls)
        var keys = py_call(b.list,mapping)
        for i in range(len(keys)):  # driver-loop: reject key protocols before class-owner lookup
            if py_type(keys[i]) is not b.str:
                raise Error("TypeError: native owner reinitialization requires native Array class keys")
        if py_get(mapping,"__del__") is not None or py_get(mapping,"__getattribute__") is not None:
            raise Error("TypeError: native owner reinitialization cannot invoke Array lifecycle hooks")
        var owners = py_call(b.list,context["owner_array_slots"])
        for i in range(len(owners)):  # driver-loop: all Array slots own references, including otherwise unused metadata
            var descriptor = context["owner_array_slots"][owners[i]]
            if py_get(mapping,owners[i]) is not descriptor:
                raise Error("TypeError: native owner reinitialization requires native Array owner slots")
            native_release_admit(py_call(descriptor.__get__,value,cls),context,depth+1)
        return
    if cls is b.memoryview:
        native_release_admit(value.obj,context,depth+1)
        return
    if cls is context["array_type"] or cls is context["owner_mmap_class"]:
        return
    if cls is py_get(context,"ndarray"):
        var dtype = value.dtype
        if not _released_dtype(dtype,context,depth+1):
            raise Error("TypeError: native owner reinitialization requires an admitted native dtype")
        if Bool(py=dtype.hasobject):
            _released_object_elements(value,context,depth+1)
        native_release_admit(value.base,context,depth+1)
        return
    if _released_dtype(value,context,depth):
        return
    var scalars = py_get(context,"owner_dtype_scalars")
    if scalars is not None:
        for i in range(len(scalars)):  # driver-loop: exact primitive NumPy scalars own no application objects
            if cls is scalars[i]:
                return
    raise Error("TypeError: native owner reinitialization cannot release Python-owned metadata; retain the same owner or release it in application code first")
