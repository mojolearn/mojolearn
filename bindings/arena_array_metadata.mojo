# SPDX-License-Identifier: Apache-2.0
# Exact native Array/ndarray metadata for arena lifecycle operations.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem


def arena_array_metadata(value: PythonObject, context: PythonObject) raises -> PythonObject:
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
            var descriptor = context["owner_array_slots"][names[i]]
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
