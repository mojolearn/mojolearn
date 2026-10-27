# SPDX-License-Identifier: Apache-2.0
# Zero-copy ndarray view of an admitted Array's retained native exporter.
# Imported by both input and public-output owners; no model-specific routing.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.native_buffer_owner import native_buffer_view
from bindings.sequential_admission import sequential_admit_value, _one_of, _ctypes_array_owner
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem


def _namespace(cls: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var fields = py_call(b.type.__dict__["__dict__"].__get__, cls)
    var names = py_call(b.list, fields.keys())
    for i in range(len(names)):  # driver-loop: over refuse foreign equality hooks before slot lookups
        if py_type(names[i]) is not b.str:
            raise Error("TypeError: native Array storage requires exact string class keys")
    return fields


def native_array_ndarray_view(value: PythonObject, context: PythonObject) raises -> PythonObject:
    # A raw Array._mv may be an ownerless memory_at view. Build the ndarray
    # from its admitted backing store and offset instead, retaining that store
    # through the native memoryview after the model/Array has been released.
    var cls = context["array_class"]
    var slots = context["array_storage_slots"]
    var namespace = _namespace(cls)
    var names = Python.tuple("_store", "_mv", "_addr", "_readonly", "shape", "dtype", "order")
    var values = Python.dict()
    for i in range(len(names)):  # driver-loop: over native Array storage metadata, no Python properties
        var name = names[i]
        if py_get(namespace, name) is not slots[name]:
            raise Error("TypeError: native output requires the connected Array storage slots")
        py_setitem(values, name, py_call(slots[name].__get__, value, cls))
    var b = native_loaded_module("builtins")
    if (py_type(values["_addr"]) is not b.int or py_type(values["_readonly"]) is not b.bool
            or py_type(values["shape"]) is not b.tuple or py_type(values["dtype"]) is not b.str
            or py_type(values["order"]) is not b.str):
        raise Error("TypeError: native output requires ordinary Array shape and storage metadata")
    if not py_contains(context["array_storage_dtypes"], values["dtype"]):
        raise Error("TypeError: native output requires a supported numeric Array dtype")
    var shape = values["shape"]
    var empty = False
    for i in range(len(shape)):  # driver-loop: over admit dimensions before ndarray's native constructor
        if py_type(shape[i]) is not b.int or Int(py=shape[i]) < 0:
            raise Error("TypeError: native output requires nonnegative native shape dimensions")
        empty = empty or Int(py=shape[i]) == 0
    var order = String(values["order"])
    if order != "C" and order != "F":
        raise Error("TypeError: native output requires contiguous C or F order")
    var store = values["_store"]
    if store is None:
        # The native mapped-preprocessing output keeps an owning memoryview
        # slice in _mv and has no _store. This is safe only when the view has
        # an actual exporter; ownerless memory_at views need their _store.
        store = values["_mv"]
        if py_type(store) is not b.memoryview or store.obj is None:
            raise Error("TypeError: native output Array has no retained storage exporter")
    var store_type = py_type(store)
    if store_type is b.memoryview:
        sequential_admit_value(store, context)
    elif (store is None or not (_one_of(store_type, context["buffer_owner_types"])
            or _one_of(store_type, context["scalar_types"]) or _ctypes_array_owner(store_type, context))):
        # General input admission accepts Array/list values too. They are not
        # native buffer exporters; never ask them for a Python buffer hook.
        raise Error("TypeError: native output requires a native storage buffer, not a Python exporter")
    var buffer = native_buffer_view(store)
    if not Bool(py=buffer.contiguous):
        raise Error("TypeError: native output requires contiguous retained storage")
    # Public Py_buffer layout, as used by native_buffer_address. ANY_CONTIGUOUS
    # (0x98) accepts both original C and Fortran storage without packing/copying.
    ref cpy = Python().cpython()
    var exported = List[Int](length=10, fill=0)
    var pointer = exported.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    if cpy.lib.get_function[Int32]("PyObject_GetBuffer")(buffer._obj_ptr, pointer, Int32(0x98)) != 0:
        raise cpy.unsafe_get_error()
    var address = exported[0]
    var size = exported[2]
    cpy.lib.get_function[NoneType]("PyBuffer_Release")(pointer)
    var offset = 0 if empty else Int(py=values["_addr"]) - address
    if offset < 0 or offset > size:
        raise Error("ValueError: native output view lies outside its retained storage")
    if Bool(py=values["_readonly"]) and not Bool(py=buffer.readonly):
        buffer = py_call(buffer.toreadonly)
    # Exact ndarray C construction only: no asarray, Array.__array__, Python
    # buffer protocol, dtype conversion, or data copy. NumPy validates span.
    return py_call(context["ndarray"], shape, dtype=values["dtype"], buffer=buffer,
                   offset=offset, order=values["order"])
