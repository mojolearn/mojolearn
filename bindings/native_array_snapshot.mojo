# SPDX-License-Identifier: Apache-2.0
"""Capture native Array slots and retained buffer metadata without input hooks."""
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module, NativeModules
from bindings.mode_metadata_native import _class_fields
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from providers.python.borrowed_buffer_pyapi import admit_native_buffer_exporter, AdmittedBorrowedBuffer


def native_array_slots(context: PythonObject) raises -> PythonObject:
    # Callers without an operation handle keep their per-call module checks.
    var mods = NativeModules()
    return native_array_slots(mods, context)


def native_array_slots(mut mods: NativeModules, context: PythonObject) raises -> PythonObject:
    # `mods`: the operation's admitted module connections (one check per GIL
    # release window instead of one per Array).
    var b = mods.module("builtins")
    var cls = context["array_class"]
    if py_type(cls) is not b.type:
        raise Error("TypeError: neural arrays require their native storage class")
    var namespace = _class_fields(cls)
    var names = Python.tuple("_store", "_base", "_pin", "_mv", "_addr", "_readonly",
        "shape", "dtype", "ndim", "size", "nbytes", "itemsize", "strides", "order")
    var slots = Python.dict()
    var types = mods.module("types")
    for i in range(len(names)):  # small-loop(Array native slot schema)
        var slot = py_get(namespace, names[i])
        if (py_type(slot) is not types.MemberDescriptorType or slot.__objclass__ is not cls
                or not Bool(py=slot.__name__ == names[i])):
            raise Error("TypeError: neural Array storage descriptors changed")
        py_setitem(slots, names[i], slot)
    var mro = py_call(b.type.__dict__["__mro__"].__get__, cls)
    for i in range(len(mro)):  # small-loop(Array access ancestry)
        var row = _class_fields(mro[i])
        if py_contains(row, "__getattr__") or py_contains(row, "__del__"):
            raise Error("TypeError: neural Array metadata hooks are not supported")
        for name in ["__getattribute__", "__setattr__"]:  # driver-loop: native admission metadata and retained owner registry
            var own = py_get(row, PythonObject(name))
            if own is not None and own is not b.object.__dict__[name]:
                raise Error("TypeError: neural Array metadata hooks are not supported")
    return slots


def native_array_snapshot(value: PythonObject, context: PythonObject, all_dtypes: Bool = False) raises -> PythonObject:
    # Callers without an operation handle keep their per-call module checks.
    var mods = NativeModules()
    return native_array_snapshot(mods, value, context, all_dtypes)


def native_array_snapshot(mut mods: NativeModules, value: PythonObject, context: PythonObject, all_dtypes: Bool = False) raises -> PythonObject:
    # Capture every Array metadata word once through native member descriptors.
    # The returned passive view retains both the original owner and a real
    # buffer export; execution never consults mutable caller shape/stride slots.
    var b = mods.module("builtins")
    var cls = context["array_class"]
    if py_type(value) is not cls:
        raise Error("TypeError: neural Array input requires its exact native class")
    var slots = native_array_slots(mods, context)
    var values = Python.dict()
    var names = py_call(b.list, slots.keys())
    for i in range(len(names)):  # small-loop(capture native Array metadata once)
        py_setitem(values, names[i], py_call(slots[names[i]].__get__, value, cls))
    for name in ["_addr", "ndim", "size", "nbytes", "itemsize"]:  # small-loop(native size and address words)
        if py_type(values[name]) is not b.int:
            raise Error("TypeError: neural Array dimensions and addresses require native integers")
    if (py_type(values["shape"]) is not b.tuple or py_type(values["strides"]) is not b.tuple
            or py_type(values["dtype"]) is not b.str or py_type(values["order"]) is not b.str
            or py_type(values["_readonly"]) is not b.bool):
        raise Error("TypeError: neural Array metadata requires native shape, dtype and layout")
    var shape = values["shape"]
    var strides = values["strides"]
    var count = 1
    var itemsize = Int(py=values["itemsize"])
    var dtype = String(values["dtype"])
    var code = "f" if dtype == "<f4" else ("i" if dtype == "<i4" else ("b" if dtype == "<i1" else ("H" if dtype == "<u2" else "")))
    var expected_itemsize = 1 if code == "b" else (2 if code == "H" else 4)
    if all_dtypes:
        if dtype == "<f8" or dtype == "<i8":
            code = "d" if dtype == "<f8" else "q"
            expected_itemsize = 8
        elif dtype == "<u4":
            code = "I"
            expected_itemsize = 4
        elif dtype == "<u1":
            code = "B"
            expected_itemsize = 1
        elif dtype == "<f2":
            code = "H"
            expected_itemsize = 2
    if code == "" or itemsize != expected_itemsize or len(shape) != len(strides) or Int(py=values["ndim"]) != len(shape):
        raise Error("TypeError: neural Array storage dtype or rank is inconsistent")
    for i in range(len(shape)):  # small-loop(admitted tensor dimensions)
        if py_type(shape[i]) is not b.int or py_type(strides[i]) is not b.int:
            raise Error("TypeError: neural Array shape/strides require native integers")
        var n = Int(py=shape[i])
        if n < 0 or (n != 0 and count > 2147483647 // n):
            raise Error("neural Array shape exceeds native indexing")
        count *= n
    if Int(py=values["size"]) != count or Int(py=values["nbytes"]) != count * itemsize:
        raise Error("neural Array size differs from shape")
    var order = String(values["order"])
    if order != "C" and order != "F":
        raise Error("neural Array requires contiguous C or F layout")
    var step = itemsize
    for i in range(len(shape)):  # small-loop(verify the declared contiguous layout)
        var axis = len(shape) - i - 1 if order == "C" else i
        if count > 0 and Int(py=shape[axis]) > 1 and Int(py=strides[axis]) != step:
            raise Error("neural Array strides differ from its declared layout")
        step *= Int(py=shape[axis])
    var store = values["_store"]
    if store is None:
        store = values["_mv"]
        if py_type(store) is not b.memoryview or store.obj is None:
            raise Error("TypeError: neural Array needs retained native storage")
    admit_native_buffer_exporter(store)
    var pin = AdmittedBorrowedBuffer(store)
    var layout = pin.layout.value().copy()
    var offset = Int(py=values["_addr"]) - layout.address
    if count == 0:
        offset = 0
    var compact = layout.contiguous()
    var f_step = layout.itemsize
    var f_compact = True
    for i in range(len(layout.shape)):  # small-loop(F-contiguous retained native exporters)
        f_compact = f_compact and layout.suboffsets[i] < 0 and (layout.shape[i] <= 1 or layout.strides[i] == f_step)
        f_step *= layout.shape[i]
    # Empty native exporters have no reachable elements. CPython/NumPy may
    # report zero strides for their nonzero axes while admitting C and F order.
    compact = compact or f_compact or layout.count == 0
    if not compact or offset < 0 or offset + count * itemsize > layout.count * layout.itemsize:
        raise Error("neural Array lies outside its retained native storage")
    # _mv is mutable too: copying it blindly would reintroduce arbitrary
    # format/shape/buffer hooks when a downstream operation reads the snapshot.
    var original_view = values["_mv"]
    if py_type(original_view) is not b.memoryview:
        raise Error("TypeError: native Array requires its native memoryview storage")
    admit_native_buffer_exporter(original_view)
    var view_pin = AdmittedBorrowedBuffer(original_view)
    var view_layout = view_pin.layout.value().copy()
    if (view_pin.format != code or view_layout.itemsize != itemsize
            or view_layout.count != count
            or (count > 0 and view_layout.address != Int(py=values["_addr"]))):
        raise Error("TypeError: native Array memoryview differs from its captured storage")
    if (layout.readonly or view_layout.readonly) and not Bool(py=values["_readonly"]):
        raise Error("TypeError: native Array cannot mark read-only storage writable")
    var retained = py_call(b.memoryview, store)
    var retained_view = py_call(b.memoryview, original_view)
    var result = py_call(b.object.__new__, cls)
    for i in range(len(names)):  # small-loop(publish passive captured metadata via native slot setters)
        py_call(slots[names[i]].__set__, result, values[names[i]])
    py_call(slots["_store"].__set__, result, retained)
    py_call(slots["_mv"].__set__, result, retained_view)
    py_call(slots["_base"].__set__, result, value)
    _ = pin^
    _ = view_pin^
    return result




struct NativeArrayAdmission(Copyable, Movable):
    """Array class admission captured by one full `native_array_slots` guard.

    `recheck()` is the post-release re-check for a long native operation such
    as a generation loop: kernel/worker calls may release the GIL, so another
    thread can mutate the Array class while they run. Re-check exactly once per
    GIL-release window; it is not redundant. It reuses the admitted builtins,
    getters and descriptors (no module-connection scan, no per-row module
    lookup) and inspects only the mutable rows of the admitted ancestry
    (immutable builtin rows such as `object` cannot acquire hooks). Any
    difference from the admitted state reruns the full guard, so a refusal
    raises the same error the full guard raises.
    """

    var cls: PythonObject
    var mro: PythonObject
    var slots: PythonObject
    var rows: PythonObject
    var type_cls: PythonObject
    var str_cls: PythonObject
    var list_fn: PythonObject
    var mro_get: PythonObject
    var dict_get: PythonObject
    var object_getattribute: PythonObject
    var object_setattr: PythonObject

    def __init__(out self, context: PythonObject) raises:
        self.slots = native_array_slots(context)
        var b = native_loaded_module("builtins")
        self.cls = context["array_class"]
        self.type_cls = b.type
        self.str_cls = b.str
        self.list_fn = b.list
        self.mro_get = b.type.__dict__["__mro__"].__get__
        self.dict_get = b.type.__dict__["__dict__"].__get__
        self.object_getattribute = b.object.__dict__["__getattribute__"]
        self.object_setattr = b.object.__dict__["__setattr__"]
        self.mro = py_call(self.mro_get, self.cls)
        self.rows = Python.list()
        ref cpy = Python().cpython()
        for i in range(len(self.mro)):  # small-loop(mutable Array ancestry rows, admitted once)
            # Py_TPFLAGS_IMMUTABLETYPE (1 << 8): attributes cannot be set.
            var flags = cpy.lib.get_function[UInt64]("PyType_GetFlags")(self.mro[i]._obj_ptr)
            if (flags & UInt64(1 << 8)) == 0:
                _ = py_call(self.rows.append, self.mro[i])

    def _unchanged(self, context: PythonObject) raises -> Bool:
        var cls = context["array_class"]
        if cls is not self.cls or py_type(cls) is not self.type_cls:
            return False
        if py_call(self.mro_get, cls) is not self.mro:
            return False
        for i in range(len(self.rows)):  # small-loop(mutable admitted Array ancestry rows)
            var row = py_call(self.dict_get, self.rows[i])
            var keys = py_call(self.list_fn, row.keys())
            for k in range(len(keys)):  # small-loop(class namespace keys before metadata reads)
                if py_type(keys[k]) is not self.str_cls:
                    return False
            if py_contains(row, "__getattr__") or py_contains(row, "__del__"):
                return False
            var own = py_get(row, PythonObject("__getattribute__"))
            if own is not None and own is not self.object_getattribute:
                return False
            own = py_get(row, PythonObject("__setattr__"))
            if own is not None and own is not self.object_setattr:
                return False
            if self.rows[i] is self.cls:
                var names = py_call(self.list_fn, self.slots.keys())
                for j in range(len(names)):  # small-loop(Array native slot schema)
                    if py_get(row, names[j]) is not self.slots[names[j]]:
                        return False
        return True

    def recheck(self, context: PythonObject) raises:
        if not self._unchanged(context):
            _ = native_array_slots(context)
