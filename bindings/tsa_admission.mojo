# SPDX-License-Identifier: Apache-2.0
# Passive scalar/storage admission shared by the legacy TSA public owners.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.native_loaded_module import native_loaded_module, NativeModules
from bindings.native_class_admission import class_record, class_admit, dict_keys_admitted, dict_keys_admit
from bindings.py_balanced import py_call, py_type, py_get, py_setitem
from bindings.classical_validate import classical_view
from bindings.classical_arrays import classical_address
from bindings.classical_attributes import classical_connect_numpy
from bindings.native_numpy_metadata import native_numpy_dtype, native_numpy_module_fields
from bindings.native_buffer_owner import native_buffer_view
from bindings.native_scalar_decode import native_scalar_buffer
from bindings.arena_materialize_native import _prepare, _array_layout
from bindings.native_array_storage import native_array_from_bytes
from bindings.classical_arrays import classical_wrap_array, _store_code
from bindings.arena_array_metadata import arena_array_metadata
from gemm.f16_widen import f16_bits_to_f32
from bindings.sequential_admission import _one_of, _ctypes_array_owner
from bindings.admitted_input import admit_view, admitted_view_kind
from bindings.tensor_view import TensorView
from core.cast_in import CAST_KIND_BOOL, CAST_KIND_FLOAT, CAST_KIND_SIGNED, CAST_KIND_UNSIGNED


def tsa_scalar(value: PythonObject, context: PythonObject, integer: Bool = False) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var cls = py_type(value)
    if cls is b.int or cls is b.float or cls is b.bool or cls is b.str or cls is b.bytes or cls is b.bytearray:
        return value
    if _one_of(cls, context["scalar_types"]):
        return value
    if py_get(context, "ndarray") is not None:
        var factory = native_numpy_dtype(context)
        for code in ["?", "i1", "u1", "i2", "u2", "i4", "u4", "i8", "u8", "q", "Q", "f2", "f4", "f8", "g", "S", "U"]:  # driver-loop: exact primitive scalar classes, including long double and native long-long aliases
            if cls is py_call(factory, code).type:
                return value
        if cls is context["ndarray"] and _primitive_dtype(value, context):
            if Int(py=value.size) != 1:
                raise Error("TypeError: TSA scalar settings require one native numeric value")
            var kind = String(value.dtype.kind)
            return native_scalar_buffer(value).to_python_number(integer or kind == "i" or kind == "u" or kind == "b")
    # Native one-element numeric buffers retain the original int/float
    # conversion interface. Arbitrary __int__/__float__/__bool__ never run.
    # The one admission captures the value (bindings/admitted_input.mojo
    # admit_view, which also reads a ctypes-backed Array from its slots); its
    # one word is decoded here from the captured address.
    var one = admit_view(value, "scalar setting", context)
    if one.view.count() != 1:
        raise Error("TypeError: TSA scalar settings require one native numeric value")
    var result = tsa_word(one.view, "scalar setting")
    _ = one^
    return result


def tsa_word(v: TensorView, name: String) raises -> PythonObject:
    """The first element of a captured numeric view as a Python bool, int or
    float (bool, signed / unsigned integers of 1, 2, 4 or 8 bytes, float16 /
    32 / 64): the number the buffer's own format reads back as."""
    var t = admitted_view_kind(v, name)
    var kind = t[0]
    var item = t[1]
    if kind == CAST_KIND_BOOL:
        return PythonObject(MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=v.addr)[0] != 0)
    if kind == CAST_KIND_FLOAT:
        if item == 2:
            return PythonObject(Float64(f16_bits_to_f32(MutPointer[UInt16, MutUntrackedOrigin](unsafe_from_address=v.addr)[0])))
        if item == 4:
            return PythonObject(Float64(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=v.addr)[0]))
        if item == 8:
            return PythonObject(MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=v.addr)[0])
    if kind == CAST_KIND_SIGNED:
        if item == 1:
            return PythonObject(Int(MutPointer[Int8, MutUntrackedOrigin](unsafe_from_address=v.addr)[0]))
        if item == 2:
            return PythonObject(Int(MutPointer[Int16, MutUntrackedOrigin](unsafe_from_address=v.addr)[0]))
        if item == 4:
            return PythonObject(Int(MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=v.addr)[0]))
        if item == 8:
            return PythonObject(Int(MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=v.addr)[0]))
    if kind == CAST_KIND_UNSIGNED:
        if item == 1:
            return PythonObject(Int(MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=v.addr)[0]))
        if item == 2:
            return PythonObject(Int(MutPointer[UInt16, MutUntrackedOrigin](unsafe_from_address=v.addr)[0]))
        if item == 4:
            return PythonObject(Int(MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=v.addr)[0]))
        if item == 8:
            var w = MutPointer[UInt64, MutUntrackedOrigin](unsafe_from_address=v.addr)[0]
            if w >= UInt64(1) << 63:
                raise Error("ValueError: TSA " + name + " is beyond the signed 64-bit range")
            return PythonObject(Int(w))
    raise Error("TypeError: TSA " + name + " requires one native numeric value")


# TOMBSTONE: tsa_buffer (the TSA owners' own admission: classical_view, a by-hand repack of ctypes-backed Arrays,
# arena _prepare, a second pinned view; superseded by bindings/admitted_input.mojo admit_view, the one admission)
# deleted 2026-11-09 by lane/m1-followups-misc; code recoverable at 5b5490960d.
# Restore: git apply experiments/removed/TSA_BUFFER_ADMISSION.patch


def tsa_hw_class(model: PythonObject, context: PythonObject) raises -> Bool:
    var cls = py_type(model)
    if cls is context["hw_class"]:
        return False
    var module = py_get(context["modules"], "mojolearn._classical_host")
    if module is not None and py_type(module) is py_type(native_loaded_module("sys")):
        if cls is py_get(native_numpy_module_fields(module), "HostExponentialSmoothing"):
            return True
    raise Error("TypeError: ExponentialSmoothing requires its connected native model class")


comptime _HW_LAYOUT_SLOT = "tsa_hw_layout_v1"  # p3-cnn-sequence-tsa class record


def tsa_hw_fields(model: PythonObject, context: PythonObject) raises -> PythonObject:
    _ = tsa_hw_class(model, context)
    var b = native_loaded_module("builtins")
    var cls = py_type(model)
    if py_type(cls) is not b.type:
        raise Error("TypeError: ExponentialSmoothing requires a native model metaclass")
    var mro = py_call(b.type.__dict__["__mro__"].__get__, cls)
    var fields = PythonObject(None)
    # p3-cnn-sequence-tsa: the class walk's verdict (the MRO row whose
    # `__dict__` descriptor serves the instance storage) is kept per class
    # under the type watcher (any edit of the class or an ancestor misses),
    # and the instance keys are re-checked only after the instance dict
    # changed (dictionary watcher). Python 3.10/3.11 keep the full walk.
    var mods = NativeModules()
    var kept = class_record(mods, _HW_LAYOUT_SLOT, cls)
    if kept is not None:
        var own = py_call(b.type.__dict__["__dict__"].__get__, mro[Int(py=kept)])
        fields = py_call(py_get(own, "__dict__").__get__, model)
    else:
        var row = -1
        for i in range(len(mro)):  # driver-loop: exact class dictionaries and native instance storage
            var own = py_call(b.type.__dict__["__dict__"].__get__, mro[i])
            if not _tsa_str_items(py_call(b.list, own), b):
                raise Error("TypeError: ExponentialSmoothing class keys must be native strings")
            if py_get(own, "__del__") is not None:
                raise Error("TypeError: ExponentialSmoothing cannot invoke lifecycle callbacks")
            var descriptor = py_get(own, "__dict__")
            if fields is None and descriptor is not None and py_type(descriptor) is py_type(b.memoryview.obj):
                fields = py_call(descriptor.__get__, model)
                row = i
        if row >= 0:
            class_admit(mods, _HW_LAYOUT_SLOT, cls, PythonObject(row))
    if py_type(fields) is not b.dict:
        raise Error("TypeError: ExponentialSmoothing requires native instance storage")
    if dict_keys_admitted(fields):
        return fields
    if not _tsa_str_items(py_call(b.list, fields), b):
        raise Error("TypeError: ExponentialSmoothing state keys must be native strings")
    dict_keys_admit(fields)
    return fields



def _tsa_str_items(keys: PythonObject, b: PythonObject) raises -> Bool:
    """True when every item of the exact list `keys` is an exact str: the
    items are read borrowed through the list's C storage and their types
    compared by pointer (no item boxed into a Python call, no key equality
    or metaclass hook runs). Every Holt-Winters call admits the class and
    instance keys, so this stays pointer reads only."""
    ref cpy = Python().cpython()
    var get_item = cpy.lib.get_function[PyObjectPtr]("PyList_GetItem")
    var str_class = b.str
    var admitted = True
    for j in range(len(keys)):  # driver-loop: over the class and instance dictionary keys (model structure)
        var key_type = cpy.PyObject_Type(get_item(keys._obj_ptr, j))
        admitted = key_type == str_class._obj_ptr
        cpy.Py_DecRef(key_type)
        if not admitted:
            break
    _ = keys  # keep the list alive through the borrowed item reads above
    _ = str_class
    return admitted


def tsa_truth(value: PythonObject, context: PythonObject) raises -> Bool:
    var b = native_loaded_module("builtins")
    var cls = py_type(value)
    if value is None:
        return False
    if cls is b.list or cls is b.tuple or cls is b.dict or cls is b.set or cls is b.frozenset or cls is b.str or cls is b.bytes or cls is b.bytearray:
        return len(value) != 0
    if cls is context["array_class"]:
        var snapshot = admit_view(value, "flag", context)
        if len(snapshot.view.shape) == 0:
            raise Error("TypeError: len() of unsized object")
        return snapshot.view.shape[0] != 0
    if cls is py_get(context, "ndarray") and _primitive_dtype(value, context):
        var count = Int(py=value.size)
        if count == 0:
            return False
        if count != 1:
            raise Error("ValueError: the truth value of an array with more than one element is ambiguous")
        return native_scalar_buffer(value).truth()
    return Bool(py=py_call(b.bool, tsa_scalar(value, context)))



def tsa_component_index(value: PythonObject, context: PythonObject) raises -> PythonObject:
    if value is None:
        return value
    var b = native_loaded_module("builtins")
    if py_type(value) is b.bool or _one_of(py_type(value), context["integer_types"]):
        return py_call(b.int, value)
    if py_type(value) is py_get(context, "ndarray"):
        var snapshot = admit_view(value, "component index", context)
        if len(snapshot.view.shape) == 0 and admitted_view_kind(snapshot.view, "component index")[0] != CAST_KIND_FLOAT:
            return py_call(b.int, tsa_scalar(value, context))
    raise Error("TypeError: Holt-Winters component indices require native integers")



def _primitive_dtype(value: PythonObject, context: PythonObject) raises -> Bool:
    # Caller has admitted an exact ndarray before native dtype descriptors.
    var factory = native_numpy_dtype(context)
    var dtype = value.dtype
    for code in ["?", "i1", "u1", "i2", "u2", "i4", "u4", "i8", "u8", "q", "Q", "f2", "f4", "f8", "g"]:  # driver-loop: fixed real scalar buffer formats
        if py_type(dtype) is py_type(py_call(factory, code)):
            return True
    return False
