# SPDX-License-Identifier: Apache-2.0
# Passive scalar/storage admission shared by the legacy TSA public owners.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
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
from providers.python.array_metadata_pyapi import ArrayMetadata, array_shape_once, array_metadata_once


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
    var one = tsa_buffer(value, "scalar setting", context)
    if one.count != 1:
        raise Error("TypeError: TSA scalar settings require one native numeric value")
    # ndarray's scalar conversion is C-owned. Array has a Python protocol,
    # so read its independently captured buffer instead.
    var view = one.owner[1]
    var flat = py_call(view.cast, "B")
    var code = String(view.format)
    if code == "e" or (one.is_array and one.dtype_number == 23):
        var bits = MutPointer[UInt16, MutUntrackedOrigin](unsafe_from_address=one.source.addr)[0]
        return PythonObject(Float64(f16_bits_to_f32(bits)))
    return py_call(flat.cast, code)[0]


def tsa_buffer(value: PythonObject, name: String, context: PythonObject) raises -> ArrayMetadata:
    _array_layout(context)
    var b = native_loaded_module("builtins")
    var mapping = py_call(b.type.__dict__["__dict__"].__get__, context["array_class"])
    if py_get(mapping, "__setattr__") is not None:
        raise Error("TypeError: TSA native preparation cannot invoke Array attribute hooks")
    var prepared = value
    if py_type(value) is not context["array_class"]:
        prepared = classical_view(value, context, name)
    else:
        # Library output stores can be native ctypes arrays. The shared
        # materializer does not admit those exporters; pack their C buffer
        # after exact native owner admission, retaining the supported Array
        # input rather than invoking its Python conversion methods.
        var slots = context["owner_array_slots"]
        var view = py_call(slots["_mv"].__get__, value, context["array_class"])
        if py_type(view) is not b.memoryview:
            raise Error("TypeError: TSA Array backing view must be a native memoryview")
        if _ctypes_array_owner(py_type(view.obj), context):
            var metadata = arena_array_metadata(value, context)
            var order = py_call(slots["order"].__get__, value, context["array_class"])
            if py_type(order) is not b.str or (String(order) != "C" and String(order) != "F"):
                raise Error("TypeError: TSA Array order must be C or F")
            var dtype = String(metadata["dtype"])
            var store = native_array_from_bytes(_store_code(dtype), py_call(view.tobytes, "C"))
            prepared = classical_wrap_array(store, metadata["shape"], dtype, String(order), context)
    # Native preparation validates exact metadata and pins an independent
    # buffer before any kernel can release the GIL. Never keep just mutable
    # caller Array slots as the ownership contract.
    if py_get(context, "ndarray") is not None:
        _ = native_numpy_dtype(context)
    var snapshot = _prepare(prepared, context, True)
    var result = array_metadata_once(snapshot, array_shape_once(snapshot, context), name, context)
    var pinned = native_buffer_view(py_call(context["array_fields"]["_mv"], snapshot, context["array_class"]))
    result.owner = Python.tuple(result.owner, pinned, value)
    return result^


def tsa_hw_class(model: PythonObject, context: PythonObject) raises -> Bool:
    var cls = py_type(model)
    if cls is context["hw_class"]:
        return False
    var module = py_get(context["modules"], "mojolearn._classical_host")
    if module is not None and py_type(module) is py_type(native_loaded_module("sys")):
        if cls is py_get(native_numpy_module_fields(module), "HostExponentialSmoothing"):
            return True
    raise Error("TypeError: ExponentialSmoothing requires its connected native model class")


def tsa_hw_fields(model: PythonObject, context: PythonObject) raises -> PythonObject:
    _ = tsa_hw_class(model, context)
    var b = native_loaded_module("builtins")
    var cls = py_type(model)
    if py_type(cls) is not b.type:
        raise Error("TypeError: ExponentialSmoothing requires a native model metaclass")
    var mro = py_call(b.type.__dict__["__mro__"].__get__, cls)
    var fields = PythonObject(None)
    for i in range(len(mro)):  # driver-loop: exact class dictionaries and native instance storage
        var own = py_call(b.type.__dict__["__dict__"].__get__, mro[i])
        var keys = py_call(b.list, own)
        for j in range(len(keys)):  # driver-loop: validate keys before any descriptor lookup
            if py_type(keys[j]) is not b.str:
                raise Error("TypeError: ExponentialSmoothing class keys must be native strings")
        if py_get(own, "__del__") is not None:
            raise Error("TypeError: ExponentialSmoothing cannot invoke lifecycle callbacks")
        var descriptor = py_get(own, "__dict__")
        if fields is None and descriptor is not None and py_type(descriptor) is py_type(b.memoryview.obj):
            fields = py_call(descriptor.__get__, model)
    if py_type(fields) is not b.dict:
        raise Error("TypeError: ExponentialSmoothing requires native instance storage")
    var keys = py_call(b.list, fields)
    for j in range(len(keys)):  # driver-loop: field lookup cannot invoke application key equality
        if py_type(keys[j]) is not b.str:
            raise Error("TypeError: ExponentialSmoothing state keys must be native strings")
    return fields


def tsa_truth(value: PythonObject, context: PythonObject) raises -> Bool:
    var b = native_loaded_module("builtins")
    var cls = py_type(value)
    if value is None:
        return False
    if cls is b.list or cls is b.tuple or cls is b.dict or cls is b.set or cls is b.frozenset or cls is b.str or cls is b.bytes or cls is b.bytearray:
        return len(value) != 0
    if cls is context["array_class"]:
        var snapshot = tsa_buffer(value, "flag", context)
        if len(snapshot.shape) == 0:
            raise Error("TypeError: len() of unsized object")
        return snapshot.shape[0] != 0
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
        var snapshot = tsa_buffer(value, "component index", context)
        if len(snapshot.shape) == 0 and snapshot.source.kind != 2:
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
