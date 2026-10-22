# SPDX-License-Identifier: Apache-2.0
# The explicit CPU density digest hashes stored model fields, not NPZ bytes.
# Array.tobytes uses physical storage order; ndarray.tobytes uses logical C
# order. Preserve that distinction while owning iteration and SHA-256 here.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_type, py_get, py_contains
from bindings.classical_attributes import classical_connect_numpy, classical_getattribute
from bindings.host_tree_native import host_tree_constructor_fields
from bindings.native_numpy_metadata import native_numpy_dtype
from bindings.sequential_admission import _one_of
from bindings.optimizer_public_words import stash_python_error
from core.sha256 import Sha256


def _density_hash_call(entry: PythonObject, *args: PythonObject) raises -> PythonObject:
    ref cpy = Python().cpython()
    var envelope = cpy.PyTuple_New(len(args))
    if not envelope:
        var failure = stash_python_error()
        raise failure^
    for i in range(len(args)):  # driver-loop: fixed native metadata call envelope
        _ = cpy.PyTuple_SetItem(envelope, i, cpy.Py_NewRef(args[i]._obj_ptr))
    var pointer = cpy.PyObject_Call(entry._obj_ptr, envelope, PyObjectPtr())
    cpy.Py_DecRef(envelope)
    if not pointer:
        var failure = stash_python_error()
        raise failure^
    return PythonObject(from_owned=pointer)


def _density_hash_buffer(value: PythonObject, context: PythonObject) raises -> PythonObject:
    var cls = py_type(value)
    var view = PythonObject(None)
    if cls is context["array_class"]:
        # Read the retained native slot rather than Array.tobytes or a
        # subsequently replaced Python property. Hash its physical bytes.
        var type_class = py_type(cls)
        var fields = py_call(type_class.__dict__["__dict__"].__get__, cls)
        var slot = context["owner_array_slots"]["_mv"]
        if py_get(fields, "_mv") is not slot:
            raise Error("TypeError: density hashes require the connected native Array buffer slot")
        view = py_call(slot.__get__, value, cls)
        if py_type(view) is not context["host_iforest_memoryview_type"]:
            raise Error("TypeError: density hash Array storage requires a native memoryview")
    else:
        var admitted = cls is context["host_iforest_memoryview_type"] or cls is context["array_type"] or cls is py_get(context, "ndarray")
        if not admitted and py_get(context, "ndarray") is not None:
            admitted = _one_of(cls, context["scalar_types"])
            if not admitted:
                var constructor = native_numpy_dtype(context)
                var codes = Python.tuple("q", "Q", "g", "F", "D", "G", "S", "U", "V", "M8", "m8")
                for i in range(len(codes)):  # driver-loop: native buffer scalar aliases
                    admitted = admitted or cls is py_call(constructor, codes[i]).type
        if not admitted:
            if _one_of(cls, context["host_hash_missing_tobytes_types"]):
                var name = py_call(py_type(cls).__getattribute__, cls, "__name__")
                raise Error("AttributeError: '" + String(name) + "' object has no attribute 'tobytes'")
            raise Error("TypeError: density hashes require native buffers; Python tobytes protocols cannot execute")
        view = value
    return view


def _density_hash_update(mut hash: Sha256, value: PythonObject) raises:
    # Request shape/strides without a PEP3118 format string. A datetime or
    # timedelta buffer has ordinary bytes even when its dtype has no PEP3118
    # spelling. Public buffer copying preserves logical C order, with no
    # NumPy conversion or scalar arithmetic. Array supplies its physical _mv.
    ref cpy = Python().cpython()
    var view = List[Int](length=10, fill=0)
    var pointer = view.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    var status = cpy.lib.get_function[Int32]("PyObject_GetBuffer")(value._obj_ptr, pointer, Int32(0x118))
    if status != 0:
        var failure = stash_python_error()
        raise failure^
    var size = view[2]
    var bytes = List[UInt8](length=max(size, 1), fill=UInt8(0))
    var destination = bytes.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    status = cpy.lib.get_function[Int32]("PyBuffer_ToContiguous")(destination, pointer, size, Int8(67))
    cpy.lib.get_function[NoneType]("PyBuffer_Release")(pointer)
    if status != 0:
        var failure = stash_python_error()
        raise failure^
    # The active export pinned mutable exporters through copying; this private
    # Mojo owner remains live until SHA-256 finishes consuming every byte.
    hash.update(Int(destination), size)
    _ = bytes^
    _ = view^


def host_density_hash(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    classical_connect_numpy(context)
    _ = host_tree_constructor_fields(model, context)
    var family = String(args[0])
    var plan = context["classical_public_layouts"][py_type(model)]
    var mro = plan["mro"]
    var known = False
    for i in range(len(mro)):  # driver-loop: registered density class ancestry
        var kind = py_get(context["host_inference_kinds"], mro[i])
        known = known or (kind is not None and String(kind) == family)
    if not known or (family != "kde" and family != "iforest"):
        raise Error("TypeError: density hash requires its connected native host class")
    var declared = classical_getattribute(model, PythonObject("_HOST_ARRAYS"), context)
    if not _one_of(py_type(declared), context["host_hash_field_collection_types"]):
        if _one_of(py_type(declared), context["host_hash_noniterable_types"]):
            var name = py_call(py_type(py_type(declared)).__getattribute__, py_type(declared), "__name__")
            raise Error("TypeError: '" + String(name) + "' object is not iterable")
        raise Error("TypeError: density hash field names require native containers without Python iteration")
    var names = _density_hash_call(py_type(Python.list()), declared)
    for i in range(len(names)):  # driver-loop: admit names before native lexical comparison
        if not _one_of(py_type(names[i]), context["host_hash_name_sort_types"]):
            raise Error("TypeError: density hash field names must be native strings")
    _ = _density_hash_call(context["host_hash_list_sort"], names)
    var hash = Sha256()
    for i in range(len(names)):  # driver-loop: sorted saved-state fields, including duplicate names
        if py_type(names[i]) is not py_type(PythonObject("")):
            var name = py_call(py_type(py_type(names[i])).__getattribute__, py_type(names[i]), "__name__")
            raise Error("TypeError: attribute name must be string, not '" + String(name) + "'")
        var value = classical_getattribute(model, names[i], context)
        if value is not None:
            var buffer = _density_hash_buffer(value, context)
            _density_hash_update(hash, buffer)
            _ = buffer^
        _ = value^
    return PythonObject(hash.hexdigest())
