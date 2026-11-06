# SPDX-License-Identifier: Apache-2.0
# DeviceRows owns the same typed msel store snapshots as model_selection.
from bindings.native_array_storage import native_array_zeros
from bindings.arena_buffer_copy import buffer_copy_to_array
from std.python import Python, PythonObject
from std.math import isinf, isnan
from gemm.f16_widen import f16_bits_to_f32
from std.python._cpython import PyObjectPtr
from bindings.cpython_table import cpython_table
from bindings.py_balanced import py_call, py_type, py_get, py_setitem
from bindings.native_buffer_owner import native_buffer_view, native_storage_allocate
from bindings.arena_array_metadata import arena_array_metadata
from bindings.classical_arrays import _store_code
from bindings.optimizer_public_words import stash_python_error
from core.array_helpers_impl import strided_copy_bytes
from bindings.native_loaded_module import native_namespace_string_keys
from bindings.p3_array_arena_records import (
    P3_ARRAY_LAYOUT, P3_RESIDENT_LAYOUT, p3_layout_admitted, p3_layout_admit,
    p3_numpy_scalar_classes, p3_numpy_dtype_class,
)


def _native_call(entry: PythonObject, args: PythonObject) raises -> PythonObject:
    ref cpy = Python().cpython()
    var result = cpy.lib.get_function[PyObjectPtr]("PyObject_CallObject")(entry._obj_ptr,args._obj_ptr)
    if not result:
        var failure = stash_python_error()
        raise failure^
    return PythonObject(from_owned=result)


def _address(view: PythonObject) raises -> Int:
    ref cpy = Python().cpython()
    var storage = List[Int](length=10,fill=0)
    var pointer = storage.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    if cpy.lib.get_function[Int32]("PyObject_GetBuffer")(view._obj_ptr,pointer,Int32(0)) != 0:
        var failure = stash_python_error()
        raise failure^
    var address = storage[0]
    cpy.lib.get_function[NoneType]("PyBuffer_Release")(pointer)
    _ = storage^
    return address


def _array_layout(context: PythonObject) raises:
    var cls = context["array_class"]
    var slots = context["owner_array_slots"]
    # p3-array-arena: admitted once per class state (type watcher), not per
    # call; any class modification re-runs the full check below.
    if p3_layout_admitted(P3_ARRAY_LAYOUT, cls, slots):
        return
    var b = context["arena_native_types"]
    var ancestry = py_call(b["type"].__dict__["__mro__"].__get__,cls)
    if len(ancestry) != 2 or ancestry[1] is not b["object"]:
        raise Error("TypeError: native row materialization requires the ordinary Array layout")
    var mapping = py_call(b["type"].__dict__["__dict__"].__get__,cls)
    var keys = py_call(b["list"],mapping)
    for i in range(len(keys)):  # driver-loop: class lookup cannot invoke application key equality
        if py_type(keys[i]) is not b["str"]:
            raise Error("TypeError: native row materialization requires native Array class keys")
    if py_get(mapping,"__del__") is not None or py_get(mapping,"__getattribute__") is not None:
        raise Error("TypeError: native row materialization cannot invoke Array lifecycle hooks")
    var names = py_call(b["list"],slots)
    for i in range(len(names)):  # driver-loop: retained slot setters bypass application __setattr__ and descriptors
        if py_get(mapping,names[i]) is not slots[names[i]]:
            raise Error("TypeError: native row materialization requires the connected Array slots")
    p3_layout_admit(P3_ARRAY_LAYOUT, cls, slots)


def _itemsize(dtype: String) raises -> Int:
    if dtype == "<f8" or dtype == "<i8":
        return 8
    if dtype == "<f4" or dtype == "<i4" or dtype == "<u4":
        return 4
    if dtype == "<f2" or dtype == "<u2":
        return 2
    if dtype == "<i1" or dtype == "<u1":
        return 1
    raise Error("TypeError: native row materialization requires a supported numeric Array dtype")


def _array(shape: PythonObject, dtype: String, view: PythonObject, store: PythonObject,
           pin: PythonObject, context: PythonObject, base: PythonObject) raises -> PythonObject:
    _array_layout(context)
    var b = context["arena_native_types"]
    var item = _itemsize(dtype)
    var size = 1
    var strides = List[Int](length=len(shape),fill=0)
    for i in range(len(shape)):  # driver-loop: native output rank and contiguous byte-stride metadata
        var axis = len(shape)-1-i
        var extent = Int(py=shape[axis])
        if extent < 0 or size > 9223372036854775807//item or (size > 0 and extent > 9223372036854775807//item//size):
            raise Error("ValueError: native row output exceeds indexing bounds")
        strides[axis] = size*item
        size *= extent
    if Int(py=view.nbytes) != size*item:
        raise Error("ValueError: native row materialization storage and shape disagree")
    var steps = PythonObject(None)
    if len(strides) == 1:
        steps = Python.tuple(strides[0])
    elif len(strides) == 2:
        steps = Python.tuple(strides[0],strides[1])
    else:
        var collected = Python.list()
        for i in range(len(shape)):  # driver-loop: publish native strides without an Array method
            _ = py_call(collected.append,strides[i])
        steps = py_call(b["tuple"],collected)
    # p3-array-arena: the fixed slots are written straight through the
    # retained setters (no field dict, no key list), in the old dict order.
    var result = native_storage_allocate(context["array_class"])
    var slots = context["owner_array_slots"]
    _ = py_call(slots["_store"].__set__,result,store)
    _ = py_call(slots["_base"].__set__,result,base)
    _ = py_call(slots["_pin"].__set__,result,pin)
    _ = py_call(slots["_mv"].__set__,result,view)
    _ = py_call(slots["_addr"].__set__,result,PythonObject(_address(view)))
    _ = py_call(slots["_readonly"].__set__,result,view.readonly)
    _ = py_call(slots["shape"].__set__,result,shape)
    _ = py_call(slots["dtype"].__set__,result,PythonObject(dtype))
    _ = py_call(slots["ndim"].__set__,result,PythonObject(len(shape)))
    _ = py_call(slots["size"].__set__,result,PythonObject(size))
    _ = py_call(slots["nbytes"].__set__,result,PythonObject(size*item))
    _ = py_call(slots["itemsize"].__set__,result,PythonObject(item))
    _ = py_call(slots["strides"].__set__,result,steps)
    _ = py_call(slots["order"].__set__,result,PythonObject("C"))
    return result


def _owned(store: PythonObject, shape: PythonObject, dtype: String, context: PythonObject) raises -> PythonObject:
    return _array(shape,dtype,native_buffer_view(store),store,PythonObject(None),context,PythonObject(None))


def _empty(shape: PythonObject, dtype: String, context: PythonObject) raises -> PythonObject:
    var item = _itemsize(dtype)
    var count = 1
    for i in range(len(shape)):  # driver-loop: allocate only the consumed output shape
        var extent = Int(py=shape[i])
        if extent < 0 or (count > 0 and extent > 9223372036854775807//item//count):
            raise Error("ValueError: native row output exceeds indexing bounds")
        count *= extent
    var store = native_array_zeros(String(_store_code(dtype)), count)
    return _owned(store,shape,dtype,context)


def _buffer_owner(view: PythonObject, context: PythonObject) raises:
    var b = context["arena_native_types"]
    if py_type(view) is not context["arena_memoryview"]:
        raise Error("TypeError: native row materialization requires a native backing memoryview")
    var owner = view.obj
    var cls = py_type(owner)
    if owner is None or cls is b["bytes"] or cls is b["bytearray"] or cls is context["array_type"] or cls is context["owner_mmap_class"] or cls is py_get(context,"ndarray"):
        return
    raise Error("TypeError: native row materialization cannot invoke a Python buffer exporter")


def _cast_code(dtype: String) raises -> Int:
    if dtype == "<f4":
        return 0
    if dtype == "<f8":
        return 1
    if dtype == "<i4":
        return 2
    if dtype == "<i8":
        return 3
    if dtype == "<u4":
        return 4
    if dtype == "<u1":
        return 5
    if dtype == "<u2":
        return 6
    if dtype == "<i1":
        return 7
    if dtype == "<f2":
        return 8
    raise Error("TypeError: native row materialization requires a supported numeric Array dtype")


def _slot(array: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    return py_call(context["owner_array_slots"][PythonObject(name)].__get__,array,context["array_class"])


def _prepared_info(array: PythonObject, context: PythonObject) raises -> PythonObject:
    if py_type(array) is not context["array_class"]:
        raise Error("TypeError: resident entries require the prepared native Array owner")
    var metadata = arena_array_metadata(array,context)
    _array_layout(context)
    var view = _slot(array,"_mv",context)
    _buffer_owner(view,context)
    var order = _slot(array,"order",context)
    if py_type(order) is not context["arena_native_types"]["str"] or String(order) != "C" or String(view.format) != _store_code(String(metadata["dtype"])) or not Bool(py=view.c_contiguous):
        raise Error("TypeError: resident entries require prepared contiguous native storage")
    var item = _itemsize(String(metadata["dtype"]))
    if Int(py=metadata["itemsize"]) != item or Int(py=metadata["size"]) > 9223372036854775807//item:
        raise Error("ValueError: resident storage dtype and item size disagree")
    var nbytes = Int(py=metadata["size"])*item
    if Int(py=view.nbytes) != nbytes:
        raise Error("ValueError: resident storage and shape disagree")
    py_setitem(metadata,"address",_address(view))
    py_setitem(metadata,"nbytes",nbytes)
    return metadata


def _cast_failure(address: Int, source: String, count: Int) raises:
    if source == "<f4" or source == "<f8" or source == "<f2":
        for i in range(count):  # driver-loop: preserve the old ordered int-conversion refusal without a Python reference path
            var number = Float64(0)
            if source == "<f8":
                number = MutPointer[Float64,MutUntrackedOrigin](unsafe_from_address=address)[i]
            elif source == "<f4":
                number = Float64(MutPointer[Float32,MutUntrackedOrigin](unsafe_from_address=address)[i])
            else:
                number = Float64(f16_bits_to_f32(MutPointer[UInt16,MutUntrackedOrigin](unsafe_from_address=address)[i]))
            if isnan(number):
                raise Error("ValueError: cannot convert float NaN to integer")
            if isinf(number):
                raise Error("OverflowError: cannot convert float infinity to integer")
            if number < -9223372036854775808.0 or number >= 9223372036854775808.0:
                raise Error("TypeError: numeric row index exceeds int64 storage")
    raise Error("OverflowError: row index exceeds the target integer range")


def _cast(array: PythonObject, dtype: String, context: PythonObject) raises -> PythonObject:
    var metadata = _prepared_info(array,context)
    var target = _empty(metadata["shape"],dtype,context)
    var count = Int(py=metadata["size"])
    if count:
        var status = Int(py=_native_call(context["arena_cast"],Python.tuple(metadata["address"],_cast_code(String(metadata["dtype"])),_slot(target,"_addr",context),_cast_code(dtype),count)))
        _ = array  # raw source address is borrowed through the complete native cast
        _ = target
        if status:
            try:
                _cast_failure(Int(py=metadata["address"]),String(metadata["dtype"]),count)
            finally:
                _ = array
    return target


def _native_item(value: PythonObject, index: PythonObject) raises -> PythonObject:
    ref cpy = Python().cpython()
    var pointer = cpy.lib.get_function[PyObjectPtr]("PyObject_GetItem")(value._obj_ptr,index._obj_ptr)
    if not pointer:
        var failure = stash_python_error()
        raise failure^
    return PythonObject(from_owned=pointer)


def _objects(value: PythonObject, context: PythonObject) raises -> PythonObject:
    # p3-array-arena: one native pass over the flat iterator (C next, no index
    # objects, no Python list), class checks against native addresses
    # resolved once per process, and the two conversion symbols resolved once
    # per call instead of per element. Same admission, order and storage.
    var b = context["arena_native_types"]
    var numpy_classes = p3_numpy_scalar_classes(context["native_numpy_dtype"])
    var int_class = Int(b["int"]._obj_ptr)
    var bool_class = Int(b["bool"]._obj_ptr)
    var float_class = Int(b["float"]._obj_ptr)
    var count = Int(py=value.size)
    var values = List[PythonObject](capacity=count)
    var wide = False
    ref cpy = Python().cpython()
    var flat = value.flat
    var raw_iterator = cpy.PyObject_GetIter(flat._obj_ptr)
    if not raw_iterator:
        raise cpy.unsafe_get_error()
    var iterator = PythonObject(from_owned=raw_iterator)
    for i in range(count):  # driver-loop: inspect object references before using a native scalar conversion
        var raw = cpy.PyIter_Next(iterator._obj_ptr)
        if not raw:
            if cpy.PyErr_Occurred():
                raise cpy.unsafe_get_error()
            raise Error("RuntimeError: object array iteration ended before its size")
        var atom = PythonObject(from_owned=raw)
        var cls = Int(py_type(atom)._obj_ptr)
        var real = cls == int_class or cls == bool_class or cls == float_class
        var numpy = False
        for k in range(len(numpy_classes)):  # driver-loop: fixed native scalar identity set
            numpy = numpy or cls == numpy_classes[k]
        if not real and not numpy:
            raise Error("TypeError: native row materialization requires ordinary real numeric object elements; Python protocols cannot execute")
        wide = wide or cls == float_class or numpy
        values.append(atom^)
    var dtype = String("<f8") if wide else String("<i8")
    var result = _empty(value.shape,dtype,context)
    var address = Int(py=_slot(result,"_addr",context))
    if wide:
        var number_float = cpy.lib.get_function[PyObjectPtr]("PyNumber_Float")
        for i in range(len(values)):  # driver-loop: admitted scalar storage conversion, preserving the old float64/int64 choice
            var converted = number_float(values[i]._obj_ptr)
            if not converted:
                _ = cpy.unsafe_get_error()
                raise Error("TypeError: numeric object element exceeds the float64 storage range")
            var scalar = PythonObject(from_owned=converted)
            MutPointer[Float64,MutUntrackedOrigin](unsafe_from_address=address)[i] = Float64(py=scalar)
    else:
        var as_long = cpy.lib.get_function[Int64]("PyLong_AsLongLong")
        for i in range(len(values)):  # driver-loop: admitted scalar storage conversion, preserving the old float64/int64 choice
            var integer = as_long(values[i]._obj_ptr)
            if cpython_table()[].PyErr_Occurred():
                _ = cpy.unsafe_get_error()
                raise Error("TypeError: numeric object element exceeds the int64 storage range")
            MutPointer[Int64,MutUntrackedOrigin](unsafe_from_address=address)[i] = integer
    return result


def _numpy_copy(value: PythonObject, view: PythonObject, kind: String, item: Int,
                context: PythonObject) raises -> PythonObject:
    var b = context["arena_native_types"]
    var dtype = "<"+kind+String(item)
    var target = dtype
    var code = String()
    if kind == "b":
        code = "B"
        target = "<u1"
    elif kind == "i" and item == 2:
        code = "h"
        target = "<i4"
    elif kind == "u" and item == 8:
        code = "Q"
        target = "<i8"
    else:
        code = _store_code(dtype)
        # Match _materialize's existing natural-width copy rule. Contiguous
        # native uint16 stays uint16; strided/endian copies become uint32.
        if dtype == "<u2":
            target = "<u4"
    # One native C-order copy into the owned store (was tobytes then frombytes).
    var store = buffer_copy_to_array(String(code), view)
    if not Bool(py=value.dtype.isnative) and item > 1:
        _ = py_call(store.byteswap)
    var count = Int(py=value.size)
    var shape = value.shape
    if len(shape) == 0:
        shape = Python.tuple(1)
    var address = Int(py=py_call(store.buffer_info)[0])
    if target != dtype or kind == "b":
        var out = _empty(shape,target,context)
        var output = Int(py=_slot(out,"_addr",context))
        if kind == "u" and item == 2:
            # p3-array-arena: uint16 -> uint32 widening is exact; run it in the
            # connected chunked cast entry (codes 6 -> 4) instead of a serial
            # host loop. bool, int16 and uint64 have no source code there yet.
            if count:
                _ = _native_call(context["arena_cast"],Python.tuple(address,6,output,4,count))
            _ = store^
            return out
        for i in range(count):  # driver-loop: existing native-width buffer conversion without NumPy arithmetic
            if kind == "b":
                MutPointer[UInt8,MutUntrackedOrigin](unsafe_from_address=output)[i] = UInt8(MutPointer[UInt8,MutUntrackedOrigin](unsafe_from_address=address)[i] != 0)
            elif kind == "i":
                MutPointer[Int32,MutUntrackedOrigin](unsafe_from_address=output)[i] = Int32(MutPointer[Int16,MutUntrackedOrigin](unsafe_from_address=address)[i])
            elif item == 2:
                MutPointer[UInt32,MutUntrackedOrigin](unsafe_from_address=output)[i] = UInt32(MutPointer[UInt16,MutUntrackedOrigin](unsafe_from_address=address)[i])
            else:
                var word = MutPointer[UInt64,MutUntrackedOrigin](unsafe_from_address=address)[i]
                if word > UInt64(9223372036854775807):
                    raise Error("OverflowError: unsigned fold data exceeds int64 storage")
                MutPointer[Int64,MutUntrackedOrigin](unsafe_from_address=output)[i] = Int64(word)
        _ = store^
        return out
    if not Bool(py=view.contiguous) and Bool(py=value.dtype.isnative) and count:
        if dtype == "<f4":
            _ = _native_call(context["arena_cast"],Python.tuple(address,0,address,0,count))
        elif dtype == "<f2" and Bool(py=context["arena_half_copy_canonical"]):
            # The established strided half path round-trips through struct's
            # native float codec. CPython 3.11 canonicalizes NaN payloads;
            # 3.14 preserves them. Its import-time codec metadata preserves
            # that existing behavior without calling struct in a product op.
            var words = MutPointer[UInt16,MutUntrackedOrigin](unsafe_from_address=address)
            for i in range(count):  # driver-loop: preserve the prior half storage conversion's NaN bits
                if (words[i] & 0x7FFF) > 0x7C00:
                    words[i] = (words[i] & 0x8000) | 0x7E00
    return _owned(store,shape,target,context)


def _prepare(value: PythonObject, context: PythonObject, snapshot: Bool = False) raises -> PythonObject:
    var metadata = arena_array_metadata(value,context)
    var b = context["arena_native_types"]
    if py_type(value) is context["array_class"]:
        _array_layout(context)
        var view = py_call(context["owner_array_slots"]["_mv"].__get__,value,context["array_class"])
        _buffer_owner(view,context)
        var order = py_call(context["owner_array_slots"]["order"].__get__,value,context["array_class"])
        if py_type(order) is not b["str"] or (String(order) != "C" and String(order) != "F"):
            raise Error("TypeError: native row materialization requires native Array order metadata")
        var dtype = String(metadata["dtype"])
        var code = _store_code(dtype)
        var item = _itemsize(dtype)
        if Int(py=metadata["itemsize"]) != item or Int(py=metadata["size"]) > 9223372036854775807//item:
            raise Error("ValueError: native row storage dtype and item size disagree")
        if String(view.format) != code or not Bool(py=view.c_contiguous) or Int(py=view.nbytes) != Int(py=metadata["size"])*Int(py=metadata["itemsize"]):
            raise Error("TypeError: native row materialization requires a complete typed Array backing buffer")
        if String(order) == "C":
            if not snapshot:
                return value
            # A public operation may retain this view across native calls.
            # Capture the caller's metadata and independent backing owners once;
            # retaining only the mutable Array object would permit its slots to
            # replace the buffer/shape while the operation still uses them.
            var slots = context["owner_array_slots"]
            var cls = context["array_class"]
            var store = py_call(slots["_store"].__get__,value,cls)
            var pin = py_call(slots["_pin"].__get__,value,cls)
            var base = py_call(slots["_base"].__get__,value,cls)
            var readonly = py_call(slots["_readonly"].__get__,value,cls)
            if py_type(readonly) is not b["bool"]:
                raise Error("TypeError: native row materialization requires boolean Array readonly metadata")
            # A reference to the caller's memoryview alone is insufficient:
            # memoryview.release() can invalidate that exact view. Independent
            # native exports keep the original storage pinned until completion.
            var snapshot_view = native_buffer_view(view)
            if py_type(store) is b["memoryview"]:
                store = native_buffer_view(store)
            if py_type(pin) is b["memoryview"]:
                pin = native_buffer_view(pin)
            if py_type(base) is b["memoryview"]:
                base = native_buffer_view(base)
            var output = _array(metadata["shape"],dtype,snapshot_view,store,pin,context,Python.tuple(value,base))
            _ = py_call(slots["_readonly"].__set__,output,PythonObject(Bool(py=readonly) or Bool(py=view.readonly)))
            return output
        var shape = metadata["shape"]
        var nonunit = 0
        for i in range(len(shape)):  # driver-loop: preserve _as_c's zero-copy view when C and F layouts coincide
            nonunit += Int(Int(py=shape[i]) != 1)
        if nonunit <= 1:
            var slots = context["owner_array_slots"]
            return _array(shape,dtype,view,py_call(slots["_store"].__get__,value,context["array_class"]),py_call(slots["_pin"].__get__,value,context["array_class"]),context,value)
        var output = _empty(shape,dtype,context)
        var rank = len(shape)
        var descriptor = List[Int64](length=3*rank+2,fill=0)
        var source_step = 1
        var output_step = 1
        for i in range(rank):  # driver-loop: generic Fortran-to-C shape traversal preserves the existing typed bytes
            descriptor[i] = Int64(Int(py=shape[i]))
            descriptor[rank+i] = Int64(source_step)
            source_step *= Int(py=shape[i])
            descriptor[2*rank+rank-1-i] = Int64(output_step)
            output_step *= Int(py=shape[rank-1-i])
        if Int(py=metadata["size"]):
            strided_copy_bytes(_address(view),Int(py=_slot(output,"_addr",context)),MutPointer[Int64,MutUntrackedOrigin](unsafe_from_address=Int(descriptor.unsafe_ptr())),rank,Int(py=metadata["itemsize"]),Int(py=metadata["size"]))
        _ = descriptor^
        return output
    var dtype = value.dtype
    # p3-array-arena: the 17 exact dtype classes are resolved once per process
    # (bindings/p3_array_arena_records.mojo), not constructed per call.
    if not p3_numpy_dtype_class(context["native_numpy_dtype"], py_type(dtype)):
        raise Error("TypeError: native row materialization requires ordinary supported numeric dtype metadata")
    var kind = String(dtype.kind)
    var item = Int(py=dtype.itemsize)
    if kind == "O":
        return _objects(value,context)
    if kind == "c":
        raise Error("ValueError: Complex data not supported: fold data is complex")
    if (kind != "f" and kind != "i" and kind != "u" and kind != "b") or (item != 1 and item != 2 and item != 4 and item != 8) or (kind == "f" and item == 1):
        raise Error("TypeError: native row materialization requires a supported numeric buffer")
    var view = native_buffer_view(value)
    var native = Bool(py=dtype.isnative)
    var supported = kind != "b" and not (kind == "i" and item == 2) and not (kind == "u" and item == 8)
    if native and supported and len(metadata["shape"]) > 0 and Bool(py=view.c_contiguous):
        var typestr = "<"+kind+String(item)
        if Int(py=metadata["size"]) == 0:
            return _empty(metadata["shape"],typestr,context)
        var flat = py_call(py_call(view.cast,"B").cast,_store_code(typestr))
        return _array(metadata["shape"],typestr,flat,value,view,context,PythonObject(None))
    # F-contiguous native buffers already have an Array dtype; the existing
    # _as_c copy preserves that width (including uint16 and half NaN bits).
    if native and supported and len(metadata["shape"]) > 0 and Bool(py=view.f_contiguous):
        var typestr = "<"+kind+String(item)
        var store = buffer_copy_to_array(String(_store_code(typestr)), view)
        return _owned(store,metadata["shape"],typestr,context)
    return _numpy_copy(value,view,kind,item,context)


def _resident_fields(owner: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = context["arena_native_types"]
    if py_type(owner) is not context["arena_resident_class"]:
        raise Error("TypeError: native row materialization requires the ordinary model-selection resident owner")
    var cls = context["arena_resident_class"]
    # p3-array-arena: the class half is admitted once per class state.
    if not p3_layout_admitted(P3_RESIDENT_LAYOUT, cls, context["arena_resident_dict"]):
        var ancestry = py_call(b["type"].__dict__["__mro__"].__get__,cls)
        if len(ancestry) != 2 or ancestry[1] is not b["object"]:
            raise Error("TypeError: native row materialization requires the ordinary resident layout")
        var mapping = py_call(b["type"].__dict__["__dict__"].__get__,cls)
        var keys = py_call(b["list"],mapping)
        for i in range(len(keys)):  # driver-loop: class keys precede lifetime/descriptor lookup
            if py_type(keys[i]) is not b["str"]:
                raise Error("TypeError: resident metadata requires native string keys")
        if py_get(mapping,"__dict__") is not context["arena_resident_dict"] or py_get(mapping,"__del__") is not None or py_get(mapping,"__getattribute__") is not None:
            raise Error("TypeError: native row materialization cannot invoke resident owner hooks")
        p3_layout_admit(P3_RESIDENT_LAYOUT, cls, context["arena_resident_dict"])
    var fields = py_call(context["arena_resident_dict"].__get__,owner,cls)
    # Instance keys: one native dictionary walk, no key list.
    if py_type(fields) is not b["dict"] or not native_namespace_string_keys(fields._obj_ptr, b["str"]):
        raise Error("TypeError: resident metadata requires native string keys")
    if fields["_put"] is not context["arena_msel_put"] or fields["_free"] is not context["arena_msel_free"] or fields["_take"] is not context["arena_msel_take_rows"]:
        raise Error("TypeError: native row materialization requires its connected native resident entries")
    return fields
