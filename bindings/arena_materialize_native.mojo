# SPDX-License-Identifier: Apache-2.0
# DeviceRows owns the same typed msel store snapshots as model_selection.
from std.python import Python, PythonObject
from std.math import isinf, isnan
from gemm.f16_widen import f16_bits_to_f32
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_type, py_get, py_setitem
from bindings.native_buffer_owner import native_buffer_view, native_storage_allocate
from bindings.arena_array_metadata import arena_array_metadata
from bindings.classical_arrays import _store_code
from bindings.optimizer_public_words import stash_python_error
from core.array_helpers_impl import strided_copy_bytes


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
    var b = context["arena_native_types"]
    var cls = context["array_class"]
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
    var slots = context["owner_array_slots"]
    var names = py_call(b["list"],slots)
    for i in range(len(names)):  # driver-loop: retained slot setters bypass application __setattr__ and descriptors
        if py_get(mapping,names[i]) is not slots[names[i]]:
            raise Error("TypeError: native row materialization requires the connected Array slots")


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
    var steps = Python.list()
    for i in range(len(shape)):  # driver-loop: publish native strides without an Array method
        _ = py_call(steps.append,strides[i])
    var fields = Python.dict(_store=store,_base=base,_pin=pin,_mv=view,
        _addr=_address(view),_readonly=view.readonly,shape=shape,dtype=PythonObject(dtype),
        ndim=len(shape),size=size,nbytes=size*item,itemsize=item,strides=py_call(b["tuple"],steps),order="C")
    var result = native_storage_allocate(context["array_class"])
    var names = py_call(b["list"],fields)
    for i in range(len(names)):  # driver-loop: initialize fresh native slots; no old owners can be released
        _ = py_call(context["owner_array_slots"][names[i]].__set__,result,fields[names[i]])
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
    var store = py_call(context["array_type"],_store_code(dtype),py_call(context["arena_native_types"]["bytes"],count*item))
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
        if status:
            _cast_failure(Int(py=metadata["address"]),String(metadata["dtype"]),count)
    return target


def _native_item(value: PythonObject, index: PythonObject) raises -> PythonObject:
    ref cpy = Python().cpython()
    var pointer = cpy.lib.get_function[PyObjectPtr]("PyObject_GetItem")(value._obj_ptr,index._obj_ptr)
    if not pointer:
        var failure = stash_python_error()
        raise failure^
    return PythonObject(from_owned=pointer)


def _objects(value: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = context["arena_native_types"]
    var factory = context["native_numpy_dtype"]
    var real_classes = Python.list()
    for code in ["i1","u1","i2","u2","i4","u4","i8","u8","f2","f4","f8","g","q","Q"]:  # driver-loop: exact native real scalar interfaces, excluding object/custom classes
        _ = py_call(real_classes.append,py_call(factory,code).type)
    var values = Python.list()
    var wide = False
    var flat = value.flat
    for i in range(Int(py=value.size)):  # driver-loop: inspect object references before using a native scalar conversion
        var atom = _native_item(flat,PythonObject(i))
        var cls = py_type(atom)
        var real = cls is b["int"] or cls is b["bool"] or cls is b["float"]
        var numpy = False
        for k in range(len(real_classes)):  # driver-loop: fixed native scalar identity set
            numpy = numpy or cls is real_classes[k]
        if not real and not numpy:
            raise Error("TypeError: native row materialization requires ordinary real numeric object elements; Python protocols cannot execute")
        wide = wide or cls is b["float"] or numpy
        _ = py_call(values.append,atom)
    var dtype = String("<f8") if wide else String("<i8")
    var result = _empty(value.shape,dtype,context)
    var address = Int(py=_slot(result,"_addr",context))
    ref cpy = Python().cpython()
    for i in range(len(values)):  # driver-loop: admitted scalar storage conversion, preserving the old float64/int64 choice
        if wide:
            var converted = cpy.lib.get_function[PyObjectPtr]("PyNumber_Float")(values[i]._obj_ptr)
            if not converted:
                _ = cpy.unsafe_get_error()
                raise Error("TypeError: numeric object element exceeds the float64 storage range")
            var scalar = PythonObject(from_owned=converted)
            MutPointer[Float64,MutUntrackedOrigin](unsafe_from_address=address)[i] = Float64(py=scalar)
        else:
            var integer = cpy.lib.get_function[Int64]("PyLong_AsLongLong")(values[i]._obj_ptr)
            if cpy.lib.get_function[PyObjectPtr]("PyErr_Occurred")():
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
    var packed = py_call(view.tobytes,"C")
    var store = py_call(context["array_type"],code,packed)
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


def _prepare(value: PythonObject, context: PythonObject) raises -> PythonObject:
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
            return value
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
    var factory = context["native_numpy_dtype"]
    var known = False
    for code in ["?","i1","u1","i2","u2","i4","u4","i8","u8","f2","f4","f8","O","c8","c16","q","Q"]:  # driver-loop: exact native dtype classes precede dtype access or buffer acquisition
        known = known or py_type(dtype) is py_type(py_call(factory,code))
    if not known:
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
        var store = py_call(context["array_type"],_store_code(typestr),py_call(view.tobytes,"C"))
        return _owned(store,metadata["shape"],typestr,context)
    return _numpy_copy(value,view,kind,item,context)


def _resident_fields(owner: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = context["arena_native_types"]
    if py_type(owner) is not context["arena_resident_class"]:
        raise Error("TypeError: native row materialization requires the ordinary model-selection resident owner")
    var cls = context["arena_resident_class"]
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
    var fields = py_call(context["arena_resident_dict"].__get__,owner,cls)
    keys = py_call(b["list"],fields)
    for i in range(len(keys)):  # driver-loop: resident dictionary lookups cannot run key equality
        if py_type(keys[i]) is not b["str"]:
            raise Error("TypeError: resident metadata requires native string keys")
    if fields["_put"] is not context["arena_msel_put"] or fields["_free"] is not context["arena_msel_free"] or fields["_take"] is not context["arena_msel_take_rows"]:
        raise Error("TypeError: native row materialization requires its connected native resident entries")
    return fields


def _entry(values: PythonObject, entries: PythonObject, indices: Bool, context: PythonObject) raises -> PythonObject:
    var b = context["arena_native_types"]
    var key = py_call(context["arena_id"],values)
    var hit = py_get(entries,key)
    if hit is not None:
        if py_type(hit) is not b["tuple"] or len(hit) != 3:
            raise Error("TypeError: native resident entries require their native owner tuple")
        if hit[0] is values:
            _ = _prepared_info(hit[1],context)
            if py_type(hit[2]) is not b["int"]:
                raise Error("TypeError: native resident entries require an integer store id")
            return hit
        raise Error("TypeError: resident identity entry does not retain its original input")
    var array = _prepare(values,context)
    if indices and String(_slot(array,"dtype",context)) != "<i8":
        array = _cast(array,"<i8",context)
    var sid = -1
    var prepared = _prepared_info(array,context)
    var nbytes = Int(py=prepared["nbytes"])
    if nbytes:
        sid = Int(py=_native_call(context["arena_msel_put"],Python.tuple(prepared["address"],nbytes)))
    try:
        var result = Python.tuple(values,array,sid)
        py_setitem(entries,key,result)
        return result
    except error:
        if sid >= 0:
            # Publication failed after allocation. Free the unpublished id,
            # preserving the primary failure if the backend also refuses
            # cleanup; do not overwrite the shared pending-exception receipt.
            try:
                _ = py_call(context["arena_msel_free"],sid)
            except cleanup_error:
                pass
        raise error^


def arena_materialize(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    var cached = fields["_host"]
    if cached is not None:
        return cached
    var take = fields["_take"]
    if py_type(take) is not context["owner_method_class"] or take.__func__ is not context["arena_resident_take"]:
        raise Error("TypeError: DeviceRows.materialize requires model_selection._Resident.take; Python gather callbacks cannot execute")
    var owner = take.__self__
    var entries = _resident_fields(owner,context)["_entries"]
    var b = context["arena_native_types"]
    if py_type(entries) is not b["dict"]:
        raise Error("TypeError: resident entries require a native dictionary")
    var keys = py_call(b["list"],entries)
    for i in range(len(keys)):  # driver-loop: identity-keyed cache lookups must not invoke application equality
        if py_type(keys[i]) is not b["int"]:
            raise Error("TypeError: resident entries require native identity keys")
    var base_entry = _entry(fields["base"],entries,False,context)
    var index_entry = _entry(fields["indices"],entries,True,context)
    var array = _prepared_info(base_entry[1],context)
    var indices = _prepared_info(index_entry[1],context)
    var shape = Python.list(indices["size"])
    var base_shape = array["shape"]
    for i in range(1,len(base_shape)):  # driver-loop: generic row shape preserves all trailing dimensions
        _ = py_call(shape.append,base_shape[i])
    var output = _empty(py_call(b["tuple"],shape),String(array["dtype"]),context)
    var n = Int(py=base_shape[0])
    var row_bytes = Int(py=array["nbytes"])//n if n else 0
    if Int(py=indices["size"]) and row_bytes:
        _ = _native_call(context["arena_msel_take_rows"],Python.tuple(base_entry[2],n,row_bytes,index_entry[2],indices["size"],_slot(output,"_addr",context)))
    py_setitem(fields,"_host",output)
    return output
