# SPDX-License-Identifier: Apache-2.0
# Native owner for the established _Prog transport contract. Arithmetic and
# stream synchronization remain in the same x_prep exported implementations.
from std.memory import memcpy
from std.os import getenv
from std.time import perf_counter_ns
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem, py_setattr
from bindings.py_typed_call import py_call_typed
from bindings.classical_arrays import classical_address, classical_is_array, classical_c_order, classical_wrap_array
from bindings.sequential_admission import sequential_admit_value
from bindings.x_prep_native_inputs import _dense_values, _values, _rows_fields

alias F32P = MutPointer[Float32, MutUntrackedOrigin]


def xprep_r3(name: String) -> Bool:
    # These three existing R3 choices default on; OFF wins over ON.
    var names = String(getenv("MOJOLEARN_XPREP_R3_OFF","")).replace("+",",").split(",")
    for i in range(len(names)):  # driver-loop: configured transport-switch names
        if String(names[i]).strip() == name or String(names[i]).strip() == "all":
            return False
    return True


def xprep_address(value: PythonObject) raises -> Int:
    # Only retained native array.array/mmap/memoryview storage reaches here.
    ref cpy = Python().cpython()
    var view = List[Int](length=10,fill=0)
    var pointer = view.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    if cpy.lib.get_function[Int32]("PyObject_GetBuffer")(value._obj_ptr,pointer,Int32(0)) != 0:
        raise cpy.unsafe_get_error()
    var address = view[0]
    cpy.lib.get_function[NoneType]("PyBuffer_Release")(pointer)
    return address


def xprep_storage(words: Int, context: PythonObject) raises -> PythonObject:
    var profile = String(getenv("MOJOLEARN_XPREP_PROFILE","0")) == "1"
    var start = Int(perf_counter_ns()) if profile else 0
    var n = max(words,1)
    var b = Python.import_module("builtins")
    var storage = PythonObject(None)
    if n >= 262144 and xprep_r3("mapped"):
        storage = py_call(py_call(b.memoryview,py_call(context["arena_mmap"],-1,4*n)).cast,"f")
    else:
        storage = py_call(context["array_type"],"f",py_call(b.bytes,4*n))
    return Python.tuple(storage,xprep_address(storage),Int(perf_counter_ns())-start if profile else 0)


def xprep_take(storage: PythonObject, offset: Int, rows: Int, cols: Int, context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var shape = Python.tuple(rows) if cols == 0 else Python.tuple(rows,cols)
    var count = rows*(1 if cols == 0 else cols)
    var part = py_call(b.memoryview,storage)[py_call(b.slice,offset,offset+count)]
    if py_type(storage) is b.memoryview and count >= 1048576 and xprep_r3("view"):
        # A large mapped output retains its slice and mapping after the
        # operation returns, exactly as _Prog._take's view path does.
        var result = py_call(b.object.__new__,context["array_class"])
        py_setattr(result,"_store",PythonObject(None))
        py_setattr(result,"_base",storage)
        py_setattr(result,"_pin",PythonObject(None))
        py_setattr(result,"_mv",part)
        py_setattr(result,"_addr",PythonObject(xprep_address(part)))
        py_setattr(result,"_readonly",PythonObject(False))
        py_setattr(result,"shape",shape)
        py_setattr(result,"dtype",PythonObject("<f4"))
        py_setattr(result,"ndim",PythonObject(1 if cols == 0 else 2))
        py_setattr(result,"size",PythonObject(count))
        py_setattr(result,"nbytes",PythonObject(count*4))
        py_setattr(result,"itemsize",PythonObject(4))
        py_setattr(result,"strides",Python.tuple(4) if cols == 0 else Python.tuple(cols*4,4))
        py_setattr(result,"order",PythonObject("C"))
        return result
    var store = storage
    if py_type(storage) is b.memoryview:
        store = py_call(context["array_type"],"f")
        _ = py_call(store.frombytes,py_call(part.cast,"B"))
    elif offset != 0 or len(storage) != count:
        store = storage[py_call(b.slice,offset,offset+count)]
    return classical_wrap_array(store,shape,"<f4","C",context)


def xprep_ranges(entries: PythonObject, scratch: Int) raises -> Bool:
    var device_scratch = scratch == 0 or py_get(entries,"x_prep_run_out") is not None or py_get(entries,"x_prep_run_scratch") is not None
    return device_scratch and String(getenv("MOJOLEARN_ARENA_RANGES","1").strip()) != "0" and py_get(entries,"x_prep_run_ranges") is not None


def xprep_values(source: PythonObject, entries: PythonObject, context: PythonObject) raises -> PythonObject:
    # (kind, native buffer/base, indices, rows, columns). Row handles remain
    # on device when the existing ranges/store ABI is available.
    if py_type(source) is context["maxabs_rows_class"]:
        var fields = _rows_fields(source,context)
        var dtype = fields["dtype"]
        if py_type(dtype) is not Python.import_module("builtins").str:
            raise Error("TypeError: native row selection requires native dtype metadata")
        if String(dtype) == "<f4" and xprep_ranges(entries,1) and py_get(entries,"x_prep_dev_take_rows") is not None and py_get(entries,"x_prep_dev_put") is not None and py_get(entries,"x_prep_dev_free") is not None:
            var base = _dense_values(fields["base"],context)
            var indices = fields["indices"]
            sequential_admit_value(indices,context)
            if not classical_is_array(indices,context) or String(indices.dtype) != "<i8" or Int(py=indices.ndim) != 1 or not classical_c_order(indices,context):
                raise Error("TypeError: native row selection requires a contiguous int64 index buffer")
            var n = Int(py=indices.size)
            var d = Int(py=base.shape[1])
            if n < 1 or n > 2147483647//d:
                raise Error("native row selection must be nonempty and within native indexing bounds")
            return Python.tuple(1,base,indices,n,d)
        # A missing GPU transport capability has the same host snapshot
        # semantics as CPU, without invoking DeviceRows._take.
        var values = _values(source,context,True)
        return Python.tuple(0,values,PythonObject(None),values.shape[0],values.shape[1])
    var values = _values(source,context)
    return Python.tuple(0,values,PythonObject(None),values.shape[0],values.shape[1])


def _cache_layout(cls: PythonObject, descriptor: PythonObject) raises:
    var b = Python.import_module("builtins")
    if py_type(cls) is not b.type:
        raise Error("TypeError: arena caches require native class metadata")
    var ancestry = py_call(b.type.__dict__["__mro__"].__get__,cls)
    if len(ancestry) != 2 or ancestry[1] is not b.object:
        raise Error("TypeError: arena cache classes cannot inherit Python lifecycle hooks")
    var mapping = py_call(b.type.__dict__["__dict__"].__get__,cls)
    var keys = py_call(b.list,mapping)
    for i in range(len(keys)):  # driver-loop: inspect class keys before lookup can compare user keys
        if py_type(keys[i]) is not b.str:
            raise Error("TypeError: arena cache class keys must remain native strings")
    if py_get(mapping,"__del__") is not None or py_get(mapping,"__getattribute__") is not None or py_get(mapping,"__dict__") is not descriptor:
        raise Error("TypeError: arena cache classes cannot install Python lifecycle hooks")


def _cache_fields(value: PythonObject, cls: PythonObject, descriptor: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    if py_type(value) is not cls:
        raise Error("TypeError: arena caches require their native metadata class")
    _cache_layout(cls,descriptor)
    var fields = py_fields(value)
    var names = py_call(b.list,fields)
    for i in range(len(names)):  # driver-loop: inspect cache metadata keys before lookup
        if py_type(names[i]) is not b.str:
            raise Error("TypeError: arena cache keys must remain native strings")
    return fields


def _resident(entries: PythonObject, words: Int, rows: Bool, context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var scope = py_call(b.getattr,context["arena_local"],"scope",PythonObject(None))
    if scope is None or py_get(entries,"x_prep_dev_put") is None or py_get(entries,"x_prep_dev_free") is None:
        return PythonObject(None)
    var fields = _cache_fields(scope,context["arena_scope_class"],context["arena_scope_dict"])
    if py_type(fields["min_words"]) is not b.int or py_type(fields["caches"]) is not b.dict:
        raise Error("TypeError: resident scope requires native cache metadata")
    if not rows and words < Int(py=fields["min_words"]):
        return PythonObject(None)
    var caches = fields["caches"]
    var keys = py_call(b.list,caches)
    for i in range(len(keys)):  # driver-loop: admit resident binding identities before dictionary lookup
        var key = keys[i]
        if py_type(key) is not b.tuple or len(key) != 2 or py_type(key[0]) is not b.int or py_type(key[1]) is not b.str:
            raise Error("TypeError: resident cache keys require native binding identities")
    var key = Python.tuple(entries["arena_binding_id"],"x_prep")
    var pair = py_get(caches,key)
    if pair is None:
        _cache_layout(context["arena_cache_class"],context["arena_cache_dict"])
        var cache = py_call(b.object.__new__,context["arena_cache_class"])
        var state = _cache_fields(cache,context["arena_cache_class"],context["arena_cache_dict"])
        py_setitem(state,"_put",entries["x_prep_dev_put"])
        py_setitem(state,"_free",entries["x_prep_dev_free"])
        py_setitem(state,"_live",py_get(entries,"x_prep_dev_live"))
        py_setitem(state,"_entries",Python.dict())
        py_setitem(state,"uploads",0)
        py_setitem(state,"hits",0)
        pair = Python.tuple(cache,entries["arena_binding"])
        py_setitem(caches,key,pair)
    if py_type(pair) is not b.tuple or len(pair) != 2 or pair[1] is not entries["arena_binding"]:
        raise Error("TypeError: resident cache binding identity changed")
    return pair[0]


def _cached_slot(cache: PythonObject, value: PythonObject, entries: PythonObject, context: PythonObject) raises -> Int:
    var b = Python.import_module("builtins")
    var state = _cache_fields(cache,context["arena_cache_class"],context["arena_cache_dict"])
    if state["_put"] is not entries["x_prep_dev_put"] or state["_free"] is not entries["x_prep_dev_free"] or py_type(state["_entries"]) is not b.dict or py_type(state["uploads"]) is not b.int or py_type(state["hits"]) is not b.int:
        raise Error("TypeError: resident cache entries must retain native binding metadata")
    var stored = state["_entries"]
    var keys = py_call(b.list,stored)
    for i in range(len(keys)):  # driver-loop: native pointer/count keys precede dictionary lookup
        if py_type(keys[i]) is not b.tuple or len(keys[i]) != 2 or py_type(keys[i][0]) is not b.int or py_type(keys[i][1]) is not b.int:
            raise Error("TypeError: resident cache keys require native buffer identities")
    var key = Python.tuple(classical_address(value),value.size)
    var hit = py_get(stored,key)
    if hit is not None:
        if py_type(hit) is not b.tuple or len(hit) != 2 or py_type(hit[0]) is not b.int:
            raise Error("TypeError: resident cache slots must be native integers")
        py_setitem(state,"hits",Int(py=state["hits"])+1)
        return Int(py=hit[0])
    # Allocate the counter before acquiring a device resource. Until cache
    # publication succeeds, this operation still owns the fresh slot.
    var next_uploads = PythonObject(Int(py=state["uploads"])+1)
    var slot = Int(py=py_call_typed(entries["x_prep_dev_put"],key[0],key[1]))
    try:
        py_setitem(stored,key,Python.tuple(slot,value))
    except error:
        try:
            _ = py_call_typed(entries["x_prep_dev_free"],slot)
        except cleanup_error:
            raise Error("RuntimeError: arena cache publication failed: "+String(error)+"; slot "+String(slot)+" cleanup also failed: "+String(cleanup_error))
        raise error
    py_setitem(state,"uploads",next_uploads)
    return slot


def _release_owned(owned: List[Int], entries: PythonObject) -> String:
    var failures = String()
    for i in range(len(owned)):  # driver-loop: attempt every gather/direct release even when an earlier free raises
        var slot = owned[len(owned)-1-i]
        try:
            _ = py_call_typed(entries["x_prep_dev_free"],slot)
        except error:
            failures += ("" if failures.byte_length() == 0 else "; ")+"slot "+String(slot)+": "+String(error)
    return failures


def xprep_run(storage: PythonObject, host: Int, scratch: Int, output: PythonObject, out_words: Int,
              var program: List[Int32], inputs: PythonObject, entries: PythonObject, context: PythonObject,
              logical_output: Int = 0) raises:
    var profile = String(getenv("MOJOLEARN_XPREP_PROFILE","0")) == "1"
    var start = Int(perf_counter_ns()) if profile else 0
    var kernel_start = start
    var arena = Int(py=storage[1])
    var op = 0 if out_words == 0 else Int(py=output[1])
    var device_scratch = scratch > 0 and (py_get(entries,"x_prep_run_out") is not None or py_get(entries,"x_prep_run_scratch") is not None)
    var ranges = xprep_ranges(entries,scratch)
    var ins = List[Int32]()
    var outs = List[Int32]()
    # Each input can own at most a direct base and a gathered selection.
    # Reserve before acquisition so recording a successful put/take cannot
    # allocate and strand a slot on List growth failure.
    var owned = List[Int](capacity=2*len(inputs))
    var last = 0
    try:
        for k in range(len(inputs)):  # driver-loop: admitted input regions and their native owners
            var offset = Int(py=inputs[k][0])
            var record = inputs[k][1]
            var rows = Int(py=record[0]) == 1
            var value = record[1]
            var words = Int(py=record[3])*Int(py=record[4])
            var source = -1
            if ranges:
                var cache = _resident(entries,Int(py=value.size),rows,context)
                if cache is not None:
                    source = _cached_slot(cache,value,entries,context)
                elif py_get(entries,"x_prep_dev_put") is not None and py_get(entries,"x_prep_dev_free") is not None and (rows or (words >= 1048576 and String(getenv("MOJOLEARN_XPREP_DIRECT","1").strip()) != "0")):
                    source = Int(py=py_call_typed(entries["x_prep_dev_put"],classical_address(value),value.size))
                    owned.append(source)
                if rows:
                    source = Int(py=py_call_typed(entries["x_prep_dev_take_rows"],source,record[4],classical_address(record[2]),record[3]))
                    owned.append(source)
            if source < 0:
                memcpy(dest=F32P(unsafe_from_address=arena+4*offset),src=F32P(unsafe_from_address=Int(py=classical_address(value))),count=words)
            if ranges:
                if source < 0 and len(ins) >= 3 and ins[len(ins)-1] < 0 and Int(ins[len(ins)-2]) == offset:
                    ins[len(ins)-2] = Int32(offset+words)
                else:
                    ins.append(Int32(offset))
                    ins.append(Int32(offset+words))
                    ins.append(Int32(source))
                if offset > last:
                    outs.append(Int32(last))
                    outs.append(Int32(offset))
                    outs.append(-1)
                    outs.append(1)
                last = offset+words
        if ranges and last < host:
            outs.append(Int32(last))
            outs.append(Int32(host))
            outs.append(-1)
            outs.append(1)
        var nst = len(program)//16
        var pa = Int(program.unsafe_ptr())
        kernel_start = Int(perf_counter_ns()) if profile else 0
        if ranges:
            _ = py_call_typed(entries["x_prep_run_ranges"],arena,pa,op,Python.tuple(host,scratch if device_scratch else 0,out_words,nst),Python.tuple(Int(ins.unsafe_ptr()),len(ins)//3,Int(outs.unsafe_ptr()),len(outs)//4))
        elif out_words > 0 or (device_scratch and py_get(entries,"x_prep_run_out") is not None):
            _ = py_call_typed(entries["x_prep_run_out"],arena,pa,op,Python.tuple(host,scratch if device_scratch else 0,out_words,nst))
        elif device_scratch:
            _ = py_call_typed(entries["x_prep_run_scratch"],arena,host,scratch,pa,nst)
        else:
            _ = py_call_typed(entries["x_prep_run"],arena,host+scratch,pa,nst)
        # The tuples above retain pointer integers, not their source Lists.
        # Keep both descriptor buffers live until the native call returns;
        # otherwise their last use is tuple construction, before py_call.
        _ = ins^
        _ = outs^
    except error:
        var cleanup = _release_owned(owned,entries)
        if cleanup.byte_length() > 0:
            raise Error("RuntimeError: x_prep operation failed: "+String(error)+"; cleanup also failed: "+cleanup)
        raise error
    var cleanup = _release_owned(owned,entries)
    if cleanup.byte_length() > 0:
        raise Error("RuntimeError: x_prep slot cleanup failed: "+cleanup)
    if profile:
        var done = Int(perf_counter_ns())
        var names = String()
        for i in range(len(program)//16):  # driver-loop: consumed stage metadata for the existing profiling line
            var op = Int(program[i*16])
            var name = String(op)
            if op == 0:
                name = "sort_cols"
            elif op == 1:
                name = "col_stats"
            elif op == 2:
                name = "quantile"
            elif op == 3:
                name = "affine"
            elif op == 4:
                name = "scale_params"
            elif op == 73:
                name = "robust_uv"
            elif op == 124:
                name = "colb_part"
            elif op == 125:
                name = "colb_fold"
            elif op == 128:
                name = "maxabs_fold"
            names += ("" if i == 0 else "+")+name
        print("XPPROG ops="+names+" arena="+String(host)+" scratch="+String(scratch)+" out="+String(logical_output)+" dev_out="+String(Int(out_words > 0))+" ranges="+String(Int(ranges))+" alloc_s="+String(Float64(Int(py=storage[2]))/1.0e9)+" inputs_s="+String(Float64(kernel_start-start)/1.0e9)+" call_s="+String(Float64(done-kernel_start)/1.0e9))
    # Own the caller's command List and retain it through execution even
    # when profiling is disabled. A raw integer address is not an owner.
    _ = program^
