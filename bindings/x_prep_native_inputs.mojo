# SPDX-License-Identifier: Apache-2.0
# Shared native scaler admission and typed row-snapshot ownership.
from std.os import getenv
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem
from bindings.classical_arrays import classical_input, classical_address, classical_typed, classical_is_array, classical_c_order, classical_kind, classical_wrap_array, _store_code
from bindings.classical_pandas import classical_admit_value, native_pandas_exact, native_pandas_f32, _pd_class, _pd_fields, _pd_get
from bindings.classical_validate import classical_refuse_sparse
from bindings.classical_warnings import classical_warn_column
from bindings.sequential_admission import _one_of, _ordinary_torch
from bindings.native_torch_admission import native_torch_numpy
from bindings.scaler_native import _input
from core.array_helpers_impl import strided_copy_bytes


def _target_finite(view: PythonObject, context: PythonObject) raises:
    var item = Int(py=view.itemsize)
    if item == 2:
        # _sk_validate._all_finite scanned only Float32/Float64 target arrays.
        return
    if item != 2 and item != 4 and item != 8:
        raise Error("TypeError: MaxAbsScaler targets require Float16, Float32 or Float64 storage")
    var native = classical_is_array(view,context) or Bool(py=view.dtype.isnative)
    var storage = view._mv if classical_is_array(view,context) else py_call(Python.import_module("builtins").memoryview,view)
    var packed = py_call(storage.tobytes,"C")
    # Keep the existing device finite scan and Float64 values. Packing and
    # byte-order normalization use the native array buffer implementation.
    var store = py_call(context["array_type"],"d" if item == 8 else "f",packed)
    if not native:
        _ = py_call(store.byteswap)
    var address = Int(py=py_call(store.buffer_info)[0])
    if Int(py=py_call(context["maxabs_finite_f64" if item == 8 else "maxabs_finite_f32"],address,view.size)) != 1:
        raise Error("Input y contains NaN or infinity (MaxAbsScaler).")
    _ = store^
    _ = packed^


def _target_pandas(value: PythonObject, context: PythonObject) raises -> Int:
    var b = Python.import_module("builtins")
    var manager = py_get(_pd_fields(value),"_mgr")
    var series = py_type(value) is _pd_class("pandas.core.series","Series")
    var blocks = py_get(_pd_fields(manager),"blocks") if series else _pd_get("pandas._libs.internals","BlockManager","blocks",manager)
    if py_type(blocks) is not b.tuple or len(blocks) == 0:
        raise Error("TypeError: target requires native pandas block storage")
    var rows = -1
    var cols = 0
    var object_result = False
    var arrays = Python.list()
    for k in range(len(blocks)):  # driver-loop: over target pandas storage blocks
        var raw = _pd_get("pandas._libs.internals","Block","values",blocks[k])
        if py_type(raw) is not context["ndarray"]:
            var cls = py_type(raw)
            if cls is not _pd_class("pandas.core.arrays.integer","IntegerArray") and cls is not _pd_class("pandas.core.arrays.floating","FloatingArray") and cls is not _pd_class("pandas.core.arrays.boolean","BooleanArray"):
                raise Error("TypeError: target requires native pandas numeric or object storage")
            raw = py_get(_pd_fields(raw),"_data")
            object_result = True
        if py_type(raw) is not context["ndarray"]:
            raise Error("TypeError: pandas target storage requires exact native arrays")
        var rank = Int(py=raw.ndim)
        var n = Int(py=raw.shape[rank-1])
        if rows >= 0 and rows != n:
            raise Error("pandas target blocks have different row counts")
        rows = n
        cols += 1 if rank == 1 else Int(py=raw.shape[0])
        var kind = String(raw.dtype.kind)
        if kind == "c":
            raise Error("Complex data not supported: MaxAbsScaler got a complex y.")
        object_result = object_result or kind == "O" or kind == "U" or kind == "S"
        _ = py_call(arrays.append,raw)
    if not series and cols == 1:
        classical_warn_column(context)
    if not object_result:
        for k in range(len(arrays)):  # driver-loop: native target buffer validation, one call per block
            if String(arrays[k].dtype.kind) == "f":
                _target_finite(arrays[k],context)
    return rows


def _target_view(value: PythonObject, context: PythonObject) raises -> PythonObject:
    var prepared = native_torch_numpy(value) if _ordinary_torch(py_type(value),context) else value
    var view = classical_input(prepared,context,"y")
    return Python.tuple(-1 if Int(py=view.ndim) == 0 else Int(py=view.shape[0]),view)


def _target(value: PythonObject, rows: Int, context: PythonObject) raises:
    var b = Python.import_module("builtins")
    var cls = py_type(value)
    if value is None or cls is b.int or cls is b.float or cls is b.bool or cls is b.str or cls is b.bytes or cls is b.dict:
        return
    if cls is b.complex:
        raise Error("Complex data not supported: MaxAbsScaler got a complex y.")
    var sequence = cls is b.list or cls is b.tuple
    var length = -1
    var view = PythonObject(None)
    var examine = True
    if native_pandas_exact(value):
        length = _target_pandas(value,context)
    elif sequence:
        length = len(value)
        var first = value
        while py_type(first) is b.list or py_type(first) is b.tuple:  # driver-loop: over target nesting rank metadata
            if len(first) == 0:
                examine = False
                break
            first = first[0]
        var atom = py_type(first)
        if atom is b.complex:
            raise Error("Complex data not supported: MaxAbsScaler got a complex y.")
        classical_admit_value(value,context)
        examine = examine and (atom is b.float or _one_of(atom,context["scalar_types"]))
        if examine:
            view = classical_input(value,context,"y")
    else:
        classical_admit_value(value,context)
        var pair = _target_view(value,context)
        length = Int(py=pair[0])
        view = pair[1]
    if view is not None:
        var kind = classical_kind(view,context)
        if kind == "c":
            raise Error("Complex data not supported: MaxAbsScaler got a complex y.")
        if not sequence and Int(py=view.ndim) == 2 and Int(py=view.shape[1]) == 1 and cls is not b.memoryview:
            classical_warn_column(context)
        if kind == "f":
            _target_finite(view,context)
    if length >= 0 and length != rows:
        raise Error("Found input variables with inconsistent numbers of samples: ["+String(rows)+", "+String(length)+"]")


def _dense_values(source: PythonObject, context: PythonObject) raises -> PythonObject:
    classical_admit_value(source,context)
    classical_refuse_sparse(source,"MaxAbsScaler")
    var prepared = source
    if native_pandas_exact(source):
        prepared = native_pandas_f32(source,context,False,True)
    elif _ordinary_torch(py_type(source),context):
        prepared = native_torch_numpy(source)
    var values = classical_input(prepared,context)
    if classical_kind(values,context) == "c":
        raise Error("Complex data not supported: MaxAbsScaler got a complex X.")
    if Int(py=values.ndim) != 2:
        raise Error("MaxAbsScaler requires a nonempty two-dimensional input")
    return _input(values,2,False,context)


def _rows_snapshot(base: PythonObject, indices: PythonObject, context: PythonObject) raises -> PythonObject:
    # DeviceRows' host materialization is stored before the estimator casts
    # its inputs. Preserve original f64/integer words for later consumers.
    var b = Python.import_module("builtins")
    classical_admit_value(base,context)
    if not classical_is_array(base,context):
        raise Error("TypeError: native DeviceRows requires an ordinary numeric Array base")
    var shape = py_call(context["array_fields"]["shape"],base,context["array_class"])
    var dtype_value = py_call(context["array_fields"]["dtype"],base,context["array_class"])
    var order = py_call(context["array_fields"]["order"],base,context["array_class"])
    if py_type(dtype_value) is not b.str or py_type(shape) is not b.tuple or len(shape) != 2 or py_type(shape[0]) is not b.int or py_type(shape[1]) is not b.int or py_type(order) is not b.str:
        raise Error("TypeError: native DeviceRows requires native numeric Array metadata")
    var dtype = String(dtype_value)
    var code = _store_code(dtype)
    var nr = Int(py=shape[0])
    var d = Int(py=shape[1])
    var n = Int(py=indices.size)
    if nr < 1 or d < 1 or n < 1 or nr > 2147483647//d or n > 2147483647//d:
        raise Error("MaxAbsScaler requires a nonempty row selection within native indexing bounds")
    var view = py_call(context["array_fields"]["_mv"],base,context["array_class"])
    if py_type(view) is not b.memoryview:
        raise Error("TypeError: native row selection requires a native backing memoryview")
    classical_admit_value(view,context)
    var item = 8 if dtype == "<f8" or dtype == "<i8" else (1 if dtype == "<u1" or dtype == "<i1" else (2 if dtype == "<f2" or dtype == "<u2" else 4))
    if not Bool(py=view.c_contiguous) or Int(py=view.nbytes) != nr*d*item:
        raise Error("TypeError: native row selection requires a complete contiguous Array store")
    var packed = PythonObject(None)
    ref cpy = Python().cpython()
    var buffer = List[Int](length=10,fill=0)
    var bp = buffer.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    if cpy.lib.get_function[Int32]("PyObject_GetBuffer")(view._obj_ptr,bp,Int32(0)) != 0:
        raise cpy.unsafe_get_error()
    var address = buffer[0]
    cpy.lib.get_function[NoneType]("PyBuffer_Release")(bp)
    if String(order) == "F":
        packed = py_call(context["array_type"],code,py_call(b.bytes,nr*d*item))
        var dims = List[Int64](length=8,fill=0)
        dims[0] = Int64(nr)
        dims[1] = Int64(d)
        dims[2] = 1
        dims[3] = Int64(nr)
        dims[4] = Int64(d)
        dims[5] = 1
        var destination = Int(py=py_call(packed.buffer_info)[0])
        strided_copy_bytes(address,destination,MutPointer[Int64,MutUntrackedOrigin](unsafe_from_address=Int(dims.unsafe_ptr())),2,item,nr*d)
        address = destination
    elif String(order) != "C":
        raise Error("TypeError: native row selection requires C or Fortran Array order")
    var store = py_call(context["array_type"],code,py_call(b.bytes,n*d*item))
    var output = classical_wrap_array(store,Python.tuple(n,d),dtype,"C",context["output_context"])
    _ = py_call(context["maxabs_gather_rows"],address,classical_address(output),classical_address(indices),nr,n,item*d)
    # PyBuffer_Release above releases its temporary export, not the native
    # gather's raw address. Retain both source and index owners to that read.
    _ = view^
    _ = indices
    _ = packed^
    return output


def _rows_fields(source: PythonObject, context: PythonObject) raises -> PythonObject:
    # Even an exact DeviceRows can contain foreign dictionary keys. Native
    # dict lookup would invoke their Python equality before buffer admission.
    # Resolve its native dictionary slot without application descriptors.
    var b = Python.import_module("builtins")
    var cls = context["maxabs_rows_class"]
    if py_type(source) is not cls or py_type(cls) is not b.type:
        raise Error("TypeError: native row selection requires its ordinary metadata class")
    var ancestry = py_call(b.type.__dict__["__mro__"].__get__,cls)
    if len(ancestry) != 2 or ancestry[1] is not b.object:
        raise Error("TypeError: native row selection cannot inherit Python lifecycle hooks")
    var mapping = py_call(b.type.__dict__["__dict__"].__get__,cls)
    var keys = py_call(b.list,mapping)
    for k in range(len(keys)):  # driver-loop: class keys precede dictionary-descriptor lookup
        if py_type(keys[k]) is not b.str:
            raise Error("TypeError: native row selection class keys must be native strings")
    var descriptor = context["maxabs_rows_dict"]
    if py_get(mapping,"__dict__") is not descriptor or py_type(descriptor) is not py_type(b.object.__dict__["__class__"]) or descriptor.__objclass__ is not cls:
        raise Error("TypeError: native row selection requires a native instance dictionary slot")
    if py_get(mapping,"__del__") is not None or py_get(mapping,"__getattribute__") is not None:
        raise Error("TypeError: native row selection cannot install Python lifecycle hooks")
    var fields = py_call(descriptor.__get__,source,cls)
    if py_type(fields) is not b.dict:
        raise Error("TypeError: native row selection requires native metadata storage")
    keys = py_call(b.list,fields)
    for k in range(len(keys)):  # driver-loop: reject equality/hash protocols before any row metadata lookup
        if py_type(keys[k]) is not b.str:
            raise Error("TypeError: native row selection metadata keys must be native strings")
    return fields


def _values(source: PythonObject, context: PythonObject, force_host_rows: Bool = False) raises -> PythonObject:
    if py_type(source) is context["maxabs_rows_class"]:
        var fields = _rows_fields(source,context)
        # DeviceRows.materialize's host-route contract is a single retained
        # snapshot. A later mutation of base/indices must not silently
        # replace an already materialized selection. GPU ranges deliberately
        # retain their separate device-gather/resident-base semantics.
        var host_rows = force_host_rows or String(context["vendor"]) == "cpu" or String(getenv("MOJOLEARN_ARENA_RANGES","1").strip()) == "0"
        if py_type(fields["dtype"]) is not Python.import_module("builtins").str:
            raise Error("TypeError: native row selection requires native dtype metadata")
        host_rows = host_rows or String(fields["dtype"]) != "<f4"
        if host_rows and py_get(fields,"_host") is not None:
            return _dense_values(fields["_host"],context)
        var indices = fields["indices"]
        classical_admit_value(indices,context)
        if not classical_is_array(indices,context) or String(indices.dtype) != "<i8" or Int(py=indices.ndim) != 1 or not classical_c_order(indices,context):
            raise Error("TypeError: native row selection requires a contiguous int64 index buffer")
        if host_rows:
            var snapshot = _rows_snapshot(fields["base"],indices,context)
            py_setitem(fields,"_host",snapshot)
            return _dense_values(snapshot,context)
        var base = _dense_values(fields["base"],context)
        var n = Int(py=indices.size)
        var d = Int(py=base.shape[1])
        if n < 1 or n > 2147483647 // d:
            raise Error("MaxAbsScaler requires a nonempty row selection within native indexing bounds")
        var output = classical_typed(Python.tuple(n,d),"<f4",context["output_context"])
        _ = py_call(context["maxabs_gather_rows"],classical_address(base),classical_address(output),classical_address(indices),base.shape[0],n,4*d)
        _ = base^
        return output
    return _dense_values(source,context)


