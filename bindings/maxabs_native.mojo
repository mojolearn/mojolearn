# SPDX-License-Identifier: Apache-2.0
# MaxAbsScaler owns its existing x_prep program and complete public operation.
from std.memory import memcpy
from std.os import getenv
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem, py_contains
from bindings.classical_attributes import classical_connect_numpy, classical_check_layout
from bindings.classical_arrays import classical_input, classical_address, classical_typed, classical_is_array, classical_c_order, classical_kind, classical_wrap_array, _store_code
from bindings.classical_pandas import classical_admit_value, native_pandas_exact, native_pandas_f32, _pd_class, _pd_fields, _pd_get
from bindings.classical_validate import classical_refuse_sparse
from bindings.classical_metadata import classical_is_fitted
from bindings.classical_exceptions import classical_not_fitted_class
from bindings.classical_warnings import classical_warn_column
from bindings.causal_exception import causal_borrowed_item, causal_set_exception
from bindings.sequential_admission import _one_of, _ordinary_torch
from bindings.native_torch_admission import native_torch_numpy
from bindings.scaler_native import _input, _ptr, _stage, _copy_row
from bindings.scaler_lifecycle_native import _run, _entries, _public_result
from core.array_helpers_impl import strided_copy_bytes

alias F32P = MutPointer[Float32, MutUntrackedOrigin]


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
    _ = packed^
    return output


def _values(source: PythonObject, context: PythonObject, force_host_rows: Bool = False) raises -> PythonObject:
    if py_type(source) is context["maxabs_rows_class"]:
        var fields = py_fields(source)
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


def maxabs_fit(model: PythonObject, source: PythonObject, target: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    var values = _values(source,context)
    var n = Int(py=values.shape[0])
    var d = Int(py=values.shape[1])
    _target(target,n,context)
    var mode = context["backend_fields"]["_DEFAULT_MODE"]
    var entries = _entries(mode,context)
    var direct = py_get(entries,"x_prep_maxabs_fit_direct")
    var maximum = PythonObject(None)
    var scale = PythonObject(None)
    if direct is not None:
        var output = classical_typed(Python.tuple(2*d),"<f4",context["output_context"])
        _ = py_call(direct,classical_address(values),classical_address(output),Python.list(n,d))
        maximum = _copy_row(_ptr(output),d,context)
        scale = _copy_row(_ptr(output)+d,d,context)
        _ = output^
    else:
        # Existing row blocks are 2048 rows, independent of benchmark shapes.
        var blocked = String(getenv("MOJOLEARN_XPREP_BLOCKED","1").strip()) != "0"
        var blocks = (n+2047)//2048
        var extra = 5*blocks*d+2*d if blocked else 7*d
        var arena = List[Float32](length=n*d+extra,fill=0)
        var ap = F32P(unsafe_from_address=Int(arena.unsafe_ptr()))
        memcpy(dest=ap,src=_ptr(values),count=n*d)
        var sc = n*d
        var ma = 0
        var program = List[Int32]()
        if blocked:
            var part = sc+d
            ma = part+5*blocks*d
            _stage(program,124,blocks*d,0,n,d,part,blocks)
            _stage(program,128,d,part,blocks,d,ma,sc)
        else:
            var st = sc+d
            ma = st+5*d
            _stage(program,1,d,0,n,d,st)
            _stage(program,4,d,ma,1,d,-1,sc,1)
        _run(arena,program,entries)
        maximum = _copy_row(ap+ma,d,context)
        scale = _copy_row(ap+sc,d,context)
        _ = arena^
    py_setitem(fields,"max_abs_",maximum)
    py_setitem(fields,"scale_",scale)
    py_setitem(fields,"n_features_in_",d)
    py_setitem(fields,"n_samples_seen_",n)
    py_setitem(fields,"numeric_mode_",mode)
    py_setitem(fields,"_mojolearn_fitted",True)
    if py_contains(fields,"feature_names_in_"):
        _ = py_call(fields.pop,"feature_names_in_")
    _ = values^
    return model


def maxabs_transform(model: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    if not Bool(py=classical_is_fitted(model)):
        _ = classical_not_fitted_class(context)
        raise Error("NotFittedError: This MaxAbsScaler instance is not fitted yet. Call 'fit' with appropriate arguments before using transform.")
    var fields = py_fields(model)
    if not py_contains(fields,"n_features_in_"):
        raise Error("RuntimeError: mojolearn: this MaxAbsScaler instance is not fitted yet")
    var values = _values(source,context)
    var n = Int(py=values.shape[0])
    var d = Int(py=values.shape[1])
    if py_type(fields["n_features_in_"]) is not Python.import_module("builtins").int or Int(py=fields["n_features_in_"]) != d:
        raise Error("MaxAbsScaler input feature count differs from fit")
    var scale = fields["scale_"]
    if scale is not None:
        scale = _input(scale,1,False,context)
        if Int(py=scale.size) != d:
            raise Error("MaxAbsScaler scale_ must hold one divisor per feature")
    var arena = List[Float32](length=2*n*d+d,fill=0)
    var ap = F32P(unsafe_from_address=Int(arena.unsafe_ptr()))
    memcpy(dest=ap,src=_ptr(values),count=n*d)
    if scale is not None:
        memcpy(dest=ap+n*d,src=_ptr(scale),count=d)
    var program = List[Int32]()
    _stage(program,3,n*d,0,n*d,d,-1,n*d if scale is not None else -1,n*d+d)
    _run(arena,program,_entries(fields["numeric_mode_"],context))
    var output = classical_typed(Python.tuple(n,d),"<f4",context["output_context"])
    memcpy(dest=_ptr(output),src=ap+n*d+d,count=n*d)
    _ = arena^
    _ = values^
    return output


def maxabs_lifecycle_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        var model = call[0]
        var operation = String(call[1])
        var args = call[2]
        var context = call[3]
        classical_connect_numpy(context)
        classical_check_layout(model,context)
        var fields = py_fields(model)
        var result = PythonObject(None)
        if operation == "init":
            py_setitem(fields,"copy",args[0])
        elif operation == "get_params":
            result = Python.dict(copy=fields["copy"])
        elif operation == "set_params":
            var options = args[0]
            var names = py_call(Python.import_module("builtins").list,options)
            for k in range(len(names)):  # driver-loop: proposed parameter names, in declared order
                if py_type(names[k]) is not Python.import_module("builtins").str or String(names[k]) != "copy":
                    raise Error("mojolearn: invalid parameter for MaxAbsScaler")
                py_setitem(fields,names[k],options[names[k]])
            result = model
        elif operation == "is_fitted":
            result = classical_is_fitted(model)
        elif operation == "fit" or operation == "fit_transform":
            if operation == "fit_transform" and len(args[2]) != 0:
                raise Error("TypeError: MaxAbsScaler.fit got an unexpected keyword argument")
            result = maxabs_fit(model,args[0],args[1],context)
            if operation == "fit_transform":
                result = _public_result(maxabs_transform(model,args[0],context),context)
        elif operation == "transform":
            result = _public_result(maxabs_transform(model,args[0],context),context)
        else:
            raise Error("unknown native MaxAbsScaler operation")
        return result.steal_data()
    except error:
        if String(error).startswith("NotFittedError:") or String(error).startswith("DataConversionWarning:"):
            try:
                ref cpy = Python().cpython()
                var context = cpy.PyTuple_GetItem(args_ptr,3)
                var warning = String(error).startswith("DataConversionWarning:")
                var cls = causal_borrowed_item(context,"warning_category" if warning else "active_not_fitted_class")
                return causal_set_exception(error_without_class(error,"DataConversionWarning" if warning else "NotFittedError"),cls)
            except nested:
                return raise_python_exception(nested)
        if String(error).startswith("TypeError:"):
            return raise_python_exception(error_without_class(error,"TypeError"),ExceptionType.TypeError)
        if String(error).startswith("ImportError:"):
            return raise_python_exception(error_without_class(error,"ImportError"),ExceptionType("PyExc_ImportError"))
        if String(error).startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error,"RuntimeError"),ExceptionType("PyExc_RuntimeError"))
        return raise_python_exception(error_without_class(error,"ValueError"),ExceptionType.ValueError)
