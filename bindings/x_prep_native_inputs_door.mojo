# SPDX-License-Identifier: Apache-2.0
# Shared native scaler admission and typed row-snapshot ownership.
from bindings.native_array_storage_door import native_array_from_bytes, native_array_unfilled
from bindings.native_loaded_module_door import native_loaded_module
from bindings.arena_owner_door import ArenaRows, arena_rows, arena_rows_publish_host
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_fields
from bindings.classical_arrays_door import classical_input, classical_address, classical_typed, classical_is_array, classical_c_order, classical_kind, classical_wrap_array, _store_code
from bindings.classical_pandas_door import classical_admit_value, native_pandas_exact, native_pandas_f32, _pd_class, _pd_fields, _pd_get
from bindings.classical_validate_unpack import classical_refuse_sparse
from bindings.classical_warnings_unpack import classical_warn_column
from bindings.sequential_admission_door import _one_of, _ordinary_torch
from bindings.native_torch_admission_door import native_torch_numpy
from bindings.scaler_native_door import _input
from core.array_helpers_impl import strided_copy_bytes
from bindings.py_optional import py_present
from bindings.py_cpython_table import cpython_table  # buffer calls without a dlsym per call (perf2-prep)
from bindings.x_prep_keys_unpack import x_prep_keys
from bindings.py_balanced import py_item


def _target_finite(view: PythonObject, context: PythonObject) raises:
    var item = Int(py=view.itemsize)
    if item == 2:
        # _sk_validate._all_finite scanned only Float32/Float64 target arrays.
        return
    if item != 2 and item != 4 and item != 8:
        raise Error("TypeError: MaxAbsScaler targets require Float16, Float32 or Float64 storage")
    var native = classical_is_array(view,context) or Bool(py=view.dtype.isnative)
    var storage = view._mv if classical_is_array(view,context) else py_call(native_loaded_module("builtins").memoryview,view)
    var size = Int(py=view.size)
    if native:
        # A native-order C-contiguous export of exactly `size` words is
        # already the packed buffer tobytes would build: scan it in place,
        # without the bytes copy and the second array.array copy below.
        # Any other layout (strided, byte-swapped, short export) keeps the
        # packing path. The buffer is released at once; `storage` and `view`
        # own the words through the scan (the xprep_address idiom).
        ref cpy = Python().cpython()
        var buffer = List[Int](length=10,fill=0)
        var bp = buffer.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
        # PyBUF_C_CONTIGUOUS (0x38): only a C-contiguous export succeeds
        if cpython_table()[].PyObject_GetBuffer(storage._obj_ptr,bp,Int32(0x38)) == 0:
            var direct = buffer[0]
            var length = buffer[2]
            cpython_table()[].PyBuffer_Release(bp)
            _ = buffer^  # the Py_buffer words stay alive through the release above
            if length == size*item:
                if Int(py=py_call(context["maxabs_finite_f64" if item == 8 else "maxabs_finite_f32"],direct,size)) != 1:
                    raise Error("Input y contains NaN or infinity (MaxAbsScaler).")
                _ = storage  # keep the exported words alive through the scan above
                return
        else:
            cpython_table()[].PyErr_Clear()
    var packed = py_call(storage.tobytes,"C")
    # Keep the existing device finite scan and Float64 values. Packing and
    # byte-order normalization use the native array buffer implementation.
    var store = native_array_from_bytes(String("d" if item == 8 else "f"), packed)
    if not native:
        _ = py_call(store.byteswap)
    var address = Int(py=py_call(store.buffer_info)[0])
    if Int(py=py_call(context["maxabs_finite_f64" if item == 8 else "maxabs_finite_f32"],address,view.size)) != 1:
        raise Error("Input y contains NaN or infinity (MaxAbsScaler).")
    _ = store^
    _ = packed^


def _target_pandas(value: PythonObject, context: PythonObject) raises -> Int:
    var nk = x_prep_keys()
    var b = native_loaded_module("builtins")
    var manager = py_get(_pd_fields(value),nk[].k__mgr)
    var series = py_type(value) is _pd_class("pandas.core.series","Series")
    var blocks = py_get(_pd_fields(manager),nk[].k_blocks) if series else _pd_get("pandas._libs.internals","BlockManager","blocks",manager)
    if py_type(blocks) is not b.tuple or len(blocks) == 0:
        raise Error("TypeError: target requires native pandas block storage")
    var rows = -1
    var cols = 0
    var object_result = False
    var arrays = Python.list()
    for k in range(len(blocks)):  # driver-loop: over target pandas storage blocks
        var raw = _pd_get("pandas._libs.internals","Block","values",blocks[k])
        if py_type(raw) is not py_item(context, nk[].k_ndarray):
            var cls = py_type(raw)
            if cls is not _pd_class("pandas.core.arrays.integer","IntegerArray") and cls is not _pd_class("pandas.core.arrays.floating","FloatingArray") and cls is not _pd_class("pandas.core.arrays.boolean","BooleanArray"):
                raise Error("TypeError: target requires native pandas numeric or object storage")
            raw = py_get(_pd_fields(raw),nk[].k__data)
            object_result = True
        if py_type(raw) is not py_item(context, nk[].k_ndarray):
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
    var nk = x_prep_keys()
    var b = native_loaded_module("builtins")
    var cls = py_type(value)
    if value is None or cls is b.int or cls is b.float or cls is b.bool or cls is b.str or cls is b.bytes or cls is b.dict:
        return
    if cls is b.complex:
        raise Error("Complex data not supported: MaxAbsScaler got a complex y.")
    var sequence = cls is b.list or cls is b.tuple
    var length = -1
    # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
    var view = Optional[PythonObject]()
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
        examine = examine and (atom is b.float or _one_of(atom,py_item(context, nk[].k_scalar_types)))
        if examine:
            view = classical_input(value,context,"y")
    else:
        classical_admit_value(value,context)
        var pair = _target_view(value,context)
        length = Int(py=pair[0])
        view = py_present(pair[1])
    if Bool(view):
        var kind = classical_kind(view.value(),context)
        if kind == "c":
            raise Error("Complex data not supported: MaxAbsScaler got a complex y.")
        if not sequence and Int(py=view.value().ndim) == 2 and Int(py=view.value().shape[1]) == 1 and cls is not b.memoryview:
            classical_warn_column(context)
        if kind == "f":
            _target_finite(view.value(),context)
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
    var nk = x_prep_keys()
    # DeviceRows' host materialization is stored before the estimator casts
    # its inputs. Preserve original f64/integer words for later consumers.
    var b = native_loaded_module("builtins")
    classical_admit_value(base,context)
    if not classical_is_array(base,context):
        raise Error("TypeError: native DeviceRows requires an ordinary numeric Array base")
    var shape = py_call(py_item(py_item(context, nk[].k_array_fields), nk[].k_shape),base,py_item(context, nk[].k_array_class))
    var dtype_value = py_call(py_item(py_item(context, nk[].k_array_fields), nk[].k_dtype),base,py_item(context, nk[].k_array_class))
    var order = py_call(py_item(py_item(context, nk[].k_array_fields), nk[].k_order),base,py_item(context, nk[].k_array_class))
    if py_type(dtype_value) is not b.str or py_type(shape) is not b.tuple or len(shape) != 2 or py_type(shape[0]) is not b.int or py_type(shape[1]) is not b.int or py_type(order) is not b.str:
        raise Error("TypeError: native DeviceRows requires native numeric Array metadata")
    var dtype = String(dtype_value)
    var code = _store_code(dtype)
    var nr = Int(py=shape[0])
    var d = Int(py=shape[1])
    var n = Int(py=indices.size)
    if nr < 1 or d < 1 or n < 1 or nr > 2147483647//d or n > 2147483647//d:
        raise Error("MaxAbsScaler requires a nonempty row selection within native indexing bounds")
    var view = py_call(py_item(py_item(context, nk[].k_array_fields), nk[].k__mv),base,py_item(context, nk[].k_array_class))
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
    if cpython_table()[].PyObject_GetBuffer(view._obj_ptr,bp,Int32(0)) != 0:
        raise cpy.unsafe_get_error()
    var address = buffer[0]
    cpython_table()[].PyBuffer_Release(bp)
    _ = buffer^  # the Py_buffer words stay alive through the release above
    if String(order) == "F":
        # lane fix-output-zero-fill (2026-11-14): unfilled; strided_copy_bytes writes all nr*d items.
        packed = native_array_unfilled(String(code), nr*d)
        var dims = List[Int64](length=8,fill=0)
        dims[0] = Int64(nr)
        dims[1] = Int64(d)
        dims[2] = 1
        dims[3] = Int64(nr)
        dims[4] = Int64(d)
        dims[5] = 1
        var destination = Int(py=py_call(packed.buffer_info)[0])
        strided_copy_bytes(address,destination,MutPointer[Int64,MutUntrackedOrigin](unsafe_from_address=Int(dims.unsafe_ptr())),2,item,nr*d)
        _ = dims^  # named after the call that writes or reads it through its address (Mojo frees a List at its last named use)
        address = destination
    elif String(order) != "C":
        raise Error("TypeError: native row selection requires C or Fortran Array order")
    # lane fix-output-zero-fill (2026-11-14): unfilled; maxabs_gather_rows writes all n rows of d.
    var store = native_array_unfilled(String(code), n*d)
    var output = classical_wrap_array(store,Python.tuple(n,d),dtype,"C",py_item(context, nk[].k_output_context))
    _ = py_call(py_item(context, nk[].k_maxabs_gather_rows),address,classical_address(output),classical_address(indices),nr,n,item*d)
    # PyBuffer_Release above releases its temporary export, not the native
    # gather's raw address. Retain both source and index owners to that read.
    _ = view^
    _ = indices
    _ = packed^
    return output


def _rows_fields(source: PythonObject, context: PythonObject) raises -> ArenaRows:
    # The fold view of an exact DeviceRows: its native state, resolved once
    # (one slot read through the class's own member descriptor, one lease).
    # No instance dictionary exists, so no foreign key or descriptor can run.
    var nk = x_prep_keys()
    var cls = py_item(context, nk[].k_maxabs_rows_class)
    if py_type(source) is not cls:
        raise Error("TypeError: native row selection requires its ordinary metadata class")
    return arena_rows(source, py_item(context, nk[].k_maxabs_rows_owner), cls)


def _values(source: PythonObject, context: PythonObject, force_host_rows: Bool = False) raises -> PythonObject:
    var nk = x_prep_keys()
    if py_type(source) is py_item(context, nk[].k_maxabs_rows_class):
        var fold = _rows_fields(source,context)
        # DeviceRows.materialize's host-route contract is a single retained
        # snapshot. A later mutation of base/indices must not silently
        # replace an already materialized selection.
        # Every caller of a row selection asks for the host snapshot
        # (`force_host_rows` True at each of them; a device gather is the
        # caller's own ranges path, bindings/x_prep_native_owner_door.mojo
        # `xprep_values`). The process question this function used to ask for
        # a caller that did not (the context's vendor word, the run-time
        # switch MOJOLEARN_ARENA_RANGES) and the unretained host gather it
        # selected had no caller left and are removed (lane dw-prep, round 3):
        # a row selection here is the host snapshot, on every device.
        if fold.dtype == "":
            raise Error("TypeError: native row selection requires native dtype metadata")
        if fold.host is not None:
            return _dense_values(fold.host,context)
        var indices = fold.indices
        classical_admit_value(indices,context)
        if not classical_is_array(indices,context) or String(indices.dtype) != "<i8" or Int(py=indices.ndim) != 1 or not classical_c_order(indices,context):
            raise Error("TypeError: native row selection requires a contiguous int64 index buffer")
        var snapshot = _rows_snapshot(fold.base,indices,context)
        # Published once on the fold's native state (the first snapshot is kept).
        var kept = arena_rows_publish_host(fold.word, snapshot, Int(py=classical_address(snapshot)))
        return _dense_values(kept,context)
    return _dense_values(source,context)


