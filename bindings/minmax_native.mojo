# SPDX-License-Identifier: Apache-2.0
# MinMaxScaler owns preparation, launches, running extrema, outputs and state
# here. The preprocessing and x_prep kernels and their order are unchanged.
from std.memory import memcpy, bitcast
from std.python import Python, PythonObject
from bindings.optimizer_public_words import PYTHON_ERROR_PENDING, optimizer_python_error
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem, py_contains
from bindings.py_typed_call import py_call_typed
from bindings.classical_attributes import classical_connect_numpy, classical_check_layout
from bindings.classical_arrays import classical_address, classical_typed, classical_is_array, classical_c_order
from bindings.classical_pandas import classical_admit_value
from bindings.classical_exceptions import classical_not_fitted_class
from bindings.causal_exception import causal_borrowed_item, causal_set_exception
from bindings.scaler_native import _input, _ptr, _stage, _copy_row
from bindings.scaler_lifecycle_native import _run, _direct, _entries, _fitted, _write_back, _public_result, _nan_fill
from core.scaler_fit_impl import scaler_all_finite, scaler_refuse_infinite_columns, scaler_feature_indices
from bindings.minmax_metadata import _flag, minmax_range_pair
from bindings.minmax_archive_native import minmax_archive_load, minmax_archive_save
from bindings.scaler_tags_native import scaler_tags

alias F32P = MutPointer[Float32, MutUntrackedOrigin]


def _configuration(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    _ = _flag(fields["copy"],"copy",context)
    _ = _flag(fields["clip"],"clip",context)
    var mode = fields["numeric_mode"]
    var b = Python.import_module("builtins")
    if mode is None:
        mode = context["backend_fields"]["_DEFAULT_MODE"]
    if py_type(mode) is not b.str:
        raise Error("numeric_mode must be fast, identical or None")
    mode = py_call(py_call(mode.strip).lower)
    if String(mode) != "identical" and String(mode) != "fast":
        raise Error("numeric_mode must be fast, identical or None")
    var pair = minmax_range_pair(fields["feature_range"],context)
    var low = Float64(py=pair[0])
    var high = Float64(py=pair[1])
    if (bitcast[DType.uint64](low) & UInt64(0x7ff0000000000000)) == UInt64(0x7ff0000000000000) or (bitcast[DType.uint64](high) & UInt64(0x7ff0000000000000)) == UInt64(0x7ff0000000000000) or abs(low) > Float64(3.4028234663852886e38) or abs(high) > Float64(3.4028234663852886e38):
        raise Error("feature_range endpoints must be finite Float32 values")
    var lower = Float32(low)
    var upper = Float32(high)
    if not lower < upper:
        raise Error("feature_range must have lower < upper after Float32 conversion")
    return Python.tuple(mode,Float64(lower),Float64(upper))


def _features(fields: PythonObject) raises -> Int:
    var value = fields["n_features_in_"]
    if py_type(value) is not Python.import_module("builtins").int:
        raise Error("TypeError: MinMaxScaler n_features_in_ must remain a native integer")
    return Int(py=value)


def _statistic(value: PythonObject, d: Int, name: String, context: PythonObject) raises -> PythonObject:
    classical_admit_value(value,context)
    if not classical_is_array(value,context):
        raise Error("MinMaxScaler " + name + " must remain a finite contiguous Float32 feature vector")
    if String(value.dtype) != "<f4" or Int(py=value.ndim) != 1 or Int(py=value.size) != d or not classical_c_order(value,context):
        raise Error("MinMaxScaler " + name + " must remain a finite contiguous Float32 feature vector")
    return value


def _minmax_entries(mode: PythonObject, context: PythonObject) raises -> PythonObject:
    if py_type(mode) is not Python.import_module("builtins").str:
        raise Error("TypeError: MinMaxScaler numeric_mode_ must remain a native string")
    if Bool(py=py_get(context,"scaler_explicit_host",False)) and String(mode) != "identical":
        raise Error("HostMinMaxScaler runs identical only on the host")
    return _entries(mode,context)


def minmax_initialize(model: PythonObject, args: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    var names = Python.tuple("feature_range","copy","clip","numeric_mode")
    for k in range(4):  # small-loop(params: constructor metadata)
        py_setitem(fields,names[k],args[k])
    return PythonObject(None)


def _fit_fill(values: PythonObject, entries: PythonObject, context: PythonObject) raises -> PythonObject:
    var n = Int(py=values.shape[0])
    var d = Int(py=values.shape[1])
    var st = n*d
    var keep = st+6*d
    var dest = keep+d
    var cn = dest+n*d
    var arena = List[Float32](length=cn+d,fill=0)
    var ap = F32P(unsafe_from_address=Int(arena.unsafe_ptr()))
    memcpy(dest=ap,src=_ptr(values),count=n*d)
    scaler_feature_indices(ap+keep,d)
    var program = List[Int32]()
    _stage(program,1,d,0,n,d,st)
    _stage(program,24,n*d,0,n,d,st+3*d,dest,keep,d)
    _stage(program,266,d,st,cn)
    _run(arena,program,entries)
    scaler_refuse_infinite_columns(ap,st,d)
    var filled = classical_typed(Python.tuple(n,d),"<f4",context["output_context"])
    memcpy(dest=_ptr(filled),src=ap+dest,count=n*d)
    var colnan = _copy_row(ap+cn,d,context)
    _ = arena^
    return Python.tuple(filled,colnan)


def _keep_extrema(fields: PythonObject, output: PythonObject, colnan: PythonObject, entries: PythonObject, context: PythonObject) raises:
    var d = Int(py=output.shape[1])
    var arena = List[Float32](length=12*d+1,fill=0)
    var ap = F32P(unsafe_from_address=Int(arena.unsafe_ptr()))
    if colnan is not None:
        memcpy(dest=ap,src=_ptr(colnan),count=d)
    memcpy(dest=ap+d,src=_ptr(output),count=5*d)
    var program = List[Int32]()
    _stage(program,267,d,d,d,0,6*d,11*d)
    _stage(program,248,1,11*d,d,12*d)
    _run(arena,program,entries)
    if bitcast[DType.int32](arena[12*d]) != 0:
        raise Error("MinMaxScaler fitted statistics overflowed or scale is not positive in Float32")
    var names = Python.tuple("data_min_","data_max_","data_range_","scale_","min_")
    for k in range(5):  # small-loop(stats: five native statistic row publications)
        py_setitem(fields,names[k],_copy_row(ap+(6+k)*d,d,context))
    _ = arena^


def minmax_fit(model: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    var configuration = _configuration(fields,context)
    var mode = configuration[0]
    var names = py_call(Python.import_module("builtins").list,fields)
    for k in range(len(names)):  # driver-loop: stale fitted attribute names on refit
        if String(names[k]).endswith("_"):
            _ = py_call(fields.pop,names[k])
    var entries = _minmax_entries(mode,context)
    if py_get(entries,"minmax_fit") is None:
        raise Error("ImportError: HostMinMaxScaler training requires _mojolearn_preprocessing_host: " + String(py_get(entries,"scaler_training_error","minmax_fit is unavailable")))
    var direct = _direct(entries,"minmax_fit_direct")
    var values = _input(source,2,True,context)
    var n = Int(py=values.shape[0])
    var d = Int(py=values.shape[1])
    var params = Python.list(n,d,configuration[1],configuration[2])
    var output = classical_typed(Python.tuple(5,d),"<f4",context["output_context"])
    var finite = True
    var colnan = PythonObject(None)
    if direct is not None:
        finite = Int(py=py_call(direct,classical_address(values),classical_address(output),params)) != 0
    else:
        finite = scaler_all_finite(_ptr(values),n*d)
        if finite:
            _ = py_call_typed(entries["minmax_fit"],classical_address(values),classical_address(output),params)
    if not finite:
        var filled = _fit_fill(values,entries,context)
        colnan = filled[1]
        _ = py_call_typed(entries["minmax_fit"],classical_address(filled[0]),classical_address(output),params)
        _ = filled^
    _keep_extrema(fields,output,colnan,entries,context)
    py_setitem(fields,"n_features_in_",d)
    py_setitem(fields,"n_samples_seen_",n)
    py_setitem(fields,"numeric_mode_",mode)
    py_setitem(fields,"feature_range_",Python.tuple(configuration[1],configuration[2]))
    py_setitem(fields,"clip_",_flag(fields["clip"],"clip",context))
    _ = output^
    _ = values^
    return model


def minmax_fit_transform(model: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    """fit_transform on one upload of X: the combined `minmax_fit_transform`
    export runs `minmax_fit_direct` then `minmax_transform_direct` on the
    resident X (the same kernels on the same words) and the fitted rows are
    published by the same c2_mm_keep program as `minmax_fit`. None when this
    route cannot run (no combined export: the host column or
    MOJOLEARN_SCALER_DIRECT=0; or X holds a NaN or an infinity: the two-step
    NaN route), and the caller runs fit then transform."""
    var fields = py_fields(model)
    var configuration = _configuration(fields,context)
    var mode = configuration[0]
    var entries = _minmax_entries(mode,context)
    var entry = _direct(entries,"minmax_fit_transform")
    if entry is None or py_get(entries,"minmax_fit") is None:
        return PythonObject(None)
    # minmax_fit's refit invalidation comes before X is read.
    var names = py_call(Python.import_module("builtins").list,fields)
    for k in range(len(names)):  # driver-loop: stale fitted attribute names on refit
        if String(names[k]).endswith("_"):
            _ = py_call(fields.pop,names[k])
    var clip = _flag(fields["clip"],"clip",context)
    var values = _input(source,2,True,context)
    var n = Int(py=values.shape[0])
    var d = Int(py=values.shape[1])
    var stats = classical_typed(Python.tuple(5,d),"<f4",context["output_context"])
    var output = classical_typed(Python.tuple(n,d),"<f4",context["output_context"])
    var got = Int(py=py_call(entry,classical_address(values),classical_address(stats),classical_address(output),
                             Python.list(n,d,configuration[1],configuration[2],Int(clip))))
    if got == 0:
        _ = values^
        return PythonObject(None)
    _keep_extrema(fields,stats,PythonObject(None),entries,context)
    py_setitem(fields,"n_features_in_",d)
    py_setitem(fields,"n_samples_seen_",n)
    py_setitem(fields,"numeric_mode_",mode)
    py_setitem(fields,"feature_range_",Python.tuple(configuration[1],configuration[2]))
    py_setitem(fields,"clip_",clip)
    _ = stats^
    if got < 0:
        raise Error("MinMaxScaler transform overflowed in Float32")
    return _write_back(source,values,output,_flag(fields["copy"],"copy",context),context)


def minmax_partial_fit(model: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    if not _fitted(fields):
        return minmax_fit(model,source,context)
    var configuration = _configuration(fields,context)
    var old_range = minmax_range_pair(fields["feature_range_"],context)
    if old_range != Python.tuple(configuration[1],configuration[2]):
        raise Error("MinMaxScaler partial_fit: feature_range changed since the first batch")
    var b = Python.import_module("builtins")
    var batch = py_call(b.object.__new__,context["scaler_class"])
    _ = minmax_initialize(batch,Python.tuple(fields["feature_range"],True,fields["clip"],fields["numeric_mode_"]))
    _ = minmax_fit(batch,source,context)
    var fresh = py_fields(batch)
    var d = _features(fields)
    if _features(fresh) != d:
        raise Error("MinMaxScaler input feature count differs from fit")
    var arena = List[Float32](length=9*d,fill=0)
    var ap = F32P(unsafe_from_address=Int(arena.unsafe_ptr()))
    var stats = Python.tuple(fields["data_min_"],fields["data_max_"],fresh["data_min_"],fresh["data_max_"])
    for k in range(4):  # small-loop(stats: old/new extrema buffers)
        var value = _statistic(stats[k],d,"extrema",context)
        memcpy(dest=ap+k*d,src=_ptr(value),count=d)
        _ = value^
    var entries = _minmax_entries(fields["numeric_mode_"],context)
    var program = List[Int32]()
    _stage(program,268,d,0,d,2*d,3*d,d,4*d,8*d)
    _run(arena,program,entries)
    var output = classical_typed(Python.tuple(5,d),"<f4",context["output_context"])
    var colnan = _copy_row(ap+8*d,d,context)
    _ = py_call_typed(entries["minmax_fit"],Int(ap+4*d),classical_address(output),Python.list(4,d,configuration[1],configuration[2]))
    _keep_extrema(fields,output,colnan,entries,context)
    if py_type(fields["n_samples_seen_"]) is not b.int or py_type(fresh["n_samples_seen_"]) is not b.int:
        raise Error("TypeError: MinMaxScaler n_samples_seen_ must remain a native integer")
    py_setitem(fields,"n_samples_seen_",fields["n_samples_seen_"] + fresh["n_samples_seen_"])
    py_setitem(fields,"clip_",_flag(fields["clip"],"clip",context))
    _ = arena^
    _ = output^
    _ = stats^
    return model


def minmax_transform(model: PythonObject, source: PythonObject, inverse: Bool, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    var copy = _flag(fields["copy"],"copy",context)
    if not _fitted(fields):
        _ = classical_not_fitted_class(context)
        raise Error("NotFittedError: MinMaxScaler is not fitted")
    var entries = _minmax_entries(fields["numeric_mode_"], context)
    var direct = _direct(entries, "minmax_transform_direct")
    var values = _input(source, 2, True, context)
    var n = Int(py=values.shape[0])
    var d = Int(py=values.shape[1])
    if d != _features(fields):
        raise Error("MinMaxScaler input feature count differs from fit")
    var mean = _statistic(fields["min_"],d,"min_",context)
    var scale = _statistic(fields["scale_"],d,"scale_",context)
    var bounds = minmax_range_pair(fields["feature_range_"],context)
    var clip = _flag(fields["clip_"],"clip_",context)
    var arena = List[Float32](length=7*d+5, fill=0)
    var ap = F32P(unsafe_from_address=Int(arena.unsafe_ptr()))
    memcpy(dest=ap, src=_ptr(mean), count=d)
    memcpy(dest=ap+d, src=_ptr(scale), count=d)
    arena[7*d+4] = 1
    var program = List[Int32]()
    _stage(program, 264, d, 0, d, 1, 1, 7*d+3, 7*d+4, 2*d, 3*d, 4*d, 5*d, 6*d)
    _stage(program, 248, 1, 4*d, d, 7*d)
    _stage(program, 248, 1, 5*d, d, 7*d+1)
    _stage(program, 248, 1, 6*d, d, 7*d+2)
    _run(arena, program, entries)
    if bitcast[DType.int32](arena[7*d+1]) != 0:
        raise Error("MinMaxScaler min_ and scale_ must remain finite (NaN: a feature never seen)")
    if bitcast[DType.int32](arena[7*d+2]) != 0:
        raise Error("MinMaxScaler scale_ must remain positive")
    var colnan = bitcast[DType.int32](arena[7*d]) != 0
    var output = classical_typed(Python.tuple(n,d), "<f4", context["output_context"])
    var params = Python.list(n,d,Int(inverse),Int(clip),bounds[0],bounds[1])
    var finite = True
    if direct is not None:
        var got = Int(py=py_call(direct, classical_address(values), Int(ap+3*d), Int(ap+2*d), classical_address(output), params))
        if got == 0:
            raise Error("MinMaxScaler transform overflowed in Float32")
        finite = got == 1
        if not finite:
            var filled = _nan_fill(values, entries, context)
            if Int(py=py_call(direct, classical_address(filled), Int(ap+3*d), Int(ap+2*d), classical_address(output), params)) != 1:
                raise Error("MinMaxScaler transform overflowed in Float32")
            _ = filled^
    else:
        finite = scaler_all_finite(_ptr(values), n*d)
        var filled = values if finite else _nan_fill(values, entries, context)
        _ = py_call_typed(entries["minmax_transform"], classical_address(filled), Int(ap+3*d), Int(ap+2*d), classical_address(output), params)
        if not scaler_all_finite(_ptr(output), n*d):
            raise Error("MinMaxScaler transform overflowed in Float32")
        _ = filled^
    if not finite or colnan:
        var keep = List[Float32](length=3*n*d+2*d, fill=0)
        var kp = F32P(unsafe_from_address=Int(keep.unsafe_ptr()))
        memcpy(dest=kp, src=_ptr(values), count=n*d)
        memcpy(dest=kp+n*d, src=_ptr(output), count=n*d)
        memcpy(dest=kp+2*n*d, src=ap+4*d, count=d)
        var restore = List[Int32]()
        # Match _Prog.put_codes: nan_keep consumes numeric Float32 flags,
        # not the int32 mask words published by c2_nan_sub.
        _stage(restore, 10, d, 2*n*d, 2*n*d+d)
        _stage(restore, 103, n*d, 0, d, n*d, 2*n*d+d, 2*n*d+2*d)
        _run(keep, restore, entries)
        memcpy(dest=_ptr(output), src=kp+2*n*d+2*d, count=n*d)
        _ = keep^
    _ = arena^
    _ = mean^
    _ = scale^
    return _write_back(source, values, output, copy, context)



def minmax_lifecycle_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var all = PythonObject(from_borrowed=args_ptr)
        var model = all[0]
        var operation = String(all[1])
        var args = all[2]
        var context = all[3]
        var b = Python.import_module("builtins")
        var cls = model if py_type(model) is b.type else py_type(model)
        var host = py_get(context["scaler_host_contexts"],cls)
        if host is not None:
            context = host
        classical_connect_numpy(context)
        classical_check_layout(model,context)
        var result = PythonObject(None)
        if operation == "init":
            result = minmax_initialize(model,args)
        elif operation == "get_params":
            var fields = py_fields(model)
            result = Python.dict(feature_range=fields["feature_range"],copy=fields["copy"],clip=fields["clip"],numeric_mode=fields["numeric_mode"])
        elif operation == "set_params":
            var fields = py_fields(model)
            var options = args[0]
            var names = py_call(b.list,options)
            var known = Python.tuple("feature_range","copy","clip","numeric_mode")
            for k in range(len(names)):  # driver-loop: proposed estimator parameter names
                if py_type(names[k]) is not b.str or not py_contains(known,names[k]):
                    raise Error("Invalid MinMaxScaler parameters")
            if len(names):
                var values = Python.tuple(py_get(options,"feature_range",fields["feature_range"]),py_get(options,"copy",fields["copy"]),py_get(options,"clip",fields["clip"]),py_get(options,"numeric_mode",fields["numeric_mode"]))
                _ = py_call(fields.clear)
                _ = minmax_initialize(model,values)
            result = model
        elif operation == "tags":
            result = scaler_tags(model,context,Python.tuple("float32"))
        elif operation == "is_fitted":
            result = PythonObject(_fitted(py_fields(model)))
        elif operation == "fit":
            result = minmax_fit(model,args[0],context)
        elif operation == "partial_fit":
            result = minmax_partial_fit(model,args[0],context)
        elif operation == "fit_transform":
            if len(args[2]):
                raise Error("TypeError: MinMaxScaler.fit got an unexpected keyword argument")
            result = minmax_fit_transform(model,args[0],context)
            if result is None:
                _ = minmax_fit(model,args[0],context)
                result = minmax_transform(model,args[0],False,context)
            result = _public_result(result,context)
        elif operation == "transform":
            result = _public_result(minmax_transform(model,args[0],Bool(py=args[1]),context),context)
        elif operation == "save":
            result = minmax_archive_save(model,args[0],context)
        elif operation == "load":
            result = minmax_archive_load(model,args[0],context)
        else:
            raise Error("unknown native MinMaxScaler operation")
        return result.steal_data()
    except error:
        if String(error) == PYTHON_ERROR_PENDING:
            return optimizer_python_error(error^)
        if String(error).startswith("NotFittedError:"):
            try:
                ref cpy = Python().cpython()
                var root = cpy.PyTuple_GetItem(args_ptr,3)
                var model = PythonObject(from_borrowed=cpy.PyTuple_GetItem(args_ptr,0))
                var model_cls = py_type(model)
                var hosts = causal_borrowed_item(root,"scaler_host_contexts")
                var host = cpy.PyDict_GetItemWithError(hosts,model_cls._obj_ptr)
                if host:
                    root = host
                var cls = causal_borrowed_item(root,"active_not_fitted_class")
                return causal_set_exception(error_without_class(error,"NotFittedError"),cls)
            except nested:
                return raise_python_exception(nested)
        if String(error).startswith("TypeError:"):
            return raise_python_exception(error_without_class(error,"TypeError"),ExceptionType.TypeError)
        if String(error).startswith("AttributeError:"):
            return raise_python_exception(error_without_class(error,"AttributeError"),ExceptionType("PyExc_AttributeError"))
        if String(error).startswith("ImportError:"):
            return raise_python_exception(error_without_class(error,"ImportError"),ExceptionType("PyExc_ImportError"))
        if String(error).startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error,"RuntimeError"),ExceptionType("PyExc_RuntimeError"))
        return raise_python_exception(error_without_class(error,"ValueError"),ExceptionType.ValueError)
