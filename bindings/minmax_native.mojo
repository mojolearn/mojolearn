# SPDX-License-Identifier: Apache-2.0
# MinMaxScaler owns preparation, launches, running extrema, outputs and state
# here. The preprocessing and x_prep kernels and their order are unchanged.
from std.memory import unsafe_memcpy, bitcast
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.optimizer_public_words import PYTHON_ERROR_PENDING, optimizer_python_error
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_contains, py_setattr
from bindings.native_loaded_module import NativeModules
from bindings.native_prepared_owner import prepared_get, prepared_put
from bindings.mode_metadata_native import _fields
from bindings.py_typed_call import py_call_typed
from bindings.classical_attributes import classical_connect_numpy, classical_check_layout
from bindings.classical_arrays import classical_address, classical_typed, classical_is_array, classical_c_order
from bindings.classical_pandas import classical_admit_value
from bindings.classical_exceptions import classical_not_fitted_class
from bindings.causal_exception import causal_borrowed_item, causal_set_exception
from providers.python.preprocessing_input_pyapi import ScalerInputSnapshot, scaler_operation_input
from bindings.scaler_native import _ptr, _copy_row, scaler_door_calls, scaler_transform_calls, ScalerDoorCalls
from core.scaler_core import (
    ScalerTransformSpec, scaler_minmax_fit_core, scaler_minmax_keep_extrema, scaler_transform_rows,
    scaler_transform_rows_valid, scaler_transform_core,
)
from core.preprocessing_program import prep_program_stage as _stage
from bindings.scaler_lifecycle_native import _run, _direct, _entries, _fitted, _write_back, _public_result
from bindings.minmax_metadata import _flag, minmax_range_pair
from bindings.minmax_archive_native import minmax_archive_load, minmax_archive_save
from bindings.py_optional import py_or_none, py_present
# TOMBSTONE: P1_NATIVE_SKLEARN_TAGS (removed: native code importing sklearn.utils._tags and building its Tag dataclasses; the tags are built by the Python compatibility layer, _mode.sklearn_tags) deleted 2026-11-09 by lane p1-python-callbacks-worst; code recoverable at e68654a757.
# Restore: experiments/removed/P1_NATIVE_SKLEARN_TAGS.patch.

comptime F32P = MutPointer[Float32, MutUntrackedOrigin]


def _configuration_mode(var mode: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    if mode is None:
        mode = context["backend_fields"]["_DEFAULT_MODE"]
    if py_type(mode) is not b.str:
        raise Error("numeric_mode must be fast, identical or None")
    mode = py_call(py_call(mode.strip).lower)
    if String(mode) != "identical" and String(mode) != "fast":
        raise Error("numeric_mode must be fast, identical or None")
    return mode


def _configuration(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    var copy = _flag(fields["copy"],"copy",context)
    var clip = _flag(fields["clip"],"clip",context)
    var mode = _configuration_mode(fields["numeric_mode"],context)
    var pair = minmax_range_pair(fields["feature_range"],context)
    var low = Float64(py=pair[0])
    var high = Float64(py=pair[1])
    if (bitcast[DType.uint64](low) & UInt64(0x7ff0000000000000)) == UInt64(0x7ff0000000000000) or (bitcast[DType.uint64](high) & UInt64(0x7ff0000000000000)) == UInt64(0x7ff0000000000000) or abs(low) > Float64(3.4028234663852886e38) or abs(high) > Float64(3.4028234663852886e38):
        raise Error("feature_range endpoints must be finite Float32 values")
    var lower = Float32(low)
    var upper = Float32(high)
    if not lower < upper:
        raise Error("feature_range must have lower < upper after Float32 conversion")
    # Own completed native scalars for the whole operation. Neither publication
    # nor a combined-fit fallback may reread mutable constructor settings.
    return Python.tuple(mode,Float64(lower),Float64(upper),copy,clip)


def _features(fields: PythonObject) raises -> Int:
    var value = fields["n_features_in_"]
    if py_type(value) is not native_loaded_module("builtins").int:
        raise Error("TypeError: MinMaxScaler n_features_in_ must remain a native integer")
    return Int(py=value)


def _statistic(value: PythonObject, d: Int, name: String, context: PythonObject) raises -> PythonObject:
    classical_admit_value(value,context)
    if not classical_is_array(value,context):
        raise Error("MinMaxScaler " + name + " must remain a finite contiguous Float32 feature vector")
    if String(value.dtype) != "<f4" or Int(py=value.ndim) != 1 or Int(py=value.size) != d or not classical_c_order(value,context):
        raise Error("MinMaxScaler " + name + " must remain a finite contiguous Float32 feature vector")
    return value


comptime _ROWS_SLOT = "minmax_transform_rows_v1"


def _minmax_entries(mode: PythonObject, context: PythonObject) raises -> PythonObject:
    if py_type(mode) is not native_loaded_module("builtins").str:
        raise Error("TypeError: MinMaxScaler numeric_mode_ must remain a native string")
    if Bool(py=py_get(context,"scaler_explicit_host",False)) and String(mode) != "identical":
        raise Error("HostMinMaxScaler runs identical only on the host")
    return _entries(mode,context)


def minmax_initialize(model: PythonObject, args: PythonObject) raises -> PythonObject:
    var fields = _fields(model)
    var names = Python.tuple("feature_range","copy","clip","numeric_mode")
    for k in range(4):  # small-loop(params: constructor metadata)
        py_setitem(fields,names[k],args[k])
    return PythonObject(None)


# TOMBSTONE: `_fit_fill` moved to core/scaler_core.mojo `scaler_minmax_fit_core` (lane
# scaler-typed-core, 2026-11-12); the NaN fill runs once there for the door and the shard worker.


def _publish_extrema(fields: PythonObject, rows: List[Float32], d: Int, context: PythonObject) raises:
    """PACK: the five published MinMax rows (core `scaler_minmax_keep_extrema`)."""
    var rp = F32P(unsafe_from_address=Int(rows.unsafe_ptr()))
    var names = Python.tuple("data_min_","data_max_","data_range_","scale_","min_")
    for k in range(5):  # small-loop(stats: five native statistic row publications)
        py_setitem(fields,names[k],_copy_row(rp+k*d,d,context))
    _ = rows


def _keep_extrema(fields: PythonObject, output: PythonObject, colnan: PythonObject, entries: PythonObject, context: PythonObject) raises:
    var d = Int(py=output.shape[1])
    var calls = ScalerDoorCalls(entries, False)  # the x_prep program only
    var rows = scaler_minmax_keep_extrema(calls, _ptr(output), 0 if colnan is None else Int(py=classical_address(colnan)), d)
    _ = output  # keep the Python objects alive through the native call above
    _ = colnan
    _publish_extrema(fields, rows, d, context)


def minmax_fit(model: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    return minmax_fit(model, source, context, PythonObject(None), PythonObject(None))


def minmax_fit(model: PythonObject, source: PythonObject, context: PythonObject,
               input_cache: PythonObject) raises -> PythonObject:
    return minmax_fit(model, source, context, input_cache, PythonObject(None))


def minmax_fit(model: PythonObject, source: PythonObject, context: PythonObject,
                  input_cache: PythonObject,
                  var configuration: PythonObject) raises -> PythonObject:
    var fields = _fields(model)
    if configuration is None:
        configuration = _configuration(fields,context)
    var mode = configuration[0]
    var names = py_call(native_loaded_module("builtins").list,fields)
    for k in range(len(names)):  # driver-loop: stale fitted attribute names on refit
        if String(names[k]).endswith("_"):
            _ = py_call(fields.pop,names[k])
    var entries = _minmax_entries(mode,context)
    if py_get(entries,"minmax_fit") is None:
        raise Error("ImportError: HostMinMaxScaler training requires _mojolearn_preprocessing_host: " + String(py_get(entries,"scaler_training_error","minmax_fit is unavailable")))
    var calls = scaler_door_calls(entries)
    var snapshot = scaler_operation_input(source, context, input_cache)
    var captured = snapshot.downcast_value_ptr[ScalerInputSnapshot]()
    var values = captured[].values
    var n = captured[].metadata.source.rows
    var d = captured[].metadata.source.cols
    # RUN: the typed core (kernel, NaN fill, extrema publication program).
    var rows = scaler_minmax_fit_core(calls, _ptr(values), n, d, Float32(Float64(py=configuration[1])),
                                      Float32(Float64(py=configuration[2])))
    _ = values  # keep the Python object alive through the native call above
    # PACK
    _publish_extrema(fields, rows, d, context)
    py_setitem(fields,"n_features_in_",d)
    py_setitem(fields,"n_samples_seen_",n)
    py_setitem(fields,"numeric_mode_",mode)
    py_setitem(fields,"feature_range_",Python.tuple(configuration[1],configuration[2]))
    py_setitem(fields,"clip_",configuration[4])
    _ = values^
    return model


def minmax_fit_transform(model: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    return minmax_fit_transform(model, source, context, PythonObject(None), PythonObject(None))


def minmax_fit_transform(model: PythonObject, source: PythonObject, context: PythonObject,
                         input_cache: PythonObject) raises -> PythonObject:
    return minmax_fit_transform(model, source, context, input_cache, PythonObject(None))


def minmax_fit_transform(model: PythonObject, source: PythonObject, context: PythonObject,
                  input_cache: PythonObject,
                  var configuration: PythonObject) raises -> PythonObject:
    """fit_transform on one upload of X: the combined `minmax_fit_transform`
    export runs `minmax_fit_direct` then `minmax_transform_direct` on the
    resident X (the same kernels on the same words) and the fitted rows are
    published by the same c2_mm_keep program as `minmax_fit`. None when this
    route cannot run (no combined export: the host column or
    MOJOLEARN_SCALER_DIRECT=0; or X holds a NaN or an infinity: the two-step
    NaN route), and the caller runs fit then transform."""
    var fields = _fields(model)
    if configuration is None:
        configuration = _configuration(fields,context)
    var mode = configuration[0]
    var entries = _minmax_entries(mode,context)
    var entry = _direct(entries,"minmax_fit_transform")
    if entry is None or py_get(entries,"minmax_fit") is None:
        return PythonObject(None)
    # minmax_fit's refit invalidation comes before X is read.
    var names = py_call(native_loaded_module("builtins").list,fields)
    for k in range(len(names)):  # driver-loop: stale fitted attribute names on refit
        if String(names[k]).endswith("_"):
            _ = py_call(fields.pop,names[k])
    var clip = Bool(py=configuration[4])
    var snapshot = scaler_operation_input(source, context, input_cache)
    var captured = snapshot.downcast_value_ptr[ScalerInputSnapshot]()
    var values = captured[].values
    var n = captured[].metadata.source.rows
    var d = captured[].metadata.source.cols
    var stats = classical_typed(Python.tuple(5,d),"<f4",context["output_context"])
    var output = classical_typed(Python.tuple(n,d),"<f4",context["output_context"])
    var got = Int(py=py_call(entry,classical_address(values),classical_address(stats),classical_address(output),
                             Python.list(n,d,configuration[1],configuration[2],Int(clip))))
    _ = output  # keep the Python object alive through the native call above
    _ = stats  # keep the Python object alive through the native call above
    _ = values  # keep the Python object alive through the native call above
    if got == 0:
        _ = values^
        return PythonObject(None)
    _ = output  # keep the Python object alive through the native calls above
    _ = stats  # keep the Python object alive through the native calls above
    _ = values  # keep the Python object alive through the native calls above
    _keep_extrema(fields,stats,PythonObject(None),entries,context)
    py_setitem(fields,"n_features_in_",d)
    py_setitem(fields,"n_samples_seen_",n)
    py_setitem(fields,"numeric_mode_",mode)
    py_setitem(fields,"feature_range_",Python.tuple(configuration[1],configuration[2]))
    py_setitem(fields,"clip_",clip)
    _ = stats^
    if got < 0:
        raise Error("MinMaxScaler transform overflowed in Float32")
    _ = output  # keep the Python object alive through the native calls above
    return _write_back(source,values,output,Bool(py=configuration[3]),context,snapshot)


def minmax_partial_fit(model: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = _fields(model)
    if not _fitted(fields):
        return minmax_fit(model,source,context)
    var configuration = _configuration(fields,context)
    var old_range = minmax_range_pair(fields["feature_range_"],context)
    if old_range != Python.tuple(configuration[1],configuration[2]):
        raise Error("MinMaxScaler partial_fit: feature_range changed since the first batch")
    var b = native_loaded_module("builtins")
    var fitted_mode = fields["numeric_mode_"]
    var batch = py_call(b.object.__new__,context["scaler_class"])
    var batch_configuration = Python.tuple(_configuration_mode(fitted_mode,context),configuration[1],configuration[2],True,configuration[4])
    _ = minmax_initialize(batch,Python.tuple(Python.tuple(configuration[1],configuration[2]),True,configuration[4],fitted_mode))
    _ = minmax_fit(batch,source,context,PythonObject(None),batch_configuration)
    var fresh = _fields(batch)
    var d = _features(fields)
    if _features(fresh) != d:
        raise Error("MinMaxScaler input feature count differs from fit")
    var arena = List[Float32](length=9*d,fill=0)
    var ap = F32P(unsafe_from_address=Int(arena.unsafe_ptr()))
    var stats = Python.tuple(fields["data_min_"],fields["data_max_"],fresh["data_min_"],fresh["data_max_"])
    for k in range(4):  # small-loop(stats: old/new extrema buffers)
        var value = _statistic(stats[k],d,"extrema",context)
        unsafe_memcpy(dest=ap+k*d,src=_ptr(value),count=d)
        _ = value  # keep the Python object alive through the native call above
        _ = value^
    var entries = _minmax_entries(fitted_mode,context)
    var program = List[Int32]()
    _stage(program,268,d,0,d,2*d,3*d,d,4*d,8*d)
    _run(arena,program,entries)
    var output = classical_typed(Python.tuple(5,d),"<f4",context["output_context"])
    var colnan = _copy_row(ap+8*d,d,context)
    _ = py_call_typed(entries["minmax_fit"],Int(ap+4*d),classical_address(output),Python.list(4,d,configuration[1],configuration[2]))
    _ = output  # keep the Python object alive through the native call above
    _keep_extrema(fields,output,colnan,entries,context)
    if py_type(fields["n_samples_seen_"]) is not b.int or py_type(fresh["n_samples_seen_"]) is not b.int:
        raise Error("TypeError: MinMaxScaler n_samples_seen_ must remain a native integer")
    py_setitem(fields,"n_samples_seen_",fields["n_samples_seen_"] + fresh["n_samples_seen_"])
    py_setitem(fields,"clip_",configuration[4])
    _ = arena^
    _ = output^
    _ = stats^
    return model


def minmax_transform(model: PythonObject, source: PythonObject, inverse: Bool, context: PythonObject) raises -> PythonObject:
    return minmax_transform(model, source, inverse, context, PythonObject(None), PythonObject(None))


def minmax_transform(model: PythonObject, source: PythonObject, inverse: Bool, context: PythonObject,
                     input_cache: PythonObject) raises -> PythonObject:
    return minmax_transform(model, source, inverse, context, input_cache, PythonObject(None))


def minmax_transform(model: PythonObject, source: PythonObject, inverse: Bool, context: PythonObject,
                  input_cache: PythonObject,
                  configuration: PythonObject) raises -> PythonObject:
    var fields = _fields(model)
    var copy = _flag(fields["copy"],"copy",context) if configuration is None else Bool(py=configuration[3])
    if not _fitted(fields):
        _ = classical_not_fitted_class(context)
        raise Error("NotFittedError: MinMaxScaler is not fitted")
    var mode = fields["numeric_mode_"] if configuration is None else configuration[0]
    var entries = _minmax_entries(mode, context)
    var calls = scaler_transform_calls(entries, context)
    var snapshot = scaler_operation_input(source, context, input_cache)
    var captured = snapshot.downcast_value_ptr[ScalerInputSnapshot]()
    var values = captured[].values
    var n = captured[].metadata.source.rows
    var d = captured[].metadata.source.cols
    if d != _features(fields):
        raise Error("MinMaxScaler input feature count differs from fit")
    var bounds = minmax_range_pair(fields["feature_range_"],context) if configuration is None else Python.tuple(configuration[1],configuration[2])
    var clip = _flag(fields["clip_"],"clip_",context) if configuration is None else Bool(py=configuration[4])
    var arena = List[Float32](length=7*d+5, fill=0)
    var ap = F32P(unsafe_from_address=Int(arena.unsafe_ptr()))
    # The derived and validated rows of the fitted min_ / scale_ (the stage
    # 264 + 248 program, core `scaler_transform_rows`) are kept per model in
    # the prepared-owner registry (weakly keyed, outside __dict__), keyed by
    # the identity of min_ and scale_, d and the numeric mode (p3-prep). When
    # the record is taken both Arrays are made read-only (hazard class A): an
    # in-place write through the public view raises NumPy's read-only error;
    # replacing min_ or scale_ (refit, partial_fit, load, assignment) misses
    # and rebuilds.
    var mods = NativeModules()
    var min_value = fields["min_"]
    var scale_value = fields["scale_"]
    var held = prepared_get(mods, _ROWS_SLOT, model)
    if (held is not None and held[0] is min_value and held[1] is scale_value and Int(py=held[3]) == d
            and String(held[4]) == String(mode)):
        unsafe_memcpy(dest=ap, src=_ptr(held[2]), count=7*d+5)
        _ = held  # keep the Python object alive through the copy above
    else:
        var mean = _statistic(min_value,d,"min_",context)
        var scale = _statistic(scale_value,d,"scale_",context)
        arena = scaler_transform_rows(calls, Int(py=classical_address(mean)), Int(py=classical_address(scale)), d, True, True)
        ap = F32P(unsafe_from_address=Int(arena.unsafe_ptr()))
        _ = mean  # keep the Python objects alive through the native call above
        _ = scale
        if scaler_transform_rows_valid(ap, d):
            var rows = classical_typed(Python.tuple(7*d+5), "<f4", context["output_context"])
            unsafe_memcpy(dest=_ptr(rows), src=ap, count=7*d+5)
            _ = rows  # keep the Python object alive through the copy above
            py_setattr(min_value, "_readonly", PythonObject(True))
            py_setattr(scale_value, "_readonly", PythonObject(True))
            prepared_put(mods, _ROWS_SLOT, model, Python.tuple(min_value, scale_value, rows, d, String(mode)))
    var output = classical_typed(Python.tuple(n,d), "<f4", context["output_context"])
    # RUN: the typed core (statistic refusals, kernel, NaN fill and restore,
    # overflow refusal).
    scaler_transform_core(calls, ScalerTransformSpec(True, inverse, True, True, clip,
                                                     Float32(Float64(py=bounds[0])), Float32(Float64(py=bounds[1]))),
                          ap, _ptr(values), n, d, _ptr(output))
    _ = arena^
    return _write_back(source, values, output, copy, context, snapshot)



def minmax_lifecycle_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var all = PythonObject(from_borrowed=args_ptr)
        var model = all[0]
        var operation = String(all[1])
        var args = all[2]
        var context = all[3]
        var b = native_loaded_module("builtins")
        var cls = model if py_type(model) is b.type else py_type(model)
        if py_type(cls) is not b.type:
            raise Error("TypeError: native MinMaxScaler refuses Python metaclass hooks")
        var host = py_get(context["scaler_host_contexts"],cls)
        if host is not None:
            context = host
        classical_connect_numpy(context)
        classical_check_layout(model,context)
        # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
        var result = Optional[PythonObject]()
        if operation == "init":
            result = py_present(minmax_initialize(model,args))
        elif operation == "get_params":
            var fields = _fields(model)
            result = Python.dict(feature_range=fields["feature_range"],copy=fields["copy"],clip=fields["clip"],numeric_mode=fields["numeric_mode"])
        elif operation == "set_params":
            var fields = _fields(model)
            var options = args[0]
            if py_type(options) is not b.dict:
                raise Error("TypeError: MinMaxScaler parameters require an ordinary keyword dictionary")
            var names = py_call(b.list,options)
            var known = Python.tuple("feature_range","copy","clip","numeric_mode")
            for k in range(len(names)):  # driver-loop: proposed estimator parameter names
                if py_type(names[k]) is not b.str or not py_contains(known,names[k]):
                    raise Error("Invalid MinMaxScaler parameters")
            if len(names):
                var values = Python.tuple(py_get(options,"feature_range",fields["feature_range"]),py_get(options,"copy",fields["copy"]),py_get(options,"clip",fields["clip"]),py_get(options,"numeric_mode",fields["numeric_mode"]))
                _ = py_call(fields.clear)
                _ = minmax_initialize(model,values)
            result = py_present(model)
        elif operation == "is_fitted":
            result = PythonObject(_fitted(_fields(model)))
        elif operation == "fit":
            result = py_present(minmax_fit(model,args[0],context))
        elif operation == "partial_fit":
            result = py_present(minmax_partial_fit(model,args[0],context))
        elif operation == "fit_transform":
            if len(args[2]):
                raise Error("TypeError: MinMaxScaler.fit got an unexpected keyword argument")
            var input_cache = Python.list()
            var configuration = _configuration(_fields(model),context)
            result = py_present(minmax_fit_transform(model,args[0],context,input_cache,configuration))
            if not result:
                _ = minmax_fit(model,args[0],context,input_cache,configuration)
                result = py_present(minmax_transform(model,args[0],False,context,input_cache,configuration))
            result = py_present(_public_result(py_or_none(result),context,input_cache))
        elif operation == "transform":
            var input_cache = Python.list()
            result = py_present(_public_result(minmax_transform(model,args[0],Bool(py=args[1]),context,input_cache),context,input_cache))
        elif operation == "save":
            result = py_present(minmax_archive_save(model,args[0],context))
        elif operation == "load":
            result = py_present(minmax_archive_load(model,args[0],context))
        else:
            raise Error("unknown native MinMaxScaler operation")
        return py_or_none(result).steal_data()
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
