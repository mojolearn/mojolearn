# SPDX-License-Identifier: Apache-2.0
# MinMaxScaler owns preparation, launches, running extrema, outputs and state
# here. The preprocessing and x_prep kernels and their order are unchanged.
from bindings.mode_answer_door import answer_mode_object
from core.mode_answer import MODE_FAMILY_SCALER
from std.memory import unsafe_memcpy, bitcast
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.optimizer_public_words_unpack import PYTHON_ERROR_PENDING, optimizer_python_error
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_contains, py_setattr
from bindings.native_loaded_module_door import NativeModules
from bindings.native_prepared_owner_unpack import prepared_get, prepared_put
from bindings.mode_metadata_native_door import _fields
from bindings.classical_attributes_door import classical_connect_numpy, classical_check_layout
from bindings.classical_arrays_door import classical_address, classical_typed, classical_is_array, classical_c_order
from bindings.classical_pandas_door import classical_admit_value
from bindings.classical_exceptions_door import classical_not_fitted_class
from bindings.causal_exception_unpack import causal_borrowed_item, causal_set_exception
from providers.python.preprocessing_input_pyapi_door import ScalerInputSnapshot, scaler_operation_input
from bindings.scaler_native_door import _ptr, _copy_row, scaler_door_calls, scaler_transform_calls, ScalerDoorCalls
from core.scaler_core import (
    ScalerTransformSpec, scaler_minmax_fit_core, scaler_minmax_keep_extrema, scaler_transform_rows,
    scaler_minmax_partial_fit_run,
    scaler_transform_rows_valid, scaler_transform_core,
)
from bindings.scaler_lifecycle_native_door import _direct, _entries, _fitted, _write_back, _public_result
from bindings.minmax_metadata_unpack import _flag
from bindings.minmax_metadata_pack import minmax_range_pair
from bindings.minmax_archive_native_unpack import minmax_archive_load, minmax_archive_save
from bindings.py_optional import py_or_none, py_present
from bindings.minmax_keys_unpack import (
    minmax_keys, MINMAX_OPN_INIT, MINMAX_OPN_GET_PARAMS, MINMAX_OPN_SET_PARAMS, MINMAX_OPN_IS_FITTED, MINMAX_OPN_FIT,
    MINMAX_OPN_PARTIAL_FIT, MINMAX_OPN_FIT_TRANSFORM, MINMAX_OPN_TRANSFORM, MINMAX_OPN_SAVE, MINMAX_OPN_LOAD,
    MINMAX_OPN_VENDOR_USED,
)
from bindings.py_keys import py_op_code
from bindings.scaler_route_unpack import (
    scaler_route_context, scaler_route_clear, scaler_device_admit, scaler_route_vendor_used, scaler_selection_held,
    scaler_selection_keep, scaler_route_refuse_fast,
)
from bindings.py_balanced import py_item
# TOMBSTONE: P1_NATIVE_SKLEARN_TAGS (removed: native code importing sklearn.utils._tags and building its Tag dataclasses; the tags are built by the Python compatibility layer, _mode.sklearn_tags) deleted 2026-11-09 by lane p1-python-callbacks-worst; code recoverable at e68654a757.
# Restore: experiments/removed/P1_NATIVE_SKLEARN_TAGS.patch.

comptime F32P = MutPointer[Float32, MutUntrackedOrigin]


def _configuration_mode(var mode: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    if mode is None:
        var nk = minmax_keys()
        mode = py_item(py_item(context, nk[].k_backend_fields), nk[].k__DEFAULT_MODE)
    if py_type(mode) is not b.str:
        raise Error("numeric_mode must be fast, identical or None")
    mode = py_call(py_call(mode.strip).lower)
    var mode_word = String(mode)
    if mode_word != "identical" and mode_word != "fast":
        raise Error("numeric_mode must be fast, identical or None")
    answer_mode_object(mode, mode_word, MODE_FAMILY_SCALER)
    return mode


def _configuration(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = minmax_keys()
    var copy = _flag(py_item(fields, nk[].k_copy),"copy",context)
    var clip = _flag(py_item(fields, nk[].k_clip),"clip",context)
    var mode = _configuration_mode(py_item(fields, nk[].k_numeric_mode),context)
    var pair = minmax_range_pair(py_item(fields, nk[].k_feature_range),context)
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
    var nk = minmax_keys()
    var value = py_item(fields, nk[].k_n_features_in_)
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


def minmax_rows(calls: ScalerDoorCalls, min_value: PythonObject, scale_value: PythonObject, d: Int,
                context: PythonObject) raises -> List[Float32]:
    """UNPACK the fitted min_ / scale_ rows and RUN the core's derived-rows
    program; shared by the public transform and the shard worker."""
    var center = _statistic(min_value,d,"min_",context)
    var spread = _statistic(scale_value,d,"scale_",context)
    var rows = scaler_transform_rows(calls, Int(py=classical_address(center)), Int(py=classical_address(spread)), d, True, True)
    _ = center  # keep the Python objects alive through the native call above
    _ = spread
    return rows^


def minmax_fit_publish(fields: PythonObject, rows: List[Float32], n: Int, d: Int, configuration: PythonObject,
                       context: PythonObject) raises:
    """PACK: a MinMaxScaler fit's fitted fields; shared by the public fit and
    the shard worker."""
    _publish_extrema(fields, rows, d, context)
    var nk = minmax_keys()
    py_setitem(fields,nk[].k_n_features_in_,d)
    py_setitem(fields,nk[].k_n_samples_seen_,n)
    py_setitem(fields,nk[].k_numeric_mode_,configuration[0])
    py_setitem(fields,nk[].k_feature_range_,Python.tuple(configuration[1],configuration[2]))
    py_setitem(fields,nk[].k_clip_,configuration[4])


def _minmax_entries(mode: PythonObject, context: PythonObject) raises -> PythonObject:
    if py_type(mode) is not native_loaded_module("builtins").str:
        raise Error("TypeError: MinMaxScaler numeric_mode_ must remain a native string")
    var nk = minmax_keys()
    if Bool(py=py_get(context,nk[].k_scaler_explicit_host,False)) and String(mode) != "identical":
        raise Error("HostMinMaxScaler runs identical only on the host")
    return _entries(mode,context)


def _minmax_entries(model: PythonObject, mode: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_minmax_entries` of an estimator's operation: the selection its device
    route holds for `mode` (one record read and pointer compares), or the
    selection made here by the same checks, in their order and words, and
    kept in the route. A scratch model without a route (partial_fit's batch)
    selects as before."""
    var held = scaler_selection_held(model, mode, context)
    if held is not None:
        return held
    scaler_route_refuse_fast(model, mode, context)
    var entries = _minmax_entries(mode, context)
    scaler_selection_keep(model, mode, entries, context)
    return entries


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
    var entries = _minmax_entries(model,mode,context)
    var nk = minmax_keys()
    if py_get(entries,nk[].k_minmax_fit) is None:
        raise Error("ImportError: HostMinMaxScaler training requires _mojolearn_preprocessing_host: " + String(py_get(entries,nk[].k_scaler_training_error,"minmax_fit is unavailable")))
    var calls = scaler_door_calls(model, entries, context)
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
    minmax_fit_publish(fields, rows, n, d, configuration, context)
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
    var entries = _minmax_entries(model,mode,context)
    var entry = _direct(entries,"minmax_fit_transform")
    var nk = minmax_keys()
    if entry is None or py_get(entries,nk[].k_minmax_fit) is None:
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
    var stats = classical_typed(Python.tuple(5,d),"<f4",py_item(context, nk[].k_output_context))
    var output = classical_typed(Python.tuple(n,d),"<f4",py_item(context, nk[].k_output_context))
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
    py_setitem(fields,nk[].k_n_features_in_,d)
    py_setitem(fields,nk[].k_n_samples_seen_,n)
    py_setitem(fields,nk[].k_numeric_mode_,mode)
    py_setitem(fields,nk[].k_feature_range_,Python.tuple(configuration[1],configuration[2]))
    py_setitem(fields,nk[].k_clip_,clip)
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
    var nk = minmax_keys()
    var old_range = minmax_range_pair(py_item(fields, nk[].k_feature_range_),context)
    if old_range != Python.tuple(configuration[1],configuration[2]):
        raise Error("MinMaxScaler partial_fit: feature_range changed since the first batch")
    var b = native_loaded_module("builtins")
    var fitted_mode = py_item(fields, nk[].k_numeric_mode_)
    if py_type(py_item(fields, nk[].k_n_samples_seen_)) is not b.int:
        raise Error("TypeError: MinMaxScaler n_samples_seen_ must remain a native integer")
    # UNPACK (lane bm-trees, 2026-11-22): the fitted extrema rows, the batch
    # and the calls of the fitted mode; no batch estimator object is built.
    var d = _features(fields)
    var old_min = _statistic(py_item(fields, nk[].k_data_min_), d, "extrema", context)
    var old_max = _statistic(py_item(fields, nk[].k_data_max_), d, "extrema", context)
    var entries = _minmax_entries(model, fitted_mode, context)
    if py_get(entries,nk[].k_minmax_fit) is None:
        raise Error("ImportError: HostMinMaxScaler training requires _mojolearn_preprocessing_host: " + String(py_get(entries,nk[].k_scaler_training_error,"minmax_fit is unavailable")))
    var calls = scaler_door_calls(model, entries, context)
    var snapshot = scaler_operation_input(source, context, PythonObject(None))
    var captured = snapshot.downcast_value_ptr[ScalerInputSnapshot]()
    var values = captured[].values
    var n = captured[].metadata.source.rows
    var batch_d = captured[].metadata.source.cols
    # RUN: the batch fit core, the width refusal and the merge (core/scaler_core.mojo).
    var rows = scaler_minmax_partial_fit_run(calls, _ptr(old_min), _ptr(old_max), d, _ptr(values), n, batch_d,
                                             Float32(Float64(py=configuration[1])),
                                             Float32(Float64(py=configuration[2])))
    _ = values  # keep the Python objects alive through the native call above
    _ = old_min
    _ = old_max
    # PACK
    _publish_extrema(fields, rows, d, context)
    py_setitem(fields,nk[].k_n_samples_seen_,py_item(fields, nk[].k_n_samples_seen_) + PythonObject(n))
    py_setitem(fields,nk[].k_clip_,configuration[4])
    _ = values^
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
    var nk = minmax_keys()
    var copy = _flag(py_item(fields, nk[].k_copy),"copy",context) if configuration is None else Bool(py=configuration[3])
    if not _fitted(fields):
        _ = classical_not_fitted_class(context)
        raise Error("NotFittedError: MinMaxScaler is not fitted")
    var mode = py_item(fields, nk[].k_numeric_mode_) if configuration is None else configuration[0]
    var entries = _minmax_entries(model, mode, context)
    var calls = scaler_transform_calls(model, entries, context)
    var snapshot = scaler_operation_input(source, context, input_cache)
    var captured = snapshot.downcast_value_ptr[ScalerInputSnapshot]()
    var values = captured[].values
    var n = captured[].metadata.source.rows
    var d = captured[].metadata.source.cols
    if d != _features(fields):
        raise Error("MinMaxScaler input feature count differs from fit")
    var bounds = minmax_range_pair(py_item(fields, nk[].k_feature_range_),context) if configuration is None else Python.tuple(configuration[1],configuration[2])
    var clip = _flag(py_item(fields, nk[].k_clip_),"clip_",context) if configuration is None else Bool(py=configuration[4])
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
    var min_value = py_item(fields, nk[].k_min_)
    var scale_value = py_item(fields, nk[].k_scale_)
    var held = prepared_get(mods, _ROWS_SLOT, model)
    if (held is not None and held[0] is min_value and held[1] is scale_value and Int(py=held[3]) == d
            and String(held[4]) == String(mode)):
        unsafe_memcpy(dest=ap, src=_ptr(held[2]), count=7*d+5)
        _ = held  # keep the Python object alive through the copy above
    else:
        arena = minmax_rows(calls, min_value, scale_value, d, context)
        ap = F32P(unsafe_from_address=Int(arena.unsafe_ptr()))
        if scaler_transform_rows_valid(ap, d):
            var rows = classical_typed(Python.tuple(7*d+5), "<f4", py_item(context, nk[].k_output_context))
            unsafe_memcpy(dest=_ptr(rows), src=ap, count=7*d+5)
            _ = rows  # keep the Python object alive through the copy above
            py_setattr(min_value, "_readonly", PythonObject(True))
            py_setattr(scale_value, "_readonly", PythonObject(True))
            prepared_put(mods, _ROWS_SLOT, model, Python.tuple(min_value, scale_value, rows, d, String(mode)))
    var output = classical_typed(Python.tuple(n,d), "<f4", py_item(context, nk[].k_output_context))
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
        var nk = minmax_keys()
        var operation = py_op_code(nk[].ops, all[1])
        var args = all[2]
        var root = all[3]
        var context = root
        var b = native_loaded_module("builtins")
        var is_class = py_type(model) is b.type
        var cls = model if is_class else py_type(model)
        if py_type(cls) is not b.type:
            raise Error("TypeError: native MinMaxScaler refuses Python metaclass hooks")
        if not is_class and (operation == MINMAX_OPN_FIT or operation == MINMAX_OPN_PARTIAL_FIT
                or operation == MINMAX_OPN_FIT_TRANSFORM or operation == MINMAX_OPN_TRANSFORM):
            # A data operation runs in the estimator's route
            # (bindings/scaler_route_unpack.mojo): the process's
            # context, the CPU one of `device="cpu"`, or an explicit Host*
            # class's own. One record read and one pointer compare; the
            # route's first resolution reads `device` and the host table.
            context = scaler_route_context(model, root)
        else:
            # Parameter, archive and class-level operations compute nothing
            # on a device: the explicit host table by class, as before.
            var host = py_get(py_item(context, nk[].k_scaler_host_contexts),cls)
            if host is not None:
                context = host
        classical_connect_numpy(context)
        classical_check_layout(model,context)
        # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
        var result = Optional[PythonObject]()
        if operation == MINMAX_OPN_INIT:
            # `device` is the fifth constructor argument (absent from a
            # four-argument caller: this installation's device). Refused in
            # words before any field changes.
            var device = PythonObject(None)
            if len(args) > 4:
                device = args[4]
            scaler_device_admit(device, cls, root)
            result = py_present(minmax_initialize(model,args))
            py_setitem(_fields(model), nk[].k_device, device)
            scaler_route_clear(model)
        elif operation == MINMAX_OPN_GET_PARAMS:
            var fields = _fields(model)
            var params = Python.dict(feature_range=py_item(fields, nk[].k_feature_range),copy=py_item(fields, nk[].k_copy),clip=py_item(fields, nk[].k_clip),numeric_mode=py_item(fields, nk[].k_numeric_mode))
            # None on an instance built before the parameter existed.
            py_setitem(params, nk[].k_device, py_get(fields, nk[].k_device))
            result = params
        elif operation == MINMAX_OPN_VENDOR_USED:
            result = py_present(scaler_route_vendor_used(model, root))
        elif operation == MINMAX_OPN_SET_PARAMS:
            var fields = _fields(model)
            var options = args[0]
            if py_type(options) is not b.dict:
                raise Error("TypeError: MinMaxScaler parameters require an ordinary keyword dictionary")
            var names = py_call(b.list,options)
            var known = Python.tuple(nk[].k_feature_range,nk[].k_copy,nk[].k_clip,nk[].k_numeric_mode,nk[].k_device)
            for k in range(len(names)):  # driver-loop: proposed estimator parameter names
                if py_type(names[k]) is not b.str or not py_contains(known,names[k]):
                    raise Error("Invalid MinMaxScaler parameters")
            if len(names):
                var device = py_get(options,nk[].k_device,py_get(fields, nk[].k_device))
                # Refused in words before the estimator changes.
                scaler_device_admit(device, cls, root)
                var values = Python.tuple(py_get(options,nk[].k_feature_range,py_item(fields, nk[].k_feature_range)),py_get(options,nk[].k_copy,py_item(fields, nk[].k_copy)),py_get(options,nk[].k_clip,py_item(fields, nk[].k_clip)),py_get(options,nk[].k_numeric_mode,py_item(fields, nk[].k_numeric_mode)))
                _ = py_call(fields.clear)
                _ = minmax_initialize(model,values)
                py_setitem(fields, nk[].k_device, device)
                scaler_route_clear(model)
            result = py_present(model)
        elif operation == MINMAX_OPN_IS_FITTED:
            result = PythonObject(_fitted(_fields(model)))
        elif operation == MINMAX_OPN_FIT:
            result = py_present(minmax_fit(model,args[0],context))
        elif operation == MINMAX_OPN_PARTIAL_FIT:
            result = py_present(minmax_partial_fit(model,args[0],context))
        elif operation == MINMAX_OPN_FIT_TRANSFORM:
            if len(args[2]):
                raise Error("TypeError: MinMaxScaler.fit got an unexpected keyword argument")
            var input_cache = Python.list()
            var configuration = _configuration(_fields(model),context)
            result = py_present(minmax_fit_transform(model,args[0],context,input_cache,configuration))
            if not result:
                _ = minmax_fit(model,args[0],context,input_cache,configuration)
                result = py_present(minmax_transform(model,args[0],False,context,input_cache,configuration))
            result = py_present(_public_result(py_or_none(result),context,input_cache))
        elif operation == MINMAX_OPN_TRANSFORM:
            var input_cache = Python.list()
            result = py_present(_public_result(minmax_transform(model,args[0],Bool(py=args[1]),context,input_cache),context,input_cache))
        elif operation == MINMAX_OPN_SAVE:
            result = py_present(minmax_archive_save(model,args[0],context))
        elif operation == MINMAX_OPN_LOAD:
            # `load(path, device=...)` (gd-verify-view, 2026-11-25): as StandardScaler's load
            # (bindings/scaler_lifecycle_native_door.mojo).
            var device = PythonObject(None)
            if len(args) > 1:
                device = args[1]
            scaler_device_admit(device, cls, root)
            var loaded = minmax_archive_load(model,args[0],context)
            if device is not None:
                py_setitem(_fields(loaded), nk[].k_device, device)
            result = py_present(loaded)
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
