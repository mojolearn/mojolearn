# SPDX-License-Identifier: Apache-2.0
# StandardScaler's public lifecycle. The original preprocessing and x_prep
# kernels, operation order, modes, and copy semantics remain unchanged.
from std.memory import unsafe_memcpy, bitcast
from std.os import getenv
from bindings.native_loaded_module import native_loaded_module
from std.python import Python, PythonObject
from bindings.optimizer_public_words import PYTHON_ERROR_PENDING, optimizer_python_error
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem, py_contains, py_setattr
from bindings.native_loaded_module import NativeModules
from bindings.native_prepared_owner import prepared_get, prepared_put
from bindings.py_typed_call import py_call_typed
from bindings.native_error_class import error_without_class
from bindings.native_caller_frame import native_caller_globals
from bindings.classical_attributes import classical_connect_numpy, classical_getattribute, classical_check_layout
from bindings.classical_arrays import classical_typed, classical_input, classical_address, classical_is_array, classical_is_f32, classical_c_order, classical_i64, classical_dtype_str
from bindings.classical_archive import _archive_class_name_unpack
from bindings.standard_archive_native import standard_archive_write, standard_archive_load as scaler_archive_load
from core.native_archive_standard import NativeStandardArchiveSaved, native_standard_statistic_names
from bindings.classical_exceptions import classical_not_fitted_class
from bindings.causal_exception import causal_borrowed_item, causal_set_exception
from bindings.classical_pandas import native_pandas_exact, classical_admit_value
# TOMBSTONE: P1_NATIVE_SKLEARN_TAGS (removed: native code importing sklearn.utils._tags and building its Tag dataclasses; the tags are built by the Python compatibility layer, _mode.sklearn_tags) deleted 2026-11-09 by lane p1-python-callbacks-worst; code recoverable at e68654a757.
# Restore: experiments/removed/P1_NATIVE_SKLEARN_TAGS.patch.
from bindings.prep_entry_cache import prep_mode_checked
from bindings.sequential_inputs import sequential_input
from bindings.sequential_admission import _ordinary_torch
from bindings.native_torch_admission import native_torch_numpy
from providers.python.preprocessing_input_pyapi import ScalerInputSnapshot, scaler_operation_input, scaler_snapshot_write_back
from bindings.scaler_native import scaler_fit, _input, _ptr, _copy_row, _bool
from core.preprocessing_program import prep_program_stage as _stage
from core.scaler_fit_impl import scaler_fill, scaler_all_finite, scaler_refuse_infinite_columns, scaler_feature_indices, scaler_counts_f32, scaler_counts_i64, scaler_equal_f32, scaler_equal_i64, scaler_counts_f64, scaler_merge_count_only, scaler_narrow_counts

alias F32P = MutPointer[Float32, MutUntrackedOrigin]
alias I64P = MutPointer[Int64, MutUntrackedOrigin]


def _entries(mode: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    if py_type(mode) is not b.str or (String(mode) != "fast" and String(mode) != "identical"):
        raise Error("numeric_mode must be fast, identical or None")
    if Bool(py=py_get(context,"scaler_explicit_host",False)) and String(mode) != "identical":
        raise Error("HostStandardScaler runs identical only on the host")
    var entries = context["scaler_bindings"][mode]
    if py_type(entries) is b.str:
        raise Error("ImportError: " + String(entries))
    var expected = 1 if String(mode) == "identical" else 0
    # The two mode probes run once per binding (prep_entry_cache).
    if not prep_mode_checked(entries, expected):
        raise Error("RuntimeError: Scaler native numeric mode disagrees with requested mode")
    return entries


def _mode(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    _ = _bool(fields["copy"], context)
    _ = _bool(fields["with_mean"], context)
    _ = _bool(fields["with_std"], context)
    var mode = fields["numeric_mode"]
    if mode is None:
        mode = context["backend_fields"]["_DEFAULT_MODE"]
    if py_type(mode) is not native_loaded_module("builtins").str:
        raise Error("numeric_mode must be fast, identical or None")
    mode = py_call(py_call(mode.strip).lower)
    if String(mode) != "fast" and String(mode) != "identical":
        raise Error("numeric_mode must be fast, identical or None")
    return mode


def _direct(entries: PythonObject, name: String) raises -> PythonObject:
    if String(getenv("MOJOLEARN_SCALER_DIRECT", "1").strip()) == "0":
        return PythonObject(None)
    return py_get(entries, name)


def _run(mut arena: List[Float32], mut program: List[Int32], entries: PythonObject) raises:
    _ = py_call_typed(entries["x_prep_run"], Int(arena.unsafe_ptr()), len(arena), Int(program.unsafe_ptr()), len(program) // 16)


def _features(fields: PythonObject) raises -> Int:
    var value = fields["n_features_in_"]
    if py_type(value) is not native_loaded_module("builtins").int:
        raise Error("TypeError: StandardScaler n_features_in_ must remain a native integer")
    return Int(py=value)


def _fitted(fields: PythonObject) raises -> Bool:
    return py_contains(fields, "scale_") and py_contains(fields, "numeric_mode_")


def scaler_initialize(model: PythonObject, args: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    var names = Python.tuple("copy", "with_mean", "with_std", "numeric_mode")
    for k in range(4):  # small-loop(params: the four stored constructor arguments)
        py_setitem(fields, names[k], args[k])
    return PythonObject(None)


def _write_back(source: PythonObject, values: PythonObject, output: PythonObject, copy: Bool, context: PythonObject) raises -> PythonObject:
    return _write_back(source, values, output, copy, context, PythonObject(None))


def _write_back(source: PythonObject, values: PythonObject, output: PythonObject, copy: Bool, context: PythonObject,
                snapshot: PythonObject) raises -> PythonObject:
    if snapshot is not None:
        return scaler_snapshot_write_back(snapshot, output, copy)
    if copy or native_pandas_exact(source):
        return output
    var b = native_loaded_module("builtins")
    if py_type(source) is b.list or py_type(source) is b.tuple or py_type(source) is b.range:
        return output
    var view = PythonObject(None)
    if classical_is_array(source,context):
        if String(source.dtype) != "<f4" or not classical_c_order(source,context) or Bool(py=source._readonly):
            return output
        view = source._mv
    else:
        # The source was admitted before _input. memoryview's C constructor
        # handles only the supported ordinary native buffer interfaces here.
        try:
            var exporter = native_torch_numpy(source) if _ordinary_torch(py_type(source),context) else source
            view = py_call(b.memoryview,exporter)
        except:
            return output
        if Bool(py=view.readonly) or not Bool(py=view.c_contiguous):
            return output
        var format = String(view.format)
        if Int(py=view.itemsize) != 4 or (format != "f" and format != "<f" and format != "=f" and format != "@f"):
            return output
    var dest = py_call(view.cast,"B")
    var payload = py_call(output._mv.cast,"B")
    py_setitem(dest,py_call(b.slice,PythonObject(None)),payload)
    # `values` is borrowed from the caller and remains live through this call.
    return source


def _statistic(value: PythonObject, d: Int, name: String, context: PythonObject) raises -> PythonObject:
    # A fitted statistic the fit published is an exact Array (our own class,
    # slot fields only): its admission is a no-op, so the per-transform
    # admission (module checks per statistic) runs only for anything else.
    if not classical_is_array(value, context):
        classical_admit_value(value,context)
        raise Error("StandardScaler " + name + " must remain a finite contiguous Float32 feature vector")
    if String(value.dtype) != "<f4" or Int(py=value.ndim) != 1 or Int(py=value.size) != d or not classical_c_order(value, context):
        raise Error("StandardScaler " + name + " must remain a finite contiguous Float32 feature vector")
    return value


def _nan_fill(values: PythonObject, entries: PythonObject, context: PythonObject) raises -> PythonObject:
    var n = Int(py=values.shape[0])
    var d = Int(py=values.shape[1])
    var st = n*d
    var keep = st+6*d
    var stat = keep+d
    var dest = stat+d
    var arena = List[Float32](length=dest+n*d, fill=0)
    var ap = F32P(unsafe_from_address=Int(arena.unsafe_ptr()))
    unsafe_memcpy(dest=ap, src=_ptr(values), count=n*d)
    scaler_feature_indices(ap+keep, d)
    var program = List[Int32]()
    _stage(program, 1, d, 0, n, d, st)
    _stage(program, 24, n*d, 0, n, d, stat, dest, keep, d)
    _run(arena, program, entries)
    scaler_refuse_infinite_columns(ap, st, d)
    var output = classical_typed(Python.tuple(n,d), "<f4", context["output_context"])
    unsafe_memcpy(dest=_ptr(output), src=ap+dest, count=n*d)
    _ = output  # keep the Python object alive through the native call above
    _ = arena^
    return output


comptime _STANDARD_ROWS_SLOT = "standard_transform_rows_v1"


def scaler_transform(model: PythonObject, source: PythonObject, inverse: Bool, copy_option: PythonObject, context: PythonObject) raises -> PythonObject:
    return scaler_transform(model, source, inverse, copy_option, context, PythonObject(None))


def scaler_transform(model: PythonObject, source: PythonObject, inverse: Bool, copy_option: PythonObject, context: PythonObject,
                     input_cache: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    var copy = _bool(fields["copy"] if copy_option is None else copy_option, context)
    if not _fitted(fields):
        _ = classical_not_fitted_class(context)
        raise Error("NotFittedError: StandardScaler is not fitted")
    var entries = _entries(fields["numeric_mode_"], context)
    var direct = _direct(entries, "standard_transform_direct")
    var snapshot = scaler_operation_input(source, context, input_cache)
    var captured = snapshot.downcast_value_ptr[ScalerInputSnapshot]()
    var values = captured[].values
    var n = captured[].metadata.source.rows
    var d = captured[].metadata.source.cols
    if d != _features(fields):
        raise Error("StandardScaler input feature count differs from fit")
    var with_mean = _bool(fields["with_mean_"], context)
    var with_std = _bool(fields["with_std_"], context)
    var mean = fields["mean_"]
    var scale = fields["scale_"]
    if with_mean and mean is None:
        raise Error("StandardScaler mean_ is missing for a centered transform")
    if with_std and scale is None:
        raise Error("StandardScaler scale_ is missing for a scaled transform")
    var arena = List[Float32](length=7*d+5, fill=0)
    var ap = F32P(unsafe_from_address=Int(arena.unsafe_ptr()))
    # The derived and validated rows of the fitted statistics (the stage 264
    # + 248 program, a separate crossing) are kept per model in the
    # prepared-owner registry (weakly keyed, outside __dict__), keyed by the
    # identity of mean_ and scale_ (or None), d, with_mean, with_std and the
    # numeric mode (p3-prep). When the record is taken the fitted Arrays are
    # made read-only (hazard class A): an in-place write through the public
    # view raises NumPy's read-only error; replacing a statistic (refit,
    # partial_fit, load, assignment) misses and rebuilds.
    var mods = NativeModules()
    var mode_text = String(fields["numeric_mode_"])
    var held = prepared_get(mods, _STANDARD_ROWS_SLOT, model)
    if (held is not None and held[0] is mean and held[1] is scale and Int(py=held[3]) == d
            and String(held[4]) == mode_text and Bool(py=held[5]) == with_mean and Bool(py=held[6]) == with_std):
        unsafe_memcpy(dest=ap, src=_ptr(held[2]), count=7*d+5)
        _ = held  # keep the Python object alive through the copy above
    else:
        var mean_raw = mean
        var scale_raw = scale
        if mean is None:
            mean = classical_typed(Python.tuple(d), "<f4", context["output_context"])
        if scale is None:
            scale = classical_typed(Python.tuple(d), "<f4", context["output_context"])
            scaler_fill(_ptr(scale), d, 1)
            _ = scale  # keep the Python object alive through the native call above
        mean = _statistic(mean, d, "mean_", context)
        scale = _statistic(scale, d, "scale_", context)
        unsafe_memcpy(dest=ap, src=_ptr(mean), count=d)
        _ = mean  # keep the Python object alive through the native call above
        unsafe_memcpy(dest=ap+d, src=_ptr(scale), count=d)
        _ = scale  # keep the Python object alive through the native call above
        arena[7*d+4] = 1
        var program = List[Int32]()
        _stage(program, 264, d, 0, d, Int(with_mean), Int(with_std), 7*d+3, 7*d+4, 2*d, 3*d, 4*d, 5*d, 6*d)
        _stage(program, 248, 1, 4*d, d, 7*d)
        _stage(program, 248, 1, 5*d, d, 7*d+1)
        _stage(program, 248, 1, 6*d, d, 7*d+2)
        _run(arena, program, entries)
        if bitcast[DType.int32](arena[7*d+1]) == 0 and bitcast[DType.int32](arena[7*d+2]) == 0:
            var rows = classical_typed(Python.tuple(7*d+5), "<f4", context["output_context"])
            unsafe_memcpy(dest=_ptr(rows), src=ap, count=7*d+5)
            _ = rows  # keep the Python object alive through the copy above
            if mean_raw is not None:
                py_setattr(mean_raw, "_readonly", PythonObject(True))
            if scale_raw is not None:
                py_setattr(scale_raw, "_readonly", PythonObject(True))
            prepared_put(mods, _STANDARD_ROWS_SLOT, model, Python.tuple(
                mean_raw, scale_raw, rows, d, mode_text, PythonObject(with_mean), PythonObject(with_std)))
    if bitcast[DType.int32](arena[7*d+1]) != 0:
        raise Error("StandardScaler mean_ and scale_ must remain finite (NaN: a feature never seen)")
    if bitcast[DType.int32](arena[7*d+2]) != 0:
        raise Error("StandardScaler scale_ must remain positive")
    var colnan = bitcast[DType.int32](arena[7*d]) != 0
    var output = classical_typed(Python.tuple(n,d), "<f4", context["output_context"])
    var params = Python.list(n,d,Int(inverse),Int(with_mean),Int(with_std))
    var finite = True
    if direct is not None:
        var got = Int(py=py_call(direct, classical_address(values), Int(ap+2*d), Int(ap+3*d), classical_address(output), params))
        _ = output  # keep the Python object alive through the native call above
        _ = values  # keep the Python object alive through the native call above
        if got == 0:
            raise Error("StandardScaler transform overflowed in Float32")
        _ = output  # keep the Python object alive through the native calls above
        _ = values  # keep the Python object alive through the native calls above
        finite = got == 1
        _ = output  # keep the Python object alive through the native call above
        _ = values  # keep the Python object alive through the native call above
        if not finite:
            var filled = _nan_fill(values, entries, context)
            if Int(py=py_call(direct, classical_address(filled), Int(ap+2*d), Int(ap+3*d), classical_address(output), params)) != 1:
                raise Error("StandardScaler transform overflowed in Float32")
            _ = filled  # keep the Python object alive through the native calls above
            _ = output  # keep the Python object alive through the native calls above
            _ = filled^
    else:
        finite = scaler_all_finite(_ptr(values), n*d)
        _ = values  # keep the Python object alive through the native call above
        var filled = values if finite else _nan_fill(values, entries, context)
        _ = py_call_typed(entries["standard_transform"], classical_address(filled), Int(ap+2*d), Int(ap+3*d), classical_address(output), params)
        _ = filled  # keep the Python object alive through the native call above
        _ = output  # keep the Python object alive through the native call above
        if not scaler_all_finite(_ptr(output), n*d):
            raise Error("StandardScaler transform overflowed in Float32")
        _ = output  # keep the Python object alive through the native calls above
        _ = filled^
    if not finite or colnan:
        var keep = List[Float32](length=3*n*d+2*d, fill=0)
        var kp = F32P(unsafe_from_address=Int(keep.unsafe_ptr()))
        unsafe_memcpy(dest=kp, src=_ptr(values), count=n*d)
        _ = values  # keep the Python object alive through the native call above
        unsafe_memcpy(dest=kp+n*d, src=_ptr(output), count=n*d)
        _ = output  # keep the Python object alive through the native call above
        unsafe_memcpy(dest=kp+2*n*d, src=ap+4*d, count=d)
        var restore = List[Int32]()
        # c2_nan_sub publishes int32 words, while nan_keep reads numeric
        # Float32 flags. Preserve the old _Prog.put_codes i2f stage: raw
        # integer 1 is a subnormal float that Metal can flush to zero.
        _stage(restore, 10, d, 2*n*d, 2*n*d+d)
        _stage(restore, 103, n*d, 0, d, n*d, 2*n*d+d, 2*n*d+2*d)
        _run(keep, restore, entries)
        unsafe_memcpy(dest=_ptr(output), src=kp+2*n*d+2*d, count=n*d)
        _ = output  # keep the Python object alive through the native call above
        _ = keep^
    _ = arena^
    _ = mean^
    _ = scale^
    return _write_back(source, values, output, copy, context, snapshot)


def _counts(value: PythonObject, d: Int, dest: F32P, context: PythonObject) raises:
    classical_admit_value(value,context)
    var b = native_loaded_module("builtins")
    if py_type(value) is b.int or py_type(value) is b.float:
        scaler_counts_f32(0, 0, Float64(py=value), dest, d)
        return
    if not classical_is_array(value, context) or Int(py=value.ndim) != 1 or Int(py=value.size) != d:
        raise Error("StandardScaler n_samples_seen_ must remain a scalar or per-feature count vector")
    var dtype = String(value.dtype)
    if dtype != "<i8" and dtype != "<f4":
        raise Error("StandardScaler n_samples_seen_ must remain int64 counts or float32 weights")
    scaler_counts_f32(Int(py=classical_address(value)), 1 if dtype == "<i8" else 2, 0, dest, d)


def _integer_counts(value: PythonObject, d: Int, dest: I64P, context: PythonObject) raises:
    var scalar = py_type(value) is native_loaded_module("builtins").int
    if not scalar and (not classical_is_array(value, context) or String(value.dtype) != "<i8" or Int(py=value.size) != d):
        raise Error("StandardScaler n_samples_seen_ must remain int64 counts")
    scaler_counts_i64(0 if scalar else Int(py=classical_address(value)), not scalar, Int64(py=value) if scalar else Int64(0), dest, d)


def _publish_counts(source: F32P, d: Int, context: PythonObject) raises -> PythonObject:
    if scaler_equal_f32(source, d):
        return PythonObject(Float64(source[0]))
    return _copy_row(source, d, context)


def scaler_partial_fit(model: PythonObject, source: PythonObject, weights: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    if not _fitted(fields):
        return scaler_fit(model, source, weights, context)
    _ = _mode(fields, context)
    if _bool(fields["with_mean"], context) != _bool(fields["with_mean_"], context) or _bool(fields["with_std"], context) != _bool(fields["with_std_"], context):
        raise Error("StandardScaler partial_fit: with_mean / with_std changed since the first batch")
    var b = native_loaded_module("builtins")
    var batch = py_call(b.object.__new__, context["scaler_class"])
    _ = scaler_initialize(batch, Python.tuple(True, fields["with_mean_"], fields["with_std_"], fields["numeric_mode_"]))
    _ = scaler_fit(batch, source, weights, context)
    var fresh = py_fields(batch)
    var d = _features(fields)
    if _features(fresh) != d:
        raise Error("StandardScaler input feature count differs from fit")
    var old_seen = fields["n_samples_seen_"]
    var new_seen = fresh["n_samples_seen_"]
    var weighted = weights is not None or py_type(old_seen) is b.float
    if classical_is_array(old_seen, context):
        weighted = weighted or String(old_seen.dtype) == "<f4"
    var entries = _entries(fields["numeric_mode_"], context)
    var arena = List[Float32](length=10*d, fill=0)
    var ap = F32P(unsafe_from_address=Int(arena.unsafe_ptr()))
    _counts(old_seen, d, ap, context)
    _counts(new_seen, d, ap+d, context)
    var count = PythonObject(None)
    if fields["mean_"] is None:
        # Preserve _per_feature's Float64 addition and _seen's publication
        # exactly when neither mean nor variance is retained.
        var wide = List[Float64](length=2*d,fill=0)
        var wp = MutPointer[Float64,MutUntrackedOrigin](unsafe_from_address=Int(wide.unsafe_ptr()))
        var past = Python.tuple(old_seen,new_seen)
        for k in range(2):  # small-loop(counts: previous and current count storage)
            var value = past[k]
            var scalar = py_type(value) is b.int or py_type(value) is b.float
            scaler_counts_f64(0 if scalar else Int(py=classical_address(value)), 0 if scalar else (1 if String(value.dtype) == "<i8" else 2), Float64(py=value) if scalar else Float64(0), wp+k*d,d)
            _ = value  # keep the Python object alive through the native call above
        if scaler_merge_count_only(wp,wp+d,d):
            count = PythonObject(wp[0]) if weighted else PythonObject(Int(wp[0]))
        else:
            count = classical_typed(Python.tuple(d),"<f4" if weighted else "<i8",context["output_context"])
            scaler_narrow_counts(wp,Int(py=classical_address(count)),weighted,d)
            _ = count  # keep the Python object alive through the native call above
        py_setitem(fields,"n_samples_seen_",count)
        _ = wide^
        _ = arena^
        return model
    else:
        var old_mean = _statistic(fields["mean_"], d, "mean_", context)
        var new_mean = _statistic(fresh["mean_"], d, "mean_", context)
        unsafe_memcpy(dest=ap+2*d, src=_ptr(old_mean), count=d)
        _ = old_mean  # keep the Python object alive through the native call above
        unsafe_memcpy(dest=ap+3*d, src=_ptr(new_mean), count=d)
        _ = new_mean  # keep the Python object alive through the native call above
        if fields["var_"] is not None:
            var old_var = _statistic(fields["var_"], d, "var_", context)
            unsafe_memcpy(dest=ap+4*d, src=_ptr(old_var), count=d)
            _ = old_var  # keep the Python object alive through the native call above
            _ = old_var^
        if fresh["var_"] is not None:
            var new_var = _statistic(fresh["var_"], d, "var_", context)
            unsafe_memcpy(dest=ap+5*d, src=_ptr(new_var), count=d)
            _ = new_var  # keep the Python object alive through the native call above
            _ = new_var^
        var program = List[Int32]()
        _stage(program, 91, d, 0, 2*d, 4*d, d, 3*d, 5*d, d, 1, 6*d, 7*d, 8*d, 1)
        _stage(program, 102, d, 8*d, 9*d)
        _run(arena, program, entries)
        py_setitem(fields, "mean_", _copy_row(ap+7*d, d, context))
        if _bool(fields["with_std_"], context):
            py_setitem(fields, "var_", _copy_row(ap+8*d, d, context))
            py_setitem(fields, "scale_", _copy_row(ap+9*d, d, context))
        _ = old_mean^
        _ = new_mean^
    if weighted:
        count = _publish_counts(ap+6*d, d, context)
    else:
        # Retain the exact original int64 device-count addition, avoiding the
        # loss of integer precision from the float32 statistics arena.
        var counts = List[Float32](length=6*d, fill=0)
        var cp = F32P(unsafe_from_address=Int(counts.unsafe_ptr()))
        var ip = I64P(unsafe_from_address=Int(cp))
        _integer_counts(old_seen, d, ip, context)
        _integer_counts(new_seen, d, ip+d, context)
        var program = List[Int32]()
        _stage(program, 269, d, 0, 2*d, 4*d)
        _run(counts, program, entries)
        if scaler_equal_i64(ip+2*d, d):
            count = PythonObject(Int(ip[2*d]))
        else:
            count = classical_typed(Python.tuple(d), "<i8", context["output_context"])
            var dest = I64P(unsafe_from_address=Int(py=classical_address(count)))
            _ = count  # keep the Python object alive through the native call above
            unsafe_memcpy(dest=dest, src=ip+2*d, count=d)
        _ = counts^
    py_setitem(fields, "n_samples_seen_", count)
    _ = arena^
    _ = old_seen^
    _ = new_seen^
    return model


def scaler_fit_transform(model: PythonObject, source: PythonObject, options: PythonObject, context: PythonObject) raises -> PythonObject:
    return scaler_fit_transform(model, source, options, context, PythonObject(None))


def scaler_fit_transform(model: PythonObject, source: PythonObject, options: PythonObject, context: PythonObject,
                         var input_cache: PythonObject) raises -> PythonObject:
    if input_cache is None:
        input_cache = Python.list()
    var b = native_loaded_module("builtins")
    var names = py_call(b.list, options)
    for k in range(len(names)):  # driver-loop: fit's keyword names, not data
        if py_type(names[k]) is not b.str or String(names[k]) != "sample_weight":
            raise Error("TypeError: StandardScaler.fit got an unexpected keyword argument")
    if len(options) == 0:
        var fields = py_fields(model)
        var mode = _mode(fields, context)
        # One upload of X for the whole operation: the combined fit +
        # transform export (every tier; `standard_fit_direct` then
        # `standard_transform_direct` on the resident X, the same kernels on
        # the same words), else the FAST Apple experiment's fused export.
        # Without either (the host column, MOJOLEARN_SCALER_DIRECT=0) the
        # two-step route below runs.
        var entries = _entries(mode, context)
        var entry = _direct(entries, "standard_fit_transform")
        if entry is None and String(mode) == "fast":
            entry = _direct(entries, "standard_fit_transform_direct")
        if entry is not None:
            # scaler_fit's refit invalidation comes before X is read.
            var stale = py_call(b.list, fields)
            for k in range(len(stale)):  # driver-loop: invalidate fitted attribute names on refit
                if String(stale[k]).endswith("_"):
                    _ = py_call(fields.pop, stale[k])
            var snapshot = scaler_operation_input(source, context, input_cache)
            var captured = snapshot.downcast_value_ptr[ScalerInputSnapshot]()
            var values = captured[].values
            var n = captured[].metadata.source.rows
            var d = captured[].metadata.source.cols
            var stats = classical_typed(Python.tuple(3,d), "<f4", context["output_context"])
            var output = classical_typed(Python.tuple(n,d), "<f4", context["output_context"])
            var wm = _bool(fields["with_mean"], context)
            var ws = _bool(fields["with_std"], context)
            var got = Int(py=py_call(entry, classical_address(values), classical_address(stats), classical_address(output), Python.list(n,d,Int(wm),Int(ws))))
            _ = output  # keep the Python object alive through the native call above
            _ = stats  # keep the Python object alive through the native call above
            _ = values  # keep the Python object alive through the native call above
            if got != 0:
                var ptr = _ptr(stats)
                _ = stats  # keep the Python object alive through the native call above
                py_setitem(fields, "mean_", _copy_row(ptr,d,context) if wm or ws else PythonObject(None))
                _ = stats  # keep the Python object alive through the native call above
                py_setitem(fields, "var_", _copy_row(ptr+d,d,context) if ws else PythonObject(None))
                _ = stats  # keep the Python object alive through the native call above
                py_setitem(fields, "scale_", _copy_row(ptr+2*d,d,context) if ws else PythonObject(None))
                _ = stats  # keep the Python object alive through the native call above
                py_setitem(fields, "n_features_in_", d)
                py_setitem(fields, "n_samples_seen_", n)
                py_setitem(fields, "numeric_mode_", mode)
                py_setitem(fields, "with_mean_", wm)
                py_setitem(fields, "with_std_", ws)
                _ = stats^
                if got < 0:
                    raise Error("StandardScaler transform overflowed in Float32")
                _ = output  # keep the Python object alive through the native calls above
                _ = values  # keep the Python object alive through the native calls above
                return _write_back(source, values, output, _bool(fields["copy"],context), context, snapshot)
            _ = output  # keep the Python object alive through the native calls above
            _ = stats  # keep the Python object alive through the native calls above
            _ = values  # keep the Python object alive through the native calls above
    _ = scaler_fit(model, source, py_get(options,"sample_weight"), context, input_cache)
    return scaler_transform(model, source, False, PythonObject(None), context, input_cache)


def scaler_archive_save(model: PythonObject, path: PythonObject, context: PythonObject) raises -> PythonObject:
    # UNPACK: public slots and optional owner pins only. Native archive code
    # owns metadata bytes, scalar count narrowing, member order and ZIP/NPY.
    var fields = py_fields(model)
    if not _fitted(fields):
        raise Error("RuntimeError: this estimator is not fitted yet")
    var seen = fields["n_samples_seen_"]
    var integer = py_type(seen) is py_type(PythonObject(0))
    var features = _features(fields)
    var integer_count = Int64(py=seen) if integer else Int64(-1)
    var with_mean = _bool(fields["with_mean_"], context)
    var with_std = _bool(fields["with_std_"], context)
    var saved = NativeStandardArchiveSaved(features, integer_count, with_mean, with_std)
    var estimator = _archive_class_name_unpack(py_type(model))
    var mode = fields["numeric_mode_"]
    var count_kind = 0
    var scalar_count = Float64(0)
    if not integer:
        count_kind = 2
        if not classical_is_array(seen, context):
            if py_type(seen) is not py_type(PythonObject(0.0)):
                raise Error("TypeError: StandardScaler counts must be native numbers or count buffers")
            count_kind = 1
            scalar_count = Float64(py=seen)
    var statistics = List[PythonObject]()
    var names = native_standard_statistic_names()
    for name in names:  # unpack: three optional public statistic owner pins
        var value = fields[PythonObject(name + "_")]
        if value is not None:
            _ = _statistic(value, features, name, context)
        statistics.append(value)
    return standard_archive_write(path, context, estimator, saved, mode, seen,
                                  count_kind, scalar_count, statistics)


def _outside(context: PythonObject, depth: Int) raises -> Bool:
    var outside = True
    try:
        var globals = native_caller_globals(depth)
        if globals is None:
            return True
        var name = py_get(globals,"__name__")
        if py_type(name) is py_type(PythonObject("")):
            var module = String(name)
            outside = module != "mojolearn" and not module.startswith("mojolearn.") and not module.startswith("mojolearn_verify_all_")
        if py_get(globals,"__mojolearn_user_boundary__") is PythonObject(True):
            outside = True
        elif py_get(globals,"__mojolearn_internal__") is PythonObject(True):
            outside = False
    except:
        outside = True
    return outside


def _public_result(result: PythonObject, context: PythonObject) raises -> PythonObject:
    return _public_result(result, context, PythonObject(None))


def _public_result(result: PythonObject, context: PythonObject,
                   input_cache: PythonObject) raises -> PythonObject:
    if classical_is_array(result,context) and py_get(context,"ndarray") is not None and _outside(context,1):
        if input_cache is not None and len(input_cache) > 0:
            var captured = input_cache[0].downcast_value_ptr[ScalerInputSnapshot]()
            if result is captured[].original:
                # Publish the captured shape and owner after copy-back; the
                # caller's mutable Array metadata is not read again.
                return sequential_input(captured[].values,context)
        return sequential_input(result,context)
    return result


def scaler_attribute_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=args_ptr)
        var context = args[0]
        var model = args[1]
        var name = args[2]
        classical_connect_numpy(context)
        var value = classical_getattribute(model,name,context)
        # The caller frame is walked only when the stored value is an Array
        # that the public conversion changed (p3-prep): method lookups and
        # plain fields return without the frame walk.
        var fields = py_fields(model)
        var original = py_get(fields,name)
        if original is not value and classical_is_array(original,context) and not _outside(context,0):
            value = original
        return value.steal_data()
    except error:
        if String(error).startswith("AttributeError:"):
            return raise_python_exception(error_without_class(error,"AttributeError"),ExceptionType("PyExc_AttributeError"))
        return raise_python_exception(error_without_class(error,"TypeError"),ExceptionType.TypeError)


def scaler_lifecycle_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var all = PythonObject(from_borrowed=args_ptr)
        var model = all[0]
        var op = String(all[1])
        var args = all[2]
        var context = all[3]
        var cls = model if py_type(model) is native_loaded_module("builtins").type else py_type(model)
        var host = py_get(context["scaler_host_contexts"],cls)
        if host is not None:
            context = host
        classical_connect_numpy(context)
        classical_check_layout(model,context)
        var result = PythonObject(None)
        if op == "init":
            result = scaler_initialize(model,args)
        elif op == "get_params":
            var fields = py_fields(model)
            result = Python.dict(copy=fields["copy"],with_mean=fields["with_mean"],with_std=fields["with_std"],numeric_mode=fields["numeric_mode"])
        elif op == "set_params":
            var fields = py_fields(model)
            var options = args[0]
            var b = native_loaded_module("builtins")
            var names = py_call(b.list,options)
            var known = Python.tuple("copy","with_mean","with_std","numeric_mode")
            for k in range(len(names)):  # driver-loop: proposed estimator parameter names
                if py_type(names[k]) is not b.str or not py_contains(known,names[k]):
                    raise Error("Invalid StandardScaler parameters")
            if len(names):
                var values = Python.tuple(py_get(options,"copy",fields["copy"]),py_get(options,"with_mean",fields["with_mean"]),py_get(options,"with_std",fields["with_std"]),py_get(options,"numeric_mode",fields["numeric_mode"]))
                _ = py_call(fields.clear)
                _ = scaler_initialize(model,values)
            result = model
        elif op == "is_fitted":
            result = PythonObject(_fitted(py_fields(model)))
        elif op == "transform":
            var input_cache = Python.list()
            result = _public_result(scaler_transform(model,args[0],Bool(py=args[1]),args[2],context,input_cache),context,input_cache)
        elif op == "partial_fit":
            result = scaler_partial_fit(model,args[0],args[2],context)
        elif op == "fit_transform":
            var input_cache = Python.list()
            result = _public_result(scaler_fit_transform(model,args[0],args[2],context,input_cache),context,input_cache)
        elif op == "save":
            result = scaler_archive_save(model,args[0],context)
        elif op == "load":
            result = scaler_archive_load(model,args[0],context)
        else:
            raise Error("unknown native StandardScaler operation")
        return result.steal_data()
    except error:
        if String(error) == PYTHON_ERROR_PENDING:
            return optimizer_python_error(error^)
        if String(error).startswith("NotFittedError:"):
            try:
                ref cpy = Python().cpython()
                var root = cpy.PyTuple_GetItem(args_ptr,3)
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
