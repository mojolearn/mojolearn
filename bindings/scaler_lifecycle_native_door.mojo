# SPDX-License-Identifier: Apache-2.0
# StandardScaler's public lifecycle. The original preprocessing and x_prep
# kernels, operation order, modes, and copy semantics remain unchanged.
from bindings.mode_answer_door import answer_mode_object
from core.mode_answer import MODE_FAMILY_SCALER
from std.memory import unsafe_memcpy
from std.os import getenv
from bindings.py_typed_call import py_call_typed
from bindings.native_loaded_module_door import native_loaded_module
from std.python import Python, PythonObject
from bindings.optimizer_public_words_unpack import PYTHON_ERROR_PENDING, optimizer_python_error
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem, py_contains, py_setattr
from bindings.native_loaded_module_door import NativeModules
from bindings.native_prepared_owner_unpack import prepared_get, prepared_put
from bindings.native_error_class import error_without_class
from bindings.native_caller_frame_pack import native_caller_globals
from bindings.classical_attributes_door import classical_connect_numpy, classical_getattribute, classical_check_layout
from bindings.classical_arrays_door import classical_wrap_layout
from bindings.native_array_storage_door import native_array_unfilled
from core.native_array_layout import NativeArrayLayout, native_matrix_shape
from bindings.classical_arrays_door import classical_typed, classical_input, classical_address, classical_is_array, classical_is_f32, classical_c_order, classical_i64, classical_dtype_str
from bindings.classical_archive_door import _archive_class_name_unpack
from bindings.standard_archive_native_unpack import standard_archive_write, standard_archive_load as scaler_archive_load
from core.native_archive_standard import NativeStandardArchiveSaved, native_standard_statistic_names
from bindings.classical_exceptions_door import classical_not_fitted_class
from bindings.causal_exception_unpack import causal_borrowed_item, causal_set_exception
from bindings.classical_pandas_door import native_pandas_exact, classical_admit_value
# TOMBSTONE: P1_NATIVE_SKLEARN_TAGS (removed: native code importing sklearn.utils._tags and building its Tag dataclasses; the tags are built by the Python compatibility layer, _mode.sklearn_tags) deleted 2026-11-09 by lane p1-python-callbacks-worst; code recoverable at e68654a757.
# Restore: experiments/removed/P1_NATIVE_SKLEARN_TAGS.patch.
from bindings.prep_entry_cache_unpack import prep_mode_checked
from bindings.sequential_inputs_door import sequential_input
from bindings.sequential_admission_door import _ordinary_torch
from bindings.native_torch_admission_door import native_torch_numpy
from providers.python.preprocessing_input_pyapi_door import ScalerInputSnapshot, scaler_operation_input, scaler_snapshot_write_back
from bindings.scaler_native_door import (
    scaler_fit, _input, _ptr, _copy_row, _bool, scaler_transform_calls, ScalerDoorCalls, scaler_door_calls,
    scaler_fit_entries, scaler_fit_input,
)
from core.scaler_core import scaler_standard_fit_core, SCALER_SEEN_INT, SCALER_SEEN_FLOAT, SCALER_SEEN_FLOAT_ROW
from bindings.scaler_lifecycle_run import (
    ScalerSeenView, ScalerStandardMerged, scaler_seen_of_fit, scaler_standard_merge_run, SCALER_SEEN_SOURCE_SCALAR,
    SCALER_SEEN_SOURCE_I64, SCALER_SEEN_SOURCE_F32,
)
from core.scaler_core import ScalerTransformSpec, scaler_transform_rows, scaler_transform_rows_valid, scaler_transform_core
from bindings.scaler_keys_unpack import (
    scaler_keys, SCALER_OPN_INIT, SCALER_OPN_GET_PARAMS, SCALER_OPN_SET_PARAMS, SCALER_OPN_IS_FITTED,
    SCALER_OPN_TRANSFORM, SCALER_OPN_PARTIAL_FIT, SCALER_OPN_FIT_TRANSFORM, SCALER_OPN_SAVE, SCALER_OPN_LOAD,
    SCALER_OPN_VENDOR_USED,
)
from bindings.scaler_route_unpack import (
    scaler_route_context, scaler_route_clear, scaler_device_admit, scaler_route_vendor_used, scaler_selection_held,
    scaler_selection_keep, scaler_route_refuse_fast,
)
from bindings.py_keys import py_op_code
from bindings.py_balanced import py_item

comptime F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime I64P = MutPointer[Int64, MutUntrackedOrigin]


def _entries(mode: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    if py_type(mode) is not b.str or (String(mode) != "fast" and String(mode) != "identical"):
        raise Error("numeric_mode must be fast, identical or None")
    var nk = scaler_keys()
    if Bool(py=py_get(context,nk[].k_scaler_explicit_host,False)) and String(mode) != "identical":
        raise Error("HostStandardScaler runs identical only on the host")
    var entries = py_item(context, nk[].k_scaler_bindings)[mode]
    if py_type(entries) is b.str:
        raise Error("ImportError: " + String(entries))
    var expected = 1 if String(mode) == "identical" else 0
    # The two mode probes run once per binding (prep_entry_cache).
    if not prep_mode_checked(entries, expected):
        raise Error("RuntimeError: Scaler native numeric mode disagrees with requested mode")
    return entries


def scaler_entries(model: PythonObject, mode: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_entries` of an estimator's operation: the selection its device route
    holds for `mode` (one record read and pointer compares; no dictionary
    read, mode probe or String build), or the selection made here by
    `_entries`' own checks, in its order and words, and kept in the route.
    `device="cpu"` with the fast mode is refused in the shared words first."""
    var held = scaler_selection_held(model, mode, context)
    if held is not None:
        return held
    scaler_route_refuse_fast(model, mode, context)
    var entries = _entries(mode, context)
    scaler_selection_keep(model, mode, entries, context)
    return entries


def _mode(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = scaler_keys()
    _ = _bool(py_item(fields, nk[].k_copy), context)
    _ = _bool(py_item(fields, nk[].k_with_mean), context)
    _ = _bool(py_item(fields, nk[].k_with_std), context)
    var mode = py_item(fields, nk[].k_numeric_mode)
    if mode is None:
        mode = py_item(py_item(context, nk[].k_backend_fields), nk[].k__DEFAULT_MODE)
    return _answered_mode(mode)


def _answered_mode(requested: PythonObject) raises -> PythonObject:
    """Validate a requested mode word (a str, stripped and lower-cased, fast
    or identical) and answer it under MODE_FAMILY_SCALER. Shared by
    StandardScaler (`_mode`) and MaxAbsScaler/RobustScaler
    (`scaler_requested_mode`); once per fit, in no loop."""
    if py_type(requested) is not native_loaded_module("builtins").str:
        raise Error("numeric_mode must be fast, identical or None")
    var mode = py_call(py_call(requested.strip).lower)
    var mode_word = String(mode)
    if mode_word != "fast" and mode_word != "identical":
        raise Error("numeric_mode must be fast, identical or None")
    answer_mode_object(mode, mode_word, MODE_FAMILY_SCALER)
    return mode


def scaler_requested_mode(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    """The mode a scaler with an optional `numeric_mode` field fits in
    (MaxAbsScaler, RobustScaler): the process default when the field is None
    or absent (an instance pickled before the parameter existed), else the
    validated word, answered as StandardScaler's is. Once per fit, in no loop;
    the None path makes the one default read the fit made before."""
    var nk = scaler_keys()
    var mode = py_get(fields, nk[].k_numeric_mode)
    if mode is None:
        return py_item(py_item(context, nk[].k_backend_fields), nk[].k__DEFAULT_MODE)
    return _answered_mode(mode)


def _direct(entries: PythonObject, name: String) raises -> PythonObject:
    if String(getenv("MOJOLEARN_SCALER_DIRECT", "1").strip()) == "0":
        return PythonObject(None)
    return py_get(entries, name)


def _run(mut arena: List[Float32], mut program: List[Int32], entries: PythonObject) raises:
    """The compiled x_prep program entry on an arena (bindings/minmax_native_door.mojo still drives MinMax through it;
    StandardScaler's fit and partial_fit run the typed ScalerDriverCalls in bindings/scaler_lifecycle_run.mojo).
    Follow-up: move MinMax onto the typed run and delete this."""
    var nk = scaler_keys()
    _ = py_call_typed(py_item(entries, nk[].k_x_prep_run), Int(arena.unsafe_ptr()), len(arena), Int(program.unsafe_ptr()), len(program) // 16)


def _features(fields: PythonObject) raises -> Int:
    var nk = scaler_keys()
    var value = py_item(fields, nk[].k_n_features_in_)
    if py_type(value) is not native_loaded_module("builtins").int:
        raise Error("TypeError: StandardScaler n_features_in_ must remain a native integer")
    return Int(py=value)


def _fitted(fields: PythonObject) raises -> Bool:
    var nk = scaler_keys()
    return py_contains(fields, nk[].k_scale_) and py_contains(fields, nk[].k_numeric_mode_)


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


# TOMBSTONE: `_nan_fill` moved to core/scaler_core.mojo `scaler_nan_fill` (lane scaler-typed-core,
# 2026-11-12); one copy for both scalers and the shard worker.


comptime _STANDARD_ROWS_SLOT = "standard_transform_rows_v1"


def scaler_standard_rows(calls: ScalerDoorCalls, mean: PythonObject, scale: PythonObject, d: Int, with_mean: Bool,
                         with_std: Bool, context: PythonObject) raises -> List[Float32]:
    """UNPACK the fitted mean_ / scale_ (None, or a contiguous Float32 row
    of d) and RUN the core's derived-rows program; shared by the public
    transform and the shard worker (lane scaler-typed-core)."""
    if with_mean and mean is None:
        raise Error("StandardScaler mean_ is missing for a centered transform")
    if with_std and scale is None:
        raise Error("StandardScaler scale_ is missing for a scaled transform")
    var center = 0
    var spread = 0
    if mean is not None:
        center = Int(py=classical_address(_statistic(mean, d, "mean_", context)))
    if scale is not None:
        spread = Int(py=classical_address(_statistic(scale, d, "scale_", context)))
    var rows = scaler_transform_rows(calls, center, spread, d, with_mean, with_std)
    _ = mean  # keep the Python objects alive through the native call above
    _ = scale
    return rows^


def scaler_transform(model: PythonObject, source: PythonObject, inverse: Bool, copy_option: PythonObject, context: PythonObject) raises -> PythonObject:
    return scaler_transform(model, source, inverse, copy_option, context, PythonObject(None))


def scaler_transform(model: PythonObject, source: PythonObject, inverse: Bool, copy_option: PythonObject, context: PythonObject,
                     input_cache: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    var nk = scaler_keys()
    var copy = _bool(py_item(fields, nk[].k_copy) if copy_option is None else copy_option, context)
    if not _fitted(fields):
        _ = classical_not_fitted_class(context)
        raise Error("NotFittedError: StandardScaler is not fitted")
    var entries = scaler_entries(model, py_item(fields, nk[].k_numeric_mode_), context)
    var calls = scaler_transform_calls(model, entries, context)
    var snapshot = scaler_operation_input(source, context, input_cache)
    var captured = snapshot.downcast_value_ptr[ScalerInputSnapshot]()
    var values = captured[].values
    var n = captured[].metadata.source.rows
    var d = captured[].metadata.source.cols
    if d != _features(fields):
        raise Error("StandardScaler input feature count differs from fit")
    var with_mean = _bool(py_item(fields, nk[].k_with_mean_), context)
    var with_std = _bool(py_item(fields, nk[].k_with_std_), context)
    var mean = py_item(fields, nk[].k_mean_)
    var scale = py_item(fields, nk[].k_scale_)
    var arena = List[Float32](length=7*d+5, fill=0)
    var ap = F32P(unsafe_from_address=Int(arena.unsafe_ptr()))
    # The derived and validated rows of the fitted statistics (the stage 264
    # + 248 program, core/scaler_core.mojo `scaler_transform_rows`) are kept
    # per model in the prepared-owner registry (weakly keyed, outside
    # __dict__), keyed by the identity of mean_ and scale_ (or None), d,
    # with_mean, with_std and the numeric mode (p3-prep). When the record is
    # taken the fitted Arrays are made read-only (hazard class A): an
    # in-place write through the public view raises NumPy's read-only error;
    # replacing a statistic (refit, partial_fit, load, assignment) misses and
    # rebuilds.
    var mods = NativeModules()
    var mode_text = String(py_item(fields, nk[].k_numeric_mode_))
    var held = prepared_get(mods, _STANDARD_ROWS_SLOT, model)
    if (held is not None and held[0] is mean and held[1] is scale and Int(py=held[3]) == d
            and String(held[4]) == mode_text and Bool(py=held[5]) == with_mean and Bool(py=held[6]) == with_std):
        unsafe_memcpy(dest=ap, src=_ptr(held[2]), count=7*d+5)
        _ = held  # keep the Python object alive through the copy above
    else:
        arena = scaler_standard_rows(calls, mean, scale, d, with_mean, with_std, context)
        ap = F32P(unsafe_from_address=Int(arena.unsafe_ptr()))
        if scaler_transform_rows_valid(ap, d):
            var rows = classical_typed(Python.tuple(7*d+5), "<f4", py_item(context, nk[].k_output_context))
            unsafe_memcpy(dest=_ptr(rows), src=ap, count=7*d+5)
            _ = rows  # keep the Python object alive through the copy above
            if mean is not None:
                py_setattr(mean, "_readonly", PythonObject(True))
            if scale is not None:
                py_setattr(scale, "_readonly", PythonObject(True))
            prepared_put(mods, _STANDARD_ROWS_SLOT, model, Python.tuple(
                mean, scale, rows, d, mode_text, PythonObject(with_mean), PythonObject(with_std)))
    # fb12-rows (2026-11-17): the output is written in FULL before it is
    # returned (the direct kernel's download, or the NaN route's second call,
    # or the column-NaN restore's whole-block copy; every other exit raises),
    # so it takes the unfilled store: classical_typed zero-filled n*d words
    # (one extra write pass over the output per transform, 4 bytes a word,
    # 0.26 ms at 4.4 MB on the 16.8 GB/s memcpy of the L40S box) that the
    # writer then overwrote. Same object type and slots; no bit moves.
    var output = classical_wrap_layout(native_array_unfilled(String("f"), n*d),
                                       NativeArrayLayout(native_matrix_shape(n, d), 4), "<f4",
                                       py_item(context, nk[].k_output_context))
    # RUN: the typed core (statistic refusals, kernel, NaN fill and restore,
    # overflow refusal) on the borrowed input and the new output.
    scaler_transform_core(calls, ScalerTransformSpec(False, inverse, with_mean, with_std, False, 0, 0),
                          ap, _ptr(values), n, d, _ptr(output))
    _ = arena^
    _ = mean^
    _ = scale^
    return _write_back(source, values, output, copy, context, snapshot)


def _seen_view(value: PythonObject, d: Int, context: PythonObject) raises -> ScalerSeenView:
    """UNPACK the fitted n_samples_seen_ (a native int or float, or a
    per-feature int64 / float32 row, borrowed) with its refusals."""
    classical_admit_value(value,context)
    var b = native_loaded_module("builtins")
    if py_type(value) is b.int:
        return ScalerSeenView(SCALER_SEEN_SOURCE_SCALAR, Float64(py=value), Int64(py=value), True, 0)
    if py_type(value) is b.float:
        return ScalerSeenView(SCALER_SEEN_SOURCE_SCALAR, Float64(py=value), Int64(0), False, 0)
    if not classical_is_array(value, context) or Int(py=value.ndim) != 1 or Int(py=value.size) != d:
        raise Error("StandardScaler n_samples_seen_ must remain a scalar or per-feature count vector")
    var dtype = String(value.dtype)
    if dtype != "<i8" and dtype != "<f4":
        raise Error("StandardScaler n_samples_seen_ must remain int64 counts or float32 weights")
    var source = SCALER_SEEN_SOURCE_I64 if dtype == "<i8" else SCALER_SEEN_SOURCE_F32
    return ScalerSeenView(source, 0, Int64(0), False, Int(py=classical_address(value)))


def _seen_pack(merged: ScalerStandardMerged, d: Int, context: PythonObject) raises -> PythonObject:
    """PACK the merged n_samples_seen_: one native number when every feature
    saw the same count, else the per-feature row."""
    if merged.seen_kind == SCALER_SEEN_INT:
        return PythonObject(Int(merged.seen_integer))
    if merged.seen_kind == SCALER_SEEN_FLOAT:
        return PythonObject(merged.seen_scalar)
    if merged.seen_kind == SCALER_SEEN_FLOAT_ROW:
        return _copy_row(F32P(unsafe_from_address=Int(merged.seen_f32.unsafe_ptr())), d, context)
    var nk = scaler_keys()
    var count = classical_typed(Python.tuple(d), "<i8", py_item(context, nk[].k_output_context))
    unsafe_memcpy(dest=I64P(unsafe_from_address=Int(py=classical_address(count))),
                  src=I64P(unsafe_from_address=Int(merged.seen_i64.unsafe_ptr())), count=d)
    _ = count  # keep the Python object alive through the copy above
    return count


def scaler_partial_fit(model: PythonObject, source: PythonObject, weights: PythonObject, context: PythonObject) raises -> PythonObject:
    """UNPACK the fitted state and the batch, RUN the typed batch fit and the
    merge (bindings/scaler_lifecycle_run.mojo; lane py-msel-prep-cnn: the
    batch was a second Python StandardScaler fitted through the door and the
    merge program was composed here), PACK the merged fields."""
    var fields = py_fields(model)
    if not _fitted(fields):
        return scaler_fit(model, source, weights, context)
    _ = _mode(fields, context)
    var nk = scaler_keys()
    if _bool(py_item(fields, nk[].k_with_mean), context) != _bool(py_item(fields, nk[].k_with_mean_), context) or _bool(py_item(fields, nk[].k_with_std), context) != _bool(py_item(fields, nk[].k_with_std_), context):
        raise Error("StandardScaler partial_fit: with_mean / with_std changed since the first batch")
    var b = native_loaded_module("builtins")
    # The batch fit's own admission, in the order StandardScaler.fit makes it
    # (copy True, the fitted flags, the fitted mode).
    var with_mean = _bool(py_item(fields, nk[].k_with_mean_), context)
    var with_std = _bool(py_item(fields, nk[].k_with_std_), context)
    var mode = py_item(fields, nk[].k_numeric_mode_)
    if py_type(mode) is not b.str:
        raise Error("numeric_mode must be fast, identical or None")
    mode = py_call(py_call(mode.strip).lower)
    var mode_word = String(mode)
    if mode_word != "fast" and mode_word != "identical":
        raise Error("numeric_mode must be fast, identical or None")
    answer_mode_object(mode, mode_word, MODE_FAMILY_SCALER)
    if Bool(py=py_get(context,nk[].k_scaler_explicit_host,False)) and String(mode) != "identical":
        raise Error("HostStandardScaler runs identical only on the host")
    var entries = scaler_fit_entries(model, mode, context)
    var batch = scaler_fit_input(source, weights, context, PythonObject(None))
    var calls = scaler_door_calls(model, entries, context)
    var x = batch.x
    var weight = batch.weight
    var weighted = batch.weighted
    var batch_d = batch.d
    # RUN: the batch fit (the typed core, NaN and weights included).
    var fit = scaler_standard_fit_core(calls, _ptr(x), batch.n, batch_d, with_mean, with_std,
                                       Int(py=classical_address(weight)) if weighted else 0)
    _ = weight^
    _ = x^
    var d = _features(fields)
    if batch_d != d:
        raise Error("StandardScaler input feature count differs from fit")
    # UNPACK the fitted state (borrowed rows, alive to the end of the merge).
    var old_seen = py_item(fields, nk[].k_n_samples_seen_)
    var old = _seen_view(old_seen, d, context)
    var kept = py_item(fields, nk[].k_mean_) is not None
    var old_mean = PythonObject(None)
    var old_var = PythonObject(None)
    var mean_addr = 0
    var var_addr = 0
    if kept:
        old_mean = _statistic(py_item(fields, nk[].k_mean_), d, "mean_", context)
        mean_addr = Int(py=classical_address(old_mean))
        if not with_mean and not with_std:
            # the batch publishes no mean_ to merge with
            classical_admit_value(PythonObject(None), context)
            raise Error("StandardScaler mean_ must remain a finite contiguous Float32 feature vector")
        if py_item(fields, nk[].k_var_) is not None:
            old_var = _statistic(py_item(fields, nk[].k_var_), d, "var_", context)
            var_addr = Int(py=classical_address(old_var))
    var scratch = List[Int64]()
    var new = scaler_seen_of_fit(fit, scratch, d)
    var stats = Int(fit.stats.unsafe_ptr())
    # RUN: the merge.
    var merged = scaler_standard_merge_run(calls, d, weighted, old, new, kept, mean_addr, stats, var_addr,
                                           stats + 4 * d if with_std else 0)
    _ = scratch^
    _ = fit^
    _ = old_mean^
    _ = old_var^
    _ = old_seen^
    # PACK
    if merged.kept:
        var ptr = F32P(unsafe_from_address=Int(merged.stats.unsafe_ptr()))
        py_setitem(fields, nk[].k_mean_, _copy_row(ptr, d, context))
        if with_std:
            py_setitem(fields, nk[].k_var_, _copy_row(ptr + d, d, context))
            py_setitem(fields, nk[].k_scale_, _copy_row(ptr + 2 * d, d, context))
    py_setitem(fields, nk[].k_n_samples_seen_, _seen_pack(merged, d, context))
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
    var nk = scaler_keys()
    if len(options) == 0:
        var fields = py_fields(model)
        var mode = _mode(fields, context)
        # One upload of X for the whole operation: the combined fit +
        # transform export (every tier; `standard_fit_direct` then
        # `standard_transform_direct` on the resident X, the same kernels on
        # the same words), else the FAST Apple experiment's fused export.
        # Without either (the host column, MOJOLEARN_SCALER_DIRECT=0) the
        # two-step route below runs.
        var entries = scaler_entries(model, mode, context)
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
            var stats = classical_typed(Python.tuple(3,d), "<f4", py_item(context, nk[].k_output_context))
            var output = classical_typed(Python.tuple(n,d), "<f4", py_item(context, nk[].k_output_context))
            var wm = _bool(py_item(fields, nk[].k_with_mean), context)
            var ws = _bool(py_item(fields, nk[].k_with_std), context)
            var got = Int(py=py_call(entry, classical_address(values), classical_address(stats), classical_address(output), Python.list(n,d,Int(wm),Int(ws))))
            _ = output  # keep the Python object alive through the native call above
            _ = stats  # keep the Python object alive through the native call above
            _ = values  # keep the Python object alive through the native call above
            if got != 0:
                var ptr = _ptr(stats)
                _ = stats  # keep the Python object alive through the native call above
                py_setitem(fields, nk[].k_mean_, _copy_row(ptr,d,context) if wm or ws else PythonObject(None))
                _ = stats  # keep the Python object alive through the native call above
                py_setitem(fields, nk[].k_var_, _copy_row(ptr+d,d,context) if ws else PythonObject(None))
                _ = stats  # keep the Python object alive through the native call above
                py_setitem(fields, nk[].k_scale_, _copy_row(ptr+2*d,d,context) if ws else PythonObject(None))
                _ = stats  # keep the Python object alive through the native call above
                py_setitem(fields, nk[].k_n_features_in_, d)
                py_setitem(fields, nk[].k_n_samples_seen_, n)
                py_setitem(fields, nk[].k_numeric_mode_, mode)
                py_setitem(fields, nk[].k_with_mean_, wm)
                py_setitem(fields, nk[].k_with_std_, ws)
                _ = stats^
                if got < 0:
                    raise Error("StandardScaler transform overflowed in Float32")
                _ = output  # keep the Python object alive through the native calls above
                _ = values  # keep the Python object alive through the native calls above
                return _write_back(source, values, output, _bool(py_item(fields, nk[].k_copy),context), context, snapshot)
            _ = output  # keep the Python object alive through the native calls above
            _ = stats  # keep the Python object alive through the native calls above
            _ = values  # keep the Python object alive through the native calls above
    _ = scaler_fit(model, source, py_get(options,nk[].k_sample_weight), context, input_cache)
    return scaler_transform(model, source, False, PythonObject(None), context, input_cache)


def scaler_archive_save(model: PythonObject, path: PythonObject, context: PythonObject) raises -> PythonObject:
    # UNPACK: public slots and optional owner pins only. Native archive code
    # owns metadata bytes, scalar count narrowing, member order and ZIP/NPY.
    var fields = py_fields(model)
    if not _fitted(fields):
        raise Error("RuntimeError: this estimator is not fitted yet")
    var nk = scaler_keys()
    var seen = py_item(fields, nk[].k_n_samples_seen_)
    var integer = py_type(seen) is py_type(PythonObject(0))
    var features = _features(fields)
    var integer_count = Int64(py=seen) if integer else Int64(-1)
    var with_mean = _bool(py_item(fields, nk[].k_with_mean_), context)
    var with_std = _bool(py_item(fields, nk[].k_with_std_), context)
    var saved = NativeStandardArchiveSaved(features, integer_count, with_mean, with_std)
    var estimator = _archive_class_name_unpack(py_type(model))
    var mode = py_item(fields, nk[].k_numeric_mode_)
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
        var nk = scaler_keys()
        var name = py_get(globals,nk[].k___name__)
        if py_type(name) is py_type(PythonObject("")):
            var module = String(name)
            outside = module != "mojolearn" and not module.startswith("mojolearn.") and not module.startswith("mojolearn_verify_all_")
        if py_get(globals,nk[].k___mojolearn_user_boundary__) is PythonObject(True):
            outside = True
        elif py_get(globals,nk[].k___mojolearn_internal__) is PythonObject(True):
            outside = False
    except:
        outside = True
    return outside


def _public_result(result: PythonObject, context: PythonObject) raises -> PythonObject:
    return _public_result(result, context, PythonObject(None))


def _public_result(result: PythonObject, context: PythonObject,
                   input_cache: PythonObject) raises -> PythonObject:
    var nk = scaler_keys()
    if classical_is_array(result,context) and py_get(context,nk[].k_ndarray) is not None and _outside(context,1):
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
        var nk = scaler_keys()
        var op = py_op_code(nk[].ops, all[1])
        var args = all[2]
        var root = all[3]
        var context = root
        var is_class = py_type(model) is native_loaded_module("builtins").type
        var cls = model if is_class else py_type(model)
        if not is_class and (op == SCALER_OPN_TRANSFORM or op == SCALER_OPN_PARTIAL_FIT or op == SCALER_OPN_FIT_TRANSFORM):
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
        var result = PythonObject(None)
        if op == SCALER_OPN_INIT:
            # `device` is the fifth constructor argument (absent from a
            # four-argument caller: this installation's device). Refused in
            # words before any field changes.
            var device = PythonObject(None)
            if len(args) > 4:
                device = args[4]
            scaler_device_admit(device, cls, root)
            result = scaler_initialize(model,args)
            py_setitem(py_fields(model), nk[].k_device, device)
            scaler_route_clear(model)
        elif op == SCALER_OPN_GET_PARAMS:
            var fields = py_fields(model)
            result = Python.dict(copy=py_item(fields, nk[].k_copy),with_mean=py_item(fields, nk[].k_with_mean),with_std=py_item(fields, nk[].k_with_std),numeric_mode=py_item(fields, nk[].k_numeric_mode))
            # None on an instance built before the parameter existed.
            py_setitem(result, nk[].k_device, py_get(fields, nk[].k_device))
        elif op == SCALER_OPN_SET_PARAMS:
            var fields = py_fields(model)
            var options = args[0]
            var b = native_loaded_module("builtins")
            var names = py_call(b.list,options)
            var known = Python.tuple(nk[].k_copy,nk[].k_with_mean,nk[].k_with_std,nk[].k_numeric_mode,nk[].k_device)
            for k in range(len(names)):  # driver-loop: proposed estimator parameter names
                if py_type(names[k]) is not b.str or not py_contains(known,names[k]):
                    raise Error("Invalid StandardScaler parameters")
            if len(names):
                var device = py_get(options,nk[].k_device,py_get(fields, nk[].k_device))
                # Refused in words before the estimator changes.
                scaler_device_admit(device, cls, root)
                var values = Python.tuple(py_get(options,nk[].k_copy,py_item(fields, nk[].k_copy)),py_get(options,nk[].k_with_mean,py_item(fields, nk[].k_with_mean)),py_get(options,nk[].k_with_std,py_item(fields, nk[].k_with_std)),py_get(options,nk[].k_numeric_mode,py_item(fields, nk[].k_numeric_mode)))
                _ = py_call(fields.clear)
                _ = scaler_initialize(model,values)
                py_setitem(fields, nk[].k_device, device)
                scaler_route_clear(model)
            result = model
        elif op == SCALER_OPN_VENDOR_USED:
            result = scaler_route_vendor_used(model, root)
        elif op == SCALER_OPN_IS_FITTED:
            result = PythonObject(_fitted(py_fields(model)))
        elif op == SCALER_OPN_TRANSFORM:
            var input_cache = Python.list()
            result = _public_result(scaler_transform(model,args[0],Bool(py=args[1]),args[2],context,input_cache),context,input_cache)
        elif op == SCALER_OPN_PARTIAL_FIT:
            result = scaler_partial_fit(model,args[0],args[2],context)
        elif op == SCALER_OPN_FIT_TRANSFORM:
            var input_cache = Python.list()
            result = _public_result(scaler_fit_transform(model,args[0],args[2],context,input_cache),context,input_cache)
        elif op == SCALER_OPN_SAVE:
            result = scaler_archive_save(model,args[0],context)
        elif op == SCALER_OPN_LOAD:
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
