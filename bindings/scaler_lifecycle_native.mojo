# SPDX-License-Identifier: Apache-2.0
# StandardScaler's public lifecycle. The original preprocessing and x_prep
# kernels, operation order, modes, and copy semantics remain unchanged.
from std.memory import memcpy, bitcast
from std.os import getenv
from std.python import Python, PythonObject
from bindings.optimizer_public_words import PYTHON_ERROR_PENDING, optimizer_python_error
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem, py_contains
from bindings.native_error_class import error_without_class
from bindings.classical_attributes import classical_connect_numpy, classical_getattribute, classical_check_layout
from bindings.classical_arrays import classical_typed, classical_input, classical_address, classical_is_array, classical_is_f32, classical_c_order, classical_i64, classical_dtype_str
from bindings.classical_archive import classical_archive_write
from bindings.classical_archive_read import classical_archive_read, _archive_exact, _archive_string
from bindings.classical_exceptions import classical_not_fitted_class
from bindings.causal_exception import causal_borrowed_item, causal_set_exception
from bindings.classical_pandas import native_pandas_exact, classical_admit_value
from bindings.scaler_tags_native import scaler_tags
from bindings.sequential_inputs import sequential_input
from bindings.sequential_admission import _ordinary_torch
from bindings.native_torch_admission import native_torch_numpy
from bindings.scaler_native import scaler_fit, _input, _ptr, _stage, _copy_row, _bool
from core.scaler_fit_impl import scaler_fill, scaler_all_finite, scaler_refuse_infinite_columns, scaler_feature_indices, scaler_counts_f32, scaler_counts_i64, scaler_equal_f32, scaler_equal_i64, scaler_counts_f64, scaler_merge_count_only, scaler_narrow_counts

alias F32P = MutPointer[Float32, MutUntrackedOrigin]
alias I64P = MutPointer[Int64, MutUntrackedOrigin]


def _entries(mode: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    if py_type(mode) is not b.str or (String(mode) != "fast" and String(mode) != "identical"):
        raise Error("numeric_mode must be fast, identical or None")
    if Bool(py=py_get(context,"scaler_explicit_host",False)) and String(mode) != "identical":
        raise Error("HostStandardScaler runs identical only on the host")
    var entries = context["scaler_bindings"][mode]
    if py_type(entries) is b.str:
        raise Error("ImportError: " + String(entries))
    var expected = 1 if String(mode) == "identical" else 0
    if Int(py=py_call(entries["preprocessing_numeric_mode"])) != expected or Int(py=py_call(entries["x_prep_numeric_mode"])) != expected:
        raise Error("RuntimeError: Scaler native numeric mode disagrees with requested mode")
    return entries


def _mode(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    _ = _bool(fields["copy"], context)
    _ = _bool(fields["with_mean"], context)
    _ = _bool(fields["with_std"], context)
    var mode = fields["numeric_mode"]
    if mode is None:
        mode = context["backend_fields"]["_DEFAULT_MODE"]
    if py_type(mode) is not Python.import_module("builtins").str:
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
    _ = py_call(entries["x_prep_run"], Int(arena.unsafe_ptr()), len(arena), Int(program.unsafe_ptr()), len(program) // 16)


def _features(fields: PythonObject) raises -> Int:
    var value = fields["n_features_in_"]
    if py_type(value) is not Python.import_module("builtins").int:
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
    if copy or native_pandas_exact(source):
        return output
    var b = Python.import_module("builtins")
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
    classical_admit_value(value,context)
    if not classical_is_array(value, context):
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
    memcpy(dest=ap, src=_ptr(values), count=n*d)
    scaler_feature_indices(ap+keep, d)
    var program = List[Int32]()
    _stage(program, 1, d, 0, n, d, st)
    _stage(program, 24, n*d, 0, n, d, stat, dest, keep, d)
    _run(arena, program, entries)
    scaler_refuse_infinite_columns(ap, st, d)
    var output = classical_typed(Python.tuple(n,d), "<f4", context["output_context"])
    memcpy(dest=_ptr(output), src=ap+dest, count=n*d)
    _ = arena^
    return output


def scaler_transform(model: PythonObject, source: PythonObject, inverse: Bool, copy_option: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    var copy = _bool(fields["copy"] if copy_option is None else copy_option, context)
    if not _fitted(fields):
        _ = classical_not_fitted_class(context)
        raise Error("NotFittedError: StandardScaler is not fitted")
    var entries = _entries(fields["numeric_mode_"], context)
    var direct = _direct(entries, "standard_transform_direct")
    var values = _input(source, 2, True, context)
    var n = Int(py=values.shape[0])
    var d = Int(py=values.shape[1])
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
    if mean is None:
        mean = classical_typed(Python.tuple(d), "<f4", context["output_context"])
    if scale is None:
        scale = classical_typed(Python.tuple(d), "<f4", context["output_context"])
        scaler_fill(_ptr(scale), d, 1)
    mean = _statistic(mean, d, "mean_", context)
    scale = _statistic(scale, d, "scale_", context)
    var arena = List[Float32](length=7*d+5, fill=0)
    var ap = F32P(unsafe_from_address=Int(arena.unsafe_ptr()))
    memcpy(dest=ap, src=_ptr(mean), count=d)
    memcpy(dest=ap+d, src=_ptr(scale), count=d)
    arena[7*d+4] = 1
    var program = List[Int32]()
    _stage(program, 264, d, 0, d, Int(with_mean), Int(with_std), 7*d+3, 7*d+4, 2*d, 3*d, 4*d, 5*d, 6*d)
    _stage(program, 248, 1, 4*d, d, 7*d)
    _stage(program, 248, 1, 5*d, d, 7*d+1)
    _stage(program, 248, 1, 6*d, d, 7*d+2)
    _run(arena, program, entries)
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
        if got == 0:
            raise Error("StandardScaler transform overflowed in Float32")
        finite = got == 1
        if not finite:
            var filled = _nan_fill(values, entries, context)
            if Int(py=py_call(direct, classical_address(filled), Int(ap+2*d), Int(ap+3*d), classical_address(output), params)) != 1:
                raise Error("StandardScaler transform overflowed in Float32")
            _ = filled^
    else:
        finite = scaler_all_finite(_ptr(values), n*d)
        var filled = values if finite else _nan_fill(values, entries, context)
        _ = py_call(entries["standard_transform"], classical_address(filled), Int(ap+2*d), Int(ap+3*d), classical_address(output), params)
        if not scaler_all_finite(_ptr(output), n*d):
            raise Error("StandardScaler transform overflowed in Float32")
        _ = filled^
    if not finite or colnan:
        var keep = List[Float32](length=3*n*d+2*d, fill=0)
        var kp = F32P(unsafe_from_address=Int(keep.unsafe_ptr()))
        memcpy(dest=kp, src=_ptr(values), count=n*d)
        memcpy(dest=kp+n*d, src=_ptr(output), count=n*d)
        memcpy(dest=kp+2*n*d, src=ap+4*d, count=d)
        var restore = List[Int32]()
        # c2_nan_sub publishes int32 words, while nan_keep reads numeric
        # Float32 flags. Preserve the old _Prog.put_codes i2f stage: raw
        # integer 1 is a subnormal float that Metal can flush to zero.
        _stage(restore, 10, d, 2*n*d, 2*n*d+d)
        _stage(restore, 103, n*d, 0, d, n*d, 2*n*d+d, 2*n*d+2*d)
        _run(keep, restore, entries)
        memcpy(dest=_ptr(output), src=kp+2*n*d+2*d, count=n*d)
        _ = keep^
    _ = arena^
    _ = mean^
    _ = scale^
    return _write_back(source, values, output, copy, context)


def _counts(value: PythonObject, d: Int, dest: F32P, context: PythonObject) raises:
    classical_admit_value(value,context)
    var b = Python.import_module("builtins")
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
    var scalar = py_type(value) is Python.import_module("builtins").int
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
    var b = Python.import_module("builtins")
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
        if scaler_merge_count_only(wp,wp+d,d):
            count = PythonObject(wp[0]) if weighted else PythonObject(Int(wp[0]))
        else:
            count = classical_typed(Python.tuple(d),"<f4" if weighted else "<i8",context["output_context"])
            scaler_narrow_counts(wp,Int(py=classical_address(count)),weighted,d)
        py_setitem(fields,"n_samples_seen_",count)
        _ = wide^
        _ = arena^
        return model
    else:
        var old_mean = _statistic(fields["mean_"], d, "mean_", context)
        var new_mean = _statistic(fresh["mean_"], d, "mean_", context)
        memcpy(dest=ap+2*d, src=_ptr(old_mean), count=d)
        memcpy(dest=ap+3*d, src=_ptr(new_mean), count=d)
        if fields["var_"] is not None:
            var old_var = _statistic(fields["var_"], d, "var_", context)
            memcpy(dest=ap+4*d, src=_ptr(old_var), count=d)
            _ = old_var^
        if fresh["var_"] is not None:
            var new_var = _statistic(fresh["var_"], d, "var_", context)
            memcpy(dest=ap+5*d, src=_ptr(new_var), count=d)
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
            memcpy(dest=dest, src=ip+2*d, count=d)
        _ = counts^
    py_setitem(fields, "n_samples_seen_", count)
    _ = arena^
    _ = old_seen^
    _ = new_seen^
    return model


def scaler_fit_transform(model: PythonObject, source: PythonObject, options: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var names = py_call(b.list, options)
    for k in range(len(names)):  # driver-loop: fit's keyword names, not data
        if py_type(names[k]) is not b.str or String(names[k]) != "sample_weight":
            raise Error("TypeError: StandardScaler.fit got an unexpected keyword argument")
    if len(options) == 0:
        var fields = py_fields(model)
        var mode = _mode(fields, context)
        if String(mode) == "fast":
            var entries = _entries(mode, context)
            var entry = _direct(entries, "standard_fit_transform_direct")
            if entry is not None:
                var values = _input(source, 2, True, context)
                var n = Int(py=values.shape[0])
                var d = Int(py=values.shape[1])
                var stale = py_call(b.list, fields)
                for k in range(len(stale)):  # driver-loop: invalidate fitted attribute names on refit
                    if String(stale[k]).endswith("_"):
                        _ = py_call(fields.pop, stale[k])
                var stats = classical_typed(Python.tuple(3,d), "<f4", context["output_context"])
                var output = classical_typed(Python.tuple(n,d), "<f4", context["output_context"])
                var wm = _bool(fields["with_mean"], context)
                var ws = _bool(fields["with_std"], context)
                var got = Int(py=py_call(entry, classical_address(values), classical_address(stats), classical_address(output), Python.list(n,d,Int(wm),Int(ws))))
                if got != 0:
                    var ptr = _ptr(stats)
                    py_setitem(fields, "mean_", _copy_row(ptr,d,context) if wm or ws else PythonObject(None))
                    py_setitem(fields, "var_", _copy_row(ptr+d,d,context) if ws else PythonObject(None))
                    py_setitem(fields, "scale_", _copy_row(ptr+2*d,d,context) if ws else PythonObject(None))
                    py_setitem(fields, "n_features_in_", d)
                    py_setitem(fields, "n_samples_seen_", n)
                    py_setitem(fields, "numeric_mode_", mode)
                    py_setitem(fields, "with_mean_", wm)
                    py_setitem(fields, "with_std_", ws)
                    _ = stats^
                    if got < 0:
                        raise Error("StandardScaler transform overflowed in Float32")
                    return _write_back(source, values, output, _bool(fields["copy"],context), context)
    _ = scaler_fit(model, source, py_get(options,"sample_weight"), context)
    return scaler_transform(model, source, False, PythonObject(None), context)


def scaler_archive_save(model: PythonObject, path: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    if not _fitted(fields):
        raise Error("RuntimeError: this estimator is not fitted yet")
    var b = Python.import_module("builtins")
    var seen = fields["n_samples_seen_"]
    var integer = py_type(seen) is b.int
    var meta = classical_typed(Python.tuple(4), "<i8", context["output_context"])
    var mp = I64P(unsafe_from_address=Int(py=classical_address(meta)))
    mp[0] = Int64(_features(fields))
    mp[1] = Int64(py=seen) if integer else Int64(-1)
    mp[2] = Int64(_bool(fields["with_mean_"],context))
    mp[3] = Int64(_bool(fields["with_std_"],context))
    var arrays = Python.dict(format="mojolearn-scaler-1", estimator=py_type(model).__name__, numeric_mode=fields["numeric_mode_"], meta=meta)
    if not integer:
        if not classical_is_array(seen,context):
            if py_type(seen) is not b.float:
                raise Error("TypeError: StandardScaler counts must be native numbers or count buffers")
            var scalar = classical_typed(Python.tuple(1), "<f4", context["output_context"])
            _ptr(scalar)[0] = Float32(py=seen)
            seen = scalar
        py_setitem(arrays,"n_samples_seen",seen)
    var names = Python.tuple("mean","var","scale")
    for k in range(3):  # small-loop(stats: three saved statistic names)
        var value = fields[names[k]+"_"]
        if value is not None:
            _ = _statistic(value,_features(fields),String(names[k]),context)
            py_setitem(arrays,names[k],value)
    return classical_archive_write(path,arrays,context)


def scaler_archive_build(cls: PythonObject, arrays: PythonObject, path: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    if String(_archive_string(arrays,"format",context)) != "mojolearn-scaler-1":
        raise Error("model archive format does not match mojolearn-scaler-1")
    var saved = _archive_string(arrays,"estimator",context)
    var ancestry = py_call(b.type.__dict__["__mro__"].__get__,cls)
    var known = False
    for k in range(len(ancestry)):  # driver-loop: native loading class ancestry names
        known = known or Bool(py=ancestry[k].__name__ == saved)
    if not known:
        raise Error("model estimator does not match loading class")
    var mode = _archive_string(arrays,"numeric_mode",context)
    if String(mode) != "fast" and String(mode) != "identical":
        raise Error("invalid saved numeric_mode")
    var meta = _archive_exact(arrays,"meta","<i8",context)
    if Int(py=meta.size) != 4:
        raise Error("StandardScaler meta must hold four fields")
    var d = classical_i64(meta,0)
    var wm = classical_i64(meta,2) != 0
    var ws = classical_i64(meta,3) != 0
    var model = py_call(b.object.__new__,cls)
    _ = scaler_initialize(model,Python.tuple(True,wm,ws,PythonObject(None)))
    var fields = py_fields(model)
    var output_context = py_get(context,"output_context",context)
    var names = Python.tuple("mean","var","scale")
    for k in range(3):  # small-loop(stats: saved StandardScaler statistics)
        var present = wm or ws if k == 0 else ws
        if present != py_contains(arrays,names[k]):
            raise Error("StandardScaler saved statistics do not match with_mean and with_std")
        var value = PythonObject(None)
        if present:
            var stored = _archive_exact(arrays,String(names[k]),"<f4",context)
            if Int(py=stored.ndim) != 1 or Int(py=stored.size) != d:
                raise Error("StandardScaler saved statistic does not match n_features_in_")
            value = classical_typed(Python.tuple(d),"<f4",output_context)
            memcpy(dest=_ptr(value),src=_ptr(stored),count=d)
            _ = stored^
        py_setitem(fields,names[k]+"_",value)
    var seen = PythonObject(classical_i64(meta,1))
    if classical_i64(meta,1) < 0:
        var stored = py_get(arrays,"n_samples_seen")
        if stored is None:
            raise Error("StandardScaler n_samples_seen is missing")
        var dtype = classical_dtype_str(stored,context)
        if (dtype != "<f4" and dtype != "<i8") or Int(py=stored.ndim) != 1 or (Int(py=stored.size) != 1 and Int(py=stored.size) != d):
            raise Error("StandardScaler n_samples_seen does not match n_features_in_")
        if Int(py=stored.size) == 1:
            seen = PythonObject(Float64(_ptr(stored)[0])) if dtype == "<f4" else PythonObject(Float64(classical_i64(stored,0)))
        else:
            seen = classical_typed(Python.tuple(d),dtype,output_context)
            memcpy(dest=MutPointer[UInt8,MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(seen))), src=MutPointer[UInt8,MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(stored))), count=d*(4 if dtype == "<f4" else 8))
        _ = stored^
    py_setitem(fields,"n_features_in_",d)
    py_setitem(fields,"n_samples_seen_",seen)
    py_setitem(fields,"numeric_mode_",mode)
    py_setitem(fields,"with_mean_",wm)
    py_setitem(fields,"with_std_",ws)
    _ = meta^
    return model


def scaler_archive_load(cls: PythonObject, path: PythonObject, context: PythonObject) raises -> PythonObject:
    return scaler_archive_build(cls,classical_archive_read(path,context),path,context)


def _outside(context: PythonObject, depth: Int) raises -> Bool:
    var outside = True
    try:
        var frame = py_call(context["classical_getframe"],depth)
        var globals = frame.f_globals
        var name = py_get(globals,"__name__")
        if py_type(name) is Python.import_module("builtins").str:
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
    if classical_is_array(result,context) and py_get(context,"ndarray") is not None and _outside(context,1):
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
        if not _outside(context,0):
            var fields = py_fields(model)
            var original = py_get(fields,name)
            if classical_is_array(original,context):
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
        var cls = model if py_type(model) is Python.import_module("builtins").type else py_type(model)
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
            var b = Python.import_module("builtins")
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
        elif op == "tags":
            result = scaler_tags(model,context)
        elif op == "is_fitted":
            result = PythonObject(_fitted(py_fields(model)))
        elif op == "transform":
            result = _public_result(scaler_transform(model,args[0],Bool(py=args[1]),args[2],context),context)
        elif op == "partial_fit":
            result = scaler_partial_fit(model,args[0],args[2],context)
        elif op == "fit_transform":
            result = _public_result(scaler_fit_transform(model,args[0],args[2],context),context)
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
