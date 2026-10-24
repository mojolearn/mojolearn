# SPDX-License-Identifier: Apache-2.0
# Explicit CPU KDE fit owns validation, retained buffers and fitted state.
# Its CPU numerical binding has no immutable-fit or resident-fit registry;
# preparing the stored float32 rows is the entire fit operation.
from std.python import Python, PythonObject
from bindings.native_buffer_owner import native_buffer_view, native_storage_allocate
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setattr
from bindings.classical_attributes import classical_connect_numpy, classical_check_layout, classical_getattribute
from bindings.classical_arrays import classical_input, classical_address, classical_is_array, classical_is_f32, classical_c_order, classical_kind, classical_cast_kind
from bindings.classical_pandas import classical_admit_value, native_pandas_exact, native_pandas_f32, _pd_fields, _pd_get, _pd_class, _pd_type_fields
from bindings.classical_validate import RESHAPE_HINT
from bindings.classical_warnings import classical_warning, _warning_category, COLUMN_WARNING
from bindings.sequential_admission import _ordinary_torch, _one_of
from bindings.native_torch_admission import native_torch_numpy
from bindings.host_tree_native import host_tree_constructor_fields
from bindings.host_sparse_native import host_sparse_format, host_sparse_dense, _sparse_fields
from bindings.host_array_native import host_archive_typed
from bindings.host_inference_scalars import host_inference_scalar
from bindings.host_logistic_native import host_logistic_classes
from core.scaler_fit_impl import scaler_cast
from core.host_inference_impl import host_inference_finite, host_inference_raw_finite
from core.classical_arrays_impl import classical_check_weights
from core.host_sparse_impl import host_sparse_uint64_bound


def _kde_shape(shape: PythonObject) raises -> String:
    var text = String("(")
    for i in range(len(shape)):  # driver-loop: exact native shape metadata formatting
        if i > 0:
            text += ", "
        text += String(Int(py=shape[i]))
    if len(shape) == 1:
        text += ","
    return text + ")"


def _kde_array_view(value: PythonObject, context: PythonObject) raises -> PythonObject:
    if classical_is_array(value, context):
        return value
    # Exact native ndarray admission precedes this function. Hold a native
    # buffer export for the complete fitted lifetime, including readonly data.
    var b = Python.import_module("builtins")
    var pin = native_buffer_view(value)
    var flat = py_call(py_call(pin.cast, "B").cast, "f")
    var out = native_storage_allocate(context["array_class"])
    py_setattr(out, "_store", value)
    py_setattr(out, "_base", PythonObject(None))
    py_setattr(out, "_pin", pin)
    py_setattr(out, "_mv", flat)
    py_setattr(out, "_addr", classical_address(value))
    py_setattr(out, "_readonly", pin.readonly)
    py_setattr(out, "shape", value.shape)
    py_setattr(out, "strides", value.strides)
    py_setattr(out, "dtype", PythonObject("<f4"))
    py_setattr(out, "ndim", value.ndim)
    py_setattr(out, "size", value.size)
    py_setattr(out, "nbytes", value.nbytes)
    py_setattr(out, "itemsize", PythonObject(4))
    py_setattr(out, "order", PythonObject("C"))
    return out


def _kde_pandas(source: PythonObject, context: PythonObject) raises -> PythonObject:
    # The ordinary pandas __array__ path borrowed a homogeneous float32
    # block. Preserve that alias when its transposed storage is C-contiguous;
    # multiple/mixed blocks already needed a new float32 allocation.
    var result = native_pandas_f32(source, context, False, True)
    var manager = _pd_fields(source)["_mgr"]
    var series = py_type(source) is _pd_class("pandas.core.series", "Series")
    var blocks = _pd_fields(manager)["blocks"] if series else _pd_get("pandas._libs.internals", "BlockManager", "blocks", manager)
    if len(blocks) == 1:
        var data = _pd_get("pandas._libs.internals", "Block", "values", blocks[0])
        if py_type(data) is context["ndarray"] and classical_is_f32(data, context) and Bool(py=data.dtype.isnative):
            if series and Bool(py=data.flags.c_contiguous):
                return Python.tuple(data, False)
            if not series and Bool(py=data.flags.f_contiguous):
                var pin = native_buffer_view(data)
                var view = py_call(context["ndarray"], result.shape, dtype=context["float32"], buffer=pin,
                    strides=Python.tuple(data.strides[1], data.strides[0]))
                return Python.tuple(view, False)
    return Python.tuple(result, True)


def _kde_prepare(source: PythonObject, context: PythonObject, name: String) raises -> PythonObject:
    classical_admit_value(source, context)
    if native_pandas_exact(source):
        return _kde_pandas(source, context)
    var b = Python.import_module("builtins")
    var prepared = native_torch_numpy(source) if _ordinary_torch(py_type(source), context) else source
    var value = classical_input(prepared, context, name)
    var copied = py_type(source) is b.list or py_type(source) is b.tuple or py_type(source) is b.range
    if py_type(source) is b.memoryview:
        copied = copied or not Bool(py=source.contiguous)
    return Python.tuple(value, copied)


def _kde_object_storage(value: PythonObject, context: PythonObject) raises -> PythonObject:
    # Ordinary object buffers containing native real scalars were supported
    # by _buffer._materialize. Read native object slots, admit every element,
    # then preserve its int64/float64 intermediate before the float32 cast.
    var b = Python.import_module("builtins")
    if py_type(value) is not context["ndarray"]:
        raise Error("TypeError: object storage requires an exact native ndarray")
    var flat = value.flat
    var integral = True
    for i in range(Int(py=value.size)):  # driver-loop: native object-buffer scalar admission
        var item = flat[i]
        var cls = py_type(item)
        if cls is b.int or cls is b.bool:
            continue
        integral = False
        if cls is not b.float and not _one_of(cls, context["scalar_types"]):
            raise Error("TypeError: KDE object buffers require native real scalar values without Python hooks")
        host_inference_scalar(item, context)
    var out = host_archive_typed(value.shape, "<i8" if integral else "<f8", context["output_context"])
    var address = Int(py=classical_address(out))
    for i in range(Int(py=value.size)):  # driver-loop: marshal admitted object scalars into native numeric storage
        var item = flat[i]
        if integral:
            if Bool(py=item < context["int64_min"]) or Bool(py=item > context["int64_max"]):
                raise Error("OverflowError: int too big to convert")
            MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=address)[i] = Int64(Int(py=item))
        else:
            MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=address)[i] = Float64(py=item)
    _ = flat^
    return out


def _kde_cast(value: PythonObject, context: PythonObject, name: String) raises -> PythonObject:
    if classical_kind(value, context) == "O":
        var numeric = _kde_object_storage(value, context)
        var result = _kde_cast(numeric, context, name)
        _ = numeric^
        return result
    var rank = Int(py=value.ndim)
    var rows = Int(py=value.shape[0]) if rank > 0 else 1
    var cols = Int(py=value.shape[1]) if rank == 2 else 1
    if rows < 1 or cols < 1 or rows > 2147483647 // cols:
        raise Error(name + " is empty or exceeds native indexing")
    var native = classical_is_array(value, context) or Bool(py=value.dtype.isnative)
    if classical_is_f32(value, context) and classical_c_order(value, context) and native:
        return _kde_array_view(value, context)
    if rank > 0 and classical_kind(value, context) == "u" and Int(py=value.itemsize) == 8:
        host_sparse_uint64_bound(Int(py=classical_address(value)), rows, cols,
            Int(py=value.strides[0]) if rank else 0, Int(py=value.strides[1]) if rank == 2 else 0, native)
    var result = host_archive_typed(value.shape, "<f4", context["output_context"])
    var kind = classical_cast_kind(value, Int(py=value.itemsize), name, context)
    var owner = value
    scaler_cast(Int(py=classical_address(value)), Int(py=value.strides[0]) if rank else 0,
        Int(py=value.strides[1]) if rank == 2 else 0, rows, cols, Int(py=value.itemsize), kind, native,
        MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(result))))
    _ = owner^
    return result


def _kde_y(source: PythonObject, rows: Int, est: String, context: PythonObject) raises:
    if source is None:
        return
    var b = Python.import_module("builtins")
    var sparse = host_sparse_format(source)
    if sparse.byte_length() > 0:
        # y is ignored by this estimator. Read only admitted sparse shape;
        # never call a sparse method or inspect unused element protocols.
        var shape = _sparse_fields(source)["_shape"]
        if py_type(shape) is not b.tuple or len(shape) != 2 or py_type(shape[0]) is not b.int or py_type(shape[1]) is not b.int:
            raise Error("TypeError: sparse target shape requires native integers")
        if rows >= 0 and Int(py=shape[0]) != rows:
            raise Error("Found input variables with inconsistent numbers of samples: [" + String(rows) + ", " + String(Int(py=shape[0])) + "]")
        return
    classical_admit_value(source, context)
    # The legacy validator ignores scalar strings, scalar numbers and
    # string/object label vectors apart from a described leading dimension.
    var sequence = py_type(source) is b.list or py_type(source) is b.tuple
    var first = source
    while py_type(first) is b.list or py_type(first) is b.tuple:  # driver-loop: target rank metadata
        if len(first) == 0:
            break
        first = first[0]
    var labels = py_type(first) is b.str or py_type(first) is b.bytes or first is None
    if labels and sequence:
        if rows >= 0 and len(source) != rows:
            raise Error("Found input variables with inconsistent numbers of samples: [" + String(rows) + ", " + String(len(source)) + "]")
        return
    if py_type(source) is b.str or py_type(source) is b.bytes or py_type(source) is b.dict:
        return
    var pair = _kde_prepare(source, context, "y")
    var value = pair[0]
    var rank = Int(py=value.ndim)
    var kind = classical_kind(value, context)
    if kind == "c":
        raise Error("Complex data not supported: " + est + " got a complex y.")
    if rank == 2 and Int(py=value.shape[1]) == 1 and not sequence and py_type(source) is not b.memoryview:
        var location = PythonObject(None)
        classical_warning(COLUMN_WARNING, _warning_category(context), context, location, "HostInferenceWarning")
    var width = Int(py=value.itemsize)
    if kind == "f" and (width == 4 or width == 8):
        try:
            var nrows = Int(py=value.shape[0]) if rank else 1
            var columns = Int(py=value.size) // max(nrows, 1) if rank > 1 else 1
            # Ordinary rank-one/two buffers retain their actual strides.
            # Higher-rank native buffers are packed through a native view.
            if rank > 2:
                if classical_is_array(value, context):
                    classical_admit_value(value._mv, context)
                    value = classical_input(value._mv, context["output_context"], "y")
                else:
                    var packed = py_call(native_buffer_view(value).tobytes, "C")
                    value = py_call(context["ndarray"], Python.tuple(Int(py=value.size)), dtype=value.dtype, buffer=packed)
                nrows = Int(py=value.size)
                columns = 1
            host_inference_raw_finite(Int(py=classical_address(value)), Int(py=value.strides[0]) if rank else 0,
                Int(py=value.strides[1]) if rank == 2 else 0, nrows, columns, width,
                classical_is_array(value, context) or Bool(py=value.dtype.isnative))
        except:
            raise Error("Input y contains NaN or infinity (" + est + ").")
    if rows >= 0 and rank > 0 and Int(py=pair[0].shape[0]) != rows:
        raise Error("Found input variables with inconsistent numbers of samples: [" + String(rows) + ", " + String(Int(py=pair[0].shape[0])) + "]")
    _ = value^
    _ = pair^


def _kde_weights(source: PythonObject, rows: Int, context: PythonObject) raises -> PythonObject:
    if source is None:
        return PythonObject(None)
    var b = Python.import_module("builtins")
    # Empty native sequences have a valid one-dimensional weight shape.
    # Check that shape before the general nonempty sequence materializer.
    var source_type = py_type(source)
    if _one_of(source_type, context["host_kde_sequence_types"]) and len(source) == 0:
        raise Error("sample_weight.shape == (0,), expected (" + String(rows) + ",)!")
    var pair = _kde_prepare(source, context, "sample_weight")
    var raw = pair[0]
    var rank = Int(py=raw.ndim)
    if classical_kind(raw, context) == "c":
        var format = native_buffer_view(source).format
        raise Error("Complex data not supported: buffer format '" + String(format) + "' is complex")
    if rank > 1 or py_type(source) is b.bool:
        raise Error("Sample weights must be 1D array or scalar")
    if rank == 1 and Int(py=raw.shape[0]) != rows:
        raise Error("sample_weight.shape == (" + String(Int(py=raw.shape[0])) + ",), expected (" + String(rows) + ",)!")
    var converted = _kde_cast(raw, context, "sample_weight")
    var scalar = rank == 0
    var out = host_archive_typed(Python.tuple(rows), "<f4", context["output_context"]) if scalar else converted
    try:
        host_inference_finite(Int(py=classical_address(converted)), 1 if scalar else rows)
    except:
        raise Error("Input sample_weight contains NaN or infinity.")
    var positive = classical_check_weights(
        MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(converted))),
        MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(out))), rows, scalar)
    if not positive:
        raise Error("Sample weights must contain at least one non-zero number.")
    _ = converted^
    _ = raw^
    _ = pair^
    return out


def host_kde_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    classical_connect_numpy(context)
    classical_check_layout(model, context)
    var fields = host_tree_constructor_fields(model, context)
    var b = Python.import_module("builtins")
    var mro = py_call(b.type.__dict__["__mro__"].__get__, py_type(model))
    var known = False
    for i in range(len(mro)):  # driver-loop: fixed KDE class ancestry and hook refusal
        var family = py_get(context["host_inference_kinds"], mro[i])
        known = known or (family is not None and String(family) == "kde")
        if py_get(_pd_type_fields(mro[i]), "_mojolearn_after_fit") is not None:
            raise Error("TypeError: KDE fit does not invoke Python after-fit hooks")
    if not known:
        raise Error("TypeError: native KDE fit requires a connected host KDE class")
    var keys = py_call(py_type(Python.list()), fields.keys())
    for i in range(len(keys)):  # driver-loop: native stored model admission before replacement/destruction
        classical_admit_value(fields[keys[i]], context)
    # Replaced state must not release object buffers with Python finalizers.
    var buffers = Python.tuple("_x", "_w")
    for i in range(len(buffers)):  # driver-loop: replaced KDE buffer lifetime admission
        var prior = py_get(fields, buffers[i])
        if py_type(prior) is py_get(context, "ndarray") and classical_kind(prior, context) == "O":
            raise Error("TypeError: KDE stored buffers cannot contain Python objects")
    var names = py_get(fields, "feature_names_in_")
    if py_type(names) is py_get(context, "ndarray") and classical_kind(names, context) == "O":
        _ = host_logistic_classes(names, context)
    var overwritten = Python.tuple("_mojolearn_fitted", "input_copied_", "n_features_in_", "n_samples_fit_")
    for i in range(len(overwritten)):  # driver-loop: overwritten scalar lifetime admission
        var prior = py_get(fields, overwritten[i])
        if py_type(prior) is py_get(context, "ndarray") and classical_kind(prior, context) == "O":
            raise Error("TypeError: KDE scalar state cannot contain Python objects")
    host_inference_scalar(py_get(fields, "_mojolearn_fitting", 0), context)
    var depth = Int(py=py_get(fields, "_mojolearn_fitting", 0))
    var est = String(py_call(b.type.__dict__["__name__"].__get__, py_type(model)))
    var source = args[0]
    var sparse = host_sparse_format(source)
    var pair = PythonObject(None)
    var raw = PythonObject(None)
    var copied = False
    var described = True
    if sparse.byte_length() > 0:
        raw = host_sparse_dense(source, sparse, context)
        var stored = _sparse_fields(source)
        var dtype = stored["dtype"] if sparse == "lil" or sparse == "dok" else stored["data"].dtype
        copied = dtype is not context["float32"] or (sparse == "csc" and Int(py=raw.shape[0]) > 1 and Int(py=raw.shape[1]) > 1)
    else:
        pair = _kde_prepare(source, context, "X")
        raw = pair[0]
        copied = Bool(py=pair[1])
        var kind = classical_kind(raw, context)
        var rank = Int(py=raw.ndim)
        described = kind == "b" or kind == "i" or kind == "u" or kind == "f" or kind == "c"
        if depth == 0 and described:
            if kind == "c":
                raise Error("Complex data not supported: " + est + " got a complex X.")
            if rank < 2:
                raise Error("Expected 2D array, got " + ("scalar" if rank == 0 else "1D") + " array instead (shape=" + _kde_shape(raw.shape) + "). " + RESHAPE_HINT)
            if rank == 2 and (Int(py=raw.shape[0]) == 0 or Int(py=raw.shape[1]) == 0):
                var dimension = "sample(s)" if Int(py=raw.shape[0]) == 0 else "feature(s)"
                raise Error("Found array with 0 " + dimension + " (shape=" + _kde_shape(raw.shape) + ") while a minimum of 1 is required by " + est + ".")
        if rank != 2:
            raise Error("mojolearn: X must be 2-D, got " + String(rank) + "-D shape " + _kde_shape(raw.shape))
        copied = copied or not classical_is_f32(raw, context) or not classical_c_order(raw, context) or (not classical_is_array(raw, context) and not Bool(py=raw.dtype.isnative))
    if depth == 0:
        _kde_y(args[1], Int(py=raw.shape[0]) if described else -1, est, context)
    try:
        _kde_fit_store(model, fields, raw, args[2], copied, est, context)
    finally:
        if depth <= 0:
            _ = py_call(fields.pop, "_mojolearn_fitting", PythonObject(None))
    fields["_mojolearn_fitted"] = PythonObject(True)
    if depth == 0:
        # The outer validator converted DataFrames before reading names.
        _ = py_call(fields.pop, "feature_names_in_", PythonObject(None))
    _ = raw^
    _ = pair^
    return model


def _kde_fit_mode(model: PythonObject, fields: PythonObject, context: PythonObject) raises -> PythonObject:
    # ModeMixin preserves an inherited numeric_mode without creating an
    # instance field. Match native attribute precedence, including a missing
    # attribute's None default, without invoking Python descriptor hooks.
    var present = py_contains(fields, "numeric_mode")
    var mro = context["classical_public_layouts"][py_type(model)]["mro"]
    for i in range(len(mro)):  # driver-loop: connected native class metadata
        present = present or py_contains(_pd_type_fields(mro[i]), "numeric_mode")
    if present:
        return classical_getattribute(model, PythonObject("numeric_mode"), context)
    return PythonObject(None)


def _kde_fit_store(model: PythonObject, fields: PythonObject, raw: PythonObject, weight: PythonObject, copied: Bool, est: String, context: PythonObject) raises:
    # The CPU owner has no resident registry or immutable-fit experiment.
    # Refit clears an existing cache before preparation, like the old method.
    var b = Python.import_module("builtins")
    if py_get(fields, "_resident") is not None:
        fields["_resident"] = PythonObject(None)
    var x = _kde_cast(raw, context, "X")
    fields["input_copied_"] = PythonObject(copied)
    try:
        host_inference_finite(Int(py=classical_address(x)), Int(py=x.size))
    except:
        raise Error("mojolearn KernelDensity: X contains a NaN or an infinity; refused by name at fit (DEVIATION 604)")
    var mode = _kde_fit_mode(model, fields, context)
    if mode is not None and py_type(mode) is not b.str:
        raise Error("TypeError: KDE numeric mode requires an exact native string")
    if mode is not None and String(mode) != "identical":
        raise Error("mojolearn: " + est + " runs IDENTICAL only on the host; this model was saved '" + String(mode) + "'")
    fields["_x"] = x
    fields["n_features_in_"] = x.shape[1]
    fields["n_samples_fit_"] = x.shape[0]
    fields["_w"] = _kde_weights(weight, Int(py=x.shape[0]), context)
    _ = x^
