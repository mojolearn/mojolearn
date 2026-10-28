# SPDX-License-Identifier: Apache-2.0
# Public lifecycle for the four decomp manifold estimators and TSNE. UMAP's
# supervised target owner is separate; this envelope validates ignored y.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.native_loaded_module import native_loaded_module
from bindings.manifold_settings_native import ManifoldSettings
from bindings.classical_attributes import classical_connect_numpy
from bindings.classical_validate import classical_view, classical_refuse_sparse, RESHAPE_HINT
from bindings.classical_pandas import classical_admit_value, native_pandas_exact
from bindings.classical_exceptions import classical_not_fitted_class
from bindings.classical_warnings import _warning_category, COLUMN_WARNING
from bindings.classifier_public_lifecycle import _feature_names
from bindings.metrics_native_support import metric_warn
from bindings.sequential_admission import _one_of, _ordinary_torch
from bindings.host_sparse_native import host_sparse_format, host_sparse_dense, _sparse_fields
from bindings.family_common import FAMILY_F32P, family_array, family_repr
from bindings.classical_arrays import classical_address
from bindings.parallel_public_result import parallel_public_result
from providers.python.array_metadata_pyapi import ArrayMetadata, array_shape_once, array_metadata_once
from core.cast_in import cast_in_kind
from core.family_cast_impl import family_cast_rows


def manifold_method(operation: String) -> String:
    if operation.endswith("_fit_transform"):
        return String("fit_transform")
    if operation.endswith("_fit"):
        return String("fit")
    if operation.endswith("_reconstruction_error"):
        return String("reconstruction_error")
    return String("transform")


def _fitted(fields: PythonObject) raises -> Bool:
    var b = native_loaded_module("builtins")
    for name in ["_mojolearn_fitted", "_mojolearn_fitting"]:  # small-loop(fitted flags: two native metadata fields)
        var value = py_get(fields, name)
        if value is not None:
            if py_type(value) is not b.bool and py_type(value) is not b.int:
                raise Error("TypeError: manifold fitted flags require native booleans or integers")
            if Bool(py=value):
                return True
    var reset = py_get(fields, "_mojolearn_params_reset")
    var names: PythonObject
    if reset is not None:
        if py_type(reset) is not b.tuple:
            raise Error("TypeError: native parameter reset metadata requires an ordinary tuple")
        names = reset
    else:
        names = py_call(b.list, fields)
    for i in range(len(names)):  # driver-loop: fitted-field metadata, never value protocols
        if py_type(names[i]) is not b.str:
            raise Error("TypeError: native parameter reset metadata requires exact string names")
        var name = String(names[i])
        if name.endswith("_") and not name.startswith("_") and py_contains(fields, names[i]) and fields[names[i]] is not None:
            return True
    return False


def manifold_require_fitted(model: PythonObject, settings: ManifoldSettings,
                            method: String, context: PythonObject) raises:
    if method != "transform" or _fitted(settings.values):
        return
    _ = classical_not_fitted_class(context)
    raise Error("NotFittedError: This " + String(py_type(model).__name__) + " instance is not fitted yet. Call 'fit' with appropriate arguments before using " + method + ".")


def _numeric_dtype(code: Int) -> Bool:
    return (code >= 0 and code <= 13) or code == 23


def _complex_dtype(code: Int) -> Bool:
    return code == 14 or code == 15 or code == 16


def _cast(var snapshot: ArrayMetadata, name: String, context: PythonObject) raises -> ArrayMetadata:
    # Unsupported metadata is descriptive only. This guard dominates every
    # CastIn-based conversion and replace_f32 call. The ignored-target byte
    # scan below separately admits its supported dtype/width and arbitrary rank.
    if not snapshot.native_byte_order:
        raise Error("TypeError: " + name + " must be in native byte order")
    if not snapshot.castable:
        # Reproduce the former cast's dtype/itemsize refusal before attempting
        # any numerical read. A valid dtype with unsupported rank was refused
        # by the public shape checks above.
        _ = cast_in_kind(snapshot.dtype_number, snapshot.source.itemsize, name)
        raise Error(name + " metadata requires an admitted scalar/vector/matrix")
    if not snapshot.f32_c():
        var output = family_array(snapshot.python_shape(), "<f4", context)
        var address = Int(py=classical_address(output))
        var src = snapshot.source
        family_cast_rows(src.addr, src.s0, src.s1, src.rows, src.cols, src.itemsize, src.kind,
                         FAMILY_F32P(unsafe_from_address=address))
        snapshot.replace_f32(output, address)
    return snapshot^


def _target_finite(snapshot: ArrayMetadata) raises -> Bool:
    # The previous ignored-target check scans only float32/float64, retaining
    # finite float64 values outside float32 range. This native strided scan
    # supports arbitrary rank; it does not execute the matrix-only CastIn.
    var width = snapshot.source.itemsize
    if snapshot.dtype_number != 11 and snapshot.dtype_number != 12:
        return True
    if (snapshot.dtype_number == 11 and width != 4) or (snapshot.dtype_number == 12 and width != 8):
        raise Error("TypeError: target floating dtype has inconsistent itemsize")
    if len(snapshot.strides) != len(snapshot.shape):
        raise Error("TypeError: target strides must match its captured shape")
    var pointer = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=snapshot.source.addr)
    var mask = UInt64(0x7F800000) if width == 4 else UInt64(0x7FF0000000000000)
    for linear in range(snapshot.count):  # cpu-route: ignored-target finiteness over captured native strides
        var remainder = linear
        var offset = 0
        for k in range(len(snapshot.shape)):  # small-loop(rank: target metadata axes)
            var axis = len(snapshot.shape) - 1 - k
            offset += (remainder % snapshot.shape[axis]) * snapshot.strides[axis]
            remainder //= snapshot.shape[axis]
        var bits = UInt64(0)
        for byte in range(width):  # small-loop(itemsize: four or eight native floating bytes)
            var index = byte if snapshot.native_byte_order else width - 1 - byte
            bits |= UInt64(pointer[offset + index]) << UInt64(8 * byte)
        if (bits & mask) == mask:
            return False
    return True


def manifold_input(source: PythonObject, name: String, sparse_allowed: Bool,
                   context: PythonObject) raises -> ArrayMetadata:
    var prepared = source
    if sparse_allowed:
        var sparse = host_sparse_format(source)
        if sparse.byte_length() > 0:
            if not py_contains(context, "output_context"):
                py_setitem(context, "output_context", context)
            prepared = host_sparse_dense(source, sparse, context)
    else:
        classical_refuse_sparse(source, name)
    var view = classical_view(prepared, context, "X")
    var shape = array_shape_once(view, context)
    var snapshot = array_metadata_once(view, shape^, "X", context,
                                      allow_non_native=True, allow_unsupported=True)
    var nd = len(snapshot.shape)
    var text = family_repr(snapshot.python_shape())
    if _complex_dtype(snapshot.dtype_number):
        raise Error("Complex data not supported: " + name + " got a complex X.")
    if _numeric_dtype(snapshot.dtype_number):
        if nd < 2:
            raise Error("Expected 2D array, got " + ("scalar" if nd == 0 else "1D") + " array instead (shape=" + text + "). " + RESHAPE_HINT)
        if nd == 2 and snapshot.shape[0] == 0:
            raise Error("Found array with 0 sample(s) (shape=" + text + ") while a minimum of 1 is required by " + name + ".")
        if nd == 2 and snapshot.shape[1] == 0:
            raise Error("Found array with 0 feature(s) (shape=" + text + ") while a minimum of 1 is required by " + name + ".")
    if nd != 2:
        var hint = String(". ") + RESHAPE_HINT if nd < 2 else String("")
        raise Error("mojolearn: X must be 2-D, got " + String(nd) + "-D shape " + text + hint)
    var result = _cast(snapshot, "X", context)
    _ = view
    _ = prepared
    _ = source
    return result^


def _target_leaves(value: PythonObject, context: PythonObject, depth: Int = 0) raises:
    if depth > 64:
        raise Error("TypeError: manifold target nesting exceeds the supported depth")
    var b = native_loaded_module("builtins")
    var cls = py_type(value)
    if cls is b.list or cls is b.tuple:
        for i in range(len(value)):  # driver-loop: ignored target references, never conversion hooks
            _target_leaves(value[i], context, depth + 1)
        return
    if cls is b.complex or cls is b.bytes:
        return
    classical_admit_value(value, context)


def _target_list_kind(value: PythonObject, context: PythonObject) raises -> Int:
    # The historical validator describes a list's first leaf; an integer-first
    # or nonnumeric-first ignored y does not acquire a new finite-value scan.
    var b = native_loaded_module("builtins")
    var item = value
    while py_type(item) is b.list or py_type(item) is b.tuple:  # driver-loop: captured native sequence rank
        if len(item) == 0:
            return 1
        item = item[0]
    var cls = py_type(item)
    if cls is b.complex:
        return 3
    if cls is b.float:
        return 2
    if cls is b.int or cls is b.bool:
        return 1
    if _one_of(cls, context["scalar_types"]):
        var kind = String(item.dtype.kind)
        if kind == "c":
            return 3
        if kind == "f":
            return 2
        if kind == "i" or kind == "u" or kind == "b":
            return 1
    return 0


def manifold_ignored_target(y: PythonObject, rows: Int, name: String, context: PythonObject) raises:
    if y is None:
        return
    var b = native_loaded_module("builtins")
    var cls = py_type(y)
    if cls is b.int or cls is b.float or cls is b.bool or cls is b.complex or cls is b.str or cls is b.bytes or cls is b.dict:
        # These exact scalar/non-tabular inputs have no shape in the existing
        # ignored-target contract. No conversion or length protocol is needed.
        return
    if cls is b.list or cls is b.tuple:
        _target_leaves(y, context)
        var kind = _target_list_kind(y, context)
        if kind == 3:
            raise Error("Complex data not supported: " + name + " got a complex y.")
        if kind == 2:
            var view = classical_view(y, context, "y")
            var shape = array_shape_once(view, context)
            var captured = array_metadata_once(view, shape^, "y", context,
                                                allow_non_native=True, allow_unsupported=True)
            if not _target_finite(captured):
                raise Error("Input y contains NaN or infinity (" + name + ").")
        if len(y) != rows:
            raise Error("Found input variables with inconsistent numbers of samples: [" + String(rows) + ", " + String(len(y)) + "]")
        return
    # Ignored sparse y has only length semantics; never densify its labels.
    var sparse = host_sparse_format(y)
    if sparse.byte_length() > 0:
        var shape = py_get(_sparse_fields(y), "_shape")
        if py_type(shape) is not b.tuple or len(shape) != 2 or py_type(shape[0]) is not b.int or py_type(shape[1]) is not b.int:
            raise Error("TypeError: sparse target storage requires a native two-dimensional shape")
        if Int(py=shape[0]) != rows:
            raise Error("Found input variables with inconsistent numbers of samples: [" + String(rows) + ", " + String(Int(py=shape[0])) + "]")
        return
    var view = classical_view(y, context, "y")
    var shape = array_shape_once(view, context)
    var snapshot = array_metadata_once(view, shape^, "y", context,
                                      allow_non_native=True, allow_unsupported=True)
    if _complex_dtype(snapshot.dtype_number):
        raise Error("Complex data not supported: " + name + " got a complex y.")
    var reshape_supported = cls is py_get(context, "ndarray") or cls is context["array_class"] or native_pandas_exact(y) or _ordinary_torch(cls, context)
    if len(snapshot.shape) == 2 and snapshot.shape[1] == 1 and reshape_supported:
        metric_warn(COLUMN_WARNING, _warning_category(context), context)
    if not _target_finite(snapshot):
        raise Error("Input y contains NaN or infinity (" + name + ").")
    if len(snapshot.shape) > 0 and snapshot.shape[0] != rows:
        raise Error("Found input variables with inconsistent numbers of samples: [" + String(rows) + ", " + String(snapshot.shape[0]) + "]")
    _ = view
    _ = y


def manifold_check_width(settings: ManifoldSettings, x: ArrayMetadata, names: PythonObject,
                         name: String, context: PythonObject) raises:
    var b = native_loaded_module("builtins")
    var metric = py_get(settings.values, "metric")
    if py_type(metric) is b.str and String(metric) == "precomputed":
        return
    var width = py_get(settings.values, "n_features_in_")
    if py_type(width) is b.int and x.source.cols != Int(py=width):
        raise Error("X has " + String(x.source.cols) + " features, but " + name + " is expecting " + String(Int(py=width)) + " features as input.")
    var fitted = py_get(settings.values, "feature_names_in_")
    if fitted is None:
        if names is not None and (py_type(width) is b.int or py_type(width) is b.bool):
            metric_warn("X has feature names, but " + name + " was fitted without feature names", b.UserWarning, context)
        return
    if names is None:
        metric_warn("X does not have valid feature names, but " + name + " was fitted with feature names", b.UserWarning, context)
        return
    classical_admit_value(fitted, context)
    var previous = py_call(b.list, fitted)
    for i in range(len(previous)):  # driver-loop: feature-name identity/type admission before equality
        if py_type(previous[i]) is not b.str:
            raise Error("TypeError: fitted feature names require exact strings")
    if previous != py_call(b.list, names):
        raise Error("The feature names should match those that were passed during fit.")


def manifold_finish(settings: ManifoldSettings, result: PythonObject, method: String,
                    names: PythonObject, context: PythonObject) raises -> PythonObject:
    if method == "fit" or method == "fit_transform":
        if names is None:
            _ = py_call(settings.fields.pop, "feature_names_in_", PythonObject(None))
        else:
            py_setitem(settings.fields, "feature_names_in_", names)
        py_setitem(settings.fields, "_mojolearn_fitted", PythonObject(True))
    if method == "fit":
        return result
    return parallel_public_result(result, method, PythonObject(None), context)
