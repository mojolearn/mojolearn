# SPDX-License-Identifier: Apache-2.0
# Explicit CPU logistic decisions, labels and probability links retain their
# established numerical entries; every surrounding operation is native.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get
from bindings.classical_arrays import classical_input, classical_address, classical_is_array, classical_kind
from bindings.classical_pandas import classical_admit_value, native_pandas_exact, _pd_type_fields
from bindings.sequential_admission import _one_of, _ordinary_torch
from bindings.native_torch_admission import native_torch_numpy
from bindings.native_numpy_metadata import native_numpy_dtype
from bindings.host_archive_native import host_archive_typed, host_archive_class_values
from bindings.host_tree_native import host_tree_input
from bindings.host_kernel_inference_native import _kernel_entry, _kernel_buffer
from bindings.host_inference_features import host_inference_feature_names
from bindings.host_inference_preparation import host_inference_pandas_finite
from core.host_inference_impl import host_inference_raw_finite
from core.host_tree_impl import host_tree_argmax
from core.host_sparse_impl import host_sparse_uint64_bound


def _logistic_label_kind(value: PythonObject, context: PythonObject) raises -> Int:
    # 1=integer, 2=real, 0=object labels. Only exact native scalar slots can
    # enter a numeric conversion; object labels are returned by reference.
    var b = Python.import_module("builtins")
    var cls = py_type(value)
    if cls is b.int:
        return 1
    if cls is b.float:
        return 2
    if _one_of(cls, Python.tuple(b.bool, b.str, b.bytes, b.complex, py_type(PythonObject(None)))):
        return 0
    if _one_of(cls, context["scalar_types"]):
        var kind = String(value.dtype.kind)
        return 1 if kind == "i" or kind == "u" else (2 if kind == "f" else 0)
    if py_get(context, "ndarray") is not None:
        var aliases = Python.tuple("q", "Q", "g", "U", "S", "F", "D", "G")
        var constructor = native_numpy_dtype(context)
        for i in range(len(aliases)):  # driver-loop: exact native scalar aliases
            if cls is py_call(constructor, aliases[i]).type:
                return 1 if i < 2 else (2 if i == 2 else 0)
    classical_admit_value(value, context)
    return 0


def _logistic_label_dtype(dtype: PythonObject, context: PythonObject) raises:
    var b = Python.import_module("builtins")
    var types = py_call(b.list, context["numeric_dtypes"].values())
    for i in range(len(types)):  # driver-loop: primitive native dtype identities before scalar reads
        if py_type(dtype) is py_type(types[i]):
            return
    var aliases = Python.tuple("q", "Q", "g", "U", "S", "O", "F", "D", "G")
    var constructor = native_numpy_dtype(context)
    for i in range(len(aliases)):  # driver-loop: native class-label storage dtypes
        if py_type(dtype) is py_type(py_call(constructor, aliases[i])):
            return
    raise Error("TypeError: class labels require native numeric, complex, string or object storage")


def host_logistic_classes(source: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var cls = py_type(source)
    var labels = Python.list()
    if cls is py_get(context, "host_inference_class_labels"):
        # Fit's ClassLabels has Python indexing helpers. Copy its native list
        # payload through list's C descriptor, never those helpers or iterators.
        var mro = py_call(b.type.__dict__["__mro__"].__get__, cls)
        for i in range(len(mro)):  # driver-loop: connected class-label lifetime admission
            if py_get(_pd_type_fields(mro[i]), "__del__") is not None:
                raise Error("TypeError: class-label storage cannot use Python finalizers")
        labels = py_call(b.list.__dict__["copy"], source)
    elif cls is b.list or cls is b.tuple:
        labels = py_call(b.list, source)
    elif cls is context["array_class"]:
        if Int(py=source.ndim) != 1:
            raise Error("class labels require one-dimensional native storage")
        labels = host_archive_class_values(source, context)
    elif cls is py_get(context, "ndarray"):
        _logistic_label_dtype(source.dtype, context)
        if Int(py=source.ndim) != 1:
            raise Error("class labels require one-dimensional native storage")
        for i in range(Int(py=source.size)):  # driver-loop: distinct native label storage
            # Exact ndarray indexing with a primitive dtype is a C slot.
            # Object elements are admitted below before any conversion.
            _ = py_call(labels.append, source[i])
    else:
        raise Error("TypeError: class labels require native list or array storage without Python protocols")
    for i in range(len(labels)):  # driver-loop: scalar label admission before numeric slots
        _ = _logistic_label_kind(labels[i], context)
    return labels


def _logistic_decode(labels: PythonObject, indices: List[Int], array_labels: Bool, context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var integers = True
    var numbers = True
    for i in range(len(labels)):  # driver-loop: distinct label result-type metadata
        var kind = _logistic_label_kind(labels[i], context)
        integers = integers and kind == 1
        numbers = numbers and kind != 0
    var selected = Python.list()
    for i in range(len(indices)):  # driver-loop: native publication of selected label references
        if indices[i] < 0 or indices[i] >= len(labels):
            raise Error("predicted class index exceeds the fitted class table")
        _ = py_call(selected.append, labels[indices[i]])
    if not numbers:
        return selected
    if array_labels:
        # The established decode's numeric gather tests `not classes`.
        # Multi-element ndarrays reject that truth conversion; preserve the
        # existing refusal without invoking a NumPy or element truth slot.
        raise Error("The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()")
    var out = host_archive_typed(Python.tuple(len(indices)), "<i8" if integers else "<f8", context["output_context"])
    # The old decode returns an Array when the selected labels fit, even if
    # an unselected table label does not. Otherwise it returns the label list.
    try:
        for i in range(len(indices)):  # driver-loop: completed native label result publication
            if integers:
                var value = py_call(b.int, selected[i])
                if Bool(py=value < PythonObject(-9223372036854775807 - 1)) or Bool(py=value > PythonObject(9223372036854775807)):
                    return selected
                MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(out)))[i] = Int64(Int(py=value))
            else:
                MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(out)))[i] = Float64(py=selected[i])
    except:
        # All conversions above have exact native scalar admission. This is
        # only the established numeric-overflow result contract, no callback.
        return selected
    return out


def host_logistic_inference(model: PythonObject, fields: PythonObject, labels: PythonObject, operation: String, source: PythonObject, context: PythonObject, public: Bool) raises -> PythonObject:
    var fitting = Bool(py=py_get(fields, "_mojolearn_fitting", False))
    classical_admit_value(source, context)
    var prepared = native_torch_numpy(source) if _ordinary_torch(py_type(source), context) else source
    if native_pandas_exact(prepared):
        if not fitting:
            host_inference_pandas_finite(prepared, context)
    else:
        var raw = classical_input(prepared, context, "X")
        if classical_kind(raw, context) == "c":
            raise Error("Complex data not supported")
        var width = Int(py=raw.itemsize)
        if Int(py=raw.ndim) == 2 and classical_kind(raw, context) == "u" and width == 8:
            host_sparse_uint64_bound(Int(py=classical_address(raw)), Int(py=raw.shape[0]), Int(py=raw.shape[1]),
                Int(py=raw.strides[0]), Int(py=raw.strides[1]), classical_is_array(raw, context) or Bool(py=raw.dtype.isnative))
        if not fitting and Int(py=raw.ndim) == 2 and classical_kind(raw, context) == "f" and (width == 4 or width == 8):
            host_inference_raw_finite(Int(py=classical_address(raw)), Int(py=raw.strides[0]), Int(py=raw.strides[1]),
                Int(py=raw.shape[0]), Int(py=raw.shape[1]), width,
                classical_is_array(raw, context) or Bool(py=raw.dtype.isnative))
        _ = raw^
    var x = host_tree_input(prepared, context)
    _ = prepared^
    var rows = Int(py=x.shape[0])
    var features = Int(py=x.shape[1])
    if features != Int(py=fields["n_features_in_"]):
        raise Error("LogisticRegression feature count differs from fit")
    var count = len(labels)
    if count < 2:
        raise Error("LogisticRegression requires at least two fitted classes")
    var targets = 1 if count == 2 else count
    if not fitting:
        # Preserve warnings from the old nested public method chain. The
        # original CPU multiclass predict called decision_function; proba
        # did likewise and log_proba called both. No Python method is called.
        var checks = 1
        if (operation == "host_infer_predict" and targets > 1) or operation == "host_infer_predict_proba":
            checks = 2
        elif operation == "host_infer_predict_log_proba":
            checks = 3
        for i in range(checks):  # driver-loop: fixed original public validation stages
            host_inference_feature_names(model, fields, context)
    var intercept = Bool(py=fields["fit_intercept"])
    var dims = features + Int(intercept)
    if targets > 2147483647 // dims or rows > 2147483647 // count:
        raise Error("LogisticRegression dimensions exceed native indexing")
    var weights = _kernel_buffer(fields["_w"], dims * targets, "<f4", context)
    var scores = PythonObject(None)
    var out = PythonObject(None)
    var indices = List[Int]()
    if operation == "host_infer_predict" and targets == 1:
        if py_get(context["host_inference_entries"], "qn_predict_binary") is None:
            raise Error("RuntimeError: LogisticRegression requires qn_predict_binary; rebuild the estimators binding")
        var codes = host_archive_typed(Python.tuple(rows), "<i8", context["output_context"])
        _ = py_call(_kernel_entry("qn_predict_binary", context), classical_address(x), classical_address(weights),
            classical_address(codes), Python.list(rows, features, Int(intercept)))
        var values = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(codes)))
        for i in range(rows):  # driver-loop: native completed class-code publication
            indices.append(Int(values[i]))
        _ = codes^
    else:
        scores = host_archive_typed(Python.tuple(rows) if targets == 1 else Python.tuple(rows, targets), "<f4", context["output_context"])
        _ = py_call(_kernel_entry("qn_decision_function", context), classical_address(x), classical_address(weights),
            classical_address(scores), Python.list(rows, features, Int(intercept), targets))
        if operation == "host_infer_predict":
            indices = host_tree_argmax(Int(py=classical_address(scores)), rows, targets)
        elif operation == "host_infer_decision_function":
            out = scores
        else:
            var probabilities = host_archive_typed(Python.tuple(rows, count), "<f8", context["output_context"])
            _ = py_call(_kernel_entry("qn_sigmoid" if targets == 1 else "qn_softmax", context), classical_address(scores),
                classical_address(probabilities), Python.list(rows) if targets == 1 else Python.list(rows, count))
            if operation == "host_infer_predict_log_proba":
                var flags = 0
                if py_get(context["host_inference_entries"], "py2mojo_linear_flags") is not None:
                    flags = Int(py=py_call(_kernel_entry("py2mojo_linear_flags", context)))
                if (flags & 2) == 0:
                    raise Error("RuntimeError: this estimators binding predates py2mojo_rows; rebuild it")
                out = host_archive_typed(Python.tuple(rows, count), "<f8", context["output_context"])
                _ = py_call(_kernel_entry("py2mojo_rows", context), 1, classical_address(probabilities), classical_address(out), Python.list(rows * count, 1))
            else:
                out = probabilities
            _ = probabilities^
    _ = x^
    _ = weights^
    _ = scores^
    if operation == "host_infer_predict":
        out = _logistic_decode(labels, indices, py_type(fields["classes_"]) is py_get(context, "ndarray"), context)
    var b = Python.import_module("builtins")
    return classical_input(out, context, "logistic result") if py_type(out) is not b.list and public and py_get(context, "ndarray") is not None else out
