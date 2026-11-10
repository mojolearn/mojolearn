# SPDX-License-Identifier: Apache-2.0
# Explicit CPU logistic decisions, labels and probability links retain their
# established numerical entries; every surrounding operation is native.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get
from core.logistic_native_api import QnQueryRequest, QN_NATIVE_BINARY, QN_NATIVE_DECISION, QN_NATIVE_PROBA
from bindings.logistic_descriptor_boundary import logistic_run_query, logistic_run_log
from bindings.classical_arrays import classical_input, classical_address, classical_is_array, classical_kind
from bindings.classical_pandas import classical_admit_value, native_pandas_exact, _pd_type_fields
from bindings.sequential_admission import _one_of, _ordinary_torch
from bindings.native_torch_admission import native_torch_numpy
from bindings.native_numpy_metadata import native_numpy_dtype
from bindings.host_archive_native import host_archive_typed, host_archive_class_values
from bindings.host_tree_native import host_tree_input
from bindings.host_kernel_inference_native import _kernel_buffer
from bindings.host_inference_features import host_inference_feature_names
from bindings.host_inference_preparation import host_inference_pandas_finite
from core.host_inference_impl import host_inference_raw_finite
from core.host_tree_impl import host_tree_argmax, host_label_codes_in_range, host_label_first_unconverted, host_label_gather_i64, host_label_gather_f64
from core.host_sparse_impl import host_sparse_uint64_bound
from bindings.p3_linear_helpers import py_list_select


struct _LabelAliases(Movable):
    # The eight native scalar alias classes (np.dtype(code).type), built at
    # most once per class-table read instead of once per label.
    var types: List[PythonObject]

    def __init__(out self):
        self.types = List[PythonObject]()

    def fill(mut self, context: PythonObject) raises:
        if len(self.types) == 0:
            var aliases = Python.tuple("q", "Q", "g", "U", "S", "F", "D", "G")
            var constructor = native_numpy_dtype(context)
            for i in range(len(aliases)):  # driver-loop: exact native scalar aliases
                self.types.append(py_call(constructor, aliases[i]).type)


def _logistic_label_kind(value: PythonObject, context: PythonObject) raises -> Int:
    var aliases = _LabelAliases()
    return _logistic_label_kind(value, context, aliases)


def _logistic_label_kind(value: PythonObject, context: PythonObject, mut alias_types: _LabelAliases) raises -> Int:
    # 1=integer, 2=real, 0=object labels. Only exact native scalar slots can
    # enter a numeric conversion; object labels are returned by reference.
    var b = native_loaded_module("builtins")
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
        alias_types.fill(context)
        for i in range(len(alias_types.types)):  # driver-loop: exact native scalar aliases
            if cls is alias_types.types[i]:
                return 1 if i < 2 else (2 if i == 2 else 0)
    classical_admit_value(value, context)
    return 0


def _logistic_label_dtype(dtype: PythonObject, context: PythonObject) raises:
    var b = native_loaded_module("builtins")
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
    var b = native_loaded_module("builtins")
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
    var alias_types = _LabelAliases()
    for i in range(len(labels)):  # driver-loop: scalar label admission before numeric slots
        _ = _logistic_label_kind(labels[i], context, alias_types)
    return labels


def _host_logistic_selected(labels: PythonObject, indices: List[Int]) raises -> PythonObject:
    """The list of selected label references (object-label result): one
    PyList_New and direct slot stores of the k stored class references."""
    return py_list_select(labels, indices)


def _host_logistic_decode(labels: PythonObject, indices: List[Int], array_labels: Bool, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var integers = True
    var numbers = True
    var alias_types = _LabelAliases()
    for i in range(len(labels)):  # driver-loop: distinct label result-type metadata
        var kind = _logistic_label_kind(labels[i], context, alias_types)
        integers = integers and kind == 1
        numbers = numbers and kind != 0
    var k = len(labels)
    if not host_label_codes_in_range(indices, k):
        raise Error("predicted class index exceeds the fitted class table")
    if not numbers:
        return _host_logistic_selected(labels, indices)
    if array_labels:
        # The established decode's numeric gather tests `not classes`.
        # Multi-element ndarrays reject that truth conversion; preserve the
        # existing refusal without invoking a NumPy or element truth slot.
        raise Error("The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()")
    # Each class is converted once, not once per predicted row: the k words
    # and whether each converts are exactly what the per-row conversion of a
    # selected reference produced (same int()/float() of the same object), so
    # the result and its fallback are unchanged; rows are gathered natively.
    var words_i = List[Int64](length=k, fill=0)
    var words_f = List[Float64](length=k, fill=0.0)
    var fits = List[Bool](length=k, fill=True)
    for i in range(k):  # driver-loop: over the fitted classes (model structure)
        try:
            if integers:
                var value = py_call(b.int, labels[i])
                if Bool(py=value < PythonObject(-9223372036854775807 - 1)) or Bool(py=value > PythonObject(9223372036854775807)):
                    fits[i] = False
                else:
                    words_i[i] = Int64(Int(py=value))
            else:
                words_f[i] = Float64(py=labels[i])
        except:
            # All conversions above have exact native scalar admission. This is
            # only the established numeric-overflow result contract, no callback.
            fits[i] = False
    # The old decode returns an Array when the selected labels fit, even if
    # an unselected table label does not. Otherwise it returns the label list.
    if host_label_first_unconverted(indices, fits) >= 0:
        return _host_logistic_selected(labels, indices)
    var out = host_archive_typed(Python.tuple(len(indices)), "<i8" if integers else "<f8", context["output_context"])
    if integers:
        host_label_gather_i64(indices, words_i, Int(py=classical_address(out)))
    else:
        host_label_gather_f64(indices, words_f, Int(py=classical_address(out)))
    _ = out  # keep the Python object alive through the native gather above
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
        # The native decoder already consumes machine-word codes. The QN
        # ABI writes int64, the same word on every supported LP64 target;
        # write directly into that owned native list instead of allocating
        # a Python Array and copying each completed code into the list.
        indices = List[Int](length=rows, fill=0)
        logistic_run_query(context["host_inference_entries"], QN_NATIVE_BINARY,
            QnQueryRequest(Int(py=classical_address(x)), Int(py=classical_address(weights)),
                Int(indices.unsafe_ptr()), 0, rows, features, 1, intercept, False))
    elif operation == "host_infer_predict_proba" or operation == "host_infer_predict_log_proba":
        # Decision and link in one native call (no Python scores array).
        var probabilities = host_archive_typed(Python.tuple(rows, count), "<f8", context["output_context"])
        logistic_run_query(context["host_inference_entries"], QN_NATIVE_PROBA,
            QnQueryRequest(Int(py=classical_address(x)), Int(py=classical_address(weights)),
                Int(py=classical_address(probabilities)), 0, rows, features, targets, intercept, False))
        if operation == "host_infer_predict_log_proba":
            out = host_archive_typed(Python.tuple(rows, count), "<f8", context["output_context"])
            logistic_run_log(context["host_inference_entries"], Int(py=classical_address(probabilities)),
                Int(py=classical_address(out)), rows * count)
        else:
            out = probabilities
        _ = probabilities^
    else:
        scores = host_archive_typed(Python.tuple(rows) if targets == 1 else Python.tuple(rows, targets), "<f4", context["output_context"])
        logistic_run_query(context["host_inference_entries"], QN_NATIVE_DECISION,
            QnQueryRequest(Int(py=classical_address(x)), Int(py=classical_address(weights)),
                Int(py=classical_address(scores)), 0, rows, features, targets, intercept, False))
        if operation == "host_infer_predict":
            indices = host_tree_argmax(Int(py=classical_address(scores)), rows, targets)
        elif operation == "host_infer_decision_function":
            out = scores
    # Raw addresses do not keep converted input/weight owners alive through
    # the synchronous descriptor call (which releases the GIL).
    _ = x^
    _ = weights^
    _ = scores^
    if operation == "host_infer_predict":
        out = _host_logistic_decode(labels, indices, py_type(fields["classes_"]) is py_get(context, "ndarray"), context)
    var b = native_loaded_module("builtins")
    return classical_input(out, context, "logistic result") if py_type(out) is not b.list and public and py_get(context, "ndarray") is not None else out
