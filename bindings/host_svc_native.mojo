# SPDX-License-Identifier: Apache-2.0
# Explicit CPU SVC owns its binary/pair-machine sequence, calibration,
# class decoding and computed attributes. Existing numerical entries are reused.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_fields
from bindings.native_error_class import error_without_class
from bindings.classical_attributes import classical_check_layout, classical_getattribute, classical_connect_numpy
from bindings.classical_arrays import classical_input, classical_address, classical_is_array, classical_kind, classical_cast_kind
from bindings.classical_pandas import classical_admit_value, native_pandas_exact, _pd_type_fields
from bindings.sequential_admission import _ordinary_torch
from bindings.native_torch_admission import native_torch_numpy
from bindings.host_archive_native import host_archive_typed
from bindings.host_tree_native import host_tree_input, host_tree_constructor_fields
from bindings.host_sparse_native import host_sparse_format, host_sparse_dense
from bindings.host_inference_features import host_inference_feature_names
from bindings.host_inference_scalars import host_inference_scalar
from bindings.host_kernel_inference_native import _kernel_entry, _kernel_buffer
from bindings.host_logistic_native import host_logistic_classes, _logistic_decode
from bindings.host_svr_native import _svr_matrix, _svr_support, _svr_bias
from bindings.scaler_lifecycle_native import _outside
from core.host_sparse_impl import host_sparse_uint64_bound
from core.host_helpers_impl import argmax_rows_f64
from core.host_svr_impl import host_svr_indices, host_svr_bias


def _svc_numeric_storage(value: PythonObject, context: PythonObject) raises:
    if py_get(context, "ndarray") is not None and py_type(value) is context["ndarray"]:
        if classical_kind(value, context) == "O":
            raise Error("TypeError: SVC numeric state cannot contain Python object storage")


def host_svc_admit_state(fields: PythonObject, context: PythonObject) raises:
    var b = Python.import_module("builtins")
    _svc_numeric_storage(py_get(fields, "dual_coef_"), context)
    _svc_numeric_storage(py_get(fields, "support_vectors_"), context)
    var names = Python.tuple("n_features_in_", "_gamma", "_hp_cache_size", "_hp_degree", "_hp_coef0", "_label0", "_label1")
    for i in range(len(names)):  # driver-loop: fixed native SVC scalar controls
        if py_contains(fields, names[i]):
            host_inference_scalar(fields[names[i]], context)
    var flags = Python.tuple("probability", "_hp_probability", "_hp_break_ties", "break_ties")
    for i in range(len(flags)):  # driver-loop: native bool-compatible controls before truth slots
        if py_contains(fields, flags[i]):
            host_inference_scalar(fields[flags[i]], context, True)
    var strings = Python.tuple("_hp_kernel", "decision_function_shape")
    for i in range(len(strings)):  # driver-loop: fixed string controls
        if py_contains(fields, strings[i]) and py_type(fields[strings[i]]) is not b.str:
            raise Error("TypeError: saved SVC kernel and decision shape require exact native strings")
    var pairs = py_get(fields, "_pairs")
    if pairs is None:
        if py_contains(fields, "n_support_"):
            host_inference_scalar(fields["n_support_"], context)
        return
    if py_type(pairs) is not b.list and py_type(pairs) is not b.tuple:
        raise Error("TypeError: SVC pair state requires native list or tuple storage")
    for i in range(len(pairs)):  # driver-loop: native pair scalar admission before launch preparation
        var pair = pairs[i]
        if py_type(pair) is not b.dict:
            raise Error("TypeError: SVC pair state requires native dictionaries")
        _svc_numeric_storage(pair["dual"], context)
        _svc_numeric_storage(pair["sv"], context)
        host_inference_scalar(pair["i"], context)
        host_inference_scalar(pair["j"], context)
        host_inference_scalar(pair["b"], context)


def host_svc_probability(model: PythonObject, fields: PythonObject, name: PythonObject, context: PythonObject) raises:
    var probability = py_get(fields, "probability", False)
    host_inference_scalar(probability, context, True)
    if not Bool(py=probability):
        var b = Python.import_module("builtins")
        var cls_name = String(py_call(b.type.__dict__["__name__"].__get__, py_type(model)))
        raise Error("AttributeError: This '" + cls_name + "' has no attribute '" + String(name) + "'")


def _svc_input(source: PythonObject, context: PythonObject) raises -> PythonObject:
    var format = host_sparse_format(source)
    if format.byte_length() > 0:
        return host_sparse_dense(source, format, context)
    classical_admit_value(source, context)
    var prepared = native_torch_numpy(source) if _ordinary_torch(py_type(source), context) else source
    if not native_pandas_exact(prepared):
        var raw = classical_input(prepared, context, "X")
        if classical_kind(raw, context) == "c":
            raise Error("Complex data not supported")
        if Int(py=raw.ndim) == 2 and classical_kind(raw, context) == "u" and Int(py=raw.itemsize) == 8:
            host_sparse_uint64_bound(Int(py=classical_address(raw)), Int(py=raw.shape[0]), Int(py=raw.shape[1]),
                Int(py=raw.strides[0]), Int(py=raw.strides[1]), classical_is_array(raw, context) or Bool(py=raw.dtype.isnative))
        _ = raw^
    var x = host_tree_input(prepared, context)
    _ = prepared^
    return x


def _svc_binary_support(value: PythonObject, count: Int, columns: Int, context: PythonObject) raises -> PythonObject:
    # Binary SVC historically calls support_.tolist() before _i32; unlike
    # pair machines, even an Array's wider scalar storage is converted.
    var b = Python.import_module("builtins")
    if py_type(value) is b.list or py_type(value) is b.tuple:
        raise Error("AttributeError: SVC support_ requires native array storage with tolist")
    var array = classical_input(value, context, "support_")
    if Int(py=array.ndim) != 1 or Int(py=array.size) != count:
        raise Error("SVC support index shape differs from n_support_")
    var width = Int(py=array.itemsize)
    var out = host_archive_typed(Python.tuple(count), "<i4", context["output_context"])
    host_svr_indices(Int(py=classical_address(array)), Int(py=array.strides[0]), count, width,
        classical_cast_kind(array, width, "support_", context),
        classical_is_array(array, context) or Bool(py=array.dtype.isnative), columns, Int(py=classical_address(out)))
    _ = array^
    return out


def _svc_machine(query: PythonObject, dual_value: PythonObject, support_value: PythonObject, columns_value: PythonObject,
                 count: Int, bias: Float64, label0: Float64, label1: Float64, classify: Bool, binary: Bool,
                 destination: Int, fields: PythonObject, context: PythonObject) raises:
    var x = query
    var features = Int(py=fields["n_features_in_"])
    var rows = Int(py=x.shape[0])
    if features < 1 or count < 0 or count > 2147483647 // features or rows > 2147483647 // max(count, 1):
        raise Error("SVC support dimensions exceed native indexing")
    var kernel = String(fields["_hp_kernel"])
    var code = 0 if kernel == "linear" else (1 if kernel == "poly" else (2 if kernel == "rbf" else (3 if kernel == "sigmoid" else (4 if kernel == "precomputed" else -1))))
    if code < 0:
        raise Error("SVC kernel name is not recognized")
    var dual = PythonObject(None)
    var support = PythonObject(None)
    var columns = PythonObject(None)
    if count > 0:
        dual = _kernel_buffer(dual_value, count, "<f4", context)
        if code == 4:
            columns = _svc_binary_support(columns_value, count, features, context) if binary else _svr_support(columns_value, count, features, context)
            var gathered = host_archive_typed(Python.tuple(rows, count), "<f4", context["output_context"])
            _ = py_call(_kernel_entry("svc_pair_epilogue", context), classical_address(x), 0,
                classical_address(columns), classical_address(gathered), Python.list(6, rows, features, rows, count))
            _ = x^
            x = gathered
            features = count
            support = dual
        else:
            support = _svr_matrix(support_value, count, features, context)
    var params = Python.list(rows, features, count, bias, label0, label1, code, fields["_gamma"], Int(classify),
        fields["_hp_cache_size"], py_call(Python.import_module("builtins").int, fields["_hp_degree"]), Float64(py=fields["_hp_coef0"]))
    _ = py_call(_kernel_entry("svc_predict", context), classical_address(x),
        classical_address(dual) if count > 0 else PythonObject(0),
        classical_address(support) if count > 0 else PythonObject(0), destination, params)
    _ = x^
    _ = dual^
    _ = support^
    _ = columns^


def _svc_pair_count(pair: PythonObject, context: PythonObject) raises -> Int:
    var dual = classical_input(pair["dual"], context, "pair dual")
    if Int(py=dual.ndim) != 2 or Int(py=dual.shape[0]) != 1:
        raise Error("SVC pair dual coefficients require shape (1, n_support)")
    return Int(py=dual.shape[1])


def _svc_epilogue(mode: Int, decisions: PythonObject, pair_codes: PythonObject, rows: Int, pairs: Int, classes: Int,
                  ab: PythonObject, label1: Float64, context: PythonObject) raises -> PythonObject:
    var shape = Python.tuple(rows) if mode == 2 or mode == 5 else Python.tuple(rows, pairs if mode == 0 else classes)
    var dtype = "<i8" if mode == 2 or mode == 5 else ("<f4" if mode == 0 else "<f8")
    var out = host_archive_typed(shape, dtype, context["output_context"])
    var decision_owner = decisions
    var code_owner = pair_codes
    var calibration_owner = ab
    _ = py_call(_kernel_entry("svc_pair_epilogue", context), classical_address(decision_owner), classical_address(code_owner),
        classical_address(calibration_owner) if calibration_owner is not None else PythonObject(0), classical_address(out), Python.list(mode, rows, pairs, classes, label1))
    _ = decision_owner^
    _ = code_owner^
    _ = calibration_owner^
    return out


def _svc_calibration_array(value: PythonObject, count: Int, vector: Bool, context: PythonObject) raises -> PythonObject:
    var array = classical_input(value, context, "SVC calibration")
    var width = Int(py=array.itemsize)
    var kind = classical_cast_kind(array, width, "SVC calibration", context)
    if Int(py=array.ndim) != (1 if vector else 2) or Int(py=array.shape[0]) != (2 if vector else count):
        raise Error("SVC probability calibration requires native (A, B) pairs")
    if not vector and Int(py=array.shape[1]) != 2:
        raise Error("SVC probability calibration requires native (A, B) pairs")
    var out = host_archive_typed(Python.tuple(count, 2), "<f8", context["output_context"])
    var words = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(out)))
    var base = Int(py=classical_address(array))
    var stride = Int(py=array.strides[0])
    var col = stride if vector else Int(py=array.strides[1])
    var native = classical_is_array(array, context) or Bool(py=array.dtype.isnative)
    for i in range(count):
        for j in range(2):
            words[2 * i + j] = host_svr_bias(base + (0 if vector else i * stride) + j * col, width, kind, native)
    _ = array^
    return out


def _svc_calibration(value: PythonObject, count: Int, context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    if py_type(value) is not b.list and py_type(value) is not b.tuple:
        return _svc_calibration_array(value, count, False, context)
    if len(value) != count:
        raise Error("SVC probability calibration requires one native pair per machine")
    var out = host_archive_typed(Python.tuple(count, 2), "<f8", context["output_context"])
    var words = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(out)))
    for i in range(count):  # driver-loop: admitted Platt calibration words
        var pair = value[i]
        if py_type(pair) is b.list or py_type(pair) is b.tuple:
            if len(pair) != 2:
                raise Error("SVC probability calibration requires native (A, B) pairs")
            host_inference_scalar(pair[0], context)
            host_inference_scalar(pair[1], context)
            words[2 * i] = Float64(py=pair[0])
            words[2 * i + 1] = Float64(py=pair[1])
        else:
            var packed = _svc_calibration_array(pair, 1, True, context)
            var source = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(packed)))
            words[2 * i] = source[0]
            words[2 * i + 1] = source[1]
            _ = packed^
    return out


def host_svc_inference(model: PythonObject, fields: PythonObject, labels: PythonObject, operation: String,
                       source: PythonObject, context: PythonObject, public: Bool) raises -> PythonObject:
    var probability = operation == "host_infer_predict_proba" or operation == "host_infer_predict_log_proba"
    if probability:
        # Availability is checked by attribute lookup. A previously bound
        # method (or class-level call) keeps the original fitted-calibration
        # contract after probability is changed on the instance.
        if py_get(fields, "_prob_ab") is None:
            raise Error("AttributeError: SVC predict_proba requires fitted probability calibration; fit with probability=True")
    var x = _svc_input(source, context)
    var rows = Int(py=x.shape[0])
    if Int(py=x.shape[1]) != Int(py=fields["n_features_in_"]):
        raise Error("SVC input width differs from fitted dimensions")
    if not Bool(py=py_get(fields, "_mojolearn_fitting", False)):
        host_inference_feature_names(model, fields, context)
    var classes = len(labels)
    if classes < 2:
        raise Error("SVC requires at least two fitted classes")
    var pairs = py_get(fields, "_pairs")
    var count = 1 if pairs is None else len(pairs)
    if count < 1 or rows > 2147483647 // count or rows > 2147483647 // classes:
        raise Error("SVC decision dimensions exceed native indexing")
    var decisions = host_archive_typed(Python.tuple(count, rows), "<f4", context["output_context"])
    var codes = host_archive_typed(Python.tuple(count, 2), "<i4", context["output_context"])
    var code_words = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(codes)))
    var base = Int(py=classical_address(decisions))
    if pairs is None:
        code_words[0] = 0
        code_words[1] = 1
        _svc_machine(x, fields["dual_coef_"], fields["support_vectors_"], fields["support_"],
            Int(py=fields["n_support_"]), _svr_bias(fields["intercept_"], context), Float64(py=fields["_label0"]),
            Float64(py=fields["_label1"]), operation == "host_infer_predict", True, base, fields, context)
    else:
        for i in range(count):  # driver-loop: existing one-vs-one machine order and pair-major writes
            var pair = pairs[i]
            var low = Int(py=pair["i"])
            var high = Int(py=pair["j"])
            if low < 0 or high < 0 or low >= classes or high >= classes:
                raise Error("SVC pair class index exceeds fitted classes")
            code_words[2 * i] = Int32(low)
            code_words[2 * i + 1] = Int32(high)
            _svc_machine(x, pair["dual"], pair["sv"], pair["support"], _svc_pair_count(pair, context),
                Float64(py=pair["b"]), 0.0, 1.0, False, False, base + 4 * rows * i, fields, context)
    var out = PythonObject(None)
    var ab = PythonObject(None)
    if probability:
        ab = _svc_calibration(fields["_prob_ab"], count, context)
        out = _svc_epilogue(4 if operation == "host_infer_predict_log_proba" else 3, decisions, codes, rows, count, classes, ab, 0.0, context)
    elif operation == "host_infer_decision_function":
        if pairs is None:
            # Copy completed words without changing binary margin bits.
            out = host_archive_typed(Python.tuple(rows), "<f4", context["output_context"])
            var src = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=base)
            var dst = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(out)))
            for i in range(rows):
                dst[i] = src[i]
        else:
            out = _svc_epilogue(0 if String(fields["decision_function_shape"]) == "ovo" else 1,
                decisions, codes, rows, count, classes, ab, 0.0, context)
    else:
        var indices = List[Int]()
        var index_array = PythonObject(None)
        if pairs is not None and Bool(py=fields["_hp_break_ties"]):
            var scores = _svc_epilogue(1, decisions, codes, rows, count, classes, ab, 0.0, context)
            index_array = host_archive_typed(Python.tuple(rows), "<i8", context["output_context"])
            argmax_rows_f64(MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(scores))), rows, classes,
                MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(index_array))))
            _ = scores^
        else:
            index_array = _svc_epilogue(5 if pairs is None else 2, decisions, codes, rows, count, classes, ab,
                Float64(py=fields["_label1"]) if pairs is None else 0.0, context)
        var values = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(index_array)))
        for i in range(rows):  # driver-loop: completed native class code publication
            indices.append(Int(values[i]))
        _ = index_array^
        out = _logistic_decode(labels, indices, py_type(fields["classes_"]) is py_get(context, "ndarray"), context)
    _ = x^
    _ = decisions^
    _ = codes^
    _ = ab^
    return classical_input(out, context, "SVC result") if py_type(out) is not Python.import_module("builtins").list and public and py_get(context, "ndarray") is not None else out


def _svc_coefficients(model: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = host_tree_constructor_fields(model, context)
    if not py_contains(fields, "dual_coef_"):
        raise Error("AttributeError: SVC coef_ requires a fitted model")
    var keys = py_call(Python.import_module("builtins").list, fields.keys())
    for i in range(len(keys)):  # driver-loop: state admission before the computed property
        if String(keys[i]) == "classes_":
            _ = host_logistic_classes(fields[keys[i]], context)
        else:
            classical_admit_value(fields[keys[i]], context)
    host_svc_admit_state(fields, context)
    if String(fields["_hp_kernel"]) != "linear":
        raise Error("AttributeError: SVC coef_ is available only for kernel='linear'")
    var features = Int(py=fields["n_features_in_"])
    var pairs = py_get(fields, "_pairs")
    var rows = 1 if pairs is None else len(pairs)
    if features < 1 or rows < 1 or rows > 2147483647 // features:
        raise Error("SVC coefficient dimensions exceed native indexing")
    var out = host_archive_typed(Python.tuple(rows, features), "<f4", context["output_context"])
    for i in range(rows):  # driver-loop: pair coefficient orientation matches the existing public property
        var count = Int(py=fields["n_support_"]) if pairs is None else _svc_pair_count(pairs[i], context)
        if count < 0 or count > 2147483647 // features:
            raise Error("SVC support dimensions exceed native indexing")
        if count == 0:
            continue
        var mode = py_get(fields, "numeric_mode")
        if mode is not None and (py_type(mode) is not Python.import_module("builtins").str or String(mode) != "identical"):
            raise Error("explicit host models run identical mode only")
        var dual = _kernel_buffer(fields["dual_coef_"] if pairs is None else pairs[i]["dual"], count, "<f4", context)
        var support = _svr_matrix(fields["support_vectors_"] if pairs is None else pairs[i]["sv"], count, features, context)
        _ = py_call(_kernel_entry("svc_dual_gemv", context), classical_address(dual), classical_address(support),
            Python.list(count, features, Int(pairs is not None)), Int(py=classical_address(out)) + 4 * features * i)
        _ = dual^
        _ = support^
    return classical_input(out, context, "coef_") if _outside(context, 0) and py_get(context, "ndarray") is not None else out


def host_svc_attribute_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=args_ptr)
        var context = args[0]
        var model = args[1]
        var name = args[2]
        var b = Python.import_module("builtins")
        classical_connect_numpy(context)
        classical_check_layout(model, context)
        if py_type(name) is not b.str:
            raise Error("TypeError: native SVC attribute names must be exact strings")
        var attr = String(name)
        if attr == "coef_" or attr == "predict_proba" or attr == "predict_log_proba":
            var mro = py_call(b.type.__dict__["__mro__"].__get__, py_type(model))
            for i in range(len(mro)):  # driver-loop: preserve normal MRO overrides and exact connected identities
                var fields = _pd_type_fields(mro[i])
                if py_contains(fields, name):
                    if attr == "coef_" and fields[name] is context["host_svc_coefficient_property"]:
                        var coefficients = _svc_coefficients(model, context)
                        return coefficients.steal_data()
                    if attr != "coef_" and fields[name] is context["host_svc_probability_methods"][name]:
                        host_svc_probability(model, host_tree_constructor_fields(model, context), name, context)
                    break
        var value = classical_getattribute(model, name, context)
        if not _outside(context, 0):
            var original = py_get(py_fields(model), name)
            if classical_is_array(original, context):
                value = original
        return value.steal_data()
    except error:
        if String(error).startswith("AttributeError:"):
            return raise_python_exception(error_without_class(error, "AttributeError"), ExceptionType("PyExc_AttributeError"))
        if String(error).startswith("TypeError:"):
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        return raise_python_exception(error, ExceptionType.ValueError)
