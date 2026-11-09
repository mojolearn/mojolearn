# SPDX-License-Identifier: Apache-2.0
# Whole public neighbor-operation envelope. The dedicated owners retain their
# numerical algorithms; packed base KNN/KDE extension dispatch is still explicit
# migration debt, not evidence of descriptor-only execution.
from std.math import isfinite, floor
from std.python import Python, PythonObject
from bindings.admitted_family import admit_family_meta, admit_family_owned
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.mode_metadata_native import _fields, _fitted
from bindings.classical_attributes import classical_connect_numpy
from bindings.classical_pandas import classical_admit_value
from bindings.classical_arrays import classical_address
from bindings.classical_validate import classical_estimator_name, classical_check_width
from bindings.classical_exceptions import classical_not_fitted_class
from bindings.classical_labels import labels_shape_of, _flatten_labels
from bindings.label_integer_storage import labels_integer_storage, labels_float_storage
from bindings.label_input_sources import labels_storage_source, labels_list_integer_storage, labels_text_storage
from core.label_storage_impl import LABELS_FLOAT_OK, LABELS_FLOAT_NONFINITE, LABELS_FLOAT_CONTINUOUS
from bindings.classical_warnings import classical_warn_column
from bindings.classifier_public_lifecycle import _public_labels
from bindings.host_inference_scalars import host_inference_scalar
from bindings.native_array_view import native_array_ndarray_view
from core.p3nc_scan import p3nc_widen_f32_f64, P3NC_REAL_OK, P3NC_REAL_NONFINITE, P3NC_REAL_CONTINUOUS
from core.host_finite_scan import host_all_finite_f32
from bindings.p3nc_label_lists import p3nc_float_list_verdict
from bindings.family_common import FAMILY_F32P, FAMILY_F64P, family_array
from bindings.metrics_native_support import metric_float, metric_is_real, metric_is_integral
from bindings.host_sparse_native import host_sparse_format, _sparse_fields
from bindings.cluster_fit_a_native import _pending, _pending_text, _drop_pending
from bindings.optimizer_public_words import PYTHON_ERROR_PENDING
from providers.python.estimator_lifecycle_pyapi import (
    estimator_outside, estimator_feature_check, estimator_fit_begin, estimator_fit_end, estimator_fit_complete,
)


struct NeighborPublicCall(Movable):
    var args: PythonObject
    var fields: PythonObject
    var source: PythonObject
    var method: String
    var fitting: Bool
    var outside: Bool
    var depth: Int
    var rows: Int
    var columns: Int

    def __init__(out self, args: PythonObject, fields: PythonObject, source: PythonObject,
                 method: String, fitting: Bool, outside: Bool, depth: Int, rows: Int, columns: Int):
        self.args = args
        self.fields = fields
        self.source = source
        self.method = method
        self.fitting = fitting
        self.outside = outside
        self.depth = depth
        self.rows = rows
        self.columns = columns


def _method(operation: String) raises -> String:
    var names = ["fit_predict", "fit_transform", "predict_log_proba", "predict_proba", "score_samples",
                 "decision_function", "transform", "predict", "score", "fit"]
    for suffix in names:  # small-loop(public neighbor operation names)
        if operation.endswith(suffix):
            return suffix
    raise Error("RuntimeError: unknown native neighbor lifecycle operation " + operation)


def _target(source: PythonObject, rows: Int, name: String, classifier: Bool, required: Bool,
            multioutput: Bool, context: PythonObject) raises -> PythonObject:
    if source is None:
        if required:
            raise Error(name + " requires y to be passed, but the target y is None.")
        return source
    var b = native_loaded_module("builtins")
    if host_sparse_format(source).byte_length() > 0:
        var shape = _sparse_fields(source)["_shape"]
        if py_type(shape) is not b.tuple or len(shape) != 2 or py_type(shape[0]) is not b.int:
            raise Error("TypeError: sparse target shape requires native dimensions")
        if Int(py=shape[0]) != rows:
            raise Error("Found input variables with inconsistent numbers of samples: [" + String(rows) + ", " + String(Int(py=shape[0])) + "]")
        return source
    classical_admit_value(source, context)
    if py_type(source) is py_get(context, "ndarray") and String(source.dtype.kind) == "c":
        raise Error("Complex data not supported: " + name + " got a complex y.")
    var shape = labels_shape_of(source, context)
    if len(shape) > 0 and shape[0] != rows:
        raise Error("Found input variables with inconsistent numbers of samples: [" + String(rows) + ", " + String(shape[0]) + "]")
    # Perf (perf2-neighbors): any rank is admitted from the buffer, not only a
    # vector. The walk below returns `source` unchanged for every shape except
    # the non-multioutput (n, 1) column, which it flattens with a warning, so
    # only that case keeps the walk (a 2-D multioutput KNN y used to pay one
    # Python object and four scalar checks per label).
    var column = not multioutput and len(shape) == 2 and shape[1] == 1
    var buffered = len(shape) >= 1 and not column
    # An exact integer/bool ndarray or Array holds no complex, non-finite or
    # non-integral element and no Python object, so the element walk below
    # cannot refuse it and it is returned unchanged either way: admit it from
    # its dtype metadata instead of label by label.
    # perf3-label-doors: a numpy-backed pandas Series is checked through its
    # stored exact ndarray (the same labels in the same order, 1-D, so never
    # the column case) and returned unchanged, as the walk returns it.
    var stored = labels_storage_source(source, context) if buffered else PythonObject(None)
    if stored is not None and labels_integer_storage(stored, context):
        return source
    # An exact flat list / tuple of exact ints or bools: every label passes
    # every check below and a list is returned unchanged by the walk.
    if labels_list_integer_storage(source):
        return source
    # An exact '<U' ndarray lists exact str only, which the walk admits and
    # never tests as real; any shape but the flattened column is returned
    # unchanged by the walk.
    if stored is not None and labels_text_storage(source, context):
        return source
    # Perf: an exact float32/float64 ndarray or Array vector holds only real
    # floats, so the element walk below reduces to its finite (and, for a
    # classifier, integral) test. The one shared native buffer check
    # (bindings/label_integer_storage.mojo, also the label door's) refuses at
    # the first refused element in C order with the same wording, instead of
    # one Python float and three scalar checks per label.
    if stored is not None:
        var refusal = labels_float_storage(stored, context, classifier)
        if refusal == LABELS_FLOAT_NONFINITE:
            raise Error("Input y contains NaN or infinity (" + name + ").")
        if refusal == LABELS_FLOAT_CONTINUOUS:
            raise Error("Unknown label type: continuous. Maybe you are trying to fit a classifier, which expects discrete classes on a regression target with continuous values.")
        if refusal == LABELS_FLOAT_OK:
            return source
    # p3-neighbors-cluster: an exact flat list / tuple of exact floats is
    # checked in one native pass (bindings/p3nc_label_lists.mojo) with the
    # walk's verdicts; the walk returns such a list unchanged.
    var verdict = p3nc_float_list_verdict(source, classifier)
    if verdict == P3NC_REAL_NONFINITE:
        raise Error("Input y contains NaN or infinity (" + name + ").")
    if verdict == P3NC_REAL_CONTINUOUS:
        raise Error("Unknown label type: continuous. Maybe you are trying to fit a classifier, which expects discrete classes on a regression target with continuous values.")
    if verdict == P3NC_REAL_OK:
        return source
    var flat = _flatten_labels(source, context)
    classical_admit_value(flat, context)
    for i in range(len(flat)):  # driver-loop: admitted label words, never Python element protocols
        if py_type(flat[i]) is b.complex:
            raise Error("Complex data not supported: " + name + " got a complex y.")
        host_inference_scalar(flat[i], context)
        if metric_is_real(flat[i], context) and not metric_is_integral(flat[i], context):
            var number = metric_float(flat[i], context)
            if not isfinite(number):
                raise Error("Input y contains NaN or infinity (" + name + ").")
            if classifier and floor(number) != number:
                raise Error("Unknown label type: continuous. Maybe you are trying to fit a classifier, which expects discrete classes on a regression target with continuous values.")
    if (not multioutput and len(shape) == 2 and shape[1] == 1 and py_type(source) is not b.list
            and py_type(source) is not b.tuple and py_type(source) is not b.memoryview):
        classical_warn_column(context)
        return flat
    return source


def neighbors_public_begin(model: PythonObject, operation: String, args: PythonObject,
                           context: PythonObject) raises -> NeighborPublicCall:
    classical_connect_numpy(context)
    var b = native_loaded_module("builtins")
    if py_type(py_type(model)) is not b.type:
        raise Error("TypeError: native neighbor operations refuse Python metaclass hooks")
    # _fields admits the actual native dictionary descriptor and every key;
    # object.__getattribute__(model, '__dict__') could invoke a replaced property.
    var fields = _fields(model)
    var method = _method(operation)
    var fitting = method == "fit" or method == "fit_predict" or method == "fit_transform"
    var depth = py_get(fields, "_mojolearn_fitting", PythonObject(0))
    if py_type(depth) is not b.int:
        raise Error("TypeError: native neighbor fit depth must be an exact integer")
    var fitted = py_get(fields, "_mojolearn_fitted", PythonObject(False))
    if py_type(fitted) is not b.bool and py_type(fitted) is not b.int:
        raise Error("TypeError: native fitted flags cannot invoke Python boolean hooks")
    if not fitting and not Bool(py=_fitted(model)):
        _ = classical_not_fitted_class(context)
        raise Error("NotFittedError: This " + classical_estimator_name(model)
                    + " instance is not fitted yet. Call 'fit' with appropriate arguments before using " + method + ".")
    var first = 0 if operation == "family_xn_knn_nn_fit" else 1
    var source = args[first]
    var name = classical_estimator_name(model)
    # KNNImputer accepts NaNs. The other local/nearest estimators had a public
    # finite scan; KDE keeps its existing fit/score scan and copy reporting.
    # The test rides in the admission (the cast's own verdict, or one
    # parallel scan of a borrowed X): no second pass over X.
    var scan = not operation.startswith("family_xn_loc_imp_") and not operation.startswith("family_xn_kde_")
    var snapshot = admit_family_meta(source, 2, "X", context, True, finite=scan,
                                     refusal="Input X contains NaN or infinity (" + name + ").")
    var rows = snapshot.shape[0]
    var columns = snapshot.shape[1]
    if not fitting and Int(py=depth) == 0:
        var width = py_get(fields, "n_features_in_")
        if width is not None:
            host_inference_scalar(width, context)
            classical_check_width(columns, Int(py=width), name)
        estimator_feature_check(model, fields, source, context)
    var values = Python.list()
    for i in range(len(args)):  # small-loop(fixed operation argument envelope)
        _ = py_call(values.append, args[i])
    # These owners already take their own float32 training/query buffers.
    # Hand them one stable native Array, rather than revisiting user metadata.
    # KDE retains its borrowing/copy-report contract; its repeated metadata
    # capture remains explicitly pending until its dense helper takes snapshots.
    if not operation.startswith("family_xn_kde_"):
        # Perf: a query (predict/transform/score/kneighbors) only reads X, so
        # an exact C-order float32 input that is not readonly is handed over
        # as admitted, without a per-call copy. A fit (the model retains X),
        # the imputer (writes into its X buffer) and a readonly view still get
        # the owned copy.
        var borrow = (not fitting and not operation.startswith("family_xn_loc_imp_")
                      and not snapshot.readonly and snapshot.f32_c())
        # 0.8.66 regression C3 (knn-reg / knn-clf fit 8.5x on istella, 2.2x
        # on taxi): a k-NN fit keeps a reference to the caller's float32
        # C-order matrix as its index, as 0.8.55 and scikit-learn's brute
        # NearestNeighbors do, instead of a zero-filled copy. The caller's
        # array is never frozen or otherwise changed (coordinator decision
        # 2026-11-08); the fit docstring tells users to refit after an
        # in-place edit.
        if (fitting and not borrow and operation.startswith("family_xn_knn_") and snapshot.f32_c()
                and not snapshot.copied):
            borrow = True
        py_setitem(values, PythonObject(first), snapshot.keep if borrow else admit_family_owned(snapshot, context))
    elif not fitting and host_sparse_format(source).byte_length() == 0:
        # Perf (perf2-neighbors): a dense KDE query (score_samples/score) only
        # reads X and never reports `input_copied_`, so it takes this float32
        # C-order snapshot as `_kde_scores` would have rebuilt it
        # (`family_as_f32` is this same snapshot's `keep`). A KDE fit keeps
        # its own capture and copy report; a sparse query keeps its exact
        # densification (bindings/neighbors_kde_native.mojo `_kde_dense`).
        py_setitem(values, PythonObject(first), snapshot.keep)
    if fitting and Int(py=depth) == 0:
        var target_index = 2
        if operation == "family_xn_knn_nn_fit":
            target_index = 1
        elif operation == "family_xn_kde_fit":
            target_index = 3
        var classifier = operation == "family_xn_knn_clf_fit" or operation == "family_xn_loc_nc_fit"
        var required = classifier or operation == "family_xn_knn_reg_fit"
        if target_index < len(args):
            var target = _target(args[target_index], rows, name, classifier, required,
                operation.startswith("family_xn_knn_"), context)
            py_setitem(values, PythonObject(target_index), target)
        elif required:
            raise Error(name + " requires y to be passed, but the target y is None.")
    var previous = estimator_fit_begin(fields) if fitting else Int(py=depth)
    return NeighborPublicCall(values, fields, source, method, fitting, estimator_outside(), previous, rows, columns)


def neighbors_public_end(model: PythonObject, call: NeighborPublicCall) raises:
    if call.fitting:
        estimator_fit_end(model, call.depth)


def _public_result(value: PythonObject, widen: Bool, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    if py_type(value) is b.list or py_type(value) is b.tuple:
        var output = Python.list()
        for i in range(len(value)):  # driver-loop: finished public multioutput result containers
            _ = py_call(output.append, _public_result(value[i], widen, context))
        return py_call(b.tuple, output) if py_type(value) is b.tuple else output
    if py_type(value) is not context["array_class"]:
        return value
    var result = value
    if widen and String(value.dtype) == "<f4":
        var count = Int(py=value.size)
        result = family_array(value.shape, "<f8", context)
        # p3-neighbors-cluster: eight lanes at a time (exact widening).
        p3nc_widen_f32_f64(FAMILY_F32P(unsafe_from_address=Int(py=classical_address(value))),
                           FAMILY_F64P(unsafe_from_address=Int(py=classical_address(result))), count)
        _ = value
        _ = result
    if py_get(context, "ndarray") is not None:
        return native_array_ndarray_view(result, context)
    return result


def neighbors_public_finish(model: PythonObject, operation: String, result: PythonObject,
                            call: NeighborPublicCall, context: PythonObject) raises -> PythonObject:
    if call.fitting:
        estimator_fit_complete(model, call.source, False, context)
        if not py_contains(call.fields, "n_features_in_"):
            py_setitem(call.fields, "n_features_in_", PythonObject(call.columns))
        if operation.startswith("family_xn_knn_"):
            # Former NearestNeighbors._mojolearn_after_fit cache initialization.
            # The fit owner already released the previous resident before work.
            py_setitem(call.fields, "_resident_holder", Python.dict())
        if operation == "family_xn_loc_nc_fit":
            py_setitem(call.fields, "classes_", _public_labels(call.fields["classes_"], context))
    if call.outside and result is not model:
        var outlier = operation.startswith("family_xn_loc_lof_") and (call.method == "score_samples" or call.method == "decision_function")
        var regressor = operation == "family_xn_knn_reg_predict" and py_get(context, "ndarray") is not None
        return _public_result(result, outlier or regressor, context)
    return result


def neighbors_public_refusal(error: Error, call: NeighborPublicCall, model: PythonObject) raises -> Error:
    var text = String(error)
    var pending = text == PYTHON_ERROR_PENDING
    var b = native_loaded_module("builtins")
    var original = _pending(0) if pending else PythonObject(None)
    if pending and original is not b.Exception and original is not b.ValueError and original is not b.ZeroDivisionError:
        return Error(text)
    if pending:
        text = _pending_text(error)
    for fault in ["device_context", "CUDA", "HIP", "hipError", "Metal", "MTL"]:  # small-loop(device errors remain native failures)
        if text.find(fault) >= 0:
            return Error(String(error))
    var message = String("")
    var name = classical_estimator_name(model)
    if call.depth == 0 and pending and original is b.Exception:
        for spelling in ["NaN", "infinity", "Infinity", "infinite", "Infinite", "nonfinite", "non-finite", "not finite", "requires finite", "must be finite"]:  # small-loop(public nonfinite refusal spellings)
            if text.find(spelling) >= 0:
                message = (String("Input") if call.fitting else String("Input X")) + " contains NaN or infinity (" + name + ": " + text + ")"
                break
    if message.byte_length() == 0 and call.fitting and call.depth == 0:
        var value_error = original is b.ValueError or (not pending and not text.startswith("TypeError:") and not text.startswith("RuntimeError:") and not text.startswith("ImportError:") and not text.startswith("AttributeError:") and not text.startswith("OverflowError:") and not text.startswith("IndexError:"))
        if call.rows == 1 and (value_error or original is b.Exception or original is b.ZeroDivisionError):
            message = name + " cannot be fit on n_samples=1: " + text
        elif call.columns == 1 and value_error:
            message = name + " cannot be fit on n_features=1: " + text
    if message.byte_length() == 0:
        return Error(String(error))
    if pending:
        _drop_pending()
    return Error("ValueError: " + message)
