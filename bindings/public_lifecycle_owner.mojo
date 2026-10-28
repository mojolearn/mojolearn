# SPDX-License-Identifier: Apache-2.0
# The estimator lifecycle the `_mode` class-creation wrappers used to run in
# Python (`_track_fit`, `_needs_fit`, `_public_output`, `_sk_validate`), for
# the native owners of the preprocessing, decomposition and UMAP classes:
# the fitted check (NotFittedError), scikit-learn's X refusals and width /
# feature-name checks, fit bookkeeping (depth, `_mojolearn_fitted`,
# `feature_names_in_`, ClassLabels `classes_`), and NumPy results for callers
# outside the package. An owner calls `lifecycle_begin` before its algorithm,
# `lifecycle_end` in a `finally`, then `lifecycle_complete` and
# `lifecycle_result` (same order as `_xl_public_operation` in
# bindings/x_linear_native.mojo, the pattern this generalizes).
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem, py_contains
from bindings.classical_metadata import classical_is_fitted
from bindings.classical_exceptions import classical_not_fitted_class
from bindings.classical_validate import classical_estimator_name, RESHAPE_HINT
from bindings.optimizer_public_words import stash_python_error
from bindings.parallel_public_result import parallel_public_result
from providers.python.array_metadata_pyapi import array_slot_once
from providers.python.estimator_lifecycle_pyapi import (
    estimator_feature_check, estimator_fit_begin, estimator_fit_end, estimator_fit_complete,
)


def lifecycle_fitting(method: String) -> Bool:
    """The methods that fit (`_mode._FITS`)."""
    return method == "fit" or method == "partial_fit" or method == "fit_transform" or method == "fit_predict"


def lifecycle_needs_fit(method: String) -> Bool:
    """The methods that refuse an unfitted instance (`_mode._NEEDS_FIT`)."""
    return (method == "predict" or method == "predict_proba" or method == "predict_log_proba"
            or method == "decision_function" or method == "transform" or method == "inverse_transform"
            or method == "score" or method == "score_samples" or method == "predict_joint_log_proba")


def lifecycle_not_fitted(model: PythonObject, method: String, context: PythonObject) raises:
    """Raise the package's NotFittedError (sklearn's when it is loaded) as a
    pending Python exception, which every native error boundary restores."""
    var cls = classical_not_fitted_class(context)
    var message = ("This " + classical_estimator_name(model) + " instance is not fitted yet. Call 'fit' with "
                   + "appropriate arguments before using " + method + ".")
    ref cpy = Python().cpython()
    cpy.PyErr_SetString(cls._obj_ptr, message.as_c_string_slice().unsafe_ptr().as_unsafe_any_origin())
    var failure = stash_python_error()
    raise failure^


def _repr(value: PythonObject) raises -> String:
    return String(py_call(native_loaded_module("builtins").repr, value))


def lifecycle_x_shape(source: PythonObject, context: PythonObject) raises -> PythonObject:
    """The shape tuple of an exact ndarray or mojolearn Array (metadata only,
    no conversion), else None (lists, frames, strings: the owner's own input
    step admits them)."""
    if py_type(source) is py_get(context, "ndarray"):
        return source.shape
    if py_type(source) is py_get(context, "array_class"):
        return array_slot_once(source, "shape", context)
    return PythonObject(None)


def _x_kind(source: PythonObject, context: PythonObject) raises -> String:
    if py_type(source) is py_get(context, "ndarray"):
        return String(source.dtype.kind)
    var dtype = String(py=array_slot_once(source, "dtype", context))
    if dtype.byte_length() >= 2:
        var code = String(dtype[byte=1:2])
        if code == "f" or code == "i" or code == "u" or code == "b" or code == "c":
            return code
    return "O"


def lifecycle_check_x(model: PythonObject, source: PythonObject, sparse_ok: Bool,
                      context: PythonObject) raises:
    """scikit-learn's `validate_data` refusals on X (`_sk_validate._check_X`):
    sparse (unless the class takes it), complex, rank below two, no rows, no
    columns. Values are read by the owner's own typed conversion."""
    var est = classical_estimator_name(model)
    var module = py_type(source).__module__
    if (not sparse_ok and py_type(module) is native_loaded_module("builtins").str
            and Bool(py=py_call(module.startswith, "scipy.sparse"))):
        raise Error("TypeError: Sparse data was passed for X, but dense data is required. Use '.toarray()' to "
                    + "convert to a dense numpy array. (" + est + " takes dense X only.)")
    var shape = lifecycle_x_shape(source, context)
    if shape is None:
        return
    var kind = _x_kind(source, context)
    if kind == "c":
        raise Error("Complex data not supported: " + est + " got a complex X.")
    if kind == "O" or kind == "U" or kind == "S":
        return  # object/string storage: the owner's encoder input step
    var nd = len(shape)
    if nd < 2:
        raise Error("Expected 2D array, got " + ("scalar" if nd == 0 else "1D") + " array instead (shape="
                    + _repr(shape) + "). " + RESHAPE_HINT)
    if nd == 2 and Int(py=shape[0]) == 0:
        raise Error("Found array with 0 sample(s) (shape=" + _repr(shape) + ") while a minimum of 1 is required by " + est + ".")
    if nd == 2 and Int(py=shape[1]) == 0:
        raise Error("Found array with 0 feature(s) (shape=" + _repr(shape) + ") while a minimum of 1 is required by " + est + ".")


def lifecycle_check_width(model: PythonObject, fields: PythonObject, source: PythonObject,
                          context: PythonObject) raises:
    """X's width against `n_features_in_` (`_sk_validate._check_width`) and
    its column names against `feature_names_in_`."""
    var b = native_loaded_module("builtins")
    var fitted = py_get(fields, "n_features_in_")
    var shape = lifecycle_x_shape(source, context)
    if shape is not None and len(shape) == 2 and py_type(fitted) is b.int:
        if Int(py=shape[1]) != Int(py=fitted):
            raise Error("X has " + String(Int(py=shape[1])) + " features, but " + classical_estimator_name(model)
                        + " is expecting " + String(Int(py=fitted)) + " features as input.")
    estimator_feature_check(model, fields, source, context)


def lifecycle_admit_state(model: PythonObject, fields: PythonObject) raises:
    var b = native_loaded_module("builtins")
    if py_type(py_get(fields, "_mojolearn_fitting", PythonObject(0))) is not b.int:
        raise Error("TypeError: native fitting depth requires an exact integer")
    if py_contains(fields, "_mojolearn_after_fit"):
        raise Error("TypeError: native fitting cannot invoke Python completion callbacks")


def lifecycle_begin(model: PythonObject, method: String, source: PythonObject, x_checks: Bool,
                    sparse_ok: Bool, requires_fit: Bool, context: PythonObject) raises -> Int:
    """Before the algorithm. `method` is the public method name; `source` the
    caller's X (None when the method takes none); `x_checks` False for a class
    whose X is not a numeric table (labels, raw documents); `requires_fit`
    False for a stateless transformer. Returns the fit depth to hand back to
    `lifecycle_end` (0 for a method that does not fit)."""
    var fields = py_fields(model)
    lifecycle_admit_state(model, fields)
    var nested = Int(py=py_get(fields, "_mojolearn_fitting", PythonObject(0))) > 0
    if lifecycle_fitting(method):
        if x_checks and not nested and source is not None:
            lifecycle_check_x(model, source, sparse_ok, context)
        return estimator_fit_begin(fields)
    if lifecycle_needs_fit(method):
        if requires_fit and not Bool(py=classical_is_fitted(model)):
            lifecycle_not_fitted(model, method, context)
        if x_checks and not nested and source is not None:
            lifecycle_check_x(model, source, sparse_ok, context)
            if requires_fit:
                lifecycle_check_width(model, fields, source, context)
    return 0


def lifecycle_end(model: PythonObject, method: String, depth: Int) raises:
    if lifecycle_fitting(method):
        estimator_fit_end(model, depth)


def lifecycle_complete(model: PythonObject, method: String, source: PythonObject, classifier: Bool,
                       context: PythonObject) raises:
    """After a successful fit: `_mojolearn_fitted`, `feature_names_in_` and,
    for a classifier, the public ClassLabels `classes_`."""
    if not lifecycle_fitting(method):
        return
    if classifier and not py_contains(py_fields(model), "classes_"):
        estimator_fit_complete(model, source, False, context)
        return
    estimator_fit_complete(model, source, classifier, context)


def lifecycle_result(model: PythonObject, method: String, result: PythonObject, kind: String,
                     context: PythonObject) raises -> PythonObject:
    """The result for the caller: `fit`/`partial_fit` return the estimator;
    other Arrays become NumPy for callers outside the package (float64 for a
    regressor's predict and a detector's scores; `kind` is the estimator's
    `_estimator_type`, "" for none)."""
    if method == "fit" or method == "partial_fit" or method == "score":
        return result
    var label = PythonObject(None) if kind.byte_length() == 0 else PythonObject(kind)
    return parallel_public_result(result, method, label, context)
