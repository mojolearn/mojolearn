# SPDX-License-Identifier: Apache-2.0
# The public lifecycle of the decomposition expansion's estimators (the
# former `_mode` class-creation wrappers), at each of their native doors:
# decomp_c, ipca_c, dict_learning_c, mcd_c and als_c. Every public method
# of IncrementalPCA, the random projections, NMF, FastICA, FactorAnalysis,
# LatentDirichletAllocation, the dictionary learners, the sparse PCAs,
# SparseCoder, MinCovDet, EllipticEnvelope and AlternatingLeastSquares is
# one native call that runs the fitted check (NotFittedError), scikit-learn's
# X refusals and width / feature-name checks, fit bookkeeping and NumPy
# results for callers outside the package (bindings/public_lifecycle_owner.mojo).
from std.python import Python, PythonObject
from bindings.py_balanced import py_get, py_fields, py_contains
from bindings.public_lifecycle_owner import (
    lifecycle_begin, lifecycle_end, lifecycle_complete, lifecycle_result, lifecycle_check_width,
    lifecycle_check_x, lifecycle_x_shape,
)
from bindings.classical_validate import classical_estimator_name


def decomp_public_method(operation: String) -> String:
    """The public method a door operation serves ("" for a private one: the
    random projection's components_ slot, get_covariance, perplexity, ...)."""
    if (operation == "fit" or operation == "partial_fit" or operation == "fit_transform"
            or operation == "fit_predict" or operation == "transform" or operation == "inverse_transform"
            or operation == "score" or operation == "score_samples" or operation == "decision_function"
            or operation == "predict"):
        return operation
    if operation == "dl_fit" or operation == "mbdl_fit" or operation == "spca_fit" or operation == "coder_fit":
        return "fit"
    if (operation == "dl_fit_transform" or operation == "mbdl_fit_transform" or operation == "spca_fit_transform"
            or operation == "coder_fit_transform"):
        return "fit_transform"
    if operation == "coding_transform" or operation == "spca_transform" or operation == "coder_transform":
        return "transform"
    if operation == "coding_inverse" or operation == "spca_inverse" or operation == "coder_inverse":
        return "inverse_transform"
    if operation == "elliptic_score":
        return "score"
    return ""


def decomp_public_begin(model: PythonObject, method: String, args: PythonObject, x_checks: Bool,
                        requires_fit: Bool, context: PythonObject) raises -> Int:
    """Before the algorithm; returns the fit depth for `lifecycle_end`."""
    if method.byte_length() == 0:
        return 0
    var source = args[0] if len(args) > 0 else PythonObject(None)
    var fields = py_fields(model)
    var nested = Int(py=py_get(fields, "_mojolearn_fitting", PythonObject(0))) > 0
    if method == "partial_fit" and x_checks and not nested and py_contains(fields, "n_features_in_"):
        # a later batch keeps the first batch's width and feature names
        # (scikit-learn's validate_data(reset=False)), before the depth moves
        lifecycle_check_x(model, source, False, context)
        lifecycle_check_width(model, fields, source, context)
    return lifecycle_begin(model, method, source, x_checks, False, requires_fit, context)


def decomp_public_finish(model: PythonObject, method: String, args: PythonObject, result: PythonObject,
                         kind: String, context: PythonObject) raises -> PythonObject:
    """After a successful algorithm: fit bookkeeping and the caller's result."""
    if method.byte_length() == 0:
        return result
    var source = args[0] if len(args) > 0 else PythonObject(None)
    lifecycle_complete(model, method, source, False, context)
    return lifecycle_result(model, method, result, kind, context)


def decomp_coder_width(model: PythonObject, args: PythonObject, context: PythonObject) raises:
    """SparseCoder.transform: X's width against its dictionary's columns
    (the stateless coder has no fitted n_features_in_ to compare with)."""
    var fields = py_fields(model)
    var dictionary = py_get(fields, "dictionary")
    var expected = lifecycle_x_shape(dictionary, context)
    var shape = lifecycle_x_shape(args[0], context)
    if expected is None or shape is None or len(expected) != 2 or len(shape) != 2:
        return
    if Int(py=shape[1]) != Int(py=expected[1]):
        raise Error("X has " + String(Int(py=shape[1])) + " features, but " + classical_estimator_name(model)
                    + " is expecting " + String(Int(py=expected[1])) + " features as input.")
