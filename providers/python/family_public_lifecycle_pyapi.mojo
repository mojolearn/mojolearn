# SPDX-License-Identifier: Apache-2.0
# The whole public lifecycle of the family estimators that formerly took the
# `_mode` class-creation wrappers (lane lifecycle-neighbors-cluster-20261028):
# OneClassSVM, KernelPCA, AdditiveChi2Sampler, SkewedChi2Sampler,
# LabelPropagation, LabelSpreading, PageRank, Louvain, SVGP, KMeans, LinearSVC
# and LinearSVR. One native public call runs, in order: model and fitted-state
# admission, the NotFittedError refusal, X validation (rank, emptiness, the
# finite scan scikit-learn's validate_data makes for these estimators), the
# width and feature-name checks, y validation at an outermost fit, fit depth
# bookkeeping, the algorithm owner, the fitted flag, feature_names_in_ and
# public class labels after a fit, and NumPy results (float64 where
# scikit-learn returns float64) for callers outside the package. The numerical
# owners are unchanged; this provider imports none of them.
from std.math import isfinite
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.mode_metadata_native import _fields, _fitted, _class_fields
from bindings.classical_attributes import classical_connect_numpy, _type_mro
from bindings.classical_validate import classical_estimator_name, classical_check_width
from bindings.classical_exceptions import classical_not_fitted_class
from bindings.host_inference_scalars import host_inference_scalar
from bindings.host_sparse_native import host_sparse_format, _sparse_fields
from bindings.family_common import FAMILY_F32P, family_f32_snapshot, family_owned_snapshot
from providers.python.estimator_lifecycle_pyapi import (
    estimator_outside, estimator_feature_check, estimator_fit_begin, estimator_fit_complete,
)
from providers.python.neighbors_public_lifecycle_pyapi import NeighborPublicCall, _target, _public_result


struct FamilyLifecycle(Copyable, Movable):
    """How one public operation is checked. `method` is the public method it
    serves ("" for an operation outside this lifecycle: a property, the
    pickle state, a composed internal call)."""
    var method: String
    var requires_fit: Bool   # False for a stateless transformer (scikit-learn's requires_fit tag)
    var scan: Bool           # the X finite scan scikit-learn's validate_data makes
    var replace: Bool        # hand the owner the admitted dense float32 snapshot of X
    var graph: Bool          # an adjacency input: an (indptr, indices, n) tuple passes through
    var target: Int          # the position of y validated at an outermost fit (-1: none)
    var classifier: Bool     # discrete labels; public ClassLabels classes_ after fit
    var required: Bool       # fit refuses y=None
    var widen: Bool          # float32 result widened exactly to float64 for outside callers

    def __init__(out self, method: String, requires_fit: Bool = True, scan: Bool = False,
                 replace: Bool = True, graph: Bool = False, target: Int = -1,
                 classifier: Bool = False, required: Bool = False, widen: Bool = False):
        self.method = method
        self.requires_fit = requires_fit
        self.scan = scan
        self.replace = replace
        self.graph = graph
        self.target = target
        self.classifier = classifier
        self.required = required
        self.widen = widen


def family_lifecycle_spec(operation: String) -> FamilyLifecycle:
    """The lifecycle of a connected public operation. The positions are the
    owner argument tuple's: 0 the resolved binding, 1 X, then the rest."""
    # OneClassSVM (bindings/svc_ocsvm_native.mojo). sample_weight is not a target.
    if operation == "family_svc_ocsvm_fit":
        return FamilyLifecycle("fit", scan=True)
    if operation == "family_svc_ocsvm_fit_predict":
        return FamilyLifecycle("fit_predict", scan=True)
    if operation == "family_svc_ocsvm_score_samples":
        return FamilyLifecycle("score_samples", scan=True, widen=True)
    if operation == "family_svc_ocsvm_decision":
        return FamilyLifecycle("decision_function", scan=True, widen=True)
    if operation == "family_svc_ocsvm_predict":
        return FamilyLifecycle("predict", scan=True)
    # AdditiveChi2Sampler is stateless (transform needs no fit); a sparse
    # input keeps its sparse result, so X passes through unreplaced when sparse.
    if operation == "family_xn_semi_achi2_fit":
        return FamilyLifecycle("fit", scan=True)
    if operation == "family_xn_semi_achi2_transform":
        return FamilyLifecycle("transform", requires_fit=False, scan=True)
    if operation == "family_xn_semi_achi2_fit_transform":
        return FamilyLifecycle("fit_transform", scan=True)
    if operation == "family_xn_semi_schi2_fit":
        return FamilyLifecycle("fit", scan=True)
    if operation == "family_xn_semi_schi2_transform":
        return FamilyLifecycle("transform", scan=True)
    if operation == "family_xn_semi_schi2_fit_transform":
        return FamilyLifecycle("fit_transform", scan=True)
    # LabelPropagation / LabelSpreading: y (-1 marks unlabeled) at position 2.
    if operation == "family_xn_semi_lp_fit":
        return FamilyLifecycle("fit", scan=True, target=2, classifier=True, required=True)
    if operation == "family_xn_semi_lp_predict_proba":
        return FamilyLifecycle("predict_proba", scan=True)
    if operation == "family_xn_semi_lp_predict":
        return FamilyLifecycle("predict", scan=True)
    if operation == "family_xn_semi_lp_score":
        return FamilyLifecycle("score", scan=True)
    # PageRank / Louvain take an adjacency (dense, scipy.sparse or a CSR tuple).
    if operation == "family_xn_graph_pagerank_fit" or operation == "family_xn_graph_louvain_fit":
        return FamilyLifecycle("fit", replace=False, graph=True)
    if operation == "family_xn_graph_louvain_fit_predict":
        return FamilyLifecycle("fit_predict", replace=False, graph=True)
    # SVGP (predict_f / predict_y are latent-function queries outside this table).
    if operation == "family_xn_graph_svgp_fit":
        return FamilyLifecycle("fit", target=2, required=True)
    if operation == "family_xn_graph_svgp_predict":
        return FamilyLifecycle("predict", widen=True)
    if operation == "family_xn_graph_svgp_score":
        return FamilyLifecycle("score")
    # KMeans (bindings/kmeans_native.mojo): its own by-name non-finite refusal;
    # X reaches the owner as the caller passed it (array init receives the rows).
    if operation == "family_kmeans_fit":
        return FamilyLifecycle("fit", replace=False)
    if operation == "family_kmeans_fit_predict":
        return FamilyLifecycle("fit_predict", replace=False)
    if operation == "family_kmeans_fit_transform":
        return FamilyLifecycle("fit_transform", replace=False)
    if operation == "family_kmeans_predict":
        return FamilyLifecycle("predict", replace=False)
    if operation == "family_kmeans_transform":
        return FamilyLifecycle("transform", replace=False)
    # LinearSVC / LinearSVR (bindings/svc_linear_native.mojo, svm_mixture_native.mojo).
    if operation == "family_svc_linear_svc_fit":
        return FamilyLifecycle("fit", scan=True, replace=False, target=2, classifier=True, required=True)
    if operation == "family_svm_linear_svc_decision":
        return FamilyLifecycle("decision_function", scan=True, replace=False)
    if operation == "family_svc_linear_svc_predict":
        return FamilyLifecycle("predict", scan=True, replace=False)
    if operation == "family_svc_linear_svc_score":
        return FamilyLifecycle("score", scan=True, replace=False)
    if operation == "family_svc_linear_svr_fit":
        return FamilyLifecycle("fit", scan=True, replace=False, target=2, required=True)
    if operation == "family_svm_linear_svr_predict":
        return FamilyLifecycle("predict", scan=True, replace=False, widen=True)
    if operation == "family_svc_linear_svr_score":
        return FamilyLifecycle("score", scan=True, replace=False)
    return FamilyLifecycle("")


def kpca_lifecycle_spec(operation: String) -> FamilyLifecycle:
    """KernelPCA through the decomposition door (X at position 0)."""
    if operation == "fit":
        return FamilyLifecycle("fit", scan=True)
    if operation == "fit_transform":
        return FamilyLifecycle("fit_transform", scan=True)
    if operation == "transform":
        return FamilyLifecycle("transform", scan=True)
    return FamilyLifecycle("")


def _precomputed(fields: PythonObject) raises -> Bool:
    # scikit-learn skips the width check for a precomputed kernel or metric:
    # the query's columns are the training rows.
    var b = native_loaded_module("builtins")
    for key in ["kernel", "metric", "affinity"]:  # small-loop(three fixed precomputed-mode parameters)
        var value = py_get(fields, key)
        if py_type(value) is b.str and String(py=value) == "precomputed":
            return True
    return False


def family_lifecycle_begin(model: PythonObject, operation: String, spec: FamilyLifecycle,
                           args: PythonObject, x_index: Int, context: PythonObject) raises -> NeighborPublicCall:
    """Admission and validation before the owner runs; the returned call
    carries the owner's argument list (X replaced by its admitted snapshot
    where `spec.replace`, y by its validated form)."""
    classical_connect_numpy(context)
    var b = native_loaded_module("builtins")
    if py_type(py_type(model)) is not b.type:
        raise Error("TypeError: native estimator operations refuse Python metaclass hooks")
    var fields = _fields(model)
    var method = spec.method
    var fitting = method == "fit" or method == "fit_predict" or method == "fit_transform"
    var depth = py_get(fields, "_mojolearn_fitting", PythonObject(0))
    if py_type(depth) is not b.int:
        raise Error("TypeError: native fit depth must be an exact integer")
    var fitted = py_get(fields, "_mojolearn_fitted", PythonObject(False))
    if py_type(fitted) is not b.bool and py_type(fitted) is not b.int:
        raise Error("TypeError: native fitted flags cannot invoke Python boolean hooks")
    var ancestry = _type_mro(py_type(model))
    for i in range(len(ancestry)):  # driver-loop: no completion callback may execute from a native fit
        if py_contains(_class_fields(ancestry[i]), "_mojolearn_after_fit"):
            raise Error("TypeError: native fitting cannot invoke Python completion callbacks")
    if py_contains(fields, "_mojolearn_after_fit"):
        raise Error("TypeError: native fitting cannot invoke Python completion callbacks")
    var name = classical_estimator_name(model)
    if spec.requires_fit and not fitting and not Bool(py=_fitted(model)):
        _ = classical_not_fitted_class(context)
        raise Error("NotFittedError: This " + name + " instance is not fitted yet. Call 'fit' with "
                    + "appropriate arguments before using " + method + ".")
    var source = args[x_index]
    var values = Python.list()
    for i in range(len(args)):  # small-loop(fixed operation argument envelope)
        _ = py_call(values.append, args[i])
    var rows = -1
    var columns = -1
    if host_sparse_format(source).byte_length() > 0:
        # Sparse X keeps its own owner path (densified there, or a sparse
        # result); only its declared shape is read here.
        var shape = _sparse_fields(source)["_shape"]
        if py_type(shape) is not b.tuple or len(shape) != 2 or py_type(shape[0]) is not b.int or py_type(shape[1]) is not b.int:
            raise Error("TypeError: sparse input shape requires native dimensions")
        rows = Int(py=shape[0])
        columns = Int(py=shape[1])
    elif not (spec.graph and py_type(source) is b.tuple):
        var snapshot = family_f32_snapshot(source, 2, "X", context, True)
        rows = snapshot.shape[0]
        columns = snapshot.shape[1]
        if rows == 0:
            raise Error("Found array with 0 sample(s) (shape=(0, " + String(columns) + ")) while a minimum of 1 is required by " + name + ".")
        if columns == 0 and not spec.graph:
            raise Error("Found array with 0 feature(s) (shape=(" + String(rows) + ", 0)) while a minimum of 1 is required by " + name + ".")
        if spec.scan:
            var data = FAMILY_F32P(unsafe_from_address=snapshot.source.addr)
            for i in range(snapshot.count):  # driver-loop: public finite-input validation
                if not isfinite(data[i]):
                    raise Error("Input X contains NaN or infinity (" + name + ").")
        if spec.replace:
            py_setitem(values, PythonObject(x_index), family_owned_snapshot(snapshot, context))
    if not fitting and Int(py=depth) == 0 and columns >= 0 and not spec.graph and not _precomputed(fields):
        var width = py_get(fields, "n_features_in_")
        if py_type(width) is b.int:
            host_inference_scalar(width, context)
            classical_check_width(columns, Int(py=width), name)
            # scikit-learn's feature-name check runs after a fit only: a
            # stateless transform before any fit warns about nothing.
            estimator_feature_check(model, fields, source, context)
    if fitting and Int(py=depth) == 0 and spec.target >= 0:
        if spec.target < len(args):
            var target = _target(args[spec.target], rows, name, spec.classifier, spec.required, False, context)
            py_setitem(values, PythonObject(spec.target), target)
        elif spec.required:
            raise Error(name + " requires y to be passed, but the target y is None.")
    var previous = estimator_fit_begin(fields) if fitting else Int(py=depth)
    # The owners index an exact argument tuple, as estimator_arguments made it.
    return NeighborPublicCall(py_call(b.tuple, values), fields, source, method, fitting, estimator_outside(),
                              previous, rows, columns)


def family_lifecycle_finish(model: PythonObject, spec: FamilyLifecycle, result: PythonObject,
                            call: NeighborPublicCall, context: PythonObject) raises -> PythonObject:
    """Fit bookkeeping after a completed owner call, then the public result."""
    if call.fitting:
        # A nested set_params may have replaced the dictionary's contents;
        # estimator_fit_complete reads the live dictionary.
        estimator_fit_complete(model, call.source, spec.classifier, context)
        var fields = _fields(model)
        if not py_contains(fields, "n_features_in_") and call.columns >= 0:
            py_setitem(fields, "n_features_in_", PythonObject(call.columns))
    if call.outside and result is not model and call.method != "score":
        # A regressor's predict is float64 alongside NumPy (scikit-learn's
        # dtype); an outlier detector's scores are float64 for every caller.
        var widen = spec.widen and (call.method != "predict" or py_get(context, "ndarray") is not None)
        return _public_result(result, widen, context)
    return result
