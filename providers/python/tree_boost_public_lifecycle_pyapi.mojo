# SPDX-License-Identifier: Apache-2.0
# Public boost/embedding lifecycle. Existing trees_boost_* remains the internal
# numerical ABI; this owner replaces the Python _mode validation/output stack.
from std.python import Python, PythonObject
from bindings.py_balanced import py_get, py_type
from bindings.classical_pandas import classical_admit_value
from bindings.classical_validate import classical_check_x, classical_check_y, classical_check_width, classical_estimator_name
from bindings.classical_labels import labels_shape_of
from bindings.classical_exceptions import classical_require_fitted
from bindings.trees_boost_admission import boost_admit
from bindings.trees_public_boost import boost_kind, boost_operation
from bindings.trees_public_mixin_score import boost_score
from bindings.trees_public_auxiliary import embedding_apply
from bindings.parallel_public_result import parallel_public_result
from providers.python.estimator_lifecycle_pyapi import estimator_feature_check, estimator_fit_begin, estimator_fit_end, estimator_fit_complete
from providers.python.estimator_tabular_boundary_pyapi import _target, _refusal
from providers.python.gp_public_lifecycle_pyapi import _gp_classifier_target
from bindings.prep_lda_admission import lda_native_labels


def _boost_classifier_target(source: PythonObject, rows: Int, name: String,
                             context: PythonObject) raises -> PythonObject:
    """The classifier target a boost fit hands to `trees_boost_fit`.

    A rank-1 integer ndarray or Array takes the forest owner's admission
    (`lda_native_labels`, trees_public_lifecycle_native) and is passed on
    unchanged: `trees_encode_labels` reads it with one native call. Every
    other target (lists, tuples, torch, bool, float, object or 2-D arrays) keeps the GP
    label snapshot, which boxes, admits and converts each label as a Python
    object. That snapshot cost about 2.3 s per classifier fit at 20000 rows
    on every column (identity records b25df4e213 -> 0412c8ee2f, train stage:
    trees-bagging-clf 0.9 -> 21 s over 9 cells on the CPU column, the
    regressors unchanged) since 2c61cfe498 moved the boost lifecycle here.
    The labels' values and order are the same either way (the four lanes'
    output hashes are equal across those two records), so no bits change."""
    var cls = py_type(source)
    var fast = False
    var ndarray = py_get(context, "ndarray")
    # A rank-1 <f4/<f8 ndarray or Array takes the same door: lda_native_labels
    # checks NaN/infinity and integrality from its buffer in one native pass,
    # and trees_encode_labels encodes it with one prepare_targets call into
    # float64 classes, the same Python-float classes and codes the snapshot's
    # float list encodes to, so no bits change.
    if ndarray is not None and cls is ndarray:
        var kind = String(source.dtype.kind)
        var typestr = String(source.dtype.str)
        fast = Int(py=source.ndim) == 1 and (kind == "i" or kind == "u" or typestr == "<f4" or typestr == "<f8")
    elif cls is context["array_class"]:
        var typestr = String(source.dtype)
        fast = Int(py=source.ndim) == 1 and (typestr.startswith("<i") or typestr.startswith("<u")
                                             or typestr.startswith("|i") or typestr.startswith("|u")
                                             or typestr == "<f4" or typestr == "<f8")
    if not fast:
        return _gp_classifier_target(source, rows, name, context)
    lda_native_labels(source, context, True, name)
    var target_shape = labels_shape_of(source, context)
    if len(target_shape) == 0:
        raise Error("mojolearn: y must be a sequence of labels")
    if target_shape[0] != rows:
        raise Error("Found input variables with inconsistent numbers of samples: [" + String(rows) + ", "
                    + String(target_shape[0]) + "]")
    return source


def tree_boost_public_operation(model: PythonObject, operation: String, args: PythonObject,
                                context: PythonObject) raises -> PythonObject:
    var method = String(operation[byte=String("trees_public_boost_").byte_length():])
    var fitting = method == "fit" or method == "fit_transform"
    var fields = boost_admit(model, context, fitting)
    classical_admit_value(args, context)
    var nested = Int(py=py_get(fields, "_mojolearn_fitting", PythonObject(0))) > 0
    var kind = boost_kind(model, context)
    var classifier = kind.endswith("_classifier")
    var name = classical_estimator_name(model)
    if not fitting:
        classical_require_fitted(model, False, method, context)
    var x = classical_check_x(args[0], name, context)
    var shape = labels_shape_of(x, context)
    var rows = shape[0]
    var columns = shape[1]
    if not fitting:
        classical_check_width(columns, Int(py=fields["n_features_in_"]), name)
        estimator_feature_check(model, fields, args[0], context)
    var call_args = Python.tuple(x)
    if fitting:
        if kind == "rte":
            _target(args[1], rows, name, context)
        else:
            var y = _boost_classifier_target(args[1], rows, name, context) if classifier else classical_check_y(args[1], rows, name, context)
            call_args = Python.tuple(x, y, args[2])
    elif method == "score":
        call_args = Python.tuple(x, args[1], args[2])
    var result = PythonObject(None)
    if fitting:
        var depth = estimator_fit_begin(fields)
        try:
            result = boost_operation(model, "trees_boost_" + method, call_args, context)
        except error:
            var refused = _refusal(error, True, depth > 0, rows, columns, name)
            raise refused^
        finally:
            if fitting:
                estimator_fit_end(model, depth)
        estimator_fit_complete(model, args[0], classifier, context)
        if method == "fit":
            return result
    elif method == "score":
        return boost_score(model, call_args, context)
    elif method == "apply":
        return embedding_apply(model, call_args, context)
    else:
        try:
            result = boost_operation(model, "trees_boost_" + method, call_args, context)
        except error:
            var refused = _refusal(error, False, nested, rows, columns, name)
            raise refused^
    return parallel_public_result(result, method, PythonObject("classifier" if classifier else "regressor"), context)
