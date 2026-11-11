# SPDX-License-Identifier: Apache-2.0
# LinearSVC `fit`, `predict`, `score` and LinearSVR `fit`, `score`
# (python/mojolearn/svm.py), one native call each. The parameter checks of
# `_LinearSVMBase._check_common`, the label encoding, the per-row weights of
# `linear_model._qn_row_weights` (sample_weight times the class weights the
# x_trees binding spreads over the rows), the one `qn_fit_ovr` (or the
# per-target `qn_fit`) call with the same parameter lists, the prediction
# entries `qn_predict_binary` / `qn_predict_multiclass` /
# `qn_decision_function` and the score metrics are the Python methods' own,
# in the same order with the same arguments: the bits do not move.
from std.memory import memcpy
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.admitted_family import admit_family, admit_family_copied, admit_family_meta, admit_family_weights, admitted_array
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem, py_fields
from bindings.classical_arrays import classical_address
from bindings.classical_attributes import classical_connect_numpy
from bindings.family_common import (
    FAMILY_F32P, FAMILY_I64P, family_repr, family_str, family_isinstance, family_call,
    family_array,
)
from bindings.svm_mixture_native import _qn_scores, _qn_scores_native
from core.admitted_weights import admitted_fill
from bindings.class_weight_boundary import CLASS_WEIGHT_SVC
from bindings.class_weight_rows_boundary import class_weight_plan_admitted
from bindings.label_weights_boundary import label_weights_run
from core.label_weights_native_api import LabelWeightsRequest, LABEL_WEIGHTS_MULTIPLY, LABEL_WEIGHTS_ROWS
from providers.python.estimator_binding_pyapi import estimator_binding_mode
from bindings.classical_labels import labels_type_name, classical_encode_labels
from bindings.linear_svc_boundary import LinearSvcWeights, LinearSvcConnection, linear_svc_connection, linear_svc_run_fit, linear_svc_run_query, LinearSvrFitResult, linear_svr_run_fit, LinearSvcPrediction
from core.linear_svc_native_api import LinearSvcOptions, LinearSvcFitRequest, LinearSvcFitPlan, LinearSvcWeightsState, linear_svc_pack
from core.qn_single_fit import QnSingleFitPlan
from core.logistic_native_api import QnFitRequest, QnQueryRequest, QN_NATIVE_BINARY, QN_NATIVE_MULTICLASS, QN_NATIVE_MULTICLASS_AVAILABLE
from core.labels_native_api import LabelsThresholdRequest
from bindings.labels_descriptor_boundary import labels_threshold, labels_argmax_rows
from bindings.svc_common import (
    svc_builtins, svc_field, svc_attr, svc_has, svc_name, svc_float, svc_int, svc_truthy,
    svc_mode, svc_binding, svc_shape_of, svc_labels_1d, svc_sorted_classes,
    svc_decode, svc_r2_sums,
)

from bindings.svc_score_boundary import svc_decode_score, svc_accuracy_native

comptime _PY2MOJO_ROWS = 2


# ---------------------------------------------------------------- parameters


def _linear_check_common(model: PythonObject, fields: PythonObject) raises:
    """`_LinearSVMBase._check_common`."""
    var b = svc_builtins()
    var operator = native_loaded_module("operator")
    var name = svc_name(model)
    var penalty = svc_field(fields, "penalty")
    var loss = svc_field(fields, "loss")
    var c = svc_field(fields, "C")
    if not py_contains(Python.tuple(PythonObject("l1"), PythonObject("l2")), penalty):
        raise Error("Expected penalty to be one of ['l1', 'l2'], got " + family_repr(penalty))
    var losses = svc_attr(model, "_LOSSES")
    if not py_contains(losses, loss):
        raise Error("Expected loss to be one of " + family_repr(family_call(b.list, losses)) + ", got "
                    + family_repr(loss))
    if not Bool(py=family_call(operator.gt, c, PythonObject(0))):
        raise Error("mojolearn " + name + ": C must be positive, got " + family_str(c))
    if svc_truthy(svc_field(fields, "penalized_intercept")):
        raise Error("NotImplementedError: mojolearn " + name + ": penalized_intercept=True is not implemented "
                    + "(the reference appends a column of ones to X and fits without "
                    + "a bias, linear.cu:150-163); the intercept is never penalized here")
    # `_check_qn_solver_fields`
    var tol = svc_field(fields, "tol")
    if not Bool(py=family_call(operator.gt, tol, PythonObject(0))):
        raise Error("mojolearn " + name + ": tol must be positive, got " + family_str(tol))
    var names: List[String] = ["max_iter", "linesearch_max_iter", "lbfgs_memory"]
    for i in range(3):  # small-loop(solver fields: three integer checks)
        var value = svc_field(fields, names[i])
        if (Bool(py=family_call(operator.ne, family_call(b.int, value), value))
                or Bool(py=family_call(operator.lt, value, PythonObject(1)))):
            raise Error("mojolearn " + name + ": " + names[i] + " must be a positive integer, got " + family_str(value))


def _linear_strengths_values(fields: PythonObject) raises -> InlineArray[Float64, 2]:
    var inv_c = 1.0 / svc_float(svc_field(fields, "C"))
    if String(svc_field(fields, "penalty")) == "l1":
        var values: InlineArray[Float64, 2] = [inv_c, 0.0]
        return values^
    var values: InlineArray[Float64, 2] = [0.0, inv_c]
    return values^


def _class_weight_message(model: PythonObject, class_weight: PythonObject) raises -> String:
    return ("The 'class_weight' parameter of " + svc_name(model) + " must be an "
            + "instance of 'dict', a str among {'balanced'} or None. "
            + "Got " + (family_repr(class_weight) if py_type(class_weight) is svc_builtins().str
                        else "a " + labels_type_name(py_type(class_weight))) + " instead.")


def _qn_row_weights(model: PythonObject, fields: PythonObject, sample_weight: PythonObject,
                    class_weight: PythonObject, classes: PythonObject, codes: PythonObject, n: Int,
                    context: PythonObject) raises -> LinearSvcWeights:
    var b = svc_builtins()
    var result = LinearSvcWeights(PythonObject(None), LinearSvcWeightsState())
    if sample_weight is None and class_weight is None:
        return result^
    if sample_weight is None:
        result.state.values = List[Float32](length=n, fill=0)
        result.state.address = Int(result.state.values.unsafe_ptr())
        admitted_fill[DType.float32](FAMILY_F32P(unsafe_from_address=result.state.address), n, Float32(1.0))
    else:
        result.input_pin = admit_family_weights(sample_weight, n, False, context).keep
        result.state.address = Int(py=classical_address(result.input_pin))
    if class_weight is None:
        return result^
    if py_type(class_weight) is b.str:
        if String(class_weight) != "balanced":
            raise Error(_class_weight_message(model, class_weight))
    elif py_type(class_weight) is not b.dict:
        raise Error(_class_weight_message(model, class_weight))
    var xb = estimator_binding_mode(svc_mode(fields), "_mojolearn_x_trees", context)
    var listed = family_call(b.list, classes)
    var count = Int(py=codes.size)
    var k = len(listed)
    var balanced = py_type(class_weight) is b.str
    var plan = class_weight_plan_admitted[CLASS_WEIGHT_SVC](class_weight, listed, codes,
        count, k, balanced, xb, context)
    if plan.all_one:
        return result^
    var class_rows = List[Float32](length=count, fill=0)
    if count:
        _ = label_weights_run[True](xb, LABEL_WEIGHTS_ROWS,
            LabelWeightsRequest(Int(py=codes._addr), Int(plan.words.unsafe_ptr()),
                Int(class_rows.unsafe_ptr()), count, k))
    var output = List[Float32](length=n, fill=0)
    _ = label_weights_run[True](xb, LABEL_WEIGHTS_MULTIPLY,
        LabelWeightsRequest(result.state.address, Int(class_rows.unsafe_ptr()),
            Int(output.unsafe_ptr()), n, 0))
    _ = class_rows
    _ = plan
    _ = result
    result.input_pin = PythonObject(None)
    result.state.values = output^
    result.state.address = Int(result.state.values.unsafe_ptr())
    return result^


def _linear_svr_fit_target(fields: PythonObject, binding: PythonObject, x: PythonObject,
    target: PythonObject, loss: Int, l1: Float64, l2: Float64, grad_tol: Float64,
    change_tol: Float64, svr_eps: Float64, sw: PythonObject, penalty_normalized: Bool,
    context: PythonObject) raises -> LinearSvrFitResult:
    var rows = Int(py=x.shape[0])
    var cols = Int(py=x.shape[1])
    var intercept = svc_truthy(svc_field(fields, "fit_intercept"))
    var weights = family_array(Python.tuple(PythonObject(cols + Int(intercept))), "<f4", context)
    var plan = QnSingleFitPlan()
    # Preserve parameter conversion, optional weight address, then X/y/w
    # address admission in the original single-target fit order.
    var max_iter = svc_int(svc_field(fields, "max_iter"))
    var linesearch = svc_int(svc_field(fields, "linesearch_max_iter"))
    var memory = svc_int(svc_field(fields, "lbfgs_memory"))
    var sw_address = 0 if sw is None else Int(py=classical_address(sw))
    var request = QnFitRequest(Int(py=classical_address(x)), Int(py=classical_address(target)),
        Int(py=classical_address(weights)), 0, sw_address, 0, rows, cols, 1,
        max_iter, linesearch, memory, intercept, penalty_normalized, loss,
        l1, l2, grad_tol, change_tol, svr_eps)
    var connection = linear_svc_connection(binding)
    linear_svr_run_fit(connection, request, plan)
    _ = x
    _ = target
    _ = sw
    _ = weights
    return LinearSvrFitResult(weights, plan^)


# ---------------------------------------------------------------- LinearSVR


def _linear_svr_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises:
    """`LinearSVR.fit(X, y, sample_weight)`; args = (binding, X, y, sample_weight)."""
    var operator = native_loaded_module("operator")
    var binding = args[0]
    var fields = py_fields(model)
    var epsilon = svc_field(fields, "epsilon")
    if not (Bool(py=family_call(operator.le, PythonObject(0.0), epsilon))
            and Bool(py=family_call(operator.lt, epsilon, family_call(svc_builtins().float, PythonObject("inf"))))):
        raise Error("mojolearn LinearSVR: epsilon must be finite and non-negative, got " + family_str(epsilon))
    _linear_check_common(model, fields)
    var xc = admit_family_meta(args[1], 2, "X", context)
    py_setitem(fields, PythonObject("input_copied_"), PythonObject(admit_family_copied(xc, args[1])))
    var x = xc.keep
    var rows = Int(py=x.shape[0])
    var cols = Int(py=x.shape[1])
    # `_target_1d`
    if len(svc_shape_of(args[2], context)) != 1:
        raise Error("mojolearn LinearSVR requires a 1-D y")
    var t = admit_family(args[2], 1, "y", context, False).array
    if Int(py=t.shape[0]) != rows:
        raise Error("mojolearn LinearSVR X and y lengths differ")
    var strengths = _linear_strengths_values(fields)
    var sw = PythonObject(None)
    if args[3] is not None:
        sw = admit_family_weights(args[3], rows, False, context).keep
    var tol = svc_float(svc_field(fields, "tol"))
    var loss = Int(py=svc_attr(model, "_LOSSES")[svc_field(fields, "loss")])
    var result = _linear_svr_fit_target(fields, binding, x, t, loss, strengths[0], strengths[1], tol, 0.1 * tol,
                                    svc_float(epsilon), sw, svc_truthy(svc_attr(model, "penalty_normalized")), context)
    var w = result.weights
    py_setitem(fields, PythonObject("objective_"), PythonObject(result.state.objective()))
    py_setitem(fields, PythonObject("retcode_"), PythonObject(result.state.retcode()))
    py_setitem(fields, PythonObject("_w"), w)
    var coef = family_array(Python.tuple(PythonObject(cols)), "<f4", context)
    memcpy(dest=FAMILY_F32P(unsafe_from_address=Int(py=classical_address(coef))),
           src=FAMILY_F32P(unsafe_from_address=Int(py=classical_address(w))), count=cols)
    _ = coef  # keep the Python object alive through the native call above
    _ = w  # keep the Python object alive through the native call above
    py_setitem(fields, PythonObject("coef_"), coef)
    if svc_truthy(svc_field(fields, "fit_intercept")):
        py_setitem(fields, PythonObject("intercept_"),
                   PythonObject(Float64(FAMILY_F32P(unsafe_from_address=Int(py=classical_address(w)))[cols])))
        _ = w  # keep the Python object alive through the native call above
    else:
        py_setitem(fields, PythonObject("intercept_"), PythonObject(0.0))
    var n_iter = family_array(Python.tuple(PythonObject(1)), "<i8", context)
    FAMILY_I64P(unsafe_from_address=Int(py=classical_address(n_iter)))[0] = Int64(result.state.iterations)
    _ = n_iter  # keep the Python object alive through the native call above
    py_setitem(fields, PythonObject("n_iter_"), n_iter)
    py_setitem(fields, PythonObject("n_features_in_"), PythonObject(cols))
    _ = x^
    _ = t^
    _ = sw^


def _linear_svr_predict(model: PythonObject, binding: PythonObject, source: PythonObject,
                        context: PythonObject) raises -> PythonObject:
    if not svc_has(py_fields(model), "_w"):
        raise Error("mojolearn LinearSVR: call fit before predict")
    return _qn_scores(model, binding, source, 1, context)


def _r2_host(pred: PythonObject, y: PythonObject, context: PythonObject) raises -> PythonObject:
    """`linear_model._r2_host`: `1 - SS_res / SS_tot`, 0.0 for a constant y."""
    var sums = svc_r2_sums(pred, y, context)
    var ss_res = Float64(py=sums[0])
    var ss_tot = Float64(py=sums[1])
    if ss_tot != 0.0:
        return PythonObject(1.0 - ss_res / ss_tot)
    return PythonObject(0.0)


# ---------------------------------------------------------------- LinearSVC


def _linear_svc_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises:
    """`LinearSVC.fit(X, y, sample_weight)`; args = (binding, X, y, sample_weight)."""
    var b = svc_builtins()
    var operator = native_loaded_module("operator")
    var binding = args[0]
    var fields = py_fields(model)
    var multi_class = svc_field(fields, "multi_class")
    if Bool(py=family_call(operator.ne, multi_class, PythonObject("ovr"))):
        raise Error("mojolearn LinearSVC: multi_class must be 'ovr', got " + family_repr(multi_class))
    _linear_check_common(model, fields)
    var xc = admit_family_meta(args[1], 2, "X", context)
    py_setitem(fields, PythonObject("input_copied_"), PythonObject(admit_family_copied(xc, args[1])))
    var x = xc.keep
    var rows = Int(py=x.shape[0])
    var cols = Int(py=x.shape[1])
    if len(svc_shape_of(args[2], context)) != 1:
        raise Error("mojolearn LinearSVC requires a 1-D y")
    var encoded = classical_encode_labels(args[2], context)
    if encoded.n != rows:
        raise Error("mojolearn LinearSVC X and y lengths differ")
    var classes = encoded.classes
    var codes = encoded.codes
    py_setitem(fields, PythonObject("classes_"), classes)
    var n_classes = len(classes)
    if n_classes < 2:
        raise Error("mojolearn LinearSVC: y has one class")
    var strengths = _linear_strengths_values(fields)
    var sw = _qn_row_weights(model, fields, args[3], svc_field(fields, "class_weight"), classes, codes, rows, context)
    # The original OVR selection: binary fits class 1 once.
    var first = 1 if n_classes == 2 else 0
    var n_fits = 1 if n_classes == 2 else n_classes
    var fi = 1 if svc_truthy(svc_field(fields, "fit_intercept")) else 0
    var n_coefs = cols + fi
    var pn = 1 if svc_truthy(svc_attr(model, "penalty_normalized")) else 0
    var tol = svc_float(svc_field(fields, "tol"))
    var loss = Int(py=svc_attr(model, "_LOSSES")[svc_field(fields, "loss")])
    var plan = LinearSvcFitPlan(n_fits, n_coefs)
    var connection = linear_svc_connection(binding)
    plan.storage[0].prepare_info(n_fits)
    var options = LinearSvcOptions(rows, cols, first, n_fits,
        svc_int(svc_field(fields, "max_iter")),
        svc_int(svc_field(fields, "linesearch_max_iter")),
        svc_int(svc_field(fields, "lbfgs_memory")), fi != 0, pn != 0,
        loss, strengths[0], strengths[1], tol, 0.1 * tol)
    var request = LinearSvcFitRequest(Int(py=classical_address(x)),
        Int(py=classical_address(codes)), sw.state.address, options.copy())
    linear_svc_run_fit(connection, request, plan)
    _ = x
    _ = codes
    _ = sw
    var n_iter = plan.iterations()
    # PACK public metadata in the same scalar-versus-list form. These lists
    # are outputs only; the solver and its per-class results use native state.
    var objectives = Python.list()
    var retcodes = Python.list()
    for fit in range(n_fits):  # boundary PACK: one public scalar per model target
        _ = py_call(objectives.append, PythonObject(Float64(plan.storage[0].info[3 * fit])))
        _ = py_call(retcodes.append, PythonObject(Int(plan.storage[0].info[3 * fit + 1])))
    var w = family_array(Python.tuple(PythonObject(n_fits * n_coefs)), "<f4", context)
    var w_address = Int(py=classical_address(w))
    var coef = family_array(Python.tuple(PythonObject(n_fits), PythonObject(cols)), "<f4", context)
    var coef_address = Int(py=classical_address(coef))
    var intercept = family_array(Python.tuple(PythonObject(n_fits)), "<f4", context)
    var intercept_address = Int(py=classical_address(intercept))
    with GILReleased(Python()):
        linear_svc_pack(plan.storage[0], n_fits, cols, fi != 0,
            w_address, coef_address, intercept_address)
    _ = plan
    py_setitem(fields, PythonObject("_w"), w)
    py_setitem(fields, PythonObject("coef_"), coef)
    py_setitem(fields, PythonObject("intercept_"), intercept)
    var it = family_array(Python.tuple(PythonObject(1)), "<i8", context)
    FAMILY_I64P(unsafe_from_address=Int(py=classical_address(it)))[0] = Int64(n_iter)
    _ = it
    py_setitem(fields, PythonObject("n_iter_"), it)
    py_setitem(fields, PythonObject("objective_"), objectives[0] if n_fits == 1 else objectives)
    py_setitem(fields, PythonObject("retcode_"), retcodes[0] if n_fits == 1 else retcodes)
    py_setitem(fields, PythonObject("n_features_in_"), PythonObject(cols))
    _ = x
    _ = sw
    _ = codes
    _ = encoded


def _linear_svc_codes(model: PythonObject, fields: PythonObject, connection: LinearSvcConnection,
    source: PythonObject, operation: Int, n_classes: Int, context: PythonObject) raises -> List[Int64]:
    if not svc_has(fields, "_w"):
        raise Error("mojolearn LinearSVC: call fit first")
    var x = admit_family(source, 2, "X", context, False).array
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    if d != svc_int(svc_field(fields, "n_features_in_")):
        raise Error("mojolearn " + svc_name(model) + " feature count differs from fit")
    var codes = List[Int64](length=n, fill=0)
    var fi = 1 if svc_truthy(svc_field(fields, "fit_intercept")) else 0
    var w = svc_field(fields, "_w")
    if n > 0:
        var request = QnQueryRequest(Int(py=classical_address(x)), Int(py=classical_address(w)),
            Int(codes.unsafe_ptr()), 0, n, d, n_classes if n_classes > 2 else 1, fi != 0, False)
        linear_svc_run_query(connection, operation, request)
    _ = x
    _ = w
    return codes^


def _linear_svc_prediction(model: PythonObject, binding: PythonObject, source: PythonObject,
                        context: PythonObject) raises -> LinearSvcPrediction:
    var fields = py_fields(model)
    var fitted = svc_has(fields, "classes_")
    if fitted:
        var classes = svc_field(fields, "classes_")
        var k = len(classes)
        var connection = linear_svc_connection(binding)
        if k == 2 and (connection.driver.flags() & _PY2MOJO_ROWS) != 0:
            var codes = _linear_svc_codes(model, fields, connection, source, QN_NATIVE_BINARY, 2, context)
            return LinearSvcPrediction(classes, codes^)
        if k > 2 and connection.driver.metadata(QN_NATIVE_MULTICLASS_AVAILABLE) != 0:
            var codes = _linear_svc_codes(model, fields, connection, source, QN_NATIVE_MULTICLASS, k, context)
            return LinearSvcPrediction(classes, codes^)
    if not svc_has(fields, "_w"):
        raise Error("mojolearn LinearSVC: call fit first")
    var classes = svc_field(fields, "classes_")
    var k = len(classes)
    var connection = linear_svc_connection(binding)
    var scores = _qn_scores_native(model, connection, source, 1 if k == 2 else k, context)
    if k != 2 and k < 3:
        # A rebound class table of length 0/1 produced a 1-D decision Array;
        # the old argmax boundary then refused its missing second dimension.
        raise Error("IndexError: tuple index out of range")
    var codes = List[Int64](length=scores.rows, fill=0)
    if k == 2:
        if scores.rows > 0:
            _ = labels_threshold(context, LabelsThresholdRequest(Int(scores.values.unsafe_ptr()),
                Int(codes.unsafe_ptr()), 0, scores.rows, True, 0, 1, 0))
    elif scores.rows > 0:
        labels_argmax_rows(context, False, Int(scores.values.unsafe_ptr()), scores.rows, k, Int(codes.unsafe_ptr()))
    _ = scores
    return LinearSvcPrediction(classes, codes^)


def _linear_svc_predict(model: PythonObject, binding: PythonObject, source: PythonObject,
                        context: PythonObject) raises -> PythonObject:
    var result = _linear_svc_prediction(model, binding, source, context)
    return svc_decode(model, result.classes, result.codes, context)


def svc_linear_operation(model: PythonObject, operation: String, args: PythonObject,
                         context: PythonObject) raises -> PythonObject:
    """The linear machines; args[0] is the resolved estimators binding."""
    classical_connect_numpy(context)
    if operation == "family_svc_linear_svr_fit":
        _linear_svr_fit(model, args, context)
        return model
    if operation == "family_svc_linear_svr_score":
        var pred = _linear_svr_predict(model, args[0], args[1], context)
        return _r2_host(pred, args[2], context)
    if operation == "family_svc_linear_svc_fit":
        _linear_svc_fit(model, args, context)
        return model
    if operation == "family_svc_linear_svc_predict":
        return _linear_svc_predict(model, args[0], args[1], context)
    if operation == "family_svc_linear_svc_score":
        var result = _linear_svc_prediction(model, args[0], args[1], context)
        var prediction = svc_decode_score(model, result.classes, result.codes, context)
        return PythonObject(svc_accuracy_native(args[2], prediction, context))
    raise Error("RuntimeError: unknown linear SVM operation " + operation)
