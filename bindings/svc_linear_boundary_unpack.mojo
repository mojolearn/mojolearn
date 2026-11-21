# SPDX-License-Identifier: Apache-2.0
"""UNPACK of LinearSVC `fit`, `predict`, `score` and LinearSVR `fit`
(python/mojolearn/svm.py): the estimator's fields and the caller's arrays
become typed requests (core/linear_svc_native_api.mojo, core/logistic_native_api.mojo)
and the pins that keep every addressed buffer alive until PACK.

The parameter checks of `_LinearSVMBase._check_common`, the label encoding,
the per-row weights of `linear_model._qn_row_weights` (sample_weight times
the class weights the x_trees programs of the ESTIMATOR'S device spread over
the rows) and the request words are the Python methods' own, in the same
order with the same arguments: the bits do not move. RUN is the typed driver
call of bindings/linear_svc_boundary_unpack.mojo (`linear_svc_run_fit`,
`linear_svr_run_fit`, `linear_svc_run_query`, the GIL released); PACK is
bindings/svc_linear_boundary_pack.mojo; the door
(bindings/svc_linear_native_door.mojo) only sequences the three.

Two fields are published during UNPACK, where the Python methods published
them: `input_copied_` after X is admitted and `classes_` after y is encoded
(both before the refusals that follow them).
"""
from std.python import Python, PythonObject
from bindings.admitted_family_door import admit_family, admit_family_copied, admit_family_meta, admit_family_weights_array
from providers.python.array_metadata_pyapi_door import ArrayPin
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_type, py_contains, py_setitem, py_fields
from bindings.classical_arrays_door import classical_address
from bindings.family_common_unpack import (
    FAMILY_F32P,
    family_repr,
    family_str,
)
from bindings.family_common_pack import (
    family_call,
    family_array,
)
from core.admitted_weights import admitted_fill
from bindings.class_weight_boundary_unpack import CLASS_WEIGHT_SVC
from bindings.class_weight_rows_boundary_pack import class_weight_plan_admitted
from bindings.label_weights_boundary_unpack import label_weights_run
from core.label_weights_native_api import LabelWeightsRequest, LABEL_WEIGHTS_MULTIPLY, LABEL_WEIGHTS_ROWS
from providers.python.estimator_binding_pyapi_door import estimator_binding_routed
from bindings.classical_labels_door import labels_type_name, classical_encode_labels
from bindings.linear_svc_boundary_unpack import LinearSvcWeights, LinearSvcConnection, linear_svc_connection
from core.linear_svc_native_api import LinearSvcOptions, LinearSvcFitRequest, LinearSvcWeightsState
from core.logistic_native_api import QnFitRequest, QnQueryRequest
from bindings.svc_common_unpack import (
    svc_builtins, svc_field, svc_attr, svc_has, svc_name, svc_float, svc_int, svc_truthy, svc_mode,
    svc_shape_of,
)
from bindings.svm_keys_unpack import svm_keys


# ---------------------------------------------------------------- parameters


def _linear_check_common(model: PythonObject, fields: PythonObject) raises:
    """`_LinearSVMBase._check_common`."""
    var b = svc_builtins()
    var operator = native_loaded_module("operator")
    var name = svc_name(model)
    var nk = svm_keys()
    var penalty = svc_field(fields, nk[].k_penalty, "penalty")
    var loss = svc_field(fields, nk[].k_loss, "loss")
    var c = svc_field(fields, nk[].k_C, "C")
    if not py_contains(Python.tuple(PythonObject("l1"), PythonObject("l2")), penalty):
        raise Error("Expected penalty to be one of ['l1', 'l2'], got " + family_repr(penalty))
    var losses = svc_attr(model, nk[].k__LOSSES)
    if not py_contains(losses, loss):
        raise Error("Expected loss to be one of " + family_repr(family_call(b.list, losses)) + ", got "
                    + family_repr(loss))
    if not Bool(py=family_call(operator.gt, c, PythonObject(0))):
        raise Error("mojolearn " + name + ": C must be positive, got " + family_str(c))
    if svc_truthy(svc_field(fields, nk[].k_penalized_intercept, "penalized_intercept")):
        raise Error("NotImplementedError: mojolearn " + name + ": penalized_intercept=True is not implemented "
                    + "(the reference appends a column of ones to X and fits without "
                    + "a bias, linear.cu:150-163); the intercept is never penalized here")
    # `_check_qn_solver_fields`
    var tol = svc_field(fields, nk[].k_tol, "tol")
    if not Bool(py=family_call(operator.gt, tol, PythonObject(0))):
        raise Error("mojolearn " + name + ": tol must be positive, got " + family_str(tol))
    var names: List[String] = ["max_iter", "linesearch_max_iter", "lbfgs_memory"]
    for i in range(3):  # small-loop(solver fields: three integer checks)
        var key = nk[].k_max_iter if i == 0 else (nk[].k_linesearch_max_iter if i == 1 else nk[].k_lbfgs_memory)
        var value = svc_field(fields, key, names[i])
        if (Bool(py=family_call(operator.ne, family_call(b.int, value), value))
                or Bool(py=family_call(operator.lt, value, PythonObject(1)))):
            raise Error("mojolearn " + name + ": " + names[i] + " must be a positive integer, got " + family_str(value))


def _linear_strengths_values(fields: PythonObject) raises -> Array[Float64, 2]:
    var nk = svm_keys()
    var inv_c = 1.0 / svc_float(svc_field(fields, nk[].k_C, "C"))
    if String(svc_field(fields, nk[].k_penalty, "penalty")) == "l1":
        var values: Array[Float64, 2] = [inv_c, 0.0]
        return values^
    var values: Array[Float64, 2] = [0.0, inv_c]
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
        result.input_pin = admit_family_weights_array(sample_weight, n, False, context)
        result.state.address = Int(py=classical_address(result.input_pin))
    if class_weight is None:
        return result^
    if py_type(class_weight) is b.str:
        if String(class_weight) != "balanced":
            raise Error(_class_weight_message(model, class_weight))
    elif py_type(class_weight) is not b.dict:
        raise Error(_class_weight_message(model, class_weight))
    # The class-weight programs of the ESTIMATOR'S device (its binding route), not the process set.
    var xb = estimator_binding_routed(model, svc_mode(fields), "_mojolearn_x_trees", context)
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


# ---------------------------------------------------------------- LinearSVR fit


@fieldwise_init
struct LinearSvrFitUnpacked(Movable):
    """A LinearSVR fit: the typed request and the Python owners of every
    buffer it addresses (X, y, the optional sample weights, the coefficient
    store the solver writes), held until PACK."""
    var fields: PythonObject
    var x: PythonObject
    var target: PythonObject
    var sample_weight: PythonObject
    var weights: PythonObject
    var connection: LinearSvcConnection
    var request: QnFitRequest
    var cols: Int
    var intercept: Bool


def linear_svr_fit_unpack(model: PythonObject, args: PythonObject, context: PythonObject) raises -> LinearSvrFitUnpacked:
    """`LinearSVR.fit(X, y, sample_weight)`; args = (binding, X, y, sample_weight)."""
    var operator = native_loaded_module("operator")
    var binding = args[0]
    var fields = py_fields(model)
    var nk = svm_keys()
    var epsilon = svc_field(fields, nk[].k_epsilon, "epsilon")
    if not (Bool(py=family_call(operator.le, PythonObject(0.0), epsilon))
            and Bool(py=family_call(operator.lt, epsilon, family_call(svc_builtins().float, PythonObject("inf"))))):
        raise Error("mojolearn LinearSVR: epsilon must be finite and non-negative, got " + family_str(epsilon))
    _linear_check_common(model, fields)
    var xc_pin = ArrayPin()
    var xc = admit_family_meta(args[1], 2, "X", context, xc_pin)
    py_setitem(fields, nk[].k_input_copied_, PythonObject(admit_family_copied(xc, args[1])))
    var x = xc_pin.keep
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
        sw = admit_family_weights_array(args[3], rows, False, context)
    var tol = svc_float(svc_field(fields, nk[].k_tol, "tol"))
    var loss = Int(py=svc_attr(model, nk[].k__LOSSES)[svc_field(fields, nk[].k_loss, "loss")])
    var svr_eps = svc_float(epsilon)
    var penalty_normalized = svc_truthy(svc_attr(model, nk[].k_penalty_normalized))
    var intercept = svc_truthy(svc_field(fields, nk[].k_fit_intercept, "fit_intercept"))
    var weights = family_array(Python.tuple(PythonObject(cols + Int(intercept))), "<f4", context)
    # Preserve parameter conversion, optional weight address, then X/y/w
    # address admission in the original single-target fit order.
    var max_iter = svc_int(svc_field(fields, nk[].k_max_iter, "max_iter"))
    var linesearch = svc_int(svc_field(fields, nk[].k_linesearch_max_iter, "linesearch_max_iter"))
    var memory = svc_int(svc_field(fields, nk[].k_lbfgs_memory, "lbfgs_memory"))
    var sw_address = 0 if sw is None else Int(py=classical_address(sw))
    var request = QnFitRequest(Int(py=classical_address(x)), Int(py=classical_address(t)),
        Int(py=classical_address(weights)), 0, sw_address, 0, rows, cols, 1,
        max_iter, linesearch, memory, intercept, penalty_normalized, loss,
        strengths[0], strengths[1], tol, 0.1 * tol, svr_eps)
    var connection = linear_svc_connection(binding)
    return LinearSvrFitUnpacked(fields, x, t, sw, weights, connection^, request^, cols, intercept)


# ---------------------------------------------------------------- LinearSVC fit


@fieldwise_init
struct LinearSvcFitUnpacked(Movable):
    """A LinearSVC fit: the typed request and the owners of every buffer it
    addresses (X, the int32 codes, the row weights: an admitted input or
    native storage), held until PACK."""
    var fields: PythonObject
    var x: PythonObject
    var classes: PythonObject
    var codes: PythonObject
    var weights: LinearSvcWeights
    var connection: LinearSvcConnection
    var request: LinearSvcFitRequest
    var n_fits: Int
    var n_coefs: Int
    var cols: Int
    var intercept: Bool


def linear_svc_fit_unpack(model: PythonObject, args: PythonObject, context: PythonObject) raises -> LinearSvcFitUnpacked:
    """`LinearSVC.fit(X, y, sample_weight)`; args = (binding, X, y, sample_weight)."""
    var operator = native_loaded_module("operator")
    var binding = args[0]
    var fields = py_fields(model)
    var nk = svm_keys()
    var multi_class = svc_field(fields, nk[].k_multi_class, "multi_class")
    if Bool(py=family_call(operator.ne, multi_class, PythonObject("ovr"))):
        raise Error("mojolearn LinearSVC: multi_class must be 'ovr', got " + family_repr(multi_class))
    _linear_check_common(model, fields)
    var xc_pin = ArrayPin()
    var xc = admit_family_meta(args[1], 2, "X", context, xc_pin)
    py_setitem(fields, nk[].k_input_copied_, PythonObject(admit_family_copied(xc, args[1])))
    var x = xc_pin.keep
    var rows = Int(py=x.shape[0])
    var cols = Int(py=x.shape[1])
    if len(svc_shape_of(args[2], context)) != 1:
        raise Error("mojolearn LinearSVC requires a 1-D y")
    var encoded = classical_encode_labels(args[2], context)
    if encoded.n != rows:
        raise Error("mojolearn LinearSVC X and y lengths differ")
    var classes = encoded.classes
    var codes = encoded.codes
    py_setitem(fields, nk[].k_classes_, classes)
    var n_classes = len(classes)
    if n_classes < 2:
        raise Error("mojolearn LinearSVC: y has one class")
    var strengths = _linear_strengths_values(fields)
    var sw = _qn_row_weights(model, fields, args[3], svc_field(fields, nk[].k_class_weight, "class_weight"), classes, codes, rows, context)
    # The original OVR selection: binary fits class 1 once.
    var first = 1 if n_classes == 2 else 0
    var n_fits = 1 if n_classes == 2 else n_classes
    var fi = 1 if svc_truthy(svc_field(fields, nk[].k_fit_intercept, "fit_intercept")) else 0
    var n_coefs = cols + fi
    var pn = 1 if svc_truthy(svc_attr(model, nk[].k_penalty_normalized)) else 0
    var tol = svc_float(svc_field(fields, nk[].k_tol, "tol"))
    var loss = Int(py=svc_attr(model, nk[].k__LOSSES)[svc_field(fields, nk[].k_loss, "loss")])
    var connection = linear_svc_connection(binding)
    var options = LinearSvcOptions(rows, cols, first, n_fits,
        svc_int(svc_field(fields, nk[].k_max_iter, "max_iter")),
        svc_int(svc_field(fields, nk[].k_linesearch_max_iter, "linesearch_max_iter")),
        svc_int(svc_field(fields, nk[].k_lbfgs_memory, "lbfgs_memory")), fi != 0, pn != 0,
        loss, strengths[0], strengths[1], tol, 0.1 * tol)
    var request = LinearSvcFitRequest(Int(py=classical_address(x)),
        Int(py=classical_address(codes)), sw.state.address, options.copy())
    _ = encoded
    return LinearSvcFitUnpacked(fields, x, classes, codes, sw^, connection^, request^, n_fits, n_coefs, cols, fi != 0)


# ---------------------------------------------------------------- LinearSVC query


@fieldwise_init
struct LinearSvcCodesUnpacked(Movable):
    """A LinearSVC class-code query: the typed request, the native code
    store it writes (one int64 per row) and the owners of X and of the
    fitted coefficients. `rows == 0` runs nothing."""
    var x: PythonObject
    var w: PythonObject
    var request: QnQueryRequest
    var codes: List[Int64]
    var rows: Int


def linear_svc_codes_unpack(model: PythonObject, fields: PythonObject, source: PythonObject,
    n_classes: Int, context: PythonObject) raises -> LinearSvcCodesUnpacked:
    var nk = svm_keys()
    if not svc_has(fields, nk[].k__w):
        raise Error("mojolearn LinearSVC: call fit first")
    var x = admit_family(source, 2, "X", context, False).array
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    if d != svc_int(svc_field(fields, nk[].k_n_features_in_, "n_features_in_")):
        raise Error("mojolearn " + svc_name(model) + " feature count differs from fit")
    var codes = List[Int64](length=n, fill=0)
    var fi = 1 if svc_truthy(svc_field(fields, nk[].k_fit_intercept, "fit_intercept")) else 0
    var w = svc_field(fields, nk[].k__w, "_w")
    var request = QnQueryRequest(0, 0, 0, 0, 0, d, n_classes if n_classes > 2 else 1, fi != 0, False)
    if n > 0:
        request = QnQueryRequest(Int(py=classical_address(x)), Int(py=classical_address(w)),
            Int(codes.unsafe_ptr()), 0, n, d, n_classes if n_classes > 2 else 1, fi != 0, False)
    return LinearSvcCodesUnpacked(x, w, request^, codes^, n)
