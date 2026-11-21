# SPDX-License-Identifier: Apache-2.0
"""PACK of LinearSVC `fit`, `predict`, `score` and LinearSVR `fit`, `score`
(python/mojolearn/svm.py): the typed fit state becomes the fitted public
fields (`_w`, `coef_`, `intercept_`, `n_iter_`, `objective_`, `retcode_`,
`n_features_in_`) and a typed prediction becomes the caller's labels or
score. UNPACK is bindings/svc_linear_boundary_unpack.mojo; the same fields,
in the same order, with the same values as the Python methods published.
"""
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.py_balanced import py_call, py_setitem
from bindings.classical_arrays_door import classical_address
from bindings.family_common_unpack import FAMILY_F32P, FAMILY_I64P
from bindings.family_common_pack import family_array
from bindings.linear_svc_boundary_unpack import LinearSvcPrediction
from bindings.svc_linear_boundary_unpack import LinearSvrFitUnpacked, LinearSvcFitUnpacked
from core.linear_svc_native_api import LinearSvcFitPlan, linear_svc_pack
from core.qn_single_fit import QnSingleFitPlan
from bindings.svc_common_pack import svc_decode
from bindings.svc_common_door import svc_r2_sums
from bindings.labels_descriptor_boundary_door import labels_scope
from bindings.svc_score_boundary_door import svc_decode_score, svc_accuracy_native
from bindings.svm_keys_unpack import svm_keys


def linear_svr_fit_pack(unpacked: LinearSvrFitUnpacked, plan: QnSingleFitPlan, context: PythonObject) raises:
    """Publish a finished LinearSVR fit."""
    var nk = svm_keys()
    var fields = unpacked.fields
    var cols = unpacked.cols
    var w = unpacked.weights
    py_setitem(fields, nk[].k_objective_, PythonObject(plan.objective()))
    py_setitem(fields, nk[].k_retcode_, PythonObject(plan.retcode()))
    py_setitem(fields, nk[].k__w, w)
    var coef = family_array(Python.tuple(PythonObject(cols)), "<f4", context)
    unsafe_memcpy(dest=FAMILY_F32P(unsafe_from_address=Int(py=classical_address(coef))),
           src=FAMILY_F32P(unsafe_from_address=Int(py=classical_address(w))), count=cols)
    _ = coef  # keep the Python object alive through the native call above
    _ = w  # keep the Python object alive through the native call above
    py_setitem(fields, nk[].k_coef_, coef)
    if unpacked.intercept:
        py_setitem(fields, nk[].k_intercept_,
                   PythonObject(Float64(FAMILY_F32P(unsafe_from_address=Int(py=classical_address(w)))[cols])))
        _ = w  # keep the Python object alive through the native call above
    else:
        py_setitem(fields, nk[].k_intercept_, PythonObject(0.0))
    var n_iter = family_array(Python.tuple(PythonObject(1)), "<i8", context)
    FAMILY_I64P(unsafe_from_address=Int(py=classical_address(n_iter)))[0] = Int64(plan.iterations)
    _ = n_iter  # keep the Python object alive through the native call above
    py_setitem(fields, nk[].k_n_iter_, n_iter)
    py_setitem(fields, nk[].k_n_features_in_, PythonObject(cols))


def linear_svc_fit_pack(unpacked: LinearSvcFitUnpacked, plan: LinearSvcFitPlan, context: PythonObject) raises:
    """Publish a finished LinearSVC fit."""
    var nk = svm_keys()
    var fields = unpacked.fields
    var n_fits = unpacked.n_fits
    var n_coefs = unpacked.n_coefs
    var cols = unpacked.cols
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
        linear_svc_pack(plan.storage[0], n_fits, cols, unpacked.intercept,
            w_address, coef_address, intercept_address)
    py_setitem(fields, nk[].k__w, w)
    py_setitem(fields, nk[].k_coef_, coef)
    py_setitem(fields, nk[].k_intercept_, intercept)
    var it = family_array(Python.tuple(PythonObject(1)), "<i8", context)
    FAMILY_I64P(unsafe_from_address=Int(py=classical_address(it)))[0] = Int64(n_iter)
    _ = it
    py_setitem(fields, nk[].k_n_iter_, it)
    py_setitem(fields, nk[].k_objective_, objectives[0] if n_fits == 1 else objectives)
    py_setitem(fields, nk[].k_retcode_, retcodes[0] if n_fits == 1 else retcodes)
    py_setitem(fields, nk[].k_n_features_in_, PythonObject(cols))


def linear_svc_predict_pack(model: PythonObject, prediction: LinearSvcPrediction,
                            context: PythonObject) raises -> PythonObject:
    """The predicted class codes as the caller's labels."""
    return svc_decode(model, prediction.classes, prediction.codes, context)


def linear_svc_score_pack(model: PythonObject, prediction: LinearSvcPrediction, truth: PythonObject,
                          context: PythonObject) raises -> PythonObject:
    """Accuracy of the predicted class codes against `truth`, counted on the
    metrics image of the estimator's device."""
    # The label steps run on the estimator's device (labels_scope), read once;
    # `context` stays the family context its binding route compares.
    var label_context = labels_scope(model, context)
    var decoded = svc_decode_score(model, prediction.classes, prediction.codes, label_context)
    return PythonObject(svc_accuracy_native(model, truth, decoded, context, label_context))


def linear_svr_score_pack(pred: PythonObject, y: PythonObject, context: PythonObject) raises -> PythonObject:
    """`linear_model._r2_host`: `1 - SS_res / SS_tot`, 0.0 for a constant y."""
    var sums = svc_r2_sums(pred, y, context)
    var ss_res = Float64(py=sums[0])
    var ss_tot = Float64(py=sums[1])
    if ss_tot != 0.0:
        return PythonObject(1.0 - ss_res / ss_tot)
    return PythonObject(0.0)
