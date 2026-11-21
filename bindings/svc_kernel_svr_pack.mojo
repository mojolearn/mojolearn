# SPDX-License-Identifier: Apache-2.0
# PACK of SVR `fit`, `predict` and `score`
# (bindings/svc_kernel_native_door.mojo sequences the operations): the
# compact public arrays cut from the fit's worst-case buffers, the fitted
# fields published together after RUN (a failed fit leaves the estimator's
# previous fitted state whole), the public prediction vector, and R^2.
from std.python import Python, PythonObject
from bindings.py_balanced import py_setitem
from bindings.family_common_pack import family_array
from bindings.svc_common_pack import svc_prefix, svc_scalar_array
from bindings.svc_common_door import svc_r2_sums
from bindings.svc_common_run import svc_round_f32
from bindings.svm_keys_unpack import svm_keys
from bindings.svc_kernel_rows_unpack import _svm_fit_refuse
from bindings.svc_kernel_svr_unpack import SvrFitUnpacked


def svr_fit_pack(unpacked: SvrFitUnpacked, context: PythonObject) raises:
    """The fitted fields of `SVR.fit`. A fit X the device entry flagged is
    refused first, in admit_x's wording, before anything is published."""
    var n_cols = unpacked.request.features
    var n_support = unpacked.plan.count
    var gamma = unpacked.gamma
    if unpacked.device_form:
        # X's finite and range words and the gamma solved with (lane a8a)
        _svm_fit_refuse(Int(unpacked.plan.info[3]), Int(unpacked.plan.info[4]), unpacked.fold, "SVR")
        gamma = unpacked.plan.info[5]
    var intercept = svc_scalar_array(svc_round_f32(unpacked.plan.info[0]), "<f4", context)
    var support = svc_prefix(unpacked.request.support, n_support, Python.tuple(PythonObject(n_support)), "<i4", context)
    var vectors = svc_prefix(unpacked.request.vectors, n_support * n_cols,
                             Python.tuple(PythonObject(n_support), PythonObject(n_cols)), "<f4", context)
    var dual = svc_prefix(unpacked.request.dual, n_support, Python.tuple(PythonObject(1), PythonObject(n_support)), "<f4", context)
    var fields = unpacked.fields
    var nk = svm_keys()
    py_setitem(fields, nk[].k_input_copied_, PythonObject(unpacked.copied))
    py_setitem(fields, nk[].k_n_features_in_, PythonObject(n_cols))
    py_setitem(fields, nk[].k_n_support_, PythonObject(n_support))
    py_setitem(fields, nk[].k_n_iter_, PythonObject(Int(unpacked.plan.info[2])))
    py_setitem(fields, nk[].k_intercept_, intercept)
    py_setitem(fields, nk[].k_support_, support)
    py_setitem(fields, nk[].k_support_vectors_, vectors)
    py_setitem(fields, nk[].k_dual_coef_, dual)
    py_setitem(fields, nk[].k__gamma, PythonObject(gamma))


def svr_query_output(rows: Int, context: PythonObject) raises -> PythonObject:
    """The public float32 prediction vector RUN writes in place."""
    return family_array(Python.tuple(PythonObject(rows)), "<f4", context)


def svr_score_pack(pred: PythonObject, y: PythonObject, context: PythonObject) raises -> PythonObject:
    """`SVR.score`: R^2, 1.0 for a perfect constant fit, 0.0 otherwise."""
    var sums = svc_r2_sums(pred, y, context)
    var ss_res = Float64(py=sums[0])
    var ss_tot = Float64(py=sums[1])
    if ss_tot == 0.0:
        return PythonObject(1.0 if ss_res == 0.0 else 0.0)
    return PythonObject(1.0 - ss_res / ss_tot)
