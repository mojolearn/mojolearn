# SPDX-License-Identifier: Apache-2.0
# PACK of the cross-validated fits (RidgeCV, LassoCV, ElasticNetCV,
# LogisticRegressionCV): the float32 result words of their RUN
# (x_linear/door_run.mojo) become the fitted attributes (the names, types,
# shapes and order the Python wrote), one function per public operation and
# RidgeCV path; nothing is published before the fit returned. UNPACK is
# bindings/x_linear_cv_unpack.mojo.
from std.python import Python, PythonObject
from x_linear.door_run import xl_ridge_singular
from bindings.x_linear_common_unpack import _b, _name, _xl_word, _xl_float, _xl_count, _xl_slice, _xl_reals
from bindings.x_linear_regressors_pack import _xl_set_linear
from bindings.x_linear_cv_unpack import XlRidgeCv, XlEnetCvFit, XlLogCvFit
from bindings.py_balanced import py_call, py_setitem
from bindings.x_linear_keys_unpack import x_linear_keys
from bindings.classical_labels_door import labels_address


# ------------------------------------------------------------------ RidgeCV


def xl_ridge_singular_refuse(model: PythonObject, words: PythonObject, d: Int) raises:
    """Refuse a ridge system that does not factor even in float-float, in
    the fit's words (`words`: that fit's result; its alpha is word d + 1)."""
    raise Error("mojolearn " + _name(model) + ": X'X + alpha I is singular even in float-float (alpha="
                + String(py=py_call(_b().format, _xl_float(words, d + 1), PythonObject("g")))
                + "); rescale X or use a larger alpha")


def xl_fit_ridge_cv_pack(model: PythonObject, fields: PythonObject, call: XlRidgeCv,
                         context: PythonObject) raises -> PythonObject:
    """RidgeCV.fit with cv=None: refuse a system that does not factor, then
    `coef_`, `intercept_`, `alpha_`, `best_score_`, `cv_results_` (with
    store_cv_results) and `n_features_in_`."""
    var nk = x_linear_keys()
    var d = call.request.d
    var A = len(call.request.alphas)
    if xl_ridge_singular(labels_address(call.result), d, A):
        xl_ridge_singular_refuse(model, call.result, d)
    _xl_set_linear(fields, call.result, d, context)
    py_setitem(fields, nk[].k_alpha_, _xl_float(call.result, d + 1))
    py_setitem(fields, nk[].k_best_score_, _xl_float(call.result, d + 2))
    if call.store:
        py_setitem(fields, nk[].k_cv_results_, _xl_slice(call.result, d + 3, A, Python.tuple(A), context))
    py_setitem(fields, nk[].k_n_features_in_, PythonObject(d))
    return model


def xl_fit_ridge_cv_kfold_pack(model: PythonObject, fields: PythonObject, call: XlRidgeCv, best: Int,
                               context: PythonObject) raises -> PythonObject:
    """RidgeCV.fit, `_fit_kfold`: refuse a refit that does not factor, then
    `coef_`, `intercept_`, `alpha_` (the user's own value), `best_score_`
    (that alpha's k-fold score) and `n_features_in_`."""
    var nk = x_linear_keys()
    var d = call.request.d
    if xl_ridge_singular(labels_address(call.result), d, 1):
        xl_ridge_singular_refuse(model, call.result, d)
    _xl_set_linear(fields, call.result, d, context)
    py_setitem(fields, nk[].k_alpha_, PythonObject(call.request.alphas[best]))
    py_setitem(fields, nk[].k_best_score_, _xl_float(call.scores, best))
    py_setitem(fields, nk[].k_n_features_in_, PythonObject(d))
    return model


def xl_fit_ridge_cv_splits_pack(model: PythonObject, fields: PythonObject, call: XlRidgeCv, best: Int,
                                undefined: Bool, context: PythonObject) raises -> PythonObject:
    """RidgeCV.fit, `_fit_splits`: refuse a refit that does not factor,
    then `n_features_in_`, `coef_`, `intercept_`, `alpha_` (the user's own
    value) and `best_score_` (that alpha's mean R^2 over the splits; NaN
    when a split's R^2 is undefined)."""
    var nk = x_linear_keys()
    var d = call.request.d
    if xl_ridge_singular(labels_address(call.result), d, 1):
        xl_ridge_singular_refuse(model, call.result, d)
    var best_score = py_call(_b().float, PythonObject("nan"))
    if not undefined:
        best_score = PythonObject(MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=labels_address(call.means))[best])
    py_setitem(fields, nk[].k_n_features_in_, PythonObject(d))
    _xl_set_linear(fields, call.result, d, context)
    py_setitem(fields, nk[].k_alpha_, PythonObject(call.request.alphas[best]))
    py_setitem(fields, nk[].k_best_score_, best_score)
    return model


# ------------------------------------------------------ LassoCV / ElasticNetCV


def _xl_enetcv_pack(fields: PythonObject, call: XlEnetCvFit, context: PythonObject) raises -> PythonObject:
    """`_enetcv_fit`, PACK: `coef_`, `intercept_`, `alpha_`, `n_iter_`,
    `alphas_`, `mse_path_` and `n_features_in_`; returns the l1_ratio grid
    entry nearest the best one."""
    var nk = x_linear_keys()
    var d = call.fit.request.d
    var L = call.count
    var grid = call.grid
    var folds = call.folds
    _xl_set_linear(fields, call.fit.result, d, context)
    py_setitem(fields, nk[].k_alpha_, _xl_float(call.fit.result, d + 1))
    var l1_best = Float64(_xl_word(call.fit.result, d + 2))
    py_setitem(fields, nk[].k_n_iter_, PythonObject(_xl_count(call.fit.result, d + 3)))
    var off = d + 4
    if L == 1:
        py_setitem(fields, nk[].k_alphas_, _xl_slice(call.fit.result, off, grid, Python.tuple(grid), context))
        py_setitem(fields, nk[].k_mse_path_, _xl_slice(call.fit.result, off + grid, grid * folds, Python.tuple(grid, folds), context))
    else:
        py_setitem(fields, nk[].k_alphas_, _xl_slice(call.fit.result, off, L * grid, Python.tuple(L, grid), context))
        py_setitem(fields, nk[].k_mse_path_, _xl_slice(call.fit.result, off + L * grid, L * grid * folds, Python.tuple(L, grid, folds), context))
    py_setitem(fields, nk[].k_n_features_in_, PythonObject(d))
    # the user's own value (the kernel carried it as float32): the first
    # entry nearest the returned one
    var pick = 0
    var gap = abs(Float64(py=call.ratios[0]) - l1_best)
    for i in range(1, L):  # driver-loop: over the l1_ratio grid
        var g = abs(Float64(py=call.ratios[i]) - l1_best)
        if g < gap:
            gap = g
            pick = i
    return call.ratios[pick]


def xl_fit_lasso_cv_pack(model: PythonObject, fields: PythonObject, call: XlEnetCvFit,
                         context: PythonObject) raises -> PythonObject:
    """LassoCV.fit: the fitted attributes of the path."""
    _ = _xl_enetcv_pack(fields, call, context)
    return model


def xl_fit_elastic_net_cv_pack(model: PythonObject, fields: PythonObject, call: XlEnetCvFit,
                               context: PythonObject) raises -> PythonObject:
    """ElasticNetCV.fit: the fitted attributes of the path, then
    `l1_ratio_` (the grid entry nearest the best one)."""
    var nk = x_linear_keys()
    var chosen = _xl_enetcv_pack(fields, call, context)
    py_setitem(fields, nk[].k_l1_ratio_, chosen)
    return model


# ---------------------------------------------------- LogisticRegressionCV


def xl_fit_logistic_cv_pack(model: PythonObject, fields: PythonObject, call: XlLogCvFit,
                            context: PythonObject) raises -> PythonObject:
    """LogisticRegressionCV.fit: `classes_`, `coef_` (kp, d), `intercept_`
    (kp,), `Cs_`, `C_` (the grid entry nearest the returned best C, per
    fitted problem), `n_iter_`, `scores_` (a (folds, Cs) table per class
    label) and `n_features_in_`."""
    var nk = x_linear_keys()
    var b = _b()
    var d = call.fit.request.d
    var kp = call.kp
    var folds = call.folds
    var nc = len(call.cs)
    py_setitem(fields, nk[].k_classes_, call.classes)
    py_setitem(fields, nk[].k_coef_, _xl_slice(call.fit.result, 0, kp * d, Python.tuple(kp, d), context))
    py_setitem(fields, nk[].k_intercept_, _xl_slice(call.fit.result, kp * d, kp, Python.tuple(kp), context))
    # the grid entry nearest the returned best C (the first on a tie)
    var best_word = Float64(_xl_word(call.fit.result, kp * d + kp))
    var pick = 0
    var gap = abs(call.cs[0] - best_word)
    for i in range(1, nc):  # driver-loop: over the Cs grid
        var g = abs(call.cs[i] - best_word)
        if g < gap:
            gap = g
            pick = i
    py_setitem(fields, nk[].k_Cs_, _xl_reals(call.cs))
    var chosen = Python.list()
    for c in range(kp):  # driver-loop: over the fitted problems (model structure)
        _ = py_call(chosen.append, PythonObject(call.cs[pick]))
    py_setitem(fields, nk[].k_C_, chosen)
    py_setitem(fields, nk[].k_n_iter_, PythonObject(_xl_count(call.fit.result, kp * d + kp + 1)))
    var off = kp * d + kp + 2
    var names = py_call(b.list, call.classes)
    var scores = Python.dict()
    var first = 1 if kp == 1 else 0
    for c in range(first, len(names)):  # driver-loop: over the class labels (one score table each)
        py_setitem(scores, names[c], _xl_slice(call.fit.result, off, folds * nc, Python.tuple(folds, nc), context))
    py_setitem(fields, nk[].k_scores_, scores)
    py_setitem(fields, nk[].k_n_features_in_, PythonObject(d))
    return model
