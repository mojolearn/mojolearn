# SPDX-License-Identifier: Apache-2.0
# PACK of the plain regressors' fits: the flat float32 result of
# x_linear/door_run.mojo `xl_fit_run` / `xl_huber_fit_run` becomes the
# fitted attributes (the names, types, shapes and order the Python wrote),
# one function per public operation; nothing is published before the fit
# returned. UNPACK is bindings/x_linear_regressors_unpack.mojo.
from bindings.x_linear_common_unpack import XlFit
from bindings.x_linear_regressors_unpack import XlHuberFit
from std.ffi import c_int, c_long
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from x_linear.door_run import LINK_IDENTITY, LINK_EXP
from bindings.x_linear_common_unpack import _truth, _str, _class_attr, _xl_word, _xl_float, _xl_count, _xl_slice
from bindings.py_balanced import py_setitem
from bindings.py_lists import py_list_int
from bindings.x_linear_keys_unpack import x_linear_keys
from bindings.classical_labels_door import labels_full_f32
from x_linear.bayes_words import bayes_v_at, bayes_ev_at, bayes_offset_at, ard_dk_at, ard_offset_at, ard_sigma_at
from bindings.x_linear_common_unpack import _b
from bindings.py_balanced import py_call


def _xl_set_linear(fields: PythonObject, out_buf: PythonObject, d: Int, context: PythonObject) raises:
    """coef_ = vals[:d] (float32 Array), intercept_ = float(vals[d])."""
    var nk = x_linear_keys()
    py_setitem(fields, nk[].k_coef_, _xl_slice(out_buf, 0, d, Python.tuple(d), context))
    py_setitem(fields, nk[].k_intercept_, _xl_float(out_buf, d))


def xl_fit_glm_pack(model: PythonObject, fields: PythonObject, call: XlFit, context: PythonObject) raises -> PythonObject:
    """`_GLMBase.fit`: refuse targets outside the loss's range (the word
    after n_iter), then publish `coef_`, `intercept_`, `n_iter_`,
    `n_features_in_` and the prediction's link `_link`."""
    var nk = x_linear_keys()
    var d = call.request.d
    if _xl_word(call.result, d + 2) < 0:
        var loss_name = _class_attr(model, "_loss_name")
        raise Error("Some value(s) of y are out of the valid range of the loss '"
                    + (_str(loss_name) if _truth(loss_name) else String("HalfTweedieLoss")) + "'.")
    _xl_set_linear(fields, call.result, d, context)
    py_setitem(fields, nk[].k_n_iter_, PythonObject(_xl_count(call.result, d + 1)))
    py_setitem(fields, nk[].k_n_features_in_, PythonObject(d))
    py_setitem(fields, nk[].k__link, PythonObject(LINK_EXP if call.request.ip[2] == 1 else LINK_IDENTITY))
    return model


def _xl_huber_mask(flags: PythonObject, address: Int, count: Int) raises -> PythonObject:
    # The private <u1 output already has one completed native flag per row.
    # Keep its owner live while publishing directly into one final bool list.
    ref cpy = Python().cpython()
    var raw = cpy.lib.get_function[PyObjectPtr]("PyList_New")(count)
    if not raw:
        raise cpy.unsafe_get_error()
    # Own the NULL-filled list immediately: partial failures release every
    # initialized slot, while CPython skips the uninitialized NULL slots.
    var out = PythonObject(from_owned=raw)
    var box = cpy.lib.get_function[PyObjectPtr]("PyBool_FromLong")
    var set_item = cpy.lib.get_function[c_int]("PyList_SetItem")
    var data = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=address)
    for i in range(count):  # driver-loop: publish completed native Huber result bytes as Python bool slots
        var item = box(c_long(data[i] != 0))
        if not item:
            raise cpy.unsafe_get_error()
        # SetItem steals the new reference on both success and failure.
        if set_item(out._obj_ptr, i, item) != 0:
            raise cpy.unsafe_get_error()
    _ = flags
    return out


def xl_fit_huber_pack(model: PythonObject, fields: PythonObject, call: XlHuberFit, context: PythonObject) raises -> PythonObject:
    """HuberRegressor.fit: publish `coef_`, `intercept_`, `scale_`,
    `n_iter_`, `n_features_in_` and `outliers_` (one bool per training
    row, from the flag bytes RUN wrote)."""
    var nk = x_linear_keys()
    var d = call.fit.request.d
    _xl_set_linear(fields, call.fit.result, d, context)
    py_setitem(fields, nk[].k_scale_, _xl_float(call.fit.result, d + 1))
    py_setitem(fields, nk[].k_n_iter_, PythonObject(_xl_count(call.fit.result, d + 2)))
    py_setitem(fields, nk[].k_n_features_in_, PythonObject(d))
    py_setitem(fields, nk[].k_outliers_, _xl_huber_mask(call.flags, call.flags_address, call.fit.request.n))
    return model


def xl_fit_bayes_pack(model: PythonObject, fields: PythonObject, call: XlFit, context: PythonObject) raises -> PythonObject:
    """BayesianRidge.fit: `coef_`, `intercept_`, `alpha_`, `lambda_`,
    `n_iter_`, `n_features_in_`."""
    var nk = x_linear_keys()
    var d = call.request.d
    _xl_set_linear(fields, call.result, d, context)
    py_setitem(fields, nk[].k_alpha_, _xl_float(call.result, d + 1))
    py_setitem(fields, nk[].k_lambda_, _xl_float(call.result, d + 2))
    py_setitem(fields, nk[].k_n_iter_, PythonObject(_xl_count(call.result, d + 3)))
    py_setitem(fields, nk[].k_n_features_in_, PythonObject(d))
    # the posterior (x_linear/bayes_words.mojo): the factor the fit holds
    # (V, ev; predict(return_std=True) reads it, the public sigma_ is formed
    # from it on its first read and kept, so a refit drops the kept one), the
    # column means and the unit scales (`_preprocess_data` no longer scales)
    xl_bayes_posterior_pack(fields, call.result, d, context)
    return model


def xl_bayes_posterior_pack(fields: PythonObject, result: PythonObject, d: Int, context: PythonObject) raises:
    """BayesianRidge's posterior fields from a fit's result words."""
    var nk = x_linear_keys()
    py_setitem(fields, nk[].k__bayes_v, _xl_slice(result, bayes_v_at(d), d * d, Python.tuple(d, d), context))
    py_setitem(fields, nk[].k__bayes_ev, _xl_slice(result, bayes_ev_at(d), d, Python.tuple(d), context))
    _ = py_call(_b().dict.pop, fields, nk[].k__bayes_sigma, PythonObject(None))
    py_setitem(fields, nk[].k_X_offset_, _xl_slice(result, bayes_offset_at(d), d, Python.tuple(d), context))
    py_setitem(fields, nk[].k_X_scale_, labels_full_f32(d, 1.0, context))


def xl_fit_ard_pack(model: PythonObject, fields: PythonObject, call: XlFit, context: PythonObject) raises -> PythonObject:
    """ARDRegression.fit: `coef_`, `intercept_`, `alpha_`, `lambda_` (one
    per feature), `n_iter_`, `n_features_in_`."""
    var nk = x_linear_keys()
    var d = call.request.d
    _xl_set_linear(fields, call.result, d, context)
    py_setitem(fields, nk[].k_alpha_, _xl_float(call.result, d + 1))
    py_setitem(fields, nk[].k_lambda_, _xl_slice(call.result, d + 2, d, Python.tuple(d), context))
    py_setitem(fields, nk[].k_n_iter_, PythonObject(_xl_count(call.result, 2 * d + 2)))
    py_setitem(fields, nk[].k_n_features_in_, PythonObject(d))
    # the posterior over the dk kept features (x_linear/bayes_words.mojo)
    var dk = _xl_count(call.result, ard_dk_at(d))
    py_setitem(fields, nk[].k_sigma_, _xl_slice(call.result, ard_sigma_at(d), dk * dk, Python.tuple(dk, dk), context))
    py_setitem(fields, nk[].k_X_offset_, _xl_slice(call.result, ard_offset_at(d), d, Python.tuple(d), context))
    py_setitem(fields, nk[].k_X_scale_, labels_full_f32(d, 1.0, context))
    return model


def xl_fit_lars_pack(model: PythonObject, fields: PythonObject, call: XlFit, context: PythonObject) raises -> PythonObject:
    """Lars.fit / LassoLars.fit: `coef_`, `intercept_`, `n_iter_`,
    `alpha_`, `active_` (the active feature indices, one list) and
    `n_features_in_`."""
    var nk = x_linear_keys()
    var d = call.request.d
    _xl_set_linear(fields, call.result, d, context)
    py_setitem(fields, nk[].k_n_iter_, PythonObject(_xl_count(call.result, d + 1)))
    py_setitem(fields, nk[].k_alpha_, _xl_float(call.result, d + 2))
    var k = _xl_count(call.result, d + 3)
    # the active feature indices read natively, published as one list
    # (py_list_int), not a bound `append` call per feature
    var active_words = List[Int](capacity=max(k, 0))
    for i in range(k):  # driver-loop: over the active features (model structure)
        active_words.append(_xl_count(call.result, d + 4 + i))
    py_setitem(fields, nk[].k_active_, py_list_int(active_words))
    py_setitem(fields, nk[].k_n_features_in_, PythonObject(d))
    return model


def xl_fit_quantile_pack(model: PythonObject, fields: PythonObject, call: XlFit, context: PythonObject) raises -> PythonObject:
    """QuantileRegressor.fit: `coef_`, `intercept_`, `n_iter_`,
    `n_features_in_`."""
    var nk = x_linear_keys()
    var d = call.request.d
    _xl_set_linear(fields, call.result, d, context)
    py_setitem(fields, nk[].k_n_iter_, PythonObject(_xl_count(call.result, d + 1)))
    py_setitem(fields, nk[].k_n_features_in_, PythonObject(d))
    return model
