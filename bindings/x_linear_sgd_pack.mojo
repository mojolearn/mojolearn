# SPDX-License-Identifier: Apache-2.0
# PACK of the online fits and of RidgeClassifier.fit: the flat float32
# result of x_linear/door_run.mojo `xl_fit_run` becomes the fitted
# attributes (the names, types, shapes and order the Python wrote), one
# function per public operation; nothing is published before the fit
# returned. UNPACK is bindings/x_linear_sgd_unpack.mojo.
from bindings.x_linear_common_unpack import XlFit
from bindings.x_linear_sgd_unpack import XlClassFit
from std.python import Python, PythonObject
from bindings.x_linear_common_unpack import _b, _name, _xl_word, _xl_float, _xl_count, _xl_slice
from bindings.py_balanced import py_call, py_setitem
from bindings.x_linear_keys_unpack import x_linear_keys
from bindings.classical_labels_door import labels_reshape


def _xl_sgd_pack(fields: PythonObject, call: XlFit, context: PythonObject) raises -> Tuple[PythonObject, PythonObject]:
    """`_sgd_fit`, PACK: refuse an overflowed fit (the last result word),
    publish `n_iter_`, `t_` and `n_features_in_`, and return (coef
    (problems, d), intercept (problems,))."""
    var nk = x_linear_keys()
    var n = call.request.n
    var d = call.request.d
    var n_classes = call.request.ip[0]
    var problems = n_classes if n_classes > 2 else 1
    var n_out = problems * d + problems + 2
    if _xl_word(call.result, n_out - 1) != 0:
        raise Error("Floating-point under-/overflow occurred. Scaling input data with "
                    + "StandardScaler or MinMaxScaler might help.")
    var coef = _xl_slice(call.result, 0, problems * d, Python.tuple(problems, d), context)
    var intercept = _xl_slice(call.result, problems * d, problems, Python.tuple(problems), context)
    var n_iter = _xl_count(call.result, n_out - 2)
    py_setitem(fields, nk[].k_n_iter_, PythonObject(n_iter))
    py_setitem(fields, nk[].k_t_, PythonObject(Float64(n_iter * n + 1)))
    py_setitem(fields, nk[].k_n_features_in_, PythonObject(d))
    return (coef, intercept)


def xl_fit_sgd_regressor_pack(model: PythonObject, fields: PythonObject, call: XlFit, context: PythonObject) raises -> PythonObject:
    """SGDRegressor.fit / PassiveAggressiveRegressor.fit: `n_iter_`, `t_`,
    `n_features_in_`, `coef_` (d,) and `intercept_` (1,)."""
    var nk = x_linear_keys()
    var fit = _xl_sgd_pack(fields, call, context)
    py_setitem(fields, nk[].k_coef_, labels_reshape(fit[0], Python.tuple(call.request.d), context))
    py_setitem(fields, nk[].k_intercept_, fit[1])
    return model


def xl_fit_one_class_pack(model: PythonObject, fields: PythonObject, call: XlFit, context: PythonObject) raises -> PythonObject:
    """SGDOneClassSVM.fit: `n_iter_`, `t_`, `n_features_in_`, `coef_` (d,)
    and `offset_` (the binding returns it in the intercept word)."""
    var nk = x_linear_keys()
    var fit = _xl_sgd_pack(fields, call, context)
    py_setitem(fields, nk[].k_coef_, labels_reshape(fit[0], Python.tuple(call.request.d), context))
    # the binding returns offset_ itself for one class
    py_setitem(fields, nk[].k_offset_, _xl_slice(fit[1], 0, 1, Python.tuple(1), context))
    return model


def xl_fit_sgd_classifier_pack(model: PythonObject, fields: PythonObject, call: XlClassFit, context: PythonObject) raises -> PythonObject:
    """SGDClassifier.fit / Perceptron.fit / PassiveAggressiveClassifier.fit:
    `classes_`, `n_iter_`, `t_`, `n_features_in_`, `coef_` (problems, d)
    and `intercept_` (problems,), the order the fit always wrote."""
    var nk = x_linear_keys()
    py_setitem(fields, nk[].k_classes_, call.classes)
    var fit = _xl_sgd_pack(fields, call.fit, context)
    py_setitem(fields, nk[].k_coef_, fit[0])
    py_setitem(fields, nk[].k_intercept_, fit[1])
    return model


def xl_fit_ridge_classifier_pack(model: PythonObject, fields: PythonObject, call: XlClassFit, context: PythonObject) raises -> PythonObject:
    """RidgeClassifier.fit: refuse a system that does not factor (status
    word 2), then `classes_`, `coef_` (T, d), `intercept_` (T,) and
    `n_features_in_`."""
    var nk = x_linear_keys()
    var d = call.fit.request.d
    var T = call.fit.request.ip[0]
    # status 2: X'X + alpha I does not factor even in float-float
    if _xl_word(call.fit.result, T * d + T + 2 + 1) == 2.0:
        raise Error("mojolearn " + _name(model) + ": X'X + alpha I is singular even in float-float (alpha="
                    + String(py=py_call(_b().format, _xl_float(call.fit.result, T * d + T), PythonObject("g")))
                    + "); rescale X or use a larger alpha")
    py_setitem(fields, nk[].k_classes_, call.classes)
    py_setitem(fields, nk[].k_coef_, _xl_slice(call.fit.result, 0, T * d, Python.tuple(T, d), context))
    py_setitem(fields, nk[].k_intercept_, _xl_slice(call.fit.result, T * d, T, Python.tuple(T), context))
    py_setitem(fields, nk[].k_n_features_in_, PythonObject(d))
    return model
