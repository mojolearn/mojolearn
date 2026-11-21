# SPDX-License-Identifier: Apache-2.0
# PACK of the coordinate-descent pair's operations (Lasso, ElasticNet):
# the typed result of x_linear/door_run.mojo `xl_cd_fit_run` /
# `xl_cd_predict_run` becomes the fitted attributes and the returned
# objects, one function per public operation. UNPACK is
# bindings/x_linear_cd_unpack.mojo.
from std.python import PythonObject
from bindings.py_balanced import py_setitem
from bindings.x_linear_keys_unpack import x_linear_keys
from bindings.x_linear_common_unpack import _xl_float
from bindings.x_linear_cd_unpack import XlCdFit, XlCdPredict


def xl_cd_fit_pack(model: PythonObject, fields: PythonObject, call: XlCdFit, n_iter: Int) raises -> PythonObject:
    """ElasticNet.fit / Lasso.fit: publish `input_copied_`,
    `fortran_copied_`, `coef_`, `intercept_`, `n_iter_` and
    `n_features_in_` (the names, types and order the fit always wrote) and
    return the model. Nothing is published before the solver returned: a
    fit that refuses or fails leaves the estimator's fields as they were."""
    var nk = x_linear_keys()
    py_setitem(fields, nk[].k_input_copied_, PythonObject(call.copied))
    py_setitem(fields, nk[].k_fortran_copied_, PythonObject(call.copied))
    py_setitem(fields, nk[].k_coef_, call.coef)
    py_setitem(fields, nk[].k_intercept_, _xl_float(call.info, 0))
    py_setitem(fields, nk[].k_n_iter_, PythonObject(n_iter))
    py_setitem(fields, nk[].k_n_features_in_, PythonObject(call.request.n_cols))
    return model


def xl_cd_predict_pack(call: XlCdPredict) raises -> PythonObject:
    """ElasticNet.predict / Lasso.predict: the (n,) float32 prediction."""
    return call.prediction
