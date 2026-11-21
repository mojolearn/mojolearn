# SPDX-License-Identifier: Apache-2.0
# PACK of IsotonicRegression's operations: the flat float32 result of
# x_linear/door_run.mojo `xl_isotonic_fit_run` / `xl_fit_run` becomes the
# fitted attributes and the returned predictions. UNPACK is
# bindings/x_linear_isotonic_unpack.mojo.
from std.python import Python, PythonObject
from bindings.py_balanced import py_setitem
from bindings.x_linear_keys_unpack import x_linear_keys
from bindings.x_linear_common_unpack import _xl_word, _xl_float, _xl_count, _xl_slice
from bindings.x_linear_isotonic_unpack import XlIsotonicFit, XlIsotonicTransform


def xl_isotonic_fit_pack(fields: PythonObject, call: XlIsotonicFit, rising: Bool, context: PythonObject) raises:
    """IsotonicRegression.fit: publish `increasing_`, `X_min_`, `X_max_`,
    `X_thresholds_`, `y_thresholds_` (the k pooled knots) and
    `n_features_in_`, the names, types and order the fit always wrote.
    Nothing is published before the fit returned."""
    var nk = x_linear_keys()
    var n = call.fit.request.n
    py_setitem(fields, nk[].k_increasing_, PythonObject(rising))
    var k = _xl_count(call.fit.result, 0)
    py_setitem(fields, nk[].k_X_min_, _xl_float(call.fit.result, 1))
    py_setitem(fields, nk[].k_X_max_, _xl_float(call.fit.result, 2))
    py_setitem(fields, nk[].k_X_thresholds_, _xl_slice(call.fit.result, 3, k, Python.tuple(k), context))
    py_setitem(fields, nk[].k_y_thresholds_, _xl_slice(call.fit.result, 3 + n, k, Python.tuple(k), context))
    py_setitem(fields, nk[].k_n_features_in_, PythonObject(1))


def xl_isotonic_transform_pack(call: XlIsotonicTransform, context: PythonObject) raises -> PythonObject:
    """IsotonicRegression.transform / predict: the (n,) float32
    predictions; with out_of_bounds='raise' the flag word after them
    refuses a query outside the fitted range."""
    var n = call.fit.request.n
    if call.raise_oob:
        if _xl_word(call.fit.result, n) != 0:
            raise Error("A value in x_new is below/above the interpolation range.")
        return _xl_slice(call.fit.result, 0, n, Python.tuple(n), context)
    return call.fit.result
