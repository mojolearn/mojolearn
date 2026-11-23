# SPDX-License-Identifier: Apache-2.0
# UNPACK of the plain regressors' fits (PoissonRegressor, GammaRegressor,
# TweedieRegressor, HuberRegressor, BayesianRidge, ARDRegression, Lars,
# LassoLars, QuantileRegressor; python/mojolearn/_expansion_linear.py): the
# refusals in the order and words the Python made them, X and y through
# the one admission, the checked weights, and the typed fit request
# (x_linear/door_run.mojo `XlFitRequest`: the route, the integer and float
# hyperparameters, the result words), one function per public operation.
# RUN (`xl_fit_run`, `xl_huber_fit_run`) holds no Python object; PACK is
# bindings/x_linear_regressors_pack.mojo.
from bindings.x_linear_common_unpack import XlFit, xl_fit_build, _xl_bayes_refuse, _xl_or
from std.python import Python, PythonObject
from x_linear.door_run import (
    ALGO_GLM, ALGO_HUBER, ALGO_BAYES, ALGO_ARD, ALGO_LARS, ALGO_QUANTILE, LINK_IDENTITY, xl_lbfgs_work,
)
from bindings.x_linear_route_unpack import XLinearOps, xl_ops_require
from bindings.x_linear_common_unpack import (
    _op, _truth, _float, _int, _str, _ge, _lt, _eq, _name, _class_attr, _attr, _xl_matrix, _xl_vector,
    _xl_with_weights,
)
from bindings.py_balanced import py_call, py_contains
from bindings.admitted_input_unpack import admitted_vector
from bindings.x_linear_keys_unpack import x_linear_keys
from bindings.py_balanced import py_item
from bindings.classical_labels_door import labels_address
from x_linear.bayes_words import bayes_out_words, ard_out_words


struct XlHuberFit(Movable):
    """One admitted HuberRegressor.fit: the fit call, the n targets alone
    (`targets`; `y` owns them), the n outlier flag bytes (`flags` at
    `flags_address`), the prediction's link word and epsilon."""
    var fit: XlFit
    var targets: Int
    var y: PythonObject
    var flags: PythonObject
    var flags_address: Int
    var link: Int
    var epsilon: Float64

    def __init__(out self, var fit: XlFit, targets: Int, var y: PythonObject, var flags: PythonObject,
                 flags_address: Int, link: Int, epsilon: Float64):
        self.fit = fit^
        self.targets = targets
        self.y = y^
        self.flags = flags^
        self.flags_address = flags_address
        self.link = link
        self.epsilon = epsilon


def _xl_tweedie_link(fields: PythonObject) raises -> Int:
    """TweedieRegressor._link_code."""
    var nk = x_linear_keys()
    var p = _float(py_item(fields, nk[].k_power))
    if p < 0 or (p > 0 and p < 1):
        raise Error("mojolearn TweedieRegressor: power=" + _str(PythonObject(p)) + " is not implemented (power 0 or >= 1)")
    var link = py_item(fields, nk[].k_link)
    if _eq(link, PythonObject("auto")):
        return 0 if p == 0 else 1
    if _eq(link, PythonObject("identity")):
        if p != 0:
            raise Error("mojolearn TweedieRegressor: the identity link is implemented for power=0 only")
        return 0
    if _eq(link, PythonObject("log")):
        return 1
    raise Error("mojolearn TweedieRegressor: link must be 'auto', 'identity' or 'log'")


def xl_fit_glm_unpack(model: PythonObject, fields: PythonObject, args: PythonObject, entries: XLinearOps,
                      context: PythonObject) raises -> XlFit:
    """`_GLMBase.fit(X, y, sample_weight)`: Poisson, Gamma and Tweedie."""
    var nk = x_linear_keys()
    var name = _name(model)
    if not py_contains(Python.tuple("lbfgs", "newton-cholesky"), py_item(fields, nk[].k_solver)):
        raise Error("mojolearn " + name + ": solver must be 'lbfgs' or 'newton-cholesky'")
    if _truth(py_item(fields, nk[].k_warm_start)):
        raise Error("mojolearn " + name + ": warm_start is not implemented")
    if not _ge(py_item(fields, nk[].k_alpha), PythonObject(0)):
        raise Error("mojolearn " + name + ": alpha must be >= 0")
    xl_ops_require(entries)
    var x = _xl_matrix(args[0], name, context)
    var n = x.rows
    var d = x.cols
    var y = _xl_vector(args[1], n, context)
    # TweedieRegressor reads its own power and link; Poisson and Gamma fix
    # them (`_power`, log link)
    var tweedie = py_contains(fields, nk[].k_power) and py_contains(fields, nk[].k_link)
    var link = _xl_tweedie_link(fields) if tweedie else 1
    var power = _float(py_item(fields, nk[].k_power)) if tweedie else _float(_class_attr(model, "_power"))
    var m = d + 1
    var has_sw = 0
    var yy = _xl_with_weights(y, args[2], n, has_sw, context)
    var ip: List[Int] = [_int(py_item(fields, nk[].k_max_iter)), 1 if _truth(py_item(fields, nk[].k_fit_intercept)) else 0, link, has_sw]
    var fp: List[Float64] = [power, _float(py_item(fields, nk[].k_alpha)), _float(py_item(fields, nk[].k_tol))]
    return xl_fit_build(entries, ALGO_GLM, x.address, n * d, n, d, x.array, yy.address, yy.rows, yy.array, ip^, fp^,
                        d + 3, 3 * n + m * m + 3 * m, 1, context)


def xl_fit_huber_unpack(model: PythonObject, fields: PythonObject, args: PythonObject, entries: XLinearOps,
                        context: PythonObject) raises -> XlHuberFit:
    """HuberRegressor.fit(X, y, sample_weight): the fit, and the outlier
    flags of the training rows from the X the fit already holds."""
    var nk = x_linear_keys()
    var epsilon = py_item(fields, nk[].k_epsilon)
    if not _ge(epsilon, PythonObject(1.0)):
        raise Error("mojolearn HuberRegressor: epsilon must be >= 1.0")
    if _truth(py_item(fields, nk[].k_warm_start)):
        raise Error("mojolearn HuberRegressor: warm_start is not implemented")
    xl_ops_require(entries)
    var x = _xl_matrix(args[0], _name(model), context)
    var n = x.rows
    var d = x.cols
    var y = _xl_vector(args[1], n, context)
    var intercept = _truth(py_item(fields, nk[].k_fit_intercept))
    var p = d + 2 if intercept else d + 1
    var has_sw = 0
    var yy = _xl_with_weights(y, args[2], n, has_sw, context)
    var ip: List[Int] = [_int(py_item(fields, nk[].k_max_iter)), 1 if intercept else 0, has_sw]
    var threshold = _float(epsilon)
    var fp: List[Float64] = [threshold, _float(py_item(fields, nk[].k_alpha)), _float(py_item(fields, nk[].k_tol))]
    var fit = xl_fit_build(entries, ALGO_HUBER, x.address, n * d, n, d, x.array, yy.address, yy.rows, yy.array, ip^,
                           fp^, d + 4 + p, xl_lbfgs_work(p) + n, 1, context)
    # `self.predict(a)`: the fitted model on the float32 X this call holds
    var link = _int(_attr(model, "_link", PythonObject(LINK_IDENTITY)))
    var flags = admitted_vector(n, "<u1", context)
    var flags_address = labels_address(flags)
    return XlHuberFit(fit^, y.address, y.array, flags, flags_address, link, threshold)


def xl_fit_bayes_unpack(model: PythonObject, fields: PythonObject, args: PythonObject, entries: XLinearOps,
                        context: PythonObject) raises -> XlFit:
    """BayesianRidge.fit(X, y, sample_weight)."""
    var nk = x_linear_keys()
    _xl_bayes_refuse(model, fields)
    xl_ops_require(entries)
    var x = _xl_matrix(args[0], _name(model), context)
    var n = x.rows
    var d = x.cols
    var y = _xl_vector(args[1], n, context)
    var has_sw = 0
    var yy = _xl_with_weights(y, args[2], n, has_sw, context)
    var ip: List[Int] = [_int(py_item(fields, nk[].k_max_iter)), 1 if _truth(py_item(fields, nk[].k_fit_intercept)) else 0, has_sw]
    var fp: List[Float64] = [_float(py_item(fields, nk[].k_tol)), _float(py_item(fields, nk[].k_alpha_1)), _float(py_item(fields, nk[].k_alpha_2)),
                             _float(py_item(fields, nk[].k_lambda_1)), _float(py_item(fields, nk[].k_lambda_2)),
                             _xl_or(py_item(fields, nk[].k_alpha_init), -1.0), _xl_or(py_item(fields, nk[].k_lambda_init), -1.0)]
    return xl_fit_build(entries, ALGO_BAYES, x.address, n * d, n, d, x.array, yy.address, yy.rows, yy.array, ip^,
                        fp^, bayes_out_words(d), 3 * d * d + 5 * d + n, 1, context)


def xl_fit_ard_unpack(model: PythonObject, fields: PythonObject, args: PythonObject, entries: XLinearOps,
                      context: PythonObject) raises -> XlFit:
    """ARDRegression.fit(X, y)."""
    var nk = x_linear_keys()
    _xl_bayes_refuse(model, fields)
    xl_ops_require(entries)
    var x = _xl_matrix(args[0], _name(model), context)
    var n = x.rows
    var d = x.cols
    if n < 2:
        raise Error("mojolearn ARDRegression: at least 2 samples are required")
    var y = _xl_vector(args[1], n, context)
    var ip: List[Int] = [_int(py_item(fields, nk[].k_max_iter)), 1 if _truth(py_item(fields, nk[].k_fit_intercept)) else 0]
    var fp: List[Float64] = [_float(py_item(fields, nk[].k_tol)), _float(py_item(fields, nk[].k_alpha_1)), _float(py_item(fields, nk[].k_alpha_2)),
                             _float(py_item(fields, nk[].k_lambda_1)), _float(py_item(fields, nk[].k_lambda_2)), _float(py_item(fields, nk[].k_threshold_lambda))]
    return xl_fit_build(entries, ALGO_ARD, x.address, n * d, n, d, x.array, y.address, y.rows, y.array, ip^, fp^,
                        ard_out_words(d), 3 * d * d + 4 * d + n, 2 * d, context)


def xl_fit_lars_unpack(model: PythonObject, fields: PythonObject, args: PythonObject, lasso: Bool,
                       entries: XLinearOps, context: PythonObject) raises -> XlFit:
    """Lars.fit(X, y) and (`lasso`) LassoLars.fit(X, y): `_lars_fit`."""
    var nk = x_linear_keys()
    var name = _name(model)
    # LassoLars has max_iter and alpha; Lars has n_nonzero_coefs
    var max_iter: PythonObject
    var alpha_min = Float64(0.0)
    if lasso:
        if not _ge(py_item(fields, nk[].k_alpha), PythonObject(0)):
            raise Error("mojolearn LassoLars: alpha must be >= 0")
        max_iter = py_item(fields, nk[].k_max_iter)
        alpha_min = _float(py_item(fields, nk[].k_alpha))
    else:
        max_iter = py_item(fields, nk[].k_n_nonzero_coefs)
    if _attr(model, "jitter", PythonObject(None)) is not None:
        raise Error("mojolearn " + name + ": jitter is not implemented")
    xl_ops_require(entries)
    var x = _xl_matrix(args[0], name, context)
    var n = x.rows
    var d = x.cols
    var y = _xl_vector(args[1], n, context)
    var ip: List[Int] = [_int(max_iter), 1 if _truth(py_item(fields, nk[].k_fit_intercept)) else 0, 1 if lasso else 0,
                         1 if _truth(_attr(model, "positive", PythonObject(False))) else 0]
    var fp: List[Float64] = [alpha_min]
    return xl_fit_build(entries, ALGO_LARS, x.address, n * d, n, d, x.array, y.address, y.rows, y.array, ip^, fp^,
                        2 * d + 4, 2 * d * d + 8 * d, 2 * d, context)


def xl_fit_quantile_unpack(model: PythonObject, fields: PythonObject, args: PythonObject, entries: XLinearOps,
                           context: PythonObject) raises -> XlFit:
    """QuantileRegressor.fit(X, y, sample_weight)."""
    var nk = x_linear_keys()
    var q = py_item(fields, nk[].k_quantile)
    if not (_lt(PythonObject(0), q) and _lt(q, PythonObject(1))):
        raise Error("mojolearn QuantileRegressor: quantile must be strictly between 0 and 1")
    if not _ge(py_item(fields, nk[].k_alpha), PythonObject(0)):
        raise Error("mojolearn QuantileRegressor: alpha must be >= 0")
    xl_ops_require(entries)
    var x = _xl_matrix(args[0], _name(model), context)
    var n = x.rows
    var d = x.cols
    var y = _xl_vector(args[1], n, context)
    var m = d + 1
    var has_sw = 0
    var yy = _xl_with_weights(y, args[2], n, has_sw, context)
    var tol = py_item(fields, nk[].k_tol)
    var ip: List[Int] = [_int(py_item(fields, nk[].k_max_iter)), 1 if _truth(py_item(fields, nk[].k_fit_intercept)) else 0, has_sw]
    var fp: List[Float64] = [_float(q), _float(py_item(fields, nk[].k_alpha)), _float(py_call(_op().truediv, tol, PythonObject(100.0))),
                             _float(tol), 1.0]  # fp[4]: unread since MOJOLEARN_XQ_ABS_BALANCE went (TOMBSTONE in x_linear/quantile.mojo)
    return xl_fit_build(entries, ALGO_QUANTILE, x.address, n * d, n, d, x.array, yy.address, yy.rows, yy.array, ip^,
                        fp^, d + 3, m * m + 3 * m + 4 * n + 3 * d, 1, context)
