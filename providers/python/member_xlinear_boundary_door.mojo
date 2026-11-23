# SPDX-License-Identifier: Apache-2.0
"""The member boundary of the XLinear family; RUN is core/xlinear_member.mojo.
Published kinds: 0 PoissonRegressor, 1 GammaRegressor, 2 TweedieRegressor,
3 HuberRegressor, 4 BayesianRidge, 5 ARDRegression, 6 Lars, 7 LassoLars, 8 QuantileRegressor, 9 RidgeClassifier, 10 RidgeCV,
11 LassoCV, 12 ElasticNetCV, 13 LogisticRegressionCV, 14 IsotonicRegression (`_mojolearn_member = (MEMBER_FAMILY_XLINEAR, kind)` on
each class).

  * `xlinear_member_admit` (UNPACK, once per member): the estimator's own
    route (bindings/x_linear_route_unpack.mojo `xl_route`: a hit is one
    record read), the metrics R^2 entry of the process's tier on the route's
    device in the route's fifth slot when that image is loaded, and the
    kind's parameter words (core/xlinear_member.mojo docstring) after the
    estimator's own checks (bindings/x_linear_regressors_unpack.mojo: the
    kind's `xl_fit_<kind>_unpack` refusals).
  * `xlinear_member_pack` (PACK): the fitted attributes the kind's
    `xl_fit_<kind>_pack` writes (bindings/x_linear_regressors_pack.mojo).
  * `xlinear_member_state` (UNPACK): a member state from a fitted estimator.

The family's classes not published yet (each takes its own kind here, the
same pattern over its own fit request and result layout) are listed in the
lane's result file.

SERIAL BY NATURE: one member's admission or publication; a caller runs its
members in parallel.
"""
from std.ffi import c_int, c_long
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.admitted_input_unpack import admit_x
from bindings.classical_arrays_door import classical_array, classical_address
from bindings.member_owner_unpack import MemberOwner, member_labels_take
from bindings.member_class_weight_unpack import member_cw_kind, member_cw_invalid_message
from core.member_class_weight import (
    MEMBER_CW_BALANCED, MEMBER_CW_DICT, MEMBER_CW_INVALID, member_cw_balanced_words, member_cw_pending_words,
    member_cw_invalid_words,
)
from core.member_families import MEMBER_FAMILY_XLINEAR
from bindings.admitted_input_unpack import admitted_array
from bindings.py_lists import py_list_f64
from bindings.py_lists import py_list_int
from bindings.py_balanced import py_call
from bindings.py_balanced import py_get, py_item, py_contains, py_setitem
from bindings.x_linear_common_unpack import _truth, _ge, _float, _int, _name, _class_attr, _attr, _lt, _b, _xl_bayes_refuse, _xl_or
from bindings.x_linear_common_unpack import _eq, _is_int, _isinstance
from bindings.x_linear_keys_unpack import x_linear_keys
from bindings.x_linear_regressors_unpack import _xl_tweedie_link
from bindings.x_linear_cv_unpack import _xl_is_int_cv, _is_int_plain, _xl_cs_grid
from bindings.x_linear_isotonic_unpack import _xl_isotonic_check
from bindings.x_linear_route_unpack import xl_route, xl_ops_require
from core.entry_table import EntryRef, entry_ref_word
from core.hostptr import f32_ptr
from core.member_abi import (
    MemberPlan, MemberCall, MemberMatrix, member_call, member_new, member_free,
    MEMBER_OP_EXPORT, MEMBER_OP_IMPORT, MEMBER_BLOCK_COEF, MEMBER_BLOCK_INTERCEPT, MEMBER_BLOCK_SCALARS, MEMBER_IMPORT_DONE, MEMBER_F64,
    MEMBER_BLOCK_BY_INDEX,
)
from core.xlinear_member import (
    XL_MEMBER_HUBER, XL_MEMBER_BAYES, XL_MEMBER_ARD, XL_MEMBER_LARS, XL_MEMBER_LASSO_LARS, XL_MEMBER_QUANTILE, XL_MEMBER_RIDGE_CLASSIFIER, XL_MEMBER_RIDGE_CV, XL_MEMBER_BLOCK_CV_RESULTS,
    XL_MEMBER_BLOCK_OUTLIERS, XL_MEMBER_BLOCK_LAMBDA,
    XL_MEMBER_BLOCK_ACTIVE, XL_MEMBER_LASSO_CV, XL_MEMBER_ELASTIC_NET_CV, XL_MEMBER_LOGISTIC_CV, XL_MEMBER_ISOTONIC,
    XL_MEMBER_BLOCK_ALPHAS, XL_MEMBER_BLOCK_MSE_PATH, XL_MEMBER_BLOCK_SCORES, XL_MEMBER_BLOCK_THRESHOLDS,
    XL_MEMBER_BLOCK_SIGMA, XL_MEMBER_BLOCK_X_OFFSET, XL_MEMBER_BLOCK_FACTOR,
)
from bindings.classical_labels_door import labels_full_f32
from core.metrics_entry_table import MetricsEntryKey, metrics_r2_entry, MR_OP_COUNT
from providers.python.device_route_boundary_door import device_route_value
from x_linear.door_run import LINK_IDENTITY, LINK_EXP


def _xl_glm_words(model: PythonObject, fields: PythonObject, kind: Int) raises -> List[Float64]:
    """Kinds 0..2: `xl_fit_glm_unpack`'s refusals, then the seven words."""
    var nk = x_linear_keys()
    var name = _name(model)
    if not py_contains(Python.tuple("lbfgs", "newton-cholesky"), py_item(fields, nk[].k_solver)):
        raise Error("mojolearn " + name + ": solver must be 'lbfgs' or 'newton-cholesky'")
    if _truth(py_item(fields, nk[].k_warm_start)):
        raise Error("mojolearn " + name + ": warm_start is not implemented")
    if not _ge(py_item(fields, nk[].k_alpha), PythonObject(0)):
        raise Error("mojolearn " + name + ": alpha must be >= 0")
    var tweedie = kind == 2
    var link = _xl_tweedie_link(fields) if tweedie else 1
    var power = _float(py_item(fields, nk[].k_power)) if tweedie else _float(_class_attr(model, "_power"))
    return [Float64(kind), _intercept_word(fields), Float64(_int(py_item(fields, nk[].k_max_iter))), Float64(link),
        power, _float(py_item(fields, nk[].k_alpha)), _float(py_item(fields, nk[].k_tol))]


def _xl_huber_words(fields: PythonObject) raises -> List[Float64]:
    """Kind 3: `xl_fit_huber_unpack`'s refusals, then the six words."""
    var nk = x_linear_keys()
    var epsilon = py_item(fields, nk[].k_epsilon)
    if not _ge(epsilon, PythonObject(1.0)):
        raise Error("mojolearn HuberRegressor: epsilon must be >= 1.0")
    if _truth(py_item(fields, nk[].k_warm_start)):
        raise Error("mojolearn HuberRegressor: warm_start is not implemented")
    return [Float64(XL_MEMBER_HUBER), _intercept_word(fields), Float64(_int(py_item(fields, nk[].k_max_iter))),
        _float(epsilon), _float(py_item(fields, nk[].k_alpha)), _float(py_item(fields, nk[].k_tol))]


def _xl_bayes_words(model: PythonObject, fields: PythonObject) raises -> List[Float64]:
    """Kind 4: `xl_fit_bayes_unpack`'s refusals, then the ten words."""
    var nk = x_linear_keys()
    _xl_bayes_refuse(model, fields)
    return [Float64(XL_MEMBER_BAYES), _intercept_word(fields), Float64(_int(py_item(fields, nk[].k_max_iter))),
        _float(py_item(fields, nk[].k_tol)), _float(py_item(fields, nk[].k_alpha_1)),
        _float(py_item(fields, nk[].k_alpha_2)), _float(py_item(fields, nk[].k_lambda_1)),
        _float(py_item(fields, nk[].k_lambda_2)), _xl_or(py_item(fields, nk[].k_alpha_init), -1.0),
        _xl_or(py_item(fields, nk[].k_lambda_init), -1.0)]


def _xl_ard_words(model: PythonObject, fields: PythonObject) raises -> List[Float64]:
    """Kind 5: `xl_fit_ard_unpack`'s refusals (the two-sample rule runs at
    the member's fit), then the nine words."""
    var nk = x_linear_keys()
    _xl_bayes_refuse(model, fields)
    return [Float64(XL_MEMBER_ARD), _intercept_word(fields), Float64(_int(py_item(fields, nk[].k_max_iter))),
        _float(py_item(fields, nk[].k_tol)), _float(py_item(fields, nk[].k_alpha_1)),
        _float(py_item(fields, nk[].k_alpha_2)), _float(py_item(fields, nk[].k_lambda_1)),
        _float(py_item(fields, nk[].k_lambda_2)), _float(py_item(fields, nk[].k_threshold_lambda))]


def _xl_lars_words(model: PythonObject, fields: PythonObject, lasso: Bool) raises -> List[Float64]:
    """Kinds 6, 7: `xl_fit_lars_unpack`'s refusals, then the four (Lars)
    or five (LassoLars) words."""
    var nk = x_linear_keys()
    var name = _name(model)
    var limit: PythonObject
    if lasso:
        if not _ge(py_item(fields, nk[].k_alpha), PythonObject(0)):
            raise Error("mojolearn LassoLars: alpha must be >= 0")
        limit = py_item(fields, nk[].k_max_iter)
    else:
        limit = py_item(fields, nk[].k_n_nonzero_coefs)
    if _attr(model, "jitter", PythonObject(None)) is not None:
        raise Error("mojolearn " + name + ": jitter is not implemented")
    var words: List[Float64] = [Float64(XL_MEMBER_LASSO_LARS if lasso else XL_MEMBER_LARS), _intercept_word(fields),
        Float64(_int(limit)), 1.0 if _truth(_attr(model, "positive", PythonObject(False))) else 0.0]
    if lasso:
        words.append(_float(py_item(fields, nk[].k_alpha)))
    return words^


def _xl_quantile_words(fields: PythonObject) raises -> List[Float64]:
    """Kind 8: `xl_fit_quantile_unpack`'s refusals, then the six words."""
    var nk = x_linear_keys()
    var q = py_item(fields, nk[].k_quantile)
    if not (_lt(PythonObject(0), q) and _lt(q, PythonObject(1))):
        raise Error("mojolearn QuantileRegressor: quantile must be strictly between 0 and 1")
    if not _ge(py_item(fields, nk[].k_alpha), PythonObject(0)):
        raise Error("mojolearn QuantileRegressor: alpha must be >= 0")
    return [Float64(XL_MEMBER_QUANTILE), _intercept_word(fields), Float64(_int(py_item(fields, nk[].k_max_iter))),
        _float(q), _float(py_item(fields, nk[].k_alpha)), _float(py_item(fields, nk[].k_tol))]


def _xl_ridge_classifier_words(model: PythonObject, fields: PythonObject) raises -> List[Float64]:
    """Kind 9: `xl_fit_ridge_classifier_unpack`'s refusals, then the three
    words."""
    var nk = x_linear_keys()
    var name = _name(model)
    if _truth(_attr(model, "positive", PythonObject(False))):
        raise Error("mojolearn " + name + ": positive=True is not implemented")
    if not py_contains(Python.tuple("auto", "cholesky"), _attr(model, "solver", PythonObject("auto"))):
        raise Error("mojolearn " + name + ": solver must be 'auto' or 'cholesky'")
    var alpha = py_item(fields, nk[].k_alpha)
    if not _ge(alpha, PythonObject(0)):
        raise Error("mojolearn RidgeClassifier: alpha must be >= 0")
    return [Float64(XL_MEMBER_RIDGE_CLASSIFIER), _intercept_word(fields), _float(alpha)]


def _xl_ridge_cv_words(fields: PythonObject, context: PythonObject, mut admitted: Bool) raises -> List[Float64]:
    """Kind 10: `xl_fit_ridge_cv_unpack`'s refusals, then 5 + A words.
    `admitted` is False for a splitter object as cv (its folds are made in
    Python by the estimator's own split path)."""
    var nk = x_linear_keys()
    var b = _b()
    if py_item(fields, nk[].k_scoring) is not None or _truth(py_item(fields, nk[].k_alpha_per_target)):
        raise Error("mojolearn RidgeCV: only scoring=None, alpha_per_target=False are implemented")
    var given = py_item(fields, nk[].k_alphas)
    var listed = given if Bool(py=py_call(b.hasattr, given, PythonObject("__len__"))) else Python.list(given)
    var floats = py_call(b.list, py_call(b.map, b.float, listed))
    var grid = len(floats)
    var words: List[Float64] = [Float64(XL_MEMBER_RIDGE_CV), _intercept_word(fields),
        1.0 if _truth(py_item(fields, nk[].k_store_cv_results)) else 0.0, 0.0, Float64(grid)]
    var refuse = grid == 0
    for i in range(grid):  # driver-loop: over the alpha grid
        var alpha = Float64(py=floats[i])
        refuse = refuse or not (alpha > 0)
        words.append(alpha)
    if refuse:
        raise Error("mojolearn RidgeCV: alphas must be positive")
    var cv = py_item(fields, nk[].k_cv)
    admitted = True
    if cv is None:
        return words^
    if not _xl_is_int_cv(cv, context):
        admitted = False
        return words^
    if _truth(py_item(fields, nk[].k_store_cv_results)):
        raise Error("cv!=None and store_cv_results=True are incompatible")
    words[3] = Float64(_int(cv))
    return words^


def _xl_enetcv_words(model: PythonObject, fields: PythonObject, context: PythonObject, elastic: Bool,
    mut admitted: Bool) raises -> List[Float64]:
    """Kinds 11, 12: `_xl_enetcv_unpack`'s refusals (selection, the
    l1_ratio range, the cv count, an empty grid), then 10 + L (+ G) words.
    `admitted` is False for a splitter object as cv (its folds are made in
    Python by the estimator's split path)."""
    var nk = x_linear_keys()
    var name = _name(model)
    var b = _b()
    if not _eq(py_item(fields, nk[].k_selection), PythonObject("cyclic")):
        raise Error("mojolearn " + name + ": selection='random' is not implemented")
    var ratios = Python.list(PythonObject(1.0))
    if elastic:
        var given = py_item(fields, nk[].k_l1_ratio)
        var listed = py_call(b.list, given) if Bool(py=py_call(b.hasattr, given, PythonObject("__len__"))) else Python.list(given)
        ratios = py_call(b.list, py_call(b.map, b.float, listed))
    var L = len(ratios)
    for i in range(L):  # driver-loop: over the l1_ratio grid
        var r = Float64(py=ratios[i])
        if not (r > 0 and r <= 1):
            raise Error("mojolearn " + name + ": l1_ratio must be in (0, 1]")
    var cv = py_item(fields, nk[].k_cv)
    admitted = True
    var folds = 5
    if cv is not None:
        if not _xl_is_int_cv(cv, context):
            admitted = False
            return List[Float64]()
        var k = py_call(b.int, cv)
        if not _is_int(k) or _lt(k, PythonObject(2)):
            raise Error("mojolearn: cv must be None or an int in [2, n_samples]")
        folds = _int(k)
    var alphas = py_item(fields, nk[].k_alphas)
    var n_alphas = _attr(model, "n_alphas", PythonObject(None))
    var explicit = False
    var grid = 0
    var values = Python.list()
    if _is_int(alphas):
        grid = _int(alphas)
    elif alphas is None or py_contains(Python.tuple("warn", "deprecated"), alphas):
        grid = _int(n_alphas) if _is_int_plain(n_alphas) else 100
    else:
        explicit = True
        values = py_call(b.sorted, py_call(b.map, b.float, alphas), reverse=PythonObject(True))
        grid = len(values)
    if grid < 1:
        raise Error("mojolearn " + name + ": at least one alpha is required")
    var kind = XL_MEMBER_ELASTIC_NET_CV if elastic else XL_MEMBER_LASSO_CV
    var words: List[Float64] = [Float64(kind), _intercept_word(fields), Float64(_int(py_item(fields, nk[].k_max_iter))),
        _float(py_item(fields, nk[].k_eps)), _float(py_item(fields, nk[].k_tol)),
        1.0 if _truth(py_item(fields, nk[].k_positive)) else 0.0, Float64(folds), Float64(grid),
        1.0 if explicit else 0.0, Float64(L)]
    for i in range(L):  # driver-loop: over the l1_ratio grid
        words.append(Float64(py=ratios[i]))
    if explicit:
        for i in range(grid):  # driver-loop: over the user alpha grid
            words.append(Float64(py=values[i]))
    return words^


def _xl_logistic_cv_words(fields: PythonObject, context: PythonObject, mut admitted: Bool) raises -> List[Float64]:
    """Kind 13: `xl_fit_logistic_cv_unpack`'s refusals (penalty, scoring,
    refit, solver), then 7 + C words (and, for an exact dict class_weight,
    word 5 = 2 and the pending class-weight tail: lane mf-adm3, filled over
    the caller's labels by `member_cw_fill`). `admitted` is False for any
    other non-word class_weight and for a splitter object as cv."""
    var nk = x_linear_keys()
    if not _eq(py_item(fields, nk[].k_penalty), PythonObject("l2")) or _truth(py_item(fields, nk[].k_dual)) or py_item(fields, nk[].k_l1_ratios) is not None:
        raise Error("mojolearn LogisticRegressionCV: only penalty='l2' (primal) is implemented")
    if py_item(fields, nk[].k_scoring) is not None or not _truth(py_item(fields, nk[].k_refit)):
        raise Error("mojolearn LogisticRegressionCV: scoring and refit=False are not implemented")
    if not py_contains(Python.tuple("lbfgs", "newton-cg", "newton-cholesky"), py_item(fields, nk[].k_solver)):
        raise Error("mojolearn LogisticRegressionCV: solver must be lbfgs (newton-* run L-BFGS too)")
    admitted = False
    var class_weight = py_item(fields, nk[].k_class_weight)
    var balanced = False
    var keyed = False
    var invalid = False
    if class_weight is not None:
        if member_cw_kind(class_weight, _b()) == MEMBER_CW_DICT:
            keyed = True
        elif not _eq(class_weight, PythonObject("balanced")):
            # A value the fit door refuses (lane mf-left4): the INVALID tail,
            # whose words the FIT caller's UNPACK raises; a value the door
            # accepts that the member does not read: not admitted.
            if len(member_cw_invalid_message(MEMBER_FAMILY_XLINEAR, PythonObject(None), class_weight, _b())) == 0:
                return List[Float64]()
            invalid = True
        else:
            balanced = True
    var cv = py_item(fields, nk[].k_cv)
    var folds = 5
    if cv is not None:
        if not _xl_is_int_cv(cv, context):
            return List[Float64]()
        folds = _int(cv)
    admitted = True
    var cs = _xl_cs_grid(py_item(fields, nk[].k_Cs), context)
    var words: List[Float64] = [Float64(XL_MEMBER_LOGISTIC_CV), _intercept_word(fields),
        Float64(_int(py_item(fields, nk[].k_max_iter))), _float(py_item(fields, nk[].k_tol)), Float64(folds),
        2.0 if (keyed or invalid) else (1.0 if balanced else 0.0), Float64(len(cs))]
    for i in range(len(cs)):  # driver-loop: over the Cs grid
        words.append(cs[i])
    if keyed:
        member_cw_pending_words(words)
    elif invalid:
        member_cw_invalid_words(words)
    return words^


def _xl_isotonic_words(fields: PythonObject) raises -> List[Float64]:
    """Kind 14: `xl_isotonic_fit_unpack`'s refusal (out_of_bounds), then
    the seven words."""
    var nk = x_linear_keys()
    _xl_isotonic_check(fields)
    var increasing = py_item(fields, nk[].k_increasing)
    var direction = 2.0 if _eq(increasing, PythonObject("auto")) else (1.0 if _truth(increasing) else 0.0)
    var y_min = py_item(fields, nk[].k_y_min)
    var y_max = py_item(fields, nk[].k_y_max)
    var mode = py_item(fields, nk[].k_out_of_bounds)
    var oob = 2.0 if _eq(mode, PythonObject("raise")) else (1.0 if _eq(mode, PythonObject("clip")) else 0.0)
    return [Float64(XL_MEMBER_ISOTONIC), direction, 0.0 if y_min is None else 1.0, _xl_or(y_min, 0.0),
        0.0 if y_max is None else 1.0, _xl_or(y_max, 0.0), oob]


def _intercept_word(fields: PythonObject) raises -> Float64:
    var nk = x_linear_keys()
    return 1.0 if _truth(py_item(fields, nk[].k_fit_intercept)) else 0.0


def xlinear_member_admit(model: PythonObject, fields: PythonObject, kind: Int, context: PythonObject,
    b: PythonObject, mut owner: MemberOwner) raises -> Bool:
    """UNPACK: fill `owner.plan.route`, `owner.plan.params` and `owner.pins`
    from the unfitted estimator."""
    var nk = x_linear_keys()
    var words: List[Float64]
    if kind >= 0 and kind <= 2:
        words = _xl_glm_words(model, fields, kind)
    elif kind == XL_MEMBER_HUBER:
        words = _xl_huber_words(fields)
    elif kind == XL_MEMBER_BAYES:
        words = _xl_bayes_words(model, fields)
    elif kind == XL_MEMBER_ARD:
        words = _xl_ard_words(model, fields)
    elif kind == XL_MEMBER_LARS or kind == XL_MEMBER_LASSO_LARS:
        words = _xl_lars_words(model, fields, kind == XL_MEMBER_LASSO_LARS)
    elif kind == XL_MEMBER_QUANTILE:
        words = _xl_quantile_words(fields)
    elif kind == XL_MEMBER_RIDGE_CLASSIFIER:
        # A class_weight is the tail from word 3 (lane mf-adm3): 'balanced',
        # or a dict's pending table the caller fills over its labels
        # (`member_cw_fill`); any other value keeps the estimator's own doors.
        var weighting = member_cw_kind(py_item(fields, nk[].k_class_weight), b)
        if weighting < 0:
            # A value the fit door refuses (lane mf-left4): the INVALID tail.
            if len(member_cw_invalid_message(MEMBER_FAMILY_XLINEAR, model, py_item(fields, nk[].k_class_weight),
                                             b)) == 0:
                return False
            weighting = MEMBER_CW_INVALID
        words = _xl_ridge_classifier_words(model, fields)
        if weighting == MEMBER_CW_BALANCED:
            member_cw_balanced_words(words)
        elif weighting == MEMBER_CW_DICT:
            member_cw_pending_words(words)
        elif weighting == MEMBER_CW_INVALID:
            member_cw_invalid_words(words)
    elif kind == XL_MEMBER_RIDGE_CV:
        var admitted = True
        words = _xl_ridge_cv_words(fields, context, admitted)
        if not admitted:
            return False
    elif kind == XL_MEMBER_LASSO_CV:
        var admitted = True
        words = _xl_enetcv_words(model, fields, context, False, admitted)
        if not admitted:
            return False
    elif kind == XL_MEMBER_ELASTIC_NET_CV:
        var admitted = True
        words = _xl_enetcv_words(model, fields, context, True, admitted)
        if not admitted:
            return False
    elif kind == XL_MEMBER_LOGISTIC_CV:
        var admitted = True
        words = _xl_logistic_cv_words(fields, context, admitted)
        if not admitted:
            return False
    elif kind == XL_MEMBER_ISOTONIC:
        words = _xl_isotonic_words(fields)
    else:
        raise Error("RuntimeError: mojolearn: unknown XLinear member kind " + String(kind))
    var ops = xl_route(model, fields, context, False)
    xl_ops_require(ops)
    var route = device_route_value(ops.owner)
    if not route.has_fifth:
        # The metrics image's R^2 entry the estimator's own score reaches
        # (its process tier, the route's device), when that image is loaded.
        var metric_mode = 1 if String(py_item(py_item(context, nk[].k_backend_fields), nk[].k__DEFAULT_MODE)) == "identical" else 0
        try:
            var entry = metrics_r2_entry(MetricsEntryKey(metric_mode, route.vendor))
            route.fifth = EntryRef(entry, entry_ref_word(metric_mode, route.vendor, MR_OP_COUNT))
            route.has_fifth = True
        except problem:
            # Not loaded: score is refused in words; the caller runs its own metric.
            _ = String(problem)
    owner.plan.route = route
    owner.plan.params = words^
    owner.pins = Python.tuple(ops.owner, ops.scope)
    return True


def _bool_list(address: Int, count: Int) raises -> PythonObject:
    """One Python list of bools from `count` int32 words (0 / 1), published
    straight into its slots (bindings/x_linear_regressors_pack.mojo
    `_xl_huber_mask`'s form)."""
    ref cpy = Python().cpython()
    var raw = cpy.lib.get_function[PyObjectPtr]("PyList_New")(count)
    if not raw:
        raise cpy.unsafe_get_error()
    var out = PythonObject(from_owned=raw)
    var box = cpy.lib.get_function[PyObjectPtr]("PyBool_FromLong")
    var set_item = cpy.lib.get_function[c_int]("PyList_SetItem")
    var data = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=address)
    for i in range(count):  # driver-loop: publish completed native result words as Python bool slots
        var item = box(c_long(data[i] != 0))
        if not item:
            raise cpy.unsafe_get_error()
        if set_item(out._obj_ptr, i, item) != 0:
            raise cpy.unsafe_get_error()
    return out


def xlinear_member_pack(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan, state: Int) raises:
    """PACK: write the fitted attributes of state `state` into `model`, as
    the estimator's own fit does (bindings/x_linear_regressors_pack.mojo)."""
    var nk = x_linear_keys()
    var kind = Int(plan.params[0])
    var call = MemberCall(plan.route, state)
    if kind == XL_MEMBER_ISOTONIC:
        _xl_isotonic_pack(fields, context, plan, call)
        return
    if kind == XL_MEMBER_RIDGE_CLASSIFIER or kind == XL_MEMBER_LOGISTIC_CV:
        # coef_ (T, d), intercept_ (T,), n_features_in_; classes_ is the caller's PACK
        # (LogisticRegressionCV takes the labels: its scores_ is keyed by them).
        var logcv = kind == XL_MEMBER_LOGISTIC_CV
        var labels = PythonObject(None)
        if logcv:
            labels = member_labels_take(fields, b)
            if labels is not None:
                py_setitem(fields, nk[].k_classes_, labels)
        call.block = MEMBER_BLOCK_COEF
        _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
        var outputs = call.dst.rows
        var d = call.dst.cols
        var weights = classical_array(outputs, d, context)
        unsafe_memcpy(dest=f32_ptr(Int(py=classical_address(weights))), src=f32_ptr(call.dst.data), count=outputs * d)
        call.block = MEMBER_BLOCK_INTERCEPT
        _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
        var offsets = classical_array(outputs, 0, context)
        unsafe_memcpy(dest=f32_ptr(Int(py=classical_address(offsets))), src=f32_ptr(call.dst.data), count=outputs)
        var classes = call.classes
        var blocks = call.out_cols
        py_setitem(fields, nk[].k_coef_, weights)
        py_setitem(fields, nk[].k_intercept_, offsets)
        if logcv:
            _xl_logistic_cv_pack(fields, context, plan, call, labels, outputs, classes, blocks)
        py_setitem(fields, nk[].k_n_features_in_, PythonObject(d))
        return
    call.block = MEMBER_BLOCK_COEF
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var cols = call.dst.cols
    var coef = classical_array(cols, 0, context)
    unsafe_memcpy(dest=f32_ptr(Int(py=classical_address(coef))), src=f32_ptr(call.dst.data), count=cols)
    var blocks = call.out_cols
    call.block = MEMBER_BLOCK_SCALARS
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var scalars = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call.dst.data)
    # float32 words, published as Python floats (the result words' rounding)
    var intercept = Float64(Float32(scalars[0]))
    py_setitem(fields, nk[].k_coef_, coef)
    py_setitem(fields, nk[].k_intercept_, PythonObject(intercept))
    if kind == XL_MEMBER_HUBER:
        py_setitem(fields, nk[].k_scale_, PythonObject(Float64(Float32(scalars[1]))))
        py_setitem(fields, nk[].k_n_iter_, PythonObject(Int(scalars[2])))
        py_setitem(fields, nk[].k_n_features_in_, PythonObject(cols))
        if blocks > 2:
            call.block = XL_MEMBER_BLOCK_OUTLIERS
            _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
            py_setitem(fields, nk[].k_outliers_, _bool_list(call.dst.data, call.dst.cols))
        return
    if kind == XL_MEMBER_BAYES:
        py_setitem(fields, nk[].k_alpha_, PythonObject(Float64(Float32(scalars[1]))))
        py_setitem(fields, nk[].k_lambda_, PythonObject(Float64(Float32(scalars[2]))))
        py_setitem(fields, nk[].k_n_iter_, PythonObject(Int(scalars[3])))
        py_setitem(fields, nk[].k_n_features_in_, PythonObject(cols))
        _xl_posterior_pack(fields, context, plan, call, blocks, cols)
        return
    if kind == XL_MEMBER_ARD:
        py_setitem(fields, nk[].k_alpha_, PythonObject(Float64(Float32(scalars[1]))))
        py_setitem(fields, nk[].k_n_iter_, PythonObject(Int(scalars[2])))
        py_setitem(fields, nk[].k_n_features_in_, PythonObject(cols))
        if blocks > 2:
            call.block = XL_MEMBER_BLOCK_LAMBDA
            _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
            var lam = classical_array(call.dst.cols, 0, context)
            unsafe_memcpy(dest=f32_ptr(Int(py=classical_address(lam))), src=f32_ptr(call.dst.data),
                count=call.dst.cols)
            py_setitem(fields, nk[].k_lambda_, lam)
        _xl_posterior_pack(fields, context, plan, call, blocks, cols)
        return
    if kind == XL_MEMBER_RIDGE_CV:
        py_setitem(fields, nk[].k_alpha_, PythonObject(scalars[1]))
        py_setitem(fields, nk[].k_best_score_, PythonObject(scalars[2]))
        if blocks > 2:
            call.block = XL_MEMBER_BLOCK_CV_RESULTS
            _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
            var table = classical_array(call.dst.cols, 0, context)
            unsafe_memcpy(dest=f32_ptr(Int(py=classical_address(table))), src=f32_ptr(call.dst.data),
                count=call.dst.cols)
            py_setitem(fields, nk[].k_cv_results_, table)
        py_setitem(fields, nk[].k_n_features_in_, PythonObject(cols))
        return
    if kind == XL_MEMBER_LASSO_CV or kind == XL_MEMBER_ELASTIC_NET_CV:
        # `_xl_enetcv_pack`'s order: alpha_, n_iter_, alphas_, mse_path_, n_features_in_ (then l1_ratio_)
        var alpha = Float64(Float32(scalars[1]))
        var l1_ratio = scalars[2]
        var n_iter = Int(scalars[3])
        py_setitem(fields, nk[].k_alpha_, PythonObject(alpha))
        py_setitem(fields, nk[].k_n_iter_, PythonObject(n_iter))
        if blocks > 2:
            var L = Int(plan.params[9])
            var grid = Int(plan.params[7])
            var folds = Int(plan.params[6])
            call.block = XL_MEMBER_BLOCK_ALPHAS
            _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
            var path = admitted_array(Python.tuple(grid) if L == 1 else Python.tuple(L, grid), "<f4", context)
            unsafe_memcpy(dest=f32_ptr(Int(py=classical_address(path))), src=f32_ptr(call.dst.data), count=L * grid)
            py_setitem(fields, nk[].k_alphas_, path)
            call.block = XL_MEMBER_BLOCK_MSE_PATH
            _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
            var mse = admitted_array(Python.tuple(grid, folds) if L == 1 else Python.tuple(L, grid, folds), "<f4",
                context)
            unsafe_memcpy(dest=f32_ptr(Int(py=classical_address(mse))), src=f32_ptr(call.dst.data),
                count=L * grid * folds)
            py_setitem(fields, nk[].k_mse_path_, mse)
        py_setitem(fields, nk[].k_n_features_in_, PythonObject(cols))
        if kind == XL_MEMBER_ELASTIC_NET_CV:
            py_setitem(fields, nk[].k_l1_ratio_, PythonObject(l1_ratio))
        return
    if kind == XL_MEMBER_LARS or kind == XL_MEMBER_LASSO_LARS:
        py_setitem(fields, nk[].k_n_iter_, PythonObject(Int(scalars[1])))
        py_setitem(fields, nk[].k_alpha_, PythonObject(Float64(Float32(scalars[2]))))
        if blocks > 2:
            call.block = XL_MEMBER_BLOCK_ACTIVE
            _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
            var words = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=call.dst.data)
            var active = List[Int](capacity=call.dst.cols)
            for i in range(call.dst.cols):  # driver-loop: over the active features (model structure)
                active.append(Int(words[i]))
            py_setitem(fields, nk[].k_active_, py_list_int(active))
        py_setitem(fields, nk[].k_n_features_in_, PythonObject(cols))
        return
    py_setitem(fields, nk[].k_n_iter_, PythonObject(Int(scalars[1])))
    py_setitem(fields, nk[].k_n_features_in_, PythonObject(cols))
    if kind <= 2:
        # the GLM's prediction link; QuantileRegressor keeps the class's identity
        py_setitem(fields, nk[].k__link, PythonObject(LINK_EXP if plan.params[3] == 1.0 else LINK_IDENTITY))


def _xl_logistic_cv_pack(fields: PythonObject, context: PythonObject, plan: MemberPlan, mut call: MemberCall,
    labels: PythonObject, kp: Int, classes: Int, blocks: Int) raises:
    """Kind 13 after coef_ / intercept_: `Cs_`, `C_` (one entry per fitted
    problem), `n_iter_`, `scores_` (one (folds, Cs) table per class label;
    the first label is skipped for two classes) as
    `xl_fit_logistic_cv_pack` writes them. Without labels the tables are
    keyed by the class codes."""
    var nk = x_linear_keys()
    var nc = Int(plan.params[6])
    var folds = Int(plan.params[4])
    call.block = MEMBER_BLOCK_SCALARS
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var scalars = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call.dst.data)
    var chosen_c = scalars[0]
    var n_iter = Int(scalars[1])
    var cs = List[Float64](capacity=nc)
    for i in range(nc):  # driver-loop: over the Cs grid
        cs.append(plan.params[7 + i])
    py_setitem(fields, nk[].k_Cs_, py_list_f64(cs))
    var chosen = Python.list()
    for c in range(kp):  # driver-loop: over the fitted problems (model structure)
        _ = py_call(chosen.append, PythonObject(chosen_c))
    py_setitem(fields, nk[].k_C_, chosen)
    py_setitem(fields, nk[].k_n_iter_, PythonObject(n_iter))
    var scores = Python.dict()
    if blocks > 3:
        call.block = XL_MEMBER_BLOCK_SCORES
        _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
        var names = py_call(_b().list, labels) if labels is not None else Python.list()
        var first = 1 if kp == 1 else 0
        for c in range(first, classes):  # driver-loop: over the class labels (one score table each)
            var table = admitted_array(Python.tuple(folds, nc), "<f4", context)
            unsafe_memcpy(dest=f32_ptr(Int(py=classical_address(table))), src=f32_ptr(call.dst.data),
                count=folds * nc)
            py_setitem(scores, names[c] if labels is not None else PythonObject(c), table)
    py_setitem(fields, nk[].k_scores_, scores)


def _xl_isotonic_pack(fields: PythonObject, context: PythonObject, plan: MemberPlan, mut call: MemberCall) raises:
    """Kind 14: `increasing_`, `X_min_`, `X_max_`, `X_thresholds_`,
    `y_thresholds_` and `n_features_in_` as `xl_isotonic_fit_pack` writes
    them."""
    var nk = x_linear_keys()
    call.block = MEMBER_BLOCK_SCALARS
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var scalars = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call.dst.data)
    var rising = scalars[0] != 0.0
    var x_min = scalars[1]
    var x_max = scalars[2]
    call.block = XL_MEMBER_BLOCK_THRESHOLDS
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var k = call.dst.cols
    var xs = admitted_array(Python.tuple(k), "<f4", context)
    var ys = admitted_array(Python.tuple(k), "<f4", context)
    unsafe_memcpy(dest=f32_ptr(Int(py=classical_address(xs))), src=f32_ptr(call.dst.data), count=k)
    unsafe_memcpy(dest=f32_ptr(Int(py=classical_address(ys))), src=f32_ptr(call.dst.data + 4 * k), count=k)
    py_setitem(fields, nk[].k_increasing_, PythonObject(rising))
    py_setitem(fields, nk[].k_X_min_, PythonObject(x_min))
    py_setitem(fields, nk[].k_X_max_, PythonObject(x_max))
    py_setitem(fields, nk[].k_X_thresholds_, xs)
    py_setitem(fields, nk[].k_y_thresholds_, ys)
    py_setitem(fields, nk[].k_n_features_in_, PythonObject(1))


def _xl_posterior_pack(fields: PythonObject, context: PythonObject, plan: MemberPlan, mut call: MemberCall,
    blocks: Int, cols: Int) raises:
    """The posterior fields `xl_fit_bayes_pack` / `xl_fit_ard_pack` write,
    from the state's posterior blocks when it holds them (lane mf-iterimp2):
    BayesianRidge its factor (V, ev; sigma_ is formed on its first read, so
    a kept one is dropped), ARDRegression its sigma_; X_offset_ and X_scale_."""
    var nk = x_linear_keys()
    var ard = Int(plan.params[0]) == XL_MEMBER_ARD
    var named = XL_MEMBER_BLOCK_SIGMA if ard else XL_MEMBER_BLOCK_FACTOR
    var held = False
    for i in range(blocks):  # driver-loop: over the state's few fitted blocks (names by index)
        call.block = i
        call.flags = MEMBER_BLOCK_BY_INDEX
        _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
        held = held or call.block == named
    call.flags = 0
    if not held:
        return
    call.block = XL_MEMBER_BLOCK_X_OFFSET
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var offset = classical_array(cols, 0, context)
    if cols > 0:
        unsafe_memcpy(dest=f32_ptr(Int(py=classical_address(offset))), src=f32_ptr(call.dst.data), count=cols)
    call.block = named
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    if ard:
        var dk = call.dst.rows
        # (0, 0) when an ARD fit kept no feature, as its own fit publishes it
        var sigma = admitted_array(Python.tuple(0, 0), "<f4", context)
        if dk > 0:
            sigma = classical_array(dk, dk, context)
            unsafe_memcpy(dest=f32_ptr(Int(py=classical_address(sigma))), src=f32_ptr(call.dst.data),
                count=dk * dk)
        py_setitem(fields, nk[].k_sigma_, sigma)
    else:
        var d = call.dst.cols
        var v = classical_array(d, d, context)
        var ev = classical_array(d, 0, context)
        if d > 0:
            unsafe_memcpy(dest=f32_ptr(Int(py=classical_address(v))), src=f32_ptr(call.dst.data), count=d * d)
            unsafe_memcpy(dest=f32_ptr(Int(py=classical_address(ev))), src=f32_ptr(call.dst.data) + d * d, count=d)
        py_setitem(fields, nk[].k__bayes_v, v)
        py_setitem(fields, nk[].k__bayes_ev, ev)
        _ = py_call(_b().dict.pop, fields, nk[].k__bayes_sigma, PythonObject(None))
    py_setitem(fields, nk[].k_X_offset_, offset)
    py_setitem(fields, nk[].k_X_scale_, labels_full_f32(cols, 1.0, context))


def _xl_posterior_state(fields: PythonObject, context: PythonObject, plan: MemberPlan, mut call: MemberCall,
    kind: Int) raises:
    """IMPORT of the fitted posterior when the estimator holds it, before
    DONE: X_offset_ and BayesianRidge's factor (its private fields, never
    the public sigma_, which is not read here) or ARDRegression's sigma_
    and lambda_."""
    var nk = x_linear_keys()
    var ard = kind == XL_MEMBER_ARD
    var s_source = py_get(fields, nk[].k_sigma_ if ard else nk[].k__bayes_v)
    var offset_source = py_get(fields, nk[].k_X_offset_)
    if s_source is None or offset_source is None:
        return
    var offset = admit_x(offset_source, 1, "X_offset_", context)
    call.flags = 0
    var s_words = PythonObject(None)
    var ev = PythonObject(None)
    if ard:
        # an ARD fit that kept no feature has an empty (0, 0) sigma_: nothing to import
        if Int(py=s_source.size) > 0:
            s_words = admit_x(s_source, 2, "sigma_", context)
            call.block = XL_MEMBER_BLOCK_SIGMA
            call.x = MemberMatrix(Int(py=classical_address(s_words)), Int(py=s_words.shape[0]),
                Int(py=s_words.shape[1]))
            _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    else:
        var ev_source = py_get(fields, nk[].k__bayes_ev)
        if ev_source is None:
            return
        s_words = admit_x(s_source, 2, "posterior factor", context)
        ev = admit_x(ev_source, 1, "posterior factor", context)
        var d = Int(py=ev.size)
        if Int(py=s_words.size) != d * d:
            raise Error("ValueError: the posterior factor does not match n_features_in_")
        # one (d + 1) x d block: V's rows, then ev
        var block = classical_array(d + 1, d, context)
        unsafe_memcpy(dest=f32_ptr(Int(py=classical_address(block))), src=f32_ptr(Int(py=classical_address(s_words))),
            count=d * d)
        unsafe_memcpy(dest=f32_ptr(Int(py=classical_address(block))) + d * d,
            src=f32_ptr(Int(py=classical_address(ev))), count=d)
        call.block = XL_MEMBER_BLOCK_FACTOR
        call.x = MemberMatrix(Int(py=classical_address(block)), d + 1, d)
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        _ = block
    call.block = XL_MEMBER_BLOCK_X_OFFSET
    call.x = MemberMatrix(Int(py=classical_address(offset)), 1, Int(py=offset.size))
    _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    if ard:
        var lam = admit_x(py_item(fields, nk[].k_lambda_), 1, "lambda_", context)
        call.block = XL_MEMBER_BLOCK_LAMBDA
        call.x = MemberMatrix(Int(py=classical_address(lam)), 1, Int(py=lam.size))
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        _ = lam
    _ = s_words
    _ = ev
    _ = offset


def _count_word(fields: PythonObject, key: PythonObject) raises -> Float64:
    var value = py_get(fields, key)
    if value is None:
        return 0.0
    return Float64(Int(py=value))


def xlinear_member_state(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan) raises -> Int:
    """UNPACK: a new member state holding the fitted attributes of `model`
    (coef_ and the kind's scalars; training records are not imported).
    The caller frees it (`member_free`)."""
    var nk = x_linear_keys()
    var kind = Int(plan.params[0])
    if kind == XL_MEMBER_ISOTONIC:
        return _xl_isotonic_state(fields, context, plan)
    var coefficient_source = py_get(fields, nk[].k_coef_)
    if coefficient_source is None:
        raise Error("NotFittedError: this member is not fitted yet")
    if kind == XL_MEMBER_RIDGE_CLASSIFIER or kind == XL_MEMBER_LOGISTIC_CV:
        return _xl_classifier_state(fields, context, plan, coefficient_source)
    var coef = admit_x(coefficient_source, 1, "coef_", context)
    var cols = Int(py=coef.size)
    var intercept = Float64(py=py_item(fields, nk[].k_intercept_))
    var scalars: List[Float64]
    if kind == XL_MEMBER_HUBER:
        scalars = [intercept, Float64(py=py_item(fields, nk[].k_scale_)), _count_word(fields, nk[].k_n_iter_)]
    elif kind == XL_MEMBER_BAYES:
        scalars = [intercept, Float64(py=py_item(fields, nk[].k_alpha_)), Float64(py=py_item(fields, nk[].k_lambda_)),
            _count_word(fields, nk[].k_n_iter_)]
    elif kind == XL_MEMBER_ARD:
        scalars = [intercept, Float64(py=py_item(fields, nk[].k_alpha_)), _count_word(fields, nk[].k_n_iter_)]
    elif kind == XL_MEMBER_RIDGE_CV:
        var best = py_get(fields, nk[].k_best_score_)
        scalars = [intercept, Float64(py=py_item(fields, nk[].k_alpha_)),
            0.0 if best is None else Float64(py=best)]
    elif kind == XL_MEMBER_LARS or kind == XL_MEMBER_LASSO_LARS:
        scalars = [intercept, _count_word(fields, nk[].k_n_iter_), Float64(py=py_item(fields, nk[].k_alpha_))]
    elif kind == XL_MEMBER_LASSO_CV or kind == XL_MEMBER_ELASTIC_NET_CV:
        var l1_ratio = 1.0
        if kind == XL_MEMBER_ELASTIC_NET_CV:
            l1_ratio = Float64(py=py_item(fields, nk[].k_l1_ratio_))
        scalars = [intercept, Float64(py=py_item(fields, nk[].k_alpha_)), l1_ratio, _count_word(fields, nk[].k_n_iter_)]
    else:
        scalars = [intercept, _count_word(fields, nk[].k_n_iter_)]
    var call = member_new(plan)
    try:
        call.block = MEMBER_BLOCK_COEF
        call.x = MemberMatrix(Int(py=classical_address(coef)), 1, cols)
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = MEMBER_BLOCK_SCALARS
        call.x = MemberMatrix(Int(scalars.unsafe_ptr()), 1, len(scalars), MEMBER_F64)
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        if kind == XL_MEMBER_BAYES or kind == XL_MEMBER_ARD:
            _xl_posterior_state(fields, context, plan, call, kind)
        call.block = 0
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    except error:
        member_free(plan, call)
        raise error
    _ = coef
    _ = len(scalars)
    return call.state


def _xl_classifier_state(fields: PythonObject, context: PythonObject, plan: MemberPlan,
    coefficient_source: PythonObject) raises -> Int:
    """Kind 9: coef_ (T, d), intercept_ (T,) and the class count of classes_."""
    var nk = x_linear_keys()
    var classes_source = py_get(fields, nk[].k_classes_)
    if classes_source is None:
        raise Error("NotFittedError: this member is not fitted yet")
    var coef = admit_x(coefficient_source, 2, "coef_", context)
    var outputs = Int(py=coef.shape[0])
    var d = Int(py=coef.shape[1])
    var intercept = admit_x(py_item(fields, nk[].k_intercept_), 1, "intercept_", context)
    var logcv = Int(plan.params[0]) == XL_MEMBER_LOGISTIC_CV
    var scalars = List[Float64]()
    if logcv:
        # [C_ (the one chosen C), n_iter_]
        scalars = [Float64(py=py_item(py_item(fields, nk[].k_C_), PythonObject(0))), _count_word(fields, nk[].k_n_iter_)]
    var call = member_new(plan)
    try:
        call.block = MEMBER_BLOCK_COEF
        call.x = MemberMatrix(Int(py=classical_address(coef)), outputs, d)
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        if logcv:
            call.block = MEMBER_BLOCK_SCALARS
            call.x = MemberMatrix(Int(scalars.unsafe_ptr()), 1, 2, MEMBER_F64)
            _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = MEMBER_BLOCK_INTERCEPT
        call.x = MemberMatrix(Int(py=classical_address(intercept)), 1, Int(py=intercept.size))
        call.classes = len(classes_source)
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    except error:
        member_free(plan, call)
        raise error
    _ = coef
    _ = intercept
    _ = len(scalars)
    return call.state


def _xl_isotonic_state(fields: PythonObject, context: PythonObject, plan: MemberPlan) raises -> Int:
    """Kind 14: [increasing_, X_min_, X_max_] and the joined thresholds."""
    var nk = x_linear_keys()
    var x_source = py_get(fields, nk[].k_X_thresholds_)
    if x_source is None:
        raise Error("NotFittedError: this member is not fitted yet")
    var xs = admit_x(x_source, 1, "X_thresholds_", context)
    var ys = admit_x(py_item(fields, nk[].k_y_thresholds_), 1, "y_thresholds_", context)
    var k = Int(py=xs.size)
    if Int(py=ys.size) != k:
        raise Error("ValueError: X_thresholds_ and y_thresholds_ have different lengths")
    var knots = List[Float32](length=2 * k, fill=0.0)
    unsafe_memcpy(dest=knots.unsafe_ptr(), src=f32_ptr(Int(py=classical_address(xs))), count=k)
    unsafe_memcpy(dest=knots.unsafe_ptr() + k, src=f32_ptr(Int(py=classical_address(ys))), count=k)
    var scalars: List[Float64] = [1.0 if _truth(py_item(fields, nk[].k_increasing_)) else 0.0,
        Float64(py=py_item(fields, nk[].k_X_min_)), Float64(py=py_item(fields, nk[].k_X_max_))]
    var call = member_new(plan)
    try:
        call.block = MEMBER_BLOCK_SCALARS
        call.x = MemberMatrix(Int(scalars.unsafe_ptr()), 1, 3, MEMBER_F64)
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = XL_MEMBER_BLOCK_THRESHOLDS
        call.x = MemberMatrix(Int(knots.unsafe_ptr()), 2, k)
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    except error:
        member_free(plan, call)
        raise error
    _ = xs
    _ = ys
    _ = len(knots)
    _ = len(scalars)
    return call.state
