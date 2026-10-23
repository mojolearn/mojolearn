# SPDX-License-Identifier: Apache-2.0
# `GradientBoosting.fit` in one native call (python/mojolearn/ensemble.py):
# the input layout choice (row-major for a FAST Apple binding with the device
# quantizer, else column-major), the target and class checks with their
# wording, the feature flags, the sample weights, the eval set, the
# 35-slot parameter list (the resolved learning rate, leaf iterations and
# random strength, the class weights and the optional tail), the multi-target
# and query/pair tails, the option strings (loss, bootstrap, detector, NaN
# mode, border type, the Ordered fold settings), the `gbdt_fit` /
# `gbdt_fit_rowmajor` launch, and the fitted state read from its outputs.
# The entry, its arguments and their order are the Python path's, so the
# fitted model text keeps its bytes.
#
# Round trips removed against the Python path: X, y, the weights and the eval
# set are staged and handed to the fit entry in the same call (Python
# returned between each conversion, scan and the launch); the output
# dimension is read from the fitted text in the same call.
from std.math import isfinite, floor, copysign
from std.memory import bitcast
from std.os import getenv
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem, py_fields
from bindings.py_attr import py_attr
from bindings.py_typed_call import py_call_typed
from bindings.host_archive_native import host_archive_gbdt_state
from bindings.classical_arrays import classical_wrap_array
from bindings.native_buffer_owner import native_buffer_view
from bindings.trees_public_inputs import (
    TreesMatrix, TreesLabels, trees_builtins, trees_entry, trees_need, trees_helper, trees_vector, trees_address,
    trees_f32, trees_f32_view, trees_view, trees_check_rank, trees_forest_layout, trees_all_finite,
    trees_check_sample_weight, trees_call, trees_encode_labels,
)
from bindings.trees_public_gbdt import gbdt_table, gbdt_vendor, gbdt_has_ctr, _field, _int, _repr
from checks.numerics import portable_exp64, portable_log64, pinned_mul_f64

comptime _UNSET = -1.0
comptime _F32_MAX = 3.4028234663852886e+38


def _b() raises -> PythonObject:
    return trees_builtins()


def _str(value: PythonObject) raises -> String:
    return String(py_call(_b().str, value))


def _in(value: PythonObject, table: PythonObject) raises -> Bool:
    return py_contains(table, value)


def _eq(value: PythonObject, text: String) raises -> Bool:
    return Bool(py=value == PythonObject(text))


def _truthy(value: PythonObject) raises -> Bool:
    return Bool(py=py_call(_b().bool, value))


def _is_bool(value: PythonObject) raises -> Bool:
    var t = py_type(value)
    return t is _b().bool or String(t.__name__) == "bool_"


def _float(value: PythonObject) raises -> Float64:
    return Float64(py=py_call_typed(_b().float, value))


def _tables(context: PythonObject) raises -> PythonObject:
    return context["trees_gbdt_tables"]


# ------------------------------------------------------------ resolution


def _c_round(number: Float64, precision: Int) -> Float64:
    """`ensemble._c_round`: C `round` (halves away from zero) at
    `precision` decimals."""
    var multiplier = Float64(1.0)
    for _ in range(precision):  # small-loop(precision): six decimal places
        multiplier = multiplier * 10.0
    var scaled = pinned_mul_f64(number, multiplier)
    var rounded = floor(abs(scaled) + 0.5)
    return copysign(rounded, scaled) / multiplier


def gbdt_auto_learning_rate(loss: PythonObject, n_rows: Float64, iterations: Float64, use_best: Bool, bfa: Bool,
                            table: PythonObject, context: PythonObject) raises -> PythonObject:
    """`ensemble.catboost_auto_learning_rate`: CatBoost's GPU automatic
    rate, or None where their table has no row. `exp` and `log` are the
    package's portable binary64 routines (the ones `_portable_math`
    loads)."""
    var target = String("")
    if _eq(loss, "Logloss") or _eq(loss, "MultiClass") or _eq(loss, "RMSE"):
        target = String(loss)
    if target == "":
        return PythonObject(None)
    var source = table if table is not None else _tables(context)["gpu_auto_learning_rate"]
    var coeffs = py_get(source, Python.tuple(PythonObject(target), PythonObject(use_best), PythonObject(bfa)))
    if coeffs is None:
        return PythonObject(None)
    var a = _float(coeffs[0])
    var bb = _float(coeffs[1])
    var c = _float(coeffs[2])
    var d = _float(coeffs[3])
    if iterations <= 0.0 or n_rows <= 0.0:
        raise Error("math domain error")
    var custom = portable_exp64(pinned_mul_f64(c, portable_log64(iterations)) + d)
    var default_iter = portable_exp64(pinned_mul_f64(c, portable_log64(1000.0)) + d)
    var default_rate = portable_exp64(pinned_mul_f64(a, portable_log64(n_rows)) + bb)
    if not isfinite(custom) or not isfinite(default_iter) or not isfinite(default_rate):
        raise Error("OverflowError: math range error")
    var rate = pinned_mul_f64(default_rate, custom) / default_iter
    return PythonObject(_c_round(min(rate, 0.5), 6))


def catboost_auto_learning_rate_op(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """The public `ensemble.catboost_auto_learning_rate(loss, n_rows,
    iterations, use_best_model, boost_from_average, table=None)`."""
    return gbdt_auto_learning_rate(args[0], _float(args[1]), _float(args[2]), _truthy(args[3]),
                                   _truthy(args[4]), args[5] if _truthy(args[5]) else PythonObject(None), context)


def _sabotaged() -> Bool:
    return getenv("MOJOLEARN_CATBOOST_DEFAULTS_SABOTAGE") == "1" and getenv("MOJOLEARN_HOST_ALLOW_SABOTAGE") == "1"


def _resolved_learning_rate(fields: PythonObject, model: PythonObject, n_rows: Int, n_eval_rows: Int,
                            eval_constant: Bool, context: PythonObject) raises -> Float64:
    """`GradientBoosting._resolved_learning_rate`."""
    var t = _tables(context)
    var lr = _field(fields, model, "learning_rate")
    if lr is not None:
        return _float(lr)
    if not _eq(_field(fields, model, "grow_policy"), "SymmetricTree"):
        return _float(t["legacy_learning_rate"])
    if (_field(fields, model, "l2_leaf_reg") is not None or _field(fields, model, "leaf_estimation_method") is not None
            or _field(fields, model, "leaf_estimation_iterations") is not None):
        return _float(t["catboost_learning_rate"])
    var use_best_obj = _field(fields, model, "use_best_model")
    var use_best = (n_eval_rows > 0 and not eval_constant) if use_best_obj is None else _truthy(use_best_obj)
    var bfa_obj = _field(fields, model, "boost_from_average")
    var bfa = _in(_field(fields, model, "loss"), t["boost_from_average_losses"]) if bfa_obj is None else _truthy(bfa_obj)
    var table = t["cpu_auto_learning_rate"] if _sabotaged() else PythonObject(None)
    var rate = gbdt_auto_learning_rate(_field(fields, model, "loss"), Float64(n_rows),
                                       Float64(Int(py=py_call_typed(_b().int, _field(fields, model, "n_estimators")))),
                                       use_best, bfa, table, context)
    return _float(t["catboost_learning_rate"]) if rate is None else _float(rate)


def _resolved_boosting_type(fields: PythonObject, model: PythonObject, n_rows: Int, context: PythonObject) raises -> String:
    """`GradientBoosting._resolved_boosting_type`."""
    var t = _tables(context)
    var explicit = _field(fields, model, "boosting_type")
    if explicit is not None:
        return _str(explicit)
    var loss = _field(fields, model, "loss")
    var fraction = py_get(fields, "feature_fraction", PythonObject(1.0))
    var score = _field(fields, model, "score_function")
    if (not _eq(_field(fields, model, "grow_policy"), "SymmetricTree") or _in(loss, t["multi_output_losses"])
            or _in(loss, t["multi_regression_losses"])
            or not (_eq(score, "Cosine") or _eq(score, "NewtonCosine"))
            or _truthy(_field(fields, model, "use_pointwise_searcher")) or Bool(py=fraction < PythonObject(1.0))):
        return "Plain"
    if n_rows >= Int(py=t["ordered_max_rows"]) or Int(py=py_call_typed(_b().int, _field(fields, model, "n_estimators"))) < Int(py=t["ordered_min_iterations"]):
        return "Plain"
    return "Ordered"


def _check_ordered_options(boosting: String, fields: PythonObject, model: PythonObject, context: PythonObject) raises:
    """`ensemble._check_ordered_options`."""
    if boosting != "Ordered":
        return
    var t = _tables(context)
    var loss = _field(fields, model, "loss")
    if not _eq(_field(fields, model, "grow_policy"), "SymmetricTree"):
        raise Error("mojolearn: Ordered boosting is not supported for nonsymmetric "
                    + "trees. (catboost_options.cpp:757-759)")
    if _in(loss, t["multi_output_losses"]):
        raise Error("mojolearn: On GPU loss " + _str(loss) + " can't be used with ordered "
                    + "boosting (catboost_options.cpp:949-967)")
    if _in(loss, t["multi_regression_losses"]):
        raise Error("NotImplementedError: mojolearn: boosting_type='Ordered' with loss=" + _repr(loss) + " is not "
                    + "carried: the Ordered arm is one-dimensional; use "
                    + "boosting_type='Plain'")
    var score = _field(fields, model, "score_function")
    if not (_eq(score, "Cosine") or _eq(score, "NewtonCosine")):
        raise Error("mojolearn: Score function " + _str(score) + " can't be used with "
                    + "ordered boosting (catboost_options.cpp:972-978)")
    if _eq(_field(fields, model, "leaf_estimation_method"), "Exact"):
        raise Error("mojolearn: Exact leaf estimation method don't work with ordered "
                    + "boosting on GPU (catboost_options.cpp:346-350)")
    if _in(loss, t["querywise_losses"]):
        raise Error("NotImplementedError: mojolearn: boosting_type='Ordered' with loss=" + _repr(loss) + " is not "
                    + "implemented: their folds follow the query grouping "
                    + "(dynamic_boosting.h:189-223); use boosting_type='Plain'")
    if _truthy(_field(fields, model, "use_pointwise_searcher")):
        raise Error("mojolearn: use_pointwise_searcher selects the doc-parallel Plain "
                    + "searcher; an Ordered fit runs the feature-parallel fold searcher")
    if Bool(py=py_get(fields, "feature_fraction", PythonObject(1.0)) < PythonObject(1.0)):
        raise Error("NotImplementedError: mojolearn: boosting_type='Ordered' with feature_fraction < 1 is "
                    + "not implemented")


def _validate_search_options(fields: PythonObject, model: PythonObject, context: PythonObject) raises -> Float64:
    """`ensemble._validate_search_options`: the strength the fit uses."""
    var t = _tables(context)
    var grow = _field(fields, model, "grow_policy")
    var score = _field(fields, model, "score_function")
    if _truthy(_field(fields, model, "use_pointwise_searcher")) and not _eq(grow, "SymmetricTree"):
        raise Error("mojolearn: use_pointwise_searcher=True is an OBLIVIOUS searcher; "
                    + "grow_policy=" + _repr(grow) + " requires the greedy subsets searcher")
    var rs = _field(fields, model, "random_strength")
    var l2_score = _eq(score, "L2") or _eq(score, "NewtonL2")
    if rs is None:
        if _eq(grow, "SymmetricTree") and not l2_score:
            return _float(t["catboost_random_strength"])
        return _float(t["legacy_random_strength"])
    var strength = Float64(0)
    var valid = not _is_bool(rs) and Bool(py=py_call(_b().isinstance, rs, context["trees_real_type"]))
    var nan = bitcast[DType.float64](UInt64(0x7FF8000000000000))
    strength = nan
    if valid:
        try:
            strength = _float(rs)
        except:
            strength = nan
    if not isfinite(strength) or not (strength >= 0 and strength <= _F32_MAX):
        raise Error("mojolearn: random_strength must be finite, nonnegative and "
                    + "<= Float32.MAX_FINITE")
    if strength != 0.0 and l2_score:
        raise Error("mojolearn: random_strength=" + _str(PythonObject(strength)) + " does nothing under "
                    + "score_function=" + _repr(score) + "; use Cosine or NewtonCosine, "
                    + "or random_strength=0.0")
    return strength


def _f(value: PythonObject) raises -> PythonObject:
    """`_params`' `f`: the unset sentinel, else `float(value)`."""
    if value is None:
        return PythonObject(_UNSET)
    return py_call_typed(_b().float, value)


def _tri(value: PythonObject) raises -> PythonObject:
    """`ensemble._tri`."""
    if value is None:
        return PythonObject(-1)
    return PythonObject(1 if _truthy(value) else 0)


def _int_or(value: PythonObject, unset: Int) raises -> PythonObject:
    if value is None:
        return PythonObject(unset)
    return py_call_typed(_b().int, value)


def _params(fields: PythonObject, model: PythonObject, n_rows: Int, n_features: Int, n_flags: Int, n_weights: Int,
            n_eval_rows: Int, eval_constant: Bool, context: PythonObject) raises -> PythonObject:
    """`GradientBoosting._params`: the 35-slot list in the binding's order,
    the class weights, then the optional min_split_gain /
    min_child_hessian / feature_fraction tail."""
    var b = _b()
    var t = _tables(context)
    var strength = _validate_search_options(fields, model, context)
    var cw = _field(fields, model, "class_weights")
    var weights = Python.list() if cw is None or not _truthy(cw) else py_call(b.list, cw)
    var method = _field(fields, model, "leaf_estimation_method")
    var method_code = PythonObject(-1) if method is None else t["leaf_estimation_names"][method]
    var iters = _field(fields, model, "leaf_estimation_iterations")
    if (iters is None and _eq(_field(fields, model, "grow_policy"), "SymmetricTree")
            and Int(py=py_call_typed(b.int, _field(fields, model, "n_estimators"))) < 200 and n_features < 20):
        iters = PythonObject(1)
    var rate = _resolved_learning_rate(fields, model, n_rows, n_eval_rows, eval_constant, context)
    var tail = Python.list()
    var fraction = py_get(fields, "feature_fraction", PythonObject(1.0))
    var gain = _field(fields, model, "min_split_gain")
    var hessian = _field(fields, model, "min_child_hessian")
    if not Bool(py=fraction == PythonObject(1.0)):
        _ = py_call(tail.append, PythonObject(-1.0) if gain is None else py_call_typed(b.float, gain))
        _ = py_call(tail.append, PythonObject(-1.0) if hessian is None else py_call_typed(b.float, hessian))
        _ = py_call(tail.append, py_call_typed(b.float, fraction))
    elif hessian is not None:
        _ = py_call(tail.append, PythonObject(-1.0) if gain is None else py_call_typed(b.float, gain))
        _ = py_call(tail.append, py_call_typed(b.float, hessian))
    elif gain is not None:
        _ = py_call(tail.append, py_call_typed(b.float, gain))
    var l2 = _field(fields, model, "l2_leaf_reg")
    var l2_value = PythonObject(0.0 if _eq(_field(fields, model, "loss"), "YetiRank") else 3.0) if l2 is None else l2
    var params = Python.list(
        PythonObject(n_rows), PythonObject(n_features), PythonObject(n_weights), PythonObject(n_flags),
        py_call_typed(b.int, _field(fields, model, "border_count")),
        py_call_typed(b.int, _field(fields, model, "n_estimators")),
        py_call_typed(b.int, _field(fields, model, "max_depth")),
        PythonObject(rate),
        py_call_typed(b.float, l2_value),
        py_call_typed(b.int, _field(fields, model, "random_state")),
        t["score_function_names"][_field(fields, model, "score_function")],
        _f(_field(fields, model, "loss_alpha")), _f(_field(fields, model, "loss_q")),
        _f(_field(fields, model, "loss_delta")), _f(_field(fields, model, "loss_variance_power")),
        _f(_field(fields, model, "loss_border")),
        PythonObject(-1) if iters is None else py_call_typed(b.int, iters),
        method_code,
        py_call_typed(b.float, _field(fields, model, "bagging_temperature")),
        _f(_field(fields, model, "subsample")),
        PythonObject(n_eval_rows),
        _f(_field(fields, model, "od_pvalue")),
        _int_or(_field(fields, model, "od_wait"), -1),
        _tri(_field(fields, model, "use_best_model")),
        py_call_typed(b.int, _field(fields, model, "best_model_min_trees")),
        PythonObject(strength),
        PythonObject(1 if _truthy(_field(fields, model, "use_pointwise_searcher")) else 0),
        py_call_typed(b.int, _field(fields, model, "border_build_max_samples")),
        _int_or(_field(fields, model, "permutation_count"), -1),
        _int_or(_field(fields, model, "ctr_estimation_permutation_id"), -1),
        _tri(_field(fields, model, "boost_from_average")),
        t["grow_policy_codes"][_field(fields, model, "grow_policy")],
        _int_or(_field(fields, model, "max_leaves"), -1),
        py_call_typed(b.int, _field(fields, model, "min_data_in_leaf")),
        PythonObject(len(weights)),
    )
    _ = py_call(params.extend, weights)
    _ = py_call(params.extend, tail)
    return params


def _padded_tail(params: PythonObject) raises -> PythonObject:
    """The slots past the class weights padded to their three defaults
    (`[-1.0, -1.0, 1.0][len(tail):]`), so a later slot lands in place."""
    var b = _b()
    var fixed = 35 + Int(py=py_call_typed(b.int, params[34]))
    var out = py_call(b.list, params)
    var defaults = Python.list(PythonObject(-1.0), PythonObject(-1.0), PythonObject(1.0))
    var have = len(params) - fixed
    for i in range(have, 3):  # small-loop(defaults): at most three tail defaults
        _ = py_call(out.append, defaults[i])
    return out


# ---------------------------------------------------------- labels/codes


def _codes(addr: Int, n: Int, context: PythonObject) raises -> PythonObject:
    """`_labels.finite_integer_codes` of a float32 vector, as `(k, dense,
    top)`: the number of distinct values, whether they are exactly
    0..k-1, and the largest; None where `finite_integer_codes` answers None
    (a value that is not a finite nonnegative integer)."""
    var classes = trees_vector(n, "<f8", context)
    var codes = trees_vector(n, "<i4", context)
    var k = 0
    try:
        k = Int(py=py_call_typed(trees_helper(context, "prepare_targets"), PythonObject(addr), PythonObject(0),
                                PythonObject(n), PythonObject(trees_address(classes)), PythonObject(trees_address(codes)),
                                PythonObject(0), PythonObject(0), PythonObject(0)))
        _ = codes  # keep the Python object alive through the native call above
        _ = classes  # keep the Python object alive through the native call above
    except error:
        if "NaN" in String(error):
            return PythonObject(None)
        raise error^
    var p = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=trees_address(classes))
    _ = classes  # keep the Python object alive through the native call above
    var lo = p[0]
    _ = classes  # keep the Python object alive through the native call above
    var hi = p[k - 1]
    _ = classes  # keep the Python object alive through the native call above
    if lo < 0 or not isfinite(hi):
        return PythonObject(None)
    var integral = Int(py=py_call_typed(trees_helper(context, "reduce_stat"), PythonObject(trees_address(classes)),
                                       PythonObject(1), PythonObject(k), PythonObject(4)))
    _ = classes  # keep the Python object alive through the native call above
    if integral == 0:
        return PythonObject(None)
    # Sorted distinct nonnegative integers are 0..k-1 exactly when the
    # largest is k - 1.
    return Python.tuple(PythonObject(k), PythonObject(Int(hi) == k - 1), PythonObject(Int(hi)))


# ------------------------------------------------------------ groups/pairs


def _group_key(value: PythonObject, index: Int, context: PythonObject) raises -> PythonObject:
    """`ensemble._group_id_key`: the bytes CatBoost hashes for a group id."""
    var b = _b()
    if Bool(py=py_call(b.isinstance, value, b.bytes)):
        return value
    if Bool(py=py_call(b.isinstance, value, b.str)):
        return py_call(value.encode, "utf-8")
    if Bool(py=py_call(b.isinstance, value, context["trees_integral_type"])) and not Bool(py=py_call(b.isinstance, value, b.bool)):
        return py_call(py_call(b.str, py_call_typed(b.int, value)).encode, "ascii")
    raise Error("mojolearn: group_id[" + String(index) + "] object (" + _repr(value) + ") is unsuitable "
                + "(should be string or integral type)")


def _listed(value: PythonObject, context: PythonObject) raises -> PythonObject:
    """`value.tolist()` for an exact ndarray or `array.array`, a list as
    is, else `list(value)`."""
    var b = _b()
    var cls = py_type(value)
    var ndarray = py_get(context, "ndarray")
    if ndarray is not None and cls is ndarray:
        return py_call(ndarray.tolist, value)
    if cls is context["array_type"]:
        return py_call(context["array_type"].tolist, value)
    if cls is b.list:
        return value
    return py_call(b.list, value)


def _group_sizes(group_id: PythonObject, n_rows: Int, context: PythonObject) raises -> PythonObject:
    """`ensemble._group_sizes`: the run lengths of consecutive equal ids, a
    repeated id refused (their `adjacent_find`)."""
    var b = _b()
    var ids = PythonObject(None)
    try:
        ids = _listed(group_id, context)
    except error:
        if String(error).startswith("TypeError"):
            raise Error("mojolearn: Invalid group_id type=" + _str(py_type(group_id)) + ": must be "
                        + "array like.")
        raise error^
    if len(ids) != n_rows:
        raise Error("mojolearn: Length of group_id=" + String(len(ids)) + " and length of "
                    + "data=" + String(n_rows) + " are different.")
    if n_rows > 0xFFFFFFFF:
        raise Error("mojolearn: group_id needs at most 2**32 - 1 rows")
    var sizes = List[Int]()
    var seen = py_call(b.set)
    var last = PythonObject(None)
    for i in range(n_rows):  # cpu-route: group_id label objects (int/str/bytes) checked for contiguous runs [py-data-loop]
        var value = ids[i]
        var key = _group_key(value, i, context)
        if i > 0 and Bool(py=key == last):
            sizes[len(sizes) - 1] += 1
            continue
        if py_contains(seen, key):
            raise Error("mojolearn: group Ids are not consecutive: the rows of group "
                        + _repr(value) + " are split into more than one run (row " + String(i) + "); "
                        + "CatBoost requires every group's rows to be contiguous")
        _ = py_call(seen.add, key)
        sizes.append(1)
        last = key
    var out = trees_vector(len(sizes), "<u4", context)
    var p = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=trees_address(out))
    for i in range(len(sizes)):  # cpu-route: group_id run lengths published from the label input step
        p[i] = UInt32(sizes[i])
    return out


def _pairs_arrays(pairs: PythonObject, pairs_weight: PythonObject, n_rows: Int,
                  context: PythonObject) raises -> PythonObject:
    """`ensemble._pairs_arrays`: (uint32 winner/loser pairs, float32 pair
    weights) with its refusals."""
    var b = _b()
    var rows = _listed(pairs, context)
    var m = len(rows)
    if m == 0:
        raise Error("mojolearn: pairs is empty")
    var flat = trees_vector(2 * m, "<u4", context)
    var fp = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=trees_address(flat))
    for i in range(m):  # cpu-route: user pairs list of Python index pairs validated and flattened [py-data-loop]
        var pair = rows[i]
        if len(pair) != 2:
            raise Error("mojolearn: Length of pairs[" + String(i) + "] isn't equal to 2.")
        for j in range(2):  # cpu-route: user pairs list of Python index pairs validated and flattened [py-data-loop]
            var index = pair[j]
            if not Bool(py=py_call(b.isinstance, index, context["trees_integral_type"])) or Bool(py=py_call(b.isinstance, index, b.bool)):
                raise Error("mojolearn: Invalid pairs[" + String(i) + "][" + String(j) + "] = '" + _str(index) + "' value "
                            + "type=" + _str(py_type(index)) + ": must be an integer.")
            var v = Int(py=py_call_typed(b.int, index))
            if not (0 <= v and v < n_rows):
                raise Error("mojolearn: pairs[" + String(i) + "][" + String(j) + "] = " + String(v) + " is outside the "
                            + String(n_rows) + " rows")
        var w0 = Int(py=py_call_typed(b.int, pair[0]))
        var w1 = Int(py=py_call_typed(b.int, pair[1]))
        if w0 == w1:
            raise Error("mojolearn: pairs[" + String(i) + "] pairs row " + String(w0) + " with itself")
        fp[2 * i] = UInt32(w0)
        fp[2 * i + 1] = UInt32(w1)
    var weights = trees_vector(m, "<f4", context)
    var wp = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=trees_address(weights))
    if pairs_weight is None:
        for i in range(m):  # cpu-route: user pairs list of Python index pairs validated and flattened [py-data-loop]
            wp[i] = 1.0
    else:
        var values = _listed(pairs_weight, context)
        if len(values) != m:
            raise Error("mojolearn: len(pairs_weight) = " + String(len(values)) + " is not equal to "
                        + "len(pairs) = " + String(m) + " ")
        for i in range(m):  # cpu-route: user pairs list of Python index pairs validated and flattened [py-data-loop]
            var w = values[i]
            var ok = Bool(py=py_call(b.isinstance, w, context["trees_real_type"]))
            if ok:
                var f = _float(w)
                ok = isfinite(f) and f >= 0
            if not ok:
                raise Error("mojolearn: pairs_weight[" + String(i) + "] must be a finite nonnegative number")
            wp[i] = Float32(_float(w))
    return Python.tuple(flat, weights)


# ----------------------------------------------------------------- inputs


def _flags(fields: PythonObject, model: PythonObject, n_features: Int, context: PythonObject) raises -> PythonObject:
    """`GradientBoosting._flags`: one packed word per feature (bit 0
    categorical, bit 1 one-hot), or None."""
    var b = _b()
    var cat = _field(fields, model, "cat_features")
    var hot = _field(fields, model, "one_hot_features")
    if not _truthy(cat) and not _truthy(hot):
        return PythonObject(None)
    var out = trees_vector(n_features, "<u4", context)
    var p = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=trees_address(out))
    var lists = Python.tuple(cat, hot)
    var names: List[String] = ["cat_features", "one_hot_features"]
    for which in range(2):  # small-loop(lists): the two feature-list arguments
        var values = lists[which]
        if not _truthy(values):
            continue
        var listed = py_call(b.list, values)
        for k in range(len(listed)):  # driver-loop: over the feature-list argument indices
            var i = listed[k]
            if not (Bool(py=PythonObject(0) <= i) and Bool(py=i < PythonObject(n_features))):
                raise Error("mojolearn: " + names[which] + " index " + _str(i) + " out of range for "
                            + String(n_features) + " features")
            var at = Int(py=py_call_typed(b.int, i))
            p[at] = p[at] | UInt32(1 if which == 0 else 2)
    return out


def _eval_set(eval_set: PythonObject, fields: PythonObject, model: PythonObject, n_features: Int,
              context: PythonObject) raises -> PythonObject:
    """`GradientBoosting._eval_arrays`: (X_eval column-major, y_eval, rows)
    as (keep_x, x_addr, keep_y, y_addr, rows), or None."""
    var b = _b()
    if eval_set is None:
        return PythonObject(None)
    var pair = eval_set
    if Bool(py=py_call(b.isinstance, eval_set, b.list)):
        if len(eval_set) != 1:
            raise Error("mojolearn: eval_set takes ONE (X, y) pair; the "
                        + "boosting carries a single test cursor. CatBoost "
                        + "accepts a list, this takes " + String(len(eval_set)) + " as an "
                        + "error rather than scoring the first and dropping "
                        + "the rest")
        pair = eval_set[0]
    var items = PythonObject(None)
    try:
        items = py_call(b.tuple, pair)
    except:
        raise Error("mojolearn: eval_set must be (X_eval, y_eval)")
    if len(items) != 2:
        raise Error("mojolearn: eval_set must be (X_eval, y_eval)")
    var xe = trees_f32(items[0], 2, "F", "eval_set X", context)
    if xe.cols != n_features:
        raise Error("mojolearn: eval_set X has " + String(xe.cols) + " features "
                    + "for a fit on " + String(n_features))
    if xe.rows == 0:
        raise Error("mojolearn: eval_set X has no rows")
    var ye = trees_f32(items[1], 1, "C", "eval_set y", context)
    if ye.rows != xe.rows:
        raise Error("mojolearn: eval_set y has " + String(ye.rows) + " values for "
                    + String(xe.rows) + " rows")
    if not trees_all_finite(ye.addr, ye.rows, False, context):
        var multi = _in(_field(fields, model, "loss"), _tables(context)["multi_output_losses"])
        raise Error("mojolearn: " + ("eval_set y (the eval_set labels)" if multi else "eval_set y")
                    + " must be finite (no NaN or infinity)")
    return Python.tuple(xe.keep, PythonObject(xe.addr), ye.keep, PythonObject(ye.addr), PythonObject(xe.rows))


def _has_nan(x: TreesMatrix, context: PythonObject) raises -> Bool:
    """Whether a float32 X holds a NaN (an infinity is not one): the label
    encoder refuses a NaN by name and accepts an infinity. Reached only when
    the finite scan already failed."""
    var n = x.rows * x.cols
    var classes = trees_vector(n, "<f8", context)
    var codes = trees_vector(n, "<i4", context)
    try:
        _ = py_call_typed(trees_helper(context, "prepare_targets"), PythonObject(x.addr), PythonObject(0), PythonObject(n),
                          PythonObject(trees_address(classes)), PythonObject(trees_address(codes)),
                          PythonObject(0), PythonObject(0), PythonObject(0))
        _ = codes  # keep the Python object alive through the native call above
        _ = classes  # keep the Python object alive through the native call above
    except error:
        if "NaN" in String(error):
            return True
        raise error^
    return False


def _bias_of(text: String) raises -> Float64:
    """The `bias ...` line's float64 bits (0.0 without one)."""
    var at = -1
    if text.startswith("bias "):
        at = 0
    else:
        var found = text.find("\nbias ")
        if found >= 0:
            at = found + 1
    if at < 0:
        return 0.0
    var end = text.find("\n", at)
    var line = String(text[byte=at:text.byte_length()]) if end < 0 else String(text[byte=at:end])
    var words = line.split()
    if len(words) < 2:
        raise Error("IndexError: list index out of range")
    var second = String(words[1])
    var slash = second.find("/")
    if slash < 0:
        raise Error("IndexError: list index out of range")
    var rest = String(second[byte=slash + 1:second.byte_length()])
    var next = rest.find("/")
    var hex = rest if next < 0 else String(rest[byte=0:next])
    var bits = UInt64(0)
    var digits = hex.as_bytes()
    for i in range(len(digits)):  # small-loop(hex): at most sixteen hex digits of one word
        var c = Int(digits[i])
        var d = -1
        if c >= 48 and c <= 57:
            d = c - 48
        elif c >= 97 and c <= 102:
            d = c - 87
        elif c >= 65 and c <= 70:
            d = c - 55
        if d < 0:
            raise Error("invalid literal for int() with base 16: '" + hex + "'")
        bits = (bits << 4) | UInt64(d)
    return bitcast[DType.float64](bits)


def _f64_array(values: PythonObject, context: PythonObject) raises -> PythonObject:
    """`Array.from_list(values, "<f8")` of a binding's list of floats."""
    var store = py_call(context["array_type"], "d", values)
    return classical_wrap_array(store, Python.tuple(len(store)), "<f8", "C", context)


# -------------------------------------------------------------------- fit


def gbdt_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`GradientBoosting.fit(X, y, sample_weight, eval_set, group_id,
    subgroup_id, pairs, pairs_weight)`."""
    var b = _b()
    var t = _tables(context)
    var fields = py_fields(model)
    var X = args[0]
    var y = args[1]
    var sample_weight = args[2]
    var eval_set = args[3]
    var group_id = args[4]
    var subgroup_id = args[5]
    var pairs = args[6]
    var pairs_weight = args[7]
    # A refit drops the previous model's device copy here (its holder
    # releases it), not at the next call.
    _ = py_call(fields.pop, "_resident", PythonObject(None))
    var table = gbdt_table(fields, context)
    var cat = _field(fields, model, "cat_features")
    var hot = _field(fields, model, "one_hot_features")
    var row_major_ok = False
    var probe = trees_entry(table, "gbdt_fit_row_major_available")
    if probe is not None:
        try:
            row_major_ok = Int(py=py_call_typed(b.int, py_call_typed(probe))) != 0
        except:
            row_major_ok = False
    var x: TreesMatrix
    if not _truthy(cat) and not _truthy(hot) and row_major_ok:
        x = trees_forest_layout(X, "X", context)
    else:
        x = trees_f32(X, 2, "F", "X", context)
    var n_rows = x.rows
    var n_features = x.cols
    if n_rows == 0 or n_features == 0:
        raise Error("mojolearn: fit requires at least one row and one feature")
    var loss = _field(fields, model, "loss")
    var target_dim = 1
    var ya: TreesMatrix
    if _in(loss, t["multi_regression_losses"]):
        ya = trees_f32(y, 2, "F", "y", context)
        target_dim = ya.cols
        if target_dim < 2:
            raise Error("mojolearn: " + _str(loss) + " needs y of shape (n_samples, "
                        + "n_targets) with n_targets >= 2, got " + String(target_dim) + " "
                        + "(CatBoost: \"Only one class found, can't learn "
                        + "multiclass objective\", multiclass_targets.h:167)")
        if eval_set is not None:
            raise Error("NotImplementedError: mojolearn: loss=" + _repr(loss) + " with eval_set is not "
                        + "carried: the held-out arm's target is one value per row")
        if group_id is not None or pairs is not None:
            raise Error("mojolearn: loss=" + _repr(loss) + " does not read group_id "
                        + "or pairs")
        if _field(fields, model, "class_weights") is not None:
            raise Error("mojolearn: class_weights do not apply to loss=" + _repr(loss))
        if _truthy(cat) or _truthy(hot):
            raise Error("NotImplementedError: mojolearn: loss=" + _repr(loss) + " with cat_features or "
                        + "one_hot_features is not carried: the CTR target "
                        + "binarization reads one target per row")
        if _truthy(_field(fields, model, "boost_from_average")):
            raise Error("NotImplementedError: mojolearn: boost_from_average=True with loss="
                        + _repr(loss) + " is not carried: its StartingPoint is "
                        + "per-dimension (optimal_const_for_loss.h:230-239) and "
                        + "this model's bias is one value (DEVIATION 5951)")
    else:
        ya = trees_f32(y, 1, "C", "y", context)
    if ya.rows != n_rows:
        raise Error("mojolearn: y has " + String(ya.rows) + " values for " + String(n_rows) + " rows")
    var multi_output = _in(loss, t["multi_output_losses"])
    if not trees_all_finite(ya.addr, ya.rows * ya.cols, False, context):
        raise Error("mojolearn: " + ("y (the class labels)" if multi_output else "y")
                    + " must be finite (no NaN or infinity)")
    if _eq(_field(fields, model, "nan_mode"), "Forbidden") and not trees_all_finite(x.addr, n_rows * n_features, False, context):
        if _has_nan(x, context):
            raise Error("mojolearn: nan_mode='Forbidden' but X contains NaN. "
                        + "CatBoost refuses this pair; this implementation would otherwise "
                        + "bin the NaNs silently with no NaN bin. Use "
                        + "nan_mode='Min' or 'Max', or clean the column.")
    var n_classes = 0
    if multi_output:
        var found = _codes(ya.addr, n_rows, context)
        if found is None:
            raise Error("mojolearn: " + _str(loss) + " labels must be finite "
                        + "nonnegative integer class codes")
        n_classes = Int(py=found[0])
        if n_classes < 2:
            raise Error("mojolearn: " + _str(loss) + " needs at least two classes")
        if not Bool(py=found[1]):
            raise Error("mojolearn: " + _str(loss) + " labels must be dense class "
                        + "codes 0..k-1; encode labels before fitting")
        var cw = _field(fields, model, "class_weights")
        if cw is not None and len(cw) != n_classes:
            raise Error("mojolearn: class_weights needs " + String(n_classes) + " entries "
                        + "for " + _str(loss))
    var flags = _flags(fields, model, n_features, context)
    var n_flags = 0 if flags is None else n_features
    var flags_holder = flags if flags is not None else trees_vector(1, "<u4", context)
    var w_addr = x.addr
    var n_weights = 0
    var weights: Optional[TreesMatrix] = None
    if sample_weight is not None:
        weights = trees_check_sample_weight(sample_weight, n_rows, context)
        w_addr = weights.value().addr
        n_weights = n_rows
    var held = _eval_set(eval_set, fields, model, n_features, context)
    var n_eval_rows = 0 if held is None else Int(py=held[4])
    if n_eval_rows and multi_output:
        var found = _codes(Int(py=held[3]), n_eval_rows, context)
        if found is None or Int(py=found[2]) >= n_classes:
            raise Error("mojolearn: eval_set labels must be integer class codes "
                        + "within the training class range")
    var eval_constant = False
    if n_eval_rows:
        var helper = trees_helper(context, "reduce_stat")
        var lo = py_call_typed(helper, held[3], PythonObject(0), PythonObject(n_eval_rows), PythonObject(0))
        var hi = py_call_typed(helper, held[3], PythonObject(0), PythonObject(n_eval_rows), PythonObject(1))
        eval_constant = Bool(py=lo == hi)
    var params = _params(fields, model, n_rows, n_features, n_flags, n_weights, n_eval_rows, eval_constant, context)
    py_setitem(fields, "learning_rate_", py_call_typed(b.float, params[7]))
    if target_dim != 1:
        params = _padded_tail(params)
        _ = py_call(params.append, PythonObject(target_dim))
    if subgroup_id is not None:
        raise Error("NotImplementedError: mojolearn: subgroup_id is read by no loss this implementation "
                    + "trains; the ranking losses that read CatBoost's subgroup ids "
                    + "are not implemented")
    if pairs_weight is not None and pairs is None:
        raise Error("mojolearn: pairs_weight needs pairs")
    if pairs is not None:
        if not _eq(loss, "PairLogit"):
            raise Error("NotImplementedError: mojolearn: pairs are read by loss='PairLogit' only; "
                        + "loss=" + _repr(loss) + " does not use them")
        if group_id is None:
            raise Error("NotImplementedError: mojolearn: pairs without group_id are not implemented: "
                        + "the CatBoost reference then regroups and reorders the "
                        + "whole pool by the pairs' connected components; pass "
                        + "group_id")
    var group_holder = PythonObject(None)
    var pair_holders = PythonObject(None)
    if group_id is not None:
        group_holder = _group_sizes(group_id, n_rows, context)
        params = _padded_tail(params)
        _ = py_call(params.append, PythonObject(trees_address(group_holder)))
        _ = py_call(params.append, PythonObject(Int(py=group_holder.size)))
        if _eq(loss, "PairLogit"):
            if pairs is None:
                _ = py_call(params.append, PythonObject(trees_address(group_holder)))
                _ = py_call(params.append, PythonObject(-1))
                _ = py_call(params.append, PythonObject(trees_address(group_holder)))
            else:
                pair_holders = _pairs_arrays(pairs, pairs_weight, n_rows, context)
                _ = py_call(params.append, PythonObject(trees_address(pair_holders[0])))
                _ = py_call(params.append, PythonObject(Int(py=pair_holders[1].size)))
                _ = py_call(params.append, PythonObject(trees_address(pair_holders[1])))
    var od = _field(fields, model, "od_type")
    if od is not None and not _in(od, t["od_types"]):
        raise Error("mojolearn: od_type must be one of " + _repr(t["od_types"]) + ", got " + _repr(od))
    var bootstrap = _field(fields, model, "bootstrap_type")
    var strs = Python.list(loss, PythonObject("") if not _truthy(bootstrap) else bootstrap,
                           PythonObject("") if not _truthy(od) else od, _field(fields, model, "nan_mode"))
    var border = py_get(fields, "feature_border_type", PythonObject("GreedyLogSum"))
    if not _in(border, t["border_types"]):
        raise Error("mojolearn: feature_border_type must be one of " + _repr(t["border_types"]) + ", got " + _repr(border))
    var boosting = _resolved_boosting_type(fields, model, n_rows, context)
    _check_ordered_options(boosting, fields, model, context)
    var fold_len = _float(py_get(fields, "fold_len_multiplier", PythonObject(2.0)))
    var fold_block = py_get(fields, "fold_permutation_block", PythonObject(None))
    if boosting == "Plain":
        if fold_len != 2.0 or fold_block is not None:
            raise Error("mojolearn: fold_len_multiplier and fold_permutation_block "
                        + "are read only by Ordered boosting, and this fit resolved "
                        + "to Plain; they would be accepted and ignored")
        if _field(fields, model, "permutation_count") is not None and not _truthy(cat):
            raise Error("mojolearn: permutation_count is read only by the "
                        + "categorical path and by Ordered boosting, and this fit "
                        + "resolved to Plain with no cat_features; it would be "
                        + "accepted and ignored")
    if not _eq(border, "GreedyLogSum") or boosting == "Ordered":
        _ = py_call(strs.append, border)
    if boosting == "Ordered":
        _ = py_call(strs.append, PythonObject("Ordered"))
        _ = py_call(strs.append, PythonObject(String(bitcast[DType.uint64](fold_len))))
        _ = py_call(strs.append, PythonObject(String(0 if fold_block is None else Int(py=py_call_typed(b.int, fold_block)))))
    py_setitem(fields, "boosting_type_", PythonObject(boosting))
    var ex_addr = x.addr if held is None else Int(py=held[1])
    var ey_addr = ya.addr if held is None else Int(py=held[3])
    var name = String("gbdt_fit_rowmajor") if x.row_major else String("gbdt_fit")
    var entry = trees_need(table, name, "mojolearn: this gbdt binding has no " + name, "AttributeError")
    var call = Python.list(PythonObject(x.addr), PythonObject(ya.addr), PythonObject(w_addr),
                           PythonObject(trees_address(flags_holder)), PythonObject(ex_addr), PythonObject(ey_addr),
                           params, strs)
    _ = flags_holder  # keep the Python object alive through the native call above
    var out = trees_call(entry, call)
    _ = weights^
    _ = held^
    _ = group_holder^
    _ = pair_holders^
    var text = out[0]
    py_setitem(fields, "model_", text)
    py_setitem(fields, "bias_", PythonObject(_bias_of(String(py_call(b.str, text)))))
    py_setitem(fields, "best_iteration_", py_call_typed(b.int, out[1]))
    py_setitem(fields, "stopped_early_", PythonObject(_truthy(out[2])))
    py_setitem(fields, "loss_curve_", _f64_array(out[3], context))
    py_setitem(fields, "test_loss_curve_", _f64_array(out[4], context) if n_eval_rows else PythonObject(None))
    py_setitem(fields, "n_features_in_", PythonObject(n_features))
    var dim = PythonObject(None)
    if gbdt_has_ctr(text) and gbdt_vendor(table) == "cpu":
        dim = py_call_typed(b.int, host_archive_gbdt_state(py_call(b.str, text), context)["dim"])
    else:
        dim = py_call_typed(trees_need(table, "gbdt_model_dim", "mojolearn: this gbdt binding has no gbdt_model_dim",
                                       "AttributeError"), text)
    py_setitem(fields, "approx_dim_", dim)
    if _eq(loss, "MultiClass"):
        py_setitem(fields, "n_classes_", dim + PythonObject(1))
    elif _eq(loss, "MultiClassOneVsAll"):
        py_setitem(fields, "n_classes_", dim)
    else:
        py_setitem(fields, "n_classes_", PythonObject(None))
    _ = x^
    _ = ya^
    return model


def gbdt_inspect(model: PythonObject, operation: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """The fit's resolved settings, for inspection: `_params(...)`,
    `_resolved_learning_rate(...)`, `_resolved_boosting_type(n_rows)`."""
    var fields = py_fields(model)
    if operation == "trees_gbdt_boosting_type":
        return PythonObject(_resolved_boosting_type(fields, model, _int(args[0]), context))
    if operation == "trees_gbdt_learning_rate":
        return PythonObject(_resolved_learning_rate(fields, model, _int(args[0]), _int(args[1]), _truthy(args[2]), context))
    return _params(fields, model, _int(args[0]), _int(args[1]), _int(args[2]), _int(args[3]), _int(args[4]),
                   _truthy(args[5]), context)


# ------------------------------------------------------------ constructor


def _real(value: PythonObject, context: PythonObject) raises -> Bool:
    return not _is_bool(value) and Bool(py=py_call(_b().isinstance, value, context["trees_real_type"]))


def _parsed_real(value: PythonObject, context: PythonObject) raises -> Float64:
    """`float(value)` of a real non-bool, NaN for anything else or a
    refused conversion (the constructor's `parsed_*` pattern)."""
    var nan = bitcast[DType.float64](UInt64(0x7FF8000000000000))
    if not _real(value, context):
        return nan
    try:
        return _float(value)
    except:
        return nan


def gbdt_initialize(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`GradientBoosting.__init__`: every refusal in the constructor's
    order and wording, then the stored parameters (normalized as it
    normalized them) and the unfitted state."""
    var b = _b()
    var t = _tables(context)
    var names = t["init_names"]
    var kw = Python.dict()
    for i in range(len(names)):  # driver-loop: over the constructor parameter names
        py_setitem(kw, names[i], args[i])
    var loss = kw["loss"]
    if not _in(loss, t["losses"]):
        raise Error("mojolearn: loss must be one of " + _repr(t["losses"]) + ", got " + _repr(loss))
    var required = py_get(t["required_param"], loss)
    if required is not None:
        if kw[required[1]] is None:
            raise Error("mojolearn: " + _str(loss) + " requires " + _str(required[1]) + "= "
                        + "(CatBoost's " + _repr(required[0]) + ", which it makes mandatory)")
    var grow = kw["grow_policy"]
    var symmetric = _eq(grow, "SymmetricTree")
    var bootstrap = kw["bootstrap_type"]
    var bootstrap_defaulted = bootstrap is None and symmetric
    var n_estimators = kw["n_estimators"]
    if n_estimators is None:
        n_estimators = t["catboost_iterations"] if symmetric else t["legacy_iterations"]
    if bootstrap is None and symmetric:
        bootstrap = t["catboost_bootstrap"]
    if bootstrap is not None and not _in(bootstrap, t["bootstrap_types"]):
        raise Error("mojolearn: bootstrap_type must be one of " + _repr(t["bootstrap_types"]) + ", got " + _repr(bootstrap))
    var subsample = kw["subsample"]
    if _eq(bootstrap, "Bayesian") and subsample is not None:
        if bootstrap_defaulted:
            raise Error("mojolearn: the default bootstrap_type='Bayesian' "
                        + "(CatBoost's GPU default, bootstrap_options.h:18) does "
                        + "not support subsample; pass bootstrap_type='Bernoulli' "
                        + "or 'Poisson' with it")
        raise Error("mojolearn: bootstrap_type='Bayesian' does not support "
                    + "subsample; it takes bagging_temperature")
    if not Bool(py=kw["bagging_temperature"] == PythonObject(1.0)) and not _eq(bootstrap, "Bayesian"):
        raise Error("mojolearn: bagging_temperature is read only by "
                    + "bootstrap_type='Bayesian'; with bootstrap_type="
                    + _repr(bootstrap) + " it would be ignored, so it is refused")
    if subsample is not None and not (_eq(bootstrap, "Bernoulli") or _eq(bootstrap, "Poisson")):
        raise Error("mojolearn: subsample is read only by bootstrap_type="
                    + "'Bernoulli' or 'Poisson'; with bootstrap_type="
                    + _repr(bootstrap) + " it would be ignored, so it is refused")
    var method = kw["leaf_estimation_method"]
    if method is not None and not _in(method, t["leaf_estimation_names"]):
        raise Error("mojolearn: leaf_estimation_method must be one of "
                    + _repr(py_call(b.tuple, t["leaf_estimation_names"])) + ", got " + _repr(method))
    if _eq(grow, "Region"):
        raise Error("NotImplementedError: mojolearn: grow_policy='Region' is EGrowPolicy::Region, "
                    + "which no lane implements (greedy_search_helper.cpp:325-350); "
                    + "reachable values are " + _repr(t["grow_policies"]))
    if not _in(grow, t["grow_policy_codes"]):
        raise Error("mojolearn: grow_policy must be one of " + _repr(t["grow_policies"]) + ", "
                    + "got " + _repr(grow))
    var fraction = _parsed_real(kw["feature_fraction"], context)
    if not isfinite(fraction) or not (fraction > 0.0 and fraction <= 1.0):
        raise Error("mojolearn: feature_fraction must be finite and in (0, 1]")
    var cat = kw["cat_features"]
    var hot = kw["one_hot_features"]
    if fraction < 1.0 and ((cat is not None and len(cat) > 0) or (hot is not None and len(hot) > 0)):
        raise Error("NotImplementedError: mojolearn: feature_fraction < 1 requires numeric features; "
                    + "categorical, one-hot and CTR features are unsupported")
    var gain_obj = kw["min_split_gain"]
    var gain = PythonObject(None)
    if gain_obj is not None:
        var parsed = _parsed_real(gain_obj, context)
        if not isfinite(parsed) or parsed < 0:
            raise Error("mojolearn: min_split_gain must be finite and nonnegative or None")
        if symmetric:
            raise Error("mojolearn: min_split_gain is only supported for Depthwise and Lossguide")
        gain = PythonObject(parsed)
    var non_symmetric = not symmetric
    if non_symmetric and not _in(loss, t["non_symmetric_losses"]):
        raise Error("NotImplementedError: mojolearn: Error: optimization scheme is not supported for "
                    + "GPU learning Loss=" + _str(loss) + ";OptimizationScheme=" + _str(grow) + " "
                    + "-- CatBoost's GPU registers a non-symmetric trainer for "
                    + "exactly eleven losses (pointwise_non_symmetric.cpp:7-29; "
                    + "train.cpp:279 is the refusal): " + _repr(py_call(b.sorted, t["non_symmetric_losses"])))
    var max_leaves = kw["max_leaves"]
    var depth = kw["max_depth"]
    if max_leaves is not None:
        var leaves = Int(py=py_call_typed(b.int, max_leaves))
        if leaves < 2:
            raise Error("mojolearn: max_leaves must be at least 2, got " + _str(max_leaves))
        var pinned = 1 << Int(py=py_call_typed(b.int, depth))
        if not _eq(grow, "Lossguide") and leaves != pinned:
            raise Error("mojolearn: max_leaves option works only with lossguide "
                        + "tree growing (catboost_options.cpp:998); under "
                        + "grow_policy=" + _repr(grow) + " CatBoost pins it to "
                        + "2 ** max_depth == " + String(pinned) + ", got " + _str(max_leaves))
        if _eq(grow, "Lossguide") and leaves > 65536:
            raise Error("mojolearn: Maximum leaves count for Lossguide grow "
                        + "policy is 65536, got " + _str(max_leaves))
    var min_data = kw["min_data_in_leaf"]
    var min_data_int = Int(py=py_call_typed(b.int, min_data))
    if min_data_int < 0:
        raise Error("mojolearn: min_data_in_leaf must be >= 0, got " + _str(min_data))
    if not non_symmetric and min_data_int != 1:
        raise Error("mojolearn: min_data_in_leaf=" + _str(min_data) + " does "
                    + "nothing under grow_policy='SymmetricTree' -- CatBoost's "
                    + "IsTerminalLeaf tests the leaf size only for Depthwise and "
                    + "Lossguide (greedy_search_helper.cpp:685) and discards the "
                    + "value on oblivious trees. Use a non-symmetric grow_policy "
                    + "or leave it at 1.")
    var score = kw["score_function"]
    if score is None:
        score = PythonObject("NewtonL2") if _eq(grow, "Lossguide") else PythonObject("Cosine")
    var unsupported = py_get(t["unsupported_score_functions"], score)
    if unsupported is not None:
        raise Error("NotImplementedError: mojolearn: score_function=" + _repr(score) + " is not "
                    + "honored here -- " + _str(unsupported) + ". Reachable values are " + _repr(t["score_functions"]) + ".")
    if not _in(score, t["score_function_names"]):
        raise Error("mojolearn: score_function must be one of " + _repr(t["score_functions"]) + ", got " + _repr(score))
    var hessian_obj = kw["min_child_hessian"]
    var hessian = PythonObject(None)
    if hessian_obj is not None:
        var parsed = _parsed_real(hessian_obj, context)
        if not isfinite(parsed) or not (parsed >= 0 and parsed <= _F32_MAX):
            raise Error("mojolearn: min_child_hessian must be finite nonnegative and <= Float32.MAX_FINITE or None")
        if symmetric:
            raise Error("mojolearn: min_child_hessian requires Depthwise or Lossguide")
        if not (_eq(score, "NewtonL2") or _eq(score, "NewtonCosine")):
            raise Error("mojolearn: min_child_hessian requires NewtonL2 or NewtonCosine")
        if not (_eq(loss, "RMSE") or _eq(loss, "Logloss") or _eq(loss, "CrossEntropy")):
            raise Error("mojolearn: min_child_hessian supports RMSE, Logloss and CrossEntropy only")
        hessian = PythonObject(parsed)
    var nan_mode = kw["nan_mode"]
    if not _in(nan_mode, t["nan_modes"]):
        raise Error("mojolearn: nan_mode must be one of " + _repr(t["nan_modes"]) + ", got " + _repr(nan_mode))
    var border = kw["feature_border_type"]
    if not _in(border, t["border_types"]):
        raise Error("mojolearn: feature_border_type must be one of " + _repr(t["border_types"]) + ", got " + _repr(border))
    # `_validate_search_options` over the arguments.
    var probe = Python.dict()
    py_setitem(probe, "grow_policy", grow)
    py_setitem(probe, "score_function", score)
    py_setitem(probe, "use_pointwise_searcher", kw["use_pointwise_searcher"])
    py_setitem(probe, "random_strength", kw["random_strength"])
    var strength = _validate_search_options(probe, model, context)
    var random_strength = PythonObject(None) if kw["random_strength"] is None else PythonObject(strength)
    var border_rows = kw["border_build_max_samples"]
    if Bool(py=border_rows < PythonObject(0)):
        raise Error("mojolearn: border_build_max_samples must be >= 0 (0 means "
                    + "every row), got " + _str(border_rows))
    var class_weights = kw["class_weights"]
    var stored_weights = PythonObject(None)
    if class_weights is not None:
        if len(class_weights) == 0:
            raise Error("mojolearn: class_weights is empty; pass None for none")
        var listed = py_call(b.list, class_weights)
        stored_weights = Python.list()
        for i in range(len(listed)):  # driver-loop: over the class_weights argument
            var w = _float(listed[i])
            if not isfinite(w) or w < 0:
                raise Error("mojolearn: class_weights must have finite nonnegative entries")
        if _eq(loss, "RMSE"):
            raise Error("mojolearn: class_weights is not accepted with "
                        + "loss='RMSE', which has no class structure to weight")
        for i in range(len(listed)):  # driver-loop: over the class_weights argument
            _ = py_call(stored_weights.append, py_call_typed(b.float, listed[i]))
    var boosting = kw["boosting_type"]
    if boosting is not None and not _in(boosting, t["boosting_types"]):
        raise Error("mojolearn: boosting_type must be one of " + _repr(t["boosting_types"]) + " or "
                    + "None, got " + _repr(boosting))
    var ordered_probe = Python.dict()
    py_setitem(ordered_probe, "loss", loss)
    py_setitem(ordered_probe, "grow_policy", grow)
    py_setitem(ordered_probe, "score_function", score)
    py_setitem(ordered_probe, "leaf_estimation_method", method)
    py_setitem(ordered_probe, "use_pointwise_searcher", kw["use_pointwise_searcher"])
    py_setitem(ordered_probe, "feature_fraction", PythonObject(fraction))
    _check_ordered_options(_str(boosting) if boosting is not None else String(""), ordered_probe, model, context)
    var fold_len = kw["fold_len_multiplier"]
    var fold_ok = _real(fold_len, context)
    if fold_ok:
        var f = _float(fold_len)
        fold_ok = isfinite(f) and f > 1.0
    if not fold_ok:
        raise Error("mojolearn: fold len multiplier should be greater than 1, got " + _repr(fold_len))
    var block = kw["fold_permutation_block"]
    if block is not None:
        var ok = not _is_bool(block) and Bool(py=py_call(b.isinstance, block, context["trees_integral_type"]))
        if ok:
            var v = Int(py=py_call_typed(b.int, block))
            ok = 0 <= v and v <= 256
        if not ok:
            raise Error("mojolearn: fold_permutation_block must be an integer in "
                        + "0..256 or None, got " + _repr(block))
    if _eq(boosting, "Plain") and (_float(fold_len) != 2.0 or block is not None):
        raise Error("mojolearn: fold_len_multiplier and fold_permutation_block are "
                    + "read only by Ordered boosting (dynamic_boosting.h:115-223); "
                    + "with boosting_type='Plain' they would be accepted and ignored")
    var perm_names: List[String] = ["permutation_count", "ctr_estimation_permutation_id"]
    for i in range(2):  # small-loop(perm_names): the two permutation arguments
        var value = kw[perm_names[i]]
        if value is None:
            continue
        if Int(py=py_call_typed(b.int, value)) < 0:
            raise Error("mojolearn: " + perm_names[i] + " must be >= 0 when given, got " + _str(value))
        var may_be_ordered = i == 0 and not _eq(boosting, "Plain") and symmetric
        if not _truthy(cat) and not may_be_ordered:
            raise Error("mojolearn: " + perm_names[i] + " is read only by the categorical "
                        + "path (doc_parallel_dataset_builder.cpp:190-262)"
                        + (" and by Ordered boosting" if i == 0 else "")
                        + "; with no cat_features it would be accepted and "
                        + "ignored. Pass cat_features= or leave it unset.")
    var fields = py_fields(model)
    var stored = Python.list(
        Python.tuple("loss", loss), Python.tuple("n_estimators", n_estimators), Python.tuple("max_depth", depth),
        Python.tuple("learning_rate", kw["learning_rate"]), Python.tuple("l2_leaf_reg", kw["l2_leaf_reg"]),
        Python.tuple("border_count", kw["border_count"]), Python.tuple("random_state", kw["random_state"]),
        Python.tuple("loss_alpha", kw["loss_alpha"]), Python.tuple("loss_q", kw["loss_q"]),
        Python.tuple("loss_delta", kw["loss_delta"]), Python.tuple("loss_variance_power", kw["loss_variance_power"]),
        Python.tuple("loss_border", kw["loss_border"]), Python.tuple("leaf_estimation_method", method),
        Python.tuple("leaf_estimation_iterations", kw["leaf_estimation_iterations"]),
        Python.tuple("bootstrap_type", bootstrap), Python.tuple("bagging_temperature", kw["bagging_temperature"]),
        Python.tuple("subsample", subsample), Python.tuple("cat_features", cat), Python.tuple("one_hot_features", hot),
        Python.tuple("od_type", kw["od_type"]), Python.tuple("od_pvalue", kw["od_pvalue"]),
        Python.tuple("od_wait", kw["od_wait"]), Python.tuple("use_best_model", kw["use_best_model"]),
        Python.tuple("best_model_min_trees", py_call_typed(b.int, kw["best_model_min_trees"])),
        Python.tuple("score_function", score), Python.tuple("nan_mode", nan_mode),
        Python.tuple("random_strength", random_strength),
        Python.tuple("use_pointwise_searcher", PythonObject(_truthy(kw["use_pointwise_searcher"]))),
        Python.tuple("boost_from_average", kw["boost_from_average"]),
        Python.tuple("border_build_max_samples", py_call_typed(b.int, border_rows)),
        Python.tuple("class_weights", stored_weights),
        Python.tuple("permutation_count", kw["permutation_count"]),
        Python.tuple("ctr_estimation_permutation_id", kw["ctr_estimation_permutation_id"]),
        Python.tuple("grow_policy", grow),
        Python.tuple("max_leaves", PythonObject(None) if max_leaves is None else py_call_typed(b.int, max_leaves)),
        Python.tuple("min_data_in_leaf", PythonObject(min_data_int)),
        Python.tuple("min_split_gain", gain), Python.tuple("min_child_hessian", hessian),
        Python.tuple("feature_fraction", PythonObject(fraction)), Python.tuple("feature_border_type", border),
        Python.tuple("boosting_type", boosting), Python.tuple("fold_len_multiplier", py_call_typed(b.float, fold_len)),
        Python.tuple("fold_permutation_block", PythonObject(None) if block is None else py_call_typed(b.int, block)),
        Python.tuple("model_", PythonObject(None)), Python.tuple("loss_curve_", PythonObject(None)),
        Python.tuple("test_loss_curve_", PythonObject(None)), Python.tuple("best_iteration_", PythonObject(None)),
        Python.tuple("stopped_early_", PythonObject(None)), Python.tuple("learning_rate_", PythonObject(None)),
        Python.tuple("boosting_type_", PythonObject(None)), Python.tuple("n_features_in_", PythonObject(None)),
        Python.tuple("approx_dim_", PythonObject(None)), Python.tuple("n_classes_", PythonObject(None)),
    )
    for i in range(len(stored)):  # driver-loop: over the constructor's stored fields
        py_setitem(fields, stored[i][0], stored[i][1])
    return PythonObject(None)


# --------------------------------------------------------------- adapters


def _adapter_tables(context: PythonObject) raises -> PythonObject:
    return context["trees_gbdt_adapter_tables"]


def _adapter_params(model: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_GBDTAdapter.get_params()`."""
    var t = _adapter_tables(context)
    var cls = py_type(model)
    if not (py_attr(cls, "__init__") is py_attr(t["base"], "__init__")):
        raise Error("TypeError: GBDT adapter subclasses with custom constructors need their own parameter protocol")
    var fields = py_fields(model)
    var names = t["names"]
    var params = Python.dict()
    for i in range(len(names)):  # driver-loop: over the adapter's constructor parameter names
        py_setitem(params, names[i], _field(fields, model, String(names[i])))
    return params


def _new_learner(model: PythonObject, mode: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_GBDTAdapter._new_learner(mode)`: the GradientBoosting learner the
    adapter fits, built by the native constructor (its refusals are the
    learner's)."""
    var b = _b()
    if mode is not None:
        var ok = Bool(py=py_call(b.isinstance, mode, b.str))
        if ok:
            var clean = String(py_call(py_call(mode.strip).lower))
            ok = clean == "fast" or clean == "identical"
        if not ok:
            raise Error("numeric_mode must be fast, identical or None")
    var loss = py_type(model)._loss
    if _eq(loss, "RMSE") and _field(py_fields(model), model, "class_weights") is not None:
        raise Error("NotImplementedError: GradientBoostingRegressor does not support class_weights")
    var params = _adapter_params(model, context)
    py_setitem(params, "numeric_mode", mode)
    var t = _tables(context)
    var names = t["init_names"]
    var learner_cls = _adapter_tables(context)["learner"]
    var learner = py_call(b.object.__new__, learner_cls)
    var args = Python.list()
    for i in range(len(names)):  # driver-loop: over the learner's constructor parameter names
        var name = names[i]
        if String(name) == "loss":
            _ = py_call(args.append, loss)
        else:
            _ = py_call(args.append, py_get(params, name, context["trees_defaults"][learner_cls][name]))
    _ = gbdt_initialize(learner, py_call(b.tuple, args), context)
    py_setitem(py_fields(learner), "numeric_mode", mode)
    return learner


def _clear_fit(fields: PythonObject) raises:
    """`_GBDTAdapter._clear_fit`: drop every fitted attribute."""
    var keys = py_call(_b().list, fields)
    for i in range(len(keys)):  # driver-loop: over the instance's attribute names
        if String(keys[i]).endswith("_"):
            _ = py_call(fields.pop, keys[i], PythonObject(None))


def _label_kind(labels: TreesLabels, values: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    """`_metrics_impl._classification_encoded`'s kind and class values over
    labels the ORDER RULE already encoded: (kind, the sorted classes as
    `str` or `int`), with its one-dimensional and type refusals."""
    var b = _b()
    var shape_ok = True
    var cls = py_type(values)
    if cls is b.list or cls is b.tuple:
        shape_ok = not (len(values) > 0 and Bool(py=py_call(b.isinstance, values[0], Python.tuple(b.list, b.tuple))))
    elif cls is context["array_class"]:
        shape_ok = Int(py=values.ndim) == 1
    else:
        try:
            shape_ok = Int(py=native_buffer_view(values).ndim) == 1
        except:
            shape_ok = True
    if not shape_ok:
        raise Error(name + " must be one-dimensional single-label data")
    var classes = labels.classes
    var k = len(classes)
    var all_str = True
    var all_int = True
    for i in range(k):  # driver-loop: over the distinct class labels
        var c = classes[i]
        all_str = all_str and Bool(py=py_call(b.isinstance, c, b.str))
        all_int = all_int and (Bool(py=py_call(b.isinstance, c, context["trees_integral_type"])) or _is_bool(c))
    if all_str and k > 0:
        var out = Python.list()
        for i in range(k):  # driver-loop: over the distinct class labels
            _ = py_call(out.append, py_call(b.str, classes[i]))
        return Python.tuple("string", out)
    if all_int and k > 0:
        var out = Python.list()
        for i in range(k):  # driver-loop: over the distinct class labels
            _ = py_call(out.append, py_call_typed(b.int, classes[i]))
        return Python.tuple("integer", out)
    raise Error("TypeError: " + name + " must contain only strings or only integers; "
                + "floating labels, missing labels and mixed types are unsupported")


def _codes_f32(labels: TreesLabels, context: PythonObject) raises -> PythonObject:
    """The class codes as float32 (`_label_map(...).astype('<f4')`)."""
    var out = trees_vector(labels.n, "<f4", context)
    _ = py_call_typed(trees_helper(context, "cast_elements"), PythonObject(trees_address(labels.codes)), PythonObject(2),
                      PythonObject(trees_address(out)), PythonObject(0), PythonObject(labels.n))
    return out


def _adapter_eval(eval_set: PythonObject) raises -> PythonObject:
    """The adapters' eval_set shape: one (X, y) pair."""
    var b = _b()
    var pair = eval_set
    if Bool(py=py_call(b.isinstance, eval_set, b.list)):
        if len(eval_set) != 1:
            raise Error("eval_set accepts one (X, y) pair")
        pair = eval_set[0]
    if not Bool(py=py_call(b.isinstance, pair, b.tuple)) or len(pair) != 2:
        raise Error("eval_set must be (X_eval, y_eval) or a one-pair list")
    return pair


def _fit_native(model: PythonObject, X: PythonObject, y: PythonObject, sample_weight: PythonObject,
                eval_set: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_GBDTAdapter._fit_native`: the learner in the resolved mode, the
    binding's compiled mode checked, the native learner fit, the fitted
    attributes copied."""
    var b = _b()
    var fields = py_fields(model)
    var raw = _field(fields, model, "numeric_mode")
    if raw is None:
        raw = context["backend_fields"]["_DEFAULT_MODE"]
    var learner = _new_learner(model, raw, context)
    var mode = String(py_call(py_call(raw.strip).lower))
    var learner_fields = py_fields(learner)
    py_setitem(learner_fields, "numeric_mode", PythonObject(mode))
    var table = gbdt_table(learner_fields, context)
    var code = Int(py=py_call_typed(trees_need(table, "gbdt_numeric_mode", "mojolearn: this gbdt binding has no gbdt_numeric_mode",
                                               "AttributeError")))
    if code != (0 if mode == "fast" else 1):
        raise Error("RuntimeError: GBDT adapter native numeric mode disagrees with requested mode")
    _ = gbdt_fit(learner, Python.tuple(X, y, sample_weight, eval_set, PythonObject(None), PythonObject(None),
                                       PythonObject(None), PythonObject(None)), context)
    py_setitem(fields, "_learner_", learner)
    py_setitem(fields, "numeric_mode_", PythonObject(mode))
    py_setitem(fields, "n_features_in_", learner_fields["n_features_in_"])
    py_setitem(fields, "model_", learner_fields["model_"])
    var copied = Python.tuple("loss_curve_", "test_loss_curve_", "best_iteration_", "stopped_early_", "learning_rate_")
    for i in range(len(copied)):  # small-loop(copied): the five learner attributes
        py_setitem(fields, copied[i], learner_fields[copied[i]])
    return model


def adapter_classifier_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`GradientBoostingClassifier.fit(X, y, sample_weight, eval_set)`."""
    var fields = py_fields(model)
    _ = _new_learner(model, _field(fields, model, "numeric_mode"), context)
    _clear_fit(fields)
    var y = args[1]
    var labels = trees_encode_labels(y, context)
    var seen = len(labels.classes)
    if seen < 2:
        raise Error("Classifier can't train when only one class is present.")
    if seen > 2:
        raise Error("Only binary classification is supported. The type of the target "
                    + "is multiclass.")
    var kind = _label_kind(labels, y, "y", context)
    var classes = kind[1]
    var encoded = _codes_f32(labels, context)
    var eval_set = args[3]
    if eval_set is not None:
        var pair = _adapter_eval(eval_set)
        var eval_labels = trees_encode_labels(pair[1], context)
        var eval_kind = _label_kind(eval_labels, pair[1], "eval_set y", context)
        if not Bool(py=eval_kind[0] == kind[0]):
            raise Error("eval_set contains labels outside the training vocabulary")
        var table = trees_vector(len(eval_kind[1]), "<i4", context)
        var tp = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=trees_address(table))
        _ = table  # keep the Python object alive through the native call above
        for i in range(len(eval_kind[1])):  # driver-loop: over the eval set's distinct labels
            var at = -1
            for j in range(len(classes)):  # small-loop(classes): the two training classes
                if Bool(py=classes[j] == eval_kind[1][i]):
                    at = j
            if at < 0:
                raise Error("eval_set contains labels outside the training vocabulary")
            tp[i] = Int32(at)
            _ = table  # keep the Python object alive through the native call above
        var mapped = trees_vector(eval_labels.n, "<i4", context)
        _ = py_call_typed(trees_helper(context, "gather_i32"), PythonObject(trees_address(table)),
                          PythonObject(len(eval_kind[1])), PythonObject(trees_address(eval_labels.codes)),
                          PythonObject(eval_labels.n), PythonObject(trees_address(mapped)))
        _ = table  # keep the Python object alive through the native call above
        _ = mapped  # keep the Python object alive through the native call above
        var eval_y = trees_vector(eval_labels.n, "<f4", context)
        _ = py_call_typed(trees_helper(context, "cast_elements"), PythonObject(trees_address(mapped)), PythonObject(2),
                          PythonObject(trees_address(eval_y)), PythonObject(0), PythonObject(eval_labels.n))
        _ = mapped  # keep the Python object alive through the native call above
        eval_set = Python.tuple(pair[0], eval_y)
    _ = _fit_native(model, args[0], encoded, args[2], eval_set, context)
    py_setitem(fields, "classes_", classes)
    py_setitem(fields, "n_classes_", PythonObject(2))
    py_setitem(fields, "_label_kind_", kind[0])
    return model


def _regression_target(source: PythonObject, name: String, context: PythonObject) raises -> TreesMatrix:
    """`as_f32_c(y, ndim=None)` and `_regression_target`'s checks."""
    var view = trees_view(source, name, context)
    trees_check_rank(view, -1, name)
    if view.ndim() != 1 or view.size() == 0:
        raise Error(name + " must be nonempty one-dimensional targets")
    var target = trees_f32_view(view^, "C", context)
    if not trees_all_finite(target.addr, target.rows, False, context):
        raise Error(name + " must be finite")
    return target^


def adapter_regressor_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`GradientBoostingRegressor.fit(X, y, sample_weight, eval_set)`."""
    var fields = py_fields(model)
    _ = _new_learner(model, _field(fields, model, "numeric_mode"), context)
    _clear_fit(fields)
    var target = _regression_target(args[1], "y", context)
    var eval_set = args[3]
    var eval_target: Optional[TreesMatrix] = None
    if eval_set is not None:
        var pair = _adapter_eval(eval_set)
        eval_target = _regression_target(pair[1], "eval_set y", context)
        eval_set = Python.tuple(pair[0], _as_array(eval_target.value(), context))
    return _fit_native(model, args[0], _as_array(target, context), args[2], eval_set, context)


def _as_array(m: TreesMatrix, context: PythonObject) raises -> PythonObject:
    """A float32 vector as the Array the learner's fit reads (the checked
    copy itself, or the caller's own Array/buffer owner)."""
    return m.keep
