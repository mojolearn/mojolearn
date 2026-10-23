# SPDX-License-Identifier: Apache-2.0
# Hyperparameter search and the validation tools of
# python/mojolearn/model_selection.py, natively (lane mcb2-msel):
# `ParameterGrid` / `ParameterSampler` iteration, the shared `_BaseSearch.fit`
# of GridSearchCV and RandomizedSearchCV, `validation_curve`,
# `learning_curve` and `permutation_test_score`. The folds are drawn ONCE
# and X, y and the fold indices stay resident for every candidate, value,
# size or permutation (`MselFoldRows`); the loops over candidates, folds,
# sizes and permutations are this owner's; the estimator and the scorers are
# the caller's objects. The score tables use the base binding's
# `row_means_f64` / `row_stds_f64` and the permutations the same counter
# draws, so every result is unchanged.
from std.memory import memcpy
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_setattr, py_contains, py_fields
from bindings.py_attr import py_attr
from bindings.native_buffer_owner import native_storage_allocate
from bindings.classical_arrays import classical_wrap_array
from bindings.family_common import (
    FAMILY_I64P, family_call, family_call_kw, family_isinstance, family_is_str, family_repr, family_str,
    family_type_name, family_as_f32,
)
from bindings.metrics_reg_native import _Prog, _reg_run, _arena_base, _arena_int
from bindings.msel_common import (
    msel_b, msel_entry, msel_try, msel_missed, msel_catch, msel_warn, msel_is_bool, msel_is_integral, msel_is_real,
    msel_int, msel_op, msel_cmp, msel_materialize, msel_try_materialize, msel_flatten, msel_is_array, msel_addr,
    msel_p64, msel_pf64, msel_new, msel_new2, msel_f64_list, msel_f64_block, msel_i64_list, msel_i32_list,
    msel_slice_i64, msel_index_i64, msel_resident_new, msel_resident_take, msel_resident_release,
    msel_resident_close, msel_get_scorer, msel_score, MselRng, MselGroups, msel_apply,
)
from bindings.msel_split import msel_truth, msel_cls, msel_sabotage
from bindings.msel_cv import (
    msel_clone, msel_cv_folds, msel_device_rows_ok, msel_require_serial, msel_scorers, MselFoldRows, MselScores,
    msel_cross_validate_run, msel_seconds, msel_fit,
)


# ---------------------------------------------------------------- parameter grids


def msel_grid_dicts(param_grid_in: PythonObject) raises -> PythonObject:
    """`ParameterGrid(param_grid).param_grid`: the constructor's checks and
    its list of dicts."""
    var b = msel_b()
    var param_grid = param_grid_in
    if family_isinstance(param_grid, b.dict):
        param_grid = py_call(b.list, Python.tuple(param_grid))
    if not family_isinstance(param_grid, Python.tuple(b.list, b.tuple)):
        raise Error("TypeError: Parameter grid should be a dict or a list, got: " + family_repr(param_grid))
    var out = Python.list()
    for i in range(len(param_grid)):  # driver-loop: over the candidate dicts
        var g = param_grid[i]
        if not family_isinstance(g, b.dict):
            raise Error("TypeError: Parameter grid is not a dict (" + family_repr(g) + ")")
        var items = py_call(b.list, family_call(g.items))
        for j in range(len(items)):  # driver-loop: over a grid's parameters
            var k = items[j][0]
            var v = items[j][1]
            if family_is_str(v) or not msel_truth(py_call(b.hasattr, v, PythonObject("__iter__"))):
                raise Error("TypeError: Parameter grid for parameter " + family_repr(k) + " needs to be a list, got: "
                            + family_repr(v))
            if len(family_call(b.list, v)) == 0:
                raise Error("Parameter grid for parameter " + family_repr(k) + " need to be a non-empty sequence.")
        _ = py_call(out.append, family_call(b.dict, g))
    return out


def msel_grid_list(param_grid: PythonObject) raises -> PythonObject:
    """`list(ParameterGrid)` from its stored dicts: every combination of
    each dict's values, keys sorted, dicts in list order."""
    var b = msel_b()
    var itertools = Python.import_module("itertools")
    var out = Python.list()
    for i in range(len(param_grid)):  # driver-loop: over the candidate dicts
        var g = param_grid[i]
        var keys = family_call(b.sorted, g)
        if len(keys) == 0:
            _ = py_call(out.append, Python.dict())
            continue
        var values = Python.list()
        for j in range(len(keys)):  # driver-loop: over a grid's parameters
            _ = py_call(values.append, family_call(b.list, family_call(py_attr(g, "__getitem__"), keys[j])))
        var combos = family_call(b.list, msel_apply(itertools.product, py_call(b.tuple, values)))
        for c in range(len(combos)):  # driver-loop: over candidates
            _ = py_call(out.append, family_call(b.dict, py_call(b.zip, keys, combos[c])))
    return out


def msel_grid_len(param_grid: PythonObject) raises -> Int:
    """`len(ParameterGrid)`."""
    var b = msel_b()
    var total = 0
    for i in range(len(param_grid)):  # driver-loop: over the candidate dicts
        var size = 1
        var values = py_call(b.list, family_call(param_grid[i].values))
        for j in range(len(values)):  # driver-loop: over a grid's parameters
            size *= len(family_call(b.list, values[j]))
        total += size
    return total


def msel_sampler_check(param_distributions_in: PythonObject) raises -> PythonObject:
    """`ParameterSampler.__init__`'s checks; returns its distributions."""
    var b = msel_b()
    var dists = param_distributions_in
    if family_isinstance(dists, b.dict):
        dists = py_call(b.list, Python.tuple(dists))
    var seq = family_call(b.list, dists)
    for i in range(len(seq)):  # driver-loop: over the candidate dicts
        var items = py_call(b.list, family_call(seq[i].items))
        for j in range(len(items)):  # driver-loop: over a grid's parameters
            var k = items[j][0]
            var v = items[j][1]
            if msel_truth(py_call(b.hasattr, v, PythonObject("rvs"))):
                raise Error("NotImplementedError: mojolearn ParameterSampler: " + family_repr(k) + " is a scipy.stats "
                            + "distribution; its draws are numpy's, so only lists are sampled here "
                            + "(x_metrics/split.mojo DEVIATION 6108)")
            if family_is_str(v) or not msel_truth(py_call(b.hasattr, v, PythonObject("__iter__"))):
                raise Error("TypeError: Parameter value for " + family_repr(k) + " is not a list or a distribution ("
                            + family_repr(v) + ")")
    return dists


def msel_sampler_list(dists: PythonObject, n_iter_in: PythonObject, random_state: PythonObject,
                      context: PythonObject) raises -> PythonObject:
    """`list(ParameterSampler)`: a random permutation of the full grid,
    its first n_iter combinations."""
    var grid = msel_grid_list(msel_grid_dicts(dists))
    var n_iter = n_iter_in
    if msel_cmp("lt", PythonObject(len(grid)), n_iter):
        msel_warn("The total space of parameters " + String(len(grid)) + " is smaller than n_iter="
                  + family_str(n_iter) + ". Running " + String(len(grid)) + " iterations. For exhaustive "
                  + "searches, use GridSearchCV.", msel_b().UserWarning)
        n_iter = PythonObject(len(grid))
    var rng = MselRng(random_state, context)
    var perm = rng.permutation(len(grid), context)
    var head = py_call(py_attr(perm, "__getitem__"), py_call(msel_b().slice, PythonObject(None), n_iter))
    return py_call(msel_b().list, py_call(msel_b().map, py_call(msel_b().getattr, grid, PythonObject("__getitem__")), head))


def msel_sampler_len(dists: PythonObject, n_iter: PythonObject) raises -> PythonObject:
    """`len(ParameterSampler)`."""
    return family_call(msel_b().min, n_iter, PythonObject(msel_grid_len(msel_grid_dicts(dists))))


# ---------------------------------------------------------------- score tables


def msel_row_means(rows: List[List[Float64]], context: PythonObject) raises -> List[Float64]:
    """`_row_means`: `fsum(row) / len(row)` of each row, by `row_means_f64`."""
    var m = len(rows)
    var k = len(rows[0]) if m else 0
    var flat = msel_new(max(m * k, 1), "<f8", context)
    var fp = msel_pf64(flat)
    _ = flat  # keep the Python object alive through the native call above
    for i in range(m):  # driver-loop: over candidates
        for j in range(k):  # driver-loop: over folds
            fp[i * k + j] = rows[i][j]
            _ = flat  # keep the Python object alive through the native call above
    var out = msel_new(max(m, 1), "<f8", context)
    if m and k:
        if msel_truth(family_call(msel_entry("row_means_f64", context), PythonObject(msel_addr(flat)), PythonObject(m),
                                  PythonObject(k), PythonObject(msel_addr(out)))):
            raise Error("-inf + inf in fsum")
        _ = flat  # keep the Python object alive through the native calls above
        _ = out  # keep the Python object alive through the native calls above
    var res = List[Float64]()
    var op = msel_pf64(out)
    _ = out  # keep the Python object alive through the native call above
    for i in range(m):  # driver-loop: over candidates
        res.append(op[i])
        _ = out  # keep the Python object alive through the native call above
    return res^


def msel_row_stds(rows: List[List[Float64]], means: List[Float64], context: PythonObject) raises -> List[Float64]:
    """`_row_stds`: each row's population standard deviation about its mean
    (`row_stds_f64`)."""
    var m = len(rows)
    var res = List[Float64]()
    if m == 0:
        return res^
    var k = len(rows[0])
    var flat = msel_new(max(m * k, 1), "<f8", context)
    var fp = msel_pf64(flat)
    _ = flat  # keep the Python object alive through the native call above
    for i in range(m):  # driver-loop: over candidates
        for j in range(k):  # driver-loop: over folds
            fp[i * k + j] = rows[i][j]
            _ = flat  # keep the Python object alive through the native call above
    var mu = msel_f64_list(means, context)
    var out = msel_new(m, "<f8", context)
    _ = family_call(msel_entry("row_stds_f64", context), PythonObject(msel_addr(flat)), PythonObject(m), PythonObject(k),
                    PythonObject(msel_addr(mu)), PythonObject(msel_addr(out)))
    _ = flat  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
    _ = mu  # keep the Python object alive through the native call above
    var op = msel_pf64(out)
    _ = out  # keep the Python object alive through the native call above
    for i in range(m):  # driver-loop: over candidates
        res.append(op[i])
        _ = out  # keep the Python object alive through the native call above
    return res^


def _floats(values: List[Float64]) raises -> PythonObject:
    var out = Python.list()
    for i in range(len(values)):  # driver-loop: over candidates
        _ = py_call(out.append, PythonObject(values[i]))
    return out


def msel_rank(values: PythonObject) raises -> PythonObject:
    """`_rank(values)`: scikit-learn's 'min' ranking, descending (1 best),
    with Python's own set, sort and count over the candidate scores."""
    var b = msel_b()
    var keywords = Python.dict()
    py_setitem(keywords, "reverse", PythonObject(True))
    var order = family_call_kw(b.sorted, keywords, py_call(b.set, values))
    var pos = Python.dict()
    var start = 1
    for i in range(len(order)):  # driver-loop: over distinct candidate scores
        py_setitem(pos, order[i], PythonObject(start))
        start += Int(py=py_call(values.count, order[i]))
    return py_call(b.list, py_call(b.map, py_call(b.getattr, pos, PythonObject("__getitem__")), values))


def _i32_from_list(values: PythonObject, context: PythonObject) raises -> PythonObject:
    """`Array.from_list(values, '<i4')` of a list of Python ints."""
    var store = family_call(Python.import_module("array").array, PythonObject("i"), values)
    return classical_wrap_array(store, Python.tuple(len(store)), "<i4", "C", context)


# ---------------------------------------------------------------- search


def _set_params(est: PythonObject, params: PythonObject) raises:
    """`est.set_params(**params)`, or one setattr per parameter."""
    var b = msel_b()
    if msel_truth(py_call(b.hasattr, est, PythonObject("set_params"))):
        _ = family_call_kw(family_call(b.getattr, est, PythonObject("set_params")), params)
    else:
        var items = py_call(b.list, family_call(params.items))
        for i in range(len(items)):  # driver-loop: over candidate parameters
            _ = family_call(b.setattr, est, items[i][0], items[i][1])


def msel_candidates(model: PythonObject, context: PythonObject) raises -> PythonObject:
    """`list(self._candidates())`: GridSearchCV's grid, RandomizedSearchCV's
    sample, or a subclass's own `_candidates`."""
    var b = msel_b()
    var kind = py_type(model)
    if kind is msel_cls("GridSearchCV", context):
        return msel_grid_list(msel_grid_dicts(family_call(b.getattr, model, PythonObject("param_grid"))))
    if kind is msel_cls("RandomizedSearchCV", context):
        var dists = msel_sampler_check(family_call(b.getattr, model, PythonObject("param_distributions")))
        return msel_sampler_list(dists, family_call(b.getattr, model, PythonObject("n_iter")),
                                 family_call(b.getattr, model, PythonObject("random_state")), context)
    return family_call(b.list, family_call(family_call(b.getattr, model, PythonObject("_candidates"))))


def msel_search_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_BaseSearch.fit(X, y, groups=...)`."""
    var b = msel_b()
    var x_in = args[0]
    var y_in = args[1]
    var groups = args[2]
    var estimator = family_call(b.getattr, model, PythonObject("estimator"))
    var error_score = family_call(b.getattr, model, PythonObject("error_score"))
    msel_require_serial(family_call(b.getattr, model, PythonObject("n_jobs")), error_score,
                        String(py=py_type(model).__name__), context)
    var candidates = msel_candidates(model, context)
    if len(candidates) == 0:
        raise Error("No fits were performed. Was the CV iterator empty? Were there no candidates?")
    var scoring_in = family_call(b.getattr, model, PythonObject("scoring"))
    var refit = family_call(b.getattr, model, PythonObject("refit"))
    var multi = msel_scorers(scoring_in, context)
    var refit_false = refit is PythonObject(False)
    if multi is not None and not refit_false and (
            not family_is_str(refit) or not py_contains(multi, refit)) and not msel_truth(py_call(b.callable, refit)):
        raise Error("For multi-metric scoring, the parameter refit must be set to a scorer key "
                    + "or a callable to refit an estimator with the best parameter setting on the "
                    + "whole data and make the best_* attributes available for that metric.")
    var scoring = multi if multi is not None else scoring_in
    var return_train = msel_truth(family_call(b.getattr, model, PythonObject("return_train_score")))
    var got = msel_cv_folds(estimator, x_in, y_in, family_call(b.getattr, model, PythonObject("cv")), groups, True, context)
    var xf = got[0]
    var yf = got[1]
    var folds = got[2]
    var per = List[List[List[Float64]]]()
    var trains = List[List[List[Float64]]]()
    var names = List[String]()
    var n_splits = 0
    var rows = MselFoldRows(xf, yf, folds, msel_device_rows_ok(estimator, scoring, context), context)
    try:
        for c in range(len(candidates)):  # driver-loop: over candidates
            var est = msel_clone(estimator, False, context)
            _set_params(est, candidates[c])
            var cvr = msel_cross_validate_run(est, yf, rows, scoring, return_train, False, False, error_score, context)
            if c == 0:
                names = cvr.names.copy()
                for _ in range(len(names)):  # driver-loop: over scorer names
                    per.append(List[List[Float64]]())
                    trains.append(List[List[Float64]]())
            for k in range(len(names)):  # driver-loop: over scorer names
                per[k].append(cvr.test[k].copy())
                if return_train:
                    trains[k].append(cvr.train[k].copy())
            n_splits = len(cvr.fit_time)
    finally:
        rows.close(context)
    var results = Python.dict()
    py_setitem(results, "params", candidates)
    var itertools = Python.import_module("itertools")
    var keys = family_call(b.sorted, py_call(b.set, py_call(itertools.chain.from_iterable, candidates)))
    for i in range(len(keys)):  # driver-loop: over candidate parameter names
        var getter = py_call(Python.import_module("operator").methodcaller, PythonObject("get"), keys[i])
        py_setitem(results, family_call(msel_op("add"), PythonObject("param_"), family_call(b.str, keys[i])),
                   py_call(b.list, py_call(b.map, getter, candidates)))
    var best_means = Python.list()
    var best_ranks = Python.list()
    var key_index = -1
    if multi is None:
        key_index = 0
    elif family_is_str(refit):
        for k in range(len(names)):  # driver-loop: over scorer names
            if names[k] == String(py=refit):
                key_index = k
    for k in range(len(names)):  # driver-loop: over scorer names
        var suffix = String("") if multi is None else "_" + names[k]
        for i in range(n_splits):  # driver-loop: over folds
            var column = List[Float64]()
            for c in range(len(candidates)):  # driver-loop: over candidates
                column.append(per[k][c][i])
            py_setitem(results, PythonObject("split" + String(i) + "_test_score" + suffix), msel_f64_list(column, context))
        var means = msel_row_means(per[k], context)
        var stds = msel_row_stds(per[k], means, context)
        var ranks = msel_rank(_floats(means))
        py_setitem(results, PythonObject("mean_test_score" + suffix), msel_f64_list(means, context))
        py_setitem(results, PythonObject("std_test_score" + suffix), msel_f64_list(stds, context))
        py_setitem(results, PythonObject("rank_test_score" + suffix), _i32_from_list(ranks, context))
        if return_train:
            py_setitem(results, PythonObject("mean_train_score" + suffix), msel_f64_list(msel_row_means(trains[k], context), context))
        if k == key_index:
            best_means = _floats(means)
            best_ranks = ranks
    py_setattr(model, "cv_results_", results)
    py_setattr(model, "n_splits_", PythonObject(n_splits))
    py_setattr(model, "multimetric_", PythonObject(multi is not None))
    var best_params = PythonObject(None)
    if key_index >= 0:
        var best_index = family_call(best_ranks.index, PythonObject(1))
        py_setattr(model, "best_index_", best_index)
        py_setattr(model, "best_score_", best_means[best_index])
        best_params = candidates[best_index]
        py_setattr(model, "best_params_", best_params)
    elif msel_truth(py_call(b.callable, refit)):
        var best_index = family_call(b.int, family_call(refit, results))
        py_setattr(model, "best_index_", best_index)
        best_params = family_call(py_attr(candidates, "__getitem__"), best_index)
        py_setattr(model, "best_params_", best_params)
    if not refit_false:
        var best = msel_clone(estimator, False, context)
        if msel_truth(py_call(b.hasattr, best, PythonObject("set_params"))):
            _ = family_call_kw(family_call(b.getattr, best, PythonObject("set_params")),
                               family_call(b.getattr, model, PythonObject("best_params_")))
        var xa = msel_materialize(x_in, "X", context)
        msel_fit(best, xa, y_in)
        py_setattr(model, "best_estimator_", best)
        py_setattr(model, "scorer_", msel_get_scorer(scoring_in, context) if multi is None else multi)
    return model


# ---------------------------------------------------------------- validation_curve


def msel_validation_curve(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`validation_curve(estimator, X, y, param_name, param_range, groups,
    cv, scoring, n_jobs, error_score)`."""
    var b = msel_b()
    var estimator = args[0]
    var scoring = args[7]
    var error_score = args[9]
    msel_require_serial(args[8], error_score, "validation_curve", context)
    var got = msel_cv_folds(estimator, args[1], args[2], args[6], args[5], True, context)
    var xf = got[0]
    var yf = got[1]
    var folds = got[2]
    var tr = List[Float64]()
    var te = List[Float64]()
    var count = 0
    var k = 0
    var values = family_call(b.list, args[4])
    var rows = MselFoldRows(xf, yf, folds, msel_device_rows_ok(estimator, scoring, context), context)
    try:
        for v in range(len(values)):  # driver-loop: over parameter values
            var est = msel_clone(estimator, False, context)
            var params = Python.dict()
            py_setitem(params, args[3], values[v])
            _ = family_call_kw(family_call(b.getattr, est, PythonObject("set_params")), params)
            var r = msel_cross_validate_run(est, yf, rows, scoring, True, False, False, error_score, context)
            if len(r.names) != 1 or r.names[0] != "score":
                _ = family_call(py_attr(Python.dict(), "__getitem__"), PythonObject("train_score"))
            for i in range(len(r.train[0])):  # driver-loop: over folds
                tr.append(r.train[0][i])
                te.append(r.test[0][i])
            if v == 0:
                k = len(r.train[0])
            count += 1
    finally:
        rows.close(context)
    if count == 0:
        raise Error("IndexError: list index out of range")
    return Python.tuple(msel_f64_block(tr, count, k, context), msel_f64_block(te, count, k, context))


# ---------------------------------------------------------------- learning_curve


def msel_learning_curve(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`learning_curve(estimator, X, y, groups, train_sizes, cv, scoring,
    n_jobs, shuffle, random_state, error_score, return_times)`."""
    var b = msel_b()
    var estimator = args[0]
    var got = msel_cv_folds(estimator, args[1], args[2], args[5], args[3], False, context)
    var x = got[0]
    var y = got[1]
    var folds = got[2]
    var nf = len(folds)
    var n_max = Int(py=folds[0][0].size)
    for f in range(nf):  # driver-loop: over folds
        n_max = min(n_max, Int(py=folds[f][0].size))
    var chosen = Python.list()
    var requested = family_call(b.list, args[4])
    var math = Python.import_module("math")
    for i in range(len(requested)):  # driver-loop: over the requested train sizes
        var s = requested[i]
        var a = 0
        if family_isinstance(s, b.float):
            a = msel_int(family_call(math.floor, family_call(msel_op("mul"), s, PythonObject(n_max))))
        else:
            a = msel_int(s)
        if not (0 < a and a <= n_max):
            raise Error("train_sizes has been interpreted as absolute numbers of training samples and "
                        + "must be within (0, " + String(n_max) + "], but is within [" + String(a) + ", " + String(a) + "].")
        _ = py_call(chosen.append, PythonObject(a))
    var sizes_py = family_call(b.sorted, py_call(b.set, chosen))
    var sizes = List[Int]()
    for i in range(len(sizes_py)):  # driver-loop: over train sizes
        sizes.append(Int(py=sizes_py[i]))
    var shuffle = msel_truth(args[8])
    var orders = Python.list()
    if shuffle:
        var rng = MselRng(args[9], context)
        var lengths = List[Int]()
        for f in range(nf):  # driver-loop: over folds
            lengths.append(Int(py=folds[f][0].size))
        var perms = rng.permutation_rows(lengths, context)
        for f in range(nf):  # driver-loop: over folds (one device gather each)
            var tr = msel_index_i64(folds[f][0], context)
            var size = lengths[f]
            if size > 0:
                var out = msel_new(size, "<i8", context)
                _ = family_call(msel_entry("gather_i64", context), PythonObject(msel_addr(tr)), PythonObject(size),
                                PythonObject(msel_addr(perms[f])), PythonObject(size), PythonObject(msel_addr(out)))
                _ = tr  # keep the Python object alive through the native call above
                _ = out  # keep the Python object alive through the native call above
                tr = out
            _ = py_call(orders.append, tr)
    else:
        for f in range(nf):  # driver-loop: over folds
            _ = py_call(orders.append, msel_index_i64(folds[f][0], context))
    var tr_s = List[Float64]()
    var te_s = List[Float64]()
    var ft = List[Float64]()
    var st = List[Float64]()
    var res = msel_resident_new(context)
    var scoring = args[6]
    try:
        for si in range(len(sizes)):  # driver-loop: over train sizes
            var a = sizes[si]
            for f in range(nf):  # driver-loop: over folds
                var test = folds[f][1]
                var sub = msel_slice_i64(orders[f], 0, a, context)
                var est = msel_clone(estimator, False, context)
                var t0 = msel_seconds()
                var xs = _take(res, x, sub, context)
                var ys = _take(res, y, sub, context)
                _ = family_call(family_call(b.getattr, est, PythonObject("fit")), xs, ys)
                var t1 = msel_seconds()
                var sc = msel_get_scorer(scoring, context)
                tr_s.append(Float64(py=msel_score(est, xs, ys, sc, context)))
                te_s.append(Float64(py=msel_score(est, _take(res, x, test, context), _take(res, y, test, context), sc, context)))
                msel_resident_release(res, sub, context)
                ft.append(t1 - t0)
                st.append(msel_seconds() - t1)
    finally:
        msel_resident_close(res, context)
    var out = Python.list()
    _ = py_call(out.append, msel_i64_list(sizes, context))
    _ = py_call(out.append, msel_f64_block(tr_s, len(sizes), nf, context))
    _ = py_call(out.append, msel_f64_block(te_s, len(sizes), nf, context))
    if msel_truth(args[11]):
        _ = py_call(out.append, msel_f64_block(ft, len(sizes), nf, context))
        _ = py_call(out.append, msel_f64_block(st, len(sizes), nf, context))
    return py_call(b.tuple, out)


def _take(res: PythonObject, values: PythonObject, idx: PythonObject, context: PythonObject) raises -> PythonObject:
    """learning_curve's `take`: a list by index, else the resident gather."""
    var b = msel_b()
    if family_isinstance(values, b.list):
        return py_call(b.list, py_call(b.map, py_call(b.getattr, values, PythonObject("__getitem__")), idx._mv))
    return msel_resident_take(res, values, idx, context)


# ---------------------------------------------------------------- permutation_test_score


def _y_free(splitter: PythonObject, context: PythonObject) raises -> Bool:
    """`_y_free(splitter)`: its split cannot read y and draws the same folds
    on every call."""
    var b = msel_b()
    var t = py_type(splitter)
    var fixed: List[String] = ["LeaveOneGroupOut", "LeavePGroupsOut", "TimeSeriesSplit", "LeaveOneOut", "LeavePOut",
                                "PredefinedSplit", "_IterableCV"]
    for i in range(len(fixed)):  # driver-loop: over the y-free splitter classes
        if t is msel_cls(fixed[i], context):
            return True
    var seeded = msel_is_integral(py_call(b.getattr, splitter, PythonObject("random_state"), PythonObject(None)), context)
    if t is msel_cls("KFold", context) or t is msel_cls("GroupKFold", context):
        return not msel_truth(family_call(b.getattr, splitter, PythonObject("shuffle"))) or seeded
    if t is msel_cls("ShuffleSplit", context) or t is msel_cls("GroupShuffleSplit", context):
        return seeded
    return False


def _perm_base(y: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_NativePermutation._base(y)`: y as the definition's first
    `mean_score` array (int32 or float32), or None."""
    var b = msel_b()
    var array_mod = Python.import_module("array")
    if family_isinstance(y, Python.tuple(b.list, b.tuple)):
        var kinds = py_call(b.set, py_call(b.map, b.type, y))
        var code = String("")
        var dtype = String("")
        var ints = py_call(b.set, Python.tuple(b.int, b.bool))
        if len(kinds) > 0 and msel_truth(py_call(kinds.issubset, ints)):
            code = "i"
            dtype = "<i4"
        elif len(kinds) == 1 and py_contains(kinds, b.float):
            code = "f"
            dtype = "<f4"
        else:
            return PythonObject(None)
        var store = msel_try(array_mod.array, Python.tuple(PythonObject(code), y), b.OverflowError, context)
        if msel_missed(store, context):
            return PythonObject(None)
        return classical_wrap_array(store, Python.tuple(len(store)), dtype, "C", context)
    var a = msel_try_materialize(y, "y", Python.tuple(b.TypeError, b.ValueError), context)
    if msel_missed(a, context):
        return PythonObject(None)
    if Int(py=a.ndim) != 1 or Int(py=a.size) < 1:
        return PythonObject(None)
    var d = String(py=a.dtype)
    if d == "<f4" or d == "<f8":
        return family_as_f32(a, 1, "y", context)
    if d == "<i4":
        return a
    if d == "<i8" or d == "<u1" or d == "<u4":
        var narrow = msel_new(Int(py=a.size), "<i4", context)
        if msel_int(family_call(msel_entry("cast_elements", context), PythonObject(msel_addr(a)),
                                context["family_msel_codes"][a.dtype], PythonObject(msel_addr(narrow)),
                                PythonObject(2), a.size)) != 0:
            return PythonObject(None)
        _ = a  # keep the Python object alive through the native calls above
        _ = narrow  # keep the Python object alive through the native calls above
        return narrow
    if d == "<i2" or d == "<u2" or d == "<i1":
        var store = family_call(array_mod.array, PythonObject("i"), a._mv)
        return classical_wrap_array(store, Python.tuple(len(store)), "<i4", "C", context)
    return PythonObject(None)


def _mean_test(estimator: PythonObject, x: PythonObject, y: PythonObject, folds: PythonObject, scoring: PythonObject,
               mut rows: MselFoldRows, context: PythonObject) raises -> Float64:
    """`_row_means([cross_validate(...)['test_score']])[0]` on `rows`."""
    var r = msel_cross_validate_run(estimator, y, rows, scoring, False, False, False, PythonObject(_nan()),
                                    context)
    var one = List[List[Float64]]()
    one.append(r.test[0].copy())
    return msel_row_means(one, context)[0]


def _nan() -> Float64:
    return Float64(0.0) / Float64(0.0)


def _mean_own(estimator: PythonObject, x: PythonObject, y: PythonObject, folds: PythonObject, scoring: PythonObject,
              device: Bool, context: PythonObject) raises -> Float64:
    """`_mean_test` on rows of its own (the folds drawn for this y)."""
    var rows = MselFoldRows(x, y, folds, device, context)
    var value: Float64 = 0.0
    try:
        value = _mean_test(estimator, x, y, folds, scoring, rows, context)
    finally:
        rows.close(context)
    return value


def _group_order(groups: PythonObject, n: Int, context: PythonObject) raises -> PythonObject:
    """`_NativePermutation._group_order`: (sizes, group-sorted rows,
    inverse) of the groups in sorted-group order, in one program."""
    var gc = MselGroups(groups, n, "groups", context)
    var m = gc.m
    var prog = _Prog(context["family_metrics_ops"])
    var key = prog.put_i32(gc.codes)
    var off_slot = prog.alloc(m + 1)
    var off = prog.want(off_slot, m + 1)
    var ordr = prog.alloc(n)
    prog.stage("group_sort", 1, [key, n, m, off, ordr])
    var off2 = prog.alloc(n + 1)
    var inv = prog.alloc(n)
    prog.stage("group_sort", 1, [ordr, n, n, off2, inv])
    var w_ord_slot = prog.alloc(2 * n)
    var w_ord = prog.want(w_ord_slot, 2 * n)
    var w_inv_slot = prog.alloc(2 * n)
    var w_inv = prog.want(w_inv_slot, 2 * n)
    prog.stage("rows64", n, [ordr, w_ord])
    prog.stage("rows64", n, [inv, w_inv])
    var arena = _reg_run(prog, PythonObject(None), context)
    var base = _arena_base(arena)
    var sizes = Python.list()
    for g in range(m):  # driver-loop: over groups
        _ = py_call(sizes.append, PythonObject(_arena_int(arena, off + g + 1) - _arena_int(arena, off + g)))
    var rows_sorted = msel_new(n, "<i8", context)
    var inverse = msel_new(n, "<i8", context)
    memcpy(dest=msel_p64(rows_sorted), src=FAMILY_I64P(unsafe_from_address=base + 4 * w_ord), count=n)
    _ = rows_sorted  # keep the Python object alive through the native call above
    memcpy(dest=msel_p64(inverse), src=FAMILY_I64P(unsafe_from_address=base + 4 * w_inv), count=n)
    _ = inverse  # keep the Python object alive through the native call above
    _ = prog^
    return Python.tuple(sizes, rows_sorted, inverse)


def _group_index(mut rng: MselRng, order: PythonObject, n: Int, context: PythonObject) raises -> PythonObject:
    """`_NativePermutation._group_index`: each group's rows permuted by its
    draw over the group-sorted rows, put back in row order."""
    var sizes_py = order[0]
    var sizes = List[Int]()
    for g in range(len(sizes_py)):  # driver-loop: over groups
        sizes.append(Int(py=sizes_py[g]))
    var gather = msel_entry("gather_i64", context)
    var src = msel_new(n, "<i8", context)
    var s0 = msel_addr(src)
    _ = src  # keep the Python object alive through the native call above
    var o0 = msel_addr(order[1])
    var perms = rng.permutation_rows(sizes, context)
    var at = 0
    for g in range(len(sizes)):  # driver-loop: over groups (one device gather each)
        _ = family_call(gather, PythonObject(o0 + 8 * at), PythonObject(sizes[g]), PythonObject(msel_addr(perms[g])),
                        PythonObject(sizes[g]), PythonObject(s0 + 8 * at))
        at += sizes[g]
    var idx = msel_new(n, "<i8", context)
    _ = family_call(gather, PythonObject(s0), PythonObject(n), PythonObject(msel_addr(order[2])), PythonObject(n),
                    PythonObject(msel_addr(idx)))
    _ = idx  # keep the Python object alive through the native call above
    return idx


def _pvalue(scores: List[Float64], score: Float64, n_permutations: Int) -> Float64:
    var hits = 0
    for i in range(len(scores)):  # driver-loop: over permutations
        if scores[i] >= score:
            hits += 1
    return (Float64(hits) + 1.0) / Float64(n_permutations + 1)


def msel_permutation_test_score(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`permutation_test_score(estimator, X, y, groups, cv, n_permutations,
    n_jobs, random_state, verbose, scoring, fit_params, params)`."""
    var b = msel_b()
    var estimator = args[0]
    var y = args[2]
    var groups = args[3]
    var cv = args[4]
    var n_permutations = msel_int(args[5])
    var random_state = args[7]
    var base = PythonObject(None)
    if not msel_sabotage():
        _ = msel_entry("gather_i64", context)
        base = _perm_base(y, context)
        if base is not None and Int(py=base.size) < 2:
            base = PythonObject(None)
    if base is None:
        return _permutation_definition(args, context)
    var x = msel_materialize(args[1], "X", context)
    var sc = msel_get_scorer(args[9], context)
    var n = Int(py=base.size)
    var reuse = not (cv is None or msel_is_integral(cv, context)) and _y_free(cv, context)
    var folds0 = msel_cv_folds(estimator, x, base, cv, groups, False, context)[2]
    var device = msel_device_rows_ok(estimator, sc, context)
    var rows0 = MselFoldRows(x, PythonObject(None), folds0, device, context)
    var score: Float64 = 0.0
    var perm_scores = List[Float64]()
    try:
        score = _mean_test(estimator, x, base, folds0, sc, rows0, context)
        var rng = MselRng(random_state, context)
        var order = PythonObject(None) if groups is None else _group_order(groups, n, context)
        for _ in range(n_permutations):  # driver-loop: over permutations
            var idx = PythonObject(None)
            if order is None:
                idx = rng.permutation_rows([n], context)[0]
            else:
                idx = _group_index(rng, order, n, context)
            # the permuted y: one device gather out of the resident base y
            var yp = msel_resident_take(rows0.res, base, idx, context)
            msel_resident_release(rows0.res, idx, context)
            if reuse:
                perm_scores.append(_mean_test(estimator, x, yp, folds0, sc, rows0, context))
            else:
                var folds = msel_cv_folds(estimator, x, yp, cv, groups, False, context)[2]
                perm_scores.append(_mean_own(estimator, x, yp, folds, sc, device, context))
    finally:
        rows0.close(context)
    return Python.tuple(PythonObject(score), msel_f64_list(perm_scores, context),
                        PythonObject(_pvalue(perm_scores, score, n_permutations)))


def _permutation_definition(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`permutation_test_score`'s definition route (label objects the
    native permutation cannot hold): y rebuilt from a list per permutation,
    the permutations by Python's own list operations."""
    var b = msel_b()
    var itertools = Python.import_module("itertools")
    var operator = Python.import_module("operator")
    var estimator = args[0]
    var groups = args[3]
    var cv = args[4]
    var n_permutations = msel_int(args[5])
    var xa = msel_materialize(args[1], "X", context)
    var yl = msel_flatten(args[2], context)
    var sc = msel_get_scorer(args[9], context)
    var score = _definition_mean(estimator, xa, yl, groups, cv, sc, context)
    var rng = MselRng(args[7], context)
    var perm_scores = List[Float64]()
    for _ in range(n_permutations):  # driver-loop: over permutations
        var yp = PythonObject(None)
        if groups is None:
            var perm = rng.permutation(len(yl), context)
            yp = py_call(b.list, py_call(b.map, py_call(b.getattr, yl, PythonObject("__getitem__")), perm))
        else:
            var g = msel_flatten(groups, context)
            yp = py_call(b.list, yl)
            var values = family_call(b.sorted, family_call(b.set, g))
            for gi in range(len(values)):  # driver-loop: over groups
                var rows = py_call(b.list, py_call(itertools.compress, py_call(b.range, PythonObject(len(g))),
                                                   py_call(b.map, operator.eq, g, py_call(itertools.repeat, values[gi]))))
                var perm = rng.permutation(len(rows), context)
                var src = py_call(b.map, py_call(b.getattr, yl, PythonObject("__getitem__")),
                                  py_call(b.map, py_call(b.getattr, rows, PythonObject("__getitem__")), perm))
                _ = py_call(b.list, py_call(b.map, py_call(b.getattr, yp, PythonObject("__setitem__")), rows, src))
        perm_scores.append(_definition_mean(estimator, xa, yp, groups, cv, sc, context))
    return Python.tuple(PythonObject(score), msel_f64_list(perm_scores, context),
                        PythonObject(_pvalue(perm_scores, score, n_permutations)))


def _definition_mean(estimator: PythonObject, xa: PythonObject, yv: PythonObject, groups: PythonObject, cv: PythonObject,
                     sc: PythonObject, context: PythonObject) raises -> Float64:
    """The definition's `mean_score(yv)`: y as a float32 / int32 Array by
    `Array.from_list`, then one cross_validate."""
    var b = msel_b()
    var dtype = PythonObject("<f4") if family_isinstance(yv[0], b.float) else PythonObject("<i4")
    var from_list = family_call(b.getattr, msel_cls("Array", context), PythonObject("from_list"))
    var yarr = msel_materialize(family_call(from_list, yv, dtype), "y", context)
    var folds = msel_cv_folds(estimator, xa, yarr, cv, groups, False, context)[2]
    return _mean_own(estimator, xa, yarr, folds, sc, msel_device_rows_ok(estimator, sc, context), context)
