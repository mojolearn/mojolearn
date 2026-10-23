# SPDX-License-Identifier: Apache-2.0
# model_selection's public scorer functions and the search estimators'
# delegations, one native call each (python/mojolearn/model_selection.py:
# `make_scorer`, `get_scorer`, `get_scorer_names`, `GridSearchCV` /
# `RandomizedSearchCV` `predict`, `predict_proba`, `decision_function`,
# `transform` and `score`). The scorer table is fixed at import; the
# refusals keep the Python functions' wording and order. A delegated method
# is the caller's fitted estimator's own (a mojolearn estimator's is itself
# one native call); its exception reaches the caller unchanged.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setattr, py_setitem
from bindings.py_attr import py_attr
from bindings.native_buffer_owner import native_storage_allocate
from bindings.family_common import family_repr, family_is_str, family_isinstance, family_call
from bindings.msel_common import msel_get_scorer, msel_score, msel_admit
from bindings.msel_scorer_native import msel_metric_plan, msel_scorer_call
from bindings.msel_estimator_native import msel_estimator_call, msel_estimator_plan
from bindings.parallel_cv_native import msel_parallel_operation
from bindings.msel_cv import msel_cross_validate, msel_cross_val_score, msel_cross_val_predict, msel_split_descriptor
from bindings.msel_search import (
    msel_grid_dicts, msel_grid_list, msel_grid_len, msel_sampler_check, msel_sampler_list, msel_sampler_len, msel_search_fit, msel_validation_curve,
    msel_learning_curve, msel_permutation_test_score,
)
from bindings.msel_search_state import msel_search_fields, msel_search_plan, msel_search_initialize
from bindings.msel_split import (
    msel_split, msel_n_splits, msel_loo_at, msel_lpo_at, msel_logo_at, msel_lpgo_at, msel_check_cv,
    msel_train_test_split, msel_truth,
)


def _msel_attr(model: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    var fields = msel_search_fields(model, context)
    if not py_contains(fields, PythonObject(name)):
        raise Error("AttributeError: native search field is unavailable: " + name)
    return fields[PythonObject(name)]


def _msel_best(model: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = msel_search_fields(model, context)
    if not py_contains(fields, "best_estimator_"):
        raise Error("AttributeError: This " + String(py_type(model).__name__)
                    + " instance was initialized with refit=False or is not fitted.")
    return fields["best_estimator_"]


def msel_operation(model: PythonObject, operation: String, args: PythonObject,
                   context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    if operation.startswith("family_msel_parallel_"):
        return msel_parallel_operation(operation, args, context)
    if operation == "family_msel_search_init":
        return msel_search_initialize(model, args, context)
    if operation == "family_msel_grid_init":
        var grid = msel_grid_dicts(args[0], context)
        return msel_search_initialize(model, Python.tuple(grid), context)
    if operation == "family_msel_sampler_init":
        _ = msel_sampler_check(args[0], context)
        return msel_search_initialize(model, args, context)
    if operation == "family_msel_search_get_params":
        var plan = msel_search_plan(model, context)
        var fields = msel_search_fields(model, context)
        var result = Python.dict()
        var names = plan["parameters"]
        for k in range(len(names)):  # driver-loop: fixed search constructor schema
            py_setitem(result, names[k], fields[names[k]])
        return result
    if operation == "family_msel_search_set_params":
        var fields = msel_search_fields(model, context)
        if py_type(args[0]) is not b.dict:
            raise Error("TypeError: search parameter keywords must be a native dict")
        var names = py_call(b.list, args[0].keys())
        for k in range(len(names)):  # driver-loop: existing search setter accepts plain stored parameter names
            if py_type(names[k]) is not b.str:
                raise Error("TypeError: search parameter names must be exact strings")
            if py_contains(msel_search_plan(model, context)["protected"], names[k]):
                raise Error("TypeError: search parameters cannot override native method or storage hooks")
        _ = py_call(fields.update, args[0])
        return model
    if operation == "family_msel_search_estimator_type":
        var plan = msel_estimator_plan(_msel_attr(model, "estimator", context), context)
        return plan["estimator_type"]
    if operation == "family_msel_get_scorer":
        return msel_get_scorer(args[0], context)
    if operation == "family_msel_scorer_names":
        return py_call(b.sorted, context["family_msel_scorers"])
    if operation == "family_msel_scorer_call":
        return msel_scorer_call(model, args[0], args[1], args[2], args[3], args[4], context)
    if operation == "family_msel_make_scorer":
        _ = msel_metric_plan(args[0], context)
        msel_admit(args[1], context)
        msel_admit(args[2], context)
        msel_admit(args[3], context)
        # `_Scorer(score_func, sign, kwargs, response_method, None)`, its
        # slots filled directly.
        var scorer = native_storage_allocate(context["family_msel_scorer_class"])
        py_setattr(scorer, "_score_func", args[0])
        py_setattr(scorer, "_sign", PythonObject(1 if Bool(py=py_call(b.bool, args[2])) else -1))
        py_setattr(scorer, "_kwargs", args[3])
        py_setattr(scorer, "_response_method", args[1])
        py_setattr(scorer, "_name", PythonObject(None))
        return scorer
    if operation == "family_msel_delegate":
        var best = _msel_best(model, context)
        if py_type(args[0]) is not b.str:
            raise Error("TypeError: native delegated method must be an exact string")
        return msel_estimator_call(best, String(args[0]), Python.tuple(args[1]), context)
    if operation == "family_msel_search_score":
        # The scorer is resolved first, as `score` resolved it before
        # reaching the best estimator.
        var scorer: PythonObject
        var multimetric = _msel_attr(model, "multimetric_", context)
        msel_admit(multimetric, context)
        if not Bool(py=py_call(b.bool, multimetric)):
            scorer = msel_get_scorer(_msel_attr(model, "scoring", context), context)
        else:
            var scorers = _msel_attr(model, "scorer_", context)
            var refit = _msel_attr(model, "refit", context)
            if py_type(scorers) is not b.dict or py_type(refit) is not b.str:
                raise Error("TypeError: fitted native search scorer state requires a dict and exact string key")
            scorer = scorers[refit]
        var best = _msel_best(model, context)
        return msel_score(best, args[0], args[1], scorer, context)
    if operation == "family_msel_split":
        # Preserve the public iterator contract without a Python generator
        # wrapper resuming around every native fold. Lazy native splitters
        # already return an iterator; iter() leaves those unchanged.
        return py_call(b.iter, msel_split(model, args[0], args[1], args[2], context))
    if operation == "family_msel_n_splits":
        return msel_n_splits(model, args[0], args[1], args[2], context)
    if operation == "family_msel_split_loo_at":
        return msel_loo_at(args, context)
    if operation == "family_msel_split_lpo_at":
        return msel_lpo_at(args, context)
    if operation == "family_msel_split_logo_at":
        return msel_logo_at(args, context)
    if operation == "family_msel_split_lpgo_at":
        return msel_lpgo_at(args, context)
    if operation == "family_msel_check_cv":
        return msel_check_cv(args[0], args[1], msel_truth(args[2]), args[3], args[4], context)
    if operation == "family_msel_train_test_split":
        return msel_train_test_split(args[0], args[1], args[2], args[3], args[4], args[5], context)
    if operation == "family_msel_cross_validate":
        return msel_cross_validate(args, context)
    if operation == "family_msel_cross_val_score":
        return msel_cross_val_score(args, context)
    if operation == "family_msel_cross_val_predict":
        return msel_cross_val_predict(args, context)
    if operation == "family_msel_split_descriptor":
        return msel_split_descriptor(args, context)
    if operation == "family_msel_grid" or operation == "family_msel_grid_iter":
        var result = msel_grid_list(msel_grid_dicts(_msel_attr(model, "param_grid", context), context), context)
        return py_call(b.iter, result) if operation.endswith("_iter") else result
    if operation == "family_msel_grid_len":
        return PythonObject(msel_grid_len(msel_grid_dicts(_msel_attr(model, "param_grid", context), context), context))
    if operation == "family_msel_grid_item":
        msel_admit(args[0], context)
        var grid = msel_grid_list(msel_grid_dicts(_msel_attr(model, "param_grid", context), context), context)
        return family_call(py_attr(grid, "__getitem__"), args[0])
    if operation == "family_msel_sampler" or operation == "family_msel_sampler_iter":
        var result = msel_sampler_list(_msel_attr(model, "param_distributions", context), _msel_attr(model, "n_iter", context),
                                       _msel_attr(model, "random_state", context), context)
        return py_call(b.iter, result) if operation.endswith("_iter") else result
    if operation == "family_msel_sampler_len":
        return msel_sampler_len(_msel_attr(model, "param_distributions", context), _msel_attr(model, "n_iter", context), context)
    if operation == "family_msel_search_fit":
        return msel_search_fit(model, args, context)
    if operation == "family_msel_validation_curve":
        return msel_validation_curve(args, context)
    if operation == "family_msel_learning_curve":
        return msel_learning_curve(args, context)
    if operation == "family_msel_permutation_test_score":
        return msel_permutation_test_score(args, context)
    raise Error("RuntimeError: unknown model_selection operation " + operation)
