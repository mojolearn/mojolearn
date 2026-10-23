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
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setattr
from bindings.py_attr import py_attr
from bindings.native_buffer_owner import native_storage_allocate
from bindings.family_common import family_repr, family_is_str, family_isinstance, family_call
from bindings.msel_common import msel_get_scorer, msel_score
from bindings.msel_cv import msel_cross_validate, msel_cross_val_score, msel_cross_val_predict, msel_split_descriptor
from bindings.msel_search import (
    msel_grid_list, msel_grid_len, msel_sampler_list, msel_sampler_len, msel_search_fit, msel_validation_curve,
    msel_learning_curve, msel_permutation_test_score,
)
from bindings.msel_split import (
    msel_split, msel_n_splits, msel_loo_at, msel_lpo_at, msel_logo_at, msel_lpgo_at, msel_check_cv,
    msel_train_test_split, msel_truth,
)


def _msel_attr(model: PythonObject, name: String) raises -> PythonObject:
    """`model.<name>` with its own AttributeError."""
    return family_call(Python.import_module("builtins").getattr, model, PythonObject(name))


def _msel_best(model: PythonObject) raises -> PythonObject:
    """`BaseSearchCV._best`: the refit estimator, refused by name."""
    var b = Python.import_module("builtins")
    if not Bool(py=family_call(b.hasattr, model, PythonObject("best_estimator_"))):
        raise Error("AttributeError: This " + String(py_type(model).__name__)
                    + " instance was initialized with refit=False or is not fitted.")
    return _msel_attr(model, "best_estimator_")


def msel_operation(model: PythonObject, operation: String, args: PythonObject,
                   context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    if operation == "family_msel_get_scorer":
        return msel_get_scorer(args[0], context)
    if operation == "family_msel_scorer_names":
        return py_call(b.sorted, context["family_msel_scorers"])
    if operation == "family_msel_make_scorer":
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
        var best = _msel_best(model)
        return family_call(_msel_attr(best, String(args[0])), args[1])
    if operation == "family_msel_search_score":
        # The scorer is resolved first, as `score` resolved it before
        # reaching the best estimator.
        var scorer: PythonObject
        if not Bool(py=py_call(b.bool, _msel_attr(model, "multimetric_"))):
            scorer = msel_get_scorer(_msel_attr(model, "scoring"), context)
        else:
            scorer = family_call(py_call(b.getattr, _msel_attr(model, "scorer_"), PythonObject("__getitem__")),
                                 _msel_attr(model, "refit"))
        var best = _msel_best(model)
        return msel_score(best, args[0], args[1], scorer, context)
    if operation == "family_msel_split":
        return msel_split(model, args[0], args[1], args[2], context)
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
    if operation == "family_msel_grid":
        return msel_grid_list(py_call(b.getattr, model, PythonObject("param_grid")))
    if operation == "family_msel_grid_len":
        return PythonObject(msel_grid_len(py_call(b.getattr, model, PythonObject("param_grid"))))
    if operation == "family_msel_grid_item":
        return family_call(py_attr(msel_grid_list(py_call(b.getattr, model, PythonObject("param_grid"))), "__getitem__"), args[0])
    if operation == "family_msel_sampler":
        return msel_sampler_list(_msel_attr(model, "param_distributions"), _msel_attr(model, "n_iter"),
                                 _msel_attr(model, "random_state"), context)
    if operation == "family_msel_sampler_len":
        return msel_sampler_len(_msel_attr(model, "param_distributions"), _msel_attr(model, "n_iter"))
    if operation == "family_msel_search_fit":
        return msel_search_fit(model, args, context)
    if operation == "family_msel_validation_curve":
        return msel_validation_curve(args, context)
    if operation == "family_msel_learning_curve":
        return msel_learning_curve(args, context)
    if operation == "family_msel_permutation_test_score":
        return msel_permutation_test_score(args, context)
    raise Error("RuntimeError: unknown model_selection operation " + operation)
