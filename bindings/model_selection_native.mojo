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
from bindings.native_buffer_owner import native_storage_allocate
from bindings.family_common import family_repr, family_is_str, family_isinstance, family_call


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


def msel_get_scorer(scoring: PythonObject, context: PythonObject) raises -> PythonObject:
    """`get_scorer(scoring)`."""
    var b = Python.import_module("builtins")
    if scoring is None or Bool(py=py_call(b.callable, scoring)):
        return scoring
    if family_is_str(scoring):
        var table = context["family_msel_scorers"]
        if not py_contains(table, scoring):
            raise Error(family_repr(scoring) + " is not a valid scoring value. Use "
                        + "sklearn.metrics.get_scorer_names() to get valid options.")
        return table[scoring]
    raise Error("scoring must be a str, a callable or None, got " + family_repr(scoring))


def msel_score(estimator: PythonObject, x: PythonObject, y: PythonObject, scorer: PythonObject,
               context: PythonObject) raises -> PythonObject:
    """`model_selection._score(estimator, X, y, scorer)`: the estimator's own
    `score` without a scorer; a real scalar, as a float."""
    var b = Python.import_module("builtins")
    var value: PythonObject
    if scorer is None:
        value = family_call(_msel_attr(estimator, "score"), x, y)
    else:
        value = family_call(scorer, estimator, x, y)
    if not family_isinstance(value, context["family_msel_real"]):
        raise Error("TypeError: scoring must return a real scalar")
    return family_call(b.float, value)


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
    raise Error("RuntimeError: unknown model_selection operation " + operation)
