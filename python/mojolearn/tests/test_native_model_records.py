# SPDX-License-Identifier: Apache-2.0
"""The native per-model records (bindings/native_model_records.mojo).

A recorded model (one whose public call already stored its admission records
and generation stamps) must behave exactly like a twin that has no record (a
deep copy: records are keyed by object identity, so a copy starts empty)
after any edit of its state, and a record must never keep its model alive.
"""
import copy
import gc
import weakref

import numpy as np
import pytest

import mojolearn

CASES = [
    ("RandomForestClassifier", {"n_estimators": 4, "random_state": 0}, True),
    ("AdaBoostClassifier", {"n_estimators": 3, "random_state": 0}, True),
    ("GradientBoostingRegressor", {"n_estimators": 3}, False),
    ("KernelRidge", {"alpha": 1.0}, False),
    ("GaussianProcessRegressor", {}, False),
    ("LinearDiscriminantAnalysis", {}, True),
]


def _data(classifier):
    rng = np.random.default_rng(0)
    X = rng.random((40, 4), dtype=np.float32)
    y = (X[:, 0] > 0.5).astype(np.int64) if classifier else (X @ np.arange(1, 5, dtype=np.float32))
    return X, y


def _estimator(name, params):
    cls = getattr(mojolearn, name, None)
    if cls is None:
        pytest.skip(f"mojolearn does not export {name}")
    return cls(**params)


def _outcome(call):
    try:
        result = call()
    except Exception as error:  # the refusal itself is the observable outcome
        return ("raise", type(error).__name__, str(error))
    return ("ok", np.asarray(result).tobytes())


def _recorded(name, params, classifier):
    X, y = _data(classifier)
    model = _estimator(name, params).fit(X, y)
    # Two calls: the first stores the records, the second takes the
    # generation-stamped path.
    first = _outcome(lambda: model.predict(X))
    second = _outcome(lambda: model.predict(X))
    assert first == second
    return model, X


def _twin(model):
    try:
        return copy.deepcopy(model)
    except Exception as error:
        pytest.skip(f"{type(model).__name__} does not deep-copy: {error}")


def _first_param(model):
    names = sorted(model.get_params(deep=False))
    if not names:
        pytest.skip("estimator has no constructor parameters")
    return names[0]


def _edit_reassign(model):
    setattr(model, _first_param(model), object())


def _edit_delete(model):
    delattr(model, _first_param(model))


def _edit_new_dict(model):
    model.__dict__ = dict(model.__dict__)


def _edit_members_in_place(model):
    for name in ("estimators_", "_estimators", "trees_"):
        members = model.__dict__.get(name)
        if type(members) is list and len(members) > 1:
            members.reverse()
            return
    pytest.skip("no exact member list to edit in place")


def _edit_container_param(model):
    for name, value in model.get_params(deep=False).items():
        if type(value) is dict and value:
            key = next(iter(value))
            value[key] = object()
            return
        if type(value) is list and value:
            value[0] = object()
            return
    pytest.skip("no exact container parameter to edit in place")


def _edit_subclass(model):
    model.__class__ = type("RecordedSubclass", (type(model),), {})


EDITS = {
    "reassign": _edit_reassign,
    "delete": _edit_delete,
    "new_dict": _edit_new_dict,
    "members_in_place": _edit_members_in_place,
    "container_param": _edit_container_param,
    "subclass": _edit_subclass,
}


@pytest.mark.parametrize("edit", sorted(EDITS))
@pytest.mark.parametrize("name,params,classifier", CASES, ids=[c[0] for c in CASES])
def test_recorded_model_matches_unrecorded_twin_after_edit(name, params, classifier, edit):
    model, X = _recorded(name, params, classifier)
    twin = _twin(model)
    try:
        EDITS[edit](model)
    except TypeError as error:
        pytest.skip(f"edit refused by Python: {error}")
    EDITS[edit](twin)
    # Twice each: the record must miss after the edit, then settle again.
    for _ in range(2):
        assert _outcome(lambda: model.predict(X)) == _outcome(lambda: twin.predict(X))


@pytest.mark.parametrize("name,params,classifier", CASES, ids=[c[0] for c in CASES])
def test_set_params_after_record_matches_fresh_fit(name, params, classifier):
    model, X = _recorded(name, params, classifier)
    X2, y2 = _data(classifier)
    current = model.get_params(deep=False)
    model.set_params(**current)
    fresh = _estimator(name, params).fit(X2, y2)
    refit = model.fit(X2, y2)
    assert _outcome(lambda: refit.predict(X)) == _outcome(lambda: fresh.predict(X))


@pytest.mark.parametrize("name,params,classifier", CASES, ids=[c[0] for c in CASES])
def test_records_never_keep_their_model_alive(name, params, classifier):
    def recorded_references():
        model, _ = _recorded(name, params, classifier)
        references = [weakref.ref(model)]
        for field in ("estimators_", "_estimators", "trees_", "_learner_"):
            value = model.__dict__.get(field)
            members = value if type(value) in (list, tuple) else [value]
            for member in members:
                try:
                    references.append(weakref.ref(member))
                except TypeError:
                    pass
        return references

    references = recorded_references()
    gc.collect()
    assert [ref for ref in references if ref() is not None] == []


# One estimator per family with a resident device store: fit, predict (the
# resident copy now exists), set_params, fit again. The reset must release
# the resident owners natively and the refit must match a fresh model.
def _tabular(kind):
    rng = np.random.default_rng(1)
    X = rng.random((48, 4), dtype=np.float32)
    if kind == "clf":
        return (X, (X[:, 0] > 0.5).astype(np.int64)), "predict", (X,)
    if kind == "reg":
        return (X, X @ np.arange(1, 5, dtype=np.float32)), "predict", (X,)
    if kind == "predict":
        return (X,), "predict", (X,)
    return (X,), "transform", (X,)


def _series(_kind):
    rng = np.random.default_rng(1)
    y = np.cumsum(rng.random(80)).astype(np.float64)
    return (y,), "forecast", (3,)


def _sequences(_kind):
    rng = np.random.default_rng(1)
    X = rng.random((24, 6, 3), dtype=np.float32)
    return (X, (X[:, 0, 0] > 0.5).astype(np.int64)), "predict", (X,)


RESIDENT_CASES = [
    ("GaussianProcessRegressor", {}, _tabular, "reg"),
    ("SVC", {}, _tabular, "clf"),
    ("KernelRidge", {"alpha": 1.0}, _tabular, "reg"),
    ("TruncatedSVD", {"n_components": 2, "random_state": 0}, _tabular, "transform"),
    ("KMeans", {"n_clusters": 3, "random_state": 0, "n_init": 1}, _tabular, "predict"),
    ("RandomForestClassifier", {"n_estimators": 4, "random_state": 0}, _tabular, "clf"),
    ("ARIMA", {"order": (1, 0, 0)}, _series, None),
    ("GRUClassifier", {"hidden_size": 4, "max_epochs": 1, "random_state": 0}, _sequences, None),
]


@pytest.mark.parametrize("name,params,data,kind", RESIDENT_CASES, ids=[c[0] for c in RESIDENT_CASES])
def test_set_params_after_resident_predict_then_refit(name, params, data, kind):
    fit_args, method, call_args = data(kind)
    try:
        model = _estimator(name, params).fit(*fit_args)
    except TypeError as error:
        pytest.skip(f"{name} does not take these settings: {error}")
    first = _outcome(lambda: getattr(model, method)(*call_args))
    assert _outcome(lambda: getattr(model, method)(*call_args)) == first
    assert model.set_params(**model.get_params(deep=False)) is model
    refit = model.fit(*fit_args)
    fresh = _estimator(name, params).fit(*fit_args)
    assert _outcome(lambda: getattr(refit, method)(*call_args)) == _outcome(lambda: getattr(fresh, method)(*call_args))
