# SPDX-License-Identifier: Apache-2.0
"""NearestNeighbors.fit and the k-NN classifier/regressor fit, predict,
predict_proba and score are one native call each
(bindings/neighbors_knn_native.mojo). Every case compares the native method
with the frozen former Python body (`_legacy_neighbors_knn`) bit for bit:
fitted state, outputs and refusals (class and wording)."""
import inspect

import numpy as np
import pytest

import mojolearn
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.tests import _legacy_neighbors_knn as legacy


def _data(seed=0, n=40, d=3, nq=9):
    rng = np.random.default_rng(seed)
    return rng.random((n, d), dtype=np.float32), rng.random((nq, d), dtype=np.float32), rng


def _bits(a):
    a = np.asarray(a)
    return a.dtype.str, a.shape, a.tobytes()


def _same(a, b):
    if isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b)
        for x, y in zip(a, b):
            _same(x, y)
        return
    assert _bits(a) == _bits(b)


def _state(model):
    fields = vars(model)
    out = {}
    for name in ("n_samples_fit_", "n_features_in_", "outputs_2d_", "_classes_list", "_label_names"):
        if name in fields:
            out[name] = fields[name]
    for name in ("_index", "_y_cols"):
        if fields.get(name) is not None:
            out[name] = _bits(fields[name])
    return out


def _raw(cls, name):
    """The class's own method without the package's Python wrappers."""
    return inspect.unwrap(getattr(cls, name))


def _error(call, *args):
    try:
        call(*args)
    except Exception as error:  # noqa: BLE001 - the class and text are compared
        return type(error), str(error)
    return None


CLASSIFIER_CASES = {
    "int": dict(y=lambda rng, n: rng.integers(-3, 4, n)),
    "int32": dict(y=lambda rng, n: rng.integers(0, 5, n).astype(np.int32)),
    "whole-float": dict(y=lambda rng, n: rng.integers(0, 3, n).astype(np.float64)),
    "str": dict(y=lambda rng, n: np.array(["b", "a", "c"])[rng.integers(0, 3, n)]),
    "list": dict(y=lambda rng, n: [int(v) for v in rng.integers(0, 4, n)]),
    "2d-c": dict(y=lambda rng, n: rng.integers(0, 4, (n, 3))),
    "2d-f": dict(y=lambda rng, n: np.asfortranarray(rng.integers(0, 4, (n, 2)))),
    "2d-one": dict(y=lambda rng, n: rng.integers(0, 4, (n, 1))),
}
PARAMS = [
    dict(),
    dict(weights="distance"),
    dict(metric="minkowski", p=3),
    dict(metric="manhattan", n_neighbors=7),
    dict(metric="jensenshannon"),
]


@pytest.mark.parametrize("case", sorted(CLASSIFIER_CASES))
@pytest.mark.parametrize("params", PARAMS, ids=str)
def test_classifier_matches_former_path(case, params):
    X, Q, rng = _data()
    y = CLASSIFIER_CASES[case]["y"](rng, X.shape[0])
    native = mojolearn.KNeighborsClassifier(**params)
    former = mojolearn.KNeighborsClassifier(**params)
    assert native.fit(X, y) is native
    legacy.clf_fit(former, X, y)
    assert _state(native) == _state(former)
    _same(native.predict(Q), legacy.clf_predict(former, Q))
    _same(native.predict_proba(Q), legacy.clf_predict_proba(former, Q))
    truth = y[: Q.shape[0]] if not isinstance(y, list) else y[: Q.shape[0]]
    if np.ndim(y) == 1:
        assert native.score(Q, truth) == legacy.clf_score(former, Q, truth)
        w = np.linspace(0.5, 2.0, Q.shape[0], dtype=np.float32)
        assert native.score(Q, truth, w) == legacy.clf_score(former, Q, truth, w)


@pytest.mark.parametrize("params", PARAMS, ids=str)
@pytest.mark.parametrize("outputs", [1, 2, "column"])
def test_regressor_matches_former_path(params, outputs):
    X, Q, rng = _data(1)
    if outputs == 1:
        y = rng.random(X.shape[0])
    elif outputs == "column":
        y = rng.random((X.shape[0], 1), dtype=np.float32)
    else:
        y = np.asfortranarray(rng.random((X.shape[0], 3)))
    native = mojolearn.KNeighborsRegressor(**params)
    former = mojolearn.KNeighborsRegressor(**params)
    native.fit(X, y)
    legacy.reg_fit(former, X, y)
    assert _state(native) == _state(former)
    _same(native.predict(Q), legacy.reg_predict(former, Q))
    truth = np.asarray(y)[: Q.shape[0]]
    assert native.score(Q, truth) == legacy.reg_score(former, Q, truth)


@pytest.mark.parametrize("params", PARAMS[:4] + [dict(algorithm="rbc"), dict(algorithm="rbc", metric="chebyshev")], ids=str)
def test_nearest_neighbors_fit_matches_former_path(params):
    X, Q, _ = _data(2)
    params = {k: v for k, v in params.items() if k != "weights"}
    native = mojolearn.NearestNeighbors(**params).fit(X)
    former = mojolearn.NearestNeighbors(**params)
    legacy.nn_fit(former, X)
    assert _state(native) == _state(former)
    _same(native.kneighbors(Q), former.kneighbors(Q))


REFUSED = [
    (mojolearn.NearestNeighbors, dict(metric="haversine")),
    (mojolearn.NearestNeighbors, dict(metric="nope")),
    (mojolearn.NearestNeighbors, dict(metric=len)),
    (mojolearn.NearestNeighbors, dict(metric="euclidean", p=3)),
    (mojolearn.NearestNeighbors, dict(metric="minkowski", p=0)),
    (mojolearn.NearestNeighbors, dict(metric="minkowski", p=float("inf"))),
    (mojolearn.NearestNeighbors, dict(metric="minkowski", p=1e-40)),
    (mojolearn.NearestNeighbors, dict(algorithm="kd_tree")),
    (mojolearn.NearestNeighbors, dict(algorithm="rbc", metric="cosine")),
    (mojolearn.NearestNeighbors, dict(algorithm="rbc", metric="minkowski", p=0.5)),
    (mojolearn.KNeighborsClassifier, dict(algorithm="rbc")),
    (mojolearn.KNeighborsClassifier, dict(weights=len)),
    (mojolearn.KNeighborsClassifier, dict(weights="gaussian")),
    (mojolearn.KNeighborsRegressor, dict(weights="distance", metric="inner_product")),
]


@pytest.mark.parametrize("cls,params", REFUSED, ids=str)
def test_refusals_keep_class_and_wording(cls, params):
    X, _, rng = _data(3)
    y = rng.integers(0, 3, X.shape[0])
    native = cls(**params)
    former = cls(**params)
    if cls is mojolearn.NearestNeighbors:
        got, want = _error(native.fit, X), _error(legacy.nn_fit, former, X)
    elif cls is mojolearn.KNeighborsClassifier:
        got, want = _error(native.fit, X, y), _error(legacy.clf_fit, former, X, y)
    else:
        got, want = _error(native.fit, X, y), _error(legacy.reg_fit, former, X, y)
    assert got is not None and got == want


@pytest.mark.parametrize("y", [np.zeros((40, 2, 2), np.int64), np.zeros(39, np.int64), np.array([0.5] * 40),
                               np.array([True, False] * 20), np.array([1 << 40] * 40)], ids=str)
def test_label_refusals_keep_class_and_wording(y):
    X, _, _ = _data(4)
    got = _error(mojolearn.KNeighborsClassifier().fit, X, y)
    want = _error(legacy.clf_fit, mojolearn.KNeighborsClassifier(), X, y)
    assert got is not None and got == want


def test_predict_refusals_keep_class_and_wording():
    X, Q, rng = _data(5)
    y = rng.integers(0, 3, X.shape[0])
    for change in (dict(n_neighbors=0), dict(n_neighbors=41), dict(metric="nope")):
        native = mojolearn.KNeighborsClassifier().fit(X, y)
        former = mojolearn.KNeighborsClassifier()
        legacy.clf_fit(former, X, y)
        native.set_params(**change)
        former.set_params(**change)
        assert _error(native.predict, Q) == _error(legacy.clf_predict, former, Q)
    native = mojolearn.KNeighborsClassifier().fit(X, y)
    assert _error(native.predict, Q[:, :2]) == (ValueError, "mojolearn: X has 2 features, index has 3")


def test_methods_run_as_one_native_call():
    X, Q, rng = _data(6)
    y = rng.integers(0, 3, X.shape[0])
    clf = mojolearn.KNeighborsClassifier()
    assert audit_native_call(_raw(mojolearn.KNeighborsClassifier, "fit"), clf, X, y).assert_native() is clf
    audit_native_call(_raw(mojolearn.KNeighborsClassifier, "predict"), clf, Q).assert_native()
    audit_native_call(_raw(mojolearn.KNeighborsClassifier, "predict_proba"), clf, Q).assert_native()
    audit_native_call(_raw(mojolearn.KNeighborsClassifier, "score"), clf, Q, y[:9]).assert_native()
    reg = mojolearn.KNeighborsRegressor()
    audit_native_call(_raw(mojolearn.KNeighborsRegressor, "fit"), reg, X, X[:, 0]).assert_native()
    audit_native_call(_raw(mojolearn.KNeighborsRegressor, "predict"), reg, Q).assert_native()
    audit_native_call(_raw(mojolearn.KNeighborsRegressor, "score"), reg, Q, Q[:, 0]).assert_native()
    nn = mojolearn.NearestNeighbors()
    audit_native_call(_raw(mojolearn.NearestNeighbors, "fit"), nn, X).assert_native()
