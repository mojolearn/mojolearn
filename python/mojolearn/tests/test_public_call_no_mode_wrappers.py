# SPDX-License-Identifier: Apache-2.0
"""A package estimator's public method is its native owner's one binding call.

`_mode` resolves at class creation what its per-call wrappers (`public`,
`tracked`, `guarded`, the `_loader` wrapper `load`) used to decide on every
call: a package class gets none of them, because each public method's native
owner validates the input, tracks the fit, refuses an unfitted call, publishes
`_mojolearn_fitted` after a load and returns the public result itself. Only an
application subclass's own overrides (and, for a base not marked native, its
inherited methods) are still wrapped (lane py-mode-wrappers, 2026-11-15).

How to see it: `test_no_python_frame_between_public_method_and_binding`
profiles one estimator per family and fails on any Python frame entered
before the public method's own frame (a wrapper or descriptor) or called from
it outside a C call (a helper between the method and the binding).
"""
import inspect
import os
import sys

import numpy as np
import pytest

import mojolearn as ml
from mojolearn import _mode

_NAMES = tuple(dict.fromkeys(_mode._OUTPUTS + _mode._FITS + _mode._NEEDS_FIT + _mode._LOADS))
_WRAPPERS = frozenset(("public", "guarded", "tracked", "load"))
_PACKAGE = os.path.dirname(os.path.abspath(ml.__file__)) + os.sep
_TESTS = os.path.join(_PACKAGE, "tests") + os.sep


def _function(raw):
    if isinstance(raw, (classmethod, staticmethod)):
        raw = raw.__func__
    if isinstance(raw, _mode.available_if):
        raw = raw.fn
    return raw


def _is_wrapper(fn):
    code = getattr(fn, "__code__", None)
    return (code is not None and os.path.abspath(code.co_filename) == os.path.abspath(_mode.__file__)
            and code.co_name in _WRAPPERS)


def _public_classes():
    for name in ml.__all__:  # glue: test inventory of the public names
        try:
            value = getattr(ml, name)
        except Exception:  # a family without its binding on this install: other tests own that
            continue
        if isinstance(value, type) and issubclass(value, _mode.PublicOutputsMixin):
            yield value


def test_no_package_class_carries_a_mode_wrapper():
    offenders = []
    for cls in _public_classes():  # glue: test inventory
        for name in _NAMES:  # glue: test inventory
            fn = _function(inspect.getattr_static(cls, name, None))
            if _is_wrapper(fn):
                offenders.append(f"{cls.__name__}.{name} ({fn.__code__.co_name})")
    assert offenders == []


class _AppForest(ml.RandomForestClassifier):
    def predict(self, X):
        return super().predict(X)


class _AppAgglomerative(ml.AgglomerativeClustering):
    pass


def test_application_subclasses_keep_their_wrappers():
    # An override on a class marked native: its own method is guarded.
    assert _is_wrapper(vars(_AppForest)["predict"])
    # A subclass of a package class not marked native: the inherited package
    # loader is wrapped on the subclass, so its load still marks it fitted.
    assert _is_wrapper(_function(inspect.getattr_static(_AppAgglomerative, "load")))
    # The package classes themselves stay unwrapped.
    assert not _is_wrapper(vars(ml.RandomForestClassifier)["predict"])
    assert not _is_wrapper(_function(inspect.getattr_static(ml.AgglomerativeClustering, "load")))


def _frames_between(call, method):
    """The package Python frames entered by `call()` before `method`'s own
    frame, or called directly from it while no C call is active."""
    code = _function(method).__code__
    state = {"top": None, "c": 0}
    extra = []

    def record(frame, event, arg):
        if event == "call":
            filename = os.path.abspath(frame.f_code.co_filename)
            if not filename.startswith(_PACKAGE) or filename.startswith(_TESTS):
                return
            if state["top"] is None:
                if frame.f_code is code:
                    state["top"] = frame
                else:
                    extra.append(getattr(frame.f_code, "co_qualname", frame.f_code.co_name))
            elif frame.f_back is state["top"] and state["c"] == 0:
                extra.append(getattr(frame.f_code, "co_qualname", frame.f_code.co_name))
        elif event == "c_call" and frame is state["top"]:
            state["c"] += 1
        elif event in ("c_return", "c_exception") and frame is state["top"]:
            state["c"] -= 1

    previous = sys.getprofile()
    sys.setprofile(record)
    try:
        result = call()
    finally:
        sys.setprofile(previous)
    assert state["top"] is not None, "the public method's own frame never ran"
    return extra, result


def _data():
    rng = np.random.default_rng(0)
    X = np.ascontiguousarray(rng.standard_normal((64, 4)).astype(np.float32))
    y_class = (X[:, 0] > 0).astype(np.int64)
    y_value = np.ascontiguousarray((X[:, 0] + 0.5 * X[:, 1]).astype(np.float32))
    return X, y_class, y_value


# One estimator per family: (factory, target kind, query method).
_FAMILIES = {
    "linear": (lambda: ml.Ridge(alpha=1.0), "value", "predict"),
    "forest": (lambda: ml.RandomForestClassifier(n_estimators=2, max_depth=2), "class", "predict"),
    "extra_trees": (lambda: ml.ExtraTreesRegressor(n_estimators=2, max_depth=2), "value", "predict"),
    "gbdt": (lambda: ml.GradientBoostingRegressor(n_estimators=2, max_depth=2), "value", "predict"),
    "iforest": (lambda: ml.IsolationForest(n_estimators=4), None, "predict"),
    "meta": (lambda: ml.VotingRegressor([("a", ml.RandomForestRegressor(n_estimators=2, max_depth=2))]),
             "value", "predict"),
    "neighbors": (lambda: ml.KNeighborsClassifier(n_neighbors=3), "class", "predict"),
    "cluster": (lambda: ml.KMeans(n_clusters=2), None, "predict"),
    "hierarchy": (lambda: ml.AgglomerativeClustering(n_clusters=2), None, None),
    "decomposition": (lambda: ml.PCA(n_components=2), None, "transform"),
    "preprocessing": (lambda: ml.StandardScaler(), None, "transform"),
    "mixture": (lambda: ml.GaussianMixture(n_components=2), None, "predict"),
    "svm": (lambda: ml.SVC(), "class", "predict"),
    "kernel": (lambda: ml.KernelRidge(), "value", "predict"),
    "manifold": (lambda: ml.Isomap(n_neighbors=5, n_components=2), None, "transform"),
}


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_no_python_frame_between_public_method_and_binding(family):
    factory, target, query = _FAMILIES[family]
    X, y_class, y_value = _data()
    model = factory()
    args = (X,) if target is None else (X, y_class if target == "class" else y_value)
    cls = type(model)
    extra, _ = _frames_between(lambda: model.fit(*args), inspect.getattr_static(cls, "fit"))
    assert extra == [], f"{cls.__name__}.fit: {extra}"
    if query is not None:
        extra, _ = _frames_between(lambda: getattr(model, query)(X), inspect.getattr_static(cls, query))
        assert extra == [], f"{cls.__name__}.{query}: {extra}"


@pytest.mark.parametrize("factory", [
    lambda: ml.RandomForestClassifier(n_estimators=2, max_depth=2),
    lambda: ml.ExtraTreesClassifier(n_estimators=2, max_depth=2),
    # save() refuses a fit without prediction data (documented; default prediction_data=False since 6743480016)
    lambda: ml.AgglomerativeClustering(n_clusters=2, prediction_data=True),
], ids=["random_forest", "extra_trees", "agglomerative"])
def test_load_is_one_binding_call_and_publishes_fitted(tmp_path, factory):
    X, y_class, _ = _data()
    model = factory()
    model = model.fit(X) if isinstance(model, ml.AgglomerativeClustering) else model.fit(X, y_class)
    path = str(tmp_path / "model.npz")
    model.save(path)
    cls = type(model)
    extra, loaded = _frames_between(lambda: cls.load(path), inspect.getattr_static(cls, "load"))
    assert extra == [], f"{cls.__name__}.load: {extra}"
    assert type(loaded) is cls
    # The native loader marks the instance fitted (formerly the Python loader
    # wrapper's job): the fitted check reads the flag, not a field scan.
    assert vars(loaded).get("_mojolearn_fitted") is True
    assert _mode.is_fitted(loaded)
