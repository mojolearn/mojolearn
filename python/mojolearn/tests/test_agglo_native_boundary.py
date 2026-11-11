# SPDX-License-Identifier: Apache-2.0
"""Agglomerative public admission, storage and ownership seams; kernels use Mojo fixtures."""
import copy
import gc
import weakref

import numpy as np
import pytest
import mojolearn as ml


def _x():
    return np.array([[0, 0], [0.1, 0], [0, 0.1], [0.1, 0.1],
                     [4, 4], [4.1, 4], [4, 4.1], [4.1, 4.1]], dtype=np.float32)


def _model(**options):
    settings = dict(n_clusters=2, linkage="single", metric="euclidean",
                    prediction_data=True, numeric_mode="identical")
    settings.update(options)
    return ml.AgglomerativeClustering(**settings)


@pytest.mark.parametrize("options", [{}, {"compute_distances": True}, {"linkage": "ward"},
                                     {"linkage": "average", "metric": "cosine"}])
def test_agglo_typed_requests_bypass_legacy_compute_entries(options, monkeypatch):
    from mojolearn import _backend

    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("legacy numerical entry was called")

    for name, entries in {
        "_mojolearn_solver": ("linkage_fit",),
        "_mojolearn_x_cluster": ("x_cluster_call",),
        "_mojolearn_x_neighbors": ("x_neighbors_call",),
        "_mojolearn_estimators": ("labeled_reference_prepare", "labeled_reference_predict", "labeled_reference_predict_resident"),
    }.items():
        binding = _backend.binding(name, "identical")
        for entry in entries:
            monkeypatch.setattr(binding, entry, forbidden, raising=False)
    model = _model(**options)
    assert model.fit(_x() + 0.25) is model
    assert np.asarray(model.labels_).shape == (8,)
    if model.linkage == "single":
        assert np.asarray(model.predict(_x()[:2])).shape == (2,)
    assert calls == []


@pytest.mark.parametrize("route,module,factory", [
    ("solver", "_mojolearn_solver", "hierarchy_solver_native_api_v1"),
    ("cluster", "_mojolearn_x_cluster", "hierarchy_cluster_native_api_v1"),
    ("predict", "_mojolearn_estimators", "dbscan_native_api_v1"),
])
def test_agglo_python_descriptor_factory_is_refused_without_call(route, module, factory, monkeypatch):
    from mojolearn import _backend

    model = _model(compute_distances=route == "cluster").fit(_x())
    calls = []

    def forbidden():
        calls.append(True)
        raise AssertionError("Python factory was called")

    monkeypatch.setattr(_backend.binding(module, "identical"), factory, forbidden)
    with pytest.raises(TypeError, match="compiled extension entry"):
        model.predict(_x()) if route == "predict" else model.fit(_x())
    assert calls == []


@pytest.mark.parametrize("field", ["n_clusters", "linkage", "metric", "c", "compute_full_tree",
                                    "distance_threshold", "compute_distances", "prediction_data"])
def test_agglo_rebound_scalar_protocols_are_refused_without_hooks(field):
    model = _model()
    calls = []

    class Protocol:
        @property
        def __class__(self):
            calls.append("class")
            return int

        def __int__(self):
            calls.append("int")
            return 2

        def __float__(self):
            calls.append("float")
            return 0.5

        def __bool__(self):
            calls.append("bool")
            return True

        def __eq__(self, other):
            calls.append("eq")
            return False

        def __hash__(self):
            calls.append("hash")
            return 0

        def __repr__(self):
            calls.append("repr")
            return "native"

    setattr(model, field, Protocol())
    with pytest.raises(TypeError, match="native|hooks"):
        model.fit(_x())
    assert calls == []


def test_agglo_connectivity_callback_is_refused_without_call():
    calls = []

    def connectivity(x):
        calls.append(True)
        return np.eye(len(x), dtype=np.float32)

    with pytest.raises(TypeError, match="connectivity callbacks"):
        _model(connectivity=connectivity).fit(_x())
    assert calls == []


def test_agglo_normalized_metadata_precedes_prediction_flag_refusal():
    model = _model(prediction_data=np.bool_(True), connectivity=None, compute_distances=1)
    with pytest.raises(TypeError, match="prediction_data must be a bool"):
        model.fit(_x())
    assert model._connectivity == "pairwise"
    assert model._compute_distances is True
    assert "labels_" not in vars(model)


def test_agglo_fit_predict_returns_public_label_storage():
    model = _model(compute_distances=True)
    labels = model.fit_predict(_x())
    assert np.shares_memory(np.asarray(labels), np.asarray(model.labels_))
    assert np.asarray(model.children_).shape == (7, 2)
    assert np.asarray(model.distances_).shape == (7,)


def test_agglo_predict_keeps_namespace_identity_and_fitted_snapshot():
    x = _x()
    model = _model().fit(x)
    expected = np.asarray(model.predict(_x())).copy()
    before = dict(vars(model))
    x[:] = 100
    np.testing.assert_array_equal(np.asarray(model.predict(_x())), expected)
    assert vars(model).keys() == before.keys()
    assert all(vars(model)[name] is value for name, value in before.items())


def test_agglo_mutable_rebound_labels_remain_visible():
    model = _model().fit(_x())
    refs = np.asarray(model._fit_X).copy()
    labels = np.asarray(model.labels_).copy()
    model._fit_X, model.labels_ = refs, labels
    before = np.asarray(model.predict(refs)).copy()
    labels += 9
    np.testing.assert_array_equal(np.asarray(model.predict(refs)), before + 9)
    assert model._fit_X is refs and model.labels_ is labels


def test_agglo_pickle_strips_native_residency_and_refits():
    model = _model().fit(_x())
    expected = np.asarray(model.predict(_x())).copy()
    assert "_resident_holder" not in model.__getstate__()
    restored = copy.deepcopy(model)
    np.testing.assert_array_equal(np.asarray(restored.predict(_x())), expected)
    assert restored.fit(_x()) is restored
    assert model.fit(_x()) is model


def test_agglo_legacy_internal_predict_keeps_direct_uploads():
    from mojolearn import _backend, _hierarchy_impl

    model = _model().fit(_x())
    expected = np.asarray(model.predict(_x())).copy()
    model._resident_holder = None
    binding = _backend.binding("_mojolearn_estimators", "identical")
    result = _hierarchy_impl._NATIVE(model, "family_xc_agglo_predict", (binding, _x()), _hierarchy_impl._NATIVE_CONTEXT)
    np.testing.assert_array_equal(np.asarray(result), expected)
    assert model._resident_holder is None


def test_agglo_resident_owner_does_not_keep_estimator_alive():
    model = _model().fit(_x())
    model.predict(_x())
    reference = weakref.ref(model)
    del model
    gc.collect()
    assert reference() is None


@pytest.mark.parametrize("value", [np.array(7), np.array([7])])
def test_agglo_refusal_does_not_invoke_numpy_formatters(value):
    calls = []

    def forbidden(value):
        calls.append(True)
        raise AssertionError("array formatter was called")

    with np.printoptions(formatter={"all": forbidden}):
        with pytest.raises(NotImplementedError, match="c=.*refused"):
            _model(c=value).fit(_x())
    assert calls == []
