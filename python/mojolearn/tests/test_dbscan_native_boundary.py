# SPDX-License-Identifier: Apache-2.0
"""DBSCAN Python admission/public ownership seams; numerical fixtures are Mojo."""
import copy
import gc
import weakref

import numpy as np
import pytest
import mojolearn as ml


def _x():
    return np.array([[0, 0], [0.1, 0], [2, 2], [2.1, 2], [10, 10]], dtype=np.float32)


def _model(**options):
    settings = dict(eps=0.25, min_samples=2, algorithm="brute", prediction_data=True, numeric_mode="identical")
    settings.update(options)
    return ml.DBSCAN(**settings)


def test_dbscan_native_requests_bypass_legacy_argument_list_entries(monkeypatch):
    from mojolearn import _backend

    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("legacy DBSCAN entry was called")

    binding = _backend.binding("_mojolearn_estimators", "identical")
    for name in ("dbscan_fit", "dbscan_fit_core", "dbscan_core_arrays", "labeled_reference_prepare", "labeled_reference_release", "labeled_reference_predict", "labeled_reference_predict_resident"):
        monkeypatch.setattr(binding, name, forbidden, raising=False)
    model = _model()
    assert model.fit(_x()) is model
    assert np.asarray(model.predict(_x()[:2])).shape == (2,)
    assert np.asarray(model.fit_predict(_x())).shape == (5,)
    assert calls == []


@pytest.mark.parametrize("operation", ["fit", "fit_predict", "predict"])
def test_dbscan_python_descriptor_factory_is_refused_before_call(operation, monkeypatch):
    from mojolearn import _backend

    model = _model().fit(_x())
    calls = []

    def forbidden():
        calls.append(True)
        raise AssertionError("Python DBSCAN descriptor factory was called")

    binding = _backend.binding("_mojolearn_estimators", "identical")
    monkeypatch.setattr(binding, "dbscan_native_api_v1", forbidden)
    with pytest.raises(TypeError, match="compiled extension entry"):
        getattr(model, operation)(_x())
    assert calls == []


@pytest.mark.parametrize("field", ["eps", "min_samples", "metric", "algorithm", "max_mbytes_per_batch", "max_iterations", "prediction_data"])
def test_dbscan_rebound_parameter_protocols_are_refused_without_hooks(field):
    model = _model().fit(_x())
    calls = []

    class Protocol:
        @property
        def __class__(self):
            calls.append("class")
            return float

        def __float__(self):
            calls.append("float")
            return 0.25

        def __int__(self):
            calls.append("int")
            return 2

        def __bool__(self):
            calls.append("bool")
            return True

        def __hash__(self):
            calls.append("hash")
            return 0

        def __str__(self):
            calls.append("str")
            return "euclidean"

        def __repr__(self):
            calls.append("repr")
            return "native"

    setattr(model, field, Protocol())
    with pytest.raises(TypeError, match="native|hooks"):
        model.fit(_x())
    assert calls == []


def test_dbscan_predict_keeps_the_public_namespace_unchanged():
    model = _model().fit(_x())
    before = dict(vars(model))
    model.predict(_x()[:2])
    after = vars(model)
    assert before.keys() == after.keys()
    assert all(after[name] is value for name, value in before.items())
    model.predict(_x()[2:4])
    assert all(vars(model)[name] is value for name, value in before.items())


def test_dbscan_legacy_internal_query_keeps_its_direct_upload_route():
    from mojolearn import _backend, density

    model = _model().fit(_x())
    expected = np.asarray(model.predict(_x())).copy()
    model._resident_holder = None
    binding = _backend.binding("_mojolearn_estimators", "identical")
    actual = density._NATIVE(model, "family_xc_dbscan_predict", (binding, _x()), density._NATIVE_CONTEXT)
    np.testing.assert_array_equal(np.asarray(actual), expected)
    assert model._resident_holder is None


def test_dbscan_fit_predict_preserves_public_label_storage_alias():
    model = _model()
    result = model.fit_predict(_x())
    assert np.shares_memory(np.asarray(result), np.asarray(model.labels_))
    assert np.asarray(model.core_sample_indices_).dtype == np.dtype("int32")
    assert np.asarray(model.components_).shape[1] == 2


def test_dbscan_mutable_rebound_reference_arrays_use_current_values():
    model = _model().fit(_x())
    components = np.asarray(model.components_).copy()
    keys = np.asarray(model.core_sample_indices_).copy()
    labels = np.asarray(model._core_labels).copy()
    model.components_, model.core_sample_indices_, model._core_labels = components, keys, labels
    query = components[:1].copy()
    first = np.asarray(model.predict(query)).copy()
    labels += 7
    np.testing.assert_array_equal(np.asarray(model.predict(query)), first + 7)
    assert model.components_ is components and model._core_labels is labels


def test_dbscan_precomputed_keeps_default_rbc_admission_and_brute_execution():
    x = _x()
    distances = np.sqrt(np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=2)).astype(np.float32)
    rbc = _model(metric="precomputed", algorithm="rbc", prediction_data=False)
    brute = _model(metric="precomputed", algorithm="brute", prediction_data=False)
    np.testing.assert_array_equal(np.asarray(rbc.fit_predict(distances)), np.asarray(brute.fit_predict(distances)))


def test_dbscan_precomputed_prediction_refusal_precedes_eps_validation():
    with pytest.raises(ValueError, match="prediction_data=True.*precomputed"):
        _model(metric="precomputed", algorithm="rbc", eps=-1).fit(np.zeros((5, 5), dtype=np.float32))


def test_dbscan_invalid_weight_preserves_early_publication_order():
    model = _model()
    with pytest.raises(ValueError):
        model.fit(_x(), sample_weight=np.ones(4, dtype=np.float32))
    assert "input_copied_" in vars(model)
    assert "labels_" not in vars(model)
    assert "core_sample_indices_" not in vars(model)


def test_dbscan_pickle_drops_residency_and_restores_reusable_public_state():
    model = _model().fit(_x())
    expected = np.asarray(model.predict(_x())).copy()
    assert "_resident_holder" not in model.__getstate__()
    restored = copy.deepcopy(model)
    np.testing.assert_array_equal(np.asarray(restored.predict(_x())), expected)
    assert restored.fit(_x()) is restored
    assert model.fit(_x()) is model


def test_dbscan_fit_core_storage_is_independent_of_caller_mutation():
    x = _x()
    model = _model().fit(x)
    expected = np.asarray(model.predict(_x())).copy()
    x[:] = 100
    np.testing.assert_array_equal(np.asarray(model.predict(_x())), expected)
    assert not np.asarray(model.components_).flags.writeable
    assert not np.asarray(model.core_sample_indices_).flags.writeable


def test_dbscan_owner_and_admission_cache_do_not_keep_model_alive():
    model = _model().fit(_x())
    model.predict(_x())
    reference = weakref.ref(model)
    del model
    gc.collect()
    assert reference() is None


def test_dbscan_invalid_array_metric_does_not_invoke_numpy_formatters():
    calls = []

    def forbidden(value):
        calls.append(True)
        raise AssertionError("NumPy formatter was called")

    with np.printoptions(formatter={"all": forbidden}):
        with pytest.raises(ValueError, match="metric=.*refused"):
            _model(metric=np.array(1.0)).fit(_x())
    assert calls == []
