"""Public HDBSCAN packing, prepared-cache invalidation and callback seams."""
import gc
import pickle

import numpy as np
import pytest

from mojolearn import HDBSCAN
from mojolearn import hdbscan as hd

__mojolearn_user_boundary__ = True


def _points():
    return np.array([[0], [1], [2], [10], [11], [12]], dtype=np.float32)


def _model():
    return HDBSCAN(min_samples=2, min_cluster_size=2, prediction_data=True,
                   numeric_mode="identical").fit(_points())


def test_prediction_record_stays_outside_public_fields_and_preserves_array_identity():
    model = _model()
    before = dict(vars(model))
    pd_before = dict(model._prediction_data)
    hd.approximate_predict(model, _points()[[1, 4]])
    hd.membership_vector(model, _points()[[1, 4]], batch_size=1)
    hd.all_points_membership_vectors(model, batch_size=2)
    assert tuple(vars(model)) == tuple(before)
    assert all(vars(model)[key] is value for key, value in before.items())
    assert tuple(model._prediction_data) == tuple(pd_before)
    assert all(model._prediction_data[key] is value for key, value in pd_before.items())


@pytest.mark.parametrize("method", [hd.approximate_predict, hd.membership_vector])
def test_prepared_query_rechecks_rebound_tree_counts(method):
    model = _model()
    method(model, _points()[[1, 4]])
    model._prediction_data["n_edges"] = -1
    with pytest.raises(TypeError, match="tree counts require non-negative native integers"):
        method(model, _points()[[1, 4]])


def test_prepared_query_rechecks_rebound_fitted_dimensions():
    model = _model()
    hd.approximate_predict(model, _points()[[1, 4]])
    model.n_features_in_ = 2
    with pytest.raises(ValueError, match="training storage differs from its fitted shape"):
        hd.approximate_predict(model, _points()[[1, 4]])


def test_prediction_metadata_foreign_key_refused_without_hooks():
    calls = []

    class Key:
        def __hash__(self):
            calls.append("hash")
            return hash("n_edges")

        def __eq__(self, other):
            calls.append("equal")
            return False

    model = _model()
    model._prediction_data[Key()] = 1
    calls.clear()
    with pytest.raises(TypeError, match="native|exact string|unsupported|callback"):
        hd.approximate_predict(model, _points()[[1, 4]])
    assert calls == []


def test_query_factory_replacement_cannot_execute_python(monkeypatch):
    calls = []

    def forbidden():
        calls.append(True)
        raise AssertionError("Python HDBSCAN query factory executed")

    model = _model()
    monkeypatch.setattr(model._bind(), "hdbscan_query_native_api_v1", forbidden)
    with pytest.raises(TypeError, match="compiled extension|compiled factory|native descriptor"):
        hd.approximate_predict(model, _points()[[1, 4]])
    assert calls == []


def test_public_prediction_exporters_survive_native_plan_and_model_teardown():
    model = _model()
    labels = np.asarray(model.labels_)
    parents = np.asarray(model._prediction_data["parents"])
    exemplars = np.asarray(model._prediction_data["exemplar_idx"])
    parent_copy = parents.copy()
    exemplar_copy = exemplars.copy()
    restored = pickle.loads(pickle.dumps(model))
    del model
    gc.collect()
    np.testing.assert_array_equal(parents, parent_copy)
    np.testing.assert_array_equal(exemplars, exemplar_copy)
    predicted, strength = hd.approximate_predict(restored, _points()[[1, 4]])
    np.testing.assert_array_equal(predicted, labels[[1, 4]])
    np.testing.assert_allclose(strength, 1, rtol=0, atol=1e-6)


def test_prediction_training_storage_retains_copy_on_borrow():
    points = _points()
    model = HDBSCAN(min_samples=2, min_cluster_size=2, prediction_data=True,
                    numeric_mode="identical").fit(points)
    points[:] = -100
    np.testing.assert_array_equal(model._raw_data, _points())
    predicted, _ = hd.approximate_predict(model, _points()[[1, 4]])
    np.testing.assert_array_equal(predicted, np.asarray(model.labels_)[[1, 4]])
