"""OPTICS public scalar, publication and storage seams for typed transport."""
import gc
import pickle

import numpy as np
import pytest

import mojolearn as ml

__mojolearn_user_boundary__ = True


def _points():
    return np.array([[0], [1], [2], [10], [11], [12]], dtype=np.float32)


@pytest.mark.parametrize("size", [2, 2.0, np.float32(0.4), np.float64(0.4)])
def test_optics_retains_scalar_fraction_and_count_admission(size):
    model = ml.OPTICS(min_samples=size, cluster_method="dbscan", max_eps=2,
                      eps=1.1, numeric_mode="identical").fit(_points())
    np.testing.assert_array_equal(model.labels_, [0, 0, 0, 1, 1, 1])
    np.testing.assert_allclose(model.core_distances_, 1, rtol=0, atol=0)


def test_optics_min_samples_refusal_precedes_later_float_conversion():
    model = ml.OPTICS(min_samples=7, max_eps="bad", xi="bad", numeric_mode="identical")
    with pytest.raises(ValueError, match="min_samples must be no greater"):
        model.fit(_points())
    assert "ordering_" not in vars(model)
    assert "labels_" not in vars(model)


def test_optics_dbscan_refit_keeps_old_hierarchy_owner():
    model = ml.OPTICS(min_samples=2, min_cluster_size=2, max_eps=2,
                      numeric_mode="identical").fit(_points())
    previous = vars(model)["cluster_hierarchy_"]
    model.cluster_method = "dbscan"
    model.eps = 1.1
    model.fit(_points())
    assert vars(model)["cluster_hierarchy_"] is previous
    np.testing.assert_array_equal(model.labels_, [0, 0, 0, 1, 1, 1])


def test_optics_exporters_survive_native_result_and_model_teardown():
    model = ml.OPTICS(min_samples=2, cluster_method="dbscan", max_eps=2,
                      eps=1.1, numeric_mode="identical").fit(_points())
    labels = np.asarray(model.labels_)
    core = np.asarray(model.core_distances_)
    restored = pickle.loads(pickle.dumps(model))
    del model
    gc.collect()
    np.testing.assert_array_equal(labels, [0, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(core, np.ones(6, dtype=np.float32))
    np.testing.assert_array_equal(restored.labels_, labels)
