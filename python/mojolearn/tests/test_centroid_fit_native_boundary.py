"""Retained public storage/publication seams for typed MeanShift/AP transport.

These are installed-boundary cases, not native kernel or performance evidence.
The broader existing cluster tests remain until replacement coverage passes.
"""
import gc
import pickle

import numpy as np
import pytest

import mojolearn as ml

__mojolearn_user_boundary__ = True


def test_meanshift_seed_shape_refusal_precedes_bad_bandwidth():
    model = ml.MeanShift(
        seeds=np.ones((1, 2), dtype=np.float32), bandwidth="bad-bandwidth",
        numeric_mode="identical",
    )
    with pytest.raises(ValueError, match="seeds have 2 features, X has 1"):
        model.fit(np.array([[0.0], [10.0]], dtype=np.float32))
    assert "cluster_centers_" not in vars(model)
    assert "labels_" not in vars(model)


def test_affinity_precomputed_shape_refusal_precedes_preference_length():
    model = ml.AffinityPropagation(
        affinity="precomputed", preference=[1.0], numeric_mode="identical",
    )
    with pytest.raises(ValueError, match="matrix of similarities must be a square array"):
        model.fit(np.ones((3, 2), dtype=np.float32))
    assert "cluster_centers_indices_" not in vars(model)
    assert "affinity_matrix_" not in vars(model)


def test_affinity_precomputed_refit_preserves_existing_centers_field():
    x = np.array([[0.0], [2.0], [6.0]], dtype=np.float32)
    model = ml.AffinityPropagation(
        preference=-1.0, damping=0.5, max_iter=1, convergence_iter=1,
        random_state=13, numeric_mode="identical",
    ).fit(x)
    owner = vars(model)["cluster_centers_"]
    model.affinity = "precomputed"
    similarities = np.array([[0.0, -4.0, -36.0], [-4.0, 0.0, -16.0],
                             [-36.0, -16.0, 0.0]], dtype=np.float32)
    assert model.fit(similarities) is model
    assert vars(model)["cluster_centers_"] is owner
    with pytest.raises(ValueError, match="Predict method is not supported"):
        model.predict(similarities)


@pytest.mark.parametrize("name", ["MeanShift", "AffinityPropagation"])
def test_centroid_public_arrays_survive_result_owner_and_model_teardown(name):
    x = np.array([[0.0], [2.0], [6.0]], dtype=np.float32)
    if name == "MeanShift":
        model = ml.MeanShift(bandwidth=0.5, numeric_mode="identical").fit(x)
    else:
        model = ml.AffinityPropagation(
            preference=-1.0, damping=0.5, max_iter=1, convergence_iter=1,
            random_state=13, numeric_mode="identical",
        ).fit(x)
    centers = np.asarray(model.cluster_centers_)
    labels = np.asarray(model.labels_)
    expected_centers = centers.copy()
    expected_labels = labels.copy()
    restored = pickle.loads(pickle.dumps(model))
    del model, x
    gc.collect()
    np.testing.assert_array_equal(centers, expected_centers)
    np.testing.assert_array_equal(labels, expected_labels)
    np.testing.assert_array_equal(restored.cluster_centers_, expected_centers)
    np.testing.assert_array_equal(restored.labels_, expected_labels)


def test_affinity_diagnostic_and_similarity_exporters_are_separate_public_storage():
    x = np.array([[0.0], [2.0], [6.0]], dtype=np.float32)
    model = ml.AffinityPropagation(
        preference=-1.0, damping=0.5, max_iter=1, convergence_iter=1,
        random_state=13, numeric_mode="identical",
    ).fit(x)
    affinity = np.asarray(model.affinity_matrix_)
    diagnostic = np.asarray(model._message_diag)
    assert affinity.shape == (3, 3) and affinity.dtype == np.dtype("float32")
    assert diagnostic.shape == (6,) and diagnostic.dtype == np.dtype("float32")
    before = diagnostic.copy()
    assert not np.shares_memory(affinity, diagnostic)
    del model
    gc.collect()
    np.testing.assert_array_equal(diagnostic, before)
