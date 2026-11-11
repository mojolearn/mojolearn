"""Retained public ownership/publication seams for the typed cluster variants.

Run against the installed compiled candidate. Numerical kernel correctness is
covered by the independent native provider fixtures, not duplicated here.
"""
import gc
import pickle

import numpy as np
import pytest

import mojolearn as ml

__mojolearn_user_boundary__ = True


def test_minibatch_failed_first_partial_preserves_early_publication_stage():
    model = ml.MiniBatchKMeans(
        n_clusters=2, init=np.array([[0.0], [10.0]], dtype=np.float32),
        batch_size=4, random_state=7, reassignment_ratio="bad-ratio",
        numeric_mode="identical",
    )
    with pytest.raises(ValueError, match="bad-ratio"):
        model.partial_fit(np.array([[0.0], [1.0], [10.0], [11.0]], dtype=np.float32))
    fields = vars(model)
    assert fields["_partial_batch"] == 4
    assert fields["_partial_init"] == (2, 4)
    assert fields["n_steps_"] == 0
    assert "cluster_centers_" not in fields
    assert "_partial_state" not in fields


@pytest.mark.parametrize("name", ["MiniBatchKMeans", "BisectingKMeans"])
def test_variant_float_refusal_precedes_native_integer_narrowing(name):
    model = getattr(ml, name)(
        n_clusters=2, max_iter=1 << 100, tol="bad-tolerance",
        random_state=7, numeric_mode="identical",
    )
    with pytest.raises(ValueError, match="bad-tolerance"):
        model.fit(np.array([[0.0], [1.0], [10.0], [11.0]], dtype=np.float32))
    assert "cluster_centers_" not in vars(model)


def test_minibatch_state_forms_and_rebound_stream_survive_pickle():
    x = np.array([[0.0], [1.0], [10.0], [11.0]], dtype=np.float32)
    model = ml.MiniBatchKMeans(
        n_clusters=2, init=np.array([[0.0], [10.0]], dtype=np.float32),
        batch_size=4, random_state=7, reassignment_ratio=0,
        numeric_mode="identical",
    ).partial_fit(x)
    fields = vars(model)
    assert type(fields["_partial_batch"]) is int
    assert type(fields["_partial_init"]) is tuple
    assert type(fields["_partial_state"]) is tuple
    assert type(fields["_partial_state"][0]) is tuple
    # Existing private attributes remain deliberately rebindable. A restored
    # instance must consume these values, not a stale internal cached stream.
    model._partial_state = ((0, 29), 0)
    restored = pickle.loads(pickle.dumps(model))
    assert restored._partial_state == ((0, 29), 0)
    assert restored.partial_fit(x) is restored
    assert vars(restored)["n_steps_"] == 2
    assert type(restored._partial_state) is tuple


def test_bisect_tree_arrays_keep_identity_aliases_and_serialization():
    x = np.array([[0.0], [1.0], [10.0], [11.0]], dtype=np.float32)
    model = ml.BisectingKMeans(n_clusters=2, random_state=7, numeric_mode="identical").fit(x)
    centers = model._tree_centers
    nodes = model._tree_nodes
    assert centers is model._tree_centers
    assert nodes is model._tree_nodes
    before = np.asarray(model.predict(x)).copy()
    view = np.asarray(nodes)
    # Swap only the two leaf labels in the observable existing tree storage.
    view[1, 2], view[2, 2] = view[2, 2], view[1, 2]
    np.testing.assert_array_equal(model.predict(x), 1 - before)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.predict(x), 1 - before)
    raw = nodes._mv
    expected = raw.tobytes()
    del restored, model, centers, nodes, view
    gc.collect()
    assert raw.tobytes() == expected
