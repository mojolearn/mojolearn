# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Resident fitted state of the clustering families (lane
ne1-a-cluster-spectral, plan workstream A; owed rows N024, N021, N326, N172).

Per family: the first query after a fit takes the fitted arrays once (the
binding's handle count rises by one), a second query reuses them (the count
stays), the answer is byte-equal to the per-call route (the same fitted
values assigned as a plain ndarray, which cannot be made read-only and so
keeps the per-call upload), an in-place write to the held array raises
NumPy's read-only error, replacing the attribute takes a new copy, and a
refit releases the old one. GPU bindings only: the CPU host bindings keep
the per-call entries."""
import gc

import numpy as np
import pytest

import mojolearn as ml


def _gpu_binding(name):
    try:
        from mojolearn import _backend
        return _backend.binding(name, "identical")
    except Exception:
        return None


def _blobs(seed=7):
    rng = np.random.default_rng(seed)
    x = np.concatenate([rng.normal(c, 0.4, size=(60, 3)) for c in (0.0, 4.0, 8.0)])
    return np.ascontiguousarray(x, dtype=np.float32)


def _bytes(v):
    return np.asarray(v).tobytes()


@pytest.fixture(scope="module")
def x_cluster():
    b = _gpu_binding("_mojolearn_x_cluster")
    if b is None or not hasattr(b, "x_cluster_centers_count"):
        pytest.skip("no GPU x_cluster binding with the resident doors in this install")
    return b


@pytest.mark.parametrize("make", [
    lambda: ml.MiniBatchKMeans(n_clusters=3, batch_size=64, max_iter=3, random_state=1),
    lambda: ml.BisectingKMeans(n_clusters=3, random_state=1),
    lambda: ml.MeanShift(bandwidth=1.5),
])
def test_x_cluster_centers_resident(x_cluster, make):
    x = _blobs()
    q = np.ascontiguousarray(x[::7] + np.float32(0.1))
    # A model's copy is released lazily: its key goes on the dead-key ring
    # and the next lookup or store drains it (bindings/cluster_resident_table.mojo).
    # So drain the previous case's copy first: a throwaway query (drain, then
    # mint) and a refit (which releases its own copy at once).
    w = make().fit(x)
    _ = w.transform(q) if hasattr(w, "transform") else w.predict(q)
    w.fit(x)
    m = make().fit(x)
    before = x_cluster.x_cluster_centers_count()
    labels = _bytes(m.predict(q)) if not isinstance(m, ml.BisectingKMeans) else None
    dist = _bytes(m.transform(q)) if hasattr(m, "transform") else None
    score = m.score(q) if hasattr(m, "score") else None
    held = x_cluster.x_cluster_centers_count()
    assert held == before + 1
    # A second query reuses the copy.
    if dist is not None:
        assert _bytes(m.transform(q)) == dist
    assert x_cluster.x_cluster_centers_count() == held
    # The held array is read-only.
    with pytest.raises(ValueError):
        np.asarray(m.cluster_centers_)[0, 0] = 0.0
    # The per-call route over the same values gives the same bytes.
    m.cluster_centers_ = np.array(np.asarray(m.cluster_centers_), dtype=np.float32, copy=True)
    if labels is not None:
        assert _bytes(m.predict(q)) == labels
    if dist is not None:
        assert _bytes(m.transform(q)) == dist
    if score is not None:
        assert m.score(q) == score
    # A refit releases the old copy at once; the next query takes one.
    m.fit(x)
    assert x_cluster.x_cluster_centers_count() == held - 1
    if dist is not None:
        _ = m.transform(q)
        assert x_cluster.x_cluster_centers_count() == held
    del m
    gc.collect()


@pytest.fixture(scope="module")
def metrics():
    b = _gpu_binding("_mojolearn_metrics")
    if b is None or not hasattr(b, "spectral_predict_count"):
        pytest.skip("no GPU metrics binding with the resident spectral doors in this install")
    return b


@pytest.mark.parametrize("affinity", ["nearest_neighbors"])
def test_spectral_predict_resident(metrics, affinity):
    x = _blobs(11)
    q = np.ascontiguousarray(x[::5] + np.float32(0.05))
    m = ml.SpectralClustering(n_clusters=3, affinity=affinity, prediction_data=True, random_state=1).fit(x)
    before = metrics.spectral_predict_count()
    first = _bytes(m.predict(q))
    held = metrics.spectral_predict_count()
    assert held == before + 1
    assert _bytes(m.predict(q)) == first
    assert metrics.spectral_predict_count() == held
    with pytest.raises(ValueError):
        np.asarray(m._pd_centroids).reshape(-1)[0] = 0.0
    # The per-call route: the same values as plain ndarrays.
    m._pd_centroids = np.array(np.asarray(m._pd_centroids), copy=True)
    assert _bytes(m.predict(q)) == first
    m.fit(x)
    assert metrics.spectral_predict_count() == held - 1


def test_sequence_resident_doors():
    b = _gpu_binding("_mojolearn_x_sequence")
    if b is None or not hasattr(b, "x_sequence_resident_slots"):
        pytest.skip("no GPU x_sequence binding with the resident doors in this install")
    n = b.x_sequence_resident_slots()
    assert n >= 0
    # Releasing a model that holds nothing is quiet and changes nothing.
    b.x_sequence_resident_release(0)
    assert b.x_sequence_resident_slots() == n
