# SPDX-License-Identifier: Apache-2.0
"""mojolearn.Leiden at the Python API boundary (lane leiden-umap-1, 2026-11-28): the input forms it admits, its
refusals, its parameters and the labels and attributes it returns. The kernels' numerical checks are native
(checks/native_tests/x_neighbors_leiden_check.mojo). Also the UMAP min_dist default (0.5, Scanpy's)."""
import numpy as np
import pytest

import mojolearn


def _two_triangles():
    a = np.zeros((6, 6), np.float32)
    for i, j in [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5), (2, 3)]:
        a[i, j] = a[j, i] = 1.0
    return a


def _blocks(n=60, k=3, seed=0):
    """k dense blocks with a few edges between them, weights 1..3."""
    rng = np.random.default_rng(seed)
    block = np.arange(n) * k // n
    p = np.where(block[:, None] == block[None, :], 0.5, 0.02)
    m = np.triu(rng.random((n, n)) < p, 1)
    w = (1 + rng.integers(0, 3, (n, n))).astype(np.float32)
    a = np.where(m, w, 0).astype(np.float32)
    return a + a.T


def _connected(a, labels):
    for c in np.unique(labels):
        members = np.flatnonzero(labels == c)
        seen, stack = {members[0]}, [members[0]]
        while stack:
            u = stack.pop()
            for v in members:
                if v not in seen and a[u, v] != 0:
                    seen.add(v)
                    stack.append(v)
        if len(seen) != len(members):
            return False
    return True


def test_defaults_and_params():
    m = mojolearn.Leiden()
    assert m.get_params()["resolution"] == 1.0
    assert m.get_params()["objective_function"] == "modularity"
    assert m.get_params()["n_iterations"] == -1
    assert m.get_params()["random_state"] == 0
    assert "Leiden" in mojolearn.__all__ or hasattr(mojolearn, "Leiden")


def test_two_triangles_labels_and_attributes():
    m = mojolearn.Leiden().fit(_two_triangles())
    labels = np.asarray(m.labels_)
    assert labels.dtype == np.int32 and labels.shape == (6,)
    assert labels.tolist() == [0, 0, 0, 1, 1, 1]
    assert m.n_communities_ == 2
    assert isinstance(m.quality_, float)
    assert abs(m.quality_ - 2 * (3 / 7 - 0.25)) < 1e-5
    assert isinstance(m.n_iterations_, int) and m.n_iterations_ >= 1


def test_fit_predict_returns_labels():
    a = _blocks()
    m = mojolearn.Leiden(random_state=3)
    got = np.asarray(m.fit_predict(a))
    assert np.array_equal(got, np.asarray(m.labels_))
    assert got.min() == 0 and got.max() == m.n_communities_ - 1
    assert _connected(a, got)


@pytest.mark.parametrize("objective", ["modularity", "CPM", "cpm"])
def test_input_forms_agree(objective):
    sp = pytest.importorskip("scipy.sparse")
    a = _blocks()
    params = dict(resolution=0.5 if objective != "modularity" else 1.0, objective_function=objective)
    dense = np.asarray(mojolearn.Leiden(**params).fit_predict(a))
    sparse = np.asarray(mojolearn.Leiden(**params).fit_predict(sp.csr_matrix(a)))
    assert np.array_equal(dense, sparse)
    assert _connected(a, dense)
    # the (indptr, indices, n) tuple: every edge weighs 1
    b = (a != 0).astype(np.float32)
    csr = sp.csr_matrix(b)
    from_tuple = np.asarray(mojolearn.Leiden(**params).fit_predict(
        (csr.indptr.astype(np.int32), csr.indices.astype(np.int32), b.shape[0])))
    assert np.array_equal(from_tuple, np.asarray(mojolearn.Leiden(**params).fit_predict(b)))


def test_reproducible_for_a_random_state():
    a = _blocks(seed=4)
    one = np.asarray(mojolearn.Leiden(random_state=11).fit_predict(a))
    two = np.asarray(mojolearn.Leiden(random_state=11).fit_predict(a))
    assert np.array_equal(one, two)


def test_n_iterations_bounds_the_run():
    m = mojolearn.Leiden(n_iterations=1).fit(_blocks())
    assert m.n_iterations_ == 1
    stable = mojolearn.Leiden(n_iterations=-1).fit(_blocks())
    assert stable.quality_ >= m.quality_


def test_cpm_resolution_splits():
    a = _blocks()
    coarse = mojolearn.Leiden(resolution=0.05, objective_function="CPM").fit(a)
    fine = mojolearn.Leiden(resolution=2.0, objective_function="CPM").fit(a)
    assert fine.n_communities_ >= coarse.n_communities_


@pytest.mark.parametrize("params, message", [
    (dict(objective_function="surprise"), "objective_function must be 'modularity' or 'CPM'"),
    (dict(n_iterations=0), "n_iterations must be a positive integer"),
])
def test_parameter_refusals(params, message):
    with pytest.raises(ValueError, match=message):
        mojolearn.Leiden(**params).fit(_two_triangles())


def test_graph_refusals():
    a = _two_triangles()
    a[0, 1] = 2.0
    with pytest.raises(ValueError, match="symmetric"):
        mojolearn.Leiden().fit(a)
    with pytest.raises(ValueError, match="no edges"):
        mojolearn.Leiden().fit(np.zeros((4, 4), np.float32))
    with pytest.raises(ValueError):
        mojolearn.Leiden().fit(np.zeros((3, 4), np.float32))


def test_umap_min_dist_default_is_scanpys():
    assert mojolearn.UMAP().min_dist == 0.5
    assert mojolearn.UMAP().get_params()["min_dist"] == 0.5
