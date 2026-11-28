# SPDX-License-Identifier: Apache-2.0
"""The public boundary of `mojolearn.knn_graph` (lane sc-ann-1): shapes and
dtypes, the exact graph against a NumPy brute force (an external oracle),
IVF with every list probed equal to the exact graph, recall of the default
IVF graph, the refusals, and one native call."""

import numpy as np
import pytest

import mojolearn as ml


_RNG = np.random.default_rng(11)
X = _RNG.standard_normal((600, 6)).astype(np.float32)


def _call(*args, **kw):
    try:
        return ml.knn_graph(*args, **kw)
    except (ImportError, ModuleNotFoundError) as error:  # noqa: BLE001 - a binding not built here
        pytest.skip(f"compiled binding unavailable: {error}")


def _brute(x, k):
    # independent oracle: float64 squared distances, self excluded, ties by id
    d = ((x[:, None, :].astype(np.float64) - x[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d, np.inf)
    return np.argsort(d, axis=1, kind="stable")[:, :k]


def test_exact_shapes_and_neighbors():
    dist, idx = _call(X, 8)
    dist, idx = np.asarray(dist), np.asarray(idx)
    assert dist.shape == (600, 8) and idx.shape == (600, 8)
    assert dist.dtype == np.float32 and idx.dtype == np.int32
    assert not (idx == np.arange(600)[:, None]).any()
    assert (np.diff(dist, axis=1) >= 0).all()
    # same neighbor sets as the float64 oracle on generic data
    want = _brute(X, 8)
    agree = np.mean([len(set(a) & set(b)) / 8 for a, b in zip(idx, want)])
    assert agree > 0.99


def test_euclidean_is_root_of_sqeuclidean():
    d2, i2 = _call(X, 5)
    d1, i1 = _call(X, 5, metric="euclidean")
    assert np.array_equal(np.asarray(i1), np.asarray(i2))
    np.testing.assert_allclose(np.asarray(d1), np.sqrt(np.asarray(d2)), rtol=1e-6)


def test_ivf_all_probes_equals_exact():
    de, ie = _call(X, 7, method="exact", numeric_mode="identical")
    da, ia = _call(X, 7, method="ivf", n_lists=6, n_probes=6, random_state=4, numeric_mode="identical")
    assert np.array_equal(np.asarray(ie), np.asarray(ia))
    assert np.asarray(de).tobytes() == np.asarray(da).tobytes()


def test_ivf_recall_on_clusters():
    centers = np.repeat(np.arange(10, dtype=np.float32)[:, None] * 50.0, 6, axis=1)
    x = (np.repeat(centers, 300, axis=0) + _RNG.standard_normal((3000, 6)).astype(np.float32))
    _, ia = _call(x, 10, method="ivf", n_lists=10, n_probes=3, random_state=0)
    want = _brute(x, 10)
    recall = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(np.asarray(ia), want)])
    assert recall >= 0.9


def test_ivf_default_small_input_is_exact():
    de, ie = _call(X, 5, method="exact")
    da, ia = _call(X, 5, method="ivf")  # 600 rows: the default plan builds the exact graph
    assert np.array_equal(np.asarray(ie), np.asarray(ia))


@pytest.mark.parametrize("kw,err", [
    (dict(n_neighbors=0), ValueError),
    (dict(n_neighbors=600), ValueError),
    (dict(n_neighbors=2.0), TypeError),
    (dict(n_neighbors=True), TypeError),
    (dict(method="hnsw"), ValueError),
    (dict(metric="cosine"), ValueError),
    (dict(method="ivf", n_lists=0), ValueError),
    (dict(method="ivf", n_lists=601), ValueError),
    (dict(method="ivf", n_lists=4, n_probes=5), ValueError),
    (dict(method="ivf", n_probes=0), ValueError),
    (dict(method="ivf", random_state=-1), ValueError),
    (dict(method="ivf", random_state="a"), TypeError),
])
def test_refusals(kw, err):
    _call(X, 3)
    args = dict(n_neighbors=3)
    args.update(kw)
    with pytest.raises(err):
        ml.knn_graph(X, **args)


def test_non_finite_refused():
    _call(X, 3)
    bad = X.copy()
    bad[3, 2] = np.nan
    with pytest.raises(ValueError):
        ml.knn_graph(bad, 3)


def test_is_one_native_call():
    from mojolearn._verify_native_runtime import audit_native_call
    _call(X, 3)
    audit_native_call(ml.knn_graph, X, 3).assert_native()
