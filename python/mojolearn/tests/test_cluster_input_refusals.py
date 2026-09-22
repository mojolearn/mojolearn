# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Refusals found by the pip-install smoke of the cluster family (2026-09-22).

Before these checks, from a plain `pip install mojolearn==0.8.13`:

  * `KMeans.fit` on an X holding one NaN returned a NaN centroid, an
    `inertia_` of FLT_MAX and label -1 for that row, silently; `predict` and
    `transform` returned -1 / garbage for a NaN query.
  * `DBSCAN.fit` labelled a NaN row noise (-1), silently.
  * `KernelDensity.fit` accepted a NaN and refused it only at the first
    `score_samples` (DEVIATION 604); it is now refused at fit as well.
  * `Embedding.forward([1.5])` read row 1: a float id was cast to int32.

scikit-learn refuses the first two (validate_data) and torch the third
(a float index tensor). Each is now refused BY NAME before any launch.

    cd python && python3 -m mojolearn.tests.test_cluster_input_refusals
"""
import sys

import numpy as np

from mojolearn import DBSCAN, Embedding, KernelDensity, KMeans


def _x():
    rng = np.random.default_rng(0)
    return rng.normal(size=(64, 3)).astype(np.float32)


def _raises(exc, needle, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc as e:
        assert needle in str(e), str(e)
        return
    raise AssertionError("ACCEPTED: %r" % (fn,))


def test_kmeans_refuses_non_finite():
    x = _x()
    xn = x.copy()
    xn[5, 1] = np.nan
    xi = x.copy()
    xi[7, 0] = np.inf
    _raises(ValueError, "X contains a NaN or an infinity", KMeans(3).fit, xn)
    _raises(ValueError, "X contains a NaN or an infinity", KMeans(3).fit, xi)
    _raises(ValueError, "init_centroids contains a NaN", KMeans(3, init="array", init_centroids=xn[4:7]).fit, x)
    w = np.ones(64)
    w[0] = np.nan
    _raises(ValueError, "sample_weight contains a NaN", KMeans(3).fit, x, sample_weight=w)
    m = KMeans(3, random_state=0).fit(x)
    _raises(ValueError, "X contains a NaN or an infinity", m.predict, xn)
    _raises(ValueError, "X contains a NaN or an infinity", m.transform, xi)
    # the finite path is untouched
    assert np.isfinite(float(m.inertia_))
    assert np.array_equal(np.asarray(m.predict(x)), np.asarray(m.labels_))


def test_dbscan_refuses_non_finite():
    x = _x()
    xn = x.copy()
    xn[3, 2] = np.nan
    _raises(ValueError, "X contains a NaN or an infinity", DBSCAN(eps=1.0).fit, xn)
    xi = x.copy()
    xi[3, 2] = -np.inf
    _raises(ValueError, "X contains a NaN or an infinity", DBSCAN(eps=1.0).fit, xi)
    assert np.asarray(DBSCAN(eps=1.0).fit(x).labels_).shape == (64,)


def test_kde_refuses_non_finite_at_fit():
    x = _x()
    xn = x.copy()
    xn[0, 0] = np.nan
    _raises(ValueError, "X contains a NaN or an infinity", KernelDensity().fit, xn)
    assert np.isfinite(np.asarray(KernelDensity().fit(x).score_samples(x))).all()


def test_embedding_refuses_float_ids():
    w = np.arange(20 * 4, dtype=np.float32).reshape(20, 4)
    e = Embedding(20, 4, weight=w)
    _raises(TypeError, "ids must be integers", e.forward, [1.5])
    _raises(TypeError, "ids must be integers", e.forward, np.array([1.0, 2.0]))
    _raises(TypeError, "ids must be integers", e.backward, np.array([1.0]), np.ones((1, 4), np.float32))
    assert np.array_equal(np.asarray(e.forward([1, 2])), w[[1, 2]])
    assert np.array_equal(np.asarray(e.forward(np.array([3], np.int64))), w[[3]])


def main():
    for fn in (test_kmeans_refuses_non_finite, test_dbscan_refuses_non_finite, test_kde_refuses_non_finite_at_fit, test_embedding_refuses_float_ids):
        fn()
        print("ok", fn.__name__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
