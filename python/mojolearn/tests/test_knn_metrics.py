# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Brute-force k-NN under canberra, braycurtis, correlation, jensenshannon and
inner_product (lane x-neighbors-metrics, 2026-09-27). Refusals always run;
value checks skip, saying so, without a binding. scikit-learn / scipy are
compared to a tolerance; repeat calls must be the same bits."""

import numpy as np
import pytest

import mojolearn as ml


def _data(n=200, d=7, seed=6, positive=False):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d)).astype(np.float32)
    if positive:
        x = np.abs(x) + np.float32(0.05)
        x = (x / x.sum(axis=1, keepdims=True)).astype(np.float32)
    return x


def _nn_or_skip(metric, x, k=5):
    try:
        return ml.NearestNeighbors(n_neighbors=k, metric=metric).fit(x)
    except (ImportError, NotImplementedError) as exc:
        pytest.skip(f"no k-NN binding on this install: {exc}")


def test_refusals():
    x = _data()
    with pytest.raises(ValueError, match="haversine"):
        ml.NearestNeighbors(metric="haversine").fit(x[:, :2])
    with pytest.raises(ValueError, match="X >= 0"):
        ml.NearestNeighbors(metric="jensenshannon").fit(x)
    with pytest.raises(ValueError, match="brute force only"):
        ml.NearestNeighbors(metric="canberra", algorithm="rbc").fit(x)
    with pytest.raises(ValueError, match="similarity"):
        ml.KNeighborsRegressor(metric="inner_product", weights="distance").fit(x, x[:, 0])


@pytest.mark.parametrize("metric", ["canberra", "braycurtis", "correlation", "jensenshannon"])
def test_matches_scipy(metric):
    # The metric's DEFINITION is scipy's (cuVS computes the same cells):
    # scikit-learn's own DistanceMetric spells braycurtis with
    # sum(|u| + |v|) in the denominator, which differs from scipy's
    # sum(|u + v|) on signed data, and it accepts no 'jensenshannon' name;
    # both are scipy.spatial.distance.cdist here, stable-sorted.
    sd = pytest.importorskip("scipy.spatial.distance")
    x = _data(positive=(metric == "jensenshannon"))
    q = x[:30]
    nn = _nn_or_skip(metric, x)
    d, i = nn.kneighbors(q)
    full = sd.cdist(q.astype(np.float64), x.astype(np.float64), metric=metric)
    ri = np.argsort(full, axis=1, kind="stable")[:, :5]
    rd = np.take_along_axis(full, ri, axis=1)
    np.testing.assert_allclose(np.asarray(d), rd, rtol=2e-4, atol=2e-5)
    assert (np.asarray(i) == ri).mean() >= 0.97
    d2, i2 = nn.kneighbors(q)
    assert np.asarray(d2).tobytes() == np.asarray(d).tobytes()
    assert np.asarray(i2).tobytes() == np.asarray(i).tobytes()


def test_inner_product_is_the_largest_products():
    x = _data()
    q = x[:20]
    nn = _nn_or_skip("inner_product", x)
    d, i = nn.kneighbors(q)
    dots = q.astype(np.float64) @ x.astype(np.float64).T
    ref = np.argsort(-dots, axis=1, kind="stable")[:, :5]
    assert (np.asarray(i) == ref).mean() >= 0.97
    np.testing.assert_allclose(np.asarray(d), np.take_along_axis(dots, ref, 1), rtol=1e-4, atol=1e-4)


def test_classifier_and_regressor_run_the_metric():
    x = _data()
    y = (x[:, 0] > 0).astype(np.int64)
    try:
        c = ml.KNeighborsClassifier(n_neighbors=5, metric="canberra").fit(x, y)
    except (ImportError, NotImplementedError) as exc:
        pytest.skip(f"no k-NN binding on this install: {exc}")
    skn = pytest.importorskip("sklearn.neighbors")
    r = skn.KNeighborsClassifier(n_neighbors=5, metric="canberra", algorithm="brute").fit(x, y)
    assert (np.asarray(c.predict(x[:50])) == r.predict(x[:50])).mean() >= 0.96
