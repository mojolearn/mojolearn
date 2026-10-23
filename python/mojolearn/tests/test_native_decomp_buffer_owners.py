# SPDX-License-Identifier: Apache-2.0
"""Public decomposition paths keep scratch alive and accept native buffers."""

from array import array
import math

import numpy as np
import pytest

import mojolearn as ml


@pytest.mark.parametrize("algorithm", ["lle", "umap_pca"])
def test_manifold_svd_keeps_resident_sort_scratch(algorithm):
    # Both public fits sort singular values through the native manifold
    # kit. On a GPU its count output is a pooled device buffer, even though
    # the caller only consumes the sorted indices. Repeat in one process
    # so pool reuse cannot conceal a lost owner.
    x = np.random.default_rng(51).normal(size=(64, 5)).astype(np.float32)
    before = x.copy()
    first = None
    for _ in range(2):
        if algorithm == "lle":
            model = ml.LocallyLinearEmbedding(
                n_neighbors=8, n_components=2, numeric_mode="identical"
            )
        else:
            model = ml.UMAP(
                n_neighbors=8, n_components=2, n_epochs=8, init="pca",
                random_state=3, numeric_mode="identical",
            )
        model.fit(x)
        result = np.asarray(model.embedding_)
        assert result.shape == (64, 2)
        assert np.isfinite(result).all()
        if first is None:
            first = result.copy()
        else:
            np.testing.assert_array_equal(result.view(np.uint32), first.view(np.uint32))
    np.testing.assert_array_equal(x, before)


@pytest.fixture(scope="module")
def elliptic_case():
    x = np.random.default_rng(52).normal(size=(96, 4)).astype(np.float32)
    model = ml.EllipticEnvelope(
        contamination=.15, random_state=1, support_fraction=.7,
        numeric_mode="identical",
    ).fit(x)
    prediction = np.asarray(model.predict(x)).copy()
    labels = prediction.copy()
    labels[::3] *= -1
    return model, x, labels, prediction


@pytest.mark.parametrize("storage", ["ndarray", "strided", "memoryview", "array", "float64"])
def test_elliptic_score_copies_admitted_weight_storage(elliptic_case, storage):
    model, x, labels, prediction = elliptic_case
    values = np.linspace(.25, 2, len(x), dtype=np.float32)
    if storage == "strided":
        backing = np.zeros(2 * len(values), dtype=np.float32)
        backing[::2] = values
        weights = backing[::2]
    elif storage == "memoryview":
        weights = memoryview(values)
    elif storage == "array":
        weights = array("f", values)
    elif storage == "float64":
        weights = values.astype(np.float64)
    else:
        weights = values.copy()
        weights.flags.writeable = False
    # The oracle consumes the completed prediction in test code. Weights
    # are exactly float32 values, including the float64 conversion case.
    expected = math.fsum(float(v) for v in values[prediction == labels]) / math.fsum(map(float, values))
    for _ in range(2):
        actual = model.score(x, labels, sample_weight=weights)
        assert actual == pytest.approx(expected, rel=0, abs=1e-15)
    np.testing.assert_array_equal(np.asarray(weights), values)
