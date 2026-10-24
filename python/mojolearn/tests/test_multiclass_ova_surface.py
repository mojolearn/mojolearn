# SPDX-License-Identifier: Apache-2.0
"""OVA boundary and serialization checks through the real public fit (small inputs).

DEVIATION 2460: predictions are `mojolearn.Array`; dtype, row sums and the
uint32 bit comparisons go through `np.asarray` (zero-copy), unchanged otherwise.
"""
from unittest.mock import patch

import numpy as np
import pytest

from mojolearn import _backend
from mojolearn.ensemble import GradientBoosting


def _ova(classes, **fit_kw):
    rng = np.random.default_rng(0)
    X = rng.random((30, 2), dtype=np.float32)
    y = np.arange(30) % classes
    class_weights = [float(k + 1) for k in range(classes)]
    model = GradientBoosting(loss="MultiClassOneVsAll", n_estimators=3, max_depth=2,
                             bootstrap_type="No", random_strength=0.0,
                             class_weights=class_weights, numeric_mode="identical")
    return model.fit(X, y, **fit_kw), X


@pytest.mark.parametrize("classes", [2, 3])
def test_ova_dimensions_link_weights_and_saved_mode(classes, tmp_path):
    if _backend._CPU_ONLY is not None:
        pytest.skip("CPU-only install: the gbdt host binding trains its declared lanes with unit weights only and refuses sample_weight by name (bindings/_mojolearn_gbdt_host.mojo); weighted GBDT fits need a GPU binding")
    weights = np.arange(1, 31, dtype=np.float32) / 2
    model, X = _ova(classes, sample_weight=weights)
    assert model.n_classes_ == model.approx_dim_ == classes
    raw, probability = model.predict(X), model.predict_proba(X)
    assert raw.shape == probability.shape == (30, classes)
    assert np.asarray(raw).dtype == np.asarray(probability).dtype == np.float32
    # one-vs-all: each class's probability is the sigmoid of its own raw
    # score, never a softmax across classes
    sigmoid = 1.0 / (1.0 + np.exp(-np.asarray(raw, dtype=np.float64)))
    np.testing.assert_allclose(np.asarray(probability), sigmoid, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(model.predict_classes(X)),
                                  np.argmax(np.asarray(probability), axis=1))
    unweighted, _ = _ova(classes)
    assert np.asarray(unweighted.predict(X)).tobytes() != np.asarray(raw).tobytes()
    path = tmp_path / "ova.npz"
    model.save(path)
    with patch.object(_backend, "_DEFAULT_MODE", "fast"):
        restored = GradientBoosting.load(path)
        assert restored.loss == "MultiClassOneVsAll"
        assert restored.numeric_mode == "identical"
        assert restored.n_classes_ == restored.approx_dim_ == classes
        np.testing.assert_array_equal(np.asarray(restored.predict(X)).view(np.uint32),
                                      np.asarray(raw).view(np.uint32))
        np.testing.assert_array_equal(np.asarray(restored.predict_proba(X)).view(np.uint32),
                                      np.asarray(probability).view(np.uint32))


@pytest.mark.parametrize("labels", [
    [-1, 0, 1], [0, 0.5, 1], [0, np.nan, 1], [0, np.inf, 1], [0, 0, 0],
])
def test_unsafe_ova_labels_refused(labels):
    model = GradientBoosting(loss="MultiClassOneVsAll", class_weights=[1, 2])
    with pytest.raises(ValueError, match="class"):
        model.fit(np.zeros((3, 2), np.float32), labels)


@pytest.mark.parametrize("policy", ["Depthwise", "Lossguide"])
def test_ova_remains_symmetric_only(policy):
    with pytest.raises(NotImplementedError, match="optimization scheme"):
        GradientBoosting(loss="MultiClassOneVsAll", grow_policy=policy)
