"""DART binary probabilities through the real public path (small inputs):
the (n, 2) rows a binary fit predicts."""
import numpy as np

from mojolearn import _expansion_trees as trees


def test_binary_probability_rows():
    rng = np.random.default_rng(0)
    X = rng.random((40, 3), dtype=np.float32)
    y = (X[:, 0] > 0.5).astype(np.int32)
    model = trees.DARTClassifier(n_estimators=3, random_state=0).fit(X, y)
    proba = np.asarray(model.predict_proba(X[:19]))
    assert proba.shape == (19, 2)
    assert np.all(np.isfinite(proba)) and np.all(proba >= 0)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=0, atol=1e-6)
