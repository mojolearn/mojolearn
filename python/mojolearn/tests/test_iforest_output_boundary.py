"""Isolation Forest outputs through the real public path (small inputs):
each method returns its own result kind, and they agree."""
import numpy as np
import pytest

from mojolearn import IsolationForest


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(7)
    X = rng.random((32, 2), dtype=np.float32)
    return IsolationForest(n_estimators=2, max_samples=4, random_state=7).fit(X), X[:3]


@pytest.mark.parametrize("method", ["score_samples", "decision_function", "predict"])
def test_iforest_method_outputs(fitted, method):
    model, X = fitted
    result = np.asarray(getattr(model, method)(X))
    assert result.shape == (3,)
    if method == "predict":
        assert set(result.tolist()) <= {-1, 1}
    else:
        assert np.issubdtype(result.dtype, np.floating) and np.all(np.isfinite(result))


def test_iforest_predict_is_the_sign_of_decision_function(fitted):
    model, X = fitted
    decision = np.asarray(model.decision_function(X))
    np.testing.assert_array_equal(np.asarray(model.predict(X)), np.where(decision < 0, -1, 1))
