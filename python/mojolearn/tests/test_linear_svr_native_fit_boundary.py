"""LinearSVR fitted storage, publication and native factory seams."""
import gc
import pickle

import numpy as np
import pytest

from mojolearn.svm import LinearSVR

__mojolearn_user_boundary__ = True


def _points():
    return np.array([[-1, 0], [1, 0]] * 4, dtype=np.float32)


def _model(**options):
    model = LinearSVR(C=2, epsilon=0.1, loss="squared_epsilon_insensitive",
                      tol=1e-6, max_iter=200, numeric_mode="identical", **options)
    model.penalty_normalized = False
    return model


def test_linear_svr_public_storage_and_scalar_metadata_survive_native_plan_release():
    model = _model(fit_intercept=False).fit(_points(), [-1, 1] * 4)
    assert isinstance(model.objective_, float)
    assert isinstance(model.retcode_, int)
    assert np.asarray(model.n_iter_).dtype == np.dtype("int64")
    assert np.asarray(model.n_iter_).shape == (1,)
    assert model.n_features_in_ == 2
    coefficients = np.asarray(model.coef_)
    assert not np.shares_memory(coefficients, np.asarray(model._w))
    restored = pickle.loads(pickle.dumps(model))
    del model
    gc.collect()
    np.testing.assert_allclose(coefficients, [0.72, 0], rtol=0, atol=2e-4)
    np.testing.assert_allclose(restored.predict(_points()), [-0.72, 0.72] * 4,
                               rtol=0, atol=2e-4)


def test_linear_svr_weighted_intercept_and_metadata_keep_original_values():
    model = _model().fit(_points(), [-1, 1] * 4, sample_weight=[1, 3] * 4)
    np.testing.assert_allclose(model.coef_, [27 / 35, 0], rtol=0, atol=2e-4)
    assert abs(model.intercept_ - 9 / 140) < 2e-4
    assert abs(model.objective_ - 243 / 1400) < 2e-4


def test_linear_svr_target_refusal_retains_input_publication_order():
    model = _model()
    with pytest.raises(ValueError, match="X and y lengths differ"):
        model.fit(_points(), [1])
    assert "input_copied_" in vars(model)
    assert "_w" not in vars(model)
    assert "objective_" not in vars(model)
    assert "coef_" not in vars(model)


def test_linear_svr_native_factory_replacement_refuses_without_callback(monkeypatch):
    model = _model()
    calls = []

    def forbidden():
        calls.append(True)
        raise AssertionError("Python QN provider invoked")

    monkeypatch.setattr(model._bind(), "linear_svc_native_api_v1", forbidden)
    with pytest.raises(TypeError, match="compiled extension entry"):
        model.fit(_points(), [-1, 1] * 4)
    assert calls == []
    assert "input_copied_" in vars(model)
    assert "objective_" not in vars(model)
    assert "_w" not in vars(model)
