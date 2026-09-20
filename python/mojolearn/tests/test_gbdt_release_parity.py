"""Numerical regressions for CPU/GPU gradient boosting release paths."""
import numpy as np
import pytest

from mojolearn import GradientBoosting


def _data():
    rng = np.random.default_rng(29)
    x = rng.normal(size=(600, 6)).astype(np.float32)
    y = (x[:, 0] + 0.4 * x[:, 1] > 0).astype(np.float32)
    return x, y


def test_heldout_cursor_starts_at_zero():
    x, y = _data()
    # An unrelated fit changes allocator contents before the held-out fit.
    GradientBoosting(n_estimators=3, loss="RMSE").fit(x, x[:, 2])
    model = GradientBoosting(
        n_estimators=6, max_depth=3, loss="Logloss",
        learning_rate=0.1, bootstrap_type="No", random_strength=0,
        boost_from_average=False, use_best_model=False,
    ).fit(x, y, eval_set=(x, y))
    np.testing.assert_allclose(
        model.test_loss_curve_, model.loss_curve_, rtol=2e-6, atol=1e-7,
    )


@pytest.mark.parametrize("grow_policy", ["Depthwise", "Lossguide"])
def test_cpu_rmse_honors_non_symmetric_grow_policy(grow_policy):
    x, _ = _data()
    model = GradientBoosting(
        n_estimators=2, max_depth=3, loss="RMSE", grow_policy=grow_policy,
    ).fit(x, x[:, 0] + x[:, 1] * x[:, 2])
    assert "\nntree 0 " in str(model.model_)
    assert np.isfinite(np.asarray(model.predict(x))).all()
