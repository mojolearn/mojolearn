"""predict_classes through the real public path (small inputs): whichever
route serves the codes (the FAST resident class output or the probability
path), they are the first-max class code of `predict_proba`."""
import numpy as np
import pytest

from mojolearn import GradientBoosting, _backend


def _fit(mode, loss, k):
    if _backend._CPU_ONLY is not None and mode == "fast":
        pytest.skip("CPU-only install: every host binding is identical only (_backend._cpu_only_binding refuses other tiers by name)")
    rng = np.random.default_rng(0)
    X = rng.random((40, 4), dtype=np.float32)
    y = np.minimum((X[:, 0] * k).astype(np.int32), k - 1).astype(np.float32)
    model = GradientBoosting(loss=loss, n_estimators=4, max_depth=2, bootstrap_type="No",
                             random_strength=0.0, numeric_mode=mode)
    try:
        model.fit(X, y)
    except ImportError as error:
        pytest.skip(f"this install has no {mode} GBDT binding: {error}")
    return model, X


@pytest.mark.parametrize("mode", ["identical", "fast"])
@pytest.mark.parametrize("loss,k", [("Logloss", 2), ("MultiClass", 3), ("MultiClass", 4)])
def test_predict_classes_is_the_argmax_of_predict_proba(mode, loss, k):
    model, X = _fit(mode, loss, k)
    proba = np.asarray(model.predict_proba(X))
    assert proba.shape == (X.shape[0], k)
    np.testing.assert_array_equal(np.asarray(model.predict_classes(X)), np.argmax(proba, axis=1))
