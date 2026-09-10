"""Minimum split gain validation and backward-compatible binding tail."""
import numpy as np
import pytest
from mojolearn.ensemble import GradientBoosting


@pytest.mark.parametrize("value", [-1, -0.1, np.nan, np.inf, -np.inf, True,
                                   "1", [1], 10**400])
def test_bad_gain_rejected(value):
    with pytest.raises(ValueError, match="min_split_gain"):
        GradientBoosting(grow_policy="Lossguide", min_split_gain=value)


def test_symmetric_gain_rejected():
    with pytest.raises(ValueError, match="Depthwise and Lossguide"):
        GradientBoosting(min_split_gain=0)


@pytest.mark.parametrize("policy", ["Depthwise", "Lossguide"])
@pytest.mark.parametrize("weights", [None, [1.25, 2.5]])
def test_optional_tail_preserves_counted_class_weights(policy, weights):
    before = GradientBoosting(loss="Logloss", grow_policy=policy,
                              class_weights=weights)._params(8, 2, 0)
    after = GradientBoosting(loss="Logloss", grow_policy=policy,
                             class_weights=weights,
                             min_split_gain=2.125)._params(8, 2, 0)
    assert len(before) == 35 + len(weights or [])
    assert before[34] == len(weights or [])
    assert before[35:] == (weights or [])
    assert after[:-1] == before
    assert after[-1] == 2.125


@pytest.mark.parametrize("value", [0, np.float32(0.5), np.float64(1e100)])
def test_finite_nonnegative_gain_accepted(value):
    model = GradientBoosting(grow_policy="Depthwise", min_split_gain=value)
    assert model.min_split_gain == float(value)
