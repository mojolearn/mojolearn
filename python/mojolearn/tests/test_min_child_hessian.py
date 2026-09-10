"""Public child-curvature validation and compatibility of the native ABI tail.

GPU split correctness belongs to checks/min_child_hessian_check.mojo. These
tests protect the independent Python input and counted-class-weight boundary.
"""
import numpy as np
import pytest

from mojolearn.ensemble import GradientBoosting


def _model(**kwargs):
    options = dict(grow_policy="Lossguide", score_function="NewtonL2")
    options.update(kwargs)
    return GradientBoosting(**options)


@pytest.mark.parametrize("value", [
    -1, -0.01, np.nan, np.inf, -np.inf, True, np.bool_(False),
    "1", [1], 1 + 0j, 10**400,
    np.nextafter(float(np.finfo(np.float32).max), np.inf),
])
def test_invalid_child_hessian_rejected(value):
    with pytest.raises(ValueError, match="min_child_hessian"):
        _model(min_child_hessian=value)


@pytest.mark.parametrize("value", [
    0, np.int64(1), np.float32(0.5), np.float64(1.00000001),
    np.nextafter(0.0, 1.0), float(np.finfo(np.float32).max),
])
def test_bound_is_preserved_as_float64_until_native_validation(value):
    model = _model(min_child_hessian=value)
    assert model.min_child_hessian == float(value)
    assert model._params(8, 2, 0)[-2:] == [-1.0, float(value)]


@pytest.mark.parametrize("policy", ["Depthwise", "Lossguide"])
@pytest.mark.parametrize("score", ["NewtonL2", "NewtonCosine"])
@pytest.mark.parametrize("loss", ["RMSE", "Logloss", "CrossEntropy"])
def test_supported_curvature_profiles(policy, score, loss):
    model = _model(grow_policy=policy, score_function=score, loss=loss,
                   min_child_hessian=0.25)
    assert model._params(8, 2, 0)[-1] == 0.25


@pytest.mark.parametrize("score", ["L2", "Cosine"])
def test_first_order_weights_are_not_exposed_as_hessians(score):
    with pytest.raises(ValueError, match="NewtonL2 or NewtonCosine"):
        _model(score_function=score, min_child_hessian=0)


def test_symmetric_policy_refuses_enabled_guard():
    with pytest.raises(ValueError, match="Depthwise or Lossguide"):
        _model(grow_policy="SymmetricTree", min_child_hessian=0)


@pytest.mark.parametrize("loss", ["Poisson", "MAE", "MAPE"])
def test_unaudited_objective_curvature_refused(loss):
    with pytest.raises(ValueError, match="RMSE, Logloss and CrossEntropy"):
        _model(loss=loss, min_child_hessian=0)


@pytest.mark.parametrize("weights", [None, [1.25, 2.5]])
@pytest.mark.parametrize("gain", [None, 2.125])
def test_counted_class_weights_and_existing_gain_tail_remain_compatible(weights, gain):
    base = dict(loss="Logloss", class_weights=weights, min_split_gain=gain)
    before = _model(**base)._params(8, 2, 0)
    explicit_off = _model(**base, min_child_hessian=None)._params(8, 2, 0)
    after = _model(**base, min_child_hessian=0.125)._params(8, 2, 0)
    fixed = 35 + len(weights or [])
    assert explicit_off == before
    assert len(before) == fixed + (gain is not None)
    assert before[34] == len(weights or [])
    assert before[35:fixed] == (weights or [])
    assert after[:fixed] == before[:fixed]
    assert after[fixed:] == [-1.0 if gain is None else gain, 0.125]


@pytest.mark.parametrize("mode", ["fast", "deterministic", "identical"])
def test_guard_retains_requested_numeric_mode(mode):
    model = _model(min_child_hessian=0.25, numeric_mode=mode)
    assert model.numeric_mode == mode
