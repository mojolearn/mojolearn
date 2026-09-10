"""Host refusal and ABI checks: no GPU or cross-device qualification claim."""
import numpy as np
import pytest
from mojolearn import GradientBoosting

MODES = ['fast', 'deterministic', 'identical']


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf, -1., True, '1', None,
    1+0j, 10**400, np.nextafter(float(np.finfo(np.float32).max), np.inf)])
def test_invalid_noise_at_constructor_and_packing(mode, value):
    with pytest.raises(ValueError, match='random_strength must be finite'):
        GradientBoosting(random_strength=value, numeric_mode=mode)
    model = GradientBoosting(numeric_mode=mode)
    model.random_strength = value
    with pytest.raises(ValueError, match='random_strength must be finite'):
        model._params(32, 4, 0)


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('score', ['L2', 'NewtonL2'])
def test_noise_score_conflict_rechecked_after_mutation(mode, score):
    with pytest.raises(ValueError, match='does nothing'):
        GradientBoosting(score_function=score, random_strength=1., numeric_mode=mode)
    model = GradientBoosting(score_function=score, numeric_mode=mode)
    model.random_strength = 1.
    with pytest.raises(ValueError, match='does nothing'):
        model._params(32, 4, 0)
    model = GradientBoosting(random_strength=1., numeric_mode=mode)
    model.score_function = score
    with pytest.raises(ValueError, match='does nothing'):
        model._params(32, 4, 0)


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('policy', ['Depthwise', 'Lossguide'])
def test_pointwise_policy_conflict_rechecked_after_mutation(mode, policy):
    with pytest.raises(ValueError, match='OBLIVIOUS'):
        GradientBoosting(grow_policy=policy, use_pointwise_searcher=True, numeric_mode=mode)
    model = GradientBoosting(grow_policy=policy, numeric_mode=mode)
    model.use_pointwise_searcher = True
    with pytest.raises(ValueError, match='OBLIVIOUS'):
        model._params(32, 4, 0)
    model = GradientBoosting(use_pointwise_searcher=True, numeric_mode=mode)
    model.grow_policy = policy
    with pytest.raises(ValueError, match='OBLIVIOUS'):
        model._params(32, 4, 0)


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('pointwise', [False, True])
@pytest.mark.parametrize('strength', [0., -0., 1., np.float32(.5), float(np.finfo(np.float32).max)])
def test_supported_noise_and_pointwise_pack_in_every_mode(mode, pointwise, strength):
    model = GradientBoosting(random_strength=strength,
                             use_pointwise_searcher=pointwise, numeric_mode=mode)
    params = model._params(32, 4, 0)
    assert len(params) == 35
    assert params[25] == strength
    assert params[26] == int(pointwise)


def test_default_noise_and_searcher_unchanged():
    params = GradientBoosting()._params(32, 4, 0)
    assert params[25:27] == [0., 0]
