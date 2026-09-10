"""Host API/optional-tail checks; GPU reach and default fingerprints are separate."""
import numpy as np
import pytest

from mojolearn import GradientBoosting, GradientBoostingClassifier, GradientBoostingRegressor


@pytest.mark.parametrize('value', [None, 0, -0., -1, 1.1, np.nextafter(1., 2.),
    np.nan, np.inf, -np.inf, True, np.bool_(False), '0.5', [0.5], 1+0j, 10**400])
def test_invalid_fraction_refused(value):
    with pytest.raises(ValueError, match='feature_fraction'):
        GradientBoosting(feature_fraction=value)


@pytest.mark.parametrize('value', [1, np.int64(1), np.float32(.5),
    np.float64(.5000000000001), np.nextafter(0., 1.)])
def test_fraction_float64_roundtrip(value):
    model = GradientBoosting(feature_fraction=value)
    assert model.feature_fraction == float(value)
    slots = model._params(32, 8, 0)
    if value == 1:
        assert len(slots) == 35
    else:
        assert slots[-3:] == [-1., -1., float(value)]


@pytest.mark.parametrize('weights', [None, [1., 2.]])
@pytest.mark.parametrize('gain,hessian', [(None,None), (0.,None), (None,.25), (.5,.25)])
def test_counted_weights_and_legacy_tail_layout(weights, gain, hessian):
    options = dict(loss='Logloss', grow_policy='Lossguide', score_function='NewtonL2',
                   class_weights=weights, min_split_gain=gain, min_child_hessian=hessian)
    default = GradientBoosting(**options)._params(32, 8, 0)
    explicit = GradientBoosting(**options, feature_fraction=1.)._params(32, 8, 0)
    assert explicit == default
    offset = 35 + len(weights or [])
    old_tail = [] if gain is None else [gain]
    if hessian is not None:
        old_tail = [-1. if gain is None else gain, hessian]
    assert default[offset:] == old_tail
    enabled = GradientBoosting(**options, feature_fraction=.5)._params(32, 8, 0)
    assert enabled[:offset] == default[:offset]
    assert enabled[offset:] == [-1. if gain is None else gain,
                                -1. if hessian is None else hessian, .5]


@pytest.mark.parametrize('name', ['cat_features', 'one_hot_features'])
@pytest.mark.parametrize('value', [[0], np.array([0, 2], dtype=np.int32)])
def test_enabled_fraction_refuses_categorical_paths(name, value):
    with pytest.raises(NotImplementedError, match='numeric features'):
        GradientBoosting(feature_fraction=.5, **{name: value})


@pytest.mark.parametrize('name', ['cat_features', 'one_hot_features'])
def test_default_does_not_disable_existing_categorical_options(name):
    model = GradientBoosting(**{name: [0]})
    assert model.feature_fraction == 1.
    assert len(model._params(32, 8, 1)) == 35


@pytest.mark.parametrize('cls', [GradientBoostingClassifier, GradientBoostingRegressor])
def test_adapter_reuses_validation_and_preserves_raw_parameter(cls):
    fraction = np.float64(.5000000000001)
    model = cls(feature_fraction=fraction)
    assert model.get_params()['feature_fraction'] is fraction
    assert model._new_learner('identical').feature_fraction == float(fraction)
    with pytest.raises(ValueError, match='feature_fraction'):
        model.set_params(feature_fraction=False)
    assert model.feature_fraction is fraction
    model.set_params(feature_fraction=.25)
    assert model._new_learner('fast')._params(32, 8, 0)[-1] == .25


@pytest.mark.parametrize('cls', [GradientBoostingClassifier, GradientBoostingRegressor])
def test_adapter_sklearn_clone_preserves_fraction(cls):
    sklearn = pytest.importorskip('sklearn.base')
    model = cls(feature_fraction=np.float64(.5), numeric_mode='identical')
    cloned = sklearn.clone(model)
    assert cloned.get_params() == model.get_params()
    assert not cloned.__sklearn_is_fitted__()
