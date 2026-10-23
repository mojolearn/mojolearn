"""Class-weight validation and dispatch through the real public fit (small inputs)."""
import numpy as np
import pytest
from mojolearn import RandomForestClassifier


def _data(balanced):
    rng = np.random.default_rng(0)
    x = rng.random((40, 4), dtype=np.float32)
    y = np.where(np.arange(40) % 2 == 0, 'a', 'b') if balanced else np.where(x[:, 0] > 0.3, 'a', 'b')
    return x, y


def _proba(x, y, **kw):
    weights = kw.pop('sample_weight', None)
    model = RandomForestClassifier(n_estimators=4, max_depth=3, random_state=0, **kw)
    return np.asarray(model.fit(x, y, sample_weight=weights).predict_proba(x)).view(np.uint32)


@pytest.mark.parametrize('value', [None, {}, {'a': 1, 'b': 1}, 'balanced'])
def test_unit_path(value):
    x, y = _data(balanced=True)
    np.testing.assert_array_equal(_proba(x, y, class_weight=value), _proba(x, y))


def test_balanced_and_partial_mapping():
    x, y = _data(balanced=False)
    n_a = int(np.count_nonzero(y == 'a'))
    n_b = len(y) - n_a
    balanced = np.where(y == 'a', len(y) / (2 * n_a), len(y) / (2 * n_b)).astype(np.float32)
    np.testing.assert_array_equal(_proba(x, y, class_weight='balanced'), _proba(x, y, sample_weight=balanced))
    partial = np.where(y == 'a', 0.0, 1.0).astype(np.float32)
    np.testing.assert_array_equal(_proba(x, y, class_weight={'a': 0}), _proba(x, y, sample_weight=partial))


@pytest.mark.parametrize('weights', [{'a': -1}, {'a': np.inf}, {'a': np.nan},
    {'a': True}, {'a': '2'}, {'a': 1e100}, {'a': 1e-100}, {'a': 0, 'b': 0}, {'z': 2}])
def test_invalid_weights(weights):
    x, y = _data(balanced=True)
    with pytest.raises(ValueError):
        RandomForestClassifier(n_estimators=2, class_weight=weights).fit(x, y)


@pytest.mark.parametrize('value', ['balanced_subsample', 'unknown', [], 3])
def test_invalid_option(value):
    # checked at fit, before any binding is touched (scikit-learn's convention)
    with pytest.raises(ValueError, match='class_weight'):
        RandomForestClassifier(class_weight=value).fit([[0.0], [1.0]], [0, 1])


def test_fit_dispatch():
    """Unit class weights take the unweighted fit; other class weights are
    the per-row weights a `sample_weight` of the same values gives."""
    x, y = _data(balanced=False)
    plain = _proba(x, y)
    for value in [None, {'a': 1}]:
        np.testing.assert_array_equal(_proba(x, y, class_weight=value), plain)
    rows = np.where(y == 'a', 2.0, 1.0).astype(np.float32)
    np.testing.assert_array_equal(_proba(x, y, class_weight={'a': 2}), _proba(x, y, sample_weight=rows))
