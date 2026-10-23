"""Class-weight validation and dispatch; the dispatch test fits for real (small inputs)."""
import numpy as np
import pytest
from mojolearn import RandomForestClassifier
from mojolearn.randomforest import _class_weight_rows


@pytest.mark.parametrize('value', [None, {}, {'a': 1, 'b': 1}, 'balanced'])
def test_unit_path(value):
    assert _class_weight_rows(value, np.array(['a', 'b']), np.array([0, 1])) is None


def test_balanced_and_partial_mapping():
    classes = np.array(['a', 'b'])
    codes = np.array([0, 0, 0, 1])
    np.testing.assert_array_equal(_class_weight_rows('balanced', classes, codes),
                                  np.array([2/3, 2/3, 2/3, 2], dtype=np.float32))
    np.testing.assert_array_equal(_class_weight_rows({'a': 0}, classes, codes),
                                  np.array([0, 0, 0, 1], dtype=np.float32))


@pytest.mark.parametrize('weights', [{'a': -1}, {'a': np.inf}, {'a': np.nan},
    {'a': True}, {'a': '2'}, {'a': 1e100}, {'a': 1e-100}, {'a': 0, 'b': 0}, {'z': 2}])
def test_invalid_weights(weights):
    with pytest.raises(ValueError):
        _class_weight_rows(weights, np.array(['a', 'b']), np.array([0, 1]))


@pytest.mark.parametrize('value', ['balanced_subsample', 'unknown', [], 3])
def test_invalid_option(value):
    # checked at fit, before any binding is touched (scikit-learn's convention)
    with pytest.raises(ValueError, match='class_weight'):
        RandomForestClassifier(class_weight=value).fit([[0.0], [1.0]], [0, 1])


def test_fit_dispatch():
    """Unit class weights take the unweighted fit; other class weights are
    the per-row weights a `sample_weight` of the same values gives."""
    rng = np.random.default_rng(0)
    x = rng.random((40, 4), dtype=np.float32)
    y = np.where(x[:, 0] > 0.3, 'a', 'b')

    def proba(**kw):
        weights = kw.pop('sample_weight', None)
        model = RandomForestClassifier(n_estimators=4, max_depth=3, random_state=0, **kw)
        return np.asarray(model.fit(x, y, sample_weight=weights).predict_proba(x)).view(np.uint32)

    plain = proba()
    for value in [None, {'a': 1}]:
        np.testing.assert_array_equal(proba(class_weight=value), plain)
    rows = np.where(y == 'a', 2.0, 1.0).astype(np.float32)
    np.testing.assert_array_equal(proba(class_weight={'a': 2}), proba(sample_weight=rows))
