"""Host class-weight validation/dispatch; these tests do not execute a GPU fit."""
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
    with pytest.raises(ValueError, match='class_weight'):
        RandomForestClassifier(class_weight=value)


def test_fit_dispatch(monkeypatch):
    # This unit test isolates weighted entry selection, not export ownership.
    monkeypatch.setenv("MOJOLEARN_FOREST_EXPORT", "legacy")
    import ctypes
    from types import SimpleNamespace
    seen = []
    def plain(*args):
        seen.append(None)
    def weighted(x, y, params, criterion, address):
        seen.append(np.ctypeslib.as_array((ctypes.c_float * 4).from_address(address)).copy())
    binding = SimpleNamespace(rf_classifier_fit=plain, rf_classifier_fit_weighted=weighted)
    monkeypatch.setattr(RandomForestClassifier, '_bind', lambda *args: binding)
    def arrays(self, x, y, nclasses, fit_fn):
        fit_fn(0, 0, [], 0)
        return self
    monkeypatch.setattr(RandomForestClassifier, '_fit_arrays', arrays)
    x = np.zeros((4, 1), np.float32)
    y = np.array(['a', 'a', 'a', 'b'])
    for value in [None, {'a': 1}, {'a': 2}]:
        RandomForestClassifier(class_weight=value).fit(x, y)
    assert seen[:2] == [None, None]
    np.testing.assert_array_equal(seen[2], [2, 2, 2, 1])
