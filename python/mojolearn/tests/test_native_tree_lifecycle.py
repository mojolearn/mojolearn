"""Public tree construction and label consumption stay within native owners."""
import inspect

import numpy as np
import pytest

import mojolearn as ml
from mojolearn._labels import ClassLabels
from mojolearn._verify_native_runtime import audit_native_call


FORESTS = (ml.RandomForestClassifier, ml.RandomForestRegressor,
           ml.ExtraTreesClassifier, ml.ExtraTreesRegressor)
TREES = (ml.DecisionTreeClassifier, ml.DecisionTreeRegressor)


@pytest.mark.parametrize('cls', FORESTS + TREES)
def test_public_constructor_binds_defaults_and_preserves_parameter_objects(cls):
    # Constructors store objects unchanged; numerical/value checks belong to fit.
    marker = object()
    supplied = dict(max_depth=marker, random_state=np.int64(7), numeric_mode='identical')
    model = audit_native_call(cls, **supplied).assert_native()
    fields = vars(model)
    signature = inspect.signature(cls)
    for name, parameter in signature.parameters.items():
        expected = supplied.get(name, parameter.default)
        assert fields[name] is expected
    assert fields['_has_constructor_parameters'] is True
    assert '_cfg' not in fields


@pytest.mark.parametrize('cls', FORESTS)
def test_forest_positional_count_and_duplicate_keyword_are_bound_natively(cls):
    count = np.int64(3)
    model = audit_native_call(cls, count, numeric_mode='identical').assert_native()
    assert vars(model)['n_estimators'] is count
    for args, kwargs, message in (
        ((3, 4), {}, 'too many positional'),
        ((3,), {'n_estimators': 4}, 'multiple values'),
        ((), {'unknown_tree_parameter': 2}, 'unexpected keyword'),
    ):
        result = audit_native_call(cls, *args, **kwargs)
        assert isinstance(result.exception, TypeError), result.report()
        assert message in str(result.exception)
        report = result.report()
        assert not report['python_frames'] and not report['forbidden_native_calls'], report


@pytest.mark.parametrize('cls', TREES)
def test_tree_constructor_keeps_keyword_only_contract(cls):
    result = audit_native_call(cls, 3)
    assert isinstance(result.exception, TypeError), result.report()
    assert 'too many positional' in str(result.exception)
    report = result.report()
    assert not report['python_frames'] and not report['forbidden_native_calls'], report


@pytest.mark.parametrize('cls', (ml.RandomForestClassifier, ml.ExtraTreesClassifier,
                                 ml.DecisionTreeClassifier))
@pytest.mark.parametrize('labels', ([2, 5], [2.0, 5.0], [False, True], ['left', 'right']))
def test_public_classifier_predict_reads_classlabels_native_storage(cls, labels):
    x = np.arange(36, dtype=np.float32).reshape(12, 3)
    y = [labels[i % 2] for i in range(len(x))]
    options = {'max_depth': 3, 'random_state': 13, 'numeric_mode': 'identical'}
    if cls in FORESTS:
        options['n_estimators'] = 3
    model = audit_native_call(cls, **options).assert_native()
    audit_native_call(model.fit, x, y).assert_native()
    # The published ClassLabels interface is retained. Its Python indexing
    # method must not be called by native prediction, even for numeric labels.
    assert type(model.classes_) is ClassLabels
    actual = audit_native_call(model.predict, x).assert_native()
    codes = np.asarray(model.predict_proba(x)).argmax(axis=1)
    expected = [list.__getitem__(model.classes_, int(code)) for code in codes]
    np.testing.assert_array_equal(np.asarray(actual), expected)
    if type(labels[0]) in (str, bool):
        assert type(actual) is list
    else:
        assert actual.dtype == ('<i8' if type(labels[0]) is int else '<f8')
    score = audit_native_call(model.score, x, y).assert_native()
    assert score == np.mean(np.asarray(expected) == np.asarray(y))


def test_modified_classlabels_hook_refuses_before_invocation(monkeypatch):
    x = np.arange(36, dtype=np.float32).reshape(12, 3)
    model = ml.RandomForestClassifier(n_estimators=3, max_depth=3,
                                     random_state=13, numeric_mode='identical').fit(x, [0, 1] * 6)
    calls = []

    def hostile(self, key):
        calls.append(key)
        raise AssertionError('ClassLabels hook executed')

    monkeypatch.setattr(ClassLabels, '__getitem__', hostile)
    result = audit_native_call(model.predict, x)
    assert isinstance(result.exception, TypeError), result.report()
    assert calls == []
    report = result.report()
    assert not report['python_frames'] and not report['forbidden_native_calls'], report


@pytest.mark.parametrize('cls', FORESTS + TREES)
def test_public_parameter_update_clears_fit_without_eager_validation(cls):
    model = cls(numeric_mode='identical')
    # This tests lifecycle state directly, without compiling/fitting a large
    # default forest merely to obtain stale state to clear.
    vars(model).update(_mojolearn_fitted=True, n_features_in_=3, old_cache=object())
    marker = object()
    params = audit_native_call(model.get_params).assert_native()
    assert params['numeric_mode'] == 'identical'
    changed = audit_native_call(model.set_params, max_depth=marker).assert_native()
    assert changed is model and vars(model)['max_depth'] is marker
    assert set(vars(model)) == set(params) | {'_has_constructor_parameters'}
    before = dict(vars(model))
    result = audit_native_call(model.set_params, unknown_tree_parameter=1)
    assert isinstance(result.exception, ValueError), result.report()
    assert vars(model) == before
    report = result.report()
    assert not report['python_frames'] and not report['forbidden_native_calls'], report


def test_public_tree_refuses_input_protocol_before_invocation():
    calls = []

    class Input:
        def __array__(self, *args, **kwargs):
            calls.append('array')
            raise AssertionError('input protocol ran')

    model = ml.RandomForestRegressor(n_estimators=3, numeric_mode='identical')
    result = audit_native_call(model.fit, Input(), [1., 2.])
    assert isinstance(result.exception, TypeError), result.report()
    assert calls == []
    report = result.report()
    assert not report['python_frames'] and not report['forbidden_native_calls'], report


@pytest.mark.parametrize('cls', (ml.RandomForestRegressor, ml.ExtraTreesRegressor,
                                 ml.DecisionTreeRegressor))
def test_public_regressor_fit_predict_and_not_fitted_checks_are_native(cls):
    x = np.arange(36, dtype=np.float32).reshape(12, 3)
    y = (x[:, 0] * np.float32(.25)).copy()
    options = dict(max_depth=3, random_state=13, numeric_mode='identical')
    if cls in FORESTS:
        options['n_estimators'] = 3
    model = audit_native_call(cls, **options).assert_native()
    unfitted = audit_native_call(model.predict, x)
    assert isinstance(unfitted.exception, ml.exceptions.NotFittedError), unfitted.report()
    report = unfitted.report()
    assert not report['python_frames'] and not report['forbidden_native_calls'], report
    audit_native_call(model.fit, x, y).assert_native()
    first = np.asarray(audit_native_call(model.predict, x).assert_native()).copy()
    second = np.asarray(audit_native_call(model.predict, x).assert_native())
    np.testing.assert_array_equal(first.view(np.uint8), second.view(np.uint8))
    assert np.isfinite(first).all()
    assert vars(model)['_mojolearn_fitted'] is True


def test_public_classifier_keeps_continuous_target_refusal():
    model = ml.DecisionTreeClassifier(numeric_mode='identical')
    x = np.arange(36, dtype=np.float32).reshape(12, 3)
    result = audit_native_call(model.fit, x, [0.25, 1.5] * 6)
    assert isinstance(result.exception, ValueError), result.report()
    assert 'continuous' in str(result.exception)
    report = result.report()
    assert not report['python_frames'] and not report['forbidden_native_calls'], report
