# SPDX-License-Identifier: Apache-2.0
"""Established boosting/NB callers keep native CV, search and permutation paths.

Explicit public fits below are numerical test controls, outside the audited CV
boundary; this does not qualify their older Python public wrappers as native.
"""
import numpy as np
import pytest

from mojolearn import GaussianNB, GradientBoostingRegressor, LinearDiscriminantAnalysis, model_selection as ms
from mojolearn._verify_native_runtime import audit_native_call


def clean(call, *args, error=None, **kwargs):
    observed = audit_native_call(call, *args, **kwargs)
    report = observed.report()
    assert not any(report[key] for key in ('python_frames', 'forbidden_native_calls', 'unresolved_native_calls',
                                           'wrapper_work', 'unobserved_existing_threads')), report
    if error is not None:
        assert isinstance(observed.exception, error), report
        return observed.exception
    return observed.assert_native()


def data():
    x = np.random.default_rng(37).normal(size=(42, 4)).astype('f4')
    y = (2 * x[:, 0] - x[:, 1] + .2 * x[:, 2] ** 2).astype('f4')
    return x, y


def boosting(**kwargs):
    return GradientBoostingRegressor(n_estimators=4, max_depth=3, numeric_mode='identical', **kwargs)


def test_native_gbr_cv_default_named_scores_and_predictions_match_explicit_fits():
    x, y = data()
    folds = list(ms.KFold(3).split(x, y))
    model = boosting()
    scores, predicted = [], np.empty_like(y)
    for train, test in folds:
        train, test = np.asarray(train), np.asarray(test)
        control = boosting().fit(x[train], y[train])
        scores.append(control.score(x[test], y[test]))
        predicted[test] = control.predict(x[test])
    actual = clean(ms.cross_val_score, model, x, y, cv=folds)
    np.testing.assert_array_equal(actual, scores)
    result = clean(ms.cross_validate, model, x, y, cv=folds,
                   scoring={'r2': 'r2', 'mae': 'neg_mean_absolute_error'}, return_train_score=True,
                   return_estimator=True, error_score='raise')
    np.testing.assert_array_equal(result['test_r2'], clean(ms.cross_val_score, model, x, y, cv=folds, scoring='r2'))
    assert np.asarray(result['train_mae']).shape == (3,)
    assert len(result['estimator']) == 3
    np.testing.assert_array_equal(clean(ms.cross_val_predict, model, x, y, cv=folds), predicted)
    assert not hasattr(model, '_learner_')
    assert all(estimator is not model for estimator in result['estimator'])


def test_native_gbr_grid_search_preserves_constructor_and_refit_semantics():
    x, y = data()
    model = boosting()
    original = dict(vars(model))
    search = ms.GridSearchCV(model, {'max_depth': [2, 3], 'learning_rate': [.05, .1]}, cv=3,
                            scoring='neg_mean_squared_error', error_score='raise')
    assert clean(search.fit, x, y) is search
    assert vars(model) == original
    assert not hasattr(model, '_learner_')
    best = search.best_estimator_
    assert best.max_depth == search.best_params_['max_depth']
    assert best.learning_rate == search.best_params_['learning_rate']
    control = GradientBoostingRegressor(**best.get_params()).fit(x, y)
    np.testing.assert_array_equal(clean(search.predict, x), control.predict(x))
    invalid = ms.GridSearchCV(model, {'not_a_parameter': [1]}, cv=2, error_score='raise')
    clean(invalid.fit, x, y, error=ValueError)
    assert vars(model) == original


@pytest.mark.parametrize('priors_kind', ['list', 'ndarray'])
def test_native_gaussian_nb_cv_copies_priors_and_keeps_native_fold_route(priors_kind):
    x, _ = data()
    y = (np.arange(len(x)) % 3).astype('i4')
    x[:, 0] += y * 2
    priors = [.2, .3, .5] if priors_kind == 'list' else np.array([.2, .3, .5], dtype='f8')
    original = np.asarray(priors).copy()
    model = GaussianNB(priors=priors)
    folds = list(ms.StratifiedKFold(3).split(x, y))
    expected = []
    for train, test in folds:
        train, test = np.asarray(train), np.asarray(test)
        expected.append(GaussianNB(priors=priors).fit(x[train], y[train]).score(x[test], y[test]))
    np.testing.assert_array_equal(clean(ms.cross_val_score, model, x, y, cv=folds), expected)
    result = clean(ms.cross_validate, model, x, y, cv=folds, scoring='accuracy',
                   return_estimator=True, error_score='raise')
    np.testing.assert_array_equal(result['test_score'], expected)
    assert not hasattr(model, 'classes_')
    assert model.priors is priors
    for estimator in result['estimator']:
        assert estimator.priors is not priors
        np.testing.assert_array_equal(estimator.priors, original)
    np.testing.assert_array_equal(priors, original)
    probability = clean(ms.cross_val_predict, model, x, y, cv=folds, method='predict_proba')
    np.testing.assert_allclose(np.asarray(probability).sum(axis=1), 1, rtol=1e-6, atol=1e-6)


def test_native_gaussian_nb_group_permutation_contract_and_parameter_search():
    x, _ = data()
    y = (np.arange(len(x)) % 3).astype('i4')
    groups = np.repeat(np.arange(7, dtype='i4'), 6)
    x[:, 0] += y * 2
    model = GaussianNB()
    folds = list(ms.GroupKFold(3).split(x, y, groups))
    expected = clean(ms.cross_val_score, model, x, y, cv=folds, scoring='accuracy')
    score, shuffled, pvalue = clean(ms.permutation_test_score, model, x, y, groups=groups,
                                  cv=ms.GroupKFold(3), n_permutations=3, random_state=19, scoring='accuracy')
    np.testing.assert_allclose(score, np.mean(expected), rtol=1e-15, atol=0)
    assert np.asarray(shuffled).shape == (3,)
    assert pvalue == (1 + int(np.count_nonzero(np.asarray(shuffled) >= score))) / 4
    search = ms.GridSearchCV(model, {'var_smoothing': [1e-9, 1e-6]}, cv=folds,
                            scoring='accuracy', error_score='raise')
    clean(search.fit, x, y)
    np.testing.assert_array_equal(clean(search.predict, x), search.best_estimator_.predict(x))
    assert not hasattr(model, 'classes_')


@pytest.mark.parametrize('cls', [GaussianNB, LinearDiscriminantAnalysis])
def test_native_cv_connects_cold_arena_context_before_ndarray_row_snapshot(cls):
    from mojolearn import _expansion_prep as prep
    from mojolearn._native_classical_bridge import _numpy_free_context

    x, _ = data()
    y = (np.arange(len(x)) % 3).astype('i4')
    x[:, 0] += y * 2
    folds = list(ms.StratifiedKFold(3).split(x, y))
    model = cls()
    # Recreate the real import-time arena admission state, independent of
    # other tests warming it via _Prog.run. Only testing changes metadata;
    # the whole public CV operation must connect and consume it natively.
    arena = prep._PROG_CONTEXT
    saved = dict(arena)
    cold = _numpy_free_context()
    try:
        arena.pop('ndarray', None)
        for key in ('native_value_types', 'buffer_owner_types', 'scalar_types', 'integer_types'):
            arena[key] = cold[key]
        assert np.ndarray not in arena['buffer_owner_types']
        actual = clean(ms.cross_val_score, model, x, y, cv=folds, scoring='accuracy')
        assert arena['ndarray'] is np.ndarray
        assert np.ndarray in arena['buffer_owner_types']
    finally:
        arena.clear()
        arena.update(saved)
    expected = []
    for train, test in folds:
        train, test = np.asarray(train), np.asarray(test)
        fitted = cls().fit(x[train], y[train])
        expected.append(fitted.score(x[test], y[test]))
    np.testing.assert_array_equal(actual, expected)
    assert not hasattr(model, 'classes_')


def test_native_gbr_and_nb_refuse_configuration_and_input_hooks_before_invocation():
    x, y = data()
    calls = []

    class Scalar:
        def __float__(self):
            calls.append('float')
            return .1

    class Input:
        @property
        def shape(self):
            calls.append('shape')
            return x.shape

        def __array__(self, *args, **kwargs):
            calls.append('array')
            return x

    for model, labels in ((boosting(learning_rate=Scalar()), y),
                          (GaussianNB(var_smoothing=Scalar()), (y > 0).astype('i4'))):
        clean(ms.cross_val_score, model, x, labels, cv=3, error=TypeError)
    for model in (boosting(), GaussianNB()):
        clean(ms.cross_val_score, model, Input(), y, cv=3, error=TypeError)
    assert calls == []


def test_native_gbr_refuses_replaced_loss_descriptor_before_its_hook():
    x, y = data()
    calls = []

    class Loss:
        def __get__(self, instance, cls):
            calls.append('loss')
            return 'RMSE'

    model = boosting()
    old = vars(GradientBoostingRegressor)['_loss']
    GradientBoostingRegressor._loss = Loss()
    try:
        clean(ms.cross_val_score, model, x, y, cv=3, error=TypeError)
        assert calls == []
    finally:
        GradientBoostingRegressor._loss = old
