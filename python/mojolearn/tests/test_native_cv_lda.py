# SPDX-License-Identifier: Apache-2.0
"""LDA constructor, parameter metadata and cross-validation use native entries."""
import numpy as np
import pytest

from mojolearn import Array, LinearDiscriminantAnalysis, model_selection as ms
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
    labels = np.arange(42, dtype='i4') % 3
    source = np.random.default_rng(43).normal(size=(42, 5)).astype('f4')
    source[:, 0] += labels * 2.5
    source[:, 1] -= labels * 1.5
    return source, labels


@pytest.mark.parametrize('solver,shrinkage', [('svd', None), ('lsqr', .2), ('eigen', 'auto')])
def test_native_lda_constructor_parameters_and_public_fit(solver, shrinkage):
    x, y = data()
    model = clean(LinearDiscriminantAnalysis, solver=solver, shrinkage=shrinkage, store_covariance=True)
    params = clean(model.get_params)
    assert params == dict(solver=solver, shrinkage=shrinkage, priors=None, n_components=None,
                          store_covariance=True, tol=1e-4, covariance_estimator=None)
    assert clean(model.set_params, tol=2e-4) is model
    assert clean(model.get_params, deep=False)['tol'] == 2e-4
    clean(model.set_params, unknown=1, error=ValueError)
    assert clean(model.fit, x, y) is model
    prediction = clean(model.predict, x)
    proba = clean(model.predict_proba, x)
    score = clean(model.score, x, y)
    np.testing.assert_allclose(np.asarray(proba).sum(axis=1), 1, rtol=1e-5, atol=1e-5)
    assert score == float(np.mean(np.asarray(prediction) == y))
    assert np.asarray(clean(model.decision_function, x)).shape == (len(x), 3)
    if solver != 'lsqr':
        assert np.asarray(clean(model.transform, x)).shape == (len(x), 2)


@pytest.mark.parametrize('priors_kind', ['none', 'list', 'ndarray', 'Array'])
@pytest.mark.parametrize('scoring', [None, 'accuracy'])
def test_native_lda_cv_matches_explicit_native_fits_without_mutating_constructor(priors_kind, scoring):
    x, y = data()
    priors = None
    if priors_kind == 'list':
        priors = [.2, .3, .5]
    elif priors_kind == 'ndarray':
        priors = np.array([.2, .3, .5], dtype='f8')
    elif priors_kind == 'Array':
        priors = Array.from_list([.2, .3, .5], '<f8')
    before = None if priors is None else np.asarray(priors).copy()
    model = LinearDiscriminantAnalysis(priors=priors)
    folds = list(ms.StratifiedKFold(3).split(x, y))
    actual = np.asarray(clean(ms.cross_val_score, model, x, y, cv=folds, scoring=scoring))
    expected = []
    for train, test in folds:
        train, test = np.asarray(train), np.asarray(test)
        fitted = LinearDiscriminantAnalysis(priors=priors).fit(x[train], y[train])
        expected.append(fitted.score(x[test], y[test]))
    np.testing.assert_array_equal(actual, np.asarray(expected))
    assert not hasattr(model, 'classes_')
    assert clean(model.get_params)['priors'] is priors
    if priors is not None:
        np.testing.assert_array_equal(np.asarray(priors), before)


def test_native_lda_permutation_score_runs_complete_native_boundary():
    x, y = data()
    score, shuffled, pvalue = clean(ms.permutation_test_score, LinearDiscriminantAnalysis(), x, y,
                                    cv=3, n_permutations=3, random_state=17, scoring='accuracy')
    assert np.asarray(shuffled).shape == (3,)
    assert pvalue == (1 + int(np.count_nonzero(np.asarray(shuffled) >= score))) / 4


def test_lda_rejects_covariance_and_scalar_callbacks_without_touching_them():
    calls = []

    class Covariance:
        def fit(self, x):
            calls.append('fit')
            return self

        @property
        def covariance_(self):
            calls.append('covariance')
            return np.eye(5)

    class Scalar:
        def __float__(self):
            calls.append('float')
            return .2

    clean(LinearDiscriminantAnalysis, solver='lsqr', covariance_estimator=Covariance(), error=TypeError)
    clean(LinearDiscriminantAnalysis, solver='lsqr', shrinkage=Scalar(), error=TypeError)
    model = LinearDiscriminantAnalysis()
    clean(model.set_params, covariance_estimator=Covariance(), error=TypeError)
    assert calls == []


def test_lda_object_labels_refuse_scalar_protocols_before_fit():
    x, _ = data()
    calls = []

    class Label:
        def __hash__(self):
            calls.append('hash')
            return 1

        def __float__(self):
            calls.append('float')
            return 1.0

    y = np.empty(len(x), dtype=object)
    y[:] = Label()
    model = LinearDiscriminantAnalysis()
    clean(model.fit, x, y, error=TypeError)
    assert calls == []
