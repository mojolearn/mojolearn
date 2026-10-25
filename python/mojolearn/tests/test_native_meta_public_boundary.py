# SPDX-License-Identifier: Apache-2.0
"""Actual public meta-estimator calls, including native lifecycle behavior."""
import numpy as np
import pytest

import mojolearn as ml
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.model_selection import KFold
from mojolearn.exceptions import NotFittedError


NAMES = ('VotingRegressor', 'VotingClassifier', 'StackingRegressor',
         'StackingClassifier', 'MultiOutputRegressor', 'MultiOutputClassifier',
         'OneVsRestClassifier', 'CalibratedClassifierCV')


def fixture(name):
    x = np.random.default_rng(29).normal(size=(24, 4)).astype('f4')
    labels = (np.arange(24) % 2).astype('i8')
    y = (x[:, 0] - 0.25 * x[:, 1]).astype('f4')
    cv = KFold(2, shuffle=True, random_state=1)
    if name.endswith('Regressor') and name.startswith(('Voting', 'Stacking')):
        members = [('ridge', ml.Ridge()), ('bayes', ml.BayesianRidge(max_iter=10))]
        model = ml.VotingRegressor(members) if name == 'VotingRegressor' else ml.StackingRegressor(members, cv=cv)
    elif name.startswith(('Voting', 'Stacking')):
        members = [('logistic', ml.LogisticRegression(max_iter=20)), ('bayes', ml.GaussianNB())]
        model = ml.VotingClassifier(members, voting='soft') if name == 'VotingClassifier' else ml.StackingClassifier(members, cv=cv)
        y = labels
    elif name == 'MultiOutputRegressor':
        model, y = ml.MultiOutputRegressor(ml.Ridge()), np.column_stack([y, -y])
    elif name == 'MultiOutputClassifier':
        model, y = ml.MultiOutputClassifier(ml.LogisticRegression(max_iter=20)), np.column_stack([labels, labels])
    elif name == 'OneVsRestClassifier':
        model, y = ml.OneVsRestClassifier(ml.LogisticRegression(max_iter=20)), labels
    else:
        model, y = ml.CalibratedClassifierCV(cv=cv), labels
    return model, x, y


@pytest.mark.parametrize('name', NAMES)
def test_unfitted_public_prediction_has_the_supported_exception(name):
    model, x, _ = fixture(name)
    audit = audit_native_call(model.predict, x)
    assert isinstance(audit.exception, NotFittedError), audit.report()
    report = audit.report()
    assert not any(report[k] for k in ('python_frames', 'wrapper_work',
        'forbidden_native_calls', 'unresolved_native_calls', 'unobserved_existing_threads')), report


@pytest.mark.parametrize('name', NAMES)
def test_public_fit_predict_score_have_no_python_lifecycle_frames(name):
    model, x, y = fixture(name)
    assert audit_native_call(model.fit, x, y).assert_native() is model
    output = audit_native_call(model.predict, x).assert_native()
    score = audit_native_call(model.score, x, y).assert_native()
    assert isinstance(output, np.ndarray)
    assert output.shape == y.shape and np.isfinite(float(score))
    with pytest.raises(ValueError, match='features'):
        model.predict(np.zeros((2, 5), dtype='f4'))


def test_public_fit_refuses_python_input_protocol_without_invoking_it():
    calls = []
    class Input:
        def __array__(self, *args, **kwargs):
            calls.append('array')
            return np.zeros((24, 4), dtype='f4')
    model, _, y = fixture('VotingRegressor')
    with pytest.raises(TypeError):
        model.fit(Input(), y)
    assert calls == []
