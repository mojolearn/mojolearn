# SPDX-License-Identifier: Apache-2.0
"""Actual public meta-estimator calls, including native lifecycle behavior."""
import numpy as np
import pytest
import json

import mojolearn as ml
from mojolearn._verify_native_runtime import audit_native_call, audit_compatible_call
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
def test_public_fit_predict_score_have_no_python_lifecycle_frames(name, record_property):
    model, x, y = fixture(name)
    fit = audit_compatible_call(model.fit, x, y)
    assert fit.assert_compatible() is model
    prediction = audit_compatible_call(model.predict, x)
    output = prediction.assert_compatible()
    scored = audit_compatible_call(model.score, x, y)
    score = scored.assert_compatible()
    # Preserve every observed Python frame and the named exception in the
    # retained JUnit evidence. Native-only qualification remains false when
    # Array.__len__ ran; no other Python helper or callback is admitted.
    reports = [audit.report() for audit in (fit, prediction, scored)]
    record_property('runtime_qualification', json.dumps(reports))
    for report in reports:
        if report['approved_python_fallbacks']:
            assert report['passed'] is False
            assert {r['id'] for r in report['approved_python_fallbacks']} == {'array-length-metadata'}
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
