# SPDX-License-Identifier: Apache-2.0
"""Descriptor classifier admission and unchanged independent fold assignment."""
import numpy as np

import mojolearn as ml
from mojolearn import model_selection as ms
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.tests._fold_reference import default_folds


def inputs():
    x = np.arange(120, dtype='f4').reshape(30, 4) / np.float32(32)
    y = np.asarray([8, -3, 5] * 10, dtype='i4')
    return x, y


def test_split_descriptor_refuses_fake_classifier_before_any_protocol_hook():
    calls = []

    class _FoldClf:
        _estimator_type = 'classifier'

        def __getattribute__(self, name):
            calls.append(name)
            raise AssertionError('fake classifier attribute hook ran')

        def get_params(self, deep=False):
            calls.append('get_params')
            raise AssertionError('fake classifier parameter hook ran')

        def __sklearn_tags__(self):
            calls.append('__sklearn_tags__')
            raise AssertionError('fake classifier tag hook ran')

    x, y = inputs()
    observed = audit_native_call(ms.split_descriptor, x, y, estimator=_FoldClf(), cv=5)
    report = observed.report()
    assert isinstance(observed.exception, TypeError), report
    assert calls == []
    assert not any(report[key] for key in ('python_frames', 'forbidden_native_calls',
                                           'unresolved_native_calls', 'wrapper_work')), report


def test_native_classifier_descriptor_matches_independent_explicit_folds():
    x, y = inputs()
    classifier = ml.LogisticRegression(numeric_mode='identical')
    before = dict(vars(classifier))
    pairs = [(np.asarray(train, dtype='i8'), np.asarray(test, dtype='i8'))
             for train, test in default_folds(y.tolist(), 5, True)]
    actual = audit_native_call(ms.split_descriptor, x, y, estimator=classifier, cv=5).assert_native()
    expected = audit_native_call(ms.split_descriptor, x, y, cv=pairs).assert_native()
    assert actual == expected
    assert vars(classifier) == before
