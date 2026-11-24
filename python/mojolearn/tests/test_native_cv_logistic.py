# SPDX-License-Identifier: Apache-2.0
"""Exercise LogisticRegression's model-selection boundary and its explicit host owner."""
import numpy as np

from mojolearn import LogisticRegression, model_selection as ms
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


def data(classes=2):
    y = np.arange(48, dtype='i4') % classes
    x = np.random.default_rng(21).normal(size=(48, 5)).astype('f4')
    x[:, 0] += y * 2
    return x, y


def test_native_logistic_cv_and_randomized_search_match_explicit_fits():
    x, y = data()
    folds = list(ms.StratifiedKFold(3).split(x, y))
    model = LogisticRegression(max_iter=30)
    actual = clean(ms.cross_val_score, model, x, y, cv=folds, scoring='accuracy')
    expected = []
    for train, test in folds:
        train, test = np.asarray(train), np.asarray(test)
        fitted = LogisticRegression(max_iter=30).fit(x[train], y[train])
        expected.append(fitted.score(x[test], y[test]))
    np.testing.assert_array_equal(actual, np.asarray(expected))
    search = ms.RandomizedSearchCV(model, {'C': [.5, 1., 2.]}, n_iter=2, cv=folds,
                                   random_state=23, scoring='accuracy')
    assert clean(search.fit, x, y) is search
    prediction = clean(search.predict, x)
    np.testing.assert_array_equal(prediction, search.best_estimator_.predict(x))
    assert not hasattr(model, '_w')


def test_explicit_host_logistic_keeps_native_constructor_and_cpu_binding():
    from mojolearn._classical_host import HostLogisticRegression
    model = clean(HostLogisticRegression, max_iter=30, numeric_mode='identical')
    assert clean(model.get_params)['numeric_mode'] == 'identical'
    # This explicit owner is exercised in GPU packages too; it must retain
    # CPU metadata rather than inherit the process vendor's estimator entries.
    from mojolearn.linear_model import _LR_CONTEXT
    owner = _LR_CONTEXT['lr_host_contexts'][HostLogisticRegression]
    entries = owner['lr_bindings']['identical']
    assert type(entries) is dict
    # 48de8844b5: the owner keeps the one typed entry; its module is the CPU estimators binding.
    assert set(entries) == {'logistic_native_api_v1'}
    assert entries['logistic_native_api_v1'].__self__.estimators_vendor() == 'cpu'
    x, y = data()
    clean(model.fit, x, y)
    assert np.asarray(clean(model.predict, x)).shape == y.shape
    assert clean(model.score, x, y) >= 0
