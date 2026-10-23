# SPDX-License-Identifier: Apache-2.0
"""Exercise LogisticRegression's actual public and model-selection boundaries."""
import numpy as np
import pytest

from mojolearn import LogisticRegression, model_selection as ms
from mojolearn.exceptions import NotFittedError
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


def test_native_logistic_constructor_and_parameter_reconstruction():
    x, y = data()
    model = clean(LogisticRegression, C=2, max_iter=30, numeric_mode='identical')
    params = clean(model.get_params)
    assert list(params) == sorted(params)
    assert params == dict(penalty='l2', tol=1e-4, C=2, fit_intercept=True, class_weight=None,
                          max_iter=30, linesearch_max_iter=50, l1_ratio=None, solver='qn', numeric_mode='identical')
    clean(model.predict, x, error=NotFittedError)
    clean(model.fit, x, y)
    assert clean(model.__sklearn_is_fitted__) is True
    weights = vars(model)['_w']
    clean(model.set_params, unsupported=1, error=ValueError)
    assert vars(model)['_w'] is weights
    assert clean(model.set_params) is model
    assert vars(model)['_w'] is weights
    assert clean(model.set_params, C=.5) is model
    assert not hasattr(model, '_w')
    assert clean(model.__sklearn_is_fitted__) is False
    clean(model.predict, x, error=NotFittedError)


@pytest.mark.parametrize('classes', [2, 3])
@pytest.mark.parametrize('weighted', [False, True])
def test_native_logistic_fit_predict_score_and_wrong_width(classes, weighted):
    x, y = data(classes)
    model = clean(LogisticRegression, max_iter=40, class_weight='balanced' if weighted else None)
    weight = np.linspace(.5, 1.5, len(y), dtype='f4') if weighted else None
    assert clean(model.fit, x, y, sample_weight=weight) is model
    labels = model.classes_
    prediction = clean(model.predict, x)
    proba = clean(model.predict_proba, x)
    assert type(proba) is np.ndarray
    np.testing.assert_allclose(proba.sum(axis=1), 1, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(clean(model.predict_log_proba, x), np.log(proba), rtol=1e-6, atol=1e-6)
    assert np.asarray(clean(model.decision_function, x)).shape == ((len(x),) if classes == 2 else (len(x), classes))
    assert clean(model.score, x, y) == float(np.mean(prediction == y))
    assert model.classes_ is labels
    for operation in (model.predict, model.predict_proba, model.decision_function):
        clean(operation, x[:, :-1].copy(), error=ValueError)


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


def test_native_logistic_refuses_callbacks_before_hooks():
    x, y = data()
    calls = []

    class Scalar:
        def __float__(self):
            calls.append('float')
            return 1.

    class Input:
        @property
        def shape(self):
            calls.append('shape')
            return x.shape

        def __array__(self, *args, **kwargs):
            calls.append('array')
            return x

    clean(LogisticRegression, C=Scalar(), error=TypeError)
    model = LogisticRegression(max_iter=30)
    clean(model.fit, Input(), y, error=TypeError)
    labels = y.astype(object)
    labels[0] = Scalar()
    clean(model.fit, x, labels, error=TypeError)
    clean(model.fit, x, y.astype('f8') + .25, error=ValueError)
    assert calls == []


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
    assert entries['estimators_vendor']() == 'cpu'
    x, y = data()
    clean(model.fit, x, y)
    assert np.asarray(clean(model.predict, x)).shape == y.shape
    assert clean(model.score, x, y) >= 0


def test_native_attribute_dispatch_refuses_metaclass_hash_before_lookup():
    calls = []

    class Meta(type):
        def __hash__(cls):
            calls.append('hash')
            return type.__hash__(cls)

    class External(LogisticRegression, metaclass=Meta):
        pass

    obj = object.__new__(External)
    calls.clear()
    clean(getattr, obj, 'predict', error=TypeError)
    assert calls == []
