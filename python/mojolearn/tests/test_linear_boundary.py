# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Python API boundary tests of the linear family: refusals, exception types and words, returned types,
buffer admission and framework interop. Numerical results are held by the verify lanes."""

# moved from python/mojolearn/tests/test_native_cv_logistic.py
import numpy as np
import pytest

from mojolearn import LogisticRegression
from mojolearn.exceptions import NotFittedError
from mojolearn._verify_native_runtime import audit_native_call


def cvl_clean(call, *args, error=None, **kwargs):
    observed = audit_native_call(call, *args, **kwargs)
    report = observed.report()
    assert not any(report[key] for key in ('python_frames', 'forbidden_native_calls', 'unresolved_native_calls',
                                           'wrapper_work', 'unobserved_existing_threads')), report
    if error is not None:
        assert isinstance(observed.exception, error), report
        return observed.exception
    return observed.assert_native()


def cvl_data(classes=2):
    y = np.arange(48, dtype='i4') % classes
    x = np.random.default_rng(21).normal(size=(48, 5)).astype('f4')
    x[:, 0] += y * 2
    return x, y


def test_native_logistic_constructor_and_parameter_reconstruction():
    x, y = cvl_data()
    model = cvl_clean(LogisticRegression, C=2, max_iter=30, numeric_mode='identical')
    params = cvl_clean(model.get_params)
    assert list(params) == sorted(params)
    assert params == dict(penalty='l2', tol=1e-4, C=2, fit_intercept=True, class_weight=None,
                          max_iter=30, linesearch_max_iter=50, l1_ratio=None, solver='qn', numeric_mode='identical',
                          device=None)  # device= since 48f24481f1 (dev2-logistic-device)
    cvl_clean(model.predict, x, error=NotFittedError)
    cvl_clean(model.fit, x, y)
    assert cvl_clean(model.__sklearn_is_fitted__) is True
    weights = vars(model)['_w']
    cvl_clean(model.set_params, unsupported=1, error=ValueError)
    assert vars(model)['_w'] is weights
    assert cvl_clean(model.set_params) is model
    assert vars(model)['_w'] is weights
    assert cvl_clean(model.set_params, C=.5) is model
    assert not hasattr(model, '_w')
    assert cvl_clean(model.__sklearn_is_fitted__) is False
    cvl_clean(model.predict, x, error=NotFittedError)


@pytest.mark.parametrize('classes', [2, 3])
@pytest.mark.parametrize('weighted', [False, True])
def test_native_logistic_fit_predict_score_and_wrong_width(classes, weighted):
    x, y = cvl_data(classes)
    model = cvl_clean(LogisticRegression, max_iter=40, class_weight='balanced' if weighted else None)
    weight = np.linspace(.5, 1.5, len(y), dtype='f4') if weighted else None
    assert cvl_clean(model.fit, x, y, sample_weight=weight) is model
    labels = model.classes_
    prediction = cvl_clean(model.predict, x)
    proba = cvl_clean(model.predict_proba, x)
    assert type(proba) is np.ndarray
    np.testing.assert_allclose(proba.sum(axis=1), 1, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(cvl_clean(model.predict_log_proba, x), np.log(proba), rtol=1e-6, atol=1e-6)
    assert np.asarray(cvl_clean(model.decision_function, x)).shape == ((len(x),) if classes == 2 else (len(x), classes))
    assert cvl_clean(model.score, x, y) == float(np.mean(prediction == y))
    assert model.classes_ is labels
    for operation in (model.predict, model.predict_proba, model.decision_function):
        cvl_clean(operation, x[:, :-1].copy(), error=ValueError)


def test_native_logistic_refuses_callbacks_before_hooks():
    x, y = cvl_data()
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

    cvl_clean(LogisticRegression, C=Scalar(), error=TypeError)
    model = LogisticRegression(max_iter=30)
    cvl_clean(model.fit, Input(), y, error=TypeError)
    labels = y.astype(object)
    labels[0] = Scalar()
    cvl_clean(model.fit, x, labels, error=TypeError)
    cvl_clean(model.fit, x, y.astype('f8') + .25, error=ValueError)
    assert calls == []


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
    cvl_clean(getattr, obj, 'predict', error=TypeError)
    assert calls == []
