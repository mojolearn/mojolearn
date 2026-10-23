"""Whole public CV calls use exact native estimator entries and fresh clones."""
import numpy as np
import pytest

import mojolearn as ml
from mojolearn import model_selection as ms
from mojolearn._verify_native_runtime import audit_native_call


def samples():
    x = np.linspace(-1, 1, 54, dtype=np.float32).reshape(18, 3)
    x[:, 1] *= x[:, 1]
    x[:, 2] = np.cos(x[:, 2])
    y = (x @ np.array([.5, -.2, .7], 'f4')).astype('f4')
    return x, y


@pytest.mark.parametrize('cls', [ml.LinearRegression, ml.Ridge, ml.RandomForestRegressor,
                                 ml.ExtraTreesRegressor, ml.DecisionTreeRegressor])
def test_public_cross_validation_calls_native_clones(cls):
    x, y = samples()
    options = {'n_estimators': 3, 'max_depth': 3} if 'Forest' in cls.__name__ or 'ExtraTrees' in cls.__name__ else {}
    original = cls(**options)
    before = dict(vars(original))
    result = audit_native_call(ms.cross_validate, original, x, y, cv=3, n_jobs=1,
                               return_estimator=True, error_score='raise').assert_native()
    assert len(result['estimator']) == 3
    assert len({id(item) for item in result['estimator']}) == 3
    assert all(type(item) is cls and item is not original for item in result['estimator'])
    assert vars(original) == before
    assert np.isfinite(np.asarray(result['test_score'])).all()


def test_external_estimator_hooks_refuse_before_protocol_lookup():
    calls = []

    class External:
        def __getattribute__(self, name):
            calls.append(name)
            raise AssertionError('external estimator protocol ran')

    x, y = samples()
    audit = audit_native_call(ms.cross_val_score, External(), x, y, cv=3, n_jobs=1)
    assert isinstance(audit.exception, TypeError), audit.report()
    assert calls == []
    report = audit.report()
    assert not report['python_frames'] and not report['forbidden_native_calls'], report


def test_replaced_native_estimator_method_refuses_without_callback(monkeypatch):
    calls = []
    model = ml.Ridge()
    x, y = samples()
    monkeypatch.setattr(ml.Ridge, 'fit', lambda *args, **kw: calls.append('fit'))
    audit = audit_native_call(ms.cross_val_score, model, x, y, cv=3, n_jobs=1)
    assert isinstance(audit.exception, TypeError), audit.report()
    assert calls == []
