# SPDX-License-Identifier: Apache-2.0
"""Public runtime regressions for native metadata snapshots.

Run against the newly compiled bindings. Source-only checks cannot qualify these
paths, and an installed wheel from before this repair is not candidate evidence.
"""
import gc
import json

import numpy as np
import pytest

import mojolearn as ml
from mojolearn._array import Array
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.model_selection import KFold


def _stack(classifier=False, passthrough=True):
    cv = KFold(3, shuffle=True, random_state=7)
    if classifier:
        return ml.StackingClassifier(
            [('logistic', ml.LogisticRegression(max_iter=20)), ('bayes', ml.GaussianNB())],
            cv=cv, passthrough=passthrough,
        )
    return ml.StackingRegressor(
        [('ridge', ml.Ridge(alpha=0.5)), ('ridge2', ml.Ridge(alpha=2.0))],
        cv=cv, passthrough=passthrough,
    )


def _data(rows=24, cols=4):
    x = np.random.default_rng(29).normal(size=(rows, cols)).astype('f4')
    return x, (x[:, 0] - 0.25 * x[:, 1]).astype('f4')


def _audited(record_property, method, *args, **kwargs):
    audit = audit_native_call(method, *args, **kwargs)
    record_property('metadata_snapshot_runtime', json.dumps(audit.report()))
    # These targeted paths must now pass without the Array.__len__ exception.
    return audit.assert_native()


@pytest.mark.parametrize('classifier', [False, True])
def test_stacking_fit_transform_predict_never_call_array_len(classifier, monkeypatch, record_property):
    model = _stack(classifier)
    x, y = _data()
    if classifier:
        y = (np.arange(x.shape[0]) % 2).astype('i8')

    def forbidden_len(self):
        raise AssertionError('native meta-estimator called Python Array.__len__')

    monkeypatch.setattr(Array, '__len__', forbidden_len)
    weights = {} if classifier else {'sample_weight': np.linspace(0.5, 1.5, x.shape[0]).astype('f4')}
    assert _audited(record_property, model.fit, x, y, **weights) is model
    transformed = np.asarray(_audited(record_property, model.transform, x[:7]))
    assert transformed.shape == (7, 6)
    np.testing.assert_array_equal(transformed[:, -4:], x[:7])
    predicted = np.asarray(_audited(record_property, model.predict, x[:7]))
    assert predicted.shape == (7,) and np.isfinite(predicted).all()
    if classifier:
        proba = np.asarray(_audited(record_property, model.predict_proba, x[:7]))
        assert proba.shape == (7, 2) and np.isfinite(proba).all()


def test_multiclass_indicators_reuse_codes_without_array_len(monkeypatch, record_property):
    x, _ = _data(27)
    y = (np.arange(27) % 3).astype('i8')
    model = ml.OneVsRestClassifier(ml.LogisticRegression(max_iter=20))

    def forbidden_len(self):
        raise AssertionError('native indicator called Python Array.__len__')

    monkeypatch.setattr(Array, '__len__', forbidden_len)
    assert _audited(record_property, model.fit, x, y) is model
    result = np.asarray(_audited(record_property, model.predict, x[:5]))
    assert result.shape == (5,) and set(result).issubset({0, 1, 2})


def test_stacking_refreshes_shape_dtype_layout_data_and_refit_options():
    model = _stack()
    x, y = _data()
    model.fit(x, y)
    # New address, rank-preserving shape, dtype and layout on each operation.
    inputs = [x[:5].copy(), np.asfortranarray(x[:9], dtype='f8'),
              np.repeat(x[:6], 2, axis=1)[:, ::2]]
    for fresh in inputs:
        result = np.asarray(model.transform(fresh))
        assert result.shape == (fresh.shape[0], 6)
        np.testing.assert_array_equal(result[:, -4:], fresh.astype('f4'))
    fresh = inputs[0]
    fresh[:] += 3
    np.testing.assert_array_equal(np.asarray(model.transform(fresh))[:, -4:], fresh)
    # The same source object can expose a new shape on the next call.
    reshaped = x.copy()
    reshaped.shape = (12, 8)
    with pytest.raises(ValueError, match='features'):
        model.transform(reshaped)
    reshaped.shape = (24, 4)
    np.testing.assert_array_equal(np.asarray(model.transform(reshaped))[:, -4:], x)
    # Changing fitted members, options and input width must not retain old
    # native metadata. Refit is compared with an independently constructed model.
    next_x, next_y = _data(30, 3)
    model.set_params(estimators=[('only', ml.Ridge(alpha=3.0))], passthrough=False)
    model.fit(next_x, next_y)
    reference = ml.StackingRegressor([('only', ml.Ridge(alpha=3.0))],
                                    cv=KFold(3, shuffle=True, random_state=7), passthrough=False)
    reference.fit(next_x, next_y)
    actual = np.asarray(model.transform(next_x[:8]))
    assert actual.shape == (8, 1)
    np.testing.assert_array_equal(actual, np.asarray(reference.transform(next_x[:8])))
    np.testing.assert_array_equal(np.asarray(model.predict(next_x[:8])),
                                  np.asarray(reference.predict(next_x[:8])))
    with pytest.raises(ValueError, match='features'):
        model.predict(x)


def test_stacking_temporaries_and_changed_fold_sizes_keep_owners_alive():
    model = _stack()
    x, y = _data(25)  # three unequal folds exercise per-fold counts
    model.fit(x.copy(), y.copy(), sample_weight=np.ones(25, dtype='f4'))
    expected = x[:4].copy()
    output = model.transform(x[:4].copy())
    del x, y
    gc.collect()
    # Retained result bytes must survive release of every caller input owner.
    np.testing.assert_array_equal(np.asarray(output)[:, -4:], expected)
    new_x, new_y = _data(28, 3)
    model.fit(new_x.copy(), new_y.copy(), sample_weight=np.ones(28, dtype='f4'))
    assert np.asarray(model.transform(new_x[:2].copy())).shape == (2, 5)
    with pytest.raises(ValueError):
        model.fit(new_x, new_y, sample_weight=np.ones(27, dtype='f4'))
    with pytest.raises(ValueError):
        model.fit(new_x, new_y[:-1])
