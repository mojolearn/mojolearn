# SPDX-License-Identifier: Apache-2.0
"""Real native fold/worker contracts; CPU process tests are not GPU placement evidence."""
import copy
import os

import numpy as np
import pytest

from mojolearn import Array, GradientBoostingRegressor, GradientBoostingClassifier, GaussianNB, LinearRegression, _backend
from mojolearn import model_selection as serial
from mojolearn import parallel_model_selection as parallel
from mojolearn._verify_native_runtime import audit_native_call


def entry(operation, args=()):
    return serial._NATIVE(None, operation, args, serial._NATIVE_CONTEXT)


def clean(call, *args, error=None, **kwargs):
    observed = audit_native_call(call, *args, **kwargs)
    report = observed.report()
    assert not any(report[key] for key in ('python_frames', 'forbidden_native_calls', 'unresolved_native_calls',
                                           'wrapper_work', 'unobserved_existing_threads')), report
    if error is not None:
        assert isinstance(observed.exception, error), report
        return observed.exception
    return observed.assert_native()


def trace():
    return entry('family_msel_parallel_last_run')


def closed(record):
    assert record['closed'], record
    for pid in record['workers']:
        with pytest.raises(ChildProcessError):
            os.waitpid(pid, os.WNOHANG)


@pytest.fixture
def data():
    x = np.random.default_rng(37).normal(size=(41, 4)).astype('f4')
    y = (2 * x[:, 0] - x[:, 1] + .2 * x[:, 2] ** 2).astype('f4')
    return x, y


@pytest.mark.parametrize('devices', [(0,), (0, 1), (3, 1, 0)])
@pytest.mark.parametrize('scoring', [None, 'neg_mean_squared_error'])
def test_scores_fold_order_and_fresh_clones_match_serial(data, devices, scoring):
    if len(devices) > 1 and _backend.vendor() != 'cpu':
        pytest.skip('multiple logical CPU processes; physical GPU schedules use the installed verifier')
    x, y = data
    original = GradientBoostingRegressor(n_estimators=3, max_depth=2, random_state=7, numeric_mode='identical')
    original.fit(x, y)  # Test preparation; the native CV boundary must discard fitted state.
    model_before = original._learner_.model_
    expected = clean(serial.cross_val_score, original, x, y, cv=5, scoring=scoring)
    actual = clean(parallel.cross_val_score, original, x, y, devices=devices, cv=5, scoring=scoring)
    assert actual.tobytes() == expected.tobytes()
    assert original._learner_.model_ == model_before
    record = trace()
    assert record['status'] == 'COMPLETE'
    assert record['stages'][:3] == ['validate', 'validated', 'start']
    assert record['stages'][3] == 'identity' and record['stages'][-1] == 'close'
    width = len(devices)
    assert record['waves'] == [min(width, 5 - start) for start in range(0, 5, width)]
    assert [row[0] for row in record['assignments']] == list(range(5))
    assert [row[1] for row in record['assignments']] == [devices[i % width] for i in range(5)]
    assert len(set(record['workers'])) == width
    assert [row[2] for row in record['assignments']] == [record['workers'][i % width] for i in range(5)]
    closed(record)


def test_validate_all_folds_before_any_worker_fit(data):
    x, y = data
    error = clean(parallel.cross_val_score, LinearRegression(), x, y, devices=(0,),
                  cv=[([0, 1], [2]), ([0, 1], [1])], error=ValueError)
    assert 'overlap' in str(error)
    assert trace()['workers'] == [] and trace()['waves'] == []
    assert trace()['stages'] == ['validate']


@pytest.mark.parametrize('callback_kind', ['estimator', 'scorer', 'splitter', 'protocol', 'devices'])
def test_python_callbacks_are_refused_before_hooks_or_workers(data, callback_kind):
    x, y = data
    calls = []
    class Foreign:
        def __getattribute__(self, name):
            calls.append(name)
            raise AssertionError('unsupported protocol invoked')
        def __call__(self, *args):
            calls.append('call')
            return 1.
        def __iter__(self):
            calls.append('iter')
            return iter(())
    kwargs = dict(devices=(0,), cv=3)
    model = LinearRegression()
    value = Foreign()
    if callback_kind == 'estimator': model = value
    elif callback_kind == 'scorer': kwargs['scoring'] = value
    elif callback_kind == 'splitter': kwargs['cv'] = value
    elif callback_kind == 'protocol': x = value
    else: kwargs['devices'] = value
    clean(parallel.cross_val_score, model, x, y, error=TypeError, **kwargs)
    assert calls == []
    assert trace()['workers'] == [] and trace()['waves'] == []


@pytest.mark.parametrize('devices', [(), (0, 0), (-1,), (True,), ('0',)])
def test_invalid_device_lists_refused(data, devices):
    clean(parallel.cross_val_score, LinearRegression(), *data, devices=devices, error=ValueError)
    assert trace()['workers'] == []


def test_fast_mode_refused_before_workers(data):
    model = GradientBoostingRegressor(n_estimators=3, numeric_mode='fast')
    error = clean(parallel.cross_val_score, model, *data, devices=(0,), error=ValueError)
    assert 'IDENTICAL' in str(error)
    assert trace()['workers'] == []


@pytest.mark.parametrize('failure', ['fit', 'score'])
def test_fold_failure_closes_pool_without_partial_result(data, failure):
    x, y = data
    if failure == 'fit':
        # Constructor only stores this value; native fold fitting rejects it.
        model = GradientBoostingRegressor(n_estimators=3, max_depth=-1, numeric_mode='identical')
        scoring = None
    else:
        # Connected native metric, incompatible with the model's continuous
        # predictions. This fails in scoring after a successful native fit.
        model = LinearRegression()
        scoring = 'accuracy'
    error = clean(parallel.cross_val_score, model, x, y, devices=(0,), cv=3,
                  scoring=scoring, error=RuntimeError)
    assert 'native model worker failed' in str(error)
    record = trace()
    assert record['status'] == 'FAILED' and record['waves'] == [1]
    assert record['stages'][-1] == 'close'
    closed(record)
    # No abandoned worker or stale response poisons the next operation.
    np.testing.assert_array_equal(clean(parallel.cross_val_score, LinearRegression(), x, y, devices=(0,), cv=3),
                                  clean(serial.cross_val_score, LinearRegression(), x, y, cv=3))
    closed(trace())


def test_native_splitter_groups_and_string_labels(data):
    x, _ = data
    y = ['one' if i % 2 else 'two' for i in range(len(x))]
    groups = np.arange(len(x), dtype='i4') // 5
    model = GaussianNB()
    split = serial.GroupKFold(3)
    expected = clean(serial.cross_val_score, model, x, y, cv=split, groups=groups)
    actual = clean(parallel.cross_val_score, model, x, y, devices=(0,), cv=split, groups=groups)
    assert actual.tobytes() == expected.tobytes()
    closed(trace())


def test_classifier_native_schema_keeps_serial_and_parallel_results(data):
    x, y = data
    y = (x[:, 0] > x[:, 1]).astype('i4')
    model = GradientBoostingClassifier(n_estimators=3, max_depth=2, border_count=16,
                                       random_state=7, numeric_mode='identical')
    expected = clean(serial.cross_val_score, model, x, y, cv=3, scoring='accuracy')
    actual = clean(parallel.cross_val_score, model, x, y, devices=(0,), cv=3, scoring='accuracy')
    assert actual.tobytes() == expected.tobytes()
    assert not hasattr(model, '_learner_')
    closed(trace())


@pytest.mark.parametrize('devices', [(1,), (0, 1), (1, 0)])
def test_metal_requires_exactly_device_zero_before_launch(data, monkeypatch, devices):
    # Metadata-only refusal: no Metal operation or device claim is performed.
    meta = serial._NATIVE_CONTEXT['family_msel_parallel']
    monkeypatch.setitem(meta, 'vendor', 'metal')
    error = clean(parallel.cross_val_score, LinearRegression(), *data, devices=devices, error=NotImplementedError)
    assert 'devices=(0,)' in str(error)
    assert trace()['workers'] == []


def test_no_implicit_cpu_route(data, monkeypatch):
    meta = serial._NATIVE_CONTEXT['family_msel_parallel']
    monkeypatch.setitem(meta, 'vendor', 'cpu')
    monkeypatch.setitem(meta, 'cpu_only', False)
    clean(parallel.cross_val_score, LinearRegression(), *data, devices=(0,), error=NotImplementedError)
    assert trace()['workers'] == []


@pytest.mark.parametrize('fault', ['missing', 'extra', 'reversed', 'nonscalar'])
def test_incomplete_excess_unordered_or_nonscalar_batch_refused(fault):
    result = [(0, 1.0, None), (1, 2.0, None)]
    if fault == 'missing': result.pop()
    elif fault == 'extra': result.append((2, 3.0, None))
    elif fault == 'reversed': result.reverse()
    else: result[0] = (0, [1.0], None)
    clean(entry, 'family_msel_parallel_require_batch', (result, 0, 2), error=(ValueError, TypeError))


def test_estimator_hash_hooks_refused_before_constructor_container_clone(monkeypatch):
    model = LinearRegression()
    calls = []
    monkeypatch.setattr(LinearRegression, '__hash__', lambda self: calls.append('hash') or 1)
    clean(entry, 'family_msel_parallel_wire', ([model],), error=TypeError)
    assert calls == []


@pytest.mark.parametrize('classifier', [False, True])
def test_native_completed_capture_retains_model_prediction_and_replay_bytes(data, tmp_path, classifier):
    import io
    from mojolearn._parallel_cv_witness import retain_completed
    x, y = data
    if classifier:
        y = (x[:, 0] > x[:, 1]).astype('i4')
    cls = GradientBoostingClassifier if classifier else GradientBoostingRegressor
    model = cls(n_estimators=3, max_depth=2, border_count=16, random_state=7, numeric_mode='identical')
    scores, completed = clean(parallel._capture, model, x, y, devices=(0,), cv=3)
    expected = clean(serial.cross_val_score, model, x, y, cv=3)
    assert scores.tobytes() == expected.tobytes()
    assert len(completed) == 3
    for value in completed:
        assert value['archive'].startswith(b'PK')
        with np.load(io.BytesIO(value['archive']), allow_pickle=False) as archive:
            assert archive.files
        np.testing.assert_array_equal(value['raw_predict'], value['reload_predict'])
        assert set(value['native_artifacts']) == {'cv', 'trees', 'archive', 'gbdt'}
    records = retain_completed(completed, directory=tmp_path, vendor=_backend.vendor())
    assert len(records) == 3
    assert len(list((tmp_path / 'models').glob('*.npz'))) == 3
    assert len(list((tmp_path / 'arrays').glob('*.npy'))) == 18
    closed(trace())
