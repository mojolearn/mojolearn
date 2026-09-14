#!/usr/bin/env python3
"""Cloud-only RF/ET pooled prediction bits against the same parallel-groves engine."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cloud', action='store_true', required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('RUNPOD_POD_ID'):
        raise SystemExit('RunPod required; no local execution')
    os.environ['MOJOLEARN_FOREST_DEVICE_COUNT'] = '1'

    import numpy as np
    from mojolearn.randomforest import RandomForestClassifier, RandomForestRegressor
    from mojolearn.extratrees import ExtraTreesClassifier, ExtraTreesRegressor
    from mojolearn.parallel_ensemble import ParallelForestPredictor

    rng = np.random.default_rng(736812)
    X = rng.normal(size=(96, 5)).astype('<f4')
    query = rng.normal(size=(131, 5)).astype('<f4')
    classifier_types = (RandomForestClassifier, ExtraTreesClassifier)
    estimator_types = classifier_types + (RandomForestRegressor, ExtraTreesRegressor)
    groups = []
    refusals = []

    def digest(value):
        if isinstance(value, list):
            raw = json.dumps(value, ensure_ascii=True, separators=(',', ':')).encode()
            return dict(shape=[len(value)], dtype='labels', sha256=hashlib.sha256(raw).hexdigest())
        value = np.asarray(value)
        return dict(shape=list(value.shape), dtype=value.dtype.str,
                    sha256=hashlib.sha256(value.tobytes()).hexdigest())

    def same(left, right):
        assert digest(left) == digest(right), (digest(left), digest(right))

    def refuse(name, function, exceptions=(ValueError, TypeError, RuntimeError)):
        try:
            function()
        except exceptions:
            refusals.append(name)
        else:
            raise AssertionError('expected refusal: ' + name)

    def labels(classes):
        if classes == 2:
            values = ['class-z', 'class-a']
        elif classes == 3:
            values = [-(2**40), 7, 2**40 + 3]
        elif classes == 8:
            values = [i + .25 for i in range(classes)]
        else:
            values = ['label-%02d' % i for i in range(classes)]
        return [values[i % classes] for i in range(len(X))]

    # Every estimator sees the 32-grove boundary; classifiers additionally
    # cover binary, small-vector, full-vector and scalar-wide output paths.
    for cls in estimator_types:
        classification = cls in classifier_types
        for index, trees in enumerate((31, 32, 33, 65)):
            classes = (2, 3, 8, 9)[index] if classification else 1
            params = dict(n_estimators=trees, max_depth=3, max_features=1.0,
                          random_state=819, inference_engine='parallel_groves',
                          numeric_mode='identical')
            if cls in (RandomForestClassifier, RandomForestRegressor):
                params.update(n_bins=16, n_streams=1)
            model = cls(**params)
            y = labels(classes) if classification else (X[:,0] * np.float32(.25) - X[:,2]).astype('<f4')
            model.fit(X, y)
            methods = ('predict', 'predict_proba') if classification else ('predict',)
            expected_dtype = '<f8' if cls in (ExtraTreesClassifier, ExtraTreesRegressor) else '<f4'
            expected = {method: getattr(model, method)(query) for method in methods}
            if classification:
                assert np.asarray(expected['predict_proba']).dtype.str == expected_dtype
                assert expected['predict_proba'].shape == (len(query), classes)
            else:
                assert np.asarray(expected['predict']).dtype.str == expected_dtype
            receipts = []
            devices = (1, 0) if trees == 65 else (0, 1)
            predictor = ParallelForestPredictor(model, devices=devices)
            with predictor:
                worker_pid = predictor._pool._workers[0].pid
                assert predictor.n_features_in_ == X.shape[1]
                assert predictor.n_estimators_ == trees
                if classification:
                    same(predictor.classes_, model.classes_)
                for rows in (1, 33, 131, 7, 131):
                    for method in methods:
                        left = getattr(model, method)(query[:rows])
                        right = getattr(predictor, method)(query[:rows])
                        same(left, right)
                        receipts.append(dict(method=method, rows=rows, **digest(right)))
                assert predictor._pool._workers[0].pid == worker_pid
                refuse('shape-' + cls.__name__, lambda: predictor.predict(query[:, :4]))
                refuse('method-' + cls.__name__, lambda: predictor._predict(query, 'fit'))
                if not classification:
                    refuse('proba-' + cls.__name__, lambda: predictor.predict_proba(query))
                same(predictor.predict(query), expected['predict'])
                # Query shape/layout changes must not recreate the snapshot.
                same(predictor.predict(query[::2]), model.predict(query[::2]))
                empty = np.empty((0, X.shape[1]), dtype='<f4')
                empty_outcomes = []
                for method in methods:
                    try:
                        baseline = getattr(model, method)(empty)
                    except (ValueError, RuntimeError):
                        refuse('empty-' + cls.__name__, lambda: getattr(predictor, method)(empty))
                        empty_outcomes.append(dict(method=method, status='matching_refusal'))
                        break  # Failed RPC correctly closes its snapshot worker.
                    else:
                        actual = getattr(predictor, method)(empty)
                        same(baseline, actual)
                        empty_outcomes.append(dict(method=method, status='PASS', **digest(actual)))
            predictor.close()
            assert not predictor._pool._workers
            refuse('closed-' + cls.__name__, lambda: predictor.predict(query))
            refuse('reenter-' + cls.__name__, predictor.__enter__)
            if trees == 65:
                # Explicit K=1 adapter preserves the original engine as well.
                with ParallelForestPredictor(model, devices=(0,)) as one:
                    for method in methods:
                        same(getattr(one, method)(query), expected[method])
                # Refitting/reconfiguring the original cannot change a snapshot.
                with ParallelForestPredictor(model, devices=(0, 1)) as frozen:
                    model.set_params(n_estimators=1)
                    for method in methods:
                        same(getattr(frozen, method)(query), expected[method])
            groups.append(dict(estimator=cls.__name__, trees=trees, classes=classes,
                               devices=list(devices), outputs=receipts, empty=empty_outcomes))

    refuse('unsupported', lambda: ParallelForestPredictor(object()))
    refuse('unfitted', lambda: ParallelForestPredictor(RandomForestRegressor(
        inference_engine='parallel_groves', numeric_mode='identical')))
    fitted = RandomForestRegressor(n_estimators=3, max_depth=2, n_bins=8,
        n_streams=1, inference_engine='parallel_groves', numeric_mode='identical').fit(X, X[:,0])
    for devices in ((), (0,0), (-1,), (True,), tuple(range(65))):
        refuse('devices-' + repr(devices), lambda: ParallelForestPredictor(fitted, devices=devices))
    fitted.inference_engine = 'sequential'
    refuse('sequential_engine', lambda: ParallelForestPredictor(fitted))
    fitted.inference_engine = 'parallel_groves'
    fitted.numeric_mode = 'fast'
    refuse('nonidentical_mode', lambda: ParallelForestPredictor(fitted))
    fitted.numeric_mode = 'identical'
    failed = ParallelForestPredictor(fitted)
    bad = query.copy()
    bad[0,0] = np.nan
    refuse('native_nonfinite', lambda: failed.predict(bad))
    refuse('failure_does_not_restart', lambda: failed.predict(query))
    assert failed._closed and not failed._pool._workers
    failed.close()

    args.report.write_text(json.dumps(dict(status='PASS', groups=groups,
        refusals=refusals, scope='RF/ET pooled inference across two GPUs versus single-GPU parallel_groves; exact original output widths, label decoding and 32-grove fold. No fit pooling or capacity measurement.'), indent=2)+'\n')
    print('PASS', len(groups), 'RF/ET configurations, persistent snapshots and refusals')


if __name__ == '__main__':
    main()
