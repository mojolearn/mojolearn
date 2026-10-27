#!/usr/bin/env python3
"""Bounded SVC public A/B; root executes rebuilt local Mac, never a timing run."""
import argparse
import json
import os
from pathlib import Path
import traceback


def qualify():
    import numpy as np
    from mojolearn import SVC
    from mojolearn._verify_native_runtime import audit_native_call

    rng = np.random.default_rng(421)
    x = rng.normal(size=(36, 5)).astype('f4')
    labels = np.tile(np.arange(3, dtype='i4'), 12)
    x[:, 0] += labels.astype('f4') * np.float32(0.3)
    weights = np.linspace(0.5, 1.5, len(x), dtype='f4')
    results = []
    def equal(a, b, name):
        a, b = np.asarray(a), np.asarray(b)
        assert a.shape == b.shape and a.dtype == b.dtype, name
        assert a.tobytes() == b.tobytes(), name
    def call(arm, method, *args, **kwargs):
        os.environ['MOJOLEARN_CLASSICAL_NATIVE_DISPATCH'] = str(arm)
        return method(*args, **kwargs)
    for kernel, binary, probability in [('linear', True, False), ('rbf', False, True),
                                        ('poly', False, False), ('precomputed', False, True)]:
        data = (x @ x.T).astype('f4') if kernel == 'precomputed' else x
        y = (labels != 0).astype('i4') if binary else labels
        opts = dict(kernel=kernel, C=1.25, tol=1e-3, max_iter=64, nochange_steps=32,
                    probability=probability, random_state=19, numeric_mode='identical')
        if kernel == 'poly':
            opts.update(gamma=0.2, degree=2, coef0=0.1)
        elif kernel == 'rbf':
            opts.update(gamma='scale')
        models = [SVC(**opts), SVC(**opts)]
        fit_reports = []
        fit_counts = None
        for arm, model in enumerate(models):
            os.environ['MOJOLEARN_CLASSICAL_NATIVE_DISPATCH'] = str(arm)
            audit = audit_native_call(model.fit, data, y, sample_weight=weights)
            audit.assert_native()
            fit_reports.append(audit.report())
            if arm:
                name, fit_counts = vars(model)['_native_svc_dispatch']
                assert name == 'mojolearn.svm.binary.v1'
                assert fit_counts[0] >= (1 if binary else 3)
                if probability:
                    assert fit_counts[1] > 0  # held-out probability-fold machines
        for attr in ('classes_', 'support_', 'support_vectors_', 'dual_coef_', 'intercept_', 'n_iter_'):
            equal(getattr(models[0], attr), getattr(models[1], attr), attr)
        methods = ['decision_function', 'predict'] + (['predict_proba', 'predict_log_proba'] if probability else [])
        for method in methods:
            a = call(0, getattr(models[0], method), data)
            b = call(1, getattr(models[1], method), data)
            equal(a, b, method)
            name, calls = vars(models[1])['_native_svc_dispatch']
            assert name == 'mojolearn.svm.binary.v1' and calls[1] == (1 if binary else 3)
        if probability:
            for attr in ('probA_', 'probB_'):
                equal(getattr(models[0], attr), getattr(models[1], attr), attr)
        # New public fit must recapture changed settings, not reuse old metadata.
        for arm, model in enumerate(models):
            model.C = 0.75
            call(arm, model.fit, data, y, sample_weight=weights)
        equal(call(0, models[0].decision_function, data), call(1, models[1].decision_function, data), 'refit')
        for arm, model in enumerate(models):
            try:
                call(arm, model.predict, np.zeros((2, data.shape[1] + 1), dtype='f4'))
            except Exception:
                pass
            else:
                raise AssertionError('feature mismatch accepted')
            call(arm, model.predict, data)
        results.append(dict(kernel=kernel, binary=binary, probability=probability,
                            fit_calls=fit_counts, runtime_audits=fit_reports, status='passed'))
    return dict(status='passed', cases=results,
                pending=['all kernels/label types/class weights', 'concurrent input mutation',
                         'CPU binding candidate', 'SVR and other families', 'performance', 'other targets'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        report = qualify()
    except BaseException:
        report = dict(status='failed', traceback=traceback.format_exc())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(status=report['status'], evidence=str(args.output))))
    return int(report['status'] != 'passed')


if __name__ == '__main__':
    raise SystemExit(main())
