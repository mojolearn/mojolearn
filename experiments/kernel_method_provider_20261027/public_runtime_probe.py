#!/usr/bin/env python3
"""Bounded public bits/ownership probe; root runs compiled candidate on Mac."""
import argparse
import gc
import json
import os
from pathlib import Path
import traceback
import weakref


def qualify():
    import numpy as np
    from mojolearn import KernelRidge, Nystroem, RBFSampler
    from mojolearn import _backend, kernel_methods as api
    from mojolearn._verify_native_runtime import audit_native_call

    rng = np.random.default_rng(412)
    x = rng.normal(size=(18, 4)).astype('f4')
    query = rng.normal(size=(7, 4)).astype('f4')
    y = rng.normal(size=(18, 2)).astype('f4')
    sw = np.linspace(.7, 1.3, 18, dtype='f8')
    records, audits = [], []
    def same(a, b):
        a, b = np.asarray(a), np.asarray(b)
        assert a.dtype == b.dtype and a.shape == b.shape
        assert a.tobytes() == b.tobytes()
    def call(arm, model, method, *args, **kwargs):
        os.environ['MOJOLEARN_CLASSICAL_NATIVE_DISPATCH'] = str(arm)
        if arm:
            audit = audit_native_call(getattr(model, method), *args, **kwargs)
            audits.append(audit.report())
            return audit.assert_native()
        return getattr(model, method)(*args, **kwargs)
    def counts(model):
        name, calls = api._NATIVE(model, 'family_km_provider_info', (), api._NATIVE_CONTEXT)
        assert name == 'mojolearn.kernel_methods.native.v1'
        return list(calls)
    for kernel in ('linear', 'rbf', 'precomputed'):
        data = (x @ x.T).astype('f4') if kernel == 'precomputed' else x
        q = (query @ x.T).astype('f4') if kernel == 'precomputed' else query
        models = [KernelRidge(alpha=.7, kernel=kernel, gamma=.2, numeric_mode='identical') for _ in range(2)]
        for arm, model in enumerate(models):
            call(arm, model, 'fit', data, y, sample_weight=sw)
        same(models[0].dual_coef_, models[1].dual_coef_)
        for _ in range(2):
            before = counts(models[1])
            same(call(0, models[0], 'predict', q), call(1, models[1], 'predict', q))
            after = counts(models[1])
            assert after[13] - before[13] == 1 and after[14] - before[14] == 1
        # Same-shaped fitted-array replacement invalidates resident ownership.
        for model in models:
            model.dual_coef_ = np.asarray(model.dual_coef_).copy()
        same(call(0, models[0], 'predict', q), call(1, models[1], 'predict', q))
        for arm, model in enumerate(models):
            call(arm, model, 'fit', data, y * np.float32(.75), sample_weight=sw)
        same(call(0, models[0], 'predict', q), call(1, models[1], 'predict', q))
        records.append(dict(family='KernelRidge', kernel=kernel, calls=counts(models[1])))
    for cls, options in [(Nystroem, dict(kernel='rbf', gamma=.2)),
                         (RBFSampler, dict(gamma=.2)), (RBFSampler, dict(gamma='scale'))]:
        models = [cls(n_components=6, random_state=17, numeric_mode='identical', **options) for _ in range(2)]
        same(call(0, models[0], 'fit_transform', x), call(1, models[1], 'fit_transform', x))
        native_counts = counts(models[1])
        assert native_counts[13] == 1 and native_counts[14] == 1
        same(call(0, models[0], 'transform', query), call(1, models[1], 'transform', query))
        for arm, model in enumerate(models):
            model.gamma = .15
            call(arm, model, 'fit', x)
        same(call(0, models[0], 'transform', query), call(1, models[1], 'transform', query))
        records.append(dict(family=cls.__name__, gamma=options['gamma'], calls=counts(models[1])))

    binding = _backend.binding('_mojolearn_kernel_methods', 'identical')
    target = KernelRidge(alpha=.7, kernel='rbf', gamma=.2, numeric_mode='identical')
    rejected_calls = []
    def poison(*args, **kwargs):
        rejected_calls.append(True)
        raise AssertionError('Python loader/legacy computation callback reached')
    names = ('kernel_methods_numeric_mode', 'kernel_ridge_fit', 'kernel_ridge_fit_ptr',
             'kernel_ridge_predict', 'kernel_ridge_predict_ptr', 'kernel_ridge_resident_prepare',
             'kernel_ridge_predict_resident', 'kernel_ridge_resident_release')
    saved = {name:getattr(binding, name) for name in names if hasattr(binding, name)}
    loader, mode_hook = _backend.binding, _backend._ModeSet.__getattr__
    try:
        for name in saved:
            setattr(binding, name, poison)
        _backend.binding = poison
        _backend._ModeSet.__getattr__ = poison
        call(1, target, 'fit', x, y)
        call(1, target, 'predict', query)
        call(1, target, 'fit', x, y)
        call(1, target, 'predict', query)
        assert not rejected_calls
    finally:
        for name, entry in saved.items():
            setattr(binding, name, entry)
        _backend.binding, _backend._ModeSet.__getattr__ = loader, mode_hook
    original = binding.kernel_methods_api_v1
    try:
        binding.kernel_methods_api_v1 = poison
        try:
            call(1, target, 'predict', query)
        except TypeError:
            pass
        else:
            raise AssertionError('replaced Python provider factory was accepted')
        assert not rejected_calls
    finally:
        binding.kernel_methods_api_v1 = original
    call(1, target, 'predict', query)
    held = vars(target).get('_resident_holder', {}).get('index')
    if held is not None:
        assert len(held) == 5
        handle = held[1]
        del held
        weak = weakref.ref(target)
        del target
        gc.collect()
        assert weak() is None
        try:
            binding.kernel_ridge_resident_release(handle)
        except Exception as error:
            assert 'unknown or released' in str(error)
        else:
            raise AssertionError('native resource did not release resident handle')
    return dict(status='passed', cases=records, runtime_audits=audits,
                pending=['CPU providers', 'other kernels and exotic input types', 'score/parallel wrappers',
                         'all other estimator families', 'performance', 'other devices'])


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
    return report['status'] != 'passed'


if __name__ == '__main__':
    raise SystemExit(main())
