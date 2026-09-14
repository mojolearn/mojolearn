#!/usr/bin/env python3
"""Cloud-only KernelRidge, Nystroem and RBFSampler multi-GPU drivers versus one device."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cloud', required=True, action='store_true')
    p.add_argument('--report', required=True, type=Path)
    p.add_argument('--devices', default='0,1')
    args = p.parse_args()
    if not os.environ.get('RUNPOD_POD_ID'):
        raise SystemExit('RunPod required; no local execution')
    for name in ('MOJOLEARN_SVM_DEVICE_COUNT', 'MOJOLEARN_CHOLESKY_DEVICE_COUNT'):
        os.environ.pop(name, None)
    devices = tuple(int(v) for v in args.devices.split(','))
    import numpy as np
    from mojolearn.kernel_methods import KernelRidge, Nystroem, RBFSampler
    from mojolearn.parallel_classical import (
        fit_kernel_method, apply_kernel_method, transform_rbf_sampler)

    checks, refusals = [], []

    def same(a, b, what):
        a, b = np.asarray(a), np.asarray(b)
        assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), what
        return hashlib.sha256(b.tobytes()).hexdigest()

    def refuse(name, function):
        try:
            function()
        except (ValueError, TypeError, RuntimeError, ImportError):
            refusals.append(name)
        else:
            raise AssertionError('expected refusal: ' + name)

    def identical(model):
        model.numeric_mode = 'identical'
        return model

    g = np.random.default_rng(4401)
    for n, d, t in ((1, 3, 1), (37, 4, 1), (129, 6, 3), (515, 9, 2)):
        X = g.normal(size=(n, d)).astype('<f4')
        y = g.normal(size=(n, t) if t > 1 else (n,)).astype('<f4')
        Q = g.normal(size=(n + 17, d)).astype('<f4')
        for kernel, extra in (('linear', {}), ('rbf', dict(gamma=0.3)), ('poly', dict(degree=2, gamma=0.2, coef0=0.5)),
                              ('sigmoid', dict(gamma=0.05, coef0=0.1))):
            params = dict(alpha=0.5, kernel=kernel, **extra)
            outcomes = []
            for fit in (lambda: identical(KernelRidge(**params)).fit(X, y),
                        lambda: fit_kernel_method(identical(KernelRidge(**params)), X, y, devices=devices)):
                try:
                    outcomes.append(fit())
                except Exception as exc:
                    # The refusal sentence, without the worker traceback.
                    text = str(exc)
                    outcomes.append(next(line for line in reversed(text.splitlines()) if 'kernel_ridge' in line))
            if isinstance(outcomes[0], str) or isinstance(outcomes[1], str):
                assert outcomes[0] == outcomes[1] or (isinstance(outcomes[0], str) and isinstance(outcomes[1], str)
                                                      and outcomes[0].split('Exception: ')[-1] in outcomes[1]), \
                    ('KernelRidge outcome differs', n, kernel, str(outcomes[0])[:200], str(outcomes[1])[:200])
                checks.append(dict(estimator='KernelRidge', n=n, d=d, targets=t, kernel=kernel,
                                   refusal=outcomes[0].split('Exception: ')[-1][:160]))
                print('PASS KernelRidge equal refusal', n, d, t, kernel, flush=True)
            else:
                one, many = outcomes
                digests = dict(dual=same(one.dual_coef_, many.dual_coef_, (n, kernel, 'dual')))
                assert one.info_ == many.info_ == 0
                digests['predict'] = same(one.predict(Q), apply_kernel_method(many, Q, devices=devices), (n, kernel, 'predict'))
                checks.append(dict(estimator='KernelRidge', n=n, d=d, targets=t, kernel=kernel, **digests))
                print('PASS KernelRidge', n, d, t, kernel, flush=True)
            if n >= 37:
                q = min(n, 64 if n > 64 else n - 5)
                nparams = dict(kernel=kernel, n_components=q, random_state=11, **extra)
                one = identical(Nystroem(**nparams)).fit(X)
                many = fit_kernel_method(identical(Nystroem(**nparams)), X, devices=devices)
                digests = {}
                for name in ('components_', 'component_indices_', 'normalization_', 'eigenvalues_'):
                    digests[name] = same(getattr(one, name), getattr(many, name), (n, kernel, name))
                digests['transform'] = same(one.transform(Q), apply_kernel_method(many, Q, devices=devices), (n, kernel, 'transform'))
                checks.append(dict(estimator='Nystroem', n=n, d=d, n_components=q, kernel=kernel, **digests))
                print('PASS Nystroem', n, d, q, kernel, flush=True)

    for d, q, rows, shard in ((3, 5, 1, 1), (8, 100, 1000, 128), (17, 257, 4097, 1000), (2, 64, 333, 4096)):
        sampler = identical(RBFSampler(gamma=0.7, n_components=q, random_state=d * 31)).fit(np.zeros((1, d), '<f4'))
        X = g.normal(size=(rows, d)).astype('<f4')
        expected = sampler.transform(X)
        actual = transform_rbf_sampler(sampler, X, devices=devices, rows_per_shard=shard)
        digest = same(expected, actual, ('RBFSampler', d, q, rows))
        checks.append(dict(estimator='RBFSampler', d=d, n_components=q, rows=rows, rows_per_shard=shard, transform=digest))
        print('PASS RBFSampler', d, q, rows, shard, flush=True)

    X = g.normal(size=(20, 3)).astype('<f4')
    refuse('laplacian', lambda: fit_kernel_method(identical(KernelRidge(kernel='laplacian')), X, X[:, 0], devices=devices))
    refuse('wrong_type', lambda: fit_kernel_method(object(), X, devices=devices))
    fast = KernelRidge()
    fast.numeric_mode = 'fast'
    refuse('nonidentical', lambda: fit_kernel_method(fast, X, X[:, 0], devices=devices))
    refuse('ridge_without_y', lambda: fit_kernel_method(identical(KernelRidge()), X, devices=devices))
    before = identical(KernelRidge(alpha=0.5)).fit(X, X[:, 0])
    snapshot = np.asarray(before.dual_coef_).tobytes()
    refuse('bad_targets', lambda: fit_kernel_method(before, X, X[:-1, 0], devices=devices))
    assert np.asarray(before.dual_coef_).tobytes() == snapshot, 'failed fit published state'
    refuse('unfitted_sampler', lambda: transform_rbf_sampler(identical(RBFSampler()), X, devices=devices))
    sampler = identical(RBFSampler(n_components=4)).fit(X)
    refuse('sampler_shape', lambda: transform_rbf_sampler(sampler, X[:, :2], devices=devices))
    refuse('sampler_shard', lambda: transform_rbf_sampler(sampler, X, devices=devices, rows_per_shard=0))

    args.report.write_text(json.dumps(dict(status='PASS', devices=list(devices), checks=checks, refusals=refusals,
        scope='KernelRidge/Nystroem kernel rows (SVM seam) plus KernelRidge Cholesky rows and target columns; '
              'RBFSampler whole query rows on separate GPUs; versus one device; no speed claim'), indent=2) + '\n')
    print('PASS', len(checks), 'kernel method configurations and', len(refusals), 'refusals')


if __name__ == '__main__':
    main()
