#!/usr/bin/env python3
"""Cloud-only GaussianMixture row-sharded E-step: fitted state and predictions vs one device."""
import argparse
import hashlib
import json
import os
import struct
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cloud', required=True, action='store_true')
    p.add_argument('--report', required=True, type=Path)
    p.add_argument('--devices', default='0,1')
    args = p.parse_args()
    if not os.environ.get('RUNPOD_POD_ID'):
        raise SystemExit('RunPod required; no local execution')
    os.environ.pop('MOJOLEARN_GMM_DEVICE_COUNT', None)
    os.environ.pop('MOJOLEARN_KMEANS_DEVICE_COUNT', None)
    devices = tuple(int(v) for v in args.devices.split(','))
    import numpy as np
    from mojolearn import GaussianMixture
    from mojolearn.parallel_classical import fit_gaussian_mixture, predict_gaussian_mixture

    names = ('weights_', 'means_', 'covariances_', 'precisions_cholesky_', 'log_det_chol_')
    rng = np.random.default_rng(90210)
    checks, refusals = [], []

    def blobs(n, d, k, seed):
        g = np.random.default_rng(seed)
        centers = g.normal(scale=6.0, size=(k, d))
        return (centers[np.arange(n) % k] + g.normal(size=(n, d))).astype('<f4')

    def digest_model(model):
        h = hashlib.sha256()
        for name in names:
            h.update(name.encode() + np.asarray(getattr(model, name)).tobytes())
        h.update(struct.pack('<ii', model.n_iter_, int(model.converged_)))
        h.update(struct.pack('<d', model.lower_bound_))
        return h.hexdigest()

    def refuse(name, function):
        try:
            function()
        except (ValueError, TypeError, RuntimeError, ImportError):
            refusals.append(name)
        else:
            raise AssertionError('expected refusal: ' + name)

    cases = [
        (37, 1, 2, 'kmeans', 100, 1e-3), (37, 1, 2, 'random', 100, 1e-3),
        (129, 3, 1, 'kmeans', 100, 1e-3), (257, 3, 3, 'random', 5, 1e-3),
        (600, 4, 4, 'kmeans', 100, 1e-4), (600, 4, 4, 'random', 100, 1e-4),
        (2003, 6, 4, 'kmeans', 100, 1e-3), (1001, 8, 5, 'random', 0, 1e-3),
        (4097, 2, 3, 'kmeans', 50, 0.0),
    ]
    for index, (n, d, k, init, max_iter, tol) in enumerate(cases):
        X = blobs(n, d, k, 1000 + index)
        params = dict(n_components=k, init_params=init, max_iter=max_iter, tol=tol,
                      reg_covar=1e-6, random_state=17 + index)
        one = GaussianMixture(**params)
        one.numeric_mode = 'identical'
        one.fit(X)
        many = GaussianMixture(**params)
        many.numeric_mode = 'identical'
        fit_gaussian_mixture(many, X, devices=devices)
        for name in names:
            a, b = np.asarray(getattr(one, name)), np.asarray(getattr(many, name))
            assert a.shape == b.shape and a.tobytes() == b.tobytes(), (index, name)
        assert (one.n_iter_, one.converged_) == (many.n_iter_, many.converged_), index
        assert struct.pack('<d', one.lower_bound_) == struct.pack('<d', many.lower_bound_), index
        query = np.concatenate([X[::3], X[:1] * np.float32(40.0)]).astype('<f4')
        outputs = {}
        for method in ('score_samples', 'predict_proba', 'predict'):
            a = np.asarray(getattr(one, method)(query))
            b = np.asarray(predict_gaussian_mixture(many, query, devices=devices, method=method))
            assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), (index, method)
            outputs[method] = hashlib.sha256(b.tobytes()).hexdigest()
        checks.append(dict(n=n, d=d, k=k, init=init, max_iter=max_iter, tol=tol,
                           n_iter=int(many.n_iter_), converged=bool(many.converged_),
                           model_sha256=digest_model(many), outputs=outputs))
        print('PASS', n, d, k, init, max_iter, tol, 'n_iter', many.n_iter_, flush=True)

    # A collapse must refuse on both paths with the same message, and a failed
    # distributed fit must not publish into the supplied estimator.
    X = np.concatenate([blobs(40, 2, 2, 5), np.repeat(np.float32([[9.0, 9.0]]), 6, axis=0)]).astype('<f4')
    params = dict(n_components=3, init_params='kmeans', reg_covar=0.0, random_state=3)
    messages = []
    one = GaussianMixture(**params)
    one.numeric_mode = 'identical'
    try:
        one.fit(X)
        messages.append('fit')
    except Exception as exc:
        messages.append(str(exc).splitlines()[-1])
    fitted = GaussianMixture(n_components=2, random_state=1)
    fitted.numeric_mode = 'identical'
    fit_gaussian_mixture(fitted, blobs(64, 2, 2, 8), devices=devices)
    before = digest_model(fitted)
    fitted.n_components, fitted.reg_covar, fitted.random_state = 3, 0.0, 3
    try:
        fit_gaussian_mixture(fitted, X, devices=devices)
        messages.append('fit')
    except Exception as exc:
        text = str(exc)
        messages.append(next((line for line in reversed(text.splitlines()) if 'GaussianMixture' in line), text))
    assert digest_model(fitted) == before, 'failed fit published state'
    collapse = dict(one=messages[0][-240:], many=messages[1][-240:])
    if messages[0] == 'fit' or messages[1] == 'fit':
        assert messages[0] == messages[1] == 'fit', collapse
    else:
        assert 'FAILED AT' in messages[0] and messages[0] in messages[1], collapse
    refusals.append('collapse_same_outcome')

    refuse('wrong_type', lambda: fit_gaussian_mixture(object(), X, devices=devices))
    fast = GaussianMixture(n_components=2)
    fast.numeric_mode = 'fast'
    refuse('nonidentical', lambda: fit_gaussian_mixture(fast, X, devices=devices))
    diag = GaussianMixture(n_components=2, covariance_type='diag')
    diag.numeric_mode = 'identical'
    refuse('covariance_type_diag', lambda: fit_gaussian_mixture(diag, X, devices=devices))
    refuse('unfitted_predict', lambda: predict_gaussian_mixture(GaussianMixture(), X, devices=devices))
    refuse('bad_method', lambda: predict_gaussian_mixture(many, X[:, :8], devices=devices, method='fit'))
    refuse('duplicate_devices', lambda: fit_gaussian_mixture(GaussianMixture(), X, devices=(0, 0)))

    args.report.write_text(json.dumps(dict(status='PASS', devices=list(devices), checks=checks,
        collapse=collapse, refusals=refusals,
        scope='GaussianMixture full covariance, kmeans/random init; row-sharded E-steps versus one device; '
              'root M-step, Cholesky and convergence test; no pooled root memory or speed claim'), indent=2) + '\n')
    print('PASS', len(checks), 'GaussianMixture configurations and', len(refusals), 'refusals')


if __name__ == '__main__':
    main()
