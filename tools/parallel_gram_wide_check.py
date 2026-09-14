#!/usr/bin/env python3
"""Cloud-only full-estimator checks for distributed pinned Gram chunks."""
import argparse
import hashlib
import json
import os
import struct
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cloud', action='store_true', required=True)
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    args = p.parse_args()
    if not os.environ.get('RUNPOD_POD_ID'):
        raise SystemExit('RunPod required; no local execution')
    import numpy as np
    from mojolearn import LinearRegression, Ridge, PCA, TruncatedSVD
    from mojolearn.parallel_classical import fit_gram_estimator
    corpus = args.corpus.read_bytes()
    raw = np.frombuffer(corpus, dtype=np.uint8)
    checks, hashes = [], []
    for columns in (129, 257):
        X = raw[:259 * columns].astype('<f4').reshape(259, columns) / np.float32(255)
        y = raw[50000:50259].astype('<f4') / np.float32(255)
        weights = np.ones(259, dtype='<f4')
        weights[::7] = np.float32(0.5)
        cases = [(LinearRegression, dict(fit_intercept=False), y, {}),
                 (LinearRegression, dict(fit_intercept=True), y, dict(sample_weight=weights)),
                 (Ridge, dict(alpha=0.25, fit_intercept=True), y, {}),
                 (PCA, dict(n_components=3, svd_solver='covariance_eigh', whiten=True), None, {}),
                 (TruncatedSVD, dict(n_components=3), None, {})]
        for cls, params, target, kwargs in cases:
            params['numeric_mode'] = 'identical'
            serial = cls(**params).fit(X, target, **kwargs)
            parallel = fit_gram_estimator(cls(**params), X, target, devices=(0, 1), **kwargs)
            digest = hashlib.sha256()
            for name in ('coef_', '_x_mean', 'components_', 'mean_', 'explained_variance_',
                         'explained_variance_ratio_', 'singular_values_'):
                if hasattr(serial, name):
                    a, b = getattr(serial, name), getattr(parallel, name)
                    assert a.shape == b.shape and a.tobytes() == b.tobytes(), (cls.__name__, columns, name)
                    digest.update(name.encode() + b.tobytes())
            for name in ('intercept_', '_y_mean', 'noise_variance_'):
                if hasattr(serial, name):
                    a, b = struct.pack('<d', getattr(serial, name)), struct.pack('<d', getattr(parallel, name))
                    assert a == b, (cls.__name__, columns, name)
                    digest.update(name.encode() + b)
            method = 'predict' if target is not None else 'transform'
            output = getattr(parallel, method)(X)
            assert getattr(serial, method)(X).tobytes() == output.tobytes(), (cls.__name__, columns, method)
            digest.update(output.tobytes())
            checks.append([cls.__name__, columns, params])
            hashes.append(digest.hexdigest())
            print('PASS', cls.__name__, columns, params, flush=True)
    for rows, columns in ((3,7),(17,65),(17,129),(33,257)):
        X = raw[:rows*columns].astype('<f4').reshape(rows,columns)/np.float32(255)
        y = raw[50000:50000+rows].astype('<f4')/np.float32(255)
        for intercept in (False,True):
            params = dict(fit_intercept=intercept,numeric_mode='identical')
            serial = LinearRegression(**params).fit(X,y)
            parallel = fit_gram_estimator(LinearRegression(**params),X,y,devices=(0,1))
            assert serial.coef_.tobytes() == parallel.coef_.tobytes()
            assert struct.pack('<d',serial.intercept_) == struct.pack('<d',parallel.intercept_)
            assert serial.predict(X).tobytes() == parallel.predict(X).tobytes()
            checks.append(['wide-OLS',rows,columns,intercept])
            hashes.append(hashlib.sha256(parallel.coef_.tobytes()+parallel.predict(X).tobytes()).hexdigest())
            print('PASS wide OLS',rows,columns,intercept,flush=True)
    args.report.write_text(json.dumps(dict(status='PASS', checks=checks, hashes=hashes,
        corpus_sha256=hashlib.sha256(corpus).hexdigest(),
        scope='Two H100s; original v1 Gram output rows; 129/257 columns; no pooled root-memory or cross-vendor claim'), indent=2) + '\n')


if __name__ == '__main__':
    main()
