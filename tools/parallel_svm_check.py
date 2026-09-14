#!/usr/bin/env python3
"""Cloud-only SVC/SVR full fitted-state and prediction equality."""
import argparse
import hashlib
import json
import os
import struct
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cloud', required=True, action='store_true')
    p.add_argument('--corpus', required=True, type=Path)
    p.add_argument('--report', required=True, type=Path)
    args = p.parse_args()
    if not os.environ.get('RUNPOD_POD_ID'):
        raise SystemExit('RunPod required; no local execution')
    import numpy as np
    from mojolearn import SVC, SVR
    from mojolearn.parallel_classical import fit_svm, predict_svm
    corpus = args.corpus.read_bytes()
    raw = np.frombuffer(corpus, dtype=np.uint8)
    checks = []
    for columns in (7, 137):
        X = raw[:65 * columns].astype('<f4').reshape(65, columns) / np.float32(255)
        X[3] = X[2]  # a distance/split tie
        for cls in (SVC, SVR):
            y = raw[20000:20065].astype('<f4')
            y = (y.astype('<i4') % 2) if cls is SVC else y / np.float32(255)
            for kernel in ('linear', 'rbf'):
                params = dict(kernel=kernel, gamma='scale', C=0.7, max_iter=5,
                              cache_size=0, numeric_mode='identical')
                one = cls(**params).fit(X, y)
                many = fit_svm(cls(**params), X, y, devices=(0, 1))
                digest = hashlib.sha256()
                for name in ('support_', 'support_vectors_', 'dual_coef_', 'intercept_',
                             'n_support_', 'n_iter_', 'n_features_in_', '_gamma'):
                    a, b = getattr(one, name), getattr(many, name)
                    if hasattr(a, 'tobytes'):
                        assert a.shape == b.shape
                        a, b = a.tobytes(), b.tobytes()
                    else:
                        a, b = struct.pack('<d', a), struct.pack('<d', b)
                    assert a == b, (columns, cls.__name__, kernel, name)
                    digest.update(name.encode() + b)
                if cls is SVC:
                    assert one.classes_ == many.classes_
                for method in (('predict', 'decision_function') if cls is SVC else ('predict',)):
                    a = getattr(one, method)(X[::3])
                    b = predict_svm(many, X[::3], devices=(0, 1), method=method)
                    assert a.shape == b.shape and a.tobytes() == b.tobytes(), (columns, cls.__name__, kernel, method)
                    digest.update(method.encode() + b.tobytes())
                before = many.dual_coef_.tobytes()
                try:
                    fit_svm(many, X, y[:-1], devices=(0, 1))
                except Exception:
                    pass
                else:
                    raise AssertionError('invalid target length accepted')
                assert many.dual_coef_.tobytes() == before
                checks.append(dict(features=columns, estimator=cls.__name__, params=params,
                                   sha256=digest.hexdigest()))
                print('PASS', columns, cls.__name__, kernel, flush=True)
    args.report.write_text(json.dumps(dict(status='PASS', checks=checks,
        corpus_sha256=hashlib.sha256(corpus).hexdigest(),
        scope='Two H100s; binary SVC and epsilon SVR, linear/RBF, five iterations, kernel rows; full root state remains'), indent=2) + '\n')


if __name__ == '__main__':
    main()
