#!/usr/bin/env python3
"""Cloud-only QN feature partitions: coefficients, objective, iterations and predictions."""
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
    from mojolearn import LogisticRegression
    from mojolearn.parallel_classical import fit_logistic
    corpus = args.corpus.read_bytes()
    raw = np.frombuffer(corpus, dtype=np.uint8)
    checks, hashes = [], []
    for columns in (7, 137):
        X = raw[:257 * columns].astype('<f4').reshape(257, columns) / np.float32(255)
        for classes in (2, 3, 5):
            y = raw[50000:50257].astype('<i4') % classes
            penalties = ('l1', 'l2', 'elasticnet', None) if classes == 2 else ('l2', None)
            for penalty in penalties:
                params = dict(penalty=penalty, fit_intercept=penalty is not None, max_iter=5,
                              linesearch_max_iter=10, numeric_mode='identical')
                if penalty == 'elasticnet':
                    params['l1_ratio'] = 0.4
                serial = LogisticRegression(**params).fit(X, y)
                parallel = fit_logistic(LogisticRegression(**params), X, y, devices=(0, 1))
                digest = hashlib.sha256()
                for name in ('_w', 'coef_', 'intercept_', 'n_iter_', 'objective_', 'retcode_'):
                    a, b = getattr(serial, name), getattr(parallel, name)
                    if hasattr(a, 'tobytes'):
                        assert a.shape == b.shape
                        a, b = a.tobytes(), b.tobytes()
                    else:
                        a, b = struct.pack('<d', a), struct.pack('<d', b)
                    assert a == b, (columns, classes, penalty, name)
                    digest.update(name.encode() + b)
                assert serial.classes_ == parallel.classes_
                for method in ('decision_function', 'predict_proba', 'predict'):
                    a, b = getattr(serial, method)(X), getattr(parallel, method)(X)
                    assert a.shape == b.shape and a.tobytes() == b.tobytes(), (columns, classes, penalty, method)
                    digest.update(method.encode() + b.tobytes())
                before = parallel._w.tobytes()
                try:
                    fit_logistic(parallel, X, y[:-1], devices=(0, 1))
                except Exception:
                    pass
                else:
                    raise AssertionError('invalid targets accepted')
                assert parallel._w.tobytes() == before, 'failed fit mutated owner'
                checks.append([columns, classes, params])
                hashes.append(digest.hexdigest())
                print('PASS', columns, classes, penalty, flush=True)
    args.report.write_text(json.dumps(dict(status='PASS', checks=checks, hashes=hashes,
        corpus_sha256=hashlib.sha256(corpus).hexdigest(),
        scope='Two H100s; binary/multinomial QN; admitted penalties; five iterations; no root-memory pooling or cross-vendor claim'), indent=2) + '\n')


if __name__ == '__main__':
    main()
