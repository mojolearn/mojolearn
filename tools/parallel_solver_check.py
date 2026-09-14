#!/usr/bin/env python3
"""Cloud-only coordinate-descent leaf partition equality."""
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
    from mojolearn import Lasso, ElasticNet
    from mojolearn.parallel_classical import fit_coordinate_descent
    corpus = args.corpus.read_bytes()
    raw = np.frombuffer(corpus, dtype=np.uint8)
    checks = []
    for rows in (127, 257, 1031):
        X = raw[:rows * 5].astype('<f4').reshape(rows, 5) / np.float32(255)
        X[:, 4] = 1
        y = raw[20000:20000 + rows].astype('<f4') / np.float32(255)
        for cls in (Lasso, ElasticNet):
            for intercept in (False, True):
                params = dict(alpha=0.01, fit_intercept=intercept, max_iter=5, tol=0)
                serial = cls(**params).fit(X, y)
                parallel = fit_coordinate_descent(cls(**params), X, y, devices=(0, 1))
                digest = hashlib.sha256()
                for name in ('coef_', 'intercept_', 'n_iter_'):
                    a, b = getattr(serial, name), getattr(parallel, name)
                    if hasattr(a, 'tobytes'):
                        assert a.shape == b.shape
                        a, b = a.tobytes(), b.tobytes()
                    else:
                        a, b = struct.pack('<d', a), struct.pack('<d', b)
                    assert a == b, (rows, cls.__name__, intercept, name)
                    digest.update(name.encode() + b)
                a, b = serial.predict(X), parallel.predict(X)
                assert a.shape == b.shape and a.tobytes() == b.tobytes()
                digest.update(b.tobytes())
                before = parallel.coef_.tobytes()
                try:
                    fit_coordinate_descent(parallel, X, y[:-1], devices=(0, 1))
                except Exception:
                    pass
                else:
                    raise AssertionError('invalid targets accepted')
                assert parallel.coef_.tobytes() == before
                checks.append(dict(rows=rows, estimator=cls.__name__, params=params, sha256=digest.hexdigest()))
                print('PASS', rows, cls.__name__, intercept, flush=True)
    args.report.write_text(json.dumps(dict(status='PASS', checks=checks,
        corpus_sha256=hashlib.sha256(corpus).hexdigest(),
        scope='Two H100s; cyclic coordinate descent, five iterations, one/odd dot leaves; root state is not pooled'), indent=2) + '\n')


if __name__ == '__main__':
    main()
