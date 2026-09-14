#!/usr/bin/env python3
"""Cloud-only GP covariance partition: full factor/dual/likelihood and predictions."""
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
    from mojolearn import GaussianProcessRegressor
    from mojolearn._gp_impl import ConstantKernel, WhiteKernel, RBF, Matern
    from mojolearn.parallel_classical import fit_gaussian_process, predict_gaussian_process
    corpus = args.corpus.read_bytes()
    raw = np.frombuffer(corpus, dtype=np.uint8)
    checks = []
    for rows in (5, 33):
        X = raw[:rows * 3].astype('<f4').reshape(rows, 3) / np.float32(255)
        X[2] = X[1]
        y = raw[30000:30000 + rows].astype('<f4') / np.float32(255)
        kernels = [RBF([0.7, 1.0, 1.3]), WhiteKernel(0.2),
                   ConstantKernel(0.8) * RBF(0.9) + WhiteKernel(0.1)]
        kernels += [Matern([0.7, 1.0, 1.3], nu=nu) + WhiteKernel(0.1) for nu in (0.5, 1.5, 2.5)]
        for kernel in kernels:
            params = dict(kernel=kernel, alpha=2.0 ** -20, numeric_mode='identical')
            one = GaussianProcessRegressor(**params).fit(X, y)
            many = fit_gaussian_process(GaussianProcessRegressor(**params), X, y, devices=(0, 1))
            assert one.info_ == many.info_ == 0
            digest = hashlib.sha256()
            for name in ('L_', 'alpha_', 'info_', 'nb_', 'log_marginal_likelihood_value_', '_logdet', '_ydotalpha'):
                a, b = getattr(one, name), getattr(many, name)
                if hasattr(a, 'tobytes'):
                    assert a.shape == b.shape
                    a, b = a.tobytes(), b.tobytes()
                else:
                    a, b = struct.pack('<d', a), struct.pack('<d', b)
                assert a == b, (rows, repr(kernel), name)
                digest.update(name.encode() + b)
            for return_std in (False, True):
                a = one.predict(X[::2], return_std=return_std)
                b = predict_gaussian_process(many, X[::2], devices=(0, 1), return_std=return_std)
                for x, z in zip(a if return_std else (a,), b if return_std else (b,)):
                    assert x.shape == z.shape and x.tobytes() == z.tobytes(), (rows, repr(kernel), 'prediction')
                    digest.update(z.tobytes())
                if return_std:
                    assert one.clamped_.tobytes() == many.clamped_.tobytes()
                    assert one.n_clamped_ == many.n_clamped_
            before = many.L_.tobytes()
            try:
                fit_gaussian_process(many, X, y[:-1], devices=(0, 1))
            except Exception:
                pass
            else:
                raise AssertionError('bad targets accepted')
            assert many.L_.tobytes() == before
            checks.append(dict(rows=rows, kernel=repr(kernel), sha256=digest.hexdigest()))
            print('PASS', rows, repr(kernel), flush=True)
    args.report.write_text(json.dumps(dict(status='PASS', checks=checks,
        corpus_sha256=hashlib.sha256(corpus).hexdigest(),
        scope='Two H100s; fixed kernels and alpha, covariance rows, original root Cholesky and variance; no pooled root memory'), indent=2) + '\n')


if __name__ == '__main__':
    main()
