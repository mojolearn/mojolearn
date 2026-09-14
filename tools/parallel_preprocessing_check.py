#!/usr/bin/env python3
"""Cloud-only scaler column partitions: fit, transform, inverse and refusal."""
import argparse
import hashlib
import json
import os
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
    from mojolearn import StandardScaler, MinMaxScaler
    from mojolearn.parallel_preprocessing import fit_scaler, transform_scaler, _statistics
    corpus = args.corpus.read_bytes()
    X = np.frombuffer(corpus[:513 * 33], dtype=np.uint8).astype('<f4').reshape(513, 33)
    X[:, 0] = np.float32(1)
    X[:, 1] = np.float32(-0.0)
    X[:, 2] = np.float32(1e-40)
    cases = [(StandardScaler, dict(with_mean=m, with_std=s)) for m in (False, True) for s in (False, True)]
    cases += [(MinMaxScaler, dict(feature_range=r, clip=c)) for r in ((0, 1), (-2, 3)) for c in (False, True)]
    checks, hashes = [], []
    for cls, params in cases:
        params['numeric_mode'] = 'identical'
        serial = cls(**params).fit(X)
        for width in (7, 16):
            parallel = fit_scaler(cls(**params), X, devices=(0, 1), columns_per_shard=width)
            digest = hashlib.sha256()
            for name in _statistics(parallel):
                a, b = getattr(serial, name), getattr(parallel, name)
                if a is None:
                    assert b is None
                else:
                    assert a.tobytes() == b.tobytes(), (cls.__name__, params, width, name)
                    digest.update(name.encode() + b.tobytes())
            transformed = transform_scaler(parallel, X, devices=(0, 1), columns_per_shard=width)
            assert serial.transform(X).tobytes() == transformed.tobytes(), (cls.__name__, width, 'transform')
            inverse = transform_scaler(parallel, transformed, devices=(0, 1), columns_per_shard=width, inverse=True)
            assert serial.inverse_transform(transformed).tobytes() == inverse.tobytes(), (cls.__name__, width, 'inverse')
            digest.update(transformed.tobytes() + inverse.tobytes())
            before = parallel.__dict__.copy()
            bad = X.copy()
            bad[-1, -1] = np.nan
            try:
                fit_scaler(parallel, bad, devices=(0, 1))
            except ValueError:
                pass
            else:
                raise AssertionError('nonfinite input accepted')
            assert all(parallel.__dict__[key] is value for key, value in before.items()), 'refused fit published state'
            checks.append([cls.__name__, params, width])
            hashes.append(digest.hexdigest())
            print('PASS', cls.__name__, params, width, flush=True)
    args.report.write_text(json.dumps(dict(status='PASS', checks=checks, hashes=hashes,
        corpus_sha256=hashlib.sha256(corpus).hexdigest(),
        scope='Two GPUs; column-sharded statistics and transforms; no large-memory/cross-vendor qualification'), indent=2) + '\n')


if __name__ == '__main__':
    main()
