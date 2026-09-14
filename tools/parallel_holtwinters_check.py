#!/usr/bin/env python3
"""Cloud-only independent-series Holt-Winters fit and component layout gate."""
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
    from mojolearn import ExponentialSmoothing
    from mojolearn.parallel_classical import fit_exponential_smoothing
    corpus = args.corpus.read_bytes()
    y = np.frombuffer(corpus[:5 * 64], dtype=np.uint8).astype('<f4').reshape(5, 64) / np.float32(255) + np.float32(1)
    checks, hashes = [], []
    for seasonal in ('additive', 'multiplicative'):
        params = dict(seasonal=seasonal, seasonal_periods=4, start_periods=2, ts_num=5)
        serial = ExponentialSmoothing(y, **params).fit()
        for width in (1, 2, 3):
            parallel = fit_exponential_smoothing(ExponentialSmoothing(y, **params), devices=(0, 1), series_per_shard=width)
            digest = hashlib.sha256()
            for name in ('_comps', 'level_', 'trend_', 'season_', 'sse_', 'alpha_', 'beta_', 'gamma_', 'n_iter_', 'criterion_'):
                a, b = getattr(serial, name), getattr(parallel, name)
                assert a.shape == b.shape and a.tobytes() == b.tobytes(), (seasonal, width, name)
                digest.update(name.encode() + b.tobytes())
            for index in (None, 0, 4):
                a, b = serial.forecast(5, index), parallel.forecast(5, index)
                assert a.shape == b.shape and a.tobytes() == b.tobytes(), (seasonal, width, 'forecast', index)
                digest.update(b.tobytes())
            before = parallel._comps.tobytes()
            bad = y.copy()
            bad[-1, -1] = np.nan
            parallel.endog = bad
            try:
                fit_exponential_smoothing(parallel, devices=(0, 1), series_per_shard=width)
            except Exception:
                pass
            else:
                raise AssertionError('invalid series accepted')
            assert parallel._comps.tobytes() == before, 'failed fit published state'
            checks.append([seasonal, width])
            hashes.append(digest.hexdigest())
            print('PASS', seasonal, width, flush=True)
    args.report.write_text(json.dumps(dict(status='PASS', checks=checks, hashes=hashes,
        corpus_sha256=hashlib.sha256(corpus).hexdigest(),
        scope='Two GPUs; five independent series; additive/multiplicative; no large-memory or cross-vendor qualification'), indent=2) + '\n')


if __name__ == '__main__':
    main()
