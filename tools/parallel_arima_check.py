#!/usr/bin/env python3
"""Cloud-only ARIMA series partition and transactional publication gate."""
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
    from mojolearn import ARIMA
    from mojolearn.parallel_classical import fit_arima
    data = args.corpus.read_bytes()
    y = np.frombuffer(data[:5 * 96], dtype=np.uint8).astype('<f4').reshape(5, 96) / np.float32(255)
    checks, hashes = [], []
    for order in ((1, 0, 0), (0, 0, 1), (1, 0, 1), (1, 1, 0)):
        options = dict(order=order, maxiter=20, numeric_mode='identical')
        serial = ARIMA(**options).fit(y)
        for width in (1, 2, 3):
            parallel = fit_arima(ARIMA(**options), y, devices=(0, 1), series_per_shard=width)
            digest = hashlib.sha256()
            for name in ('params_', 'x_', 'x0_', 'n_iter_', 'retcode_', 'llf_', 'fx_', 'aic_', 'bic_'):
                expected, actual = getattr(serial, name), getattr(parallel, name)
                assert expected.shape == actual.shape and expected.tobytes() == actual.tobytes(), (order, width, name)
                digest.update(name.encode() + actual.tobytes())
            forecast = parallel.forecast(5)
            assert serial.forecast(5).tobytes() == forecast.tobytes(), (order, width, 'forecast')
            digest.update(forecast.tobytes())
            before = parallel.params_.tobytes()
            bad = y.copy()
            bad[-1, -1] = np.nan
            try:
                fit_arima(parallel, bad, devices=(0, 1), series_per_shard=width)
            except Exception:
                pass
            else:
                raise AssertionError('invalid series accepted')
            assert parallel.params_.tobytes() == before, 'failed fit mutated owner'
            checks.append([order, width])
            hashes.append(digest.hexdigest())
            print('PASS', order, width, flush=True)
    args.report.write_text(json.dumps(dict(status='PASS', checks=checks, hashes=hashes,
        corpus_sha256=hashlib.sha256(data).hexdigest(),
        scope='Two GPUs, five independent series; exact fit and forecasts; no large-memory or cross-vendor qualification'), indent=2) + '\n')


if __name__ == '__main__':
    main()
