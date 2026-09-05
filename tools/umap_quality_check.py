#!/usr/bin/env python3
"""Independent neighborhood-quality gate for two fixed UMAP input profiles.

Run in the main lane against an installed wheel. Requires numpy and
scikit-learn. Scores are not bitwise certificates or upstream layout parity.
Thresholds are fixed before the initial run: trustworthiness >= .90 and
at least .20 above a deterministic row-permutation sabotage control.
"""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import sklearn
from sklearn.manifold import trustworthiness

import mojolearn
from mojolearn import UMAP


def fixtures():
    # Dyadic arithmetic gives exact input bytes without trig or RNG libraries.
    t = (np.arange(64, dtype=np.float32) - 32) / 32
    curve = np.column_stack((t, t * t, t * t * t))
    i = np.arange(64, dtype=np.float32)
    u, v = (i % 8 - 3.5) / 4, (i // 8 - 3.5) / 4
    saddle = np.column_stack((u, v, (u * u - v * v) / 2))
    return [('cubic64x3', curve, 2, 19), ('saddle64x3', saddle, 3, 7)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['fast', 'deterministic', 'identical'], required=True)
    parser.add_argument('--device', required=True, help='Actual device, recorded by the operator')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    record = {'schema': 'umap.quality.v1', 'device': args.device,
              'platform': platform.platform(), 'mode': args.mode,
              'mojolearn_version': mojolearn.__version__,
              'package_file': mojolearn.__file__, 'numpy': np.__version__,
              'sklearn': sklearn.__version__,
              'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'metric': 'sklearn.manifold.trustworthiness, Euclidean, k=5',
              'threshold': .90, 'minimum_control_margin': .20,
              'scope': 'Named local quality fixtures; no cross-vendor or upstream-layout claim',
              'results': [], 'status': 'RUNNING'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name, x, dimensions, seed in fixtures():
        config = dict(n_neighbors=8, n_components=dimensions, n_epochs=200,
                      random_state=seed, min_dist=.1, spread=1.)
        before = x.copy()
        layout = UMAP(numeric_mode=args.mode, **config).fit_transform(x)
        np.testing.assert_array_equal(x, before)
        if layout.shape != (64, dimensions) or not np.isfinite(layout).all():
            raise RuntimeError('Invalid layout ' + name)
        score = float(trustworthiness(x.astype(np.float64), layout.astype(np.float64), n_neighbors=5))
        # 37 is coprime to 64: a fixed permutation destroys row correspondence.
        control = float(trustworthiness(x.astype(np.float64),
                        layout[(np.arange(64) * 37) % 64].astype(np.float64), n_neighbors=5))
        passed = score >= .90 and score - control >= .20
        record['results'].append({'profile': name, 'parameters': config,
            'input_uint32': x.view(np.uint32).tolist(),
            'layout_uint32': layout.view(np.uint32).tolist(),
            'trustworthiness': score, 'permuted_control': control,
            'passed': passed})
        args.output.write_text(json.dumps(record, indent=2) + '\n')
        print(name, args.mode, 'trustworthiness', score, 'control', control,
              'PASS' if passed else 'FAIL', flush=True)
    record['status'] = 'PASS' if all(r['passed'] for r in record['results']) else 'FAIL'
    args.output.write_text(json.dumps(record, indent=2) + '\n')
    return 0 if record['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
