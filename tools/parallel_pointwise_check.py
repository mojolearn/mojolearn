#!/usr/bin/env python3
"""Cloud-only pointwise feature-group qualification over the pinned R2 corpus."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cloud', action='store_true', required=True)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('RUNPOD_POD_ID'):
        raise SystemExit('RunPod required; no local execution')
    import numpy as np
    from mojolearn import GradientBoosting
    from mojolearn.parallel_ensemble import fit_boosting
    corpus = args.corpus.read_bytes()
    raw = np.frombuffer(corpus, dtype=np.uint8)
    checks, hashes = [], []
    fixtures = [(513, 'SymmetricTree', border, 'RMSE') for border in (1, 7, 31, 63, 127, 254)]
    fixtures += [(8193, 'SymmetricTree', border, 'RMSE') for border in (1, 7)]
    fixtures += [(513, 'SymmetricTree', 31, 'Logloss')]
    for rows, policy, border, loss in fixtures:
        X = raw[:rows * 73].astype('<f4').reshape(rows, 73)
        # Mixed packing policies and an incomplete last feature group.
        X[:, :33] %= np.float32(2)
        X[:, 33:50] %= np.float32(8)
        y = raw[1000000:1000000 + rows].astype('<f4')
        if loss == 'Logloss':
            y %= np.float32(2)
        elif loss == 'MultiClass':
            y %= np.float32(3)
        else:
            y /= np.float32(255)
        weights = np.ones(rows, dtype='<f4')
        weights[::7] = np.float32(0)
        weights[::11] = np.float32(0.5)
        options = dict(loss=loss, n_estimators=3, max_depth=3, border_count=border,
                       grow_policy=policy, random_state=491,
                       random_strength=0.1 if policy == 'SymmetricTree' else 0.0,
                       numeric_mode='identical', bootstrap_type='No', use_pointwise_searcher=True)
        serial = GradientBoosting(**options).fit(X, y, sample_weight=weights)
        parallel = fit_boosting(GradientBoosting(**options), X, y,
                               devices=(0, 1), sample_weight=weights)
        fixture = [rows, policy, border, loss]
        assert serial.model_ == parallel.model_, (fixture, 'model')
        assert serial.loss_curve_.tobytes() == parallel.loss_curve_.tobytes(), (fixture, 'loss curve')
        prediction = parallel.predict(X)
        assert serial.predict(X).tobytes() == prediction.tobytes(), (fixture, 'predictions')
        old = parallel.model_
        try:
            fit_boosting(parallel, X, y[:-1], devices=(0, 1))
        except Exception:
            pass
        else:
            raise AssertionError('invalid target length accepted')
        assert parallel.model_ == old, 'refused fit mutated owner'
        checks.append(fixture)
        hashes.append(hashlib.sha256(parallel.model_.encode() + prediction.tobytes()
                                     + parallel.loss_curve_.tobytes()).hexdigest())
        print('PASS', fixture, flush=True)
    args.report.write_text(json.dumps(dict(status='PASS', checks=checks, hashes=hashes,
        corpus_sha256=hashlib.sha256(corpus).hexdigest(),
        scope='Two devices; pointwise packed feature groups, original row reductions; no pooled root-state capacity claim'), indent=2) + '\n')


if __name__ == '__main__':
    main()
