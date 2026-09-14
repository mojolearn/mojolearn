#!/usr/bin/env python3
"""Cloud-only DBSCAN labels, convergence counts and identity-stage equality."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cloud', required=True, action='store_true')
    parser.add_argument('--corpus', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    if not os.environ.get('RUNPOD_POD_ID'):
        raise SystemExit('RunPod required; no local execution')
    import numpy as np
    from mojolearn import DBSCAN
    from mojolearn.parallel_classical import fit_dbscan
    raw_bytes = args.corpus.read_bytes()
    raw = np.frombuffer(raw_bytes, dtype=np.uint8)
    checks = []
    trace_root = args.report.parent / 'traces'
    trace_root.mkdir(exist_ok=True)
    for n in (41, 1025):
        X = raw[:n*3].astype('<f4').reshape(n, 3) / np.float32(2048)
        X[1::2, 0] += np.float32(2)
        X[3] = X[2]
        X[-3:, 0] = np.array([7, 9, 11], dtype='<f4')
        for algorithm, metric in (('rbc', 'euclidean'), ('brute', 'euclidean'), ('brute', 'manhattan')):
            for weighting in ('none', 'uniform', 'signed'):
                weights = None if weighting == 'none' else np.ones(n, dtype='<f4')
                if weighting == 'signed':
                    weights[::7] = np.float32(-0.25)
                    weights[1::7] = np.float32(0)
                    weights[2::7] = np.float32(2)
                params = dict(eps=0.2, min_samples=5, algorithm=algorithm, metric=metric,
                              max_mbytes_per_batch=1, numeric_mode='identical')
                stem = f'{n}-{algorithm}-{metric}-{weighting}'
                one_path = trace_root / (stem + '-one.tsv')
                two_path = trace_root / (stem + '-two.tsv')
                os.environ['MOJOLEARN_IDENTITY_TRACE'] = str(one_path)
                one = DBSCAN(**params).fit(X, sample_weight=weights)
                os.environ['MOJOLEARN_IDENTITY_TRACE'] = str(two_path)
                two = fit_dbscan(DBSCAN(**params), X, sample_weight=weights, devices=(0, 1))
                os.environ.pop('MOJOLEARN_IDENTITY_TRACE')
                assert one.labels_.tobytes() == two.labels_.tobytes(), stem
                assert one.n_iter_ == two.n_iter_ and one.n_features_in_ == two.n_features_in_
                stages = lambda path: [s for s in path.read_text().splitlines() if s and not s.startswith('#')]
                one_stages, two_stages = stages(one_path), stages(two_path)
                assert len(one_stages) == 3 and one_stages == two_stages, (stem, one_stages, two_stages)
                before = two.labels_.tobytes()
                try:
                    fit_dbscan(two, X, sample_weight=np.ones(n-1, dtype='<f4'), devices=(0, 1))
                except Exception:
                    pass
                else:
                    raise AssertionError('invalid weights accepted')
                assert two.labels_.tobytes() == before
                checks.append(dict(rows=n, algorithm=algorithm, metric=metric, weights=weighting,
                                   iterations=two.n_iter_, sha256=hashlib.sha256(before).hexdigest(),
                                   stages=two_stages))
                print('PASS', stem, flush=True)
    args.report.write_text(json.dumps(dict(status='PASS', checks=checks,
        corpus_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        scope='Two H100s; neighborhood row partitions and original root weighted core/label merges; replicated index'), indent=2)+'\n')


if __name__ == '__main__':
    main()
