#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One fit per arm, board data, real robust-fit quality against current main.

fit DATA_DIR DATASET OUT.npz [--rows N] [--lane min-cov-det|elliptic-envelope]
compare A.npz B.npz
No opponents are fitted. The CPU only evaluates saved fitted quantities.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np


def fit(args):
    import bench_board_algos as board
    import mojolearn as ml
    from scipy.stats import chi2

    block, _ = board._load_block(args.lane, args.dataset, args.data)
    data = board.lane_arrays(args.lane, block)
    x = np.ascontiguousarray(data['X'], dtype=np.float32)
    q = np.ascontiguousarray(data['Xq'], dtype=np.float32)
    if args.rows:
        x, q = x[:args.rows], q[:args.rows]
    cls = ml.MinCovDet if args.lane == 'min-cov-det' else ml.EllipticEnvelope
    model = cls(random_state=board.SEED, numeric_mode='fast')
    before = time.perf_counter()
    model.fit(x)
    ms = 1000 * (time.perf_counter() - before)
    distances = np.asarray(model.mahalanobis(q), dtype=np.float64)
    flags = (distances > chi2.ppf(.975, x.shape[1]) if args.lane == 'min-cov-det'
             else np.asarray(model.predict(q)) < 0)
    payload = {name: np.asarray(getattr(model, name)) for name in (
        'location_', 'covariance_', 'precision_', 'support_', 'raw_location_',
        'raw_covariance_', 'raw_support_', 'dist_')}
    payload.update(distances=distances, flags=flags, fit_ms=ms,
                   dataset=args.dataset, lane=args.lane, shape=x.shape,
                   data_sha=hashlib.sha256(x.tobytes() + q.tobytes()).hexdigest())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **payload)
    print('MCDQ-FIT ' + json.dumps(dict(lane=args.lane, dataset=args.dataset,
          shape=list(x.shape), fit_ms=ms, fraction_flagged=float(flags.mean()),
          raw_rank=int(np.linalg.matrix_rank(payload['raw_covariance_'])), out=args.out)), flush=True)


def compare(args):
    with np.load(args.a) as a, np.load(args.b) as b:
        for key in ('dataset', 'lane', 'shape', 'data_sha'):
            if not np.array_equal(a[key], b[key]):
                raise ValueError('Different inputs: ' + key)
        metrics = {}
        ok = True
        for key in ('location_', 'covariance_', 'precision_', 'raw_location_', 'raw_covariance_', 'distances', 'dist_'):
            av, bv = a[key].astype(np.float64), b[key].astype(np.float64)
            finite = bool(np.isfinite(av).all() and np.isfinite(bv).all())
            rel = float(np.linalg.norm(av - bv) / max(np.linalg.norm(av), 1e-12))
            metrics[key + '_rel'] = rel
            ok &= finite and rel <= .01
        for key in ('flags', 'support_', 'raw_support_'):
            av, bv = a[key].astype(bool), b[key].astype(bool)
            union = int(np.count_nonzero(av | bv))
            jac = float(np.count_nonzero(av & bv) / union) if union else 1.0
            metrics[key + '_jaccard'] = jac
            ok &= jac >= .99
        # A rank loss was the concrete failure of the earlier 215 ms route.
        ra = int(np.linalg.matrix_rank(a['raw_covariance_']))
        rb = int(np.linalg.matrix_rank(b['raw_covariance_']))
        metrics.update(raw_rank_a=ra, raw_rank_b=rb,
                       fit_ms_a=float(a['fit_ms']), fit_ms_b=float(b['fit_ms']))
        ok &= ra == rb
        print('MCDQ ' + json.dumps(dict(status='PASS' if ok else 'FAIL', **metrics)), flush=True)
        return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('fit')
    p.add_argument('data')
    p.add_argument('dataset')
    p.add_argument('out')
    p.add_argument('--rows', type=int)
    p.add_argument('--lane', choices=('min-cov-det', 'elliptic-envelope'), default='min-cov-det')
    p = sub.add_parser('compare')
    p.add_argument('a')
    p.add_argument('b')
    args = parser.parse_args()
    return fit(args) if args.command == 'fit' else compare(args)


if __name__ == '__main__':
    raise SystemExit(main())
