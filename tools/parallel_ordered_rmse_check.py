#!/usr/bin/env python3
"""Cloud-only ordered fold and feature-partition identity."""
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
    from mojolearn import OrderedRMSE
    from mojolearn.parallel_ensemble import fit_ordered_rmse
    corpus = args.corpus.read_bytes()
    raw = np.frombuffer(corpus, dtype=np.uint8)
    checks = []
    for n in (65, 513):
        X = raw[:n*73].astype('<f4').reshape(n, 73)
        X[:, :33] %= np.float32(2)
        X[:, 33:50] %= np.float32(8)
        y = raw[1000000:1000000+n].astype('<f4') / np.float32(255)
        for border in (1, 7, 31, 254):
            for reverse in (False, True):
                order = np.arange(n, dtype='<i8')
                if reverse:
                    order = order[::-1].copy()
                weights = np.ones(n, dtype='<f4')
                weights[::7] = 0
                weights[::11] = 0.5
                options = dict(n_estimators=3, max_depth=3, border_count=border)
                trace_root = args.report.parent / ('ordered-trace-' + str(len(checks)))
                trace_root.mkdir(exist_ok=True)
                os.environ['MOJOLEARN_IDENTITY_TRACE'] = str(trace_root / 'one.trace')
                one = OrderedRMSE(**options).fit(X,y,permutation=order,sample_weight=weights)
                os.environ['MOJOLEARN_IDENTITY_TRACE'] = str(trace_root / 'many.trace')
                many = fit_ordered_rmse(OrderedRMSE(**options),X,y,permutation=order,
                                        sample_weight=weights,devices=(0,1))
                os.environ.pop('MOJOLEARN_IDENTITY_TRACE', None)
                records = lambda p: [line for line in p.read_text().splitlines() if line and not line.startswith('#')]
                assert records(trace_root / 'one.trace') == records(trace_root / 'many.trace')
                assert one.model_ == many.model_
                assert one.loss_curve_ is None and many.loss_curve_ is None
                assert one.predict(X).tobytes() == many.predict(X).tobytes()
                before = many.model_
                try:
                    fit_ordered_rmse(many,X,y,permutation=np.zeros(n,dtype='<i8'),devices=(0,1))
                except (ValueError, RuntimeError):
                    pass
                else:
                    raise AssertionError('invalid permutation accepted')
                assert many.model_ == before
                checks.append(dict(rows=n,border=border,reverse=reverse,
                    sha256=hashlib.sha256(many.model_.encode()+many.predict(X).tobytes()).hexdigest()))
                print('PASS ordered RMSE',n,border,reverse,flush=True)
    args.report.write_text(json.dumps(dict(status='PASS',checks=checks,
        corpus_sha256=hashlib.sha256(corpus).hexdigest(),
        scope='Two H100s; original ordered permutation/fold state with feature-group histograms; no pooled state claim'),indent=2)+'\n')


if __name__ == '__main__':
    main()
