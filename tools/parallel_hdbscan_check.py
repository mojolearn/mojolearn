#!/usr/bin/env python3
"""Cloud-only HDBSCAN with distributed k-NN and dense distance rows versus one device.

Each configuration is fitted once in this process (one device) and once through
fit_hdbscan (all selected devices), both with MOJOLEARN_IDENTITY_TRACE pointed
at a file; the two traces (input, core distances, every mutual reachability
cell, the MST, the condensed tree, stabilities, selection, labels) and the
fitted attributes must be equal.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def records(path):
    return [line for line in Path(path).read_text().splitlines() if line and not line.startswith('#')]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cloud', required=True, action='store_true')
    p.add_argument('--report', required=True, type=Path)
    p.add_argument('--traces', required=True, type=Path)
    p.add_argument('--devices', default='0,1')
    args = p.parse_args()
    if not os.environ.get('RUNPOD_POD_ID'):
        raise SystemExit('RunPod required; no local execution')
    for name in ('MOJOLEARN_NEIGHBORS_DEVICE_COUNT', 'MOJOLEARN_HIERARCHY_DEVICE_COUNT', 'MOJOLEARN_IDENTITY_TRACE'):
        os.environ.pop(name, None)
    args.traces.mkdir(parents=True, exist_ok=True)
    devices = tuple(int(v) for v in args.devices.split(','))
    import numpy as np
    from mojolearn.hdbscan import HDBSCAN
    from mojolearn.parallel_classical import fit_hdbscan

    names = ('labels_', 'core_distances_', 'n_clusters_', 'n_outliers_', 'n_boruvka_rounds_', 'n_condensed_clusters_')
    checks, refusals = [], []

    def digest(model):
        h = hashlib.sha256()
        for name in names:
            value = getattr(model, name)
            h.update(name.encode() + (np.asarray(value).tobytes() if hasattr(value, 'tobytes') else repr(value).encode()))
        return h.hexdigest()

    def trace_fit(path, function):
        Path(path).write_text('')
        os.environ['MOJOLEARN_IDENTITY_TRACE'] = str(path)
        try:
            return function()
        finally:
            os.environ.pop('MOJOLEARN_IDENTITY_TRACE', None)

    g = np.random.default_rng(2718)
    cases = [
        (5, 2, dict(min_cluster_size=2, min_samples=2)),
        (37, 3, dict(min_cluster_size=4)),
        (200, 2, dict(min_cluster_size=8, min_samples=3)),
        (515, 5, dict(min_cluster_size=10, cluster_selection_method='leaf')),
        (1024, 4, dict(min_cluster_size=15, alpha=1.5, allow_single_cluster=True)),
    ]
    for index, (n, d, params) in enumerate(cases):
        centers = g.normal(scale=5.0, size=(4, d))
        X = (centers[np.arange(n) % 4] + g.normal(size=(n, d))).astype('<f4')
        if n > 10:
            X[5] = X[3]
        one_path = args.traces / f'case{index}-one.trace'
        many_path = args.traces / f'case{index}-many.trace'
        one = trace_fit(one_path, lambda: _identical(HDBSCAN(**params)).fit(X))
        many = trace_fit(many_path, lambda: fit_hdbscan(_identical(HDBSCAN(**params)), X, devices=devices))
        a, b = records(one_path), records(many_path)
        assert len(a) == len(b) and len(a) > 0, (index, len(a), len(b))
        for i, (ra, rb) in enumerate(zip(a, b)):
            assert ra == rb, ('trace differs', index, i, ra, rb)
        assert digest(one) == digest(many), ('attributes differ', index)
        checks.append(dict(n=n, d=d, params=params, trace_records=len(a),
                           trace_sha256=hashlib.sha256('\n'.join(b).encode()).hexdigest(),
                           n_clusters=int(many.n_clusters_), sha256=digest(many)))
        print('PASS HDBSCAN', n, d, params, 'records', len(a), 'clusters', many.n_clusters_, flush=True)

    def refuse(name, function):
        try:
            function()
        except (ValueError, TypeError, RuntimeError, ImportError):
            refusals.append(name)
        else:
            raise AssertionError('expected refusal: ' + name)

    X = g.normal(size=(20, 2)).astype('<f4')
    refuse('wrong_type', lambda: fit_hdbscan(object(), X, devices=devices))
    fast = HDBSCAN()
    fast.numeric_mode = 'fast'
    refuse('nonidentical', lambda: fit_hdbscan(fast, X, devices=devices))
    bad = _identical(HDBSCAN(min_cluster_size=2, min_samples=50))
    refuse('min_samples_exceeds_rows', lambda: fit_hdbscan(bad, X, devices=devices))
    assert not hasattr(bad, 'labels_'), 'failed fit published state'

    args.report.write_text(json.dumps(dict(status='PASS', devices=list(devices), checks=checks, refusals=refusals,
        scope='HDBSCAN dense graph: k-NN query rows and pairwise distance rows across devices versus one device; '
              'root mutual reachability, MST, hierarchy and selection; no speed claim'), indent=2) + '\n')
    print('PASS', len(checks), 'HDBSCAN configurations and', len(refusals), 'refusals')


def _identical(model):
    model.numeric_mode = 'identical'
    return model


if __name__ == '__main__':
    main()
