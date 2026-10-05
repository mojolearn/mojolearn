#!/usr/bin/env python3
"""Targeted opt-in candidate identity/quality/timing worker (GMM, KMeans, OCSVM).

One fit (+ inference) on the canonical medium fixture named in
docs/identical/candidate-priority-queue.json, run against ONE source tree
(baseline = wave ON tree, candidate = candidate build tree). Saves every
compared output raw (npz) and a SHA256 digest over names/dtypes/shapes/bytes.
--operation identity: no timer (GPU or host column).
--operation timing: GPU only; one untimed warmup fit+infer, then ONE measured
fit+infer sample. No opponents, no installs; the package and every mapped
_mojolearn binary must come from --source.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

FIXTURES = {
    # lane: (block, fit_rows, query_rows)
    'gmm': ('reg-taxi.npz', 50000, 50000),
    'kmeans': ('reg-taxi.npz', 50000, 0),
    'ocsvm': ('cls-taxi.npz', 10000, 10000),
}


def provenance(root, vendor):
    import mojolearn as ml
    package = Path(ml.__file__).resolve()
    if not package.is_relative_to(root / 'python/mojolearn'):
        raise RuntimeError('refused non-source mojolearn: ' + str(package))
    if ml.vendor() != vendor:
        raise RuntimeError('vendor %r != %r' % (ml.vendor(), vendor))
    paths = set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        tail = line.split()[-1]
        if '_mojolearn' in tail and '.so' in tail:
            p = Path(tail).resolve()
            if not p.is_relative_to(root / 'python/mojolearn'):
                raise RuntimeError('refused external binding: ' + str(p))
            paths.add(str(p))
    if not paths:
        raise RuntimeError('no source binding mapped')
    return sorted(paths)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--lane', required=True, choices=sorted(FIXTURES))
    p.add_argument('--data', required=True, type=Path, help='rows-small directory')
    p.add_argument('--vendor', required=True, choices=('cuda', 'hip', 'cpu'))
    p.add_argument('--operation', required=True, choices=('identity', 'timing'))
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    if a.operation == 'timing' and a.vendor == 'cpu':
        p.error('host timing forbidden')
    if os.environ.get('MOJOLEARN_NUMERIC_MODE') != 'identical':
        p.error('IDENTICAL mode required')
    a.out.mkdir(parents=True, exist_ok=False)
    root = a.source.resolve()
    sys.path.insert(0, str(root / 'python'))
    import numpy as np
    import mojolearn as ml
    block, fit_rows, query_rows = FIXTURES[a.lane]
    with np.load(a.data / block) as z:
        X = np.ascontiguousarray(z['X'][:fit_rows], dtype=np.float32)
        Xq = np.ascontiguousarray(z['Xq'][:query_rows], dtype=np.float32) if query_rows else None
    state = {}
    if a.lane == 'gmm':
        def fit():
            state['m'] = ml.GaussianMixture(n_components=4, max_iter=20, random_state=7).fit(X)
        def infer():
            m = state['m']; state['proba'] = m.predict_proba(Xq); state['score'] = m.score_samples(Xq)
        def outputs():
            m = state['m']
            return {'means': m.means_, 'covariances': m.covariances_, 'weights': m.weights_,
                    'precisions_cholesky': m.precisions_cholesky_, 'n_iter': np.int64(m.n_iter_),
                    'converged': np.int64(m.converged_), 'lower_bound': np.float64(m.lower_bound_),
                    'responsibilities': state['proba'], 'score_samples': state['score']}
    elif a.lane == 'kmeans':
        def fit():
            state['m'] = ml.KMeans(n_clusters=8, n_init=1, random_state=7).fit(X)
        def infer():
            pass
        def outputs():
            m = state['m']
            return {'centers': m.cluster_centers_, 'labels': m.labels_, 'inertia': np.float64(m.inertia_),
                    'n_iter': np.int64(m.n_iter_)}
    else:
        def fit():
            state['m'] = ml.OneClassSVM(kernel='rbf', nu=0.1, tol=1e-3, gamma='scale', max_iter=-1).fit(X)
        def infer():
            m = state['m']; state['dec'] = m.decision_function(Xq); state['pred'] = m.predict(Xq)
        def outputs():
            m = state['m']
            return {'dual_coef': m.dual_coef_, 'intercept': m.intercept_, 'support': m.support_,
                    'n_iter': np.int64(m.n_iter_), 'decision': state['dec'], 'labels': state['pred']}
    metrics = {}
    if a.operation == 'timing':
        fit(); infer()  # one untimed warmup
        start = time.perf_counter(); fit(); fit_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter(); infer(); infer_ms = (time.perf_counter() - start) * 1000
        metrics = {'fit_ms': fit_ms, 'infer_ms': infer_ms if a.lane != 'kmeans' else None}
    else:
        fit(); infer()
    readback = getattr(state['m'], 'numeric_mode_used', None)
    mode = readback() if callable(readback) else 'unavailable'
    if mode not in ('identical', 'unavailable'):
        raise RuntimeError('numeric mode readback ' + repr(mode))
    mapped = provenance(root, a.vendor)
    digest = hashlib.sha256(); meta = {}; saved = {}
    for name, value in sorted(outputs().items()):
        arr = np.ascontiguousarray(np.asarray(value))
        if arr.dtype.hasobject:
            raise RuntimeError('object output ' + name)
        blob = arr.tobytes()
        m = {'dtype': arr.dtype.str, 'shape': list(arr.shape), 'bytes': arr.nbytes}
        digest.update(json.dumps([name, m], sort_keys=True).encode() + b'\0' + blob)
        meta[name] = dict(m, sha256=hashlib.sha256(blob).hexdigest()); saved[name] = arr
    np.savez(a.out / 'outputs.npz', **saved)
    n_iter = int(saved['n_iter'].reshape(-1)[0])
    result = {'status': 'PASS', 'lane': a.lane, 'vendor': a.vendor, 'operation': a.operation,
              'fixture': {'block': block, 'fit_rows': fit_rows, 'query_rows': query_rows},
              'warmups': 1 if a.operation == 'timing' else 0, 'timing_samples': 1 if a.operation == 'timing' else 0,
              'digest': digest.hexdigest(), 'outputs': meta, 'n_iter': n_iter, 'bindings': mapped,
              'numeric_mode_env': os.environ.get('MOJOLEARN_NUMERIC_MODE'), 'numeric_mode_used': mode, **metrics}
    (a.out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print('CANDIDATE_GATE', a.lane, a.vendor, a.operation, 'n_iter', n_iter, digest.hexdigest(), json.dumps(metrics))


if __name__ == '__main__':
    main()
