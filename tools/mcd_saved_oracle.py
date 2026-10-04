#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Analyze saved MCD captures only: no model import, fitting, GPU work or timing.

DATA A.npz B.npz --source MEASURED_SHA --tag UNIQUE_TAG --out REPORT.json
Reconstruct the exact captured fixture, require its saved SHA, then compare
both saved models against ONE float64 oracle conditional on their identical
saved reweighting support. This tests numerical quality, not whether the
robust estimator chose an optimal support. Float64 covariance uses empirical
normalization and the normal-consistency factor in MinCovDet.fit. Precision
uses the application's full-d float32-epsilon spectral cutoff.

Fixed conservative gate: exact raw fields/support/flags, finite values, and
B's error <= A's error for EVERY reported oracle metric. No fitted tolerance,
noise allowance, or averaging of gains against regressions. A failure means
HOLD; it does not establish that the changed search code caused the error.

Source diagnosis at measured d0b30bfe: _expansion_decomp.py::_masked_cov
uses the unchanged k.mm(Xc,Xc,ta=True). x_decomp/device.mojo::_launch_gemm_mma
splits k when tiles<640 and k>=1024; d220/n3000 uses five split windows.
gemm/afn_apple_fast.mojo's Atomic.fetch_add sums those windows in unspecified
order. Thus identical raw search results/support/location can still produce
slightly different final covariance and amplified pseudoinverse differences.
This explains a possible mechanism; it does NOT demonstrate a noise range.
The declared fit fixture uses defaults (assume_centered=False, contamination=.1).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from scipy.stats import chi2
from mcd_compat_quality import load_inputs


def json_ready(value):
    """Normalize NumPy scalars before writing or printing evidence."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def error_metrics(value, reference):
    value = np.asarray(value, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if value.shape != reference.shape:
        raise ValueError(f'oracle shape mismatch: {value.shape} != {reference.shape}')
    diff = value - reference
    if not np.isfinite(diff).all():
        raise ValueError('nonfinite saved value or oracle')
    denom = max(float(np.linalg.norm(reference.ravel())), np.finfo(np.float64).tiny)
    return dict(rel_l2=float(np.linalg.norm(diff.ravel()) / denom),
                max_abs=float(np.max(np.abs(diff), initial=0.0)))


def distances(x, location, precision):
    centered = x - location
    return np.einsum('ij,ij->i', centered @ precision, centered)


def analyze(a, b, x, q, lane):
    exact_names = ('raw_location_', 'raw_covariance_', 'raw_support_', 'support_', 'flags')
    exact = {key: (a[key].shape == b[key].shape and a[key].dtype == b[key].dtype
                   and a[key].tobytes() == b[key].tobytes()) for key in exact_names}
    # A common conditional oracle is only valid when the saved masks coincide.
    if not exact['support_']:
        return dict(status='HOLD', reason='different reweighting supports', exact=exact)
    for saved in (a, b):
        for key in ('raw_location_', 'raw_covariance_', 'location_', 'covariance_',
                    'precision_', 'dist_', 'distances'):
            if not np.isfinite(saved[key]).all():
                raise ValueError('nonfinite captured field: ' + key)
        if not np.isin(saved['support_'], (0, 1)).all():
            raise ValueError('support contains values other than zero/one')
        if np.asarray(saved['flags']).shape != (len(q),):
            raise ValueError('query flag shape mismatch')
    support = np.asarray(a['support_']).astype(bool)
    if support.shape != (len(x),) or not support.any():
        raise ValueError('invalid saved reweighting support')
    selected = x[support].astype(np.float64)
    location = selected.mean(axis=0)
    centered = selected - location
    d = x.shape[1]
    factor = .975 / chi2.cdf(chi2.ppf(.975, d), d + 2)
    covariance = centered.T @ centered / len(selected) * factor
    eigenvalues, vectors = np.linalg.eigh(covariance)
    cutoff = float(float(np.max(np.abs(eigenvalues))) * d * float(np.finfo(np.float32).eps))
    keep = np.abs(eigenvalues) > cutoff
    precision = (vectors[:, keep] / eigenvalues[keep]) @ vectors[:, keep].T
    train_dist = distances(x.astype(np.float64), location, precision)
    query_dist = distances(q.astype(np.float64), location, precision)
    oracle = dict(location_=location, covariance_=covariance, precision_=precision,
                  dist_=train_dist, distances=query_dist)
    if lane == 'elliptic-envelope':
        offset = np.percentile(-train_dist, 10.0, method='linear')
        oracle.update(offset_=np.asarray(offset), decision_function=-query_dist-offset)
        oracle_flags = oracle['decision_function'] < 0
    else:
        oracle_flags = query_dist > chi2.ppf(.975, d)
    metrics = {}
    passed = all(exact.values())
    for key, reference in oracle.items():
        ma, mb = error_metrics(a[key], reference), error_metrics(b[key], reference)
        checks = {name: mb[name] <= ma[name] for name in ma}
        metrics[key] = dict(A=ma, B=mb, no_worse=checks)
        passed &= all(checks.values())
    flag_errors = {arm: int(np.count_nonzero(np.asarray(saved['flags'], dtype=bool) != oracle_flags))
                   for arm, saved in (('A', a), ('B', b))}
    metrics['oracle_flag_disagreement_count'] = dict(**flag_errors,
                                                    no_worse=flag_errors['B'] <= flag_errors['A'])
    passed &= flag_errors['B'] <= flag_errors['A']
    # Same-input covariance drift is consistent with split-K atomics, but this
    # diagnostic neither measures a noise range nor relaxes any oracle gate.
    ab = {key: error_metrics(b[key], a[key]) for key in oracle}
    return dict(status='PASS' if passed else 'HOLD', exact=exact, metrics=metrics,
                A_B_differences=ab, oracle_rank=int(keep.sum()), oracle_cutoff=cutoff,
                support_count=int(support.sum()), consistency_factor=float(factor),
                criterion='all exact checks and every B error <= corresponding A error; zero allowance',
                scope='float64 numerical accuracy conditional on common saved support; no opponent refit')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('data')
    p.add_argument('a', type=Path)
    p.add_argument('b', type=Path)
    p.add_argument('--source', required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--extra-baseline', type=Path,
                   help='another saved A from same fixture/source; diagnostic only, never relaxes gate')
    args = p.parse_args()
    if not re.fullmatch('[0-9a-f]{40}', args.source):
        p.error('--source must be full measured source SHA')
    if args.out.exists():
        raise FileExistsError('refusing to overwrite oracle report')
    with np.load(args.a, allow_pickle=False) as aa, np.load(args.b, allow_pickle=False) as bb:
        a, b = dict(aa), dict(bb)
    for key in ('dataset', 'lane', 'shape', 'data_sha'):
        if not np.array_equal(a[key], b[key]):
            raise ValueError('capture identity mismatch: ' + key)
    dataset, lane = str(a['dataset']), str(a['lane'])
    if lane not in ('min-cov-det', 'elliptic-envelope'):
        raise ValueError('unsupported lane')
    source_records = {}
    for arm, saved in (('A', a), ('B', b)):
        recorded = str(saved.get('compiled_source', 'legacy-unrecorded'))
        source_records[arm] = recorded
        if recorded not in ('legacy-unrecorded', args.source):
            raise ValueError('capture source mismatch: ' + arm)
    n = int(a['shape'][0])
    x, q = load_inputs(SimpleNamespace(data=args.data, dataset=dataset, lane=lane, rows=n))
    data_sha = hashlib.sha256(x.tobytes() + q.tobytes()).hexdigest()
    if data_sha != str(a['data_sha']) or tuple(x.shape) != tuple(a['shape']):
        raise ValueError('reconstructed fixture differs from saved input hash/shape')
    result = analyze(a, b, x, q, lane)
    result['provenance'] = dict(tag=args.tag, measured_source_declared=args.source,
        source_recorded=source_records,
        source_verified_in_captures=all(v == args.source for v in source_records.values()),
        capture_A=str(args.a.resolve()), capture_B=str(args.b.resolve()),
        capture_sha256_A=sha(args.a), capture_sha256_B=sha(args.b),
        fixture_sha256=data_sha, dataset=dataset, lane=lane, shape=list(x.shape),
        script_sha256=sha(__file__), script_commit=subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[1], text=True).strip(),
        numpy_version=np.__version__, analysis_only=True)
    if args.extra_baseline:
        with np.load(args.extra_baseline, allow_pickle=False) as capture:
            extra = dict(capture)
        for key in ('dataset', 'lane', 'shape', 'data_sha'):
            if not np.array_equal(a[key], extra[key]):
                raise ValueError('extra baseline fixture mismatch: ' + key)
        extra_source = str(extra.get('compiled_source', 'legacy-unrecorded'))
        if extra_source != source_records['A']:
            raise ValueError('extra baseline recorded source differs from A')
        extra_arm = str(extra.get('arm', 'unrecorded'))
        if extra_arm not in ('A', 'unrecorded'):
            raise ValueError('extra baseline must be saved arm A')
        aa = analyze(a, extra, x, q, lane)
        eb = analyze(extra, b, x, q, lane)
        result['extra_baseline'] = dict(
            path=str(args.extra_baseline.resolve()), sha256=sha(args.extra_baseline),
            recorded_source=extra_source,
            source_verified_in_capture=extra_source == args.source,
            A_vs_extra_A=aa, extra_A_vs_B=eb,
            interpretation='two observed A captures; no variance estimate or threshold relaxation; primary A/B gate unchanged')
    result = json_ready(result)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    for key, values in result.get('metrics', {}).items():
        print('MCD-SAVED-ORACLE-METRIC ' + json.dumps(dict(field=key, **values), sort_keys=True))
    if 'extra_baseline' in result:
        extra = result['extra_baseline']
        for label in ('A_vs_extra_A', 'extra_A_vs_B'):
            diagnostic = extra[label]
            print('MCD-SAVED-BASELINE ' + json.dumps(dict(comparison=label,
                status=diagnostic['status'], exact=diagnostic.get('exact'),
                A_B_differences=diagnostic.get('A_B_differences'),
                metrics=diagnostic.get('metrics')), sort_keys=True))
    print('MCD-SAVED-ORACLE ' + json.dumps(dict(status=result['status'], tag=args.tag,
        exact=result.get('exact'), report=str(args.out), provenance=result['provenance']), sort_keys=True))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
