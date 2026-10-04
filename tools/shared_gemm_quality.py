#!/usr/bin/env python3
"""Unscored M3 quality for incumbent/G1/G5; exact source and binary provenance.

SOURCE TAG: load staged shared_gemm_probe B.so (-D MOJOLEARN_APPLE_FAST_SHARED_GEMM_AUDIT).
No timing, no opponent calls, no queue changes, no builds. Actual dispatch routes.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

DEFINE = 'MOJOLEARN_APPLE_FAST_SHARED_GEMM_AUDIT'
ROOT = Path(__file__).resolve().parents[1]
FIXTURE = 'shared-gemm-routes-g1g5-v1'
CATALOG = '9ab2d3d3fb770498ef025db08f595a0149792bb7'
# Bounds fixed before B results. Compare both candidates to incumbent too.
ABS_SCALED_BOUND = 5e-6
# This is route-level quality, not fitted-estimator evidence. Labels state
# representative products; no claim about real board shape/path coverage.
ROUTES = {0: 'core-nt', 1: 'core-gram', 2: 'vendor-nn', 3: 'vendor-nt'}
SHAPES = [
    ('square', 128, 128, 128, 'random'),
    ('ragged', 65, 71, 33, 'random'),
    ('small', 7, 3, 9, 'random'),
    ('vector', 79, 1, 65, 'random'),
    ('cancel', 73, 65, 64, 'cancel'),
    ('dynamic', 79, 83, 65, 'dynamic'),
    ('zero-k', 5, 7, 0, 'random'),
    ('distance-block', 1024, 128, 64, 'random'),
    ('projection', 1024, 32, 257, 'random'),
    ('rbf-block', 256, 256, 129, 'random'),
    ('long-reduction', 67, 65, 2049, 'random'),
]
CASES = [(f'{ROUTES[r]}-{name}', m, n, k, r != 2, False, kind, r)
         for r in (0, 2, 3) for name, m, n, k, kind in SHAPES]
CASES += [(f'core-gram-{name}', m, m, k, True, True, kind, 1)
          for name, m, n, k, kind in SHAPES if name != 'vector']



def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source')
    p.add_argument('tag')
    args = p.parse_args()
    assert re.fullmatch('[0-9a-f]{40}', args.source)
    assert re.fullmatch('[A-Za-z0-9_.-]+', args.tag)
    os.chdir(ROOT)
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == args.source
    subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', 'bindings/', 'core/', 'gemm/',
                    'experiments/apple_fast/gemm/', 'tools/shared_gemm_quality.py'], check=True)
    arms = Path.home() / 'mq/verified-arms' / args.source / 'shared_gemm_probe'
    manifest = json.loads((arms / 'manifest.json').read_text())
    assert manifest['source_sha'] == args.source and manifest['binding'] == 'shared_gemm_probe'
    assert manifest['numeric_mode'] == 'fast'
    assert manifest['defines_A'] == '' and manifest['defines_B'] == '-D ' + DEFINE
    so = arms / 'B.so'
    digest = hashlib.sha256(so.read_bytes()).hexdigest()
    assert digest == manifest['hashes']['B']
    out = Path.home() / 'mq/out' / (args.tag + '-quality')
    out.mkdir(parents=True, exist_ok=False)
    os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    import numpy as np
    spec = importlib.util.spec_from_file_location('_mojolearn_shared_gemm_probe', so)
    b = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b)
    assert b.abi_version() == 1 and b.enabled() == 1
    calls = {arm: int(b.count(arm)) for arm in (0, 1, 5)}
    assert calls == {0: 0, 1: 0, 5: 0}
    records, failures = [], []
    route_counts = {r: [0, 0, 0, 0] for r in ROUTES}
    assert all(int(b.route_count(r, c)) == 0 for r in ROUTES for c in range(4))
    for name, m, n, k, nt, alias, kind, route in CASES:
        rng = np.random.default_rng(907)
        a = rng.standard_normal((m, k)).astype('float32')
        physical_b = rng.standard_normal((n, k) if nt else (k, n)).astype('float32')
        if kind == 'cancel':
            a[:, 1::2] = -a[:, ::2]
            if nt:
                physical_b[:, 1::2] = physical_b[:, ::2]
            else:
                physical_b[1::2] = physical_b[::2]
        if kind == 'dynamic':
            a *= np.exp2(rng.integers(-12, 12, size=a.shape)).astype('float32')
            physical_b *= np.exp2(rng.integers(-12, 12, size=physical_b.shape)).astype('float32')
        if alias:
            physical_b = a
        oracle = a.astype('float64') @ (physical_b.T if nt else physical_b).astype('float64')
        scale = max(float(np.linalg.norm(a.astype('float64')) * np.linalg.norm(physical_b.astype('float64'))), 1.0)
        # K0 buffers have one allocated sentinel because DeviceBuffer has min1.
        ap = a if k else np.zeros(1, dtype='float32')
        bp = physical_b if k else (ap if alias else np.zeros(1, dtype='float32'))
        metrics, outputs = {}, {}
        for arm in (0, 1, 5):
            c = np.full((m, n), np.nan, dtype='float32')
            reached = b.gemm(ap.ctypes.data, bp.ctypes.data, c.ctypes.data, [m, n, k, int(nt), arm, int(alias), route])
            assert reached == arm
            candidate = arm != 0 and n > 1 and k > 0
            column = (1 if arm == 1 else 2) if candidate else 0
            route_counts[route][column] += 1
            if candidate:
                route_counts[route][3] += 1
            actual_counts = {r: [int(b.route_count(r, c)) for c in range(4)] for r in ROUTES}
            assert actual_counts == route_counts, (name, arm, actual_counts, route_counts)
            calls[arm] += 1
            assert {v: int(b.count(v)) for v in calls} == calls
            finite = bool(np.isfinite(c).all())
            error = float(np.linalg.norm(c.astype('float64') - oracle) / scale)
            outputs[arm] = c
            metrics[arm] = dict(finite=finite, scaled_error=error, max_error=float(np.max(np.abs(c.astype('float64') - oracle))), sha256=hashlib.sha256(c.tobytes()).hexdigest())
            np.save(out / f'{name}-G{arm}.npy', c)
        # G1/G5 share shape and order; require exact output words, not a claim
        # that the SDK incumbent necessarily uses that same rounding order.
        same = outputs[1].tobytes() == outputs[5].tobytes()
        good = same and all(v['finite'] and 0 <= v['scaled_error'] <= ABS_SCALED_BOUND for v in metrics.values())
        good = good and all(metrics[arm]['scaled_error'] <= metrics[0]['scaled_error'] and metrics[arm]['max_error'] <= metrics[0]['max_error'] for arm in (1, 5))
        if not good:
            failures.append(name)
        row = dict(case=name, shape=[m, n, k], nt=nt, alias=alias, metrics=metrics,
                   incumbent='gpu-zero-contract+adapter-refusal' if k == 0 else ROUTES[route], route=route, candidate_eligible=n > 1 and k > 0,
                   direct_shared_exact=same, status='PASS' if good else 'FAIL')
        records.append(row)
        print('SHARED-GEMM-QUALITY ' + json.dumps(row, sort_keys=True), flush=True)
    result = dict(source_sha=args.source, catalog_source=CATALOG, binding_sha256=digest,
                  fixture=FIXTURE, abi_version=1, bound=ABS_SCALED_BOUND, no_regression_tolerance=0,
                  route_counts=route_counts, call_counts=calls, estimator_quality_covered=False, cases=records, failures=failures, status='FAIL' if failures else 'PASS')
    (out / 'report.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    if not failures:
        (out / 'PASS.json').write_text(json.dumps(result, sort_keys=True) + '\n')
    print('SHARED-GEMM-END status=' + result['status'], flush=True)
    return bool(failures)


if __name__ == '__main__':
    sys.exit(main())
