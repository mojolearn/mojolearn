#!/usr/bin/env python3
"""Unscored M3 quality for incumbent/G1/G5; exact source and binary provenance.

SOURCE TAG: load staged gemm_probe B.so (-D MOJOLEARN_APPLE_GEMM_PROBE).
No timing, no opponent calls, no queue changes, no builds, no production hook.
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

DEFINE = 'MOJOLEARN_APPLE_GEMM_PROBE'
ROOT = Path(__file__).resolve().parents[1]
FIXTURE = 'catalog-gemm-g1g5-v1'
CATALOG = '9ab2d3d3fb770498ef025db08f595a0149792bb7'
# Bounds fixed before B results. Compare both candidates to incumbent too.
ABS_SCALED_BOUND = 5e-6
CASES = [
    ('nn-square', 128, 128, 128, False, False, 'random'),
    ('nt-square', 128, 128, 128, True, False, 'random'),
    ('nn-ragged', 65, 71, 33, False, False, 'random'),
    ('nt-ragged', 65, 71, 33, True, False, 'random'),
    ('nn-small', 7, 3, 9, False, False, 'random'),
    ('nt-small', 7, 3, 9, True, False, 'random'),
    ('nt-vector', 79, 1, 65, True, False, 'random'),
    ('gram-alias', 67, 67, 35, True, True, 'random'),
    ('nn-cancel', 73, 65, 64, False, False, 'cancel'),
    ('nt-dynamic', 79, 83, 65, True, False, 'dynamic'),
    ('nn-zero-k', 5, 7, 0, False, False, 'random'),
    ('nt-zero-k', 5, 7, 0, True, False, 'random'),
]


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
                    'experiments/apple_fast/gemm/', 'tools/catalog_gemm_quality.py'], check=True)
    arms = Path.home() / 'mq/verified-arms' / args.source / 'gemm_probe'
    manifest = json.loads((arms / 'manifest.json').read_text())
    assert manifest['source_sha'] == args.source and manifest['binding'] == 'gemm_probe'
    assert manifest['numeric_mode'] == 'fast'
    assert manifest['defines_A'] == '' and manifest['defines_B'] == '-D ' + DEFINE
    so = arms / 'B.so'
    digest = hashlib.sha256(so.read_bytes()).hexdigest()
    assert digest == manifest['hashes']['B']
    out = Path.home() / 'mq/out' / (args.tag + '-quality')
    out.mkdir(parents=True, exist_ok=False)
    os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    import numpy as np
    spec = importlib.util.spec_from_file_location('_mojolearn_gemm_probe', so)
    b = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b)
    assert b.abi_version() == 1 and b.enabled() == 1
    calls = {arm: int(b.count(arm)) for arm in (0, 1, 5)}
    assert calls == {0: 0, 1: 0, 5: 0}
    records, failures = [], []
    for name, m, n, k, nt, alias, kind in CASES:
        rng = np.random.default_rng(907)
        a = rng.standard_normal((m, k)).astype('float32')
        physical_b = rng.standard_normal((n, k) if nt else (k, n)).astype('float32')
        if kind == 'cancel':
            a[:, 1::2] = -a[:, ::2]
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
        bp = physical_b if k else np.zeros(1, dtype='float32')
        metrics, outputs = {}, {}
        for arm in (0, 1, 5):
            c = np.full((m, n), np.nan, dtype='float32')
            reached = b.gemm(ap.ctypes.data, bp.ctypes.data, c.ctypes.data, [m, n, k, int(nt), arm, int(alias)])
            assert reached == arm
            calls[arm] += 1
            assert {v: int(b.count(v)) for v in calls} == calls
            finite = bool(np.isfinite(c).all())
            error = float(np.linalg.norm(c.astype('float64') - oracle) / scale)
            outputs[arm] = c
            metrics[arm] = dict(finite=finite, scaled_error=error, sha256=hashlib.sha256(c.tobytes()).hexdigest())
            np.save(out / f'{name}-G{arm}.npy', c)
        # G1/G5 share shape and order; require exact output words, not a claim
        # that the SDK incumbent necessarily uses that same rounding order.
        same = outputs[1].tobytes() == outputs[5].tobytes()
        good = same and all(v['finite'] and 0 <= v['scaled_error'] <= ABS_SCALED_BOUND for v in metrics.values())
        good = good and all(metrics[arm]['scaled_error'] <= metrics[0]['scaled_error'] for arm in (1, 5))
        if not good:
            failures.append(name)
        row = dict(case=name, shape=[m, n, k], nt=nt, alias=alias, metrics=metrics,
                   incumbent='gpu-zero-contract' if k == 0 else ('core-nt' if nt else 'sdk-nn'),
                   direct_shared_exact=same, status='PASS' if good else 'FAIL')
        records.append(row)
        print('CATALOG-GEMM-QUALITY ' + json.dumps(row, sort_keys=True), flush=True)
    result = dict(source_sha=args.source, catalog_source=CATALOG, binding_sha256=digest,
                  fixture=FIXTURE, abi_version=1, bound=ABS_SCALED_BOUND, no_regression_tolerance=0,
                  call_counts=calls, cases=records, failures=failures, status='FAIL' if failures else 'PASS')
    (out / 'report.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    if not failures:
        (out / 'PASS.json').write_text(json.dumps(result, sort_keys=True) + '\n')
    print('CATALOG-GEMM-END status=' + result['status'], flush=True)
    return bool(failures)


if __name__ == '__main__':
    sys.exit(main())
