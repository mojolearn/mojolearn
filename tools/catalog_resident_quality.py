#!/usr/bin/env python3
"""Unscored M3 quality for incumbent/G1-G10; exact source and binary provenance.

SOURCE TAG: load staged resident_gemm_probe B.so (-D MOJOLEARN_APPLE_GEMM_RESIDENT_PROBE).
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

DEFINE = 'MOJOLEARN_APPLE_GEMM_RESIDENT_PROBE'
ROOT = Path(__file__).resolve().parents[1]
FIXTURE = 'catalog-resident-matrix-v1'
PRIOR_SHA256 = 'f2bcde131ffea5ff54cea85920c3a49ec7888b5d4b2824e2077189b50aa973a0'
ARMS = tuple(range(11))
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
    p.add_argument('prior_report',help='reviewed all10 quality JSON; not cold timing output')
    args = p.parse_args()
    prior_path = Path(args.prior_report).expanduser().resolve()
    assert hashlib.sha256(prior_path.read_bytes()).hexdigest() == PRIOR_SHA256
    prior = json.loads(prior_path.read_text())
    assert prior['source_sha'] == '9f1a1657c6e054573a495425e223fd73e0c174db'
    assert prior['failures'] == ['nt-vector']
    prior_cases = {row['case']:row for row in prior['cases']}
    assert re.fullmatch('[0-9a-f]{40}', args.source)
    assert re.fullmatch('[A-Za-z0-9_.-]+', args.tag)
    os.chdir(ROOT)
    assert subprocess.check_output(["sysctl","-n","machdep.cpu.brand_string"],text=True).strip() == "Apple M3 Ultra"
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == args.source
    subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', 'bindings/', 'core/', 'gemm/',
                    'experiments/apple_fast/gemm/', 'tools/catalog_resident_quality.py'], check=True)
    arms = Path.home() / 'mq/verified-arms' / args.source / 'resident_gemm_probe'
    manifest = json.loads((arms / 'manifest.json').read_text())
    assert manifest['source_sha'] == args.source and manifest['binding'] == 'resident_gemm_probe'
    assert manifest['numeric_mode'] == 'fast'
    assert manifest['defines_A'] == '' and manifest['defines_B'] == '-D ' + DEFINE
    so = arms / 'B.so'
    digest = hashlib.sha256(so.read_bytes()).hexdigest()
    assert digest == manifest['hashes']['B']
    out = Path.home() / 'mq/out' / (args.tag + '-quality')
    out.mkdir(parents=True, exist_ok=False)
    os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    import numpy as np
    spec = importlib.util.spec_from_file_location('_mojolearn_resident_gemm_probe', so)
    b = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b)
    assert b.abi_version() == 1 and b.enabled() == 1
    # Predeclared n=1 refusal, not a relaxed vector-quality threshold.
    sentinel = np.zeros(1,dtype='float32')
    try:
        b.prepare(sentinel.ctypes.data,sentinel.ctypes.data,[1,1,1,1,0])
    except Exception as exc:
        assert 'matrix-only contract' in str(exc), str(exc)
    else:
        raise AssertionError('resident matrix contract must refuse n=1')
    calls = {arm: int(b.count(arm)) for arm in ARMS}
    assert calls == {arm: 0 for arm in ARMS}
    records, failures = [], []
    variant_failures = {arm: [] for arm in ARMS[1:]}
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
        for arm in ARMS:
            c = np.full((m, n), np.nan, dtype='float32')
            assert b.prepare(ap.ctypes.data, bp.ctypes.data, [m,n,k,int(nt),int(alias)]) == 1
            reached = b.run_read(arm, c.ctypes.data)
            # The prepared buffers are one-shot. A refusal is not a repeat.
            try:
                b.run_read(arm,c.ctypes.data)
            except Exception as exc:
                assert 'unused prepared buffers' in str(exc), str(exc)
            else:
                raise AssertionError('resident preparation must reject replay')
            assert b.release() == 1
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
        variant_status = {}
        for arm in ARMS[1:]:
            metric = metrics[arm]
            good_arm = (metric['finite'] and 0 <= metric['scaled_error'] <= ABS_SCALED_BOUND
                        and metric['scaled_error'] <= metrics[0]['scaled_error']
                        and metrics[0]['finite'])
            good_arm = good_arm and metric['sha256'] == prior_cases[name]['metrics'][str(arm)]['sha256']
            if arm in (1, 5):
                good_arm = good_arm and same
            variant_status[arm] = 'PASS' if good_arm else 'FAIL'
            if not good_arm:
                variant_failures[arm].append(name)
        prior_words_equal = all(metrics[arm]['sha256'] == prior_cases[name]['metrics'][str(arm)]['sha256'] for arm in ARMS)
        good = prior_words_equal and all(v == 'PASS' for v in variant_status.values())
        if not good:
            failures.append(name)
        row = dict(case=name, shape=[m, n, k], nt=nt, alias=alias, metrics=metrics,
                   incumbent='gpu-zero-contract' if k == 0 else ('core-nt' if nt else 'sdk-nn'),
                   prior_words_equal=prior_words_equal, direct_shared_exact=same, variant_status=variant_status, status='PASS' if good else 'FAIL')
        records.append(row)
        print('CATALOG-GEMM-QUALITY ' + json.dumps(row, sort_keys=True), flush=True)
    result = dict(source_sha=args.source, catalog_source=CATALOG, binding_sha256=digest,
                  prior_report_sha256=PRIOR_SHA256, fixture=FIXTURE, abi_version=1, bound=ABS_SCALED_BOUND, no_regression_tolerance=0,
                  contract="resident-input-matrix", n1_refused=True, call_counts=calls, cases=records, failures=failures, variant_failures=variant_failures, status='FAIL' if failures else 'PASS')
    (out / 'report.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    for arm, failed_cases in variant_failures.items():
        if not failed_cases:
            receipt = dict(source_sha=args.source, catalog_source=CATALOG,
                           binding_sha256=digest, fixture=FIXTURE, abi_version=1,
                           variant=arm, no_regression_tolerance=0, status='PASS',
                           cases=[row['case'] for row in records], timing=False)
            (out / f'PASS_G{arm}.json').write_text(json.dumps(receipt, sort_keys=True) + '\n')
    if not failures:
        (out / 'PASS.json').write_text(json.dumps(result, sort_keys=True) + '\n')
    print('CATALOG-GEMM-END status=' + result['status'], flush=True)
    return bool(failures)


if __name__ == '__main__':
    sys.exit(main())
