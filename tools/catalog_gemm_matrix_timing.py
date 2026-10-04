#!/usr/bin/env python3
"""M3-only matrix geometry screen, not unrestricted quality or caller admission.

SOURCE TAG REPORT_PATH REPORT_SHA256. Root must review before queueing.
All 66 calls are predeclared: six shapes x G0..G10, one fresh process each.
No warmups, repetitions, opponent calls, production changes or builds.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import catalog_gemm_quality as quality

ROOT = Path(__file__).resolve().parents[1]
SOURCE = '9f1a1657c6e054573a495425e223fd73e0c174db'
REPORT_SHA256 = 'f2bcde131ffea5ff54cea85920c3a49ec7888b5d4b2824e2077189b50aa973a0'
ARMS = tuple(range(11))
# name, m, n, k, NT, aliased Gram. n>=2 is mandatory for this screen.
SHAPES = (
    ('dense-nn', 2048, 512, 512, False, False),
    ('square-nn', 1024, 1024, 1024, False, False),
    ('tall-projection-nn', 32768, 64, 220, False, False),
    ('lowwidth-kmeans-nt', 32768, 8, 220, True, False),
    ('odd-nt', 4097, 71, 221, True, False),
    ('gram-nt', 1024, 1024, 220, True, True),
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(args):
    assert args.source == SOURCE
    assert re.fullmatch('[A-Za-z0-9_.-]+', args.tag)
    assert args.report_sha256 == REPORT_SHA256, 'not the reviewed matrix-only quality report'
    os.chdir(ROOT)
    brand = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
    assert brand == 'Apple M3 Ultra', 'timing requires Apple M3 Ultra, found: ' + brand
    subprocess.run(['git', 'merge-base', '--is-ancestor', SOURCE, 'HEAD'], check=True)
    paths = ['*.mojo', 'bindings/', 'python/', 'pixi.toml', 'pixi.lock',
             'tools/catalog_gemm_quality.py']
    subprocess.run(['git', 'diff', '--quiet', SOURCE + '..HEAD', '--', *paths], check=True)
    subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', *paths,
                    'tools/catalog_gemm_matrix_timing.py'], check=True)
    report_path = Path(args.report).expanduser().resolve()
    assert digest(report_path) == args.report_sha256, 'quality report hash mismatch'
    report = json.loads(report_path.read_text())
    arms = Path.home() / 'mq/verified-arms' / SOURCE / 'gemm_probe'
    manifest = json.loads((arms / 'manifest.json').read_text())
    assert manifest['source_sha'] == SOURCE and manifest['binding'] == 'gemm_probe'
    assert manifest['numeric_mode'] == 'fast'
    assert manifest['defines_A'] == ''
    assert manifest['defines_B'] == '-D ' + quality.DEFINE
    assert {a: digest(arms / (a + '.so')) for a in ('A', 'B')} == manifest['hashes']
    assert report['source_sha'] == SOURCE and report['catalog_source'] == quality.CATALOG
    assert report['binding_sha256'] == manifest['hashes']['B']
    assert report['fixture'] == quality.FIXTURE and report['abi_version'] == 2
    assert report['bound'] == quality.ABS_SCALED_BOUND
    assert report['no_regression_tolerance'] == 0
    assert report['status'] == 'FAIL' and report['failures'] == ['nt-vector']
    assert report['variant_failures'] == {str(a): ['nt-vector'] for a in ARMS[1:]}
    assert report['call_counts'] == {str(a): len(quality.CASES) for a in ARMS}
    assert len(report['cases']) == len(quality.CASES)
    for row, case in zip(report['cases'], quality.CASES):
        name, m, n, k, nt, alias, kind = case
        assert row['case'] == name and row['shape'] == [m, n, k]
        assert row['nt'] is nt and row['alias'] is alias
        assert row['incumbent'] == ('gpu-zero-contract' if k == 0 else ('core-nt' if nt else 'sdk-nn'))
        assert set(row['metrics']) == {str(a) for a in ARMS}
        assert set(row['variant_status']) == {str(a) for a in ARMS[1:]}
        base = row['metrics']['0']
        for metric in row['metrics'].values():
            assert isinstance(metric['finite'], bool)
            assert isinstance(metric['scaled_error'], (int, float))
            assert re.fullmatch('[0-9a-f]{64}', metric['sha256'])
        statuses = {}
        for a in ARMS[1:]:
            metric = row['metrics'][str(a)]
            err = metric['scaled_error']
            good = (metric['finite'] and base['finite'] and math.isfinite(err)
                    and math.isfinite(base['scaled_error'])
                    and 0 <= err <= quality.ABS_SCALED_BOUND
                    and err <= base['scaled_error'])
            if a in (1, 5):
                good = good and row['direct_shared_exact']
            statuses[str(a)] = 'PASS' if good else 'FAIL'
        assert statuses == row['variant_status']
        if name == 'nt-vector':
            assert n == 1 and row['status'] == 'FAIL'
            assert all(v == 'FAIL' for v in statuses.values())
        else:
            assert n >= 2 and row['status'] == 'PASS'
            assert all(v == 'PASS' for v in statuses.values())
            assert row['direct_shared_exact'] is True
            assert all(v['sha256'] == base['sha256'] for v in row['metrics'].values())
    return arms / 'B.so', manifest, report_path


def child(args, so):
    import numpy as np
    name, m, n, k, nt, alias = SHAPES[args.shape]
    assert n >= 2 and args.arm in ARMS
    spec = importlib.util.spec_from_file_location('_mojolearn_gemm_probe', so)
    b = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b)
    assert b.abi_version() == 2 and b.enabled() == 1
    assert {a: int(b.count(a)) for a in ARMS} == dict.fromkeys(ARMS, 0)
    rng = np.random.default_rng(102604)
    a = rng.standard_normal((m, k)).astype('float32')
    physical_b = a if alias else rng.standard_normal((n, k) if nt else (k, n)).astype('float32')
    c = np.full((m, n), np.nan, dtype='float32')
    inputs = [hashlib.sha256(x.tobytes()).hexdigest() for x in (a, physical_b)]
    t0 = time.perf_counter_ns()
    reached = b.gemm(a.ctypes.data, physical_b.ctypes.data, c.ctypes.data,
                     [m, n, k, int(nt), args.arm, int(alias)])
    # Binding downloads C and synchronizes before return: call includes completion.
    t1 = time.perf_counter_ns()
    checksum = float(c.sum(dtype=np.float64))  # Full first read, no oracle computation.
    t2 = time.perf_counter_ns()
    counts = {a: int(b.count(a)) for a in ARMS}
    assert reached == args.arm
    assert counts == {a: int(a == args.arm) for a in ARMS}
    assert bool(np.isfinite(c).all()) and math.isfinite(checksum)
    result = dict(shape=name, dimensions_M_N_K=[m, n, k], nt=nt, alias=alias,
                  variant=args.arm, reached=reached, call_counts=counts,
                  call_completion_ms=(t1-t0)/1e6, first_read_ms=(t2-t1)/1e6,
                  total_ms=(t2-t0)/1e6, checksum=checksum, input_sha256=inputs,
                  output_sha256=hashlib.sha256(c.tobytes()).hexdigest())
    print('CATALOG-MATRIX-TIME ' + json.dumps(result, sort_keys=True), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('source', 'tag', 'report', 'report_sha256'):
        p.add_argument(arg)
    p.add_argument('--shape', type=int, choices=range(len(SHAPES)))
    p.add_argument('--arm', type=int, choices=ARMS)
    args = p.parse_args()
    os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                      MOJOLEARN_VENDOR='apple', MOJOLEARN_NUMERIC_MODE='fast')
    so, manifest, report_path = validate(args)
    if args.shape is not None:
        assert args.arm is not None
        child(args, so)
        return
    assert args.arm is None
    out = Path.home() / 'mq/out' / (args.tag + '-timing')
    out.mkdir(parents=True, exist_ok=False)
    records = []
    for shape_index, shape in enumerate(SHAPES):
        for arm in ARMS:
            cmd = [sys.executable, str(Path(__file__).resolve()), SOURCE, args.tag,
                   str(report_path), args.report_sha256, '--shape', str(shape_index), '--arm', str(arm)]
            log = out / f'{shape[0]}-G{arm}.log'
            with log.open('x') as stream:
                result = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT)
            assert result.returncode == 0, 'child failed; inspect ' + str(log)
            lines = [x for x in log.read_text().splitlines() if x.startswith('CATALOG-MATRIX-TIME ')]
            assert len(lines) == 1
            row = json.loads(lines[0].split(' ', 1)[1])
            assert row['variant'] == arm and row['shape'] == shape[0]
            incumbent = records[-arm] if arm else row
            assert row['input_sha256'] == incumbent['input_sha256']
            row['output_equal_incumbent'] = row['output_sha256'] == incumbent['output_sha256']
            row['quality_needs_review'] = not row['output_equal_incumbent']
            # Preserve the sole scored result even if words differ; no retries.
            records.append(row)
            print('CATALOG-MATRIX-TIME ' + json.dumps(row, sort_keys=True), flush=True)
    report = dict(source_sha=SOURCE, harness_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  binding_hashes=manifest['hashes'], quality_report_sha256=args.report_sha256,
                  eligibility='matrix n>=2 only; vector quality FAIL excluded by predeclared contract',
                  quality_status='MATRIX_ONLY: 11 fixtures PASS; unrestricted FAIL nt-vector',
                  scored_calls_per_shape_arm=1, fresh_process_per_call=True,
                  warmups=0, opponents=0, production_admission=False,
                  quality_needs_review=any(r['quality_needs_review'] for r in records),
                  machine='Apple M3 Ultra',
                  required_next='actual estimator quality before any production integration', records=records)
    (out / 'report.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
