#!/usr/bin/env python3
"""Bounded all-fixture clean/native-control records, one CPU lane at a time.

Run after runpod_cpu_leg.sh builds production and sabotage host directories.
The output preserves failures and a summary after every lane; no refusal,
unstable result or unchanged native control counts as successful evidence.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import verification_evidence as evidence
import verification_matrix as matrix


def evaluate_pair(clean, sabotage, lane, fixtures):
    expected = {f'{lane}/{fixture}' for fixture in fixtures}
    rows = []
    for key in sorted(expected):
        a, b = clean.get('cells', {}).get(key), sabotage.get('cells', {}).get(key)
        context = (evidence.same_cell_context(clean, sabotage, key, 'train')
                   and clean.get('vendor') == sabotage.get('vendor')
                   and evidence.devices(clean) == evidence.devices(sabotage))
        rows.append(dict(cell=key, context_matches=context,
                         clean=matrix.stable_digest(a, 'train') if a else None,
                         sabotage=matrix.stable_digest(b, 'train') if b else None,
                         detected=bool(context and a and b and matrix.negative_control_moves(b, a, 'train'))))
    native = any(f.get('sabotage') for f in sabotage.get('host', {}).get('families', {}).values())
    clean_native = not any(f.get('sabotage') for f in clean.get('host', {}).get('families', {}).values())
    complete = (set(clean.get('cells', {})) == expected == set(sabotage.get('cells', {}))
                and clean.get('complete', True) and sabotage.get('complete', True))
    return dict(cells=rows, native_sabotage=native, clean_native=clean_native,
                complete=complete, passed=bool(complete and native and clean_native and all(r['detected'] for r in rows)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lanes', required=True)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('--properties', action='store_true',
                        help='also record step/full, gradients, batch-size and ragged properties')
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    root = Path(__file__).resolve().parents[1]
    prod = root / 'python/mojolearn/host'
    bad = root / 'python/mojolearn/host-sabotage'
    if not list(prod.glob('*.so')) or not list(bad.glob('*.so')):
        parser.error('both production and native sabotage builds are required')
    # Dependencies not being sabotaged must still resolve in the control arm.
    for source in prod.glob('*.so'):
        if not (bad / source.name).exists():
            shutil.copy2(source, bad / source.name)
    harness = matrix.load('tools/identity_break.py', '_cpu_batch_harness')
    lanes = args.lanes.split(',')
    if len(set(lanes)) != len(lanes) or set(lanes) - set(harness.LANES):
        parser.error('lanes must be known and unique')
    args.out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({key: '1' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                                   'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')})
    env.update(PYTHONPATH=str(root / 'python'), MOJOLEARN_NUMERIC_MODE='identical')
    summary = dict(lanes={}, passed=False)
    for lane in lanes:
        directory = args.out / lane
        directory.mkdir(exist_ok=True)
        statuses, records = {}, {}
        for arm, host in (('clean', prod), ('sabotage', bad)):
            arm_env = dict(env, MOJOLEARN_HOST_DIR=str(host))
            arm_env.pop('MOJOLEARN_HOST_ALLOW_SABOTAGE', None)
            if arm == 'sabotage':
                arm_env['MOJOLEARN_HOST_ALLOW_SABOTAGE'] = '1'
            path = directory / f'cpu-{arm}.json'
            # Refuse accidental overwrite of a prior run, including failures.
            if path.exists():
                raise SystemExit(f'record already exists: {path}')
            command = [sys.executable, str(root / 'tools/identity_break.py'), '--lanes', lane,
                       '--repeats', '2', '--require-cpu', '--step-full', '--json', str(path)]
            if args.properties:
                command.extend(['--batch-grad', '--batch-scale', '--ragged'])
            print(f'{lane}: {arm}', flush=True)
            with (directory / f'{arm}.log').open('w') as log:
                try:
                    statuses[arm] = subprocess.run(command, cwd=root, env=arm_env, stdout=log,
                                                   stderr=subprocess.STDOUT, timeout=args.timeout).returncode
                except subprocess.TimeoutExpired:
                    statuses[arm] = 124
            records[arm] = json.loads(path.read_text()) if path.exists() else {}
        result = evaluate_pair(records['clean'], records['sabotage'], lane, harness.FIXTURES)
        result['exit_codes'] = statuses
        result['passed'] &= all(code == 0 for code in statuses.values())
        (directory / 'negative-controls.json').write_text(json.dumps(result, indent=2) + '\n')
        summary['lanes'][lane] = result
        (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(f'{lane}: {sum(r["detected"] for r in result["cells"])}/{len(result["cells"])} controls; passed={result["passed"]}', flush=True)
    summary['passed'] = all(r['passed'] for r in summary['lanes'].values())
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
