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
    clean_failures = []
    for key in sorted(expected):
        a, b = clean.get('cells', {}).get(key), sabotage.get('cells', {}).get(key)
        if a:
            for field, verdict in a.items():
                if (field == 'verdict' or field.endswith('_verdict')) and verdict not in ('STABLE', 'N/A'):
                    clean_failures.append(dict(cell=key, part=field, verdict=verdict))
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
                clean_failures=clean_failures, complete=complete,
                passed=bool(complete and native and clean_native and not clean_failures
                            and all(r['detected'] for r in rows)))


def expected_oracle_failure(record):
    """Exit one is expected only for repeated training or property failures.

    Native faults can break an independent batch/RL-pair comparison while
    returning stable wrong training bytes. Refusals and repeat instability
    remain failures of the control run, not successful negative evidence.
    """
    cells = record.get('cells', {})
    observed = False
    mismatch_kinds = {part: 'BATCH_MOVED' for part in ('batch', 'batchgrad', 'batchscale', 'ragged', 'stepfull')}
    mismatch_kinds['rlpair'] = 'RLPAIR_MOVED'
    for cell in cells.values():
        hashes = cell.get('hashes', [])
        if (not isinstance(hashes, list) or len(hashes) < 2
                or not all(isinstance(h, str) and h for h in hashes) or len(set(hashes)) != 1):
            return False
        if any(value for field, value in cell.items() if field == 'error' or field.endswith('_error')):
            return False
        if cell.get('verdict') == 'DIVERGENT':
            errors = cell.get('oracle_errors', [])
            if len(errors) != len(hashes) or not all(errors):
                return False
            observed = True
        elif cell.get('verdict') != 'STABLE' or cell.get('oracle_errors'):
            return False
        for field, verdict in cell.items():
            if not field.endswith('_verdict') or verdict in ('STABLE', 'N/A'):
                continue
            part = field[:-len('_verdict')]
            values = cell.get(part, [])
            if (verdict != mismatch_kinds.get(part) or not isinstance(values, list)
                    or len(values) != len(hashes)
                    or not all(isinstance(v, str) and v.startswith(verdict + ':') for v in values)
                    or len(set(values)) != 1):
                return False
            observed = True
    return observed


def arm_environment(env, host, sabotage):
    """Keep the direct forest/byte-LM loaders on the same native arm."""
    result = dict(env, MOJOLEARN_HOST_DIR=str(host))
    for family in ('FOREST', 'BYTE_LM'):
        result[f'MOJOLEARN_{family}_HOST_BINARY'] = str(host / f'_mojolearn_{family.lower()}_host.so')
    for flag in ('MOJOLEARN_HOST_ALLOW_SABOTAGE', 'MOJOLEARN_FOREST_HOST_ALLOW_SABOTAGE',
                 'MOJOLEARN_BYTE_LM_HOST_ALLOW_SABOTAGE'):
        result.pop(flag, None)
        if sabotage:
            result[flag] = '1'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lanes', required=True)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--timeout', type=int, default=300)
    # ACCEPTED AND INERT SINCE 2026-09-20. It used to add --batch-grad,
    # --batch-scale and --ragged, so a tool named verify_cpu_BATCH recorded
    # no batch-size property unless the caller remembered a flag -- and its
    # help text claimed it added step/full, which line 139 already passed
    # unconditionally, so a reader who saw --step-full in the log assumed the
    # other three had come with it. All four parts are now the harness
    # default and this asks for nothing.
    parser.add_argument('--properties', action='store_true',
                        help='accepted and inert: every property part is recorded by default')
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
            arm_env = arm_environment(env, host, arm == 'sabotage')
            path = directory / f'cpu-{arm}.json'
            # Refuse accidental overwrite of a prior run, including failures.
            if path.exists():
                raise SystemExit(f'record already exists: {path}')
            command = [sys.executable, str(root / 'tools/identity_break.py'), '--lanes', lane,
                       '--repeats', '2', '--require-cpu', '--json', str(path)]
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
        result['passed'] &= (statuses['clean'] == 0 and
                             (statuses['sabotage'] == 0 or
                              (statuses['sabotage'] == 1 and expected_oracle_failure(records['sabotage']))))
        (directory / 'negative-controls.json').write_text(json.dumps(result, indent=2) + '\n')
        summary['lanes'][lane] = result
        (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(f'{lane}: {sum(r["detected"] for r in result["cells"])}/{len(result["cells"])} controls; passed={result["passed"]}', flush=True)
    summary['passed'] = all(r['passed'] for r in summary['lanes'].values())
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
