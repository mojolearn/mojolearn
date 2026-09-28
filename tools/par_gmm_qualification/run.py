#!/usr/bin/env python3
"""Bounded par-gmm qualification on a prebuilt, exclusively queued two-GPU host."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

SOURCE = 'c46f776140e997a3db602c97dea53985601e5131'
AMD_SHA = '5abd39df5941bd63cc0990b4357c787caa8102064e3f279414a35ee3c17160e7'
FIXTURES = ('base', 'ties', 'hashed', 'wide', 'denormal', 'denormal_ftz', 'dupes', 'odd', 'negative')
PARTS = ('train', 'infer', 'model', 'batch', 'stepfull', 'batchgrad', 'batchscale', 'ragged')
NA = dict(stepfull='n/a:no-decode-state', batchgrad='n/a:no-backward',
          batchscale='n/a:not-in-the-serving-set', ragged='n/a:no-sequence-axis')


def strict_candidate(table):
    if table.get('lane_revisions', {}).get('par-gmm') != 'classic-kmeanspp-init-1':
        raise ValueError('candidate has the wrong par-gmm revision')
    for fixture in FIXTURES:
        cell = table.get('cells', {}).get('par-gmm/' + fixture, {})
        for part in PARTS:
            entry = cell.get(part, {})
            ref = entry.get('ref', '')
            cols = entry.get('cols', {})
            if entry.get('conflict') or any(isinstance(value, list) for value in cols.values()):
                raise ValueError(f'{fixture}/{part}: conflicting witnesses')
            if part in NA:
                if ref != NA[part]:
                    raise ValueError(f'{fixture}/{part}: unexpected applicability declaration')
            elif not re.fullmatch('[0-9a-f]{16}', ref):
                raise ValueError(f'{fixture}/{part}: missing numerical reference')
            for vendor in ('amd', 'nvidia'):
                index = cols.get(vendor)
                if type(index) is not int or not 0 <= index < len(table['records']):
                    raise ValueError(f'{fixture}/{part}: missing {vendor} witness')
                if table['records'][index]['class'] != vendor:
                    raise ValueError(f'{fixture}/{part}: inconsistent witness class')



def strict_physical(report, fixture):
    if (report.get('state') != 'VERIFIED' or report.get('passed') is not True
            or report.get('vendor') != 'cuda' or report.get('devices') != [0, 1]
            or report.get('fixtures') != [fixture] or report.get('witness_refusal')):
        raise ValueError(f'{fixture}: not a verified CUDA two-device column')
    witnesses = report.get('cell_witnesses', [])
    if len(witnesses) != 1 or witnesses[0].get('fixture') != fixture or witnesses[0].get('witness_refusal'):
        raise ValueError(f'{fixture}: missing per-fixture placement witness')
    cell = witnesses[0].get('witness', {})
    pools = cell.get('detail', [])
    if cell.get('placement') != 'physical' or cell.get('devices') != [0, 1] or not pools:
        raise ValueError(f'{fixture}: no witnessed physical pool')
    for pool in pools:
        workers = pool.get('workers', [])
        if pool.get('devices') != [0, 1] or not pool.get('cooperative') or len(workers) != 1:
            raise ValueError(f'{fixture}: not the requested cooperative device group')
        uuids = workers[0].get('devices', [])
        if len(uuids) != 2 or not all(uuids) or len(set(uuids)) != 2:
            raise ValueError(f'{fixture}: repeated or missing physical GPU identity')
    if report.get('witness', {}).get('placement') != 'physical':
        raise ValueError(f'{fixture}: placement is not physical')
    if not report.get('cells') or any(c['verdict'] not in ('IDENTICAL', 'N/A') for c in report['cells']):
        raise ValueError(f'{fixture}: missing or failed compared part')


def bounded(command, cwd, env, log, stdout=None):
    with log.open('wb') as err, (stdout or log.with_suffix('.stdout')).open('wb') as out:
        child = subprocess.Popen(command, cwd=cwd, env=env, stdout=out, stderr=err, start_new_session=True)
        try:
            code = child.wait(timeout=120)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            raise
        if code:
            raise RuntimeError(f'exit {code}: {command}; see {log} and {stdout or log.with_suffix(".stdout")}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tree', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--amd-record', type=Path, required=True)
    ap.add_argument('--execute', action='store_true')
    args = ap.parse_args()
    print(json.dumps(dict(execute=args.execute, source=SOURCE, fixtures=FIXTURES,
                         one_device_captures=9, two_device_checks=9, timeout_seconds=120)), flush=True)
    if not args.execute:
        return 0
    tree, out = args.tree.resolve(), args.out.resolve()
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=tree, text=True).strip() != SOURCE:
        raise ValueError('qualification source does not match the pinned commit')
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=tree, text=True).strip():
        raise ValueError('qualification source is dirty')
    if hashlib.sha256(args.amd_record.read_bytes()).hexdigest() != AMD_SHA:
        raise ValueError('AMD witness SHA-256 mismatch')
    mask = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if len(mask) != 2 or len(set(mask)) != 2 or '-1' in mask or not all(mask):
        raise ValueError('the exclusive queue job must expose two distinct GPUs')
    # The queue owns the physical visibility mask. Ordinals below are local to it.
    env = dict(os.environ, MOJOLEARN_NUMERIC_MODE='identical', PYTHONPATH=str(tree / 'python'))
    for name in ('MOJOLEARN_PAR_DEVICES', 'MOJOLEARN_GMM_DEVICE_COUNT', 'MOJOLEARN_KMEANS_DEVICE_COUNT'):
        env.pop(name, None)
    out.mkdir(parents=True, exist_ok=False)  # preserve old evidence; no automatic retry/overwrite
    amd = out / 'amd.par-gmm.json'
    amd.write_bytes(args.amd_record.read_bytes())
    python = str(tree / '.pixi/envs/default/bin/python')
    bounded([python, '-c', 'from mojolearn import _backend as b; assert b.vendor()=="cuda"; '
             'assert b.binding("_mojolearn_mixture").gmm_parallel_available()==1'],
            tree, env, out / 'preflight.log')
    records = [amd]
    for fixture in FIXTURES:
        record = out / ('cuda.' + fixture + '.json')
        bounded([python, 'tools/identity_break.py', '--lanes', 'par-gmm', '--fixtures', fixture,
                 '--repeats', '1', '--require-backend', 'cuda', '--fail-on-refused', '--json', str(record)],
                tree, dict(env, MOJOLEARN_PAR_DEVICES='0'), out / ('capture.' + fixture + '.log'))
        records.append(record)
    candidate = out / 'candidate-table.json'
    command = [python, '-m', 'mojolearn', 'verify', '--all', '--lanes', 'par-gmm', '--batch-checks',
               '--reference-table', 'python/mojolearn/verify_reference/table.json', '--emit-reference', str(candidate)]
    for record in records:
        command += ['--records', str(record)]
    bounded(command, tree, env, out / 'candidate.log')
    strict_candidate(json.loads(candidate.read_text()))
    for fixture in FIXTURES:
        report = out / ('two-device.' + fixture + '.json')
        bounded([python, '-m', 'mojolearn', 'verify', '--par', 'all', '--fixtures', fixture, '--lanes', 'par-gmm',
                 '--par-devices', '0,1', '--repeats', '1', '--reference-table', str(candidate), '--json'],
                tree, env, out / ('two-device.' + fixture + '.log'), stdout=report)
        strict_physical(json.loads(report.read_text()), fixture)
    summary = dict(source=SOURCE, amd_sha256=AMD_SHA, passed=True, fixtures=list(FIXTURES),
                   qualification_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   records={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in records + [candidate] + sorted(out.glob('two-device.*.json'))})
    (out / 'qualification.json').write_text(json.dumps(summary, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
