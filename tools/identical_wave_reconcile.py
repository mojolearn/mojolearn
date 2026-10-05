#!/usr/bin/env python3
"""Reconcile a revised wave's quality receipt from verified successes only.

Inputs: the composed revision (identical_wave_revision.py), its original
quality.json (hash bound in revision-provenance.json) and append-only
supplements (identical_wave_supplement.py). Per required gate and arm:
  original PASS / NOT_APPLICABLE  -> carried, linked to the original receipt;
  original FAIL + supplement PASS -> PASS, linked to the supplement receipt
                                     (latest attempt decides; all attempts listed);
  dart_reference failure          -> PASS only with an adjudication receipt
                                     whose status is PASS for that arm, else PENDING;
  anything else                   -> FAIL.
Always writes a fresh quality-reconciliation-<n>.json. Writes quality.json (the
receipt identical_wave_runner.py timing requires) only when every cell is
PASS/NOT_APPLICABLE, and never overwrites one. Nothing is rerun here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wave', required=True, type=Path)
    p.add_argument('--plan', required=True, type=Path)
    p.add_argument('--dart-adjudication', type=Path, help='dart-quality-adjudication.json from the DART statistics lane (ON vs OFF retained predictions)')
    p.add_argument('--dart-on-receipt', type=Path, help='The adjudicated ON prediction receipt (receipt.json) it names; its prediction hashes must equal this box\'s retained ON outputs')
    a = p.parse_args()
    wave = a.wave.resolve()
    identity = load(wave / 'wave.json')
    # box-run-2 (2026-10-05): a wave that was never composed into a revision is
    # reconciled IN PLACE: its own quality.json is the original receipt, and a
    # PASS is written to quality-reconciled.json (quality.json is never touched);
    # timing takes it with identical_wave_runner.py --quality-receipt.
    in_place = not (wave / 'revision-provenance.json').exists()
    if in_place:
        prov = {'status': 'PRODUCTS_FROZEN_QUALITY_RECONCILIATION_REQUIRED', 'new_plan_sha256': digest(a.plan),
                'old_plan_sha256': digest(a.plan), 'old_revision': str(wave), 'old_quality_sha256': digest(wave / 'quality.json')}
    else:
        prov = load(wave / 'revision-provenance.json')
    prepared = load(wave / 'prepare.json')
    if prepared.get('status') != 'PASS' or prepared.get('identity') != identity:
        p.error('revision prepare receipt invalid')
    if prov.get('status') != 'PRODUCTS_FROZEN_QUALITY_RECONCILIATION_REQUIRED' or prov['new_plan_sha256'] != digest(a.plan):
        p.error('revision provenance incomplete or for another plan')
    if identity['plan_sha256'] != digest(a.plan):
        p.error('plan differs from wave identity')
    old_root = Path(prov['old_revision'])
    old_quality_path = old_root / 'quality.json'
    if digest(old_quality_path) != prov['old_quality_sha256']:
        p.error('original quality receipt changed since composition')
    old = load(old_quality_path)
    old_identity = dict(identity, plan_sha256=prov['old_plan_sha256'])
    if old.get('identity') != old_identity:
        p.error('original quality receipt identity mismatch')
    plan = load(a.plan)
    required = plan['required_quality_gates']
    gates = {g['id']: g for g in plan['quality_gates']}
    adjudication = None
    if a.dart_adjudication:
        adjudication = load(a.dart_adjudication)
        adjudication_sha = digest(a.dart_adjudication)
    report = {'identity': identity, 'phase': 'quality', 'status': 'INCOMPLETE', 'arms': {},
              'method': 'reconciled: original successes carried, failed cells replaced only by append-only PASS supplements',
              'original_quality': {'path': str(old_quality_path), 'sha256': prov['old_quality_sha256']},
              'opponents': 'store only; never executed', 'reconciled_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    counts = {}
    for arm in ('on', 'off'):
        original = {s['id']: s for s in old['arms'][arm]}
        if set(original) != set(required):
            p.error('original gate inventory incomplete: ' + arm)
        steps = []
        for gate_id in required:
            prior = original[gate_id]
            applicable = arm in gates[gate_id].get('arms', ['on', 'off'])
            row = {'id': gate_id, 'original': {k: prior.get(k) for k in ('status', 'rc')}}
            if prior.get('status') == 'NOT_APPLICABLE':
                row['status'] = 'NOT_APPLICABLE' if not applicable else 'FAIL'
                row['source'] = 'original'
            elif prior.get('status') == 'PASS' and prior.get('rc') == 0:
                row.update(status='PASS', rc=0, source='original',
                           log=str(old_root / arm / 'quality' / (gate_id + '.log')))
            else:
                # Every attempt is listed; the LATEST attempt decides. Earlier
                # failed attempts (e.g. harness packaging defects) stay visible.
                base = wave / 'supplements' / (gate_id + '--' + arm)
                attempts, n = [], 1
                folder = base
                while folder.is_dir():
                    attempts.append(folder)
                    n += 1
                    folder = base.with_name(base.name + '--attempt' + str(n))
                if attempts:
                    row['attempts'] = [{'folder': str(f), 'status': (load(f / 'receipt.json').get('status') if (f / 'receipt.json').is_file() else 'NO_RECEIPT')} for f in attempts]
                folder = attempts[-1] if attempts else base
                receipt_path = folder / 'receipt.json'
                if gate_id == 'dart_reference':
                    # The adjudication compares retained ON predictions with OFF on
                    # frozen OFF-only margins. It transfers to this box only when this
                    # box's retained ON prediction files are byte-identical to the
                    # adjudicated ones (same-bits contract); otherwise it stays pending.
                    verdict, evidence = None, {}
                    if adjudication is not None and a.dart_on_receipt and arm == 'on':
                        on_receipt = load(a.dart_on_receipt)
                        local = {name: digest(old_root / 'on/quality/dart_reference-outputs' / (name + '.npz'))
                                 for name in on_receipt.get('prediction_sources', {})}
                        matched = (bool(local) and on_receipt.get('arm') == 'on' and on_receipt.get('source_sha') == identity['sha']
                                   and all(local[n] == on_receipt['prediction_sources'][n]['sha256'] for n in local))
                        decisions = adjudication.get('decisions', {})
                        if adjudication.get('source_sha') != identity['sha'] or not decisions:
                            verdict = None
                        elif not matched:
                            verdict = 'PREDICTIONS_DIFFER'
                        elif adjudication.get('status') == 'PASS' and all(d.get('status') == 'PASS' for d in decisions.values()):
                            verdict = 'PASS'
                        else:
                            verdict = adjudication.get('status')
                        evidence = {'adjudication': str(a.dart_adjudication), 'adjudication_sha256': adjudication_sha,
                                    'on_receipt': str(a.dart_on_receipt), 'on_receipt_sha256': digest(a.dart_on_receipt),
                                    'local_prediction_sha256': local, 'predictions_match_adjudicated': matched,
                                    'decisions': {k: d.get('status') for k, d in decisions.items()}}
                    if verdict == 'PASS':
                        row.update(status='PASS', rc=0, source='adjudication', **evidence)
                    elif verdict in ('FAIL', 'INCONCLUSIVE', 'INSUFFICIENT', 'PREDICTIONS_DIFFER'):
                        row.update(status='FAIL', source='adjudication', verdict=verdict, **evidence)
                    else:
                        row.update(status='PENDING_ADJUDICATION', source='dart statistics lane', **evidence)
                elif receipt_path.is_file():
                    sup = load(receipt_path)
                    log = folder / (gate_id + '.log')
                    valid = (sup.get('identity') == identity and sup.get('gate') == gate_id and sup.get('arm') == arm
                             and sup.get('build_products_sha256') == digest(wave / arm / 'build-products.json')
                             and log.is_file() and sup.get('log_sha256') == digest(log))
                    ok = valid and sup.get('status') == 'PASS' and sup.get('rc') == 0
                    row.update(status='PASS' if ok else 'FAIL', rc=sup.get('rc'), source='supplement',
                               receipt=str(receipt_path), receipt_sha256=digest(receipt_path), receipt_valid=valid)
                else:
                    row.update(status='FAIL', source='no supplement')
            steps.append(row)
        report['arms'][arm] = steps
        counts[arm] = {}
        for row in steps:
            counts[arm][row['status']] = counts[arm].get(row['status'], 0) + 1
    good = all(row['status'] in ('PASS', 'NOT_APPLICABLE') for steps in report['arms'].values() for row in steps)
    report['status'] = 'PASS' if good else 'INCOMPLETE_OR_FAILED'
    report['counts'] = counts
    n = 1
    while (wave / ('quality-reconciliation-%d.json' % n)).exists():
        n += 1
    out = wave / ('quality-reconciliation-%d.json' % n)
    with out.open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    if good:
        with (wave / ('quality-reconciled.json' if in_place else 'quality.json')).open('x') as stream:  # never overwrite an existing quality receipt
            stream.write(json.dumps(report, indent=2) + '\n')
    print('QUALITY_RECONCILED', report['status'], json.dumps(counts, sort_keys=True), out, flush=True)
    return 0 if good else 1


if __name__ == '__main__':
    sys.exit(main())
