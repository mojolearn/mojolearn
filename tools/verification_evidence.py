#!/usr/bin/env python3
"""Generate the wheel's historical evidence inventory; never certifies a release.

Pairs must name the same full commit, input bytes, held-out bytes and protocol.
Multi-device comparisons additionally require identical binding digests, distinct
requested GPU indices and a matching single-device record. This trusts recorded
metadata; it does not attest that a device was physically used.
"""
import argparse
import hashlib
import json
import pprint
from pathlib import Path
import subprocess

import verification_matrix as matrix

ROOT = Path(matrix.ROOT)
OUTPUT = ROOT / 'python/mojolearn/_verification_evidence_data.py'
PARTS = ('train', 'infer', 'model', 'batch', 'stepfull', 'batchgrad', 'batchscale', 'ragged', 'rlpair')


def devices(record):
    try:
        values = tuple(int(v) for v in str(record.get('package', {}).get('par_devices') or '0').split(','))
        return values if values and len(set(values)) == len(values) and min(values) >= 0 else ()
    except (ValueError, TypeError):
        return ()


def same_cell_context(a, b, key, part):
    commit = a.get('commit', '')
    if len(commit) != 40 or commit != b.get('commit') or a.get('mode') != 'identical' or b.get('mode') != 'identical':
        return False
    lane, fixture = key.split('/', 1)
    for field in ('fixtures', 'heldout'):
        av, bv = a.get(field, {}).get(fixture), b.get(field, {}).get(fixture)
        if not av or av != bv:
            return False
    if a.get('lane_revisions', {}).get(lane) != b.get('lane_revisions', {}).get(lane):
        return False
    if part not in ('train', 'infer', 'model'):
        protocol = part + '_protocol'
        if not a.get(protocol) or a[protocol] != b.get(protocol):
            return False
    for field in ('fixture_n', 'wide'):
        if a.get('package', {}).get(field) != b.get('package', {}).get(field):
            return False
    return True


def bindings(record):
    return {b['module']: b['sha256'] for b in record.get('package', {}).get('bindings', [])
            if b.get('module') and b.get('sha256')}


def admitted_multi(column, reference):
    j = column['record']
    if column['sabotage'] or column['cls'] not in matrix.GPU_CLASSES or len(devices(j)) < 2:
        return False
    # Reuse all normal exclusions, changing only the two-device restrictions.
    normalized = dict(j, package=dict(j.get('package', {}), par_devices='0'))
    normalized['vendor'] = str(j.get('vendor', '')).removesuffix('-two')
    return reference.admit(normalized, column['rel']) is None


def build():
    reference = matrix.load('python/mojolearn/_verify_reference.py', '_evidence_reference')
    harness = matrix.load('tools/identity_break.py', '_evidence_harness')
    columns = matrix.read_columns(reference)
    clean = [c for c in columns if not c['sabotage'] and c['admit'] is None and c['cls'] in ('cpu', *matrix.GPU_CLASSES)]
    partners = {}
    for c in clean:
        partners.setdefault((c['cls'], c['commit']), []).append(c)
    out = {name: dict(backend_records={}, negative_controls=[], multi_device=[]) for name in harness.LANES}
    sources = {}

    def source(c):
        path = c['rel']
        if path not in sources:
            sources[path] = dict(sha256=hashlib.sha256((ROOT / path).read_bytes()).hexdigest(),
                                 commit=c['commit'], vendor=c['vendor'])
        return path

    for c in clean:
        for key, cell in c['cells'].items():
            lane, fixture = key.split('/', 1)
            if lane in out and matrix.stable_digest(cell, 'train'):
                out[lane]['backend_records'].setdefault(c['cls'], {}).setdefault(fixture, source(c))

    for c in columns:
        multi = admitted_multi(c, reference)
        if not c['sabotage'] and not multi:
            continue
        for key, cell in c['cells'].items():
            lane, fixture = key.split('/', 1)
            if lane not in out or (multi and not lane.startswith('par-')):
                continue
            for part in PARTS:
                for base in partners.get((c['cls'], c['commit']), []):
                    if base['cls'] != c['cls'] or not same_cell_context(c['record'], base['record'], key, part):
                        continue
                    if multi:
                        if (len(devices(base['record'])) != 1 or not bindings(c['record'])
                                or bindings(c['record']) != bindings(base['record'])):
                            continue
                        a, b = matrix.stable_digest(cell, part), matrix.stable_digest(base['cells'].get(key), part)
                        if not a or not b:
                            continue
                        bucket = 'multi_device'
                        evidence = dict(result='identical' if a == b else 'different',
                                        devices=list(devices(c['record'])))
                    else:
                        if c['vendor'] != base['vendor']:
                            continue
                        if not devices(c['record']) or devices(c['record']) != devices(base['record']):
                            continue
                        if not matrix.negative_control_moves(cell, base['cells'].get(key), part):
                            continue
                        bucket = 'negative_controls'
                        evidence = dict(kind=c['sab_kind'], result='detected')
                    item = dict(fixture=fixture, part=part, record=source(c), clean_record=source(base), **evidence)
                    if item not in out[lane][bucket]:
                        out[lane][bucket].append(item)
                    break
    return dict(format='mojolearn.historical-evidence.v1', release_qualified=False,
                source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                source_tree_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).strip()),
                generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                harness_sha256=hashlib.sha256((ROOT / 'tools/identity_break.py').read_bytes()).hexdigest(),
                sources=sources, lanes=out,
                limitations=['Historical records are not qualification of this wheel or current source.',
                             'Backend records show repeated stable training hashes, not cross-backend agreement.',
                             'Negative controls cover only the listed fixtures and parts; build and harness controls differ.',
                             'Multi-device evidence compares recorded requests, not independently attested device use.',
                             'An empty evidence list means no qualifying pair found, not proof that none exists elsewhere.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    result = build()
    if args.write:
        # A Python module is included by both wheel builders without data-file rules.
        payload = pprint.pformat(result, sort_dicts=True, width=120)
        OUTPUT.write_text('"""Generated by tools/verification_evidence.py --write. Historical evidence only."""\n'
                          'DATA = ' + payload + '\n')
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
