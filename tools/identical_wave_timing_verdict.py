#!/usr/bin/env python3
"""KEEP/DROP per timed case from one ON and one OFF sample on NVIDIA and AMD.

Inputs are harvested wave directories (or box paths) holding the runner's
timing receipt and per-case result.json files. A case is KEEP only when ON is
faster than OFF (fit + infer, the single reported sample after one untimed
warmup) on BOTH vendors, its cross-vendor identity is PROVEN in the partial
proof, and the wave quality receipt is PASS. Everything else is DROP with the
reason. Single samples: no medians, intervals or speedup claims beyond the
recorded numbers. Opponents are never involved.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

BACKENDS = {'nvidia': 'cuda', 'amd': 'hip'}


def load(path):
    return json.loads(Path(path).read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--nvidia-wave', required=True, type=Path)
    p.add_argument('--amd-wave', required=True, type=Path)
    p.add_argument('--tag', required=True)
    p.add_argument('--proof', required=True, type=Path)
    p.add_argument('--plan', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    plan = load(a.plan)
    family = {c['lane'] + '--' + c['dataset']: c.get('family') for c in plan['cases']}
    proof = load(a.proof)
    waves = {'nvidia': a.nvidia_wave, 'amd': a.amd_wave}
    receipts = {v: load(w / ('timing-' + a.tag + '.json')) for v, w in waves.items()}
    quality = {v: load(w / 'quality.json').get('status') for v, w in waves.items()}
    cases = None
    for vendor, receipt in receipts.items():
        if receipt.get('phase') != 'timing':
            p.error('not a timing receipt: ' + vendor)
        if cases is None:
            cases = receipt.get('cases')
        elif receipt.get('cases') != cases:
            p.error('vendors timed different case sets')
    rows = []
    for tag in cases:
        row = {'case': tag, 'family': family.get(tag), 'samples': {}}
        reasons = []
        for vendor, wave in waves.items():
            backend = BACKENDS[vendor]
            for arm in ('on', 'off'):
                path = wave / arm / ('timing-' + a.tag) / (tag + '--' + backend) / 'result.json'
                if not path.is_file():
                    reasons.append('missing ' + vendor + '/' + arm)
                    continue
                result = load(path)
                if result.get('warmups') != 1 or result.get('timing_samples') != 1:
                    reasons.append('contract ' + vendor + '/' + arm)
                total = result['fit_ms'] + (result.get('infer_ms') or 0.0)
                row['samples'][vendor + '/' + arm] = {'fit_ms': result['fit_ms'], 'infer_ms': result.get('infer_ms'),
                                                     'total_ms': total, 'digest': result.get('digest'),
                                                     'result_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        faster = {}
        for vendor in waves:
            on, off = row['samples'].get(vendor + '/on'), row['samples'].get(vendor + '/off')
            if on and off:
                faster[vendor] = on['total_ms'] < off['total_ms']
                row[vendor + '_on_over_off'] = round(on['total_ms'] / off['total_ms'], 4) if off['total_ms'] else None
        proven = all(tag in proof['classes'][arm].get('PROVEN', []) for arm in ('on', 'off'))
        if not proven:
            reasons.append('cross-vendor identity not PROVEN')
        if any(q != 'PASS' for q in quality.values()):
            reasons.append('wave quality not PASS')
        if len(faster) != 2:
            reasons.append('incomplete samples')
        elif not all(faster.values()):
            reasons.append('ON not faster on ' + ','.join(v for v, f in faster.items() if not f))
        row['verdict'] = 'DROP' if reasons else 'KEEP'
        row['reasons'] = reasons
        rows.append(row)
    report = {'schema': 1, 'rule': 'KEEP iff ON total (fit+infer, one sample after one untimed warmup) < OFF on NVIDIA AND AMD, '
                                   'case PROVEN cross-vendor in both arms, quality PASS on both boxes',
              'tag': a.tag, 'proof_sha256': hashlib.sha256(a.proof.read_bytes()).hexdigest(),
              'timing_receipts': {v: str(w / ('timing-' + a.tag + '.json')) for v, w in waves.items()},
              'quality_status': quality, 'rows': rows,
              'counts': {k: sum(r['verdict'] == k for r in rows) for k in ('KEEP', 'DROP')}}
    with a.out.open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print('TIMING_VERDICT', json.dumps(report['counts']), a.out)
    for r in rows:
        print('%-36s %-6s nv=%s amd=%s %s' % (r['case'], r['verdict'], r.get('nvidia_on_over_off'), r.get('amd_on_over_off'), ';'.join(r['reasons'])))
    return 0


if __name__ == '__main__':
    sys.exit(main())
