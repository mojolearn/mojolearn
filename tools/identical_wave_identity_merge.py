#!/usr/bin/env python3
"""Merge a broad identity receipt with reruns of ONLY its failed cases.

Base: <wave>/identity.json (complete case inventory, some cases failed, e.g. on
worker harness defects). Reruns: identity-<tag>.json from
identical_wave_runner.py identity --tag <tag> --cases <failed cases>. A rerun
may replace a case only when that case failed in the base (no re-rolling of
passing cases); superseded base steps are kept inside the merged receipt. The
output is a new file (never overwritten) in the runner's identity schema, which
identical_wave_compare.py and timing (--identity-receipt) accept.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wave', required=True, type=Path)
    p.add_argument('--rerun', required=True, type=Path, action='append')
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    base_path = a.wave / 'identity.json'
    base = json.loads(base_path.read_text())
    merged = {'identity': base['identity'], 'phase': 'identity', 'status': 'INCOMPLETE', 'arms': {},
              'opponents': 'store only; never executed',
              'merged_from': {'base': {'path': str(base_path), 'sha256': digest(base_path)}, 'reruns': []}}
    reruns = []
    for path in a.rerun:
        doc = json.loads(path.read_text())
        if doc.get('identity') != base['identity'] or doc.get('phase') != 'identity':
            p.error('rerun identity differs: ' + str(path))
        reruns.append(doc)
        merged['merged_from']['reruns'].append({'path': str(path), 'sha256': digest(path), 'cases': doc.get('cases'),
                                                'worker': doc.get('worker')})
    for arm in ('on', 'off'):
        steps = base['arms'][arm]
        cases = {}
        for step in steps:
            cases.setdefault(step['id'].rsplit('--', 1)[0], []).append(step)
        superseded = []
        for doc in reruns:
            new = {}
            for step in doc['arms'][arm]:
                new.setdefault(step['id'].rsplit('--', 1)[0], []).append(step)
            for case, rows in new.items():
                if case not in cases:
                    p.error('rerun case not in base: ' + case)
                if all(r.get('status') == 'PASS' for r in cases[case]):
                    p.error('refused: rerun replaces a case that passed in the base: ' + case)
                superseded.append({'case': case, 'base_steps': cases[case]})
                cases[case] = [dict(r, rerun=True) for r in rows]
        merged['arms'][arm] = [r for rows in cases.values() for r in rows]
        merged.setdefault('superseded', {})[arm] = superseded
    good = all(r.get('status') == 'PASS' for rows in merged['arms'].values() for r in rows)
    merged['status'] = 'PASS' if good else 'INCOMPLETE_OR_FAILED'
    with a.out.open('x') as stream:
        stream.write(json.dumps(merged, indent=2) + '\n')
    print('IDENTITY_MERGED', merged['status'], a.out, flush=True)
    return 0 if good else 1


if __name__ == '__main__':
    sys.exit(main())
