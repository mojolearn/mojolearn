#!/usr/bin/env python3
"""Index retained compile receipts without importing or executing a binding."""
from __future__ import annotations
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / 'experiments/six_lane_integration'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def build(roots):
    plan = json.loads((STORE / 'build_plan.json').read_text())
    catalog = json.loads((STORE / 'catalog.json').read_text())
    matrix = json.loads((STORE / 'matrix.json').read_text())
    current_hashes = {}
    records = []
    skipped = []
    for root in roots:
        for receipt in sorted(root.resolve().rglob('receipt.json')):
            try:
                r = json.loads(receipt.read_text())
            except json.JSONDecodeError:
                skipped.append(dict(path=str(receipt),reason='receipt incomplete at snapshot'))
                continue
            if not r.get('source_files') or not r.get('key'):
                continue
            mismatches = []
            for name, old_hash in r['source_files'].items():
                if name not in current_hashes:
                    path = ROOT / name
                    current_hashes[name] = digest(path) if path.is_file() else None
                if current_hashes[name] != old_hash:
                    mismatches.append(name)
            artifact = receipt.parent / Path(r['artifact']).name
            if not artifact.is_file() and Path(r['artifact']).is_file():
                artifact = Path(r['artifact'])
            artifact_valid = artifact.is_file() and digest(artifact) == r.get('artifact_sha256')
            log = receipt.parent / 'build.log'
            if not log.exists() and Path(r['log']).is_file():
                log = Path(r['log'])
            record = dict(key=r['key'],binding=r['binding'],mode=r['mode'],vendor=r['vendor'],
                source_sha=r['source_sha'],source_closure_sha256=r['source_closure_sha256'],
                status=r['status'],returncode=r.get('returncode'),defines=r['defines'],
                configurations=r['configurations'],compiler=r['compiler'],
                compiler_sha256=r['compiler_sha256'],hardware=r['hardware'],argv=r['argv'],
                receipt=str(receipt),receipt_sha256=digest(receipt),log=str(log),
                log_sha256=digest(log) if log.exists() else None,
                artifact=str(artifact),artifact_sha256=r.get('artifact_sha256'),
                artifact_retained_and_hash_checked=bool(artifact_valid),
                current_source_closure_matches=not mismatches,changed_source_files=mismatches,
                qualification='compile only; no runtime reach, timing, quality or identity evidence')
            records.append(record)
    by_key = {}
    for r in records:
        by_key.setdefault(r['key'],[]).append(r)
    jobs = []
    for job in plan['jobs']:
        evidence = by_key.get(job['key'],[])
        matched = [r for r in evidence if r['current_source_closure_matches']]
        compiled = [r for r in matched if r['status']=='COMPILED' and r['artifact_retained_and_hash_checked']]
        status = 'COMPILED' if compiled else 'FAILED' if any(r['status']=='FAILED' for r in matched) else 'NOT_COMPILED'
        jobs.append(dict(key=job['key'],binding=job['binding'],mode=job['mode'],vendor=job['vendor'],
            status=status,configurations=job['configurations'],defines=job['defines'],
            evidence=[r['receipt'] for r in evidence],
            compilation_basis='exact binding/vendor/mode/defines and current source closure; original source SHA retained'))
    configs = {c['id']:c for c in matrix['configurations']}
    entries = []
    for e in catalog['entries']:
        related = [j for j in jobs if any(e['id'] in configs[c['configuration']]['members'] for c in j['configurations'] if c['configuration'] in configs)]
        entries.append(dict(id=e['id'],source_id=e['source_id'],original_id=e['original_id'],
            implementation=e['implementation'],qualification=e['qualification'],
            gaps=e['gaps'],source_record=e['source_record'],arms=[a['id'] for a in e['arms']],
            build_configuration_counts=dict(Counter(j['status'] for j in related)),
            build_keys=[j['key'] for j in related],
            note='Combined-configuration builds cover only their recorded defines. They do not qualify omitted alternatives or establish runtime reach.'))
    return dict(schema='mojolearn.six-lane-build-coverage/1',records=records,jobs=jobs,implementation_ledger=entries,
        summary=dict(planned=len(jobs),statuses=dict(Counter(j['status'] for j in jobs)),retained_receipts=len(records),
            by_vendor={v:dict(Counter(j['status'] for j in jobs if j['vendor']==v)) for v in ('apple','host','nvidia')}),
        skipped_incomplete_receipts=skipped,unsupported=plan['unsupported'],blocked=plan['blocked'],
        qualification='NO ESTIMATOR EXECUTION; NO RUNTIME TESTS; NO TIMING, QUALITY, IDENTITY OR PROMOTION CLAIM',
        policy='This ledger does not build, import, run, measure, promote, admit a board row, or remove original evidence.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence-root',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,default=STORE/'build_coverage.json')
    a=p.parse_args();d=build(a.evidence_root)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
    print(json.dumps(d['summary'],sort_keys=True))


if __name__=='__main__':
    main()
