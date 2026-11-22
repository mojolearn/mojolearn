#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Adapt retained AFCL results to the existing measurement-board input format.

Offline evidence glue only; never execute a workload, render a board, overwrite
old attempts or change defaults. Authored without running this program.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

GATES = ('full_dataset_coverage', 'actual_candidate_route',
         'task_quality_no_regression', 'api_contract', 'matched_full_operation')


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def positive(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def retained_json(value):
    """Keep nonfinite diagnostics explicit without emitting invalid JSON numbers."""
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite_value': repr(value)}
    if isinstance(value, dict):
        return {key: retained_json(child) for key, child in value.items()}
    if isinstance(value, list):
        return [retained_json(child) for child in value]
    return value


def gate_pass(gate) -> bool:
    return gate == 'PASS' or isinstance(gate, dict) and gate.get('status') == 'PASS'


def admission(summary: dict, quality: dict | None, idea: str) -> bool:
    if quality is None or len(summary.get('ideas', [])) != 1 or summary.get('stage') not in ('time', 'run'):
        return False
    expected = {'id': idea, 'mode': 'fast', 'vendor': 'apple', 'status': 'PASS',
                'source_sha': summary.get('source_sha'),
                'manifest_sha256': summary.get('recipe_digest'),
                'workloads_sha256': summary.get('workloads_sha256'),
                'paired_build_sha256': summary.get('paired_build_sha256'),
                'artifact_hashes': summary.get('artifact_hashes'),
                'machine': summary.get('machine')}
    return all(value and quality.get(key) == value for key, value in expected.items()) and all(
        gate_pass(quality.get('gates', {}).get(gate)) and gate_pass(summary.get('gates', {}).get(gate))
        for gate in GATES
    )


def export(summary: dict, path: Path, quality: dict | None) -> list[dict]:
    evidence_hash = digest(path)
    ideas = summary.get('ideas', [])
    if not ideas:
        raise ValueError('Result must retain its selected ideas, including failed attempts')
    machine = summary.get('machine')
    if isinstance(machine, dict):
        machine = json.dumps(machine, sort_keys=True)
    source = summary.get('source_sha')
    artifacts = summary.get('artifact_hashes', {})
    workloads = summary.get('workloads') or [{'id': 'preflight', 'arms': {}, 'status': summary.get('status')}]
    cells = []
    for item in workloads:
        arms = item.get('arms', {})
        a = arms.get('baseline', {})
        b = arms.get('candidate', {})
        ad = a.get('data', {})
        bd = b.get('data', {})
        scored = []
        for data in (ad, bd):
            rounds = data.get('rounds', [])
            scored.append(rounds[1] if len(rounds) == 2 and rounds[0].get('excluded_warmup') is True
                          and rounds[1].get('excluded_warmup') is False else {})
        at = scored[0].get('whole_operation_ms')
        bt = scored[1].get('whole_operation_ms')
        complete = a.get('exit_code') == 0 and b.get('exit_code') == 0 and positive(at) and positive(bt)
        failed = any(arm.get('exit_code', 0) != 0 for arm in arms.values()) or bool(summary.get('error'))
        for idea in ideas:
            cell = {'id': idea, 'vendor': 'apple', 'route': 'fast',
                    'case': '+'.join(ideas) + '/' + item['id'], 'scope': 'full_workload',
                    'status': 'FAILED' if failed else 'UNQUALIFIED',
                    'source_sha': source, 'machine': machine,
                    'evidence': str(path.resolve()), 'evidence_sha256': evidence_hash,
                    'attempt_id': evidence_hash + ':' + idea + ':' + item['id'],
                    'selected_ids': ideas, 'recipe': item.get('recipe'),
                    'workloads_sha256': summary.get('workloads_sha256'),
                    'paired_build_sha256': summary.get('paired_build_sha256'),
                    'gates': item.get('gates', summary.get('gates', {})),
                    'observed': {'baseline_ms': at, 'candidate_ms': bt,
                                 'baseline_exit_code': a.get('exit_code'), 'candidate_exit_code': b.get('exit_code')},
                    'note': 'Own A/B only. No opponent ratio or automatic promotion. Combined selections remain interaction evidence.'}
            if (complete and admission(summary, quality, idea) and machine and source
                    and all(artifacts.get(arm) for arm in ('baseline', 'candidate'))
                    and ad.get('source_sha') == source and bd.get('source_sha') == source
                    and ad.get('actual_arrays') == bd.get('actual_arrays')
                    and scored[0].get('params') == scored[1].get('params')
                    and all(gate_pass(item.get('gates', {}).get(gate)) for gate in GATES)):
                cell.update(status='MEASURED', artifact_hashes=artifacts,
                            baseline_machine=machine, candidate_machine=machine,
                            baseline_source_sha=source, candidate_source_sha=source,
                            warmups=1, scored_samples=1, baseline_ms=at, candidate_ms=bt, returncode=0)
            cells.append(cell)
    return cells


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, action='append', required=True)
    parser.add_argument('--quality-receipt', type=Path, help='Optional matched genuine gate receipt for a single result')
    parser.add_argument('--index', type=Path, help='Existing immutable-attempt index to extend')
    parser.add_argument('--output', type=Path, required=True, help='New index file; input history is not overwritten')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output index; existing attempt history is retained')
    if args.quality_receipt and len(args.result) != 1:
        parser.error('Attach a quality receipt to one result at a time')
    index = read(args.index) if args.index else {'schema': 1, 'campaign': 'apple-fast-classical-20261006',
                                                'machines': [], 'cells': [], 'notes': [], 'decisions': []}
    if index.get('campaign') != 'apple-fast-classical-20261006':
        parser.error('Use the AFCL campaign index; historical campaigns are separate')
    quality = read(args.quality_receipt) if args.quality_receipt else None
    existing = {cell.get('attempt_id') for cell in index['cells']}
    for path in args.result:
        for cell in export(read(path), path, quality):
            if cell['attempt_id'] not in existing:
                index['cells'].append(cell)
                existing.add(cell['attempt_id'])
    index['notes'].append('Imported retained AFCL evidence without executing workloads or updating boards. Missing gates remain UNQUALIFIED.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(retained_json(index), indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
