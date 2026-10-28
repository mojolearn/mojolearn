#!/usr/bin/env python3
"""Deduplicate a public-runtime inventory into a source-review worklist.

This never imports mojolearn or qualifies runtime behavior. Phase tags are
review hints, not defects: native callees and installed replacements still need
review even when a Python wrapper is one call. Existing exceptions stay explicit.
"""
import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path


def call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return call_name(node.value) + '.' + node.attr
    return '<dynamic>'


def phase_hints(node, native_entries):
    if node is None:
        return ['source_resolution_pending'], []
    calls = [n for n in ast.walk(node) if isinstance(n, ast.Call)]
    names = sorted({call_name(n.func) for n in calls})
    hints = set()
    if any(isinstance(n, (ast.If, ast.Raise, ast.Assert, ast.Compare)) for n in ast.walk(node)):
        hints.add('validation_or_branch')
    text = ast.unparse(node)
    if any(s in text for s in ('not fitted', 'not been fitted', 'call fit', 'call forward first', '_check_fitted', '_require_fitted')):
        hints.add('fitted_state')
    if any(name in ('int', 'float', 'bool', 'str', 'bytes', 'list', 'tuple', 'memoryview')
           or any(term in name for term in ('asarray', 'ascontiguous', 'reshape', 'astype', '_f32', '_buffer')) for name in names):
        hints.add('argument_or_result_preparation')
    if any(name.endswith(('._bind', '._binding', '._extension', '.binding', '.load_set')) for name in names):
        hints.add('binding_selection')
    if any(name.startswith('self.') for name in names):
        hints.add('instance_method_dispatch')
    if any(isinstance(n, (ast.For, ast.While, ast.With, ast.ListComp, ast.DictComp, ast.GeneratorExp)) for n in ast.walk(node)):
        hints.add('loop_or_resource_orchestration')
    native_lines = [n.lineno for n in calls if call_name(n.func).split('.')[-1] in native_entries]
    if native_lines and any(isinstance(n, (ast.Assign, ast.AugAssign, ast.For, ast.If)) and n.lineno > min(native_lines)
                            for n in ast.walk(node)):
        hints.add('post_native_result_work')
    # Every native owner needs transitive review; zero visible Python work is
    # deliberately not represented as an approved or completed runtime status.
    hints.add('transitive_native_review_pending' if native_entries else 'callee_resolution_pending')
    return sorted(hints), names


def group_rows(endpoints):
    groups = {}
    for row in endpoints:
        definition = row.get('definition') or row.get('declared_definition')
        connections = row.get('import_connections', [])
        # A source body shared by aliases/inheritance is one implementation.
        # Generated replacements with different connection records remain
        # separate. No-source endpoints are never coalesced merely by status.
        key = json.dumps([definition, row.get('source_ast_sha256'), row.get('native_entries', []), connections,
                          row['endpoint'] if not definition else None], sort_keys=True)
        group = groups.setdefault(key, {'id': hashlib.sha256(key.encode()).hexdigest()[:16],
            'source': definition or '', 'source_ast_sha256': row.get('source_ast_sha256') or '',
            'endpoints': [], 'public_aliases': set(), 'families': set(), 'statuses': set(), 'native_entries': set(),
            'qualification_debt': set(), 'import_connections': connections})
        group['endpoints'].append(row['endpoint'])
        public = row.get('public', row['endpoint'])
        suffix = row['endpoint'][len(public):] if row['endpoint'].startswith(public) else ''
        group['public_aliases'].update(alias + suffix for alias in row.get('aliases', []))
        group['families'].add(row['family'])
        group['statuses'].add(row['runtime_status'])
        group['native_entries'].update(row.get('native_entries', []))
        group['qualification_debt'].update(row.get('qualification_debt', []))
    return sorted(groups.values(), key=lambda g: (g['source'], g['id']))


def worklist(report, root):
    parsed = {}
    rows = []
    for group in group_rows(report['endpoints']):
        node = None
        path, _, line = group['source'].rpartition(':')
        if line.isdigit() and (root / path).is_file():
            if path not in parsed:
                parsed[path] = {n.lineno: n for n in ast.walk(ast.parse((root / path).read_text()))
                                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
            node = parsed[path].get(int(line))
        stale = node is not None and group['source_ast_sha256'] and hashlib.sha256(ast.dump(node).encode()).hexdigest() != group['source_ast_sha256']
        hints, calls = phase_hints(None if stale else node, group['native_entries'])
        if stale:
            hints.append('inventory_source_changed_rerun_required')
        rows.append(dict(group, phase_hints=hints, direct_call_names=calls,
                         runtime_evidence='NOT_REVIEWED_BY_WORKLIST'))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--summary-json', type=Path, required=True)
    parser.add_argument('--source-ref', required=True, help='Commit of the source used to create the inventory')
    args = parser.parse_args()
    raw = args.inventory.read_bytes()
    report = json.loads(raw)
    rows = worklist(report, args.root)
    columns = ('id', 'source', 'statuses', 'families', 'phase_hints', 'native_entries',
               'qualification_debt', 'runtime_evidence', 'direct_call_names', 'public_aliases', 'endpoints')
    with args.out.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter='\t', lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: '|'.join(sorted(row[k])) if isinstance(row[k], (set, list)) else row[k] for k in columns})
    summary = {'schema': 'mojolearn.provider-boundary-worklist.v1',
        'input_sha256': hashlib.sha256(raw).hexdigest(), 'input_path': str(args.inventory),
        'source_ref': args.source_ref,
        'endpoints': len(report['endpoints']), 'source_implementation_groups': len(rows),
        'groups_by_status_combination': dict(Counter('|'.join(sorted(r['statuses'])) for r in rows)),
        'groups_by_phase_hint': dict(Counter(h for r in rows for h in r['phase_hints'])),
        'limitations': ['Source identity deduplicates aliases/inherited bodies, not algorithms or data-driven options.',
            'Phase tags are heuristic triage hints and never qualification or proof of a defect.',
            'Native-boundary wrappers still need transitive native review and installed runtime evidence.',
            'Installed replacements may make retained source bodies unreachable; retain inventory qualification debt.',
            'Tag metadata is an allowed category. Other approved exceptions require exact named operation evidence.'],
        'runtime_executed': False}
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(f"worklist: {summary['endpoints']} endpoints -> {len(rows)} source implementation groups; no runtime qualification")


if __name__ == '__main__':
    main()
