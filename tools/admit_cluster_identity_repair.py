"""Account for a completed broad capture plus a bounded cluster repair.

Original reports are never rewritten into a fictitious passing broad run.
Native bytes and every other shipped Python module must remain unchanged.
This is numerical evidence admission, not permission to publish a release.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

from compare_installed_identity import PARTS, require, validate


def sha(data):
    return hashlib.sha256(data).hexdigest()


def payloads(wheels):
    require(len(wheels) == 3 and {p.name.split('-')[0] for p in wheels}
            == {'mojolearn', 'mojolearn_nvidia', 'mojolearn_amd'}, 'Need core and both plugins')
    result = {}
    for path in wheels:
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if name.endswith(('.py', '.dylib')) or '.so' in Path(name).name:
                    require(name not in result, 'Duplicate packaged member: ' + name)
                    result[name] = sha(archive.read(name))
    return result


def admit(initial, repair, table, lanes, commit):
    require(initial.get('format') == 'mojolearn.verify-all-report.v1', 'Wrong initial report')
    require(initial.get('fixtures') == ['base'] and initial.get('repeats') == 1, 'Wrong initial fixture scope')
    require(len(initial.get('lanes', [])) == len(lanes) and set(initial['lanes']) == set(lanes), 'Incomplete initial lane scope')
    execution = initial.get('execution', {})
    require(execution.get('mode') == 'fresh-process-per-cell' and execution.get('interrupted') is None
            and execution.get('completed_cells') == execution.get('total_cells') == len(lanes),
            'Initial sweep did not finish')
    require(initial.get('device', {}).get('vendor') == 'cuda'
            and initial['device'].get('numeric_mode') == 'identical', 'Wrong initial backend')
    require(initial['device'].get('commit_source') == 'wheel COMMIT witness'
            and initial.get('models_checked') == 0 and initial.get('bindings')
            and not initial.get('bindings_error'), 'Missing initial installed/native witness')
    affected = sorted(lane for lane in lanes if lane.startswith('x-cluster-'))
    require(affected and set(repair.get('lanes', [])) == set(affected), 'Repair must cover the whole affected cluster family')
    replacement = validate(repair, 'cuda', commit, affected)
    expected = {(lane, 'base', part) for lane in lanes for part in PARTS}
    rows = initial.get('cells', [])
    original = {(r['lane'], r['fixture'], r['part']): r for r in rows}
    require(len(original) == len(rows) and set(original) == expected, 'Missing or duplicate initial parts')
    failures = set()
    unreferenced_structural = []
    numeric = structural = 0
    for key, old in original.items():
        lane, fixture, part = key
        new = replacement.get(key, old)
        require(new.get('state') in ('IDENTICAL', 'N/A') and not new.get('error'), 'Unresolved cell: ' + '/'.join(key))
        reference = table['cells'].get(lane + '/' + fixture, {}).get(part)
        if reference is None:
            require(new.get('state') == 'N/A' and part == 'stepfull'
                    and new.get('value') == 'n/a:no-decode-state',
                    'Missing applicable final reference: ' + '/'.join(key))
            unreferenced_structural.append('/'.join(key))
        else:
            require(new.get('value') == reference['ref'], 'Value differs from final reference: ' + '/'.join(key))
        if old.get('state') not in ('IDENTICAL', 'N/A') or old.get('error'):
            failures.add(lane)
            require(lane in affected, 'Failure outside repaired family')
        elif lane in affected:
            require(old['value'] == new['value'] and old['state'] == new['state'],
                    'Previously passing cluster part changed: ' + '/'.join(key))
        if new['state'] == 'IDENTICAL':
            require(re.fullmatch('[0-9a-f]{16}', str(new['value'])), 'Invalid numeric hash')
            numeric += 1
        else:
            require(isinstance(new['value'], str) and new['value'].startswith('n/a:')
                    and new['value'] not in ('n/a:UNDECLARED', 'n/a:skipped'), 'Undeclared structural part')
            structural += 1
    return dict(schema='mojolearn.cluster-identity-repair-admission.v1', status='PASS',
                source_commit=commit, initial_source_commit=initial['device'].get('commit'),
                lanes=len(lanes), carried_lanes=len(lanes) - len(affected),
                repaired_lanes=affected, originally_failing_lanes=sorted(failures),
                numeric_parts=numeric, structural_na_parts=structural,
                unreferenced_structural_parts=unreferenced_structural,
                synthesized_raw_column=False, release_qualified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('initial', 'repair', 'selection', 'output'):
        parser.add_argument('--' + field, type=Path, required=True)
    parser.add_argument('--initial-wheel', type=Path, action='append', required=True)
    parser.add_argument('--final-wheel', type=Path, action='append', required=True)
    args = parser.parse_args()
    initial = json.loads(args.initial.read_text())
    repair = json.loads(args.repair.read_text())
    selection = json.loads(args.selection.read_text())
    before, after = payloads(args.initial_wheel), payloads(args.final_wheel)
    require(set(before) == set(after), 'Packaged Python/native inventory changed')
    changed = {name for name in before if before[name] != after[name]}
    require(changed == {'mojolearn/_expansion_cluster.py'}, 'Unexpected shipped code/native changes: ' + str(changed))
    core = [p for p in args.final_wheel if p.name.startswith('mojolearn-')]
    require(len(core) == 1, 'Need one final core wheel')
    with zipfile.ZipFile(core[0]) as archive:
        commit = archive.read('mojolearn/identity_columns/COMMIT').decode().strip()
        table_bytes = archive.read('mojolearn/verify_reference/table.json')
        harness_sha = sha(archive.read('mojolearn/_identity_break.py'))
    require(repair['table']['sha256'] == sha(table_bytes), 'Repair did not use final table')
    require(initial['harness']['sha256'] == repair['harness']['sha256'] == harness_sha, 'Harness changed')
    old_core = [p for p in args.initial_wheel if p.name.startswith('mojolearn-')]
    require(len(old_core) == 1, 'Need one initial core wheel')
    with zipfile.ZipFile(old_core[0]) as archive:
        require(initial['device']['commit'] == archive.read('mojolearn/identity_columns/COMMIT').decode().strip(),
                'Initial source witness differs')
    result = admit(initial, repair, json.loads(table_bytes), selection['lanes'], commit)
    result['inputs'] = [dict(path=str(p.resolve()), sha256=sha(p.read_bytes())) for p in
                       [args.initial, args.repair, args.selection, *args.initial_wheel, *args.final_wheel]]
    result['unchanged_native_files'] = sum(name.endswith(('.so', '.dylib')) for name in before)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
