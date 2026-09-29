"""Compare original installed-verifier reports without synthesizing columns.

This compares the declared overlap only. Native-source qualification, physical
multi-GPU checks and exact-wheel installation receipts remain separate gates.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

PARTS = ('train', 'infer', 'model', 'batch', 'stepfull')
VENDORS = ('metal', 'cuda', 'hip')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(report, vendor, commit, lanes):
    require(report.get('format') == 'mojolearn.verify-all-report.v1', 'Wrong report format')
    require(report.get('exit') == 0 and report.get('verdict') == 'VERIFIED', 'Report did not verify')
    device = report.get('device', {})
    require(device.get('vendor') == vendor and device.get('numeric_mode') == 'identical', 'Wrong device or mode')
    require(device.get('commit') == commit and device.get('commit_source') == 'wheel COMMIT witness',
            'Report is not from the selected installed source')
    require(report.get('fixtures') == ['base'] and report.get('repeats') == 1, 'Wrong fixture scope')
    selected = report.get('lanes', [])
    require(selected and len(selected) == len(set(selected)) and set(lanes) <= set(selected), 'Missing or duplicate lanes')
    require(report.get('models_checked') == 0, 'Unexpected portable-model rows')
    execution = report.get('execution', {})
    require(execution.get('mode') == 'fresh-process-per-cell' and execution.get('interrupted') is None
            and execution.get('completed_cells') == execution.get('total_cells') == len(selected),
            'Incomplete or unisolated execution')
    require(report.get('bindings') and not report.get('bindings_error'), 'Missing native witnesses')
    for binding in report['bindings']:
        require(binding.get('module') and re.fullmatch('[0-9a-f]{64}', str(binding.get('sha256', '')))
                and binding.get('size', 0) > 0, 'Invalid native witness')
    rows = report.get('cells', [])
    indexed = {(r['lane'], r['fixture'], r['part']): r for r in rows}
    expected = {(lane, 'base', part) for lane in selected for part in PARTS}
    require(len(indexed) == len(rows) and set(indexed) == expected, 'Missing, duplicate or unexpected cell parts')
    for row in rows:
        value = row.get('value')
        require(not row.get('error'), 'Cell execution error')
        if row.get('state') == 'IDENTICAL':
            require(re.fullmatch('[0-9a-f]{16}', str(value)) and value == row.get('reference'),
                    'Numeric value does not equal its reference')
        else:
            require(row.get('state') == 'N/A' and isinstance(value, str) and value.startswith('n/a:')
                    and value not in ('n/a:UNDECLARED', 'n/a:skipped'), 'Unverified cell part')
    return indexed


def compare(reports, selection):
    require(set(reports) == set(VENDORS), 'Apple, NVIDIA and AMD reports are all required')
    lanes = selection['lanes']
    require(lanes and len(lanes) == len(set(lanes)), 'Invalid comparison selection')
    require(selection.get('fixtures') == ['base'] and selection.get('parts') == list(PARTS), 'Wrong selected parts')
    indexed = {v: validate(r, v, selection['package_source'], lanes) for v, r in reports.items()}
    for section in ('harness', 'table'):
        hashes = {r.get(section, {}).get('sha256') for r in reports.values()}
        require(len(hashes) == 1 and re.fullmatch('[0-9a-f]{64}', str(next(iter(hashes)))),
                'Different or absent ' + section + ' bytes')
    numeric = structural = 0
    for lane in lanes:
        for part in PARTS:
            key = (lane, 'base', part)
            values = {(index[key]['state'], index[key]['value']) for index in indexed.values()}
            require(len(values) == 1, 'Cross-vendor difference: ' + '/'.join(key))
            if next(iter(values))[0] == 'IDENTICAL':
                numeric += 1
            else:
                structural += 1
    return dict(schema='mojolearn.installed-identity-comparison.v1', status='AGREE',
                source_commit=selection['package_source'], lanes=lanes, fixtures=['base'],
                numeric_parts=numeric, structural_na_parts=structural,
                report_lane_counts={v: len(r['lanes']) for v, r in reports.items()},
                full_algorithm_coverage=False, release_qualified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    for vendor in VENDORS:
        parser.add_argument('--' + vendor, type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    paths = {v: getattr(args, v) for v in VENDORS}
    result = compare({v: json.loads(p.read_text()) for v, p in paths.items()},
                     json.loads(args.selection.read_text()))
    result['inputs'] = {v: dict(path=str(p.resolve()), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                        for v, p in dict(paths, selection=args.selection).items()}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
