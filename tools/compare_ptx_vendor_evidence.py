#!/usr/bin/env python3
"""Compare retained measured scope only; never admit columns or qualify a release.

Raw columns stay unchanged. Apple/AMD's intentionally omitted batch/rlpair and
PTX's additional fixtures remain explicit exclusions, not synthetic witnesses.
Receipts are trusted collector evidence, not cryptographic attestations.
"""
import argparse
import json
from pathlib import Path
import re

import identity_columns
import nvidia_baseline_qualification as baseline

PARTS = ('train', 'infer', 'model', 'batchgrad', 'batchscale', 'ragged', 'stepfull')
FIXTURES = ('base', 'denormal', 'odd')
require = baseline.require


def stable(values, repeats, label):
    require(isinstance(values, list) and len(values) == repeats and values,
            'Missing/repeat-count mismatch: ' + label)
    require(all(value == values[0] for value in values), 'Unstable: ' + label)
    return values[0]


def compare_columns(ptx, reference, source, vendor, *, left_vendor="nvidia", left_minimum=2):
    """Pure measured-scope comparison. Invalid evidence raises; mismatches report."""
    v = identity_columns.vref()
    for label, column, minimum in [(left_vendor, ptx, left_minimum), (vendor, reference, 1)]:
        require(column.get('commit') == source and column.get('mode') == 'identical'
                and column.get('complete') is True and not column.get('skipped'),
                label + ': incomplete/wrong source or mode')
        require(type(column.get('repeats')) is int and column['repeats'] >= minimum,
                label + ': insufficient repeats')
        require(not any(value for key, value in column.items() if key.endswith('_sabotage')),
                label + ': sabotage evidence')
        require(set(PARTS[3:]) <= set(column.get('parts_collected', [])),
                label + ': missing part declaration')
        require(not column.get('package', {}).get('bindings_error'), label + ': binding error')
        bindings = column.get('package', {}).get('bindings', [])
        require(bindings and all(row.get('module') and re.fullmatch('[0-9a-f]{64}', str(row.get('sha256', '')))
                                 for row in bindings)
                and len({row['module'] for row in bindings}) == len(bindings), label + ': invalid binding evidence')
    require(v.record_device_class(ptx, 'column.json') == (left_vendor, None), 'Wrong left evidence vendor')
    require(v.record_device_class(reference, 'column.json') == (vendor, None), 'Wrong reference vendor')
    require(set(reference.get('fixtures', {})) == set(FIXTURES), 'Reference fixture scope differs')
    for fixture in FIXTURES:
        for field in ('fixtures', 'heldout'):
            left = ptx.get(field, {}).get(fixture)
            require(isinstance(left, dict) and set(left) == ({'X', 'y_clf', 'y_reg'} if field == 'fixtures' else {'X'})
                    and all(re.fullmatch('[0-9a-f]{16}', str(value)) for value in left.values())
                    and left == reference.get(field, {}).get(fixture),
                    'Different/missing ' + field + ': ' + fixture)
    require(ptx.get('heldout_seed') is not None and ptx['heldout_seed'] == reference.get('heldout_seed'),
            'Different/missing heldout seed')
    for part in PARTS[3:]:
        require(ptx.get(part + '_protocol') and ptx[part + '_protocol'] == reference.get(part + '_protocol'),
                'Different/missing protocol: ' + part)
    lanes = sorted({key.split('/')[0] for key in reference['cells']})
    expected = {lane + '/' + fixture for lane in lanes for fixture in FIXTURES}
    require(expected and set(reference['cells']) == expected, 'Reference coverage has holes')
    ptx_shared = {key for key in ptx['cells'] if key.split('/')[-1] in FIXTURES}
    require(ptx_shared == expected, 'Shared lane coverage differs')
    for revisions in ('lane_revisions', 'batch_revisions'):
        require(isinstance(ptx.get(revisions), dict) and isinstance(reference.get(revisions), dict),
                'Missing revision metadata')
        for lane in lanes:
            require(ptx[revisions].get(lane) == reference[revisions].get(lane),
                    'Different revision: ' + lane + '/' + revisions)
    rows, mismatches = [], []
    for key in sorted(expected):
        cells = (ptx['cells'][key], reference['cells'][key])
        for cell in cells:
            require(not any(value for name, value in cell.items() if 'error' in name), 'Cell error: ' + key)
        for part in PARTS:
            values = []
            for column, cell in zip((ptx, reference), cells):
                value = v._part_value(cell, part, min_repeats=column['repeats'])
                require(value is not None, 'Missing/refused part: ' + key + '/' + part)
                stable(cell['hashes' if part == 'train' else part], column['repeats'], key + '/' + part)
                values.append(value)
            is_na = any(value.startswith('n/a:') for value in values)
            if is_na:
                require(values[0] == values[1], 'Structural N/A differs: ' + key + '/' + part)
                require(not any(word in values[0].lower() for word in ('skipped', 'undeclared', 'unavailable', 'refused', 'error')),
                        'Nonstructural N/A: ' + key + '/' + part)
            row = dict(cell=key, part=part, ptx=values[0], reference=values[1],
                       status='structural-n/a' if is_na else ('equal' if values[0] == values[1] else 'mismatch'))
            rows.append(row)
            if row['status'] == 'mismatch':
                mismatches.append(row)
        # Bind the constituent output hashes as well as aggregate train hashes.
        for field in ('parts', 'reload'):
            left, right = cells[0].get(field), cells[1].get(field)
            if field == 'reload' and left is None and right is None:
                require(cells[0]['model'][0].startswith('n/a:') and cells[1]['model'][0].startswith('n/a:'),
                        'Missing reload for saved model: ' + key)
                continue
            a = stable(left, ptx['repeats'], key + '/' + field)
            b = stable(right, reference['repeats'], key + '/' + field)
            require(a and b, 'Empty component evidence: ' + key + '/' + field)
            if field == 'parts':
                require(isinstance(a, dict) and isinstance(b, dict) and set(a) == set(b),
                        'Component coverage differs: ' + key)
                pairs = [(field + '/' + name, a[name], b[name]) for name in sorted(a)]
            else:
                pairs = [(field, a, b)]
                require(a == cells[0]['infer'][0] and b == cells[1]['infer'][0],
                        'Reload differs from inference: ' + key)
            for name, x, y in pairs:
                require(all(isinstance(value, str) and re.fullmatch('[0-9a-f]{16}', value) for value in (x, y)),
                        'Missing concrete component hash: ' + key + '/' + name)
                row = dict(cell=key, part=name, ptx=x, reference=y, status='equal' if x == y else 'mismatch')
                rows.append(row)
                if x != y:
                    mismatches.append(row)
    return dict(vendor=vendor, lanes=lanes, fixtures=list(FIXTURES), parts=list(PARTS),
                reference_repeats=reference['repeats'], ptx_repeats=ptx['repeats'],
                compared_hashes=sum(row['status'] != 'structural-n/a' for row in rows),
                structural_na=sum(row['status'] == 'structural-n/a' for row in rows),
                rows=rows, mismatches=mismatches, passed=not mismatches,
                excluded=dict(ptx_fixtures=sorted(set(ptx['fixtures']) - set(FIXTURES)),
                              parts=['batch', 'rlpair'], reason='No matching reference measurements'),
                hardware_evidence={key: reference.get(key) for key in ('platform', 'vendor', 'package', 'merged_from')})


def report(manifest_path, receipt_path, references, source, harness_sha256, reference_hashes):
    require(re.fullmatch('[0-9a-f]{40}', source), 'Expected full source SHA')
    require(re.fullmatch('[0-9a-f]{64}', harness_sha256), 'Expected harness SHA256')
    require(set(references) == {'apple', 'amd'} and set(reference_hashes) == set(references),
            'Both pinned reference vendors required')
    manifest_path, receipt_path = Path(manifest_path), Path(receipt_path)
    manifest, receipt = baseline.read(manifest_path), baseline.read(receipt_path)
    require(manifest.get('source_commit') == source and receipt.get('role') == 'baseline', 'Wrong source or role')
    files = baseline.manifest_files(manifest)
    for name, entry in files.items():
        require(baseline.sha(manifest_path.parent / name) == entry['sha256'], 'Payload bytes changed: ' + name)
    member = Path(receipt.get('column_file', ''))
    require(member.name == str(member) and member.name, 'Unsafe column path')
    column_path = receipt_path.parent / member
    require(baseline.sha(column_path) == receipt.get('column_sha256'), 'Column hash changed')
    column = baseline.read(column_path)
    require(set(column.get('cells', {})) == {lane + '/' + fixture for lane in receipt.get('lanes', [])
                                          for fixture in receipt.get('fixtures', [])}, 'Receipt scope differs')
    require(set(column.get('fixtures', {})) == set(receipt.get('fixtures', [])), 'Receipt fixtures differ')
    baseline.validate_receipt(receipt, column, manifest, baseline.sha(manifest_path), files, harness_sha256)
    comparisons = []
    inputs = [dict(file=str(p.resolve()), sha256=baseline.sha(p)) for p in (manifest_path, receipt_path, column_path)]
    for vendor, path in references.items():
        path = Path(path)
        require(baseline.sha(path) == reference_hashes[vendor], 'Reference file hash changed: ' + vendor)
        comparisons.append(compare_columns(column, baseline.read(path), source, vendor))
        inputs.append(dict(file=str(path.resolve()), sha256=baseline.sha(path)))
    return dict(schema='mojolearn.ptx-cross-vendor-measurements.v1', source_commit=source,
                expected_harness_sha256=harness_sha256, inputs=inputs, comparisons=comparisons,
                tooling_sha256=baseline.sha(__file__),
                ptx_hardware=receipt['hardware'], ptx_driver=receipt['driver_version'],
                passed=all(item['passed'] for item in comparisons), release_qualified=False,
                identical_qualified=False, universal_gpu_support=False,
                scope='Only retained shared fixture/part hashes; no admission or unmeasured-device claim')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'receipt', 'apple', 'amd', 'source', 'harness-sha256', 'apple-sha256', 'amd-sha256', 'out'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    result = report(args.manifest, args.receipt, {'apple': args.apple, 'amd': args.amd}, args.source, args.harness_sha256,
                    {'apple': args.apple_sha256, 'amd': args.amd_sha256})
    baseline.write(args.out, result)
    print(json.dumps({key: result[key] for key in ('passed', 'release_qualified', 'identical_qualified')}))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
