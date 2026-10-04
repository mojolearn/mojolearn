#!/usr/bin/env python3
"""Generate a bounded PTX release decision from complete retained evidence.

This is a separate admission step, not a flag added to an experimental build.
It rechecks the actual payload, full NVIDIA comparisons, pinned Apple/AMD
references, and measured driver witnesses. It never changes input artifacts or
their source identities. Shared vendor coverage and NVIDIA-only coverage remain
distinct. No record is emitted on an incomplete or failing comparison.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import compare_ptx_vendor_evidence as cross_vendor
import nvidia_baseline_qualification as baseline

ROOT = Path(__file__).resolve().parents[1]
require = baseline.require


def admission_api():
    spec = importlib.util.spec_from_file_location(
        'release_ptx_admission', ROOT / 'python/mojolearn/ptx_admission.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def encoded(document):
    return (json.dumps(document, indent=2, sort_keys=True) + '\n').encode()


def encoded_sha(document):
    return hashlib.sha256(encoded(document)).hexdigest()


def configuration_for(receipt, witness, script_sha256):
    """Join a separate measured configuration witness without editing receipts."""
    require(witness.get('schema') == 'mojolearn.cuda-runtime-config-witness.v1'
            and witness.get('source_commit') == receipt['source_commit'],
            'Configuration witness source/schema differs')
    require(witness.get('script_sha256') == script_sha256,
            'Configuration capture script differs')
    require(witness.get('source_commit_file_sha256')
            == receipt.get('installed_source', {}).get('core_commit_sha256'),
            'Configuration witness installed core differs')
    require(witness.get('timestamp_utc') and witness.get('cuDriverGetVersion_return') == 0
            and type(witness.get('cuDriverGetVersion')) is int
            and witness['cuDriverGetVersion'] > 0, 'Missing measured CUDA driver API version')
    hardware, device = receipt['hardware'], witness.get('device', {})
    require(device.get('uuid') == hardware['uuid']
            and device.get('name') == hardware['name']
            and device.get('compute_capability') == '.'.join(map(str, hardware['compute_capability']))
            and device.get('driver_version') == receipt['driver_version'],
            'Configuration witness GPU/driver differs')
    return dict(device_name=hardware['name'], compute_capability=hardware['compute_capability'],
                driver_version=receipt['driver_version'],
                cuda_driver_version=witness['cuDriverGetVersion'])


def build(manifest_path, receipts, references, reference_hashes, witness_paths, witness_script):
    api = admission_api()
    manifest_path = Path(manifest_path)
    manifest = baseline.read(manifest_path)
    source = manifest['source_commit']
    h, _ = baseline.modules()
    inventory = baseline.inventory(h)
    require(not inventory['applicability_gaps'], 'Unresolved applicability metadata')
    require(set(inventory['fixtures']) == set(api.NVIDIA_FIXTURES),
            'Inventory fixture contract changed; review admission policy')
    require(set(inventory['parts']) == set(api.NVIDIA_PARTS),
            'Inventory output-part contract changed; review admission policy')
    # This validates full lane/fixture/part coverage, two repeats, loaded-file
    # provenance, actual payload bytes, and at least two compute capabilities.
    nvidia = baseline.check(manifest_path, receipts)
    require(nvidia.get('status') == 'OBSERVED_CONFIGURATION_AGREEMENT'
            and nvidia.get('full_applicable_single_gpu_coverage') is True,
            'Full NVIDIA comparison did not pass')
    require(set(nvidia['lanes']) == set(inventory['lanes']), 'NVIDIA lane inventory differs')
    script_sha = baseline.sha(witness_script)
    witnesses = {}
    witness_inputs = []
    for path in map(Path, witness_paths):
        witness = baseline.read(path)
        uuid = witness.get('device', {}).get('uuid')
        require(uuid and uuid not in witnesses, 'Missing or duplicate witness GPU UUID')
        witnesses[uuid] = witness
        witness_inputs.append(dict(file=str(path.resolve()), sha256=baseline.sha(path)))
    shared_reports, configs, used = [], {}, set()
    for path in map(Path, receipts):
        receipt = baseline.read(path)
        if receipt['role'] != 'baseline':
            continue
        report = cross_vendor.report(manifest_path, path, references, source,
                                     baseline.harness_digest(), reference_hashes)
        require(report['passed'], 'PTX does not match the pinned Apple/AMD references')
        require({row['vendor'] for row in report['comparisons']} == {'apple', 'amd'},
                'Missing reference vendor')
        for row in report['comparisons']:
            require(set(row['lanes']) == set(inventory['lanes'])
                    and set(row['fixtures']) == set(api.SHARED_FIXTURES)
                    and set(row['parts']) == set(api.SHARED_PARTS),
                    'Cross-vendor coverage differs from the complete canonical contract')
        uuid = receipt['hardware']['uuid']
        require(uuid in witnesses, 'Missing measured CUDA configuration: ' + uuid)
        config = configuration_for(receipt, witnesses[uuid], script_sha)
        configs[json.dumps(config, sort_keys=True)] = config
        used.add(uuid)
        shared_reports.append(report)
    require(shared_reports and used == set(witnesses), 'Missing baseline or unused configuration witness')
    shared = dict(schema='mojolearn.ptx-admission-shared-evidence.v1', comparisons=shared_reports,
                  configuration_inputs=witness_inputs,
                  capture_script=dict(file=str(Path(witness_script).resolve()), sha256=script_sha))
    lanes = sorted(inventory['lanes'])
    record = dict(schema=api.SCHEMA, qualified=True, numeric_mode='identical',
                  source_commit=source, manifest_sha256=baseline.sha(manifest_path),
                  coverage_contract=api.COVERAGE_CONTRACT,
                  coverage=dict(inventory_sha256=encoded_sha(inventory),
                      harness_sha256=baseline.harness_digest(),
                      shared=dict(lanes=lanes, fixtures=list(api.SHARED_FIXTURES),
                                  parts=list(api.SHARED_PARTS), vendors=['cuda', 'hip', 'metal'],
                                  comparison_sha256=encoded_sha(shared)),
                      nvidia=dict(lanes=lanes, fixtures=list(api.NVIDIA_FIXTURES),
                                  parts=list(api.NVIDIA_PARTS), comparison_sha256=encoded_sha(nvidia))),
                  configurations=[configs[key] for key in sorted(configs)])
    api.validate_admission(record, source_commit=source, manifest_sha256=baseline.sha(manifest_path))
    return record, {'inventory.json': inventory, 'nvidia-comparison.json': nvidia,
                    'shared-comparison.json': shared}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--receipt', required=True, type=Path, action='append')
    parser.add_argument('--configuration-witness', required=True, type=Path, action='append')
    parser.add_argument('--configuration-script', required=True, type=Path)
    for vendor in ('apple', 'amd'):
        parser.add_argument('--' + vendor, required=True, type=Path)
        parser.add_argument('--' + vendor + '-sha256', required=True)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    require(not args.out.exists(), 'Refusing to overwrite retained admission evidence')
    record, reports = build(args.manifest, args.receipt,
        {'apple': args.apple, 'amd': args.amd},
        {'apple': args.apple_sha256, 'amd': args.amd_sha256},
        args.configuration_witness, args.configuration_script)
    args.out.mkdir(parents=True)
    for name, report in reports.items():
        (args.out / name).write_bytes(encoded(report))
    # Write the decision last; failed verification never produces this file.
    output = args.out / admission_api().ADMISSION_FILE
    output.write_bytes(encoded(record))
    print(json.dumps(dict(admission=str(output), source_commit=record['source_commit'],
                          configurations=len(record['configurations']),
                          shared_fixtures=record['coverage']['shared']['fixtures'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
