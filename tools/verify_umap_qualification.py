#!/usr/bin/env python3
"""Check retained UMAP qualification against the exact macOS upload candidate.

Reads files only: no imports of mojolearn, installs, builds or GPU execution.
This verifies the qualification record, not a new hardware qualification.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from qualify_umap_wheel import sha


def verify(wheel, results, source_root, expected_version):
    record = json.loads(Path(results).read_text())
    if record.get('status') != 'PASSED':
        raise ValueError('Qualification did not pass')
    if record.get('wheel_sha256') != sha(wheel):
        raise ValueError('Upload wheel differs from the qualified wheel')
    if record.get('expected_version') != expected_version:
        raise ValueError('Qualification version differs from the release version')
    required_sources = (
        'python/mojolearn/_umap_impl.py',
        'python/mojolearn/tests/test_umap_surface.py',
        'python/mojolearn/tests/test_umap_transform.py',
        'tools/umap_transform_quality_check.py',
    )
    sources = record.get('source_files', {})
    for relative in required_sources:
        if sources.get(relative) != sha(Path(source_root) / relative):
            raise ValueError('Qualification source changed: ' + relative)
    modes = {'fast': 0, 'deterministic': 2, 'identical': 1}
    expected_jobs = {'create-venv', 'install-wheel', 'check-dependencies',
                     'install-test-dependencies', 'installed-packages'}
    expected_jobs.update(f'{surface}-{mode}' for mode in modes
                         for surface in ('fit', 'transform', 'quality'))
    jobs = record.get('jobs', [])
    if (len(jobs) != len(expected_jobs) or
            {job.get('name') for job in jobs} != expected_jobs):
        raise ValueError('Qualification jobs are missing, duplicated or unexpected')
    for job in jobs:
        if job.get('exit_code') != 0 or job.get('timed_out') is not False:
            raise ValueError('Qualification job did not succeed: ' + job['name'])
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Wheel contains ambiguous duplicate members')
        wrapper_sha = hashlib.sha256(archive.read('mojolearn/_umap_impl.py')).hexdigest()
        if wrapper_sha != sources[required_sources[0]]:
            raise ValueError('Wheel wrapper differs from the qualified source')
        for mode, code in modes.items():
            member = ('mojolearn/' + (mode + '/' if mode != 'fast' else '') +
                      '_mojolearn_metrics.so')
            if mode != 'identical':
                # DEVIATION 2490: UMAP ships IDENTICAL only. The wheel must not
                # carry a lower-tier metrics binding, and each lower-tier job
                # must have been refused by name by the installed package.
                if member in names:
                    raise ValueError('Wheel carries an identical-only binding in a lower tier: ' + member)
                for surface in ('fit', 'transform', 'quality'):
                    name = f'{surface}-{mode}'
                    installed = next(job for job in jobs if job['name'] == name).get('installed', {})
                    if (installed.get('version') != expected_version or
                            installed.get('mode') != mode or
                            installed.get('refused') is not True or
                            '_mojolearn_metrics' not in installed.get('refusal', '') or
                            installed.get('wrapper_sha256') != wrapper_sha):
                        raise ValueError('Lower tier was not refused by the installed wheel: ' + name)
                continue
            binding_sha = hashlib.sha256(archive.read(member)).hexdigest()
            for surface in ('fit', 'transform', 'quality'):
                name = f'{surface}-{mode}'
                installed = next(job for job in jobs if job['name'] == name).get('installed', {})
                if (installed.get('version') != expected_version or
                        installed.get('mode') != mode or
                        installed.get('binding_mode_code') != code or
                        installed.get('wrapper_sha256') != wrapper_sha or
                        installed.get('binding_sha256') != binding_sha):
                    raise ValueError('Installed artifact evidence differs from wheel: ' + name)
    return record['wheel_sha256']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('wheel', type=Path)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--expected-version', required=True)
    args = parser.parse_args()
    try:
        digest = verify(args.wheel, args.results, args.source_root, args.expected_version)
    except (ValueError, OSError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        parser.exit(1, f'UMAP qualification verification failed: {exc}\n')
    print(f'PASS retained UMAP qualification matches upload wheel: {digest}')


if __name__ == '__main__':
    main()
