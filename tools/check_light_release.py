#!/usr/bin/env python3
"""Admit a bounded release smoke, without claiming full numerical certification."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile


JOBS = {'create-venv', 'install', 'dependencies', 'installed', 'expanded-api',
        'loaded-lm-cpu', 'loaded-lm-gpu', 'loaded-lm-cpu-gpu-compare',
        'coverage', 'models', 'batch', 'extended', 'self-test'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def check(directory, source_commit, platform="all"):
    vendors = {"all": {"metal", "cuda"}, "macos": {"metal"}, "linux": {"cuda"}}[platform]
    directory = Path(directory)
    require(re.fullmatch('[0-9a-f]{40}', source_commit), 'invalid source commit')
    manifest = json.loads((directory / 'alpha-manifest.json').read_text())
    contract = manifest.get('light_smoke', {})
    require(contract.get('source_commit') == source_commit, 'smoke source differs from frozen release source')
    receipts = contract.get('receipts', {})
    require(isinstance(receipts, dict) and len(receipts) == len(vendors), 'required platform smoke receipts missing')
    wheels = manifest.get('files', {})
    require(isinstance(wheels, dict) and len(wheels) == len(vendors), 'exactly one wheel per selected platform required')
    seen_vendors, seen_wheels = set(), set()
    for name, expected in receipts.items():
        require(re.fullmatch(r'light-smoke-[a-z0-9-]+\.json', name), 'unsafe smoke filename')
        path = directory / name
        require(path.is_file() and not path.is_symlink() and path.stat().st_size < 2 * 1024**2,
                'missing or oversized smoke receipt')
        require(digest(path) == expected, 'smoke receipt digest mismatch')
        report = json.loads(path.read_text())
        require(report.get('status') == 'PASSED' and report.get('scope') == 'expanded',
                'installed smoke did not pass')
        require(report.get('source_commit') == source_commit, 'receipt source mismatch')
        require(report.get('release_qualified') is False, 'smoke cannot claim full release qualification')
        vendor = report.get('installed', {}).get('vendor')
        require(vendor in vendors and vendor not in seen_vendors, 'wrong or duplicate vendor')
        wheel = Path(report.get('wheel', '')).name
        require(wheel in wheels and wheel not in seen_wheels, 'unlisted or duplicate wheel')
        require(('macosx' in wheel) if vendor == 'metal' else ('manylinux' in wheel), 'vendor/platform mismatch')
        require(report.get('wheel_sha256') == wheels[wheel] == digest(directory / wheel), 'wheel digest mismatch')
        with zipfile.ZipFile(directory / wheel) as archive:
            require(archive.read('mojolearn/identity_columns/COMMIT').decode().strip() == source_commit,
                    'packaged source mismatch')
        jobs = report.get('jobs', [])
        require(len(jobs) == len({j['name'] for j in jobs}), 'duplicate smoke job')
        require(JOBS <= {j['name'] for j in jobs}, 'missing required smoke job')
        require(all(j.get('exit_code') in ((0, 5) if j['name'] == 'extended' else (0,)) for j in jobs),
                'failed smoke job')
        require(report.get('expanded', {}).get('scope') == 'expanded', 'missing expanded API/loaded-model proof')
        seen_vendors.add(vendor)
        seen_wheels.add(wheel)
    require(seen_vendors == vendors and seen_wheels == set(wheels), 'incomplete artifact smoke')
    return {'status': 'PASSED_LIGHT_RELEASE', 'source_commit': source_commit,
            'wheels': wheels, 'runtime_vendors': sorted(seen_vendors),
            'full_numerical_certification': False,
            'scope': 'Exact installed smoke for the published platforms; other architectures and experimental contracts are not newly certified'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--platform', choices=('all', 'macos', 'linux'), default='all')
    args = parser.parse_args()
    print(json.dumps(check(args.directory, args.source_commit, args.platform), indent=2))


if __name__ == '__main__':
    main()
