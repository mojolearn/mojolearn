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


#: THE SPLIT LINUX PACKAGES (python/mojolearn/gpu_plugins.py): wheel-name
#: prefix of each GPU plugin -> the runtime vendor its receipt must report.
PLUGIN_VENDORS = {'mojolearn_cuda': 'cuda', 'mojolearn_rocm': 'hip'}
#: The marker the split core carries in its .dist-info (gpu_plugins.CORE_MARKER).
CORE_MARKER = 'gpu_plugins.json'


def wheel_kind(path):
    """(platform, role, vendors a receipt may report) of one manifest wheel.

    macOS: the Metal wheel. Linux: the combined wheel (the NVIDIA smoke, as
    always); the split core `mojolearn` (either vendor's smoke installed it,
    with its plugin); a split plugin (only its own vendor's smoke, whose
    receipt names the plugin by name and sha256)."""
    name = path.name
    prefix = name.split('-', 1)[0]
    if prefix in PLUGIN_VENDORS:
        return 'linux', 'plugin', {PLUGIN_VENDORS[prefix]}
    if 'macosx' in name:
        return 'macos', 'wheel', {'metal'}
    with zipfile.ZipFile(path) as archive:
        split_core = any(n.endswith('.dist-info/' + CORE_MARKER) for n in archive.namelist())
    return 'linux', 'wheel', ({'cuda', 'hip'} if split_core else {'cuda'})


def check(directory, source_commit, platform="all"):
    platforms = {"all": {"macos", "linux"}, "macos": {"macos"}, "linux": {"linux"}}[platform]
    directory = Path(directory)
    require(re.fullmatch('[0-9a-f]{40}', source_commit), 'invalid source commit')
    manifest = json.loads((directory / 'alpha-manifest.json').read_text())
    contract = manifest.get('light_smoke', {})
    require(contract.get('source_commit') == source_commit, 'smoke source differs from frozen release source')
    receipts = contract.get('receipts', {})
    wheels = manifest.get('files', {})
    require(isinstance(wheels, dict) and len(wheels) == len(platforms), 'exactly one wheel per selected platform required')
    require(isinstance(receipts, dict) and len(receipts) == len(wheels), 'required platform smoke receipts missing')
    kinds = {}
    for wheel in wheels:
        require('/' not in wheel and (directory / wheel).is_file(), 'missing wheel ' + wheel)
        kinds[wheel] = wheel_kind(directory / wheel)
    require({k[0] for k in kinds.values()} == platforms, 'exactly one wheel per selected platform required')
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
        core = Path(report.get('wheel', '')).name
        plugins = {Path(p.get('wheel', '')).name: p.get('wheel_sha256')
                   for p in report.get('plugins') or [] if isinstance(p, dict)}
        # The receipt covers the one manifest wheel it installed: its core
        # wheel, or (for a split plugin) a plugin it installed beside the core.
        covered = [w for w in wheels if (kinds[w][1] == 'wheel' and w == core)
                   or (kinds[w][1] == 'plugin' and w in plugins)]
        require(len(covered) == 1, 'unlisted or duplicate wheel')
        wheel = covered[0]
        require(wheel not in seen_wheels, 'unlisted or duplicate wheel')
        _, role, allowed = kinds[wheel]
        require(vendor in allowed and vendor not in seen_vendors, 'wrong or duplicate vendor')
        require(('macosx' in wheel) if vendor == 'metal' else ('manylinux' in wheel), 'vendor/platform mismatch')
        if role == 'plugin':
            require(plugins[wheel] == wheels[wheel] == digest(directory / wheel), 'wheel digest mismatch')
            # A plugin holds no Python and no source witness; it is bound to
            # the receipt by sha256 and to its core by the exact version pin.
            require(core.split('-')[:2] == ['mojolearn', wheel.split('-')[1]],
                    'plugin receipt installed a core of another version')
        else:
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
    require(seen_wheels == set(wheels), 'incomplete artifact smoke')
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
