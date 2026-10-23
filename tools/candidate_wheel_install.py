#!/usr/bin/env python3
"""Install unpublished release artifacts with explicit, non-PyPI provenance.

The orchestrator supplies every mojolearn distribution required by this vendor.
Only third-party dependencies may resolve from an index. Qualification receipts
identify the exact wheel hashes and source freeze, never merely a version.
"""
import argparse
import base64
import csv
from email.parser import BytesParser
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import zipfile

import pypi_wheel_install as published

SCHEMA = 'mojolearn-candidate-wheel-manifest/1'


def canonical(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def wheel_record(path, version, source_commit):
    path = Path(path).resolve()
    if not path.is_file() or path.suffix != '.whl':
        raise ValueError('candidate wheel absent: '+str(path))
    if not re.fullmatch(r'[0-9a-f]{40}', source_commit):
        raise ValueError('candidate source commit must be a full Git SHA')
    with zipfile.ZipFile(path) as wheel:
        names = wheel.namelist()
        if len(names) != len(set(names)) or any(n.startswith('/') or '..' in Path(n).parts for n in names):
            raise ValueError('duplicate or unsafe wheel members: '+path.name)
        metadata = [n for n in names if n.endswith('.dist-info/METADATA')]
        if len(metadata) != 1:
            raise ValueError('candidate must contain exactly one distribution: '+path.name)
        prefix = metadata[0].rsplit('/', 1)[0]
        meta = BytesParser().parsebytes(wheel.read(metadata[0]))
        name = canonical(meta['Name'] or '')
        if meta['Version'] != version or not name:
            raise ValueError('candidate metadata version/name differs: '+path.name)
        filename_parts = path.name[:-4].split('-')
        if (len(filename_parts) not in (5, 6) or canonical(filename_parts[0]) != name
                or filename_parts[1] != version):
            raise ValueError('candidate filename differs from metadata: '+path.name)
        tags = BytesParser().parsebytes(wheel.read(prefix+'/WHEEL')).get_all('Tag', [])
        py, abi, platform = filename_parts[-3:]
        filename_tags = {f'{p}-{a}-{s}' for p in py.split('.') for a in abi.split('.') for s in platform.split('.')}
        if not tags or set(tags) != filename_tags:
            raise ValueError('wheel filename/ABI/platform tags differ: '+path.name)
        payload = json.loads(wheel.read(prefix+'/LINUX_PAYLOAD.json'))
        if (payload.get('schema') != 'mojolearn.linux-payload.v1' or payload.get('version') != version
                or payload.get('source_commit') != source_commit
                or canonical((payload.get('split') or {}).get('distribution', '')) != name):
            raise ValueError('candidate payload/source provenance differs: '+path.name)
        modes = payload.get('release_modes', ['fast', 'identical'])
        if (not isinstance(modes, list) or len(modes) != len(set(modes))
                or not set(modes) <= {'fast', 'identical'} or 'identical' not in modes):
            raise ValueError('candidate release mode metadata is invalid or lacks identical')
        modes = [mode for mode in ('fast', 'identical') if mode in modes]
        if name == published.CORE:
            if wheel.read('mojolearn/identity_columns/COMMIT').decode().strip() != source_commit:
                raise ValueError('core identity table belongs to another source freeze')
        record_name = prefix+'/RECORD'
        rows = list(csv.reader(io.StringIO(wheel.read(record_name).decode())))
        indexed = set()
        for row in rows:
            if len(row) != 3 or row[0] in indexed:
                raise ValueError('invalid or duplicate wheel RECORD row')
            member, digest, size = row
            indexed.add(member)
            if member == record_name:
                if digest or size:
                    raise ValueError('RECORD must not hash itself')
                continue
            data = wheel.read(member)
            actual = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()
            if digest != 'sha256='+actual or size != str(len(data)):
                raise ValueError('candidate RECORD mismatch: '+member)
        if indexed != {n for n in names if not n.endswith('/')}:
            raise ValueError('unrecorded candidate wheel payload')
        return dict(distribution=name, version=version, file=path.name, path=str(path),
                    sha256=published.sha256_file(path), bytes=path.stat().st_size, tags=sorted(tags),
                    source_commit=source_commit, requires_dist=meta.get_all('Requires-Dist', []),
                    release_modes=modes,
                    payload_sha256=hashlib.sha256(wheel.read(prefix+'/LINUX_PAYLOAD.json')).hexdigest())


def build_manifest(version, vendor, wheels, source_commit):
    if vendor not in ('nvidia', 'amd'):
        raise ValueError('candidate qualification supports NVIDIA and AMD')
    files = [wheel_record(path, version, source_commit) for path in wheels]
    expected = set(published.distributions(vendor))
    if len(files) != len(expected) or {row['distribution'] for row in files} != expected:
        raise ValueError('supply exactly the core and all vendor artifacts: '+', '.join(sorted(expected)))
    if len({tuple(row['release_modes']) for row in files}) != 1:
        raise ValueError('candidate core/vendor release modes differ; missing metadata means historical dual mode')
    for row in files:
        for requirement in row['requires_dist']:
            match = re.match(r'\s*([A-Za-z0-9_.-]+)\s*(.*)', requirement)
            if not match:
                raise ValueError('unreadable wheel dependency: '+requirement)
            dependency = canonical(match[1])
            if dependency.startswith('mojolearn'):
                # An absent same-version package must never be obtained from
                # PyPI while supposedly testing an unpublished candidate.
                spec = match[2].split(';', 1)[0].strip().strip('()').replace(' ', '')
                if dependency not in expected or spec != '=='+version:
                    raise ValueError('unprovided or unpinned mojolearn dependency: '+requirement)
    files.sort(key=lambda row: row['distribution'])
    return dict(schema=SCHEMA, source='candidate', version=version, vendor=vendor,
                source_commit=source_commit, files=files,
                release_modes=files[0]['release_modes'],
                requested=[name+'=='+version for name in sorted(expected)],
                validation='wheel RECORD, metadata, tags and frozen payload provenance')


def identity(manifest):
    return dict(source='candidate', vendor=manifest['vendor'], version=manifest['version'],
                source_commit=manifest['source_commit'],
                release_modes=manifest['release_modes'],
                wheels={row['distribution']: row['sha256'] for row in manifest['files']})


def refusal(manifest, loaded):
    if manifest.get('schema') != SCHEMA or manifest.get('source') != 'candidate':
        return 'not a candidate artifact manifest'
    try:
        files = manifest['files']
        if (not re.fullmatch(r'[0-9a-f]{40}', manifest['source_commit'])
                or len(files) != len(published.distributions(manifest['vendor']))
                or {row['distribution'] for row in files} != set(published.distributions(manifest['vendor']))
                or any(row['source_commit'] != manifest['source_commit'] or row['version'] != manifest['version']
                       or row['release_modes'] != manifest['release_modes']
                       or not re.fullmatch(r'[0-9a-f]{64}', row['sha256']) for row in files)):
            return 'candidate identity/version/source records differ'
        why = published.installed_refusal(manifest, loaded)
        if why:
            return why
        for row in files:
            archive = (loaded[row['distribution']].get('direct_url') or {}).get('archive_info') or {}
            digest = archive.get('hashes', {}).get('sha256') or archive.get('hash', '').removeprefix('sha256=')
            if digest != row['sha256']:
                return 'candidate installation lacks its exact artifact hash: '+row['distribution']
    except (KeyError, TypeError, ValueError):
        return 'incomplete candidate artifact manifest'
    return None


def install(args):
    manifest = build_manifest(args.version, args.vendor, args.wheel, args.source_commit)
    supported = subprocess.run([args.python, '-c',
        'import json; from pip._vendor.packaging.tags import sys_tags; print(json.dumps([str(t) for t in sys_tags()]))'],
        capture_output=True, text=True, check=True)
    tags = set(json.loads(supported.stdout))
    if any(not tags.intersection(row['tags']) for row in manifest['files']):
        raise ValueError('candidate wheel ABI/platform is incompatible with the qualification interpreter')
    paths = [row['path'] for row in manifest['files']]
    pip = [args.python, '-m', 'pip', 'install', '--no-input', '--disable-pip-version-check']
    if published.run(pip+paths, args.log) or published.run(pip+['--force-reinstall', '--no-deps']+paths, args.log):
        raise RuntimeError('candidate wheel installation failed; see '+args.log)
    # Retain the actual bytes identity again after pip opened the files.
    if any(published.sha256_file(row['path']) != row['sha256'] for row in manifest['files']):
        raise RuntimeError('candidate artifact changed during installation')
    manifest['installed_into'] = args.python
    manifest['loaded'] = published.loaded_receipt(args.python, manifest)
    published._write(args.manifest, manifest)
    why = refusal(manifest, manifest['loaded'])
    if why:
        raise RuntimeError(why)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('version', 'vendor', 'python', 'source-commit', 'manifest', 'log'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--wheel', action='append', required=True)
    return install(parser.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
