#!/usr/bin/env python3
"""File-only alpha publication staging gate; no numerical qualification claim."""
import argparse
import ast
import csv
from email import policy
from email.parser import BytesParser
import hashlib
import io
import json
from pathlib import Path
import re
import stat
import zipfile

from alpha_overlay import (
    MAX_FILES, MAX_MEMBER, MAX_TOTAL, MAX_WHEEL, NOTICE, hash_stream, record_hash,
    require, safe_name,
)


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key: ' + key)
        result[key] = value
    return result


def decode(raw):
    return json.loads(raw, object_pairs_hook=unique)


def hex_digest(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def wheel_digest(path):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= MAX_WHEEL,
            'wheel must be a bounded regular file')
    digest, size = hashlib.sha256(), 0
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            size += len(chunk)
            require(size <= MAX_WHEEL, 'wheel grew past bound')
            digest.update(chunk)
    return digest.hexdigest()


def verify_wheel(path, version, release_profile=None):
    parts = path.name[:-4].split('-')
    require(path.name.endswith('.whl') and len(parts) in (5, 6)
            and parts[0] == 'mojolearn' and parts[1] == version
            and parts[-1] != 'any' and all(re.fullmatch('[A-Za-z0-9_.]+', p) for p in parts[-3:]),
            'wheel filename version/platform mismatch')
    dist = 'mojolearn-' + version + '.dist-info/'
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        require(len(infos) <= MAX_FILES and sum(i.file_size for i in infos) <= MAX_TOTAL,
                'wheel expansion exceeds bounds')
        names = set()
        for info in infos:
            safe_name(info.filename.rstrip('/') if info.is_dir() else info.filename)
            require(info.filename not in names and not info.flag_bits & 1
                    and not stat.S_ISLNK(info.external_attr >> 16)
                    and info.file_size <= MAX_MEMBER, 'unsafe/duplicate/encrypted/oversize member')
            names.add(info.filename)
        files = {i.filename for i in infos if not i.is_dir()}
        require({n.split('/')[0] for n in files if '.dist-info/' in n} == {dist[:-1]},
                'unexpected dist-info directory')

        def small(name):
            require(name in files and archive.getinfo(name).file_size <= 16 * 1024**2,
                    'missing/oversize metadata: ' + name)
            return archive.read(name)

        record_name = dist + 'RECORD'
        records = {}
        for row in csv.reader(io.StringIO(small(record_name).decode('utf-8'))):
            require(len(row) == 3 and row[0] not in records, 'invalid/duplicate RECORD row')
            safe_name(row[0])
            records[row[0]] = row[1:]
        require(set(records) == files and records[record_name] == ['', ''], 'RECORD inventory mismatch')
        hashes = {}
        for name in sorted(files - {record_name}):
            with archive.open(name) as stream:
                digest, size = hash_stream(stream)
            require(records[name] == [record_hash(digest), str(size)], 'RECORD hash mismatch: ' + name)
            hashes[name] = digest.hex()
        wheel = BytesParser(policy=policy.compat32).parsebytes(small(dist + 'WHEEL'))
        tags = {p + '-' + a + '-' + platform for p in parts[-3].split('.')
                for a in parts[-2].split('.') for platform in parts[-1].split('.')}
        require(wheel.get_all('Root-Is-Purelib') == ['false']
                and set(wheel.get_all('Tag', [])) == tags, 'platform WHEEL tags mismatch')
        metadata = BytesParser(policy=policy.compat32).parsebytes(small(dist + 'METADATA'))
        require(metadata.get_all('Name') == ['mojolearn'] and metadata.get_all('Version') == [version]
                and [v for v in metadata.get_all('Classifier', []) if v.startswith('Development Status ::')]
                    == ['Development Status :: 3 - Alpha'], 'alpha package metadata mismatch')
        provenance = decode(small(dist + 'ALPHA_PROVENANCE.json'))
        require(provenance.get('schema') == 'mojolearn.alpha-overlay.v1'
                and provenance.get('version') == version
                and hex_digest(provenance.get('base_wheel_sha256'))
                and provenance.get('current_numerical_qualification') ==
                    'NOT INHERITED; requires separate root validation', 'alpha provenance contract mismatch')
        if release_profile == 'alpha-api':
            require(provenance.get('release_profile') == 'alpha-api', 'wheel lacks manifest alpha-api release profile')
        native = {n: h for n, h in hashes.items() if n.endswith(('.so', '.dylib', '.dll', '.pyd')) or '.so.' in n}
        require(native and provenance.get('native_runtime_sha256') == native,
                'native/runtime provenance mismatch')
        modules = None
        for node in ast.parse(small('mojolearn/_backend.py')).body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_MODULES' for t in node.targets):
                modules = ast.literal_eval(node.value)
        require(isinstance(modules, tuple) and modules and all(isinstance(n, str) for n in modules)
                and provenance.get('required_module_registry') == list(modules), 'native registry provenance mismatch')
        directories = sorted({str(Path(n).parent) for n in native if Path(n).name.startswith('_mojolearn')})
        missing = {d: [module for module in modules if not any(str(Path(n).parent) == d and
                    (Path(n).name == module + '.so' or Path(n).name.startswith(module + '.'))
                    for n in native)] for d in directories}
        require(provenance.get('missing_optional_native_modules_by_present_directory') == missing,
                'missing-native availability inventory mismatch')
        require(provenance.get('notice') == NOTICE and small(dist + 'ALPHA_NOTICE.md') == (NOTICE + '\n').encode()
                and NOTICE.encode() in small(dist + 'METADATA'), 'alpha qualification notice missing')
        python = {n: h for n, h in hashes.items() if n.endswith('.py') and
                  (n.startswith('mojolearn/') or n == 'mojolearn_diagnostics.py')}
        require(python and provenance.get('overlay_python_sha256') == python,
                'overlaid Python provenance mismatch')
        docs = provenance.get('documentation_source_sha256')
        require(isinstance(docs, dict) and set(docs) <= {'mojolearn/ALPHA_API.md'}
                and all(hashes.get(n) == h and hex_digest(h) for n, h in docs.items()),
                'overlaid documentation provenance mismatch')
        require(('mojolearn/ALPHA_API.md' not in hashes) or 'mojolearn/ALPHA_API.md' in docs,
                'alpha API documentation is not provenance-bound')
        require(small('mojolearn/_version.py') == ('__version__ = ' + repr(version) + '\n').encode(),
                'runtime Python version differs from alpha metadata')
        require(not any(n.startswith('mojolearn/') and any(p in ('tests', '__pycache__') for p in Path(n).parts)
                        for n in files), 'test/cache artifacts must not ship')
    return sorted(tags)


def verify(directory, manifest_sha256):
    require(hex_digest(manifest_sha256), 'expected manifest SHA256 must be lowercase hexadecimal')
    require(directory.is_dir() and not directory.is_symlink(), 'expected regular artifact directory')
    manifest_path = directory / 'alpha-manifest.json'
    require(manifest_path.is_file() and not manifest_path.is_symlink()
            and manifest_path.stat().st_size <= 2 * 1024**2, 'missing/oversize manifest')
    with manifest_path.open('rb') as stream:
        raw = stream.read(2 * 1024**2 + 1)
    require(len(raw) <= 2 * 1024**2 and hashlib.sha256(raw).hexdigest() == manifest_sha256,
            'manifest SHA256 mismatch')
    manifest = decode(raw)
    require(isinstance(manifest, dict) and manifest.get('schema') == 'mojolearn.alpha-release.v1',
            'alpha manifest schema mismatch')
    version, files = manifest.get('version'), manifest.get('files')
    require(isinstance(version, str), 'manifest version must be a string')
    alpha_version = re.fullmatch(r'(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)a(0|[1-9][0-9]*)', version)
    final_version = re.fullmatch(r'(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)', version)
    release_profile = manifest.get('release_profile')
    require(alpha_version or (final_version and release_profile == 'alpha-api'),
            'manifest requires explicit alpha version or alpha-api release profile')
    require(isinstance(files, dict) and 1 <= len(files) <= 32, 'manifest wheel inventory missing/oversize')
    for name, digest in files.items():
        safe_name(name)
        require('/' not in name and name.endswith('.whl') and hex_digest(digest), 'invalid manifest wheel entry')
    require({p.name for p in directory.iterdir()} == set(files) | {'alpha-manifest.json'},
            'artifact directory has missing or injected files')
    tags = {}
    for name, expected in sorted(files.items()):
        path = directory / name
        require(wheel_digest(path) == expected, 'wheel SHA256 mismatch: ' + name)
        tags[name] = verify_wheel(path, version, release_profile)
    return dict(schema='mojolearn.alpha-artifact-verification.v1', passed=True, version=version,
                manifest_sha256=manifest_sha256, files=files, tags=tags, release_profile=release_profile,
                scope='File integrity, alpha metadata and recorded overlay provenance only; '
                      'no package execution, native compatibility or current numerical qualification')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--manifest-sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.directory, args.manifest_sha256), sort_keys=True, indent=2))
