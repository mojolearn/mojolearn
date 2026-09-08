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
import sys
import tarfile
import tempfile
import zipfile

from alpha_overlay import (
    MAX_FILES, MAX_MEMBER, MAX_TOTAL, MAX_WHEEL, NOTICE, hash_stream, record_hash,
    require, safe_name,
)

# DEVIATION 2290. The version the combined Linux profile ships and the profile's
# name come from the tools-side reader, never from a literal here:
# python/mojolearn/_version.py is the one source of truth (0.7.0 at the time of
# writing; 0.6.1 was never published) and `release-linux3` the profile, with
# `release-0.6.1` accepted as its deprecated alias.
REPO = Path(__file__).resolve().parents[1]
TOOLS = str(REPO / 'tools')
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)
from verify_linux_surface_qualification import RELEASE_PROFILE, RELEASE_PROFILES, release_version  # noqa: E402


def released_version(source_root=None):
    """The version SOURCE_ROOT's _version.py declares; this checkout's when no root is given."""
    return release_version(REPO if source_root is None else Path(source_root))


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


def verify_wheel(path, version, release_profile=None, qualification_root=None, source_root=None):
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
        require(all('\n' not in value and '\r' not in value
                    for value in metadata.get_all('Summary', [])),
                'package Summary must be a single line')
        released = released_version(source_root)  # DEVIATION 2290: never a literal
        if version == released and ('linux' in parts[-1]):
            require(dist + 'LINUX_PAYLOAD.json' in files,
                    'Linux ' + released + ' requires a fresh combined payload, not an inherited native overlay')
        if dist + 'LINUX_PAYLOAD.json' in files:
            require(version == released and release_profile == 'alpha-api'
                    and parts[-1].startswith('manylinux_')
                    and dist + 'ALPHA_PROVENANCE.json' not in files,
                    'fresh combined Linux payload cannot masquerade as inherited alpha overlay')
            payload = decode(small(dist + 'LINUX_PAYLOAD.json'))
            require(payload.get('schema') == 'mojolearn.linux-payload.v1'
                    and payload.get('release_profile') == 'alpha-api'
                    and payload.get('assembly_profile') in RELEASE_PROFILES
                    and payload.get('version') == version, 'fresh Linux payload profile mismatch')
            require(qualification_root is not None and source_root is not None,
                    'fresh Linux wheel requires full final-wheel installed qualification and source root')
            # Trusted repository file-only checker; never import package/native code.
            # (tools/ is on sys.path from the module top, DEVIATION 2290.)
            from check_linux_release_qualification import check_release061
            result = check_release061(path, qualification_root, source_root)
            require(result.get('status') == 'PASSED'
                    and result.get('wheel_sha256') == wheel_digest(path)
                    and set(result.get('runtime_coverage', {})) == {'cuda/sm_89', 'cuda/sm_90', 'hip/gfx942'},
                    'exact final combined wheel runtime qualification missing')
            return sorted(tags)
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


def extract_qualification(path, expected_sha, output):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 512 * 1024**2,
            'qualification archive missing/unsafe/over 512 MiB')
    require(wheel_digest(path) == expected_sha, 'qualification archive SHA mismatch')
    with tarfile.open(path, 'r:gz') as archive:
        members, names, total = [], set(), 0
        for member in archive:
            require(len(members) < 20000, 'too many qualification members')
            name = member.name
            if name.startswith('./'):
                name = name[2:]
            if name in ('', '.') and member.isdir():
                continue
            safe_name(name)
            require(name not in names and (member.isdir() or member.isfile())
                    and member.size <= 128 * 1024**2, 'unsafe/duplicate qualification member')
            total += member.size
            require(total <= 2 * 1024**3, 'qualification expansion exceeds two GiB')
            names.add(name)
            members.append((member, name))
        for member, name in members:
            target = output / name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as incoming, target.open('xb') as outgoing:
                    while True:
                        chunk = incoming.read(1024 * 1024)
                        if not chunk:
                            break
                        outgoing.write(chunk)


def verify(directory, manifest_sha256, qualification_archive=None, source_root=None):
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
    qualification = manifest.get('linux_qualification')
    if qualification is not None:
        require(version == released_version(source_root) and release_profile == 'alpha-api'  # DEVIATION 2290
                and isinstance(qualification, dict)
                and set(qualification) == {'file', 'sha256', 'wheel'}
                and qualification['file'] == 'linux-qualification.tar.gz'
                and qualification['wheel'] in files
                and hex_digest(qualification['sha256']), 'invalid Linux qualification manifest')
        require(qualification_archive is not None and source_root is not None
                and qualification_archive.name == qualification['file'],
                'root-pinned Linux qualification archive/source required')
    else:
        require(qualification_archive is None, 'unlisted qualification archive refused')
    tags = {}
    for name, expected in sorted(files.items()):
        path = directory / name
        require(wheel_digest(path) == expected, 'wheel SHA256 mismatch: ' + name)
        if qualification is not None and name == qualification['wheel']:
            with tempfile.TemporaryDirectory(prefix='mojolearn-alpha-linux-qualification-') as temporary:
                extracted = Path(temporary)
                extract_qualification(qualification_archive, qualification['sha256'], extracted)
                with zipfile.ZipFile(path) as archive:
                    require('mojolearn-' + version + '.dist-info/LINUX_PAYLOAD.json' in archive.namelist(),
                            'qualified wheel must carry a fresh Linux payload inventory')  # DEVIATION 2290
                tags[name] = verify_wheel(path, version, release_profile, extracted, source_root)
        else:
            # DEVIATION 2290: the source root only decides which _version.py is read.
            tags[name] = verify_wheel(path, version, release_profile, None, source_root)
    return dict(schema='mojolearn.alpha-artifact-verification.v1', passed=True, version=version,
                manifest_sha256=manifest_sha256, files=files, tags=tags, release_profile=release_profile,
                linux_qualification=qualification,
                scope='File integrity, alpha metadata and recorded overlay provenance only; '
                      'no package execution. A listed fresh Linux payload additionally requires its exact final-wheel '
                      'three-architecture installed admission; overlay artifacts inherit no current numerical qualification')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--qualification-archive', type=Path)
    parser.add_argument('--source-root', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.directory, args.manifest_sha256, args.qualification_archive, args.source_root), sort_keys=True, indent=2))
