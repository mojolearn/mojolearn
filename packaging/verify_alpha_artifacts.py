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
from verify_linux_surface_qualification import (  # noqa: E402
    RELEASE_PROFILE, RELEASE_PROFILES, load_gpu_plugins, release_version)

# THE SPLIT LINUX PACKAGES (python/mojolearn/gpu_plugins.py, 2026-09-25): the
# core `mojolearn` (no GPU set, `[cuda]`/`[rocm]` extras pinning the plugins)
# and one plugin per vendor, `mojolearn-cuda` and `mojolearn-rocm` (only
# mojolearn/<vendor>/..., requiring exactly `mojolearn==<version>`). Their
# payload records the split packer profile.
GPU_PLUGINS = load_gpu_plugins()
SPLIT_PROFILE = 'release-split'
#: wheel-name prefix -> (distribution, vendor directory or None for the core)
DISTRIBUTIONS = {'mojolearn': ('mojolearn', None),
                 **{row['wheel_name']: (row['distribution'], vendor) for vendor, row in GPU_PLUGINS.PLUGINS.items()}}


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


def verify_wheel(path, version, release_profile=None, qualification_root=None, source_root=None,
                 macos_smoke_source=None):
    parts = path.name[:-4].split('-')
    require(path.name.endswith('.whl') and len(parts) in (5, 6)
            and parts[0] in DISTRIBUTIONS and parts[1] == version
            and parts[-1] != 'any' and all(re.fullmatch('[A-Za-z0-9_.]+', p) for p in parts[-3:]),
            'wheel filename version/platform mismatch')
    distribution, plugin_vendor = DISTRIBUTIONS[parts[0]]
    dist = parts[0] + '-' + version + '.dist-info/'
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
        require(metadata.get_all('Name') == [distribution] and metadata.get_all('Version') == [version]
                and [v for v in metadata.get_all('Classifier', []) if v.startswith('Development Status ::')]
                    == ['Development Status :: 3 - Alpha'], 'alpha package metadata mismatch')
        require(all('\n' not in value and '\r' not in value
                    for value in metadata.get_all('Summary', [])),
                'package Summary must be a single line')
        released = released_version(source_root)  # DEVIATION 2290: never a literal
        split_core = plugin_vendor is None and dist + GPU_PLUGINS.CORE_MARKER in files
        if plugin_vendor is not None or split_core:
            verify_split_wheel(path, version, released, release_profile, qualification_root, plugin_vendor,
                               distribution, dist, files, metadata, small)
            return sorted(tags)
        overlay_present = dist + 'ALPHA_PROVENANCE.json' in files
        if version == released and ('linux' in parts[-1]) and not overlay_present:
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
            if qualification_root is None:
                # Alpha policy, 2026-09-09: the installed qualification archive is
                # optional. Without it the wheel's payload, RECORD, metadata and
                # hashes are still checked; runtime behavior is not certified.
                return sorted(tags)
            require(source_root is not None, 'qualification archive needs the source root')
            # Trusted repository file-only checker; never import package/native code.
            # (tools/ is on sys.path from the module top, DEVIATION 2290.)
            from check_linux_release_qualification import check_release061
            # DEVIATION 2293: one owner for the Hopper slot's two spellings.
            import verify_linux_surface_qualification as surface
            result = check_release061(path, qualification_root, source_root)
            require(result.get('status') == 'PASSED'
                    and result.get('wheel_sha256') == wheel_digest(path)
                    # DEVIATION 2293: Hopper spelled sm_90 or sm_90a, never both.
                    and surface.arch_set_ok(set(result.get('runtime_coverage', {}))),
                    'exact final combined wheel runtime qualification missing')
            return sorted(tags)
        if (parts[-1].startswith('macosx_') and parts[-1].endswith('_arm64') and not overlay_present
                and dist + 'LINUX_PAYLOAD.json' not in files):
            # A FRESH NATIVE macOS BUILD (0.8.12). Until then a macOS wheel could
            # reach the light route only as an overlay inheriting native bytes,
            # so a release with native changes had no light path on macOS. It is
            # admitted only as the released version, only with a macOS smoke
            # receipt in the manifest, and only when its source witness names the
            # commit that receipt was taken from (tools/check_light_release.py
            # then ties the receipt to this exact wheel's SHA256).
            require(version == released and release_profile == 'alpha-api',
                    'a fresh macOS build is admitted only as the released alpha-api version')
            require(macos_smoke_source is not None,
                    'a fresh macOS build needs its light-smoke-macos.json receipt in the manifest')
            require(small('mojolearn/identity_columns/COMMIT').decode().strip() == macos_smoke_source,
                    'fresh macOS build source witness differs from its smoke receipt source')
            require(dist + 'portable-math.json' in files, 'fresh macOS build lacks its platform-math audit record')
            require(any(n.endswith(('.so', '.dylib')) and n.startswith('mojolearn/identical/') for n in files),
                    'fresh macOS build carries no identical-mode native binaries')
            declared = [ast.literal_eval(node.value) for node in ast.parse(small('mojolearn/_version.py')).body
                        if isinstance(node, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == '__version__' for t in node.targets)]
            require(declared == [version], 'runtime Python version differs from wheel metadata')
            require(not any(n.startswith('mojolearn/') and any(p in ('tests', '__pycache__') for p in Path(n).parts)
                            for n in files), 'test/cache artifacts must not ship')
            return sorted(tags)
        provenance = decode(small(dist + 'ALPHA_PROVENANCE.json'))
        require(provenance.get('schema') == 'mojolearn.alpha-overlay.v1'
                and provenance.get('version') == version
                and hex_digest(provenance.get('base_wheel_sha256'))
                and provenance.get('current_numerical_qualification') ==
                    'NOT INHERITED; requires separate root validation', 'alpha provenance contract mismatch')
        reuse = provenance.get('native_reuse')
        if reuse is not None:
            require(isinstance(reuse, dict) and reuse.get('schema') == 'mojolearn.native-reuse.v1'
                    and all(re.fullmatch('[0-9a-f]{40}', str(reuse.get(k, '')))
                            for k in ('package_source_commit', 'native_source_commit'))
                    and hex_digest(reuse.get('compile_inputs_sha256'))
                    and isinstance(reuse.get('compile_input_count'), int)
                    and reuse['compile_input_count'] > 0, 'invalid native reuse contract')
            require(small('mojolearn/identity_columns/COMMIT').decode().strip()
                    == reuse['package_source_commit'], 'native reuse package source mismatch')
            resources = provenance.get('resource_overlay_sha256', {})
            require(set(resources) == {'mojolearn/verify_reference/table.json',
                                       'mojolearn/identity_columns/COMMIT'}
                    and all(hex_digest(h) and hashes.get(n) == h for n, h in resources.items()),
                    'resource overlay provenance mismatch')
        if version == released and 'linux' in parts[-1]:
            require(release_profile == 'alpha-api' and reuse is not None
                    and parts[-1].startswith('manylinux_'), 'Linux overlay requires explicit native reuse provenance')
            parent = decode(small(dist + 'BASE_LINUX_PAYLOAD.json'))
            require(hashes[dist + 'BASE_LINUX_PAYLOAD.json'] == provenance.get('base_linux_payload_sha256')
                    and parent.get('schema') == 'mojolearn.linux-payload.v1'
                    and parent.get('source_commit') == reuse['native_source_commit'],
                    'native reuse parent Linux build provenance mismatch')
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
        require(isinstance(docs, dict) and set(docs) <= {'mojolearn/ALPHA_API.md', 'mojolearn/CITATION.cff',
                    'mojolearn/Hendel_2026_bitwise_identical_gpu_ml_preprint.pdf'}
                and all(hashes.get(n) == h and hex_digest(h) for n, h in docs.items()),
                'overlaid documentation provenance mismatch')
        require(('mojolearn/ALPHA_API.md' not in hashes) or 'mojolearn/ALPHA_API.md' in docs,
                'alpha API documentation is not provenance-bound')
        require(small('mojolearn/_version.py') == ('__version__ = ' + repr(version) + '\n').encode(),
                'runtime Python version differs from alpha metadata')
        require(not any(n.startswith('mojolearn/') and any(p in ('tests', '__pycache__') for p in Path(n).parts)
                        for n in files), 'test/cache artifacts must not ship')
    return sorted(tags)


def verify_split_wheel(path, version, released, release_profile, qualification_root, vendor,
                       distribution, dist, files, metadata, small):
    """One wheel of the split Linux set, file checks only: a fresh payload of
    the split packer profile, the exact version pins both ways, the marker
    documents gpu_plugins.py writes, and each wheel's ownership (the core no
    GPU set, a plugin only its own vendor's directory and no Python). The
    set-level partition proof is tools/wheel_api_audit.py --split, run by the
    packer; installed qualification of a split set is admitted on the whole
    set (tools/check_linux_release_qualification.py --profile release-split),
    never on one wheel of it, so a qualification archive bound to one split
    wheel is refused here."""
    role = GPU_PLUGINS.CORE_PROFILE if vendor is None else GPU_PLUGINS.PLUGINS[vendor]['profile']
    require(version == released and release_profile == 'alpha-api' and path.name[:-4].split('-')[-1].startswith('manylinux_')
            and path.name.endswith('_x86_64.whl'),
            'a split Linux wheel is admitted only as the released alpha-api version with a manylinux x86_64 tag')
    require(qualification_root is None,
            'installed qualification of a split set is admitted on the whole set, not on one wheel of it')
    require(dist + 'ALPHA_PROVENANCE.json' not in files and dist + 'LINUX_PAYLOAD.json' in files,
            'a split Linux wheel needs a fresh payload inventory, not an inherited overlay')
    payload = decode(small(dist + 'LINUX_PAYLOAD.json'))
    split = payload.get('split') if isinstance(payload, dict) else None
    require(payload.get('schema') == 'mojolearn.linux-payload.v1' and payload.get('version') == version
            and payload.get('release_profile') == 'alpha-api' and payload.get('assembly_profile') == SPLIT_PROFILE
            and isinstance(split, dict) and split.get('role') == role and split.get('distribution') == distribution,
            'split Linux payload profile/role mismatch')
    requires = metadata.get_all('Requires-Dist', [])
    payload_members = [n for n in files if not n.startswith(dist)]
    if vendor is None:
        stray = [n for n in payload_members if GPU_PLUGINS.member_vendor(n)]
        require(not stray, 'the split core carries a GPU set member: ' + (stray[0] if stray else ''))
        require(set(metadata.get_all('Provides-Extra', [])) == {r['extra'] for r in GPU_PLUGINS.PLUGINS.values()}
                and all(f'{r["distribution"]}=={version}; extra == "{r["extra"]}"' in requires
                        for r in GPU_PLUGINS.PLUGINS.values()),
                'the split core does not pin every plugin at exactly its version')
        require(decode(small(dist + GPU_PLUGINS.CORE_MARKER)) == GPU_PLUGINS.core_marker(version),
                'split core marker disagrees with gpu_plugins.py')
        require(small('mojolearn/identity_columns/COMMIT').decode().strip() == payload.get('source_commit'),
                'split core source witness differs from its payload')
        return
    stray = [n for n in payload_members if GPU_PLUGINS.member_vendor(n) != vendor]
    require(not stray, distribution + ' carries a member outside mojolearn/' + vendor + '/: '
            + (stray[0] if stray else ''))
    require(any(n.endswith('.so') for n in payload_members) and not any(n.endswith('.py') for n in payload_members),
            distribution + ' must carry binaries and no Python')
    require(requires == ['mojolearn==' + version] and not metadata.get_all('Provides-Extra', []),
            distribution + ' must require exactly mojolearn==' + version)
    arches = {n.split('/')[2] for n in payload_members}
    require(decode(small(dist + GPU_PLUGINS.PLUGIN_MARKER)) == GPU_PLUGINS.plugin_marker(vendor, version, arches),
            distribution + ' marker disagrees with its payload or gpu_plugins.py')


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
    smoke = manifest.get('light_smoke')
    smoke_files = {}
    if smoke is not None:
        require(isinstance(smoke, dict) and set(smoke) == {'source_commit', 'receipts'}
                and re.fullmatch('[0-9a-f]{40}', str(smoke['source_commit']))
                and isinstance(smoke['receipts'], dict) and 1 <= len(smoke['receipts']) <= 2,
                'invalid light smoke contract')
        smoke_files = smoke['receipts']
        for name, expected in smoke_files.items():
            require(re.fullmatch(r'light-smoke-[a-z0-9-]+\.json', name) and hex_digest(expected),
                    'invalid smoke receipt entry')
            path = directory / name
            require(path.is_file() and not path.is_symlink() and path.stat().st_size < 2 * 1024**2
                    and hashlib.sha256(path.read_bytes()).hexdigest() == expected,
                    'missing or changed smoke receipt')
    require({p.name for p in directory.iterdir()} == set(files) | set(smoke_files) | {'alpha-manifest.json'},
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
            macos_source = (smoke['source_commit'] if smoke is not None
                            and 'light-smoke-macos.json' in smoke_files else None)
            tags[name] = verify_wheel(path, version, release_profile, None, source_root, macos_source)
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
