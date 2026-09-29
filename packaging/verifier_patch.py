#!/usr/bin/env python3
"""Package/admit verifier-only updates of published wheels without rerunning numerics.

Every member outside the explicit verifier/version allowlist must remain byte-identical
to the PyPI base. Original native provenance is retained; patch provenance is separate.
"""
import argparse
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import urllib.request
import zipfile

VERIFIER_FILES = ('__main__.py', '_verify_all.py', '_verification_profiles.py')
PROJECTS = {'mojolearn', 'mojolearn_nvidia', 'mojolearn_amd'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def members(path):
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if not n.endswith('/')]
        if len(names) != len(set(names)):
            raise ValueError('duplicate wheel members')
        return {n: z.read(n) for n in names}


def record(files, dist):
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator='\n')
    for name, raw in sorted(files.items()):
        writer.writerow((name, 'sha256=' + base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip('='), len(raw)))
    writer.writerow((dist + '/RECORD', '', ''))
    return buf.getvalue().encode()


def expected(base, base_name, version, source_root, source_commit):
    project, old, tags = base_name.split('-', 2)
    if project not in PROJECTS or not re.fullmatch(r'\d+\.\d+\.\d+(?:rc\d+)?', version):
        raise ValueError('unsupported project/version')
    old_dist, dist = f'{project}-{old}.dist-info', f'{project}-{version}.dist-info'
    files = {n.replace(old_dist + '/', dist + '/', 1): raw for n, raw in base.items()
             if n != old_dist + '/RECORD'}
    # Version changes are confined to package metadata, not native provenance.
    for name in ('METADATA', 'gpu_plugin.json', 'gpu_plugins.json'):
        path = dist + '/' + name
        if path in files:
            files[path] = files[path].replace(old.encode(), version.encode())
    if project == 'mojolearn':
        files['mojolearn/_version.py'] = files['mojolearn/_version.py'].replace(old.encode(), version.encode())
        for name in VERIFIER_FILES:
            files['mojolearn/' + name] = (source_root / 'python/mojolearn' / name).read_bytes()
    proof = dict(schema='mojolearn.verifier-patch.v1', base_wheel=base_name,
                 source_commit=source_commit, version=version,
                 base_members_sha256={n: sha(raw) for n, raw in sorted(base.items())},
                 scope='verifier and version metadata only; numerical payload reused unchanged')
    files[dist + '/VERIFIER_PATCH.json'] = (json.dumps(proof, sort_keys=True, indent=2) + '\n').encode()
    files[dist + '/RECORD'] = record(files, dist)
    return f'{project}-{version}-{tags}', files


def published_base(name, cache):
    project, version, _ = name.split('-', 2)
    if project not in PROJECTS:
        raise ValueError('unsupported base project')
    with urllib.request.urlopen(f'https://pypi.org/pypi/{project.replace("_", "-")}/{version}/json') as response:
        metadata = json.load(response)
    item, = [x for x in metadata['urls'] if x['filename'] == name]
    path = cache / name
    if not path.exists():
        with urllib.request.urlopen(item['url']) as response:
            path.write_bytes(response.read())
    if sha(path.read_bytes()) != item['digests']['sha256']:
        raise ValueError('base wheel differs from PyPI digest: ' + name)
    return members(path), item['digests']['sha256']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--base-version', default='0.8.25')
    parser.add_argument('--version', default='0.8.26')
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--manifest-sha256')
    args = parser.parse_args()
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.source_root, text=True).strip()
    if head != args.source_commit:
        raise ValueError('source commit does not match checkout')
    for name in VERIFIER_FILES:
        path = 'python/mojolearn/' + name
        committed = subprocess.check_output(['git', 'show', head + ':' + path], cwd=args.source_root)
        if committed != (args.source_root / path).read_bytes():
            raise ValueError('uncommitted verifier source: ' + name)
    args.cache.mkdir(parents=True, exist_ok=True)
    args.directory.mkdir(parents=True, exist_ok=True)
    manifest_path = args.directory / 'alpha-manifest.json'
    if args.build:
        names = [f'mojolearn-{args.base_version}-py3-none-macosx_11_0_arm64.whl'] + [
            f'{p}-{args.base_version}-py3-none-manylinux_2_35_x86_64.whl' for p in sorted(PROJECTS)]
        manifest = dict(schema='mojolearn.verifier-patch-release.v1', version=args.version,
                        source_commit=head, files={}, bases={})
        for name in names:
            base, digest = published_base(name, args.cache)
            target, files = expected(base, name, args.version, args.source_root, head)
            with zipfile.ZipFile(args.directory / target, 'w', zipfile.ZIP_DEFLATED) as z:
                for n, raw in files.items():
                    z.writestr(n, raw)
            manifest['files'][target] = sha((args.directory / target).read_bytes())
            manifest['bases'][target] = dict(filename=name, sha256=digest)
        manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n')
    raw = manifest_path.read_bytes()
    if args.manifest_sha256 and sha(raw) != args.manifest_sha256:
        raise ValueError('manifest digest mismatch')
    manifest = json.loads(raw)
    if manifest['schema'] != 'mojolearn.verifier-patch-release.v1' or manifest['source_commit'] != head:
        raise ValueError('manifest schema/source mismatch')
    version = manifest['version']
    required = {f'mojolearn-{version}-py3-none-macosx_11_0_arm64.whl'} | {
        f'{p}-{version}-py3-none-manylinux_2_35_x86_64.whl' for p in PROJECTS}
    if set(manifest['files']) != required:
        raise ValueError('verifier patch requires both core platforms and both Linux plugins')
    if set(manifest['files']) != {p.name for p in args.directory.glob('*.whl')}:
        raise ValueError('unexpected or missing wheels')
    for name, digest in manifest['files'].items():
        if Path(name).name != name or sha((args.directory / name).read_bytes()) != digest:
            raise ValueError('wheel name/digest mismatch')
        identity = manifest['bases'][name]
        base, base_digest = published_base(identity['filename'], args.cache)
        if base_digest != identity['sha256']:
            raise ValueError('base identity mismatch')
        target, files = expected(base, identity['filename'], manifest['version'], args.source_root, head)
        actual = members(args.directory / name)
        if target != name or actual != files:
            changed = sorted(n for n in set(actual) | set(files) if actual.get(n) != files.get(n))
            raise ValueError('unapproved payload changes: ' + str(changed))
        print(json.dumps(dict(wheel=name, status='PASSED', numerical_payload='byte-identical to published base')))
    print('manifest_sha256=' + sha(raw))


if __name__ == '__main__':
    main()
