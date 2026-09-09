#!/usr/bin/env python3
"""Admit one final dual-vendor Linux wheel using retained hardware evidence.

Staging convention: WHEEL_DIR/qualification/{hip,cuda}/ contains each complete
24-job qualification directory and its original build-provenance.json. The
proof is bound by wheel-audit.json's build_provenance_sha256, even when copied
into this directory after qualification. Both runs must install the exact
final staged wheel; separately qualified vendor candidates are insufficient.
Read-only: no imports of mojolearn, builds, GPU access or publication.
"""
import argparse
import base64
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import zipfile

import compare_ordered_python
import verify_linux_surface_qualification as surface

require = surface.require
EXTENSION = re.compile(r'mojolearn/(cuda|hip)/(sm_[0-9]+a?|gfx[0-9a-f]+)/'
                       r'(?:(deterministic|identical)/)?(_mojolearn[^/]*)\.so')
MODE_READBACK = surface.BINDINGS - {'_mojolearn_estimators', '_mojolearn_rf',
                                 '_mojolearn_trees', '_mojolearn_solver', '_mojolearn_tsa'}
# The three SLOTS a release wheel must fill. DEVIATION 2293: this stays the
# canonical triple, because it is what says WHICH slots exist. Which spelling
# the Hopper slot actually carries is a property of the built wheel, read off
# the wheel, never assumed from here -- widening this constant instead made a
# test fixture build four architectures and fill Hopper twice.
RELEASE_ARCHES = frozenset(surface.RELEASE_ARCHES)
RELEASE_ARCHES_ACCEPTED = RELEASE_ARCHES | {'cuda/sm_90a'}
# DEVIATION 2290. No version literal on this path: the version is whatever the
# source root's python/mojolearn/_version.py says (surface.release_version) and
# the profile is surface.RELEASE_PROFILE, with `release-0.6.1` as its alias.


def digest_file(path):
    with Path(path).open('rb') as stream:
        return digest_stream(stream)[0]


def digest_stream(stream):
    digest, size = hashlib.sha256(), 0
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def native_inventory(root):
    """Exactly the build snapshot policy in linux_surface_qualification.sh.

    Compare the complete inventory, not just listed paths: adding a new native
    file must invalidate older builds too. Qualification-only tools/workflow
    edits are outside this native-build inventory by design.
    """
    root = Path(root)
    files = []
    for directory, dirs, names in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in (
            '.git', '.pixi', '.venv', '__pycache__', 'results', 'dist',
            'archive', 'upstream') and not d.startswith('.'))
        for name in sorted(names):
            path = Path(directory) / name
            rel = path.relative_to(root).as_posix()
            if (name.endswith('.mojo') or rel.startswith((
                    'bindings/', 'packaging/linux/', 'python/mojolearn/'))
                    and name.endswith(('.py', '.sh')) or rel in (
                    'pixi.toml', 'pixi.lock', 'tools/linux_surface_qualification.sh')):
                files.append([rel, digest_file(path)])
    return sorted(files)


def inventory_digest(inventory):
    return hashlib.sha256(json.dumps(inventory, separators=(',', ':')).encode()).hexdigest()


def inspect_wheel(wheel, root, flat_python=False, byte_lm=False):
    """Verify RECORD and every advertised architecture/mode on final bytes."""
    extensions, sets = {}, {}
    require(re.fullmatch(r'.+-manylinux_[A-Za-z0-9_.]+_x86_64\.whl', wheel.name),
            'Final Linux wheel requires a repaired manylinux x86_64 tag')
    with zipfile.ZipFile(wheel) as archive:
        # DEVIATION 2295: zip DIRECTORY entries are not RECORD rows. The wheel
        # spec lists files; RECORD has never named a directory. pack_wheel.py
        # writes none, so this never mattered, but `auditwheel repair` rewrites
        # the archive and adds 15 of them, and this check then refused the very
        # wheel it exists to admit. Found by running the installed qualification
        # on a real device for the first time.
        paths = [n for n in archive.namelist() if not n.endswith('/')]
        require(len(paths) == len(set(paths)), 'Duplicate wheel member')
        require(all(not PurePosixPath(p).is_absolute() and '..' not in PurePosixPath(p).parts
                    and '\\' not in p for p in paths), 'Invalid wheel member path')
        records = [p for p in paths if p.endswith('.dist-info/RECORD')]
        require(len(records) == 1, 'Wheel requires one RECORD')
        declared = set()
        for row in csv.reader(io.StringIO(archive.read(records[0]).decode())):
            require(len(row) == 3, 'Invalid wheel RECORD row')
            path, hashed, size = row
            require(path not in declared, 'Duplicate RECORD row')
            declared.add(path)
            if path == records[0]:
                require(not hashed and not size, 'RECORD self row must be unhashed')
                continue
            with archive.open(path) as stream:
                digest, actual_size = digest_stream(stream)
            encoded = base64.urlsafe_b64encode(bytes.fromhex(digest)).rstrip(b'=').decode()
            require(hashed == 'sha256=' + encoded and size == str(actual_size),
                    'Wheel RECORD hash/size mismatch: ' + path)
            match = EXTENSION.fullmatch(path)
            if match:
                vendor, arch, mode, name = match.groups()
                extensions[path.removeprefix('mojolearn/')] = digest
                sets.setdefault((vendor, arch, mode or 'fast'), set()).add(name)
            elif path.endswith('.so') and '/_mojolearn' in path:
                raise ValueError('Unexpected wheel extension location: ' + path)
        require(declared == set(paths), 'RECORD does not cover the complete wheel')
        require({v for v, _, _ in sets} == {'hip', 'cuda'},
                'Final release wheel must contain both HIP and CUDA')
        for vendor, arch in {(v, a) for v, a, _ in sets}:
            for mode in surface.MODES:
                require(sets.get((vendor, arch, mode)) == surface.expected_bindings(mode, byte_lm),
                        'Incomplete final wheel set: ' + '/'.join((vendor, arch, mode)))
        python_sources = ((root / 'python/mojolearn').glob('*.py') if flat_python
                          else (root / 'python/mojolearn').rglob('*.py'))
        for source in python_sources:
            member = 'mojolearn/' + source.relative_to(root / 'python/mojolearn').as_posix()
            require(archive.read(member) == source.read_bytes(), 'Stale packaged Python source: ' + member)
    return extensions, {'/'.join(k): len(v) for k, v in sorted(sets.items())}


def check_vendor(directory, vendor, wheel_sha, inventory, extensions, sets, arch=None):
    qualification, _ = surface.retained(directory)
    require(qualification.get('vendor') == vendor, 'Wrong staged qualification vendor')
    require(qualification.get('wheel_sha256') == wheel_sha,
            vendor + ' qualification did not install the exact final staged wheel')
    audit = json.loads((directory / 'wheel-audit.json').read_text())
    require(audit.get('sha256') == wheel_sha and audit.get('qualification_vendor') == vendor,
            'Wheel audit differs from staged artifact/vendor')
    require(audit.get('advertised_vendors') == ['cuda', 'hip']
            and audit.get('extension_hashes') == extensions and audit.get('sets') == sets,
            'Qualification audited a different wheel layout or extension inventory')
    proof_path = directory / 'build-provenance.json'
    require(digest_file(proof_path) == audit.get('build_provenance_sha256'),
            'Build proof differs from qualification audit')
    proof = json.loads(proof_path.read_text())
    require(proof.get('schema') == 'mojolearn.linux.build-provenance.v1'
            and proof.get('complete') is True and type(proof.get('build_exit')) is int
            and proof['build_exit'] == 0 and proof.get('action') == 'build',
            'Incomplete or failed vendor build proof')
    require(re.fullmatch(r'[0-9a-f]{40}', proof.get('source_commit', '')) is not None,
            'Invalid native source commit')
    require(bool(inventory) and proof.get('source_inventory') == inventory,
            'Native build inventory differs from current source checkout')
    source_sha = inventory_digest(inventory)
    require(proof.get('source_sha256') == source_sha
            and audit.get('source_sha256') == source_sha
            and qualification.get('source_sha256') == source_sha,
            'Native source inventory/hash disagreement')
    expected_extensions = {'mojolearn/' + p: h for p, h in extensions.items()
                           if p.startswith(vendor + '/' + (arch + '/' if arch else ''))}
    require(proof.get('extensions') == expected_extensions,
            'Final wheel vendor binaries differ from qualified build proof')
    expected_records = {s + '-' + m for s, m in surface.expected_jobs(audit)}
    require(set(qualification.get('installed_records', {})) == expected_records,
            'Final admission requires the exact profile job inventory')
    for surface_name, mode in sorted(surface.expected_jobs(audit)):
        for code in (surface.MODES[mode],):
            name = surface_name + '-' + mode
            path = directory / (name + '.installed.json')
            require(qualification['installed_records'][name] == digest_file(path),
                    'Installed record differs from qualification manifest')
            installed = json.loads(path.read_text())
            if arch is not None:
                require(surface.is_release_profile(audit)  # DEVIATION 2290
                        and audit.get('runtime_architecture') == arch,
                        'Missing architecture-specific wheel audit')
                require(installed.get('device_architecture') == arch
                        and installed.get('selected_architecture') == arch
                        and installed.get('architecture_probe')
                        and installed.get('architecture_override_absent') is True,
                        'Job did not execute on the claimed native architecture')
            require(installed.get('vendor') == vendor and installed.get('mode') == mode
                    and installed.get('wheel_sha256') == wheel_sha,
                    'Installed job provenance mismatch: ' + name)
            package = PurePosixPath(installed['package']).parent
            require(package.is_absolute() and 'site-packages' in package.parts
                    and 'venv' in package.parts, 'Job package is not an isolated installed package')
            bindings = installed['installed_bindings']
            require(set(bindings) == surface.expected_bindings(mode, arch is not None), 'Incomplete installed binding inventory')
            for binding_name, binding in bindings.items():
                member = PurePosixPath(binding['path']).relative_to(package).as_posix()
                match = EXTENSION.fullmatch('mojolearn/' + member)
                require(match is not None and match[1] == vendor
                        and (arch is None or match[2] == arch)
                        and (match[3] or 'fast') == mode and match[4] == binding_name
                        and binding.get('sha256') == extensions.get(member),
                        'Installed binding does not match final wheel vendor/mode')
                if binding_name in MODE_READBACK | {'_mojolearn_byte_lm'} or 'mode_code' in binding:
                    require('mode_code' in binding, 'Missing required native mode readback')
                    require(type(binding['mode_code']) is int and binding['mode_code'] == code,
                            'Installed native mode readback mismatch')
            if surface_name == 'umap-quality':
                surface.check_quality(json.loads((directory / (name + '.json')).read_text()),
                                      mode, bindings['_mojolearn_metrics']['sha256'])
    return qualification


def release_audit(wheel, source_root, proof_root, runtime_key):
    """File-only three-architecture preflight, also recomputed at final admission."""
    wheel, source_root, proof_root = map(Path, (wheel, source_root, proof_root))
    require(runtime_key in RELEASE_ARCHES_ACCEPTED, 'Unknown runtime architecture')
    extensions, sets = inspect_wheel(wheel, source_root, flat_python=True, byte_lm=True)
    version = surface.release_version(source_root)  # DEVIATION 2290: the source root's, never a literal
    carried = {'/'.join(k.split('/')[:2]) for k in sets}
    require(surface.arch_set_ok(carried),
            version + ' requires exactly sm_89, sm_90 (or sm_90a) and gfx942')
    require(runtime_key in carried, 'Qualified architecture is not one this wheel carries')
    inventory = native_inventory(source_root)
    source_sha = inventory_digest(inventory)
    with zipfile.ZipFile(wheel) as archive:
        member = 'mojolearn-' + version + '.dist-info/LINUX_PAYLOAD.json'
        payload = json.loads(archive.read(member))
        require(payload.get('schema') == 'mojolearn.linux-payload.v1'
                and payload.get('version') == version
                and surface.is_release_profile(payload), 'Wrong payload profile')
        require(payload.get('extensions') == {'mojolearn/' + n: h for n, h in extensions.items()}
                and payload.get('source_inventory') == inventory,
                'Final payload or source differs from assembly inventory')
        require(set(payload.get('sets', {})) == carried, 'Missing assembly set proofs')
        require(payload.get('optional_native', {}).get('_mojolearn_byte_lm') == dict(included=True,
                    supported_modes=['identical'], unsupported_modes=['fast', 'deterministic']),
                'Release byte-LM must be included in IDENTICAL only')
        for field, select in [('python_sha256', lambda n: n.endswith('.py')),
                              ('runtime_sha256', lambda n: '/.libs/' in n)]:
            actual = {n: hashlib.sha256(archive.read(n)).hexdigest()
                      for n in archive.namelist() if select(n)}
            require(payload.get(field) == actual, 'Final wheel differs from ' + field)
    proof_hashes = {}
    for key in sorted(carried):
        proof_path = proof_root / (key.replace('/', '-') + '.json')
        proof = json.loads(proof_path.read_text())
        require(proof.get('schema') == 'mojolearn.linux.build-provenance.v1'
                and proof.get('complete') is True and type(proof.get('build_exit')) is int
                and proof['build_exit'] == 0 and proof.get('action') == 'build', 'Failed architecture build')
        require(proof.get('source_inventory') == inventory and proof.get('source_sha256') == source_sha
                and re.fullmatch('[0-9a-f]{40}', proof.get('source_commit', ''))
                and proof['source_commit'] == payload.get('source_commit'), 'Different architecture source')
        expected = {'mojolearn/' + n: h for n, h in extensions.items() if n.startswith(key + '/')}
        require(len(expected) == 46 and proof.get('extensions') == expected, 'Architecture build bytes differ')
        proof_hashes[key] = digest_file(proof_path)
        require(payload['sets'][key] == {'sha256': proof_hashes[key], 'source_sha256': source_sha},
                'Assembly proof linkage differs')
    vendor, arch = runtime_key.split('/')
    return dict(sha256=digest_file(wheel), wheel=str(wheel.resolve()), advertised_vendors=['cuda', 'hip'],
                assembly_profile=surface.RELEASE_PROFILE, qualification_vendor=vendor, runtime_architecture=arch,
                source_sha256=source_sha, build_provenance_sha256=proof_hashes[runtime_key],
                architecture_build_proofs=proof_hashes, extension_hashes=extensions, sets=sets)


def check_release061(wheel, qualification_root, source_root):
    wheel, qualification_root, source_root = map(Path, (wheel, qualification_root, source_root))
    proof_root = qualification_root / 'build-proofs'
    inventory = native_inventory(source_root)
    # DEVIATION 2293: the architectures are the ones the WHEEL carries, with
    # the Hopper slot spelled however it was built. arch_set_ok still requires
    # exactly the three slots, filled once each.
    carried = {'/'.join(k.split('/')[:2])
               for k in inspect_wheel(wheel, source_root, flat_python=True, byte_lm=True)[1]}
    require(surface.arch_set_ok(carried),
            surface.release_version(source_root)
            + ' requires exactly sm_89, sm_90 (or sm_90a) and gfx942')
    directories = {}
    for key in sorted(carried):
        vendor, arch = key.split('/')
        directory = qualification_root / vendor / arch
        expected = release_audit(wheel, source_root, proof_root, key)
        recorded = json.loads((directory / 'wheel-audit.json').read_text())
        # DEVIATION 2290: an audit retained under the deprecated alias name
        # compares equal to the one recomputed under the current name.
        require(surface.is_release_profile(recorded) and
                {k: v for k, v in recorded.items() if k not in ('wheel', 'assembly_profile')} ==
                {k: v for k, v in expected.items() if k not in ('wheel', 'assembly_profile')},
                'Architecture audit differs')
        check_vendor(directory, vendor, expected['sha256'], inventory,
                     expected['extension_hashes'], expected['sets'], arch)
        check_corpora(directory, source_root)
        directories[key] = directory
    comparisons = {}
    # DEVIATION 2293: whichever Hopper spelling this wheel actually carries.
    for cuda in sorted(k for k in directories if k.startswith('cuda/')):
        umap = surface.compare(directories['hip/gfx942'], directories[cuda])
        ordered = compare_ordered_python.compare(directories['hip/gfx942'], directories[cuda])
        require(umap.get('status') == ordered.get('status') == 'PASSED', 'Architecture identity comparison failed')
        comparisons[cuda] = dict(umap=umap, ordered=ordered)
    return dict(schema='mojolearn.linux.release-admission.v2', status='PASSED',
                assembly_profile=surface.RELEASE_PROFILE, wheel=wheel.name, wheel_sha256=digest_file(wheel),
                source_sha256=inventory_digest(inventory), jobs_per_runtime_architecture=25,
                runtime_coverage={key: digest_file(path / 'qualification.json') for key, path in directories.items()},
                comparisons=comparisons,
                scope='Exact final wheel; 25 installed jobs on each of sm_89, sm_90 (or sm_90a), gfx942; byte-LM one-step/checkpoint functionality only; bounded UMAP/Ordered identity only')


def check_corpora(directory, source_root):
    """Refuse legacy fingerprints which silently omitted nested Mamba fixtures."""
    snapshot = json.loads((directory / 'qualification-sources.json').read_text())
    for case in surface.CORPUS_CASES:
        corpus = source_root / 'mamba/corpus' / case
        require((corpus / 'x.f32').is_file(), 'Missing current Mamba corpus: ' + case)
        prefix = 'mamba/corpus/' + case + '/'
        current = {p.relative_to(source_root).as_posix(): digest_file(p)
                   for p in corpus.rglob('*') if p.is_file()}
        recorded = {p: h for p, h in snapshot.items() if p.startswith(prefix)}
        require(recorded == current, 'Missing or changed qualified Mamba corpus: ' + case)


def check(wheel, qualification_root, source_root):
    wheel, qualification_root, source_root = map(Path, (wheel, qualification_root, source_root))
    wheel_sha = digest_file(wheel)
    inventory = native_inventory(source_root)
    extensions, sets = inspect_wheel(wheel, source_root)
    directories = {v: qualification_root / v for v in ('hip', 'cuda')}
    for vendor, directory in directories.items():
        check_vendor(directory, vendor, wheel_sha, inventory, extensions, sets)
        check_corpora(directory, source_root)
    # Recompute comparators from retained inputs; a standalone PASS JSON is
    # not evidence. They independently admit hashes, fixtures and raw bits.
    umap = surface.compare(directories['hip'], directories['cuda'])
    ordered = compare_ordered_python.compare(directories['hip'], directories['cuda'])
    require(umap.get('status') == 'PASSED' and ordered.get('status') == 'PASSED',
            'Retained installed identity comparator failed')
    return {'schema': 'mojolearn.linux.release-admission.v1', 'status': 'PASSED',
            'wheel': wheel.name, 'wheel_sha256': wheel_sha,
            'source_sha256': inventory_digest(inventory), 'jobs_per_vendor': 24,
            'qualification_sha256': {v: digest_file(p / 'qualification.json')
                                     for v, p in directories.items()},
            'umap_identity': umap, 'ordered_identity': ordered,
            'scope': 'Exact final dual-vendor wheel; 24 installed jobs per vendor and bounded IDENTICAL UMAP/Ordered fixtures'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('wheel', type=Path)
    parser.add_argument('--qualification-root', required=True, type=Path)
    parser.add_argument('--source-root', required=True, type=Path)
    # DEVIATION 2290: `release-0.6.1` is the deprecated alias of release-linux3.
    parser.add_argument('--profile', choices=('legacy', surface.RELEASE_PROFILE, 'release-0.6.1'),
                        default='legacy')
    args = parser.parse_args()
    try:
        action = check_release061 if args.profile in surface.RELEASE_PROFILES else check
        result = action(args.wheel, args.qualification_root, args.source_root)
    except (ValueError, OSError, KeyError, TypeError, AttributeError, zipfile.BadZipFile) as exc:
        result = {'status': 'FAILED', 'reason': str(exc)}
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'PASSED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
