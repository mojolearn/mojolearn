#!/usr/bin/env python3
"""Compare source public names with wheel contents without importing either.

An export is packaging evidence, not numerical qualification or a distinct
algorithm. Release builders use --require-complete to reject a candidate that
omits or carries stale Python implementation/verifier bytes or bundled reference
assets from this source.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tempfile
import zipfile

import verification_matrix as matrix


def source_payload(root):
    """All package implementations, including a newly added subpackage.

    Do not restrict this to the packer's allow-list: that would miss a package
    accidentally omitted from both setuptools and the Linux packing loop.
    Tests and bytecode are development inputs, not installed algorithms.
    """
    root = Path(root)
    package = root / 'python' / 'mojolearn'
    result = {}
    for path in package.rglob('*.py'):
        relative = path.relative_to(package)
        if any(part in ('tests', '__pycache__') for part in relative.parts):
            continue
        result['mojolearn/' + relative.as_posix()] = path
    # Both builders must embed the current executable verifier, not just its
    # CLI. Source checkouts need not contain these generated copies.
    for module, tool in (('_identity_break.py', 'identity_break.py'),
                         ('_identity_trace_diff.py', 'identity_trace_diff.py')):
        result['mojolearn/' + module] = root / 'tools' / tool
    for fragment in sorted((root / "tools" / "identity_lanes").glob("*.py")):
        result["mojolearn/_identity_lane_" + fragment.name] = fragment
    return result


def reference_payload(root):
    """Committed reference table and portable models used by installed checks."""
    package = Path(root) / 'python' / 'mojolearn'
    base = package / 'verify_reference'
    paths = (list(base.glob('*.json')) + list((base / 'models').glob('*'))
             + list((base / 'ctr_models').glob('*.npz')))
    return {'mojolearn/' + path.relative_to(package).as_posix(): path
            for path in paths if path.is_file()}


def payload_gaps(wheel, root):
    result = {}
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        for kind, sources in (('python', source_payload(root)),
                              ('reference', reference_payload(root))):
            missing, changed = [], []
            for member, source in sources.items():
                if member not in names:
                    missing.append(member)
                elif archive.read(member) != source.read_bytes():
                    changed.append(member)
            result['missing_' + kind + '_payload'] = sorted(missing)
            result['changed_' + kind + '_payload'] = sorted(changed)
    return result


def inspect_wheel(wheel):
    wheel = Path(wheel)
    original_root = matrix.ROOT
    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        for name in names:
            path = PurePosixPath(name)
            if not name.startswith('mojolearn/') or not name.endswith('.py'):
                continue
            if '..' in path.parts or path.is_absolute():
                raise ValueError(f'unsafe wheel member: {name}')
            target = Path(tmp) / 'python' / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
        if not (Path(tmp) / 'python/mojolearn/__init__.py').is_file():
            raise ValueError('wheel has no mojolearn package')
        try:
            matrix.ROOT = tmp
            surface, modules = matrix.public_surface()
        finally:
            matrix.ROOT = original_root
    host_binaries = sorted(n for n in names if n.startswith('mojolearn/host/') and n.endswith('.so'))
    return dict(wheel=wheel.name, sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
                public_names=sorted(surface), public_modules=modules,
                host_binaries=host_binaries, host_bindings=host_bindings(wheel, host_binaries))


#: The index a merged build writes beside its files (bindings/merged_build.sh).
MERGED_INDEX = '_mojolearn_merged.json'


def host_bindings(wheel, host_binaries):
    """{binding: wheel member that serves it} for host/: a binding's own
    `<name>.so`, else the merged file host/'s index names for it (the merged
    layout ships one file for every host binding)."""
    served = {}
    with zipfile.ZipFile(wheel) as archive:
        if 'mojolearn/host/' + MERGED_INDEX in archive.namelist():
            index = json.loads(archive.read('mojolearn/host/' + MERGED_INDEX))
            for fname, members in index.get('files', {}).items():
                if 'mojolearn/host/' + fname in host_binaries:
                    served.update({m: 'mojolearn/host/' + fname for m in members})
    for member in host_binaries:
        name = PurePosixPath(member).stem
        if member.endswith('.so') and not any(member == f for f in served.values()):
            served[name] = member
    return dict(sorted(served.items()))


def audit(wheels):
    surface, modules = matrix.public_surface()
    results = []
    for wheel in wheels:
        row = inspect_wheel(wheel)
        row['source_names_absent_from_wheel'] = sorted(set(surface) - set(row['public_names']))
        row['wheel_names_absent_from_source'] = sorted(set(row['public_names']) - set(surface))
        row.update(payload_gaps(wheel, matrix.ROOT))
        row['source_payload_complete'] = not any(row[key] for key in (
            'source_names_absent_from_wheel', 'missing_python_payload', 'changed_python_payload',
            'missing_reference_payload', 'changed_reference_payload'))
        results.append(row)
    return dict(scope='Public export inventory; includes aliases, helpers and constants. '
                      'Neither an algorithm count nor numerical qualification.',
                source_public_names=sorted(surface), source_public_modules=modules, wheels=results)


def _wheel_facts(wheel):
    """(dist-info dir, METADATA message, WHEEL tags, payload names, the small
    .dist-info files read) of one wheel, without extracting it; None when the
    wheel does not hold exactly one .dist-info directory."""
    from email import policy
    from email.parser import BytesParser
    with zipfile.ZipFile(wheel) as archive:
        names = [n for n in archive.namelist() if not n.endswith('/')]
        dists = sorted({n.split('/')[0] for n in names if n.split('/')[0].endswith('.dist-info')})
        if len(dists) != 1:
            return None
        dist = dists[0]
        read = {n: archive.read(n) for n in names if n.startswith(dist + '/')
                and n.rsplit('/', 1)[-1] in ('METADATA', 'WHEEL', 'gpu_plugins.json', 'gpu_plugin.json', 'gpu_payload.json')}
    metadata = BytesParser(policy=policy.compat32).parsebytes(read.get(dist + '/METADATA', b''))
    tags = BytesParser(policy=policy.compat32).parsebytes(read.get(dist + '/WHEEL', b'')).get_all('Tag', [])
    payload = sorted(n for n in names if not n.startswith(dist + '/'))
    return dist, metadata, tags, payload, read


def split_audit(wheels):
    """Check core, vendor bundles and experimental payload ownership.

    This is file inspection, not numerical qualification. Experimental payloads
    may be inspected here; the publisher independently refuses them.
    """
    from verify_linux_surface_qualification import load_gpu_plugins
    plugins = load_gpu_plugins()
    packages = {r['distribution']: r for r in plugins.distribution_rows(include_experimental=True)}
    problems, rows, owner, versions, tagsets = [], [], {}, set(), set()
    for wheel in map(Path, wheels):
        facts = _wheel_facts(wheel)
        if facts is None:
            problems.append(f'{wheel.name}: not exactly one dist-info directory')
            continue
        dist, metadata, tags, payload, read = facts
        name, version = metadata.get('Name', ''), metadata.get('Version', '')
        versions.add(version)
        tagsets.add(tuple(sorted(tags)))
        requires = metadata.get_all('Requires-Dist', [])
        for member in payload:
            if member in owner:
                problems.append(f'{member} is in both {owner[member]} and {wheel.name}')
            owner[member] = wheel.name
        row = dict(wheel=wheel.name, path=str(wheel), distribution=name, version=version,
                   tags=sorted(tags), members=len(payload),
                   binaries=sum(m.endswith('.so') for m in payload))
        if wheel.stat().st_size > plugins.wheel_size_limit(name):
            problems.append(f'{wheel.name}: exceeds the configured project upload budget')
        if name == plugins.CORE_DISTRIBUTION:
            row['role'] = plugins.CORE_PROFILE
            stray = [m for m in payload if plugins.member_vendor(m)]
            if stray:
                problems.append(f'{wheel.name}: the core carries GPU members, e.g. {stray[0]}')
            if 'mojolearn/__init__.py' not in payload:
                problems.append(f'{wheel.name}: the core carries no mojolearn/__init__.py')
            extras = sorted(metadata.get_all('Provides-Extra', []))
            if set(extras) - {'verify', 'numpy'}:
                problems.append(f'{wheel.name}: the core declares Provides-Extra {extras}; only numpy and verify are allowed')
            pins = [r for r in requires if re.split(r'[\s;=<>!~\[(]', r, maxsplit=1)[0]
                    .strip().lower().replace('_', '-') in packages]
            if sorted(pins) != sorted(plugins.core_requirements(version)):
                problems.append(f'{wheel.name}: core GPU dependency pins disagree with the package registry')
            marker_name, expected = plugins.CORE_MARKER, plugins.core_marker(version)
        elif name == plugins.META_DISTRIBUTION:
            # THE LINUX `mojolearn` WHEEL (2026-10-20): metadata only, requiring
            # exactly the core and both vendors at its version.
            row['role'] = plugins.META_PROFILE
            if not wheel.name.startswith(f"{plugins.META_WHEEL_NAME}-{version}-"):
                problems.append(f'{wheel.name}: filename differs from package metadata')
            if payload:
                problems.append(f'{wheel.name}: the mojolearn meta wheel carries files, e.g. {payload[0]}')
            if requires != plugins.meta_requirements(version):
                problems.append(f'{wheel.name}: Requires-Dist differs from the core and both vendors at {version}')
            if metadata.get_all('Provides-Extra', []):
                problems.append(f'{wheel.name}: the mojolearn meta wheel must not declare extras')
            marker_name, expected = plugins.PLUGIN_MARKER, plugins.meta_marker(version)
        elif name in packages:
            package = packages[name]
            profile = package['profile']
            row['role'] = profile
            if not wheel.name.startswith(f"{package['wheel_name']}-{version}-"):
                problems.append(f'{wheel.name}: filename differs from package metadata')
            if requires != plugins.package_requirements(profile, version):
                problems.append(f'{wheel.name}: Requires-Dist differs from exact package dependencies')
            if metadata.get_all('Provides-Extra', []):
                problems.append(f'{wheel.name}: GPU package must not declare extras')
            marker_name = plugins.PLUGIN_MARKER
            try:
                actual_marker = json.loads(read[dist + '/' + marker_name])
                bundle = actual_marker.get('bundled_ptx')
            except (KeyError, ValueError, AttributeError):
                bundle = None
            if package.get('ptx'):
                try:
                    plugins.validate_bundle_descriptor(bundle)
                    row['ptx_reference'] = bundle
                except ValueError as exc:
                    problems.append(f'{wheel.name}: invalid PTX dependency manifest: {exc}')
            elif package['code_format'] == 'ptx':
                # THE PTX SLOT (Andrew 2026-10-10: PTX is a normal target; no flag):
                # owned by the PTX dependency, with its manifest bound by both
                # package markers and its bytes by the release inventory.
                try:
                    if bundle is None:
                        raise ValueError('the vendor marker binds no PTX set')
                    prefix = plugins.BUNDLED_PTX_ROOT + '/'
                    manifest_member = prefix + plugins.BASELINE_MANIFEST
                    plugins.validate_bundle_descriptor(bundle)
                    with zipfile.ZipFile(wheel) as archive:
                        files = {m[len(prefix):]: hashlib.sha256(archive.read(m)).hexdigest()
                                 for m in payload if m.startswith(prefix) and m.endswith('.so')}
                        manifest_bytes = archive.read(manifest_member)
                        # the merged-file indexes (bindings/merged_build.sh) are bound too
                        indexes = {m[len(prefix):]: hashlib.sha256(archive.read(m)).hexdigest()
                                   for m in payload if m.startswith(prefix) and m.endswith('/_mojolearn_merged.json')
                                   or m == prefix + '_mojolearn_merged.json'}
                        plugins.validate_bundled_ptx(bundle, manifest_bytes, files, indexes)
                        ptx_source = json.loads(manifest_bytes)['source_commit']
                        inventory_member = dist + '/LINUX_PAYLOAD.json'
                        if inventory_member in archive.namelist():
                            inventory = json.loads(archive.read(inventory_member))
                            if inventory.get('source_commit') != ptx_source:
                                raise ValueError('the PTX set and the release inventory name different commits')
                            for name, digest in files.items():
                                if inventory.get('extensions', {}).get(prefix + name) != digest:
                                    raise ValueError('PTX set bytes differ from the release inventory')
                    allowed = {prefix + name for name in files} | {manifest_member}
                    if {m for m in payload if m.startswith(prefix)} != allowed:
                        raise ValueError('undeclared PTX set files')
                    row['ptx'] = dict(bundle, source_commit=ptx_source, arch=plugins.PTX_ARCH)
                except (ValueError, KeyError, TypeError) as exc:
                    problems.append(f'{wheel.name}: invalid PTX slot: {exc}')
            elif bundle is not None:
                problems.append(f'{wheel.name}: only the NVIDIA wheel carries a PTX slot')
            arches = []
            stray = []
            for member in payload:
                try:
                    valid = plugins.owns_member(profile, member) and member == plugins.installed_member(member)
                except ValueError:
                    valid = False
                if not valid:
                    stray.append(member)
            if stray:
                problems.append(f'{wheel.name}: members outside its architecture payload: {stray[0]}')
            if not row['binaries'] or any(m.endswith('.py') for m in payload):
                problems.append(f'{wheel.name}: payload must contain binaries and no Python')
            arches = sorted({m.split('/')[2] for m in payload if m.endswith('.so')
                             and m.split('/')[1] == package['directory']})
            if not plugins.valid_arches(profile, arches):
                problems.append(f'{wheel.name}: GPU wheel must contain one set per registered architecture slot')
            row['arches'] = arches
            try:
                expected = plugins.package_marker(profile, version, arches, bundled_ptx=bundle)
            except ValueError as exc:
                problems.append(f'{wheel.name}: invalid vendor marker ownership: {exc}')
                expected = None
        else:
            problems.append(f'{wheel.name}: unknown distribution {name!r}')
            rows.append(row)
            continue
        try:
            marker = json.loads(read[dist + '/' + marker_name])
        except (KeyError, ValueError):
            marker = None
        if marker != expected:
            problems.append(f'{wheel.name}: {marker_name} is missing or disagrees with package ownership')
        rows.append(row)
    if len(versions) > 1:
        problems.append(f'the split wheels carry different versions: {sorted(versions)}')
    if len(tagsets) > 1:
        problems.append(f'the split wheels carry different tags: {sorted(tagsets)}')
    ptx = next((r.get('ptx') for r in rows if r.get('role') == plugins.PTX_PROFILE), None)
    if ptx is not None:
        for row in rows:
            if 'ptx_reference' in row and row['ptx_reference']['manifest_sha256'] != ptx['manifest_sha256']:
                problems.append(f"{row['wheel']}: PTX dependency manifest differs from the vendor marker")
    return dict(scope='Split wheel ownership, dependencies, size and metadata; no numerical qualification.',
                wheels=rows, problems=problems)


#: The JSON API root of each index a release publishes to.
INDEX_JSON = {'pypi': 'https://pypi.org/pypi', 'testpypi': 'https://test.pypi.org/pypi'}


def _index_files(index, project, version, timeout=20):
    """The file names `index` serves for project==version ([] when none)."""
    import urllib.request
    url = f'{INDEX_JSON[index]}/{project}/{version}/json'
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return [f.get('filename', '') for f in json.load(response).get('urls', [])]
    except Exception:
        return []


def plugins_on_index(wheels, index, files=None, attempts=6, sleep=None):
    """DEPENDENCIES PUBLISH FIRST (2026-10-20). mojolearn-core requires no
    vendor; each vendor requires mojolearn-core (and mojolearn-nvidia its PTX
    package); the Linux `mojolearn` wheel requires the core and both vendors.
    For each such wheel among `wheels`, every one of its mojolearn
    requirements at its version must already be on `index` ('pypi' or
    'testpypi'), so the index never serves an unresolvable requirement.
    Returns the problems; empty is the pass.
    `files(index, project, version)` stands in for the index in tests; the
    lookup is retried `attempts` times, the index may still be propagating an
    upload made minutes earlier."""
    from verify_linux_surface_qualification import load_gpu_plugins
    plugins = load_gpu_plugins()
    files = files or _index_files
    if sleep is None:
        import time
        sleep = time.sleep
    problems = []
    for wheel in map(Path, wheels):
        facts = _wheel_facts(wheel)
        if facts is None:
            continue
        dist, metadata, _, _, read = facts
        name = metadata.get('Name', '')
        version = metadata.get('Version', '')
        packages = {r['distribution']: r for r in plugins.distribution_rows()}
        if name == plugins.CORE_DISTRIBUTION and dist + '/' + plugins.CORE_MARKER in read:
            required = plugins.core_requirements(version)
        elif name == plugins.META_DISTRIBUTION and dist + '/' + plugins.PLUGIN_MARKER in read:
            required = plugins.meta_requirements(version)
        elif name in packages:
            required = plugins.package_requirements(packages[name]['profile'], version)
        else:
            continue
        wheel_names = {r['distribution']: r['wheel_name'] for r in packages.values()}
        wheel_names[plugins.CORE_DISTRIBUTION] = plugins.CORE_WHEEL_NAME
        for requirement in required:
            dependency = requirement.split('==', 1)[0]
            row = dict(distribution=dependency, wheel_name=wheel_names[dependency])
            prefix = f'{row["wheel_name"]}-{version}-'
            for attempt in range(attempts):
                if any(n.startswith(prefix) and n.endswith('.whl') for n in files(index, row['distribution'], version)):
                    break
                if attempt + 1 < attempts:
                    sleep(30)
            else:
                problems.append(f'{wheel.name}: {row["distribution"]}=={version} is not on {index}; the package '
                                f'requires it and publishes only after its dependencies')
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('wheels', nargs='+', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--require-complete', action='store_true',
                        help='fail for missing public exports or missing/stale source Python or reference payload')
    parser.add_argument('--split', action='store_true',
                        help='the wheels are split Linux core, vendor bundles, or experimental payloads: '
                             'run split_audit over all and the API audit over the core alone')
    parser.add_argument('--plugins-on-index', choices=sorted(INDEX_JSON),
                        help="require each wheel's exact mojolearn dependencies on the selected index")
    args = parser.parse_args()
    if args.plugins_on_index:
        problems = plugins_on_index(args.wheels, args.plugins_on_index)
        for problem in problems:
            print('::error::' + problem)
        if not problems:
            print(f'the mojolearn dependencies of every wheel are available on {args.plugins_on_index}')
        return int(bool(problems))
    if args.split:
        split = split_audit(args.wheels)
        cores = [Path(row['path']) for row in split['wheels'] if row.get('role') == 'core-linux']
        report = audit(cores) if cores else dict(wheels=[])
        report['split'] = split
        result = json.dumps(report, indent=2) + '\n'
        if args.output:
            args.output.write_text(result)
        else:
            print(result, end='')
        return int(bool(split['problems']) or (args.require_complete and any(
            not row['source_payload_complete'] for row in report['wheels'])))
    report = audit(args.wheels)
    result = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.write_text(result)
    else:
        print(result, end='')
    return int(args.require_complete and any(not row['source_payload_complete']
                                            for row in report['wheels']))


if __name__ == '__main__':
    raise SystemExit(main())
