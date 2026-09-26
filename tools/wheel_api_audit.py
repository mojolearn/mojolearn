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
    return result


def reference_payload(root):
    """Committed reference table and portable models used by installed checks."""
    package = Path(root) / 'python' / 'mojolearn'
    base = package / 'verify_reference'
    paths = list(base.glob('*.json')) + list((base / 'models').glob('*'))
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
    return dict(wheel=wheel.name, sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
                public_names=sorted(surface), public_modules=modules,
                host_binaries=sorted(n for n in names if n.startswith('mojolearn/host/')
                                     and n.endswith('.so')))


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
                and n.rsplit('/', 1)[-1] in ('METADATA', 'WHEEL', 'gpu_plugins.json', 'gpu_plugin.json')}
    metadata = BytesParser(policy=policy.compat32).parsebytes(read.get(dist + '/METADATA', b''))
    tags = BytesParser(policy=policy.compat32).parsebytes(read.get(dist + '/WHEEL', b'')).get_all('Tag', [])
    payload = sorted(n for n in names if not n.startswith(dist + '/'))
    return dist, metadata, tags, payload, read


def split_audit(wheels):
    """THE SPLIT LINUX WHEELS (2026-09-25, python/mojolearn/gpu_plugins.py).

    The core `mojolearn` holds no GPU set; each plugin holds exactly its own
    vendor's sets (mojolearn/<vendor>/...) and nothing else, no Python and no
    runtime; every plugin requires exactly `mojolearn==<its version>`; the
    core requires EVERY plugin at its own version exactly and nothing else of
    them (2026-09-26, `pip install mojolearn` works for everyone:
    gpu_plugins.core_requirements) and declares no extras; the .dist-info markers agree
    with the payload; all wheels share one version and one tag; and no member
    is in two wheels. File inspection only. Returns {'wheels': [...],
    'problems': [...]}, and an empty `problems` is the pass."""
    from verify_linux_surface_qualification import load_gpu_plugins
    plugins = load_gpu_plugins()
    by_distribution = {row['distribution']: vendor for vendor, row in plugins.PLUGINS.items()}
    problems, rows, owner, versions, tagsets = [], [], {}, set(), set()
    for wheel in map(Path, wheels):
        facts = _wheel_facts(wheel)
        if facts is None:
            problems.append(f'{wheel.name}: not exactly one .dist-info directory')
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
        row = dict(wheel=wheel.name, path=str(wheel), distribution=name, version=version, tags=sorted(tags),
                   members=len(payload), binaries=sum(1 for m in payload if m.endswith('.so')))
        if name == plugins.CORE_DISTRIBUTION:
            row['role'] = plugins.CORE_PROFILE
            stray = [m for m in payload if plugins.member_vendor(m)]
            if stray:
                problems.append(f'{wheel.name}: the core carries {len(stray)} GPU set member(s), e.g. {stray[0]}')
            if 'mojolearn/__init__.py' not in payload:
                problems.append(f'{wheel.name}: the core carries no mojolearn/__init__.py')
            # `pip install mojolearn` WORKS FOR EVERYONE (Andrew, 2026-09-26):
            # the core requires BOTH plugins at its own version exactly, no
            # marker, no extra, and declares no extras at all
            extras = sorted(metadata.get_all('Provides-Extra', []))
            if extras:
                problems.append(f'{wheel.name}: the core declares Provides-Extra {extras}; it must declare none')
            on_plugin = [r for r in requires
                         if re.split(r'[\s;=<>!~\[(]', r, maxsplit=1)[0].strip().lower().replace('_', '-') in by_distribution]
            want = plugins.core_requirements(version)
            if sorted(on_plugin) != sorted(want) or len(on_plugin) != len(set(on_plugin)):
                problems.append(f'{wheel.name}: the core requires the GPU plugins as {on_plugin}; '
                                f'it must require exactly {want}')
            try:
                marker = json.loads(read[dist + '/' + plugins.CORE_MARKER])
            except (KeyError, ValueError):
                marker = None
            if marker != plugins.core_marker(version):
                problems.append(f'{wheel.name}: {plugins.CORE_MARKER} is missing or disagrees with gpu_plugins.py')
        elif name in by_distribution:
            vendor = by_distribution[name]
            row['role'] = plugins.PLUGINS[vendor]['profile']
            stray = [m for m in payload if plugins.member_vendor(m) != vendor]
            if stray:
                problems.append(f'{wheel.name}: carries {len(stray)} member(s) outside mojolearn/{vendor}/, '
                                f'e.g. {stray[0]}')
            if not row['binaries']:
                problems.append(f'{wheel.name}: carries no binary')
            if any(m.endswith('.py') for m in payload):
                problems.append(f'{wheel.name}: a plugin must carry no Python module')
            if requires != [f'{plugins.CORE_DISTRIBUTION}=={version}']:
                problems.append(f'{wheel.name}: Requires-Dist {requires}, want exactly '
                                f'[{plugins.CORE_DISTRIBUTION}=={version}]')
            if not wheel.name.startswith(f'{plugins.PLUGINS[vendor]["wheel_name"]}-{version}-'):
                problems.append(f'{wheel.name}: file name is not '
                                f'{plugins.PLUGINS[vendor]["wheel_name"]}-{version}-...')
            arches = sorted({m.split('/')[2] for m in payload if len(m.split('/')) > 3})
            row['arches'] = arches
            try:
                marker = json.loads(read[dist + '/' + plugins.PLUGIN_MARKER])
            except (KeyError, ValueError):
                marker = None
            if marker != plugins.plugin_marker(vendor, version, arches):
                problems.append(f'{wheel.name}: {plugins.PLUGIN_MARKER} is missing or does not name '
                                f'{vendor} {version} {arches}')
        else:
            problems.append(f'{wheel.name}: {name!r} is neither the core nor a plugin gpu_plugins.py names')
        rows.append(row)
    if len(versions) > 1:
        problems.append(f'the split wheels carry different versions: {sorted(versions)}')
    if len(tagsets) > 1:
        problems.append(f'the split wheels carry different tags: {sorted(tagsets)}')
    return dict(scope='Split Linux wheels: ownership, pins and markers; file inspection only.',
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
    """THE CORE PUBLISHES LAST (2026-09-26). The split Linux core requires
    mojolearn-nvidia==<v> and mojolearn-amd==<v>, so an index serving the core
    before both plugins would hand every `pip install mojolearn` an
    unresolvable requirement. For each split core among `wheels` (one whose
    .dist-info carries gpu_plugins.json), every plugin at the core's version
    must already be on `index` ('pypi' or 'testpypi'). Returns the problems;
    empty is the pass. Wheels other than a split core need nothing.
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
        if metadata.get('Name', '') != plugins.CORE_DISTRIBUTION or dist + '/' + plugins.CORE_MARKER not in read:
            continue
        version = metadata.get('Version', '')
        for row in plugins.PLUGINS.values():
            prefix = f'{row["wheel_name"]}-{version}-'
            for attempt in range(attempts):
                if any(n.startswith(prefix) and n.endswith('.whl') for n in files(index, row['distribution'], version)):
                    break
                if attempt + 1 < attempts:
                    sleep(30)
            else:
                problems.append(f'{wheel.name}: {row["distribution"]}=={version} is not on {index}; the split core '
                                f'requires it and publishes only after both plugins (publish the plugins first)')
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('wheels', nargs='+', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--require-complete', action='store_true',
                        help='fail for missing public exports or missing/stale source Python or reference payload')
    parser.add_argument('--split', action='store_true',
                        help='the wheels are the split Linux set (mojolearn, mojolearn-nvidia, mojolearn-amd): '
                             'run split_audit over all and the API audit over the core alone')
    parser.add_argument('--plugins-on-index', choices=sorted(INDEX_JSON),
                        help='refuse a split Linux core among the wheels unless mojolearn-nvidia and mojolearn-amd '
                             'of its version are already on this index (the core publishes last)')
    args = parser.parse_args()
    if args.plugins_on_index:
        problems = plugins_on_index(args.wheels, args.plugins_on_index)
        for problem in problems:
            print('::error::' + problem)
        if not problems:
            print(f'every split core among the wheels has both plugins on {args.plugins_on_index} (or none is a split core)')
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
