#!/usr/bin/env python3
"""Compare source public names with wheel contents without importing either.

An export is packaging evidence, not numerical qualification or a distinct
algorithm. Release builders use --require-complete to reject a candidate that
omits or carries stale Python implementation/verifier bytes from this source.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
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


def payload_gaps(wheel, root):
    missing, changed = [], []
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        for member, source in source_payload(root).items():
            if member not in names:
                missing.append(member)
            elif archive.read(member) != source.read_bytes():
                changed.append(member)
    return dict(missing_python_payload=sorted(missing),
                changed_python_payload=sorted(changed))


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
            'source_names_absent_from_wheel', 'missing_python_payload', 'changed_python_payload'))
        results.append(row)
    return dict(scope='Public export inventory; includes aliases, helpers and constants. '
                      'Neither an algorithm count nor numerical qualification.',
                source_public_names=sorted(surface), source_public_modules=modules, wheels=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('wheels', nargs='+', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--require-complete', action='store_true',
                        help='fail for missing public exports or missing/stale source Python payload')
    args = parser.parse_args()
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
