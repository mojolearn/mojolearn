#!/usr/bin/env python3
"""Compare source public names with wheel contents without importing either.

An export is packaging evidence, not numerical qualification or a distinct
algorithm. Pass both the published wheel and the next candidate to distinguish
already-packaged additions from work that still needs a public API.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tempfile
import zipfile

import verification_matrix as matrix


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
        results.append(row)
    return dict(scope='Public export inventory; includes aliases, helpers and constants. '
                      'Neither an algorithm count nor numerical qualification.',
                source_public_names=sorted(surface), source_public_modules=modules, wheels=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('wheels', nargs='+', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = json.dumps(audit(args.wheels), indent=2) + '\n'
    if args.output:
        args.output.write_text(result)
    else:
        print(result, end='')


if __name__ == '__main__':
    main()
