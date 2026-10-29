#!/usr/bin/env python3
"""Check literal native package-owner lookups against declared setup connections.

This checks source declarations, not import order, dynamic module names, optional
third-party modules, or runtime availability. Public installed checks cover order.
The small export contract follows the native consumers; a new owner needs review.
No product module is imported to inspect its setup.
"""
import argparse
import ast
import json
from pathlib import Path
import re
import subprocess

EXPORTS = {
    'mojolearn._array': {'_ARRAY_CONTEXT'},
    'mojolearn._expansion_trees': {'_TREES_CONTEXT'},
    'mojolearn._mode': {'_MODE_METADATA_CONTEXT'},
    'mojolearn._native_family_bridge': {'_CONTEXT'},
    'mojolearn._native_prep_stateless_bridge': {'CONTEXT_CONNECTION'},
    'mojolearn._native_array_storage': {'ARRAY_TYPE', 'ZERO_STORES'},
    'mojolearn._native_loaded_modules': {'PATH_TYPES', 'PATH_BASES'},
    'mojolearn._native_public_attributes': {'_CONTEXT'},
    'mojolearn.importers._program': {'_IMPORT_CONTEXT'},
    'mojolearn._native_family_layouts': {'LAYOUTS'},
}
LOOKUP = re.compile(r'\bnative_loaded_module\(\s*["\'](mojolearn\.[\w.]+)["\']')


def relevant(path):
    return (path.startswith('python/mojolearn/') and path.endswith('.py')
            and '/tests/' not in path) or (
        path.endswith('.mojo') and not path.startswith(('tools/', 'experiments/', 'tests/', 'benchmarks/')))


def sources(root, tree=None):
    if tree is None:
        paths = subprocess.check_output(['git', 'ls-files'], cwd=root, text=True).splitlines()
        return {p: (root / p).read_text() for p in paths if relevant(p) and (root / p).is_file()}
    entries = subprocess.check_output(['git', 'ls-tree', '-r', tree], cwd=root, text=True).splitlines()
    rows = [(line.split('\t', 1)[1], line.split()[2]) for line in entries
            if relevant(line.split('\t', 1)[1]) and line.split()[1] == 'blob']
    data = subprocess.check_output(['git', 'cat-file', '--batch'], cwd=root,
                                   input=('\n'.join(sha for _, sha in rows) + '\n').encode())
    result, offset = {}, 0
    for path, _ in rows:
        end = data.index(b'\n', offset)
        size = int(data[offset:end].split()[2])
        result[path] = data[end + 1:end + 1 + size].decode()
        offset = end + 2 + size
    return result


def inspect(files):
    consumers, declarations, errors = {}, {}, []
    for path, source in files.items():
        if path.endswith('.mojo'):
            # Mojo line comments cannot introduce a product consumer.
            for line, text in enumerate(source.splitlines(), 1):
                for match in LOOKUP.finditer(text.split('#', 1)[0]):
                    consumers.setdefault(match[1], []).append(f'{path}:{line}')
            continue
        module = path.removeprefix('python/').removesuffix('.py').replace('/', '.')
        if module.endswith('.__init__'):
            module = module.removesuffix('.__init__')
        parsed = ast.parse(source, path)
        names = {'connect_module'} if module == 'mojolearn._native_loaded_modules' else set()
        for node in ast.walk(parsed):
            if isinstance(node, ast.ImportFrom) and node.module in ('_native_loaded_modules', 'mojolearn._native_loaded_modules'):
                names.update(alias.asname or alias.name for alias in node.names if alias.name == 'connect_module')
        for node in ast.walk(parsed):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id in names and len(node.args) >= 3):
                continue
            owner = node.args[0]
            if isinstance(owner, ast.Name) and owner.id == '__name__':
                name = module
            elif isinstance(owner, ast.Constant) and isinstance(owner.value, str):
                name = owner.value
            else:
                continue
            if not name.startswith('mojolearn.'):
                continue
            try:
                exports = ast.literal_eval(node.args[2])
                if not isinstance(exports, tuple) or any(type(v) is not str for v in exports):
                    raise ValueError('not a literal string tuple')
            except (ValueError, TypeError):
                errors.append(f'{path}:{node.lineno}: {name}: exports must be a literal string tuple')
                continue
            declarations.setdefault(name, []).append({'source': f'{path}:{node.lineno}', 'exports': sorted(exports)})
    for owner in sorted(consumers):
        if owner not in EXPORTS:
            errors.append(f'{owner}: new package owner needs a reviewed export contract')
        rows = declarations.get(owner, [])
        if len(rows) != 1:
            errors.append(f'{owner}: expected one setup declaration, found {len(rows)}')
        elif owner in EXPORTS and not EXPORTS[owner].issubset(rows[0]['exports']):
            errors.append(f'{owner}: missing connected exports {sorted(EXPORTS[owner] - set(rows[0]["exports"]))}')
    return {'consumers': consumers, 'declarations': declarations, 'errors': errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--tree', help='Read committed source instead of the working tree')
    parser.add_argument('--json-out', type=Path)
    args = parser.parse_args()
    report = inspect(sources(args.root, args.tree))
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2) + '\n')
    print(f'native-package-connections: {len(report["consumers"])} owners, '
          f'{sum(map(len, report["consumers"].values()))} literal consumers, {len(report["errors"])} errors')
    for error in report['errors']:
        print('ERROR:', error)
    return bool(report['errors'])


if __name__ == '__main__':
    raise SystemExit(main())
