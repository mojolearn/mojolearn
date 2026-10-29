#!/usr/bin/env python3
"""Import real host owners in every order, then the real installed package.

No stubs, monkeypatching, missing-binding skips or arithmetic benchmarks.
This is a development-family loader regression; wheel assembly has its own
boundary smoke. Each subprocess uses the same extension namespace as _backend.
"""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

CHILD = r"""
import importlib.machinery, importlib.util, json, pathlib, sys
package = pathlib.Path(sys.argv[1]).resolve()
order = json.loads(sys.argv[2])
modules = []
for family in order:
    basename = '_mojolearn_' + family + '_host'
    path = package / 'host' / (basename + '.so')
    assert path.is_file(), path
    full = 'mojolearn._host.' + basename
    loader = importlib.machinery.ExtensionFileLoader(full, str(path))
    spec = importlib.util.spec_from_loader(full, loader, origin=str(path))
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    sys.modules[full] = module
    modules.append(module)
canonical = modules[0]._NativeIdentityKey
assert all(module._NativeIdentityKey is canonical for module in modules)
import mojolearn
assert pathlib.Path(mojolearn.__file__).resolve().parent == package
from mojolearn import _backend
for family, original in zip(order, modules):
    assert _backend.load_host_module('_mojolearn_' + family + '_host') is original
print(json.dumps({'order': order, 'canonical_identity': True, 'package': str(package)}))
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', required=True)
    parser.add_argument('--package-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    families = ('tokenizer', 'mamba', 'runtime_sequential')
    env = os.environ.copy()
    for key in ('PYTHONPATH', 'DYLD_LIBRARY_PATH', 'LD_LIBRARY_PATH', 'MOJOLEARN_HOST_DIR'):
        env.pop(key, None)
    env.update(MOJOLEARN_VENDOR='cpu', MOJOLEARN_NUMERIC_MODE='identical')
    rows = []
    for order in itertools.permutations(families):
        log = args.output / ('-'.join(order) + '.log')
        command = [args.python, '-I', '-c', CHILD, str(args.package_root), json.dumps(order)]
        with log.open('w') as stream:
            process = subprocess.run(command, cwd=args.output, env=env,
                                     stdout=stream, stderr=subprocess.STDOUT)
        rows.append({'order': order, 'exit_code': process.returncode, 'log': str(log)})
    files = {family: args.package_root / 'host' / ('_mojolearn_' + family + '_host.so') for family in families}
    report = {'scope': 'six real native owner import orders + actual installed package import',
              'python': args.python, 'package': str(args.package_root), 'cases': rows,
              'binary_sha256': {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()},
              'failed': sum(row['exit_code'] != 0 for row in rows)}
    (args.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'cases': len(rows), 'failed': report['failed'], 'output': str(args.output)}))
    return bool(report['failed'])


if __name__ == '__main__':
    raise SystemExit(main())
