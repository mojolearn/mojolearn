#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reject provider/CPython dependencies in selected native neural closures.

This checks source dependencies, not runtime behavior or the purity of every file
in the historically mixed core/autograd directories. There is no debt baseline.
"""
import argparse
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
NATIVE_ROOTS = (
    'core.cast_in', 'core.small_mlp_state_impl', 'autograd.program_io',
    'autograd.model', 'autograd.program_runtime', 'autograd.ops_api',
)
IMPORT = re.compile(r'^\s*(?:from\s+([\w.]+)\s+import\b|import\s+([\w.]+))', re.M)


def imports(source):
    # Ignore comments/docstrings so documentation naming a provider is harmless.
    source = re.sub(r'""".*?"""|\'\'\'.*?\'\'\'', '', source, flags=re.S)
    source = '\n'.join(line.split('#', 1)[0] for line in source.splitlines())
    return [a or b for a, b in IMPORT.findall(source)]


def audit(root=ROOT, roots=NATIVE_ROOTS):
    root = Path(root)
    seen, pending, failures = set(), [(name, [name]) for name in roots], []
    while pending:
        module, chain = pending.pop()
        if module in seen:
            continue
        seen.add(module)
        path = root.joinpath(*module.split('.')).with_suffix('.mojo')
        if not path.is_file():
            failures.append({'chain': chain, 'reason': 'missing native source'})
            continue
        for imported in imports(path.read_text()):
            if (imported == 'std.python' or imported.startswith('std.python.')
                    or imported == 'bindings' or imported.startswith('bindings.')
                    or imported.startswith('adapters.') or imported.endswith('_pyapi')):
                failures.append({'chain': chain + [imported], 'reason': 'provider or CPython dependency'})
                continue
            local = root.joinpath(*imported.split('.')).with_suffix('.mojo')
            if local.is_file():
                pending.append((imported, chain + [imported]))
            elif imported.split('.')[0] in {'core', 'autograd', 'sequence', 'training', 'checks', 'gemm', 'mamba'}:
                failures.append({'chain': chain + [imported], 'reason': 'unresolved repository dependency'})
    return {'schema': 'mojolearn.neural-native-dependencies.v1', 'roots': list(roots),
            'source_modules': sorted(seen), 'failures': failures, 'passed': not failures,
            'runtime_validated': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    result = audit()
    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + '\n')
    print(f"neural native dependencies: {len(result['source_modules'])} modules, "
          f"{len(result['failures'])} failures")
    for item in result['failures'][:10]:
        print(item['reason'] + ': ' + ' -> '.join(item['chain']))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
