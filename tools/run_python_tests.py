#!/usr/bin/env python3
"""Run pytest files only; module-style gates run via check_python_gates.py.

The inventory is exhaustive and syntax based: importing a gate during pytest
collection can terminate the entire suite with sys.exit, even when it passes.
No failed assertions are filtered. Both entrypoints must pass for release.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from python_test_inventory import inventory

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tests-root', type=Path, default=ROOT / 'python/mojolearn/tests')
    ap.add_argument('--plan', action='store_true')
    ap.add_argument('--installed', action='store_true',
                    help='test installed runtime while retaining source test paths; no package files modified')
    args, pytest_args = ap.parse_known_args(argv)
    suites = inventory(args.tests_root)
    if not suites['pytest']:
        ap.error('empty pytest suite')
    if args.plan:
        print(json.dumps({key: [str(p) for p in paths] for key, paths in suites.items()}, indent=2))
        return 0
    print(f"pytest files: {len(suites['pytest'])}; separate module gates: {len(suites['gates'])}", flush=True)
    if pytest_args[:1] == ['--']:
        pytest_args = pytest_args[1:]
    if args.installed:
        # Preload the installed runtime under isolated Python, then attach ONLY
        # its test namespace to source. Relative fixtures retain their package
        # names and Path(__file__).resolve() still sees repository evidence.
        code = '''import importlib.util, pathlib, sys
import mojolearn, pytest
root = pathlib.Path(sys.argv.pop(1)).resolve()
runtime = pathlib.Path(mojolearn.__file__).resolve()
if root.parent in runtime.parents:
    raise SystemExit('refusing source runtime in installed qualification')
print('Installed runtime:', runtime, flush=True)
spec = importlib.util.spec_from_file_location('mojolearn.tests', root/'__init__.py', submodule_search_locations=[str(root)])
tests = importlib.util.module_from_spec(spec)
sys.modules['mojolearn.tests'] = tests
spec.loader.exec_module(tests)
raise SystemExit(pytest.main(sys.argv[1:]))
'''
        return subprocess.call([sys.executable, '-I', '-c', code, str(args.tests_root),
                                '-q', '--import-mode=importlib',
                                *map(str, suites['pytest']), *pytest_args])
    return subprocess.call([sys.executable, '-m', 'pytest', '-q',
                            *map(str, suites['pytest']), *pytest_args], cwd=ROOT / 'python')


if __name__ == '__main__':
    raise SystemExit(main())
