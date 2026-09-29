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
    return subprocess.call([sys.executable, '-m', 'pytest', '-q',
                            *map(str, suites['pytest']), *pytest_args], cwd=ROOT / 'python')


if __name__ == '__main__':
    raise SystemExit(main())
