#!/usr/bin/env python3
"""Run pytest files only; module-style gates run via check_python_gates.py.

The inventory is exhaustive and syntax based: importing a gate during pytest
collection can terminate the entire suite with sys.exit, even when it passes.
No failed assertions are filtered. Both entrypoints must pass for release.

The suite runs on every core with pytest-xdist (`-n <workers> --dist loadfile`:
each test file stays in one worker process, so module fixtures and module
state behave as in one process). Workers default to the cores this process
may use; MOJOLEARN_TEST_WORKERS overrides it and MOJOLEARN_TEST_WORKERS=1 is
the old serial run. Tests marked `serial` (python/pytest.ini) are left out of
the parallel pass and run afterwards in one process. Without pytest-xdist in
the environment the run is serial and says so.

python/mojolearn/tests/conftest.py adds two switches: MOJOLEARN_TEST_SLOW=0
deselects the tests marked `slow` (kept by default), and
MOJOLEARN_TEST_DURATIONS_DIR=<dir> records every test's duration for
tools/test_duration_report.py. MOJOLEARN_TEST_ORDER=<tsv> (that tool's
`--tsv` output from an earlier run) hands the parallel pass its files heaviest
first, so a long file starts at once instead of finishing last; files the tsv
does not name follow in their usual order.
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from python_test_inventory import inventory

ROOT = Path(__file__).resolve().parents[1]
SERIAL_MARK = 'pytest.mark.serial'


def default_workers():
    """Cores this process may use (affinity/container quota where the OS says)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def worker_count(env=None):
    env = os.environ if env is None else env
    text = env.get('MOJOLEARN_TEST_WORKERS', '').strip()
    if not text:
        return default_workers()
    if not text.isdigit() or int(text) < 1:
        raise SystemExit(f'MOJOLEARN_TEST_WORKERS must be a positive integer, got {text!r}')
    return int(text)


def user_chose_workers(pytest_args):
    """The caller passed its own -n/--numprocesses or turned xdist off."""
    return any(a == '-n' or a.startswith('--numprocesses') or (a.startswith('-n') and a[2:].isdigit())
               or a in ('no:xdist', '-pno:xdist') for a in pytest_args)


def serial_files(paths):
    """Files that mark tests `serial`, found by text like the inventory (never imported)."""
    return [p for p in paths if SERIAL_MARK in p.read_text(errors='replace')]


def heaviest_first(files, order_tsv):
    """`files` sorted by their total seconds in `order_tsv` (seconds<TAB>outcome<TAB>nodeid
    lines), heaviest first; files it does not name keep their order after them."""
    totals = {}
    for line in Path(order_tsv).read_text(encoding='utf-8').splitlines():
        parts = line.split('\t')
        if len(parts) >= 3:
            name = parts[2].split('::', 1)[0]
            totals[name] = totals.get(name, 0.0) + float(parts[0])
    def total(path):
        # nodeids are relative to python/ (the rootdir); match on that suffix
        text = Path(path).as_posix()
        for name, seconds in totals.items():
            if text == name or text.endswith('/' + name):
                return seconds
        return None
    known = sorted((f for f in files if total(f) is not None), key=lambda f: -total(f))
    return known + [f for f in files if total(f) is None]


DURATIONS_ROOT = Path.home() / 'mojolearn-evidence' / 'test-durations'


def durations_dir():
    """This run's per-test duration directory (conftest records into it; on by default, "0" turns it off)."""
    value = os.environ.get('MOJOLEARN_TEST_DURATIONS_DIR', '').strip()
    if value == '0':
        return None
    if value:
        return Path(value)
    import time
    path = DURATIONS_ROOT / time.strftime('%Y%m%d-%H%M%S')
    os.environ['MOJOLEARN_TEST_DURATIONS_DIR'] = str(path)
    return path


def newest_order():
    """The newest run's heaviest-first tsv, so the parallel pass starts long files first without being told."""
    try:
        runs = sorted(p for p in DURATIONS_ROOT.glob('*/all.tsv'))
    except OSError:
        return ''
    return str(runs[-1]) if runs else ''


def write_duration_report(directory, workers):
    """After the run: the per-file and per-test report (tools/test_duration_report.py) and all.tsv next to it."""
    if directory is None or not any(directory.glob('durations-*.json')):
        return
    report = ROOT / 'tools' / 'test_duration_report.py'
    subprocess.call([sys.executable, str(report), str(directory), '--workers', str(workers), '--top', '40',
                     '--tsv', str(directory / 'all.tsv')])
    print(f'durations: {directory} (all.tsv feeds the next run\'s order)', flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
        # Stays one process: xdist workers are fresh interpreters and would not
        # carry the mojolearn.tests namespace attached below.
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
mojolearn.tests = tests
spec.loader.exec_module(tests)
raise SystemExit(pytest.main(sys.argv[1:]))
'''
        return subprocess.call([sys.executable, '-I', '-c', code, str(args.tests_root),
                                '-q', '--import-mode=importlib',
                                *map(str, suites['pytest']), *pytest_args])
    durations = durations_dir()
    files = [str(p) for p in suites['pytest']]
    pytest = [sys.executable, '-m', 'pytest', '-q']
    workers = worker_count()
    # The slowest tests are printed at the end of every run (Andrew, 2026-11-06: the suite's heavy tail is
    # cut from measured durations, never guessed). MOJOLEARN_TEST_DURATIONS=0 turns the report off.
    # (named slowest_n: `durations` above is the per-test duration directory)
    slowest_n = os.environ.get('MOJOLEARN_TEST_DURATIONS', '80').strip()
    if slowest_n != '0' and not any(a.startswith('--durations') for a in pytest_args):
        pytest_args = ['--durations', slowest_n, '--durations-min', '0.5', *pytest_args]
    have_xdist = importlib.util.find_spec('xdist') is not None
    if user_chose_workers(pytest_args) or workers == 1:
        rc = subprocess.call([*pytest, *files, *pytest_args], cwd=ROOT / 'python')
        write_duration_report(durations, 1)
        return rc
    if not have_xdist:
        print('pytest-xdist is not installed in this environment; running serially', flush=True)
        rc = subprocess.call([*pytest, *files, *pytest_args], cwd=ROOT / 'python')
        write_duration_report(durations, 1)
        return rc
    serial = [str(p) for p in serial_files(suites['pytest'])]
    order = os.environ.get('MOJOLEARN_TEST_ORDER', '').strip() or newest_order()
    if order:
        files = heaviest_first(files, order)
        print(f'parallel pass order: heaviest first from {order}', flush=True)
    print(f'parallel pass: {workers} workers, --dist loadfile; serial pass: {len(serial)} file(s)', flush=True)
    rc = subprocess.call([*pytest, '-n', str(workers), '--dist', 'loadfile', '-m', 'not serial',
                          *files, *pytest_args], cwd=ROOT / 'python')
    if not serial:
        write_duration_report(durations, workers)
        return rc
    print('serial pass: tests marked serial, one process', flush=True)
    rc_serial = subprocess.call([*pytest, '-p', 'no:xdist', '-m', 'serial', *serial, *pytest_args],
                                cwd=ROOT / 'python')
    if rc_serial == 5:  # no test selected: the files only mention the mark
        rc_serial = 0
    write_duration_report(durations, workers)
    return rc or rc_serial


if __name__ == '__main__':
    raise SystemExit(main())
