"""Suite-wide collection rules: environment requirements and the `slow` switch.

Environment markers (registered in python/pytest.ini) skip a test at
collection, with the exact reason, when this environment cannot run it. A
skip is reported as a skip; it never passes. Each probe runs at most once per
session and only when a collected test carries its marker.

    needs_fast_tier          a `fast` tier binary set is installed
    needs_identity_harness   `mojolearn._identity_break` is present (the wheel build
                             copies tools/identity_break.py into the package)
    needs_numpy_free_python  MOJOLEARN_NUMPY_FREE_PYTHON names an interpreter
    needs_installed_package  `import mojolearn` works in a child with no PYTHONPATH
                             (the package is installed or editable-installed)

`slow` marks a test whose work is long by nature (each mark is listed with its
reason in the lane table). MOJOLEARN_TEST_SLOW=0 deselects them; the default
keeps them in.

The probes live in _suite_env.py, which test modules also use
(`identity_harness_or_skip` for a module-level import of the harness).
"""
import os

import pytest

from mojolearn.tests._suite_env import ENVIRONMENT


def pytest_collection_modifyitems(config, items):
    keep_slow = os.environ.get("MOJOLEARN_TEST_SLOW", "1").strip() != "0"
    kept, dropped = [], []
    for item in items:
        if not keep_slow and item.get_closest_marker("slow") is not None:
            dropped.append(item)
            continue
        kept.append(item)
        for name, probe in ENVIRONMENT.items():
            if item.get_closest_marker(name) is None:
                continue
            reason = probe()
            if reason is not None:
                item.add_marker(pytest.mark.skip(reason=f"{name}: {reason}"))
                break
    if dropped:
        config.hook.pytest_deselected(items=dropped)
        items[:] = kept


# Per-test durations for the whole run (setup + call + teardown, seconds), one
# JSON file per process, when MOJOLEARN_TEST_DURATIONS_DIR names a directory:
# tools/test_duration_report.py sums them per file and per test so the heavy
# tail is cut from measurements. On by default; costs one dict entry per test.
# Under pytest-xdist the controller receives every worker's reports, so only
# the controller (or a run without xdist) records; workers write nothing.
_DURATIONS = {}


def _durations_dir():
    """Where this run's per-test durations go. ON BY DEFAULT (Andrew, 2026-11-07: timing rides on every run,
    never a separate run): MOJOLEARN_TEST_DURATIONS_DIR names the directory, "0" turns recording off, unset
    means ~/mojolearn-evidence/test-durations/latest (overwritten by each run; the runner archives it)."""
    value = os.environ.get("MOJOLEARN_TEST_DURATIONS_DIR", "").strip()
    if value == "0":
        return ""
    return value or os.path.join(os.path.expanduser("~"), "mojolearn-evidence", "test-durations", "latest")


def _records():
    return bool(_durations_dir()) and not os.environ.get("PYTEST_XDIST_WORKER")


def pytest_runtest_logreport(report):
    if _records():
        entry = _DURATIONS.setdefault(report.nodeid, {"seconds": 0.0, "outcome": "passed"})
        entry["seconds"] += report.duration
        if report.outcome != "passed" and entry["outcome"] == "passed":
            entry["outcome"] = report.outcome if report.when == "call" or report.outcome != "failed" else "error"


def pytest_sessionfinish(session):
    if not _records() or not _DURATIONS:
        return
    import json
    directory = _durations_dir()
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"durations-{os.getpid()}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"tests": _DURATIONS}, handle)
