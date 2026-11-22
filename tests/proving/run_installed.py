# SPDX-License-Identifier: Apache-2.0
"""Run named pytest files of python/mojolearn/tests (and tests/api, tools/) against the INSTALLED wheel.

    python tests/proving/run_installed.py [--workers N] FILE [FILE ...] [-- PYTEST ARGS]

Each file runs in its own interpreter under tools/run_python_tests.py INSTALLED_BOOTSTRAP (the source tree's
`mojolearn.tests` mounted over the installed runtime, never the source runtime), the files spread across every
core (N defaults to the core count, limited by memory); one log per file under ./proving-installed-logs and one
summary line per file. Exit status is nonzero when any file fails. Test tooling only.
"""
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "tools"))
from run_python_tests import INSTALLED_BOOTSTRAP  # noqa: E402


def _workers(units, requested):
    if requested:
        return max(1, min(units, requested))
    cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    try:
        memory = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        cores = min(cores, max(1, int(memory / (1.5 * (1 << 30)))))
    except (ValueError, OSError, AttributeError):
        pass
    return max(1, min(units, cores))


def main(argv):
    workers = None
    if argv[:1] == ["--workers"]:
        workers, argv = int(argv[1]), argv[2:]
    extra = []
    if "--" in argv:
        at = argv.index("--")
        argv, extra = argv[:at], argv[at + 1:]
    if not argv:
        print(__doc__)
        return 2
    logs = Path("proving-installed-logs")
    logs.mkdir(exist_ok=True)
    root = REPO / "python" / "mojolearn" / "tests"

    def one(path):
        log = logs / (Path(path).name + ".log")
        with open(log, "w") as out:
            rc = subprocess.call([sys.executable, "-I", "-c", INSTALLED_BOOTSTRAP, str(root), "-q", "-rfEs",
                                  "--import-mode=importlib", str(path), *extra], stdout=out, stderr=subprocess.STDOUT)
        return path, rc, log

    with ThreadPoolExecutor(_workers(len(argv), workers)) as pool:
        results = list(pool.map(one, argv))
    failed = 0
    for path, rc, log in results:
        tail = log.read_text(errors="replace").strip().splitlines()[-1:] or [""]
        print(f"rc={rc} {path} :: {tail[0][:200]} (log {log})")
        failed += rc not in (0, 5)          # 5: every test skipped or none collected
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
