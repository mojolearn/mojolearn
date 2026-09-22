#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE RELEASE REHEARSAL: the checks that failed 0.8.12 AFTER boxes were
rented, run locally in minutes BEFORE anything is (2026-09-21).

    pixi run release-rehearsal

A PRE-RELEASE REHEARSAL THE RELEASER RUNS BY HAND. It is not a per-merge
check, not a CI job and not a gate on anything: nothing calls it but a person
about to cut a release (docs/RELEASE_CHECKLIST.md, step 0). It rents nothing,
compiles nothing and holds the Mac's Metal slot only for the Python tests.

WHAT 0.8.12 LOST TIME TO, and the step that catches each:
  * the macOS wheel build failed after 16 minutes of compiling because
    `mojolearn/cross_vendor.py` imported NumPy and `math` outside the
    independent-verification files. `platform-math` stages the package the
    way the wheel ships it (every shipped .py, plus the two tools copies
    pack_wheel.py and build_release_wheel.sh add) and runs the wheel audit's
    Python rules over it.
  * a byte LM surface test broke on a signature change and was patched by
    hand mid-release. `python-tests` runs the whole suite in IDENTICAL mode.
  * the wheel audit's own tests, the docs facts, the extension lists and the
    host list are each a line of the checklist that was run late or not at
    all.
  * the Linux legs refused at launch more than once (r2, r3). Their DRY RUNS
    rent nothing and still run every local pre-flight a real launch runs.

One line per step, PASS or FAIL with the seconds it took; a failing step
prints the tail of its log and the log's path. Every step runs even after a
failure, so one rehearsal names every problem. The exit status is non-zero
when any step failed.

    python3 tools/release_rehearsal.py --list        # the steps, nothing run
    python3 tools/release_rehearsal.py --only python-tests,platform-math
    MOJOLEARN_DO_TOKEN_FILE=~/.mojolearn_do_token     # the AMD leg's token (the default)
"""
import argparse
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "python" / "mojolearn"


def stage_package(dest):
    """The Python half of the wheel, laid out as it ships: the packages
    pyproject.toml lists (mojolearn, mojolearn.models), the top-level
    `mojolearn_diagnostics` module, and the two tools the builders copy in
    under package names. Tests do not ship and are not staged."""
    dest = Path(dest)
    (dest / "mojolearn" / "models").mkdir(parents=True)
    for src in sorted(PKG.glob("*.py")):
        shutil.copy2(src, dest / "mojolearn" / src.name)
    for src in sorted((PKG / "models").glob("*.py")):
        shutil.copy2(src, dest / "mojolearn" / "models" / src.name)
    shutil.copy2(ROOT / "python" / "mojolearn_diagnostics.py", dest / "mojolearn_diagnostics.py")
    # packaging/linux/pack_wheel.py and packaging/macos/build_release_wheel.sh
    shutil.copy2(ROOT / "tools" / "identity_break.py", dest / "mojolearn" / "_identity_break.py")
    shutil.copy2(ROOT / "tools" / "identity_trace_diff.py", dest / "mojolearn" / "_identity_trace_diff.py")
    return dest


def steps(work):
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    token = os.path.expanduser(os.environ.get("MOJOLEARN_DO_TOKEN_FILE", "~/.mojolearn_do_token"))
    py = sys.executable
    stage = Path(work) / "staged-wheel-tree"
    return [
        ("wheel-audit-tests", ["pixi", "run", "test-wheel-audit"], None),
        # THE ONE STEP THAT CAN TOUCH METAL: the suite imports the package and
        # some tests fit on the default device. It takes the Mac's Metal slot
        # like every other GPU job, so it waits for a running Apple pass
        # instead of corrupting it.
        ("python-tests", [py, str(ROOT / "tools" / "mac_slot.py"), "--wait-timeout", "3600", "metal",
                          "pixi", "run", "-e", "test", "test-python"],
         dict(MOJOLEARN_NUMERIC_MODE="identical")),
        ("docs-facts", ["pixi", "run", "check-docs-facts"], None),
        ("ext-lists", [py, "packaging/check_ext_lists.py"], None),
        ("ext-lists-host", [py, "packaging/check_ext_lists.py", "--host"], None),
        ("platform-math", ["sh", "-c", "%s tools/release_rehearsal.py --stage-only %s && "
                           "pixi run -e pkg python packaging/portable_math/wheel.py --python-tree %s"
                           % (shlex.quote(py), shlex.quote(str(stage)), shlex.quote(str(stage)))], None),
        ("nvidia-leg-dry-run", ["sh", "tools/gemm_remote_leg.sh", "nvidia", "--payload", "mamba",
                                "--source-ref", head, "--gpu", "NVIDIA H100 80GB HBM3",
                                "--allow-concurrent", "--minutes", "60"],
         dict(MOJOLEARN_NVIDIA_CAMPAIGN="7", MOJOLEARN_GPU_ARCHS="sm_90a")),
        # The real AMD leg runs in the pinned Ubuntu 22.04 container; dry-run that.
        ("amd-leg-dry-run", ["bash", "tools/do_release061_leg.sh", head, token],
         dict(MOJOLEARN_RELEASE_UBUNTU22="1")),
    ]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--list", action="store_true", help="print the steps and run nothing")
    ap.add_argument("--only", default="", help="comma-separated step names")
    ap.add_argument("--keep", default="", metavar="DIR", help="write logs here instead of a fresh temp dir")
    ap.add_argument("--stage-only", default="", metavar="DIR", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.stage_only:
        shutil.rmtree(args.stage_only, ignore_errors=True)
        stage_package(args.stage_only)
        return 0
    work = Path(args.keep) if args.keep else Path(tempfile.gettempdir()) / "mojolearn-rehearsal-(temp)"
    plan = steps(work)
    names = [n for n, _, _ in plan]
    only = [n for n in args.only.split(",") if n]
    unknown = sorted(set(only) - set(names))
    if unknown:
        ap.error(f"unknown step(s) {unknown}; the steps are {names}")
    if args.list:
        for name, cmd, env in plan:
            extra = " ".join(f"{k}={v}" for k, v in (env or {}).items())
            print(f"{name}: {extra + ' ' if extra else ''}{shlex.join(cmd)}")
        return 0
    work = Path(args.keep) if args.keep else Path(tempfile.mkdtemp(prefix="mojolearn-rehearsal-"))
    work.mkdir(parents=True, exist_ok=True)
    plan = steps(work)
    print(f"# release rehearsal at {subprocess.run(['git', '-C', str(ROOT), 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()}, logs in {work}", flush=True)
    failed = []
    for name, cmd, env in plan:
        if only and name not in only:
            continue
        log = work / f"{name}.log"
        t0 = time.monotonic()
        with open(log, "w") as fh:
            try:
                rc = subprocess.call(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                                     env=dict(os.environ, **(env or {})))
            except OSError as exc:
                fh.write(f"could not start: {exc}\n")
                rc = 127
        seconds = time.monotonic() - t0
        if rc == 0:
            print(f"PASS {name} ({seconds:.0f}s)", flush=True)
            continue
        failed.append(name)
        print(f"FAIL {name} ({seconds:.0f}s, exit {rc}) log {log}", flush=True)
        tail = log.read_text(errors="replace").splitlines()[-8:]
        for line in tail:
            print(f"    {line}", flush=True)
    print(f"# {'REHEARSAL PASSED' if not failed else 'REHEARSAL FAILED: ' + ', '.join(failed)}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
