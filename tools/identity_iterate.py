#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Run changed identity lanes in fair, bounded jobs with live progress.

The default base fixture is an iteration check, never a full release record.
Each job runs two fits and inference/save/reload. Batch and decode checks
are separate explicit groups. Routine jobs use existing CPU oracle binaries,
with one-minute queue and execution limits. Product routing is unchanged.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def plan(paths, base, fixtures, lanes=()):
    import lane_select
    import identity_break
    if lanes and paths:
        raise ValueError("choose named lanes OR changed paths, not both")
    if lanes:
        unknown = set(lanes) - set(identity_break.LANES)
        if unknown:
            raise ValueError(f"unknown lanes: {sorted(unknown)}")
        # Naming one algorithm does not need the whole dependency graph.
        selection = dict(lanes=list(dict.fromkeys(lanes)), fallback=False,
                         reasons={name: "explicit lane" for name in lanes}, unattributed=[])
    else:
        selection = lane_select.select(paths if paths else lane_select.changed_paths(base), ref=base)
    unknown = set(fixtures) - set(identity_break.FIXTURES)
    if unknown:
        raise ValueError(f"unknown fixtures: {sorted(unknown)}")
    return selection


def cpu_package(out, host_dir, source=None):
    """Stage only Python sources, exercising the existing CPU-only install route.

    No product code or installed binaries are edited. A stable directory keeps
    resume provenance stable. Native host files stay at their recorded paths.
    """
    source = source or ROOT / "python" / "mojolearn"
    host_dir = Path(host_dir).resolve()
    if not host_dir.is_dir() or not list(host_dir.glob("_mojolearn_*_host.so")):
        raise ValueError(f"no CPU host bindings under {host_dir}; pass --host-dir")
    root = out.resolve() / "cpu-package"
    root.mkdir(parents=True, exist_ok=True)
    marker = root / ".identity-iterate"
    if any(root.iterdir()) and not marker.exists():
        raise ValueError(f"refusing to replace an unowned package directory: {root}")
    marker.touch()
    package = root / "mojolearn"
    package.mkdir(exist_ok=True)
    wanted = set()
    for src in source.rglob("*.py"):
        rel = src.relative_to(source)
        if "__pycache__" in rel.parts:
            continue
        dst = package / rel
        wanted.add(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_symlink() and dst.resolve() == src.resolve():
            continue
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())
    for dst in package.rglob("*"):
        if dst.suffix in (".so", ".dylib"):
            raise ValueError(f"native library in CPU source staging: {dst}")
        if dst.suffix == ".py" and dst not in wanted:
            dst.unlink()
    return root, host_dir


def command(python, lane, fixture, record, timeout, mode, resume, group="all", wait_timeout=60, deadline=None):
    cmd = [python, str(ROOT / "tools/mac_slot.py"), "--timeout", str(timeout),
           "--wait-timeout", str(wait_timeout),
           "--timing-json", str(record.with_suffix(".timing.json")), "run" if mode == "cpu" else mode,
           python, "-u", str(ROOT / "tools/identity_break.py"), "--lanes", lane,
           "--fixtures", fixture, "--repeats", "1", "--fail-on-refused", "--json", str(record)]
    if deadline is not None:
        cmd[2:2] = ["--deadline", str(deadline)]
    if mode == "cpu":
        cmd.append("--require-cpu")
    if mode != "run":
        cmd += ["--require-backend", mode]
    if group in ("core", "rlpair"):
        cmd.append("--no-batch")
    if group in ("core", "batch"):
        cmd.append("--no-rlpair")
    if resume and record.exists():
        cmd.append("--resume")
    return cmd


def run_job(cmd, env, **popen_options):
    child = subprocess.Popen(cmd, env=env, **popen_options)
    def forward(signum, frame):
        if child.poll() is None:
            child.send_signal(signum)
    old = {s: signal.signal(s, forward) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        code = child.wait()
        return 128 - code if code < 0 else code
    finally:
        for s, handler in old.items():
            signal.signal(s, handler)


def main(argv=None):
    started = time.monotonic()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", help="changed source paths; otherwise diff --base plus local changes")
    ap.add_argument("--base", default="HEAD^")
    ap.add_argument("--lane", action="append", default=[], help="one algorithm lane; repeat to select several")
    ap.add_argument("--lanes", default="", help="comma-separated algorithm lanes")
    ap.add_argument("--exhaustive", action="store_true", help="all nine fixtures for the selected lanes")
    ap.add_argument("--fixtures", default=None, help="comma-separated fixture names; default base is iteration coverage")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--plan", action="store_true", help="print selection without taking any slot")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--budget", type=float, default=None, help="total seconds including planning, queueing and execution (CPU/run 300; Metal 60)")
    ap.add_argument("--timeout", type=float, default=60, help="maximum seconds per job, excluding wait (default 60)")
    ap.add_argument("--mode", choices=("cpu", "metal", "cuda", "hip", "run"), default="cpu")
    ap.add_argument("--metal-diagnostic", action="store_true",
                    help="explicitly run one bounded Apple diagnostic between releases")
    ap.add_argument("--metal-expanded", action="store_true",
                    help="explicitly permit multiple Metal diagnostic jobs; never implied by a release marker")
    ap.add_argument("--host-dir", type=Path, help="prebuilt internal CPU oracle bindings; no builds are launched")
    ap.add_argument("--wait-timeout", type=float, default=60, help="queue limit in seconds (default 60)")
    ap.add_argument("--probe-group", choices=("core", "batch", "rlpair", "all"), default="core",
                    help="core = training/inference/save/reload; all runs separate bounded jobs")
    ap.add_argument("--full-selection", action="store_true", help="explicitly run a selector fallback; Apple release policy still applies")
    args = ap.parse_args(argv)
    if args.budget is None:
        args.budget = 60 if args.mode == "metal" else 300
    if any(not math.isfinite(v) or v <= 0 for v in (args.timeout, args.wait_timeout, args.budget)):
        ap.error("timeouts must be finite and positive")
    import identity_break
    if args.exhaustive and args.fixtures is not None:
        ap.error("choose --exhaustive OR --fixtures")
    fixture_names = ",".join(identity_break.FIXTURES) if args.exhaustive else (args.fixtures if args.fixtures is not None else "base")
    fixtures = list(dict.fromkeys(filter(None, fixture_names.split(","))))
    if not fixtures:
        ap.error("at least one fixture is required")
    lanes = args.lane + [name for name in args.lanes.split(",") if name]
    try:
        selected = plan(args.paths, args.base, fixtures, lanes) if lanes else plan(args.paths, args.base, fixtures)
    except ValueError as exc:
        ap.error(str(exc))
    selected["fixtures"] = fixtures
    selected["repeats"] = 1
    selected["cell_count"] = len(selected["lanes"]) * len(fixtures)
    # The audit is a preflight, before staging sources or taking a lease.
    import lane_applicability
    column = {"cpu": "cpu-host", "metal": "apple-metal", "cuda": "nvidia-1gpu", "hip": "amd-1gpu"}.get(args.mode)
    scopes = lane_applicability.scopes() if column and selected["lanes"] else {}
    selected["inapplicable"] = {
        lane: scopes[lane].applicable(column)[1]
        for lane in selected["lanes"]
        if column and not scopes[lane].applicable(column)[0]
    }
    jobs = []
    for lane in selected["lanes"]:
        has_batch = callable(identity_break.BATCH.get(lane))
        if args.probe_group == "batch" and not has_batch:
            ap.error(f"{lane} has no applicable batch probe: {identity_break.BATCH.get(lane, 'undeclared')}")
        groups = (["core"] + (["batch"] if has_batch else [])
                  + (["rlpair"] if lane in identity_break.RLPAIR else [])
                  if args.probe_group == "all" else [args.probe_group])
        if args.probe_group == "rlpair" and lane not in identity_break.RLPAIR:
            ap.error(f"{lane} does not declare an rlpair probe")
        jobs.extend(dict(lane=lane, fixture=fixture, group=group) for fixture in fixtures for group in groups)
    selected.update(jobs=jobs, job_count=len(jobs), fit_count=2 * len(jobs),
                    mode=args.mode, timeout=args.timeout, wait_timeout=args.wait_timeout, budget=args.budget)

    print(json.dumps(selected, indent=2), flush=True)
    if args.plan:
        return 0
    if selected["inapplicable"]:
        ap.error("selected lanes are inapplicable to this backend: " +
                 json.dumps(selected["inapplicable"], sort_keys=True))
    if selected["fallback"] and not args.full_selection:
        ap.error("selection fell back to all lanes; inspect the plan or explicitly pass --full-selection")
    # Metal runs whatever was selected (2026-09-19). Each cell is fitted once
    # under the shared budget, so the refusals that kept a ten-hour column off
    # the Mac have nothing left to refuse. --metal-diagnostic and
    # --metal-expanded are accepted and ignored.
    import identity_break
    if not selected["lanes"]:
        print("No affected identity lanes. No GPU work requested.")
        return 0
    if not args.out:
        ap.error("--out is required for checkpointed execution")
    for job in jobs:
        record = args.out.resolve() / "{lane}--{fixture}--{group}.json".format(**job)
        if record.exists() and not args.resume:
            ap.error(f"{record} exists; use --resume or a new output directory")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "selection.json").write_text(json.dumps(selected, indent=2) + "\n")
    env = dict(os.environ, MOJOLEARN_NUMERIC_MODE="identical")
    # Run this checkout's routing code, even when the interpreter has a wheel.
    env["PYTHONPATH"] = str(ROOT / "python") + os.pathsep + env.get("PYTHONPATH", "")
    if args.mode == "cpu":
        host = args.host_dir or os.environ.get("MOJOLEARN_HOST_DIR") or ROOT / "python" / "mojolearn" / "host"
        try:
            package_root, host = cpu_package(args.out, host)
        except ValueError as exc:
            ap.error(str(exc))
        env["PYTHONPATH"] = str(package_root) + os.pathsep + env.get("PYTHONPATH", "")
        env["MOJOLEARN_HOST_DIR"] = str(host)
    deadline = started + args.budget
    completed = []
    def report(status, pending, failed=None):
        result = dict(status=status, complete=status == "passed", budget=args.budget,
                      elapsed_seconds=time.monotonic() - started,
                      completed=completed, pending=pending, failed=failed)
        path = args.out / "run-summary.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(path)
    report("running", jobs)
    for index, job in enumerate(jobs):
        if time.monotonic() >= deadline:
            report("budget-exhausted", jobs[index:])
            print(f"Total run budget exhausted; {len(jobs) - index} jobs remain. See run-summary.json.", flush=True)
            return 124
        lane, fixture, group = job["lane"], job["fixture"], job["group"]
        record = args.out.resolve() / f"{lane}--{fixture}--{group}.json"
        print(f"# QUEUE {lane}/{fixture}/{group}", flush=True)
        code = run_job(command(sys.executable, lane, fixture, record,
                               args.timeout, args.mode, args.resume, group, args.wait_timeout, deadline), env)
        if code:
            report("budget-exhausted" if code == 124 and time.monotonic() >= deadline else "failed",
                   jobs[index + 1:], dict(job, exit_code=code))
            return code
        completed.append(job)
        report("running", jobs[index + 1:])
    report("passed", [])
    print("Selected iteration cells passed. This is not a full release qualification.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
